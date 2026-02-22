use ryft_xla_sys::bindings::{
    MlirType, mlirComplexTypeGet, mlirFloatTypeGetWidth, mlirTypeIsAFloat8E3M4, mlirTypeIsAFloat8E4M3,
    mlirTypeIsAFloat8E4M3B11FNUZ, mlirTypeIsAFloat8E4M3FN, mlirTypeIsAFloat8E4M3FNUZ, mlirTypeIsAFloat8E5M2,
    mlirTypeIsAFloat8E5M2FNUZ, mlirTypeIsAFloat8E8M0FNU,
};

use crate::{Context, Type, mlir_subtype_trait_impls};

/// Built-in MLIR [`Type`] that represents a floating-point type.
///
/// This `trait` acts effectively as the super-type of all MLIR floating-point [`Type`]s and can be checked and
/// specialized using the [`FloatType::is`](crate::TypeRef::is) and [`FloatType::cast`](crate::TypeRef::cast) functions.
pub trait FloatType<'c, 't: 'c>: Type<'c, 't> {
    /// Returns the bit width of this floating-point type. The bit width of a floating-point type is defined as the
    /// number of bits that each value that belongs to that type occupies.
    fn bit_width(&self) -> usize {
        unsafe { mlirFloatTypeGetWidth(self.to_c_api()) as usize }
    }
}

/// Reference to a [`FloatType`] that is owned by a [`Context`].
#[derive(Copy, Clone)]
pub struct FloatTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> FloatType<'c, 't> for FloatTypeRef<'c, 't> {}

mlir_subtype_trait_impls!(FloatTypeRef<'c, 't> as Type, mlir_type = Type, mlir_subtype = Float);

// We need the following macro because not all of the relevant MLIR C API functions are named consistently
// and we want to be able to synthesize the function name for each floating-point type in order to implement
// the `mlir_float_type` macro below.
macro_rules! mlir_float_type_is_a_function {
    (BFloat16) => {
        ryft_xla_sys::bindings::mlirTypeIsABF16
    };
    (Float16) => {
        ryft_xla_sys::bindings::mlirTypeIsAF16
    };
    (FloatTF32) => {
        ryft_xla_sys::bindings::mlirTypeIsATF32
    };
    (Float32) => {
        ryft_xla_sys::bindings::mlirTypeIsAF32
    };
    (Float64) => {
        ryft_xla_sys::bindings::mlirTypeIsAF64
    };
    ($type:ident) => {
        paste::paste! { ryft_xla_sys::bindings::[<mlirTypeIsA $type>] }
    };
}

macro_rules! mlir_float_type {
    ($type:ident, url = $url:expr) => {
        paste::paste! {
            #[doc = "Built-in MLIR [`FloatType`]. Refer to the [official MLIR documentation]("]
            #[doc = $url]
            #[doc = ") for information."]
            #[derive(Copy, Clone)]
            pub struct [<$type TypeRef>]<'c, 't> {
                /// Handle that represents this [`Type`] in the MLIR C API.
                handle: MlirType,

                /// [`Context`] that owns this [`Type`].
                context: &'c Context<'t>,
            }

            impl<'c, 't> Type<'c, 't> for [<$type TypeRef>]<'c, 't> {
                unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
                    if !handle.ptr.is_null() && unsafe { mlir_float_type_is_a_function!($type)(handle) } {
                        Some(Self { handle, context })
                    } else {
                        None
                    }
                }

                unsafe fn to_c_api(&self) -> MlirType {
                    self.handle
                }

                fn context(&self) -> &'c Context<'t> {
                    &self.context
                }
            }

            impl<'c, 't> FloatType<'c, 't> for [<$type TypeRef>]<'c, 't> {}

            mlir_subtype_trait_impls!([<$type TypeRef>]<'c, 't> as Type, mlir_type = Type);
        }
    };
}

mlir_float_type!(Float4E2M1FN, url = "https://mlir.llvm.org/docs/Dialects/Builtin/#float4e2m1fntype");
mlir_float_type!(Float6E2M3FN, url = "https://mlir.llvm.org/docs/Dialects/Builtin/#float6e2m3fntype");
mlir_float_type!(Float6E3M2FN, url = "https://mlir.llvm.org/docs/Dialects/Builtin/#float6e3m2fntype");
mlir_float_type!(Float8E3M4, url = "https://mlir.llvm.org/docs/Dialects/Builtin/#float8e3m4type");
mlir_float_type!(Float8E4M3, url = "https://mlir.llvm.org/docs/Dialects/Builtin/#float8e4m3type");
mlir_float_type!(Float8E4M3B11FNUZ, url = "https://mlir.llvm.org/docs/Dialects/Builtin/#float8e4m3b11fnuztype");
mlir_float_type!(Float8E4M3FN, url = "https://mlir.llvm.org/docs/Dialects/Builtin/#float8e4m3fntype");
mlir_float_type!(Float8E4M3FNUZ, url = "https://mlir.llvm.org/docs/Dialects/Builtin/#float8e4m3fnuztype");
mlir_float_type!(Float8E5M2, url = "https://mlir.llvm.org/docs/Dialects/Builtin/#float8e5m2type");
mlir_float_type!(Float8E5M2FNUZ, url = "https://mlir.llvm.org/docs/Dialects/Builtin/#float8e5m2fnuztype");
mlir_float_type!(Float8E8M0FNU, url = "https://mlir.llvm.org/docs/Dialects/Builtin/#float8e8m0fnutype");
mlir_float_type!(BFloat16, url = "https://mlir.llvm.org/docs/Dialects/Builtin/#bfloat16type");
mlir_float_type!(Float16, url = "https://mlir.llvm.org/docs/Dialects/Builtin/#float16type");
mlir_float_type!(FloatTF32, url = "https://mlir.llvm.org/docs/Dialects/Builtin/#floattf32type");
mlir_float_type!(Float32, url = "https://mlir.llvm.org/docs/Dialects/Builtin/#float32type");
mlir_float_type!(Float64, url = "https://mlir.llvm.org/docs/Dialects/Builtin/#float64type");
// mlir_float_type!(Float80, url = "https://mlir.llvm.org/docs/Dialects/Builtin/#float80type");
// mlir_float_type!(Float128, url = "https://mlir.llvm.org/docs/Dialects/Builtin/#float128type");
mlir_float_type!(Complex, url = "https://mlir.llvm.org/docs/Dialects/Builtin/#complextype");

/// Built-in MLIR [`FloatType`] that represents any 8-bit floating-point type.
///
/// This `trait` acts effectively as the super-type of all MLIR 8-bit floating-point [`Type`]s and can be checked and
/// specialized using the [`Float8Type::is`](crate::TypeRef::is) and [`Float8Type::cast`](crate::TypeRef::cast)
/// functions.
pub trait Float8Type<'c, 't: 'c>: FloatType<'c, 't> {}

/// Reference to a [`Float8Type`] that is owned by a [`Context`].
#[derive(Copy, Clone)]
pub struct Float8TypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> Type<'c, 't> for Float8TypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null()
            && unsafe {
                mlirTypeIsAFloat8E3M4(handle)
                    || mlirTypeIsAFloat8E4M3(handle)
                    || mlirTypeIsAFloat8E4M3B11FNUZ(handle)
                    || mlirTypeIsAFloat8E4M3FN(handle)
                    || mlirTypeIsAFloat8E4M3FNUZ(handle)
                    || mlirTypeIsAFloat8E5M2(handle)
                    || mlirTypeIsAFloat8E5M2FNUZ(handle)
                    || mlirTypeIsAFloat8E8M0FNU(handle)
            }
        {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirType {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        &self.context
    }
}

impl<'c, 't> FloatType<'c, 't> for Float8TypeRef<'c, 't> {}
impl<'c, 't> Float8Type<'c, 't> for Float8TypeRef<'c, 't> {}

mlir_subtype_trait_impls!(Float8TypeRef<'c, 't> as Type, mlir_type = Type);

impl<'c, 't> Float8Type<'c, 't> for Float8E3M4TypeRef<'c, 't> {}
impl<'c, 't> Float8Type<'c, 't> for Float8E4M3TypeRef<'c, 't> {}
impl<'c, 't> Float8Type<'c, 't> for Float8E4M3B11FNUZTypeRef<'c, 't> {}
impl<'c, 't> Float8Type<'c, 't> for Float8E4M3FNTypeRef<'c, 't> {}
impl<'c, 't> Float8Type<'c, 't> for Float8E4M3FNUZTypeRef<'c, 't> {}
impl<'c, 't> Float8Type<'c, 't> for Float8E5M2TypeRef<'c, 't> {}
impl<'c, 't> Float8Type<'c, 't> for Float8E5M2FNUZTypeRef<'c, 't> {}
impl<'c, 't> Float8Type<'c, 't> for Float8E8M0FNUTypeRef<'c, 't> {}

// We need the following macro because not all of the relevant MLIR C API functions are named consistently
// and we want to be able to synthesize the function name for each floating-point type in order to implement
// the `mlir_float_type_constructor` macro below.
macro_rules! mlir_float_type_get_function {
    (BFloat16) => {
        ryft_xla_sys::bindings::mlirBF16TypeGet
    };
    (Float16) => {
        ryft_xla_sys::bindings::mlirF16TypeGet
    };
    (FloatTF32) => {
        ryft_xla_sys::bindings::mlirTF32TypeGet
    };
    (Float32) => {
        ryft_xla_sys::bindings::mlirF32TypeGet
    };
    (Float64) => {
        ryft_xla_sys::bindings::mlirF64TypeGet
    };
    ($type:ident) => {
        paste::paste! { ryft_xla_sys::bindings::[<mlir $type TypeGet>] }
    };
}

macro_rules! mlir_float_type_constructor {
    ($type:ident, $doc:expr $(,)*) => {
        paste::paste! {
            impl<'t> Context<'t> {
                #[doc = $doc]
                pub fn [<$type:lower _type>]<'c>(&'c self) -> [<$type TypeRef>]<'c, 't> {
                    // While this operation can mutate the context (in that it might add an entry to its corresponding
                    // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
                    // function quite inconvenient/annoying in practice. This should have no negative consequences in
                    // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
                    // should be no possibility for this function to cause problems with an immutable borrow.
                    unsafe {
                        [<$type TypeRef>]::from_c_api(
                            mlir_float_type_get_function!($type)(*self.handle.borrow()),
                            &self,
                        ).unwrap()
                    }
                }
            }
        }
    };
}

mlir_float_type_constructor!(Float4E2M1FN, "Creates a new [`Float4E2M1FNTypeRef`] owned by this [`Context`].");
mlir_float_type_constructor!(Float6E2M3FN, "Creates a new [`Float6E2M3FNTypeRef`] owned by this [`Context`].");
mlir_float_type_constructor!(Float6E3M2FN, "Creates a new [`Float6E3M2FNTypeRef`] owned by this [`Context`].");
mlir_float_type_constructor!(Float8E3M4, "Creates a new [`Float8E3M4TypeRef`] owned by this [`Context`].");
mlir_float_type_constructor!(Float8E4M3, "Creates a new [`Float8E4M3TypeRef`] owned by this [`Context`].");
mlir_float_type_constructor!(
    Float8E4M3B11FNUZ,
    "Creates a new [`Float8E4M3B11FNUZTypeRef`] owned by this [`Context`].",
);
mlir_float_type_constructor!(Float8E4M3FN, "Creates a new [`Float8E4M3FNTypeRef`] owned by this [`Context`].");
mlir_float_type_constructor!(Float8E4M3FNUZ, "Creates a new [`Float8E4M3FNUZTypeRef`] owned by this [`Context`].");
mlir_float_type_constructor!(Float8E5M2, "Creates a new [`Float8E5M2TypeRef`] owned by this [`Context`].");
mlir_float_type_constructor!(Float8E5M2FNUZ, "Creates a new [`Float8E5M2FNUZTypeRef`] owned by this [`Context`].");
mlir_float_type_constructor!(Float8E8M0FNU, "Creates a new [`Float8E8M0FNUTypeRef`] owned by this [`Context`].");
mlir_float_type_constructor!(BFloat16, "Creates a new [`BFloat16TypeRef`] owned by this [`Context`].");
mlir_float_type_constructor!(Float16, "Creates a new [`Float16TypeRef`] owned by this [`Context`].");
mlir_float_type_constructor!(FloatTF32, "Creates a new [`FloatTF32TypeRef`] owned by this [`Context`].");
mlir_float_type_constructor!(Float32, "Creates a new [`Float32TypeRef`] owned by this [`Context`].");
mlir_float_type_constructor!(Float64, "Creates a new [`Float64TypeRef`] owned by this [`Context`].");
// mlir_float_type_constructor!(Float80, "Creates a new [`Float80TypeRef`] owned by this [`Context`].");
// mlir_float_type_constructor!(Float128, "Creates a new [`Float128TypeRef`] owned by this [`Context`].");

impl<'t> Context<'t> {
    /// Creates a new [`ComplexTypeRef`] owned by this [`Context`], using the provided element type for its real
    /// and imaginary parts. Note that the element type must be either a floating-point type or an integer type.
    pub fn complex_type<'c, T: Type<'c, 't>>(&'c self, element_type: T) -> ComplexTypeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        let _guard = self.borrow();
        unsafe { ComplexTypeRef::from_c_api(mlirComplexTypeGet(element_type.to_c_api()), &self).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::types::tests::{test_type_casting, test_type_display_and_debug};

    use super::*;

    #[test]
    fn test_float_type() {
        let context = Context::new();

        let r#type = context.bfloat16_type();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.bit_width(), 16);

        let r#type = context.float16_type();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.bit_width(), 16);

        let r#type = context.float32_type();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.bit_width(), 32);

        let r#type = context.float64_type();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.bit_width(), 64);

        let element_type = context.float32_type();
        let r#type = context.complex_type(element_type);
        assert_eq!(&context, r#type.context());
    }

    #[test]
    fn test_float_type_equality() {
        let context = Context::new();

        // Same types from the same context must be equal because they are "uniqued".
        let type_1 = context.float32_type();
        let type_2 = context.float32_type();
        assert_eq!(type_1, type_2);

        // Different types from the same context must not be equal.
        let type_2 = context.float64_type();
        assert_ne!(type_1, type_2);

        // Same types from different contexts must not be equal.
        let context = Context::new();
        let type_2 = context.float32_type();
        assert_ne!(type_1, type_2);

        // Also test using complex types.
        let element_type = context.float32_type();
        let type_1 = context.complex_type(element_type);
        let type_2 = context.complex_type(element_type);
        assert_eq!(type_1, type_2);

        let other_element_type = context.float64_type();
        let type_2 = context.complex_type(other_element_type);
        assert_ne!(type_1, type_2);

        let context = Context::new();
        let other_element_type = context.float32_type();
        let type_2 = context.complex_type(other_element_type);
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_float_type_display_and_debug() {
        let context = Context::new();
        test_type_display_and_debug(context.bfloat16_type(), "bf16");
        test_type_display_and_debug(context.float16_type(), "f16");
        test_type_display_and_debug(context.float32_type(), "f32");
        test_type_display_and_debug(context.float64_type(), "f64");
        test_type_display_and_debug(context.complex_type(context.float32_type()), "complex<f32>");
    }

    #[test]
    fn test_float_type_parsing() {
        let context = Context::new();
        assert_eq!(context.parse_type("bf16").unwrap(), context.bfloat16_type());
        assert_eq!(context.parse_type("f32").unwrap(), context.float32_type());
        assert_eq!(context.parse_type("complex<f32>").unwrap(), context.complex_type(context.float32_type()));
    }

    #[test]
    fn test_float_type_casting() {
        let context = Context::new();
        test_type_casting(context.float32_type());
        test_type_casting(context.complex_type(context.float32_type()));
    }

    #[test]
    fn test_float8_type() {
        let context = Context::new();
        let r#type = context.float8e4m3fn_type().as_type_ref().cast::<Float8TypeRef>().unwrap();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.bit_width(), 8);
        assert_eq!(context.float16_type().as_type_ref().cast::<Float8TypeRef>(), None);
    }

    #[test]
    fn test_float8_type_casting() {
        let context = Context::new();
        test_type_casting(context.float8e4m3fn_type().as_type_ref().cast::<Float8TypeRef>().unwrap());
    }
}
