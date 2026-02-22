use ryft_xla_sys::bindings::{
    MlirAttribute, mlirAttributeIsADenseBoolArray, mlirAttributeIsADenseF32Array, mlirAttributeIsADenseF64Array,
    mlirAttributeIsADenseI8Array, mlirAttributeIsADenseI16Array, mlirAttributeIsADenseI32Array,
    mlirAttributeIsADenseI64Array, mlirDenseArrayAttrGetTypeID,
};

use crate::{Attribute, Context, TypeId, mlir_subtype_trait_impls};

/// Built-in MLIR [`Attribute`] that represents a dense unidimensional array of boolean, integer, or floating-point
/// values. This is different from [`DenseElementsAttribute`](super::DenseElementsAttribute) in that it is a flat
/// unidimensional array that does not use a storage optimization for splat and is guaranteed to store the underlying
/// elements contiguously in memory. Also, the underlying value type always has a bit width that is a multiple of 8
/// (i.e., the number of bits that each value occupies is a multiple of 8; boolean values are stored as bytes).
///
/// This `trait` acts effectively as the super-type of all MLIR [`DenseArrayAttribute`]s and can be checked and
/// specialized using the [`DenseArrayAttribute::is`](crate::AttributeRef::is) and
/// [`DenseArrayAttribute::cast`](crate::AttributeRef::cast) functions.
///
/// # Examples
///
/// The following are examples of [`DenseArrayAttribute`]s represented using their
/// [`Display`](std::fmt::Display) rendering:
///
/// ```text
/// array<i8>
/// array<i32: 10, 42>
/// array<f64: 42., 12.>
/// ```
///
/// Note that when specific subtypes/specializations are used as arguments to operations, the rendering might omit
/// the value type and only contain the underlying values (e.g., `[1, 2, 3]`).
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#densearrayattr)
/// for more information.
pub trait DenseArrayAttribute<'c, 't: 'c>: Attribute<'c, 't> {}

/// Reference to a [`DenseArrayAttribute`] that is owned by a [`Context`].
#[derive(Copy, Clone)]
pub struct DenseArrayAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> DenseArrayAttributeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`DenseArrayAttributeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirDenseArrayAttrGetTypeID()).unwrap() }
    }
}

impl<'c, 't> Attribute<'c, 't> for DenseArrayAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null()
            && unsafe {
                mlirAttributeIsADenseBoolArray(handle)
                    || mlirAttributeIsADenseI8Array(handle)
                    || mlirAttributeIsADenseI16Array(handle)
                    || mlirAttributeIsADenseI32Array(handle)
                    || mlirAttributeIsADenseI64Array(handle)
                    || mlirAttributeIsADenseF32Array(handle)
                    || mlirAttributeIsADenseF64Array(handle)
            }
        {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

impl<'c, 't> DenseArrayAttribute<'c, 't> for DenseArrayAttributeRef<'c, 't> {}

mlir_subtype_trait_impls!(DenseArrayAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

// We need the following macros because not all of the relevant MLIR C API functions are named consistently
// and we want to be able to synthesize the function name for each value type in order to implement
// the `mlir_dense_array_attribute` macro below.
macro_rules! mlir_dense_array_is_a_function {
    (Boolean) => {
        ryft_xla_sys::bindings::mlirAttributeIsADenseBoolArray
    };
    (Integer8) => {
        ryft_xla_sys::bindings::mlirAttributeIsADenseI8Array
    };
    (Integer16) => {
        ryft_xla_sys::bindings::mlirAttributeIsADenseI16Array
    };
    (Integer32) => {
        ryft_xla_sys::bindings::mlirAttributeIsADenseI32Array
    };
    (Integer64) => {
        ryft_xla_sys::bindings::mlirAttributeIsADenseI64Array
    };
    (Float32) => {
        ryft_xla_sys::bindings::mlirAttributeIsADenseF32Array
    };
    (Float64) => {
        ryft_xla_sys::bindings::mlirAttributeIsADenseF64Array
    };
}

macro_rules! mlir_dense_array_get_element_function {
    (Boolean) => {
        ryft_xla_sys::bindings::mlirDenseBoolArrayGetElement
    };
    (Integer8) => {
        ryft_xla_sys::bindings::mlirDenseI8ArrayGetElement
    };
    (Integer16) => {
        ryft_xla_sys::bindings::mlirDenseI16ArrayGetElement
    };
    (Integer32) => {
        ryft_xla_sys::bindings::mlirDenseI32ArrayGetElement
    };
    (Integer64) => {
        ryft_xla_sys::bindings::mlirDenseI64ArrayGetElement
    };
    (Float32) => {
        ryft_xla_sys::bindings::mlirDenseF32ArrayGetElement
    };
    (Float64) => {
        ryft_xla_sys::bindings::mlirDenseF64ArrayGetElement
    };
}

macro_rules! mlir_dense_array_constructor_function {
    (Boolean) => {
        ryft_xla_sys::bindings::mlirDenseBoolArrayGet
    };
    (Integer8) => {
        ryft_xla_sys::bindings::mlirDenseI8ArrayGet
    };
    (Integer16) => {
        ryft_xla_sys::bindings::mlirDenseI16ArrayGet
    };
    (Integer32) => {
        ryft_xla_sys::bindings::mlirDenseI32ArrayGet
    };
    (Integer64) => {
        ryft_xla_sys::bindings::mlirDenseI64ArrayGet
    };
    (Float32) => {
        ryft_xla_sys::bindings::mlirDenseF32ArrayGet
    };
    (Float64) => {
        ryft_xla_sys::bindings::mlirDenseF64ArrayGet
    };
}

macro_rules! mlir_dense_array_constructor_call {
    (Boolean, $context:ident, $values:ident) => {{
        let values = $values.iter().map(|&value| value as std::ffi::c_int).collect::<Vec<_>>();
        DenseBooleanArrayAttributeRef::from_c_api(
            mlir_dense_array_constructor_function!(Boolean)(
                // While this operation can mutate the context (in that it might add an entry to its corresponding
                // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
                // function quite inconvenient/annoying in practice. This should have no negative consequences in
                // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
                // should be no possibility for this function to cause problems with an immutable borrow.
                *$context.handle.borrow(),
                values.len().cast_signed(),
                values.as_ptr(),
            ),
            &$context,
        )
    }};
    ($type_name:ident, $context:ident, $values:ident) => {
        paste::paste! {
            [<Dense $type_name ArrayAttributeRef>]::from_c_api(mlir_dense_array_constructor_function!($type_name)(
                // While this operation can mutate the context (in that it might add an entry to its corresponding
                // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
                // function quite inconvenient/annoying in practice. This should have no negative consequences in
                // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
                // should be no possibility for this function to cause problems with an immutable borrow.
                *$context.handle.borrow(),
                $values.len().cast_signed(),
                $values.as_ptr(),
            ), &$context)
        }
    };
}

macro_rules! mlir_dense_array_attribute {
    ($type:ty, $type_name:ident, description = $type_description:expr) => {
        paste::paste! {
            #[doc = "Built-in MLIR [`DenseArrayAttributeRef`] that contains "]
            #[doc = $type_description]
            #[doc = " values."]
            #[derive(Copy, Clone)]
            pub struct [<Dense $type_name ArrayAttributeRef>]<'c, 't> {
                /// Handle that represents this [`Attribute`] in the MLIR C API.
                handle: MlirAttribute,

                /// [`Context`] that owns this [`Attribute`].
                context: &'c Context<'t>,
            }

            impl<'c, 't> [<Dense $type_name ArrayAttributeRef>]<'c, 't> {
                /// Returns the length of this dense array attribute (i.e., the number of values it contains).
                pub fn len(&self) -> usize {
                    unsafe { ryft_xla_sys::bindings::mlirDenseArrayGetNumElements(self.handle).cast_unsigned() }
                }

                /// Returns the values contained in this dense array attribute.
                pub fn values(&self) -> impl Iterator<Item = $type> {
                    (0..self.len()).map(|index| self.value(index))
                }

                /// Returns the `index`-th value of this dense array attribute.
                pub fn value(&self, index: usize) -> $type {
                    unsafe { mlir_dense_array_get_element_function!($type_name)(self.handle, index.cast_signed()) }
                }
            }

            impl<'c, 't> Attribute<'c, 't> for [<Dense $type_name ArrayAttributeRef>]<'c, 't> {
                unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
                    if !handle.ptr.is_null() && unsafe { mlir_dense_array_is_a_function!($type_name)(handle) } {
                        Some(Self { handle, context })
                    } else {
                        None
                    }
                }

                unsafe fn to_c_api(&self) -> MlirAttribute {
                    self.handle
                }

                fn context(&self) -> &'c Context<'t> {
                    &self.context
                }
            }

            impl<'c, 't> DenseArrayAttribute<'c, 't> for [<Dense $type_name ArrayAttributeRef>]<'c, 't> {}

            mlir_subtype_trait_impls!([<Dense $type_name ArrayAttributeRef>]<'c, 't> as Attribute, mlir_type = Attribute);

            impl<'c, 't> From<[<Dense $type_name ArrayAttributeRef>]<'c, 't>> for Vec<$type> {
                fn from(value: [<Dense $type_name ArrayAttributeRef>]) -> Self {
                    value.values().collect()
                }
            }

            impl<'c, 't, const N: usize> crate::FromWithContext<'c, 't, &[$type; N]>
                for [<Dense $type_name ArrayAttributeRef>]<'c, 't>
            {
                fn from_with_context(value: &[$type; N], context: &'c Context<'t>) -> Self {
                    context.[<dense_ $type _array_attribute>](value).unwrap()
                }
            }

            impl<'c, 't> crate::FromWithContext<'c, 't, &[$type]> for [<Dense $type_name ArrayAttributeRef>]<'c, 't> {
                fn from_with_context(value: &[$type], context: &'c Context<'t>) -> Self {
                    context.[<dense_ $type _array_attribute>](value).unwrap()
                }
            }

            impl<'t> Context<'t> {
                /// Creates a new dense array attribute which is owned by this [`Context`].
                pub fn [<dense_ $type _array_attribute>]<'c>(
                    &'c self,
                    values: &[$type],
                ) -> Option<[<Dense $type_name ArrayAttributeRef>]<'c, 't>> {
                    unsafe { mlir_dense_array_constructor_call!($type_name, self, values) }
                }
            }
        }
    };
}

mlir_dense_array_attribute!(bool, Boolean, description = "boolean");
mlir_dense_array_attribute!(i8, Integer8, description = "8-bit integer");
mlir_dense_array_attribute!(i16, Integer16, description = "16-bit integer");
mlir_dense_array_attribute!(i32, Integer32, description = "32-bit integer");
mlir_dense_array_attribute!(i64, Integer64, description = "64-bit integer");
mlir_dense_array_attribute!(f32, Float32, description = "32-bit floating-point number");
mlir_dense_array_attribute!(f64, Float64, description = "64-bit floating-point number");

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::IntoWithContext;
    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_dense_array_attribute_type_id() {
        let context = Context::new();
        let dense_array_attribute_id = DenseArrayAttributeRef::type_id();
        let dense_array_attribute_1 = context.dense_i32_array_attribute(&[1, 2, 3]).unwrap();
        let dense_array_attribute_2 = context.dense_i32_array_attribute(&[1, 2, 3]).unwrap();
        assert_eq!(dense_array_attribute_1.type_id(), dense_array_attribute_2.type_id());
        assert_eq!(dense_array_attribute_id, dense_array_attribute_1.type_id());
    }

    #[test]
    fn test_dense_bool_array_attribute() {
        let context = Context::new();
        let attribute: DenseBooleanArrayAttributeRef<'_, '_> = (&[true, false, true]).into_with_context(&context);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.len(), 3);
        assert_eq!(attribute.value(0), true);
        assert_eq!(attribute.value(1), false);
        assert_eq!(attribute.value(2), true);
        let values: Vec<_> = attribute.into();
        assert_eq!(values, vec![true, false, true]);
    }

    #[test]
    fn test_dense_i32_array_attribute() {
        let context = Context::new();
        let attribute = context.dense_i32_array_attribute(&[1, 2, 3]).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.len(), 3);
        assert_eq!(attribute.value(0), 1);
        assert_eq!(attribute.value(1), 2);
        assert_eq!(attribute.value(2), 3);
        assert_eq!(attribute.values().collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    #[test]
    fn test_dense_i64_array_attribute() {
        let context = Context::new();
        let attribute = context.dense_i64_array_attribute(&[10, 20, 30]).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.len(), 3);
        assert_eq!(attribute.value(0), 10);
        assert_eq!(attribute.value(1), 20);
        assert_eq!(attribute.value(2), 30);
        assert_eq!(attribute.values().collect::<Vec<_>>(), vec![10, 20, 30]);
    }

    #[test]
    fn test_dense_f32_array_attribute() {
        let context = Context::new();
        let attribute = context.dense_f32_array_attribute(&[1.5, 2.5, 3.5]).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.len(), 3);
        assert_eq!(attribute.value(0), 1.5);
        assert_eq!(attribute.value(1), 2.5);
        assert_eq!(attribute.value(2), 3.5);
        assert_eq!(attribute.values().collect::<Vec<_>>(), vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_dense_array_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.dense_i32_array_attribute(&[1, 2, 3]).unwrap();
        let attribute_2 = context.dense_i32_array_attribute(&[1, 2, 3]).unwrap();
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.dense_i32_array_attribute(&[4, 5, 6]).unwrap();
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.dense_i32_array_attribute(&[1, 2, 3]).unwrap();
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_dense_array_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.dense_i32_array_attribute(&[1, 2, 3]).unwrap();
        test_attribute_display_and_debug(attribute, "array<i32: 1, 2, 3>");
    }

    #[test]
    fn test_dense_array_attribute_parsing() {
        let context = Context::new();
        let attribute = context.dense_i32_array_attribute(&[1, 2, 3]).unwrap();
        let parsed = context.parse_attribute("array<i32: 1, 2, 3>").unwrap();
        assert_eq!(parsed, attribute);
    }

    #[test]
    fn test_dense_array_attribute_casting() {
        let context = Context::new();
        let attribute = context.dense_i32_array_attribute(&[1, 2, 3]).unwrap();
        test_attribute_casting(attribute);
        test_attribute_casting(attribute.cast::<DenseArrayAttributeRef>().unwrap());
    }
}
