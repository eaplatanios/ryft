use half::{bf16, f16};

use ryft_xla_sys::bindings::{
    MlirAttribute, mlirDenseElementsAttrGetRawData, mlirDenseElementsAttrGetSplatValue, mlirDenseElementsAttrIsSplat,
    mlirDenseElementsAttrReshapeGet, mlirDenseIntOrFPElementsAttrGetTypeID, mlirElementsAttrGetNumElements,
    mlirElementsAttrGetValue, mlirElementsAttrIsValidIndex, mlirSparseElementsAttrGetIndices,
    mlirSparseElementsAttrGetTypeID, mlirSparseElementsAttrGetValues, mlirSparseElementsAttribute,
    mlirUnmanagedDenseResourceElementsAttrGet,
};

use crate::{
    Attribute, AttributeRef, Context, FromWithContext, ShapedType, StringRef, TypeId, VectorTypeDimension,
    mlir_subtype_trait_impls,
};

use super::{FloatAttributeRef, IntegerAttributeRef};

/// Built-in MLIR [`Attribute`] that represents a multidimensional array of values. This `trait` acts effectively
/// as the super-type of all MLIR [`ElementsAttribute`]s and can be checked and specialized using the
/// [`ElementsAttribute::is`](AttributeRef::is) and [`ElementsAttribute::cast`](AttributeRef::cast) functions.
///
/// This is different from [`DenseArrayAttribute`](super::DenseArrayAttribute) in that it is multidimensional, it
/// supports more than just 8-bit aligned primitive types for the underlying values, and it supports various layouts
/// for the underlying values array, including sparse layouts.
pub trait ElementsAttribute<'c, 't: 'c>: Attribute<'c, 't> {
    /// Returns the total number of elements in this attribute.
    fn elements_count(&self) -> usize {
        unsafe { mlirElementsAttrGetNumElements(self.to_c_api()) as usize }
    }

    /// Returns the element at the specified multidimensional index of this attribute,
    /// or [`None`] if the index is out of bounds.
    fn element(&self, index: &[usize]) -> Option<AttributeRef<'c, 't>> {
        unsafe {
            let mut index = index.iter().map(|&index| index as u64).collect::<Vec<_>>();
            if mlirElementsAttrIsValidIndex(self.to_c_api(), index.len().cast_signed(), index.as_mut_ptr()) {
                AttributeRef::from_c_api(
                    mlirElementsAttrGetValue(self.to_c_api(), index.len().cast_signed(), index.as_mut_ptr()),
                    self.context(),
                )
            } else {
                None
            }
        }
    }
}

/// Reference to an [`ElementsAttribute`] that is owned by a [`Context`].
#[derive(Copy, Clone)]
pub struct ElementsAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> ElementsAttribute<'c, 't> for ElementsAttributeRef<'c, 't> {}

mlir_subtype_trait_impls!(
    ElementsAttributeRef<'c, 't> as Attribute,
    mlir_type = Attribute,
    mlir_subtype = Elements,
);

/// Built-in MLIR [`ElementsAttribute`] that represents a dense multidimensional array of integer or floating-point
/// values. This attribute contains a densely packed vector or tensor of integer or floating-point values. The element
/// type of this attribute is required to be either an [`IntegerTypeRef`](crate::IntegerTypeRef) or a
/// [`FloatTypeRef`](crate::FloatTypeRef).
///
/// # Examples
///
/// The following are examples of [`DenseElementsAttribute`]s represented using their
/// [`Display`](std::fmt::Display) rendering:
///
/// ```text
/// // A "splat" tensor with 2 `i32` values:
/// dense<10> : tensor<2xi32>
///
/// // A tensor with 2 `f32` values:
/// dense<[10.0, 11.0]> : tensor<2xf32>
///
/// // A "splat" tensor with 2 strings:
/// dense<"example"> : tensor<2x!foo.string>
///
/// // A tensor with 2 string values:
/// dense<["example1", "example2"]> : tensor<2x!foo.string>
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#denseintorfpelementsattr)
/// for more information.
pub trait DenseElementsAttribute<'c, 't: 'c>: ElementsAttribute<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`DenseElementsAttributeRef`].
    fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirDenseIntOrFPElementsAttrGetTypeID()).unwrap() }
    }

    /// Returns `true` if this attribute contains a single replicated value.
    fn is_splat(&self) -> bool {
        unsafe { mlirDenseElementsAttrIsSplat(self.to_c_api()) }
    }

    /// Returns the value of the replicated element if this attribute is a splat and [`None`] otherwise.
    fn splat(&self) -> Option<AttributeRef<'c, 't>> {
        if !self.is_splat() {
            None
        } else {
            unsafe { AttributeRef::from_c_api(mlirDenseElementsAttrGetSplatValue(self.to_c_api()), self.context()) }
        }
    }

    /// Returns a pointer to the raw underlying data of this attribute.
    unsafe fn raw_data(&self) -> *const std::ffi::c_void {
        unsafe { mlirDenseElementsAttrGetRawData(self.to_c_api()) }
    }
}

/// Reference to a [`DenseElementsAttribute`] that is owned by a [`Context`].
#[derive(Copy, Clone)]
pub struct DenseElementsAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

macro_rules! mlir_dense_elements_attribute_splat {
    (mlir_type = Bool, rust_type = bool) => {
        /// Returns the value of the replicated element if this attribute is a splat and [`None`] otherwise.
        /// This function is unsafe because it can panic if this attribute's element type does not match
        /// the return type of this function.
        pub unsafe fn bool_splat(&self) -> Option<bool> {
            if !self.is_splat() {
                None
            } else {
                unsafe {
                    let value = ryft_xla_sys::bindings::mlirDenseElementsAttrGetBoolSplatValue(self.to_c_api());
                    Some(value != 0)
                }
            }
        }
    };
    (mlir_type = String, rust_type = StringRef) => {
        /// Returns the value of the replicated element if this attribute is a splat and [`None`] otherwise.
        /// This function is unsafe because it can panic if this attribute's element type does not match
        /// the return type of this function.
        pub unsafe fn string_ref_splat(&self) -> Option<StringRef<'c>> {
            if !self.is_splat() {
                None
            } else {
                unsafe {
                    Some(StringRef::from_c_api(ryft_xla_sys::bindings::mlirDenseElementsAttrGetStringSplatValue(
                        self.to_c_api(),
                    )))
                }
            }
        }
    };
    (mlir_type = $mlir_type:ident, rust_type = $rust_type:ident) => {
        paste::paste! {
            /// Returns the value of the replicated element if this attribute is a splat and [`None`] otherwise.
            /// This function is unsafe because it can panic if this attribute's element type does not match
            /// the return type of this function.
            pub unsafe fn [<$rust_type _splat>](&self) -> Option<$rust_type> {
                if !self.is_splat() {
                    None
                } else {
                    unsafe {
                        Some(ryft_xla_sys::bindings::[<mlirDenseElementsAttrGet $mlir_type SplatValue>](
                            self.to_c_api(),
                        ))
                    }
                }
            }
        }
    };
}

macro_rules! mlir_dense_elements_attribute_elements {
    (mlir_type = String, rust_type = StringRef) => {
        /// Returns all the elements of this attribute, flattened.
        /// This function is unsafe because it can panic if this attribute's element type does not match
        /// the return type of this function.
        pub unsafe fn string_ref_elements(&self) -> impl Iterator<Item = StringRef<'c>> {
            unsafe { (0..self.elements_count()).map(|index| self.string_ref_element(index).unwrap()) }
        }
    };
    (mlir_type = $mlir_type:ident, rust_type = $rust_type:ident) => {
        paste::paste! {
            /// Returns all the elements of this attribute, flattened.
            /// This function is unsafe because it can panic if this attribute's element type does not match
            /// the return type of this function.
            pub unsafe fn [<$rust_type:snake _elements>](&self) -> impl Iterator<Item = $rust_type> {
                unsafe { (0..self.elements_count()).map(|index| self.[<$rust_type:snake _element>](index).unwrap()) }
            }
        }
    };
}

macro_rules! mlir_dense_elements_attribute_element {
    (mlir_type = String, rust_type = StringRef) => {
        paste::paste! {
            /// Returns the element at the `index`-th position of this attribute, assuming flat contiguous indexing.
            /// This function is unsafe because it can panic if this attribute's element type does not match
            /// the return type of this function.
            pub unsafe fn string_ref_element(&self, index: usize) -> Option<StringRef<'c>> {
                if index >= self.elements_count() {
                    None
                } else {
                    unsafe {
                        Some(StringRef::from_c_api(ryft_xla_sys::bindings::mlirDenseElementsAttrGetStringValue(
                            self.to_c_api(),
                            index.cast_signed(),
                        )))
                    }
                }
            }
        }
    };
    (mlir_type = $mlir_type:ident, rust_type = $rust_type:ident) => {
        paste::paste! {
            /// Returns the element at the `index`-th position of this attribute, assuming flat contiguous indexing.
            /// This function is unsafe because it can panic if this attribute's element type does not match
            /// the return type of this function.
            pub unsafe fn [<$rust_type:snake _element>](&self, index: usize) -> Option<$rust_type> {
                if index >= self.elements_count() {
                    None
                } else {
                    unsafe {
                        Some(ryft_xla_sys::bindings::[<mlirDenseElementsAttrGet $mlir_type Value>](
                            self.to_c_api(),
                            index.cast_signed(),
                        ) as $rust_type)
                    }
                }
            }
        }
    };
}

impl<'c, 't> DenseElementsAttributeRef<'c, 't> {
    mlir_dense_elements_attribute_splat!(mlir_type = Bool, rust_type = bool);
    mlir_dense_elements_attribute_splat!(mlir_type = UInt8, rust_type = u8);
    mlir_dense_elements_attribute_splat!(mlir_type = Int8, rust_type = i8);
    mlir_dense_elements_attribute_splat!(mlir_type = UInt32, rust_type = u32);
    mlir_dense_elements_attribute_splat!(mlir_type = Int32, rust_type = i32);
    mlir_dense_elements_attribute_splat!(mlir_type = UInt64, rust_type = u64);
    mlir_dense_elements_attribute_splat!(mlir_type = Int64, rust_type = i64);
    mlir_dense_elements_attribute_splat!(mlir_type = Float, rust_type = f32);
    mlir_dense_elements_attribute_splat!(mlir_type = Double, rust_type = f64);
    mlir_dense_elements_attribute_splat!(mlir_type = String, rust_type = StringRef);

    mlir_dense_elements_attribute_elements!(mlir_type = Bool, rust_type = bool);
    mlir_dense_elements_attribute_elements!(mlir_type = Index, rust_type = usize);
    mlir_dense_elements_attribute_elements!(mlir_type = UInt8, rust_type = u8);
    mlir_dense_elements_attribute_elements!(mlir_type = Int8, rust_type = i8);
    mlir_dense_elements_attribute_elements!(mlir_type = UInt32, rust_type = u32);
    mlir_dense_elements_attribute_elements!(mlir_type = Int32, rust_type = i32);
    mlir_dense_elements_attribute_elements!(mlir_type = UInt64, rust_type = u64);
    mlir_dense_elements_attribute_elements!(mlir_type = Int64, rust_type = i64);
    mlir_dense_elements_attribute_elements!(mlir_type = Float, rust_type = f32);
    mlir_dense_elements_attribute_elements!(mlir_type = Double, rust_type = f64);
    mlir_dense_elements_attribute_elements!(mlir_type = String, rust_type = StringRef);

    mlir_dense_elements_attribute_element!(mlir_type = Bool, rust_type = bool);
    mlir_dense_elements_attribute_element!(mlir_type = Index, rust_type = usize);
    mlir_dense_elements_attribute_element!(mlir_type = UInt8, rust_type = u8);
    mlir_dense_elements_attribute_element!(mlir_type = Int8, rust_type = i8);
    mlir_dense_elements_attribute_element!(mlir_type = UInt32, rust_type = u32);
    mlir_dense_elements_attribute_element!(mlir_type = Int32, rust_type = i32);
    mlir_dense_elements_attribute_element!(mlir_type = UInt64, rust_type = u64);
    mlir_dense_elements_attribute_element!(mlir_type = Int64, rust_type = i64);
    mlir_dense_elements_attribute_element!(mlir_type = Float, rust_type = f32);
    mlir_dense_elements_attribute_element!(mlir_type = Double, rust_type = f64);
    mlir_dense_elements_attribute_element!(mlir_type = String, rust_type = StringRef);
}

impl<'c, 't> ElementsAttribute<'c, 't> for DenseElementsAttributeRef<'c, 't> {}

impl<'c, 't> DenseElementsAttribute<'c, 't> for DenseElementsAttributeRef<'c, 't> {}

mlir_subtype_trait_impls!(
    DenseElementsAttributeRef<'c, 't> as Attribute,
    mlir_type = Attribute,
    mlir_subtype = DenseElements,
);

macro_rules! mlir_dense_elements_attribute_constructor {
    ($attribute_type:ident, $attribute_name:ident $(,)?) => {
        impl<'t> Context<'t> {
            /// Creates a new dense elements attribute with the specified [`ShapedType`] and elements.
            /// Returns [`None`] if the provided type is not supported or if it is not compatible with the
            /// provided elements.
            pub fn $attribute_name<'c, T: ShapedType<'c, 't>, A: Attribute<'c, 't>>(
                &'c self,
                shaped_type: T,
                elements: &[A],
            ) -> Option<$attribute_type<'c, 't>> {
                // While this operation can mutate the context (in that it might add an entry to its corresponding
                // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
                // function quite inconvenient/annoying in practice. This should have no negative consequences in
                // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
                // should be no possibility for this function to cause problems with an immutable borrow.
                let _guard = self.borrow();
                unsafe {
                    let elements = elements.iter().map(|element| element.to_c_api()).collect::<Vec<_>>();
                    $attribute_type::from_c_api(
                        ryft_xla_sys::bindings::mlirDenseElementsAttrGet(
                            shaped_type.to_c_api(),
                            elements.len().cast_signed(),
                            elements.as_ptr() as *const _,
                        ),
                        &self,
                    )
                }
            }
        }
    };
}

macro_rules! mlir_dense_elements_attribute_from_raw_buffer {
    ($attribute_type:ident, $attribute_name:ident $(,)?) => {
        paste::paste! {
            impl<'t> Context<'t> {
                /// Creates a new dense elements attribute with the specified [`ShapedType`] and elements
                /// populated from the provided packed row-major buffer. The format of the raw buffer is a
                /// densely packed array of values that can be bitcast to the storage format of the element
                /// type in the provided shaped type. Types that are not byte aligned will be treated
                /// differently, depending on their bit width:
                ///
                ///   - **Bit Width > 1:** Rounded up to the next byte.
                ///   - **Bit Width = 1:** Packed into 8-bit bytes with bits corresponding to the linear
                ///     order of the shaped type from the most significant bit to the least significant bit,
                ///     padded on the right.
                ///
                /// A raw buffer of a single element (or for 1-bit, a byte of value `0` or `255`) will be
                /// interpreted as a splat (i.e., a value that is the same for all elements in the attribute).
                ///
                /// The resulting attribute is owned by this context.
                pub fn [<$attribute_name _from_raw_buffer>]<'c, T: ShapedType<'c, 't>, D>(
                    &'c self,
                    shaped_type: T,
                    buffer: &[D],
                ) -> Option<$attribute_type<'c, 't>> {
                    // While this operation can mutate the context (in that it might add an entry to its corresponding
                    // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
                    // function quite inconvenient/annoying in practice. This should have no negative consequences in
                    // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
                    // should be no possibility for this function to cause problems with an immutable borrow.
                    let _guard = self.borrow();
                    unsafe {
                        $attribute_type::from_c_api(ryft_xla_sys::bindings::mlirDenseElementsAttrRawBufferGet(
                            shaped_type.to_c_api(),
                            buffer.len() * std::mem::size_of::<D>(),
                            buffer.as_ptr() as *const _,
                        ), &self)
                    }
                }
            }
        }
    };
}

macro_rules! mlir_dense_elements_attribute_from_element {
    ($attribute_type:ident, mlir_type = Attr, rust_type = Attribute $(,)?) => {
        paste::paste! {
            impl<'t> Context<'t> {
                /// Creates a new dense elements attribute with the specified [`ShapedType`] whose elements
                /// are all set to the provided value. The resulting attribute is owned by this context.
                pub fn splatted_dense_attribute_elements_attribute<
                    'c,
                    T: ShapedType<'c, 't>,
                    A: Attribute<'c, 't>,
                >(
                    &'c self,
                    shaped_type: T,
                    value: A,
                ) -> Option<$attribute_type<'c, 't>> {
                    // While this operation can mutate the context (in that it might add an entry to its corresponding
                    // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
                    // function quite inconvenient/annoying in practice. This should have no negative consequences in
                    // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
                    // should be no possibility for this function to cause problems with an immutable borrow.
                    let _guard = self.borrow();
                    unsafe {
                        $attribute_type::from_c_api(ryft_xla_sys::bindings::mlirDenseElementsAttrSplatGet(
                            shaped_type.to_c_api(),
                            value.to_c_api(),
                        ), &self)
                    }
                }
            }
        }
    };
    ($attribute_type:ident, mlir_type = Attr, rust_type = $rust_type:ident $(,)?) => {
        paste::paste! {
            impl<'t> Context<'t> {
                /// Creates a new dense elements attribute with the specified [`ShapedType`] whose elements
                /// are all set to the provided value. The resulting attribute is owned by this context.
                pub fn [<splatted_dense_ $rust_type:snake _elements_attribute>]<'c, T: ShapedType<'c, 't>>(
                    &'c self,
                    shaped_type: T,
                    value: [<$rust_type Ref>]<'c, 't>,
                ) -> Option<$attribute_type<'c, 't>> {
                    // While this operation can mutate the context (in that it might add an entry to its corresponding
                    // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
                    // function quite inconvenient/annoying in practice. This should have no negative consequences in
                    // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
                    // should be no possibility for this function to cause problems with an immutable borrow.
                    let _guard = self.borrow();
                    unsafe {
                        $attribute_type::from_c_api(ryft_xla_sys::bindings::mlirDenseElementsAttrSplatGet(
                            shaped_type.to_c_api(),
                            value.to_c_api(),
                        ), &self)
                    }
                }
            }
        }
    };
    ($attribute_type:ident, mlir_type = $mlir_type:ident, rust_type = $rust_type:ident $(,)?) => {
        paste::paste! {
            impl<'t> Context<'t> {
                /// Creates a new dense elements attribute with the specified [`ShapedType`] whose elements
                /// are all set to the provided value. The resulting attribute is owned by this context.
                pub fn [<splatted_dense_ $rust_type:snake _elements_attribute>]<'c, T: ShapedType<'c, 't>>(
                    &'c self,
                    shaped_type: T,
                    value: $rust_type,
                ) -> Option<$attribute_type<'c, 't>> {
                    // While this operation can mutate the context (in that it might add an entry to its corresponding
                    // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
                    // function quite inconvenient/annoying in practice. This should have no negative consequences in
                    // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
                    // should be no possibility for this function to cause problems with an immutable borrow.
                    let _guard = self.borrow();
                    unsafe {
                        $attribute_type::from_c_api(
                            ryft_xla_sys::bindings::[<mlirDenseElementsAttr $mlir_type SplatGet>](
                                shaped_type.to_c_api(),
                                value,
                            ),
                            &self,
                        )
                    }
                }
            }
        }
    };
}

macro_rules! mlir_dense_elements_attribute_from_elements {
    ($attribute_type:ident, mlir_type = Bool, rust_type = bool $(,)?) => {
        paste::paste! {
            impl<'t> Context<'t> {
                /// Creates a new dense elements attribute with the specified [`ShapedType`] whose elements
                /// are set to the provided values. The resulting attribute is owned by this context.
                pub fn dense_bool_elements_attribute<'c, T: ShapedType<'c, 't>>(
                    &'c self,
                    shaped_type: T,
                    elements: &[bool],
                ) -> Option<$attribute_type<'c, 't>> {
                    // While this operation can mutate the context (in that it might add an entry to its corresponding
                    // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
                    // function quite inconvenient/annoying in practice. This should have no negative consequences in
                    // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
                    // should be no possibility for this function to cause problems with an immutable borrow.
                    let _guard = self.borrow();
                    unsafe {
                        let elements = elements.iter().map(|&element| element as std::ffi::c_int).collect::<Vec<_>>();
                        $attribute_type::from_c_api(
                            ryft_xla_sys::bindings::mlirDenseElementsAttrBoolGet(
                                shaped_type.to_c_api(),
                                elements.len().cast_signed(),
                                elements.as_ptr(),
                            ),
                            &self,
                        )
                    }
                }
            }
        }
    };
    ($attribute_type:ident, mlir_type = $mlir_type:ident, rust_type = $rust_type:ident $(,)?) => {
        paste::paste! {
            impl<'t> Context<'t> {
                /// Creates a new dense elements attribute with the specified [`ShapedType`] whose elements
                /// are set to the provided values. The resulting attribute is owned by this context.
                pub fn [<dense_ $rust_type:snake _elements_attribute>]<'c, T: ShapedType<'c, 't>>(
                    &'c self,
                    shaped_type: T,
                    elements: &[$rust_type],
                ) -> Option<$attribute_type<'c, 't>> {
                    // While this operation can mutate the context (in that it might add an entry to its corresponding
                    // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
                    // function quite inconvenient/annoying in practice. This should have no negative consequences in
                    // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
                    // should be no possibility for this function to cause problems with an immutable borrow.
                    let _guard = self.borrow();
                    unsafe {
                        $attribute_type::from_c_api(
                            ryft_xla_sys::bindings::[<mlirDenseElementsAttr $mlir_type Get>](
                                shaped_type.to_c_api(),
                                elements.len().cast_signed(),
                                elements.as_ptr() as _,
                            ),
                            &self,
                        )
                    }
                }
            }
        }
    };
}

mlir_dense_elements_attribute_constructor!(DenseElementsAttributeRef, dense_elements_attribute);
mlir_dense_elements_attribute_from_raw_buffer!(DenseElementsAttributeRef, dense_elements_attribute);
mlir_dense_elements_attribute_from_element!(DenseElementsAttributeRef, mlir_type = Attr, rust_type = Attribute);
mlir_dense_elements_attribute_from_elements!(DenseElementsAttributeRef, mlir_type = String, rust_type = StringRef);

/// [`DenseElementsAttribute`] that holds either integer- or float-valued elements.
pub trait DenseIntegerOrFloatElementsAttribute<'c, 't: 'c>: DenseElementsAttribute<'c, 't> {
    /// Reshapes this attribute to the provided [`ShapedType`], returning a new attribute if successful.
    /// The provided type must have the same total number of elements as this attribute.
    fn reshape<T: ShapedType<'c, 't>>(&self, shaped_type: T) -> Option<DenseElementsAttributeRef<'c, 't>> {
        unsafe {
            DenseElementsAttributeRef::from_c_api(
                mlirDenseElementsAttrReshapeGet(self.to_c_api(), shaped_type.to_c_api()),
                self.context(),
            )
        }
    }
}

/// [`DenseElementsAttributeRef`] with an [`IntegerTypeRef`](crate::IntegerTypeRef) as its elements type.
#[derive(Copy, Clone)]
pub struct DenseIntegerElementsAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> DenseIntegerElementsAttributeRef<'c, 't> {
    mlir_dense_elements_attribute_splat!(mlir_type = Bool, rust_type = bool);
    mlir_dense_elements_attribute_splat!(mlir_type = UInt8, rust_type = u8);
    mlir_dense_elements_attribute_splat!(mlir_type = Int8, rust_type = i8);
    mlir_dense_elements_attribute_splat!(mlir_type = UInt32, rust_type = u32);
    mlir_dense_elements_attribute_splat!(mlir_type = Int32, rust_type = i32);
    mlir_dense_elements_attribute_splat!(mlir_type = UInt64, rust_type = u64);
    mlir_dense_elements_attribute_splat!(mlir_type = Int64, rust_type = i64);

    mlir_dense_elements_attribute_elements!(mlir_type = Bool, rust_type = bool);
    mlir_dense_elements_attribute_elements!(mlir_type = Index, rust_type = usize);
    mlir_dense_elements_attribute_elements!(mlir_type = UInt8, rust_type = u8);
    mlir_dense_elements_attribute_elements!(mlir_type = Int8, rust_type = i8);
    mlir_dense_elements_attribute_elements!(mlir_type = UInt32, rust_type = u32);
    mlir_dense_elements_attribute_elements!(mlir_type = Int32, rust_type = i32);
    mlir_dense_elements_attribute_elements!(mlir_type = UInt64, rust_type = u64);
    mlir_dense_elements_attribute_elements!(mlir_type = Int64, rust_type = i64);

    mlir_dense_elements_attribute_element!(mlir_type = Bool, rust_type = bool);
    mlir_dense_elements_attribute_element!(mlir_type = Index, rust_type = usize);
    mlir_dense_elements_attribute_element!(mlir_type = UInt8, rust_type = u8);
    mlir_dense_elements_attribute_element!(mlir_type = Int8, rust_type = i8);
    mlir_dense_elements_attribute_element!(mlir_type = UInt32, rust_type = u32);
    mlir_dense_elements_attribute_element!(mlir_type = Int32, rust_type = i32);
    mlir_dense_elements_attribute_element!(mlir_type = UInt64, rust_type = u64);
    mlir_dense_elements_attribute_element!(mlir_type = Int64, rust_type = i64);
}

impl<'c, 't> ElementsAttribute<'c, 't> for DenseIntegerElementsAttributeRef<'c, 't> {}

impl<'c, 't> DenseElementsAttribute<'c, 't> for DenseIntegerElementsAttributeRef<'c, 't> {}

impl<'c, 't> DenseIntegerOrFloatElementsAttribute<'c, 't> for DenseIntegerElementsAttributeRef<'c, 't> {}

mlir_subtype_trait_impls!(
    DenseIntegerElementsAttributeRef<'c, 't> as Attribute,
    mlir_type = Attribute,
    mlir_subtype = DenseIntElements,
);

mlir_dense_elements_attribute_constructor!(DenseIntegerElementsAttributeRef, dense_integer_elements_attribute);
mlir_dense_elements_attribute_from_raw_buffer!(DenseIntegerElementsAttributeRef, dense_integer_elements_attribute);
mlir_dense_elements_attribute_from_element!(
    DenseIntegerElementsAttributeRef,
    mlir_type = Attr,
    rust_type = IntegerAttribute,
);
mlir_dense_elements_attribute_from_element!(DenseIntegerElementsAttributeRef, mlir_type = Bool, rust_type = bool);
mlir_dense_elements_attribute_from_element!(DenseIntegerElementsAttributeRef, mlir_type = UInt8, rust_type = u8);
mlir_dense_elements_attribute_from_element!(DenseIntegerElementsAttributeRef, mlir_type = Int8, rust_type = i8);
mlir_dense_elements_attribute_from_element!(DenseIntegerElementsAttributeRef, mlir_type = UInt32, rust_type = u32);
mlir_dense_elements_attribute_from_element!(DenseIntegerElementsAttributeRef, mlir_type = Int32, rust_type = i32);
mlir_dense_elements_attribute_from_element!(DenseIntegerElementsAttributeRef, mlir_type = UInt64, rust_type = u64);
mlir_dense_elements_attribute_from_element!(DenseIntegerElementsAttributeRef, mlir_type = Int64, rust_type = i64);
mlir_dense_elements_attribute_from_elements!(DenseIntegerElementsAttributeRef, mlir_type = Bool, rust_type = bool);
mlir_dense_elements_attribute_from_elements!(DenseIntegerElementsAttributeRef, mlir_type = UInt8, rust_type = u8);
mlir_dense_elements_attribute_from_elements!(DenseIntegerElementsAttributeRef, mlir_type = Int8, rust_type = i8);
mlir_dense_elements_attribute_from_elements!(DenseIntegerElementsAttributeRef, mlir_type = UInt32, rust_type = u32);
mlir_dense_elements_attribute_from_elements!(DenseIntegerElementsAttributeRef, mlir_type = Int32, rust_type = i32);
mlir_dense_elements_attribute_from_elements!(DenseIntegerElementsAttributeRef, mlir_type = UInt64, rust_type = u64);
mlir_dense_elements_attribute_from_elements!(DenseIntegerElementsAttributeRef, mlir_type = Int64, rust_type = i64);

impl<'c, 't> FromWithContext<'c, 't, &[usize]> for DenseIntegerElementsAttributeRef<'c, 't> {
    fn from_with_context(value: &[usize], context: &'c Context<'t>) -> Self {
        context
            .dense_integer_elements_attribute(
                context
                    .vector_type(
                        context.index_type(),
                        &[VectorTypeDimension::Fixed(value.len())],
                        context.unknown_location(),
                    )
                    .unwrap(),
                &value
                    .iter()
                    .map(|index| context.integer_attribute(context.index_type(), *index as i64))
                    .collect::<Vec<_>>(),
            )
            .unwrap()
    }
}

/// [`DenseElementsAttributeRef`] with a [`FloatTypeRef`](crate::FloatTypeRef) as its elements type.
#[derive(Copy, Clone)]
pub struct DenseFloatElementsAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> DenseFloatElementsAttributeRef<'c, 't> {
    mlir_dense_elements_attribute_splat!(mlir_type = Float, rust_type = f32);
    mlir_dense_elements_attribute_splat!(mlir_type = Double, rust_type = f64);

    mlir_dense_elements_attribute_elements!(mlir_type = Float, rust_type = f32);
    mlir_dense_elements_attribute_elements!(mlir_type = Double, rust_type = f64);

    mlir_dense_elements_attribute_element!(mlir_type = Float, rust_type = f32);
    mlir_dense_elements_attribute_element!(mlir_type = Double, rust_type = f64);
}

impl<'c, 't> ElementsAttribute<'c, 't> for DenseFloatElementsAttributeRef<'c, 't> {}

impl<'c, 't> DenseElementsAttribute<'c, 't> for DenseFloatElementsAttributeRef<'c, 't> {}

impl<'c, 't> DenseIntegerOrFloatElementsAttribute<'c, 't> for DenseFloatElementsAttributeRef<'c, 't> {}

mlir_subtype_trait_impls!(
    DenseFloatElementsAttributeRef<'c, 't> as Attribute,
    mlir_type = Attribute,
    mlir_subtype = DenseFPElements,
);

mlir_dense_elements_attribute_constructor!(DenseFloatElementsAttributeRef, dense_float_elements_attribute);
mlir_dense_elements_attribute_from_raw_buffer!(DenseFloatElementsAttributeRef, dense_float_elements_attribute);
mlir_dense_elements_attribute_from_element!(
    DenseFloatElementsAttributeRef,
    mlir_type = Attr,
    rust_type = FloatAttribute,
);
mlir_dense_elements_attribute_from_element!(DenseFloatElementsAttributeRef, mlir_type = Float, rust_type = f32);
mlir_dense_elements_attribute_from_element!(DenseFloatElementsAttributeRef, mlir_type = Double, rust_type = f64);
mlir_dense_elements_attribute_from_elements!(DenseFloatElementsAttributeRef, mlir_type = Float16, rust_type = f16);
mlir_dense_elements_attribute_from_elements!(DenseFloatElementsAttributeRef, mlir_type = BFloat16, rust_type = bf16);
mlir_dense_elements_attribute_from_elements!(DenseFloatElementsAttributeRef, mlir_type = Float, rust_type = f32);
mlir_dense_elements_attribute_from_elements!(DenseFloatElementsAttributeRef, mlir_type = Double, rust_type = f64);

/// Built-in MLIR [`ElementsAttributeRef`] that is backed by a handle to a built-in dialect resource containing
/// a densely packed array of values.
///
/// This is different from [`DenseArrayAttributeRef`](crate::DenseArrayAttributeRef) in that it is multidimensional,
/// it supports more than just 8-bit aligned primitive types for the underlying values, and it supports various layouts
/// for the underlying values array, including sparse layouts.
///
/// # Examples
///
/// The following is an example of a [`DenseResourceElementsAttributeRef`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```text
/// "example.user_op"() {attr = dense_resource<blob1> : tensor<3xi64> } : () -> ()
///
/// {-#
/// dialect_resources: {
///     builtin: {
///       blob1: "0x08000000010000000000000002000000000000000300000000000000"
///     }
///   }
/// #-}
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#denseresourceelementsattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct DenseResourceElementsAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> ElementsAttribute<'c, 't> for DenseResourceElementsAttributeRef<'c, 't> {}

mlir_subtype_trait_impls!(
    DenseResourceElementsAttributeRef<'c, 't> as Attribute,
    mlir_type = Attribute,
    mlir_subtype = DenseResourceElements,
);

impl<'t> Context<'t> {
    /// Creates a new [`DenseResourceElementsAttributeRef`] by copying and reinterpreting the provided `data`,
    /// without performing any checks for its type, size, or alignment. This function is marked unsafe because it does
    /// minimal validation or massaging of the provided `data`, and it is up to the caller to ensure that the buffer
    /// meets the characteristics implied by the provided [`ShapedType`]. It should generally be avoided unless you are
    /// absolutely sure that the provided data is valid. The typed `dense_<T>_resource_elements_attribute` (e.g.,
    /// [`Context::dense_bool_resource_elements_attribute`]) functions should be used instead whenever possible.
    ///
    /// Note that the backing buffer and any user objects will be retained for the lifetime of the resulting
    /// [`DenseResourceElementsAttributeRef`]. This is typically bounded to the lifetime of this [`Context`] but the
    /// resource can have a shorter lifespan depending on how it is used in subsequent processing.
    ///
    /// The resulting [`DenseResourceElementsAttributeRef`] is owned by this context.
    pub unsafe fn unmanaged_dense_resource_elements_attribute<'c, 's, T: ShapedType<'c, 't>, D>(
        &'c self,
        shaped_type: T,
        name: StringRef<'s>,
        data: &[D],
    ) -> Option<DenseResourceElementsAttributeRef<'c, 't>> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        let _guard = self.borrow();

        unsafe extern "C" fn deallocate(
            _user_data: *mut std::ffi::c_void,
            data: *const std::ffi::c_void,
            size: usize,
            alignment: usize,
        ) {
            if !data.is_null() {
                unsafe {
                    let layout = std::alloc::Layout::from_size_align_unchecked(size, alignment);
                    std::alloc::dealloc(data as *mut _, layout);
                }
            }
        }

        unsafe {
            let data_size = data.len() * size_of::<D>();
            let data_alignment = align_of::<D>();
            let data_layout = std::alloc::Layout::from_size_align_unchecked(data_size, data_alignment);
            let data_ptr = std::alloc::alloc(data_layout) as *mut D;
            std::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, data.len());
            DenseResourceElementsAttributeRef::from_c_api(
                mlirUnmanagedDenseResourceElementsAttrGet(
                    shaped_type.to_c_api(),
                    name.to_c_api(),
                    data_ptr as *mut std::ffi::c_void,
                    data_size,
                    data_alignment,
                    false,
                    Some(deallocate),
                    std::ptr::null_mut(),
                ),
                &self,
            )
        }
    }
}

macro_rules! mlir_dense_resource_elements_attribute_element {
    (mlir_type = $mlir_type:ident, rust_type = $rust_type:ident) => {
        paste::paste! {
            impl<'c, 't> DenseResourceElementsAttributeRef<'c, 't> {
                /// Returns the element at the `index`-th position of this [`DenseResourceElementsAttributeRef`],
                /// assuming flat contiguous indexing. This function is unsafe because it can panic if this attribute's
                /// element type does not match the return type of this function.
                pub unsafe fn [<$rust_type:snake _element>](&self, index: usize) -> Option<$rust_type> {
                    if index >= self.elements_count() {
                        None
                    } else {
                        unsafe {
                            Some(ryft_xla_sys::bindings::[<mlirDense $mlir_type ResourceElementsAttrGetValue>](
                                self.to_c_api(),
                                index.cast_signed(),
                            ))
                        }
                    }
                }

                /// Returns all the elements of this [`DenseResourceElementsAttributeRef`], flattened.
                /// This function is unsafe because it can panic if this attribute's element type does not match
                /// the return type of this function.
                pub unsafe fn [<$rust_type:snake _elements>](&self) -> impl Iterator<Item = $rust_type> {
                    unsafe {
                        (0..self.elements_count()).map(|index| self.[<$rust_type:snake _element>](index).unwrap())
                    }
                }
            }
        }
    };
}

macro_rules! mlir_dense_resource_elements_attribute_from_elements {
    (mlir_type = Bool, rust_type = bool) => {
        impl<'t> Context<'t> {
            /// Creates a new [`DenseResourceElementsAttributeRef`] with the specified [`ShapedType`]
            /// whose elements are set to the provided values. The resulting attribute is owned by this context.
            pub fn dense_bool_resource_elements_attribute<'c, 's, T: ShapedType<'c, 't>>(
                &'c self,
                shaped_type: T,
                name: StringRef<'s>,
                elements: &[bool],
            ) -> Option<DenseResourceElementsAttributeRef<'c, 't>> {
                // While this operation can mutate the context (in that it might add an entry to its corresponding
                // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
                // function quite inconvenient/annoying in practice. This should have no negative consequences in
                // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
                // should be no possibility for this function to cause problems with an immutable borrow.
                let _guard = self.borrow();
                unsafe {
                    let elements = elements.iter().map(|&element| element as std::ffi::c_int).collect::<Vec<_>>();
                    DenseResourceElementsAttributeRef::from_c_api(
                        ryft_xla_sys::bindings::mlirUnmanagedDenseBoolResourceElementsAttrGet(
                            shaped_type.to_c_api(),
                            name.to_c_api(),
                            elements.len().cast_signed(),
                            elements.as_ptr(),
                        ),
                        &self,
                    )
                }
            }
        }
    };
    (mlir_type = $mlir_type:ident, rust_type = $rust_type:ident) => {
        paste::paste! {
            impl<'t> Context<'t> {
                /// Creates a new [`DenseResourceElementsAttributeRef`] with the specified [`ShapedType`] whose elements
                /// are set to the provided values. The resulting attribute is owned by this context.
                pub fn [<dense_ $rust_type:snake _resource_elements_attribute>]<'c, 's, T: ShapedType<'c, 't>>(
                    &'c self,
                    shaped_type: T,
                    name: StringRef<'s>,
                    elements: &[$rust_type],
                ) -> Option<DenseResourceElementsAttributeRef<'c, 't>> {
                    // While this operation can mutate the context (in that it might add an entry to its corresponding
                    // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
                    // function quite inconvenient/annoying in practice. This should have no negative consequences in
                    // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
                    // should be no possibility for this function to cause problems with an immutable borrow.
                    let _guard = self.borrow();
                    unsafe {
                        DenseResourceElementsAttributeRef::from_c_api(
                            ryft_xla_sys::bindings::[<mlirUnmanagedDense $mlir_type ResourceElementsAttrGet>](
                                shaped_type.to_c_api(),
                                name.to_c_api(),
                                elements.len().cast_signed(),
                                elements.as_ptr() as _,
                            ),
                            &self,
                        )
                    }
                }
            }
        }
    };
}

macro_rules! mlir_dense_resource_elements_functions {
    (mlir_type = $mlir_type:ident, rust_type = $rust_type:ident) => {
        mlir_dense_resource_elements_attribute_element!(mlir_type = $mlir_type, rust_type = $rust_type);
        mlir_dense_resource_elements_attribute_from_elements!(mlir_type = $mlir_type, rust_type = $rust_type);
    };
}

mlir_dense_resource_elements_functions!(mlir_type = Bool, rust_type = bool);
mlir_dense_resource_elements_functions!(mlir_type = UInt8, rust_type = u8);
mlir_dense_resource_elements_functions!(mlir_type = Int8, rust_type = i8);
mlir_dense_resource_elements_functions!(mlir_type = UInt16, rust_type = u16);
mlir_dense_resource_elements_functions!(mlir_type = Int16, rust_type = i16);
mlir_dense_resource_elements_functions!(mlir_type = UInt32, rust_type = u32);
mlir_dense_resource_elements_functions!(mlir_type = Int32, rust_type = i32);
mlir_dense_resource_elements_functions!(mlir_type = UInt64, rust_type = u64);
mlir_dense_resource_elements_functions!(mlir_type = Int64, rust_type = i64);
mlir_dense_resource_elements_functions!(mlir_type = Float, rust_type = f32);
mlir_dense_resource_elements_functions!(mlir_type = Double, rust_type = f64);

/// Built-in MLIR [`ElementsAttributeRef`] that represents a sparse multidimensional array using a coordinate list
/// (COO) encoding to represent the sparse elements. The element indices are stored in a two-dimensional tensor of
/// 64-bit integer values with shape `[N, dimension_count]`. This tensor specifies the indices of the elements in the
/// sparse tensor that have non-zero values (in this case there are `N` non-zero elements). The element values are
/// stored in a one-dimensional tensor with shape `[N]`, that supplies the corresponding values for the sparse indices.
///
/// # Examples
///
/// The following is an example of a [`SparseElementsAttributeRef`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```text
/// sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>
///
/// // The above attribute represents the following tensor:
/// //  [[1, 0, 0, 0],
/// //   [0, 0, 5, 0],
/// //   [0, 0, 0, 0]]
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#sparseelementsattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct SparseElementsAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> SparseElementsAttributeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`SparseElementsAttributeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirSparseElementsAttrGetTypeID()).unwrap() }
    }

    /// Returns the indices of this [`SparseElementsAttributeRef`]. These are the indices of the non-zero elements of
    /// the underlying multidimensional array, represented as 64-bit integer values.
    pub fn indices(&self) -> DenseIntegerElementsAttributeRef<'c, 't> {
        unsafe {
            DenseIntegerElementsAttributeRef::from_c_api(mlirSparseElementsAttrGetIndices(self.handle), self.context)
                .unwrap()
        }
    }

    /// Returns the values of this [`SparseElementsAttributeRef`]. These are the values of the non-zero elements
    /// of the underlying multidimensional array. They correspond to the indices returned by
    /// [`SparseElementsAttributeRef::indices`].
    pub fn values(&self) -> DenseElementsAttributeRef<'c, 't> {
        unsafe {
            DenseElementsAttributeRef::from_c_api(mlirSparseElementsAttrGetValues(self.handle), self.context).unwrap()
        }
    }
}

impl<'c, 't> ElementsAttribute<'c, 't> for SparseElementsAttributeRef<'c, 't> {}

mlir_subtype_trait_impls!(
    SparseElementsAttributeRef<'c, 't> as Attribute,
    mlir_type = Attribute,
    mlir_subtype = SparseElements,
);

impl<'t> Context<'t> {
    /// Creates a new [`SparseElementsAttributeRef`] with the specified shape with the provided indices and values.
    /// `indices` and `values` must have the same number of elements and `indices` is expected to contain 64-bit
    /// integer values.
    ///
    /// The resulting [`SparseElementsAttributeRef`] is owned by this context.
    pub fn sparse_elements_attribute<'c, T: ShapedType<'c, 't>, V: DenseElementsAttribute<'c, 't>>(
        &'c self,
        shaped_type: T,
        indices: DenseIntegerElementsAttributeRef<'c, 't>,
        values: V,
    ) -> Option<SparseElementsAttributeRef<'c, 't>> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        let _guard = self.borrow();
        unsafe {
            SparseElementsAttributeRef::from_c_api(
                mlirSparseElementsAttribute(shaped_type.to_c_api(), indices.to_c_api(), values.to_c_api()),
                &self,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::Size;
    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_dense_elements_attribute_type_id() {
        let context = Context::new();
        let location = context.unknown_location();
        let dense_elements_attribute_id = <DenseElementsAttributeRef as DenseElementsAttribute>::type_id();
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(2)], None, location).unwrap();
        let dense_elements_attribute_1 = context.dense_i32_elements_attribute(tensor_type, &[1, 2]).unwrap();
        let dense_elements_attribute_2 = context.dense_i32_elements_attribute(tensor_type, &[1, 2]).unwrap();
        assert_eq!(dense_elements_attribute_1.type_id(), dense_elements_attribute_2.type_id());
        assert_eq!(dense_elements_attribute_id, dense_elements_attribute_1.type_id());
    }

    #[test]
    fn test_dense_elements_attribute() {
        let context = Context::new();
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();

        let tensor_type = context.tensor_type(i32_type, &[Size::Static(3)], None, context.unknown_location()).unwrap();
        let attribute = context
            .dense_elements_attribute(
                tensor_type,
                &[
                    context.integer_attribute(i32_type, 1),
                    context.integer_attribute(i32_type, 2),
                    context.integer_attribute(i32_type, 3),
                ],
            )
            .unwrap();
        assert_eq!(&context, attribute.context());
        assert!(unsafe { !attribute.raw_data().is_null() });
        assert_eq!(attribute.elements_count(), 3);
        assert!(!attribute.is_splat());
        assert!(attribute.splat().is_none());
        assert!(unsafe { attribute.i32_splat().is_none() });
        assert_eq!(unsafe { attribute.i32_element(0) }, Some(1));
        assert_eq!(unsafe { attribute.i32_element(1) }, Some(2));
        assert_eq!(unsafe { attribute.i32_element(2) }, Some(3));
        assert_eq!(unsafe { attribute.i32_elements().collect::<Vec<_>>() }, vec![1, 2, 3]);
        assert_eq!(attribute.element(&[0]).unwrap(), context.integer_attribute(i32_type, 1));
        assert_eq!(attribute.element(&[1]).unwrap(), context.integer_attribute(i32_type, 2));
        assert_eq!(attribute.element(&[2]).unwrap(), context.integer_attribute(i32_type, 3));
        assert!(attribute.element(&[3]).is_none());

        let tensor_type = context.shaped_type(context.none_type(), &[Size::Static(3)]).unwrap();
        let attribute = context
            .dense_string_ref_elements_attribute(tensor_type, &["1".into(), "2".into(), "3".into()])
            .unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.elements_count(), 3);
        assert!(!attribute.is_splat());
        assert!(unsafe { attribute.string_ref_splat().is_none() });
        assert_eq!(unsafe { attribute.string_ref_element(0).unwrap().as_str() }, Ok("1"));
        assert_eq!(unsafe { attribute.string_ref_element(1).unwrap().as_str() }, Ok("2"));
        assert_eq!(unsafe { attribute.string_ref_element(2).unwrap().as_str() }, Ok("3"));
        assert!(unsafe { attribute.string_ref_element(3).is_none() });
        assert_eq!(
            unsafe { attribute.string_ref_elements().collect::<Vec<_>>() },
            vec![StringRef::from("1"), StringRef::from("2"), StringRef::from("3")]
        );

        let tensor_type = context.tensor_type(i32_type, &[Size::Static(3)], None, context.unknown_location()).unwrap();
        let attribute = context.splatted_dense_i32_elements_attribute(tensor_type, 42).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.elements_count(), 3);
        assert!(attribute.is_splat());
        assert_eq!(unsafe { attribute.i32_splat() }, Some(42));
        assert!(attribute.splat().is_some());
        assert_eq!(attribute.splat().unwrap(), context.integer_attribute(i32_type, 42));

        let tensor_type = context.tensor_type(i1_type, &[Size::Static(3)], None, context.unknown_location()).unwrap();
        let attribute = context.dense_bool_elements_attribute(tensor_type, &[true, false, true]).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.elements_count(), 3);
        assert!(!attribute.is_splat());
        assert!(attribute.splat().is_none());
        assert!(unsafe { attribute.bool_splat().is_none() });
        assert_eq!(unsafe { attribute.bool_elements().collect::<Vec<_>>() }, vec![true, false, true]);

        let tensor_type = context.tensor_type(i1_type, &[Size::Static(3)], None, context.unknown_location()).unwrap();
        let attribute = context.splatted_dense_bool_elements_attribute(tensor_type, true).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.elements_count(), 3);
        assert!(attribute.is_splat());
        assert_eq!(unsafe { attribute.bool_splat() }, Some(true));

        let tensor_type = context.shaped_type(context.none_type(), &[Size::Static(3)]).unwrap();
        let attribute = context
            .splatted_dense_attribute_elements_attribute(tensor_type, context.string_attribute("42"))
            .unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.elements_count(), 3);
        assert!(attribute.is_splat());
        assert_eq!(unsafe { attribute.string_ref_splat() }, Some("42".into()));

        let tensor_type = context.tensor_type(f32_type, &[Size::Static(2)], None, context.unknown_location()).unwrap();
        let attribute = context.dense_f32_elements_attribute(tensor_type, &[1.5, 2.5]).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.elements_count(), 2);
        assert!(unsafe { attribute.f32_element(2).is_none() });
        let elements = unsafe { attribute.f32_elements().collect::<Vec<_>>() };
        assert_eq!(elements.len(), 2);
        assert!((elements[0] - 1.5).abs() < 1e-6);
        assert!((elements[1] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_dense_elements_attribute_from_raw_buffer() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(3)], None, context.unknown_location()).unwrap();
        let attribute =
            context.dense_integer_elements_attribute_from_raw_buffer(tensor_type, &[1i32, 2i32, 3i32]).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.elements_count(), 3);
        assert!(!attribute.is_splat());
        assert_eq!(unsafe { attribute.i32_element(0) }, Some(1));
        assert_eq!(unsafe { attribute.i32_element(1) }, Some(2));
        assert_eq!(unsafe { attribute.i32_element(2) }, Some(3));
        assert_eq!(unsafe { attribute.i32_elements().collect::<Vec<_>>() }, vec![1, 2, 3]);
        assert_eq!(attribute.element(&[0]).unwrap(), context.integer_attribute(i32_type, 1));
        assert_eq!(attribute.element(&[1]).unwrap(), context.integer_attribute(i32_type, 2));
        assert_eq!(attribute.element(&[2]).unwrap(), context.integer_attribute(i32_type, 3));
    }

    #[test]
    fn test_dense_elements_attribute_reshape() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(3)], None, context.unknown_location()).unwrap();
        let attribute = context
            .dense_integer_elements_attribute(
                tensor_type,
                &[
                    context.integer_attribute(i32_type, 1),
                    context.integer_attribute(i32_type, 2),
                    context.integer_attribute(i32_type, 3),
                ],
            )
            .unwrap();
        let tensor_type = context.shaped_type(i32_type, &[Size::Static(1), Size::Static(3)]).unwrap();
        let attribute = attribute.reshape(tensor_type).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.elements_count(), 3);
        assert!(!attribute.is_splat());
        assert_eq!(unsafe { attribute.i32_element(0) }, Some(1));
        assert_eq!(unsafe { attribute.i32_element(1) }, Some(2));
        assert_eq!(unsafe { attribute.i32_element(2) }, Some(3));
        assert_eq!(unsafe { attribute.i32_elements().collect::<Vec<_>>() }, vec![1, 2, 3]);
        assert_eq!(attribute.element(&[0, 0]).unwrap(), context.integer_attribute(i32_type, 1));
        assert_eq!(attribute.element(&[0, 1]).unwrap(), context.integer_attribute(i32_type, 2));
        assert_eq!(attribute.element(&[0, 2]).unwrap(), context.integer_attribute(i32_type, 3));
    }

    #[test]
    fn test_dense_elements_attribute_equality() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(2)], None, context.unknown_location()).unwrap();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context
            .splatted_dense_integer_attribute_elements_attribute(tensor_type, context.integer_attribute(i32_type, 42))
            .unwrap();
        let attribute_2 = context.splatted_dense_i32_elements_attribute(tensor_type, 42).unwrap();
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.splatted_dense_i32_elements_attribute(tensor_type, 100).unwrap();
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(2)], None, context.unknown_location()).unwrap();
        let attribute_2 = context.splatted_dense_i32_elements_attribute(tensor_type, 42).unwrap();
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_dense_elements_attribute_display_and_debug() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(2)], None, context.unknown_location()).unwrap();
        let attribute = context.splatted_dense_i32_elements_attribute(tensor_type, 10).unwrap();
        test_attribute_display_and_debug(attribute, "dense<10> : tensor<2xi32>");
    }

    #[test]
    fn test_dense_elements_attribute_parse() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(2)], None, context.unknown_location()).unwrap();
        assert_eq!(
            context.parse_attribute("dense<10> : tensor<2xi32>").unwrap(),
            context.splatted_dense_i32_elements_attribute(tensor_type, 10).unwrap(),
        );
    }

    #[test]
    fn test_dense_elements_attribute_casting() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(2)], None, context.unknown_location()).unwrap();
        let attribute = context.splatted_dense_i32_elements_attribute(tensor_type, 10).unwrap();
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_dense_integer_elements_attribute() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);

        let tensor_type = context.tensor_type(i64_type, &[Size::Static(3)], None, context.unknown_location()).unwrap();
        let attribute = context.dense_i64_elements_attribute(tensor_type, &[10, 20, 30]).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.elements_count(), 3);
        assert_eq!(unsafe { attribute.i64_element(0) }, Some(10));
        assert_eq!(unsafe { attribute.i64_element(1) }, Some(20));
        assert_eq!(unsafe { attribute.i64_element(2) }, Some(30));
        assert_eq!(unsafe { attribute.i64_elements().collect::<Vec<_>>() }, vec![10, 20, 30]);

        let tensor_type = context.tensor_type(i32_type, &[Size::Static(4)], None, context.unknown_location()).unwrap();
        let attribute = context.splatted_dense_i32_elements_attribute(tensor_type, 7).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.elements_count(), 4);
        assert!(attribute.is_splat());
        assert_eq!(unsafe { attribute.i32_splat() }, Some(7));
    }

    #[test]
    fn test_dense_integer_elements_attribute_equality() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(2)], None, context.unknown_location()).unwrap();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.dense_i32_elements_attribute(tensor_type, &[5, 10]).unwrap();
        let attribute_2 = context.dense_i32_elements_attribute(tensor_type, &[5, 10]).unwrap();
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.dense_i32_elements_attribute(tensor_type, &[5, 20]).unwrap();
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(2)], None, context.unknown_location()).unwrap();
        let attribute_2 = context.dense_i32_elements_attribute(tensor_type, &[5, 10]).unwrap();
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_dense_integer_elements_attribute_display_and_debug() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(2)], None, context.unknown_location()).unwrap();
        let attribute = context.dense_i32_elements_attribute(tensor_type, &[1, 2]).unwrap();
        test_attribute_display_and_debug(attribute, "dense<[1, 2]> : tensor<2xi32>");
    }

    #[test]
    fn test_dense_integer_elements_attribute_parse() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(2)], None, context.unknown_location()).unwrap();
        assert_eq!(
            context.parse_attribute("dense<[1, 2]> : tensor<2xi32>").unwrap(),
            context.dense_i32_elements_attribute(tensor_type, &[1, 2]).unwrap(),
        );
    }

    #[test]
    fn test_dense_integer_elements_attribute_casting() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(2)], None, context.unknown_location()).unwrap();
        let attribute = context.dense_i32_elements_attribute(tensor_type, &[1, 2]).unwrap();
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_dense_float_elements_attribute() {
        let context = Context::new();
        let f32_type = context.float32_type();
        let f64_type = context.float64_type();

        let tensor_type = context.tensor_type(f32_type, &[Size::Static(3)], None, context.unknown_location()).unwrap();
        let attribute = context.dense_f32_elements_attribute(tensor_type, &[1.0, 2.5, 3.7]).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.elements_count(), 3);
        assert!((unsafe { attribute.f32_element(0) }.unwrap() - 1.0).abs() < 1e-6);
        assert!((unsafe { attribute.f32_element(1) }.unwrap() - 2.5).abs() < 1e-6);
        assert!((unsafe { attribute.f32_element(2) }.unwrap() - 3.7).abs() < 1e-6);
        assert_eq!(unsafe { attribute.f32_elements() }.collect::<Vec<_>>().len(), 3);

        let tensor_type = context.tensor_type(f64_type, &[Size::Static(4)], None, context.unknown_location()).unwrap();
        let attribute = context.splatted_dense_f64_elements_attribute(tensor_type, 3.14).unwrap();
        assert_eq!(&context, attribute.context());
        assert!(attribute.is_splat());
        assert!((unsafe { attribute.f64_splat() }.unwrap() - 3.14).abs() < 1e-6);
    }

    #[test]
    fn test_dense_float_elements_attribute_equality() {
        let context = Context::new();
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(2)], None, context.unknown_location()).unwrap();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.dense_f32_elements_attribute(tensor_type, &[1.5, 2.5]).unwrap();
        let attribute_2 = context.dense_f32_elements_attribute(tensor_type, &[1.5, 2.5]).unwrap();
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.dense_f32_elements_attribute(tensor_type, &[1.5, 3.5]).unwrap();
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(2)], None, context.unknown_location()).unwrap();
        let attribute_2 = context.dense_f32_elements_attribute(tensor_type, &[1.5, 2.5]).unwrap();
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_dense_float_elements_attribute_display_and_debug() {
        let context = Context::new();
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(2)], None, context.unknown_location()).unwrap();
        let attribute = context.dense_f32_elements_attribute(tensor_type, &[1.5, 2.5]).unwrap();
        test_attribute_display_and_debug(attribute, "dense<[1.500000e+00, 2.500000e+00]> : tensor<2xf32>");
    }

    #[test]
    fn test_dense_float_elements_attribute_parse() {
        let context = Context::new();
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(2)], None, context.unknown_location()).unwrap();
        assert_eq!(
            context.parse_attribute("dense<[1.500000e+00, 2.500000e+00]> : tensor<2xf32>").unwrap(),
            context.dense_f32_elements_attribute(tensor_type, &[1.5, 2.5]).unwrap(),
        );
    }

    #[test]
    fn test_dense_float_elements_attribute_casting() {
        let context = Context::new();
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(2)], None, context.unknown_location()).unwrap();
        let attribute = context.dense_f32_elements_attribute(tensor_type, &[1.5, 2.5]).unwrap();
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_dense_resource_elements_attribute() {
        let context = Context::new();
        let i8_type = context.signless_integer_type(8);
        let tensor_type = context.tensor_type(i8_type, &[Size::Static(3)], None, context.unknown_location()).unwrap();
        let attribute = context
            .dense_i8_resource_elements_attribute(tensor_type, StringRef::from("test_blob"), &[10, 20, 30])
            .unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.elements_count(), 3);
        assert_eq!(unsafe { attribute.i8_element(0) }, Some(10));
        assert_eq!(unsafe { attribute.i8_element(1) }, Some(20));
        assert_eq!(unsafe { attribute.i8_element(2) }, Some(30));
        assert!(unsafe { attribute.i8_element(3).is_none() });
        assert_eq!(unsafe { attribute.i8_elements().collect::<Vec<_>>() }, vec![10, 20, 30]);
    }

    #[test]
    fn test_dense_resource_elements_attribute_equality() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(2)], None, context.unknown_location()).unwrap();

        // Different resource names should create different attributes.
        let attribute_1 = context
            .dense_i32_resource_elements_attribute(tensor_type, StringRef::from("blob1"), &[5, 10])
            .unwrap();
        let attribute_2 = context
            .dense_i32_resource_elements_attribute(tensor_type, StringRef::from("blob2"), &[5, 10])
            .unwrap();
        assert_ne!(attribute_1, attribute_2);

        // Same resource name from different contexts must not be equal.
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(2)], None, context.unknown_location()).unwrap();
        let attribute_2 = context
            .dense_i32_resource_elements_attribute(tensor_type, StringRef::from("blob1"), &[5, 10])
            .unwrap();
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_dense_resource_elements_attribute_casting() {
        let context = Context::new();
        let i1_type = context.signless_integer_type(1);
        let tensor_type = context.tensor_type(i1_type, &[Size::Static(2)], None, context.unknown_location()).unwrap();
        let attribute = context
            .dense_bool_resource_elements_attribute(tensor_type, StringRef::from("test"), &[true, false])
            .unwrap();
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_unmanaged_dense_resource_elements_attribute() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(3)], None, context.unknown_location()).unwrap();
        let attribute = unsafe {
            context
                .unmanaged_dense_resource_elements_attribute(
                    tensor_type,
                    StringRef::from("test_blob"),
                    &[10i32, 20i32, 30i32],
                )
                .unwrap()
        };
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.elements_count(), 3);
        assert_eq!(unsafe { attribute.i32_element(0) }, Some(10));
        assert_eq!(unsafe { attribute.i32_element(1) }, Some(20));
        assert_eq!(unsafe { attribute.i32_element(2) }, Some(30));
        assert!(unsafe { attribute.i32_element(3).is_none() });
        assert_eq!(unsafe { attribute.i32_elements().collect::<Vec<_>>() }, vec![10, 20, 30]);
    }

    #[test]
    fn test_unmanaged_dense_resource_elements_attribute_equality() {
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(2)], None, context.unknown_location()).unwrap();

        // Different resource names should create different attributes.
        let attribute_1 = unsafe {
            context
                .unmanaged_dense_resource_elements_attribute(tensor_type, StringRef::from("blob1"), &[5i32, 10i32])
                .unwrap()
        };
        let attribute_2 = unsafe {
            context
                .unmanaged_dense_resource_elements_attribute(tensor_type, StringRef::from("blob2"), &[5i32, 10i32])
                .unwrap()
        };
        assert_ne!(attribute_1, attribute_2);

        // Same resource name from different contexts must not be equal.
        let context = Context::new();
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(2)], None, context.unknown_location()).unwrap();
        let attribute_2 = unsafe {
            context
                .unmanaged_dense_resource_elements_attribute(tensor_type, StringRef::from("blob1"), &[5i32, 10i32])
                .unwrap()
        };
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_unmanaged_dense_resource_elements_attribute_casting() {
        let context = Context::new();
        let f64_type = context.float64_type();
        let tensor_type = context.tensor_type(f64_type, &[Size::Static(2)], None, context.unknown_location()).unwrap();
        let attribute = unsafe {
            context
                .unmanaged_dense_resource_elements_attribute(tensor_type, StringRef::from("test"), &[1.5f64, 2.5f64])
                .unwrap()
        };
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_sparse_elements_attribute_type_id() {
        let context = Context::new();
        let location = context.unknown_location();
        let sparse_elements_attribute_id = SparseElementsAttributeRef::type_id();
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(2)], None, location).unwrap();
        let indices_type = context.tensor_type(i32_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        let values_type = context.tensor_type(i32_type, &[Size::Static(2)], None, location).unwrap();
        let indices = context.dense_i64_elements_attribute(indices_type, &[0, 0, 1, 2]).unwrap();
        let values = context.dense_i32_elements_attribute(values_type, &[1, 5]).unwrap();
        let sparse_elements_attribute_1 = context.sparse_elements_attribute(tensor_type, indices, values).unwrap();
        let sparse_elements_attribute_2 = context.sparse_elements_attribute(tensor_type, indices, values).unwrap();
        assert_eq!(sparse_elements_attribute_1.type_id(), sparse_elements_attribute_2.type_id());
        assert_eq!(sparse_elements_attribute_id, sparse_elements_attribute_1.type_id());
        assert_ne!(sparse_elements_attribute_id, <DenseElementsAttributeRef as DenseElementsAttribute>::type_id());
    }

    #[test]
    fn test_sparse_elements_attribute() {
        let context = Context::new();
        let location = context.unknown_location();
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(3), Size::Static(4)], None, location).unwrap();

        // Create indices tensor [[0, 0], [1, 2]] that corresponds to 2 non-zero elements, at (0, 0) and (1, 2).
        let indices_type = context.tensor_type(i64_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        let indices = context.dense_i64_elements_attribute(indices_type, &[0, 0, 1, 2]).unwrap();

        // Create values [1, 5].
        let values_type = context.tensor_type(i32_type, &[Size::Static(2)], None, location).unwrap();
        let values = context.dense_i32_elements_attribute(values_type, &[1, 5]).unwrap();

        // Create sparse elements attribute.
        let attribute = context.sparse_elements_attribute(tensor_type, indices, values).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.elements_count(), 12);
        assert_eq!(attribute.indices().elements_count(), 4);
        assert_eq!(attribute.values().elements_count(), 2);
        assert_eq!(unsafe { attribute.indices().i64_elements().collect::<Vec<_>>() }, vec![0, 0, 1, 2]);
        assert_eq!(unsafe { attribute.values().i32_elements().collect::<Vec<_>>() }, vec![1, 5]);
    }

    #[test]
    fn test_sparse_elements_attribute_equality() {
        let context = Context::new();
        let location = context.unknown_location();
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        let indices_type = context.tensor_type(i64_type, &[Size::Static(1), Size::Static(2)], None, location).unwrap();
        let values_type = context.tensor_type(i32_type, &[Size::Static(1)], None, location).unwrap();
        let indices = context.dense_i64_elements_attribute(indices_type, &[0, 0]).unwrap();
        let values_1 = context.dense_i32_elements_attribute(values_type, &[42]).unwrap();
        let values_2 = context.dense_i32_elements_attribute(values_type, &[42]).unwrap();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.sparse_elements_attribute(tensor_type, indices, values_1).unwrap();
        let attribute_2 = context.sparse_elements_attribute(tensor_type, indices, values_2).unwrap();
        assert_eq!(attribute_1, attribute_2);

        // Different values should create different attributes.
        let values_3 = context.dense_i32_elements_attribute(values_type, &[100]).unwrap();
        let attribute_3 = context.sparse_elements_attribute(tensor_type, indices, values_3).unwrap();
        assert_ne!(attribute_1, attribute_3);
    }

    #[test]
    fn test_sparse_elements_attribute_display_and_debug() {
        let context = Context::new();
        let location = context.unknown_location();
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(3), Size::Static(4)], None, location).unwrap();
        let indices_type = context.tensor_type(i64_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        let values_type = context.tensor_type(i32_type, &[Size::Static(2)], None, location).unwrap();
        let indices = context.dense_i64_elements_attribute(indices_type, &[0, 0, 1, 2]).unwrap();
        let values = context.dense_i32_elements_attribute(values_type, &[1, 5]).unwrap();
        let attribute = context.sparse_elements_attribute(tensor_type, indices, values).unwrap();
        test_attribute_display_and_debug(attribute, "sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>");
    }

    #[test]
    fn test_sparse_elements_attribute_parse() {
        let context = Context::new();
        let location = context.unknown_location();
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(3), Size::Static(4)], None, location).unwrap();
        let indices_type = context.tensor_type(i64_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        let values_type = context.tensor_type(i32_type, &[Size::Static(2)], None, location).unwrap();
        let indices = context.dense_i64_elements_attribute(indices_type, &[0, 0, 1, 2]).unwrap();
        let values = context.dense_i32_elements_attribute(values_type, &[1, 5]).unwrap();
        assert_eq!(
            context.parse_attribute("sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>").unwrap(),
            context.sparse_elements_attribute(tensor_type, indices, values).unwrap(),
        );
    }

    #[test]
    fn test_sparse_elements_attribute_casting() {
        let context = Context::new();
        let location = context.unknown_location();
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        let indices_type = context.tensor_type(i64_type, &[Size::Static(1), Size::Static(2)], None, location).unwrap();
        let values_type = context.tensor_type(i32_type, &[Size::Static(1)], None, location).unwrap();
        let indices = context.dense_i64_elements_attribute(indices_type, &[0, 0]).unwrap();
        let values = context.dense_i32_elements_attribute(values_type, &[42]).unwrap();
        let attribute = context.sparse_elements_attribute(tensor_type, indices, values).unwrap();
        test_attribute_casting(attribute);
    }
}
