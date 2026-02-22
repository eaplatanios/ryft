use ryft_xla_sys::bindings::{
    MlirType, mlirRankedTensorTypeGetChecked, mlirRankedTensorTypeGetEncoding, mlirRankedTensorTypeGetTypeID,
    mlirShapedTypeGetDimSize, mlirShapedTypeGetRank, mlirShapedTypeIsStaticDim, mlirUnrankedTensorTypeGetChecked,
    mlirUnrankedTensorTypeGetTypeID,
};

use crate::{Attribute, AttributeRef, Context, Location, Type, TypeId, mlir_subtype_trait_impls};

use super::{ShapedType, Size};

/// Built-in MLIR [`Type`] that represents multidimensional arrays with a fixed number of dimensions. This is in
/// contrast to [`UnrankedTensorTypeRef`] which is used to represent tensors with an unknown number of dimensions.
///
/// Values of this type are `N`-dimensional arrays that have a known element [`Type`] and a fixed rank `N`.
/// Each dimension may have a statically known size or a dynamically determined size (indicated by the `?`
/// character in the string rendering of the type).
///
/// The runtime representation of MLIR [`TensorTypeRef`]s is intentionally abstracted; you cannot control
/// its layout or get a pointer to the underlying data. For low level buffer access, you should use the
/// [`MemRefTypeRef`](crate::MemRefTypeRef). This abstracted runtime representation holds the tensor data values
/// as well as information about the (potentially dynamic) shape of the tensor.
///
/// The [`TensorTypeRef::encoding`] attribute provides additional information about the tensors the type
/// represents. An empty attribute denotes a straightforward tensor without any specific structure. But particular
/// properties, like sparsity or other specific characteristics of the data of the tensor can be encoded through this
/// attribute. The semantics are defined by a type and attribute interface and must be respected by all passes that
/// operate on [`TensorTypeRef`]s.
///
/// # Examples
///
/// The following are examples of [`TensorTypeRef`]s represented using their [`Display`](std::fmt::Display) rendering:
///
/// ```text
/// tensor<?x?x?x?xf32>         => 4D tensor with all of its dimensions being dynamic.
/// tensor<?x?x13x?xf32>        => 4D tensor with its third dimension being static and the rest dynamic.
/// tensor<17x4x13x4xf32>       => 4D tensor with all of its dimensions being static.
/// tensor<f32>                 => 0D tensor (i.e., a scalar).
/// tensor<0x42xf32>            => 0-sized dimensions are allowed in tensors.
/// tensor<?x?xf64, #ENCODING>  => 2D tensor with an encoding attribute (where #ENCODING is a named alias).
/// ```
///
/// Refer to the [MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#rankedtensortype)
/// for more information.
#[derive(Copy, Clone)]
pub struct TensorTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> TensorTypeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`TensorTypeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirRankedTensorTypeGetTypeID()).unwrap() }
    }

    /// Returns the rank of this [`TensorTypeRef`] (i.e., the number of dimensions it has).
    pub fn rank(&self) -> usize {
        unsafe { mlirShapedTypeGetRank(self.handle) as usize }
    }

    /// Returns all dimension [`Size`]s of this [`TensorTypeRef`].
    pub fn dimensions(&self) -> impl Iterator<Item = Size> {
        (0..self.rank()).map(|dimension| self.dimension(dimension))
    }

    /// Returns `true` if all dimensions of this [`TensorTypeRef`] has a static size.
    pub fn has_static_shape(&self) -> bool {
        self.dimensions().all(|dimension| dimension.is_static())
    }

    /// Returns the `dimension`-th dimension [`Size`] of this [`TensorTypeRef`].
    ///
    /// Note that this function will panic if the provided dimension is out of bounds.
    pub fn dimension(&self, dimension: usize) -> Size {
        if dimension >= self.rank() {
            panic!("dimension is out of bounds");
        }
        unsafe {
            if mlirShapedTypeIsStaticDim(self.handle, dimension.cast_signed()) {
                Size::Static(mlirShapedTypeGetDimSize(self.handle, dimension.cast_signed()) as usize)
            } else {
                Size::Dynamic
            }
        }
    }

    /// Returns the encoding [`Attribute`] of this [`TensorTypeRef`].
    pub fn encoding(&self) -> Option<AttributeRef<'c, 't>> {
        unsafe { AttributeRef::from_c_api(mlirRankedTensorTypeGetEncoding(self.handle), self.context) }
    }
}

impl<'c, 't> ShapedType<'c, 't> for TensorTypeRef<'c, 't> {}

mlir_subtype_trait_impls!(TensorTypeRef<'c, 't> as Type, mlir_type = Type, mlir_subtype = RankedTensor);

impl<'t> Context<'t> {
    /// Creates a new [`TensorTypeRef`] owned by this [`Context`]. If any of the arguments are invalid, then this
    /// function will return [`None`] and will also emit the appropriate diagnostics at the provided location. For
    /// example, this can happen if `element_type` is not an [`IndexTypeRef`](crate::IndexTypeRef), an
    /// [`IntegerTypeRef`](crate::IntegerTypeRef), or a [`FloatType`](crate::FloatTypeRef).
    pub fn tensor_type<'c, T: Type<'c, 't>, L: Location<'c, 't>>(
        &'c self,
        element_type: T,
        shape: &[Size],
        encoding: Option<AttributeRef<'c, 't>>,
        location: L,
    ) -> Option<TensorTypeRef<'c, 't>> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        let _guard = self.borrow();
        unsafe {
            let dimensions = shape.iter().map(|dimension| dimension.to_c_api()).collect::<Vec<_>>();
            TensorTypeRef::from_c_api(
                mlirRankedTensorTypeGetChecked(
                    location.to_c_api(),
                    dimensions.len().cast_signed(),
                    dimensions.as_ptr(),
                    element_type.to_c_api(),
                    encoding.unwrap_or_else(|| self.null_attribute()).to_c_api(),
                ),
                &self,
            )
        }
    }
}

/// Built-in MLIR [`Type`] that represents multidimensional arrays with an unknown number of dimensions.
/// This is in contrast to [`TensorTypeRef`] which is used to represent tensors with a fixed number of dimensions.
///
/// # Examples
///
/// The following are examples of [`UnrankedTensorTypeRef`]s represented using their
/// [`Display`](std::fmt::Display) rendering:
///
/// ```text
/// tensor<*xf32>   => Unranked tensor that contains f32 elements.
/// tensor<*xbf16>  => Unranked tensor that contains bf16 elements.
/// ```
///
/// Refer to the [MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#unrankedtensortype)
/// for more information.
#[derive(Copy, Clone)]
pub struct UnrankedTensorTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> UnrankedTensorTypeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`UnrankedTensorTypeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirUnrankedTensorTypeGetTypeID()).unwrap() }
    }
}

impl<'c, 't> ShapedType<'c, 't> for UnrankedTensorTypeRef<'c, 't> {}

mlir_subtype_trait_impls!(UnrankedTensorTypeRef<'c, 't> as Type, mlir_type = Type, mlir_subtype = UnrankedTensor);

impl<'t> Context<'t> {
    /// Creates a new [`UnrankedTensorTypeRef`] owned by this [`Context`]. If any of the arguments are invalid, then
    /// this function will return [`None`] and will also emit the appropriate diagnostics at the provided location. For
    /// example, this can happen if `element_type` is not an [`IndexTypeRef`](crate::IndexTypeRef), an
    /// [`IntegerTypeRef`](crate::IntegerTypeRef), or a [`FloatType`](crate::FloatTypeRef).
    pub fn unranked_tensor_type<'c, T: Type<'c, 't>, L: Location<'c, 't>>(
        &'c self,
        element_type: T,
        location: L,
    ) -> Option<UnrankedTensorTypeRef<'c, 't>> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        let _guard = self.borrow();
        unsafe {
            UnrankedTensorTypeRef::from_c_api(
                mlirUnrankedTensorTypeGetChecked(location.to_c_api(), element_type.to_c_api()),
                &self,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::types::tests::{test_type_casting, test_type_display_and_debug};

    use super::*;

    #[test]
    fn test_tensor_type_type_id() {
        let context = Context::new();
        let location = context.unknown_location();
        let tensor_type = TensorTypeRef::type_id();
        let tensor_type_1 = context.tensor_type(context.float32_type(), &[Size::Static(32)], None, location);
        let tensor_type_2 = context.tensor_type(context.float32_type(), &[Size::Static(64)], None, location);
        assert_eq!(tensor_type_1.unwrap().type_id(), tensor_type_2.unwrap().type_id());
        assert_eq!(tensor_type, tensor_type_1.unwrap().type_id());
    }

    #[test]
    fn test_tensor_type() {
        let context = Context::new();
        let location = context.unknown_location();

        // Valid element type.
        let element_type = context.bfloat16_type();
        let shape = vec![Size::Static(32), Size::Dynamic, Size::Dynamic, Size::Static(2)];
        let r#type = context.tensor_type(element_type, &shape, None, location).unwrap();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.rank(), 4);
        assert_eq!(r#type.dimensions().collect::<Vec<_>>(), shape);
        assert_eq!(r#type.dimension(0), Size::Static(32));
        assert_eq!(r#type.dimension(1), Size::Dynamic);
        assert_eq!(r#type.dimension(2), Size::Dynamic);
        assert_eq!(r#type.dimension(3), Size::Static(2));
        assert!(!r#type.has_static_shape());
        assert_eq!(r#type.element_type(), element_type);
        assert!(r#type.encoding().is_none());

        // Invalid element type.
        let element_type = context.none_type();
        let shape = vec![Size::Static(32), Size::Dynamic, Size::Dynamic, Size::Static(2)];
        let r#type = context.tensor_type(element_type, &shape, None, location);
        assert!(r#type.is_none());
    }

    #[test]
    fn test_tensor_type_equality() {
        let context = Context::new();
        let location = context.unknown_location();
        let element_type = context.bfloat16_type();
        let shape = vec![Size::Static(32), Size::Dynamic];

        // Same types from the same context must be equal because they are "uniqued".
        let type_1 = context.tensor_type(element_type, &shape, None, location);
        let type_2 = context.tensor_type(element_type, &shape, None, location);
        assert_eq!(type_1, type_2);

        // Different shapes from the same context must not be equal.
        let shape = vec![Size::Static(32), Size::Static(16)];
        let type_2 = context.tensor_type(element_type, &shape, None, location);
        assert_ne!(type_1, type_2);

        // Same types from different contexts must not be equal.
        let context = Context::new();
        let element_type = context.bfloat16_type();
        let type_2 = context.tensor_type(element_type, &shape, None, location);
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_tensor_type_display_and_debug() {
        let context = Context::new();
        let location = context.unknown_location();
        let element_type = context.bfloat16_type();
        let shape = vec![Size::Static(32), Size::Dynamic, Size::Dynamic, Size::Static(2)];
        let r#type = context.tensor_type(element_type, &shape, None, location).unwrap();
        test_type_display_and_debug(r#type, "tensor<32x?x?x2xbf16>");
    }

    #[test]
    fn test_tensor_type_parsing() {
        let context = Context::new();
        let location = context.unknown_location();
        let element_type = context.bfloat16_type();
        let shape = vec![Size::Static(32), Size::Dynamic, Size::Dynamic, Size::Static(2)];
        assert_eq!(
            context.parse_type("tensor<32x?x?x2xbf16>").unwrap(),
            context.tensor_type(element_type, &shape, None, location).unwrap()
        );
    }

    #[test]
    fn test_tensor_type_casting() {
        let context = Context::new();
        let location = context.unknown_location();
        let element_type = context.bfloat16_type();
        let shape = vec![Size::Static(32), Size::Dynamic];
        let r#type = context.tensor_type(element_type, &shape, None, location).unwrap();
        test_type_casting(r#type);
    }

    #[test]
    fn test_unranked_tensor_type_type_id() {
        let context = Context::new();
        let location = context.unknown_location();
        let unranked_tensor_type = UnrankedTensorTypeRef::type_id();
        let unranked_tensor_type_1 = context.unranked_tensor_type(context.float32_type(), location);
        let unranked_tensor_type_2 = context.unranked_tensor_type(context.bfloat16_type(), location);
        assert_eq!(unranked_tensor_type_1.unwrap().type_id(), unranked_tensor_type_2.unwrap().type_id());
        assert_eq!(unranked_tensor_type, unranked_tensor_type_1.unwrap().type_id());
    }

    #[test]
    fn test_unranked_tensor_type() {
        let context = Context::new();
        let location = context.unknown_location();

        // Valid element type.
        let element_type = context.signless_integer_type(1);
        let r#type = context.unranked_tensor_type(element_type, location).unwrap();
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.element_type(), element_type);

        // Invalid element type.
        let element_type = context.none_type();
        let r#type = context.unranked_tensor_type(element_type, location);
        assert!(r#type.is_none());
    }

    #[test]
    fn test_unranked_tensor_type_equality() {
        let context = Context::new();
        let location = context.unknown_location();
        let element_type = context.signless_integer_type(1);

        // Same types from the same context must be equal because they are "uniqued".
        let type_1 = context.unranked_tensor_type(element_type, location);
        let type_2 = context.unranked_tensor_type(element_type, location);
        assert_eq!(type_1, type_2);

        // Different element types from the same context must not be equal.
        let other_element_type = context.float32_type();
        let type_2 = context.unranked_tensor_type(other_element_type, location);
        assert_ne!(type_1, type_2);

        // Same types from different contexts must not be equal.
        let context = Context::new();
        let other_element_type = context.bfloat16_type();
        let type_2 = context.unranked_tensor_type(other_element_type, location);
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_unranked_tensor_type_display_and_debug() {
        let context = Context::new();
        let location = context.unknown_location();
        let element_type = context.unsigned_integer_type(16);
        let r#type = context.unranked_tensor_type(element_type, location).unwrap();
        test_type_display_and_debug(r#type, "tensor<*xui16>");
    }

    #[test]
    fn test_unranked_tensor_type_parsing() {
        let context = Context::new();
        let location = context.unknown_location();
        let element_type = context.unsigned_integer_type(16);
        assert_eq!(
            context.parse_type("tensor<*xui16>").unwrap(),
            context.unranked_tensor_type(element_type, location).unwrap()
        );
    }

    #[test]
    fn test_unranked_tensor_type_casting() {
        let context = Context::new();
        let location = context.unknown_location();
        let element_type = context.complex_type(context.bfloat16_type());
        let r#type = context.unranked_tensor_type(element_type, location).unwrap();
        test_type_casting(r#type);
    }
}
