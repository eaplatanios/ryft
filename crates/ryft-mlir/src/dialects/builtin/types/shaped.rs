use ryft_xla_sys::bindings::{
    MlirType, mlirRankedTensorTypeGet, mlirShapedTypeGetDynamicSize, mlirShapedTypeGetElementType,
};

use crate::{Attribute, Context, Type, TypeRef, mlir_subtype_trait_impls};

/// Represents a size quantity (e.g., of a tensor dimension, stride, or offset) that can be
/// either statically known or dynamically determined.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Size {
    Static(usize),
    Dynamic,
}

impl Size {
    /// Constructs a new [`Size`] from its corresponding representation in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn from_c_api(value: i64) -> Self {
        if value == unsafe { mlirShapedTypeGetDynamicSize() } { Self::Dynamic } else { Self::Static(value as usize) }
    }

    /// Returns the MLIR C API representation of this [`Size`].
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn to_c_api(&self) -> i64 {
        match *self {
            Size::Static(value) => value as i64,
            Size::Dynamic => unsafe { mlirShapedTypeGetDynamicSize() },
        }
    }

    /// Returns the value of this [`Size`], if it is a [`Size::Static`], and [`None`] otherwise.
    pub fn value(&self) -> Option<usize> {
        match self {
            Size::Static(value) => Some(*value),
            Size::Dynamic => None,
        }
    }

    /// Returns `true` if this [`Size`] is statically known.
    pub fn is_static(&self) -> bool {
        matches!(self, Self::Static(_))
    }

    /// Returns `true` if this [`Size`] is dynamically determined.
    pub fn is_dynamic(&self) -> bool {
        matches!(self, Self::Dynamic)
    }
}

/// Built-in MLIR [`Type`] that represents types which themselves represent various forms of multidimensional arrays
/// that have a (potentially dynamic) shape and an element [`Type`] (e.g., [`VectorTypeRef`](crate::VectorTypeRef),
/// [`TensorTypeRef`](crate::TensorTypeRef), [`UnrankedTensorTypeRef`](crate::UnrankedTensorTypeRef),
/// [`MemRefTypeRef`](crate::MemRefTypeRef), and [`UnrankedMemRefTypeRef`](crate::UnrankedMemRefTypeRef)).
///
/// This `struct` acts effectively as the super-type of all MLIR shaped [`Type`]s and can be checked and
/// specialized using the [`ShapedType::is`](TypeRef::is) and [`ShapedType::cast`](TypeRef::cast) functions.
pub trait ShapedType<'c, 't: 'c>: Type<'c, 't> {
    /// Returns element [`Type`] of this shaped type.
    fn element_type(&self) -> TypeRef<'c, 't> {
        unsafe { TypeRef::from_c_api(mlirShapedTypeGetElementType(self.to_c_api()), self.context()).unwrap() }
    }
}

/// Reference to a [`ShapedType`] that is owned by a [`Context`].
#[derive(Copy, Clone)]
pub struct ShapedTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

mlir_subtype_trait_impls!(ShapedTypeRef<'c, 't> as Type, mlir_type = Type, mlir_subtype = Shaped);

impl<'c, 't> ShapedType<'c, 't> for ShapedTypeRef<'c, 't> {}

impl<'t> Context<'t> {
    /// Creates a new [`ShapedTypeRef`] owned by this [`Context`]. If any of the arguments are invalid, then this
    /// function will return [`None`] and will also emit the appropriate diagnostics at the provided location.
    pub fn shaped_type<'c, T: Type<'c, 't>>(
        &'c self,
        element_type: T,
        shape: &[Size],
    ) -> Option<ShapedTypeRef<'c, 't>> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        let _guard = self.borrow();
        unsafe {
            let dimensions = shape.iter().map(|dimension| dimension.to_c_api()).collect::<Vec<_>>();
            ShapedTypeRef::from_c_api(
                mlirRankedTensorTypeGet(
                    dimensions.len().cast_signed(),
                    dimensions.as_ptr(),
                    element_type.to_c_api(),
                    self.null_attribute().to_c_api(),
                ),
                self,
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
    fn test_size() {
        assert_eq!(Size::Static(4).value(), Some(4));
        assert_eq!(Size::Dynamic.value(), None);
        assert!(!Size::Static(4).is_dynamic());
        assert!(Size::Dynamic.is_dynamic());
        assert_eq!(unsafe { Size::from_c_api(Size::Static(4).to_c_api()) }, Size::Static(4));
        assert_eq!(unsafe { Size::from_c_api(Size::Dynamic.to_c_api()) }, Size::Dynamic);
    }

    #[test]
    fn test_shaped_type() {
        let context = Context::new();
        let element_type = context.float32_type();
        let shape = vec![Size::Static(4), Size::Dynamic];
        let tensor_type = context.shaped_type(element_type, &shape).unwrap();
        let shaped_type = tensor_type.cast::<ShapedTypeRef>().unwrap();
        assert_eq!(&context, shaped_type.context());
        assert_eq!(shaped_type.element_type(), element_type);
    }

    #[test]
    fn test_shaped_type_equality() {
        let context = Context::new();
        let element_type = context.bfloat16_type();
        let shape = vec![Size::Static(32), Size::Dynamic];

        // Same types from the same context must be equal because they are "uniqued".
        let type_1 = context.shaped_type(element_type, &shape);
        let type_2 = context.shaped_type(element_type, &shape);
        assert_eq!(type_1, type_2);

        // Different shapes from the same context must not be equal.
        let shape = vec![Size::Static(32), Size::Static(16)];
        let type_2 = context.shaped_type(element_type, &shape);
        assert_ne!(type_1, type_2);

        // Same types from different contexts must not be equal.
        let context = Context::new();
        let element_type = context.bfloat16_type();
        let type_2 = context.shaped_type(element_type, &shape);
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_shaped_type_display_and_debug() {
        let context = Context::new();
        let element_type = context.float32_type();
        let shape = vec![Size::Static(4), Size::Dynamic];
        let tensor_type = context.shaped_type(element_type, &shape).unwrap();
        let shaped_type = tensor_type.cast::<ShapedTypeRef>().unwrap();
        test_type_display_and_debug(shaped_type, "tensor<4x?xf32>");
    }

    #[test]
    fn test_shaped_type_casting() {
        let context = Context::new();
        let element_type = context.float32_type();
        let shape = vec![Size::Static(4), Size::Dynamic];
        let tensor_type = context.shaped_type(element_type, &shape).unwrap();
        let shaped_type = tensor_type.cast::<ShapedTypeRef>().unwrap();
        test_type_casting(shaped_type);
    }
}
