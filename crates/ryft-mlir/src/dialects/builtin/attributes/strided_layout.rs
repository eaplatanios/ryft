use ryft_xla_sys::bindings::{
    MlirAttribute, mlirStridedLayoutAttrGet, mlirStridedLayoutAttrGetNumStrides, mlirStridedLayoutAttrGetOffset,
    mlirStridedLayoutAttrGetStride, mlirStridedLayoutAttrGetTypeID,
};

use crate::{Attribute, Context, TypeId, mlir_subtype_trait_impls};

/// Built-in MLIR [`Attribute`] that represents the strided layout of a [`MemRefTypeRef`](crate::MemRefTypeRef).
///
/// This attribute captures layout information of [`MemRefTypeRef`](crate::MemRefTypeRef)s in a canonical form.
/// Specifically, it contains a list of strides, one for each dimension. A stride is the number of elements in the
/// linear storage one must step over to reflect an increment in the given dimension. For example, an  M x N row-major
/// contiguous [`MemRefTypeRef`](crate::MemRefTypeRef) would have the strides `[N, 1]`. [`StridedLayoutAttributeRef`]s
/// also contain the offset from the base pointer of the [`MemRefTypeRef`](crate::MemRefTypeRef) to the first
/// effectively accessed element, expressed as a number of contiguously stored elements.
///
/// Strides must be positive and the offset must be non-negative. Both the strides and the offset may be dynamic
/// (i.e. their value may not be known at compile time; this is expressed as a `?` character in the assembly syntax).
///
/// Refer to the documentation of [`MemRefTypeRef::strides_and_offset`](crate::MemRefTypeRef::strides_and_offset)
/// and to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#stridedlayoutattr) for more
/// information.
#[derive(Copy, Clone)]
pub struct StridedLayoutAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> StridedLayoutAttributeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`StridedLayoutAttributeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirStridedLayoutAttrGetTypeID()).unwrap() }
    }

    /// Returns the number of dimensions of this [`StridedLayoutAttributeRef`].
    pub fn dimension_count(&self) -> usize {
        unsafe { mlirStridedLayoutAttrGetNumStrides(self.handle).cast_unsigned() }
    }

    /// Returns the offset of this [`StridedLayoutAttributeRef`].
    pub fn offset(&self) -> usize {
        unsafe { mlirStridedLayoutAttrGetOffset(self.handle) as usize }
    }

    /// Returns the strides of this [`StridedLayoutAttributeRef`].
    pub fn strides(&self) -> impl Iterator<Item = usize> {
        (0..self.dimension_count())
            .map(|index| unsafe { mlirStridedLayoutAttrGetStride(self.handle, index.cast_signed()) as usize })
    }
}

mlir_subtype_trait_impls!(
    StridedLayoutAttributeRef<'c, 't> as Attribute,
    mlir_type = Attribute,
    mlir_subtype = StridedLayout,
);

impl<'t> Context<'t> {
    /// Creates a new [`StridedLayoutAttributeRef`] with the specified offset and strides.
    /// The resulting [`StridedLayoutAttributeRef`] is owned by this context.
    pub fn strided_layout_attribute<'c>(
        &'c self,
        offset: usize,
        strides: &[usize],
    ) -> StridedLayoutAttributeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        let strides = strides.iter().map(|&stride| stride as i64).collect::<Vec<_>>();
        unsafe {
            StridedLayoutAttributeRef::from_c_api(
                mlirStridedLayoutAttrGet(
                    *self.handle.borrow(),
                    offset as i64,
                    strides.len().cast_signed(),
                    strides.as_ptr(),
                ),
                self,
            )
            .unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_strided_layout_attribute_type_id() {
        let context = Context::new();
        let strided_layout_attribute_id = StridedLayoutAttributeRef::type_id();
        let strided_layout_attribute_1 = context.strided_layout_attribute(0, &[4, 1]);
        let strided_layout_attribute_2 = context.strided_layout_attribute(0, &[4, 1]);
        assert_eq!(strided_layout_attribute_1.type_id(), strided_layout_attribute_2.type_id());
        assert_eq!(strided_layout_attribute_id, strided_layout_attribute_1.type_id());
    }

    #[test]
    fn test_strided_layout_attribute() {
        let context = Context::new();

        // Test with offset and strides.
        let attribute = context.strided_layout_attribute(0, &[4, 1]);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.offset(), 0);
        assert_eq!(attribute.dimension_count(), 2);
        assert_eq!(attribute.strides().collect::<Vec<_>>(), vec![4, 1]);

        // Test with non-zero offset.
        let attribute = context.strided_layout_attribute(10, &[8, 2, 1]);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.offset(), 10);
        assert_eq!(attribute.dimension_count(), 3);
        assert_eq!(attribute.strides().collect::<Vec<_>>(), vec![8, 2, 1]);
    }

    #[test]
    fn test_strided_layout_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.strided_layout_attribute(0, &[4, 1]);
        let attribute_2 = context.strided_layout_attribute(0, &[4, 1]);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.strided_layout_attribute(0, &[8, 1]);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.strided_layout_attribute(0, &[4, 1]);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_strided_layout_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.strided_layout_attribute(0, &[4, 1]);
        test_attribute_display_and_debug(attribute, "strided<[4, 1]>");
    }

    #[test]
    fn test_strided_layout_attribute_parsing() {
        let context = Context::new();
        let attribute = context.strided_layout_attribute(0, &[4, 1]);
        let parsed = context.parse_attribute("strided<[4, 1]>").unwrap();
        assert_eq!(parsed, attribute);
    }

    #[test]
    fn test_strided_layout_attribute_casting() {
        let context = Context::new();
        let attribute = context.strided_layout_attribute(0, &[4, 1]);
        test_attribute_casting(attribute);
    }
}
