use ryft_xla_sys::bindings::{
    MlirAttribute, mlirShapedTypeGetDynamicSize, stablehloAttributeIsTypeExtensions, stablehloTypeExtensionsGet,
    stablehloTypeExtensionsGetBoundsElem, stablehloTypeExtensionsGetBoundsSize,
};

use crate::{Attribute, Context, DialectHandle, mlir_subtype_trait_impls};

/// StableHLO [`Attribute`] that is used to extend the built-in MLIR [`TensorTypeRef`](crate::TensorTypeRef) with
/// StableHLO tensor-specific properties. These properties are not modeled in the built-in MLIR type. This is included
/// in [`TensorTypeRef`](crate::TensorTypeRef) for StableHLO types via its
/// [`TensorTypeRef::encoding`](crate::TensorTypeRef::encoding) attribute.
#[derive(Copy, Clone)]
pub struct TensorTypeExtensionsAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> TensorTypeExtensionsAttributeRef<'c, 't> {
    /// Returns the bounds for the dimensions of the associated [`TensorTypeRef`](crate::TensorTypeRef). The returned
    /// vector a length equal to the number of dimensions of the associated [`TensorTypeRef`](crate::TensorTypeRef)
    /// (i.e., equal to its _rank_). For each dimension, it contains either a bound on its size if it is a dimension
    /// with a [`Size::Dynamic`](crate::Size::Dynamic) size, or [`None`] if it has either a
    /// [`Size::Static`](crate::Size::Static) size, or a [`Size::Dynamic`](crate::Size::Dynamic)
    /// size and no bound specified for it.
    pub fn bounds(&self) -> Vec<Option<usize>> {
        unsafe {
            let count = stablehloTypeExtensionsGetBoundsSize(self.handle).cast_unsigned();
            let mut bounds = Vec::with_capacity(count);
            for i in 0..count {
                let bound = stablehloTypeExtensionsGetBoundsElem(self.handle, i.cast_signed());
                bounds.push(if bound == mlirShapedTypeGetDynamicSize() { None } else { Some(bound as usize) });
            }
            bounds
        }
    }
}

impl<'c, 't> Attribute<'c, 't> for TensorTypeExtensionsAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { stablehloAttributeIsTypeExtensions(handle) } {
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

mlir_subtype_trait_impls!(TensorTypeExtensionsAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'t> Context<'t> {
    /// Creates a new StableHLO [`TensorTypeExtensionsAttributeRef`] owned by this [`Context`].
    ///
    /// Refer to the documentation of [`TensorTypeExtensionsAttributeRef::bounds`] for information on the `bounds`
    /// argument of this function.
    pub fn stable_hlo_tensor_type_extensions<'c>(
        &'c self,
        bounds: &[Option<usize>],
    ) -> TensorTypeExtensionsAttributeRef<'c, 't> {
        // Make sure that the StableHLO dialect is loaded into the current context to prevent segmentation faults.
        self.load_dialect(DialectHandle::stable_hlo());
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            let bounds = bounds
                .iter()
                .map(|bound| match bound {
                    None => mlirShapedTypeGetDynamicSize(),
                    Some(bound) => *bound as i64,
                })
                .collect::<Vec<_>>();
            TensorTypeExtensionsAttributeRef::from_c_api(
                stablehloTypeExtensionsGet(*self.handle.borrow(), bounds.len().cast_signed(), bounds.as_ptr()),
                self,
            )
            .unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_tensor_type_extensions_attribute() {
        let context = Context::new();
        let bounds = vec![Some(10), None, Some(20), None];
        let attribute = context.stable_hlo_tensor_type_extensions(&bounds);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.bounds(), bounds);
    }

    #[test]
    fn test_tensor_type_extensions_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_tensor_type_extensions(&[Some(10), None, Some(20), None]);
        let attribute_2 = context.stable_hlo_tensor_type_extensions(&[Some(10), None, Some(20), None]);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.stable_hlo_tensor_type_extensions(&[None, None, Some(20)]);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.stable_hlo_tensor_type_extensions(&[Some(10), None, Some(20), None]);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_tensor_type_extensions_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.stable_hlo_tensor_type_extensions(&[Some(10), None, Some(20), None]);
        test_attribute_display_and_debug(attribute, "#stablehlo.bounds<10, ?, 20, ?>");
    }

    #[test]
    fn test_tensor_type_extensions_attribute_casting() {
        let context = Context::new();
        let attribute = context.stable_hlo_tensor_type_extensions(&[Some(10), None, Some(20), None]);
        test_attribute_casting(attribute);
    }
}
