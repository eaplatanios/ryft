use ryft_xla_sys::bindings::{
    MlirType, mlirTupleTypeGet, mlirTupleTypeGetNumTypes, mlirTupleTypeGetType, mlirTupleTypeGetTypeID,
};

use crate::{Context, Type, TypeId, TypeRef, mlir_subtype_trait_impls};

/// Built-in MLIR [`Type`] that represents a fixed-size collection of other [`Type`]. Note that, while [`TupleTypeRef`]s
/// are first-class in the MLIR type system, MLIR provides no standard operations for operating on [`TupleTypeRef`]s.
/// The rationale for this decision is described [here](https://mlir.llvm.org/docs/Rationale/Rationale/#tuple-types).
///
/// # Examples
///
/// The following are examples of [`TupleTypeRef`]s represented using their [`Display`](std::fmt::Display) rendering:
///
/// ```text
/// tuple<>
/// tuple<f32>
/// tuple<i32, f32, tensor<i1>, i5>
/// ```
///
/// Refer to the [MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#tupletype) for more information.
#[derive(Copy, Clone)]
pub struct TupleTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> TupleTypeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`TupleTypeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirTupleTypeGetTypeID()).unwrap() }
    }

    /// Returns the length of this [`TupleTypeRef`] (i.e., the number of element [`Type`]s it contains).
    pub fn len(&self) -> usize {
        unsafe { mlirTupleTypeGetNumTypes(self.handle).cast_unsigned() }
    }

    /// Returns all element [`Type`]s of this [`TupleTypeRef`].
    pub fn elements(&self) -> impl Iterator<Item = TypeRef<'c, 't>> {
        (0..self.len()).map(|index| self.element(index))
    }

    /// Returns the element [`Type`] of this [`TupleTypeRef`] at the specified index.
    ///
    /// Note that this function will panic if the provided index is out of bounds.
    pub fn element(&self, index: usize) -> TypeRef<'c, 't> {
        if index >= self.len() {
            panic!("index is out of bounds");
        }
        unsafe { TypeRef::from_c_api(mlirTupleTypeGetType(self.handle, index.cast_signed()), self.context).unwrap() }
    }
}

mlir_subtype_trait_impls!(TupleTypeRef<'c, 't> as Type, mlir_type = Type, mlir_subtype = Tuple);

impl<'t> Context<'t> {
    /// Creates a new [`TupleTypeRef`] with the provided element types, which is owned by this [`Context`].
    pub fn tuple_type<'c, T: Type<'c, 't>>(&'c self, elements: &[T]) -> TupleTypeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            let elements = elements.iter().map(|element| element.to_c_api()).collect::<Vec<_>>();
            TupleTypeRef::from_c_api(
                mlirTupleTypeGet(*self.handle.borrow(), elements.len().cast_signed(), elements.as_ptr() as *const _),
                &self,
            )
            .unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::types::tests::{test_type_casting, test_type_display_and_debug};

    use super::*;

    #[test]
    fn test_tuple_type_type_id() {
        let context = Context::new();
        let tuple_type = TupleTypeRef::type_id();
        let tuple_type_1 = context.tuple_type(&[context.index_type()]);
        let tuple_type_2 = context.tuple_type(&[context.float32_type()]);
        assert_eq!(tuple_type_1.type_id(), tuple_type_2.type_id());
        assert_eq!(tuple_type, tuple_type_1.type_id());
    }

    #[test]
    fn test_tuple_type() {
        let context = Context::new();

        let r#type = context.tuple_type::<TypeRef>(&[]);
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.len(), 0);

        let element = context.float32_type();
        let r#type = context.tuple_type(&[element]);
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.len(), 1);
        assert_eq!(r#type.elements().collect::<Vec<_>>(), vec![element]);

        let element_1 = context.signless_integer_type(32).as_ref();
        let element_2 = context.float32_type().as_ref();
        let element_3 = context.index_type().as_ref();
        let r#type = context.tuple_type(&[element_1, element_2, element_3]);
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.len(), 3);
        assert_eq!(r#type.elements().collect::<Vec<_>>(), vec![element_1, element_2, element_3]);
    }

    #[test]
    fn test_tuple_type_equality() {
        let context = Context::new();
        let element_1 = context.index_type().as_ref();
        let element_2 = context.float32_type().as_ref();

        // Same types from the same context must be equal because they are "uniqued".
        let type_1 = context.tuple_type(&[element_1, element_2]);
        let type_2 = context.tuple_type(&[element_1, element_2]);
        assert_eq!(type_1, type_2);

        // Different tuples from the same context must not be equal.
        let type_2 = context.tuple_type(&[element_1]);
        assert_ne!(type_1, type_2);

        // Same types from different contexts must not be equal.
        let context = Context::new();
        let element_1 = context.index_type().as_ref();
        let element_2 = context.float32_type().as_ref();
        let type_2 = context.tuple_type(&[element_1, element_2]);
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_tuple_type_display_and_debug() {
        let context = Context::new();
        let r#type = context.tuple_type::<TypeRef>(&[]);
        test_type_display_and_debug(r#type, "tuple<>");

        let element = context.float32_type();
        let r#type = context.tuple_type(&[element]);
        test_type_display_and_debug(r#type, "tuple<f32>");

        let element_1 = context.signless_integer_type(32).as_ref();
        let element_2 = context.float32_type().as_ref();
        let element_3 = context.index_type().as_ref();
        let r#type = context.tuple_type(&[element_1, element_2, element_3]);
        test_type_display_and_debug(r#type, "tuple<i32, f32, index>");
    }

    #[test]
    fn test_tuple_type_parsing() {
        let context = Context::new();
        let element_1 = context.signless_integer_type(32).as_ref();
        let element_2 = context.float32_type().as_ref();
        let element_3 = context.index_type().as_ref();
        assert_eq!(context.parse_type("tuple<>").unwrap(), context.tuple_type::<TypeRef>(&[]));
        assert_eq!(context.parse_type("tuple<f32>").unwrap(), context.tuple_type(&[element_2]));
        assert_eq!(
            context.parse_type("tuple<i32, f32, index>").unwrap(),
            context.tuple_type(&[element_1, element_2, element_3])
        );
    }

    #[test]
    fn test_tuple_type_casting() {
        let context = Context::new();
        let element_1 = context.index_type().as_ref();
        let element_2 = context.float32_type().as_ref();
        let r#type = context.tuple_type(&[element_1, element_2]);
        test_type_casting(r#type);
    }
}
