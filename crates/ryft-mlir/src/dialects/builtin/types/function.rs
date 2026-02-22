use ryft_xla_sys::bindings::{
    MlirType, mlirFunctionTypeGet, mlirFunctionTypeGetInput, mlirFunctionTypeGetNumInputs,
    mlirFunctionTypeGetNumResults, mlirFunctionTypeGetResult, mlirFunctionTypeGetTypeID,
};

use crate::{Context, Type, TypeId, TypeRef, mlir_subtype_trait_impls};

/// Built-in MLIR [`Type`] that represents the type of functions where a function is defined as a mapping from a list
/// of inputs to a list of outputs. A [`FunctionTypeRef`] can be thought of as a function signature. It consists of a
/// list of formal input/parameter [`Type`]s and a list of formal output/result [`Type`]s.
///
/// Refer to the [MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#functiontype)
/// for more information.
#[derive(Copy, Clone)]
pub struct FunctionTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> FunctionTypeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`FunctionTypeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirFunctionTypeGetTypeID()).unwrap() }
    }

    /// Returns the number of inputs (i.e., parameters) of this [`FunctionTypeRef`].
    pub fn input_count(&self) -> usize {
        unsafe { mlirFunctionTypeGetNumInputs(self.handle).cast_unsigned() }
    }

    /// Returns the number of outputs (i.e., results) of this [`FunctionTypeRef`].
    pub fn output_count(&self) -> usize {
        unsafe { mlirFunctionTypeGetNumResults(self.handle).cast_unsigned() }
    }

    /// Returns the input (i.e., parameter) [`Type`]s of this [`FunctionTypeRef`].
    pub fn inputs(&self) -> impl Iterator<Item = TypeRef<'c, 't>> {
        (0..self.input_count()).map(|index| self.input(index))
    }

    /// Returns the output (i.e., result) [`Type`]s of this [`FunctionTypeRef`].
    pub fn outputs(&self) -> impl Iterator<Item = TypeRef<'c, 't>> {
        (0..self.output_count()).map(|index| self.output(index))
    }

    /// Returns the `index`-th input (i.e., parameter) [`Type`] of this [`FunctionTypeRef`].
    ///
    /// Note that this function will panic if the provided index is out of bounds.
    pub fn input(&self, index: usize) -> TypeRef<'c, 't> {
        if index >= self.input_count() {
            panic!("function type input index is out of bounds");
        }
        unsafe {
            let element = mlirFunctionTypeGetInput(self.handle, index.cast_signed());
            TypeRef::from_c_api(element, self.context).unwrap()
        }
    }

    /// Returns the `index`-th output (i.e., result) [`Type`] of this [`FunctionTypeRef`].
    ///
    /// Note that this function will panic if the provided index is out of bounds.
    pub fn output(&self, index: usize) -> TypeRef<'c, 't> {
        if index >= self.output_count() {
            panic!("function type output index is out of bounds");
        }
        unsafe {
            let element = mlirFunctionTypeGetResult(self.handle, index.cast_signed());
            TypeRef::from_c_api(element, self.context).unwrap()
        }
    }
}

mlir_subtype_trait_impls!(FunctionTypeRef<'c, 't> as Type, mlir_type = Type, mlir_subtype = Function);

impl<'t> Context<'t> {
    /// Creates a new [`FunctionTypeRef`] with the provided input and output [`Type`]s,
    /// which is owned by this [`Context`].
    pub fn function_type<'c, I: Type<'c, 't>, O: Type<'c, 't>>(
        &'c self,
        inputs: &[I],
        outputs: &[O],
    ) -> FunctionTypeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            let inputs = inputs.iter().map(|input| input.to_c_api()).collect::<Vec<_>>();
            let outputs = outputs.iter().map(|output| output.to_c_api()).collect::<Vec<_>>();
            FunctionTypeRef::from_c_api(
                mlirFunctionTypeGet(
                    *self.handle.borrow(),
                    inputs.len().cast_signed(),
                    inputs.as_ptr() as *const _,
                    outputs.len().cast_signed(),
                    outputs.as_ptr() as *const _,
                ),
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
    fn test_function_type_type_id() {
        let context = Context::new();
        let function_type = FunctionTypeRef::type_id();
        let function_type_1 = context.function_type::<TypeRef, TypeRef>(&[], &[]);
        let function_type_2 = context.function_type::<TypeRef, TypeRef>(&[], &[]);
        assert_eq!(function_type_1.type_id(), function_type_2.type_id());
        assert_eq!(function_type, function_type_1.type_id());
    }

    #[test]
    fn test_function_type() {
        let context = Context::new();
        let input_1 = context.index_type().as_ref();
        let input_2 = context.float32_type().as_ref();
        let output_1 = context.signless_integer_type(64).as_ref();
        let output_2 = context.float32_type().as_ref();

        // Multiple inputs and single output.
        let r#type = context.function_type(&[input_1, input_2], &[output_1]);
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.input_count(), 2);
        assert_eq!(r#type.output_count(), 1);
        assert_eq!(r#type.inputs().collect::<Vec<_>>(), vec![input_1, input_2]);
        assert_eq!(r#type.outputs().collect::<Vec<_>>(), vec![output_1]);
        assert_eq!(r#type.input(0), input_1);
        assert_eq!(r#type.input(1), input_2);
        assert_eq!(r#type.output(0), output_1);

        // No inputs and single output.
        let r#type = context.function_type::<TypeRef, _>(&[], &[output_1]);
        assert_eq!(r#type.input_count(), 0);
        assert_eq!(r#type.output_count(), 1);

        // Single input and multiple outputs.
        let r#type = context.function_type::<_, TypeRef>(&[input_1], &[]);
        assert_eq!(r#type.input_count(), 1);
        assert_eq!(r#type.output_count(), 0);

        // Multiple inputs and multiple outputs.
        let r#type = context.function_type(&[input_1], &[output_1, output_2]);
        assert_eq!(r#type.input_count(), 1);
        assert_eq!(r#type.output_count(), 2);
        assert_eq!(r#type.inputs().collect::<Vec<_>>(), vec![input_1]);
        assert_eq!(r#type.outputs().collect::<Vec<_>>(), vec![output_1, output_2]);
        assert_eq!(r#type.input(0), input_1);
        assert_eq!(r#type.output(0), output_1);
        assert_eq!(r#type.output(1), output_2);
    }

    #[test]
    fn test_function_type_equality() {
        let context = Context::new();
        let input = context.index_type();
        let output = context.float32_type();

        // Same types from the same context must be equal because they are "uniqued".
        let type_1 = context.function_type(&[input], &[output]);
        let type_2 = context.function_type(&[input], &[output]);
        assert_eq!(type_1, type_2);

        // Different signatures from the same context must not be equal.
        let type_2 = context.function_type(&[input, input], &[output]);
        assert_ne!(type_1, type_2);

        // Same types from different contexts must not be equal.
        let context = Context::new();
        let input = context.index_type();
        let output = context.float32_type();
        let type_2 = context.function_type(&[input], &[output]);
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_function_type_display_and_debug() {
        let context = Context::new();
        let input_1 = context.index_type().as_ref();
        let input_2 = context.float32_type().as_ref();
        let output_1 = context.signless_integer_type(64).as_ref();
        let output_2 = context.signless_integer_type(1).as_ref();
        let r#type = context.function_type(&[input_1, input_2], &[output_1, output_2]);
        test_type_display_and_debug(r#type, "(index, f32) -> (i64, i1)");
    }

    #[test]
    fn test_function_type_parsing() {
        let context = Context::new();
        let input_1 = context.index_type().as_ref();
        let input_2 = context.float32_type().as_ref();
        let output_1 = context.signless_integer_type(64).as_ref();
        let output_2 = context.signless_integer_type(1).as_ref();
        assert_eq!(
            context.parse_type("(index, f32) -> (i64, i1)").unwrap(),
            context.function_type(&[input_1, input_2], &[output_1, output_2])
        );
    }

    #[test]
    fn test_function_type_casting() {
        let context = Context::new();
        let input_1 = context.index_type().as_ref();
        let input_2 = context.float32_type().as_ref();
        let output_1 = context.signless_integer_type(64).as_ref();
        let output_2 = context.float32_type().as_ref();
        let r#type = context.function_type(&[input_1, input_2], &[output_1, output_2]);
        test_type_casting(r#type);
    }
}
