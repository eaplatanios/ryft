use crate::{
    DetachedOp, DialectHandle, Location, Operation, OperationBuilder, Value, ValueRef, mlir_op, mlir_op_trait,
};

/// StableHLO [`Operation`] that constructs a complex tensor from two real tensors representing the real and imaginary
/// parts of the complex tensor. The shapes of the two input tensors must match. They also match the shape of the
/// output tensor, except for the fact that it has a different data type.
///
/// Note that this is an elementwise operation over the input tensors.
///
/// # Example
///
/// The following is an example of a [`ComplexOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %real: [1.0, 3.0]
/// // %imag: [2.0, 4.0]
/// %result = stablehlo.complex %real, %imag : tensor<2xcomplex<f32>>
/// // %result: [(1.0, 2.0), (3.0, 4.0)]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#complex) for more information.
pub trait ComplexOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the real part of the complex tensor that this [`ComplexOperation`] constructs.
    fn real(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the imaginary part of the complex tensor that this [`ComplexOperation`] constructs.
    fn imag(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }
}

mlir_op!(Complex);
mlir_op_trait!(Complex, OneResult);
mlir_op_trait!(Complex, ZeroRegions);
mlir_op_trait!(Complex, ZeroSuccessors);

/// Constructs a new detached/owned [`ComplexOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, the function may panic!
pub fn complex<'r, 'i, 'c: 'r + 'i, 't: 'c, R: Value<'r, 'c, 't>, I: Value<'i, 'c, 't>, L: Location<'c, 't>>(
    real: R,
    imag: I,
    location: L,
) -> DetachedComplexOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.complex", location)
        .add_operand(real)
        .add_operand(imag)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::complex`")
}

/// StableHLO [`Operation`] that extracts the imaginary part from a complex tensor. The shape of the output tensor
/// matches the shape of the input tensor, except for the fact that the output tensor has a different data type.
///
/// Note that this is an elementwise operation over the input tensor.
///
/// # Example
///
/// The following is an example of an [`ImagOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [(1.0, 2.0), (3.0, 4.0)]
/// %result = stablehlo.imag %operand : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
/// // %result: [2.0, 4.0]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#imag) for more information.
pub trait ImagOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Imag);
mlir_op_trait!(Imag, OneOperand);
mlir_op_trait!(Imag, OneResult);
mlir_op_trait!(Imag, ZeroRegions);
mlir_op_trait!(Imag, ZeroSuccessors);

/// Constructs a new detached/owned [`ImagOperation`] at the specified [`Location`].
///
/// Note that if the input to this function is invalid, the function may panic!
pub fn imag<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> DetachedImagOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.imag", location)
        .add_operand(input)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::imag`")
}

/// StableHLO [`Operation`] that extracts the real part from a complex tensor. The shape of the output tensor
/// matches the shape of the input tensor, except for the fact that the output tensor has a different data type.
///
/// Note that this is an elementwise operation over the input tensor.
///
/// # Example
///
/// The following is an example of a [`RealOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [(1.0, 2.0), (3.0, 4.0)]
/// %result = stablehlo.real %operand : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
/// // %result: [1.0, 3.0]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#real) for more information.
pub trait RealOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Real);
mlir_op_trait!(Real, OneOperand);
mlir_op_trait!(Real, OneResult);
mlir_op_trait!(Real, ZeroRegions);
mlir_op_trait!(Real, ZeroSuccessors);

/// Constructs a new detached/owned [`RealOperation`] at the specified [`Location`].
///
/// Note that if the input to this function is invalid, the function may panic!
pub fn real<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    location: L,
) -> DetachedRealOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.real", location)
        .add_operand(input)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::real`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Block, Context, OneOperand, OneResult, Operation, Size, Type, Value};

    use super::{ComplexOperation, complex, imag, real};

    #[test]
    fn test_complex() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let complex_type = context.complex_type(f32_type);
        let real_tensor_type = context.tensor_type(f32_type, &[Size::Static(2)], None, location).unwrap();
        let complex_tensor_type = context.tensor_type(complex_type, &[Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(real_tensor_type, location), (real_tensor_type, location)]);
            let real_value = block.argument(0).unwrap();
            let imag_value = block.argument(1).unwrap();
            let complex_op = complex(real_value, imag_value, location);
            assert_eq!(complex_op.operands().count(), 2);
            assert_eq!(complex_op.real(), real_value);
            assert_eq!(complex_op.imag(), imag_value);
            assert_eq!(complex_op.results().count(), 1);
            assert_eq!(complex_op.result(0).unwrap().r#type(), complex_tensor_type);
            let complex_block = block.append_operation(complex_op);
            block.append_operation(func::r#return(&[complex_block.result(0).unwrap()], location));
            func::func(
                "complex_test",
                func::FuncAttributes {
                    arguments: vec![real_tensor_type.into(), real_tensor_type.into()],
                    results: vec![complex_tensor_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @complex_test(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xcomplex<f32>> {
                    %0 = stablehlo.complex %arg0, %arg1 : tensor<2xcomplex<f32>>
                    return %0 : tensor<2xcomplex<f32>>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_imag() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let complex_type = context.complex_type(f32_type);
        let input_tensor_type = context.tensor_type(complex_type, &[Size::Static(2)], None, location).unwrap();
        let output_tensor_type = context.tensor_type(f32_type, &[Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_tensor_type, location)]);
            let input_value = block.argument(0).unwrap();
            let imag_op = imag(input_value, location);
            assert_eq!(imag_op.input(), input_value);
            assert_eq!(imag_op.output().r#type(), output_tensor_type);
            assert_eq!(imag_op.operands().count(), 1);
            assert_eq!(imag_op.results().count(), 1);
            let imag_block = block.append_operation(imag_op);
            block.append_operation(func::r#return(&[imag_block.result(0).unwrap()], location));
            func::func(
                "imag_test",
                func::FuncAttributes {
                    arguments: vec![input_tensor_type.into()],
                    results: vec![output_tensor_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @imag_test(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xf32> {
                    %0 = stablehlo.imag %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
                    return %0 : tensor<2xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_real() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let complex_type = context.complex_type(f32_type);
        let input_tensor_type = context.tensor_type(complex_type, &[Size::Static(2)], None, location).unwrap();
        let output_tensor_type = context.tensor_type(f32_type, &[Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_tensor_type, location)]);
            let input_value = block.argument(0).unwrap();
            let real_op = real(input_value, location);
            assert_eq!(real_op.input(), input_value);
            assert_eq!(real_op.output().r#type(), output_tensor_type.as_ref());
            assert_eq!(real_op.operands().count(), 1);
            assert_eq!(real_op.results().count(), 1);
            let real_block = block.append_operation(real_op);
            block.append_operation(func::r#return(&[real_block.result(0).unwrap()], location));
            func::func(
                "real_test",
                func::FuncAttributes {
                    arguments: vec![input_tensor_type.into()],
                    results: vec![output_tensor_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @real_test(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xf32> {
                    %0 = stablehlo.real %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
                    return %0 : tensor<2xf32>
                  }
                }
            "},
        );
    }
}
