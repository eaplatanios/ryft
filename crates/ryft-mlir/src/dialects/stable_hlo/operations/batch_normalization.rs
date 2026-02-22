use crate::{
    Attribute, DetachedOp, DialectHandle, FloatAttributeRef, IntegerAttributeRef, Location, Operation,
    OperationBuilder, Value, mlir_op, mlir_op_trait,
};

/// Name of the [`Attribute`] that is used to store [`BatchNormOperation::feature_index`].
pub const BATCH_NORM_FEATURE_INDEX_ATTRIBUTE: &str = "feature_index";

/// Name of the [`Attribute`] that is used to store [`BatchNormOperation::epsilon`].
pub const BATCH_NORM_EPSILON_ATTRIBUTE: &str = "epsilon";

/// Trait that is shared among all StableHLO [`Operation`]s that are related to
/// [batch normalization](https://en.wikipedia.org/wiki/Batch_normalization).
pub trait BatchNormOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the feature index dimension used by this [`BatchNormOperation`].
    fn feature_index(&self) -> u32 {
        self.attribute(BATCH_NORM_FEATURE_INDEX_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value() as u32)
            .unwrap_or_else(|| {
                panic!("invalid '{}' attribute in `stable_hlo::{}`", BATCH_NORM_FEATURE_INDEX_ATTRIBUTE, self.name())
            })
    }

    /// Returns the ε parameter used by this [`BatchNormOperation`] for numerical stability.
    fn epsilon(&self) -> f32 {
        self.attribute(BATCH_NORM_EPSILON_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<FloatAttributeRef>())
            .map(|attribute| attribute.value() as f32)
            .unwrap_or_else(|| {
                panic!("invalid '{}' attribute in `stable_hlo::{}`", BATCH_NORM_EPSILON_ATTRIBUTE, self.name())
            })
    }
}

/// StableHLO [`Operation`] that normalizes a tensor across all dimensions except for the
/// [`BatchNormOperation::feature_index`] dimension.
///
/// More formally, this operation can be expressed as a decomposition into existing StableHLO operations, as follows
/// (using Python syntax for simplicity):
///
/// ```python
/// def batch_norm_inference(
///     input: Tensor,
///     scale: Tensor,
///     offset: Tensor,
///     mean: Tensor,
///     variance: Tensor,
///     epsilon: float,
///     feature_index: int,
/// ) -> Tensor:
///     # Broadcast inputs to the shape of the input tensor.
///     scale = broadcast_in_dim(scale, [feature_index], type(input))
///     offset = broadcast_in_dim(offset, [feature_index], type(input))
///     mean = broadcast_in_dim(mean, [feature_index], type(input))
///     variance = broadcast_in_dim(variance, [feature_index], type(input))
///     epsilon = broadcast_in_dim(constant(epsilon, element_type(input)), [], type(input))
///
///     # Perform normalization using the provided `mean` and `variance`.
///     normalized = divide(subtract(input, mean), sqrt(add(variance, epsilon)))
///     return add(multiply(scale, normalized), offset)
/// ```
///
/// For quantized types, this operation first dequantizes the input, applies the batch normalization inference operation
/// as described above, and then quantizes the result.
///
/// # Example
///
/// The following is an example of a [`BatchNormInferenceOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [
/// //            [[1.0, 2.0], [3.0, 4.0]],
/// //            [[3.0, 4.0], [1.0, 2.0]]
/// //           ]
/// // %scale: [1.0, 1.0]
/// // %offset: [1.0, 1.0]
/// // %mean: [2.0, 3.0]
/// // %variance: [1.0, 1.0]
/// %result = "stablehlo.batch_norm_inference"(%operand, %scale, %offset, %mean, %variance) <{
///   epsilon = 0.0 : f32,
///   feature_index = 2 : i64
/// }> : (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) -> tensor<2x2x2xf64>
/// // %result: [
/// //           [[0.0, 0.0], [2.0, 2.0]],
/// //           [[2.0, 2.0], [0.0, 0.0]]
/// //          ]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#batch_norm_inference)
/// for more information.
pub trait BatchNormInferenceOperation<'o, 'c: 'o, 't: 'c>: BatchNormOperation<'o, 'c, 't> {}

mlir_op!(BatchNormInference);
mlir_op_trait!(BatchNormInference, OneResult);
mlir_op_trait!(BatchNormInference, ZeroRegions);
mlir_op_trait!(BatchNormInference, ZeroSuccessors);
mlir_op_trait!(BatchNormInference, @local BatchNormOperation);

/// Constructs a new detached/owned [`BatchNormInferenceOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`BatchNormInferenceOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
#[allow(clippy::too_many_arguments)]
pub fn batch_norm_inference<
    'i,
    's,
    'o,
    'm,
    'v,
    'c: 'i + 's + 'o + 'm + 'v,
    't: 'c,
    I: Value<'i, 'c, 't>,
    S: Value<'s, 'c, 't>,
    O: Value<'o, 'c, 't>,
    M: Value<'m, 'c, 't>,
    V: Value<'v, 'c, 't>,
    L: Location<'c, 't>,
>(
    input: I,
    scale: S,
    offset: O,
    mean: M,
    variance: V,
    epsilon: f32,
    feature_index: u32,
    location: L,
) -> DetachedBatchNormInferenceOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.batch_norm_inference", location)
        .add_operand(input)
        .add_operand(scale)
        .add_operand(offset)
        .add_operand(mean)
        .add_operand(variance)
        .add_attribute(BATCH_NORM_EPSILON_ATTRIBUTE, context.float_attribute(context.float32_type(), epsilon as f64))
        .add_attribute(
            BATCH_NORM_FEATURE_INDEX_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), feature_index as i64),
        )
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::batch_norm_inference`")
}

/// StableHLO [`Operation`] that computes the mean and variance of a tensor across all dimensions except for the
/// [`BatchNormOperation::feature_index`] dimension, and then normalizes that tensor using those statistics.
/// Its outputs consist of the normalized tensor as well as the computed mean and variance tensors.
///
/// More formally, this operation can be expressed as a decomposition into existing StableHLO operations, as follows
/// (using Python syntax for simplicity):
///
/// ```python
/// def compute_mean(input: Tensor, feature_index: int) -> Tensor:
///     sum = reduce(
///         inputs=[input],
///         initial_values=[constant(0, element_type(input))],
///         dimensions=[i for i in range(rank(input)) if i != feature_index],
///         body=lambda x, y: add(x, y))
///     denominator = constant(size(input) / dim(input, feature_index), element_type(input))
///     return divide(sum, broadcast_in_dim(denominator, [], type(sum)))
///
/// def compute_variance(input: Tensor, feature_index: int) -> Tensor:
///     mean = compute_mean(input, feature_index)
///     mean = broadcast_in_dim(mean, [feature_index], type(input))
///     centered_input = subtract(input, mean)
///     return compute_mean(mul(centered_input, centered_input), feature_index)
///
/// def batch_norm_training(
///     input: Tensor,
///     scale: Tensor,
///     offset: Tensor,
///     epsilon: float,
///     feature_index: int,
/// ) -> tuple[Tensor, Tensor, Tensor]:
///     mean = compute_mean(input, feature_index)
///     variance = compute_variance(input, feature_index)
///     return batch_norm_inference(input, scale, offset, mean, variance, epsilon, feature_index), mean, variance
/// ```
///
/// For quantized types, this operation first dequantizes the input, applies the batch normalization inference operation
/// as described above, and then quantizes the result.
///
/// # Example
///
/// The following is an example of a [`BatchNormTrainingOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [
/// //            [[1.0, 2.0], [3.0, 4.0]],
/// //            [[3.0, 4.0], [1.0, 2.0]]
/// //           ]
/// // %scale: [1.0, 1.0]
/// // %offset: [1.0, 1.0]
/// %output, %batch_mean, %batch_variance = "stablehlo.batch_norm_training"(%operand, %scale, %offset) <{
///   epsilon = 0.0 : f32,
///   feature_index = 2 : i64
/// }> : (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>) -> (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>)
/// // %output: [
/// //           [[0.0, 0.0], [2.0, 2.0]],
/// //           [[2.0, 2.0], [0.0, 0.0]]
/// //          ]
/// // %batch_mean: [2.0, 3.0]
/// // %batch_variance: [1.0, 1.0]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#batch_norm_training)
/// for more information.
pub trait BatchNormTrainingOperation<'o, 'c: 'o, 't: 'c>: BatchNormOperation<'o, 'c, 't> {}

mlir_op!(BatchNormTraining);
mlir_op_trait!(BatchNormTraining, ZeroRegions);
mlir_op_trait!(BatchNormTraining, ZeroSuccessors);
mlir_op_trait!(BatchNormTraining, @local BatchNormOperation);

/// Constructs a new detached/owned [`BatchNormTrainingOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`BatchNormTrainingOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn batch_norm_training<
    'i,
    's,
    'o,
    'c: 'i + 's + 'o,
    't: 'c,
    I: Value<'i, 'c, 't>,
    S: Value<'s, 'c, 't>,
    O: Value<'o, 'c, 't>,
    L: Location<'c, 't>,
>(
    input: I,
    scale: S,
    offset: O,
    epsilon: f32,
    feature_index: u32,
    location: L,
) -> DetachedBatchNormTrainingOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.batch_norm_training", location)
        .add_operand(input)
        .add_operand(scale)
        .add_operand(offset)
        .add_attribute(BATCH_NORM_EPSILON_ATTRIBUTE, context.float_attribute(context.float32_type(), epsilon as f64))
        .add_attribute(
            BATCH_NORM_FEATURE_INDEX_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), feature_index as i64),
        )
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::batch_norm_training`")
}

/// StableHLO [`Operation`] that computes gradients for [`BatchNormTrainingOperation`] for back-propagation.
/// It takes as input the gradient of the [`BatchNormTrainingOperation`] output/result and returns the gradients
/// of its inputs/operands.
///
/// More formally, this operation can be expressed as a decomposition into existing StableHLO operations, as follows
/// (using Python syntax for simplicity):
///
/// ```python
/// def compute_sum(input: Tensor, feature_index: int) -> Tensor:
///     return reduce(
///         inputs=[input],
///         initial_values=[constant(0, element_type(input))],
///         dimensions=[i for i in range(rank(input)) if i != feature_index],
///         body=lambda x, y: add(x, y))
///
/// def compute_mean(input: Tensor, feature_index: int) -> Tensor:
///     sum = compute_sum(input, feature_index)
///     denominator = constant(size(input) / dim(input, feature_index), element_type(input))
///     return divide(sum, broadcast_in_dim(denominator, [], type(sum)))
///
/// def batch_norm_grad(
///     input: Tensor,
///     scale: Tensor,
///     mean: Tensor,
///     variance: Tensor,
///     grad_output: Tensor,
///     epsilon: float,
///     feature_index: int,
/// ) -> tuple[Tensor, Tensor, Tensor]:
///     # Broadcast inputs to the shape of the input tensor.
///     scale = broadcast_in_dim(scale, [feature_index], type(input))
///     mean = broadcast_in_dim(mean, [feature_index], type(input))
///     variance = broadcast_in_dim(variance, [feature_index], type(input))
///     epsilon = broadcast_in_dim(constant(epsilon, element_type(input)), [], type(input))
///
///     # Perform normalization using the provided `mean` and `variance`.
///     # The intermediate values will be useful for computing gradients later on.
///     centered_input = subtract(input, mean)
///     standard_deviation = sqrt(add(variance, epsilon))
///     normalized_input = divide(centered_input, standard_deviation)
///
///     # Compute the gradients.
///     elements_per_feature = broadcast_in_dim(
///         constant(divide(size(input), dim(input, feature_index)), element_type(grad_output)),
///         [],
///         type(input))
///     i1 = multiply(grad_output, elements_per_feature)
///     i2 = broadcast_in_dim(compute_sum(grad_output, feature_index), [feature_index], type(input))
///     i3 = broadcast_in_dim(
///         compute_sum(multiply(grad_output, centered_input), feature_index),
///         [feature_index],
///         type(input))
///     i4 = multiply(i3, centered_input)
///     i5 = divide(i4, add(variance, epsilon))
///     i6 = subtract(subtract(i1, i2), i5)
///
///     grad_input = multiply(divide(divide(scale, standard_deviation), elements_per_feature), i6)
///     grad_scale = compute_sum(multiply(grad_output, normalized_input), feature_index)
///     grad_offset = compute_sum(grad_output, feature_index)
///
///     return grad_input, grad_scale, grad_offset
/// ```
///
/// For quantized types, this operation first dequantizes the input, applies the batch normalization inference operation
/// as described above, and then quantizes the result.
///
/// # Example
///
/// The following is an example of a [`BatchNormGradOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: [
/// //            [[1.0, 2.0], [3.0, 4.0]],
/// //            [[3.0, 4.0], [1.0, 2.0]]
/// //           ]
/// // %scale: [1.0, 1.0]
/// // %mean: [2.0, 3.0]
/// // %variance: [1.0, 1.0]
/// // %grad_output: [
/// //                [[0.1, 0.1], [0.1, 0.1]],
/// //                [[0.1, 0.1], [0.1, 0.1]]
/// //               ]
/// %grad_operand, %grad_scale, %grad_offset = "stablehlo.batch_norm_grad"(
///   %operand,
///   %scale,
///   %mean,
///   %variance,
///   %grad_output
/// ) <{
///   epsilon = 0.0 : f32,
///   feature_index = 2 : i64
/// }> : (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2x2x2xf64>)
///   -> (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>)
/// // %grad_operand: [
/// //                 [[0.0, 0.0], [0.0, 0.0]],
/// //                 [[0.0, 0.0], [0.0, 0.0]]
/// //                ]
/// // %grad_scale: [0.0, 0.0]
/// // %grad_offset: [0.4, 0.4]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#batch_norm_grad)
/// for more information.
pub trait BatchNormGradOperation<'o, 'c: 'o, 't: 'c>: BatchNormOperation<'o, 'c, 't> {}

mlir_op!(BatchNormGrad);
mlir_op_trait!(BatchNormGrad, ZeroRegions);
mlir_op_trait!(BatchNormGrad, ZeroSuccessors);
mlir_op_trait!(BatchNormGrad, @local BatchNormOperation);

/// Constructs a new detached/owned [`BatchNormGradOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`BatchNormGradOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
#[allow(clippy::too_many_arguments)]
pub fn batch_norm_grad<
    'i,
    's,
    'm,
    'v,
    'g,
    'c: 'i + 's + 'm + 'v + 'g,
    't: 'c,
    I: Value<'i, 'c, 't>,
    S: Value<'s, 'c, 't>,
    M: Value<'m, 'c, 't>,
    V: Value<'v, 'c, 't>,
    G: Value<'g, 'c, 't>,
    L: Location<'c, 't>,
>(
    input: I,
    scale: S,
    mean: M,
    variance: V,
    grad_output: G,
    epsilon: f32,
    feature_index: u32,
    location: L,
) -> DetachedBatchNormGradOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.batch_norm_grad", location)
        .add_operand(input)
        .add_operand(scale)
        .add_operand(mean)
        .add_operand(variance)
        .add_operand(grad_output)
        .add_attribute(BATCH_NORM_EPSILON_ATTRIBUTE, context.float_attribute(context.float32_type(), epsilon as f64))
        .add_attribute(
            BATCH_NORM_FEATURE_INDEX_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), feature_index as i64),
        )
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::batch_norm_grad`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Block, Context, Operation, Size};

    use super::*;

    #[test]
    fn test_batch_norm_inference() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let tensor_type = context
            .tensor_type(f32_type, &[Size::Static(2), Size::Static(2), Size::Static(2)], None, location)
            .unwrap();
        let param_type = context.tensor_type(f32_type, &[Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (tensor_type, location),
                (param_type, location),
                (param_type, location),
                (param_type, location),
                (param_type, location),
            ]);
            let operand = block.argument(0).unwrap();
            let scale = block.argument(1).unwrap();
            let offset = block.argument(2).unwrap();
            let mean = block.argument(3).unwrap();
            let variance = block.argument(4).unwrap();
            let batch_norm_op = batch_norm_inference(operand, scale, offset, mean, variance, 0.001, 2, location);
            assert_eq!(batch_norm_op.operands().count(), 5);
            assert_eq!(batch_norm_op.results().count(), 1);
            assert_eq!(batch_norm_op.epsilon(), 0.001);
            assert_eq!(batch_norm_op.feature_index(), 2);
            let batch_norm_op = block.append_operation(batch_norm_op);
            block.append_operation(func::r#return(&[batch_norm_op.result(0).unwrap()], location));
            func::func(
                "batch_norm_inference_test",
                func::FuncAttributes {
                    arguments: vec![
                        tensor_type.into(),
                        param_type.into(),
                        param_type.into(),
                        param_type.into(),
                        param_type.into(),
                    ],
                    results: vec![tensor_type.into()],
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
                  func.func @batch_norm_inference_test(\
                    %arg0: tensor<2x2x2xf32>, \
                    %arg1: tensor<2xf32>, \
                    %arg2: tensor<2xf32>, \
                    %arg3: tensor<2xf32>, \
                    %arg4: tensor<2xf32>\
                  ) -> tensor<2x2x2xf32> {
                    %0 = \"stablehlo.batch_norm_inference\"(%arg0, %arg1, %arg2, %arg3, %arg4) <{\
                      epsilon = 1.000000e-03 : f32, \
                      feature_index = 2 : i64\
                    }> : (tensor<2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) \
                      -> tensor<2x2x2xf32>
                    return %0 : tensor<2x2x2xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_batch_norm_training() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let tensor_type = context
            .tensor_type(f32_type, &[Size::Static(2), Size::Static(2), Size::Static(2)], None, location)
            .unwrap();
        let param_type = context.tensor_type(f32_type, &[Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location), (param_type, location), (param_type, location)]);
            let operand = block.argument(0).unwrap();
            let scale = block.argument(1).unwrap();
            let offset = block.argument(2).unwrap();
            let batch_norm_op = batch_norm_training(operand, scale, offset, 0.001, 2, location);
            assert_eq!(batch_norm_op.operands().count(), 3);
            assert_eq!(batch_norm_op.results().count(), 3);
            assert_eq!(batch_norm_op.epsilon(), 0.001);
            assert_eq!(batch_norm_op.feature_index(), 2);
            let batch_norm_op = block.append_operation(batch_norm_op);
            block.append_operation(func::r#return(batch_norm_op.results().collect::<Vec<_>>().as_slice(), location));
            func::func(
                "batch_norm_training_test",
                func::FuncAttributes {
                    arguments: vec![tensor_type.into(), param_type.into(), param_type.into()],
                    results: vec![tensor_type.into(), param_type.into(), param_type.into()],
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
                  func.func @batch_norm_training_test(\
                    %arg0: tensor<2x2x2xf32>, \
                    %arg1: tensor<2xf32>, \
                    %arg2: tensor<2xf32>\
                  ) -> (tensor<2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) {
                    %output, %batch_mean, %batch_var = \"stablehlo.batch_norm_training\"(%arg0, %arg1, %arg2) <{\
                      epsilon = 1.000000e-03 : f32, \
                      feature_index = 2 : i64\
                    }> : (tensor<2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) \
                      -> (tensor<2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
                    return %output, %batch_mean, %batch_var : tensor<2x2x2xf32>, tensor<2xf32>, tensor<2xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_batch_norm_grad() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let tensor_type = context
            .tensor_type(f32_type, &[Size::Static(2), Size::Static(2), Size::Static(2)], None, location)
            .unwrap();
        let param_type = context.tensor_type(f32_type, &[Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (tensor_type, location),
                (param_type, location),
                (param_type, location),
                (param_type, location),
                (tensor_type, location),
            ]);
            let operand = block.argument(0).unwrap();
            let scale = block.argument(1).unwrap();
            let mean = block.argument(2).unwrap();
            let variance = block.argument(3).unwrap();
            let grad_output = block.argument(4).unwrap();
            let batch_norm_op = batch_norm_grad(operand, scale, mean, variance, grad_output, 0.001, 2, location);
            assert_eq!(batch_norm_op.operands().count(), 5);
            assert_eq!(batch_norm_op.results().count(), 3);
            assert_eq!(batch_norm_op.epsilon(), 0.001);
            assert_eq!(batch_norm_op.feature_index(), 2);
            let batch_norm_op = block.append_operation(batch_norm_op);
            block.append_operation(func::r#return(batch_norm_op.results().collect::<Vec<_>>().as_slice(), location));
            func::func(
                "batch_norm_grad_test",
                func::FuncAttributes {
                    arguments: vec![
                        tensor_type.into(),
                        param_type.into(),
                        param_type.into(),
                        param_type.into(),
                        tensor_type.into(),
                    ],
                    results: vec![tensor_type.into(), param_type.into(), param_type.into()],
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
                  func.func @batch_norm_grad_test(\
                    %arg0: tensor<2x2x2xf32>, \
                    %arg1: tensor<2xf32>, \
                    %arg2: tensor<2xf32>, \
                    %arg3: tensor<2xf32>, \
                    %arg4: tensor<2x2x2xf32>\
                  ) -> (tensor<2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) {
                    %grad_operand, %grad_scale, %grad_offset = \"stablehlo.batch_norm_grad\"(\
                      %arg0, \
                      %arg1, \
                      %arg2, \
                      %arg3, \
                      %arg4\
                    ) <{\
                      epsilon = 1.000000e-03 : f32, \
                      feature_index = 2 : i64\
                    }> : (tensor<2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2xf32>) \
                      -> (tensor<2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
                    return %grad_operand, %grad_scale, %grad_offset : tensor<2x2x2xf32>, tensor<2xf32>, tensor<2xf32>
                  }
                }
            "},
        );
    }
}
