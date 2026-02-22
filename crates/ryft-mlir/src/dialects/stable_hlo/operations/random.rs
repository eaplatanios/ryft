#![allow(deprecated)]

use crate::{
    Attribute, DetachedOp, DialectHandle, IntoWithContext, Location, Operation, OperationBuilder, Type, Value,
    mlir_enum_attribute, mlir_op, mlir_op_trait,
};

mlir_enum_attribute!(
    rust_name = RngDistribution,
    mlir_name = RngDistribution,
    description = "StableHLO random number generator distribution",
    variants = {
        Uniform => "UNIFORM",
        Normal => "NORMAL",
    },
    rust_prefix = stable_hlo,
    mlir_prefix = stablehlo,
    mlir_dialect_handle_constructor = stable_hlo,
);

mlir_enum_attribute!(
    rust_name = RngAlgorithm,
    mlir_name = RngAlgorithm,
    description = "StableHLO random number generator algorithm",
    variants = {
        Default => "DEFAULT",
        ThreeFry => "THREE_FRY",
        Philox => "PHILOX",
    },
    rust_prefix = stable_hlo,
    mlir_prefix = stablehlo,
    mlir_dialect_handle_constructor = stable_hlo,
);

/// Name of the [`Attribute`] that is used to store [`RngOperation::rng_distribution`].
pub const RNG_DISTRIBUTION_ATTRIBUTE: &str = "rng_distribution";

/// StableHLO [`Operation`] that generates random numbers using a specific distribution, populating tensors of
/// a specific data type and shape. This operation takes three inputs/operands, `a`, `b`, and `shape`. `a` and `b` are
/// typically scalar values that are used to parameterize the random distribution that is used (i.e., based on the value
/// of [`RngOperation::rng_distribution`]), and `shape` is a one-dimensional integer tensor that determines the shape of
/// the output/result of this operation. `a` and `b` must have the same data type and that is also the data type of the
/// output/result of this operation.
///
/// If [`RngOperation::rng_distribution`] is [`RngDistribution::Uniform`], then the random numbers are generated
/// following the uniform distribution over the interval `[a, b)`. If `a >= b`, the behavior is undefined.
///
/// If [`RngOperation::rng_distribution`] is [`RngDistribution::Normal`], then the random numbers are generated
/// following the normal distribution with mean `a` and standard deviation `b`. If `b < 0`, the behavior is undefined.
///
/// The exact way in which random numbers are generated is implementation-specific. For example, they may or may not be
/// deterministic, and they may or may not use a hidden state internal to the random number generator. For this reason,
/// this operation is not very useful when it comes to reproducibility and is considered deprecated. You should
/// instead use [`RngBitGeneratorOperation`].
///
/// # Example
///
/// The following is an example of an [`RngOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %a = 0
/// // %b = 2
/// // %shape = [3, 3]
/// %result = stablehlo.rng %a, %b, %shape, distribution =  UNIFORM
///   : (tensor<i32>, tensor<i32>, tensor<2xi64>) -> tensor<3x3xi32>
/// // %result: [[1, 0, 1], [1, 1, 1], [0, 0, 0]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#rng)
/// for more information.
#[deprecated(
    note = "Per [StableHLO v1.0 Cleanup #2283](https://github.com/openxla/stablehlo/pull/2283), this operation is \
    being explored for deprecation as it appears to be unused by both frameworks and compilers. As such, it has \
    limited compatibility guarantees (6 months)."
)]
pub trait RngOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the [`RngDistribution`] that this [`RngOperation`] is using.
    fn rng_distribution(&self) -> RngDistribution {
        self.attribute(RNG_DISTRIBUTION_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<RngDistributionAttributeRef>())
            .map(|attribute| attribute.value())
            .unwrap_or_else(|| panic!("invalid '{RNG_DISTRIBUTION_ATTRIBUTE}' attribute in `stable_hlo::rng`"))
    }
}

mlir_op!(Rng);
mlir_op_trait!(Rng, OneResult);
mlir_op_trait!(Rng, ZeroRegions);
mlir_op_trait!(Rng, ZeroSuccessors);

/// Constructs a new detached/owned [`RngOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`RngOperation`] for information on the arguments of this function.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
#[deprecated(
    note = "Per [StableHLO v1.0 Cleanup #2283](https://github.com/openxla/stablehlo/pull/2283), this operation is \
    being explored for deprecation as it appears to be unused by both frameworks and compilers. As such, it has \
    limited compatibility guarantees (6 months)."
)]
pub fn rng<
    'a,
    'b,
    's,
    'c: 'a + 'b + 's,
    't: 'c,
    A: Value<'a, 'c, 't>,
    B: Value<'b, 'c, 't>,
    S: Value<'s, 'c, 't>,
    D: IntoWithContext<'c, 't, RngDistributionAttributeRef<'c, 't>>,
    L: Location<'c, 't>,
>(
    a: A,
    b: B,
    shape: S,
    distribution: D,
    location: L,
) -> DetachedRngOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.rng", location)
        .add_operand(a)
        .add_operand(b)
        .add_operand(shape)
        .add_attribute(RNG_DISTRIBUTION_ATTRIBUTE, distribution.into_with_context(location.context()))
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::rng`")
}

/// Name of the [`Attribute`] that is used to store [`RngBitGeneratorOperation::rng_algorithm`].
pub const RNG_ALGORITHM_ATTRIBUTE: &str = "rng_algorithm";

/// StableHLO [`Operation`] that generates random bits using the provided initial state (as its only input/operand)
/// and algorithm specified by [`RngBitGeneratorOperation::rng_algorithm`]. It produces as outputs the updated
/// state (that can be used later on to generate more random numbers) and the sampled output tensor.
///
/// The outputs of this operation are guaranteed to be a deterministic function of its input state, but they are not
/// guaranteed to be deterministic across different implementations of this operation (e.g., targeting different
/// hardware accelerators).
///
/// The type of the output tensor is determined by (and matches) and the type of the input state and must be an
/// integer or a floating-point type.
///
/// Note that not all [`RngAlgorithm`] support all data types.
///
/// # Example
///
/// The following is an example of an [`RngBitGeneratorOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %output_state, %output = stablehlo.rng_bit_generator %initial_state, algorithm =  THREE_FRY
///   : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2x3xui32>)
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#rng_bit_generator)
/// for more information.
pub trait RngBitGeneratorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the [`RngAlgorithm`] that this [`RngBitGeneratorOperation`] is using.
    fn rng_algorithm(&self) -> RngAlgorithm {
        self.attribute(RNG_ALGORITHM_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<RngAlgorithmAttributeRef<'c, 't>>())
            .map(|attribute| attribute.value())
            .unwrap_or_else(|| {
                panic!("invalid '{RNG_ALGORITHM_ATTRIBUTE}' attribute in `stable_hlo::rng_bit_generator`")
            })
    }
}

mlir_op!(RngBitGenerator);
mlir_op_trait!(RngBitGenerator, ZeroRegions);
mlir_op_trait!(RngBitGenerator, ZeroSuccessors);

/// Constructs a new detached/owned [`RngBitGeneratorOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`RngBitGeneratorOperation`] for information on the arguments of this function.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn rng_bit_generator<
    's,
    'c: 's,
    't: 'c,
    S: Value<'s, 'c, 't>,
    A: IntoWithContext<'c, 't, RngAlgorithmAttributeRef<'c, 't>>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    state: S,
    algorithm: A,
    output_type: T,
    location: L,
) -> DetachedRngBitGeneratorOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.rng_bit_generator", location)
        .add_operand(state)
        .add_attribute(RNG_ALGORITHM_ATTRIBUTE, algorithm.into_with_context(location.context()))
        .add_result(state.r#type())
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::rng_bit_generator`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};
    use crate::dialects::func;
    use crate::{Attribute, Block, Context, Operation, Size};

    use super::{RngAlgorithm, RngBitGeneratorOperation, RngDistribution, RngOperation, rng, rng_bit_generator};

    #[test]
    fn test_rng_distribution_attribute() {
        let context = Context::new();
        let attribute = context.stable_hlo_rng_distribution(RngDistribution::Normal);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), RngDistribution::Normal);
    }

    #[test]
    fn test_rng_distribution_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_rng_distribution(RngDistribution::Normal);
        let attribute_2 = context.stable_hlo_rng_distribution(RngDistribution::Normal);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.stable_hlo_rng_distribution(RngDistribution::Uniform);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.stable_hlo_rng_distribution(RngDistribution::Normal);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_rng_distribution_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.stable_hlo_rng_distribution(RngDistribution::Normal);
        test_attribute_display_and_debug(attribute, "#stablehlo<rng_distribution NORMAL>");
    }

    #[test]
    fn test_rng_distribution_attribute_casting() {
        let context = Context::new();
        let attribute = context.stable_hlo_rng_distribution(RngDistribution::Normal);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_rng_algorithm_attribute() {
        let context = Context::new();
        let attribute = context.stable_hlo_rng_algorithm(RngAlgorithm::ThreeFry);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value(), RngAlgorithm::ThreeFry);
    }

    #[test]
    fn test_rng_algorithm_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_rng_algorithm(RngAlgorithm::ThreeFry);
        let attribute_2 = context.stable_hlo_rng_algorithm(RngAlgorithm::ThreeFry);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.stable_hlo_rng_algorithm(RngAlgorithm::Default);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.stable_hlo_rng_algorithm(RngAlgorithm::ThreeFry);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_rng_algorithm_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.stable_hlo_rng_algorithm(RngAlgorithm::ThreeFry);
        test_attribute_display_and_debug(attribute, "#stablehlo<rng_algorithm THREE_FRY>");
    }

    #[test]
    fn test_rng_algorithm_attribute_casting() {
        let context = Context::new();
        let attribute = context.stable_hlo_rng_algorithm(RngAlgorithm::ThreeFry);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_rng() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let i64_type = context.signless_integer_type(64);
        let bound_type = context.tensor_type(f32_type, &[], None, location).unwrap();
        let shape_type = context.tensor_type(i64_type, &[Size::Static(2)], None, location).unwrap();
        let result_type = context.tensor_type(f32_type, &[Size::Dynamic, Size::Dynamic], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(bound_type, location), (bound_type, location), (shape_type, location)]);
            let a = block.argument(0).unwrap();
            let b = block.argument(1).unwrap();
            let shape = block.argument(2).unwrap();
            let rng_op = rng(a, b, shape, RngDistribution::Uniform, location);
            assert_eq!(rng_op.rng_distribution(), RngDistribution::Uniform);
            assert_eq!(rng_op.operands().count(), 3);
            assert_eq!(rng_op.results().count(), 1);
            let rng_block = block.append_operation(rng_op);
            block.append_operation(func::r#return(&[rng_block.result(0).unwrap()], location));
            func::func(
                "rng_uniform_test",
                func::FuncAttributes {
                    arguments: vec![bound_type.into(), bound_type.into(), shape_type.into()],
                    results: vec![result_type.into()],
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
                  func.func @rng_uniform_test(\
                    %arg0: tensor<f32>, \
                    %arg1: tensor<f32>, \
                    %arg2: tensor<2xi64>\
                  ) -> tensor<?x?xf32> {
                    %0 = stablehlo.rng %arg0, %arg1, %arg2, distribution =  UNIFORM \
                      : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<?x?xf32>
                    return %0 : tensor<?x?xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_rng_bit_generator() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let ui64_type = context.unsigned_integer_type(64);
        let state_type = context.tensor_type(ui64_type, &[Size::Static(2)], None, location).unwrap();
        let ui32_type = context.unsigned_integer_type(32);
        let output_type = context.tensor_type(ui32_type, &[Size::Static(2), Size::Static(3)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(state_type, location)]);
            let initial_state = block.argument(0).unwrap();
            let rng_op = rng_bit_generator(initial_state, RngAlgorithm::ThreeFry, output_type, location);
            assert_eq!(rng_op.rng_algorithm(), RngAlgorithm::ThreeFry);
            assert_eq!(rng_op.operands().count(), 1);
            assert_eq!(rng_op.results().count(), 2);
            let rng_block = block.append_operation(rng_op);
            block.append_operation(func::r#return(
                &[rng_block.result(0).unwrap(), rng_block.result(1).unwrap()],
                location,
            ));
            func::func(
                "rng_bit_generator_test",
                func::FuncAttributes {
                    arguments: vec![state_type.into()],
                    results: vec![state_type.into(), output_type.into()],
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
                  func.func @rng_bit_generator_test(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<2x3xui32>) {
                    %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm =  THREE_FRY \
                      : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2x3xui32>)
                    return %output_state, %output : tensor<2xui64>, tensor<2x3xui32>
                  }
                }
            "},
        );
    }
}
