#![allow(deprecated)]

use crate::{
    Attribute, DetachedOp, DialectHandle, IntegerAttributeRef, Location, Operation, OperationBuilder, Value, mlir_op,
    mlir_op_trait,
};

/// StableHLO [`Operation`] that constructs a tuple (i.e., it combines multiple values into a single composite tuple
/// structure). Tuples are a legacy feature in StableHLO that exists primarily for compatibility with HLO, where they
/// are used to represent variadic inputs and outputs. StableHLO natively supports variadic inputs and outputs, making
/// tuples less essential in modern usage.
///
/// # Example
///
/// The following is an example of a [`TupleOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %val0: [1.0, 2.0]
/// // %val1: 3
/// %0 = stablehlo.tuple %arg0, %arg1 : tuple<tensor<2xf64>, tensor<i64>>
/// // %result: ([1.0, 2.0], 3)
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#tuple)
/// for more information.
#[deprecated(
    note = "Per [StableHLO v1.0 Cleanup #2283](https://github.com/openxla/stablehlo/pull/2283), this operation is \
    being explored for deprecation as it appears to be unused by both frameworks and compilers. As such, it has \
    limited compatibility guarantees (6 months)."
)]
pub trait TupleOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Tuple);
mlir_op_trait!(Tuple, OneResult);
mlir_op_trait!(Tuple, ZeroRegions);
mlir_op_trait!(Tuple, ZeroSuccessors);

/// Constructs a new detached/owned [`TupleOperation`] at the specified [`Location`].
#[deprecated(
    note = "Per [StableHLO v1.0 Cleanup #2283](https://github.com/openxla/stablehlo/pull/2283), this operation is \
    being explored for deprecation as it appears to be unused by both frameworks and compilers. As such, it has \
    limited compatibility guarantees (6 months)."
)]
pub fn tuple<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    values: &[V],
    location: L,
) -> DetachedTupleOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.tuple", location)
        .add_operands(values)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::tuple`")
}

/// Name of the [`Attribute`] that is used to store [`GetTupleElementOperation::index`].
pub const GET_TUPLE_ELEMENT_INDEX_ATTRIBUTE: &'static str = "index";

/// StableHLO [`Operation`] that extracts an element from a tuple based on its index in the tuple. Tuples are a legacy
/// feature in StableHLO that exists primarily for compatibility with HLO, where they are used to represent variadic
/// inputs and outputs. StableHLO natively supports variadic inputs and outputs, making tuples less essential in modern
/// usage.
///
/// # Example
///
/// The following is an example of a [`GetTupleElementOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand: ([1.0, 2.0], 3)
/// %result = stablehlo.get_tuple_element %operand[0] : (tuple<tensor<2xf64>, tensor<i64>>) -> tensor<2xf64>
/// // %result: [1.0, 2.0]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#get_tuple_element)
/// for more information.
#[deprecated(
    note = "Per [StableHLO v1.0 Cleanup #2283](https://github.com/openxla/stablehlo/pull/2283), this operation is \
    being explored for deprecation as it appears to be unused by both frameworks and compilers. As such, it has \
    limited compatibility guarantees (6 months)."
)]
pub trait GetTupleElementOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the index of the tuple element that this [`GetTupleElementOperation`] extracts.
    fn index(&self) -> usize {
        self.attribute(GET_TUPLE_ELEMENT_INDEX_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value() as usize)
            .expect(&format!(
                "invalid '{GET_TUPLE_ELEMENT_INDEX_ATTRIBUTE}' attribute in `stable_hlo::get_tuple_element`",
            ))
    }
}

mlir_op!(GetTupleElement);
mlir_op_trait!(GetTupleElement, OneResult);
mlir_op_trait!(GetTupleElement, ZeroRegions);

/// Constructs a new detached/owned [`GetTupleElementOperation`] at the specified [`Location`].
#[deprecated(
    note = "Per [StableHLO v1.0 Cleanup #2283](https://github.com/openxla/stablehlo/pull/2283), this operation is \
    being explored for deprecation as it appears to be unused by both frameworks and compilers. As such, it has \
    limited compatibility guarantees (6 months)."
)]
pub fn get_tuple_element<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    operand: V,
    index: usize,
    location: L,
) -> DetachedGetTupleElementOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.get_tuple_element", location)
        .add_operand(operand)
        .add_attribute(
            GET_TUPLE_ELEMENT_INDEX_ATTRIBUTE,
            location.context().integer_attribute(location.context().signless_integer_type(32), index as i64),
        )
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::get_tuple_element`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Block, Context, Operation, Size, Value};

    use super::{GetTupleElementOperation, get_tuple_element, tuple};

    #[test]
    fn test_tuple() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f64_type = context.float64_type();
        let i64_type = context.signless_integer_type(64);
        let tensor_f64_type = context.tensor_type(f64_type, &[Size::Static(2)], None, location).unwrap();
        let tensor_i64_type = context.tensor_type(i64_type, &[], None, location).unwrap();
        let tuple_type = context.tuple_type(&[tensor_f64_type, tensor_i64_type]);
        module.body().append_operation({
            let mut block = context.block(&[(tensor_f64_type, location), (tensor_i64_type, location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let tuple_op = tuple(&[arg_0, arg_1], location);
            assert_eq!(tuple_op.operands().count(), 2);
            assert_eq!(tuple_op.results().count(), 1);
            assert_eq!(tuple_op.result(0).unwrap().r#type(), tuple_type);
            let tuple_block = block.append_operation(tuple_op);
            block.append_operation(func::r#return(&[tuple_block.result(0).unwrap()], location));
            func::func(
                "tuple_test",
                func::FuncAttributes {
                    arguments: vec![tensor_f64_type.into(), tensor_i64_type.into()],
                    results: vec![tuple_type.into()],
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
                  func.func @tuple_test(%arg0: tensor<2xf64>, %arg1: tensor<i64>) -> tuple<tensor<2xf64>, tensor<i64>> {
                    %0 = stablehlo.tuple %arg0, %arg1 : tuple<tensor<2xf64>, tensor<i64>>
                    return %0 : tuple<tensor<2xf64>, tensor<i64>>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_get_tuple_element() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f64_type = context.float64_type();
        let i64_type = context.signless_integer_type(64);
        let tensor_f64_type = context.tensor_type(f64_type, &[Size::Static(2)], None, location).unwrap();
        let tensor_i64_type = context.tensor_type(i64_type, &[], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_f64_type, location), (tensor_i64_type, location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let inner_tuple_op = tuple(&[arg_1], location);
            let inner_tuple_block = block.append_operation(inner_tuple_op);
            let inner_tuple_value = inner_tuple_block.result(0).unwrap().as_ref();
            let outer_tuple_op = tuple(&[arg_0.as_ref(), inner_tuple_value], location);
            let outer_tuple_block = block.append_operation(outer_tuple_op);
            let outer_tuple_result = outer_tuple_block.result(0).unwrap();
            let get_element_op = get_tuple_element(outer_tuple_result, 0, location);
            assert_eq!(get_element_op.index(), 0);
            assert_eq!(get_element_op.operands().count(), 1);
            assert_eq!(get_element_op.results().count(), 1);
            let get_element_block = block.append_operation(get_element_op);
            block.append_operation(func::r#return(&[get_element_block.result(0).unwrap()], location));
            func::func(
                "nested_tuple_test",
                func::FuncAttributes {
                    arguments: vec![tensor_f64_type.into(), tensor_i64_type.into()],
                    results: vec![tensor_f64_type.into()],
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
                  func.func @nested_tuple_test(%arg0: tensor<2xf64>, %arg1: tensor<i64>) -> tensor<2xf64> {
                    %0 = stablehlo.tuple %arg1 : tuple<tensor<i64>>
                    %1 = stablehlo.tuple %arg0, %0 : tuple<tensor<2xf64>, tuple<tensor<i64>>>
                    %2 = stablehlo.get_tuple_element %1[0] : (tuple<tensor<2xf64>, tuple<tensor<i64>>>) -> tensor<2xf64>
                    return %2 : tensor<2xf64>
                  }
                }
            "},
        );
    }
}
