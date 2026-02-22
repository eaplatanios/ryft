#![allow(deprecated)]

use ryft_xla_sys::bindings::mlirOperationGetOperand;

use crate::{
    Attribute, DenseInteger64ArrayAttributeRef, DetachedOp, DetachedRegion, DialectHandle, FromWithContext, Location,
    OneRegion, Operation, OperationBuilder, RegionRef, Value, ValueRef, mlir_op, mlir_op_trait,
};

use super::{HasPadding, PADDING_ATTRIBUTE};

/// Name of the [`Attribute`] that is used to store [`MapOperation::dimensions`].
pub const MAP_DIMENSIONS_ATTRIBUTE: &str = "dimensions";

/// StableHLO [`Operation`] that applies a computation (i.e., a function) elementwise to input tensors "zipped"
/// together, across specific dimensions, producing a resulting tensor.
///
/// This operation has a variable number of inputs (i.e., operands) and a single output (i.e., result) tensor.
/// It also has an attribute storing the dimensions of the input tensors over which to apply the mapping function
/// (i.e., [`MapOperation::dimensions`]). The mapping function itself is stored in the only [`Region`](crate::Region)
/// that this [`Operation`] contains.
///
/// Note that all inputs (i.e., operands) of this operation must have the same shape. Furthermore, the types of
/// the arguments and result of the underlying mapping function must match the types of the arguments and the result
/// of this operation itself.
///
/// # Example
///
/// The following is an example of a [`MapOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %input0: [[0, 1], [2, 3]]
/// // %input1: [[4, 5], [6, 7]]
/// %result = \"stablehlo.map\"(%arg0, %arg1) <{dimensions = array<i64: 0, 1>}> ({
/// ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>):
///   %1 = stablehlo.multiply %arg2, %arg3 : tensor<i64>
///   stablehlo.return %1 : tensor<i64>
/// }) : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi64>
/// // %result: [[0, 5], [12, 21]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#map) for more information.
#[deprecated(
    note = "Per [StableHLO v1.0 Cleanup #2283](https://github.com/openxla/stablehlo/pull/2283), this operation is \
    being explored for deprecation as it appears to be unused by both frameworks and compilers. As such, it has \
    limited compatibility guarantees (6 months)."
)]
pub trait MapOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneRegion<'o, 'c, 't> {
    /// Returns the dimensions over which this [`MapOperation`] applies the underlying function.
    fn dimensions(&self) -> Vec<usize> {
        self.attribute(MAP_DIMENSIONS_ATTRIBUTE)
            .and_then(|attribute| {
                attribute
                    .cast::<DenseInteger64ArrayAttributeRef>()
                    .map(|attribute| attribute.values().map(|value| value as usize).collect())
            })
            .unwrap_or_else(|| panic!("invalid '{MAP_DIMENSIONS_ATTRIBUTE}' attribute in `stable_hlo::map`"))
    }

    /// Returns a reference to the [`Region`](crate::Region) that contains the mapping computation
    /// used by this [`MapOperation`].
    fn computation(&self) -> RegionRef<'o, 'c, 't> {
        self.body_region()
    }
}

mlir_op!(Map);
mlir_op_trait!(Map, OneRegion);
mlir_op_trait!(Map, OneResult);
mlir_op_trait!(Map, SingleBlock);
mlir_op_trait!(Map, SingleBlockRegions);
mlir_op_trait!(Map, ZeroSuccessors);

/// Constructs a new detached/owned [`MapOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`MapOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
#[deprecated(
    note = "Per [StableHLO v1.0 Cleanup #2283](https://github.com/openxla/stablehlo/pull/2283), this operation is \
    being explored for deprecation as it appears to be unused by both frameworks and compilers. As such, it has \
    limited compatibility guarantees (6 months)."
)]
pub fn map<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    inputs: &[V],
    dimensions: &[usize],
    computation: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedMapOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.map", location)
        .add_operands(inputs)
        .add_attribute(
            MAP_DIMENSIONS_ATTRIBUTE,
            context
                .dense_i64_array_attribute(dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice())
                .unwrap(),
        )
        .add_region(computation)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::map`")
}

/// Name of the [`Attribute`] that is used to store [`ReduceOperation::dimensions`].
pub const REDUCE_DIMENSIONS_ATTRIBUTE: &str = "dimensions";

/// StableHLO [`Operation`] that applies a reduction computation (i.e., function) along the specified
/// dimensions of its input tensors. The order of reductions is implementation-specific, which means that
/// [`ReduceOperation::computation`] and [`ReduceOperation::initial_values`] must form a monoid to guarantee
/// that the operation produces the same results for all inputs on all implementations. However, this condition does
/// not hold for many popular reductions. E.g., if [`ReduceOperation::computation`] uses floating-point addition and
/// zeros for [`ReduceOperation::initial_values`], then we do not actually have a monoid because floating-point
/// addition is not associative.
///
/// Note that [`ReduceOperation::initial_values`] has the same length as [`ReduceOperation::inputs`] and also,
/// all initial values are scalars with the same data/element type as the corresponding input tensor.
///
/// # Example
///
/// The following is an example of a [`ReduceOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %input = [[0, 1, 2, 3, 4, 5]]
/// // %init_value = 0
/// %result = stablehlo.reduce(%input init: %init_value)
///   applies stablehlo.add across dimensions = [1]
/// : (tensor<3x4xi32>, tensor<i32>) -> tensor<3xi32>
/// // %result = [15]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#reduce)
/// for more information.
pub trait ReduceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneRegion<'o, 'c, 't> {
    /// Returns an [`Iterator`] over the input values of this [`ReduceOperation`].
    ///
    /// Note that the returned iterator does not hold a borrowed reference to the underlying [`Context`](crate::Context)
    /// because that would make it impossible to perform mutating operations on that context (e.g., from within
    /// [`Pass`](crate::Pass)es) while iterating over the contents of that iterator.
    fn inputs<'r>(&'r self) -> impl Iterator<Item = ValueRef<'o, 'c, 't>> {
        let operand_count = self.operand_count();
        (0..operand_count / 2).map(|index| unsafe {
            ValueRef::from_c_api(mlirOperationGetOperand(self.to_c_api(), index.cast_signed()), self.context()).unwrap()
        })
    }

    /// Returns an [`Iterator`] over the initial values of this [`ReduceOperation`].
    ///
    /// Note that the returned iterator does not hold a borrowed reference to the underlying [`Context`](crate::Context)
    /// because that would make it impossible to perform mutating operations on that context (e.g., from within
    /// [`Pass`](crate::Pass)es) while iterating over the contents of that iterator.
    fn initial_values<'r>(&'r self) -> impl Iterator<Item = ValueRef<'o, 'c, 't>> {
        let operand_count = self.operand_count();
        (operand_count / 2..operand_count).map(|index| unsafe {
            ValueRef::from_c_api(mlirOperationGetOperand(self.to_c_api(), index.cast_signed()), self.context()).unwrap()
        })
    }

    /// Returns the dimensions over which this [`ReduceOperation`] applies the underlying function.
    fn dimensions(&self) -> Vec<usize> {
        self.attribute(REDUCE_DIMENSIONS_ATTRIBUTE)
            .and_then(|attribute| {
                attribute
                    .cast::<DenseInteger64ArrayAttributeRef>()
                    .map(|attribute| attribute.values().map(|value| value as usize).collect())
            })
            .unwrap_or_else(|| panic!("invalid '{REDUCE_DIMENSIONS_ATTRIBUTE}' attribute in `stable_hlo::reduce`"))
    }

    /// Returns a reference to the [`Region`](crate::Region) that contains the reduction computation
    /// used by this [`ReduceOperation`].
    fn computation(&self) -> RegionRef<'o, 'c, 't> {
        self.body_region()
    }
}

mlir_op!(Reduce);
mlir_op_trait!(Reduce, OneRegion);
mlir_op_trait!(Reduce, SingleBlock);
mlir_op_trait!(Reduce, SingleBlockRegions);
mlir_op_trait!(Reduce, ZeroSuccessors);

/// Constructs a new detached/owned [`ReduceOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`ReduceOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn reduce<
    'input,
    'initial_value,
    'c: 'input + 'initial_value,
    't: 'c,
    Input: Value<'input, 'c, 't>,
    InitialValue: Value<'initial_value, 'c, 't>,
    L: Location<'c, 't>,
>(
    inputs: &[Input],
    initial_values: &[InitialValue],
    dimensions: &[usize],
    computation: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedReduceOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.reduce", location)
        .add_operands(inputs)
        .add_operands(initial_values)
        .add_attribute(
            REDUCE_DIMENSIONS_ATTRIBUTE,
            context
                .dense_i64_array_attribute(dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice())
                .unwrap(),
        )
        .add_region(computation)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::reduce`")
}

/// Name of the [`Attribute`] that is used to store [`ReduceWindowOperation::window_dimensions`].
pub const REDUCE_WINDOW_DIMENSIONS_ATTRIBUTE: &str = "window_dimensions";

/// Name of the [`Attribute`] that is used to store [`ReduceWindowOperation::window_strides`].
pub const REDUCE_WINDOW_STRIDES_ATTRIBUTE: &str = "window_strides";

/// Name of the [`Attribute`] that is used to store [`ReduceWindowOperation::base_dilations`].
pub const REDUCE_WINDOW_BASE_DILATIONS_ATTRIBUTE: &str = "base_dilations";

/// Name of the [`Attribute`] that is used to store [`ReduceWindowOperation::window_dilations`].
pub const REDUCE_WINDOW_DILATIONS_ATTRIBUTE: &str = "window_dilations";

/// StableHLO [`Operation`] that applies a reduction computation (i.e., function) to window that slide over its input
/// tensors, producing output tensors with the same shape. For each window position, the operation applies the reduction
/// [`ReduceWindowOperation::computation`] to all elements in that window, starting with the corresponding
/// initial value from [`ReduceWindowOperation::initial_values`].
///
/// # Example
///
/// The following is an example of a [`ReduceWindowOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %input = [[1, 2], [3, 4], [5, 6]]
/// // %init_value = 0
/// %result = "stablehlo.reduce_window"(%input, %init_value) <{
///   window_dimensions = array<i64: 2, 1>,
///   window_strides = array<i64: 4, 1>,
///   base_dilations = array<i64: 2, 1>,
///   window_dilations = array<i64: 3, 1>,
///   padding = dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>,
/// }> ({
///   ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
///     %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
///     stablehlo.return %0 : tensor<i64>
/// }) : (tensor<3x2xi64>, tensor<i64>) -> tensor<2x2xi64>
/// // %result = [[0, 0], [3, 4]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#reduce_window)
/// for more information.
pub trait ReduceWindowOperation<'o, 'c: 'o, 't: 'c>:
    Operation<'o, 'c, 't> + HasPadding<'o, 'c, 't> + OneRegion<'o, 'c, 't>
{
    /// Returns an [`Iterator`] over the input values of this [`ReduceWindowOperation`].
    ///
    /// Note that the returned iterator does not hold a borrowed reference to the underlying [`Context`](crate::Context)
    /// because that would make it impossible to perform mutating operations on that context (e.g., from within
    /// [`Pass`](crate::Pass)es) while iterating over the contents of that iterator.
    fn inputs<'r>(&'r self) -> impl Iterator<Item = ValueRef<'o, 'c, 't>> {
        let operand_count = self.operand_count();
        (0..operand_count / 2).map(|index| unsafe {
            ValueRef::from_c_api(mlirOperationGetOperand(self.to_c_api(), index.cast_signed()), self.context()).unwrap()
        })
    }

    /// Returns an [`Iterator`] over the initial values of this [`ReduceWindowOperation`].
    ///
    /// Note that the returned iterator does not hold a borrowed reference to the underlying [`Context`](crate::Context)
    /// because that would make it impossible to perform mutating operations on that context (e.g., from within
    /// [`Pass`](crate::Pass)es) while iterating over the contents of that iterator.
    fn initial_values<'r>(&'r self) -> impl Iterator<Item = ValueRef<'o, 'c, 't>> {
        let operand_count = self.operand_count();
        (operand_count / 2..operand_count).map(|index| unsafe {
            ValueRef::from_c_api(mlirOperationGetOperand(self.to_c_api(), index.cast_signed()), self.context()).unwrap()
        })
    }

    /// Returns the window dimensions for this [`ReduceWindowOperation`].
    fn window_dimensions(&self) -> Vec<usize> {
        self.attribute(REDUCE_WINDOW_DIMENSIONS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().map(|value| value as usize).collect())
            .unwrap_or_else(|| panic!("invalid '{REDUCE_WINDOW_DIMENSIONS_ATTRIBUTE}' attribute in `stable_hlo::reduce_window`"))
    }

    /// Returns the window strides for this [`ReduceWindowOperation`], if specified.
    fn window_strides(&self) -> Option<Vec<usize>> {
        self.attribute(REDUCE_WINDOW_STRIDES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().map(|value| value as usize).collect())
    }

    /// Returns the base dilations for this [`ReduceWindowOperation`], if specified.
    fn base_dilations(&self) -> Option<Vec<usize>> {
        self.attribute(REDUCE_WINDOW_BASE_DILATIONS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().map(|value| value as usize).collect())
    }

    /// Returns the window dilations for this [`ReduceWindowOperation`], if specified.
    fn window_dilations(&self) -> Option<Vec<usize>> {
        self.attribute(REDUCE_WINDOW_DILATIONS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().map(|value| value as usize).collect())
    }

    /// Returns a reference to the [`Region`](crate::Region) that contains the reduction computation
    /// used by this [`ReduceWindowOperation`].
    fn computation(&self) -> RegionRef<'o, 'c, 't> {
        self.body_region()
    }
}

mlir_op!(ReduceWindow);
mlir_op_trait!(ReduceWindow, OneRegion);
mlir_op_trait!(ReduceWindow, SingleBlock);
mlir_op_trait!(ReduceWindow, SingleBlockRegions);
mlir_op_trait!(ReduceWindow, ZeroSuccessors);
mlir_op_trait!(ReduceWindow, @local HasPadding);

/// Constructs a new detached/owned [`ReduceWindowOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`ReduceWindowOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
#[allow(clippy::too_many_arguments)]
pub fn reduce_window<
    'input,
    'initial_value,
    'c: 'input + 'initial_value,
    't: 'c,
    Input: Value<'input, 'c, 't>,
    InitialValue: Value<'initial_value, 'c, 't>,
    L: Location<'c, 't>,
>(
    inputs: &[Input],
    initial_values: &[InitialValue],
    window_dimensions: &[usize],
    window_strides: Option<&[usize]>,
    base_dilations: Option<&[usize]>,
    window_dilations: Option<&[usize]>,
    padding: Option<&[(usize, usize)]>,
    computation: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedReduceWindowOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    let mut builder = OperationBuilder::new("stablehlo.reduce_window", location)
        .add_operands(inputs)
        .add_operands(initial_values)
        .add_attribute(
            REDUCE_WINDOW_DIMENSIONS_ATTRIBUTE,
            DenseInteger64ArrayAttributeRef::from_with_context(
                window_dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice(),
                context,
            ),
        );

    if let Some(window_strides) = window_strides {
        builder = builder.add_attribute(
            REDUCE_WINDOW_STRIDES_ATTRIBUTE,
            DenseInteger64ArrayAttributeRef::from_with_context(
                window_strides.iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice(),
                context,
            ),
        );
    }

    if let Some(base_dilations) = base_dilations {
        builder = builder.add_attribute(
            REDUCE_WINDOW_BASE_DILATIONS_ATTRIBUTE,
            DenseInteger64ArrayAttributeRef::from_with_context(
                base_dilations.iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice(),
                context,
            ),
        );
    }

    if let Some(window_dilations) = window_dilations {
        builder = builder.add_attribute(
            REDUCE_WINDOW_DILATIONS_ATTRIBUTE,
            DenseInteger64ArrayAttributeRef::from_with_context(
                window_dilations.iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice(),
                context,
            ),
        );
    }

    if let Some(padding) = padding {
        builder = builder.add_attribute(PADDING_ATTRIBUTE, location.context().stable_hlo_padding(padding, location));
    }

    builder
        .add_region(computation)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::reduce_window`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::{func, stable_hlo};
    use crate::{Block, Context, Operation, Region, Size, Value};

    use super::*;

    #[test]
    fn test_map() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let input_type = context.tensor_type(i64_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        let scalar_i64_type = context.tensor_type(i64_type, &[], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_type, location), (input_type, location)]);
            let mut computation_region = context.region();
            let mut computation_block = context.block(&[(scalar_i64_type, location), (scalar_i64_type, location)]);
            let multiply_op = stable_hlo::multiply(
                computation_block.argument(0).unwrap(),
                computation_block.argument(1).unwrap(),
                location,
            );
            let multiply_op = computation_block.append_operation(multiply_op);
            let multiply_result = multiply_op.result(0).unwrap().as_ref();
            computation_block.append_operation(stable_hlo::r#return(&[multiply_result], location));
            computation_region.append_block(computation_block);
            let map_op = map(
                &[block.argument(0).unwrap(), block.argument(1).unwrap()],
                &[0, 1],
                computation_region.into(),
                location,
            );
            assert_eq!(map_op.dimensions(), vec![0, 1]);
            assert_eq!(map_op.computation().blocks().count(), 1);
            assert_eq!(map_op.operands().count(), 2);
            assert_eq!(map_op.results().count(), 1);
            assert_eq!(map_op.regions().count(), 1);
            let map_op = block.append_operation(map_op);
            block.append_operation(func::r#return(&[map_op.result(0).unwrap()], location));
            func::func(
                "map_test",
                func::FuncAttributes {
                    arguments: vec![input_type.into(), input_type.into()],
                    results: vec![input_type.into()],
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
                  func.func @map_test(%arg0: tensor<2x2xi64>, %arg1: tensor<2x2xi64>) -> tensor<2x2xi64> {
                    %0 = \"stablehlo.map\"(%arg0, %arg1) <{dimensions = array<i64: 0, 1>}> ({
                    ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>):
                      %1 = stablehlo.multiply %arg2, %arg3 : tensor<i64>
                      stablehlo.return %1 : tensor<i64>
                    }) : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi64>
                    return %0 : tensor<2x2xi64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_reduce() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let input_type = context.tensor_type(i32_type, &[Size::Static(3), Size::Static(4)], None, location).unwrap();
        let initial_value_type = context.tensor_type(i32_type, &[], None, location).unwrap();
        let output_type = context.tensor_type(i32_type, &[Size::Static(3)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_type, location), (initial_value_type, location)]);
            let input = block.argument(0).unwrap();
            let initial_value = block.argument(1).unwrap();
            let mut region = context.region();
            let mut region_block = context.block(&[(initial_value_type, location), (initial_value_type, location)]);
            let lhs = region_block.argument(0).unwrap();
            let rhs = region_block.argument(1).unwrap();
            let add_op = stable_hlo::add(lhs, rhs, location);
            let add_op = region_block.append_operation(add_op);
            let return_op = stable_hlo::r#return(&[add_op.result(0).unwrap()], location);
            region_block.append_operation(return_op);
            region.append_block(region_block);
            let reduce_op = reduce(&[input], &[initial_value], &[1], region.into(), location);
            assert_eq!(reduce_op.inputs().collect::<Vec<_>>(), vec![input]);
            assert_eq!(reduce_op.initial_values().collect::<Vec<_>>(), vec![initial_value]);
            assert_eq!(reduce_op.dimensions(), vec![1]);
            assert_eq!(reduce_op.computation().blocks().count(), 1);
            assert_eq!(reduce_op.operands().count(), 2);
            assert_eq!(reduce_op.results().count(), 1);
            let reduce_op = block.append_operation(reduce_op);
            block.append_operation(func::r#return(&[reduce_op.result(0).unwrap()], location));
            func::func(
                "reduce_test",
                func::FuncAttributes {
                    arguments: vec![input_type.into(), initial_value_type.into()],
                    results: vec![output_type.into()],
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
                  func.func @reduce_test(%arg0: tensor<3x4xi32>, %arg1: tensor<i32>) -> tensor<3xi32> {
                    %0 = stablehlo.reduce(%arg0 init: %arg1) \
                      applies stablehlo.add across dimensions = [1] \
                    : (tensor<3x4xi32>, tensor<i32>) -> tensor<3xi32>
                    return %0 : tensor<3xi32>
                  }
                }
            "}
        );
    }

    #[test]
    fn test_reduce_window() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let input_type = context
            .tensor_type(
                f32_type,
                &[Size::Static(1), Size::Static(4), Size::Static(4), Size::Static(1)],
                None,
                location,
            )
            .unwrap();
        let initial_value_type = context.tensor_type(f32_type, &[], None, location).unwrap();
        let output_type = context
            .tensor_type(
                f32_type,
                &[Size::Static(7), Size::Static(3), Size::Static(2), Size::Static(1)],
                None,
                location,
            )
            .unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_type, location), (initial_value_type, location)]);
            let input = block.argument(0).unwrap();
            let initial_value = block.argument(1).unwrap();
            let mut region = context.region();
            let mut region_block = context.block(&[(initial_value_type, location), (initial_value_type, location)]);
            let lhs = region_block.argument(0).unwrap();
            let rhs = region_block.argument(1).unwrap();
            let max_op = stable_hlo::maximum(lhs, rhs, location);
            let max_op = region_block.append_operation(max_op);
            let return_op = stable_hlo::r#return(&[max_op.result(0).unwrap()], location);
            region_block.append_operation(return_op);
            region.append_block(region_block);
            let reduce_window_op = reduce_window(
                &[input],
                &[initial_value],
                &[1, 2, 2, 1],
                Some(&[1, 2, 2, 1]),
                Some(&[1, 1, 1, 1]),
                Some(&[1, 1, 1, 1]),
                Some(&[(4, 2), (0, 2), (0, 0), (0, 0)]),
                region.into(),
                location,
            );
            assert_eq!(reduce_window_op.inputs().collect::<Vec<_>>(), vec![input]);
            assert_eq!(reduce_window_op.initial_values().collect::<Vec<_>>(), vec![initial_value]);
            assert_eq!(reduce_window_op.window_dimensions(), vec![1, 2, 2, 1]);
            assert_eq!(reduce_window_op.window_strides(), Some(vec![1, 2, 2, 1]));
            assert_eq!(reduce_window_op.base_dilations(), Some(vec![1, 1, 1, 1]));
            assert_eq!(reduce_window_op.window_dilations(), Some(vec![1, 1, 1, 1]));
            assert_eq!(reduce_window_op.padding(), Some(vec![(4, 2), (0, 2), (0, 0), (0, 0)]));
            assert_eq!(reduce_window_op.computation().blocks().count(), 1);
            assert_eq!(reduce_window_op.operands().count(), 2);
            assert_eq!(reduce_window_op.results().count(), 1);
            let reduce_window_op = block.append_operation(reduce_window_op);
            block.append_operation(func::r#return(&[reduce_window_op.result(0).unwrap()], location));
            func::func(
                "reduce_window_test",
                func::FuncAttributes {
                    arguments: vec![input_type.into(), initial_value_type.into()],
                    results: vec![output_type.into()],
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
                  func.func @reduce_window_test(%arg0: tensor<1x4x4x1xf32>, %arg1: tensor<f32>) -> tensor<7x3x2x1xf32> {
                    %0 = \"stablehlo.reduce_window\"(%arg0, %arg1) <{\
                      base_dilations = array<i64: 1, 1, 1, 1>, \
                      padding = dense<[[4, 2], [0, 2], [0, 0], [0, 0]]> : tensor<4x2xi64>, \
                      window_dilations = array<i64: 1, 1, 1, 1>, \
                      window_dimensions = array<i64: 1, 2, 2, 1>, \
                      window_strides = array<i64: 1, 2, 2, 1>\
                    }> ({
                    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
                      stablehlo.return %1 : tensor<f32>
                    }) : (tensor<1x4x4x1xf32>, tensor<f32>) -> tensor<7x3x2x1xf32>
                    return %0 : tensor<7x3x2x1xf32>
                  }
                }
            "}
        );
    }
}
