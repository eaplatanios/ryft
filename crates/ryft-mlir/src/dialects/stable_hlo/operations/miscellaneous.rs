use std::collections::HashMap;

use ryft_xla_sys::bindings::{MlirAttribute, stablehloOutputOperandAliasGet};

use crate::{
    ArrayAttributeRef, Attribute, AttributeRef, BooleanAttributeRef, Context, DenseInteger64ArrayAttributeRef,
    DenseIntegerElementsAttributeRef, DetachedOp, DetachedRegion, DialectHandle, DictionaryAttributeRef,
    ElementsAttribute, ElementsAttributeRef, FlatSymbolRefAttributeRef, FromWithContext, IntegerAttributeRef,
    IntoWithContext, Location, OneRegion, Operation, OperationBuilder, RegionRef, ShapedType, StringAttributeRef,
    StringRef, Type, Value, ValueRef, mlir_attribute_field, mlir_op, mlir_op_trait, mlir_subtype_trait_impls,
};

/// Name of the [`Attribute`] that is used to store [`ConstantOperation::value`].
pub const CONSTANT_VALUE_ATTRIBUTE: &str = "value";

/// StableHLO [`Operation`] that produces an output tensor from a constant value. That value is represented as an
/// [`ElementsAttribute`] that is stored in this [`Operation`] and is thus known at compile time. This operation
/// serves as the fundamental way to introduce literal values into StableHLO programs.
///
/// # Example
///
/// The following is an example of a [`ConstantOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %output = stablehlo.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
/// // %output: [[0.0, 1.0], [2.0, 3.0]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#constant) for more information.
pub trait ConstantOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the constant value that is stored in this [`Operation`].
    fn value(&self) -> ElementsAttributeRef<'c, 't> {
        self.attribute(CONSTANT_VALUE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast())
            .unwrap_or_else(|| panic!("invalid '{CONSTANT_VALUE_ATTRIBUTE}' attribute in `stable_hlo::constant`"))
    }
}

mlir_op!(Constant);
mlir_op_trait!(Constant, ConstantLike);
mlir_op_trait!(Constant, OneResult);
mlir_op_trait!(Constant, ZeroOperands);
mlir_op_trait!(Constant, ZeroRegions);
mlir_op_trait!(Constant, ZeroSuccessors);

/// Constructs a new detached/owned [`ConstantOperation`] at the specified [`Location`] and with the provided value.
/// The result type is automatically inferred from the provided value. Refer to the documentation of
/// [`ConstantOperation`] for more information on the operation semantics.
pub fn constant<'c, 't, A: ElementsAttribute<'c, 't>, L: Location<'c, 't>>(
    value: A,
    location: L,
) -> DetachedConstantOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.constant", location)
        .add_attribute(CONSTANT_VALUE_ATTRIBUTE, value)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::constant`")
}

/// Name of the [`Attribute`] that is used to store [`DynamicIotaOperation::iota_dimension`]
/// and [`IotaOperation::iota_dimension`].
pub const IOTA_DIMENSION_ATTRIBUTE: &str = "iota_dimension";

/// StableHLO [`Operation`] that fills an output tensor with values in increasing order starting from zero
/// along the [`IotaOperation::iota_dimension`] dimension.
///
/// # Example
///
/// The following are examples of [`IotaOperation`]s represented using their [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %output = stablehlo.iota dim = 0 : tensor<4x5xi32>
/// // %output: [
/// //           [0, 0, 0, 0, 0],
/// //           [1, 1, 1, 1, 1],
/// //           [2, 2, 2, 2, 2],
/// //           [3, 3, 3, 3, 3]
/// //          ]
///
/// %output = stablehlo.iota dim = 1 : tensor<4x5xi32>
/// // %output: [
/// //           [0, 1, 2, 3, 4],
/// //           [0, 1, 2, 3, 4],
/// //           [0, 1, 2, 3, 4],
/// //           [0, 1, 2, 3, 4]
/// //          ]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#iota) for more information.
pub trait IotaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the dimension along which the values of the output tensor of this [`IotaOperation`] increase.
    fn iota_dimension(&self) -> usize {
        self.attribute(IOTA_DIMENSION_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{IOTA_DIMENSION_ATTRIBUTE}' attribute in `stable_hlo::dynamic_iota`"))
            .signless_value() as usize
    }
}

mlir_op!(Iota);
mlir_op_trait!(Iota, OneResult);
mlir_op_trait!(Iota, ZeroRegions);
mlir_op_trait!(Iota, ZeroSuccessors);

/// Constructs a new detached/owned [`IotaOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`IotaOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn iota<'c, 't: 'c, T: ShapedType<'c, 't>, L: Location<'c, 't>>(
    output_type: T,
    iota_dimension: usize,
    location: L,
) -> DetachedIotaOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.iota", location)
        .add_attribute(
            IOTA_DIMENSION_ATTRIBUTE,
            location
                .context()
                .integer_attribute(location.context().signless_integer_type(64), iota_dimension as i64),
        )
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::iota`")
}

/// StableHLO [`Operation`] that fills an output tensor with values in increasing order starting from zero along the
/// [`DynamicIotaOperation::iota_dimension`] dimension. This is equivalent to [`IotaOperation`] except for the fact that
/// the shape of the output tensor is dynamic and provided as the only input/operand of this operation
/// (as a one-dimensional tensor).
///
/// # Example
///
/// The following is an example of a [`DynamicIotaOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %output_shape = stablehlo.constant dense<[4, 5]> : tensor<2xi64>
/// %result = stablehlo.dynamic_iota %output_shape, dim = 0 : (tensor<2xi64>) -> tensor<4x5xi64>
/// // %result: [
/// //           [0, 0, 0, 0, 0],
/// //           [1, 1, 1, 1, 1],
/// //           [2, 2, 2, 2, 2],
/// //           [3, 3, 3, 3, 3]
/// //          ]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#dynamic_iota)
/// for more information.
pub trait DynamicIotaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the dimension along which the values of the output tensor of this [`DynamicIotaOperation`] increase.
    fn iota_dimension(&self) -> usize {
        self.attribute(IOTA_DIMENSION_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{IOTA_DIMENSION_ATTRIBUTE}' attribute in `stable_hlo::dynamic_iota`"))
            .signless_value() as usize
    }
}

mlir_op!(DynamicIota);
mlir_op_trait!(DynamicIota, OneResult);
mlir_op_trait!(DynamicIota, ZeroRegions);
mlir_op_trait!(DynamicIota, ZeroSuccessors);

/// Constructs a new detached/owned [`DynamicIotaOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`DynamicIotaOperation`] for more information on the operation semantics.
///
/// Note that since this operation supports dynamic shapes (as opposed to [`iota`] which only supports static shapes),
/// the provided `output_type` can have certain dimensions set to [`Size::Dynamic`](crate::Size::Dynamic).
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn dynamic_iota<'s, 'c: 's, 't: 'c, S: Value<'s, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    output_shape: S,
    output_type: T,
    iota_dimension: usize,
    location: L,
) -> DetachedDynamicIotaOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.dynamic_iota", location)
        .add_operand(output_shape)
        .add_attribute(
            IOTA_DIMENSION_ATTRIBUTE,
            location
                .context()
                .integer_attribute(location.context().signless_integer_type(64), iota_dimension as i64),
        )
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::dynamic_iota`")
}

/// Name of the [`Attribute`] that is used to store [`SortOperation::dimension`].
pub const SORT_DIMENSION_ATTRIBUTE: &str = "dimension";

/// Name of the [`Attribute`] that is used to store [`SortOperation::is_stable`].
pub const SORT_IS_STABLE_ATTRIBUTE: &str = "is_stable";

/// StableHLO [`Operation`] that sorts 1-dimensional slices of its inputs/operands along their
/// [`SortOperation::dimension`] dimension, together, according to its [`SortOperation::comparator`], to produce its
/// outputs/results. Furthermore, if [`SortOperation::is_stable`] is `true`, then the sorting is stable (i.e., the
/// relative order of elements considered to be equal by the comparator is preserved).
///
/// For the case where there is only one input, two elements `e_0` and `e_1` of that input are considered to be equal
/// by the comparator if and only if `comparator(e_0, e_1) = comparator(e_1, e_0) = false`.
///
/// More formally, the following holds for this operation, for all `r = [r_0, r_1, ..., r_D] ∈ index_space(results[0])`:
///
///   - `r_slice = [r_0, ..., :, ..., r_D]` where the slicing operator, `:`, is inserted at the dimension specified
///     by [`SortOperation::dimension`].
///   - `zipped_inputs = (inputs[0], ..., inputs[N])`, where `N + 1` is the number of inputs/operands of this operation.
///   - `zipped_results[r_slice] = sort(zipped_inputs[r_slice], zipped_comparator)`, where `sort` sorts a 1-dimensional
///     slice in non-descending order expecting that `zipped_comparator` returns `true` if the left-hand side argument
///     is less than the right-hand side argument. Concretely, `zipped_comparator` is defined as (using Python-like
///     pseudocode):
///
///     ```python
///     def zipped_comparator[T](zipped_lhs: list[T], zipped_rhs: list[T]) -> bool:
///         comparator_args = []
///         for lhs_element, rhs_element in zip(zipped_lhs, zipped_rhs):
///             comparator_args.append(lhs_element)
///             comparator_args.append(rhs_element)
///         return comparator(*comparator_args)
///     ```
///
/// The `comparator` function is represented by the only [`Region`](crate::Region) that this [`Operation`] holds. It
/// must have `2 * K` arguments, where `K` is the size of the [`SortOperation::dimension`]th dimension of each of the
/// input and output tensors (which must all have the same shape), and it must return a boolean value with a
/// [`stable_hlo::return`](crate::dialects::stable_hlo::return) operation.
///
/// # Example
///
/// The following is an example of a [`SortOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %input0 = [[1, 2, 3], [3, 2, 1]]
/// // %input1 = [[3, 2, 1], [1, 2, 3]]
/// %result0, %result1 = "stablehlo.sort"(%input0, %input1) <{dimension = 0 : i64, is_stable = true}> ({
/// ^bb0(%input2: tensor<i64>, %input3: tensor<i64>, %input4: tensor<i64>, %input5: tensor<i64>):
///   %1 = stablehlo.compare  GT, %input2, %input3,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
///   stablehlo.return %1 : tensor<i1>
/// }) : (tensor<2x3xi64>, tensor<2x3xi64>) -> (tensor<2x3xi64>, tensor<2x3xi64>)
/// // %result0 = [[3, 2, 3], [1, 2, 1]]
/// // %result1 = [[1, 2, 1], [3, 2, 3]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#sort)
/// for more information.
pub trait SortOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneRegion<'o, 'c, 't> {
    /// Returns the dimension over which this [`SortOperation`] sorts its inputs.
    fn dimension(&self) -> usize {
        self.attribute(SORT_DIMENSION_ATTRIBUTE)
            .and_then(|attribute| {
                attribute.cast::<IntegerAttributeRef<'c, 't>>().map(|attribute| attribute.signless_value() as usize)
            })
            .unwrap_or_else(|| panic!("invalid '{SORT_DIMENSION_ATTRIBUTE}' attribute in `stable_hlo::sort`"))
    }

    /// Returns `true` if the sorting performed by this [`SortOperation`] is stable (i.e., the relative order of
    /// elements considered to be equal by the comparator is preserved).
    fn is_stable(&self) -> bool {
        self.attribute(SORT_IS_STABLE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<BooleanAttributeRef<'c, 't>>().map(|attribute| attribute.value()))
            .unwrap_or_else(|| panic!("invalid '{SORT_IS_STABLE_ATTRIBUTE}' attribute in `stable_hlo::sort`"))
    }

    /// Returns a reference to the [`Region`](crate::Region) that contains the comparator
    /// used by this [`SortOperation`].
    fn comparator(&self) -> RegionRef<'o, 'c, 't> {
        self.body_region()
    }
}

mlir_op!(Sort);
mlir_op_trait!(Sort, OneRegion);
mlir_op_trait!(Sort, ZeroSuccessors);

/// Constructs a new detached/owned [`SortOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`SortOperation`] for more information on the operation semantics and the arguments of this function.
///
/// Note that if any of the inputs to this function are invalid, the function may panic!
pub fn sort<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    inputs: &[V],
    dimension: usize,
    is_stable: bool,
    comparator: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedSortOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.sort", location)
        .add_operands(inputs)
        .add_attribute(
            SORT_DIMENSION_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), dimension as i64),
        )
        .add_attribute(SORT_IS_STABLE_ATTRIBUTE, context.boolean_attribute(is_stable))
        .add_region(comparator)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::sort`")
}

/// Name of the [`Attribute`] that is used to store [`ReverseOperation::reverse_dimensions`].
pub const REVERSE_DIMENSIONS_ATTRIBUTE: &str = "dimensions";

/// StableHLO [`Operation`] that reverses the order of elements in its input tensor along the dimensions specified in
/// [`ReverseOperation::reverse_dimensions`]. More formally, `output[output_index] = input[input_index]`, where:
///
///   - `input_index[d] = dim(output, d) - output_index[d] - 1` if `d` in [`ReverseOperation::reverse_dimensions`], and
///   - `input_index[d] = output_index[d]`, otherwise.
///
/// # Example
///
/// The following is an example of a [`ReverseOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand = [[1, 2], [3, 4], [5, 6]]
/// %result = stablehlo.reverse %operand, dims = [1] : tensor<3x2xi32>
/// // %result: [[2, 1], [4, 3], [6, 5]]
/// ```
///
/// Refer to the [StableHLO specification](https://openxla.org/stablehlo/spec#reverse) for more information.
pub trait ReverseOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the dimensions along which to reverse the elements of the input to this [`ReverseOperation`].
    fn reverse_dimensions(&self) -> Vec<usize> {
        self.attribute(REVERSE_DIMENSIONS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().map(|value| value as usize).collect())
            .unwrap_or_else(|| panic!("invalid '{REVERSE_DIMENSIONS_ATTRIBUTE}' attribute in `stable_hlo::reverse`"))
    }
}

mlir_op!(Reverse);
mlir_op_trait!(Reverse, OneResult);
mlir_op_trait!(Reverse, ZeroRegions);
mlir_op_trait!(Reverse, ZeroSuccessors);

/// Constructs a new detached/owned [`ReverseOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`ReverseOperation`] for more information on the operation semantics and the arguments of this function.
///
/// Note that if any of the inputs to this function are invalid, the function may panic!
pub fn reverse<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    dimensions: &[usize],
    location: L,
) -> DetachedReverseOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.reverse", location)
        .add_operand(input)
        .add_attribute(
            REVERSE_DIMENSIONS_ATTRIBUTE,
            location
                .context()
                .dense_i64_array_attribute(dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice())
                .unwrap(),
        )
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::reverse`")
}

/// StableHLO [`Operation`] that represents a return operation from within a StableHLO function. It is equivalent to
/// [`func::ReturnOperation`](crate::dialects::func::ReturnOperation) except that it is supposed to be used in
/// [`Region`](crate::Region)s that are nested within other StableHLO [`Operation`]s (as opposed to the body of
/// [`FuncOperation`](crate::dialects::func::FuncOperation)s).
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec) for more information and look
/// for instances where the `"stablehlo.return"` [`Operation`]s is referenced in that documentation.
pub trait ReturnOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns an [`Iterator`] over the return [`Value`]s (i.e., the operands) of this [`ReturnOperation`].
    ///
    /// Note that the returned iterator does not hold a borrowed reference to the underlying [`Context`]
    /// because that would make it impossible to perform mutating operations on that context (e.g., from within
    /// [`Pass`](crate::Pass)es) while iterating over the contents of that iterator.
    fn values(&self) -> impl Iterator<Item = ValueRef<'o, 'c, 't>> {
        self.operands()
    }
}

mlir_op!(Return);
mlir_op_trait!(Return, AlwaysSpeculatable);
mlir_op_trait!(Return, MemRefsNormalizable);
mlir_op_trait!(Return, NoMemoryEffect);
mlir_op_trait!(Return, Pure);
mlir_op_trait!(Return, ZeroRegions);

/// Constructs a new detached/owned [`ReturnOperation`] at the specified [`Location`] and with the provided operands.
/// Refer to the documentation of [`ReturnOperation`] for more information.
pub fn r#return<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    values: &[V],
    location: L,
) -> DetachedReturnOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.return", location)
        .add_operands(values)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::return`")
}

/// StableHLO [`Operation`] that ensures that the [`Operation`]s that produce its operands (i.e., inputs) are executed
/// before any [`Operation`]s that depend on its result, preventing any compiler transformations from moving operations
/// across that barrier. Other than that, it acts as an identity function (i.e., its results/outputs are the same as its
/// operands/inputs). This [`Operation`] is useful for controlling compiler optimization behavior (e.g., for timing
/// measurements or debugging).
///
/// # Example
///
/// The following is an example of an [`OptimizationBarrierOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // %operand0: 1.0
/// // %operand1: 2.0
/// %result0, %result1 = stablehlo.optimization_barrier %operand0, %operand1 : tensor<f32>, tensor<f32>
/// // %result0: 1.0
/// // %result1: 2.0
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#optimization_barrier)
/// for more information.
pub trait OptimizationBarrierOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(OptimizationBarrier);
mlir_op_trait!(OptimizationBarrier, ZeroRegions);
mlir_op_trait!(OptimizationBarrier, ZeroSuccessors);

/// Constructs a new detached/owned [`OptimizationBarrierOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`OptimizationBarrierOperation`] for more information on the operation semantics.
pub fn optimization_barrier<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    operands: &[V],
    location: L,
) -> DetachedOptimizationBarrierOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.optimization_barrier", location)
        .add_operands(operands)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::optimization_barrier`")
}

/// Name of the [`Attribute`] that is used to store [`CompositeOperation::composite_name`].
pub const COMPOSITE_NAME_ATTRIBUTE: &str = "name";

/// Name of the [`Attribute`] that is used to store [`CompositeOperation::composite_version`].
pub const COMPOSITE_VERSION_ATTRIBUTE: &str = "version";

/// Name of the [`Attribute`] that is used to store [`CompositeOperation::composite_attributes`].
pub const COMPOSITE_ATTRIBUTES_ATTRIBUTE: &str = "composite_attributes";

/// Name of the [`Attribute`] that is used to store [`CompositeOperation::composite_decomposition`].
pub const COMPOSITE_DECOMPOSITION_ATTRIBUTE: &str = "decomposition";

/// StableHLO [`Operation`] that is composed of other StableHLO operations. You can think of it as a _named_
/// [`func::call`](crate::dialects::func::call) to [`CompositeOperation::composite_decomposition`] which is a reference
/// to a [`func::func`](crate::dialects::func::func). That is, instances of this [`Operation`] can be replaced with its
/// [`CompositeOperation::composite_decomposition`] without changing program semantics. The main difference with normal
/// function calls is that this [`Operation`] is primarily meant to be used for making it easier to pattern match
/// against certain composite operations when implementing accelerator-specific compiler optimizations (e.g., for scaled
/// dot product attention). In cases where inlining [`CompositeOperation::composite_decomposition`] does not provide the
/// same [`Operation`] semantics you should instead use [`custom_call`]. Note that there are also optionally
/// [`Attribute`]s stored in this [`Operation`] under the [`CompositeOperation::composite_attributes`] key. This is
/// meant to support custom metadata that may be used in [`CompositeOperation::composite_decomposition`]. Finally, this
/// operation is _versioned_ in order to enable providing compatibility guarantees. Its version is stored in
/// [`CompositeOperation::composite_version`].
///
/// The number and types of the operands and results of this [`Operation`] match the number and types of the operands
/// and results of its [`CompositeOperation::composite_decomposition`].
///
/// # Example
///
/// The following is an example of a [`CompositeOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// func.func private @my_op(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
///   %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
///   return %0 : tensor<f32>
/// }
///
/// func.func @composite_example(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
///   %0 = stablehlo.composite \"my_namespace.my_op\" %arg0, %arg1 {
///     composite_attributes = {my_op_attribute},
///     decomposition = @my_op,
///     version = 1 : i32
///   } : (tensor<f32>, tensor<f32>) -> tensor<f32>
///   return %0 : tensor<f32>
/// }
/// ```
///
/// Refer to [this video](https://www.youtube.com/watch?v=QEJzPLRhFzg) and to the
/// [official StableHLO specification](https://openxla.org/stablehlo/spec#composite) for more information.
pub trait CompositeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the name of this [`CompositeOperation`] that follows namespaced operation naming conventions.
    fn composite_name(&self) -> StringRef<'c> {
        self.attribute(COMPOSITE_NAME_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<StringAttributeRef>().map(|attribute| attribute.string()))
            .unwrap_or_else(|| panic!("invalid '{COMPOSITE_NAME_ATTRIBUTE}' attribute in `stable_hlo::composite`"))
    }

    /// Returns the optional version of this [`CompositeOperation`]. Typically, version 0 means that a composite
    /// operation is under development and does not imply any compatibility guarantees, whereas higher versions do.
    fn composite_version(&self) -> Option<u64> {
        self.attribute(COMPOSITE_VERSION_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>().map(|attribute| attribute.unsigned_value()))
    }

    /// Returns the composite [`Attribute`]s of this [`CompositeOperation`] that will be propagated to
    /// [`CompositeOperation::composite_decomposition`] when this operation is invoked.
    fn composite_attributes(&self) -> Option<HashMap<StringRef<'c>, AttributeRef<'c, 't>>> {
        self.attribute(COMPOSITE_ATTRIBUTES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DictionaryAttributeRef>().map(|attribute| attribute.into()))
    }

    /// Returns the name/symbol of the decomposition [`func::func`](crate::dialects::func::func)  of this
    /// [`CompositeOperation`]. The referred function must be defined in the parent scope of this operation.
    fn composite_decomposition(&self) -> StringRef<'c> {
        self.attribute(COMPOSITE_DECOMPOSITION_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<FlatSymbolRefAttributeRef>().map(|attribute| attribute.reference()))
            .unwrap_or_else(|| panic!("invalid '{COMPOSITE_DECOMPOSITION_ATTRIBUTE}' attribute in `stable_hlo::composite`"))
    }
}

mlir_op!(Composite);
mlir_op_trait!(Composite, ZeroRegions);
mlir_op_trait!(Composite, ZeroSuccessors);

/// Constructs a new detached/owned [`CompositeOperation`] at the specified [`Location`]. Refer to the documentation
/// of [`CompositeOperation`], [`CompositeOperation::composite_name`], [`CompositeOperation::composite_version`],
/// [`CompositeOperation::composite_attributes`], and [`CompositeOperation::composite_decomposition`], for more
/// information on the operation semantics and the arguments of this function.
///
/// Note that if any of the inputs to this function are invalid, the function may panic!
pub fn composite<
    'v,
    'c: 'v,
    't: 'c,
    's,
    N: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
    A: Attribute<'c, 't>,
    V: Value<'v, 'c, 't>,
    D: IntoWithContext<'c, 't, FlatSymbolRefAttributeRef<'c, 't>>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    name: N,
    version: Option<u64>,
    attributes: Option<&HashMap<StringRef<'s>, A>>,
    operands: &[V],
    decomposition: D,
    result_types: &[T],
    location: L,
) -> DetachedCompositeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    let mut builder = OperationBuilder::new("stablehlo.composite", location)
        .add_operands(operands)
        .add_attribute(COMPOSITE_NAME_ATTRIBUTE, name.into_with_context(context))
        .add_attribute(COMPOSITE_DECOMPOSITION_ATTRIBUTE, decomposition.into_with_context(context));
    if let Some(attributes) = attributes {
        builder = builder.add_attribute(
            COMPOSITE_ATTRIBUTES_ATTRIBUTE,
            DictionaryAttributeRef::from_with_context(attributes, context),
        )
    }
    if let Some(version) = version {
        builder = builder.add_attribute(
            COMPOSITE_VERSION_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(32), version.cast_signed()),
        );
    }
    builder
        .add_results(result_types)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::composite`")
}

/// API version used by a [`CustomCallOperation`]. This determines the format in which the custom operation metadata
/// are specified (i.e., as a [`StringAttributeRef`] or a [`DictionaryAttributeRef`] among other things related to how
/// it should be invoked.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Default)]
pub enum CustomCallApiVersion {
    Unspecified,
    Original,
    StatusReturning,
    StatusReturningUnified,
    #[default]
    TypedFfi,
}


impl<'c, 't> From<IntegerAttributeRef<'c, 't>> for CustomCallApiVersion {
    fn from(value: IntegerAttributeRef<'c, 't>) -> Self {
        match value.signless_value() {
            1 => Self::Original,
            2 => Self::StatusReturning,
            3 => Self::StatusReturningUnified,
            4 => Self::TypedFfi,
            _ => Self::Unspecified,
        }
    }
}

impl<'c, 't> FromWithContext<'c, 't, CustomCallApiVersion> for IntegerAttributeRef<'c, 't> {
    fn from_with_context(value: CustomCallApiVersion, context: &'c Context<'t>) -> Self {
        let r#type = context.signless_integer_type(32);
        match value {
            CustomCallApiVersion::Unspecified => context.integer_attribute(r#type, 0),
            CustomCallApiVersion::Original => context.integer_attribute(r#type, 1),
            CustomCallApiVersion::StatusReturning => context.integer_attribute(r#type, 2),
            CustomCallApiVersion::StatusReturningUnified => context.integer_attribute(r#type, 3),
            CustomCallApiVersion::TypedFfi => context.integer_attribute(r#type, 4),
        }
    }
}

/// Memory layouts of the operands and the results of a [`CustomCallOperation`]. A memory layout for a tensor is
/// specified as a [`Vec`] that contains the indices of its dimensions ranked in minor-to-major order. For example,
/// consider a three-dimensional tensor with shape `[2, 3, 2]` and layout `[2, 0, 1]`, where we denote the element at
/// position `[i, j, k]` as `x_ijk`. In this case, the tensor will be represented as follows in memory:
///
/// ```text
/// [x_000, x_001, x_100, x_101, x_010, x_011, x_110, x_111, x_020, x_021, x_120, x_121]
/// ```
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CustomCallMemoryLayouts {
    /// Memory layouts of the operands/inputs of the [`CustomCallOperation`]. The length of this vector must match
    /// the number of operands of the corresponding operation.
    pub operands: Vec<Vec<usize>>,

    /// Memory layouts of the results/outputs of the [`CustomCallOperation`]. The length of this vector must match
    /// the number of results of the corresponding operation.
    pub results: Vec<Vec<usize>>,
}

/// StableHLO [`Attribute`] that models the alias relationship between outputs and operands in [`CustomCallOperation`]s.
///
/// This attribute captures the alias relationship of outputs to operands for [`CustomCallOperation`]s. Specifically,
/// for a specific output, it captures an aliasing relationship with the operand denoted by
/// [`OutputOperandAliasAttributeRef::operand_index`]. [`OutputOperandAliasAttributeRef::output_tuple_indices`] and
/// [`OutputOperandAliasAttributeRef::operand_tuple_indices`] are used to index into the [`Operation`] output and
/// operand types. These index lists are empty if the corresponding types are not tuple types, and can be arbitrarily
/// long in the case of arbitrarily nested tuple types.
///
/// # Example
///
/// The following is an example for the use of this attribute in an MLIR program:
///
/// ```mlir
/// %0 = "stablehlo.custom_call"(%arg0, %arg1) {
///   // Other attributes...
///   output_operand_alias = [
///     #stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = [1]>
///   ]
/// } : (tuple<tensor<1x1xf32>, tensor<2x3xf32>>, tensor<5x5xf32>) -> tuple<tensor<2x3xf32>>
/// ```
///
/// In this example, the operation output and the `0`-th (i.e., first) operand are both tuples. The alias attribute
/// shows the relationship between the `0`-th element in the output tuple and the `1`-st element in the `0`-th operand.
/// Note that both of these elements have the same [`Type`] (i.e., `tensor<2x3xf32>`).
///
/// Refer to the [official XLA documentation](https://www.tensorflow.org/xla/aliasing) for more information.
#[derive(Copy, Clone)]
pub struct OutputOperandAliasAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> OutputOperandAliasAttributeRef<'c, 't> {
    mlir_attribute_field!(
        output_tuple_indices,
        OutputOperandAliasGetOutputTupleIndices,
        [usize],
        mlir_prefix = stablehlo,
    );

    mlir_attribute_field!(operand_index, OutputOperandAliasGetOperandIndex, i64, mlir_prefix = stablehlo);

    mlir_attribute_field!(
        operand_tuple_indices,
        OutputOperandAliasGetOperandTupleIndices,
        [usize],
        mlir_prefix = stablehlo,
    );
}

mlir_subtype_trait_impls!(
    OutputOperandAliasAttributeRef<'c, 't> as Attribute,
    mlir_type = Attribute,
    mlir_subtype = OutputOperandAlias,
    mlir_prefix = stablehlo,
);

impl<'t> Context<'t> {
    /// Creates a new StableHLO [`OutputOperandAliasAttributeRef`] owned by this [`Context`].
    pub fn stable_hlo_output_operand_alias<'c>(
        &'c self,
        output_tuple_indices: &[usize],
        operand_index: usize,
        operand_tuple_indices: &[usize],
    ) -> OutputOperandAliasAttributeRef<'c, 't> {
        // Make sure that the StableHLO dialect is loaded into the current context to prevent segmentation faults.
        self.load_dialect(DialectHandle::stable_hlo());
        let output_tuple_indices = output_tuple_indices.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let operand_tuple_indices = operand_tuple_indices.iter().map(|v| *v as i64).collect::<Vec<_>>();
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            OutputOperandAliasAttributeRef::from_c_api(
                stablehloOutputOperandAliasGet(
                    *self.handle.borrow(),
                    output_tuple_indices.len().cast_signed(),
                    output_tuple_indices.as_ptr(),
                    operand_index as i64,
                    operand_tuple_indices.len().cast_signed(),
                    operand_tuple_indices.as_ptr(),
                ),
                self,
            )
            .unwrap()
        }
    }
}

/// Name of the [`Attribute`] that is used to store [`CustomCallOperation::custom_call_target_name`].
pub const CUSTOM_CALL_TARGET_NAME_ATTRIBUTE: &str = "call_target_name";

/// Name of the [`Attribute`] that is used to store [`CustomCallOperation::custom_call_has_side_effect`].
pub const CUSTOM_CALL_HAS_SIDE_EFFECT_ATTRIBUTE: &str = "has_side_effect";

/// Name of the [`Attribute`] that is used to store [`CustomCallOperation::custom_call_backend_config`].
pub const CUSTOM_CALL_BACKEND_CONFIG_ATTRIBUTE: &str = "backend_config";

/// Name of the [`Attribute`] that is used to store [`CustomCallOperation::custom_call_api_version`].
pub const CUSTOM_CALL_API_VERSION_ATTRIBUTE: &str = "api_version";

/// Name of the [`Attribute`] that is used to store [`CustomCallOperation::custom_call_called_computations`].
pub const CUSTOM_CALL_CALLED_COMPUTATIONS_ATTRIBUTE: &str = "called_computations";

/// Name of the [`Attribute`] that is used to store part of [`CustomCallOperation::custom_call_memory_layouts`].
pub const CUSTOM_CALL_OPERAND_LAYOUTS_ATTRIBUTE: &str = "operand_layouts";

/// Name of the [`Attribute`] that is used to store part of [`CustomCallOperation::custom_call_memory_layouts`].
pub const CUSTOM_CALL_RESULT_LAYOUTS_ATTRIBUTE: &str = "result_layouts";

/// Name of the [`Attribute`] that is used to store [`CustomCallOperation::custom_call_output_operand_aliases`].
pub const CUSTOM_CALL_OUTPUT_OPERAND_ALIASES_ATTRIBUTE: &str = "output_operand_aliases";

/// [`CustomCallOperation::custom_call_target_name`] for the XLA GPU custom call that creates an uninitialized `memref`.
pub const XLA_GPU_CREATE_BUFFER_CUSTOM_CALL_TARGET_NAME: &str = "CreateBuffer";

/// [`CustomCallOperation::custom_call_target_name`] for the XLA GPU custom call that creates an initialized `memref`
/// from a `tensor`.
pub const XLA_GPU_PIN_CUSTOM_CALL_TARGET_NAME: &str = "Pin";

/// [`CustomCallOperation::custom_call_target_name`] for the XLA GPU custom call that deallocates a `memref`
/// and returns a `tensor`.
pub const XLA_GPU_UNPIN_CUSTOM_CALL_TARGET_NAME: &str = "Unpin";

/// StableHLO [`Operation`] that encapsulates a call to a custom implementation called
/// [`CustomCallOperation::custom_call_target_name`]. This operation provides a mechanism for invoking operations that
/// are not part of the standard StableHLO operation set. This enables integration with external libraries, custom
/// kernels, and platform-specific optimizations.
///
/// The semantics of [`CustomCallOperation`] are entirely implementation-specific and are determined by the
/// implementation referenced by [`CustomCallOperation::custom_call_target_name`]. Different implementations may define
/// their custom semantics and behaviors. Note that this operation supports both side-effect-free and side-effecting
/// implementations as determined by [`CustomCallOperation::custom_call_has_side_effect`].
///
/// Optionally, a [`CustomCallOperation`] may also specify the memory layout of its operands/inputs and results/outputs
/// via [`CustomCallOperation::custom_call_memory_layouts`].
///
/// ## Special XLA GPU Target Names
///
/// XLA GPU defines three special [`CustomCallOperation::custom_call_target_name`]s for buffer operations:
///
/// - `CreateBuffer`: Creates an uninitialized `memref` buffer.
/// - `Pin`: Converts a `tensor` to an initialized `memref` buffer.
/// - `Unpin`: Deallocates a `memref` buffer and returns its contents as a `tensor`.
///
/// # Examples
///
/// The following are examples of [`CustomCallOperation`]s represented using their
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = stablehlo.custom_call @my_custom_op(%arg0) {
///   backend_config = {},
///   api_version = 4 : i32,
///   called_computations = [@helper_fn],
/// } : (tensor<f32>) -> tensor<f32>
///
/// %uninitialized_buffer = stablehlo.custom_call @CreateBuffer() {
///   api_version = 4 : i32,
/// } : () -> memref<4xf64>
///
/// %initialized_buffer = stablehlo.custom_call @Pin(%init_value) {
///   api_version = 4 : i32,
/// } : (tensor<4xf64>) -> memref<4xf64>
///
/// %dealloc_buffer = stablehlo.custom_call @Unpin(%initialized_buffer) {
///   api_version = 4 : i32,
/// } : (memref<4xf64>) -> tensor<4xf64>
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#custom_call) and the
/// [official XLA documentation](https://openxla.org/xla/custom_call) for more information.
pub trait CustomCallOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the name of the target implementation of this [`CustomCallOperation`].
    fn custom_call_target_name(&self) -> StringRef<'c> {
        self.attribute(CUSTOM_CALL_TARGET_NAME_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<StringAttributeRef>().map(|attribute| attribute.string()))
            .unwrap_or_else(|| panic!("invalid '{CUSTOM_CALL_TARGET_NAME_ATTRIBUTE}' attribute in `stable_hlo::custom_call`"))
    }

    /// Returns `true` if executing this [`CustomCallOperation`] can result in side effects
    /// (i.e., if this operation is not _pure_).
    fn custom_call_has_side_effect(&self) -> bool {
        self.attribute(CUSTOM_CALL_HAS_SIDE_EFFECT_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<BooleanAttributeRef>().map(|attribute| attribute.value()))
            .unwrap_or_else(|| panic!("invalid '{CUSTOM_CALL_HAS_SIDE_EFFECT_ATTRIBUTE}' attribute in `stable_hlo::custom_call`"))
    }

    /// Returns the backend configuration of this [`CustomCallOperation`]. This is either a [`StringAttributeRef`]
    /// (when [`CustomCallOperation::custom_call_api_version`] is not [`CustomCallApiVersion::TypedFfi`])
    /// or a [`DictionaryAttributeRef`] (when [`CustomCallOperation::custom_call_api_version`] is
    /// [`CustomCallApiVersion::TypedFfi`]) that contains implementation-specific metadata.
    fn custom_call_backend_config(&self) -> AttributeRef<'c, 't> {
        self.attribute(CUSTOM_CALL_BACKEND_CONFIG_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<AttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{CUSTOM_CALL_BACKEND_CONFIG_ATTRIBUTE}' attribute in `stable_hlo::custom_call`"))
    }

    /// Returns the [`CustomCallApiVersion`] of this [`CustomCallOperation`].
    fn custom_call_api_version(&self) -> CustomCallApiVersion {
        self.attribute(CUSTOM_CALL_API_VERSION_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>().map(|attribute| attribute.into()))
            .unwrap_or_else(|| panic!("invalid '{CUSTOM_CALL_API_VERSION_ATTRIBUTE}' attribute in `stable_hlo::custom_call`"))
    }

    /// Returns the names/symbols of functions that are used by this [`CustomCallOperation`].
    fn custom_call_called_computations(&self) -> Vec<StringRef<'c>> {
        let error_message =
            format!("invalid '{CUSTOM_CALL_CALLED_COMPUTATIONS_ATTRIBUTE}' attribute in `stable_hlo::custom_call`");
        self.attribute(CUSTOM_CALL_CALLED_COMPUTATIONS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<ArrayAttributeRef>())
            .expect(&error_message)
            .elements()
            .map(|attribute| attribute.cast::<FlatSymbolRefAttributeRef>().expect(&error_message).reference())
            .collect()
    }

    /// Returns the optional memory layout information for the operands/inputs and the results/outputs of this
    /// [`CustomCallOperation`]. Refer to the documentation of [`CustomCallMemoryLayouts`] for information on the
    /// semantics of these memory layouts.
    fn custom_call_memory_layouts(&self) -> Option<CustomCallMemoryLayouts> {
        let error_message =
            format!("invalid '{CUSTOM_CALL_OPERAND_LAYOUTS_ATTRIBUTE}' attribute in `stable_hlo::custom_call`");
        self.attribute(CUSTOM_CALL_OPERAND_LAYOUTS_ATTRIBUTE)
            .expect(&error_message)
            .cast::<ArrayAttributeRef>()
            .zip(
                self.attribute(CUSTOM_CALL_RESULT_LAYOUTS_ATTRIBUTE)
                    .expect(&error_message)
                    .cast::<ArrayAttributeRef>(),
            )
            .map(|(operands, results)| CustomCallMemoryLayouts {
                operands: operands
                    .elements()
                    .map(|attribute| unsafe {
                        attribute
                            .cast::<DenseIntegerElementsAttributeRef<'c, 't>>()
                            .expect(&error_message)
                            .usize_elements()
                            .collect()
                    })
                    .collect(),
                results: results
                    .elements()
                    .map(|attribute| unsafe {
                        attribute
                            .cast::<DenseIntegerElementsAttributeRef<'c, 't>>()
                            .expect(&error_message)
                            .usize_elements()
                            .collect()
                    })
                    .collect(),
            })
    }

    /// Returns the alias relationship between outputs and operands of this [`CustomCallOperation`].
    fn custom_call_output_operand_aliases(&self) -> Vec<OutputOperandAliasAttributeRef<'c, 't>> {
        let error_message =
            format!("invalid '{CUSTOM_CALL_OUTPUT_OPERAND_ALIASES_ATTRIBUTE}' attribute in `stable_hlo::custom_call`");
        self.attribute(CUSTOM_CALL_OUTPUT_OPERAND_ALIASES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<ArrayAttributeRef>())
            .expect(&error_message)
            .elements()
            .map(|attribute| attribute.cast::<OutputOperandAliasAttributeRef>().expect(&error_message))
            .collect()
    }
}

mlir_op!(CustomCall);
mlir_op_trait!(CustomCall, ZeroRegions);
mlir_op_trait!(CustomCall, ZeroSuccessors);

/// Constructs a new detached/owned [`CustomCallOperation`] at the specified [`Location`]. Refer to the documentation
/// of [`CustomCallOperation`], [`CustomCallOperation::custom_call_target_name`],
/// [`CustomCallOperation::custom_call_has_side_effect`], [`CustomCallOperation::custom_call_backend_config`],
/// [`CustomCallOperation::custom_call_api_version`], [`CustomCallOperation::custom_call_called_computations`],
/// [`CustomCallOperation::custom_call_memory_layouts`], and [`CustomCallOperation::custom_call_output_operand_aliases`]
/// for more information on the operation semantics and the arguments of this function.
///
/// Note that if any of the inputs to this function are invalid, the function may panic!
#[allow(clippy::too_many_arguments)]
pub fn custom_call<
    'v,
    'c: 'v,
    't: 'c,
    V: Value<'v, 'c, 't>,
    N: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    inputs: &[V],
    target_name: N,
    has_side_effect: bool,
    backend_config: Option<AttributeRef<'c, 't>>,
    api_version: CustomCallApiVersion,
    called_computations: &[FlatSymbolRefAttributeRef<'c, 't>],
    memory_layouts: Option<CustomCallMemoryLayouts>,
    output_operand_aliases: &[OutputOperandAliasAttributeRef<'c, 't>],
    output_types: &[T],
    location: L,
) -> DetachedCustomCallOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    let mut builder = OperationBuilder::new("stablehlo.custom_call", location)
        .add_operands(inputs)
        .add_attribute(CUSTOM_CALL_TARGET_NAME_ATTRIBUTE, target_name.into_with_context(context))
        .add_attribute(CUSTOM_CALL_HAS_SIDE_EFFECT_ATTRIBUTE, context.boolean_attribute(has_side_effect));

    if let Some(backend_config) = backend_config {
        builder = builder.add_attribute(CUSTOM_CALL_BACKEND_CONFIG_ATTRIBUTE, backend_config);
    }

    if let Some(memory_layouts) = memory_layouts {
        builder = builder
            .add_attribute(
                CUSTOM_CALL_OPERAND_LAYOUTS_ATTRIBUTE,
                context.array_attribute(
                    &memory_layouts
                        .operands
                        .iter()
                        .map(|layout| DenseIntegerElementsAttributeRef::from_with_context(layout.as_slice(), context))
                        .collect::<Vec<_>>(),
                ),
            )
            .add_attribute(
                CUSTOM_CALL_RESULT_LAYOUTS_ATTRIBUTE,
                context.array_attribute(
                    &memory_layouts
                        .results
                        .iter()
                        .map(|layout| DenseIntegerElementsAttributeRef::from_with_context(layout.as_slice(), context))
                        .collect::<Vec<_>>(),
                ),
            );
    }

    builder
        .add_attribute(CUSTOM_CALL_API_VERSION_ATTRIBUTE, IntegerAttributeRef::from_with_context(api_version, context))
        .add_attribute(CUSTOM_CALL_CALLED_COMPUTATIONS_ATTRIBUTE, context.array_attribute(called_computations))
        .add_attribute(CUSTOM_CALL_OUTPUT_OPERAND_ALIASES_ATTRIBUTE, context.array_attribute(output_operand_aliases))
        .add_results(output_types)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::custom_call`")
}

/// Constructs a new detached/owned [`CustomCallOperation`] for the XLA GPU `CreateBuffer` built-in target.
/// In StableHLO's buffer model, this target creates an uninitialized `memref` buffer.
///
/// Refer to the [StableHLO buffer RFC](
/// https://github.com/openxla/stablehlo/blob/main/rfcs/20250729-buffer.md#xla-gpu-support-special-custom_call-targets)
/// for more information.
pub fn gpu_create_buffer_custom_call<'c, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    output_type: T,
    location: L,
) -> DetachedCustomCallOperation<'c, 't> {
    custom_call::<ValueRef, _, _, _>(
        &[],
        XLA_GPU_CREATE_BUFFER_CUSTOM_CALL_TARGET_NAME,
        false,
        None,
        CustomCallApiVersion::TypedFfi,
        &[],
        None,
        &[],
        &[output_type],
        location,
    )
}

/// Constructs a new detached/owned [`CustomCallOperation`] for the XLA GPU `Pin` built-in target.
/// In StableHLO's buffer model, this target creates an initialized `memref` buffer from a tensor value.
///
/// Refer to the [StableHLO buffer RFC](
/// https://github.com/openxla/stablehlo/blob/main/rfcs/20250729-buffer.md#xla-gpu-support-special-custom_call-targets)
/// for more information.
pub fn gpu_pin_custom_call<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    input: V,
    output_type: T,
    location: L,
) -> DetachedCustomCallOperation<'c, 't> {
    custom_call(
        &[input],
        XLA_GPU_PIN_CUSTOM_CALL_TARGET_NAME,
        false,
        None,
        CustomCallApiVersion::TypedFfi,
        &[],
        None,
        &[],
        &[output_type],
        location,
    )
}

/// Constructs a new detached/owned [`CustomCallOperation`] for the XLA GPU `Unpin` built-in target.
/// In StableHLO's buffer model, this target deallocates a `memref` buffer and returns its contents as a tensor.
///
/// Refer to the [StableHLO buffer RFC](
/// https://github.com/openxla/stablehlo/blob/main/rfcs/20250729-buffer.md#xla-gpu-support-special-custom_call-targets)
/// for more information.
pub fn gpu_unpin_custom_call<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    input: V,
    output_type: T,
    location: L,
) -> DetachedCustomCallOperation<'c, 't> {
    custom_call(
        &[input],
        XLA_GPU_UNPIN_CUSTOM_CALL_TARGET_NAME,
        false,
        None,
        CustomCallApiVersion::TypedFfi,
        &[],
        None,
        &[],
        &[output_type],
        location,
    )
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};
    use crate::dialects::{func, stable_hlo};
    use crate::{Attribute, Block, Context, Operation, Region, Size, StringRef, SymbolVisibility, Value};

    use super::{
        CompositeOperation, CustomCallApiVersion, CustomCallMemoryLayouts, CustomCallOperation, DynamicIotaOperation,
        IotaOperation, ReverseOperation, SortOperation, XLA_GPU_CREATE_BUFFER_CUSTOM_CALL_TARGET_NAME,
        XLA_GPU_PIN_CUSTOM_CALL_TARGET_NAME, XLA_GPU_UNPIN_CUSTOM_CALL_TARGET_NAME, composite, constant, custom_call,
        dynamic_iota, gpu_create_buffer_custom_call, gpu_pin_custom_call, gpu_unpin_custom_call, iota,
        optimization_barrier, r#return, reverse, sort,
    };

    #[test]
    fn test_constant() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let tensor_type = context.tensor_type(i64_type, &[Size::Static(2), Size::Static(4)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = constant(
                context.dense_i64_elements_attribute(tensor_type, &[0, 1, 2, 3, 4, 5, 6, 7]).unwrap(),
                location,
            );
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "constant_test",
                func::FuncAttributes { arguments: vec![], results: vec![tensor_type.into()], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @constant_test() -> tensor<2x4xi64> {
                    %c = stablehlo.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>
                    return %c : tensor<2x4xi64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_iota() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(4), Size::Static(5)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = iota(tensor_type, 1, location);
            assert_eq!(op.iota_dimension(), 1);
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.result(0).unwrap().r#type(), tensor_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "iota_test",
                func::FuncAttributes { arguments: vec![], results: vec![tensor_type.into()], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @iota_test() -> tensor<4x5xi32> {
                    %0 = stablehlo.iota dim = 1 : tensor<4x5xi32>
                    return %0 : tensor<4x5xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_dynamic_iota() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let shape_tensor_type = context.tensor_type(i64_type, &[Size::Static(2)], None, location).unwrap();
        let tensor_type = context.tensor_type(i64_type, &[Size::Dynamic, Size::Dynamic], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(shape_tensor_type, location)]);
            let op = dynamic_iota(block.argument(0).unwrap(), tensor_type, 1, location);
            assert_eq!(op.iota_dimension(), 1);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "dynamic_iota_test",
                func::FuncAttributes {
                    arguments: vec![shape_tensor_type.into()],
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
                  func.func @dynamic_iota_test(%arg0: tensor<2xi64>) -> tensor<?x?xi64> {
                    %0 = stablehlo.dynamic_iota %arg0, dim = 1 : (tensor<2xi64>) -> tensor<?x?xi64>
                    return %0 : tensor<?x?xi64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_sort() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let input_type = context.tensor_type(i64_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        let scalar_i64_type = context.tensor_type(i64_type, &[], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_type, location), (input_type, location)]);
            let mut comparator_region = context.region();
            let mut comparator_block = context.block(&[
                (scalar_i64_type, location),
                (scalar_i64_type, location),
                (scalar_i64_type, location),
                (scalar_i64_type, location),
            ]);
            let compare_op = comparator_block.append_operation(stable_hlo::compare(
                comparator_block.argument(0).unwrap(),
                comparator_block.argument(1).unwrap(),
                stable_hlo::ComparisonDirection::GreaterThan,
                stable_hlo::ComparisonType::Signed,
                location,
            ));
            comparator_block.append_operation(r#return(&[compare_op.result(0).unwrap()], location));
            comparator_region.append_block(comparator_block);
            let op = sort(&block.arguments().collect::<Vec<_>>(), 1, true, comparator_region, location);
            assert_eq!(op.dimension(), 1);
            assert_eq!(op.is_stable(), true);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 2);
            assert_eq!(op.regions().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&op.results().collect::<Vec<_>>(), location));
            func::func(
                "sort_test",
                func::FuncAttributes {
                    arguments: vec![input_type.into(), input_type.into()],
                    results: vec![input_type.into(), input_type.into()],
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
                  func.func @sort_test(%arg0: tensor<2x2xi64>, %arg1: tensor<2x2xi64>) -> (tensor<2x2xi64>, tensor<2x2xi64>) {
                    %0:2 = \"stablehlo.sort\"(%arg0, %arg1) <{dimension = 1 : i64, is_stable = true}> ({
                    ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>, %arg4: tensor<i64>, %arg5: tensor<i64>):
                      %1 = stablehlo.compare  GT, %arg2, %arg3,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %1 : tensor<i1>
                    }) : (tensor<2x2xi64>, tensor<2x2xi64>) -> (tensor<2x2xi64>, tensor<2x2xi64>)
                    return %0#0, %0#1 : tensor<2x2xi64>, tensor<2x2xi64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_reverse() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let input_tensor_type = context
            .tensor_type(i32_type, &[Size::Static(3), Size::Static(4), Size::Static(5)], None, location)
            .unwrap();
        let output_tensor_type = context
            .tensor_type(i32_type, &[Size::Static(3), Size::Static(4), Size::Static(5)], None, location)
            .unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_tensor_type, location)]);
            let input = block.argument(0).unwrap();
            let op = reverse(input, &[0, 2], location);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.reverse_dimensions(), vec![0, 2]);
            assert_eq!(op.result(0).unwrap().r#type(), output_tensor_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "reverse_test",
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
                  func.func @reverse_test(%arg0: tensor<3x4x5xi32>) -> tensor<3x4x5xi32> {
                    %0 = stablehlo.reverse %arg0, dims = [0, 2] : tensor<3x4x5xi32>
                    return %0 : tensor<3x4x5xi32>
                  }
                }
            "}
        );
    }

    #[allow(deprecated)]
    #[test]
    fn test_return() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        let scalar_tensor_type = context.tensor_type(f32_type, &[], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let mut map_region = context.region();
            let mut map_block = context.block(&[(scalar_tensor_type, location)]);
            let input = map_block.argument(0).unwrap();
            let negate_op = stable_hlo::negate(input, location);
            let negate_op = map_block.append_operation(negate_op);
            let return_op = r#return(&[negate_op.result(0).unwrap()], location);
            assert_eq!(return_op.operands().count(), 1);
            assert_eq!(return_op.results().count(), 0);
            assert_eq!(return_op.regions().count(), 0);
            map_block.append_operation(return_op);
            map_region.append_block(map_block);
            let map_op = stable_hlo::map(&[block.argument(0).unwrap()], &[0, 1], map_region.into(), location);
            let map_op = block.append_operation(map_op);
            block.append_operation(func::r#return(&[map_op.result(0).unwrap()], location));
            func::func(
                "return_test",
                func::FuncAttributes {
                    arguments: vec![tensor_type.into()],
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
                  func.func @return_test(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
                    %0 = \"stablehlo.map\"(%arg0) <{dimensions = array<i64: 0, 1>}> ({
                    ^bb0(%arg1: tensor<f32>):
                      %1 = stablehlo.negate %arg1 : tensor<f32>
                      stablehlo.return %1 : tensor<f32>
                    }) : (tensor<2x2xf32>) -> tensor<2x2xf32>
                    return %0 : tensor<2x2xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_optimization_barrier() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let tensor_type = context.tensor_type(context.float32_type(), &[], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location), (tensor_type, location)]);
            let op = optimization_barrier(&block.arguments().collect::<Vec<_>>(), location);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 2);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&op.results().collect::<Vec<_>>(), location));
            func::func(
                "optimization_barrier_test",
                func::FuncAttributes {
                    arguments: vec![tensor_type.into(), tensor_type.into()],
                    results: vec![tensor_type.into(), tensor_type.into()],
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
                  func.func @optimization_barrier_test(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
                    %0:2 = stablehlo.optimization_barrier %arg0, %arg1 : tensor<f32>, tensor<f32>
                    return %0#0, %0#1 : tensor<f32>, tensor<f32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_composite() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[], None, location).unwrap();
        module.body().append_operation({
            let mut decomposition = context.block(&[(tensor_type, location), (tensor_type, location)]);
            let op = stable_hlo::add(decomposition.argument(0).unwrap(), decomposition.argument(1).unwrap(), location);
            let op = decomposition.append_operation(op);
            decomposition.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "my_op",
                func::FuncAttributes {
                    arguments: vec![tensor_type.into(), tensor_type.into()],
                    results: vec![tensor_type.into()],
                    visibility: SymbolVisibility::Private,
                    ..Default::default()
                },
                decomposition.into(),
                location,
            )
        });
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location), (tensor_type, location)]);
            let composite_attributes =
                HashMap::from([(StringRef::from("my_op_attribute"), context.unit_attribute().as_ref())]);
            let composite_op = composite(
                "my_namespace.my_op",
                Some(1),
                Some(&composite_attributes),
                &block.arguments().collect::<Vec<_>>(),
                "my_op",
                &[tensor_type],
                location,
            );
            assert_eq!(composite_op.composite_name().as_str().unwrap(), "my_namespace.my_op");
            assert_eq!(composite_op.composite_version(), Some(1));
            assert_eq!(composite_op.composite_attributes(), Some(composite_attributes));
            assert_eq!(composite_op.composite_decomposition().as_str().unwrap(), "my_op");
            assert_eq!(composite_op.operands().count(), 2);
            assert_eq!(composite_op.results().count(), 1);
            let composite_op = block.append_operation(composite_op);
            block.append_operation(func::r#return(&[composite_op.result(0).unwrap()], location));
            func::func(
                "composite_test",
                func::FuncAttributes {
                    arguments: vec![tensor_type.into(), tensor_type.into()],
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
                  func.func private @my_op(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
                    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
                    return %0 : tensor<f32>
                  }
                  func.func @composite_test(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
                    %0 = stablehlo.composite \"my_namespace.my_op\" %arg0, %arg1 {\
                      composite_attributes = {my_op_attribute}, \
                      decomposition = @my_op, \
                      version = 1 : i32\
                    } : (tensor<f32>, tensor<f32>) -> tensor<f32>
                    return %0 : tensor<f32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_output_operand_alias_attribute() {
        let context = Context::new();
        let attribute = context.stable_hlo_output_operand_alias(&[0, 1], 2, &[3, 4]);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.output_tuple_indices(), vec![0, 1]);
        assert_eq!(attribute.operand_index(), 2);
        assert_eq!(attribute.operand_tuple_indices(), vec![3, 4]);
    }

    #[test]
    fn test_output_operand_alias_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_output_operand_alias(&[0, 1], 2, &[3, 4]);
        let attribute_2 = context.stable_hlo_output_operand_alias(&[0, 1], 2, &[3, 4]);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.stable_hlo_output_operand_alias(&[1, 0], 2, &[3, 4]);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.stable_hlo_output_operand_alias(&[0, 1], 2, &[3, 4]);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_output_operand_alias_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.stable_hlo_output_operand_alias(&[0, 1], 2, &[3, 4]);
        test_attribute_display_and_debug(
            attribute,
            "#stablehlo.output_operand_alias<\
              output_tuple_indices = [0, 1], \
              operand_index = 2, \
              operand_tuple_indices = [3, 4]\
            >",
        );
    }

    #[test]
    fn test_output_operand_alias_attribute_casting() {
        let context = Context::new();
        let attribute = context.stable_hlo_output_operand_alias(&[0, 1], 2, &[3, 4]);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_custom_call() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(4), Size::Static(2)], None, location).unwrap();
        let memref_type =
            context.mem_ref_type(f32_type, &[Size::Static(4), Size::Static(2)], None, None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location), (tensor_type, location)]);
            let backend_config = context.string_attribute("status_returning_attribute");
            let op = custom_call(
                &block.arguments().collect::<Vec<_>>(),
                "my_custom_op",
                true,
                Some(backend_config.as_ref()),
                CustomCallApiVersion::StatusReturning,
                &[context.flat_symbol_ref_attribute("add_0"), context.flat_symbol_ref_attribute("add_1")],
                Some(CustomCallMemoryLayouts { operands: vec![vec![0, 1], vec![1, 0]], results: vec![vec![1, 0]] }),
                &[
                    context.stable_hlo_output_operand_alias(&[], 1, &[]),
                    context.stable_hlo_output_operand_alias(&[], 0, &[]),
                ],
                &[tensor_type],
                location,
            );
            assert_eq!(op.custom_call_target_name().as_str().unwrap(), "my_custom_op");
            assert!(op.custom_call_has_side_effect());
            assert_eq!(op.custom_call_backend_config(), backend_config);
            assert_eq!(op.custom_call_api_version(), CustomCallApiVersion::StatusReturning);
            assert_eq!(
                op.custom_call_called_computations()
                    .iter()
                    .map(|string_ref| string_ref.as_str().unwrap())
                    .collect::<Vec<_>>(),
                ["add_0", "add_1"],
            );
            assert_eq!(
                op.custom_call_memory_layouts(),
                Some(CustomCallMemoryLayouts { operands: vec![vec![0, 1], vec![1, 0]], results: vec![vec![1, 0]] })
            );
            assert_eq!(op.custom_call_output_operand_aliases().len(), 2);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.regions().count(), 0);
            let op = block.append_operation(op);

            // Add a couple more custom calls testing the XLA GPU built-in constructors.
            let create_buffer_op = gpu_create_buffer_custom_call(memref_type, location);
            assert_eq!(
                create_buffer_op.custom_call_target_name().as_str().unwrap(),
                XLA_GPU_CREATE_BUFFER_CUSTOM_CALL_TARGET_NAME,
            );
            assert_eq!(create_buffer_op.custom_call_api_version(), CustomCallApiVersion::TypedFfi);
            block.append_operation(create_buffer_op);

            let pin_op = gpu_pin_custom_call(op.result(0).unwrap(), memref_type, location);
            assert_eq!(pin_op.custom_call_target_name().as_str().unwrap(), XLA_GPU_PIN_CUSTOM_CALL_TARGET_NAME);
            assert_eq!(pin_op.custom_call_api_version(), CustomCallApiVersion::TypedFfi);
            let pin_op = block.append_operation(pin_op);

            let unpin_op = gpu_unpin_custom_call(pin_op.result(0).unwrap(), tensor_type, location);
            assert_eq!(unpin_op.custom_call_target_name().as_str().unwrap(), XLA_GPU_UNPIN_CUSTOM_CALL_TARGET_NAME);
            assert_eq!(unpin_op.custom_call_api_version(), CustomCallApiVersion::TypedFfi);
            let unpin_op = block.append_operation(unpin_op);

            block.append_operation(func::r#return(&[unpin_op.result(0).unwrap()], location));
            func::func(
                "custom_call_test",
                func::FuncAttributes {
                    arguments: vec![tensor_type.into(), tensor_type.into()],
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
                  func.func @custom_call_test(%arg0: tensor<4x2xf32>, %arg1: tensor<4x2xf32>) -> tensor<4x2xf32> {
                    %0 = stablehlo.custom_call @my_custom_op(%arg0, %arg1) {\
                      api_version = 2 : i32, \
                      backend_config = \"status_returning_attribute\", \
                      called_computations = [@add_0, @add_1], \
                      has_side_effect = true, \
                      operand_layouts = [dense<[0, 1]> : vector<2xindex>, dense<[1, 0]> : vector<2xindex>], \
                      output_operand_aliases = [\
                        #stablehlo.output_operand_alias<\
                          output_tuple_indices = [], \
                          operand_index = 1, \
                          operand_tuple_indices = []\
                        >, \
                        #stablehlo.output_operand_alias<\
                          output_tuple_indices = [], \
                          operand_index = 0, \
                          operand_tuple_indices = []\
                        >\
                      ], \
                      result_layouts = [dense<[1, 0]> : vector<2xindex>]\
                    } : (tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
                    %1 = stablehlo.custom_call @CreateBuffer() {api_version = 4 : i32} : () -> memref<4x2xf32>
                    %2 = stablehlo.custom_call @Pin(%0) {api_version = 4 : i32} : (tensor<4x2xf32>) -> memref<4x2xf32>
                    %3 = stablehlo.custom_call @Unpin(%2) {api_version = 4 : i32} : (memref<4x2xf32>) -> tensor<4x2xf32>
                    return %3 : tensor<4x2xf32>
                  }
                }
            "},
        );
    }
}
