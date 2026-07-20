use std::collections::HashMap;

use ryft_xla_sys::bindings::{MlirAttribute, stablehloOutputOperandAliasGet};

use crate::{
    ArrayAttributeRef, Attribute, AttributeRef, Context, DenseIntegerElementsAttributeRef, DetachedOp, DetachedRegion,
    DialectHandle, DictionaryAttributeRef, ElementsAttribute, ElementsAttributeRef, Error, FlatSymbolRefAttributeRef,
    IndexTypeRef, IntegerAttributeRef, Location, OneRegion, Operation, OperationBuilder, RegionRef, ShapedType,
    ShapedTypeRef, StringAttributeRef, StringRef, TryFromWithContext, TryIntoWithContext, Type, Value, ValueRef,
    mlir_attribute_field, mlir_op, mlir_op_trait, mlir_subtype_trait_impls,
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
    fn value(&self) -> Result<ElementsAttributeRef<'c, 't>, Error> {
        self.elements_attribute(CONSTANT_VALUE_ATTRIBUTE)
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
) -> Result<DetachedConstantOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::stable_hlo()?)?;
    OperationBuilder::new("stablehlo.constant", location)
        .add_attribute(CONSTANT_VALUE_ATTRIBUTE, value)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `stable_hlo::constant`"))
        })
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
    fn iota_dimension(&self) -> Result<usize, Error> {
        let value = self.integer_attribute(IOTA_DIMENSION_ATTRIBUTE)?.signless_value();
        usize::try_from(value)
            .map_err(|_| Error::invalid_argument("invalid `iota_dimension` attribute in `stablehlo.iota`"))
    }
}

mlir_op!(Iota);
mlir_op_trait!(Iota, OneResult);
mlir_op_trait!(Iota, ZeroRegions);
mlir_op_trait!(Iota, ZeroSuccessors);

/// Constructs a new detached/owned [`IotaOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`IotaOperation`] for more information on the operation semantics.
pub fn iota<'c, 't: 'c, T: ShapedType<'c, 't>, L: Location<'c, 't>>(
    output_type: T,
    iota_dimension: usize,
    location: L,
) -> Result<DetachedIotaOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::stable_hlo()?)?;
    OperationBuilder::new("stablehlo.iota", location)
        .add_attribute(
            IOTA_DIMENSION_ATTRIBUTE,
            location
                .context()
                .integer_attribute(location.context().signless_integer_type(64), iota_dimension as i64),
        )
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `stable_hlo::iota`"))
        })
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
    fn iota_dimension(&self) -> Result<usize, Error> {
        let value = self.integer_attribute(IOTA_DIMENSION_ATTRIBUTE)?.signless_value();
        usize::try_from(value)
            .map_err(|_| Error::invalid_argument("invalid `iota_dimension` attribute in `stablehlo.dynamic_iota`"))
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
pub fn dynamic_iota<'s, 'c: 's, 't: 'c, S: Value<'s, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    output_shape: S,
    output_type: T,
    iota_dimension: usize,
    location: L,
) -> Result<DetachedDynamicIotaOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::stable_hlo()?)?;
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
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `stable_hlo::dynamic_iota`"))
        })
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
///   %1 = stablehlo.compare GT, %input2, %input3, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
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
    fn dimension(&self) -> Result<usize, Error> {
        let value = self.integer_attribute(SORT_DIMENSION_ATTRIBUTE)?.signless_value();
        usize::try_from(value).map_err(|_| Error::invalid_argument("invalid `dimension` attribute in `stablehlo.sort`"))
    }

    /// Returns `true` if the sorting performed by this [`SortOperation`] is stable (i.e., the relative order of
    /// elements considered to be equal by the comparator is preserved).
    fn is_stable(&self) -> Result<bool, Error> {
        Ok(self.boolean_attribute(SORT_IS_STABLE_ATTRIBUTE)?.value())
    }

    /// Returns a reference to the [`Region`](crate::Region) that contains the comparator
    /// used by this [`SortOperation`].
    fn comparator(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedSortOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo()?)?;
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
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `stable_hlo::sort`"))
        })
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
    fn reverse_dimensions(&self) -> Result<Vec<usize>, Error> {
        self.dense_integer_64_array_attribute(REVERSE_DIMENSIONS_ATTRIBUTE)?
            .values()
            .map(|value| {
                usize::try_from(value)
                    .map_err(|_| Error::invalid_argument("invalid `dimensions` attribute in `stablehlo.reverse`"))
            })
            .collect()
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
) -> Result<DetachedReverseOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::stable_hlo()?)?;
    OperationBuilder::new("stablehlo.reverse", location)
        .add_operand(input)
        .add_attribute(
            REVERSE_DIMENSIONS_ATTRIBUTE,
            location
                .context()
                .dense_i64_array_attribute(dimensions.iter().map(|v| *v as i64).collect::<Vec<_>>().as_slice())?,
        )
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `stable_hlo::reverse`"))
        })
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
    fn values(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        self.operand_values()
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
) -> Result<DetachedReturnOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::stable_hlo()?)?;
    OperationBuilder::new("stablehlo.return", location)
        .add_operands(values)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `stable_hlo::return`"))
        })
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
) -> Result<DetachedOptimizationBarrierOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::stable_hlo()?)?;
    OperationBuilder::new("stablehlo.optimization_barrier", location)
        .add_operands(operands)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `stable_hlo::optimization_barrier`"))
        })
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
/// [`CompositeOperation::composite_version`]. [`CompositeOperation`]s may also carry zero or more
/// [`Region`](crate::Region)s through [`CompositeOperation::composite_regions`] in order to model composite operations
/// with bodies (e.g., operations like `while` or `reduce`). If the decomposition is inlined, the `regions` will be
/// ignored.
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
    fn composite_name(&self) -> Result<StringRef<'c>, Error> {
        Ok(self.string_attribute(COMPOSITE_NAME_ATTRIBUTE)?.string())
    }

    /// Returns the version of this [`CompositeOperation`]. Typically, version 0 means that a composite operation is
    /// under development and does not imply any compatibility guarantees, whereas higher versions do.
    fn composite_version(&self) -> Result<u32, Error> {
        if self.has_attribute(COMPOSITE_VERSION_ATTRIBUTE) {
            u32::try_from(self.integer_attribute(COMPOSITE_VERSION_ATTRIBUTE)?.unsigned_value())
                .map_err(|_| Error::invalid_argument("invalid `version` attribute in `stable_hlo::composite`"))
        } else {
            Ok(0)
        }
    }

    /// Returns the composite [`Attribute`]s of this [`CompositeOperation`] that will be propagated to
    /// [`CompositeOperation::composite_decomposition`] when this operation is invoked.
    fn composite_attributes(&self) -> Result<HashMap<StringRef<'c>, AttributeRef<'c, 't>>, Error> {
        if self.has_attribute(COMPOSITE_ATTRIBUTES_ATTRIBUTE) {
            HashMap::try_from(self.dictionary_attribute(COMPOSITE_ATTRIBUTES_ATTRIBUTE)?)
        } else {
            Ok(HashMap::new())
        }
    }

    /// Returns the name/symbol of the decomposition [`func::func`](crate::dialects::func::func)  of this
    /// [`CompositeOperation`]. The referred function must be defined in the parent scope of this operation.
    fn composite_decomposition(&self) -> Result<StringRef<'c>, Error> {
        Ok(self.flat_symbol_ref_attribute(COMPOSITE_DECOMPOSITION_ATTRIBUTE)?.reference())
    }

    /// Returns an [`Iterator`] over the [`Region`](crate::Region)s carried by this [`CompositeOperation`].
    fn composite_regions(&self) -> impl Iterator<Item = Result<RegionRef<'o, 'c, 't>, Error>> {
        self.regions()
    }
}

mlir_op!(Composite);
mlir_op_trait!(Composite, ZeroSuccessors);

/// Constructs a new detached/owned [`CompositeOperation`] at the specified [`Location`]. Refer to the documentation
/// of [`CompositeOperation`], [`CompositeOperation::composite_name`], [`CompositeOperation::composite_version`],
/// [`CompositeOperation::composite_attributes`], [`CompositeOperation::composite_decomposition`], and
/// [`CompositeOperation::composite_regions`], for more information on the operation semantics and the arguments of
/// this function.
///
/// Note that if any of the inputs to this function are invalid, the function may panic!
#[allow(clippy::too_many_arguments)]
pub fn composite<
    'v,
    'c: 'v,
    't: 'c,
    's,
    N: TryIntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
    A: Attribute<'c, 't>,
    V: Value<'v, 'c, 't>,
    D: TryIntoWithContext<'c, 't, FlatSymbolRefAttributeRef<'c, 't>>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    name: N,
    version: u32,
    attributes: Option<&HashMap<StringRef<'s>, A>>,
    operands: &[V],
    decomposition: D,
    regions: Vec<DetachedRegion<'c, 't>>,
    result_types: &[T],
    location: L,
) -> Result<DetachedCompositeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo()?)?;
    let mut builder = OperationBuilder::new("stablehlo.composite", location)
        .add_operands(operands)
        .add_attribute(COMPOSITE_NAME_ATTRIBUTE, name.try_into_with_context(context)?)
        .add_attribute(COMPOSITE_DECOMPOSITION_ATTRIBUTE, decomposition.try_into_with_context(context)?);
    if let Some(attributes) = attributes.filter(|attributes| !attributes.is_empty()) {
        builder = builder.add_attribute(
            COMPOSITE_ATTRIBUTES_ATTRIBUTE,
            DictionaryAttributeRef::try_from_with_context(attributes, context)?,
        )
    }
    builder = builder.add_attribute(
        COMPOSITE_VERSION_ATTRIBUTE,
        context.integer_attribute(context.signless_integer_type(32), i64::from(version)),
    );
    if !regions.is_empty() {
        builder = builder.add_regions(regions);
    }
    builder.add_results(result_types).build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `stable_hlo::composite`"))
    })
}

/// API version used by a [`CustomCallOperation`]. This determines the format in which the custom operation metadata
/// are specified (i.e., as a [`StringAttributeRef`] or a [`DictionaryAttributeRef`] among other things related to how
/// it should be invoked.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

impl<'c, 't> TryFromWithContext<'c, 't, CustomCallApiVersion> for IntegerAttributeRef<'c, 't> {
    fn try_from_with_context(value: CustomCallApiVersion, context: &'c Context<'t>) -> Result<Self, Error> {
        let r#type = context.signless_integer_type(32);
        Ok(match value {
            CustomCallApiVersion::Unspecified => context.integer_attribute(r#type, 0),
            CustomCallApiVersion::Original => context.integer_attribute(r#type, 1),
            CustomCallApiVersion::StatusReturning => context.integer_attribute(r#type, 2),
            CustomCallApiVersion::StatusReturningUnified => context.integer_attribute(r#type, 3),
            CustomCallApiVersion::TypedFfi => context.integer_attribute(r#type, 4),
        })
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
    ) -> Result<OutputOperandAliasAttributeRef<'c, 't>, Error> {
        // Make sure that the StableHLO dialect is loaded into the current context to prevent segmentation faults.
        self.load_dialect(DialectHandle::stable_hlo()?)?;
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
            .map_err(|_| Error::invalid_argument("invalid arguments to `Context::stable_hlo_output_operand_alias`"))
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

/// Name of the [`Attribute`] that is used to store [`CustomCallOperation::custom_call_result_tilings`].
pub const CUSTOM_CALL_RESULT_TILINGS_ATTRIBUTE: &str = "result_tilings";

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
/// via [`CustomCallOperation::custom_call_memory_layouts`], as well as tiling information for its results/outputs via
/// [`CustomCallOperation::custom_call_result_tilings`].
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
    fn custom_call_target_name(&self) -> Result<StringRef<'c>, Error> {
        Ok(self.string_attribute(CUSTOM_CALL_TARGET_NAME_ATTRIBUTE)?.string())
    }

    /// Returns `true` if executing this [`CustomCallOperation`] can result in side effects
    /// (i.e., if this operation is not _pure_).
    fn custom_call_has_side_effect(&self) -> Result<bool, Error> {
        Ok(self.boolean_attribute(CUSTOM_CALL_HAS_SIDE_EFFECT_ATTRIBUTE)?.value())
    }

    /// Returns the backend configuration of this [`CustomCallOperation`]. This is either a [`StringAttributeRef`]
    /// (when [`CustomCallOperation::custom_call_api_version`] is not [`CustomCallApiVersion::TypedFfi`])
    /// or a [`DictionaryAttributeRef`] (when [`CustomCallOperation::custom_call_api_version`] is
    /// [`CustomCallApiVersion::TypedFfi`]) that contains implementation-specific metadata.
    fn custom_call_backend_config(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute(CUSTOM_CALL_BACKEND_CONFIG_ATTRIBUTE)?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                CUSTOM_CALL_BACKEND_CONFIG_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the [`CustomCallApiVersion`] of this [`CustomCallOperation`].
    fn custom_call_api_version(&self) -> Result<CustomCallApiVersion, Error> {
        Ok(self.integer_attribute(CUSTOM_CALL_API_VERSION_ATTRIBUTE)?.into())
    }

    /// Returns the names/symbols of functions that are used by this [`CustomCallOperation`].
    fn custom_call_called_computations(&self) -> Result<Vec<StringRef<'c>>, Error> {
        self.array_attribute(CUSTOM_CALL_CALLED_COMPUTATIONS_ATTRIBUTE)?
            .elements()
            .map(|attribute| {
                attribute?
                    .cast::<FlatSymbolRefAttributeRef>()
                    .map(|attribute| attribute.reference())
                    .ok_or_else(|| {
                        Error::invalid_argument("invalid `called_computations` attribute in `stablehlo.custom_call`")
                    })
            })
            .collect()
    }

    /// Returns the optional memory layout information for the operands/inputs and the results/outputs of this
    /// [`CustomCallOperation`]. Refer to the documentation of [`CustomCallMemoryLayouts`] for information on the
    /// semantics of these memory layouts.
    fn custom_call_memory_layouts(&self) -> Result<Option<CustomCallMemoryLayouts>, Error> {
        let operands = if self.has_attribute(CUSTOM_CALL_OPERAND_LAYOUTS_ATTRIBUTE) {
            Some(self.array_attribute(CUSTOM_CALL_OPERAND_LAYOUTS_ATTRIBUTE)?)
        } else {
            None
        };
        let results = if self.has_attribute(CUSTOM_CALL_RESULT_LAYOUTS_ATTRIBUTE) {
            Some(self.array_attribute(CUSTOM_CALL_RESULT_LAYOUTS_ATTRIBUTE)?)
        } else {
            None
        };
        let (operands, results) = match (operands, results) {
            (Some(operands), Some(results)) => (operands, results),
            (None, None) => return Ok(None),
            _ => {
                return Err(Error::invalid_argument(
                    "custom call operand and result layouts must be provided together",
                ));
            }
        };
        Ok(Some(CustomCallMemoryLayouts {
            operands: operands
                .elements()
                .map(|attribute| {
                    let attribute = attribute?.cast::<DenseIntegerElementsAttributeRef<'c, 't>>().ok_or_else(|| {
                        Error::invalid_argument("invalid `operand_layouts` attribute in `stablehlo.custom_call`")
                    })?;
                    let r#type = attribute.r#type()?.cast::<ShapedTypeRef>().ok_or_else(|| {
                        Error::invalid_argument("invalid `operand_layouts` attribute in `stablehlo.custom_call`")
                    })?;
                    if !r#type.element_type()?.is::<IndexTypeRef>() {
                        return Err(Error::invalid_argument(
                            "invalid `operand_layouts` attribute in `stablehlo.custom_call`",
                        ));
                    }
                    unsafe { attribute.usize_elements().collect() }
                })
                .collect::<Result<Vec<_>, Error>>()?,
            results: results
                .elements()
                .map(|attribute| {
                    let attribute = attribute?.cast::<DenseIntegerElementsAttributeRef<'c, 't>>().ok_or_else(|| {
                        Error::invalid_argument("invalid `result_layouts` attribute in `stablehlo.custom_call`")
                    })?;
                    let r#type = attribute.r#type()?.cast::<ShapedTypeRef>().ok_or_else(|| {
                        Error::invalid_argument("invalid `result_layouts` attribute in `stablehlo.custom_call`")
                    })?;
                    if !r#type.element_type()?.is::<IndexTypeRef>() {
                        return Err(Error::invalid_argument(
                            "invalid `result_layouts` attribute in `stablehlo.custom_call`",
                        ));
                    }
                    unsafe { attribute.usize_elements().collect() }
                })
                .collect::<Result<Vec<_>, Error>>()?,
        }))
    }

    /// Returns the alias relationship between outputs and operands of this [`CustomCallOperation`].
    fn custom_call_output_operand_aliases(&self) -> Result<Vec<OutputOperandAliasAttributeRef<'c, 't>>, Error> {
        self.array_attribute(CUSTOM_CALL_OUTPUT_OPERAND_ALIASES_ATTRIBUTE)?
            .elements()
            .map(|attribute| {
                attribute?.cast::<OutputOperandAliasAttributeRef>().ok_or_else(|| {
                    Error::invalid_argument("invalid `output_operand_aliases` attribute in `stablehlo.custom_call`")
                })
            })
            .collect()
    }

    /// Returns the optional tiling information for the results/outputs of this [`CustomCallOperation`]. When present,
    /// the returned vector contains one tiling per result of this operation, where each tiling is a sequence of tiles
    /// and each tile is a sequence of tile dimension sizes (represented as a 1-D tensor of index type in the
    /// underlying [`Attribute`]).
    fn custom_call_result_tilings(&self) -> Result<Option<Vec<Vec<Vec<usize>>>>, Error> {
        if !self.has_attribute(CUSTOM_CALL_RESULT_TILINGS_ATTRIBUTE) {
            return Ok(None);
        }
        self.array_attribute(CUSTOM_CALL_RESULT_TILINGS_ATTRIBUTE)?
            .elements()
            .map(|tiling| {
                tiling?
                    .cast::<ArrayAttributeRef<'c, 't>>()
                    .ok_or_else(|| {
                        Error::invalid_argument("invalid `result_tilings` attribute in `stablehlo.custom_call`")
                    })?
                    .elements()
                    .map(|tile| {
                        let tile = tile?.cast::<DenseIntegerElementsAttributeRef<'c, 't>>().ok_or_else(|| {
                            Error::invalid_argument("invalid `result_tilings` attribute in `stablehlo.custom_call`")
                        })?;
                        let r#type = tile.r#type()?.cast::<ShapedTypeRef>().ok_or_else(|| {
                            Error::invalid_argument("invalid `result_tilings` attribute in `stablehlo.custom_call`")
                        })?;
                        if !r#type.element_type()?.is::<IndexTypeRef>() {
                            return Err(Error::invalid_argument(
                                "invalid `result_tilings` attribute in `stablehlo.custom_call`",
                            ));
                        }
                        unsafe { tile.usize_elements().collect() }
                    })
                    .collect::<Result<Vec<_>, Error>>()
            })
            .collect::<Result<Vec<_>, Error>>()
            .map(Some)
    }
}

mlir_op!(CustomCall);
mlir_op_trait!(CustomCall, ZeroRegions);
mlir_op_trait!(CustomCall, ZeroSuccessors);

/// Constructs a new detached/owned [`CustomCallOperation`] at the specified [`Location`]. Refer to the documentation
/// of [`CustomCallOperation`], [`CustomCallOperation::custom_call_target_name`],
/// [`CustomCallOperation::custom_call_has_side_effect`], [`CustomCallOperation::custom_call_backend_config`],
/// [`CustomCallOperation::custom_call_api_version`], [`CustomCallOperation::custom_call_called_computations`],
/// [`CustomCallOperation::custom_call_memory_layouts`], [`CustomCallOperation::custom_call_output_operand_aliases`],
/// and [`CustomCallOperation::custom_call_result_tilings`] for more information on the operation semantics and the
/// arguments of this function.
///
/// Note that if any of the inputs to this function are invalid, the function may panic!
#[allow(clippy::too_many_arguments)]
pub fn custom_call<
    'v,
    'c: 'v,
    't: 'c,
    V: Value<'v, 'c, 't>,
    N: TryIntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
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
    result_tilings: Option<Vec<Vec<Vec<usize>>>>,
    output_types: &[T],
    location: L,
) -> Result<DetachedCustomCallOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo()?)?;
    let mut builder = OperationBuilder::new("stablehlo.custom_call", location)
        .add_operands(inputs)
        .add_attribute(CUSTOM_CALL_TARGET_NAME_ATTRIBUTE, target_name.try_into_with_context(context)?)
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
                        .map(|layout| {
                            DenseIntegerElementsAttributeRef::try_from_with_context(layout.as_slice(), context)
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                ),
            )
            .add_attribute(
                CUSTOM_CALL_RESULT_LAYOUTS_ATTRIBUTE,
                context.array_attribute(
                    &memory_layouts
                        .results
                        .iter()
                        .map(|layout| {
                            DenseIntegerElementsAttributeRef::try_from_with_context(layout.as_slice(), context)
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                ),
            );
    }

    if let Some(result_tilings) = result_tilings {
        builder = builder.add_attribute(
            CUSTOM_CALL_RESULT_TILINGS_ATTRIBUTE,
            context.array_attribute(
                &result_tilings
                    .iter()
                    .map(|tiling| {
                        Ok(context.array_attribute(
                            &tiling
                                .iter()
                                .map(|tile| {
                                    DenseIntegerElementsAttributeRef::try_from_with_context(tile.as_slice(), context)
                                })
                                .collect::<Result<Vec<_>, _>>()?,
                        ))
                    })
                    .collect::<Result<Vec<_>, Error>>()?,
            ),
        );
    }

    builder
        .add_attribute(
            CUSTOM_CALL_API_VERSION_ATTRIBUTE,
            IntegerAttributeRef::try_from_with_context(api_version, context)?,
        )
        .add_attribute(CUSTOM_CALL_CALLED_COMPUTATIONS_ATTRIBUTE, context.array_attribute(called_computations))
        .add_attribute(CUSTOM_CALL_OUTPUT_OPERAND_ALIASES_ATTRIBUTE, context.array_attribute(output_operand_aliases))
        .add_results(output_types)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `stable_hlo::custom_call`"))
        })
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
) -> Result<DetachedCustomCallOperation<'c, 't>, Error> {
    custom_call::<ValueRef, _, _, _>(
        &[],
        XLA_GPU_CREATE_BUFFER_CUSTOM_CALL_TARGET_NAME,
        false,
        None,
        CustomCallApiVersion::TypedFfi,
        &[],
        None,
        &[],
        None,
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
) -> Result<DetachedCustomCallOperation<'c, 't>, Error> {
    custom_call(
        &[input],
        XLA_GPU_PIN_CUSTOM_CALL_TARGET_NAME,
        false,
        None,
        CustomCallApiVersion::TypedFfi,
        &[],
        None,
        &[],
        None,
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
) -> Result<DetachedCustomCallOperation<'c, 't>, Error> {
    custom_call(
        &[input],
        XLA_GPU_UNPIN_CUSTOM_CALL_TARGET_NAME,
        false,
        None,
        CustomCallApiVersion::TypedFfi,
        &[],
        None,
        &[],
        None,
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
        CompositeOperation, ConstantOperation, CustomCallApiVersion, CustomCallMemoryLayouts, CustomCallOperation,
        DynamicIotaOperation, IotaOperation, ReverseOperation, SortOperation,
        XLA_GPU_CREATE_BUFFER_CUSTOM_CALL_TARGET_NAME, XLA_GPU_PIN_CUSTOM_CALL_TARGET_NAME,
        XLA_GPU_UNPIN_CUSTOM_CALL_TARGET_NAME, composite, constant, custom_call, dynamic_iota,
        gpu_create_buffer_custom_call, gpu_pin_custom_call, gpu_unpin_custom_call, iota, optimization_barrier,
        r#return, reverse, sort,
    };

    #[test]
    fn test_constant() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i64_type = context.signless_integer_type(64);
        let tensor_type = context.tensor_type(i64_type, &[Size::Static(2), Size::Static(4)], None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let value = context.dense_i64_elements_attribute(tensor_type, &[0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
                let op = constant(value, location).unwrap();
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.value().unwrap().to_string(), value.to_string());
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "constant_test",
                    func::FuncAttributes { arguments: vec![], results: vec![tensor_type.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
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
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(4), Size::Static(5)], None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let op = iota(tensor_type, 1, location).unwrap();
                assert_eq!(op.iota_dimension().unwrap(), 1);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.result(0).unwrap().r#type().unwrap(), tensor_type);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "iota_test",
                    func::FuncAttributes { arguments: vec![], results: vec![tensor_type.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
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
        let module = context.module(location).unwrap();
        let i64_type = context.signless_integer_type(64);
        let shape_tensor_type = context.tensor_type(i64_type, &[Size::Static(2)], None, location).unwrap();
        let tensor_type = context.tensor_type(i64_type, &[Size::Dynamic, Size::Dynamic], None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(shape_tensor_type, location)]);
                let op = dynamic_iota(block.argument(0).unwrap(), tensor_type, 1, location).unwrap();
                assert_eq!(op.iota_dimension().unwrap(), 1);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "dynamic_iota_test",
                    func::FuncAttributes {
                        arguments: vec![shape_tensor_type.into()],
                        results: vec![tensor_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
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
        let module = context.module(location).unwrap();
        let i64_type = context.signless_integer_type(64);
        let input_type = context.tensor_type(i64_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        let scalar_i64_type = context.tensor_type(i64_type, &[], None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(input_type, location), (input_type, location)]);
                let mut comparator_region = context.region();
                let mut comparator_block = context.block(&[
                    (scalar_i64_type, location),
                    (scalar_i64_type, location),
                    (scalar_i64_type, location),
                    (scalar_i64_type, location),
                ]);
                let compare_op = comparator_block
                    .append_operation(
                        stable_hlo::compare(
                            comparator_block.argument(0).unwrap(),
                            comparator_block.argument(1).unwrap(),
                            stable_hlo::ComparisonDirection::GreaterThan,
                            stable_hlo::ComparisonType::Signed,
                            location,
                        )
                        .unwrap(),
                    )
                    .unwrap();
                comparator_block
                    .append_operation(r#return(&[compare_op.result(0).unwrap()], location).unwrap())
                    .unwrap();
                comparator_region.append_block(comparator_block).unwrap();
                let op = sort(
                    &block.arguments().collect::<Result<Vec<_>, _>>().unwrap(),
                    1,
                    true,
                    comparator_region,
                    location,
                )
                .unwrap();
                assert_eq!(op.dimension().unwrap(), 1);
                assert!(op.is_stable().unwrap());
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.regions().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(
                        func::r#return(
                            &op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().collect::<Vec<_>>(),
                            location,
                        )
                        .unwrap(),
                    )
                    .unwrap();
                func::func(
                    "sort_test",
                    func::FuncAttributes {
                        arguments: vec![input_type.into(), input_type.into()],
                        results: vec![input_type.into(), input_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @sort_test(%arg0: tensor<2x2xi64>, %arg1: tensor<2x2xi64>) -> (tensor<2x2xi64>, tensor<2x2xi64>) {
                    %0:2 = \"stablehlo.sort\"(%arg0, %arg1) <{dimension = 1 : i64, is_stable = true}> ({
                    ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>, %arg4: tensor<i64>, %arg5: tensor<i64>):
                      %1 = stablehlo.compare GT, %arg2, %arg3, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
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
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let input_tensor_type = context
            .tensor_type(i32_type, &[Size::Static(3), Size::Static(4), Size::Static(5)], None, location)
            .unwrap();
        let output_tensor_type = context
            .tensor_type(i32_type, &[Size::Static(3), Size::Static(4), Size::Static(5)], None, location)
            .unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(input_tensor_type, location)]);
                let input = block.argument(0).unwrap();
                let op = reverse(input, &[0, 2], location).unwrap();
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.reverse_dimensions().unwrap(), vec![0, 2]);
                assert_eq!(op.reverse_dimensions().unwrap(), vec![0, 2]);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "reverse_test",
                    func::FuncAttributes {
                        arguments: vec![input_tensor_type.into()],
                        results: vec![output_tensor_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
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
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        let scalar_tensor_type = context.tensor_type(f32_type, &[], None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let mut map_region = context.region();
                let mut map_block = context.block(&[(scalar_tensor_type, location)]);
                let input = map_block.argument(0).unwrap();
                let negate_op = stable_hlo::negate(input, location).unwrap();
                let negate_op = map_block.append_operation(negate_op).unwrap();
                let return_op = r#return(&[negate_op.result(0).unwrap()], location).unwrap();
                assert_eq!(return_op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(return_op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(return_op.regions().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                map_block.append_operation(return_op).unwrap();
                map_region.append_block(map_block).unwrap();
                let map_op =
                    stable_hlo::map(&[block.argument(0).unwrap()], &[0, 1], map_region.into(), location).unwrap();
                let map_op = block.append_operation(map_op).unwrap();
                block.append_operation(func::r#return(&[map_op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "return_test",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into()],
                        results: vec![tensor_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
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
        let module = context.module(location).unwrap();
        let tensor_type = context.tensor_type(context.float32_type(), &[], None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location), (tensor_type, location)]);
                let op =
                    optimization_barrier(&block.arguments().collect::<Result<Vec<_>, _>>().unwrap(), location).unwrap();
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                let op = block.append_operation(op).unwrap();
                block
                    .append_operation(
                        func::r#return(
                            &op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().collect::<Vec<_>>(),
                            location,
                        )
                        .unwrap(),
                    )
                    .unwrap();
                func::func(
                    "optimization_barrier_test",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into(), tensor_type.into()],
                        results: vec![tensor_type.into(), tensor_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
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
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[], None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut decomposition = context.block(&[(tensor_type, location), (tensor_type, location)]);
                let op =
                    stable_hlo::add(decomposition.argument(0).unwrap(), decomposition.argument(1).unwrap(), location)
                        .unwrap();
                let op = decomposition.append_operation(op).unwrap();
                decomposition.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "my_op",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into(), tensor_type.into()],
                        results: vec![tensor_type.into()],
                        visibility: SymbolVisibility::Private,
                        ..Default::default()
                    },
                    decomposition.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location), (tensor_type, location)]);
                let composite_attributes =
                    HashMap::from([(StringRef::from("my_op_attribute"), context.unit_attribute().as_ref())]);
                let mut composite_region = context.region();
                let mut composite_region_block = context.block(&[(tensor_type, location), (tensor_type, location)]);
                composite_region_block
                    .append_operation(
                        stable_hlo::r#return(
                            &composite_region_block.arguments().collect::<Result<Vec<_>, _>>().unwrap(),
                            location,
                        )
                        .unwrap(),
                    )
                    .unwrap();
                composite_region.append_block(composite_region_block).unwrap();
                let composite_op = composite(
                    "my_namespace.my_op",
                    1,
                    Some(&composite_attributes),
                    &block.arguments().collect::<Result<Vec<_>, _>>().unwrap(),
                    "my_op",
                    vec![composite_region.into()],
                    &[tensor_type],
                    location,
                )
                .unwrap();
                assert_eq!(composite_op.composite_name().unwrap().as_str().unwrap(), "my_namespace.my_op");
                assert_eq!(composite_op.composite_version().unwrap(), 1);
                assert_eq!(composite_op.composite_attributes().unwrap(), composite_attributes);
                assert_eq!(composite_op.composite_decomposition().unwrap().as_str().unwrap(), "my_op");
                assert_eq!(
                    composite_op.composite_regions().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(),
                    1,
                );
                assert_eq!(composite_op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(composite_op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(composite_op.regions().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let composite_op = block.append_operation(composite_op).unwrap();
                block
                    .append_operation(func::r#return(&[composite_op.result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "composite_test",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into(), tensor_type.into()],
                        results: vec![tensor_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func private @my_op(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
                    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
                    return %0 : tensor<f32>
                  }
                  func.func @composite_test(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
                    %0 = stablehlo.composite \"my_namespace.my_op\" %arg0, %arg1 ({
                    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                      stablehlo.return %arg2, %arg3 : tensor<f32>, tensor<f32>
                    }) {\
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
        let attribute = context.stable_hlo_output_operand_alias(&[0, 1], 2, &[3, 4]).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.output_tuple_indices(), vec![0, 1]);
        assert_eq!(attribute.operand_index(), 2);
        assert_eq!(attribute.operand_tuple_indices(), vec![3, 4]);
    }

    #[test]
    fn test_output_operand_alias_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_output_operand_alias(&[0, 1], 2, &[3, 4]).unwrap();
        let attribute_2 = context.stable_hlo_output_operand_alias(&[0, 1], 2, &[3, 4]).unwrap();
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.stable_hlo_output_operand_alias(&[1, 0], 2, &[3, 4]).unwrap();
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.stable_hlo_output_operand_alias(&[0, 1], 2, &[3, 4]).unwrap();
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_output_operand_alias_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.stable_hlo_output_operand_alias(&[0, 1], 2, &[3, 4]).unwrap();
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
        let attribute = context.stable_hlo_output_operand_alias(&[0, 1], 2, &[3, 4]).unwrap();
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_custom_call() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(4), Size::Static(2)], None, location).unwrap();
        let memref_type =
            context.mem_ref_type(f32_type, &[Size::Static(4), Size::Static(2)], None, None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location), (tensor_type, location)]);
                let backend_config = context.string_attribute("status_returning_attribute");
                let op = custom_call(
                    &block.arguments().collect::<Result<Vec<_>, _>>().unwrap(),
                    "my_custom_op",
                    true,
                    Some(backend_config.as_ref()),
                    CustomCallApiVersion::StatusReturning,
                    &[context.flat_symbol_ref_attribute("add_0"), context.flat_symbol_ref_attribute("add_1")],
                    Some(CustomCallMemoryLayouts { operands: vec![vec![0, 1], vec![1, 0]], results: vec![vec![1, 0]] }),
                    &[
                        context.stable_hlo_output_operand_alias(&[], 1, &[]).unwrap(),
                        context.stable_hlo_output_operand_alias(&[], 0, &[]).unwrap(),
                    ],
                    Some(vec![vec![vec![2, 4], vec![1, 2]]]),
                    &[tensor_type],
                    location,
                )
                .unwrap();
                assert_eq!(op.custom_call_target_name().unwrap().as_str().unwrap(), "my_custom_op");
                assert!(op.custom_call_has_side_effect().unwrap());
                assert_eq!(op.custom_call_backend_config().unwrap(), backend_config);
                assert_eq!(op.custom_call_api_version().unwrap(), CustomCallApiVersion::StatusReturning);
                assert_eq!(
                    op.custom_call_called_computations()
                        .unwrap()
                        .iter()
                        .map(|string_ref| string_ref.as_str().unwrap())
                        .collect::<Vec<_>>(),
                    ["add_0", "add_1"],
                );
                assert_eq!(
                    op.custom_call_memory_layouts().unwrap(),
                    Some(CustomCallMemoryLayouts { operands: vec![vec![0, 1], vec![1, 0]], results: vec![vec![1, 0]] })
                );
                assert_eq!(op.custom_call_output_operand_aliases().unwrap().len(), 2);
                assert_eq!(op.custom_call_result_tilings().unwrap(), Some(vec![vec![vec![2, 4], vec![1, 2]]]));
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.regions().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                let op = block.append_operation(op).unwrap();

                // Add a couple more custom calls testing the XLA GPU built-in constructors.
                let create_buffer_op = gpu_create_buffer_custom_call(memref_type, location).unwrap();
                assert_eq!(
                    create_buffer_op.custom_call_target_name().unwrap().as_str().unwrap(),
                    XLA_GPU_CREATE_BUFFER_CUSTOM_CALL_TARGET_NAME,
                );
                assert_eq!(create_buffer_op.custom_call_api_version().unwrap(), CustomCallApiVersion::TypedFfi);
                assert_eq!(create_buffer_op.custom_call_result_tilings().unwrap(), None);
                block.append_operation(create_buffer_op).unwrap();

                let pin_op = gpu_pin_custom_call(op.result(0).unwrap(), memref_type, location).unwrap();
                assert_eq!(
                    pin_op.custom_call_target_name().unwrap().as_str().unwrap(),
                    XLA_GPU_PIN_CUSTOM_CALL_TARGET_NAME
                );
                assert_eq!(pin_op.custom_call_api_version().unwrap(), CustomCallApiVersion::TypedFfi);
                let pin_op = block.append_operation(pin_op).unwrap();

                let unpin_op = gpu_unpin_custom_call(pin_op.result(0).unwrap(), tensor_type, location).unwrap();
                assert_eq!(
                    unpin_op.custom_call_target_name().unwrap().as_str().unwrap(),
                    XLA_GPU_UNPIN_CUSTOM_CALL_TARGET_NAME,
                );
                assert_eq!(unpin_op.custom_call_api_version().unwrap(), CustomCallApiVersion::TypedFfi);
                let unpin_op = block.append_operation(unpin_op).unwrap();

                block.append_operation(func::r#return(&[unpin_op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "custom_call_test",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into(), tensor_type.into()],
                        results: vec![tensor_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
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
                      operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], \
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
                      result_layouts = [dense<[1, 0]> : tensor<2xindex>], \
                      result_tilings = [[dense<[2, 4]> : tensor<2xindex>, dense<[1, 2]> : tensor<2xindex>]]\
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
