use std::fmt::Display;

use crate::batching::{
    ArrayBatch, BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::impl_non_transposable_operation;
use crate::operations::constants::IotaOperation;
use crate::operations::manipulation::{Broadcast, Reshape, Slice, Transpose};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::ProgramError;
use crate::programs::atoms::MaybeZero;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{TypeError, Typed};
use crate::programs::values::Value;
use crate::sharding::ShardingDimension;
use crate::types::{ArrayType, DataType, Shape, Size};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`SortOperation`].
pub const SORT_OPERATION_NAME: &str = "sort";

/// Direction in which a [`SortOperation`] orders its key operand.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SortDirection {
    /// Orders keys from smallest to largest.
    Ascending,

    /// Orders keys from largest to smallest.
    Descending,
}

impl Display for SortDirection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ascending => formatter.write_str("ascending"),
            Self::Descending => formatter.write_str("descending"),
        }
    }
}

/// [`Operation`] that sorts one or more same-shaped operands along one axis by the values of its first operand (the
/// key): the key is ordered by [`SortDirection`] and every other operand is co-permuted by the key's order, like
/// [JAX's `lax.sort`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.sort.html) with `num_keys = 1`. The sort
/// is always stable, so elements with equal keys keep their original relative order (which is what routes
/// [`argmax`](ArgMax::argmax)-style ties to the lowest index). Floating-point keys are ordered by the IEEE 754
/// total order (`-NaN < -∞ < … < -0.0 < +0.0 < … < +∞ < NaN`), matching
/// [StableHLO's `TOTALORDER` comparison](https://openxla.org/stablehlo/spec#compare); complex keys are unordered
/// and rejected. Operands must agree on shape (element types may differ), the sorted axis must not be sharded
/// (sorting across shards would require communication), and operands that still carry partial sums are rejected.
///
/// There is no user-provided comparator: the fixed key-ordering policy covers the ranking use cases
/// ([`top_k`](TopK::top_k), [`argmax`](ArgMax::argmax), [`argmin`](ArgMin::argmin)) without carrying a comparator
/// region through every program transform.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SortOperation {
    /// Axis along which the operands are sorted.
    axis: usize,

    /// Direction in which the key operand is ordered.
    direction: SortDirection,
}

impl SortOperation {
    /// Creates a new [`SortOperation`] sorting along `axis` in the provided `direction`.
    #[inline]
    pub fn new(axis: usize, direction: SortDirection) -> Self {
        Self { axis, direction }
    }

    /// Returns the axis along which the operands are sorted.
    #[inline]
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// Returns the direction in which the key operand is ordered.
    #[inline]
    pub fn direction(&self) -> SortDirection {
        self.direction
    }
}

impl Display for SortOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Operation::<ArrayType>::render(self, formatter, 0)
    }
}

impl Operation<ArrayType> for SortOperation {
    #[inline]
    fn name(&self) -> &'static str {
        SORT_OPERATION_NAME
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, SORT_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("axis", &self.axis)?;
            operation.field("direction", &self.direction)
        })
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        let Some(key_type) = input_types.first() else {
            return Err(TypeError { message: "'sort' needs at least one input".to_string() });
        };
        if matches!(key_type.data_type(), DataType::Token | DataType::Zero | DataType::C64 | DataType::C128) {
            return Err(TypeError {
                message: format!("'sort' does not support key data type {}", key_type.data_type()),
            });
        }
        if self.axis >= key_type.rank() {
            return Err(TypeError {
                message: format!("'sort' axis {} is out of bounds for rank {}", self.axis, key_type.rank()),
            });
        }
        for input_type in input_types {
            if input_type.shape() != key_type.shape() {
                return Err(TypeError {
                    message: format!(
                        "'sort' operands must agree on shape but got {} and {}",
                        key_type.shape(),
                        input_type.shape(),
                    ),
                });
            }
            if !input_type.unreduced_axes().is_empty() {
                return Err(TypeError { message: "'sort' does not support unreduced operands".to_string() });
            }
            if let Some(sharding) = input_type.sharding() {
                if matches!(sharding.dimensions()[self.axis], ShardingDimension::Sharded(_)) {
                    return Err(TypeError { message: format!("'sort' cannot sort along sharded axis {}", self.axis) });
                }
            }
        }
        Ok(input_types.to_vec())
    }
}

impl<C: Domain<Type = ArrayType, Value: Sort>> InterpretableOperation<C> for SortOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        C::Value::sort(inputs, self.axis, self.direction)
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for SortOperation where
    C::Operation: From<SortOperation>
{
}

/// Forward-mode rule for [`SortOperation`]: sorting co-permutes every non-key operand by the key's order, so the
/// live tangents ride one staged sort as extra passenger operands after the primals — the first half of the outputs
/// are the primal outputs and the rest are the co-permuted tangents (the same trick JAX's sort JVP uses).
/// Structural-zero tangents stay symbolic because any permutation of zeros is zero.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for SortOperation
where
    C::Operation: From<SortOperation>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let mut operands = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let live_indices = inputs
            .iter()
            .enumerate()
            .filter_map(|(index, input)| input.tangent().as_value().map(|tangent| (index, tangent.clone())))
            .collect::<Vec<_>>();
        operands.extend(live_indices.iter().map(|(_, tangent)| tangent.clone()));
        let mut outputs = context.bind(*self, Vec::new(), operands.as_slice())?;
        let output_tangents = outputs.split_off(inputs.len());
        let mut tangent_by_output = vec![None; inputs.len()];
        for ((index, _), tangent) in live_indices.iter().zip(output_tangents) {
            tangent_by_output[*index] = Some(tangent);
        }
        outputs
            .into_iter()
            .zip(tangent_by_output)
            .map(|(primal, tangent)| {
                let tangent = match tangent {
                    Some(tangent) => MaybeZero::Value(tangent),
                    None => MaybeZero::Zero(primal.r#type().tangent()),
                };
                DifferentiationDual::new(primal, tangent).map_err(DifferentiationError::from)
            })
            .collect()
    }
}

impl_non_transposable_operation!(SortOperation);

/// Batching rule for [`SortOperation`]: every mapped operand's batch axis moves to the leading physical position,
/// replicated operands broadcast to the batched physical shape (all sort operands must agree on shape), and the
/// sort axis lifts past the inserted leading batch dimension.
impl<C: Context<Type = ArrayType, Value: Broadcast + Transpose>> BatchableOperation<C> for SortOperation
where
    SortOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        let Some(axis_size) = ArrayBatch::common_batch_size(inputs)? else {
            return self.interpret_with_batch_axes(context, inputs, &vec![BatchAxis::replicated(); inputs.len()]);
        };
        let axis_sharding = ArrayBatch::sharding_for_inputs(inputs)?;
        let batched_inputs = inputs
            .iter()
            .map(|input| {
                if !input.batch_axis().is_replicated() {
                    return input.move_axis(0);
                }
                let unbatched_type = input.unbatched_type();
                let mut physical_type = unbatched_type.with_inserted_dimension(0, Size::Static(axis_size))?;
                if let Some(sharding) = unbatched_type.sharding() {
                    physical_type.sharding = Some(
                        sharding
                            .with_inserted_dimension(0, axis_sharding.clone())
                            .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?,
                    );
                }
                let output_axes = (1..physical_type.rank()).collect::<Vec<_>>();
                let broadcasted = input.value().clone().broadcast(physical_type.clone(), output_axes.as_slice())?;
                ArrayBatch::new(physical_type, broadcasted, 0)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let lifted = SortOperation::new(self.axis + 1, self.direction);
        lifted.interpret_with_batch_axes(
            context,
            batched_inputs.as_slice(),
            &vec![BatchAxis::from_position(0); inputs.len()],
        )
    }
}

/// Represents the ability to sort same-shaped operands along one axis by the first operand's values. [`Sort`]
/// stages or executes a [`SortOperation`]; refer to its documentation for the ordering policy and the transform
/// rules. The capability method dispatches through the first operand's context.
pub trait Sort: Sized {
    /// Sorts `operands` along `axis` by the first operand's values in the provided `direction`, co-permuting every
    /// other operand by the key's order, and returning a [`ProgramError`] if something goes wrong.
    fn sort(operands: &[Self], axis: usize, direction: SortDirection) -> Result<Vec<Self>, ProgramError>;
}

/// Any context-carrying value sorts by binding a [`SortOperation`] through its own context. The
/// `From<SortOperation>` bound makes this disjoint from the eager reference value types (whose context operation is
/// [`ConstantOperation`](crate::operations::constants::ConstantOperation)), so it covers the transform tracers and
/// backend-owned values without conflicting with concrete implementations.
impl<V: Value<Type = ArrayType>> Sort for V
where
    V::DispatchDomain: Context<Operation: From<SortOperation>>,
{
    fn sort(operands: &[Self], axis: usize, direction: SortDirection) -> Result<Vec<Self>, ProgramError> {
        let Some(first) = operands.first() else {
            return Err(ProgramError::UnsupportedOperation { message: "'sort' needs at least one input".to_string() });
        };
        first.dispatch_domain().bind(SortOperation::new(axis, direction), Vec::new(), operands)
    }
}

/// Applies the permutation computed from precomputed key ranks to every operand's values, sorting along `axis` of
/// an array with the provided static `dimensions`. This is the shared reference-backend evaluator behind the
/// concrete [`Sort`] implementations: `key_ranks` holds one order-preserving `u64` rank per key element (in
/// row-major order), the sort is stable, and [`SortDirection::Descending`] reverses the key comparison while
/// keeping equal-rank elements in their original order.
pub fn sort_evaluate<T: Clone>(
    key_ranks: &[u64],
    operand_values: &[&[T]],
    dimensions: &[usize],
    axis: usize,
    direction: SortDirection,
) -> Vec<Vec<T>> {
    let axis_size = dimensions[axis];
    let inner_stride: usize = dimensions[axis + 1..].iter().product();
    let outer_count: usize = dimensions[..axis].iter().product();
    let mut outputs = operand_values.iter().map(|values| values.to_vec()).collect::<Vec<_>>();
    if axis_size == 0 {
        return outputs;
    }
    let mut permutation = Vec::with_capacity(axis_size);
    for outer in 0..outer_count {
        for inner in 0..inner_stride {
            let base = outer * axis_size * inner_stride + inner;
            permutation.clear();
            permutation.extend(0..axis_size);
            match direction {
                SortDirection::Ascending => {
                    permutation.sort_by_key(|&position| key_ranks[base + position * inner_stride]);
                }
                SortDirection::Descending => {
                    permutation.sort_by_key(|&position| std::cmp::Reverse(key_ranks[base + position * inner_stride]));
                }
            }
            for (operand, output) in operand_values.iter().zip(outputs.iter_mut()) {
                for (target_position, &source_position) in permutation.iter().enumerate() {
                    output[base + target_position * inner_stride] =
                        operand[base + source_position * inner_stride].clone();
                }
            }
        }
    }
    outputs
}

/// Value-level top-k capability, selecting the `k` largest elements along one axis together with their indices.
/// [`TopK`] is not a primitive operation: it is provided for every sortable, sliceable value as a stable descending
/// [`Sort`] of the value and an index [`iota`](IotaOperation) followed by a [`Slice`] of the leading `k` entries.
/// This staged form is exactly the sort-plus-slice idiom that XLA's top-k rewriter recognizes and replaces with its
/// fast top-k implementation. Ties select the lowest index (the sort is stable), NaNs order above `+∞` (the IEEE
/// 754 total order), and the returned indices are `i32`.
pub trait TopK: Sized {
    /// Returns the `k` largest elements of this value along `axis` together with their `i32` indices, both with the
    /// `axis` dimension resized to `k`, returning a [`ProgramError`] if something goes wrong.
    fn top_k(&self, k: usize, axis: usize) -> Result<(Self, Self), ProgramError>;
}

impl<V: Value<Type = ArrayType> + Sort + Slice> TopK for V
where
    V::DispatchDomain: Context<Operation: From<IotaOperation<ArrayType>>>,
{
    fn top_k(&self, k: usize, axis: usize) -> Result<(Self, Self), ProgramError> {
        let (indices, dimensions) = sorted_index_passenger(self, axis)?;
        top_k_from_index_passenger(self, indices, dimensions.as_slice(), k, axis)
    }
}

/// Selects the `k` leading entries of a descending ranking sort with a prebuilt index passenger. This is the shared
/// composition behind every [`TopK`] implementation: the blanket implementation stages the index passenger as an
/// [`IotaOperation`], while concrete eager backends materialize it directly.
pub(crate) fn top_k_from_index_passenger<V: Clone + Sort + Slice>(
    value: &V,
    indices: V,
    dimensions: &[usize],
    k: usize,
    axis: usize,
) -> Result<(V, V), ProgramError> {
    if k > dimensions[axis] {
        return Err(ProgramError::UnsupportedOperation {
            message: format!("'top_k' k {k} exceeds axis {axis} size {}", dimensions[axis]),
        });
    }
    let mut sorted = Sort::sort(&[value.clone(), indices], axis, SortDirection::Descending)?;
    let sorted_indices = sorted.remove(1);
    let sorted_values = sorted.remove(0);
    let start_indices = vec![0; dimensions.len()];
    let mut limit_indices = dimensions.to_vec();
    limit_indices[axis] = k;
    let strides = vec![1; dimensions.len()];
    Ok((
        sorted_values.slice(start_indices.as_slice(), limit_indices.as_slice(), strides.as_slice())?,
        sorted_indices.slice(start_indices.as_slice(), limit_indices.as_slice(), strides.as_slice())?,
    ))
}

/// Value-level argmax capability, computing the index of the largest element along one axis. [`ArgMax`] is not a
/// primitive operation: it is a [`top_k`](TopK::top_k)-style descending [`Sort`] with an index passenger, sliced to
/// the leading entry and reshaped to drop the reduced axis. Ties select the lowest index (the sort is stable) and
/// NaNs order above `+∞`, so an axis containing a NaN reports the NaN's index, matching
/// [`jnp.argmax`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argmax.html). The returned indices are
/// `i32`.
pub trait ArgMax: Sized {
    /// Returns the `i32` indices of the largest elements of this value along `axis`, with that axis dropped from
    /// the result shape, returning a [`ProgramError`] if something goes wrong.
    fn argmax(&self, axis: usize) -> Result<Self, ProgramError>;
}

impl<V: Value<Type = ArrayType> + Sort + Slice + Reshape> ArgMax for V
where
    V::DispatchDomain: Context<Operation: From<IotaOperation<ArrayType>>>,
{
    fn argmax(&self, axis: usize) -> Result<Self, ProgramError> {
        let (indices, dimensions) = sorted_index_passenger(self, axis)?;
        extremal_index_from_index_passenger(self, indices, dimensions.as_slice(), axis, SortDirection::Descending)
    }
}

/// Value-level argmin capability, computing the index of the smallest element along one axis. Refer to the
/// documentation of [`ArgMax`] for the composition and the tie and NaN policies (for [`ArgMin`], NaNs order below
/// `-∞` in the ascending total order, so an axis containing a `-NaN`-signed NaN reports its index).
pub trait ArgMin: Sized {
    /// Returns the `i32` indices of the smallest elements of this value along `axis`, with that axis dropped from
    /// the result shape, returning a [`ProgramError`] if something goes wrong.
    fn argmin(&self, axis: usize) -> Result<Self, ProgramError>;
}

impl<V: Value<Type = ArrayType> + Sort + Slice + Reshape> ArgMin for V
where
    V::DispatchDomain: Context<Operation: From<IotaOperation<ArrayType>>>,
{
    fn argmin(&self, axis: usize) -> Result<Self, ProgramError> {
        let (indices, dimensions) = sorted_index_passenger(self, axis)?;
        extremal_index_from_index_passenger(self, indices, dimensions.as_slice(), axis, SortDirection::Ascending)
    }
}

/// Stages the `i32` index iota that rides a ranking sort as a passenger operand, and returns it together with the
/// operand's static dimensions.
fn sorted_index_passenger<V: Value<Type = ArrayType> + Sort>(
    value: &V,
    axis: usize,
) -> Result<(V, Vec<usize>), ProgramError>
where
    V::DispatchDomain: Context<Operation: From<IotaOperation<ArrayType>>>,
{
    let value_type = value.r#type();
    let dimensions = value_type
        .shape()
        .dimensions()
        .iter()
        .map(|size| match size {
            Size::Static(size) => Ok(*size),
            Size::Dynamic(_) => Err(ProgramError::UnsupportedOperation {
                message: "ranking operations do not support dynamic dimensions".to_string(),
            }),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut iota_type = ArrayType::new(DataType::I32, Shape::new(value_type.shape().dimensions().to_vec()));
    if let Some(sharding) = value_type.sharding() {
        iota_type = iota_type
            .with_sharding(sharding.clone())
            .map_err(|error| ProgramError::from(TypeError { message: error.to_string() }))?;
    }
    let indices = value.dispatch_domain().bind(IotaOperation::new(iota_type, axis), Vec::new(), &[])?.remove(0);
    Ok((indices, dimensions))
}

/// Computes the extremal-element index along `axis` shared by every [`ArgMax`] and [`ArgMin`] implementation: a
/// stable ranking sort with a prebuilt index passenger, sliced to the leading entry and reshaped to drop the
/// reduced axis.
pub(crate) fn extremal_index_from_index_passenger<V: Clone + Sort + Slice + Reshape>(
    value: &V,
    indices: V,
    dimensions: &[usize],
    axis: usize,
    direction: SortDirection,
) -> Result<V, ProgramError> {
    let mut sorted = Sort::sort(&[value.clone(), indices], axis, direction)?;
    let sorted_indices = sorted.remove(1);
    let start_indices = vec![0; dimensions.len()];
    let mut limit_indices = dimensions.to_vec();
    limit_indices[axis] = 1;
    let strides = vec![1; dimensions.len()];
    let leading = sorted_indices.slice(start_indices.as_slice(), limit_indices.as_slice(), strides.as_slice())?;
    let output_dimensions = dimensions
        .iter()
        .enumerate()
        .filter_map(|(dimension, &size)| (dimension != axis).then_some(Size::Static(size)))
        .collect::<Vec<_>>();
    leading.reshape(Shape::new(output_dimensions))
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::tracing::{DomainTracer, Trace};

    use super::*;

    /// Returns the static `f64` vector type of the provided length used throughout these tests.
    fn vector_type(length: usize) -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(length)]))
    }

    #[test]
    fn test_sort() {
        let operation = SortOperation::new(0, SortDirection::Ascending);
        assert_eq!(operation.axis(), 0);
        assert_eq!(operation.direction(), SortDirection::Ascending);
        assert_eq!(operation.name(), SORT_OPERATION_NAME);
        assert_eq!(operation.to_string(), "sort [axis=0, direction=ascending]");
        assert_eq!(SortOperation::new(1, SortDirection::Descending).to_string(), "sort [axis=1, direction=descending]",);

        // An ascending key-value sort co-permutes the passenger by the key's order, and the sort is stable: both
        // `3.0` keys keep their original relative order, so the first one's payload `10.0` precedes `30.0`.
        let keys = Array::vector(vec![3.0, 1.0, 3.0, 2.0]);
        let payload = Array::vector(vec![10.0, 20.0, 30.0, 40.0]);
        let outputs = InterpretableOperation::<EagerContext<Array>>::interpret(
            &operation,
            &EagerContext::new(),
            &EmptyRegionDriver,
            &[keys.clone(), payload.clone()],
        )
        .unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], Array::vector(vec![1.0, 2.0, 3.0, 3.0]));
        assert_eq!(outputs[1], Array::vector(vec![20.0, 40.0, 10.0, 30.0]));

        // Descending reverses the key comparison while keeping equal keys in their original order.
        let outputs = InterpretableOperation::<EagerContext<Array>>::interpret(
            &SortOperation::new(0, SortDirection::Descending),
            &EagerContext::new(),
            &EmptyRegionDriver,
            &[keys, payload],
        )
        .unwrap();
        assert_eq!(outputs[0], Array::vector(vec![3.0, 3.0, 2.0, 1.0]));
        assert_eq!(outputs[1], Array::vector(vec![10.0, 30.0, 40.0, 20.0]));

        // A one-operand sort stages as a single-output instruction.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(vector_type(4));
        let outputs = builder.add_instruction(operation, Vec::new(), vec![input]).unwrap().to_vec();
        let program = builder.build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[4] .
                let %1:f64[4] = sort [axis=0, direction=ascending] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_sort_type_inference() {
        let operation = SortOperation::new(0, SortDirection::Ascending);
        let complex = ArrayType::new(DataType::C64, Shape::new(vec![Size::Static(4)]));
        // Sort operands only need to agree on shape; passenger element types pass through unchanged.
        let passenger = ArrayType::new(DataType::I32, Shape::new(vec![Size::Static(4)]));
        check_operation_type_inference!(
            operation = operation,
            cases = [
                {
                    type = ArrayType,
                    input_types = [],
                    error = "'sort' needs at least one input",
                },
                {
                    input_types = [complex],
                    error = "'sort' does not support key data type c64",
                },
                {
                    input_types = [vector_type(4), vector_type(3)],
                    error = "'sort' operands must agree on shape but got [4] and [3]",
                },
                {
                    input_types = [vector_type(4), passenger.clone()],
                    output_types = [vector_type(4), passenger],
                },
            ],
        );
        check_operation_type_inference!(
            operation = SortOperation::new(1, SortDirection::Ascending),
            cases = [{
                input_types = [vector_type(4)],
                error = "'sort' axis 1 is out of bounds for rank 1",
            }],
        );
    }

    #[test]
    fn test_sort_multi_axis() {
        let input = Array::matrix(2, 3, vec![3.0, 1.0, 2.0, 0.0, 5.0, 4.0]);
        // Axis 0 sorts every column independently.
        let sorted = Sort::sort(std::slice::from_ref(&input), 0, SortDirection::Ascending).unwrap().remove(0);
        assert_eq!(sorted, Array::matrix(2, 3, vec![0.0, 1.0, 2.0, 3.0, 5.0, 4.0]));
        // Axis 1 sorts every row independently.
        let sorted = Sort::sort(std::slice::from_ref(&input), 1, SortDirection::Ascending).unwrap().remove(0);
        assert_eq!(sorted, Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0]));
    }

    #[test]
    fn test_sort_total_order() {
        // Floating-point keys follow the IEEE 754 total order: `-∞ < -0.0 < +0.0 < NaN`.
        let input = Array::vector(vec![f64::NAN, -0.0, 0.0, f64::NEG_INFINITY]);
        let sorted = Sort::sort(&[input], 0, SortDirection::Ascending).unwrap().remove(0);
        let values = sorted.to_f64s();
        assert_eq!(values[0], f64::NEG_INFINITY);
        assert_eq!(values[1].to_bits(), (-0.0f64).to_bits());
        assert!(values[1].is_sign_negative());
        assert_eq!(values[2].to_bits(), 0.0f64.to_bits());
        assert!(values[3].is_nan());
    }

    #[test]
    fn test_sort_batching() {
        // A mapped key moves its batch axis to the leading position and the sort axis lifts past it, so each batch
        // item's vector is sorted independently.
        check_operation_batching!(
            @exact,
            operation = SortOperation::new(0, SortDirection::Ascending),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::matrix(2, 2, vec![3.0, 1.0, 2.0, 5.0]))],
                outputs = [(@mapped(axis = 0), Array::matrix(2, 2, vec![1.0, 3.0, 2.0, 5.0]))],
            }],
        );
        // A replicated passenger broadcasts to the batched physical shape and is co-permuted per batch item by its
        // item's key order: item 0's keys reverse, item 1's keys are already sorted.
        check_operation_batching!(
            @exact,
            operation = SortOperation::new(0, SortDirection::Ascending),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::matrix(2, 2, vec![3.0, 1.0, 2.0, 5.0])),
                    (@replicated, Array::vector(vec![7.0, 8.0])),
                ],
                outputs = [
                    (@mapped(axis = 0), Array::matrix(2, 2, vec![1.0, 3.0, 2.0, 5.0])),
                    (@mapped(axis = 0), Array::matrix(2, 2, vec![8.0, 7.0, 7.0, 8.0])),
                ],
            }],
        );
    }

    #[test]
    fn test_sort_differentiation() {
        // The tangent rides the staged sort as a passenger operand, so it is co-permuted by the primal key's order.
        check_operation_differentiation!(
            @approx(step = 1e-3, epsilon = 1e-6),
            operation = SortOperation::new(0, SortDirection::Ascending),
            cases = [{
                primals = [Array::vector(vec![3.0, 1.0, 2.0])],
                tangents = [Array::vector(vec![30.0, 10.0, 20.0])],
                primal_outputs = [Array::vector(vec![1.0, 2.0, 3.0])],
                tangent_outputs = [Array::vector(vec![10.0, 20.0, 30.0])],
                jvp = indoc! {"
                    lambda %0:f64[3], %1:f64[3] .
                    let %2:f64[3], %3:f64[3] = sort [axis=0, direction=ascending] %0 %1
                    in (%2, %3)
                "},
            }],
        );
    }

    #[test]
    fn test_sort_partial_evaluation() {
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = SortOperation::new(0, SortDirection::Ascending),
            cases = [
                {
                    inputs = [(@known, Array::vector(vec![3.0, 1.0, 2.0]))],
                    outputs = [(@known, Array::vector(vec![1.0, 2.0, 3.0]))],
                    residual_instructions = 0,
                },
                {
                    inputs = [(@unknown(
                        type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)])),
                        replay = Array::vector(vec![3.0, 1.0, 2.0])
                    ))],
                    outputs = [(@residual, Array::vector(vec![1.0, 2.0, 3.0]))],
                    residual_instructions = 1,
                },
            ],
        );
    }

    #[test]
    fn test_sort_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = SortOperation::new(0, SortDirection::Ascending),
            input_types = [ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]))],
        );
    }

    #[test]
    fn test_top_k() {
        let input = Array::vector(vec![3.0, 1.0, 3.0, -0.0, 0.0, 2.0]);
        let (values, indices) = input.top_k(3, 0).unwrap();
        assert_eq!(values, Array::vector(vec![3.0, 3.0, 2.0]));
        // Ties select the lowest index first because the descending ranking sort is stable.
        assert_eq!(
            indices,
            Array::new(
                ArrayType::new(DataType::I32, Shape::new(vec![Size::Static(3)])),
                vec![Scalar::I32(0), Scalar::I32(2), Scalar::I32(5)],
            )
            .unwrap(),
        );
        assert!(matches!(
            input.top_k(7, 0),
            Err(ProgramError::UnsupportedOperation { message }) if message == "'top_k' k 7 exceeds axis 0 size 6",
        ));
    }

    #[test]
    fn test_argmax_and_argmin() {
        /// Returns the expected `i32` index array of the provided static dimensions.
        fn index_array(dimensions: Vec<usize>, indices: Vec<i32>) -> Array {
            let shape = Shape::new(dimensions.into_iter().map(Size::Static).collect());
            Array::new(ArrayType::new(DataType::I32, shape), indices.into_iter().map(Scalar::I32).collect()).unwrap()
        }

        // NaN orders above `+∞` in the descending total order, so `argmax` reports the NaN's index, while `argmin`
        // reports the smallest ordinary value's index (a positive NaN orders last ascending as well).
        let with_nan = Array::vector(vec![1.0, f64::NAN, 3.0]);
        assert_eq!(with_nan.argmax(0), Ok(index_array(vec![], vec![1])));
        assert_eq!(with_nan.argmin(0), Ok(index_array(vec![], vec![0])));

        // Ties select the lowest index because the ranking sort is stable.
        let tie = Array::vector(vec![2.0, 2.0]);
        assert_eq!(tie.argmax(0), Ok(index_array(vec![], vec![0])));
        assert_eq!(tie.argmin(0), Ok(index_array(vec![], vec![0])));

        // The reduced axis is dropped from the result shape, and the indices are `i32`.
        let matrix = Array::matrix(2, 3, vec![1.0, 5.0, 3.0, 4.0, 0.0, 2.0]);
        assert_eq!(matrix.argmax(0), Ok(index_array(vec![3], vec![1, 0, 0])));
        assert_eq!(matrix.argmax(1), Ok(index_array(vec![2], vec![1, 0])));
        assert_eq!(matrix.argmin(0), Ok(index_array(vec![3], vec![0, 1, 1])));
        assert_eq!(matrix.argmin(1), Ok(index_array(vec![2], vec![0, 1])));
    }

    #[test]
    fn test_top_k_stages_through_the_tracer_capability() {
        // Staging `top_k` on a tracer composes the sort-plus-slice idiom that XLA's top-k rewriter recognizes: an
        // index iota rides a descending sort as a passenger and both outputs are sliced to the leading `k` entries.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok(x.top_k(2, 0)?.0),
            vector_type(4),
        )
        .unwrap();
        let program = program.to_flat_program();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[4] .
                let %1:i32[4] = iota [type=i32[4], dimension=0]
                    %2:f64[4], %3:i32[4] = sort [axis=0, direction=descending] %0 %1
                    %4:f64[2] = slice [start_indices=[0], limit_indices=[2]] %2
                    %5:i32[2] = slice [start_indices=[0], limit_indices=[2]] %3
                in (%4)
            "}
            .trim_end(),
        );
    }
}
