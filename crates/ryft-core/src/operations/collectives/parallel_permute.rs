//! Contains the named-axis [`ParallelPermuteOperation`], which sends every participant's operand to another
//! participant along a named axis, together with its interpretation, partial-evaluation, batching, forward-mode
//! differentiation, and transposition rules.

// TODO(eaplatanios): Review this module.

use std::fmt::Display;

use crate::arrays::{ArrayBatch, ArrayBatching, ArrayBatchingPolicy, ArrayIrType, ArrayType};
use crate::axes::NamedAxes;
use crate::batching::{BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    TransposableOperation, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::constants::zero_like::ZeroLike;
use crate::operations::manipulation::concatenation::Concatenate;
use crate::operations::manipulation::slicing::Slice;
use crate::operations::manipulation::transposition::Transpose;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, ProgramError, ProjectedValue, RegionInterface, TypeError, Typed, Value,
    ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

use super::{
    forward_collective_to_parent, interpret_degenerate_collective, reject_ragged_collective_inputs,
    resolve_named_axis_size, shape_changing_collective, shape_changing_collective_dimensions,
    shape_changing_collective_output_type, transpose_shape_changing_collective, validate_collective_axis_size,
};

shape_changing_collective! {
    /// [`Operation`] that sends every participant's operand to another participant along the named axis according
    /// to explicit `(source, target)` pairs — the analogue of
    /// [JAX's `ppermute`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ppermute.html) and
    /// [StableHLO's `collective_permute`](https://openxla.org/stablehlo/spec#collective_permute). Participants that
    /// no pair targets receive zeros. The output shape is unchanged. The collective is linear and its transpose is
    /// the permutation with every pair inverted. A matching `batch` level consumes the mapped batch axis by
    /// reassembling it in target order from per-item slices, with zero slices at untargeted positions.
    operation = ParallelPermuteOperation,
    name = PARALLEL_PERMUTE_OPERATION_NAME = "parallel_permute",
    /// Value-level entry point for staging a [`ParallelPermuteOperation`]. Refer to its documentation for the semantics
    /// and transform rules.
    capability = ParallelPermute::parallel_permute,
    fields = {
        /// Pairs of `(source, target)` positions along the named axis: the value of participant `source` is sent to
        /// participant `target`.
        source_target_pairs: Vec<(usize, usize)>,
    },
    infer = |operation, input_type, dimensions| {
        let mut seen_sources = std::collections::BTreeSet::new();
        let mut seen_targets = std::collections::BTreeSet::new();
        for (source, target) in &operation.source_target_pairs {
            if *source >= operation.axis_size || *target >= operation.axis_size {
                return Err(TypeError::invalid(format!(
                        "`parallel_permute` pair ({source}, {target}) is out of bounds for axis size {}",
                        operation.axis_size,
                    )));
            }
            if !seen_sources.insert(*source) || !seen_targets.insert(*target) {
                return Err(TypeError::invalid(format!(
                        "`parallel_permute` pairs must have unique sources and targets but ({source}, {target}) \
                         repeats one",
                    )));
            }
        }
        shape_changing_collective_output_type(PARALLEL_PERMUTE_OPERATION_NAME, input_type, dimensions)
    },
}

impl ParallelPermuteOperation {
    /// Returns the `(source, target)` pairs of participant positions along the named axis.
    #[inline]
    pub fn source_target_pairs(&self) -> &[(usize, usize)] {
        self.source_target_pairs.as_slice()
    }
}

/// Convenience permutation encoded as the source participant selected for each output participant.
pub trait ParallelShuffle: Sized {
    /// Permutes a named axis using `permutation[output] = input` encoding.
    fn parallel_shuffle(&self, axis_name: &str, permutation: &[usize]) -> Result<Self, ProgramError>;
}

impl<V> ParallelShuffle for V
where
    V: Value<Type = ArrayIrType>,
    V::DispatchDomain: Context<Type = ArrayIrType> + NamedAxes,
    <V::DispatchDomain as Domain>::Operation: From<ParallelPermuteOperation>,
{
    fn parallel_shuffle(&self, axis_name: &str, permutation: &[usize]) -> Result<Self, ProgramError> {
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        if permutation.len() != axis_size {
            return Err(TypeError::invalid(format!(
                "`parallel_shuffle` permutation length {} must equal axis size {axis_size}",
                permutation.len(),
            ))
            .into());
        }
        let mut seen = vec![false; axis_size];
        for &source in permutation {
            let Some(source_seen) = seen.get_mut(source) else {
                return Err(TypeError::invalid(format!(
                    "`parallel_shuffle` source index {source} is out of bounds for axis size {axis_size}",
                ))
                .into());
            };
            if *source_seen {
                return Err(TypeError::invalid(format!(
                    "`parallel_shuffle` permutation contains source index {source} more than once",
                ))
                .into());
            }
            *source_seen = true;
        }
        Ok(context
            .bind(
                ParallelPermuteOperation::new(
                    axis_name.to_string(),
                    axis_size,
                    permutation.iter().copied().zip(0..axis_size).collect(),
                ),
                Vec::new(),
                std::slice::from_ref(self),
            )?
            .remove(0))
    }
}

impl<V> ParallelShuffle for ProjectedValue<ArrayType, V>
where
    V: ParallelShuffle + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>,
{
    fn parallel_shuffle(&self, axis_name: &str, permutation: &[usize]) -> Result<Self, ProgramError> {
        self.value().parallel_shuffle(axis_name, permutation)?.into_projected().map_err(Into::into)
    }
}

// Batching rule for [`ParallelPermuteOperation`]. A matching `batch` level consumes the mapped batch axis by
// reassembling it in target order: for each position `t` along the batch axis, the output receives the slice of the
// source item that sends to `t`, or a zero slice when no pair targets `t`. A non-matching level forwards the
// collective untouched to the parent context via [`forward_collective_to_parent`].
impl<C, P: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>> for ParallelPermuteOperation
where
    C: Context<Type = ArrayType>,
    C::Operation: From<ParallelPermuteOperation>,
    <C as Domain>::Value: Concatenate + Slice + Transpose + ZeroLike,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<P>>, BatchingError> {
        reject_ragged_collective_inputs(self.name(), inputs)?;
        if context.axis_name() != Some(self.axis_name.as_str()) {
            return Ok(forward_collective_to_parent(context, C::Operation::from(self.clone()), inputs)?.into());
        }
        let [input] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
        };
        let input = P::match_axis(context, input, 0.into())?;
        let batch_size = P::axis_size(context)?;
        if batch_size != self.axis_size {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`parallel_permute` over axis `{}` resolved axis size {} but the mapped batch axis has size \
                     {batch_size}",
                    self.axis_name, self.axis_size,
                ),
            });
        }
        let Some(shape) = input.value().r#type().static_shape() else {
            return Err(BatchingError::UnsupportedOperation {
                message: "`parallel_permute` batching requires statically shaped operands".to_string(),
            });
        };
        let dimensions = shape.dimensions().to_vec();
        let value = input.into_value();
        // Map each target position along the batch axis to the source item that sends to it; positions that no pair
        // targets receive zeros. Pair uniqueness is enforced by output type inference, so it is not revalidated here.
        let mut sources = vec![None; batch_size];
        for (source, target) in &self.source_target_pairs {
            if *source >= batch_size || *target >= batch_size {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "`parallel_permute` pair ({source}, {target}) is out of bounds for axis size {batch_size}",
                    ),
                });
            }
            sources[*target] = Some(*source);
        }
        // Slice each item `[i, i + 1)` from the leading batch axis and concatenate the slices back in target order.
        let rank = dimensions.len();
        let strides = vec![1; rank];
        let slice_item = |item: usize| -> Result<<C as Domain>::Value, ProgramError> {
            let mut start_indices = vec![0; rank];
            let mut limit_indices = dimensions.clone();
            start_indices[0] = item;
            limit_indices[0] = item + 1;
            value.slice(&start_indices, &limit_indices, &strides)
        };
        let mut zero_item = None;
        let mut items = Vec::with_capacity(batch_size);
        for source in sources {
            match source {
                Some(source) => items.push(slice_item(source)?),
                None => {
                    if zero_item.is_none() {
                        zero_item = Some(slice_item(0)?.zero_like());
                    }
                    // The zero slice was materialized right above when absent.
                    items.push(zero_item.clone().unwrap());
                }
            }
        }
        let permuted = Concatenate::concatenate(&items, 0)?;
        Ok(vec![ArrayBatch::new(permuted, Some(0))?].into())
    }
}

// Transpose rule for [`ParallelPermuteOperation`]: sending along `(source, target)` pulls cotangents back along
// `(target, source)`, so the operand cotangent is the permutation with every pair inverted.
impl<V, O> TransposableOperation<V, O> for ParallelPermuteOperation
where
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType> + From<ParallelPermuteOperation>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        let inverted_pairs =
            self.source_target_pairs.iter().map(|(source, target)| (*target, *source)).collect::<Vec<_>>();
        transpose_shape_changing_collective(
            context,
            inputs,
            outputs,
            ParallelPermuteOperation::new(self.axis_name.clone(), self.axis_size, inverted_pairs),
        )
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatching, ArrayIrOperation, ArrayIrValue, ArrayOperation, DataType, Dimension,
        DimensionValue, Shape,
    };
    use crate::batching::{BatchAxis, BatchAxisSpecification, BatchingContext, BatchingTracer, batch};
    use crate::contexts::EagerContext;
    use crate::operations::collectives::tests::f32_vector;

    use super::*;

    #[test]
    fn test_parallel_shuffle_composes_parallel_permute_in_the_composite_domain() {
        type Parent = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let context = BatchingContext::<_, ArrayIrBatching>::new(
            Parent::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
        )
        .with_axis_name("x".to_string());
        let input = ArrayIrValue::Array(Array::matrix(3, 2, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let input = ArrayIrBatch::new(input, BatchAxis::new(0)).unwrap();
        let input = BatchingTracer::new(context, input);
        let output = input.parallel_shuffle("x", &[2, 0, 1]).unwrap().into_batch();

        assert_eq!(output.batch_axis(), BatchAxis::new(0));
        let ArrayIrValue::Array(output) = output.into_value() else {
            panic!("parallel_shuffle must preserve the array member kind");
        };
        assert_eq!(output.to_f64s(), vec![5.0, 6.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_parallel_permute_type_inference() {
        use crate::macros::check_operation_type_inference;

        check_operation_type_inference!(
            operation = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1), (1, 0)]),
            cases = [{
                input_types = [f32_vector(3)],
                output_types = [f32_vector(3)],
            }],
        );
        check_operation_type_inference!(
            operation = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 2)]),
            cases = [{
                input_types = [f32_vector(3)],
                error = "`parallel_permute` pair (0, 2) is out of bounds for axis size 2",
            }],
        );
        check_operation_type_inference!(
            operation = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1), (0, 0)]),
            cases = [{
                input_types = [f32_vector(3)],
                error = "`parallel_permute` pairs must have unique sources and targets but (0, 0) repeats one",
            }],
        );
    }

    #[test]
    fn test_parallel_permute_over_batched_axis_permutes_the_items() {
        use crate::batching::BatchingTracer;

        // The rotation `[(0, 1), (1, 0)]` swaps the two batch items: item 0 receives item 1's `[3, 4]` and item 1
        // receives item 0's `[1, 2]`.
        let output: Array = batch(
            |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatching>| {
                item.parallel_permute("x", vec![(0, 1), (1, 0)])
            },
            Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
        );
        assert_eq!(output.to_f64s(), vec![3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn test_parallel_permute_over_batched_axis_zeros_untargeted_items() {
        use crate::batching::BatchingTracer;

        // With the single pair `(0, 1)`, item 1 receives item 0's `[1, 2]` while no pair targets item 0, so it
        // receives zeros, matching JAX's `ppermute` semantics for untargeted participants.
        let output: Array = batch(
            |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatching>| {
                item.parallel_permute("x", vec![(0, 1)])
            },
            Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(output.to_f64s(), vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_parallel_permute_transposes_to_inverted_pairs() {
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;

        // Sending along `(source, target)` pulls cotangents back along `(target, source)`, so the pullback stages the
        // permutation with every pair inverted.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(f32_vector(2));
        let output = builder
            .add_instruction(
                ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1), (1, 0)]),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let pullback = program.transpose_with_respect_to(&[0]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc::indoc! {r#"
                lambda %0:f32[2] .
                let %1:f32[2] = parallel_permute [axis_name="x", axis_size=2, source_target_pairs=[(1, 0), (0, 1)]] %0
                in (%1)
            "#}
            .trim_end(),
        );
    }
}
