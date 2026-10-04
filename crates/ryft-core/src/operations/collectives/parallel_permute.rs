use std::collections::BTreeSet;

use crate::arrays::{
    ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayOperation,
    ArrayType, MeshAxisType, RaggedAxis,
};
use crate::axes::{NamedAxes, NamedAxis};
use crate::batching::{BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain};
use crate::macros::check_count;
use crate::operations::collectives::parallel_vary::{PARALLEL_VARY_OPERATION_NAME, ParallelVary};
use crate::operations::collectives::{
    define_linear_collective_operation, impl_differentiable_linear_collective_operation,
    infer_linear_collective_operation_output_type, resolve_named_axis_size,
};
use crate::operations::constants::zero_like::ZeroLike;
use crate::operations::manipulation::concatenation::Concatenate;
use crate::operations::manipulation::slicing::Slice;
use crate::operations::manipulation::transposition::Transpose;
use crate::programs::{Operation, ProgramError, ProjectedValue, TypeError, Typed, Value, ValueProjection};

/// Canonical operation name for [`ParallelPermuteOperation`].
pub const PARALLEL_PERMUTE_OPERATION_NAME: &str = "parallel_permute";

define_linear_collective_operation!(
    /// [`Operation`] that routes input arrays between positions along the named axis according to explicit
    /// `(source, target)` pairs. Both indices are zero-based coordinates along `axis_name`, in `0..axis_size`, rather
    /// than global device IDs or element indices within an input array. Each position represents one execution of the
    /// enclosing function with its own input array. A pair sends the entire input array at `source` to the output at
    /// `target`. Sources must be unique, targets must be unique, and positions that no pair targets receive zeros.
    /// The output type is the input type.
    ///
    /// For example, inside `shard_map` over a manual mesh axis `x` of size three, `(0, 2)` sends the input at mesh
    /// coordinate `x = 0` to the output at `x = 2`. On a multidimensional mesh, this routing is repeated separately
    /// for each fixed combination of coordinates along the other axes. The physical devices at these coordinates can
    /// have arbitrary device IDs; the pair still uses axis coordinates `0` and `2`.
    ///
    /// Inside [`batch`](crate::batch) over a named axis of size three, `(0, 2)` instead sends batch item zero's input
    /// array to batch item two's output. For either interpretation, if positions zero, one, and two hold arrays `A`,
    /// `B`, and `C`, then pairs `[(0, 1), (2, 0)]` produce `C`, `A`, and a zero array at those respective positions.
    ///
    /// This is the Ryft analogue of JAX's
    /// [`jax.lax.ppermute`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ppermute.html) and StableHLO's
    /// [`collective_permute`](https://openxla.org/stablehlo/spec#collective_permute).
    ///
    /// Over a manual mesh axis, a permutation generally gives the axis positions different values, so the input must
    /// vary over the axis (refer to [`ParallelVary`]) and the output varies over it as well. The collective is linear
    /// and its transpose is the permutation with every pair inverted. Outside any binder, the single position of a
    /// degenerate axis keeps its value when the pair `(0, 0)` is present and receives zeros otherwise.
    ///
    /// A matching `batch` level consumes the mapped batch axis by reassembling it in target order from per-item slices,
    /// with zero slices at untargeted positions, and passes a replicated input through unchanged when every position is
    /// targeted. Unlike JAX's batching rule, which requires a full permutation, partial permutations are supported.
    /// Bounded ragged extents follow the same source-to-target routing as their packed values, and untargeted
    /// positions receive zero extents together with their zero-filled values.
    ParallelPermuteOperation,
    PARALLEL_PERMUTE_OPERATION_NAME,
    fields = {
        /// Pairs of `(source, target)` positions along the named axis. For each pair, the value of participant `source`
        /// is sent to participant `target`.
        source_target_pairs: Vec<(usize, usize)>,
    },
    check_array_types = [@no_unreduced],
    infer_output_type = |operation, input_type, dimensions| {
        let mut sources = BTreeSet::new();
        let mut targets = BTreeSet::new();
        for &(source, target) in &operation.source_target_pairs {
            if source >= operation.axis_size || target >= operation.axis_size {
                return Err(TypeError::invalid(format!(
                    "`{}` pair ({}, {}) is out of bounds for axis size {}",
                    PARALLEL_PERMUTE_OPERATION_NAME,
                    source,
                    target,
                    operation.axis_size,
                )));
            }
            if !sources.insert(source) || !targets.insert(target) {
                return Err(TypeError::invalid(format!(
                    "`{PARALLEL_PERMUTE_OPERATION_NAME}` pairs must have unique sources and targets but \
                     ({source}, {target}) repeats one",
                )));
            }
        }

        // A permutation over a manual mesh axis gives the participants different values, so an input that is still
        // invariant over that axis would yield an output whose type wrongly claims that it is invariant.
        if let Some(sharding) = input_type.sharding()
            && sharding.mesh().axis_type(&operation.axis_name) == Some(MeshAxisType::Manual)
            && !sharding.varying_manual_axes().contains(&operation.axis_name)
        {
            return Err(TypeError::invalid(format!(
                "`{}` input must vary over manual axis `{}`; pass an invariant value \
                 through `{}` first so that the permuted output is typed as varying",
                PARALLEL_PERMUTE_OPERATION_NAME,
                operation.axis_name,
                PARALLEL_VARY_OPERATION_NAME,
            )));
        }

        infer_linear_collective_operation_output_type(PARALLEL_PERMUTE_OPERATION_NAME, input_type, dimensions)
    },
    interpret<C> where C::Value: ZeroLike {
        |operation, input| {
            // With one participant, the only valid pairs are none at all and `(0, 0)`.
            // Without a pair, nothing targets the participant, so it receives zeros.
            if operation.source_target_pairs.is_empty() { input.zero_like() } else { Ok(input.clone()) }
        }
    },
);

impl ParallelPermuteOperation {
    /// Returns the `(source, target)` pairs of participant positions along the named axis.
    #[inline]
    pub fn source_target_pairs(&self) -> &[(usize, usize)] {
        self.source_target_pairs.as_slice()
    }
}

impl<
    C: Context<
            Type = ArrayType,
            Value: ZeroLike + Concatenate + Slice + Transpose,
            Operation: From<ParallelPermuteOperation>,
        >,
    P: ArrayExtentBatchingPolicy<C>,
> BatchableOperation<C, ArrayBatchingPolicy<P>> for ParallelPermuteOperation
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        // A matching `batch` level consumes the mapped batch axis by reassembling it in target order: for each position
        // `t` along the batch axis, the output receives the slice of the source item that sends to `t`, or a zero slice
        // when no pair targets `t`. A non-matching level forwards the collective untouched to the parent context.
        if context.axis_name() != Some(self.axis_name.as_str()) {
            ArrayBatch::reject_ragged_inputs(self, inputs)?;
            return Ok(context.forward_to_parent(C::Operation::from(self.clone()), inputs)?.into());
        }

        let [input] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
        };

        // A batching rule runs before any context infers the operation's output type, so the operation is validated
        // here, against the physical (i.e., padded) item type, before its pairs index the batch axis.
        self.infer_output_types(&[input.value().r#type().unbatched(input.batch_axis())?], &[])?;
        let batch_size = P::axis_size(context)?;
        if batch_size != self.axis_size {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{}` over axis `{}` resolved axis size {} but the mapped batch axis has size {}",
                    PARALLEL_PERMUTE_OPERATION_NAME, self.axis_name, self.axis_size, batch_size,
                ),
            });
        }

        // Map each target position along the batch axis to the source item that sends to it. The pairs were validated
        // against the axis size above, which equals the batch size.
        let mut sources = vec![None; batch_size];
        for &(source, target) in &self.source_target_pairs {
            sources[target] = Some(source);
        }
        let has_untargeted_participant = sources.iter().any(Option::is_none);

        // A replicated input holds the same value for every batch item, so a full permutation leaves it unchanged.
        if !has_untargeted_participant && input.batch_axis_position().is_none() {
            return Ok(vec![input.clone()].into());
        }

        if has_untargeted_participant
            && let Some(ragged_axis) =
                input.ragged_axes().iter().find(|ragged_axis| ragged_axis.dimension().bounds().lower() != 0)
        {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{}` cannot assign a zero extent to bounded ragged dimension `{}` whose lower bound is {}",
                    PARALLEL_PERMUTE_OPERATION_NAME,
                    ragged_axis.dimension(),
                    ragged_axis.dimension().bounds().lower(),
                ),
            });
        }

        let input = P::match_axis(context, input, 0.into())?;
        let permuted = route_axis_slices(input.value(), 0, sources.as_slice())?;
        let ragged_axes = input
            .ragged_axes()
            .iter()
            .map(|ragged_axis| {
                // Extents that do not already vary over the participant axis must first acquire that axis so an
                // untargeted participant receives zero metadata together with its zero-filled packed value.
                let mut extent_axes = ragged_axis.extent_axes().to_vec();
                let extents = if let Some(extent_axis) = extent_axes.iter().position(|axis| *axis == 0) {
                    route_axis_slices(ragged_axis.extents(), extent_axis, sources.as_slice())?
                } else if has_untargeted_participant {
                    let extents =
                        P::match_axis(context, &ArrayBatch::replicated(ragged_axis.extents().clone()), 0.into())?
                            .into_value();
                    extent_axes.insert(0, 0);
                    route_axis_slices(&extents, 0, sources.as_slice())?
                } else {
                    ragged_axis.extents().clone()
                };
                Ok(RaggedAxis::new(ragged_axis.axis(), extents, ragged_axis.dimension().clone(), extent_axes))
            })
            .collect::<Result<Vec<_>, BatchingError>>()?;
        Ok(vec![ArrayBatch::new(permuted, Some(0))?.with_ragged_axes(ragged_axes)?].into())
    }
}

impl_differentiable_linear_collective_operation! {
    ParallelPermuteOperation,
    transpose = |operation| -> ParallelPermuteOperation {
        // Sending along `(source, target)` pulls cotangents back along `(target, source)`, so the input cotangent
        // is the permutation with every pair inverted.
        ParallelPermuteOperation::new(
            operation.axis_name.clone(),
            operation.axis_size,
            operation.source_target_pairs.iter().map(|(source, target)| (*target, *source)).collect::<Vec<_>>(),
        )
    },
}

impl<A: Value<Type = ArrayType>> From<ParallelPermuteOperation> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: ParallelPermuteOperation) -> Self {
        Self::Array(ArrayOperation::ParallelPermute(operation))
    }
}

/// Represents the ability to permute values across the participants of a named axis. Refer to the documentation
/// of [`ParallelPermuteOperation`] for the semantics of this operation and its transformation rules.
///
/// # Example
///
/// Every batch item sends its row to the next item. No item sends to the first item, which receives zeros:
///
/// ```rust
/// # use ryft_core::{
/// #     Array, ArrayBatchingPolicy, ArrayOperation, BatchAxis, BatchAxisSpecification, BatchingTracer, EagerContext,
/// #     ParallelPermute, batch,
/// # };
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let rows = Array::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
/// let shifted = batch(
///     |row: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>| {
///         row.parallel_permute("rows", vec![(0, 1), (1, 2)])
///     },
///     rows,
///     BatchAxis::new(0),
///     BatchAxis::new(0),
///     BatchAxisSpecification::named("rows"),
/// )?;
/// assert_eq!(shifted, Array::matrix(3, 2, vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0])?);
/// # Ok(())
/// # }
/// ```
pub trait ParallelPermute: Sized {
    /// Returns this value permuted across the participants of the named axis `axis_name`. For every `(source, target)`
    /// pair, participant `target` receives the value of participant `source`, and every participant that no pair
    /// targets receives zeros. Over a manual mesh axis, an input that does not vary over the axis is first made
    /// varying through [`ParallelVary`], because the permuted participants generally hold different values.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of an axis bound by an enclosing `batch` level or manual region.
    ///   - `source_target_pairs`: Pairs of `(source, target)` positions along the named axis,
    ///     with unique sources and unique targets.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError::Axis`] error wrapping [`AxisError::UnboundAxisName`](crate::axes::AxisError) when no
    /// enclosing binder binds `axis_name`, and a [`ProgramError`] if a pair references a participant outside the axis,
    /// if two pairs share a source or a target, or if this value carries unreduced axes.
    fn parallel_permute(&self, axis_name: &str, source_target_pairs: Vec<(usize, usize)>)
    -> Result<Self, ProgramError>;
}

impl<
    V: Value<Type = ArrayType, DispatchDomain: Context<Operation: From<ParallelPermuteOperation>> + NamedAxes>
        + ParallelVary,
> ParallelPermute for V
{
    fn parallel_permute(
        &self,
        axis_name: &str,
        source_target_pairs: Vec<(usize, usize)>,
    ) -> Result<Self, ProgramError> {
        // Any context-carrying value permutes by resolving the axis size from the active `NamedAxes` environment and
        // binding a `ParallelPermuteOperation` through its own context. Over a manual mesh axis, an input that is still
        // invariant over the axis is first made varying, exactly as JAX's `ppermute` does, so that the output type
        // records that the participants hold different values.
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        let mut input = self.clone();
        if matches!(context.named_axis(axis_name), Some(NamedAxis::Mesh { .. }))
            && !input.r#type().sharding().is_some_and(|sharding| sharding.varying_manual_axes().contains(axis_name))
        {
            input = input.parallel_vary(axis_name)?;
        }
        let operation = ParallelPermuteOperation::new(axis_name.to_string(), axis_size, source_target_pairs);
        let mut outputs = context.bind(operation, Vec::new(), &[input])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Represents the ability to shuffle values across the participants of a named axis by listing,
/// for every output participant, the participant whose value it receives. This is the Ryft analogue of JAX's
/// [`jax.lax.pshuffle`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.pshuffle.html), and it stages a
/// [`ParallelPermuteOperation`] with the pair `(permutation[target], target)` for every target through
/// [`ParallelPermute`], whose manual-axis behavior therefore applies to shuffles as well.
pub trait ParallelShuffle: Sized {
    /// Returns this value shuffled across the participants of the named axis `axis_name`, where participant `target`
    /// receives the value of participant `permutation[target]`. A permutation shorter than the axis shuffles only the
    /// leading participants, and every remaining participant receives zeros.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of an axis bound by an enclosing `batch` level or manual region.
    ///   - `permutation`: Source participant of every output participant, which must be a permutation
    ///     of `0..permutation.len()`.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if `permutation` is not a permutation of `0..permutation.len()`, and any error
    /// of [`ParallelPermute::parallel_permute`] otherwise (e.g., when `permutation` is longer than the axis).
    fn parallel_shuffle(&self, axis_name: &str, permutation: &[usize]) -> Result<Self, ProgramError>;
}

impl<V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>>
    ParallelShuffle for V
where
    ProjectedValue<ArrayType, V>: ParallelShuffle,
{
    #[inline]
    fn parallel_shuffle(&self, axis_name: &str, permutation: &[usize]) -> Result<Self, ProgramError> {
        // A composite value shuffles through its array view, so that every shuffle shares the staging path
        // of `ParallelPermute`.
        Ok(V::from_projected(
            ValueProjection::<ArrayType>::into_projected(self.clone())?.parallel_shuffle(axis_name, permutation)?,
        ))
    }
}

impl<V> ParallelShuffle for ProjectedValue<ArrayType, V>
where
    Self: ParallelPermute,
{
    fn parallel_shuffle(&self, axis_name: &str, permutation: &[usize]) -> Result<Self, ProgramError> {
        let mut seen = vec![false; permutation.len()];
        for &source in permutation {
            let Some(source_seen) = seen.get_mut(source) else {
                return Err(TypeError::invalid(format!(
                    "`parallel_shuffle` source index {} is out of bounds for a permutation of length {}",
                    source,
                    permutation.len(),
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
        self.parallel_permute(axis_name, permutation.iter().copied().zip(0..).collect())
    }
}

/// Returns `value` with its slices along `axis` routed from their source positions to their target positions.
/// Specifically, slice `target` of the result is slice `sources[target]` of `value`, or a slice of zeros when
/// `sources[target]` is `None`. The result has `sources.len()` slices along `axis` and otherwise the shape of `value`.
/// For example, routing the rows `[a, b, c]` with `sources = [None, Some(0), Some(1)]` yields `[0, a, b]`.
///
/// The batching rule of [`ParallelPermuteOperation`] applies this function along the batch axis, where every slice is
/// one batch item, both to the packed values and to the extents of their bounded ragged axes, so that the extents move
/// together with their values and untargeted items receive zero extents together with their zero-filled values.
///
/// # Errors
///
/// Returns a [`BatchingError`] if `value` is not statically shaped, or if staging one of the slices, the zero slice,
/// or the concatenation fails.
fn route_axis_slices<V: Value<Type = ArrayType> + ZeroLike + Concatenate + Slice + Transpose>(
    value: &V,
    axis: usize,
    sources: &[Option<usize>],
) -> Result<V, BatchingError> {
    let value = if axis == 0 { value.clone() } else { value.clone().move_axis(axis, 0)? };
    let Some(shape) = value.r#type().static_shape() else {
        return Err(BatchingError::UnsupportedOperation {
            message: format!(
                "`{PARALLEL_PERMUTE_OPERATION_NAME}` batching requires statically shaped inputs and ragged extents",
            ),
        });
    };
    let dimensions = shape.dimensions().to_vec();
    let rank = dimensions.len();
    let strides = vec![1; rank];
    let slice_item = |item: usize| -> Result<V, ProgramError> {
        let mut start_indices = vec![0; rank];
        let mut limit_indices = dimensions.clone();
        start_indices[0] = item;
        limit_indices[0] = item + 1;
        value.slice(&start_indices, &limit_indices, &strides)
    };
    let mut zero_item = None;
    let mut items = Vec::with_capacity(sources.len());
    for source in sources {
        match source {
            Some(source) => items.push(slice_item(*source)?),
            None => {
                if zero_item.is_none() {
                    zero_item = Some(slice_item(0)?.zero_like()?);
                }
                items.push(zero_item.clone().unwrap());
            }
        }
    }
    let permuted = Concatenate::concatenate(&items, 0)?;
    if axis == 0 { Ok(permuted) } else { Ok(permuted.move_axis(0, axis)?) }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::batching::DynamicArrayExtentBatchingPolicy;
    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrValue, DataType, DimensionBounds, DimensionValue,
        DimensionVariable, LogicalMesh, MeshAxis, Sharding,
    };
    use crate::axes::AxisError;
    use crate::batching::{BatchAxis, BatchAxisSpecification, BatchingTracer, batch};
    use crate::contexts::{EagerContext, ProjectedContext, StagingContext};
    use crate::differentiation::{
        DifferentiableOperation, DifferentiationContext, DifferentiationDual, differentiate_at,
    };
    use crate::interpretation::InterpretableOperation;
    use crate::macros::check_operation_type_inference;
    use crate::operations::collectives::tests::f32_vector;
    use crate::operations::reductions::{Reduce, ReductionKind};
    use crate::parameters::Placeholder;
    use crate::partial::{
        PartialEvaluationContext, PartialEvaluationOutput, PartialEvaluationValue, PartialValue,
        PartiallyEvaluatableOperation,
    };
    use crate::programs::{EmptyRegionDriver, Program, ProgramBuilder};
    use crate::tracing::TracingContext;

    use super::*;

    /// Creates a manual mesh whose axis `"x"` has two participants and whose axis `"y"` has one.
    fn manual_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 1, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap()
    }

    /// Builds the single-instruction program that applies `operation` to one input of type `input_type`.
    fn parallel_permute_program(
        operation: ParallelPermuteOperation,
        input_type: ArrayType,
    ) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(input_type);
        let outputs = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap().to_vec();
        builder.build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Batches `operation` on `input` under an eager batching level of size `axis_size` that binds the axis `"x"`.
    fn batch_parallel_permute(
        operation: &ParallelPermuteOperation,
        axis_size: usize,
        input: ArrayBatch<Array>,
    ) -> Result<Vec<ArrayBatch<Array>>, BatchingError> {
        let context = BatchingContext::<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>::new(
            EagerContext::new(),
            axis_size,
        )
        .with_axis_name("x".to_string());
        Ok(operation.batch(&context, &EmptyRegionDriver, &[input])?.into_parts().0)
    }

    #[test]
    fn test_parallel_permute() {
        let operation = ParallelPermuteOperation::new("x".to_string(), 3, vec![(0, 1), (2, 0)]);
        assert_eq!(operation.name(), PARALLEL_PERMUTE_OPERATION_NAME);
        assert_eq!(operation.axis_name(), "x");
        assert_eq!(operation.axis_size(), 3);
        assert_eq!(operation.source_target_pairs(), &[(0, 1), (2, 0)]);
        assert_eq!(
            operation.to_string(),
            "parallel_permute [axis_name=\"x\", axis_size=3, source_target_pairs=[(0, 1), (2, 0)]]",
        );
        assert_eq!(operation, operation.clone());
        assert_ne!(operation, ParallelPermuteOperation::new("x".to_string(), 3, vec![(0, 1)]));
        assert_ne!(operation, ParallelPermuteOperation::new("y".to_string(), 3, vec![(0, 1), (2, 0)]));
    }

    #[test]
    fn test_parallel_permute_type_inference() {
        // A permutation preserves its input type, including the variation over a manual mesh axis, and an empty
        // permutation, which zeros every participant, is valid.
        let sharding = Sharding::replicated(manual_mesh(), 1);
        let varying = f32_vector(4).with_sharding(sharding.clone().with_varying_manual_axes(["x"]).unwrap()).unwrap();
        check_operation_type_inference!(
            operation = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1), (1, 0)]),
            cases = [
                { input_types = [f32_vector(3)], output_types = [f32_vector(3)] },
                { input_types = [varying.clone()], output_types = [varying.clone()] },
            ],
        );
        check_operation_type_inference!(
            operation = ParallelPermuteOperation::new("x".to_string(), 2, Vec::new()),
            cases = [{ input_types = [f32_vector(3)], output_types = [f32_vector(3)] }],
        );

        // Every pair must reference participants of the axis, and no two pairs may share a source or a target.
        for (pairs, message) in [
            (vec![(0, 2)], "`parallel_permute` pair (0, 2) is out of bounds for axis size 2"),
            (vec![(2, 0)], "`parallel_permute` pair (2, 0) is out of bounds for axis size 2"),
            (
                vec![(0, 1), (0, 0)],
                "`parallel_permute` pairs must have unique sources and targets but (0, 0) repeats one",
            ),
            (
                vec![(0, 1), (1, 1)],
                "`parallel_permute` pairs must have unique sources and targets but (1, 1) repeats one",
            ),
        ] {
            check_operation_type_inference!(
                operation = ParallelPermuteOperation::new("x".to_string(), 2, pairs),
                cases = [{ input_types = [f32_vector(3)], error = message }],
            );
        }

        // An input that is invariant over the manual axis would wrongly type the permuted output as invariant, and
        // inputs with a pending cross-device sum are rejected.
        check_operation_type_inference!(
            operation = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1)]),
            cases = [
                {
                    input_types = [f32_vector(4).with_sharding(sharding.clone()).unwrap()],
                    error = "`parallel_permute` input must vary over manual axis `x`; pass an invariant value through \
                             `parallel_vary` first so that the permuted output is typed as varying",
                },
                {
                    input_types = [f32_vector(4).with_sharding(sharding.with_unreduced_axes(["y"]).unwrap()).unwrap()],
                    error = "`parallel_permute` does not support unreduced inputs",
                },
            ],
        );
    }

    #[test]
    fn test_parallel_permute_interpretation() {
        // Outside any binder, the single participant of a degenerate axis keeps its value only when it sends to itself,
        // and receives zeros otherwise, while a larger axis has no per-item semantics.
        let context = EagerContext::<Array>::new();
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
        assert_eq!(
            ParallelPermuteOperation::new("x".to_string(), 1, vec![(0, 0)]).interpret(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&input),
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            ParallelPermuteOperation::new("x".to_string(), 1, Vec::new()).interpret(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&input),
            ),
            Ok(vec![Array::vector(vec![0.0f32, 0.0]).unwrap()]),
        );
        assert_eq!(
            ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1)]).interpret(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&input),
            ),
            Err(ProgramError::UnsupportedOperation {
                message: "cannot interpret `parallel_permute` over axis `x` of size 2 without an enclosing binder"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_parallel_permute_partial_evaluation() {
        // A known input over a degenerate axis folds through interpretation, here into zeros for the untargeted
        // participant.
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let program =
            parallel_permute_program(ParallelPermuteOperation::new("x".to_string(), 1, Vec::new()), f32_vector(2));
        let evaluation = program.partially_evaluate(&[PartialValue::Known(input.clone())]).unwrap();
        assert_eq!(evaluation.outputs(), &[PartialEvaluationOutput::Known(Array::vector(vec![0.0f32, 0.0]).unwrap())]);

        // A known input over a larger axis under an eager parent residualizes the operation, which has no per-item
        // value, so the residual program is the source program itself.
        let operation = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1)]);
        let program = parallel_permute_program(operation.clone(), f32_vector(2));
        let evaluation = program.partially_evaluate(&[PartialValue::Known(input)]).unwrap();
        assert_eq!(evaluation.program().to_string(), program.to_string());
        assert!(evaluation.outputs()[0].is_unknown());

        // A known input under a staging parent stays known, because the operation is staged into the parent trace.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = PartialEvaluationValue::known(trace.input(f32_vector(2)));
        let outputs = operation
            .partially_evaluate(&PartialEvaluationContext::new(trace), &EmptyRegionDriver, &[input])
            .unwrap();
        assert!(outputs[0].is_known());
        assert_eq!(outputs[0].r#type().as_ref(), &f32_vector(2));
    }

    #[test]
    fn test_parallel_permute_batching() {
        // A level that binds the permuted axis reassembles its mapped axis in target order, wherever that axis sits,
        // and zeros every item that no pair targets.
        let swap = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1), (1, 0)]);
        let shift = ParallelPermuteOperation::new("x".to_string(), 3, vec![(0, 1), (1, 2)]);
        assert_eq!(
            batch_parallel_permute(
                &swap,
                2,
                ArrayBatch::new(Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap(), BatchAxis::new(0)).unwrap(),
            ),
            Ok(vec![
                ArrayBatch::new(Array::matrix(2, 2, vec![3.0, 4.0, 1.0, 2.0]).unwrap(), BatchAxis::new(0)).unwrap()
            ]),
        );
        assert_eq!(
            batch_parallel_permute(
                &swap,
                2,
                ArrayBatch::new(Array::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(), BatchAxis::new(1))
                    .unwrap(),
            ),
            Ok(vec![
                ArrayBatch::new(Array::matrix(2, 3, vec![2.0, 4.0, 6.0, 1.0, 3.0, 5.0]).unwrap(), BatchAxis::new(0))
                    .unwrap(),
            ]),
        );
        assert_eq!(
            batch_parallel_permute(
                &shift,
                3,
                ArrayBatch::new(Array::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(), BatchAxis::new(0))
                    .unwrap(),
            ),
            Ok(vec![
                ArrayBatch::new(Array::matrix(3, 2, vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0]).unwrap(), BatchAxis::new(0))
                    .unwrap(),
            ]),
        );

        // A replicated input holds the same value for every item, so a full permutation leaves it unchanged, while a
        // partial permutation materializes it so that the untargeted items can receive zeros.
        let input = ArrayBatch::replicated(Array::vector(vec![1.0, 2.0]).unwrap());
        assert_eq!(batch_parallel_permute(&swap, 2, input.clone()), Ok(vec![input.clone()]));
        assert_eq!(
            batch_parallel_permute(&shift, 3, input),
            Ok(vec![
                ArrayBatch::new(Array::matrix(3, 2, vec![0.0, 0.0, 1.0, 2.0, 1.0, 2.0]).unwrap(), BatchAxis::new(0))
                    .unwrap(),
            ]),
        );

        // The operation's axis size must equal the size of the level that binds its axis.
        assert_eq!(
            batch_parallel_permute(
                &swap,
                3,
                ArrayBatch::new(Array::vector(vec![1.0, 2.0, 3.0]).unwrap(), BatchAxis::new(0)).unwrap(),
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_permute` over axis `x` resolved axis size 2 but the mapped batch axis has size 3"
                    .to_string(),
            }),
        );

        // A level that binds another axis forwards the permutation to its parent and preserves its mapped axis.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input =
            ArrayBatch::new(trace.input(ArrayType::new_static(DataType::F32, [2, 3])), BatchAxis::new(1)).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace.clone(), 3).with_axis_name("y".to_string());
        let outputs = swap.batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input)).unwrap().into_parts().0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(1));
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(
                vec![outputs[0].value().atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 3] .
                let %1:f32[2, 3] = parallel_permute [axis_name=\"x\", axis_size=2, source_target_pairs=[(0, 1), (1, 0)]] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_parallel_permute_batching_ragged() {
        // Bounded ragged extents follow the same routing as their packed values, and an untargeted item receives a zero
        // extent together with its zero-filled value.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let ragged = |values: Vec<f64>, extents: Array, extent_axes: Vec<usize>| {
            ArrayBatch::new(Array::matrix(2, 3, values).unwrap(), BatchAxis::new(0))
                .unwrap()
                .with_ragged_axes(vec![RaggedAxis::new(1, extents, length.clone(), extent_axes)])
                .unwrap()
        };
        let swap = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1), (1, 0)]);
        let send = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1)]);
        let input = ragged(vec![1.0, 0.0, 0.0, 2.0, 3.0, 4.0], Array::vector(vec![1i32, 3]).unwrap(), vec![0]);
        assert_eq!(
            batch_parallel_permute(&swap, 2, input.clone()),
            Ok(vec![ragged(vec![2.0, 3.0, 4.0, 1.0, 0.0, 0.0], Array::vector(vec![3i32, 1]).unwrap(), vec![0])]),
        );
        assert_eq!(
            batch_parallel_permute(&send, 2, input),
            Ok(vec![ragged(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0], Array::vector(vec![0i32, 1]).unwrap(), vec![0])]),
        );

        // Extents that are the same for every item stay replicated under a full permutation, but must vary over the
        // batch axis before a partial permutation can zero the extent of an untargeted item.
        let input = ragged(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], Array::scalar(1i32).unwrap(), Vec::new());
        assert_eq!(
            batch_parallel_permute(&swap, 2, input.clone()),
            Ok(vec![ragged(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], Array::scalar(1i32).unwrap(), Vec::new())]),
        );
        assert_eq!(
            batch_parallel_permute(&send, 2, input),
            Ok(vec![ragged(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0], Array::vector(vec![0i32, 1]).unwrap(), vec![0])]),
        );

        // The dynamic batching policy of the composite family materializes replicated extents in the same way.
        let input =
            ArrayBatch::new(Array::matrix(2, 3, vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(), BatchAxis::new(0))
                .unwrap()
                .with_ragged_axes(vec![RaggedAxis::new(1, Array::scalar(1i32).unwrap(), length.clone(), Vec::new())])
                .unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
            ProjectedContext::new(EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new()),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        )
        .with_axis_name("x".to_string());
        assert_eq!(
            send.batch(&context, &EmptyRegionDriver, &[input]).unwrap().into_parts().0,
            vec![
                ArrayBatch::new(Array::matrix(2, 3, vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(), BatchAxis::new(0))
                    .unwrap()
                    .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![0i32, 1]).unwrap(), length, vec![0],)])
                    .unwrap(),
            ],
        );

        // A zero extent is not representable for a dimension whose lower bound is positive, which only a partial
        // permutation can require, and a level that binds another axis cannot route the extents of its own items.
        let positive_length = DimensionVariable::new("length", DimensionBounds::new(1, Some(4)).unwrap());
        let input =
            ArrayBatch::new(Array::matrix(2, 3, vec![1.0, 0.0, 0.0, 2.0, 3.0, 4.0]).unwrap(), BatchAxis::new(0))
                .unwrap()
                .with_ragged_axes(vec![RaggedAxis::new(
                    1,
                    Array::vector(vec![1i32, 3]).unwrap(),
                    positive_length,
                    vec![0],
                )])
                .unwrap();
        assert!(batch_parallel_permute(&swap, 2, input.clone()).is_ok());
        assert_eq!(
            batch_parallel_permute(&send, 2, input.clone()),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_permute` cannot assign a zero extent to bounded ragged dimension `length` whose \
                          lower bound is 1"
                    .to_string(),
            }),
        );
        let context = BatchingContext::<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>::new(
            EagerContext::new(),
            2,
        )
        .with_axis_name("y".to_string());
        assert_eq!(
            swap.batch(&context, &EmptyRegionDriver, &[input]).map(|outputs| outputs.into_parts().0),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_permute` does not support bounded ragged dimension `length` on input 0".to_string(),
            }),
        );
    }

    #[test]
    fn test_parallel_permute_differentiation() {
        // The collective is linear, so the tangent rides the same permutation as the primal.
        let operation = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1)]);
        assert_eq!(
            parallel_permute_program(operation.clone(), f32_vector(2)).jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[2] .
                let %2:f32[2] = parallel_permute [axis_name=\"x\", axis_size=2, source_target_pairs=[(0, 1)]] %0
                    %3:f32[2] = parallel_permute [axis_name=\"x\", axis_size=2, source_target_pairs=[(0, 1)]] %1
                in (%2, %3)"
            },
        );

        // A structural zero tangent stays a structural zero, and only the primal permutation is staged.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = context.input(f32_vector(2));
        let outputs = operation
            .jvp(
                &DifferentiationContext::fused(context.clone()),
                &EmptyRegionDriver,
                &[DifferentiationDual::new_with_zero_tangent(input).unwrap()],
            )
            .unwrap();
        assert!(outputs[0].tangent().is_zero());
        assert_eq!(context.builder().borrow().instructions().len(), 1);

        // Reverse mode through a level that binds the axis pulls every cotangent back to the item that sent the value,
        // so the last item, whose value no pair receives, gets a zero gradient.
        let inputs = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(
            differentiate_at(inputs).value_and_gradient(|inputs| {
                let shifted = batch(
                    |item| item.parallel_permute("x", vec![(0, 1), (1, 2)]),
                    inputs,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("x"),
                )?;
                Ok(shifted.reduce(&[0], ReductionKind::Sum)?)
            }),
            Ok((Array::scalar(3.0).unwrap(), Array::vector(vec![1.0, 1.0, 0.0]).unwrap())),
        );
    }

    #[test]
    fn test_parallel_permute_transposition() {
        // Sending along `(source, target)` pulls cotangents back along `(target, source)`, so the transpose is the
        // permutation with every pair inverted, and transposing it again recovers the original permutation.
        let program = parallel_permute_program(
            ParallelPermuteOperation::new("x".to_string(), 3, vec![(0, 1), (1, 2)]),
            f32_vector(2),
        );
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = parallel_permute [axis_name=\"x\", axis_size=3, source_target_pairs=[(1, 0), (2, 1)]] %0
                in (%1)"
            },
        );
        assert_eq!(transposed.transpose_with_respect_to(&[0], &[]).unwrap().to_string(), program.to_string());
    }

    #[test]
    fn test_parallel_permute_parallel_permute() {
        // A name that no enclosing binder binds fails fast instead of silently acting as identity.
        let inputs = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(
            batch(
                |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>| {
                    item.parallel_permute("y", vec![(0, 1)])
                },
                inputs.clone(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::named("x"),
            ),
            Err::<Array, _>(BatchingError::Axis(AxisError::UnboundAxisName { name: "y".to_string() })),
        );

        // A name that a `batch` level binds resolves the size of that level and permutes its batch items.
        assert_eq!(
            batch(
                |item| item.parallel_permute("x", vec![(0, 2), (2, 0)]),
                inputs,
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::named("x"),
            ),
            Ok(Array::vector(vec![3.0, 0.0, 1.0]).unwrap()),
        );

        // Over a manual mesh axis, a varying value is permuted directly, while an invariant value, and a value without
        // a sharding, are first made varying, because the permuted participants generally hold different values.
        let mesh = manual_mesh();
        let sharding = Sharding::replicated(mesh.clone(), 0);
        let varying = ArrayType::scalar(DataType::F32)
            .with_sharding(sharding.clone().with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let invariant = ArrayType::scalar(DataType::F32).with_sharding(sharding).unwrap();
        for (input_type, expected) in [
            (
                varying.clone(),
                indoc! {"
                    lambda %0:f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [], varying_manual={'x'}}] .
                    let %1:f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [], varying_manual={'x'}}] = \
                        parallel_permute [axis_name=\"x\", axis_size=2, source_target_pairs=[(0, 1)]] %0
                    in (%1)"
                },
            ),
            (
                invariant,
                indoc! {"
                    lambda %0:f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, []}] .
                    let %1:f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [], varying_manual={'x'}}] = \
                            parallel_vary [axis_name=\"x\"] %0
                        %2:f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [], varying_manual={'x'}}] = \
                            parallel_permute [axis_name=\"x\", axis_size=2, source_target_pairs=[(0, 1)]] %1
                    in (%2)"
                },
            ),
            (
                ArrayType::scalar(DataType::F32),
                indoc! {"
                    lambda %0:f32[] .
                    let %1:f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, []}] = broadcast [
                        output_type=f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, []}],
                        output_axes=[],
                    ] %0
                        %2:f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [], varying_manual={'x'}}] = \
                            parallel_vary [axis_name=\"x\"] %1
                        %3:f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [], varying_manual={'x'}}] = \
                            parallel_permute [axis_name=\"x\", axis_size=2, source_target_pairs=[(0, 1)]] %2
                    in (%3)"
                },
            ),
        ] {
            let (output, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_permute("x", vec![(0, 1)]),
                input_type,
                vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
            )
            .unwrap();
            assert_eq!(output, varying);
            assert_eq!(program.to_string(), expected);
        }
    }

    #[test]
    fn test_parallel_shuffle_parallel_shuffle() {
        // A composite value shuffles through its array view: output item `i` receives input item `permutation[i]`,
        // and a permutation shorter than the axis gives every remaining item zeros.
        let shuffle = |permutation: &[usize]| {
            let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
                EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
                ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
            )
            .with_axis_name("x".to_string());
            let input = ArrayIrValue::Array(Array::matrix(3, 2, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
            let input = BatchingTracer::new(context, ArrayIrBatch::new(input, BatchAxis::new(0)).unwrap());
            input.parallel_shuffle("x", permutation).map(|output| output.into_batch().into_value())
        };
        assert_eq!(
            shuffle(&[2, 0, 1]),
            Ok(ArrayIrValue::Array(Array::matrix(3, 2, vec![5.0f32, 6.0, 1.0, 2.0, 3.0, 4.0]).unwrap())),
        );
        assert_eq!(
            shuffle(&[1, 0]),
            Ok(ArrayIrValue::Array(Array::matrix(3, 2, vec![3.0f32, 4.0, 1.0, 2.0, 0.0, 0.0]).unwrap())),
        );

        // The permutation must be a permutation of its own positions, and it cannot be longer than the axis.
        assert_eq!(
            shuffle(&[0, 2]),
            Err(ProgramError::Type(TypeError::invalid(
                "`parallel_shuffle` source index 2 is out of bounds for a permutation of length 2",
            ))),
        );
        assert_eq!(
            shuffle(&[1, 1]),
            Err(ProgramError::Type(TypeError::invalid(
                "`parallel_shuffle` permutation contains source index 1 more than once",
            ))),
        );
        assert_eq!(
            shuffle(&[3, 2, 1, 0]),
            Err(BatchingError::Type(TypeError::invalid(
                "`parallel_permute` pair (3, 0) is out of bounds for axis size 3",
            ))
            .into()),
        );
    }
}
