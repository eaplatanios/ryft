//! Operations that order array elements along one axis. Sorting is defined by the [`SortOperation`] type together with
//! the [`Sort`] value capability trait, whose functions apply it to eager [`Array`]s and traced values alike, so the
//! same code executes immediately or records into a program depending on the value it runs on. The capabilities fall
//! into two groups:
//!
//!   - **Sorting:** [`Sort`] stably sorts one or more same-shaped arrays along one axis in a [`SortDirection`],
//!     by the lexicographic order of their leading key inputs, and co-permutes every other input as a passenger.
//!   - **Ranking:** [`TopK`] selects the `k` largest elements along one axis together with their indices. It is a
//!     composition rather than a primitive. It sorts the ranked value together with an `i32` index
//!     [`iota`](IotaOperation) passenger and slices the leading entries, and so every program transform supports
//!     it through the rules of [`SortOperation`], [`Slice`], and [`Reshape`], and its indices are always `i32`.
//!     The indices of the largest and smallest elements are reductions rather than rankings, and are supported
//!     via [`ArgMax`](crate::ArgMax) and [`ArgMin`](crate::ArgMin), respectively.
//!
//! A [`SortOrdering`] selects how floating-point keys, including the real and imaginary parts of complex keys, compare.
//! Under the default [`SortOrdering::Canonical`] ordering, `-0.0` and `+0.0` compare equal, every NaN compares equal
//! to every other NaN and greater than `+∞`, and complex keys order lexicographically by their real part and then their
//! imaginary part, same as JAX's [`lax.sort`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.sort.html).
//! Under [`SortOrdering::Total`], keys order by the IEEE 754 total order of StableHLO's
//! [`TOTALORDER`](https://openxla.org/stablehlo/spec#compare) comparison (i.e., `-NaN < -∞ < … < -0.0 < +0.0
//! < … < +∞ < +NaN`), and complex keys are rejected. [`TopK`] ranks under the total ordering, following JAX's
//! [`lax.top_k`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.top_k.html). Every sort is stable, so elements
//! that tie on every key keep their original relative order and ranking ties select the lowest index.
//!
//! # Batching
//!
//! Every mapped input moves its batch axis to the leading position and every replicated input is broadcast to the
//! batched shape, because all inputs of a sort must agree on shape. The sorted axis then lifts past the leading batch
//! axis. Bounded ragged batches are rejected, because sorting an axis that carries padding would move that padding into
//! valid data.
//!
//! # Differentiation
//!
//! The permutation that a sort applies is piecewise constant in its keys, so every input, including each key,
//! differentiates as a passenger of that permutation: tangents ride a sort of the primal keys, and the transpose of
//! the resulting linear map sorts the output cotangents by the forward permutation itself, which applies its inverse.
//! Reverse mode differentiation therefore works through every sorting and ranking capability, while the integer indices
//! returned by the ranking capabilities have zero derivatives.
//!
//! # Examples
//!
//! Sorting co-permutes passengers by the order of the keys, and the ordering decides how signed zeros and NaNs tie:
//!
//! ```rust
//! # use ryft_core::{Array, ProgramError, Sort, SortDirection, SortOrdering};
//! # fn main() -> Result<(), ProgramError> {
//! let keys = Array::vector(vec![0.0, -0.0, f64::NAN, -f64::NAN])?;
//! let positions = Array::vector(vec![0i32, 1, 2, 3])?;
//! let canonical = Array::sort(&[keys.clone(), positions.clone()], 0, SortDirection::Ascending)?;
//! assert_eq!(canonical[1], Array::vector(vec![0i32, 1, 2, 3])?);
//! let total = Array::sort_with_ordering(&[keys, positions], 0, 1, SortDirection::Ascending, SortOrdering::Total)?;
//! assert_eq!(total[1], Array::vector(vec![3i32, 1, 0, 2])?);
//! # Ok(())
//! # }
//! ```
//!
//! Ranking selects the leading entries of a stable sort, so ties select the lowest index:
//!
//! ```rust
//! # use ryft_core::{Array, ProgramError, TopK};
//! # fn main() -> Result<(), ProgramError> {
//! let scores = Array::vector(vec![1.0, 3.0, 2.0, 3.0])?;
//! let (values, indices) = scores.top_k(2, 0)?;
//! assert_eq!(values, Array::vector(vec![3.0, 3.0])?);
//! assert_eq!(indices, Array::vector(vec![1i32, 3])?);
//! # Ok(())
//! # }
//! ```

use std::fmt::Display;
use std::num::NonZeroUsize;

use ryft_macros::capability;

use crate::arrays::{
    Array, ArrayAddressing, ArrayBatch, ArrayBatchingPolicy, ArrayElement, ArrayExtentBatchingPolicy, ArrayIrType,
    ArrayIrValue, ArrayType, Complex, DataType, Dimension, Shape, ShardingDimension,
};
use crate::axes::Axis;
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{DifferentiableType, DifferentiationDual, ElementwiseDerivativeAlignment};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::Capability;
use crate::operations::collectives::parallel_vary::ManualVariationAlignment;
use crate::operations::constants::iota::{Iota, IotaOperation};
use crate::operations::manipulation::broadcasting::Broadcast;
use crate::operations::manipulation::reshaping::Reshape;
use crate::operations::manipulation::slicing::Slice;
use crate::operations::manipulation::transposition::Transpose;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, ProgramError, ProjectedValue, RegionInterface, TypeError, Typed, Value,
    ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

/// Direction in which a [`SortOperation`] orders its key inputs.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SortDirection {
    /// Orders keys from smallest to largest.
    Ascending,

    /// Orders keys from largest to smallest.
    Descending,
}

impl Display for SortDirection {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ascending => formatter.write_str("ascending"),
            Self::Descending => formatter.write_str("descending"),
        }
    }
}

/// Order in which a [`SortOperation`] compares floating-point keys, including the real and imaginary parts of complex
/// keys. Boolean keys order `false` before `true`, and integer keys order by value, under both orderings.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum SortOrdering {
    /// Orders floating-point keys by value: `-0.0` and `+0.0` compare equal, and every NaN, regardless of its sign and
    /// payload, compares equal to every other NaN and greater than `+∞`. Complex keys order lexicographically by their
    /// real part and then their imaginary part, each compared this way. This is the ordering of JAX's
    /// [`lax.sort`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.sort.html) and of NumPy's sorts.
    #[default]
    Canonical,

    /// Orders floating-point keys by the IEEE 754 total order, `-NaN < -∞ < … < -0.0 < +0.0 < … < +∞ < +NaN`,
    /// where NaNs of the same sign are further ordered by payload. This is the ordering of StableHLO's
    /// [`TOTALORDER`](https://openxla.org/stablehlo/spec#compare) comparison and of JAX's
    /// [`lax.top_k`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.top_k.html).
    /// Complex keys have no total order and are rejected.
    Total,
}

impl Display for SortOrdering {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Canonical => formatter.write_str("canonical"),
            Self::Total => formatter.write_str("total"),
        }
    }
}

/// Canonical operation name for [`SortOperation`].
pub const SORT_OPERATION_NAME: &str = "sort";

/// [`Operation`] that sorts one or more same-shaped inputs along one axis by the values of its first `key_count` inputs
/// (i.e., its keys). Elements are ordered lexicographically by the keys in input order (i.e., key 0 decides, ties on
/// key 0 fall through to key 1, and so on), with every key compared in the same [`SortDirection`] and under the same
/// [`SortOrdering`], and every other input is co-permuted as a passenger. The sort is always stable, so elements that
/// are equal on every key keep their original relative order, which is what routes ranking ties (e.g., in [`TopK`]) to
/// the lowest index. An ascending sort under the default [`SortOrdering::Canonical`] ordering computes JAX's
/// [`lax.sort`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.sort.html) with `num_keys = key_count`, and
/// a descending one computes the stable descending sort of `jnp.sort(..., descending=True)`.
///
/// Inputs must agree on shape, while their element types may differ. Token and structural-zero keys are rejected, as
/// are complex keys under [`SortOrdering::Total`]. The sorted axis must not be sharded, because sorting across shards
/// would require communication. Inputs that carry unreduced partial sums are rejected. Reduced keys are rejected too,
/// because their zero-filled replicas can select different permutations, while reduced passengers retain their
/// reduction state, because permutations preserve their zero-filled replicas. Inputs must have matching manual
/// variation, and [`Sort`] inserts the required transitions before binding.
///
/// There is no user-provided comparator: the fixed lexicographic key ordering covers ranking (e.g., [`TopK`]) and
/// multi-key sorts without carrying a comparator region through every program transform.
///
/// The permutation is piecewise constant in the keys, so every input, including each key, differentiates as a
/// passenger: tangents ride a sort of their primal keys, and the transpose of that permutation sorts the output
/// cotangents by the permutation itself, which applies its inverse.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SortOperation {
    /// Axis along which the inputs are sorted.
    axis: usize,

    /// Number of leading inputs that act as lexicographic sort keys.
    key_count: NonZeroUsize,

    /// Direction in which the key inputs are ordered.
    direction: SortDirection,

    /// Order in which floating-point keys are compared.
    ordering: SortOrdering,
}

impl SortOperation {
    /// Creates a new [`SortOperation`] that sorts along `axis` in the provided `direction` with a single key input and
    /// the default [`SortOrdering::Canonical`] ordering.
    #[inline]
    pub fn new(axis: usize, direction: SortDirection) -> Self {
        Self { axis, key_count: NonZeroUsize::MIN, direction, ordering: SortOrdering::default() }
    }

    /// Creates the [`SortOperation`] that the [`Sort`] capability functions execute or stage for `inputs`, normalizing
    /// the possibly negative `axis` against the rank of the first input and rejecting a zero `key_count`. Type
    /// inference validates the remaining input requirements.
    fn from_sort_arguments<V: Typed<Type = ArrayType>>(
        inputs: &[V],
        axis: Axis,
        key_count: usize,
        direction: SortDirection,
        ordering: SortOrdering,
    ) -> Result<Self, ProgramError> {
        let Some(first) = inputs.first() else {
            return Err(TypeError::invalid(format!("`{SORT_OPERATION_NAME}` needs at least one input")).into());
        };
        let rank = first.r#type().rank();
        let axis = axis.normalize(rank).map_err(|_| {
            TypeError::invalid(format!("`{SORT_OPERATION_NAME}` axis {axis} is out of bounds for rank {rank}"))
        })?;
        let key_count = NonZeroUsize::new(key_count).ok_or_else(|| ProgramError::InvalidArgument {
            message: format!("`{SORT_OPERATION_NAME}` `key_count` must be at least 1"),
        })?;
        Ok(Self { axis, key_count, direction, ordering })
    }

    /// Returns this [`SortOperation`] with the provided number of leading key inputs compared lexicographically.
    #[inline]
    pub fn with_key_count(mut self, key_count: NonZeroUsize) -> Self {
        self.key_count = key_count;
        self
    }

    /// Returns this [`SortOperation`] with the provided floating-point key [`SortOrdering`].
    #[inline]
    pub fn with_ordering(mut self, ordering: SortOrdering) -> Self {
        self.ordering = ordering;
        self
    }

    /// Returns the axis along which the inputs are sorted for this [`SortOperation`].
    #[inline]
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// Returns the number of leading inputs that act as lexicographic sort keys for this [`SortOperation`].
    #[inline]
    pub fn key_count(&self) -> NonZeroUsize {
        self.key_count
    }

    /// Returns the [`SortDirection`] in which the key inputs are ordered for this [`SortOperation`].
    #[inline]
    pub fn direction(&self) -> SortDirection {
        self.direction
    }

    /// Returns the [`SortOrdering`] in which floating-point keys are compared for this [`SortOperation`].
    #[inline]
    pub fn ordering(&self) -> SortOrdering {
        self.ordering
    }
}

impl Display for SortOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for SortOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        SORT_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        let Some(key_type) = input_types.first() else {
            return Err(TypeError::invalid(format!("`{SORT_OPERATION_NAME}` needs at least one input")));
        };

        let key_count = self.key_count.get();
        if key_count > input_types.len() {
            return Err(TypeError::invalid(format!(
                "`{}` `key_count` {} exceeds input count {}",
                SORT_OPERATION_NAME,
                key_count,
                input_types.len(),
            )));
        }

        for input_type in &input_types[..key_count] {
            // A reduced key can select a different permutation on its zero-filled replicas, so invariant passengers
            // would need additional variation tracking. Reduced passengers remain valid.
            if !input_type.reduced_axes().is_empty() {
                return Err(TypeError::invalid(format!("`{SORT_OPERATION_NAME}` does not support reduced keys")));
            }

            let data_type = input_type.data_type();
            if data_type.is_token() || data_type.is_zero() {
                return Err(TypeError::invalid(format!(
                    "`{SORT_OPERATION_NAME}` does not support key data type `{data_type}`",
                )));
            }

            if data_type.is_complex() && self.ordering == SortOrdering::Total {
                return Err(TypeError::invalid(format!(
                    "`{SORT_OPERATION_NAME}` does not support key data type `{data_type}` under the `total` ordering",
                )));
            }
        }

        if self.axis >= key_type.rank() {
            return Err(TypeError::invalid(format!(
                "`{}` axis {} is out of bounds for rank {}",
                SORT_OPERATION_NAME,
                self.axis,
                key_type.rank(),
            )));
        }

        for input_type in input_types {
            if input_type.shape() != key_type.shape() {
                return Err(TypeError::invalid(format!(
                    "`{}` inputs must agree on shape but got {} and {}",
                    SORT_OPERATION_NAME,
                    key_type.shape(),
                    input_type.shape(),
                )));
            }

            if !input_type.unreduced_axes().is_empty() {
                return Err(TypeError::invalid(format!("`{SORT_OPERATION_NAME}` does not support unreduced inputs")));
            }

            if let Some(sharding) = input_type.sharding()
                && matches!(sharding.dimensions()[self.axis], ShardingDimension::Sharded(_))
            {
                return Err(TypeError::invalid(format!(
                    "`{}` cannot sort along sharded axis {}",
                    SORT_OPERATION_NAME, self.axis,
                )));
            }
        }

        ArrayType::check_matching_manual_variation(SORT_OPERATION_NAME, &input_types.iter().collect::<Vec<_>>())?;
        Ok(input_types.to_vec())
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        // Renders as `sort [axis=..., direction=...]` for the default single-key canonical sort, adding the
        // `key_count` and `ordering` fields only when they differ from their defaults, so that the overwhelmingly
        // common renderings stay short.
        OperationFormatter::new(formatter, indentation, SORT_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("axis", self.axis)?;
            if self.key_count > NonZeroUsize::MIN {
                operation.field("key_count", self.key_count)?;
            }
            operation.field("direction", self.direction)?;
            if self.ordering != SortOrdering::default() {
                operation.field("ordering", self.ordering)?;
            }
            Ok(())
        })
    }
}

impl<C: Domain<Type = ArrayType, Value: Sort>> InterpretableOperation<C> for SortOperation {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        C::Value::sort_with_ordering(inputs, self.axis, self.key_count.get(), self.direction, self.ordering)
    }
}

impl<C: Context<Type = ArrayType, Operation: From<SortOperation>>> PartiallyEvaluatableOperation<C> for SortOperation {}

impl<C: Context<Type = ArrayType, Value: Sort + Broadcast + Transpose>, P: ArrayExtentBatchingPolicy<C>>
    BatchableOperation<C, ArrayBatchingPolicy<P>> for SortOperation
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        // Every mapped input's batch axis moves to the leading physical position, replicated inputs broadcast to the
        // batched physical shape (because all sort inputs must agree on shape), and the sort axis lifts past the
        // inserted leading batch axis while the key count and ordering carry through unchanged.

        // Sorting a padded ragged axis would move padding into valid data, and repacking the inputs would drop the
        // ragged metadata of every other axis.
        ArrayBatch::reject_ragged_inputs(self, inputs)?;

        let Some(axis_size) = ArrayBatch::common_batch_size(inputs)? else {
            return Ok(self
                .interpret_with_batch_axes(context, inputs, &vec![BatchAxis::replicated(); inputs.len()])?
                .into());
        };
        let axis_sharding = ArrayBatch::sharding_for_inputs(inputs)?;
        let batched_inputs = inputs
            .iter()
            .map(|input| {
                if !input.batch_axis().is_replicated() {
                    return input.move_axis(0);
                }
                let unbatched_type = input.unbatched_type();
                let physical_type = unbatched_type.batched(0, Dimension::Static(axis_size), axis_sharding.clone())?;
                let output_axes = (1..physical_type.rank()).collect::<Vec<_>>();
                let broadcasted = input.value().clone().broadcast(physical_type.clone(), output_axes.as_slice())?;
                ArrayBatch::new(broadcasted, 0)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let lifted = SortOperation { axis: self.axis + 1, ..*self };
        Ok(lifted
            .interpret_with_batch_axes(
                context,
                batched_inputs.as_slice(),
                &vec![BatchAxis::from_position(0); inputs.len()],
            )?
            .into())
    }
}

impl_differentiable_operation! {
    SortOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Operation: From<SortOperation>,
    {
        |operation, context, _driver, inputs| {
            // Live tangents ride a sort of the primal keys as passengers, so they are permuted exactly like their
            // primals. When primal and tangent work share one context, a single sort permutes the primal inputs and
            // the tangents together. Otherwise, the primal sort runs in the primal context and the tangent sort
            // receives only the transferred keys, because the keys alone determine the permutation. Structural-zero
            // tangents remain symbolic because every permutation of zeros is zero.
            let key_count = operation.key_count().get();
            let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            let live_tangents = inputs
                .iter()
                .enumerate()
                .filter_map(|(index, input)| input.tangent().as_value().map(|tangent| (index, tangent.clone())))
                .collect::<Vec<_>>();
            let (outputs, sorted_tangents) = if live_tangents.is_empty() {
                (context.primal().bind(*operation, Vec::new(), primal_inputs.as_slice())?, Vec::new())
            } else if std::ptr::eq(context.primal(), context.tangent()) {
                let mut sort_inputs = primal_inputs;
                sort_inputs.extend(live_tangents.iter().map(|(_, tangent)| tangent.clone()));
                let mut outputs = context.primal().bind(*operation, Vec::new(), sort_inputs.as_slice())?;
                let sorted_tangents = outputs.split_off(inputs.len());
                (outputs, sorted_tangents)
            } else {
                let outputs = context.primal().bind(*operation, Vec::new(), primal_inputs.as_slice())?;
                let mut sort_inputs = primal_inputs
                    .into_iter()
                    .take(key_count)
                    .map(|key| context.primal_to_tangent(key))
                    .collect::<Result<Vec<_>, _>>()?;
                sort_inputs.extend(live_tangents.iter().map(|(_, tangent)| tangent.clone()));
                let mut sorted_tangents = context.tangent().bind(*operation, Vec::new(), sort_inputs.as_slice())?;
                (outputs, sorted_tangents.split_off(key_count))
            };

            let mut tangents = vec![None; inputs.len()];
            for ((index, _), tangent) in live_tangents.iter().zip(sorted_tangents) {
                tangents[*index] = Some(tangent);
            }

            outputs
                .into_iter()
                .zip(tangents)
                .map(|(primal, tangent)| {
                    let tangent = match tangent {
                        Some(tangent) => MaybeZero::Value(tangent),
                        None => MaybeZero::Zero(primal.r#type().tangent()?),
                    };
                    DifferentiationDual::new(primal, tangent)
                })
                .collect()
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType> + From<IotaOperation<ArrayType>> + From<SortOperation>,
        Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<ArrayType> + Sort,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // With known keys, the forward map permutes every passenger by the same permutation `p` along the sorted
            // axis (i.e., `y[i] = x[p[i]]`), so its transpose applies the inverse permutation to the output cotangents
            // (i.e., `x̄[p[i]] = ȳ[i]`). The rule recovers `p` by sorting the known keys with an index iota passenger
            // and then sorts the output cotangents by `p` itself in ascending order, which moves the cotangent at
            // position `i` to position `p[i]`. `p` is a permutation, so this second sort has no ties. Keys and known
            // passengers receive no cotangents, and structural-zero cotangents remain symbolic.
            check_count!("output", outputs, inputs.len(), ProgramError);
            check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
            let mut keys = inputs[..operation.key_count().get()]
                .iter()
                .map(|input| {
                    input.as_known().cloned().ok_or_else(|| ProgramError::InvalidArgument {
                        message: format!("`{SORT_OPERATION_NAME}` is not linear in its keys and requires known keys"),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;

            let cotangents = (operation.key_count().get()..inputs.len())
                .filter_map(|index| match &outputs[index] {
                    MaybeZero::Value(cotangent) if accumulators[index].is_needed() => Some((index, cotangent.clone())),
                    _ => None,
                })
                .collect::<Vec<_>>();
            if cotangents.is_empty() {
                return Ok(());
            }

            // The `i32` index iota shares the sharding of the keys, including their varying manual axes. An iota
            // holds the same elements on every device, so typing it as varying is always valid (i.e., it is what
            // a `parallel_vary` of the invariant iota produces, and an integer iota has no cotangent for that
            // transition to affect).
            let index_type = ArrayType::new(DataType::I32, keys[0].r#type().shape().clone())
                .with_sharding(keys[0].r#type().sharding().cloned())
                .map_err(|error| TypeError::invalid(error.to_string()))?;
            let index_operation = IotaOperation::new(index_type, operation.axis())?;
            let mut indices = context.stage_nullary_operation(index_operation)?;
            check_count!("output", indices, 1, ProgramError);
            keys.push(indices.remove(0));
            let permutation = Sort::sort_with_ordering(
                keys.as_slice(),
                operation.axis(),
                operation.key_count().get(),
                operation.direction(),
                operation.ordering(),
            )?
            .pop()
            .unwrap();

            let mut inverse_inputs = vec![permutation];
            inverse_inputs.extend(cotangents.iter().map(|(_, cotangent)| cotangent.clone()));
            let inverted = Sort::sort(inverse_inputs.as_slice(), operation.axis(), SortDirection::Ascending)?;
            for ((index, _), cotangent) in cotangents.iter().zip(inverted.into_iter().skip(1)) {
                let contribution = cotangent.unalign_cotangent(&inputs[*index].r#type().cotangent()?)?;
                accumulators[*index].accumulate(context, MaybeZero::Value(contribution))?;
            }
            Ok(())
        }
    },
}

/// Represents the ability to sort same-shaped inputs along one axis by the values of their leading key inputs. Elements
/// are ordered lexicographically by the keys (i.e., ties on earlier keys fall through to later keys), every key is
/// compared in the same [`SortDirection`] and under the same [`SortOrdering`], and every non-key input is co-permuted
/// as a passenger. Sorting is stable, so elements that tie on every key keep their original relative order. Concrete
/// [`Array`]s sort immediately, while context-carrying values align their manual variation and bind a [`SortOperation`]
/// through the context of their first input. Refer to the documentation of [`SortOperation`] for the input requirements
/// and the transform rules.
///
/// Besides the general [`Self::sort_with_ordering`] function, this trait provides the [`Self::sort`] and
/// [`Self::sort_with_key_count`] shortcut functions for single-key and canonically ordered sorts, which share
/// the axis and error contract of [`Self::sort_with_ordering`].
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and composite array IR values implement it for [`ArrayIrType`].
///
/// # Example
///
/// The following example sorts by a primary key whose ties are broken by a secondary key:
///
/// ```rust
/// # use ryft_core::{Array, ProgramError, Sort, SortDirection};
/// # fn main() -> Result<(), ProgramError> {
/// let primary = Array::vector(vec![2i32, 1, 2])?;
/// let secondary = Array::vector(vec![5.0, 9.0, 4.0])?;
/// let sorted = Array::sort_with_key_count(&[primary, secondary], 0, 2, SortDirection::Ascending)?;
/// assert_eq!(sorted[0], Array::vector(vec![1i32, 2, 2])?);
/// assert_eq!(sorted[1], Array::vector(vec![9.0, 4.0, 5.0])?);
/// # Ok(())
/// # }
/// ```
#[capability]
pub trait Sort<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Sorts `inputs` along `axis` by the values of their first input in the provided `direction` under the default
    /// [`SortOrdering::Canonical`] ordering, co-permuting every other input. Refer to [`Self::sort_with_ordering`]
    /// for the semantics of the parameters and for the errors that this function may return.
    #[inline]
    fn sort<A: Into<Axis>>(inputs: &[Self], axis: A, direction: SortDirection) -> Result<Vec<Self>, ProgramError> {
        Self::sort_with_key_count(inputs, axis, 1, direction)
    }

    /// Sorts `inputs` along `axis` lexicographically by the values of their first `key_count` inputs in the provided
    /// `direction` under the default [`SortOrdering::Canonical`] ordering, co-permuting every remaining input. Refer
    /// to [`Self::sort_with_ordering`] for the semantics of the parameters and for the errors that this function may
    /// return.
    #[inline]
    fn sort_with_key_count<A: Into<Axis>>(
        inputs: &[Self],
        axis: A,
        key_count: usize,
        direction: SortDirection,
    ) -> Result<Vec<Self>, ProgramError> {
        Self::sort_with_ordering(inputs, axis, key_count, direction, SortOrdering::default())
    }

    /// Sorts `inputs` along `axis` lexicographically by the values of their first `key_count` inputs in the provided
    /// `direction`, comparing floating-point keys under `ordering` and co-permuting every remaining input.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: Same-shaped values to sort, starting with the keys.
    ///   - `axis`: [`Axis`] along which the inputs are sorted. Negative axes count from the end.
    ///     Its dimension must not be sharded.
    ///   - `key_count`: Number of leading `inputs` that act as lexicographic sort keys, which must be at least 1.
    ///   - `direction`: [`SortDirection`] in which every key is ordered.
    ///   - `ordering`: [`SortOrdering`] under which floating-point keys are compared.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if `inputs` is empty, if `axis` is out of bounds, if `key_count` is zero or exceeds
    /// the number of inputs, if the inputs violate the requirements documented on [`SortOperation`], or if the context
    /// of the first input fails to bind the sort.
    fn sort_with_ordering<A: Into<Axis>>(
        inputs: &[Self],
        axis: A,
        key_count: usize,
        direction: SortDirection,
        ordering: SortOrdering,
    ) -> Result<Vec<Self>, ProgramError>;
}

impl Sort for Array {
    fn sort_with_ordering<A: Into<Axis>>(
        inputs: &[Self],
        axis: A,
        key_count: usize,
        direction: SortDirection,
        ordering: SortOrdering,
    ) -> Result<Vec<Self>, ProgramError> {
        // Type inference validates the inputs (e.g., their count, key data types, shapes, axis, sharding, and manual
        // variation) exactly as it does for staged sorts. The keys are then converted to an order-preserving encoding
        // of each element and the resulting gather map moves whole element encodings, so non-key inputs of any element
        // data type (including the sub-byte ones without a scalar representation) sort without being decoded.
        let operation = SortOperation::from_sort_arguments(inputs, axis.into(), key_count, direction, ordering)?;
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        operation.infer_output_types(input_types.as_slice(), &[])?;

        // Every sort key contributes order-preserving `u64` keys for its elements in row-major order. Comparing these
        // order keys compares the elements themselves (refer to `DataType::element_order_key` for the encoding), with
        // floating-point elements canonicalized under the canonical ordering. Complex sort keys contribute two order
        // key vectors (i.e., one for their real parts followed by one for their imaginary parts), which then compare
        // lexicographically like two separate sort keys.
        let canonicalize = ordering == SortOrdering::Canonical;
        let mut order_keys = Vec::<Vec<u64>>::with_capacity(operation.key_count().get());
        for key in &inputs[..operation.key_count().get()] {
            let data_type = key.r#type().data_type();
            let addressing = ArrayAddressing::new(key.r#type().into_owned())?;
            let elements = (0..addressing.element_count())
                .map(|index| &key.storage_bytes()[addressing.byte_range_for_flat_index(index)]);
            match data_type {
                DataType::C64 | DataType::C128 => {
                    let parts = elements
                        .map(|bytes| match data_type {
                            DataType::C64 => {
                                let value = Complex::<f32>::decode(bytes);
                                (f64::from(value.re), f64::from(value.im))
                            }
                            _ => {
                                let value = Complex::<f64>::decode(bytes);
                                (value.re, value.im)
                            }
                        })
                        .collect::<Vec<_>>();
                    order_keys.push(
                        parts
                            .iter()
                            .map(|(real, _)| DataType::floating_point_element_order_key(*real, canonicalize))
                            .collect(),
                    );
                    order_keys.push(
                        parts
                            .iter()
                            .map(|(_, imaginary)| DataType::floating_point_element_order_key(*imaginary, canonicalize))
                            .collect(),
                    );
                }
                _ => {
                    // Type inference rejects token and structural-zero keys, so every other key has element order keys.
                    order_keys
                        .push(elements.map(|bytes| data_type.element_order_key(bytes, canonicalize).unwrap()).collect())
                }
            }
        }

        // The sort permutes every slice along the sorted axis independently. In row-major order, the elements of one
        // such slice are `inner_stride` apart (i.e., the product of the dimensions after the axis), and the slices are
        // identified by an `outer` index over the dimensions before the axis and an `inner` offset within one stride.
        // The `gather` map records, for every flat output position, the flat input position whose element the sorted
        // output takes, and so it starts as the identity, which positions outside the sorted axis never leave.
        let dimensions = input_types[0].static_shape().unwrap().dimensions().to_vec();
        let axis = operation.axis();
        let axis_size = dimensions[axis];
        let inner_stride = dimensions[axis + 1..].iter().product::<usize>();
        let outer_count = dimensions[..axis].iter().product::<usize>();
        let mut gather = (0..dimensions.iter().product()).collect::<Vec<usize>>();
        let mut permutation = Vec::with_capacity(axis_size);
        for outer in 0..outer_count {
            for inner in 0..inner_stride {
                // `base` is the flat position of the first element of this slice, so its element at position `i`
                // along the sorted axis lives at `base + i * inner_stride`.
                let base = outer * axis_size * inner_stride + inner;

                // Sorting the positions `0..axis_size` of the slice by their keys yields the permutation that maps
                // every sorted position to the original position whose element it takes. The keys compare
                // lexicographically (i.e., the first key component that differs decides), descending sorts reverse
                // every comparison, and `sort_by` is stable, so elements that tie on every key keep their order.
                permutation.clear();
                permutation.extend(0..axis_size);
                permutation.sort_by(|&left, &right| {
                    order_keys
                        .iter()
                        .map(|keys| {
                            let left_key = keys[base + left * inner_stride];
                            let right_key = keys[base + right * inner_stride];
                            match direction {
                                SortDirection::Ascending => left_key.cmp(&right_key),
                                SortDirection::Descending => right_key.cmp(&left_key),
                            }
                        })
                        .find(|ordering| ordering.is_ne())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                for (target_position, &source_position) in permutation.iter().enumerate() {
                    gather[base + target_position * inner_stride] = base + source_position * inner_stride;
                }
            }
        }

        // Applying the gather map moves whole element encodings, so every input is permuted the same way without being
        // decoded.
        inputs
            .iter()
            .zip(input_types)
            .map(|(input, input_type)| input.gather_elements(input_type, |index| gather[index]))
            .collect()
    }
}

impl<
    V: Value<Type = ArrayType, DispatchDomain: Context<Operation: From<SortOperation>>>
        + ManualVariationAlignment<ArrayType>,
> Sort<ArrayType> for V
{
    fn sort_with_ordering<A: Into<Axis>>(
        inputs: &[Self],
        axis: A,
        key_count: usize,
        direction: SortDirection,
        ordering: SortOrdering,
    ) -> Result<Vec<Self>, ProgramError> {
        // Context-carrying values bind a `SortOperation` through the first input's context after aligning the manual
        // variation of the inputs. The `From<SortOperation>` bound keeps this implementation disjoint from the eager
        // reference arrays, whose dispatch domain does not provide a sort operation.
        let operation = SortOperation::from_sort_arguments(inputs, axis.into(), key_count, direction, ordering)?;
        let aligned_inputs = ManualVariationAlignment::align_manual_variation(inputs)?;
        inputs[0].dispatch_domain().bind(operation, Vec::new(), &aligned_inputs)
    }
}

// Composite values sort through their array views. Sorting has no receiver, which the shared projection macro cannot
// express, and so this projection and the concrete one below are written out.
impl<V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>>
    Sort<ArrayIrType> for V
where
    ProjectedValue<ArrayType, V>: Sort<ArrayType>,
{
    fn sort_with_ordering<A: Into<Axis>>(
        inputs: &[Self],
        axis: A,
        key_count: usize,
        direction: SortDirection,
        ordering: SortOrdering,
    ) -> Result<Vec<Self>, ProgramError> {
        let inputs = inputs
            .iter()
            .map(|input| ValueProjection::<ArrayType>::into_projected(input.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let outputs =
            ProjectedValue::<ArrayType, V>::sort_with_ordering(&inputs, axis, key_count, direction, ordering)?;
        Ok(outputs.into_iter().map(V::from_projected).collect())
    }
}

impl<A: Value<Type = ArrayType> + Sort<ArrayType>> Sort<ArrayIrType> for ArrayIrValue<A> {
    fn sort_with_ordering<AxisValue: Into<Axis>>(
        inputs: &[Self],
        axis: AxisValue,
        key_count: usize,
        direction: SortDirection,
        ordering: SortOrdering,
    ) -> Result<Vec<Self>, ProgramError> {
        let inputs = inputs
            .iter()
            .map(|input| ValueProjection::<ArrayType>::into_projected(input.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(A::sort_with_ordering(&inputs, axis, key_count, direction, ordering)?
            .into_iter()
            .map(Self::Array)
            .collect())
    }
}

/// Represents the ability to select the `k` largest elements of a value along one axis together with their indices.
/// Values rank by the IEEE 754 total order of [`SortOrdering::Total`] (i.e., `+NaN` ranks above `+∞`, `+0.0` ranks
/// above `-0.0`, and `-NaN` ranks below `-∞`), ties select the lowest index first, and the indices are `i32`.
/// Complex values are rejected, because they have no total order. These are the semantics of JAX's
/// [`jax.lax.top_k`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.top_k.html),
/// generalized from the trailing axis to any axis.
///
/// [`TopK`] is not a primitive operation. [`Self::top_k`] stably sorts the value in descending order together with an
/// `i32` index [`iota`](IotaOperation) passenger and slices the leading `k` entries of both sorted outputs, so every
/// program transform supports it through the rules of [`SortOperation`] and [`Slice`]. This staged form is the
/// sort-plus-slice idiom that the XLA backend's top-k rewriter replaces with its fast top-k implementation. That
/// rewriter only accepts `iota` or `broadcast(iota)` index passengers, while the StableHLO-to-HLO import turns the
/// index iota of an input with leading size-1 dimensions (e.g., `f32[1, 32000]`) into `reshape(iota)`. Therefore,
/// when the ranked axis is the trailing axis, the leading size-1 dimensions are reshaped away before the composition
/// and reinserted afterward, which leaves the values and indices unchanged.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and composite array IR values implement it for [`ArrayIrType`].
///
/// Note that composite values select through their array views, so that the composition (and its index `iota`)
/// runs in the array universe.
///
/// # Example
///
/// The following example selects the three largest scores, where the tie between the two `3.0` scores selects the
/// lower index first:
///
/// ```rust
/// # use ryft_core::{Array, ProgramError, TopK};
/// # fn main() -> Result<(), ProgramError> {
/// let (values, indices) = Array::vector(vec![1.0, 3.0, 2.0, 3.0])?.top_k(3, 0)?;
/// assert_eq!(values, Array::vector(vec![3.0, 3.0, 2.0])?);
/// assert_eq!(indices, Array::vector(vec![1i32, 3, 2])?);
/// # Ok(())
/// # }
/// ```
#[capability(projection(ArrayIrType => ArrayType))]
pub trait TopK<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Returns the `k` largest elements of this value along `axis` in descending order together with their indices,
    /// both with the `axis` dimension resized to `k`.
    ///
    /// # Parameters
    ///
    ///   - `k`: Number of elements to select, which must not exceed the size of `axis`.
    ///   - `axis`: [`Axis`] along which the elements are ranked. Negative axes count from the end.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if the value has complex elements or dynamic dimensions, if `axis` is out of bounds,
    /// if `k` exceeds the size of `axis`, or if executing or staging the composition fails.
    fn top_k<A: Into<Axis>>(&self, k: usize, axis: A) -> Result<(Self, Self), ProgramError>;
}

impl<V: Value<Type = ArrayType, DispatchDomain: Iota<V>> + Sort + Slice + Reshape> TopK<ArrayType> for V {
    fn top_k<A: Into<Axis>>(&self, k: usize, axis: A) -> Result<(Self, Self), ProgramError> {
        // Complex values have no total order, and the index passenger and the slices below need static extents,
        // so both are rejected before the axis is normalized and `k` is checked against its extent.
        let value_type = self.r#type();
        let data_type = value_type.data_type();
        if data_type.is_complex() {
            return Err(TypeError::invalid(format!("`top_k` does not support data type `{data_type}`")).into());
        }

        let dimensions = value_type
            .shape()
            .dimensions()
            .iter()
            .map(|dimension| {
                dimension.value().ok_or_else(|| ProgramError::UnsupportedOperation {
                    message: "`top_k` does not support dynamic dimensions".to_string(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let rank = dimensions.len();
        let axis = axis.into();
        let axis = axis
            .normalize(rank)
            .map_err(|_| TypeError::invalid(format!("`top_k` axis {axis} is out of bounds for rank {rank}")))?;
        if k > dimensions[axis] {
            return Err(ProgramError::InvalidArgument {
                message: format!("`top_k` `k` {k} exceeds size {} of axis {axis}", dimensions[axis]),
            });
        }

        // When the ranked axis is the trailing axis and follows unsharded size-1 dimensions, the composition runs on
        // a squeezed view without those dimensions, and both outputs are reshaped back to the original rank afterward.
        // The squeeze exists for XLA: its top-k rewriter only accepts `iota` or `broadcast(iota)` index passengers, and
        // the StableHLO-to-HLO import canonicalizes the degenerate higher-rank index iota of a batch-size-1 input
        // (e.g., `f32[1, 32000]`) into `reshape(iota)`, so without the squeeze such inputs would never reach XLA's fast
        // top-k implementation. The index iota must therefore be created at the squeezed shape (reshaping the
        // higher-rank iota would stage the same rejected `reshape(iota)` pattern), which is why the squeeze recurses
        // into `top_k` on the squeezed value instead of reshaping the outputs computed below. Values and indices are
        // identical either way, and the validation above already reported errors in terms of the original axis.
        let squeezed_count = dimensions[..axis].iter().take_while(|&&size| size == 1).count();
        let squeezes_sharded_dimension = value_type.sharding().is_some_and(|sharding| {
            sharding.dimensions()[..squeezed_count]
                .iter()
                .any(|dimension| matches!(dimension, ShardingDimension::Sharded(_)))
        });
        if axis + 1 == rank && squeezed_count > 0 && !squeezes_sharded_dimension {
            let squeezed_shape =
                Shape::new(dimensions[squeezed_count..].iter().copied().map(Dimension::Static).collect());
            let (values, indices) = self.reshape(squeezed_shape)?.top_k(k, axis - squeezed_count)?;
            let mut output_dimensions = dimensions.iter().copied().map(Dimension::Static).collect::<Vec<_>>();
            output_dimensions[axis] = Dimension::Static(k);
            let output_shape = Shape::new(output_dimensions);
            return Ok((values.reshape(output_shape.clone())?, indices.reshape(output_shape)?));
        }

        // The `i32` index passenger holds the index of every element along the ranked axis, and it is built through the
        // `Iota` capability of the dispatch domain of the value, which materializes it for eager arrays and stages an
        // `IotaOperation` for context-carrying values. The passenger shares the sharding of the value, including its
        // varying manual axes. An iota holds the same elements on every device, so typing it as varying is always valid
        // (i.e., it is what a `parallel_vary` of the invariant iota produces, and an integer iota has no cotangent for
        // that transition to affect), and it lets values that insert no variation transitions (e.g., eager arrays) sort
        // it together with the value. A stable descending sort under the total ordering then moves the largest elements
        // to the front of the axis, with ties keeping their original order (and so the lowest index first), and carries
        // the passenger along, so that it holds the original index of every sorted element.
        let index_type = ArrayType::new(DataType::I32, value_type.shape().clone())
            .with_sharding(value_type.sharding().cloned())
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        let indices = self.dispatch_domain().iota(&index_type, axis)?;
        let mut sorted = Self::sort_with_ordering(
            &[self.clone(), indices],
            axis,
            1,
            SortDirection::Descending,
            SortOrdering::Total,
        )?;
        let sorted_indices = sorted.pop().unwrap();
        let sorted_values = sorted.pop().unwrap();

        // The `k` leading entries of both sorted outputs along the ranked axis are the `k` largest values and their
        // indices, which is the sort-plus-slice idiom that XLA's top-k rewriter replaces with its fast implementation.
        let start_indices = vec![0; rank];
        let mut limit_indices = dimensions;
        limit_indices[axis] = k;
        let strides = vec![1; rank];
        Ok((
            sorted_values.slice(start_indices.as_slice(), limit_indices.as_slice(), strides.as_slice())?,
            sorted_indices.slice(start_indices.as_slice(), limit_indices.as_slice(), strides.as_slice())?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, DimensionBounds, DimensionVariable, Layout,
        LogicalMesh, MeshAxis, MeshAxisType, RaggedAxis, Sharding, StridedLayout, i4,
    };
    use crate::axes::NamedAxis;
    use crate::contexts::EagerContext;
    use crate::differentiation::{DifferentiationError, TransposableOperation, TranspositionContext};
    use crate::macros::{
        check_gradient, check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::operations::reductions::Reduce;
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder};
    use crate::tracing::{DomainTracer, DomainTracingContext, Trace, Tracer, TracingContext};

    use super::*;

    #[test]
    fn test_sort_direction() {
        assert_eq!(SortDirection::Ascending.to_string(), "ascending");
        assert_eq!(SortDirection::Descending.to_string(), "descending");
    }

    #[test]
    fn test_sort_ordering() {
        assert_eq!(SortOrdering::default(), SortOrdering::Canonical);
        assert_eq!(SortOrdering::Canonical.to_string(), "canonical");
        assert_eq!(SortOrdering::Total.to_string(), "total");
    }

    #[test]
    fn test_sort() {
        let operation = SortOperation::new(0, SortDirection::Ascending);
        assert_eq!(operation.name(), SORT_OPERATION_NAME);
        assert_eq!(operation.axis(), 0);
        assert_eq!(operation.direction(), SortDirection::Ascending);
        assert_eq!(operation.key_count(), NonZeroUsize::MIN);
        assert_eq!(operation.ordering(), SortOrdering::Canonical);
        assert_eq!(operation.to_string(), "sort [axis=0, direction=ascending]");
        assert_eq!(SortOperation::new(1, SortDirection::Descending).to_string(), "sort [axis=1, direction=descending]");

        // Non-default key counts and orderings render, while their defaults keep the rendering unchanged.
        let multi_key = operation.with_key_count(NonZeroUsize::new(2).unwrap()).with_ordering(SortOrdering::Total);
        assert_eq!(multi_key.key_count().get(), 2);
        assert_eq!(multi_key.ordering(), SortOrdering::Total);
        assert_eq!(multi_key.to_string(), "sort [axis=0, key_count=2, direction=ascending, ordering=total]");
        assert_eq!(multi_key.with_key_count(NonZeroUsize::MIN).with_ordering(SortOrdering::Canonical), operation);

        // A sort stages as one instruction with one output per input.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let keys = builder.add_input(ArrayType::new_static(DataType::F64, [4]));
        let values = builder.add_input(ArrayType::new_static(DataType::I32, [4]));
        let outputs = builder.add_instruction(multi_key, Vec::new(), vec![keys, values], None).unwrap().to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder, Placeholder], vec![Placeholder, Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[4], %1:i32[4] .
                let %2:f64[4], %3:i32[4] = sort [axis=0, key_count=2, direction=ascending, ordering=total] %0 %1
                in (%2, %3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_sort_type_inference() {
        let vector = ArrayType::new_static(DataType::F64, [4]);
        let complex = ArrayType::new_static(DataType::C64, [4]);
        let passenger = ArrayType::new_static(DataType::I32, [4]);
        check_operation_type_inference!(
            operation = SortOperation::new(0, SortDirection::Ascending),
            cases = [
                {
                    type = ArrayType,
                    input_types = [],
                    error = "`sort` needs at least one input",
                },
                {
                    input_types = [vector.clone(), ArrayType::new_static(DataType::F64, [3])],
                    error = "`sort` inputs must agree on shape but got [4] and [3]",
                },
                {
                    input_types = [ArrayType::new_static(DataType::Token, [4])],
                    error = "`sort` does not support key data type `token`",
                },
                {
                    input_types = [complex.clone(), passenger.clone()],
                    output_types = [complex.clone(), passenger.clone()],
                },
                {
                    input_types = [vector.clone(), passenger.clone()],
                    output_types = [vector.clone(), passenger.clone()],
                },
            ],
        );
        check_operation_type_inference!(
            operation = SortOperation::new(1, SortDirection::Ascending),
            cases = [{
                input_types = [vector.clone()],
                error = "`sort` axis 1 is out of bounds for rank 1",
            }],
        );

        // The total ordering has no order for complex keys, while complex passengers remain valid.
        check_operation_type_inference!(
            operation = SortOperation::new(0, SortDirection::Ascending).with_ordering(SortOrdering::Total),
            cases = [
                {
                    input_types = [complex.clone()],
                    error = "`sort` does not support key data type `c64` under the `total` ordering",
                },
                {
                    input_types = [vector.clone(), complex.clone()],
                    output_types = [vector.clone(), complex.clone()],
                },
            ],
        );

        // A multi-key sort validates every key: the key count must not exceed the input count, and every key data type
        // must be sortable, while passengers pass through unchanged.
        check_operation_type_inference!(
            operation = SortOperation::new(0, SortDirection::Ascending).with_key_count(NonZeroUsize::new(2).unwrap()),
            cases = [
                {
                    input_types = [vector.clone()],
                    error = "`sort` `key_count` 2 exceeds input count 1",
                },
                {
                    input_types = [vector.clone(), ArrayType::new_static(DataType::Zero, [4])],
                    error = "`sort` does not support key data type `zero`",
                },
                {
                    input_types = [vector.clone(), vector.clone(), passenger.clone()],
                    output_types = [vector.clone(), vector, passenger],
                },
            ],
        );
    }

    #[test]
    fn test_sort_type_inference_sharding() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let vector = ArrayType::new_static(DataType::F64, [4]);
        let invariant = vector.clone().with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap();
        let varying = vector
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let reduced = vector
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_reduced_axes(["x"]).unwrap())
            .unwrap();
        let unreduced = vector
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let sharded =
            vector.with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["y"])]).unwrap()).unwrap();

        // Inputs must share their manual variation, and reduced passengers keep their reduction state, while reduced
        // keys, unreduced inputs, and sharded sort axes are rejected.
        check_operation_type_inference!(
            operation = SortOperation::new(0, SortDirection::Ascending),
            cases = [
                {
                    input_types = [varying.clone(), invariant.clone()],
                    error = "`sort` inputs must have matching varying manual axes; insert `parallel_vary` on the \
                             inputs that lack an axis, as `align_manual_variation` does",
                },
                {
                    input_types = [varying.clone(), varying.clone()],
                    output_types = [varying.clone(), varying],
                },
                {
                    input_types = [invariant.clone(), reduced.clone()],
                    output_types = [invariant.clone(), reduced.clone()],
                },
                {
                    input_types = [reduced],
                    error = "`sort` does not support reduced keys",
                },
                {
                    input_types = [invariant, unreduced],
                    error = "`sort` does not support unreduced inputs",
                },
                {
                    input_types = [sharded],
                    error = "`sort` cannot sort along sharded axis 0",
                },
            ],
        );
    }

    #[test]
    fn test_sort_interpretation() {
        // An ascending key-value sort co-permutes the passenger by the key order, and the sort is stable: both `3.0`
        // keys keep their original relative order, so the first one's value `10` precedes `30`. Descending reverses
        // the key comparison while keeping equal keys in their original order.
        let keys = Array::vector(vec![3.0, 1.0, 3.0, 2.0]).unwrap();
        let values = Array::vector(vec![10i32, 20, 30, 40]).unwrap();
        assert_eq!(
            Array::sort(&[keys.clone(), values.clone()], 0, SortDirection::Ascending),
            Ok(vec![Array::vector(vec![1.0, 2.0, 3.0, 3.0]).unwrap(), Array::vector(vec![20i32, 40, 10, 30]).unwrap()]),
        );
        assert_eq!(
            Array::sort(&[keys, values], 0, SortDirection::Descending),
            Ok(vec![Array::vector(vec![3.0, 3.0, 2.0, 1.0]).unwrap(), Array::vector(vec![10i32, 30, 40, 20]).unwrap()]),
        );

        // Every column sorts independently along axis 0, and every row along axis 1. Negative axes count from the end.
        let matrix = Array::matrix(2, 3, vec![3.0, 1.0, 2.0, 0.0, 5.0, 4.0]).unwrap();
        assert_eq!(
            Array::sort(std::slice::from_ref(&matrix), 0, SortDirection::Ascending),
            Ok(vec![Array::matrix(2, 3, vec![0.0, 1.0, 2.0, 3.0, 5.0, 4.0]).unwrap()]),
        );
        assert_eq!(
            Array::sort(std::slice::from_ref(&matrix), 1, SortDirection::Ascending),
            Ok(vec![Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0]).unwrap()]),
        );
        assert_eq!(
            Array::sort(std::slice::from_ref(&matrix), -1, SortDirection::Ascending),
            Array::sort(std::slice::from_ref(&matrix), 1, SortDirection::Ascending),
        );
        assert!(matches!(
            Array::sort(std::slice::from_ref(&matrix), -3, SortDirection::Ascending),
            Err(ProgramError::Type(TypeError::Invalid { message, .. }))
                if message == "`sort` axis -3 is out of bounds for rank 2",
        ));

        // Two keys compare lexicographically: ties on key 0 fall through to key 1, and the full tie `(1.0, 9.0)` keeps
        // its original order (element 1 before element 3), which the passenger shows. Descending reverses every key
        // comparison while keeping full ties in their original order, so its result is not the ascending one reversed.
        let inputs = [
            Array::vector(vec![2.0, 1.0, 2.0, 1.0]).unwrap(),
            Array::vector(vec![5.0, 9.0, 4.0, 9.0]).unwrap(),
            Array::vector(vec![10i32, 20, 30, 40]).unwrap(),
        ];
        assert_eq!(
            Array::sort_with_key_count(&inputs, 0, 2, SortDirection::Ascending),
            Ok(vec![
                Array::vector(vec![1.0, 1.0, 2.0, 2.0]).unwrap(),
                Array::vector(vec![9.0, 9.0, 4.0, 5.0]).unwrap(),
                Array::vector(vec![20i32, 40, 30, 10]).unwrap(),
            ]),
        );
        assert_eq!(
            Array::sort_with_key_count(&inputs, 0, 2, SortDirection::Descending),
            Ok(vec![
                Array::vector(vec![2.0, 2.0, 1.0, 1.0]).unwrap(),
                Array::vector(vec![5.0, 4.0, 9.0, 9.0]).unwrap(),
                Array::vector(vec![10i32, 30, 20, 40]).unwrap(),
            ]),
        );

        // The eager implementation validates its inputs through type inference.
        assert!(matches!(
            Array::sort(&[], 0, SortDirection::Ascending),
            Err(ProgramError::Type(TypeError::Invalid { message, .. })) if message == "`sort` needs at least one input",
        ));
        assert!(matches!(
            Array::sort_with_key_count(&inputs[..1], 0, 2, SortDirection::Ascending),
            Err(ProgramError::Type(TypeError::Invalid { message, .. }))
                if message == "`sort` `key_count` 2 exceeds input count 1",
        ));
        assert!(matches!(
            Array::sort_with_key_count(&inputs, 0, 0, SortDirection::Ascending),
            Err(ProgramError::InvalidArgument { message }) if message == "`sort` `key_count` must be at least 1",
        ));
    }

    #[test]
    fn test_sort_interpretation_ordering() {
        // The canonical ordering treats `-0.0` and `+0.0` as equal and every NaN as equal and greater than `+∞`, so
        // stability keeps those ties in input order, which the index passenger shows. The total ordering separates
        // signed zeros and orders NaNs by sign. The expected permutations are those of JAX's `jax.lax.sort` and
        // `jax.lax.top_k`, respectively.
        let negative_nan = -f64::NAN;
        let keys = Array::vector(vec![f64::NAN, 0.0, -0.0, negative_nan, f64::NEG_INFINITY, 1.0]).unwrap();
        let indices = Array::vector(vec![0i32, 1, 2, 3, 4, 5]).unwrap();
        let sorted = Array::sort(&[keys.clone(), indices.clone()], 0, SortDirection::Ascending).unwrap();
        assert_eq!(sorted[1], Array::vector(vec![4i32, 1, 2, 5, 0, 3]).unwrap());
        assert_eq!(
            sorted[0].elements::<f64>().unwrap().iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            [f64::NEG_INFINITY, 0.0, -0.0, 1.0, f64::NAN, negative_nan].map(f64::to_bits).to_vec(),
        );
        let sorted =
            Array::sort_with_ordering(&[keys, indices], 0, 1, SortDirection::Descending, SortOrdering::Total).unwrap();
        assert_eq!(sorted[1], Array::vector(vec![0i32, 5, 1, 2, 4, 3]).unwrap());

        // Canonical complex keys order lexicographically by their canonicalized real and imaginary parts, so that the
        // signed-zero real parts tie and NaN parts order last.
        let keys = Array::vector(vec![
            Complex::new(1.0f64, f64::NAN),
            Complex::new(f64::NAN, 0.0),
            Complex::new(1.0, 2.0),
            Complex::new(1.0, 1.0),
            Complex::new(-0.0, 1.0),
            Complex::new(0.0, 1.0),
            Complex::new(0.0, 5.0),
        ])
        .unwrap();
        let indices = Array::vector(vec![0i32, 1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(
            Array::sort(&[keys, indices], 0, SortDirection::Ascending).unwrap()[1],
            Array::vector(vec![4i32, 5, 6, 3, 2, 0, 1]).unwrap(),
        );

        // Booleans order `false` before `true`, and signed and unsigned integers order by value.
        assert_eq!(
            Array::sort(&[Array::vector(vec![true, false, true]).unwrap()], 0, SortDirection::Ascending),
            Ok(vec![Array::vector(vec![false, true, true]).unwrap()]),
        );
        assert_eq!(
            Array::sort(&[Array::vector(vec![3i8, -1, 2, -128]).unwrap()], 0, SortDirection::Ascending),
            Ok(vec![Array::vector(vec![-128i8, -1, 2, 3]).unwrap()]),
        );
        assert_eq!(
            Array::sort(&[Array::vector(vec![3u64, u64::MAX, 0]).unwrap()], 0, SortDirection::Descending),
            Ok(vec![Array::vector(vec![u64::MAX, 3, 0]).unwrap()]),
        );
    }

    #[test]
    fn test_sort_interpretation_element_layouts() {
        // Non-key inputs sort by moving whole element encodings, so sub-byte passengers (which have no scalar
        // representation) ride an `f32` key without being decoded.
        let key = Array::vector(vec![3.0f32, 1.0, 2.0]).unwrap();
        let passenger = Array::from_elements(
            ArrayType::new_static(DataType::I4, [3]),
            &[i4::new(-8).unwrap(), i4::new(0).unwrap(), i4::new(7).unwrap()],
        )
        .unwrap();
        let outputs = Array::sort(&[key, passenger], 0, SortDirection::Ascending).unwrap();
        assert_eq!(outputs[0], Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        assert_eq!(outputs[1].storage_bytes(), [0x00, 0x07, 0x08]);

        // Sub-byte keys decode through arbitrary physical layouts, and outputs keep the layouts of their inputs.
        let key_type =
            ArrayType::new_static(DataType::I4, [3]).with_layout(Layout::Strided(StridedLayout::new(vec![-1])));
        let key =
            Array::from_elements(key_type.clone(), &[i4::new(3).unwrap(), i4::new(-2).unwrap(), i4::new(1).unwrap()])
                .unwrap();
        let output = Array::sort(&[key], 0, SortDirection::Ascending).unwrap().remove(0);
        assert_eq!(output.r#type().into_owned(), key_type);
        assert_eq!(output.elements::<i4>(), Ok(vec![i4::new(-2).unwrap(), i4::new(1).unwrap(), i4::new(3).unwrap()]));
    }

    #[test]
    fn test_sort_interpretation_manual_variation() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let invariant_type = ArrayType::new_static(DataType::I32, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1))
            .unwrap();
        let varying_type = ArrayType::new_static(DataType::I32, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();

        // Eager arrays insert no variation transitions, so mismatched manual variation is rejected, while matching
        // varying inputs sort and keep their variation.
        let invariant = Array::from_elements(invariant_type.clone(), &[1i32, 2]).unwrap();
        let varying = Array::from_elements(varying_type.clone(), &[2i32, 1]).unwrap();
        assert!(matches!(
            Array::sort(&[varying.clone(), invariant], 0, SortDirection::Ascending),
            Err(ProgramError::Type(TypeError::Invalid { message, .. }))
                if message == "`sort` inputs must have matching varying manual axes; insert `parallel_vary` on the \
                               inputs that lack an axis, as `align_manual_variation` does",
        ));
        assert_eq!(
            Array::sort(&[varying.clone(), varying], 0, SortDirection::Ascending),
            Ok(vec![
                Array::from_elements(varying_type.clone(), &[1i32, 2]).unwrap(),
                Array::from_elements(varying_type.clone(), &[1i32, 2]).unwrap(),
            ]),
        );

        // Traced values align their manual variation before binding the sort.
        let (outputs, program) =
            DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::trace_with_named_axes(
                |inputs| Sort::sort(&inputs, 0, SortDirection::Ascending),
                vec![varying_type.clone(), invariant_type],
                vec![("x".to_string(), NamedAxis::Mesh { mesh, axis: 0, size: 2 })],
            )
            .unwrap();
        assert_eq!(outputs, vec![varying_type.clone(), varying_type]);
        assert_eq!(
            program.instructions().iter().map(|instruction| instruction.operation().name()).collect::<Vec<_>>(),
            vec!["parallel_vary", "sort"],
        );
    }

    #[test]
    fn test_sort_partial_evaluation() {
        let keys = Array::vector(vec![2.0, 1.0, 2.0]).unwrap();
        let values = Array::vector(vec![5.0, 9.0, 4.0]).unwrap();
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = SortOperation::new(0, SortDirection::Ascending),
            cases = [
                {
                    inputs = [(@known, keys.clone())],
                    outputs = [(@known, Array::vector(vec![1.0, 2.0, 2.0]).unwrap())],
                    residual_instructions = 0,
                },
                {
                    inputs = [(@unknown(type = keys.r#type().into_owned(), replay = keys.clone()))],
                    outputs = [(@residual, Array::vector(vec![1.0, 2.0, 2.0]).unwrap())],
                    residual_instructions = 1,
                },
            ],
        );

        // A multi-key sort folds and residualizes like a single-key sort, resolving key-0 ties through key 1.
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = SortOperation::new(0, SortDirection::Ascending).with_key_count(NonZeroUsize::new(2).unwrap()),
            cases = [
                {
                    inputs = [(@known, keys.clone()), (@known, values.clone())],
                    outputs = [
                        (@known, Array::vector(vec![1.0, 2.0, 2.0]).unwrap()),
                        (@known, Array::vector(vec![9.0, 4.0, 5.0]).unwrap()),
                    ],
                    residual_instructions = 0,
                },
                {
                    inputs = [(@unknown(type = keys.r#type().into_owned(), replay = keys.clone())), (@known, values)],
                    outputs = [
                        (@residual, Array::vector(vec![1.0, 2.0, 2.0]).unwrap()),
                        (@residual, Array::vector(vec![9.0, 4.0, 5.0]).unwrap()),
                    ],
                    residual_instructions = 1,
                },
            ],
        );
    }

    #[test]
    fn test_sort_batching() {
        // A mapped key moves its batch axis to the leading position and the sort axis lifts past it, so each batch
        // item sorts independently. A replicated passenger broadcasts to the batched physical shape and is
        // co-permuted per batch item by that item's key order.
        check_operation_batching!(
            @exact,
            operation = SortOperation::new(0, SortDirection::Ascending),
            axis_size = 2,
            cases = [
                {
                    inputs = [(@mapped(axis = 0), Array::matrix(2, 2, vec![3.0, 1.0, 2.0, 5.0]).unwrap())],
                    outputs = [(@mapped(axis = 0), Array::matrix(2, 2, vec![1.0, 3.0, 2.0, 5.0]).unwrap())],
                },
                {
                    inputs = [(@mapped(axis = 1), Array::matrix(2, 2, vec![3.0, 2.0, 1.0, 5.0]).unwrap())],
                    outputs = [(@mapped(axis = 0), Array::matrix(2, 2, vec![1.0, 3.0, 2.0, 5.0]).unwrap())],
                },
                {
                    inputs = [
                        (@mapped(axis = 0), Array::matrix(2, 2, vec![3.0, 1.0, 2.0, 5.0]).unwrap()),
                        (@replicated, Array::vector(vec![7.0, 8.0]).unwrap()),
                    ],
                    outputs = [
                        (@mapped(axis = 0), Array::matrix(2, 2, vec![1.0, 3.0, 2.0, 5.0]).unwrap()),
                        (@mapped(axis = 0), Array::matrix(2, 2, vec![8.0, 7.0, 7.0, 8.0]).unwrap()),
                    ],
                },
                {
                    inputs = [(@replicated, Array::vector(vec![2.0, 1.0]).unwrap())],
                    outputs = [(@replicated, Array::vector(vec![1.0, 2.0]).unwrap())],
                },
            ],
        );

        // A multi-key total-order sort carries its key count and ordering through batching unchanged: item 0 ties on
        // key 0 and reorders by key 1, while item 1 orders `-0.0` before `+0.0` under the total ordering.
        check_operation_batching!(
            @exact,
            operation = SortOperation::new(0, SortDirection::Ascending)
                .with_key_count(NonZeroUsize::new(2).unwrap())
                .with_ordering(SortOrdering::Total),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::matrix(2, 2, vec![3.0, 3.0, 0.0, -0.0]).unwrap()),
                    (@mapped(axis = 0), Array::matrix(2, 2, vec![1.0, 0.0, 9.0, 8.0]).unwrap()),
                ],
                outputs = [
                    (@mapped(axis = 0), Array::matrix(2, 2, vec![3.0, 3.0, -0.0, 0.0]).unwrap()),
                    (@mapped(axis = 0), Array::matrix(2, 2, vec![0.0, 1.0, 8.0, 9.0]).unwrap()),
                ],
            }],
        );
    }

    #[test]
    fn test_sort_batching_ragged() {
        // Sorting would move padding into valid data, so bounded ragged batches are rejected.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
        let input =
            ArrayBatch::new(Array::matrix(2, 3, vec![2.0, 1.0, 0.0, 3.0, 0.0, 0.0]).unwrap(), BatchAxis::new(0))
                .unwrap()
                .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![2i32, 1]).unwrap(), length, vec![0])])
                .unwrap();
        let context = BatchingContext::new(EagerContext::<Array>::new(), 2);
        assert!(matches!(
            SortOperation::new(0, SortDirection::Ascending).batch(&context, &EmptyRegionDriver, &[input]),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "`sort` does not support bounded ragged dimension `length` on input 0",
        ));
    }

    #[test]
    fn test_sort_differentiation() {
        // The tangent rides the sort as a passenger, so it is co-permuted by the order of its primal key.
        check_operation_differentiation!(
            @approx(step = 1e-3, epsilon = 1e-6),
            operation = SortOperation::new(0, SortDirection::Ascending),
            cases = [{
                primals = [Array::vector(vec![3.0, 1.0, 2.0]).unwrap()],
                tangents = [Array::vector(vec![30.0, 10.0, 20.0]).unwrap()],
                primal_outputs = [Array::vector(vec![1.0, 2.0, 3.0]).unwrap()],
                tangent_outputs = [Array::vector(vec![10.0, 20.0, 30.0]).unwrap()],
                jvp = indoc! {"
                    lambda %0:f64[3], %1:f64[3] .
                    let %2:f64[3], %3:f64[3] = sort [axis=0, direction=ascending] %0 %1
                    in (%2, %3)
                "},
            }],
        );

        // A multi-key sort keeps its key count on the jvp sort, so the tangents (appended after every primal input)
        // ride as passengers permuted by the lexicographic order: the key-0 tie between elements 0 and 2 is resolved
        // by key 1. The tied key-0 elements carry equal tangents so that the tie survives the finite-difference
        // perturbation.
        check_operation_differentiation!(
            @approx(step = 1e-3, epsilon = 1e-6),
            operation = SortOperation::new(0, SortDirection::Ascending).with_key_count(NonZeroUsize::new(2).unwrap()),
            cases = [{
                primals = [
                    Array::vector(vec![2.0, 1.0, 2.0]).unwrap(),
                    Array::vector(vec![5.0, 9.0, 4.0]).unwrap(),
                    Array::vector(vec![7.0, 8.0, 9.0]).unwrap(),
                ],
                tangents = [
                    Array::vector(vec![10.0, 20.0, 10.0]).unwrap(),
                    Array::vector(vec![100.0, 200.0, 300.0]).unwrap(),
                    Array::vector(vec![1000.0, 2000.0, 3000.0]).unwrap(),
                ],
                primal_outputs = [
                    Array::vector(vec![1.0, 2.0, 2.0]).unwrap(),
                    Array::vector(vec![9.0, 4.0, 5.0]).unwrap(),
                    Array::vector(vec![8.0, 9.0, 7.0]).unwrap(),
                ],
                tangent_outputs = [
                    Array::vector(vec![20.0, 10.0, 10.0]).unwrap(),
                    Array::vector(vec![200.0, 300.0, 100.0]).unwrap(),
                    Array::vector(vec![2000.0, 3000.0, 1000.0]).unwrap(),
                ],
                jvp = indoc! {"
                    lambda %0:f64[3], %1:f64[3], %2:f64[3], %3:f64[3], %4:f64[3], %5:f64[3] .
                    let %6:f64[3], %7:f64[3], %8:f64[3], %9:f64[3], %10:f64[3], %11:f64[3] = \
                        sort [axis=0, key_count=2, direction=ascending] %0 %1 %2 %3 %4 %5
                    in (%6, %7, %8, %9, %10, %11)
                "},
            }],
        );
    }

    #[test]
    fn test_sort_differentiation_linearization() {
        // A linearization transfers only the keys into the tangent program, because the keys alone determine the
        // permutation, so the passenger primal is not retained as a residual.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let keys = builder.add_input(ArrayType::new_static(DataType::F64, [3]));
        let values = builder.add_input(ArrayType::new_static(DataType::F64, [3]));
        let operation = SortOperation::new(0, SortDirection::Descending);
        let outputs = builder.add_instruction(operation, Vec::new(), vec![keys, values], None).unwrap().to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder, Placeholder], vec![Placeholder, Placeholder])
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3], %2:f64[3] .
                let %3:f64[3], %4:f64[3], %5:f64[3] = sort [axis=0, direction=descending] %2 %0 %1
                in (%4, %5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_sort_differentiation_reverse_mode() {
        // Reverse mode transposes the permutation of the passengers, so the gradient of the sum of the two smallest
        // sorted values selects their original positions, for keys of either ordering and multi-key sorts alike, which
        // finite differences independently confirm.
        check_gradient!(
            |input| Sort::sort(std::slice::from_ref(&input), 0, SortDirection::Ascending)?
                .remove(0)
                .slice(&[0], &[2], &[1])?
                .reduce_sum(&[0], None),
            at = Array::vector(vec![3.0, 1.0, 2.0]).unwrap(),
            step = 1e-3,
            tolerance = 1e-6,
        );
        check_gradient!(
            |input| Sort::sort_with_ordering(
                &[input.clone(), input.clone() * input],
                1,
                1,
                SortDirection::Descending,
                SortOrdering::Total,
            )?
            .remove(1)
            .slice(&[0, 0], &[2, 1], &[1, 1])?
            .reduce_sum(&[0, 1], None),
            at = Array::matrix(2, 3, vec![3.0, 1.0, 2.0, -1.0, 4.0, 0.5]).unwrap(),
            step = 1e-3,
            tolerance = 1e-6,
        );
        check_gradient!(
            |input| Sort::sort_with_key_count(
                &[input.clone(), input.clone() * input.clone(), input.clone() + input.clone() + input],
                0,
                2,
                SortDirection::Ascending,
            )?
            .remove(2)
            .slice(&[0], &[2], &[1])?
            .reduce_sum(&[0], None),
            at = Array::vector(vec![3.0, 1.0, 2.0]).unwrap(),
            step = 1e-3,
            tolerance = 1e-6,
        );
    }

    #[test]
    fn test_sort_transposition() {
        // With a known key `[3, 1, 2]` (i.e., the permutation `[1, 2, 0]`), the passenger cotangent moves back to the
        // original positions of the sorted elements through a sort of the permutation itself.
        check_operation_transposition!(
            @exact,
            operation = SortOperation::new(0, SortDirection::Ascending),
            cases = [{
                inputs = [
                    (@known, Array::vector(vec![3.0, 1.0, 2.0]).unwrap()),
                    (@linear(type = ArrayType::new_static(DataType::F64, [3]))),
                ],
                output_cotangents = [
                    Array::vector(vec![0.0, 0.0, 0.0]).unwrap(),
                    Array::vector(vec![10.0, 20.0, 30.0]).unwrap(),
                ],
                input_cotangents = [Array::vector(vec![30.0, 10.0, 20.0]).unwrap()],
                pullback = indoc! {"
                    lambda %0:f64[3], %1:f64[3], %2:f64[3] .
                    let %3:i32[3] = iota [type=i32[3], dimension=0]
                        %4:f64[3], %5:i32[3] = sort [axis=0, direction=ascending] %2 %3
                        %6:i32[3], %7:f64[3] = sort [axis=0, direction=ascending] %5 %1
                    in (%7)
                "},
            }],
        );

        // Sorting is not linear in its keys, and structural-zero passenger cotangents stage nothing.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input_type = ArrayType::new_static(DataType::F64, [3]);
        let operation = SortOperation::new(0, SortDirection::Ascending);
        let inputs = [PartialValue::Unknown(input_type.clone()), PartialValue::Unknown(input_type.clone())];
        let mut rule_context = TranspositionContext::new(context.clone());
        let accumulators = rule_context.cotangent_accumulators(&inputs, &[]).unwrap();
        let cotangents = [MaybeZero::Zero(input_type.cotangent().unwrap()), MaybeZero::Zero(input_type.clone())];
        assert!(matches!(
            operation.transpose(&mut rule_context, &EmptyRegionDriver, &inputs, &cotangents, &accumulators),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "`sort` is not linear in its keys and requires known keys",
        ));
        let inputs =
            [PartialValue::Known(context.input(input_type.clone())), PartialValue::Unknown(input_type.clone())];
        let accumulators = rule_context.cotangent_accumulators(&inputs, &[]).unwrap();
        operation
            .transpose(&mut rule_context, &EmptyRegionDriver, &inputs, &cotangents, &accumulators)
            .unwrap();
        assert!(rule_context.take_cotangents(&accumulators).unwrap().iter().all(MaybeZero::is_zero));
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_sort_composite() {
        type CompositeContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        // Composite tracers stage the array sort and the array top-k composition of their members.
        let vector_type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3]));
        let (_, program) = CompositeContext::trace(
            |inputs: Vec<Tracer<CompositeContext>>| Tracer::sort(&inputs, 0, SortDirection::Ascending),
            vec![vector_type.clone(), vector_type],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3], %1:f32[3] .
                let %2:f32[3], %3:f32[3] = sort [axis=0, direction=ascending] %0 %1
                in (%2, %3)"
            },
        );

        // Concrete composite values sort and select through their array members.
        let keys = Array::vector(vec![3.0f32, 1.0, 2.0]).unwrap();
        let values = Array::vector(vec![1i32, 2, 3]).unwrap();
        let sorted = Array::sort(&[keys.clone(), values.clone()], 0, SortDirection::Ascending).unwrap();
        assert_eq!(
            ArrayIrValue::sort(
                &[ArrayIrValue::Array(keys.clone()), ArrayIrValue::Array(values)],
                0,
                SortDirection::Ascending,
            ),
            Ok(sorted.into_iter().map(ArrayIrValue::Array).collect()),
        );
        let (top_values, top_indices) = keys.top_k(2, 0).unwrap();
        assert_eq!(
            ArrayIrValue::Array(keys).top_k(2, 0),
            Ok((ArrayIrValue::Array(top_values), ArrayIrValue::Array(top_indices))),
        );
    }

    #[test]
    fn test_top_k() {
        // Ties select the lowest index first because the descending ranking sort is stable, and values rank by the
        // total order, so `+NaN` ranks first, `+0.0` ranks above `-0.0`, and `-NaN` ranks last, which are the
        // indices that JAX's `jax.lax.top_k` returns.
        let input = Array::vector(vec![3.0, 1.0, 3.0, -0.0, 0.0, 2.0]).unwrap();
        assert_eq!(
            input.top_k(3, 0),
            Ok((Array::vector(vec![3.0, 3.0, 2.0]).unwrap(), Array::vector(vec![0i32, 2, 5]).unwrap())),
        );
        let input = Array::vector(vec![f64::NAN, 2.0, -f64::NAN, 0.0, -0.0]).unwrap();
        assert_eq!(input.top_k(5, 0).unwrap().1, Array::vector(vec![0i32, 1, 3, 4, 2]).unwrap());

        // Every column of a matrix ranks independently along a non-trailing axis as well, and negative axes count from
        // the end.
        let matrix = Array::matrix(2, 3, vec![3.0, 1.0, 2.0, 0.0, 5.0, 4.0]).unwrap();
        assert_eq!(
            matrix.top_k(1, 0),
            Ok((Array::matrix(1, 3, vec![3.0, 5.0, 4.0]).unwrap(), Array::matrix(1, 3, vec![0i32, 1, 1]).unwrap())),
        );
        assert_eq!(matrix.top_k(1, -2), matrix.top_k(1, 0));

        // Oversized `k`, complex values, and out-of-bounds axes are rejected.
        assert!(matches!(
            input.top_k(6, 0),
            Err(ProgramError::InvalidArgument { message }) if message == "`top_k` `k` 6 exceeds size 5 of axis 0",
        ));
        assert!(matches!(
            Array::vector(vec![Complex::new(1.0f32, 0.0)]).unwrap().top_k(1, 0),
            Err(ProgramError::Type(TypeError::Invalid { message, .. }))
                if message == "`top_k` does not support data type `c64`",
        ));
        assert!(matches!(
            input.top_k(1, 1),
            Err(ProgramError::Type(TypeError::Invalid { message, .. }))
                if message == "`top_k` axis 1 is out of bounds for rank 1",
        ));
        assert!(matches!(
            input.top_k(1, -2),
            Err(ProgramError::Type(TypeError::Invalid { message, .. }))
                if message == "`top_k` axis -2 is out of bounds for rank 1",
        ));
    }

    #[test]
    fn test_top_k_squeezed_view() {
        // A trailing ranked axis behind leading size-1 dimensions squeezes those dimensions away before the
        // composition and reinserts them afterward, so values and indices match the rank-1 results with the leading
        // size-1 dimensions restored.
        let input = Array::matrix(1, 6, vec![3.0, 1.0, 3.0, -0.0, 0.0, 2.0]).unwrap();
        assert_eq!(
            input.top_k(3, 1),
            Ok((Array::matrix(1, 3, vec![3.0, 3.0, 2.0]).unwrap(), Array::matrix(1, 3, vec![0i32, 2, 5]).unwrap())),
        );
        let input = Array::from_elements(ArrayType::new_static(DataType::F64, [1, 1, 3]), &[3.0, 1.0, 3.0]).unwrap();
        assert_eq!(
            input.top_k(2, 2),
            Ok((
                Array::from_elements(ArrayType::new_static(DataType::F64, [1, 1, 2]), &[3.0, 3.0]).unwrap(),
                Array::from_elements(ArrayType::new_static(DataType::I32, [1, 1, 2]), &[0i32, 2]).unwrap(),
            )),
        );

        // The out-of-bounds `k` error names the caller's axis rather than the squeezed axis.
        assert!(matches!(
            Array::matrix(1, 2, vec![1.0, 2.0]).unwrap().top_k(3, 1),
            Err(ProgramError::InvalidArgument { message }) if message == "`top_k` `k` 3 exceeds size 2 of axis 1",
        ));
    }

    #[test]
    fn test_top_k_staging() {
        // Staging `top_k` composes the sort-plus-slice idiom that XLA's top-k rewriter recognizes: an index iota
        // rides a descending total-order sort as a passenger and both outputs are sliced to the leading `k` entries.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok(x.top_k(2, 0)?.0),
            ArrayType::new_static(DataType::F64, [4]),
        )
        .unwrap();
        assert_eq!(
            program.to_flat_program().to_string(),
            indoc! {"
                lambda %0:f64[4] .
                let %1:i32[4] = iota [type=i32[4], dimension=0]
                    %2:f64[4], %3:i32[4] = sort [axis=0, direction=descending, ordering=total] %0 %1
                    %4:f64[2] = slice [start_indices=[0], limits=[2]] %2
                    %5:i32[2] = slice [start_indices=[0], limits=[2]] %3
                in (%4)
            "}
            .trim_end(),
        );

        // Along the trailing axis (here given as `-1`) of a batch-size-1 input, the leading size-1 dimension is
        // squeezed away before the composition, so the index passenger is a rank-1 iota (not the `reshape(iota)` that
        // XLA's top-k rewriter rejects), and both outputs are reshaped back to the original rank afterward.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok(x.top_k(2, -1)?.0),
            ArrayType::new_static(DataType::F64, [1, 4]),
        )
        .unwrap();
        assert_eq!(
            program.to_flat_program().to_string(),
            indoc! {"
                lambda %0:f64[1, 4] .
                let %1:f64[4] = reshape [shape=[4]] %0
                    %2:i32[4] = iota [type=i32[4], dimension=0]
                    %3:f64[4], %4:i32[4] = sort [axis=0, direction=descending, ordering=total] %1 %2
                    %5:f64[2] = slice [start_indices=[0], limits=[2]] %3
                    %6:i32[2] = slice [start_indices=[0], limits=[2]] %4
                    %7:f64[1, 2] = reshape [shape=[1, 2]] %5
                    %8:i32[1, 2] = reshape [shape=[1, 2]] %6
                in (%7)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_top_k_manual_variation() {
        // The index passenger shares the varying manual axes of the ranked value, so eager arrays (which insert no
        // variation transitions) rank varying values, and traced values stage no `parallel_vary` for the passenger.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap();
        let input_type = ArrayType::new_static(DataType::F64, [3]).with_sharding(sharding.clone()).unwrap();
        let values_type = ArrayType::new_static(DataType::F64, [2]).with_sharding(sharding.clone()).unwrap();
        let indices_type = ArrayType::new_static(DataType::I32, [2]).with_sharding(sharding).unwrap();
        assert_eq!(
            Array::from_elements(input_type.clone(), &[1.0, 3.0, 2.0]).unwrap().top_k(2, 0),
            Ok((
                Array::from_elements(values_type.clone(), &[3.0, 2.0]).unwrap(),
                Array::from_elements(indices_type.clone(), &[1i32, 2]).unwrap(),
            )),
        );
        let (outputs, program) =
            DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::trace_with_named_axes(
                |inputs| {
                    let (values, indices) = inputs[0].top_k(2, 0)?;
                    Ok(vec![values, indices])
                },
                vec![input_type],
                vec![("x".to_string(), NamedAxis::Mesh { mesh, axis: 0, size: 2 })],
            )
            .unwrap();
        assert_eq!(outputs, vec![values_type, indices_type]);
        assert_eq!(
            program.instructions().iter().map(|instruction| instruction.operation().name()).collect::<Vec<_>>(),
            vec!["iota", "sort", "slice", "slice"],
        );
    }
}
