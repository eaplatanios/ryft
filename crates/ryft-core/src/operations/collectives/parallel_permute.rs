use std::collections::BTreeSet;
use std::fmt::Display;

use ryft_macros::capability;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayIrValue,
    ArrayOperation, ArrayType, LogicalMesh, MeshAxisType, RaggedAxis, ShardingDimension,
};
use crate::axes::{AxisError, NamedAxes, NamedAxis};
use crate::batching::{BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    CotangentAccumulator, DifferentiableOperation, DifferentiationContext, DifferentiationDriver, DifferentiationDual,
    DifferentiationError, DifferentiationPolicy, TransposableOperation, TranspositionContext, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::Capability;
use crate::operations::arithmetic::AddOperation;
use crate::operations::collectives::parallel_vary::{PARALLEL_VARY_OPERATION_NAME, ParallelVary};
use crate::operations::collectives::{LinearCollectiveOperation, resolve_named_axis_size, validate_manual_mesh_input};
use crate::operations::constants::zero_like::ZeroLike;
use crate::operations::manipulation::concatenation::Concatenate;
use crate::operations::manipulation::slicing::Slice;
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::sharding::Reshard;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, ProgramError, ProjectedValue, RegionInterface, TypeError, Typed, Value,
    ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

/// Canonical operation name for [`ParallelPermuteOperation`].
pub const PARALLEL_PERMUTE_OPERATION_NAME: &str = "parallel_permute";

/// [`Operation`] that routes input arrays between positions along the named axis according to explicit
/// `(source, target)` pairs. Both indices are zero-based coordinates along `axis_name`, in `0..axis_size`, rather than
/// global device IDs or element indices within an input array. Each position represents one execution of the enclosing
/// function with its own input array. A pair sends the entire input array at `source` to the output at `target`.
/// Sources must be unique, targets must be unique, and positions that no pair targets receive zeros. The output
/// type is the input type.
///
/// For example, inside `shard_map` over a manual mesh axis `x` of size three, `(0, 2)` sends the input at mesh
/// coordinate `x = 0` to the output at `x = 2`. On a multidimensional mesh, this routing is repeated separately for
/// each fixed combination of coordinates along the other axes. The physical devices at these coordinates can have
/// arbitrary device IDs; the pair still uses axis coordinates `0` and `2`.
///
/// Inside [`batch`](crate::batch) over a named axis of size three, `(0, 2)` instead sends batch item zero's input array
/// to batch item two's output. For either interpretation, if positions zero, one, and two hold arrays `A`, `B`, and
/// `C`, then pairs `[(0, 1), (2, 0)]` produce `C`, `A`, and a zero array at those respective positions.
///
/// This is the Ryft analogue of JAX's
/// [`jax.lax.ppermute`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ppermute.html) and StableHLO's
/// [`collective_permute`](https://openxla.org/stablehlo/spec#collective_permute).
///
/// A permutation over a manual mesh axis is created by [`with_mesh`](Self::with_mesh), and
/// [`ParallelPermute::parallel_permute`] supplies the mesh automatically from the enclosing manual region. Such a
/// permutation generally gives the axis positions different values, so its input must vary over the axis (refer to
/// [`ParallelVary`]) and its output varies over it as well. Pending cross-device sums over that axis are rejected; sums
/// over unrelated manual axes are preserved. An ordinary permutation carries no mesh and preserves the input's mesh
/// variation and pending sums, even when its input carries a manual mesh axis with the same name, because a `batch`
/// level whose axis name shadows that mesh axis may bind it instead. The collective is linear and its transpose is the
/// permutation with every pair inverted, over the same mesh. Outside any binder, the single position of a degenerate
/// axis keeps its value when the pair `(0, 0)` is present and receives zeros otherwise.
///
/// A matching `batch` level consumes the mapped batch axis of an ordinary permutation by reassembling it in target
/// order from per-item slices, with zero slices at untargeted positions, and passes a replicated input through
/// unchanged when every position is targeted. Unlike JAX's batching rule, which requires a full permutation, partial
/// permutations are supported. Bounded ragged extents follow the same source-to-target routing as their packed values,
/// and untargeted positions receive zero extents together with their zero-filled values. A permutation over a manual
/// mesh axis cannot be consumed by a `batch` level.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParallelPermuteOperation {
    /// Axis name referenced by this collective.
    axis_name: String,

    /// Number of participants along the named axis, resolved from the active [`NamedAxes`] environment
    /// when the operation is staged.
    axis_size: usize,

    /// Pairs of zero-based `(source, target)` coordinates along the named axis. Each pair routes the entire input
    /// array at `source` to the output at `target`.
    source_target_pairs: Vec<(usize, usize)>,

    /// Refer to the documentation of [`mesh`](Self::mesh) for more information.
    mesh: Option<LogicalMesh>,
}

impl ParallelPermuteOperation {
    /// Creates a new [`ParallelPermuteOperation`] over the axis with the provided name and resolved axis size.
    #[inline]
    pub fn new(axis_name: String, axis_size: usize, source_target_pairs: Vec<(usize, usize)>) -> Self {
        Self { axis_name, axis_size, source_target_pairs, mesh: None }
    }

    /// Returns this [`ParallelPermuteOperation`] configured to permute over a manual axis of `mesh`. The input must
    /// vary over [`axis_name`](Self::axis_name) on that mesh, whose size must equal [`axis_size`](Self::axis_size).
    /// Type inference validates these requirements. [`ParallelPermute::parallel_permute`] supplies the mesh
    /// automatically from the enclosing manual region.
    #[inline]
    pub fn with_mesh(mut self, mesh: LogicalMesh) -> Self {
        self.mesh = Some(mesh);
        self
    }

    /// Returns the axis name referenced by this collective.
    #[inline]
    pub fn axis_name(&self) -> &str {
        &self.axis_name
    }

    /// Returns the number of participants along the named axis.
    #[inline]
    pub fn axis_size(&self) -> usize {
        self.axis_size
    }

    /// Returns the `(source, target)` pairs of participant positions along the named axis.
    #[inline]
    pub fn source_target_pairs(&self) -> &[(usize, usize)] {
        self.source_target_pairs.as_slice()
    }

    /// Returns the logical mesh whose manual axis this [`ParallelPermuteOperation`] permutes over, or [`None`] for an
    /// ordinary permutation, whose named axis may be bound by any enclosing binder. Only a permutation over a manual
    /// mesh axis requires its input to vary over that axis.
    #[inline]
    pub fn mesh(&self) -> Option<&LogicalMesh> {
        self.mesh.as_ref()
    }
}

impl Display for ParallelPermuteOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ParallelPermuteOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        PARALLEL_PERMUTE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        // The permutation preserves its single input's type, including dynamic dimensions.
        let input_type = self.check_input(input_types, region_interfaces)?;
        let axis_name = &self.axis_name;
        let mut sources = BTreeSet::new();
        let mut targets = BTreeSet::new();
        for &(source, target) in &self.source_target_pairs {
            if source >= self.axis_size || target >= self.axis_size {
                return Err(TypeError::invalid(format!(
                    "`{}` pair ({}, {}) is out of bounds for axis size {}",
                    PARALLEL_PERMUTE_OPERATION_NAME, source, target, self.axis_size,
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
        // invariant over that axis would yield an output whose type wrongly claims that it is invariant. An ordinary
        // permutation never inspects the input's mesh, including its pending sums, because a `batch` level may bind a
        // shadowing axis name.
        if let Some(mesh) = &self.mesh {
            validate_manual_mesh_input(
                PARALLEL_PERMUTE_OPERATION_NAME,
                axis_name,
                Some(self.axis_size),
                mesh,
                input_type,
            )?;

            // Routing values cannot complete a pending sum over the same axis. Sums over unrelated axes commute with
            // this permutation and retain their pending state.
            let sharding = input_type.sharding().unwrap();
            if sharding.unreduced_axes().contains(axis_name) {
                return Err(TypeError::invalid(format!(
                    "`{PARALLEL_PERMUTE_OPERATION_NAME}` does not support unreduced inputs",
                )));
            }

            if !sharding.varying_manual_axes().contains(axis_name) {
                return Err(TypeError::invalid(format!(
                    "`{}` input must vary over manual axis `{}`; pass an invariant value \
                     through `{}` first so that the permuted output is typed as varying",
                    PARALLEL_PERMUTE_OPERATION_NAME, axis_name, PARALLEL_VARY_OPERATION_NAME,
                )));
            }
        }

        Ok(vec![input_type.clone()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, PARALLEL_PERMUTE_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("axis_name", format_args!("{:?}", self.axis_name))?;
            operation.field("axis_size", self.axis_size)?;
            operation.field("source_target_pairs", format_args!("{:?}", &self.source_target_pairs))?;
            if let Some(value) = &self.mesh {
                operation.field("mesh", value)?;
            }
            Ok(())
        })
    }
}

impl LinearCollectiveOperation for ParallelPermuteOperation {
    type Adjoint = ParallelPermuteOperation;

    #[inline]
    fn axis_name(&self) -> &str {
        &self.axis_name
    }

    #[inline]
    fn axis_size(&self) -> usize {
        self.axis_size
    }

    #[inline]
    fn mesh(&self) -> Option<&LogicalMesh> {
        self.mesh.as_ref()
    }

    #[inline]
    fn adjoint(&self, _input_type: &ArrayType) -> Result<ParallelPermuteOperation, ProgramError> {
        // Sending along `(source, target)` pulls cotangents back along `(target, source)`, so the input cotangent
        // is the permutation with every pair inverted, over the same axis and mesh.
        Ok(ParallelPermuteOperation {
            source_target_pairs: self.source_target_pairs.iter().map(|(source, target)| (*target, *source)).collect(),
            ..self.clone()
        })
    }

    #[inline]
    fn adapt_to_batch_axis(&self, input_batch_axis: usize) -> (Self, usize) {
        // A permutation preserves the shape of its input, so it applies to the packed values of a mapped input
        // unchanged and leaves the mapped axis where it is.
        (self.clone(), input_batch_axis)
    }
}

impl<C: Domain<Type = ArrayType, Value: ZeroLike>> InterpretableOperation<C> for ParallelPermuteOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[<C as Domain>::Value],
    ) -> Result<Vec<<C as Domain>::Value>, ProgramError> {
        // Eager binding does not infer output types, so interpretation validates the shared input contract and the
        // operation payload before applying the degenerate-axis rule.
        check_count!("input", inputs, 1, ProgramError);
        self.check_degenerate_interpretation()?;
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(&input_types, &[])?;

        // With one participant, the only valid pairs are none at all and `(0, 0)`. Without a pair, nothing targets
        // the participant, so it receives zeros.
        let input = &inputs[0];
        Ok(vec![if self.source_target_pairs.is_empty() { input.zero_like()? } else { input.clone() }])
    }
}

impl<C: Context<Type = ArrayType, Operation: From<ParallelPermuteOperation>>> PartiallyEvaluatableOperation<C>
    for ParallelPermuteOperation
{
}

impl<
    C: Context<
            Type = ArrayType,
            Value: ZeroLike + Concatenate + Slice + Transpose + Reshard,
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
            return context.forward_collective(self, inputs);
        }

        self.reject_mesh_form()?;

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

impl<C: Context<Type = ArrayType, Operation: From<ParallelPermuteOperation>>> DifferentiableOperation<C>
    for ParallelPermuteOperation
{
    #[inline]
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        self.linear_collective_jvp(context, inputs)
    }
}

impl<
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType> + From<AddOperation<ArrayType>> + From<ParallelPermuteOperation>,
> TransposableOperation<V, O> for ParallelPermuteOperation
{
    #[inline]
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        self.linear_collective_transpose(context, inputs, outputs, accumulators)
    }
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
/// The type-family parameter defaults to this value's type, so that homogeneous array values and composite array
/// values, which permute through their array views, share the same call syntax.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and composite array IR values implement it for [`ArrayIrType`].
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
#[capability]
pub trait ParallelPermute<T = <Self as Capability>::Universe>: Capability + Sized {
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
    /// if two pairs share a source or a target, or if this value carries a pending sum over the participating manual
    /// mesh axis.
    fn parallel_permute(&self, axis_name: &str, source_target_pairs: Vec<(usize, usize)>)
    -> Result<Self, ProgramError>;

    /// Returns this value shuffled across the participants of the named axis `axis_name`, where participant `target`
    /// receives the value of participant `permutation[target]`. A permutation shorter than the axis shuffles only the
    /// leading participants, and every remaining participant receives zeros. This is the Ryft analogue of JAX's
    /// [`jax.lax.pshuffle`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.pshuffle.html), and it permutes with
    /// the pair `(permutation[target], target)` for every target through
    /// [`parallel_permute`](Self::parallel_permute), whose manual-axis
    /// behavior therefore applies to shuffles as well.
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
    /// of [`parallel_permute`](Self::parallel_permute) otherwise (e.g., when `permutation` is longer than the axis).
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

impl ParallelPermute<ArrayType> for Array {
    // A concrete `Array` never executes inside an axis binder, because the values under a `batch` level or inside
    // a manual region are tracers, so every axis name is unbound for it.

    #[inline]
    fn parallel_permute(
        &self,
        axis_name: &str,
        _source_target_pairs: Vec<(usize, usize)>,
    ) -> Result<Self, ProgramError> {
        Err(AxisError::UnboundAxisName { name: axis_name.to_string() }.into())
    }
}

impl<A: Value<Type = ArrayType> + ParallelPermute<ArrayType>> ParallelPermute<ArrayIrType> for ArrayIrValue<A> {
    #[inline]
    fn parallel_permute(
        &self,
        axis_name: &str,
        source_target_pairs: Vec<(usize, usize)>,
    ) -> Result<Self, ProgramError> {
        let input = <Self as ValueProjection<ArrayType>>::projected(self)?;
        Ok(Self::Array(input.parallel_permute(axis_name, source_target_pairs)?))
    }
}

impl<
    V: Value<Type = ArrayType, DispatchDomain: Context<Operation: From<ParallelPermuteOperation>> + NamedAxes>
        + ParallelVary,
> ParallelPermute<ArrayType> for V
{
    fn parallel_permute(
        &self,
        axis_name: &str,
        source_target_pairs: Vec<(usize, usize)>,
    ) -> Result<Self, ProgramError> {
        // Any context-carrying value permutes by resolving the axis size from the active `NamedAxes` environment and
        // binding a `ParallelPermuteOperation` through its own context. Over a manual mesh axis, the operation records
        // the mesh, and an input that is still invariant over the axis is first made varying, exactly as JAX's
        // `ppermute` does, so that the output type records that the participants hold different values.
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        let mut operation = ParallelPermuteOperation::new(axis_name.to_string(), axis_size, source_target_pairs);
        let mut input = self.clone();
        if let Some(NamedAxis::Mesh { mesh, .. }) = context.named_axis(axis_name) {
            if input.r#type().unreduced_axes().contains(axis_name) {
                return Err(TypeError::invalid(format!(
                    "`{PARALLEL_PERMUTE_OPERATION_NAME}` does not support unreduced inputs",
                ))
                .into());
            }
            if !input.r#type().sharding().is_some_and(|sharding| sharding.varying_manual_axes().contains(axis_name)) {
                input = input.parallel_vary(axis_name)?;
            }
            operation = operation.with_mesh(mesh);
        }
        let mut outputs = context.bind(operation, Vec::new(), &[input])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>>
    ParallelPermute<ArrayIrType> for V
where
    ProjectedValue<ArrayType, V>: ParallelPermute<ArrayType>,
{
    #[inline]
    fn parallel_permute(
        &self,
        axis_name: &str,
        source_target_pairs: Vec<(usize, usize)>,
    ) -> Result<Self, ProgramError> {
        // A composite value permutes through its array view, which owns named-axis resolution and manual-axis
        // variation, so that every permutation shares one staging path.
        Ok(V::from_projected(
            ValueProjection::<ArrayType>::into_projected(self.clone())?
                .parallel_permute(axis_name, source_target_pairs)?,
        ))
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
fn route_axis_slices<V: Value<Type = ArrayType> + ZeroLike + Concatenate + Slice + Transpose + Reshard>(
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
    let output_sharding = value.r#type().sharding().cloned();

    // Extent-one slices cannot retain an explicit mesh placement whose axis product exceeds one.
    // Route slices with this axis replicated, then restore its placement once on the complete result.
    let value = if let Some(sharding) = &output_sharding
        && matches!(sharding.dimensions().first(), Some(ShardingDimension::Sharded(axes)) if axes.iter().any(|axis| {
            sharding.mesh().axis_type(axis) == Some(MeshAxisType::Explicit)
                && sharding.mesh().axis_size(axis).is_some_and(|size| size > 1)
        })) {
        let mut dimensions = sharding.dimensions().to_vec();
        dimensions[0] = ShardingDimension::Replicated;
        let replicated = sharding.with_dimensions(dimensions).map_err(TypeError::from)?;
        value.reshard(&replicated)?
    } else {
        value
    };

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

    let mut permuted = Concatenate::concatenate(&items, 0)?;
    if let Some(sharding) = output_sharding
        && permuted.r#type().sharding() != Some(&sharding)
    {
        permuted = permuted.reshard(&sharding)?;
    }

    if axis == 0 { Ok(permuted) } else { Ok(permuted.move_axis(0, axis)?) }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::batching::DynamicArrayExtentBatchingPolicy;
    use crate::arrays::{
        ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrValue, DataType, Dimension, DimensionBounds, DimensionValue,
        DimensionVariable, Layout, Memory, MeshAxis, Shape, Sharding, StridedLayout,
    };
    use crate::batching::{BatchAxis, BatchAxisSpecification, BatchingTracer, batch};
    use crate::contexts::{EagerContext, ProjectedContext, StagingContext};
    use crate::differentiation::differentiate_at;
    use crate::macros::{check_gradient, check_operation_partial_evaluation, check_operation_type_inference};
    use crate::operations::collectives::tests::{batch_collective, collective_program, eager_collective_context};
    use crate::operations::reductions::{Reduce, ReductionKind};
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationOutput, PartialEvaluationValue};
    use crate::programs::{EmptyRegionDriver, Program};
    use crate::tests::hash_of;

    use super::*;

    /// Creates a manual mesh whose axis `"x"` has two participants and whose axis `"y"` has one.
    fn manual_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 1, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap()
    }

    /// Traces [`ParallelPermute::parallel_permute`] over axis `"x"` with `source_target_pairs` under a named `batch`
    /// level of extent 2 that binds `"x"` inside a manual region over [`manual_mesh`], so that the batch level shadows
    /// the manual mesh axis with the same name, and returns the traced output type and program.
    fn trace_permutation_shadowing_manual_axis(
        input_type: ArrayType,
        source_target_pairs: Vec<(usize, usize)>,
    ) -> (ArrayType, Program<Array, ArrayOperation<Array>, Array, Array>) {
        TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| {
                let context = BatchingContext::<_, ArrayBatchingPolicy>::new(input.dispatch_domain(), 2)
                    .with_axis_name("x".to_string());
                let input = BatchingTracer::new(context, ArrayBatch::new(input, BatchAxis::new(0))?);
                Ok(input.parallel_permute("x", source_target_pairs)?.into_batch().into_value())
            },
            input_type,
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: manual_mesh() })],
        )
        .unwrap()
    }

    #[test]
    fn test_parallel_permute() {
        let operation = ParallelPermuteOperation::new("x".to_string(), 3, vec![(0, 1), (2, 0)]);
        assert_eq!(operation.name(), PARALLEL_PERMUTE_OPERATION_NAME);
        assert_eq!(operation.axis_name(), "x");
        assert_eq!(operation.axis_size(), 3);
        assert_eq!(operation.source_target_pairs(), &[(0, 1), (2, 0)]);
        assert_eq!(operation.mesh(), None);
        assert_eq!(
            operation.to_string(),
            "parallel_permute [axis_name=\"x\", axis_size=3, source_target_pairs=[(0, 1), (2, 0)]]",
        );
    }

    #[test]
    fn test_parallel_permute_with_mesh() {
        // A permutation over a manual mesh axis records and renders its mesh.
        let mesh_permutation = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1)]).with_mesh(manual_mesh());
        assert_eq!(mesh_permutation.mesh(), Some(&manual_mesh()));
        assert_eq!(
            mesh_permutation.to_string(),
            indoc! {"
                parallel_permute [
                    axis_name=\"x\",
                    axis_size=2,
                    source_target_pairs=[(0, 1)],
                    mesh=['x'=2:manual, 'y'=1:manual],
                ]"
            },
        );
    }

    #[test]
    fn test_parallel_permute_equality_and_hashing() {
        // Independently constructed operations are equal, and hash alike, when their axes, pairs, and meshes agree.
        let ordinary = ParallelPermuteOperation::new("x".to_string(), 3, vec![(0, 1), (2, 0)]);
        let same_ordinary = ParallelPermuteOperation::new("x".to_string(), 3, vec![(0, 1), (2, 0)]);
        let mesh_permutation = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1)]).with_mesh(manual_mesh());
        let same_mesh_permutation =
            ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1)]).with_mesh(manual_mesh());
        assert_eq!(ordinary, same_ordinary);
        assert_eq!(mesh_permutation, same_mesh_permutation);
        assert_eq!(hash_of(&ordinary), hash_of(&same_ordinary));
        assert_eq!(hash_of(&mesh_permutation), hash_of(&same_mesh_permutation));

        // Every attribute participates in the identity.
        let unmeshed = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1)]);
        assert_ne!(ordinary, ParallelPermuteOperation::new("x".to_string(), 3, vec![(0, 1)]));
        assert_ne!(ordinary, ParallelPermuteOperation::new("x".to_string(), 4, vec![(0, 1), (2, 0)]));
        assert_ne!(ordinary, ParallelPermuteOperation::new("y".to_string(), 3, vec![(0, 1), (2, 0)]));
        assert_ne!(mesh_permutation, unmeshed);

        // Map lookups find independently constructed equal operations and miss distinct ones.
        let operations = HashMap::from([(ordinary, 0), (mesh_permutation, 1)]);
        assert_eq!(operations.get(&same_ordinary), Some(&0));
        assert_eq!(operations.get(&same_mesh_permutation), Some(&1));
        assert_eq!(operations.get(&unmeshed), None);
    }

    #[test]
    fn test_parallel_permute_type_inference() {
        // A permutation preserves its input type, including its layout and memory, and an empty permutation, which
        // zeros every participant, is valid. An ordinary permutation also accepts an input that is invariant over a
        // manual mesh axis with the same name, because a `batch` level that shadows that mesh axis may bind it.
        let mesh = manual_mesh();
        let sharding = Sharding::replicated(mesh.clone(), 1);
        let vector = ArrayType::new_static(DataType::F32, [3]);
        let invariant = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap();
        let varying = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(sharding.clone().with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let placed = varying
            .clone()
            .with_layout(Layout::Strided(StridedLayout::new(vec![4])))
            .with_memory(Memory::Host { pinned: true });
        check_operation_type_inference!(
            operation = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1), (1, 0)]),
            cases = [
                { input_types = [vector.clone()], output_types = [vector.clone()] },
                { input_types = [placed.clone()], output_types = [placed.clone()] },
                { input_types = [invariant.clone()], output_types = [invariant.clone()] },
            ],
        );
        check_operation_type_inference!(
            operation = ParallelPermuteOperation::new("x".to_string(), 2, Vec::new()),
            cases = [{ input_types = [vector.clone()], output_types = [vector.clone()] }],
        );

        // Bounded and unbounded dynamic dimensions are preserved for ordinary and manual-mesh permutations.
        let bounded_length = DimensionVariable::new("length", DimensionBounds::new(0, Some(8)).unwrap());
        let bounded = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(bounded_length)]));
        let unbounded_length = DimensionVariable::new("length", DimensionBounds::unbounded());
        let unbounded = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(unbounded_length)]));
        check_operation_type_inference!(
            operation = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1)]),
            cases = [
                { input_types = [bounded.clone()], output_types = [bounded.clone()] },
                { input_types = [unbounded.clone()], output_types = [unbounded.clone()] },
            ],
        );
        let varying_sharding = sharding.clone().with_varying_manual_axes(["x"]).unwrap();
        let bounded_varying = bounded.with_sharding(varying_sharding.clone()).unwrap();
        let unbounded_varying = unbounded.with_sharding(varying_sharding).unwrap();
        check_operation_type_inference!(
            operation = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1)]).with_mesh(mesh.clone()),
            cases = [
                { input_types = [bounded_varying.clone()], output_types = [bounded_varying] },
                { input_types = [unbounded_varying.clone()], output_types = [unbounded_varying] },
            ],
        );

        // Every pair must reference participants of the axis, and no two pairs may share a source or a target.
        check_operation_type_inference!(
            operation = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 2)]),
            cases = [{
                input_types = [vector.clone()],
                error = "`parallel_permute` pair (0, 2) is out of bounds for axis size 2",
            }],
        );
        check_operation_type_inference!(
            operation = ParallelPermuteOperation::new("x".to_string(), 2, vec![(2, 0)]),
            cases = [{
                input_types = [vector.clone()],
                error = "`parallel_permute` pair (2, 0) is out of bounds for axis size 2",
            }],
        );
        check_operation_type_inference!(
            operation = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1), (0, 0)]),
            cases = [{
                input_types = [vector.clone()],
                error = "`parallel_permute` pairs must have unique sources and targets but (0, 0) repeats one",
            }],
        );
        check_operation_type_inference!(
            operation = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1), (1, 1)]),
            cases = [{
                input_types = [vector.clone()],
                error = "`parallel_permute` pairs must have unique sources and targets but (1, 1) repeats one",
            }],
        );

        // A permutation over a manual mesh axis preserves a varying input type, but an input that is invariant over
        // the axis would wrongly type the permuted output as invariant, so the input must vary over the axis of the
        // operation's mesh. Routing cannot complete a pending sum over its own axis, so it rejects one, while an
        // ordinary permutation preserves every pending sum.
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let other_varying = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(other_mesh, 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let unreduced = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(sharding.with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        check_operation_type_inference!(
            operation = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1)]),
            cases = [{ input_types = [unreduced.clone()], output_types = [unreduced.clone()] }],
        );
        check_operation_type_inference!(
            operation = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1)]).with_mesh(mesh.clone()),
            cases = [
                { input_types = [varying.clone()], output_types = [varying.clone()] },
                {
                    input_types = [invariant],
                    error = "`parallel_permute` input must vary over manual axis `x`; pass an invariant value through \
                             `parallel_vary` first so that the permuted output is typed as varying",
                },
                {
                    input_types = [vector],
                    error = "`parallel_permute` input must carry a mesh containing manual axis `x`",
                },
                {
                    input_types = [other_varying],
                    error = "`parallel_permute` input mesh does not match the operation mesh",
                },
                { input_types = [unreduced], error = "`parallel_permute` does not support unreduced inputs" },
            ],
        );

        // The mesh axis must be manual, and its size must equal the axis size of the operation.
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        check_operation_type_inference!(
            operation = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1)]).with_mesh(explicit_mesh),
            cases = [{ input_types = [varying.clone()], error = "`parallel_permute` mesh axis `x` must be manual" }],
        );
        check_operation_type_inference!(
            operation = ParallelPermuteOperation::new("x".to_string(), 3, vec![(0, 1)]).with_mesh(mesh),
            cases = [{
                input_types = [varying],
                error = "`parallel_permute` axis size 3 does not match the size of manual mesh axis `x`",
            }],
        );
    }

    #[test]
    fn test_parallel_permute_type_inference_preserves_unrelated_pending_sums() {
        // Routing along `x` commutes with a pending sum along the unrelated manual axis `y`, so a permutation over `x`
        // preserves it, unlike a pending sum along `x` itself, which `test_parallel_permute_type_inference` rejects.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let pending_y = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(
                Sharding::replicated(mesh.clone(), 1)
                    .with_unreduced_axes(["y"])
                    .unwrap()
                    .with_varying_manual_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        check_operation_type_inference!(
            operation = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1), (1, 0)]).with_mesh(mesh),
            cases = [{ input_types = [pending_y.clone()], output_types = [pending_y] }],
        );
    }

    #[test]
    fn test_parallel_permute_interpretation() {
        // Outside any binder, the single participant of a degenerate axis keeps its value only when it sends to itself,
        // and receives zeros otherwise, while a larger axis has no per-item semantics.
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
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

        // Even a degenerate collective validates its mapping before interpreting it as identity or zero.
        assert_eq!(
            context.bind(
                ParallelPermuteOperation::new("x".to_string(), 1, vec![(1, 0)]),
                Vec::new(),
                std::slice::from_ref(&input),
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "`parallel_permute` pair (1, 0) is out of bounds for axis size 1",
            ))),
        );
        assert_eq!(
            context.bind(ParallelPermuteOperation::new("x".to_string(), 1, vec![(0, 0), (0, 0)]), Vec::new(), &[input]),
            Err(ProgramError::Type(TypeError::invalid(
                "`parallel_permute` pairs must have unique sources and targets but (0, 0) repeats one",
            ))),
        );
    }

    #[test]
    fn test_parallel_permute_partial_evaluation() {
        // A known input over a degenerate axis folds through interpretation, here into zeros for the untargeted
        // participant, and an unknown input residualizes the operation.
        check_operation_partial_evaluation!(
            operation = ParallelPermuteOperation::new("x".to_string(), 1, Vec::new()),
            inputs = [Array::vector(vec![1.0f32, 2.0]).unwrap()],
            expected = Array::vector(vec![0.0f32, 0.0]).unwrap(),
        );

        // A known input over a larger axis under an eager parent residualizes the operation, which has no per-item
        // value, so the residual program is the source program itself.
        let operation = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1)]);
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let program = collective_program(operation.clone(), ArrayType::new_static(DataType::F32, [2]));
        let evaluation = program.partially_evaluate(&[PartialValue::Known(input)]).unwrap();
        assert_eq!(evaluation.program().to_string(), program.to_string());
        assert_eq!(evaluation.outputs(), &[PartialEvaluationOutput::Unknown(0)]);

        // A known input under a staging parent stays known, because the operation is staged into the parent trace.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = PartialEvaluationValue::known(trace.input(ArrayType::new_static(DataType::F32, [2])));
        let outputs = operation
            .partially_evaluate(&PartialEvaluationContext::new(trace.clone()), &EmptyRegionDriver, &[input])
            .unwrap();
        assert!(outputs[0].is_known());
        assert_eq!(outputs[0].r#type().as_ref(), &ArrayType::new_static(DataType::F32, [2]));
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(
                vec![outputs[0].as_known().unwrap().atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = parallel_permute [axis_name=\"x\", axis_size=2, source_target_pairs=[(0, 1)]] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_parallel_permute_batching() {
        // A level that binds the permuted axis reassembles its mapped axis in target order, wherever that axis sits,
        // and zeros every item that no pair targets.
        let swap = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1), (1, 0)]);
        let shift = ParallelPermuteOperation::new("x".to_string(), 3, vec![(0, 1), (1, 2)]);
        assert_eq!(
            batch_collective(
                &swap,
                "x",
                2,
                ArrayBatch::new(Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap(), BatchAxis::new(0)).unwrap(),
            ),
            Ok(vec![
                ArrayBatch::new(Array::matrix(2, 2, vec![3.0, 4.0, 1.0, 2.0]).unwrap(), BatchAxis::new(0)).unwrap()
            ]),
        );
        assert_eq!(
            batch_collective(
                &swap,
                "x",
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
            batch_collective(
                &shift,
                "x",
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
        assert_eq!(batch_collective(&swap, "x", 2, input.clone()), Ok(vec![input.clone()]));
        assert_eq!(
            batch_collective(&shift, "x", 3, input),
            Ok(vec![
                ArrayBatch::new(Array::matrix(3, 2, vec![0.0, 0.0, 1.0, 2.0, 1.0, 2.0]).unwrap(), BatchAxis::new(0))
                    .unwrap(),
            ]),
        );

        // The operation's axis size must equal the size of the level that binds its axis.
        assert_eq!(
            batch_collective(
                &swap,
                "x",
                3,
                ArrayBatch::new(Array::vector(vec![1.0, 2.0, 3.0]).unwrap(), BatchAxis::new(0)).unwrap(),
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_permute` over axis `x` resolved axis size 2 but the mapped batch axis has size 3"
                    .to_string(),
            }),
        );

        // A permutation over a manual mesh axis cannot be consumed by a level that binds a batch axis with its name.
        assert_eq!(
            batch_collective(
                &swap.clone().with_mesh(manual_mesh()),
                "x",
                2,
                ArrayBatch::new(Array::vector(vec![1.0, 2.0]).unwrap(), BatchAxis::new(0)).unwrap(),
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_permute` over a manual mesh axis cannot bind a named batch axis".to_string(),
            }),
        );

        // Routing slices every batch item by its static extents, so a level that binds the axis rejects an input whose
        // items have a dynamic dimension.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(length)]));
        let input = ArrayBatch::new(trace.input(input_type), BatchAxis::new(0)).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace, 2).with_axis_name("x".to_string());
        assert!(matches!(
            swap.batch(&context, &EmptyRegionDriver, &[input]),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "`parallel_permute` batching requires statically shaped inputs and ragged extents",
        ));

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
                let %1:f32[2, 3] = \
                    parallel_permute [axis_name=\"x\", axis_size=2, source_target_pairs=[(0, 1), (1, 0)]] %0
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
            batch_collective(&swap, "x", 2, input.clone()),
            Ok(vec![ragged(vec![2.0, 3.0, 4.0, 1.0, 0.0, 0.0], Array::vector(vec![3i32, 1]).unwrap(), vec![0])]),
        );
        assert_eq!(
            batch_collective(&send, "x", 2, input),
            Ok(vec![ragged(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0], Array::vector(vec![0i32, 1]).unwrap(), vec![0])]),
        );

        // Extents that are the same for every item stay replicated under a full permutation, but must vary over the
        // batch axis before a partial permutation can zero the extent of an untargeted item.
        let input = ragged(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], Array::scalar(1i32).unwrap(), Vec::new());
        assert_eq!(
            batch_collective(&swap, "x", 2, input.clone()),
            Ok(vec![ragged(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], Array::scalar(1i32).unwrap(), Vec::new())]),
        );
        assert_eq!(
            batch_collective(&send, "x", 2, input),
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
                    .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![0i32, 1]).unwrap(), length, vec![0])])
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
                    positive_length.clone(),
                    vec![0],
                )])
                .unwrap();
        assert_eq!(
            batch_collective(&swap, "x", 2, input.clone()),
            Ok(vec![
                ArrayBatch::new(Array::matrix(2, 3, vec![2.0, 3.0, 4.0, 1.0, 0.0, 0.0]).unwrap(), BatchAxis::new(0))
                    .unwrap()
                    .with_ragged_axes(vec![RaggedAxis::new(
                        1,
                        Array::vector(vec![3i32, 1]).unwrap(),
                        positive_length,
                        vec![0],
                    )])
                    .unwrap()
            ]),
        );
        assert_eq!(
            batch_collective(&send, "x", 2, input.clone()),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_permute` cannot assign a zero extent to bounded ragged dimension `length` whose \
                          lower bound is 1"
                    .to_string(),
            }),
        );
        let context = eager_collective_context("y", 2);
        assert_eq!(
            swap.batch(&context, &EmptyRegionDriver, &[input]).map(|outputs| outputs.into_parts().0),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_permute` does not support bounded ragged dimension `length` on input 0".to_string(),
            }),
        );
    }

    #[test]
    fn test_parallel_permute_batching_sharding() {
        // Route explicitly sharded participants using replicated singleton slices, restoring their original placement
        // on the complete output. Ragged extents with the same placement must follow that routing too.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["devices"]), ShardingDimension::Replicated])
                .unwrap();
        let extent_sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["devices"])]).unwrap();
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let ragged = |values: Vec<f64>, extents: Vec<i32>| {
            ArrayBatch::new(Array::matrix(2, 3, values).unwrap().reshard(&sharding).unwrap(), BatchAxis::new(0))
                .unwrap()
                .with_ragged_axes(vec![RaggedAxis::new(
                    1,
                    Array::vector(extents).unwrap().reshard(&extent_sharding).unwrap(),
                    length.clone(),
                    vec![0],
                )])
                .unwrap()
        };
        let input = ragged(vec![1.0, 0.0, 0.0, 2.0, 3.0, 4.0], vec![1, 3]);
        let swap = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1), (1, 0)]);
        let send = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1)]);
        assert_eq!(
            batch_collective(&swap, "x", 2, input.clone()),
            Ok(vec![ragged(vec![2.0, 3.0, 4.0, 1.0, 0.0, 0.0], vec![3, 1])]),
        );
        assert_eq!(
            batch_collective(&send, "x", 2, input),
            Ok(vec![ragged(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![0, 1])]),
        );
    }

    #[test]
    fn test_parallel_permute_batching_shadows_manual_axis() {
        // The inner named batch binds `x`, so an invariant value on an enclosing manual mesh axis with the same name
        // needs no mesh-axis variation to permute its local batch items.
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(manual_mesh(), 1))
            .unwrap();
        let (output_type, program) = trace_permutation_shadowing_manual_axis(input_type.clone(), vec![(0, 1), (1, 0)]);
        assert_eq!(output_type, input_type);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}]}] .
                let %1:f32[1][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}]}] = \
                        slice [start_indices=[1], limits=[2]] %0
                    %2:f32[1][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}]}] = \
                        slice [start_indices=[0], limits=[1]] %0
                    %3:f32[2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}]}] = concatenate [axis=0] %1 %2
                in (%3)"
            },
        );

        // The local permutation also preserves a pending cross-device sum over the shadowed mesh axis,
        // including for the zero-filled item at an untargeted position.
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(manual_mesh(), 1).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let (output_type, program) = trace_permutation_shadowing_manual_axis(input_type.clone(), vec![(0, 1)]);
        assert_eq!(output_type, input_type);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], unreduced={'x'}}] .
                let %1:f32[1][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], unreduced={'x'}}] = \
                        slice [start_indices=[0], limits=[1]] %0
                    %2:f32[1][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], unreduced={'x'}}] = zero_like %1
                    %3:f32[1][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], unreduced={'x'}}] = \
                        slice [start_indices=[0], limits=[1]] %0
                    %4:f32[2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], unreduced={'x'}}] = \
                        concatenate [axis=0] %2 %3
                in (%4)"
            },
        );

        // A value that varies over the unrelated manual axis `y` keeps that variation on every routed item, including
        // the zero-filled item at an untargeted position, so that their concatenation needs no variation alignment.
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(manual_mesh(), 1).with_varying_manual_axes(["y"]).unwrap())
            .unwrap();
        let (output_type, program) = trace_permutation_shadowing_manual_axis(input_type.clone(), vec![(0, 1)]);
        assert_eq!(output_type, input_type);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'y'}}] .
                let %1:f32[1][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'y'}}] = \
                        slice [start_indices=[0], limits=[1]] %0
                    %2:f32[1][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'y'}}] = zero_like %1
                    %3:f32[1][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'y'}}] = \
                        slice [start_indices=[0], limits=[1]] %0
                    %4:f32[2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'y'}}] = \
                        concatenate [axis=0] %2 %3
                in (%4)"
            },
        );
    }

    #[test]
    fn test_parallel_permute_differentiation() {
        // The collective is linear, so the tangent rides the same permutation as the primal.
        let operation = ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1)]);
        assert_eq!(
            collective_program(operation.clone(), ArrayType::new_static(DataType::F32, [2]))
                .jvp()
                .unwrap()
                .to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[2] .
                let %2:f32[2] = parallel_permute [axis_name=\"x\", axis_size=2, source_target_pairs=[(0, 1)]] %0
                    %3:f32[2] = parallel_permute [axis_name=\"x\", axis_size=2, source_target_pairs=[(0, 1)]] %1
                in (%2, %3)"
            },
        );

        // A structural zero tangent stays a structural zero, and only the primal permutation is staged.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = context.input(ArrayType::new_static(DataType::F32, [2]));
        let outputs = operation
            .jvp(
                &DifferentiationContext::fused(context.clone()),
                &EmptyRegionDriver,
                &[DifferentiationDual::new_with_zero_tangent(input).unwrap()],
            )
            .unwrap();
        assert!(outputs[0].tangent().is_zero());
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(
                vec![outputs[0].primal().atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = parallel_permute [axis_name=\"x\", axis_size=2, source_target_pairs=[(0, 1)]] %0
                in (%1)"
            },
        );

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

        // Unequal output weights distinguish the routing direction; finite differences also check that an input
        // which sends nowhere contributes no derivative.
        check_gradient!(
            |inputs| {
                let routed = batch(
                    |item| item.parallel_permute("x", vec![(0, 2), (2, 1)]),
                    inputs,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("x"),
                )?;
                let last = routed.slice(&[2], &[3], &[1])?;
                let last = last.clone() + last;
                let last = last.clone() + last;
                let middle = routed.slice(&[1], &[2], &[1])?;
                let weighted = last + middle.clone() + middle;
                weighted.reduce(&[0], ReductionKind::Sum)
            },
            at = Array::vector(vec![1.0, 2.0, 3.0]).unwrap(),
            step = 1e-6,
            tolerance = 1e-6,
        );
    }

    #[test]
    fn test_parallel_permute_transposition() {
        // Sending along `(source, target)` pulls cotangents back along `(target, source)`, so the transpose is the
        // permutation with every pair inverted, and transposing it again recovers the original permutation.
        let program = collective_program(
            ParallelPermuteOperation::new("x".to_string(), 3, vec![(0, 1), (1, 2)]),
            ArrayType::new_static(DataType::F32, [2]),
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

        // A permutation over a manual mesh axis transposes over the same mesh.
        let varying = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(manual_mesh(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let program = collective_program(
            ParallelPermuteOperation::new("x".to_string(), 2, vec![(0, 1)]).with_mesh(manual_mesh()),
            varying,
        );
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[]).unwrap().to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'x'}}] .
                let %1:f32[2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'x'}}] = \
                    parallel_permute [
                    axis_name=\"x\",
                    axis_size=2,
                    source_target_pairs=[(1, 0)],
                    mesh=['x'=2:manual, 'y'=1:manual],
                ] %0
                in (%1)"
            },
        );
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

        // A composite value permutes through its array view.
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
        )
        .with_axis_name("x".to_string());
        let input = ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let input = BatchingTracer::new(context, ArrayIrBatch::new(input, BatchAxis::new(0)).unwrap());
        assert_eq!(
            input.parallel_permute("x", vec![(0, 2), (2, 0)]).map(|output| output.into_batch().into_value()),
            Ok(ArrayIrValue::Array(Array::vector(vec![3.0f32, 0.0, 1.0]).unwrap())),
        );

        // Over a manual mesh axis, a varying value is permuted directly, while an invariant value, and a value without
        // a sharding, are first made varying, because the permuted participants generally hold different values.
        let mesh = manual_mesh();
        let axes = vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })];
        let sharding = Sharding::replicated(mesh, 0);
        let varying = ArrayType::scalar(DataType::F32)
            .with_sharding(sharding.clone().with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let invariant = ArrayType::scalar(DataType::F32).with_sharding(sharding.clone()).unwrap();
        let (output, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_permute("x", vec![(0, 1)]),
            varying.clone(),
            axes.clone(),
        )
        .unwrap();
        assert_eq!(output, varying);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [], varying_manual={'x'}}] .
                let %1:f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [], varying_manual={'x'}}] = \
                    parallel_permute [
                    axis_name=\"x\",
                    axis_size=2,
                    source_target_pairs=[(0, 1)],
                    mesh=['x'=2:manual, 'y'=1:manual],
                ] %0
                in (%1)"
            },
        );
        let (output, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_permute("x", vec![(0, 1)]),
            invariant,
            axes.clone(),
        )
        .unwrap();
        assert_eq!(output, varying);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, []}] .
                let %1:f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %0
                    %2:f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [], varying_manual={'x'}}] = \
                        parallel_permute [
                        axis_name=\"x\",
                        axis_size=2,
                        source_target_pairs=[(0, 1)],
                        mesh=['x'=2:manual, 'y'=1:manual],
                    ] %1
                in (%2)"
            },
        );
        let (output, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_permute("x", vec![(0, 1)]),
            ArrayType::scalar(DataType::F32),
            axes.clone(),
        )
        .unwrap();
        assert_eq!(output, varying);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, []}] = broadcast [
                    output_type=f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, []}],
                    output_axes=[],
                ] %0
                    %2:f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %1
                    %3:f32[][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [], varying_manual={'x'}}] = \
                        parallel_permute [
                        axis_name=\"x\",
                        axis_size=2,
                        source_target_pairs=[(0, 1)],
                        mesh=['x'=2:manual, 'y'=1:manual],
                    ] %2
                in (%3)"
            },
        );

        // Over a manual mesh axis, a value with a pending cross-device sum is rejected before it is made varying.
        assert_eq!(
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_permute("x", vec![(0, 1)]),
                ArrayType::scalar(DataType::F32)
                    .with_sharding(sharding.with_unreduced_axes(["x"]).unwrap())
                    .unwrap(),
                axes,
            )
            .map(|(output, _)| output),
            Err(ProgramError::Type(TypeError::invalid("`parallel_permute` does not support unreduced inputs"))),
        );

        // A concrete array is never inside an axis binder, so every axis name is unbound for it.
        assert_eq!(
            Array::vector(vec![1.0, 2.0]).unwrap().parallel_permute("i", vec![(0, 1)]),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "i".to_string() })),
        );
    }

    #[test]
    fn test_parallel_permute_parallel_permute_composite() {
        // Concrete composite values permute through their array members, which for concrete arrays outside a
        // named-axis environment reports the unbound axis.
        let vector = Array::vector(vec![1.0f32, 2.0]).unwrap();
        assert_eq!(
            ArrayIrValue::Array(vector.clone()).parallel_permute("x", vec![(0, 1)]),
            vector.parallel_permute("x", vec![(0, 1)]).map(ArrayIrValue::Array),
        );
    }

    #[test]
    fn test_parallel_permute_parallel_permute_preserves_unrelated_pending_sums() {
        // Over a manual mesh axis, an invariant value is made varying over the permuted axis while it keeps its pending
        // sum over an unrelated axis, and a composite value stages the same permutation through its array view.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let axes = vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })];
        let pending_y = Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["y"]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(pending_y.clone()).unwrap();
        let output_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(pending_y.with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let (output, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_permute("x", vec![(0, 1), (1, 0)]),
            input_type.clone(),
            axes.clone(),
        )
        .unwrap();
        assert_eq!(output, output_type);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], unreduced={'y'}}] .
                let %1:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], unreduced={'y'}, \
                        varying_manual={'x'}}] = parallel_vary [axis_name=\"x\"] %0
                    %2:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], unreduced={'y'}, \
                        varying_manual={'x'}}] = \
                        parallel_permute [
                        axis_name=\"x\",
                        axis_size=2,
                        source_target_pairs=[(0, 1), (1, 0)],
                        mesh=['x'=2:manual, 'y'=2:manual],
                    ] %1
                in (%2)"
            },
        );
        assert_eq!(
            TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_permute("x", vec![(0, 1), (1, 0)]),
                ArrayIrType::Array(input_type),
                axes.clone(),
            )
            .map(|(output, _)| output),
            Ok(ArrayIrType::Array(output_type)),
        );

        // A pending sum over the permuted axis is rejected for both families before the value is made varying.
        let pending_x = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh, 1).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        assert_eq!(
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_permute("x", vec![(0, 1), (1, 0)]),
                pending_x.clone(),
                axes.clone(),
            )
            .map(|(output, _)| output),
            Err(ProgramError::Type(TypeError::invalid("`parallel_permute` does not support unreduced inputs"))),
        );
        assert_eq!(
            TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_permute("x", vec![(0, 1), (1, 0)]),
                ArrayIrType::Array(pending_x),
                axes,
            )
            .map(|(output, _)| output),
            Err(ProgramError::Type(TypeError::invalid("`parallel_permute` does not support unreduced inputs"))),
        );
    }

    #[test]
    fn test_parallel_permute_parallel_shuffle() {
        // Under a `batch` level that binds the axis, output item `target` receives input item `permutation[target]`,
        // and an empty permutation gives every item zeros.
        let input = Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap();
        assert_eq!(
            batch(
                |item| item.parallel_shuffle("x", &[2, 0, 1]),
                input.clone(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::named("x"),
            ),
            Ok(Array::vector(vec![3.0f32, 1.0, 2.0]).unwrap()),
        );
        assert_eq!(
            batch(
                |item| item.parallel_shuffle("x", &[]),
                input,
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::named("x"),
            ),
            Ok(Array::vector(vec![0.0f32, 0.0, 0.0]).unwrap()),
        );

        // A composite value shuffles through its array view in the same way, and a permutation shorter than the axis
        // gives every remaining item zeros.
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
        )
        .with_axis_name("x".to_string());
        let input = ArrayIrValue::Array(Array::matrix(3, 2, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        let input = BatchingTracer::new(context, ArrayIrBatch::new(input, BatchAxis::new(0)).unwrap());
        assert_eq!(
            input.parallel_shuffle("x", &[2, 0, 1]).map(|output| output.into_batch().into_value()),
            Ok(ArrayIrValue::Array(Array::matrix(3, 2, vec![5.0f32, 6.0, 1.0, 2.0, 3.0, 4.0]).unwrap())),
        );
        assert_eq!(
            input.parallel_shuffle("x", &[1, 0]).map(|output| output.into_batch().into_value()),
            Ok(ArrayIrValue::Array(Array::matrix(3, 2, vec![3.0f32, 4.0, 1.0, 2.0, 0.0, 0.0]).unwrap())),
        );

        // The permutation must be a permutation of its own positions, and it cannot be longer than the axis.
        assert_eq!(
            input.parallel_shuffle("x", &[0, 2]).map(|output| output.into_batch().into_value()),
            Err(ProgramError::Type(TypeError::invalid(
                "`parallel_shuffle` source index 2 is out of bounds for a permutation of length 2",
            ))),
        );
        assert_eq!(
            input.parallel_shuffle("x", &[1, 1]).map(|output| output.into_batch().into_value()),
            Err(ProgramError::Type(TypeError::invalid(
                "`parallel_shuffle` permutation contains source index 1 more than once",
            ))),
        );
        // `parallel_shuffle` itself accepts this permutation, so the axis bound is enforced by the batching rule of
        // `parallel_permute`. That rule converts its `TypeError` into a `BatchingError::Type`, which reaches the
        // caller wrapped in a `ProgramError`, unlike the plain `ProgramError::Type` errors of `parallel_shuffle` above.
        assert_eq!(
            input.parallel_shuffle("x", &[3, 2, 1, 0]).map(|output| output.into_batch().into_value()),
            Err(BatchingError::Type(TypeError::invalid(
                "`parallel_permute` pair (3, 0) is out of bounds for axis size 3",
            ))
            .into()),
        );
    }
}
