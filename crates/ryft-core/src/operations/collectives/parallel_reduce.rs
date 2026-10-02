use std::fmt::Display;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayType, DataType, LogicalMesh, MeshAxisType,
    RaggedArrayExtentBatchingPolicy, RaggedAxis, RaggedMaskIdentity,
};
use crate::axes::{Axis, AxisError, NamedAxes, NamedAxis};
use crate::batching::{BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain};
use crate::differentiation::DifferentiationDual;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::arithmetic::DivOperation;
use crate::operations::collectives::parallel_vary::{ManualVariationAlignment, ParallelVary, ParallelVaryOperation};
use crate::operations::collectives::{effective_collective_axis_size, resolve_named_axis_size};
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::manipulation::conversions::ConvertElementType;
use crate::operations::manipulation::memory::TransferToMemory;
use crate::operations::reductions::{Reduce, ReductionKind};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{MaybeZero, Operation, OperationFormatter, ProgramError, RegionInterface, TypeError, Value};

// TODO(eaplatanios): Review from here onwards.

/// Combining operator of a [`ParallelReduceOperation`], which also determines the operation's name. Every kind combines
/// the values that the participants of a named axis hold into one result that each participant holds in full, and every
/// kind preserves the element data type: integer sums wrap and integer means truncate toward zero in the input's own
/// type, so a fractional mean requires converting the input to a floating-point type first.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ParallelReductionKind {
    /// Sum of the participants' values, which is the analogue of `jax.lax.psum`. Numeric inputs and the structural zero
    /// are supported. Unlike JAX, which converts Boolean inputs to `i32` before summing them, Boolean inputs are
    /// rejected and must be converted explicitly.
    Sum,

    /// Arithmetic mean of the participants' values (i.e., their sum divided by the participant count), which is the
    /// analogue of `jax.lax.pmean`. Numeric inputs and the structural zero are supported. Integer means truncate toward
    /// zero in the input's own type, whereas JAX's `pmean` promotes them to a floating-point type through its division.
    Mean,

    /// Maximum of the participants' values, which is the analogue of `jax.lax.pmax`. Boolean, numeric, and
    /// structural-zero inputs are supported, with the ordering and exceptional-value semantics of
    /// [`ReductionKind::Max`]. A maximum is not differentiable.
    Max,
}

impl ParallelReductionKind {
    /// Returns the name of this [`ParallelReductionKind`], which program renderings and diagnostics use as the `kind`
    /// attribute of a [`ParallelReduceOperation`] (e.g., `parallel_reduce [kind=sum, axis_name="i"]`).
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Mean => "mean",
            Self::Max => "max",
        }
    }

    /// Returns the [`ReductionKind`] that combines the participants' values when a `batch` level reduces its mapped
    /// axis, and whose scalar combiner a backend uses to reduce across devices.
    #[inline]
    pub fn reduction_kind(self) -> ReductionKind {
        match self {
            Self::Sum => ReductionKind::Sum,
            Self::Mean => ReductionKind::Mean,
            Self::Max => ReductionKind::Max,
        }
    }
}

impl Display for ParallelReductionKind {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.name())
    }
}

/// Name of [`ParallelReduceOperation`]. The operation's [`ParallelReductionKind`] is rendered as its `kind` attribute.
pub const PARALLEL_REDUCE_OPERATION_NAME: &str = "parallel_reduce";

/// [`Operation`] that reduces a value across the participants of a named axis using a [`ParallelReductionKind`],
/// leaving every participant with the full result. Refer to the documentation of [`ParallelReduce`] for more
/// information. One payload represents three forms of the operation:
///
///   - **Ordinary reductions**, created by [`new`](Self::new), name an axis that an enclosing `batch` level binds.
///     Their output type is their input type. The `batch` level whose axis name matches reduces its mapped axis, and
///     every other level forwards the reduction to its parent. A replicated input holds the same value for every batch
///     item, so its sum is that value multiplied by the axis size, while its mean and its maximum are the value itself.
///   - **Grouped reductions**, created by [`grouped`](Self::grouped), reduce independently within each group of an
///     equal-sized exact partition of the axis's participants and preserve the input's manual variation, because
///     distinct groups can produce distinct results. A `batch` level that binds their axis rejects them.
///   - **Mesh reductions**, created by [`with_mesh`](Self::with_mesh), reduce over a manual axis of that mesh inside a
///     manual region (e.g., the body of a `shard_map` operation in the XLA backend). Their input must vary over the
///     axis, and their output no longer does, because every device holds the full reduction. This is the analogue of
///     JAX's `psum_invariant` primitive, and of `pmax` inside a `shard_map`. A `batch` level that binds their axis name
///     rejects them, and every other level forwards them to its parent with their mapped and ragged axes intact.
///
/// No form has per-item semantics, because the other participants exist only inside an enclosing binder. Interpreting
/// the operation outside one is therefore an error, and partial evaluation preserves it for the binder or backend that
/// owns the axis.
///
/// A `batch` level that binds the axis of a sum or a maximum accepts bounded ragged inputs. Padding is replaced with
/// the reduction identity before the participants are combined, and each surviving ragged extent is the elementwise
/// maximum across the participants. Participants whose local extents exclude a position contribute the identity, so
/// with multiple ragged axes, a position inside the coordinatewise-maximum output bounds that no participant covers
/// holds the identity. For a sum, that is zero. For a maximum, it is the lowest value of the element type (e.g., zero
/// or `false` for unsigned integers and Booleans), so negative live values still win. A mean rejects ragged inputs,
/// because its denominator has no single implied meaning: the participant count, the present-value count, and the
/// logical-element count define different operations.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParallelReduceOperation {
    /// Refer to the documentation of [`axis_name`](Self::axis_name) for more information.
    axis_name: String,

    /// Refer to the documentation of [`kind`](Self::kind) for more information.
    kind: ParallelReductionKind,

    /// Refer to the documentation of [`axis_size`](Self::axis_size) for more information.
    axis_size: Option<usize>,

    /// Refer to the documentation of [`axis_index_groups`](Self::axis_index_groups) for more information.
    axis_index_groups: Option<Vec<Vec<usize>>>,

    /// Refer to the documentation of [`mesh`](Self::mesh) for more information.
    mesh: Option<LogicalMesh>,
}

impl ParallelReduceOperation {
    /// Creates a new ordinary [`ParallelReduceOperation`] over `axis_name` with the provided `kind`.
    #[inline]
    pub fn new(axis_name: String, kind: ParallelReductionKind) -> Self {
        Self { axis_name, kind, axis_size: None, axis_index_groups: None, mesh: None }
    }

    /// Creates a new grouped [`ParallelReduceOperation`] over `axis_name` with the provided `kind`, after validating
    /// that `axis_index_groups` is an equal-sized exact partition of `0..axis_size`.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if `axis_size` is zero or if `axis_index_groups` is empty, contains an empty group,
    /// contains groups of different sizes, or does not contain every participant in `0..axis_size` exactly once.
    pub fn grouped(
        axis_name: String,
        kind: ParallelReductionKind,
        axis_size: usize,
        axis_index_groups: Vec<Vec<usize>>,
    ) -> Result<Self, TypeError> {
        effective_collective_axis_size(PARALLEL_REDUCE_OPERATION_NAME, axis_size, Some(axis_index_groups.as_slice()))?;
        Ok(Self { axis_name, kind, axis_size: Some(axis_size), axis_index_groups: Some(axis_index_groups), mesh: None })
    }

    /// Returns this [`ParallelReduceOperation`] configured to reduce over a manual axis of `mesh`. The input must vary
    /// over [`axis_name`](Self::axis_name) on that mesh, and the output is invariant over it. Type inference validates
    /// these requirements. [`ParallelReduce::parallel_reduce`] supplies the mesh automatically from the enclosing
    /// manual region.
    #[inline]
    pub fn with_mesh(mut self, mesh: LogicalMesh) -> Self {
        self.mesh = Some(mesh);
        self
    }

    /// Returns the name of the axis across whose participants this [`ParallelReduceOperation`] reduces.
    #[inline]
    pub fn axis_name(&self) -> &str {
        &self.axis_name
    }

    /// Returns the [`ParallelReductionKind`] of this [`ParallelReduceOperation`].
    #[inline]
    pub fn kind(&self) -> ParallelReductionKind {
        self.kind
    }

    /// Returns the full size of the named axis of a grouped [`ParallelReduceOperation`], or [`None`] for an ungrouped
    /// one, whose axis size is supplied by the enclosing binder.
    #[inline]
    pub fn axis_size(&self) -> Option<usize> {
        self.axis_size
    }

    /// Returns the ordered participant groups of a grouped [`ParallelReduceOperation`], or [`None`] for an ungrouped
    /// one.
    #[inline]
    pub fn axis_index_groups(&self) -> Option<&[Vec<usize>]> {
        self.axis_index_groups.as_deref()
    }

    /// Returns the logical mesh whose manual axis this [`ParallelReduceOperation`] reduces over, or [`None`] for an
    /// ordinary or grouped reduction. A mesh reduction removes the axis from its input's varying manual axes, while
    /// ordinary and grouped reductions preserve the input's manual variation.
    #[inline]
    pub fn mesh(&self) -> Option<&LogicalMesh> {
        self.mesh.as_ref()
    }

    /// Returns the number of participants in each group of a grouped [`ParallelReduceOperation`], or [`None`] for an
    /// ungrouped one, whose participant count is supplied by the enclosing binder.
    #[inline]
    pub fn group_size(&self) -> Option<usize> {
        self.axis_index_groups.as_ref().and_then(|groups| groups.first()).map(Vec::len)
    }
}

impl Display for ParallelReduceOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ParallelReduceOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        PARALLEL_REDUCE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        let (name, kind) = (self.name(), self.kind);
        let input = &input_types[0];
        match (&self.axis_index_groups, self.axis_size) {
            (Some(axis_index_groups), Some(axis_size)) => {
                effective_collective_axis_size(name, axis_size, Some(axis_index_groups.as_slice()))?;
            }
            (None, None) => {}
            _ => {
                return Err(TypeError::invalid(format!(
                    "`{name}` must store both the full axis size and axis index groups, or neither",
                )));
            }
        }

        // The element data type is validated for every form, so that a reduction that a `batch` level later collapses
        // fails here, naming this operation, rather than inside the batched `reduce`.
        let data_type = input.data_type();
        let (requirement, supports_kind) = match kind {
            ParallelReductionKind::Sum | ParallelReductionKind::Mean => {
                ("numeric", data_type.is_numeric() || data_type == DataType::Zero)
            }
            ParallelReductionKind::Max => {
                ("Boolean or numeric", data_type.is_boolean() || data_type.is_numeric() || data_type == DataType::Zero)
            }
        };
        if !supports_kind {
            return Err(TypeError::invalid(format!(
                "`{name}` with kind `{kind}` requires {requirement} inputs but got `{data_type}`",
            )));
        }

        // Ordinary and grouped reductions preserve the input type: the named axis exists physically only inside an
        // enclosing binder, which consumes it. A mesh reduction instead removes the axis from the input's variation.
        let output = match &self.mesh {
            None => input.clone(),
            Some(mesh) => {
                let axis_name = &self.axis_name;
                if self.axis_index_groups.is_some() {
                    return Err(TypeError::invalid(format!(
                        "`{name}` over a manual mesh axis must not use axis index groups",
                    )));
                }
                if kind == ParallelReductionKind::Mean {
                    return Err(TypeError::invalid(format!(
                        "`{name}` with kind `{kind}` over a manual mesh axis must be staged as a `{name}` with kind \
                         `sum` divided by the axis size",
                    )));
                }
                if mesh.axis_type(axis_name) != Some(MeshAxisType::Manual) {
                    return Err(TypeError::invalid(format!("`{name}` mesh axis `{axis_name}` must be manual")));
                }
                let Some(sharding) = input.sharding() else {
                    return Err(TypeError::invalid(format!(
                        "`{name}` input must carry a mesh containing manual axis `{axis_name}`",
                    )));
                };
                if sharding.mesh() != mesh {
                    return Err(TypeError::invalid(format!("`{name}` input mesh does not match the operation mesh")));
                }
                if sharding.unreduced_axes().contains(axis_name) || sharding.reduced_axes().contains(axis_name) {
                    return Err(TypeError::invalid(format!(
                        "`{name}` axis `{axis_name}` must not carry reduction state",
                    )));
                }
                if !sharding.varying_manual_axes().contains(axis_name) {
                    return Err(TypeError::invalid(format!(
                        "`{name}` input must vary over manual axis `{axis_name}`; pass an invariant value through \
                         `parallel_vary` first so that every copy is counted",
                    )));
                }
                let mut axes = sharding.varying_manual_axes().clone();
                axes.remove(axis_name);
                let sharding = sharding
                    .clone()
                    .with_varying_manual_axes(axes)
                    .map_err(|error| TypeError::invalid(error.to_string()))?;
                input.clone().with_sharding(sharding).map_err(|error| TypeError::invalid(error.to_string()))?
            }
        };

        // A maximum, and a truncating integer mean, do not commute with the pending cross-device sum of an unreduced
        // input, whereas a sum and a floating-point mean do.
        if !input.unreduced_axes().is_empty() {
            if kind == ParallelReductionKind::Max {
                return Err(TypeError::invalid(format!(
                    "`{name}` with kind `{kind}` does not support unreduced inputs",
                )));
            }
            if kind == ParallelReductionKind::Mean && data_type.is_integer() {
                return Err(TypeError::invalid(format!(
                    "`{name}` with kind `{kind}` does not support unreduced integer inputs",
                )));
            }
        }

        Ok(vec![output])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("kind", self.kind)?;
            operation.field("axis_name", format_args!("{:?}", self.axis_name))?;
            if let Some(mesh) = &self.mesh {
                operation.field("mesh", mesh)?;
            }
            if let Some(axis_size) = self.axis_size {
                operation.field("axis_size", axis_size)?;
            }
            if let Some(axis_index_groups) = &self.axis_index_groups {
                operation.field("axis_index_groups", format_args!("{axis_index_groups:?}"))?;
            }
            Ok(())
        })
    }
}

impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for ParallelReduceOperation {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // The other participants of the named axis exist only inside an enclosing binder: a `batch` level consumes the
        // reduction through the batching rule, and a backend that owns a manual region lowers it to a cross-device
        // reduction. There is therefore no per-item value to produce, and partial evaluation residualizes the operation
        // through the deferral of unsupported operations in its default rule.
        check_count!("input", inputs, 1, ProgramError);
        let message = match &self.mesh {
            None => {
                format!("cannot interpret `{}` over axis `{}` without an enclosing binder", self.name(), self.axis_name)
            }
            Some(_) => format!(
                "`{}` over manual mesh axis `{}` requires an execution backend owning the mesh",
                self.name(),
                self.axis_name,
            ),
        };
        Err(ProgramError::UnsupportedOperation { message })
    }
}

impl<C: Context<Type = ArrayType, Operation: From<ParallelReduceOperation>>> PartiallyEvaluatableOperation<C>
    for ParallelReduceOperation
{
}

impl<
    C: Context<Type = ArrayType, Operation: From<ParallelReduceOperation>, Value: Reduce>,
    P: RaggedArrayExtentBatchingPolicy<C>,
> BatchableOperation<C, ArrayBatchingPolicy<P>> for ParallelReduceOperation
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        // This rule owns named-axis resolution. A level whose axis name matches consumes its mapped axis, while every
        // other level forwards the reduction untouched to its parent, where the next level repeats the resolution.
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        if self.mesh.is_some() {
            if context.axis_name() == Some(self.axis_name.as_str()) {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!("`{}` over a manual mesh axis cannot bind a named batch axis", self.name()),
                });
            }

            // A mesh reduction combines the local shards elementwise, so unrelated mapped and ragged axes pass through
            // untouched, exactly as they do through its adjoint `parallel_vary`.
            return Ok(context.forward_to_parent(C::Operation::from(self.clone()), inputs)?.into());
        }

        if context.axis_name() != Some(self.axis_name.as_str()) {
            ArrayBatch::reject_ragged_inputs(self, inputs)?;
            return Ok(context.forward_to_parent(C::Operation::from(self.clone()), inputs)?.into());
        }

        if self.axis_index_groups.is_some() {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{}` axis index groups are not supported when a batch transform binds the collective axis",
                    self.name(),
                ),
            });
        }

        if self.kind == ParallelReductionKind::Mean && !input.ragged_axes().is_empty() {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{}` with kind `{}` does not define a denominator for bounded ragged inputs",
                    self.name(),
                    self.kind,
                ),
            });
        }

        // A replicated input holds the same value for every batch item. A sum counts it once per item, so it is first
        // materialized across the mapped extent, while the mean and the maximum of identical values are that value.
        let input = match (self.kind, input.batch_axis_position()) {
            (ParallelReductionKind::Sum, None) => P::match_axis(context, input, Axis::from(0usize))?,
            (_, None) => return Ok(vec![input.clone()].into()),
            (_, Some(_)) => input.clone(),
        };
        let batch_axis = input.batch_axis_position().unwrap();

        // Padding along every ragged axis is replaced with the reduction identity, so that it cannot contribute to the
        // result. Each surviving ragged extent is the maximum of the participants' extents.
        let masked_axes = input.ragged_axes().iter().map(RaggedAxis::axis).collect::<Vec<_>>();
        let input = match self.kind {
            ParallelReductionKind::Sum => {
                P::mask_identity_input(context, &input, masked_axes.as_slice(), RaggedMaskIdentity::Zero)?
            }
            ParallelReductionKind::Max => {
                P::mask_identity_input(context, &input, masked_axes.as_slice(), RaggedMaskIdentity::Lowest)?
            }
            ParallelReductionKind::Mean => input,
        };
        let output = input.value().reduce(&[batch_axis], self.kind.reduction_kind())?;
        let ragged_axes = input
            .ragged_axes()
            .iter()
            .map(|ragged_axis| {
                let extents = match ragged_axis.extent_axes().iter().position(|axis| *axis == batch_axis) {
                    None => ragged_axis.extents().clone(),
                    Some(extent_batch_axis) => {
                        ragged_axis.extents().reduce(&[extent_batch_axis], ReductionKind::Max)?
                    }
                };
                Ok(RaggedAxis::new(
                    ragged_axis.axis() - usize::from(batch_axis < ragged_axis.axis()),
                    extents,
                    ragged_axis.dimension().clone(),
                    ragged_axis
                        .extent_axes()
                        .iter()
                        .filter_map(|axis| (*axis != batch_axis).then_some(*axis - usize::from(batch_axis < *axis)))
                        .collect(),
                ))
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        Ok(vec![ArrayBatch::new(output, BatchAxis::replicated())?.with_ragged_axes(ragged_axes)?].into())
    }
}

impl_differentiable_operation! {
    ParallelReduceOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Operation: From<ParallelReduceOperation>,
    {
        |operation, context, _driver, inputs| {
            // Sums and means are linear, so the tangent is the same reduction of the input tangent. A structural zero
            // tangent stays a structural zero, whose type follows type inference because a mesh reduction changes the
            // manual variation. A maximum is nonlinear and, as in JAX, has no differentiation rule.
            check_count!("input", inputs, 1, ProgramError);
            if operation.kind == ParallelReductionKind::Max {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!("`{}` with kind `{}` is not differentiable", operation.name(), operation.kind),
                }
                .into());
            }
            let primal = stage_parallel_reduce(context.primal(), operation, inputs[0].primal())?;
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(r#type) => {
                    MaybeZero::Zero(operation.infer_output_types(std::slice::from_ref(r#type), &[])?.remove(0))
                }
                MaybeZero::Value(tangent) => {
                    MaybeZero::Value(stage_parallel_reduce(context.tangent(), operation, tangent)?)
                }
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType> + From<ParallelReduceOperation> + From<ParallelVaryOperation>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // An ordinary or grouped sum is self-adjoint: summing the cotangents of a shared total hands each
            // participant the total cotangent, which is the same reduction of the output cotangent. A mean is the sum
            // scaled by a constant, so it is self-adjoint as well. A mesh sum is the linear map `(x_1, …, x_n) ↦ Σ x_i`
            // whose adjoint hands the shared cotangent to every device, which is a `parallel_vary` over the same axis.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 1, DifferentiationError);
            if operation.kind == ParallelReductionKind::Max {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!("`{}` with kind `{}` is not transposable", operation.name(), operation.kind),
                }
                .into());
            }
            match &outputs[0] {
                MaybeZero::Value(_) if !accumulators[0].is_needed() => Ok(()),
                MaybeZero::Value(cotangent) if operation.mesh.is_some() => {
                    let mut contributions = context.bind(
                        ParallelVaryOperation::new(operation.axis_name.clone()),
                        Vec::new(),
                        std::slice::from_ref(cotangent),
                    )?;
                    check_count!("output", contributions, 1, ProgramError);
                    accumulators[0].accumulate(context, MaybeZero::Value(contributions.remove(0)))?;
                    Ok(())
                }
                MaybeZero::Value(cotangent) => {
                    let contribution = stage_parallel_reduce(&**context, operation, cotangent)?;
                    accumulators[0].accumulate(context, MaybeZero::Value(contribution))?;
                    Ok(())
                }
                MaybeZero::Zero(_) => Ok(()),
            }
        }
    },
}

/// Represents the ability to reduce a value across the participants of a named axis, producing one result that every
/// batch item or device shard holds in full. The axis is resolved by name against the active [`NamedAxes`] environment
/// at staging time, so an unbound name fails fast rather than silently acting as identity. A name bound by an enclosing
/// `batch` level stages an ordinary [`ParallelReduceOperation`], which that level collapses by reducing its mapped
/// batch axis. A value that is the same for every batch item is counted once per item by a sum, so summing a constant
/// `c` over an axis of size `n` yields `n · c`, exactly as `jax.lax.psum` does. A name bound to a device mesh axis by a
/// manual region (e.g., the body of a `shard_map` operation in the XLA backend) stays in the staged body program and
/// lowers to a cross-device `all_reduce` over that mesh axis. This is the analogue of JAX's `jax.lax.psum`,
/// `jax.lax.pmean`, and `jax.lax.pmax`. Refer to [`ParallelReductionKind`] for the supported data types of each kind
/// and for how integer means differ from JAX.
///
/// Inside a manual region, every value is a per-device local shard and its type records over which manual axes the
/// shards may differ. A value that varies over the axis holds a different shard on every device along it, and reducing
/// it stages a [`ParallelReduceOperation`] carrying the region's mesh, which combines those shards over the whole axis
/// and hands every device the same result, so the output is _invariant_ over the axis. This is what JAX stages as its
/// `psum_invariant` primitive. A value that is invariant over the axis holds the same shard on every device. Reducing
/// it counts that shard once per device, which is why the operation only accepts varying inputs: an invariant value is
/// first passed through [`parallel_vary`](ParallelVary::parallel_vary), which makes the multiplication by the device
/// count an explicit, differentiable step rather than an accident. [`parallel_reduce`](Self::parallel_reduce) inserts
/// that step itself, so summing an invariant constant `c` over an axis of size `n` yields `n · c` here as well. The sum
/// and the variation transition are adjoints of each other, which is what makes gradients through a manual region
/// correct without any bookkeeping at the region boundary: the transpose of the mesh-form sum is a
/// [`ParallelVaryOperation`], because the cotangent of a shared total is the same value handed to every device, and the
/// transpose of a [`ParallelVaryOperation`] is the mesh-form sum. A mean over a manual axis is the mesh-form sum
/// divided by the axis size, as `jax.lax.pmean` is `psum(x) / n`, and a maximum is not differentiable.
///
/// Participant subgroups retain the ordinary collective contract: the staged operation carries no mesh and preserves
/// the input variation, because distinct groups can produce distinct results.
///
/// # Example
///
/// Each device squares its own weight, and the sum of the squares is shared by all of them. Transposing the program
/// turns the sum back into a variation, so the gradient of the shared total reaches every device's weight:
///
/// ```rust
/// # use indoc::indoc;
/// # use ryft_core::{
/// #     Array, ArrayOperation, ArrayType, DataType, LogicalMesh, MeshAxis, MeshAxisType, NamedAxis, Operation,
/// #     ParallelReduce, ParallelReductionKind, Sharding, TracingContext,
/// # };
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual)?])?;
/// let invariant = ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(mesh.clone(), 0))?;
/// let varying = ArrayType::scalar(DataType::F32)
///     .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["x"])?)?;
/// let axes = vec![("x".to_string(), NamedAxis::Mesh { mesh, axis: 0, size: 2 })];
/// let (output, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
///     |weight| weight.parallel_reduce("x", ParallelReductionKind::Sum),
///     varying.clone(),
///     axes,
/// )?;
/// assert_eq!(output, invariant.clone());
/// assert_eq!(
///     program.to_string(),
///     indoc! {"
///         lambda %0:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] .
///         let %1:f32[][sharding={mesh<['x'=2:manual]>, []}] = \
///             parallel_reduce [kind=sum, axis_name=\"x\", mesh=['x'=2:manual]] %0
///         in (%1)"},
/// );
/// let transposed = program.transpose_with_respect_to(&[0], &[])?;
/// assert_eq!(transposed.input_types(), vec![invariant]);
/// assert_eq!(transposed.output_types(), vec![varying]);
/// let operations = transposed.instructions().iter().map(|instruction| instruction.operation().name());
/// assert_eq!(operations.collect::<Vec<_>>(), ["parallel_vary"]);
/// # Ok(())
/// # }
/// ```
pub trait ParallelReduce: Sized {
    /// Returns the reduction of this value across the participants of the named axis `axis_name`, staging a
    /// [`ParallelReduceOperation`] of the provided `kind`. Over a manual mesh axis, an input that is invariant over the
    /// axis is first passed through [`parallel_vary`](ParallelVary::parallel_vary), the staged operation records the
    /// axis's mesh, and a [`Mean`](ParallelReductionKind::Mean) is staged as the sum divided by the axis size.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of an axis bound by an enclosing `batch` level or manual region.
    ///   - `kind`: [`ParallelReductionKind`] that determines how the participants' values are combined.
    ///
    /// # Errors
    ///
    /// Returns [`AxisError::UnboundAxisName`] (surfaced as [`BatchingError::Axis`] riding a [`ProgramError::Custom`]
    /// payload) when no enclosing binder binds `axis_name`, and a [`ProgramError`] if `kind` does not support the
    /// element data type of this value, if this value carries reduction state along a manual `axis_name`, or if its
    /// sharding is on a different mesh than the region's.
    fn parallel_reduce(&self, axis_name: &str, kind: ParallelReductionKind) -> Result<Self, ProgramError>;

    /// Returns the reduction of this value within each participant group of the named axis `axis_name`, staging a
    /// grouped [`ParallelReduceOperation`] after validating that the groups cover the axis exactly once. Grouped
    /// reductions preserve manual variation, because distinct groups can produce distinct results.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of an axis bound by an enclosing `batch` level or manual region.
    ///   - `kind`: [`ParallelReductionKind`] that determines how the participants' values are combined.
    ///   - `axis_index_groups`: Equal-sized exact partition of `0..axis_size` into participant groups.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if `axis_name` is not bound by an enclosing binder, if `axis_index_groups` is not an
    /// equal-sized exact partition of the axis, or if `kind` does not support the element data type of this value.
    fn parallel_reduce_with_axis_index_groups(
        &self,
        axis_name: &str,
        kind: ParallelReductionKind,
        axis_index_groups: Vec<Vec<usize>>,
    ) -> Result<Self, ProgramError>;
}

// Any context-carrying value reduces by validating the axis name against the active `NamedAxes` environment and binding
// a `ParallelReduceOperation` through its own context: a staged tracer records the operation, a batching tracer
// resolves the named axis against the batching context stack, and a JVP dual forwards to the primal-side resolution.
impl<V: Value<Type = ArrayType> + ManualVariationAlignment<ArrayType> + ParallelVary> ParallelReduce for V
where
    V::DispatchDomain: Context + NamedAxes,
    <V::DispatchDomain as Domain>::Operation:
        From<ParallelReduceOperation> + From<ConstantOperation<Array>> + From<DivOperation<ArrayType>>,
{
    fn parallel_reduce(&self, axis_name: &str, kind: ParallelReductionKind) -> Result<Self, ProgramError> {
        let context = self.dispatch_domain();
        let named_axis = context
            .named_axis(axis_name)
            .ok_or_else(|| AxisError::UnboundAxisName { name: axis_name.to_string() })?;
        let NamedAxis::Mesh { mesh, size, .. } = named_axis else {
            let operation = ParallelReduceOperation::new(axis_name.to_string(), kind);
            let mut outputs = context.bind(operation, Vec::new(), std::slice::from_ref(self))?;
            check_count!("output", outputs, 1, ProgramError);
            return Ok(outputs.remove(0));
        };

        // Reducing an invariant value counts its shard once per device, so the value is first made varying, which
        // records that multiplication as an explicit, differentiable step.
        let mut input = self.clone();
        if !input.r#type().sharding().is_some_and(|sharding| sharding.varying_manual_axes().contains(axis_name)) {
            input = input.parallel_vary(axis_name)?;
        }

        // A mean over a manual mesh axis is the mesh-form sum divided by the axis size, exactly as JAX's `pmean` is
        // `psum(x) / n` at the wrapper level with no `pmean` primitive. The division happens in the element data type.
        let operation_kind = if kind == ParallelReductionKind::Mean { ParallelReductionKind::Sum } else { kind };
        let operation = ParallelReduceOperation::new(axis_name.to_string(), operation_kind).with_mesh(mesh);
        let mut outputs = context.bind(operation, Vec::new(), &[input])?;
        check_count!("output", outputs, 1, ProgramError);
        let output = outputs.remove(0);
        if kind != ParallelReductionKind::Mean {
            return Ok(output);
        }
        let output_type = output.r#type().into_owned();
        let divisor = Array::scalar(size as f64)?
            .convert_element_type(output_type.data_type())?
            .transfer_to_memory(output_type.memory())?;
        let mut divisors = context.bind(ConstantOperation::new(divisor), Vec::new(), &[])?;
        check_count!("output", divisors, 1, ProgramError);
        let inputs = ManualVariationAlignment::align_manual_variation(&[output, divisors.remove(0)])?;
        let mut quotients = context.bind(DivOperation::<ArrayType>::new(), Vec::new(), &inputs)?;
        check_count!("output", quotients, 1, ProgramError);
        Ok(quotients.remove(0))
    }

    fn parallel_reduce_with_axis_index_groups(
        &self,
        axis_name: &str,
        kind: ParallelReductionKind,
        axis_index_groups: Vec<Vec<usize>>,
    ) -> Result<Self, ProgramError> {
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        let operation = ParallelReduceOperation::grouped(axis_name.to_string(), kind, axis_size, axis_index_groups)?;
        let mut outputs = context.bind(operation, Vec::new(), std::slice::from_ref(self))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Binds `operation` to `input` in `context` and returns its single output. The forward-mode and transposition rules
/// use this to repeat a [`ParallelReduceOperation`] on a primal, a tangent, or an output cotangent.
fn stage_parallel_reduce<C: Context<Type = ArrayType, Operation: From<ParallelReduceOperation>>>(
    context: &C,
    operation: &ParallelReduceOperation,
    input: &C::Value,
) -> Result<C::Value, ProgramError> {
    let mut outputs = context.bind(operation.clone(), Vec::new(), std::slice::from_ref(input))?;
    check_count!("output", outputs, 1, ProgramError);
    Ok(outputs.remove(0))
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::batching::DynamicArrayExtentBatchingPolicy;
    use crate::arrays::{
        ArrayIrOperation, ArrayIrValue, ArrayOperation, Dimension, DimensionBounds, DimensionType, DimensionValue,
        DimensionVariable, MeshAxis, Shape, Sharding,
    };
    use crate::batching::{BatchAxisSpecification, BatchingTracer, batch};
    use crate::contexts::{EagerContext, ProjectedContext, StagingContext};
    use crate::differentiation::{
        DifferentiableOperation, DifferentiationContext, DifferentiationError, TransposableOperation,
        TranspositionContext, differentiate_at,
    };
    use crate::macros::check_operation_type_inference;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationValue, PartialValue};
    use crate::programs::{EmptyRegionDriver, Program, ProgramBuilder, Typed, ValueProjection};
    use crate::tracing::TracingContext;

    use super::*;

    /// Creates a two-axis manual mesh together with the invariant and varying (over `"m"`) types of one local scalar.
    fn mesh_scalar_types() -> (LogicalMesh, ArrayType, ArrayType) {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("m", 4, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("outer", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 0);
        let invariant = ArrayType::scalar(DataType::F32).with_sharding(sharding.clone()).unwrap();
        let varying = ArrayType::scalar(DataType::F32)
            .with_sharding(sharding.with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        (mesh, invariant, varying)
    }

    /// Binds `"m"` as a manual axis of `mesh` for [`TracingContext::trace_with_named_axes`].
    fn manual_mesh_axes(mesh: &LogicalMesh) -> Vec<(String, NamedAxis)> {
        vec![("m".to_string(), NamedAxis::Mesh { axis: 0, size: mesh.axis_size("m").unwrap(), mesh: mesh.clone() })]
    }

    /// Builds the single-instruction program that applies `operation` to one input of type `input_type`.
    fn parallel_reduce_program(
        operation: ParallelReduceOperation,
        input_type: ArrayType,
    ) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(input_type);
        let outputs = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap().to_vec();
        builder.build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Batches `operation` on `input` under an eager batching level of size `axis_size` that binds the axis `"i"`.
    fn batch_parallel_reduce(
        operation: &ParallelReduceOperation,
        axis_size: usize,
        input: ArrayBatch<Array>,
    ) -> Result<Vec<ArrayBatch<Array>>, BatchingError> {
        let context = BatchingContext::<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>::new(
            EagerContext::new(),
            axis_size,
        )
        .with_axis_name("i".to_string());
        Ok(operation.batch(&context, &EmptyRegionDriver, &[input])?.into_parts().0)
    }

    #[test]
    fn test_parallel_reduction_kind() {
        for (kind, name, reduction_kind) in [
            (ParallelReductionKind::Sum, "sum", ReductionKind::Sum),
            (ParallelReductionKind::Mean, "mean", ReductionKind::Mean),
            (ParallelReductionKind::Max, "max", ReductionKind::Max),
        ] {
            assert_eq!(kind.name(), name);
            assert_eq!(kind.to_string(), name);
            assert_eq!(kind.reduction_kind(), reduction_kind);
        }
    }

    #[test]
    fn test_parallel_reduce() {
        let operation = ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Sum);
        assert_eq!(operation.name(), PARALLEL_REDUCE_OPERATION_NAME);
        assert_eq!(operation.axis_name(), "i");
        assert_eq!(operation.kind(), ParallelReductionKind::Sum);
        assert_eq!(operation.axis_size(), None);
        assert_eq!(operation.axis_index_groups(), None);
        assert_eq!(operation.group_size(), None);
        assert_eq!(operation.mesh(), None);
        assert_eq!(operation.to_string(), "parallel_reduce [kind=sum, axis_name=\"i\"]");

        // Grouped reductions record the full axis size and their participant groups in order.
        let grouped = ParallelReduceOperation::grouped(
            "x".to_string(),
            ParallelReductionKind::Mean,
            4,
            vec![vec![0, 2], vec![3, 1]],
        )
        .unwrap();
        assert_eq!(grouped.name(), PARALLEL_REDUCE_OPERATION_NAME);
        assert_eq!(grouped.axis_size(), Some(4));
        assert_eq!(grouped.axis_index_groups(), Some([vec![0, 2], vec![3, 1]].as_slice()));
        assert_eq!(grouped.group_size(), Some(2));
        assert_eq!(grouped.mesh(), None);
        assert_eq!(
            grouped.to_string(),
            "parallel_reduce [kind=mean, axis_name=\"x\", axis_size=4, axis_index_groups=[[0, 2], [3, 1]]]",
        );

        // The participant groups must form an equal-sized exact partition of a non-empty axis.
        for (axis_size, axis_index_groups, message) in [
            (0, vec![vec![0]], "`parallel_reduce` axis size must be greater than zero"),
            (
                4,
                vec![vec![0, 1], vec![1, 2]],
                "`parallel_reduce` axis index groups contain participant 1 more than once",
            ),
            (
                4,
                vec![vec![0, 1, 2], vec![3]],
                "`parallel_reduce` axis index group 1 has size 1 but every group must have size 3",
            ),
            (3, vec![vec![0], vec![1]], "`parallel_reduce` axis index groups do not contain participant 2"),
            (2, vec![vec![0, 2]], "`parallel_reduce` axis index 2 is out of bounds for axis size 2"),
        ] {
            assert_eq!(
                ParallelReduceOperation::grouped(
                    "x".to_string(),
                    ParallelReductionKind::Sum,
                    axis_size,
                    axis_index_groups,
                ),
                Err(TypeError::invalid(message)),
            );
        }

        // Mesh reductions record and render their mesh.
        let (mesh, _, _) = mesh_scalar_types();
        let operation =
            ParallelReduceOperation::new("m".to_string(), ParallelReductionKind::Max).with_mesh(mesh.clone());
        assert_eq!(operation.name(), PARALLEL_REDUCE_OPERATION_NAME);
        assert_eq!(operation.mesh(), Some(&mesh));
        assert_eq!(
            operation.to_string(),
            "parallel_reduce [kind=max, axis_name=\"m\", mesh=['m'=4:manual, 'outer'=2:manual]]",
        );
        assert_ne!(operation, ParallelReduceOperation::new("m".to_string(), ParallelReductionKind::Max));
    }

    #[test]
    fn test_parallel_reduce_type_inference() {
        // Ordinary and grouped reductions preserve their input type, including its manual variation, and every form
        // validates the element data type of its kind.
        let (mesh, invariant, varying) = mesh_scalar_types();
        check_operation_type_inference!(
            operation = ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Sum),
            cases = [
                {
                    input_types = [ArrayType::new_static(DataType::F32, [2, 3])],
                    output_types = [ArrayType::new_static(DataType::F32, [2, 3])],
                },
                { input_types = [varying.clone()], output_types = [varying.clone()] },
                {
                    input_types = [ArrayType::new_static(DataType::Zero, [2])],
                    output_types = [ArrayType::new_static(DataType::Zero, [2])],
                },
                {
                    input_types = [ArrayType::new_static(DataType::Boolean, [2])],
                    error = "`parallel_reduce` with kind `sum` requires numeric inputs but got `bool`",
                },
            ],
        );
        check_operation_type_inference!(
            operation = ParallelReduceOperation::grouped(
                "i".to_string(),
                ParallelReductionKind::Mean,
                4,
                vec![vec![0, 1], vec![2, 3]],
            )
            .unwrap(),
            cases = [
                { input_types = [varying.clone()], output_types = [varying.clone()] },
                {
                    input_types = [ArrayType::new_static(DataType::Boolean, [2])],
                    error = "`parallel_reduce` with kind `mean` requires numeric inputs but got `bool`",
                },
            ],
        );
        check_operation_type_inference!(
            operation = ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Max),
            cases = [
                {
                    input_types = [ArrayType::new_static(DataType::Boolean, [2])],
                    output_types = [ArrayType::new_static(DataType::Boolean, [2])],
                },
                {
                    input_types = [ArrayType::new_static(DataType::Token, [])],
                    error = "`parallel_reduce` with kind `max` requires Boolean or numeric inputs but got `token`",
                },
            ],
        );

        // A maximum and a truncating integer mean do not commute with the pending cross-device sum of an unreduced
        // input, whereas a sum and a floating-point mean do.
        let unreduced = |data_type: DataType| {
            ArrayType::new_static(data_type, [2])
                .with_sharding(Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["outer"]).unwrap())
                .unwrap()
        };
        check_operation_type_inference!(
            operation = ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Sum),
            cases = [{ input_types = [unreduced(DataType::I32)], output_types = [unreduced(DataType::I32)] }],
        );
        check_operation_type_inference!(
            operation = ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Mean),
            cases = [
                { input_types = [unreduced(DataType::F32)], output_types = [unreduced(DataType::F32)] },
                {
                    input_types = [unreduced(DataType::I32)],
                    error = "`parallel_reduce` with kind `mean` does not support unreduced integer inputs",
                },
            ],
        );
        check_operation_type_inference!(
            operation = ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Max),
            cases = [{
                input_types = [unreduced(DataType::F32)],
                error = "`parallel_reduce` with kind `max` does not support unreduced inputs",
            }],
        );

        // A mesh reduction removes its axis from the input's manual variation and preserves every other piece of
        // sharding state, including reduced axes and the variation over other manual axes.
        let sharding = invariant.sharding().unwrap().clone();
        let with_sharding = |sharding: Sharding| ArrayType::scalar(DataType::F32).with_sharding(sharding).unwrap();
        for kind in [ParallelReductionKind::Sum, ParallelReductionKind::Max] {
            let operation = ParallelReduceOperation::new("m".to_string(), kind).with_mesh(mesh.clone());
            assert_eq!(operation.infer_output_types(&[varying.clone()], &[]), Ok(vec![invariant.clone()]));
            assert_eq!(operation.fold(&[varying.clone()], &[]), Ok(None));
            assert_eq!(
                operation.infer_output_types(
                    &[with_sharding(sharding.clone().with_varying_manual_axes(["m", "outer"]).unwrap())],
                    &[],
                ),
                Ok(vec![with_sharding(sharding.clone().with_varying_manual_axes(["outer"]).unwrap())]),
            );
            let reduced = sharding.clone().with_reduced_axes(["outer"]).unwrap();
            assert_eq!(
                operation.infer_output_types(
                    &[with_sharding(reduced.clone().with_varying_manual_axes(["m"]).unwrap())],
                    &[],
                ),
                Ok(vec![with_sharding(reduced)]),
            );

            // The input must vary over the manual axis of the operation's own mesh and carry no reduction state
            // along it, and a mesh reduction cannot also be grouped.
            assert_eq!(
                operation.infer_output_types(&[invariant.clone()], &[]),
                Err(TypeError::invalid(
                    "`parallel_reduce` input must vary over manual axis `m`; pass an invariant value through \
                     `parallel_vary` first so that every copy is counted",
                )),
            );
            assert_eq!(
                operation.infer_output_types(&[ArrayType::scalar(DataType::F32)], &[]),
                Err(TypeError::invalid("`parallel_reduce` input must carry a mesh containing manual axis `m`")),
            );
            for sharding in [
                sharding.clone().with_unreduced_axes(["m"]).unwrap(),
                sharding.clone().with_reduced_axes(["m"]).unwrap(),
            ] {
                assert_eq!(
                    operation.infer_output_types(&[with_sharding(sharding)], &[]),
                    Err(TypeError::invalid("`parallel_reduce` axis `m` must not carry reduction state")),
                );
            }
            let other_mesh = LogicalMesh::new(vec![MeshAxis::new("m", 4, MeshAxisType::Manual).unwrap()]).unwrap();
            assert_eq!(
                ParallelReduceOperation::new("m".to_string(), kind)
                    .with_mesh(other_mesh)
                    .infer_output_types(&[varying.clone()], &[]),
                Err(TypeError::invalid("`parallel_reduce` input mesh does not match the operation mesh")),
            );
            let explicit = LogicalMesh::new(vec![MeshAxis::new("m", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
            assert_eq!(
                ParallelReduceOperation::new("m".to_string(), kind)
                    .with_mesh(explicit)
                    .infer_output_types(&[varying.clone()], &[]),
                Err(TypeError::invalid("`parallel_reduce` mesh axis `m` must be manual")),
            );
            assert_eq!(
                ParallelReduceOperation::grouped("m".to_string(), kind, 4, vec![vec![0, 1], vec![2, 3]])
                    .unwrap()
                    .with_mesh(mesh.clone())
                    .infer_output_types(&[varying.clone()], &[]),
                Err(TypeError::invalid("`parallel_reduce` over a manual mesh axis must not use axis index groups")),
            );
        }

        // The unreduced-input rule applies to mesh reductions as well, and a mesh mean is staged as a sum instead.
        let unreduced = sharding.with_unreduced_axes(["outer"]).unwrap();
        let input = with_sharding(unreduced.clone().with_varying_manual_axes(["m"]).unwrap());
        assert_eq!(
            ParallelReduceOperation::new("m".to_string(), ParallelReductionKind::Sum)
                .with_mesh(mesh.clone())
                .infer_output_types(std::slice::from_ref(&input), &[]),
            Ok(vec![with_sharding(unreduced)]),
        );
        assert_eq!(
            ParallelReduceOperation::new("m".to_string(), ParallelReductionKind::Max)
                .with_mesh(mesh.clone())
                .infer_output_types(&[input], &[]),
            Err(TypeError::invalid("`parallel_reduce` with kind `max` does not support unreduced inputs")),
        );
        assert_eq!(
            ParallelReduceOperation::new("m".to_string(), ParallelReductionKind::Mean)
                .with_mesh(mesh)
                .infer_output_types(&[varying], &[]),
            Err(TypeError::invalid(
                "`parallel_reduce` with kind `mean` over a manual mesh axis must be staged as a `parallel_reduce` with \
                 kind `sum` divided by the axis size",
            )),
        );
    }

    #[test]
    fn test_parallel_reduce_interpretation() {
        // No form has per-item semantics, because the other participants exist only inside an enclosing binder.
        let (mesh, _, _) = mesh_scalar_types();
        let context = EagerContext::<Array>::new();
        let input = Array::scalar(2.0f32).unwrap();
        for (operation, message) in [
            (
                ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Sum),
                "cannot interpret `parallel_reduce` over axis `i` without an enclosing binder",
            ),
            (
                ParallelReduceOperation::grouped(
                    "i".to_string(),
                    ParallelReductionKind::Mean,
                    2,
                    vec![vec![0], vec![1]],
                )
                .unwrap(),
                "cannot interpret `parallel_reduce` over axis `i` without an enclosing binder",
            ),
            (
                ParallelReduceOperation::new("m".to_string(), ParallelReductionKind::Max).with_mesh(mesh),
                "`parallel_reduce` over manual mesh axis `m` requires an execution backend owning the mesh",
            ),
        ] {
            assert_eq!(
                operation.interpret(&context, &EmptyRegionDriver, std::slice::from_ref(&input)),
                Err(ProgramError::UnsupportedOperation { message: message.to_string() }),
            );
        }
    }

    #[test]
    fn test_parallel_reduce_partial_evaluation() {
        let (mesh, invariant, varying) = mesh_scalar_types();
        let ordinary = ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Sum);
        let mesh_sum = ParallelReduceOperation::new("m".to_string(), ParallelReductionKind::Sum).with_mesh(mesh);
        for (operation, input_type, output_type) in [
            (ordinary, ArrayType::scalar(DataType::F32), ArrayType::scalar(DataType::F32)),
            (mesh_sum, varying.clone(), invariant.clone()),
        ] {
            // A known input under an eager parent residualizes the operation, which has no per-item value, so the
            // residual program is the source program itself.
            let input = Array::from_elements(input_type.clone(), &[2.0f32]).unwrap();
            let program = parallel_reduce_program(operation.clone(), input_type.clone());
            let evaluation = program.partially_evaluate(&[PartialValue::Known(input.clone())]).unwrap();
            assert_eq!(evaluation.program().to_string(), program.to_string());
            assert!(evaluation.outputs()[0].is_unknown());

            // The composite family residualizes the operation in the same way.
            let eager =
                PartialEvaluationContext::new(EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new());
            let outputs = ArrayIrOperation::Array(ArrayOperation::ParallelReduce(operation.clone()))
                .partially_evaluate(
                    &eager,
                    &EmptyRegionDriver,
                    &[PartialEvaluationValue::known(ArrayIrValue::Array(input))],
                )
                .unwrap();
            assert!(outputs[0].is_unknown());
            assert_eq!(outputs[0].r#type().as_ref(), &output_type.clone().into());

            // A known input under a staging parent stays known, because the operation is staged into the parent trace.
            let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
            let input = PartialEvaluationValue::known(trace.input(input_type));
            let outputs = operation
                .partially_evaluate(&PartialEvaluationContext::new(trace), &EmptyRegionDriver, &[input])
                .unwrap();
            assert!(outputs[0].is_known());
            assert_eq!(outputs[0].r#type().as_ref(), &output_type);
        }
    }

    #[test]
    fn test_parallel_reduce_batching() {
        // A level that binds the reduction's axis collapses its mapped axis, wherever that axis sits, and returns a
        // replicated result that preserves the element data type (so integer means truncate toward zero).
        let sum = ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Sum);
        let mean = ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Mean);
        let max = ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Max);
        let mapped = |value: Array, axis: usize| ArrayBatch::new(value, BatchAxis::new(axis)).unwrap();
        assert_eq!(
            batch_parallel_reduce(&sum, 2, mapped(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(), 0)),
            Ok(vec![ArrayBatch::replicated(Array::vector(vec![5.0, 7.0, 9.0]).unwrap())]),
        );
        assert_eq!(
            batch_parallel_reduce(&sum, 2, mapped(Array::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(), 1)),
            Ok(vec![ArrayBatch::replicated(Array::vector(vec![3.0, 7.0, 11.0]).unwrap())]),
        );
        assert_eq!(
            batch_parallel_reduce(&max, 3, mapped(Array::vector(vec![1i32, 4, 2]).unwrap(), 0)),
            Ok(vec![ArrayBatch::replicated(Array::scalar(4i32).unwrap())]),
        );
        assert_eq!(
            batch_parallel_reduce(&mean, 3, mapped(Array::vector(vec![2.0, 4.0, 9.0]).unwrap(), 0)),
            Ok(vec![ArrayBatch::replicated(Array::scalar(5.0).unwrap())]),
        );
        assert_eq!(
            batch_parallel_reduce(&mean, 2, mapped(Array::matrix(2, 2, vec![1i32, 2, 2, 5]).unwrap(), 0)),
            Ok(vec![ArrayBatch::replicated(Array::vector(vec![1i32, 3]).unwrap())]),
        );

        // A replicated input holds the same value for every batch item, so its sum counts that value once per item,
        // while its mean and its maximum are the value itself.
        let input = ArrayBatch::replicated(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());
        assert_eq!(
            batch_parallel_reduce(&sum, 3, input.clone()),
            Ok(vec![ArrayBatch::replicated(Array::vector(vec![3.0, 6.0, 9.0]).unwrap())]),
        );
        assert_eq!(batch_parallel_reduce(&mean, 3, input.clone()), Ok(vec![input.clone()]));
        assert_eq!(batch_parallel_reduce(&max, 3, input.clone()), Ok(vec![input.clone()]));

        // A level that binds the axis rejects participant groups.
        let grouped =
            ParallelReduceOperation::grouped("i".to_string(), ParallelReductionKind::Sum, 2, vec![vec![0], vec![1]])
                .unwrap();
        assert_eq!(
            batch_parallel_reduce(&grouped, 2, mapped(Array::vector(vec![1.0, 2.0]).unwrap(), 0)),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_reduce` axis index groups are not supported when a batch transform binds the \
                          collective axis"
                    .to_string(),
            }),
        );

        // A level that binds another axis forwards the reduction to its parent and preserves its mapped axis.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input =
            ArrayBatch::new(trace.input(ArrayType::new_static(DataType::F32, [2, 3])), BatchAxis::new(1)).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace.clone(), 3).with_axis_name("j".to_string());
        let outputs = sum.batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input)).unwrap().into_parts().0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(1));
        assert_eq!(outputs[0].value().r#type().as_ref(), &ArrayType::new_static(DataType::F32, [2, 3]));
        assert_eq!(trace.builder().borrow().instructions().len(), 1);
        assert!(matches!(
            trace.builder().borrow().instructions()[0].operation(),
            ArrayOperation::ParallelReduce(operation) if operation == &sum,
        ));
    }

    #[test]
    fn test_parallel_reduce_batching_mesh() {
        // A level that binds another axis forwards a mesh reduction to its parent with its mapped and ragged axes
        // intact, exactly as it does its adjoint `parallel_vary`, while a level that binds the same name cannot stand
        // in for the manual mesh axis.
        let (mesh, _, _) = mesh_scalar_types();
        let operation =
            ParallelReduceOperation::new("m".to_string(), ParallelReductionKind::Sum).with_mesh(mesh.clone());
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let packed = trace.input(
            ArrayType::new_static(DataType::F32, [2, 3])
                .with_sharding(Sharding::replicated(mesh.clone(), 2).with_varying_manual_axes(["m"]).unwrap())
                .unwrap(),
        );
        let extents = trace.input(ArrayType::new_static(DataType::I32, [2]));
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
        let input = ArrayBatch::new(packed, BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, extents, length, vec![0])])
            .unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace, 2).with_axis_name("i".to_string());
        let outputs =
            operation.batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input)).unwrap().into_parts().0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), input.batch_axis());
        assert_eq!(outputs[0].ragged_axes(), input.ragged_axes());
        assert_eq!(
            outputs[0].value().r#type().as_ref(),
            &ArrayType::new_static(DataType::F32, [2, 3]).with_sharding(Sharding::replicated(mesh, 2)).unwrap(),
        );

        let context =
            BatchingContext::<_, ArrayBatchingPolicy>::new(TracingContext::<Array, ArrayOperation<Array>>::new(), 2)
                .with_axis_name("m".to_string());
        assert!(matches!(
            operation.batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input)),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "`parallel_reduce` over a manual mesh axis cannot bind a named batch axis",
        ));
    }

    #[test]
    fn test_parallel_reduce_batching_ragged() {
        // Padding is replaced with the reduction identity before the participants are combined, and each surviving
        // ragged extent is the maximum of the participants' extents, so positions that no participant covers hold the
        // identity.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
        let ragged = |values: Vec<f32>| {
            ArrayBatch::new(Array::matrix(2, 3, values).unwrap(), BatchAxis::new(0))
                .unwrap()
                .with_ragged_axes(vec![RaggedAxis::new(
                    1,
                    Array::vector(vec![1i32, 2]).unwrap(),
                    length.clone(),
                    vec![0],
                )])
                .unwrap()
        };
        let expected = |values: Vec<f32>| {
            ArrayBatch::replicated(Array::vector(values).unwrap())
                .with_ragged_axes(vec![RaggedAxis::new(0, Array::scalar(2i32).unwrap(), length.clone(), Vec::new())])
                .unwrap()
        };
        let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
            ProjectedContext::new(EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new()),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        )
        .with_axis_name("i".to_string());
        let batch_ragged = |kind: ParallelReductionKind, input: ArrayBatch<Array>| {
            Ok::<_, BatchingError>(
                ParallelReduceOperation::new("i".to_string(), kind)
                    .batch(&context, &EmptyRegionDriver, &[input])?
                    .into_parts()
                    .0,
            )
        };
        assert_eq!(
            batch_ragged(ParallelReductionKind::Sum, ragged(vec![1.0, 100.0, 100.0, 2.0, 3.0, 100.0])),
            Ok(vec![expected(vec![3.0, 3.0, 0.0])]),
        );
        assert_eq!(
            batch_ragged(ParallelReductionKind::Max, ragged(vec![-5.0, 100.0, 100.0, -3.0, -4.0, 100.0])),
            Ok(vec![expected(vec![-3.0, -4.0, f32::NEG_INFINITY])]),
        );

        // A mean has no single implied denominator for ragged inputs.
        assert_eq!(
            batch_ragged(ParallelReductionKind::Mean, ragged(vec![1.0; 6])),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_reduce` with kind `mean` does not define a denominator for bounded ragged inputs"
                    .to_string(),
            }),
        );

        // Under a staging parent with a dynamic extent, the masks and both reductions are staged.
        for (kind, expected) in [
            (
                ParallelReductionKind::Sum,
                indoc! {"
                    lambda %0:dimension<items ∈ [1, 9)>, %1:f32[items, 3], %2:i32[items] .
                    let %3:dimension<items ∈ [1, 9)> = dimension_size [axis=0] %1
                        %4:dimension<3> = constant [value=3]
                        %5:i32[3] = iota [type=i32[3], dimension=0]
                        %6:i32[items, 3] = broadcast [output_axes=[1]] %5 %3 %4
                        %7:i32[items, 3] = broadcast [output_axes=[0]] %2 %3 %4
                        %8:bool[items, 3] = compare [direction=LessThan] %6 %7
                        %9:f32[items, 3] = zero_like %1
                        %10:f32[items, 3] = select %8 %1 %9
                        %11:f32[3] = reduce [kind=sum, axes=[0]] %10
                        %12:i32[] = reduce [kind=max, axes=[0]] %2
                    in (%11, %12)"
                },
            ),
            (
                ParallelReductionKind::Max,
                indoc! {"
                    lambda %0:dimension<items ∈ [1, 9)>, %1:f32[items, 3], %2:i32[items] .
                    let %3:dimension<items ∈ [1, 9)> = dimension_size [axis=0] %1
                        %4:dimension<3> = constant [value=3]
                        %5:i32[3] = iota [type=i32[3], dimension=0]
                        %6:i32[items, 3] = broadcast [output_axes=[1]] %5 %3 %4
                        %7:i32[items, 3] = broadcast [output_axes=[0]] %2 %3 %4
                        %8:bool[items, 3] = compare [direction=LessThan] %6 %7
                        %9:f32[] = constant [value=-inf]
                        %10:f32[items, 3] = broadcast [output_axes=[]] %9 %3 %4
                        %11:f32[items, 3] = select %8 %1 %10
                        %12:f32[3] = reduce [kind=max, axes=[0]] %11
                        %13:i32[] = reduce [kind=max, axes=[0]] %2
                    in (%12, %13)"
                },
            ),
        ] {
            let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
            let batch_extent = trace.input(DimensionType::from(items.clone()).into());
            let packed = trace.input(
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(items.clone()), Dimension::Static(3)]),
                )
                .into(),
            );
            let extents =
                trace.input(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(items)])).into());
            let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
                ProjectedContext::new(trace.clone()),
                batch_extent,
            )
            .with_axis_name("i".to_string());
            let input = ArrayBatch::new(packed.into_projected().unwrap(), BatchAxis::new(0))
                .unwrap()
                .with_ragged_axes(vec![RaggedAxis::new(1, extents.into_projected().unwrap(), length.clone(), vec![0])])
                .unwrap();
            let output = ParallelReduceOperation::new("i".to_string(), kind)
                .batch(&context, &EmptyRegionDriver, &[input])
                .unwrap()
                .into_parts()
                .0
                .remove(0);
            assert_eq!(output.batch_axis(), BatchAxis::replicated());
            assert_eq!(output.ragged_axes().len(), 1);
            let ragged_axis = &output.ragged_axes()[0];
            assert_eq!((ragged_axis.axis(), ragged_axis.dimension(), ragged_axis.extent_axes()), (0, &length, &[][..]));
            let extent = ragged_axis.extents().clone().into_value().atom_id().unwrap();
            let value = output.into_value().into_value().atom_id().unwrap();
            let program = trace
                .builder()
                .borrow()
                .clone()
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![value, extent],
                    vec![Placeholder, Placeholder, Placeholder],
                    vec![Placeholder, Placeholder],
                )
                .unwrap();
            assert_eq!(program.to_string(), expected);
        }
    }

    #[test]
    fn test_parallel_reduce_batching_dynamic_extent() {
        // A mean over a dynamic mapped extent divides by the runtime extent through the staged reduction.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_extent = trace.input(DimensionType::from(items.clone()).into());
        let packed = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(items), Dimension::Static(3)])).into(),
        );
        let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
            ProjectedContext::new(trace.clone()),
            batch_extent,
        )
        .with_axis_name("i".to_string());
        let input = ArrayBatch::new(packed.into_projected().unwrap(), BatchAxis::new(0)).unwrap();
        let output = ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Mean)
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts()
            .0
            .remove(0);
        assert_eq!(output.batch_axis(), BatchAxis::replicated());
        let value = output.into_value().into_value().atom_id().unwrap();
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![value],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<items ∈ [1, 9)>, %1:f32[items, 3] .
                let %2:f32[3] = reduce [kind=mean, axes=[0]] %1
                in (%2)"
            },
        );
    }

    #[test]
    fn test_parallel_reduce_differentiation() {
        // Sums and means are linear, so their tangent is the same reduction of the input tangent.
        let (mesh, invariant, varying) = mesh_scalar_types();
        let input_type = ArrayType::scalar(DataType::F32);
        assert_eq!(
            parallel_reduce_program(
                ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Mean),
                input_type,
            )
            .jvp()
            .unwrap()
            .to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[] = parallel_reduce [kind=mean, axis_name=\"i\"] %0
                    %3:f32[] = parallel_reduce [kind=mean, axis_name=\"i\"] %1
                in (%2, %3)"
            },
        );
        let mesh_sum = ParallelReduceOperation::new("m".to_string(), ParallelReductionKind::Sum).with_mesh(mesh);
        assert_eq!(
            parallel_reduce_program(mesh_sum.clone(), varying.clone()).jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['m'=4:manual, 'outer'=2:manual]>, [], varying_manual={'m'}}], \
                %1:f32[][sharding={mesh<['m'=4:manual, 'outer'=2:manual]>, [], varying_manual={'m'}}] .
                let %2:f32[][sharding={mesh<['m'=4:manual, 'outer'=2:manual]>, []}] = parallel_reduce [kind=sum, axis_name=\"m\", mesh=['m'=4:manual, 'outer'=2:manual]] %0
                    %3:f32[][sharding={mesh<['m'=4:manual, 'outer'=2:manual]>, []}] = parallel_reduce [kind=sum, axis_name=\"m\", mesh=['m'=4:manual, 'outer'=2:manual]] %1
                in (%2, %3)"
            },
        );

        // A structural zero tangent stays a structural zero, whose type follows the mesh reduction's type inference,
        // and only the primal reduction is staged.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = context.input(varying.clone());
        let outputs = mesh_sum
            .jvp(
                &DifferentiationContext::fused(context.clone()),
                &EmptyRegionDriver,
                &[DifferentiationDual::new_with_zero_tangent(input.clone()).unwrap()],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].tangent().is_zero());
        assert_eq!(outputs[0].tangent().r#type().as_ref(), &invariant);
        assert_eq!(context.builder().borrow().instructions().len(), 1);

        // A maximum is nonlinear and has no differentiation rule.
        assert!(matches!(
            ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Max).jvp(
                &DifferentiationContext::fused(context.clone()),
                &EmptyRegionDriver,
                &[DifferentiationDual::new_with_zero_tangent(input).unwrap()],
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`parallel_reduce` with kind `max` is not differentiable",
        ));

        // Reverse mode through a level that binds the axis gives every item the cotangent of the shared total, which
        // a mean scales by the inverse batch size.
        let inputs = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        for (kind, value, gradient) in
            [(ParallelReductionKind::Sum, 6.0, 1.0), (ParallelReductionKind::Mean, 2.0, 1.0 / 3.0)]
        {
            assert_eq!(
                differentiate_at(inputs.clone()).value_and_gradient(|inputs| {
                    Ok(batch(
                        |item| item.parallel_reduce("i", kind),
                        inputs,
                        BatchAxis::new(0),
                        BatchAxis::replicated(),
                        BatchAxisSpecification::named("i"),
                    )?)
                }),
                Ok((Array::scalar(value).unwrap(), Array::vector(vec![gradient; 3]).unwrap())),
            );
        }
    }

    #[test]
    fn test_parallel_reduce_transposition() {
        // Ordinary sums and means are self-adjoint.
        let (mesh, invariant, varying) = mesh_scalar_types();
        for kind in [ParallelReductionKind::Sum, ParallelReductionKind::Mean] {
            let program = parallel_reduce_program(
                ParallelReduceOperation::new("i".to_string(), kind),
                ArrayType::scalar(DataType::F32),
            );
            assert_eq!(
                program.transpose_with_respect_to(&[0], &[]).unwrap().to_string(),
                format!("lambda %0:f32[] .\nlet %1:f32[] = parallel_reduce [kind={kind}, axis_name=\"i\"] %0\nin (%1)"),
            );
        }

        // A mesh sum hands the shared cotangent to every device, which is a `parallel_vary`, and transposing that
        // recovers the sum.
        let mesh_sum = ParallelReduceOperation::new("m".to_string(), ParallelReductionKind::Sum).with_mesh(mesh);
        let program = parallel_reduce_program(mesh_sum.clone(), varying.clone());
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['m'=4:manual, 'outer'=2:manual]>, []}] .
                let %1:f32[][sharding={mesh<['m'=4:manual, 'outer'=2:manual]>, [], varying_manual={'m'}}] = parallel_vary [axis_name=\"m\"] %0
                in (%1)"
            },
        );
        assert_eq!(transposed.transpose_with_respect_to(&[0], &[]).unwrap().to_string(), program.to_string());

        // Structural zero and unrequested cotangents stage nothing.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let inputs = [PartialValue::Unknown(varying.clone())];
        let mut transposition = TranspositionContext::new(context.clone());
        let accumulators = transposition.cotangent_accumulators(&inputs, &[]).unwrap();
        mesh_sum
            .transpose(
                &mut transposition,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Zero(invariant.clone())],
                &accumulators,
            )
            .unwrap();
        let contributions = transposition.take_cotangents(&accumulators).unwrap();
        assert_eq!(contributions.len(), 1);
        assert!(contributions[0].is_zero());
        let accumulators = transposition.cotangent_accumulators(&inputs, &[false]).unwrap();
        let cotangent = context.input(invariant.clone());
        mesh_sum
            .transpose(
                &mut transposition,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Value(cotangent.clone())],
                &accumulators,
            )
            .unwrap();
        assert!(transposition.take_cotangents(&accumulators).unwrap()[0].is_zero());
        assert!(context.builder().borrow().instructions().is_empty());

        // A maximum is nonlinear and cannot be transposed.
        let accumulators = transposition.cotangent_accumulators(&inputs, &[]).unwrap();
        assert!(matches!(
            ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Max).transpose(
                &mut transposition,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Value(cotangent)],
                &accumulators,
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`parallel_reduce` with kind `max` is not transposable",
        ));
    }

    #[test]
    fn test_parallel_reduce_parallel_reduce() {
        // A name that no enclosing binder binds fails fast instead of silently acting as identity.
        let inputs = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(
            batch(
                |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>| {
                    item.parallel_reduce("j", ParallelReductionKind::Sum)
                },
                inputs.clone(),
                BatchAxis::new(0),
                BatchAxis::replicated(),
                BatchAxisSpecification::named("i"),
            ),
            Err::<Array, _>(BatchingError::Axis(AxisError::UnboundAxisName { name: "j".to_string() })),
        );

        // A name that a `batch` level binds reduces across its batch items, and a value that is the same for every
        // item is counted once per item by a sum.
        for (kind, mapped, replicated) in [
            (ParallelReductionKind::Sum, 6.0, 30.0),
            (ParallelReductionKind::Mean, 2.0, 10.0),
            (ParallelReductionKind::Max, 3.0, 10.0),
        ] {
            assert_eq!(
                batch(
                    |(item, constant): (
                        BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>,
                        _,
                    )| {
                        Ok((item.parallel_reduce("i", kind)?, constant.parallel_reduce("i", kind)?))
                    },
                    (inputs.clone(), Array::scalar(10.0).unwrap()),
                    (BatchAxis::new(0), BatchAxis::replicated()),
                    (BatchAxis::replicated(), BatchAxis::replicated()),
                    BatchAxisSpecification::named("i"),
                ),
                Ok((Array::scalar(mapped).unwrap(), Array::scalar(replicated).unwrap())),
            );
        }

        // Names bound by nested `batch` levels resolve to the matching level: the inner reduction over the outer axis
        // sums the columns of the outer items.
        let matrix = Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(
            batch(
                |row| {
                    Ok(batch(
                        |scalar| scalar.parallel_reduce("outer", ParallelReductionKind::Sum),
                        row,
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        BatchAxisSpecification::named("inner"),
                    )?)
                },
                matrix,
                BatchAxis::new(0),
                BatchAxis::replicated(),
                BatchAxisSpecification::named("outer"),
            ),
            Ok(Array::vector(vec![4.0, 6.0]).unwrap()),
        );

        // Over a manual mesh axis, a varying value is reduced directly, while an invariant value, and a value without a
        // sharding, are first made varying, so that every copy is counted.
        let (mesh, invariant, varying) = mesh_scalar_types();
        for (input_type, kind, expected) in [
            (
                varying.clone(),
                ParallelReductionKind::Sum,
                indoc! {"
                    lambda %0:f32[][sharding={mesh<['m'=4:manual, 'outer'=2:manual]>, [], varying_manual={'m'}}] .
                    let %1:f32[][sharding={mesh<['m'=4:manual, 'outer'=2:manual]>, []}] = parallel_reduce [kind=sum, axis_name=\"m\", mesh=['m'=4:manual, 'outer'=2:manual]] %0
                    in (%1)"
                },
            ),
            (
                invariant.clone(),
                ParallelReductionKind::Sum,
                indoc! {"
                    lambda %0:f32[][sharding={mesh<['m'=4:manual, 'outer'=2:manual]>, []}] .
                    let %1:f32[][sharding={mesh<['m'=4:manual, 'outer'=2:manual]>, [], varying_manual={'m'}}] = parallel_vary [axis_name=\"m\"] %0
                        %2:f32[][sharding={mesh<['m'=4:manual, 'outer'=2:manual]>, []}] = parallel_reduce [kind=sum, axis_name=\"m\", mesh=['m'=4:manual, 'outer'=2:manual]] %1
                    in (%2)"
                },
            ),
            (
                ArrayType::scalar(DataType::F32),
                ParallelReductionKind::Sum,
                indoc! {"
                    lambda %0:f32[] .
                    let %1:f32[][sharding={mesh<['m'=4:manual, 'outer'=2:manual]>, []}] = broadcast [
                        output_type=f32[][sharding={mesh<['m'=4:manual, 'outer'=2:manual]>, []}],
                        output_axes=[],
                    ] %0
                        %2:f32[][sharding={mesh<['m'=4:manual, 'outer'=2:manual]>, [], varying_manual={'m'}}] = parallel_vary [axis_name=\"m\"] %1
                        %3:f32[][sharding={mesh<['m'=4:manual, 'outer'=2:manual]>, []}] = parallel_reduce [kind=sum, axis_name=\"m\", mesh=['m'=4:manual, 'outer'=2:manual]] %2
                    in (%3)"
                },
            ),
            (
                varying.clone(),
                ParallelReductionKind::Max,
                indoc! {"
                    lambda %0:f32[][sharding={mesh<['m'=4:manual, 'outer'=2:manual]>, [], varying_manual={'m'}}] .
                    let %1:f32[][sharding={mesh<['m'=4:manual, 'outer'=2:manual]>, []}] = parallel_reduce [kind=max, axis_name=\"m\", mesh=['m'=4:manual, 'outer'=2:manual]] %0
                    in (%1)"
                },
            ),
            // A mean over a manual mesh axis is the mesh sum divided by the axis size.
            (
                varying.clone(),
                ParallelReductionKind::Mean,
                indoc! {"
                    lambda %0:f32[][sharding={mesh<['m'=4:manual, 'outer'=2:manual]>, [], varying_manual={'m'}}] .
                    let %1:f32[][sharding={mesh<['m'=4:manual, 'outer'=2:manual]>, []}] = parallel_reduce [kind=sum, axis_name=\"m\", mesh=['m'=4:manual, 'outer'=2:manual]] %0
                        %2:f32[] = constant [value=4.0]
                        %3:f32[][sharding={mesh<['m'=4:manual, 'outer'=2:manual]>, []}] = div %1 %2
                    in (%3)"
                },
            ),
        ] {
            let (output, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_reduce("m", kind),
                input_type,
                manual_mesh_axes(&mesh),
            )
            .unwrap();
            assert_eq!(output, invariant);
            assert_eq!(program.to_string(), expected);
        }
    }

    #[test]
    fn test_parallel_reduce_parallel_reduce_with_axis_index_groups() {
        // Grouped reductions record the resolved axis size and preserve the input's manual variation.
        let (mesh, _, varying) = mesh_scalar_types();
        let (output, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| {
                input.parallel_reduce_with_axis_index_groups(
                    "m",
                    ParallelReductionKind::Sum,
                    vec![vec![0, 1], vec![2, 3]],
                )
            },
            varying.clone(),
            manual_mesh_axes(&mesh),
        )
        .unwrap();
        assert_eq!(output, varying);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['m'=4:manual, 'outer'=2:manual]>, [], varying_manual={'m'}}] .
                let %1:f32[][sharding={mesh<['m'=4:manual, 'outer'=2:manual]>, [], varying_manual={'m'}}] = parallel_reduce [kind=sum, axis_name=\"m\", axis_size=4, axis_index_groups=[[0, 1], [2, 3]]] %0
                in (%1)"
            },
        );

        // The groups must partition the resolved axis, and the axis must be bound.
        assert_eq!(
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_reduce_with_axis_index_groups("m", ParallelReductionKind::Sum, vec![vec![0, 1]]),
                varying.clone(),
                manual_mesh_axes(&mesh),
            )
            .map(|(output, _)| output),
            Err(ProgramError::Type(TypeError::invalid(
                "`parallel_reduce` axis index groups do not contain participant 2",
            ))),
        );
        assert_eq!(
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_reduce_with_axis_index_groups("x", ParallelReductionKind::Sum, vec![vec![0]]),
                varying,
                manual_mesh_axes(&mesh),
            )
            .map(|(output, _)| output),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "x".to_string() })),
        );
    }
}
