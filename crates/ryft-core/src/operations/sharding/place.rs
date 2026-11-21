//! Moving values between concrete placements, which is the Ryft analogue of JAX's
//! [`device_put`](https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html). A [`Placement`] names where an
//! array lives concretely (i.e., a [`Sharding`] over the devices of a [`DeviceMesh`](crate::DeviceMesh), optionally
//! with a [`Memory`](crate::Memory) space and a storage [`Layout`](crate::Layout)). Moving a value to another placement
//! never changes its elements, shape, or data type, and unlike a [`Reshard`](crate::Reshard), which redistributes a
//! value over the mesh that it already lives on, it can move the value onto the devices of another mesh.
//!
//! The module is organized around the following entities:
//!
//!   - [`PlaceOperation`] is the operation that a placement stages. It carries an optional source placement, an
//!     optional target placement, and a copy policy that states whether its output may share storage with its input,
//!     and it owns the type inference rule that validates both placements (e.g., that the target preserves the
//!     reduction state of the input and splits every static extent evenly) along with the interpretation, partial
//!     evaluation, batching, differentiation, and transposition rules of a placement. Without a target, the value
//!     stays where it is.
//!   - [`Placed`] reports the placement that a value concretely lives on. Only values that hold device buffers (e.g.,
//!     the arrays of a backend) have one, because devices and placement over [`MeshAxisType::Auto`] mesh axes are
//!     runtime state that types do not record. Staged values and reference [`Array`]s have none, wrappers such as
//!     tracers report the placement of the value that they wrap, and batched values report that of their batch items.
//!   - [`PlaceDispatch`] places one leaf with an optional target and a given copy policy through the
//!     [`Value::Dispatch`] policy of its value type. Values that dispatch through their domain bind a
//!     [`PlaceOperation`] in their own context, reference [`Array`]s record the placement in their types,
//!     and array IR values place their array members.
//!   - [`Place`] is the value capability through which callers place individual values and the leaves of structured
//!     values, with one target per leaf, preserving their structure and static fields. Its results may share storage
//!     with its inputs, and callers that need results with storage of their own use [`PlaceDispatch`] directly.
//!
//! The source of a placement is informative. It states where the value is expected to live when the placement is
//! created, which transposition needs in order to move the cotangent back to those devices, because types do not record
//! them. Type inference checks a source against what the input type tracks, while backends that execute a placement on
//! a concrete value move it from wherever it actually lives. When a caller of [`Place`] omits the source of a leaf, the
//! leaf's [`Placed`] placement is used instead, so concrete backend arrays record where they actually live, while
//! staged values leave the source unknown. Inside batched functions, the placement of a batch item also records how its
//! batch is distributed (refer to [`Placement::batch_shardings`]), so a source inferred from a batch item is lifted
//! back to where the packed value lives, an explicit target places every batch item where it says, and a placement
//! without a target leaves the packed value where it is.

use std::fmt::Display;
use std::sync::Arc;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrBatch, ArrayIrType, ArrayIrValue,
    ArrayType, Dimension, MeshAxisType, Placement, Sharding, ShardingDimension,
};
use crate::batching::{
    BatchableOperation, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy, BatchingTracer,
};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{DifferentiableType, DifferentiationDual, DifferentiationPolicy, DifferentiationTracer};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::Capability;
use crate::operations::manipulation::broadcasting::{Broadcast, BroadcastOperation};
use crate::operations::sharding::validate_preserved_mesh_axes;
use crate::parameters::{Parameter, Parameterized, ParameterizedFamily};
use crate::partial::{PartialTracer, PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, OperationProjection, OperationProvider, ProgramError, ProjectedValue,
    RegionInterface, Type, TypeError, Typed, Value, ValueDirectDispatch, ValueDomainDispatch, ValueProjection,
};
use crate::tracing::Tracer;

/// Canonical operation name for [`PlaceOperation`].
pub const PLACE_OPERATION_NAME: &str = "place";

/// [`Operation`] that moves its input to a target [`Placement`], possibly on the devices of another
/// [`DeviceMesh`](crate::DeviceMesh). This operation is the Ryft analogue of JAX's
/// [`jax.device_put`](https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html). The array's elements, shape,
/// and data type are unchanged. Type inference returns the input type with its sharding replaced by the target sharding
/// without its [`MeshAxisType::Auto`] mesh axes (which array types cannot record, refer to
/// [`Sharding::without_auto_axes`]), its memory replaced by the target memory when the target
/// names one, and its layout replaced by the target layout when the target names one.
///
/// The target is optional. Without a target, the value stays where it is (i.e., its devices, sharding, memory, and
/// layout are unchanged), and type inference returns the input type. Such an operation still records its source, which
/// transposition uses to move the cotangent back to it, and its copy policy, so that it can force a copy of a concrete
/// value. An operation without a target, without a source, and whose output may share storage with its input is
/// therefore an identity.
///
/// Unlike a [`ReshardOperation`](crate::ReshardOperation), which redistributes a value over the mesh that it already
/// lives on, a placement may move the value to another mesh, so the target mesh need not define the axes of the input's
/// mesh. A placement still changes only where a value lives and never its reduction state, so the target sharding must
/// restate the unreduced and reduced axes of the input's tracked sharding (and its unreduced kind) exactly, and the
/// target mesh must define each of those axes with the same size and type as the input's mesh. Every shard of a
/// partitioned dimension has the same extent, so for each dimension that the target sharding partitions, the number
/// of shards along it (i.e., the product of the sizes of the mesh axes that shard it, [`MeshAxisType::Auto`] axes
/// included) must divide its static extent, as JAX's `device_put` also requires. A dynamic extent is only known when a
/// backend places a concrete value, so type inference accepts any split of it and backends check it when they place the
/// value.
///
/// A value inside a manual region lives on the devices of that region, so it cannot be moved, and nothing can be moved
/// into such a region either. Types show such values through their variation along [`MeshAxisType::Manual`] mesh axes
/// (refer to [`Sharding::varying_manual_axes`]) or their reduction state over them, so neither the target or source
/// sharding nor the tracked sharding of an input that is moved to a target may have either. An operation without a
/// target leaves such a value where it is, so it accepts it unless its source describes it. A global array outside
/// any manual region can still be tiled over a manual axis of its mesh, and it is moved like any other array. Backends
/// separately reject placements onto a target inside active manual regions when they lower a program. A target layout
/// must address the shape of the input (refer to [`ArrayType::validate_layout`]).
///
/// The optional source [`Placement`] is informative. It states where the input is expected to live when the operation
/// is created, and transposition uses it to move the cotangent back to those devices. Type inference checks it against
/// what the input type tracks. The source sharding must have the rank of the input, must not describe a value inside
/// a manual region, and must restate the reduction state of the input exactly (i.e., its unreduced axes, reduced axes,
/// and unreduced kind, which are empty and a sum for an input without a tracked sharding) over a mesh that defines
/// those axes like the input's mesh. When the input type tracks a sharding, the source sharding without its auto axes
/// must also equal it. These checks guarantee that the cotangent that transposition moves back to the source devices
/// has the cotangent type of the input (up to a placement-only broadcast for an input without a tracked sharding).
/// The memory and layout of the source are not compared with the input type. Backends that execute the operation on a
/// concrete value move the value from wherever it actually lives, without comparing that placement with the source.
///
/// The operation also carries a copy policy (refer to [`Self::with_may_alias`]), which is the Ryft analogue of the
/// `copy_semantics` of JAX's `device_put` primitive. By default, the output may share storage with the input (e.g.,
/// a placement that leaves a concrete value where it already lives can return that value), and with `may_alias` set to
/// `false`, the output has storage of its own, which no other value shares. The policy does not affect type inference.
/// It only matters for concrete values, whose storage other handles can observe. A compiled program has no observable
/// buffer identity, because its outputs are buffers that its execution allocates (or donated inputs, which no other
/// handle can observe), so backends ignore the policy when they lower the operation.
///
/// Interpretation moves the value through the [`PlaceDispatch`] primitive of [`Place`] with the operation's source,
/// target, and copy policy. Batching keeps the copy policy and extends the source and target shardings with an entry
/// for the batch dimension. A target or source that records how its batch is distributed (refer to
/// [`Placement::batch_shardings`]), such as the placement that a batch item reports (refer to [`Placed`]), gets that
/// distribution back. Otherwise, an explicit target replicates the batch dimension, so that it places every batch item
/// where it says, and a source gets the placement that the packed input type records for its batch dimension. This
/// differs from JAX, whose batching rule applies an unchanged target to the packed value positionally, so that a target
/// that splits the first dimension of a batch item splits the batch dimension instead when the batch is mapped at
/// position `0`. An omitted target stays omitted and leaves the packed value where it is. Differentiation places the
/// tangent with the same target, source, and copy policy as the primal but without the layouts of the target and
/// source, which need not be valid for the element type of the tangent. When the operation has no source, both use the
/// concrete placement of the primal as their source, if it has one. Transposition moves the cotangent back to where
/// the input lived, and it always forces a copy, as JAX's `device_put` transpose does (refer to the transposition
/// rule of this operation for the details).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PlaceOperation {
    /// Optional source [`Placement`] where the input is expected to live, which is informative only.
    source: Option<Placement>,

    /// Optional target [`Placement`] that the input is moved to. When it is [`None`], the input stays where it is.
    target: Option<Placement>,

    /// Boolean indicating whether the output may share storage with the input. When it is `false`, the output must
    /// have storage of its own.
    may_alias: bool,
}

impl PlaceOperation {
    /// Creates a new [`PlaceOperation`] that records `source` as the expected placement of its input and moves it to
    /// `target`. A [`None`] source means the expected placement is unknown. A [`None`] target leaves the input where
    /// it is. The output may share storage with its input (refer to [`Self::with_may_alias`]).
    #[inline]
    pub fn new(source: Option<Placement>, target: impl Into<Option<Placement>>) -> Self {
        Self { source, target: target.into(), may_alias: true }
    }

    /// Returns this [`PlaceOperation`] with its copy policy replaced by `may_alias`. With `true`, which is the default,
    /// the output may share storage with the input (e.g., when the placement leaves the input where it already lives),
    /// and with `false`, the output has storage of its own, which no other value shares. This is the analogue of the
    /// `copy_semantics` of JAX's `device_put` primitive, where `false` corresponds to a forced copy.
    #[inline]
    pub fn with_may_alias(mut self, may_alias: bool) -> Self {
        self.may_alias = may_alias;
        self
    }

    /// Returns the source [`Placement`] where the input is expected to live, or [`None`] if it is not known.
    #[inline]
    pub fn source(&self) -> Option<&Placement> {
        self.source.as_ref()
    }

    /// Returns the target [`Placement`] that the input is moved to, or [`None`] if the input stays where it is.
    #[inline]
    pub fn target(&self) -> Option<&Placement> {
        self.target.as_ref()
    }

    /// Returns whether the output may share storage with the input (refer to [`Self::with_may_alias`]).
    #[inline]
    pub fn may_alias(&self) -> bool {
        self.may_alias
    }
}

impl Display for PlaceOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for PlaceOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        PLACE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);

        let input = &input_types[0];
        let target = self.target.as_ref().map(Placement::sharding);
        if let Some(target) = target
            && input.rank() != target.rank()
        {
            return Err(TypeError::invalid(format!(
                "`{}` target sharding rank ({}) does not match the input rank ({})",
                PLACE_OPERATION_NAME,
                target.rank(),
                input.rank(),
            )));
        }

        let source = self.source.as_ref().map(Placement::sharding);
        if let Some(source) = source
            && source.rank() != input.rank()
        {
            return Err(TypeError::invalid(format!(
                "`{}` source sharding rank ({}) does not match the input rank ({})",
                PLACE_OPERATION_NAME,
                source.rank(),
                input.rank(),
            )));
        }

        // Values of a manual region live on the devices of that region, so they cannot be moved anywhere else, and
        // nothing can be moved into such a region either. Their types show it through their variation along manual
        // axes or their reduction state over them (i.e., per device values or pending reductions across the devices of
        // the region), whereas a global array can be tiled over a manual axis outside any manual region and is moved
        // like any other array. Backends reject placements inside active manual regions separately. A placement without
        // a target leaves its input where it is, so it accepts such an input, unless its source describes it.
        let is_manual_region_sharding = |sharding: &Sharding| {
            !sharding.varying_manual_axes().is_empty()
                || sharding
                    .unreduced_axes()
                    .iter()
                    .chain(sharding.reduced_axes())
                    .any(|axis| sharding.mesh().axis_type(axis) == Some(MeshAxisType::Manual))
        };
        if target.is_some_and(|target| {
            input.sharding().is_some_and(is_manual_region_sharding) || is_manual_region_sharding(target)
        }) || source.is_some_and(is_manual_region_sharding)
        {
            return Err(TypeError::invalid(format!(
                "`{PLACE_OPERATION_NAME}` cannot move values that vary along manual mesh axes \
                 or carry reduction state over them",
            )));
        }

        // A placement changes where a value lives but never its reduction state, which records pending or completed
        // cross-device reductions. The target therefore restates the input's unreduced and reduced axes exactly, and
        // so does a given source, which describes the input. The state is restated by axis name, so the target and
        // source meshes must also define every axis that carries it identically (e.g., an explicit unreduced axis must
        // not become an auto axis, over which array types record no reduction state, so that the output would silently
        // lose its pending reduction). Array types never record reduction state over auto axes, so a target or source
        // with such state is rejected here as well. An input without a tracked sharding has no reduction state.
        if let Some(input_sharding) = input.sharding() {
            let reduction_axes = || input_sharding.unreduced_axes().iter().chain(input_sharding.reduced_axes());
            for mesh in source.into_iter().chain(target).map(Sharding::mesh) {
                validate_preserved_mesh_axes(PLACE_OPERATION_NAME, input_sharding.mesh(), reduction_axes(), mesh)?;
            }
        }

        let (unreduced_axes, reduced_axes) = match input.sharding() {
            Some(sharding) => (sharding.unreduced_axes().clone(), sharding.reduced_axes().clone()),
            None => Default::default(),
        };

        let has_input_reduction_state = |sharding: &Sharding| {
            sharding.unreduced_axes() == &unreduced_axes
                && sharding.reduced_axes() == &reduced_axes
                && sharding.unreduced_kind() == input.unreduced_kind()
        };

        if let Some(source) = source
            && !has_input_reduction_state(source)
        {
            return Err(TypeError::invalid(format!(
                "`{PLACE_OPERATION_NAME}` source sharding `{source}` does not match the reduction state of the input \
                 type `{input}`",
            )));
        }

        if target.is_some_and(|target| !has_input_reduction_state(target)) {
            return Err(TypeError::invalid(format!(
                "`{PLACE_OPERATION_NAME}` cannot change reduction state; use a reduction (or the corresponding \
                 collective) instead",
            )));
        }

        // A given source must also describe where the input lives as far as its type can tell, which excludes the
        // placement over auto axes that types do not record.
        if let Some(source) = source
            && let Some(input_sharding) = input.sharding()
        {
            let source = source.without_auto_axes();
            if input_sharding != &source {
                return Err(TypeError::invalid(format!(
                    "`{PLACE_OPERATION_NAME}` source sharding `{source}` does not match the input type `{input}`",
                )));
            }
        }

        // Without a target, the value stays where it is, so it keeps its type.
        let Some(placement) = &self.target else {
            return Ok(vec![input.clone()]);
        };
        let target = placement.sharding();

        // Every shard of a partitioned dimension has the same extent, so the number of shards along each dimension that
        // the target partitions (i.e., the product of the sizes of the mesh axes that shard it) must divide its static
        // extent. This includes auto axes, which the output type does not record but along which the value is still
        // split. A dynamic extent is only known when a backend places a concrete value, which is when it is checked.
        for (dimension, (extent, sharding)) in input.shape().dimensions().iter().zip(target.dimensions()).enumerate() {
            let (Dimension::Static(extent), ShardingDimension::Sharded(axis_names)) = (extent, sharding) else {
                continue;
            };

            // The product is over distinct axes of one mesh, so it never exceeds the device count of that mesh,
            // which fits `usize`.
            let shard_count = axis_names.iter().filter_map(|name| target.mesh().axis_size(name)).product::<usize>();
            if extent % shard_count != 0 {
                return Err(TypeError::invalid(format!(
                    "`{PLACE_OPERATION_NAME}` target sharding splits dimension #{dimension} with extent {extent} into \
                     {shard_count} shards, which do not divide it evenly",
                )));
            }
        }

        let mut output = input.clone().with_sharding(target.without_auto_axes()).map_err(TypeError::from)?;
        if let Some(memory) = placement.memory() {
            output = output.with_memory(memory);
        }

        if let Some(layout) = placement.layout() {
            // A target layout must address the output shape, so that a malformed request fails here rather than only
            // when a backend materializes or lowers the output.
            output = output.with_layout(layout.clone());
            output.validate_layout().map_err(|error| {
                TypeError::invalid(format!("`{PLACE_OPERATION_NAME}` target layout is invalid: {error}"))
            })?;
        }

        Ok(vec![output])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        // An omitted source or target is not rendered, and neither is the default copy policy, so that the rendering
        // of a placement stays compact. An identity placement therefore renders as its name alone.
        let formatter = OperationFormatter::new(formatter, indentation, self.name())?;
        if self.source.is_none() && self.target.is_none() && self.may_alias {
            return Ok(());
        }
        formatter.bracketed(|operation| {
            if let Some(source) = &self.source {
                operation.field("source", source)?;
            }
            if let Some(target) = &self.target {
                operation.field("target", target)?;
            }
            if !self.may_alias {
                operation.field("may_alias", false)?;
            }
            Ok(())
        })
    }
}

impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for PlaceOperation
where
    C::Value: Placed + Value<Dispatch: PlaceDispatch<ArrayType, C::Value>>,
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // The placement flows through the dispatch primitive of `Place`, which also carries the optional target and the
        // copy policy, so that interpretation over staging values (e.g., program batching, re-tracing, etc.) preserves
        // them, while concrete values move to the target or stay where they are without one.
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        let source = self.source.clone().or_else(|| input.placement());
        Ok(vec![<C::Value as Value>::Dispatch::place(input, source, self.target.as_ref(), self.may_alias)?])
    }
}

impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for PlaceOperation
where
    C::Operation: From<PlaceOperation>,
{
    // Partial evaluation defers to the default fold-or-residualize behavior of `Program::partially_evaluate`.
}

impl<C: Context<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>>
    for PlaceOperation
where
    C::Value: Placed + Value<Dispatch: PlaceDispatch<ArrayType, C::Value>>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        _context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        // The batching rule places the packed value, so the target and source shardings gain an entry for the batch
        // dimension. Every batching level records exactly one batch sharding in the placements of the batch items that
        // it reports (refer to `Placed`), so this level takes the innermost recorded batch sharding off the target and
        // the source, also for an unmapped input, which has no batch dimension to put it back into and keeps the target
        // and source otherwise unchanged. For a mapped input, a placement that records how its batch is distributed
        // puts that sharding back at the batch dimension. That makes the source that a batch item reports for itself
        // (e.g., the source that a placement inside a batched function infers from its input) the placement of the
        // packed input, including the split of the batch dimension over auto mesh axes that the packed input type does
        // not record, and it makes the transposed placement of such a source move the cotangent back there. Every
        // recorded batch sharding names only axes of the mesh of its placement, so it can always be put back.
        //
        // An explicit target that records nothing places every batch item where it says, so its batch dimension is
        // replicated (e.g., a replicated target replicates the packed value, also when it is split along mesh axes).
        // This deliberately differs from JAX, whose batching rule binds `device_put` on the packed value with the
        // unchanged target, so that its partition specification applies positionally to the packed dimensions (e.g.,
        // a target `P(x)` for a `[6]` batch item mapped at position 0 of a `[4, 6]` value splits the batch dimension,
        // whereas this rule splits the item dimension). An omitted target stays omitted, so that the packed value stays
        // where it is. A source that records nothing gains the placement that the packed input type records for its
        // batch dimension, so that it keeps describing where the packed input lives as far as its type can tell.
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        let mut target = self.target.clone();
        let mut source = self.source.clone();
        let target_batch_sharding = target.as_mut().and_then(Placement::pop_batch_sharding);
        let source_batch_sharding = source.as_mut().and_then(Placement::pop_batch_sharding);
        if let Some(batch_axis) = input.batch_axis_position() {
            if target.as_ref().and_then(Placement::layout).is_some() {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "`{PLACE_OPERATION_NAME}` cannot batch a target with an explicit layout, because the layout \
                         does not describe the batch dimension",
                    ),
                });
            }

            let batched = |placement: &Placement, batch_sharding: ShardingDimension| -> Result<_, BatchingError> {
                let sharding = placement.sharding().batched(batch_axis, batch_sharding)?;
                let mut result = Placement::new(placement.mesh().clone(), sharding).map_err(TypeError::from)?;
                if let Some(memory) = placement.memory() {
                    result = result.with_memory(memory);
                }
                if let Some(layout) = placement.layout() {
                    result = result.with_layout(layout.clone());
                }
                for enclosing in placement.batch_shardings() {
                    result = result.with_batch_sharding(enclosing.clone()).map_err(TypeError::from)?;
                }
                Ok(result)
            };

            if let Some(placement) = &target {
                target = Some(batched(placement, target_batch_sharding.unwrap_or(ShardingDimension::Replicated))?);
            }

            if let Some(placement) = &source {
                let batch_sharding = source_batch_sharding.unwrap_or_else(|| {
                    input
                        .r#type()
                        .sharding()
                        .map_or(ShardingDimension::Replicated, |sharding| sharding.dimensions()[batch_axis].clone())
                });
                source = Some(batched(placement, batch_sharding)?);
            }
        }

        // Placement changes preserve the packed shape and mapped batch axis, and the copy policy applies to the packed
        // value unchanged.
        let packed = input.value();
        let source = source.or_else(|| packed.placement());
        let output = <C::Value as Value>::Dispatch::place(packed, source, target.as_ref(), self.may_alias)?;
        Ok(vec![ArrayBatch::new(output, input.batch_axis())?])
    }
}

impl_differentiable_operation! {
    PlaceOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Value: Placed + Value<Dispatch: PlaceDispatch<ArrayType, C::Value>>,
        C::Operation: From<PlaceOperation>,
    {
        |operation, _context, _driver, inputs| {
            // A placement is linear, so the tangent is placed on the same devices, with the same sharding, in the same
            // memory, and with the same copy policy as the primal, and symbolic zero tangents stay symbolic without
            // allocation. The tangent can have another element type than the primal (e.g., an `f8e8m0fnu` primal has
            // an `f32` tangent), for which the layouts of the target and source can be invalid, so it is placed without
            // them. Derivative types carry no layout either. Without a source, the primal records where it concretely
            // lives as its source, if anywhere, and so does the tangent, because the tangent is placed by the same
            // linear map, whose transpose must move the cotangent back to where the primal lived.
            check_count!("input", inputs, 1, ProgramError);
            let may_alias = operation.may_alias();
            let input = inputs[0].primal();
            let source = operation.source().cloned().or_else(|| input.placement());
            let primal = <C::Value as Value>::Dispatch::place(input, source.clone(), operation.target(), may_alias)?;
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                MaybeZero::Value(tangent) => MaybeZero::Value(<C::Value as Value>::Dispatch::place(
                    tangent,
                    source.as_ref().map(placement_without_layout).or_else(|| tangent.placement()),
                    operation.target().map(placement_without_layout).as_ref(),
                    may_alias,
                )?),
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType> + From<BroadcastOperation> + From<PlaceOperation>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // A placement is the identity linear map between two placements, so its transpose moves the output
            // cotangent back to where the input lived, as JAX's `device_put` transpose does by swapping its devices
            // and sources. The cotangent is sharded like the cotangent dual of the input's sharding (i.e., with its
            // unreduced and reduced axes swapped) and lives in the input's memory. It carries no layout, since
            // derivative types carry none. Whatever the copy policy of the forward placement is, the transposed
            // placement forces a copy, as JAX's `device_put` transpose does, so that the input cotangent never shares
            // storage with the output cotangent that the caller provided. Symbolic zero cotangents propagate unchanged.
            // Every placement that the transposed placement derives from the target or the source keeps the batch
            // shardings that it records (refer to `Placement::batch_shardings`), so that the transposed placement of
            // a batch item inside a batched function still moves the cotangent back to how its batch was distributed.
            // A source that splits a static extent unevenly (e.g., an extent of 3 over 2 devices) cannot be the target
            // of the transposed placement, because type inference requires even splits, so transposition fails for it.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 1, DifferentiationError);

            let MaybeZero::Value(cotangent) = &outputs[0] else { return Ok(()) };
            let input_type = inputs[0].r#type();
            let input_cotangent_type = input_type.cotangent()?;
            let cotangent_placement = |placement: &Placement| -> Result<Placement, TypeError> {
                let mut result = Placement::new(placement.mesh().clone(), placement.sharding().cotangent()?)?;
                for enclosing in placement.batch_shardings() {
                    result = result.with_batch_sharding(enclosing.clone())?;
                }
                Ok(result)
            };

            let transposed = match (operation.source(), operation.target()) {
                (Some(source), Some(target)) => {
                    // With a known source, the cotangent moves back to the source mesh, and the old target
                    // becomes the new source.
                    let transposed_target = cotangent_placement(source)?.with_memory(input_type.memory());
                    let transposed_source = cotangent_placement(target)?;
                    PlaceOperation::new(Some(transposed_source), transposed_target)
                }
                (None, Some(target)) => {
                    // Without a source, the value is not known to have changed meshes, so the cotangent stays on the
                    // target mesh and gets the input's tracked sharding and memory back. This deliberately differs from
                    // JAX, whose transpose swaps the devices and sources of `device_put` and so leaves a cotangent with
                    // an unknown source where it is. Restoring the input's tracked sharding and memory keeps the
                    // cotangent well typed, and it is exact inside a compiled program, which runs on a single mesh.
                    // An input that is tracked over another mesh did change meshes, and its devices are not known.
                    let sharding = match input_cotangent_type.sharding() {
                        Some(sharding) if sharding.mesh().has_same_topology(target.mesh().logical_mesh()) => {
                            sharding.clone()
                        }
                        Some(_) => {
                            return Err(ProgramError::UnsupportedOperation {
                                message: format!(
                                    "cannot transpose `{PLACE_OPERATION_NAME}` across device meshes without a source \
                                     placement",
                                ),
                            }
                            .into());
                        }
                        None => target.sharding().cotangent().map_err(TypeError::from)?,
                    };
                    let mut placement = Placement::new(target.mesh().clone(), sharding).map_err(TypeError::from)?;
                    for enclosing in target.batch_shardings() {
                        placement = placement.with_batch_sharding(enclosing.clone()).map_err(TypeError::from)?;
                    }
                    PlaceOperation::new(None, placement.with_memory(input_type.memory()))
                }
                (Some(source), None) => {
                    // Without a target, the forward placement left the value where it was, which its source describes
                    // when it has one. The cotangent therefore moves to the source, as JAX's transpose does when it
                    // swaps the devices and sources of `device_put`, and it has no source of its own, because the
                    // forward placement did not say where the output lives.
                    PlaceOperation::new(None, cotangent_placement(source)?.with_memory(input_type.memory()))
                }
                (None, None) => {
                    // Without a target and a source, the forward placement is an identity up to its copy policy,
                    // and so is its transpose, which still forces a copy, as JAX's `device_put` transpose does.
                    PlaceOperation::new(None, None)
                },
            };

            let transposed = transposed.with_may_alias(false);
            let mut placed = context.stage_operation(transposed, Vec::new(), std::slice::from_ref(cotangent))?;
            check_count!("output", placed, 1, ProgramError);
            let placed = placed.remove(0);

            // An input without a tracked sharding receives its exact cotangent type through a placement-only
            // broadcast, as in `ReshardOperation` transposition.
            let contribution = if placed.r#type().as_ref() == &input_cotangent_type {
                placed
            } else {
                placed.broadcast(
                    input_cotangent_type.clone(),
                    &(0..input_cotangent_type.shape().rank()).collect::<Vec<_>>(),
                )?
            };

            accumulators[0].accumulate(context, MaybeZero::Value(contribution))?;

            Ok(())
        }
    },
}

impl<O: Operation<Type = ArrayType> + From<PlaceOperation>> OperationProvider<ArrayType, PlaceOperation> for O {
    type Operation = Self;

    #[inline]
    fn provide(request: PlaceOperation, input_types: &[&ArrayType]) -> Result<Self, ProgramError> {
        check_count!("input", input_types, 1, ProgramError);
        Ok(request.into())
    }
}

impl<O: Operation<Type = ArrayIrType> + OperationProjection<ArrayType, Projected: From<PlaceOperation>>>
    OperationProvider<ArrayIrType, PlaceOperation> for O
{
    type Operation = Self;

    #[inline]
    fn provide(request: PlaceOperation, input_types: &[&ArrayIrType]) -> Result<Self, ProgramError> {
        // Array IR families select the placement through their array projection, so that the provided operation
        // is the canonical array member (e.g., `ArrayIrOperation::Array(ArrayOperation::Place(_))`).
        check_count!("input", input_types, 1, ProgramError);
        let operation: <Self as OperationProjection<ArrayType>>::Projected = request.into();
        Ok(operation.into())
    }
}

/// Reports the [`Placement`] that a value concretely lives on. Devices and placement over [`MeshAxisType::Auto`] mesh
/// axes are runtime state that types do not record, so only the values that hold device buffers have a placement
/// (e.g., the arrays of a backend). Staged values have none, because where they live is decided when their program is
/// compiled and executed, and neither do reference [`Array`]s, which are host values. A value that wraps another value
/// (e.g., a [`PartialTracer`] or a [`DifferentiationTracer`]) reports the placement of the value that it wraps.
/// A batched value (i.e., an [`ArrayBatch`], an [`ArrayIrBatch`], or a [`BatchingTracer`] whose batch reports a
/// placement) reports the placement of the batch items that a batched function observes. That is the placement of its
/// packed value with the batch dimension removed from the sharding, because the placement of that dimension describes
/// how the batch is distributed rather than how each batch item is placed, and without a layout, because a layout of
/// the packed value does not describe a batch item. The placement of the removed dimension is recorded as the innermost
/// batch sharding of the reported placement instead (refer to [`Placement::batch_shardings`]), because a batch item
/// does not live on the whole mesh when its batch is split, and the batching rule of [`PlaceOperation`] puts it back.
/// A replicated value is its own batch item, so it reports the placement of its packed value with a replicated batch
/// recorded. Every batching level thus records exactly one batch sharding, so that nested batching levels each take
/// their own off. When a source is omitted, [`Place`] uses this trait on each leaf to record its concrete placement.
pub trait Placed {
    /// Returns the [`Placement`] that this value concretely lives on, or [`None`] if it has none.
    fn placement(&self) -> Option<Placement>;
}

impl Placed for Array {
    #[inline]
    fn placement(&self) -> Option<Placement> {
        // Reference arrays are host values that no device holds, so they have no placement even when their type records
        // a sharding (which then only describes how a backend would distribute the array).
        None
    }
}

impl<A: Value<Type = ArrayType> + Placed> Placed for ArrayIrValue<A> {
    #[inline]
    fn placement(&self) -> Option<Placement> {
        // Only array members live on devices. First-class dimensions and references have no placement.
        match self {
            Self::Array(value) => value.placement(),
            Self::Dimension(_) | Self::Reference(_) => None,
        }
    }
}

impl<T: Type, V: Placed> Placed for ProjectedValue<T, V> {
    #[inline]
    fn placement(&self) -> Option<Placement> {
        self.value().placement()
    }
}

impl<C: Context> Placed for Tracer<C> {
    #[inline]
    fn placement(&self) -> Option<Placement> {
        // A staged value lives wherever the program that computes it places it, which is decided only when that
        // program is compiled and executed.
        None
    }
}

impl<C: Context<Value: Placed>> Placed for PartialTracer<C> {
    #[inline]
    fn placement(&self) -> Option<Placement> {
        // A partial evaluation value lives where its known payload lives, while an unknown value has no placement yet.
        // A poisoned value reports none either, and its deferred error surfaces when the value is used.
        self.value().ok().and_then(|value| match value.value() {
            PartialValue::Known(known) => known.placement(),
            PartialValue::Unknown(_) => None,
        })
    }
}

impl<V: Value<Type = ArrayType> + Placed> Placed for ArrayBatch<V> {
    #[inline]
    fn placement(&self) -> Option<Placement> {
        unbatched_placement(self.value().placement()?, self.batch_axis_position())
    }
}

impl<V: Value<Type = ArrayIrType> + Placed> Placed for ArrayIrBatch<V> {
    #[inline]
    fn placement(&self) -> Option<Placement> {
        unbatched_placement(self.value().placement()?, self.batch_axis_position())
    }
}

impl<C: Context, P: BatchingPolicy<C, Batch: Placed>> Placed for BatchingTracer<C, P> {
    #[inline]
    fn placement(&self) -> Option<Placement> {
        // The policy-selected batch owns the batch dimension of the packed value, so it also owns how the placement
        // of a batch item follows from the placement of that packed value.
        self.batch().placement()
    }
}

impl<C: Context<Value: Placed>, P: DifferentiationPolicy<C>> Placed for DifferentiationTracer<C, P> {
    #[inline]
    fn placement(&self) -> Option<Placement> {
        // A differentiation tracer lives where its primal lives.
        self.primal().placement()
    }
}

/// Executes one [`PlaceOperation`] for a leaf through its [`Value::Dispatch`] policy. A domain-dispatched leaf binds
/// the operation in its own context. Direct backends can implement this trait on [`ValueDirectDispatch`] for their
/// values. [`Place`] supplies the leaf's inferred source when available, while this primitive uses the source that it
/// is given as is. Array IR values project the array member, place it, and lift the result back into their original
/// universe.
///
/// Unlike [`Place`], this primitive also takes an optional target and the copy policy of the placement (refer to
/// [`PlaceOperation::with_may_alias`]). Without a target, the value stays where it is (i.e., its devices, sharding,
/// memory, and layout are unchanged), which places a value with its source recorded or, with `may_alias` set to
/// `false`, gives it storage of its own. With `may_alias` set to `false`, implementations for concrete values must
/// return a value whose storage no other value shares, and implementations for staged values must stage the placement
/// with that policy, so that the copy happens when the staged placement executes on concrete values.
pub trait PlaceDispatch<T: Type, V: Value<Type = T>> {
    /// Records `source` as the expected placement of `input` and moves it to `target`. A [`None`] target leaves it
    /// where it is. The result may share storage with `input` only when `may_alias` is `true`.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::Type`] when [`PlaceOperation`] type inference rejects the target or the source for
    /// the type of `input` (e.g., when their ranks differ from that of `input` or the target changes its reduction
    /// state), and [`ProgramError::InvalidInputCount`] or [`ProgramError::InvalidOutputCount`] when a context binds
    /// the placement with an unexpected number of values. Array IR values return [`ProgramError::Type`] for a member
    /// that is not an array. Backends that execute the placement on concrete values return their own errors (e.g.,
    /// when a transfer to the devices of `target` fails), and staging contexts return the errors of staging it.
    fn place(
        input: &V,
        source: Option<Placement>,
        target: Option<&Placement>,
        may_alias: bool,
    ) -> Result<V, ProgramError>;
}

impl PlaceDispatch<ArrayType, Array> for ValueDirectDispatch {
    fn place(
        input: &Array,
        source: Option<Placement>,
        target: Option<&Placement>,
        may_alias: bool,
    ) -> Result<Array, ProgramError> {
        // A reference array is a host value without devices, so placing it keeps its elements unchanged. Its type still
        // records the placement, and the output type comes from the `PlaceOperation` type inference rule itself, so
        // that eager evaluation validates the target and source exactly like staged programs do. The layout determines
        // how the physical storage addresses the elements, so the storage is shared only when the layout is unchanged
        // and the copy policy allows it, and it is copied when that policy forbids sharing it. Otherwise, the logical
        // bytes are re-encoded for the output layout into new storage, which preserves the exact element bits
        // (including low-precision encodings and NaN payloads), as in `Cast` for `Array`. Without a target,
        // the output type is the input type, so the storage is shared or copied as the copy policy says.
        let operation = PlaceOperation::new(source, target.cloned()).with_may_alias(may_alias);
        let mut output_types = operation.infer_output_types(&[input.r#type().into_owned()], &[])?;
        check_count!("output", output_types, 1, ProgramError);
        let output_type = output_types.remove(0);
        if output_type.layout() != input.r#type().layout() {
            Array::from_logical_bytes(output_type, &input.logical_bytes())
        } else if may_alias {
            Ok(Array::new_unchecked(output_type, input.shared_storage_bytes().clone()))
        } else {
            Ok(Array::new_unchecked(output_type, Arc::new(input.storage_bytes().to_vec())))
        }
    }
}

impl<
    V: Value<
            Type = ArrayType,
            Dispatch = ValueDomainDispatch,
            Domain: Context<Type = ArrayType, Operation: From<PlaceOperation>>,
        >,
> PlaceDispatch<ArrayType, V> for ValueDomainDispatch
{
    #[inline]
    fn place(
        input: &V,
        source: Option<Placement>,
        target: Option<&Placement>,
        may_alias: bool,
    ) -> Result<V, ProgramError> {
        let outputs = input.domain().bind(
            PlaceOperation::new(source, target.cloned()).with_may_alias(may_alias),
            Vec::new(),
            std::slice::from_ref(input),
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.into_iter().next().unwrap())
    }
}

impl<
    V: Value<Type = ArrayIrType>
        + ValueProjection<ArrayType, Projected: Value<Type = ArrayType, Dispatch: PlaceDispatch<ArrayType, V::Projected>>>,
> PlaceDispatch<ArrayIrType, V> for V::Dispatch
{
    #[inline]
    fn place(
        input: &V,
        source: Option<Placement>,
        target: Option<&Placement>,
        may_alias: bool,
    ) -> Result<V, ProgramError> {
        let input = input.clone().into_projected()?;
        let output = <V::Projected as Value>::Dispatch::place(&input, source, target, may_alias)?;
        Ok(V::from_projected(output))
    }
}

/// Source argument accepted by [`Place::place`]. A borrowed tree of `Leaf` values has the same ordered
/// parameter paths as the receiver. `Option<&S>` supplies or omits a whole tree of [`Placement`]s, while
/// `&S` with `Option<Placement>` leaves lets each leaf omit its own source.
pub trait PlacementSources<P: Parameter, S: Parameterized<P>> {
    /// Returns the supplied source tree, or [`None`] to infer the source of every leaf.
    fn as_parameterized(&self) -> Option<&S>;
}

impl<S: Parameterized<Placement>> PlacementSources<Placement, S> for Option<&S> {
    #[inline]
    fn as_parameterized(&self) -> Option<&S> {
        *self
    }
}

impl<S: Parameterized<Option<Placement>>> PlacementSources<Option<Placement>, S> for &S {
    #[inline]
    fn as_parameterized(&self) -> Option<&S> {
        Some(self)
    }
}

/// Moves a value to a target [`Placement`], possibly on the devices of another [`DeviceMesh`](crate::DeviceMesh),
/// without changing its elements, shape, or data type. This is the Ryft analogue of
/// [`jax.device_put`](https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html), and like JAX's `device_put`
/// primitive it works inside transformations, so that a value can be moved within staged programs and functions that
/// are batched or differentiated. Its type records the target sharding (without its [`MeshAxisType::Auto`] mesh axes)
/// and the target [`Memory`](crate::Memory) and [`Layout`](crate::Layout) when the target names them, while an omitted
/// memory or layout keeps the value's current storage. Refer to [`PlaceOperation`] for the validation that the target
/// and source go through.
///
/// The source is informative. It states where the value is expected to live when the placement is created. Placement
/// over auto mesh axes and devices are runtime state that types do not record, so transposition uses the source to move
/// the cotangent back to those devices. [`PlaceOperation`] type inference checks the source against what the type of
/// the value tracks, while backends that execute the placement on a concrete value move the value from wherever it
/// actually lives without comparing that placement with the source. When the source is [`None`], each leaf uses the
/// placement that it concretely lives on (refer to [`Placed`]) if it has one, so that concrete backend arrays record
/// their actual placement, while staged values leave the source unknown. Reference [`Array`]s have no devices, so they
/// record the output type that [`PlaceOperation`] type inference computes without changing their elements, and they
/// re-encode their storage only for a target layout that differs from theirs. Unlike [`Reshard`](crate::Reshard), which
/// redistributes a value over the mesh that it already lives on, this capability can express placements across meshes.
/// Host scalar types have no placement and do not implement it.
///
/// Every placement through this capability uses the default copy policy of [`PlaceOperation`], so its results may share
/// storage with its inputs (e.g., a reference [`Array`] keeps sharing its storage unless its layout changes). To place
/// a leaf whose result must have storage of its own, call [`PlaceDispatch::place`] with `may_alias` set to `false`
/// instead, which also forces that copy when the placement is staged and later executes on concrete values.
/// [`PlaceDispatch::place`] also places a leaf without a target, which leaves it where it is.
///
/// A single value accepts an optional source followed by its target. A [`Parameterized`] receiver accepts placements
/// with the same ordered parameter paths and preserves its structure and static fields. The universe parameter `T`
/// defaults to the [`Capability`] universe of a single-value receiver, which is [`ArrayType`] for homogeneous arrays
/// and [`ArrayIrType`] for array IR values. The leaf parameter `P` defaults to `Self`, so single-value bounds
/// can use `Place<T>`. A structured receiver implements `Place<T, P>` for the universe `T` of its leaves.
///
/// # Examples
///
/// ```
/// # use ryft_core::{
/// #     Array, ArrayType, DataType, Device, DeviceMesh, LogicalMesh, Memory, MeshAxis, MeshAxisType, Place, Placement,
/// #     Sharding, ShardingDimension, Typed,
/// # };
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit)?])?;
/// let devices = DeviceMesh::new(mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)])?;
/// let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])])?;
/// let target = Placement::new(devices, sharding.clone())?.with_memory(Memory::Host { pinned: true });
/// let input = Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[1i32, 2])?;
/// let output = input.place(None, &target)?;
/// assert_eq!(output.r#type().sharding(), Some(&sharding));
/// assert_eq!(output.r#type().memory(), Memory::Host { pinned: true });
/// assert_eq!(output.elements::<i32>()?, vec![1, 2]);
///
/// let inputs = (input, output);
/// let targets = (target.clone(), target.clone());
/// let outputs = inputs.place(&(None, Some(target)), &targets)?;
/// assert_eq!(outputs.0, outputs.1);
/// assert_eq!(outputs.place(Some(&targets), &targets)?, outputs);
/// # Ok(())
/// # }
/// ```
pub trait Place<T = <Self as Capability>::Universe, P: Parameter + Capability = Self>: Sized {
    /// Places each leaf on its corresponding target. Pass `None` to infer all sources, `Some(&sources)` to supply
    /// every source, or a borrowed tree of `Option<Placement>` to infer just its `None` leaves. Each inferred source
    /// comes from the leaf's [`Placed::placement`], which can itself be unknown.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidArgument`] before placing any leaf if the ordered parameter paths of the
    /// receiver, targets, or supplied sources differ, and otherwise the first error that placing a leaf returns (refer
    /// to [`PlaceDispatch::place`]), such as a [`ProgramError::Type`] when [`PlaceOperation`] type inference rejects
    /// the target or source of that leaf.
    fn place<Source: Parameter>(
        &self,
        sources: impl PlacementSources<Source, <Self as Parameterized<P>>::To<Source>>,
        targets: &<Self as Parameterized<P>>::To<Placement>,
    ) -> Result<Self, ProgramError>
    where
        Self: Parameterized<P, Family: ParameterizedFamily<Placement> + ParameterizedFamily<Source>>,
        for<'s> &'s Source: Into<Option<&'s Placement>>;
}

impl<T, P: Value<Dispatch: PlaceDispatch<P::Type, P>> + Capability<Universe = T> + Placed, S: Parameterized<P>>
    Place<T, P> for S
{
    fn place<Source: Parameter>(
        &self,
        sources: impl PlacementSources<Source, S::To<Source>>,
        targets: &S::To<Placement>,
    ) -> Result<Self, ProgramError>
    where
        S::Family: ParameterizedFamily<Placement> + ParameterizedFamily<Source>,
        for<'s> &'s Source: Into<Option<&'s Placement>>,
    {
        let sources = sources.as_parameterized();
        if !self.parameter_paths().eq(targets.parameter_paths()) {
            return Err(ProgramError::InvalidArgument {
                message: "receiver and `targets` must have the same parameter structure".to_string(),
            });
        }
        if sources.is_some_and(|sources| !self.parameter_paths().eq(sources.parameter_paths())) {
            return Err(ProgramError::InvalidArgument {
                message: "receiver and `sources` must have the same parameter structure".to_string(),
            });
        }
        let mut sources = sources.into_iter().flat_map(|sources| sources.parameters());
        let outputs = self
            .parameters()
            .zip(targets.parameters())
            .map(|(input, target)| {
                let source = sources.next().and_then(Into::into).cloned().or_else(|| input.placement());
                P::Dispatch::place(input, source, Some(target), true)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(S::from_parameters(self.parameter_structure(), outputs)?)
    }
}

/// Returns the placement of the batch items of a packed value that lives on `placement` and whose batch dimension is
/// at position `batch_axis`, or [`None`] if the sharding of `placement` has no such dimension. The batch dimension is
/// removed from the sharding, because its placement describes how the batch is distributed rather than how each batch
/// item is placed (refer to [`ArrayType::unbatched`]), and so is the layout, which describes the storage of the packed
/// value. The placement of the removed dimension is recorded as the innermost batch sharding of the result instead
/// (refer to [`Placement::batch_shardings`]), because a batch item does not live on the whole mesh when its batch is
/// split, and the batching rule of [`PlaceOperation`] puts it back. A replicated value (i.e., one without a batch
/// dimension) is its own batch item, so `placement` is returned with a replicated batch recorded, which keeps one
/// recorded batch sharding per enclosing batching level, so that each level takes its own off.
fn unbatched_placement(placement: Placement, batch_axis: Option<usize>) -> Option<Placement> {
    let Some(batch_axis) = batch_axis else {
        return placement.with_batch_sharding(ShardingDimension::Replicated).ok();
    };
    let sharding = placement.sharding();
    if batch_axis >= sharding.rank() {
        return None;
    }
    let mut dimensions = sharding.dimensions().to_vec();
    let batch_sharding = dimensions.remove(batch_axis);
    let sharding = sharding.with_dimensions(dimensions).ok()?;
    let mut result = Placement::new(placement.mesh().clone(), sharding).ok()?;
    if let Some(memory) = placement.memory() {
        result = result.with_memory(memory);
    }
    for enclosing in placement.batch_shardings() {
        result = result.with_batch_sharding(enclosing.clone()).ok()?;
    }
    result.with_batch_sharding(batch_sharding).ok()
}

/// Returns `placement` without its layout, keeping its mesh, sharding, memory, and recorded batch shardings.
fn placement_without_layout(placement: &Placement) -> Placement {
    let mut result = Placement::new(placement.mesh().clone(), placement.sharding().clone()).unwrap();
    if let Some(memory) = placement.memory() {
        result = result.with_memory(memory);
    }

    // The batch shardings were recorded over the same sharding, so recording them again cannot fail.
    for enclosing in placement.batch_shardings() {
        result = result.with_batch_sharding(enclosing.clone()).unwrap();
    }
    result
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::cell::RefCell;
    use std::collections::{BTreeMap, HashSet};
    use std::rc::Rc;
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrOperation, ArrayOperation, DataType, Device, DeviceMesh, Dimension, DimensionBounds, DimensionValue,
        DimensionVariable, Layout, LogicalMesh, Memory, MeshAxis, Shape, StridedLayout, TiledLayout, UnreducedKind,
    };
    use crate::axes::{NamedAxes, NamedAxis};
    use crate::batching::{BatchAxis, batch};
    use crate::contexts::EagerContext;
    use crate::differentiation::{
        DifferentiableOperation, DifferentiationContext, DifferentiationError, TransposableOperation,
        TranspositionContext, differentiate_at,
    };
    use crate::macros::{check_operation_partial_evaluation, check_operation_type_inference};
    use crate::operations::constants::constant::Constant;
    use crate::operations::constants::fill::Fill;
    use crate::operations::constants::iota::Iota;
    use crate::operations::constants::one::One;
    use crate::operations::constants::zero::Zero;
    use crate::parameters::Placeholder;
    use crate::partial::PartialEvaluationContext;
    use crate::programs::{
        AttachedRegions, Concretizable, EffectClasses, EmptyRegionDriver, Provenance, ProvenanceScope,
    };
    use crate::tracing::TracingContext;

    use super::*;

    /// Memory that placement fixtures move values into.
    const PINNED_HOST: Memory = Memory::Host { pinned: true };

    /// Array that dispatches through [`RecordingContext`] and reports `placement` as where it concretely lives.
    #[derive(Clone, Debug)]
    struct PlacedArray {
        array: Array,
        placement: Option<Placement>,
        context: RecordingContext,
    }

    impl Display for PlacedArray {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "{}", self.array)
        }
    }

    impl Parameter for PlacedArray {}

    impl Typed for PlacedArray {
        type Type = ArrayType;

        fn r#type(&self) -> Cow<'_, ArrayType> {
            self.array.r#type()
        }
    }

    impl Value for PlacedArray {
        type Dispatch = ValueDomainDispatch;
        type Domain = RecordingContext;

        fn domain(&self) -> Self::Domain {
            self.context.clone()
        }
    }

    impl Placed for PlacedArray {
        fn placement(&self) -> Option<Placement> {
            self.placement.clone()
        }
    }

    /// Eager context that records the operations bound in it and interprets them over reference arrays. The outputs
    /// of a placement live on its target, or where its input lives when it has no target, and all other outputs have
    /// no placement.
    #[derive(Clone, Debug, Default)]
    struct RecordingContext(Rc<RefCell<Vec<ArrayOperation<Array>>>>);

    impl Domain for RecordingContext {
        type Type = ArrayType;
        type Value = PlacedArray;
        type Constant = Array;
        type Operation = ArrayOperation<Array>;
    }

    impl Context for RecordingContext {
        fn lift(&self, array: Array) -> Result<PlacedArray, ProgramError> {
            Ok(PlacedArray { array, placement: None, context: self.clone() })
        }

        fn bind<'r, O: Into<Self::Operation>, R: Into<AttachedRegions<'r, Self::Constant, Self::Operation>>>(
            &self,
            operation: O,
            regions: R,
            inputs: &[PlacedArray],
        ) -> Result<Vec<PlacedArray>, ProgramError>
        where
            Self::Constant: 'r,
            Self::Operation: 'r,
        {
            let operation = operation.into();
            let placement = match &operation {
                ArrayOperation::Place(place) => {
                    place.target().cloned().or_else(|| inputs.first().and_then(|input| input.placement.clone()))
                }
                _ => None,
            };
            let inputs = inputs.iter().map(|input| input.array.clone()).collect::<Vec<_>>();
            let outputs =
                EagerContext::<Array, ArrayOperation<Array>>::new().bind(operation.clone(), regions, &inputs)?;
            self.0.borrow_mut().push(operation);
            Ok(outputs
                .into_iter()
                .map(|array| PlacedArray { array, placement: placement.clone(), context: self.clone() })
                .collect())
        }

        fn is_eager(&self) -> bool {
            true
        }

        fn provenance(&self) -> Provenance {
            Provenance::unknown()
        }

        fn invoke_with_provenance_origin<R, F: FnOnce() -> R>(&self, _origin: Provenance, function: F) -> R {
            function()
        }

        fn invoke_with_provenance_scope<R, F: FnOnce() -> R>(&self, _scope: ProvenanceScope, function: F) -> R {
            function()
        }
    }

    impl NamedAxes for RecordingContext {
        fn named_axis(&self, _name: &str) -> Option<NamedAxis> {
            None
        }

        fn named_axes(&self) -> Vec<(String, NamedAxis)> {
            Vec::new()
        }
    }

    impl Zero<PlacedArray> for RecordingContext {
        fn zero(&self, r#type: &ArrayType) -> Result<PlacedArray, ProgramError> {
            self.lift(EagerContext::<Array>::new().zero(r#type)?)
        }
    }

    impl One<PlacedArray> for RecordingContext {
        fn one(&self, r#type: &ArrayType) -> Result<PlacedArray, ProgramError> {
            self.lift(EagerContext::<Array>::new().one(r#type)?)
        }
    }

    impl Iota<PlacedArray> for RecordingContext {
        fn iota(&self, r#type: &ArrayType, dimension: usize) -> Result<PlacedArray, ProgramError> {
            self.lift(EagerContext::<Array>::new().iota(r#type, dimension)?)
        }
    }

    impl Fill<u64, PlacedArray> for RecordingContext {
        fn fill(&self, r#type: &ArrayType, value: u64) -> Result<PlacedArray, ProgramError> {
            self.lift(EagerContext::<Array>::new().fill(r#type, value)?)
        }
    }

    impl Concretizable<bool> for PlacedArray {
        fn concretize(&self) -> Result<bool, ProgramError> {
            self.array.concretize()
        }
    }

    /// Returns a device mesh with the explicit axis `x` of size 2 over the devices 0 and 1.
    fn mesh_x() -> DeviceMesh {
        DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![Device::new(0, 0), Device::new(1, 0)],
        )
        .unwrap()
    }

    /// Returns a device mesh with the explicit axis `y` of size 4 over the devices 2 through 5, which has another
    /// topology than [`mesh_x`].
    fn mesh_y() -> DeviceMesh {
        DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("y", 4, MeshAxisType::Explicit).unwrap()]).unwrap(),
            (2..6).map(|id| Device::new(id, 0)).collect(),
        )
        .unwrap()
    }

    /// Returns a placement on `mesh` whose sharding over the logical mesh of `mesh` has the provided `dimensions`.
    fn placement(mesh: &DeviceMesh, dimensions: Vec<ShardingDimension>) -> Placement {
        Placement::new(mesh.clone(), Sharding::new(mesh.logical_mesh().clone(), dimensions).unwrap()).unwrap()
    }

    #[test]
    fn test_place_operation() {
        let source = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        let operation = PlaceOperation::new(Some(source.clone()), target.clone());
        assert_eq!(operation.name(), PLACE_OPERATION_NAME);
        assert_eq!(operation.source(), Some(&source));
        assert_eq!(operation.target(), Some(&target));
        assert!(operation.may_alias());
        assert_eq!(
            operation.to_string(),
            indoc! {"
                place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                    target={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], sharding={mesh<['y'=4:explicit]>, [{'y'}]}},
                ]"
            },
        );

        // An unknown source is not rendered.
        let operation = PlaceOperation::new(None, target);
        assert_eq!(operation.source(), None);
        assert_eq!(
            operation.to_string(),
            indoc! {"
                place [
                    target={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], sharding={mesh<['y'=4:explicit]>, [{'y'}]}},
                ]"
            },
        );

        // An omitted target leaves the input where it is and is not rendered either, so that a placement without a
        // target and a source renders as its name alone.
        let operation = PlaceOperation::new(Some(source.clone()), None);
        assert_eq!(operation.source(), Some(&source));
        assert_eq!(operation.target(), None);
        assert_eq!(
            operation.to_string(),
            indoc! {"
                place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                ]"
            },
        );
        let operation = PlaceOperation::new(None, None);
        assert_eq!(operation.source(), None);
        assert_eq!(operation.target(), None);
        assert_eq!(operation.to_string(), PLACE_OPERATION_NAME);
    }

    #[test]
    fn test_place_operation_with_may_alias() {
        let source = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        let aliasing = PlaceOperation::new(Some(source.clone()), target.clone());
        let copying = aliasing.clone().with_may_alias(false);
        assert!(!copying.may_alias());
        assert_eq!(copying.source(), Some(&source));
        assert_eq!(copying.target(), Some(&target));
        assert!(copying.clone().with_may_alias(true).may_alias());

        // The copy policy takes part in equality and hashing.
        assert_eq!(copying, copying.clone());
        assert_ne!(copying, aliasing);
        assert_eq!(copying.clone().with_may_alias(true), aliasing);
        let operations = HashSet::from([aliasing.clone(), copying.clone()]);
        assert_eq!(operations.len(), 2);
        assert!(operations.contains(&aliasing));
        assert!(operations.contains(&copying));

        // A forced copy is rendered after the placements, or alone when both are omitted.
        assert_eq!(
            copying.to_string(),
            indoc! {"
                place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                    target={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], sharding={mesh<['y'=4:explicit]>, [{'y'}]}},
                    may_alias=false,
                ]"
            },
        );
        assert_eq!(
            PlaceOperation::new(None, target).with_may_alias(false).to_string(),
            indoc! {"
                place [
                    target={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], sharding={mesh<['y'=4:explicit]>, [{'y'}]}},
                    may_alias=false,
                ]"
            },
        );
        assert_eq!(PlaceOperation::new(None, None).with_may_alias(false).to_string(), "place [may_alias=false]");
    }

    #[test]
    fn test_place_operation_type_inference() {
        let input_type = ArrayType::new_static(DataType::F32, [4]);
        let x_sharding =
            Sharding::new(mesh_x().logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let x_input_type = input_type.clone().with_sharding(x_sharding.clone()).unwrap();
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);

        // The output adopts the target sharding, also when the input is tracked on another mesh, and keeps its storage
        // unless the target names one.
        check_operation_type_inference!(
            operation = PlaceOperation::new(None, target.clone()),
            cases = [{
                input_types = [input_type.clone()],
                output_types = [input_type.clone().with_sharding(target.sharding().clone()).unwrap()],
            }, {
                input_types = [x_input_type.clone().with_memory(PINNED_HOST)],
                output_types = [input_type
                    .clone()
                    .with_sharding(target.sharding().clone())
                    .unwrap()
                    .with_memory(PINNED_HOST)],
            }, {
                input_types = [ArrayType::new_static(DataType::F32, [4, 2])],
                error = format!("`{PLACE_OPERATION_NAME}` target sharding rank (1) does not match the input rank (2)"),
            }, {
                input_types = [],
                error = "expected 1 input but got 0",
            }],
        );
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        check_operation_type_inference!(
            operation = PlaceOperation::new(None, target.clone().with_memory(PINNED_HOST).with_layout(layout.clone())),
            cases = [{
                input_types = [x_input_type.clone()],
                output_types = [input_type
                    .clone()
                    .with_sharding(target.sharding().clone())
                    .unwrap()
                    .with_memory(PINNED_HOST)
                    .with_layout(layout)],
            }],
        );

        // Types cannot record placement over auto axes, so the output drops them from the target sharding.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let auto_target = Placement::new(
            mesh_x(),
            Sharding::new(auto_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
        )
        .unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(None, auto_target),
            cases = [{
                input_types = [x_input_type.clone()],
                output_types = [input_type.clone().with_sharding(Sharding::replicated(auto_mesh.clone(), 1)).unwrap()],
            }],
        );

        // A target layout must address the output shape.
        let rank_3_layout = Layout::Tiled(TiledLayout::new(vec![0, 1, 2], vec![]));
        check_operation_type_inference!(
            operation = PlaceOperation::new(None, target.clone().with_layout(rank_3_layout)),
            cases = [{
                input_types = [input_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` target layout is invalid: tiled layout rank 3 does not match array \
                     rank 1",
                ),
            }],
        );

        // A source must agree with the tracked sharding of the input without its auto axes.
        let auto_source = Placement::new(
            mesh_x(),
            Sharding::new(auto_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
        )
        .unwrap();
        let replicated_source = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        check_operation_type_inference!(
            operation = PlaceOperation::new(Some(auto_source), target.clone()),
            cases = [{
                input_types = [input_type.clone().with_sharding(Sharding::replicated(auto_mesh.clone(), 1)).unwrap()],
                output_types = [input_type.clone().with_sharding(target.sharding().clone()).unwrap()],
            }],
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(Some(replicated_source.clone()), target.clone()),
            cases = [{
                input_types = [input_type.clone()],
                output_types = [input_type.clone().with_sharding(target.sharding().clone()).unwrap()],
            }, {
                input_types = [x_input_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` source sharding `{}` does not match the input type `{x_input_type}`",
                    replicated_source.sharding(),
                ),
            }],
        );
        let rank_2_source = placement(&mesh_x(), vec![ShardingDimension::Replicated; 2]);
        check_operation_type_inference!(
            operation = PlaceOperation::new(Some(rank_2_source), target.clone()),
            cases = [{
                input_types = [input_type.clone()],
                error = format!("`{PLACE_OPERATION_NAME}` source sharding rank (2) does not match the input rank (1)"),
            }],
        );

        // A source restates the reduction state of the input exactly, also when the input type tracks no sharding
        // (i.e., when the input has no reduction state), so that every accepted placement can be transposed.
        let unreduced_sharding =
            Sharding::replicated(mesh_x().logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let unreduced_source = Placement::new(mesh_x(), unreduced_sharding.clone()).unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(Some(unreduced_source.clone()), target.clone()),
            cases = [{
                input_types = [input_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` source sharding `{unreduced_sharding}` does not match the reduction \
                     state of the input type `{input_type}`",
                ),
            }],
        );
        let max_input_type = input_type
            .clone()
            .with_sharding(unreduced_sharding.clone().with_unreduced_kind(UnreducedKind::Max).unwrap())
            .unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(
                Some(unreduced_source),
                Placement::new(
                    mesh_x(),
                    unreduced_sharding.clone().with_unreduced_kind(UnreducedKind::Max).unwrap(),
                )
                .unwrap(),
            ),
            cases = [{
                input_types = [max_input_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` source sharding `{unreduced_sharding}` does not match the reduction \
                     state of the input type `{max_input_type}`",
                ),
            }],
        );
        let reduced_sharding =
            Sharding::replicated(mesh_x().logical_mesh().clone(), 1).with_reduced_axes(["x"]).unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(
                Some(replicated_source.clone()),
                Placement::new(mesh_x(), reduced_sharding.clone()).unwrap(),
            ),
            cases = [{
                input_types = [input_type.clone().with_sharding(reduced_sharding.clone()).unwrap()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` source sharding `{}` does not match the reduction state of the input \
                     type `{}`",
                    replicated_source.sharding(),
                    input_type.clone().with_sharding(reduced_sharding).unwrap(),
                ),
            }],
        );

        // Values inside manual regions, which vary along manual axes or carry reduction state over them, cannot be
        // moved, and nothing can be moved into such a region either.
        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying_sharding = Sharding::replicated(manual_mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap();
        let manual_error = format!(
            "`{PLACE_OPERATION_NAME}` cannot move values that vary along manual mesh axes or carry reduction state \
             over them",
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(None, target.clone()),
            cases = [{
                input_types = [input_type.clone().with_sharding(varying_sharding.clone()).unwrap()],
                error = manual_error.clone(),
            }, {
                input_types = [input_type
                    .clone()
                    .with_sharding(Sharding::replicated(manual_mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap())
                    .unwrap()],
                error = manual_error.clone(),
            }, {
                input_types = [input_type
                    .clone()
                    .with_sharding(Sharding::replicated(manual_mesh.clone(), 1).with_reduced_axes(["x"]).unwrap())
                    .unwrap()],
                error = manual_error.clone(),
            }],
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(None, Placement::new(mesh_x(), varying_sharding.clone()).unwrap()),
            cases = [{
                input_types = [input_type.clone()],
                error = manual_error.clone(),
            }],
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(Some(Placement::new(mesh_x(), varying_sharding).unwrap()), target.clone()),
            cases = [{
                input_types = [input_type.clone()],
                error = manual_error,
            }],
        );

        // A global array outside any manual region can be tiled over a manual axis of its mesh, and it can be moved
        // like any other array (e.g., onto a single device), also onto a mesh with manual axes.
        let single_device_mesh =
            LogicalMesh::new(vec![MeshAxis::new("d", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let single_device = Placement::new(
            DeviceMesh::new(single_device_mesh.clone(), vec![Device::new(0, 0)]).unwrap(),
            Sharding::replicated(single_device_mesh.clone(), 1),
        )
        .unwrap();
        let manual_tiled_sharding =
            Sharding::new(manual_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(
                Some(Placement::new(mesh_x(), manual_tiled_sharding.clone()).unwrap()),
                single_device,
            ),
            cases = [{
                input_types = [input_type.clone().with_sharding(manual_tiled_sharding.clone()).unwrap()],
                output_types = [input_type.clone().with_sharding(Sharding::replicated(single_device_mesh, 1)).unwrap()],
            }],
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(None, Placement::new(mesh_x(), manual_tiled_sharding.clone()).unwrap()),
            cases = [{
                input_types = [input_type.clone()],
                output_types = [input_type.clone().with_sharding(manual_tiled_sharding).unwrap()],
            }],
        );

        // A placement restates the reduction state of its input exactly.
        let unreduced_input_type = input_type
            .clone()
            .with_sharding(Sharding::replicated(mesh_x().logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let reduction_error = format!(
            "`{PLACE_OPERATION_NAME}` cannot change reduction state; use a reduction (or the corresponding collective) \
             instead",
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(
                None,
                Placement::new(
                    mesh_x(),
                    Sharding::replicated(mesh_x().logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap(),
                )
                .unwrap(),
            ),
            cases = [{
                input_types = [unreduced_input_type.clone()],
                output_types = [unreduced_input_type.clone()],
            }, {
                input_types = [x_input_type],
                error = reduction_error,
            }],
        );

        // Reduction state is restated by axis name, so the target and source meshes must define the axes that carry
        // it like the input's mesh does. Otherwise, the unreduced input would silently lose its pending reduction over
        // an auto axis `x` (over which types record no reduction state), or change it over an axis of another size or
        // over an axis that the target mesh does not define.
        let explicit_axis = MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(None, target.clone()),
            cases = [{
                input_types = [unreduced_input_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` cannot redefine preserved explicit mesh axis `{explicit_axis}` as an \
                     undefined axis",
                ),
            }],
        );
        let auto_axis = MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap();
        let wide_axis = MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap();
        let wide_mesh = LogicalMesh::new(vec![wide_axis.clone()]).unwrap();
        let wide_devices = DeviceMesh::new(wide_mesh.clone(), (2..6).map(|id| Device::new(id, 0)).collect()).unwrap();
        let auto_unreduced =
            Placement::new(mesh_x(), Sharding::replicated(auto_mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap())
                .unwrap();
        let wide_unreduced =
            Placement::new(wide_devices, Sharding::replicated(wide_mesh, 1).with_unreduced_axes(["x"]).unwrap())
                .unwrap();
        let unreduced_placement = Placement::new(mesh_x(), unreduced_input_type.sharding().unwrap().clone()).unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(None, auto_unreduced.clone()),
            cases = [{
                input_types = [unreduced_input_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` cannot redefine preserved explicit mesh axis `{explicit_axis}` as \
                     `{auto_axis}`",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(None, wide_unreduced.clone()),
            cases = [{
                input_types = [unreduced_input_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` cannot redefine preserved explicit mesh axis `{explicit_axis}` as \
                     `{wide_axis}`",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(Some(auto_unreduced), unreduced_placement.clone()),
            cases = [{
                input_types = [unreduced_input_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` cannot redefine preserved explicit mesh axis `{explicit_axis}` as \
                     `{auto_axis}`",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(Some(wide_unreduced), unreduced_placement.clone()),
            cases = [{
                input_types = [unreduced_input_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` cannot redefine preserved explicit mesh axis `{explicit_axis}` as \
                     `{wide_axis}`",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(Some(unreduced_placement.clone()), unreduced_placement),
            cases = [{
                input_types = [unreduced_input_type.clone()],
                output_types = [unreduced_input_type],
            }],
        );

        // The operation cannot own nested regions.
        assert_eq!(
            PlaceOperation::new(None, target).infer_output_types(
                std::slice::from_ref(&input_type),
                &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_place_operation_type_inference_without_target() {
        // Without a target, the value stays where it is, so the output keeps the sharding, memory, and layout of the
        // input, also when the input varies along a manual mesh axis or carries reduction state.
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let input_type = ArrayType::new_static(DataType::F32, [4]);
        let x_sharding =
            Sharding::new(mesh_x().logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let pinned_input_type =
            input_type.clone().with_sharding(x_sharding).unwrap().with_memory(PINNED_HOST).with_layout(layout);
        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying_input_type = input_type
            .clone()
            .with_sharding(Sharding::replicated(manual_mesh, 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let unreduced_input_type = input_type
            .clone()
            .with_sharding(Sharding::replicated(mesh_x().logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(None, None).with_may_alias(false),
            cases = [{
                input_types = [input_type.clone()],
                output_types = [input_type.clone()],
            }, {
                input_types = [pinned_input_type.clone()],
                output_types = [pinned_input_type.clone()],
            }, {
                input_types = [varying_input_type.clone()],
                output_types = [varying_input_type],
            }, {
                input_types = [unreduced_input_type.clone()],
                output_types = [unreduced_input_type],
            }, {
                input_types = [],
                error = "expected 1 input but got 0",
            }],
        );

        // A given source is still validated against the input type.
        let source = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let replicated_input_type =
            input_type.clone().with_sharding(Sharding::replicated(mesh_x().logical_mesh().clone(), 1)).unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(Some(source.clone()), None),
            cases = [{
                input_types = [pinned_input_type.clone()],
                output_types = [pinned_input_type],
            }, {
                input_types = [input_type.clone()],
                output_types = [input_type],
            }, {
                input_types = [replicated_input_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` source sharding `{}` does not match the input type \
                     `{replicated_input_type}`",
                    source.sharding(),
                ),
            }, {
                input_types = [ArrayType::new_static(DataType::F32, [4, 2])],
                error = format!("`{PLACE_OPERATION_NAME}` source sharding rank (1) does not match the input rank (2)"),
            }],
        );
    }

    #[test]
    fn test_place_operation_type_inference_shapes() {
        // Rank-0 arrays are placed with rank-0 shardings, which can only replicate them.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let scalar_target = placement(&mesh_y(), vec![]);
        check_operation_type_inference!(
            operation = PlaceOperation::new(Some(placement(&mesh_x(), vec![])), scalar_target.clone()),
            cases = [{
                input_types = [scalar_type.clone()],
                output_types = [scalar_type.clone().with_sharding(scalar_target.sharding().clone()).unwrap()],
            }],
        );

        // Placement needs only the rank of the input, so a dynamic extent keeps its variable and gets the target
        // sharding like a static one.
        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                "n",
                DimensionBounds::new(2, Some(8)).unwrap(),
            ))]),
        );
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        check_operation_type_inference!(
            operation = PlaceOperation::new(None, target.clone()),
            cases = [{
                input_types = [dynamic_type.clone()],
                output_types = [dynamic_type.clone().with_sharding(target.sharding().clone()).unwrap()],
            }],
        );
    }

    #[test]
    fn test_place_operation_type_inference_uneven_splits() {
        // A target can only split a static extent into shards of equal size, also along auto axes, which the output
        // type does not record, and along every mesh axis that shards the dimension together.
        let mixed_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("a", 3, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let mixed_devices = DeviceMesh::new(mixed_mesh.clone(), (0..6).map(|id| Device::new(id, 0)).collect()).unwrap();
        let target = Placement::new(
            mixed_devices,
            Sharding::new(
                mixed_mesh.clone(),
                vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x", "a"])],
            )
            .unwrap(),
        )
        .unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(None, target),
            cases = [{
                input_types = [ArrayType::new_static(DataType::I32, [5, 12])],
                output_types = [ArrayType::new_static(DataType::I32, [5, 12])
                    .with_sharding(
                        Sharding::new(
                            mixed_mesh,
                            vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])],
                        )
                        .unwrap(),
                    )
                    .unwrap()],
            }, {
                input_types = [ArrayType::new_static(DataType::I32, [5, 4])],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` target sharding splits dimension #1 with extent 4 into 6 shards, which \
                     do not divide it evenly",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(None, placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])])),
            cases = [{
                input_types = [ArrayType::new_static(DataType::I32, [5])],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` target sharding splits dimension #0 with extent 5 into 2 shards, which \
                     do not divide it evenly",
                ),
            }],
        );
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(
                None,
                Placement::new(mesh_x(), Sharding::new(auto_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
                    .unwrap(),
            ),
            cases = [{
                input_types = [ArrayType::new_static(DataType::I32, [5])],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` target sharding splits dimension #0 with extent 5 into 2 shards, which \
                     do not divide it evenly",
                ),
            }],
        );

        // A dynamic extent is only known when a backend places a concrete value, so type inference accepts any split
        // of it, also one that its bounds do not admit evenly.
        let dynamic_type = ArrayType::new(
            DataType::I32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                "n",
                DimensionBounds::new(3, Some(5)).unwrap(),
            ))]),
        );
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        check_operation_type_inference!(
            operation = PlaceOperation::new(None, target.clone()),
            cases = [{
                input_types = [dynamic_type.clone()],
                output_types = [dynamic_type.with_sharding(target.sharding().clone()).unwrap()],
            }],
        );
    }

    #[test]
    fn test_place_operation_interpretation() {
        // Reference arrays have no devices, so interpretation keeps the payload unchanged while recording the
        // placement in the value's type so that it matches the declared output type.
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]).with_memory(PINNED_HOST);
        let operation = PlaceOperation::new(None, target.clone());
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let outputs = operation
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(
            outputs,
            vec![
                Array::from_elements(
                    ArrayType::new_static(DataType::F32, [2])
                        .with_sharding(target.sharding().clone())
                        .unwrap()
                        .with_memory(PINNED_HOST),
                    &[1.0f32, 2.0],
                )
                .unwrap(),
            ],
        );
        assert!(Arc::ptr_eq(input.shared_storage_bytes(), outputs[0].shared_storage_bytes()));
        assert_eq!(
            operation.interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // A forced copy gives the output storage of its own with the same elements.
        let copies = operation
            .clone()
            .with_may_alias(false)
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(copies, outputs);
        assert!(!Arc::ptr_eq(input.shared_storage_bytes(), copies[0].shared_storage_bytes()));

        // Rank-0 arrays are placed like any other array.
        let scalar_target = placement(&mesh_x(), vec![]);
        let scalar = Array::scalar(3.0f32).unwrap();
        assert_eq!(
            PlaceOperation::new(None, scalar_target.clone()).interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                std::slice::from_ref(&scalar),
            ),
            Ok(vec![
                Array::from_elements(
                    ArrayType::scalar(DataType::F32).with_sharding(scalar_target.sharding().clone()).unwrap(),
                    &[3.0f32],
                )
                .unwrap(),
            ]),
        );

        // Interpreting the operation in a staging context stages it, also for an input with a dynamic extent.
        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                "n",
                DimensionBounds::new(2, Some(8)).unwrap(),
            ))]),
        );
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_context_and_named_axes(
            |context, input| Ok(operation.interpret(context, &EmptyRegionDriver, &[input])?.remove(0)),
            dynamic_type.clone(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            output_type,
            dynamic_type.with_sharding(target.sharding().clone()).unwrap().with_memory(PINNED_HOST),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[n] .
                let %1:f32[n][sharding={mesh<['x'=2:explicit]>, [{'x'}]}]@Host[Pinned] = place [
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{'x'}]}, memory=Host[Pinned]},
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_interpretation_without_target() {
        // Without a target, a reference array keeps its type and shares its storage, unless the copy policy forbids
        // sharing it.
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let outputs = PlaceOperation::new(None, None)
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(outputs, vec![input.clone()]);
        assert!(Arc::ptr_eq(input.shared_storage_bytes(), outputs[0].shared_storage_bytes()));

        // A placement without a target and a source that forces a copy is still staged, so that replaying the program
        // on concrete values copies them.
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input| <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&input, None, None, false),
            input.r#type().into_owned(),
        )
        .unwrap();
        assert_eq!(output_type, input.r#type().into_owned());
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = place [may_alias=false] %0
                in (%1)"
            },
        );
        let output = program.interpret(input.clone()).unwrap();
        assert_eq!(output, input);
        assert!(!Arc::ptr_eq(input.shared_storage_bytes(), output.shared_storage_bytes()));
    }

    #[test]
    fn test_place_operation_partial_evaluation() {
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
        check_operation_partial_evaluation!(
            operation = PlaceOperation::new(None, target.clone()),
            inputs = [input.clone()],
            expected = input.place(None, &target).unwrap(),
        );
        check_operation_partial_evaluation!(
            operation = PlaceOperation::new(None, None).with_may_alias(false),
            inputs = [input.clone()],
            expected = input,
        );

        // Known inputs fold through the reference interpreter, which re-encodes them for a target layout.
        let input = Array::from_logical_bytes(ArrayType::new_static(DataType::U8, [2]), &[10, 20]).unwrap();
        let layout = Layout::Strided(StridedLayout::new(vec![-1]));
        check_operation_partial_evaluation!(
            operation = PlaceOperation::new(None, target.clone().with_layout(layout.clone())),
            inputs = [input],
            expected = Array::new(
                ArrayType::new_static(DataType::U8, [2])
                    .with_sharding(target.sharding().clone())
                    .unwrap()
                    .with_layout(layout),
                vec![20, 10],
            )
            .unwrap(),
        );
    }

    #[test]
    fn test_place_operation_batching() {
        // The batch item is placed with rank-1 shardings, so the lifted placement extends the target sharding with
        // a replicated batch dimension and the source sharding with the placement that the packed input type records
        // for its batch dimension. The target therefore splits the item dimension of the packed value and not its
        // batch dimension, unlike JAX, which applies the target to the packed value positionally.
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        let source = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x| {
                let (target, source) = (target.clone(), source.clone());
                Ok(batch(
                    move |item| item.place(Some(&source), &target),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )?)
            },
            ArrayType::new_static(DataType::F32, [2, 4]),
        )
        .unwrap();
        assert_eq!(
            output_type,
            ArrayType::new_static(DataType::F32, [2, 4])
                .with_sharding(target.sharding().with_inserted_dimension(0, ShardingDimension::Replicated).unwrap())
                .unwrap(),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 4] .
                let %1:f32[2, 4][sharding={mesh<['y'=4:explicit]>, [{}, {'y'}]}] = place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{}, {}]}},
                    target={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], \
                        sharding={mesh<['y'=4:explicit]>, [{}, {'y'}]}},
                ] %0
                in (%1)"
            },
        );

        // The target applies to each batch item, so it splits the item dimension wherever the batch dimension is, and
        // the batch dimension stays replicated also when it comes after the item dimension.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x| {
                let (target, source) = (target.clone(), source.clone());
                Ok(batch(
                    move |item| item.place(Some(&source), &target),
                    x,
                    BatchAxis::new(1),
                    BatchAxis::new(1),
                    None,
                )?)
            },
            ArrayType::new_static(DataType::F32, [4, 2]),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4, 2] .
                let %1:f32[4, 2][sharding={mesh<['y'=4:explicit]>, [{'y'}, {}]}] = place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{}, {}]}},
                    target={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], \
                        sharding={mesh<['y'=4:explicit]>, [{'y'}, {}]}},
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_batching_may_alias() {
        // The packed value is placed with the copy policy of the batch item placement.
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x| {
                Ok(batch(
                    move |item| {
                        <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&item, None, Some(&target), false)
                    },
                    x,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )?)
            },
            ArrayType::new_static(DataType::F32, [2, 4]),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 4] .
                let %1:f32[2, 4][sharding={mesh<['y'=4:explicit]>, [{}, {'y'}]}] = place [
                    target={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], \
                        sharding={mesh<['y'=4:explicit]>, [{}, {'y'}]}},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_batching_axis_sharding() {
        // An explicit target that records no batch sharding places every batch item where it says, so the batch
        // dimension of the packed value is replicated, as in JAX, whatever the placement of the batch dimension that
        // the batching level selects is. A target that records how its batch is distributed (e.g., the placement of
        // another batch item) puts that distribution back instead.
        let target = placement(&mesh_y(), vec![ShardingDimension::Replicated]);
        let input =
            ArrayBatch::new(Array::matrix(4, 1, vec![1.0f32, 2.0, 3.0, 4.0]).unwrap(), BatchAxis::new(0)).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array>::new(), 4)
            .with_axis_sharding(ShardingDimension::sharded(["y"]));
        let outputs = PlaceOperation::new(None, target.clone())
            .batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis_position(), Some(0));
        assert_eq!(
            outputs[0].value(),
            &Array::from_elements(
                ArrayType::new_static(DataType::F32, [4, 1])
                    .with_sharding(Sharding::replicated(mesh_y().logical_mesh().clone(), 2))
                    .unwrap(),
                &[1.0f32, 2.0, 3.0, 4.0],
            )
            .unwrap(),
        );

        // A target that records the batch sharding `y` splits the batch dimension along `y`, whatever the batching
        // level selects.
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array>::new(), 4)
            .with_axis_sharding(ShardingDimension::sharded(["x"]));
        let split_target = target.with_batch_sharding(ShardingDimension::sharded(["y"])).unwrap();
        let outputs = PlaceOperation::new(None, split_target)
            .batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis_position(), Some(0));
        assert_eq!(
            outputs[0].value(),
            &Array::from_elements(
                ArrayType::new_static(DataType::F32, [4, 1])
                    .with_sharding(
                        Sharding::new(
                            mesh_y().logical_mesh().clone(),
                            vec![ShardingDimension::sharded(["y"]), ShardingDimension::Replicated],
                        )
                        .unwrap(),
                    )
                    .unwrap(),
                &[1.0f32, 2.0, 3.0, 4.0],
            )
            .unwrap(),
        );
    }

    #[test]
    fn test_place_operation_batching_sharded_source() {
        // The packed input is sharded along `x` in its batch dimension, so the source of the lifted placement records
        // that placement for the batch dimension and keeps describing where the packed input lives.
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        let source = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        let input_type = ArrayType::new_static(DataType::F32, [2, 4])
            .with_sharding(
                Sharding::new(
                    mesh_x().logical_mesh().clone(),
                    vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated],
                )
                .unwrap(),
            )
            .unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x| {
                Ok(batch(
                    move |item| item.place(Some(&source), &target),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )?)
            },
            input_type,
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 4][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}] .
                let %1:f32[2, 4][sharding={mesh<['y'=4:explicit]>, [{}, {'y'}]}] = place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}},
                    target={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], \
                        sharding={mesh<['y'=4:explicit]>, [{}, {'y'}]}},
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_batching_identity() {
        // The packed input is split along the auto axis `x` in its batch dimension, which its type does not record. A
        // batched function that places each batch item without a target leaves the packed value where it is, so the
        // lifted placement has no target either and keeps that split, while its source is the placement of the packed
        // input.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let packed = Placement::new(
            mesh_x(),
            Sharding::new(auto_mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated])
                .unwrap(),
        )
        .unwrap();
        let array = Array::from_elements(
            ArrayType::new_static(DataType::F32, [2, 2])
                .with_sharding(Sharding::replicated(auto_mesh, 2))
                .unwrap(),
            &[1.0f32, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let context = RecordingContext::default();
        let input = PlacedArray { array: array.clone(), placement: Some(packed.clone()), context: context.clone() };
        let output = batch(
            |item| <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&item, None, None, true),
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(
            context.0.borrow().as_slice(),
            &[ArrayOperation::Place(PlaceOperation::new(Some(packed.clone()), None))],
        );
        assert_eq!(output.array, array);
        assert_eq!(output.placement, Some(packed));
    }

    #[test]
    fn test_place_operation_batching_replicated_target() {
        // The packed vector is split along the auto axis `x`, which its type does not record. An explicit replicated
        // target for each batch item is not a request to leave it where it is, so the lifted target replicates the
        // batch dimension, and the packed result is replicated on both devices, as in JAX. Omitting the target instead
        // keeps the split.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let packed = Placement::new(
            mesh_x(),
            Sharding::new(auto_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
        )
        .unwrap();
        let array = Array::from_elements(
            ArrayType::new_static(DataType::F32, [4])
                .with_sharding(Sharding::replicated(auto_mesh.clone(), 1))
                .unwrap(),
            &[1.0f32, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let context = RecordingContext::default();
        let input = PlacedArray { array: array.clone(), placement: Some(packed.clone()), context: context.clone() };
        let target = Placement::new(mesh_x(), Sharding::replicated(auto_mesh.clone(), 0)).unwrap();
        let replicated =
            batch(|item| item.place(None, &target), input.clone(), BatchAxis::new(0), BatchAxis::new(0), None).unwrap();
        let kept = batch(
            |item| <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&item, None, None, true),
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        let replicated_placement = Placement::new(mesh_x(), Sharding::replicated(auto_mesh, 1)).unwrap();
        assert_eq!(
            context.0.borrow().as_slice(),
            &[
                ArrayOperation::Place(PlaceOperation::new(Some(packed.clone()), replicated_placement.clone())),
                ArrayOperation::Place(PlaceOperation::new(Some(packed.clone()), None)),
            ],
        );
        assert_eq!(replicated.array, array);
        assert_eq!(replicated.placement, Some(replicated_placement));
        assert_eq!(kept.array, array);
        assert_eq!(kept.placement, Some(packed));

        // The same holds when the vector is split along the explicit axis `x`, which its type records, so that the
        // batching level is also split along `x`. The explicit replicated target still replicates the packed result.
        let packed = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let array = Array::from_elements(
            ArrayType::new_static(DataType::F32, [4]).with_sharding(packed.sharding().clone()).unwrap(),
            &[1.0f32, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let context = RecordingContext::default();
        let input = PlacedArray { array: array.clone(), placement: Some(packed.clone()), context: context.clone() };
        let target = placement(&mesh_x(), vec![]);
        let replicated =
            batch(|item| item.place(None, &target), input.clone(), BatchAxis::new(0), BatchAxis::new(0), None).unwrap();
        let kept = batch(
            |item| <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&item, None, None, true),
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        let replicated_placement = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        assert_eq!(
            context.0.borrow().as_slice(),
            &[
                ArrayOperation::Place(PlaceOperation::new(Some(packed.clone()), replicated_placement.clone())),
                ArrayOperation::Place(PlaceOperation::new(Some(packed.clone()), None)),
            ],
        );
        assert_eq!(
            replicated.array,
            Array::from_elements(
                ArrayType::new_static(DataType::F32, [4])
                    .with_sharding(replicated_placement.sharding().clone())
                    .unwrap(),
                &[1.0f32, 2.0, 3.0, 4.0],
            )
            .unwrap(),
        );
        assert_eq!(replicated.placement, Some(replicated_placement));
        assert_eq!(kept.array, array);
        assert_eq!(kept.placement, Some(packed));
    }

    #[test]
    fn test_place_operation_batching_unmapped_input() {
        // An unmapped input has no batch dimension at this batching level, so the batching rule drops the batch
        // sharding that the target records for this level instead of putting it back, and the source that the input
        // reports for itself (i.e., its packed placement with a replicated batch recorded) loses that record too.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let mapped_placement = Placement::new(
            mesh_x(),
            Sharding::new(auto_mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated])
                .unwrap(),
        )
        .unwrap();
        let unmapped_placement = Placement::new(mesh_x(), Sharding::replicated(auto_mesh.clone(), 1)).unwrap();
        let context = RecordingContext::default();
        let mapped = PlacedArray {
            array: Array::from_elements(
                ArrayType::new_static(DataType::F32, [2, 2])
                    .with_sharding(Sharding::replicated(auto_mesh.clone(), 2))
                    .unwrap(),
                &[1.0f32, 2.0, 3.0, 4.0],
            )
            .unwrap(),
            placement: Some(mapped_placement),
            context: context.clone(),
        };
        let unmapped = PlacedArray {
            array: Array::from_elements(
                ArrayType::new_static(DataType::F32, [2]).with_sharding(Sharding::replicated(auto_mesh, 1)).unwrap(),
                &[5.0f32, 6.0],
            )
            .unwrap(),
            placement: Some(unmapped_placement.clone()),
            context: context.clone(),
        };
        let output = batch(
            |(mapped, unmapped)| unmapped.place(None, &mapped.placement().unwrap()),
            (mapped, unmapped),
            (BatchAxis::new(0), BatchAxis::replicated()),
            BatchAxis::replicated(),
            None,
        )
        .unwrap();
        assert_eq!(
            context.0.borrow().as_slice(),
            &[ArrayOperation::Place(PlaceOperation::new(Some(unmapped_placement.clone()), unmapped_placement.clone()))],
        );
        assert_eq!(output.placement, Some(unmapped_placement));
    }

    #[test]
    fn test_place_operation_batching_layouts() {
        // A layout describes every dimension of the placed value, so a mapped input cannot be placed with one, whereas
        // a replicated input keeps its rank and is placed with the unchanged target.
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]).with_layout(layout.clone());
        let operation = PlaceOperation::new(None, target.clone());
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array>::new(), 2);
        let mapped =
            ArrayBatch::new(Array::matrix(2, 2, vec![1.0f32, 2.0, 3.0, 4.0]).unwrap(), BatchAxis::new(0)).unwrap();
        assert_eq!(
            operation.batch(&context, &EmptyRegionDriver, &[mapped]),
            Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{PLACE_OPERATION_NAME}` cannot batch a target with an explicit layout, because the layout does \
                     not describe the batch dimension",
                ),
            }),
        );
        let array = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let replicated = ArrayBatch::new(array.clone(), BatchAxis::replicated()).unwrap();
        let outputs = operation.batch(&context, &EmptyRegionDriver, &[replicated]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis_position(), None);
        assert_eq!(outputs[0].value(), &array.place(None, &target).unwrap());
        assert_eq!(outputs[0].r#type().layout(), Some(&layout));
    }

    #[test]
    fn test_place_operation_differentiation() {
        // A placement is linear, so the tangent is placed with the same target as the primal.
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]).with_memory(PINNED_HOST);
        let (primal, tangent) = differentiate_at(Array::vector(vec![1.0f32, 2.0]).unwrap())
            .jvp(Array::vector(vec![3.0f32, 4.0]).unwrap(), |x| x.place(None, &target))
            .unwrap();
        let expected_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(target.sharding().clone())
            .unwrap()
            .with_memory(PINNED_HOST);
        assert_eq!(primal, Array::from_elements(expected_type.clone(), &[1.0f32, 2.0]).unwrap());
        assert_eq!(tangent, Array::from_elements(expected_type.clone(), &[3.0f32, 4.0]).unwrap());

        // Symbolic zero tangents keep the type of the placed primal without staging a placement of their own.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let primal = context.input(ArrayType::new_static(DataType::F32, [2]));
        let duals = PlaceOperation::new(None, target)
            .jvp(
                &DifferentiationContext::fused(context.clone()),
                &EmptyRegionDriver,
                &[DifferentiationDual::new_with_zero_tangent(primal).unwrap()],
            )
            .unwrap();
        assert!(duals[0].tangent().is_zero());
        assert_eq!(duals[0].tangent().r#type().as_ref(), &expected_type);
        assert_eq!(context.builder().borrow().instructions().len(), 1);
    }

    #[test]
    fn test_place_operation_differentiation_may_alias() {
        // The primal and the tangent are both placed with the copy policy of the primal placement.
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(x, t)| {
                Ok(differentiate_at(x).jvp(t, |x| {
                    <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&x, None, Some(&target), false)
                })?)
            },
            (ArrayType::new_static(DataType::F32, [4]), ArrayType::new_static(DataType::F32, [4])),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4], %1:f32[4] .
                let %2:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = place [
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                    may_alias=false,
                ] %0
                    %3:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = place [
                        target={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                        may_alias=false,
                    ] %1
                in (%2, %3)"
            },
        );
    }

    #[test]
    fn test_place_operation_differentiation_layouts() {
        // A one-byte stride is valid for the one-byte elements of an `f8e8m0fnu` primal, but its tangent has `f32`
        // elements, which that stride would overlap, so the tangent is placed without the target layout.
        let target = placement(&mesh_x(), vec![ShardingDimension::Replicated])
            .with_layout(Layout::Strided(StridedLayout::new(vec![1])));
        let primal = Array::from_logical_bytes(ArrayType::new_static(DataType::F8E8M0FNU, [2]), &[127, 128]).unwrap();
        let (primal, tangent) = differentiate_at(primal)
            .jvp(Array::vector(vec![3.0f32, 4.0]).unwrap(), |x| x.place(None, &target))
            .unwrap();
        assert_eq!(
            primal,
            Array::from_logical_bytes(
                ArrayType::new_static(DataType::F8E8M0FNU, [2])
                    .with_sharding(target.sharding().clone())
                    .unwrap()
                    .with_layout(Layout::Strided(StridedLayout::new(vec![1]))),
                &[127, 128],
            )
            .unwrap(),
        );
        assert_eq!(
            tangent,
            Array::from_elements(
                ArrayType::new_static(DataType::F32, [2]).with_sharding(target.sharding().clone()).unwrap(),
                &[3.0f32, 4.0],
            )
            .unwrap(),
        );
    }

    #[test]
    fn test_place_operation_differentiation_batching() {
        // Batching a differentiated placement places the packed primal and the packed tangent with the same lifted
        // target and source.
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        let source = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(x, t)| {
                Ok(batch(
                    move |(item, tangent)| {
                        Ok(differentiate_at(item).jvp(tangent, |item| item.place(Some(&source), &target))?)
                    },
                    (x, t),
                    (BatchAxis::new(0), BatchAxis::new(0)),
                    (BatchAxis::new(0), BatchAxis::new(0)),
                    None,
                )?)
            },
            (ArrayType::new_static(DataType::F32, [2, 4]), ArrayType::new_static(DataType::F32, [2, 4])),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 4], %1:f32[2, 4] .
                let %2:f32[2, 4][sharding={mesh<['y'=4:explicit]>, [{}, {'y'}]}] = place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{}, {}]}},
                    target={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], \
                        sharding={mesh<['y'=4:explicit]>, [{}, {'y'}]}},
                ] %0
                    %3:f32[2, 4][sharding={mesh<['y'=4:explicit]>, [{}, {'y'}]}] = place [
                        source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{}, {}]}},
                        target={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], \
                            sharding={mesh<['y'=4:explicit]>, [{}, {'y'}]}},
                    ] %1
                in (%2, %3)"
            },
        );

        // Differentiating a batched placement transposes the lifted placement, so the batched cotangent moves back to
        // the source mesh with the lifted source sharding.
        let input_type = ArrayType::new_static(DataType::F32, [2, 4])
            .with_sharding(
                Sharding::new(
                    mesh_x().logical_mesh().clone(),
                    vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])],
                )
                .unwrap(),
            )
            .unwrap();
        let input = Array::from_elements(input_type, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        let source = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let (_, pullback) = differentiate_at(input)
            .vjp(|x| {
                Ok(batch(|item| item.place(Some(&source), &target), x, BatchAxis::new(0), BatchAxis::new(0), None)?)
            })
            .unwrap();
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[2, 4][sharding={mesh<['y'=4:explicit]>, [{}, {'y'}]}] .
                let %1:f32[2, 4][sharding={mesh<['x'=2:explicit]>, [{}, {'x'}]}] = place [
                    source={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], \
                        sharding={mesh<['y'=4:explicit]>, [{}, {'y'}]}},
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{}, {'x'}]}, memory=Device},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_differentiation_batching_inferred_source() {
        // A gradient taken inside a batched function moves the cotangent back to the devices that the packed input
        // concretely lives on, which the placement infers as its source from the batch item view of that input,
        // also when the target has the topology of the source and no source is given.
        let other_mesh_x =
            DeviceMesh::new(mesh_x().logical_mesh().clone(), vec![Device::new(2, 0), Device::new(3, 0)]).unwrap();
        let packed = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated]);
        let target = placement(&other_mesh_x, vec![ShardingDimension::Replicated]);
        let context = RecordingContext::default();
        let input = PlacedArray {
            array: Array::from_elements(
                ArrayType::new_static(DataType::F32, [2, 2]).with_sharding(packed.sharding().clone()).unwrap(),
                &[1.0f32, 2.0, 3.0, 4.0],
            )
            .unwrap(),
            placement: Some(packed.clone()),
            context: context.clone(),
        };
        let gradient = batch(
            |item| {
                let (output, pullback) = differentiate_at(item).vjp(|x| x.place(None, &target))?;
                pullback.apply_materialized(output)
            },
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();

        // The forward placement records the packed input placement as its source, so the lifted transpose moves the
        // cotangent back to the devices 0 and 1 with the sharding of the packed input, which the source of each batch
        // item records for its batch dimension. The explicit replicated target of each batch item replicates the
        // batch dimension of the forward output, as in JAX, so that is where the transpose moves the cotangent from.
        let target = placement(&other_mesh_x, vec![ShardingDimension::Replicated, ShardingDimension::Replicated]);
        let gradient_placement = packed.clone().with_memory(Memory::Device);
        assert_eq!(
            context.0.borrow().as_slice(),
            &[
                ArrayOperation::Place(PlaceOperation::new(Some(packed.clone()), target.clone())),
                ArrayOperation::Place(
                    PlaceOperation::new(Some(target), gradient_placement.clone()).with_may_alias(false),
                ),
            ],
        );
        assert_eq!(
            gradient.array,
            Array::from_elements(
                ArrayType::new_static(DataType::F32, [2, 2]).with_sharding(packed.sharding().clone()).unwrap(),
                &[1.0f32, 2.0, 3.0, 4.0],
            )
            .unwrap(),
        );
        assert_eq!(gradient.placement, Some(gradient_placement));
    }

    #[test]
    fn test_place_operation_differentiation_batching_auto_axis() {
        // The packed input is split along the auto axis `x` in its batch dimension on the devices 0 and 1, which its
        // type does not record, and each batch item is moved to the devices 2 and 3 of a mesh with the same axes. The
        // source of each batch item records that split, so the forward placement records the packed input placement as
        // its source, and the transpose moves the cotangent back to the devices 0 and 1 with its batch dimension split
        // along `x` again, rather than replicated.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let other_mesh_x =
            DeviceMesh::new(mesh_x().logical_mesh().clone(), vec![Device::new(2, 0), Device::new(3, 0)]).unwrap();
        let packed = Placement::new(
            mesh_x(),
            Sharding::new(auto_mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated])
                .unwrap(),
        )
        .unwrap();
        let array = Array::from_elements(
            ArrayType::new_static(DataType::F32, [2, 2])
                .with_sharding(Sharding::replicated(auto_mesh.clone(), 2))
                .unwrap(),
            &[1.0f32, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let target = Placement::new(other_mesh_x.clone(), Sharding::replicated(auto_mesh.clone(), 1)).unwrap();
        let context = RecordingContext::default();
        let input = PlacedArray { array: array.clone(), placement: Some(packed.clone()), context: context.clone() };
        let gradient = batch(
            |item| {
                let (output, pullback) = differentiate_at(item).vjp(|x| x.place(None, &target))?;
                pullback.apply_materialized(output)
            },
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        let target = Placement::new(other_mesh_x, Sharding::replicated(auto_mesh, 2)).unwrap();
        let gradient_placement = packed.clone().with_memory(Memory::Device);
        assert_eq!(
            context.0.borrow().as_slice(),
            &[
                ArrayOperation::Place(PlaceOperation::new(Some(packed), target.clone())),
                ArrayOperation::Place(
                    PlaceOperation::new(Some(target), gradient_placement.clone()).with_may_alias(false),
                ),
            ],
        );
        assert_eq!(gradient.array, array);
        assert_eq!(gradient.placement, Some(gradient_placement));
    }

    #[test]
    fn test_place_operation_differentiation_batching_identity() {
        // A gradient taken inside a batched function through a placement of each batch item without a target keeps the
        // primal where it lives (i.e., split along the auto axis `x` in its batch dimension, which its type does not
        // record) and records the batch item view of that placement, which records the split, as the source of the
        // tangent. The transposed placement moves the cotangent to that source, so batching puts the split back, and
        // the gradient lives where the packed input does.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let packed = Placement::new(
            mesh_x(),
            Sharding::new(auto_mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated])
                .unwrap(),
        )
        .unwrap();
        let array = Array::from_elements(
            ArrayType::new_static(DataType::F32, [2, 2])
                .with_sharding(Sharding::replicated(auto_mesh.clone(), 2))
                .unwrap(),
            &[1.0f32, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let context = RecordingContext::default();
        let input = PlacedArray { array: array.clone(), placement: Some(packed.clone()), context: context.clone() };
        let gradient = batch(
            |item| {
                let (output, pullback) = differentiate_at(item)
                    .vjp(|x| <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&x, None, None, true))?;
                pullback.apply_materialized(output)
            },
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        let gradient_placement = packed.clone().with_memory(Memory::Device);
        assert_eq!(
            context.0.borrow().as_slice(),
            &[
                ArrayOperation::Place(PlaceOperation::new(Some(packed.clone()), None)),
                ArrayOperation::Place(
                    PlaceOperation::new(Some(packed), gradient_placement.clone()).with_may_alias(false),
                ),
            ],
        );
        assert_eq!(gradient.array, array);
        assert_eq!(gradient.placement, Some(gradient_placement));
    }

    #[test]
    fn test_place_operation_differentiation_batching_nested() {
        // The packed input is split along `x` in its outer batch dimension and along `y` in its inner batch dimension,
        // so each batch item records both splits, from the outermost to the innermost, and each batching level puts
        // back its own. The placement of each batch item without a target therefore records the packed input placement
        // as its source, and the transpose moves the cotangent back to that placement.
        let auto_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let devices = DeviceMesh::new(auto_mesh.clone(), (0..4).map(|id| Device::new(id, 0)).collect()).unwrap();
        let packed = Placement::new(
            devices.clone(),
            Sharding::new(
                auto_mesh.clone(),
                vec![
                    ShardingDimension::sharded(["x"]),
                    ShardingDimension::sharded(["y"]),
                    ShardingDimension::Replicated,
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let array = Array::from_elements(
            ArrayType::new_static(DataType::F32, [2, 2, 1])
                .with_sharding(Sharding::replicated(auto_mesh.clone(), 3))
                .unwrap(),
            &[1.0f32, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let context = RecordingContext::default();
        let input = PlacedArray { array: array.clone(), placement: Some(packed.clone()), context: context.clone() };
        let item_placement = RefCell::new(None);
        let gradient = batch(
            |outer| {
                Ok(batch(
                    |item| {
                        *item_placement.borrow_mut() = item.placement();
                        let (output, pullback) = differentiate_at(item).vjp(|x| {
                            <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&x, None, None, true)
                        })?;
                        pullback.apply_materialized(output)
                    },
                    outer,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )?)
            },
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(
            item_placement.into_inner(),
            Some(
                Placement::new(devices, Sharding::replicated(auto_mesh, 1))
                    .unwrap()
                    .with_batch_sharding(ShardingDimension::sharded(["x"]))
                    .unwrap()
                    .with_batch_sharding(ShardingDimension::sharded(["y"]))
                    .unwrap(),
            ),
        );
        let gradient_placement = packed.clone().with_memory(Memory::Device);
        assert_eq!(
            context.0.borrow().as_slice(),
            &[
                ArrayOperation::Place(PlaceOperation::new(Some(packed.clone()), None)),
                ArrayOperation::Place(
                    PlaceOperation::new(Some(packed), gradient_placement.clone()).with_may_alias(false),
                ),
            ],
        );
        assert_eq!(gradient.array, array);
        assert_eq!(gradient.placement, Some(gradient_placement));
    }

    #[test]
    fn test_place_operation_differentiation_batching_without_source() {
        // A reference array reports no placement, so a placement of each batch item without a source records none,
        // and its transpose restores the tracked sharding of the batch item on the target mesh. That transposed target
        // records no batch sharding, so batching replicates its batch dimension like that of the forward target, and
        // the gradient is replicated even though the packed input type records a split of its batch dimension along
        // `x`. This pins a remaining limitation. Without a source, nothing records how the batch of the input was
        // distributed (JAX leaves such a cotangent where it is as well).
        let input_type = ArrayType::new_static(DataType::F32, [2, 2])
            .with_sharding(
                Sharding::new(
                    mesh_x().logical_mesh().clone(),
                    vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated],
                )
                .unwrap(),
            )
            .unwrap();
        let input = Array::from_elements(input_type, &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let target = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        let gradient = batch(
            |item| {
                let (output, pullback) = differentiate_at(item).vjp(|x| x.place(None, &target))?;
                pullback.apply_materialized(output)
            },
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(
            gradient,
            Array::from_elements(
                ArrayType::new_static(DataType::F32, [2, 2])
                    .with_sharding(Sharding::replicated(mesh_x().logical_mesh().clone(), 2))
                    .unwrap(),
                &[1.0f32, 2.0, 3.0, 4.0],
            )
            .unwrap(),
        );
    }

    #[test]
    fn test_place_operation_transposition() {
        // With a known source, the cotangent moves back to the source mesh with the cotangent dual of the source
        // sharding and the memory of the input, and the old target becomes the new source.
        let source = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]).with_memory(PINNED_HOST);
        let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(source.sharding().clone()).unwrap();
        let input = Array::from_elements(input_type.clone(), &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let (output, pullback) = differentiate_at(input).vjp(|x| x.place(Some(&source), &target)).unwrap();
        let cotangent = pullback
            .apply_materialized(
                Array::from_elements(output.r#type().cotangent().unwrap(), &[5.0f32, 6.0, 7.0, 8.0]).unwrap(),
            )
            .unwrap();
        assert_eq!(cotangent, Array::from_elements(input_type.cotangent().unwrap(), &[5.0f32, 6.0, 7.0, 8.0]).unwrap());
        let (pullback, residuals) = pullback.into_transposed_parts().unwrap();
        assert!(residuals.is_empty());
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['y'=4:explicit]>, [{'y'}]}]@Host[Pinned] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = place [
                    source={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], \
                        sharding={mesh<['y'=4:explicit]>, [{'y'}]}},
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{'x'}]}, memory=Device},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );

        // Without a source, the cotangent stays on the target mesh and gets the tracked sharding of the input back.
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let input_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh_x().logical_mesh().clone(), 1))
            .unwrap();
        let input = Array::from_elements(input_type, &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let (_, pullback) = differentiate_at(input).vjp(|x| x.place(None, &target)).unwrap();
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] = place [
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{}]}, memory=Device},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );

        // An input without a tracked sharding gets its exact cotangent type through a placement-only broadcast.
        let input = Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let (_, pullback) = differentiate_at(input).vjp(|x| x.place(None, &target)).unwrap();
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = place [
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{'x'}]}, memory=Device},
                    may_alias=false,
                ] %0
                    %2:f32[4] = broadcast [output_type=f32[4], output_axes=[0]] %1
                in (%2)"
            },
        );

        // An input that is tracked over another mesh did change meshes, so its cotangent cannot be moved back to
        // devices that are not known.
        let input_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh_y().logical_mesh().clone(), 1))
            .unwrap();
        let input = Array::from_elements(input_type, &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let (_, pullback) = differentiate_at(input).vjp(|x| x.place(None, &target)).unwrap();
        assert!(matches!(
            pullback.into_transposed_parts(),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == format!(
                    "cannot transpose `{PLACE_OPERATION_NAME}` across device meshes without a source placement",
                ),
        ));
    }

    #[test]
    fn test_place_operation_transposition_without_target() {
        // Without a target, the primal stays where it is, and the cotangent moves to the source with the memory of the
        // input and no source of its own. The input type tracks no sharding here, so it gets its exact cotangent type
        // through a placement-only broadcast.
        let source = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        let input = Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let (output, pullback) = differentiate_at(input.clone())
            .vjp(|x| <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&x, Some(source.clone()), None, true))
            .unwrap();
        assert_eq!(output, input);
        let cotangent = Array::vector(vec![5.0f32, 6.0, 7.0, 8.0]).unwrap();
        assert_eq!(pullback.apply_materialized(cotangent.clone()), Ok(cotangent.clone()));
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4] .
                let %1:f32[4][sharding={mesh<['y'=4:explicit]>, [{'y'}]}] = place [
                    target={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], \
                        sharding={mesh<['y'=4:explicit]>, [{'y'}]}, memory=Device},
                    may_alias=false,
                ] %0
                    %2:f32[4] = broadcast [output_type=f32[4], output_axes=[0]] %1
                in (%2)"
            },
        );

        // Without a target and a source, the placement is an identity, and so is its transpose, which still forces a
        // copy of the cotangent.
        let (_, pullback) = differentiate_at(input)
            .vjp(|x| <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&x, None, None, true))
            .unwrap();
        let copied = pullback.apply_materialized(cotangent.clone()).unwrap();
        assert_eq!(copied, cotangent);
        assert!(!Arc::ptr_eq(cotangent.shared_storage_bytes(), copied.shared_storage_bytes()));
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4] .
                let %1:f32[4] = place [may_alias=false] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_transposition_may_alias() {
        // The transposed placement forces a copy of the cotangent whatever the copy policy of the forward placement is,
        // so a forward placement that forces a copy itself transposes to the same placement as one that does not.
        let source = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]).with_memory(PINNED_HOST);
        let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(source.sharding().clone()).unwrap();
        let input = Array::from_elements(input_type.clone(), &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let (output, pullback) = differentiate_at(input)
            .vjp(|x| {
                <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(
                    &x,
                    Some(source.clone()),
                    Some(&target),
                    false,
                )
            })
            .unwrap();
        let cotangent = pullback
            .apply_materialized(
                Array::from_elements(output.r#type().cotangent().unwrap(), &[5.0f32, 6.0, 7.0, 8.0]).unwrap(),
            )
            .unwrap();
        assert_eq!(cotangent, Array::from_elements(input_type.cotangent().unwrap(), &[5.0f32, 6.0, 7.0, 8.0]).unwrap());
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['y'=4:explicit]>, [{'y'}]}]@Host[Pinned] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = place [
                    source={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], \
                        sharding={mesh<['y'=4:explicit]>, [{'y'}]}},
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{'x'}]}, memory=Device},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_transposition_unsharded_input() {
        // An input without a tracked sharding moves back to the source mesh and then gets its exact cotangent type
        // through a placement-only broadcast.
        let source = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        let input = Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let (_, pullback) = differentiate_at(input).vjp(|x| x.place(Some(&source), &target)).unwrap();
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['y'=4:explicit]>, [{'y'}]}] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] = place [
                    source={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], sharding={mesh<['y'=4:explicit]>, [{'y'}]}},
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{}]}, memory=Device},
                    may_alias=false,
                ] %0
                    %2:f32[4] = broadcast [output_type=f32[4], output_axes=[0]] %1
                in (%2)"
            },
        );
    }

    #[test]
    fn test_place_operation_transposition_reduction_state() {
        // The cotangent moves back with the cotangent duals of the source and target shardings, which swap their
        // unreduced and reduced axes, so an unreduced input receives a reduced cotangent and vice versa.
        let other_mesh_x =
            DeviceMesh::new(mesh_x().logical_mesh().clone(), vec![Device::new(2, 0), Device::new(3, 0)]).unwrap();
        let unreduced = Sharding::replicated(mesh_x().logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let input = Array::from_elements(
            ArrayType::new_static(DataType::F32, [4]).with_sharding(unreduced.clone()).unwrap(),
            &[1.0f32, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let source = Placement::new(mesh_x(), unreduced.clone()).unwrap();
        let target = Placement::new(other_mesh_x.clone(), unreduced).unwrap();
        let (_, pullback) = differentiate_at(input).vjp(|x| x.place(Some(&source), &target)).unwrap();
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{}], reduced={'x'}}] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{}], reduced={'x'}}] = place [
                    source={mesh<['x'=2:explicit]>, devices=[2, 3], \
                        sharding={mesh<['x'=2:explicit]>, [{}], reduced={'x'}}},
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{}], reduced={'x'}}, memory=Device},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );

        // A reduced input receives an unreduced cotangent.
        let reduced = Sharding::replicated(mesh_x().logical_mesh().clone(), 1).with_reduced_axes(["x"]).unwrap();
        let input = Array::from_elements(
            ArrayType::new_static(DataType::F32, [4]).with_sharding(reduced.clone()).unwrap(),
            &[1.0f32, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let source = Placement::new(mesh_x(), reduced.clone()).unwrap();
        let target = Placement::new(other_mesh_x, reduced).unwrap();
        let (_, pullback) = differentiate_at(input).vjp(|x| x.place(Some(&source), &target)).unwrap();
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{}], unreduced={'x'}}] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{}], unreduced={'x'}}] = place [
                    source={mesh<['x'=2:explicit]>, devices=[2, 3], \
                        sharding={mesh<['x'=2:explicit]>, [{}], unreduced={'x'}}},
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{}], unreduced={'x'}}, memory=Device},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_transposition_undefined_cotangent() {
        // A value that is unreduced with a kind other than a sum has no cotangent, so the placement cannot be
        // transposed. Transposition contexts already reject such inputs when they create their accumulators, so the
        // accumulators here come from the corresponding sum, which isolates the rejection by the rule itself.
        let other_mesh_x =
            DeviceMesh::new(mesh_x().logical_mesh().clone(), vec![Device::new(2, 0), Device::new(3, 0)]).unwrap();
        let sum = Sharding::replicated(mesh_x().logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let maximum = sum.clone().with_unreduced_kind(UnreducedKind::Max).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(maximum.clone()).unwrap();
        let target = Placement::new(other_mesh_x, maximum.clone()).unwrap();
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let mut rule_context = TranspositionContext::new(context.clone());
        let sum_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sum).unwrap();
        assert_eq!(
            rule_context.cotangent_accumulators(&[PartialValue::Unknown(input_type.clone())], &[]).err(),
            Some(DifferentiationError::UndefinedCotangentType { primal_type: input_type.to_string() }),
        );
        let accumulators =
            rule_context.cotangent_accumulators(&[PartialValue::Unknown(sum_type.clone())], &[]).unwrap();
        let inputs = [PartialValue::Unknown(input_type.clone())];
        let cotangent = context.input(sum_type.cotangent().unwrap());
        assert_eq!(
            PlaceOperation::new(Some(Placement::new(mesh_x(), maximum).unwrap()), target).transpose(
                &mut rule_context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Value(cotangent)],
                &accumulators,
            ),
            Err(DifferentiationError::UndefinedCotangentType { primal_type: input_type.to_string() }),
        );
    }

    #[test]
    fn test_place_operation_transposition_auto_source() {
        // The source sharding places the input along the auto axis `x`, which the input type does not record, so the
        // cotangent moves back with that runtime placement and its type matches the input cotangent type exactly.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let source = Placement::new(
            mesh_x(),
            Sharding::new(auto_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
        )
        .unwrap();
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        let input = Array::from_elements(
            ArrayType::new_static(DataType::F32, [4]).with_sharding(Sharding::replicated(auto_mesh, 1)).unwrap(),
            &[1.0f32, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let (_, pullback) = differentiate_at(input).vjp(|x| x.place(Some(&source), &target)).unwrap();
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['y'=4:explicit]>, [{'y'}]}] .
                let %1:f32[4][sharding={mesh<['x'=2:auto]>, [{}]}] = place [
                    source={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], sharding={mesh<['y'=4:explicit]>, [{'y'}]}},
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:auto]>, [{'x'}]}, memory=Device},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_transposition_uneven_source() {
        // The source splits an extent of 3 over the 2 devices along the auto axis `x`, which the input type does not
        // record. The transposed placement moves the cotangent back to that source, which type inference rejects
        // because a placement cannot describe an uneven split.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let source = Placement::new(
            mesh_x(),
            Sharding::new(auto_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
        )
        .unwrap();
        let target = placement(&mesh_y(), vec![ShardingDimension::Replicated]);
        let input = Array::from_elements(
            ArrayType::new_static(DataType::F32, [3]).with_sharding(Sharding::replicated(auto_mesh, 1)).unwrap(),
            &[1.0f32, 2.0, 3.0],
        )
        .unwrap();
        let (_, pullback) = differentiate_at(input).vjp(|x| x.place(Some(&source), &target)).unwrap();
        assert!(matches!(
            pullback.into_transposed_parts(),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == format!(
                    "`{PLACE_OPERATION_NAME}` target sharding splits dimension #0 with extent 3 into 2 shards, which \
                     do not divide it evenly",
                ),
        ));
    }

    #[test]
    fn test_place_operation_transposition_host_memory() {
        // The cotangent returns to the memory of the input, which is pinned host memory here.
        let source = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]).with_memory(Memory::Device);
        let input = Array::from_elements(
            ArrayType::new_static(DataType::F32, [4])
                .with_sharding(source.sharding().clone())
                .unwrap()
                .with_memory(PINNED_HOST),
            &[1.0f32, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let (_, pullback) = differentiate_at(input).vjp(|x| x.place(Some(&source), &target)).unwrap();
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['y'=4:explicit]>, [{'y'}]}] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}]@Host[Pinned] = place [
                    source={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], sharding={mesh<['y'=4:explicit]>, [{'y'}]}},
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{}]}, memory=Host[Pinned]},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_transposition_manual_mesh() {
        // A replicated scalar over a mesh with a manual axis is a global value outside any manual region, so placing it
        // where it already lives without a source transposes to a placement that gives the cotangent its sharding back.
        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let target = Placement::new(mesh_x(), Sharding::replicated(manual_mesh.clone(), 0)).unwrap();
        let input_type = ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(manual_mesh, 0)).unwrap();
        let input = Array::from_elements(input_type.clone(), &[3.0f32]).unwrap();
        let (output, pullback) = differentiate_at(input).vjp(|x| x.place(None, &target)).unwrap();
        assert_eq!(output, Array::from_elements(input_type.clone(), &[3.0f32]).unwrap());
        let cotangent =
            pullback.apply_materialized(Array::from_elements(input_type.clone(), &[5.0f32]).unwrap()).unwrap();
        assert_eq!(cotangent, Array::from_elements(input_type, &[5.0f32]).unwrap());
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=2:manual]>, []}] .
                let %1:f32[][sharding={mesh<['x'=2:manual]>, []}] = place [
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:manual]>, []}, memory=Device},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_transposition_second_order() {
        // The pullback of a placement is itself a placement, so differentiating a function that applies the pullback
        // transposes it again and moves the cotangent of the cotangent to the target of the original placement.
        let source = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]).with_memory(PINNED_HOST);
        let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(source.sharding().clone()).unwrap();
        let input = Array::from_elements(input_type.clone(), &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let output_cotangent_type = input_type
            .with_sharding(target.sharding().clone())
            .unwrap()
            .with_memory(PINNED_HOST)
            .cotangent()
            .unwrap();
        let output_cotangent = Array::from_elements(output_cotangent_type, &[5.0f32, 6.0, 7.0, 8.0]).unwrap();
        let (_, pullback) = differentiate_at(output_cotangent)
            .vjp(|cotangent| {
                let input = cotangent.context().constant(input)?;
                let (_, pullback) = differentiate_at(input).vjp(|x| x.place(Some(&source), &target))?;
                pullback.apply_materialized(cotangent)
            })
            .unwrap();
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] .
                let %1:f32[4][sharding={mesh<['y'=4:explicit]>, [{'y'}]}]@Host[Pinned] = place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                    target={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], \
                        sharding={mesh<['y'=4:explicit]>, [{'y'}]}, memory=Host[Pinned]},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_transposition_zero_cotangent() {
        // A symbolic zero cotangent propagates unchanged without staging a placement.
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let input_type = ArrayType::new_static(DataType::F32, [2]);
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let mut rule_context = TranspositionContext::new(context.clone());
        let inputs = [PartialValue::Unknown(input_type.clone())];
        let accumulators = rule_context.cotangent_accumulators(&inputs, &[]).unwrap();
        PlaceOperation::new(None, target.clone())
            .transpose(
                &mut rule_context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Zero(input_type.clone().with_sharding(target.sharding().clone()).unwrap())],
                &accumulators,
            )
            .unwrap();
        let contributions = rule_context.take_cotangents(&accumulators).unwrap();
        assert!(contributions[0].is_zero());
        assert_eq!(contributions[0].r#type().as_ref(), &input_type);
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_place_operation_provider() {
        // Homogeneous array families provide the placement directly, and array IR families through their array
        // projection.
        let operation = PlaceOperation::new(None, placement(&mesh_x(), vec![ShardingDimension::Replicated]));
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        assert_eq!(
            ArrayOperation::<Array>::provide(operation.clone(), &[&array_type]).unwrap().to_string(),
            operation.to_string(),
        );
        assert_eq!(
            ArrayOperation::<Array>::provide(operation.clone(), &[]).unwrap_err(),
            ProgramError::InvalidInputCount { expected: 1, actual: 0 },
        );
        let ir_type = ArrayIrType::Array(array_type);
        assert!(matches!(
            ArrayIrOperation::<Array>::provide(operation.clone(), &[&ir_type]).unwrap(),
            ArrayIrOperation::Array(ArrayOperation::Place(provided)) if provided == operation,
        ));
    }

    #[test]
    fn test_placed() {
        /// Value that reports a fixed placement.
        struct FixedPlacement(Placement);

        impl Placed for FixedPlacement {
            fn placement(&self) -> Option<Placement> {
                Some(self.0.clone())
            }
        }

        // Reference arrays, staged and unknown values, and batches of reference arrays have no placement.
        let array = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let array_type = array.r#type().into_owned();
        assert_eq!(array.placement(), None);
        let tracer = TracingContext::<Array, ArrayOperation<Array>>::new().input(array_type.clone());
        assert_eq!(tracer.placement(), None);
        let partial = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        assert_eq!(PartialTracer::new(partial.clone(), partial.unknown_input(array_type.clone(), 0)).placement(), None);
        let batching = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);
        assert_eq!(batching.lift(array.clone()).unwrap().placement(), None);

        // Wrappers report the placement of the value that they wrap, which a reference array does not have.
        assert_eq!(partial.lift(array.clone()).unwrap().placement(), None);
        let differentiation = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(array.clone()).unwrap(),
            DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new()),
        );
        assert_eq!(differentiation.placement(), None);
        assert_eq!(ProjectedValue::new(array.clone(), array_type.clone()).placement(), None);
        let fixed = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        assert_eq!(ProjectedValue::new(FixedPlacement(fixed.clone()), array_type).placement(), Some(fixed));

        // Only the array members of array IR values live on devices.
        assert_eq!(ArrayIrValue::Array(array).placement(), None);
        assert_eq!(ArrayIrValue::<Array>::Dimension(DimensionValue::constant(3).unwrap()).placement(), None);

        // Batched values report the placement of their batch items, which drops the batch dimension from the sharding
        // of the packed value along with its layout, keeps its memory, and records the sharding of the batch dimension
        // as how the batch is distributed. Replicated values report the placement of their packed value with a
        // replicated batch recorded, because each of their batch items is the whole packed value.
        let context = RecordingContext::default();
        let packed = placement(&mesh_x(), vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])])
            .with_memory(PINNED_HOST)
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])));
        let item = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])])
            .with_memory(PINNED_HOST)
            .with_batch_sharding(ShardingDimension::Replicated)
            .unwrap();
        let split_item = placement(&mesh_x(), vec![ShardingDimension::Replicated])
            .with_memory(PINNED_HOST)
            .with_batch_sharding(ShardingDimension::sharded(["x"]))
            .unwrap();
        let replicated_item = packed.clone().with_batch_sharding(ShardingDimension::Replicated).unwrap();
        let packed_array = PlacedArray {
            array: Array::matrix(2, 2, vec![1.0f32, 2.0, 3.0, 4.0]).unwrap(),
            placement: Some(packed.clone()),
            context: context.clone(),
        };
        let mapped = ArrayBatch::new(packed_array.clone(), BatchAxis::new(0)).unwrap();
        assert_eq!(mapped.placement(), Some(item.clone()));
        assert_eq!(ArrayBatch::new(packed_array.clone(), BatchAxis::new(1)).unwrap().placement(), Some(split_item));
        assert_eq!(ArrayBatch::replicated(packed_array.clone()).placement(), Some(replicated_item.clone()));
        let packed_ir_value = ArrayIrValue::Array(packed_array.clone());
        assert_eq!(
            ArrayIrBatch::new(packed_ir_value.clone(), BatchAxis::new(0)).unwrap().placement(),
            Some(item.clone()),
        );
        assert_eq!(ArrayIrBatch::replicated(packed_ir_value).placement(), Some(replicated_item));
        let batching = BatchingContext::<_, ArrayBatchingPolicy>::new(context.clone(), 2);
        assert_eq!(BatchingTracer::new(batching, mapped).placement(), Some(item));
        assert_eq!(
            ArrayBatch::new(context.lift(packed_array.array).unwrap(), BatchAxis::new(0)).unwrap().placement(),
            None,
        );
    }

    #[test]
    fn test_place_dispatch() {
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let expected = Array::from_elements(
            ArrayType::new_static(DataType::F32, [2]).with_sharding(target.sharding().clone()).unwrap(),
            &[1.0f32, 2.0],
        )
        .unwrap();

        // A reference array shares its storage by default and copies it when the copy policy forbids sharing it.
        let shared = ValueDirectDispatch::place(&input, None, Some(&target), true).unwrap();
        assert_eq!(shared, expected);
        assert!(Arc::ptr_eq(input.shared_storage_bytes(), shared.shared_storage_bytes()));
        let copied = ValueDirectDispatch::place(&input, None, Some(&target), false).unwrap();
        assert_eq!(copied, expected);
        assert!(!Arc::ptr_eq(input.shared_storage_bytes(), copied.shared_storage_bytes()));

        // Without a target, a reference array keeps its type and follows the copy policy in the same way.
        let shared = ValueDirectDispatch::place(&input, None, None, true).unwrap();
        assert_eq!(shared, input);
        assert!(Arc::ptr_eq(input.shared_storage_bytes(), shared.shared_storage_bytes()));
        let copied = ValueDirectDispatch::place(&input, None, None, false).unwrap();
        assert_eq!(copied, input);
        assert!(!Arc::ptr_eq(input.shared_storage_bytes(), copied.shared_storage_bytes()));
        assert_eq!(
            ValueDirectDispatch::place(&input, Some(placement(&mesh_x(), vec![])), None, true),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PLACE_OPERATION_NAME}` source sharding rank (0) does not match the input rank (1)",
            )))),
        );

        // Array IR values pass the copy policy on to their array members.
        let ir_input = ArrayIrValue::Array(input.clone());
        let ArrayIrValue::Array(copied) = ValueDirectDispatch::place(&ir_input, None, Some(&target), false).unwrap()
        else {
            panic!("placing an array member must produce an array member");
        };
        assert_eq!(copied, expected);
        assert!(!Arc::ptr_eq(input.shared_storage_bytes(), copied.shared_storage_bytes()));

        // Values that dispatch through their domain bind the placement with the copy policy.
        let context = RecordingContext::default();
        let source = placement(&mesh_y(), vec![ShardingDimension::Replicated]);
        let placed = context.lift(Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap()).unwrap();
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        ValueDomainDispatch::place(&placed, Some(source.clone()), Some(&target), false).unwrap();
        ValueDomainDispatch::place(&placed, None, Some(&target), true).unwrap();
        ValueDomainDispatch::place(&placed, None, None, false).unwrap();
        assert_eq!(
            context.0.borrow().as_slice(),
            &[
                ArrayOperation::Place(PlaceOperation::new(Some(source), target.clone()).with_may_alias(false)),
                ArrayOperation::Place(PlaceOperation::new(None, target)),
                ArrayOperation::Place(PlaceOperation::new(None, None).with_may_alias(false)),
            ],
        );
    }

    #[test]
    fn test_placement_sources() {
        let x = placement(&mesh_x(), vec![]);

        // An optional structure of placements supplies the source of every leaf or of none.
        let sources = (x.clone(), vec![x.clone()]);
        assert_eq!(PlacementSources::as_parameterized(&Some(&sources)), Some(&sources));
        assert_eq!(PlacementSources::<Placement, (Placement, Vec<Placement>)>::as_parameterized(&None), None);

        // A structure of optional placements is always supplied, and each of its leaves can omit its own source.
        let sources = (Some(x), vec![None]);
        assert_eq!(PlacementSources::as_parameterized(&&sources), Some(&sources));
    }

    #[test]
    fn test_place() {
        // Staged values have no concrete placement, so placing them without a source stages an unknown source.
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input| input.place(None, &target),
            ArrayType::new_static(DataType::F32, [4]),
        )
        .unwrap();
        assert_eq!(
            output_type,
            ArrayType::new_static(DataType::F32, [4]).with_sharding(target.sharding().clone()).unwrap(),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = place [
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                ] %0
                in (%1)"
            },
        );

        // Array IR values place their array members, with a given source staged as is.
        let source = placement(&mesh_y(), vec![ShardingDimension::Replicated]);
        let (output_type, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |input| input.place(Some(&source), &target),
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4])),
        )
        .unwrap();
        assert_eq!(
            output_type,
            ArrayIrType::Array(
                ArrayType::new_static(DataType::F32, [4]).with_sharding(target.sharding().clone()).unwrap()
            ),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = place [
                    source={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], \
                        sharding={mesh<['y'=4:explicit]>, [{}]}},
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                ] %0
                in (%1)"
            },
        );
        assert_eq!(
            ArrayIrValue::<Array>::Dimension(DimensionValue::constant(1).unwrap()).place(None, &target),
            Err(ProgramError::Type(TypeError::invalid("expected array type but got dimension type"))),
        );

        // Reference arrays have no devices, so placing them keeps their payload and records the placement in their
        // type, and they validate the target and source exactly like staged programs do.
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let output = input.place(None, &target.clone().with_memory(PINNED_HOST)).unwrap();
        assert_eq!(
            output,
            Array::from_elements(
                ArrayType::new_static(DataType::F32, [2])
                    .with_sharding(target.sharding().clone())
                    .unwrap()
                    .with_memory(PINNED_HOST),
                &[1.0f32, 2.0],
            )
            .unwrap(),
        );
        assert!(Arc::ptr_eq(input.shared_storage_bytes(), output.shared_storage_bytes()));
        assert_eq!(
            input.place(None, &placement(&mesh_x(), vec![])),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PLACE_OPERATION_NAME}` target sharding rank (0) does not match the input rank (1)",
            )))),
        );
        assert_eq!(
            input.place(Some(&placement(&mesh_x(), vec![])), &target),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PLACE_OPERATION_NAME}` source sharding rank (0) does not match the input rank (1)",
            )))),
        );
    }

    #[test]
    fn test_place_layouts() {
        // A layout determines how the storage addresses the elements, so a reference array placed with another layout
        // re-encodes its elements instead of reinterpreting its storage. Reversing the byte stride stores the elements
        // in reverse order, and widening it leaves a hole between them, while the elements stay the same.
        let input = Array::from_logical_bytes(ArrayType::new_static(DataType::U8, [2]), &[10, 20]).unwrap();
        let target = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        let output_type = |stride| {
            ArrayType::new_static(DataType::U8, [2])
                .with_sharding(target.sharding().clone())
                .unwrap()
                .with_layout(Layout::Strided(StridedLayout::new(vec![stride])))
        };
        let output = input.place(None, &target.clone().with_layout(Layout::Strided(StridedLayout::new(vec![-1]))));
        assert_eq!(output, Array::new(output_type(-1), vec![20, 10]));
        let output = output.unwrap();
        assert_eq!(output.elements::<u8>(), Ok(vec![10, 20]));
        assert_eq!(output.logical_bytes(), vec![10, 20]);
        let output = input.place(None, &target.clone().with_layout(Layout::Strided(StridedLayout::new(vec![2]))));
        assert_eq!(output, Array::new(output_type(2), vec![10, 0, 20]));
        let output = output.unwrap();
        assert_eq!(output.elements::<u8>(), Ok(vec![10, 20]));
        assert_eq!(output.logical_bytes(), vec![10, 20]);

        // An unchanged layout keeps sharing the storage.
        let output = input.place(None, &target).unwrap();
        assert!(Arc::ptr_eq(input.shared_storage_bytes(), output.shared_storage_bytes()));
    }

    #[test]
    fn test_place_source_inference() {
        let context = RecordingContext::default();
        let concrete = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        let explicit = placement(&mesh_y(), vec![ShardingDimension::Replicated]);
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        let array = Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let placed = PlacedArray { array: array.clone(), placement: Some(concrete.clone()), context: context.clone() };
        let unplaced = context.lift(array).unwrap();

        // A value with a concrete placement records it as the source, an explicit source wins over it, and a value
        // without a concrete placement stages no source.
        placed.place(None, &target).unwrap();
        placed.place(Some(&explicit), &target).unwrap();
        unplaced.place(None, &target).unwrap();
        assert_eq!(
            context.0.borrow().as_slice(),
            &[
                ArrayOperation::Place(PlaceOperation::new(Some(concrete), target.clone())),
                ArrayOperation::Place(PlaceOperation::new(Some(explicit), target.clone())),
                ArrayOperation::Place(PlaceOperation::new(None, target)),
            ],
        );
    }

    #[test]
    fn test_place_structured() {
        /// Exercises a single-value caller that specifies its universe.
        fn place_array<V: Value + Place<ArrayType>>(input: &V, target: &Placement) -> Result<V, ProgramError> {
            input.place(None, target)
        }

        /// Only the placeholder structure needs to be cloneable to rebuild this receiver.
        #[derive(ryft_macros::Parameterized)]
        struct Inputs<P: Parameter> {
            /// Vector and scalar groups with different placement ranks.
            values: (P, Vec<P>),

            /// Static metadata supplied by the receiver.
            label: &'static str,
        }

        impl Clone for Inputs<Placeholder> {
            fn clone(&self) -> Self {
                Self { values: self.values.clone(), label: self.label }
            }
        }

        let vector = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let scalar = Array::scalar(3.0f32).unwrap();
        let x = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let y = placement(&mesh_y(), vec![]).with_memory(PINNED_HOST);

        // Every leaf is placed on the target at its parameter path, and the rest of the structure is kept.
        let inputs = Inputs { values: (vector.clone(), vec![scalar.clone()]), label: "inputs" };
        let targets = Inputs { values: (x.clone(), vec![y.clone()]), label: "targets" };
        let outputs = inputs.place(None, &targets).unwrap();
        assert_eq!(outputs.label, "inputs");
        assert_eq!(outputs.values, (place_array(&vector, &x).unwrap(), vec![scalar.place(None, &y).unwrap()]));

        // Structures of array IR values place their array members, and their other members cannot be placed.
        let inputs = vec![ArrayIrValue::Array(scalar.clone())];
        assert_eq!(
            inputs.place(None, &vec![y.clone()]),
            Ok(vec![ArrayIrValue::Array(scalar.place(None, &y).unwrap())]),
        );
        let dimension = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(1).unwrap());
        assert_eq!(
            vec![dimension].place(None, &vec![y]),
            Err(ProgramError::Type(TypeError::invalid("expected array type but got dimension type"))),
        );

        // Structures without leaves place nothing.
        assert!(Vec::<Array>::new().place(None, &Vec::new()).unwrap().is_empty());
        assert_eq!(<() as Place<ArrayType, Array>>::place(&(), Some(&()), &()), Ok(()));
    }

    #[test]
    fn test_place_structured_sources() {
        let context = RecordingContext::default();
        let x = placement(&mesh_x(), vec![]);
        let y = placement(&mesh_y(), vec![]);
        let array = Array::scalar(1.0f32).unwrap();
        let first = PlacedArray { array: array.clone(), placement: Some(x.clone()), context: context.clone() };
        let second = PlacedArray { array: array.clone(), placement: Some(y.clone()), context: context.clone() };
        let inputs = (first, [second, context.lift(array).unwrap()]);
        let targets = (y.clone(), [x.clone(), x.clone()]);

        // Without sources, every leaf records where it concretely lives, if it reports that.
        inputs.place(None, &targets).unwrap();
        assert_eq!(
            context.0.take(),
            vec![
                ArrayOperation::Place(PlaceOperation::new(Some(x.clone()), y.clone())),
                ArrayOperation::Place(PlaceOperation::new(Some(y.clone()), x.clone())),
                ArrayOperation::Place(PlaceOperation::new(None, x.clone())),
            ],
        );

        // A structure of sources overrides the placement that every leaf reports.
        let sources = (y.clone(), [x.clone(), y.clone()]);
        inputs.place(Some(&sources), &targets).unwrap();
        assert_eq!(
            context.0.take(),
            vec![
                ArrayOperation::Place(PlaceOperation::new(Some(y.clone()), y.clone())),
                ArrayOperation::Place(PlaceOperation::new(Some(x.clone()), x.clone())),
                ArrayOperation::Place(PlaceOperation::new(Some(y.clone()), x.clone())),
            ],
        );

        // A structure of optional sources overrides it only for the leaves that it names a source for.
        inputs.place(&(None, [Some(x.clone()), None]), &targets).unwrap();
        assert_eq!(
            context.0.take(),
            vec![
                ArrayOperation::Place(PlaceOperation::new(Some(x.clone()), y)),
                ArrayOperation::Place(PlaceOperation::new(Some(x.clone()), x.clone())),
                ArrayOperation::Place(PlaceOperation::new(None, x)),
            ],
        );
    }

    #[test]
    fn test_place_structured_mismatched_parameter_paths() {
        let context = RecordingContext::default();
        let input = context.lift(Array::scalar(1.0f32).unwrap()).unwrap();
        let inputs = BTreeMap::from([("a", input.clone()), ("b", input)]);
        let x = placement(&mesh_x(), vec![]);
        let targets = BTreeMap::from([("a", x.clone()), ("b", x.clone())]);
        let mismatched = BTreeMap::from([("a", x.clone()), ("c", x)]);
        let target_error = ProgramError::InvalidArgument {
            message: "receiver and `targets` must have the same parameter structure".to_string(),
        };
        let source_error = ProgramError::InvalidArgument {
            message: "receiver and `sources` must have the same parameter structure".to_string(),
        };

        // The targets and the sources must have the parameter paths of the receiver, and no leaf is placed otherwise.
        assert_eq!(inputs.place(None, &mismatched).unwrap_err(), target_error);
        assert_eq!(inputs.place(Some(&mismatched), &targets).unwrap_err(), source_error);
        let missing = BTreeMap::from([("a", None)]);
        assert_eq!(inputs.place(&missing, &targets).unwrap_err(), source_error);
        let wrong_paths = BTreeMap::from([("a", None), ("c", None)]);
        assert_eq!(inputs.place(&wrong_paths, &targets).unwrap_err(), source_error);
        assert!(context.0.borrow().is_empty());
    }

    #[test]
    fn test_unbatched_placement() {
        let packed = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated])
            .with_memory(PINNED_HOST)
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])));

        // The batch dimension leaves the sharding and is recorded as how the batch is distributed, the memory is kept,
        // and the layout, which describes the storage of the packed value, is dropped.
        assert_eq!(
            unbatched_placement(packed.clone(), Some(0)),
            Some(
                placement(&mesh_x(), vec![ShardingDimension::Replicated])
                    .with_memory(PINNED_HOST)
                    .with_batch_sharding(ShardingDimension::sharded(["x"]))
                    .unwrap(),
            ),
        );
        assert_eq!(
            unbatched_placement(packed.clone(), Some(1)),
            Some(
                placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])])
                    .with_memory(PINNED_HOST)
                    .with_batch_sharding(ShardingDimension::Replicated)
                    .unwrap(),
            ),
        );

        // A sharding that has no dimension at the batch axis describes no batch items.
        assert_eq!(unbatched_placement(packed.clone(), Some(2)), None);

        // A value without a batch dimension is its own batch item, so it keeps its placement, including its layout,
        // with a replicated batch recorded.
        assert_eq!(
            unbatched_placement(packed.clone(), None),
            Some(packed.clone().with_batch_sharding(ShardingDimension::Replicated).unwrap()),
        );

        // The batch shardings of enclosing batching levels are kept from the outermost to the innermost.
        let item = unbatched_placement(unbatched_placement(packed, Some(0)).unwrap(), Some(0)).unwrap();
        assert_eq!(item.sharding().rank(), 0);
        assert_eq!(item.batch_shardings(), &[ShardingDimension::sharded(["x"]), ShardingDimension::Replicated]);
    }

    #[test]
    fn test_placement_without_layout() {
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));

        // Only the layout is dropped, so the memory and the recorded batch shardings are kept.
        let placed = placement(&mesh_x(), vec![ShardingDimension::Replicated])
            .with_memory(PINNED_HOST)
            .with_batch_sharding(ShardingDimension::sharded(["x"]))
            .unwrap();
        assert_eq!(placement_without_layout(&placed.clone().with_layout(layout.clone())), placed);
        assert_eq!(placement_without_layout(&placed), placed);

        // A placement that names no memory still names none.
        let placed = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        assert_eq!(placement_without_layout(&placed.clone().with_layout(layout)), placed);
    }
}
