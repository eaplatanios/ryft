use std::fmt::Display;

use crate::axes::Axis;
use crate::batching::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, BatchAxis, BatchableOperation, BatchingContext, BatchingDriver,
    BatchingError, BatchingPolicy, ProgramBatchingOutputAxesPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::DifferentiationError;
use crate::differentiation::forward::{DifferentiableOperation, DifferentiationDriver, DifferentiationDual};
use crate::differentiation::reverse::{TransposableOperation, TranspositionDriver};
use crate::differentiation::types::DifferentiableType;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, check_types};
use crate::operations::constants::{Zero, ZeroOperation, ZeroOperationProvider};
use crate::operations::math::{Reduce, ReduceOperation, ReductionKind};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::ProgramError;
use crate::programs::atoms::{AtomId, MaybeZero};
use crate::programs::builders::ProgramBuilder;
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::{OutputRegionProvenance, RegionInterface, RegionSlot};
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::Value;
use crate::tracing::{NestedTracingContext, Tracer, TracingContext};
use crate::types::ArrayType;

/// Differentiation-owned protocol through which an operation family materializes zeros whose runtime geometry must
/// be supplied by explicitly captured _residual_ values, because it is not derivable from the zero's [`Type`] alone.
///
/// # Why this Protocol Exists
///
/// Differentiation is the one transform that must synthesize values with no data edge to derive them from. For example,
/// transposition is *defined* to return a cotangent for every differentiated input, including inputs that are
/// disconnected from every output, and the mathematically determined value for such an input is a zero of its cotangent
/// type. For a static type this is easy as [`ZeroOperationProvider::zero_operation`] constructs the zero from the type,
/// with no operands. For a type with dynamic axes it is impossible as a [`Type`] carries only dimension _identities_
/// and bounds, never defining values, and so the zero operation needs one explicit dimension operand per dynamic axis.
/// Also, the value that could supply those operands (i.e., the primal input the cotangent corresponds to) is _not an
/// input of the pullback program_ where the zero must be staged. The only moment both the need and the geometry are in
/// scope is during linearization, and so the required extents must be captured then and threaded to transposition as
/// ordinary residuals. This trait is that capture/spend contract, expressed once per operation *family*. It is a set of
/// associated functions with no receiver (i.e., `self` argument), because it is invoked precisely when no [`Operation`]
/// instance exists.
///
/// # How to Use It
///
/// The protocol has three steps, executed by the differentiation machinery rather than by implementors:
///
///   1. **Declare:** When linearization finds a differentiated input whose cotangent will be disconnected, it calls
///      [`Self::zero_residual_types`] with the zero's type to learn which residual values the eventual zero needs
///      (e.g., one first-class dimension per *distinct* dynamic identity for a composite array type, and nothing for
///      a static type).
///   2. **Capture:** While the primal value is still in scope, linearization records those residuals from it.
///      [`Self::capture_zero_residuals`] stages the reads into the program being built (i.e., the program-level
///      [`Program::linearize`](crate::Program::linearize) path), and [`Self::capture_zero_residual_values`] is its
///      value-level counterpart for reusable pullback callables that close over concrete or tracer values. The captured
///      residuals ride the tangent program's ordinary trailing residual suffix.
///   3. **Spend:** During transposition, [`Self::add_zero_from_residuals`] stages the zero inside the pullback
///      program, consuming exactly the residuals captured in step 2 as its explicit operands.
///
/// The three steps must agree on residual count and order. Every mismatch is a loud typed error (the capture sites
/// validate against the declared types, and the spend site validates the residual count), never a silently wrong-shaped
/// zero.
///
/// # Who Implements It
///
/// Almost nobody needs to implement this trait, by design. Every operation family with an input-free zero (i.e., every
/// family with a `From<ZeroOperation<T>>` conversion) receives the whole protocol through a blanket implementation that
/// declares nothing, captures nothing, and spends by constructing the type-only zero (i.e., the fail-loud default
/// rejects unexpected residuals rather than ignoring them, so a mismatched linearize/transpose pairing cannot be
/// silently accepted). Only families whose zero genuinely consumes runtime-geometry operands (e.g., the composite
/// program family and its XLA counterpart) override all four methods coherently.
///
/// [`LinearCallOperation`] below is this protocol's sibling. It retains residual geometry for the transpose of a
/// *non-trivial* residual-parameterized linear map by attaching explicit forward/transpose regions to an instruction,
/// while this trait retains it for the degenerate zero map, which has no instruction to attach anything to. Both exist
/// for the same reason (i.e., reverse mode needs geometry at a moment when its defining values would otherwise be out
/// of scope) and both keep residual selection and threading owned by the differentiation transform rather than leaking
/// into primal operation payloads.
pub trait ResidualZeroProvider<T: Type>: ZeroOperationProvider<T> {
    /// Returns the types of the residual values that a zero of `r#type` needs, in the exact order in which
    /// [`Self::capture_zero_residuals`] captures them and [`Self::add_zero_from_residuals`] consumes them. Input-free
    /// [`Operation`] families use the empty default. The array-dimension composite family returns one dimension type
    /// per _distinct_ dynamic identity of `r#type`, in first-occurrence order, so repeated axes share one residual.
    #[inline]
    fn zero_residual_types(_type: &T) -> Vec<T> {
        Vec::new()
    }

    /// Stages instructions into `builder` that read the residual values declared by [`Self::zero_residual_types`] from
    /// the primal value `source` (e.g., one `dimension_size` read per declared residual), returning the new atoms in
    /// declaration order. Linearization calls this while `source` is still in scope of the program being built. The
    /// returned atoms are then threaded to transposition as ordinary residuals. Input-free [`Operation`] families
    /// capture nothing.
    #[inline]
    fn capture_zero_residuals<V: Value<Type = T>>(
        _builder: &mut ProgramBuilder<V, Self>,
        _source: AtomId,
        _type: &T,
    ) -> Result<Vec<AtomId>, ProgramError> {
        Ok(Vec::new())
    }

    /// Captures the residual values declared by [`Self::zero_residual_types`] from the primal value `source`
    /// in a live `context`, returning them in declaration order. This is the value-level counterpart of
    /// [`Self::capture_zero_residuals`] used by reusable pullback callables, whose captured residuals are
    /// concrete values or tracers closed over by the callable rather than atoms of a program under construction.
    #[inline]
    fn capture_zero_residual_values<C: Context<Type = T, Operation = Self>>(
        _context: &C,
        _source: &C::Value,
        _type: &T,
    ) -> Result<Vec<C::Value>, ProgramError> {
        Ok(Vec::new())
    }

    /// Stages a zero of `r#type` into `builder`, consuming the `residuals` captured by [`Self::capture_zero_residuals`]
    /// as its explicit operands, in declaration order. Transposition calls this inside the pullback program, where the
    /// primal that supplied the residuals is no longer an input.
    fn add_zero_from_residuals<V: Value<Type = T>>(
        builder: &mut ProgramBuilder<V, Self>,
        r#type: T,
        residuals: &[AtomId],
    ) -> Result<AtomId, ProgramError> {
        // The default represents genuinely input-free zero families. Rejecting residuals rather than ignoring them
        // keeps a mismatched linearization/transposition boundary from being silently accepted.
        if !residuals.is_empty() {
            return Err(ProgramError::InvalidArgument {
                message: format!("input-free zero expected 0 residuals but got {}", residuals.len()),
            });
        }
        Ok(builder.add_instruction(Self::zero_operation(r#type)?, Vec::new(), Vec::new())?[0])
    }
}

// Every operation family that absorbs a type-only `ZeroOperation` has an input-free zero, and so the defaulted
// residual protocol applies verbatim. Composite families without that conversion implement the protocol directly.
impl<T: Type, O: Operation<T> + From<ZeroOperation<T>>> ResidualZeroProvider<T> for O {}

/// Interface form implemented by a [`LinearCallOperation`].
#[derive(Clone, Debug, PartialEq)]
enum LinearCallInterface<T: DifferentiableType> {
    /// Executable linear map. Both the map and its transpose have attached regions (i.e., `forward` followed by
    /// `transpose`), and the complete operation interface is derived from the `forward` [`Region`](crate::Region)'s
    /// boundary, so no [`Type`](crate::Type)s are stored.
    ForwardAndTranspose,

    /// Reverse-only linear map that supplies a transpose program but no executable forward program, so exactly one
    /// region is attached, named `transpose`. The forward map `u ↦ Lᵣ(u)` therefore exists mathematically but has
    /// no region boundary to derive the operation interface from, and its input and output types are stored here
    /// explicitly. Interpreting this form is deliberately an error (i.e., the canonical reverse-only diagnostic).
    /// The call exists in a linearized program only so that reverse mode can transpose it by replaying the attached
    /// [`Region`](crate::Region). For example, [`CustomVjpOperation`](crate::CustomVjpOperation) stages this form
    /// because `custom_vjp` supplies a user-written backward program without a tangent program. Refer to the
    /// documentation of [`LinearCallOperation`] for how this form relates to [`Self::ForwardAndTranspose`].
    TransposeOnly {
        /// Input [`Type`](crate::Type)s of the unavailable forward map (one per linear operand).
        input_types: Vec<T>,

        /// Output [`Type`](crate::Type)s of the unavailable forward map.
        output_types: Vec<T>,
    },
}

/// [`Operation`] that represents a call to a residual-parameterized linear map together with its transpose. For
/// fixed residual values `r`, this operation represents a linear map `Lᵣ : U → V` and its transpose `Lᵣᵀ : V* → U*`.
/// Linearity applies only to `u ∈ U`: for scalars `a` and `b`, `Lᵣ(a · u₁ + b · u₂) = a · Lᵣ(u₁) + b · Lᵣ(u₂)`. The
/// residuals `r` are fixed primal values that parameterize the map, so transposition produces cotangents for `u` but
/// neither differentiates nor accumulates cotangents for `r`. Its two region interfaces are deliberately symmetric:
///
///   - `forward`: `(r, u) ↦ v = Lᵣ(u)`, and
///   - `transpose`: `(r, v̄) ↦ ū = Lᵣᵀ(v̄)`.
///
/// Operation operands are ordered as `[residuals..., linear_inputs...]`, matching both region interfaces, and
/// [`residual_count`](Self::residual_count) separates the two operand roles. That symmetry makes transposition a
/// *swap*: the transpose of `Lᵣ` staged with regions `(forward, transpose)` is the same operation staged with the
/// regions `(transpose, forward)` over `[residuals..., output_cotangents...]`, so transposing twice restores the
/// original call and a pullback program retains the linear-call boundary (and its explicit residual edges) instead
/// of dissolving it. This mirrors the transpose rule of [JAX's
/// `jax.custom_derivatives.linear_call`](https://github.com/jax-ml/jax/blob/main/jax/_src/custom_derivatives.py),
/// which has no rendered documentation page and is therefore linked at its source. The swap re-derives each side's
/// expected interface with the cotangent type mapping, which requires `cotangent(cotangent(u)) = u` for the linear
/// types. Tangent types (the only types linearization rules stage as linear operands) satisfy this even where primal
/// storage types do not (e.g., `f8e8m0fnu`, whose tangent and cotangent representations are both `f32`).
///
/// Every residual is an ordinary typed Single Static Assignment (SSA) edge rather than differentiation-only payload
/// metadata. Partial evaluation can lift those values into the enclosing [`Linearization`](crate::Linearization)
/// residual environment, and partition-aware transposition receives the same values as known operands in
/// deterministic order.
///
/// An explicit operation boundary is necessary because a tangent program must retain more than the computation of
/// `Lᵣ(u)`. After linearization, that program may be cloned, imported, simplified, differentiated again, or transposed
/// independently of the primal program. Merely inlining the forward computation would lose its association with both
/// the residual values and the program that implements `Lᵣᵀ`. Attaching both regions to this operation makes that
/// association survive through the ordinary [`Program`](crate::Program) region, operand, identity-renaming, and import
/// machinery, without an ambient value lookup or side table.
///
/// The dynamic reshape operation illustrates the need for this representation. Its tangent map reshapes an input
/// tangent using the output extents, whereas its transpose reshapes an output cotangent using the original input
/// extents. Those input extents cannot always be recovered from the output shape (e.g., `[n, 4] → [2, 2·n]`). The
/// reshape rule therefore carries both sets of extents as explicit residual operands and attaches forward and inverse
/// reshape regions to one [`LinearCallOperation`]. Other shape-dependent linear rules can use the same mechanism; the
/// operation itself has no array- or dimension-specific semantics.
///
/// The _forward-and-transpose_ form of this operation derives its complete interface from attached `forward` and
/// `transpose` [`Region`](crate::Region)s and can be interpreted, lowered, transposed, and differentiated again. The
/// _transpose-only_ form is the deliberate exception for linear maps that supply only a reverse rule: it stores the
/// unavailable forward input/output types and attaches only the transpose region, so attempting to execute or
/// differentiate it in forward mode is an error (e.g., [`CustomVjpOperation`](crate::CustomVjpOperation) stages
/// this form because it supplies a backward program without a tangent program).
#[derive(Clone, Debug, PartialEq)]
pub struct LinearCallOperation<T: DifferentiableType> {
    /// Number of leading residual operands.
    residual_count: usize,

    /// Specifies whether this call has an executable forward program or is a reverse-only transpose-only map.
    interface: LinearCallInterface<T>,
}

impl<T: DifferentiableType> LinearCallOperation<T> {
    /// Creates a _forward-and-transpose_ [`LinearCallOperation`] with two attached regions,
    /// `forward` and `transpose`.
    #[inline]
    pub fn new(residual_count: usize) -> Self {
        Self { residual_count, interface: LinearCallInterface::ForwardAndTranspose }
    }

    /// Creates a reverse-only _transpose-only_ [`LinearCallOperation`] for a linear map that supplies a transpose
    /// program but no executable forward program, stating the unavailable forward map's interface explicitly.
    #[inline]
    pub fn transpose_only(residual_count: usize, input_types: Vec<T>, output_types: Vec<T>) -> Self {
        Self { residual_count, interface: LinearCallInterface::TransposeOnly { input_types, output_types } }
    }

    /// Returns the number of leading residual operands.
    #[inline]
    pub fn residual_count(&self) -> usize {
        self.residual_count
    }

    /// Returns `true` if this call has the _transpose-only_ form, which attaches no executable forward
    /// [`Region`](crate::Region).
    #[inline]
    pub fn is_transpose_only(&self) -> bool {
        matches!(self.interface, LinearCallInterface::TransposeOnly { .. })
    }

    /// Splits the provided `input_types` into its leading residual types and trailing linear types.
    fn split_inputs<'a>(&self, input_types: &'a [T]) -> Result<(&'a [T], &'a [T]), TypeError> {
        if self.residual_count > input_types.len() {
            return Err(TypeError::invalid(format!(
                "linear call residual count {} exceeds input count {}",
                self.residual_count,
                input_types.len(),
            )));
        }
        Ok(input_types.split_at(self.residual_count))
    }

    /// Stages a [`LinearCallInterface::ForwardAndTranspose`] residual-parameterized [`LinearCallOperation`].
    /// It traces its two attached [`Region`](crate::Region)s and binds the resulting _forward-and-transpose_
    /// [`LinearCallOperation`] in `context`. This is a canonical constructor rather than a separate binding path as the
    /// final step is still an ordinary [`Context::bind`], and everything before it only constructs the operation's two
    /// region programs. It exists because every producer of an [`LinearCallInterface::ForwardAndTranspose`] linear call
    /// must uphold the same boundary convention, which is easy to get subtly wrong at any one of many call sites:
    ///
    ///   - the operands are ordered as `[residuals..., linear_inputs...]`,
    ///   - the `forward` region receives the same values in the same order,
    ///   - the `transpose` region receives the same residuals followed by one cotangent-typed input
    ///     per traced forward output, and
    ///   - the operation carries its regions in `[forward, transpose]` order, with
    ///     [`residual_count`](Self::residual_count) separating the two operand roles.
    ///
    /// Callers therefore provide only the operation-specific mathematics of the two maps, as closures over tracers that
    /// are already split into their residual and linear/cotangent groups.
    ///
    /// The _transpose-only_ form (i.e., [`LinearCallInterface::TransposeOnly`]) deliberately has no staging counterpart
    /// (e.g., an `Option`al `transpose_fn` or a `stage_transpose_only` sibling) as it shares _none_ of this function's
    /// mechanics, because there is no forward map to trace and therefore no traced outputs to derive the transpose
    /// boundary from (its interface types are stated explicitly instead).
    ///
    /// # Parameters
    ///
    ///   - `context`: [`Context`] in which the linear call is staged.
    ///   - `residuals`: Fixed primal values parameterizing both regions.
    ///   - `linear_inputs`: Tangent values to which the forward linear map is applied.
    ///   - `forward`: Function that builds the forward map `v = Lᵣ(u)` from `(residuals, linear_inputs)`.
    ///   - `transpose`: Function that builds the transpose map `ū = Lᵣᵀ(v̄)` from `(residuals, output_cotangents)`.
    pub(crate) fn stage<
        C: Context<Type = T, Operation: From<LinearCallOperation<T>>>,
        ForwardFn: FnOnce(
            &[Tracer<NestedTracingContext<C>>],
            &[Tracer<NestedTracingContext<C>>],
        ) -> Result<Vec<Tracer<NestedTracingContext<C>>>, ProgramError>,
        TransposeFn: FnOnce(
            &[Tracer<NestedTracingContext<C>>],
            &[Tracer<NestedTracingContext<C>>],
        ) -> Result<Vec<Tracer<NestedTracingContext<C>>>, ProgramError>,
    >(
        context: &C,
        residuals: Vec<C::Value>,
        linear_inputs: Vec<C::Value>,
        forward_fn: ForwardFn,
        transpose_fn: TransposeFn,
    ) -> Result<Vec<C::Value>, ProgramError> {
        let residual_count = residuals.len();
        let forward_input_types =
            residuals.iter().chain(&linear_inputs).map(|value| value.r#type().into_owned()).collect();
        let (_, forward) = NestedTracingContext::trace(
            context.clone(),
            move |inputs| {
                let (residuals, linear_inputs) = inputs.split_at(residual_count);
                forward_fn(residuals, linear_inputs)
            },
            forward_input_types,
        )?;
        let transpose_input_types = residuals
            .iter()
            .map(|value| value.r#type().into_owned())
            .chain(forward.outputs().map(|output| output.r#type().into_owned().cotangent()))
            .collect();
        let (_, transpose) = NestedTracingContext::trace(
            context.clone(),
            move |inputs| {
                let (residuals, output_cotangents) = inputs.split_at(residual_count);
                transpose_fn(residuals, output_cotangents)
            },
            transpose_input_types,
        )?;
        let mut inputs = residuals;
        inputs.extend(linear_inputs);
        context.bind(Self::new(residual_count), vec![forward, transpose], inputs.as_slice())
    }

    /// Batches an invocation to this [`LinearCallOperation`] by structurally batching its two attached
    /// [`Region`](crate::Region)s. Both [`LinearCallInterface`] forms preserve a completely replicated call unchanged.
    ///
    /// A mapped [`LinearCallInterface::ForwardAndTranspose`] call batches its forward region to discover the batched
    /// output axes, batches its transpose region under those axes, and aligns the transpose outputs with the original
    /// linear-input axes (i.e., a cotangent for a replicated linear input is summed across the batch by `collapse_fn`,
    /// while cotangents for mapped linear inputs retain their packed axes).
    ///
    /// A mapped [`LinearCallInterface::TransposeOnly`] call is rejected because its unavailable forward program cannot
    /// determine the batched output axes.
    ///
    /// The batching policy owns the boundary shape of its structurally batched programs.
    /// [`BatchingPolicy::adapt_batched_program`] adapts each batched region to the plain two-region linear-call
    /// boundary, and any [`BatchingPolicy::boundary_operands`] (e.g., a composite program's first-class mapped extent)
    /// become additional leading residuals of the batched call.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active [`BatchingContext`] for the transform level being applied.
    ///   - `driver`: [`BatchingDriver`] exposing this call's attached regions.
    ///   - `inputs`: Batched operands, ordered as `[residuals..., linear_inputs...]`.
    ///   - `input_axes`: Batch axis of each operand in `inputs`.
    ///   - `collapse_fn`: Function that sums one mapped transpose output along the provided axis within the adapted
    ///     program's own [`TracingContext`]. Summation is the correct collapse here because these outputs are
    ///     cotangents (i.e., a replicated linear input `u` was broadcast across the batch, the batched transpose
    ///     therefore produced one cotangent `ūᵢ` per batch item, and the transpose of a broadcast is a summation,
    ///     so the one shared cotangent is `ū = Σᵢ ūᵢ`). Callers own this closure only because its mechanics are
    ///     universe-specific (e.g., the homogeneous tracer has the direct reduction capability, while the composite
    ///     universe reaches it through a projected value).
    pub(crate) fn batch_regions<
        C: Context<Type = T, Operation: From<LinearCallOperation<T>>>,
        P: BatchingPolicy<C>,
        D: BatchingDriver<C, P>,
        CollapseFn: Fn(
            &TracingContext<C::Constant, C::Operation>,
            Tracer<TracingContext<C::Constant, C::Operation>>,
            Axis,
        ) -> Result<Tracer<TracingContext<C::Constant, C::Operation>>, BatchingError>,
    >(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
        input_axes: Vec<BatchAxis>,
        collapse_fn: CollapseFn,
    ) -> Result<Vec<P::Batch>, BatchingError> {
        if self.residual_count > inputs.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "linear call residual count {} exceeds input count {}",
                self.residual_count,
                inputs.len(),
            ))
            .into());
        }
        check_count!("input", input_axes, inputs.len(), ProgramError);
        let input_values = inputs.iter().map(P::value).cloned().collect::<Vec<_>>();

        // A completely replicated call needs no structural region rewrite. Keeping the original call also avoids
        // manufacturing a batch axis that neither region observes.
        if input_axes.iter().all(BatchAxis::is_replicated) {
            let outputs = context.parent().bind(
                self.clone(),
                driver.regions().map(|region| region.to_program()).collect::<Vec<_>>(),
                input_values.as_slice(),
            )?;
            return Ok(outputs.into_iter().map(P::replicated).collect());
        }

        match &self.interface {
            LinearCallInterface::ForwardAndTranspose => {}
            LinearCallInterface::TransposeOnly { .. } => {
                return Err(BatchingError::UnsupportedOperation {
                    message: "a transpose-only linear call cannot be batched with mapped inputs because its \
                             unavailable forward program does not determine output batch axes"
                        .to_string(),
                });
            }
        }

        let (residual_axes, linear_axes) = input_axes.split_at(self.residual_count);
        let (forward, output_axes) = P::adapt_batched_program(
            driver.batch_program(
                context,
                driver.region(0)?,
                input_axes.as_slice(),
                ProgramBatchingOutputAxesPolicy::Natural,
            )?,
            None,
            &collapse_fn,
        )?
        .into_parts();

        // Batch the supplied transpose under the forward result axes. Cotangents for replicated linear inputs are
        // summed during adaptation; mapped linear inputs instead retain their packed axes.
        let transpose_input_axes = residual_axes.iter().copied().chain(output_axes.iter().copied()).collect::<Vec<_>>();
        let (transpose, transpose_output_axes) = P::adapt_batched_program(
            driver.batch_program(
                context,
                driver.region(1)?,
                transpose_input_axes.as_slice(),
                ProgramBatchingOutputAxesPolicy::AlignEachTo(linear_axes.to_vec()),
            )?,
            Some(linear_axes),
            &collapse_fn,
        )?
        .into_parts();
        check_count!("output", transpose_output_axes, linear_axes.len(), ProgramError);
        check_count!("output", transpose.output_ids(), linear_axes.len(), ProgramError);

        let boundary_operands = P::boundary_operands(context.axis_extent());
        let mut packed_inputs = Vec::with_capacity(boundary_operands.len() + input_values.len());
        let boundary_operand_count = boundary_operands.len();
        packed_inputs.extend(boundary_operands);
        packed_inputs.extend(input_values);
        context
            .parent()
            .bind(
                LinearCallOperation::new(self.residual_count + boundary_operand_count),
                vec![forward, transpose],
                packed_inputs.as_slice(),
            )?
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| P::batch(output, axis))
            .collect()
    }
}

impl<T: DifferentiableType> Display for LinearCallOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: DifferentiableType> Operation<T> for LinearCallOperation<T> {
    #[inline]
    fn name(&self) -> &'static str {
        // The two forms render under distinct names so rendered programs and the diagnostics built from this name
        // distinguish a reverse-only call from an executable one without inspecting region counts.
        if self.is_transpose_only() { "transpose_only_linear_call" } else { "linear_call" }
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        if self.is_transpose_only() {
            const { &[RegionSlot::rule("transpose")] }
        } else {
            const { &[RegionSlot::computation("forward"), RegionSlot::rule("transpose")] }
        }
    }

    fn infer_region_input_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<Option<Vec<T>>>, TypeError> {
        match &self.interface {
            LinearCallInterface::ForwardAndTranspose => {
                check_count!("region", region_interfaces, 2, TypeError);
                let forward = &region_interfaces[0];
                let renaming = T::derive_identity_renaming(forward.input_types(), input_types)?;
                let output_types = forward
                    .output_types()
                    .iter()
                    .map(|r#type| r#type.rename_identities(&renaming))
                    .collect::<Result<Vec<_>, _>>()?;
                let (residual_types, _) = self.split_inputs(input_types)?;
                let transpose_input_types = residual_types
                    .iter()
                    .cloned()
                    .chain(output_types.iter().map(DifferentiableType::cotangent))
                    .collect();
                Ok(vec![Some(input_types.to_vec()), Some(transpose_input_types)])
            }
            LinearCallInterface::TransposeOnly { output_types, .. } => {
                check_count!("region", region_interfaces, 1, TypeError);
                let (residual_types, _) = self.split_inputs(input_types)?;
                Ok(vec![Some(
                    residual_types
                        .iter()
                        .cloned()
                        .chain(output_types.iter().map(DifferentiableType::cotangent))
                        .collect(),
                )])
            }
        }
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        match &self.interface {
            LinearCallInterface::ForwardAndTranspose => {
                check_count!("region", region_interfaces, 2, TypeError);
                let forward = &region_interfaces[0];
                let transpose = &region_interfaces[1];
                check_types!(@same, "linear call forward input", [input_types, forward.input_types()]);
                let (residual_types, linear_types) = self.split_inputs(input_types)?;
                let transpose_input_types = residual_types
                    .iter()
                    .cloned()
                    .chain(forward.output_types().iter().map(DifferentiableType::cotangent))
                    .collect::<Vec<_>>();
                let transpose_output_types = linear_types.iter().map(DifferentiableType::cotangent).collect::<Vec<_>>();
                check_types!(@same, "linear call transpose input", [
                    &transpose_input_types,
                    transpose.input_types(),
                ]);
                check_types!(@same, "linear call transpose output", [
                    &transpose_output_types,
                    transpose.output_types(),
                ]);
                Ok(forward.output_types().to_vec())
            }
            LinearCallInterface::TransposeOnly { input_types: linear_types, output_types } => {
                check_count!("region", region_interfaces, 1, TypeError);
                let transpose = &region_interfaces[0];
                let (residual_types, actual_linear_types) = self.split_inputs(input_types)?;
                check_types!(@same, "transpose-only linear call input", [linear_types, actual_linear_types]);
                let transpose_input_types = residual_types
                    .iter()
                    .cloned()
                    .chain(output_types.iter().map(DifferentiableType::cotangent))
                    .collect::<Vec<_>>();
                let transpose_output_types = linear_types.iter().map(DifferentiableType::cotangent).collect::<Vec<_>>();
                check_types!(@same, "transpose-only linear call transpose input", [
                    &transpose_input_types,
                    transpose.input_types(),
                ]);
                check_types!(@same, "transpose-only linear call transpose output", [
                    &transpose_output_types,
                    transpose.output_types(),
                ]);
                Ok(output_types.clone())
            }
        }
    }

    #[inline]
    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        // The forward-and-transpose form's outputs are exactly its forward region's outputs.
        // The transpose-only form has no forward region and therefore no region provenance.
        if self.is_transpose_only() {
            Vec::new()
        } else {
            vec![OutputRegionProvenance { region_index: 0, output_index }]
        }
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("residual_count", self.residual_count))
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<T::Identity>) -> Result<Self, TypeError> {
        Ok(Self {
            residual_count: self.residual_count,
            interface: match &self.interface {
                LinearCallInterface::ForwardAndTranspose => LinearCallInterface::ForwardAndTranspose,
                LinearCallInterface::TransposeOnly { input_types, output_types } => {
                    LinearCallInterface::TransposeOnly {
                        input_types: input_types
                            .iter()
                            .map(|r#type| r#type.rename_identities(renaming))
                            .collect::<Result<Vec<_>, _>>()?,
                        output_types: output_types
                            .iter()
                            .map(|r#type| r#type.rename_identities(renaming))
                            .collect::<Result<Vec<_>, _>>()?,
                    }
                }
            },
        })
    }
}

impl<C: Domain<Type: DifferentiableType>> InterpretableOperation<C> for LinearCallOperation<C::Type> {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        if self.is_transpose_only() {
            return Err(ProgramError::UnsupportedOperation {
                message: "a transpose-only linear call has no forward program to execute; it supports only \
                          reverse-mode differentiation (e.g., 'vjp', 'value_and_gradient', or 'jacobian_reverse')"
                    .to_string(),
            });
        }
        driver.interpret_region(context, 0, inputs.to_vec())
    }
}

impl<C: Context<Type: DifferentiableType, Operation: From<LinearCallOperation<C::Type>>>>
    PartiallyEvaluatableOperation<C> for LinearCallOperation<C::Type>
{
}

impl<
    C: Context<Type = ArrayType, Operation: From<LinearCallOperation<ArrayType>> + From<ReduceOperation>>,
    P: ArrayBatchingPolicy<C>,
> BatchableOperation<C, ArrayBatching<P>> for LinearCallOperation<ArrayType>
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        let input_axes = inputs.iter().map(ArrayBatch::batch_axis).collect::<Vec<_>>();
        self.batch_regions(context, driver, inputs, input_axes, |_, output, axis| {
            Ok(output.reduce(
                &[axis.normalize(output.r#type().rank()).map_err(|_| BatchingError::BatchAxisOutOfBounds {
                    r#type: Box::new(output.r#type().into_owned()),
                    axis,
                })?],
                ReductionKind::Sum,
            ))
        })
    }
}

impl<C: Context<Type: DifferentiableType> + Zero<C::Value>> DifferentiableOperation<C>
    for LinearCallOperation<C::Type>
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        if self.is_transpose_only() {
            return Err(ProgramError::UnsupportedOperation {
                message: "a transpose-only linear call has no forward-mode (JVP) rule; it supports only \
                          reverse-mode differentiation"
                    .to_string(),
            }
            .into());
        }

        inputs.len().checked_sub(self.residual_count).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "linear call residual count {} exceeds input count {}",
                self.residual_count,
                inputs.len(),
            ))
        })?;

        let forward = driver.region(0)?;
        let primals = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        if inputs.iter().all(|input| input.tangent().is_zero()) {
            let primal_outputs = forward.interpret_in_context(context, primals)?;
            return Ok(primal_outputs.into_iter().map(DifferentiationDual::new_with_zero_tangent).collect());
        }

        // Higher-order differentiation must include the dependence of the linear map on its residual parameters.
        // Replay the ordinary fused JVP of the attached forward region instead of assuming residual tangents are zero.
        let jvp = driver.jvp_program(forward)?;
        let mut jvp_inputs = primals;
        for input in inputs {
            if !input.tangent().r#type().is_zero_space() {
                jvp_inputs.push(input.tangent().clone().materialize(context)?);
            }
        }

        let mut outputs = jvp.interpret_in_context(context, jvp_inputs)?;
        let output_types = forward.output_types();
        let tangent_outputs = outputs.split_off(output_types.len());
        let mut tangent_outputs = tangent_outputs.into_iter();
        outputs
            .into_iter()
            .zip(output_types)
            .map(|(primal, output_type)| {
                if output_type.tangent().is_zero_space() {
                    Ok(DifferentiationDual::new_with_zero_tangent(primal))
                } else {
                    DifferentiationDual::new(primal, tangent_outputs.next().unwrap())
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }
}

impl<
    V: Value<Type: DifferentiableType>,
    O: Operation<V::Type> + ZeroOperationProvider<V::Type> + From<LinearCallOperation<V::Type>>,
> TransposableOperation<V, O> for LinearCallOperation<V::Type>
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        if self.residual_count > inputs.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "linear call residual count {} exceeds input count {}",
                self.residual_count,
                inputs.len(),
            ))
            .into());
        }
        let (residual_inputs, linear_inputs) = inputs.split_at(self.residual_count);
        let residuals = residual_inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                input.as_known().cloned().ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "linear call residual operand {index} is not known during transposition",
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        if outputs.iter().all(MaybeZero::is_zero) {
            return Ok(inputs.iter().map(|input| MaybeZero::Zero(input.r#type().cotangent())).collect());
        }

        // Classify each transpose-region output as structurally zero by inspecting its producing instruction in the
        // source region, so zero cotangents stay symbolic instead of accumulating. Outputs with no producing
        // instruction (forwarded region inputs and constants) conservatively classify as nonzero. The transpose
        // region follows the forward-and-transpose form's leading forward region (index 1) and is the transpose-only
        // form's only region (index 0).
        let transpose = driver.region(if self.is_transpose_only() { 0 } else { 1 })?;
        let output_is_zero = transpose
            .output_ids()
            .iter()
            .map(|output| {
                transpose
                    .instructions()
                    .iter()
                    .find_map(|instruction| {
                        instruction
                            .outputs()
                            .iter()
                            .position(|candidate| candidate == output)
                            .map(|output_index| instruction.operation().is_zero(output_index))
                    })
                    .unwrap_or(false)
            })
            .collect::<Vec<_>>();

        let mut transpose_inputs = residuals;
        transpose_inputs
            .extend(outputs.iter().cloned().map(|output| output.materialize(context)).collect::<Result<Vec<_>, _>>()?);
        let input_cotangents = if self.is_transpose_only() {
            // The transpose-only form's region is a user-supplied backward program with no linearity contract of its
            // own, so it cannot be re-transposed and is replayed inline into the pullback.
            transpose.interpret_in_context(context, transpose_inputs)?
        } else {
            // The forward-and-transpose form transposes by *swapping* its regions: the pullback stages the same
            // operation with the transpose region leading, over the same residuals followed by the output
            // cotangents. Transposing twice therefore restores the original call, and the pullback retains the
            // linear-call boundary together with its explicit residual edges.
            let swapped = LinearCallOperation::new(self.residual_count);
            let transpose_program = transpose.to_program();
            let forward_program = driver.region(0)?.to_program();
            context.bind(swapped, vec![transpose_program, forward_program], &transpose_inputs)?
        };
        check_count!("output", input_cotangents, linear_inputs.len(), ProgramError);

        let mut cotangents =
            residual_inputs.iter().map(|input| MaybeZero::Zero(input.r#type().cotangent())).collect::<Vec<_>>();
        cotangents.extend(linear_inputs.iter().zip(input_cotangents.into_iter().zip(output_is_zero)).map(
            |(input, (cotangent, is_zero))| {
                if input.is_unknown() {
                    if is_zero { MaybeZero::Zero(cotangent.r#type().into_owned()) } else { MaybeZero::Value(cotangent) }
                } else {
                    MaybeZero::Zero(input.r#type().cotangent())
                }
            },
        ));
        Ok(cotangents)
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::batching::{BatchAxis, ProgramBatchingOutputAxesPolicy};
    use crate::contexts::StagingContext;
    use crate::contexts::tests::{
        ProjectedMemberType, ProjectedMemberValue, ProjectedProgramType, ProjectedProgramValue,
    };
    use crate::differentiation::DifferentiationError;
    use crate::differentiation::reverse::{TransposableOperation, TranspositionDriver};
    use crate::operations::math::{AddOperation, MulOperation};
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::ProgramError;
    use crate::programs::atoms::MaybeZero;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::programs::Program;
    use crate::programs::regions::{RegionDriver, RegionRef, RegionSlot};
    use crate::sharding::ShardingDimension;
    use crate::tracing::TracingContext;
    use crate::types::{ArrayType, DataType};

    use super::*;

    /// Builds the scalar program `(r, u) ↦ r · u` used as a residual-parameterized linear-map region.
    fn scalar_multiply_program() -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let r#type = ArrayType::scalar(DataType::F64);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let residual = builder.add_input(r#type.clone());
        let linear = builder.add_input(r#type);
        let output = builder.add_instruction(MulOperation, Vec::new(), vec![linear, residual]).unwrap()[0];
        builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap()
    }

    #[test]
    fn test_linear_call_operation() {
        let r#type = ArrayType::scalar(DataType::F64);

        // The forward-and-transpose form derives its interface from two attached regions.
        let operation = LinearCallOperation::<ArrayType>::new(1);
        assert_eq!(operation.residual_count(), 1);
        assert!(!operation.is_transpose_only());
        assert_eq!(operation.to_string(), "linear_call [residual_count=1]");
        assert_eq!(operation.region_slots(), &[RegionSlot::computation("forward"), RegionSlot::rule("transpose")]);
        assert_eq!(
            format!("{operation:?}"),
            "LinearCallOperation { residual_count: 1, interface: ForwardAndTranspose }",
        );

        // The transpose-only form stores the unavailable forward interface and renders under its distinct name.
        let operation = LinearCallOperation::transpose_only(1, vec![r#type.clone()], vec![r#type]);
        assert_eq!(operation.residual_count(), 1);
        assert!(operation.is_transpose_only());
        assert_eq!(operation.to_string(), "transpose_only_linear_call [residual_count=1]");
        assert_eq!(operation.region_slots(), &[RegionSlot::rule("transpose")]);
    }

    #[test]
    fn test_linear_call_operation_supports_a_third_composite_member() {
        // A third-member residual paired with a first-member linear value proves that the operation and its attached
        // regions use generic program storage without an array/dimension-specific projection.
        let linear_type = ProjectedProgramType::First(ProjectedMemberType);
        let residual_type = ProjectedProgramType::Third(ProjectedMemberType);
        let forward = {
            let mut builder = ProgramBuilder::<ProjectedProgramValue, LinearCallOperation<ProjectedProgramType>>::new();
            builder.add_input(residual_type.clone());
            let linear = builder.add_input(linear_type.clone());
            builder
                .build::<Vec<ProjectedProgramValue>, Vec<ProjectedProgramValue>>(
                    vec![linear],
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let transpose = {
            let mut builder = ProgramBuilder::<ProjectedProgramValue, LinearCallOperation<ProjectedProgramType>>::new();
            builder.add_input(residual_type.clone());
            let cotangent = builder.add_input(linear_type.clone());
            builder
                .build::<Vec<ProjectedProgramValue>, Vec<ProjectedProgramValue>>(
                    vec![cotangent],
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<ProjectedProgramValue, LinearCallOperation<ProjectedProgramType>>::new();
        let forward = builder.import_region(forward.entry_region_ref());
        let transpose = builder.import_region(transpose.entry_region_ref());
        let residual = builder.add_input(residual_type);
        let linear = builder.add_input(linear_type);
        let output = builder
            .add_instruction(LinearCallOperation::new(1), vec![forward, transpose], vec![residual, linear])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ProjectedProgramValue>, Vec<ProjectedProgramValue>>(
                vec![output],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();

        let linear = ProjectedProgramValue::First(ProjectedMemberValue(2));
        let residual = ProjectedProgramValue::Third(ProjectedMemberValue(7));
        assert_eq!(program.interpret(vec![residual, linear.clone()]), Ok(vec![linear]));
        assert_eq!(program.instructions()[0].regions().len(), 2);
    }

    #[test]
    fn test_linear_call_operation_transpose_only_form_rejects_forward_execution() {
        let r#type = ArrayType::scalar(DataType::F64);
        let transpose = scalar_multiply_program();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let transpose = builder.import_region(transpose.entry_region_ref());
        let residual = builder.add_input(r#type.clone());
        let linear = builder.add_input(r#type.clone());
        let output = builder
            .add_instruction(
                LinearCallOperation::transpose_only(1, vec![r#type.clone()], vec![r#type]),
                vec![transpose],
                vec![residual, linear],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        assert!(matches!(
            program.interpret(vec![Array::scalar(2.0), Array::scalar(3.0)]),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "a transpose-only linear call has no forward program to execute; it supports only \
                               reverse-mode differentiation (e.g., 'vjp', 'value_and_gradient', or \
                               'jacobian_reverse')",
        ));
        assert!(matches!(
            program.batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            ),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "a transpose-only linear call cannot be batched with mapped inputs because its \
                               unavailable forward program does not determine output batch axes",
        ));
    }

    #[test]
    fn test_linear_call_operation_batching_preserves_its_transpose() {
        let r#type = ArrayType::scalar(DataType::F64);
        let forward = scalar_multiply_program();
        let transpose = forward.clone();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let forward = builder.import_region(forward.entry_region_ref());
        let transpose = builder.import_region(transpose.entry_region_ref());
        let residual = builder.add_input(r#type.clone());
        let linear = builder.add_input(r#type);
        let output = builder
            .add_instruction(LinearCallOperation::new(1), vec![forward, transpose], vec![residual, linear])
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // A varying residual with a shared linear input produces mapped outputs. Its transpose must sum the per-item
        // cotangents back into the single shared input cotangent.
        let (batched, output_axes) = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            batched.interpret(vec![Array::vector(vec![2.0, 3.0]), Array::scalar(4.0)]),
            Ok(vec![Array::vector(vec![8.0, 12.0])]),
        );

        let instruction = &batched.instructions()[0];
        let transpose = batched.region_ref(instruction.regions()[1]).unwrap().to_program();
        assert_eq!(
            transpose.interpret(vec![Array::vector(vec![2.0, 3.0]), Array::vector(vec![5.0, 7.0])]),
            Ok(vec![Array::scalar(31.0)]),
        );

        // A replicated residual and mapped linear input preserve the mapped cotangent instead of reducing it.
        let (batched_linear, output_axes) = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            batched_linear.interpret(vec![Array::scalar(2.0), Array::vector(vec![3.0, 4.0])]),
            Ok(vec![Array::vector(vec![6.0, 8.0])]),
        );
        let instruction = &batched_linear.instructions()[0];
        let transpose = batched_linear.region_ref(instruction.regions()[1]).unwrap().to_program();
        assert_eq!(
            transpose.interpret(vec![Array::scalar(2.0), Array::vector(vec![5.0, 7.0])]),
            Ok(vec![Array::vector(vec![10.0, 14.0])]),
        );

        // Rebatching an executable linear call composes mapped axes without losing either attached region.
        let (nested, output_axes) = batched_linear
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            nested.interpret(vec![Array::vector(vec![2.0, 3.0]), Array::matrix(2, 2, vec![4.0, 5.0, 6.0, 7.0]),]),
            Ok(vec![Array::matrix(2, 2, vec![8.0, 10.0, 18.0, 21.0])]),
        );
    }

    #[test]
    fn test_linear_call_operation_batching_preserves_a_replicated_transpose_only_call() {
        let r#type = ArrayType::scalar(DataType::F64);
        let transpose = scalar_multiply_program();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let transpose = builder.import_region(transpose.entry_region_ref());
        let residual = builder.add_input(r#type.clone());
        let linear = builder.add_input(r#type.clone());
        let output = builder
            .add_instruction(
                LinearCallOperation::transpose_only(1, vec![r#type.clone()], vec![r#type]),
                vec![transpose],
                vec![residual, linear],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // A completely replicated call needs no structural region rewrite, so batching is the one transform the
        // transpose-only form supports: the call rebinds itself over its untouched backward region.
        let (batched, output_axes) = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::replicated()]);
        let instruction = &batched.instructions()[0];
        assert_eq!(instruction.operation().name(), "transpose_only_linear_call");
        assert_eq!(instruction.regions().len(), 1);
    }

    #[test]
    fn test_linear_call_operation_nested_jvp_differentiates_residual_parameters() {
        let r#type = ArrayType::scalar(DataType::F64);
        let forward = scalar_multiply_program();
        let transpose = forward.clone();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let forward = builder.import_region(forward.entry_region_ref());
        let transpose = builder.import_region(transpose.entry_region_ref());
        let residual = builder.add_input(r#type.clone());
        let linear = builder.add_input(r#type);
        let output = builder
            .add_instruction(LinearCallOperation::new(1), vec![forward, transpose], vec![residual, linear])
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();

        // For `Lᵣ(u) = r · u`, the nested JVP is `(r, u, ṙ, u̇) ↦ (r · u, ṙ · u + r · u̇)`. At
        // `(r, u, ṙ, u̇) = (2, 3, 5, 7)`, the primal is `6` and the tangent is `5 · 3 + 2 · 7 = 29`.
        assert_eq!(
            program.jvp().unwrap().interpret(vec![
                Array::scalar(2.0),
                Array::scalar(3.0),
                Array::scalar(5.0),
                Array::scalar(7.0),
            ]),
            Ok(vec![Array::scalar(6.0), Array::scalar(29.0)]),
        );
    }

    #[test]
    fn test_linear_call_operation_transposition_swaps_the_attached_regions() {
        /// Exposes an executable linear call's two regions directly to its transposition rule.
        struct TwoRegionDriver<'r> {
            forward: RegionRef<'r, Array, ArrayOperation<Array>>,
            transpose: RegionRef<'r, Array, ArrayOperation<Array>>,
        }

        impl RegionDriver<Array, ArrayOperation<Array>> for TwoRegionDriver<'_> {
            fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, Array, ArrayOperation<Array>>>
            where
                Array: 'r,
                ArrayOperation<Array>: 'r,
            {
                [self.forward, self.transpose].into_iter()
            }
        }

        impl TranspositionDriver<Array, ArrayOperation<Array>> for TwoRegionDriver<'_> {
            fn transpose_program(
                &self,
                _region: RegionRef<'_, Array, ArrayOperation<Array>>,
                _input_linearity: &[bool],
            ) -> Result<Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>, DifferentiationError>
            {
                unreachable!("linear call transposition swaps regions and never re-enters transposition")
            }
        }

        let r#type = ArrayType::scalar(DataType::F64);
        let forward = scalar_multiply_program();
        let transpose = {
            // Deliberately use a different body from `forward` so the rendered program proves the region order was
            // swapped instead of merely proving that two indistinguishable regions remain attached.
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let residual = builder.add_input(r#type.clone());
            let output_cotangent = builder.add_input(r#type.clone());
            let input_cotangent =
                builder.add_instruction(AddOperation, Vec::new(), vec![output_cotangent, residual]).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![input_cotangent], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let driver = TwoRegionDriver { forward: forward.entry_region_ref(), transpose: transpose.entry_region_ref() };

        // A structural-zero output cotangent returns structural zeros for both the residual and the linear operand
        // without replaying either region or staging a materialized array zero.
        let mut zero_context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let residual = zero_context.input(r#type.clone());
        let zero_cotangents = LinearCallOperation::new(1)
            .transpose(
                &mut zero_context,
                &driver,
                &[PartialValue::Known(residual), PartialValue::Unknown(r#type.clone())],
                &[MaybeZero::Zero(r#type.clone())],
            )
            .unwrap();
        assert_eq!(zero_cotangents.len(), 2);
        assert!(zero_cotangents.iter().all(MaybeZero::is_zero));
        assert!(zero_context.builder().borrow().instructions().is_empty());

        let mut context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let residual = context.input(r#type.clone());
        let output_cotangent = context.input(r#type.clone());

        // Transposition must assign a structural zero to the known residual and stage one swapped linear call for the
        // unknown linear input's cotangent.
        let cotangents = LinearCallOperation::new(1)
            .transpose(
                &mut context,
                &driver,
                &[PartialValue::Known(residual), PartialValue::Unknown(r#type)],
                &[MaybeZero::Value(output_cotangent)],
            )
            .unwrap();
        assert_eq!(cotangents.len(), 2);
        assert!(cotangents[0].is_zero());
        let output = match &cotangents[1] {
            MaybeZero::Value(value) => value.atom_id().unwrap(),
            MaybeZero::Zero(_) => panic!("expected a live linear-input cotangent"),
        };

        let builder = context.builder().clone();
        drop((context, cotangents));
        let builder = Rc::try_unwrap(builder).expect("the builder is uniquely owned once every tracer is dropped");
        let program = builder
            .into_inner()
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = linear_call [residual_count=1] %0 %1 [
                    forward={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = add %1 %0
                        in (%2)
                    },
                    transpose={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = mul %1 %0
                        in (%2)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
    }
}
