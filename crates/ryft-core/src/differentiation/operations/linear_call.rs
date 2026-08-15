use std::fmt::Display;

use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchedProgram, BatchingContext, BatchingDriver, BatchingError,
    ProgramBatchingOutputAxesPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::DifferentiationError;
use crate::differentiation::batching::CotangentBatchingPolicy;
use crate::differentiation::forward::{DifferentiableOperation, DifferentiationDriver, DifferentiationDual};
use crate::differentiation::reverse::{TransposableOperation, TranspositionDriver};
use crate::differentiation::types::DifferentiableType;
use crate::differentiation::zeros::ResidualZeroProvider;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, check_types};
use crate::operations::Zero;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, OutputRegionProvenance, ProgramError, RegionInterface, RegionSlot,
    TypeError, TypeIdentityRenaming, Typed, Value,
};
use crate::tracing::{NestedTracingContext, Tracer, TracingContext};

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

    /// Maps this operation from type universe `T` into type universe `U` while preserving its interface form and
    /// [`residual_count`](Self::residual_count). A _forward-and-transpose_ call stores no interface types and therefore
    /// changes only its type parameter. A _transpose-only_ call applies `map` to every stored input and output type of
    /// its unavailable forward map.
    ///
    /// This operation is consuming so lifting a linear call into a composite type universe can move and map its stored
    /// types without cloning them or exposing the private [`LinearCallInterface`] representation.
    ///
    /// # Parameters
    ///
    ///   - `map`: Function that converts each stored type from `T` into `U`.
    #[inline]
    pub fn map_types<U: DifferentiableType, F: FnMut(T) -> U>(self, mut map: F) -> LinearCallOperation<U> {
        LinearCallOperation {
            residual_count: self.residual_count,
            interface: match self.interface {
                LinearCallInterface::ForwardAndTranspose => LinearCallInterface::ForwardAndTranspose,
                LinearCallInterface::TransposeOnly { input_types, output_types } => {
                    let input_types = input_types.into_iter().map(&mut map).collect();
                    let output_types = output_types.into_iter().map(map).collect();
                    LinearCallInterface::TransposeOnly { input_types, output_types }
                }
            },
        }
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
        C: Context<Type = T, Operation: Clone + From<LinearCallOperation<T>>>,
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
    /// [`Region`](crate::Region)s. Both [`LinearCallInterface`] forms preserve a completely replicated call unchanged
    /// when the batching level is _unnamed_, which is precisely when no attached region can observe it.
    ///
    /// Every other [`LinearCallInterface::ForwardAndTranspose`] call batches its forward region to discover the batched
    /// output axes, batches its transpose region under those axes, and aligns the transpose outputs with the original
    /// linear-input axes (i.e., a cotangent for a replicated linear input is summed across the batch by `collapse_fn`,
    /// while cotangents for mapped linear inputs retain their packed axes).
    ///
    /// Every other [`LinearCallInterface::TransposeOnly`] call is rejected because its unavailable forward program
    /// cannot determine the batched output axes.
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
    pub(crate) fn batch_regions<
        C: Context<Type = T, Operation: From<LinearCallOperation<T>>>,
        P: CotangentBatchingPolicy<C>,
        D: BatchingDriver<C, P>,
    >(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
        input_axes: Vec<BatchAxis>,
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

        // A completely replicated call at an unnamed batching level needs no structural region rewrite, and keeping the
        // original call avoids manufacturing a batch axis that neither region observes. What makes that shortcut sound
        // is the level being unnamed rather than the operands being replicated: an attached region's value can vary per
        // batch item with no mapped operand at all, but only by addressing the level _by name_ (e.g., an `axis_index`
        // or a collective over this level's axis, both of which a transposed program naturally contains because
        // `all_gather` transposes to `psum_scatter` and vice versa). Every named-axis operation resolves the level it
        // belongs to by comparing its own axis name against `BatchingContext::axis_name`, so an unnamed level is
        // provably invisible to all of them, including those of operation families this crate does not know about. A
        // named level therefore batches its regions structurally, which is also what lets the named-axis operations
        // inside them resolve against it.
        if input_axes.iter().all(BatchAxis::is_replicated) && context.axis_name().is_none() {
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
                    message: "a transpose-only linear call cannot be batched structurally because its unavailable \
                              forward program does not determine output batch axes; it is preserved unchanged only \
                              when every operand is replicated at an unnamed batching level"
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
            P::sum_mapped_cotangents,
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
            P::sum_mapped_cotangents,
        )?
        .into_parts();
        check_count!("output", transpose_output_axes, linear_axes.len(), ProgramError);
        check_count!("output", transpose.output_ids(), linear_axes.len(), ProgramError);

        // Rebinding the call must reconcile type views that agree only for dense batches. For example, a ragged mapped
        // operand is packed at its declared bound, so its physical type strictly refines the logical boundary retained
        // by the structurally batched regions, while `linear_call` inference requires its regions' boundaries to match
        // the operand types exactly. Both regions are therefore specialized to the packed operand types (which is a
        // no-op whenever the boundaries already agree, which is true for every dense batch), and the transpose's
        // cotangent inputs derive from the specialized forward's physical outputs for the same reason. The bound call's
        // results are plain values that cannot carry batch metadata, so each output carrier is restored through the
        // driver from the source region's per-item logical output type and the input carriers; wrapping them as bare
        // `P::batch` carriers would present bound padding as live data.
        let boundary_operands = P::boundary_operands(context.axis_extent());
        let mut packed_inputs = Vec::with_capacity(boundary_operands.len() + input_values.len());
        let boundary_operand_count = boundary_operands.len();
        packed_inputs.extend(boundary_operands);
        packed_inputs.extend(input_values);
        let forward_input_types = packed_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let forward_output_types = driver.region(0)?.output_types();
        let forward = forward.specialize(forward_input_types.as_slice())?;
        let physical_output_types = forward.output_types();
        let transpose_input_types = packed_inputs[..boundary_operand_count + self.residual_count]
            .iter()
            .map(|input| input.r#type().into_owned())
            .chain(physical_output_types.iter().map(DifferentiableType::cotangent))
            .collect::<Vec<_>>();
        let transpose = transpose.specialize(transpose_input_types.as_slice())?;
        let outputs = context.parent().bind(
            LinearCallOperation::new(self.residual_count + boundary_operand_count),
            vec![forward, transpose],
            packed_inputs.as_slice(),
        )?;
        check_count!("output", outputs, output_axes.len(), ProgramError);
        check_count!("output", forward_output_types, outputs.len(), ProgramError);
        outputs
            .into_iter()
            .zip(output_axes.into_iter().zip(forward_output_types))
            .map(|(output, (axis, logical_type))| driver.restore_batch(output, axis, &logical_type, inputs))
            .collect()
    }
}

impl<T: DifferentiableType> Display for LinearCallOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: DifferentiableType> Operation for LinearCallOperation<T> {
    type Type = T;

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
                          reverse-mode differentiation (e.g., `vjp`, `value_and_gradient`, or `jacobian_reverse`)"
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
    T: DifferentiableType,
    C: Context<Type = T, Operation: Clone + From<LinearCallOperation<T>>>,
    P: CotangentBatchingPolicy<C>,
> BatchableOperation<C, P> for LinearCallOperation<T>
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        let input_axes = inputs.iter().map(P::batch_axis).collect::<Vec<_>>();
        Ok(self.batch_regions(context, driver, inputs, input_axes)?.into())
    }
}

impl<C: Context<Type: DifferentiableType, Operation: ResidualZeroProvider<C::Type>> + Zero<C::Value>>
    DifferentiableOperation<C> for LinearCallOperation<C::Type>
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
        // The derived program is only interpreted here, never re-attached, so the shared handle is simply dereferenced.
        let jvp = driver.jvp_program(forward)?;
        let mut jvp_inputs = primals;
        for input in inputs {
            if !input.tangent().r#type().is_zero_space() {
                // The operand primal names every runtime quantity a reference-bearing tangent type omits, because the
                // tangent type derivation preserves geometry exactly; statically shaped operands keep the nullary
                // zero and stage the same instruction sequence as before.
                jvp_inputs.push(C::Operation::materialize_zero_from_residual_sources(
                    context,
                    input.tangent().clone(),
                    std::iter::once(input.primal()),
                )?);
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
    O: Operation<Type = V::Type> + ResidualZeroProvider<V::Type> + From<LinearCallOperation<V::Type>>,
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

        // A dead output's structural-zero cotangent still becomes a real operand of the transposed call. Its type
        // alone cannot construct it when it references runtime identities, but the boundary collectively names every
        // such quantity: at least one peer cotangent is live here (the all-zero case returned above) and the retained
        // residuals are live too, so the zero is assembled from them one identity at a time before falling back to the
        // nullary zero that every identity-free type keeps.
        let materialized_cotangents = outputs
            .iter()
            .cloned()
            .map(|output| {
                O::materialize_zero_from_residual_sources(
                    context,
                    output,
                    outputs.iter().filter_map(MaybeZero::as_value).chain(&residuals),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut transpose_inputs = residuals;
        transpose_inputs.extend(materialized_cotangents);
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
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayBatch, ArrayBatching, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType,
        DataType, Dimension, DimensionBounds, DimensionValue, DimensionVariable, LogicalMesh, MeshAxis, MeshAxisType,
        Shape, Sharding, ShardingDimension,
    };
    use crate::axes::AxisIndexOperation;
    use crate::batching::{BatchAxis, ProgramBatchingOutputAxesPolicy, RecursiveBatchingDriver, batch};
    use crate::contexts::tests::{
        ProjectedMemberType, ProjectedMemberValue, ProjectedProgramType, ProjectedProgramValue,
    };
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::DifferentiationError;
    use crate::differentiation::reverse::{TransposableOperation, TranspositionDriver};
    use crate::operations::{
        AddOperation, ConvertElementTypeOperation, DimensionFromScalar, DynamicBroadcast, MulOperation, Reduce,
        ReductionKind, ZeroLikeOperation, ZeroOperation,
    };
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationOutput, PartialValue};
    use crate::programs::{
        Effects, MaybeZero, Program, ProgramBuilder, ProgramError, RegionDriver, RegionRef, RegionSlot, ValueProjection,
    };
    use crate::tracing::TracingContext;

    use super::*;

    /// Builds the scalar program `(r, u) ↦ r · u` used as a residual-parameterized linear-map region.
    fn scalar_multiply_program() -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let r#type = ArrayType::scalar(DataType::F64);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let residual = builder.add_input(r#type.clone());
        let linear = builder.add_input(r#type);
        let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![linear, residual]).unwrap()[0];
        builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap()
    }

    /// Builds the scalar program `u ↦ u · axis_index("items")`, a linear map whose factor is read from the named
    /// `items` axis rather than from an operand. It serves as both regions of a linear call whose value varies per
    /// batch item even though every operand is replicated.
    fn axis_scaled_multiply_program() -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let linear = builder.add_input(ArrayType::scalar(DataType::F64));
        let index = builder
            .add_instruction(AxisIndexOperation::new("items".to_string()), Vec::new(), Vec::new())
            .unwrap()[0];
        let factor = builder
            .add_instruction(ConvertElementTypeOperation::new(DataType::F64), Vec::new(), vec![index])
            .unwrap()[0];
        let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![linear, factor]).unwrap()[0];
        builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Test-only transposition driver exposing one attached region, used to invoke the _transpose-only_ form's
    /// transposition rule directly on the transpose region it replays.
    struct TestTranspositionDriver<'r> {
        /// Transpose region exposed by this driver.
        region: RegionRef<'r, Array, ArrayOperation<Array>>,
    }

    impl RegionDriver<Array, ArrayOperation<Array>> for TestTranspositionDriver<'_> {
        fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, Array, ArrayOperation<Array>>>
        where
            Array: 'r,
            ArrayOperation<Array>: 'r,
        {
            std::iter::once(self.region)
        }
    }

    impl TranspositionDriver<Array, ArrayOperation<Array>> for TestTranspositionDriver<'_> {
        fn transpose_program(
            &self,
            _region: RegionRef<'_, Array, ArrayOperation<Array>>,
            _input_linearity: &[bool],
        ) -> Result<Arc<Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>>, DifferentiationError> {
            Err(ProgramError::UnsupportedOperation {
                message: "test driver does not transpose nested regions".to_string(),
            }
            .into())
        }
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

        // Mapping changes only the type universe: the executable form retains its marker-only interface, while the
        // transpose-only form maps every stored input and output type without changing the residual boundary.
        assert_eq!(
            LinearCallOperation::<ArrayType>::new(2).map_types(ArrayIrType::Array),
            LinearCallOperation::<ArrayIrType>::new(2),
        );
        let scalar_type = ArrayType::scalar(DataType::F64);
        assert_eq!(
            LinearCallOperation::transpose_only(1, vec![scalar_type.clone()], vec![scalar_type.clone()])
                .map_types(ArrayIrType::Array),
            LinearCallOperation::transpose_only(
                1,
                vec![ArrayIrType::Array(scalar_type.clone())],
                vec![ArrayIrType::Array(scalar_type)],
            ),
        );

        // The transpose-only form derives its transpose region's expected interface from the stored forward interface
        // through the cotangent type mapping, so it infers the stored output types whenever the attached region agrees.
        // The linear types are differential types rather than primal storage types, which the mapping must respect.
        let residual_type = ArrayType::scalar(DataType::F64);
        let tangent_type = ArrayType::scalar(DataType::F32);
        let transpose_interface = RegionInterface::new(
            vec![residual_type.clone(), tangent_type.clone()],
            vec![tangent_type.clone()],
            Effects::PURE,
        );
        assert_eq!(
            LinearCallOperation::transpose_only(1, vec![tangent_type.clone()], vec![tangent_type.clone()])
                .infer_output_types(&[residual_type, tangent_type.clone()], std::slice::from_ref(&transpose_interface)),
            Ok(vec![tangent_type]),
        );
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
    fn test_linear_call_operation_transpose_only_validates_residual_count() {
        let r#type = ArrayType::scalar(DataType::F64);

        // A residual count larger than the operand count is a malformed call rather than a silently truncated split,
        // and inference rejects it before any region interface is consulted.
        let transpose_interface = RegionInterface::new(vec![r#type.clone()], vec![r#type.clone()], Effects::PURE);
        assert!(matches!(
            LinearCallOperation::transpose_only(2, Vec::new(), Vec::new())
                .infer_output_types(&[], std::slice::from_ref(&transpose_interface)),
            Err(TypeError::Invalid { message }) if message == "linear call residual count 2 exceeds input count 0",
        ));

        // Transposition enforces the same split independently, because a pullback may be built from an imported call
        // whose operands were pruned after inference ran.
        let transpose = scalar_multiply_program();
        let driver = TestTranspositionDriver { region: transpose.entry_region_ref() };
        let mut context = TracingContext::<Array, ArrayOperation<Array>>::new();
        assert!(matches!(
            LinearCallOperation::transpose_only(3, vec![r#type.clone()], vec![r#type.clone()]).transpose(
                &mut context,
                &driver,
                &[],
                &[],
            ),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message)))
                if message == "linear call residual count 3 exceeds input count 0",
        ));

        // Every residual must be known during transposition, because the replayed transpose region consumes the
        // residual values themselves rather than cotangents for them.
        let output_cotangent = context.input(r#type.clone());
        assert!(matches!(
            LinearCallOperation::transpose_only(1, vec![r#type.clone()], vec![r#type.clone()]).transpose(
                &mut context,
                &driver,
                &[PartialValue::Unknown(r#type.clone()), PartialValue::Unknown(r#type)],
                &[MaybeZero::Value(output_cotangent)],
            ),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message)))
                if message == "linear call residual operand 0 is not known during transposition",
        ));
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
                               reverse-mode differentiation (e.g., `vjp`, `value_and_gradient`, or \
                               `jacobian_reverse`)",
        ));
        assert!(matches!(
            program.batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            ),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "a transpose-only linear call cannot be batched structurally because its unavailable \
                               forward program does not determine output batch axes; it is preserved unchanged only \
                               when every operand is replicated at an unnamed batching level",
        ));
    }

    #[test]
    fn test_linear_call_operation_transpose_only_remains_opaque_to_partial_evaluation() {
        // A transpose-only carrier (the form `custom_vjp` stages into its tangent program) must survive partial
        // evaluation as one unknown-producing instruction. Splitting or folding it would separate the backward region
        // from the residuals that parameterize it, so the default opaque rule is the correct behavior here.
        let r#type = ArrayType::scalar(DataType::F64);
        let transpose = scalar_multiply_program();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let transpose = builder.import_region(transpose.entry_region_ref());
        let residual = builder.add_input(r#type.clone());
        let linear = builder.add_input(r#type.clone());
        let output = builder
            .add_instruction(
                ArrayOperation::LinearCall(LinearCallOperation::transpose_only(
                    1,
                    vec![r#type.clone()],
                    vec![r#type.clone()],
                )),
                vec![transpose],
                vec![residual, linear],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let evaluation = program
            .partially_evaluate(&[PartialValue::Unknown(r#type.clone()), PartialValue::Unknown(r#type)])
            .unwrap();

        assert!(matches!(evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), ArrayOperation::LinearCall(_)));
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
    fn test_linear_call_operation_batching_rewrites_replicated_regions_naming_the_batch_axis() {
        // Both regions of `u ↦ u · axis_index("items")` address the enclosing named batching level, so the call's
        // value varies per batch item even though its single operand is replicated.
        let regions = vec![axis_scaled_multiply_program(), axis_scaled_multiply_program()];
        let driver = RecursiveBatchingDriver::new(&regions);
        let context = BatchingContext::<_, ArrayBatching>::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 3)
            .with_axis_name("items".to_string());

        // Batching must therefore rewrite both regions instead of taking the all-replicated fast path, which would bind
        // the call unchanged and leave the `axis_index` unresolved (it then reaches eager interpretation and reports
        // "`axis_index` for the device mesh axis `items` has no eager value").
        let outputs = LinearCallOperation::new(0)
            .batch(&context, &driver, &[ArrayBatch::replicated(Array::scalar(2.0))])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &Array::vector(vec![0.0, 2.0, 4.0]));
    }

    #[test]
    fn test_linear_call_operation_batching_preserves_a_replicated_unnamed_call() {
        let forward = scalar_multiply_program();
        let transpose = forward.clone();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let forward = builder.import_region(forward.entry_region_ref());
        let transpose = builder.import_region(transpose.entry_region_ref());
        let residual = builder.add_input(ArrayType::scalar(DataType::F64));
        let linear = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(LinearCallOperation::new(1), vec![forward, transpose], vec![residual, linear])
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // An unnamed level is the shape every differentiation-generated linear call is batched under, and no operation
        // can address it, so a completely replicated call keeps its fast path: the staged instruction retains its
        // residual count, its operand order, and both regions byte for byte.
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
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = linear_call [residual_count=1] %0 %1 [
                    forward={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = mul %1 %0
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
        assert_eq!(batched.to_string(), program.to_string());

        // A *named* level is conservatively structural even for regions that name no axis, because the level's name is
        // all this rule can decide on. The rewritten regions are semantically the source regions, so the batched call
        // still computes `r · u` and reports its output replicated.
        let regions = vec![scalar_multiply_program(), scalar_multiply_program()];
        let driver = RecursiveBatchingDriver::new(&regions);
        let context = BatchingContext::<_, ArrayBatching>::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
            .with_axis_name("items".to_string());
        let outputs = LinearCallOperation::new(1)
            .batch(
                &context,
                &driver,
                &[ArrayBatch::replicated(Array::scalar(2.0)), ArrayBatch::replicated(Array::scalar(4.0))],
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value(), &Array::scalar(8.0));
    }

    #[test]
    fn test_linear_call_operation_batching_specializes_ragged_region_boundaries() -> Result<(), ProgramError> {
        // A mapped `dimension_from_scalar` gives each batch item its own extent, so the value entering the linear
        // call is packed at the declared bound while the structurally batched regions retain the logical dynamic
        // boundary. Batching must specialize both attached regions to the packed operand types and restore a ragged
        // carrier for the call's output.
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4))?);
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let values = trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)])).into());
        let extents = trace.input(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2)])).into());
        let output = batch(
            |(value, extent)| {
                let extent = extent.to_dimension(variable.clone())?;
                let repeated = value.dynamic_broadcast_to(&[extent])?;
                let mut repeated = LinearCallOperation::stage(
                    repeated.context(),
                    Vec::new(),
                    vec![repeated.clone()],
                    |_, inputs| Ok(inputs.to_vec()),
                    |_, cotangents| Ok(cotangents.to_vec()),
                )?;
                let repeated = ValueProjection::<ArrayType>::into_projected(repeated.remove(0))?;
                Ok(repeated.reduce(&[0], ReductionKind::Sum).into_value())
            },
            (values, extents),
            (BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        )?;
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, ArrayIrValue<Array>>(
            vec![output.atom_id()?],
            vec![Placeholder, Placeholder],
            Placeholder,
        )?;

        // Both attached regions were specialized to the packed operand types. Every array-member boundary type is
        // the physical bound-shaped storage, with no logical dynamic dimension remaining.
        let instruction = program
            .instructions()
            .iter()
            .find(|instruction| instruction.operation().name() == "linear_call")
            .unwrap();
        for region in instruction.regions() {
            let region = program.region_ref(*region).unwrap();
            for r#type in region.input_types().iter().chain(region.output_types().iter()) {
                if let ArrayIrType::Array(array_type) = r#type {
                    assert!(array_type.static_shape().is_some(), "{array_type}");
                }
            }
        }

        // The rebuilt transpose is the identity over the physical cotangent, taking the relayed batch extent first.
        let transpose = program.region_ref(instruction.regions()[1]).unwrap().to_program();
        let cotangent = Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            transpose.interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::constant(2)?),
                ArrayIrValue::Array(cotangent.clone()),
            ])?,
            vec![ArrayIrValue::Array(cotangent)],
        );

        // End to end, the masked sums match the per-item logical extents rather than the packed bound.
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![2.0_f32, 3.0])),
                ArrayIrValue::Array(Array::vector(vec![1_i32, 3])),
            ])?,
            ArrayIrValue::Array(Array::vector(vec![2.0_f32, 9.0])),
        );
        Ok(())
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
            ) -> Result<Arc<Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>>, DifferentiationError>
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
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![output_cotangent, residual]).unwrap()[0];
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

    #[test]
    fn test_linear_call_operation_transposition_zeros_known_operand_cotangents() {
        // The transpose region of a transpose-only call returns one cotangent per linear operand, in operand order
        // after the leading residuals. Residual operands are fixed primal parameters rather than linear inputs, and
        // a *known* linear operand is not being differentiated, so both receive structural zeros while the replayed
        // region's cotangents are assigned only to the unknown linear operands.
        let r#type = ArrayType::scalar(DataType::F64);
        let mut transpose_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let residual = transpose_builder.add_input(r#type.clone());
        let output_cotangent = transpose_builder.add_input(r#type.clone());
        let first_input_cotangent = transpose_builder
            .add_instruction(MulOperation::new(), Vec::new(), vec![residual, output_cotangent])
            .unwrap()[0];
        let transpose = transpose_builder
            .build::<Vec<Array>, Vec<Array>>(
                vec![first_input_cotangent, output_cotangent],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let driver = TestTranspositionDriver { region: transpose.entry_region_ref() };
        let mut context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let residual = context.input(r#type.clone());
        let known_linear = context.input(r#type.clone());
        let output_cotangent = context.input(r#type.clone());

        let linear_types = vec![r#type.clone(), r#type.clone()];
        let cotangents = LinearCallOperation::transpose_only(1, linear_types, vec![r#type.clone()])
            .transpose(
                &mut context,
                &driver,
                &[PartialValue::Known(residual), PartialValue::Unknown(r#type), PartialValue::Known(known_linear)],
                &[MaybeZero::Value(output_cotangent)],
            )
            .unwrap();

        assert_eq!(cotangents.len(), 3);
        assert!(cotangents[0].is_zero());
        assert!(matches!(cotangents[1], MaybeZero::Value(_)));
        assert!(cotangents[2].is_zero());
    }

    #[test]
    fn test_linear_call_operation_transposition_preserves_structural_zero_outputs() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let primal_type = ArrayType::scalar(DataType::F64)
            .with_sharding(Sharding::new(mesh, Vec::new()).unwrap().with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let tangent_type = primal_type.tangent();
        let cotangent_type = primal_type.cotangent();
        assert_ne!(tangent_type, cotangent_type);

        // A canonical `zero` transpose-region output is already typed in the linear input's cotangent space.
        // Recovering its structural zero must retain that type instead of dualizing its sharding a second time.
        let mut transpose_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        transpose_builder.add_input(cotangent_type.clone());
        let zero = transpose_builder
            .add_instruction(ZeroOperation::new(cotangent_type.clone()), Vec::new(), Vec::new())
            .unwrap()[0];
        let transpose = transpose_builder
            .build::<Vec<Array>, Vec<Array>>(vec![zero], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let driver = TestTranspositionDriver { region: transpose.entry_region_ref() };
        let mut context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = context.input(cotangent_type.clone());
        let cotangents = LinearCallOperation::transpose_only(0, vec![tangent_type.clone()], vec![tangent_type.clone()])
            .transpose(
                &mut context,
                &driver,
                &[PartialValue::Unknown(tangent_type.clone())],
                &[MaybeZero::Value(output_cotangent)],
            )
            .unwrap();
        assert!(matches!(&cotangents[0], MaybeZero::Zero(r#type) if r#type == &cotangent_type));

        // `zero_like` is equally structural even though it consumes an exemplar input. Opaque region replay must
        // recognize it instead of turning the result into a live zero-valued tracer.
        let mut transpose_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = transpose_builder.add_input(cotangent_type.clone());
        let zero_like = transpose_builder
            .add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![output_cotangent])
            .unwrap()[0];
        let transpose = transpose_builder
            .build::<Vec<Array>, Vec<Array>>(vec![zero_like], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let driver = TestTranspositionDriver { region: transpose.entry_region_ref() };
        let mut context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = context.input(cotangent_type.clone());
        let cotangents = LinearCallOperation::transpose_only(0, vec![tangent_type.clone()], vec![tangent_type])
            .transpose(
                &mut context,
                &driver,
                &[PartialValue::Unknown(primal_type.tangent())],
                &[MaybeZero::Value(output_cotangent)],
            )
            .unwrap();
        assert!(matches!(&cotangents[0], MaybeZero::Zero(r#type) if r#type == &cotangent_type));
    }
}
