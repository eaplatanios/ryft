use std::fmt::Display;

use crate::arrays::{ArrayType, DataType};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchedProgram, BatchingContext, BatchingDriver, BatchingError,
    BatchingPolicy, ProgramBatchingOutputAxesPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    CotangentAccumulator, CotangentDestinationKind, DifferentiableOperation, DifferentiableType,
    DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationError, DifferentiationPolicy,
    NOTHING_SAVEABLE_POLICY_NAME, ResidualZeroProvider, TransposableOperation, TranspositionContext,
    TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, check_types};
use crate::operations::arithmetic::AddOperation;
use crate::operations::manipulation::conversions::ReducePrecisionOperation;
use crate::operations::manipulation::memory::TransferToMemoryOperation;
use crate::operations::references::ReferenceNewOperation;
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationInput, PartialEvaluationOutput,
    PartialEvaluationValue, PartialValue, PartiallyEvaluatableOperation, PartitionedProgram, ResidualPolicyReference,
};
use crate::programs::{
    CalleeRegionDriver, ErasedOperation, InputRegionProvenance, MaybeZero, Operation, OperationBoundaryPruning,
    OperationFormatter, OperationPayloadProjection, OperationProvider, OutputRegionProvenance, ProgramError,
    ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeValue,
    ReferenceDischargeableOperation, ReferenceMemberType, RegionInterface, RegionLiveness, RegionSlot, Type, TypeError,
    Typed, Value, discharge_positional_region_operation,
};
use crate::tracing::{Tracer, TracingContext};

/// Selection of the instruction inputs of a [`RematerializeOperation`] on which backends place an optimization barrier
/// when the call is [differentiated](RematerializeOperation::differentiated). The barrier keeps compilers from merging
/// the recomputation with the original computation (e.g., through common subexpression elimination) and from scheduling
/// it before the selected inputs are available. This is the analogue of the `prevent_cse` parameter of
/// [`jax.checkpoint`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html), whose per-argument
/// form corresponds to [`RematerializationOptimizationBarrier::Inputs`].
///
/// Transforms keep an [`Inputs`](RematerializationOptimizationBarrier::Inputs) selection aligned with the inputs of the
/// calls that they derive. Inputs that carry derivative values (i.e., saved residuals, tangents, and cotangents) are
/// selected, inputs that only carry bookkeeping state (i.e., the extents that batching prepends and the captured
/// reference state that reference discharge appends) are not, and pruned inputs drop their entries.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RematerializationOptimizationBarrier {
    /// Every input, which is the default.
    All,

    /// No input, which disables the barrier (e.g., for a call in a loop body, where the loop already keeps the
    /// recomputation from being merged with the original computation).
    None,

    /// The inputs whose positional entries are `true`, with one entry per instruction input of the call.
    Inputs(Vec<bool>),
}

impl RematerializationOptimizationBarrier {
    /// Returns whether this selection includes the instruction input at `index`.
    #[inline]
    pub fn selects(&self, index: usize) -> bool {
        match self {
            Self::All => true,
            Self::None => false,
            Self::Inputs(selected) => selected.get(index).copied().unwrap_or(false),
        }
    }
}

/// Canonical operation name for [`RematerializeOperation`].
pub const REMATERIALIZE_OPERATION_NAME: &str = "rematerialize";

/// [`Operation`] that represents a rematerialized (i.e., checkpointed) call of its attached `body` region, which is the
/// analogue of JAX's [`jax.checkpoint`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html) primitive.
/// Outside of differentiation, the call computes exactly what its body computes, and its outputs are the body's
/// outputs. Under differentiation, the residual policy of the call decides which of the values that the body computes
/// are saved for the derivative computation and which are recomputed from the saved values instead (based on a
/// [`ResidualPolicy`](crate::ResidualPolicy)), which trades computation for the memory that the saved values occupy.
///
/// The instruction inputs of the call map positionally onto the region inputs of its body, and its outputs map
/// positionally onto the outputs of its body. The call carries no stored derivative regions: the transforms derive the
/// body's derivatives when they need them, so rematerialization composes with every transform that applies to its body.
///
/// The call also records whether it is the residual side of a differentiated computation (via
/// [`differentiated`](Self::differentiated)), in which case backends place an optimization barrier on the inputs that
/// its [`optimization_barrier`](Self::optimization_barrier) selects, so that the recomputation is neither merged with
/// the original computation nor scheduled before those inputs are available.
///
/// # Example
///
/// The following call saves nothing, so differentiating it recomputes `sin(x)` from `x`:
///
/// ```text
/// %1:f64[] = rematerialize %0 [
///     body={
///         lambda %0:f64[] .
///         let %1:f64[] = sin %0
///         in (%1)
///     },
/// ]
/// ```
///
/// It renders only the fields that differ from their defaults (e.g.,
/// `rematerialize [policy="dots_saveable", optimization_barrier=false, differentiated=true]`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RematerializeOperation<T: Type> {
    /// Residual policy that decides which values the call saves under differentiation.
    policy: ResidualPolicyReference<T>,

    /// Inputs of the call on which backends place an optimization barrier when it is
    /// [differentiated](Self::differentiated).
    optimization_barrier: RematerializationOptimizationBarrier,

    /// Whether the call is the residual side of a differentiated computation.
    differentiated: bool,
}

impl<T: Type> RematerializeOperation<T> {
    /// Creates a new [`RematerializeOperation`] with the provided residual policy, which places an optimization barrier
    /// on all of its inputs when it is differentiated and is not yet differentiated.
    #[inline]
    pub fn new(policy: ResidualPolicyReference<T>) -> Self {
        Self { policy, optimization_barrier: RematerializationOptimizationBarrier::All, differentiated: false }
    }

    /// Sets the inputs of this [`RematerializeOperation`] on which backends place an optimization barrier when it is
    /// differentiated. Refer to [`optimization_barrier`](Self::optimization_barrier) for more information.
    #[inline]
    pub fn with_optimization_barrier(mut self, optimization_barrier: RematerializationOptimizationBarrier) -> Self {
        self.optimization_barrier = optimization_barrier;
        self
    }

    /// Returns this [`RematerializeOperation`] with the provided differentiated flag.
    /// Refer to [`differentiated`](Self::differentiated) for more information.
    #[inline]
    pub fn with_differentiated(mut self, differentiated: bool) -> Self {
        self.differentiated = differentiated;
        self
    }

    /// Returns the residual policy that decides which values this call saves under differentiation.
    #[inline]
    pub fn policy(&self) -> &ResidualPolicyReference<T> {
        &self.policy
    }

    /// Returns the inputs of this call on which backends place an optimization barrier when it is
    /// [differentiated](Self::differentiated). This is the analogue of the `prevent_cse` parameter of
    /// [`jax.checkpoint`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html).
    #[inline]
    pub fn optimization_barrier(&self) -> &RematerializationOptimizationBarrier {
        &self.optimization_barrier
    }

    /// Returns whether this call is the residual side of a differentiated computation. Splitting a call into the work
    /// that its derivative computation needs up front and the work that it recomputes marks the recomputing call as
    /// differentiated, and every other transform preserves the flag.
    #[inline]
    pub fn differentiated(&self) -> bool {
        self.differentiated
    }

    /// Returns this [`RematerializeOperation`] lifted into a type universe `U` whose types project into `T` (e.g.,
    /// from [`ArrayType`] into [`ArrayIrType`](crate::ArrayIrType)), with its policy lifted through
    /// [`ResidualPolicyReference::lift`], which keeps the identity of the policy, and with its flags unchanged.
    #[inline]
    pub fn lift<U: 'static + Type>(&self) -> RematerializeOperation<U>
    where
        T: 'static,
        for<'t> &'t T: TryFrom<&'t U>,
    {
        RematerializeOperation {
            policy: self.policy.lift(),
            optimization_barrier: self.optimization_barrier.clone(),
            differentiated: self.differentiated,
        }
    }

    /// Returns this [`RematerializeOperation`] with its [optimization barrier](Self::optimization_barrier) selection
    /// remapped by `map_fn` onto the inputs of a call that a transform derives from this one. A selection of all inputs
    /// or of no inputs applies to every derived call unchanged.
    fn with_remapped_optimization_barrier<F: FnOnce(&[bool]) -> Vec<bool>>(&self, map_fn: F) -> Self {
        let mut operation = self.clone();
        if let RematerializationOptimizationBarrier::Inputs(selected) = &self.optimization_barrier {
            operation.optimization_barrier = RematerializationOptimizationBarrier::Inputs(map_fn(selected));
        }
        operation
    }

    /// Returns this [`RematerializeOperation`] with its [optimization barrier](Self::optimization_barrier) selection
    /// remapped onto the inputs of the residual program of `partition`, whose partitioned program has the inputs of
    /// this call. An unknown input keeps its selection, and so does a known input that the known program forwards to
    /// the residual program unchanged, while every residual that the known program computes is selected, so that the
    /// recomputation stays separate from the computation of the values that it starts from.
    fn with_residual_optimization_barrier<V: Value<Type = T>, O: Operation<Type = T>>(
        &self,
        partition: &PartitionedProgram<V, O>,
    ) -> Self {
        self.with_remapped_optimization_barrier(|selected| {
            let known_program = partition.known_program();
            let known_output_count = partition.outputs().iter().filter(|output| output.is_known()).count();
            partition
                .residual_inputs()
                .iter()
                .map(|input| match input {
                    PartialEvaluationInput::Unknown(index) => selected[*index],
                    PartialEvaluationInput::Known(index) => {
                        let output = known_program.output_ids()[known_output_count + index];
                        match known_program.input_ids().iter().position(|input| *input == output) {
                            Some(position) => selected[partition.known_input_indices()[position]],
                            None => true,
                        }
                    }
                })
                .collect()
        })
    }

    /// Returns the operation that rounds a saved residual of type `r#type` to the precision of its type, which
    /// [`PartitionedProgram::with_rounded_residuals`] stages in the known program of a rematerialized call so that
    /// its known and residual consumers observe the same value. This rounding policy targets arrays of IEEE-style
    /// floating-point types narrower than `f32` (i.e., `bf16`, `f16`, and the `f8` formats with infinities), which
    /// backends may compute at a higher precision (e.g., in `f32`). [`ReducePrecisionOperation`] simulates their
    /// formats without changing the array type. The policy leaves `f32` and wider types unchanged. It also excludes
    /// finite-only formats (e.g., `f8e4m3fn`), because an IEEE-style simulation of their bit widths would map their
    /// largest finite values to infinities.
    fn excess_precision_rounding(r#type: &T) -> Option<ErasedOperation>
    where
        for<'t> &'t ArrayType: TryFrom<&'t T>,
    {
        let (exponent_bits, mantissa_bits) = match <&ArrayType>::try_from(r#type).ok()?.data_type() {
            DataType::BF16 => (8, 7),
            DataType::F16 => (5, 10),
            DataType::F8E5M2 => (5, 2),
            DataType::F8E4M3 => (4, 3),
            DataType::F8E3M4 => (3, 4),
            _ => return None,
        };
        Some(ErasedOperation::new(ReducePrecisionOperation::<ArrayType>::new(exponent_bits, mantissa_bits)))
    }
}

impl<T: 'static + Type> Display for RematerializeOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: 'static + Type> Operation for RematerializeOperation<T> {
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        REMATERIALIZE_OPERATION_NAME
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        const { &[RegionSlot::computation("body")] }
    }

    fn infer_region_input_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<Option<Vec<T>>>, TypeError> {
        // The body is always requested at the instruction input types, whose type identities may differ from the
        // declared ones even when the types compare equal, and staging decides whether that requires instantiating
        // or specializing the body.
        check_count!("region", region_interfaces, 1, TypeError);
        T::derive_identity_renaming(region_interfaces[0].input_types(), input_types)?;
        Ok(vec![Some(input_types.to_vec())])
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        check_count!("region", region_interfaces, 1, TypeError);
        let body = &region_interfaces[0];
        check_types!(@same, format!("`{REMATERIALIZE_OPERATION_NAME}` body input"), [
            input_types,
            body.input_types(),
        ]);
        if let RematerializationOptimizationBarrier::Inputs(selected) = &self.optimization_barrier
            && selected.len() != input_types.len()
        {
            return Err(TypeError::invalid(format!(
                "`{}` optimization barrier has {} entries but the call has {} inputs",
                REMATERIALIZE_OPERATION_NAME,
                selected.len(),
                input_types.len(),
            )));
        }
        Ok(body.output_types().to_vec())
    }

    #[inline]
    fn input_region_provenance(&self, region_index: usize, input_index: usize) -> InputRegionProvenance {
        if region_index == 0 {
            InputRegionProvenance::Input { index: input_index }
        } else {
            InputRegionProvenance::None
        }
    }

    #[inline]
    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        vec![OutputRegionProvenance { region_index: 0, output_index }]
    }

    fn prune_boundary(
        &self,
        input_count: usize,
        used_outputs: &[bool],
        regions: &mut dyn RegionLiveness,
    ) -> Result<Option<OperationBoundaryPruning<Self>>, ProgramError> {
        // Instruction inputs map onto the body inputs one for one, so the inputs that the body does not use are dropped
        // together with its unused outputs.
        let kept_inputs = regions.used_region_inputs(0, used_outputs)?;
        check_count!("input", kept_inputs, input_count, ProgramError);
        let operation = self.with_remapped_optimization_barrier(|selected| {
            selected.iter().zip(&kept_inputs).filter_map(|(selected, kept)| kept.then_some(*selected)).collect()
        });
        Ok(Some(OperationBoundaryPruning { operation, kept_inputs, kept_outputs: used_outputs.to_vec() }))
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        let operation = OperationFormatter::new(formatter, indentation, REMATERIALIZE_OPERATION_NAME)?;
        let renders_policy = self.policy.name() != NOTHING_SAVEABLE_POLICY_NAME;
        let renders_optimization_barrier = self.optimization_barrier != RematerializationOptimizationBarrier::All;
        if !renders_policy && !renders_optimization_barrier && !self.differentiated {
            return Ok(());
        }
        operation.bracketed(|operation| {
            if renders_policy {
                operation.field("policy", format_args!("{:?}", self.policy.name()))?;
            }
            match &self.optimization_barrier {
                RematerializationOptimizationBarrier::All => {}
                RematerializationOptimizationBarrier::None => operation.field("optimization_barrier", false)?,
                RematerializationOptimizationBarrier::Inputs(selected) => {
                    operation.field("optimization_barrier", format_args!("{selected:?}"))?
                }
            }
            if self.differentiated {
                operation.field("differentiated", true)?;
            }
            Ok(())
        })
    }
}

impl<
    T: 'static + Type + From<P::Referent>,
    C: Context<Type = T, Operation: From<RematerializeOperation<T>>>,
    P: ReferenceDischargePolicy<C>,
> ReferenceDischargeableOperation<C, P> for RematerializeOperation<T>
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        // A rematerialized call forwards its instruction inputs onto its body's inputs one for one and reports the
        // body's outputs as its own, which is the positionally forwarding shape that the shared structured rewrite
        // serves with no leading inputs. The captured reference state that the rewrite appends carries no derivative
        // values, so its inputs are not selected by the optimization barrier.
        discharge_positional_region_operation(self, context, driver, inputs, 0, |appended_input_count| {
            self.with_remapped_optimization_barrier(|selected| {
                selected.iter().copied().chain(std::iter::repeat_n(false, appended_input_count)).collect()
            })
        })
    }
}

impl<C: Domain<Type: 'static>> InterpretableOperation<C> for RematerializeOperation<C::Type> {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        driver.interpret_region(context, 0, inputs.to_vec())
    }
}

impl<C: Context<Type: 'static, Operation: From<RematerializeOperation<C::Type>> + OperationPayloadProjection>>
    PartiallyEvaluatableOperation<C> for RematerializeOperation<C::Type>
where
    for<'t> &'t ArrayType: TryFrom<&'t C::Type>,
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // Partial evaluation follows JAX's `remat_partial_eval`. A call whose inputs are all known folds whole, which
        // keeps the rematerialization boundary in the known-side context. Otherwise, the body is partitioned by input
        // knownness with this call's residual policy, which decides which known values the residual side receives and
        // which ones it recomputes. The known program is then replayed in the known-side context (i.e., hoisted out of
        // the call), and the residual program becomes a differentiated call. Partitioning a call that is already
        // differentiated (e.g., the residual side of a linearization, whose known inputs are the saved values)
        // therefore re-plans its body with the same policy rather than hoisting its recomputation. A body with effects,
        // deferred work, or references stays whole instead, because replaying its known side outside of the call would
        // bypass the effect ordering and the reference placement of the active partial evaluation, as for `jit_call`.
        let body = driver.region(0)?;
        check_count!("input", inputs, body.input_types().len(), ProgramError);
        let effects = body.effects();
        let has_references = inputs.iter().any(|input| input.r#type().is_reference())
            || body.output_types().iter().any(Type::is_reference)
            || body
                .instructions_in_closure()
                .any(|(_, instruction)| instruction.operation().effects().has_reference_declarations());
        if inputs.iter().all(PartialEvaluationValue::is_known)
            || !effects.classes().is_empty()
            || effects.is_retained_when_unused()
            || has_references
        {
            return context.fold_or_residualize(self.clone(), vec![body.to_program()], inputs);
        }
        if let Some(error) = context.error() {
            return Err(error);
        }

        let input_known = inputs.iter().map(PartialEvaluationValue::is_known).collect::<Vec<_>>();

        // Look through memory transfers and round before offloading, so known consumers and derivative
        // consumers use the same rounded value. Rounding after the transfer would leave known consumers
        // using the value computed with excess precision.
        let partition = driver
            .partition_program(&context.clone().with_residual_policy(&self.policy), body, &input_known)?
            .with_rounded_residuals(Self::excess_precision_rounding, |operation| {
                operation.projected_payload::<TransferToMemoryOperation>().is_some()
            })?;
        let residual_operation = self.with_residual_optimization_barrier(&partition).with_differentiated(true);
        let (known_program, residual_program, known_input_indices, residual_inputs, outputs) = partition.into_parts();

        // The known program returns the known outputs of the body followed by the values that the residual program
        // receives, which are therefore offset by the number of known outputs. An eager known-side context may be
        // unable to execute some pure operations (e.g., operations that only a backend can execute), which keeps the
        // call whole as for any other operation. Known inputs that the known program forwards keep their original
        // values, so that they are materialized in the residual program as they would be without the call.
        let known_inputs = known_input_indices.iter().map(|&index| inputs[index].as_known().cloned().unwrap());
        let known_outputs = match known_program.interpret_in_context(context.parent(), known_inputs.collect()) {
            Err(ProgramError::UnsupportedOperation { .. }) => {
                return context.fold_or_residualize(self.clone(), vec![body.to_program()], inputs);
            }
            known_outputs => known_outputs?,
        };
        let known_outputs = known_program
            .output_ids()
            .iter()
            .zip(known_outputs)
            .map(|(output, value)| match known_program.input_ids().iter().position(|input| input == output) {
                Some(position) => inputs[known_input_indices[position]].clone(),
                None => context.known_value(value),
            })
            .collect::<Vec<_>>();
        let known_output_count = outputs.iter().filter(|output| output.is_known()).count();
        let residual_inputs = residual_inputs
            .iter()
            .map(|input| match input {
                PartialEvaluationInput::Unknown(index) => inputs[*index].clone(),
                PartialEvaluationInput::Known(index) => known_outputs[known_output_count + index].clone(),
            })
            .collect::<Vec<_>>();
        let residual_outputs =
            context.residualize(residual_operation, vec![residual_program], residual_inputs.as_slice())?;
        Ok(outputs
            .iter()
            .map(|output| match output {
                PartialEvaluationOutput::Known(index) => known_outputs[*index].clone(),
                PartialEvaluationOutput::Unknown(index) => residual_outputs[*index].clone(),
            })
            .collect())
    }
}

impl<T: 'static + Type, C: Context<Type = T, Operation: From<RematerializeOperation<T>>>, P: BatchingPolicy<C>>
    BatchableOperation<C, P> for RematerializeOperation<T>
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        // Batching batches the body with its natural output axes and binds the same call over the batched body.
        // Any aBatchingPolicy::boundary_inputs` (e.g., the first-class mapped extent of a composite program) become
        // additional leading instruction inputs, and therefore leading body inputs, of the batched call, whose tangents
        // are zero-space. As for `linear_call`, a completely replicated call at an unnamed batching level is bound
        // unchanged, and the body is specialized to the packed input types, so that type views that agree only for
        // dense batches (e.g., ragged inputs packed at their declared bounds) reconcile with the exact interface that
        // inference requires.
        let input_axes = inputs.iter().map(P::batch_axis).collect::<Vec<_>>();
        let input_values = inputs.iter().map(P::value).cloned().collect::<Vec<_>>();
        if input_axes.iter().all(BatchAxis::is_replicated) && context.axis_name().is_none() {
            let regions = driver.regions().map(|region| region.to_program()).collect::<Vec<_>>();
            let outputs = context.parent().bind(self.clone(), regions, input_values.as_slice())?;
            return Ok(outputs.into_iter().map(P::replicated).collect::<Vec<_>>().into());
        }
        let body = driver.region(0)?;
        let batched_body =
            driver.batch_program(context, body, input_axes.as_slice(), ProgramBatchingOutputAxesPolicy::Natural)?;
        let output_axes = batched_body.output_axes().to_vec();
        let batched_body = context.align_and_adapt_batched_program_outputs(
            driver,
            body,
            input_axes.as_slice(),
            batched_body,
            output_axes.as_slice(),
        )?;
        let mut packed_inputs = P::boundary_inputs(context.axis_extent());
        let boundary_input_count = packed_inputs.len();
        packed_inputs.extend(input_values);
        let packed_input_types = packed_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let logical_input_types = inputs.iter().map(|input| P::unbatched_type(input).into_owned()).collect::<Vec<_>>();
        let logical_output_types = body.to_program().specialize(&logical_input_types)?.output_types();
        let batched_body = batched_body.specialize(packed_input_types.as_slice())?;
        let operation = self.with_remapped_optimization_barrier(|selected| {
            std::iter::repeat_n(false, boundary_input_count).chain(selected.iter().copied()).collect()
        });
        let outputs = context.parent().bind(operation, vec![batched_body], packed_inputs.as_slice())?;
        check_count!("output", outputs, output_axes.len(), ProgramError);
        check_count!("output", logical_output_types, outputs.len(), ProgramError);
        Ok(outputs
            .into_iter()
            .zip(output_axes.into_iter().zip(logical_output_types))
            .map(|(output, (axis, logical_type))| driver.restore_batch(output, axis, &logical_type, inputs))
            .collect::<Result<Vec<_>, _>>()?
            .into())
    }
}

impl<
    C: Context<
            Type: 'static + DifferentiableType,
            Operation: From<RematerializeOperation<C::Type>>
                           + OperationPayloadProjection
                           + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>,
        >,
> DifferentiableOperation<C> for RematerializeOperation<C::Type>
where
    for<'t> &'t ArrayType: TryFrom<&'t C::Type>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // Differentiation follows JAX's `remat_jvp`. The body is differentiated with respect to the inputs that have
        // live tangents (structural zeros get no tangent slot, as JAX drops symbolic-zero tangents), and the driver
        // decides whether that uses the `jvp` or the `jvp_for_transpose` rules of the body, so the default
        // `jvp_for_transpose` of this operation delegates here. When the primal and tangent contexts are shared, the
        // call is bound over the fused derivative program with its incoming flags, which interprets it in eager
        // contexts and keeps the rematerialization boundary in staged ones, for later transforms. Otherwise (e.g., for
        // linearization and reverse-mode differentiation), the fused program is partitioned with this call's residual
        // policy: its known program computes the primal outputs together with the values that the policy saves in the
        // primal context, and its residual program recomputes everything else in the tangent context as a
        // differentiated call.
        let body = driver.region(0)?;
        let output_count = body.output_types().len();
        check_count!("input", inputs, body.input_types().len(), ProgramError);
        let active_input_indices = inputs
            .iter()
            .enumerate()
            .filter_map(|(index, input)| match input.tangent() {
                MaybeZero::Value(tangent) if !tangent.r#type().is_zero_space() => Some(index),
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut fused_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        if active_input_indices.is_empty() {
            let outputs = context.primal().bind(self.clone(), vec![body.to_program()], fused_inputs.as_slice())?;
            return outputs.into_iter().map(DifferentiationDual::new_with_zero_tangent).collect();
        }

        // The fused program maps the primal inputs followed by the live input tangents to the primal outputs followed
        // by the output tangents that are not structurally zero.
        let output_activity = body.tangent_output_mask(active_input_indices.as_slice())?;
        let fused = driver.jvp_program(body, active_input_indices.as_slice())?;
        fused_inputs
            .extend(active_input_indices.iter().map(|&index| inputs[index].tangent().as_value().unwrap().clone()));

        // The optimization barrier of the fused call additionally selects every live input tangent.
        let fused_operation = self.with_remapped_optimization_barrier(|selected| {
            selected.iter().copied().chain(std::iter::repeat_n(true, active_input_indices.len())).collect()
        });

        let mut outputs = if std::ptr::eq(context.primal(), context.tangent()) {
            context.primal().bind(fused_operation, vec![(*fused).clone()], fused_inputs.as_slice())?
        } else {
            let mut input_known = vec![true; inputs.len()];
            input_known.resize(fused_inputs.len(), false);
            let required_known_outputs = (0..output_count).collect::<Vec<_>>();
            let partition = driver.partition_jvp_program_with_residual_policy(
                fused.entry_region_ref(),
                input_known.as_slice(),
                required_known_outputs.as_slice(),
                &self.policy,
            )?;

            // Look through memory transfers and round before offloading, so known consumers and derivative consumers
            // use the same rounded value. Rounding after the transfer would leave known consumers using the value
            // computed with excess precision.
            let partition = partition.with_rounded_residuals(Self::excess_precision_rounding, |operation| {
                operation.projected_payload::<TransferToMemoryOperation>().is_some()
            })?;
            let residual_operation =
                fused_operation.with_residual_optimization_barrier(&partition).with_differentiated(true);
            partition.interpret_in_context_with(context, fused_inputs.as_slice(), output_count, |program, inputs| {
                Ok(context.tangent().bind(residual_operation, vec![program.clone()], inputs.as_slice())?)
            })?
        };

        check_count!(
            "output",
            outputs,
            output_count + output_activity.iter().filter(|active| **active).count(),
            ProgramError,
        );
        let mut tangents = outputs.split_off(output_count).into_iter();
        outputs
            .into_iter()
            .zip(output_activity)
            .map(|(primal, active)| match active {
                true => DifferentiationDual::new(primal, MaybeZero::Value(tangents.next().unwrap())),
                false => DifferentiationDual::new_with_zero_tangent(primal),
            })
            .collect()
    }
}

impl<
    V: Value<Type: 'static + DifferentiableType + ReferenceMemberType>,
    O: Operation<Type = V::Type>
        + From<AddOperation<V::Type>>
        + From<RematerializeOperation<V::Type>>
        + ResidualZeroProvider<V::Type, Operation = O>
        + OperationProvider<
            V::Type,
            ReferenceNewOperation<<V::Type as ReferenceMemberType>::Referent, V::Type>,
            Operation = O,
        >,
> TransposableOperation<V, O> for RematerializeOperation<V::Type>
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        // Transposition follows JAX's `remat_transpose`. The body is transposed with respect to the unknown (i.e.,
        // linear) inputs, and the call is bound with its incoming flags and policy over the transposed body,
        // so that the backward computation keeps its rematerialization boundary and higher-order derivatives keep
        // rematerializing. Any work over the known inputs of the body (e.g., values that the residual program
        // recomputes from the saved ones) is replayed inside the transposed body. The transposed call consumes the
        // cotangents of the non-reference outputs, the cotangent references of the reference inputs whose state
        // cotangents are live, and the known inputs, in that order, and it produces the cotangents of the linear
        // inputs that return one.
        let body = driver.region(0)?;
        check_count!("input", inputs, body.input_types().len(), ProgramError);
        check_count!("output", outputs, body.output_types().len(), ProgramError);
        check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
        let cotangents = context.cotangent_destinations(driver, inputs, accumulators)?;

        // A call without live output cotangents and without live reference state is a zero linear map, unless its body
        // has deferred work or observable effects that must run with structurally zero cotangents.
        if outputs.iter().all(MaybeZero::is_zero)
            && !cotangents.has_reference_state_destinations()
            && !body.must_transpose()
        {
            return Ok(());
        }

        let (linear_input_indices, destination_kinds): (Vec<_>, Vec<_>) = inputs
            .iter()
            .zip(cotangents.kinds())
            .enumerate()
            .filter_map(|(index, (input, kind))| input.is_unknown().then_some((index, *kind)))
            .unzip();
        let transposed_body = driver.transpose_program(body, &linear_input_indices, &destination_kinds)?;

        // A reference output forwards an input root whose state cotangent lives in the cotangent reference of that
        // input, so it has no cotangent input. A structurally zero output cotangent is materialized from the live
        // cotangents and the known inputs, which carry any runtime dimensions that its type refers to.
        let known_inputs = inputs.iter().filter_map(PartialValue::as_known).cloned().collect::<Vec<_>>();
        let mut call_inputs = Vec::with_capacity(outputs.len() + cotangents.references().len() + known_inputs.len());
        for (cotangent, output_type) in outputs.iter().zip(body.output_types().iter()) {
            if !output_type.is_reference() {
                call_inputs.push(O::materialize_zero_from_residual_sources(
                    &**context,
                    cotangent.clone(),
                    outputs.iter().filter_map(MaybeZero::as_value).chain(&known_inputs),
                )?);
            }
        }
        call_inputs.extend(cotangents.references().iter().cloned());
        let cotangent_input_count = call_inputs.len();
        call_inputs.extend(known_inputs);

        // The optimization barrier of the transposed call selects every cotangent input, followed by the selection
        // of the known inputs.
        let operation = self.with_remapped_optimization_barrier(|selected| {
            std::iter::repeat_n(true, cotangent_input_count)
                .chain(
                    inputs.iter().zip(selected).filter_map(|(input, selected)| input.is_known().then_some(*selected)),
                )
                .collect()
        });
        let input_cotangents =
            context.bind(O::from(operation), CalleeRegionDriver::new(&[transposed_body]), call_inputs.as_slice())?;
        let returned_cotangent_count =
            linear_input_indices.iter().filter(|&&index| cotangents.returns_cotangent(index)).count();
        check_count!("output", input_cotangents, returned_cotangent_count, ProgramError);

        // Linear inputs that return a cotangent receive the outputs of the transposed call in input order. The
        // cotangent reference of a reference input was accumulated into in place and is returned by identity, and
        // every other input receives a structural zero.
        let mut input_cotangents = input_cotangents.into_iter();
        for (index, (input, accumulator)) in inputs.iter().zip(accumulators).enumerate() {
            let cotangent = match cotangents.kind(index) {
                CotangentDestinationKind::Return if input.is_unknown() => {
                    MaybeZero::Value(input_cotangents.next().unwrap())
                }
                CotangentDestinationKind::Reference => {
                    if cotangents.is_reference_input(index) {
                        input_cotangents.next();
                    }
                    MaybeZero::Zero(input.r#type().cotangent()?)
                }
                CotangentDestinationKind::Return | CotangentDestinationKind::Ignore => {
                    MaybeZero::Zero(input.r#type().cotangent()?)
                }
            };
            accumulator.accumulate(context, cotangent)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReference, ArrayType, DataType,
        DimensionBounds, DimensionType, DimensionValue, LogicalMesh, Memory, MeshAxis, MeshAxisType, Sharding,
        ShardingDimension,
    };
    use crate::batching::{BatchAxis, ProgramBatchingOutputAxesPolicy};
    use crate::captures::{CaptureReference, ClosedProgram};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{
        CotangentDestination, CotangentSeed, DotsSaveable, EverythingSaveable, NothingSaveable,
        OffloadDotsWithNoBatchDimensions, SaveOnlyTheseNames, differentiate_at, rematerialize,
    };
    use crate::operations::arithmetic::{AddOperation, MulOperation};
    use crate::operations::collectives::parallel_reduce::{ParallelReduceOperation, ParallelReductionKind};
    use crate::operations::comparisons::{CompareOperation, ComparisonDirection};
    use crate::operations::control_flow::scan::ScanOperation;
    use crate::operations::control_flow::r#while::WhileOperation;
    use crate::operations::custom_functions::functions::custom_function;
    use crate::operations::debugging::PrintOperation;
    use crate::operations::dimensions::dimension_add::DimensionAddOperation;
    use crate::operations::dot::{Dot, DotDimensionNumbers, DotOperation};
    use crate::operations::manipulation::memory::TransferToMemory;
    use crate::operations::references::{
        ReferenceAddUpdateOperation, ReferenceFreezeOperation, ReferenceNewOperation, ReferenceReadOperation,
    };
    use crate::operations::tagging::Tag;
    use crate::operations::trigonometric::{Cos, CosOperation, Sin, SinOperation};
    use crate::parameters::Placeholder;
    use crate::partial::ResidualPolicy;
    use crate::programs::{
        EffectClasses, ExternalReferenceBinding, Program, ProgramBuilder, ReferenceSource, ReferenceType, RegionDriver,
        RegionRef,
    };
    use crate::tracing::{Tracer, TracingContext};

    use super::*;

    type TestIrValue = ArrayIrValue<Array>;
    type TestIrOperation = ArrayIrOperation<Array>;
    type TestTracer = Tracer<TracingContext<Array, ArrayOperation<Array>>>;

    // The transform tests below stage complete programs rather than using the `check_operation_*` macros, because their
    // subject is the transformation of the attached body region, for which those macros recommend explicit setup. They
    // also compare derivatives with analytic values rather than with `check_gradient!`, which instantiates the function
    // under test over both linearization tracers and concrete arrays, while a rematerialized function fixes the type of
    // its tracer input.

    /// Returns a [`RematerializeOperation`] over [`ArrayType`] with the default [`NothingSaveable`] policy.
    fn rematerialize_operation() -> RematerializeOperation<ArrayType> {
        RematerializeOperation::new(ResidualPolicyReference::new(NothingSaveable))
    }

    /// Builds the body `x ↦ sin(x) * x` over `f64[]` scalars.
    fn sine_product_body() -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![sine, x], None).unwrap()[0];
        builder
            .build::<Vec<Array>, Vec<Array>>(vec![product], vec![Placeholder], vec![Placeholder])
            .unwrap()
    }

    /// Returns the dot product dimensions that contract two vectors.
    fn vector_dot_dimensions() -> DotDimensionNumbers {
        DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![])
    }

    /// Builds the program `(x, y) ↦ rematerialize(sin(dot(x, x)) * y)` over an `f64[3]` vector `x` and an `f64[]`
    /// scalar `y`, whose call uses the provided operation.
    fn sine_of_dot_program(
        operation: RematerializeOperation<ArrayType>,
    ) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let vector_type = ArrayType::new_static(DataType::F64, [3]);
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut body = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let x = body.add_input(vector_type.clone());
        let y = body.add_input(scalar_type.clone());
        let dot = DotOperation::new(vector_dot_dimensions());
        let dot = body.add_instruction(dot, Vec::new(), vec![x, x], None).unwrap()[0];
        let sine = body.add_instruction(SinOperation::new(), Vec::new(), vec![dot], None).unwrap()[0];
        let product = body.add_instruction(MulOperation::new(), Vec::new(), vec![sine, y], None).unwrap()[0];
        let body = body
            .build::<Vec<Array>, Vec<Array>>(vec![product], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let body = builder.import_program(body);
        let x = builder.add_input(vector_type);
        let y = builder.add_input(scalar_type);
        let output = builder.add_instruction(operation, vec![body], vec![x, y], None).unwrap()[0];
        builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap()
    }

    /// Computes `sin(tag(dot(x, x), "dot"))` for a vector `x`.
    fn tagged_sine_of_dot(x: TestTracer) -> Result<TestTracer, ProgramError> {
        Ok(x.dot(&x, &vector_dot_dimensions())?.tag("dot")?.sin()?)
    }

    /// Differentiates [`tagged_sine_of_dot`] rematerialized with `policy` at `[0.1, 0.2, 0.3]` in reverse mode and
    /// returns its value, its gradient, and the residuals that its pullback saved.
    fn tagged_sine_of_dot_vjp<P: Clone + ResidualPolicy<ArrayType>>(policy: P) -> (Array, Array, Vec<Array>) {
        let function = rematerialize(tagged_sine_of_dot).with_policy(policy);
        let x = Array::vector(vec![0.1f64, 0.2, 0.3]).unwrap();
        let (value, pullback) = differentiate_at(x).vjp(|x| function.call(x)).unwrap();
        let gradient = pullback.apply(Array::scalar(1.0f64).unwrap()).unwrap();
        (value, gradient, pullback.residuals().to_vec())
    }

    #[test]
    fn test_rematerialization_optimization_barrier_selects() {
        assert!(RematerializationOptimizationBarrier::All.selects(0));
        assert!(RematerializationOptimizationBarrier::All.selects(7));
        assert!(!RematerializationOptimizationBarrier::None.selects(0));
        let barrier = RematerializationOptimizationBarrier::Inputs(vec![true, false]);
        assert!(barrier.selects(0));
        assert!(!barrier.selects(1));

        // Indices beyond the selection are not selected.
        assert!(!barrier.selects(2));
    }

    #[test]
    fn test_rematerialize() {
        let policy = ResidualPolicyReference::<ArrayType>::new(NothingSaveable);
        let operation = RematerializeOperation::new(policy.clone());
        assert_eq!(operation.policy(), &policy);
        assert_eq!(operation.optimization_barrier(), &RematerializationOptimizationBarrier::All);
        assert!(!operation.differentiated());
        assert_eq!(operation.name(), REMATERIALIZE_OPERATION_NAME);
        assert_eq!(operation.region_slots(), &[RegionSlot::computation("body")]);
        assert_eq!(operation.input_region_provenance(0, 2), InputRegionProvenance::Input { index: 2 });
        assert_eq!(operation.input_region_provenance(1, 0), InputRegionProvenance::None);
        assert_eq!(
            operation.output_region_provenance(1),
            vec![OutputRegionProvenance { region_index: 0, output_index: 1 }],
        );

        // Only the fields that differ from their defaults render, and the default policy is omitted.
        assert_eq!(operation.to_string(), "rematerialize");
        let configured = RematerializeOperation::new(ResidualPolicyReference::<ArrayType>::new(DotsSaveable))
            .with_optimization_barrier(RematerializationOptimizationBarrier::None)
            .with_differentiated(true);
        assert_eq!(configured.optimization_barrier(), &RematerializationOptimizationBarrier::None);
        assert!(configured.differentiated());
        assert_eq!(
            configured.to_string(),
            "rematerialize [policy=\"dots_saveable\", optimization_barrier=false, differentiated=true]",
        );
        assert_eq!(operation.clone().with_differentiated(true).to_string(), "rematerialize [differentiated=true]");
        assert_eq!(
            operation
                .clone()
                .with_optimization_barrier(RematerializationOptimizationBarrier::Inputs(vec![true, false]))
                .to_string(),
            "rematerialize [optimization_barrier=[true, false]]",
        );

        // Operations compare and hash by the identity of their policy definition and by their flags.
        assert_eq!(operation.clone(), operation);
        assert_ne!(operation.clone().with_differentiated(true), operation);
        assert_ne!(rematerialize_operation(), operation);
        let operations = HashSet::from([operation.clone(), configured.clone()]);
        assert!(operations.contains(&operation));
        assert!(operations.contains(&configured));
        assert!(!operations.contains(&operation.clone().with_differentiated(true)));
        assert!(!operations.contains(&rematerialize_operation()));
    }

    #[test]
    fn test_rematerialize_lift() {
        // Lifting keeps the identity of the policy and the flags of the operation.
        let operation = RematerializeOperation::new(ResidualPolicyReference::<ArrayType>::new(DotsSaveable))
            .with_optimization_barrier(RematerializationOptimizationBarrier::Inputs(vec![false, true]))
            .with_differentiated(true);
        let lifted = operation.lift::<ArrayIrType>();
        assert_eq!(lifted.policy().id(), operation.policy().id());
        assert_eq!(lifted.policy().name(), "dots_saveable");
        assert_eq!(lifted.optimization_barrier(), &RematerializationOptimizationBarrier::Inputs(vec![false, true]));
        assert!(lifted.differentiated());
    }

    #[test]
    fn test_rematerialize_type_inference() {
        let scalar = ArrayType::scalar(DataType::F64);
        let operation = rematerialize_operation();
        let body =
            RegionInterface::new(vec![scalar.clone()], vec![scalar.clone(), scalar.clone()], EffectClasses::NONE);

        // The outputs are the outputs of the body, whose inputs must match the instruction inputs.
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&scalar), std::slice::from_ref(&body)),
            Ok(vec![scalar.clone(), scalar.clone()]),
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&scalar), &[]),
            Err(TypeError::invalid("expected 1 region but got 0")),
        );
        assert_eq!(
            operation.infer_output_types(&[ArrayType::scalar(DataType::F32)], std::slice::from_ref(&body)),
            Err(TypeError::invalid(
                "`rematerialize` body input type signature mismatch: expected [f32[]] but got [f64[]]",
            )),
        );
        assert_eq!(
            operation
                .clone()
                .with_optimization_barrier(RematerializationOptimizationBarrier::Inputs(vec![true, false]))
                .infer_output_types(std::slice::from_ref(&scalar), std::slice::from_ref(&body)),
            Err(TypeError::invalid("`rematerialize` optimization barrier has 2 entries but the call has 1 inputs")),
        );

        // The body is requested at the instruction input types, which staging instantiates when their type identities
        // differ from the declared ones. The output of the body below is a computed dimension, whose identity the
        // instantiation derives from the input's dimension while keeping its diagnostic label.
        assert_eq!(
            operation.infer_region_input_types(std::slice::from_ref(&scalar), std::slice::from_ref(&body)),
            Ok(vec![Some(vec![scalar.clone()])]),
        );
        assert_eq!(
            operation.infer_region_input_types(std::slice::from_ref(&scalar), &[]),
            Err(TypeError::invalid("expected 1 region but got 0")),
        );
        assert_eq!(
            operation.infer_region_input_types(&[scalar.clone(), scalar.clone()], std::slice::from_ref(&body)),
            Err(TypeError::invalid("declared type count 1 does not match actual type count 2")),
        );
        let bounds = DimensionBounds::new(1, Some(5)).unwrap();
        let extent = DimensionType::new("n", bounds);
        let two = DimensionValue::constant(2).unwrap();
        let addition = DimensionAddOperation::new(&extent, two.r#type().as_ref()).unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let input = builder.add_input(extent.into());
        let two = builder.add_constant(TestIrValue::Dimension(two));
        let sum = builder.add_instruction(addition, Vec::new(), vec![input, two], None).unwrap()[0];
        let body = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![sum], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(body.output_types()[0].to_string(), "dimension<n + 2 ∈ [3, 7)>");
        let (_, program) = TracingContext::<TestIrValue, TestIrOperation>::trace(
            |input: Tracer<TracingContext<TestIrValue, TestIrOperation>>| {
                let context = input.context().clone();
                Ok(context.bind(operation.lift::<ArrayIrType>(), vec![body], std::slice::from_ref(&input))?.remove(0))
            },
            ArrayIrType::from(DimensionType::new("m", bounds)),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<m ∈ [1, 5)> .
                let %1:dimension<n + 2 ∈ [3, 7)> = rematerialize %0 [
                    body={
                        lambda %0:dimension<m ∈ [1, 5)> .
                        let %1:dimension<2> = const 2
                            %2:dimension<n + 2 ∈ [3, 7)> = dimension_add %0 %1
                        in (%2)
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_rematerialize_boundary_pruning() {
        // The body maps `(x, y)` to `(sin(x), y)`, so using only the first output drops the input `y` and the second
        // output, together with the optimization barrier selection entry of `y`.
        let scalar = ArrayType::scalar(DataType::F64);
        let mut body = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let x = body.add_input(scalar.clone());
        let y = body.add_input(scalar.clone());
        let sine = body.add_instruction(SinOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
        let body = body
            .build::<Vec<Array>, Vec<Array>>(vec![sine, y], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let body = builder.import_program(body);
        let x = builder.add_input(scalar.clone());
        let y = builder.add_input(scalar);
        let operation = rematerialize_operation()
            .with_optimization_barrier(RematerializationOptimizationBarrier::Inputs(vec![false, true]));
        let outputs = builder.add_instruction(operation, vec![body], vec![x, y], None).unwrap().to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![outputs[0]], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.into_pruned().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = rematerialize [optimization_barrier=[false]] %0 [
                    body={
                        lambda %0:f64[] .
                        let %1:f64[] = sin %0
                        in (%1)
                    },
                ]
                in (%2)"},
        );
    }

    #[test]
    fn test_rematerialize_discharge_references() {
        // The body adds `x` into the reference `r` and returns the value that it then holds. Discharging threads the
        // state of the reference through the body positionally and publishes the mutated state.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = body.add_input(reference_type.clone().into());
        let x = body.add_input(scalar_type.clone().into());
        body.add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, x], None)
            .unwrap();
        let value = body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let reference = builder.add_input(reference_type.into());
        let x = builder.add_input(scalar_type.into());
        let output = builder
            .add_instruction(rematerialize_operation().lift::<ArrayIrType>(), vec![body], vec![reference, x], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[], %3:f32[] = rematerialize %0 %1 [
                    body={
                        lambda %0:f32[], %1:f32[] .
                        let %2:f32[] = add %0 %1
                        in (%2, %2)
                    },
                ]
                in (%2, %3)"},
        );
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(1))],
        );
    }

    #[test]
    fn test_rematerialize_discharge_references_captured_reference() {
        // The body `x ↦ (c += x; read(c))` mutates a captured outer reference `c`. Discharging appends the state of `c`
        // as an input of the call, which carries no derivative values and which its optimization barrier therefore does
        // not select, and publishes the mutated state of `c`.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let mut body =
            ProgramBuilder::<CaptureReference<ArrayIrType>, ArrayIrOperation<CaptureReference<ArrayType>>>::new();
        let x = body.add_input(scalar_type.clone().into());
        let reference = body.add_constant(CaptureReference::new(0, reference_type.into()));
        body.add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, x], None)
            .unwrap();
        let value = body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let body = body
            .build::<Vec<CaptureReference<ArrayIrType>>, Vec<CaptureReference<ArrayIrType>>>(
                vec![value],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let mut builder =
            ProgramBuilder::<CaptureReference<ArrayIrType>, ArrayIrOperation<CaptureReference<ArrayType>>>::new();
        let body = builder.import_program(body);
        let x = builder.add_input(scalar_type.into());
        let operation = rematerialize_operation()
            .with_optimization_barrier(RematerializationOptimizationBarrier::Inputs(vec![true]))
            .lift::<ArrayIrType>();
        let output = builder.add_instruction(operation, vec![body], vec![x], None).unwrap()[0];
        let program = builder
            .build::<Vec<CaptureReference<ArrayIrType>>, Vec<CaptureReference<ArrayIrType>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let reference = ArrayReference::new(Array::scalar(1.0f32).unwrap());
        let closed = ClosedProgram::new(program, vec![TestIrValue::Reference(reference)]).unwrap();
        let discharged = closed.discharge_references().unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[], %3:f32[] = rematerialize [optimization_barrier=[true, false]] %1 %0 [
                    body={
                        lambda %0:f32[], %1:f32[] .
                        let %2:f32[] = add %1 %0
                        in (%2, %2)
                    },
                ]
                in (%2, %3)"},
        );
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, Some(1))],
        );
    }

    #[test]
    fn test_rematerialize_interpretation() {
        // Outside of differentiation, the call computes what its body computes.
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let outputs = context
            .bind(rematerialize_operation(), vec![sine_product_body()], &[Array::scalar(0.5f64).unwrap()])
            .unwrap();
        assert_eq!(outputs, vec![Array::scalar(0.5f64.sin() * 0.5).unwrap()]);
    }

    #[test]
    fn test_rematerialize_partial_evaluation() {
        // With a known `x` and an unknown `y`, the policy decides what the known side computes: saving nothing forwards
        // `x` to a differentiated call that recomputes the dot product, while saving dot products hoists the dot
        // product out of the call and recomputes only the sine.
        let program = sine_of_dot_program(rematerialize_operation());
        assert_eq!(
            program.partition(&[true, false]).unwrap().to_string(),
            indoc! {"
                partition [known_inputs=[0], residual_inputs=[Unknown(1), Known(0)], outputs=[Unknown(0)]]
                known={
                    lambda %0:f64[3] .
                    in (%0)
                }
                residual={
                    lambda %0:f64[], %1:f64[3] .
                    let %2:f64[] = rematerialize [differentiated=true] %0 %1 [
                        body={
                            lambda %0:f64[], %1:f64[3] .
                            let %2:f64[] = dot [
                                dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            ] %1 %1
                                %3:f64[] = sin %2
                                %4:f64[] = mul %3 %0
                            in (%4)
                        },
                    ]
                    in (%2)
                }"},
        );
        let program = sine_of_dot_program(RematerializeOperation::new(ResidualPolicyReference::new(DotsSaveable)));
        assert_eq!(
            program.partition(&[true, false]).unwrap().to_string(),
            indoc! {"
                partition [known_inputs=[0], residual_inputs=[Unknown(1), Known(0)], outputs=[Unknown(0)]]
                known={
                    lambda %0:f64[3] .
                    let %1:f64[] = dot [
                        dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    ] %0 %0
                    in (%1)
                }
                residual={
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = rematerialize [policy=\"dots_saveable\", differentiated=true] %0 %1 [
                        body={
                            lambda %0:f64[], %1:f64[] .
                            let %2:f64[] = sin %1
                                %3:f64[] = mul %2 %0
                            in (%3)
                        },
                    ]
                    in (%2)
                }"},
        );

        // A call whose inputs are all known folds whole, which keeps it undifferentiated on the known side.
        assert_eq!(
            program.partition(&[true, true]).unwrap().to_string(),
            indoc! {"
                partition [known_inputs=[0, 1], residual_inputs=[], outputs=[Known(0)]]
                known={
                    lambda %0:f64[3], %1:f64[] .
                    let %2:f64[] = rematerialize [policy=\"dots_saveable\"] %0 %1 [
                        body={
                            lambda %0:f64[3], %1:f64[] .
                            let %2:f64[] = dot [
                                dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            ] %0 %0
                                %3:f64[] = sin %2
                                %4:f64[] = mul %3 %1
                            in (%4)
                        },
                    ]
                    in (%2)
                }
                residual={
                    lambda  .
                    in ()
                }"},
        );

        // The residual call keeps the optimization barrier selection of the unknown `y` and of the known `x`, which the
        // known program forwards unchanged, while it selects the dot product that the known program computes.
        let operation = rematerialize_operation()
            .with_optimization_barrier(RematerializationOptimizationBarrier::Inputs(vec![false, true]));
        assert_eq!(
            sine_of_dot_program(operation).partition(&[true, false]).unwrap().to_string(),
            indoc! {"
                partition [known_inputs=[0], residual_inputs=[Unknown(1), Known(0)], outputs=[Unknown(0)]]
                known={
                    lambda %0:f64[3] .
                    in (%0)
                }
                residual={
                    lambda %0:f64[], %1:f64[3] .
                    let %2:f64[] = rematerialize [optimization_barrier=[true, false], differentiated=true] %0 %1 [
                        body={
                            lambda %0:f64[], %1:f64[3] .
                            let %2:f64[] = dot [
                                dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            ] %1 %1
                                %3:f64[] = sin %2
                                %4:f64[] = mul %3 %0
                            in (%4)
                        },
                    ]
                    in (%2)
                }"},
        );
        let operation = RematerializeOperation::new(ResidualPolicyReference::new(DotsSaveable))
            .with_optimization_barrier(RematerializationOptimizationBarrier::Inputs(vec![false, false]));
        assert_eq!(
            sine_of_dot_program(operation).partition(&[true, false]).unwrap().to_string(),
            indoc! {"
                partition [known_inputs=[0], residual_inputs=[Unknown(1), Known(0)], outputs=[Unknown(0)]]
                known={
                    lambda %0:f64[3] .
                    let %1:f64[] = dot [
                        dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    ] %0 %0
                    in (%1)
                }
                residual={
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = rematerialize [policy=\"dots_saveable\", optimization_barrier=[false, true], differentiated=true] %0 %1 [
                        body={
                            lambda %0:f64[], %1:f64[] .
                            let %2:f64[] = sin %1
                                %3:f64[] = mul %2 %0
                            in (%3)
                        },
                    ]
                    in (%2)
                }"},
        );

        // A call over a reference input stays whole and undifferentiated on the residual side, because replaying its
        // known side outside of the call would bypass the reference placement of the active partial evaluation.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = body.add_input(reference_type.clone().into());
        let x = body.add_input(scalar_type.clone().into());
        let value = body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let sine = ArrayOperation::<Array>::from(SinOperation::<ArrayType>::new());
        let sine = body.add_instruction(sine, Vec::new(), vec![value], None).unwrap()[0];
        let product = ArrayOperation::<Array>::from(MulOperation::<ArrayType>::new());
        let product = body.add_instruction(product, Vec::new(), vec![sine, x], None).unwrap()[0];
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![product], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let reference = builder.add_input(reference_type.into());
        let x = builder.add_input(scalar_type.into());
        let output = builder
            .add_instruction(rematerialize_operation().lift::<ArrayIrType>(), vec![body], vec![reference, x], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.partition(&[true, false]).unwrap().to_string(),
            indoc! {"
                partition [known_inputs=[0], residual_inputs=[Unknown(1), Known(0)], outputs=[Unknown(0)]]
                known={
                    lambda %0:ref<f32[]> .
                    in (%0)
                }
                residual={
                    lambda %0:f32[], %1:ref<f32[]> .
                    let %2:f32[] = rematerialize %1 %0 [
                        body={
                            lambda %0:ref<f32[]>, %1:f32[] .
                            let %2:f32[] = reference_read %0
                                %3:f32[] = sin %2
                                %4:f32[] = mul %3 %1
                            in (%4)
                        },
                    ]
                    in (%2)
                }"},
        );

        // A body with observable effects stays whole and undifferentiated on the residual side as well, because replaying
        // its known side outside of the call would bypass the effect ordering of the active partial evaluation.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut body = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let x = body.add_input(scalar_type.clone());
        let y = body.add_input(scalar_type.clone());
        let printed = body.add_instruction(PrintOperation::new("x"), Vec::new(), vec![x], None).unwrap()[0];
        let product = body.add_instruction(MulOperation::new(), Vec::new(), vec![printed, y], None).unwrap()[0];
        let body = body
            .build::<Vec<Array>, Vec<Array>>(vec![product], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let body = builder.import_program(body);
        let x = builder.add_input(scalar_type.clone());
        let y = builder.add_input(scalar_type);
        let output = builder.add_instruction(rematerialize_operation(), vec![body], vec![x, y], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.partition(&[true, false]).unwrap().to_string(),
            indoc! {"
                partition [known_inputs=[0], residual_inputs=[Unknown(1), Known(0)], outputs=[Unknown(0)]]
                known={
                    lambda %0:f64[] .
                    in (%0)
                }
                residual={
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = rematerialize %1 %0 [
                        body={
                            lambda %0:f64[], %1:f64[] .
                            let %2:f64[] = print [label=x] %0
                                %3:f64[] = mul %2 %1
                            in (%3)
                        },
                    ]
                    in (%2)
                }"},
        );
    }

    #[test]
    fn test_rematerialize_partial_evaluation_forwarded_known_inputs() {
        // `(a, b) ↦ a · b + rematerialize((a, b) ↦ a · b)` with an eager known side forwards the known `a` into the
        // differentiated call, which receives it as the same residual input that the outer product uses.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut body = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let a = body.add_input(scalar_type.clone());
        let b = body.add_input(scalar_type.clone());
        let product = body.add_instruction(MulOperation::new(), Vec::new(), vec![a, b], None).unwrap()[0];
        let body = body
            .build::<Vec<Array>, Vec<Array>>(vec![product], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let body = builder.import_program(body);
        let a = builder.add_input(scalar_type.clone());
        let b = builder.add_input(scalar_type.clone());
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![a, b], None).unwrap()[0];
        let call = builder.add_instruction(rematerialize_operation(), vec![body], vec![a, b], None).unwrap()[0];
        let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![product, call], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![sum], vec![Placeholder; 2], vec![Placeholder]).unwrap();
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(2.0f64).unwrap()),
                PartialValue::Unknown(scalar_type),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %0
                    %3:f64[] = rematerialize [differentiated=true] %0 %1 [
                        body={
                            lambda %0:f64[], %1:f64[] .
                            let %2:f64[] = mul %1 %0
                            in (%2)
                        },
                    ]
                    %4:f64[] = add %2 %3
                in (%4)"},
        );
    }

    #[test]
    fn test_rematerialize_partial_evaluation_pending_error() {
        // A partial evaluation that already retained a binding error reports that error before it hoists the known side
        // of a call into the known-side context, which would otherwise stage the dot product into the enclosing trace.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let context = PartialEvaluationContext::new(trace.clone());
        let left = context.lift(Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap()).unwrap();
        let right = context.lift(Array::vector(vec![1.0f64, 2.0]).unwrap()).unwrap();
        let error = context.bind(MulOperation::new(), Vec::new(), &[left, right]).err().unwrap();
        assert_eq!(error, ProgramError::Type(TypeError::invalid("`mul` input types are not broadcast-compatible")));
        let program = sine_of_dot_program(RematerializeOperation::new(ResidualPolicyReference::new(DotsSaveable)));
        let inputs = vec![
            PartialEvaluationValue::known(trace.input(ArrayType::new_static(DataType::F64, [3]))),
            context.unknown_input(ArrayType::scalar(DataType::F64), 1),
        ];
        assert_eq!(context.inline_program(&program, inputs).err(), Some(error));
        assert!(trace.builder().borrow().instructions().is_empty());

        // Without a pending error, the same call hoists the dot product into the enclosing trace.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let context = PartialEvaluationContext::new(trace.clone());
        let inputs = vec![
            PartialEvaluationValue::known(trace.input(ArrayType::new_static(DataType::F64, [3]))),
            context.unknown_input(ArrayType::scalar(DataType::F64), 1),
        ];
        assert!(context.inline_program(&program, inputs).is_ok());
        let instructions = trace
            .builder()
            .borrow()
            .instructions()
            .iter()
            .map(|instruction| instruction.operation().name())
            .collect::<Vec<_>>();
        assert_eq!(instructions, vec!["dot"]);
    }

    #[test]
    fn test_rematerialize_partial_evaluation_unsupported_known_operation() {
        // The body `(x, y) ↦ parallel_sum(x) · y` reduces `x` over a manual mesh axis, which an eager known side cannot
        // execute. Saving everything would hoist that reduction out of the call, so the call stays whole and
        // undifferentiated on the residual side instead, for an execution backend that owns the mesh.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 0);
        let invariant_type = ArrayType::scalar(DataType::F32).with_sharding(sharding.clone()).unwrap();
        let varying_type = ArrayType::scalar(DataType::F32)
            .with_sharding(sharding.with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        let mut body = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let x = body.add_input(varying_type.clone());
        let y = body.add_input(invariant_type.clone());
        let sum = ParallelReduceOperation::new("m".to_string(), ParallelReductionKind::Sum).with_mesh(mesh);
        let sum = body.add_instruction(sum, Vec::new(), vec![x], None).unwrap()[0];
        let product = body.add_instruction(MulOperation::new(), Vec::new(), vec![sum, y], None).unwrap()[0];
        let body = body
            .build::<Vec<Array>, Vec<Array>>(vec![product], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let body = builder.import_program(body);
        let x = builder.add_input(varying_type.clone());
        let y = builder.add_input(invariant_type.clone());
        let operation = RematerializeOperation::new(ResidualPolicyReference::new(EverythingSaveable));
        let output = builder.add_instruction(operation, vec![body], vec![x, y], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::from_elements(varying_type, &[2.0f32]).unwrap()),
                PartialValue::Unknown(invariant_type),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['m'=4:manual]>, []}], %1:f32[][sharding={mesh<['m'=4:manual]>, [], varying_manual={'m'}}] .
                let %2:f32[][sharding={mesh<['m'=4:manual]>, []}] = rematerialize [policy=\"everything_saveable\"] %1 %0 [
                    body={
                        lambda %0:f32[][sharding={mesh<['m'=4:manual]>, [], varying_manual={'m'}}], %1:f32[][sharding={mesh<['m'=4:manual]>, []}] .
                        let %2:f32[][sharding={mesh<['m'=4:manual]>, []}] = parallel_sum [axis_name=\"m\", mesh=['m'=4:manual]] %0
                            %3:f32[][sharding={mesh<['m'=4:manual]>, []}] = mul %2 %1
                        in (%3)
                    },
                ]
                in (%2)"},
        );
    }

    #[test]
    fn test_rematerialize_batching() {
        // Batching batches the body with its natural output axes and keeps the rematerialization boundary.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let body = builder.import_program(sine_product_body());
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(rematerialize_operation(), vec![body], vec![x], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let (batched, output_axes) = program
            .batched(2, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f64[2] .
                let %1:f64[2] = rematerialize %0 [
                    body={
                        lambda %0:f64[2] .
                        let %1:f64[2] = sin %0
                            %2:f64[2] = mul %1 %0
                        in (%2)
                    },
                ]
                in (%1)"},
        );
        assert_eq!(
            batched.interpret(vec![Array::vector(vec![0.5f64, 1.5]).unwrap()]),
            Ok(vec![Array::vector(vec![0.5f64.sin() * 0.5, 1.5f64.sin() * 1.5]).unwrap()]),
        );

        // A completely replicated call at an unnamed batching level is bound unchanged.
        let (batched, output_axes) = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::replicated()]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = rematerialize %0 [
                    body={
                        lambda %0:f64[] .
                        let %1:f64[] = sin %0
                            %2:f64[] = mul %1 %0
                        in (%2)
                    },
                ]
                in (%1)"},
        );

        // A mapped reference input enters the batched body as a reference to the stacked values, and the body may
        // forward it as an output. The threaded extent becomes a leading input of the batched call, which its
        // optimization barrier does not select.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = body.add_input(reference_type.clone().into());
        let x = body.add_input(scalar_type.clone().into());
        body.add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, x], None)
            .unwrap();
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference, x], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let reference = builder.add_input(reference_type.into());
        let x = builder.add_input(scalar_type.into());
        let operation = rematerialize_operation()
            .with_optimization_barrier(RematerializationOptimizationBarrier::Inputs(vec![true, true]));
        let outputs = builder
            .add_instruction(operation.lift::<ArrayIrType>(), vec![body], vec![reference, x], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let extent = DimensionValue::constant(3).unwrap();
        let (batched, output_axes) = program
            .batched_with_threaded_extent(
                extent.r#type().into_owned(),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0), BatchAxis::new(0)]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:dimension<3>, %1:ref<f32[3]>, %2:f32[3] .
                let %3:ref<f32[3]>, %4:f32[3] = rematerialize [optimization_barrier=[false, true, true]] %0 %1 %2 [
                    body={
                        lambda %0:dimension<3>, %1:ref<f32[3]>, %2:f32[3] .
                        let () = reference_add_update %1 %2
                        in (%1, %2)
                    },
                ]
                in (%0, %3, %4)"},
        );
        let counter = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let outputs = batched
            .interpret(vec![
                TestIrValue::Dimension(extent),
                TestIrValue::Reference(counter.clone()),
                TestIrValue::Array(Array::vector(vec![10.0f32, 20.0, 30.0]).unwrap()),
            ])
            .unwrap();
        assert_eq!(outputs.len(), 3);
        assert_eq!(counter.read(), Ok(Array::vector(vec![11.0f32, 22.0, 33.0]).unwrap()));
    }

    #[test]
    fn test_rematerialize_differentiation() {
        // Forward mode in a shared staged context binds the same call over the fused derivative program, keeping its
        // policy and flags.
        let operation = RematerializeOperation::new(ResidualPolicyReference::new(DotsSaveable))
            .with_optimization_barrier(RematerializationOptimizationBarrier::None)
            .with_differentiated(true);
        assert_eq!(
            sine_of_dot_program(operation).jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[], %2:f64[3], %3:f64[] .
                let %4:f64[], %5:f64[] = rematerialize [policy=\"dots_saveable\", optimization_barrier=false, differentiated=true] %0 %1 %2 %3 [
                    body={
                        lambda %0:f64[3], %1:f64[], %2:f64[3], %3:f64[] .
                        let %4:f64[] = dot [
                            dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        ] %0 %0
                            %5:f64[] = dot [
                                dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            ] %2 %0
                            %6:f64[] = dot [
                                dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            ] %0 %2
                            %7:f64[] = add %5 %6
                            %8:f64[] = sin %4
                            %9:f64[] = cos %4
                            %10:f64[] = mul %9 %7
                            %11:f64[] = mul %8 %1
                            %12:f64[] = mul %1 %10
                            %13:f64[] = mul %8 %3
                            %14:f64[] = add %12 %13
                        in (%11, %14)
                    },
                ]
                in (%4, %5)"},
        );

        // Inputs whose tangents are zero-space (e.g., integer inputs) get no tangent slot in the fused call, and a call
        // without any live input tangent is bound unchanged over its primal inputs, with structurally zero output
        // tangents.
        let body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let x = builder.add_input(ArrayType::scalar(DataType::F64));
            let n = builder.add_input(ArrayType::scalar(DataType::I64));
            let square = builder.add_instruction(MulOperation::new(), Vec::new(), vec![x, x], None).unwrap()[0];
            let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![n, n], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![square, product], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let body = builder.import_program(body);
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let n = builder.add_input(ArrayType::scalar(DataType::I64));
        let outputs =
            builder.add_instruction(rematerialize_operation(), vec![body], vec![x, n], None).unwrap().to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[], %1:i64[], %2:f64[] .
                let %3:f64[], %4:i64[], %5:f64[] = rematerialize %0 %1 %2 [
                    body={
                        lambda %0:f64[], %1:i64[], %2:f64[] .
                        let %3:f64[] = mul %0 %0
                            %4:f64[] = mul %0 %2
                            %5:f64[] = mul %0 %2
                            %6:f64[] = add %4 %5
                            %7:i64[] = mul %1 %1
                        in (%3, %7, %6)
                    },
                ]
                in (%3, %4, %5)"},
        );
        let body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let n = builder.add_input(ArrayType::scalar(DataType::I64));
            let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![n, n], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![product], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let body = builder.import_program(body);
        let n = builder.add_input(ArrayType::scalar(DataType::I64));
        let output = builder.add_instruction(rematerialize_operation(), vec![body], vec![n], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:i64[] .
                let %1:i64[] = rematerialize %0 [
                    body={
                        lambda %0:i64[] .
                        let %1:i64[] = mul %0 %0
                        in (%1)
                    },
                ]
                in (%1)"},
        );

        // The optimization barrier of the fused call additionally selects the live input tangents, and that of the
        // differentiated call of a linearization keeps the selection of the primal inputs that it receives unchanged.
        let operation = rematerialize_operation()
            .with_optimization_barrier(RematerializationOptimizationBarrier::Inputs(vec![false, true]));
        assert_eq!(
            sine_of_dot_program(operation.clone()).jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[], %2:f64[3], %3:f64[] .
                let %4:f64[], %5:f64[] = rematerialize [optimization_barrier=[false, true, true, true]] %0 %1 %2 %3 [
                    body={
                        lambda %0:f64[3], %1:f64[], %2:f64[3], %3:f64[] .
                        let %4:f64[] = dot [
                            dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        ] %0 %0
                            %5:f64[] = dot [
                                dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            ] %2 %0
                            %6:f64[] = dot [
                                dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            ] %0 %2
                            %7:f64[] = add %5 %6
                            %8:f64[] = sin %4
                            %9:f64[] = cos %4
                            %10:f64[] = mul %9 %7
                            %11:f64[] = mul %8 %1
                            %12:f64[] = mul %1 %10
                            %13:f64[] = mul %8 %3
                            %14:f64[] = add %12 %13
                        in (%11, %14)
                    },
                ]
                in (%4, %5)"},
        );
        let linearization = sine_of_dot_program(operation).linearize().unwrap();
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[], %2:f64[3], %3:f64[] .
                let %4:f64[] = rematerialize [optimization_barrier=[true, true, false, true], differentiated=true] %0 %1 %2 %3 [
                    body={
                        lambda %0:f64[3], %1:f64[], %2:f64[3], %3:f64[] .
                        let %4:f64[] = dot [
                            dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        ] %2 %2
                            %5:f64[] = cos %4
                            %6:f64[] = dot [
                                dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            ] %0 %2
                            %7:f64[] = dot [
                                dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            ] %2 %0
                            %8:f64[] = add %6 %7
                            %9:f64[] = mul %5 %8
                            %10:f64[] = mul %3 %9
                            %11:f64[] = sin %4
                            %12:f64[] = mul %11 %1
                            %13:f64[] = add %10 %12
                        in (%13)
                    },
                ]
                in (%4)"},
        );

        // Linearization partitions the fused program with the policy of the call: its primal program computes the
        // values that the policy saves, and its tangent program recomputes the others in a differentiated call.
        let linearization = sine_of_dot_program(rematerialize_operation()).linearize().unwrap();
        assert_eq!(
            format!("{}\n{}", linearization.primal(), linearization.tangent()),
            indoc! {"
                lambda %0:f64[3], %1:f64[] .
                let %2:f64[] = dot [
                    dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                ] %0 %0
                    %3:f64[] = sin %2
                    %4:f64[] = mul %3 %1
                in (%4, %0, %1)
                lambda %0:f64[3], %1:f64[], %2:f64[3], %3:f64[] .
                let %4:f64[] = rematerialize [differentiated=true] %0 %1 %2 %3 [
                    body={
                        lambda %0:f64[3], %1:f64[], %2:f64[3], %3:f64[] .
                        let %4:f64[] = dot [
                            dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        ] %2 %2
                            %5:f64[] = cos %4
                            %6:f64[] = dot [
                                dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            ] %0 %2
                            %7:f64[] = dot [
                                dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            ] %2 %0
                            %8:f64[] = add %6 %7
                            %9:f64[] = mul %5 %8
                            %10:f64[] = mul %3 %9
                            %11:f64[] = sin %4
                            %12:f64[] = mul %11 %1
                            %13:f64[] = add %10 %12
                        in (%13)
                    },
                ]
                in (%4)"},
        );
        let operation = RematerializeOperation::new(ResidualPolicyReference::new(DotsSaveable));
        let linearization = sine_of_dot_program(operation).linearize().unwrap();
        assert_eq!(
            format!("{}\n{}", linearization.primal(), linearization.tangent()),
            indoc! {"
                lambda %0:f64[3], %1:f64[] .
                let %2:f64[] = dot [
                    dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                ] %0 %0
                    %3:f64[] = sin %2
                    %4:f64[] = mul %3 %1
                in (%4, %0, %1, %2)
                lambda %0:f64[3], %1:f64[], %2:f64[3], %3:f64[], %4:f64[] .
                let %5:f64[] = rematerialize [policy=\"dots_saveable\", differentiated=true] %0 %1 %2 %3 %4 [
                    body={
                        lambda %0:f64[3], %1:f64[], %2:f64[3], %3:f64[], %4:f64[] .
                        let %5:f64[] = cos %4
                            %6:f64[] = dot [
                                dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            ] %0 %2
                            %7:f64[] = dot [
                                dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            ] %2 %0
                            %8:f64[] = add %6 %7
                            %9:f64[] = mul %5 %8
                            %10:f64[] = mul %3 %9
                            %11:f64[] = sin %4
                            %12:f64[] = mul %11 %1
                            %13:f64[] = add %10 %12
                        in (%13)
                    },
                ]
                in (%5)"},
        );
        let operation = RematerializeOperation::new(ResidualPolicyReference::new(EverythingSaveable));
        let linearization = sine_of_dot_program(operation).linearize().unwrap();
        assert_eq!(
            format!("{}\n{}", linearization.primal(), linearization.tangent()),
            indoc! {"
                lambda %0:f64[3], %1:f64[] .
                let %2:f64[] = dot [
                    dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                ] %0 %0
                    %3:f64[] = sin %2
                    %4:f64[] = mul %3 %1
                    %5:f64[] = cos %2
                in (%4, %0, %5, %1, %3)
                lambda %0:f64[3], %1:f64[], %2:f64[3], %3:f64[], %4:f64[], %5:f64[] .
                let %6:f64[] = rematerialize [policy=\"everything_saveable\", differentiated=true] %0 %1 %2 %3 %4 %5 [
                    body={
                        lambda %0:f64[3], %1:f64[], %2:f64[3], %3:f64[], %4:f64[], %5:f64[] .
                        let %6:f64[] = dot [
                            dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        ] %0 %2
                            %7:f64[] = dot [
                                dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            ] %2 %0
                            %8:f64[] = add %6 %7
                            %9:f64[] = mul %3 %8
                            %10:f64[] = mul %4 %9
                            %11:f64[] = mul %5 %1
                            %12:f64[] = add %10 %11
                        in (%12)
                    },
                ]
                in (%6)"},
        );
    }

    #[test]
    fn test_rematerialize_differentiation_policies() {
        // The policy of the call decides which residuals its pullback saves, while the value and the gradient are the
        // same for every policy: saving nothing saves only `x`, saving everything also saves the cosine, saving dot
        // products also saves the dot product, offloading policies save it in host memory, and name-based policies
        // follow the tags in the body. The classification of every built-in policy is tested in the `policies` module.
        // The gradient of `sin(x · x)` is `2 cos(x · x) x`.
        let x = [0.1f64, 0.2, 0.3];
        let dot = 0.14f64;
        let value = Array::scalar(dot.sin()).unwrap();
        let gradient = Array::vector(x.iter().map(|x| 2.0 * dot.cos() * x).collect::<Vec<_>>()).unwrap();
        let vector = Array::vector(x.to_vec()).unwrap();
        let dot = Array::scalar(dot).unwrap();
        let cosine = Array::scalar(0.14f64.cos()).unwrap();
        let host = Memory::Host { pinned: true };
        assert_eq!(tagged_sine_of_dot_vjp(NothingSaveable), (value.clone(), gradient.clone(), vec![vector.clone()]));
        assert_eq!(
            tagged_sine_of_dot_vjp(EverythingSaveable),
            (value.clone(), gradient.clone(), vec![vector.clone(), cosine]),
        );
        assert_eq!(
            tagged_sine_of_dot_vjp(DotsSaveable),
            (value.clone(), gradient.clone(), vec![vector.clone(), dot.clone()]),
        );
        assert_eq!(
            tagged_sine_of_dot_vjp(OffloadDotsWithNoBatchDimensions::new(host)),
            (value.clone(), gradient.clone(), vec![vector.clone(), dot.transfer_to_memory(host).unwrap()]),
        );
        assert_eq!(tagged_sine_of_dot_vjp(SaveOnlyTheseNames::new(["dot"])), (value, gradient, vec![vector, dot]));
    }

    #[test]
    fn test_rematerialize_differentiation_entry_points() {
        // Forward mode, linearization, and reverse mode agree, and linearization saves the residuals that the policy
        // selects: `x` and the dot product.
        let function = rematerialize(tagged_sine_of_dot).with_policy(DotsSaveable);
        let x = Array::vector(vec![0.1f64, 0.2, 0.3]).unwrap();
        let tangent = Array::vector(vec![1.0f64, 0.0, 0.0]).unwrap();
        let value = Array::scalar(0.14f64.sin()).unwrap();
        let output_tangent = Array::scalar(0.14f64.cos() * 0.2).unwrap();
        assert_eq!(
            differentiate_at(x.clone()).jvp(tangent.clone(), |x| function.call(x)).unwrap(),
            (value.clone(), output_tangent.clone()),
        );
        let (linearized_value, pushforward) = differentiate_at(x.clone()).linearize(|x| function.call(x)).unwrap();
        assert_eq!(linearized_value, value);
        assert_eq!(pushforward.residuals(), &[x.clone(), Array::scalar(0.14f64).unwrap()]);
        assert_eq!(pushforward.apply(tangent), Ok(output_tangent));
        let gradient = Array::vector(vec![0.2 * 0.14f64.cos(), 0.4 * 0.14f64.cos(), 0.6 * 0.14f64.cos()]).unwrap();
        assert_eq!(differentiate_at(x).value_and_gradient(|x| function.call(x)).unwrap(), (value, gradient));
    }

    #[test]
    fn test_rematerialize_differentiation_custom_function_body() {
        // The body calls a custom function whose JVP rule doubles the derivative of `sin` and whose VJP rule triples
        // it. Forward mode through the call uses the JVP rule, while reverse mode uses the VJP rule, because the
        // driver selects the rule with which the body is differentiated.
        let sine = custom_function(|x: TestTracer| Ok(x.sin()?))
            .with_jvp(|x, tangent| {
                let tangent = x.cos()? * tangent;
                Ok((x.sin()?, tangent.clone() + tangent))
            })
            .with_vjp(
                |x: TestTracer| Ok((x.sin()?, x.cos()?)),
                |cosine, cotangent| {
                    let cotangent = cosine * cotangent;
                    Ok(cotangent.clone() + cotangent.clone() + cotangent)
                },
            );
        let function = rematerialize(move |x: TestTracer| sine.call(x));
        let x = Array::scalar(0.5f64).unwrap();
        assert_eq!(
            differentiate_at(x.clone()).jvp(Array::scalar(1.0f64).unwrap(), |x| function.call(x)).unwrap(),
            (Array::scalar(0.5f64.sin()).unwrap(), Array::scalar(2.0 * 0.5f64.cos()).unwrap()),
        );
        assert_eq!(differentiate_at(x).gradient(|x| function.call(x)), Ok(Array::scalar(3.0 * 0.5f64.cos()).unwrap()));
    }

    #[test]
    fn test_rematerialize_differentiation_higher_order() {
        // The staged gradient of `sin(x²)` keeps the differentiated call that recomputes `cos(x²)`, and so does
        // differentiating that gradient in forward mode (i.e., forward-over-reverse Hessian-vector products) and in
        // reverse mode (i.e., the gradient of the gradient), as well as the gradient of its forward-mode derivative.
        let function = rematerialize(|x: TestTracer| Ok((x.clone() * x).sin()?));
        let (_, gradient) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x: TestTracer| Ok(differentiate_at(x).gradient(|x| function.call(x))?),
            ArrayType::scalar(DataType::F64),
        )
        .unwrap();
        let gradient = gradient.into_flat_program();
        assert_eq!(
            gradient.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = mul %0 %0
                    %2:f64[] = sin %1
                    %3:f64[] = one [type=f64[]]
                    %4:f64[] = rematerialize [differentiated=true] %3 %0 [
                        body={
                            lambda %0:f64[], %1:f64[] .
                            let %2:f64[] = mul %1 %1
                                %3:f64[] = cos %2
                                %4:f64[] = mul %3 %0
                                %5:f64[] = mul %1 %4
                                %6:f64[] = mul %1 %4
                                %7:f64[] = add %5 %6
                            in (%7)
                        },
                    ]
                in (%4)"},
        );
        let hessian_vector_product = gradient.jvp().unwrap();
        assert_eq!(
            hessian_vector_product.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %0 %0
                    %3:f64[] = mul %0 %1
                    %4:f64[] = mul %0 %1
                    %5:f64[] = add %3 %4
                    %6:f64[] = sin %2
                    %7:f64[] = cos %2
                    %8:f64[] = mul %7 %5
                    %9:f64[] = one [type=f64[]]
                    %10:f64[], %11:f64[] = rematerialize [differentiated=true] %9 %0 %1 [
                        body={
                            lambda %0:f64[], %1:f64[], %2:f64[] .
                            let %3:f64[] = mul %1 %1
                                %4:f64[] = mul %1 %2
                                %5:f64[] = mul %1 %2
                                %6:f64[] = add %4 %5
                                %7:f64[] = cos %3
                                %8:f64[] = sin %3
                                %9:f64[] = mul %8 %6
                                %10:f64[] = neg %9
                                %11:f64[] = mul %7 %0
                                %12:f64[] = mul %0 %10
                                %13:f64[] = mul %1 %11
                                %14:f64[] = mul %11 %2
                                %15:f64[] = mul %1 %12
                                %16:f64[] = add %14 %15
                                %17:f64[] = mul %1 %11
                                %18:f64[] = mul %11 %2
                                %19:f64[] = mul %1 %12
                                %20:f64[] = add %18 %19
                                %21:f64[] = add %13 %17
                                %22:f64[] = add %16 %20
                            in (%21, %22)
                        },
                    ]
                in (%10, %11)"},
        );
        let gradient_of_gradient = gradient.linearize().unwrap().pullback().unwrap();
        assert_eq!(
            gradient_of_gradient.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                let %4:f64[] = rematerialize [differentiated=true] %0 %1 %3 [
                    body={
                        lambda %0:f64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = mul %1 %0
                            %4:f64[] = mul %1 %1
                            %5:f64[] = cos %4
                            %6:f64[] = mul %5 %2
                            %7:f64[] = mul %6 %0
                            %8:f64[] = mul %1 %0
                            %9:f64[] = add %3 %8
                            %10:f64[] = mul %2 %9
                            %11:f64[] = neg %10
                            %12:f64[] = sin %4
                            %13:f64[] = mul %12 %11
                            %14:f64[] = mul %1 %13
                            %15:f64[] = add %7 %14
                            %16:f64[] = mul %1 %13
                            %17:f64[] = add %15 %16
                            %18:f64[] = mul %6 %0
                            %19:f64[] = add %17 %18
                        in (%19)
                    },
                ]
                in (%4)"},
        );

        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x: TestTracer| function.call(x),
            ArrayType::scalar(DataType::F64),
        )
        .unwrap();
        let gradient_of_jvp = program.into_flat_program().jvp().unwrap().linearize().unwrap().pullback().unwrap();
        assert_eq!(
            gradient_of_jvp.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                let %4:f64[], %5:f64[] = rematerialize [differentiated=true] %0 %1 %2 %3 [
                    body={
                        lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = mul %2 %2
                            %5:f64[] = cos %4
                            %6:f64[] = mul %5 %1
                            %7:f64[] = mul %2 %6
                            %8:f64[] = mul %3 %6
                            %9:f64[] = mul %2 %6
                            %10:f64[] = add %7 %9
                            %11:f64[] = mul %3 %6
                            %12:f64[] = add %8 %11
                            %13:f64[] = mul %2 %3
                            %14:f64[] = mul %2 %3
                            %15:f64[] = add %13 %14
                            %16:f64[] = mul %15 %1
                            %17:f64[] = neg %16
                            %18:f64[] = sin %4
                            %19:f64[] = mul %18 %17
                            %20:f64[] = cos %4
                            %21:f64[] = mul %20 %0
                            %22:f64[] = add %19 %21
                            %23:f64[] = mul %2 %22
                            %24:f64[] = add %12 %23
                            %25:f64[] = mul %2 %22
                            %26:f64[] = add %24 %25
                        in (%26, %10)
                    },
                ]
                in (%4, %5)"},
        );

        // The second derivative of `sin(x²)` is `2 cos(x²) - 4 x² sin(x²)`.
        let x = 0.5f64;
        let second_derivative = 2.0 * (x * x).cos() - 4.0 * x * x * (x * x).sin();
        let outputs = hessian_vector_product.interpret(vec![Array::scalar(x).unwrap(), Array::scalar(1.0).unwrap()]);
        assert_eq!(
            outputs,
            Ok(vec![Array::scalar(2.0 * x * (x * x).cos()).unwrap(), Array::scalar(second_derivative).unwrap()]),
        );
    }

    #[test]
    fn test_rematerialize_differentiation_batching() {
        // Batching a pullback keeps its differentiated call, and so does differentiating a batched program.
        let linearization = sine_of_dot_program(rematerialize_operation()).linearize().unwrap();
        let (batched_pullback, _) = linearization
            .pullback()
            .unwrap()
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(
            batched_pullback.to_string(),
            indoc! {"
                lambda %0:f64[2], %1:f64[2, 3], %2:f64[2] .
                let %3:f64[2, 3], %4:f64[2] = rematerialize [differentiated=true] %0 %1 %2 [
                    body={
                        lambda %0:f64[2], %1:f64[2, 3], %2:f64[2] .
                        let %3:f64[2] = dot [
                            dimensions=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[0], rhs_batching=[0]),
                        ] %1 %1
                            %4:f64[2] = cos %3
                            %5:f64[2] = mul %2 %0
                            %6:f64[2] = mul %4 %5
                            %7:f64[2, 3] = dot [
                                dimensions=(lhs_contracting=[], rhs_contracting=[], lhs_batching=[0], rhs_batching=[0]),
                            ] %1 %6
                            %8:f64[2, 3] = dot [
                                dimensions=(lhs_contracting=[], rhs_contracting=[], lhs_batching=[0], rhs_batching=[0]),
                            ] %6 %1
                            %9:f64[2, 3] = add %7 %8
                            %10:f64[2] = sin %3
                            %11:f64[2] = mul %10 %0
                        in (%9, %11)
                    },
                ]
                in (%3, %4)"},
        );
        let (batched, _) = sine_of_dot_program(rematerialize_operation())
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        let pullback_of_batched = batched.linearize().unwrap().pullback().unwrap();
        assert_eq!(
            pullback_of_batched.to_string(),
            indoc! {"
                lambda %0:f64[2], %1:f64[2, 3], %2:f64[2] .
                let %3:f64[2, 3], %4:f64[2] = rematerialize [differentiated=true] %0 %1 %2 [
                    body={
                        lambda %0:f64[2], %1:f64[2, 3], %2:f64[2] .
                        let %3:f64[2] = dot [
                            dimensions=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[0], rhs_batching=[0]),
                        ] %1 %1
                            %4:f64[2] = sin %3
                            %5:f64[2] = mul %4 %0
                            %6:f64[2] = mul %2 %0
                            %7:f64[2] = cos %3
                            %8:f64[2] = mul %7 %6
                            %9:f64[2, 3] = dot [
                                dimensions=(lhs_contracting=[], rhs_contracting=[], lhs_batching=[0], rhs_batching=[0]),
                            ] %1 %8
                            %10:f64[2, 3] = dot [
                                dimensions=(lhs_contracting=[], rhs_contracting=[], lhs_batching=[0], rhs_batching=[0]),
                            ] %8 %1
                            %11:f64[2, 3] = add %9 %10
                        in (%11, %5)
                    },
                ]
                in (%3, %4)"},
        );
    }

    #[test]
    fn test_rematerialize_differentiation_scan_body() {
        // The body scans `c * cos(dot(x, x))` over the rows `x` of `xs`. Saving dot products applies to every iteration
        // of the scan: the primal scan stacks the dot products, and the differentiated call recomputes their cosines.
        let scan_body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let _index = builder.add_input(ArrayType::scalar(DataType::I64));
            let c = builder.add_input(ArrayType::scalar(DataType::F64));
            let x = builder.add_input(ArrayType::new_static(DataType::F64, [3]));
            let dot = DotOperation::new(vector_dot_dimensions());
            let dot = builder.add_instruction(dot, Vec::new(), vec![x, x], None).unwrap()[0];
            let cosine = builder.add_instruction(CosOperation::new(), Vec::new(), vec![dot], None).unwrap()[0];
            let next = builder.add_instruction(MulOperation::new(), Vec::new(), vec![c, cosine], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![next], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let scan_body = builder.import_program(scan_body);
            let c = builder.add_input(ArrayType::scalar(DataType::F64));
            let xs = builder.add_input(ArrayType::new_static(DataType::F64, [2, 3]));
            let scan = ScanOperation::new(1, 2);
            let output = builder.add_instruction(scan, vec![scan_body], vec![c, xs], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let body = builder.import_program(body);
        let c = builder.add_input(ArrayType::scalar(DataType::F64));
        let xs = builder.add_input(ArrayType::new_static(DataType::F64, [2, 3]));
        let operation = RematerializeOperation::new(ResidualPolicyReference::new(DotsSaveable));
        let output = builder.add_instruction(operation, vec![body], vec![c, xs], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(
            format!("{}\n{}", linearization.primal(), linearization.tangent()),
            indoc! {"
                lambda %0:f64[], %1:f64[2, 3] .
                let %2:f64[], %3:f64[2], %4:f64[2] = scan [carry_count=1, length=2, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[3] .
                        let %3:f64[] = dot [
                            dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        ] %2 %2
                            %4:f64[] = cos %3
                            %5:f64[] = mul %1 %4
                        in (%5, %1, %3)
                    },
                ]
                in (%2, %1, %3, %4)
                lambda %0:f64[], %1:f64[2, 3], %2:f64[2, 3], %3:f64[2], %4:f64[2] .
                let %5:f64[] = rematerialize [policy=\"dots_saveable\", differentiated=true] %0 %1 %2 %3 %4 [
                    body={
                        lambda %0:f64[], %1:f64[2, 3], %2:f64[2, 3], %3:f64[2], %4:f64[2] .
                        let %5:f64[] = scan [carry_count=1, length=2, reverse=false] %0 %1 %2 %3 %4 [
                            body={
                                lambda %0:i64[], %1:f64[], %2:f64[3], %3:f64[3], %4:f64[], %5:f64[] .
                                let %6:f64[] = cos %5
                                    %7:f64[] = mul %6 %1
                                    %8:f64[] = sin %5
                                    %9:f64[] = dot [
                                        dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                                    ] %2 %3
                                    %10:f64[] = dot [
                                        dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                                    ] %3 %2
                                    %11:f64[] = add %9 %10
                                    %12:f64[] = mul %8 %11
                                    %13:f64[] = neg %12
                                    %14:f64[] = mul %4 %13
                                    %15:f64[] = add %7 %14
                                in (%15)
                            },
                        ]
                        in (%5)
                    },
                ]
                in (%5)"},
        );
    }

    #[test]
    fn test_rematerialize_differentiation_unbounded_while_body() {
        // The body `while (x < 16) { x = x * x }` has no reverse-mode derivative, but evaluating it and differentiating
        // it in forward mode derive nothing that only reverse mode needs. At `x = 2`, the loop computes `x⁴` locally.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let condition = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let x = builder.add_input(scalar_type.clone());
            let threshold = builder.add_constant(Array::scalar(16.0f64).unwrap());
            let comparison = CompareOperation::new(ComparisonDirection::LessThan);
            let predicate = builder.add_instruction(comparison, Vec::new(), vec![x, threshold], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let while_body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let x = builder.add_input(scalar_type.clone());
            let square = builder.add_instruction(MulOperation::new(), Vec::new(), vec![x, x], None).unwrap()[0];
            builder.build::<Vec<Array>, Vec<Array>>(vec![square], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let condition = builder.import_program(condition);
            let while_body = builder.import_program(while_body);
            let x = builder.add_input(scalar_type.clone());
            let output =
                builder.add_instruction(WhileOperation::new(), vec![condition, while_body], vec![x], None).unwrap()[0];
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let body = builder.import_program(body);
        let x = builder.add_input(scalar_type);
        let output = builder.add_instruction(rematerialize_operation(), vec![body], vec![x], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(program.interpret(vec![Array::scalar(2.0f64).unwrap()]), Ok(vec![Array::scalar(16.0f64).unwrap()]));
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.interpret(vec![Array::scalar(2.0f64).unwrap(), Array::scalar(1.0f64).unwrap()]),
            Ok(vec![Array::scalar(16.0f64).unwrap(), Array::scalar(32.0f64).unwrap()]),
        );
    }

    #[test]
    fn test_rematerialize_differentiation_excess_precision() {
        // Saving everything for `x ↦ sin(x)²` over `bf16` saves `sin(x)` and `cos(x)`. The primal program also squares
        // `sin(x)`, so it is rounded to `bf16` right after its producer, and both its primal consumer and the tangent
        // program observe the rounded value even when a backend computes `sin(x)` in a wider type. The cosine feeds
        // only the tangent program and is not rounded.
        let scalar_type = ArrayType::scalar(DataType::BF16);
        let mut body = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let x = body.add_input(scalar_type.clone());
        let sine = body.add_instruction(SinOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
        let square = body.add_instruction(MulOperation::new(), Vec::new(), vec![sine, sine], None).unwrap()[0];
        let body = body.build::<Vec<Array>, Vec<Array>>(vec![square], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let body = builder.import_program(body);
        let x = builder.add_input(scalar_type);
        let operation = RematerializeOperation::new(ResidualPolicyReference::new(EverythingSaveable));
        let output = builder.add_instruction(operation, vec![body], vec![x], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(
            format!("{}\n{}", linearization.primal(), linearization.tangent()),
            indoc! {"
                lambda %0:bf16[] .
                let %1:bf16[] = sin %0
                    %2:bf16[] = reduce_precision [exponent_bits=8, mantissa_bits=7] %1
                    %3:bf16[] = mul %2 %2
                    %4:bf16[] = cos %0
                in (%3, %4, %2)
                lambda %0:bf16[], %1:bf16[], %2:bf16[] .
                let %3:bf16[] = rematerialize [policy=\"everything_saveable\", differentiated=true] %0 %1 %2 [
                    body={
                        lambda %0:bf16[], %1:bf16[], %2:bf16[] .
                        let %3:bf16[] = mul %1 %0
                            %4:bf16[] = mul %2 %3
                            %5:bf16[] = mul %2 %3
                            %6:bf16[] = add %4 %5
                        in (%6)
                    },
                ]
                in (%3)"},
        );
    }

    #[test]
    fn test_rematerialize_differentiation_external_reference_mutation() {
        // The body `(r, x) ↦ (r += x²; read(r) · x)` mutates an external reference. Its mutation runs once, in the
        // primal computation, and is never recomputed, so the reference holds `r + x²` afterwards, and the derivative
        // of `(r + x²) · x` with respect to `x` is `r + 3x²`, as without rematerialization.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = body.add_input(reference_type.clone().into());
        let x = body.add_input(scalar_type.clone().into());
        let square = ArrayOperation::<Array>::from(MulOperation::<ArrayType>::new());
        let square = body.add_instruction(square, Vec::new(), vec![x, x], None).unwrap()[0];
        body.add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, square], None)
            .unwrap();
        let value = body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let product = ArrayOperation::<Array>::from(MulOperation::<ArrayType>::new());
        let product = body.add_instruction(product, Vec::new(), vec![value, x], None).unwrap()[0];
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![product], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let reference = builder.add_input(reference_type.into());
        let x = builder.add_input(scalar_type.into());
        let output = builder
            .add_instruction(rematerialize_operation().lift::<ArrayIrType>(), vec![body], vec![reference, x], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let scalar = |value: f32| TestIrValue::Array(Array::scalar(value).unwrap());
        let reference = ArrayReference::new(Array::scalar(1.0f32).unwrap());
        let (value, pullback) = differentiate_at((TestIrValue::Reference(reference.clone()), scalar(2.0)))
            .vjp(|(reference, x)| Ok(program.interpret_in_context(&x.dispatch_domain(), vec![reference, x])?.remove(0)))
            .unwrap();
        assert_eq!(value, scalar(10.0));
        assert_eq!(reference.read(), Ok(Array::scalar(5.0f32).unwrap()));
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(scalar(1.0)),
                (CotangentDestination::Ignore, CotangentDestination::Return),
            ),
            Ok((None, Some(scalar(13.0)))),
        );
        assert_eq!(reference.read(), Ok(Array::scalar(5.0f32).unwrap()));
    }

    #[test]
    fn test_rematerialize_transposition() {
        // Transposing the tangent program binds the same differentiated call over the transposed body, which recomputes
        // the cosine of the dot product from the saved `x` before using it.
        let linearization = sine_of_dot_program(rematerialize_operation()).linearize().unwrap();
        let pullback = linearization.pullback().unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3], %2:f64[] .
                let %3:f64[3], %4:f64[] = rematerialize [differentiated=true] %0 %1 %2 [
                    body={
                        lambda %0:f64[], %1:f64[3], %2:f64[] .
                        let %3:f64[] = dot [
                            dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        ] %1 %1
                            %4:f64[] = sin %3
                            %5:f64[] = mul %4 %0
                            %6:f64[] = mul %2 %0
                            %7:f64[] = cos %3
                            %8:f64[] = mul %7 %6
                            %9:f64[3] = dot [
                                dimensions=(lhs_contracting=[], rhs_contracting=[], lhs_batching=[], rhs_batching=[]),
                            ] %1 %8
                            %10:f64[3] = dot [
                                dimensions=(lhs_contracting=[], rhs_contracting=[], lhs_batching=[], rhs_batching=[]),
                            ] %8 %1
                            %11:f64[3] = add %9 %10
                        in (%11, %5)
                    },
                ]
                in (%3, %4)"},
        );

        // The optimization barrier of the transposed call selects its cotangent input, followed by the selection of
        // the known inputs.
        let operation = rematerialize_operation()
            .with_optimization_barrier(RematerializationOptimizationBarrier::Inputs(vec![false, true]));
        let linearization = sine_of_dot_program(operation).linearize().unwrap();
        assert_eq!(
            linearization.pullback().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3], %2:f64[] .
                let %3:f64[3], %4:f64[] = rematerialize [optimization_barrier=[true, false, true], differentiated=true] %0 %1 %2 [
                    body={
                        lambda %0:f64[], %1:f64[3], %2:f64[] .
                        let %3:f64[] = dot [
                            dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        ] %1 %1
                            %4:f64[] = sin %3
                            %5:f64[] = mul %4 %0
                            %6:f64[] = mul %2 %0
                            %7:f64[] = cos %3
                            %8:f64[] = mul %7 %6
                            %9:f64[3] = dot [
                                dimensions=(lhs_contracting=[], rhs_contracting=[], lhs_batching=[], rhs_batching=[]),
                            ] %1 %8
                            %10:f64[3] = dot [
                                dimensions=(lhs_contracting=[], rhs_contracting=[], lhs_batching=[], rhs_batching=[]),
                            ] %8 %1
                            %11:f64[3] = add %9 %10
                        in (%11, %5)
                    },
                ]
                in (%3, %4)"},
        );
    }

    #[test]
    fn test_rematerialize_transposition_zero_linear_map() {
        // A call without live output cotangents and without live reference state is a zero linear map, so its rule
        // stages nothing and leaves structural zeros for its inputs. Program transposition skips such a call before
        // reaching the rule, so the rule is invoked directly, through a driver that exposes the body `(x, t) ↦ x · t`
        // and refuses to transpose it, because a zero linear map never needs its transposed body.
        struct BodyDriver<'r> {
            body: RegionRef<'r, Array, ArrayOperation<Array>>,
        }

        impl RegionDriver<Array, ArrayOperation<Array>> for BodyDriver<'_> {
            fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, Array, ArrayOperation<Array>>>
            where
                Array: 'r,
                ArrayOperation<Array>: 'r,
            {
                std::iter::once(self.body)
            }
        }

        impl TranspositionDriver<Array, ArrayOperation<Array>> for BodyDriver<'_> {
            fn transpose_program(
                &self,
                _region: RegionRef<'_, Array, ArrayOperation<Array>>,
                _input_indices: &[usize],
                _destination_kinds: &[CotangentDestinationKind],
            ) -> Result<Arc<Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>>, DifferentiationError>
            {
                Err(ProgramError::UnsupportedOperation {
                    message: "a zero linear map never transposes its body".to_string(),
                }
                .into())
            }
        }

        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut body = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let x = body.add_input(scalar_type.clone());
        let t = body.add_input(scalar_type.clone());
        let product = body.add_instruction(MulOperation::new(), Vec::new(), vec![x, t], None).unwrap()[0];
        let body = body
            .build::<Vec<Array>, Vec<Array>>(vec![product], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let driver = BodyDriver { body: body.entry_region_ref() };
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let mut context = TranspositionContext::new(trace.clone());
        let inputs =
            [PartialValue::Known(trace.input(scalar_type.clone())), PartialValue::Unknown(scalar_type.clone())];
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        rematerialize_operation()
            .transpose(&mut context, &driver, &inputs, &[MaybeZero::Zero(scalar_type)], &accumulators)
            .unwrap();
        let cotangents = context.take_cotangents(&accumulators).unwrap();
        assert_eq!(cotangents.len(), 2);
        assert!(cotangents.iter().all(MaybeZero::is_zero));
        assert!(trace.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_rematerialize_transposition_local_reference_lifecycle() {
        // The body `x ↦ freeze(new(x) += x²) · x` computes `x² + x³` through a local reference whose lifecycle the
        // pullback recomputes from the saved `x`, afresh for every application.
        let scalar_type = ArrayIrType::from(ArrayType::scalar(DataType::F32));
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let x = body.add_input(scalar_type.clone());
        let reference = body.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
        let square = ArrayOperation::<Array>::from(MulOperation::<ArrayType>::new());
        let square = body.add_instruction(square, Vec::new(), vec![x, x], None).unwrap()[0];
        body.add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, square], None)
            .unwrap();
        let frozen =
            body.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let product = ArrayOperation::<Array>::from(MulOperation::<ArrayType>::new());
        let product = body.add_instruction(product, Vec::new(), vec![frozen, x], None).unwrap()[0];
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![product], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body = builder.import_program(body);
        let x = builder.add_input(scalar_type);
        let output = builder
            .add_instruction(rematerialize_operation().lift::<ArrayIrType>(), vec![body], vec![x], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(
            format!("{}\n{}", linearization.primal(), linearization.tangent()),
            indoc! {"
                lambda %0:f32[] .
                let %1:ref<f32[]> = reference_new %0
                    %2:f32[] = mul %0 %0
                    () = reference_add_update %1 %2
                    %3:f32[] = reference_freeze %1
                    %4:f32[] = mul %3 %0
                in (%4, %0)
                lambda %0:f32[], %1:f32[] .
                let %2:f32[] = rematerialize [differentiated=true] %0 %1 [
                    body={
                        lambda %0:f32[], %1:f32[] .
                        let %2:ref<f32[]> = reference_new %1
                            %3:f32[] = mul %1 %1
                            () = reference_add_update %2 %3
                            %4:f32[] = reference_freeze %2
                            %5:ref<f32[]> = reference_new %0
                            %6:f32[] = mul %1 %0
                            %7:f32[] = mul %1 %0
                            %8:f32[] = add %6 %7
                            () = reference_add_update %5 %8
                            %9:f32[] = reference_freeze %5
                            %10:f32[] = mul %1 %9
                            %11:f32[] = mul %4 %0
                            %12:f32[] = add %10 %11
                        in (%12)
                    },
                ]
                in (%2)"},
        );
        let pullback = linearization.pullback().unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[] = rematerialize [differentiated=true] %0 %1 [
                    body={
                        lambda %0:f32[], %1:f32[] .
                        let %2:ref<f32[]> = reference_new %1
                            %3:f32[] = mul %1 %1
                            () = reference_add_update %2 %3
                            %4:f32[] = reference_freeze %2
                            %5:f32[] = mul %4 %0
                            %6:f32[] = mul %1 %0
                            %7:f32[] = zero [type=f32[]]
                            %8:ref<f32[]> = reference_new %7
                            () = reference_add_update %8 %6
                            %9:f32[] = reference_read %8
                            %10:f32[] = mul %1 %9
                            %11:f32[] = add %5 %10
                            %12:f32[] = mul %1 %9
                            %13:f32[] = add %11 %12
                            %14:f32[] = reference_freeze %8
                            %15:f32[] = add %13 %14
                        in (%15)
                    },
                ]
                in (%2)"},
        );
        let scalar = |value: f32| TestIrValue::Array(Array::scalar(value).unwrap());
        assert_eq!(pullback.interpret(vec![scalar(1.0), scalar(2.0)]), Ok(vec![scalar(16.0)]));
        assert_eq!(pullback.interpret(vec![scalar(1.0), scalar(2.0)]), Ok(vec![scalar(16.0)]));

        // Reverse mode over the call itself recomputes the lifecycle in the pullback too.
        let (value, pullback) = differentiate_at(scalar(2.0))
            .vjp(|x| Ok(program.interpret_in_context(&x.dispatch_domain(), vec![x])?.remove(0)))
            .unwrap();
        assert_eq!(value, scalar(12.0));
        assert_eq!(pullback.apply(scalar(1.0)), Ok(scalar(16.0)));
        assert_eq!(pullback.apply(scalar(1.0)), Ok(scalar(16.0)));
    }
}
