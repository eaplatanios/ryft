use std::fmt::Display;

use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchedProgram, BatchingContext, BatchingDriver, BatchingError,
    BatchingPolicy, ProgramBatchingOutputAxesPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver, DifferentiationDual,
    DifferentiationError, DifferentiationPolicy, NOTHING_SAVEABLE_POLICY_NAME,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, check_types, impl_non_transposable_operation};
use crate::partial::{PartiallyEvaluatableOperation, ResidualPolicyReference};
use crate::programs::{
    InputRegionProvenance, Operation, OperationBoundaryPruning, OperationFormatter, OutputRegionProvenance,
    ProgramError, ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy,
    ReferenceDischargeValue, ReferenceDischargeableOperation, RegionInterface, RegionLiveness, RegionSlot, Type,
    TypeError, Typed, discharge_positional_region_operation,
};

/// Canonical operation name for [`RematerializeOperation`].
pub const REMATERIALIZE_OPERATION_NAME: &str = "rematerialize";

/// [`Operation`] that represents a rematerialized (i.e., checkpointed) call of its attached `body` region, which is the
/// analogue of JAX's [`jax.checkpoint`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html) primitive.
/// Outside of differentiation, the call computes exactly what its body computes, and its outputs are the body's
/// outputs. Under differentiation, the residual policy of the call decides which of the values that the body computes
/// are saved for the derivative computation and which are recomputed from the saved values instead (based on a
/// [`ResidualPolicy`]), which trades computation for the memory that the saved values occupy.
///
/// The operands of the call map positionally onto the inputs of its body, and its outputs map positionally onto the
/// outputs of its body. The call carries no stored derivative regions: the transforms derive the body's derivatives
/// when they need them, so rematerialization composes with every transform that applies to its body.
///
/// The call also records whether it is the residual side of a differentiated computation (via
/// [`differentiated`](Self::differentiated)), in which case backends place an optimization barrier on its inputs when
/// [`optimization_barrier`](Self::optimization_barrier) is set, so that the recomputation is neither merged with the
/// original computation nor scheduled before its inputs are available.
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

    /// Whether backends place an optimization barrier on the inputs of the call when it is
    /// [differentiated](Self::differentiated).
    optimization_barrier: bool,

    /// Whether the call is the residual side of a differentiated computation.
    differentiated: bool,
}

impl<T: Type> RematerializeOperation<T> {
    /// Creates a new [`RematerializeOperation`] with the provided residual policy, which places an optimization barrier
    /// when it is differentiated and is not yet differentiated.
    #[inline]
    pub fn new(policy: ResidualPolicyReference<T>) -> Self {
        Self { policy, optimization_barrier: true, differentiated: false }
    }

    /// Returns this [`RematerializeOperation`] with the provided optimization-barrier flag.
    /// Refer to [`optimization_barrier`](Self::optimization_barrier) for more information.
    #[inline]
    pub fn with_optimization_barrier(mut self, optimization_barrier: bool) -> Self {
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

    /// Returns whether backends place an optimization barrier on the inputs of this call when it is
    /// [differentiated](Self::differentiated). This is the analogue of the `prevent_cse` parameter of
    /// [`jax.checkpoint`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html).
    #[inline]
    pub fn optimization_barrier(&self) -> bool {
        self.optimization_barrier
    }

    /// Returns whether this call is the residual side of a differentiated computation. Splitting a call into the work
    /// that its derivative computation needs up front and the work that it recomputes marks the recomputing call as
    /// differentiated, and every other transform preserves the flag.
    #[inline]
    pub fn differentiated(&self) -> bool {
        self.differentiated
    }

    /// Returns this [`RematerializeOperation`] lifted into a type universe `U` whose types project into `T` (e.g.,
    /// from [`ArrayType`](crate::ArrayType) into [`ArrayIrType`](crate::ArrayIrType)), with its policy lifted through
    /// [`ResidualPolicyReference::lift`], which keeps the identity of the policy, and with its flags unchanged.
    #[inline]
    pub fn lift<U: 'static + Type>(&self) -> RematerializeOperation<U>
    where
        T: 'static,
        for<'t> &'t T: TryFrom<&'t U>,
    {
        RematerializeOperation {
            policy: self.policy.lift(),
            optimization_barrier: self.optimization_barrier,
            differentiated: self.differentiated,
        }
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
        // The body is always requested at the operand types, whose type identities may differ from the declared ones
        // even when the types compare equal, and staging decides whether that requires instantiating or specializing
        // the body.
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
        // Operands map onto the body inputs one for one, so the operands that the body does not use are dropped
        // together with its unused outputs.
        let kept_inputs = regions.used_region_inputs(0, used_outputs)?;
        check_count!("input", kept_inputs, input_count, ProgramError);
        Ok(Some(OperationBoundaryPruning { operation: self.clone(), kept_inputs, kept_outputs: used_outputs.to_vec() }))
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        let operation = OperationFormatter::new(formatter, indentation, REMATERIALIZE_OPERATION_NAME)?;
        let renders_policy = self.policy.name() != NOTHING_SAVEABLE_POLICY_NAME;
        if !renders_policy && self.optimization_barrier && !self.differentiated {
            return Ok(());
        }
        operation.bracketed(|operation| {
            if renders_policy {
                operation.field("policy", format_args!("{:?}", self.policy.name()))?;
            }
            if !self.optimization_barrier {
                operation.field("optimization_barrier", false)?;
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
    #[inline]
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        // A rematerialized call forwards its operands onto its body's inputs one for one and reports the body's outputs
        // as its own, which is the positionally forwarding shape that the shared structured rewrite serves with no
        // leading operands.
        discharge_positional_region_operation(self, context, driver, inputs, 0)
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

// TODO(eaplatanios): Review from here onwards.

// TODO(eaplatanios): Phase B2 of `.tasks/plan_rematerialization_redesign.md` replaces this default, which folds calls
//  whose inputs are all known and otherwise residualizes them unchanged, with a rule that partitions the body according
//  to the residual policy.
impl<C: Context<Type: 'static, Operation: From<RematerializeOperation<C::Type>>>> PartiallyEvaluatableOperation<C>
    for RematerializeOperation<C::Type>
{
}

// Batching batches the body with its natural output axes and binds the same call over the batched body. Any
// `BatchingPolicy::boundary_operands` (e.g., the first-class mapped extent of a composite program) become additional
// leading operands, and therefore leading body inputs, of the batched call, whose tangents are zero-space. As for
// `linear_call`, a completely replicated call at an unnamed batching level is bound unchanged, and the body is
// specialized to the packed operand types, so that type views that agree only for dense batches (e.g., ragged operands
// packed at their declared bounds) reconcile with the exact interface that inference requires.
impl<T: 'static + Type, C: Context<Type = T, Operation: From<RematerializeOperation<T>>>, P: BatchingPolicy<C>>
    BatchableOperation<C, P> for RematerializeOperation<T>
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
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
        let mut packed_inputs = P::boundary_operands(context.axis_extent());
        packed_inputs.extend(input_values);
        let packed_input_types = packed_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let logical_input_types = inputs.iter().map(|input| P::unbatched_type(input).into_owned()).collect::<Vec<_>>();
        let logical_output_types = body.to_program().specialize(&logical_input_types)?.output_types();
        let batched_body = batched_body.specialize(packed_input_types.as_slice())?;
        let outputs = context.parent().bind(self.clone(), vec![batched_body], packed_inputs.as_slice())?;
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

// TODO(eaplatanios): Phase B2 of `.tasks/plan_rematerialization_redesign.md` replaces this rejection with the
//  policy-driven `jvp` and `jvp_for_transpose` rules.
impl<C: Context<Type: 'static + DifferentiableType, Operation: From<RematerializeOperation<C::Type>>>>
    DifferentiableOperation<C> for RematerializeOperation<C::Type>
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        _context: &DifferentiationContext<C, P>,
        _driver: &D,
        _inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!("operation `{REMATERIALIZE_OPERATION_NAME}` is not differentiable yet"),
        }
        .into())
    }
}

// TODO(eaplatanios): Phase B2 of `.tasks/plan_rematerialization_redesign.md` replaces this rejection with the
//  transposition rule that re-binds the call over its transposed body.
impl_non_transposable_operation!(<T> RematerializeOperation<T> where T: 'static + Type);

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReference, ArrayType, DataType,
        DimensionBounds, DimensionType, DimensionValue, ShardingDimension,
    };
    use crate::batching::{BatchAxis, ProgramBatchingOutputAxesPolicy};
    use crate::contexts::EagerContext;
    use crate::differentiation::rematerialization::{DotsSaveable, NothingSaveable};
    use crate::operations::arithmetic::MulOperation;
    use crate::operations::dimensions::DimensionAddOperation;
    use crate::operations::references::{ReferenceAddUpdateOperation, ReferenceReadOperation};
    use crate::operations::trigonometric::SinOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{EffectClasses, Program, ProgramBuilder, ReferenceType};
    use crate::tracing::{Tracer, TracingContext};

    use super::*;

    type TestIrValue = ArrayIrValue<Array>;
    type TestIrOperation = ArrayIrOperation<Array>;

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

    #[test]
    fn test_rematerialize_operation() {
        let policy = ResidualPolicyReference::<ArrayType>::new(NothingSaveable);
        let operation = RematerializeOperation::new(policy.clone());
        assert_eq!(operation.policy(), &policy);
        assert!(operation.optimization_barrier());
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
            .with_optimization_barrier(false)
            .with_differentiated(true);
        assert!(!configured.optimization_barrier());
        assert!(configured.differentiated());
        assert_eq!(
            configured.to_string(),
            "rematerialize [policy=\"dots_saveable\", optimization_barrier=false, differentiated=true]",
        );
        assert_eq!(operation.clone().with_differentiated(true).to_string(), "rematerialize [differentiated=true]");

        // Operations compare by the identity of their policy definition and by their flags.
        assert_eq!(operation.clone(), operation);
        assert_ne!(operation.clone().with_differentiated(true), operation);
        assert_ne!(rematerialize_operation(), operation);
    }

    #[test]
    fn test_rematerialize_operation_lift() {
        // Lifting keeps the identity of the policy and the flags of the operation.
        let operation = RematerializeOperation::new(ResidualPolicyReference::<ArrayType>::new(DotsSaveable))
            .with_optimization_barrier(false)
            .with_differentiated(true);
        let lifted = operation.lift::<ArrayIrType>();
        assert_eq!(lifted.policy().id(), operation.policy().id());
        assert_eq!(lifted.policy().name(), "dots_saveable");
        assert!(!lifted.optimization_barrier());
        assert!(lifted.differentiated());
    }

    #[test]
    fn test_rematerialize_operation_type_inference() {
        let scalar = ArrayType::scalar(DataType::F64);
        let operation = rematerialize_operation();
        let body =
            RegionInterface::new(vec![scalar.clone()], vec![scalar.clone(), scalar.clone()], EffectClasses::NONE);

        // The outputs are the outputs of the body, whose inputs must match the operands.
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

        // The body is requested at the operand types, which staging instantiates when their type identities differ
        // from the declared ones. The output of the body below is a computed dimension, whose identity the
        // instantiation derives from the operand's dimension while keeping its diagnostic label.
        assert_eq!(
            operation.infer_region_input_types(std::slice::from_ref(&scalar), std::slice::from_ref(&body)),
            Ok(vec![Some(vec![scalar.clone()])]),
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
    fn test_rematerialize_operation_boundary_pruning() {
        // The body maps `(x, y)` to `(sin(x), y)`, so using only the first output drops the operand `y` and the second
        // output.
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
        let outputs =
            builder.add_instruction(rematerialize_operation(), vec![body], vec![x, y], None).unwrap().to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![outputs[0]], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.into_pruned().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = rematerialize %0 [
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
    fn test_rematerialize_operation_discharge_references() {
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
        assert_eq!(discharged.external_reference_bindings().len(), 1);
        assert!(discharged.external_reference_bindings()[0].is_mutated());
    }

    #[test]
    fn test_rematerialize_operation_interpretation() {
        // Outside of differentiation, the call computes what its body computes.
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let outputs = context
            .bind(rematerialize_operation(), vec![sine_product_body()], &[Array::scalar(0.5f64).unwrap()])
            .unwrap();
        assert_eq!(outputs, vec![Array::scalar(0.5f64.sin() * 0.5).unwrap()]);
    }

    #[test]
    fn test_rematerialize_operation_batching() {
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

        // A mapped reference input enters the batched body as a reference to the stacked values, and the body may
        // forward it as an output. The threaded extent becomes a leading operand of the batched call.
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
        let outputs = builder
            .add_instruction(rematerialize_operation().lift::<ArrayIrType>(), vec![body], vec![reference, x], None)
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
                let %3:ref<f32[3]>, %4:f32[3] = rematerialize %0 %1 %2 [
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
    fn test_rematerialize_operation_differentiation() {
        // TODO(eaplatanios): Phase B2 of `.tasks/plan_rematerialization_redesign.md` replaces this rejection.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let body = builder.import_program(sine_product_body());
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(rematerialize_operation(), vec![body], vec![x], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(program.jvp().unwrap_err().to_string(), "operation `rematerialize` is not differentiable yet");
    }
}
