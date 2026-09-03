//! Generic ordered additive reference update operation and its value-level capability.

// TODO(eaplatanios): Review this module.

use std::borrow::Cow;
use std::sync::LazyLock;

use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    TransposableOperation, TranspositionContext, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::AddOperation;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::ProgramError;
use crate::programs::atoms::MaybeZero;
use crate::programs::effects::{Effect, Effects};
use crate::programs::operations::Operation;
use crate::programs::references::discharge::{
    ReferenceAccumulationPolicy, ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargeValue,
    ReferenceDischargeableOperation,
};
use crate::programs::references::operations::ReferenceRead;
use crate::programs::references::semantics::{ReferenceAccessMode, ReferenceInput, ReferenceOperationSemantics};
use crate::programs::references::types::ReferenceType;
use crate::programs::references::views::ReferenceViewOperation;
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::Value;
use crate::tracing::{Tracer, TracingContext};

use super::{align_stored_batch, stored_tangents, validate_operand_types};

/// Canonical operation name for [`ReferenceAddUpdateOperation`].
pub const REFERENCE_ADD_UPDATE_OPERATION_NAME: &str = "reference_add_update";

/// Adds an update into the value stored by a reference in program order.
pub trait ReferenceAddUpdate<Update = Self>: Sized {
    /// Adds `update` to the stored value in program order.
    fn add_update(&self, update: &Update) -> Result<(), ProgramError>;
}

static REFERENCE_ADD_UPDATE_OPERATION_SEMANTICS: LazyLock<ReferenceOperationSemantics> = LazyLock::new(|| {
    ReferenceOperationSemantics::new(vec![ReferenceInput::new(0, ReferenceAccessMode::Accumulate)], Vec::new())
});

define_reference_primitive_payload!(
    /// Applies an ordered additive update whose result must retain the reference's exact referent type.
    ReferenceAddUpdateOperation
);

impl_reference_primitive_display!(ReferenceAddUpdateOperation, REFERENCE_ADD_UPDATE_OPERATION_NAME);

impl<T, U> Operation for ReferenceAddUpdateOperation<T, U>
where
    T: Type,
    U: Type,
    for<'t> &'t T: TryFrom<&'t U, Error = TypeError>,
    for<'t> &'t ReferenceType<T>: TryFrom<&'t U, Error = TypeError>,
    AddOperation<T>: Operation<Type = T>,
{
    type Type = U;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_ADD_UPDATE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[U],
        region_interfaces: &[RegionInterface<U>],
    ) -> Result<Vec<U>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let reference = <&ReferenceType<T>>::try_from(&input_types[0])?;
        let update = <&T>::try_from(&input_types[1])?;
        let addition_results =
            AddOperation::<T>::new().infer_output_types(&[reference.referent().clone(), update.clone()], &[])?;
        check_count!("output", addition_results, 1, TypeError);
        let addition_result = &addition_results[0];
        if addition_result != reference.referent() {
            return Err(TypeError::invalid(format!(
                "`{REFERENCE_ADD_UPDATE_OPERATION_NAME}` addition result type `{addition_result}` must exactly match \
                 reference referent type `{}`",
                reference.referent(),
            )));
        }
        Ok(Vec::new())
    }

    #[inline]
    fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
        Cow::Borrowed(&REFERENCE_ADD_UPDATE_OPERATION_SEMANTICS)
    }

    #[inline]
    fn effects(&self) -> Effects {
        Effects::single(Effect::OrderedState)
    }
}

impl<T, U, C> InterpretableOperation<C> for ReferenceAddUpdateOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceAddUpdateOperation<T, U>: Operation<Type = U>,
    C: Domain<Type = U, Value: ReferenceAddUpdate<C::Value>>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        inputs[0].add_update(&inputs[1])?;
        Ok(Vec::new())
    }
}

impl<T, U, C, P> ReferenceDischargeableOperation<C, P> for ReferenceAddUpdateOperation<T, U>
where
    T: Type,
    U: From<T> + From<ReferenceType<T>> + Type,
    ReferenceAddUpdateOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceAddUpdateOperation<T, U>>>,
    P: ReferenceAccumulationPolicy<C, Referent = T>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        _driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        let reference = inputs[0].try_as_reference("a reference to accumulate into")?;
        let update = inputs[1].try_as_value("an update value")?.clone();

        // The sum of the handle's referent and the update must itself be the handle's referent, which is exactly what
        // this operation's own inference states and what a universe's addition alone does not guarantee.
        validate_operand_types(self, inputs)?;
        context.accumulate(reference, update)?;
        Ok(Vec::new())
    }
}

impl<T, U, C> PartiallyEvaluatableOperation<C> for ReferenceAddUpdateOperation<T, U>
where
    T: Type,
    U: Type,
    C: Context<Type = U, Operation: From<ReferenceAddUpdateOperation<T, U>>>,
{
    // The default partial-evaluation behavior applies: the primitive's ordered-state effect is placed centrally
    // before any operation rule runs.
}

impl<T, U, C> DifferentiableOperation<C> for ReferenceAddUpdateOperation<T, U>
where
    T: Type,
    U: DifferentiableType,
    ReferenceAddUpdateOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Value: ReferenceAddUpdate<C::Value>>,
{
    // Addition is linear, so the update's tangent is accumulated into the tangent reference exactly as the primal
    // update is accumulated into the primal reference. Accumulating a symbolic zero tangent is a no-op and stages
    // nothing. The tangent pairing is resolved before either accumulation so that a rejected plumbing store leaves both
    // references untouched.
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        let stored = stored_tangents(REFERENCE_ADD_UPDATE_OPERATION_NAME, &inputs[0], &inputs[1])?;
        inputs[0].primal().add_update(inputs[1].primal())?;
        if let Some((tangent_reference, MaybeZero::Value(tangent))) = stored {
            tangent_reference.add_update(&tangent)?;
        }
        Ok(Vec::new())
    }
}

impl<T, U, C, P> BatchableOperation<C, P> for ReferenceAddUpdateOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceAddUpdateOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Value: ReferenceAddUpdate<C::Value>>,
    P: BatchingPolicy<C>,
{
    // The update is aligned with the reference's fixed batch axis before the packed accumulation.
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        check_count!("input", inputs, 2, ProgramError);
        let update =
            align_stored_batch(context, driver, REFERENCE_ADD_UPDATE_OPERATION_NAME, &inputs[0], inputs[1].clone())?;
        P::value(&inputs[0]).add_update(P::value(&update))?;
        Ok(Vec::new().into())
    }
}

impl<T, U, V, O> TransposableOperation<V, O> for ReferenceAddUpdateOperation<T, U>
where
    T: Type,
    U: DifferentiableType,
    ReferenceAddUpdateOperation<T, U>: Operation<Type = U>,
    V: Value<Type = U>,
    O: ReferenceViewOperation<Type = U>,
    Tracer<TracingContext<V, O>>: ReferenceRead<Tracer<TracingContext<V, O>>>,
{
    // An accumulation maps `(state, x) ↦ state + x`, so its transpose reads the cotangent reference as the cotangent of
    // the update and leaves the reference's contents unchanged for the earlier accesses. An accumulator that nothing
    // has reached yet holds zero, so nothing is staged and the update's cotangent stays symbolic.
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<'_, V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        check_count!("output", outputs, 0, ProgramError);
        let reference_cotangent = MaybeZero::Zero(inputs[0].r#type().cotangent()?);
        let Some(accumulator) = context.cotangent_reference_if_allocated(0)? else {
            return Ok(vec![reference_cotangent, MaybeZero::Zero(inputs[1].r#type().cotangent()?)]);
        };
        Ok(vec![reference_cotangent, MaybeZero::Value(accumulator.read()?)])
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatching, ArrayIrOperation, ArrayIrValue, ArrayType, DataType, DimensionBounds,
        DimensionType, DimensionValue, DimensionVariable,
    };
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
    use crate::contexts::EagerContext;
    use crate::differentiation::{DifferentiationContext, DifferentiationDual, DifferentiationTracer};
    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::references::operations::tests::*;
    use crate::programs::references::operations::{ReferenceNew, ReferenceRead};
    use crate::programs::regions::EmptyRegionDriver;

    use super::*;

    #[test]
    fn test_reference_add_update_operation() {
        let referent = TestReferent::new(7, 16);
        let promoted_refinement = TestReferent::new(7, 32);
        let value = TestType::Value(referent);
        let reference = TestType::Reference(ReferenceType::new(referent));

        assert_parameter_roundtrip(AddUpdate::new());
        assert_eq!(AddUpdate::new().to_string(), REFERENCE_ADD_UPDATE_OPERATION_NAME);
        assert_eq!(AddUpdate::new().effects(), Effects::single(Effect::OrderedState));
        assert_eq!(AddUpdate::new().reference_semantics().outputs(), &[]);
        assert_eq!(
            AddUpdate::new().reference_semantics().inputs(),
            &[ReferenceInput::new(0, ReferenceAccessMode::Accumulate)],
        );

        assert_eq!(
            ReferenceAddUpdateOperation::<TestReferent, AddUpdateUniverse>::new().infer_output_types(
                &[AddUpdateUniverse::Reference(ReferenceType::new(referent)), AddUpdateUniverse::Value(referent)],
                &[],
            ),
            Ok(Vec::new()),
        );
        assert_eq!(AddUpdate::new().infer_output_types(&[reference.clone(), value.clone()], &[]), Ok(Vec::new()));
        assert_eq!(
            AddUpdate::new().infer_output_types(&[reference.clone(), TestType::Value(promoted_refinement)], &[]),
            Err(TypeError::invalid(
                "`reference_add_update` addition result type `value<i7,p32>` must exactly match reference referent \
                 type `value<i7,p16>`",
            )),
        );
        assert_eq!(
            AddUpdate::new().infer_output_types(std::slice::from_ref(&reference), &[]),
            Err(TypeError::invalid("expected 2 inputs but got 1")),
        );
        assert_eq!(
            AddUpdate::new().infer_output_types(&[value.clone(), value.clone()], &[]),
            Err(TypeError::invalid("expected reference type but got value type")),
        );
        assert_eq!(
            AddUpdate::new().infer_output_types(&[reference.clone(), reference.clone()], &[]),
            Err(TypeError::invalid("expected value type but got reference type")),
        );
        let region = RegionInterface::new(Vec::new(), Vec::new(), Effects::PURE);
        assert_eq!(
            AddUpdate::new().infer_output_types(&[reference, value], std::slice::from_ref(&region)),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_reference_add_update_operation_reference_discharge() {
        // An accumulation produces no result and replaces the current state with its sum with the update.
        let (context, reference) = allocated_allocation(4);
        let inputs = vec![
            ReferenceDischargeValue::Reference(reference.clone()),
            ReferenceDischargeValue::Value(TestValue::new(REFERENT, 9)),
        ];
        assert_eq!(
            AddUpdate::new().discharge_references(&context, &EmptyRegionDriver, inputs.as_slice()),
            Ok(Vec::new()),
        );
        assert_eq!(context.read(&reference), Ok(TestValue::new(REFERENT, 13)));
        assert_eq!(context.is_mutated(reference.allocation_id()), Ok(true));

        // An update whose sum with the referent would not itself be the referent is rejected by this operation's own
        // inference before the universe accumulates anything, so the allocation keeps its previous state.
        let promoted = vec![
            ReferenceDischargeValue::Reference(reference.clone()),
            ReferenceDischargeValue::Value(TestValue::new(TestReferent::new(7, 32), 1)),
        ];
        assert_eq!(
            AddUpdate::new().discharge_references(&context, &EmptyRegionDriver, promoted.as_slice()),
            Err(TypeError::invalid(
                "`reference_add_update` addition result type `value<i7,p32>` must exactly match reference referent \
             type `value<i7,p16>`",
            )
            .into()),
        );
        assert_eq!(context.read(&reference), Ok(TestValue::new(REFERENT, 13)));
    }

    #[test]
    fn test_reference_add_update_operation_jvp() {
        type TestValue = ArrayIrValue<Array>;

        let context = DifferentiationContext::new(EagerContext::<TestValue, ArrayIrOperation<Array>>::new());
        let reference = TestValue::Array(Array::vector(vec![1.0_f32, 2.0])).reference_new().unwrap();
        let tangent_reference = TestValue::Array(Array::vector(vec![3.0_f32, 4.0])).reference_new().unwrap();
        let active = DifferentiationTracer::new(
            DifferentiationDual::new(reference.clone(), tangent_reference.clone()).unwrap(),
            context.clone(),
        );

        // Addition is linear, so the update's tangent accumulates into the tangent reference.
        let update = DifferentiationTracer::new(
            DifferentiationDual::new(
                TestValue::Array(Array::vector(vec![5.0_f32, 6.0])),
                TestValue::Array(Array::vector(vec![7.0_f32, 8.0])),
            )
            .unwrap(),
            context.clone(),
        );
        let outputs = context.bind(ReferenceAddUpdateOperation::new(), Vec::new(), &[active.clone(), update]).unwrap();
        assert!(outputs.is_empty());
        assert_eq!(reference.read(), Ok(TestValue::Array(Array::vector(vec![6.0_f32, 8.0]))));
        assert_eq!(tangent_reference.read(), Ok(TestValue::Array(Array::vector(vec![10.0_f32, 12.0]))));

        // A symbolic zero update tangent accumulates nothing and is not instantiated.
        let update = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(TestValue::Array(Array::vector(vec![1.0_f32, 1.0]))).unwrap(),
            context.clone(),
        );
        context.bind(ReferenceAddUpdateOperation::new(), Vec::new(), &[active, update]).unwrap();
        assert_eq!(reference.read(), Ok(TestValue::Array(Array::vector(vec![7.0_f32, 9.0]))));
        assert_eq!(tangent_reference.read(), Ok(TestValue::Array(Array::vector(vec![10.0_f32, 12.0]))));

        // Staged, the zero-tangent accumulation is therefore elided from the tangent side entirely: the fused program
        // accumulates the constant into the primal reference and leaves the tangent reference untouched.
        let mut builder = ProgramBuilder::<TestValue, ArrayIrOperation<Array>>::new();
        let reference_atom = builder.add_input(ReferenceType::new(ArrayType::scalar(DataType::F32)).into());
        let constant = builder.add_constant(TestValue::Array(Array::scalar(1.0_f32)));
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference_atom, constant], None)
            .unwrap();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference_atom], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:ref<f32[]>, %1:ref<f32[]> .
                let %2:f32[] = const 1.0
                    reference_add_update %0 %2
                in (%0, %1)
            "}
            .trim_end(),
        );

        // A plumbing reference accepts an update without a live tangent, while a live update tangent has no tangent
        // reference to accumulate into and the rejection precedes the primal accumulation.
        let plumbing = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(reference.clone()).unwrap(),
            context.clone(),
        );
        let update = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(TestValue::Array(Array::vector(vec![1.0_f32, 1.0]))).unwrap(),
            context.clone(),
        );
        context.bind(ReferenceAddUpdateOperation::new(), Vec::new(), &[plumbing.clone(), update]).unwrap();
        assert_eq!(reference.read(), Ok(TestValue::Array(Array::vector(vec![8.0_f32, 10.0]))));
        let update = DifferentiationTracer::new(
            DifferentiationDual::new(
                TestValue::Array(Array::vector(vec![1.0_f32, 1.0])),
                TestValue::Array(Array::vector(vec![1.0_f32, 1.0])),
            )
            .unwrap(),
            context.clone(),
        );
        let error = context.bind(ReferenceAddUpdateOperation::new(), Vec::new(), &[plumbing, update]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<DifferentiationError>(),
            Some(&DifferentiationError::PlumbingReferenceTangent { operation: REFERENCE_ADD_UPDATE_OPERATION_NAME }),
        );
        assert_eq!(reference.read(), Ok(TestValue::Array(Array::vector(vec![8.0_f32, 10.0]))));
        assert_eq!(tangent_reference.read(), Ok(TestValue::Array(Array::vector(vec![10.0_f32, 12.0]))));
    }

    #[test]
    fn test_reference_add_update_operation_batching() {
        type TestValue = ArrayIrValue<Array>;

        let extent = TestValue::Dimension(
            DimensionValue::new(DimensionType::new(DimensionVariable::new("batch", DimensionBounds::unbounded())), 2)
                .unwrap(),
        );
        let context = BatchingContext::<_, ArrayIrBatching>::new(
            EagerContext::<TestValue, ArrayIrOperation<Array>>::new(),
            extent,
        );
        let packed_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let initial = TestValue::Array(Array::from_f64s(packed_type.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let reference = initial.reference_new().unwrap();
        let batched =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(reference.clone(), BatchAxis::new(0)).unwrap());

        // An update mapped at the reference's batch axis accumulates packed.
        let update = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::new(TestValue::Array(Array::from_f64s(packed_type.clone(), vec![1.0; 6])), BatchAxis::new(0))
                .unwrap(),
        );
        let outputs = context.bind(ReferenceAddUpdateOperation::new(), Vec::new(), &[batched, update]).unwrap();
        assert!(outputs.is_empty());
        assert_eq!(
            reference.read(),
            Ok(TestValue::Array(Array::from_f64s(packed_type.clone(), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))),
        );

        // A batched update cannot accumulate into an unbatched reference.
        let replicated = BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(reference.clone()));
        let update = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::new(TestValue::Array(Array::from_f64s(packed_type, vec![1.0; 6])), BatchAxis::new(0))
                .unwrap(),
        );
        let error = context.bind(ReferenceAddUpdateOperation::new(), Vec::new(), &[replicated, update]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<BatchingError>(),
            Some(&BatchingError::UnsupportedOperation {
                message: "`reference_add_update` cannot store a batched value into an unbatched reference; pass the \
                          reference as a batched input instead"
                    .to_string(),
            }),
        );
    }
}
