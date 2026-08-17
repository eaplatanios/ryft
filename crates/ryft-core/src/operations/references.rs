use std::borrow::Cow;
use std::fmt::Display;
use std::sync::LazyLock;

use ryft_macros::Parameter;

use crate::arrays::{ArrayIrType, ArrayType};
use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiationDriver, DifferentiationDual, DifferentiationError,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_transposable_operation};
use crate::parameters::Parameter;
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue, PartiallyEvaluatableOperation,
};
use crate::programs::{
    Effect, Effects, Operation, ProgramError, ReferenceAccessMode, ReferenceInputAccess, ReferenceOperationSemantics,
    ReferenceOutputSemantics, ReferenceType, RegionInterface, TypeError, Value,
};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`NewReferenceOperation`].
pub const NEW_REFERENCE_OPERATION_NAME: &str = "new_reference";

/// Canonical operation name for [`ReferenceReadOperation`].
pub const REFERENCE_READ_OPERATION_NAME: &str = "reference_read";

/// Creates a new reference initialized from this value.
pub trait NewReference<Output = Self>: Sized {
    /// Creates an independent reference whose initial state is this value.
    fn new_reference(&self) -> Result<Output, ProgramError>;
}

impl<V: Value<Type = ArrayIrType>> NewReference<V> for V
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<NewReferenceOperation>,
{
    fn new_reference(&self) -> Result<V, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(NewReferenceOperation, Vec::new(), std::slice::from_ref(self))?
            .remove(0))
    }
}

/// Reads an immutable snapshot from a reference value.
pub trait ReferenceRead<Output = Self>: Sized {
    /// Returns the reference's current value as an immutable snapshot.
    fn read(&self) -> Result<Output, ProgramError>;
}

impl<V: Value<Type = ArrayIrType>> ReferenceRead<V> for V
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceReadOperation>,
{
    fn read(&self) -> Result<V, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(ReferenceReadOperation, Vec::new(), std::slice::from_ref(self))?
            .remove(0))
    }
}

// Reference semantics descriptors are constant per operation type. Sharing them through `LazyLock` statics lets the
// per-instruction program analysis read them through `Cow::Borrowed` without allocating.
static NEW_REFERENCE_OPERATION_SEMANTICS: LazyLock<ReferenceOperationSemantics> = LazyLock::new(|| {
    ReferenceOperationSemantics::new(vec![ReferenceOutputSemantics::NewRoot { output_index: 0 }], Vec::new())
});

static REFERENCE_READ_OPERATION_SEMANTICS: LazyLock<ReferenceOperationSemantics> = LazyLock::new(|| {
    ReferenceOperationSemantics::new(Vec::new(), vec![ReferenceInputAccess::new(0, ReferenceAccessMode::Read)])
});

/// Composite array-to-reference allocation operation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct NewReferenceOperation;

impl Display for NewReferenceOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for NewReferenceOperation {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        NEW_REFERENCE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let referent = <&ArrayType>::try_from(&input_types[0])?;
        Ok(vec![ReferenceType::new(referent.clone()).into()])
    }

    #[inline]
    fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
        Cow::Borrowed(&NEW_REFERENCE_OPERATION_SEMANTICS)
    }

    #[inline]
    fn effects(&self) -> Effects {
        Effects::single(Effect::OrderedState)
    }
}

impl<C: Domain<Type = ArrayIrType, Value: NewReference<C::Value>>> InterpretableOperation<C> for NewReferenceOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        // Member-kind checking is owned by the value-level `NewReference` implementation, whose projection produces
        // the same diagnostic type inference would, so no inference re-run is needed here.
        Ok(vec![inputs[0].new_reference()?])
    }
}

/// Composite reference-to-array snapshot read operation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct ReferenceReadOperation;

impl Display for ReferenceReadOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ReferenceReadOperation {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_READ_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let reference = <&ReferenceType<ArrayType>>::try_from(&input_types[0])?;
        Ok(vec![reference.referent().clone().into()])
    }

    #[inline]
    fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
        Cow::Borrowed(&REFERENCE_READ_OPERATION_SEMANTICS)
    }

    #[inline]
    fn effects(&self) -> Effects {
        Effects::single(Effect::OrderedState)
    }
}

impl<C: Domain<Type = ArrayIrType, Value: ReferenceRead<C::Value>>> InterpretableOperation<C>
    for ReferenceReadOperation
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        // Member-kind checking is owned by the value-level `ReferenceRead` implementation, whose projection produces
        // the same diagnostic type inference would, so no inference re-run is needed here.
        Ok(vec![inputs[0].read()?])
    }
}

macro_rules! impl_unsupported_reference_transforms {
    // Each invocation installs the same conservative transform rejection for one unresolved reference operation.
    ($operation:ty) => {
        impl<C: Context<Type = ArrayIrType, Operation: From<$operation>>> PartiallyEvaluatableOperation<C>
            for $operation
        {
            fn partially_evaluate<D: PartialEvaluationDriver<C>>(
                &self,
                _context: &PartialEvaluationContext<C>,
                _driver: &D,
                _inputs: &[PartialEvaluationValue<C::Value>],
            ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
                Err(ProgramError::UnsupportedOperation {
                    message: format!("`{}` must be discharged before partial evaluation", self.name()),
                })
            }
        }

        impl<C: Context<Type = ArrayIrType, Operation: From<$operation>>, P: BatchingPolicy<C>> BatchableOperation<C, P>
            for $operation
        {
            fn batch<D: BatchingDriver<C, P>>(
                &self,
                _context: &BatchingContext<C, P>,
                _driver: &D,
                _inputs: &[P::Batch],
            ) -> Result<BatchedOutputs<C, P>, BatchingError> {
                Err(BatchingError::UnsupportedOperation {
                    message: format!("`{}` must be discharged before batching", self.name()),
                })
            }
        }

        impl<C: Context<Type = ArrayIrType, Operation: From<$operation>>> DifferentiableOperation<C> for $operation {
            fn jvp<D: DifferentiationDriver<C>>(
                &self,
                _context: &C,
                _driver: &D,
                _inputs: &[DifferentiationDual<C::Value>],
            ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
                Err(ProgramError::UnsupportedOperation {
                    message: format!("`{}` must be discharged before differentiation", self.name()),
                }
                .into())
            }
        }
    };
}

impl_unsupported_reference_transforms!(NewReferenceOperation);
impl_unsupported_reference_transforms!(ReferenceReadOperation);

// Transposition rejection reuses the shared non-transposable diagnostic: reference operations never transpose
// directly because reverse-mode differentiation always discharges them first (refer to `plan-references.md`).
impl_non_transposable_operation!(NewReferenceOperation);
impl_non_transposable_operation!(ReferenceReadOperation);

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayIrBatching, ArrayIrOperation, ArrayIrValue, DataType};
    use crate::contexts::EagerContext;
    use crate::differentiation::{
        CustomJvpOperation, CustomVjpOperation, DifferentiationError, DifferentiationTracer, ForwardModeDifferentiate,
        Linearization, TransposableOperation,
    };
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{EmptyRegionDriver, Program, ProgramBuilder, Reference, RegionDriver, RegionRef};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_reference_operation_type_and_access_contracts() {
        let array_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(array_type.clone());
        assert_eq!(
            NewReferenceOperation.infer_output_types(std::slice::from_ref(&array_type.clone().into()), &[]),
            Ok(vec![reference_type.clone().into()]),
        );
        assert_eq!(
            ReferenceReadOperation.infer_output_types(std::slice::from_ref(&reference_type.into()), &[]),
            Ok(vec![array_type.into()]),
        );
        assert_eq!(NewReferenceOperation.effects(), Effects::single(Effect::OrderedState));
        assert_eq!(ReferenceReadOperation.effects(), Effects::single(Effect::OrderedState));
        assert_eq!(
            NewReferenceOperation.reference_semantics().outputs(),
            &[ReferenceOutputSemantics::NewRoot { output_index: 0 }],
        );
        assert_eq!(
            ReferenceReadOperation.reference_semantics().accesses(),
            &[ReferenceInputAccess::new(0, ReferenceAccessMode::Read)],
        );
    }

    #[test]
    fn test_reference_operations_execute_eagerly_and_stage_as_composite_native_variants() {
        let initial = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0]));
        let reference = initial.new_reference().unwrap();
        assert_eq!(ReferenceRead::read(&reference).unwrap(), initial);

        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let (output_type, program) = TestContext::trace(
            |input| ReferenceRead::read(&input.new_reference()?),
            ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let [allocation, read] = program.instructions() else {
            panic!("expected allocation followed by one read");
        };
        assert!(matches!(allocation.operation(), ArrayIrOperation::NewReference(_)));
        assert!(matches!(read.operation(), ArrayIrOperation::ReferenceRead(_)));
        assert_eq!(program.effects(), Effects::single(Effect::OrderedState));
    }

    #[test]
    fn test_reference_operations_reject_transforms_until_discharge() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let input_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let (_, program) =
            TestContext::trace(|input| ReferenceRead::read(&input.new_reference()?), input_type.clone()).unwrap();
        let program = program.into_flat_program();

        // Partial evaluation must never execute, fold, or split an unresolved state chain, regardless of whether the
        // reference-producing input is known or unknown.
        for input in [
            PartialValue::Unknown(input_type.clone()),
            PartialValue::Known(ArrayIrValue::Array(Array::scalar(1.0_f32))),
        ] {
            assert_eq!(
                program.partially_evaluate(std::slice::from_ref(&input)).map(|_| ()),
                Err(ProgramError::UnsupportedOperation {
                    message: "`new_reference` must be discharged before partial evaluation".to_string(),
                }),
            );
        }

        // Structural differentiation (direct `Program::jvp` and the linearization built on it) rejects at the fused
        // replay's entry, before its all-zero shortcut could stage any primal instruction.
        for result in [program.jvp().map(|_| ()), program.linearize().map(|_| ())] {
            assert!(matches!(
                result,
                Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                    if message
                        == "program carries unresolved state and must be discharged before differentiation",
            ));
        }

        // Transposition reuses the shared non-transposable diagnostic because reference operations never transpose
        // directly (reverse mode discharges them first).
        let mut transposition_context = TestContext::new();
        assert!(matches!(
            <NewReferenceOperation as TransposableOperation<ArrayIrValue<Array>, ArrayIrOperation<Array>>>::transpose(
                &NewReferenceOperation,
                &mut transposition_context,
                &EmptyRegionDriver,
                &[],
                &[],
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `new_reference` is not transposable",
        ));

        // Batching rejects before inspecting any batch inputs.
        let batching_context = BatchingContext::<_, ArrayIrBatching>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Array(Array::scalar(2_i64)),
        );
        assert!(matches!(
            <NewReferenceOperation as BatchableOperation<_, ArrayIrBatching>>::batch(
                &NewReferenceOperation,
                &batching_context,
                &EmptyRegionDriver,
                &[],
            ),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "`new_reference` must be discharged before batching",
        ));

        // A lifted reference must not ride through batching as a replicated batch that could cross the output
        // boundary unchanged, so `BatchingContext::lift` routes through the policy's checked batch constructor.
        let error = batching_context
            .lift(ArrayIrValue::Reference(Reference::new(Array::scalar(1.0_f32))))
            .map(|_| ())
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<BatchingError>(),
            Some(&BatchingError::UnsupportedOperation {
                message: "references must be discharged before batching".to_string(),
            }),
        );
    }

    #[test]
    fn test_custom_derivative_rules_reject_unresolved_state() {
        // Custom derivative rule regions are interpreted directly rather than routed through the per-operation
        // transform rejections, so a rule whose body touches unresolved state must be rejected before any
        // of it executes during differentiation.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let primal = {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let input = builder.add_input(scalar_type.clone());
            builder.build::<Vec<_>, Vec<_>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let rule = {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let input = builder.add_input(scalar_type.clone());
            let input_tangent = builder.add_input(scalar_type.clone());
            let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![input]).unwrap()[0];
            let output = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
            builder
                .build::<Vec<_>, Vec<_>>(vec![output, input_tangent], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        // The central `DifferentiationContext::bind` guard rejects the state before the rule is ever consulted (the
        // operation-local rule-region guards remain as defense in depth behind it).
        let result = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new().jvp(
            {
                let primal = primal.clone();
                let rule = rule.clone();
                move |input: DifferentiationTracer<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>, ()| {
                    let operation = ArrayIrOperation::CustomJvp(CustomJvpOperation::new());
                    let regions = vec![primal.clone(), rule.clone()];
                    Ok(input.context().bind(operation, regions, std::slice::from_ref(&input))?.remove(0))
                }
            },
            ArrayIrValue::Array(Array::scalar(1.0_f32)),
            ArrayIrValue::Array(Array::scalar(1.0_f32)),
            (),
        );
        assert!(matches!(
            result,
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`custom_jvp` carries unresolved state in an attached region and must be \
                    discharged before differentiation",
        ));

        // The all-zero tangent fast path binds the primal directly without reaching any operation rule, so state must
        // be rejected on that path too: a lifted constant input carries a structural-zero tangent.
        let result = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new().jvp(
            move |input: DifferentiationTracer<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>, ()| {
                let lifted = input.context().lift(ArrayIrValue::Array(Array::scalar(1.0_f32)))?;
                let operation = ArrayIrOperation::CustomJvp(CustomJvpOperation::new());
                let regions = vec![primal.clone(), rule.clone()];
                Ok(input.context().bind(operation, regions, std::slice::from_ref(&lifted))?.remove(0))
            },
            ArrayIrValue::Array(Array::scalar(1.0_f32)),
            ArrayIrValue::Array(Array::scalar(1.0_f32)),
            (),
        );
        assert!(matches!(
            result,
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`custom_jvp` carries unresolved state in an attached region and must be \
                    discharged before differentiation",
        ));
    }

    #[test]
    fn test_operation_local_custom_derivative_guards_reject_state_in_nested_dormant_rules() {
        type TestValue = ArrayIrValue<Array>;
        type TestOperation = ArrayIrOperation<Array>;
        type TestContext = EagerContext<TestValue, TestOperation>;
        type TestProgram = Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>;

        struct TestDifferentiationDriver {
            programs: Vec<TestProgram>,
        }

        impl RegionDriver<TestValue, TestOperation> for TestDifferentiationDriver {
            fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, TestValue, TestOperation>>
            where
                TestValue: 'r,
                TestOperation: 'r,
            {
                self.programs.iter().map(Program::entry_region_ref)
            }
        }

        impl DifferentiationDriver<TestContext> for TestDifferentiationDriver {
            fn jvp_program(
                &self,
                _region: RegionRef<'_, TestValue, TestOperation>,
            ) -> Result<Arc<TestProgram>, DifferentiationError> {
                unreachable!("the operation-local state guard must reject before recursive differentiation")
            }

            fn linearize_program(
                &self,
                _region: RegionRef<'_, TestValue, TestOperation>,
            ) -> Result<Linearization<TestValue, TestOperation>, DifferentiationError> {
                unreachable!("the operation-local state guard must reject before recursive linearization")
            }

            fn jvp_operation(
                &self,
                _operation: &TestOperation,
                _programs: Vec<TestProgram>,
                _context: &TestContext,
                _inputs: &[DifferentiationDual<TestValue>],
            ) -> Result<Vec<DifferentiationDual<TestValue>>, DifferentiationError> {
                unreachable!("the operation-local state guard must reject before recursive differentiation")
            }
        }

        fn nested_custom_jvp_regions(scalar_type: &ArrayIrType) -> Vec<TestProgram> {
            let primal = {
                let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
                let input = builder.add_input(scalar_type.clone());
                builder
                    .build::<Vec<TestValue>, Vec<TestValue>>(vec![input], vec![Placeholder], vec![Placeholder])
                    .unwrap()
            };
            let rule = {
                let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
                let input = builder.add_input(scalar_type.clone());
                let tangent = builder.add_input(scalar_type.clone());
                let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![input]).unwrap()[0];
                let output = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
                builder
                    .build::<Vec<TestValue>, Vec<TestValue>>(
                        vec![output, tangent],
                        vec![Placeholder; 2],
                        vec![Placeholder; 2],
                    )
                    .unwrap()
            };
            vec![primal, rule]
        }

        fn nested_rule_state_program(scalar_type: &ArrayIrType, include_tangent_output: bool) -> TestProgram {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let regions = nested_custom_jvp_regions(scalar_type)
                .iter()
                .map(|region| builder.import_region(region.entry_region_ref()))
                .collect::<Vec<_>>();
            let input = builder.add_input(scalar_type.clone());
            let tangent = include_tangent_output.then(|| builder.add_input(scalar_type.clone()));
            let output =
                builder.add_instruction(CustomJvpOperation::<ArrayIrType>::new(), regions, vec![input]).unwrap()[0];
            let mut outputs = vec![output];
            if let Some(tangent) = tangent {
                outputs.push(tangent);
            }
            let input_count = usize::from(include_tangent_output) + 1;
            let output_count = outputs.len();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(
                    outputs,
                    vec![Placeholder; input_count],
                    vec![Placeholder; output_count],
                )
                .unwrap()
        }

        fn identity_program(scalar_type: &ArrayIrType) -> TestProgram {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let input = builder.add_input(scalar_type.clone());
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![input], vec![Placeholder], vec![Placeholder])
                .unwrap()
        }

        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let context = TestContext::new();
        let input = DifferentiationDual::new(
            TestValue::Array(Array::scalar(1.0_f32)),
            TestValue::Array(Array::scalar(1.0_f32)),
        )
        .unwrap();

        // The outer JVP rule is pure under ordinary effect aggregation because its nested custom-JVP call keeps the
        // state in a dormant rule. Direct operation-rule invocation must nevertheless inspect the complete closure.
        let primal = identity_program(&scalar_type);
        let jvp = nested_rule_state_program(&scalar_type, true);
        assert!(jvp.effects().is_pure());
        assert!(jvp.entry_region_ref().contains_effect_in_closure(Effect::OrderedState));
        let driver = TestDifferentiationDriver { programs: vec![primal, jvp] };
        assert!(matches!(
            CustomJvpOperation::<ArrayIrType>::new().jvp(&context, &driver, std::slice::from_ref(&input)),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`custom_jvp` rule regions must not contain unresolved state",
        ));

        // Custom VJP retains both its directly interpreted forward rule and its transpose-time backward rule in the
        // differentiated program. Exercise each position independently so neither can hide state in a dormant child.
        let primal = identity_program(&scalar_type);
        let forward = nested_rule_state_program(&scalar_type, false);
        let backward = identity_program(&scalar_type);
        assert!(forward.effects().is_pure());
        let driver = TestDifferentiationDriver { programs: vec![primal, forward, backward] };
        assert!(matches!(
            CustomVjpOperation::<ArrayIrType>::new().jvp(&context, &driver, std::slice::from_ref(&input)),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`custom_vjp` rule regions must not contain unresolved state",
        ));

        let primal = identity_program(&scalar_type);
        let forward = identity_program(&scalar_type);
        let backward = nested_rule_state_program(&scalar_type, false);
        assert!(backward.effects().is_pure());
        let driver = TestDifferentiationDriver { programs: vec![primal, forward, backward] };
        assert!(matches!(
            CustomVjpOperation::<ArrayIrType>::new().jvp(&context, &driver, std::slice::from_ref(&input)),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`custom_vjp` rule regions must not contain unresolved state",
        ));
    }

    #[test]
    fn test_program_jvp_rejects_reference_state_hidden_in_dormant_rule_regions() {
        // Sealed program effects deliberately exclude dormant rule regions, so a program whose only unresolved state
        // lives inside a custom-derivative rule is `Effects::PURE`. The fused JVP entry must therefore scan the whole
        // attached-region closure rather than trusting program effects.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let primal = {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let input = builder.add_input(scalar_type.clone());
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![input],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let rule = {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let input = builder.add_input(scalar_type.clone());
            let input_tangent = builder.add_input(scalar_type.clone());
            let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![input]).unwrap()[0];
            let output = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![output, input_tangent],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let wrapped = {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let primal_id = builder.import_region(primal.entry_region_ref());
            let rule_id = builder.import_region(rule.entry_region_ref());
            let input = builder.add_input(scalar_type.clone());
            let outputs = builder
                .add_instruction(
                    ArrayIrOperation::CustomJvp(CustomJvpOperation::new()),
                    vec![primal_id, rule_id],
                    vec![input],
                )
                .unwrap()
                .to_vec();
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    outputs,
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };
        assert!(wrapped.effects().is_pure());
        assert!(matches!(
            wrapped.jvp(),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message
                    == "program carries unresolved state and must be discharged before differentiation",
        ));
    }
}
