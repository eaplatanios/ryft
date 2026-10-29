use std::borrow::Cow;
use std::fmt::{Display, Formatter};

use indoc::indoc;
use pretty_assertions::assert_eq;

use ryft_core::{
    Add, Array, ArrayMemberType, ArrayOperation, ArrayType, Broadcast, ConditionOperation, Context, DataType,
    DifferentiableOperation, DifferentiableType, DifferentiationError, Domain, GuardOperation, NoIdentity, One,
    Operation, OperationProjection, Parameter, ProgramError, ProjectedContext, RegionInterface, RegionSlot,
    ResidualStorageOperation, ResidualZeroProvider, ScanOperation, Tracer, TracingContext, Type, TypeError, Typed,
    Value, ValueDirectDispatch, ValueProjection, WhileOperation, WhileType, ZeroOperation,
};

#[test]
fn test_external_while_type_supports_shared_differentiation() {
    // These local types expose their array members through the public projection contracts. The external crate
    // cannot implement foreign array capabilities directly on Tracer<TracingContext<ExternalValue, ExternalOperation>>.
    #[derive(Clone, Debug, PartialEq)]
    struct ExternalType(ArrayType);

    impl Display for ExternalType {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(formatter)
        }
    }

    impl Parameter for ExternalType {}

    impl Type for ExternalType {
        type Identity = NoIdentity;
        type Refinements = ();

        fn is_compatible_with(&self, other: &Self) -> bool {
            self.0.is_compatible_with(&other.0)
        }

        fn is_refined_by(&self, other: &Self) -> bool {
            self.0.is_refined_by(&other.0)
        }

        fn is_scalar(&self) -> bool {
            self.0.is_scalar()
        }

        fn is_complex(&self) -> bool {
            self.0.is_complex()
        }
    }

    impl From<ArrayType> for ExternalType {
        fn from(r#type: ArrayType) -> Self {
            Self(r#type)
        }
    }

    impl<'t> TryFrom<&'t ExternalType> for &'t ArrayType {
        type Error = TypeError;

        fn try_from(r#type: &'t ExternalType) -> Result<Self, TypeError> {
            Ok(&r#type.0)
        }
    }

    impl ArrayMemberType for ExternalType {
        fn as_array_type(&self) -> Result<&ArrayType, TypeError> {
            Ok(&self.0)
        }
    }

    impl WhileType for ExternalType {
        fn validate_while_predicate(&self, state_types: &[Self]) -> Result<(), TypeError> {
            self.0
                .validate_while_predicate(&state_types.iter().map(|r#type| r#type.0.clone()).collect::<Vec<_>>())
        }

        fn is_batched_while_predicate(&self) -> bool {
            self.0.is_batched_while_predicate()
        }

        fn is_loop_invariant_state(&self) -> bool {
            false
        }
    }

    impl DifferentiableType for ExternalType {
        fn is_zero_space(&self) -> bool {
            self.0.is_zero_space()
        }

        fn tangent(&self) -> Result<Self, DifferentiationError> {
            self.0.tangent().map(Self)
        }

        fn cotangent(&self) -> Result<Self, DifferentiationError> {
            self.0.cotangent().map(Self)
        }
    }

    #[derive(Clone, Debug)]
    struct ExternalValue(Array);

    impl Display for ExternalValue {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(formatter)
        }
    }

    impl Parameter for ExternalValue {}

    impl Typed for ExternalValue {
        type Type = ExternalType;

        fn r#type(&self) -> Cow<'_, Self::Type> {
            Cow::Owned(ExternalType(self.0.r#type().into_owned()))
        }
    }

    struct ExternalDomain;

    impl Domain for ExternalDomain {
        type Type = ExternalType;
        type Value = ExternalValue;
        type Constant = ExternalValue;
        type Operation = ExternalOperation;
    }

    impl Value for ExternalValue {
        type Dispatch = ValueDirectDispatch;
        type Domain = ExternalDomain;

        fn domain(&self) -> Self::Domain {
            ExternalDomain
        }
    }

    impl ValueProjection<ArrayType> for ExternalValue {
        type Projected = Array;
        type ProjectedRef<'v> = &'v Array;

        fn from_projected(value: Array) -> Self {
            Self(value)
        }

        fn projected<'v>(&'v self) -> Result<Self::ProjectedRef<'v>, TypeError>
        where
            ArrayType: 'v,
        {
            Ok(&self.0)
        }

        fn into_projected(self) -> Result<Array, TypeError> {
            Ok(self.0)
        }
    }

    #[derive(Clone)]
    enum ExternalOperation {
        Array(ArrayOperation<Array>),
        While(WhileOperation<ExternalType>),
        Guard(GuardOperation<ExternalType>),
        Zero(ZeroOperation<ExternalType>),
    }

    impl Operation for ExternalOperation {
        type Type = ExternalType;

        fn name(&self) -> &'static str {
            match self {
                Self::Array(operation) => operation.name(),
                Self::While(operation) => operation.name(),
                Self::Guard(operation) => operation.name(),
                Self::Zero(operation) => operation.name(),
            }
        }

        fn region_slots(&self) -> &'static [RegionSlot] {
            match self {
                Self::Array(operation) => operation.region_slots(),
                Self::While(operation) => operation.region_slots(),
                Self::Guard(operation) => operation.region_slots(),
                Self::Zero(operation) => operation.region_slots(),
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[ExternalType],
            region_interfaces: &[RegionInterface<ExternalType>],
        ) -> Result<Vec<ExternalType>, TypeError> {
            match self {
                Self::Array(operation) => {
                    let input_types = input_types.iter().map(|r#type| r#type.0.clone()).collect::<Vec<_>>();
                    let region_interfaces = region_interfaces
                        .iter()
                        .map(|region| {
                            RegionInterface::new(
                                region.input_types().iter().map(|r#type| r#type.0.clone()).collect(),
                                region.output_types().iter().map(|r#type| r#type.0.clone()).collect(),
                                region.effects(),
                            )
                            .with_deferred_work(region.has_deferred_work())
                        })
                        .collect::<Vec<_>>();
                    operation
                        .infer_output_types(&input_types, &region_interfaces)
                        .map(|types| types.into_iter().map(ExternalType).collect())
                }
                Self::While(operation) => operation.infer_output_types(input_types, region_interfaces),
                Self::Guard(operation) => operation.infer_output_types(input_types, region_interfaces),
                Self::Zero(operation) => operation.infer_output_types(input_types, region_interfaces),
            }
        }

        fn render(&self, formatter: &mut Formatter<'_>, indentation: usize) -> std::fmt::Result {
            match self {
                Self::Array(operation) => operation.render(formatter, indentation),
                Self::While(operation) => operation.render(formatter, indentation),
                Self::Guard(operation) => operation.render(formatter, indentation),
                Self::Zero(operation) => operation.render(formatter, indentation),
            }
        }
    }

    impl From<ArrayOperation<Array>> for ExternalOperation {
        fn from(operation: ArrayOperation<Array>) -> Self {
            Self::Array(operation)
        }
    }

    impl From<WhileOperation<ExternalType>> for ExternalOperation {
        fn from(operation: WhileOperation<ExternalType>) -> Self {
            Self::While(operation)
        }
    }

    impl From<ConditionOperation<ExternalType>> for ExternalOperation {
        fn from(_operation: ConditionOperation<ExternalType>) -> Self {
            Self::Array(ArrayOperation::Condition(ConditionOperation::new()))
        }
    }

    impl From<ScanOperation<ExternalType>> for ExternalOperation {
        fn from(operation: ScanOperation<ExternalType>) -> Self {
            // This external universe contains arrays only, so its scan descriptor uses the ordinary array semantics.
            let scan = ScanOperation::new(operation.carry_count(), operation.iteration_count().clone())
                .with_reverse(operation.reverse())
                .with_unroll(operation.unroll())
                .unwrap()
                .with_stacked_axis_shardings(operation.stacked_axis_shardings().to_vec());
            Self::Array(ArrayOperation::Scan(scan))
        }
    }

    impl From<GuardOperation<ExternalType>> for ExternalOperation {
        fn from(operation: GuardOperation<ExternalType>) -> Self {
            Self::Guard(operation)
        }
    }

    impl From<ZeroOperation<ExternalType>> for ExternalOperation {
        fn from(operation: ZeroOperation<ExternalType>) -> Self {
            Self::Zero(operation)
        }
    }

    impl OperationProjection<ArrayType> for ExternalOperation {
        type Projected = ArrayOperation<Array>;
    }

    impl ResidualZeroProvider<ExternalType> for ExternalOperation {}

    impl ResidualStorageOperation<ExternalType> for ExternalOperation {
        fn residual_to_storage(_residual_type: &ExternalType) -> Result<Option<Self>, TypeError> {
            Ok(None)
        }

        fn residual_from_storage(_residual_type: &ExternalType) -> Result<Option<Self>, TypeError> {
            Ok(None)
        }

        fn residual_from_carrier(
            _residual_type: &ExternalType,
            _carrier_type: &ExternalType,
        ) -> Result<Option<Self>, TypeError> {
            Ok(None)
        }
    }

    fn assert_shared_rule<C: Context<Type = ExternalType>>()
    where
        WhileOperation<ExternalType>: DifferentiableOperation<C>,
    {
    }

    type ExternalTrace = TracingContext<ExternalValue, ExternalOperation>;
    assert_shared_rule::<ExternalTrace>();

    // Staging through projected members retains the external program universe and lifts the resulting array type.
    let input_type = ArrayType::new_static(DataType::I64, [2]);
    let (output_type, program) = ExternalTrace::trace(
        |input: Tracer<ExternalTrace>| -> Result<Tracer<ExternalTrace>, ProgramError> {
            let context = ProjectedContext::<_, ArrayType>::new(input.context().clone());
            let input = input.into_projected()?;
            let one = context.one(&ArrayType::scalar(DataType::I64))?;
            let expanded = one.broadcast(input_type.clone(), &[])?;
            let output = input.add(&expanded)?;
            Ok(<Tracer<ExternalTrace> as ValueProjection<ArrayType>>::from_projected(output))
        },
        ExternalType(input_type.clone()),
    )
    .unwrap();
    assert_eq!(output_type, ExternalType(input_type));
    assert_eq!(
        program.to_string(),
        indoc! {"
            lambda %0:i64[2] .
            let %1:i64[] = one [type=i64[]]
                %2:i64[2] = broadcast [output_type=i64[2], output_axes=[]] %1
                %3:i64[2] = add %0 %2
            in (%3)"},
    );
}
