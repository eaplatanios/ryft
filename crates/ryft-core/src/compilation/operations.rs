use crate::programs::{Operation, TypeError, Value};

/// Operation-family capability for representing a call to a staged program. The call operation carries only metadata.
/// Its callee is a flat program, whose captures have been lifted into leading inputs, composed into the region driver
/// passed to [`Context::bind`](crate::Context::bind), which interns it as a shared callee root region by
/// [`Arc`](std::sync::Arc) identity. The concrete operation family decides how that boundary lowers and how batching,
/// differentiation, partial evaluation, and other transforms rewrite it. This keeps higher-order call semantics with
/// the operation that owns them while the lifecycle and capture plumbing remain backend-neutral.
pub trait StagedCallOperation<Constant: Value>: Operation<Type = Constant::Type> + Sized {
    /// Constructs a call operation. The accompanying [`Context::bind`](crate::Context::bind) supplies its callee
    /// through a shared-callee region driver, while `capture_count` records the exact leading lifted-capture prefix
    /// of that callee boundary.
    fn staged_call(capture_count: usize) -> Self;

    /// Constructs one call with its exact callee signature. Families that export callee-local nominal identities
    /// override this hook to allocate instruction-owned result identities while forwarding input identities.
    /// Allocation happens when the call is built, and output inference must remain deterministic. The default returns
    /// [`Self::staged_call`], leaving result inference to the family. A family whose inference reports the callee's
    /// result types then makes each call instruction define the callee-local identities itself. Program construction
    /// rejects a second call to the same callee in one region, because it would define those identities again although
    /// sharing a callee root does not make the runtime results of separate calls the same quantity. Override this
    /// function to instantiate fresh result identities for every call instead.
    #[inline]
    fn staged_call_with_types(
        capture_count: usize,
        input_types: &[Constant::Type],
        output_types: &[Constant::Type],
    ) -> Result<Self, TypeError> {
        let _ = input_types;
        let _ = output_types;
        Ok(Self::staged_call(capture_count))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use ryft_macros::Operation;

    use crate::arrays::{
        Array, ArrayIrType, ArrayIrValue, ArrayType, DataType, DimensionBounds, DimensionType, DimensionVariable,
    };
    use crate::operations::DimensionFromScalarOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError, RegionInterface, RegionSlot, TypedProgram};

    use super::*;

    #[test]
    fn test_staged_call_operation_staged_call_with_types() {
        /// Call operation whose results are exactly the results of its callee.
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        struct TestCallOperation;

        impl Operation for TestCallOperation {
            type Type = ArrayIrType;

            fn name(&self) -> &'static str {
                "test_call"
            }

            fn region_slots(&self) -> &'static [RegionSlot] {
                const { &[RegionSlot::computation("callee")] }
            }

            fn infer_output_types(
                &self,
                _input_types: &[ArrayIrType],
                region_interfaces: &[RegionInterface<ArrayIrType>],
            ) -> Result<Vec<ArrayIrType>, TypeError> {
                Ok(region_interfaces[0].output_types().to_vec())
            }
        }

        /// Operation family that implements only the required [`StagedCallOperation::staged_call`] constructor.
        #[derive(Clone, Debug, Operation)]
        #[ryft(type(ArrayIrType), constant(ArrayIrValue<Array>))]
        enum TestOperation {
            DimensionFromScalar(DimensionFromScalarOperation),
            Call(TestCallOperation),
        }

        impl StagedCallOperation<ArrayIrValue<Array>> for TestOperation {
            fn staged_call(_capture_count: usize) -> Self {
                Self::Call(TestCallOperation)
            }
        }

        // The callee defines its result dimension itself, so the identity is local to the callee.
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(8)).unwrap());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, TestOperation>::new();
        let extent = builder.add_input(ArrayType::scalar(DataType::I64).into());
        let dimension = builder
            .add_instruction(
                DimensionFromScalarOperation::new(DimensionType::from(rows)),
                Vec::new(),
                vec![extent],
                None,
            )
            .unwrap()[0];
        let callee = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![dimension],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        // The default binds `staged_call`, so the call instruction itself defines the callee-local result identity.
        let operation =
            TestOperation::staged_call_with_types(0, &callee.input_types(), &callee.output_types()).unwrap();
        assert!(matches!(operation, TestOperation::Call(TestCallOperation)));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, TestOperation>::new();
        let extent = builder.add_input(ArrayType::scalar(DataType::I64).into());
        let region = builder.import_program(callee.clone());
        let output = builder.add_instruction(operation.clone(), vec![region], vec![extent], None).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:i64[] .
                let %1:dimension<rows ∈ [0, 8)> = test_call %0 [
                    callee={
                        lambda %0:i64[] .
                        let %1:dimension<rows ∈ [0, 8)> = dimension_from_scalar [bounds=[0, 8)] %0
                        in (%1)
                    },
                ]
                in (%1)"},
        );

        // A second call to the same callee cannot define that identity again, because separate calls need not produce
        // the same extent.
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, TestOperation>::new();
        let region = builder.import_program(callee);
        let first = builder.add_input(ArrayType::scalar(DataType::I64).into());
        let second = builder.add_input(ArrayType::scalar(DataType::I64).into());
        let first = builder.add_instruction(operation.clone(), vec![region], vec![first], None).unwrap()[0];
        let second = builder.add_instruction(operation, vec![region], vec![second], None).unwrap()[0];
        assert!(matches!(
            builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![first, second],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            ),
            Err(ProgramError::Type(TypeError::Invalid { message, .. }))
                if message == "operation `test_call` output defines identity rows more than once in this region",
        ));
    }
}
