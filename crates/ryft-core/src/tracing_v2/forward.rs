#[cfg(test)]
mod tests {
    use crate::contexts::EagerContext;
    use crate::contexts::StagingContext;
    use crate::operations::scalars::ScalarOperation;
    use crate::parameters::{ParameterError, Parameterized};
    use crate::programs::ProgramError;
    use crate::scalars::Scalar;
    use crate::tracing::{DomainTracer, DomainTracingContext};
    use crate::tracing_v2::DifferentiationContext;
    use crate::types::DataType;

    #[test]
    fn jvp_rejects_mismatched_parameter_structures() {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let result: Result<(Scalar, Scalar), ProgramError> =
            domain.jvp(|xs| xs[0].clone(), vec![Scalar::from(2.0)], vec![Scalar::from(1.0), Scalar::from(2.0)]);
        assert!(matches!(
            result,
            Err(ProgramError::Parameter(ParameterError::MismatchedParameterStructures {
                left_structure,
                right_structure,
            })) if left_structure == format!("{:?}", vec![Scalar::from(2.0)].parameter_structure())
                && right_structure
                    == format!("{:?}", vec![Scalar::from(1.0), Scalar::from(2.0)].parameter_structure())
        ));
    }

    #[test]
    fn traced_jvp_requires_input_leaves() {
        let context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let empty_primals: Vec<DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>> = Vec::new();
        let empty_tangents: Vec<DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>> = Vec::new();

        let result: Result<
            (
                Vec<DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>>,
                Vec<DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>>,
            ),
            ProgramError,
        > = DifferentiationContext::jvp(&context, |inputs| inputs, empty_primals, empty_tangents);

        assert!(matches!(result, Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 })));
    }

    #[test]
    fn traced_jvp_rejects_mismatched_program_builders() {
        let context_a = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let context_b = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let primal_a = context_a.input(DataType::F64);
        let primal_b = context_b.input(DataType::F64);
        let tangent_a = context_a.input(DataType::F64);
        let tangent_b = context_a.input(DataType::F64);

        let result: Result<
            (
                DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>,
                DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>,
            ),
            ProgramError,
        > = DifferentiationContext::jvp(
            &context_a,
            |inputs| inputs[0].clone() + inputs[1].clone(),
            vec![primal_a, primal_b],
            vec![tangent_a, tangent_b],
        );

        assert!(matches!(result, Err(ProgramError::MismatchedProgramBuilders)));
    }
}
