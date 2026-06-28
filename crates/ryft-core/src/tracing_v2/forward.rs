#[cfg(test)]
mod tests {
    use crate::contexts::StagingContext;
    use crate::parameters::{ParameterError, Parameterized};
    use crate::programs::ProgramError;
    use crate::scalars::{Scalar, ScalarDomain};
    use crate::tracing::{DomainTracer, DomainTracingContext};
    use crate::tracing_v2::DifferentiationContext;
    use crate::types::DataType;

    #[test]
    fn jvp_rejects_mismatched_parameter_structures() {
        let domain = ScalarDomain::new();
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
        let context = DomainTracingContext::<ScalarDomain>::new();
        let empty_primals: Vec<DomainTracer<ScalarDomain>> = Vec::new();
        let empty_tangents: Vec<DomainTracer<ScalarDomain>> = Vec::new();

        let result: Result<(Vec<DomainTracer<ScalarDomain>>, Vec<DomainTracer<ScalarDomain>>), ProgramError> =
            DifferentiationContext::jvp(&context, |inputs| inputs, empty_primals, empty_tangents);

        assert!(matches!(result, Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 })));
    }

    #[test]
    fn traced_jvp_rejects_mismatched_program_builders() {
        let context_a = DomainTracingContext::<ScalarDomain>::new();
        let context_b = DomainTracingContext::<ScalarDomain>::new();
        let primal_a = context_a.input(DataType::F64);
        let primal_b = context_b.input(DataType::F64);
        let tangent_a = context_a.input(DataType::F64);
        let tangent_b = context_a.input(DataType::F64);

        let result: Result<(DomainTracer<ScalarDomain>, DomainTracer<ScalarDomain>), ProgramError> =
            DifferentiationContext::jvp(
                &context_a,
                |inputs| inputs[0].clone() + inputs[1].clone(),
                vec![primal_a, primal_b],
                vec![tangent_a, tangent_b],
            );

        assert!(matches!(result, Err(ProgramError::MismatchedProgramBuilders)));
    }
}
