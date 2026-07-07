#[cfg(test)]
mod tests {
    use crate::contexts::StagingContext;
    use crate::contexts::{Domain, EagerContext};
    use crate::differentiation::DifferentiationError;
    use crate::operations::arithmetic::Add;
    use crate::operations::scalars::ScalarOperation;
    use crate::parameters::{ParameterError, Parameterized};
    use crate::programs::ProgramError;
    use crate::scalars::Scalar;
    use crate::tracing::{DomainTracer, DomainTracingContext, NestedTracingContext};
    use crate::tracing_v2::Differentiate;
    use crate::types::DataType;

    #[test]
    fn jvp_rejects_mismatched_parameter_structures() {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let result: Result<(Scalar, Scalar), ProgramError> =
            domain.jvp(|xs| Ok(xs[0].clone()), vec![Scalar::from(2.0)], vec![Scalar::from(1.0), Scalar::from(2.0)]);
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
        > = Differentiate::jvp(&context, |inputs| Ok(inputs), empty_primals, empty_tangents);

        assert!(matches!(result, Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 })));
    }

    #[test]
    fn traced_jvp_rejects_mismatched_program_builders_at_bind_time() {
        // Foreign tracers are detected lazily, like everything else about staging: the first operation that binds a
        // primal from another trace fails the builder-identity check in `stage_operation`, which poisons the
        // receiving trace and defers the error to its boundary, exactly like any other staged failure.
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
        > = Differentiate::jvp(
            &context_a,
            |inputs| inputs[0].add(&inputs[1]),
            vec![primal_a, primal_b],
            vec![tangent_a, tangent_b],
        );

        let (primal, _tangent) = result.unwrap();
        assert_eq!(primal.atom_id(), Err(ProgramError::PoisonedValue));
        assert_eq!(context_a.builder().borrow_mut().error.take(), Some(ProgramError::MismatchedProgramBuilders));
    }

    #[test]
    fn trace_rejects_mismatched_program_builder_outputs() {
        // A foreign tracer that escapes a trace as an *output* never flows through a bind, so the trace boundary
        // itself performs the builder-identity check: without it, the foreign atom id would silently alias whichever
        // atom shares its index in the local builder.
        let foreign_context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let foreign = foreign_context.input(DataType::F64);
        let result = EagerContext::<Scalar, ScalarOperation<Scalar>>::trace(move |_input| Ok(foreign), DataType::F64);
        assert!(matches!(result, Err(ProgramError::MismatchedProgramBuilders)));
    }

    #[test]
    fn primal_program_trace_rejects_mismatched_program_builder_outputs() {
        // The reverse-mode tracing prologue performs the same boundary check on the closure's outputs, so a foreign
        // nested tracer smuggled out of a differentiated closure is rejected instead of aliasing a local atom.
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let foreign_context = NestedTracingContext::new(domain.clone());
        let foreign = foreign_context.input(DataType::F64);
        let result = domain.value_and_gradient(move |_input| foreign, Scalar::from(1.0));
        assert!(matches!(result, Err(DifferentiationError::Program(ProgramError::MismatchedProgramBuilders))));
    }
}
