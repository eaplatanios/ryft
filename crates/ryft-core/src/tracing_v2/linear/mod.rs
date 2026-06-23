/// Structured differential materialization helpers (forward- and reverse-mode Jacobians, Hessian).
mod differential;
/// Public reverse-mode APIs built from traced programs and staged pullbacks.
mod reverse;

pub use differential::{
    CoordinateValue, DifferentiableDomainExtension, Differential, DifferentialBlock, DifferentialRow, Hessian,
    Jacobian, jacrev,
};
pub use reverse::{grad, grad_with_aux, value_and_grad, value_and_grad_with_aux};

#[cfg(test)]
mod tests {
    use indoc::indoc;

    use crate::contexts::StagingContext;
    use crate::operations::scalars::ScalarOperation;
    use crate::operations::trigonometric::Sin;
    use crate::programs::Program;
    use crate::scalars::ScalarDomain;
    use crate::tracing::TracingContext;
    use crate::tracing_v2::DifferentiationContext;
    use crate::types::DataType;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn test_linearize_returns_the_primal_output_and_pushforward() {
        let domain = ScalarDomain::<f64>::new();
        let (primal, pushforward) = domain.linearize(|x| Ok(x.clone() * x.clone() + x.sin()), 2.0f64).unwrap();

        approx_eq(primal, 2.0f64.powi(2) + 2.0f64.sin());
        approx_eq(
            pushforward.apply(&crate::contexts::EagerContext::new(), 1.5f64).unwrap(),
            (4.0 + 2.0f64.cos()) * 1.5,
        );
        let pushforward = pushforward.instantiate_program().unwrap();
        assert_eq!(
            pushforward.to_string(),
            indoc! {"
            lambda %0:f64 .
            let %1:f64 = scale [factor=2] %0
                %2:f64 = scale [factor=2] %0
                %3:f64 = add %1 %2
                %4:f64 = scale [factor=-0.4161468365471424] %0
                %5:f64 = add %3 %4
            in (%5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_transposed_linear_program_matches_the_reverse_mode_pullback() {
        let domain = ScalarDomain::<f64>::new();
        let (primal, pushforward) = domain
            .linearize(|inputs| Ok(inputs.0.clone() * inputs.1 + inputs.0.sin()), (2.0f64, 3.0f64))
            .unwrap();
        let pullback = pushforward.instantiate_program().unwrap().transpose().unwrap();
        let cotangent = pullback.interpret(1.0f64).unwrap();

        approx_eq(primal, 2.0 * 3.0 + 2.0f64.sin());
        approx_eq(cotangent.0, 3.0 + 2.0f64.cos());
        approx_eq(cotangent.1, 2.0);
        assert_eq!(
            pullback.to_string(),
            indoc! {"
            lambda %0:f64 .
            let %1:f64 = scale [factor=-0.4161468365471424] %0
                %2:f64 = scale [factor=2] %0
                %3:f64 = scale [factor=3] %0
                %4:f64 = add %1 %3
            in (%4, %2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn linear_program_display_delegates_to_the_underlying_program() {
        let domain = ScalarDomain::<f64>::new();
        let (_, pushforward) = domain.linearize(|x| Ok(x.clone() * x.clone() + x.sin()), 2.0f64).unwrap();
        let pushforward = pushforward.instantiate_program().unwrap();

        assert_eq!(
            pushforward.to_string(),
            indoc! {"
            lambda %0:f64 .
            let %1:f64 = scale [factor=2] %0
                %2:f64 = scale [factor=2] %0
                %3:f64 = add %1 %2
                %4:f64 = scale [factor=-0.4161468365471424] %0
                %5:f64 = add %3 %4
            in (%5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn active_and_program_linearization_have_matching_residualized_pushforwards() {
        let domain = ScalarDomain::<f64>::new();
        let (active_output, active_pushforward) =
            domain.linearize(|x| Ok(x.clone() * x.clone() + x.sin()), 2.0f64).unwrap();
        let (program_output, program): (f64, Program<DataType, f64, ScalarOperation<f64>, f64, f64>) =
            TracingContext::interpret_and_trace(&domain, |x| Ok(x.clone() * x.clone() + x.sin()), 2.0f64).unwrap();
        let (replayed_output, replayed_pushforward) = domain.linearize_program(&program, vec![2.0f64]).unwrap();

        approx_eq(active_output, program_output);
        approx_eq(active_output, replayed_output);
        assert_eq!(active_pushforward.residuals().len(), replayed_pushforward.residuals().len());
        approx_eq(
            active_pushforward.apply(&crate::contexts::EagerContext::new(), 1.5f64).unwrap(),
            replayed_pushforward.apply(&crate::contexts::EagerContext::new(), 1.5f64).unwrap(),
        );
    }

    #[test]
    fn duplicate_primal_factors_share_one_residual() {
        let domain = ScalarDomain::<f64>::new();
        let (_, pushforward) = domain.linearize(|x| Ok(x.clone() * x), 2.0f64).unwrap();

        assert_eq!(pushforward.residuals().len(), 1);
        approx_eq(pushforward.apply(&crate::contexts::EagerContext::new(), 3.0f64).unwrap(), 12.0);
    }

    #[test]
    fn dead_primal_factors_are_not_kept_as_residuals() {
        let domain = ScalarDomain::<f64>::new();
        let (_, pushforward) = domain
            .linearize(
                |x| {
                    let _dead = x.clone() * x.clone();
                    Ok(x)
                },
                2.0f64,
            )
            .unwrap();

        assert_eq!(pushforward.residuals().len(), 0);
        approx_eq(pushforward.apply(&crate::contexts::EagerContext::new(), 7.0f64).unwrap(), 7.0);
    }

    #[test]
    fn closed_constants_do_not_become_residuals() {
        let domain = ScalarDomain::<f64>::new();
        let (_, pushforward) = domain
            .linearize(
                |x| {
                    let constant = x.context().constant(3.0);
                    Ok(x * constant)
                },
                2.0f64,
            )
            .unwrap();

        assert_eq!(pushforward.residuals().len(), 0);
        approx_eq(pushforward.apply(&crate::contexts::EagerContext::new(), 4.0f64).unwrap(), 12.0);
    }
}
