//! Contains test support utilities shared by `ryft` unit tests, doctests, and downstream crates: the
//! [`check_gradient!`](crate::check_gradient) finite-difference gradient oracle and the region-carrying
//! [`TestRegionOperation`]. Concrete reference values live in the [`backends`](crate::backends) module instead:
//! [`Scalar`](crate::backends::scalars::Scalar) for the scalar universe and
//! [`Array`](crate::backends::arrays::Array) for the array universe.
//!
//! The module is part of `ryft-core`'s public API so downstream tests and documentation examples can use it without
//! feature configuration, but its contents exist only for tests and documentation examples.

use crate::macros::check_count;
use crate::programs::effects::{Effect, Effects};
use crate::programs::operations::Operation;
use crate::programs::regions::RegionInterface;
use crate::programs::types::TypeError;
use crate::types::DataType;

/// Asserts that the reverse-mode gradient of `$function` at `$input` matches a central finite-difference estimate
/// of its derivative within absolute tolerance `$tolerance`. The leading selector picks the value universe:
/// `@scalar` checks a [`Scalar`](crate::backends::scalars::Scalar)-valued function, while `@array` checks an
/// [`Array`](crate::backends::arrays::Array)-valued function of any input shape whose output is a rank-0 real `f64`
/// array (the only shape the plain [`gradient`](crate::differentiation::gradient) entry point accepts). This is the
/// standard oracle for testing operation gradient rules without hand-deriving the expected derivative — and without
/// trusting the machinery under test: the gradient side runs `$function` through
/// [`gradient`](crate::differentiation::gradient), while the finite-difference side evaluates `$function` directly on
/// concrete values at the perturbed points, never touching the differentiation machinery it is checking. That double
/// instantiation is why this is a macro: `$function` must be a closure literal (or a generic function), and the
/// internal `@check` rule shared by both selectors instantiates it once over
/// [`LinearizationTracer`](crate::differentiation::LinearizationTracer) inputs and once over concrete
/// [`Scalar`](crate::backends::scalars::Scalar) or [`Array`](crate::backends::arrays::Array) inputs before
/// dispatching to the selector's internal assertion rule.
///
/// An `f64`(-typed) input estimates the ordinary derivative `(f(x + h) - f(x - h)) / (2h)` — under `@array`, once per
/// input element with all other elements held fixed, assembling the estimated gradient array. A `c128`(-typed) input
/// requires a ℂ → ℝ `$function` and estimates both real partials (again per element under `@array`) with central
/// differences along the real and imaginary axes, assembling `complex(∂f/∂re, -∂f/∂im)` — the conjugate
/// steepest-ascent gradient the bilinear transposition pairing returns (e.g., `2z̄` for `f(z) = |z|²`). Other input
/// data types (including `c64`, whose `f32` precision cannot support a meaningful central difference) panic. Pick an
/// `$input` away from any non-differentiable point of `$function` (e.g., the kink of `abs` at zero) and a
/// `$tolerance` compatible with the O(`$step`²) truncation error of the central difference.
#[macro_export]
macro_rules! check_gradient {
    (@scalar, $function:expr, $input:expr, $step:expr, $tolerance:expr $(,)?) => {
        $crate::check_gradient!(
            @check(
                $crate::backends::scalars::Scalar,
                $crate::backends::scalars::ScalarOperation<$crate::backends::scalars::Scalar>,
                @assert_scalar,
            )
            $function, $input, $step, $tolerance
        )
    };
    (@array, $function:expr, $input:expr, $step:expr, $tolerance:expr $(,)?) => {
        $crate::check_gradient!(
            @check(
                $crate::backends::arrays::Array,
                $crate::backends::arrays::ArrayOperation<$crate::backends::arrays::Array>,
                @assert_array,
            )
            $function, $input, $step, $tolerance
        )
    };
    // Internal rule shared by both selectors: `$value` and `$operation` pick the eager context whose linearization
    // tracer pins the traced instantiation of `$function`, and `@$assert` names the internal rule below that checks
    // the resulting gradient against the concrete-side finite-difference estimate.
    (
        @check($value:ty, $operation:ty, @$assert:ident $(,)?)
        $function:expr, $input:expr, $step:expr, $tolerance:expr
    ) => {{
        // Closure parameter types infer from an expected type, so each instantiation of `$function` flows through
        // an identity function pinning the signature that instantiation is used at.
        type EagerCheckContext = $crate::contexts::EagerContext<$value, $operation>;
        fn pin_traced<F>(function: F) -> F
        where
            F: Fn(
                $crate::differentiation::LinearizationTracer<EagerCheckContext>,
            ) -> $crate::differentiation::LinearizationTracer<EagerCheckContext>,
        {
            function
        }
        fn pin_eager<F: Fn($value) -> $value>(function: F) -> F {
            function
        }
        let input: $value = ::core::convert::Into::into($input);
        let step: f64 = $step;
        let tolerance: f64 = $tolerance;
        let gradient = $crate::differentiation::gradient(pin_traced($function), input.clone()).unwrap();
        $crate::check_gradient!(@$assert(gradient, pin_eager($function), input, step, tolerance))
    }};
    // Internal rule behind `@scalar`: checks a reverse-mode `$gradient` of the ℝ → ℝ or ℂ → ℝ function `$evaluate`
    // at `$input` against the central finite-difference estimate of its derivative.
    (@assert_scalar($gradient:expr, $evaluate:expr, $input:expr, $step:expr, $tolerance:expr $(,)?)) => {{
        let gradient = $gradient;
        let evaluate = $evaluate;
        let input: $crate::backends::scalars::Scalar = $input;
        let step: f64 = $step;
        let tolerance: f64 = $tolerance;
        let central_difference = |plus: $crate::backends::scalars::Scalar, minus: $crate::backends::scalars::Scalar| {
            (evaluate(plus) - evaluate(minus)) / $crate::backends::scalars::Scalar::from(2.0 * step)
        };
        match input {
            $crate::backends::scalars::Scalar::F64(input) => {
                let estimate = central_difference(
                    $crate::backends::scalars::Scalar::from(input + step),
                    $crate::backends::scalars::Scalar::from(input - step),
                );
                ::approx::assert_abs_diff_eq!(gradient, estimate, epsilon = tolerance);
            }
            $crate::backends::scalars::Scalar::C128(_) => {
                // The two central differences estimate the real partials that assemble the conjugate steepest-ascent
                // gradient `complex(∂f/∂re, -∂f/∂im)`.
                let (real_step, imaginary_step) = $crate::check_gradient!(@complex_perturbation_steps(step));
                let real_estimate = central_difference(input + real_step, input - real_step);
                let imaginary_estimate = central_difference(input + imaginary_step, input - imaginary_step);
                let estimate =
                    $crate::operations::complex::Complex::complex(&real_estimate, &(-imaginary_estimate)).unwrap();
                ::approx::assert_abs_diff_eq!(gradient, estimate, epsilon = tolerance);
            }
            other => panic!(
                "finite-difference gradient checking requires an f64 or c128 input but got {}",
                $crate::programs::types::Typed::r#type(&other).into_owned(),
            ),
        }
    }};
 
    // Internal rule behind the `@array` branch of this macro. It checks a reverse-mode `$gradient` of the ℝⁿ → ℝ or
    // ℂⁿ → ℝ function `$evaluate` at `$input` (an array of any shape whose output is a rank-0 real `f64` array) against
    // the central finite-difference estimates of its partials, perturbing one input element at a time with all others
    // held fixed.
    (@assert_array($gradient:expr, $evaluate:expr, $input:expr, $step:expr, $tolerance:expr $(,)?)) => {{
        let gradient = $gradient;
        let evaluate = $evaluate;
        let input: $crate::backends::arrays::Array = $input;
        let step: f64 = $step;
        let tolerance: f64 = $tolerance;
        // The function output is a rank-0 real array, so the central difference reads its single `f64` element.
        let central_difference = |plus: $crate::backends::arrays::Array, minus: $crate::backends::arrays::Array| {
            (evaluate(plus).to_f64s()[0] - evaluate(minus).to_f64s()[0]) / (2.0 * step)
        };
        let input_type = $crate::programs::types::Typed::r#type(&input).into_owned();
        let element_count = input.values().len();
        match input_type.data_type() {
            $crate::types::DataType::F64 => {
                let perturbed = |index: usize, delta: f64| {
                    let mut values = input.to_f64s();
                    values[index] += delta;
                    $crate::backends::arrays::Array::from_f64s(input_type.clone(), values)
                };
                let estimates = (0..element_count)
                    .map(|index| central_difference(perturbed(index, step), perturbed(index, -step)))
                    .collect::<Vec<_>>();
                let estimate = $crate::backends::arrays::Array::from_f64s(input_type.clone(), estimates);
                ::approx::assert_abs_diff_eq!(gradient, estimate, epsilon = tolerance);
            }
            $crate::types::DataType::C128 => {
                // Per input element, the two central differences estimate the real partials that assemble the
                // conjugate steepest-ascent gradient `complex(∂f/∂re, -∂f/∂im)`.
                let (real_step, imaginary_step) = $crate::check_gradient!(@complex_perturbation_steps(step));
                let perturbed = |index: usize, delta: $crate::backends::scalars::Scalar| {
                    let mut values = input.values().to_vec();
                    values[index] = values[index] + delta;
                    $crate::backends::arrays::Array::new(input_type.clone(), values).unwrap()
                };
                let mut real_estimates = Vec::with_capacity(element_count);
                let mut imaginary_estimates = Vec::with_capacity(element_count);
                for index in 0..element_count {
                    real_estimates
                        .push(central_difference(perturbed(index, real_step), perturbed(index, -real_step)));
                    imaginary_estimates.push(-central_difference(
                        perturbed(index, imaginary_step),
                        perturbed(index, -imaginary_step),
                    ));
                }
                let part_type = input_type.clone().with_data_type($crate::types::DataType::F64);
                let estimate = $crate::operations::complex::Complex::complex(
                    &$crate::backends::arrays::Array::from_f64s(part_type.clone(), real_estimates),
                    &$crate::backends::arrays::Array::from_f64s(part_type, imaginary_estimates),
                )
                .unwrap();
                ::approx::assert_abs_diff_eq!(gradient, estimate, epsilon = tolerance);
            }
            other => panic!("finite-difference gradient checking requires an f64 or c128 input but got {other}"),
        }
    }};
    
    // Internal rule shared by the complex arms of both assertion rules of this macro. It builds the complex-valued
    // real- and imaginary-axis perturbation steps so that the central differences remain in the complex tangent space.
    (@complex_perturbation_steps($step:expr)) => {{
        let real_step = $crate::operations::complex::Complex::complex(
            &$crate::backends::scalars::Scalar::from($step),
            &$crate::backends::scalars::Scalar::from(0.0),
        )
        .unwrap();
        let imaginary_step = $crate::operations::complex::Complex::complex(
            &$crate::backends::scalars::Scalar::from(0.0),
            &$crate::backends::scalars::Scalar::from($step),
        )
        .unwrap();
        (real_step, imaginary_step)
    }};
}

pub use check_gradient;

/// Test [`Operation`] with declared attached-region slots, used to exercise the region-carrying construction,
/// inference, validation, effects, rendering, and rebuild paths before any production operation family migrates onto
/// attached regions. Like the rest of this module, it exists only for tests and documentation examples.
#[derive(Clone, Debug, PartialEq)]
pub enum TestRegionOperation {
    /// Region-free binary addition stand-in used inside region bodies.
    Add,

    /// Region-free unary identity stand-in with an observable ordered-IO effect.
    Effectful,

    /// Region-carrying operation declaring the provided region slot names. Its inferred output types are the first
    /// attached region's output types, which pins that region interfaces are derived and delivered during inference.
    WithRegions(&'static [&'static str]),
}

impl Operation<DataType> for TestRegionOperation {
    fn name(&self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Effectful => "effectful",
            Self::WithRegions(_) => "with_regions",
        }
    }

    fn infer_output_types(
        &self,
        input_types: &[DataType],
        region_interfaces: &[RegionInterface<DataType>],
    ) -> Result<Vec<DataType>, TypeError> {
        match self {
            Self::Add => {
                check_count!("input", input_types, 2, TypeError);
                Ok(vec![input_types[0]])
            }
            Self::Effectful => {
                check_count!("input", input_types, 1, TypeError);
                Ok(vec![input_types[0]])
            }
            Self::WithRegions(names) => {
                check_count!("input", input_types, 1, TypeError);
                if region_interfaces.len() != names.len() {
                    return Err(TypeError {
                        message: format!(
                            "expected {} region interfaces but got {}",
                            names.len(),
                            region_interfaces.len(),
                        ),
                    });
                }
                Ok(region_interfaces[0].output_types().to_vec())
            }
        }
    }

    fn region_names(&self) -> &'static [&'static str] {
        match self {
            Self::Add | Self::Effectful => &[],
            Self::WithRegions(names) => names,
        }
    }

    fn effects(&self) -> Effects {
        match self {
            Self::Add | Self::WithRegions(_) => Effects::PURE,
            Self::Effectful => Effects::single(Effect::OrderedIo),
        }
    }
}
