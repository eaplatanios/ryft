use std::fmt::Debug;

use crate::operations::InterpretableOperation;
use crate::parameters::{Parameterized, ParameterizedFamily};

pub mod contexts;
pub mod domains;
pub mod errors;
pub mod programs;

pub use contexts::{CaptureContext, Context, ProgramTracingContext, TracingContext};
pub use domains::{
    CapturingDomain, Domain, DomainTracer, LinearScalarDomain, ProgramTracer, ProgramTracingDomain, RuntimeDomain,
    ScalarDomain, Tracer, TracerState, TracingDomain,
};
pub use errors::TracingError;
pub use programs::{Atom, AtomId, Instruction, Program, ProgramBuilder, Traceable, Value};

/// Traces the provided `function` into a [`Program`]. This is the module-level equivalent of [`TracingContext::trace`].
///
/// # Parameters
///
///   - `domain`: [`TracingDomain`] that provides the traced operation, type, and constant representations.
///   - `function`: Function/closure to trace.
///   - `input_type`: Type of the input to the function being traced. This is used to determine the types of the
///     traced [`Program`] output.
#[inline]
pub fn trace<
    'domain,
    D: TracingDomain,
    F: FnOnce(I::To<DomainTracer<'domain, D>>) -> Result<O, TracingError>,
    I: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<DomainTracer<'domain, D>>>,
    O: Parameterized<DomainTracer<'domain, D>, Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>>,
>(
    domain: &'domain D,
    function: F,
    input_type: I,
) -> Result<
    (O::To<D::Type>, Program<D::Type, D::Constant, D::Operation, I::To<D::Constant>, O::To<D::Constant>>),
    TracingError,
> {
    TracingContext::trace(domain, function, input_type)
}

/// Traces the provided `function` into a [`Program`] and interprets that program on the provided `input`. This is the
/// module-level equivalent of [`TracingContext::interpret_and_trace`].
///
/// # Parameters
///
///   - `domain`: [`TracingDomain`] that provides the traced operation, type, and constant representations.
///   - `function`: Function/closure to trace and interpret/execute.
///   - `input`: Input value to use for tracing and interpreting the provided function.
#[inline]
pub fn interpret_and_trace<
    'domain,
    D: TracingDomain<Operation: Clone + InterpretableOperation<D::Type, D::Value>>,
    F: FnOnce(I::To<DomainTracer<'domain, D>>) -> Result<O, TracingError>,
    I: Parameterized<
            D::Value,
            Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<DomainTracer<'domain, D>>,
            ParameterStructure: Debug + PartialEq,
        >,
    O: Parameterized<DomainTracer<'domain, D>, Family: ParameterizedFamily<D::Value> + ParameterizedFamily<D::Constant>>,
>(
    domain: &'domain D,
    function: F,
    input: I,
) -> Result<
    (O::To<D::Value>, Program<D::Type, D::Constant, D::Operation, I::To<D::Constant>, O::To<D::Constant>>),
    TracingError,
> {
    TracingContext::interpret_and_trace(domain, function, input)
}

/// Traces the provided `function` against `input_type` and returns its inferred output type. This is the module-level
/// equivalent of [`TracingContext::infer_output_type`].
///
/// # Parameters
///
///   - `domain`: [`TracingDomain`] that provides the traced operation, type, and constant representations.
///   - `function`: Function/closure whose output type to infer.
///   - `input_type`: Type of the input to the function that will be used to infer the type of its output.
#[inline]
pub fn infer_output_type<
    'domain,
    D: TracingDomain,
    F: FnOnce(I::To<DomainTracer<'domain, D>>) -> Result<O, TracingError>,
    I: Parameterized<D::Type, Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<DomainTracer<'domain, D>>>,
    O: Parameterized<DomainTracer<'domain, D>, Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>>,
>(
    domain: &'domain D,
    function: F,
    input_type: I,
) -> Result<O::To<D::Type>, TracingError> {
    TracingContext::infer_output_type(domain, function, input_type)
}

#[cfg(test)]
mod tests {
    use crate::{DataType, ScalarDomain, Sin, infer_output_type, interpret_and_trace, trace};

    #[test]
    fn test_trace_function_delegates_to_context_tracing() {
        let domain = ScalarDomain::<f64>::new();
        let (output_type, program) = trace(&domain, |x| Ok(x.clone() * x), DataType::F64).unwrap();
        assert_eq!(output_type, DataType::F64);
        assert_eq!(program.interpret(3.0), Ok(9.0));
    }

    #[test]
    fn test_interpret_and_trace_function_delegates_to_context_tracing() {
        let domain = ScalarDomain::<f64>::new();
        let (output, program) = interpret_and_trace(&domain, |x| Ok(x.clone() * x.clone() + x.sin()), 2.0).unwrap();
        assert_eq!(output, 2.0 * 2.0 + 2.0f64.sin());
        assert_eq!(program.interpret(3.0), Ok(3.0 * 3.0 + 3.0f64.sin()));
    }

    #[test]
    fn test_infer_output_type_function_delegates_to_context_tracing() {
        let domain = ScalarDomain::<f64>::new();
        let output_type = infer_output_type(&domain, |x| Ok(x.sin()), DataType::F64).unwrap();
        assert_eq!(output_type, DataType::F64);
    }
}
