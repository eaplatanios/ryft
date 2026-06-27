use crate::operations::Operation;
use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::programs::{Program, ProgramError, Value};
use crate::tracing::{DomainTracer, DomainTracingContext};
use crate::types::Type;

/// Type/value universe at the core of Ryft that is used by program interpretation, tracing, and transformations like
/// batching and automatic differentiation. A [`Domain`] is purely the type, value, constant, and operation universe
/// that a backend or value model understands. It carries no behavior. It does not describe an active tracing run, and
/// it does not decide what happens when a primitive operation is bound. Active bind handling, and lifting of staged
/// constants into runtime values, live in [`Context`](crate::Context) implementations. This separation allows the
/// same [`Domain`] to be reused by ordinary tracing, batching, linearization, and other transformation contexts. A
/// [`Domain`] that can additionally *apply* operations to values is a [`Context`](crate::Context). Eager backends do
/// so by implementing [`Context`](crate::Context) directly (i.e., binding interprets operations over concrete values),
/// while staging contexts bind through [`StagingContext`](crate::StagingContext)s.
pub trait Domain: Sized {
    /// [`Type`]s that this [`Domain`] uses to represent the abstract metadata associated with its [`Value`]s.
    /// A commonly used [`Type`] is [`ArrayType`](crate::ArrayType), though scalar-only domains can use
    /// [`DataType`](crate::DataType) and richer backends may use richer type representations.
    type Type: Type;

    /// [`Value`] types supported by this [`Domain`]. Instances of this type are what [`Program`] interpretation and
    /// eager transforms operate on. [`Domain::Type`] represents abstract staging metadata, while [`Domain::Value`]
    /// represents the runtime values that inhabit traced programs during execution.
    type Value: Value<Self::Type>;

    /// Constant payload type stored in traced [`Program`]s for this [`Domain`]. For eager domains this is usually the
    /// same type as [`Domain::Value`]. Compiled backends may use a lifetime-free abstract representation here while
    /// reserving [`Domain::Value`] for concrete runtime values.
    type Constant: Value<Self::Type>;

    /// [`Operation`] representation supported by this [`Domain`] for ordinary traced [`Program`]s.
    type Operation: Operation<Self::Type>;

    /// Traces `function` into a [`Program`] for the provided input types, in this [`Domain`]'s type, staged-constant,
    /// and operation universe. This is the canonical ordinary-tracing entry point named by a [`Domain`]. It invokes
    /// `function` once over [`DomainTracer`] inputs standing in for `input_type` through a fresh
    /// [`DomainTracingContext`], and returns the output types plus the finalized program.
    #[inline]
    fn trace<
        F: FnOnce(Input::To<DomainTracer<Self>>) -> Result<Output, ProgramError>,
        Input: Parameterized<
                Self::Type,
                Family: ParameterizedFamily<Self::Constant> + ParameterizedFamily<DomainTracer<Self>>,
            >,
        Output: Parameterized<
                DomainTracer<Self>,
                Family: ParameterizedFamily<Self::Type> + ParameterizedFamily<Self::Constant>,
            >,
    >(
        function: F,
        input_type: Input,
    ) -> Result<
        (
            Output::To<Self::Type>,
            Program<Self::Type, Self::Constant, Self::Operation, Input::To<Self::Constant>, Output::To<Self::Constant>>,
        ),
        ProgramError,
    > {
        DomainTracingContext::<Self>::trace(function, input_type)
    }

    /// Traces `function` against `input_type` in this [`Domain`]'s universe and returns only the inferred output type,
    /// without retaining the traced [`Program`]. Use this when callers need just the output types of an ordinary
    /// symbolic trace named by a [`Domain`].
    #[inline]
    fn infer_output_type<
        F: FnOnce(Input::To<DomainTracer<Self>>) -> Result<Output, ProgramError>,
        Input: Parameterized<
                Self::Type,
                Family: ParameterizedFamily<Self::Constant> + ParameterizedFamily<DomainTracer<Self>>,
            >,
        Output: Parameterized<
                DomainTracer<Self>,
                Family: ParameterizedFamily<Self::Type> + ParameterizedFamily<Self::Constant>,
            >,
    >(
        function: F,
        input_type: Input,
    ) -> Result<Output::To<Self::Type>, ProgramError> {
        DomainTracingContext::<Self>::infer_output_type(function, input_type)
    }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use pretty_assertions::assert_eq;

    use crate::contexts::Context;
    use crate::operations::constants::{OneOperation, ZeroOperation};
    use crate::scalars::{Scalar, ScalarDomain};
    use crate::types::DataType;

    #[test]
    fn test_domain() {
        // [`ScalarDomain`] is an eager `Context` over the self-describing [`Scalar`] value type, so binding a nullary
        // zero/one `Operation` interprets it directly to the [`Scalar`] variant matching the requested [`DataType`].
        let domain = ScalarDomain::new();
        assert_eq!(domain.bind(ZeroOperation::new(DataType::BF16), &[]), Ok(vec![Scalar::BF16(bf16::ZERO)]));
        assert_eq!(domain.bind(OneOperation::new(DataType::BF16), &[]), Ok(vec![Scalar::BF16(bf16::ONE)]));
        assert_eq!(domain.bind(ZeroOperation::new(DataType::F16), &[]), Ok(vec![Scalar::F16(f16::ZERO)]));
        assert_eq!(domain.bind(OneOperation::new(DataType::F16), &[]), Ok(vec![Scalar::F16(f16::ONE)]));
        assert_eq!(domain.bind(ZeroOperation::new(DataType::F32), &[]), Ok(vec![Scalar::F32(0.0)]));
        assert_eq!(domain.bind(OneOperation::new(DataType::F32), &[]), Ok(vec![Scalar::F32(1.0)]));
        assert_eq!(domain.bind(ZeroOperation::new(DataType::F64), &[]), Ok(vec![Scalar::F64(0.0)]));
        assert_eq!(domain.bind(OneOperation::new(DataType::F64), &[]), Ok(vec![Scalar::F64(1.0)]));
    }
}
