use std::marker::PhantomData;

use crate::operations::Operation;
use crate::programs::Value;
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

    /// [`Value`] types supported by this [`Domain`]. Instances of this type are what [`Program`](crate::Program)
    /// interpretation and eager transforms operate on. [`Domain::Type`] represents abstract staging metadata, while
    /// [`Domain::Value`] represents the runtime values that inhabit traced programs during execution.
    type Value: Value<Self::Type>;

    /// Constant payload type stored in traced [`Program`](crate::Program)s for this [`Domain`]. For eager domains
    /// this is usually the same type as [`Domain::Value`]. Compiled backends may use a lifetime-free abstract
    /// representation here while reserving [`Domain::Value`] for concrete runtime values.
    type Constant: Value<Self::Type>;

    /// [`Operation`] representation supported by this [`Domain`] for ordinary traced [`Program`](crate::Program)s.
    type Operation: Operation<Self::Type>;
}

/// *Abstract* [`Domain`] that simply defines a type universe `(T, V, O)` with no concrete backend behind it. An
/// [`AbstractDomain`] is purely a type universe defining the `Type`, `Value`, `Constant`, and `Operation` types,
/// and nothing more. Concrete backend domains pair that universe with execution semantics and also typically
/// implement [`Context`](crate::Context), so that they can [`lift`](crate::Context::lift) constants and
/// [`bind`](crate::Context::bind) [`Operation`]s. [`AbstractDomain`] sits at the opposite end. It is a zero-sized
/// token that fixes the four associated types and carries no behavior of its own. It exists so that value-level
/// tracing APIs can operate over an explicitly named type universe when there is no backend to borrow, such as when
/// building or transposing [`Program`](crate::Program)s purely symbolically. The active staging, owning an underlying
/// [`ProgramBuilder`](crate::ProgramBuilder) and recording [`Instruction`](crate::Instruction)s, is performed by
/// a wrapping [`TracingContext`](crate::TracingContext). This type only pins down which types that program is
/// expressed over.
#[derive(Copy, Clone, Debug, Default)]
pub struct AbstractDomain<T: Type, V: Value<T>, O: Operation<T>> {
    /// [`PhantomData`] marker tying this zero-sized abstract [`Domain`] to its associated types.
    marker: PhantomData<fn() -> (T, V, O)>,
}

impl<T: Type, V: Value<T>, O: Operation<T>> AbstractDomain<T, V, O> {
    /// Creates a new [`AbstractDomain`].
    #[inline]
    pub const fn new() -> Self {
        Self { marker: PhantomData }
    }
}

impl<T: Type, V: Value<T>, O: Operation<T>> Domain for AbstractDomain<T, V, O> {
    type Type = T;
    type Value = V;
    type Constant = V;
    type Operation = O;
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use pretty_assertions::assert_eq;

    use crate::contexts::Context;
    use crate::operations::constants::{OneOperation, ZeroOperation};
    use crate::programs::ProgramError;
    use crate::scalars::ScalarDomain;
    use crate::types::{DataType, TypeError};

    #[test]
    fn test_domain() {
        // Only the floating-point element types provide eager `Context` support, because that capability routes
        // through operation interpretation and the integer/boolean `ScalarOperation` enums are not interpretable.
        // The integer and boolean zero/one values themselves are covered by the `Zero`/`One` tests in
        // `crate::operations::constants`. A nullary zero/one `Operation` bound on an eager `Context` interprets
        // directly to the corresponding scalar identity, and binding against a mismatched `DataType` fails.
        let bf16_domain = ScalarDomain::<bf16>::new();
        assert_eq!(
            bf16_domain.bind(ZeroOperation::new(DataType::BF16), &[]),
            Ok(vec![bf16::ZERO]),
        );
        assert_eq!(
            bf16_domain.bind(OneOperation::new(DataType::BF16), &[]),
            Ok(vec![bf16::ONE]),
        );

        let f16_domain = ScalarDomain::<f16>::new();
        assert_eq!(
            f16_domain.bind(ZeroOperation::new(DataType::F16), &[]),
            Ok(vec![f16::ZERO]),
        );
        assert_eq!(
            f16_domain.bind(OneOperation::new(DataType::F16), &[]),
            Ok(vec![f16::ONE]),
        );

        let f32_domain = ScalarDomain::<f32>::new();
        assert_eq!(
            f32_domain.bind(ZeroOperation::new(DataType::F32), &[]),
            Ok(vec![0.0f32]),
        );
        assert_eq!(
            f32_domain.bind(OneOperation::new(DataType::F32), &[]),
            Ok(vec![1.0f32]),
        );

        let f64_domain = ScalarDomain::<f64>::new();
        assert_eq!(
            f64_domain.bind(ZeroOperation::new(DataType::F64), &[]),
            Ok(vec![0.0f64]),
        );
        assert_eq!(
            f64_domain.bind(OneOperation::new(DataType::F64), &[]),
            Ok(vec![1.0f64]),
        );
        assert!(matches!(
            f64_domain.bind(ZeroOperation::new(DataType::F32), &[]),
            Err(ProgramError::Type(TypeError { message }))
                if message == "scalar value expected data type f64 but got f32",
        ));
        assert!(matches!(
            f64_domain.bind(OneOperation::new(DataType::F32), &[]),
            Err(ProgramError::Type(TypeError { message }))
                if message == "scalar value expected data type f64 but got f32",
        ));
    }
}
