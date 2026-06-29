use std::fmt::Display;

use crate::contexts::StagingContext;
use crate::differentiation::{Cotangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::constants::{MaybeZeroOperation, ZeroOperation};
use crate::operations::{ElementwiseOperation, InterpretableOperation, Operation};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};
use crate::tracing_v2::linearization::{DifferentiableOperation, JvpTracer};
use crate::types::{Type, TypeError};

/// Canonical operation name for [`TagOperation`].
pub const TAG_OPERATION_NAME: &'static str = "tag";

/// [`Operation`] that returns its input unchanged while tagging it with a key that is visible to program transforms.
/// This is useful for features like key-based rematerialization in automatic differentiation transforms. Refer to the
/// documentation of [`Tag`] for more information.
///
/// Interpretation, batching, and backend lowering all treat this operation as an identity function. Differentiation
/// passes the tangent through unchanged while re-tagging the primal value so that the tag is visible to instructions
/// that define linearization residuals, which is exactly what consumers such as key-based rematerialization strategies
/// need.
#[derive(Clone, Debug)]
pub struct TagOperation {
    /// Key tagging the operation's output [`Value`].
    key: String,
}

impl TagOperation {
    /// Creates a new [`TagOperation`] with the provided key.
    #[inline]
    pub fn new<K: Into<String>>(key: K) -> Self {
        Self { key: key.into() }
    }

    /// Returns the key carried by this [`TagOperation`].
    #[inline]
    pub fn key(&self) -> &str {
        self.key.as_str()
    }
}

impl Display for TagOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{TAG_OPERATION_NAME}[{}]", self.key)
    }
}

impl<T: Type> Operation<T> for TagOperation {
    #[inline]
    fn name(&self) -> &'static str {
        TAG_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![input_types[0].clone()])
    }
}

impl ElementwiseOperation for TagOperation {
    #[inline]
    fn input_count(&self) -> usize {
        1
    }
}

impl<T: Type, V: Clone + Value<T>> InterpretableOperation<T, V> for TagOperation {
    #[inline]
    fn interpret(
        &self,
        _context: &<V as Value<T>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].clone()])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<T: Type, V: Value<T>, O> PartiallyEvaluatableOperation<T, V, O> for TagOperation {}

/// Represents the ability to tag values in programs with keys. [`Tag`] stages a [`TagOperation`], which is effectively
/// an identity function carrying a string-valued key. The tag gets attached to traced values and survives forward-mode
/// differentiation (the [`DifferentiableOperation`] rule re-tags the primal value and passes the tangent value
/// through), so that it marks the instructions that define linearization residuals. [`MaybeTag`] enables transforms
/// to classify staged instructions by key without knowing the concrete operation type.
pub trait Tag: Sized {
    /// Returns this value unchanged while tagging it with `key`.
    fn tag(self, key: &str) -> Self;
}

/// Trait that enables checking if an operation is a [`TagOperation`] and, if so, extract the underlying key.
pub trait MaybeTag {
    /// Returns the underlying key if this operation is a [`TagOperation`], and [`None`] otherwise.
    fn key(&self) -> Option<&str>;
}

impl<C: StagingContext<Operation: From<TagOperation>>> Tag for Tracer<C, C::Meta> {
    #[inline]
    fn tag(self, key: &str) -> Self {
        self.unary(TagOperation::new(key))
    }
}

impl<
    C: StagingContext<
            Value: Tag,
            Operation: Clone + MaybeZeroOperation<C::Type> + From<ZeroOperation<C::Type>> + From<TagOperation>,
        >,
> DifferentiableOperation<C> for TagOperation
{
    fn jvp(&self, _context: &C, inputs: &[JvpTracer<C>]) -> Result<Vec<JvpTracer<C>>, ProgramError> {
        // We re-tag the input primal for downstream classification while letting the input tangent pass through
        // unchanged, matching the identity tangent of the tag.
        check_count!("input", inputs, 1, ProgramError);
        let primal = inputs[0].primal().clone().tag(self.key());
        Ok(vec![JvpTracer::new(primal, inputs[0].tangent().clone())])
    }
}

impl<T: Type, V: Value<T>, O: Operation<T>> TransposableOperation<T, V, O> for TagOperation {
    #[inline]
    fn transpose(
        &self,
        _context: &mut TracingContext<T, V, O>,
        _inputs: &[PartialValue<T, Tracer<TracingContext<T, V, O>>>],
        outputs: &[Cotangent<T, V, O>],
    ) -> Result<Vec<Cotangent<T, V, O>>, ProgramError> {
        // `TagOperation` acts as a linear identity function (i.e., `y = x`), and so its adjoint is the identity
        // function. The single output cotangent passes straight through to the single input, staging nothing and
        // leaving the cotangent untagged. That is because the tag is meant to mark forward residuals and not adjoints.
        // Based on the `DifferentiableOperation` implementation for `TagOperation`, the tag rides only the primal
        // value and so a tag should never reach a transposed program, in practice. This implementation exists mainly
        // for completeness and so that owning closed operation families (i.e., operation enums) can implement
        // `TransposableOperation` themselves if one of their variants holds a `TagOperation` payload.
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![outputs[0].clone()])
    }
}

// TODO(eaplatanios): Add unit tests mirroring the structure and style of the tests in
//  `ryft_core::operations::arithmetic::add`, including checks for the `DifferentiableOperation` and the
//  `TransposableOperation` implementations.
