use std::fmt::Display;

use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiationDriver, DifferentiationDual, DifferentiationError, TransposableOperation,
    TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::ElementwiseOperation;
use crate::operations::constants::ZeroOperation;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::ProgramError;
use crate::programs::atoms::MaybeZero;
use crate::programs::operations::Operation;
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError};
use crate::programs::values::Value;
use crate::tracing::{Tracer, TracingContext};

/// Canonical operation name for [`TagOperation`].
pub const TAG_OPERATION_NAME: &str = "tag";

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
    fn infer_output_types(
        &self,
        input_types: &[T],
        _region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
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

impl<C: Domain> InterpretableOperation<C> for TagOperation {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].clone()])
    }
}

impl<C: Context<Operation: From<TagOperation>>> PartiallyEvaluatableOperation<C> for TagOperation {}

/// Represents the ability to tag values in programs with keys. [`Tag`] stages a [`TagOperation`], which is effectively
/// an identity function carrying a string-valued key. The tag gets attached to traced values and survives forward-mode
/// differentiation (the [`DifferentiableOperation`] rule re-tags the primal value and passes the tangent value
/// through), so that it marks the instructions that define linearization residuals, which rematerialization
/// policies classify by key through the producing [`TagOperation`].
pub trait Tag: Sized {
    /// Returns this value unchanged while tagging it with `key`.
    fn tag(self, key: &str) -> Self;
}

impl<V: Value<DispatchDomain: Context<Operation: From<TagOperation>>>> Tag for V {
    #[inline]
    fn tag(self, key: &str) -> Self {
        self.dispatch_domain()
            .bind(TagOperation::new(key), Vec::new(), std::slice::from_ref(&self))
            .expect("`tag` operation failed")
            .remove(0)
    }
}

impl<C: Context> DifferentiableOperation<C> for TagOperation
where
    C::Operation: From<ZeroOperation<C::Type>> + From<TagOperation>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // We re-tag the input primal for downstream classification while letting the input tangent pass through
        // unchanged, matching the identity tangent of the tag. The tag binds through the context so the rule works
        // uniformly under staging and eager contexts.
        check_count!("input", inputs, 1, ProgramError);
        let mut primal =
            context.bind(TagOperation::new(self.key()), Vec::new(), std::slice::from_ref(inputs[0].primal()))?;
        check_count!("output", primal, 1, ProgramError);
        Ok(vec![DifferentiationDual::new(primal.remove(0), inputs[0].tangent().clone())])
    }
}

impl<V: Value, O: Operation<V::Type>> TransposableOperation<V, O> for TagOperation {
    #[inline]
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
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
//  `ryft_core::operations::math::add`, including checks for the `DifferentiableOperation` and the
//  `TransposableOperation` implementations.
