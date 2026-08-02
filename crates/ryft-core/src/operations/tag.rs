use std::fmt::Display;
use std::marker::PhantomData;

use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_elementwise_operation};
use crate::operations::ElementwiseOperation;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::ProgramError;
use crate::programs::operations::Operation;
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError};
use crate::programs::values::Value;

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
pub struct TagOperation<T: Type> {
    /// Key tagging the operation's output [`Value`].
    key: String,

    /// Type universe in which this operation is valid.
    marker: PhantomData<fn() -> T>,
}

impl<T: Type> TagOperation<T> {
    /// Creates a new [`TagOperation`] with the provided key.
    #[inline]
    pub fn new<K: Into<String>>(key: K) -> Self {
        Self { key: key.into(), marker: PhantomData }
    }

    /// Returns the key carried by this [`TagOperation`].
    #[inline]
    pub fn key(&self) -> &str {
        self.key.as_str()
    }
}

impl<T: Type> Display for TagOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{TAG_OPERATION_NAME}[{}]", self.key)
    }
}

impl<T: Type> Operation<T> for TagOperation<T> {
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

impl ElementwiseOperation for TagOperation<crate::types::ArrayType> {
    #[inline]
    fn input_count(&self) -> usize {
        1
    }
}

impl<C: Domain> InterpretableOperation<C> for TagOperation<C::Type> {
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

impl<C: Context<Operation: From<TagOperation<C::Type>>>> PartiallyEvaluatableOperation<C> for TagOperation<C::Type> {}

/// Represents the ability to tag values in programs with keys. [`Tag`] stages a [`TagOperation`], which is effectively
/// an identity function carrying a string-valued key. The tag gets attached to traced values and survives forward-mode
/// differentiation (the [`DifferentiableOperation`] rule re-tags the primal value and passes the tangent value
/// through), so that it marks the instructions that define linearization residuals, which rematerialization
/// policies classify by key through the producing [`TagOperation`].
pub trait Tag: Sized {
    /// Returns this value unchanged while tagging it with `key`.
    fn tag(self, key: &str) -> Self;
}

impl<V: Value<DispatchDomain: Context<Operation: From<TagOperation<V::Type>>>>> Tag for V {
    #[inline]
    fn tag(self, key: &str) -> Self {
        self.dispatch_domain()
            .bind(TagOperation::new(key), Vec::new(), std::slice::from_ref(&self))
            .expect("`tag` operation failed")
            .remove(0)
    }
}

impl_differentiable_elementwise_operation! {
    @linear<T>
    TagOperation<T>,
    rule = [@positive]
}

// TODO(eaplatanios): Add unit tests mirroring the structure and style of the tests in
//  `ryft_core::operations::math::add`, including checks for the `DifferentiableOperation` and the
//  `TransposableOperation` implementations.
