use std::fmt::Display;
use std::marker::PhantomData;

use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_elementwise_operation};
use crate::operations::ElementwiseOperation;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{Operation, OperationFormatter, ProgramError, RegionInterface, Type, TypeError, Value};

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
        self.render(formatter, 0)
    }
}

impl<T: Type> Operation for TagOperation<T> {
    type Type = T;

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

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, TAG_OPERATION_NAME)?
            .bracketed(|operation| operation.field("key", &self.key))
    }
}

impl ElementwiseOperation for TagOperation<crate::arrays::ArrayType> {
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

impl_differentiable_elementwise_operation! {
    @linear<T>
    TagOperation<T>,
    rule = [@positive]
}

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

// TODO(eaplatanios): Add unit tests mirroring the structure and style of the tests in
//  `ryft_core::operations::math::add`, including checks for the `DifferentiableOperation` and the
//  `TransposableOperation` implementations.

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayType, DataType};
    use crate::contexts::EagerContext;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::programs::EmptyRegionDriver;

    use super::*;

    #[test]
    fn test_tag() {
        let operation = TagOperation::<ArrayType>::new("residual");
        assert_eq!(operation.key(), "residual");
        assert_eq!(operation.to_string(), "tag [key=residual]");
    }

    #[test]
    fn test_tag_type_inference() {
        check_operation_type_inference!(
            operation = TagOperation::new("residual"),
            cases = [{
                input_types = [ArrayType::scalar(DataType::F64)],
                output_types = [ArrayType::scalar(DataType::F64)],
            }],
        );
    }

    #[test]
    fn test_tag_interpretation() {
        assert_eq!(
            TagOperation::new("residual").interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(3.0)],
            ),
            Ok(vec![Array::scalar(3.0)]),
        );
    }

    #[test]
    fn test_tag_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = TagOperation::new("residual"),
            inputs = [Array::scalar(3.0)],
            expected = Array::scalar(3.0),
        );
    }

    #[test]
    fn test_tag_batching() {
        check_operation_batching!(
            @exact,
            operation = TagOperation::new("residual"),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![3.0, -2.0]))],
                outputs = [(@mapped(axis = 0), Array::vector(vec![3.0, -2.0]))],
            }],
        );
    }

    #[test]
    fn test_tag_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = TagOperation::new("residual"),
            cases = [{
                primals = [Array::scalar(3.0)],
                tangents = [Array::scalar(2.0)],
                primal_outputs = [Array::scalar(3.0)],
                tangent_outputs = [Array::scalar(2.0)],
            }],
        );
    }

    #[test]
    fn test_tag_transposition() {
        check_operation_transposition!(
            @exact,
            operation = TagOperation::new("residual"),
            cases = [{
                inputs = [(@linear(type = ArrayType::scalar(DataType::F64)))],
                output_cotangents = [Array::scalar(2.0)],
                input_cotangents = [Array::scalar(2.0)],
            }],
        );
    }
}
