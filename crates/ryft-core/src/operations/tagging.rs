//! Operations that attach string keys to values so that program transforms can recognize them later. Tags are identity
//! functions on their data: interpretation, batching, partial evaluation, and backend lowering all pass the input
//! through unchanged, and the key lives only in the staged instruction. This module provides the following:
//!
//!   - The [`Tag`] value capability, whose [`tag`](Tag::tag) returns its input unchanged while marking the producing
//!     instruction with a key.
//!   - The [`TagOperation`] that the capability stages, whose [`key`](TagOperation::key) is what consumers match on.
//!     Forward-mode differentiation re-tags the primal value and passes the tangent through, so a tag placed in the
//!     body of a function is still present on the instructions that define the residuals of its linearization.
//!
//! The main consumer of tags is key-based rematerialization. Policies such as
//! [`SaveOnlyTheseNames`](crate::SaveOnlyTheseNames), [`SaveAnyNamesButThese`](crate::SaveAnyNamesButThese), and
//! [`SaveAndOffloadOnlyTheseNames`](crate::SaveAndOffloadOnlyTheseNames) decide whether to save, offload, or recompute
//! each residual of a [`rematerialize`](crate::rematerialize)d function by looking at the key of the [`TagOperation`]
//! that produced it, which mirrors the role of [`checkpoint_name` in JAX](
//! https://docs.jax.dev/en/latest/_autosummary/jax.ad_checkpoint.checkpoint_name.html).
//! Tags carry no other semantics, so they are safe to leave in production programs.
//!
//! # Example
//!
//! Tagging a traced value stages a `tag` instruction that carries the key and forwards its input:
//!
//! ```rust
//! # use indoc::indoc;
//! # use ryft_core::{Array, ArrayOperation, ArrayType, DataType, Mul, ProgramError, Tag, Trace, TracingContext};
//! # fn main() -> Result<(), ProgramError> {
//! let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
//!     |input| input.clone().mul(&input)?.tag("squared"),
//!     ArrayType::scalar(DataType::F64),
//! )?;
//! assert_eq!(
//!     program.to_string(),
//!     indoc! {"
//!         lambda %0:f64[] .
//!         let %1:f64[] = mul %0 %0
//!             %2:f64[] = tag [key=squared] %1
//!         in (%2)"},
//! );
//! assert_eq!(program.interpret(Array::scalar(3.0f64)?)?, Array::scalar(9.0f64)?);
//! # Ok(())
//! # }
//! ```

use std::fmt::Display;
use std::marker::PhantomData;

use crate::arrays::{Array, ArrayType};
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
    /// Refer to the documentation of [`key`](Self::key) for more information.
    key: String,

    /// [`PhantomData`] marker tying this [`Operation`] to the [`Type`] universe in which it is valid.
    marker: PhantomData<fn() -> T>,
}

impl<T: Type> TagOperation<T> {
    /// Creates a new [`TagOperation`] with the provided key.
    #[inline]
    pub fn new<K: Into<String>>(key: K) -> Self {
        Self { key: key.into(), marker: PhantomData }
    }

    /// Returns the key that tags the output [`Value`] of this [`TagOperation`].
    #[inline]
    pub fn key(&self) -> &str {
        self.key.as_str()
    }
}

impl<T: Type> Display for TagOperation<T> {
    #[inline]
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

impl ElementwiseOperation for TagOperation<ArrayType> {
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
    rule = [@positive],
}

/// Represents the ability to tag values in programs with keys. [`Tag`] stages a [`TagOperation`], which is effectively
/// an identity function carrying a string-valued key. The tag gets attached to traced values and survives forward-mode
/// differentiation rule (i.e., the [`DifferentiableOperation`](crate::DifferentiableOperation) implementation re-tags
/// the primal value and passes the tangent value through), so that it marks the instructions that define linearization
/// residuals, which rematerialization policies classify by key through the producing [`TagOperation`].
pub trait Tag: Sized {
    /// Returns this value unchanged while tagging it with `key`, and a [`ProgramError`] if the value's context fails
    /// to bind the operation. Tracing contexts never fail here, because the value is always native to its own context,
    /// but contexts that execute operations eagerly (e.g., a backend running operation by operation) may.
    fn tag(self, key: &str) -> Result<Self, ProgramError>;
}

impl Tag for Array {
    #[inline]
    fn tag(self, _key: &str) -> Result<Self, ProgramError> {
        // Tagging is metadata for staged programs only, so a concrete array carries itself through unchanged.
        Ok(self)
    }
}

impl<V: Value<DispatchDomain: Context<Operation: From<TagOperation<V::Type>>>>> Tag for V {
    #[inline]
    fn tag(self, key: &str) -> Result<Self, ProgramError> {
        let mut outputs =
            self.dispatch_domain().bind(TagOperation::new(key), Vec::new(), std::slice::from_ref(&self))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayOperation, DataType};
    use crate::contexts::EagerContext;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::programs::{EffectClasses, EmptyRegionDriver};
    use crate::tracing::{DomainTracer, Trace};

    use super::*;

    #[test]
    fn test_tag() {
        let operation = TagOperation::<ArrayType>::new("residual");

        // Operation identity, accessors, and rendering.
        assert_eq!(operation.name(), TAG_OPERATION_NAME);
        assert_eq!(operation.key(), "residual");
        assert_eq!(operation.input_count(), 1);
        assert_eq!(operation.to_string(), "tag [key=residual]");
    }

    #[test]
    fn test_tag_type_inference() {
        check_operation_type_inference!(
            operation = TagOperation::new("residual"),
            cases = [{
                input_types = [ArrayType::scalar(DataType::F64)],
                output_types = [ArrayType::scalar(DataType::F64)],
            }, {
                input_types = [ArrayType::new_static(DataType::I32, [2, 3])],
                output_types = [ArrayType::new_static(DataType::I32, [2, 3])],
            }, {
                input_types = [],
                error = "expected 1 input but got 0",
            }, {
                input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
                error = "expected 1 input but got 2",
            }],
        );
    }

    #[test]
    fn test_tag_interpretation() {
        let context = EagerContext::<Array>::new();
        let input = Array::scalar(3.0).unwrap();
        let operation = TagOperation::new("residual");

        // Interpretation passes the input through unchanged and validates the input count.
        assert_eq!(
            operation.interpret(&context, &EmptyRegionDriver, std::slice::from_ref(&input)),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            operation.interpret(&context, &EmptyRegionDriver, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
        assert_eq!(
            operation.interpret(&context, &EmptyRegionDriver, &[input.clone(), input]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 2 }),
        );
    }

    #[test]
    fn test_tag_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = TagOperation::new("residual"),
            inputs = [Array::scalar(3.0).unwrap()],
            expected = Array::scalar(3.0).unwrap(),
        );
    }

    #[test]
    fn test_tag_batching() {
        check_operation_batching!(
            @exact,
            operation = TagOperation::new("residual"),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![3.0, -2.0]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![3.0, -2.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_tag_differentiation() {
        // The JVP re-tags the primal and passes the tangent through untagged, so the key marks the instruction that
        // defines the linearization residual rather than the tangent program.
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = TagOperation::new("residual"),
            cases = [{
                primals = [Array::scalar(3.0).unwrap()],
                tangents = [Array::scalar(2.0).unwrap()],
                primal_outputs = [Array::scalar(3.0).unwrap()],
                tangent_outputs = [Array::scalar(2.0).unwrap()],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = tag [key=residual] %0
                    in (%2, %1)
                "},
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
                output_cotangents = [Array::scalar(2.0).unwrap()],
                input_cotangents = [Array::scalar(2.0).unwrap()],
            }],
        );
    }

    #[test]
    fn test_array_tag() {
        // Tags are metadata for staged programs, so an eager array is returned unchanged.
        let input = Array::vector(vec![1.0_f32, 2.0]).unwrap();
        assert_eq!(input.clone().tag("residual").unwrap(), input);
    }

    #[test]
    fn test_tag_staging() {
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |input: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok(input.tag("residual")?),
            ArrayType::scalar(DataType::F64),
        )
        .unwrap();
        let program = program.to_flat_program();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = tag [key=residual] %0
                in (%1)"},
        );

        // Tags declare no effects, so unlike prints they do not pin an otherwise dead value.
        assert_eq!(program.effects().classes(), EffectClasses::NONE);
    }
}
