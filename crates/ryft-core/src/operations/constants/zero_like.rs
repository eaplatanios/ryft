use std::fmt::Display;
use std::marker::PhantomData;

use crate::arrays::{Array, ArrayElement, ArrayType, DataType, dispatch_on_array_element_type};
use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_elementwise_operation};
use crate::operations::ElementwiseOperation;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{Operation, ProgramError, RegionInterface, Type, TypeError, Typed, Value};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`ZeroLikeOperation`].
pub const ZERO_LIKE_OPERATION_NAME: &str = "zero_like";

/// [`Operation`] that has one exemplar input and that produces a single output that corresponds to the _zero_ value
/// with the same [`Type`] as that input.
#[derive(Clone, Debug, Default)]
pub struct ZeroLikeOperation<T: Type>(PhantomData<fn() -> T>);

impl<T: Type> Copy for ZeroLikeOperation<T> {}

impl<T: Type> ZeroLikeOperation<T> {
    /// Constructs a zero-like operation for the `T` type universe.
    #[inline]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Type> Display for ZeroLikeOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(ZERO_LIKE_OPERATION_NAME)
    }
}

impl<T: Type> Operation for ZeroLikeOperation<T> {
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        ZERO_LIKE_OPERATION_NAME
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
    fn is_zero(&self, output_index: usize) -> bool {
        output_index == 0
    }
}

impl ElementwiseOperation for ZeroLikeOperation<ArrayType> {
    #[inline]
    fn input_count(&self) -> usize {
        1
    }
}

impl<C: Domain<Value: ZeroLike>> InterpretableOperation<C> for ZeroLikeOperation<C::Type> {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].zero_like()])
    }
}

impl<C: Context<Operation: From<ZeroLikeOperation<C::Type>>>> PartiallyEvaluatableOperation<C>
    for ZeroLikeOperation<C::Type>
{
}

impl_differentiable_elementwise_operation!(@constant<T> ZeroLikeOperation<T>);

/// Synthesizes a _zero_ value from an exemplar. [`ZeroLike`] is the value-driven counterpart to [`Zero`](super::Zero).
/// It is what [`ZeroLikeOperation`] needs for its [`InterpretableOperation`] implementation.
pub trait ZeroLike {
    /// Returns a _zero_ value with the same structure as `self`.
    fn zero_like(&self) -> Self;
}

impl ZeroLike for Array {
    fn zero_like(&self) -> Self {
        match self.r#type().data_type() {
            DataType::Token | DataType::Zero | DataType::F8E8M0FNU => self.clone(),
            data_type => dispatch_on_array_element_type!(data_type, |Element| {
                let element = Element::from_unsigned(0).unwrap();
                Self::from_fn_elements(self.r#type().into_owned(), |_| Ok(element)).unwrap()
            }),
        }
    }
}

impl<V: Value<DispatchDomain: Context<Operation: From<ZeroLikeOperation<V::Type>>>>> ZeroLike for V {
    #[inline]
    fn zero_like(&self) -> Self {
        self.dispatch_domain()
            .bind(ZeroLikeOperation::new(), Vec::new(), &[self.clone()])
            .expect("`zero_like` operation failed")
            .remove(0)
    }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayBatch, ArrayBatching, ArrayOperation, ArrayType, DataType, f8e8m0fnu};
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::differentiate_at;
    use crate::interpretation::InterpretableOperation;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationValue, PartialTracer};
    use crate::programs::{EmptyRegionDriver, Operation, ProgramBuilder};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_zero_like() {
        // Verify the operation's identity, zero metadata, rendering, and eager interpretation.
        let operation = ZeroLikeOperation::<ArrayType>::new();
        assert!(operation.is_zero(0));
        assert!(!operation.is_zero(1));
        assert_eq!(format!("{operation}"), ZERO_LIKE_OPERATION_NAME);
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.5)],
            ),
            Ok(vec![Array::scalar(0.0)]),
        );

        // Verify the operation's textual form when it appears in a program.
        let mut builder = ProgramBuilder::<Array, ZeroLikeOperation<ArrayType>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(operation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                    lambda %0:f64[] .
                    let %1:f64[] = zero_like %0
                    in (%1)
                "}
            .trim_end(),
        );
    }

    #[test]
    fn test_array_zero_like() {
        // Verify value-driven zero synthesis across representative rank-zero array data-type families.
        for (input, expected) in [
            (Array::scalar(false), Array::scalar(false)),
            (Array::scalar(5i32), Array::scalar(0i32)),
            (Array::scalar(5u32), Array::scalar(0u32)),
            (Array::scalar(bf16::from_f32(5.0)), Array::scalar(bf16::ZERO)),
            (Array::scalar(f16::from_f32(5.0)), Array::scalar(f16::ZERO)),
            (Array::scalar(3.0f32), Array::scalar(0.0f32)),
            (Array::scalar(7.0f64), Array::scalar(0.0f64)),
        ] {
            assert_eq!(input.zero_like(), expected);
        }

        let input = Array::vector(vec![1.5f32, -2.5]);
        let output = input.zero_like();
        assert_eq!(output.elements::<f32>(), Ok(vec![0.0, 0.0]));
        assert_eq!(output.r#type().into_owned(), ArrayType::new_static(DataType::F32, [2]));

        // `f8e8m0fnu` cannot represent zero, so zero-like retains the exemplar exactly.
        let input = Array::from_elements(
            ArrayType::new_static(DataType::F8E8M0FNU, [2]),
            &[f8e8m0fnu::from_bits(0x7e), f8e8m0fnu::from_bits(0x80)],
        )
        .unwrap();
        assert_eq!(input.zero_like(), input);
    }

    #[test]
    fn test_staging_zero_like() {
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = context.input(ArrayType::new_static(DataType::F32, [2]));
        let output = input.zero_like();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![output.atom_id().unwrap()], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = zero_like %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_partial_evaluation_zero_like() {
        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let input = PartialTracer::new(context, PartialEvaluationValue::known(Array::vector(vec![1.5f32, -2.5])));
        let output = input.zero_like();
        assert_eq!(output.value().unwrap().as_known(), Some(&Array::vector(vec![0.0f32, 0.0])));
    }

    #[test]
    fn test_batching_zero_like() {
        let context = BatchingContext::<_, ArrayBatching>::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 4);
        let input = BatchingTracer::new(
            context,
            ArrayBatch::new(Array::vector(vec![1.5f32, -2.5]), BatchAxis::replicated()).unwrap(),
        );
        let output = input.zero_like();
        assert_eq!(output.batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(output.batch().value(), &Array::vector(vec![0.0f32, 0.0]));
    }

    #[test]
    fn test_differentiation_zero_like() {
        // Dense reverse-mode differentiation batches the constant rule while constructing the identity Jacobian.
        let jacobian = differentiate_at(Array::scalar(2.0))
            .jacobian_reverse(|input| Ok(input.clone() + input.zero_like()))
            .unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.value().to_f64s(), vec![1.0]);
    }
}
