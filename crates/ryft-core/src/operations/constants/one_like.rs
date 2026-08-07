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

/// Canonical operation name for [`OneLikeOperation`].
pub const ONE_LIKE_OPERATION_NAME: &str = "one_like";

/// [`Operation`] that has one exemplar input and that produces a single output that corresponds to the _one_ value
/// with the same [`Type`] as that input.
#[derive(Clone, Debug, Default)]
pub struct OneLikeOperation<T: Type>(PhantomData<fn() -> T>);

impl<T: Type> Copy for OneLikeOperation<T> {}

impl<T: Type> OneLikeOperation<T> {
    /// Constructs a one-like operation for the `T` type universe.
    #[inline]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Type> Display for OneLikeOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(ONE_LIKE_OPERATION_NAME)
    }
}

impl<T: Type> Operation for OneLikeOperation<T> {
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        ONE_LIKE_OPERATION_NAME
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

impl ElementwiseOperation for OneLikeOperation<crate::arrays::ArrayType> {
    #[inline]
    fn input_count(&self) -> usize {
        1
    }
}

impl<C: Domain<Value: OneLike>> InterpretableOperation<C> for OneLikeOperation<C::Type> {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].one_like()])
    }
}

impl<C: Context<Operation: From<OneLikeOperation<C::Type>>>> PartiallyEvaluatableOperation<C>
    for OneLikeOperation<C::Type>
{
}

impl_differentiable_elementwise_operation!(@constant<T> OneLikeOperation<T>);

/// Synthesizes a _one_ value from an exemplar. [`OneLike`] is the value-driven counterpart to [`One`](super::One).
/// It is what [`OneLikeOperation`] needs for its [`InterpretableOperation`] implementation.
pub trait OneLike {
    /// Returns a _one_ value with the same structure as `self`.
    fn one_like(&self) -> Self;
}

impl<V: Value<DispatchDomain: Context<Operation: From<OneLikeOperation<V::Type>>>>> OneLike for V {
    #[inline]
    fn one_like(&self) -> Self {
        self.dispatch_domain()
            .bind(OneLikeOperation::new(), Vec::new(), &[self.clone()])
            .expect("`one_like` operation failed")
            .remove(0)
    }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{ArrayType, DataType};
    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::contexts::EagerContext;
    use crate::differentiation::jacobian::JacobianDifferentiate;
    use crate::interpretation::InterpretableOperation;
    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::regions::EmptyRegionDriver;

    use super::*;

    #[test]
    fn test_one_like() {
        // Verify value-driven one synthesis across representative rank-zero array data-type families.
        for (input, expected) in [
            (Array::scalar(false), Array::scalar(true)),
            (Array::scalar(5i32), Array::scalar(1i32)),
            (Array::scalar(5u32), Array::scalar(1u32)),
            (Array::scalar(bf16::from_f32(5.0)), Array::scalar(bf16::ONE)),
            (Array::scalar(f16::from_f32(5.0)), Array::scalar(f16::ONE)),
            (Array::scalar(3.0f32), Array::scalar(1.0f32)),
            (Array::scalar(7.0f64), Array::scalar(1.0f64)),
        ] {
            assert_eq!(input.one_like(), expected);
        }

        // Verify the operation's identity, rendering, and eager interpretation.
        let operation = OneLikeOperation::<ArrayType>::new();
        assert_eq!(format!("{operation}"), ONE_LIKE_OPERATION_NAME);
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.5)],
            ),
            Ok(vec![Array::scalar(1.0)]),
        );

        // Verify the operation's textual form when it appears in a program.
        let mut builder = ProgramBuilder::<Array, OneLikeOperation<ArrayType>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(operation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = one_like %0
                in (%1)
            "}
            .trim_end(),
        );

        // Dense forward-mode differentiation batches the constant rule while constructing the identity Jacobian.
        let jacobian = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jacobian_forward(|input| Ok(input.clone() + input.one_like()), Array::scalar(2.0))
            .unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.value().to_f64s(), vec![1.0]);
    }
}
