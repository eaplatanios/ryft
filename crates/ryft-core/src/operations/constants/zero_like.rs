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

impl<T: Type> Operation<T> for ZeroLikeOperation<T> {
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

impl ElementwiseOperation for ZeroLikeOperation<crate::types::ArrayType> {
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

    use crate::backends::arrays::Array;
    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::differentiation::jacobian::jacobian_reverse;
    use crate::interpretation::InterpretableOperation;
    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::operations::Operation;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::types::DataType;

    use super::*;

    #[test]
    fn test_zero_like() {
        // Verify value-driven zero synthesis across representative scalar data-type families.
        for (input, expected) in [
            (Scalar::from(false), Scalar::from(false)),
            (Scalar::from(5i32), Scalar::from(0i32)),
            (Scalar::from(5u32), Scalar::from(0u32)),
            (Scalar::from(bf16::from_f32(5.0)), Scalar::from(bf16::ZERO)),
            (Scalar::from(f16::from_f32(5.0)), Scalar::from(f16::ZERO)),
            (Scalar::from(3.0f32), Scalar::from(0.0f32)),
            (Scalar::from(7.0f64), Scalar::from(0.0f64)),
        ] {
            assert_eq!(input.zero_like(), expected);
        }

        // Verify the operation's identity, zero metadata, rendering, and eager interpretation.
        let operation = ZeroLikeOperation::<DataType>::new();
        assert!(Operation::<DataType>::is_zero(&operation, 0));
        assert!(!Operation::<DataType>::is_zero(&operation, 1));
        assert_eq!(format!("{operation}"), ZERO_LIKE_OPERATION_NAME);
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(2.5)],
            ),
            Ok(vec![Scalar::from(0.0)]),
        );

        // Verify the operation's textual form when it appears in a program.
        let mut builder = ProgramBuilder::<Scalar, ZeroLikeOperation<DataType>>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                    lambda %0:f64 .
                    let %1:f64 = zero_like %0
                    in (%1)
                "}
            .trim_end(),
        );

        // Dense reverse-mode differentiation batches the constant rule while constructing the identity Jacobian.
        let jacobian = jacobian_reverse(|input| Ok(input.clone() + input.zero_like()), Array::scalar(2.0)).unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.value().to_f64s(), vec![1.0]);
    }
}
