use std::fmt::Display;

use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
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
#[derive(Copy, Clone, Debug, Default)]
pub struct OneLikeOperation;

impl Display for OneLikeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(ONE_LIKE_OPERATION_NAME)
    }
}

impl<T: Type> Operation<T> for OneLikeOperation {
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

impl ElementwiseOperation for OneLikeOperation {
    #[inline]
    fn input_count(&self) -> usize {
        1
    }
}

impl<C: Domain<Value: OneLike>> InterpretableOperation<C> for OneLikeOperation {
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

impl<C: Context<Operation: From<OneLikeOperation>>> PartiallyEvaluatableOperation<C> for OneLikeOperation {}

/// Synthesizes a _one_ value from an exemplar. [`OneLike`] is the value-driven counterpart to [`One`](super::One).
/// It is what [`OneLikeOperation`] needs for its [`InterpretableOperation`] implementation.
pub trait OneLike {
    /// Returns a _one_ value with the same structure as `self`.
    fn one_like(&self) -> Self;
}

impl<V: Value<DispatchDomain: Context<Operation: From<OneLikeOperation>>>> OneLike for V {
    #[inline]
    fn one_like(&self) -> Self {
        self.dispatch_domain()
            .bind(OneLikeOperation, Vec::new(), &[self.clone()])
            .expect("`one_like` operation failed")
            .remove(0)
    }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::operations::Operation;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::TypeError;
    use crate::types::{ArrayType, DataType};

    use super::*;

    #[test]
    fn test_one_like() {
        assert_eq!(Scalar::from(false).one_like(), Scalar::from(true));
        assert_eq!(Scalar::from(5i32).one_like(), Scalar::from(1i32));
        assert_eq!(Scalar::from(5u32).one_like(), Scalar::from(1u32));
        assert_eq!(Scalar::from(bf16::from_f32(5.0)).one_like(), Scalar::from(bf16::ONE));
        assert_eq!(Scalar::from(f16::from_f32(5.0)).one_like(), Scalar::from(f16::ONE));
        assert_eq!(Scalar::from(3.0f32).one_like(), Scalar::from(1.0f32));
        assert_eq!(Scalar::from(7.0f64).one_like(), Scalar::from(1.0f64));

        let operation = OneLikeOperation;
        assert_eq!(Operation::<DataType>::name(&operation), ONE_LIKE_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "OneLikeOperation");
        assert_eq!(format!("{operation}"), ONE_LIKE_OPERATION_NAME);
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64], &[]),
            Ok(vec![DataType::F64])
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(2.5)],
            ),
            Ok(vec![Scalar::from(1.0)]),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[ArrayType::scalar(DataType::F32)], &[]),
            Ok(vec![ArrayType::scalar(DataType::F32)]),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[], &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        let mut builder = ProgramBuilder::<Scalar, OneLikeOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = one_like %0
                in (%1)
            "}
            .trim_end(),
        );
    }
}
