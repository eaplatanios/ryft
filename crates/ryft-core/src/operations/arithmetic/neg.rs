use std::fmt::Display;

use crate::contexts::StagingContext;
use crate::macros::check_count;
use crate::operations::{ElementwiseOperation, InterpretableOperation, Operation};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{ProgramError, Value};
use crate::tracing::Tracer;
use crate::types::{ArrayType, DataType, Type, TypeError};

/// Canonical operation name for [`NegOperation`].
pub const NEG_OPERATION_NAME: &'static str = "neg";

/// [`Operation`] that negates one value while preserving its type metadata.
#[derive(Clone, Debug, Default)]
pub struct NegOperation;

impl Display for NegOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(NEG_OPERATION_NAME)
    }
}

impl Operation<DataType> for NegOperation {
    #[inline]
    fn name(&self) -> &'static str {
        NEG_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![input_types[0].clone()])
    }
}

impl Operation<ArrayType> for NegOperation {
    #[inline]
    fn name(&self) -> &'static str {
        NEG_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        ElementwiseOperation::infer_output_types(self, input_types)
    }
}

impl ElementwiseOperation for NegOperation {
    #[inline]
    fn input_count(&self) -> usize {
        1
    }
}

impl<T: Type, V: Clone + Value<T> + Neg> InterpretableOperation<T, V> for NegOperation
where
    Self: Operation<T>,
{
    #[inline]
    fn interpret(
        &self,
        _context: &<V as Value<T>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].neg()?])
    }
}

impl<T: Type, V: Value<T>, O> PartiallyEvaluatableOperation<T, V, O> for NegOperation {}

/// Value-level elementwise negation capability. [`Neg`] is the fallible Ryft counterpart to [`std::ops::Neg`]
/// that [`NegOperation`] interprets through, surfacing a [`ProgramError`] when something goes wrong, instead of
/// panicking. Value types additionally provide [`std::ops::Neg`] as ergonomic (albeit panicking) sugar layered on top
/// of this capability.
pub trait Neg: Sized {
    /// Negates `self`, returning a [`ProgramError`] if something goes wrong.
    fn neg(&self) -> Result<Self, ProgramError>;
}

impl<C: StagingContext<Operation: From<NegOperation>>> Neg for Tracer<C, C::Meta> {
    #[inline]
    fn neg(&self) -> Result<Self, ProgramError> {
        Ok(self.unary(NegOperation))
    }
}

impl<C: StagingContext<Operation: From<NegOperation>>> std::ops::Neg for Tracer<C, C::Meta> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self.unary(NegOperation)
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::scalars::Scalar;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tests::TestArray;
    use crate::types::{ArrayType, Layout, Shape, Size, StridedLayout};

    use super::*;

    #[test]
    fn test_neg() {
        let operation = NegOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<DataType>::name(&operation), NEG_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "NegOperation");
        assert_eq!(format!("{operation}"), NEG_OPERATION_NAME);
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[DataType::F32]), Ok(vec![DataType::F32]),);
        assert_eq!(
            InterpretableOperation::<DataType, Scalar>::interpret(
                &operation,
                &EagerContext::new(),
                &[Scalar::from(2.0)],
            ),
            Ok(vec![Scalar::from(-2.0)]),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, TestArray>::interpret(
                &operation,
                &EagerContext::new(),
                &[TestArray::scalar(2.0)]
            ),
            Ok(vec![TestArray::scalar(-2.0)]),
        );

        // Array type inference preserves shape, layout, and sharding metadata for its single input.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let input = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![3, 1])))
            .with_sharding(
                Sharding::with_manual_axes(
                    mesh,
                    vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["x"],
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(
            <NegOperation as Operation<ArrayType>>::infer_output_types(&operation, std::slice::from_ref(&input)),
            Ok(vec![input]),
        );

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<DataType, Scalar>::interpret(&operation, &EagerContext::new(), &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, TestArray>::interpret(&operation, &EagerContext::new(), &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<DataType, Scalar, NegOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, vec![input]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = neg %0
                in (%1)
            "}
            .trim_end(),
        );
    }
}
