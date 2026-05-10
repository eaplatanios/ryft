use std::fmt::Display;
use std::ops::Neg;

use crate::macros::check_count;
use crate::operations::{ElementwiseOperation, InterpretableOperation, Operation};
use crate::tracing::{Traceable, Tracer, TracingDomain, TracingError};
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

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

impl ElementwiseOperation for NegOperation {
    #[inline]
    fn name(&self) -> &'static str {
        NEG_OPERATION_NAME
    }

    #[inline]
    fn input_count(&self) -> usize {
        1
    }
}

impl<V: Clone + Typed<DataType> + Neg<Output = V>> InterpretableOperation<DataType, V> for NegOperation {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![-inputs[0].clone()])
    }
}

impl<V: Clone + Typed<ArrayType> + Neg<Output = V>> InterpretableOperation<ArrayType, V> for NegOperation {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![-inputs[0].clone()])
    }
}

/// Trait that represents [`Operation`] carrier types that support/include [`NegOperation`]. Backend-owned closed
/// [`Operation`] carrier types implement this trait so that generic transform code can stage [`NegOperation`] without
/// knowing which carrier is in use.
pub trait SupportsNeg<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of [`NegOperation`].
    fn neg_operation() -> Self;
}

impl<'domain, D: TracingDomain<OperationCarrier: SupportsNeg<D::Type, D::Value>>> Neg for Tracer<'domain, D> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self.unary(D::OperationCarrier::neg_operation())
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing::{ProgramBuilder, TracingError};
    use crate::types::{Layout, Shape, Size, StridedLayout};

    use super::*;

    #[test]
    fn test_neg() {
        let operation = NegOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<DataType>::name(&operation), NEG_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "NegOperation");
        assert_eq!(format!("{operation}"), NEG_OPERATION_NAME);
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[DataType::F32]), Ok(vec![DataType::F32]),);
        assert_eq!(InterpretableOperation::<DataType, f64>::interpret(&operation, &[2.0]), Ok(vec![-2.0]));
        assert_eq!(InterpretableOperation::<ArrayType, f64>::interpret(&operation, &[2.0]), Ok(vec![-2.0]));

        // Array type inference preserves shape, layout, and sharding metadata for its single input.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let input = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Size::Static(2), Size::Static(3)]),
            Some(Layout::Strided(StridedLayout::new(vec![3, 1]))),
            Some(
                Sharding::with_manual_axes(
                    mesh,
                    vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["x"],
                )
                .unwrap(),
            ),
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
            InterpretableOperation::<DataType, f64>::interpret(&operation, &[]),
            Err(TracingError::InvalidInputCount { expected: 1, got: 0 }),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, f64>::interpret(&operation, &[]),
            Err(TracingError::InvalidInputCount { expected: 1, got: 0 }),
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<DataType, f64, NegOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, vec![input]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![output], Placeholder, Placeholder).unwrap();
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
