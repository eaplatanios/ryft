use std::fmt::Display;

use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::{ElementwiseOperation, Operation};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{ProgramError, Value};
use crate::regions::RegionInterface;
use crate::types::{ArrayType, DataType, Type, TypeError};

/// Canonical operation name for [`Atan2Operation`].
pub const ATAN2_OPERATION_NAME: &str = "atan2";

/// [`Operation`] that computes the elementwise two-argument arc tangent of its operands (i.e., `(y, x) ↦ atan2(y, x)`,
/// the angle of the point `(x, y)` in the correct quadrant) while preserving their type metadata. This is the analogue
/// of [JAX's `lax.atan2`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.atan2.html). The two operands must have
/// identical types, and complex operands are rejected (the angle of a complex value is instead the composition
/// `atan2(imaginary(z), real(z))`).
#[derive(Clone, Debug, Default)]
pub struct Atan2Operation;

impl Display for Atan2Operation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(ATAN2_OPERATION_NAME)
    }
}

impl Operation<DataType> for Atan2Operation {
    #[inline]
    fn name(&self) -> &'static str {
        ATAN2_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[DataType],
        _region_interfaces: &[RegionInterface<DataType>],
    ) -> Result<Vec<DataType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        if input_types[0] != input_types[1] {
            return Err(TypeError {
                message: format!(
                    "'{ATAN2_OPERATION_NAME}' requires identical operand types but got {} and {}",
                    input_types[0], input_types[1],
                ),
            });
        }
        if input_types[0].is_complex() {
            return Err(TypeError {
                message: format!("'{ATAN2_OPERATION_NAME}' requires real operands but got {}", input_types[0],),
            });
        }
        Ok(vec![input_types[0].clone()])
    }
}

impl Operation<ArrayType> for Atan2Operation {
    #[inline]
    fn name(&self) -> &'static str {
        ATAN2_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        if input_types[0].data_type().is_complex() || input_types[1].data_type().is_complex() {
            return Err(TypeError {
                message: format!(
                    "'{ATAN2_OPERATION_NAME}' requires real operands but got {} and {}",
                    input_types[0], input_types[1],
                ),
            });
        }
        ElementwiseOperation::infer_output_types(self, input_types)
    }
}

impl ElementwiseOperation for Atan2Operation {
    #[inline]
    fn input_count(&self) -> usize {
        2
    }
}

impl<C: Domain<Value: Atan2>> InterpretableOperation<C> for Atan2Operation
where
    Self: Operation<C::Type>,
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].atan2(&inputs[1])?])
    }
}

impl<C: Context> PartiallyEvaluatableOperation<C> for Atan2Operation where C::Operation: From<Atan2Operation> {}

/// Value-level elementwise two-argument arc-tangent capability, computing `atan2(self, x)`. [`Atan2`] fills the
/// same role for [`Atan2Operation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
pub trait Atan2: Sized {
    /// Computes the elementwise two-argument arc tangent `atan2(self, x)` (i.e., with this value as the `y`
    /// coordinate), returning a [`ProgramError`] if something goes wrong (e.g., when the operands are not
    /// floating-point valued).
    fn atan2(&self, x: &Self) -> Result<Self, ProgramError>;
}

impl<V: Value<DispatchDomain: Context<Operation: From<Atan2Operation>>>> Atan2 for V {
    #[inline]
    fn atan2(&self, x: &Self) -> Result<Self, ProgramError> {
        Ok(self.dispatch_domain().bind(Atan2Operation, Vec::new(), &[self.clone(), x.clone()])?.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::regions::EmptyRegionDriver;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tests::TestArray;
    use crate::types::{Layout, Shape, Size, StridedLayout};

    use super::*;

    #[test]
    fn test_atan2() {
        assert_eq!(Scalar::from(0.5f32).atan2(&Scalar::from(-0.25f32)).unwrap(), 0.5f32.atan2(-0.25f32));
        assert_eq!(Scalar::from(0.5f64).atan2(&Scalar::from(-0.25f64)).unwrap(), 0.5f64.atan2(-0.25f64));
        assert_eq!(
            Scalar::from(bf16::from_f32(0.5)).atan2(&Scalar::from(bf16::from_f32(-0.25))).unwrap(),
            bf16::from_f32(0.5f32.atan2(-0.25f32)),
        );
        assert_eq!(
            Scalar::from(f16::from_f32(0.5)).atan2(&Scalar::from(f16::from_f32(-0.25))).unwrap(),
            f16::from_f32(0.5f32.atan2(-0.25f32)),
        );

        let operation = Atan2Operation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<DataType>::name(&operation), ATAN2_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "Atan2Operation");
        assert_eq!(format!("{operation}"), ATAN2_OPERATION_NAME);
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F32, DataType::F32], &[]),
            Ok(vec![DataType::F32]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(0.5), Scalar::from(-0.25)],
            ),
            Ok(vec![Scalar::from(0.5f64.atan2(-0.25f64))]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<TestArray>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[TestArray::scalar(0.5), TestArray::scalar(-0.25)],
            ),
            Ok(vec![TestArray::scalar(0.5f64.atan2(-0.25f64))]),
        );

        // Array type inference preserves shape, layout, and sharding metadata for its identical inputs.
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
            <Atan2Operation as Operation<ArrayType>>::infer_output_types(
                &operation,
                &[input.clone(), input.clone()],
                &[],
            ),
            Ok(vec![input]),
        );

        // Mismatched and complex operand types report precise inference errors.
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F32, DataType::F64], &[]),
            Err(TypeError { message: "'atan2' requires identical operand types but got f32 and f64".to_string() }),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::C64, DataType::C64], &[]),
            Err(TypeError { message: "'atan2' requires real operands but got c64".to_string() }),
        );
        let complex_type = ArrayType::new(DataType::C64, Shape::new(vec![Size::Static(2)]));
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[complex_type.clone(), complex_type], &[]),
            Err(TypeError { message: "'atan2' requires real operands but got c64[2] and c64[2]".to_string() }),
        );

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[], &[]),
            Err(TypeError { message: "expected 2 inputs but got 0".to_string() }),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[], &[]),
            Err(TypeError { message: "expected 2 inputs but got 0".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<TestArray>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(0.5f32), Scalar::from(-0.25f64)],
            ),
            Err(ProgramError::Type(TypeError {
                message: "cannot compute the arc tangent of scalars of data types f32 and f64".to_string(),
            })),
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<Scalar, Atan2Operation>::new();
        let y = builder.add_input(DataType::F64);
        let x = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, vec![y, x], Vec::new()).unwrap()[0];
        let program = builder
            .build::<(Scalar, Scalar), Scalar>(vec![output], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = atan2 %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }
}
