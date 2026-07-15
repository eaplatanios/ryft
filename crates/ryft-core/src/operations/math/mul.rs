use std::collections::BTreeSet;
use std::fmt::Display;

use crate::broadcasting::Broadcastable;
use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, define_tracer_operator};
use crate::operations::{ElementwiseOperation, Operation};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{ProgramError, Value};
use crate::regions::RegionInterface;
use crate::sharding::Sharding;
use crate::types::{ArrayType, DataType, TypeError};

/// Canonical operation name for [`MulOperation`].
pub const MUL_OPERATION_NAME: &str = "mul";

/// [`Operation`] that multiplies two values and typically supports broadcasting semantics for arrays.
#[derive(Clone, Debug, Default)]
pub struct MulOperation;

impl Display for MulOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(MUL_OPERATION_NAME)
    }
}

impl Operation<DataType> for MulOperation {
    #[inline]
    fn name(&self) -> &'static str {
        MUL_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[DataType],
        _region_interfaces: &[RegionInterface<DataType>],
    ) -> Result<Vec<DataType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        input_types[0].broadcast(&input_types[1]).map(|output| vec![output]).map_err(|_| TypeError {
            message: format!("'{MUL_OPERATION_NAME}' input types are not broadcast-compatible"),
        })
    }
}

impl Operation<ArrayType> for MulOperation {
    #[inline]
    fn name(&self) -> &'static str {
        MUL_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        ElementwiseOperation::infer_output_types(self, input_types)
    }
}

impl ElementwiseOperation for MulOperation {
    #[inline]
    fn input_count(&self) -> usize {
        2
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        // Multiplication is bilinear, so its output sharding combines the operands' unreduced/reduced reduction
        // state by the bilinear rule (this is also what JAX does with its `_mul_ur_rule` implementation; refer to
        // https://blog.ezyang.com/2026/01/jax-sharding-type-system/ for more information on that) rather than the
        // congruent rule that generic elementwise broadcasting applies. The reduction state is combined independently
        // of the per-dimension placement, so the placement is broadcast with that state stripped (otherwise the shared
        // broadcast would reject a legitimate unreduced/reduced pairing) and the recomputed state is reattached
        // afterward.
        check_count!("input", input_types, 2, TypeError);
        let stripped = [input_types[0].without_reduction_axes(), input_types[1].without_reduction_axes()];
        let output = self.broadcast_output_type(&stripped)?;
        let left_unreduced = input_types[0].unreduced_axes();
        let left_reduced = input_types[0].reduced_axes();
        let right_unreduced = input_types[1].unreduced_axes();
        let right_reduced = input_types[1].reduced_axes();

        // An operand unreduced over some axes is a partial sum still awaiting an all-reduce over them. The product of
        // two partial sums is not a partial sum, so at most one operand may be unreduced. The other must then be
        // reduced over exactly those axes, and the product stays unreduced over them (its matching reduced marker is
        // consumed when the reduced set is computed below).
        let output_unreduced = match (left_unreduced.is_empty(), right_unreduced.is_empty()) {
            (false, false) => {
                return Err(TypeError {
                    message: format!("'{MUL_OPERATION_NAME}' cannot multiply two operands that are both unreduced"),
                });
            }
            (false, true) => {
                if left_unreduced != right_reduced {
                    return Err(TypeError {
                        message: format!(
                            "'{MUL_OPERATION_NAME}' requires the second operand to be reduced over the axes \
                             the first is unreduced over",
                        ),
                    });
                }
                left_unreduced.clone()
            }
            (true, false) => {
                if right_unreduced != left_reduced {
                    return Err(TypeError {
                        message: format!(
                            "'{MUL_OPERATION_NAME}' requires the first operand to be reduced over the axes \
                             the second is unreduced over",
                        ),
                    });
                }
                right_unreduced.clone()
            }
            (true, true) => BTreeSet::new(),
        };

        // Plain reduced axes must agree when both operands carry them (either operand may leave the set unset), and any
        // axis that just became unreduced is dropped from the reduced set since the product now tracks it as unreduced.
        let mut output_reduced = if left_reduced.is_empty() {
            right_reduced.clone()
        } else if right_reduced.is_empty() || left_reduced == right_reduced {
            left_reduced.clone()
        } else {
            return Err(TypeError {
                message: format!("'{MUL_OPERATION_NAME}' operands must be reduced over the same axes"),
            });
        };
        output_reduced.retain(|axis| !output_unreduced.contains(axis));

        // A non-empty result reduction state means some operand was sharded, so the broadcast output (already stripped
        // of reduction axes) carries a sharding onto which the recomputed state is reattached; otherwise it is already
        // correct as is.
        if output_unreduced.is_empty() && output_reduced.is_empty() {
            return Ok(vec![output]);
        }
        let sharding = output.sharding().expect("bilinear reduction state implies a sharded output");
        let rebuilt = Sharding::with_manual_axes(
            sharding.mesh().clone(),
            sharding.dimensions().to_vec(),
            output_unreduced,
            output_reduced,
            sharding.varying_manual_axes().clone(),
        )
        .map_err(|error| TypeError { message: error.to_string() })?;
        Ok(vec![output.with_sharding(rebuilt).map_err(|error| TypeError { message: error.to_string() })?])
    }
}

impl<C: Domain<Value: Mul>> InterpretableOperation<C> for MulOperation
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
        Ok(vec![inputs[0].mul(&inputs[1])?])
    }
}

impl<C: Context> PartiallyEvaluatableOperation<C> for MulOperation where C::Operation: From<MulOperation> {}

/// Value-level elementwise multiplication capability. [`Mul`] is the fallible Ryft counterpart to [`std::ops::Mul`]
/// that [`MulOperation`] interprets through, surfacing a [`ProgramError`] when something goes wrong, instead of
/// panicking. Value types additionally provide [`std::ops::Mul`] as ergonomic (albeit panicking) sugar layered on top
/// of this capability.
pub trait Mul: Sized {
    /// Multiplies `self` by `rhs`, returning a [`ProgramError`] if something goes wrong.
    fn mul(&self, rhs: &Self) -> Result<Self, ProgramError>;
}

impl<V: Value<DispatchDomain: Context<Operation: From<MulOperation>>>> Mul for V {
    #[inline]
    fn mul(&self, rhs: &Self) -> Result<Self, ProgramError> {
        Ok(self.dispatch_domain().bind(MulOperation, Vec::new(), &[self.clone(), rhs.clone()])?.remove(0))
    }
}

define_tracer_operator!(@binary std::ops::Mul, mul, MulOperation, "`mul` operation failed");

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

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
    fn test_mul() {
        let operation = MulOperation;

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<DataType>::name(&operation), MUL_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "MulOperation");
        assert_eq!(format!("{operation}"), MUL_OPERATION_NAME);
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F32, DataType::F64], &[]),
            Ok(vec![DataType::F64]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(2.0), Scalar::from(3.5)],
            ),
            Ok(vec![Scalar::from(7.0)]),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<TestArray>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[TestArray::scalar(2.0), TestArray::scalar(3.5)],
            ),
            Ok(vec![TestArray::scalar(7.0)]),
        );

        // Array type inference broadcasts shapes and promotes data types.
        let output = <MulOperation as Operation<ArrayType>>::infer_output_types(
            &operation,
            &[
                ArrayType::scalar(DataType::F32),
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)])),
            ],
            &[],
        )
        .unwrap();
        assert_eq!(output, vec![ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]))]);

        // Array type inference drops layout metadata when inputs disagree.
        let output = <MulOperation as Operation<ArrayType>>::infer_output_types(
            &operation,
            &[
                ArrayType::new(DataType::F32, Shape::scalar()).with_layout(Layout::Strided(StridedLayout::new(vec![]))),
                ArrayType::scalar(DataType::F32),
            ],
            &[],
        )
        .unwrap();
        assert_eq!(output, vec![ArrayType::scalar(DataType::F32)]);

        // Array type inference tolerates compatible inputs that only disagree on varying manual axes.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let left = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(
                Sharding::with_manual_axes(
                    mesh.clone(),
                    vec![ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["x"],
                )
                .unwrap(),
            )
            .unwrap();
        let right = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(
                Sharding::with_manual_axes(
                    mesh,
                    vec![ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["y"],
                )
                .unwrap(),
            )
            .unwrap();
        let output =
            <MulOperation as Operation<ArrayType>>::infer_output_types(&operation, &[left, right], &[]).unwrap();
        assert_eq!(
            output[0].sharding().as_ref().unwrap().varying_manual_axes(),
            &BTreeSet::from(["x".to_string(), "y".to_string()]),
        );

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64], &[]),
            Err(TypeError { message: "expected 2 inputs but got 1".to_string() }),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[ArrayType::scalar(DataType::F64)], &[]),
            Err(TypeError { message: "expected 2 inputs but got 1".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Scalar::from(2.0)],
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
        assert_eq!(
            InterpretableOperation::<EagerContext<TestArray>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[TestArray::scalar(2.0)]
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F8E3M4, DataType::F32], &[]),
            Err(TypeError { message: format!("'{MUL_OPERATION_NAME}' input types are not broadcast-compatible") }),
        );
        let error = <MulOperation as Operation<ArrayType>>::infer_output_types(
            &operation,
            &[
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)])),
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)])),
            ],
            &[],
        )
        .unwrap_err();
        assert_eq!(
            error,
            TypeError { message: format!("'{MUL_OPERATION_NAME}' input types are not broadcast-compatible") },
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<Scalar, MulOperation>::new();
        let left = builder.add_input(DataType::F64);
        let right = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, Vec::new(), vec![left, right]).unwrap()[0];
        let program = builder
            .build::<(Scalar, Scalar), Scalar>(vec![output], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = mul %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_mul_combines_unreduced_and_reduced_operands_bilinearly() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let vector_type = || ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]));
        let unreduced = |axis: &str| {
            vector_type()
                .with_sharding(
                    Sharding::with_unreduced_axes(mesh.clone(), vec![ShardingDimension::replicated()], [axis]).unwrap(),
                )
                .unwrap()
        };
        let reduced = |axis: &str| {
            vector_type()
                .with_sharding(
                    Sharding::with_manual_axes(
                        mesh.clone(),
                        vec![ShardingDimension::replicated()],
                        Vec::<&str>::new(),
                        [axis],
                        Vec::<&str>::new(),
                    )
                    .unwrap(),
                )
                .unwrap()
        };

        // Unreduced over `x` times reduced over `x` is the partial-sum-times-replicated case: the product stays
        // unreduced over `x`, and the reduced marker is cleared.
        let output = <MulOperation as Operation<ArrayType>>::infer_output_types(
            &MulOperation,
            &[unreduced("x"), reduced("x")],
            &[],
        )
        .unwrap();
        assert_eq!(output[0].sharding().unwrap().unreduced_axes(), &BTreeSet::from(["x".to_string()]));
        assert_eq!(output[0].sharding().unwrap().reduced_axes(), &BTreeSet::new());

        // Two operands both unreduced cannot be multiplied (the product of two partial sums is not a partial sum).
        assert_eq!(
            <MulOperation as Operation<ArrayType>>::infer_output_types(
                &MulOperation,
                &[unreduced("x"), unreduced("x")],
                &[],
            ),
            Err(TypeError {
                message: format!("'{MUL_OPERATION_NAME}' cannot multiply two operands that are both unreduced")
            }),
        );

        // Unreduced over `x` requires the other operand to be reduced over exactly `x`, not a different axis.
        assert_eq!(
            <MulOperation as Operation<ArrayType>>::infer_output_types(
                &MulOperation,
                &[unreduced("x"), reduced("y")],
                &[]
            ),
            Err(TypeError {
                message: format!(
                    "'{MUL_OPERATION_NAME}' requires the second operand to be reduced over the axes the first is \
                     unreduced over",
                ),
            }),
        );

        // Two operands reduced over the same axis multiply to a value reduced over that axis.
        let output = <MulOperation as Operation<ArrayType>>::infer_output_types(
            &MulOperation,
            &[reduced("x"), reduced("x")],
            &[],
        )
        .unwrap();
        assert_eq!(output[0].sharding().unwrap().reduced_axes(), &BTreeSet::from(["x".to_string()]));
        assert_eq!(output[0].sharding().unwrap().unreduced_axes(), &BTreeSet::new());
    }
}
