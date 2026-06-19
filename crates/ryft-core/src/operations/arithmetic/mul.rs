use std::collections::BTreeSet;
use std::fmt::Display;
use std::ops::Mul;

use crate::broadcasting::Broadcastable;
use crate::contexts::StagingContext;
use crate::macros::check_count;
use crate::operations::{ElementwiseOperation, InterpretableOperation, Operation};
use crate::programs::{ProgramError, Value};
use crate::sharding::Sharding;
use crate::tracing::Tracer;
use crate::types::{ArrayType, DataType, Type, TypeError};

/// Canonical operation name for [`MulOperation`].
pub const MUL_OPERATION_NAME: &'static str = "mul";

/// [`Operation`] that multiplies two values and typically supports broadcasting semantics for arrays.
#[derive(Clone, Debug, Default)]
pub struct MulOperation;

impl Display for MulOperation {
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
    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        input_types[0].broadcast(&input_types[1]).map(|output| vec![output]).map_err(|_| TypeError {
            message: format!("{MUL_OPERATION_NAME} input types are not broadcast-compatible"),
        })
    }
}

impl ElementwiseOperation for MulOperation {
    #[inline]
    fn name(&self) -> &'static str {
        MUL_OPERATION_NAME
    }

    #[inline]
    fn input_count(&self) -> usize {
        2
    }

    // TODO(eaplatanios): Review this function. Also, test.
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        // Multiplication is bilinear, so its output sharding combines the operands' unreduced/reduced reduction
        // state by the bilinear rule (JAX's `_mul_ur_rule`) rather than the congruent rule that generic elementwise
        // broadcasting applies: the operands cannot both be unreduced (the product of two distributed partial sums
        // is not itself a partial sum), but an operand unreduced over some axes times one reduced over exactly those
        // axes yields a result unreduced over them. The combination does not depend on the per-dimension placement,
        // so it applies for every `MeshAxisType` rather than being gated to explicit axes. The placement is broadcast
        // with the reduction state stripped (so the shared broadcast does not reject a legitimate unreduced/reduced
        // pairing), then the reduction state is recomputed and reattached.
        //
        // Background on JAX's unreduced/reduced sharding type system that this rule mirrors:
        // https://blog.ezyang.com/2026/01/jax-sharding-type-system/ (the `_mul_ur_rule` it references lives in JAX's
        // `jax/_src/lax/lax.py`).
        check_count!("input", input_types, 2, TypeError);
        let stripped = [strip_reduction_state(&input_types[0]), strip_reduction_state(&input_types[1])];
        let output = self.broadcast_output_type(&stripped)?;

        let (left_unreduced, left_reduced) = reduction_state(&input_types[0]);
        let (right_unreduced, right_reduced) = reduction_state(&input_types[1]);
        let (output_unreduced, output_reduced) =
            combine_bilinear_reduction_state(&left_unreduced, &left_reduced, &right_unreduced, &right_reduced)?;
        if output_unreduced.is_empty() && output_reduced.is_empty() {
            return Ok(vec![output]);
        }

        // Reduction state is only non-empty when an operand carried it, so the broadcast output has a sharding to
        // rebuild with the recomputed reduction state.
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

// TODO(eaplatanios): Review this function. Also, test.
/// Returns the `(unreduced, reduced)` axis sets of an [`ArrayType`]'s [`Sharding`], or empty sets when it has none.
fn reduction_state(input_type: &ArrayType) -> (BTreeSet<String>, BTreeSet<String>) {
    match input_type.sharding() {
        Some(sharding) => (sharding.unreduced_axes().clone(), sharding.reduced_axes().clone()),
        None => (BTreeSet::new(), BTreeSet::new()),
    }
}

// TODO(eaplatanios): Review this function. Also, test.
/// Returns a copy of `input_type` whose [`Sharding`] (if any) has its unreduced and reduced axis sets cleared while
/// its per-dimension placement and varying-manual axes are preserved, so the shared elementwise broadcast does not
/// reject operands that disagree on their reduction state (which the bilinear rule combines separately).
fn strip_reduction_state(input_type: &ArrayType) -> ArrayType {
    let Some(sharding) = input_type.sharding() else {
        return input_type.clone();
    };
    let stripped = Sharding::with_manual_axes(
        sharding.mesh().clone(),
        sharding.dimensions().to_vec(),
        Vec::<String>::new(),
        Vec::<String>::new(),
        sharding.varying_manual_axes().clone(),
    )
    .expect("clearing reduction-state axes preserves a valid sharding");
    input_type.clone().with_sharding(stripped).expect("a same-rank sharding stays valid")
}

// TODO(eaplatanios): Review this function. Also, test.
/// Combines two operands' `(unreduced, reduced)` reduction-state axis sets under the bilinear rule (JAX's
/// `_mul_ur_rule`): the operands cannot both be unreduced; an operand unreduced over a set of axes requires the
/// other to be reduced over exactly those axes and yields an unreduced result over them; reduced axes otherwise
/// propagate when the operands agree, and any axis that ends up unreduced is removed from the reduced set.
fn combine_bilinear_reduction_state(
    left_unreduced: &BTreeSet<String>,
    left_reduced: &BTreeSet<String>,
    right_unreduced: &BTreeSet<String>,
    right_reduced: &BTreeSet<String>,
) -> Result<(BTreeSet<String>, BTreeSet<String>), TypeError> {
    let output_unreduced = match (left_unreduced.is_empty(), right_unreduced.is_empty()) {
        (false, false) => {
            return Err(TypeError {
                message: format!("{MUL_OPERATION_NAME} cannot multiply two operands that are both unreduced"),
            });
        }
        (false, true) => {
            if left_unreduced != right_reduced {
                return Err(TypeError {
                    message: format!(
                        "{MUL_OPERATION_NAME} requires the second operand to be reduced over the axes the first is \
                         unreduced over"
                    ),
                });
            }
            left_unreduced.clone()
        }
        (true, false) => {
            if right_unreduced != left_reduced {
                return Err(TypeError {
                    message: format!(
                        "{MUL_OPERATION_NAME} requires the first operand to be reduced over the axes the second is \
                         unreduced over"
                    ),
                });
            }
            right_unreduced.clone()
        }
        (true, true) => BTreeSet::new(),
    };

    let mut output_reduced = if left_reduced.is_empty() {
        right_reduced.clone()
    } else if right_reduced.is_empty() || left_reduced == right_reduced {
        left_reduced.clone()
    } else {
        return Err(TypeError { message: format!("{MUL_OPERATION_NAME} operands must be reduced over the same axes") });
    };
    output_reduced.retain(|axis| !output_unreduced.contains(axis));
    Ok((output_unreduced, output_reduced))
}

impl<T: Type, V: Clone + Value<T> + Mul<Output = V>> InterpretableOperation<T, V> for MulOperation
where
    Self: Operation<T>,
{
    #[inline]
    fn interpret(
        &self,
        _context: &<V as Value<T>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].clone() * inputs[1].clone()])
    }
}

impl<C: StagingContext<Operation: From<MulOperation>>> Mul for Tracer<C> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.binary(&rhs, MulOperation)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
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
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F32, DataType::F64]),
            Ok(vec![DataType::F64]),
        );
        assert_eq!(InterpretableOperation::<DataType, f64>::interpret(&operation, &mut (), &[2.0, 3.5]), Ok(vec![7.0]));
        assert_eq!(
            InterpretableOperation::<ArrayType, TestArray>::interpret(
                &operation,
                &mut (),
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
        let output = <MulOperation as Operation<ArrayType>>::infer_output_types(&operation, &[left, right]).unwrap();
        assert_eq!(
            output[0].sharding().as_ref().unwrap().varying_manual_axes(),
            &BTreeSet::from(["x".to_string(), "y".to_string()]),
        );

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64]),
            Err(TypeError { message: "expected 2 inputs but got 1".to_string() }),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[ArrayType::scalar(DataType::F64)]),
            Err(TypeError { message: "expected 2 inputs but got 1".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<DataType, f64>::interpret(&operation, &mut (), &[2.0]),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, TestArray>::interpret(&operation, &mut (), &[TestArray::scalar(2.0)]),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F8E3M4, DataType::F32]),
            Err(TypeError { message: format!("{MUL_OPERATION_NAME} input types are not broadcast-compatible") }),
        );
        let error = <MulOperation as Operation<ArrayType>>::infer_output_types(
            &operation,
            &[
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)])),
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)])),
            ],
        )
        .unwrap_err();
        assert_eq!(
            error,
            TypeError { message: format!("{MUL_OPERATION_NAME} input types are not broadcast-compatible") }
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<DataType, f64, MulOperation>::new();
        let left = builder.add_input(DataType::F64);
        let right = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, vec![left, right]).unwrap()[0];
        let program = builder.build::<(f64, f64), f64>(vec![output], (Placeholder, Placeholder), Placeholder).unwrap();
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

    // TODO(eaplatanios): Review this function.
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
        let output =
            <MulOperation as Operation<ArrayType>>::infer_output_types(&MulOperation, &[unreduced("x"), reduced("x")])
                .unwrap();
        assert_eq!(output[0].sharding().unwrap().unreduced_axes(), &BTreeSet::from(["x".to_string()]));
        assert_eq!(output[0].sharding().unwrap().reduced_axes(), &BTreeSet::new());

        // Two operands both unreduced cannot be multiplied (the product of two partial sums is not a partial sum).
        assert_eq!(
            <MulOperation as Operation<ArrayType>>::infer_output_types(
                &MulOperation,
                &[unreduced("x"), unreduced("x")]
            ),
            Err(TypeError {
                message: format!("{MUL_OPERATION_NAME} cannot multiply two operands that are both unreduced")
            }),
        );

        // Unreduced over `x` requires the other operand to be reduced over exactly `x`, not a different axis.
        assert_eq!(
            <MulOperation as Operation<ArrayType>>::infer_output_types(&MulOperation, &[unreduced("x"), reduced("y")]),
            Err(TypeError {
                message: format!(
                    "{MUL_OPERATION_NAME} requires the second operand to be reduced over the axes the first is \
                     unreduced over"
                ),
            }),
        );

        // Two operands reduced over the same axis multiply to a value reduced over that axis.
        let output =
            <MulOperation as Operation<ArrayType>>::infer_output_types(&MulOperation, &[reduced("x"), reduced("x")])
                .unwrap();
        assert_eq!(output[0].sharding().unwrap().reduced_axes(), &BTreeSet::from(["x".to_string()]));
        assert_eq!(output[0].sharding().unwrap().unreduced_axes(), &BTreeSet::new());
    }
}
