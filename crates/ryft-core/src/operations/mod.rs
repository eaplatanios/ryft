use std::collections::BTreeSet;

use crate::arrays::{ArrayType, Broadcastable};
use crate::macros::check_count;
use crate::programs::{Operation, TypeError};

pub mod attention;
pub mod collectives;
pub mod compare;
pub mod complex;
pub mod constants;
pub mod control_flow;
pub mod custom_call;
pub mod debugging;
pub mod differentiation;
pub mod dimensions;
pub mod logical;
pub mod manipulation;
pub mod math;
pub mod memory;
pub mod random;
pub mod sharding;
pub mod sort;
pub mod tag;

// TODO(eaplatanios): We should be importing specific symbols here wherever possible / relevant.
pub use collectives::{Collective, CollectiveKind, CollectiveOperation, forward_collective_to_parent};
pub use compare::*;
pub use constants::*;
pub use control_flow::*;
pub use debugging::{PRINT_OPERATION_NAME, Print, PrintOperation};
pub use differentiation::*;
pub use dimensions::{
    ArithmeticDimensionOperation, DIMENSION_ADD_OPERATION_NAME, DIMENSION_DIV_FLOOR_OPERATION_NAME,
    DIMENSION_FROM_SCALAR_OPERATION_NAME, DIMENSION_MAX_OPERATION_NAME, DIMENSION_MIN_OPERATION_NAME,
    DIMENSION_MUL_OPERATION_NAME, DIMENSION_POW_OPERATION_NAME, DIMENSION_REM_OPERATION_NAME,
    DIMENSION_REQUIRE_BOUNDS_OPERATION_NAME, DIMENSION_REQUIRE_DIVISIBLE_BY_OPERATION_NAME,
    DIMENSION_REQUIRE_EQUAL_OPERATION_NAME, DIMENSION_REQUIRE_LESS_THAN_OR_EQUAL_OPERATION_NAME,
    DIMENSION_SATURATING_SUB_OPERATION_NAME, DIMENSION_SIZE_OPERATION_NAME, DIMENSION_SUB_OPERATION_NAME,
    DIMENSION_TO_SCALAR_OPERATION_NAME, DimensionAddOperation, DimensionDivFloorOperation, DimensionFromScalar,
    DimensionFromScalarOperation, DimensionMax, DimensionMaxOperation, DimensionMin, DimensionMinOperation,
    DimensionMulOperation, DimensionPow, DimensionPowOperation, DimensionRemOperation, DimensionRequirement,
    DimensionRequirementOperation, DimensionRequirementPredicate, DimensionSaturatingSub,
    DimensionSaturatingSubOperation, DimensionSize, DimensionSizeOperation, DimensionSubOperation, DimensionToScalar,
    DimensionToScalarOperation, RUNTIME_DIMENSION_DATA_TYPE,
};
pub use logical::*;
pub use manipulation::*;
pub use math::*;
pub use memory::{TRANSFER_TO_MEMORY_OPERATION_NAME, TransferToMemory, TransferToMemoryOperation};
pub use sharding::*;
pub use tag::{TAG_OPERATION_NAME, Tag, TagOperation};

/// Represents [`Operation`]s that operate elementwise on arrays and that support _broadcasting_ semantics.
/// [`ElementwiseOperation`] captures the shared type inference behavior of elementwise array operations.
/// Implementations declare their fixed input count, while the default type inference implementation checks
/// the input count, broadcasts all input [`ArrayType`]s while tolerating [`Sharding`](crate::Sharding)s that
/// differ only by [`Sharding::varying_manual_axes`](crate::Sharding::varying_manual_axes).
pub trait ElementwiseOperation: Operation<Type = ArrayType> {
    /// Returns the number of input arrays consumed by this elementwise [`Operation`].
    fn input_count(&self) -> usize;

    /// Infers the broadcasted output [`ArrayType`] for this elementwise [`Operation`]. Operations whose output
    /// [`Sharding`](crate::Sharding) does not follow plain broadcasting semantics (e.g., [`MulOperation`], which is
    /// bilinear in its operands and combines their reduction state accordingly) must override this function, typically
    /// using [`infer_elementwise_broadcast_type`](Self::infer_elementwise_broadcast_type) for the data type, shapes,
    /// and placement, and layering their own sharding rule on top.
    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, self.input_count(), TypeError);
        Ok(vec![self.infer_elementwise_broadcast_type(input_types)?])
    }

    /// Broadcasts the elementwise operands into a single output [`ArrayType`], tolerating shardings that differ only by
    /// their [`Sharding::varying_manual_axes`](crate::Sharding::varying_manual_axes). Ryft keeps generic [`ArrayType`]
    /// broadcasting conservative, and so this function retries inference after erasing only the varying-manual-axis
    /// (VMA) metadata and then restores the union of that metadata on the result, instead of weakening generic
    /// [`ArrayType`] broadcasting everywhere.
    ///
    /// This is effectively a shared helper function for the default [`infer_output_types`](Self::infer_output_types)
    /// implementation and for operations that override that default to layer extra sharding rules on top of the
    /// broadcasted placement (e.g., [`MulOperation`]'s bilinear reduction-state rule).
    fn infer_elementwise_broadcast_type(&self, input_types: &[ArrayType]) -> Result<ArrayType, TypeError> {
        match ArrayType::broadcasted(input_types) {
            Ok(output) => Ok(output),
            Err(_) => {
                let original_varying_manual_axes = input_types
                    .iter()
                    .filter_map(|input_type| input_type.sharding.as_ref())
                    .flat_map(|sharding| sharding.varying_manual_axes().iter().cloned())
                    .collect::<BTreeSet<_>>();
                let mut input_types = input_types.to_vec();
                for sharding in input_types.iter_mut().filter_map(|input_type| input_type.sharding.as_mut()) {
                    sharding.clear_varying_manual_axes();
                }
                let mut output = ArrayType::broadcasted(input_types.as_slice()).map_err(|_| {
                    TypeError::invalid(format!("'{}' input types are not broadcast-compatible", self.name()))
                })?;
                if let Some(sharding) = &mut output.sharding {
                    sharding.set_varying_manual_axes(original_varying_manual_axes).map_err(TypeError::custom)?;
                }
                Ok(output)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable, Layout, LogicalMesh, MeshAxis,
        MeshAxisType, Shape, Sharding, ShardingDimension, StridedLayout,
    };
    use crate::programs::RegionInterface;

    use super::*;

    #[test]
    fn test_elementwise_operation_type_inference() {
        #[derive(Clone, Debug)]
        struct TestElementwiseArrayOperation {
            input_count: usize,
        }

        impl Operation for TestElementwiseArrayOperation {
            type Type = ArrayType;

            #[inline]
            fn name(&self) -> &'static str {
                "elementwise_test"
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

        impl ElementwiseOperation for TestElementwiseArrayOperation {
            #[inline]
            fn input_count(&self) -> usize {
                self.input_count
            }
        }

        let operation = TestElementwiseArrayOperation { input_count: 1 };
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        assert_eq!(Operation::infer_output_types(&operation, &[input_type.clone()], &[]), Ok(vec![input_type]));
        assert_eq!(
            Operation::infer_output_types(&operation, &[], &[]),
            Err(TypeError::invalid("expected 1 input but got 0".to_string())),
        );

        let operation = TestElementwiseArrayOperation { input_count: 2 };
        assert_eq!(
            Operation::infer_output_types(
                &operation,
                &[
                    ArrayType::scalar(DataType::F32).with_layout(Layout::Strided(StridedLayout::new(Vec::new()))),
                    ArrayType::scalar(DataType::F32),
                ],
                &[],
            ),
            Ok(vec![ArrayType::scalar(DataType::F32)]),
        );
        assert_eq!(
            Operation::infer_output_types(
                &operation,
                &[
                    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)])),
                    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])),
                ],
                &[],
            ),
            Err(TypeError::invalid("'elementwise_test' input types are not broadcast-compatible".to_string())),
        );

        let operation = TestElementwiseArrayOperation { input_count: 3 };
        let output = Operation::infer_output_types(
            &operation,
            &[
                ArrayType::scalar(DataType::F32),
                ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])),
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1), Dimension::Static(3)])),
            ],
            &[],
        )
        .unwrap();
        assert_eq!(
            output,
            vec![ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))],
        );

        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let first = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
                    .unwrap()
                    .with_varying_manual_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        let second = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
                    .unwrap()
                    .with_varying_manual_axes(["y"])
                    .unwrap(),
            )
            .unwrap();
        let third = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])])
                    .unwrap()
                    .with_varying_manual_axes(["z"])
                    .unwrap(),
            )
            .unwrap();
        let output = Operation::infer_output_types(&operation, &[first, second, third], &[]).unwrap();
        assert_eq!(
            output[0].sharding().unwrap().varying_manual_axes(),
            &BTreeSet::from(["x".to_string(), "y".to_string(), "z".to_string()]),
        );

        // Dynamic dimensions flow through elementwise congruence when they match exactly, while static-vs-dynamic
        // mismatches are rejected.
        let operation = TestElementwiseArrayOperation { input_count: 2 };
        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
                Dimension::Static(3),
            ]),
        );
        assert_eq!(
            Operation::infer_output_types(&operation, &[dynamic_type.clone(), dynamic_type.clone()], &[]),
            Ok(vec![dynamic_type.clone()]),
        );
        assert_eq!(
            Operation::infer_output_types(
                &operation,
                &[
                    dynamic_type,
                    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])),
                ],
                &[],
            ),
            Err(TypeError::invalid("'elementwise_test' input types are not broadcast-compatible".to_string())),
        );
    }
}
