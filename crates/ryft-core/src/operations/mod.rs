use crate::arrays::{ArrayType, Broadcastable};
use crate::macros::check_count;
use crate::programs::{Operation, TypeError};

pub mod arithmetic;
pub mod assertions;
pub mod attention;
pub mod collectives;
pub mod comparisons;
pub mod complex;
pub mod constants;
pub mod control_flow;
pub mod cumulative;
pub mod custom_call;
pub mod debugging;
pub mod differentiation;
pub mod dimensions;
pub mod dot;
pub mod exponential;
pub mod extrema;
pub mod logical;
pub mod manipulation;
pub mod quantization;
pub mod random;
pub mod reductions;
pub mod references;
pub mod rounding;
pub mod sharding;
pub mod sort;
pub mod special;
pub mod tagging;
pub mod trigonometric;

// TODO(eaplatanios): We should be importing specific symbols here wherever possible / relevant.
pub use arithmetic::{
    ABS_OPERATION_NAME, ADD_OPERATION_NAME, Abs, AbsOperation, Add, AddOperation, DIV_OPERATION_NAME, Div,
    DivOperation, MUL_OPERATION_NAME, Mul, MulOperation, NEG_OPERATION_NAME, Neg, NegOperation, POW_OPERATION_NAME,
    Pow, PowOperation, REM_OPERATION_NAME, RSQRT_OPERATION_NAME, Rem, RemOperation, Rsqrt, RsqrtOperation,
    SIGN_OPERATION_NAME, SQRT_OPERATION_NAME, SUB_OPERATION_NAME, Sign, SignOperation, Sqrt, SqrtOperation, Sub,
    SubOperation,
};
pub use assertions::{
    ASSERT_OPERATION_NAME, Assert, AssertOperation, AssertionError, AssertionFailure, AssertionValue,
};
pub use collectives::{
    ManualVariationAlignment, PARALLEL_VARY_OPERATION_NAME, ParallelReduce, ParallelReduceOperation,
    ParallelReductionKind, ParallelVary, ParallelVaryOperation, forward_collective_to_parent,
};
pub use comparisons::*;
pub use constants::*;
pub use control_flow::*;
pub use cumulative::*;
pub use debugging::{PRINT_OPERATION_NAME, Print, PrintOperation};
pub use differentiation::{
    CUSTOM_JVP_OPERATION_NAME, CUSTOM_VJP_OPERATION_NAME, CustomJvp, CustomJvpOperation, CustomVjp, CustomVjpOperation,
    LinearCallOperation, STOP_GRADIENT_OPERATION_NAME, StopGradient, StopGradientOperation, StopGradients, custom_jvp,
    custom_vjp,
};
pub use dimensions::{
    ArithmeticDimensionOperation, DIMENSION_ADD_OPERATION_NAME, DIMENSION_DATA_TYPE, DIMENSION_DIV_OPERATION_NAME,
    DIMENSION_FROM_SCALAR_OPERATION_NAME, DIMENSION_MAX_OPERATION_NAME, DIMENSION_MIN_OPERATION_NAME,
    DIMENSION_MUL_OPERATION_NAME, DIMENSION_POW_OPERATION_NAME, DIMENSION_REM_OPERATION_NAME,
    DIMENSION_SATURATING_SUB_OPERATION_NAME, DIMENSION_SIZE_OPERATION_NAME, DIMENSION_SUB_OPERATION_NAME,
    DIMENSION_TO_SCALAR_OPERATION_NAME, DimensionAddOperation, DimensionDivOperation, DimensionFromScalar,
    DimensionFromScalarOperation, DimensionMax, DimensionMaxOperation, DimensionMin, DimensionMinOperation,
    DimensionMulOperation, DimensionPow, DimensionPowOperation, DimensionRemOperation, DimensionSaturatingSub,
    DimensionSaturatingSubOperation, DimensionSize, DimensionSizeOperation, DimensionSubOperation, DimensionToScalar,
    DimensionToScalarOperation,
};
pub use dot::{
    DOT_OPERATION_NAME, Dot, DotDimensionNumbers, DotOperation, DotOps, RAGGED_DOT_OPERATION_NAME, RaggedDot,
    RaggedDotDimensionNumbers, RaggedDotMode, RaggedDotOperation,
};
pub use exponential::{
    EXP_OPERATION_NAME, Exp, ExpOperation, LN_1P_OPERATION_NAME, LOG_ADD_EXP_OPERATION_NAME, LOG_OPERATION_NAME,
    LOGISTIC_OPERATION_NAME, Ln1p, Ln1pOperation, Log, LogAddExp, LogAddExpOperation, LogOperation, Logistic,
    LogisticOperation,
};
pub use extrema::{
    CLAMP_OPERATION_NAME, Clamp, ClampOperation, MAX_OPERATION_NAME, MIN_OPERATION_NAME, Max, MaxOperation, Min,
    MinOperation,
};
pub use logical::*;
pub use manipulation::*;
pub use quantization::{BlockQuantize, SCALED_DOT_OPERATION_NAME, ScaledDot, ScaledDotOperation};
pub use reductions::{LogSumExp, Reduce, ReduceOperation, ReductionKind};
pub use references::{
    REFERENCE_ADD_UPDATE_OPERATION_NAME, REFERENCE_ATOMIC_ADD_UPDATE_OPERATION_NAME,
    REFERENCE_DYNAMIC_INDEX_OPERATION_NAME, REFERENCE_FREEZE_OPERATION_NAME, REFERENCE_INDEX_OPERATION_NAME,
    REFERENCE_NEW_OPERATION_NAME, REFERENCE_READ_OPERATION_NAME, REFERENCE_SLICE_OPERATION_NAME,
    REFERENCE_SWAP_OPERATION_NAME, REFERENCE_WRITE_OPERATION_NAME, ReferenceAddUpdate, ReferenceAddUpdateOperation,
    ReferenceAtomicAddUpdate, ReferenceAtomicAddUpdateOperation, ReferenceDynamicIndex, ReferenceDynamicIndexOperation,
    ReferenceFreeze, ReferenceFreezeOperation, ReferenceIndex, ReferenceIndexOperation, ReferenceNew,
    ReferenceNewOperation, ReferenceRead, ReferenceReadOperation, ReferenceSlice, ReferenceSliceOperation,
    ReferenceSwap, ReferenceSwapOperation, ReferenceWrite, ReferenceWriteOperation,
};
pub use rounding::{
    CEIL_OPERATION_NAME, Ceil, CeilOperation, FLOOR_OPERATION_NAME, Floor, FloorOperation, ROUND_OPERATION_NAME, Round,
    RoundOperation,
};
pub use sharding::*;
pub use special::{ERF_OPERATION_NAME, Erf, ErfOperation};
pub use tagging::{TAG_OPERATION_NAME, Tag, TagOperation};
pub use trigonometric::{
    ATAN2_OPERATION_NAME, Atan2, Atan2Operation, COS_OPERATION_NAME, Cos, CosOperation, SIN_OPERATION_NAME, Sin,
    SinOperation, TANH_OPERATION_NAME, Tanh, TanhOperation,
};

/// Represents [`Operation`]s that operate elementwise on arrays and that support _broadcasting_ semantics.
/// [`ElementwiseOperation`] captures the shared type inference behavior of elementwise array operations.
/// Implementations declare their fixed input count, while the default type inference implementation checks
/// the input count and matching manual variation, then broadcasts all input [`ArrayType`]s. Binding must
/// insert explicit variation transitions before inference when invariant and varying values are combined.
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

    /// Broadcasts input geometry and placement after validating matching manual variation. Operations with specialized
    /// reduction-state rules may normalize those states before calling this function and restore their output state.
    fn infer_elementwise_broadcast_type(&self, input_types: &[ArrayType]) -> Result<ArrayType, TypeError> {
        ArrayType::check_matching_manual_variation(self.name(), &input_types.iter().collect::<Vec<_>>())?;
        ArrayType::broadcasted(input_types)
            .map_err(|_| TypeError::invalid(format!("`{}` input types are not broadcast-compatible", self.name())))
    }
}

#[cfg(test)]
mod tests {
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
            Err(TypeError::invalid("`elementwise_test` input types are not broadcast-compatible".to_string())),
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
        assert_eq!(
            Operation::infer_output_types(&operation, &[first.clone(), second, third], &[]),
            Err(TypeError::invalid(
                "`elementwise_test` inputs must have matching varying manual axes; insert `parallel_vary` on the \
                 inputs that lack an axis, as `align_manual_variation` does",
            )),
        );
        assert_eq!(
            Operation::infer_output_types(
                &operation,
                &[first.clone(), first.clone(), ArrayType::scalar(DataType::F32)],
                &[],
            ),
            Err(TypeError::invalid(
                "`elementwise_test` inputs must have matching varying manual axes; insert `parallel_vary` on the \
                 inputs that lack an axis, as `align_manual_variation` does",
            )),
        );
        assert_eq!(
            Operation::infer_output_types(&operation, &[first.clone(), first.clone(), first.clone()], &[]),
            Ok(vec![first]),
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
            Err(TypeError::invalid("`elementwise_test` input types are not broadcast-compatible".to_string())),
        );
    }
}
