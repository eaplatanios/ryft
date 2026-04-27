use std::collections::HashMap;

#[cfg(feature = "ndarray")]
use ndarray::Array2;
use ryft_mlir::dialects::{func, shardy, stable_hlo, stable_hlo::Accuracy, stable_hlo::Precision};
use ryft_mlir::{
    Attribute, Block, BlockRef, Context as MlirContext, DenseElementsAttributeRef, Location, LocationRef,
    Operation as MlirOperation, Region, Size as MlirSize, Type, TypeAndAttributes, TypeRef, Value, ValueRef,
};

use ryft_core::parameters::Parameterized;
use ryft_core::sharding::{LogicalMesh, ShardingError};
use ryft_core::tracing::{AtomId, Instruction, Operation, Program, Traceable, TracingError};
use ryft_core::tracing_v2::{
    CustomPrimitive, LinearPrimitiveOperation, MatrixOps, PrimitiveOperation,
    operations::{
        AddOperation, CosOperation, LeftMatMulOperation, LinearRematerializeOperation, MatMulOperation,
        MatrixTransposeOperation, MulOperation, NegOperation, RematerializeOperation, ReshapeOperation,
        RightMatMulOperation, ScaleOperation, SinOperation,
        control_flow::{ConditionOperation, ConditionPredicate, WhileOperation},
    },
};
use ryft_core::types::{ArrayType, DataType, Size, Typed};

use crate::experimental::operations::{
    LinearShardMapEvalMode, LinearShardMapOperation, ShardMapOperation, WithShardingConstraintOperation,
};
use crate::experimental::ops::XlaPrimitiveOperation;
use crate::mlir::ToMlir;

use super::shard_map::{ShardMap, ShardMapConstantKind, ShardMapError, ShardMapTensor};
/// Error type for StableHLO/Shardy lowering.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum LoweringError {
    /// Underlying shard-map error returned while building manual-computation attributes.
    #[error("{0}")]
    ShardMapError(#[from] ShardMapError),

    /// Underlying sharding error returned while building mesh or sharding attributes.
    #[error("{0}")]
    ShardingError(#[from] ShardingError),

    /// Error returned when a lowered function name is empty or contains whitespace.
    #[error("invalid function name '{function_name}' used during XLA lowering")]
    InvalidFunctionName { function_name: String },

    /// Error returned when lowering encounters a traced tensor type that MLIR rejects.
    #[error("invalid tensor type '{array_type}' used during XLA lowering")]
    InvalidTensorType { array_type: ArrayType },

    /// Error returned when lowering encounters a staged op that does not yet have StableHLO support.
    #[error("unsupported staged op '{op}' during XLA lowering")]
    UnsupportedOp { op: String },

    /// Error returned when lowering encounters a constant value that it does not know how to build.
    #[error("unsupported traced constant at atom %{atom_id} during XLA lowering")]
    UnsupportedConstant { atom_id: AtomId },

    /// Error returned when lowering encounters a type that does not have StableHLO support yet.
    #[error("unsupported data type '{data_type}' during XLA lowering")]
    UnsupportedDataType { data_type: DataType },

    /// Error returned when lowering needs a staged value that was never assigned.
    #[error("missing lowered value for staged atom %{atom_id}")]
    MissingAtomValue { atom_id: AtomId },

    /// Error returned when MLIR rejects the constructed dense-elements attribute.
    #[error("invalid dense elements attribute for data type '{data_type}' during XLA lowering")]
    InvalidDenseElementsAttribute { data_type: DataType },

    /// Error returned when the constructed MLIR module fails verification.
    #[error("constructed MLIR module failed verification during XLA lowering")]
    MlirVerificationFailure,

    /// Error returned when one traced XLA program mixes shard maps from incompatible meshes.
    #[error("traced XLA lowering requires all nested shard maps to use compatible logical meshes")]
    IncompatibleNestedMeshes,

    /// Error returned when simplifying a staged program prior to lowering fails.
    #[error("failed to simplify staged XLA program before lowering: {message}")]
    SimplificationFailure { message: String },

    /// Error returned when one custom primitive does not provide StableHLO lowering.
    #[error("custom primitive '{op}' does not provide StableHLO lowering")]
    MissingCustomLowering { op: String },

    /// Underlying tracing error returned while replaying a staged program through the generic
    /// [`Program::interpret_with`] engine.
    #[error("{0}")]
    Tracing(#[from] TracingError),
}

/// Lowering mode used for plain `tracing_v2` MLIR emission.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum PlainMlirLoweringMode {
    /// Lower the program exactly as traced.
    Unpacked,
}

/// Lowering helper passed to op-owned plain StableHLO lowering hooks.
pub(crate) struct PlainMlirLowerer<'b, 'c: 'b, 't: 'c> {
    /// Owning block receiving the lowered operations.
    pub(crate) block: BlockRef<'b, 'c, 't>,

    /// MLIR context owning the block and created operations.
    pub(crate) context: &'c MlirContext<'t>,

    /// Shared MLIR location used for emitted operations.
    pub(crate) location: LocationRef<'c, 't>,
}

impl<'b, 'c: 'b, 't: 'c> PlainMlirLowerer<'b, 'c, 't> {
    /// Lowers one tensor type inside this lowering context.
    pub(crate) fn lower_tensor_type(
        &self,
        array_type: &ArrayType,
    ) -> Result<ryft_mlir::TensorTypeRef<'c, 't>, LoweringError> {
        lower_tensor_type(array_type, self.context, self.location)
    }

    /// Lowers one plain traced literal value inside this lowering context.
    pub(crate) fn lower_literal_value<V: MlirLowerableValue>(
        &mut self,
        value: &V,
    ) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
        lower_literal_value(value, &mut self.block, self.context, self.location)
    }

    /// Lowers one nested `rematerialize` op by inlining the body sub-program into the current block
    /// and placing an optimization barrier on the boundary outputs.
    pub(crate) fn lower_rematerialize<V: MlirLowerableValue, O: Clone + XlaOperation<V>, L: Clone>(
        &mut self,
        remat_op: &RematerializeOperation<ArrayType, V, O, L>,
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_rematerialize_inline(
            remat_op.body().program(),
            input_values,
            &mut self.block,
            self.context,
            self.location,
        )
    }

    /// Lowers one nested condition operation inside this lowering context.
    pub(crate) fn lower_condition<V: MlirLowerableValue, O: Clone + XlaOperation<V>>(
        &mut self,
        condition_op: &ConditionOperation<V, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_condition_to_if(condition_op, input_values, &mut self.block, self.context, self.location)
    }

    /// Lowers one nested while operation inside this lowering context.
    pub(crate) fn lower_while<V: MlirLowerableValue, O: Clone + XlaOperation<V>>(
        &mut self,
        while_op: &WhileOperation<V, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_while_to_while(while_op, input_values, &mut self.block, self.context, self.location)
    }
}

/// StableHLO lowering hook carried by one [`CustomPrimitive`].
pub(crate) trait StableHloCustomLowering<V: Traceable<ArrayType>> {
    /// Lowers one custom primitive to StableHLO/Shardy operations.
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        op: &CustomPrimitive<ArrayType, V>,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>;
}

/// Typed StableHLO lowering extension stored inside one [`CustomPrimitive`].
#[derive(Clone)]
pub(crate) struct StableHloCustomLoweringExtension<V: Traceable<ArrayType>> {
    lowering: std::sync::Arc<dyn StableHloCustomLowering<V>>,
}

impl<V: Traceable<ArrayType>> StableHloCustomLoweringExtension<V> {
    /// Creates one StableHLO lowering extension from a registered lowering rule.
    pub(crate) fn new(lowering: std::sync::Arc<dyn StableHloCustomLowering<V>>) -> Self {
        Self { lowering }
    }

    /// Lowers one custom primitive through the registered StableHLO lowering rule.
    pub(crate) fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        op: &CustomPrimitive<ArrayType, V>,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        self.lowering.lower_to_mlir(op, input_values, output_types, lowerer)
    }
}

/// Operations that can be lowered to StableHLO for XLA compilation.
///
/// Implementing this trait makes an operation eligible for MLIR lowering via
/// [`to_mlir_module_for_plain_program`] and related entry points. The core [`PrimitiveOperation`] and
/// [`LinearPrimitiveOperation`] enums provide the default blanket implementations, and backends can add
/// their own closed op carriers by implementing this trait for those enums.
pub(crate) trait XlaOperation<V: MlirLowerableValue>: Operation<ArrayType> {
    /// Lowers this operation to one or more StableHLO operations.
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>;
}

impl<V: MlirLowerableValue> XlaOperation<V> for AddOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result =
            lowerer.block.append_operation(stable_hlo::add(input_values[0], input_values[1], lowerer.location));
        Ok(vec![result.result(0).expect("stablehlo.add should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> XlaOperation<V> for MulOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result =
            lowerer
                .block
                .append_operation(stable_hlo::multiply(input_values[0], input_values[1], lowerer.location));
        Ok(vec![result.result(0).expect("stablehlo.multiply should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> XlaOperation<V> for NegOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::negate(input_values[0], lowerer.location));
        Ok(vec![result.result(0).expect("stablehlo.negate should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> XlaOperation<V> for SinOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result =
            lowerer
                .block
                .append_operation(stable_hlo::sine(input_values[0], Accuracy::Default, lowerer.location));
        Ok(vec![result.result(0).expect("stablehlo.sine should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> XlaOperation<V> for CosOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result =
            lowerer
                .block
                .append_operation(stable_hlo::cosine(input_values[0], Accuracy::Default, lowerer.location));
        Ok(vec![result.result(0).expect("stablehlo.cosine should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> XlaOperation<V> for MatrixTransposeOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::transpose(input_values[0], &[1, 0], lowerer.location));
        Ok(vec![result.result(0).expect("stablehlo.transpose should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> XlaOperation<V> for MatMulOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let output_tensor_type = lowerer.lower_tensor_type(&output_types[0])?;
        let dimensions = lowerer.context.stable_hlo_dot_dimensions(&[], &[], &[1], &[0]);
        let result = lowerer.block.append_operation(stable_hlo::dot_general(
            input_values[0],
            input_values[1],
            dimensions,
            Some((Precision::Default, Precision::Default)),
            None,
            output_tensor_type,
            lowerer.location,
        ));
        Ok(vec![result.result(0).expect("stablehlo.dot_general should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> XlaOperation<V> for ScaleOperation<ArrayType, V> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let factor = self.factor();
        let factor_value = lowerer.lower_literal_value(factor)?;
        let output_tensor_type = lowerer.lower_tensor_type(&output_types[0])?;
        let factor_type = factor.r#type();
        let factor_broadcast = if *factor_type != output_types[0] {
            let broadcast = lowerer.block.append_operation(stable_hlo::broadcast(
                factor_value,
                output_tensor_type,
                &[],
                lowerer.location,
            ));
            broadcast.result(0).expect("stablehlo.broadcast should return one result").as_ref()
        } else {
            factor_value
        };
        let result =
            lowerer
                .block
                .append_operation(stable_hlo::multiply(input_values[0], factor_broadcast, lowerer.location));
        Ok(vec![result.result(0).expect("stablehlo.multiply should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue + ryft_core::tracing_v2::MatrixOps> XlaOperation<V> for LeftMatMulOperation<V> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let factor = self.factor();
        let factor_value = lowerer.lower_literal_value(factor)?;
        let output_tensor_type = lowerer.lower_tensor_type(&output_types[0])?;
        let dimensions = lowerer.context.stable_hlo_dot_dimensions(&[], &[], &[1], &[0]);
        let result = lowerer.block.append_operation(stable_hlo::dot_general(
            factor_value,
            input_values[0],
            dimensions,
            Some((Precision::Default, Precision::Default)),
            None,
            output_tensor_type,
            lowerer.location,
        ));
        Ok(vec![result.result(0).expect("stablehlo.dot_general should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue + ryft_core::tracing_v2::MatrixOps> XlaOperation<V> for RightMatMulOperation<V> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let factor = self.factor();
        let factor_value = lowerer.lower_literal_value(factor)?;
        let output_tensor_type = lowerer.lower_tensor_type(&output_types[0])?;
        let dimensions = lowerer.context.stable_hlo_dot_dimensions(&[], &[], &[1], &[0]);
        let result = lowerer.block.append_operation(stable_hlo::dot_general(
            input_values[0],
            factor_value,
            dimensions,
            Some((Precision::Default, Precision::Default)),
            None,
            output_tensor_type,
            lowerer.location,
        ));
        Ok(vec![result.result(0).expect("stablehlo.dot_general should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> XlaOperation<V> for ReshapeOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let output_shape = static_dimensions(self.output_type())?;
        let result = lowerer.block.append_operation(stable_hlo::reshape(
            input_values[0],
            output_shape.as_slice(),
            lowerer.location,
        ));
        Ok(vec![result.result(0).expect("stablehlo.reshape should return one result").as_ref()])
    }
}

impl XlaOperation<ShardMapTensor> for XlaPrimitiveOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        match self {
            Self::Add => <AddOperation as XlaOperation<ShardMapTensor>>::lower_to_mlir(
                &AddOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Mul => <MulOperation as XlaOperation<ShardMapTensor>>::lower_to_mlir(
                &MulOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Neg => <NegOperation as XlaOperation<ShardMapTensor>>::lower_to_mlir(
                &NegOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Sin => <SinOperation as XlaOperation<ShardMapTensor>>::lower_to_mlir(
                &SinOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Cos => <CosOperation as XlaOperation<ShardMapTensor>>::lower_to_mlir(
                &CosOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::MatMul => <MatMulOperation as XlaOperation<ShardMapTensor>>::lower_to_mlir(
                &MatMulOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::MatrixTranspose => <MatrixTransposeOperation as XlaOperation<ShardMapTensor>>::lower_to_mlir(
                &MatrixTransposeOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Scale { factor } => {
                <ScaleOperation<ArrayType, ShardMapTensor> as XlaOperation<ShardMapTensor>>::lower_to_mlir(
                    &ScaleOperation::new(factor.clone()),
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            Self::LeftMatMul { factor } => {
                <LeftMatMulOperation<ShardMapTensor> as XlaOperation<ShardMapTensor>>::lower_to_mlir(
                    &LeftMatMulOperation::new(factor.clone()),
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            Self::RightMatMul { factor } => {
                <RightMatMulOperation<ShardMapTensor> as XlaOperation<ShardMapTensor>>::lower_to_mlir(
                    &RightMatMulOperation::new(factor.clone()),
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            Self::Reshape { input_type, output_type } => {
                <ReshapeOperation as XlaOperation<ShardMapTensor>>::lower_to_mlir(
                    &ReshapeOperation::new(input_type.clone(), output_type.clone()),
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            Self::Rematerialize(remat) => lowerer.lower_rematerialize(remat.as_ref(), input_values),
            Self::Condition(condition) => lowerer.lower_condition(condition.as_ref(), input_values),
            Self::While(while_operation) => lowerer.lower_while(while_operation.as_ref(), input_values),
            Self::ShardMap(shard_map_op) => {
                let simplified_body = shard_map_op
                    .body()
                    .simplified()
                    .map_err(|error| LoweringError::SimplificationFailure { message: error.to_string() })?;
                lower_manual_computation(
                    &mut lowerer.block,
                    input_values,
                    &simplified_body.shard_map,
                    &simplified_body.program,
                    simplified_body.local_input_types.as_slice(),
                    simplified_body.global_output_types.as_slice(),
                    lowerer.context,
                    lowerer.location,
                )
            }
            Self::LinearShardMap(shard_map_op) => lower_linear_shard_map_eval_mode(
                shard_map_op.eval_mode(),
                &[],
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            Self::WithShardingConstraint(op) => {
                let operation = lowerer.block.append_operation(shardy::sharding_constraint(
                    input_values[0],
                    op.sharding().to_mlir(lowerer.location),
                    lowerer.location,
                ));
                Ok(vec![operation.result(0).expect("sdy.sharding_constraint should return one result").as_ref()])
            }
            Self::Custom(custom_op) => {
                let mut shard_map_lowerer =
                    ShardMapMlirLowerer { block: lowerer.block, context: lowerer.context, location: lowerer.location };
                custom_op
                    .extensions()
                    .get::<StableHloCustomLoweringExtension<ShardMapTensor>>()
                    .ok_or_else(|| LoweringError::MissingCustomLowering { op: self.name().to_string() })?
                    .lower_to_mlir(custom_op.as_ref(), input_values, output_types, &mut shard_map_lowerer)
            }
        }
    }
}

impl<V: MlirLowerableValue, O: Clone + XlaOperation<V>> XlaOperation<V>
    for LinearRematerializeOperation<ArrayType, V, O>
{
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_rematerialize_inline(
            self.body().program(),
            input_values,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )
    }
}

impl<V: MlirLowerableValue, O: Clone + XlaOperation<V>> XlaOperation<V> for ConditionOperation<V, O> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lowerer.lower_condition(self, input_values)
    }
}

impl<V: MlirLowerableValue, O: Clone + XlaOperation<V>> XlaOperation<V> for WhileOperation<V, O> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lowerer.lower_while(self, input_values)
    }
}

impl<V: MlirLowerableValue + MatrixOps> XlaOperation<V> for PrimitiveOperation<V> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        match self {
            PrimitiveOperation::Add => <AddOperation as XlaOperation<V>>::lower_to_mlir(
                &AddOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            PrimitiveOperation::Mul => <MulOperation as XlaOperation<V>>::lower_to_mlir(
                &MulOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            PrimitiveOperation::Neg => <NegOperation as XlaOperation<V>>::lower_to_mlir(
                &NegOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            PrimitiveOperation::Sin => <SinOperation as XlaOperation<V>>::lower_to_mlir(
                &SinOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            PrimitiveOperation::Cos => <CosOperation as XlaOperation<V>>::lower_to_mlir(
                &CosOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            PrimitiveOperation::MatrixTranspose => <MatrixTransposeOperation as XlaOperation<V>>::lower_to_mlir(
                &MatrixTransposeOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            PrimitiveOperation::MatMul => <MatMulOperation as XlaOperation<V>>::lower_to_mlir(
                &MatMulOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            PrimitiveOperation::Scale { factor } => <ScaleOperation<ArrayType, V> as XlaOperation<V>>::lower_to_mlir(
                &ScaleOperation::new(factor.clone()),
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            PrimitiveOperation::LeftMatMul { factor } => <LeftMatMulOperation<V> as XlaOperation<V>>::lower_to_mlir(
                &LeftMatMulOperation::new(factor.clone()),
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            PrimitiveOperation::RightMatMul { factor } => <RightMatMulOperation<V> as XlaOperation<V>>::lower_to_mlir(
                &RightMatMulOperation::new(factor.clone()),
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            PrimitiveOperation::Reshape { input_type, output_type } => {
                <ReshapeOperation as XlaOperation<V>>::lower_to_mlir(
                    &ReshapeOperation::new(input_type.clone(), output_type.clone()),
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            PrimitiveOperation::Rematerialize(remat) => lowerer.lower_rematerialize(remat, input_values),
            PrimitiveOperation::Condition(condition) => {
                condition.lower_to_mlir(input_values, output_types, mode, lowerer)
            }
            PrimitiveOperation::While(while_operation) => {
                while_operation.lower_to_mlir(input_values, output_types, mode, lowerer)
            }
            PrimitiveOperation::Custom(_) => {
                Err(LoweringError::UnsupportedOp { op: Operation::name(self).to_string() })
            }
        }
    }
}

impl<V: MlirLowerableValue + MatrixOps> XlaOperation<V> for LinearPrimitiveOperation<V> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        match self {
            LinearPrimitiveOperation::Add => <AddOperation as XlaOperation<V>>::lower_to_mlir(
                &AddOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            LinearPrimitiveOperation::Neg => <NegOperation as XlaOperation<V>>::lower_to_mlir(
                &NegOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            LinearPrimitiveOperation::MatrixTranspose => <MatrixTransposeOperation as XlaOperation<V>>::lower_to_mlir(
                &MatrixTransposeOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            LinearPrimitiveOperation::Scale { factor } => {
                <ScaleOperation<ArrayType, V> as XlaOperation<V>>::lower_to_mlir(
                    &ScaleOperation::new(factor.clone()),
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            LinearPrimitiveOperation::LeftMatMul { factor } => {
                <LeftMatMulOperation<V> as XlaOperation<V>>::lower_to_mlir(
                    &LeftMatMulOperation::new(factor.clone()),
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            LinearPrimitiveOperation::RightMatMul { factor } => {
                <RightMatMulOperation<V> as XlaOperation<V>>::lower_to_mlir(
                    &RightMatMulOperation::new(factor.clone()),
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            LinearPrimitiveOperation::Reshape { input_type, output_type } => {
                <ReshapeOperation as XlaOperation<V>>::lower_to_mlir(
                    &ReshapeOperation::new(input_type.clone(), output_type.clone()),
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            LinearPrimitiveOperation::Zero(zero) => {
                if !input_values.is_empty() {
                    return Err(TracingError::InvalidInputCount { expected: 0, got: input_values.len() }.into());
                }
                if output_types.len() != 1 {
                    return Err(TracingError::InvalidOutputCount { expected: 1, got: output_types.len() }.into());
                }
                let zero_type = zero.output_type();
                let block = &mut lowerer.block;
                let context = lowerer.context;
                let location = lowerer.location;
                let tensor_type = lower_tensor_type(zero_type, context, location)?;
                let elements = lower_constant_elements_attribute(
                    zero_type.data_type,
                    tensor_type,
                    ShardMapConstantKind::Zero,
                    context,
                )?;
                let constant = block.append_operation(stable_hlo::constant(elements, location));
                Ok(vec![constant.result(0).expect("stablehlo.constant should return one result").as_ref()])
            }
            LinearPrimitiveOperation::Rematerialize(remat) => {
                <LinearRematerializeOperation<ArrayType, V> as XlaOperation<V>>::lower_to_mlir(
                    remat,
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            LinearPrimitiveOperation::Condition(condition) => {
                condition.lower_to_mlir(input_values, output_types, mode, lowerer)
            }
            LinearPrimitiveOperation::While(while_operation) => {
                while_operation.lower_to_mlir(input_values, output_types, mode, lowerer)
            }
            LinearPrimitiveOperation::Custom(custom_op) => {
                let mut shard_map_lowerer =
                    ShardMapMlirLowerer { block: lowerer.block, context: lowerer.context, location: lowerer.location };
                custom_op
                    .primitive()
                    .extensions()
                    .get::<StableHloCustomLoweringExtension<V>>()
                    .ok_or_else(|| LoweringError::MissingCustomLowering {
                        op: custom_op.primitive().name().to_string(),
                    })?
                    .lower_to_mlir(custom_op.primitive().as_ref(), input_values, output_types, &mut shard_map_lowerer)
            }
        }
    }
}

/// Lowering helper passed to op-owned traced XLA MLIR lowering hooks.
pub(crate) struct ShardMapMlirLowerer<'b, 'c: 'b, 't: 'c> {
    /// Owning block receiving the lowered operations.
    pub(crate) block: BlockRef<'b, 'c, 't>,

    /// MLIR context owning the block and created operations.
    pub(crate) context: &'c MlirContext<'t>,

    /// Shared MLIR location used for emitted operations.
    pub(crate) location: LocationRef<'c, 't>,
}

impl<'b, 'c: 'b, 't: 'c> ShardMapMlirLowerer<'b, 'c, 't> {
    /// Lowers one tensor type inside this lowering context.
    pub(crate) fn lower_tensor_type(
        &self,
        array_type: &ArrayType,
    ) -> Result<ryft_mlir::TensorTypeRef<'c, 't>, LoweringError> {
        lower_tensor_type(array_type, self.context, self.location)
    }

    /// Lowers one nested `rematerialize` op by inlining the body sub-program into the current block
    /// and placing an optimization barrier on the boundary outputs.
    pub(crate) fn lower_rematerialize<V: MlirLowerableValue, O: Clone + XlaOperation<V>, L: Clone>(
        &mut self,
        remat_op: &RematerializeOperation<ArrayType, V, O, L>,
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_rematerialize_inline(
            remat_op.body().program(),
            input_values,
            &mut self.block,
            self.context,
            self.location,
        )
    }

    /// Lowers one nested condition operation inside this lowering context.
    pub(crate) fn lower_condition<V: MlirLowerableValue, O: Clone + XlaOperation<V>>(
        &mut self,
        condition_op: &ConditionOperation<V, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_condition_to_if(condition_op, input_values, &mut self.block, self.context, self.location)
    }

    /// Lowers one nested while operation inside this lowering context.
    pub(crate) fn lower_while<V: MlirLowerableValue, O: Clone + XlaOperation<V>>(
        &mut self,
        while_op: &WhileOperation<V, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_while_to_while(while_op, input_values, &mut self.block, self.context, self.location)
    }

    /// Lowers one nested Shardy manual computation operation inside this lowering context.
    pub(crate) fn lower_manual_computation<
        ProgramInput: Parameterized<ShardMapTensor>,
        ProgramOutput: Parameterized<ShardMapTensor>,
    >(
        &mut self,
        outer_inputs: &[ValueRef<'b, 'c, 't>],
        shard_map: &ShardMap,
        program: &Program<ArrayType, ShardMapTensor, XlaPrimitiveOperation, ProgramInput, ProgramOutput>,
        local_input_types: &[ArrayType],
        global_output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_manual_computation(
            &mut self.block,
            outer_inputs,
            shard_map,
            program,
            local_input_types,
            global_output_types,
            self.context,
            self.location,
        )
    }

    /// Lowers one linear shard-map evaluation mode inside this lowering context.
    pub(crate) fn lower_linear_shard_map_eval_mode(
        &mut self,
        eval_mode: &LinearShardMapEvalMode,
        captured_values: &[ValueRef<'b, 'c, 't>],
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_linear_shard_map_eval_mode(
            eval_mode,
            captured_values,
            input_values,
            &mut self.block,
            self.context,
            self.location,
        )
    }
}

/// Lowers a traced shard-map program to a textual StableHLO/Shardy MLIR module.
pub(crate) fn to_mlir_module<
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
    ProgramInput: Parameterized<ShardMapTensor>,
    ProgramOutput: Parameterized<ShardMapTensor>,
    S: AsRef<str>,
>(
    shard_map: &ShardMap,
    program: &Program<ArrayType, ShardMapTensor, XlaPrimitiveOperation, ProgramInput, ProgramOutput>,
    global_input_types: &Input,
    local_input_types: &Input,
    global_output_types: &Output,
    _local_output_types: &Output,
    function_name: S,
) -> Result<String, LoweringError> {
    let function_name = normalize_function_name(function_name.as_ref())?;
    let global_input_types = global_input_types.parameters().cloned().collect::<Vec<_>>();
    let local_input_types = local_input_types.parameters().cloned().collect::<Vec<_>>();
    let global_output_types = global_output_types.parameters().cloned().collect::<Vec<_>>();

    let context = MlirContext::new();
    let location = context.unknown_location();
    let module = context.module(location);

    let global_input_tensor_types = global_input_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let global_output_tensor_types = global_output_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let mesh_operation = shard_map.mesh().to_mlir(location);
    module.body().append_operation(mesh_operation);

    let function_arguments = global_input_tensor_types
        .iter()
        .zip(shard_map.in_shardings().iter())
        .map(|(tensor_type, sharding)| {
            let sharding = sharding.to_mlir(location);
            Ok(TypeAndAttributes {
                r#type: tensor_type.as_ref(),
                attributes: Some(HashMap::from([("sdy.sharding".into(), sharding.as_ref())])),
            })
        })
        .collect::<Result<Vec<_>, LoweringError>>()?;
    let function_results = global_output_tensor_types
        .iter()
        .zip(shard_map.out_shardings().iter())
        .map(|(tensor_type, sharding)| {
            let sharding = sharding.to_mlir(location);
            Ok(TypeAndAttributes {
                r#type: tensor_type.as_ref(),
                attributes: Some(HashMap::from([("sdy.sharding".into(), sharding.as_ref())])),
            })
        })
        .collect::<Result<Vec<_>, LoweringError>>()?;

    module.body().append_operation({
        let function_block = context.block(
            global_input_tensor_types
                .iter()
                .map(|tensor_type| (*tensor_type, location))
                .collect::<Vec<_>>()
                .as_slice(),
        );
        let outer_inputs = (0..global_input_tensor_types.len())
            .map(|index| function_block.argument(index).expect("function block arguments should exist").as_ref())
            .collect::<Vec<_>>();
        let mut function_block_ref = function_block.as_ref();
        let manual_results = lower_manual_computation(
            &mut function_block_ref,
            outer_inputs.as_slice(),
            shard_map,
            program,
            local_input_types.as_slice(),
            global_output_types.as_slice(),
            &context,
            location.as_ref(),
        )?;
        function_block_ref.append_operation(func::r#return(manual_results.as_slice(), location));

        func::func(
            function_name.as_str(),
            func::FuncAttributes { arguments: function_arguments, results: function_results, ..Default::default() },
            function_block.into(),
            location,
        )
    });

    if !module.verify() {
        return Err(LoweringError::MlirVerificationFailure);
    }

    Ok(module.to_string())
}

/// Lowers an arbitrary traced XLA program to a textual StableHLO/Shardy MLIR module.
pub(crate) fn to_mlir_module_for_program<Input, Output, ProgramInput, ProgramOutput, S>(
    program: &Program<ArrayType, ShardMapTensor, XlaPrimitiveOperation, ProgramInput, ProgramOutput>,
    global_input_types: &Input,
    global_output_types: &Output,
    function_name: S,
) -> Result<String, LoweringError>
where
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
    ProgramInput: Parameterized<ShardMapTensor>,
    ProgramOutput: Parameterized<ShardMapTensor>,
    S: AsRef<str>,
{
    let function_name = normalize_function_name(function_name.as_ref())?;
    let global_input_types = global_input_types.parameters().cloned().collect::<Vec<_>>();
    let global_output_types = global_output_types.parameters().cloned().collect::<Vec<_>>();

    let context = MlirContext::new();
    let location = context.unknown_location();
    let module = context.module(location);

    if let Some(mesh) = collect_nested_sharding_mesh(program, None)? {
        let mesh_operation = mesh.to_mlir(location);
        module.body().append_operation(mesh_operation);
    }

    let global_input_tensor_types = global_input_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let global_output_tensor_types = global_output_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let function_arguments = global_input_tensor_types
        .iter()
        .map(|tensor_type| TypeAndAttributes { r#type: tensor_type.as_ref(), attributes: None })
        .collect::<Vec<_>>();
    let function_results = global_output_tensor_types
        .iter()
        .map(|tensor_type| TypeAndAttributes { r#type: tensor_type.as_ref(), attributes: None })
        .collect::<Vec<_>>();

    module.body().append_operation({
        let function_block = context.block(
            global_input_tensor_types
                .iter()
                .map(|tensor_type| (*tensor_type, location))
                .collect::<Vec<_>>()
                .as_slice(),
        );
        {
            let mut function_block_ref = function_block.as_ref();
            let outputs = lower_program_outputs(program, &mut function_block_ref, &context, location.as_ref())?;
            function_block_ref.append_operation(func::r#return(outputs.as_slice(), location));
        }
        func::func(
            function_name.as_str(),
            func::FuncAttributes { arguments: function_arguments, results: function_results, ..Default::default() },
            function_block.into(),
            location,
        )
    });

    if !module.verify() {
        return Err(LoweringError::MlirVerificationFailure);
    }
    Ok(module.to_string())
}

/// Value type that can be materialized as a StableHLO dense constant during benchmark lowering.
pub(crate) trait MlirLowerableValue: Clone + Traceable<ArrayType> + Typed<ArrayType> + 'static {
    /// Builds a dense-elements attribute containing this value.
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError>;

    /// Builds a scalar dense-elements attribute when this value can be represented as a scalar splat.
    #[inline]
    fn to_scalar_dense_elements_attribute<'c, 't>(
        &self,
        _tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        _context: &'c MlirContext<'t>,
    ) -> Result<Option<DenseElementsAttributeRef<'c, 't>>, LoweringError> {
        Ok(None)
    }
}

impl MlirLowerableValue for f64 {
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
        context
            .dense_f64_elements_attribute(tensor_type, std::slice::from_ref(self))
            .and_then(|attribute| attribute.cast::<DenseElementsAttributeRef>())
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type: DataType::F64 })
    }

    fn to_scalar_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<Option<DenseElementsAttributeRef<'c, 't>>, LoweringError> {
        Ok(Some(self.to_dense_elements_attribute(tensor_type, context)?))
    }
}

#[cfg(feature = "ndarray")]
impl MlirLowerableValue for Array2<f64> {
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
        let elements = self.iter().copied().collect::<Vec<_>>();
        context
            .dense_f64_elements_attribute(tensor_type, elements.as_slice())
            .and_then(|attribute| attribute.cast::<DenseElementsAttributeRef>())
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type: DataType::F64 })
    }

    fn to_scalar_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<Option<DenseElementsAttributeRef<'c, 't>>, LoweringError> {
        if self.shape() == [1, 1] {
            return Ok(Some(
                context
                    .dense_f64_elements_attribute(tensor_type, std::slice::from_ref(&self[(0, 0)]))
                    .and_then(|attribute| attribute.cast::<DenseElementsAttributeRef>())
                    .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type: DataType::F64 })?,
            ));
        }
        Ok(None)
    }
}

impl MlirLowerableValue for ShardMapTensor {
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
        let constant_kind =
            self.constant_kind().ok_or(LoweringError::UnsupportedConstant { atom_id: AtomId { index: 0 } })?;
        lower_constant_elements_attribute(self.r#type().data_type, tensor_type, constant_kind, context)
    }

    fn to_scalar_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<Option<DenseElementsAttributeRef<'c, 't>>, LoweringError> {
        let Some(constant_kind) = self.constant_kind() else {
            return Ok(None);
        };
        Ok(Some(lower_constant_elements_attribute(self.r#type().data_type, tensor_type, constant_kind, context)?))
    }
}

/// Lowers a plain traced `tracing_v2` program to a textual StableHLO MLIR module.
#[cfg(any(test, feature = "benchmarking"))]
#[allow(dead_code)]
pub(crate) fn to_mlir_module_for_plain_program<
    V: MlirLowerableValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
    O: Clone + XlaOperation<V>,
    S: AsRef<str>,
>(
    program: &Program<ArrayType, V, O, Input, Output>,
    function_name: S,
) -> Result<String, LoweringError> {
    let function_name = normalize_function_name(function_name.as_ref())?;
    let context = MlirContext::new();
    let location = context.unknown_location();
    let module = context.module(location);

    let input_tensor_types = program
        .input_ids
        .iter()
        .map(|atom_id| {
            let input_atom = &program.atoms[atom_id.index];
            lower_tensor_type(&input_atom.r#type(), &context, location)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output_tensor_types = program
        .output_ids
        .iter()
        .map(|atom_id| {
            let output_atom = &program.atoms[atom_id.index];
            lower_tensor_type(&output_atom.r#type(), &context, location)
        })
        .collect::<Result<Vec<_>, _>>()?;

    module.body().append_operation({
        let function_block = context.block(
            input_tensor_types.iter().map(|tensor_type| (*tensor_type, location)).collect::<Vec<_>>().as_slice(),
        );
        {
            let mut function_block_ref = function_block.as_ref();
            let outputs = lower_plain_program_outputs(program, &mut function_block_ref, &context, location.as_ref())?;
            function_block_ref.append_operation(func::r#return(outputs.as_slice(), location));
        }
        func::func(
            function_name.as_str(),
            func::FuncAttributes {
                arguments: input_tensor_types
                    .iter()
                    .map(|tensor_type| TypeAndAttributes { r#type: tensor_type.as_ref(), attributes: None })
                    .collect(),
                results: output_tensor_types
                    .iter()
                    .map(|tensor_type| TypeAndAttributes { r#type: tensor_type.as_ref(), attributes: None })
                    .collect(),
                ..Default::default()
            },
            function_block.into(),
            location,
        )
    });

    if !module.verify() {
        return Err(LoweringError::MlirVerificationFailure);
    }

    Ok(module.to_string())
}

fn collect_nested_sharding_mesh<ProgramInput, ProgramOutput>(
    program: &Program<ArrayType, ShardMapTensor, XlaPrimitiveOperation, ProgramInput, ProgramOutput>,
    existing: Option<LogicalMesh>,
) -> Result<Option<LogicalMesh>, LoweringError>
where
    ProgramInput: Parameterized<ShardMapTensor>,
    ProgramOutput: Parameterized<ShardMapTensor>,
{
    let mut mesh = existing;
    for instruction in &program.instructions {
        match &instruction.operation {
            XlaPrimitiveOperation::ShardMap(shard_map_op) => {
                mesh = Some(match mesh.take() {
                    Some(existing_mesh) => merge_logical_meshes(&existing_mesh, shard_map_op.body().shard_map.mesh())?,
                    None => shard_map_op.body().shard_map.mesh().clone(),
                });
                mesh = collect_nested_sharding_mesh(&shard_map_op.body().program, mesh)?;
            }
            XlaPrimitiveOperation::LinearShardMap(shard_map_op) => {
                mesh = collect_nested_linear_shard_map_mesh(shard_map_op.eval_mode(), mesh)?;
            }
            XlaPrimitiveOperation::Condition(condition_op) => {
                mesh = collect_nested_sharding_mesh(condition_op.true_branch(), mesh)?;
                mesh = collect_nested_sharding_mesh(condition_op.false_branch(), mesh)?;
            }
            XlaPrimitiveOperation::While(while_op) => {
                mesh = collect_nested_sharding_mesh(while_op.condition(), mesh)?;
                mesh = collect_nested_sharding_mesh(while_op.body(), mesh)?;
            }
            XlaPrimitiveOperation::WithShardingConstraint(sharding_constraint_op) => {
                mesh = Some(match mesh.take() {
                    Some(existing_mesh) => {
                        merge_logical_meshes(&existing_mesh, &sharding_constraint_op.sharding().mesh)?
                    }
                    None => sharding_constraint_op.sharding().mesh.clone(),
                });
            }
            XlaPrimitiveOperation::Custom(custom_op) => {
                if let Some(shard_map_op) = custom_op.extensions().get::<LinearShardMapOperation<ShardMapTensor>>() {
                    mesh = collect_nested_linear_shard_map_mesh(shard_map_op.eval_mode(), mesh)?;
                } else if let Some(shard_map_op) = custom_op.extensions().get::<ShardMapOperation<ShardMapTensor>>() {
                    mesh = Some(match mesh.take() {
                        Some(existing_mesh) => {
                            merge_logical_meshes(&existing_mesh, shard_map_op.body().shard_map.mesh())?
                        }
                        None => shard_map_op.body().shard_map.mesh().clone(),
                    });
                    mesh = collect_nested_sharding_mesh(&shard_map_op.body().program, mesh)?;
                } else if let Some(sharding_constraint_op) =
                    custom_op.extensions().get::<WithShardingConstraintOperation>()
                {
                    mesh = Some(match mesh.take() {
                        Some(existing_mesh) => {
                            merge_logical_meshes(&existing_mesh, &sharding_constraint_op.sharding().mesh)?
                        }
                        None => sharding_constraint_op.sharding().mesh.clone(),
                    });
                }
            }
            _ => {}
        }
    }
    Ok(mesh)
}

/// Collects nested logical meshes referenced by one linear shard-map evaluation mode.
fn collect_nested_linear_shard_map_mesh(
    eval_mode: &LinearShardMapEvalMode,
    existing: Option<LogicalMesh>,
) -> Result<Option<LogicalMesh>, LoweringError> {
    match eval_mode {
        LinearShardMapEvalMode::Body(body) => {
            let mesh = Some(match existing {
                Some(existing_mesh) => merge_logical_meshes(&existing_mesh, body.shard_map.mesh())?,
                None => body.shard_map.mesh().clone(),
            });
            collect_nested_sharding_mesh(&body.program, mesh)
        }
        LinearShardMapEvalMode::FactorizedTranspose(factorized) => {
            let mesh = Some(match existing {
                Some(existing_mesh) => merge_logical_meshes(&existing_mesh, factorized.residual_body.shard_map.mesh())?,
                None => factorized.residual_body.shard_map.mesh().clone(),
            });
            let mesh = collect_nested_sharding_mesh(&factorized.residual_body.program, mesh)?;
            let mesh = Some(match mesh {
                Some(existing_mesh) => merge_logical_meshes(&existing_mesh, factorized.apply_body.shard_map.mesh())?,
                None => factorized.apply_body.shard_map.mesh().clone(),
            });
            collect_nested_sharding_mesh(&factorized.apply_body.program, mesh)
        }
    }
}

fn merge_logical_meshes(existing: &LogicalMesh, incoming: &LogicalMesh) -> Result<LogicalMesh, LoweringError> {
    let mut merged_axes = existing.axes.clone();
    for incoming_axis in &incoming.axes {
        match existing.axis_size(incoming_axis.name.as_str()) {
            Some(existing_size) if existing_size != incoming_axis.size => {
                return Err(LoweringError::IncompatibleNestedMeshes);
            }
            Some(_) => {}
            None => merged_axes.push(incoming_axis.clone()),
        }
    }
    LogicalMesh::new(merged_axes).map_err(LoweringError::from)
}

/// Returns the static dimensions for one tensor type.
fn static_dimensions(array_type: &ArrayType) -> Result<Vec<usize>, LoweringError> {
    array_type
        .shape
        .dimensions
        .iter()
        .map(|size| match size {
            Size::Static(value) => Ok(*value),
            Size::Dynamic(_) => Err(LoweringError::InvalidTensorType { array_type: array_type.clone() }),
        })
        .collect()
}

fn lower_control_flow_region<'b, 'c: 'b, 't: 'c, V, O>(
    program: &Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    input_values: &[ValueRef<'b, 'c, 't>],
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ryft_mlir::DetachedRegion<'c, 't>, LoweringError>
where
    V: MlirLowerableValue,
    O: Clone + XlaOperation<V>,
{
    let mut region = context.region();
    let block = context.block_with_no_arguments();
    {
        let mut block_ref = block.as_ref();
        let outputs = lower_nested_program_inline(program, input_values, &mut block_ref, context, location, false)?;
        block_ref.append_operation(stable_hlo::r#return(outputs.as_slice(), location));
    }
    region.append_block(block);
    Ok(region)
}

fn lower_condition_to_if<'b, 'c: 'b, 't: 'c, V, O>(
    condition_op: &ConditionOperation<V, O>,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: MlirLowerableValue,
    O: Clone + XlaOperation<V>,
{
    let operand_count = condition_op.input_types().len();
    match condition_op.predicate() {
        ConditionPredicate::Captured(predicate) => {
            if input_values.len() != operand_count {
                return Err(LoweringError::UnsupportedOp {
                    op: format!("condition expected {operand_count} lowered inputs but got {}", input_values.len()),
                });
            }
            let branch = if *predicate { condition_op.true_branch() } else { condition_op.false_branch() };
            lower_nested_program_inline(branch, input_values, block, context, location, false)
        }
        ConditionPredicate::RuntimeInput(_) => {
            let expected_input_count = operand_count + 1;
            if input_values.len() != expected_input_count {
                return Err(LoweringError::UnsupportedOp {
                    op: format!(
                        "condition expected {expected_input_count} lowered inputs but got {}",
                        input_values.len(),
                    ),
                });
            }
            let branch_inputs = &input_values[1..];
            let true_branch_region =
                lower_control_flow_region(condition_op.true_branch(), branch_inputs, context, location)?;
            let false_branch_region =
                lower_control_flow_region(condition_op.false_branch(), branch_inputs, context, location)?;
            let operation = block.append_operation(stable_hlo::r#if(
                input_values[0],
                true_branch_region.into(),
                false_branch_region.into(),
                location,
            ));
            Ok((0..condition_op.output_types().len())
                .map(|index| {
                    operation.result(index).expect("stablehlo.if should return one result per output").as_ref()
                })
                .collect())
        }
    }
}

fn lower_while_to_while<'b, 'c: 'b, 't: 'c, V, O>(
    while_op: &WhileOperation<V, O>,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: MlirLowerableValue,
    O: Clone + XlaOperation<V>,
{
    let state_types = while_op.state_types();
    if input_values.len() != state_types.len() {
        return Err(LoweringError::UnsupportedOp {
            op: format!("while expected {} lowered inputs but got {}", state_types.len(), input_values.len()),
        });
    }
    let lowered_state_types = state_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, context, location).map(|tensor_type| tensor_type.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    let block_arguments = lowered_state_types.iter().map(|r#type| (*r#type, location)).collect::<Vec<_>>();

    let mut condition_region = context.region();
    let condition_block = context.block(block_arguments.as_slice());
    {
        let mut condition_block_ref = condition_block.as_ref();
        let condition_inputs = (0..state_types.len())
            .map(|index| {
                condition_block_ref.argument(index).expect("while condition should have state arguments").as_ref()
            })
            .collect::<Vec<_>>();
        let condition_outputs = lower_nested_program_inline(
            while_op.condition(),
            condition_inputs.as_slice(),
            &mut condition_block_ref,
            context,
            location,
            false,
        )?;
        if condition_outputs.len() != 1 {
            return Err(LoweringError::UnsupportedOp {
                op: format!("while condition lowered to {} outputs", condition_outputs.len()),
            });
        }
        condition_block_ref.append_operation(stable_hlo::r#return(condition_outputs.as_slice(), location));
    }
    condition_region.append_block(condition_block);

    let mut body_region = context.region();
    let body_block = context.block(block_arguments.as_slice());
    {
        let mut body_block_ref = body_block.as_ref();
        let body_inputs = (0..state_types.len())
            .map(|index| body_block_ref.argument(index).expect("while body should have state arguments").as_ref())
            .collect::<Vec<_>>();
        let body_outputs = lower_nested_program_inline(
            while_op.body(),
            body_inputs.as_slice(),
            &mut body_block_ref,
            context,
            location,
            false,
        )?;
        if body_outputs.len() != state_types.len() {
            return Err(LoweringError::UnsupportedOp {
                op: format!("while body lowered to {} outputs", body_outputs.len()),
            });
        }
        body_block_ref.append_operation(stable_hlo::r#return(body_outputs.as_slice(), location));
    }
    body_region.append_block(body_block);

    let operation = block.append_operation(stable_hlo::r#while(
        input_values,
        condition_region.into(),
        body_region.into(),
        location,
    ));
    Ok((0..state_types.len())
        .map(|index| operation.result(index).expect("stablehlo.while should return one result per state leaf").as_ref())
        .collect())
}

/// Inlines a nested sub-program into the given block by mapping the provided input
/// MLIR values to the body's input atoms, lowering constants and instructions in topological
/// order, and returning lowered values corresponding to the program's output atoms.
fn lower_nested_program_inline<'b, 'c: 'b, 't: 'c, O, V>(
    program: &Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    add_optimization_barrier: bool,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: MlirLowerableValue,
    O: Clone + XlaOperation<V>,
{
    let outputs = replay_program_into_block(
        program,
        input_values.to_vec(),
        block,
        context,
        location,
        |_, value, block, context, location| lower_literal_value(value, block, context, location),
        |instruction, inputs, block, context, location| {
            let output_types = instruction
                .outputs
                .iter()
                .map(|output| program.atoms[output.index].r#type().into_owned())
                .collect::<Vec<_>>();
            let mut lowerer = PlainMlirLowerer { block: *block, context, location };
            instruction.operation.lower_to_mlir(
                inputs,
                output_types.as_slice(),
                PlainMlirLoweringMode::Unpacked,
                &mut lowerer,
            )
        },
    )?;
    if outputs.is_empty() || !add_optimization_barrier {
        return Ok(outputs);
    }
    let barrier = block.append_operation(stable_hlo::optimization_barrier(outputs.as_slice(), location));
    Ok((0..outputs.len())
        .map(|index| {
            barrier
                .result(index)
                .expect("stablehlo.optimization_barrier should return one result per operand")
                .as_ref()
        })
        .collect::<Vec<_>>())
}

/// Drives [`Program::interpret_with`] to lower a staged program into MLIR ops appended to `block`.
///
/// The two callbacks plug in lowering policies for [`Atom::Constant`]s and [`Instruction`]s respectively while the
/// generic interpreter handles use-count tracking and atom bookkeeping. Each callback receives a mutable [`BlockRef`]
/// because [`BlockRef`] is `Copy` and the helper hands each closure its own copy backed by the same MLIR block.
fn replay_program_into_block<'b, 'c: 'b, 't: 'c, O, V, Input, Output, LiftConstant, ApplyOp>(
    program: &Program<ArrayType, V, O, Input, Output>,
    input_values: Vec<ValueRef<'b, 'c, 't>>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    mut lift_constant: LiftConstant,
    mut apply_op: ApplyOp,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: Traceable<ArrayType>,
    O: Clone + Operation<ArrayType>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
    LiftConstant: FnMut(
        AtomId,
        &V,
        &mut BlockRef<'b, 'c, 't>,
        &'c MlirContext<'t>,
        LocationRef<'c, 't>,
    ) -> Result<ValueRef<'b, 'c, 't>, LoweringError>,
    ApplyOp: FnMut(
        &Instruction<O>,
        &[ValueRef<'b, 'c, 't>],
        &mut BlockRef<'b, 'c, 't>,
        &'c MlirContext<'t>,
        LocationRef<'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>,
{
    let mut block_for_constants = *block;
    let mut block_for_ops = *block;
    program.interpret_with(
        input_values,
        |atom_id, value| lift_constant(atom_id, value, &mut block_for_constants, context, location),
        |instruction, inputs| apply_op(instruction, inputs, &mut block_for_ops, context, location),
    )
}

/// Inlines a rematerialize body's sub-program and places an optimization barrier on its boundary outputs.
fn lower_rematerialize_inline<'b, 'c: 'b, 't: 'c, O, V>(
    program: &Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: MlirLowerableValue,
    O: Clone + XlaOperation<V>,
{
    lower_nested_program_inline(program, input_values, block, context, location, true)
}

/// Lowers one plain traced program to values inside a block.
#[cfg(any(test, feature = "benchmarking"))]
#[allow(dead_code)]
fn lower_plain_program_outputs<'b, 'c: 'b, 't: 'c, O, V, Input, Output>(
    program: &Program<ArrayType, V, O, Input, Output>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: MlirLowerableValue,
    O: Clone + XlaOperation<V>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    let input_values = (0..program.input_ids.len())
        .map(|index| block.argument(index).expect("body block arguments should exist").as_ref())
        .collect::<Vec<_>>();
    replay_program_into_block(
        program,
        input_values,
        block,
        context,
        location,
        |_, value, block, context, location| lower_literal_value(value, block, context, location),
        |instruction, inputs, block, context, location| {
            let output_types = instruction
                .outputs
                .iter()
                .map(|output| program.atoms[output.index].r#type().into_owned())
                .collect::<Vec<_>>();
            let mut lowerer = PlainMlirLowerer { block: *block, context, location };
            instruction.operation.lower_to_mlir(
                inputs,
                output_types.as_slice(),
                PlainMlirLoweringMode::Unpacked,
                &mut lowerer,
            )
        },
    )
}

/// Lowers one traced program to values inside a block.
fn lower_program_outputs<'b, 'c: 'b, 't: 'c, ProgramInput, ProgramOutput>(
    program: &Program<ArrayType, ShardMapTensor, XlaPrimitiveOperation, ProgramInput, ProgramOutput>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    ProgramInput: Parameterized<ShardMapTensor>,
    ProgramOutput: Parameterized<ShardMapTensor>,
{
    // Mirror table of every lowered atom value. Shard-map operations look up captured global primals by `AtomId`,
    // so we keep a parallel table alongside [`Program::interpret_with`]'s use-count-tracked one. [`ValueRef`] is
    // `Copy`, so this mirror is cheap.
    let mut atom_values = vec![None; program.atoms.len()];
    let input_values = program
        .input_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(index, atom_id)| {
            let value = block.argument(index).expect("body block arguments should exist").as_ref();
            atom_values[atom_id.index] = Some(value);
            value
        })
        .collect::<Vec<_>>();
    let atom_values = std::cell::RefCell::new(atom_values);
    replay_program_into_block(
        program,
        input_values,
        block,
        context,
        location,
        |atom_id, value, block, context, location| {
            let lowered = lower_constant(atom_id, value, block, context, location)?;
            atom_values.borrow_mut()[atom_id.index] = Some(lowered);
            Ok(lowered)
        },
        |instruction, inputs, block, context, location| {
            let mut table = atom_values.borrow_mut();
            let lowered_outputs =
                lower_instruction(program, instruction, table.as_slice(), inputs, block, context, location)?;
            for (output_atom, lowered_output) in
                instruction.outputs.iter().copied().zip(lowered_outputs.iter().copied())
            {
                table[output_atom.index] = Some(lowered_output);
            }
            Ok(lowered_outputs)
        },
    )
}

/// Lowers one `sdy.manual_computation` operation, including its nested body program.
fn lower_manual_computation<'b, 'c: 'b, 't: 'c, ProgramInput, ProgramOutput>(
    block: &mut BlockRef<'b, 'c, 't>,
    outer_inputs: &[ValueRef<'b, 'c, 't>],
    shard_map: &ShardMap,
    program: &Program<ArrayType, ShardMapTensor, XlaPrimitiveOperation, ProgramInput, ProgramOutput>,
    local_input_types: &[ArrayType],
    global_output_types: &[ArrayType],
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    ProgramInput: Parameterized<ShardMapTensor>,
    ProgramOutput: Parameterized<ShardMapTensor>,
{
    let local_input_tensor_types = local_input_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let global_output_tensor_types = global_output_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, context, location))
        .collect::<Result<Vec<_>, _>>()?;

    let mut body_region = context.region();
    let body_block = context.block(
        local_input_tensor_types
            .iter()
            .map(|tensor_type| (*tensor_type, location))
            .collect::<Vec<_>>()
            .as_slice(),
    );
    {
        let mut body_block_ref = body_block.as_ref();
        let body_outputs = lower_program_outputs(program, &mut body_block_ref, context, location.as_ref())?;
        body_block_ref.append_operation(shardy::r#return(body_outputs.as_slice(), location));
    }
    body_region.append_block(body_block);

    let manual_computation = block.append_operation(shardy::manual_computation(
        outer_inputs,
        global_output_tensor_types.as_slice(),
        shard_map.to_shardy_in_shardings(context),
        shard_map.to_shardy_out_shardings(context),
        shard_map.to_shardy_manual_axes(context),
        body_region,
        location,
    ));
    Ok(manual_computation.results().map(|result| result.as_ref()).collect::<Vec<_>>())
}

/// Lowers one linear shard-map evaluation mode and returns its resulting values.
fn lower_linear_shard_map_eval_mode<'b, 'c: 'b, 't: 'c>(
    eval_mode: &LinearShardMapEvalMode,
    captured_values: &[ValueRef<'b, 'c, 't>],
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    match eval_mode {
        LinearShardMapEvalMode::Body(body) => {
            let simplified_body = body
                .simplified()
                .map_err(|error| LoweringError::SimplificationFailure { message: error.to_string() })?;
            let combined_inputs =
                captured_values.iter().copied().chain(input_values.iter().copied()).collect::<Vec<_>>();
            lower_manual_computation(
                block,
                combined_inputs.as_slice(),
                &simplified_body.shard_map,
                &simplified_body.program,
                simplified_body.local_input_types.as_slice(),
                simplified_body.global_output_types.as_slice(),
                context,
                location,
            )
        }
        LinearShardMapEvalMode::FactorizedTranspose(factorized) => {
            let residual_body = factorized
                .residual_body
                .simplified()
                .map_err(|error| LoweringError::SimplificationFailure { message: error.to_string() })?;
            let residual_results = lower_manual_computation(
                block,
                &captured_values[..residual_body.global_input_types.len()],
                &residual_body.shard_map,
                &residual_body.program,
                residual_body.local_input_types.as_slice(),
                residual_body.global_output_types.as_slice(),
                context,
                location,
            )?;
            let apply_body = factorized
                .apply_body
                .simplified()
                .map_err(|error| LoweringError::SimplificationFailure { message: error.to_string() })?;
            let apply_inputs = input_values
                .iter()
                .copied()
                .take(apply_body.global_input_types.len() - residual_results.len())
                .chain(residual_results)
                .collect::<Vec<_>>();
            lower_manual_computation(
                block,
                apply_inputs.as_slice(),
                &apply_body.shard_map,
                &apply_body.program,
                apply_body.local_input_types.as_slice(),
                apply_body.global_output_types.as_slice(),
                context,
                location,
            )
        }
    }
}

/// Lowers one concrete traced value to a StableHLO constant operation and returns its result value.
fn lower_literal_value<'b, 'c: 'b, 't: 'c, B, V, L>(
    value: &V,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    V: MlirLowerableValue,
    L: Location<'c, 't> + Copy,
{
    let value_type = value.r#type();
    if !value_type.shape.dimensions.is_empty() {
        let scalar_tensor_type = context
            .tensor_type(lower_element_type(value_type.data_type, context)?, &[], None, location)
            .ok_or_else(|| LoweringError::InvalidTensorType { array_type: ArrayType::scalar(value_type.data_type) })?;
        if let Some(scalar_elements) = value.to_scalar_dense_elements_attribute(scalar_tensor_type, context)? {
            let scalar_constant = block.append_operation(stable_hlo::constant(scalar_elements, location));
            let tensor_type = lower_tensor_type(&value_type, context, location)?;
            let broadcast = block.append_operation(stable_hlo::broadcast(
                scalar_constant.result(0).unwrap().as_ref(),
                tensor_type,
                &[],
                location,
            ));
            return Ok(broadcast.result(0).expect("stablehlo.broadcast should return one result").as_ref());
        }
    }

    let tensor_type = lower_tensor_type(&value_type, context, location)?;
    let elements = value.to_dense_elements_attribute(tensor_type, context)?;
    let constant = block.append_operation(stable_hlo::constant(elements, location));
    Ok(constant.result(0).expect("stablehlo.constant should return one result").as_ref())
}

/// Lowers a traced constant atom to a StableHLO constant operation and returns its result value.
fn lower_constant<'b, 'c: 'b, 't: 'c, B, L>(
    atom_id: AtomId,
    value: &ShardMapTensor,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Location<'c, 't> + Copy,
{
    let constant_kind = value.constant_kind().ok_or(LoweringError::UnsupportedConstant { atom_id })?;
    let array_type = value.r#type();
    let tensor_type = lower_tensor_type(&array_type, context, location)?;
    if !array_type.shape.dimensions.is_empty() {
        let scalar_tensor_type = context
            .tensor_type(lower_element_type(array_type.data_type, context)?, &[], None, location)
            .ok_or_else(|| LoweringError::InvalidTensorType { array_type: ArrayType::scalar(array_type.data_type) })?;
        let scalar_elements =
            lower_constant_elements_attribute(array_type.data_type, scalar_tensor_type, constant_kind, context)?;
        let scalar_constant = block.append_operation(stable_hlo::constant(scalar_elements, location));
        let broadcast = block.append_operation(stable_hlo::broadcast(
            scalar_constant.result(0).unwrap().as_ref(),
            tensor_type,
            &[],
            location,
        ));
        return Ok(broadcast.result(0).expect("stablehlo.broadcast should return one result").as_ref());
    }
    let elements = lower_constant_elements_attribute(array_type.data_type, tensor_type, constant_kind, context)?;
    let constant = block.append_operation(stable_hlo::constant(elements, location));
    Ok(constant.result(0).expect("stablehlo.constant should return one result").as_ref())
}

/// Dispatches shard-map StableHLO lowering for one traced operation by matching on primitive variants.
fn dispatch_lower_shard_map_mlir<'b, 'c: 'b, 't: 'c>(
    op: &XlaPrimitiveOperation,
    captured_values: &[ValueRef<'b, 'c, 't>],
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayType],
    lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    match op {
        XlaPrimitiveOperation::Add => {
            let result =
                lowerer.block.append_operation(stable_hlo::add(input_values[0], input_values[1], lowerer.location));
            Ok(vec![result.result(0).expect("stablehlo.add should return one result").as_ref()])
        }
        XlaPrimitiveOperation::Mul => {
            let result = lowerer.block.append_operation(stable_hlo::multiply(
                input_values[0],
                input_values[1],
                lowerer.location,
            ));
            Ok(vec![result.result(0).expect("stablehlo.multiply should return one result").as_ref()])
        }
        XlaPrimitiveOperation::Neg => {
            let result = lowerer.block.append_operation(stable_hlo::negate(input_values[0], lowerer.location));
            Ok(vec![result.result(0).expect("stablehlo.negate should return one result").as_ref()])
        }
        XlaPrimitiveOperation::Sin => {
            let result =
                lowerer
                    .block
                    .append_operation(stable_hlo::sine(input_values[0], Accuracy::Default, lowerer.location));
            Ok(vec![result.result(0).expect("stablehlo.sine should return one result").as_ref()])
        }
        XlaPrimitiveOperation::Cos => {
            let result = lowerer.block.append_operation(stable_hlo::cosine(
                input_values[0],
                Accuracy::Default,
                lowerer.location,
            ));
            Ok(vec![result.result(0).expect("stablehlo.cosine should return one result").as_ref()])
        }
        XlaPrimitiveOperation::MatMul => {
            let output_tensor_type = lowerer.lower_tensor_type(&output_types[0])?;
            let dimensions = lowerer.context.stable_hlo_dot_dimensions(&[], &[], &[1], &[0]);
            let result = lowerer.block.append_operation(stable_hlo::dot_general(
                input_values[0],
                input_values[1],
                dimensions,
                Some((Precision::Default, Precision::Default)),
                None,
                output_tensor_type,
                lowerer.location,
            ));
            Ok(vec![result.result(0).expect("stablehlo.dot_general should return one result").as_ref()])
        }
        XlaPrimitiveOperation::MatrixTranspose => {
            let result =
                lowerer.block.append_operation(stable_hlo::transpose(input_values[0], &[1, 0], lowerer.location));
            Ok(vec![result.result(0).expect("stablehlo.transpose should return one result").as_ref()])
        }
        XlaPrimitiveOperation::Scale { factor } => {
            let output_tensor_type = lowerer.lower_tensor_type(&output_types[0])?;
            let factor_value =
                lower_constant(AtomId { index: 0 }, factor, &mut lowerer.block, lowerer.context, lowerer.location)?;
            let factor_type = factor.r#type();
            let factor_broadcast = if *factor_type != output_types[0] {
                let broadcast = lowerer.block.append_operation(stable_hlo::broadcast(
                    factor_value,
                    output_tensor_type,
                    &[],
                    lowerer.location,
                ));
                broadcast.result(0).expect("stablehlo.broadcast should return one result").as_ref()
            } else {
                factor_value
            };
            let result = lowerer.block.append_operation(stable_hlo::multiply(
                input_values[0],
                factor_broadcast,
                lowerer.location,
            ));
            Ok(vec![result.result(0).expect("stablehlo.multiply should return one result").as_ref()])
        }
        XlaPrimitiveOperation::LeftMatMul { factor } => {
            let factor_value =
                lower_constant(AtomId { index: 0 }, factor, &mut lowerer.block, lowerer.context, lowerer.location)?;
            let output_tensor_type = lowerer.lower_tensor_type(&output_types[0])?;
            let dimensions = lowerer.context.stable_hlo_dot_dimensions(&[], &[], &[1], &[0]);
            let result = lowerer.block.append_operation(stable_hlo::dot_general(
                factor_value,
                input_values[0],
                dimensions,
                Some((Precision::Default, Precision::Default)),
                None,
                output_tensor_type,
                lowerer.location,
            ));
            Ok(vec![result.result(0).expect("stablehlo.dot_general should return one result").as_ref()])
        }
        XlaPrimitiveOperation::RightMatMul { factor } => {
            let factor_value =
                lower_constant(AtomId { index: 0 }, factor, &mut lowerer.block, lowerer.context, lowerer.location)?;
            let output_tensor_type = lowerer.lower_tensor_type(&output_types[0])?;
            let dimensions = lowerer.context.stable_hlo_dot_dimensions(&[], &[], &[1], &[0]);
            let result = lowerer.block.append_operation(stable_hlo::dot_general(
                input_values[0],
                factor_value,
                dimensions,
                Some((Precision::Default, Precision::Default)),
                None,
                output_tensor_type,
                lowerer.location,
            ));
            Ok(vec![result.result(0).expect("stablehlo.dot_general should return one result").as_ref()])
        }
        XlaPrimitiveOperation::Reshape { output_type, .. } => {
            let output_shape = static_dimensions(output_type)?;
            let result = lowerer.block.append_operation(stable_hlo::reshape(
                input_values[0],
                output_shape.as_slice(),
                lowerer.location,
            ));
            Ok(vec![result.result(0).expect("stablehlo.reshape should return one result").as_ref()])
        }
        XlaPrimitiveOperation::Rematerialize(remat_op) => lowerer.lower_rematerialize(remat_op.as_ref(), input_values),
        XlaPrimitiveOperation::Condition(condition_op) => lowerer.lower_condition(condition_op.as_ref(), input_values),
        XlaPrimitiveOperation::While(while_op) => lowerer.lower_while(while_op.as_ref(), input_values),
        XlaPrimitiveOperation::ShardMap(shard_map_op) => {
            let simplified_body = shard_map_op
                .body()
                .simplified()
                .map_err(|error| LoweringError::SimplificationFailure { message: error.to_string() })?;
            lowerer.lower_manual_computation(
                input_values,
                &simplified_body.shard_map,
                &simplified_body.program,
                simplified_body.local_input_types.as_slice(),
                simplified_body.global_output_types.as_slice(),
            )
        }
        XlaPrimitiveOperation::LinearShardMap(shard_map_op) => {
            lowerer.lower_linear_shard_map_eval_mode(shard_map_op.eval_mode(), captured_values, input_values)
        }
        XlaPrimitiveOperation::WithShardingConstraint(op) => {
            let operation = lowerer.block.append_operation(shardy::sharding_constraint(
                input_values[0],
                op.sharding().to_mlir(lowerer.location),
                lowerer.location,
            ));
            Ok(vec![operation.result(0).expect("sdy.sharding_constraint should return one result").as_ref()])
        }
        XlaPrimitiveOperation::Custom(custom_op) => custom_op
            .extensions()
            .get::<LinearShardMapOperation<ShardMapTensor>>()
            .map(|shard_map_op| {
                lowerer.lower_linear_shard_map_eval_mode(shard_map_op.eval_mode(), captured_values, input_values)
            })
            .unwrap_or_else(|| {
                custom_op
                    .extensions()
                    .get::<StableHloCustomLoweringExtension<ShardMapTensor>>()
                    .ok_or_else(|| LoweringError::MissingCustomLowering { op: op.name().to_string() })?
                    .lower_to_mlir(custom_op.as_ref(), input_values, output_types, lowerer)
            }),
    }
}

/// Lowers one traced instruction to the corresponding StableHLO operation and returns its result value.
fn lower_instruction<'b, 'c: 'b, 't: 'c, ProgramInput, ProgramOutput>(
    program: &Program<ArrayType, ShardMapTensor, XlaPrimitiveOperation, ProgramInput, ProgramOutput>,
    instruction: &Instruction<XlaPrimitiveOperation>,
    atom_values: &[Option<ValueRef<'b, 'c, 't>>],
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    ProgramInput: Parameterized<ShardMapTensor>,
    ProgramOutput: Parameterized<ShardMapTensor>,
{
    let output_types = instruction
        .outputs
        .iter()
        .map(|output| program.atoms[output.index].r#type().into_owned())
        .collect::<Vec<_>>();
    let captured_values = match &instruction.operation {
        XlaPrimitiveOperation::LinearShardMap(shard_map_op) => shard_map_op
            .captured_global_primals()
            .iter()
            .map(|atom_id| atom_values[atom_id.index].ok_or(LoweringError::MissingAtomValue { atom_id: *atom_id }))
            .collect::<Result<Vec<_>, _>>()?,
        XlaPrimitiveOperation::Custom(custom_op) => custom_op
            .extensions()
            .get::<LinearShardMapOperation<ShardMapTensor>>()
            .map(|shard_map_op| {
                shard_map_op
                    .captured_global_primals()
                    .iter()
                    .map(|atom_id| {
                        atom_values[atom_id.index].ok_or(LoweringError::MissingAtomValue { atom_id: *atom_id })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?
            .unwrap_or_default(),
        _ => Vec::new(),
    };
    let mut lowerer = ShardMapMlirLowerer { block: *block, context, location };
    dispatch_lower_shard_map_mlir(
        &instruction.operation,
        captured_values.as_slice(),
        input_values,
        output_types.as_slice(),
        &mut lowerer,
    )
}

/// Normalizes a user-provided MLIR symbol name.
fn normalize_function_name(function_name: &str) -> Result<String, LoweringError> {
    let function_name = function_name.trim();
    if function_name.is_empty() || function_name.chars().any(char::is_whitespace) {
        return Err(LoweringError::InvalidFunctionName { function_name: function_name.to_string() });
    }
    Ok(function_name.strip_prefix('@').unwrap_or(function_name).to_string())
}

/// Lowers an [`ArrayType`] to a typed MLIR tensor type.
fn lower_tensor_type<'c, 't, L: Location<'c, 't>>(
    array_type: &ArrayType,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ryft_mlir::TensorTypeRef<'c, 't>, LoweringError> {
    let element_type = lower_element_type(array_type.data_type, context)?;
    let dimensions = array_type
        .shape
        .dimensions
        .iter()
        .map(|size| match size {
            Size::Static(value) => MlirSize::Static(*value),
            Size::Dynamic(_) => MlirSize::Dynamic,
        })
        .collect::<Vec<_>>();
    context
        .tensor_type(element_type, dimensions.as_slice(), None, location)
        .ok_or_else(|| LoweringError::InvalidTensorType { array_type: array_type.clone() })
}

/// Lowers one [`DataType`] to the corresponding MLIR element type.
fn lower_element_type<'c, 't>(
    data_type: DataType,
    context: &'c MlirContext<'t>,
) -> Result<TypeRef<'c, 't>, LoweringError> {
    Ok(match data_type {
        DataType::Token => return Err(LoweringError::UnsupportedDataType { data_type }),
        DataType::Boolean => context.signless_integer_type(1).as_ref(),
        DataType::I1 => context.signless_integer_type(1).as_ref(),
        DataType::I2 => context.signless_integer_type(2).as_ref(),
        DataType::I4 => context.signless_integer_type(4).as_ref(),
        DataType::I8 => context.signless_integer_type(8).as_ref(),
        DataType::I16 => context.signless_integer_type(16).as_ref(),
        DataType::I32 => context.signless_integer_type(32).as_ref(),
        DataType::I64 => context.signless_integer_type(64).as_ref(),
        DataType::U1 => context.unsigned_integer_type(1).as_ref(),
        DataType::U2 => context.unsigned_integer_type(2).as_ref(),
        DataType::U4 => context.unsigned_integer_type(4).as_ref(),
        DataType::U8 => context.unsigned_integer_type(8).as_ref(),
        DataType::U16 => context.unsigned_integer_type(16).as_ref(),
        DataType::U32 => context.unsigned_integer_type(32).as_ref(),
        DataType::U64 => context.unsigned_integer_type(64).as_ref(),
        DataType::BF16 => context.bfloat16_type().as_ref(),
        DataType::F16 => context.float16_type().as_ref(),
        DataType::F32 => context.float32_type().as_ref(),
        DataType::F64 => context.float64_type().as_ref(),
        DataType::F4E2M1FN => context.float4e2m1fn_type().as_ref(),
        DataType::F8E3M4 => context.float8e3m4_type().as_ref(),
        DataType::F8E4M3 => context.float8e4m3_type().as_ref(),
        DataType::F8E4M3FN => context.float8e4m3fn_type().as_ref(),
        DataType::F8E4M3FNUZ => context.float8e4m3fnuz_type().as_ref(),
        DataType::F8E4M3B11FNUZ => context.float8e4m3b11fnuz_type().as_ref(),
        DataType::F8E5M2 => context.float8e5m2_type().as_ref(),
        DataType::F8E5M2FNUZ => context.float8e5m2fnuz_type().as_ref(),
        DataType::F8E8M0FNU => context.float8e8m0fnu_type().as_ref(),
        DataType::C64 => context.complex_type(context.float32_type()).as_ref(),
        DataType::C128 => context.complex_type(context.float64_type()).as_ref(),
    })
}

/// Builds the dense-elements attribute for one traced splat constant.
fn lower_constant_elements_attribute<'c, 't>(
    data_type: DataType,
    tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
    constant_kind: ShardMapConstantKind,
    context: &'c MlirContext<'t>,
) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
    let integer_value = match constant_kind {
        ShardMapConstantKind::Zero => 0,
        ShardMapConstantKind::One => 1,
    };
    let float_value = integer_value as f64;

    match data_type {
        DataType::Boolean => context
            .splatted_dense_attribute_elements_attribute(tensor_type, context.boolean_attribute(integer_value != 0))
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::I1 | DataType::I2 | DataType::I4 | DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64 => {
            context
                .splatted_dense_attribute_elements_attribute(
                    tensor_type,
                    context.integer_attribute(
                        context.signless_integer_type(signed_integer_width(data_type)?),
                        integer_value,
                    ),
                )
                .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type })
        }
        DataType::U1 | DataType::U2 | DataType::U4 | DataType::U8 | DataType::U16 | DataType::U32 | DataType::U64 => {
            context
                .splatted_dense_attribute_elements_attribute(
                    tensor_type,
                    context.integer_attribute(
                        context.unsigned_integer_type(unsigned_integer_width(data_type)?),
                        integer_value,
                    ),
                )
                .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type })
        }
        DataType::BF16 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.bfloat16_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F16 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float16_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F32 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float32_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F64 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float64_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F4E2M1FN => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float4e2m1fn_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E3M4 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e3m4_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E4M3 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e4m3_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E4M3FN => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e4m3fn_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E4M3FNUZ => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e4m3fnuz_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E4M3B11FNUZ => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e4m3b11fnuz_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E5M2 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e5m2_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E5M2FNUZ => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e5m2fnuz_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E8M0FNU => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e8m0fnu_type(), float_value),
            )
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::Token | DataType::C64 | DataType::C128 => Err(LoweringError::UnsupportedDataType { data_type }),
    }
}

/// Returns the bit width of a signed integer [`DataType`].
fn signed_integer_width(data_type: DataType) -> Result<usize, LoweringError> {
    Ok(match data_type {
        DataType::I1 => 1,
        DataType::I2 => 2,
        DataType::I4 => 4,
        DataType::I8 => 8,
        DataType::I16 => 16,
        DataType::I32 => 32,
        DataType::I64 => 64,
        _ => return Err(LoweringError::UnsupportedDataType { data_type }),
    })
}

/// Returns the bit width of an unsigned integer [`DataType`].
fn unsigned_integer_width(data_type: DataType) -> Result<usize, LoweringError> {
    Ok(match data_type {
        DataType::U1 => 1,
        DataType::U2 => 2,
        DataType::U4 => 4,
        DataType::U8 => 8,
        DataType::U16 => 16,
        DataType::U32 => 32,
        DataType::U64 => 64,
        _ => return Err(LoweringError::UnsupportedDataType { data_type }),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    #[cfg(feature = "ndarray")]
    use ndarray::{Array2, arr2};

    use ryft_core::parameters::Placeholder;
    use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use ryft_core::tracing::{InterpretableOperation, Operation, ProgramBuilder, TracingError};
    use ryft_core::tracing_v2::{
        Cos, CustomPrimitive, MatrixOps, Sin,
        operations::constants::{OneLike, ZeroLike},
    };
    use ryft_core::types::{Shape, TypeError};

    use super::super::shard_map::{TracedShardMap, shard_map as traced_shard_map};
    use super::*;

    fn test_manual_mesh(axis_name: &str, axis_size: usize) -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::new(axis_name, axis_size, MeshAxisType::Manual).unwrap()]).unwrap()
    }

    fn test_vector_type(length: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(length)]), None, None).unwrap()
    }

    fn test_matrix_type(rows: usize, cols: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(rows), Size::Static(cols)]), None, None).unwrap()
    }

    fn xla_identity_branch(
        input_type: ArrayType,
    ) -> Program<ArrayType, ShardMapTensor, XlaPrimitiveOperation, Vec<ShardMapTensor>, Vec<ShardMapTensor>> {
        let mut builder = ProgramBuilder::<ArrayType, ShardMapTensor, XlaPrimitiveOperation>::new();
        let input = builder.add_input(input_type);
        builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn xla_neg_branch(
        input_type: ArrayType,
    ) -> Program<ArrayType, ShardMapTensor, XlaPrimitiveOperation, Vec<ShardMapTensor>, Vec<ShardMapTensor>> {
        let mut builder = ProgramBuilder::<ArrayType, ShardMapTensor, XlaPrimitiveOperation>::new();
        let input = builder.add_input(input_type);
        let output = builder.add_instruction(XlaPrimitiveOperation::Neg, vec![input]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn lower_traced_module(
        traced: &TracedShardMap<ArrayType, ArrayType>,
        function_name: &str,
    ) -> Result<String, super::super::shard_map::ShardMapTraceError> {
        traced.to_mlir_module(function_name)
    }

    #[derive(Clone, Debug)]
    struct TestCustomLoweredOp;

    impl std::fmt::Display for TestCustomLoweredOp {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "test_custom_lowered")
        }
    }

    impl Operation<ArrayType> for TestCustomLoweredOp {
        fn name(&self) -> &'static str {
            "test_custom_lowered"
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
            if input_types.len() != 1 {
                return Err(TypeError {
                    message: format!("test_custom_lowered expected 1 input type but got {}", input_types.len()),
                });
            }
            Ok(vec![input_types[0].clone()])
        }
    }

    impl InterpretableOperation<ArrayType, ShardMapTensor> for TestCustomLoweredOp {
        fn interpret(&self, inputs: &[ShardMapTensor]) -> Result<Vec<ShardMapTensor>, TracingError> {
            if inputs.len() != 1 {
                return Err(TracingError::InvalidInputCount { expected: 1, got: inputs.len() });
            }
            Ok(vec![inputs[0].clone()])
        }
    }

    struct TestCustomLowering;

    impl StableHloCustomLowering<ShardMapTensor> for TestCustomLowering {
        fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
            &self,
            _op: &CustomPrimitive<ArrayType, ShardMapTensor>,
            input_values: &[ValueRef<'b, 'c, 't>],
            _output_types: &[ArrayType],
            lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't>,
        ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
            let operation = lowerer.block.append_operation(stable_hlo::negate(input_values[0], lowerer.location));
            Ok(vec![operation.result(0).expect("stablehlo.negate should return one result").as_ref()])
        }
    }

    fn custom_program(
        op: XlaPrimitiveOperation,
    ) -> Program<ArrayType, ShardMapTensor, XlaPrimitiveOperation, ShardMapTensor, ShardMapTensor> {
        let input_type = test_vector_type(4);
        let mut builder =
            ProgramBuilder::<ArrayType, ShardMapTensor, crate::experimental::ops::XlaPrimitiveOperation>::new();
        let input = builder.add_input(input_type.clone());
        let output = builder.add_instruction(op, vec![input]).unwrap()[0];
        builder.build(vec![output], Placeholder, Placeholder).unwrap()
    }

    #[cfg(feature = "ndarray")]
    fn bilinear_matmul<M>(inputs: (M, M)) -> M
    where
        M: MatrixOps,
    {
        inputs.0.matmul(inputs.1)
    }

    #[test]
    fn test_to_mlir_module_renders_a_full_add_module() {
        let global_input_type = test_vector_type(8);
        let mesh = test_manual_mesh("x", 4);
        let traced: TracedShardMap<ArrayType, ArrayType> = traced_shard_map(
            |x| x.clone() + x,
            global_input_type,
            mesh.clone(),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap(),
        )
        .unwrap();

        assert_eq!(
            lower_traced_module(&traced, "main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = stablehlo.add %arg1, %arg1 : tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_to_mlir_module_renders_constants_and_supported_ops() {
        let global_input_type = test_matrix_type(4, 4);
        let mesh = test_manual_mesh("x", 2);
        let traced: TracedShardMap<ArrayType, ArrayType> = traced_shard_map(
            |x| {
                let product = x.clone().transpose_matrix().matmul(x);
                let waveform = (-product).cos().sin();
                (waveform.clone() * waveform.one_like()) + waveform.zero_like()
            },
            global_input_type,
            mesh.clone(),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap(),
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap(),
        )
        .unwrap();

        assert_eq!(
            lower_traced_module(&traced, "kernel").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @kernel(%arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}, {}]>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x"} (%arg1: tensor<2x4xf32>) {
                      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                      %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x4xf32>
                      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                      %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4x4xf32>
                      %3 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<2x4xf32>) -> tensor<4x2xf32>
                      %4 = stablehlo.dot_general %3, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
                      %5 = stablehlo.negate %4 : tensor<4x4xf32>
                      %6 = stablehlo.cosine %5 : tensor<4x4xf32>
                      %7 = stablehlo.sine %6 : tensor<4x4xf32>
                      %8 = stablehlo.multiply %7, %1 : tensor<4x4xf32>
                      %9 = stablehlo.add %8, %2 : tensor<4x4xf32>
                      sdy.return %9 : tensor<4x4xf32>
                    } : (tensor<4x4xf32>) -> tensor<8x4xf32>
                    return %0 : tensor<8x4xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_condition_to_stablehlo_if() {
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let input_type = ArrayType::scalar(DataType::F32);
        let condition = ConditionOperation::new(
            predicate_type.clone(),
            xla_neg_branch(input_type.clone()),
            xla_identity_branch(input_type.clone()),
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayType, ShardMapTensor, XlaPrimitiveOperation>::new();
        let predicate = builder.add_input(predicate_type);
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(XlaPrimitiveOperation::Condition(Box::new(condition)), vec![predicate, input])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert!(stablehlo.contains("\"stablehlo.if\""), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.negate"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.return"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_while_to_stablehlo_while() {
        let state_type = ArrayType::scalar(DataType::Boolean);
        let while_operation =
            WhileOperation::new(xla_identity_branch(state_type.clone()), xla_identity_branch(state_type.clone()))
                .unwrap();
        let mut builder = ProgramBuilder::<ArrayType, ShardMapTensor, XlaPrimitiveOperation>::new();
        let state = builder.add_input(state_type);
        let output = builder
            .add_instruction(XlaPrimitiveOperation::While(Box::new(while_operation)), vec![state])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert!(stablehlo.contains("stablehlo.while"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.return"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_uses_registered_custom_lowering() {
        let primitive = CustomPrimitive::new(TestCustomLoweredOp)
            .with_extension(StableHloCustomLoweringExtension::new(Arc::new(TestCustomLowering)));
        let program = custom_program(XlaPrimitiveOperation::Custom(Arc::new(primitive)));
        let input_type = test_vector_type(4);

        assert_eq!(
            to_mlir_module_for_program(&program, &input_type, &input_type, "main").unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.negate %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_reports_missing_custom_lowering() {
        let program =
            custom_program(XlaPrimitiveOperation::Custom(Arc::new(CustomPrimitive::new(TestCustomLoweredOp))));
        let input_type = test_vector_type(4);

        assert_eq!(
            to_mlir_module_for_program(&program, &input_type, &input_type, "main"),
            Err(LoweringError::MissingCustomLowering { op: "test_custom_lowered".to_string() }),
        );
    }

    // ---------------------------------------------------------------------------
    // Plain-program StableHLO lowering tests for scalar programs
    // ---------------------------------------------------------------------------

    fn scalar_bilinear_sin<T>(inputs: (T, T)) -> T
    where
        T: Clone + ryft_core::tracing_v2::Sin + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    {
        inputs.0.clone() * inputs.1 + inputs.0.sin()
    }

    fn scalar_quartic_plus_sin<T>(x: T) -> T
    where
        T: Clone + ryft_core::tracing_v2::Sin + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    {
        x.clone() * x.clone() * x.clone() * x.clone() + x.sin()
    }

    #[test]
    fn test_plain_scalar_bilinear_sin_jit_stablehlo() {
        let engine = ryft_core::tracing_v2::engines::ScalarEngine::<f64>::new();
        let (_, compiled): (
            f64,
            ryft_core::tracing::Program<
                ArrayType,
                f64,
                ryft_core::tracing_v2::PrimitiveOperation<f64>,
                (f64, f64),
                f64,
            >,
        ) = ryft_core::tracing_v2::interpret_and_trace(
            &engine,
            |inputs| Ok(scalar_bilinear_sin(inputs)),
            (2.0f64, 3.0f64),
        )
        .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&compiled, "main").unwrap();
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
                    %0 = stablehlo.multiply %arg0, %arg1 : tensor<f64>
                    %1 = stablehlo.sine %arg0 : tensor<f64>
                    %2 = stablehlo.add %0, %1 : tensor<f64>
                    return %2 : tensor<f64>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_plain_rematerialize_lowers_optimization_barrier() {
        let engine = ryft_core::tracing_v2::engines::ScalarEngine::<f64>::new();
        let (_, compiled): (
            f64,
            ryft_core::tracing::Program<ArrayType, f64, ryft_core::tracing_v2::PrimitiveOperation<f64>, f64, f64>,
        ) = ryft_core::tracing_v2::interpret_and_trace(
            &engine,
            |x| Ok(ryft_core::tracing_v2::rematerialize(|y| y.sin(), x).unwrap()),
            2.0f64,
        )
        .unwrap();

        assert_eq!(
            to_mlir_module_for_plain_program(&compiled, "main").unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>) -> tensor<f64> {
                    %0 = stablehlo.sine %arg0 : tensor<f64>
                    %1 = stablehlo.optimization_barrier %0 : tensor<f64>
                    return %1 : tensor<f64>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_plain_scalar_quartic_plus_sin_grad_stablehlo() {
        let engine = ryft_core::tracing_v2::engines::ScalarEngine::<f64>::new();
        let (_, compiled): (
            f64,
            ryft_core::tracing::Program<ArrayType, f64, ryft_core::tracing_v2::PrimitiveOperation<f64>, f64, f64>,
        ) = ryft_core::tracing_v2::interpret_and_trace(
            &engine,
            |x| {
                Ok(ryft_core::tracing_v2::grad(
                    &ryft_core::tracing_v2::engines::ScalarEngine::<f64>::new(),
                    scalar_quartic_plus_sin,
                    x,
                )?)
            },
            2.0f64,
        )
        .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&compiled, "main").unwrap();
        println!("=== ryft grad(x^4 + sin(x)) StableHLO ===\n{stablehlo}");

        // Verify key structural properties matching JAX's output:
        // 1. Single cosine for the sin(x) derivative
        assert_eq!(stablehlo.matches("stablehlo.cosine").count(), 1, "should have exactly one cosine");
        // 2. Multiple multiplies for the x^4 chain rule
        assert!(stablehlo.matches("stablehlo.multiply").count() >= 5, "should have several multiplies for x^4 chain");
        // 3. Multiple adds accumulating cotangent contributions
        assert!(stablehlo.matches("stablehlo.add").count() >= 3, "should have adds for cotangent accumulation");
        // 4. No sine in the gradient (it's consumed in forward, derivative is cosine)
        assert_eq!(stablehlo.matches("stablehlo.sine").count(), 0, "gradient should not contain sine");
    }

    #[test]
    fn test_plain_scalar_bilinear_sin_vjp_pullback_standalone_stablehlo() {
        // Standalone pullback â€” specialized to primal point (x=2.0, y=3.0), like JAX's standalone vjp_fn.
        let (_, pullback): (
            f64,
            ryft_core::tracing::Program<
                ArrayType,
                f64,
                ryft_core::tracing_v2::LinearPrimitiveOperation<f64>,
                f64,
                (f64, f64),
            >,
        ) = ryft_core::tracing_v2::vjp(
            &ryft_core::tracing_v2::engines::ScalarEngine::<f64>::new(),
            |inputs| Ok(scalar_bilinear_sin(inputs)),
            (2.0f64, 3.0f64),
        )
        .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&pullback, "main").unwrap();
        println!("=== ryft standalone vjp_pullback(x*y + sin(x)) StableHLO ===\n{stablehlo}");

        // Pullback takes one cotangent, returns two cotangent outputs (for x and y).
        assert!(stablehlo.contains("-> (tensor<f64>, tensor<f64>)"), "pullback should return two outputs");
        // Scale ops with baked-in primal values (cos(2.0), y=3.0, x=2.0) lower to multiply-by-constant.
        assert!(stablehlo.matches("stablehlo.constant").count() >= 2, "should have baked-in primal constants");
    }

    #[test]
    fn test_plain_scalar_bilinear_sin_grad_jitted_stablehlo() {
        // grad(f) wrapped in JIT â€” symbolic, like JAX's jit(grad(f)).
        // Uses the ValueAndGradInvocationLeaf<Tracer<V>> dispatch that traces through vjp+pullback.
        let engine = ryft_core::tracing_v2::engines::ScalarEngine::<f64>::new();
        let (_, compiled): (
            (f64, f64),
            ryft_core::tracing::Program<
                ArrayType,
                f64,
                ryft_core::tracing_v2::PrimitiveOperation<f64>,
                (f64, f64),
                (f64, f64),
            >,
        ) = ryft_core::tracing_v2::interpret_and_trace(
            &engine,
            |inputs| {
                Ok(ryft_core::tracing_v2::grad(
                    &ryft_core::tracing_v2::engines::ScalarEngine::<f64>::new(),
                    scalar_bilinear_sin,
                    inputs,
                )?)
            },
            (2.0f64, 3.0f64),
        )
        .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&compiled, "main").unwrap();
        println!("=== ryft jit(grad(bilinear_sin)) StableHLO ===\n{stablehlo}");

        // cos(x) should be computed symbolically from %arg0, NOT as a baked-in constant.
        assert!(stablehlo.contains("stablehlo.cosine %arg0"), "cos(x) should be computed from input");
        // Should reference both inputs.
        assert!(stablehlo.contains("%arg0") && stablehlo.contains("%arg1"), "should reference both inputs");
        // No sine (sin derivative = cosine, not sine).
        assert!(!stablehlo.contains("stablehlo.sine"), "gradient should not contain sine");
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_to_mlir_module_for_plain_program_renders_transposed_matrix_pullback_factors() {
        let left = arr2(&[[1.0f64, 2.0], [3.0, 4.0]]);
        let right = arr2(&[[5.0f64, 6.0], [7.0, 8.0]]);
        let (_, pullback): (
            Array2<f64>,
            ryft_core::tracing::Program<
                ArrayType,
                Array2<f64>,
                ryft_core::tracing_v2::LinearPrimitiveOperation<Array2<f64>>,
                Array2<f64>,
                (Array2<f64>, Array2<f64>),
            >,
        ) = ryft_core::tracing_v2::vjp(
            &ryft_core::tracing_v2::operations::matrix::ndarray_support::Array2Engine::<f64>::new(),
            |inputs| Ok(bilinear_matmul(inputs)),
            (left, right),
        )
        .unwrap();

        assert_eq!(
            to_mlir_module_for_plain_program(&pullback, "main").unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x2xf64>) -> (tensor<2x2xf64>, tensor<2x2xf64>) {
                    %cst = stablehlo.constant dense<[[5.000000e+00, 7.000000e+00], [6.000000e+00, 8.000000e+00]]> : tensor<2x2xf64>
                    %0 = stablehlo.dot_general %arg0, %cst, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
                    %cst_0 = stablehlo.constant dense<[[1.000000e+00, 3.000000e+00], [2.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>
                    %1 = stablehlo.dot_general %cst_0, %arg0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
                    return %0, %1 : tensor<2x2xf64>, tensor<2x2xf64>
                  }
                }
            "#}
        );
    }
}
