use std::collections::HashMap;

use ryft_mlir::dialects::stable_hlo::{Accuracy, Precision};
use ryft_mlir::dialects::{func, shardy, stable_hlo};
use ryft_mlir::{
    Attribute, Block, BlockRef, Context as MlirContext, DenseElementsAttributeRef, FloatTypeRef, IntegerTypeRef,
    Location, LocationRef, Operation as MlirOperation, Region, Size as MlirSize, TensorTypeRef, Type,
    TypeAndAttributes, TypeRef, Value, ValueRef,
};
#[cfg(feature = "ndarray")]
use ryft_ndarray::Array as NdArrayValue;

use ryft_core::macros::check_count;
use ryft_core::operations::Operation;
use ryft_core::operations::arithmetic::{
    AddOperation, DivOperation, MulOperation, NegOperation, ScaleOperation, SubOperation,
};
use ryft_core::operations::constants::ConstantLikeOperation;
use ryft_core::operations::trigonometric::{CosOperation, SinOperation};
use ryft_core::parameters::Parameterized;
use ryft_core::sharding::{LogicalMesh, Sharding, ShardingError};
use ryft_core::tracing::{AtomId, Instruction, Program, Traceable, TracingError};
use ryft_core::tracing_v2::operations::compare::CompareKind;
use ryft_core::tracing_v2::operations::control_flow::{ConditionOperation, ConditionPredicate, WhileOperation};
use ryft_core::tracing_v2::operations::logical::LogicalKind;
use ryft_core::tracing_v2::operations::reduce::ReductionKind;
use ryft_core::tracing_v2::operations::{
    BroadcastInDimOperation, DotOperation, LeftDotOperation, ReshapeOperation, RightDotOperation, TransposeOperation,
};
use ryft_core::tracing_v2::{ArrayOperation, DotOps, LinearArrayOperation, NoOperationExtension};
use ryft_core::types::{ArrayType, DataType, Size, Typed};

use crate::experimental::operations::LinearShardMapEvalMode;
use crate::experimental::ops::{LinearXlaOperationExtension, XlaOperation, XlaOperationExtension};
use crate::mlir::ToMlir;

use super::shard_map::{ShardMap, ShardMapConstantKind, ShardMapError, XlaValue};
/// Error type for StableHLO/Shardy lowering.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum LoweringError {
    /// Underlying shard-map error returned while building manual-computation attributes.
    #[error("{0}")]
    ShardMapError(#[from] ShardMapError),

    /// Underlying sharding error returned while building mesh or sharding attributes.
    #[error("{0}")]
    ShardingError(#[from] ShardingError),

    /// Underlying MLIR error returned while building or mutating MLIR objects.
    #[error("{0}")]
    MlirError(#[from] ryft_mlir::Error),

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

    /// Underlying tracing error returned while replaying a staged program through the generic
    /// [`Program::interpret_with`] domain.
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
    block: BlockRef<'b, 'c, 't>,

    /// MLIR context owning the block and created operations.
    context: &'c MlirContext<'t>,

    /// Shared MLIR location used for emitted operations.
    location: LocationRef<'c, 't>,
}

impl<'b, 'c: 'b, 't: 'c> PlainMlirLowerer<'b, 'c, 't> {
    /// Creates a plain MLIR lowerer for operations emitted into `block`.
    pub(crate) fn new(
        block: BlockRef<'b, 'c, 't>,
        context: &'c MlirContext<'t>,
        location: LocationRef<'c, 't>,
    ) -> Self {
        Self { block, context, location }
    }

    /// Returns the block receiving the lowered operations.
    #[allow(dead_code)]
    pub(crate) fn block_mut(&mut self) -> &mut BlockRef<'b, 'c, 't> {
        &mut self.block
    }

    /// Returns the MLIR context owning emitted operations.
    #[allow(dead_code)]
    pub(crate) fn context(&self) -> &'c MlirContext<'t> {
        self.context
    }

    /// Returns the shared MLIR location used for emitted operations.
    #[allow(dead_code)]
    pub(crate) fn location(&self) -> LocationRef<'c, 't> {
        self.location
    }

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

    /// Lowers one nested condition operation inside this lowering context.
    pub(crate) fn lower_condition<V: MlirLowerableValue, O>(
        &mut self,
        condition_op: &ConditionOperation<V, O, ArrayType>,
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        O: Clone + LowerableXlaOperation<V>,
    {
        lower_condition_to_if(condition_op, input_values, &mut self.block, self.context, self.location)
    }

    /// Lowers one nested while operation inside this lowering context.
    pub(crate) fn lower_while<V: MlirLowerableValue, O>(
        &mut self,
        while_op: &WhileOperation<V, O, ArrayType>,
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        O: Clone + LowerableXlaOperation<V>,
    {
        lower_while_to_while(while_op, input_values, &mut self.block, self.context, self.location)
    }
}

/// Operations that can be lowered to StableHLO for XLA compilation.
///
/// Implementing this trait makes an operation eligible for MLIR lowering via
/// [`to_mlir_module_for_plain_program`] and related entry points. The core [`ArrayOperation`] and
/// [`LinearArrayOperation`] enums provide the default blanket implementations, and backends can add
/// their own closed op carriers by implementing this trait for those enums.
pub(crate) trait LowerableXlaOperation<V: MlirLowerableValue>: Operation<ArrayType> {
    /// Lowers this operation to one or more StableHLO operations.
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>;
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for NoOperationExtension {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        _input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        _lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        match *self {}
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for AddOperation {
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
                .append_operation(stable_hlo::add(input_values[0], input_values[1], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.add should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for SubOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::subtract(
            input_values[0],
            input_values[1],
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.subtract should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for MulOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::multiply(
            input_values[0],
            input_values[1],
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.multiply should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for DivOperation {
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
                .append_operation(stable_hlo::divide(input_values[0], input_values[1], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.divide should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for NegOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::negate(input_values[0], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.negate should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for SinOperation {
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
                .append_operation(stable_hlo::sine(input_values[0], Accuracy::Default, lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.sine should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for CosOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::cosine(
            input_values[0],
            Accuracy::Default,
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.cosine should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for TransposeOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::transpose(
            input_values[0],
            self.permutation(),
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.transpose should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for DotOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let output_tensor_type = lowerer.lower_tensor_type(&output_types[0])?;
        let dimension_numbers = self.dimensions();
        let dimensions = lowerer.context.stable_hlo_dot_dimensions(
            dimension_numbers.lhs_batching_dimensions(),
            dimension_numbers.rhs_batching_dimensions(),
            dimension_numbers.lhs_contracting_dimensions(),
            dimension_numbers.rhs_contracting_dimensions(),
        )?;
        let result = lowerer.block.append_operation(stable_hlo::dot_general(
            input_values[0],
            input_values[1],
            dimensions,
            Some((Precision::Default, Precision::Default)),
            None,
            output_tensor_type,
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.dot_general should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for ScaleOperation<ArrayType, V> {
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
            )?)?;
            broadcast.result(0).expect("stablehlo.broadcast should return one result").as_ref()
        } else {
            factor_value
        };
        let result = lowerer.block.append_operation(stable_hlo::multiply(
            input_values[0],
            factor_broadcast,
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.multiply should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for ConstantLikeOperation<ArrayType, f64> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("input", input_values, 1, TracingError);
        check_count!("output", output_types, 1, TracingError);
        let output_type = &output_types[0];
        let output_tensor_type = lowerer.lower_tensor_type(output_type)?;
        let constant_value = lower_f64_constant_splat(
            *self.value(),
            output_type,
            output_tensor_type,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        Ok(vec![constant_value])
    }
}

impl<V: MlirLowerableValue + DotOps> LowerableXlaOperation<V> for LeftDotOperation<V> {
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
        let dimension_numbers = self.dimensions();
        let dimensions = lowerer.context.stable_hlo_dot_dimensions(
            dimension_numbers.lhs_batching_dimensions(),
            dimension_numbers.rhs_batching_dimensions(),
            dimension_numbers.lhs_contracting_dimensions(),
            dimension_numbers.rhs_contracting_dimensions(),
        )?;
        let result = lowerer.block.append_operation(stable_hlo::dot_general(
            factor_value,
            input_values[0],
            dimensions,
            Some((Precision::Default, Precision::Default)),
            None,
            output_tensor_type,
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.dot_general should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue + DotOps> LowerableXlaOperation<V> for RightDotOperation<V> {
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
        let dimension_numbers = self.dimensions();
        let dimensions = lowerer.context.stable_hlo_dot_dimensions(
            dimension_numbers.lhs_batching_dimensions(),
            dimension_numbers.rhs_batching_dimensions(),
            dimension_numbers.lhs_contracting_dimensions(),
            dimension_numbers.rhs_contracting_dimensions(),
        )?;
        let result = lowerer.block.append_operation(stable_hlo::dot_general(
            input_values[0],
            factor_value,
            dimensions,
            Some((Precision::Default, Precision::Default)),
            None,
            output_tensor_type,
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.dot_general should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for ReshapeOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("output", output_types, 1, TracingError);
        let output_type = &output_types[0];
        let output_shape = static_dimensions(output_type)?;
        let result = lowerer.block.append_operation(stable_hlo::reshape(
            input_values[0],
            output_shape.as_slice(),
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.reshape should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for BroadcastInDimOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("output", output_types, 1, TracingError);
        let output_tensor_type = lowerer.lower_tensor_type(&output_types[0])?;
        let result = lowerer.block.append_operation(stable_hlo::broadcast(
            input_values[0],
            output_tensor_type,
            self.broadcast_dimensions(),
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.broadcast_in_dim should return one result").as_ref()])
    }
}

fn lower_constant_output<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    output_types: &[ArrayType],
    constant_kind: ShardMapConstantKind,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    check_count!("output", output_types, 1, TracingError);
    let output_type = &output_types[0];
    let tensor_type = lower_tensor_type(output_type, context, location)?;
    if !output_type.shape().dimensions().is_empty() {
        let scalar_tensor_type =
            context.tensor_type(lower_element_type(output_type.data_type(), context)?, &[], None, location)?;
        let scalar_elements =
            lower_constant_elements_attribute(output_type.data_type(), scalar_tensor_type, constant_kind, context)?;
        let scalar_constant = block.append_operation(stable_hlo::constant(scalar_elements, location)?)?;
        let broadcast = block.append_operation(stable_hlo::broadcast(
            scalar_constant.result(0).unwrap().as_ref(),
            tensor_type,
            &[],
            location,
        )?)?;
        return Ok(vec![broadcast.result(0).expect("stablehlo.broadcast should return one result").as_ref()]);
    }
    let elements = lower_constant_elements_attribute(output_type.data_type(), tensor_type, constant_kind, context)?;
    let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
    Ok(vec![constant.result(0).expect("stablehlo.constant should return one result").as_ref()])
}

fn lower_like_constant<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayType],
    constant_kind: ShardMapConstantKind,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    if input_values.len() != 1 {
        return Err(TracingError::InvalidInputCount { expected: 1, got: input_values.len() }.into());
    }
    lower_constant_output(output_types, constant_kind, block, context, location)
}

impl LowerableXlaOperation<XlaValue<'static>> for XlaOperationExtension<XlaValue<'static>> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        match self {
            Self::ShardMap(shard_map_op) => {
                let simplified_body = shard_map_op
                    .body()
                    .simplified()
                    .map_err(|error| LoweringError::SimplificationFailure { message: error.to_string() })?;
                lower_manual_computation(
                    &mut lowerer.block,
                    input_values,
                    simplified_body.shard_map(),
                    simplified_body.program(),
                    simplified_body.local_input_types(),
                    simplified_body.global_output_types(),
                    lowerer.context,
                    lowerer.location,
                )
            }
            Self::LinearShardMap(shard_map_op) => lower_linear_shard_map_eval_mode(
                shard_map_op.linear_state().eval_mode(),
                &[],
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            Self::WithShardingConstraint(op) => {
                let mut shard_map_lowerer = ShardMapMlirLowerer::new(lowerer.block, lowerer.context, lowerer.location);
                op.lower_to_mlir(input_values, &mut shard_map_lowerer)
            }
        }
    }
}

impl<V: MlirLowerableValue, O> LowerableXlaOperation<V> for ConditionOperation<V, O, ArrayType>
where
    O: Clone + LowerableXlaOperation<V>,
{
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

impl<V: MlirLowerableValue, O> LowerableXlaOperation<V> for WhileOperation<V, O, ArrayType>
where
    O: Clone + LowerableXlaOperation<V>,
{
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

impl<V, Extension> LowerableXlaOperation<V> for ArrayOperation<V, ArrayType, Extension>
where
    V: DotOps,
    V: MlirLowerableValue,
    Extension: Clone + LowerableXlaOperation<V>,
{
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        match self {
            ArrayOperation::Zero(_) => {
                if !input_values.is_empty() {
                    return Err(TracingError::InvalidInputCount { expected: 0, got: input_values.len() }.into());
                }
                lower_constant_output(
                    output_types,
                    ShardMapConstantKind::Zero,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )
            }
            ArrayOperation::One(_) => {
                if !input_values.is_empty() {
                    return Err(TracingError::InvalidInputCount { expected: 0, got: input_values.len() }.into());
                }
                lower_constant_output(
                    output_types,
                    ShardMapConstantKind::One,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )
            }
            ArrayOperation::Add => <AddOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                &AddOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Sub => <SubOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                &SubOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Mul => <MulOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                &MulOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Div => <DivOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                &DivOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Neg => <NegOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                &NegOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Sin => <SinOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                &SinOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Cos => <CosOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                &CosOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::ZeroLike => lower_like_constant(
                input_values,
                output_types,
                ShardMapConstantKind::Zero,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            ArrayOperation::OneLike => lower_like_constant(
                input_values,
                output_types,
                ShardMapConstantKind::One,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            ArrayOperation::Transpose { permutation } => {
                <TransposeOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                    &TransposeOperation::new(permutation.clone()),
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            ArrayOperation::Dot { dimensions } => <DotOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                &DotOperation::new(dimensions.clone()),
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Scale { factor } => {
                <ScaleOperation<ArrayType, V> as LowerableXlaOperation<V>>::lower_to_mlir(
                    &ScaleOperation::new(factor.clone()),
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            ArrayOperation::ConstantLike { value } => {
                <ConstantLikeOperation<ArrayType, f64> as LowerableXlaOperation<V>>::lower_to_mlir(
                    &ConstantLikeOperation::<ArrayType, f64>::new(*value),
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            ArrayOperation::Reshape { input_shape, output_shape } => {
                <ReshapeOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                    &ReshapeOperation::new(input_shape.clone(), output_shape.clone()),
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            ArrayOperation::BroadcastInDim { target_type, broadcast_dimensions } => {
                <BroadcastInDimOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                    &BroadcastInDimOperation::new(target_type.clone(), broadcast_dimensions.clone()),
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            ArrayOperation::Reduce { axes, kind, .. } => {
                check_count!("output", output_types, 1, TracingError);
                let value = lower_reduce_to_mlir(
                    *kind,
                    axes.as_slice(),
                    input_values[0],
                    &output_types[0],
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?;
                Ok(vec![value])
            }
            ArrayOperation::Compare { kind } => {
                let value = lower_compare_to_mlir(
                    *kind,
                    input_values[0],
                    input_values[1],
                    &mut lowerer.block,
                    lowerer.location,
                )?;
                Ok(vec![value])
            }
            ArrayOperation::Logical { kind } => {
                let value = lower_logical_to_mlir(*kind, input_values, &mut lowerer.block, lowerer.location)?;
                Ok(vec![value])
            }
            ArrayOperation::Collective { .. } => {
                // Collectives are per-lane identity at the operation type level (the named axis
                // only exists physically inside a matching `BatchingDomain`). When the named-axis
                // `vmap` consumes the collective, the batching rule produces either a `Reduce`
                // op or an unchanged lane-uniform passthrough — so reaching this lowering site
                // means the staged Collective is acting as identity, which is the right
                // semantics outside the matching batching level. Future work will rewrite
                // collectives inside `BatchingDomain::stage` so they always lower to `Reduce`.
                check_count!("input", input_values, 1, TracingError);
                Ok(vec![input_values[0]])
            }
            ArrayOperation::Select => {
                let result = lowerer.block.append_operation(stable_hlo::select(
                    input_values[0],
                    input_values[1],
                    input_values[2],
                    lowerer.location,
                )?)?;
                Ok(vec![result.result(0).expect("stablehlo.select should return one result").as_ref()])
            }
            ArrayOperation::Condition(condition) => condition.lower_to_mlir(input_values, output_types, mode, lowerer),
            ArrayOperation::While(while_operation) => {
                while_operation.lower_to_mlir(input_values, output_types, mode, lowerer)
            }
            ArrayOperation::Extension(extension) => extension.lower_to_mlir(input_values, output_types, mode, lowerer),
        }
    }
}

impl<V, Extension> LowerableXlaOperation<V> for LinearArrayOperation<V, ArrayType, Extension>
where
    V: MlirLowerableValue + DotOps,
    Extension: Clone + LowerableXlaOperation<V>,
{
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        match self {
            LinearArrayOperation::Zero(_) => {
                if !input_values.is_empty() {
                    return Err(TracingError::InvalidInputCount { expected: 0, got: input_values.len() }.into());
                }
                lower_constant_output(
                    output_types,
                    ShardMapConstantKind::Zero,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )
            }
            LinearArrayOperation::One(_) => {
                if !input_values.is_empty() {
                    return Err(TracingError::InvalidInputCount { expected: 0, got: input_values.len() }.into());
                }
                lower_constant_output(
                    output_types,
                    ShardMapConstantKind::One,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )
            }
            LinearArrayOperation::ZeroLike => lower_like_constant(
                input_values,
                output_types,
                ShardMapConstantKind::Zero,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            LinearArrayOperation::OneLike => lower_like_constant(
                input_values,
                output_types,
                ShardMapConstantKind::One,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            LinearArrayOperation::Add => <AddOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                &AddOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            LinearArrayOperation::Sub => <SubOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                &SubOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            LinearArrayOperation::Neg => <NegOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                &NegOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            LinearArrayOperation::Transpose { permutation } => {
                <TransposeOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                    &TransposeOperation::new(permutation.clone()),
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            LinearArrayOperation::Scale { factor } => {
                <ScaleOperation<ArrayType, V> as LowerableXlaOperation<V>>::lower_to_mlir(
                    &ScaleOperation::new(factor.clone()),
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            LinearArrayOperation::ConstantLike { value } => {
                <ConstantLikeOperation<ArrayType, f64> as LowerableXlaOperation<V>>::lower_to_mlir(
                    &ConstantLikeOperation::<ArrayType, f64>::new(*value),
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            LinearArrayOperation::Mul => <MulOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                &MulOperation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            LinearArrayOperation::LeftDot { factor, dimensions } => {
                <LeftDotOperation<V> as LowerableXlaOperation<V>>::lower_to_mlir(
                    &LeftDotOperation::new(factor.clone(), dimensions.clone()),
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            LinearArrayOperation::RightDot { factor, dimensions } => {
                <RightDotOperation<V> as LowerableXlaOperation<V>>::lower_to_mlir(
                    &RightDotOperation::new(factor.clone(), dimensions.clone()),
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            LinearArrayOperation::Reshape { input_shape, output_shape } => {
                <ReshapeOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                    &ReshapeOperation::new(input_shape.clone(), output_shape.clone()),
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            LinearArrayOperation::BroadcastInDim { target_type, broadcast_dimensions } => {
                <BroadcastInDimOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                    &BroadcastInDimOperation::new(target_type.clone(), broadcast_dimensions.clone()),
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            LinearArrayOperation::Reduce { axes, kind, .. } => {
                check_count!("output", output_types, 1, TracingError);
                let value = lower_reduce_to_mlir(
                    *kind,
                    axes.as_slice(),
                    input_values[0],
                    &output_types[0],
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?;
                Ok(vec![value])
            }
            LinearArrayOperation::Condition(condition) => {
                condition.lower_to_mlir(input_values, output_types, mode, lowerer)
            }
            LinearArrayOperation::While(while_operation) => {
                while_operation.lower_to_mlir(input_values, output_types, mode, lowerer)
            }
            LinearArrayOperation::Extension(extension) => {
                extension.lower_to_mlir(input_values, output_types, mode, lowerer)
            }
        }
    }
}

impl LowerableXlaOperation<XlaValue<'static>> for LinearXlaOperationExtension<XlaValue<'static>> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        match self {
            Self::LinearShardMap(op) => {
                let mut shard_map_lowerer = ShardMapMlirLowerer::new(lowerer.block, lowerer.context, lowerer.location);
                shard_map_lowerer.lower_linear_shard_map_eval_mode(op.linear_state().eval_mode(), &[], input_values)
            }
            Self::WithShardingConstraint(op) => {
                let mut shard_map_lowerer = ShardMapMlirLowerer::new(lowerer.block, lowerer.context, lowerer.location);
                op.lower_to_mlir(input_values, &mut shard_map_lowerer)
            }
        }
    }
}

/// Lowering helper passed to op-owned traced XLA MLIR lowering hooks.
pub(crate) struct ShardMapMlirLowerer<'b, 'c: 'b, 't: 'c> {
    /// Owning block receiving the lowered operations.
    block: BlockRef<'b, 'c, 't>,

    /// MLIR context owning the block and created operations.
    context: &'c MlirContext<'t>,

    /// Shared MLIR location used for emitted operations.
    location: LocationRef<'c, 't>,
}

impl<'b, 'c: 'b, 't: 'c> ShardMapMlirLowerer<'b, 'c, 't> {
    /// Creates a shard-map MLIR lowerer for operations emitted into `block`.
    pub(crate) fn new(
        block: BlockRef<'b, 'c, 't>,
        context: &'c MlirContext<'t>,
        location: LocationRef<'c, 't>,
    ) -> Self {
        Self { block, context, location }
    }

    /// Returns the block receiving the lowered operations.
    pub(crate) fn block_mut(&mut self) -> &mut BlockRef<'b, 'c, 't> {
        &mut self.block
    }

    /// Returns the MLIR context owning emitted operations.
    #[allow(dead_code)]
    pub(crate) fn context(&self) -> &'c MlirContext<'t> {
        self.context
    }

    /// Returns the shared MLIR location used for emitted operations.
    pub(crate) fn location(&self) -> LocationRef<'c, 't> {
        self.location
    }

    /// Lowers one tensor type inside this lowering context.
    pub(crate) fn lower_tensor_type(
        &self,
        array_type: &ArrayType,
    ) -> Result<ryft_mlir::TensorTypeRef<'c, 't>, LoweringError> {
        lower_tensor_type(array_type, self.context, self.location)
    }

    /// Lowers one nested condition operation inside this lowering context.
    pub(crate) fn lower_condition<V: MlirLowerableValue, O>(
        &mut self,
        condition_op: &ConditionOperation<V, O, ArrayType>,
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        O: Clone + LowerableXlaOperation<V>,
    {
        lower_condition_to_if(condition_op, input_values, &mut self.block, self.context, self.location)
    }

    /// Lowers one nested while operation inside this lowering context.
    pub(crate) fn lower_while<V: MlirLowerableValue, O>(
        &mut self,
        while_op: &WhileOperation<V, O, ArrayType>,
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        O: Clone + LowerableXlaOperation<V>,
    {
        lower_while_to_while(while_op, input_values, &mut self.block, self.context, self.location)
    }

    /// Lowers one nested Shardy manual computation operation inside this lowering context.
    pub(crate) fn lower_manual_computation<
        'o,
        ProgramInput: Parameterized<XlaValue<'static>>,
        ProgramOutput: Parameterized<XlaValue<'static>>,
    >(
        &mut self,
        outer_inputs: &[ValueRef<'b, 'c, 't>],
        shard_map: &ShardMap,
        program: &Program<ArrayType, XlaValue<'static>, XlaOperation<'static>, ProgramInput, ProgramOutput>,
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
    'o,
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
    ProgramInput: Parameterized<XlaValue<'static>>,
    ProgramOutput: Parameterized<XlaValue<'static>>,
    S: AsRef<str>,
>(
    shard_map: &ShardMap,
    program: &Program<ArrayType, XlaValue<'static>, XlaOperation<'static>, ProgramInput, ProgramOutput>,
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
    let module = context.module(location)?;

    let global_input_tensor_types = global_input_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let global_output_tensor_types = global_output_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let mesh_operation = shard_map.mesh().to_mlir(location)?;
    module.body()?.append_operation(mesh_operation)?;

    let function_arguments = global_input_tensor_types
        .iter()
        .zip(shard_map.in_shardings().iter())
        .map(|(tensor_type, sharding)| {
            let sharding = sharding.to_mlir(location)?;
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
            let sharding = sharding.to_mlir(location)?;
            Ok(TypeAndAttributes {
                r#type: tensor_type.as_ref(),
                attributes: Some(HashMap::from([("sdy.sharding".into(), sharding.as_ref())])),
            })
        })
        .collect::<Result<Vec<_>, LoweringError>>()?;

    module.body()?.append_operation({
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
        function_block_ref.append_operation(func::r#return(manual_results.as_slice(), location)?)?;

        let mut function_region = context.region();
        function_region.append_block(function_block)?;
        func::func(
            function_name.as_str(),
            func::FuncAttributes { arguments: function_arguments, results: function_results, ..Default::default() },
            function_region,
            location,
        )?
    })?;

    if !module.verify()? {
        return Err(LoweringError::MlirVerificationFailure);
    }

    Ok(module.to_string())
}

/// Lowers an arbitrary traced XLA program to a textual StableHLO/Shardy MLIR module.
///
/// When `arg_shardings` and/or `result_shardings` are provided, the corresponding `sdy.sharding`
/// attribute is attached to each func argument or result, mirroring what the XLA SPMD partitioner
/// expects to drive per-device boundary slicing (including uneven splits). When `None`, the func
/// signature has no sharding attributes — the legacy behavior used by traced programs that don't
/// participate in SPMD compilation.
pub(crate) fn to_mlir_module_for_program<'o, Input, Output, ProgramInput, ProgramOutput, S>(
    program: &Program<ArrayType, XlaValue<'static>, XlaOperation<'static>, ProgramInput, ProgramOutput>,
    global_input_types: &Input,
    global_output_types: &Output,
    function_name: S,
    arg_shardings: Option<&[Sharding]>,
    result_shardings: Option<&[Sharding]>,
) -> Result<String, LoweringError>
where
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
    ProgramInput: Parameterized<XlaValue<'static>>,
    ProgramOutput: Parameterized<XlaValue<'static>>,
    S: AsRef<str>,
{
    let function_name = normalize_function_name(function_name.as_ref())?;
    let global_input_types = global_input_types.parameters().cloned().collect::<Vec<_>>();
    let global_output_types = global_output_types.parameters().cloned().collect::<Vec<_>>();

    let context = MlirContext::new();
    let location = context.unknown_location();
    let module = context.module(location)?;

    // Emit `sdy.mesh` declarations for any sharding referenced either by inner ops or by the
    // optional signature shardings, so the func attributes can refer to `@mesh`.
    let mut signature_mesh = None;
    for sharding in arg_shardings.into_iter().flatten().chain(result_shardings.into_iter().flatten()) {
        if signature_mesh.is_none() {
            signature_mesh = Some(sharding.mesh().clone());
            break;
        }
    }
    let nested_mesh = collect_nested_sharding_mesh(program, None)?;
    if let Some(mesh) = nested_mesh.as_ref().or(signature_mesh.as_ref()) {
        let mesh_operation = mesh.to_mlir(location)?;
        module.body()?.append_operation(mesh_operation)?;
    }

    let global_input_tensor_types = global_input_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let global_output_tensor_types = global_output_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let arg_sharding_attributes = match arg_shardings {
        Some(shardings) => {
            Some(shardings.iter().map(|sharding| sharding.to_mlir(location)).collect::<Result<Vec<_>, _>>()?)
        }
        None => None,
    };
    let result_sharding_attributes = match result_shardings {
        Some(shardings) => {
            Some(shardings.iter().map(|sharding| sharding.to_mlir(location)).collect::<Result<Vec<_>, _>>()?)
        }
        None => None,
    };
    let function_arguments = global_input_tensor_types
        .iter()
        .enumerate()
        .map(|(index, tensor_type)| {
            let attributes = arg_sharding_attributes
                .as_ref()
                .map(|shardings| HashMap::from([("sdy.sharding".into(), shardings[index].as_ref())]));
            TypeAndAttributes { r#type: tensor_type.as_ref(), attributes }
        })
        .collect::<Vec<_>>();
    let function_results = global_output_tensor_types
        .iter()
        .enumerate()
        .map(|(index, tensor_type)| {
            let attributes = result_sharding_attributes
                .as_ref()
                .map(|shardings| HashMap::from([("sdy.sharding".into(), shardings[index].as_ref())]));
            TypeAndAttributes { r#type: tensor_type.as_ref(), attributes }
        })
        .collect::<Vec<_>>();

    module.body()?.append_operation({
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
            function_block_ref.append_operation(func::r#return(outputs.as_slice(), location)?)?;
        }
        let mut function_region = context.region();
        function_region.append_block(function_block)?;
        func::func(
            function_name.as_str(),
            func::FuncAttributes { arguments: function_arguments, results: function_results, ..Default::default() },
            function_region,
            location,
        )?
    })?;

    if !module.verify()? {
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

#[cfg(feature = "ndarray")]
impl MlirLowerableValue for NdArrayValue<f64> {
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
        let standard_layout = self.as_ndarray().as_standard_layout();
        let elements = standard_layout.iter().copied().collect::<Vec<_>>();
        let attribute = context
            .dense_f64_elements_attribute(tensor_type, elements.as_slice())
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: DataType::F64 })?;
        attribute
            .cast::<DenseElementsAttributeRef>()
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type: DataType::F64 })
    }

    fn to_scalar_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<Option<DenseElementsAttributeRef<'c, 't>>, LoweringError> {
        let Some(element) = self.as_ndarray().iter().next().filter(|_| self.as_ndarray().len() == 1) else {
            return Ok(None);
        };
        let attribute = context
            .dense_f64_elements_attribute(tensor_type, std::slice::from_ref(element))
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: DataType::F64 })?;
        Ok(Some(
            attribute
                .cast::<DenseElementsAttributeRef>()
                .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type: DataType::F64 })?,
        ))
    }
}

impl MlirLowerableValue for XlaValue<'static> {
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
        let constant_kind =
            self.constant_kind().ok_or(LoweringError::UnsupportedConstant { atom_id: AtomId::new(0) })?;
        lower_constant_elements_attribute(self.r#type().data_type(), tensor_type, constant_kind, context)
    }

    fn to_scalar_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<Option<DenseElementsAttributeRef<'c, 't>>, LoweringError> {
        let Some(constant_kind) = self.constant_kind() else {
            return Ok(None);
        };
        Ok(Some(lower_constant_elements_attribute(self.r#type().data_type(), tensor_type, constant_kind, context)?))
    }
}

/// Lowers a plain traced `tracing_v2` program to a textual StableHLO MLIR module.
#[cfg(any(test, feature = "benchmarking"))]
#[allow(dead_code)]
pub(crate) fn to_mlir_module_for_plain_program<
    V: MlirLowerableValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
    O: Clone + LowerableXlaOperation<V>,
    S: AsRef<str>,
>(
    program: &Program<ArrayType, V, O, Input, Output>,
    function_name: S,
) -> Result<String, LoweringError> {
    let function_name = normalize_function_name(function_name.as_ref())?;
    let context = MlirContext::new();
    let location = context.unknown_location();
    let module = context.module(location)?;

    let input_tensor_types = program
        .input_ids()
        .iter()
        .map(|atom_id| {
            let input_atom = &program.atoms()[atom_id.index()];
            lower_tensor_type(&input_atom.r#type(), &context, location)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output_tensor_types = program
        .output_ids()
        .iter()
        .map(|atom_id| {
            let output_atom = &program.atoms()[atom_id.index()];
            lower_tensor_type(&output_atom.r#type(), &context, location)
        })
        .collect::<Result<Vec<_>, _>>()?;

    module.body()?.append_operation({
        let function_block = context.block(
            input_tensor_types.iter().map(|tensor_type| (*tensor_type, location)).collect::<Vec<_>>().as_slice(),
        );
        {
            let mut function_block_ref = function_block.as_ref();
            let outputs = lower_plain_program_outputs(program, &mut function_block_ref, &context, location.as_ref())?;
            function_block_ref.append_operation(func::r#return(outputs.as_slice(), location)?)?;
        }
        let mut function_region = context.region();
        function_region.append_block(function_block)?;
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
            function_region,
            location,
        )?
    })?;

    if !module.verify()? {
        return Err(LoweringError::MlirVerificationFailure);
    }

    Ok(module.to_string())
}

fn collect_nested_sharding_mesh<ProgramInput, ProgramOutput>(
    program: &Program<ArrayType, XlaValue<'static>, XlaOperation<'static>, ProgramInput, ProgramOutput>,
    existing: Option<LogicalMesh>,
) -> Result<Option<LogicalMesh>, LoweringError>
where
    ProgramInput: Parameterized<XlaValue<'static>>,
    ProgramOutput: Parameterized<XlaValue<'static>>,
{
    let mut mesh = existing;
    for instruction in program.instructions() {
        match &instruction.operation() {
            XlaOperation::Extension(XlaOperationExtension::ShardMap(shard_map_op)) => {
                let body = shard_map_op.body();
                mesh = Some(match mesh.take() {
                    Some(existing_mesh) => merge_logical_meshes(&existing_mesh, body.shard_map().mesh())?,
                    None => body.shard_map().mesh().clone(),
                });
                mesh = collect_nested_sharding_mesh(body.program(), mesh)?;
            }
            XlaOperation::Extension(XlaOperationExtension::LinearShardMap(shard_map_op)) => {
                mesh = collect_nested_linear_shard_map_mesh(shard_map_op.linear_state().eval_mode(), mesh)?;
            }
            XlaOperation::Condition(condition_op) => {
                mesh = collect_nested_sharding_mesh(condition_op.true_branch(), mesh)?;
                mesh = collect_nested_sharding_mesh(condition_op.false_branch(), mesh)?;
            }
            XlaOperation::While(while_op) => {
                mesh = collect_nested_sharding_mesh(while_op.condition(), mesh)?;
                mesh = collect_nested_sharding_mesh(while_op.body(), mesh)?;
            }
            XlaOperation::Extension(XlaOperationExtension::WithShardingConstraint(sharding_constraint_op)) => {
                mesh = Some(match mesh.take() {
                    Some(existing_mesh) => {
                        merge_logical_meshes(&existing_mesh, sharding_constraint_op.sharding().mesh())?
                    }
                    None => sharding_constraint_op.sharding().mesh().clone(),
                });
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
                Some(existing_mesh) => merge_logical_meshes(&existing_mesh, body.shard_map().mesh())?,
                None => body.shard_map().mesh().clone(),
            });
            collect_nested_sharding_mesh(body.program(), mesh)
        }
        LinearShardMapEvalMode::FactorizedTranspose(factorized) => {
            let residual_body = factorized.residual_body();
            let mesh = Some(match existing {
                Some(existing_mesh) => merge_logical_meshes(&existing_mesh, residual_body.shard_map().mesh())?,
                None => residual_body.shard_map().mesh().clone(),
            });
            let mesh = collect_nested_sharding_mesh(residual_body.program(), mesh)?;
            let apply_body = factorized.apply_body();
            let mesh = Some(match mesh {
                Some(existing_mesh) => merge_logical_meshes(&existing_mesh, apply_body.shard_map().mesh())?,
                None => apply_body.shard_map().mesh().clone(),
            });
            collect_nested_sharding_mesh(apply_body.program(), mesh)
        }
    }
}

fn merge_logical_meshes(existing: &LogicalMesh, incoming: &LogicalMesh) -> Result<LogicalMesh, LoweringError> {
    let mut merged_axes = existing.axes().to_vec();
    for incoming_axis in incoming.axes() {
        match existing.axis_size(incoming_axis.name()) {
            Some(existing_size) if existing_size != incoming_axis.size() => {
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
        .shape()
        .dimensions()
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
    O: Clone + LowerableXlaOperation<V>,
{
    let mut region = context.region();
    let block = context.block_with_no_arguments();
    {
        let mut block_ref = block.as_ref();
        let outputs = lower_nested_program_inline(program, input_values, &mut block_ref, context, location, false)?;
        block_ref.append_operation(stable_hlo::r#return(outputs.as_slice(), location)?)?;
    }
    region.append_block(block)?;
    Ok(region)
}

fn lower_condition_to_if<'b, 'c: 'b, 't: 'c, V, O>(
    condition_op: &ConditionOperation<V, O, ArrayType>,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: MlirLowerableValue,
    O: Clone + LowerableXlaOperation<V>,
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
            )?)?;
            Ok((0..condition_op.output_types().len())
                .map(|index| {
                    operation.result(index).expect("stablehlo.if should return one result per output").as_ref()
                })
                .collect())
        }
    }
}

fn lower_while_to_while<'b, 'c: 'b, 't: 'c, V, O>(
    while_op: &WhileOperation<V, O, ArrayType>,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: MlirLowerableValue,
    O: Clone + LowerableXlaOperation<V>,
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
        condition_block_ref.append_operation(stable_hlo::r#return(condition_outputs.as_slice(), location)?)?;
    }
    condition_region.append_block(condition_block)?;

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
        body_block_ref.append_operation(stable_hlo::r#return(body_outputs.as_slice(), location)?)?;
    }
    body_region.append_block(body_block)?;

    let operation = block.append_operation(stable_hlo::r#while(
        input_values,
        condition_region.into(),
        body_region.into(),
        location,
    )?)?;
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
    O: Clone + LowerableXlaOperation<V>,
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
                .outputs()
                .iter()
                .map(|output| program.atoms()[output.index()].r#type().into_owned())
                .collect::<Vec<_>>();
            let mut lowerer = PlainMlirLowerer::new(*block, context, location);
            instruction.operation().lower_to_mlir(
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
    let barrier = block.append_operation(stable_hlo::optimization_barrier(outputs.as_slice(), location)?)?;
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
    O: Clone + LowerableXlaOperation<V>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    let input_values = (0..program.input_ids().len())
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
                .outputs()
                .iter()
                .map(|output| program.atoms()[output.index()].r#type().into_owned())
                .collect::<Vec<_>>();
            let mut lowerer = PlainMlirLowerer::new(*block, context, location);
            instruction.operation().lower_to_mlir(
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
    program: &Program<ArrayType, XlaValue<'static>, XlaOperation<'static>, ProgramInput, ProgramOutput>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    ProgramInput: Parameterized<XlaValue<'static>>,
    ProgramOutput: Parameterized<XlaValue<'static>>,
{
    // Mirror table of every lowered atom value. Shard-map operations look up captured global primals by `AtomId`,
    // so we keep a parallel table alongside [`Program::interpret_with`]'s use-count-tracked one. [`ValueRef`] is
    // `Copy`, so this mirror is cheap.
    let mut atom_values = vec![None; program.atoms().len()];
    let input_values = program
        .input_ids()
        .iter()
        .copied()
        .enumerate()
        .map(|(index, atom_id)| {
            let value = block.argument(index).expect("body block arguments should exist").as_ref();
            atom_values[atom_id.index()] = Some(value);
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
            atom_values.borrow_mut()[atom_id.index()] = Some(lowered);
            Ok(lowered)
        },
        |instruction, inputs, block, context, location| {
            let mut table = atom_values.borrow_mut();
            let lowered_outputs =
                lower_instruction(program, instruction, table.as_slice(), inputs, block, context, location)?;
            for (output_atom, lowered_output) in
                instruction.outputs().iter().copied().zip(lowered_outputs.iter().copied())
            {
                table[output_atom.index()] = Some(lowered_output);
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
    program: &Program<ArrayType, XlaValue<'static>, XlaOperation<'static>, ProgramInput, ProgramOutput>,
    local_input_types: &[ArrayType],
    global_output_types: &[ArrayType],
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    ProgramInput: Parameterized<XlaValue<'static>>,
    ProgramOutput: Parameterized<XlaValue<'static>>,
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
        body_block_ref.append_operation(shardy::r#return(body_outputs.as_slice(), location)?)?;
    }
    body_region.append_block(body_block)?;

    let manual_computation = block.append_operation(shardy::manual_computation(
        outer_inputs,
        global_output_tensor_types.as_slice(),
        shard_map.to_shardy_in_shardings(context)?,
        shard_map.to_shardy_out_shardings(context)?,
        shard_map.to_shardy_manual_axes(context)?,
        body_region,
        location,
    )?)?;
    manual_computation
        .results()
        .map(|result| result.map(|result| result.as_ref()).map_err(LoweringError::from))
        .collect()
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
                simplified_body.shard_map(),
                simplified_body.program(),
                simplified_body.local_input_types(),
                simplified_body.global_output_types(),
                context,
                location,
            )
        }
        LinearShardMapEvalMode::FactorizedTranspose(factorized) => {
            let residual_body = factorized
                .residual_body()
                .simplified()
                .map_err(|error| LoweringError::SimplificationFailure { message: error.to_string() })?;
            let residual_results = lower_manual_computation(
                block,
                &captured_values[..residual_body.global_input_types().len()],
                residual_body.shard_map(),
                residual_body.program(),
                residual_body.local_input_types(),
                residual_body.global_output_types(),
                context,
                location,
            )?;
            let apply_body = factorized
                .apply_body()
                .simplified()
                .map_err(|error| LoweringError::SimplificationFailure { message: error.to_string() })?;
            let apply_inputs = input_values
                .iter()
                .copied()
                .take(apply_body.global_input_types().len() - residual_results.len())
                .chain(residual_results)
                .collect::<Vec<_>>();
            lower_manual_computation(
                block,
                apply_inputs.as_slice(),
                apply_body.shard_map(),
                apply_body.program(),
                apply_body.local_input_types(),
                apply_body.global_output_types(),
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
    L: Copy + Location<'c, 't>,
{
    let value_type = value.r#type();
    if !value_type.shape().dimensions().is_empty() {
        let scalar_tensor_type = context
            .tensor_type(lower_element_type(value_type.data_type(), context)?, &[], None, location)
            .map_err(|_| LoweringError::InvalidTensorType { array_type: ArrayType::scalar(value_type.data_type()) })?;
        if let Some(scalar_elements) = value.to_scalar_dense_elements_attribute(scalar_tensor_type, context)? {
            let scalar_constant = block.append_operation(stable_hlo::constant(scalar_elements, location)?)?;
            let tensor_type = lower_tensor_type(&value_type, context, location)?;
            let broadcast = block.append_operation(stable_hlo::broadcast(
                scalar_constant.result(0).unwrap().as_ref(),
                tensor_type,
                &[],
                location,
            )?)?;
            return Ok(broadcast.result(0).expect("stablehlo.broadcast should return one result").as_ref());
        }
    }

    let tensor_type = lower_tensor_type(&value_type, context, location)?;
    let elements = value.to_dense_elements_attribute(tensor_type, context)?;
    let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
    Ok(constant.result(0).expect("stablehlo.constant should return one result").as_ref())
}

/// Lowers a traced constant atom to a StableHLO constant operation and returns its result value.
fn lower_constant<'b, 'c: 'b, 't: 'c, B, L>(
    atom_id: AtomId,
    value: &XlaValue,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    let constant_kind = value.constant_kind().ok_or(LoweringError::UnsupportedConstant { atom_id })?;
    let array_type = value.r#type();
    let tensor_type = lower_tensor_type(&array_type, context, location)?;
    if !array_type.shape().dimensions().is_empty() {
        let scalar_tensor_type = context
            .tensor_type(lower_element_type(array_type.data_type(), context)?, &[], None, location)
            .map_err(|_| LoweringError::InvalidTensorType { array_type: ArrayType::scalar(array_type.data_type()) })?;
        let scalar_elements =
            lower_constant_elements_attribute(array_type.data_type(), scalar_tensor_type, constant_kind, context)?;
        let scalar_constant = block.append_operation(stable_hlo::constant(scalar_elements, location)?)?;
        let broadcast = block.append_operation(stable_hlo::broadcast(
            scalar_constant.result(0).unwrap().as_ref(),
            tensor_type,
            &[],
            location,
        )?)?;
        return Ok(broadcast.result(0).expect("stablehlo.broadcast should return one result").as_ref());
    }
    let elements = lower_constant_elements_attribute(array_type.data_type(), tensor_type, constant_kind, context)?;
    let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
    Ok(constant.result(0).expect("stablehlo.constant should return one result").as_ref())
}

/// Dispatches shard-map StableHLO lowering for one traced operation by matching on primitive variants.
fn dispatch_lower_shard_map_mlir<'b, 'c: 'b, 't: 'c>(
    op: &XlaOperation<'static>,
    captured_values: &[ValueRef<'b, 'c, 't>],
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayType],
    lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    match op {
        XlaOperation::Zero(_) => {
            if !input_values.is_empty() {
                return Err(TracingError::InvalidInputCount { expected: 0, got: input_values.len() }.into());
            }
            lower_constant_output(
                output_types,
                ShardMapConstantKind::Zero,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )
        }
        XlaOperation::One(_) => {
            if !input_values.is_empty() {
                return Err(TracingError::InvalidInputCount { expected: 0, got: input_values.len() }.into());
            }
            lower_constant_output(
                output_types,
                ShardMapConstantKind::One,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )
        }
        XlaOperation::Add => {
            let result =
                lowerer
                    .block
                    .append_operation(stable_hlo::add(input_values[0], input_values[1], lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.add should return one result").as_ref()])
        }
        XlaOperation::Sub => {
            let result = lowerer.block.append_operation(stable_hlo::subtract(
                input_values[0],
                input_values[1],
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.subtract should return one result").as_ref()])
        }
        XlaOperation::Mul => {
            let result = lowerer.block.append_operation(stable_hlo::multiply(
                input_values[0],
                input_values[1],
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.multiply should return one result").as_ref()])
        }
        XlaOperation::Div => {
            let result = lowerer.block.append_operation(stable_hlo::divide(
                input_values[0],
                input_values[1],
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.divide should return one result").as_ref()])
        }
        XlaOperation::Neg => {
            let result = lowerer.block.append_operation(stable_hlo::negate(input_values[0], lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.negate should return one result").as_ref()])
        }
        XlaOperation::Sin => {
            let result = lowerer.block.append_operation(stable_hlo::sine(
                input_values[0],
                Accuracy::Default,
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.sine should return one result").as_ref()])
        }
        XlaOperation::Cos => {
            let result = lowerer.block.append_operation(stable_hlo::cosine(
                input_values[0],
                Accuracy::Default,
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.cosine should return one result").as_ref()])
        }
        XlaOperation::ZeroLike => lower_like_constant(
            input_values,
            output_types,
            ShardMapConstantKind::Zero,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        ),
        XlaOperation::OneLike => lower_like_constant(
            input_values,
            output_types,
            ShardMapConstantKind::One,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        ),
        XlaOperation::Dot { dimensions } => {
            let output_tensor_type = lowerer.lower_tensor_type(&output_types[0])?;
            let dimensions_attribute = lowerer.context.stable_hlo_dot_dimensions(
                dimensions.lhs_batching_dimensions(),
                dimensions.rhs_batching_dimensions(),
                dimensions.lhs_contracting_dimensions(),
                dimensions.rhs_contracting_dimensions(),
            )?;
            let result = lowerer.block.append_operation(stable_hlo::dot_general(
                input_values[0],
                input_values[1],
                dimensions_attribute,
                Some((Precision::Default, Precision::Default)),
                None,
                output_tensor_type,
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.dot_general should return one result").as_ref()])
        }
        XlaOperation::Transpose { permutation } => {
            let result = lowerer.block.append_operation(stable_hlo::transpose(
                input_values[0],
                permutation.as_slice(),
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.transpose should return one result").as_ref()])
        }
        XlaOperation::Scale { factor } => {
            let output_tensor_type = lowerer.lower_tensor_type(&output_types[0])?;
            let factor_value =
                lower_constant(AtomId::new(0), factor, &mut lowerer.block, lowerer.context, lowerer.location)?;
            let factor_type = factor.r#type();
            let factor_broadcast = if *factor_type != output_types[0] {
                let broadcast = lowerer.block.append_operation(stable_hlo::broadcast(
                    factor_value,
                    output_tensor_type,
                    &[],
                    lowerer.location,
                )?)?;
                broadcast.result(0).expect("stablehlo.broadcast should return one result").as_ref()
            } else {
                factor_value
            };
            let result = lowerer.block.append_operation(stable_hlo::multiply(
                input_values[0],
                factor_broadcast,
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.multiply should return one result").as_ref()])
        }
        XlaOperation::ConstantLike { value } => {
            let output_tensor_type = lowerer.lower_tensor_type(&output_types[0])?;
            let constant_value = lower_f64_constant_splat(
                *value,
                &output_types[0],
                output_tensor_type,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            Ok(vec![constant_value])
        }
        XlaOperation::Reshape { .. } => {
            check_count!("output", output_types, 1, TracingError);
            let output_type = &output_types[0];
            let output_shape = static_dimensions(output_type)?;
            let result = lowerer.block.append_operation(stable_hlo::reshape(
                input_values[0],
                output_shape.as_slice(),
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.reshape should return one result").as_ref()])
        }
        XlaOperation::BroadcastInDim { broadcast_dimensions, .. } => {
            check_count!("output", output_types, 1, TracingError);
            let output_tensor_type = lowerer.lower_tensor_type(&output_types[0])?;
            let result = lowerer.block.append_operation(stable_hlo::broadcast(
                input_values[0],
                output_tensor_type,
                broadcast_dimensions.as_slice(),
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.broadcast_in_dim should return one result").as_ref()])
        }
        XlaOperation::Reduce { axes, kind, .. } => {
            check_count!("output", output_types, 1, TracingError);
            let value = lower_reduce_to_mlir(
                *kind,
                axes.as_slice(),
                input_values[0],
                &output_types[0],
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            Ok(vec![value])
        }
        XlaOperation::Compare { kind } => {
            let value =
                lower_compare_to_mlir(*kind, input_values[0], input_values[1], &mut lowerer.block, lowerer.location)?;
            Ok(vec![value])
        }
        XlaOperation::Logical { kind } => {
            let value = lower_logical_to_mlir(*kind, input_values, &mut lowerer.block, lowerer.location)?;
            Ok(vec![value])
        }
        XlaOperation::Collective { .. } => {
            check_count!("input", input_values, 1, TracingError);
            Ok(vec![input_values[0]])
        }
        XlaOperation::Select => {
            let result = lowerer.block.append_operation(stable_hlo::select(
                input_values[0],
                input_values[1],
                input_values[2],
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.select should return one result").as_ref()])
        }
        XlaOperation::Condition(condition_op) => lowerer.lower_condition(condition_op.as_ref(), input_values),
        XlaOperation::While(while_op) => lowerer.lower_while(while_op.as_ref(), input_values),
        XlaOperation::Extension(XlaOperationExtension::ShardMap(shard_map_op)) => {
            let simplified_body = shard_map_op
                .body()
                .simplified()
                .map_err(|error| LoweringError::SimplificationFailure { message: error.to_string() })?;
            lowerer.lower_manual_computation(
                input_values,
                simplified_body.shard_map(),
                simplified_body.program(),
                simplified_body.local_input_types(),
                simplified_body.global_output_types(),
            )
        }
        XlaOperation::Extension(XlaOperationExtension::LinearShardMap(shard_map_op)) => lowerer
            .lower_linear_shard_map_eval_mode(shard_map_op.linear_state().eval_mode(), captured_values, input_values),
        XlaOperation::Extension(XlaOperationExtension::WithShardingConstraint(op)) => {
            op.lower_to_mlir(input_values, lowerer)
        }
    }
}

/// Lowers one traced instruction to the corresponding StableHLO operation and returns its result value.
fn lower_instruction<'b, 'c: 'b, 't: 'c, ProgramInput, ProgramOutput>(
    program: &Program<ArrayType, XlaValue<'static>, XlaOperation<'static>, ProgramInput, ProgramOutput>,
    instruction: &Instruction<XlaOperation<'static>>,
    atom_values: &[Option<ValueRef<'b, 'c, 't>>],
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    ProgramInput: Parameterized<XlaValue<'static>>,
    ProgramOutput: Parameterized<XlaValue<'static>>,
{
    let output_types = instruction
        .outputs()
        .iter()
        .map(|output| program.atoms()[output.index()].r#type().into_owned())
        .collect::<Vec<_>>();
    let captured_values = match &instruction.operation() {
        XlaOperation::Extension(XlaOperationExtension::LinearShardMap(shard_map_op)) => shard_map_op
            .linear_state()
            .captured_global_primals()
            .iter()
            .map(|atom_id| atom_values[atom_id.index()].ok_or(LoweringError::MissingAtomValue { atom_id: *atom_id }))
            .collect::<Result<Vec<_>, _>>()?,
        _ => Vec::new(),
    };
    let mut lowerer = ShardMapMlirLowerer::new(*block, context, location);
    dispatch_lower_shard_map_mlir(
        &instruction.operation(),
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

/// Maps a [`CompareKind`] to the matching StableHLO [`stable_hlo::ComparisonDirection`].
fn compare_kind_to_direction(kind: CompareKind) -> stable_hlo::ComparisonDirection {
    match kind {
        CompareKind::Eq => stable_hlo::ComparisonDirection::Equal,
        CompareKind::Ne => stable_hlo::ComparisonDirection::NotEqual,
        CompareKind::Lt => stable_hlo::ComparisonDirection::LessThan,
        CompareKind::Le => stable_hlo::ComparisonDirection::LessThanOrEqual,
        CompareKind::Gt => stable_hlo::ComparisonDirection::GreaterThan,
        CompareKind::Ge => stable_hlo::ComparisonDirection::GreaterThanOrEqual,
    }
}

/// Determines the StableHLO comparison semantic type for an element [`DataType`].
///
/// Currently unused — [`lower_compare_to_mlir`] hardcodes [`ComparisonType::Float`] until we
/// extract the operand's element type from its MLIR tensor type. Kept here so the routing logic
/// is documented and can be wired up trivially once that helper exists.
#[allow(dead_code)]
fn compare_type_for_data_type(data_type: DataType) -> Result<stable_hlo::ComparisonType, LoweringError> {
    match data_type {
        DataType::F4E2M1FN
        | DataType::F8E3M4
        | DataType::F8E4M3
        | DataType::F8E4M3B11FNUZ
        | DataType::F8E4M3FN
        | DataType::F8E4M3FNUZ
        | DataType::F8E5M2
        | DataType::F8E5M2FNUZ
        | DataType::F8E8M0FNU
        | DataType::BF16
        | DataType::F16
        | DataType::F32
        | DataType::F64 => Ok(stable_hlo::ComparisonType::Float),
        DataType::I1 | DataType::I2 | DataType::I4 | DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64 => {
            Ok(stable_hlo::ComparisonType::Signed)
        }
        DataType::Boolean
        | DataType::U1
        | DataType::U2
        | DataType::U4
        | DataType::U8
        | DataType::U16
        | DataType::U32
        | DataType::U64 => Ok(stable_hlo::ComparisonType::Unsigned),
        DataType::Token | DataType::C64 | DataType::C128 => Err(LoweringError::UnsupportedDataType { data_type }),
    }
}

/// Lowers an [`ArrayOperation::Compare`] / [`LinearArrayOperation::Compare`]-style dispatch to
/// `stablehlo.compare`. The resulting value has the broadcasted shape of the inputs and Boolean
/// element type. The comparison semantic is routed based on the LHS value's element type
/// (Float / Signed / Unsigned).
fn lower_compare_to_mlir<'b, 'c: 'b, 't: 'c>(
    kind: CompareKind,
    lhs: ValueRef<'b, 'c, 't>,
    rhs: ValueRef<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let direction = compare_kind_to_direction(kind);
    let lhs_type = lhs.r#type()?;
    let comparison_type = comparison_type_for_mlir_type(lhs_type)?;
    let result = block.append_operation(stable_hlo::compare(lhs, rhs, direction, comparison_type, location)?)?;
    Ok(result.result(0).expect("stablehlo.compare should return one result").as_ref())
}

/// Picks the right StableHLO comparison semantic based on the element type of an MLIR value.
///
/// Tensor values are unwrapped to their element type; non-tensor scalar types are inspected
/// directly. Float-family types route to [`stable_hlo::ComparisonType::Float`]; explicitly
/// unsigned integers route to [`stable_hlo::ComparisonType::Unsigned`]; everything else
/// (signless / signed integers, including Boolean as a signless `i1`) routes to
/// [`stable_hlo::ComparisonType::Signed`], which `stablehlo.compare` interprets sign-aware for
/// the actual width.
fn comparison_type_for_mlir_type<'c, 't>(r#type: TypeRef<'c, 't>) -> Result<stable_hlo::ComparisonType, LoweringError> {
    let element_type = if let Some(tensor) = r#type.cast::<TensorTypeRef>() {
        tensor.element_type().map_err(|error| LoweringError::MlirError(error))?
    } else {
        r#type
    };
    if element_type.is::<FloatTypeRef>() {
        return Ok(stable_hlo::ComparisonType::Float);
    }
    if let Some(integer) = element_type.cast::<IntegerTypeRef>() {
        if integer.is_unsigned() {
            return Ok(stable_hlo::ComparisonType::Unsigned);
        }
        return Ok(stable_hlo::ComparisonType::Signed);
    }
    // Default: treat as float for unknown element types (matches StableHLO's lenient handling
    // of non-integer non-float numeric types like complex).
    Ok(stable_hlo::ComparisonType::Float)
}

/// Lowers an [`ArrayOperation::Logical`] dispatch to one of `stablehlo.{and, or, xor, not}`.
fn lower_logical_to_mlir<'b, 'c: 'b, 't: 'c>(
    kind: LogicalKind,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let result = match kind {
        LogicalKind::And => block.append_operation(stable_hlo::and(input_values[0], input_values[1], location)?)?,
        LogicalKind::Or => block.append_operation(stable_hlo::or(input_values[0], input_values[1], location)?)?,
        LogicalKind::Xor => block.append_operation(stable_hlo::xor(input_values[0], input_values[1], location)?)?,
        LogicalKind::Not => block.append_operation(stable_hlo::not(input_values[0], location)?)?,
    };
    Ok(result.result(0).expect("stablehlo logical op should return one result").as_ref())
}

/// Builds a single-instruction reduction-body region for [`stable_hlo::reduce`] over the given
/// scalar `element_type`. The generated region has one block taking two scalar tensor arguments
/// of `tensor<{element_type}>` and produces a single scalar result via the binary `combiner`
/// matching the reduction kind. Returns the constructed [`DetachedRegion`].
fn build_reduce_body_region<'c, 't>(
    kind: ReductionKind,
    element_type: DataType,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ryft_mlir::DetachedRegion<'c, 't>, LoweringError> {
    let scalar_array_type = ArrayType::scalar(element_type);
    let scalar_tensor_type = lower_tensor_type(&scalar_array_type, context, location)?;
    let block = context.block(&[(scalar_tensor_type, location), (scalar_tensor_type, location)]);
    let mut region = context.region();
    let mut block_ref = region.append_block(block)?;
    let lhs = block_ref.argument(0)?.as_ref();
    let rhs = block_ref.argument(1)?.as_ref();
    let body_result = match kind {
        ReductionKind::Sum | ReductionKind::Mean => block_ref.append_operation(stable_hlo::add(lhs, rhs, location)?)?,
        ReductionKind::Max => block_ref.append_operation(stable_hlo::maximum(lhs, rhs, location)?)?,
        ReductionKind::Min => block_ref.append_operation(stable_hlo::minimum(lhs, rhs, location)?)?,
        ReductionKind::Any => block_ref.append_operation(stable_hlo::or(lhs, rhs, location)?)?,
        ReductionKind::All => block_ref.append_operation(stable_hlo::and(lhs, rhs, location)?)?,
    };
    let body_value = body_result.result(0).expect("stablehlo body combiner should return one result").as_ref();
    block_ref.append_operation(stable_hlo::r#return(&[body_value], location)?)?;
    Ok(region)
}

/// Lowers an [`ArrayOperation::Reduce`] dispatch to `stablehlo.reduce` with the appropriate
/// scalar body region and an initial-value constant matching the reduction's identity element.
fn lower_reduce_to_mlir<'b, 'c: 'b, 't: 'c>(
    kind: ReductionKind,
    axes: &[usize],
    input_value: ValueRef<'b, 'c, 't>,
    output_array_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let element_type = output_array_type.data_type();
    let initial_value = build_reduction_identity_constant(kind, element_type, block, context, location)?;
    let body_region = build_reduce_body_region(kind, element_type, context, location)?;
    let reduce_op = stable_hlo::reduce(&[input_value], &[initial_value], axes, body_region, location)?;
    let result = block.append_operation(reduce_op)?;
    let sum_result = result.result(0).expect("stablehlo.reduce should return one result").as_ref();
    if matches!(kind, ReductionKind::Mean) {
        // Mean = Sum / axis_size_product. Stage a constant divisor of the right element type and
        // divide. Skip if any reduced axis has a dynamic size; the caller's `infer_output_types`
        // would have rejected that earlier.
        let _ = sum_result;
        return Err(LoweringError::UnsupportedOp { op: "reduce_mean".to_string() });
    }
    Ok(sum_result)
}

/// Builds a scalar constant equal to the identity element for the given reduction kind, returned
/// as an MLIR `tensor<{element_type}>` value. Used as the `initial_values` argument of
/// `stablehlo.reduce`.
fn build_reduction_identity_constant<'b, 'c: 'b, 't: 'c>(
    kind: ReductionKind,
    element_type: DataType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let scalar_array_type = ArrayType::scalar(element_type);
    let scalar_tensor_type = lower_tensor_type(&scalar_array_type, context, location)?;
    let attribute = build_reduction_identity_attribute(kind, element_type, scalar_tensor_type, context)?;
    let result = block.append_operation(stable_hlo::constant(attribute, location)?)?;
    Ok(result.result(0).expect("stablehlo.constant should return one result").as_ref())
}

/// Builds a dense-elements attribute holding the identity element of the given reduction kind at
/// the given element type.
fn build_reduction_identity_attribute<'c, 't>(
    kind: ReductionKind,
    element_type: DataType,
    tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
    context: &'c MlirContext<'t>,
) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
    match (kind, element_type) {
        (ReductionKind::Sum | ReductionKind::Mean, DataType::F32) => context
            .dense_f32_elements_attribute(tensor_type, &[0.0])
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: element_type })?
            .cast::<DenseElementsAttributeRef>()
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type: element_type }),
        (ReductionKind::Sum | ReductionKind::Mean, DataType::F64) => context
            .dense_f64_elements_attribute(tensor_type, &[0.0])
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: element_type })?
            .cast::<DenseElementsAttributeRef>()
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type: element_type }),
        (ReductionKind::Max, DataType::F32) => context
            .dense_f32_elements_attribute(tensor_type, &[f32::NEG_INFINITY])
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: element_type })?
            .cast::<DenseElementsAttributeRef>()
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type: element_type }),
        (ReductionKind::Max, DataType::F64) => context
            .dense_f64_elements_attribute(tensor_type, &[f64::NEG_INFINITY])
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: element_type })?
            .cast::<DenseElementsAttributeRef>()
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type: element_type }),
        (ReductionKind::Min, DataType::F32) => context
            .dense_f32_elements_attribute(tensor_type, &[f32::INFINITY])
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: element_type })?
            .cast::<DenseElementsAttributeRef>()
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type: element_type }),
        (ReductionKind::Min, DataType::F64) => context
            .dense_f64_elements_attribute(tensor_type, &[f64::INFINITY])
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: element_type })?
            .cast::<DenseElementsAttributeRef>()
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type: element_type }),
        (ReductionKind::Any, DataType::Boolean) => context
            .dense_bool_elements_attribute(tensor_type, &[false])
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: element_type })?
            .cast::<DenseElementsAttributeRef>()
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type: element_type }),
        (ReductionKind::All, DataType::Boolean) => context
            .dense_bool_elements_attribute(tensor_type, &[true])
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: element_type })?
            .cast::<DenseElementsAttributeRef>()
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type: element_type }),
        _ => Err(LoweringError::UnsupportedDataType { data_type: element_type }),
    }
}

/// Lowers an [`ArrayType`] to a typed MLIR tensor type.
fn lower_tensor_type<'c, 't, L: Location<'c, 't>>(
    array_type: &ArrayType,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ryft_mlir::TensorTypeRef<'c, 't>, LoweringError> {
    let element_type = lower_element_type(array_type.data_type(), context)?;
    let dimensions = array_type
        .shape()
        .dimensions()
        .iter()
        .map(|size| match size {
            Size::Static(value) => MlirSize::Static(*value),
            Size::Dynamic(_) => MlirSize::Dynamic,
        })
        .collect::<Vec<_>>();
    context
        .tensor_type(element_type, dimensions.as_slice(), None, location)
        .map_err(|_| LoweringError::InvalidTensorType { array_type: array_type.clone() })
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
/// Lowers an arbitrary `f64` factor into a splatted scalar StableHLO constant whose element type
/// matches `output_type`, then broadcasts that scalar to the full output shape. Used by both
/// [`ArrayOperation::ScaleByConstant`](
/// ryft_core::ArrayOperation::ScaleByConstant) and
/// [`LinearArrayOperation::ScaleByConstant`](ryft_core::LinearArrayOperation::ScaleByConstant)
/// lowerings.
fn lower_f64_constant_splat<'b, 'c: 'b, 't: 'c, B, L>(
    factor: f64,
    output_type: &ArrayType,
    output_tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    let data_type = output_type.data_type();
    let scalar_tensor_type = context
        .tensor_type(lower_element_type(data_type, context)?, &[], None, location)
        .map_err(|_| LoweringError::InvalidTensorType { array_type: ArrayType::scalar(data_type) })?;
    let elements = lower_f64_scalar_elements_attribute(data_type, scalar_tensor_type, factor, context)?;
    let scalar_constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
    if output_type.shape().dimensions().is_empty() {
        return Ok(scalar_constant.result(0).expect("stablehlo.constant should return one result").as_ref());
    }
    let broadcast = block.append_operation(stable_hlo::broadcast(
        scalar_constant.result(0).unwrap().as_ref(),
        output_tensor_type,
        &[],
        location,
    )?)?;
    Ok(broadcast.result(0).expect("stablehlo.broadcast should return one result").as_ref())
}

fn lower_f64_scalar_elements_attribute<'c, 't>(
    data_type: DataType,
    tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
    factor: f64,
    context: &'c MlirContext<'t>,
) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
    match data_type {
        DataType::Boolean => context
            .splatted_dense_attribute_elements_attribute(tensor_type, context.boolean_attribute(factor != 0.0))
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::I1 | DataType::I2 | DataType::I4 | DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64 => {
            context
                .splatted_dense_attribute_elements_attribute(
                    tensor_type,
                    context.integer_attribute(
                        context.signless_integer_type(signed_integer_width(data_type)?),
                        factor as i64,
                    ),
                )
                .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type })
        }
        DataType::U1 | DataType::U2 | DataType::U4 | DataType::U8 | DataType::U16 | DataType::U32 | DataType::U64 => {
            context
                .splatted_dense_attribute_elements_attribute(
                    tensor_type,
                    context.integer_attribute(
                        context.unsigned_integer_type(unsigned_integer_width(data_type)?),
                        factor as i64,
                    ),
                )
                .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type })
        }
        DataType::BF16 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.bfloat16_type(), factor),
            )
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F16 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float16_type(), factor),
            )
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F32 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float32_type(), factor),
            )
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F64 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float64_type(), factor),
            )
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type }),
        unsupported => Err(LoweringError::InvalidDenseElementsAttribute { data_type: unsupported }),
    }
}

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
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::I1 | DataType::I2 | DataType::I4 | DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64 => {
            context
                .splatted_dense_attribute_elements_attribute(
                    tensor_type,
                    context.integer_attribute(
                        context.signless_integer_type(signed_integer_width(data_type)?),
                        integer_value,
                    ),
                )
                .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type })
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
                .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type })
        }
        DataType::BF16 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.bfloat16_type(), float_value),
            )
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F16 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float16_type(), float_value),
            )
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F32 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float32_type(), float_value),
            )
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F64 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float64_type(), float_value),
            )
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F4E2M1FN => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float4e2m1fn_type(), float_value),
            )
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E3M4 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e3m4_type(), float_value),
            )
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E4M3 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e4m3_type(), float_value),
            )
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E4M3FN => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e4m3fn_type(), float_value),
            )
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E4M3FNUZ => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e4m3fnuz_type(), float_value),
            )
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E4M3B11FNUZ => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e4m3b11fnuz_type(), float_value),
            )
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E5M2 => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e5m2_type(), float_value),
            )
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E5M2FNUZ => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e5m2fnuz_type(), float_value),
            )
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F8E8M0FNU => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float8e8m0fnu_type(), float_value),
            )
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type }),
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
    use std::borrow::Cow;
    use std::fmt::Display;
    use std::ops::{Add, Div, Mul, Neg, Sub};

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use ryft_core::broadcasting::Broadcastable;
    use ryft_core::operations::arithmetic::Scale;
    use ryft_core::operations::constants::{One, OneLike, Zero, ZeroLike};
    use ryft_core::operations::trigonometric::{Cos, Sin};
    use ryft_core::parameters::{Parameter, Placeholder};
    use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use ryft_core::tracing::domains::{Domain, RuntimeDomain, TracingDomain};
    use ryft_core::tracing::{ProgramBuilder, Traceable, TracingError, Value as TraceValue};
    use ryft_core::tracing_v2::operations::broadcast::{BroadcastInDim, broadcast_in_dim_evaluate};
    use ryft_core::tracing_v2::operations::control_flow::{ControlFlowError, ControlFlowValue};
    use ryft_core::tracing_v2::operations::dot::{Dot, DotDimensionNumbers, LeftDot, RightDot, dot_general_evaluate};
    use ryft_core::tracing_v2::operations::transpose::{Transpose, transpose_evaluate, transpose_is_identity};
    use ryft_core::tracing_v2::{
        ArrayOperation, CoordinateValue, DifferentiableDomain, LinearArrayOperation, LinearizableDomain, Reshape,
    };
    use ryft_core::types::{Shape, Typed};
    #[cfg(feature = "ndarray")]
    use ryft_ndarray::{Array as NdArrayValue, NdArrayDomain};

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

    #[derive(Clone, Debug, PartialEq)]
    struct TestArray {
        r#type: ArrayType,
        values: Vec<f64>,
    }

    impl TestArray {
        fn scalar(value: f64) -> Self {
            Self { r#type: ArrayType::scalar(DataType::F64), values: vec![value] }
        }

        fn element_count(r#type: &ArrayType) -> usize {
            r#type.element_count().unwrap().unwrap()
        }

        fn binary(self, rhs: Self, function: impl Fn(f64, f64) -> f64) -> Self {
            Self {
                r#type: self.r#type.clone().broadcast(&rhs.r#type).unwrap(),
                values: self.values.into_iter().zip(rhs.values).map(|(left, right)| function(left, right)).collect(),
            }
        }
    }

    impl Parameter for TestArray {}

    impl Display for TestArray {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "{:?}", self.values)
        }
    }

    impl Typed<ArrayType> for TestArray {
        fn r#type(&self) -> Cow<'_, ArrayType> {
            Cow::Borrowed(&self.r#type)
        }
    }

    impl Traceable<ArrayType> for TestArray {}

    impl TraceValue<ArrayType> for TestArray {}

    impl ControlFlowValue for TestArray {
        fn control_flow_predicate(&self) -> Result<bool, TracingError> {
            Err(ControlFlowError::InvalidPredicateValue { type_: self.r#type.clone() }.into())
        }
    }

    impl Zero<ArrayType> for TestArray {
        fn zero(r#type: &ArrayType) -> Result<Self, TracingError> {
            Ok(Self { r#type: r#type.clone(), values: vec![0.0; Self::element_count(r#type)] })
        }
    }

    impl One<ArrayType> for TestArray {
        fn one(r#type: &ArrayType) -> Result<Self, TracingError> {
            Ok(Self { r#type: r#type.clone(), values: vec![1.0; Self::element_count(r#type)] })
        }
    }

    impl ZeroLike for TestArray {
        fn zero_like(&self) -> Self {
            Self { r#type: self.r#type.clone(), values: vec![0.0; self.values.len()] }
        }
    }

    impl OneLike for TestArray {
        fn one_like(&self) -> Self {
            Self { r#type: self.r#type.clone(), values: vec![1.0; self.values.len()] }
        }
    }

    impl CoordinateValue for TestArray {
        type Coordinate = f64;

        fn coordinate_count(&self) -> usize {
            self.values.len()
        }

        fn coordinate_basis(&self) -> Vec<Self> {
            (0..self.values.len())
                .map(|index| {
                    let mut values = vec![0.0; self.values.len()];
                    values[index] = 1.0;
                    Self { r#type: self.r#type.clone(), values }
                })
                .collect()
        }

        fn coordinates(&self) -> Vec<Self::Coordinate> {
            self.values.clone()
        }

        fn stack(values: Vec<Self>) -> Result<Self, TracingError> {
            let lane_count = values.len();
            assert!(lane_count > 0, "cannot stack zero values");
            let first_type = &values[0].r#type;
            for value in values.iter().skip(1) {
                assert_eq!(value.r#type, *first_type, "stacked test arrays must share the same type");
            }
            let stacked_dimensions = std::iter::once(Size::Static(lane_count))
                .chain(first_type.shape().dimensions().iter().copied())
                .collect::<Vec<_>>();
            let stacked_type =
                ArrayType::new(first_type.data_type(), Shape::new(stacked_dimensions), None, None).unwrap();
            let mut stacked_values = Vec::with_capacity(lane_count * values[0].values.len());
            for value in values {
                stacked_values.extend(value.values);
            }
            Ok(Self { r#type: stacked_type, values: stacked_values })
        }
    }

    impl Add for TestArray {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            self.binary(rhs, |left, right| left + right)
        }
    }

    impl Sub for TestArray {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            self.binary(rhs, |left, right| left - right)
        }
    }

    impl Mul for TestArray {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            self.binary(rhs, |left, right| left * right)
        }
    }

    impl Scale for TestArray {
        type Output = Self;

        fn scale(self, factor: Self) -> Self::Output {
            factor * self
        }
    }

    impl Scale<f64> for TestArray {
        type Output = Self;

        fn scale(self, factor: f64) -> Self::Output {
            Self { r#type: self.r#type, values: self.values.into_iter().map(|value| value * factor).collect() }
        }
    }

    impl ryft_core::ConstantLike<f64> for TestArray {
        fn constant_like(&self, value: f64) -> Self {
            Self { r#type: self.r#type.clone(), values: vec![value; self.values.len()] }
        }
    }

    impl Div for TestArray {
        type Output = Self;

        fn div(self, rhs: Self) -> Self::Output {
            self.binary(rhs, |left, right| left / right)
        }
    }

    impl Neg for TestArray {
        type Output = Self;

        fn neg(self) -> Self::Output {
            Self { r#type: self.r#type, values: self.values.into_iter().map(|value| -value).collect() }
        }
    }

    impl Sin for TestArray {
        fn sin(self) -> Self {
            Self { r#type: self.r#type, values: self.values.into_iter().map(f64::sin).collect() }
        }
    }

    impl Cos for TestArray {
        fn cos(self) -> Self {
            Self { r#type: self.r#type, values: self.values.into_iter().map(f64::cos).collect() }
        }
    }

    impl Dot for TestArray {
        fn dot(self, rhs: Self, dimensions: &DotDimensionNumbers) -> Self {
            let lhs_shape: Vec<usize> =
                self.r#type.shape().dimensions().iter().map(|size| size.value().unwrap()).collect();
            let rhs_shape: Vec<usize> =
                rhs.r#type.shape().dimensions().iter().map(|size| size.value().unwrap()).collect();
            let (values, output_shape) = dot_general_evaluate(
                self.values.as_slice(),
                lhs_shape.as_slice(),
                rhs.values.as_slice(),
                rhs_shape.as_slice(),
                dimensions,
                || 0.0f64,
                |accumulator, lhs_value, rhs_value| accumulator + lhs_value * rhs_value,
            );
            let output_dimensions: Vec<Size> = output_shape.iter().map(|size| Size::Static(*size)).collect();
            let output_type =
                ArrayType::new(self.r#type.data_type(), Shape::new(output_dimensions), None, None).unwrap();
            Self { r#type: output_type, values }
        }
    }

    impl LeftDot for TestArray {
        #[inline]
        fn left_dot(self, factor: Self, dimensions: &DotDimensionNumbers) -> Self {
            factor.dot(self, dimensions)
        }
    }

    impl BroadcastInDim for TestArray {
        fn broadcast_in_dim(self, target_type: ArrayType, broadcast_dimensions: Vec<usize>) -> Self {
            let input_shape: Vec<usize> =
                self.r#type.shape().dimensions().iter().map(|size| size.value().unwrap()).collect();
            let target_shape: Vec<usize> =
                target_type.shape().dimensions().iter().map(|size| size.value().unwrap()).collect();
            let values = broadcast_in_dim_evaluate(
                self.values.as_slice(),
                input_shape.as_slice(),
                target_shape.as_slice(),
                broadcast_dimensions.as_slice(),
            );
            Self { r#type: target_type, values }
        }
    }

    impl RightDot for TestArray {
        #[inline]
        fn right_dot(self, factor: Self, dimensions: &DotDimensionNumbers) -> Self {
            self.dot(factor, dimensions)
        }
    }

    impl Transpose for TestArray {
        fn transpose(self, permutation: Vec<usize>) -> Self {
            if transpose_is_identity(&permutation) {
                return self;
            }
            let shape: Vec<usize> = self.r#type.shape().dimensions().iter().map(|size| size.value().unwrap()).collect();
            let (values, output_shape) =
                transpose_evaluate(self.values.as_slice(), shape.as_slice(), permutation.as_slice());
            let output_dimensions: Vec<Size> = output_shape.iter().map(|size| Size::Static(*size)).collect();
            let output_type =
                ArrayType::new(self.r#type.data_type(), Shape::new(output_dimensions), None, None).unwrap();
            Self { r#type: output_type, values }
        }
    }

    impl Reshape for TestArray {
        fn reshape(self, target_shape: Shape) -> Result<Self, TracingError> {
            Ok(Self {
                r#type: ArrayType::new(self.r#type.data_type(), target_shape, None, None).unwrap(),
                values: self.values,
            })
        }
    }

    impl ryft_core::tracing_v2::operations::select::Select for TestArray {
        fn select(predicate: Self, on_true: Self, on_false: Self) -> Result<Self, TracingError> {
            let values: Vec<f64> = predicate
                .values
                .iter()
                .zip(on_true.values.iter())
                .zip(on_false.values.iter())
                .map(|((pred, t), f)| if *pred != 0.0 { *t } else { *f })
                .collect();
            Ok(Self { r#type: on_true.r#type, values })
        }
    }

    impl ryft_core::tracing_v2::operations::compare::Compare for TestArray {
        type Output = Self;

        fn compare(self, rhs: Self, kind: ryft_core::tracing_v2::operations::compare::CompareKind) -> Self {
            use ryft_core::tracing_v2::operations::compare::CompareKind;
            let values: Vec<f64> = self
                .values
                .iter()
                .zip(rhs.values.iter())
                .map(|(left, right)| {
                    let predicate = match kind {
                        CompareKind::Eq => left == right,
                        CompareKind::Ne => left != right,
                        CompareKind::Lt => left < right,
                        CompareKind::Le => left <= right,
                        CompareKind::Gt => left > right,
                        CompareKind::Ge => left >= right,
                    };
                    if predicate { 1.0 } else { 0.0 }
                })
                .collect();
            let output_type = ArrayType::new(DataType::Boolean, self.r#type.shape().clone(), None, None).unwrap();
            Self { r#type: output_type, values }
        }
    }

    impl ryft_core::tracing_v2::operations::logical::LogicalBinary for TestArray {
        fn logical_binary(self, rhs: Self, kind: ryft_core::tracing_v2::operations::logical::LogicalKind) -> Self {
            use ryft_core::tracing_v2::operations::logical::LogicalKind;
            let values: Vec<f64> = self
                .values
                .iter()
                .zip(rhs.values.iter())
                .map(|(left, right)| {
                    let left_bool = *left != 0.0;
                    let right_bool = *right != 0.0;
                    let result = match kind {
                        LogicalKind::And => left_bool && right_bool,
                        LogicalKind::Or => left_bool || right_bool,
                        LogicalKind::Xor => left_bool ^ right_bool,
                        LogicalKind::Not => unreachable!("LogicalKind::Not is unary"),
                    };
                    if result { 1.0 } else { 0.0 }
                })
                .collect();
            Self { r#type: self.r#type, values }
        }
    }

    impl ryft_core::tracing_v2::operations::logical::LogicalNot for TestArray {
        fn logical_not(self) -> Self {
            let values: Vec<f64> = self.values.into_iter().map(|value| if value != 0.0 { 0.0 } else { 1.0 }).collect();
            Self { r#type: self.r#type, values }
        }
    }

    impl ryft_core::tracing_v2::operations::reduce::Reduce for TestArray {
        fn reduce(self, axes: &[usize], kind: ryft_core::tracing_v2::operations::reduce::ReductionKind) -> Self {
            use ryft_core::tracing_v2::operations::reduce::{ReductionKind, reduce_evaluate};
            if axes.is_empty() {
                return self;
            }
            let shape: Vec<usize> = self.r#type.shape().dimensions().iter().map(|size| size.value().unwrap()).collect();
            let (reduced_values, reduced_shape) = match kind {
                ReductionKind::Sum | ReductionKind::Mean => {
                    reduce_evaluate(self.values.as_slice(), shape.as_slice(), axes, || 0.0, |acc, value| acc + value)
                }
                ReductionKind::Max => reduce_evaluate(
                    self.values.as_slice(),
                    shape.as_slice(),
                    axes,
                    || f64::NEG_INFINITY,
                    |acc, value| acc.max(value),
                ),
                ReductionKind::Min => reduce_evaluate(
                    self.values.as_slice(),
                    shape.as_slice(),
                    axes,
                    || f64::INFINITY,
                    |acc, value| acc.min(value),
                ),
                ReductionKind::Any => reduce_evaluate(
                    self.values.as_slice(),
                    shape.as_slice(),
                    axes,
                    || 0.0,
                    |acc, value| if acc != 0.0 || value != 0.0 { 1.0 } else { 0.0 },
                ),
                ReductionKind::All => reduce_evaluate(
                    self.values.as_slice(),
                    shape.as_slice(),
                    axes,
                    || 1.0,
                    |acc, value| if acc != 0.0 && value != 0.0 { 1.0 } else { 0.0 },
                ),
            };
            let mut values = reduced_values;
            if matches!(kind, ReductionKind::Mean) {
                let reduced_count: usize = axes.iter().map(|axis| shape[*axis]).product();
                let divisor = reduced_count.max(1) as f64;
                for value in values.iter_mut() {
                    *value /= divisor;
                }
            }
            let output_dimensions: Vec<Size> = reduced_shape.iter().map(|size| Size::Static(*size)).collect();
            let data_type = self.r#type.data_type();
            let output_type = ArrayType::new(data_type, Shape::new(output_dimensions), None, None).unwrap();
            Self { r#type: output_type, values }
        }
    }

    impl MlirLowerableValue for TestArray {
        fn to_dense_elements_attribute<'c, 't>(
            &self,
            tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
            context: &'c MlirContext<'t>,
        ) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
            let attribute = context
                .dense_f64_elements_attribute(tensor_type, self.values.as_slice())
                .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: DataType::F64 })?;
            attribute
                .cast::<DenseElementsAttributeRef>()
                .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type: DataType::F64 })
        }

        fn to_scalar_dense_elements_attribute<'c, 't>(
            &self,
            tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
            context: &'c MlirContext<'t>,
        ) -> Result<Option<DenseElementsAttributeRef<'c, 't>>, LoweringError> {
            let [value] = self.values.as_slice() else {
                return Ok(None);
            };
            let attribute = context
                .dense_f64_elements_attribute(tensor_type, std::slice::from_ref(value))
                .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: DataType::F64 })?;
            Ok(Some(
                attribute
                    .cast::<DenseElementsAttributeRef>()
                    .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type: DataType::F64 })?,
            ))
        }
    }

    fn xla_identity_branch(
        input_type: ArrayType,
    ) -> Program<ArrayType, XlaValue<'static>, XlaOperation<'static>, Vec<XlaValue<'static>>, Vec<XlaValue<'static>>>
    {
        let mut builder = ProgramBuilder::<ArrayType, XlaValue<'static>, XlaOperation<'static>>::new();
        let input = builder.add_input(input_type);
        builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn xla_neg_branch(
        input_type: ArrayType,
    ) -> Program<ArrayType, XlaValue<'static>, XlaOperation<'static>, Vec<XlaValue<'static>>, Vec<XlaValue<'static>>>
    {
        let mut builder = ProgramBuilder::<ArrayType, XlaValue<'static>, XlaOperation<'static>>::new();
        let input = builder.add_input(input_type);
        let output = builder.add_instruction(XlaOperation::Neg, vec![input]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn lower_traced_module(
        traced: &TracedShardMap<ArrayType, ArrayType>,
        function_name: &str,
    ) -> Result<String, super::super::shard_map::ShardMapTraceError> {
        traced.to_mlir_module(function_name)
    }

    #[cfg(feature = "ndarray")]
    fn bilinear_matmul<M>(inputs: (M, M)) -> M
    where
        M: ryft_core::tracing_v2::DotOps,
    {
        inputs.0.dot(inputs.1, &DotDimensionNumbers::matmul())
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
                let product = x.clone().transpose(vec![1, 0]).dot(x, &DotDimensionNumbers::matmul());
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
                      %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<2x4xf32>) -> tensor<4x2xf32>
                      %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
                      %3 = stablehlo.negate %2 : tensor<4x4xf32>
                      %4 = stablehlo.cosine %3 : tensor<4x4xf32>
                      %5 = stablehlo.sine %4 : tensor<4x4xf32>
                      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                      %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x4xf32>
                      %7 = stablehlo.multiply %5, %6 : tensor<4x4xf32>
                      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                      %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4x4xf32>
                      %9 = stablehlo.add %7, %8 : tensor<4x4xf32>
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
        let mut builder = ProgramBuilder::<ArrayType, XlaValue, XlaOperation>::new();
        let predicate = builder.add_input(predicate_type);
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(XlaOperation::Condition(Box::new(condition)), vec![predicate, input])
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaValue>, Vec<XlaValue>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
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
        let mut builder = ProgramBuilder::<ArrayType, XlaValue, XlaOperation>::new();
        let state = builder.add_input(state_type);
        let output = builder.add_instruction(XlaOperation::While(Box::new(while_operation)), vec![state]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaValue>, Vec<XlaValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert!(stablehlo.contains("stablehlo.while"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.return"), "{stablehlo}");
    }

    // ---------------------------------------------------------------------------
    // Plain-program StableHLO lowering tests for scalar programs
    // ---------------------------------------------------------------------------

    fn scalar_bilinear_sin<T>(inputs: (T, T)) -> T
    where
        T: Clone + ryft_core::operations::trigonometric::Sin + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    {
        inputs.0.clone() * inputs.1 + inputs.0.sin()
    }

    fn scalar_quartic_plus_sin<T>(x: T) -> T
    where
        T: Clone + ryft_core::operations::trigonometric::Sin + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    {
        x.clone() * x.clone() * x.clone() * x.clone() + x.sin()
    }

    #[derive(Copy, Clone, Debug)]
    struct TestArrayDomain;

    impl Domain for TestArrayDomain {
        type Type = ArrayType;
        type Value = TestArray;
    }

    impl RuntimeDomain for TestArrayDomain {
        fn zero(&self, r#type: &ArrayType) -> Result<Self::Value, TracingError> {
            TestArray::zero(r#type)
        }

        fn one(&self, r#type: &ArrayType) -> Result<Self::Value, TracingError> {
            TestArray::one(r#type)
        }
    }

    impl TracingDomain for TestArrayDomain {
        type OperationCarrier = ArrayOperation<TestArray, ArrayType>;
    }

    #[derive(Copy, Clone, Debug)]
    struct TestArrayLinearDomain;

    impl Domain for TestArrayLinearDomain {
        type Type = ArrayType;
        type Value = TestArray;
    }

    impl RuntimeDomain for TestArrayLinearDomain {
        fn zero(&self, r#type: &ArrayType) -> Result<Self::Value, TracingError> {
            TestArray::zero(r#type)
        }

        fn one(&self, r#type: &ArrayType) -> Result<Self::Value, TracingError> {
            TestArray::one(r#type)
        }
    }

    impl TracingDomain for TestArrayLinearDomain {
        type OperationCarrier = LinearArrayOperation<TestArray, ArrayType>;
    }

    static TEST_ARRAY_LINEAR_DOMAIN: TestArrayLinearDomain = TestArrayLinearDomain;

    impl LinearizableDomain for TestArrayDomain {
        type LinearDomain = TestArrayLinearDomain;

        fn linear_domain(&self) -> &Self::LinearDomain {
            &TEST_ARRAY_LINEAR_DOMAIN
        }
    }

    #[test]
    fn test_plain_scalar_bilinear_sin_jit_stablehlo() {
        let domain = TestArrayDomain;
        let (_, compiled): (
            TestArray,
            ryft_core::tracing::Program<
                ArrayType,
                TestArray,
                ryft_core::tracing_v2::ArrayOperation<TestArray, ArrayType>,
                (TestArray, TestArray),
                TestArray,
            >,
        ) = domain
            .interpret_and_trace(
                |inputs| Ok(scalar_bilinear_sin(inputs)),
                (TestArray::scalar(2.0), TestArray::scalar(3.0)),
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
    fn test_plain_scalar_quartic_plus_sin_grad_stablehlo() {
        let domain = TestArrayDomain;
        let (_, compiled): (
            TestArray,
            ryft_core::tracing::Program<
                ArrayType,
                TestArray,
                ryft_core::tracing_v2::ArrayOperation<TestArray, ArrayType>,
                TestArray,
                TestArray,
            >,
        ) = domain
            .interpret_and_trace(|x| Ok(TestArrayDomain.grad(scalar_quartic_plus_sin, x)?), TestArray::scalar(2.0))
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
            TestArray,
            ryft_core::tracing::Program<
                ArrayType,
                TestArray,
                ryft_core::tracing_v2::LinearArrayOperation<TestArray, ArrayType>,
                TestArray,
                (TestArray, TestArray),
            >,
        ) = ryft_core::tracing_v2::vjp(
            &TestArrayDomain,
            |inputs| Ok(scalar_bilinear_sin(inputs)),
            (TestArray::scalar(2.0), TestArray::scalar(3.0)),
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
        // Uses the traced value-and-gradient path that traces through vjp+pullback.
        let domain = TestArrayDomain;
        let (_, compiled): (
            (TestArray, TestArray),
            ryft_core::tracing::Program<
                ArrayType,
                TestArray,
                ryft_core::tracing_v2::ArrayOperation<TestArray, ArrayType>,
                (TestArray, TestArray),
                (TestArray, TestArray),
            >,
        ) = domain
            .interpret_and_trace(
                |inputs| Ok(TestArrayDomain.grad(scalar_bilinear_sin, inputs)?),
                (TestArray::scalar(2.0), TestArray::scalar(3.0)),
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
        let left = NdArrayValue::from_shape_vec([2, 2], vec![1.0f64, 2.0, 3.0, 4.0]).unwrap();
        let right = NdArrayValue::from_shape_vec([2, 2], vec![5.0f64, 6.0, 7.0, 8.0]).unwrap();
        let (_, pullback): (
            NdArrayValue<f64>,
            ryft_core::tracing::Program<
                ArrayType,
                NdArrayValue<f64>,
                ryft_core::tracing_v2::LinearArrayOperation<NdArrayValue<f64>, ArrayType>,
                NdArrayValue<f64>,
                (NdArrayValue<f64>, NdArrayValue<f64>),
            >,
        ) = ryft_core::tracing_v2::vjp(
            &NdArrayDomain::<f64>::new(),
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
