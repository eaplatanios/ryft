use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;

use ryft_core::backends::scalars::Scalar;
use ryft_core::macros::check_count;
use ryft_core::operations::arithmetic::{
    AbsOperation, AddOperation, DivOperation, MulOperation, NegOperation, SubOperation,
};
use ryft_core::operations::compare::ComparisonDirection;
use ryft_core::operations::complex::{ComplexOperation, ConjugateOperation, ImaginaryOperation, RealOperation};
use ryft_core::operations::constants::{ConstantOperation, FillOperation, IotaOperation};
use ryft_core::operations::control_flow::{ConditionOperation, ScanOperation, WhileOperation};
use ryft_core::operations::exponential::{ExponentialOperation, LogarithmOperation, SquareRootOperation};
use ryft_core::operations::manipulation::{
    BroadcastOperation, GatherOperation, GatherScatterMode, Reshape, ReshapeOperation, ScatterOperation,
    ScatterReductionKind, Slice, TransposeOperation, UpdateSlice,
};
use ryft_core::operations::trigonometric::{Atan2Operation, CosOperation, SinOperation};
use ryft_core::operations::{BooleanLike, Operation};
use ryft_core::parameters::Parameterized;
use ryft_core::programs::{AtomId, Instruction, Program, ProgramError, Value};
use ryft_core::sharding::{LogicalMesh, Sharding, ShardingError};
#[cfg(any(test, feature = "benchmarking"))]
use ryft_core::tests::TestArray;
use ryft_core::tracing_v2::ArrayOperation;
use ryft_core::tracing_v2::operations::DotOperation;
use ryft_core::tracing_v2::operations::collective::{AxisIndexOperation, CollectiveKind, CollectiveOperation};
use ryft_core::tracing_v2::operations::reduce::ReductionKind;
use ryft_core::types::{ArrayType, DataType, Memory, Size, Typed};
use ryft_mlir::dialects::stable_hlo::{Accuracy, CustomCallApiVersion, Precision};
use ryft_mlir::dialects::{func, shardy, stable_hlo};
use ryft_mlir::{
    Attribute, Block, BlockRef, Context as MlirContext, DenseElementsAttributeRef, FloatTypeRef, IntegerTypeRef,
    Location, LocationRef, Operation as MlirOperation, Region, Size as MlirSize, SymbolVisibility, TensorTypeRef, Type,
    TypeAndAttributes, TypeRef, Value as MlirValue, ValueAndAttributes, ValueRef,
};

use crate::experimental::debugging::{PRINT_CUSTOM_CALL_TARGET, PRINT_LABEL_ATTRIBUTE};
#[cfg(test)]
use crate::experimental::ops::XlaProgramBuilder;
use crate::experimental::ops::{FlatXlaProgram, XlaConstant, XlaOperation, XlaProgram};
use crate::mlir::ToMlir;

use super::shard_map::{ShardMap, ShardMapError};

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

    /// Error returned when lowering encounters a captured constant reference without a matching hidden argument.
    #[error("missing captured constant #{index} during XLA lowering")]
    MissingCapturedConstant { index: usize },

    /// Error returned when lowering tries to materialize abstract XLA type metadata as a literal value.
    #[error("abstract XLA value '{array_type}' cannot be materialized as a StableHLO literal")]
    AbstractValueLiteral { array_type: ArrayType },

    /// Error returned when signature sharding metadata does not match the lowered function signature.
    #[error("invalid {kind} sharding count during XLA lowering: expected {expected}, got {actual}")]
    InvalidShardingCount {
        /// Name of the sharding group being validated.
        kind: &'static str,

        /// Number of shardings required by the signature.
        expected: usize,

        /// Number of shardings provided.
        actual: usize,
    },

    /// Error returned when lowering encounters a type that does not have StableHLO support yet.
    #[error("unsupported data type '{data_type}' during XLA lowering")]
    UnsupportedDataType { data_type: DataType },

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
    Tracing(#[from] ProgramError),
}

/// Lowering mode used for plain `tracing_v2` MLIR emission.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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

    /// Shared private functions emitted for deduplicated `jit_call` callees, consulted at `jit_call` lowering sites.
    /// Shared via [`Rc`] so it threads through nested lowering scopes without lifetime entanglement.
    nested_functions: Option<Rc<JitCallFunctionMap>>,

    /// Current StableHLO effect token of the lowering scope this lowerer emits into, or `None` when the scope has
    /// not lowered an effectful instruction yet. Lowerers are constructed per instruction by the instruction replay
    /// loops, which copy the scope-level token in through [`Self::with_token`] and read the updated token back out
    /// after the instruction lowers, so the token chain threads through effectful instructions in program order.
    token: Option<ValueRef<'b, 'c, 't>>,

    /// Collective lowering state of the lowering scope this lowerer emits into. Refer to the documentation of
    /// [`CollectiveLoweringState`] for more information.
    collective_state: CollectiveLoweringState,
}

impl<'b, 'c: 'b, 't: 'c> PlainMlirLowerer<'b, 'c, 't> {
    /// Creates a plain MLIR lowerer for operations emitted into `block`.
    pub(crate) fn new(
        block: BlockRef<'b, 'c, 't>,
        context: &'c MlirContext<'t>,
        location: LocationRef<'c, 't>,
    ) -> Self {
        Self {
            block,
            context,
            location,
            nested_functions: None,
            token: None,
            collective_state: CollectiveLoweringState::new(),
        }
    }

    /// Attaches the shared deduplicated `jit_call` functions consulted while lowering.
    pub(crate) fn with_nested_functions(mut self, nested_functions: Option<Rc<JitCallFunctionMap>>) -> Self {
        self.nested_functions = nested_functions;
        self
    }

    /// Attaches the current effect token of the enclosing lowering scope.
    pub(crate) fn with_token(mut self, token: Option<ValueRef<'b, 'c, 't>>) -> Self {
        self.token = token;
        self
    }

    /// Attaches the collective lowering state of the enclosing lowering scope.
    pub(crate) fn with_collective_state(mut self, collective_state: CollectiveLoweringState) -> Self {
        self.collective_state = collective_state;
        self
    }

    /// Lowers one tensor type inside this lowering context.
    pub(crate) fn lower_tensor_type(
        &self,
        array_type: &ArrayType,
    ) -> Result<ryft_mlir::TensorTypeRef<'c, 't>, LoweringError> {
        lower_tensor_type(array_type, self.context, self.location)
    }

    /// Lowers one plain traced literal value inside this lowering context.
    pub(crate) fn lower_literal_value<V: MlirLowerableValue + BooleanLike>(
        &mut self,
        value: &V,
    ) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
        lower_literal_value(value, &mut self.block, self.context, self.location)
    }

    /// Lowers one nested condition operation inside this lowering context.
    pub(crate) fn lower_condition<V: MlirLowerableValue + BooleanLike, O>(
        &mut self,
        condition_op: &ConditionOperation<V, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        O: LowerableXlaOperation<V>,
    {
        lower_condition_to_if(
            condition_op,
            input_values,
            &mut self.block,
            self.context,
            self.location,
            self.nested_functions.as_ref(),
            &self.collective_state,
            &mut self.token,
        )
    }

    /// Lowers one nested while operation inside this lowering context.
    pub(crate) fn lower_while<V: MlirLowerableValue + BooleanLike, O, Payload>(
        &mut self,
        while_op: &WhileOperation<V, O, Payload>,
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        O: LowerableXlaOperation<V>,
    {
        lower_while_to_while(
            while_op,
            input_values,
            &mut self.block,
            self.context,
            self.location,
            self.nested_functions.as_ref(),
            &self.collective_state,
            &mut self.token,
        )
    }

    /// Lowers one nested scan operation inside this lowering context.
    pub(crate) fn lower_scan<V: MlirLowerableValue + BooleanLike, O, Capture, Payload>(
        &mut self,
        scan_op: &ScanOperation<V, O, Capture, Payload>,
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        Capture: Value<Type = ArrayType>,
        O: LowerableXlaOperation<V>,
    {
        lower_scan_to_while(
            scan_op.body(),
            scan_op.carry_count(),
            scan_op.length(),
            scan_op.reverse(),
            scan_op.unroll(),
            input_values,
            &mut self.block,
            self.context,
            self.location,
            self.nested_functions.as_ref(),
            &self.collective_state,
            &mut self.token,
        )
    }
}

/// Operations that can be lowered to StableHLO for XLA compilation.
///
/// Implementing this trait makes an operation eligible for MLIR lowering via
/// [`to_mlir_module_for_plain_program`] and related entry points. The core [`ArrayOperation`] enum provides the default
/// blanket implementation, and backends can add their own closed operation enums by implementing this trait for those
/// enums.
pub(crate) trait LowerableXlaOperation<V: MlirLowerableValue + BooleanLike>: Operation<ArrayType> {
    /// Lowers this operation to one or more StableHLO operations.
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>;
}

impl<V: MlirLowerableValue + BooleanLike> LowerableXlaOperation<V> for AddOperation {
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

impl<V: MlirLowerableValue + BooleanLike> LowerableXlaOperation<V> for SubOperation {
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

impl<V: MlirLowerableValue + BooleanLike> LowerableXlaOperation<V> for MulOperation {
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

impl<V: MlirLowerableValue + BooleanLike> LowerableXlaOperation<V> for DivOperation {
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

impl<V: MlirLowerableValue + BooleanLike> LowerableXlaOperation<V> for NegOperation {
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

impl<V: MlirLowerableValue + BooleanLike> LowerableXlaOperation<V> for SinOperation {
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

impl<V: MlirLowerableValue + BooleanLike> LowerableXlaOperation<V> for CosOperation {
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

impl<V: MlirLowerableValue + BooleanLike> LowerableXlaOperation<V> for Atan2Operation {
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
                .append_operation(stable_hlo::atan2(input_values[0], input_values[1], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.atan2 should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue + BooleanLike> LowerableXlaOperation<V> for ExponentialOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::exponential(
            input_values[0],
            Accuracy::Default,
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.exponential should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue + BooleanLike> LowerableXlaOperation<V> for LogarithmOperation {
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
                .append_operation(stable_hlo::log(input_values[0], Accuracy::Default, lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.log should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue + BooleanLike> LowerableXlaOperation<V> for SquareRootOperation {
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
                .append_operation(stable_hlo::sqrt(input_values[0], Accuracy::Default, lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.sqrt should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue + BooleanLike> LowerableXlaOperation<V> for AbsOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::abs(input_values[0], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.abs should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue + BooleanLike> LowerableXlaOperation<V> for ComplexOperation {
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
                .append_operation(stable_hlo::complex(input_values[0], input_values[1], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.complex should return one result").as_ref()])
    }
}

/// StableHLO has no conjugation operation, so `conjugate` lowers to the `complex(real(z), negate(imag(z)))`
/// composition (the same decomposition JAX's `conj` lowering uses).
impl<V: MlirLowerableValue + BooleanLike> LowerableXlaOperation<V> for ConjugateOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let real = lowerer.block.append_operation(stable_hlo::real(input_values[0], lowerer.location)?)?;
        let imaginary = lowerer.block.append_operation(stable_hlo::imag(input_values[0], lowerer.location)?)?;
        let negated = lowerer.block.append_operation(stable_hlo::negate(
            imaginary.result(0).expect("stablehlo.imag should return one result").as_ref(),
            lowerer.location,
        )?)?;
        let result = lowerer.block.append_operation(stable_hlo::complex(
            real.result(0).expect("stablehlo.real should return one result").as_ref(),
            negated.result(0).expect("stablehlo.negate should return one result").as_ref(),
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.complex should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue + BooleanLike> LowerableXlaOperation<V> for RealOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::real(input_values[0], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.real should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue + BooleanLike> LowerableXlaOperation<V> for ImaginaryOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::imag(input_values[0], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.imag should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue + BooleanLike> LowerableXlaOperation<V> for TransposeOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::transpose(
            input_values[0],
            self.permutation().as_slice(),
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.transpose should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue + BooleanLike> LowerableXlaOperation<V> for DotOperation {
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

impl<V: MlirLowerableValue + BooleanLike> LowerableXlaOperation<V> for FillOperation<ArrayType, Scalar> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("input", input_values, 0, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        let output_type = &output_types[0];
        let output_tensor_type = lowerer.lower_tensor_type(output_type)?;
        let constant_value = lower_scalar_constant_splat(
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

impl<V: MlirLowerableValue + BooleanLike> LowerableXlaOperation<V> for IotaOperation<ArrayType> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("input", input_values, 0, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        let output_tensor_type = lowerer.lower_tensor_type(&output_types[0])?;
        let result = lowerer.block.append_operation(stable_hlo::iota(
            output_tensor_type,
            self.iota_dimension(),
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.iota should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue + BooleanLike, Payload> LowerableXlaOperation<V> for ConstantOperation<V, Payload> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("input", input_values, 0, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        // A typed literal constant lowers to a StableHLO constant materialized from the captured value's elements.
        let constant_value = lowerer.lower_literal_value(self.value())?;
        Ok(vec![constant_value])
    }
}

impl<V: MlirLowerableValue + BooleanLike> LowerableXlaOperation<V> for ReshapeOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("output", output_types, 1, ProgramError);
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

impl<V: MlirLowerableValue + BooleanLike> LowerableXlaOperation<V> for BroadcastOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("output", output_types, 1, ProgramError);
        let output_tensor_type = lowerer.lower_tensor_type(&output_types[0])?;
        let result = lowerer.block.append_operation(stable_hlo::broadcast(
            input_values[0],
            output_tensor_type,
            self.output_axes(),
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.broadcast_in_dim should return one result").as_ref()])
    }
}

/// Lowers static start indices to scalar `i64` StableHLO constants, as consumed by the index operands of
/// `stablehlo.dynamic_update_slice` when lowering the statically indexed `update_slice` operation (StableHLO has no
/// statically indexed update operation).
fn lower_static_index_constants<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    start_indices: &[usize],
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let element_type = lower_element_type(DataType::I64, context)?;
    let tensor_type = context
        .tensor_type(element_type, &[], None, location)
        .map_err(|_| LoweringError::InvalidTensorType { array_type: ArrayType::scalar(DataType::I64) })?;
    start_indices
        .iter()
        .map(|index| {
            let elements = lower_constant_elements_attribute(DataType::I64, tensor_type, *index as i64, context)?;
            let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
            Ok(constant.result(0).expect("stablehlo.constant should return one result").as_ref())
        })
        .collect()
}

fn lower_constant_output<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    output_types: &[ArrayType],
    integer_value: i64,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    check_count!("output", output_types, 1, ProgramError);
    let output_type = &output_types[0];
    let tensor_type = lower_tensor_type(output_type, context, location)?;
    let data_type = output_type.data_type();

    // Complex constants are composed as `value + 0i` from two real part constants through `stablehlo.complex`, so
    // that the dense-elements attribute helpers stay real-valued (MLIR has no complex scalar attribute to splat).
    if matches!(data_type, DataType::C64 | DataType::C128) {
        let part_data_type = if data_type == DataType::C64 { DataType::F32 } else { DataType::F64 };
        let part_tensor_type =
            context.tensor_type(lower_element_type(part_data_type, context)?, &[], None, location)?;
        let real_elements =
            lower_constant_elements_attribute(part_data_type, part_tensor_type, integer_value, context)?;
        let real = block.append_operation(stable_hlo::constant(real_elements, location)?)?;
        let imaginary_elements = lower_constant_elements_attribute(part_data_type, part_tensor_type, 0, context)?;
        let imaginary = block.append_operation(stable_hlo::constant(imaginary_elements, location)?)?;
        let complex = block.append_operation(stable_hlo::complex(
            real.result(0).expect("stablehlo.constant should return one result").as_ref(),
            imaginary.result(0).expect("stablehlo.constant should return one result").as_ref(),
            location,
        )?)?;
        let complex = complex.result(0).expect("stablehlo.complex should return one result").as_ref();
        if output_type.shape().dimensions().is_empty() {
            return Ok(vec![complex]);
        }
        let broadcast = block.append_operation(stable_hlo::broadcast(complex, tensor_type, &[], location)?)?;
        return Ok(vec![broadcast.result(0).expect("stablehlo.broadcast should return one result").as_ref()]);
    }

    if !output_type.shape().dimensions().is_empty() {
        let scalar_tensor_type = context.tensor_type(lower_element_type(data_type, context)?, &[], None, location)?;
        let scalar_elements = lower_constant_elements_attribute(data_type, scalar_tensor_type, integer_value, context)?;
        let scalar_constant = block.append_operation(stable_hlo::constant(scalar_elements, location)?)?;
        let broadcast = block.append_operation(stable_hlo::broadcast(
            scalar_constant.result(0).unwrap().as_ref(),
            tensor_type,
            &[],
            location,
        )?)?;
        return Ok(vec![broadcast.result(0).expect("stablehlo.broadcast should return one result").as_ref()]);
    }
    let elements = lower_constant_elements_attribute(data_type, tensor_type, integer_value, context)?;
    let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
    Ok(vec![constant.result(0).expect("stablehlo.constant should return one result").as_ref()])
}

fn lower_like_constant<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayType],
    integer_value: i64,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    if input_values.len() != 1 {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: input_values.len() }.into());
    }
    lower_constant_output(output_types, integer_value, block, context, location)
}

/// Returns the XLA buffer-placement kind string for `memory`, as consumed by the `_xla_buffer_placement` frontend
/// attribute on `annotate_device_placement` custom calls. This mapping is owned by the lowering on purpose: core's
/// [`Memory`] exposes no backend vocabulary (its `Display` rendering is diagnostics-only), mirroring how
/// [`Sharding`] converts to MLIR through backend-owned conversions.
fn memory_placement_kind(memory: Memory) -> &'static str {
    match memory {
        Memory::Device => "device",
        Memory::Host { pinned: true } => "pinned_host",
        Memory::Host { pinned: false } => "unpinned_host",
    }
}

/// Lowers one staged memory transfer to the `stablehlo.custom_call @annotate_device_placement` annotation that
/// XLA's `ConvertMemoryPlacementToInternalAnnotations` and
/// [`HostOffloader`](https://openxla.org/xla/tools_and_passes/host_offloading) passes legalize into memory-space
/// annotated asynchronous copies — exactly the form JAX emits for memory-kind `device_put`s: API version 1,
/// `has_side_effect = true`, an empty `backend_config` string, and the destination kind string carried as
/// `_xla_buffer_placement` inside the `mhlo.frontend_attributes` dictionary. The empty `backend_config` carries no
/// information, but emitting it keeps the rendered custom call byte-identical to JAX's so module diffs against JAX
/// stay clean.
///
/// Placement does not affect the MLIR tensor type, so the result type is the operand's type unchanged. Identity
/// transfers (destination equal to the operand's current space) still lower to the annotation: placement round
/// trips are meaningful to `HostOffloader` and must not be optimized away here.
fn lower_transfer_to_memory<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    destination: Memory,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    check_count!("input", input_values, 1, ProgramError);
    let empty_backend_config = context.string_attribute("");
    let mut operation = stable_hlo::custom_call(
        input_values,
        "annotate_device_placement",
        true,
        Some(empty_backend_config.as_ref()),
        CustomCallApiVersion::Original,
        &[],
        None,
        &[],
        None,
        &[input_values[0].r#type()?],
        location,
    )?;
    operation.set_discardable_attribute(
        "mhlo.frontend_attributes",
        context.dictionary_attribute(&[context.named_attribute(
            context.identifier("_xla_buffer_placement"),
            context.string_attribute(memory_placement_kind(destination)),
        )]),
    );
    let operation = block.append_operation(operation)?;
    Ok(vec![operation.result(0).expect("stablehlo.custom_call should return one result").as_ref()])
}

impl<V> LowerableXlaOperation<V> for XlaOperation<V>
where
    V: MlirLowerableValue + BooleanLike,
{
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        match self {
            Self::Zero(_) => {
                check_count!("input", input_values, 0, ProgramError);
                lower_constant_output(output_types, 0, &mut lowerer.block, lowerer.context, lowerer.location)
            }
            Self::ZeroLike(_) => lower_like_constant(
                input_values,
                output_types,
                0,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            Self::One(_) => {
                check_count!("input", input_values, 0, ProgramError);
                lower_constant_output(output_types, 1, &mut lowerer.block, lowerer.context, lowerer.location)
            }
            Self::OneLike(_) => lower_like_constant(
                input_values,
                output_types,
                1,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            Self::Constant(operation) => operation.lower_to_mlir(input_values, output_types, mode, lowerer),
            Self::Fill(operation) => <FillOperation<ArrayType, Scalar> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Iota(operation) => <IotaOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Neg(operation) => <NegOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Add(operation) => <AddOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Sub(operation) => <SubOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Mul(operation) => <MulOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Div(operation) => <DivOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Sin(operation) => <SinOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Cos(operation) => <CosOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Atan2(operation) => <Atan2Operation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Exponential(operation) => <ExponentialOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Logarithm(operation) => <LogarithmOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::SquareRoot(operation) => <SquareRootOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Abs(operation) => <AbsOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Complex(operation) => <ComplexOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Conjugate(operation) => <ConjugateOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Real(operation) => <RealOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Imaginary(operation) => <ImaginaryOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::StopGradient(_) => {
                check_count!("input", input_values, 1, ProgramError);
                Ok(vec![input_values[0]])
            }
            Self::Tag(_) => {
                check_count!("input", input_values, 1, ProgramError);
                Ok(vec![input_values[0]])
            }
            // `print` is the identity on its dataflow output; its observable effect lowers to a host-callback
            // custom call that consumes and produces a StableHLO token, so the effect ordering rides the scope's
            // token chain instead of the value dataflow.
            Self::Print(operation) => {
                check_count!("input", input_values, 1, ProgramError);
                lower_print_to_custom_call(
                    operation.label(),
                    input_values[0],
                    &mut lowerer.token,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?;
                Ok(vec![input_values[0]])
            }
            Self::TransferToMemory(operation) => lower_transfer_to_memory(
                operation.destination(),
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            Self::Dot(operation) => <DotOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Transpose(operation) => <TransposeOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Reshape(operation) => <ReshapeOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Reshard(operation) => {
                lower_sharding_constraint(input_values, operation.sharding(), &mut lowerer.block, lowerer.location)
            }
            Self::ShardingConstraint(operation) => {
                lower_sharding_constraint(input_values, operation.sharding(), &mut lowerer.block, lowerer.location)
            }
            Self::Broadcast(operation) => <BroadcastOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            Self::Slice(operation) => {
                let result = lowerer.block.append_operation(stable_hlo::slice(
                    input_values[0],
                    operation.start_indices(),
                    operation.limit_indices(),
                    operation.strides(),
                    lowerer.location,
                )?)?;
                Ok(vec![result.result(0).expect("stablehlo.slice should return one result").as_ref()])
            }
            Self::UpdateSlice(operation) => {
                let index_values = lower_static_index_constants(
                    operation.start_indices(),
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?;
                let result = lowerer.block.append_operation(stable_hlo::dynamic_update_slice(
                    input_values[0],
                    input_values[1],
                    index_values.as_slice(),
                    lowerer.location,
                )?)?;
                Ok(vec![result.result(0).expect("stablehlo.dynamic_update_slice should return one result").as_ref()])
            }
            Self::DynamicSlice(operation) => {
                let result = lowerer.block.append_operation(stable_hlo::dynamic_slice(
                    input_values[0],
                    &input_values[1..],
                    operation.sizes(),
                    lowerer.location,
                )?)?;
                Ok(vec![result.result(0).expect("stablehlo.dynamic_slice should return one result").as_ref()])
            }
            Self::DynamicUpdateSlice(_) => {
                let result = lowerer.block.append_operation(stable_hlo::dynamic_update_slice(
                    input_values[0],
                    input_values[1],
                    &input_values[2..],
                    lowerer.location,
                )?)?;
                Ok(vec![result.result(0).expect("stablehlo.dynamic_update_slice should return one result").as_ref()])
            }
            Self::Pad(operation) => {
                let edge_padding_low: Vec<i64> =
                    operation.edge_padding_low().iter().map(|&padding| padding as i64).collect();
                let edge_padding_high: Vec<i64> =
                    operation.edge_padding_high().iter().map(|&padding| padding as i64).collect();
                let result = lowerer.block.append_operation(stable_hlo::pad(
                    input_values[0],
                    input_values[1],
                    edge_padding_low.as_slice(),
                    edge_padding_high.as_slice(),
                    operation.interior_padding(),
                    lowerer.location,
                )?)?;
                Ok(vec![result.result(0).expect("stablehlo.pad should return one result").as_ref()])
            }
            Self::Concatenate(operation) => {
                reject_dynamic_concatenate_output(output_types)?;
                let result = lowerer.block.append_operation(stable_hlo::concatenate(
                    input_values,
                    operation.axis(),
                    lowerer.location,
                )?)?;
                Ok(vec![result.result(0).expect("stablehlo.concatenate should return one result").as_ref()])
            }
            Self::Gather(operation) => lower_gather_to_mlir(
                operation,
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            Self::Scatter(operation) => lower_scatter_to_mlir(
                operation,
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            Self::Reduce(operation) => {
                check_count!("output", output_types, 1, ProgramError);
                let value = lower_reduce_to_mlir(
                    operation.kind(),
                    operation.axes(),
                    input_values[0],
                    &output_types[0],
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?;
                Ok(vec![value])
            }
            Self::Compare(operation) => {
                let value = lower_compare_to_mlir(
                    operation.direction(),
                    input_values[0],
                    input_values[1],
                    &mut lowerer.block,
                    lowerer.location,
                )?;
                Ok(vec![value])
            }
            Self::Not(_) => {
                let result = lowerer.block.append_operation(stable_hlo::not(input_values[0], lowerer.location)?)?;
                Ok(vec![result.result(0).expect("stablehlo.not should return one result").as_ref()])
            }
            Self::And(_) => {
                let result = lowerer.block.append_operation(stable_hlo::and(
                    input_values[0],
                    input_values[1],
                    lowerer.location,
                )?)?;
                Ok(vec![result.result(0).expect("stablehlo.and should return one result").as_ref()])
            }
            Self::Or(_) => {
                let result = lowerer.block.append_operation(stable_hlo::or(
                    input_values[0],
                    input_values[1],
                    lowerer.location,
                )?)?;
                Ok(vec![result.result(0).expect("stablehlo.or should return one result").as_ref()])
            }
            Self::Xor(_) => {
                let result = lowerer.block.append_operation(stable_hlo::xor(
                    input_values[0],
                    input_values[1],
                    lowerer.location,
                )?)?;
                Ok(vec![result.result(0).expect("stablehlo.xor should return one result").as_ref()])
            }
            Self::Collective(operation) => {
                // This plain dispatch serves nested programs (control-flow bodies, inlined custom-derivative and
                // rematerialized primals), which can sit inside a shard_map manual region: the threaded
                // `CollectiveLoweringState` resolves the collective's mesh axis there and errors outside manual
                // regions (a batched axis would have been consumed into a `Reduce` at trace time).
                check_count!("input", input_values, 1, ProgramError);
                check_count!("output", output_types, 1, ProgramError);
                let collective_state = lowerer.collective_state.clone();
                let result = lower_collective_to_all_reduce(
                    operation,
                    &collective_state,
                    input_values[0],
                    &output_types[0],
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?;
                Ok(vec![result])
            }
            Self::AxisIndex(operation) => {
                check_count!("input", input_values, 0, ProgramError);
                check_count!("output", output_types, 1, ProgramError);
                let collective_state = lowerer.collective_state.clone();
                let result = lower_axis_index_to_coordinate(
                    operation,
                    &collective_state,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?;
                Ok(vec![result])
            }
            Self::Select(_) => {
                let result = lowerer.block.append_operation(stable_hlo::select(
                    input_values[0],
                    input_values[1],
                    input_values[2],
                    lowerer.location,
                )?)?;
                Ok(vec![result.result(0).expect("stablehlo.select should return one result").as_ref()])
            }
            Self::Condition(condition) => condition.lower_to_mlir(input_values, output_types, mode, lowerer),
            Self::While(while_operation) => while_operation.lower_to_mlir(input_values, output_types, mode, lowerer),
            Self::Scan(scan) => scan.lower_to_mlir(input_values, output_types, mode, lowerer),
            Self::CustomJvp(operation) => lower_nested_program_inline(
                operation.primal(),
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
                false,
                lowerer.nested_functions.as_ref(),
                &lowerer.collective_state,
                &mut lowerer.token,
            ),
            Self::CustomVjp(operation) => lower_nested_program_inline(
                operation.primal(),
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
                false,
                lowerer.nested_functions.as_ref(),
                &lowerer.collective_state,
                &mut lowerer.token,
            ),
            // The opaque custom-VJP tangent carrier is a forward-mode tangent map that reverse mode transposes away
            // before lowering, so it never reaches the backend; reaching here means a forward-mode use of `custom_vjp`
            // slipped through, which is reverse-mode-only.
            Self::CustomVjpTangent(operation) => Err(ProgramError::UnsupportedOperation {
                message: format!("operation `{}` cannot be lowered to StableHLO", operation.name()),
            }
            .into()),
            Self::Rematerialize(operation) => lower_nested_program_inline(
                operation.primal(),
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
                false,
                lowerer.nested_functions.as_ref(),
                &lowerer.collective_state,
                &mut lowerer.token,
            ),
            Self::JitCall(jit_call_op) => lower_jit_call(
                jit_call_op.program_rc(),
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
                lowerer.nested_functions.as_ref(),
                &lowerer.collective_state,
                &mut lowerer.token,
            ),
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
                    &lowerer.collective_state,
                )
            }
        }
    }
}

impl<V: MlirLowerableValue + BooleanLike, O> LowerableXlaOperation<V> for ConditionOperation<V, O>
where
    O: LowerableXlaOperation<V>,
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

impl<V: MlirLowerableValue + BooleanLike, O, Payload> LowerableXlaOperation<V> for WhileOperation<V, O, Payload>
where
    O: LowerableXlaOperation<V>,
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

impl<V: MlirLowerableValue + BooleanLike, O, Capture, Payload> LowerableXlaOperation<V>
    for ScanOperation<V, O, Capture, Payload>
where
    Capture: Value<Type = ArrayType>,
    O: LowerableXlaOperation<V>,
{
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lowerer.lower_scan(self, input_values)
    }
}

/// Lowers a sharding-control operation to the Shardy
/// [`sdy.sharding_constraint`](https://openxla.org/shardy/sdy_dialect#sdysharding_constraint-sdyshardingconstraintop)
/// operation. Both the tracked [`ArrayOperation::Reshard`](ryft_core::operations::ReshardOperation) sharding
/// transition and the [`ArrayOperation::ShardingConstraint`](ryft_core::operations::ShardingConstraintOperation)
/// auto-axis propagation hint emit this single operation; they differ only in their `ryft` type-level semantics
/// (which mesh axes they govern and how they transpose), not in the emitted MLIR.
fn lower_sharding_constraint<'b, 'c: 'b, 't: 'c>(
    input_values: &[ValueRef<'b, 'c, 't>],
    sharding: &Sharding,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    check_count!("input", input_values, 1, ProgramError);
    let sharding_attribute = sharding.to_mlir(location)?;
    let operation =
        block.append_operation(shardy::sharding_constraint(input_values[0], sharding_attribute, location)?)?;
    Ok(vec![operation.result(0).expect("sdy.sharding_constraint should return one result").as_ref()])
}

/// Returns the current StableHLO effect token of one lowering scope, creating it lazily with a zero-operand
/// `stablehlo.after_all` at the current insertion point when the scope has not needed a token yet.
///
/// One token chain orders all effectful instructions lowered into one scope (a function body or a control-flow
/// region), in program order. The chain is dropped at the end of the scope, so this v1 design orders effects within
/// one dispatch; carrying tokens across separately dispatched executions is out of scope.
fn current_or_new_token<'b, 'c: 'b, 't: 'c>(
    token: &mut Option<ValueRef<'b, 'c, 't>>,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    if let Some(token) = token {
        return Ok(*token);
    }
    let created = block.append_operation(stable_hlo::after_all::<ValueRef, _>(&[], location)?)?;
    let created = created.result(0).expect("stablehlo.after_all should return one result").as_ref();
    *token = Some(created);
    Ok(created)
}

/// Lowers one `print` operation to the [`PRINT_CUSTOM_CALL_TARGET`] host-callback custom call and advances the
/// scope's effect token chain past it.
///
/// The emitted operation follows the calling convention decoded by the FFI handler registered by
/// [`ensure_print_handler_registered`](crate::experimental::debugging::ensure_print_handler_registered):
///
/// ```mlir
/// %token_out = stablehlo.custom_call @"ryft.print"(%value, %token_in)
///   {api_version = 4 : i32, backend_config = {label = "<label>"}, has_side_effect = true}
///   : (tensor<...>, !stablehlo.token) -> !stablehlo.token
/// ```
///
/// The custom call's only result is the continuation token; the `print` operation's dataflow output is the
/// forwarded input value, which the caller returns directly.
fn lower_print_to_custom_call<'b, 'c: 'b, 't: 'c>(
    label: &str,
    value: ValueRef<'b, 'c, 't>,
    token: &mut Option<ValueRef<'b, 'c, 't>>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), LoweringError> {
    let input_token = current_or_new_token(token, block, location)?;
    let token_type = context.stable_hlo_token_type()?;
    let backend_config = context.dictionary_attribute(&[
        context.named_attribute(context.identifier(PRINT_LABEL_ATTRIBUTE), context.string_attribute(label))
    ]);
    let operation = block.append_operation(stable_hlo::custom_call(
        &[value, input_token],
        PRINT_CUSTOM_CALL_TARGET,
        true,
        Some(backend_config.as_ref()),
        CustomCallApiVersion::TypedFfi,
        &[],
        None,
        &[],
        None,
        &[token_type],
        location,
    )?)?;
    *token = Some(operation.result(0).expect("the print custom call should return one token result").as_ref());
    Ok(())
}

impl<V> LowerableXlaOperation<V> for ArrayOperation<V>
where
    V: MlirLowerableValue + BooleanLike + Slice + UpdateSlice + Reshape,
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
                    return Err(ProgramError::InvalidInputCount { expected: 0, actual: input_values.len() }.into());
                }
                lower_constant_output(output_types, 0, &mut lowerer.block, lowerer.context, lowerer.location)
            }
            ArrayOperation::One(_) => {
                if !input_values.is_empty() {
                    return Err(ProgramError::InvalidInputCount { expected: 0, actual: input_values.len() }.into());
                }
                lower_constant_output(output_types, 1, &mut lowerer.block, lowerer.context, lowerer.location)
            }
            ArrayOperation::Constant(constant) => constant.lower_to_mlir(input_values, output_types, mode, lowerer),
            ArrayOperation::Fill(fill) => {
                <FillOperation<ArrayType, Scalar> as LowerableXlaOperation<V>>::lower_to_mlir(
                    fill,
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            ArrayOperation::Iota(iota) => <IotaOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                iota,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Add(operation) => <AddOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Sub(operation) => <SubOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Mul(operation) => <MulOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Div(operation) => <DivOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Neg(operation) => <NegOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Sin(operation) => <SinOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Cos(operation) => <CosOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Atan2(operation) => <Atan2Operation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Exponential(operation) => {
                <ExponentialOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                    operation,
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            ArrayOperation::Logarithm(operation) => <LogarithmOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::SquareRoot(operation) => <SquareRootOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Abs(operation) => <AbsOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Complex(operation) => <ComplexOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Conjugate(operation) => <ConjugateOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Real(operation) => <RealOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Imaginary(operation) => <ImaginaryOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            // `stop_gradient` only affects differentiation; by lowering time it is the identity, so
            // forward the operand without emitting any MLIR operation (matching JAX's lowering).
            ArrayOperation::StopGradient(_) => {
                if input_values.len() != 1 {
                    return Err(ProgramError::InvalidInputCount { expected: 1, actual: input_values.len() }.into());
                }
                Ok(vec![input_values[0]])
            }
            // `tag` only affects rematerialization policies; by lowering time it is the identity, so
            // forward the operand without emitting any MLIR operation.
            ArrayOperation::Tag(_) => {
                if input_values.len() != 1 {
                    return Err(ProgramError::InvalidInputCount { expected: 1, actual: input_values.len() }.into());
                }
                Ok(vec![input_values[0]])
            }
            // `print` is the identity on its dataflow output; its observable effect lowers to a host-callback
            // custom call that consumes and produces a StableHLO token, so the effect ordering rides the scope's
            // token chain instead of the value dataflow.
            ArrayOperation::Print(operation) => {
                check_count!("input", input_values, 1, ProgramError);
                lower_print_to_custom_call(
                    operation.label(),
                    input_values[0],
                    &mut lowerer.token,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?;
                Ok(vec![input_values[0]])
            }
            ArrayOperation::TransferToMemory(operation) => lower_transfer_to_memory(
                operation.destination(),
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            // Custom-derivative calls lower as their primal program: the derivative programs only exist for the
            // benefit of transforms and never reach the backend.
            ArrayOperation::CustomJvp(operation) => lower_nested_program_inline(
                operation.primal(),
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
                false,
                lowerer.nested_functions.as_ref(),
                &lowerer.collective_state,
                &mut lowerer.token,
            ),
            ArrayOperation::CustomVjp(operation) => lower_nested_program_inline(
                operation.primal(),
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
                false,
                lowerer.nested_functions.as_ref(),
                &lowerer.collective_state,
                &mut lowerer.token,
            ),
            // The opaque custom-VJP tangent carrier is a forward-mode tangent map that reverse mode transposes away
            // before lowering, so it never reaches the backend; reaching here means a forward-mode use of `custom_vjp`
            // slipped through, which is reverse-mode-only.
            ArrayOperation::CustomVjpTangent(operation) => Err(ProgramError::UnsupportedOperation {
                message: format!("operation `{}` cannot be lowered to StableHLO", operation.name()),
            }
            .into()),
            ArrayOperation::Rematerialize(operation) => lower_nested_program_inline(
                operation.primal(),
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
                false,
                lowerer.nested_functions.as_ref(),
                &lowerer.collective_state,
                &mut lowerer.token,
            ),
            ArrayOperation::ZeroLike(_) => lower_like_constant(
                input_values,
                output_types,
                0,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            ArrayOperation::OneLike(_) => lower_like_constant(
                input_values,
                output_types,
                1,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            ArrayOperation::Transpose(operation) => <TransposeOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Dot(operation) => <DotOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Reshape(operation) => <ReshapeOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Reshard(operation) => {
                lower_sharding_constraint(input_values, operation.sharding(), &mut lowerer.block, lowerer.location)
            }
            ArrayOperation::ShardingConstraint(operation) => {
                lower_sharding_constraint(input_values, operation.sharding(), &mut lowerer.block, lowerer.location)
            }
            ArrayOperation::Broadcast(operation) => <BroadcastOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Reduce(operation) => {
                check_count!("output", output_types, 1, ProgramError);
                let value = lower_reduce_to_mlir(
                    operation.kind(),
                    operation.axes(),
                    input_values[0],
                    &output_types[0],
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?;
                Ok(vec![value])
            }
            ArrayOperation::Compare(operation) => {
                let value = lower_compare_to_mlir(
                    operation.direction(),
                    input_values[0],
                    input_values[1],
                    &mut lowerer.block,
                    lowerer.location,
                )?;
                Ok(vec![value])
            }
            ArrayOperation::Not(_) => {
                let result = lowerer.block.append_operation(stable_hlo::not(input_values[0], lowerer.location)?)?;
                Ok(vec![result.result(0).expect("stablehlo.not should return one result").as_ref()])
            }
            ArrayOperation::And(_) => {
                let result = lowerer.block.append_operation(stable_hlo::and(
                    input_values[0],
                    input_values[1],
                    lowerer.location,
                )?)?;
                Ok(vec![result.result(0).expect("stablehlo.and should return one result").as_ref()])
            }
            ArrayOperation::Or(_) => {
                let result = lowerer.block.append_operation(stable_hlo::or(
                    input_values[0],
                    input_values[1],
                    lowerer.location,
                )?)?;
                Ok(vec![result.result(0).expect("stablehlo.or should return one result").as_ref()])
            }
            ArrayOperation::Xor(_) => {
                let result = lowerer.block.append_operation(stable_hlo::xor(
                    input_values[0],
                    input_values[1],
                    lowerer.location,
                )?)?;
                Ok(vec![result.result(0).expect("stablehlo.xor should return one result").as_ref()])
            }
            ArrayOperation::Collective(operation) => {
                // This plain dispatch serves nested programs (control-flow bodies, inlined custom-derivative and
                // rematerialized primals), which can sit inside a shard_map manual region: the threaded
                // `CollectiveLoweringState` resolves the collective's mesh axis there and errors outside manual
                // regions (a batched axis would have been consumed into a `Reduce` at trace time).
                check_count!("input", input_values, 1, ProgramError);
                check_count!("output", output_types, 1, ProgramError);
                let collective_state = lowerer.collective_state.clone();
                let result = lower_collective_to_all_reduce(
                    operation,
                    &collective_state,
                    input_values[0],
                    &output_types[0],
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?;
                Ok(vec![result])
            }
            ArrayOperation::AxisIndex(operation) => {
                check_count!("input", input_values, 0, ProgramError);
                check_count!("output", output_types, 1, ProgramError);
                let collective_state = lowerer.collective_state.clone();
                let result = lower_axis_index_to_coordinate(
                    operation,
                    &collective_state,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?;
                Ok(vec![result])
            }
            ArrayOperation::Select(_) => {
                let result = lowerer.block.append_operation(stable_hlo::select(
                    input_values[0],
                    input_values[1],
                    input_values[2],
                    lowerer.location,
                )?)?;
                Ok(vec![result.result(0).expect("stablehlo.select should return one result").as_ref()])
            }
            ArrayOperation::Slice(operation) => {
                let result = lowerer.block.append_operation(stable_hlo::slice(
                    input_values[0],
                    operation.start_indices(),
                    operation.limit_indices(),
                    operation.strides(),
                    lowerer.location,
                )?)?;
                Ok(vec![result.result(0).expect("stablehlo.slice should return one result").as_ref()])
            }
            ArrayOperation::UpdateSlice(operation) => {
                let index_values = lower_static_index_constants(
                    operation.start_indices(),
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?;
                let result = lowerer.block.append_operation(stable_hlo::dynamic_update_slice(
                    input_values[0],
                    input_values[1],
                    index_values.as_slice(),
                    lowerer.location,
                )?)?;
                Ok(vec![result.result(0).expect("stablehlo.dynamic_update_slice should return one result").as_ref()])
            }
            ArrayOperation::DynamicSlice(operation) => {
                let result = lowerer.block.append_operation(stable_hlo::dynamic_slice(
                    input_values[0],
                    &input_values[1..],
                    operation.sizes(),
                    lowerer.location,
                )?)?;
                Ok(vec![result.result(0).expect("stablehlo.dynamic_slice should return one result").as_ref()])
            }
            ArrayOperation::DynamicUpdateSlice(_) => {
                let result = lowerer.block.append_operation(stable_hlo::dynamic_update_slice(
                    input_values[0],
                    input_values[1],
                    &input_values[2..],
                    lowerer.location,
                )?)?;
                Ok(vec![result.result(0).expect("stablehlo.dynamic_update_slice should return one result").as_ref()])
            }
            ArrayOperation::Pad(operation) => {
                let edge_padding_low: Vec<i64> =
                    operation.edge_padding_low().iter().map(|&padding| padding as i64).collect();
                let edge_padding_high: Vec<i64> =
                    operation.edge_padding_high().iter().map(|&padding| padding as i64).collect();
                let result = lowerer.block.append_operation(stable_hlo::pad(
                    input_values[0],
                    input_values[1],
                    edge_padding_low.as_slice(),
                    edge_padding_high.as_slice(),
                    operation.interior_padding(),
                    lowerer.location,
                )?)?;
                Ok(vec![result.result(0).expect("stablehlo.pad should return one result").as_ref()])
            }
            ArrayOperation::Concatenate(operation) => {
                reject_dynamic_concatenate_output(output_types)?;
                let result = lowerer.block.append_operation(stable_hlo::concatenate(
                    input_values,
                    operation.axis(),
                    lowerer.location,
                )?)?;
                Ok(vec![result.result(0).expect("stablehlo.concatenate should return one result").as_ref()])
            }
            ArrayOperation::Gather(operation) => lower_gather_to_mlir(
                operation,
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            ArrayOperation::Scatter(operation) => lower_scatter_to_mlir(
                operation,
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            ArrayOperation::Condition(condition) => condition.lower_to_mlir(input_values, output_types, mode, lowerer),
            ArrayOperation::While(while_operation) => {
                while_operation.lower_to_mlir(input_values, output_types, mode, lowerer)
            }
            ArrayOperation::Scan(scan) => scan.lower_to_mlir(input_values, output_types, mode, lowerer),
        }
    }
}

/// Lowering state consulted when lowering collectives: the module-scoped channel-id allocator (each channeled
/// StableHLO collective in a module must carry a distinct channel id, so one shared counter serves every manual region
/// in the module) and the innermost enclosing `sdy.manual_computation` region's [`ShardMap`] (whose manual device mesh
/// axes collectives resolve by name), or `None` outside manual regions.
///
/// The state is created once per lowered module, cloned per instruction lowerer (both shared pieces sit behind
/// [`Rc`]s), and [`enter_manual_region`](Self::enter_manual_region) derives the state used inside one manual region's
/// body — sharing the module's channel allocator while swapping in the region's [`ShardMap`].
#[derive(Clone)]
pub(crate) struct CollectiveLoweringState {
    /// Module-scoped counter producing the next collective channel id.
    channel_ids: Rc<Cell<usize>>,

    /// Innermost enclosing manual region's [`ShardMap`], or `None` outside manual regions.
    manual_shard_map: Option<Rc<ShardMap>>,
}

impl CollectiveLoweringState {
    /// Creates the lowering state for one module, outside any manual region.
    pub(crate) fn new() -> Self {
        Self { channel_ids: Rc::new(Cell::new(1)), manual_shard_map: None }
    }

    /// Derives the state used inside one `sdy.manual_computation` region's body: the module's channel allocator is
    /// shared and the region's [`ShardMap`] becomes the innermost manual region.
    pub(crate) fn enter_manual_region(&self, shard_map: ShardMap) -> Self {
        Self { channel_ids: self.channel_ids.clone(), manual_shard_map: Some(Rc::new(shard_map)) }
    }

    /// Returns the innermost enclosing manual region's [`ShardMap`], or `None` outside manual regions.
    pub(crate) fn manual_shard_map(&self) -> Option<&ShardMap> {
        self.manual_shard_map.as_deref()
    }

    /// Returns a fresh module-unique channel id for one channeled collective.
    pub(crate) fn next_channel_id(&self) -> usize {
        let channel_id = self.channel_ids.get();
        self.channel_ids.set(channel_id + 1);
        channel_id
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

    /// Shared private functions emitted for deduplicated `jit_call` callees, consulted at `jit_call` lowering sites.
    /// Shared via [`Rc`] so it threads through nested lowering scopes without lifetime entanglement.
    nested_functions: Option<Rc<JitCallFunctionMap>>,

    /// Current StableHLO effect token of the lowering scope this lowerer emits into, or `None` when the scope has
    /// not lowered an effectful instruction yet. Refer to the documentation of the equivalent
    /// [`PlainMlirLowerer`] field for the copy-in/copy-out threading protocol.
    token: Option<ValueRef<'b, 'c, 't>>,

    /// Collective lowering state of the lowering scope this lowerer emits into. Refer to the documentation of
    /// [`CollectiveLoweringState`] for more information.
    collective_state: CollectiveLoweringState,
}

impl<'b, 'c: 'b, 't: 'c> ShardMapMlirLowerer<'b, 'c, 't> {
    /// Creates a shard-map MLIR lowerer for operations emitted into `block`.
    pub(crate) fn new(
        block: BlockRef<'b, 'c, 't>,
        context: &'c MlirContext<'t>,
        location: LocationRef<'c, 't>,
    ) -> Self {
        Self {
            block,
            context,
            location,
            nested_functions: None,
            token: None,
            collective_state: CollectiveLoweringState::new(),
        }
    }

    /// Attaches the shared deduplicated `jit_call` functions consulted while lowering.
    pub(crate) fn with_nested_functions(mut self, nested_functions: Option<Rc<JitCallFunctionMap>>) -> Self {
        self.nested_functions = nested_functions;
        self
    }

    /// Attaches the current effect token of the enclosing lowering scope.
    pub(crate) fn with_token(mut self, token: Option<ValueRef<'b, 'c, 't>>) -> Self {
        self.token = token;
        self
    }

    /// Attaches the collective lowering state of the enclosing lowering scope.
    pub(crate) fn with_collective_state(mut self, collective_state: CollectiveLoweringState) -> Self {
        self.collective_state = collective_state;
        self
    }

    /// Lowers one tensor type inside this lowering context.
    pub(crate) fn lower_tensor_type(
        &self,
        array_type: &ArrayType,
    ) -> Result<ryft_mlir::TensorTypeRef<'c, 't>, LoweringError> {
        lower_tensor_type(array_type, self.context, self.location)
    }

    /// Lowers one nested condition operation inside this lowering context.
    pub(crate) fn lower_condition<V: MlirLowerableValue + BooleanLike, O>(
        &mut self,
        condition_op: &ConditionOperation<V, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        O: LowerableXlaOperation<V>,
    {
        lower_condition_to_if(
            condition_op,
            input_values,
            &mut self.block,
            self.context,
            self.location,
            self.nested_functions.as_ref(),
            &self.collective_state,
            &mut self.token,
        )
    }

    /// Lowers one nested while operation inside this lowering context.
    pub(crate) fn lower_while<V: MlirLowerableValue + BooleanLike, O, Payload>(
        &mut self,
        while_op: &WhileOperation<V, O, Payload>,
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        O: LowerableXlaOperation<V>,
    {
        lower_while_to_while(
            while_op,
            input_values,
            &mut self.block,
            self.context,
            self.location,
            self.nested_functions.as_ref(),
            &self.collective_state,
            &mut self.token,
        )
    }

    /// Lowers one nested scan operation inside this lowering context.
    pub(crate) fn lower_scan<V: MlirLowerableValue + BooleanLike, O, Capture, Payload>(
        &mut self,
        scan_op: &ScanOperation<V, O, Capture, Payload>,
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        Capture: Value<Type = ArrayType>,
        O: LowerableXlaOperation<V>,
    {
        lower_scan_to_while(
            scan_op.body(),
            scan_op.carry_count(),
            scan_op.length(),
            scan_op.reverse(),
            scan_op.unroll(),
            input_values,
            &mut self.block,
            self.context,
            self.location,
            self.nested_functions.as_ref(),
            &self.collective_state,
            &mut self.token,
        )
    }

    /// Lowers one nested Shardy manual computation operation inside this lowering context.
    pub(crate) fn lower_manual_computation<
        'o,
        ProgramInput: Parameterized<XlaConstant>,
        ProgramOutput: Parameterized<XlaConstant>,
    >(
        &mut self,
        outer_inputs: &[ValueRef<'b, 'c, 't>],
        shard_map: &ShardMap,
        program: &XlaProgram<ProgramInput, ProgramOutput>,
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
            &self.collective_state,
        )
    }
}

/// Lowers a traced shard-map program to a textual StableHLO/Shardy MLIR module.
pub(crate) fn to_mlir_module<
    'o,
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
    ProgramInput: Parameterized<XlaConstant>,
    ProgramOutput: Parameterized<XlaConstant>,
    S: AsRef<str>,
>(
    shard_map: &ShardMap,
    program: &XlaProgram<ProgramInput, ProgramOutput>,
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
        let collective_state = CollectiveLoweringState::new();
        let manual_results = lower_manual_computation(
            &mut function_block_ref,
            outer_inputs.as_slice(),
            shard_map,
            program,
            local_input_types.as_slice(),
            global_output_types.as_slice(),
            &context,
            location.as_ref(),
            &collective_state,
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
    program: &XlaProgram<ProgramInput, ProgramOutput>,
    capture_types: &[ArrayType],
    global_input_types: &Input,
    global_output_types: &Output,
    function_name: S,
    arg_shardings: Option<&[Sharding]>,
    result_shardings: Option<&[Sharding]>,
) -> Result<String, LoweringError>
where
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
    ProgramInput: Parameterized<XlaConstant>,
    ProgramOutput: Parameterized<XlaConstant>,
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

    // Deduplicate `jit_call` callees that occur more than once into shared private `func.func`s, so repeated nested
    // programs (identical transformer blocks, or the per-block primal and pullback programs produced by `grad`) lower
    // to one function plus N `func.call`s instead of N inlined copies. The map is empty for modules without repeated
    // calls, in which case every `jit_call` inlines exactly as before.
    let nested_functions = Rc::new(collect_jit_call_functions(program));
    {
        let mut module_block = module.body()?;
        for key in &nested_functions.order {
            let function = nested_functions.functions.get(key).expect("ordered keys are present in the map");
            emit_jit_call_function(&mut module_block, function, &nested_functions, &context, location.as_ref())?;
        }
    }

    let capture_tensor_types = capture_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let global_input_tensor_types = global_input_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let global_output_tensor_types = global_output_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let argument_tensor_types = capture_tensor_types
        .iter()
        .copied()
        .chain(global_input_tensor_types.iter().copied())
        .collect::<Vec<_>>();
    let arg_sharding_attributes = match arg_shardings {
        Some(shardings) => {
            if shardings.len() != argument_tensor_types.len() {
                return Err(LoweringError::InvalidShardingCount {
                    kind: "argument",
                    expected: argument_tensor_types.len(),
                    actual: shardings.len(),
                });
            }
            Some(shardings.iter().map(|sharding| sharding.to_mlir(location)).collect::<Result<Vec<_>, _>>()?)
        }
        None => None,
    };
    let result_sharding_attributes = match result_shardings {
        Some(shardings) => {
            if shardings.len() != global_output_tensor_types.len() {
                return Err(LoweringError::InvalidShardingCount {
                    kind: "result",
                    expected: global_output_tensor_types.len(),
                    actual: shardings.len(),
                });
            }
            Some(shardings.iter().map(|sharding| sharding.to_mlir(location)).collect::<Result<Vec<_>, _>>()?)
        }
        None => None,
    };
    let function_arguments = argument_tensor_types
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
            argument_tensor_types
                .iter()
                .map(|tensor_type| (*tensor_type, location))
                .collect::<Vec<_>>()
                .as_slice(),
        );
        {
            let mut function_block_ref = function_block.as_ref();
            let capture_values = (0..capture_tensor_types.len())
                .map(|index| function_block.argument(index).expect("capture block arguments should exist").as_ref())
                .collect::<Vec<_>>();
            let outputs = lower_program_outputs(
                program,
                capture_values.as_slice(),
                &mut function_block_ref,
                &context,
                location.as_ref(),
                Some(&nested_functions),
                &CollectiveLoweringState::new(),
            )?;
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
pub(crate) trait MlirLowerableValue: Value<Type = ArrayType> + 'static {
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

impl MlirLowerableValue for XlaConstant {
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        _tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        _context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
        Err(LoweringError::MissingCapturedConstant { index: self.index() })
    }
}

/// [`ArrayType`] is used as the value representation for abstract linear XLA programs. It can type
/// program atoms, but it is not a concrete literal; lowering paths that need a real value must
/// supply it through captured arguments instead of materializing it from type metadata.
impl MlirLowerableValue for ArrayType {
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        _tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        _context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
        Err(LoweringError::AbstractValueLiteral { array_type: self.clone() })
    }

    fn to_scalar_dense_elements_attribute<'c, 't>(
        &self,
        _tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        _context: &'c MlirContext<'t>,
    ) -> Result<Option<DenseElementsAttributeRef<'c, 't>>, LoweringError> {
        Ok(None)
    }
}

/// Concrete test/benchmark value lowering used by MLIR snapshot tooling.
#[cfg(any(test, feature = "benchmarking"))]
impl MlirLowerableValue for TestArray {
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
        // Integer-typed payloads (e.g., dynamic slicing start indices) are stored in-band as `f64` values, so they
        // convert through integer dense attributes matching the lowered integer tensor type.
        let data_type = self.r#type.data_type();
        let attribute = match data_type {
            DataType::I32 => {
                let values = self.values.iter().map(|value| *value as i32).collect::<Vec<_>>();
                context
                    .dense_i32_elements_attribute(tensor_type, values.as_slice())
                    .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type })?
                    .cast::<DenseElementsAttributeRef>()
            }
            DataType::I64 => {
                let values = self.values.iter().map(|value| *value as i64).collect::<Vec<_>>();
                context
                    .dense_i64_elements_attribute(tensor_type, values.as_slice())
                    .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type })?
                    .cast::<DenseElementsAttributeRef>()
            }
            _ => context
                .dense_f64_elements_attribute(tensor_type, self.values.as_slice())
                .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type })?
                .cast::<DenseElementsAttributeRef>(),
        };
        attribute.ok_or(LoweringError::InvalidDenseElementsAttribute { data_type })
    }

    fn to_scalar_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<Option<DenseElementsAttributeRef<'c, 't>>, LoweringError> {
        if self.values.len() != 1 {
            return Ok(None);
        }
        Ok(Some(self.to_dense_elements_attribute(tensor_type, context)?))
    }
}

/// Lowers a plain traced `tracing_v2` program to a textual StableHLO MLIR module.
#[cfg(any(test, feature = "benchmarking"))]
pub(crate) fn to_mlir_module_for_plain_program<
    V: MlirLowerableValue + BooleanLike,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
    O: LowerableXlaOperation<V>,
    S: AsRef<str>,
>(
    program: &Program<V, O, Input, Output>,
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
    program: &XlaProgram<ProgramInput, ProgramOutput>,
    existing: Option<LogicalMesh>,
) -> Result<Option<LogicalMesh>, LoweringError>
where
    ProgramInput: Parameterized<XlaConstant>,
    ProgramOutput: Parameterized<XlaConstant>,
{
    let mut mesh = existing;
    for instruction in program.instructions() {
        match &instruction.operation() {
            XlaOperation::JitCall(jit_call_op) => {
                mesh = collect_nested_sharding_mesh(jit_call_op.program(), mesh)?;
            }
            XlaOperation::ShardMap(shard_map_op) => {
                let body = shard_map_op.body();
                mesh = Some(match mesh.take() {
                    Some(existing_mesh) => merge_logical_meshes(&existing_mesh, body.shard_map().mesh())?,
                    None => body.shard_map().mesh().clone(),
                });
                mesh = collect_nested_sharding_mesh(body.program(), mesh)?;
            }
            XlaOperation::Condition(condition_op) => {
                mesh = collect_nested_sharding_mesh(condition_op.true_branch(), mesh)?;
                mesh = collect_nested_sharding_mesh(condition_op.false_branch(), mesh)?;
            }
            XlaOperation::While(while_op) => {
                mesh = collect_nested_sharding_mesh(while_op.condition(), mesh)?;
                mesh = collect_nested_sharding_mesh(while_op.body(), mesh)?;
            }
            XlaOperation::Scan(scan_op) => {
                mesh = collect_nested_sharding_mesh(scan_op.body(), mesh)?;
            }
            XlaOperation::Reshard(operation) => {
                mesh = Some(match mesh.take() {
                    Some(existing_mesh) => merge_logical_meshes(&existing_mesh, operation.sharding().mesh())?,
                    None => operation.sharding().mesh().clone(),
                });
            }
            XlaOperation::ShardingConstraint(operation) => {
                mesh = Some(match mesh.take() {
                    Some(existing_mesh) => merge_logical_meshes(&existing_mesh, operation.sharding().mesh())?,
                    None => operation.sharding().mesh().clone(),
                });
            }
            _ => {}
        }
    }
    Ok(mesh)
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

/// Rejects a `concatenate` whose output type carries a [`Size::Dynamic`] dimension. StableHLO `concatenate` lowering
/// only supports static shapes; a dynamic-axis concatenate appears only inside unbounded-while pullbacks, which the
/// while lowering already rejects (recommending `with_iteration_bound`), so this surfaces a precise error if a
/// dynamic operand ever reaches the concatenate lowering directly.
fn reject_dynamic_concatenate_output(output_types: &[ArrayType]) -> Result<(), LoweringError> {
    for output_type in output_types {
        static_dimensions(output_type)?;
    }
    Ok(())
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

/// Lowers one nested control-flow branch program into a fresh single-block region.
///
/// `entry_token` is the enclosing scope's effect token, referenced inside the region through StableHLO's implicit
/// region capture (the same mechanism that already feeds `input_values` into `stablehlo.if` branches). When
/// `return_final_token` is set, the region's `stablehlo.return` yields the branch's final effect token as one extra
/// trailing output — the entry token unchanged for a pure branch — so the enclosing operation can expose it as an
/// extra result.
fn lower_control_flow_region<'b, 'c: 'b, 't: 'c, V, O>(
    program: &Program<V, O, Vec<V>, Vec<V>>,
    input_values: &[ValueRef<'b, 'c, 't>],
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
    entry_token: Option<ValueRef<'b, 'c, 't>>,
    return_final_token: bool,
) -> Result<ryft_mlir::DetachedRegion<'c, 't>, LoweringError>
where
    V: MlirLowerableValue + BooleanLike,
    O: LowerableXlaOperation<V>,
{
    let mut region = context.region();
    let block = context.block_with_no_arguments();
    {
        let mut block_ref = block.as_ref();
        let mut region_token = entry_token;
        let mut outputs = lower_nested_program_inline(
            program,
            input_values,
            &mut block_ref,
            context,
            location,
            false,
            nested_functions,
            collective_state,
            &mut region_token,
        )?;
        if return_final_token {
            outputs.push(region_token.expect("token-returning control-flow regions receive an entry token"));
        }
        block_ref.append_operation(stable_hlo::r#return(outputs.as_slice(), location)?)?;
    }
    region.append_block(block)?;
    Ok(region)
}

fn lower_condition_to_if<'b, 'c: 'b, 't: 'c, V, O>(
    condition_op: &ConditionOperation<V, O>,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
    token: &mut Option<ValueRef<'b, 'c, 't>>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: MlirLowerableValue + BooleanLike,
    O: LowerableXlaOperation<V>,
{
    let expected_input_count = condition_op.true_branch().input_types().len() + 1;
    if input_values.len() != expected_input_count {
        return Err(LoweringError::UnsupportedOp {
            op: format!("condition expected {expected_input_count} lowered inputs but got {}", input_values.len()),
        });
    }
    let branch_inputs = &input_values[1..];
    // When either branch is effectful, the enclosing scope's token is captured implicitly by both branch regions
    // (StableHLO `if` regions capture enclosing values, which is also how `branch_inputs` flow in) and each branch
    // returns its own final token as one extra trailing result; that extra `stablehlo.if` result then becomes the
    // scope's current token. Pure conditions emit no token machinery at all.
    let threads_token =
        !condition_op.true_branch().effects().is_pure() || !condition_op.false_branch().effects().is_pure();
    let entry_token = if threads_token { Some(current_or_new_token(token, block, location)?) } else { None };
    let true_branch_region = lower_control_flow_region(
        condition_op.true_branch(),
        branch_inputs,
        context,
        location,
        nested_functions,
        collective_state,
        entry_token,
        threads_token,
    )?;
    let false_branch_region = lower_control_flow_region(
        condition_op.false_branch(),
        branch_inputs,
        context,
        location,
        nested_functions,
        collective_state,
        entry_token,
        threads_token,
    )?;
    let operation = block.append_operation(stable_hlo::r#if(
        input_values[0],
        true_branch_region.into(),
        false_branch_region.into(),
        location,
    )?)?;
    let output_count = condition_op.output_types().len();
    if threads_token {
        *token = Some(
            operation
                .result(output_count)
                .expect("a token-threaded stablehlo.if should return one trailing token result")
                .as_ref(),
        );
    }
    Ok((0..output_count)
        .map(|index| operation.result(index).expect("stablehlo.if should return one result per output").as_ref())
        .collect())
}

fn lower_while_to_while<'b, 'c: 'b, 't: 'c, V, O, Payload>(
    while_op: &WhileOperation<V, O, Payload>,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
    token: &mut Option<ValueRef<'b, 'c, 't>>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: MlirLowerableValue + BooleanLike,
    O: LowerableXlaOperation<V>,
{
    let state_types = while_op.state_types();
    let state_count = state_types.len();
    if input_values.len() != state_count {
        return Err(LoweringError::UnsupportedOp {
            op: format!("while expected {state_count} lowered inputs but got {}", input_values.len()),
        });
    }
    // A batched (non-scalar) predicate lowers with the masked semantics owned by this primitive, mirroring JAX's
    // `_while_lowering`. Rather than re-evaluating the condition in both regions, the per-item predicate is threaded
    // through the loop state as one extra carried value (JAX's `(pred, ..., carry)` layout): the condition region
    // only reduces the carried predicate with a Boolean `or` into the scalar continuation decision, and the body
    // region selects per state element between the body's candidate update and the carried (incoming-state)
    // predicate, then recomputes the predicate on the updated state for the next iteration. The condition is
    // therefore evaluated exactly once per iteration (plus once before the loop to seed the initial predicate),
    // never twice. A batched-predicate loop is always pure (`WhileOperation::new` rejects effects in a
    // batched-predicate loop, since observable effects cannot be masked for finished items), so predicate threading
    // and token threading never coexist.
    let predicate_type = while_op.condition().output_types()[0].clone();
    let batched_predicate = predicate_type.rank() > 0;
    let predicate_dimensions = (0..predicate_type.rank()).collect::<Vec<_>>();
    let predicate_offset = if batched_predicate { 1 } else { 0 };
    // A semantic iteration bound is enforced by threading an internal `i64` iteration counter through the
    // `stablehlo.while` state (element 0, starting at zero and incremented once per body run) and conjoining
    // `counter < bound` into the lowered condition. The counter is internal extra state: the operation's outputs
    // remain exactly the original state elements. Unbounded loops emit no counter machinery at all.
    let iteration_bound = while_op.iteration_bound();
    let counter_offset = if iteration_bound.is_some() { 1 } else { 0 };
    // When the nested programs are effectful, the enclosing scope's effect token is carried through the loop as one
    // extra trailing state element (appended after the states and the optional carried predicate, so the existing
    // state index math is untouched): both regions receive it as one extra block argument, the condition region only
    // reads it (its own final token cannot leave the region and is discarded), the body region threads it through the
    // body's effectful instructions, and the extra `stablehlo.while` result becomes the scope's current token. Pure
    // loops emit no token machinery at all.
    let threads_token = !while_op.condition().effects().is_pure() || !while_op.body().effects().is_pure();
    // State layout: `[counter?, states..., predicate?, token?]`.
    let predicate_index = counter_offset + state_count;
    let token_index = counter_offset + state_count + predicate_offset;
    let mut full_state_types = Vec::with_capacity(counter_offset + state_count + predicate_offset);
    if iteration_bound.is_some() {
        full_state_types.push(ArrayType::scalar(DataType::I64));
    }
    full_state_types.extend(state_types.iter().cloned());
    if batched_predicate {
        full_state_types.push(predicate_type.clone());
    }
    let mut lowered_state_types = full_state_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, context, location).map(|tensor_type| tensor_type.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    if threads_token {
        lowered_state_types.push(context.stable_hlo_token_type()?.as_ref());
    }
    let block_arguments = lowered_state_types.iter().map(|r#type| (*r#type, location)).collect::<Vec<_>>();

    // Seed the loop state. The initial per-item predicate is the condition evaluated once on the entry state in the
    // enclosing block; a batched-predicate loop is pure, so this seeding never threads a token.
    let mut state_values = Vec::with_capacity(lowered_state_types.len());
    if iteration_bound.is_some() {
        state_values.push(lower_static_index_constants(&[0], block, context, location)?[0]);
    }
    state_values.extend_from_slice(input_values);
    if batched_predicate {
        let mut no_token = None;
        let initial_predicate = lower_nested_program_inline(
            while_op.condition(),
            input_values,
            block,
            context,
            location,
            false,
            nested_functions,
            collective_state,
            &mut no_token,
        )?;
        if initial_predicate.len() != 1 {
            return Err(LoweringError::UnsupportedOp {
                op: format!("while condition lowered to {} outputs", initial_predicate.len()),
            });
        }
        state_values.push(initial_predicate[0]);
    }
    if threads_token {
        state_values.push(current_or_new_token(token, block, location)?);
    }

    let mut condition_region = context.region();
    let condition_block = context.block(block_arguments.as_slice());
    {
        let mut condition_block_ref = condition_block.as_ref();
        // The scalar continuation decision. A batched predicate is carried through the loop state, so the condition
        // region only `or`-reduces it — the loop keeps running while any per-item predicate is still true. A scalar
        // predicate is evaluated inline on the state arguments.
        let loop_predicate = if batched_predicate {
            let carried_predicate = condition_block_ref
                .argument(predicate_index)
                .expect("batched-predicate while state should include the carried predicate")
                .as_ref();
            lower_reduce_to_mlir(
                ReductionKind::Any,
                predicate_dimensions.as_slice(),
                carried_predicate,
                &ArrayType::scalar(DataType::Boolean),
                &mut condition_block_ref,
                context,
                location,
            )?
        } else {
            let condition_inputs = (counter_offset..counter_offset + state_count)
                .map(|index| {
                    condition_block_ref.argument(index).expect("while condition should have state arguments").as_ref()
                })
                .collect::<Vec<_>>();
            let mut condition_token = if threads_token {
                Some(
                    condition_block_ref
                        .argument(token_index)
                        .expect("token-threaded while state should include the token")
                        .as_ref(),
                )
            } else {
                None
            };
            let condition_outputs = lower_nested_program_inline(
                while_op.condition(),
                condition_inputs.as_slice(),
                &mut condition_block_ref,
                context,
                location,
                false,
                nested_functions,
                collective_state,
                &mut condition_token,
            )?;
            if condition_outputs.len() != 1 {
                return Err(LoweringError::UnsupportedOp {
                    op: format!("while condition lowered to {} outputs", condition_outputs.len()),
                });
            }
            condition_outputs[0]
        };
        let predicate = match iteration_bound {
            Some(bound) => {
                let counter =
                    condition_block_ref.argument(0).expect("bounded while state should include the counter").as_ref();
                let bound_constant =
                    lower_static_index_constants(&[bound], &mut condition_block_ref, context, location)?[0];
                let counter_predicate = lower_compare_to_mlir(
                    ComparisonDirection::LessThan,
                    counter,
                    bound_constant,
                    &mut condition_block_ref,
                    location,
                )?;
                let fused = condition_block_ref.append_operation(stable_hlo::and(
                    loop_predicate,
                    counter_predicate,
                    location,
                )?)?;
                fused.result(0).expect("stablehlo.and should return one result").as_ref()
            }
            None => loop_predicate,
        };
        condition_block_ref.append_operation(stable_hlo::r#return(&[predicate], location)?)?;
    }
    condition_region.append_block(condition_block)?;

    let mut body_region = context.region();
    let body_block = context.block(block_arguments.as_slice());
    {
        let mut body_block_ref = body_block.as_ref();
        let body_inputs = (counter_offset..counter_offset + state_count)
            .map(|index| body_block_ref.argument(index).expect("while body should have state arguments").as_ref())
            .collect::<Vec<_>>();
        let mut body_token = if threads_token {
            Some(
                body_block_ref
                    .argument(token_index)
                    .expect("token-threaded while state should include the token")
                    .as_ref(),
            )
        } else {
            None
        };
        let body_outputs = lower_nested_program_inline(
            while_op.body(),
            body_inputs.as_slice(),
            &mut body_block_ref,
            context,
            location,
            false,
            nested_functions,
            collective_state,
            &mut body_token,
        )?;
        if body_outputs.len() != state_count {
            return Err(LoweringError::UnsupportedOp {
                op: format!("while body lowered to {} outputs", body_outputs.len()),
            });
        }
        // For a batched predicate, mask each carry update under the carried (incoming-state) predicate so finished
        // items freeze, then recompute the predicate on the updated state and thread it as the next carried
        // predicate. A frozen item's predicate is recomputed from its frozen state, so it can never rejoin the loop.
        let (next_state_values, next_predicate) = if batched_predicate {
            let carried_predicate = body_block_ref
                .argument(predicate_index)
                .expect("batched-predicate while state should include the carried predicate")
                .as_ref();
            let masked = body_outputs
                .into_iter()
                .zip(body_inputs.iter())
                .zip(state_types.iter())
                .map(|((candidate, carried), state_type)| {
                    // The predicate broadcasts to each state element's shape along its leading (prefix) axes; a
                    // state element already shaped like the predicate reuses it directly.
                    let element_mask = if state_type.shape() == predicate_type.shape() {
                        carried_predicate
                    } else {
                        let mask_type = lower_tensor_type(
                            &ArrayType::new(DataType::Boolean, state_type.shape().clone()),
                            context,
                            location,
                        )?;
                        let broadcast = body_block_ref.append_operation(stable_hlo::broadcast(
                            carried_predicate,
                            mask_type,
                            predicate_dimensions.as_slice(),
                            location,
                        )?)?;
                        broadcast.result(0).expect("stablehlo.broadcast_in_dim should return one result").as_ref()
                    };
                    let select = body_block_ref.append_operation(stable_hlo::select(
                        element_mask,
                        candidate,
                        *carried,
                        location,
                    )?)?;
                    Ok(select.result(0).expect("stablehlo.select should return one result").as_ref())
                })
                .collect::<Result<Vec<_>, LoweringError>>()?;
            let mut no_token = None;
            let next_predicate = lower_nested_program_inline(
                while_op.condition(),
                masked.as_slice(),
                &mut body_block_ref,
                context,
                location,
                false,
                nested_functions,
                collective_state,
                &mut no_token,
            )?;
            if next_predicate.len() != 1 {
                return Err(LoweringError::UnsupportedOp {
                    op: format!("while condition lowered to {} outputs", next_predicate.len()),
                });
            }
            (masked, Some(next_predicate[0]))
        } else {
            (body_outputs, None)
        };
        let mut next_state = Vec::with_capacity(lowered_state_types.len());
        if iteration_bound.is_some() {
            let counter = body_block_ref.argument(0).expect("bounded while state should include the counter").as_ref();
            let one = lower_static_index_constants(&[1], &mut body_block_ref, context, location)?[0];
            let next_counter = body_block_ref.append_operation(stable_hlo::add(counter, one, location)?)?;
            next_state.push(next_counter.result(0).expect("stablehlo.add should return one result").as_ref());
        }
        next_state.extend(next_state_values);
        if let Some(next_predicate) = next_predicate {
            next_state.push(next_predicate);
        }
        if threads_token {
            next_state.push(body_token.expect("token-threaded while bodies receive an entry token"));
        }
        body_block_ref.append_operation(stable_hlo::r#return(next_state.as_slice(), location)?)?;
    }
    body_region.append_block(body_block)?;

    let operation = block.append_operation(stable_hlo::r#while(
        state_values.as_slice(),
        condition_region.into(),
        body_region.into(),
        location,
    )?)?;
    if threads_token {
        *token = Some(
            operation
                .result(token_index)
                .expect("a token-threaded stablehlo.while should return one trailing token result")
                .as_ref(),
        );
    }
    Ok((0..state_types.len())
        .map(|index| {
            operation
                .result(counter_offset + index)
                .expect("stablehlo.while should return one result per state leaf")
                .as_ref()
        })
        .collect())
}

/// Lowers one statically counted scan loop to a `stablehlo.while` over the state
/// `[counter, carries..., stacks..., ys...]`.
///
/// The `i64` counter starts at zero and the loop runs while `counter < length`. Each loop trip runs `unroll`
/// consecutive logical iterations (body copies) and advances the counter by `unroll`, so the loop performs
/// `length / unroll` trips (the unroll factor must be at least `1` and evenly divide `length`, which
/// [`ScanOperation::with_unroll`] guarantees by construction). Logical iteration `i` computes its iteration index (`i`,
/// or `length - 1 - i` when `reverse` is set), reads one slice of every stacked input with
/// `stablehlo.dynamic_slice` (dropping the unit iteration axis with `stablehlo.reshape`), inlines the lowered body
/// program over `[carries..., iteration_slices...]`, and writes each per-iteration output into its preallocated stacked
/// zero accumulator with `stablehlo.dynamic_update_slice`. This is the same strategy JAX uses to lower `lax.scan`,
/// which is not an XLA primitive. When `unroll == length` no `stablehlo.while` is emitted at all: the body copies
/// inline as straight-line operations at static iteration indices. The provided `input_values` must align with the body
/// program's input signature: the first `carry_count` values are the carries and every remaining body input
/// receives one stacked operand.
fn lower_scan_to_while<'b, 'c: 'b, 't: 'c, V, O>(
    body_program: &Program<V, O, Vec<V>, Vec<V>>,
    carry_count: usize,
    length: usize,
    reverse: bool,
    unroll: usize,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
    token: &mut Option<ValueRef<'b, 'c, 't>>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: MlirLowerableValue + BooleanLike,
    O: LowerableXlaOperation<V>,
{
    // When the scan body is effectful, the enclosing scope's effect token is carried through the loop as one extra
    // trailing state element that each body copy threads through its effectful instructions (mirroring the while
    // lowering); the fully unrolled form needs no extra state because its body copies inline directly into the
    // enclosing scope and thread the scope token itself. Pure scans emit no token machinery at all.
    let threads_token = !body_program.effects().is_pure();
    let body_input_types = body_program.input_types();
    let body_output_types = body_program.output_types();
    if input_values.len() != body_input_types.len() {
        return Err(LoweringError::UnsupportedOp {
            op: format!("scan expected {} lowered inputs but got {}", body_input_types.len(), input_values.len()),
        });
    }
    if unroll == 0 || length % unroll != 0 {
        return Err(LoweringError::UnsupportedOp {
            op: format!("scan unroll factor {unroll} must be at least 1 and evenly divide the scan length {length}"),
        });
    }
    let carry_types = &body_input_types[..carry_count];
    let x_slice_types = &body_input_types[carry_count..];
    let y_slice_types = &body_output_types[carry_count..];
    let stacked = |slice_type: &ArrayType| -> Result<ArrayType, LoweringError> {
        let mut dimensions = vec![length];
        dimensions.extend(static_dimensions(slice_type)?);
        Ok(ArrayType::new(
            slice_type.data_type(),
            ryft_core::types::Shape::new(dimensions.into_iter().map(Size::Static).collect()),
        ))
    };

    // A fully unrolled scan (`unroll == length`) needs no loop at all: the body copies inline as straight-line
    // operations at static iteration indices, reading and writing the same stacked inputs and zero accumulators the
    // loop form would thread through its state.
    if unroll == length && length > 0 {
        let mut carries = input_values[..carry_count].to_vec();
        let x_stacks = input_values[carry_count..].to_vec();
        let mut y_accumulators = Vec::with_capacity(y_slice_types.len());
        for y_slice_type in y_slice_types {
            let stacked_type = stacked(y_slice_type)?;
            let accumulators = lower_constant_output(std::slice::from_ref(&stacked_type), 0, block, context, location)?;
            y_accumulators.push(accumulators[0]);
        }
        let zero_index = lower_static_index_constants(&[0], block, context, location)?[0];
        let mut iterations: Vec<usize> = (0..length).collect();
        if reverse {
            iterations.reverse();
        }
        for iteration in iterations {
            let index_value = lower_static_index_constants(&[iteration], block, context, location)?[0];
            (carries, y_accumulators) = lower_scan_iteration(
                body_program,
                x_slice_types,
                y_slice_types,
                index_value,
                zero_index,
                carries,
                x_stacks.as_slice(),
                y_accumulators,
                block,
                context,
                location,
                nested_functions,
                collective_state,
                token,
            )?;
        }
        carries.extend(y_accumulators);
        return Ok(carries);
    }

    // Assemble the loop state `[counter, carries..., stacks..., ys...]`, preallocating one zero accumulator per
    // stacked output.
    let mut state_types = Vec::with_capacity(1 + body_input_types.len() + y_slice_types.len());
    state_types.push(ArrayType::scalar(DataType::I64));
    state_types.extend(carry_types.iter().cloned());
    for x_slice_type in x_slice_types {
        state_types.push(stacked(x_slice_type)?);
    }
    let mut state_values = Vec::with_capacity(state_types.len() + y_slice_types.len());
    state_values.push(lower_static_index_constants(&[0], block, context, location)?[0]);
    state_values.extend_from_slice(input_values);
    for y_slice_type in y_slice_types {
        let stacked_type = stacked(y_slice_type)?;
        let accumulators = lower_constant_output(std::slice::from_ref(&stacked_type), 0, block, context, location)?;
        state_values.push(accumulators[0]);
        state_types.push(stacked_type);
    }
    // The effect token rides at the very end of the loop state, so all counter/carry/stack/accumulator index math
    // stays untouched.
    let token_index = state_types.len();
    let mut lowered_state_types = state_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, context, location).map(|tensor_type| tensor_type.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    if threads_token {
        lowered_state_types.push(context.stable_hlo_token_type()?.as_ref());
        state_values.push(current_or_new_token(token, block, location)?);
    }
    let block_arguments = lowered_state_types.iter().map(|r#type| (*r#type, location)).collect::<Vec<_>>();

    let mut condition_region = context.region();
    let condition_block = context.block(block_arguments.as_slice());
    {
        let mut condition_block_ref = condition_block.as_ref();
        let counter = condition_block_ref.argument(0).expect("scan while state should include the counter").as_ref();
        let length_constant = lower_static_index_constants(&[length], &mut condition_block_ref, context, location)?[0];
        let predicate = lower_compare_to_mlir(
            ComparisonDirection::LessThan,
            counter,
            length_constant,
            &mut condition_block_ref,
            location,
        )?;
        condition_block_ref.append_operation(stable_hlo::r#return(&[predicate], location)?)?;
    }
    condition_region.append_block(condition_block)?;

    let mut body_region = context.region();
    let body_block = context.block(block_arguments.as_slice());
    {
        let mut body_block_ref = body_block.as_ref();
        let arguments = (0..state_types.len())
            .map(|index| body_block_ref.argument(index).expect("scan while body should have state arguments").as_ref())
            .collect::<Vec<_>>();
        let counter = arguments[0];
        let zero_index = lower_static_index_constants(&[0], &mut body_block_ref, context, location)?[0];
        // When the visit order is reversed, logical iteration `i` reads iteration `length - 1 - i` (a zero-length
        // reversed scan never runs its body, so the saturated limit constant is inert).
        let reverse_limit = if reverse {
            Some(lower_static_index_constants(&[length.saturating_sub(1)], &mut body_block_ref, context, location)?[0])
        } else {
            None
        };

        // Each loop trip runs `unroll` consecutive logical iterations (`counter + copy` for each body copy), so the
        // counter advances by `unroll` per trip and the unchanged `counter < length` condition yields
        // `length / unroll` trips.
        let mut carries = arguments[1..1 + carry_count].to_vec();
        let x_stacks = arguments[1 + carry_count..1 + carry_count + x_slice_types.len()].to_vec();
        let mut y_accumulators = arguments[1 + carry_count + x_slice_types.len()..].to_vec();
        let mut body_token = if threads_token {
            Some(
                body_block_ref
                    .argument(token_index)
                    .expect("token-threaded scan state should include the token")
                    .as_ref(),
            )
        } else {
            None
        };
        for copy in 0..unroll {
            let iteration = if copy == 0 {
                counter
            } else {
                let offset = lower_static_index_constants(&[copy], &mut body_block_ref, context, location)?[0];
                let addition = body_block_ref.append_operation(stable_hlo::add(counter, offset, location)?)?;
                addition.result(0).expect("stablehlo.add should return one result").as_ref()
            };
            let index_value = match reverse_limit {
                Some(limit) => {
                    let subtraction =
                        body_block_ref.append_operation(stable_hlo::subtract(limit, iteration, location)?)?;
                    subtraction.result(0).expect("stablehlo.subtract should return one result").as_ref()
                }
                None => iteration,
            };
            (carries, y_accumulators) = lower_scan_iteration(
                body_program,
                x_slice_types,
                y_slice_types,
                index_value,
                zero_index,
                carries,
                x_stacks.as_slice(),
                y_accumulators,
                &mut body_block_ref,
                context,
                location,
                nested_functions,
                collective_state,
                &mut body_token,
            )?;
        }

        // Assemble the next state: advance the counter by the unroll factor, thread the new carries, pass the input
        // stacks through unchanged, and thread the updated stacked accumulators (and the effect token, if any).
        let step = lower_static_index_constants(&[unroll], &mut body_block_ref, context, location)?[0];
        let next_counter = body_block_ref.append_operation(stable_hlo::add(counter, step, location)?)?;
        let mut next_state = vec![next_counter.result(0).expect("stablehlo.add should return one result").as_ref()];
        next_state.extend(carries);
        next_state.extend(x_stacks);
        next_state.extend(y_accumulators);
        if threads_token {
            next_state.push(body_token.expect("token-threaded scan bodies receive an entry token"));
        }
        body_block_ref.append_operation(stable_hlo::r#return(next_state.as_slice(), location)?)?;
    }
    body_region.append_block(body_block)?;

    let operation = block.append_operation(stable_hlo::r#while(
        state_values.as_slice(),
        condition_region.into(),
        body_region.into(),
        location,
    )?)?;
    if threads_token {
        *token = Some(
            operation
                .result(token_index)
                .expect("a token-threaded stablehlo.while should return one trailing token result")
                .as_ref(),
        );
    }
    let result = |index: usize| {
        operation.result(index).expect("stablehlo.while should return one result per state leaf").as_ref()
    };
    let mut outputs = Vec::with_capacity(carry_count + y_slice_types.len());
    outputs.extend((0..carry_count).map(|index| result(1 + index)));
    outputs.extend((0..y_slice_types.len()).map(|index| result(1 + carry_count + x_slice_types.len() + index)));
    Ok(outputs)
}

/// Emits one scan iteration at iteration index `index_value` into `block`: reads slice `index_value` of every stacked
/// input (dropping the unit iteration axis), inlines the body program over `[carries..., x_slices...]`, writes each
/// per-iteration output into its stacked accumulator at `index_value`, and returns the new carries and accumulators.
/// This is the per-iteration building block shared by the looped and fully unrolled scan lowerings in
/// [`lower_scan_to_while`].
fn lower_scan_iteration<'b, 'c: 'b, 't: 'c, V, O>(
    body_program: &Program<V, O, Vec<V>, Vec<V>>,
    x_slice_types: &[ArrayType],
    y_slice_types: &[ArrayType],
    index_value: ValueRef<'b, 'c, 't>,
    zero_index: ValueRef<'b, 'c, 't>,
    carries: Vec<ValueRef<'b, 'c, 't>>,
    x_stacks: &[ValueRef<'b, 'c, 't>],
    y_accumulators: Vec<ValueRef<'b, 'c, 't>>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
    token: &mut Option<ValueRef<'b, 'c, 't>>,
) -> Result<(Vec<ValueRef<'b, 'c, 't>>, Vec<ValueRef<'b, 'c, 't>>), LoweringError>
where
    V: MlirLowerableValue + BooleanLike,
    O: LowerableXlaOperation<V>,
{
    // Read one slice of every stacked input and drop the unit iteration axis.
    let carry_count = carries.len();
    let mut iteration_inputs = carries;
    for (stack_offset, x_slice_type) in x_slice_types.iter().enumerate() {
        let slice_dimensions = static_dimensions(x_slice_type)?;
        let mut sizes = vec![1];
        sizes.extend(slice_dimensions.iter().copied());
        let mut start_values = vec![index_value];
        start_values.extend(std::iter::repeat_n(zero_index, slice_dimensions.len()));
        let iteration = block.append_operation(stable_hlo::dynamic_slice(
            x_stacks[stack_offset],
            start_values.as_slice(),
            sizes.as_slice(),
            location,
        )?)?;
        let squeezed = block.append_operation(stable_hlo::reshape(
            iteration.result(0).expect("stablehlo.dynamic_slice should return one result").as_ref(),
            slice_dimensions.as_slice(),
            location,
        )?)?;
        iteration_inputs.push(squeezed.result(0).expect("stablehlo.reshape should return one result").as_ref());
    }

    let body_outputs = lower_nested_program_inline(
        body_program,
        iteration_inputs.as_slice(),
        block,
        context,
        location,
        false,
        nested_functions,
        collective_state,
        token,
    )?;
    if body_outputs.len() != carry_count + y_slice_types.len() {
        return Err(LoweringError::UnsupportedOp {
            op: format!("scan body lowered to {} outputs", body_outputs.len()),
        });
    }

    // Thread the new carries and write each per-iteration output into its stacked accumulator.
    let new_carries = body_outputs[..carry_count].to_vec();
    let mut new_accumulators = Vec::with_capacity(y_slice_types.len());
    for (y_offset, y_slice_type) in y_slice_types.iter().enumerate() {
        let slice_dimensions = static_dimensions(y_slice_type)?;
        let mut expanded_dimensions = vec![1];
        expanded_dimensions.extend(slice_dimensions.iter().copied());
        let expanded = block.append_operation(stable_hlo::reshape(
            body_outputs[carry_count + y_offset],
            expanded_dimensions.as_slice(),
            location,
        )?)?;
        let mut start_values = vec![index_value];
        start_values.extend(std::iter::repeat_n(zero_index, slice_dimensions.len()));
        let updated = block.append_operation(stable_hlo::dynamic_update_slice(
            y_accumulators[y_offset],
            expanded.result(0).expect("stablehlo.reshape should return one result").as_ref(),
            start_values.as_slice(),
            location,
        )?)?;
        new_accumulators
            .push(updated.result(0).expect("stablehlo.dynamic_update_slice should return one result").as_ref());
    }
    Ok((new_carries, new_accumulators))
}

/// Identity of a flat callee program staged behind a `jit_call`, used to deduplicate repeated nested programs into
/// shared private `func.func`s at lowering time.
///
/// Eligible programs (see [`supports_structural_dedup`]) are keyed structurally by their canonical rendering plus
/// flat input types: type inference is deterministic, so two eligible programs that render identically with equal
/// input types compute the same function and may share one emitted function — even when they are distinct staged
/// programs produced by separate transform passes (for example the per-block primal and pullback programs of
/// `grad(jit(f))` over repeated blocks). Programs that cannot be rendered faithfully fall back to pointer identity,
/// so only literally-shared programs merge and structurally-distinct ones never do.
#[derive(Clone, PartialEq, Eq, Hash)]
enum JitCallProgramKey {
    /// Structural identity for dedup-eligible programs: canonical rendering plus flat input types.
    Structural {
        /// Canonical [`Program`] rendering (operation names plus all bracketed attributes).
        rendered: String,

        /// Flat input [`ArrayType`]s, which together with the rendering pin the full callee signature.
        input_types: Vec<ArrayType>,
    },

    /// Pointer identity for programs that cannot be rendered faithfully (custom-derivative bodies carrying closures,
    /// or programs whose instructions hide nested bodies). Two occurrences merge only when they share one program.
    Pointer(usize),
}

/// Returns whether `program` may be deduplicated by structural identity.
///
/// A program is eligible only when every instruction's operation renders faithfully: its canonical rendering
/// captures the full operation semantics. Operations that carry hidden bodies (`jit_call`, control flow, `shard_map`)
/// or user closures (custom-derivative operations) render without those payloads, so two structurally-distinct such
/// programs can render identically; those programs fall back to [`JitCallProgramKey::Pointer`] instead, which never
/// merges distinct programs.
fn supports_structural_dedup<Input, Output>(program: &XlaProgram<Input, Output>) -> bool
where
    Input: Parameterized<XlaConstant>,
    Output: Parameterized<XlaConstant>,
{
    program.instructions().iter().all(|instruction| {
        !matches!(
            instruction.operation(),
            XlaOperation::Condition(_)
                | XlaOperation::While(_)
                | XlaOperation::Scan(_)
                | XlaOperation::CustomJvp(_)
                | XlaOperation::CustomVjp(_)
                | XlaOperation::CustomVjpTangent(_)
                | XlaOperation::Rematerialize(_)
                | XlaOperation::JitCall(_)
                | XlaOperation::ShardMap(_)
        )
    })
}

/// Computes the deduplication key for a flat callee program.
fn jit_call_program_key(program: &Rc<FlatXlaProgram>) -> JitCallProgramKey {
    if supports_structural_dedup(program.as_ref()) {
        JitCallProgramKey::Structural { rendered: program.to_string(), input_types: program.input_types() }
    } else {
        JitCallProgramKey::Pointer(Rc::as_ptr(program) as *const () as usize)
    }
}

/// One deduplicated callee emitted as a shared private `func.func`.
struct JitCallFunction {
    /// Symbol name of the emitted private function.
    symbol: String,

    /// Representative callee program for this key, lowered once as the function body.
    program: Rc<FlatXlaProgram>,

    /// Flat input [`ArrayType`]s of the callee, also the emitted function's argument types.
    input_types: Vec<ArrayType>,

    /// Flat output [`ArrayType`]s of the callee, also the emitted function's result types.
    output_types: Vec<ArrayType>,
}

/// Shared private functions emitted for `jit_call` callees that occur more than once in a module.
///
/// Built once by [`collect_jit_call_functions`] before a module is lowered and threaded read-only through the
/// lowering pass. At each `jit_call` lowering site, a callee whose key is present is emitted as a `func.call` to the
/// shared function instead of being inlined; absent callees inline as before.
#[derive(Default)]
pub(crate) struct JitCallFunctionMap {
    /// Shared functions keyed by callee identity.
    functions: HashMap<JitCallProgramKey, JitCallFunction>,

    /// Keys in first-occurrence order, so emitted symbol names and module layout are deterministic.
    order: Vec<JitCallProgramKey>,
}

impl JitCallFunctionMap {
    /// Returns the shared function for `program`, if one was emitted for its identity.
    fn get(&self, program: &Rc<FlatXlaProgram>) -> Option<&JitCallFunction> {
        self.functions.get(&jit_call_program_key(program))
    }
}

/// Counts `jit_call` callee occurrences in `instructions`, recursing into callee bodies and control-flow bodies.
///
/// `counts` accumulates the occurrence count and a representative program per identity, `order` records keys in
/// first-occurrence order, and `memo` caches the (possibly expensive) key computation per shared program pointer.
/// Shard-map bodies are intentionally not traversed: their `jit_call`s lower with shard-local types and always
/// inline.
fn count_jit_calls<Input, Output>(
    program: &XlaProgram<Input, Output>,
    counts: &mut HashMap<JitCallProgramKey, (usize, Rc<FlatXlaProgram>)>,
    order: &mut Vec<JitCallProgramKey>,
    memo: &mut HashMap<usize, JitCallProgramKey>,
) where
    Input: Parameterized<XlaConstant>,
    Output: Parameterized<XlaConstant>,
{
    for instruction in program.instructions() {
        match instruction.operation() {
            XlaOperation::Condition(condition) => {
                count_jit_calls(condition.true_branch(), counts, order, memo);
                count_jit_calls(condition.false_branch(), counts, order, memo);
            }
            XlaOperation::While(while_op) => {
                count_jit_calls(while_op.condition(), counts, order, memo);
                count_jit_calls(while_op.body(), counts, order, memo);
            }
            XlaOperation::Scan(scan) => count_jit_calls(scan.body(), counts, order, memo),
            XlaOperation::CustomJvp(custom) => count_jit_calls(custom.primal(), counts, order, memo),
            XlaOperation::CustomVjp(custom) => count_jit_calls(custom.primal(), counts, order, memo),
            XlaOperation::CustomVjpTangent(carrier) => count_jit_calls(carrier.backward(), counts, order, memo),
            XlaOperation::Rematerialize(operation) => count_jit_calls(operation.primal(), counts, order, memo),
            XlaOperation::JitCall(call) => {
                let program = call.program_rc();
                let pointer = Rc::as_ptr(program) as *const () as usize;
                let key = memo.entry(pointer).or_insert_with(|| jit_call_program_key(program)).clone();
                let entry = counts.entry(key.clone()).or_insert_with(|| {
                    order.push(key.clone());
                    (0, program.clone())
                });
                entry.0 += 1;
                count_jit_calls(call.program(), counts, order, memo);
            }
            _ => {}
        }
    }
}

/// Builds the [`JitCallFunctionMap`] for a module by emitting a shared private function for every `jit_call` callee
/// that occurs at least twice (per [`JitCallProgramKey`] identity). Single-occurrence callees are left to inline, so
/// modules without repeated calls lower exactly as before.
fn collect_jit_call_functions<Input, Output>(program: &XlaProgram<Input, Output>) -> JitCallFunctionMap
where
    Input: Parameterized<XlaConstant>,
    Output: Parameterized<XlaConstant>,
{
    let mut counts: HashMap<JitCallProgramKey, (usize, Rc<FlatXlaProgram>)> = HashMap::new();
    let mut order: Vec<JitCallProgramKey> = Vec::new();
    let mut memo: HashMap<usize, JitCallProgramKey> = HashMap::new();
    count_jit_calls(program, &mut counts, &mut order, &mut memo);

    let mut map = JitCallFunctionMap::default();
    for key in order {
        let (count, program) = counts.remove(&key).expect("every ordered key was counted");
        // Effectful callees always inline (even when repeated) so their effectful instructions chain onto the
        // caller's effect token in program order; a shared token-free function could not preserve that ordering.
        if count < 2 || !program.effects().is_pure() {
            continue;
        }
        let symbol = format!("jit_call_{}", map.order.len());
        let input_types = program.input_types();
        let output_types = program.output_types();
        map.functions.insert(key.clone(), JitCallFunction { symbol, program, input_types, output_types });
        map.order.push(key);
    }
    map
}

/// Emits the shared private `func.func` for one deduplicated callee into `module_block`.
///
/// The body is lowered with `nested_functions` in scope so that any repeated `jit_call`s inside this callee also
/// lower to `func.call`s (calls between shared functions are resolved by symbol, so emission order does not matter).
fn emit_jit_call_function<'b, 'c: 'b, 't: 'c>(
    module_block: &mut BlockRef<'b, 'c, 't>,
    function: &JitCallFunction,
    nested_functions: &Rc<JitCallFunctionMap>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), LoweringError> {
    let argument_tensor_types = function
        .input_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let result_tensor_types = function
        .output_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, context, location))
        .collect::<Result<Vec<_>, _>>()?;

    let function_block = context.block(
        argument_tensor_types
            .iter()
            .map(|tensor_type| (*tensor_type, location))
            .collect::<Vec<_>>()
            .as_slice(),
    );
    {
        let mut function_block_ref = function_block.as_ref();
        let input_values = (0..function.input_types.len())
            .map(|index| function_block.argument(index).expect("shared function block arguments should exist").as_ref())
            .collect::<Vec<_>>();
        // Deduplicated callees are pure by construction ([`collect_jit_call_functions`] skips effectful programs),
        // so the shared function body never needs an effect token.
        let mut token = None;
        let outputs = lower_nested_program_inline(
            function.program.as_ref(),
            input_values.as_slice(),
            &mut function_block_ref,
            context,
            location,
            false,
            Some(nested_functions),
            &CollectiveLoweringState::new(),
            &mut token,
        )?;
        function_block_ref.append_operation(func::r#return(outputs.as_slice(), location)?)?;
    }
    let mut function_region = context.region();
    function_region.append_block(function_block)?;
    module_block.append_operation(func::func(
        function.symbol.as_str(),
        func::FuncAttributes {
            arguments: argument_tensor_types
                .iter()
                .map(|tensor_type| TypeAndAttributes { r#type: tensor_type.as_ref(), attributes: None })
                .collect(),
            results: result_tensor_types
                .iter()
                .map(|tensor_type| TypeAndAttributes { r#type: tensor_type.as_ref(), attributes: None })
                .collect(),
            visibility: SymbolVisibility::Private,
            ..Default::default()
        },
        function_region,
        location,
    )?)?;
    Ok(())
}

/// Lowers one `jit_call` to either a `func.call` of a shared private function (when its callee was deduplicated) or
/// an inlined copy of the callee body (otherwise).
///
/// `input_values` are the lowered call operands in callee-input order; for a `linear_jit_call` they are the lowered
/// captured prefix followed by the lowered linear inputs.
fn lower_jit_call<'b, 'c: 'b, 't: 'c>(
    program: &Rc<FlatXlaProgram>,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
    token: &mut Option<ValueRef<'b, 'c, 't>>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    // Only pure callees are ever deduplicated ([`collect_jit_call_functions`] skips effectful programs), so the
    // shared-function path below never interacts with the caller's effect token; effectful callees always take the
    // inline path, which threads the caller's token through the callee body in program order.
    if let Some(map) = nested_functions {
        if let Some(function) = map.get(program) {
            // The `jit_call` operation's type inference already pins its operands to the callee input types, so a
            // matching arity is the only guard needed before emitting the symbol call; anything else inlines.
            if input_values.len() == function.input_types.len() {
                let result_tensor_types = function
                    .output_types
                    .iter()
                    .map(|array_type| lower_tensor_type(array_type, context, location))
                    .collect::<Result<Vec<_>, _>>()?;
                let operation = block.append_operation(func::call(
                    function.symbol.as_str(),
                    func::CallProperties {
                        arguments: input_values
                            .iter()
                            .map(|value| ValueAndAttributes { value: *value, attributes: None })
                            .collect(),
                        results: result_tensor_types
                            .iter()
                            .map(|tensor_type| TypeAndAttributes { r#type: tensor_type.as_ref(), attributes: None })
                            .collect(),
                        no_inline: false,
                    },
                    location,
                )?)?;
                return Ok((0..function.output_types.len())
                    .map(|index| {
                        operation.result(index).expect("func.call should return one result per output").as_ref()
                    })
                    .collect());
            }
        }
    }
    lower_nested_program_inline(
        program.as_ref(),
        input_values,
        block,
        context,
        location,
        false,
        nested_functions,
        collective_state,
        token,
    )
}

/// Inlines a nested sub-program into the given block by mapping the provided input
/// MLIR values to the body's input atoms, lowering constants and instructions in topological
/// order, and returning lowered values corresponding to the program's output atoms.
///
/// `token` is the effect token of the lowering scope the program inlines into: it flows into each instruction's
/// lowerer and the updated token is read back out after the instruction lowers, so effectful instructions chain in
/// program order and the caller observes the program's final token.
#[allow(clippy::too_many_arguments)]
fn lower_nested_program_inline<'b, 'c: 'b, 't: 'c, O, V>(
    program: &Program<V, O, Vec<V>, Vec<V>>,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    add_optimization_barrier: bool,
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
    token: &mut Option<ValueRef<'b, 'c, 't>>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: MlirLowerableValue + BooleanLike,
    O: LowerableXlaOperation<V>,
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
            let mut lowerer = PlainMlirLowerer::new(*block, context, location)
                .with_nested_functions(nested_functions.cloned())
                .with_token(*token)
                .with_collective_state(collective_state.clone());
            let outputs = instruction.operation().lower_to_mlir(
                inputs,
                output_types.as_slice(),
                PlainMlirLoweringMode::Unpacked,
                &mut lowerer,
            )?;
            *token = lowerer.token;
            Ok(outputs)
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
fn replay_program_into_block<'b, 'c: 'b, 't: 'c, O, V: Value<Type = ArrayType>, Input, Output, LiftConstant, ApplyOp>(
    program: &Program<V, O, Input, Output>,
    input_values: Vec<ValueRef<'b, 'c, 't>>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    mut lift_constant: LiftConstant,
    mut apply_op: ApplyOp,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    O: Operation<ArrayType>,
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
fn lower_plain_program_outputs<'b, 'c: 'b, 't: 'c, O, V, Input, Output>(
    program: &Program<V, O, Input, Output>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: MlirLowerableValue + BooleanLike,
    O: LowerableXlaOperation<V>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    let input_values = (0..program.input_ids().len())
        .map(|index| block.argument(index).expect("body block arguments should exist").as_ref())
        .collect::<Vec<_>>();
    // Function-body-scoped effect token chain, created lazily by the first effectful instruction and dropped at the
    // end of the function body.
    let mut token = None;
    // Module-scoped collective lowering state: this entry point lowers one whole module.
    let collective_state = CollectiveLoweringState::new();
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
            let mut lowerer = PlainMlirLowerer::new(*block, context, location)
                .with_token(token)
                .with_collective_state(collective_state.clone());
            let outputs = instruction.operation().lower_to_mlir(
                inputs,
                output_types.as_slice(),
                PlainMlirLoweringMode::Unpacked,
                &mut lowerer,
            )?;
            token = lowerer.token;
            Ok(outputs)
        },
    )
}

/// Lowers one traced program to values inside a block.
fn lower_program_outputs<'b, 'c: 'b, 't: 'c, ProgramInput, ProgramOutput>(
    program: &XlaProgram<ProgramInput, ProgramOutput>,
    captured_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    ProgramInput: Parameterized<XlaConstant>,
    ProgramOutput: Parameterized<XlaConstant>,
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
            let value =
                block.argument(captured_values.len() + index).expect("body block arguments should exist").as_ref();
            atom_values[atom_id.index()] = Some(value);
            value
        })
        .collect::<Vec<_>>();
    let atom_values = std::cell::RefCell::new(atom_values);
    // Function-body-scoped effect token chain, created lazily by the first effectful instruction and dropped at the
    // end of the function body: this v1 design orders effects within one dispatch, and carrying tokens across
    // separately dispatched executions is out of scope.
    let mut token = None;
    replay_program_into_block(
        program,
        input_values,
        block,
        context,
        location,
        |atom_id, value, block, context, location| {
            let lowered = lower_constant(atom_id, value, captured_values, block, context, location)?;
            atom_values.borrow_mut()[atom_id.index()] = Some(lowered);
            Ok(lowered)
        },
        |instruction, inputs, block, context, location| {
            let mut table = atom_values.borrow_mut();
            let lowered_outputs = lower_instruction(
                program,
                instruction,
                table.as_slice(),
                inputs,
                block,
                context,
                location,
                nested_functions,
                collective_state,
                &mut token,
            )?;
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
    program: &XlaProgram<ProgramInput, ProgramOutput>,
    local_input_types: &[ArrayType],
    global_output_types: &[ArrayType],
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    collective_state: &CollectiveLoweringState,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    ProgramInput: Parameterized<XlaConstant>,
    ProgramOutput: Parameterized<XlaConstant>,
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
        // Shard-map bodies lower with shard-local types, so their `jit_call`s always inline; do not thread the
        // module's deduplicated functions (which are typed against global shapes) into them.
        let body_collective_state = collective_state.enter_manual_region(shard_map.clone());
        let body_outputs = lower_program_outputs(
            program,
            &[],
            &mut body_block_ref,
            context,
            location.as_ref(),
            None,
            &body_collective_state,
        )?;
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

/// Lowers one concrete traced value to a StableHLO constant operation and returns its result value.
fn lower_literal_value<'b, 'c: 'b, 't: 'c, B, V, L>(
    value: &V,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    V: MlirLowerableValue + BooleanLike,
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

/// Lowers one captured constant reference by forwarding its runtime captured value.
fn lower_captured_constant<'b, 'c: 'b, 't: 'c>(
    value: &XlaConstant,
    captured_values: &[ValueRef<'b, 'c, 't>],
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    captured_values
        .get(value.index())
        .copied()
        .ok_or(LoweringError::MissingCapturedConstant { index: value.index() })
}

/// Lowers a traced constant atom to a StableHLO constant operation and returns its result value.
fn lower_constant<'b, 'c: 'b, 't: 'c, B, L>(
    _atom_id: AtomId,
    value: &XlaConstant,
    captured_values: &[ValueRef<'b, 'c, 't>],
    _block: &mut B,
    _context: &'c MlirContext<'t>,
    _location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    lower_captured_constant(value, captured_values)
}

/// Dispatches shard-map StableHLO lowering for one traced operation by matching on primitive variants.
fn dispatch_lower_shard_map_mlir<'b, 'c: 'b, 't: 'c>(
    op: &XlaOperation,
    captured_values: &[ValueRef<'b, 'c, 't>],
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayType],
    lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    match op {
        XlaOperation::Zero(_) => {
            if !input_values.is_empty() {
                return Err(ProgramError::InvalidInputCount { expected: 0, actual: input_values.len() }.into());
            }
            lower_constant_output(output_types, 0, &mut lowerer.block, lowerer.context, lowerer.location)
        }
        XlaOperation::One(_) => {
            if !input_values.is_empty() {
                return Err(ProgramError::InvalidInputCount { expected: 0, actual: input_values.len() }.into());
            }
            lower_constant_output(output_types, 1, &mut lowerer.block, lowerer.context, lowerer.location)
        }
        XlaOperation::Constant(constant) => {
            check_count!("input", input_values, 0, ProgramError);
            check_count!("output", output_types, 1, ProgramError);
            // A typed literal constant captures its value in the enclosing program's capture table; resolve it by
            // forwarding the corresponding captured runtime value.
            let constant_value = lower_captured_constant(constant.value(), captured_values)?;
            Ok(vec![constant_value])
        }
        XlaOperation::Add(_) => {
            let result =
                lowerer
                    .block
                    .append_operation(stable_hlo::add(input_values[0], input_values[1], lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.add should return one result").as_ref()])
        }
        XlaOperation::Sub(_) => {
            let result = lowerer.block.append_operation(stable_hlo::subtract(
                input_values[0],
                input_values[1],
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.subtract should return one result").as_ref()])
        }
        XlaOperation::Mul(_) => {
            let result = lowerer.block.append_operation(stable_hlo::multiply(
                input_values[0],
                input_values[1],
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.multiply should return one result").as_ref()])
        }
        XlaOperation::Div(_) => {
            let result = lowerer.block.append_operation(stable_hlo::divide(
                input_values[0],
                input_values[1],
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.divide should return one result").as_ref()])
        }
        XlaOperation::Neg(_) => {
            let result = lowerer.block.append_operation(stable_hlo::negate(input_values[0], lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.negate should return one result").as_ref()])
        }
        XlaOperation::Sin(_) => {
            let result = lowerer.block.append_operation(stable_hlo::sine(
                input_values[0],
                Accuracy::Default,
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.sine should return one result").as_ref()])
        }
        XlaOperation::Cos(_) => {
            let result = lowerer.block.append_operation(stable_hlo::cosine(
                input_values[0],
                Accuracy::Default,
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.cosine should return one result").as_ref()])
        }
        XlaOperation::Atan2(_) => {
            let result = lowerer.block.append_operation(stable_hlo::atan2(
                input_values[0],
                input_values[1],
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.atan2 should return one result").as_ref()])
        }
        XlaOperation::Exponential(_) => {
            let result = lowerer.block.append_operation(stable_hlo::exponential(
                input_values[0],
                Accuracy::Default,
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.exponential should return one result").as_ref()])
        }
        XlaOperation::Logarithm(_) => {
            let result = lowerer.block.append_operation(stable_hlo::log(
                input_values[0],
                Accuracy::Default,
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.log should return one result").as_ref()])
        }
        XlaOperation::SquareRoot(_) => {
            let result = lowerer.block.append_operation(stable_hlo::sqrt(
                input_values[0],
                Accuracy::Default,
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.sqrt should return one result").as_ref()])
        }
        XlaOperation::Abs(_) => {
            let result = lowerer.block.append_operation(stable_hlo::abs(input_values[0], lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.abs should return one result").as_ref()])
        }
        XlaOperation::Complex(_) => {
            let result = lowerer.block.append_operation(stable_hlo::complex(
                input_values[0],
                input_values[1],
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.complex should return one result").as_ref()])
        }
        // StableHLO has no conjugation operation, so `conjugate` lowers to the `complex(real(z), negate(imag(z)))`
        // composition (the same decomposition JAX's `conj` lowering uses).
        XlaOperation::Conjugate(_) => {
            let real = lowerer.block.append_operation(stable_hlo::real(input_values[0], lowerer.location)?)?;
            let imaginary = lowerer.block.append_operation(stable_hlo::imag(input_values[0], lowerer.location)?)?;
            let negated = lowerer.block.append_operation(stable_hlo::negate(
                imaginary.result(0).expect("stablehlo.imag should return one result").as_ref(),
                lowerer.location,
            )?)?;
            let result = lowerer.block.append_operation(stable_hlo::complex(
                real.result(0).expect("stablehlo.real should return one result").as_ref(),
                negated.result(0).expect("stablehlo.negate should return one result").as_ref(),
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.complex should return one result").as_ref()])
        }
        XlaOperation::Real(_) => {
            let result = lowerer.block.append_operation(stable_hlo::real(input_values[0], lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.real should return one result").as_ref()])
        }
        XlaOperation::Imaginary(_) => {
            let result = lowerer.block.append_operation(stable_hlo::imag(input_values[0], lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.imag should return one result").as_ref()])
        }
        // `stop_gradient` only affects differentiation; by lowering time it is the identity, so
        // forward the operand without emitting any MLIR operation (matching JAX's lowering).
        XlaOperation::StopGradient(_) => {
            if input_values.len() != 1 {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: input_values.len() }.into());
            }
            Ok(vec![input_values[0]])
        }
        // `tag` only affects rematerialization policies; by lowering time it is the identity, so
        // forward the operand without emitting any MLIR operation.
        XlaOperation::Tag(_) => {
            if input_values.len() != 1 {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: input_values.len() }.into());
            }
            Ok(vec![input_values[0]])
        }
        // `print` is the identity on its dataflow output; its observable effect lowers to a host-callback custom
        // call that consumes and produces a StableHLO token, so the effect ordering rides the scope's token chain
        // instead of the value dataflow.
        XlaOperation::Print(operation) => {
            check_count!("input", input_values, 1, ProgramError);
            lower_print_to_custom_call(
                operation.label(),
                input_values[0],
                &mut lowerer.token,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            Ok(vec![input_values[0]])
        }
        XlaOperation::TransferToMemory(operation) => lower_transfer_to_memory(
            operation.destination(),
            input_values,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        ),
        // Custom-derivative calls lower as their primal program; the derivative programs never reach the backend.
        XlaOperation::CustomJvp(operation) => lower_nested_program_inline(
            operation.primal(),
            input_values,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
            false,
            lowerer.nested_functions.as_ref(),
            &lowerer.collective_state,
            &mut lowerer.token,
        ),
        XlaOperation::CustomVjp(operation) => lower_nested_program_inline(
            operation.primal(),
            input_values,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
            false,
            lowerer.nested_functions.as_ref(),
            &lowerer.collective_state,
            &mut lowerer.token,
        ),
        // The opaque custom-VJP tangent carrier is a forward-mode tangent map that reverse mode transposes away before
        // lowering, so it never reaches the backend; reaching here means a forward-mode use of `custom_vjp` slipped
        // through, which is reverse-mode-only.
        XlaOperation::CustomVjpTangent(operation) => Err(ProgramError::UnsupportedOperation {
            message: format!("operation `{}` cannot be lowered to StableHLO", operation.name()),
        }
        .into()),
        XlaOperation::Rematerialize(operation) => lower_nested_program_inline(
            operation.primal(),
            input_values,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
            false,
            lowerer.nested_functions.as_ref(),
            &lowerer.collective_state,
            &mut lowerer.token,
        ),
        XlaOperation::ZeroLike(_) => {
            lower_like_constant(input_values, output_types, 0, &mut lowerer.block, lowerer.context, lowerer.location)
        }
        XlaOperation::OneLike(_) => {
            lower_like_constant(input_values, output_types, 1, &mut lowerer.block, lowerer.context, lowerer.location)
        }
        XlaOperation::Dot(operation) => {
            // The requested output sharding has already been folded into `output_types[0]` by type inference.
            let dimensions = operation.dimensions();
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
        XlaOperation::Transpose(operation) => {
            let result = lowerer.block.append_operation(stable_hlo::transpose(
                input_values[0],
                operation.permutation().as_slice(),
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.transpose should return one result").as_ref()])
        }
        XlaOperation::Fill(fill) => {
            check_count!("input", input_values, 0, ProgramError);
            check_count!("output", output_types, 1, ProgramError);
            let output_tensor_type = lowerer.lower_tensor_type(&output_types[0])?;
            let constant_value = lower_scalar_constant_splat(
                *fill.value(),
                &output_types[0],
                output_tensor_type,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            Ok(vec![constant_value])
        }
        XlaOperation::Iota(iota) => {
            check_count!("input", input_values, 0, ProgramError);
            check_count!("output", output_types, 1, ProgramError);
            let output_tensor_type = lowerer.lower_tensor_type(&output_types[0])?;
            let result = lowerer.block.append_operation(stable_hlo::iota(
                output_tensor_type,
                iota.iota_dimension(),
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.iota should return one result").as_ref()])
        }
        XlaOperation::Reshape(_) => {
            check_count!("output", output_types, 1, ProgramError);
            let output_type = &output_types[0];
            let output_shape = static_dimensions(output_type)?;
            let result = lowerer.block.append_operation(stable_hlo::reshape(
                input_values[0],
                output_shape.as_slice(),
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.reshape should return one result").as_ref()])
        }
        XlaOperation::Reshard(operation) => {
            lower_sharding_constraint(input_values, operation.sharding(), &mut lowerer.block, lowerer.location)
        }
        XlaOperation::ShardingConstraint(operation) => {
            lower_sharding_constraint(input_values, operation.sharding(), &mut lowerer.block, lowerer.location)
        }
        XlaOperation::Broadcast(operation) => {
            check_count!("output", output_types, 1, ProgramError);
            let output_tensor_type = lowerer.lower_tensor_type(&output_types[0])?;
            let result = lowerer.block.append_operation(stable_hlo::broadcast(
                input_values[0],
                output_tensor_type,
                operation.output_axes(),
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.broadcast_in_dim should return one result").as_ref()])
        }
        XlaOperation::Reduce(operation) => {
            check_count!("output", output_types, 1, ProgramError);
            let value = lower_reduce_to_mlir(
                operation.kind(),
                operation.axes(),
                input_values[0],
                &output_types[0],
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            Ok(vec![value])
        }
        XlaOperation::Compare(operation) => {
            let value = lower_compare_to_mlir(
                operation.direction(),
                input_values[0],
                input_values[1],
                &mut lowerer.block,
                lowerer.location,
            )?;
            Ok(vec![value])
        }
        XlaOperation::Not(_) => {
            let result = lowerer.block.append_operation(stable_hlo::not(input_values[0], lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.not should return one result").as_ref()])
        }
        XlaOperation::And(_) => {
            let result =
                lowerer
                    .block
                    .append_operation(stable_hlo::and(input_values[0], input_values[1], lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.and should return one result").as_ref()])
        }
        XlaOperation::Or(_) => {
            let result =
                lowerer
                    .block
                    .append_operation(stable_hlo::or(input_values[0], input_values[1], lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.or should return one result").as_ref()])
        }
        XlaOperation::Xor(_) => {
            let result =
                lowerer
                    .block
                    .append_operation(stable_hlo::xor(input_values[0], input_values[1], lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.xor should return one result").as_ref()])
        }
        XlaOperation::Collective(operation) => {
            check_count!("input", input_values, 1, ProgramError);
            check_count!("output", output_types, 1, ProgramError);
            let collective_state = lowerer.collective_state.clone();
            let result = lower_collective_to_all_reduce(
                operation,
                &collective_state,
                input_values[0],
                &output_types[0],
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            Ok(vec![result])
        }
        XlaOperation::AxisIndex(operation) => {
            check_count!("input", input_values, 0, ProgramError);
            check_count!("output", output_types, 1, ProgramError);
            let collective_state = lowerer.collective_state.clone();
            let result = lower_axis_index_to_coordinate(
                operation,
                &collective_state,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            Ok(vec![result])
        }
        XlaOperation::Select(_) => {
            let result = lowerer.block.append_operation(stable_hlo::select(
                input_values[0],
                input_values[1],
                input_values[2],
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.select should return one result").as_ref()])
        }
        XlaOperation::Slice(operation) => {
            let result = lowerer.block.append_operation(stable_hlo::slice(
                input_values[0],
                operation.start_indices(),
                operation.limit_indices(),
                operation.strides(),
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.slice should return one result").as_ref()])
        }
        XlaOperation::UpdateSlice(operation) => {
            let index_values = lower_static_index_constants(
                operation.start_indices(),
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            let result = lowerer.block.append_operation(stable_hlo::dynamic_update_slice(
                input_values[0],
                input_values[1],
                index_values.as_slice(),
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.dynamic_update_slice should return one result").as_ref()])
        }
        XlaOperation::DynamicSlice(operation) => {
            let result = lowerer.block.append_operation(stable_hlo::dynamic_slice(
                input_values[0],
                &input_values[1..],
                operation.sizes(),
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.dynamic_slice should return one result").as_ref()])
        }
        XlaOperation::DynamicUpdateSlice(_) => {
            let result = lowerer.block.append_operation(stable_hlo::dynamic_update_slice(
                input_values[0],
                input_values[1],
                &input_values[2..],
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.dynamic_update_slice should return one result").as_ref()])
        }
        XlaOperation::Pad(operation) => {
            let edge_padding_low: Vec<i64> =
                operation.edge_padding_low().iter().map(|&padding| padding as i64).collect();
            let edge_padding_high: Vec<i64> =
                operation.edge_padding_high().iter().map(|&padding| padding as i64).collect();
            let result = lowerer.block.append_operation(stable_hlo::pad(
                input_values[0],
                input_values[1],
                edge_padding_low.as_slice(),
                edge_padding_high.as_slice(),
                operation.interior_padding(),
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.pad should return one result").as_ref()])
        }
        XlaOperation::Concatenate(operation) => {
            reject_dynamic_concatenate_output(output_types)?;
            let result = lowerer.block.append_operation(stable_hlo::concatenate(
                input_values,
                operation.axis(),
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.concatenate should return one result").as_ref()])
        }
        XlaOperation::Gather(operation) => lower_gather_to_mlir(
            operation,
            input_values,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        ),
        XlaOperation::Scatter(operation) => lower_scatter_to_mlir(
            operation,
            input_values,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        ),
        XlaOperation::Condition(condition_op) => lowerer.lower_condition(condition_op.as_ref(), input_values),
        XlaOperation::While(while_op) => lowerer.lower_while(while_op.as_ref(), input_values),
        XlaOperation::Scan(scan_op) => lowerer.lower_scan(scan_op.as_ref(), input_values),
        XlaOperation::JitCall(jit_call_op) => lower_jit_call(
            jit_call_op.program_rc(),
            input_values,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
            lowerer.nested_functions.as_ref(),
            &lowerer.collective_state,
            &mut lowerer.token,
        ),
        XlaOperation::ShardMap(shard_map_op) => {
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
    }
}

/// Lowers one traced instruction to the corresponding StableHLO operation and returns its result value.
#[allow(clippy::too_many_arguments)]
fn lower_instruction<'b, 'c: 'b, 't: 'c, ProgramInput, ProgramOutput>(
    program: &XlaProgram<ProgramInput, ProgramOutput>,
    instruction: &Instruction<XlaOperation>,
    _atom_values: &[Option<ValueRef<'b, 'c, 't>>],
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
    token: &mut Option<ValueRef<'b, 'c, 't>>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    ProgramInput: Parameterized<XlaConstant>,
    ProgramOutput: Parameterized<XlaConstant>,
{
    let output_types = instruction
        .outputs()
        .iter()
        .map(|output| program.atoms()[output.index()].r#type().into_owned())
        .collect::<Vec<_>>();
    let captured_values: Vec<ValueRef<'b, 'c, 't>> = Vec::new();
    let mut lowerer = ShardMapMlirLowerer::new(*block, context, location)
        .with_nested_functions(nested_functions.cloned())
        .with_token(*token)
        .with_collective_state(collective_state.clone());
    let outputs = dispatch_lower_shard_map_mlir(
        &instruction.operation(),
        captured_values.as_slice(),
        input_values,
        output_types.as_slice(),
        &mut lowerer,
    )?;
    *token = lowerer.token;
    Ok(outputs)
}

/// Normalizes a user-provided MLIR symbol name.
fn normalize_function_name(function_name: &str) -> Result<String, LoweringError> {
    let function_name = function_name.trim();
    if function_name.is_empty() || function_name.chars().any(char::is_whitespace) {
        return Err(LoweringError::InvalidFunctionName { function_name: function_name.to_string() });
    }
    Ok(function_name.strip_prefix('@').unwrap_or(function_name).to_string())
}

impl ToMlir for ComparisonDirection {
    type Output<'c, 't: 'c> = stable_hlo::ComparisonDirection;

    fn to_mlir<'c, 't: 'c, L: Location<'c, 't>>(&self, _location: L) -> Result<Self::Output<'c, 't>, ryft_mlir::Error> {
        Ok(match self {
            ComparisonDirection::Equal => stable_hlo::ComparisonDirection::Equal,
            ComparisonDirection::NotEqual => stable_hlo::ComparisonDirection::NotEqual,
            ComparisonDirection::LessThan => stable_hlo::ComparisonDirection::LessThan,
            ComparisonDirection::LessThanOrEqual => stable_hlo::ComparisonDirection::LessThanOrEqual,
            ComparisonDirection::GreaterThan => stable_hlo::ComparisonDirection::GreaterThan,
            ComparisonDirection::GreaterThanOrEqual => stable_hlo::ComparisonDirection::GreaterThanOrEqual,
        })
    }
}

/// Lowers an [`ArrayOperation::Compare`]-style dispatch to
/// `stablehlo.compare`. The resulting value has the broadcasted shape of the inputs and Boolean
/// element type. The comparison semantic is routed based on the LHS value's element type
/// (Float / Signed / Unsigned).
fn lower_compare_to_mlir<'b, 'c: 'b, 't: 'c>(
    direction: ComparisonDirection,
    lhs: ValueRef<'b, 'c, 't>,
    rhs: ValueRef<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let direction = direction.to_mlir(location)?;
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

/// Lowers one [`CollectiveOperation`] inside a `sdy.manual_computation` region to a `stablehlo.all_reduce` over the
/// device mesh axis the collective names.
///
/// The replica groups are dense global device-id groups derived from the mesh's row-major device linearization: one
/// group per combination of the other axes' coordinates, each listing the devices along the named axis. The emitted
/// operation carries a module-unique channel id with a device-to-device channel type and `use_global_device_ids`, the
/// standard SPMD emission for cross-partition collectives. `PSum`/`PMean` reduce with `add` (a `PMean` divides the
/// result by the axis size) and `PMax` reduces with `maximum`, reusing the same scalar combiner regions as
/// `stablehlo.reduce` lowering. A collective lowered outside any manual region, or naming an axis the innermost
/// region does not bind as a manual mesh axis, is an error.
fn lower_collective_to_all_reduce<'b, 'c: 'b, 't: 'c>(
    operation: &CollectiveOperation,
    collective_state: &CollectiveLoweringState,
    input_value: ValueRef<'b, 'c, 't>,
    output_array_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let axis_name = operation.axis_name();
    let Some(shard_map) = collective_state.manual_shard_map() else {
        return Err(
            ProgramError::UnsupportedOperation {
                message: format!(
                    "collective over axis '{axis_name}' can only be lowered inside a shard_map manual region",
                ),
            }
            .into(),
        );
    };
    let mesh = shard_map.mesh();
    if !shard_map.manual_axes().iter().any(|manual_axis| manual_axis == axis_name) {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "collective over axis '{axis_name}' cannot lower inside this shard_map manual region because the \
                region does not bind that axis as a manual mesh axis",
            ),
        }
        .into());
    }
    let axis_index = mesh.axis_index(axis_name).unwrap();
    let axis_size = mesh.axis_size(axis_name).unwrap();

    // Devices are linearized row-major over the mesh axes, so the devices along `axis_name` with all other
    // coordinates fixed form one replica group. The group base ids enumerate the other axes' coordinate
    // combinations, and the group members step by the named axis's row-major stride.
    let axis_sizes: Vec<usize> = mesh.axes().iter().map(|axis| axis.size()).collect();
    let stride: usize = axis_sizes[axis_index + 1..].iter().product();
    let device_count: usize = axis_sizes.iter().product();
    let mut replica_groups: Vec<Vec<usize>> = Vec::with_capacity(device_count / axis_size);
    for device in 0..device_count {
        // A device is a group base exactly when its coordinate along the named axis is zero.
        if (device / stride) % axis_size == 0 {
            replica_groups.push((0..axis_size).map(|position| device + position * stride).collect());
        }
    }
    let replica_groups: Vec<&[usize]> = replica_groups.iter().map(Vec::as_slice).collect();

    let element_type = output_array_type.data_type();
    let computation = build_reduce_body_region(operation.kind().reduction_kind(), element_type, context, location)?;
    let result = block.append_operation(stable_hlo::all_reduce(
        &[input_value],
        stable_hlo::ReplicaGroups::dense(replica_groups.as_slice()),
        Some(collective_state.next_channel_id()),
        Some(stable_hlo::ChannelHandleType::DeviceToDevice),
        true,
        computation,
        location,
    )?)?;
    let reduced = result.result(0).expect("stablehlo.all_reduce should return one result").as_ref();
    if !matches!(operation.kind(), CollectiveKind::PMean) {
        return Ok(reduced);
    }
    // `PMean` is the mean over the named axis: divide the all-reduced sum by the axis size.
    let output_tensor_type = lower_tensor_type(output_array_type, context, location)?;
    let divisor =
        lower_f64_constant_splat(axis_size as f64, output_array_type, output_tensor_type, block, context, location)?;
    let mean = block.append_operation(stable_hlo::divide(reduced, divisor, location)?)?;
    Ok(mean.result(0).expect("stablehlo.divide should return one result").as_ref())
}

/// Lowers a scalar `u64` constant used by the [`AxisIndexOperation`] coordinate arithmetic below.
fn lower_u64_scalar_constant<'b, 'c: 'b, 't: 'c>(
    value: u64,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let scalar_tensor_type = lower_tensor_type(&ArrayType::scalar(DataType::U64), context, location)?;
    let attribute = context
        .dense_u64_elements_attribute(scalar_tensor_type, &[value])
        .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: DataType::U64 })?;
    let constant = block.append_operation(stable_hlo::constant(attribute, location)?)?;
    Ok(constant.result(0).expect("stablehlo.constant should return one result").as_ref())
}

/// Lowers one [`AxisIndexOperation`] inside a `sdy.manual_computation` region to the executing device's coordinate
/// along the named device mesh axis, as a scalar `u64`.
///
/// Devices are linearized row-major over the mesh axes, so the coordinate along the named axis is
/// `(partition_id / stride) % size`, where `stride` is the product of the mesh axis sizes after the named axis (the
/// same linearization [`lower_collective_to_all_reduce`] uses for replica groups) and `size` is the named axis's size.
/// `partition_id` yields the executing device's linear id; it is converted to `u64` and the coordinate arithmetic is
/// emitted as scalar StableHLO divide/remainder. A single-device axis collapses to the constant `0`.
fn lower_axis_index_to_coordinate<'b, 'c: 'b, 't: 'c>(
    operation: &AxisIndexOperation,
    collective_state: &CollectiveLoweringState,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let axis_name = operation.axis_name();
    let Some(shard_map) = collective_state.manual_shard_map() else {
        return Err(ProgramError::UnsupportedOperation {
            message: format!("axis_index for axis '{axis_name}' can only be lowered inside a shard_map manual region"),
        }
        .into());
    };
    let mesh = shard_map.mesh();
    if !shard_map.manual_axes().iter().any(|manual_axis| manual_axis == axis_name) {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "axis_index for axis '{axis_name}' cannot lower inside this shard_map manual region because the \
                region does not bind that axis as a manual mesh axis",
            ),
        }
        .into());
    }
    let axis_index = mesh.axis_index(axis_name).unwrap();
    let axis_size = mesh.axis_size(axis_name).unwrap();
    let axis_sizes: Vec<usize> = mesh.axes().iter().map(|axis| axis.size()).collect();
    let stride: usize = axis_sizes[axis_index + 1..].iter().product();

    let scalar_u64_type = lower_tensor_type(&ArrayType::scalar(DataType::U64), context, location)?;
    let partition_id = block.append_operation(stable_hlo::partition_id(location)?)?;
    let partition_id = partition_id.result(0).expect("stablehlo.partition_id should return one result").as_ref();
    let partition_id = block.append_operation(stable_hlo::convert(partition_id, scalar_u64_type, location)?)?;
    let partition_id = partition_id.result(0).expect("stablehlo.convert should return one result").as_ref();
    // `partition_id / stride` selects the coordinate's digit group; `% size` isolates this axis's coordinate. A unit
    // stride and a full-size axis make the respective step an identity, so they are skipped.
    let strided = if stride == 1 {
        partition_id
    } else {
        let stride_constant = lower_u64_scalar_constant(stride as u64, block, context, location)?;
        let divided = block.append_operation(stable_hlo::divide(partition_id, stride_constant, location)?)?;
        divided.result(0).expect("stablehlo.divide should return one result").as_ref()
    };
    if axis_size == mesh.device_count() {
        return Ok(strided);
    }
    let size_constant = lower_u64_scalar_constant(axis_size as u64, block, context, location)?;
    let coordinate = block.append_operation(stable_hlo::remainder(strided, size_constant, location)?)?;
    Ok(coordinate.result(0).expect("stablehlo.remainder should return one result").as_ref())
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
        return Err(LoweringError::UnsupportedOp { op: "reduce_mean".to_string() });
    }
    Ok(sum_result)
}

/// Lowers an [`ArrayOperation::Gather`] dispatch to `stablehlo.gather`. StableHLO `gather` clamps out-of-bounds start
/// indices into range by default, which is exactly [`GatherScatterMode::Clip`] semantics, so both `Clip` and
/// [`GatherScatterMode::PromiseInBounds`] (whose promise only lets the clamp be a no-op) lower to the bare op.
/// [`GatherScatterMode::FillOrDrop`] instead fills out-of-bounds windows and needs an explicit out-of-bounds
/// mask/select that is not yet emitted. The implicit index vector dimension is the last indices axis, which the gather
/// shape rule fixes at `output_rank - offset_dimensions.len()`.
fn lower_gather_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &GatherOperation,
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    check_count!("input", input_values, 2, ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    if operation.mode() == GatherScatterMode::FillOrDrop {
        return Err(LoweringError::UnsupportedOp { op: format!("gather with mode {}", operation.mode()) });
    }
    let dimensions = operation.dimensions();
    let index_vector_dimension = output_types[0].rank() - dimensions.offset_dimensions().len();
    let attribute = context.stable_hlo_gather_dimensions(
        dimensions.offset_dimensions(),
        dimensions.collapsed_slice_dimensions(),
        dimensions.operand_batching_dimensions(),
        dimensions.start_indices_batching_dimensions(),
        dimensions.start_index_map(),
        index_vector_dimension,
    )?;
    let result = block.append_operation(stable_hlo::gather(
        input_values[0],
        input_values[1],
        attribute,
        operation.slice_sizes(),
        operation.indices_are_sorted(),
        location,
    )?)?;
    Ok(vec![result.result(0).expect("stablehlo.gather should return one result").as_ref()])
}

/// Builds the scalar combiner region of a `stablehlo.scatter` for the given [`ScatterReductionKind`], modeled on
/// [`build_reduce_body_region`]. The region's block takes the existing operand scalar and the update scalar and
/// returns the combined value: `Overwrite` returns the update directly (no combine op), and the others apply the
/// matching elementwise StableHLO op.
fn build_scatter_combiner_region<'c, 't>(
    kind: ScatterReductionKind,
    element_type: DataType,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ryft_mlir::DetachedRegion<'c, 't>, LoweringError> {
    let scalar_tensor_type = lower_tensor_type(&ArrayType::scalar(element_type), context, location)?;
    let block = context.block(&[(scalar_tensor_type, location), (scalar_tensor_type, location)]);
    let mut region = context.region();
    let mut block_ref = region.append_block(block)?;
    let lhs = block_ref.argument(0)?.as_ref();
    let rhs = block_ref.argument(1)?.as_ref();
    let body_value = match kind {
        ScatterReductionKind::Overwrite => rhs,
        ScatterReductionKind::Add => block_ref
            .append_operation(stable_hlo::add(lhs, rhs, location)?)?
            .result(0)
            .expect("stablehlo.add should return one result")
            .as_ref(),
        ScatterReductionKind::Mul => block_ref
            .append_operation(stable_hlo::multiply(lhs, rhs, location)?)?
            .result(0)
            .expect("stablehlo.multiply should return one result")
            .as_ref(),
        ScatterReductionKind::Min => block_ref
            .append_operation(stable_hlo::minimum(lhs, rhs, location)?)?
            .result(0)
            .expect("stablehlo.minimum should return one result")
            .as_ref(),
        ScatterReductionKind::Max => block_ref
            .append_operation(stable_hlo::maximum(lhs, rhs, location)?)?
            .result(0)
            .expect("stablehlo.maximum should return one result")
            .as_ref(),
    };
    block_ref.append_operation(stable_hlo::r#return(&[body_value], location)?)?;
    Ok(region)
}

/// Lowers an [`ArrayOperation::Scatter`] dispatch to `stablehlo.scatter` with the combiner region selected by the
/// operation's [`ScatterReductionKind`]. As with gather, StableHLO `scatter` clamps out-of-bounds start indices by
/// default, so both [`GatherScatterMode::Clip`] and [`GatherScatterMode::PromiseInBounds`] lower to the bare op while
/// [`GatherScatterMode::FillOrDrop`] (which drops out-of-bounds writes) is not yet emitted. The implicit index vector
/// dimension is the last indices axis (`indices_rank - 1`).
fn lower_scatter_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &ScatterOperation,
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    check_count!("input", input_values, 3, ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    if operation.mode() == GatherScatterMode::FillOrDrop {
        return Err(LoweringError::UnsupportedOp { op: format!("scatter with mode {}", operation.mode()) });
    }
    let indices_rank = input_values[1]
        .r#type()?
        .cast::<TensorTypeRef>()
        .ok_or_else(|| LoweringError::UnsupportedOp { op: "scatter with non-tensor indices".to_string() })?
        .rank();
    let dimensions = operation.dimensions();
    let attribute = context.stable_hlo_scatter_dimensions(
        dimensions.update_window_dimensions(),
        dimensions.inserted_window_dimensions(),
        dimensions.operand_batching_dimensions(),
        dimensions.scatter_indices_batching_dimensions(),
        dimensions.scatter_dimensions_to_operand_dimensions(),
        indices_rank - 1,
    )?;
    let combiner = build_scatter_combiner_region(operation.kind(), output_types[0].data_type(), context, location)?;
    let result = block.append_operation(stable_hlo::scatter(
        &[input_values[0]],
        input_values[1],
        &[input_values[2]],
        attribute,
        combiner,
        operation.indices_are_sorted(),
        operation.unique_indices(),
        location,
    )?)?;
    Ok(vec![result.result(0).expect("stablehlo.scatter should return one result").as_ref()])
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
        DataType::F6E2M3FN => context.float6e2m3fn_type().as_ref(),
        DataType::F6E3M2FN => context.float6e3m2fn_type().as_ref(),
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
/// matches `output_type`, then broadcasts that scalar to the full output shape. Used by the
/// constant-fill and `fill` lowerings.
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

/// Lowers a [`Scalar`] fill value as a splatted constant of `output_type`. Real fill values route through
/// [`lower_f64_constant_splat`] after an exact widening cast to `f64`, while complex fill values compose two real
/// part splats through `stablehlo.complex` (MLIR has no complex scalar attribute to splat) and broadcast the
/// composed scalar when the output is not rank zero. A complex fill value with a real output data type surfaces the
/// cast's promotion error.
fn lower_scalar_constant_splat<'b, 'c: 'b, 't: 'c, B, L>(
    value: Scalar,
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
    if matches!(data_type, DataType::C64 | DataType::C128) {
        let (real, imaginary) = match value {
            Scalar::C64(value) => (value.re as f64, value.im as f64),
            Scalar::C128(value) => (value.re, value.im),
            value => {
                let Scalar::F64(real) = value.cast(DataType::F64)? else {
                    unreachable!("a cast to f64 yields an f64 scalar")
                };
                (real, 0.0)
            }
        };
        let part_data_type = if data_type == DataType::C64 { DataType::F32 } else { DataType::F64 };
        let part_type = ArrayType::scalar(part_data_type);
        let part_tensor_type = context
            .tensor_type(lower_element_type(part_data_type, context)?, &[], None, location)
            .map_err(|_| LoweringError::InvalidTensorType { array_type: part_type.clone() })?;
        let real_value = lower_f64_constant_splat(real, &part_type, part_tensor_type, block, context, location)?;
        let imaginary_value =
            lower_f64_constant_splat(imaginary, &part_type, part_tensor_type, block, context, location)?;
        let complex = block.append_operation(stable_hlo::complex(real_value, imaginary_value, location)?)?;
        let complex = complex.result(0).expect("stablehlo.complex should return one result").as_ref();
        if output_type.shape().dimensions().is_empty() {
            return Ok(complex);
        }
        let broadcast = block.append_operation(stable_hlo::broadcast(complex, output_tensor_type, &[], location)?)?;
        return Ok(broadcast.result(0).expect("stablehlo.broadcast should return one result").as_ref());
    }
    let Scalar::F64(value) = value.cast(DataType::F64)? else { unreachable!("a cast to f64 yields an f64 scalar") };
    lower_f64_constant_splat(value, output_type, output_tensor_type, block, context, location)
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
    integer_value: i64,
    context: &'c MlirContext<'t>,
) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
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
        DataType::F6E2M3FN => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float6e2m3fn_type(), float_value),
            )
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::F6E3M2FN => context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                context.float_attribute(context.float6e3m2fn_type(), float_value),
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
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use ryft_core::EagerContext;
    use ryft_core::contexts::Context;
    use ryft_core::operations::constants::{OneLike, OneOperation, ZeroLike};
    use ryft_core::operations::manipulation::{
        ConcatenateOperation, DynamicSliceOperation, DynamicUpdateSliceOperation, PadOperation, SliceOperation,
        Transpose, UpdateSliceOperation,
    };
    use ryft_core::operations::trigonometric::{Cos, Sin};
    use ryft_core::parameters::Placeholder;
    use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use ryft_core::tests::TestArray;
    use ryft_core::tracing_v2::ArrayOperation;
    use ryft_core::tracing_v2::ReverseModeDifferentiate;
    use ryft_core::tracing_v2::operations::dot::{Dot, DotDimensionNumbers};
    use ryft_core::types::{Shape, Size};

    use super::super::shard_map::{TracedShardMap, shard_map as traced_shard_map};
    use ryft_core::tracing::Trace;

    use super::*;

    fn test_manual_mesh(axis_name: &str, axis_size: usize) -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::new(axis_name, axis_size, MeshAxisType::Manual).unwrap()]).unwrap()
    }

    fn test_vector_type(length: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(length)]))
    }

    fn test_matrix_type(rows: usize, cols: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(rows), Size::Static(cols)]))
    }

    fn xla_identity_branch(input_type: ArrayType) -> FlatXlaProgram {
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn xla_neg_branch(input_type: ArrayType) -> FlatXlaProgram {
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let output = builder.add_instruction(NegOperation, vec![input]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn lower_traced_module(
        traced: &TracedShardMap<ArrayType, ArrayType>,
        function_name: &str,
    ) -> Result<String, super::super::shard_map::ShardMapTraceError> {
        traced.to_mlir_module(function_name)
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_captures_as_hidden_arguments() {
        let array_type = test_vector_type(4);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(array_type.clone());
        let capture = builder.add_constant(XlaConstant::new(0, array_type.clone()));
        let output = builder.add_instruction(AddOperation, vec![input, capture]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let capture_types = vec![array_type.clone()];
        let input_types = vec![array_type.clone()];
        let output_types = vec![array_type];
        let stablehlo = to_mlir_module_for_program(
            &program,
            capture_types.as_slice(),
            &input_types,
            &output_types,
            "main",
            None,
            None,
        )
        .unwrap();

        assert!(
            stablehlo.contains("func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32>"),
            "{stablehlo}",
        );
        assert!(stablehlo.contains("stablehlo.add %arg1, %arg0 : tensor<4xf32>"), "{stablehlo}");
    }

    /// Builds the flat callee `f(x) = x + x` over a vector type, returned behind an [`Rc`] so callers control
    /// whether `jit_call` sites share one program (pointer identity) or use structurally-identical distinct programs.
    fn xla_add_self_callee(input_type: ArrayType) -> std::rc::Rc<FlatXlaProgram> {
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let output = builder.add_instruction(AddOperation, vec![input, input]).unwrap()[0];
        std::rc::Rc::new(builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap())
    }

    /// Wraps `callee` in a `jit_call` operation.
    fn xla_jit_call(callee: std::rc::Rc<FlatXlaProgram>) -> XlaOperation {
        XlaOperation::JitCall(Box::new(crate::experimental::ops::JitCallOperation::new(callee)))
    }

    /// Lowers an outer program that calls `callees` (one `jit_call` each) and sums the results, returning the
    /// module text. Each callee is `f(x) = x + x`; the outer function is `g(x) = sum_i callee_i(x)`.
    fn lower_two_jit_call_module(callees: Vec<std::rc::Rc<FlatXlaProgram>>) -> String {
        let array_type = test_vector_type(4);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(array_type.clone());
        let mut accumulator: Option<AtomId> = None;
        for callee in callees {
            let call_output = builder.add_instruction(xla_jit_call(callee), vec![input]).unwrap()[0];
            accumulator = Some(match accumulator {
                None => call_output,
                Some(previous) => builder.add_instruction(AddOperation, vec![previous, call_output]).unwrap()[0],
            });
        }
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![accumulator.expect("at least one callee")],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let input_types = vec![array_type.clone()];
        let output_types = vec![array_type];
        to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap()
    }

    #[test]
    fn test_repeated_jit_call_sharing_one_program_emits_one_shared_function() {
        let callee = xla_add_self_callee(test_vector_type(4));
        let module = lower_two_jit_call_module(vec![callee.clone(), callee]);

        // Both calls of the one shared program collapse to a single private `func.func` plus two `func.call`s.
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func private @jit_call_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = call @jit_call_0(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %1 = call @jit_call_0(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %2 = stablehlo.add %0, %1 : tensor<4xf32>
                    return %2 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_structurally_identical_jit_calls_share_one_function() {
        // Two distinct programs (separate `Rc`s) with identical structure — the shape produced when a transform such
        // as `grad` linearizes each of several identical blocks into its own staged program — must still deduplicate
        // into the same single shared function as the pointer-identical case above.
        let module = lower_two_jit_call_module(vec![
            xla_add_self_callee(test_vector_type(4)),
            xla_add_self_callee(test_vector_type(4)),
        ]);

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func private @jit_call_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = call @jit_call_0(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %1 = call @jit_call_0(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %2 = stablehlo.add %0, %1 : tensor<4xf32>
                    return %2 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_single_jit_call_is_inlined() {
        // A single-occurrence callee stays below the dedup threshold and inlines, so no shared function is emitted
        // and the callee body appears directly in `@main`.
        let module = lower_two_jit_call_module(vec![xla_add_self_callee(test_vector_type(4))]);

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
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
                let product = x.transpose(vec![1, 0]).unwrap().dot(&x, &DotDimensionNumbers::matmul());
                let waveform = (-product).cos().unwrap().sin().unwrap();
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
        let condition =
            ConditionOperation::new(xla_neg_branch(input_type.clone()), xla_identity_branch(input_type.clone()))
                .unwrap();
        let mut builder = XlaProgramBuilder::new();
        let predicate = builder.add_input(predicate_type);
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(XlaOperation::Condition(Box::new(condition)), vec![predicate, input])
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
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
    fn test_to_mlir_module_for_plain_program_lowers_gather() {
        use ryft_core::operations::manipulation::{GatherDimensionNumbers, GatherOperation};
        use ryft_core::types::{Shape, Size};

        // Take whole rows of a [3, 2] matrix at the row indices in a [2, 1] index array: offset axis 1 carries the
        // row (slice sizes [1, 2]); axis 0 is collapsed (start-index driven). Output is [2, 2].
        let operand_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3), Size::Static(2)]));
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![Size::Static(2), Size::Static(1)]));
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        let mut builder = XlaProgramBuilder::new();
        let operand = builder.add_input(operand_type);
        let indices = builder.add_input(indices_type);
        let output = builder.add_instruction(XlaOperation::Gather(operation), vec![operand, indices]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert!(stablehlo.contains("stablehlo.gather"), "{stablehlo}");
        assert!(stablehlo.contains("offset_dims = [1]"), "{stablehlo}");
        assert!(stablehlo.contains("collapsed_slice_dims = [0]"), "{stablehlo}");
        assert!(stablehlo.contains("start_index_map = [0]"), "{stablehlo}");
        assert!(stablehlo.contains("slice_sizes = array<i64: 1, 2>"), "{stablehlo}");
        assert!(stablehlo.contains("-> tensor<2x2xf32>"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_clip_mode_gather_to_bare_op() {
        use ryft_core::operations::manipulation::{GatherDimensionNumbers, GatherOperation, GatherScatterMode};
        use ryft_core::types::{Shape, Size};

        // `Clip` is StableHLO `gather`'s default out-of-bounds behavior, so a `Clip`-mode gather lowers to the bare
        // `stablehlo.gather` (no extra clamp ops) just like the in-bounds default rather than erroring.
        let operand_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3), Size::Static(2)]));
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![Size::Static(2), Size::Static(1)]));
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2])
            .with_mode(GatherScatterMode::Clip);
        let mut builder = XlaProgramBuilder::new();
        let operand = builder.add_input(operand_type);
        let indices = builder.add_input(indices_type);
        let output = builder.add_instruction(XlaOperation::Gather(operation), vec![operand, indices]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert!(stablehlo.contains("stablehlo.gather"), "{stablehlo}");
        assert!(stablehlo.contains("-> tensor<2x2xf32>"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_scatter() {
        use ryft_core::operations::manipulation::{ScatterDimensionNumbers, ScatterOperation, ScatterReductionKind};
        use ryft_core::types::{Shape, Size};

        // Scatter-add row updates into a [3, 2] operand at the row indices in a [2, 1] index array. Output is [3, 2],
        // and the Add combiner lowers to a `stablehlo.add` inside the scatter region.
        let operand_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3), Size::Static(2)]));
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![Size::Static(2), Size::Static(1)]));
        let updates_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(2)]));
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]), ScatterReductionKind::Add);
        let mut builder = XlaProgramBuilder::new();
        let operand = builder.add_input(operand_type);
        let indices = builder.add_input(indices_type);
        let updates = builder.add_input(updates_type);
        let output =
            builder.add_instruction(XlaOperation::Scatter(operation), vec![operand, indices, updates]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert!(stablehlo.contains("stablehlo.scatter"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.add"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.return"), "{stablehlo}");
        assert!(stablehlo.contains("tensor<3x2xf32>"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_constant_predicate_condition_to_stablehlo_if() {
        // A condition whose predicate input is fed by a staged constant still lowers to `stablehlo.if`; folding the
        // constant predicate away is the backend's job (StableHLO canonicalization and XLA's conditional
        // simplification), not ryft's.
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let input_type = ArrayType::scalar(DataType::F32);
        let condition =
            ConditionOperation::new(xla_neg_branch(input_type.clone()), xla_identity_branch(input_type.clone()))
                .unwrap();
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let predicate = builder.add_instruction(OneOperation::new(predicate_type), vec![]).unwrap()[0];
        let output = builder
            .add_instruction(XlaOperation::Condition(Box::new(condition)), vec![predicate, input])
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert!(stablehlo.contains("\"stablehlo.if\""), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.constant"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.negate"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_while_to_stablehlo_while() {
        let state_type = ArrayType::scalar(DataType::Boolean);
        let while_operation =
            WhileOperation::new(xla_identity_branch(state_type.clone()), xla_identity_branch(state_type.clone()))
                .unwrap();
        let mut builder = XlaProgramBuilder::new();
        let state = builder.add_input(state_type);
        let output = builder.add_instruction(XlaOperation::While(Box::new(while_operation)), vec![state]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert!(stablehlo.contains("stablehlo.while"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.return"), "{stablehlo}");
        // An unbounded while emits no iteration-counter machinery.
        assert!(!stablehlo.contains("stablehlo.and"), "{stablehlo}");
        assert!(!stablehlo.contains("stablehlo.add"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_bounded_while_with_fused_counter_condition() {
        // A semantic iteration bound threads an internal i64 counter through the `stablehlo.while` state: the
        // condition region conjoins `counter < bound` into the original predicate via `stablehlo.compare` plus
        // `stablehlo.and`, and the body region increments the counter via `stablehlo.add`. The operation's outputs
        // remain the original state elements.
        let state_type = ArrayType::scalar(DataType::Boolean);
        let while_operation =
            WhileOperation::new(xla_identity_branch(state_type.clone()), xla_identity_branch(state_type.clone()))
                .unwrap()
                .with_iteration_bound(3)
                .unwrap();
        let mut builder = XlaProgramBuilder::new();
        let state = builder.add_input(state_type);
        let output = builder.add_instruction(XlaOperation::While(Box::new(while_operation)), vec![state]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert!(stablehlo.contains("stablehlo.while"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.compare"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.and"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.add"), "{stablehlo}");
        assert!(stablehlo.contains("tensor<i64>"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_batched_predicate_while_with_masked_state_updates() {
        // A batched (per-item) predicate lowers with the masked semantics owned by the while lowering: the condition
        // region reduces the `tensor<3xi1>` predicate to the scalar loop-continuation decision with a Boolean `or`
        // reduction, and the body region recomputes the per-item predicate on the incoming state and selects per
        // state element between the body's candidate update and the carried state, freezing finished items. The
        // predicate shape equals the state shape here, so no broadcast is needed for the mask.
        use ryft_core::operations::compare::CompareOperation;
        use ryft_core::operations::constants::{OneLikeOperation, ZeroLikeOperation};
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let condition = {
            let mut builder = XlaProgramBuilder::new();
            let state = builder.add_input(state_type.clone());
            let zero = builder.add_instruction(ZeroLikeOperation, vec![state]).unwrap()[0];
            let predicate = builder
                .add_instruction(CompareOperation::new(ComparisonDirection::GreaterThan), vec![state, zero])
                .unwrap()[0];
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![predicate], vec![Placeholder], vec![Placeholder])
        }
        .unwrap();
        let body = {
            let mut builder = XlaProgramBuilder::new();
            let state = builder.add_input(state_type.clone());
            let one = builder.add_instruction(OneLikeOperation, vec![state]).unwrap()[0];
            let next = builder.add_instruction(SubOperation, vec![state, one]).unwrap()[0];
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![next], vec![Placeholder], vec![Placeholder])
        }
        .unwrap();
        let while_operation = WhileOperation::new(condition, body).unwrap();
        let mut builder = XlaProgramBuilder::new();
        let state = builder.add_input(state_type);
        let output = builder.add_instruction(XlaOperation::While(Box::new(while_operation)), vec![state]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert!(stablehlo.contains("stablehlo.while"), "{stablehlo}");
        // Condition region: `or`-reduce the per-item predicate into the scalar continuation decision.
        assert!(stablehlo.contains("stablehlo.reduce"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.or"), "{stablehlo}");
        // Body region: per-item masked state update under the recomputed predicate.
        assert!(stablehlo.contains("stablehlo.select"), "{stablehlo}");
        assert!(stablehlo.contains("tensor<3xi1>"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_scan_to_while() {
        // A primal scan lowers to a `stablehlo.while` over `[counter, carries..., xs..., ys...]`: each iteration
        // reads one slice of the stacked inputs with `stablehlo.dynamic_slice`, inlines the body, and writes the
        // per-iteration outputs into preallocated zero accumulators with `stablehlo.dynamic_update_slice` (the
        // strategy JAX uses for `lax.scan`, which is not an XLA primitive).
        use ryft_core::operations::control_flow::ScanOperation as CoreScanOperation;

        let scalar_f32 = ArrayType::scalar(DataType::F32);
        let mut body_builder = XlaProgramBuilder::new();
        let carry = body_builder.add_input(scalar_f32.clone());
        let x = body_builder.add_input(scalar_f32.clone());
        let product = body_builder.add_instruction(MulOperation, vec![carry, x]).unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![product, product],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let scan = CoreScanOperation::<_, _>::new(body, 1, 3).unwrap();

        let mut builder = XlaProgramBuilder::new();
        let init = builder.add_input(scalar_f32);
        let stacked_inputs = builder.add_input(test_vector_type(3));
        let outputs = builder
            .add_instruction(XlaOperation::Scan(Box::new(scan)), vec![init, stacked_inputs])
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                outputs,
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert!(stablehlo.contains("stablehlo.while"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.compare"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.dynamic_slice"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.dynamic_update_slice"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.multiply"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_fully_unrolled_scan_without_while() {
        // A scan whose unroll factor equals its length lowers to straight-line operations: no `stablehlo.while` is
        // emitted at all and the body inlines once per iteration (three `stablehlo.multiply` copies for `length = 3`).
        use ryft_core::operations::control_flow::ScanOperation as CoreScanOperation;

        let scalar_f32 = ArrayType::scalar(DataType::F32);
        let mut body_builder = XlaProgramBuilder::new();
        let carry = body_builder.add_input(scalar_f32.clone());
        let x = body_builder.add_input(scalar_f32.clone());
        let product = body_builder.add_instruction(MulOperation, vec![carry, x]).unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![product, product],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let scan = CoreScanOperation::<_, _>::new(body, 1, 3).unwrap().with_unroll(3).unwrap();

        let mut builder = XlaProgramBuilder::new();
        let init = builder.add_input(scalar_f32);
        let stacked_inputs = builder.add_input(test_vector_type(3));
        let outputs = builder
            .add_instruction(XlaOperation::Scan(Box::new(scan)), vec![init, stacked_inputs])
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                outputs,
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert!(!stablehlo.contains("stablehlo.while"), "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.multiply").count(), 3, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.dynamic_slice").count(), 3, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.dynamic_update_slice").count(), 3, "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_partially_unrolled_scan() {
        // A scan with `unroll = 2` over `length = 4` keeps the `stablehlo.while` skeleton but runs two body copies
        // per loop trip: the body region contains two `stablehlo.multiply` copies (and one iteration read/write pair
        // per copy) while the counter advances by the unroll factor.
        use ryft_core::operations::control_flow::ScanOperation as CoreScanOperation;

        let scalar_f32 = ArrayType::scalar(DataType::F32);
        let mut body_builder = XlaProgramBuilder::new();
        let carry = body_builder.add_input(scalar_f32.clone());
        let x = body_builder.add_input(scalar_f32.clone());
        let product = body_builder.add_instruction(MulOperation, vec![carry, x]).unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![product, product],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let scan = CoreScanOperation::<_, _>::new(body, 1, 4).unwrap().with_unroll(2).unwrap();

        let mut builder = XlaProgramBuilder::new();
        let init = builder.add_input(scalar_f32);
        let stacked_inputs = builder.add_input(test_vector_type(4));
        let outputs = builder
            .add_instruction(XlaOperation::Scan(Box::new(scan)), vec![init, stacked_inputs])
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                outputs,
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert!(stablehlo.contains("stablehlo.while"), "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.multiply").count(), 2, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.dynamic_slice").count(), 2, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.dynamic_update_slice").count(), 2, "{stablehlo}");
    }

    // ---------------------------------------------------------------------------
    // Print lowering / effect-token threading tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_to_mlir_module_for_program_lowers_two_prints_through_one_token_chain() {
        use ryft_core::operations::debugging::PrintOperation;

        // Two prints in one flat program: the token chain is created lazily by one zero-operand
        // `stablehlo.after_all` at the first print, the second print consumes the first print's token result, and
        // each print's dataflow output is its forwarded operand (the final add reads `%arg0` and `%0`, not custom
        // call results).
        let array_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(array_type.clone());
        let doubled = builder.add_instruction(AddOperation, vec![input, input]).unwrap()[0];
        let first = builder.add_instruction(PrintOperation::new("first"), vec![input]).unwrap()[0];
        let second = builder.add_instruction(PrintOperation::new("second"), vec![doubled]).unwrap()[0];
        let output = builder.add_instruction(AddOperation, vec![first, second]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let input_types = vec![array_type.clone()];
        let output_types = vec![array_type];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xf64>) -> tensor<2xf64> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<2xf64>
                    %1 = stablehlo.after_all  : !stablehlo.token
                    %2 = stablehlo.custom_call @ryft.print(%arg0, %1) {api_version = 4 : i32, backend_config = {label = "first"}, has_side_effect = true} : (tensor<2xf64>, !stablehlo.token) -> !stablehlo.token
                    %3 = stablehlo.custom_call @ryft.print(%0, %2) {api_version = 4 : i32, backend_config = {label = "second"}, has_side_effect = true} : (tensor<2xf64>, !stablehlo.token) -> !stablehlo.token
                    %4 = stablehlo.add %arg0, %0 : tensor<2xf64>
                    return %4 : tensor<2xf64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_scan_body_print_with_token_loop_state() {
        use ryft_core::operations::control_flow::ScanOperation as CoreScanOperation;
        use ryft_core::operations::debugging::PrintOperation;

        // A print inside a scan body makes the lowered `stablehlo.while` carry the effect token as one extra
        // trailing state element: the entry token is created before the loop, both regions receive it as an extra
        // block argument, the body threads it through the print custom call, and the loop's trailing result
        // continues the chain (unused here because the program ends right after the scan).
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut body_builder = XlaProgramBuilder::new();
        let carry = body_builder.add_input(scalar_f64.clone());
        let x = body_builder.add_input(scalar_f64.clone());
        let printed = body_builder.add_instruction(PrintOperation::new("iteration"), vec![x]).unwrap()[0];
        let sum = body_builder.add_instruction(AddOperation, vec![carry, printed]).unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        let scan = CoreScanOperation::<_, _>::new(body, 1, 3).unwrap();

        let stacked_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let mut builder = XlaProgramBuilder::new();
        let init = builder.add_input(scalar_f64.clone());
        let stacked_inputs = builder.add_input(stacked_type.clone());
        let output =
            builder.add_instruction(XlaOperation::Scan(Box::new(scan)), vec![init, stacked_inputs]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let input_types = vec![scalar_f64.clone(), stacked_type];
        let output_types = vec![scalar_f64];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<3xf64>) -> tensor<f64> {
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1:4 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0, %iterArg_1 = %arg1, %iterArg_2 = %0) : tensor<i64>, tensor<f64>, tensor<3xf64>, !stablehlo.token
                    cond {
                      %c_3 = stablehlo.constant dense<3> : tensor<i64>
                      %2 = stablehlo.compare LT, %iterArg, %c_3, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                      stablehlo.return %2 : tensor<i1>
                    } do {
                      %c_3 = stablehlo.constant dense<0> : tensor<i64>
                      %2 = stablehlo.dynamic_slice %iterArg_1, %iterArg, sizes = [1] : (tensor<3xf64>, tensor<i64>) -> tensor<1xf64>
                      %3 = stablehlo.reshape %2 : (tensor<1xf64>) -> tensor<f64>
                      %4 = stablehlo.custom_call @ryft.print(%3, %iterArg_2) {api_version = 4 : i32, backend_config = {label = "iteration"}, has_side_effect = true} : (tensor<f64>, !stablehlo.token) -> !stablehlo.token
                      %5 = stablehlo.add %iterArg_0, %3 : tensor<f64>
                      %c_4 = stablehlo.constant dense<1> : tensor<i64>
                      %6 = stablehlo.add %iterArg, %c_4 : tensor<i64>
                      stablehlo.return %6, %5, %iterArg_1, %4 : tensor<i64>, tensor<f64>, tensor<3xf64>, !stablehlo.token
                    }
                    return %1#1 : tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_condition_branch_print_with_token_result() {
        use ryft_core::operations::debugging::PrintOperation;

        // A print inside one condition branch makes the lowered `stablehlo.if` return the branch's final effect
        // token as one extra trailing result: both branches capture the entry token implicitly, the effectful branch
        // returns its print custom call's token, and the pure branch returns the entry token unchanged.
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let input_type = ArrayType::scalar(DataType::F64);
        let mut true_builder = XlaProgramBuilder::new();
        let true_input = true_builder.add_input(input_type.clone());
        let printed = true_builder.add_instruction(PrintOperation::new("taken"), vec![true_input]).unwrap()[0];
        let negated = true_builder.add_instruction(NegOperation, vec![printed]).unwrap()[0];
        let true_branch = true_builder.build(vec![negated], vec![Placeholder], vec![Placeholder]).unwrap();
        let condition = ConditionOperation::new(true_branch, xla_identity_branch(input_type.clone())).unwrap();

        let mut builder = XlaProgramBuilder::new();
        let predicate = builder.add_input(predicate_type.clone());
        let input = builder.add_input(input_type.clone());
        let output = builder
            .add_instruction(XlaOperation::Condition(Box::new(condition)), vec![predicate, input])
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let input_types = vec![predicate_type, input_type.clone()];
        let output_types = vec![input_type];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<i1>, %arg1: tensor<f64>) -> tensor<f64> {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1:2 = "stablehlo.if"(%arg0) ({
                      %2 = stablehlo.custom_call @ryft.print(%arg1, %0) {api_version = 4 : i32, backend_config = {label = "taken"}, has_side_effect = true} : (tensor<f64>, !stablehlo.token) -> !stablehlo.token
                      %3 = stablehlo.negate %arg1 : tensor<f64>
                      stablehlo.return %3, %2 : tensor<f64>, !stablehlo.token
                    }, {
                      stablehlo.return %arg1, %0 : tensor<f64>, !stablehlo.token
                    }) : (tensor<i1>) -> (tensor<f64>, !stablehlo.token)
                    return %1#0 : tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_repeated_effectful_jit_call_callees_inline_and_chain_prints() {
        use ryft_core::operations::debugging::PrintOperation;

        // A repeated `jit_call` callee that prints is excluded from function deduplication and inlines at every
        // call site, so both inlined prints chain onto the caller's single token chain in program order (a shared
        // token-free `func.func` could not preserve that ordering).
        let array_type = test_vector_type(4);
        let mut callee_builder = XlaProgramBuilder::new();
        let callee_input = callee_builder.add_input(array_type.clone());
        let printed = callee_builder.add_instruction(PrintOperation::new("callee"), vec![callee_input]).unwrap()[0];
        let callee_output = callee_builder.add_instruction(AddOperation, vec![printed, printed]).unwrap()[0];
        let callee =
            std::rc::Rc::new(callee_builder.build(vec![callee_output], vec![Placeholder], vec![Placeholder]).unwrap());
        let module = lower_two_jit_call_module(vec![callee.clone(), callee]);

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1 = stablehlo.custom_call @ryft.print(%arg0, %0) {api_version = 4 : i32, backend_config = {label = "callee"}, has_side_effect = true} : (tensor<4xf32>, !stablehlo.token) -> !stablehlo.token
                    %2 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    %3 = stablehlo.custom_call @ryft.print(%arg0, %1) {api_version = 4 : i32, backend_config = {label = "callee"}, has_side_effect = true} : (tensor<4xf32>, !stablehlo.token) -> !stablehlo.token
                    %4 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    %5 = stablehlo.add %2, %4 : tensor<4xf32>
                    return %5 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_jit_two_prints_execute_in_order_on_cpu() {
        use ryft_core::operations::debugging::Print;
        use ryft_core::sharding::{Device, DeviceMesh};
        use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};

        use crate::experimental::debugging::{ensure_print_handler_registered, with_captured_prints};
        use crate::experimental::domains::XlaDomain;
        use crate::tests::{values_from_bytes, values_to_bytes};
        use crate::{Array, CompiledXlaFunction, FromPjrt, compile};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        ensure_print_handler_registered(&client).unwrap();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![device],
        )
        .unwrap();
        let engine = XlaDomain::new(&client);

        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> = compile(
            |x| {
                let x = x.print("x");
                (x.clone() + x).print("doubled")
            },
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        let values = [1.5f64, 2.5];
        let source =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f64>(&values).as_slice())
                .unwrap();
        let (output, lines) =
            with_captured_prints(|| engine.interpret(&compiled.executable_program(), source).unwrap());

        // Both prints fire in program order, and the printed values are the forwarded operands.
        assert_eq!(lines, vec!["x: [1.5, 2.5]".to_string(), "doubled: [3.0, 5.0]".to_string()]);

        // The compiled function still computes its dataflow output (each print is the identity on its operand).
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let output_bytes = output
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(values_from_bytes::<f64>(output_bytes.as_slice()), vec![3.0, 5.0]);
    }

    #[test]
    fn test_scan_body_print_with_token_loop_state_executes_on_cpu() {
        use std::sync::Arc;

        use ryft_core::operations::control_flow::ScanOperation as CoreScanOperation;
        use ryft_core::operations::debugging::PrintOperation;
        use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
        use ryft_pjrt::{
            BufferType, ClientOptions, CpuClientOptions, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram,
            load_cpu_plugin,
        };

        use crate::experimental::debugging::{ensure_print_handler_registered, with_captured_prints};
        use crate::tests::{values_from_bytes, values_to_bytes};

        // Executes the token-in-loop-state lowering on the real CPU plugin: the effect token is a
        // `stablehlo.while` carry and each iteration's print custom call consumes and produces it, so this pins that
        // XLA accepts and runs token-carrying loops (not just the flat token chain).
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut body_builder = XlaProgramBuilder::new();
        let carry = body_builder.add_input(scalar_f64.clone());
        let x = body_builder.add_input(scalar_f64.clone());
        let printed = body_builder.add_instruction(PrintOperation::new("iteration"), vec![x]).unwrap()[0];
        let sum = body_builder.add_instruction(AddOperation, vec![carry, printed]).unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        let scan = CoreScanOperation::<_, _>::new(body, 1, 3).unwrap();

        let stacked_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let mut builder = XlaProgramBuilder::new();
        let init = builder.add_input(scalar_f64.clone());
        let stacked_inputs = builder.add_input(stacked_type.clone());
        let output =
            builder.add_instruction(XlaOperation::Scan(Box::new(scan)), vec![init, stacked_inputs]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let input_types = vec![scalar_f64.clone(), stacked_type];
        let output_types = vec![scalar_f64];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        ensure_print_handler_registered(&client).unwrap();
        let options = CompilationOptions {
            argument_layouts: Vec::new(),
            parameter_is_tupled_arguments: false,
            executable_build_options: Some(ExecutableCompilationOptions {
                device_ordinal: -1,
                replica_count: 1,
                partition_count: 1,
                ..Default::default()
            }),
            compile_portable_executable: false,
            profile_version: 0,
            serialized_multi_slice_configuration: Vec::new(),
            environment_option_overrides: std::collections::HashMap::new(),
            target_config: None,
            allow_in_place_mlir_modification: false,
            matrix_unit_operand_precision: Precision::Default as i32,
        };
        let executable = client.compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &options).unwrap();
        let device = executable.addressable_devices().unwrap()[0].clone();
        let init_bytes = values_to_bytes::<f64>(&[1.0]);
        let stacked_bytes = values_to_bytes::<f64>(&[1.0, 2.0, 3.0]);

        let ((), lines) = with_captured_prints(|| {
            let inputs = ExecutionDeviceInputs {
                inputs: &[
                    ExecutionInput {
                        buffer: Arc::new(
                            client
                                .buffer(init_bytes.as_slice(), BufferType::F64, &[], None, device.clone(), None)
                                .unwrap(),
                        ),
                        donatable: false,
                    },
                    ExecutionInput {
                        buffer: Arc::new(
                            client
                                .buffer(stacked_bytes.as_slice(), BufferType::F64, &[3], None, device.clone(), None)
                                .unwrap(),
                        ),
                        donatable: false,
                    },
                ],
                ..Default::default()
            };
            let execution = executable.execute(vec![inputs], Vec::new(), 0, None, None, None, None).unwrap();
            let mut outputs = execution.block_until_ready().unwrap().remove(0);
            assert_eq!(outputs.outputs.len(), 1);
            let output_bytes = outputs.outputs.remove(0).copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(values_from_bytes::<f64>(output_bytes.as_slice()), vec![7.0]);
        });

        // One print per scan iteration, in iteration order.
        assert_eq!(
            lines,
            vec!["iteration: 1.0".to_string(), "iteration: 2.0".to_string(), "iteration: 3.0".to_string()],
        );
    }

    #[test]
    fn test_batched_predicate_while_executes_with_masked_semantics_on_cpu() {
        use std::sync::Arc;

        use ryft_core::operations::compare::CompareOperation;
        use ryft_core::operations::constants::{OneLikeOperation, ZeroLikeOperation};
        use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
        use ryft_pjrt::{
            BufferType, ClientOptions, CpuClientOptions, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram,
            load_cpu_plugin,
        };

        use crate::tests::{values_from_bytes, values_to_bytes};

        // End-to-end proof of the threaded-predicate lowering on the real CPU plugin: a per-item countdown
        // `while (x > 0) { x - 1 }` over `f64[3]` carries the batched `bool[3]` predicate through the loop state, so
        // the condition is evaluated once per iteration (seeded before the loop, recomputed in the body) rather than
        // twice. Items [3, 1, 2] terminate after 3, 1, and 2 iterations and all land at 0; a masking bug (or a
        // rejoining finished item) would leave nonzero or negative entries.
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let condition = {
            let mut builder = XlaProgramBuilder::new();
            let state = builder.add_input(state_type.clone());
            let zero = builder.add_instruction(ZeroLikeOperation, vec![state]).unwrap()[0];
            let predicate = builder
                .add_instruction(CompareOperation::new(ComparisonDirection::GreaterThan), vec![state, zero])
                .unwrap()[0];
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![predicate], vec![Placeholder], vec![Placeholder])
        }
        .unwrap();
        let body = {
            let mut builder = XlaProgramBuilder::new();
            let state = builder.add_input(state_type.clone());
            let one = builder.add_instruction(OneLikeOperation, vec![state]).unwrap()[0];
            let next = builder.add_instruction(SubOperation, vec![state, one]).unwrap()[0];
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![next], vec![Placeholder], vec![Placeholder])
        }
        .unwrap();
        let while_operation = WhileOperation::new(condition, body).unwrap();
        let mut builder = XlaProgramBuilder::new();
        let state = builder.add_input(state_type.clone());
        let output = builder.add_instruction(XlaOperation::While(Box::new(while_operation)), vec![state]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let input_types = vec![state_type.clone()];
        let output_types = vec![state_type.clone()];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let options = CompilationOptions {
            argument_layouts: Vec::new(),
            parameter_is_tupled_arguments: false,
            executable_build_options: Some(ExecutableCompilationOptions {
                device_ordinal: -1,
                replica_count: 1,
                partition_count: 1,
                ..Default::default()
            }),
            compile_portable_executable: false,
            profile_version: 0,
            serialized_multi_slice_configuration: Vec::new(),
            environment_option_overrides: std::collections::HashMap::new(),
            target_config: None,
            allow_in_place_mlir_modification: false,
            matrix_unit_operand_precision: Precision::Default as i32,
        };
        let executable = client.compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &options).unwrap();
        let device = executable.addressable_devices().unwrap()[0].clone();
        let input_bytes = values_to_bytes::<f64>(&[3.0, 1.0, 2.0]);
        let inputs = ExecutionDeviceInputs {
            inputs: &[ExecutionInput {
                buffer: Arc::new(
                    client.buffer(input_bytes.as_slice(), BufferType::F64, &[3], None, device.clone(), None).unwrap(),
                ),
                donatable: false,
            }],
            ..Default::default()
        };
        let execution = executable.execute(vec![inputs], Vec::new(), 0, None, None, None, None).unwrap();
        let mut outputs = execution.block_until_ready().unwrap().remove(0);
        assert_eq!(outputs.outputs.len(), 1);
        let output_bytes = outputs.outputs.remove(0).copy_to_host(None).unwrap().r#await().unwrap();
        assert_eq!(values_from_bytes::<f64>(output_bytes.as_slice()), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_grad_of_jitted_print_function_prints_once_on_cpu() {
        use ryft_core::operations::debugging::Print;
        use ryft_core::sharding::{Device, DeviceMesh};
        use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};

        use crate::experimental::debugging::{ensure_print_handler_registered, with_captured_prints};
        use crate::experimental::domains::XlaDomain;
        use crate::tests::{values_from_bytes, values_to_bytes};
        use crate::{Array, CompiledXlaFunction, FromPjrt, compile};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        ensure_print_handler_registered(&client).unwrap();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![device],
        )
        .unwrap();
        let engine = XlaDomain::new(&client);

        let input_type = ArrayType::new(DataType::F64, Shape::new(Vec::new()))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let compiled: CompiledXlaFunction<'_, ArrayType, ArrayType> =
            compile(|x| (x.clone() * x).print("y"), input_type.clone(), &engine, mesh.clone()).unwrap();
        let gradient: CompiledXlaFunction<'_, ArrayType, ArrayType> = compiled.gradient(&engine).unwrap();

        let input =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes::<f64>(&[3.0]).as_slice())
                .unwrap();
        let (output, lines) = with_captured_prints(|| engine.interpret(&gradient.executable_program(), input).unwrap());

        // The print fires exactly once, during the forward pass of the differentiated program; transposition is the
        // identity on the cotangent and re-prints nothing.
        assert_eq!(lines, vec!["y: 9.0".to_string()]);

        // The gradient of `x * x` at `x = 3` is `6`.
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let output_bytes = output
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(values_from_bytes::<f64>(output_bytes.as_slice()), vec![6.0]);
    }

    // ---------------------------------------------------------------------------
    // Plain-program StableHLO lowering tests for scalar programs
    // ---------------------------------------------------------------------------

    fn scalar_bilinear_sin<T>(inputs: (T, T)) -> T
    where
        T: Clone + ryft_core::operations::trigonometric::Sin + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    {
        inputs.0.clone() * inputs.1 + inputs.0.sin().unwrap()
    }

    fn scalar_quartic_plus_sin<T>(x: T) -> T
    where
        T: Clone + ryft_core::operations::trigonometric::Sin + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    {
        x.clone() * x.clone() * x.clone() * x.clone() + x.sin().unwrap()
    }

    static TEST_ARRAY_DOMAIN: EagerContext<TestArray, ArrayOperation<TestArray>> =
        EagerContext::<TestArray, ArrayOperation<TestArray>>::new();

    #[test]
    fn test_plain_scalar_bilinear_sin_jit_stablehlo() {
        let (_, compiled): (
            TestArray,
            ryft_core::programs::Program<
                TestArray,
                ryft_core::tracing_v2::ArrayOperation<TestArray>,
                (TestArray, TestArray),
                TestArray,
            >,
        ) = TEST_ARRAY_DOMAIN
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
        let (_, compiled): (
            TestArray,
            ryft_core::programs::Program<
                TestArray,
                ryft_core::tracing_v2::ArrayOperation<TestArray>,
                TestArray,
                TestArray,
            >,
        ) = TEST_ARRAY_DOMAIN
            .interpret_and_trace(
                |x| {
                    let context = x.context().clone();
                    Ok(context.gradient(scalar_quartic_plus_sin, x).expect("scalar gradient should succeed"))
                },
                TestArray::scalar(2.0),
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
    fn test_to_mlir_module_for_plain_program_lowers_slicing_operations() {
        let input_type = test_matrix_type(2, 3);
        let update_type = test_matrix_type(1, 2);
        let index_type = ArrayType::scalar(DataType::I32);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let update = builder.add_input(update_type);
        let index_0 = builder.add_input(index_type.clone());
        let index_1 = builder.add_input(index_type);
        let sliced = builder
            .add_instruction(SliceOperation::new(vec![1, 1], vec![2, 3]).with_strides(vec![1, 1]).unwrap(), vec![input])
            .unwrap()[0];
        let updated = builder.add_instruction(UpdateSliceOperation::new(vec![0, 1]), vec![input, update]).unwrap()[0];
        let dynamic_sliced = builder
            .add_instruction(DynamicSliceOperation::new(vec![1, 2]), vec![input, index_0, index_1])
            .unwrap()[0];
        let dynamic_updated =
            builder.add_instruction(DynamicUpdateSliceOperation, vec![input, update, index_0, index_1]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![sliced, updated, dynamic_sliced, dynamic_updated],
                vec![Placeholder, Placeholder, Placeholder, Placeholder],
                vec![Placeholder, Placeholder, Placeholder, Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<1x2xf32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<1x2xf32>, tensor<2x3xf32>, tensor<1x2xf32>, tensor<2x3xf32>) {
                    %0 = stablehlo.slice %arg0 [1:2, 1:3] : (tensor<2x3xf32>) -> tensor<1x2xf32>
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %c_0 = stablehlo.constant dense<1> : tensor<i64>
                    %1 = stablehlo.dynamic_update_slice %arg0, %arg1, %c, %c_0 : (tensor<2x3xf32>, tensor<1x2xf32>, tensor<i64>, tensor<i64>) -> tensor<2x3xf32>
                    %2 = stablehlo.dynamic_slice %arg0, %arg2, %arg3, sizes = [1, 2] : (tensor<2x3xf32>, tensor<i32>, tensor<i32>) -> tensor<1x2xf32>
                    %3 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3 : (tensor<2x3xf32>, tensor<1x2xf32>, tensor<i32>, tensor<i32>) -> tensor<2x3xf32>
                    return %0, %1, %2, %3 : tensor<1x2xf32>, tensor<2x3xf32>, tensor<1x2xf32>, tensor<2x3xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_concatenate() {
        // A static-shaped concatenate along axis 0 lowers to a single `stablehlo.concatenate` joining the operands.
        let first_type = test_matrix_type(1, 2);
        let second_type = test_matrix_type(3, 2);
        let mut builder = XlaProgramBuilder::new();
        let first = builder.add_input(first_type);
        let second = builder.add_input(second_type);
        let joined = builder.add_instruction(ConcatenateOperation::new(0), vec![first, second]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, XlaConstant>(vec![joined], vec![Placeholder, Placeholder], Placeholder)
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<1x2xf32>, %arg1: tensor<3x2xf32>) -> tensor<4x2xf32> {
                    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<1x2xf32>, tensor<3x2xf32>) -> tensor<4x2xf32>
                    return %0 : tensor<4x2xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_strided_slice_and_pad() {
        let vector_type = test_vector_type(6);
        let pad_input_type = test_vector_type(3);
        let padding_value_type = ArrayType::scalar(DataType::F32);
        let mut builder = XlaProgramBuilder::new();
        let vector = builder.add_input(vector_type);
        let pad_input = builder.add_input(pad_input_type);
        let padding_value = builder.add_input(padding_value_type);
        let strided = builder
            .add_instruction(SliceOperation::new(vec![1], vec![6]).with_strides(vec![2]).unwrap(), vec![vector])
            .unwrap()[0];
        let padded = builder
            .add_instruction(PadOperation::new(vec![1], vec![2], vec![1]).unwrap(), vec![pad_input, padding_value])
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![strided, padded],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<6xf32>, %arg1: tensor<3xf32>, %arg2: tensor<f32>) -> (tensor<3xf32>, tensor<8xf32>) {
                    %0 = stablehlo.slice %arg0 [1:6:2] : (tensor<6xf32>) -> tensor<3xf32>
                    %1 = stablehlo.pad %arg1, %arg2, low = [1], high = [2], interior = [1] : (tensor<3xf32>, tensor<f32>) -> tensor<8xf32>
                    return %0, %1 : tensor<3xf32>, tensor<8xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_slicing_vjp_pullbacks_lower_to_stablehlo() {
        use ryft_core::operations::manipulation::{DynamicSlice, Slice};

        // The static slice pullback writes the cotangent into a zero array at the static offsets via the
        // statically indexed update-slice, which lowers to `stablehlo.dynamic_update_slice` with constant indices.
        // The structural-zero destination is emitted as a `ZeroOperation` instruction in the pullback, which lowers
        // through the canonical zero path to a scalar constant broadcast to the array shape. The reverse path
        // stages the pullback over the primal operation family taking `[output_cotangents ++ residuals]`; this slice
        // pullback captures no residuals, so the pullback consumes only the single output cotangent.
        let (_, pullback): (TestArray, _) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .vjp(|x| Ok(x.slice(&[1], &[3], &[1]).unwrap()), TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]))
            .unwrap();
        let (pullback, _residuals) = pullback.into_parts();
        let stablehlo = to_mlir_module_for_plain_program(&pullback, "main").unwrap();
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xf64>) -> tensor<4xf64> {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64>
                    %c = stablehlo.constant dense<1> : tensor<i64>
                    %1 = stablehlo.dynamic_update_slice %0, %arg0, %c : (tensor<4xf64>, tensor<2xf64>, tensor<i64>) -> tensor<4xf64>
                    return %1 : tensor<4xf64>
                  }
                }
            "#}
        );

        // The strided slice pullback pads the cotangent with a zero scalar at the inverse geometry
        // (`low = start`, `interior = stride - 1`), which lowers to `stablehlo.pad`.
        let (_, pullback): (TestArray, _) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .vjp(|x| Ok(x.slice(&[1], &[6], &[2]).unwrap()), TestArray::vector(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .unwrap();
        let (pullback, _residuals) = pullback.into_parts();
        let stablehlo = to_mlir_module_for_plain_program(&pullback, "main").unwrap();
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<3xf64>) -> tensor<6xf64> {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %0 = stablehlo.pad %arg0, %cst, low = [1], high = [0], interior = [1] : (tensor<3xf64>, tensor<f64>) -> tensor<6xf64>
                    return %0 : tensor<6xf64>
                  }
                }
            "#}
        );

        // The pad pullback splits the cotangent into the strided slice at the pad geometry (for the input) and the
        // full-sum-minus-sliced-sum subtraction (for the padding value), all of which lower to StableHLO.
        let (_, pullback): (TestArray, _) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .vjp(
                |(x, padding_value)| {
                    use ryft_core::operations::manipulation::Pad;
                    Ok(x.pad(&padding_value, &[1], &[2], &[1]).unwrap())
                },
                (TestArray::vector(vec![1.0, 2.0, 3.0]), TestArray::scalar(9.0)),
            )
            .unwrap();
        let (pullback, _residuals) = pullback.into_parts();
        let stablehlo = to_mlir_module_for_plain_program(&pullback, "main").unwrap();
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<8xf64>) -> (tensor<3xf64>, tensor<f64>) {
                    %0 = stablehlo.slice %arg0 [1:6:2] : (tensor<8xf64>) -> tensor<3xf64>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<8xf64>, tensor<f64>) -> tensor<f64>
                    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %2 = stablehlo.reduce(%0 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
                    %3 = stablehlo.subtract %1, %2 : tensor<f64>
                    return %0, %3 : tensor<3xf64>, tensor<f64>
                  }
                }
            "#}
        );

        // The dynamic slice pullback scatters the cotangent at the captured index factors, which materialize as
        // integer constants through `lower_literal_value`.
        let (_, pullback): (TestArray, _) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .vjp(
                |x| {
                    let start = x.context().lift(TestArray::new(ArrayType::scalar(DataType::I32), vec![1.0])).unwrap();
                    Ok(x.dynamic_slice(&[start], &[2]).unwrap())
                },
                TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]),
            )
            .unwrap();
        let (pullback, _residuals) = pullback.into_parts();
        let stablehlo = to_mlir_module_for_plain_program(&pullback, "main").unwrap();
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xf64>) -> tensor<4xf64> {
                    %c = stablehlo.constant dense<1> : tensor<i32>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64>
                    %1 = stablehlo.dynamic_update_slice %0, %arg0, %c : (tensor<4xf64>, tensor<2xf64>, tensor<i32>) -> tensor<4xf64>
                    return %1 : tensor<4xf64>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_plain_scalar_bilinear_sin_vjp_pullback_standalone_stablehlo() {
        // Standalone pullback over the primal operation family, produced by the partition-aware reverse path.
        // The reverse program of `f(x, y) = x * y + sin(x)` consumes `[output_cotangent ++ residuals]`, where the
        // residuals are the linearization-point factors `cos(x)`, `x`, and `y`, and lowers the residual-weighted
        // backward pass as `stablehlo.multiply`s of the cotangent against those residual inputs rather than baking the
        // primal point in as constants — the analogue of JAX's standalone `vjp_fn`, with the residuals threaded as
        // explicit arguments.
        let (_, pullback): (TestArray, _) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .vjp(|inputs| Ok(scalar_bilinear_sin(inputs)), (TestArray::scalar(2.0), TestArray::scalar(3.0)))
            .unwrap();
        let (pullback, _residuals) = pullback.into_parts();

        let stablehlo = to_mlir_module_for_plain_program(&pullback, "main").unwrap();
        println!("=== ryft standalone vjp_pullback(x*y + sin(x)) StableHLO ===\n{stablehlo}");

        // The pullback takes the output cotangent plus three residual inputs and returns two cotangent outputs (for x
        // and y).
        assert!(
            stablehlo.contains(
                "func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>) -> \
                 (tensor<f64>, tensor<f64>)",
            ),
            "pullback should take a cotangent plus residual inputs and return two outputs, but got:\n{stablehlo}",
        );
        // The residual-weighted reverse multiplies the cotangent by the residual inputs rather than baking the primal
        // point in as constants, so the lowering is multiplies-and-adds with no `stablehlo.constant`.
        assert!(stablehlo.contains("stablehlo.multiply"), "residual-weighted reverse should multiply: \n{stablehlo}");
        assert!(
            !stablehlo.contains("stablehlo.constant"),
            "residuals are runtime inputs, not baked-in constants, but got:\n{stablehlo}",
        );
    }

    #[test]
    fn test_rematerialized_vjp_pullback_lowers_without_a_rematerialization_boundary() {
        use ryft_core::tracing_v2::rematerialize;

        // The value-level `vjp` runs on the partition-aware reverse path, which splices the rematerialized
        // region's recompute-and-pushforward into the program as ordinary straight-line primal operations and
        // transposes them like any other computation. The rematerialization boundary is purely a forward-pass memory
        // tradeoff with no effect on the differentiated result, so the resulting pullback carries no
        // `stablehlo.optimization_barrier`: it is the same residual-weighted backward pass as the un-rematerialized
        // body. The `prevent_cse` optimization-barrier hint applies to the forward/JVP lowering of a retained
        // `RematerializeOperation`, not to this reverse pullback, so toggling it leaves the pullback unchanged. For
        // `f(x) = sin(x · x)` the pullback consumes `[output_cotangent ++ residuals]` (residuals `cos(x²)` and `x`)
        // and lowers to the symmetric `d(x · x) = 2x · dx` transpose — two `stablehlo.multiply` branches summed —
        // scaled by `cos(x²)`.
        let expected = indoc! {r#"
            module {
              func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<f64> {
                %0 = stablehlo.multiply %arg2, %arg0 : tensor<f64>
                %1 = stablehlo.multiply %arg1, %0 : tensor<f64>
                %2 = stablehlo.multiply %arg1, %0 : tensor<f64>
                %3 = stablehlo.add %1, %2 : tensor<f64>
                return %3 : tensor<f64>
              }
            }
        "#};

        let function = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(
            |x: ryft_core::tracing::DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| {
                Ok((x.clone() * x).sin()?)
            },
        );
        let (_, pullback): (TestArray, _) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .vjp(|x| function.call(x), TestArray::scalar(2.0))
            .unwrap();
        let (pullback, _residuals) = pullback.into_parts();
        let stablehlo = to_mlir_module_for_plain_program(&pullback, "main").unwrap();
        assert!(
            !stablehlo.contains("stablehlo.optimization_barrier"),
            "the reverse path strips the rematerialization boundary, so the pullback should lower without an \
             optimization barrier, but got:\n{stablehlo}",
        );
        assert_eq!(stablehlo, expected);

        // Disabling `prevent_cse` changes nothing about the reverse pullback: the hint only affects forward lowering.
        let function = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(
            |x: ryft_core::tracing::DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| {
                Ok((x.clone() * x).sin()?)
            },
        )
        .with_prevent_cse(false);
        let (_, pullback): (TestArray, _) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .vjp(|x| function.call(x), TestArray::scalar(2.0))
            .unwrap();
        let (pullback, _residuals) = pullback.into_parts();
        let stablehlo = to_mlir_module_for_plain_program(&pullback, "main").unwrap();
        assert_eq!(stablehlo, expected, "the reverse pullback is independent of the prevent_cse hint");
    }

    #[test]
    fn test_transfer_to_memory_lowers_to_device_placement_annotations() {
        use ryft_core::tracing_v2::operations::TransferToMemory;

        // A compute-flanked host-and-back round trip lowers to one `annotate_device_placement` custom call per
        // transfer, carrying the destination kind in the `_xla_buffer_placement` frontend attribute — including the
        // identity-looking transfer back to device memory, which `HostOffloader` needs to see. The program mirrors
        // the JAX example in `python/scripts/dump_transfer_to_memory_mlir_from_jax.py`, and the asserted custom
        // calls are byte-identical to the ones JAX emits for it.
        let (_, program) = EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(
            |x: ryft_core::tracing::DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| {
                let y = x.clone() * x;
                let on_host = y.transfer_to_memory(Memory::Host { pinned: true });
                let back = on_host.transfer_to_memory(Memory::Device);
                Ok(back.clone() * back)
            },
            test_vector_type(4),
        )
        .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
        println!("=== ryft transfer_to_memory StableHLO ===\n{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.custom_call @annotate_device_placement").count(), 2, "{stablehlo}");
        assert!(
            stablehlo.contains(
                "stablehlo.custom_call @annotate_device_placement(%0) {backend_config = \"\", has_side_effect = \
                 true, mhlo.frontend_attributes = {_xla_buffer_placement = \"pinned_host\"}} : (tensor<4xf32>) -> \
                 tensor<4xf32>",
            ),
            "{stablehlo}",
        );
        assert!(
            stablehlo.contains(
                "stablehlo.custom_call @annotate_device_placement(%1) {backend_config = \"\", has_side_effect = \
                 true, mhlo.frontend_attributes = {_xla_buffer_placement = \"device\"}} : (tensor<4xf32>) -> \
                 tensor<4xf32>",
            ),
            "{stablehlo}",
        );
    }

    #[test]
    fn test_transfer_to_memory_vjp_pullback_lowers_with_a_placement_annotation() {
        use ryft_core::tracing_v2::operations::TransferToMemory;

        // The pullback of a transfer moves the cotangent back to the operand's source memory (the default device
        // space here), so it lowers to an `annotate_device_placement` custom call targeting `device`.
        let (_, pullback): (TestArray, _) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .vjp(|x| Ok(x.transfer_to_memory(Memory::Host { pinned: true })), TestArray::scalar(2.0))
            .unwrap();
        let (pullback, _residuals) = pullback.into_parts();
        let stablehlo = to_mlir_module_for_plain_program(&pullback, "main").unwrap();
        assert_eq!(stablehlo.matches("stablehlo.custom_call @annotate_device_placement").count(), 1, "{stablehlo}");
        assert!(stablehlo.contains("_xla_buffer_placement = \"device\""), "{stablehlo}");
    }

    #[test]
    fn test_plain_scalar_bilinear_sin_grad_jitted_stablehlo() {
        // grad(f) wrapped in JIT â€” symbolic, like JAX's jit(grad(f)).
        // Uses the traced value-and-gradient path that traces through vjp+pullback.
        let (_, compiled): (
            (TestArray, TestArray),
            ryft_core::programs::Program<
                TestArray,
                ryft_core::tracing_v2::ArrayOperation<TestArray>,
                (TestArray, TestArray),
                (TestArray, TestArray),
            >,
        ) = TEST_ARRAY_DOMAIN
            .interpret_and_trace(
                |inputs| {
                    let context = inputs.0.context().clone();
                    Ok(context.gradient(scalar_bilinear_sin, inputs).expect("scalar gradient should succeed"))
                },
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
}
