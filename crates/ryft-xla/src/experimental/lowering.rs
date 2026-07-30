use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use ryft_core::axes::AxisIndexOperation;
use ryft_core::backends::arrays::Array as CpuArray;
use ryft_core::backends::arrays::ArrayOperation;
use ryft_core::backends::scalars::Scalar;
use ryft_core::macros::check_count;
use ryft_core::operations::attention::{
    AttentionMask, DotProductAttentionBackwardOperation, DotProductAttentionOperation,
};
use ryft_core::operations::collectives::{
    AllGatherOperation, AllToAllOperation, CollectiveKind, CollectiveMode, CollectiveOperation, PSumScatterOperation,
    PpermuteOperation,
};
use ryft_core::operations::compare::ComparisonDirection;
use ryft_core::operations::complex::{ComplexOperation, ConjugateOperation, ImaginaryOperation, RealOperation};
use ryft_core::operations::constants::{ConstantOperation, IotaOperation};
use ryft_core::operations::control_flow::{ConditionOperation, ScanOperation, WhileOperation};
use ryft_core::operations::custom_call::{CustomCallAttribute, CustomCallOperation};
use ryft_core::operations::differentiation::CoordinateBasisOperation;
#[cfg(test)]
use ryft_core::operations::manipulation::ReshapeParameters;
use ryft_core::operations::manipulation::{
    ConvertElementType, ConvertElementTypeOperation, GatherOperation, GatherScatterMode, LegacyBroadcastOperation,
    LegacyReshapeOperation, PadOperation, ReshapeDimensionExpression, ScatterOperation, ScatterReductionKind,
    SliceOperation, TransposeOperation,
};
use ryft_core::operations::math::{
    AbsOperation, AddOperation, Atan2Operation, CeilOperation, CosOperation, DivOperation, DotOperation, ErfOperation,
    ExpOperation, FloorOperation, LogOperation, LogisticOperation, MaxOperation, MinOperation, MulOperation,
    NegOperation, PowOperation, ReductionKind, RemOperation, RoundOperation, RsqrtOperation, ScaledDotOperation,
    SignOperation, SinOperation, SqrtOperation, SubOperation, TanhOperation,
};
use ryft_core::operations::random::{RandomAlgorithm, RngBitGeneratorOperation};
use ryft_core::operations::sort::{SortDirection, SortOperation};
use ryft_core::parameters::Parameterized;
use ryft_core::programs::operations::Operation;
use ryft_core::programs::regions::{RegionId, RegionRef};
use ryft_core::programs::types::{Type as RyftType, Typed};
use ryft_core::programs::{AtomId, Instruction, Program, ProgramError, Value};
use ryft_core::sharding::{LogicalMesh, MeshAxisType, Sharding, ShardingDimension, ShardingError};
use ryft_core::types::{ArrayType, DataType, Dimension, Memory, Shape};
use ryft_mlir::dialects::stable_hlo::{Accuracy, CustomCallApiVersion, CustomCallMemoryLayouts, Precision};
use ryft_mlir::dialects::{chlo, func, shardy, stable_hlo, tensor};
use ryft_mlir::{
    Attribute, Block, BlockRef, Context as MlirContext, DenseElementsAttributeRef, FloatTypeRef, IntegerTypeRef,
    Location, LocationRef, Operation as MlirOperation, Region, Size as MlirSize, SymbolVisibility, TensorTypeRef, Type,
    TypeAndAttributes, TypeRef, Value as MlirValue, ValueAndAttributes, ValueRef,
};

use crate::experimental::debugging::{PRINT_CUSTOM_CALL_TARGET, PRINT_LABEL_ATTRIBUTE};
#[cfg(test)]
use crate::experimental::ops::XlaProgramBuilder;
use crate::experimental::ops::{FlatXlaProgram, XlaArrayConstant, XlaConstant, XlaOperation, XlaProgram};
use crate::mlir::ToMlir;

use super::shard_map::{ShardMap, ShardMapError};

mod composite;
pub use composite::{ArrayProgramLoweringError, lower_array_program_to_stable_hlo};

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

    /// Error returned when a reshape dimension cannot be represented by StableHLO's signed shape element type.
    #[error("reshape dimension {value} cannot be represented as a StableHLO i{bit_width} shape value")]
    ReshapeDimensionOutOfRange { value: usize, bit_width: u8 },

    /// Error returned when a pad interior amount cannot be represented by StableHLO's signed attribute type.
    #[error("pad interior padding {value} cannot be represented as a StableHLO i64 attribute")]
    PadInteriorPaddingOutOfRange { value: usize },

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

/// Logical-to-physical argument and result mapping of one XLA executable boundary.
///
/// Ryft retains every logical leaf in its staged and compiled function metadata. Statically shaped
/// [`DataType::Zero`] leaves carry no runtime payload and are omitted from the StableHLO/PJRT signature; each mapping
/// stores the corresponding physical ordinal for materialized leaves and `None` for erased leaves. Dynamically shaped
/// zero-space leaves deliberately retain private `i1` carriers because the current executable ABI has no independent
/// shape transport. Current XLA compilation may impose stricter limits on dynamic tensors than this projection layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct XlaExecutableSignature {
    /// Physical argument ordinal for each logical flattened input.
    input_mapping: Arc<[Option<usize>]>,

    /// Physical result ordinal for each logical flattened output.
    output_mapping: Arc<[Option<usize>]>,
}

impl XlaExecutableSignature {
    /// Derives the executable boundary mapping for the provided logical input and output types.
    pub(crate) fn new(input_types: &[ArrayType], output_types: &[ArrayType]) -> Self {
        fn mapping(types: &[ArrayType]) -> Arc<[Option<usize>]> {
            let mut physical_index = 0;
            types
                .iter()
                .map(|r#type| {
                    if r#type.data_type().is_zero() && r#type.static_shape().is_some() {
                        None
                    } else {
                        let index = physical_index;
                        physical_index += 1;
                        Some(index)
                    }
                })
                .collect()
        }
        Self { input_mapping: mapping(input_types), output_mapping: mapping(output_types) }
    }

    /// Returns the per-logical-input physical argument mapping.
    #[inline]
    pub(crate) fn input_mapping(&self) -> &[Option<usize>] {
        &self.input_mapping
    }

    /// Returns the per-logical-output physical result mapping.
    #[inline]
    pub(crate) fn output_mapping(&self) -> &[Option<usize>] {
        &self.output_mapping
    }

    /// Projects logical inputs into physical executable order.
    pub(crate) fn project_inputs<T: Clone>(&self, inputs: &[T]) -> Vec<T> {
        Self::project(&self.input_mapping, inputs)
    }

    /// Projects logical outputs into physical executable order.
    pub(crate) fn project_outputs<T: Clone>(&self, outputs: &[T]) -> Vec<T> {
        Self::project(&self.output_mapping, outputs)
    }

    /// Projects one logical sequence according to `mapping`.
    fn project<T: Clone>(mapping: &[Option<usize>], values: &[T]) -> Vec<T> {
        assert_eq!(mapping.len(), values.len());
        mapping
            .iter()
            .zip(values)
            .filter_map(|(physical_index, value)| physical_index.map(|_| value.clone()))
            .collect()
    }
}

/// Textual StableHLO module paired with the exact executable signature used to emit its entry function.
pub(crate) struct LoweredXlaModule {
    /// Textual StableHLO/Shardy module.
    stable_hlo: String,

    /// Logical-to-physical entry-function signature.
    signature: XlaExecutableSignature,
}

impl LoweredXlaModule {
    /// Consumes this lowering and returns its textual module and executable signature.
    #[inline]
    pub(crate) fn into_parts(self) -> (String, XlaExecutableSignature) {
        (self.stable_hlo, self.signature)
    }
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

    /// Declared input types of the instruction currently being lowered, in operand order.
    input_types: Vec<ArrayType>,

    /// Shared private functions emitted for deduplicated `jit_call` callees, consulted at `jit_call` lowering sites.
    /// Shared via [`Rc`] so it threads through nested lowering scopes without lifetime entanglement.
    nested_functions: Option<Rc<JitCallFunctionMap>>,

    /// Hidden capture arguments of the function currently being lowered, in capture-table order.
    captured_values: Vec<ValueRef<'b, 'c, 't>>,

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
            input_types: Vec::new(),
            nested_functions: None,
            captured_values: Vec::new(),
            token: None,
            collective_state: CollectiveLoweringState::new(),
        }
    }

    /// Attaches the declared input types of the instruction currently being lowered.
    pub(crate) fn with_input_types(mut self, input_types: Vec<ArrayType>) -> Self {
        self.input_types = input_types;
        self
    }

    /// Attaches the shared deduplicated `jit_call` functions consulted while lowering.
    pub(crate) fn with_nested_functions(mut self, nested_functions: Option<Rc<JitCallFunctionMap>>) -> Self {
        self.nested_functions = nested_functions;
        self
    }

    /// Attaches the hidden capture arguments of the function currently being lowered.
    pub(crate) fn with_captured_values(mut self, captured_values: &[ValueRef<'b, 'c, 't>]) -> Self {
        self.captured_values = captured_values.to_vec();
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
    pub(crate) fn lower_condition<V: MlirLowerableValue>(
        &mut self,
        branch_regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_condition_to_if(
            branch_regions,
            input_values,
            &mut self.block,
            self.context,
            self.location,
            self.captured_values.as_slice(),
            self.nested_functions.as_ref(),
            &self.collective_state,
            &mut self.token,
        )
    }

    /// Lowers one nested while operation inside this lowering context.
    pub(crate) fn lower_while<V: MlirLowerableValue>(
        &mut self,
        while_op: &WhileOperation,
        loop_regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_while_to_while(
            while_op,
            loop_regions,
            input_values,
            &mut self.block,
            self.context,
            self.location,
            self.captured_values.as_slice(),
            self.nested_functions.as_ref(),
            &self.collective_state,
            &mut self.token,
        )
    }

    /// Lowers one nested scan operation inside this lowering context.
    pub(crate) fn lower_scan<V: MlirLowerableValue, Capture>(
        &mut self,
        scan_op: &ScanOperation<Capture>,
        scan_regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        Capture: Value<Type = ArrayType>,
    {
        let [body] = scan_regions else {
            return Err(LoweringError::UnsupportedOp {
                op: format!("scan expected 1 attached region but got {}", scan_regions.len()),
            });
        };
        lower_scan_to_while(
            body,
            scan_op.carry_count(),
            scan_op.length(),
            scan_op.reverse(),
            scan_op.unroll(),
            input_values,
            &mut self.block,
            self.context,
            self.location,
            self.captured_values.as_slice(),
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
pub(crate) trait LowerableXlaOperation<V: MlirLowerableValue>: Operation<ArrayType> {
    /// Lowers this operation to one or more StableHLO operations.
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>;
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for ConvertElementTypeOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("input", input_values, 1, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        let output_type = lower_tensor_type(&output_types[0], lowerer.context, lowerer.location)?;
        let result =
            lowerer
                .block
                .append_operation(stable_hlo::convert(input_values[0], output_type, lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.convert should return one result").as_ref()])
    }
}

/// Converts and broadcasts one implicitly compatible elementwise operand to the exact StableHLO result tensor type.
fn normalize_elementwise_operand<'b, 'c: 'b, 't: 'c, B, L>(
    input: ValueRef<'b, 'c, 't>,
    output_type: &ArrayType,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    let input_type = input.r#type()?;
    let input_tensor_type = input_type.cast::<TensorTypeRef>().ok_or_else(|| LoweringError::UnsupportedOp {
        op: format!("elementwise operand has non-tensor MLIR type '{input_type}'"),
    })?;
    let output_tensor_type = lower_tensor_type(output_type, context, location)?;
    let output_element_type = output_tensor_type.element_type()?;
    let mut input = input;
    if input_tensor_type.element_type()? != output_element_type {
        let dimensions = input_tensor_type.dimensions().collect::<Vec<_>>();
        let converted_type = context.tensor_type(output_element_type, dimensions.as_slice(), None, location)?;
        let converted = block.append_operation(stable_hlo::convert(input, converted_type, location)?)?;
        input = converted.result(0).expect("stablehlo.convert should return one result").as_ref();
    }
    if input_tensor_type.dimensions().ne(output_tensor_type.dimensions()) {
        let output_rank = output_tensor_type.rank();
        let input_rank = input_tensor_type.rank();
        let first_dimension = output_rank.checked_sub(input_rank).ok_or_else(|| LoweringError::UnsupportedOp {
            op: format!("cannot broadcast rank-{input_rank} elementwise operand to rank-{output_rank} output"),
        })?;
        let dimensions = (first_dimension..output_rank).collect::<Vec<_>>();
        let broadcast = block.append_operation(stable_hlo::broadcast(
            input,
            output_tensor_type,
            dimensions.as_slice(),
            location,
        )?)?;
        input = broadcast.result(0).expect("stablehlo.broadcast_in_dim should return one result").as_ref();
    }
    Ok(input)
}

/// Normalizes both operands of one implicitly broadcasting binary elementwise operation to its exact result type.
fn normalize_binary_elementwise_operands<'b, 'c: 'b, 't: 'c, B, L>(
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayType],
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<[ValueRef<'b, 'c, 't>; 2], LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    check_count!("input", input_values, 2, ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    Ok([
        normalize_elementwise_operand(input_values[0], &output_types[0], block, context, location)?,
        normalize_elementwise_operand(input_values[1], &output_types[0], block, context, location)?,
    ])
}

/// Returns the promoted numeric operand descriptor for one comparison result descriptor.
fn comparison_operand_type(input_types: &[ArrayType], output_types: &[ArrayType]) -> Result<ArrayType, LoweringError> {
    check_count!("input", input_types, 2, ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    let data_type = DataType::promoted(&[input_types[0].data_type(), input_types[1].data_type()]).map_err(|error| {
        LoweringError::UnsupportedOp { op: format!("cannot normalize comparison operands: {error}") }
    })?;
    Ok(output_types[0].clone().with_data_type(data_type))
}

/// Normalizes a select condition and both branches to the exact tensor descriptors required by StableHLO.
fn normalize_select_operands<'b, 'c: 'b, 't: 'c, B, L>(
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayType],
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<[ValueRef<'b, 'c, 't>; 3], LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    check_count!("input", input_values, 3, ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    let condition_type = output_types[0].clone().with_data_type(DataType::Boolean);
    Ok([
        normalize_elementwise_operand(input_values[0], &condition_type, block, context, location)?,
        normalize_elementwise_operand(input_values[1], &output_types[0], block, context, location)?,
        normalize_elementwise_operand(input_values[2], &output_types[0], block, context, location)?,
    ])
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for AddOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let [left, right] = normalize_binary_elementwise_operands(
            input_values,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        let result = lowerer.block.append_operation(stable_hlo::add(left, right, lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.add should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for SubOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let [left, right] = normalize_binary_elementwise_operands(
            input_values,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        let result = lowerer.block.append_operation(stable_hlo::subtract(left, right, lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.subtract should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for MulOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let [left, right] = normalize_binary_elementwise_operands(
            input_values,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        let result = lowerer.block.append_operation(stable_hlo::multiply(left, right, lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.multiply should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for DivOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let [left, right] = normalize_binary_elementwise_operands(
            input_values,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        let result = lowerer.block.append_operation(stable_hlo::divide(left, right, lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.divide should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for NegOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::negate(input_values[0], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.negate should return one result").as_ref()])
    }
}

/// Lowers complex sine or cosine through real operations while preserving an exact zero component for a zero real
/// input. The direct StableHLO complex trigonometric operations can evaluate a mathematically zero product as
/// `0 * inf`, yielding NaN when the imaginary component is large enough to overflow the hyperbolic factor.
///
/// # Parameters
///
///   - `input`: Complex tensor whose elementwise sine or cosine is lowered.
///   - `output_type`: Inferred complex output type, used to construct same-shaped real constants.
///   - `sine`: If `true`, lowers sine; otherwise, lowers cosine.
///   - `block`: Destination block for the StableHLO decomposition.
///   - `context`: MLIR context that owns all emitted types and attributes.
///   - `location`: Source location attached to emitted operations.
fn lower_complex_sine_or_cosine<'b, 'c: 'b, 't: 'c, B, L>(
    input: ValueRef<'b, 'c, 't>,
    output_type: &ArrayType,
    sine: bool,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    let part_data_type = match output_type.data_type() {
        DataType::C64 => DataType::F32,
        DataType::C128 => DataType::F64,
        _ => unreachable!(),
    };
    let part_type = ArrayType::new(part_data_type, output_type.shape().clone());
    let part_tensor_type = lower_tensor_type(&part_type, context, location)?;
    let zero = lower_f64_constant_splat(0.0, &part_type, part_tensor_type, block, context, location)?;
    let two = lower_f64_constant_splat(2.0, &part_type, part_tensor_type, block, context, location)?;

    let real = block.append_operation(stable_hlo::real(input, location)?)?;
    let real = real.result(0).expect("stablehlo.real should return one result").as_ref();
    let imaginary = block.append_operation(stable_hlo::imag(input, location)?)?;
    let imaginary = imaginary.result(0).expect("stablehlo.imag should return one result").as_ref();
    let real_is_zero = block.append_operation(stable_hlo::compare(
        real,
        zero,
        stable_hlo::ComparisonDirection::Equal,
        stable_hlo::ComparisonType::Float,
        location,
    )?)?;
    let real_is_zero = real_is_zero.result(0).expect("stablehlo.compare should return one result").as_ref();
    let real_sine = block.append_operation(stable_hlo::sine(real, Accuracy::Default, location)?)?;
    let real_sine = real_sine.result(0).expect("stablehlo.sine should return one result").as_ref();
    let real_cosine = block.append_operation(stable_hlo::cosine(real, Accuracy::Default, location)?)?;
    let real_cosine = real_cosine.result(0).expect("stablehlo.cosine should return one result").as_ref();
    let negative_imaginary = block.append_operation(stable_hlo::negate(imaginary, location)?)?;
    let negative_imaginary = negative_imaginary.result(0).expect("stablehlo.negate should return one result").as_ref();
    let positive_exponential_minus_one =
        block.append_operation(stable_hlo::exponential_minus_one(imaginary, Accuracy::Default, location)?)?;
    let positive_exponential_minus_one = positive_exponential_minus_one
        .result(0)
        .expect("stablehlo.exponential_minus_one should return one result")
        .as_ref();
    let negative_exponential_minus_one =
        block.append_operation(stable_hlo::exponential_minus_one(negative_imaginary, Accuracy::Default, location)?)?;
    let negative_exponential_minus_one = negative_exponential_minus_one
        .result(0)
        .expect("stablehlo.exponential_minus_one should return one result")
        .as_ref();
    let sinh_numerator = block.append_operation(stable_hlo::subtract(
        positive_exponential_minus_one,
        negative_exponential_minus_one,
        location,
    )?)?;
    let sinh_numerator = sinh_numerator.result(0).expect("stablehlo.subtract should return one result").as_ref();
    let sinh = block.append_operation(stable_hlo::divide(sinh_numerator, two, location)?)?;
    let sinh = sinh.result(0).expect("stablehlo.divide should return one result").as_ref();
    let cosh_without_two = block.append_operation(stable_hlo::add(
        positive_exponential_minus_one,
        negative_exponential_minus_one,
        location,
    )?)?;
    let cosh_without_two = cosh_without_two.result(0).expect("stablehlo.add should return one result").as_ref();
    let cosh_numerator = block.append_operation(stable_hlo::add(cosh_without_two, two, location)?)?;
    let cosh_numerator = cosh_numerator.result(0).expect("stablehlo.add should return one result").as_ref();
    let cosh = block.append_operation(stable_hlo::divide(cosh_numerator, two, location)?)?;
    let cosh = cosh.result(0).expect("stablehlo.divide should return one result").as_ref();

    let (real_result, imaginary_result) = if sine {
        // Mask the overflowing factor before multiplication as well as selecting the final complex result. Some XLA
        // optimization paths otherwise speculate the unselected `0 * inf` expression and preserve its NaN.
        let safe_cosh = block.append_operation(stable_hlo::select(real_is_zero, zero, cosh, location)?)?;
        let safe_cosh = safe_cosh.result(0).expect("stablehlo.select should return one result").as_ref();
        let real_result = block.append_operation(stable_hlo::multiply(real_sine, safe_cosh, location)?)?;
        let real_result = real_result.result(0).expect("stablehlo.multiply should return one result").as_ref();
        let imaginary_result = block.append_operation(stable_hlo::multiply(real_cosine, sinh, location)?)?;
        let imaginary_result =
            imaginary_result.result(0).expect("stablehlo.multiply should return one result").as_ref();
        (real_result, imaginary_result)
    } else {
        let real_result = block.append_operation(stable_hlo::multiply(real_cosine, cosh, location)?)?;
        let real_result = real_result.result(0).expect("stablehlo.multiply should return one result").as_ref();
        let negative_real_sine = block.append_operation(stable_hlo::negate(real_sine, location)?)?;
        let negative_real_sine =
            negative_real_sine.result(0).expect("stablehlo.negate should return one result").as_ref();
        let safe_sinh = block.append_operation(stable_hlo::select(real_is_zero, zero, sinh, location)?)?;
        let safe_sinh = safe_sinh.result(0).expect("stablehlo.select should return one result").as_ref();
        let imaginary_result =
            block.append_operation(stable_hlo::multiply(negative_real_sine, safe_sinh, location)?)?;
        let imaginary_result =
            imaginary_result.result(0).expect("stablehlo.multiply should return one result").as_ref();
        (real_result, imaginary_result)
    };
    let ordinary_result = block.append_operation(stable_hlo::complex(real_result, imaginary_result, location)?)?;
    let ordinary_result = ordinary_result.result(0).expect("stablehlo.complex should return one result").as_ref();
    let zero_real_result = if sine {
        block.append_operation(stable_hlo::complex(zero, imaginary_result, location)?)?
    } else {
        block.append_operation(stable_hlo::complex(real_result, zero, location)?)?
    };
    let zero_real_result = zero_real_result.result(0).expect("stablehlo.complex should return one result").as_ref();
    let result =
        block.append_operation(stable_hlo::select(real_is_zero, zero_real_result, ordinary_result, location)?)?;
    Ok(result.result(0).expect("stablehlo.select should return one result").as_ref())
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for SinOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        if output_types[0].data_type().is_complex() {
            return Ok(vec![lower_complex_sine_or_cosine(
                input_values[0],
                &output_types[0],
                true,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?]);
        }
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
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        if output_types[0].data_type().is_complex() {
            return Ok(vec![lower_complex_sine_or_cosine(
                input_values[0],
                &output_types[0],
                false,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?]);
        }
        let result = lowerer.block.append_operation(stable_hlo::cosine(
            input_values[0],
            Accuracy::Default,
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.cosine should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for Atan2Operation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let [left, right] = normalize_binary_elementwise_operands(
            input_values,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        let result = lowerer.block.append_operation(stable_hlo::atan2(left, right, lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.atan2 should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for ExpOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for LogOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for SqrtOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for RsqrtOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result =
            lowerer
                .block
                .append_operation(stable_hlo::rsqrt(input_values[0], Accuracy::Default, lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.rsqrt should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for TanhOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result =
            lowerer
                .block
                .append_operation(stable_hlo::tanh(input_values[0], Accuracy::Default, lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.tanh should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for LogisticOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::logistic(
            input_values[0],
            Accuracy::Default,
            lowerer.location,
        )?)?;
        Ok(vec![result.result(0).expect("stablehlo.logistic should return one result").as_ref()])
    }
}

/// [`ErfOperation`] lowers to `chlo.erf`, which the XLA compiler legalizes to a rational polynomial approximation
/// over StableHLO operations during compilation.
impl<V: MlirLowerableValue> LowerableXlaOperation<V> for ErfOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(chlo::erf(input_values[0], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("chlo.erf should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for PowOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let [left, right] = normalize_binary_elementwise_operands(
            input_values,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        let result = lowerer.block.append_operation(stable_hlo::power(left, right, lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.power should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for SignOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::sign(input_values[0], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.sign should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for FloorOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::floor(input_values[0], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.floor should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for CeilOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::ceil(input_values[0], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.ceil should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for RoundOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer
            .block
            .append_operation(stable_hlo::round_with_nearest_even_tie_break(input_values[0], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.round_nearest_even should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for MaxOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let [left, right] = normalize_binary_elementwise_operands(
            input_values,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        let result = lowerer.block.append_operation(stable_hlo::maximum(left, right, lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.maximum should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for MinOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let [left, right] = normalize_binary_elementwise_operands(
            input_values,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        let result = lowerer.block.append_operation(stable_hlo::minimum(left, right, lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.minimum should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for RemOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let [left, right] = normalize_binary_elementwise_operands(
            input_values,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        let result = lowerer.block.append_operation(stable_hlo::remainder(left, right, lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.remainder should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for AbsOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::abs(input_values[0], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.abs should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for ComplexOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
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
impl<V: MlirLowerableValue> LowerableXlaOperation<V> for ConjugateOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for RealOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::real(input_values[0], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.real should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for ImaginaryOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::imag(input_values[0], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.imag should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for TransposeOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for DotOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for IotaOperation<ArrayType> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("input", input_values, 0, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        let output_tensor_type = lowerer.lower_tensor_type(&output_types[0])?;
        let result = lowerer.block.append_operation(stable_hlo::iota(
            output_tensor_type,
            self.dimension(),
            lowerer.location,
        )?)?;
        let result = result.result(0).expect("stablehlo.iota should return one result").as_ref();
        Ok(vec![annotate_output_memory(
            result,
            &output_types[0],
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for CoordinateBasisOperation<ArrayType> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("input", input_values, 0, ProgramError);
        lower_coordinate_basis_to_mlir(self, output_types, &mut lowerer.block, lowerer.context, lowerer.location)
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for ConstantOperation<CpuArray> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("input", input_values, 0, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        let constant_value = self.value().lower_constant_value(
            lowerer.captured_values.as_slice(),
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        Ok(vec![constant_value])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for LegacyReshapeOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_reshape_to_mlir(self, input_values, output_types, &mut lowerer.block, lowerer.context, lowerer.location)
    }
}

/// Lowers a [`LegacyReshapeOperation`] after validating its unary input and single output contract.
fn lower_reshape_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &LegacyReshapeOperation,
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    check_count!("input", input_values, 1, ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    let input = if let Some(dimensions) = operation.parameters().dimensions() {
        let transpose =
            block.append_operation(stable_hlo::transpose(input_values[0], dimensions.as_slice(), location)?)?;
        transpose.result(0).expect("stablehlo.transpose should return one result").as_ref()
    } else {
        input_values[0]
    };
    let result = if let Some(expressions) = operation.parameters().output_dimension_expressions() {
        let shape = lower_reshape_shape(expressions, input_values[0], block, context, location)?;
        let output_bounds = output_types[0]
            .shape()
            .dimensions()
            .iter()
            .map(|size| match size {
                Dimension::Static(value) => Some(*value),
                Dimension::Dynamic(_) => stable_hlo_dynamic_dimension_bound(size),
            })
            .collect::<Vec<_>>();
        let reshape = block.append_operation(stable_hlo::dynamic_reshape(input, shape, &output_bounds, location)?)?;
        let result = reshape.result(0).expect("stablehlo.dynamic_reshape should return one result").as_ref();
        let output_type = lower_tensor_type(&output_types[0], context, location)?;
        if result.r#type()? == output_type.as_ref() {
            result
        } else {
            let cast = block.append_operation(tensor::cast(result, output_type, location)?)?;
            cast.result(0).expect("tensor.cast should return one result").as_ref()
        }
    } else if output_types[0].static_shape().is_none() {
        // Core type inference only admits a fixed dynamic target when it is exactly the input shape after applying
        // `dimensions`, so the identity or transpose above already has the required result type.
        input
    } else {
        let output_shape = static_dimensions(&output_types[0])?;
        for dimension in &output_shape {
            reshape_dimension_i64(*dimension)?;
        }
        let reshape = block.append_operation(stable_hlo::reshape(input, output_shape.as_slice(), location)?)?;
        reshape.result(0).expect("stablehlo.reshape should return one result").as_ref()
    };
    if operation.parameters().output_sharding().is_some() {
        let output_sharding = output_types[0]
            .sharding()
            .expect("reshape type inference should preserve a requested output sharding");
        lower_sharding_constraint(&[result], output_sharding, block, location)
    } else {
        Ok(vec![result])
    }
}

/// Lowers one symbolic reshape dimension to a scalar `i32` StableHLO value.
fn lower_reshape_dimension_expression<'b, 'c: 'b, 't: 'c>(
    expression: &ReshapeDimensionExpression,
    input: ValueRef<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let scalar_type = context
        .tensor_type(context.signless_integer_type(32), &[], None, location)
        .map_err(|_| LoweringError::InvalidTensorType { array_type: ArrayType::scalar(DataType::I32) })?;
    let constant = |value: usize, block: &mut BlockRef<'b, 'c, 't>| -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
        let value = reshape_dimension_i32(value)?;
        let elements = lower_constant_elements_attribute(DataType::I32, scalar_type, i64::from(value), context)?;
        let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
        Ok(constant.result(0).expect("stablehlo.constant should return one result").as_ref())
    };
    match expression {
        ReshapeDimensionExpression::Constant(value) => constant(*value, block),
        ReshapeDimensionExpression::InputDimension(dimension) => {
            let size = block.append_operation(stable_hlo::get_dimension_size(input, *dimension, location)?)?;
            Ok(size.result(0).expect("stablehlo.get_dimension_size should return one result").as_ref())
        }
        ReshapeDimensionExpression::Product(factors) => {
            let mut factors = factors.iter();
            let Some(first) = factors.next() else {
                return constant(1, block);
            };
            let mut product = lower_reshape_dimension_expression(first, input, block, context, location)?;
            for factor in factors {
                let factor = lower_reshape_dimension_expression(factor, input, block, context, location)?;
                let multiply = block.append_operation(stable_hlo::multiply(product, factor, location)?)?;
                product = multiply.result(0).expect("stablehlo.multiply should return one result").as_ref();
            }
            Ok(product)
        }
        ReshapeDimensionExpression::ExactDivision { numerator, denominator } => {
            let numerator = lower_reshape_dimension_expression(numerator, input, block, context, location)?;
            let denominator = lower_reshape_dimension_expression(denominator, input, block, context, location)?;
            let divide = block.append_operation(stable_hlo::divide(numerator, denominator, location)?)?;
            Ok(divide.result(0).expect("stablehlo.divide should return one result").as_ref())
        }
    }
}

/// Converts one Ryft reshape dimension to StableHLO's signed shape element type.
fn reshape_dimension_i64(value: usize) -> Result<i64, LoweringError> {
    i64::try_from(value).map_err(|_| LoweringError::ReshapeDimensionOutOfRange { value, bit_width: 64 })
}

/// Converts one Ryft reshape dimension to the signed shape element type required by dynamic StableHLO reshape.
fn reshape_dimension_i32(value: usize) -> Result<i32, LoweringError> {
    i32::try_from(value).map_err(|_| LoweringError::ReshapeDimensionOutOfRange { value, bit_width: 32 })
}

/// Lowers symbolic reshape dimensions to the rank-1 `i32` shape tensor consumed by `stablehlo.dynamic_reshape`.
fn lower_reshape_shape<'b, 'c: 'b, 't: 'c>(
    expressions: &[ReshapeDimensionExpression],
    input: ValueRef<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let shape_type = context
        .tensor_type(context.signless_integer_type(32), &[MlirSize::Static(expressions.len())], None, location)
        .map_err(|_| LoweringError::InvalidTensorType {
            array_type: ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(expressions.len())])),
        })?;
    if expressions.is_empty() {
        let elements = context
            .dense_i32_elements_attribute(shape_type, &[])
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: DataType::I32 })?;
        let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
        return Ok(constant.result(0).expect("stablehlo.constant should return one result").as_ref());
    }

    let mut dimensions = Vec::with_capacity(expressions.len());
    for expression in expressions {
        let dimension = lower_reshape_dimension_expression(expression, input, block, context, location)?;
        let reshape = block.append_operation(stable_hlo::reshape(dimension, &[1], location)?)?;
        dimensions.push(reshape.result(0).expect("stablehlo.reshape should return one result").as_ref());
    }
    let shape = block.append_operation(stable_hlo::concatenate(dimensions.as_slice(), 0, location)?)?;
    Ok(shape.result(0).expect("stablehlo.concatenate should return one result").as_ref())
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for PadOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_pad_to_mlir(self, input_values, output_types, &mut lowerer.block, lowerer.context, lowerer.location)
    }
}

/// Validates that one pad interior amount can be represented by StableHLO's signed 64-bit attribute storage.
fn validate_pad_interior_padding(value: usize) -> Result<(), LoweringError> {
    i64::try_from(value).map(|_| ()).map_err(|_| LoweringError::PadInteriorPaddingOutOfRange { value })
}

/// Lowers a pad through the shared StableHLO path used by plain, generic-array, and shard-map dispatch.
fn lower_pad_to_mlir<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    operation: &PadOperation,
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayType],
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    check_count!("input", input_values, 2, ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    operation.interior_padding().iter().copied().try_for_each(validate_pad_interior_padding)?;
    let pad = block.append_operation(stable_hlo::pad(
        input_values[0],
        input_values[1],
        operation.edge_padding_low(),
        operation.edge_padding_high(),
        operation.interior_padding(),
        location,
    )?)?;
    let result = pad.result(0).expect("stablehlo.pad should return one result").as_ref();
    let output_type = lower_tensor_type(&output_types[0], context, location)?;
    if result.r#type()? == output_type.as_ref() {
        Ok(vec![result])
    } else {
        let cast = block.append_operation(tensor::cast(result, output_type, location)?)?;
        Ok(vec![cast.result(0).expect("tensor.cast should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for LegacyBroadcastOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_broadcast_to_mlir(
            self,
            input_values,
            lowerer.input_types.as_slice(),
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )
    }
}

/// Returns whether a broadcast requires an explicit output-placement constraint.
fn broadcast_changes_explicit_sharding(input_type: &ArrayType, output_type: &ArrayType, output_axes: &[usize]) -> bool {
    let Some(output_sharding) = output_type.sharding() else {
        return false;
    };
    let input_sharding = input_type.sharding();
    let uses_explicit_axis = |dimension: &ShardingDimension, sharding: &Sharding| match dimension {
        ShardingDimension::Sharded(axis_names) => {
            axis_names.iter().any(|name| sharding.mesh().axis_type(name) == Some(MeshAxisType::Explicit))
        }
        ShardingDimension::Replicated | ShardingDimension::Unconstrained => false,
    };
    if input_sharding.is_none() {
        return output_sharding.mesh().axes().iter().any(|axis| axis.r#type() == MeshAxisType::Explicit);
    }
    if let Some(input_sharding) = input_sharding
        && input_sharding.mesh() != output_sharding.mesh()
    {
        return input_sharding
            .mesh()
            .axes()
            .iter()
            .chain(output_sharding.mesh().axes())
            .any(|axis| axis.r#type() == MeshAxisType::Explicit);
    }

    let projected_input_sharding = input_sharding
        .map(|sharding| {
            sharding
                .with_broadcasted_dimensions(output_type.rank(), output_axes)
                .expect("broadcast output axes are validated before lowering")
        })
        .unwrap_or_else(|| Sharding::replicated(output_sharding.mesh().clone(), output_type.rank()));

    // Even when the ranked placement labels are unchanged, expanding a unit dimension that is partitioned over an
    // explicit mesh axis changes which partitions own the replicated values and must remain visible to Shardy.
    let expands_explicit_dimension = output_axes.iter().copied().enumerate().any(|(input_axis, output_axis)| {
        input_type.dimension(input_axis as isize) == Dimension::Static(1)
            && output_type.dimension(output_axis as isize) != Dimension::Static(1)
            && (uses_explicit_axis(&projected_input_sharding.dimensions()[output_axis], &projected_input_sharding)
                || uses_explicit_axis(&output_sharding.dimensions()[output_axis], output_sharding))
    });

    expands_explicit_dimension || projected_input_sharding.conflicts_on_explicit_axes_with(output_sharding)
}

/// Lowers a broadcast and explicitly constrains any placement transition over an explicit mesh axis.
#[allow(clippy::too_many_arguments)]
fn lower_broadcast_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &LegacyBroadcastOperation,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    check_count!("input", input_values, 1, ProgramError);
    check_count!("input", input_types, 1, ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    let output_tensor_type = lower_tensor_type(&output_types[0], context, location)?;
    let broadcast = block.append_operation(stable_hlo::broadcast(
        input_values[0],
        output_tensor_type,
        operation.output_axes(),
        location,
    )?)?;
    let result = broadcast.result(0).expect("stablehlo.broadcast_in_dim should return one result").as_ref();
    if broadcast_changes_explicit_sharding(&input_types[0], &output_types[0], operation.output_axes()) {
        lower_sharding_constraint(&[result], output_types[0].sharding().unwrap(), block, location)
    } else {
        Ok(vec![result])
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

/// Lowers a static slice.
fn lower_slice_to_mlir<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    operation: &SliceOperation,
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayType],
    block: &mut B,
    _context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    check_count!("input", input_values, 1, ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    let result = block.append_operation(stable_hlo::slice(
        input_values[0],
        operation.start_indices(),
        operation.limit_indices(),
        operation.strides(),
        location,
    )?)?;
    Ok(vec![result.result(0).expect("stablehlo.slice should return one result").as_ref()])
}

fn lower_unplaced_constant_output<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
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
    if data_type.is_complex() {
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

/// Lowers one integer-valued constant and applies its declared non-device memory placement to the final value.
fn lower_constant_output<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    output_types: &[ArrayType],
    integer_value: i64,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let values = lower_unplaced_constant_output(output_types, integer_value, block, context, location)?;
    check_count!("output", output_types, 1, ProgramError);
    Ok(vec![annotate_output_memory(values[0], &output_types[0], block, context, location)?])
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
    // The zero-space type has exactly one value. Both `zero_like` and `one_like` therefore materialize that value,
    // even though a typed `one` operation remains invalid because the type has no numeric multiplicative identity.
    let integer_value = match output_types {
        [output_type] if output_type.data_type().is_zero() => 0,
        _ => integer_value,
    };
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

/// Applies the non-default memory placement declared by `output_type` to `value`.
fn annotate_output_memory<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    value: ValueRef<'b, 'c, 't>,
    output_type: &ArrayType,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    if output_type.memory() == Memory::Device {
        return Ok(value);
    }
    Ok(lower_transfer_to_memory(output_type.memory(), &[value], block, context, location)?[0])
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
    V: MlirLowerableValue,
{
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
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
            Self::Constant(operation) => operation.lower_to_mlir(input_values, regions, output_types, mode, lowerer),
            Self::ConvertElementType(operation) => {
                operation.lower_to_mlir(input_values, regions, output_types, mode, lowerer)
            }
            Self::Iota(operation) => <IotaOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::CoordinateBasis(operation) => {
                <CoordinateBasisOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                    operation,
                    input_values,
                    regions,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            Self::Neg(operation) => <NegOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Add(operation) => <AddOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Sub(operation) => <SubOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Mul(operation) => <MulOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Div(operation) => <DivOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Sin(operation) => <SinOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Cos(operation) => <CosOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Atan2(operation) => <Atan2Operation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Exp(operation) => <ExpOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Log(operation) => <LogOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Sqrt(operation) => <SqrtOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Rsqrt(operation) => <RsqrtOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Tanh(operation) => <TanhOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Logistic(operation) => <LogisticOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Erf(operation) => <ErfOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Pow(operation) => <PowOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Sign(operation) => <SignOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Floor(operation) => <FloorOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Ceil(operation) => <CeilOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Round(operation) => <RoundOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Max(operation) => <MaxOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Min(operation) => <MinOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Rem(operation) => <RemOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Abs(operation) => <AbsOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Complex(operation) => <ComplexOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Conjugate(operation) => <ConjugateOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Real(operation) => <RealOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Imaginary(operation) => <ImaginaryOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
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
            Self::CustomCall(operation) => lower_custom_call_to_mlir(
                operation,
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
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
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Transpose(operation) => <TransposeOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Reshape(operation) => <LegacyReshapeOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
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
            Self::Broadcast(operation) => <LegacyBroadcastOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            Self::Slice(operation) => lower_slice_to_mlir(
                operation,
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
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
            Self::Pad(operation) => lower_pad_to_mlir(
                operation,
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            Self::Concatenate(operation) => {
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
            Self::Sort(operation) => lower_sort_to_mlir(
                operation,
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            Self::RngBitGenerator(operation) => lower_rng_bit_generator_to_mlir(
                operation,
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            Self::ScaledDot(operation) => {
                let collective_state = lowerer.collective_state.clone();
                lower_scaled_dot_to_mlir(
                    operation,
                    &collective_state,
                    input_values,
                    &lowerer.input_types,
                    output_types,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )
            }
            Self::DotProductAttention(operation) => {
                let collective_state = lowerer.collective_state.clone();
                lower_dot_product_attention_to_mlir(
                    operation,
                    &collective_state,
                    input_values,
                    &lowerer.input_types,
                    output_types,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )
            }
            Self::DotProductAttentionBackward(operation) => {
                let collective_state = lowerer.collective_state.clone();
                lower_dot_product_attention_backward_to_mlir(
                    operation,
                    &collective_state,
                    input_values,
                    &lowerer.input_types,
                    output_types,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )
            }
            Self::Compare(operation) => {
                let operand_type = comparison_operand_type(&lowerer.input_types, output_types)?;
                let [left, right] = normalize_binary_elementwise_operands(
                    input_values,
                    &[operand_type],
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?;
                let value =
                    lower_compare_to_mlir(operation.direction(), left, right, &mut lowerer.block, lowerer.location)?;
                Ok(vec![value])
            }
            Self::Not(_) => {
                let result = lowerer.block.append_operation(stable_hlo::not(input_values[0], lowerer.location)?)?;
                Ok(vec![result.result(0).expect("stablehlo.not should return one result").as_ref()])
            }
            Self::And(_) => {
                let [left, right] = normalize_binary_elementwise_operands(
                    input_values,
                    output_types,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?;
                let result = lowerer.block.append_operation(stable_hlo::and(left, right, lowerer.location)?)?;
                Ok(vec![result.result(0).expect("stablehlo.and should return one result").as_ref()])
            }
            Self::Or(_) => {
                let [left, right] = normalize_binary_elementwise_operands(
                    input_values,
                    output_types,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?;
                let result = lowerer.block.append_operation(stable_hlo::or(left, right, lowerer.location)?)?;
                Ok(vec![result.result(0).expect("stablehlo.or should return one result").as_ref()])
            }
            Self::Xor(_) => {
                let [left, right] = normalize_binary_elementwise_operands(
                    input_values,
                    output_types,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?;
                let result = lowerer.block.append_operation(stable_hlo::xor(left, right, lowerer.location)?)?;
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
            Self::AllGather(operation) => {
                check_count!("input", input_values, 1, ProgramError);
                check_count!("output", output_types, 1, ProgramError);
                let collective_state = lowerer.collective_state.clone();
                lower_all_gather_to_mlir(
                    operation,
                    &collective_state,
                    input_values[0],
                    &output_types[0],
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )
            }
            Self::PSumScatter(operation) => {
                check_count!("input", input_values, 1, ProgramError);
                check_count!("output", output_types, 1, ProgramError);
                let collective_state = lowerer.collective_state.clone();
                lower_psum_scatter_to_mlir(
                    operation,
                    &collective_state,
                    input_values[0],
                    &output_types[0],
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )
            }
            Self::Ppermute(operation) => {
                check_count!("input", input_values, 1, ProgramError);
                let collective_state = lowerer.collective_state.clone();
                lower_ppermute_to_mlir(
                    operation,
                    &collective_state,
                    input_values[0],
                    &mut lowerer.block,
                    lowerer.location,
                )
            }
            Self::AllToAll(operation) => {
                check_count!("input", input_values, 1, ProgramError);
                check_count!("output", output_types, 1, ProgramError);
                let collective_state = lowerer.collective_state.clone();
                lower_all_to_all_to_mlir(
                    operation,
                    &collective_state,
                    input_values[0],
                    &lowerer.input_types[0],
                    &output_types[0],
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )
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
                let [condition, on_true, on_false] = normalize_select_operands(
                    input_values,
                    output_types,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?;
                let result = lowerer.block.append_operation(stable_hlo::select(
                    condition,
                    on_true,
                    on_false,
                    lowerer.location,
                )?)?;
                Ok(vec![result.result(0).expect("stablehlo.select should return one result").as_ref()])
            }
            Self::Condition(condition) => condition.lower_to_mlir(input_values, regions, output_types, mode, lowerer),
            Self::While(while_operation) => {
                while_operation.lower_to_mlir(input_values, regions, output_types, mode, lowerer)
            }
            Self::Scan(scan) => scan.lower_to_mlir(input_values, regions, output_types, mode, lowerer),
            Self::CustomJvp(_) => lower_nested_program_inline(
                &regions[0],
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
                lowerer.captured_values.as_slice(),
                false,
                lowerer.nested_functions.as_ref(),
                &lowerer.collective_state,
                &mut lowerer.token,
            ),
            Self::CustomVjp(_) => lower_nested_program_inline(
                &regions[0],
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
                lowerer.captured_values.as_slice(),
                false,
                lowerer.nested_functions.as_ref(),
                &lowerer.collective_state,
                &mut lowerer.token,
            ),
            // The opaque custom-VJP tangent carrier is a forward-mode tangent map that reverse mode transposes away
            // before lowering, so it never reaches the backend; reaching here means a forward-mode use of `custom_vjp`
            // slipped through, which is reverse-mode-only.
            Self::CustomVjpTangent(operation) => Err(ProgramError::UnsupportedOperation {
                message: format!("operation `{}` cannot be lowered to StableHLO", operation.name(),),
            }
            .into()),
            Self::Rematerialize(_) => lower_nested_program_inline(
                &regions[0],
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
                lowerer.captured_values.as_slice(),
                false,
                lowerer.nested_functions.as_ref(),
                &lowerer.collective_state,
                &mut lowerer.token,
            ),
            Self::JitCall(_) => lower_jit_call(
                &regions[0],
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
                lowerer.nested_functions.as_ref(),
                &lowerer.collective_state,
                &mut lowerer.token,
            ),
            Self::ShardMap(shard_map_op) => {
                let simplified_body = regions[0]
                    .simplified()
                    .map_err(|error| LoweringError::SimplificationFailure { message: error.to_string() })?;
                lower_manual_computation_inline(
                    &mut lowerer.block,
                    input_values,
                    shard_map_op.shard_map(),
                    &simplified_body,
                    shard_map_op.global_output_types(),
                    lowerer.context,
                    lowerer.location,
                    &lowerer.collective_state,
                )
            }
        }
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for ConditionOperation<V> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lowerer.lower_condition(regions, input_values)
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for WhileOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lowerer.lower_while(self, regions, input_values)
    }
}

impl<V: MlirLowerableValue, Capture> LowerableXlaOperation<V> for ScanOperation<Capture>
where
    Capture: Value<Type = ArrayType>,
{
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lowerer.lower_scan(self, regions, input_values)
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

/// Lowers one traced random bit generation to a `stablehlo.rng_bit_generator`, mapping the algorithm to the
/// corresponding StableHLO algorithm attribute. The two results are the advanced generator state and the bits.
fn lower_rng_bit_generator_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &RngBitGeneratorOperation,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    check_count!("input", input_values, 1, ProgramError);
    let algorithm = match operation.algorithm() {
        RandomAlgorithm::ThreeFry => stable_hlo::RngAlgorithm::ThreeFry,
        RandomAlgorithm::Philox => stable_hlo::RngAlgorithm::Philox,
    };
    let output_type = lower_tensor_type(operation.output_type(), context, location)?;
    let generated =
        block.append_operation(stable_hlo::rng_bit_generator(input_values[0], algorithm, output_type, location)?)?;
    Ok(vec![
        generated.result(0).expect("stablehlo.rng_bit_generator should return a state result").as_ref(),
        generated.result(1).expect("stablehlo.rng_bit_generator should return a bits result").as_ref(),
    ])
}

/// Lowers one traced sort to a `stablehlo.sort` with a synthesized comparator region: two scalar block arguments
/// per operand, comparing the first `key_count` operand pairs lexicographically with the sort's direction (`LT`
/// for ascending and `GT` for descending) so that non-key operands ride along as passengers. For keys `0..N` the
/// synthesized result is `cmp_0 OR (eq_0 AND (cmp_1 OR (eq_1 AND … cmp_{N-1})))`, built right to left, where
/// `cmp_i` is the direction comparison of key pair `i` and `eq_i` its equality comparison. Each key derives its
/// own comparison type from that key's data type: floating-point keys compare with `TOTALORDER` semantics (for
/// both the direction and the equality comparison, so NaN ties fall through deterministically, matching XLA's
/// `num_keys` comparator), Boolean and unsigned-integer keys compare `UNSIGNED`, and signed-integer keys compare
/// `SIGNED`. The emitted sort is always stable, which is what routes ranking ties to the lowest index.
fn lower_sort_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &SortOperation,
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    if output_types.is_empty() {
        return Err(
            ProgramError::UnsupportedOperation { message: "'sort' needs at least one input".to_string() }.into()
        );
    }
    let mut comparator_arguments = Vec::with_capacity(2 * output_types.len());
    for output_type in output_types {
        let scalar_type = lower_tensor_type(&ArrayType::scalar(output_type.data_type()), context, location)?;
        comparator_arguments.push((scalar_type, location));
        comparator_arguments.push((scalar_type, location));
    }
    let comparator_block = context.block(comparator_arguments.as_slice());
    let mut comparator = context.region();
    let mut comparator_block = comparator.append_block(comparator_block)?;
    let comparison_direction = match operation.direction() {
        SortDirection::Ascending => stable_hlo::ComparisonDirection::LessThan,
        SortDirection::Descending => stable_hlo::ComparisonDirection::GreaterThan,
    };
    // The lexicographic chain builds right to left: the innermost term is the last key's direction comparison, and
    // every earlier key wraps the accumulated tail as `cmp_i OR (eq_i AND tail)`.
    let mut compared = None;
    for key_index in (0..operation.key_count()).rev() {
        let left_key = comparator_block.argument(2 * key_index)?.as_ref();
        let right_key = comparator_block.argument(2 * key_index + 1)?.as_ref();
        let comparison_type = match output_types[key_index].data_type() {
            DataType::Boolean
            | DataType::U1
            | DataType::U2
            | DataType::U4
            | DataType::U8
            | DataType::U16
            | DataType::U32
            | DataType::U64 => stable_hlo::ComparisonType::Unsigned,
            DataType::I1
            | DataType::I2
            | DataType::I4
            | DataType::I8
            | DataType::I16
            | DataType::I32
            | DataType::I64 => stable_hlo::ComparisonType::Signed,
            _ => stable_hlo::ComparisonType::TotalOrder,
        };
        let directed = comparator_block.append_operation(stable_hlo::compare(
            left_key,
            right_key,
            comparison_direction,
            comparison_type,
            location,
        )?)?;
        let directed = directed.result(0).expect("stablehlo.compare should return one result").as_ref();
        compared = Some(match compared {
            None => directed,
            Some(tail) => {
                let equal = comparator_block.append_operation(stable_hlo::compare(
                    left_key,
                    right_key,
                    stable_hlo::ComparisonDirection::Equal,
                    comparison_type,
                    location,
                )?)?;
                let equal = equal.result(0).expect("stablehlo.compare should return one result").as_ref();
                let tied = comparator_block.append_operation(stable_hlo::and(equal, tail, location)?)?;
                let tied = tied.result(0).expect("stablehlo.and should return one result").as_ref();
                let chained = comparator_block.append_operation(stable_hlo::or(directed, tied, location)?)?;
                chained.result(0).expect("stablehlo.or should return one result").as_ref()
            }
        });
    }
    // A `SortOperation` always carries at least one key, so the chain is never empty.
    let compared = compared.unwrap();
    comparator_block.append_operation(stable_hlo::r#return(&[compared], location)?)?;
    let sorted =
        block.append_operation(stable_hlo::sort(input_values, operation.axis(), true, comparator, location)?)?;
    Ok((0..output_types.len())
        .map(|index| sorted.result(index).expect("stablehlo.sort should return one result per operand").as_ref())
        .collect())
}

/// Returns whether one block-scaled operand pair uses a format and block-size combination that XLA's GPU
/// block-scaling rewriter accepts for the `__op$block_scaled_dot` custom call: MXFP8 (`f8e4m3fn` or `f8e5m2`
/// elements with `f8e8m0fnu` scales over blocks of 32) or NVFP4 (`f4e2m1fn` elements with `f8e4m3fn` scales over
/// blocks of 16).
fn block_scaled_formats_qualify(elements: DataType, scales: DataType, block_size: usize) -> bool {
    matches!(
        (elements, scales, block_size),
        (DataType::F8E4M3FN | DataType::F8E5M2, DataType::F8E8M0FNU, 32) | (DataType::F4E2M1FN, DataType::F8E4M3FN, 16)
    )
}

/// Lowers one traced block-scaled dot. On CUDA targets whose formats and block size qualify (see
/// [`block_scaled_formats_qualify`]), it emits the `__op$block_scaled_dot` custom call — operand order
/// `(lhs, rhs, lhs_scales, rhs_scales)` plus the optional trailing scalar global scale, with the contracting
/// dimension last on both element operands and rank-3 operands carrying one shared leading batch dimension — which
/// XLA's GPU block-scaling rewriter lowers to cuDNN's native block-scaled tensor-core dot (cuDNN 9.10+) or to
/// expanded reference HLO. Everywhere else it emits the portable dequantization composition (upcast, expand the
/// scales across their blocks, multiply, contract, and multiply in a present global scale), which XLA fuses like
/// any other dequantized dot.
fn lower_scaled_dot_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &ScaledDotOperation,
    collective_state: &CollectiveLoweringState,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    if input_values.len() != 4 && input_values.len() != 5 {
        return Err(ProgramError::InvalidInputCount { expected: 4, actual: input_values.len() }.into());
    }
    check_count!("input", input_types, input_values.len(), ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    let is_cuda_target = collective_state.target_platform().is_some_and(|platform| platform == "cuda");
    if is_cuda_target
        && block_scaled_formats_qualify(input_types[0].data_type(), input_types[1].data_type(), operation.block_size())
        && block_scaled_formats_qualify(input_types[2].data_type(), input_types[3].data_type(), operation.block_size())
    {
        let custom_call = CustomCallOperation::new("__op$block_scaled_dot", vec![output_types[0].clone()]);
        let mut reordered_inputs = vec![input_values[0], input_values[2], input_values[1], input_values[3]];
        reordered_inputs.extend(input_values.get(4).copied());
        return lower_custom_call_to_mlir(
            &custom_call,
            reordered_inputs.as_slice(),
            output_types,
            block,
            context,
            location,
        );
    }

    // Portable fallback: dequantize both operands (upcast the elements and scales to the accumulation type,
    // expand each scale across its block along the trailing contracting dimension, and multiply), contract over
    // the last dimension of both sides (batching over the shared leading dimension for rank-3 operands), and
    // multiply in a present global scale.
    let accumulation_type = output_types[0].data_type();
    let mut dequantized = Vec::with_capacity(2);
    for (elements_index, scales_index) in [(0usize, 1usize), (2, 3)] {
        let element_dimensions = input_types[elements_index]
            .static_shape()
            .ok_or_else(|| LoweringError::UnsupportedOp { op: "dynamically shaped scaled_dot".to_string() })?
            .dimensions()
            .to_vec();
        let scale_dimensions = input_types[scales_index]
            .static_shape()
            .ok_or_else(|| LoweringError::UnsupportedOp { op: "dynamically shaped scaled_dot".to_string() })?
            .dimensions()
            .to_vec();
        let element_type = ArrayType::new(
            accumulation_type,
            Shape::new(element_dimensions.iter().map(|&size| Dimension::Static(size)).collect()),
        );
        let expanded_type = ArrayType::new(
            accumulation_type,
            Shape::new(
                scale_dimensions
                    .iter()
                    .map(|&size| Dimension::Static(size))
                    .chain(std::iter::once(Dimension::Static(operation.block_size())))
                    .collect(),
            ),
        );
        let element_tensor_type = lower_tensor_type(&element_type, context, location)?;
        let scale_tensor_type = lower_tensor_type(
            &ArrayType::new(
                accumulation_type,
                Shape::new(scale_dimensions.iter().map(|&size| Dimension::Static(size)).collect()),
            ),
            context,
            location,
        )?;
        let expanded_tensor_type = lower_tensor_type(&expanded_type, context, location)?;
        let converted_elements = block
            .append_operation(stable_hlo::convert(input_values[elements_index], element_tensor_type, location)?)?
            .result(0)
            .expect("stablehlo.convert should return one result")
            .as_ref();
        let converted_scales = block
            .append_operation(stable_hlo::convert(input_values[scales_index], scale_tensor_type, location)?)?
            .result(0)
            .expect("stablehlo.convert should return one result")
            .as_ref();
        let scale_axes = (0..scale_dimensions.len()).collect::<Vec<_>>();
        let broadcasted_scales = block
            .append_operation(stable_hlo::broadcast(
                converted_scales,
                expanded_tensor_type,
                scale_axes.as_slice(),
                location,
            )?)?
            .result(0)
            .expect("stablehlo.broadcast_in_dim should return one result")
            .as_ref();
        let merged_scales = block
            .append_operation(stable_hlo::reshape(broadcasted_scales, element_dimensions.as_slice(), location)?)?
            .result(0)
            .expect("stablehlo.reshape should return one result")
            .as_ref();
        let product = block
            .append_operation(stable_hlo::multiply(converted_elements, merged_scales, location)?)?
            .result(0)
            .expect("stablehlo.multiply should return one result")
            .as_ref();
        dequantized.push(product);
    }
    let rank = input_types[0].rank();
    let dimensions = match rank {
        3 => context.stable_hlo_dot_dimensions(&[0], &[0], &[2], &[2])?,
        _ => context.stable_hlo_dot_dimensions(&[], &[], &[1], &[1])?,
    };
    let output_tensor_type = lower_tensor_type(&output_types[0], context, location)?;
    let result = block.append_operation(stable_hlo::dot_general(
        dequantized[0],
        dequantized[1],
        dimensions,
        Some((Precision::Default, Precision::Default)),
        None,
        output_tensor_type,
        location,
    )?)?;
    let mut result = result.result(0).expect("stablehlo.dot_general should return one result").as_ref();
    if let Some(global_scale) = input_values.get(4) {
        let broadcasted_global_scale = block
            .append_operation(stable_hlo::broadcast(*global_scale, output_tensor_type, &[], location)?)?
            .result(0)
            .expect("stablehlo.broadcast_in_dim should return one result")
            .as_ref();
        result = block
            .append_operation(stable_hlo::multiply(result, broadcasted_global_scale, location)?)?
            .result(0)
            .expect("stablehlo.multiply should return one result")
            .as_ref();
    }
    Ok(vec![result])
}

/// Static `[batch, sequence, heads, head_dim]`-style dimensions of one attention operand type, rejecting dynamic
/// shapes and any rank other than 4 for the attention lowerings.
fn attention_static_dimensions(input_type: &ArrayType) -> Result<[usize; 4], LoweringError> {
    input_type
        .static_shape()
        .and_then(|shape| <[usize; 4]>::try_from(shape.dimensions()).ok())
        .ok_or_else(|| LoweringError::UnsupportedOp { op: "dynamically shaped dot_product_attention".to_string() })
}

/// Returns the static [`ArrayType`] with the provided data type and dimensions used by the attention lowerings.
fn attention_array_type(data_type: DataType, dimensions: &[usize]) -> ArrayType {
    ArrayType::new(data_type, Shape::new(dimensions.iter().map(|&size| Dimension::Static(size)).collect()))
}

/// The cuDNN fMHA proto-JSON dot-dimension-number block of the two forward attention matrix products under the
/// `BTNH` operand convention (`bmm1 = Q·Kᵀ` contracting the head axis, `bmm2 = P·V` contracting the key/value
/// sequence axis), exactly as validated on hardware.
const FMHA_FORWARD_DOT_DIMENSION_NUMBERS: &str = "\"bmm1_dot_dimension_numbers\":\
     {\"lhs_contracting_dimensions\":[\"3\"],\"rhs_contracting_dimensions\":[\"3\"],\
     \"lhs_batch_dimensions\":[\"0\",\"2\"],\"rhs_batch_dimensions\":[\"0\",\"2\"]},\
     \"bmm2_dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"3\"],\
     \"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[\"0\",\"1\"],\
     \"rhs_batch_dimensions\":[\"0\",\"2\"]}";

/// The cuDNN fMHA proto-JSON dot-dimension-number block of the four backward gradient matrix products
/// (`bmm1_grad_gemm1 = dS·K`, `bmm1_grad_gemm2 = dSᵀ·Q`, `bmm2_grad_gemm1 = dO·Vᵀ`, `bmm2_grad_gemm2 = Pᵀ·dO`),
/// exactly as validated on hardware. These replace the forward `bmm1`/`bmm2` blocks in the backward call's
/// configuration.
const FMHA_BACKWARD_DOT_DIMENSION_NUMBERS: &str = "\"bmm1_grad_gemm1_dot_dimension_numbers\":\
     {\"lhs_contracting_dimensions\":[\"2\"],\"rhs_contracting_dimensions\":[\"1\"],\
     \"lhs_batch_dimensions\":[\"0\",\"1\"],\"rhs_batch_dimensions\":[\"0\",\"2\"]},\
     \"bmm1_grad_gemm2_dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"3\"],\
     \"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[\"0\",\"1\"],\
     \"rhs_batch_dimensions\":[\"0\",\"2\"]},\
     \"bmm2_grad_gemm1_dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"2\"],\
     \"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[\"0\",\"1\"],\
     \"rhs_batch_dimensions\":[\"0\",\"2\"]},\
     \"bmm2_grad_gemm2_dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"3\"],\
     \"rhs_contracting_dimensions\":[\"3\"],\"lhs_batch_dimensions\":[\"0\",\"2\"],\
     \"rhs_batch_dimensions\":[\"0\",\"2\"]}";

/// Renders the `GpuBackendConfig` proto-JSON string of one cuDNN fMHA custom call — the raw backend-config form
/// (not typed-FFI dictionaries) that XLA's cuDNN custom-call compiler parses — with the score scale, the
/// intermediate `[batch, q_heads, q_seq, kv_seq]` score shape at the operand element type (load-bearing on the
/// backward path, where the `P` descriptor and statistic strides derive from it), the mask kind, the
/// forward or backward dot-dimension-number block, the dropout rate and seed (the hardware-validated defaults
/// `0.0`/`42` when dropout is off), and the sliding-window length.
fn fmha_backend_config(
    element_type: &str,
    intermediate_dimensions: [usize; 4],
    scale: f64,
    mask_type: &str,
    dot_dimension_numbers: &str,
    dropout: Option<(f64, u64)>,
    sliding_window_length: usize,
) -> String {
    let [batch, heads, query_sequence, key_value_sequence] = intermediate_dimensions;
    // The `{:?}` rate formatting keeps the validated `0.0` spelling for the off state while rendering set rates
    // with their shortest round-trip form.
    let dropout_rate = dropout.map_or(0.0, |(rate, _)| rate);
    let seed = dropout.map_or(42, |(_, seed)| seed);
    format!(
        "{{\"operation_queue_id\":\"0\",\"cudnn_fmha_backend_config\":{{\
         \"algorithm\":{{\"algo_id\":\"0\",\"math_type\":\"TENSOR_OP_MATH\",\"tuning_knobs\":\
         {{\"17\":\"1\",\"24\":\"0\"}},\"is_cudnn_frontend\":true,\"workspace_size\":\"0\"}},\
         \"fmha_scale\":{scale},\"intermediate_tensor_shape\":{{\"element_type\":\"{element_type}\",\
         \"dimensions\":[\"{batch}\",\"{heads}\",\"{query_sequence}\",\"{key_value_sequence}\"],\
         \"tuple_shapes\":[],\"layout\":{{\"dim_level_types\":[],\"dim_unique\":[],\"dim_ordered\":[],\
         \"minor_to_major\":[\"3\",\"2\",\"1\",\"0\"],\"tiles\":[],\"element_size_in_bits\":\"0\",\
         \"memory_space\":\"0\",\"index_primitive_type\":\"PRIMITIVE_TYPE_INVALID\",\
         \"pointer_primitive_type\":\"PRIMITIVE_TYPE_INVALID\",\
         \"dynamic_shape_metadata_prefix_bytes\":\"0\"}},\
         \"is_dynamic_dimension\":[false,false,false,false]}},\
         \"is_flash_attention\":true,\"mask_type\":\"{mask_type}\",\
         {dot_dimension_numbers},\
         \"dropout_rate\":{dropout_rate:?},\"seed\":{seed},\"sliding_window_length\":{sliding_window_length},\
         \"max_seg_per_batch\":1,\"is_paged_attention\":false}}}}",
    )
}

/// Returns the cuDNN fMHA custom-call target name for the provided feature set: the name varies only with the bias
/// operand and dropout (`__cudnn$fmha[ScaleBias]Softmax[Dropout][Backward]`) — padding, sliding windows, and
/// grouped-query head counts are carried by the backend configuration and the operand shapes instead.
fn fmha_target_name(has_bias: bool, has_dropout: bool, backward: bool) -> String {
    format!(
        "__cudnn$fmha{}Softmax{}{}",
        if has_bias { "ScaleBias" } else { "" },
        if has_dropout { "Dropout" } else { "" },
        if backward { "Backward" } else { "" },
    )
}

/// Returns the cuDNN fMHA `mask_type` configuration value for the provided built-in mask and sequence-length
/// presence: the padding mask kinds compose the built-in mask with the appended `i32[batch]` sequence-length
/// operands.
fn fmha_mask_type(mask: AttentionMask, has_sequence_lengths: bool) -> &'static str {
    match (mask, has_sequence_lengths) {
        (AttentionMask::None, false) => "NO_MASK",
        (AttentionMask::Causal, false) => "CAUSAL",
        (AttentionMask::None, true) => "PADDING",
        (AttentionMask::Causal, true) => "PADDING_CAUSAL",
    }
}

/// Whether the attention operations qualify for the fused cuDNN fMHA custom calls: a CUDA target, `bf16`/`f16`
/// operands, and a head dimension that is a multiple of 8 (cuDNN's compile-time gate).
fn fmha_fast_path_qualifies(
    collective_state: &CollectiveLoweringState,
    data_type: DataType,
    head_dimension: usize,
) -> bool {
    collective_state.target_platform().is_some_and(|platform| platform == "cuda")
        && matches!(data_type, DataType::BF16 | DataType::F16)
        && head_dimension % 8 == 0
}

/// Mirrors [`expand_key_value_heads`](ryft_core::operations::attention) for the StableHLO composition fallbacks: a
/// grouped `[b, s, kv_heads, h]` key/value operand broadcasts to `[b, s, kv_heads, group, h]` and reshapes to
/// `[b, s, heads, h]`, so each key/value head repeats `group` times consecutively and query head `i` attends
/// key/value head `i / group`. Operands that already carry one key/value head per query head pass through.
#[allow(clippy::too_many_arguments)]
fn lower_attention_expand_key_value_heads<'b, 'c: 'b, 't: 'c>(
    operand: ValueRef<'b, 'c, 't>,
    data_type: DataType,
    [batch, key_value_sequence, key_value_heads, head_dimension]: [usize; 4],
    heads: usize,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    if key_value_heads == heads {
        return Ok(operand);
    }
    let group = heads / key_value_heads;
    let expanded_tensor_type = lower_tensor_type(
        &attention_array_type(data_type, &[batch, key_value_sequence, key_value_heads, group, head_dimension]),
        context,
        location,
    )?;
    let expanded = block
        .append_operation(stable_hlo::broadcast(operand, expanded_tensor_type, &[0, 1, 2, 4], location)?)?
        .result(0)
        .expect("stablehlo.broadcast_in_dim should return one result")
        .as_ref();
    let reshaped = block.append_operation(stable_hlo::reshape(
        expanded,
        &[batch, key_value_sequence, heads, head_dimension],
        location,
    )?)?;
    Ok(reshaped.result(0).expect("stablehlo.reshape should return one result").as_ref())
}

/// Mirrors [`apply_attention_masks`](ryft_core::operations::attention) for the StableHLO composition fallbacks:
/// causal visibility (`column <= row`), the optional sliding-window lower bound (`column > row - window`), and the
/// optional key/value sequence-length column exclusion (`column < key_value_sequence_lengths[b]`, with the
/// `i32[batch]` lengths broadcast per batch item), replacing masked positions with `-1e30`.
#[allow(clippy::too_many_arguments)]
fn lower_attention_masks<'b, 'c: 'b, 't: 'c>(
    scores: ValueRef<'b, 'c, 't>,
    mask: AttentionMask,
    sliding_window: Option<usize>,
    key_value_sequence_lengths: Option<ValueRef<'b, 'c, 't>>,
    softmax_scores_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    if mask == AttentionMask::None && key_value_sequence_lengths.is_none() {
        return Ok(scores);
    }
    let score_dimensions = softmax_scores_type.static_shape().expect("attention scores have a static shape");
    let index_type = attention_array_type(DataType::I32, score_dimensions.dimensions());
    let index_tensor_type = lower_tensor_type(&index_type, context, location)?;
    let columns = block
        .append_operation(stable_hlo::iota(index_tensor_type, 3, location)?)?
        .result(0)
        .expect("stablehlo.iota should return one result")
        .as_ref();
    let mut visible = None;
    if mask == AttentionMask::Causal {
        // A score position is visible when its column (key/value) index does not exceed its row (query) index, and
        // a sliding window additionally requires `column > row - window`.
        let rows = block
            .append_operation(stable_hlo::iota(index_tensor_type, 2, location)?)?
            .result(0)
            .expect("stablehlo.iota should return one result")
            .as_ref();
        let mut causal_visible =
            lower_compare_to_mlir(ComparisonDirection::LessThanOrEqual, columns, rows, block, location)?;
        if let Some(window) = sliding_window {
            let window_splat =
                lower_f64_constant_splat(window as f64, &index_type, index_tensor_type, block, context, location)?;
            let lower_bound = block
                .append_operation(stable_hlo::subtract(rows, window_splat, location)?)?
                .result(0)
                .expect("stablehlo.subtract should return one result")
                .as_ref();
            let in_window =
                lower_compare_to_mlir(ComparisonDirection::GreaterThan, columns, lower_bound, block, location)?;
            causal_visible = block
                .append_operation(stable_hlo::and(causal_visible, in_window, location)?)?
                .result(0)
                .expect("stablehlo.and should return one result")
                .as_ref();
        }
        visible = Some(causal_visible);
    }
    if let Some(lengths) = key_value_sequence_lengths {
        // The `[batch]` lengths broadcast against the `[batch, heads, q_seq, kv_seq]` column indices.
        let bounds = block
            .append_operation(stable_hlo::broadcast(lengths, index_tensor_type, &[0], location)?)?
            .result(0)
            .expect("stablehlo.broadcast_in_dim should return one result")
            .as_ref();
        let in_range = lower_compare_to_mlir(ComparisonDirection::LessThan, columns, bounds, block, location)?;
        visible = Some(match visible {
            None => in_range,
            Some(visible) => block
                .append_operation(stable_hlo::and(visible, in_range, location)?)?
                .result(0)
                .expect("stablehlo.and should return one result")
                .as_ref(),
        });
    }
    let softmax_scores_tensor_type = lower_tensor_type(softmax_scores_type, context, location)?;
    let masked =
        lower_f64_constant_splat(-1.0e30, softmax_scores_type, softmax_scores_tensor_type, block, context, location)?;
    // At least one mask contributed a visibility condition given the early return above.
    let selected = block.append_operation(stable_hlo::select(visible.unwrap(), scores, masked, location)?)?;
    Ok(selected.result(0).expect("stablehlo.select should return one result").as_ref())
}

/// Mirrors [`attention_logits`](ryft_core::operations::attention) for the StableHLO composition fallbacks: the
/// `query · expanded-keyᵀ` scores per batch item and head (`[b, n, t, s]`, contracting the head axis with batch
/// dimensions `[0, 2]` on both sides) at the operand type, converted to the softmax type, scaled, shifted by the
/// optional broadcast bias (converted to the softmax type alongside the scores), and masked via
/// [`lower_attention_masks`]. Returns the masked logits together with their softmax-typed [`ArrayType`].
#[allow(clippy::too_many_arguments)]
fn lower_attention_logits<'b, 'c: 'b, 't: 'c>(
    query: ValueRef<'b, 'c, 't>,
    expanded_key: ValueRef<'b, 'c, 't>,
    bias: Option<(ValueRef<'b, 'c, 't>, &ArrayType)>,
    key_value_sequence_lengths: Option<ValueRef<'b, 'c, 't>>,
    scale: f64,
    mask: AttentionMask,
    sliding_window: Option<usize>,
    data_type: DataType,
    score_dimensions: [usize; 4],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(ValueRef<'b, 'c, 't>, ArrayType), LoweringError> {
    let softmax_type = if data_type == DataType::F64 { DataType::F64 } else { DataType::F32 };
    let scores_tensor_type = lower_tensor_type(&attention_array_type(data_type, &score_dimensions), context, location)?;
    let softmax_scores_type = attention_array_type(softmax_type, &score_dimensions);
    let softmax_scores_tensor_type = lower_tensor_type(&softmax_scores_type, context, location)?;
    // Scores over `[batch, heads]`: `query [b, t, n, h] · key [b, s, n, h]` contracting `h` -> `[b, n, t, s]`.
    let scores = block.append_operation(stable_hlo::dot_general(
        query,
        expanded_key,
        context.stable_hlo_dot_dimensions(&[0, 2], &[0, 2], &[3], &[3])?,
        Some((Precision::Default, Precision::Default)),
        None,
        scores_tensor_type,
        location,
    )?)?;
    let mut scores = scores.result(0).expect("stablehlo.dot_general should return one result").as_ref();
    if data_type != softmax_type {
        scores = block
            .append_operation(stable_hlo::convert(scores, softmax_scores_tensor_type, location)?)?
            .result(0)
            .expect("stablehlo.convert should return one result")
            .as_ref();
    }
    let scale =
        lower_f64_constant_splat(scale, &softmax_scores_type, softmax_scores_tensor_type, block, context, location)?;
    scores = block
        .append_operation(stable_hlo::multiply(scores, scale, location)?)?
        .result(0)
        .expect("stablehlo.multiply should return one result")
        .as_ref();
    if let Some((bias, bias_type)) = bias {
        // The bias converts to the softmax type at its own (possibly broadcast) shape and then broadcasts against
        // the scaled scores.
        let mut bias = bias;
        if bias_type.data_type() != softmax_type {
            let softmax_bias_type =
                lower_tensor_type(&ArrayType::new(softmax_type, bias_type.shape().clone()), context, location)?;
            bias = block
                .append_operation(stable_hlo::convert(bias, softmax_bias_type, location)?)?
                .result(0)
                .expect("stablehlo.convert should return one result")
                .as_ref();
        }
        let broadcast_bias = block
            .append_operation(stable_hlo::broadcast(bias, softmax_scores_tensor_type, &[0, 1, 2, 3], location)?)?
            .result(0)
            .expect("stablehlo.broadcast_in_dim should return one result")
            .as_ref();
        scores = block
            .append_operation(stable_hlo::add(scores, broadcast_bias, location)?)?
            .result(0)
            .expect("stablehlo.add should return one result")
            .as_ref();
    }
    let logits = lower_attention_masks(
        scores,
        mask,
        sliding_window,
        key_value_sequence_lengths,
        &softmax_scores_type,
        block,
        context,
        location,
    )?;
    Ok((logits, softmax_scores_type))
}

/// Mirrors [`zero_out_of_range_query_rows`](ryft_core::operations::attention) for the StableHLO composition
/// fallbacks: replaces the rows of `value` whose query index along `row_axis` is at or beyond
/// `query_sequence_lengths[b]` with exact zeros, matching XLA memzeroing every fMHA output.
#[allow(clippy::too_many_arguments)]
fn lower_attention_zero_out_of_range_query_rows<'b, 'c: 'b, 't: 'c>(
    value: ValueRef<'b, 'c, 't>,
    value_type: &ArrayType,
    query_sequence_lengths: ValueRef<'b, 'c, 't>,
    row_axis: usize,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let index_type = ArrayType::new(DataType::I32, value_type.shape().clone());
    let index_tensor_type = lower_tensor_type(&index_type, context, location)?;
    let rows = block
        .append_operation(stable_hlo::iota(index_tensor_type, row_axis, location)?)?
        .result(0)
        .expect("stablehlo.iota should return one result")
        .as_ref();
    let bounds = block
        .append_operation(stable_hlo::broadcast(query_sequence_lengths, index_tensor_type, &[0], location)?)?
        .result(0)
        .expect("stablehlo.broadcast_in_dim should return one result")
        .as_ref();
    let in_range = lower_compare_to_mlir(ComparisonDirection::LessThan, rows, bounds, block, location)?;
    let value_tensor_type = lower_tensor_type(value_type, context, location)?;
    let zero = lower_f64_constant_splat(0.0, value_type, value_tensor_type, block, context, location)?;
    let selected = block.append_operation(stable_hlo::select(in_range, value, zero, location)?)?;
    Ok(selected.result(0).expect("stablehlo.select should return one result").as_ref())
}

/// Lowers one traced dot-product attention. On CUDA targets with `bf16`/`f16` operands and a head dimension that is
/// a multiple of 8 (cuDNN's compile-time gate), it emits the fused cuDNN flash-attention custom call — target
/// `__cudnn$fmha[ScaleBias]Softmax[Dropout]` (the name varies only with the bias operand and dropout) — with the
/// hardware-validated legacy contract: proto-JSON `backend_config` (mask kind `NO_MASK`/`CAUSAL`/`PADDING`/
/// `PADDING_CAUSAL`, sliding-window length, dropout rate and seed, and the `[b, n_q, t, s]` intermediate score
/// shape), `api_version = 2`, operands `(Q, K, V[, bias][, q_seqlen, kv_seqlen])` with `[3, 2, 1, 0]` layouts
/// (`dense<0>` for the `i32[batch]` sequence lengths), and results `(out [b, n, t, h] {3, 1, 2, 0}[, activation
/// f32[b, n, t] {2, 1, 0}], u8[0])` — the training form adds the activation statistic result exactly when the
/// operation requests it. The attention output transposes back to the logical `BTNH` layout, which compiles to a
/// pure bitcast given the declared result layout. Grouped-query attention needs no special handling: the key/value
/// operands carry their own head count while the configuration keeps the query head count through the intermediate
/// shape.
///
/// Everywhere else it inlines the portable StableHLO composition matching the reference semantics of
/// [`DotProductAttentionOperation`] helper-for-helper (grouped-head expansion, masked logits, max-stabilized
/// softmax, activation statistic, and padded-row zeroing) — except dropout, which only the fused kernels implement,
/// so a dropout-carrying operation that misses the fast-path gate reports an explicit error instead of silently
/// computing dropout-free attention.
#[allow(clippy::too_many_arguments)]
fn lower_dot_product_attention_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &DotProductAttentionOperation,
    collective_state: &CollectiveLoweringState,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    if !matches!(input_values.len(), 3..=6) || input_values.len() != input_types.len() {
        return Err(ProgramError::InvalidInputCount { expected: 3, actual: input_values.len() }.into());
    }
    let expected_output_count = if operation.activation_output() { 2 } else { 1 };
    check_count!("output", output_types, expected_output_count, ProgramError);
    let has_bias = matches!(input_values.len(), 4 | 6);
    let has_sequence_lengths = matches!(input_values.len(), 5 | 6);
    let data_type = input_types[0].data_type();
    let [batch, query_sequence, heads, head_dimension] = attention_static_dimensions(&input_types[0])?;
    let [_, key_value_sequence, key_value_heads, _] = attention_static_dimensions(&input_types[1])?;
    if fmha_fast_path_qualifies(collective_state, data_type, head_dimension) {
        let element_type = if data_type == DataType::BF16 { "BF16" } else { "F16" };
        let backend_config = fmha_backend_config(
            element_type,
            [batch, heads, query_sequence, key_value_sequence],
            operation.scale(),
            fmha_mask_type(operation.mask(), has_sequence_lengths),
            FMHA_FORWARD_DOT_DIMENSION_NUMBERS,
            operation.dropout(),
            operation.sliding_window().unwrap_or(0),
        );
        // The operand order matches the traced operand order exactly: the bias sits after the value and the
        // `i32[batch]` sequence lengths trail. The workspace result is declared at size zero (the compiler resizes
        // it as needed), and the training form inserts the `f32[b, n, t]` activation statistic before it.
        let mut operand_layouts = vec![vec![3, 2, 1, 0]; if has_bias { 4 } else { 3 }];
        if has_sequence_lengths {
            operand_layouts.extend([vec![0], vec![0]]);
        }
        let attended_type = attention_array_type(data_type, &[batch, heads, query_sequence, head_dimension]);
        let mut custom_call_output_types = vec![lower_tensor_type(&attended_type, context, location)?];
        let mut result_layouts = vec![vec![3, 1, 2, 0]];
        if operation.activation_output() {
            let activation_type = attention_array_type(DataType::F32, &[batch, heads, query_sequence]);
            custom_call_output_types.push(lower_tensor_type(&activation_type, context, location)?);
            result_layouts.push(vec![2, 1, 0]);
        }
        custom_call_output_types.push(lower_tensor_type(&attention_array_type(DataType::U8, &[0]), context, location)?);
        result_layouts.push(vec![0]);
        let custom_call = block.append_operation(stable_hlo::custom_call(
            input_values,
            fmha_target_name(has_bias, operation.dropout().is_some(), false).as_str(),
            false,
            Some(context.string_attribute(backend_config.as_str()).as_ref()),
            CustomCallApiVersion::StatusReturning,
            &[],
            Some(CustomCallMemoryLayouts { operands: operand_layouts, results: result_layouts }),
            &[],
            None,
            &custom_call_output_types,
            location,
        )?)?;
        let attended =
            custom_call.result(0).expect("stablehlo.custom_call should return the attention output").as_ref();
        let result = block.append_operation(stable_hlo::transpose(attended, &[0, 2, 1, 3], location)?)?;
        let mut results = vec![result.result(0).expect("stablehlo.transpose should return one result").as_ref()];
        if operation.activation_output() {
            results.push(
                custom_call
                    .result(1)
                    .expect("stablehlo.custom_call should return the activation statistic")
                    .as_ref(),
            );
        }
        return Ok(results);
    }
    if operation.dropout().is_some() {
        return Err(LoweringError::UnsupportedOp {
            op: "'dot_product_attention' dropout is only supported by the fused CUDA lowering".to_string(),
        });
    }

    // Portable fallback mirroring the reference composition helper-for-helper: grouped key/value heads expand to
    // one head per query head, the masked logits run at the softmax type, a max-stabilized softmax over the
    // key/value sequence axis recovers the weights, the context contraction transposes back to `BTNH`, and the
    // optional activation statistic and padded-row zeroing follow the core semantics exactly.
    let softmax_type = if data_type == DataType::F64 { DataType::F64 } else { DataType::F32 };
    let key_value_dimensions = [batch, key_value_sequence, key_value_heads, head_dimension];
    let expanded_key = lower_attention_expand_key_value_heads(
        input_values[1],
        data_type,
        key_value_dimensions,
        heads,
        block,
        context,
        location,
    )?;
    let expanded_value = lower_attention_expand_key_value_heads(
        input_values[2],
        data_type,
        key_value_dimensions,
        heads,
        block,
        context,
        location,
    )?;
    let sequence_lengths =
        has_sequence_lengths.then(|| (input_values[input_values.len() - 2], input_values[input_values.len() - 1]));
    let bias = has_bias.then(|| (input_values[3], &input_types[3]));
    let score_dimensions = [batch, heads, query_sequence, key_value_sequence];
    let (logits, softmax_scores_type) = lower_attention_logits(
        input_values[0],
        expanded_key,
        bias,
        sequence_lengths.map(|(_, key_value_lengths)| key_value_lengths),
        operation.scale(),
        operation.mask(),
        operation.sliding_window(),
        data_type,
        score_dimensions,
        block,
        context,
        location,
    )?;
    let softmax_scores_tensor_type = lower_tensor_type(&softmax_scores_type, context, location)?;
    // Max-stabilized softmax over the key/value sequence (last) axis.
    let reduced_type = attention_array_type(softmax_type, &[batch, heads, query_sequence]);
    let score_axes: &[usize] = &[0, 1, 2];
    let maxima = lower_reduce_to_mlir(ReductionKind::Max, &[3], logits, &reduced_type, block, context, location)?;
    let broadcast_maxima = block
        .append_operation(stable_hlo::broadcast(maxima, softmax_scores_tensor_type, score_axes, location)?)?
        .result(0)
        .expect("stablehlo.broadcast_in_dim should return one result")
        .as_ref();
    let shifted = block
        .append_operation(stable_hlo::subtract(logits, broadcast_maxima, location)?)?
        .result(0)
        .expect("stablehlo.subtract should return one result")
        .as_ref();
    let exponentials = block
        .append_operation(stable_hlo::exponential(shifted, Accuracy::Default, location)?)?
        .result(0)
        .expect("stablehlo.exponential should return one result")
        .as_ref();
    let sums = lower_reduce_to_mlir(ReductionKind::Sum, &[3], exponentials, &reduced_type, block, context, location)?;
    let broadcast_sums = block
        .append_operation(stable_hlo::broadcast(sums, softmax_scores_tensor_type, score_axes, location)?)?
        .result(0)
        .expect("stablehlo.broadcast_in_dim should return one result")
        .as_ref();
    let mut weights = block
        .append_operation(stable_hlo::divide(exponentials, broadcast_sums, location)?)?
        .result(0)
        .expect("stablehlo.divide should return one result")
        .as_ref();
    if data_type != softmax_type {
        let scores_tensor_type =
            lower_tensor_type(&attention_array_type(data_type, &score_dimensions), context, location)?;
        weights = block
            .append_operation(stable_hlo::convert(weights, scores_tensor_type, location)?)?
            .result(0)
            .expect("stablehlo.convert should return one result")
            .as_ref();
    }
    // Context values: `weights [b, n, t, s] · value [b, s, n, h]` contracting `s` -> `[b, n, t, h]`, then
    // transposed back to the `BTNH` output layout `[b, t, n, h]`.
    let attended_tensor_type = lower_tensor_type(
        &attention_array_type(data_type, &[batch, heads, query_sequence, head_dimension]),
        context,
        location,
    )?;
    let attended = block.append_operation(stable_hlo::dot_general(
        weights,
        expanded_value,
        context.stable_hlo_dot_dimensions(&[0, 1], &[0, 2], &[3], &[1])?,
        Some((Precision::Default, Precision::Default)),
        None,
        attended_tensor_type,
        location,
    )?)?;
    let attended = attended.result(0).expect("stablehlo.dot_general should return one result").as_ref();
    let transposed = block.append_operation(stable_hlo::transpose(attended, &[0, 2, 1, 3], location)?)?;
    let mut output = transposed.result(0).expect("stablehlo.transpose should return one result").as_ref();
    if let Some((query_lengths, _)) = sequence_lengths {
        output = lower_attention_zero_out_of_range_query_rows(
            output,
            &input_types[0],
            query_lengths,
            1,
            block,
            context,
            location,
        )?;
    }
    let mut results = vec![output];
    if operation.activation_output() {
        // The log-sum-exp statistic reuses the softmax reductions: `stat = max + ln(sum)` rowwise over the kv axis,
        // always produced at `f32`.
        let logarithms = block
            .append_operation(stable_hlo::log(sums, Accuracy::Default, location)?)?
            .result(0)
            .expect("stablehlo.log should return one result")
            .as_ref();
        let mut statistic = block
            .append_operation(stable_hlo::add(maxima, logarithms, location)?)?
            .result(0)
            .expect("stablehlo.add should return one result")
            .as_ref();
        let activation_type = attention_array_type(DataType::F32, &[batch, heads, query_sequence]);
        if softmax_type != DataType::F32 {
            let activation_tensor_type = lower_tensor_type(&activation_type, context, location)?;
            statistic = block
                .append_operation(stable_hlo::convert(statistic, activation_tensor_type, location)?)?
                .result(0)
                .expect("stablehlo.convert should return one result")
                .as_ref();
        }
        if let Some((query_lengths, _)) = sequence_lengths {
            statistic = lower_attention_zero_out_of_range_query_rows(
                statistic,
                &activation_type,
                query_lengths,
                2,
                block,
                context,
                location,
            )?;
        }
        results.push(statistic);
    }
    Ok(results)
}

/// Lowers one traced dot-product attention backward pass. Under the same gate as the forward fast path (a CUDA
/// target, `bf16`/`f16` operands, and a head dimension that is a multiple of 8) it emits the fused
/// `__cudnn$fmha[ScaleBias]Softmax[Dropout]Backward` custom call with the hardware-validated contract: the traced
/// operand order `(q, k, v[, bias], output, activation, output_cotangent[, q_seqlen, kv_seqlen])` reorders to the
/// kernel's `(Q, K, V, activation, dO[, bias], O[, q_seqlen, kv_seqlen])` call order (the bias sits between `dO`
/// and `O`), the backend configuration swaps in the four gradient-GEMM dot-dimension-number blocks and keeps the
/// operand-typed `[b, n_q, t, s]` intermediate score shape (load-bearing: the `P` descriptor and statistic strides
/// derive from it), and the results are `(dQ [b, n_q, t, h], dK [b, n_kv, s, h], dV [b, n_kv, s, h][, dBias],
/// u8[0])` with `{3, 1, 2, 0}` gradient layouts — each gradient transposes back to `BTNH`/`BSNH` (a pure bitcast)
/// while the bias cotangent keeps its own shape and default layout.
///
/// Everywhere else it inlines the portable StableHLO composition mirroring
/// [`dot_product_attention_backward_composition`](ryft_core::operations::attention): the masked logits are
/// recomputed, the weights recover as `P = exp(S - stat)`, the four documented `dot_general`s produce the
/// cotangents at the softmax type, grouped-query attention sums the key/value cotangents over the per-head group
/// axis, the bias cotangent sums over the bias's broadcast leading dimensions, and variable sequence lengths zero
/// the out-of-range output-cotangent and query-cotangent rows. Dropout outside the fast path reports an explicit
/// error because only the fused kernels implement it.
#[allow(clippy::too_many_arguments)]
fn lower_dot_product_attention_backward_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &DotProductAttentionBackwardOperation,
    collective_state: &CollectiveLoweringState,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    if !matches!(input_values.len(), 6..=9) || input_values.len() != input_types.len() {
        return Err(ProgramError::InvalidInputCount { expected: 6, actual: input_values.len() }.into());
    }
    let has_bias = matches!(input_values.len(), 7 | 9);
    let has_sequence_lengths = matches!(input_values.len(), 8 | 9);
    check_count!("output", output_types, if has_bias { 4 } else { 3 }, ProgramError);
    let data_type = input_types[0].data_type();
    let [batch, query_sequence, heads, head_dimension] = attention_static_dimensions(&input_types[0])?;
    let [_, key_value_sequence, key_value_heads, _] = attention_static_dimensions(&input_types[1])?;
    let offset = if has_bias { 4 } else { 3 };
    let (output, activation, output_cotangent) =
        (input_values[offset], input_values[offset + 1], input_values[offset + 2]);
    let sequence_lengths =
        has_sequence_lengths.then(|| (input_values[input_values.len() - 2], input_values[input_values.len() - 1]));
    if fmha_fast_path_qualifies(collective_state, data_type, head_dimension) {
        let element_type = if data_type == DataType::BF16 { "BF16" } else { "F16" };
        let backend_config = fmha_backend_config(
            element_type,
            [batch, heads, query_sequence, key_value_sequence],
            operation.scale(),
            fmha_mask_type(operation.mask(), has_sequence_lengths),
            FMHA_BACKWARD_DOT_DIMENSION_NUMBERS,
            operation.dropout(),
            operation.sliding_window().unwrap_or(0),
        );
        // The kernel call order differs from the traced operand order: `(Q, K, V, activation, dO[, bias], O
        // [, q_seqlen, kv_seqlen])`, with the bias between the output cotangent and the forward output.
        let mut operands = vec![input_values[0], input_values[1], input_values[2], activation, output_cotangent];
        let mut operand_layouts =
            vec![vec![3, 2, 1, 0], vec![3, 2, 1, 0], vec![3, 2, 1, 0], vec![2, 1, 0], vec![3, 2, 1, 0]];
        if has_bias {
            operands.push(input_values[3]);
            operand_layouts.push(vec![3, 2, 1, 0]);
        }
        operands.push(output);
        operand_layouts.push(vec![3, 2, 1, 0]);
        if let Some((query_lengths, key_value_lengths)) = sequence_lengths {
            operands.extend([query_lengths, key_value_lengths]);
            operand_layouts.extend([vec![0], vec![0]]);
        }
        let query_gradient_type = attention_array_type(data_type, &[batch, heads, query_sequence, head_dimension]);
        let key_value_gradient_type =
            attention_array_type(data_type, &[batch, key_value_heads, key_value_sequence, head_dimension]);
        let mut custom_call_output_types = vec![
            lower_tensor_type(&query_gradient_type, context, location)?,
            lower_tensor_type(&key_value_gradient_type, context, location)?,
            lower_tensor_type(&key_value_gradient_type, context, location)?,
        ];
        let mut result_layouts = vec![vec![3, 1, 2, 0], vec![3, 1, 2, 0], vec![3, 1, 2, 0]];
        if has_bias {
            custom_call_output_types.push(lower_tensor_type(&input_types[3], context, location)?);
            result_layouts.push(vec![3, 2, 1, 0]);
        }
        custom_call_output_types.push(lower_tensor_type(&attention_array_type(DataType::U8, &[0]), context, location)?);
        result_layouts.push(vec![0]);
        let custom_call = block.append_operation(stable_hlo::custom_call(
            operands.as_slice(),
            fmha_target_name(has_bias, operation.dropout().is_some(), true).as_str(),
            false,
            Some(context.string_attribute(backend_config.as_str()).as_ref()),
            CustomCallApiVersion::StatusReturning,
            &[],
            Some(CustomCallMemoryLayouts { operands: operand_layouts, results: result_layouts }),
            &[],
            None,
            &custom_call_output_types,
            location,
        )?)?;
        // Each gradient comes back in the physical `[b, n, seq, h]` layout and transposes to the logical
        // `BTNH`/`BSNH` layout (a pure bitcast given the declared result layouts); the bias cotangent keeps its
        // own shape.
        let mut results = Vec::with_capacity(if has_bias { 4 } else { 3 });
        for index in 0..3 {
            let gradient = custom_call.result(index).expect("stablehlo.custom_call should return the gradient");
            let transposed =
                block.append_operation(stable_hlo::transpose(gradient.as_ref(), &[0, 2, 1, 3], location)?)?;
            results.push(transposed.result(0).expect("stablehlo.transpose should return one result").as_ref());
        }
        if has_bias {
            results
                .push(custom_call.result(3).expect("stablehlo.custom_call should return the bias cotangent").as_ref());
        }
        return Ok(results);
    }
    if operation.dropout().is_some() {
        return Err(LoweringError::UnsupportedOp {
            op: "'dot_product_attention_backward' dropout is only supported by the fused CUDA lowering".to_string(),
        });
    }

    // Portable fallback mirroring the reference backward composition helper-for-helper.
    let softmax_type = if data_type == DataType::F64 { DataType::F64 } else { DataType::F32 };
    let key_value_dimensions = [batch, key_value_sequence, key_value_heads, head_dimension];
    let expanded_key = lower_attention_expand_key_value_heads(
        input_values[1],
        data_type,
        key_value_dimensions,
        heads,
        block,
        context,
        location,
    )?;
    let expanded_value = lower_attention_expand_key_value_heads(
        input_values[2],
        data_type,
        key_value_dimensions,
        heads,
        block,
        context,
        location,
    )?;
    let bias = has_bias.then(|| (input_values[3], &input_types[3]));
    let score_dimensions = [batch, heads, query_sequence, key_value_sequence];
    // Recompute the masked logits exactly as the forward does and recover the attention weights from the stashed
    // log-sum-exp statistic: `P = exp(S - stat)`.
    let (logits, softmax_scores_type) = lower_attention_logits(
        input_values[0],
        expanded_key,
        bias,
        sequence_lengths.map(|(_, key_value_lengths)| key_value_lengths),
        operation.scale(),
        operation.mask(),
        operation.sliding_window(),
        data_type,
        score_dimensions,
        block,
        context,
        location,
    )?;
    let softmax_scores_tensor_type = lower_tensor_type(&softmax_scores_type, context, location)?;
    let reduced_dimensions = [batch, heads, query_sequence];
    let mut statistic = activation;
    if softmax_type != DataType::F32 {
        let statistic_tensor_type =
            lower_tensor_type(&attention_array_type(softmax_type, &reduced_dimensions), context, location)?;
        statistic = block
            .append_operation(stable_hlo::convert(statistic, statistic_tensor_type, location)?)?
            .result(0)
            .expect("stablehlo.convert should return one result")
            .as_ref();
    }
    let score_axes: &[usize] = &[0, 1, 2];
    let broadcast_statistic = block
        .append_operation(stable_hlo::broadcast(statistic, softmax_scores_tensor_type, score_axes, location)?)?
        .result(0)
        .expect("stablehlo.broadcast_in_dim should return one result")
        .as_ref();
    let shifted = block
        .append_operation(stable_hlo::subtract(logits, broadcast_statistic, location)?)?
        .result(0)
        .expect("stablehlo.subtract should return one result")
        .as_ref();
    let weights = block
        .append_operation(stable_hlo::exponential(shifted, Accuracy::Default, location)?)?
        .result(0)
        .expect("stablehlo.exponential should return one result")
        .as_ref();
    // Out-of-range query rows of the incoming output cotangent are zeroed before any contraction so the key/value
    // cotangents receive no contribution from them.
    let mut output_cotangent = output_cotangent;
    if let Some((query_lengths, _)) = sequence_lengths {
        output_cotangent = lower_attention_zero_out_of_range_query_rows(
            output_cotangent,
            &input_types[0],
            query_lengths,
            1,
            block,
            context,
            location,
        )?;
    }
    // The gradient contractions all run at the softmax data type, like the forward softmax.
    let query_dimensions = [batch, query_sequence, heads, head_dimension];
    let expanded_key_value_dimensions = [batch, key_value_sequence, heads, head_dimension];
    let convert = |operand: ValueRef<'b, 'c, 't>,
                   dimensions: &[usize],
                   block: &mut BlockRef<'b, 'c, 't>|
     -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
        if data_type == softmax_type {
            return Ok(operand);
        }
        let tensor_type = lower_tensor_type(&attention_array_type(softmax_type, dimensions), context, location)?;
        Ok(block
            .append_operation(stable_hlo::convert(operand, tensor_type, location)?)?
            .result(0)
            .expect("stablehlo.convert should return one result")
            .as_ref())
    };
    let softmax_query = convert(input_values[0], &query_dimensions, block)?;
    let softmax_key = convert(expanded_key, &expanded_key_value_dimensions, block)?;
    let softmax_value = convert(expanded_value, &expanded_key_value_dimensions, block)?;
    let softmax_output = convert(output, &query_dimensions, block)?;
    let softmax_output_cotangent = convert(output_cotangent, &query_dimensions, block)?;
    // `dP[b, n, t, s] = Σ_h dO[b, t, n, h] · V[b, s, n, h]`: batch `[0, 2]/[0, 2]`, contract the head axis `3/3`.
    let weight_cotangents = block.append_operation(stable_hlo::dot_general(
        softmax_output_cotangent,
        softmax_value,
        context.stable_hlo_dot_dimensions(&[0, 2], &[0, 2], &[3], &[3])?,
        Some((Precision::Default, Precision::Default)),
        None,
        softmax_scores_tensor_type,
        location,
    )?)?;
    let weight_cotangents =
        weight_cotangents.result(0).expect("stablehlo.dot_general should return one result").as_ref();
    // `delta[b, n, t] = Σ_h dO[b, t, n, h] · O[b, t, n, h]`, transposed from `[b, t, n]` to `[b, n, t]`.
    let products = block
        .append_operation(stable_hlo::multiply(softmax_output_cotangent, softmax_output, location)?)?
        .result(0)
        .expect("stablehlo.multiply should return one result")
        .as_ref();
    let delta_type = attention_array_type(softmax_type, &[batch, query_sequence, heads]);
    let delta = lower_reduce_to_mlir(ReductionKind::Sum, &[3], products, &delta_type, block, context, location)?;
    let delta = block
        .append_operation(stable_hlo::transpose(delta, &[0, 2, 1], location)?)?
        .result(0)
        .expect("stablehlo.transpose should return one result")
        .as_ref();
    let broadcast_delta = block
        .append_operation(stable_hlo::broadcast(delta, softmax_scores_tensor_type, score_axes, location)?)?
        .result(0)
        .expect("stablehlo.broadcast_in_dim should return one result")
        .as_ref();
    // `dS = P ∘ (dP - delta)` with `delta` broadcast over the kv axis.
    let centered = block
        .append_operation(stable_hlo::subtract(weight_cotangents, broadcast_delta, location)?)?
        .result(0)
        .expect("stablehlo.subtract should return one result")
        .as_ref();
    let logit_cotangents = block
        .append_operation(stable_hlo::multiply(weights, centered, location)?)?
        .result(0)
        .expect("stablehlo.multiply should return one result")
        .as_ref();
    // The logits are `scale · (Q·Kᵀ) + bias`, so the query/key cotangents carry one extra `scale` factor while the
    // bias cotangent reads `dS` unscaled.
    let scale_splat = lower_f64_constant_splat(
        operation.scale(),
        &softmax_scores_type,
        softmax_scores_tensor_type,
        block,
        context,
        location,
    )?;
    let scaled_logit_cotangents = block
        .append_operation(stable_hlo::multiply(logit_cotangents, scale_splat, location)?)?
        .result(0)
        .expect("stablehlo.multiply should return one result")
        .as_ref();
    // `dQ[b, t, n, h] = scale · Σ_s dS[b, n, t, s] · K[b, s, n, h]`: batch `[0, 1]/[0, 2]`, contract the
    // kv-sequence axis `3/1`; the result `[b, n, t, h]` transposes to the `BTNH` layout, with out-of-range query
    // rows forced to exact zeros.
    let query_cotangent_tensor_type = lower_tensor_type(
        &attention_array_type(softmax_type, &[batch, heads, query_sequence, head_dimension]),
        context,
        location,
    )?;
    let query_cotangent = block.append_operation(stable_hlo::dot_general(
        scaled_logit_cotangents,
        softmax_key,
        context.stable_hlo_dot_dimensions(&[0, 1], &[0, 2], &[3], &[1])?,
        Some((Precision::Default, Precision::Default)),
        None,
        query_cotangent_tensor_type,
        location,
    )?)?;
    let query_cotangent = query_cotangent.result(0).expect("stablehlo.dot_general should return one result").as_ref();
    let mut query_cotangent = block
        .append_operation(stable_hlo::transpose(query_cotangent, &[0, 2, 1, 3], location)?)?
        .result(0)
        .expect("stablehlo.transpose should return one result")
        .as_ref();
    if let Some((query_lengths, _)) = sequence_lengths {
        query_cotangent = lower_attention_zero_out_of_range_query_rows(
            query_cotangent,
            &attention_array_type(softmax_type, &query_dimensions),
            query_lengths,
            1,
            block,
            context,
            location,
        )?;
    }
    // `dK[b, s, n, h] = scale · Σ_t dS[b, n, t, s] · Q[b, t, n, h]` and `dV[b, s, n, h] = Σ_t P[b, n, t, s] ·
    // dO[b, t, n, h]`: batch `[0, 1]/[0, 2]`, contract the query-sequence axis `2/1`; the results `[b, n, s, h]`
    // transpose to `[b, s, n, h]`.
    let key_value_cotangent_tensor_type = lower_tensor_type(
        &attention_array_type(softmax_type, &[batch, heads, key_value_sequence, head_dimension]),
        context,
        location,
    )?;
    let key_value_cotangent = |lhs: ValueRef<'b, 'c, 't>,
                               rhs: ValueRef<'b, 'c, 't>,
                               block: &mut BlockRef<'b, 'c, 't>|
     -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
        let cotangent = block.append_operation(stable_hlo::dot_general(
            lhs,
            rhs,
            context.stable_hlo_dot_dimensions(&[0, 1], &[0, 2], &[2], &[1])?,
            Some((Precision::Default, Precision::Default)),
            None,
            key_value_cotangent_tensor_type,
            location,
        )?)?;
        let cotangent = cotangent.result(0).expect("stablehlo.dot_general should return one result").as_ref();
        let transposed = block.append_operation(stable_hlo::transpose(cotangent, &[0, 2, 1, 3], location)?)?;
        Ok(transposed.result(0).expect("stablehlo.transpose should return one result").as_ref())
    };
    let mut key_cotangent = key_value_cotangent(scaled_logit_cotangents, softmax_query, block)?;
    let mut value_cotangent = key_value_cotangent(weights, softmax_output_cotangent, block)?;
    if key_value_heads != heads {
        // Grouped-query attention: each key/value head serves `group` consecutive query heads, so its cotangent
        // sums over the per-head group axis.
        let group = heads / key_value_heads;
        let grouped_dimensions = [batch, key_value_sequence, key_value_heads, group, head_dimension];
        let summed_type =
            attention_array_type(softmax_type, &[batch, key_value_sequence, key_value_heads, head_dimension]);
        let sum_groups = |cotangent: ValueRef<'b, 'c, 't>,
                          block: &mut BlockRef<'b, 'c, 't>|
         -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
            let grouped = block
                .append_operation(stable_hlo::reshape(cotangent, &grouped_dimensions, location)?)?
                .result(0)
                .expect("stablehlo.reshape should return one result")
                .as_ref();
            lower_reduce_to_mlir(ReductionKind::Sum, &[3], grouped, &summed_type, block, context, location)
        };
        key_cotangent = sum_groups(key_cotangent, block)?;
        value_cotangent = sum_groups(value_cotangent, block)?;
    }
    let convert_back = |cotangent: ValueRef<'b, 'c, 't>,
                        output_type: &ArrayType,
                        block: &mut BlockRef<'b, 'c, 't>|
     -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
        if data_type == softmax_type {
            return Ok(cotangent);
        }
        let tensor_type = lower_tensor_type(output_type, context, location)?;
        Ok(block
            .append_operation(stable_hlo::convert(cotangent, tensor_type, location)?)?
            .result(0)
            .expect("stablehlo.convert should return one result")
            .as_ref())
    };
    let mut results = vec![
        convert_back(query_cotangent, &output_types[0], block)?,
        convert_back(key_cotangent, &output_types[1], block)?,
        convert_back(value_cotangent, &output_types[2], block)?,
    ];
    if let Some((_, bias_type)) = bias {
        // The bias enters the logits unscaled, so its cotangent is `dS` summed over the bias's broadcast leading
        // dimensions and reshaped back to the bias shape.
        let bias_dimensions = attention_static_dimensions(bias_type)?;
        let logit_dimensions = [batch, heads];
        let reduce_axes =
            (0..2).filter(|&axis| bias_dimensions[axis] == 1 && logit_dimensions[axis] != 1).collect::<Vec<_>>();
        let mut bias_cotangent = logit_cotangents;
        if !reduce_axes.is_empty() {
            let mut summed_dimensions = score_dimensions.to_vec();
            for &axis in &reduce_axes {
                summed_dimensions[axis] = 0;
            }
            let summed_dimensions = summed_dimensions.into_iter().filter(|&size| size != 0).collect::<Vec<_>>();
            let summed_type = attention_array_type(softmax_type, summed_dimensions.as_slice());
            bias_cotangent = lower_reduce_to_mlir(
                ReductionKind::Sum,
                reduce_axes.as_slice(),
                bias_cotangent,
                &summed_type,
                block,
                context,
                location,
            )?;
        }
        let bias_shape = bias_dimensions.to_vec();
        let bias_cotangent = block
            .append_operation(stable_hlo::reshape(bias_cotangent, bias_shape.as_slice(), location)?)?
            .result(0)
            .expect("stablehlo.reshape should return one result")
            .as_ref();
        results.push(convert_back(bias_cotangent, &output_types[3], block)?);
    }
    Ok(results)
}

/// Lowers one traced custom call to a `stablehlo.custom_call` using the typed FFI calling convention
/// (`api_version = 4`): the operation's typed attributes become the `backend_config` dictionary entries (strings as
/// string attributes, Booleans as `i1` Boolean attributes, integers as signless `i64` attributes, and floats as
/// `f64` attributes — the encodings the XLA FFI decodes into typed call-frame attributes), its side-effect flag
/// becomes `has_side_effect`, and its declared output types are lowered verbatim. Handlers are resolved by the XLA
/// runtime through the target name at execution time (e.g., registered via `ryft-pjrt`'s
/// `Client::register_ffi_handler`).
fn lower_custom_call_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &CustomCallOperation,
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let attributes = operation
        .attributes()
        .iter()
        .map(|(name, value)| {
            let value = match value {
                CustomCallAttribute::String(string) => context.string_attribute(string.as_str()).as_ref(),
                CustomCallAttribute::Boolean(boolean) => context.boolean_attribute(*boolean).as_ref(),
                CustomCallAttribute::I64(integer) => {
                    context.integer_attribute(context.signless_integer_type(64), *integer).as_ref()
                }
                CustomCallAttribute::F64(float) => context.float_attribute(context.float64_type(), *float).as_ref(),
            };
            context.named_attribute(context.identifier(name.as_str()), value)
        })
        .collect::<Vec<_>>();
    let backend_config = context.dictionary_attribute(&attributes);
    let lowered_output_types = output_types
        .iter()
        .map(|output_type| lower_tensor_type(output_type, context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let lowered = block.append_operation(stable_hlo::custom_call(
        input_values,
        operation.target_name(),
        operation.has_side_effect(),
        Some(backend_config.as_ref()),
        CustomCallApiVersion::TypedFfi,
        &[],
        None,
        &[],
        None,
        &lowered_output_types,
        location,
    )?)?;
    Ok((0..output_types.len())
        .map(|index| {
            lowered
                .result(index)
                .expect("stablehlo.custom_call should return one result per declared output type")
                .as_ref()
        })
        .collect())
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for ArrayOperation<V> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
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
            ArrayOperation::Constant(constant) => {
                constant.lower_to_mlir(input_values, regions, output_types, mode, lowerer)
            }
            ArrayOperation::ConvertElementType(operation) => {
                operation.lower_to_mlir(input_values, regions, output_types, mode, lowerer)
            }
            ArrayOperation::Iota(iota) => <IotaOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                iota,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::CoordinateBasis(operation) => {
                <CoordinateBasisOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                    operation,
                    input_values,
                    regions,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            ArrayOperation::Add(operation) => <AddOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Sub(operation) => <SubOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Mul(operation) => <MulOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Div(operation) => <DivOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Neg(operation) => <NegOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Sin(operation) => <SinOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Cos(operation) => <CosOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Atan2(operation) => <Atan2Operation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Exp(operation) => <ExpOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Log(operation) => <LogOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Sqrt(operation) => <SqrtOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Rsqrt(operation) => <RsqrtOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Tanh(operation) => <TanhOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Logistic(operation) => <LogisticOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Erf(operation) => <ErfOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Pow(operation) => <PowOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Sign(operation) => <SignOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Floor(operation) => <FloorOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Ceil(operation) => <CeilOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Round(operation) => <RoundOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Max(operation) => <MaxOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Min(operation) => <MinOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Rem(operation) => <RemOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Abs(operation) => <AbsOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Complex(operation) => <ComplexOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Conjugate(operation) => <ConjugateOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Real(operation) => <RealOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Imaginary(operation) => <ImaginaryOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
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
            ArrayOperation::CustomCall(operation) => lower_custom_call_to_mlir(
                operation,
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            ArrayOperation::TransferToMemory(operation) => lower_transfer_to_memory(
                operation.destination(),
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            // Custom-derivative calls lower as their primal program: the derivative programs only exist for the
            // benefit of transforms and never reach the backend.
            ArrayOperation::CustomJvp(_) => lower_nested_program_inline(
                &regions[0],
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
                lowerer.captured_values.as_slice(),
                false,
                lowerer.nested_functions.as_ref(),
                &lowerer.collective_state,
                &mut lowerer.token,
            ),
            ArrayOperation::CustomVjp(_) => lower_nested_program_inline(
                &regions[0],
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
                lowerer.captured_values.as_slice(),
                false,
                lowerer.nested_functions.as_ref(),
                &lowerer.collective_state,
                &mut lowerer.token,
            ),
            // The opaque custom-VJP tangent carrier is a forward-mode tangent map that reverse mode transposes away
            // before lowering, so it never reaches the backend; reaching here means a forward-mode use of `custom_vjp`
            // slipped through, which is reverse-mode-only.
            ArrayOperation::CustomVjpTangent(operation) => Err(ProgramError::UnsupportedOperation {
                message: format!("operation `{}` cannot be lowered to StableHLO", operation.name(),),
            }
            .into()),
            ArrayOperation::Rematerialize(_) => lower_nested_program_inline(
                &regions[0],
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
                lowerer.captured_values.as_slice(),
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
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Dot(operation) => <DotOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Reshape(operation) => <LegacyReshapeOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                regions,
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
            ArrayOperation::Broadcast(operation) => {
                <LegacyBroadcastOperation as LowerableXlaOperation<V>>::lower_to_mlir(
                    operation,
                    input_values,
                    regions,
                    output_types,
                    mode,
                    lowerer,
                )
            }
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
            ArrayOperation::Sort(operation) => lower_sort_to_mlir(
                operation,
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            ArrayOperation::RngBitGenerator(operation) => lower_rng_bit_generator_to_mlir(
                operation,
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            ArrayOperation::ScaledDot(operation) => {
                let collective_state = lowerer.collective_state.clone();
                lower_scaled_dot_to_mlir(
                    operation,
                    &collective_state,
                    input_values,
                    &lowerer.input_types,
                    output_types,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )
            }
            ArrayOperation::DotProductAttention(operation) => {
                let collective_state = lowerer.collective_state.clone();
                lower_dot_product_attention_to_mlir(
                    operation,
                    &collective_state,
                    input_values,
                    &lowerer.input_types,
                    output_types,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )
            }
            ArrayOperation::DotProductAttentionBackward(operation) => {
                let collective_state = lowerer.collective_state.clone();
                lower_dot_product_attention_backward_to_mlir(
                    operation,
                    &collective_state,
                    input_values,
                    &lowerer.input_types,
                    output_types,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )
            }
            ArrayOperation::Compare(operation) => {
                let operand_type = comparison_operand_type(&lowerer.input_types, output_types)?;
                let [left, right] = normalize_binary_elementwise_operands(
                    input_values,
                    &[operand_type],
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?;
                let value =
                    lower_compare_to_mlir(operation.direction(), left, right, &mut lowerer.block, lowerer.location)?;
                Ok(vec![value])
            }
            ArrayOperation::Not(_) => {
                let result = lowerer.block.append_operation(stable_hlo::not(input_values[0], lowerer.location)?)?;
                Ok(vec![result.result(0).expect("stablehlo.not should return one result").as_ref()])
            }
            ArrayOperation::And(_) => {
                let [left, right] = normalize_binary_elementwise_operands(
                    input_values,
                    output_types,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?;
                let result = lowerer.block.append_operation(stable_hlo::and(left, right, lowerer.location)?)?;
                Ok(vec![result.result(0).expect("stablehlo.and should return one result").as_ref()])
            }
            ArrayOperation::Or(_) => {
                let [left, right] = normalize_binary_elementwise_operands(
                    input_values,
                    output_types,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?;
                let result = lowerer.block.append_operation(stable_hlo::or(left, right, lowerer.location)?)?;
                Ok(vec![result.result(0).expect("stablehlo.or should return one result").as_ref()])
            }
            ArrayOperation::Xor(_) => {
                let [left, right] = normalize_binary_elementwise_operands(
                    input_values,
                    output_types,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?;
                let result = lowerer.block.append_operation(stable_hlo::xor(left, right, lowerer.location)?)?;
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
            ArrayOperation::AllGather(operation) => {
                check_count!("input", input_values, 1, ProgramError);
                check_count!("output", output_types, 1, ProgramError);
                let collective_state = lowerer.collective_state.clone();
                lower_all_gather_to_mlir(
                    operation,
                    &collective_state,
                    input_values[0],
                    &output_types[0],
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )
            }
            ArrayOperation::PSumScatter(operation) => {
                check_count!("input", input_values, 1, ProgramError);
                check_count!("output", output_types, 1, ProgramError);
                let collective_state = lowerer.collective_state.clone();
                lower_psum_scatter_to_mlir(
                    operation,
                    &collective_state,
                    input_values[0],
                    &output_types[0],
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )
            }
            ArrayOperation::Ppermute(operation) => {
                check_count!("input", input_values, 1, ProgramError);
                let collective_state = lowerer.collective_state.clone();
                lower_ppermute_to_mlir(
                    operation,
                    &collective_state,
                    input_values[0],
                    &mut lowerer.block,
                    lowerer.location,
                )
            }
            ArrayOperation::AllToAll(operation) => {
                check_count!("input", input_values, 1, ProgramError);
                check_count!("output", output_types, 1, ProgramError);
                let collective_state = lowerer.collective_state.clone();
                lower_all_to_all_to_mlir(
                    operation,
                    &collective_state,
                    input_values[0],
                    &lowerer.input_types[0],
                    &output_types[0],
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )
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
                let [condition, on_true, on_false] = normalize_select_operands(
                    input_values,
                    output_types,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?;
                let result = lowerer.block.append_operation(stable_hlo::select(
                    condition,
                    on_true,
                    on_false,
                    lowerer.location,
                )?)?;
                Ok(vec![result.result(0).expect("stablehlo.select should return one result").as_ref()])
            }
            ArrayOperation::Slice(operation) => lower_slice_to_mlir(
                operation,
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
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
            ArrayOperation::Pad(operation) => lower_pad_to_mlir(
                operation,
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            ),
            ArrayOperation::Concatenate(operation) => {
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
            ArrayOperation::Condition(condition) => {
                condition.lower_to_mlir(input_values, regions, output_types, mode, lowerer)
            }
            ArrayOperation::While(while_operation) => {
                while_operation.lower_to_mlir(input_values, regions, output_types, mode, lowerer)
            }
            ArrayOperation::Scan(scan) => scan.lower_to_mlir(input_values, regions, output_types, mode, lowerer),
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

    /// PJRT platform name of the compilation target (e.g., `"cuda"` or `"cpu"`), or `None` when the lowering has
    /// no target information. Platform-gated lowerings (e.g., the block-scaled dot fast path) consult this and
    /// fall back to their portable form when it is absent.
    target_platform: Option<Rc<str>>,
}

impl CollectiveLoweringState {
    /// Creates the lowering state for one module, outside any manual region and without target information.
    pub(crate) fn new() -> Self {
        Self { channel_ids: Rc::new(Cell::new(1)), manual_shard_map: None, target_platform: None }
    }

    /// Returns a copy of this state carrying the PJRT platform name of the compilation target.
    pub(crate) fn with_target_platform(mut self, target_platform: Option<&str>) -> Self {
        self.target_platform = target_platform.map(Rc::from);
        self
    }

    /// Returns the PJRT platform name of the compilation target, or `None` when the lowering has no target
    /// information.
    pub(crate) fn target_platform(&self) -> Option<&str> {
        self.target_platform.as_deref()
    }

    /// Derives the state used inside one `sdy.manual_computation` region's body: the module's channel allocator and
    /// target platform are shared and the region's [`ShardMap`] becomes the innermost manual region.
    pub(crate) fn enter_manual_region(&self, shard_map: ShardMap) -> Self {
        Self {
            channel_ids: self.channel_ids.clone(),
            manual_shard_map: Some(Rc::new(shard_map)),
            target_platform: self.target_platform.clone(),
        }
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

    /// Declared input types of the instruction currently being lowered, in operand order.
    input_types: Vec<ArrayType>,

    /// Shared private functions emitted for deduplicated `jit_call` callees, consulted at `jit_call` lowering sites.
    /// Shared via [`Rc`] so it threads through nested lowering scopes without lifetime entanglement.
    nested_functions: Option<Rc<JitCallFunctionMap>>,

    /// Hidden capture arguments of the function currently being lowered, in capture-table order.
    captured_values: Vec<ValueRef<'b, 'c, 't>>,

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
            input_types: Vec::new(),
            nested_functions: None,
            captured_values: Vec::new(),
            token: None,
            collective_state: CollectiveLoweringState::new(),
        }
    }

    /// Attaches the declared input types of the instruction currently being lowered.
    pub(crate) fn with_input_types(mut self, input_types: Vec<ArrayType>) -> Self {
        self.input_types = input_types;
        self
    }

    /// Attaches the shared deduplicated `jit_call` functions consulted while lowering.
    pub(crate) fn with_nested_functions(mut self, nested_functions: Option<Rc<JitCallFunctionMap>>) -> Self {
        self.nested_functions = nested_functions;
        self
    }

    /// Attaches the hidden capture arguments of the function currently being lowered.
    pub(crate) fn with_captured_values(mut self, captured_values: &[ValueRef<'b, 'c, 't>]) -> Self {
        self.captured_values = captured_values.to_vec();
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
    pub(crate) fn lower_condition<V: MlirLowerableValue>(
        &mut self,
        branch_regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_condition_to_if(
            branch_regions,
            input_values,
            &mut self.block,
            self.context,
            self.location,
            self.captured_values.as_slice(),
            self.nested_functions.as_ref(),
            &self.collective_state,
            &mut self.token,
        )
    }

    /// Lowers one nested while operation inside this lowering context.
    pub(crate) fn lower_while<V: MlirLowerableValue>(
        &mut self,
        while_op: &WhileOperation,
        loop_regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        lower_while_to_while(
            while_op,
            loop_regions,
            input_values,
            &mut self.block,
            self.context,
            self.location,
            self.captured_values.as_slice(),
            self.nested_functions.as_ref(),
            &self.collective_state,
            &mut self.token,
        )
    }

    /// Lowers one nested scan operation inside this lowering context.
    pub(crate) fn lower_scan<V: MlirLowerableValue, Capture>(
        &mut self,
        scan_op: &ScanOperation<Capture>,
        scan_regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        Capture: Value<Type = ArrayType>,
    {
        let [body] = scan_regions else {
            return Err(LoweringError::UnsupportedOp {
                op: format!("scan expected 1 attached region but got {}", scan_regions.len()),
            });
        };
        lower_scan_to_while(
            body,
            scan_op.carry_count(),
            scan_op.length(),
            scan_op.reverse(),
            scan_op.unroll(),
            input_values,
            &mut self.block,
            self.context,
            self.location,
            self.captured_values.as_slice(),
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
            self.captured_values.as_slice(),
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
            &[],
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
    Ok(lower_mlir_module_for_program(
        program,
        capture_types,
        global_input_types,
        global_output_types,
        function_name,
        arg_shardings,
        result_shardings,
        None,
    )?
    .stable_hlo)
}

/// Lowers an arbitrary traced XLA program and returns both its textual module and exact physical entry signature.
pub(crate) fn lower_mlir_module_for_program<'o, Input, Output, ProgramInput, ProgramOutput, S>(
    program: &XlaProgram<ProgramInput, ProgramOutput>,
    capture_types: &[ArrayType],
    global_input_types: &Input,
    global_output_types: &Output,
    function_name: S,
    arg_shardings: Option<&[Sharding]>,
    result_shardings: Option<&[Sharding]>,
    target_platform: Option<&str>,
) -> Result<LoweredXlaModule, LoweringError>
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
    let logical_argument_types =
        capture_types.iter().cloned().chain(global_input_types.iter().cloned()).collect::<Vec<_>>();
    let signature = XlaExecutableSignature::new(logical_argument_types.as_slice(), global_output_types.as_slice());
    let physical_argument_types = signature.project_inputs(logical_argument_types.as_slice());
    let physical_output_types = signature.project_outputs(global_output_types.as_slice());

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

    // Module-scoped collective lowering state, shared between the entry function body and the deduplicated callee
    // functions below so channel ids stay unique module-wide and the target platform reaches nested callee bodies.
    let collective_state = CollectiveLoweringState::new().with_target_platform(target_platform);

    // Deduplicate `jit_call` callees that occur more than once into shared private `func.func`s, so repeated nested
    // programs (identical transformer blocks, or the per-block primal and pullback programs produced by `grad`) lower
    // to one function plus N `func.call`s instead of N inlined copies. The map is empty for modules without repeated
    // calls, in which case every `jit_call` inlines exactly as before.
    let nested_functions = Rc::new(collect_jit_call_functions(program));
    {
        let mut module_block = module.body()?;
        for key in &nested_functions.order {
            let function = nested_functions.functions.get(key).expect("ordered keys are present in the map");
            emit_jit_call_function(
                &mut module_block,
                function,
                &nested_functions,
                &collective_state,
                &context,
                location.as_ref(),
            )?;
        }
    }

    let logical_argument_tensor_types = logical_argument_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let physical_argument_tensor_types = physical_argument_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let physical_output_tensor_types = physical_output_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let arg_sharding_attributes = match arg_shardings {
        Some(shardings) => {
            if shardings.len() != logical_argument_types.len() {
                return Err(LoweringError::InvalidShardingCount {
                    kind: "argument",
                    expected: logical_argument_types.len(),
                    actual: shardings.len(),
                });
            }
            Some(
                signature
                    .project_inputs(shardings)
                    .iter()
                    .map(|sharding| sharding.to_mlir(location))
                    .collect::<Result<Vec<_>, _>>()?,
            )
        }
        None => None,
    };
    let result_sharding_attributes = match result_shardings {
        Some(shardings) => {
            if shardings.len() != global_output_types.len() {
                return Err(LoweringError::InvalidShardingCount {
                    kind: "result",
                    expected: global_output_types.len(),
                    actual: shardings.len(),
                });
            }
            Some(
                signature
                    .project_outputs(shardings)
                    .iter()
                    .map(|sharding| sharding.to_mlir(location))
                    .collect::<Result<Vec<_>, _>>()?,
            )
        }
        None => None,
    };
    let function_arguments = physical_argument_tensor_types
        .iter()
        .enumerate()
        .map(|(index, tensor_type)| {
            let attributes = arg_sharding_attributes
                .as_ref()
                .map(|shardings| HashMap::from([("sdy.sharding".into(), shardings[index].as_ref())]));
            TypeAndAttributes { r#type: tensor_type.as_ref(), attributes }
        })
        .collect::<Vec<_>>();
    let function_results = physical_output_tensor_types
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
            physical_argument_tensor_types
                .iter()
                .map(|tensor_type| (*tensor_type, location))
                .collect::<Vec<_>>()
                .as_slice(),
        );
        {
            let mut function_block_ref = function_block.as_ref();
            let mut logical_argument_values = Vec::with_capacity(logical_argument_types.len());
            for (logical_index, (array_type, tensor_type)) in
                logical_argument_types.iter().zip(logical_argument_tensor_types.iter()).enumerate()
            {
                let value = match signature.input_mapping()[logical_index] {
                    Some(physical_index) => function_block
                        .argument(physical_index)
                        .expect("physical function block arguments should exist")
                        .as_ref(),
                    None => lower_unplaced_constant_output(
                        std::slice::from_ref(array_type),
                        0,
                        &mut function_block_ref,
                        &context,
                        location.as_ref(),
                    )?
                    .into_iter()
                    .next()
                    .expect("a zero-space input constant should have one output"),
                };
                debug_assert_eq!(value.r#type()?, tensor_type.as_ref());
                logical_argument_values.push(value);
            }
            let (capture_values, input_values) = logical_argument_values.split_at(capture_types.len());
            let logical_outputs = lower_program_outputs_with_inputs(
                program,
                capture_values,
                input_values,
                &mut function_block_ref,
                &context,
                location.as_ref(),
                Some(&nested_functions),
                &collective_state,
            )?;
            let physical_outputs = signature.project_outputs(logical_outputs.as_slice());
            function_block_ref.append_operation(func::r#return(physical_outputs.as_slice(), location)?)?;
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
    Ok(LoweredXlaModule { stable_hlo: module.to_string(), signature })
}

/// Value type that can be materialized as a StableHLO dense constant during benchmark lowering.
pub(crate) trait MlirLowerableValue: Value<Type = ArrayType> + 'static {
    /// Builds a dense-elements attribute containing this value.
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError>;

    /// Returns this value's capture-table index when it is a captured-constant reference.
    #[inline]
    fn capture_index(&self) -> Option<usize> {
        None
    }

    /// Lowers this program constant inside a function body.
    #[inline]
    fn lower_constant_value<'b, 'c: 'b, 't: 'c, B, L>(
        &self,
        _captured_values: &[ValueRef<'b, 'c, 't>],
        block: &mut B,
        context: &'c MlirContext<'t>,
        location: L,
    ) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
    where
        B: Block<'b, 'c, 't>,
        L: Copy + Location<'c, 't>,
    {
        lower_literal_value(self, block, context, location)
    }

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

impl MlirLowerableValue for XlaArrayConstant {
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        _tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        _context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
        Err(LoweringError::MissingCapturedConstant { index: self.index() })
    }

    #[inline]
    fn capture_index(&self) -> Option<usize> {
        Some(self.index())
    }

    #[inline]
    fn lower_constant_value<'b, 'c: 'b, 't: 'c, B, L>(
        &self,
        captured_values: &[ValueRef<'b, 'c, 't>],
        _block: &mut B,
        _context: &'c MlirContext<'t>,
        _location: L,
    ) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
    where
        B: Block<'b, 'c, 't>,
        L: Copy + Location<'c, 't>,
    {
        lower_captured_constant(self, captured_values)
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

/// Concrete host literal lowering used by [`ConstantOperation`] and MLIR snapshot tooling.
impl MlirLowerableValue for CpuArray {
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
        let data_type = self.r#type().data_type();
        macro_rules! typed_elements {
            // Extract one natively supported scalar family and construct its matching typed MLIR dense attribute.
            ($method:ident, $variant:ident, $element:ty) => {{
                let values = self
                    .values()
                    .iter()
                    .map(|value| match value {
                        Scalar::$variant(value) => *value,
                        _ => unreachable!("array payload types are validated during construction"),
                    })
                    .collect::<Vec<$element>>();
                context
                    .$method(tensor_type, values.as_slice())
                    .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type })?
                    .cast::<DenseElementsAttributeRef>()
                    .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type })
            }};
        }
        macro_rules! raw_elements {
            // Preserve an unsupported-by-typed-MLIR scalar family's native storage bits through the raw constructor.
            ($variant:ident, $element:ty) => {{
                let values = self
                    .values()
                    .iter()
                    .map(|value| match value {
                        Scalar::$variant(value) => *value,
                        _ => unreachable!("array payload types are validated during construction"),
                    })
                    .collect::<Vec<$element>>();
                context
                    .dense_elements_attribute_from_raw_buffer(tensor_type, values.as_slice())
                    .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type })
            }};
        }

        match data_type {
            DataType::Boolean => typed_elements!(dense_bool_elements_attribute, Bool, bool),
            DataType::I8 => typed_elements!(dense_i8_elements_attribute, I8, i8),
            DataType::I16 => typed_elements!(dense_i16_elements_attribute, I16, i16),
            DataType::I32 => typed_elements!(dense_i32_elements_attribute, I32, i32),
            DataType::I64 => typed_elements!(dense_i64_elements_attribute, I64, i64),
            DataType::U8 => typed_elements!(dense_u8_elements_attribute, U8, u8),
            DataType::U16 => typed_elements!(dense_u16_elements_attribute, U16, u16),
            DataType::U32 => typed_elements!(dense_u32_elements_attribute, U32, u32),
            DataType::U64 => typed_elements!(dense_u64_elements_attribute, U64, u64),
            DataType::F4E2M1FN => raw_elements!(F4E2M1FN, u8),
            DataType::F6E2M3FN => raw_elements!(F6E2M3FN, u8),
            DataType::F6E3M2FN => raw_elements!(F6E3M2FN, u8),
            DataType::F8E3M4 => raw_elements!(F8E3M4, u8),
            DataType::F8E4M3 => raw_elements!(F8E4M3, u8),
            DataType::F8E4M3FN => raw_elements!(F8E4M3FN, u8),
            DataType::F8E4M3FNUZ => raw_elements!(F8E4M3FNUZ, u8),
            DataType::F8E4M3B11FNUZ => raw_elements!(F8E4M3B11FNUZ, u8),
            DataType::F8E5M2 => raw_elements!(F8E5M2, u8),
            DataType::F8E5M2FNUZ => raw_elements!(F8E5M2FNUZ, u8),
            DataType::F8E8M0FNU => raw_elements!(F8E8M0FNU, u8),
            DataType::BF16 => typed_elements!(dense_bf16_elements_attribute, BF16, half::bf16),
            DataType::F16 => typed_elements!(dense_f16_elements_attribute, F16, half::f16),
            DataType::F32 => typed_elements!(dense_f32_elements_attribute, F32, f32),
            DataType::F64 => typed_elements!(dense_f64_elements_attribute, F64, f64),
            DataType::C64 => {
                let values = self
                    .values()
                    .iter()
                    .flat_map(|value| match value {
                        Scalar::C64(value) => [value.re, value.im],
                        _ => unreachable!("array payload types are validated during construction"),
                    })
                    .collect::<Vec<_>>();
                context
                    .dense_elements_attribute_from_raw_buffer(tensor_type, values.as_slice())
                    .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type })
            }
            DataType::C128 => {
                let values = self
                    .values()
                    .iter()
                    .flat_map(|value| match value {
                        Scalar::C128(value) => [value.re, value.im],
                        _ => unreachable!("array payload types are validated during construction"),
                    })
                    .collect::<Vec<_>>();
                context
                    .dense_elements_attribute_from_raw_buffer(tensor_type, values.as_slice())
                    .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type })
            }
            DataType::Token
            | DataType::Zero
            | DataType::I1
            | DataType::I2
            | DataType::I4
            | DataType::U1
            | DataType::U2
            | DataType::U4 => return Err(LoweringError::UnsupportedDataType { data_type }),
        }
    }

    fn lower_constant_value<'b, 'c: 'b, 't: 'c, B, L>(
        &self,
        _captured_values: &[ValueRef<'b, 'c, 't>],
        block: &mut B,
        context: &'c MlirContext<'t>,
        location: L,
    ) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
    where
        B: Block<'b, 'c, 't>,
        L: Copy + Location<'c, 't>,
    {
        let value_type = self.r#type().into_owned();
        if value_type.is_scalar() {
            // A scalar array has exactly one value by construction.
            let value = self.values().first().copied().unwrap();
            let scalar_type = ArrayType::scalar(value_type.data_type());
            let scalar_tensor_type = lower_tensor_type(&scalar_type, context, location)?;
            let constant =
                lower_scalar_constant_splat(value, &scalar_type, scalar_tensor_type, block, context, location)?;
            return annotate_output_memory(constant, &value_type, block, context, location);
        }
        lower_literal_value(self, block, context, location)
    }
}

/// Lowers a plain traced `tracing_v2` program to a textual StableHLO MLIR module.
#[cfg(any(test, feature = "benchmarking"))]
pub(crate) fn to_mlir_module_for_plain_program<
    V: MlirLowerableValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
    O: LowerableXlaOperation<V>,
    S: AsRef<str>,
>(
    program: &Program<V, O, Input, Output>,
    function_name: S,
) -> Result<String, LoweringError>
where
    XlaOperation<V>: From<O>,
{
    let function_name = normalize_function_name(function_name.as_ref())?;
    let context = MlirContext::new();
    let location = context.unknown_location();
    let module = context.module(location)?;
    let mut mesh = None;
    for region in program.regions().iter() {
        for atom in region.atoms() {
            let atom_type = atom.r#type();
            let Some(sharding) = atom_type.sharding() else {
                continue;
            };
            mesh = Some(match mesh.take() {
                Some(existing_mesh) => merge_logical_meshes(&existing_mesh, sharding.mesh())?,
                None => sharding.mesh().clone(),
            });
        }
    }
    if let Some(mesh) = mesh {
        module.body()?.append_operation(mesh.to_mlir(location)?)?;
    }

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
    // Attached nested computations (control-flow bodies, custom-derivative programs, and `jit_call` callees) are
    // regions of this program's one canonical arena, so iterating every arena region covers them without any
    // per-operation recursion; only `shard_map` still carries its body as a payload and recurses explicitly.
    let mut mesh = existing;
    for region in program.regions().iter() {
        for instruction in region.instructions() {
            match &instruction.operation() {
                XlaOperation::ShardMap(shard_map_op) => {
                    // The body is an attached arena region covered by this same walk; only the boundary metadata's
                    // mesh needs merging here.
                    mesh = Some(match mesh.take() {
                        Some(existing_mesh) => merge_logical_meshes(&existing_mesh, shard_map_op.shard_map().mesh())?,
                        None => shard_map_op.shard_map().mesh().clone(),
                    });
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
                XlaOperation::Broadcast(operation)
                    if broadcast_changes_explicit_sharding(
                        region.atoms()[instruction.inputs()[0].index()].r#type().as_ref(),
                        operation.output_type(),
                        operation.output_axes(),
                    ) =>
                {
                    let output_sharding = operation.output_type().sharding().unwrap();
                    mesh = Some(match mesh.take() {
                        Some(existing_mesh) => merge_logical_meshes(&existing_mesh, output_sharding.mesh())?,
                        None => output_sharding.mesh().clone(),
                    });
                }
                _ => {}
            }
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

/// Returns the static dimensions for one tensor type.
fn static_dimensions(array_type: &ArrayType) -> Result<Vec<usize>, LoweringError> {
    array_type
        .shape()
        .dimensions()
        .iter()
        .map(|size| match size {
            Dimension::Static(value) => Ok(*value),
            Dimension::Dynamic(_) => Err(LoweringError::InvalidTensorType { array_type: array_type.clone() }),
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
    captured_values: &[ValueRef<'b, 'c, 't>],
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
    entry_token: Option<ValueRef<'b, 'c, 't>>,
    return_final_token: bool,
) -> Result<ryft_mlir::DetachedRegion<'c, 't>, LoweringError>
where
    V: MlirLowerableValue,
    O: LowerableXlaOperation<V>,
    XlaOperation<V>: From<O>,
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
            captured_values,
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

fn lower_condition_to_if<'b, 'c: 'b, 't: 'c, V>(
    branch_regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    captured_values: &[ValueRef<'b, 'c, 't>],
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
    token: &mut Option<ValueRef<'b, 'c, 't>>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: MlirLowerableValue,
{
    let [true_branch, false_branch] = branch_regions else {
        return Err(LoweringError::UnsupportedOp {
            op: format!("condition expected 2 attached regions but got {}", branch_regions.len()),
        });
    };
    let expected_input_count = true_branch.input_types().len() + 1;
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
    let threads_token = !true_branch.effects().is_pure() || !false_branch.effects().is_pure();
    let entry_token = if threads_token { Some(current_or_new_token(token, block, location)?) } else { None };
    let true_branch_region = lower_control_flow_region(
        true_branch,
        branch_inputs,
        context,
        location,
        captured_values,
        nested_functions,
        collective_state,
        entry_token,
        threads_token,
    )?;
    let false_branch_region = lower_control_flow_region(
        false_branch,
        branch_inputs,
        context,
        location,
        captured_values,
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
    let output_count = true_branch.output_types().len();
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

fn lower_while_to_while<'b, 'c: 'b, 't: 'c, V>(
    while_op: &WhileOperation,
    loop_regions: &[Program<V, XlaOperation<V>, Vec<V>, Vec<V>>],
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    captured_values: &[ValueRef<'b, 'c, 't>],
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
    token: &mut Option<ValueRef<'b, 'c, 't>>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: MlirLowerableValue,
{
    let [condition, body] = loop_regions else {
        return Err(LoweringError::UnsupportedOp {
            op: format!("while expected 2 attached regions but got {}", loop_regions.len()),
        });
    };
    let state_types = body.input_types();
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
    let predicate_type = condition.output_types()[0].clone();
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
    let threads_token = !condition.effects().is_pure() || !body.effects().is_pure();
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
            condition,
            input_values,
            block,
            context,
            location,
            captured_values,
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
                condition,
                condition_inputs.as_slice(),
                &mut condition_block_ref,
                context,
                location,
                captured_values,
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
            body,
            body_inputs.as_slice(),
            &mut body_block_ref,
            context,
            location,
            captured_values,
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
                condition,
                masked.as_slice(),
                &mut body_block_ref,
                context,
                location,
                captured_values,
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
    captured_values: &[ValueRef<'b, 'c, 't>],
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
    token: &mut Option<ValueRef<'b, 'c, 't>>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: MlirLowerableValue,
    O: LowerableXlaOperation<V>,
    XlaOperation<V>: From<O>,
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
            ryft_core::types::Shape::new(dimensions.into_iter().map(Dimension::Static).collect()),
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
                captured_values,
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
                captured_values,
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
    captured_values: &[ValueRef<'b, 'c, 't>],
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
    token: &mut Option<ValueRef<'b, 'c, 't>>,
) -> Result<(Vec<ValueRef<'b, 'c, 't>>, Vec<ValueRef<'b, 'c, 't>>), LoweringError>
where
    V: MlirLowerableValue,
    O: LowerableXlaOperation<V>,
    XlaOperation<V>: From<O>,
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
        captured_values,
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

/// Structural identity of a flat callee program attached behind a `jit_call`, used to deduplicate repeated nested
/// programs into shared private `func.func`s at lowering time.
///
/// Eligible programs (see [`supports_structural_dedup`]) are keyed by their canonical rendering plus their complete
/// flat input/output signature: type inference is deterministic and attached regions (control-flow bodies,
/// custom-derivative programs, and nested `jit_call` callees) render contextually inside their instructions, so two
/// programs that render identically with equal boundary types compute the same function and may share one emitted
/// function — even when they are distinct staged programs produced by separate transform passes (for example the
/// per-block primal and pullback programs of `grad(jit(f))` over repeated blocks).
#[derive(Clone, PartialEq, Eq, Hash)]
struct JitCallProgramKey {
    /// Canonical [`Program`] rendering (operation names, all bracketed attributes, and attached region bodies).
    rendered: String,

    /// Flat input [`ArrayType`]s, which together with the rendering pin the full callee signature.
    input_types: Vec<ArrayType>,

    /// Flat output [`ArrayType`]s, completing the callee signature when output-only placement metadata is not visible
    /// in the rendered body.
    output_types: Vec<ArrayType>,
}

/// Returns whether `program` may be deduplicated by structural identity.
///
/// A program is eligible only when the canonical rendering captures the full semantics needed to safely share a
/// private function. Constant atom payloads intentionally render only as `const`, so programs with constants are
/// ineligible; otherwise two callees that differ only by capture reference or literal payload could merge incorrectly.
/// `shard_map` is also ineligible because its operation payload still owns additional body metadata that is not fully
/// represented by the rendered instruction line.
fn supports_structural_dedup<V: MlirLowerableValue>(program: &Program<V, XlaOperation<V>, Vec<V>, Vec<V>>) -> bool {
    program.regions().iter().all(|region| {
        region.atoms().iter().all(|atom| !atom.is_constant())
            && region
                .instructions()
                .iter()
                .all(|instruction| !matches!(instruction.operation(), XlaOperation::ShardMap(_)))
    })
}

/// Returns whether a borrowed rooted region may be deduplicated by structural identity.
fn supports_structural_dedup_region<V: MlirLowerableValue>(region: RegionRef<'_, V, XlaOperation<V>>) -> bool {
    fn walk<V: MlirLowerableValue>(region: RegionRef<'_, V, XlaOperation<V>>, visited: &mut HashSet<RegionId>) -> bool {
        if !visited.insert(region.id()) {
            return true;
        }
        region.atoms().iter().all(|atom| !atom.is_constant())
            && region.instructions().iter().all(|instruction| {
                !matches!(instruction.operation(), XlaOperation::ShardMap(_))
                    && instruction
                        .regions()
                        .iter()
                        .copied()
                        .all(|nested| RegionRef::new(region.arena(), nested).is_ok_and(|nested| walk(nested, visited)))
            })
    }

    walk(region, &mut HashSet::new())
}

/// Computes the deduplication key for a flat callee program, or [`None`] for programs that are not eligible for
/// structural deduplication (see [`supports_structural_dedup`]) and therefore always inline. The key is generic over
/// the program's constant universe so that lookups from the value-generic lowering path type-check; the map itself is
/// only ever populated from the staged [`XlaConstant`]-keyed pipeline.
fn jit_call_program_key<V: MlirLowerableValue>(
    program: &Program<V, XlaOperation<V>, Vec<V>, Vec<V>>,
) -> Option<JitCallProgramKey> {
    supports_structural_dedup(program).then(|| JitCallProgramKey {
        rendered: program.to_string(),
        input_types: program.input_types(),
        output_types: program.output_types(),
    })
}

/// Computes the deduplication key for a borrowed callee region, materializing only after borrowed eligibility checks.
fn jit_call_region_key<V: MlirLowerableValue>(region: RegionRef<'_, V, XlaOperation<V>>) -> Option<JitCallProgramKey> {
    supports_structural_dedup_region(region).then(|| {
        let program = region.to_program();
        JitCallProgramKey {
            rendered: program.to_string(),
            input_types: program.input_types(),
            output_types: program.output_types(),
        }
    })
}

/// Returns the leading lowered input values that correspond to the capture table referenced by `program`.
fn captured_prefix_values<'b, 'c: 'b, 't: 'c, V>(
    program: &Program<V, XlaOperation<V>, Vec<V>, Vec<V>>,
    input_values: &[ValueRef<'b, 'c, 't>],
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: MlirLowerableValue,
{
    let capture_count = program
        .regions()
        .iter()
        .flat_map(|region| region.atoms())
        .filter_map(|atom| atom.as_constant().and_then(MlirLowerableValue::capture_index))
        .max()
        .map(|index| index + 1)
        .unwrap_or(0);
    if input_values.len() < capture_count {
        return Err(LoweringError::MissingCapturedConstant { index: capture_count.saturating_sub(1) });
    }
    Ok(input_values[..capture_count].to_vec())
}

/// One deduplicated callee emitted as a shared private `func.func`.
struct JitCallFunction {
    /// Symbol name of the emitted private function.
    symbol: String,

    /// Representative callee program for this key (materialized from its callee region), lowered once as the
    /// function body.
    program: FlatXlaProgram,

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
    fn get<V: MlirLowerableValue>(
        &self,
        program: &Program<V, XlaOperation<V>, Vec<V>, Vec<V>>,
    ) -> Option<&JitCallFunction> {
        self.functions.get(&jit_call_program_key(program)?)
    }
}

/// Counts `jit_call` callee occurrences in `program`, covering nested computations at every depth: attached regions
/// (control-flow bodies, custom-derivative programs, and callee bodies) all live in the program's one canonical
/// region arena, so the walk descends region edges without materializing nested programs. Shard-map body regions are
/// intentionally skipped: their `jit_call`s lower with shard-local types and always inline.
///
/// `counts` accumulates the occurrence count and a representative materialized program per identity, `order` records
/// keys in first-occurrence order, and `memo` caches the (possibly expensive) key computation per callee root
/// [`RegionId`] — interned shared callees repeat their root region id, so each shared callee is rendered once.
fn count_jit_calls<Input, Output>(
    program: &XlaProgram<Input, Output>,
    counts: &mut HashMap<JitCallProgramKey, (usize, FlatXlaProgram)>,
    order: &mut Vec<JitCallProgramKey>,
    memo: &mut HashMap<RegionId, Option<JitCallProgramKey>>,
) where
    Input: Parameterized<XlaConstant>,
    Output: Parameterized<XlaConstant>,
{
    fn walk<Input, Output>(
        program: &XlaProgram<Input, Output>,
        region: RegionId,
        counts: &mut HashMap<JitCallProgramKey, (usize, FlatXlaProgram)>,
        order: &mut Vec<JitCallProgramKey>,
        memo: &mut HashMap<RegionId, Option<JitCallProgramKey>>,
        visited: &mut HashSet<RegionId>,
    ) where
        Input: Parameterized<XlaConstant>,
        Output: Parameterized<XlaConstant>,
    {
        // Interned shared callees repeat their root region id; each region's instructions are walked once while
        // every referencing `jit_call` instruction still contributes one occurrence to the counts below.
        if !visited.insert(region) {
            return;
        }
        for instruction in
            program.region(region).expect("region ids in the arena are valid by construction").instructions()
        {
            if matches!(instruction.operation(), XlaOperation::ShardMap(_)) {
                continue;
            }
            for &nested in instruction.regions() {
                walk(program, nested, counts, order, memo, visited);
            }
            if let XlaOperation::JitCall(_) = instruction.operation() {
                let Some(&callee_region) = instruction.regions().first() else {
                    continue;
                };
                let key = memo
                    .entry(callee_region)
                    .or_insert_with(|| {
                        let callee = program
                            .region_ref(callee_region)
                            .expect("jit_call callee regions are validated at build time");
                        jit_call_region_key(callee)
                    })
                    .clone();
                let Some(key) = key else {
                    continue;
                };
                let entry = counts.entry(key.clone()).or_insert_with(|| {
                    order.push(key.clone());
                    (
                        0,
                        program
                            .region_ref(callee_region)
                            .expect("jit_call callee regions are validated at build time")
                            .to_program(),
                    )
                });
                entry.0 += 1;
            }
        }
    }
    let mut visited = HashSet::new();
    walk(program, program.entry(), counts, order, memo, &mut visited);
}

/// Builds the [`JitCallFunctionMap`] for a module by emitting a shared private function for every `jit_call` callee
/// that occurs at least twice (per [`JitCallProgramKey`] identity). Single-occurrence callees are left to inline, so
/// modules without repeated calls lower exactly as before.
fn collect_jit_call_functions<Input, Output>(program: &XlaProgram<Input, Output>) -> JitCallFunctionMap
where
    Input: Parameterized<XlaConstant>,
    Output: Parameterized<XlaConstant>,
{
    let mut counts: HashMap<JitCallProgramKey, (usize, FlatXlaProgram)> = HashMap::new();
    let mut order: Vec<JitCallProgramKey> = Vec::new();
    let mut memo: HashMap<RegionId, Option<JitCallProgramKey>> = HashMap::new();
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
/// lower to `func.call`s (calls between shared functions are resolved by symbol, so emission order does not matter),
/// and with the module's `collective_state` so module-scoped lowering state — the shared channel-id counter and the
/// target platform that gates platform-specific fast paths such as the block-scaled dot custom call — reaches the
/// callee body exactly like inlined callees (the mesh/manual-region state intentionally resets per function).
fn emit_jit_call_function<'b, 'c: 'b, 't: 'c>(
    module_block: &mut BlockRef<'b, 'c, 't>,
    function: &JitCallFunction,
    nested_functions: &Rc<JitCallFunctionMap>,
    collective_state: &CollectiveLoweringState,
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
        // Deduplicated callees are pure by construction (`collect_jit_call_functions` skips effectful programs),
        // so the shared function body never needs an effect token.
        let mut token = None;
        let outputs = lower_nested_program_inline(
            &function.program,
            input_values.as_slice(),
            &mut function_block_ref,
            context,
            location,
            &[],
            false,
            Some(nested_functions),
            collective_state,
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
fn lower_jit_call<'b, 'c: 'b, 't: 'c, V: MlirLowerableValue>(
    program: &Program<V, XlaOperation<V>, Vec<V>, Vec<V>>,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
    token: &mut Option<ValueRef<'b, 'c, 't>>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    // Only pure callees are ever deduplicated (`collect_jit_call_functions` skips effectful programs), so the
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
    let captured_values = captured_prefix_values(program, input_values)?;
    lower_nested_program_inline(
        program,
        input_values,
        block,
        context,
        location,
        captured_values.as_slice(),
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
    captured_values: &[ValueRef<'b, 'c, 't>],
    add_optimization_barrier: bool,
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
    token: &mut Option<ValueRef<'b, 'c, 't>>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: MlirLowerableValue,
    O: LowerableXlaOperation<V>,
    XlaOperation<V>: From<O>,
{
    lower_nested_region_inline(
        program.entry_region_ref(),
        input_values,
        block,
        context,
        location,
        captured_values,
        add_optimization_barrier,
        nested_functions,
        collective_state,
        token,
    )
}

/// Inlines a borrowed nested region into the given block without materializing the region itself.
#[allow(clippy::too_many_arguments)]
fn lower_nested_region_inline<'b, 'c: 'b, 't: 'c, O, V>(
    region: RegionRef<'_, V, O>,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    captured_values: &[ValueRef<'b, 'c, 't>],
    add_optimization_barrier: bool,
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
    token: &mut Option<ValueRef<'b, 'c, 't>>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: MlirLowerableValue,
    O: LowerableXlaOperation<V>,
    XlaOperation<V>: From<O>,
{
    let outputs = replay_region_ref_into_block(
        region,
        input_values.to_vec(),
        block,
        context,
        location,
        |_, value, block, context, location| value.lower_constant_value(captured_values, block, context, location),
        |instruction, inputs, block, context, location| {
            let input_types = instruction
                .inputs()
                .iter()
                .map(|input| region.atoms()[input.index()].r#type().into_owned())
                .collect::<Vec<_>>();
            let output_types = instruction
                .outputs()
                .iter()
                .map(|output| region.atoms()[output.index()].r#type().into_owned())
                .collect::<Vec<_>>();
            // Region programs lower through the trait's canonical `XlaOperation` surface, so embed the enclosing
            // operation type into it while materializing each attached region. The operation hook still takes owned
            // nested programs, so this remains a genuine operation-family mapping boundary.
            let regions = instruction
                .regions()
                .iter()
                .map(|attached| {
                    RegionRef::new(region.arena(), *attached)?
                        .to_program()
                        .map_operations(|operation| Ok(XlaOperation::from(operation.clone())))
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            let mut lowerer = PlainMlirLowerer::new(*block, context, location)
                .with_input_types(input_types)
                .with_nested_functions(nested_functions.cloned())
                .with_captured_values(captured_values)
                .with_token(*token)
                .with_collective_state(collective_state.clone());
            let outputs = instruction.operation().lower_to_mlir(
                inputs,
                regions.as_slice(),
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

/// Drives [`RegionRef::interpret_with`] to lower a borrowed region into MLIR ops appended to `block`.
fn replay_region_ref_into_block<'b, 'c: 'b, 't: 'c, O, V, LiftConstant, ApplyOp>(
    region: RegionRef<'_, V, O>,
    input_values: Vec<ValueRef<'b, 'c, 't>>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    mut lift_constant: LiftConstant,
    mut apply_op: ApplyOp,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: Value,
    O: Operation<V::Type>,
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
    region.interpret_with(
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
    V: MlirLowerableValue,
    O: LowerableXlaOperation<V>,
    XlaOperation<V>: From<O>,
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
            let input_types = instruction
                .inputs()
                .iter()
                .map(|input| program.atoms()[input.index()].r#type().into_owned())
                .collect::<Vec<_>>();
            let output_types = instruction
                .outputs()
                .iter()
                .map(|output| program.atoms()[output.index()].r#type().into_owned())
                .collect::<Vec<_>>();
            // Region programs lower through the trait's canonical `XlaOperation` surface, so embed the enclosing
            // program's operation type into it while materializing each attached region.
            let regions = instruction
                .regions()
                .iter()
                .map(|region| {
                    program
                        .region_ref(*region)?
                        .to_program()
                        .map_operations(|operation| Ok(XlaOperation::from(operation.clone())))
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            let mut lowerer = PlainMlirLowerer::new(*block, context, location)
                .with_input_types(input_types)
                .with_token(token)
                .with_collective_state(collective_state.clone());
            let outputs = instruction.operation().lower_to_mlir(
                inputs,
                regions.as_slice(),
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
    let input_values = program
        .input_ids()
        .iter()
        .enumerate()
        .map(|(index, _)| {
            block.argument(captured_values.len() + index).expect("body block arguments should exist").as_ref()
        })
        .collect::<Vec<_>>();
    lower_program_outputs_with_inputs(
        program,
        captured_values,
        input_values.as_slice(),
        block,
        context,
        location,
        nested_functions,
        collective_state,
    )
}

/// Lowers one traced program using explicitly provided logical input values.
#[allow(clippy::too_many_arguments)]
fn lower_program_outputs_with_inputs<'b, 'c: 'b, 't: 'c, ProgramInput, ProgramOutput>(
    program: &XlaProgram<ProgramInput, ProgramOutput>,
    captured_values: &[ValueRef<'b, 'c, 't>],
    input_values: &[ValueRef<'b, 'c, 't>],
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
    if input_values.len() != program.input_ids().len() {
        return Err(ProgramError::InvalidInputCount {
            expected: program.input_ids().len(),
            actual: input_values.len(),
        }
        .into());
    }
    // Mirror table of every lowered atom value. Shard-map operations look up captured global primals by `AtomId`,
    // so we keep a parallel table alongside `Program::interpret_with`'s use-count-tracked one. `ValueRef` is
    // `Copy`, so this mirror is cheap.
    let mut atom_values = vec![None; program.atoms().len()];
    for (atom_id, value) in program.input_ids().iter().copied().zip(input_values.iter().copied()) {
        atom_values[atom_id.index()] = Some(value);
    }
    let atom_values = std::cell::RefCell::new(atom_values);
    // Function-body-scoped effect token chain, created lazily by the first effectful instruction and dropped at the
    // end of the function body: this v1 design orders effects within one dispatch, and carrying tokens across
    // separately dispatched executions is out of scope.
    let mut token = None;
    replay_program_into_block(
        program,
        input_values.to_vec(),
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
                captured_values,
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
    captured_values: &[ValueRef<'b, 'c, 't>],
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
            captured_values,
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

/// Lowers one `sdy.manual_computation` operation for the value-generic plain pipeline, replaying the attached
/// local body region through the same inline machinery as every other region-carrying operation (constants lower as
/// literals through [`MlirLowerableValue`], and nested `jit_call`s always inline because manual bodies use
/// shard-local types).
#[allow(clippy::too_many_arguments)]
fn lower_manual_computation_inline<'b, 'c: 'b, 't: 'c, V>(
    block: &mut BlockRef<'b, 'c, 't>,
    outer_inputs: &[ValueRef<'b, 'c, 't>],
    shard_map: &ShardMap,
    program: &Program<V, XlaOperation<V>, Vec<V>, Vec<V>>,
    global_output_types: &[ArrayType],
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    collective_state: &CollectiveLoweringState,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: MlirLowerableValue,
{
    let local_input_tensor_types = program
        .input_types()
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
        let input_values = (0..program.input_types().len())
            .map(|index| body_block.argument(index).expect("body block arguments should exist").as_ref())
            .collect::<Vec<_>>();
        // Manual bodies lower with shard-local types, so their `jit_call`s always inline; do not thread the
        // module's deduplicated functions (which are typed against global shapes) into them. The body is also its
        // own effect scope, mirroring the captured manual-computation lowering.
        let body_collective_state = collective_state.enter_manual_region(shard_map.clone());
        let mut token = None;
        let body_outputs = lower_nested_program_inline(
            program,
            input_values.as_slice(),
            &mut body_block_ref,
            context,
            location,
            &[],
            false,
            None,
            &body_collective_state,
            &mut token,
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

/// Lowers one concrete traced value to a StableHLO constant operation and applies its declared memory placement.
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
            let broadcast = broadcast.result(0).expect("stablehlo.broadcast should return one result").as_ref();
            return annotate_output_memory(broadcast, &value_type, block, context, location);
        }
    }

    let tensor_type = lower_tensor_type(&value_type, context, location)?;
    let elements = value.to_dense_elements_attribute(tensor_type, context)?;
    let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
    let constant = constant.result(0).expect("stablehlo.constant should return one result").as_ref();
    annotate_output_memory(constant, &value_type, block, context, location)
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
    regions: &[FlatXlaProgram],
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
            let constant_value = constant.value().lower_constant_value(
                captured_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            Ok(vec![constant_value])
        }
        XlaOperation::ConvertElementType(_) => {
            check_count!("input", input_values, 1, ProgramError);
            check_count!("output", output_types, 1, ProgramError);
            let output_type = lower_tensor_type(&output_types[0], lowerer.context, lowerer.location)?;
            let result =
                lowerer
                    .block
                    .append_operation(stable_hlo::convert(input_values[0], output_type, lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.convert should return one result").as_ref()])
        }
        XlaOperation::Add(_) => {
            let [left, right] = normalize_binary_elementwise_operands(
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            let result = lowerer.block.append_operation(stable_hlo::add(left, right, lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.add should return one result").as_ref()])
        }
        XlaOperation::Sub(_) => {
            let [left, right] = normalize_binary_elementwise_operands(
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            let result = lowerer.block.append_operation(stable_hlo::subtract(left, right, lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.subtract should return one result").as_ref()])
        }
        XlaOperation::Mul(_) => {
            let [left, right] = normalize_binary_elementwise_operands(
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            let result = lowerer.block.append_operation(stable_hlo::multiply(left, right, lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.multiply should return one result").as_ref()])
        }
        XlaOperation::Div(_) => {
            let [left, right] = normalize_binary_elementwise_operands(
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            let result = lowerer.block.append_operation(stable_hlo::divide(left, right, lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.divide should return one result").as_ref()])
        }
        XlaOperation::Neg(_) => {
            let result = lowerer.block.append_operation(stable_hlo::negate(input_values[0], lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.negate should return one result").as_ref()])
        }
        XlaOperation::Sin(_) => {
            if output_types[0].data_type().is_complex() {
                return Ok(vec![lower_complex_sine_or_cosine(
                    input_values[0],
                    &output_types[0],
                    true,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?]);
            }
            let result = lowerer.block.append_operation(stable_hlo::sine(
                input_values[0],
                Accuracy::Default,
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.sine should return one result").as_ref()])
        }
        XlaOperation::Cos(_) => {
            if output_types[0].data_type().is_complex() {
                return Ok(vec![lower_complex_sine_or_cosine(
                    input_values[0],
                    &output_types[0],
                    false,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?]);
            }
            let result = lowerer.block.append_operation(stable_hlo::cosine(
                input_values[0],
                Accuracy::Default,
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.cosine should return one result").as_ref()])
        }
        XlaOperation::Atan2(_) => {
            let [left, right] = normalize_binary_elementwise_operands(
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            let result = lowerer.block.append_operation(stable_hlo::atan2(left, right, lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.atan2 should return one result").as_ref()])
        }
        XlaOperation::Exp(_) => {
            let result = lowerer.block.append_operation(stable_hlo::exponential(
                input_values[0],
                Accuracy::Default,
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.exponential should return one result").as_ref()])
        }
        XlaOperation::Log(_) => {
            let result = lowerer.block.append_operation(stable_hlo::log(
                input_values[0],
                Accuracy::Default,
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.log should return one result").as_ref()])
        }
        XlaOperation::Sqrt(_) => {
            let result = lowerer.block.append_operation(stable_hlo::sqrt(
                input_values[0],
                Accuracy::Default,
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.sqrt should return one result").as_ref()])
        }
        XlaOperation::Rsqrt(_) => {
            let result = lowerer.block.append_operation(stable_hlo::rsqrt(
                input_values[0],
                Accuracy::Default,
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.rsqrt should return one result").as_ref()])
        }
        XlaOperation::Tanh(_) => {
            let result = lowerer.block.append_operation(stable_hlo::tanh(
                input_values[0],
                Accuracy::Default,
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.tanh should return one result").as_ref()])
        }
        XlaOperation::Logistic(_) => {
            let result = lowerer.block.append_operation(stable_hlo::logistic(
                input_values[0],
                Accuracy::Default,
                lowerer.location,
            )?)?;
            Ok(vec![result.result(0).expect("stablehlo.logistic should return one result").as_ref()])
        }
        XlaOperation::Erf(_) => {
            let result = lowerer.block.append_operation(chlo::erf(input_values[0], lowerer.location)?)?;
            Ok(vec![result.result(0).expect("chlo.erf should return one result").as_ref()])
        }
        XlaOperation::Pow(_) => {
            let [left, right] = normalize_binary_elementwise_operands(
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            let result = lowerer.block.append_operation(stable_hlo::power(left, right, lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.power should return one result").as_ref()])
        }
        XlaOperation::Sign(_) => {
            let result = lowerer.block.append_operation(stable_hlo::sign(input_values[0], lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.sign should return one result").as_ref()])
        }
        XlaOperation::Floor(_) => {
            let result = lowerer.block.append_operation(stable_hlo::floor(input_values[0], lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.floor should return one result").as_ref()])
        }
        XlaOperation::Ceil(_) => {
            let result = lowerer.block.append_operation(stable_hlo::ceil(input_values[0], lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.ceil should return one result").as_ref()])
        }
        XlaOperation::Round(_) => {
            let result = lowerer
                .block
                .append_operation(stable_hlo::round_with_nearest_even_tie_break(input_values[0], lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.round_nearest_even should return one result").as_ref()])
        }
        XlaOperation::Max(_) => {
            let [left, right] = normalize_binary_elementwise_operands(
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            let result = lowerer.block.append_operation(stable_hlo::maximum(left, right, lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.maximum should return one result").as_ref()])
        }
        XlaOperation::Min(_) => {
            let [left, right] = normalize_binary_elementwise_operands(
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            let result = lowerer.block.append_operation(stable_hlo::minimum(left, right, lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.minimum should return one result").as_ref()])
        }
        XlaOperation::Rem(_) => {
            let [left, right] = normalize_binary_elementwise_operands(
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            let result = lowerer.block.append_operation(stable_hlo::remainder(left, right, lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.remainder should return one result").as_ref()])
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
        XlaOperation::CustomCall(operation) => lower_custom_call_to_mlir(
            operation,
            input_values,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        ),
        XlaOperation::TransferToMemory(operation) => lower_transfer_to_memory(
            operation.destination(),
            input_values,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        ),
        // Custom-derivative calls lower as their primal program; the derivative programs never reach the backend.
        XlaOperation::CustomJvp(_) => lower_nested_program_inline(
            &regions[0],
            input_values,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
            lowerer.captured_values.as_slice(),
            false,
            lowerer.nested_functions.as_ref(),
            &lowerer.collective_state,
            &mut lowerer.token,
        ),
        XlaOperation::CustomVjp(_) => lower_nested_program_inline(
            &regions[0],
            input_values,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
            lowerer.captured_values.as_slice(),
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
        XlaOperation::Rematerialize(_) => lower_nested_program_inline(
            &regions[0],
            input_values,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
            lowerer.captured_values.as_slice(),
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
        XlaOperation::Iota(iota) => {
            check_count!("input", input_values, 0, ProgramError);
            check_count!("output", output_types, 1, ProgramError);
            let output_tensor_type = lowerer.lower_tensor_type(&output_types[0])?;
            let result = lowerer.block.append_operation(stable_hlo::iota(
                output_tensor_type,
                iota.dimension(),
                lowerer.location,
            )?)?;
            let result = result.result(0).expect("stablehlo.iota should return one result").as_ref();
            Ok(vec![annotate_output_memory(
                result,
                &output_types[0],
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?])
        }
        XlaOperation::CoordinateBasis(operation) => {
            check_count!("input", input_values, 0, ProgramError);
            lower_coordinate_basis_to_mlir(
                operation,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )
        }
        XlaOperation::Reshape(operation) => lower_reshape_to_mlir(
            operation,
            input_values,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        ),
        XlaOperation::Reshard(operation) => {
            lower_sharding_constraint(input_values, operation.sharding(), &mut lowerer.block, lowerer.location)
        }
        XlaOperation::ShardingConstraint(operation) => {
            lower_sharding_constraint(input_values, operation.sharding(), &mut lowerer.block, lowerer.location)
        }
        XlaOperation::Broadcast(operation) => lower_broadcast_to_mlir(
            operation,
            input_values,
            lowerer.input_types.as_slice(),
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        ),
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
        XlaOperation::Sort(operation) => lower_sort_to_mlir(
            operation,
            input_values,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        ),
        XlaOperation::RngBitGenerator(operation) => lower_rng_bit_generator_to_mlir(
            operation,
            input_values,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        ),
        XlaOperation::ScaledDot(operation) => {
            let collective_state = lowerer.collective_state.clone();
            lower_scaled_dot_to_mlir(
                operation,
                &collective_state,
                input_values,
                &lowerer.input_types,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )
        }
        XlaOperation::DotProductAttention(operation) => {
            let collective_state = lowerer.collective_state.clone();
            lower_dot_product_attention_to_mlir(
                operation,
                &collective_state,
                input_values,
                &lowerer.input_types,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )
        }
        XlaOperation::DotProductAttentionBackward(operation) => {
            let collective_state = lowerer.collective_state.clone();
            lower_dot_product_attention_backward_to_mlir(
                operation,
                &collective_state,
                input_values,
                &lowerer.input_types,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )
        }
        XlaOperation::Compare(operation) => {
            let operand_type = comparison_operand_type(&lowerer.input_types, output_types)?;
            let [left, right] = normalize_binary_elementwise_operands(
                input_values,
                &[operand_type],
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            let value =
                lower_compare_to_mlir(operation.direction(), left, right, &mut lowerer.block, lowerer.location)?;
            Ok(vec![value])
        }
        XlaOperation::Not(_) => {
            let result = lowerer.block.append_operation(stable_hlo::not(input_values[0], lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.not should return one result").as_ref()])
        }
        XlaOperation::And(_) => {
            let [left, right] = normalize_binary_elementwise_operands(
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            let result = lowerer.block.append_operation(stable_hlo::and(left, right, lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.and should return one result").as_ref()])
        }
        XlaOperation::Or(_) => {
            let [left, right] = normalize_binary_elementwise_operands(
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            let result = lowerer.block.append_operation(stable_hlo::or(left, right, lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.or should return one result").as_ref()])
        }
        XlaOperation::Xor(_) => {
            let [left, right] = normalize_binary_elementwise_operands(
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            let result = lowerer.block.append_operation(stable_hlo::xor(left, right, lowerer.location)?)?;
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
        XlaOperation::AllGather(operation) => {
            check_count!("input", input_values, 1, ProgramError);
            check_count!("output", output_types, 1, ProgramError);
            let collective_state = lowerer.collective_state.clone();
            lower_all_gather_to_mlir(
                operation,
                &collective_state,
                input_values[0],
                &output_types[0],
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )
        }
        XlaOperation::PSumScatter(operation) => {
            check_count!("input", input_values, 1, ProgramError);
            check_count!("output", output_types, 1, ProgramError);
            let collective_state = lowerer.collective_state.clone();
            lower_psum_scatter_to_mlir(
                operation,
                &collective_state,
                input_values[0],
                &output_types[0],
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )
        }
        XlaOperation::Ppermute(operation) => {
            check_count!("input", input_values, 1, ProgramError);
            let collective_state = lowerer.collective_state.clone();
            lower_ppermute_to_mlir(operation, &collective_state, input_values[0], &mut lowerer.block, lowerer.location)
        }
        XlaOperation::AllToAll(operation) => {
            check_count!("input", input_values, 1, ProgramError);
            check_count!("output", output_types, 1, ProgramError);
            let collective_state = lowerer.collective_state.clone();
            lower_all_to_all_to_mlir(
                operation,
                &collective_state,
                input_values[0],
                &lowerer.input_types[0],
                &output_types[0],
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )
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
            let [condition, on_true, on_false] = normalize_select_operands(
                input_values,
                output_types,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
            let result =
                lowerer
                    .block
                    .append_operation(stable_hlo::select(condition, on_true, on_false, lowerer.location)?)?;
            Ok(vec![result.result(0).expect("stablehlo.select should return one result").as_ref()])
        }
        XlaOperation::Slice(operation) => lower_slice_to_mlir(
            operation,
            input_values,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        ),
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
        XlaOperation::Pad(operation) => lower_pad_to_mlir(
            operation,
            input_values,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        ),
        XlaOperation::Concatenate(operation) => {
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
        XlaOperation::Condition(_) => lowerer.lower_condition(regions, input_values),
        XlaOperation::While(while_op) => lowerer.lower_while(while_op, regions, input_values),
        XlaOperation::Scan(scan_op) => lowerer.lower_scan(scan_op, regions, input_values),
        XlaOperation::JitCall(_) => lower_jit_call(
            &regions[0],
            input_values,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
            lowerer.nested_functions.as_ref(),
            &lowerer.collective_state,
            &mut lowerer.token,
        ),
        XlaOperation::ShardMap(shard_map_op) => {
            let simplified_body = regions[0]
                .simplified()
                .map_err(|error| LoweringError::SimplificationFailure { message: error.to_string() })?;
            lowerer.lower_manual_computation(
                input_values,
                shard_map_op.shard_map(),
                &simplified_body,
                simplified_body.input_types().as_slice(),
                shard_map_op.global_output_types(),
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
    captured_values: &[ValueRef<'b, 'c, 't>],
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
    token: &mut Option<ValueRef<'b, 'c, 't>>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    ProgramInput: Parameterized<XlaConstant>,
    ProgramOutput: Parameterized<XlaConstant>,
{
    let input_types = instruction
        .inputs()
        .iter()
        .map(|input| program.atoms()[input.index()].r#type().into_owned())
        .collect::<Vec<_>>();
    let output_types = instruction
        .outputs()
        .iter()
        .map(|output| program.atoms()[output.index()].r#type().into_owned())
        .collect::<Vec<_>>();
    let mut lowerer = ShardMapMlirLowerer::new(*block, context, location)
        .with_input_types(input_types)
        .with_nested_functions(nested_functions.cloned())
        .with_captured_values(captured_values)
        .with_token(*token)
        .with_collective_state(collective_state.clone());
    let regions = instruction
        .regions()
        .iter()
        .map(|region| program.region_ref(*region).map(|region| region.to_program()))
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = dispatch_lower_shard_map_mlir(
        &instruction.operation(),
        captured_values,
        input_values,
        regions.as_slice(),
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

/// Lowers a packed [`CoordinateBasisOperation`] directly into the current StableHLO block.
fn lower_coordinate_basis_to_mlir<'b, 'c: 'b, 't: 'c, B, L>(
    operation: &CoordinateBasisOperation<ArrayType>,
    output_types: &[ArrayType],
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    check_count!("output", output_types, 1, ProgramError);
    let output_type = &output_types[0];
    let leaf_dimensions = operation
        .leaf_type()
        .shape()
        .dimensions()
        .iter()
        .map(|dimension_size| match dimension_size {
            Dimension::Static(dimension_size) => Ok(*dimension_size),
            Dimension::Dynamic(_) => Err(ProgramError::InvalidArgument {
                message: format!(
                    "coordinate basis requires a fully static leaf type but got {}",
                    operation.leaf_type(),
                ),
            }),
        })
        .collect::<Result<Vec<_>, _>>()?;

    // A zero-sized leaf contributes no local coordinates. Lowering its fragment directly as a typed empty zero avoids
    // constructing irrelevant row-major index arithmetic and, in particular, makes stride computation independent of
    // where the zero-sized dimension occurs.
    if leaf_dimensions.contains(&0) {
        return lower_constant_output(output_types, 0, block, context, location);
    }

    let index_type = output_type.clone().with_data_type(DataType::U64);
    let index_tensor_type = lower_tensor_type(&index_type, context, location)?;
    let basis_index = block.append_operation(stable_hlo::iota(index_tensor_type, 0, location)?)?;
    let basis_index = basis_index.result(0).expect("stablehlo.iota should return one result").as_ref();

    // Compute each leaf element's row-major flat coordinate in the physical `[basis] ++ leaf_shape` tensor. Keeping
    // all index arithmetic in u64 preserves exact coordinates throughout the generated graph.
    let mut flat_coordinate = None;
    let mut stride = 1u64;
    for (leaf_axis, dimension_size) in leaf_dimensions.iter().copied().enumerate().rev() {
        let coordinate = block.append_operation(stable_hlo::iota(index_tensor_type, leaf_axis + 1, location)?)?;
        let coordinate = coordinate.result(0).expect("stablehlo.iota should return one result").as_ref();
        let coordinate = if stride == 1 {
            coordinate
        } else {
            let stride_value =
                lower_u64_constant_splat(stride, &index_type, index_tensor_type, block, context, location)?;
            let product = block.append_operation(stable_hlo::multiply(coordinate, stride_value, location)?)?;
            product.result(0).expect("stablehlo.multiply should return one result").as_ref()
        };
        flat_coordinate = Some(match flat_coordinate {
            Some(accumulated) => {
                let sum = block.append_operation(stable_hlo::add(accumulated, coordinate, location)?)?;
                sum.result(0).expect("stablehlo.add should return one result").as_ref()
            }
            None => coordinate,
        });
        stride = stride
            .checked_mul(u64::try_from(dimension_size).map_err(|_| ProgramError::InvalidArgument {
                message: format!("leaf dimension {dimension_size} does not fit in u64"),
            })?)
            .ok_or_else(|| ProgramError::InvalidArgument {
                message: format!("coordinate count overflows u64 for leaf type {}", operation.leaf_type()),
            })?;
    }
    let mut flat_coordinate = match flat_coordinate {
        Some(flat_coordinate) => flat_coordinate,
        None => lower_u64_constant_splat(0, &index_type, index_tensor_type, block, context, location)?,
    };
    if operation.coordinate_offset() != 0 {
        let offset = u64::try_from(operation.coordinate_offset()).map_err(|_| ProgramError::InvalidArgument {
            message: format!("coordinate offset {} does not fit in u64", operation.coordinate_offset()),
        })?;
        let offset = lower_u64_constant_splat(offset, &index_type, index_tensor_type, block, context, location)?;
        let sum = block.append_operation(stable_hlo::add(flat_coordinate, offset, location)?)?;
        flat_coordinate = sum.result(0).expect("stablehlo.add should return one result").as_ref();
    }

    let selected = block.append_operation(stable_hlo::compare(
        basis_index,
        flat_coordinate,
        stable_hlo::ComparisonDirection::Equal,
        stable_hlo::ComparisonType::Unsigned,
        location,
    )?)?;
    let selected = selected.result(0).expect("stablehlo.compare should return one result").as_ref();
    let one = lower_unplaced_constant_output(output_types, 1, block, context, location)?[0];
    let zero = lower_unplaced_constant_output(output_types, 0, block, context, location)?[0];
    let result = block.append_operation(stable_hlo::select(selected, one, zero, location)?)?;
    let result = result.result(0).expect("stablehlo.select should return one result").as_ref();
    Ok(vec![annotate_output_memory(result, output_type, block, context, location)?])
}

/// Lowers an exact `u64` value as a splatted constant of `output_type`.
fn lower_u64_constant_splat<'b, 'c: 'b, 't: 'c, B, L>(
    value: u64,
    output_type: &ArrayType,
    output_tensor_type: TensorTypeRef<'c, 't>,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    let scalar_tensor_type = lower_tensor_type(&ArrayType::scalar(DataType::U64), context, location)?;
    let attribute = context
        .dense_u64_elements_attribute(scalar_tensor_type, &[value])
        .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: DataType::U64 })?;
    let constant = block.append_operation(stable_hlo::constant(attribute, location)?)?;
    let constant = constant.result(0).expect("stablehlo.constant should return one result").as_ref();
    if output_type.shape().dimensions().is_empty() {
        return Ok(constant);
    }
    let broadcast = block.append_operation(stable_hlo::broadcast(constant, output_tensor_type, &[], location)?)?;
    Ok(broadcast.result(0).expect("stablehlo.broadcast should return one result").as_ref())
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
/// standard SPMD emission for cross-partition collectives. Explicit logical participant groups are expanded inside
/// every fixed coordinate of the mesh's other axes without reordering either groups or their members. `PSum`/`PMean`
/// reduce with `add` (a `PMean` divides by the effective group size) and `PMax` reduces with `maximum`, reusing the
/// same scalar combiner regions as `stablehlo.reduce` lowering. A collective lowered outside any manual region, or
/// naming an axis the innermost region does not bind as a manual mesh axis, is an error.
/// Resolves the replica groups of one named mesh axis inside the innermost enclosing `shard_map` manual region,
/// returning the groups together with the axis size. Devices are linearized row-major over the mesh axes, so the
/// devices along the named axis with all other coordinates fixed form one replica group: the group base ids
/// enumerate the other axes' coordinate combinations, and the group members step by the named axis's row-major
/// stride. Errors outside manual regions and for axes the region does not bind as manual mesh axes.
fn mesh_axis_replica_groups(
    collective_state: &CollectiveLoweringState,
    axis_name: &str,
) -> Result<(Vec<Vec<usize>>, usize), LoweringError> {
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
    Ok((replica_groups, axis_size))
}

/// Resolves the physical replica groups for one collective, expanding each logical subgroup within every fixed
/// coordinate of the enclosing mesh's other axes.
fn collective_replica_groups(
    collective_state: &CollectiveLoweringState,
    axis_name: &str,
    axis_size: usize,
    axis_index_groups: Option<&[Vec<usize>]>,
) -> Result<(Vec<Vec<usize>>, usize), LoweringError> {
    let (mesh_groups, mesh_axis_size) = mesh_axis_replica_groups(collective_state, axis_name)?;
    if axis_size != mesh_axis_size {
        return Err(ProgramError::MalformedProgram(format!(
            "collective over axis '{axis_name}' records size {axis_size}, but the enclosing mesh axis has size \
             {mesh_axis_size}",
        ))
        .into());
    }
    let Some(axis_index_groups) = axis_index_groups else {
        return Ok((mesh_groups, mesh_axis_size));
    };
    let group_size = axis_index_groups.first().map(Vec::len).unwrap_or(0);
    let mut replica_groups = Vec::with_capacity(mesh_groups.len() * axis_index_groups.len());
    for mesh_group in mesh_groups {
        for axis_index_group in axis_index_groups {
            replica_groups.push(
                axis_index_group
                    .iter()
                    .map(|&index| {
                        mesh_group.get(index).copied().ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "collective over axis '{axis_name}' has group member {index} outside axis size \
                                 {mesh_axis_size}",
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
    }
    Ok((replica_groups, group_size))
}

/// Removes one statically singleton tensor axis while preserving every remaining dynamic dimension.
fn collapse_singleton_axis<'b, 'c: 'b, 't: 'c>(
    value: ValueRef<'b, 'c, 't>,
    output_array_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let output_type = lower_tensor_type(output_array_type, context, location)?;
    let reshape = block.append_operation(stable_hlo::reshape_with_output_type(value, output_type, location)?)?;
    Ok(reshape.result(0).expect("stablehlo.reshape should return one result").as_ref())
}

/// Lowers one traced `all_gather` to a channeled `stablehlo.all_gather` over the named mesh axis's replica groups.
pub(super) fn lower_all_gather_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &AllGatherOperation,
    collective_state: &CollectiveLoweringState,
    input_value: ValueRef<'b, 'c, 't>,
    output_array_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let (replica_groups, _) = collective_replica_groups(
        collective_state,
        operation.axis_name(),
        operation.axis_size(),
        operation.options().axis_index_groups(),
    )?;
    let replica_groups: Vec<&[usize]> = replica_groups.iter().map(Vec::as_slice).collect();
    let input_value = match operation.options().mode() {
        CollectiveMode::Tiled => input_value,
        CollectiveMode::Untiled => {
            let mut dimensions = output_array_type.shape().dimensions().to_vec();
            dimensions[operation.concat_axis()] = Dimension::Static(1);
            let input_type =
                lower_tensor_type(&output_array_type.clone().with_shape(Shape::new(dimensions)), context, location)?;
            let broadcast_dimensions =
                (0..output_array_type.rank()).filter(|&axis| axis != operation.concat_axis()).collect::<Vec<_>>();
            let broadcast = block.append_operation(stable_hlo::broadcast(
                input_value,
                input_type,
                broadcast_dimensions.as_slice(),
                location,
            )?)?;
            broadcast.result(0).expect("stablehlo.broadcast_in_dim should return one result").as_ref()
        }
    };
    let output_type = lower_tensor_type(output_array_type, context, location)?;
    let result = block.append_operation(stable_hlo::all_gather(
        &[input_value],
        operation.concat_axis(),
        stable_hlo::ReplicaGroups::dense(replica_groups.as_slice()),
        Some(collective_state.next_channel_id()),
        Some(stable_hlo::ChannelHandleType::DeviceToDevice),
        true,
        &[output_type],
        location,
    )?)?;
    Ok(vec![result.result(0).expect("stablehlo.all_gather should return one result").as_ref()])
}

/// Lowers one traced `psum_scatter` to a channeled `stablehlo.reduce_scatter` with a sum reduction over the named
/// mesh axis's replica groups.
pub(super) fn lower_psum_scatter_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &PSumScatterOperation,
    collective_state: &CollectiveLoweringState,
    input_value: ValueRef<'b, 'c, 't>,
    output_array_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let (replica_groups, _) = collective_replica_groups(
        collective_state,
        operation.axis_name(),
        operation.axis_size(),
        operation.options().axis_index_groups(),
    )?;
    let replica_groups: Vec<&[usize]> = replica_groups.iter().map(Vec::as_slice).collect();
    let computation = build_reduce_body_region(ReductionKind::Sum, output_array_type.data_type(), context, location)?;
    let native_output_type = match operation.options().mode() {
        CollectiveMode::Tiled => output_array_type.clone(),
        CollectiveMode::Untiled => output_array_type
            .with_inserted_dimension(operation.scatter_axis(), Dimension::Static(1))
            .map_err(ProgramError::from)?,
    };
    let output_type = lower_tensor_type(&native_output_type, context, location)?;
    let result = block.append_operation(stable_hlo::reduce_scatter(
        input_value,
        operation.scatter_axis(),
        stable_hlo::ReplicaGroups::dense(replica_groups.as_slice()),
        Some(collective_state.next_channel_id()),
        Some(stable_hlo::ChannelHandleType::DeviceToDevice),
        true,
        computation,
        output_type,
        location,
    )?)?;
    let result = result.result(0).expect("stablehlo.reduce_scatter should return one result").as_ref();
    if operation.options().mode() == CollectiveMode::Tiled {
        return Ok(vec![result]);
    }
    Ok(vec![collapse_singleton_axis(result, output_array_type, block, context, location)?])
}

/// Lowers one traced `ppermute` to a channeled `stablehlo.collective_permute`, expanding the axis-local
/// `(source, target)` pairs to global device pairs across the named mesh axis's replica groups.
fn lower_ppermute_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &PpermuteOperation,
    collective_state: &CollectiveLoweringState,
    input_value: ValueRef<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let (replica_groups, _) = mesh_axis_replica_groups(collective_state, operation.axis_name())?;
    let mut source_target_pairs = Vec::with_capacity(replica_groups.len() * operation.source_target_pairs().len());
    for group in &replica_groups {
        for (source, target) in operation.source_target_pairs() {
            source_target_pairs.push((group[*source], group[*target]));
        }
    }
    let result = block.append_operation(stable_hlo::collective_permute(
        input_value,
        source_target_pairs.as_slice(),
        Some(collective_state.next_channel_id()),
        Some(stable_hlo::ChannelHandleType::DeviceToDevice),
        location,
    )?)?;
    Ok(vec![result.result(0).expect("stablehlo.collective_permute should return one result").as_ref()])
}

/// Lowers one traced `all_to_all` to a channeled `stablehlo.all_to_all` over the named mesh axis's replica groups.
pub(super) fn lower_all_to_all_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &AllToAllOperation,
    collective_state: &CollectiveLoweringState,
    input_value: ValueRef<'b, 'c, 't>,
    input_array_type: &ArrayType,
    output_array_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let (replica_groups, axis_size) = collective_replica_groups(
        collective_state,
        operation.axis_name(),
        operation.axis_size(),
        operation.options().axis_index_groups(),
    )?;
    let replica_groups: Vec<&[usize]> = replica_groups.iter().map(Vec::as_slice).collect();
    let (input_value, split_axis, concat_axis) = match operation.options().mode() {
        CollectiveMode::Tiled => (input_value, operation.split_axis(), operation.concat_axis()),
        CollectiveMode::Untiled if operation.split_axis() == operation.concat_axis() => {
            (input_value, operation.split_axis(), operation.concat_axis())
        }
        CollectiveMode::Untiled => {
            let mut split_axis = operation.split_axis();
            let mut concat_axis = operation.concat_axis();
            if split_axis < concat_axis {
                concat_axis += 1;
            } else {
                split_axis += 1;
            }
            let expanded_type = input_array_type
                .with_inserted_dimension(concat_axis, Dimension::Static(1))
                .map_err(ProgramError::from)?;
            let expanded_type = lower_tensor_type(&expanded_type, context, location)?;
            let broadcast_dimensions =
                (0..input_array_type.rank() + 1).filter(|&axis| axis != concat_axis).collect::<Vec<_>>();
            let expanded = block.append_operation(stable_hlo::broadcast(
                input_value,
                expanded_type,
                broadcast_dimensions.as_slice(),
                location,
            )?)?;
            (
                expanded.result(0).expect("stablehlo.broadcast_in_dim should return one result").as_ref(),
                split_axis,
                concat_axis,
            )
        }
    };
    let result = block.append_operation(stable_hlo::all_to_all(
        &[input_value],
        split_axis,
        axis_size,
        concat_axis,
        stable_hlo::ReplicaGroups::dense(replica_groups.as_slice()),
        Some(collective_state.next_channel_id()),
        Some(stable_hlo::ChannelHandleType::DeviceToDevice),
        true,
        location,
    )?)?;
    let result = result.result(0).expect("stablehlo.all_to_all should return one result").as_ref();
    if operation.options().mode() == CollectiveMode::Tiled || split_axis == concat_axis {
        return Ok(vec![result]);
    }
    Ok(vec![collapse_singleton_axis(result, output_array_type, block, context, location)?])
}

fn lower_collective_to_all_reduce<'b, 'c: 'b, 't: 'c>(
    operation: &CollectiveOperation,
    collective_state: &CollectiveLoweringState,
    input_value: ValueRef<'b, 'c, 't>,
    output_array_type: &ArrayType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let (replica_groups, effective_axis_size) = match operation.axis_index_groups() {
        Some(axis_index_groups) => collective_replica_groups(
            collective_state,
            operation.axis_name(),
            operation.axis_size().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "grouped '{}' over axis '{}' does not record the full axis size",
                    operation.name(),
                    operation.axis_name(),
                ))
            })?,
            Some(axis_index_groups),
        )?,
        None => mesh_axis_replica_groups(collective_state, operation.axis_name())?,
    };
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
    // `PMean` is the mean over the effective participant group: divide the all-reduced sum by the group size.
    let output_tensor_type = lower_tensor_type(output_array_type, context, location)?;
    let divisor = lower_f64_constant_splat(
        effective_axis_size as f64,
        output_array_type,
        output_tensor_type,
        block,
        context,
        location,
    )?;
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
    let output_type = ArrayType::scalar(DataType::U64);
    let output_tensor_type = lower_tensor_type(&output_type, context, location)?;
    lower_u64_constant_splat(value, &output_type, output_tensor_type, block, context, location)
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
/// A [`ReductionKind::Mean`] reduction lowers as the sum divided by the number of reduced elements.
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
    let reduced = result.result(0).expect("stablehlo.reduce should return one result").as_ref();
    if !matches!(kind, ReductionKind::Mean) {
        return Ok(reduced);
    }
    // `Mean` is the sum divided by the number of reduced elements. The divisor is staged as a splatted constant of
    // the output type, clamped to one so that a zero-sized reduction yields zero rather than NaN, mirroring the
    // eager reference backend.
    let input_type = input_value.r#type()?;
    let input_tensor_type = input_type.cast::<TensorTypeRef>().ok_or_else(|| LoweringError::UnsupportedOp {
        op: format!("'reduce' operand has non-tensor MLIR type '{input_type}'"),
    })?;
    let dimensions = input_tensor_type.dimensions().collect::<Vec<_>>();
    let mut count = 1usize;
    for axis in axes {
        match dimensions.get(*axis) {
            Some(MlirSize::Static(size)) => count *= size,
            _ => {
                return Err(LoweringError::UnsupportedOp {
                    op: format!("'reduce' mean over dynamically sized axis {axis}"),
                });
            }
        }
    }
    let output_tensor_type = lower_tensor_type(output_array_type, context, location)?;
    let divisor =
        lower_f64_constant_splat(count.max(1) as f64, output_array_type, output_tensor_type, block, context, location)?;
    let mean = block.append_operation(stable_hlo::divide(reduced, divisor, location)?)?;
    Ok(mean.result(0).expect("stablehlo.divide should return one result").as_ref())
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
/// the given element type. `Sum` and `Mean` use zero; `Max` and `Min` use the bound returned by
/// [`float_reduction_identity_bound`] (negated for `Max`) at float element types and the bounds returned by
/// [`integer_reduction_identity_bounds`] at integer element types; Boolean `Any` and `All` use `false` and `true`.
/// Every other combination — including all reductions over [`DataType::F8E8M0FNU`], whose unsigned, zero-free
/// encoding has no representable identity — fails with [`LoweringError::UnsupportedDataType`].
fn build_reduction_identity_attribute<'c, 't>(
    kind: ReductionKind,
    element_type: DataType,
    tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
    context: &'c MlirContext<'t>,
) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
    if let Some(bound) = float_reduction_identity_bound(element_type) {
        let identity = match kind {
            ReductionKind::Sum | ReductionKind::Mean => 0.0,
            ReductionKind::Max => -bound,
            ReductionKind::Min => bound,
            ReductionKind::Any | ReductionKind::All => {
                return Err(LoweringError::UnsupportedDataType { data_type: element_type });
            }
        };
        return lower_f64_scalar_elements_attribute(element_type, tensor_type, identity, context);
    }
    if let Some((minimum, maximum)) = integer_reduction_identity_bounds(element_type) {
        let identity = match kind {
            ReductionKind::Sum | ReductionKind::Mean => 0,
            ReductionKind::Max => minimum,
            ReductionKind::Min => maximum,
            ReductionKind::Any | ReductionKind::All => {
                return Err(LoweringError::UnsupportedDataType { data_type: element_type });
            }
        };
        return lower_constant_elements_attribute(element_type, tensor_type, identity, context);
    }
    match (kind, element_type) {
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

/// Returns the `Min` reduction identity of the given float data type: positive infinity for formats with
/// infinities and the greatest finite value for the finite-only `f8`/`f6`/`f4` formats (for example, `448` for
/// [`DataType::F8E4M3FN`]). The `Max` identity is its negation, which every supported format represents because
/// its value range is sign-symmetric. Returns `None` for non-float data types and for [`DataType::F8E8M0FNU`],
/// whose unsigned, zero-free exponent-only encoding represents neither zero nor a sign-symmetric ordering bound.
fn float_reduction_identity_bound(data_type: DataType) -> Option<f64> {
    match data_type {
        DataType::BF16
        | DataType::F16
        | DataType::F32
        | DataType::F64
        | DataType::F8E3M4
        | DataType::F8E4M3
        | DataType::F8E5M2 => Some(f64::INFINITY),
        DataType::F4E2M1FN => Some(6.0),
        DataType::F6E2M3FN => Some(7.5),
        DataType::F6E3M2FN => Some(28.0),
        DataType::F8E4M3FN => Some(448.0),
        DataType::F8E4M3FNUZ => Some(240.0),
        DataType::F8E4M3B11FNUZ => Some(30.0),
        DataType::F8E5M2FNUZ => Some(57344.0),
        _ => None,
    }
}

/// Returns the (minimum, maximum) values representable by the given integer data type, used as the `Max` and `Min`
/// reduction identities. The values are raw `i64` payloads for [`lower_constant_elements_attribute`], which encodes
/// them at the type's width and signedness — [`DataType::U64`]'s maximum therefore wraps to `-1`, whose
/// two's-complement bits are exactly `u64::MAX`. Returns `None` for non-integer data types.
fn integer_reduction_identity_bounds(data_type: DataType) -> Option<(i64, i64)> {
    if data_type.is_signed() {
        let width = signed_integer_width(data_type).unwrap();
        Some((i64::MIN >> (64 - width), i64::MAX >> (64 - width)))
    } else if data_type.is_unsigned() {
        let width = unsigned_integer_width(data_type).unwrap();
        Some((0, (u64::MAX >> (64 - width)) as i64))
    } else {
        None
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
            Dimension::Static(value) => MlirSize::Static(*value),
            Dimension::Dynamic(_) => MlirSize::Dynamic,
        })
        .collect::<Vec<_>>();
    let bounds = array_type.shape().dimensions().iter().map(stable_hlo_dynamic_dimension_bound).collect::<Vec<_>>();
    let encoding = bounds
        .iter()
        .any(Option::is_some)
        .then(|| context.stable_hlo_tensor_type_extensions(bounds.as_slice()))
        .transpose()?
        .map(|attribute| attribute.as_ref());
    context
        .tensor_type(element_type, dimensions.as_slice(), encoding, location)
        .map_err(|_| LoweringError::InvalidTensorType { array_type: array_type.clone() })
}

/// Converts a Ryft exclusive dynamic dimension bound to StableHLO's inclusive maximum dimension size.
fn stable_hlo_dynamic_dimension_bound(size: &Dimension) -> Option<usize> {
    match size {
        Dimension::Static(_) => None,
        Dimension::Dynamic(variable) => variable.bounds().upper().and_then(|upper| upper.checked_sub(1)),
    }
}

/// Lowers one [`DataType`] to the corresponding MLIR element type.
fn lower_element_type<'c, 't>(
    data_type: DataType,
    context: &'c MlirContext<'t>,
) -> Result<TypeRef<'c, 't>, LoweringError> {
    Ok(match data_type {
        DataType::Token => return Err(LoweringError::UnsupportedDataType { data_type }),
        // StableHLO has no zero-information element type. Use an i1 tensor as an unobservable physical carrier while
        // retaining the logical `DataType::Zero` in Ryft's lowered/compiled metadata.
        DataType::Zero => context.signless_integer_type(1).as_ref(),
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
/// zero/one synthesis and literal-constant lowering.
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

/// Lowers a [`Scalar`] literal after ordinary conversion to `output_type`'s element type. Integer values use exact
/// integer attributes, floating-point and Boolean values route through [`lower_f64_constant_splat`], and complex
/// values compose two real scalar constants through `stablehlo.complex` because MLIR has no complex scalar attribute
/// to splat.
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
    let value = value.convert_element_type(data_type)?;
    if data_type.is_zero() {
        if value != Scalar::Zero {
            return Err(LoweringError::UnsupportedDataType { data_type });
        }
        return lower_f64_constant_splat(0.0, output_type, output_tensor_type, block, context, location);
    }
    if data_type.is_complex() {
        let (real, imaginary) = match (data_type, value) {
            (DataType::C64, Scalar::C64(value)) => (value.re as f64, value.im as f64),
            (DataType::C128, Scalar::C128(value)) => (value.re, value.im),
            _ => unreachable!("conversion to a complex data type yields its matching complex scalar"),
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
        return Ok(complex.result(0).expect("stablehlo.complex should return one result").as_ref());
    }
    let integer_value = match value {
        Scalar::I8(value) => Some(i64::from(value)),
        Scalar::I16(value) => Some(i64::from(value)),
        Scalar::I32(value) => Some(i64::from(value)),
        Scalar::I64(value) => Some(value),
        Scalar::U8(value) => Some(i64::from(value)),
        Scalar::U16(value) => Some(i64::from(value)),
        Scalar::U32(value) => Some(i64::from(value)),
        Scalar::U64(value) => {
            let elements = context
                .dense_u64_elements_attribute(output_tensor_type, &[value])
                .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type })?;
            let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
            return Ok(constant.result(0).expect("stablehlo.constant should return one result").as_ref());
        }
        _ => None,
    };
    if let Some(integer_value) = integer_value {
        let elements = lower_constant_elements_attribute(data_type, output_tensor_type, integer_value, context)?;
        let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
        return Ok(constant.result(0).expect("stablehlo.constant should return one result").as_ref());
    }
    let Scalar::F64(value) = value.promote_element_type(DataType::F64)? else {
        unreachable!("promotion to f64 yields an f64 scalar")
    };
    lower_f64_constant_splat(value, output_type, output_tensor_type, block, context, location)
}

/// Builds a splatted dense-elements attribute holding `factor` converted to the given `data_type`. Boolean splats
/// hold `factor != 0.0`, integer splats truncate `factor`, and float splats round `factor` to the nearest
/// representable value of the element type (covering every float format that [`lower_element_type`] supports,
/// including the `f8`/`f6`/`f4` families). [`DataType::Zero`] admits only a zero factor, lowered as its `i1` carrier.
fn lower_f64_scalar_elements_attribute<'c, 't>(
    data_type: DataType,
    tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
    factor: f64,
    context: &'c MlirContext<'t>,
) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
    match data_type {
        DataType::Zero if factor == 0.0 => context
            .splatted_dense_attribute_elements_attribute(tensor_type, context.boolean_attribute(false))
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type }),
        DataType::Zero => Err(LoweringError::UnsupportedDataType { data_type }),
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
        DataType::BF16
        | DataType::F16
        | DataType::F32
        | DataType::F64
        | DataType::F4E2M1FN
        | DataType::F6E2M3FN
        | DataType::F6E3M2FN
        | DataType::F8E3M4
        | DataType::F8E4M3
        | DataType::F8E4M3FN
        | DataType::F8E4M3FNUZ
        | DataType::F8E4M3B11FNUZ
        | DataType::F8E5M2
        | DataType::F8E5M2FNUZ
        | DataType::F8E8M0FNU => {
            let element_type = lower_element_type(data_type, context)?
                .cast::<FloatTypeRef>()
                .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type })?;
            context
                .splatted_dense_attribute_elements_attribute(tensor_type, context.float_attribute(element_type, factor))
                .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type })
        }
        DataType::Token | DataType::C64 | DataType::C128 => Err(LoweringError::UnsupportedDataType { data_type }),
    }
}

/// Builds a splatted dense-elements attribute holding `integer_value` converted to the given `data_type`. Integer
/// splats hold the exact `i64` value; every other data type routes through
/// [`lower_f64_scalar_elements_attribute`] after a lossless widening of `integer_value` to `f64`.
fn lower_constant_elements_attribute<'c, 't>(
    data_type: DataType,
    tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
    integer_value: i64,
    context: &'c MlirContext<'t>,
) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
    match data_type {
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
        _ => lower_f64_scalar_elements_attribute(data_type, tensor_type, integer_value as f64, context),
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

    use ryft_mlir::dialects::builtin::attributes::DenseElementsAttribute;

    use ryft_core::backends::arrays::Array as CpuArray;
    use ryft_core::backends::arrays::ArrayOperation;
    use ryft_core::contexts::Context;
    use ryft_core::differentiation::ReverseModeDifferentiate;
    use ryft_core::operations::compare::CompareOperation;
    use ryft_core::operations::constants::{
        ConstantOperation, Fill, OneLike, OneLikeOperation, OneOperation, ZeroLike, ZeroOperation,
    };
    use ryft_core::operations::control_flow::SelectOperation;
    use ryft_core::operations::logical::{AndOperation, OrOperation, XorOperation};
    use ryft_core::operations::manipulation::{
        ConcatenateOperation, DynamicSliceOperation, DynamicUpdateSliceOperation, LegacyBroadcastOperation,
        LegacyReshapeOperation, PadOperation, SliceOperation, Transpose, UpdateSliceOperation,
    };
    use ryft_core::operations::math::{
        Atan2Operation, Cos, DivOperation, Dot, DotDimensionNumbers, ReduceOperation, Sin,
    };
    use ryft_core::parameters::Placeholder;
    use ryft_core::programs::builders::ProgramBuilder;
    use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use ryft_core::types::dimensions::{DimensionBounds, DimensionVariable};
    use ryft_core::types::{Dimension, Shape};
    use ryft_core::{EagerContext, TypeError};

    use super::super::shard_map::{TracedShardMap, shard_map as traced_shard_map};
    use ryft_core::tracing::{Trace, TracingContext};

    use crate::tests::values_to_bytes;

    use super::*;

    fn test_manual_mesh(axis_name: &str, axis_size: usize) -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::new(axis_name, axis_size, MeshAxisType::Manual).unwrap()]).unwrap()
    }

    fn test_vector_type(length: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(length)]))
    }

    fn test_matrix_type(rows: usize, cols: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(rows), Dimension::Static(cols)]))
    }

    /// Creates a rank-one literal whose element type is inferred from its homogeneous scalar payload.
    fn test_literal(values: Vec<Scalar>) -> CpuArray {
        let data_type = values.first().unwrap().r#type().into_owned();
        CpuArray::new(ArrayType::new(data_type, Shape::new(vec![Dimension::Static(values.len())])), values).unwrap()
    }

    /// Copies the exact raw storage bytes from the MLIR dense attribute built for `literal`.
    fn test_literal_dense_bytes(literal: &CpuArray, byte_count: usize) -> Vec<u8> {
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = lower_tensor_type(literal.r#type().as_ref(), &context, location).unwrap();
        let attribute = literal.to_dense_elements_attribute(tensor_type, &context).unwrap();
        unsafe { std::slice::from_raw_parts(attribute.raw_data().cast::<u8>(), byte_count).to_vec() }
    }

    fn dynamic_dimension(name: &str, exclusive_upper_bound: Option<usize>) -> Dimension {
        DimensionVariable::new(name, DimensionBounds::non_negative(exclusive_upper_bound).unwrap()).into()
    }

    fn xla_identity_branch(input_type: ArrayType) -> FlatXlaProgram {
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn xla_neg_branch(input_type: ArrayType) -> FlatXlaProgram {
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let output = builder.add_instruction(NegOperation, Vec::new(), vec![input]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn lower_traced_module(
        traced: &TracedShardMap<ArrayType, ArrayType>,
        function_name: &str,
    ) -> Result<String, super::super::shard_map::ShardMapTraceError> {
        traced.to_mlir_module(function_name)
    }

    fn xla_elementwise_normalization_program() -> FlatXlaProgram {
        let mut builder = XlaProgramBuilder::new();
        let scalar = builder.add_input(ArrayType::scalar(DataType::F32));
        let left = builder
            .add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Static(1)])));
        let right = builder
            .add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1), Dimension::Static(4)])));
        let condition = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let boolean_vector =
            builder.add_input(ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(4)])));
        let divide = builder.add_instruction(DivOperation, Vec::new(), vec![left, right]).unwrap()[0];
        let atan2 = builder.add_instruction(Atan2Operation, Vec::new(), vec![right, left]).unwrap()[0];
        let compare = builder
            .add_instruction(CompareOperation::new(ComparisonDirection::GreaterThan), Vec::new(), vec![left, right])
            .unwrap()[0];
        let select = builder.add_instruction(SelectOperation, Vec::new(), vec![condition, scalar, right]).unwrap()[0];
        let and = builder.add_instruction(AndOperation, Vec::new(), vec![condition, boolean_vector]).unwrap()[0];
        let or = builder.add_instruction(OrOperation, Vec::new(), vec![boolean_vector, condition]).unwrap()[0];
        let xor = builder.add_instruction(XorOperation, Vec::new(), vec![condition, boolean_vector]).unwrap()[0];
        builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![divide, atan2, compare, select, and, or, xor],
                vec![Placeholder; 5],
                vec![Placeholder; 7],
            )
            .unwrap()
    }

    fn assert_elementwise_operands_are_normalized(stablehlo: &str) {
        for operation in ["divide", "atan2", "compare", "select", "and", "or", "xor"] {
            assert!(stablehlo.contains(&format!("stablehlo.{operation}")), "{stablehlo}");
        }
        assert!(stablehlo.contains("tensor<3x4xf64>"), "{stablehlo}");
        assert!(stablehlo.contains("tensor<3x4xi1>"), "{stablehlo}");
        assert!(stablehlo.matches("stablehlo.convert").count() >= 4, "{stablehlo}");
        assert!(stablehlo.matches("stablehlo.broadcast_in_dim").count() >= 8, "{stablehlo}");
    }

    #[test]
    fn test_broadcast_sharding_transition_detection() {
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let replicated = test_vector_type(4).with_sharding(Sharding::replicated(explicit_mesh.clone(), 1)).unwrap();
        let explicitly_sharded = test_vector_type(4)
            .with_sharding(Sharding::new(explicit_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert!(broadcast_changes_explicit_sharding(&test_vector_type(4), &replicated, &[0]));
        assert!(broadcast_changes_explicit_sharding(&replicated, &explicitly_sharded, &[0]));
        assert!(!broadcast_changes_explicit_sharding(&explicitly_sharded, &explicitly_sharded, &[0]));

        let first_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let second_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let first_mesh_replicated = test_vector_type(4).with_sharding(Sharding::replicated(first_mesh, 1)).unwrap();
        let second_mesh_replicated = test_vector_type(4).with_sharding(Sharding::replicated(second_mesh, 1)).unwrap();
        assert!(broadcast_changes_explicit_sharding(&first_mesh_replicated, &second_mesh_replicated, &[0]));

        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let replicated = test_vector_type(4).with_sharding(Sharding::replicated(manual_mesh.clone(), 1)).unwrap();
        let manually_sharded = test_vector_type(4)
            .with_sharding(
                Sharding::new(manual_mesh, vec![ShardingDimension::sharded(["x"])])
                    .unwrap()
                    .with_varying_manual_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        assert!(!broadcast_changes_explicit_sharding(&replicated, &manually_sharded, &[0]));

        // Expanding a partitioned singleton and changing explicit reduction state both require constraints even when
        // the ranked sharding dimensions themselves remain unchanged.
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let singleton = test_vector_type(1)
            .with_sharding(Sharding::new(explicit_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let expanded = test_vector_type(4)
            .with_sharding(Sharding::new(explicit_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert!(broadcast_changes_explicit_sharding(&singleton, &expanded, &[0]));

        let unreduced = test_vector_type(4)
            .with_sharding(Sharding::replicated(explicit_mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let reduced = test_vector_type(4)
            .with_sharding(Sharding::replicated(explicit_mesh, 1).with_reduced_axes(["x"]).unwrap())
            .unwrap();
        assert!(broadcast_changes_explicit_sharding(&unreduced, &reduced, &[0]));
    }

    #[test]
    fn test_plain_broadcast_explicit_singleton_expansion_lowers_to_constraint() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = test_vector_type(1)
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let output_type = test_vector_type(4)
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let mut builder = ryft_core::ProgramBuilder::<CpuArray, LegacyBroadcastOperation>::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(LegacyBroadcastOperation::new(output_type, vec![0]), Vec::new(), vec![input])
            .unwrap()[0];
        let program = builder
            .build::<Vec<CpuArray>, Vec<CpuArray>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<1xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1xf32>) -> tensor<4xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{"x"}]> : tensor<4xf32>
                    return %1 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_plain_reshape_dimensions_lower_transpose_before_reshape() {
        let input_type = test_matrix_type(2, 3);
        let mut builder = ryft_core::ProgramBuilder::<CpuArray, LegacyReshapeOperation>::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(
                LegacyReshapeOperation::new(
                    ReshapeParameters::new(Shape::new(vec![Dimension::Static(6)])).with_dimensions([1, 0]),
                ),
                Vec::new(),
                vec![input],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<CpuArray>, Vec<CpuArray>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<6xf32> {
                    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
                    %1 = stablehlo.reshape %0 : (tensor<3x2xf32>) -> tensor<6xf32>
                    return %1 : tensor<6xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_plain_fixed_dynamic_identity_reshape_lowers_without_an_operation() {
        let shape = Shape::new(vec![dynamic_dimension("rows", None), Dimension::Static(3)]);
        let mut builder = ryft_core::ProgramBuilder::<CpuArray, LegacyReshapeOperation>::new();
        let input = builder.add_input(ArrayType::new(DataType::F32, shape.clone()));
        let output = builder.add_instruction(LegacyReshapeOperation::new(shape), Vec::new(), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<CpuArray>, Vec<CpuArray>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<?x3xf32>) -> tensor<?x3xf32> {
                    return %arg0 : tensor<?x3xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_plain_fixed_dynamic_permuted_reshape_lowers_to_transpose() {
        let rows = DimensionVariable::new("rows", DimensionBounds::unbounded());
        let columns = DimensionVariable::new("columns", DimensionBounds::non_negative(Some(5)).unwrap());
        let input_shape = Shape::new(vec![rows.clone().into(), Dimension::Static(3), columns.clone().into()]);
        let output_shape = Shape::new(vec![columns.into(), rows.into(), Dimension::Static(3)]);
        let mut builder = ryft_core::ProgramBuilder::<CpuArray, LegacyReshapeOperation>::new();
        let input = builder.add_input(ArrayType::new(DataType::F32, input_shape));
        let output = builder
            .add_instruction(
                LegacyReshapeOperation::new(ReshapeParameters::new(output_shape).with_dimensions([2, 0, 1])),
                Vec::new(),
                vec![input],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<CpuArray>, Vec<CpuArray>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<?x3x?xf32, #stablehlo.bounds<?, ?, 4>>) -> tensor<?x?x3xf32, #stablehlo.bounds<4, ?, ?>> {
                    %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<?x3x?xf32, #stablehlo.bounds<?, ?, 4>>) -> tensor<?x?x3xf32, #stablehlo.bounds<4, ?, ?>>
                    return %0 : tensor<?x?x3xf32, #stablehlo.bounds<4, ?, ?>>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_plain_symbolic_reshape_lowers_runtime_shape_from_original_input_dimensions() {
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![dynamic_dimension("rows", None), dynamic_dimension("columns", None)]),
        );
        let normalized_input_dimension = |dimension, factor| ReshapeDimensionExpression::ExactDivision {
            numerator: Box::new(ReshapeDimensionExpression::Product(vec![
                ReshapeDimensionExpression::InputDimension(dimension),
                ReshapeDimensionExpression::Constant(factor),
            ])),
            denominator: Box::new(ReshapeDimensionExpression::Constant(factor)),
        };
        let mut builder = ryft_core::ProgramBuilder::<CpuArray, LegacyReshapeOperation>::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(
                LegacyReshapeOperation::new(
                    ReshapeParameters::from_dimension_expressions(vec![
                        normalized_input_dimension(0, 2),
                        normalized_input_dimension(1, 3),
                    ])
                    .with_dimensions([1, 0]),
                ),
                Vec::new(),
                vec![input],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<CpuArray>, Vec<CpuArray>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
                    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<?x?xf32>) -> tensor<?x?xf32>
                    %1 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x?xf32>) -> tensor<i32>
                    %c = stablehlo.constant dense<2> : tensor<i32>
                    %2 = stablehlo.multiply %1, %c : tensor<i32>
                    %c_0 = stablehlo.constant dense<2> : tensor<i32>
                    %3 = stablehlo.divide %2, %c_0 : tensor<i32>
                    %4 = stablehlo.reshape %3 : (tensor<i32>) -> tensor<1xi32>
                    %5 = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<?x?xf32>) -> tensor<i32>
                    %c_1 = stablehlo.constant dense<3> : tensor<i32>
                    %6 = stablehlo.multiply %5, %c_1 : tensor<i32>
                    %c_2 = stablehlo.constant dense<3> : tensor<i32>
                    %7 = stablehlo.divide %6, %c_2 : tensor<i32>
                    %8 = stablehlo.reshape %7 : (tensor<i32>) -> tensor<1xi32>
                    %9 = stablehlo.concatenate %4, %8, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
                    %10 = stablehlo.dynamic_reshape %0, %9 : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
                    return %10 : tensor<?x?xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_plain_symbolic_reshape_refines_inferred_mixed_output_dimensions() {
        let mut builder = ryft_core::ProgramBuilder::<CpuArray, LegacyReshapeOperation>::new();
        let input = builder.add_input(ArrayType::new(
            DataType::F32,
            Shape::new(vec![dynamic_dimension("rows", None), Dimension::Static(6)]),
        ));
        let output = builder
            .add_instruction(
                LegacyReshapeOperation::new(ReshapeParameters::from_dimension_expressions(vec![
                    ReshapeDimensionExpression::InputDimension(0),
                    ReshapeDimensionExpression::Constant(2),
                    ReshapeDimensionExpression::Constant(3),
                ])),
                Vec::new(),
                vec![input],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<CpuArray>, Vec<CpuArray>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<?x6xf32>) -> tensor<?x2x3xf32> {
                    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x6xf32>) -> tensor<i32>
                    %1 = stablehlo.reshape %0 : (tensor<i32>) -> tensor<1xi32>
                    %c = stablehlo.constant dense<2> : tensor<i32>
                    %2 = stablehlo.reshape %c : (tensor<i32>) -> tensor<1xi32>
                    %c_0 = stablehlo.constant dense<3> : tensor<i32>
                    %3 = stablehlo.reshape %c_0 : (tensor<i32>) -> tensor<1xi32>
                    %4 = stablehlo.concatenate %1, %2, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
                    %5 = stablehlo.dynamic_reshape %arg0, %4 : (tensor<?x6xf32>, tensor<3xi32>) -> tensor<?x?x?xf32, #stablehlo.bounds<?, 2, 3>>
                    %cast = tensor.cast %5 : tensor<?x?x?xf32, #stablehlo.bounds<?, 2, 3>> to tensor<?x2x3xf32>
                    return %cast : tensor<?x2x3xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_plain_symbolic_reshape_rejects_derived_dynamic_bounds_without_result_operands() {
        let mut builder = ryft_core::ProgramBuilder::<CpuArray, LegacyReshapeOperation>::new();
        let input = builder.add_input(ArrayType::new(
            DataType::F32,
            Shape::new(vec![dynamic_dimension("rows", Some(6)), Dimension::Static(4)]),
        ));
        assert_eq!(
            builder.add_instruction(
                LegacyReshapeOperation::new(ReshapeParameters::from_dimension_expressions(vec![
                    ReshapeDimensionExpression::Product(vec![
                        ReshapeDimensionExpression::InputDimension(0),
                        ReshapeDimensionExpression::Constant(2),
                    ]),
                    ReshapeDimensionExpression::Constant(2),
                ])),
                Vec::new(),
                vec![input],
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "'reshape' dynamic dimension arithmetic requires explicit result-dimension operands".to_string(),
            ))),
        );
    }

    #[test]
    fn test_reshape_dimension_i64_rejects_out_of_range_values() {
        if usize::MAX <= i64::MAX as usize {
            return;
        }

        assert_eq!(
            reshape_dimension_i64(usize::MAX),
            Err(LoweringError::ReshapeDimensionOutOfRange { value: usize::MAX, bit_width: 64 }),
        );
    }

    #[test]
    fn test_reshape_dimension_i32_rejects_out_of_range_values() {
        let value = i32::MAX as usize + 1;
        assert_eq!(
            reshape_dimension_i32(value),
            Err(LoweringError::ReshapeDimensionOutOfRange { value, bit_width: 32 }),
        );
    }

    #[test]
    fn test_pad_interior_padding_rejects_out_of_range_values() {
        if usize::MAX <= i64::MAX as usize {
            return;
        }

        assert_eq!(
            validate_pad_interior_padding(usize::MAX),
            Err(LoweringError::PadInteriorPaddingOutOfRange { value: usize::MAX }),
        );
    }

    #[test]
    fn test_pad_lowering_casts_inferred_type_to_requested_dynamic_bound() {
        let context = MlirContext::new();
        let location = context.unknown_location();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("input", Some(5))]));
        let padding_value_type = ArrayType::scalar(DataType::F32);
        let requested_output_type =
            ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("output", Some(12))]));
        let input_tensor_type = lower_tensor_type(&input_type, &context, location).unwrap();
        let padding_value_tensor_type = lower_tensor_type(&padding_value_type, &context, location).unwrap();
        let mut block = context.block(&[(input_tensor_type, location), (padding_value_tensor_type, location)]);
        let input = block.argument(0).unwrap().as_ref();
        let padding_value = block.argument(1).unwrap().as_ref();

        let results = lower_pad_to_mlir(
            &PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
            &[input, padding_value],
            std::slice::from_ref(&requested_output_type),
            &mut block,
            &context,
            location,
        )
        .unwrap();

        assert_eq!(
            results[0].r#type().unwrap(),
            lower_tensor_type(&requested_output_type, &context, location).unwrap().as_ref(),
        );
        let operation_names = block
            .operations()
            .unwrap()
            .map(|operation| operation.unwrap().name().as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        assert_eq!(operation_names, vec!["stablehlo.pad", "tensor.cast"]);
    }

    #[test]
    fn test_stable_hlo_dynamic_dimension_bound_converts_exclusive_bounds() {
        assert_eq!(stable_hlo_dynamic_dimension_bound(&Dimension::Static(4)), None);
        assert_eq!(stable_hlo_dynamic_dimension_bound(&dynamic_dimension("unbounded", None)), None);
        assert_eq!(stable_hlo_dynamic_dimension_bound(&dynamic_dimension("bounded", Some(6))), Some(5));
    }

    #[test]
    fn test_reshape_lowering_validates_input_arity_before_indexing() {
        let context = MlirContext::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let mut block = module.body().unwrap();
        let error = lower_reshape_to_mlir(
            &LegacyReshapeOperation::new(Shape::new(vec![Dimension::Static(4)])),
            &[],
            &[test_vector_type(4)],
            &mut block,
            &context,
            location.as_ref(),
        )
        .unwrap_err();

        assert_eq!(error, LoweringError::Tracing(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),);
    }

    #[test]
    fn test_xla_operation_reshape_dimensions_use_shared_lowering() {
        let input_type = test_matrix_type(2, 3);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(
                LegacyReshapeOperation::new(
                    ReshapeParameters::new(Shape::new(vec![Dimension::Static(6)])).with_dimensions([1, 0]),
                ),
                Vec::new(),
                vec![input],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<6xf32> {
                    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
                    %1 = stablehlo.reshape %0 : (tensor<3x2xf32>) -> tensor<6xf32>
                    return %1 : tensor<6xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_plain_reshape_explicit_output_sharding_lowers_constraint() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let output_sharding =
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap();
        let mut builder = ryft_core::ProgramBuilder::<CpuArray, LegacyReshapeOperation>::new();
        let input = builder.add_input(test_vector_type(4));
        let output = builder
            .add_instruction(
                LegacyReshapeOperation::new(
                    ReshapeParameters::new(Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]))
                        .with_output_sharding(output_sharding),
                ),
                Vec::new(),
                vec![input],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<CpuArray>, Vec<CpuArray>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<2x2xf32> {
                    %0 = stablehlo.reshape %arg0 : (tensor<4xf32>) -> tensor<2x2xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{"x"}, {}]> : tensor<2x2xf32>
                    return %1 : tensor<2x2xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_plain_broadcast_explicit_sharding_transition_lowers_to_constraint() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = test_vector_type(4).with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap();
        let output_type = test_vector_type(4)
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let mut builder = ryft_core::ProgramBuilder::<CpuArray, LegacyBroadcastOperation>::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(LegacyBroadcastOperation::new(output_type, vec![0]), Vec::new(), vec![input])
            .unwrap()[0];
        let program = builder
            .build::<Vec<CpuArray>, Vec<CpuArray>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{"x"}]> : tensor<4xf32>
                    return %1 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_broadcast_explicit_sharding_transition_lowers_to_constraint() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = test_vector_type(4).with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap();
        let output_type = test_vector_type(4)
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let input_sharding = input_type.sharding().unwrap().clone();
        let output_sharding = output_type.sharding().unwrap().clone();
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type.clone());
        let output = builder
            .add_instruction(LegacyBroadcastOperation::new(output_type.clone(), vec![0]), Vec::new(), vec![input])
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &vec![input_type],
            &vec![output_type],
            "main",
            Some(&[input_sharding]),
            Some(&[output_sharding]),
        )
        .unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], replicated={"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{"x"}]> : tensor<4xf32>
                    return %1 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_broadcast_explicit_sharding_transition_executes_on_cpu() {
        use std::collections::HashMap;

        use ryft_core::sharding::{Device, DeviceMesh};
        use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
        use ryft_pjrt::{ClientOptions, CpuClientOptions, Program as PjrtProgram, load_cpu_plugin};

        use crate::tests::{values_from_bytes, values_to_bytes};
        use crate::{Array, FromPjrt};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let client_devices = client.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap(),
            client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let input_type =
            test_vector_type(4).with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap();
        let output_type = test_vector_type(4)
            .with_sharding(Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let input_sharding = input_type.sharding().unwrap().clone();
        let output_sharding = output_type.sharding().unwrap().clone();
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type.clone());
        let output = builder
            .add_instruction(LegacyBroadcastOperation::new(output_type.clone(), vec![0]), Vec::new(), vec![input])
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &vec![input_type.clone()],
            &vec![output_type],
            "main",
            Some(&[input_sharding]),
            Some(&[output_sharding]),
        )
        .unwrap();
        let executable = client
            .compile(
                &PjrtProgram::Mlir { bytecode: module.into_bytes() },
                &CompilationOptions {
                    argument_layouts: Vec::new(),
                    parameter_is_tupled_arguments: false,
                    executable_build_options: Some(ExecutableCompilationOptions {
                        device_ordinal: -1,
                        replica_count: 1,
                        partition_count: 2,
                        use_spmd_partitioning: true,
                        use_shardy_partitioner: true,
                        ..Default::default()
                    }),
                    compile_portable_executable: false,
                    profile_version: 0,
                    serialized_multi_slice_configuration: Vec::new(),
                    environment_option_overrides: HashMap::new(),
                    target_config: None,
                    allow_in_place_mlir_modification: false,
                    matrix_unit_operand_precision: Precision::Default as i32,
                },
            )
            .unwrap();
        let input = Array::from_host_buffer(
            &client,
            input_type,
            mesh,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let execution_devices = executable.addressable_devices().unwrap();
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let arguments = Array::into_execute_arguments(vec![input], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();

        assert_eq!(outputs.len(), 2);
        for (output, expected) in outputs.into_iter().zip([[1.0f32, 2.0], [3.0, 4.0]]) {
            assert_eq!(output.outputs.len(), 1);
            let bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(values_from_bytes::<f32>(bytes.as_slice()), expected);
        }
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_captures_as_hidden_arguments() {
        let array_type = test_vector_type(4);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(array_type.clone());
        let capture = builder.add_constant(XlaConstant::new(0, array_type.clone()));
        let output = builder.add_instruction(AddOperation, Vec::new(), vec![input, capture]).unwrap()[0];
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

    #[test]
    fn test_to_mlir_module_for_program_lowers_capture_output_as_a_hidden_argument() {
        let array_type = test_vector_type(4);
        let mut builder = XlaProgramBuilder::new();
        let output = builder.add_constant(XlaConstant::new(0, array_type.clone()));
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], Vec::new(), vec![Placeholder])
            .unwrap();
        let input_types: [ArrayType; 0] = [];
        let output_types = vec![array_type.clone()];
        let stablehlo = to_mlir_module_for_program(
            &program,
            std::slice::from_ref(&array_type),
            &input_types,
            &output_types,
            "main",
            None,
            None,
        )
        .unwrap();

        assert!(stablehlo.contains("func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32>"), "{stablehlo}");
        assert!(stablehlo.contains("return %arg0 : tensor<4xf32>"), "{stablehlo}");
        assert!(!stablehlo.contains("stablehlo.constant"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_forwards_capture_into_nested_regions() {
        let array_type = test_vector_type(4);
        let branch = || {
            let mut builder = XlaProgramBuilder::new();
            let output = builder.add_constant(XlaConstant::new(0, array_type.clone()));
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], Vec::new(), vec![Placeholder])
                .unwrap()
        };
        let mut builder = XlaProgramBuilder::new();
        let true_region = builder.import_program(branch());
        let false_region = builder.import_program(branch());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let output_types = vec![array_type.clone()];
        let stablehlo = to_mlir_module_for_program(
            &program,
            std::slice::from_ref(&array_type),
            &[ArrayType::scalar(DataType::Boolean)],
            &output_types,
            "main",
            None,
            None,
        )
        .unwrap();

        assert!(
            stablehlo.contains("func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<i1>) -> tensor<4xf32>",),
            "{stablehlo}",
        );
        assert_eq!(stablehlo.matches("stablehlo.return %arg0 : tensor<4xf32>").count(), 2, "{stablehlo}");
        assert!(!stablehlo.contains("stablehlo.constant"), "{stablehlo}");
    }

    #[test]
    fn test_plain_elementwise_lowering_normalizes_all_implicit_operands() {
        let program = xla_elementwise_normalization_program();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert_elementwise_operands_are_normalized(&stablehlo);
    }

    #[test]
    fn test_traced_elementwise_lowering_normalizes_all_implicit_operands() {
        let program = xla_elementwise_normalization_program();
        let input_types = program.input_types();
        let output_types = program.output_types();
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();
        assert_elementwise_operands_are_normalized(&stablehlo);
    }

    #[test]
    fn test_elementwise_lowering_normalizes_zero_sized_operands() {
        let mut builder = XlaProgramBuilder::new();
        let scalar = builder.add_input(ArrayType::scalar(DataType::F32));
        let empty = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0)])));
        let output = builder.add_instruction(DivOperation, Vec::new(), vec![scalar, empty]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert!(stablehlo.contains("stablehlo.convert"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.broadcast_in_dim"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.divide"), "{stablehlo}");
        assert!(stablehlo.contains("tensor<0xf64>"), "{stablehlo}");
    }

    /// Builds the flat callee `f(x) = x + x` over a vector type, returned behind an [`Rc`] so callers control
    /// whether `jit_call` sites share one program (pointer identity) or use structurally-identical distinct programs.
    fn xla_add_self_callee(input_type: ArrayType) -> std::rc::Rc<FlatXlaProgram> {
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let output = builder.add_instruction(AddOperation, Vec::new(), vec![input, input]).unwrap()[0];
        std::rc::Rc::new(builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap())
    }

    /// Builds a nullary callee that materializes one scalar leaf's fragment of a packed coordinate basis.
    fn xla_coordinate_basis_callee(coordinate_offset: usize) -> std::rc::Rc<FlatXlaProgram> {
        let mut builder = XlaProgramBuilder::new();
        let output = builder
            .add_instruction(
                XlaOperation::CoordinateBasis(CoordinateBasisOperation::new(
                    ArrayType::scalar(DataType::F32),
                    coordinate_offset,
                    2,
                )),
                Vec::new(),
                Vec::new(),
            )
            .unwrap()[0];
        std::rc::Rc::new(builder.build(vec![output], Vec::new(), vec![Placeholder]).unwrap())
    }

    /// Stages one `jit_call` to `callee` (interned as a shared callee root region) over `inputs` in `builder`.
    fn add_xla_jit_call(
        builder: &mut XlaProgramBuilder,
        callee: &std::rc::Rc<FlatXlaProgram>,
        inputs: Vec<AtomId>,
    ) -> AtomId {
        let callee_region = builder.intern_callee(callee, None).unwrap();
        builder
            .add_instruction(
                XlaOperation::JitCall(crate::experimental::ops::JitCallOperation::new()),
                vec![callee_region],
                inputs,
            )
            .unwrap()[0]
    }

    /// Lowers an outer program that calls `callees` (one `jit_call` each) and sums the results, returning the
    /// module text. Each callee is `f(x) = x + x`; the outer function is `g(x) = sum_i callee_i(x)`.
    fn lower_two_jit_call_module(callees: Vec<std::rc::Rc<FlatXlaProgram>>) -> String {
        let array_type = test_vector_type(4);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(array_type.clone());
        let mut accumulator: Option<AtomId> = None;
        for callee in callees {
            let call_output = add_xla_jit_call(&mut builder, &callee, vec![input]);
            accumulator = Some(match accumulator {
                None => call_output,
                Some(previous) => {
                    builder.add_instruction(AddOperation, Vec::new(), vec![previous, call_output]).unwrap()[0]
                }
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
    fn test_jit_call_dedup_distinguishes_coordinate_basis_attributes() {
        let first_callee = xla_coordinate_basis_callee(0);
        let second_callee = xla_coordinate_basis_callee(1);
        assert!(jit_call_program_key(&first_callee) != jit_call_program_key(&second_callee));

        let mut builder = XlaProgramBuilder::new();
        let first = add_xla_jit_call(&mut builder, &first_callee, Vec::new());
        let second = add_xla_jit_call(&mut builder, &second_callee, Vec::new());
        let output = builder.add_instruction(AddOperation, Vec::new(), vec![first, second]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], Vec::new(), vec![Placeholder])
            .unwrap();
        let input_types = Vec::<ArrayType>::new();
        let output_types = vec![test_vector_type(2)];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        // Each semantic basis operation occurs only once, so both callees inline rather than incorrectly sharing a
        // private function under a cache key that omits the coordinate offset.
        assert!(!module.contains("func.func private"), "{module}");
        assert!(!module.contains("call @jit_call"), "{module}");
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
    fn test_jit_call_dedup_does_not_traverse_shard_map_bodies() {
        use crate::experimental::operations::ShardMapOperation;
        use crate::experimental::shard_map::FlatTracedShardMap;

        // Phase 0 boundary pin for the first-class-program-regions plan: `count_jit_calls` intentionally skips
        // shard-map bodies, so a callee that occurs twice inside a `shard_map` body never gets a shared
        // `func.func private @jit_call_*` and both occurrences inline into the `sdy.manual_computation` region.
        let vector_type = test_vector_type(4);
        let callee = xla_add_self_callee(vector_type.clone());
        let body_program = {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(vector_type.clone());
            let first = add_xla_jit_call(&mut builder, &callee, vec![input]);
            let second = add_xla_jit_call(&mut builder, &callee, vec![input]);
            let output = builder.add_instruction(AddOperation, Vec::new(), vec![first, second]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mesh = test_manual_mesh("x", 1);
        let sharding = Sharding::replicated(mesh.clone(), 1);
        let body = FlatTracedShardMap::from_parts(
            ShardMap::from_shardings(mesh, vec![sharding.clone()], vec![sharding], vec!["x".to_string()], true),
            vec![vector_type.clone()],
            vec![vector_type.clone()],
            vec![vector_type.clone()],
            vec![vector_type.clone()],
            body_program,
        );
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(vector_type.clone());
        let (shard_map_operation, shard_map_body) = ShardMapOperation::from_body(body);
        let body_region = builder.import_program(shard_map_body);
        let output = builder
            .add_instruction(XlaOperation::ShardMap(Box::new(shard_map_operation)), vec![body_region], vec![input])
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let input_types = vec![vector_type.clone()];
        let output_types = vec![vector_type];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}], replicated={"x"}>] out_shardings=[<@mesh, [{}], replicated={"x"}>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
                      %1 = stablehlo.add %arg1, %arg1 : tensor<4xf32>
                      %2 = stablehlo.add %arg1, %arg1 : tensor<4xf32>
                      %3 = stablehlo.add %1, %2 : tensor<4xf32>
                      sdy.return %3 : tensor<4xf32>
                    } : (tensor<4xf32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_custom_jvp_lowering_inlines_only_the_primal_program() {
        use ryft_core::tracing_v2::custom_derivatives::CustomJvpOperation;

        // Phase 0 boundary pin for the first-class-program-regions plan: a retained `custom_jvp` call lowers only
        // its primal program; nothing from the user-supplied JVP program (marked here by the multiply on the
        // tangent side) reaches the emitted module.
        let vector_type = test_vector_type(4);
        let primal = {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(vector_type.clone());
            let output = builder.add_instruction(AddOperation, Vec::new(), vec![input, input]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let jvp = {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(vector_type.clone());
            let tangent = builder.add_input(vector_type.clone());
            let output = builder.add_instruction(AddOperation, Vec::new(), vec![input, input]).unwrap()[0];
            let output_tangent = builder.add_instruction(MulOperation, Vec::new(), vec![tangent, tangent]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![output, output_tangent],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder, Placeholder],
                )
                .unwrap()
        };
        let operation = CustomJvpOperation::new();
        let mut builder = XlaProgramBuilder::new();
        let primal_region = builder.import_region(primal.entry_region_ref());
        let jvp_region = builder.import_region(jvp.entry_region_ref());
        let input = builder.add_input(vector_type.clone());
        let output = builder
            .add_instruction(XlaOperation::CustomJvp(operation), vec![primal_region, jvp_region], vec![input])
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let input_types = vec![vector_type.clone()];
        let output_types = vec![vector_type];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

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
    fn test_custom_vjp_tangent_lowering_is_rejected() {
        use ryft_core::tracing_v2::custom_derivatives::CustomVjpTangentOperation;

        // Phase 0 boundary pin for the first-class-program-regions plan: the un-transposed `custom_vjp_tangent`
        // carrier is reverse-mode-only and must be transposed away before lowering, so lowering it is rejected.
        let vector_type = test_vector_type(4);
        let backward = {
            let mut builder = XlaProgramBuilder::new();
            let residual = builder.add_input(vector_type.clone());
            let cotangent = builder.add_input(vector_type.clone());
            let output = builder.add_instruction(MulOperation, Vec::new(), vec![residual, cotangent]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![output],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let operation = CustomVjpTangentOperation::new(1, false, vec![vector_type.clone()], vec![vector_type.clone()]);
        let mut builder = XlaProgramBuilder::new();
        let backward_region = builder.import_region(backward.entry_region_ref());
        let tangent = builder.add_input(vector_type.clone());
        let residual = builder.add_input(vector_type.clone());
        let output = builder
            .add_instruction(XlaOperation::CustomVjpTangent(operation), vec![backward_region], vec![tangent, residual])
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let input_types = vec![vector_type.clone(), vector_type.clone()];
        let output_types = vec![vector_type];

        assert!(matches!(
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None),
            Err(LoweringError::Tracing(ProgramError::UnsupportedOperation { message }))
                if message == "operation `custom_vjp_tangent` cannot be lowered to StableHLO",
        ));
    }

    /// Builds a flat callee whose body contains a `condition` instruction (`f(p, x) = if p { -x } else { x }`),
    /// making it ineligible for structural `jit_call` deduplication because its nested branch bodies do not render
    /// into the callee's canonical program text.
    fn xla_condition_callee() -> std::rc::Rc<FlatXlaProgram> {
        let vector_type = test_vector_type(4);
        let mut builder = XlaProgramBuilder::new();
        let true_region = builder.import_region(xla_neg_branch(vector_type.clone()).entry_region_ref());
        let false_region = builder.import_region(xla_identity_branch(vector_type.clone()).entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let input = builder.add_input(vector_type);
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
            )
            .unwrap()[0];
        std::rc::Rc::new(builder.build(vec![output], vec![Placeholder, Placeholder], vec![Placeholder]).unwrap())
    }

    #[test]
    fn test_structurally_identical_jit_calls_with_nested_bodies_merge() {
        // With nested computations attached as regions, a callee's `condition` bodies render contextually inside its
        // canonical rendering, so two structurally identical but separately constructed callees share one structural
        // identity and merge into a single emitted function — exactly like the body-free callees of
        // `test_structurally_identical_jit_calls_share_one_function`. (Before first-class regions, nested bodies were
        // payloads hidden from rendering and such callees deduplicated only by pointer identity.)
        let first = xla_condition_callee();
        let second = xla_condition_callee();
        let vector_type = test_vector_type(4);
        let mut builder = XlaProgramBuilder::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let input = builder.add_input(vector_type.clone());
        let mut accumulator: Option<AtomId> = None;
        for callee in [first.clone(), first, second.clone(), second] {
            let call_output = add_xla_jit_call(&mut builder, &callee, vec![predicate, input]);
            accumulator = Some(match accumulator {
                None => call_output,
                Some(previous) => {
                    builder.add_instruction(AddOperation, Vec::new(), vec![previous, call_output]).unwrap()[0]
                }
            });
        }
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![accumulator.unwrap()],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let input_types = vec![ArrayType::scalar(DataType::Boolean), vector_type.clone()];
        let output_types = vec![vector_type];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func private @jit_call_0(%arg0: tensor<i1>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = "stablehlo.if"(%arg0) ({
                      %1 = stablehlo.negate %arg1 : tensor<4xf32>
                      stablehlo.return %1 : tensor<4xf32>
                    }, {
                      stablehlo.return %arg1 : tensor<4xf32>
                    }) : (tensor<i1>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func @main(%arg0: tensor<i1>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = call @jit_call_0(%arg0, %arg1) : (tensor<i1>, tensor<4xf32>) -> tensor<4xf32>
                    %1 = call @jit_call_0(%arg0, %arg1) : (tensor<i1>, tensor<4xf32>) -> tensor<4xf32>
                    %2 = stablehlo.add %0, %1 : tensor<4xf32>
                    %3 = call @jit_call_0(%arg0, %arg1) : (tensor<i1>, tensor<4xf32>) -> tensor<4xf32>
                    %4 = stablehlo.add %2, %3 : tensor<4xf32>
                    %5 = call @jit_call_0(%arg0, %arg1) : (tensor<i1>, tensor<4xf32>) -> tensor<4xf32>
                    %6 = stablehlo.add %4, %5 : tensor<4xf32>
                    return %6 : tensor<4xf32>
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
    fn test_to_mlir_module_for_plain_program_lowers_coordinate_basis() {
        let mut builder = XlaProgramBuilder::new();
        let output = builder
            .add_instruction(
                XlaOperation::CoordinateBasis(CoordinateBasisOperation::new(test_matrix_type(2, 3), 1, 8)),
                Vec::new(),
                Vec::new(),
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], Vec::new(), vec![Placeholder])
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(stablehlo.matches("stablehlo.iota").count(), 3);
        assert_eq!(stablehlo.matches("stablehlo.multiply").count(), 1);
        assert_eq!(stablehlo.matches("stablehlo.add").count(), 2);
        assert_eq!(stablehlo.matches("stablehlo.compare").count(), 1);
        assert_eq!(stablehlo.matches("stablehlo.select").count(), 1);
        assert!(stablehlo.contains("tensor<8x2x3xf32>"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_complex_sine_and_cosine() {
        let complex_type = ArrayType::scalar(DataType::C64);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(complex_type);
        let sine = builder.add_instruction(XlaOperation::Sin(SinOperation), Vec::new(), vec![input]).unwrap()[0];
        let cosine = builder.add_instruction(XlaOperation::Cos(CosOperation), Vec::new(), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![sine, cosine],
                vec![Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(stablehlo.matches("stablehlo.real").count(), 2, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.imag").count(), 2, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.exponential_minus_one").count(), 4, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.select").count(), 4, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.complex").count(), 4, "{stablehlo}");
    }

    /// Builds a one-instruction program reducing an input of the given data type and dimensions, and returns the
    /// result of lowering it to a rendered StableHLO module.
    fn lowered_reduce_module(
        data_type: DataType,
        kind: ReductionKind,
        axes: Vec<usize>,
        dimensions: Vec<usize>,
    ) -> Result<String, LoweringError> {
        let input_type = ArrayType::new(data_type, Shape::new(dimensions.into_iter().map(Dimension::Static).collect()));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(XlaOperation::Reduce(ReduceOperation::new(axes, kind)), Vec::new(), vec![input])
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        to_mlir_module_for_plain_program(&program, "main")
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_bf16_reduce_max() {
        assert_eq!(
            lowered_reduce_module(DataType::BF16, ReductionKind::Max, vec![1], vec![2, 3]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x3xbf16>) -> tensor<2xbf16> {
                    %cst = stablehlo.constant dense<0xFF80> : tensor<bf16>
                    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [1] : (tensor<2x3xbf16>, tensor<bf16>) -> tensor<2xbf16>
                    return %0 : tensor<2xbf16>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_bf16_reduce_sum() {
        assert_eq!(
            lowered_reduce_module(DataType::BF16, ReductionKind::Sum, vec![0], vec![2, 3]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x3xbf16>) -> tensor<3xbf16> {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
                    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2x3xbf16>, tensor<bf16>) -> tensor<3xbf16>
                    return %0 : tensor<3xbf16>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_preserves_dynamic_reduce_axes() {
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5)).unwrap());
        let input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(3)]));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(
                XlaOperation::Reduce(ReduceOperation::new(vec![1], ReductionKind::Sum)),
                Vec::new(),
                vec![input],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert!(stablehlo.contains("tensor<?x3xf32, #stablehlo.bounds<4, ?>>"), "{stablehlo}");
        assert!(stablehlo.contains("-> tensor<?xf32, #stablehlo.bounds<4>>"), "{stablehlo}");
        assert!(stablehlo.contains("across dimensions = [1]"), "{stablehlo}");

        // Mean needs the removed-axis runtime extent as a divisor. That future dimension residual is an explicit
        // Phase 6/7 boundary rather than an operand added to the primal reduce instruction.
        let reduced = DimensionVariable::new("reduced", DimensionBounds::new(1, Some(5)).unwrap());
        let input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(reduced), Dimension::Static(3)]));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(
                XlaOperation::Reduce(ReduceOperation::new(vec![0], ReductionKind::Mean)),
                Vec::new(),
                vec![input],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main"),
            Err(LoweringError::UnsupportedOp { op: "'reduce' mean over dynamically sized axis 0".to_string() }),
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_bf16_reduce_mean() {
        assert_eq!(
            lowered_reduce_module(DataType::BF16, ReductionKind::Mean, vec![1], vec![2, 3]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x3xbf16>) -> tensor<2xbf16> {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
                    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<2x3xbf16>, tensor<bf16>) -> tensor<2xbf16>
                    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<bf16>
                    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<2xbf16>
                    %2 = stablehlo.divide %0, %1 : tensor<2xbf16>
                    return %2 : tensor<2xbf16>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_full_reduce_mean_without_broadcast() {
        assert_eq!(
            lowered_reduce_module(DataType::F32, ReductionKind::Mean, vec![0], vec![4]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<f32> {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
                    %cst_0 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
                    %1 = stablehlo.divide %0, %cst_0 : tensor<f32>
                    return %1 : tensor<f32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_f16_reduce_min() {
        assert_eq!(
            lowered_reduce_module(DataType::F16, ReductionKind::Min, vec![0], vec![4]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf16>) -> tensor<f16> {
                    %cst = stablehlo.constant dense<0x7C00> : tensor<f16>
                    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.minimum across dimensions = [0] : (tensor<4xf16>, tensor<f16>) -> tensor<f16>
                    return %0 : tensor<f16>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_f8e5m2_reduce_max_with_infinite_identity() {
        assert_eq!(
            lowered_reduce_module(DataType::F8E5M2, ReductionKind::Max, vec![0], vec![4]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf8E5M2>) -> tensor<f8E5M2> {
                    %cst = stablehlo.constant dense<0xFC> : tensor<f8E5M2>
                    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [0] : (tensor<4xf8E5M2>, tensor<f8E5M2>) -> tensor<f8E5M2>
                    return %0 : tensor<f8E5M2>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_f8e4m3fn_reduce_max_with_finite_identity() {
        assert_eq!(
            lowered_reduce_module(DataType::F8E4M3FN, ReductionKind::Max, vec![0], vec![4]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf8E4M3FN>) -> tensor<f8E4M3FN> {
                    %cst = stablehlo.constant dense<-4.480000e+02> : tensor<f8E4M3FN>
                    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [0] : (tensor<4xf8E4M3FN>, tensor<f8E4M3FN>) -> tensor<f8E4M3FN>
                    return %0 : tensor<f8E4M3FN>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_f8e4m3fn_reduce_sum() {
        assert_eq!(
            lowered_reduce_module(DataType::F8E4M3FN, ReductionKind::Sum, vec![0], vec![4]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf8E4M3FN>) -> tensor<f8E4M3FN> {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f8E4M3FN>
                    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4xf8E4M3FN>, tensor<f8E4M3FN>) -> tensor<f8E4M3FN>
                    return %0 : tensor<f8E4M3FN>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_f4e2m1fn_reduce_min_with_finite_identity() {
        assert_eq!(
            lowered_reduce_module(DataType::F4E2M1FN, ReductionKind::Min, vec![0], vec![4]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf4E2M1FN>) -> tensor<f4E2M1FN> {
                    %cst = stablehlo.constant dense<6.000000e+00> : tensor<f4E2M1FN>
                    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.minimum across dimensions = [0] : (tensor<4xf4E2M1FN>, tensor<f4E2M1FN>) -> tensor<f4E2M1FN>
                    return %0 : tensor<f4E2M1FN>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_i32_reduce_max_with_minimum_identity() {
        assert_eq!(
            lowered_reduce_module(DataType::I32, ReductionKind::Max, vec![0], vec![4]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xi32>) -> tensor<i32> {
                    %c = stablehlo.constant dense<-2147483648> : tensor<i32>
                    %0 = stablehlo.reduce(%arg0 init: %c) applies stablehlo.maximum across dimensions = [0] : (tensor<4xi32>, tensor<i32>) -> tensor<i32>
                    return %0 : tensor<i32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_i64_reduce_sum() {
        assert_eq!(
            lowered_reduce_module(DataType::I64, ReductionKind::Sum, vec![0], vec![4]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xi64>) -> tensor<i64> {
                    %c = stablehlo.constant dense<0> : tensor<i64>
                    %0 = stablehlo.reduce(%arg0 init: %c) applies stablehlo.add across dimensions = [0] : (tensor<4xi64>, tensor<i64>) -> tensor<i64>
                    return %0 : tensor<i64>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_i32_reduce_mean_with_integer_division() {
        assert_eq!(
            lowered_reduce_module(DataType::I32, ReductionKind::Mean, vec![1], vec![2, 3]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x3xi32>) -> tensor<2xi32> {
                    %c = stablehlo.constant dense<0> : tensor<i32>
                    %0 = stablehlo.reduce(%arg0 init: %c) applies stablehlo.add across dimensions = [1] : (tensor<2x3xi32>, tensor<i32>) -> tensor<2xi32>
                    %c_0 = stablehlo.constant dense<3> : tensor<i32>
                    %1 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2xi32>
                    %2 = stablehlo.divide %0, %1 : tensor<2xi32>
                    return %2 : tensor<2xi32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_u8_reduce_min_with_maximum_identity() {
        assert_eq!(
            lowered_reduce_module(DataType::U8, ReductionKind::Min, vec![0], vec![4]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xui8>) -> tensor<ui8> {
                    %c = stablehlo.constant dense<255> : tensor<ui8>
                    %0 = stablehlo.reduce(%arg0 init: %c) applies stablehlo.minimum across dimensions = [0] : (tensor<4xui8>, tensor<ui8>) -> tensor<ui8>
                    return %0 : tensor<ui8>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_u64_reduce_min_with_maximum_identity() {
        assert_eq!(
            lowered_reduce_module(DataType::U64, ReductionKind::Min, vec![0], vec![4]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xui64>) -> tensor<ui64> {
                    %c = stablehlo.constant dense<18446744073709551615> : tensor<ui64>
                    %0 = stablehlo.reduce(%arg0 init: %c) applies stablehlo.minimum across dimensions = [0] : (tensor<4xui64>, tensor<ui64>) -> tensor<ui64>
                    return %0 : tensor<ui64>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_rejects_f8e8m0fnu_reduce() {
        assert_eq!(
            lowered_reduce_module(DataType::F8E8M0FNU, ReductionKind::Sum, vec![0], vec![4]).unwrap_err(),
            LoweringError::UnsupportedDataType { data_type: DataType::F8E8M0FNU },
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_zero_sized_coordinate_basis() {
        let leaf_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(0), Dimension::Static(3)]),
        );
        let mut builder = XlaProgramBuilder::new();
        let output = builder
            .add_instruction(
                XlaOperation::CoordinateBasis(CoordinateBasisOperation::new(leaf_type, 0, 0)),
                Vec::new(),
                Vec::new(),
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], Vec::new(), vec![Placeholder])
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(stablehlo.matches("stablehlo.iota").count(), 0);
        assert_eq!(stablehlo.matches("stablehlo.multiply").count(), 0);
        assert_eq!(stablehlo.matches("stablehlo.add").count(), 0);
        assert_eq!(stablehlo.matches("stablehlo.compare").count(), 0);
        assert_eq!(stablehlo.matches("stablehlo.select").count(), 0);
        assert_eq!(stablehlo.matches("stablehlo.broadcast_in_dim").count(), 1);
        assert!(stablehlo.contains("tensor<0x2x0x3xf32>"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_condition_to_stablehlo_if() {
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let input_type = ArrayType::scalar(DataType::F32);
        let mut builder = XlaProgramBuilder::new();
        let true_region = builder.import_region(xla_neg_branch(input_type.clone()).entry_region_ref());
        let false_region = builder.import_region(xla_identity_branch(input_type.clone()).entry_region_ref());
        let predicate = builder.add_input(predicate_type);
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
            )
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
        use ryft_core::types::{Dimension, DimensionBounds, DimensionVariable, Shape};

        // Take whole rows of a [3, 2] matrix at the row indices in a [2, 1] index array: offset axis 1 carries the
        // row (slice sizes [1, 2]); axis 0 is collapsed (start-index driven). Output is [2, 2].
        let operand_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]));
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]));
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        let mut builder = XlaProgramBuilder::new();
        let operand = builder.add_input(operand_type);
        let indices = builder.add_input(indices_type);
        let output = builder
            .add_instruction(XlaOperation::Gather(operation), Vec::new(), vec![operand, indices])
            .unwrap()[0];
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

        // A dynamic query-batch axis remains the same array identity through gather; it is not reconstructed from a
        // shape operand or lowered through `stablehlo.dynamic_gather`.
        let query = DimensionVariable::new("query", DimensionBounds::new(1, Some(5)).unwrap());
        let indices_type =
            ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(query), Dimension::Static(1)]));
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        let mut builder = XlaProgramBuilder::new();
        let operand = builder
            .add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])));
        let indices = builder.add_input(indices_type);
        let output = builder
            .add_instruction(XlaOperation::Gather(operation), Vec::new(), vec![operand, indices])
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert!(stablehlo.contains("stablehlo.gather"), "{stablehlo}");
        assert!(!stablehlo.contains("stablehlo.dynamic_gather"), "{stablehlo}");
        assert!(stablehlo.contains("tensor<?x2xf32, #stablehlo.bounds<4, ?>>"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_clip_mode_gather_to_bare_op() {
        use ryft_core::operations::manipulation::{GatherDimensionNumbers, GatherOperation, GatherScatterMode};
        use ryft_core::types::{Dimension, Shape};

        // `Clip` is StableHLO `gather`'s default out-of-bounds behavior, so a `Clip`-mode gather lowers to the bare
        // `stablehlo.gather` (no extra clamp ops) just like the in-bounds default rather than erroring.
        let operand_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]));
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]));
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2])
            .with_mode(GatherScatterMode::Clip);
        let mut builder = XlaProgramBuilder::new();
        let operand = builder.add_input(operand_type);
        let indices = builder.add_input(indices_type);
        let output = builder
            .add_instruction(XlaOperation::Gather(operation), Vec::new(), vec![operand, indices])
            .unwrap()[0];
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
        use ryft_core::types::{Dimension, Shape};

        // Scatter-add row updates into a [3, 2] operand at the row indices in a [2, 1] index array. Output is [3, 2],
        // and the Add combiner lowers to a `stablehlo.add` inside the scatter region.
        let operand_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]));
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]));
        let updates_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]), ScatterReductionKind::Add);
        let mut builder = XlaProgramBuilder::new();
        let operand = builder.add_input(operand_type);
        let indices = builder.add_input(indices_type);
        let updates = builder.add_input(updates_type);
        let output = builder
            .add_instruction(XlaOperation::Scatter(operation), Vec::new(), vec![operand, indices, updates])
            .unwrap()[0];
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
        let mut builder = XlaProgramBuilder::new();
        let true_region = builder.import_region(xla_neg_branch(input_type.clone()).entry_region_ref());
        let false_region = builder.import_region(xla_identity_branch(input_type.clone()).entry_region_ref());
        let input = builder.add_input(input_type);
        let predicate = builder.add_instruction(OneOperation::new(predicate_type), Vec::new(), vec![]).unwrap()[0];
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
            )
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
        let mut builder = XlaProgramBuilder::new();
        let condition_region = builder.import_region(xla_identity_branch(state_type.clone()).entry_region_ref());
        let body_region = builder.import_region(xla_identity_branch(state_type.clone()).entry_region_ref());
        let state = builder.add_input(state_type);
        let output = builder
            .add_instruction(
                XlaOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                vec![state],
            )
            .unwrap()[0];
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
        let while_operation = WhileOperation::new().with_iteration_bound(3).unwrap();
        let mut builder = XlaProgramBuilder::new();
        let condition_region = builder.import_region(xla_identity_branch(state_type.clone()).entry_region_ref());
        let body_region = builder.import_region(xla_identity_branch(state_type.clone()).entry_region_ref());
        let state = builder.add_input(state_type);
        let output = builder
            .add_instruction(XlaOperation::While(while_operation), vec![condition_region, body_region], vec![state])
            .unwrap()[0];
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
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let condition = {
            let mut builder = XlaProgramBuilder::new();
            let state = builder.add_input(state_type.clone());
            let zero = builder.add_instruction(ZeroLikeOperation, Vec::new(), vec![state]).unwrap()[0];
            let predicate = builder
                .add_instruction(CompareOperation::new(ComparisonDirection::GreaterThan), Vec::new(), vec![state, zero])
                .unwrap()[0];
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![predicate], vec![Placeholder], vec![Placeholder])
        }
        .unwrap();
        let body = {
            let mut builder = XlaProgramBuilder::new();
            let state = builder.add_input(state_type.clone());
            let one = builder.add_instruction(OneLikeOperation, Vec::new(), vec![state]).unwrap()[0];
            let next = builder.add_instruction(SubOperation, Vec::new(), vec![state, one]).unwrap()[0];
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![next], vec![Placeholder], vec![Placeholder])
        }
        .unwrap();
        let mut builder = XlaProgramBuilder::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(body.entry_region_ref());
        let state = builder.add_input(state_type);
        let output = builder
            .add_instruction(
                XlaOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                vec![state],
            )
            .unwrap()[0];
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
        let product = body_builder.add_instruction(MulOperation, Vec::new(), vec![carry, x]).unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![product, product],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let scan = CoreScanOperation::<XlaArrayConstant>::new(1, 3);

        let mut builder = XlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let init = builder.add_input(scalar_f32);
        let stacked_inputs = builder.add_input(test_vector_type(3));
        let outputs = builder
            .add_instruction(XlaOperation::Scan(scan), vec![body_region], vec![init, stacked_inputs])
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
        let product = body_builder.add_instruction(MulOperation, Vec::new(), vec![carry, x]).unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![product, product],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let scan = CoreScanOperation::<XlaArrayConstant>::new(1, 3).with_unroll(3).unwrap();

        let mut builder = XlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let init = builder.add_input(scalar_f32);
        let stacked_inputs = builder.add_input(test_vector_type(3));
        let outputs = builder
            .add_instruction(XlaOperation::Scan(scan), vec![body_region], vec![init, stacked_inputs])
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
        let product = body_builder.add_instruction(MulOperation, Vec::new(), vec![carry, x]).unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![product, product],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let scan = CoreScanOperation::<XlaArrayConstant>::new(1, 4).with_unroll(2).unwrap();

        let mut builder = XlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let init = builder.add_input(scalar_f32);
        let stacked_inputs = builder.add_input(test_vector_type(4));
        let outputs = builder
            .add_instruction(XlaOperation::Scan(scan), vec![body_region], vec![init, stacked_inputs])
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
        let array_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(array_type.clone());
        let doubled = builder.add_instruction(AddOperation, Vec::new(), vec![input, input]).unwrap()[0];
        let first = builder.add_instruction(PrintOperation::new("first"), Vec::new(), vec![input]).unwrap()[0];
        let second = builder.add_instruction(PrintOperation::new("second"), Vec::new(), vec![doubled]).unwrap()[0];
        let output = builder.add_instruction(AddOperation, Vec::new(), vec![first, second]).unwrap()[0];
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
    fn test_to_mlir_module_for_program_lowers_custom_call() {
        use ryft_core::operations::custom_call::CustomCallOperation;

        // A side-effecting custom call with one attribute of each supported type lowers to a typed-FFI
        // `stablehlo.custom_call` whose `backend_config` dictionary carries the typed encodings (string, `i1`
        // Boolean, signless `i64`, and `f64`); a pure attribute-free call lowers with an empty dictionary and
        // no `has_side_effect` marker.
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let mut builder = XlaProgramBuilder::new();
        let left = builder.add_input(array_type.clone());
        let right = builder.add_input(array_type.clone());
        let effectful = CustomCallOperation::new("ryft.test.scaled_add", vec![array_type.clone()])
            .with_attribute("scale", 2.0)
            .with_attribute("count", 4i64)
            .with_attribute("flag", true)
            .with_attribute("label", "x")
            .with_side_effect();
        let scaled = builder.add_instruction(effectful, Vec::new(), vec![left, right]).unwrap()[0];
        let pure = CustomCallOperation::new("ryft.test.add_one", vec![array_type.clone()]);
        let output = builder.add_instruction(pure, Vec::new(), vec![scaled]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let input_types = vec![array_type.clone(), array_type.clone()];
        let output_types = vec![array_type];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
                    %0 = stablehlo.custom_call @ryft.test.scaled_add(%arg0, %arg1) {api_version = 4 : i32, backend_config = {count = 4 : i64, flag = true, label = "x", scale = 2.000000e+00 : f64}, has_side_effect = true} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
                    %1 = stablehlo.custom_call @ryft.test.add_one(%0) {api_version = 4 : i32, backend_config = {}} : (tensor<2xf32>) -> tensor<2xf32>
                    return %1 : tensor<2xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_sort_with_synthesized_comparator() {
        use ryft_core::operations::sort::{SortDirection, SortOperation};

        // A two-operand descending sort lowers to a stable `stablehlo.sort` whose synthesized comparator compares
        // the key pair with `GT` and `TOTALORDER` semantics, and whose index passenger rides along unexamined.
        let key_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let index_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(4)]));
        let mut builder = XlaProgramBuilder::new();
        let keys = builder.add_input(key_type.clone());
        let indices = builder.add_input(index_type.clone());
        let outputs = builder
            .add_instruction(SortOperation::new(0, SortDirection::Descending), Vec::new(), vec![keys, indices])
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let input_types = vec![key_type.clone(), index_type.clone()];
        let output_types = vec![key_type, index_type];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xi32>) -> (tensor<4xf32>, tensor<4xi32>) {
                    %0:2 = "stablehlo.sort"(%arg0, %arg1) <{dimension = 0 : i64, is_stable = true}> ({
                    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<i32>, %arg5: tensor<i32>):
                      %1 = stablehlo.compare GT, %arg2, %arg3, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      stablehlo.return %1 : tensor<i1>
                    }) : (tensor<4xf32>, tensor<4xi32>) -> (tensor<4xf32>, tensor<4xi32>)
                    return %0#0, %0#1 : tensor<4xf32>, tensor<4xi32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_multi_key_sort_with_lexicographic_comparator() {
        use ryft_core::operations::sort::{SortDirection, SortOperation};

        // A two-key ascending sort lowers to a stable `stablehlo.sort` whose synthesized comparator chains the key
        // comparisons lexicographically as `cmp_0 OR (eq_0 AND cmp_1)`, with each key's comparison type derived from
        // that key's data type (`TOTALORDER` for the `f32` primary key, including its equality comparison so NaN
        // ties fall through deterministically, and `SIGNED` for the `i32` secondary key), while the passenger rides
        // along unexamined.
        let primary_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let secondary_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(4)]));
        let passenger_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let mut builder = XlaProgramBuilder::new();
        let primary = builder.add_input(primary_type.clone());
        let secondary = builder.add_input(secondary_type.clone());
        let passenger = builder.add_input(passenger_type.clone());
        let operation = SortOperation::new(0, SortDirection::Ascending).with_key_count(2).unwrap();
        let outputs = builder
            .add_instruction(operation, Vec::new(), vec![primary, secondary, passenger])
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 3], vec![Placeholder; 3])
            .unwrap();
        let input_types = vec![primary_type.clone(), secondary_type.clone(), passenger_type.clone()];
        let output_types = vec![primary_type, secondary_type, passenger_type];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xi32>, %arg2: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xi32>, tensor<4xf32>) {
                    %0:3 = "stablehlo.sort"(%arg0, %arg1, %arg2) <{dimension = 0 : i64, is_stable = true}> ({
                    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<i32>, %arg6: tensor<i32>, %arg7: tensor<f32>, %arg8: tensor<f32>):
                      %1 = stablehlo.compare LT, %arg5, %arg6, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
                      %2 = stablehlo.compare LT, %arg3, %arg4, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %3 = stablehlo.compare EQ, %arg3, %arg4, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %4 = stablehlo.and %3, %1 : tensor<i1>
                      %5 = stablehlo.or %2, %4 : tensor<i1>
                      stablehlo.return %5 : tensor<i1>
                    }) : (tensor<4xf32>, tensor<4xi32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xi32>, tensor<4xf32>)
                    return %0#0, %0#1, %0#2 : tensor<4xf32>, tensor<4xi32>, tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_accumulation_typed_dot() {
        use ryft_core::operations::math::{DotDimensionNumbers, DotOperation};

        // An accumulation-typed dot lowers to a `stablehlo.dot_general` whose result type is the accumulation type
        // (XLA's `preferred_element_type` contract), with the operands kept at their narrow element type.
        let operand_type =
            ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let mut builder = XlaProgramBuilder::new();
        let lhs = builder.add_input(operand_type.clone());
        let rhs = builder.add_input(operand_type.clone());
        let operation = DotOperation::new(DotDimensionNumbers::matmul()).with_accumulation_type(DataType::F32);
        let output = builder.add_instruction(operation, Vec::new(), vec![lhs, rhs]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let input_types = vec![operand_type.clone(), operand_type];
        let output_types = vec![output_type];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x2xf8E4M3FN>, %arg1: tensor<2x2xf8E4M3FN>) -> tensor<2x2xf32> {
                    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf8E4M3FN>, tensor<2x2xf8E4M3FN>) -> tensor<2x2xf32>
                    return %0 : tensor<2x2xf32>
                  }
                }
            "#},
        );
    }

    /// Builds the NVFP4 scaled-dot program shared by the platform-gated lowering fixtures below:
    /// `f4e2m1fn [2, 16]` operands with `f8e4m3fn [2, 1]` scales over blocks of 16.
    fn scaled_dot_fixture_program() -> (XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>, Vec<ArrayType>, Vec<ArrayType>)
    {
        use ryft_core::operations::math::ScaledDotOperation;

        let element_type =
            ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(16)]));
        let scale_type =
            ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let mut builder = XlaProgramBuilder::new();
        let inputs = vec![
            builder.add_input(element_type.clone()),
            builder.add_input(scale_type.clone()),
            builder.add_input(element_type.clone()),
            builder.add_input(scale_type.clone()),
        ];
        let output =
            builder.add_instruction(ScaledDotOperation::new(16, DataType::F32), Vec::new(), inputs).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 4], vec![Placeholder])
            .unwrap();
        let input_types = vec![element_type.clone(), scale_type.clone(), element_type, scale_type];
        (program, input_types, vec![output_type])
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_scaled_dot_composition_without_target_platform() {
        // Without target information the scaled dot lowers to the portable dequantization composition.
        let (program, input_types, output_types) = scaled_dot_fixture_program();
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x16xf4E2M1FN>, %arg1: tensor<2x1xf8E4M3FN>, %arg2: tensor<2x16xf4E2M1FN>, %arg3: tensor<2x1xf8E4M3FN>) -> tensor<2x2xf32> {
                    %0 = stablehlo.convert %arg0 : (tensor<2x16xf4E2M1FN>) -> tensor<2x16xf32>
                    %1 = stablehlo.convert %arg1 : (tensor<2x1xf8E4M3FN>) -> tensor<2x1xf32>
                    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x16xf32>
                    %3 = stablehlo.reshape %2 : (tensor<2x1x16xf32>) -> tensor<2x16xf32>
                    %4 = stablehlo.multiply %0, %3 : tensor<2x16xf32>
                    %5 = stablehlo.convert %arg2 : (tensor<2x16xf4E2M1FN>) -> tensor<2x16xf32>
                    %6 = stablehlo.convert %arg3 : (tensor<2x1xf8E4M3FN>) -> tensor<2x1xf32>
                    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x16xf32>
                    %8 = stablehlo.reshape %7 : (tensor<2x1x16xf32>) -> tensor<2x16xf32>
                    %9 = stablehlo.multiply %5, %8 : tensor<2x16xf32>
                    %10 = stablehlo.dot_general %4, %9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x16xf32>, tensor<2x16xf32>) -> tensor<2x2xf32>
                    return %10 : tensor<2x2xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_mlir_module_for_program_emits_block_scaled_dot_on_cuda() {
        // With a CUDA target and qualifying NVFP4 formats, the scaled dot lowers to the `__op$block_scaled_dot`
        // custom call (operand order lhs, rhs, lhs scales, rhs scales), which XLA's GPU block-scaling rewriter
        // lowers to cuDNN's native block-scaled dot or expanded reference HLO.
        let (program, input_types, output_types) = scaled_dot_fixture_program();
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x16xf4E2M1FN>, %arg1: tensor<2x1xf8E4M3FN>, %arg2: tensor<2x16xf4E2M1FN>, %arg3: tensor<2x1xf8E4M3FN>) -> tensor<2x2xf32> {
                    %0 = stablehlo.custom_call @__op$block_scaled_dot(%arg0, %arg2, %arg1, %arg3) {api_version = 4 : i32, backend_config = {}} : (tensor<2x16xf4E2M1FN>, tensor<2x16xf4E2M1FN>, tensor<2x1xf8E4M3FN>, tensor<2x1xf8E4M3FN>) -> tensor<2x2xf32>
                    return %0 : tensor<2x2xf32>
                  }
                }
            "#},
        );
    }

    /// Builds a rank-3 NVFP4 scaled-dot program with a scalar global scale: `f4e2m1fn [2, 2, 16]` operands with
    /// `f8e4m3fn [2, 2, 1]` scales over blocks of 16 and an `f32` global scale.
    fn scaled_dot_rank_3_global_scale_fixture_program()
    -> (XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>, Vec<ArrayType>, Vec<ArrayType>) {
        use ryft_core::operations::math::ScaledDotOperation;

        let element_type = ArrayType::new(
            DataType::F4E2M1FN,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(16)]),
        );
        let scale_type = ArrayType::new(
            DataType::F8E4M3FN,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(1)]),
        );
        let global_scale_type = ArrayType::scalar(DataType::F32);
        let output_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(2)]),
        );
        let mut builder = XlaProgramBuilder::new();
        let inputs = vec![
            builder.add_input(element_type.clone()),
            builder.add_input(scale_type.clone()),
            builder.add_input(element_type.clone()),
            builder.add_input(scale_type.clone()),
            builder.add_input(global_scale_type.clone()),
        ];
        let output =
            builder.add_instruction(ScaledDotOperation::new(16, DataType::F32), Vec::new(), inputs).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 5], vec![Placeholder])
            .unwrap();
        let input_types = vec![element_type.clone(), scale_type.clone(), element_type, scale_type, global_scale_type];
        (program, input_types, vec![output_type])
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_rank_3_scaled_dot_with_global_scale_composition() {
        // Without target information the rank-3 form with a global scale lowers to the portable dequantization
        // composition: batched contraction plus a broadcast global-scale multiply.
        let (program, input_types, output_types) = scaled_dot_rank_3_global_scale_fixture_program();
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x2x16xf4E2M1FN>, %arg1: tensor<2x2x1xf8E4M3FN>, %arg2: tensor<2x2x16xf4E2M1FN>, %arg3: tensor<2x2x1xf8E4M3FN>, %arg4: tensor<f32>) -> tensor<2x2x2xf32> {
                    %0 = stablehlo.convert %arg0 : (tensor<2x2x16xf4E2M1FN>) -> tensor<2x2x16xf32>
                    %1 = stablehlo.convert %arg1 : (tensor<2x2x1xf8E4M3FN>) -> tensor<2x2x1xf32>
                    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2] : (tensor<2x2x1xf32>) -> tensor<2x2x1x16xf32>
                    %3 = stablehlo.reshape %2 : (tensor<2x2x1x16xf32>) -> tensor<2x2x16xf32>
                    %4 = stablehlo.multiply %0, %3 : tensor<2x2x16xf32>
                    %5 = stablehlo.convert %arg2 : (tensor<2x2x16xf4E2M1FN>) -> tensor<2x2x16xf32>
                    %6 = stablehlo.convert %arg3 : (tensor<2x2x1xf8E4M3FN>) -> tensor<2x2x1xf32>
                    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2] : (tensor<2x2x1xf32>) -> tensor<2x2x1x16xf32>
                    %8 = stablehlo.reshape %7 : (tensor<2x2x1x16xf32>) -> tensor<2x2x16xf32>
                    %9 = stablehlo.multiply %5, %8 : tensor<2x2x16xf32>
                    %10 = stablehlo.dot_general %4, %9, batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x2x16xf32>, tensor<2x2x16xf32>) -> tensor<2x2x2xf32>
                    %11 = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<2x2x2xf32>
                    %12 = stablehlo.multiply %10, %11 : tensor<2x2x2xf32>
                    return %12 : tensor<2x2x2xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_mlir_module_for_program_emits_rank_3_block_scaled_dot_with_global_scale_on_cuda() {
        // With a CUDA target the rank-3 form with a global scale lowers to the `__op$block_scaled_dot` custom call
        // with the scalar global scale appended as its fifth operand.
        let (program, input_types, output_types) = scaled_dot_rank_3_global_scale_fixture_program();
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x2x16xf4E2M1FN>, %arg1: tensor<2x2x1xf8E4M3FN>, %arg2: tensor<2x2x16xf4E2M1FN>, %arg3: tensor<2x2x1xf8E4M3FN>, %arg4: tensor<f32>) -> tensor<2x2x2xf32> {
                    %0 = stablehlo.custom_call @__op$block_scaled_dot(%arg0, %arg2, %arg1, %arg3, %arg4) {api_version = 4 : i32, backend_config = {}} : (tensor<2x2x16xf4E2M1FN>, tensor<2x2x16xf4E2M1FN>, tensor<2x2x1xf8E4M3FN>, tensor<2x2x1xf8E4M3FN>, tensor<f32>) -> tensor<2x2x2xf32>
                    return %0 : tensor<2x2x2xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_mlir_module_for_program_emits_block_scaled_dot_in_shared_jit_call_callee() {
        // The module's target platform reaches deduplicated `jit_call` callee functions, so a qualifying NVFP4
        // scaled dot inside a shared callee still lowers to the `__op$block_scaled_dot` custom call on CUDA.
        use ryft_core::operations::math::ScaledDotOperation;

        let element_type =
            ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(16)]));
        let scale_type =
            ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let mut callee_builder = XlaProgramBuilder::new();
        let callee_inputs = vec![
            callee_builder.add_input(element_type.clone()),
            callee_builder.add_input(scale_type.clone()),
            callee_builder.add_input(element_type.clone()),
            callee_builder.add_input(scale_type.clone()),
        ];
        let callee_output = callee_builder
            .add_instruction(ScaledDotOperation::new(16, DataType::F32), Vec::new(), callee_inputs)
            .unwrap()[0];
        let callee: std::rc::Rc<FlatXlaProgram> = std::rc::Rc::new(
            callee_builder.build(vec![callee_output], vec![Placeholder; 4], vec![Placeholder]).unwrap(),
        );

        let mut builder = XlaProgramBuilder::new();
        let inputs = vec![
            builder.add_input(element_type.clone()),
            builder.add_input(scale_type.clone()),
            builder.add_input(element_type.clone()),
            builder.add_input(scale_type.clone()),
        ];
        let first = add_xla_jit_call(&mut builder, &callee, inputs.clone());
        let second = add_xla_jit_call(&mut builder, &callee, inputs);
        let output = builder.add_instruction(AddOperation, Vec::new(), vec![first, second]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 4], vec![Placeholder])
            .unwrap();
        let input_types = vec![element_type.clone(), scale_type.clone(), element_type, scale_type];
        let output_types = vec![output_type];
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func private @jit_call_0(%arg0: tensor<2x16xf4E2M1FN>, %arg1: tensor<2x1xf8E4M3FN>, %arg2: tensor<2x16xf4E2M1FN>, %arg3: tensor<2x1xf8E4M3FN>) -> tensor<2x2xf32> {
                    %0 = stablehlo.custom_call @__op$block_scaled_dot(%arg0, %arg2, %arg1, %arg3) {api_version = 4 : i32, backend_config = {}} : (tensor<2x16xf4E2M1FN>, tensor<2x16xf4E2M1FN>, tensor<2x1xf8E4M3FN>, tensor<2x1xf8E4M3FN>) -> tensor<2x2xf32>
                    return %0 : tensor<2x2xf32>
                  }
                  func.func @main(%arg0: tensor<2x16xf4E2M1FN>, %arg1: tensor<2x1xf8E4M3FN>, %arg2: tensor<2x16xf4E2M1FN>, %arg3: tensor<2x1xf8E4M3FN>) -> tensor<2x2xf32> {
                    %0 = call @jit_call_0(%arg0, %arg1, %arg2, %arg3) : (tensor<2x16xf4E2M1FN>, tensor<2x1xf8E4M3FN>, tensor<2x16xf4E2M1FN>, tensor<2x1xf8E4M3FN>) -> tensor<2x2xf32>
                    %1 = call @jit_call_0(%arg0, %arg1, %arg2, %arg3) : (tensor<2x16xf4E2M1FN>, tensor<2x1xf8E4M3FN>, tensor<2x16xf4E2M1FN>, tensor<2x1xf8E4M3FN>) -> tensor<2x2xf32>
                    %2 = stablehlo.add %0, %1 : tensor<2x2xf32>
                    return %2 : tensor<2x2xf32>
                  }
                }
            "#},
        );
    }

    /// Builds the dot-product attention program shared by the platform-gated lowering fixtures below: `BTNH`
    /// operands `query`/`key`/`value` `[1, 4, 2, head_dimension]` at the provided data type with scale `0.125`.
    fn dot_product_attention_fixture_program(
        data_type: DataType,
        head_dimension: usize,
        mask: AttentionMask,
    ) -> (XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>, Vec<ArrayType>, Vec<ArrayType>) {
        let operand_type = ArrayType::new(
            data_type,
            Shape::new(vec![
                Dimension::Static(1),
                Dimension::Static(4),
                Dimension::Static(2),
                Dimension::Static(head_dimension),
            ]),
        );
        let mut builder = XlaProgramBuilder::new();
        let inputs = vec![
            builder.add_input(operand_type.clone()),
            builder.add_input(operand_type.clone()),
            builder.add_input(operand_type.clone()),
        ];
        let output =
            builder.add_instruction(DotProductAttentionOperation::new(0.125, mask), Vec::new(), inputs).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let input_types = vec![operand_type.clone(), operand_type.clone(), operand_type.clone()];
        (program, input_types, vec![operand_type])
    }

    #[test]
    fn test_lower_mlir_module_for_program_emits_fmha_softmax_on_cuda() {
        // With a CUDA target, `bf16` operands, and a head dimension that is a multiple of 8, the dot-product
        // attention lowers to the `__cudnn$fmhaSoftmax` custom call with the empirically validated legacy custom-call
        // contract — proto-JSON `backend_config` string, `api_version = 2`, explicit operand/result layouts, and an
        // unused zero-sized workspace result — followed by the transpose back to the logical `BTNH` layout.
        let (program, input_types, output_types) =
            dot_product_attention_fixture_program(DataType::BF16, 8, AttentionMask::Causal);
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<1x4x2x8xbf16>, %arg1: tensor<1x4x2x8xbf16>, %arg2: tensor<1x4x2x8xbf16>) -> tensor<1x4x2x8xbf16> {
                    %0:2 = stablehlo.custom_call @__cudnn$fmhaSoftmax(%arg0, %arg1, %arg2) {api_version = 2 : i32, backend_config = "{\22operation_queue_id\22:\220\22,\22cudnn_fmha_backend_config\22:{\22algorithm\22:{\22algo_id\22:\220\22,\22math_type\22:\22TENSOR_OP_MATH\22,\22tuning_knobs\22:{\2217\22:\221\22,\2224\22:\220\22},\22is_cudnn_frontend\22:true,\22workspace_size\22:\220\22},\22fmha_scale\22:0.125,\22intermediate_tensor_shape\22:{\22element_type\22:\22BF16\22,\22dimensions\22:[\221\22,\222\22,\224\22,\224\22],\22tuple_shapes\22:[],\22layout\22:{\22dim_level_types\22:[],\22dim_unique\22:[],\22dim_ordered\22:[],\22minor_to_major\22:[\223\22,\222\22,\221\22,\220\22],\22tiles\22:[],\22element_size_in_bits\22:\220\22,\22memory_space\22:\220\22,\22index_primitive_type\22:\22PRIMITIVE_TYPE_INVALID\22,\22pointer_primitive_type\22:\22PRIMITIVE_TYPE_INVALID\22,\22dynamic_shape_metadata_prefix_bytes\22:\220\22},\22is_dynamic_dimension\22:[false,false,false,false]},\22is_flash_attention\22:true,\22mask_type\22:\22CAUSAL\22,\22bmm1_dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\223\22],\22rhs_contracting_dimensions\22:[\223\22],\22lhs_batch_dimensions\22:[\220\22,\222\22],\22rhs_batch_dimensions\22:[\220\22,\222\22]},\22bmm2_dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\223\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[\220\22,\221\22],\22rhs_batch_dimensions\22:[\220\22,\222\22]},\22dropout_rate\22:0.0,\22seed\22:42,\22sliding_window_length\22:0,\22max_seg_per_batch\22:1,\22is_paged_attention\22:false}}", operand_layouts = [dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>], result_layouts = [dense<[3, 1, 2, 0]> : tensor<4xindex>, dense<0> : tensor<1xindex>]} : (tensor<1x4x2x8xbf16>, tensor<1x4x2x8xbf16>, tensor<1x4x2x8xbf16>) -> (tensor<1x2x4x8xbf16>, tensor<0xui8>)
                    %1 = stablehlo.transpose %0#0, dims = [0, 2, 1, 3] : (tensor<1x2x4x8xbf16>) -> tensor<1x4x2x8xbf16>
                    return %1 : tensor<1x4x2x8xbf16>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_mlir_module_for_program_emits_fmha_softmax_without_mask_on_cuda() {
        // The unmasked `f16` variant emits `NO_MASK` and `F16` in the backend configuration.
        let (program, input_types, output_types) =
            dot_product_attention_fixture_program(DataType::F16, 8, AttentionMask::None);
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<1x4x2x8xf16>, %arg1: tensor<1x4x2x8xf16>, %arg2: tensor<1x4x2x8xf16>) -> tensor<1x4x2x8xf16> {
                    %0:2 = stablehlo.custom_call @__cudnn$fmhaSoftmax(%arg0, %arg1, %arg2) {api_version = 2 : i32, backend_config = "{\22operation_queue_id\22:\220\22,\22cudnn_fmha_backend_config\22:{\22algorithm\22:{\22algo_id\22:\220\22,\22math_type\22:\22TENSOR_OP_MATH\22,\22tuning_knobs\22:{\2217\22:\221\22,\2224\22:\220\22},\22is_cudnn_frontend\22:true,\22workspace_size\22:\220\22},\22fmha_scale\22:0.125,\22intermediate_tensor_shape\22:{\22element_type\22:\22F16\22,\22dimensions\22:[\221\22,\222\22,\224\22,\224\22],\22tuple_shapes\22:[],\22layout\22:{\22dim_level_types\22:[],\22dim_unique\22:[],\22dim_ordered\22:[],\22minor_to_major\22:[\223\22,\222\22,\221\22,\220\22],\22tiles\22:[],\22element_size_in_bits\22:\220\22,\22memory_space\22:\220\22,\22index_primitive_type\22:\22PRIMITIVE_TYPE_INVALID\22,\22pointer_primitive_type\22:\22PRIMITIVE_TYPE_INVALID\22,\22dynamic_shape_metadata_prefix_bytes\22:\220\22},\22is_dynamic_dimension\22:[false,false,false,false]},\22is_flash_attention\22:true,\22mask_type\22:\22NO_MASK\22,\22bmm1_dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\223\22],\22rhs_contracting_dimensions\22:[\223\22],\22lhs_batch_dimensions\22:[\220\22,\222\22],\22rhs_batch_dimensions\22:[\220\22,\222\22]},\22bmm2_dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\223\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[\220\22,\221\22],\22rhs_batch_dimensions\22:[\220\22,\222\22]},\22dropout_rate\22:0.0,\22seed\22:42,\22sliding_window_length\22:0,\22max_seg_per_batch\22:1,\22is_paged_attention\22:false}}", operand_layouts = [dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>], result_layouts = [dense<[3, 1, 2, 0]> : tensor<4xindex>, dense<0> : tensor<1xindex>]} : (tensor<1x4x2x8xf16>, tensor<1x4x2x8xf16>, tensor<1x4x2x8xf16>) -> (tensor<1x2x4x8xf16>, tensor<0xui8>)
                    %1 = stablehlo.transpose %0#0, dims = [0, 2, 1, 3] : (tensor<1x2x4x8xf16>) -> tensor<1x4x2x8xf16>
                    return %1 : tensor<1x4x2x8xf16>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_dot_product_attention_composition_without_target_platform() {
        // Without target information the dot-product attention lowers to the portable StableHLO composition: scores,
        // scale, causal mask (two iotas, compare, select), max-stabilized softmax, context contraction, and the
        // transpose back to `BTNH`. The `f32` operands keep the softmax at `f32` with no conversions.
        let (program, input_types, output_types) =
            dot_product_attention_fixture_program(DataType::F32, 8, AttentionMask::Causal);
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<1x4x2x8xf32>, %arg1: tensor<1x4x2x8xf32>, %arg2: tensor<1x4x2x8xf32>) -> tensor<1x4x2x8xf32> {
                    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x4x2x8xf32>, tensor<1x4x2x8xf32>) -> tensor<1x2x4x4xf32>
                    %cst = stablehlo.constant dense<1.250000e-01> : tensor<f32>
                    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x2x4x4xf32>
                    %2 = stablehlo.multiply %0, %1 : tensor<1x2x4x4xf32>
                    %3 = stablehlo.iota dim = 3 : tensor<1x2x4x4xi32>
                    %4 = stablehlo.iota dim = 2 : tensor<1x2x4x4xi32>
                    %5 = stablehlo.compare LE, %3, %4, SIGNED : (tensor<1x2x4x4xi32>, tensor<1x2x4x4xi32>) -> tensor<1x2x4x4xi1>
                    %cst_0 = stablehlo.constant dense<-1.000000e+30> : tensor<f32>
                    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x2x4x4xf32>
                    %7 = stablehlo.select %5, %2, %6 : tensor<1x2x4x4xi1>, tensor<1x2x4x4xf32>
                    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
                    %8 = stablehlo.reduce(%7 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x2x4x4xf32>, tensor<f32>) -> tensor<1x2x4xf32>
                    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2] : (tensor<1x2x4xf32>) -> tensor<1x2x4x4xf32>
                    %10 = stablehlo.subtract %7, %9 : tensor<1x2x4x4xf32>
                    %11 = stablehlo.exponential %10 : tensor<1x2x4x4xf32>
                    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %12 = stablehlo.reduce(%11 init: %cst_2) applies stablehlo.add across dimensions = [3] : (tensor<1x2x4x4xf32>, tensor<f32>) -> tensor<1x2x4xf32>
                    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2] : (tensor<1x2x4xf32>) -> tensor<1x2x4x4xf32>
                    %14 = stablehlo.divide %11, %13 : tensor<1x2x4x4xf32>
                    %15 = stablehlo.dot_general %14, %arg2, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x2x4x4xf32>, tensor<1x4x2x8xf32>) -> tensor<1x2x4x8xf32>
                    %16 = stablehlo.transpose %15, dims = [0, 2, 1, 3] : (tensor<1x2x4x8xf32>) -> tensor<1x4x2x8xf32>
                    return %16 : tensor<1x4x2x8xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_mlir_module_for_program_lowers_dot_product_attention_composition_for_f32_on_cuda() {
        // `f32` operands do not qualify for the cuDNN flash-attention call, so even a CUDA target lowers the
        // portable composition.
        let (program, input_types, output_types) =
            dot_product_attention_fixture_program(DataType::F32, 8, AttentionMask::Causal);
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;
        assert!(!module.contains("fmhaSoftmax"));
        assert!(module.contains("stablehlo.dot_general"));
    }

    #[test]
    fn test_lower_mlir_module_for_program_lowers_dot_product_attention_composition_for_unaligned_head_dim_on_cuda() {
        // A head dimension that is not a multiple of 8 fails cuDNN's compile-time gate, so a CUDA target with `bf16`
        // operands still lowers the portable composition (which upcasts its softmax to `f32`).
        let (program, input_types, output_types) =
            dot_product_attention_fixture_program(DataType::BF16, 4, AttentionMask::Causal);
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;
        assert!(!module.contains("fmhaSoftmax"));
        assert!(module.contains("stablehlo.dot_general"));
        assert!(module.contains("stablehlo.convert"));
    }

    /// Returns the inferred output [`ArrayType`]s of the provided staged output atoms.
    fn builder_output_types(builder: &XlaProgramBuilder, outputs: &[AtomId]) -> Vec<ArrayType> {
        outputs.iter().map(|output| builder.atoms()[output.index()].r#type().into_owned()).collect()
    }

    /// Builds an extended dot-product attention fixture program: `query [2, 4, query_heads, 8]` over
    /// `key`/`value [2, 4, key_value_heads, 8]` at the provided data type, with an optional broadcast-batch bias
    /// `[1, query_heads, 4, 4]` and an optional trailing `i32[2]` sequence-length pair.
    fn dot_product_attention_extended_fixture_program(
        data_type: DataType,
        query_heads: usize,
        key_value_heads: usize,
        bias: bool,
        sequence_lengths: bool,
        operation: DotProductAttentionOperation,
    ) -> (XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>, Vec<ArrayType>, Vec<ArrayType>) {
        let query_type = ArrayType::new(
            data_type,
            Shape::new(vec![
                Dimension::Static(2),
                Dimension::Static(4),
                Dimension::Static(query_heads),
                Dimension::Static(8),
            ]),
        );
        let key_value_type = ArrayType::new(
            data_type,
            Shape::new(vec![
                Dimension::Static(2),
                Dimension::Static(4),
                Dimension::Static(key_value_heads),
                Dimension::Static(8),
            ]),
        );
        let mut builder = XlaProgramBuilder::new();
        let mut input_types = vec![query_type.clone(), key_value_type.clone(), key_value_type];
        if bias {
            input_types.push(ArrayType::new(
                data_type,
                Shape::new(vec![
                    Dimension::Static(1),
                    Dimension::Static(query_heads),
                    Dimension::Static(4),
                    Dimension::Static(4),
                ]),
            ));
        }
        if sequence_lengths {
            let lengths_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2)]));
            input_types.push(lengths_type.clone());
            input_types.push(lengths_type);
        }
        let inputs = input_types.iter().map(|input_type| builder.add_input(input_type.clone())).collect::<Vec<_>>();
        let outputs = builder.add_instruction(operation, Vec::new(), inputs).unwrap().to_vec();
        let output_count = outputs.len();
        let output_types = builder_output_types(&builder, outputs.as_slice());
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                outputs,
                vec![Placeholder; input_types.len()],
                vec![Placeholder; output_count],
            )
            .unwrap();
        (program, input_types, output_types)
    }

    /// Builds the matching backward fixture program for [`dot_product_attention_extended_fixture_program`]'s
    /// operand convention: `(query, key, value[, bias], output, activation, output_cotangent[, sequence lengths])`.
    fn dot_product_attention_backward_fixture_program(
        data_type: DataType,
        query_heads: usize,
        key_value_heads: usize,
        bias: bool,
        sequence_lengths: bool,
        operation: DotProductAttentionBackwardOperation,
    ) -> (XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>, Vec<ArrayType>, Vec<ArrayType>) {
        let query_type = ArrayType::new(
            data_type,
            Shape::new(vec![
                Dimension::Static(2),
                Dimension::Static(4),
                Dimension::Static(query_heads),
                Dimension::Static(8),
            ]),
        );
        let key_value_type = ArrayType::new(
            data_type,
            Shape::new(vec![
                Dimension::Static(2),
                Dimension::Static(4),
                Dimension::Static(key_value_heads),
                Dimension::Static(8),
            ]),
        );
        let activation_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(query_heads), Dimension::Static(4)]),
        );
        let mut builder = XlaProgramBuilder::new();
        let mut input_types = vec![query_type.clone(), key_value_type.clone(), key_value_type];
        if bias {
            input_types.push(ArrayType::new(
                data_type,
                Shape::new(vec![
                    Dimension::Static(1),
                    Dimension::Static(query_heads),
                    Dimension::Static(4),
                    Dimension::Static(4),
                ]),
            ));
        }
        input_types.extend([query_type.clone(), activation_type, query_type]);
        if sequence_lengths {
            let lengths_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2)]));
            input_types.push(lengths_type.clone());
            input_types.push(lengths_type);
        }
        let inputs = input_types.iter().map(|input_type| builder.add_input(input_type.clone())).collect::<Vec<_>>();
        let outputs = builder.add_instruction(operation, Vec::new(), inputs).unwrap().to_vec();
        let output_count = outputs.len();
        let output_types = builder_output_types(&builder, outputs.as_slice());
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                outputs,
                vec![Placeholder; input_types.len()],
                vec![Placeholder; output_count],
            )
            .unwrap();
        (program, input_types, output_types)
    }

    #[test]
    fn test_lower_mlir_module_for_program_emits_fmha_scale_bias_softmax_on_cuda() {
        // A bias operand switches the fused target to `__cudnn$fmhaScaleBiasSoftmax`: the bias rides as the fourth
        // operand with the standard `[3, 2, 1, 0]` layout while the backend configuration is unchanged (the bias
        // enters the kernel post-scale and pre-mask).
        let (program, input_types, output_types) = dot_product_attention_extended_fixture_program(
            DataType::BF16,
            2,
            2,
            true,
            false,
            DotProductAttentionOperation::new(0.125, AttentionMask::Causal),
        );
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x4x2x8xbf16>, %arg1: tensor<2x4x2x8xbf16>, %arg2: tensor<2x4x2x8xbf16>, %arg3: tensor<1x2x4x4xbf16>) -> tensor<2x4x2x8xbf16> {
                    %0:2 = stablehlo.custom_call @__cudnn$fmhaScaleBiasSoftmax(%arg0, %arg1, %arg2, %arg3) {api_version = 2 : i32, backend_config = "{\22operation_queue_id\22:\220\22,\22cudnn_fmha_backend_config\22:{\22algorithm\22:{\22algo_id\22:\220\22,\22math_type\22:\22TENSOR_OP_MATH\22,\22tuning_knobs\22:{\2217\22:\221\22,\2224\22:\220\22},\22is_cudnn_frontend\22:true,\22workspace_size\22:\220\22},\22fmha_scale\22:0.125,\22intermediate_tensor_shape\22:{\22element_type\22:\22BF16\22,\22dimensions\22:[\222\22,\222\22,\224\22,\224\22],\22tuple_shapes\22:[],\22layout\22:{\22dim_level_types\22:[],\22dim_unique\22:[],\22dim_ordered\22:[],\22minor_to_major\22:[\223\22,\222\22,\221\22,\220\22],\22tiles\22:[],\22element_size_in_bits\22:\220\22,\22memory_space\22:\220\22,\22index_primitive_type\22:\22PRIMITIVE_TYPE_INVALID\22,\22pointer_primitive_type\22:\22PRIMITIVE_TYPE_INVALID\22,\22dynamic_shape_metadata_prefix_bytes\22:\220\22},\22is_dynamic_dimension\22:[false,false,false,false]},\22is_flash_attention\22:true,\22mask_type\22:\22CAUSAL\22,\22bmm1_dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\223\22],\22rhs_contracting_dimensions\22:[\223\22],\22lhs_batch_dimensions\22:[\220\22,\222\22],\22rhs_batch_dimensions\22:[\220\22,\222\22]},\22bmm2_dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\223\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[\220\22,\221\22],\22rhs_batch_dimensions\22:[\220\22,\222\22]},\22dropout_rate\22:0.0,\22seed\22:42,\22sliding_window_length\22:0,\22max_seg_per_batch\22:1,\22is_paged_attention\22:false}}", operand_layouts = [dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>], result_layouts = [dense<[3, 1, 2, 0]> : tensor<4xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x2x8xbf16>, tensor<2x4x2x8xbf16>, tensor<2x4x2x8xbf16>, tensor<1x2x4x4xbf16>) -> (tensor<2x2x4x8xbf16>, tensor<0xui8>)
                    %1 = stablehlo.transpose %0#0, dims = [0, 2, 1, 3] : (tensor<2x2x4x8xbf16>) -> tensor<2x4x2x8xbf16>
                    return %1 : tensor<2x4x2x8xbf16>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_mlir_module_for_program_emits_fmha_training_forward_with_padding_on_cuda() {
        // The training forward with variable sequence lengths emits the hardware-validated 3-tuple form: the
        // `i32[batch]` sequence lengths trail the operands with `dense<0>` layouts, the mask kind composes to
        // `PADDING_CAUSAL`, and the `f32[b, n, t]` activation statistic result (layout `{2, 1, 0}`) sits between
        // the attention output and the zero-sized workspace.
        let (program, input_types, output_types) = dot_product_attention_extended_fixture_program(
            DataType::BF16,
            2,
            2,
            false,
            true,
            DotProductAttentionOperation::new(0.125, AttentionMask::Causal).with_activation_output(),
        );
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x4x2x8xbf16>, %arg1: tensor<2x4x2x8xbf16>, %arg2: tensor<2x4x2x8xbf16>, %arg3: tensor<2xi32>, %arg4: tensor<2xi32>) -> (tensor<2x4x2x8xbf16>, tensor<2x2x4xf32>) {
                    %0:3 = stablehlo.custom_call @__cudnn$fmhaSoftmax(%arg0, %arg1, %arg2, %arg3, %arg4) {api_version = 2 : i32, backend_config = "{\22operation_queue_id\22:\220\22,\22cudnn_fmha_backend_config\22:{\22algorithm\22:{\22algo_id\22:\220\22,\22math_type\22:\22TENSOR_OP_MATH\22,\22tuning_knobs\22:{\2217\22:\221\22,\2224\22:\220\22},\22is_cudnn_frontend\22:true,\22workspace_size\22:\220\22},\22fmha_scale\22:0.125,\22intermediate_tensor_shape\22:{\22element_type\22:\22BF16\22,\22dimensions\22:[\222\22,\222\22,\224\22,\224\22],\22tuple_shapes\22:[],\22layout\22:{\22dim_level_types\22:[],\22dim_unique\22:[],\22dim_ordered\22:[],\22minor_to_major\22:[\223\22,\222\22,\221\22,\220\22],\22tiles\22:[],\22element_size_in_bits\22:\220\22,\22memory_space\22:\220\22,\22index_primitive_type\22:\22PRIMITIVE_TYPE_INVALID\22,\22pointer_primitive_type\22:\22PRIMITIVE_TYPE_INVALID\22,\22dynamic_shape_metadata_prefix_bytes\22:\220\22},\22is_dynamic_dimension\22:[false,false,false,false]},\22is_flash_attention\22:true,\22mask_type\22:\22PADDING_CAUSAL\22,\22bmm1_dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\223\22],\22rhs_contracting_dimensions\22:[\223\22],\22lhs_batch_dimensions\22:[\220\22,\222\22],\22rhs_batch_dimensions\22:[\220\22,\222\22]},\22bmm2_dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\223\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[\220\22,\221\22],\22rhs_batch_dimensions\22:[\220\22,\222\22]},\22dropout_rate\22:0.0,\22seed\22:42,\22sliding_window_length\22:0,\22max_seg_per_batch\22:1,\22is_paged_attention\22:false}}", operand_layouts = [dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>], result_layouts = [dense<[3, 1, 2, 0]> : tensor<4xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x2x8xbf16>, tensor<2x4x2x8xbf16>, tensor<2x4x2x8xbf16>, tensor<2xi32>, tensor<2xi32>) -> (tensor<2x2x4x8xbf16>, tensor<2x2x4xf32>, tensor<0xui8>)
                    %1 = stablehlo.transpose %0#0, dims = [0, 2, 1, 3] : (tensor<2x2x4x8xbf16>) -> tensor<2x4x2x8xbf16>
                    return %1, %0#1 : tensor<2x4x2x8xbf16>, tensor<2x2x4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_mlir_module_for_program_emits_fmha_dropout_with_sliding_window_on_cuda() {
        // Dropout switches the fused target to `__cudnn$fmhaSoftmaxDropout` and threads its rate and seed through
        // the backend configuration, while the sliding window rides purely as the `sliding_window_length`
        // configuration field.
        let (program, input_types, output_types) = dot_product_attention_extended_fixture_program(
            DataType::BF16,
            2,
            2,
            false,
            false,
            DotProductAttentionOperation::new(0.125, AttentionMask::Causal)
                .with_sliding_window(2)
                .with_dropout((0.5, 123)),
        );
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x4x2x8xbf16>, %arg1: tensor<2x4x2x8xbf16>, %arg2: tensor<2x4x2x8xbf16>) -> tensor<2x4x2x8xbf16> {
                    %0:2 = stablehlo.custom_call @__cudnn$fmhaSoftmaxDropout(%arg0, %arg1, %arg2) {api_version = 2 : i32, backend_config = "{\22operation_queue_id\22:\220\22,\22cudnn_fmha_backend_config\22:{\22algorithm\22:{\22algo_id\22:\220\22,\22math_type\22:\22TENSOR_OP_MATH\22,\22tuning_knobs\22:{\2217\22:\221\22,\2224\22:\220\22},\22is_cudnn_frontend\22:true,\22workspace_size\22:\220\22},\22fmha_scale\22:0.125,\22intermediate_tensor_shape\22:{\22element_type\22:\22BF16\22,\22dimensions\22:[\222\22,\222\22,\224\22,\224\22],\22tuple_shapes\22:[],\22layout\22:{\22dim_level_types\22:[],\22dim_unique\22:[],\22dim_ordered\22:[],\22minor_to_major\22:[\223\22,\222\22,\221\22,\220\22],\22tiles\22:[],\22element_size_in_bits\22:\220\22,\22memory_space\22:\220\22,\22index_primitive_type\22:\22PRIMITIVE_TYPE_INVALID\22,\22pointer_primitive_type\22:\22PRIMITIVE_TYPE_INVALID\22,\22dynamic_shape_metadata_prefix_bytes\22:\220\22},\22is_dynamic_dimension\22:[false,false,false,false]},\22is_flash_attention\22:true,\22mask_type\22:\22CAUSAL\22,\22bmm1_dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\223\22],\22rhs_contracting_dimensions\22:[\223\22],\22lhs_batch_dimensions\22:[\220\22,\222\22],\22rhs_batch_dimensions\22:[\220\22,\222\22]},\22bmm2_dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\223\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[\220\22,\221\22],\22rhs_batch_dimensions\22:[\220\22,\222\22]},\22dropout_rate\22:0.5,\22seed\22:123,\22sliding_window_length\22:2,\22max_seg_per_batch\22:1,\22is_paged_attention\22:false}}", operand_layouts = [dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>], result_layouts = [dense<[3, 1, 2, 0]> : tensor<4xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x2x8xbf16>, tensor<2x4x2x8xbf16>, tensor<2x4x2x8xbf16>) -> (tensor<2x2x4x8xbf16>, tensor<0xui8>)
                    %1 = stablehlo.transpose %0#0, dims = [0, 2, 1, 3] : (tensor<2x2x4x8xbf16>) -> tensor<2x4x2x8xbf16>
                    return %1 : tensor<2x4x2x8xbf16>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_mlir_module_for_program_emits_fmha_backward_with_bias_on_cuda() {
        // The fused backward call reorders the traced operands `(q, k, v, bias, output, activation, output
        // cotangent)` into the kernel's `(Q, K, V, activation, dO, bias, O)` call order (the bias sits between `dO`
        // and `O`), swaps the four gradient-GEMM dot-dimension-number blocks into the configuration, keeps the
        // operand-typed intermediate score shape, and returns `(dQ, dK, dV, dBias, workspace)` with the gradient
        // transposes back to `BTNH` (the bias cotangent keeps its own shape and default layout).
        let (program, input_types, output_types) = dot_product_attention_backward_fixture_program(
            DataType::BF16,
            2,
            2,
            true,
            false,
            DotProductAttentionBackwardOperation::new(0.125, AttentionMask::Causal),
        );
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x4x2x8xbf16>, %arg1: tensor<2x4x2x8xbf16>, %arg2: tensor<2x4x2x8xbf16>, %arg3: tensor<1x2x4x4xbf16>, %arg4: tensor<2x4x2x8xbf16>, %arg5: tensor<2x2x4xf32>, %arg6: tensor<2x4x2x8xbf16>) -> (tensor<2x4x2x8xbf16>, tensor<2x4x2x8xbf16>, tensor<2x4x2x8xbf16>, tensor<1x2x4x4xbf16>) {
                    %0:5 = stablehlo.custom_call @__cudnn$fmhaScaleBiasSoftmaxBackward(%arg0, %arg1, %arg2, %arg5, %arg6, %arg3, %arg4) {api_version = 2 : i32, backend_config = "{\22operation_queue_id\22:\220\22,\22cudnn_fmha_backend_config\22:{\22algorithm\22:{\22algo_id\22:\220\22,\22math_type\22:\22TENSOR_OP_MATH\22,\22tuning_knobs\22:{\2217\22:\221\22,\2224\22:\220\22},\22is_cudnn_frontend\22:true,\22workspace_size\22:\220\22},\22fmha_scale\22:0.125,\22intermediate_tensor_shape\22:{\22element_type\22:\22BF16\22,\22dimensions\22:[\222\22,\222\22,\224\22,\224\22],\22tuple_shapes\22:[],\22layout\22:{\22dim_level_types\22:[],\22dim_unique\22:[],\22dim_ordered\22:[],\22minor_to_major\22:[\223\22,\222\22,\221\22,\220\22],\22tiles\22:[],\22element_size_in_bits\22:\220\22,\22memory_space\22:\220\22,\22index_primitive_type\22:\22PRIMITIVE_TYPE_INVALID\22,\22pointer_primitive_type\22:\22PRIMITIVE_TYPE_INVALID\22,\22dynamic_shape_metadata_prefix_bytes\22:\220\22},\22is_dynamic_dimension\22:[false,false,false,false]},\22is_flash_attention\22:true,\22mask_type\22:\22CAUSAL\22,\22bmm1_grad_gemm1_dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\222\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[\220\22,\221\22],\22rhs_batch_dimensions\22:[\220\22,\222\22]},\22bmm1_grad_gemm2_dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\223\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[\220\22,\221\22],\22rhs_batch_dimensions\22:[\220\22,\222\22]},\22bmm2_grad_gemm1_dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\222\22],\22rhs_contracting_dimensions\22:[\221\22],\22lhs_batch_dimensions\22:[\220\22,\221\22],\22rhs_batch_dimensions\22:[\220\22,\222\22]},\22bmm2_grad_gemm2_dot_dimension_numbers\22:{\22lhs_contracting_dimensions\22:[\223\22],\22rhs_contracting_dimensions\22:[\223\22],\22lhs_batch_dimensions\22:[\220\22,\222\22],\22rhs_batch_dimensions\22:[\220\22,\222\22]},\22dropout_rate\22:0.0,\22seed\22:42,\22sliding_window_length\22:0,\22max_seg_per_batch\22:1,\22is_paged_attention\22:false}}", operand_layouts = [dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>], result_layouts = [dense<[3, 1, 2, 0]> : tensor<4xindex>, dense<[3, 1, 2, 0]> : tensor<4xindex>, dense<[3, 1, 2, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x2x8xbf16>, tensor<2x4x2x8xbf16>, tensor<2x4x2x8xbf16>, tensor<2x2x4xf32>, tensor<2x4x2x8xbf16>, tensor<1x2x4x4xbf16>, tensor<2x4x2x8xbf16>) -> (tensor<2x2x4x8xbf16>, tensor<2x2x4x8xbf16>, tensor<2x2x4x8xbf16>, tensor<1x2x4x4xbf16>, tensor<0xui8>)
                    %1 = stablehlo.transpose %0#0, dims = [0, 2, 1, 3] : (tensor<2x2x4x8xbf16>) -> tensor<2x4x2x8xbf16>
                    %2 = stablehlo.transpose %0#1, dims = [0, 2, 1, 3] : (tensor<2x2x4x8xbf16>) -> tensor<2x4x2x8xbf16>
                    %3 = stablehlo.transpose %0#2, dims = [0, 2, 1, 3] : (tensor<2x2x4x8xbf16>) -> tensor<2x4x2x8xbf16>
                    return %1, %2, %3, %0#3 : tensor<2x4x2x8xbf16>, tensor<2x4x2x8xbf16>, tensor<2x4x2x8xbf16>, tensor<1x2x4x4xbf16>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_dot_product_attention_composition_with_extensions() {
        // Without target information the extended features all lower through the portable StableHLO composition in
        // one module: the grouped key/value heads expand via broadcast + reshape, the broadcast-batch bias adds to
        // the scaled scores, the sliding window tightens the causal mask with a second iota compare, and the
        // requested activation statistic is `max + ln(sum)` from the softmax's own reductions.
        let (program, input_types, output_types) = dot_product_attention_extended_fixture_program(
            DataType::F32,
            4,
            2,
            true,
            false,
            DotProductAttentionOperation::new(0.125, AttentionMask::Causal)
                .with_sliding_window(2)
                .with_activation_output(),
        );
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x4x4x8xf32>, %arg1: tensor<2x4x2x8xf32>, %arg2: tensor<2x4x2x8xf32>, %arg3: tensor<1x4x4x4xf32>) -> (tensor<2x4x4x8xf32>, tensor<2x4x4xf32>) {
                    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2, 4] : (tensor<2x4x2x8xf32>) -> tensor<2x4x2x2x8xf32>
                    %1 = stablehlo.reshape %0 : (tensor<2x4x2x2x8xf32>) -> tensor<2x4x4x8xf32>
                    %2 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2, 4] : (tensor<2x4x2x8xf32>) -> tensor<2x4x2x2x8xf32>
                    %3 = stablehlo.reshape %2 : (tensor<2x4x2x2x8xf32>) -> tensor<2x4x4x8xf32>
                    %4 = stablehlo.dot_general %arg0, %1, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x4x4x8xf32>, tensor<2x4x4x8xf32>) -> tensor<2x4x4x4xf32>
                    %cst = stablehlo.constant dense<1.250000e-01> : tensor<f32>
                    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4x4x4xf32>
                    %6 = stablehlo.multiply %4, %5 : tensor<2x4x4x4xf32>
                    %7 = stablehlo.broadcast_in_dim %arg3, dims = [0, 1, 2, 3] : (tensor<1x4x4x4xf32>) -> tensor<2x4x4x4xf32>
                    %8 = stablehlo.add %6, %7 : tensor<2x4x4x4xf32>
                    %9 = stablehlo.iota dim = 3 : tensor<2x4x4x4xi32>
                    %10 = stablehlo.iota dim = 2 : tensor<2x4x4x4xi32>
                    %11 = stablehlo.compare LE, %9, %10, SIGNED : (tensor<2x4x4x4xi32>, tensor<2x4x4x4xi32>) -> tensor<2x4x4x4xi1>
                    %c = stablehlo.constant dense<2> : tensor<i32>
                    %12 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x4x4x4xi32>
                    %13 = stablehlo.subtract %10, %12 : tensor<2x4x4x4xi32>
                    %14 = stablehlo.compare GT, %9, %13, SIGNED : (tensor<2x4x4x4xi32>, tensor<2x4x4x4xi32>) -> tensor<2x4x4x4xi1>
                    %15 = stablehlo.and %11, %14 : tensor<2x4x4x4xi1>
                    %cst_0 = stablehlo.constant dense<-1.000000e+30> : tensor<f32>
                    %16 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2x4x4x4xf32>
                    %17 = stablehlo.select %15, %8, %16 : tensor<2x4x4x4xi1>, tensor<2x4x4x4xf32>
                    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
                    %18 = stablehlo.reduce(%17 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<2x4x4x4xf32>, tensor<f32>) -> tensor<2x4x4xf32>
                    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1, 2] : (tensor<2x4x4xf32>) -> tensor<2x4x4x4xf32>
                    %20 = stablehlo.subtract %17, %19 : tensor<2x4x4x4xf32>
                    %21 = stablehlo.exponential %20 : tensor<2x4x4x4xf32>
                    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %22 = stablehlo.reduce(%21 init: %cst_2) applies stablehlo.add across dimensions = [3] : (tensor<2x4x4x4xf32>, tensor<f32>) -> tensor<2x4x4xf32>
                    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1, 2] : (tensor<2x4x4xf32>) -> tensor<2x4x4x4xf32>
                    %24 = stablehlo.divide %21, %23 : tensor<2x4x4x4xf32>
                    %25 = stablehlo.dot_general %24, %3, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x4x4xf32>, tensor<2x4x4x8xf32>) -> tensor<2x4x4x8xf32>
                    %26 = stablehlo.transpose %25, dims = [0, 2, 1, 3] : (tensor<2x4x4x8xf32>) -> tensor<2x4x4x8xf32>
                    %27 = stablehlo.log %22 : tensor<2x4x4xf32>
                    %28 = stablehlo.add %18, %27 : tensor<2x4x4xf32>
                    return %26, %28 : tensor<2x4x4x8xf32>, tensor<2x4x4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_dot_product_attention_backward_composition() {
        // The backward composition fallback mirrors the reference backward composition: the masked logits are
        // recomputed, the weights recover as `P = exp(S - stat)`, the four documented `dot_general`s produce the
        // cotangents, the grouped-query key/value cotangents sum over the per-head group axis, and the bias
        // cotangent sums over the bias's broadcast batch dimension.
        let (program, input_types, output_types) = dot_product_attention_backward_fixture_program(
            DataType::F32,
            4,
            2,
            true,
            false,
            DotProductAttentionBackwardOperation::new(0.125, AttentionMask::Causal),
        );
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x4x4x8xf32>, %arg1: tensor<2x4x2x8xf32>, %arg2: tensor<2x4x2x8xf32>, %arg3: tensor<1x4x4x4xf32>, %arg4: tensor<2x4x4x8xf32>, %arg5: tensor<2x4x4xf32>, %arg6: tensor<2x4x4x8xf32>) -> (tensor<2x4x4x8xf32>, tensor<2x4x2x8xf32>, tensor<2x4x2x8xf32>, tensor<1x4x4x4xf32>) {
                    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2, 4] : (tensor<2x4x2x8xf32>) -> tensor<2x4x2x2x8xf32>
                    %1 = stablehlo.reshape %0 : (tensor<2x4x2x2x8xf32>) -> tensor<2x4x4x8xf32>
                    %2 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2, 4] : (tensor<2x4x2x8xf32>) -> tensor<2x4x2x2x8xf32>
                    %3 = stablehlo.reshape %2 : (tensor<2x4x2x2x8xf32>) -> tensor<2x4x4x8xf32>
                    %4 = stablehlo.dot_general %arg0, %1, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x4x4x8xf32>, tensor<2x4x4x8xf32>) -> tensor<2x4x4x4xf32>
                    %cst = stablehlo.constant dense<1.250000e-01> : tensor<f32>
                    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4x4x4xf32>
                    %6 = stablehlo.multiply %4, %5 : tensor<2x4x4x4xf32>
                    %7 = stablehlo.broadcast_in_dim %arg3, dims = [0, 1, 2, 3] : (tensor<1x4x4x4xf32>) -> tensor<2x4x4x4xf32>
                    %8 = stablehlo.add %6, %7 : tensor<2x4x4x4xf32>
                    %9 = stablehlo.iota dim = 3 : tensor<2x4x4x4xi32>
                    %10 = stablehlo.iota dim = 2 : tensor<2x4x4x4xi32>
                    %11 = stablehlo.compare LE, %9, %10, SIGNED : (tensor<2x4x4x4xi32>, tensor<2x4x4x4xi32>) -> tensor<2x4x4x4xi1>
                    %cst_0 = stablehlo.constant dense<-1.000000e+30> : tensor<f32>
                    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2x4x4x4xf32>
                    %13 = stablehlo.select %11, %8, %12 : tensor<2x4x4x4xi1>, tensor<2x4x4x4xf32>
                    %14 = stablehlo.broadcast_in_dim %arg5, dims = [0, 1, 2] : (tensor<2x4x4xf32>) -> tensor<2x4x4x4xf32>
                    %15 = stablehlo.subtract %13, %14 : tensor<2x4x4x4xf32>
                    %16 = stablehlo.exponential %15 : tensor<2x4x4x4xf32>
                    %17 = stablehlo.dot_general %arg6, %3, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x4x4x8xf32>, tensor<2x4x4x8xf32>) -> tensor<2x4x4x4xf32>
                    %18 = stablehlo.multiply %arg6, %arg4 : tensor<2x4x4x8xf32>
                    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %19 = stablehlo.reduce(%18 init: %cst_1) applies stablehlo.add across dimensions = [3] : (tensor<2x4x4x8xf32>, tensor<f32>) -> tensor<2x4x4xf32>
                    %20 = stablehlo.transpose %19, dims = [0, 2, 1] : (tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
                    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2] : (tensor<2x4x4xf32>) -> tensor<2x4x4x4xf32>
                    %22 = stablehlo.subtract %17, %21 : tensor<2x4x4x4xf32>
                    %23 = stablehlo.multiply %16, %22 : tensor<2x4x4x4xf32>
                    %cst_2 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
                    %24 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<2x4x4x4xf32>
                    %25 = stablehlo.multiply %23, %24 : tensor<2x4x4x4xf32>
                    %26 = stablehlo.dot_general %25, %1, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x4x4xf32>, tensor<2x4x4x8xf32>) -> tensor<2x4x4x8xf32>
                    %27 = stablehlo.transpose %26, dims = [0, 2, 1, 3] : (tensor<2x4x4x8xf32>) -> tensor<2x4x4x8xf32>
                    %28 = stablehlo.dot_general %25, %arg0, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x4x4xf32>, tensor<2x4x4x8xf32>) -> tensor<2x4x4x8xf32>
                    %29 = stablehlo.transpose %28, dims = [0, 2, 1, 3] : (tensor<2x4x4x8xf32>) -> tensor<2x4x4x8xf32>
                    %30 = stablehlo.dot_general %16, %arg6, batching_dims = [0, 1] x [0, 2], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4x4x4xf32>, tensor<2x4x4x8xf32>) -> tensor<2x4x4x8xf32>
                    %31 = stablehlo.transpose %30, dims = [0, 2, 1, 3] : (tensor<2x4x4x8xf32>) -> tensor<2x4x4x8xf32>
                    %32 = stablehlo.reshape %29 : (tensor<2x4x4x8xf32>) -> tensor<2x4x2x2x8xf32>
                    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %33 = stablehlo.reduce(%32 init: %cst_3) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x2x8xf32>, tensor<f32>) -> tensor<2x4x2x8xf32>
                    %34 = stablehlo.reshape %31 : (tensor<2x4x4x8xf32>) -> tensor<2x4x2x2x8xf32>
                    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %35 = stablehlo.reduce(%34 init: %cst_4) applies stablehlo.add across dimensions = [3] : (tensor<2x4x2x2x8xf32>, tensor<f32>) -> tensor<2x4x2x8xf32>
                    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %36 = stablehlo.reduce(%23 init: %cst_5) applies stablehlo.add across dimensions = [0] : (tensor<2x4x4x4xf32>, tensor<f32>) -> tensor<4x4x4xf32>
                    %37 = stablehlo.reshape %36 : (tensor<4x4x4xf32>) -> tensor<1x4x4x4xf32>
                    return %27, %33, %35, %37 : tensor<2x4x4x8xf32>, tensor<2x4x2x8xf32>, tensor<2x4x2x8xf32>, tensor<1x4x4x4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_mlir_module_for_program_rejects_dot_product_attention_dropout_off_the_fast_path() {
        // No composition implements dropout, so a dropout-carrying operation that misses the fused fast-path gate
        // (here: `f32` operands on CUDA, and any operands without a target platform) reports an explicit error
        // instead of silently computing dropout-free attention, for both the forward and the backward operations.
        let (program, input_types, output_types) = dot_product_attention_extended_fixture_program(
            DataType::F32,
            2,
            2,
            false,
            false,
            DotProductAttentionOperation::new(0.125, AttentionMask::Causal).with_dropout((0.5, 123)),
        );
        for platform in [Some("cuda"), None] {
            let result =
                lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, platform);
            assert!(matches!(
                result,
                Err(error) if error.to_string().contains(
                    "'dot_product_attention' dropout is only supported by the fused CUDA lowering",
                ),
            ));
        }
        let (program, input_types, output_types) = dot_product_attention_backward_fixture_program(
            DataType::F32,
            2,
            2,
            false,
            false,
            DotProductAttentionBackwardOperation::new(0.125, AttentionMask::Causal).with_dropout((0.5, 123)),
        );
        let result =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"));
        assert!(matches!(
            result,
            Err(error) if error.to_string().contains(
                "'dot_product_attention_backward' dropout is only supported by the fused CUDA lowering",
            ),
        ));
    }

    #[test]
    fn test_lower_mlir_module_for_program_never_emits_fmha_for_f8_attention() {
        // F8 attention is a validated NO-GO on the pinned cuDNN (the fused kernels reject F8 I/O at compile time),
        // so `f8` operands never qualify for the fMHA fast path and take the ordinary composition route even on
        // CUDA targets.
        let (program, input_types, output_types) = dot_product_attention_extended_fixture_program(
            DataType::F8E4M3FN,
            2,
            2,
            false,
            false,
            DotProductAttentionOperation::new(0.125, AttentionMask::Causal),
        );
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;
        assert!(!module.contains("fmha"));
        assert!(module.contains("stablehlo.dot_general"));
        assert!(module.contains("stablehlo.convert"));
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_rng_bit_generator() {
        use ryft_core::operations::random::{RandomAlgorithm, RngBitGeneratorOperation};

        let state_type = ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(2)]));
        let bits_type = ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Static(4)]));
        let mut builder = XlaProgramBuilder::new();
        let state = builder.add_input(state_type.clone());
        let outputs = builder
            .add_instruction(
                RngBitGeneratorOperation::new(RandomAlgorithm::ThreeFry, bits_type.clone()),
                Vec::new(),
                vec![state],
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        let input_types = vec![state_type.clone()];
        let output_types = vec![state_type, bits_type];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>) {
                    %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm = THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
                    return %output_state, %output : tensor<2xui64>, tensor<4xui32>
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
        let printed = body_builder.add_instruction(PrintOperation::new("iteration"), Vec::new(), vec![x]).unwrap()[0];
        let sum = body_builder.add_instruction(AddOperation, Vec::new(), vec![carry, printed]).unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        let scan = CoreScanOperation::<XlaArrayConstant>::new(1, 3);

        let stacked_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = XlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let init = builder.add_input(scalar_f64.clone());
        let stacked_inputs = builder.add_input(stacked_type.clone());
        let output = builder
            .add_instruction(XlaOperation::Scan(scan), vec![body_region], vec![init, stacked_inputs])
            .unwrap()[0];
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
        let printed =
            true_builder.add_instruction(PrintOperation::new("taken"), Vec::new(), vec![true_input]).unwrap()[0];
        let negated = true_builder.add_instruction(NegOperation, Vec::new(), vec![printed]).unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![negated], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = XlaProgramBuilder::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(xla_identity_branch(input_type.clone()).entry_region_ref());
        let predicate = builder.add_input(predicate_type.clone());
        let input = builder.add_input(input_type.clone());
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
            )
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
        let printed = callee_builder
            .add_instruction(PrintOperation::new("callee"), Vec::new(), vec![callee_input])
            .unwrap()[0];
        let callee_output =
            callee_builder.add_instruction(AddOperation, Vec::new(), vec![printed, printed]).unwrap()[0];
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

        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))
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
        let printed = body_builder.add_instruction(PrintOperation::new("iteration"), Vec::new(), vec![x]).unwrap()[0];
        let sum = body_builder.add_instruction(AddOperation, Vec::new(), vec![carry, printed]).unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        let scan = CoreScanOperation::<XlaArrayConstant>::new(1, 3);

        let stacked_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = XlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let init = builder.add_input(scalar_f64.clone());
        let stacked_inputs = builder.add_input(stacked_type.clone());
        let output = builder
            .add_instruction(XlaOperation::Scan(scan), vec![body_region], vec![init, stacked_inputs])
            .unwrap()[0];
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
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let condition = {
            let mut builder = XlaProgramBuilder::new();
            let state = builder.add_input(state_type.clone());
            let zero = builder.add_instruction(ZeroLikeOperation, Vec::new(), vec![state]).unwrap()[0];
            let predicate = builder
                .add_instruction(CompareOperation::new(ComparisonDirection::GreaterThan), Vec::new(), vec![state, zero])
                .unwrap()[0];
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![predicate], vec![Placeholder], vec![Placeholder])
        }
        .unwrap();
        let body = {
            let mut builder = XlaProgramBuilder::new();
            let state = builder.add_input(state_type.clone());
            let one = builder.add_instruction(OneLikeOperation, Vec::new(), vec![state]).unwrap()[0];
            let next = builder.add_instruction(SubOperation, Vec::new(), vec![state, one]).unwrap()[0];
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![next], vec![Placeholder], vec![Placeholder])
        }
        .unwrap();
        let mut builder = XlaProgramBuilder::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(body.entry_region_ref());
        let state = builder.add_input(state_type.clone());
        let output = builder
            .add_instruction(
                XlaOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                vec![state],
            )
            .unwrap()[0];
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
        T: Clone + ryft_core::operations::math::Sin + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    {
        inputs.0.clone() * inputs.1 + inputs.0.sin().unwrap()
    }

    fn scalar_quartic_plus_sin<T>(x: T) -> T
    where
        T: Clone + ryft_core::operations::math::Sin + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    {
        x.clone() * x.clone() * x.clone() * x.clone() + x.sin().unwrap()
    }

    static TEST_ARRAY_DOMAIN: EagerContext<CpuArray, ArrayOperation<CpuArray>> =
        EagerContext::<CpuArray, ArrayOperation<CpuArray>>::new();

    #[test]
    fn test_plain_scalar_bilinear_sin_jit_stablehlo() {
        let (_, compiled): (
            CpuArray,
            ryft_core::programs::Program<
                CpuArray,
                ryft_core::backends::arrays::ArrayOperation<CpuArray>,
                (CpuArray, CpuArray),
                CpuArray,
            >,
        ) = TEST_ARRAY_DOMAIN
            .interpret_and_trace(
                |inputs| Ok(scalar_bilinear_sin(inputs)),
                (CpuArray::scalar(2.0), CpuArray::scalar(3.0)),
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
            CpuArray,
            ryft_core::programs::Program<
                CpuArray,
                ryft_core::backends::arrays::ArrayOperation<CpuArray>,
                CpuArray,
                CpuArray,
            >,
        ) = TEST_ARRAY_DOMAIN
            .interpret_and_trace(
                |x| {
                    let context = x.context().clone();
                    Ok(context.gradient(scalar_quartic_plus_sin, x).expect("scalar gradient should succeed"))
                },
                CpuArray::scalar(2.0),
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
            .add_instruction(
                SliceOperation::new(vec![1, 1], vec![2, 3]).with_strides(vec![1, 1]).unwrap(),
                Vec::new(),
                vec![input],
            )
            .unwrap()[0];
        let updated = builder
            .add_instruction(UpdateSliceOperation::new(vec![0, 1]), Vec::new(), vec![input, update])
            .unwrap()[0];
        let dynamic_sliced = builder
            .add_instruction(DynamicSliceOperation::new(vec![1, 2]), Vec::new(), vec![input, index_0, index_1])
            .unwrap()[0];
        let dynamic_updated = builder
            .add_instruction(DynamicUpdateSliceOperation, Vec::new(), vec![input, update, index_0, index_1])
            .unwrap()[0];
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

        // Static `sizes` make the result exact even when an operand axis is dynamic; no first-class result extent or
        // dynamic-slice-size operand is needed.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(5)).unwrap());
        let input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows), Dimension::Static(3)]));
        let index_type = ArrayType::scalar(DataType::I32);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let index_0 = builder.add_input(index_type.clone());
        let index_1 = builder.add_input(index_type);
        let output = builder
            .add_instruction(DynamicSliceOperation::new(vec![1, 2]), Vec::new(), vec![input, index_0, index_1])
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert!(stablehlo.contains("stablehlo.dynamic_slice"), "{stablehlo}");
        assert!(stablehlo.contains("sizes = [1, 2]"), "{stablehlo}");
        assert!(stablehlo.contains("-> tensor<1x2xf32>"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_concatenate() {
        // A static-shaped concatenate along axis 0 lowers to a single `stablehlo.concatenate` joining the operands.
        let first_type = test_matrix_type(1, 2);
        let second_type = test_matrix_type(3, 2);
        let mut builder = XlaProgramBuilder::new();
        let first = builder.add_input(first_type);
        let second = builder.add_input(second_type);
        let joined = builder
            .add_instruction(ConcatenateOperation::new(0, 2).unwrap(), Vec::new(), vec![first, second])
            .unwrap()[0];
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

        // Until P3 supplies the derived result extent as an explicit operand, dynamic concatenation fails before
        // lowering rather than manufacturing an unstable result identity.
        let dynamic_type =
            ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("rows", None), Dimension::Static(2)]));
        let mut builder = XlaProgramBuilder::new();
        let first = builder.add_input(dynamic_type.clone());
        let second = builder.add_input(dynamic_type);
        assert_eq!(
            builder.add_instruction(ConcatenateOperation::new(0, 2).unwrap(), Vec::new(), vec![first, second]),
            Err(ProgramError::Type(TypeError::invalid(
                "'concatenate' dynamic axis 0 requires an explicit result-dimension operand".to_string(),
            ))),
        );

        // Dynamic non-concatenated dimensions need transform residuals rather than a hidden runtime-size lookup in a
        // nominally static slice operation.
        let columns = DimensionVariable::new("columns", DimensionBounds::unbounded());
        let left_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), columns.clone().into()]));
        let right_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), columns.into()]));
        let mut builder = XlaProgramBuilder::new();
        let left = builder.add_input(left_type);
        let right = builder.add_input(right_type);
        let joined = builder
            .add_instruction(ConcatenateOperation::new(0, 2).unwrap(), Vec::new(), vec![left, right])
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, XlaConstant>(vec![joined], vec![Placeholder, Placeholder], Placeholder)
            .unwrap();
        assert_eq!(
            program.transpose().unwrap_err().to_string(),
            "'concatenate' transpose requires a static size on axis 1 but operand 0 has size columns",
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
            .add_instruction(
                SliceOperation::new(vec![1], vec![6]).with_strides(vec![2]).unwrap(),
                Vec::new(),
                vec![vector],
            )
            .unwrap()[0];
        let padded = builder
            .add_instruction(
                PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
                Vec::new(),
                vec![pad_input, padding_value],
            )
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
    fn test_to_mlir_module_for_plain_program_lowers_negative_and_mixed_pad() {
        let input_type = test_vector_type(5);
        let padding_value_type = ArrayType::scalar(DataType::F32);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let padding_value = builder.add_input(padding_value_type);
        let trimmed = builder
            .add_instruction(
                PadOperation::new(vec![-1], vec![-2], vec![0]).unwrap(),
                Vec::new(),
                vec![input, padding_value],
            )
            .unwrap()[0];
        let mixed = builder
            .add_instruction(
                PadOperation::new(vec![-1], vec![2], vec![2]).unwrap(),
                Vec::new(),
                vec![input, padding_value],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![trimmed, mixed],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<5xf32>, %arg1: tensor<f32>) -> (tensor<2xf32>, tensor<14xf32>) {
                    %0 = stablehlo.pad %arg0, %arg1, low = [-1], high = [-2], interior = [0] : (tensor<5xf32>, tensor<f32>) -> tensor<2xf32>
                    %1 = stablehlo.pad %arg0, %arg1, low = [-1], high = [2], interior = [2] : (tensor<5xf32>, tensor<f32>) -> tensor<14xf32>
                    return %0, %1 : tensor<2xf32>, tensor<14xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_preserves_dynamic_pad_bound() {
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("input", Some(5))]));
        let padding_value_type = ArrayType::scalar(DataType::F32);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let padding_value = builder.add_input(padding_value_type);
        assert_eq!(
            builder.add_instruction(
                PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
                Vec::new(),
                vec![input, padding_value],
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "'pad' dynamic axis 0 requires an explicit result-dimension operand".to_string(),
            ))),
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
        let (_, pullback): (CpuArray, _) = EagerContext::<CpuArray, ArrayOperation<CpuArray>>::new()
            .vjp(|x| Ok(x.slice(&[1], &[3], &[1]).unwrap()), CpuArray::vector(vec![1.0, 2.0, 3.0, 4.0]))
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
        let (_, pullback): (CpuArray, _) = EagerContext::<CpuArray, ArrayOperation<CpuArray>>::new()
            .vjp(|x| Ok(x.slice(&[1], &[6], &[2]).unwrap()), CpuArray::vector(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
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

        // The pad pullback first edge-unpads the cotangent and then slices it with the non-unit interior stride for the
        // input. It constructs a Boolean padding-position mask for the padding-value cotangent so non-finite values at
        // input positions cannot contaminate its sum.
        let (_, pullback): (CpuArray, _) = EagerContext::<CpuArray, ArrayOperation<CpuArray>>::new()
            .vjp(
                |(x, padding_value)| {
                    use ryft_core::operations::manipulation::Pad;
                    Ok(x.pad(&padding_value, &[1], &[2], &[1]).unwrap())
                },
                (CpuArray::vector(vec![1.0, 2.0, 3.0]), CpuArray::scalar(9.0)),
            )
            .unwrap();
        let (pullback, _residuals) = pullback.into_parts();
        let stablehlo = to_mlir_module_for_plain_program(&pullback, "main").unwrap();
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<8xf64>) -> (tensor<3xf64>, tensor<f64>) {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %0 = stablehlo.pad %arg0, %cst, low = [-1], high = [-2], interior = [0] : (tensor<8xf64>, tensor<f64>) -> tensor<5xf64>
                    %1 = stablehlo.slice %0 [0:5:2] : (tensor<5xf64>) -> tensor<3xf64>
                    %c = stablehlo.constant dense<false> : tensor<i1>
                    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<3xi1>
                    %c_0 = stablehlo.constant dense<true> : tensor<i1>
                    %3 = stablehlo.pad %2, %c_0, low = [1], high = [2], interior = [1] : (tensor<3xi1>, tensor<i1>) -> tensor<8xi1>
                    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %4 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<8xf64>
                    %5 = stablehlo.select %3, %arg0, %4 : tensor<8xi1>, tensor<8xf64>
                    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %6 = stablehlo.reduce(%5 init: %cst_2) applies stablehlo.add across dimensions = [0] : (tensor<8xf64>, tensor<f64>) -> tensor<f64>
                    return %1, %6 : tensor<3xf64>, tensor<f64>
                  }
                }
            "#}
        );

        // The dynamic slice pullback scatters the cotangent at the captured index factors, which materialize as
        // integer constants through `lower_literal_value`.
        let (_, pullback): (CpuArray, _) = EagerContext::<CpuArray, ArrayOperation<CpuArray>>::new()
            .vjp(
                |x| {
                    let start =
                        x.context().lift(CpuArray::from_f64s(ArrayType::scalar(DataType::I32), vec![1.0])).unwrap();
                    Ok(x.dynamic_slice(&[start], &[2]).unwrap())
                },
                CpuArray::vector(vec![1.0, 2.0, 3.0, 4.0]),
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
        let (_, pullback): (CpuArray, _) = EagerContext::<CpuArray, ArrayOperation<CpuArray>>::new()
            .vjp(|inputs| Ok(scalar_bilinear_sin(inputs)), (CpuArray::scalar(2.0), CpuArray::scalar(3.0)))
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

        let function = rematerialize::<EagerContext<CpuArray, ArrayOperation<CpuArray>>, _, _, _>(
            |x: ryft_core::tracing::DomainTracer<EagerContext<CpuArray, ArrayOperation<CpuArray>>>| {
                Ok((x.clone() * x).sin()?)
            },
        );
        let (_, pullback): (CpuArray, _) = EagerContext::<CpuArray, ArrayOperation<CpuArray>>::new()
            .vjp(|x| function.call(x), CpuArray::scalar(2.0))
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
        let function = rematerialize::<EagerContext<CpuArray, ArrayOperation<CpuArray>>, _, _, _>(
            |x: ryft_core::tracing::DomainTracer<EagerContext<CpuArray, ArrayOperation<CpuArray>>>| {
                Ok((x.clone() * x).sin()?)
            },
        )
        .with_prevent_cse(false);
        let (_, pullback): (CpuArray, _) = EagerContext::<CpuArray, ArrayOperation<CpuArray>>::new()
            .vjp(|x| function.call(x), CpuArray::scalar(2.0))
            .unwrap();
        let (pullback, _residuals) = pullback.into_parts();
        let stablehlo = to_mlir_module_for_plain_program(&pullback, "main").unwrap();
        assert_eq!(stablehlo, expected, "the reverse pullback is independent of the prevent_cse hint");
    }

    #[test]
    fn test_transfer_to_memory_lowers_to_device_placement_annotations() {
        use ryft_core::operations::memory::TransferToMemory;

        // A compute-flanked host-and-back round trip lowers to one `annotate_device_placement` custom call per
        // transfer, carrying the destination kind in the `_xla_buffer_placement` frontend attribute — including the
        // identity-looking transfer back to device memory, which `HostOffloader` needs to see. The program mirrors
        // the JAX example in `python/scripts/dump_transfer_to_memory_mlir_from_jax.py`, and the asserted custom
        // calls are byte-identical to the ones JAX emits for it.
        let (_, program) = EagerContext::<CpuArray, ArrayOperation<CpuArray>>::trace(
            |x: ryft_core::tracing::DomainTracer<EagerContext<CpuArray, ArrayOperation<CpuArray>>>| {
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
    fn test_synthesized_constant_output_lowers_with_its_memory_placement() {
        for (memory, placement) in [
            (Memory::Device, None),
            (Memory::Host { pinned: true }, Some("pinned_host")),
            (Memory::Host { pinned: false }, Some("unpinned_host")),
        ] {
            let output_type = test_vector_type(4).with_memory(memory);
            let mut builder = XlaProgramBuilder::new();
            let output = builder.add_instruction(ZeroOperation::new(output_type), Vec::new(), Vec::new()).unwrap()[0];
            let program = builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], Vec::new(), vec![Placeholder])
                .unwrap();

            let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
            assert_eq!(
                stablehlo.matches("stablehlo.custom_call @annotate_device_placement").count(),
                usize::from(placement.is_some()),
                "{stablehlo}",
            );
            if let Some(placement) = placement {
                assert!(stablehlo.contains(&format!("_xla_buffer_placement = \"{placement}\"")), "{stablehlo}");
            }
        }
    }

    #[test]
    fn test_zero_space_type_and_constant_lower_to_a_false_i1_carrier() {
        let output_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = XlaProgramBuilder::new();
        let output = builder.add_instruction(ZeroOperation::new(output_type), Vec::new(), Vec::new()).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], Vec::new(), vec![Placeholder])
            .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert!(stablehlo.contains("func.func @main() -> tensor<3xi1>"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.constant dense<false> : tensor<i1>"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.broadcast_in_dim"), "{stablehlo}");
    }

    #[test]
    fn test_zero_space_one_like_lowers_to_the_unique_zero_value() {
        let input_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let output = builder.add_instruction(OneLikeOperation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert!(stablehlo.contains("func.func @main(%arg0: tensor<3xi1>) -> tensor<3xi1>"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.constant dense<false> : tensor<i1>"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.broadcast_in_dim"), "{stablehlo}");
    }

    #[test]
    fn test_xla_executable_signature_erases_only_static_zero_space_leaves() {
        let static_zero = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let dynamic_zero = ArrayType::new(DataType::Zero, Shape::new(vec![dynamic_dimension("dynamic_zero", Some(3))]));
        let boolean = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(3)]));
        let signature = XlaExecutableSignature::new(
            &[static_zero.clone(), boolean.clone(), dynamic_zero.clone()],
            &[dynamic_zero, static_zero],
        );

        assert_eq!(signature.input_mapping(), &[None, Some(0), Some(1)]);
        assert_eq!(signature.output_mapping(), &[Some(0), None]);
        assert_eq!(signature.project_inputs(&[0, 1, 2]), vec![1, 2]);
        assert_eq!(signature.project_outputs(&[0, 1]), vec![0]);
    }

    #[test]
    fn test_to_mlir_module_for_program_erases_static_zero_space_boundary() {
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(zero_type.clone());
        let program: FlatXlaProgram = builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();

        let stablehlo = to_mlir_module_for_program(&program, &[], &zero_type, &zero_type, "main", None, None).unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main() {
                    %c = stablehlo.constant dense<false> : tensor<i1>
                    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<3xi1>
                    return
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_preserves_effects_that_produce_an_erased_output() {
        use ryft_core::operations::debugging::PrintOperation;

        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(zero_type.clone());
        let output = builder.add_instruction(PrintOperation::new("zero"), Vec::new(), vec![input]).unwrap()[0];
        let program: FlatXlaProgram = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        let stablehlo = to_mlir_module_for_program(&program, &[], &zero_type, &zero_type, "main", None, None).unwrap();

        assert!(stablehlo.contains("stablehlo.custom_call @ryft.print"), "{stablehlo}");
        assert!(stablehlo.contains("has_side_effect = true"), "{stablehlo}");
        assert!(stablehlo.contains("return\n"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_preserves_dynamic_zero_space_shape_carrier() {
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![dynamic_dimension("zero", Some(3))]));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(zero_type.clone());
        let program: FlatXlaProgram = builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();

        let stablehlo = to_mlir_module_for_program(&program, &[], &zero_type, &zero_type, "main", None, None).unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<?xi1, #stablehlo.bounds<2>>) -> tensor<?xi1, #stablehlo.bounds<2>> {
                    return %arg0 : tensor<?xi1, #stablehlo.bounds<2>>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_literal_constant_operation_lowers_with_its_memory_placement() {
        for (memory, placement) in [
            (Memory::Device, None),
            (Memory::Host { pinned: true }, Some("pinned_host")),
            (Memory::Host { pinned: false }, Some("unpinned_host")),
        ] {
            let value = CpuArray::from_f64s(test_vector_type(4).with_memory(memory), vec![1.0, 2.0, 3.0, 4.0]);
            let mut builder = ProgramBuilder::<CpuArray, ArrayOperation<CpuArray>>::new();
            let output = builder
                .add_instruction(ArrayOperation::Constant(ConstantOperation::new(value)), Vec::new(), Vec::new())
                .unwrap()[0];
            let program =
                builder.build::<Vec<CpuArray>, Vec<CpuArray>>(vec![output], Vec::new(), vec![Placeholder]).unwrap();

            let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
            assert_eq!(
                stablehlo.matches("stablehlo.custom_call @annotate_device_placement").count(),
                usize::from(placement.is_some()),
                "{stablehlo}",
            );
            if let Some(placement) = placement {
                assert!(stablehlo.contains(&format!("_xla_buffer_placement = \"{placement}\"")), "{stablehlo}");
            }
        }
    }

    #[test]
    fn test_fill_lowers_as_scalar_constant_plus_broadcast() {
        for (memory, placement) in [
            (Memory::Device, None),
            (Memory::Host { pinned: true }, Some("pinned_host")),
            (Memory::Host { pinned: false }, Some("unpinned_host")),
        ] {
            let context = TracingContext::<CpuArray, ArrayOperation<CpuArray>>::new();
            let output_type = test_vector_type(4).with_memory(memory);
            let output = context.fill(&output_type, Scalar::F64(2.5)).unwrap();
            assert_eq!(*output.r#type(), output_type);
            let program = context
                .builder()
                .borrow()
                .clone()
                .build::<Vec<CpuArray>, Vec<CpuArray>>(vec![output.atom_id().unwrap()], Vec::new(), vec![Placeholder])
                .unwrap();

            let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
            assert_eq!(stablehlo.matches("stablehlo.constant").count(), 1, "{stablehlo}");
            assert_eq!(stablehlo.matches("stablehlo.broadcast_in_dim").count(), 1, "{stablehlo}");
            assert!(stablehlo.contains("stablehlo.constant dense<2.500000e+00> : tensor<f32>"), "{stablehlo}");
            assert_eq!(
                stablehlo.matches("stablehlo.custom_call @annotate_device_placement").count(),
                usize::from(placement.is_some()),
                "{stablehlo}",
            );
            if let Some(placement) = placement {
                assert!(stablehlo.contains(&format!("_xla_buffer_placement = \"{placement}\"")), "{stablehlo}");
            }

            let context = TracingContext::<CpuArray, ArrayOperation<CpuArray>>::new();
            let output_type = ArrayType::scalar(DataType::F32).with_memory(memory);
            let output = context.fill(&output_type, Scalar::F64(2.5)).unwrap();
            assert_eq!(*output.r#type(), output_type);
            let program = context
                .builder()
                .borrow()
                .clone()
                .build::<Vec<CpuArray>, Vec<CpuArray>>(vec![output.atom_id().unwrap()], Vec::new(), vec![Placeholder])
                .unwrap();
            let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
            assert_eq!(stablehlo.matches("stablehlo.constant").count(), 1, "{stablehlo}");
            assert_eq!(stablehlo.matches("stablehlo.broadcast_in_dim").count(), 0, "{stablehlo}");
            assert_eq!(
                stablehlo.matches("stablehlo.custom_call @annotate_device_placement").count(),
                usize::from(placement.is_some()),
                "{stablehlo}",
            );
            if let Some(placement) = placement {
                assert!(stablehlo.contains(&format!("_xla_buffer_placement = \"{placement}\"")), "{stablehlo}");
            }
        }
    }

    #[test]
    fn test_rank_positive_literal_constant_preserves_signed_zero_elements() {
        let context = TracingContext::<CpuArray, ArrayOperation<CpuArray>>::new();
        let literal = CpuArray::new(test_vector_type(2), vec![Scalar::from(-0.0_f32), Scalar::from(0.0_f32)]).unwrap();
        let output = context.bind(ConstantOperation::new(literal), Vec::new(), &[]).unwrap().remove(0);
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<CpuArray>, Vec<CpuArray>>(vec![output.atom_id().unwrap()], Vec::new(), vec![Placeholder])
            .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert_eq!(stablehlo.matches("stablehlo.constant").count(), 1, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.broadcast_in_dim").count(), 0, "{stablehlo}");
        assert!(stablehlo.contains("dense<[-0.000000e+00, 0.000000e+00]>"), "{stablehlo}");

        // A rank-positive one-element literal remains an ordinary shaped dense constant rather than being rewritten
        // as a rank-zero constant plus broadcast.
        let literal = CpuArray::new(test_vector_type(1), vec![Scalar::from(2.5_f32)]).unwrap();
        let mut builder = ProgramBuilder::<CpuArray, ArrayOperation<CpuArray>>::new();
        let output = builder
            .add_instruction(ArrayOperation::Constant(ConstantOperation::new(literal)), Vec::new(), Vec::new())
            .unwrap()[0];
        let program =
            builder.build::<Vec<CpuArray>, Vec<CpuArray>>(vec![output], Vec::new(), vec![Placeholder]).unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert_eq!(stablehlo.matches("stablehlo.constant").count(), 1, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.broadcast_in_dim").count(), 0, "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.constant dense<2.500000e+00> : tensor<1xf32>"), "{stablehlo}");
    }

    #[test]
    fn test_rank_positive_literal_dense_attributes_preserve_exact_payloads() {
        // Boolean values retain their logical order through MLIR's bit-packed dense representation.
        let boolean = test_literal(vec![Scalar::Bool(true), Scalar::Bool(false), Scalar::Bool(true)]);
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = lower_tensor_type(boolean.r#type().as_ref(), &context, location).unwrap();
        let attribute = boolean.to_dense_elements_attribute(tensor_type, &context).unwrap();
        assert_eq!(
            unsafe { attribute.bool_elements().collect::<Result<Vec<_>, _>>().unwrap() },
            vec![true, false, true],
        );

        // Every byte-aligned integer family preserves signedness, magnitude, and source order without floating-point
        // conversion. The `u64` case deliberately exceeds f64's exact-integer range.
        let integer_cases = vec![
            (test_literal(vec![Scalar::I8(-127), Scalar::I8(126)]), values_to_bytes(&[-127_i8, 126])),
            (test_literal(vec![Scalar::I16(-0x1234), Scalar::I16(0x2345)]), values_to_bytes(&[-0x1234_i16, 0x2345])),
            (
                test_literal(vec![Scalar::I32(-0x1234_567), Scalar::I32(0x2345_678)]),
                values_to_bytes(&[-0x1234_567_i32, 0x2345_678]),
            ),
            (
                test_literal(vec![Scalar::I64(-0x1234_5678_9abc_def), Scalar::I64(0x2345_6789_abcd_ef0)]),
                values_to_bytes(&[-0x1234_5678_9abc_def_i64, 0x2345_6789_abcd_ef0]),
            ),
            (test_literal(vec![Scalar::U8(0x12), Scalar::U8(0xfe)]), values_to_bytes(&[0x12_u8, 0xfe])),
            (test_literal(vec![Scalar::U16(0x1234), Scalar::U16(0xfedc)]), values_to_bytes(&[0x1234_u16, 0xfedc])),
            (
                test_literal(vec![Scalar::U32(0x1234_5678), Scalar::U32(0xfedc_ba98)]),
                values_to_bytes(&[0x1234_5678_u32, 0xfedc_ba98]),
            ),
            (
                test_literal(vec![Scalar::U64((1_u64 << 53) + 1), Scalar::U64(u64::MAX - 1)]),
                values_to_bytes(&[(1_u64 << 53) + 1, u64::MAX - 1]),
            ),
        ];
        for (literal, expected) in integer_cases {
            assert_eq!(
                test_literal_dense_bytes(&literal, expected.len()),
                expected,
                "integer literal type {}",
                literal.r#type(),
            );
        }

        // Sub-byte and eight-bit floating-point formats retain their exact encodings, including NaN payloads.
        for (data_type, bits) in [
            (DataType::F4E2M1FN, [0x01, 0x0f]),
            (DataType::F6E2M3FN, [0x01, 0x3f]),
            (DataType::F6E3M2FN, [0x02, 0x3e]),
            (DataType::F8E3M4, [0x01, 0xff]),
            (DataType::F8E4M3, [0x02, 0xfe]),
            (DataType::F8E4M3FN, [0x03, 0x7f]),
            (DataType::F8E4M3FNUZ, [0x04, 0x80]),
            (DataType::F8E4M3B11FNUZ, [0x05, 0x80]),
            (DataType::F8E5M2, [0x06, 0x7f]),
            (DataType::F8E5M2FNUZ, [0x07, 0x80]),
            (DataType::F8E8M0FNU, [0x08, 0xff]),
        ] {
            let literal = test_literal(
                bits.into_iter()
                    .map(|bits| Scalar::from_low_precision_float_bits(data_type, bits).unwrap())
                    .collect(),
            );
            assert_eq!(test_literal_dense_bytes(&literal, bits.len()), bits, "low-precision literal type {data_type}");
            let mut builder = ProgramBuilder::<CpuArray, ArrayOperation<CpuArray>>::new();
            let output = builder
                .add_instruction(ArrayOperation::Constant(ConstantOperation::new(literal)), Vec::new(), Vec::new())
                .unwrap()[0];
            let program =
                builder.build::<Vec<CpuArray>, Vec<CpuArray>>(vec![output], Vec::new(), vec![Placeholder]).unwrap();
            assert!(to_mlir_module_for_plain_program(&program, "main").is_ok(), "low-precision literal {data_type}");
        }

        // Standard floating-point families preserve signed zero, infinities, and NaN payload bits.
        let bf16_values = [half::bf16::from_bits(0x8000), half::bf16::from_bits(0x7fc1)];
        let f16_values = [half::f16::from_bits(0x8000), half::f16::from_bits(0x7e01)];
        let f32_values = [f32::from_bits(0x8000_0000), f32::INFINITY, f32::from_bits(0x7fc0_1234)];
        let f64_values =
            [f64::from_bits(0x8000_0000_0000_0000), f64::NEG_INFINITY, f64::from_bits(0x7ff8_0000_0000_1234)];
        for (literal, expected) in [
            (test_literal(bf16_values.into_iter().map(Scalar::BF16).collect()), values_to_bytes(&bf16_values)),
            (test_literal(f16_values.into_iter().map(Scalar::F16).collect()), values_to_bytes(&f16_values)),
            (test_literal(f32_values.into_iter().map(Scalar::F32).collect()), values_to_bytes(&f32_values)),
            (test_literal(f64_values.into_iter().map(Scalar::F64).collect()), values_to_bytes(&f64_values)),
        ] {
            assert_eq!(
                test_literal_dense_bytes(&literal, expected.len()),
                expected,
                "floating-point literal type {}",
                literal.r#type(),
            );
        }

        // Complex storage interleaves independently exact real and imaginary components in source order.
        let c64_components =
            [f32::from_bits(0x8000_0000), f32::from_bits(0x7fc0_1234), f32::INFINITY, f32::NEG_INFINITY];
        let c64 = test_literal(vec![
            Scalar::C64(num_complex::Complex::new(c64_components[0], c64_components[1])),
            Scalar::C64(num_complex::Complex::new(c64_components[2], c64_components[3])),
        ]);
        assert_eq!(test_literal_dense_bytes(&c64, size_of_val(&c64_components)), values_to_bytes(&c64_components),);
        let c128_components = [
            f64::from_bits(0x8000_0000_0000_0000),
            f64::from_bits(0x7ff8_0000_0000_1234),
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        let c128 = test_literal(vec![
            Scalar::C128(num_complex::Complex::new(c128_components[0], c128_components[1])),
            Scalar::C128(num_complex::Complex::new(c128_components[2], c128_components[3])),
        ]);
        assert_eq!(test_literal_dense_bytes(&c128, size_of_val(&c128_components)), values_to_bytes(&c128_components),);

        // Payload-free logical types never enter raw construction and report the standard structured lowering error.
        let tensor_type = context
            .tensor_type(context.signless_integer_type(1), &[MlirSize::Static(2)], None, location)
            .unwrap();
        for (data_type, values) in
            [(DataType::Zero, vec![Scalar::Zero, Scalar::Zero]), (DataType::Token, vec![Scalar::Token, Scalar::Token])]
        {
            let literal =
                CpuArray::new(ArrayType::new(data_type, Shape::new(vec![Dimension::Static(2)])), values).unwrap();
            assert_eq!(
                literal.to_dense_elements_attribute(tensor_type, &context),
                Err(LoweringError::UnsupportedDataType { data_type }),
            );
        }
    }

    #[test]
    fn test_rank_positive_literal_constants_execute_exactly_on_cpu() {
        use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
        use ryft_pjrt::{
            ClientOptions, CpuClientOptions, ExecutionDeviceInputs, Program as PjrtProgram, load_cpu_plugin,
        };

        let bf16_values = [half::bf16::from_bits(0x3f80), half::bf16::from_bits(0xc000)];
        let f16_values = [half::f16::from_bits(0x3c00), half::f16::from_bits(0xc000)];
        let f32_values = [f32::from_bits(0x8000_0000), f32::INFINITY];
        let f64_values = [f64::from_bits(0x8000_0000_0000_0000), f64::NEG_INFINITY];
        let c64_components = [1.5_f32, -2.0, f32::from_bits(0x8000_0000), f32::INFINITY];
        let c128_components = [1.5_f64, -2.0, f64::from_bits(0x8000_0000_0000_0000), f64::NEG_INFINITY];
        let cases = vec![
            (test_literal(vec![Scalar::Bool(true), Scalar::Bool(false), Scalar::Bool(true)]), vec![1_u8, 0, 1]),
            (test_literal(vec![Scalar::I16(-0x1234), Scalar::I16(0x2345)]), values_to_bytes(&[-0x1234_i16, 0x2345])),
            (
                test_literal(vec![Scalar::U64((1_u64 << 53) + 1), Scalar::U64(u64::MAX - 1)]),
                values_to_bytes(&[(1_u64 << 53) + 1, u64::MAX - 1]),
            ),
            (test_literal(bf16_values.into_iter().map(Scalar::BF16).collect()), values_to_bytes(&bf16_values)),
            (test_literal(f16_values.into_iter().map(Scalar::F16).collect()), values_to_bytes(&f16_values)),
            (test_literal(f32_values.into_iter().map(Scalar::F32).collect()), values_to_bytes(&f32_values)),
            (test_literal(f64_values.into_iter().map(Scalar::F64).collect()), values_to_bytes(&f64_values)),
            (
                test_literal(vec![
                    Scalar::C64(num_complex::Complex::new(c64_components[0], c64_components[1])),
                    Scalar::C64(num_complex::Complex::new(c64_components[2], c64_components[3])),
                ]),
                values_to_bytes(&c64_components),
            ),
            (
                test_literal(vec![
                    Scalar::C128(num_complex::Complex::new(c128_components[0], c128_components[1])),
                    Scalar::C128(num_complex::Complex::new(c128_components[2], c128_components[3])),
                ]),
                values_to_bytes(&c128_components),
            ),
        ];

        let mut builder = ProgramBuilder::<CpuArray, ArrayOperation<CpuArray>>::new();
        let outputs = cases
            .iter()
            .map(|(literal, _)| {
                builder
                    .add_instruction(
                        ArrayOperation::Constant(ConstantOperation::new(literal.clone())),
                        Vec::new(),
                        Vec::new(),
                    )
                    .unwrap()[0]
            })
            .collect::<Vec<_>>();
        let program = builder
            .build::<Vec<CpuArray>, Vec<CpuArray>>(outputs, Vec::new(), vec![Placeholder; cases.len()])
            .unwrap();
        let module = to_mlir_module_for_plain_program(&program, "main").unwrap();

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let executable = client
            .compile(
                &PjrtProgram::Mlir { bytecode: module.into_bytes() },
                &CompilationOptions {
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
                },
            )
            .unwrap();
        let mut executions = executable
            .execute(vec![ExecutionDeviceInputs::default()], Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();
        let outputs = executions.remove(0).outputs;
        assert_eq!(outputs.len(), cases.len());
        for (output, (literal, expected)) in outputs.into_iter().zip(cases) {
            let actual = output.copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(actual, expected, "executed literal type {}", literal.r#type());
        }
    }

    #[test]
    fn test_literal_constant_atom_lowers_with_its_memory_placement() {
        for (memory, placement) in [
            (Memory::Device, None),
            (Memory::Host { pinned: true }, Some("pinned_host")),
            (Memory::Host { pinned: false }, Some("unpinned_host")),
        ] {
            let value = CpuArray::from_f64s(test_vector_type(4).with_memory(memory), vec![1.0, 2.0, 3.0, 4.0]);
            let mut builder = ProgramBuilder::<CpuArray, ArrayOperation<CpuArray>>::new();
            let output = builder.add_constant(value);
            let program =
                builder.build::<Vec<CpuArray>, Vec<CpuArray>>(vec![output], Vec::new(), vec![Placeholder]).unwrap();

            let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
            assert_eq!(
                stablehlo.matches("stablehlo.custom_call @annotate_device_placement").count(),
                usize::from(placement.is_some()),
                "{stablehlo}",
            );
            if let Some(placement) = placement {
                assert!(stablehlo.contains(&format!("_xla_buffer_placement = \"{placement}\"")), "{stablehlo}");
            }
        }
    }

    #[test]
    fn test_transfer_to_memory_vjp_pullback_lowers_with_a_placement_annotation() {
        use ryft_core::operations::memory::TransferToMemory;

        // The pullback of a transfer moves the cotangent back to the operand's source memory (the default device
        // space here), so it lowers to an `annotate_device_placement` custom call targeting `device`.
        let (_, pullback): (CpuArray, _) = EagerContext::<CpuArray, ArrayOperation<CpuArray>>::new()
            .vjp(|x| Ok(x.transfer_to_memory(Memory::Host { pinned: true })), CpuArray::scalar(2.0))
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
            (CpuArray, CpuArray),
            ryft_core::programs::Program<
                CpuArray,
                ryft_core::backends::arrays::ArrayOperation<CpuArray>,
                (CpuArray, CpuArray),
                (CpuArray, CpuArray),
            >,
        ) = TEST_ARRAY_DOMAIN
            .interpret_and_trace(
                |inputs| {
                    let context = inputs.0.context().clone();
                    Ok(context.gradient(scalar_bilinear_sin, inputs).expect("scalar gradient should succeed"))
                },
                (CpuArray::scalar(2.0), CpuArray::scalar(3.0)),
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
