use std::borrow::Cow;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use ryft_core::differentiation::operations::{CUSTOM_JVP_OPERATION_NAME, CUSTOM_VJP_OPERATION_NAME};
use ryft_core::macros::check_count;
use ryft_core::operations::attention::{
    AttentionInputs, DotProductAttentionBackwardOperation, DotProductAttentionOperation,
    dot_product_attention_backward_ir_composition, dot_product_attention_ir_composition,
};
use ryft_core::operations::collectives::{
    AllGatherOperation, AllToAllOperation, CollectiveMode, PSumScatterOperation, PpermuteOperation,
};
use ryft_core::operations::complex::{ComplexOperation, ConjugateOperation, ImaginaryOperation, RealOperation};
use ryft_core::operations::custom_call::{CUSTOM_CALL_OPERATION_NAME, CustomCallAttribute, CustomCallOperation};
use ryft_core::operations::dot::{lhs_result_axes, rhs_result_axes};
use ryft_core::operations::quantization::scaled_dot_ir_composition;
use ryft_core::operations::random::{RandomAlgorithm, RngBitGeneratorOperation};
use ryft_core::operations::sort::{SORT_OPERATION_NAME, SortDirection, SortOperation};
use ryft_core::{
    AXIS_INDEX_OPERATION_NAME, AbsOperation, AddOperation, Array as CpuArray, ArrayIrType, ArrayOperation, ArrayType,
    Atan2Operation, AtomId, AxisIndexOperation, BroadcastOperation, CONDITION_OPERATION_NAME, CaptureReference,
    CeilOperation, CollectiveKind, CollectiveOperation, ComparisonDirection, ConstantOperation,
    ConvertElementTypeOperation, CosOperation, DataType, Dimension, DimensionOperation, DimensionRequirementOperation,
    DimensionRequirementPredicate, DimensionType, DimensionValue, DivOperation, DomainTracingContext, DotOperation,
    Effect, Effects, ErfOperation, ExpOperation, FloorOperation, GATHER_OPERATION_NAME, GatherOperation,
    GatherScatterMode, Instruction, IotaOperation, Layout, LogOperation, LogicalMesh, LogisticOperation,
    MAX_DIMENSION_EXTENT, MaxOperation, Memory, MeshAxisType, MinOperation, MulOperation, NegOperation, Operation,
    PadOperation, Parameterized, PowOperation, Program, ProgramError, ProjectedValue, Provenance,
    REMATERIALIZE_OPERATION_NAME, ReductionKind, ReferenceStateBinding, RegionId, RegionRef, RemOperation,
    ReshapeOperation, RoundOperation, RsqrtOperation, SCAN_OPERATION_NAME, SCATTER_OPERATION_NAME, ScaledDotOperation,
    ScanOperation, ScatterOperation, ScatterReductionKind, Shape, Sharding, ShardingDimension, ShardingError,
    SignOperation, SinOperation, SliceOperation, SqrtOperation, SubOperation, TanhOperation, TransposeOperation,
    Type as RyftType, Typed, Value, WHILE_OPERATION_NAME, WhileOperation,
};
#[cfg(test)]
use ryft_core::{Complex as ComplexNumber, ReshapeParameters};
use ryft_mlir::dialects::stable_hlo::{Accuracy, CustomCallApiVersion, CustomCallMemoryLayouts, Precision};
use ryft_mlir::dialects::{chlo, func, shardy, stable_hlo, tensor};
use ryft_mlir::{
    Attribute, AttributeRef, Block, BlockRef, Context as MlirContext, DenseElementsAttributeRef,
    DictionaryAttributeRef, FloatTypeRef, IntegerTypeRef, Location, LocationRef, Module, Operation as MlirOperation,
    OperationPrintingFlags, Region, Size as MlirSize, StringRef, SymbolVisibility, TensorTypeRef, Type,
    TypeAndAttributes, TypeRef, Value as MlirValue, ValueAndAttributes, ValueRef,
};

use crate::ToMlir;
use crate::experimental::assertions::{
    ASSERT_ACTOR_ATTRIBUTE, ASSERT_ADD_KIND, ASSERT_BOUNDS_KIND, ASSERT_CONCATENATE_KIND, ASSERT_CUSTOM_CALL_TARGET,
    ASSERT_DETAIL_ATTRIBUTE, ASSERT_DIV_FLOOR_KIND, ASSERT_DIVISIBLE_BY_KIND, ASSERT_DYNAMIC_SHAPE_SLICE_KIND,
    ASSERT_EQUAL_KIND, ASSERT_KIND_ATTRIBUTE, ASSERT_LEFT_ATTRIBUTE, ASSERT_LESS_THAN_OR_EQUAL_KIND, ASSERT_MUL_KIND,
    ASSERT_POW_KIND, ASSERT_REM_KIND, ASSERT_RIGHT_ATTRIBUTE, ASSERT_SUB_KIND,
};
use crate::experimental::debugging::{PRINT_CUSTOM_CALL_TARGET, PRINT_LABEL_ATTRIBUTE};
use crate::experimental::domains::{XlaDomain, XlaTracer};
#[cfg(test)]
use crate::experimental::lowering::attention::attention_array_type;
use crate::experimental::lowering::attention::{
    lower_dot_product_attention_backward_to_mlir, lower_dot_product_attention_to_mlir,
};
use crate::experimental::ops::{FlatXlaProgram, XlaArrayConstant, XlaConstant, XlaOperation, XlaProgram};

use crate::experimental::operations::SHARD_MAP_OPERATION_NAME;

use super::shard_map::{ShardMap, ShardMapError};

mod attention;
mod composite;

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
    #[error("invalid function name `{function_name}` used during XLA lowering")]
    InvalidFunctionName { function_name: String },

    /// Error returned when lowering encounters a traced tensor type that MLIR rejects.
    #[error("invalid tensor type `{array_type}` used during XLA lowering")]
    InvalidTensorType { array_type: ArrayType },

    /// Error returned when a reshape dimension cannot be represented by StableHLO's signed shape element type.
    #[error("reshape dimension {value} cannot be represented as a StableHLO i{bit_width} shape value")]
    ReshapeDimensionOutOfRange { value: usize, bit_width: u8 },

    /// Error returned when a pad interior amount cannot be represented by StableHLO's signed attribute type.
    #[error("pad interior padding {value} cannot be represented as a StableHLO i64 attribute")]
    PadInteriorPaddingOutOfRange { value: usize },

    /// Error returned when lowering encounters a staged op that does not yet have StableHLO support.
    #[error("unsupported staged op `{op}` during XLA lowering")]
    UnsupportedOp { op: String },

    /// Error returned when unresolved reference semantics reach XLA lowering before functionalization.
    #[error("unresolved reference construct `{construct}` must be discharged before XLA lowering")]
    UnresolvedReference { construct: String },

    /// Error returned when unresolved mutable state reaches ordinary XLA lowering before functionalization.
    #[error("unresolved state in `{construct}` must be discharged before XLA lowering")]
    UnresolvedState { construct: String },

    /// Error returned when a shard-map body carries ordered effects, whose tokens `sdy.manual_computation` cannot
    /// thread across its boundary.
    #[error(
        "effectful shard_map bodies are unsupported because sdy.manual_computation cannot preserve effect \
             ordering across its boundary"
    )]
    EffectfulShardMapBody,

    /// Error returned when lowering encounters a captured constant reference without a matching hidden argument.
    #[error("missing captured constant #{index} during XLA lowering")]
    MissingCapturedConstant { index: usize },

    /// Error returned when lowering tries to materialize abstract XLA type metadata as a literal value.
    #[error("abstract XLA value `{array_type}` cannot be materialized as a StableHLO literal")]
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
    #[error("unsupported data type `{data_type}` during XLA lowering")]
    UnsupportedDataType { data_type: DataType },

    /// Error returned when MLIR rejects the constructed dense-elements attribute.
    #[error("invalid dense elements attribute for data type `{data_type}` during XLA lowering")]
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

    /// Error returned when logical reference-state metadata cannot map to a safe executable alias.
    #[error("invalid XLA reference-state ABI: {message}")]
    InvalidReferenceStateAbi { message: String },

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

    /// Hidden scalar arguments that restore bounded dynamic dimensions on physicalized input tensors.
    input_dimensions: Arc<[XlaInputDimension]>,

    /// Hidden scalar results that report bounded dynamic logical output extents.
    output_dimensions: Arc<[XlaOutputDimension]>,
}

/// One bounded dynamic input axis transported as a hidden scalar executable argument.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct XlaInputDimension {
    /// Logical flattened input containing the dynamic axis.
    logical_input_index: usize,

    /// Axis within the logical input.
    axis: usize,

    /// Physical executable argument containing the runtime axis extent.
    physical_input_index: usize,
}

impl XlaInputDimension {
    /// Returns the logical flattened input containing this dynamic axis.
    #[inline]
    pub(crate) fn logical_input_index(self) -> usize {
        self.logical_input_index
    }

    /// Returns the dynamic axis within the logical input.
    #[inline]
    pub(crate) fn axis(self) -> usize {
        self.axis
    }

    /// Returns the physical executable argument containing the runtime extent.
    #[inline]
    pub(crate) fn physical_input_index(self) -> usize {
        self.physical_input_index
    }
}

/// One bounded dynamic output axis transported as a hidden scalar executable result.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct XlaOutputDimension {
    /// Logical flattened output containing the dynamic axis.
    logical_output_index: usize,

    /// Axis within the logical output.
    axis: usize,

    /// Physical executable result containing the runtime axis extent.
    physical_output_index: usize,
}

impl XlaOutputDimension {
    /// Returns the logical flattened output containing this dynamic axis.
    #[inline]
    pub(crate) fn logical_output_index(self) -> usize {
        self.logical_output_index
    }

    /// Returns the dynamic axis within the logical output.
    #[inline]
    pub(crate) fn axis(self) -> usize {
        self.axis
    }

    /// Returns the physical executable result containing the runtime extent.
    #[inline]
    pub(crate) fn physical_output_index(self) -> usize {
        self.physical_output_index
    }
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
        let input_mapping = mapping(input_types);
        let mut physical_input_index = input_mapping.iter().flatten().count();
        let mut input_dimensions = Vec::new();
        for (logical_input_index, input_type) in input_types.iter().enumerate() {
            for (axis, dimension) in input_type.shape().dimensions().iter().enumerate() {
                if dimension.value().is_none() && dimension.upper_bound().is_some() {
                    input_dimensions.push(XlaInputDimension { logical_input_index, axis, physical_input_index });
                    physical_input_index += 1;
                }
            }
        }
        let output_mapping = mapping(output_types);
        let mut physical_output_index = output_mapping.iter().flatten().count();
        let mut output_dimensions = Vec::new();
        for (logical_output_index, output_type) in output_types.iter().enumerate() {
            for (axis, dimension) in output_type.shape().dimensions().iter().enumerate() {
                if dimension.value().is_none() && dimension.upper_bound().is_some() {
                    output_dimensions.push(XlaOutputDimension { logical_output_index, axis, physical_output_index });
                    physical_output_index += 1;
                }
            }
        }
        Self {
            input_mapping,
            output_mapping,
            input_dimensions: input_dimensions.into(),
            output_dimensions: output_dimensions.into(),
        }
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

    /// Returns hidden bounded-dimension scalar arguments in physical executable order.
    #[inline]
    pub(crate) fn input_dimensions(&self) -> &[XlaInputDimension] {
        &self.input_dimensions
    }

    /// Returns hidden bounded dynamic-output extent results in physical executable order.
    #[inline]
    pub(crate) fn output_dimensions(&self) -> &[XlaOutputDimension] {
        &self.output_dimensions
    }

    /// Returns the total number of physical executable arguments.
    #[inline]
    pub(crate) fn physical_input_count(&self) -> usize {
        self.input_mapping.iter().flatten().count() + self.input_dimensions.len()
    }

    /// Projects logical inputs into physical executable order.
    pub(crate) fn project_inputs<T: Clone>(&self, inputs: &[T]) -> Vec<T> {
        Self::project(&self.input_mapping, inputs)
    }

    /// Projects logical outputs into physical executable order.
    pub(crate) fn project_outputs<T: Clone>(&self, outputs: &[T]) -> Vec<T> {
        Self::project(&self.output_mapping, outputs)
    }

    /// Projects logical data inputs to bounded physical tensor types and appends hidden extent scalars.
    pub(crate) fn physical_input_types(&self, inputs: &[ArrayType]) -> Vec<ArrayType> {
        let mut physical = Self::project(&self.input_mapping, inputs);
        let mut converted = vec![false; physical.len()];
        for input_dimension in self.input_dimensions.iter() {
            // The conversion rewrites every axis of the owning tensor at once, so an input with several dynamic axes
            // is converted only for its first hidden extent entry.
            let physical_index = self.input_mapping[input_dimension.logical_input_index].unwrap();
            if std::mem::replace(&mut converted[physical_index], true) {
                continue;
            }
            let dimensions = physical[physical_index]
                .shape()
                .dimensions()
                .iter()
                .map(|dimension| {
                    // Dimension bounds validate an exclusive upper above a non-negative lower, so a bounded dynamic
                    // axis always yields a static bound here; only an unbounded axis stays dynamic, and bounded-input
                    // materialization rejects it as having no static upper shape.
                    dimension
                        .upper_bound()
                        .and_then(|upper| upper.checked_sub(1))
                        .map_or_else(|| dimension.clone(), Dimension::Static)
                })
                .collect::<Vec<_>>();
            physical[physical_index] = physical[physical_index].clone().with_shape(Shape::new(dimensions));
        }
        physical.extend(self.input_dimensions.iter().map(|_| ArrayType::scalar(DataType::I32)));
        physical
    }

    /// Projects logical outputs and appends hidden output-extent scalar types.
    pub(crate) fn physical_output_types(&self, outputs: &[ArrayType]) -> Vec<ArrayType> {
        let mut physical = Self::project(&self.output_mapping, outputs);
        physical.extend(self.output_dimensions.iter().map(|_| ArrayType::scalar(DataType::I64)));
        physical
    }

    /// Projects logical input shardings and appends a replicated scalar sharding per hidden extent argument, on the
    /// mesh of the argument that owns the extent.
    pub(crate) fn physical_input_shardings(&self, shardings: &[Sharding]) -> Vec<Sharding> {
        let mut physical = Self::project(&self.input_mapping, shardings);
        physical.extend(
            self.input_dimensions
                .iter()
                .map(|dimension| Sharding::replicated(shardings[dimension.logical_input_index].mesh().clone(), 0)),
        );
        physical
    }

    /// Projects logical output shardings and appends a replicated scalar sharding per hidden extent result, on the
    /// mesh of the output that owns the extent.
    pub(crate) fn physical_output_shardings(&self, shardings: &[Sharding]) -> Vec<Sharding> {
        let mut physical = Self::project(&self.output_mapping, shardings);
        physical.extend(
            self.output_dimensions
                .iter()
                .map(|dimension| Sharding::replicated(shardings[dimension.logical_output_index].mesh().clone(), 0)),
        );
        physical
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

    /// Whether execution requires the host runtime assertion handler.
    requires_assertion_handler: bool,
}

impl LoweredXlaModule {
    /// Consumes this lowering and returns its textual module, executable signature, and assertion-runtime requirement.
    #[inline]
    pub(crate) fn into_parts(self) -> (String, XlaExecutableSignature, bool) {
        (self.stable_hlo, self.signature, self.requires_assertion_handler)
    }
}

/// Per-class StableHLO tokens owned by one lowering scope.
///
/// Ordered assertions and ordered I/O intentionally use independent chains. This preserves program order within
/// each observable class without introducing an ordering relationship between unrelated classes. Unordered I/O has
/// no token slot.
#[derive(Copy, Clone, Default)]
struct EffectTokens<'b, 'c: 'b, 't: 'c> {
    /// Current ordered-assertion token, created lazily by the first runtime assertion in this scope.
    ordered_assertion: Option<ValueRef<'b, 'c, 't>>,

    /// Current ordered-I/O token, created lazily by the first ordered I/O operation in this scope.
    ordered_io: Option<ValueRef<'b, 'c, 't>>,
}

impl<'b, 'c: 'b, 't: 'c> EffectTokens<'b, 'c, 't> {
    /// Returns the current token for `effect`.
    fn get(&self, effect: Effect) -> Option<ValueRef<'b, 'c, 't>> {
        match effect {
            Effect::OrderedState => unreachable!("ordered state effects must be discharged before XLA lowering"),
            Effect::OrderedAssertion => self.ordered_assertion,
            Effect::OrderedIo => self.ordered_io,
            Effect::UnorderedIo => None,
        }
    }

    /// Replaces the current token for one ordered effect class.
    fn set(&mut self, effect: Effect, token: ValueRef<'b, 'c, 't>) {
        match effect {
            Effect::OrderedState => unreachable!("ordered state effects must be discharged before XLA lowering"),
            Effect::OrderedAssertion => self.ordered_assertion = Some(token),
            Effect::OrderedIo => self.ordered_io = Some(token),
            Effect::UnorderedIo => panic!("unordered effects do not have token slots"),
        }
    }
}

/// Returns `true` if any operation in `program`'s complete region arena carries unresolved ordered state.
///
/// This scans every region rather than consulting the program's effect summary because that summary intentionally
/// excludes dormant rule regions. Ordinary XLA cannot preserve state in either computation or rule regions, so both
/// must be rejected before lowering.
pub(crate) fn contains_unresolved_state<ProgramInput, ProgramOutput>(
    program: &XlaProgram<ProgramInput, ProgramOutput>,
) -> bool
where
    ProgramInput: Parameterized<XlaConstant>,
    ProgramOutput: Parameterized<XlaConstant>,
{
    program.regions().iter().any(|region| {
        region
            .instructions()
            .iter()
            .any(|instruction| instruction.operation().effects().contains(Effect::OrderedState))
    })
}

/// Returns `true` if `program` contains a reference-typed atom or intrinsic reference semantics in any region.
///
/// This check is independent from [`contains_unresolved_state`]: a pure reference pass-through or forwarded
/// reference capture can carry reference semantics without executing a stateful instruction. Both predicates back the
/// pre-compilation state checks and the two direct module-lowering entries; eager binding, dispatch, and staging
/// enforce their corresponding boundary invariants separately.
pub(crate) fn contains_unresolved_references<ProgramInput, ProgramOutput>(
    program: &XlaProgram<ProgramInput, ProgramOutput>,
) -> bool
where
    ProgramInput: Parameterized<XlaConstant>,
    ProgramOutput: Parameterized<XlaConstant>,
{
    // The atom scan dominates for every well-formed program (reference operations always touch a reference-typed
    // atom); the semantics scan is the independent-verifier arm that still catches a core discharge bug which
    // retyped the boundary atoms but left a reference operation behind.
    program.entry_region_ref().contains_atom_type_in_closure(RyftType::is_reference)
        || program.regions().iter().any(|region| {
            region
                .instructions()
                .iter()
                .any(|instruction| !instruction.operation().reference_semantics().is_empty())
        })
}

/// Returns the effect classes that have StableHLO token slots, in canonical token/result order.
///
/// Token slots are an XLA/StableHLO representation decision, so the classification lives here rather than on the
/// core [`Effect`] type — but the match is deliberately exhaustive so that adding an effect class forces an explicit
/// token-slot decision in this backend instead of a silent omission. [`Effect::OrderedState`] has no slot: ordinary
/// XLA lowering rejects unresolved state at its module entry boundaries, and no defensive path may accidentally turn
/// state into an ordinary token-threaded effect.
fn token_threaded_effects(effects: Effects) -> impl Iterator<Item = Effect> {
    effects.into_iter().filter(|effect| match effect {
        Effect::OrderedAssertion | Effect::OrderedIo => true,
        Effect::UnorderedIo | Effect::OrderedState => false,
    })
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

    /// Current per-class effect tokens of the lowering scope this lowerer emits into. Lowerers are constructed per
    /// instruction by the instruction replay loops, which copy the scope-level tokens in through
    /// [`Self::with_effect_tokens`] and read the updated tokens back out after the instruction lowers.
    effect_tokens: EffectTokens<'b, 'c, 't>,

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
            effect_tokens: EffectTokens::default(),
            collective_state: CollectiveLoweringState::new(),
        }
    }

    /// Attaches the declared input types of the instruction currently being lowered.
    pub(crate) fn with_input_types(mut self, input_types: Vec<ArrayType>) -> Self {
        self.input_types = input_types;
        self
    }

    /// Attaches the current per-class effect tokens of the enclosing lowering scope.
    fn with_effect_tokens(mut self, effect_tokens: EffectTokens<'b, 'c, 't>) -> Self {
        self.effect_tokens = effect_tokens;
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
}

/// Operations that can be lowered to StableHLO for XLA compilation.
///
/// Implementing this trait makes an operation eligible for MLIR lowering via
/// [`to_mlir_module_for_plain_program`] and related entry points. The core [`ArrayOperation`] enum provides the default
/// blanket implementation, and backends can add their own closed operation enums by implementing this trait for those
/// enums.
pub(crate) trait LowerableXlaOperation<V: MlirLowerableValue>: Operation<Type = ArrayType> {
    /// Lowers this operation to one or more StableHLO operations.
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>;
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for ConvertElementTypeOperation<ArrayType> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
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
        op: format!("elementwise operand has non-tensor MLIR type `{input_type}`"),
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for AddOperation<ArrayType> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for SubOperation<ArrayType> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for MulOperation<ArrayType> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for DivOperation<ArrayType> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for NegOperation<ArrayType> {
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for SinOperation<ArrayType> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for CosOperation<ArrayType> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for Atan2Operation<ArrayType> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for ExpOperation<ArrayType> {
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for LogOperation<ArrayType> {
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for SqrtOperation<ArrayType> {
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for RsqrtOperation<ArrayType> {
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
                .append_operation(stable_hlo::rsqrt(input_values[0], Accuracy::Default, lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.rsqrt should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for TanhOperation<ArrayType> {
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
                .append_operation(stable_hlo::tanh(input_values[0], Accuracy::Default, lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.tanh should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for LogisticOperation<ArrayType> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
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
impl<V: MlirLowerableValue> LowerableXlaOperation<V> for ErfOperation<ArrayType> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(chlo::erf(input_values[0], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("chlo.erf should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for PowOperation<ArrayType> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for SignOperation<ArrayType> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::sign(input_values[0], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.sign should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for FloorOperation<ArrayType> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::floor(input_values[0], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.floor should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for CeilOperation<ArrayType> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let result = lowerer.block.append_operation(stable_hlo::ceil(input_values[0], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.ceil should return one result").as_ref()])
    }
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for RoundOperation<ArrayType> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for MaxOperation<ArrayType> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for MinOperation<ArrayType> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for RemOperation<ArrayType> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for AbsOperation<ArrayType> {
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for ComplexOperation<ArrayType> {
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
impl<V: MlirLowerableValue> LowerableXlaOperation<V> for ConjugateOperation<ArrayType> {
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for RealOperation<ArrayType> {
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for ImaginaryOperation<ArrayType> {
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for ConstantOperation<CpuArray> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("input", input_values, 0, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        let constant_value =
            self.value().lower_constant_value(&[], &mut lowerer.block, lowerer.context, lowerer.location)?;
        Ok(vec![constant_value])
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
        lower_reshape_to_mlir(self, input_values, output_types, &mut lowerer.block, lowerer.location)
    }
}

/// Lowers a [`ReshapeOperation`] after validating its unary input and single output contract.
fn lower_reshape_to_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &ReshapeOperation,
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayType],
    block: &mut BlockRef<'b, 'c, 't>,
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
    let result = if output_types[0].static_shape().is_none() {
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

/// Reads one runtime tensor dimension and widens StableHLO's `i32` size result to Ryft's `i64` dimension ABI.
pub(super) fn lower_runtime_dimension_size_i64<'b, 'c: 'b, 't: 'c>(
    input: ValueRef<'b, 'c, 't>,
    axis: usize,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let size = block.append_operation(stable_hlo::get_dimension_size(input, axis, location)?)?;
    let size = size.result(0).expect("stablehlo.get_dimension_size should return one result").as_ref();
    let i64_scalar = context.tensor_type(context.signless_integer_type(64), &[], None, location)?;
    let size = block.append_operation(stable_hlo::convert(size, i64_scalar, location)?)?;
    Ok(size.result(0).expect("stablehlo.convert should return one result").as_ref())
}

/// Returns the statically shaped physical-bound carrier type of one bounded dynamic array type.
fn physical_bound_type(r#type: &ArrayType) -> Result<ArrayType, LoweringError> {
    let dimensions = r#type
        .shape()
        .dimensions()
        .iter()
        .map(|dimension| match dimension {
            Dimension::Static(extent) => Ok(Dimension::Static(*extent)),
            Dimension::Dynamic(variable) => {
                variable.bounds().upper().and_then(|upper| upper.checked_sub(1)).map(Dimension::Static).ok_or_else(
                    || LoweringError::UnsupportedOp {
                        op: format!("dynamic dimension {variable} needs a finite positive physical bound"),
                    },
                )
            }
        })
        .collect::<Result<Vec<_>, LoweringError>>()?;
    Ok(r#type.clone().with_shape(Shape::new(dimensions)))
}

/// Materializes the full physical buffer of one bounded dynamic value for an XLA custom call that cannot consume
/// dynamic dimensions. The original runtime sizes are retained before each dynamic axis is widened to its physical
/// bound, and every newly exposed lane is replaced with `padding_value` before the buffer reaches the custom call.
fn lower_static_custom_call_input<'b, 'c: 'b, 't: 'c>(
    value: ValueRef<'b, 'c, 't>,
    r#type: &ArrayType,
    padding_value: f64,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    if r#type.static_shape().is_some() {
        return Ok(value);
    }
    let runtime_sizes = r#type
        .shape()
        .dimensions()
        .iter()
        .enumerate()
        .map(|(axis, dimension)| {
            if matches!(dimension, Dimension::Static(_)) {
                return Ok(None);
            }
            let size = block.append_operation(stable_hlo::get_dimension_size(value, axis, location)?)?;
            Ok(Some(size.result(0).expect("stablehlo.get_dimension_size should return one result").as_ref()))
        })
        .collect::<Result<Vec<_>, LoweringError>>()?;
    let physical_type = physical_bound_type(r#type)?;
    let size_type = lower_tensor_type(&ArrayType::scalar(DataType::I32), context, location)?;
    let mut data = value;
    let mut physicalized_dimensions = r#type.shape().dimensions().to_vec();
    for (axis, dimension) in r#type.shape().dimensions().iter().enumerate() {
        if matches!(dimension, Dimension::Static(_)) {
            continue;
        }
        let physical_extent = physical_type.shape().dimensions()[axis].value().unwrap();
        let extent = reshape_dimension_i32(physical_extent)?;
        let elements = context
            .dense_i32_elements_attribute(size_type, &[extent])
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: DataType::I32 })?;
        let size = block.append_operation(stable_hlo::constant(elements, location)?)?;
        let size = size.result(0).expect("stablehlo.constant should return one result").as_ref();
        physicalized_dimensions[axis] = Dimension::Static(physical_extent);
        let physicalized_type = r#type.clone().with_shape(Shape::new(physicalized_dimensions.clone()));
        let physicalized = block.append_operation(stable_hlo::set_dimension_size(
            data,
            size,
            lower_tensor_type(&physicalized_type, context, location)?,
            axis,
            location,
        )?)?;
        data = physicalized.result(0).expect("stablehlo.set_dimension_size should return one result").as_ref();
    }
    let index_type = ArrayType::new(DataType::I32, physical_type.shape().clone());
    let index_tensor_type = lower_tensor_type(&index_type, context, location)?;
    let mut in_bounds = None;
    for (axis, dimension) in r#type.shape().dimensions().iter().enumerate() {
        if matches!(dimension, Dimension::Static(_)) {
            continue;
        }
        let indices = block.append_operation(stable_hlo::iota(index_tensor_type, axis, location)?)?;
        let indices = indices.result(0).expect("stablehlo.iota should return one result").as_ref();
        let size = runtime_sizes[axis].unwrap();
        let size = block.append_operation(stable_hlo::broadcast(size, index_tensor_type, &[], location)?)?;
        let size = size.result(0).expect("stablehlo.broadcast should return one result").as_ref();
        let axis_in_bounds = lower_compare_to_mlir(ComparisonDirection::LessThan, indices, size, block, location)?;
        in_bounds = Some(match in_bounds {
            None => axis_in_bounds,
            Some(in_bounds) => block
                .append_operation(stable_hlo::and(in_bounds, axis_in_bounds, location)?)?
                .result(0)
                .expect("stablehlo.and should return one result")
                .as_ref(),
        });
    }
    let physical_tensor_type = lower_tensor_type(&physical_type, context, location)?;
    let padding =
        lower_f64_constant_splat(padding_value, &physical_type, physical_tensor_type, block, context, location)?;
    let masked = block.append_operation(stable_hlo::select(in_bounds.unwrap(), data, padding, location)?)?;
    Ok(masked.result(0).expect("stablehlo.select should return one result").as_ref())
}

/// Restores the dynamic dimensions of `output_type` on a physical-bound value. Each entry in `sources` identifies
/// the value and axis that define the corresponding output dimension.
fn lower_restore_dynamic_dimensions<'b, 'c: 'b, 't: 'c>(
    mut value: ValueRef<'b, 'c, 't>,
    output_type: &ArrayType,
    sources: &[(ValueRef<'b, 'c, 't>, usize)],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    if sources.len() != output_type.rank() {
        return Err(ProgramError::InvalidInputCount { expected: output_type.rank(), actual: sources.len() }.into());
    }
    let mut dimensions = physical_bound_type(output_type)?.shape().dimensions().to_vec();
    for (axis, dimension) in output_type.shape().dimensions().iter().enumerate() {
        if matches!(dimension, Dimension::Static(_)) {
            continue;
        }
        let size =
            block.append_operation(stable_hlo::get_dimension_size(sources[axis].0, sources[axis].1, location)?)?;
        let size = size.result(0).expect("stablehlo.get_dimension_size should return one result").as_ref();
        dimensions[axis] = dimension.clone();
        let refined_type = output_type.clone().with_shape(Shape::new(dimensions.clone()));
        let refined = block.append_operation(stable_hlo::set_dimension_size(
            value,
            size,
            lower_tensor_type(&refined_type, context, location)?,
            axis,
            location,
        )?)?;
        value = refined.result(0).expect("stablehlo.set_dimension_size should return one result").as_ref();
    }
    Ok(value)
}

/// Converts one Ryft reshape dimension to StableHLO's signed shape element type.
fn reshape_dimension_i64(value: usize) -> Result<i64, LoweringError> {
    i64::try_from(value).map_err(|_| LoweringError::ReshapeDimensionOutOfRange { value, bit_width: 64 })
}

/// Converts one Ryft reshape dimension to the signed shape element type required by dynamic StableHLO reshape.
fn reshape_dimension_i32(value: usize) -> Result<i32, LoweringError> {
    i32::try_from(value).map_err(|_| LoweringError::ReshapeDimensionOutOfRange { value, bit_width: 32 })
}

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for PadOperation<ArrayType> {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
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
fn lower_pad_to_mlir<'b, 'c: 'b, 't: 'c, T: RyftType, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    operation: &PadOperation<T>,
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

impl<V: MlirLowerableValue> LowerableXlaOperation<V> for BroadcastOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
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
    operation: &BroadcastOperation,
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

/// Returns the current StableHLO token for one ordered effect class, creating that class's chain lazily with a
/// zero-operand `stablehlo.after_all` when the lowering scope has not used it yet.
///
/// Each ordered class has an independent chain that preserves program order within that class without imposing an
/// artificial order across classes. The enclosing lowering helpers carry active chains through nested computations;
/// the final tokens are intentionally omitted from the public function results because their custom calls are marked
/// side-effecting and the tokens exist only to encode intra-execution ordering.
fn current_or_new_token<'b, 'c: 'b, 't: 'c>(
    effect: Effect,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    if let Some(token) = effect_tokens.get(effect) {
        return Ok(token);
    }
    let created = block.append_operation(stable_hlo::after_all::<ValueRef, _>(&[], location)?)?;
    let created = created.result(0).expect("stablehlo.after_all should return one result").as_ref();
    effect_tokens.set(effect, created);
    Ok(created)
}

/// Emits one typed assertion callback and advances only the ordered-assertion chain.
fn lower_assertion_custom_call<'b, 'c: 'b, 't: 'c>(
    predicate: ValueRef<'b, 'c, 't>,
    observed_values: &[ValueRef<'b, 'c, 't>],
    backend_config: DictionaryAttributeRef<'c, 't>,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), LoweringError> {
    let input_token = current_or_new_token(Effect::OrderedAssertion, effect_tokens, block, location)?;
    let mut inputs = Vec::with_capacity(2 + observed_values.len());
    inputs.push(predicate);
    inputs.extend_from_slice(observed_values);
    inputs.push(input_token);
    let operation = block.append_operation(stable_hlo::custom_call(
        inputs.as_slice(),
        ASSERT_CUSTOM_CALL_TARGET,
        true,
        Some(backend_config.as_ref()),
        CustomCallApiVersion::TypedFfi,
        &[],
        None,
        &[],
        None,
        &[context.stable_hlo_token_type()?],
        location,
    )?)?;
    effect_tokens.set(
        Effect::OrderedAssertion,
        operation.result(0).expect("the assertion custom call should return one token result").as_ref(),
    );
    Ok(())
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
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), LoweringError> {
    let input_token = current_or_new_token(Effect::OrderedIo, effect_tokens, block, location)?;
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
    effect_tokens.set(
        Effect::OrderedIo,
        operation.result(0).expect("the print custom call should return one token result").as_ref(),
    );
    Ok(())
}

/// Lowers one retained first-class-dimension requirement to a typed XLA FFI assertion custom call.
///
/// The predicate is computed in StableHLO from the concrete scalar extent operands. The custom call receives that
/// predicate, the observed extents, and only the [`Effect::OrderedAssertion`] token. Its backend configuration keeps
/// the canonical actor and variable names needed to reconstruct the eager diagnostic if the predicate is false.
fn lower_dimension_requirement_to_assertion<'b, 'c: 'b, 't: 'c>(
    operation: &DimensionRequirementOperation,
    actor: &str,
    input_values: &[ValueRef<'b, 'c, 't>],
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), LoweringError> {
    let left = *input_values.first().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?;
    let right = match operation.right_type() {
        Some(_) => Some(
            *input_values
                .get(1)
                .ok_or(ProgramError::InvalidInputCount { expected: 2, actual: input_values.len() })?,
        ),
        None => None,
    };
    let (predicate, kind, requirement) = match operation.predicate() {
        DimensionRequirementPredicate::Equal => (
            lower_compare_to_mlir(ComparisonDirection::Equal, left, right.unwrap(), block, location)?,
            ASSERT_EQUAL_KIND,
            None,
        ),
        DimensionRequirementPredicate::LessThanOrEqual => (
            lower_compare_to_mlir(ComparisonDirection::LessThanOrEqual, left, right.unwrap(), block, location)?,
            ASSERT_LESS_THAN_OR_EQUAL_KIND,
            None,
        ),
        DimensionRequirementPredicate::DivisibleBy => {
            let right = right.unwrap();
            let constants = lower_static_index_constants(&[0, 1], block, context, location)?;
            let zero = constants[0];
            let one = constants[1];
            let positive = lower_compare_to_mlir(ComparisonDirection::GreaterThan, right, zero, block, location)?;
            let safe_divisor = block.append_operation(stable_hlo::select(positive, right, one, location)?)?;
            let safe_divisor = safe_divisor.result(0).expect("stablehlo.select should return one result").as_ref();
            let remainder = block.append_operation(stable_hlo::remainder(left, safe_divisor, location)?)?;
            let remainder = remainder.result(0).expect("stablehlo.remainder should return one result").as_ref();
            let divisible = lower_compare_to_mlir(ComparisonDirection::Equal, remainder, zero, block, location)?;
            let predicate = block.append_operation(stable_hlo::and(positive, divisible, location)?)?;
            (
                predicate.result(0).expect("stablehlo.and should return one result").as_ref(),
                ASSERT_DIVISIBLE_BY_KIND,
                None,
            )
        }
        DimensionRequirementPredicate::Bounds(bounds) => {
            let lower = lower_static_index_constants(&[bounds.lower()], block, context, location)?[0];
            let at_least_lower =
                lower_compare_to_mlir(ComparisonDirection::GreaterThanOrEqual, left, lower, block, location)?;
            // An exclusive upper bound above the maximum representable runtime extent is redundant. Omitting it also
            // avoids attempting to encode `MAX_DIMENSION_EXTENT + 1` in the signed StableHLO index representation.
            let predicate = match bounds.upper().filter(|upper| *upper <= MAX_DIMENSION_EXTENT) {
                Some(upper) => {
                    let upper = lower_static_index_constants(&[upper], block, context, location)?[0];
                    let below_upper =
                        lower_compare_to_mlir(ComparisonDirection::LessThan, left, upper, block, location)?;
                    let predicate = block.append_operation(stable_hlo::and(at_least_lower, below_upper, location)?)?;
                    predicate.result(0).expect("stablehlo.and should return one result").as_ref()
                }
                None => at_least_lower,
            };
            (predicate, ASSERT_BOUNDS_KIND, Some(bounds.to_string()))
        }
    };

    let left_name = operation.left_type().variable().to_string();
    let right_name = operation.right_type().map(|right_type| right_type.variable().to_string());
    let mut attributes = vec![
        context.named_attribute(context.identifier(ASSERT_ACTOR_ATTRIBUTE), context.string_attribute(actor)),
        context.named_attribute(context.identifier(ASSERT_KIND_ATTRIBUTE), context.string_attribute(kind)),
        context
            .named_attribute(context.identifier(ASSERT_LEFT_ATTRIBUTE), context.string_attribute(left_name.as_str())),
    ];
    if let Some(right_name) = right_name.as_deref() {
        attributes.push(
            context.named_attribute(context.identifier(ASSERT_RIGHT_ATTRIBUTE), context.string_attribute(right_name)),
        );
    }
    if let Some(requirement) = requirement.as_deref() {
        attributes.push(
            context.named_attribute(context.identifier(ASSERT_DETAIL_ATTRIBUTE), context.string_attribute(requirement)),
        );
    }
    lower_assertion_custom_call(
        predicate,
        input_values,
        context.dictionary_attribute(attributes.as_slice()),
        effect_tokens,
        block,
        context,
        location,
    )
}

/// Builds the deliberately false `value < 0` predicate attached to diagnostic assertion custom calls whose safety
/// condition is evaluated host-side with checked operations: a nonnegative dimension value can never satisfy it, so
/// the callback always inspects the concrete observed operands, and the side-effecting call cannot be folded away by
/// a constant-true predicate.
fn lower_deliberately_false_assertion_predicate<'b, 'c: 'b, 't: 'c>(
    value: ValueRef<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    let zero = lower_static_index_constants(&[0], block, context, location)?[0];
    lower_compare_to_mlir(ComparisonDirection::LessThan, value, zero, block, location)
}

/// Lowers one runtime safety check for dimension arithmetic whose declared bounds do not prove totality.
fn lower_dimension_arithmetic_assertion<'b, 'c: 'b, 't: 'c>(
    operation: &DimensionOperation<DimensionValue>,
    left_type: &DimensionType,
    right_type: &DimensionType,
    left: ValueRef<'b, 'c, 't>,
    right: ValueRef<'b, 'c, 't>,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), LoweringError> {
    let kind = match operation {
        DimensionOperation::Add(_) => ASSERT_ADD_KIND,
        DimensionOperation::Sub(_) => ASSERT_SUB_KIND,
        DimensionOperation::Mul(_) => ASSERT_MUL_KIND,
        DimensionOperation::Pow(_) => ASSERT_POW_KIND,
        DimensionOperation::DivFloor(_) => ASSERT_DIV_FLOOR_KIND,
        DimensionOperation::Rem(_) => ASSERT_REM_KIND,
        _ => return Ok(()),
    };
    let predicate = lower_deliberately_false_assertion_predicate(left, block, context, location)?;
    let left_name = left_type.variable().to_string();
    let right_name = right_type.variable().to_string();
    let backend_config = context.dictionary_attribute(&[
        context.named_attribute(context.identifier(ASSERT_ACTOR_ATTRIBUTE), context.string_attribute(operation.name())),
        context.named_attribute(context.identifier(ASSERT_KIND_ATTRIBUTE), context.string_attribute(kind)),
        context
            .named_attribute(context.identifier(ASSERT_LEFT_ATTRIBUTE), context.string_attribute(left_name.as_str())),
        context
            .named_attribute(context.identifier(ASSERT_RIGHT_ATTRIBUTE), context.string_attribute(right_name.as_str())),
    ]);
    lower_assertion_custom_call(predicate, &[left, right], backend_config, effect_tokens, block, context, location)
}

/// Lowers the dynamic explicit-result-extent check owned by composite concatenation.
fn lower_concatenate_extent_assertion<'b, 'c: 'b, 't: 'c>(
    axis: usize,
    actual: ValueRef<'b, 'c, 't>,
    input_extents: &[ValueRef<'b, 'c, 't>],
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), LoweringError> {
    let predicate = lower_deliberately_false_assertion_predicate(actual, block, context, location)?;
    let axis = axis.to_string();
    let backend_config = context.dictionary_attribute(&[
        context.named_attribute(context.identifier(ASSERT_ACTOR_ATTRIBUTE), context.string_attribute("concatenate")),
        context.named_attribute(
            context.identifier(ASSERT_KIND_ATTRIBUTE),
            context.string_attribute(ASSERT_CONCATENATE_KIND),
        ),
        context.named_attribute(context.identifier(ASSERT_DETAIL_ATTRIBUTE), context.string_attribute(axis.as_str())),
    ]);
    let observed = std::iter::once(actual).chain(input_extents.iter().copied()).collect::<Vec<_>>();
    lower_assertion_custom_call(predicate, observed.as_slice(), backend_config, effect_tokens, block, context, location)
}

/// Lowers one runtime bounds check for an axis of a dynamic-shape-slice operation.
fn lower_dynamic_shape_slice_assertion<'b, 'c: 'b, 't: 'c>(
    axis: usize,
    stride: usize,
    input_size: ValueRef<'b, 'c, 't>,
    start: ValueRef<'b, 'c, 't>,
    size: ValueRef<'b, 'c, 't>,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), LoweringError> {
    let predicate = lower_deliberately_false_assertion_predicate(start, block, context, location)?;
    let detail = format!("{axis}:{stride}");
    let backend_config = context.dictionary_attribute(&[
        context.named_attribute(
            context.identifier(ASSERT_ACTOR_ATTRIBUTE),
            context.string_attribute("dynamic_shape_slice"),
        ),
        context.named_attribute(
            context.identifier(ASSERT_KIND_ATTRIBUTE),
            context.string_attribute(ASSERT_DYNAMIC_SHAPE_SLICE_KIND),
        ),
        context.named_attribute(context.identifier(ASSERT_DETAIL_ATTRIBUTE), context.string_attribute(detail.as_str())),
    ]);
    lower_assertion_custom_call(
        predicate,
        &[input_size, start, size],
        backend_config,
        effect_tokens,
        block,
        context,
        location,
    )
}

/// Lowers one traced random bit generation to a `stablehlo.rng_bit_generator`, mapping the algorithm to the
/// corresponding StableHLO algorithm attribute. The two results are the advanced generator state and the bits.
fn lower_rng_bit_generator_to_mlir<'b, 'c: 'b, 't: 'c, T: RyftType>(
    operation: &RngBitGeneratorOperation<T>,
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
        return Err(ProgramError::UnsupportedOperation {
            message: format!("`{SORT_OPERATION_NAME}` needs at least one input"),
        }
        .into());
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

/// Traces the canonical scaled-dot decomposition for one validated operation boundary.
fn trace_scaled_dot_composition(
    operation: &ScaledDotOperation,
    input_types: &[ArrayType],
) -> Result<FlatXlaProgram, LoweringError> {
    let input_types = scaled_dot_composite_input_types(operation, input_types)?
        .into_iter()
        .map(ArrayIrType::from)
        .collect::<Vec<_>>();
    let (_, program) = DomainTracingContext::<XlaDomain<'static>>::trace(
        |inputs: Vec<XlaTracer<'static>>| {
            let [lhs, rhs, lhs_scale, rhs_scale] = inputs.as_slice() else {
                return Err(ProgramError::InvalidInputCount { expected: 4, actual: inputs.len() });
            };
            Ok(vec![
                scaled_dot_ir_composition(
                    lhs,
                    rhs,
                    operation.has_lhs_scale().then_some(lhs_scale),
                    operation.has_rhs_scale().then_some(rhs_scale),
                    operation.dimensions(),
                    operation.preferred_element_type(),
                )?
                .into_value(),
            ])
        },
        input_types,
    )?;
    Ok(program.simplified()?)
}

/// Traces the canonical portable attention composition for one validated operation boundary.
fn trace_attention_composition(
    operation: &DotProductAttentionOperation,
    input_types: &[ArrayType],
) -> Result<FlatXlaProgram, LoweringError> {
    let input_types = input_types.iter().cloned().map(ArrayIrType::from).collect::<Vec<_>>();
    let (_, program) = DomainTracingContext::<XlaDomain<'static>>::trace(
        |inputs: Vec<XlaTracer<'static>>| {
            let signature = operation.signature();
            let attention_inputs = AttentionInputs::from_values(signature, &inputs)?;
            let (output, activation) =
                dot_product_attention_ir_composition(&attention_inputs, operation.configuration())?;
            Ok(std::iter::once(output.into_value())
                .chain(activation.map(ProjectedValue::into_value))
                .collect::<Vec<_>>())
        },
        input_types,
    )?;
    Ok(program.simplified()?)
}

/// Traces the canonical portable attention-backward composition for one validated operation boundary.
fn trace_attention_backward_composition(
    operation: &DotProductAttentionBackwardOperation,
    input_types: &[ArrayType],
) -> Result<FlatXlaProgram, LoweringError> {
    let input_types = input_types.iter().cloned().map(ArrayIrType::from).collect::<Vec<_>>();
    let (_, program) = DomainTracingContext::<XlaDomain<'static>>::trace(
        |inputs: Vec<XlaTracer<'static>>| {
            let signature = operation.signature();
            let optional_count = signature.count();
            let attention_inputs = AttentionInputs::from_values(signature, &inputs[..3 + optional_count])?;
            dot_product_attention_backward_ir_composition(
                &attention_inputs,
                &inputs[3 + optional_count],
                &inputs[4 + optional_count],
                &inputs[5 + optional_count],
                operation.configuration(),
            )
            .map(|outputs| outputs.into_iter().map(ProjectedValue::into_value).collect::<Vec<_>>())
        },
        input_types,
    )?;
    Ok(program.simplified()?)
}

/// Returns the dummy scale type used by JAX's four-operand `xla.scaled_dot` boundary for one absent scale.
fn scaled_dot_dummy_scale_type(elements: &ArrayType, contracting_dimensions: &[usize]) -> ArrayType {
    ArrayType::new(
        DataType::F8E8M0FNU,
        Shape::new(
            elements
                .shape()
                .dimensions()
                .iter()
                .enumerate()
                .map(
                    |(axis, dimension)| {
                        if contracting_dimensions.contains(&axis) { Dimension::Static(1) } else { dimension.clone() }
                    },
                )
                .collect(),
        ),
    )
}

/// Expands one public scaled-dot boundary to the exact four inputs consumed by the StableHLO composite.
fn scaled_dot_composite_input_types(
    operation: &ScaledDotOperation,
    input_types: &[ArrayType],
) -> Result<Vec<ArrayType>, LoweringError> {
    let expected_input_count = 2 + usize::from(operation.has_lhs_scale()) + usize::from(operation.has_rhs_scale());
    check_count!("input", input_types, expected_input_count, ProgramError);
    let mut scale_index = 2;
    let lhs_scale = if operation.has_lhs_scale() {
        let r#type = input_types[scale_index].clone();
        scale_index += 1;
        r#type
    } else {
        scaled_dot_dummy_scale_type(&input_types[0], operation.dimensions().lhs_contracting_dimensions())
    };
    let rhs_scale = if operation.has_rhs_scale() {
        input_types[scale_index].clone()
    } else {
        scaled_dot_dummy_scale_type(&input_types[1], operation.dimensions().rhs_contracting_dimensions())
    };
    Ok(vec![input_types[0].clone(), input_types[1].clone(), lhs_scale, rhs_scale])
}

/// Returns whether scaled dot can use a physical CUDA composite boundary without losing a runtime contracting-axis
/// requirement. Dynamic contracting element or scale dimensions retain the logical portable decomposition, whose
/// ordinary dimension SSA proves their block ratio before computing the result.
fn scaled_dot_uses_cuda_physical_boundary(
    operation: &ScaledDotOperation,
    input_types: &[ArrayType],
    target_platform: Option<&str>,
) -> Result<bool, LoweringError> {
    if target_platform != Some("cuda") {
        return Ok(false);
    }
    let expected_input_count = 2 + usize::from(operation.has_lhs_scale()) + usize::from(operation.has_rhs_scale());
    check_count!("input", input_types, expected_input_count, ProgramError);
    let mut scale_index = 2;
    for (element_index, has_scale, contracting_dimensions) in [
        (0, operation.has_lhs_scale(), operation.dimensions().lhs_contracting_dimensions()),
        (1, operation.has_rhs_scale(), operation.dimensions().rhs_contracting_dimensions()),
    ] {
        if contracting_dimensions
            .iter()
            .any(|axis| matches!(input_types[element_index].dimension(*axis), Dimension::Dynamic(_)))
        {
            return Ok(false);
        }
        if has_scale {
            if contracting_dimensions
                .iter()
                .any(|axis| matches!(input_types[scale_index].dimension(*axis), Dimension::Dynamic(_)))
            {
                return Ok(false);
            }
            scale_index += 1;
        }
    }
    Ok(true)
}

/// Returns the typed scaled-dot composition boundary used for the target platform. XLA's CUDA block-scaled-dot
/// replacement cannot propagate dynamic-dimension annotations through its fused HLO, so eligible CUDA calls use the
/// bounded static carriers while the caller masks physical suffix lanes and restores the logical result dimensions.
/// Other calls retain the logical boundary consumed by the portable decomposition.
fn scaled_dot_composition_types(
    operation: &ScaledDotOperation,
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    target_platform: Option<&str>,
) -> Result<(Vec<ArrayType>, Vec<ArrayType>), LoweringError> {
    if !scaled_dot_uses_cuda_physical_boundary(operation, input_types, target_platform)? {
        return Ok((input_types.to_vec(), output_types.to_vec()));
    }
    Ok((
        input_types.iter().map(physical_bound_type).collect::<Result<Vec<_>, _>>()?,
        output_types.iter().map(physical_bound_type).collect::<Result<Vec<_>, _>>()?,
    ))
}

/// Calls one emitted typed decomposition function at its registered array boundary.
fn lower_decomposition_call<'b, 'c: 'b, 't: 'c>(
    function: &NamedCompositionFunction,
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let result_types = output_types
        .iter()
        .map(|r#type| lower_tensor_type(r#type, context, location).map(|r#type| r#type.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    let call = block.append_operation(func::call(
        function.symbol.as_str(),
        func::CallProperties {
            arguments: input_values
                .iter()
                .map(|value| ValueAndAttributes { value: *value, attributes: None })
                .collect(),
            results: result_types
                .iter()
                .map(|r#type| TypeAndAttributes { r#type: *r#type, attributes: None })
                .collect(),
            no_inline: false,
        },
        location,
    )?)?;
    Ok((0..output_types.len())
        .map(|index| call.result(index).expect("func.call should return one result per output").as_ref())
        .collect())
}

/// Lowers one scaled dot as the exact four-operand `stablehlo.composite "xla.scaled_dot"` boundary consumed by XLA's
/// block-scaling replacement pass. Either scale omitted by the public Ryft call is represented by the same
/// JAX-compatible identity scale used by the pinned composite contract. The typed decomposition remains the portable
/// fallback when XLA does not replace the composite.
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
    let expected_input_count = 2 + usize::from(operation.has_lhs_scale()) + usize::from(operation.has_rhs_scale());
    check_count!("input", input_values, expected_input_count, ProgramError);
    check_count!("input", input_types, input_values.len(), ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    let uses_physical_boundary =
        scaled_dot_uses_cuda_physical_boundary(operation, input_types, collective_state.target_platform())?;
    let (composition_input_types, composition_output_types) =
        scaled_dot_composition_types(operation, input_types, output_types, collective_state.target_platform())?;
    let decomposition = match collective_state.named_compositions.as_ref() {
        Some(functions) => functions.get(operation, &composition_input_types, &composition_output_types)?,
        None => None,
    };
    if let Some(function) = decomposition {
        if collective_state.target_platform() == Some("cuda") && !uses_physical_boundary {
            return lower_decomposition_call(function, input_values, output_types, block, context, location);
        }
        let composition_input_values = if uses_physical_boundary {
            input_values
                .iter()
                .zip(input_types)
                .map(|(value, r#type)| lower_static_custom_call_input(*value, r#type, 0.0, block, context, location))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            input_values.to_vec()
        };
        let mut composite_values = vec![composition_input_values[0], composition_input_values[1]];
        let mut scale_index = 2;
        for (present, element_index, contracting_dimensions) in [
            (operation.has_lhs_scale(), 0, operation.dimensions().lhs_contracting_dimensions()),
            (operation.has_rhs_scale(), 1, operation.dimensions().rhs_contracting_dimensions()),
        ] {
            if present {
                composite_values.push(composition_input_values[scale_index]);
                scale_index += 1;
                continue;
            }
            let dummy_type =
                scaled_dot_dummy_scale_type(&composition_input_types[element_index], contracting_dimensions);
            let physical_type = physical_bound_type(&dummy_type)?;
            let physical_tensor_type = lower_tensor_type(&physical_type, context, location)?;
            let dummy = lower_f64_constant_splat(1.0, &physical_type, physical_tensor_type, block, context, location)?;
            if uses_physical_boundary {
                composite_values.push(dummy);
            } else {
                let sources =
                    (0..dummy_type.rank()).map(|axis| (input_values[element_index], axis)).collect::<Vec<_>>();
                composite_values.push(lower_restore_dynamic_dimensions(
                    dummy,
                    &dummy_type,
                    sources.as_slice(),
                    block,
                    context,
                    location,
                )?);
            }
        }
        let render_axes = |axes: &[usize]| axes.iter().map(usize::to_string).collect::<Vec<_>>().join(", ");
        let dimensions = format!(
            "[[[{}], [{}]], [[{}], [{}]]]",
            render_axes(operation.dimensions().lhs_contracting_dimensions()),
            render_axes(operation.dimensions().rhs_contracting_dimensions()),
            render_axes(operation.dimensions().lhs_batching_dimensions()),
            render_axes(operation.dimensions().rhs_batching_dimensions()),
        );
        let attributes = HashMap::from([
            (StringRef::from("dimension_numbers"), context.parse_attribute(dimensions.as_str())?),
            (
                StringRef::from("preferred_element_type"),
                context.type_attribute(lower_element_type(operation.preferred_element_type(), context)?).as_ref(),
            ),
        ]);
        let result_types = composition_output_types
            .iter()
            .map(|r#type| lower_tensor_type(r#type, context, location).map(|r#type| r#type.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        let composite = block.append_operation(stable_hlo::composite(
            "xla.scaled_dot",
            0,
            Some(&attributes),
            composite_values.as_slice(),
            function.symbol.as_str(),
            Vec::new(),
            result_types.as_slice(),
            location,
        )?)?;
        let results =
            (0..output_types.len()).map(|index| composite.result(index).unwrap().as_ref()).collect::<Vec<_>>();
        if !uses_physical_boundary {
            return Ok(results);
        }
        let dimensions = operation.dimensions();
        let output_sources = dimensions
            .lhs_batching_dimensions()
            .iter()
            .copied()
            .map(|axis| (input_values[0], axis))
            .chain(lhs_result_axes(dimensions, input_types[0].rank()).into_iter().map(|axis| (input_values[0], axis)))
            .chain(rhs_result_axes(dimensions, input_types[1].rank()).into_iter().map(|axis| (input_values[1], axis)))
            .collect::<Vec<_>>();
        return Ok(vec![lower_restore_dynamic_dimensions(
            results[0],
            &output_types[0],
            output_sources.as_slice(),
            block,
            context,
            location,
        )?]);
    }

    Err(LoweringError::UnsupportedOp { op: format!("missing typed decomposition for `{}`", operation.name()) })
}

/// Returns the minor-to-major layout required by one custom-call array type.
fn lower_custom_call_layout(r#type: &ArrayType) -> Result<Option<Vec<usize>>, LoweringError> {
    let Some(layout) = r#type.layout() else {
        return Ok(None);
    };
    let Layout::Tiled(layout) = layout else {
        return Err(LoweringError::UnsupportedOp {
            op: format!("{CUSTOM_CALL_OPERATION_NAME} with strided array layout `{layout}`"),
        });
    };
    if !layout.tiles().is_empty() {
        return Err(LoweringError::UnsupportedOp {
            op: format!("{CUSTOM_CALL_OPERATION_NAME} with tiled array layout `{layout}`"),
        });
    }
    if layout.rank() != r#type.rank()
        || layout.minor_to_major().iter().any(|axis| *axis >= r#type.rank())
        || layout.minor_to_major().iter().collect::<HashSet<_>>().len() != r#type.rank()
    {
        return Err(LoweringError::UnsupportedOp {
            op: format!(
                "{} with invalid array layout `{}` for rank-{} type `{}`",
                CUSTOM_CALL_OPERATION_NAME,
                layout,
                r#type.rank(),
                r#type,
            ),
        });
    }
    Ok(Some(layout.minor_to_major().to_vec()))
}

/// Returns the complete StableHLO custom-call layout lists when at least one array type requests an explicit layout.
fn lower_custom_call_memory_layouts(
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    has_effect_token: bool,
) -> Result<Option<CustomCallMemoryLayouts>, LoweringError> {
    let input_layouts = input_types.iter().map(lower_custom_call_layout).collect::<Result<Vec<_>, _>>()?;
    let output_layouts = output_types.iter().map(lower_custom_call_layout).collect::<Result<Vec<_>, _>>()?;
    if input_layouts.iter().chain(&output_layouts).all(Option::is_none) {
        return Ok(None);
    }
    let mut operands = input_types
        .iter()
        .zip(input_layouts)
        .map(|(r#type, layout)| layout.unwrap_or_else(|| (0..r#type.rank()).rev().collect()))
        .collect::<Vec<_>>();
    let mut results = output_types
        .iter()
        .zip(output_layouts)
        .map(|(r#type, layout)| layout.unwrap_or_else(|| (0..r#type.rank()).rev().collect()))
        .collect::<Vec<_>>();
    if has_effect_token {
        operands.push(Vec::new());
        results.push(Vec::new());
    }
    Ok(Some(CustomCallMemoryLayouts { operands, results }))
}

/// Lowers one traced custom call to a `stablehlo.custom_call` using the typed FFI calling convention
/// (`api_version = 4`). Typed attributes become the `backend_config` dictionary, array layouts become complete
/// StableHLO operand/result layout lists, and flat input/output aliases become StableHLO output-operand aliases.
/// A side-effecting call additionally consumes and produces the current ordered-I/O token so multiple such calls
/// remain ordered even when their array results do not carry a data dependency. Handlers are resolved by the XLA
/// runtime through the target name at execution time (e.g., registered via `ryft-pjrt`'s
/// `Client::register_ffi_handler`).
fn lower_custom_call_to_mlir<'b, 'c: 'b, 't: 'c, T: RyftType>(
    operation: &CustomCallOperation<T>,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[ArrayType],
    output_types: &[ArrayType],
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    check_count!("input", input_types, input_values.len(), ProgramError);
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
    let memory_layouts = lower_custom_call_memory_layouts(input_types, output_types, operation.has_side_effect())?;
    let mut lowered_inputs = input_values.to_vec();
    if operation.has_side_effect() {
        lowered_inputs.push(current_or_new_token(Effect::OrderedIo, effect_tokens, block, location)?);
    }
    let mut lowered_output_types = output_types
        .iter()
        .map(|output_type| lower_tensor_type(output_type, context, location).map(|r#type| r#type.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    if operation.has_side_effect() {
        lowered_output_types.push(context.stable_hlo_token_type()?.as_ref());
    }
    let output_operand_aliases = operation
        .input_output_aliases()
        .iter()
        .map(|alias| {
            let output_tuple_indices =
                if lowered_output_types.len() == 1 { Vec::new() } else { vec![alias.output_index()] };
            context.stable_hlo_output_operand_alias(output_tuple_indices.as_slice(), alias.input_index(), &[])
        })
        .collect::<Result<Vec<_>, _>>()?;
    let lowered = block.append_operation(stable_hlo::custom_call(
        lowered_inputs.as_slice(),
        operation.target_name(),
        operation.has_side_effect(),
        Some(backend_config.as_ref()),
        CustomCallApiVersion::TypedFfi,
        &[],
        memory_layouts,
        output_operand_aliases.as_slice(),
        None,
        &lowered_output_types,
        location,
    )?)?;
    if operation.has_side_effect() {
        effect_tokens.set(
            Effect::OrderedIo,
            lowered
                .result(output_types.len())
                .expect("a side-effecting custom call should return one trailing token result")
                .as_ref(),
        );
    }
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
                <ConstantOperation<CpuArray> as LowerableXlaOperation<V>>::lower_to_mlir(
                    constant,
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            ArrayOperation::ConvertElementType(operation) => {
                <ConvertElementTypeOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                    operation,
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
            ArrayOperation::Add(operation) => <AddOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Sub(operation) => <SubOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Mul(operation) => <MulOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Div(operation) => <DivOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Neg(operation) => <NegOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Sin(operation) => <SinOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Cos(operation) => <CosOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Atan2(operation) => <Atan2Operation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Exp(operation) => <ExpOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Log(operation) => <LogOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Sqrt(operation) => <SqrtOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Rsqrt(operation) => <RsqrtOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Tanh(operation) => <TanhOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Logistic(operation) => {
                <LogisticOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                    operation,
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            ArrayOperation::Erf(operation) => <ErfOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Pow(operation) => <PowOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Sign(operation) => <SignOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Floor(operation) => <FloorOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Ceil(operation) => <CeilOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Round(operation) => <RoundOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Max(operation) => <MaxOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Min(operation) => <MinOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Rem(operation) => <RemOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Abs(operation) => <AbsOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Complex(operation) => {
                <ComplexOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                    operation,
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            ArrayOperation::Conjugate(operation) => {
                <ConjugateOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                    operation,
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            ArrayOperation::Real(operation) => <RealOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                operation,
                input_values,
                output_types,
                mode,
                lowerer,
            ),
            ArrayOperation::Imaginary(operation) => {
                <ImaginaryOperation<ArrayType> as LowerableXlaOperation<V>>::lower_to_mlir(
                    operation,
                    input_values,
                    output_types,
                    mode,
                    lowerer,
                )
            }
            // `stop_gradient` only affects differentiation; by lowering time it is the identity, so
            // forward every operand without emitting any MLIR operation (matching JAX's lowering).
            ArrayOperation::StopGradient(_) => Ok(input_values.to_vec()),
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
                    &mut lowerer.effect_tokens,
                    &mut lowerer.block,
                    lowerer.context,
                    lowerer.location,
                )?;
                Ok(vec![input_values[0]])
            }
            ArrayOperation::CustomCall(operation) => lower_custom_call_to_mlir(
                operation,
                input_values,
                &lowerer.input_types,
                output_types,
                &mut lowerer.effect_tokens,
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
            ArrayOperation::CustomJvp(_) | ArrayOperation::CustomVjp(_) | ArrayOperation::Rematerialize(_) => {
                Err(ProgramError::UnsupportedOperation {
                    message: "higher-order operation must be stored directly in the enclosing backend operation family"
                        .to_string(),
                }
                .into())
            }
            ArrayOperation::LinearCall(operation) => Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "higher-order operation `{}` must be stored directly in the enclosing backend operation family",
                    operation.name(),
                ),
            }
            .into()),
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
            ArrayOperation::Condition(operation) => Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "higher-order operation `{}` must be stored directly in the enclosing backend operation family",
                    operation.name(),
                ),
            }
            .into()),
            ArrayOperation::While(operation) => Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "higher-order operation `{}` must be stored directly in the enclosing backend operation family",
                    operation.name(),
                ),
            }
            .into()),
            ArrayOperation::Scan(operation) => Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "higher-order operation `{}` must be stored directly in the enclosing backend operation family",
                    operation.name(),
                ),
            }
            .into()),
        }
    }
}

/// Lowering state consulted when lowering collectives: the module-scoped channel-id allocator (each channeled
/// Typed attribute value that participates in named-composition identity.
///
/// This intentionally models only the structural values required by StableHLO named compositions. It is independent
/// of any one Ryft operation family, so adding another named composition does not require a bespoke key type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum NamedCompositionAttribute {
    /// Nested axis lists, such as StableHLO dot dimension numbers.
    AxisLists(Vec<Vec<Vec<usize>>>),

    /// An array element type.
    DataType(DataType),

    /// Boolean configuration value.
    Boolean(bool),

    /// Optional bit-preserving `f64` configuration value.
    OptionalFloat64(Option<u64>),

    /// Optional non-negative integer configuration value.
    OptionalUnsigned(Option<usize>),

    /// Optional pair of a bit-preserving `f64` and an unsigned integer.
    OptionalFloat64Unsigned(Option<(u64, u64)>),
}

/// Identity of one generated named-composition decomposition at a concrete typed boundary.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct NamedCompositionKey {
    /// Stable semantic name consumed by the backend replacement pass.
    name: &'static str,

    /// Stable semantic version consumed by the backend replacement pass.
    version: u32,

    /// Stable identity of the typed program that defines the decomposition semantics.
    decomposition: &'static str,

    /// Canonically ordered semantic attributes.
    attributes: Vec<(&'static str, NamedCompositionAttribute)>,

    /// Logical operand types of the composition boundary.
    input_types: Vec<ArrayType>,

    /// Logical result types of the composition boundary.
    output_types: Vec<ArrayType>,
}

/// Private decomposition function emitted for one [`NamedCompositionKey`].
struct NamedCompositionFunction {
    /// Module-unique decomposition symbol.
    symbol: String,

    /// Canonical typed decomposition program.
    program: FlatXlaProgram,
}

/// Module-owned registry of canonical named compositions and their private decomposition functions.
#[derive(Default)]
struct NamedCompositionFunctionMap {
    /// Functions indexed by semantic operation and typed boundary.
    functions: HashMap<NamedCompositionKey, NamedCompositionFunction>,

    /// Stable first-occurrence order used for deterministic function emission.
    order: Vec<NamedCompositionKey>,
}

impl NamedCompositionFunctionMap {
    /// Registers one typed decomposition lazily, sharing an existing private function when the complete semantic key
    /// is already present. The registry is independent of the operation family that produced `key` and `program`;
    /// closed backend operation enums only decide when to call this seam.
    fn register<F: FnOnce() -> Result<FlatXlaProgram, LoweringError>>(
        &mut self,
        key: NamedCompositionKey,
        program: F,
    ) -> Result<(), LoweringError> {
        if self.functions.contains_key(&key) {
            return Ok(());
        }
        let base_symbol = key.name.to_string();
        let symbol = if self.order.iter().all(|existing| existing.name != key.name) {
            base_symbol
        } else {
            format!("{base_symbol}_{}", self.order.iter().filter(|existing| existing.name == key.name).count())
        };
        self.functions.insert(key.clone(), NamedCompositionFunction { symbol, program: program()? });
        self.order.push(key);
        Ok(())
    }

    /// Returns the registered decomposition for `operation` at the provided typed boundary.
    fn get(
        &self,
        operation: &ScaledDotOperation,
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Option<&NamedCompositionFunction>, LoweringError> {
        Ok(self.functions.get(&scaled_dot_composition_key(operation, input_types, output_types)?))
    }

    /// Returns the registered portable decomposition for `operation` at the provided typed boundary.
    fn get_attention(
        &self,
        operation: &DotProductAttentionOperation,
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Option<&NamedCompositionFunction> {
        self.functions.get(&attention_composition_key(operation, input_types, output_types))
    }

    /// Returns the registered portable backward decomposition for `operation` at the provided typed boundary.
    fn get_attention_backward(
        &self,
        operation: &DotProductAttentionBackwardOperation,
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Option<&NamedCompositionFunction> {
        self.functions.get(&attention_backward_composition_key(operation, input_types, output_types))
    }
}

/// Builds the canonical named-composition identity for one scaled-dot boundary.
fn scaled_dot_composition_key(
    operation: &ScaledDotOperation,
    input_types: &[ArrayType],
    output_types: &[ArrayType],
) -> Result<NamedCompositionKey, LoweringError> {
    let dimensions = operation.dimensions();
    Ok(NamedCompositionKey {
        name: "xla.scaled_dot",
        version: 0,
        decomposition: "ryft.scaled_dot.v0",
        attributes: vec![
            (
                "dimension_numbers",
                NamedCompositionAttribute::AxisLists(vec![
                    vec![
                        dimensions.lhs_contracting_dimensions().to_vec(),
                        dimensions.rhs_contracting_dimensions().to_vec(),
                    ],
                    vec![dimensions.lhs_batching_dimensions().to_vec(), dimensions.rhs_batching_dimensions().to_vec()],
                ]),
            ),
            ("preferred_element_type", NamedCompositionAttribute::DataType(operation.preferred_element_type())),
        ],
        input_types: scaled_dot_composite_input_types(operation, input_types)?,
        output_types: output_types.to_vec(),
    })
}

/// Builds the canonical identity of one portable attention composition.
fn attention_composition_key(
    operation: &DotProductAttentionOperation,
    input_types: &[ArrayType],
    output_types: &[ArrayType],
) -> NamedCompositionKey {
    let configuration = operation.configuration();
    let signature = operation.signature();
    NamedCompositionKey {
        name: "ryft.dot_product_attention",
        version: 0,
        decomposition: "ryft.dot_product_attention.v0",
        attributes: vec![
            ("scale", NamedCompositionAttribute::OptionalFloat64(configuration.scale().map(f64::to_bits))),
            ("causal", NamedCompositionAttribute::Boolean(configuration.causal())),
            (
                "local_window_left",
                NamedCompositionAttribute::OptionalUnsigned(configuration.local_window().map(|window| window.0)),
            ),
            (
                "local_window_right",
                NamedCompositionAttribute::OptionalUnsigned(configuration.local_window().map(|window| window.1)),
            ),
            (
                "dropout",
                NamedCompositionAttribute::OptionalFloat64Unsigned(
                    configuration.dropout().map(|(rate, seed)| (rate.to_bits(), seed)),
                ),
            ),
            ("residual", NamedCompositionAttribute::Boolean(configuration.return_residual())),
            ("bias", NamedCompositionAttribute::Boolean(signature.has_bias())),
            ("mask", NamedCompositionAttribute::Boolean(signature.has_mask())),
            ("query_lengths", NamedCompositionAttribute::Boolean(signature.has_query_sequence_lengths())),
            ("key_value_lengths", NamedCompositionAttribute::Boolean(signature.has_key_value_sequence_lengths())),
        ],
        input_types: input_types.to_vec(),
        output_types: output_types.to_vec(),
    }
}

/// Builds the canonical identity of one portable attention-backward composition.
fn attention_backward_composition_key(
    operation: &DotProductAttentionBackwardOperation,
    input_types: &[ArrayType],
    output_types: &[ArrayType],
) -> NamedCompositionKey {
    let configuration = operation.configuration();
    let signature = operation.signature();
    NamedCompositionKey {
        name: "ryft.dot_product_attention_backward",
        version: 0,
        decomposition: "ryft.dot_product_attention_backward.v0",
        attributes: vec![
            ("scale", NamedCompositionAttribute::OptionalFloat64(configuration.scale().map(f64::to_bits))),
            ("causal", NamedCompositionAttribute::Boolean(configuration.causal())),
            (
                "local_window_left",
                NamedCompositionAttribute::OptionalUnsigned(configuration.local_window().map(|window| window.0)),
            ),
            (
                "local_window_right",
                NamedCompositionAttribute::OptionalUnsigned(configuration.local_window().map(|window| window.1)),
            ),
            (
                "dropout",
                NamedCompositionAttribute::OptionalFloat64Unsigned(
                    configuration.dropout().map(|(rate, seed)| (rate.to_bits(), seed)),
                ),
            ),
            ("bias", NamedCompositionAttribute::Boolean(signature.has_bias())),
            ("mask", NamedCompositionAttribute::Boolean(signature.has_mask())),
            ("query_lengths", NamedCompositionAttribute::Boolean(signature.has_query_sequence_lengths())),
            ("key_value_lengths", NamedCompositionAttribute::Boolean(signature.has_key_value_sequence_lengths())),
        ],
        input_types: input_types.to_vec(),
        output_types: output_types.to_vec(),
    }
}

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
    /// no target information. Platform-gated lowerings such as fused attention consult this and
    /// fall back to their portable form when it is absent.
    target_platform: Option<Rc<str>>,

    /// Module-owned private decompositions used by named StableHLO composites.
    named_compositions: Option<Rc<NamedCompositionFunctionMap>>,

    /// Module-scoped flag recording whether any recursive lowering path constructed an instruction-specific
    /// (non-base) location from a non-unknown [`Provenance`]. It selects the module serialization mode after
    /// lowering completes: only a module that actually carries provenance locations is printed with debug
    /// information, so provenance-free modules keep their existing byte-identical StableHLO text and cache keys.
    has_provenance: Rc<Cell<bool>>,
}

impl CollectiveLoweringState {
    /// Creates the lowering state for one module, outside any manual region and without target information.
    pub(crate) fn new() -> Self {
        Self {
            channel_ids: Rc::new(Cell::new(1)),
            manual_shard_map: None,
            target_platform: None,
            named_compositions: None,
            has_provenance: Rc::new(Cell::new(false)),
        }
    }

    /// Returns a copy of this state carrying the PJRT platform name of the compilation target.
    pub(crate) fn with_target_platform(mut self, target_platform: Option<&str>) -> Self {
        self.target_platform = target_platform.map(Rc::from);
        self
    }

    /// Returns a copy of this state carrying the module's named-composition decomposition registry.
    fn with_named_compositions(mut self, named_compositions: Rc<NamedCompositionFunctionMap>) -> Self {
        self.named_compositions = Some(named_compositions);
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
            named_compositions: self.named_compositions.clone(),
            has_provenance: self.has_provenance.clone(),
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

    /// Lowers one instruction's [`Provenance`] onto MLIR locations above the provided base location and records
    /// module-wide whether any instruction-specific location was constructed. Unknown provenance uses the base
    /// location unchanged, each scope level becomes one
    /// [`NameLoc`](https://mlir.llvm.org/docs/Dialects/Builtin/#nameloc) above its lowered origin, and fused
    /// provenance becomes one metadata-free
    /// [`FusedLoc`](https://mlir.llvm.org/docs/Dialects/Builtin/#fusedloc) over its recursively lowered origins. An
    /// existing file/line base location is thereby preserved as the innermost child of the named scopes rather than
    /// being replaced.
    pub(crate) fn instruction_location<'c, 't: 'c>(
        &self,
        context: &'c MlirContext<'t>,
        provenance: &Provenance,
        base: LocationRef<'c, 't>,
    ) -> LocationRef<'c, 't> {
        /// Recursively lowers `provenance` above `base` without touching the module-wide flag.
        fn lower<'c, 't: 'c>(
            context: &'c MlirContext<'t>,
            provenance: &Provenance,
            base: LocationRef<'c, 't>,
        ) -> LocationRef<'c, 't> {
            if let Some((scope, origin)) = provenance.as_scope() {
                let child = lower(context, origin, base);
                context.named_location(scope.name(), Some(child)).as_ref()
            } else if let Some(origins) = provenance.as_fused() {
                let locations = origins.iter().map(|origin| lower(context, origin, base)).collect::<Vec<_>>();
                context.fused_location::<_, AttributeRef>(locations.as_slice(), None).as_ref()
            } else {
                base
            }
        }

        if provenance.is_unknown() {
            return base;
        }
        self.has_provenance.set(true);
        lower(context, provenance, base)
    }

    /// Returns `true` if any recursive lowering path constructed an instruction-specific location from a
    /// non-unknown [`Provenance`].
    pub(crate) fn has_provenance(&self) -> bool {
        self.has_provenance.get()
    }
}

/// Serializes one lowered module to StableHLO text, printing MLIR debug information (i.e., locations) exactly when
/// the lowering constructed instruction-specific provenance locations. Provenance-free modules keep the plain
/// rendering, so their StableHLO text, snapshots, and compilation cache keys remain byte-identical to before
/// provenance existed, while provenance-carrying modules embed their locations in the text and therefore in the
/// persistent compilation key derived from it. The pretty debug form is documented as unparsable and never used, and
/// the elision thresholds are disabled because the plain rendering performs no elision either.
///
/// Exact backend provenance can reduce compiled-artifact cache reuse, because two semantically equal modules with
/// different provenance serialize (and therefore key) differently. If that cost ever matters, the correct extension
/// is an explicit compilation option that strips provenance before lowering (making this helper fall back to the
/// plain rendering); silently returning an artifact labeled with another program's provenance is never acceptable.
fn serialize_lowered_module(
    module: &Module<'_, '_>,
    collective_state: &CollectiveLoweringState,
) -> Result<String, LoweringError> {
    if !collective_state.has_provenance() {
        return Ok(module.to_string());
    }
    module
        .as_operation()?
        .to_string_with_flags(OperationPrintingFlags {
            elements_attribute_size_threshold: None,
            resource_string_size_threshold: None,
            enable_debug_information: true,
            pretty_print_debug_information: false,
            ..OperationPrintingFlags::default()
        })
        .map_err(|error| ryft_mlir::Error::internal(format!("serialized module is not valid UTF-8: {error}")).into())
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
    input_types: Vec<ArrayIrType>,

    /// Shared private functions emitted for deduplicated `jit_call` callees, consulted at `jit_call` lowering sites.
    /// Shared via [`Rc`] so it threads through nested lowering scopes without lifetime entanglement.
    nested_functions: Option<Rc<JitCallFunctionMap>>,

    /// Hidden capture arguments of the function currently being lowered, in capture-table order.
    captured_values: Vec<ValueRef<'b, 'c, 't>>,

    /// Current per-class effect tokens of the lowering scope this lowerer emits into. Refer to the documentation of
    /// the equivalent [`PlainMlirLowerer`] field for the copy-in/copy-out threading protocol.
    effect_tokens: EffectTokens<'b, 'c, 't>,

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
            effect_tokens: EffectTokens::default(),
            collective_state: CollectiveLoweringState::new(),
        }
    }

    /// Attaches the declared input types of the instruction currently being lowered.
    pub(crate) fn with_input_types(mut self, input_types: Vec<ArrayIrType>) -> Self {
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

    /// Attaches the current per-class effect tokens of the enclosing lowering scope.
    fn with_effect_tokens(mut self, effect_tokens: EffectTokens<'b, 'c, 't>) -> Self {
        self.effect_tokens = effect_tokens;
        self
    }

    /// Attaches the collective lowering state of the enclosing lowering scope.
    pub(crate) fn with_collective_state(mut self, collective_state: CollectiveLoweringState) -> Self {
        self.collective_state = collective_state;
        self
    }

    /// Lowers one nested condition operation inside this lowering context.
    pub(crate) fn lower_condition(
        &mut self,
        branch_regions: &[FlatXlaProgram],
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
            &mut self.effect_tokens,
        )
    }

    /// Lowers one nested while operation inside this lowering context.
    pub(crate) fn lower_while(
        &mut self,
        while_op: &WhileOperation<ArrayIrType>,
        loop_regions: &[FlatXlaProgram],
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
            &mut self.effect_tokens,
        )
    }

    /// Lowers one nested scan operation inside this lowering context.
    pub(crate) fn lower_scan(
        &mut self,
        scan_op: &ScanOperation<XlaConstant>,
        scan_regions: &[FlatXlaProgram],
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let [body] = scan_regions else {
            return Err(LoweringError::UnsupportedOp {
                op: format!("{} expected 1 attached region but got {}", SCAN_OPERATION_NAME, scan_regions.len()),
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
            &mut self.effect_tokens,
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
    // This module entry must enforce the same discharge preconditions as `lower_mlir_module_for_program`: these are
    // the only guards keeping unresolved state and references out of the shard-map token-threading machinery.
    if contains_unresolved_references(program) {
        return Err(LoweringError::UnresolvedReference { construct: "program with unresolved references".to_string() });
    }
    if contains_unresolved_state(program) {
        return Err(LoweringError::UnresolvedState { construct: "program".to_string() });
    }
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

    // Module-scoped collective lowering state: this entry point lowers one whole module.
    let collective_state = CollectiveLoweringState::new();
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

    serialize_lowered_module(&module, &collective_state)
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
    lower_mlir_module_for_program_with_reference_state(
        program,
        capture_types,
        global_input_types,
        global_output_types,
        function_name,
        arg_shardings,
        result_shardings,
        target_platform,
        &[],
    )
}

/// Lowers a discharged program with logical external reference-state aliases on its entry boundary.
///
/// `reference_states` are the discharge artifact's external-state recipes; each must name a distinct logical state
/// input. Each state input must survive the executable boundary. A mutated state must be a static device array and
/// name a hidden output of the identical type and sharding; mutated pairs are recorded as `tf.aliasing_output`
/// input-output aliases, which are non-semantic may-alias hints that merely permit backend buffer reuse. The runtime
/// never donates reference-state inputs, so an alias never authorizes in-place mutation of the caller's state buffer.
/// Read-only state may be finite bounded-dynamic because it has no hidden alias. The validation here owns the
/// physical alias contract; the XLA domain additionally enforces the supported memory, sharding, and dynamic-shape
/// classes before calling it.
pub(crate) fn lower_mlir_module_for_program_with_reference_state<'o, Input, Output, ProgramInput, ProgramOutput, S>(
    program: &XlaProgram<ProgramInput, ProgramOutput>,
    capture_types: &[ArrayType],
    global_input_types: &Input,
    global_output_types: &Output,
    function_name: S,
    arg_shardings: Option<&[Sharding]>,
    result_shardings: Option<&[Sharding]>,
    target_platform: Option<&str>,
    reference_states: &[ReferenceStateBinding],
) -> Result<LoweredXlaModule, LoweringError>
where
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
    ProgramInput: Parameterized<XlaConstant>,
    ProgramOutput: Parameterized<XlaConstant>,
    S: AsRef<str>,
{
    if contains_unresolved_references(program) {
        return Err(LoweringError::UnresolvedReference { construct: "program with unresolved references".to_string() });
    }
    if contains_unresolved_state(program) {
        return Err(LoweringError::UnresolvedState { construct: "program".to_string() });
    }
    let function_name = normalize_function_name(function_name.as_ref())?;
    let global_input_types = global_input_types.parameters().cloned().collect::<Vec<_>>();
    let global_output_types = global_output_types.parameters().cloned().collect::<Vec<_>>();
    let logical_argument_types =
        capture_types.iter().cloned().chain(global_input_types.iter().cloned()).collect::<Vec<_>>();
    let signature = XlaExecutableSignature::new(logical_argument_types.as_slice(), global_output_types.as_slice());
    if let Some(shardings) = arg_shardings
        && shardings.len() != logical_argument_types.len()
    {
        return Err(LoweringError::InvalidShardingCount {
            kind: "argument",
            expected: logical_argument_types.len(),
            actual: shardings.len(),
        });
    }
    if let Some(shardings) = result_shardings
        && shardings.len() != global_output_types.len()
    {
        return Err(LoweringError::InvalidShardingCount {
            kind: "result",
            expected: global_output_types.len(),
            actual: shardings.len(),
        });
    }
    let mut aliases = HashMap::with_capacity(reference_states.len());
    let mut state_inputs = HashSet::with_capacity(reference_states.len());
    let mut aliased_outputs = HashSet::with_capacity(reference_states.len());
    for state in reference_states {
        let logical_input_index = state.discharged_input_index();
        if !state_inputs.insert(logical_input_index) {
            return Err(LoweringError::InvalidReferenceStateAbi {
                message: format!("logical state input {logical_input_index} appears more than once"),
            });
        }
        let input_type =
            logical_argument_types
                .get(logical_input_index)
                .ok_or_else(|| LoweringError::InvalidReferenceStateAbi {
                    message: format!("logical input index {logical_input_index} is out of range"),
                })?;
        let physical_input_index =
            signature.input_mapping()[logical_input_index].ok_or_else(|| LoweringError::InvalidReferenceStateAbi {
                message: format!("logical state input {logical_input_index} is erased from the executable boundary"),
            })?;
        if let Some(logical_output_index) = state.final_state_output_index() {
            let output_type = global_output_types.get(logical_output_index).ok_or_else(|| {
                LoweringError::InvalidReferenceStateAbi {
                    message: format!("logical output index {logical_output_index} is out of range"),
                }
            })?;
            let physical_output_index = signature.output_mapping()[logical_output_index].ok_or_else(|| {
                LoweringError::InvalidReferenceStateAbi {
                    message: format!(
                        "logical state output {logical_output_index} is erased from the executable boundary",
                    ),
                }
            })?;
            if input_type.static_shape().is_none() || output_type.static_shape().is_none() {
                return Err(LoweringError::InvalidReferenceStateAbi {
                    message: format!(
                        "state input {logical_input_index} and output {logical_output_index} must be static because \
                         bounded-dynamic mutation alias compatibility is unsupported",
                    ),
                });
            }
            if input_type.memory() != Memory::Device || output_type.memory() != Memory::Device {
                return Err(LoweringError::InvalidReferenceStateAbi {
                    message: format!(
                        "state input {logical_input_index} and output {logical_output_index} must use device \
                         memory",
                    ),
                });
            }
            if input_type != output_type {
                return Err(LoweringError::InvalidReferenceStateAbi {
                    message: format!(
                        "state input {logical_input_index} type `{input_type}` is incompatible with output \
                         {logical_output_index} type `{output_type}`",
                    ),
                });
            }
            match (arg_shardings, result_shardings) {
                (Some(argument_shardings), Some(result_shardings))
                    if argument_shardings[logical_input_index] != result_shardings[logical_output_index] =>
                {
                    return Err(LoweringError::InvalidReferenceStateAbi {
                        message: format!(
                            "state input {logical_input_index} and output {logical_output_index} must use the same \
                             sharding",
                        ),
                    });
                }
                (Some(_), None) | (None, Some(_)) => {
                    return Err(LoweringError::InvalidReferenceStateAbi {
                        message:
                            "reference-state aliases require both argument and result sharding metadata or neither"
                                .to_string(),
                    });
                }
                _ => {}
            }
            if !aliased_outputs.insert(physical_output_index) {
                return Err(LoweringError::InvalidReferenceStateAbi {
                    message: "reference-state aliases must be injective".to_string(),
                });
            }
            // Distinct logical inputs (enforced through `state_inputs` above) map to distinct physical inputs, so
            // this insertion can never displace an earlier alias.
            aliases.insert(physical_input_index, physical_output_index);
        }
    }
    let physical_argument_types = signature.physical_input_types(logical_argument_types.as_slice());
    let physical_output_types = signature.physical_output_types(global_output_types.as_slice());

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

    // Trace each distinct named semantic decomposition once and retain it in the module-scoped lowering state so
    // entry and nested function bodies resolve the same private symbol.
    let named_compositions = Rc::new(collect_named_composition_functions(program, target_platform)?);

    // Module-scoped collective lowering state, shared between the entry function body and private functions below so
    // channel ids stay unique module-wide and target/composition information reaches nested bodies.
    let collective_state = CollectiveLoweringState::new()
        .with_target_platform(target_platform)
        .with_named_compositions(named_compositions.clone());

    // Deduplicate `jit_call` callees that occur more than once into shared private `func.func`s, so repeated nested
    // programs (identical transformer blocks, or the per-block primal and pullback programs produced by `grad`) lower
    // to one function plus N `func.call`s instead of N inlined copies. The map is empty for modules without repeated
    // calls, in which case every `jit_call` inlines exactly as before.
    let nested_functions = Rc::new(collect_jit_call_functions(program));
    {
        let mut module_block = module.body()?;
        for key in &named_compositions.order {
            let function = named_compositions.functions.get(key).unwrap();
            emit_named_composition_function(
                &mut module_block,
                function,
                &nested_functions,
                &collective_state,
                &context,
                location.as_ref(),
            )?;
        }
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
            let physical_shardings = signature.physical_input_shardings(shardings);
            Some(
                physical_shardings
                    .iter()
                    .map(|sharding| sharding.to_mlir(location))
                    .collect::<Result<Vec<_>, _>>()?,
            )
        }
        None => None,
    };
    let result_sharding_attributes = match result_shardings {
        Some(shardings) => {
            let physical_shardings = signature.physical_output_shardings(shardings);
            Some(
                physical_shardings
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
            let mut attributes = HashMap::new();
            if let Some(shardings) = &arg_sharding_attributes {
                attributes.insert("sdy.sharding".into(), shardings[index].as_ref());
            }
            if let Some(output_index) = aliases.get(&index) {
                attributes.insert(
                    "tf.aliasing_output".into(),
                    context.integer_attribute(context.signless_integer_type(64), *output_index as i64).as_ref(),
                );
            }
            let attributes = (!attributes.is_empty()).then_some(attributes);
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
                    Some(physical_index) => {
                        let mut value = function_block
                            .argument(physical_index)
                            .expect("physical function block arguments should exist")
                            .as_ref();
                        let mut refined_type = physical_argument_types[physical_index].clone();
                        for input_dimension in signature
                            .input_dimensions()
                            .iter()
                            .filter(|dimension| dimension.logical_input_index() == logical_index)
                        {
                            let mut dimensions = refined_type.shape().dimensions().to_vec();
                            dimensions[input_dimension.axis()] =
                                array_type.shape().dimensions()[input_dimension.axis()].clone();
                            refined_type = refined_type.with_shape(Shape::new(dimensions));
                            let size = function_block
                                .argument(input_dimension.physical_input_index())
                                .expect("hidden input dimension arguments should exist")
                                .as_ref();
                            let operation = function_block_ref.append_operation(stable_hlo::set_dimension_size(
                                value,
                                size,
                                lower_tensor_type(&refined_type, &context, location)?,
                                input_dimension.axis(),
                                location,
                            )?)?;
                            value = operation
                                .result(0)
                                .expect("stablehlo.set_dimension_size should return one result")
                                .as_ref();
                        }
                        value
                    }
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
            let (capture_values, public_input_values) = logical_argument_values.split_at(capture_types.len());
            // Ordinary module callers lower an unlifted program whose entry inputs are only the public arguments.
            // The compilation domain instead supplies the capture-lifted program so core reference discharge and
            // lowering observe the same arena; in that form the complete logical argument list is also the entry
            // input list, while the leading prefix still serves attached `CaptureReference` constants. No third
            // boundary form exists, so the entry input count must match one of those two shapes exactly.
            debug_assert!(
                program.input_count() == logical_argument_values.len()
                    || program.input_count() == public_input_values.len(),
            );
            let input_values = if program.input_count() == logical_argument_values.len() {
                logical_argument_values.as_slice()
            } else {
                public_input_values
            };
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
            let mut physical_outputs = signature.project_outputs(logical_outputs.as_slice());
            for output_dimension in signature.output_dimensions() {
                physical_outputs.push(lower_runtime_dimension_size_i64(
                    logical_outputs[output_dimension.logical_output_index()],
                    output_dimension.axis(),
                    &mut function_block_ref,
                    &context,
                    location.as_ref(),
                )?);
            }
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
    // Preserve the source effect classification explicitly. Persistent cache loads no longer have the core program
    // available, and scanning rendered StableHLO for a target name would be a brittle substitute for typed metadata.
    Ok(LoweredXlaModule {
        stable_hlo: serialize_lowered_module(&module, &collective_state)?,
        signature,
        requires_assertion_handler: program.effects().contains(Effect::OrderedAssertion),
    })
}

/// Value type that can be materialized as a StableHLO dense constant during benchmark lowering.
pub(crate) trait MlirLowerableValue: Value<Type = ArrayType> + 'static {
    /// Builds a dense-elements attribute containing this value.
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError>;

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
}

/// Concrete host literal lowering used by [`ConstantOperation`] and MLIR snapshot tooling.
impl MlirLowerableValue for CpuArray {
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
        let data_type = self.r#type().data_type();
        if matches!(data_type, DataType::Token | DataType::Zero) {
            return Err(LoweringError::UnsupportedDataType { data_type });
        }

        // Layout-free arrays already store values in MLIR's logical row-major order. Explicit layouts instead require
        // traversal through `ArrayAddressing`, which `logical_bytes` performs while omitting holes and tile padding.
        let bytes = if self.r#type().layout().is_none() {
            Cow::Borrowed(self.storage_bytes())
        } else {
            Cow::Owned(self.logical_bytes())
        };

        // Ryft storage uses portable little-endian element encodings, whereas MLIR's raw-buffer API consumes native
        // storage representations. Normalize multi-byte components only on big-endian targets.
        #[cfg(target_endian = "big")]
        let bytes = {
            let mut bytes = bytes;
            let component_byte_count = match data_type {
                DataType::I16 | DataType::U16 | DataType::BF16 | DataType::F16 => 2,
                DataType::I32 | DataType::U32 | DataType::F32 | DataType::C64 => 4,
                DataType::I64 | DataType::U64 | DataType::F64 | DataType::C128 => 8,
                _ => 1,
            };
            if component_byte_count > 1 {
                bytes.to_mut().chunks_exact_mut(component_byte_count).for_each(<[u8]>::reverse);
            }
            bytes
        };

        // MLIR's typed constructors own the required packing for one-bit values. Ryft deliberately keeps these values
        // byte-padded in host arrays, so decode only their low bit at this representation boundary.
        if matches!(data_type, DataType::Boolean | DataType::I1 | DataType::U1) {
            return context
                .dense_bool_elements_attribute(
                    tensor_type,
                    bytes.iter().map(|value| value & 1 != 0).collect::<Vec<_>>().as_slice(),
                )
                .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type })?
                .cast::<DenseElementsAttributeRef>()
                .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type });
        }

        context
            .dense_elements_attribute_from_raw_buffer(tensor_type, bytes.as_ref())
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type })
    }
}

/// Lowers a plain traced `tracing_v2` program to a textual StableHLO MLIR module.
#[cfg(test)]
pub(crate) fn to_mlir_module_for_plain_program<
    V: MlirLowerableValue,
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
    // Module-scoped collective lowering state: this entry point lowers one whole module.
    let collective_state = CollectiveLoweringState::new();
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
            lower_tensor_type(&input_atom.r#type().as_ref(), &context, location)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output_tensor_types = program
        .output_ids()
        .iter()
        .map(|atom_id| {
            let output_atom = &program.atoms()[atom_id.index()];
            lower_tensor_type(&output_atom.r#type().as_ref(), &context, location)
        })
        .collect::<Result<Vec<_>, _>>()?;

    module.body()?.append_operation({
        let function_block = context.block(
            input_tensor_types.iter().map(|tensor_type| (*tensor_type, location)).collect::<Vec<_>>().as_slice(),
        );
        {
            let mut function_block_ref = function_block.as_ref();
            let outputs = lower_plain_program_outputs(
                program,
                &mut function_block_ref,
                &context,
                location.as_ref(),
                &collective_state,
            )?;
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

    serialize_lowered_module(&module, &collective_state)
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
                XlaOperation::Array(ArrayOperation::Reshard(operation)) => {
                    mesh = Some(match mesh.take() {
                        Some(existing_mesh) => merge_logical_meshes(&existing_mesh, operation.sharding().mesh())?,
                        None => operation.sharding().mesh().clone(),
                    });
                }
                XlaOperation::Array(ArrayOperation::ShardingConstraint(operation)) => {
                    mesh = Some(match mesh.take() {
                        Some(existing_mesh) => merge_logical_meshes(&existing_mesh, operation.sharding().mesh())?,
                        None => operation.sharding().mesh().clone(),
                    });
                }
                XlaOperation::Broadcast(operation)
                    if {
                        let input_type = region.atoms()[instruction.inputs()[0].index()].r#type();
                        let input_type = <&ArrayType>::try_from(input_type.as_ref()).map_err(ProgramError::from)?;
                        let output_type = region.atoms()[instruction.outputs()[0].index()].r#type();
                        let output_type = <&ArrayType>::try_from(output_type.as_ref()).map_err(ProgramError::from)?;
                        broadcast_changes_explicit_sharding(input_type, output_type, operation.output_axes())
                    } =>
                {
                    let output_type = region.atoms()[instruction.outputs()[0].index()].r#type();
                    let output_type = <&ArrayType>::try_from(output_type.as_ref()).map_err(ProgramError::from)?;
                    let output_sharding = output_type.sharding().unwrap();
                    mesh = Some(match mesh.take() {
                        Some(existing_mesh) => merge_logical_meshes(&existing_mesh, output_sharding.mesh())?,
                        None => output_sharding.mesh().clone(),
                    });
                }
                XlaOperation::Array(ArrayOperation::Broadcast(operation))
                    if {
                        let input_type = region.atoms()[instruction.inputs()[0].index()].r#type();
                        let input_type = <&ArrayType>::try_from(input_type.as_ref()).map_err(ProgramError::from)?;
                        let output_type = region.atoms()[instruction.outputs()[0].index()].r#type();
                        let output_type = <&ArrayType>::try_from(output_type.as_ref()).map_err(ProgramError::from)?;
                        broadcast_changes_explicit_sharding(input_type, output_type, operation.output_axes())
                    } =>
                {
                    let output_type = region.atoms()[instruction.outputs()[0].index()].r#type();
                    let output_type = <&ArrayType>::try_from(output_type.as_ref()).map_err(ProgramError::from)?;
                    let output_sharding = output_type.sharding().unwrap();
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
/// `entry_effect_tokens` are the enclosing scope's active per-class tokens, referenced inside the region through
/// StableHLO's implicit region capture (the same mechanism that feeds `input_values` into `stablehlo.if` branches).
/// The region returns one trailing token for each class in `threaded_effects`, in canonical effect order. A branch
/// that is pure for one of those classes returns that class's entry token unchanged.
fn lower_control_flow_region<'b, 'c: 'b, 't: 'c>(
    program: &FlatXlaProgram,
    input_values: &[ValueRef<'b, 'c, 't>],
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    captured_values: &[ValueRef<'b, 'c, 't>],
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
    entry_effect_tokens: EffectTokens<'b, 'c, 't>,
    threaded_effects: Effects,
) -> Result<ryft_mlir::DetachedRegion<'c, 't>, LoweringError> {
    let mut region = context.region();
    let block = context.block_with_no_arguments();
    {
        let mut block_ref = block.as_ref();
        let mut region_effect_tokens = entry_effect_tokens;
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
            &mut region_effect_tokens,
        )?;
        for effect in token_threaded_effects(threaded_effects) {
            outputs.push(
                region_effect_tokens
                    .get(effect)
                    .expect("token-returning control-flow regions receive one entry token per active class"),
            );
        }
        block_ref.append_operation(stable_hlo::r#return(outputs.as_slice(), location)?)?;
    }
    region.append_block(block)?;
    Ok(region)
}

fn lower_condition_to_if<'b, 'c: 'b, 't: 'c>(
    branch_regions: &[FlatXlaProgram],
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    captured_values: &[ValueRef<'b, 'c, 't>],
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let [true_branch, false_branch] = branch_regions else {
        return Err(LoweringError::UnsupportedOp {
            op: format!("{} expected 2 attached regions but got {}", CONDITION_OPERATION_NAME, branch_regions.len(),),
        });
    };
    let expected_input_count = true_branch.input_types().len() + 1;
    if input_values.len() != expected_input_count {
        return Err(LoweringError::UnsupportedOp {
            op: format!(
                "{} expected {} lowered inputs but got {}",
                CONDITION_OPERATION_NAME,
                expected_input_count,
                input_values.len(),
            ),
        });
    }
    let branch_inputs = &input_values[1..];
    // Each ordered class used by either branch is captured and returned independently. Both branches carry the union
    // so their result signatures agree, returning an entry token unchanged when that branch is pure for the class.
    let threaded_effects = true_branch.effects().union(false_branch.effects());
    let mut entry_effect_tokens = *effect_tokens;
    for effect in token_threaded_effects(threaded_effects) {
        let token = current_or_new_token(effect, effect_tokens, block, location)?;
        entry_effect_tokens.set(effect, token);
    }
    let true_branch_region = lower_control_flow_region(
        true_branch,
        branch_inputs,
        context,
        location,
        captured_values,
        nested_functions,
        collective_state,
        entry_effect_tokens,
        threaded_effects,
    )?;
    let false_branch_region = lower_control_flow_region(
        false_branch,
        branch_inputs,
        context,
        location,
        captured_values,
        nested_functions,
        collective_state,
        entry_effect_tokens,
        threaded_effects,
    )?;
    let operation = block.append_operation(stable_hlo::r#if(
        input_values[0],
        true_branch_region.into(),
        false_branch_region.into(),
        location,
    )?)?;
    let output_count = true_branch.output_types().len();
    for (token_index, effect) in token_threaded_effects(threaded_effects).enumerate() {
        effect_tokens.set(
            effect,
            operation
                .result(output_count + token_index)
                .expect("a token-threaded stablehlo.if should return one trailing result per active effect class")
                .as_ref(),
        );
    }
    Ok((0..output_count)
        .map(|index| operation.result(index).expect("stablehlo.if should return one result per output").as_ref())
        .collect())
}

fn lower_while_to_while<'b, 'c: 'b, 't: 'c>(
    while_op: &WhileOperation<ArrayIrType>,
    loop_regions: &[FlatXlaProgram],
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    captured_values: &[ValueRef<'b, 'c, 't>],
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let [condition, body] = loop_regions else {
        return Err(LoweringError::UnsupportedOp {
            op: format!("{} expected 2 attached regions but got {}", WHILE_OPERATION_NAME, loop_regions.len()),
        });
    };
    let state_types = body.input_types();
    let state_count = state_types.len();
    if input_values.len() != state_count {
        return Err(LoweringError::UnsupportedOp {
            op: format!(
                "{} expected {} lowered inputs but got {}",
                WHILE_OPERATION_NAME,
                state_count,
                input_values.len(),
            ),
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
    let condition_output_types = condition.output_types();
    let predicate_type = <&ArrayType>::try_from(&condition_output_types[0]).map_err(ProgramError::from)?.clone();
    let batched_predicate = predicate_type.rank() > 0;
    // StableHLO's while-condition region can return only the continuation predicate, not updated effect tokens.
    // Therefore, an effectful scalar condition is evaluated before the loop and after each body execution, with its
    // predicate carried through the loop state. This preserves exactly-once condition effects and lets the body
    // return their updated tokens. Batched predicates already use the same carried-predicate shape for masking.
    let condition_effects = condition.effects();
    let threaded_predicate = batched_predicate || !condition_effects.is_pure();
    let predicate_dimensions = (0..predicate_type.rank()).collect::<Vec<_>>();
    let predicate_offset = if threaded_predicate { 1 } else { 0 };
    // A semantic iteration bound is enforced by threading an internal `i64` iteration counter through the
    // `stablehlo.while` state (element 0, starting at zero and incremented once per body run) and conjoining
    // `counter < bound` into the lowered condition. The counter is internal extra state: the operation's outputs
    // remain exactly the original state elements. Unbounded loops emit no counter machinery at all.
    let iteration_bound = while_op.iteration_bound();
    let counter_offset = if iteration_bound.is_some() { 1 } else { 0 };
    // Carry one trailing token for each ordered class used by either nested program. Each class advances independently
    // through the body; a body that is pure for one active class returns that class's entry token unchanged.
    let threaded_effects = condition_effects.union(body.effects());
    let threaded_effect_count = token_threaded_effects(threaded_effects).count();
    // State layout: `[counter?, states..., predicate?, ordered-effect tokens...]`.
    let predicate_index = counter_offset + state_count;
    let token_start_index = counter_offset + state_count + predicate_offset;
    let mut full_state_types = Vec::with_capacity(counter_offset + state_count + predicate_offset);
    if iteration_bound.is_some() {
        full_state_types.push(ArrayType::scalar(DataType::I64).into());
    }
    full_state_types.extend(state_types.iter().cloned());
    if threaded_predicate {
        full_state_types.push(predicate_type.clone().into());
    }
    let mut lowered_state_types = full_state_types
        .iter()
        .map(|r#type| composite::lower_array_ir_type(r#type, context, location).map(|tensor_type| tensor_type.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    for _ in 0..threaded_effect_count {
        lowered_state_types.push(context.stable_hlo_token_type()?.as_ref());
    }
    let block_arguments = lowered_state_types.iter().map(|r#type| (*r#type, location)).collect::<Vec<_>>();

    // Seed the loop state. A threaded predicate is evaluated once on the entry state in the enclosing block. For an
    // effectful scalar condition, this also advances its ordered effect chains before those tokens enter the loop.
    let mut state_values = Vec::with_capacity(lowered_state_types.len());
    if iteration_bound.is_some() {
        state_values.push(lower_static_index_constants(&[0], block, context, location)?[0]);
    }
    state_values.extend_from_slice(input_values);
    if threaded_predicate {
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
            effect_tokens,
        )?;
        if initial_predicate.len() != 1 {
            return Err(LoweringError::UnsupportedOp {
                op: format!("{} condition lowered to {} outputs", WHILE_OPERATION_NAME, initial_predicate.len(),),
            });
        }
        state_values.push(initial_predicate[0]);
    }
    for effect in token_threaded_effects(threaded_effects) {
        state_values.push(current_or_new_token(effect, effect_tokens, block, location)?);
    }

    let mut condition_region = context.region();
    let condition_block = context.block(block_arguments.as_slice());
    {
        let mut condition_block_ref = condition_block.as_ref();
        // A threaded predicate is read from the loop state. Batched predicates are reduced with Boolean `or`, so the
        // loop continues while any mapped item remains active; effectful scalar predicates are already scalar. Pure
        // scalar predicates retain the smaller representation and are evaluated directly in this region.
        let loop_predicate = if threaded_predicate {
            let carried_predicate = condition_block_ref
                .argument(predicate_index)
                .expect("predicate-threaded while state should include the carried predicate")
                .as_ref();
            if batched_predicate {
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
                carried_predicate
            }
        } else {
            let condition_inputs = (counter_offset..counter_offset + state_count)
                .map(|index| {
                    condition_block_ref.argument(index).expect("while condition should have state arguments").as_ref()
                })
                .collect::<Vec<_>>();
            let mut condition_effect_tokens = EffectTokens::default();
            for (token_offset, effect) in token_threaded_effects(threaded_effects).enumerate() {
                condition_effect_tokens.set(
                    effect,
                    condition_block_ref
                        .argument(token_start_index + token_offset)
                        .expect("token-threaded while state should include every active effect token")
                        .as_ref(),
                );
            }
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
                &mut condition_effect_tokens,
            )?;
            if condition_outputs.len() != 1 {
                return Err(LoweringError::UnsupportedOp {
                    op: format!("{} condition lowered to {} outputs", WHILE_OPERATION_NAME, condition_outputs.len(),),
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
        let mut body_effect_tokens = EffectTokens::default();
        for (token_offset, effect) in token_threaded_effects(threaded_effects).enumerate() {
            body_effect_tokens.set(
                effect,
                body_block_ref
                    .argument(token_start_index + token_offset)
                    .expect("token-threaded while state should include every active effect token")
                    .as_ref(),
            );
        }
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
            &mut body_effect_tokens,
        )?;
        if body_outputs.len() != state_count {
            return Err(LoweringError::UnsupportedOp {
                op: format!("{} body lowered to {} outputs", WHILE_OPERATION_NAME, body_outputs.len()),
            });
        }
        // For a batched predicate, mask each carry update under the carried (incoming-state) predicate so finished
        // items freeze. A scalar loop uses the body outputs directly.
        let next_state_values = if batched_predicate {
            let carried_predicate = body_block_ref
                .argument(predicate_index)
                .expect("batched-predicate while state should include the carried predicate")
                .as_ref();
            let masked = body_outputs
                .into_iter()
                .zip(body_inputs.iter())
                .zip(state_types.iter())
                .map(|((candidate, carried), state_type)| {
                    // A first-class dimension carry is loop-invariant under a batched predicate (the contract
                    // documented on `WhileTypeSemantics`), so masking it is the identity and the body's candidate
                    // result is threaded on directly. This matches eager interpretation, whose `mask_select` returns
                    // equal dimension carries unchanged.
                    if matches!(state_type, ArrayIrType::Dimension(_)) {
                        return Ok(candidate);
                    }
                    let state_type = <&ArrayType>::try_from(state_type).map_err(ProgramError::from)?;
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
            masked
        } else {
            body_outputs
        };
        // Recompute a threaded predicate after the body. An effectful scalar condition shares the body's token set,
        // so its effects become the tokens returned as the next loop state. Batched conditions are required to be
        // pure, but follow the same value path.
        let next_predicate = if threaded_predicate {
            let next_predicate = lower_nested_program_inline(
                condition,
                next_state_values.as_slice(),
                &mut body_block_ref,
                context,
                location,
                captured_values,
                false,
                nested_functions,
                collective_state,
                &mut body_effect_tokens,
            )?;
            if next_predicate.len() != 1 {
                return Err(LoweringError::UnsupportedOp {
                    op: format!("{} condition lowered to {} outputs", WHILE_OPERATION_NAME, next_predicate.len(),),
                });
            }
            Some(next_predicate[0])
        } else {
            None
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
        for effect in token_threaded_effects(threaded_effects) {
            next_state.push(
                body_effect_tokens
                    .get(effect)
                    .expect("token-threaded while bodies receive every active effect token"),
            );
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
    for (token_offset, effect) in token_threaded_effects(threaded_effects).enumerate() {
        effect_tokens.set(
            effect,
            operation
                .result(token_start_index + token_offset)
                .expect("a token-threaded stablehlo.while should return one result per active effect class")
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

/// Lowers one shape-counted scan loop to a `stablehlo.while` over the state
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
fn lower_scan_to_while<'b, 'c: 'b, 't: 'c>(
    body_program: &FlatXlaProgram,
    carry_count: usize,
    length: &Dimension,
    reverse: bool,
    unroll: usize,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    captured_values: &[ValueRef<'b, 'c, 't>],
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    // The loop carries one trailing token per ordered class used by the body. Fully unrolled bodies update the
    // enclosing scope's chains directly, while pure scans emit no token machinery.
    let threaded_effects = body_program.effects();
    let body_input_types = body_program.input_types();
    let body_output_types = body_program.output_types();
    let runtime_length_count = usize::from(length.variable().is_some());
    if input_values.len() != body_input_types.len() + runtime_length_count {
        return Err(LoweringError::UnsupportedOp {
            op: format!(
                "{} expected {} lowered inputs but got {}",
                SCAN_OPERATION_NAME,
                body_input_types.len() + runtime_length_count,
                input_values.len(),
            ),
        });
    }
    let (input_values, runtime_length) = if runtime_length_count == 1 {
        (&input_values[..body_input_types.len()], Some(input_values[body_input_types.len()]))
    } else {
        (input_values, None)
    };
    let static_length = length.value();
    if unroll == 0
        || static_length.is_some_and(|length| length % unroll != 0)
        || (static_length.is_none() && unroll != 1)
    {
        return Err(LoweringError::UnsupportedOp {
            op: format!(
                "{SCAN_OPERATION_NAME} unroll factor {unroll} must be at least 1 and evenly divide the scan length \
                 {length}",
            ),
        });
    }
    let carry_types = &body_input_types[..carry_count];
    let x_slice_types = body_input_types[carry_count..]
        .iter()
        .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
        .collect::<Result<Vec<_>, _>>()
        .map_err(ProgramError::from)?;
    let y_slice_types = body_output_types[carry_count..]
        .iter()
        .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
        .collect::<Result<Vec<_>, _>>()
        .map_err(ProgramError::from)?;
    let stacked = |slice_type: &ArrayType| -> Result<ArrayType, LoweringError> {
        let mut dimensions = vec![length.clone()];
        dimensions.extend(slice_type.shape().dimensions().iter().cloned());
        Ok(ArrayType::new(slice_type.data_type(), ryft_core::arrays::Shape::new(dimensions)))
    };

    let initialize_accumulator =
        |slice_type: &ArrayType, block: &mut BlockRef<'b, 'c, 't>| -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
            let stacked_type = stacked(slice_type)?;
            let physical_length = match length {
                Dimension::Static(length) => *length,
                Dimension::Dynamic(_) => {
                    stable_hlo_dynamic_dimension_bound(length).ok_or_else(|| LoweringError::UnsupportedOp {
                        op: format!(
                            "{SCAN_OPERATION_NAME} length {length} needs a finite upper bound for physical accumulator \
                         allocation",
                        ),
                    })?
                }
            };
            let mut physical_dimensions = stacked_type.shape().dimensions().to_vec();
            physical_dimensions[0] = Dimension::Static(physical_length);
            let physical_type = stacked_type.clone().with_shape(Shape::new(physical_dimensions));
            let accumulator =
                lower_constant_output(std::slice::from_ref(&physical_type), 0, block, context, location)?.remove(0);
            let Some(runtime_length) = runtime_length else {
                return Ok(accumulator);
            };
            let i32_scalar_type = context.tensor_type(context.signless_integer_type(32), &[], None, location)?;
            let converted = block.append_operation(stable_hlo::convert(runtime_length, i32_scalar_type, location)?)?;
            let converted = converted.result(0).expect("stablehlo.convert should return one result").as_ref();
            let tensor_type = lower_tensor_type(&stacked_type, context, location)?;
            let refined = block.append_operation(stable_hlo::set_dimension_size(
                accumulator,
                converted,
                tensor_type,
                0,
                location,
            )?)?;
            Ok(refined.result(0).expect("stablehlo.set_dimension_size should return one result").as_ref())
        };

    // A fully unrolled scan (`unroll == length`) needs no loop at all: the body copies inline as straight-line
    // operations at static iteration indices, reading and writing the same stacked inputs and zero accumulators the
    // loop form would thread through its state.
    if static_length.is_some_and(|length| unroll == length && length > 0) {
        let length = static_length.unwrap();
        let mut carries = input_values[..carry_count].to_vec();
        let x_stacks = input_values[carry_count..].to_vec();
        let mut y_accumulators = Vec::with_capacity(y_slice_types.len());
        for y_slice_type in &y_slice_types {
            y_accumulators.push(initialize_accumulator(y_slice_type, block)?);
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
                x_slice_types.as_slice(),
                y_slice_types.as_slice(),
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
                effect_tokens,
            )?;
        }
        carries.extend(y_accumulators);
        return Ok(carries);
    }

    // Assemble the loop state `[counter, carries..., stacks..., ys...]`, preallocating one zero accumulator per
    // stacked output.
    let mut state_types = Vec::with_capacity(1 + body_input_types.len() + y_slice_types.len());
    state_types.push(ArrayType::scalar(DataType::I64).into());
    state_types.extend(carry_types.iter().cloned());
    for x_slice_type in &x_slice_types {
        state_types.push(stacked(x_slice_type)?.into());
    }
    let mut state_values = Vec::with_capacity(state_types.len() + y_slice_types.len());
    state_values.push(lower_static_index_constants(&[0], block, context, location)?[0]);
    state_values.extend_from_slice(input_values);
    for y_slice_type in &y_slice_types {
        let stacked_type = stacked(y_slice_type)?;
        state_values.push(initialize_accumulator(y_slice_type, block)?);
        state_types.push(stacked_type.into());
    }
    // Effect tokens ride at the end of the loop state, so counter/carry/stack/accumulator index math stays untouched.
    let token_start_index = state_types.len();
    let mut lowered_state_types = state_types
        .iter()
        .map(|r#type| composite::lower_array_ir_type(r#type, context, location).map(|tensor_type| tensor_type.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    for effect in token_threaded_effects(threaded_effects) {
        lowered_state_types.push(context.stable_hlo_token_type()?.as_ref());
        state_values.push(current_or_new_token(effect, effect_tokens, block, location)?);
    }
    let block_arguments = lowered_state_types.iter().map(|r#type| (*r#type, location)).collect::<Vec<_>>();

    let mut condition_region = context.region();
    let condition_block = context.block(block_arguments.as_slice());
    {
        let mut condition_block_ref = condition_block.as_ref();
        let counter = condition_block_ref.argument(0).expect("scan while state should include the counter").as_ref();
        let length_value = match static_length {
            Some(length) => lower_static_index_constants(&[length], &mut condition_block_ref, context, location)?[0],
            None => runtime_length.unwrap(),
        };
        let predicate = lower_compare_to_mlir(
            ComparisonDirection::LessThan,
            counter,
            length_value,
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
            let length_value = match static_length {
                Some(length) => lower_static_index_constants(&[length], &mut body_block_ref, context, location)?[0],
                None => runtime_length.unwrap(),
            };
            let one = lower_static_index_constants(&[1], &mut body_block_ref, context, location)?[0];
            let limit = body_block_ref.append_operation(stable_hlo::subtract(length_value, one, location)?)?;
            Some(limit.result(0).expect("stablehlo.subtract should return one result").as_ref())
        } else {
            None
        };

        // Each loop trip runs `unroll` consecutive logical iterations (`counter + copy` for each body copy), so the
        // counter advances by `unroll` per trip and the unchanged `counter < length` condition yields
        // `length / unroll` trips.
        let mut carries = arguments[1..1 + carry_count].to_vec();
        let x_stacks = arguments[1 + carry_count..1 + carry_count + x_slice_types.len()].to_vec();
        let mut y_accumulators = arguments[1 + carry_count + x_slice_types.len()..].to_vec();
        let mut body_effect_tokens = EffectTokens::default();
        for (token_offset, effect) in token_threaded_effects(threaded_effects).enumerate() {
            body_effect_tokens.set(
                effect,
                body_block_ref
                    .argument(token_start_index + token_offset)
                    .expect("token-threaded scan state should include every active effect token")
                    .as_ref(),
            );
        }
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
                x_slice_types.as_slice(),
                y_slice_types.as_slice(),
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
                &mut body_effect_tokens,
            )?;
        }

        // Assemble the next state: advance the counter by the unroll factor, thread the new carries, pass the input
        // stacks through unchanged, and thread the updated stacked accumulators and active effect tokens.
        let step = lower_static_index_constants(&[unroll], &mut body_block_ref, context, location)?[0];
        let next_counter = body_block_ref.append_operation(stable_hlo::add(counter, step, location)?)?;
        let mut next_state = vec![next_counter.result(0).expect("stablehlo.add should return one result").as_ref()];
        next_state.extend(carries);
        next_state.extend(x_stacks);
        next_state.extend(y_accumulators);
        for effect in token_threaded_effects(threaded_effects) {
            next_state.push(
                body_effect_tokens
                    .get(effect)
                    .expect("token-threaded scan bodies receive every active effect token"),
            );
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
    for (token_offset, effect) in token_threaded_effects(threaded_effects).enumerate() {
        effect_tokens.set(
            effect,
            operation
                .result(token_start_index + token_offset)
                .expect("a token-threaded scan should return one result per active effect class")
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
fn lower_scan_iteration<'b, 'c: 'b, 't: 'c>(
    body_program: &FlatXlaProgram,
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
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
) -> Result<(Vec<ValueRef<'b, 'c, 't>>, Vec<ValueRef<'b, 'c, 't>>), LoweringError> {
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
        effect_tokens,
    )?;
    if body_outputs.len() != carry_count + y_slice_types.len() {
        return Err(LoweringError::UnsupportedOp {
            op: format!("{} body lowered to {} outputs", SCAN_OPERATION_NAME, body_outputs.len()),
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

/// Collects every named semantic composition required by `program` and traces each distinct typed decomposition once.
fn collect_named_composition_functions<Input, Output>(
    program: &XlaProgram<Input, Output>,
    target_platform: Option<&str>,
) -> Result<NamedCompositionFunctionMap, LoweringError>
where
    Input: Parameterized<XlaConstant>,
    Output: Parameterized<XlaConstant>,
{
    fn walk(
        region: RegionRef<'_, XlaConstant, XlaOperation>,
        map: &mut NamedCompositionFunctionMap,
        visited: &mut HashSet<RegionId>,
        target_platform: Option<&str>,
    ) -> Result<(), LoweringError> {
        if !visited.insert(region.id()) {
            return Ok(());
        }
        for instruction in region.instructions() {
            for nested in instruction.regions() {
                walk(RegionRef::new(region.arena(), *nested)?, map, visited, target_platform)?;
            }
            let XlaOperation::Array(operation) = instruction.operation() else {
                continue;
            };
            if !matches!(
                operation,
                ArrayOperation::ScaledDot(_)
                    | ArrayOperation::DotProductAttention(_)
                    | ArrayOperation::DotProductAttentionBackward(_)
            ) {
                continue;
            }
            let input_types = instruction
                .inputs()
                .iter()
                .map(|input| {
                    let r#type = region.atoms()[input.index()].r#type();
                    <&ArrayType>::try_from(r#type.as_ref()).cloned().map_err(ProgramError::from)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let output_types = instruction
                .outputs()
                .iter()
                .map(|output| {
                    let r#type = region.atoms()[output.index()].r#type();
                    <&ArrayType>::try_from(r#type.as_ref()).cloned().map_err(ProgramError::from)
                })
                .collect::<Result<Vec<_>, _>>()?;
            match operation {
                ArrayOperation::ScaledDot(operation) => {
                    let (composition_input_types, composition_output_types) = scaled_dot_composition_types(
                        operation,
                        input_types.as_slice(),
                        output_types.as_slice(),
                        target_platform,
                    )?;
                    map.register(
                        scaled_dot_composition_key(
                            operation,
                            composition_input_types.as_slice(),
                            composition_output_types.as_slice(),
                        )?,
                        || trace_scaled_dot_composition(operation, composition_input_types.as_slice()),
                    )?;
                }
                ArrayOperation::DotProductAttention(operation) if operation.configuration().dropout().is_none() => {
                    map.register(
                        attention_composition_key(operation, input_types.as_slice(), output_types.as_slice()),
                        || trace_attention_composition(operation, input_types.as_slice()),
                    )?;
                }
                ArrayOperation::DotProductAttentionBackward(operation)
                    if operation.configuration().dropout().is_none() =>
                {
                    map.register(
                        attention_backward_composition_key(operation, input_types.as_slice(), output_types.as_slice()),
                        || trace_attention_backward_composition(operation, input_types.as_slice()),
                    )?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    let mut map = NamedCompositionFunctionMap::default();
    walk(program.entry_region_ref(), &mut map, &mut HashSet::new(), target_platform)?;
    Ok(map)
}

/// Emits one private named-composition decomposition function.
fn emit_named_composition_function<'b, 'c: 'b, 't: 'c>(
    module_block: &mut BlockRef<'b, 'c, 't>,
    function: &NamedCompositionFunction,
    nested_functions: &Rc<JitCallFunctionMap>,
    collective_state: &CollectiveLoweringState,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), LoweringError> {
    let input_types = function.program.input_types();
    let output_types = function.program.output_types();
    let argument_tensor_types = input_types
        .iter()
        .map(|r#type| composite::lower_array_ir_type(r#type, context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let result_tensor_types = output_types
        .iter()
        .map(|r#type| composite::lower_array_ir_type(r#type, context, location))
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
        let input_values = (0..input_types.len())
            .map(|index| function_block.argument(index).unwrap().as_ref())
            .collect::<Vec<_>>();
        let mut effect_tokens = EffectTokens::default();
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
            &mut effect_tokens,
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

    /// Flat input types, which together with the rendering pin the full callee signature.
    input_types: Vec<ArrayIrType>,

    /// Flat output types, completing the callee signature when output-only placement metadata is not visible
    /// in the rendered body.
    output_types: Vec<ArrayIrType>,
}

/// Returns whether `program` may be deduplicated by structural identity.
///
/// A program is eligible only when the canonical rendering captures the full semantics needed to safely share a
/// private function. Although immediate constant payloads are rendered, a [`CaptureReference`] identifies a slot and
/// type rather than the runtime value in the surrounding capture table, so programs with constants remain ineligible.
/// `shard_map` also remains outside this deduplication path because its lowering owns a separate outlined boundary.
fn supports_structural_dedup(program: &FlatXlaProgram) -> bool {
    program.regions().iter().all(|region| {
        region.atoms().iter().all(|atom| !atom.is_constant())
            && region
                .instructions()
                .iter()
                .all(|instruction| !matches!(instruction.operation(), XlaOperation::ShardMap(_)))
    })
}

/// Returns whether a borrowed rooted region may be deduplicated by structural identity.
fn supports_structural_dedup_region(region: RegionRef<'_, XlaConstant, XlaOperation>) -> bool {
    fn walk(region: RegionRef<'_, XlaConstant, XlaOperation>, visited: &mut HashSet<RegionId>) -> bool {
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
fn jit_call_program_key(program: &FlatXlaProgram) -> Option<JitCallProgramKey> {
    supports_structural_dedup(program).then(|| JitCallProgramKey {
        rendered: program.to_string(),
        input_types: program.input_types(),
        output_types: program.output_types(),
    })
}

/// Computes the deduplication key for a borrowed callee region, materializing only after borrowed eligibility checks.
fn jit_call_region_key(region: RegionRef<'_, XlaConstant, XlaOperation>) -> Option<JitCallProgramKey> {
    supports_structural_dedup_region(region).then(|| {
        let program = region.to_program();
        JitCallProgramKey {
            rendered: program.to_string(),
            input_types: program.input_types(),
            output_types: program.output_types(),
        }
    })
}

/// One deduplicated callee emitted as a shared private `func.func`.
struct JitCallFunction {
    /// Symbol name of the emitted private function.
    symbol: String,

    /// Representative callee program for this key (materialized from its callee region), lowered once as the
    /// function body.
    program: FlatXlaProgram,

    /// Flat input types of the callee, also the emitted function's argument types.
    input_types: Vec<ArrayIrType>,

    /// Flat output types of the callee, also the emitted function's result types.
    output_types: Vec<ArrayIrType>,
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
    fn get(&self, program: &FlatXlaProgram) -> Option<&JitCallFunction> {
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
/// target platform that gates platform-specific fast paths such as fused attention — reaches the
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
        .map(|r#type| composite::lower_array_ir_type(r#type, context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let result_tensor_types = function
        .output_types
        .iter()
        .map(|r#type| composite::lower_array_ir_type(r#type, context, location))
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
        // so the shared function body never creates an effect token.
        let mut effect_tokens = EffectTokens::default();
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
            &mut effect_tokens,
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
/// `input_values` are the lowered call operands in callee-input order. `capture_count` is the operation payload's
/// exact leading lifted-capture prefix length; deriving it from capture-constant indices instead would conflate the
/// independent capture namespaces of nested calls inside the callee arena.
fn lower_jit_call<'b, 'c: 'b, 't: 'c>(
    program: &FlatXlaProgram,
    capture_count: usize,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    // Only pure callees are ever deduplicated (`collect_jit_call_functions` skips effectful programs), so the
    // shared-function path below never interacts with the caller's effect tokens; effectful callees always take the
    // inline path, which threads the caller's per-class tokens through the callee body in program order.
    if let Some(map) = nested_functions {
        if let Some(function) = map.get(program) {
            // The `jit_call` operation's type inference already pins its operands to the callee input types, so a
            // matching arity is the only guard needed before emitting the symbol call; anything else inlines.
            if input_values.len() == function.input_types.len() {
                let result_tensor_types = function
                    .output_types
                    .iter()
                    .map(|r#type| composite::lower_array_ir_type(r#type, context, location))
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
    if input_values.len() < capture_count {
        return Err(LoweringError::MissingCapturedConstant { index: capture_count.saturating_sub(1) });
    }
    let captured_values = input_values[..capture_count].to_vec();
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
        effect_tokens,
    )
}

/// Inlines a nested sub-program into the given block by mapping the provided input
/// MLIR values to the body's input atoms, lowering constants and instructions in topological
/// order, and returning lowered values corresponding to the program's output atoms.
///
/// `effect_tokens` are the per-class tokens of the lowering scope the program inlines into. They flow into each
/// instruction's lowerer and the updated tokens are read back out after the instruction lowers, so same-class effects
/// chain in program order and the caller observes the program's final tokens.
#[allow(clippy::too_many_arguments)]
fn lower_nested_program_inline<'b, 'c: 'b, 't: 'c>(
    program: &FlatXlaProgram,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    captured_values: &[ValueRef<'b, 'c, 't>],
    add_optimization_barrier: bool,
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
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
        effect_tokens,
    )
}

/// Inlines a borrowed nested region into the given block without materializing the region itself.
#[allow(clippy::too_many_arguments)]
fn lower_nested_region_inline<'b, 'c: 'b, 't: 'c>(
    region: RegionRef<'_, XlaConstant, XlaOperation>,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    captured_values: &[ValueRef<'b, 'c, 't>],
    add_optimization_barrier: bool,
    nested_functions: Option<&Rc<JitCallFunctionMap>>,
    collective_state: &CollectiveLoweringState,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    let outputs = replay_region_ref_into_block(
        region,
        input_values.to_vec(),
        block,
        context,
        location,
        |atom_id, value, block, context, location| {
            lower_constant(atom_id, value, captured_values, block, context, location)
        },
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
            let regions = instruction
                .regions()
                .iter()
                .map(|attached| RegionRef::new(region.arena(), *attached).map(RegionRef::to_program))
                .collect::<Result<Vec<_>, ProgramError>>()?;
            // Every StableHLO operation generated from this one Ryft instruction inherits the instruction's
            // provenance through the lowerer's shared location, with the ambient location as its base.
            let location = collective_state.instruction_location(context, instruction.provenance(), location);
            let mut lowerer = ShardMapMlirLowerer::new(*block, context, location)
                .with_input_types(input_types)
                .with_nested_functions(nested_functions.cloned())
                .with_captured_values(captured_values)
                .with_effect_tokens(*effect_tokens)
                .with_collective_state(collective_state.clone());
            let outputs = dispatch_lower_shard_map_mlir(
                instruction.operation(),
                captured_values,
                inputs,
                regions.as_slice(),
                output_types.as_slice(),
                &mut lowerer,
            )?;
            *effect_tokens = lowerer.effect_tokens;
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
fn replay_program_into_block<
    'b,
    'c: 'b,
    't: 'c,
    T: RyftType,
    O,
    V: Value<Type = T>,
    Input,
    Output,
    LiftConstant,
    ApplyOp,
>(
    program: &Program<V, O, Input, Output>,
    input_values: Vec<ValueRef<'b, 'c, 't>>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    mut lift_constant: LiftConstant,
    mut apply_op: ApplyOp,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    O: Operation<Type = T>,
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
    O: Operation<Type = V::Type>,
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
#[cfg(test)]
fn lower_plain_program_outputs<'b, 'c: 'b, 't: 'c, O, V, Input, Output>(
    program: &Program<V, O, Input, Output>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    collective_state: &CollectiveLoweringState,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
where
    V: MlirLowerableValue,
    O: LowerableXlaOperation<V>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    let input_values = (0..program.input_ids().len())
        .map(|index| block.argument(index).expect("body block arguments should exist").as_ref())
        .collect::<Vec<_>>();
    // Function-body-scoped per-class effect chains are created lazily and dropped at the end of the function body.
    let mut effect_tokens = EffectTokens::default();
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
            if !instruction.regions().is_empty() {
                return Err(LoweringError::UnsupportedOp {
                    op: format!(
                        "plain-program lowering does not support attached regions for `{}`; use the production \
                         composite XLA lowerer",
                        instruction.operation().name(),
                    ),
                });
            }
            let location = collective_state.instruction_location(context, instruction.provenance(), location);
            let mut lowerer = PlainMlirLowerer::new(*block, context, location)
                .with_input_types(input_types)
                .with_effect_tokens(effect_tokens)
                .with_collective_state(collective_state.clone());
            let outputs = instruction.operation().lower_to_mlir(
                inputs,
                output_types.as_slice(),
                PlainMlirLoweringMode::Unpacked,
                &mut lowerer,
            )?;
            effect_tokens = lowerer.effect_tokens;
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
    // Function-body-scoped per-class effect chains are created lazily and dropped at the end of the function body.
    let mut effect_tokens = EffectTokens::default();
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
                &mut effect_tokens,
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
        // module's deduplicated functions (which are typed against global shapes) into them. The body is traced
        // through a fresh-root context and lowers with an empty capture namespace; refer to the `CustomJvp` arm of
        // `lower_operation` for the rationale.
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
    let tensor_type = lower_tensor_type(&value_type, context, location)?;
    let elements = value.to_dense_elements_attribute(tensor_type, context)?;
    let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
    let constant = constant.result(0).expect("stablehlo.constant should return one result").as_ref();
    annotate_output_memory(constant, &value_type, block, context, location)
}

/// Lowers one captured constant reference by forwarding its runtime captured value.
fn lower_captured_constant<'b, 'c: 'b, 't: 'c, T: RyftType>(
    value: &CaptureReference<T>,
    captured_values: &[ValueRef<'b, 'c, 't>],
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    captured_values
        .get(value.index())
        .copied()
        .ok_or(LoweringError::MissingCapturedConstant { index: value.index() })
}

/// Lowers one first-class dimension extent to the scalar `i64` [`stablehlo.constant`](stable_hlo::constant) that
/// represents a dimension in StableHLO. Immediate [`XlaConstant::Dimension`] atoms and staged
/// [`DimensionOperation::Constant`] instructions must agree on this representation, and so both lowering paths route
/// through this function.
fn lower_dimension_extent<'b, 'c: 'b, 't: 'c, B, L>(
    value: &DimensionValue,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    let tensor_type = lower_tensor_type(&ArrayType::scalar(DataType::I64), context, location)?;
    let extent = i64::try_from(value.extent()).map_err(|_| LoweringError::UnsupportedOp {
        op: format!("first-class dimension extent {} does not fit in an i64", value.extent()),
    })?;
    let elements = lower_constant_elements_attribute(DataType::I64, tensor_type, extent, context)?;
    let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
    Ok(constant.result(0).expect("stablehlo.constant should return one result").as_ref())
}

/// Lowers a traced constant atom to a StableHLO value and returns it. A captured constant forwards the hidden capture
/// argument that carries its runtime value, while an immediate dimension extent is materialized in place — which is
/// exactly what makes extents usable inside a `shard_map` manual region, where no capture table is reachable.
fn lower_constant<'b, 'c: 'b, 't: 'c, B, L>(
    _atom_id: AtomId,
    value: &XlaConstant,
    captured_values: &[ValueRef<'b, 'c, 't>],
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    match value {
        XlaConstant::Captured(value) => lower_captured_constant(value, captured_values),
        XlaConstant::Dimension(value) => lower_dimension_extent(value, block, context, location),
    }
}

/// Lowers one production composite XLA operation while preserving the enclosing lowering state.
fn dispatch_lower_shard_map_mlir<'b, 'c: 'b, 't: 'c>(
    operation: &XlaOperation,
    captured_values: &[ValueRef<'b, 'c, 't>],
    input_values: &[ValueRef<'b, 'c, 't>],
    regions: &[FlatXlaProgram],
    output_types: &[ArrayIrType],
    lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
    if let Some(operation) = operation.to_core_operation() {
        return composite::lower_array_ir_operation(
            &operation,
            input_values,
            lowerer.input_types.as_slice(),
            output_types,
            &lowerer.collective_state,
            &mut lowerer.effect_tokens,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        );
    }

    match operation {
        XlaOperation::Condition(_) => lowerer.lower_condition(regions, input_values),
        XlaOperation::While(operation) => lowerer.lower_while(operation, regions, input_values),
        XlaOperation::Scan(operation) => lowerer.lower_scan(operation, regions, input_values),
        XlaOperation::CustomJvp(_) => {
            let [primal, _jvp] = regions else {
                return Err(LoweringError::UnsupportedOp {
                    op: format!("{} expected 2 attached regions but got {}", CUSTOM_JVP_OPERATION_NAME, regions.len(),),
                });
            };
            // Custom-JVP regions are traced through fresh-root contexts whose local capture tables are discarded, so
            // they can never legally reference the enclosing function's captures (the trace boundary rejects bodies
            // that register captures). Lowering them with an empty capture namespace turns any capture-referencing
            // constant that still sneaks in into a loud `MissingCapturedConstant` error instead of silently aliasing
            // whatever value occupies the referenced slot of the enclosing capture prefix. Nested-traced regions
            // (`while`/`scan`/`condition`/`linear_call`) share the enclosing capture scope and keep inheriting it.
            lower_nested_program_inline(
                primal,
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
                &[],
                false,
                lowerer.nested_functions.as_ref(),
                &lowerer.collective_state,
                &mut lowerer.effect_tokens,
            )
        }
        XlaOperation::CustomVjp(_) => {
            let [primal, _forward, _backward] = regions else {
                return Err(LoweringError::UnsupportedOp {
                    op: format!("{} expected 3 attached regions but got {}", CUSTOM_VJP_OPERATION_NAME, regions.len(),),
                });
            };
            // Custom-VJP regions are traced through fresh-root contexts and lower with an empty capture namespace;
            // refer to the `CustomJvp` arm above for the rationale.
            lower_nested_program_inline(
                primal,
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
                &[],
                false,
                lowerer.nested_functions.as_ref(),
                &lowerer.collective_state,
                &mut lowerer.effect_tokens,
            )
        }
        XlaOperation::Rematerialize(_) => {
            let [primal, _forward, _backward, _tangent] = regions else {
                return Err(LoweringError::UnsupportedOp {
                    op: format!(
                        "{} expected 4 attached regions but got {}",
                        REMATERIALIZE_OPERATION_NAME,
                        regions.len(),
                    ),
                });
            };
            // Rematerialized regions are traced through fresh-root contexts and lower with an empty capture
            // namespace; refer to the `CustomJvp` arm above for the rationale.
            lower_nested_program_inline(
                primal,
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
                &[],
                false,
                lowerer.nested_functions.as_ref(),
                &lowerer.collective_state,
                &mut lowerer.effect_tokens,
            )
        }
        XlaOperation::LinearCall(operation) => {
            if operation.is_transpose_only() {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!("operation `{}` cannot be lowered to StableHLO", operation.name()),
                }
                .into());
            }
            let [forward, _transpose] = regions else {
                return Err(LoweringError::UnsupportedOp {
                    op: format!("linear_call expected 2 attached regions but got {}", regions.len()),
                });
            };
            lower_nested_program_inline(
                forward,
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
                captured_values,
                false,
                lowerer.nested_functions.as_ref(),
                &lowerer.collective_state,
                &mut lowerer.effect_tokens,
            )
        }
        XlaOperation::JitCall(operation) => {
            let [callee] = regions else {
                return Err(LoweringError::UnsupportedOp {
                    op: format!("jit_call expected 1 attached region but got {}", regions.len()),
                });
            };
            lower_jit_call(
                callee,
                operation.capture_count(),
                input_values,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
                lowerer.nested_functions.as_ref(),
                &lowerer.collective_state,
                &mut lowerer.effect_tokens,
            )
        }
        XlaOperation::ShardMap(operation) => {
            let [body] = regions else {
                return Err(LoweringError::UnsupportedOp {
                    op: format!("{} expected 1 attached region but got {}", SHARD_MAP_OPERATION_NAME, regions.len(),),
                });
            };
            // Only ordered effects need token threading, which `sdy.manual_computation` cannot express; a body
            // whose effects are all unordered lowers without any token state.
            if body.effects().is_ordered() {
                return Err(LoweringError::EffectfulShardMapBody);
            }
            let simplified_body = body
                .simplified()
                .map_err(|error| LoweringError::SimplificationFailure { message: error.to_string() })?;
            let local_input_types = simplified_body
                .input_types()
                .iter()
                .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
                .collect::<Result<Vec<_>, _>>()
                .map_err(ProgramError::from)?;
            lowerer.lower_manual_computation(
                input_values,
                operation.shard_map(),
                &simplified_body,
                local_input_types.as_slice(),
                operation.global_output_types(),
            )
        }
        _ => unreachable!("member and mixed operations are handled by the canonical core operation family"),
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
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
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
    // Every StableHLO operation generated from this one Ryft instruction inherits the instruction's provenance
    // through the lowerer's shared location, with the ambient location as its base.
    let location = collective_state.instruction_location(context, instruction.provenance(), location);
    let mut lowerer = ShardMapMlirLowerer::new(*block, context, location)
        .with_input_types(input_types)
        .with_nested_functions(nested_functions.cloned())
        .with_captured_values(captured_values)
        .with_effect_tokens(*effect_tokens)
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
    *effect_tokens = lowerer.effect_tokens;
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
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "collective over axis `{axis_name}` can only be lowered inside a \
                     {SHARD_MAP_OPERATION_NAME} manual region",
            ),
        }
        .into());
    };
    let mesh = shard_map.mesh();
    if !shard_map.manual_axes().iter().any(|manual_axis| manual_axis == axis_name) {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "collective over axis `{axis_name}` cannot lower inside this shard_map manual region because the \
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
            "collective over axis `{axis_name}` records size {axis_size}, but the enclosing mesh axis has size \
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
                                "collective over axis `{axis_name}` has group member {index} outside axis size \
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
                    "grouped `{}` over axis `{}` does not record the full axis size",
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
            message: format!(
                "{AXIS_INDEX_OPERATION_NAME} for axis `{axis_name}` can only be lowered inside a \
                 {SHARD_MAP_OPERATION_NAME} manual region",
            ),
        }
        .into());
    };
    let mesh = shard_map.mesh();
    if !shard_map.manual_axes().iter().any(|manual_axis| manual_axis == axis_name) {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "{AXIS_INDEX_OPERATION_NAME} for axis `{axis_name}` cannot lower inside this \
                 {SHARD_MAP_OPERATION_NAME} manual region because the region does not bind that axis as a manual mesh \
                 axis",
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

/// Lowers one JAX-compatible minimum or maximum. Integer and Boolean values use the corresponding StableHLO
/// operation. Floating-point values use explicit total-order comparisons and NaN propagation so every XLA backend
/// preserves IEEE signed-zero and NaN semantics. Complex values use JAX's lexicographic `(real, imaginary)`
/// compare/select sequence.
fn lower_extremum_to_mlir<'b, 'c: 'b, 't: 'c>(
    maximum: bool,
    element_type: DataType,
    left: ValueRef<'b, 'c, 't>,
    right: ValueRef<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    if !element_type.is_complex() && !element_type.is_floating_point() {
        let result = if maximum {
            block.append_operation(stable_hlo::maximum(left, right, location)?)?
        } else {
            block.append_operation(stable_hlo::minimum(left, right, location)?)?
        };
        return Ok(result.result(0).expect("stablehlo extremum should return one result").as_ref());
    }
    if element_type.is_floating_point() {
        let direction = if maximum {
            stable_hlo::ComparisonDirection::GreaterThan
        } else {
            stable_hlo::ComparisonDirection::LessThan
        };
        let ordered = block.append_operation(stable_hlo::compare(
            left,
            right,
            direction,
            stable_hlo::ComparisonType::TotalOrder,
            location,
        )?)?;
        let ordered = ordered.result(0).expect("stablehlo.compare should return one result").as_ref();
        let result = block.append_operation(stable_hlo::select(ordered, left, right, location)?)?;
        let result = result.result(0).expect("stablehlo.select should return one result").as_ref();

        // Some XLA backends implement native min/max using instructions that drop NaNs or do not distinguish signed
        // zeros. StableHLO and JAX require IEEE maximum/minimum, so make both behaviors explicit and portable.
        let left_is_nan = block.append_operation(stable_hlo::compare(
            left,
            left,
            stable_hlo::ComparisonDirection::NotEqual,
            stable_hlo::ComparisonType::Float,
            location,
        )?)?;
        let left_is_nan = left_is_nan.result(0).expect("stablehlo.compare should return one result").as_ref();
        let right_is_nan = block.append_operation(stable_hlo::compare(
            right,
            right,
            stable_hlo::ComparisonDirection::NotEqual,
            stable_hlo::ComparisonType::Float,
            location,
        )?)?;
        let right_is_nan = right_is_nan.result(0).expect("stablehlo.compare should return one result").as_ref();
        let right_or_result = block.append_operation(stable_hlo::select(right_is_nan, right, result, location)?)?;
        let right_or_result = right_or_result.result(0).expect("stablehlo.select should return one result").as_ref();
        let result = block.append_operation(stable_hlo::select(left_is_nan, left, right_or_result, location)?)?;
        return Ok(result.result(0).expect("stablehlo.select should return one result").as_ref());
    }

    let left_real = block.append_operation(stable_hlo::real(left, location)?)?;
    let left_real = left_real.result(0).expect("stablehlo.real should return one result").as_ref();
    let right_real = block.append_operation(stable_hlo::real(right, location)?)?;
    let right_real = right_real.result(0).expect("stablehlo.real should return one result").as_ref();
    let left_imaginary = block.append_operation(stable_hlo::imag(left, location)?)?;
    let left_imaginary = left_imaginary.result(0).expect("stablehlo.imag should return one result").as_ref();
    let right_imaginary = block.append_operation(stable_hlo::imag(right, location)?)?;
    let right_imaginary = right_imaginary.result(0).expect("stablehlo.imag should return one result").as_ref();
    let real_equal = block.append_operation(stable_hlo::compare(
        left_real,
        right_real,
        stable_hlo::ComparisonDirection::Equal,
        stable_hlo::ComparisonType::Float,
        location,
    )?)?;
    let real_equal = real_equal.result(0).expect("stablehlo.compare should return one result").as_ref();
    let direction =
        if maximum { stable_hlo::ComparisonDirection::GreaterThan } else { stable_hlo::ComparisonDirection::LessThan };
    let real_ordered = block.append_operation(stable_hlo::compare(
        left_real,
        right_real,
        direction,
        stable_hlo::ComparisonType::Float,
        location,
    )?)?;
    let real_ordered = real_ordered.result(0).expect("stablehlo.compare should return one result").as_ref();
    let imaginary_ordered = block.append_operation(stable_hlo::compare(
        left_imaginary,
        right_imaginary,
        direction,
        stable_hlo::ComparisonType::Float,
        location,
    )?)?;
    let imaginary_ordered = imaginary_ordered.result(0).expect("stablehlo.compare should return one result").as_ref();
    let ordered = block.append_operation(stable_hlo::select(real_equal, imaginary_ordered, real_ordered, location)?)?;
    let ordered = ordered.result(0).expect("stablehlo.select should return one result").as_ref();
    let result = block.append_operation(stable_hlo::select(ordered, left, right, location)?)?;
    Ok(result.result(0).expect("stablehlo.select should return one result").as_ref())
}

/// Builds a reduction-body region for [`stable_hlo::reduce`] over the given scalar `element_type`. The generated
/// region has one block taking two scalar tensor arguments of `tensor<{element_type}>` and produces one scalar result
/// through the combiner matching the reduction kind.
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
    let body_value = match kind {
        ReductionKind::Sum | ReductionKind::Mean => block_ref
            .append_operation(stable_hlo::add(lhs, rhs, location)?)?
            .result(0)
            .expect("stablehlo.add should return one result")
            .as_ref(),
        ReductionKind::Max | ReductionKind::Min => {
            lower_extremum_to_mlir(kind == ReductionKind::Max, element_type, lhs, rhs, &mut block_ref, location)?
        }
        ReductionKind::Any => block_ref
            .append_operation(stable_hlo::or(lhs, rhs, location)?)?
            .result(0)
            .expect("stablehlo.or should return one result")
            .as_ref(),
        ReductionKind::All => block_ref
            .append_operation(stable_hlo::and(lhs, rhs, location)?)?
            .result(0)
            .expect("stablehlo.and should return one result")
            .as_ref(),
    };
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
        op: format!("`reduce` operand has non-tensor MLIR type `{input_type}`"),
    })?;
    let dimensions = input_tensor_type.dimensions().collect::<Vec<_>>();
    let mut count = 1usize;
    for axis in axes {
        match dimensions.get(*axis) {
            Some(MlirSize::Static(size)) => count *= size,
            _ => {
                return Err(LoweringError::UnsupportedOp {
                    op: format!("`reduce` mean over dynamically sized axis {axis}"),
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
        return Err(LoweringError::UnsupportedOp {
            op: format!("{} with mode {}", GATHER_OPERATION_NAME, operation.mode()),
        });
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
/// returns the combined value: `Overwrite` returns the update directly (no combine op), arithmetic kinds apply the
/// matching elementwise StableHLO operation, and extrema use [`lower_extremum_to_mlir`].
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
        ScatterReductionKind::Min | ScatterReductionKind::Max => {
            lower_extremum_to_mlir(kind == ScatterReductionKind::Max, element_type, lhs, rhs, &mut block_ref, location)?
        }
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
        return Err(LoweringError::UnsupportedOp {
            op: format!("{} with mode {}", SCATTER_OPERATION_NAME, operation.mode()),
        });
    }
    let indices_rank = input_values[1]
        .r#type()?
        .cast::<TensorTypeRef>()
        .ok_or_else(|| LoweringError::UnsupportedOp {
            op: format!("{SCATTER_OPERATION_NAME} with non-tensor indices"),
        })?
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
    if element_type.is_complex() {
        let real = match kind {
            ReductionKind::Sum | ReductionKind::Mean => 0.0,
            ReductionKind::Max => f64::NEG_INFINITY,
            ReductionKind::Min => f64::INFINITY,
            ReductionKind::Any | ReductionKind::All => {
                return Err(LoweringError::UnsupportedDataType { data_type: element_type });
            }
        };
        let part_data_type = if element_type == DataType::C64 { DataType::F32 } else { DataType::F64 };
        let part_type = ArrayType::scalar(part_data_type);
        let part_tensor_type = lower_tensor_type(&part_type, context, location)?;
        let real_value = lower_f64_constant_splat(real, &part_type, part_tensor_type, block, context, location)?;
        let imaginary_value = lower_f64_constant_splat(0.0, &part_type, part_tensor_type, block, context, location)?;
        let complex = block.append_operation(stable_hlo::complex(real_value, imaginary_value, location)?)?;
        return Ok(complex.result(0).expect("stablehlo.complex should return one result").as_ref());
    }
    let attribute = build_reduction_identity_attribute(kind, element_type, scalar_tensor_type, context)?;
    let result = block.append_operation(stable_hlo::constant(attribute, location)?)?;
    Ok(result.result(0).expect("stablehlo.constant should return one result").as_ref())
}

/// Builds a dense-elements attribute holding the identity element of the given reduction kind at
/// the given element type. `Sum` and `Mean` use zero; `Max` and `Min` use the bounds returned by
/// [`float_reduction_identity_bounds`] at float element types and the bounds returned by
/// [`integer_reduction_identity_bounds`] at integer element types. Boolean `Any`/`Max` use `false`, while Boolean
/// `All`/`Min` use `true`. Every other combination fails with [`LoweringError::UnsupportedDataType`].
fn build_reduction_identity_attribute<'c, 't>(
    kind: ReductionKind,
    element_type: DataType,
    tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
    context: &'c MlirContext<'t>,
) -> Result<DenseElementsAttributeRef<'c, 't>, LoweringError> {
    if let Some((minimum, maximum)) = float_reduction_identity_bounds(element_type) {
        let identity = match kind {
            ReductionKind::Sum | ReductionKind::Mean if element_type != DataType::F8E8M0FNU => 0.0,
            ReductionKind::Max => minimum,
            ReductionKind::Min => maximum,
            ReductionKind::Sum | ReductionKind::Mean => {
                return Err(LoweringError::UnsupportedDataType { data_type: element_type });
            }
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
        (ReductionKind::Any | ReductionKind::Max, DataType::Boolean) => context
            .dense_bool_elements_attribute(tensor_type, &[false])
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: element_type })?
            .cast::<DenseElementsAttributeRef>()
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type: element_type }),
        (ReductionKind::All | ReductionKind::Min, DataType::Boolean) => context
            .dense_bool_elements_attribute(tensor_type, &[true])
            .map_err(|_| LoweringError::InvalidDenseElementsAttribute { data_type: element_type })?
            .cast::<DenseElementsAttributeRef>()
            .ok_or(LoweringError::InvalidDenseElementsAttribute { data_type: element_type }),
        _ => Err(LoweringError::UnsupportedDataType { data_type: element_type }),
    }
}

/// Returns the `(maximum identity, minimum identity)` pair for the given floating-point data type. Formats with
/// infinities use `(-inf, +inf)`, finite signed formats use their finite bounds, and [`DataType::F8E8M0FNU`] uses its
/// smallest and largest positive powers of two. Returns `None` for non-floating-point data types.
fn float_reduction_identity_bounds(data_type: DataType) -> Option<(f64, f64)> {
    match data_type {
        DataType::BF16
        | DataType::F16
        | DataType::F32
        | DataType::F64
        | DataType::F8E3M4
        | DataType::F8E4M3
        | DataType::F8E5M2 => Some((f64::NEG_INFINITY, f64::INFINITY)),
        DataType::F4E2M1FN => Some((-6.0, 6.0)),
        DataType::F6E2M3FN => Some((-7.5, 7.5)),
        DataType::F6E3M2FN => Some((-28.0, 28.0)),
        DataType::F8E4M3FN => Some((-448.0, 448.0)),
        DataType::F8E4M3FNUZ => Some((-240.0, 240.0)),
        DataType::F8E4M3B11FNUZ => Some((-30.0, 30.0)),
        DataType::F8E5M2FNUZ => Some((-57344.0, 57344.0)),
        DataType::F8E8M0FNU => Some((2f64.powi(-127), 2f64.powi(127))),
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
        // StableHLO admits signless `i1` as its Boolean/one-bit carrier but rejects `ui1`. U1's unsigned semantics
        // remain selected by Ryft operation metadata (e.g., comparison direction), while its sole bit is unchanged.
        DataType::U1 => context.signless_integer_type(1).as_ref(),
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
    use std::ops::{Deref, DerefMut};

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use ryft_mlir::ElementsAttribute;
    use ryft_mlir::dialects::builtin::attributes::DenseElementsAttribute;

    use ryft_core::operations::attention::{
        AttentionConfiguration, AttentionImplementation, AttentionOperandSignature,
    };
    use ryft_core::operations::random::{RandomAlgorithm, RngBitGeneratorOperation};
    use ryft_core::{
        AndOperation, Array as CpuArray, ArrayOperation, Atan2Operation, BroadcastOperation, CompareOperation,
        ConcatenateOperation, ConditionOperation, ConstantOperation, Context, Cos, Differentiate, Dimension,
        DimensionAddOperation, DimensionBounds, DimensionOperation, DimensionSizeOperation, DimensionType,
        DimensionVariable, DivOperation, Dot, DotDimensionNumbers, DynamicBroadcastOperation, DynamicSliceOperation,
        DynamicUpdateSliceOperation, EagerContext, Fill, LogicalMesh, MeshAxis, MeshAxisType, OneLike,
        OneLikeOperation, OneOperation, OrOperation, PadOperation, Placeholder, ProgramBuilder, Provenance,
        ProvenanceScope, ReduceOperation, ReshapeOperation, ReverseModeDifferentiate, ScanOperation, SelectOperation,
        Shape, Sharding, ShardingDimension, Sin, SliceOperation, Transpose, TypeError, UpdateSliceOperation,
        WhileOperation, XorOperation, ZeroLike, ZeroOperation, i1, i2, i4, u1, u2, u4,
    };

    use super::super::shard_map::{TracedShardMap, shard_map as traced_shard_map};
    use ryft_core::{Trace, TracingContext};

    use crate::tests::values_to_bytes;

    use super::*;

    /// Homogeneous builder retained only for tests of the test-only plain-program lowering helper.
    type XlaProgramBuilder = ProgramBuilder<XlaArrayConstant, ArrayOperation<XlaArrayConstant>>;

    /// Homogeneous program retained only for tests of the test-only plain-program lowering helper.
    type PlainXlaProgram =
        Program<XlaArrayConstant, ArrayOperation<XlaArrayConstant>, Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>;

    fn attention_operation(configuration: AttentionConfiguration) -> DotProductAttentionOperation {
        DotProductAttentionOperation::new(configuration, AttentionOperandSignature::default())
    }

    fn attention_backward_operation(configuration: AttentionConfiguration) -> DotProductAttentionBackwardOperation {
        DotProductAttentionBackwardOperation::new(configuration, AttentionOperandSignature::default())
    }

    /// Array-oriented facade over the production composite program builder.
    struct CompositeXlaProgramBuilder(crate::experimental::ops::XlaProgramBuilder);

    impl CompositeXlaProgramBuilder {
        /// Creates an empty production composite program builder.
        fn new() -> Self {
            Self(crate::experimental::ops::XlaProgramBuilder::new())
        }

        /// Adds an array input while lifting its descriptor into the composite type universe.
        fn add_input(&mut self, r#type: ArrayType) -> AtomId {
            self.0.add_input(ArrayIrType::Array(r#type))
        }

        /// Finalizes the composite program.
        fn build<Input: Parameterized<XlaConstant>, Output: Parameterized<XlaConstant>>(
            self,
            output_ids: Vec<AtomId>,
            input_structure: Input::ParameterStructure,
            output_structure: Output::ParameterStructure,
        ) -> Result<XlaProgram<Input, Output>, ProgramError> {
            self.0.build(output_ids, input_structure, output_structure)
        }
    }

    impl Deref for CompositeXlaProgramBuilder {
        type Target = crate::experimental::ops::XlaProgramBuilder;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl DerefMut for CompositeXlaProgramBuilder {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    fn test_manual_mesh(axis_name: &str, axis_size: usize) -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::new(axis_name, axis_size, MeshAxisType::Manual).unwrap()]).unwrap()
    }

    fn test_vector_type(length: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(length)]))
    }

    fn test_matrix_type(rows: usize, cols: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(rows), Dimension::Static(cols)]))
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

    fn xla_identity_branch(input_type: ArrayType) -> PlainXlaProgram {
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn xla_neg_branch(input_type: ArrayType) -> PlainXlaProgram {
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let output = builder.add_instruction(NegOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Lifts an array-only test fixture into the production composite XLA program representation.
    fn unproject_plain_program(program: PlainXlaProgram) -> FlatXlaProgram {
        program.into_unprojected::<XlaConstant, XlaOperation>().unwrap()
    }

    fn lower_traced_module(
        traced: &TracedShardMap<ArrayType, ArrayType>,
        function_name: &str,
    ) -> Result<String, super::super::shard_map::ShardMapTraceError> {
        traced.to_mlir_module(function_name)
    }

    fn xla_elementwise_normalization_program() -> PlainXlaProgram {
        let mut builder = XlaProgramBuilder::new();
        let scalar = builder.add_input(ArrayType::scalar(DataType::F32));
        let left = builder
            .add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Static(1)])));
        let right = builder
            .add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1), Dimension::Static(4)])));
        let condition = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let boolean_vector =
            builder.add_input(ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(4)])));
        let divide = builder.add_instruction(DivOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let atan2 = builder.add_instruction(Atan2Operation::new(), Vec::new(), vec![right, left], None).unwrap()[0];
        let compare = builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                Vec::new(),
                vec![left, right],
                None,
            )
            .unwrap()[0];
        let select = builder
            .add_instruction(SelectOperation::new(), Vec::new(), vec![condition, scalar, right], None)
            .unwrap()[0];
        let and = builder
            .add_instruction(AndOperation::new(), Vec::new(), vec![condition, boolean_vector], None)
            .unwrap()[0];
        let or = builder
            .add_instruction(OrOperation::new(), Vec::new(), vec![boolean_vector, condition], None)
            .unwrap()[0];
        let xor = builder
            .add_instruction(XorOperation::new(), Vec::new(), vec![condition, boolean_vector], None)
            .unwrap()[0];
        builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
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
        let mut builder = ryft_core::ProgramBuilder::<CpuArray, BroadcastOperation>::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(BroadcastOperation::new(output_type, vec![0]), Vec::new(), vec![input], None)
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
    fn test_plain_broadcast_with_dynamic_result_type_lowers_to_bounded_broadcast_in_dim() {
        // Homogeneous broadcast admits a dynamic output extent that is identity-equal to the mapped input extent,
        // which is exactly the payload that program batching stages for a dynamic per-item dimension. Lowering keeps
        // that extent dynamic in the result tensor type and supplies no extent operand, so `stablehlo.broadcast_in_dim`
        // verifies only when the dynamic axis carries an upper bound that lowers to a `#stablehlo.bounds` encoding.
        let bounded = dynamic_dimension("n", Some(5));
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![bounded.clone()]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), bounded]));
        let mut builder = ryft_core::ProgramBuilder::<CpuArray, BroadcastOperation>::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(BroadcastOperation::new(output_type, vec![1]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<CpuArray>, Vec<CpuArray>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<?xf32, #stablehlo.bounds<4>>) -> tensor<2x?xf32, #stablehlo.bounds<?, 4>> {
                    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<?xf32, #stablehlo.bounds<4>>) -> tensor<2x?xf32, #stablehlo.bounds<?, 4>>
                    return %0 : tensor<2x?xf32, #stablehlo.bounds<?, 4>>
                  }
                }
            "#},
        );

        // An unbounded dynamic result has no bounded form to lower into, so the module fails StableHLO verification
        // rather than producing an unverified graph.
        let unbounded = dynamic_dimension("n", None);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![unbounded.clone()]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), unbounded]));
        let mut builder = ryft_core::ProgramBuilder::<CpuArray, BroadcastOperation>::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(BroadcastOperation::new(output_type, vec![1]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<CpuArray>, Vec<CpuArray>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(to_mlir_module_for_plain_program(&program, "main"), Err(LoweringError::MlirVerificationFailure));
    }

    #[test]
    fn test_plain_reshape_dimensions_lower_transpose_before_reshape() {
        let input_type = test_matrix_type(2, 3);
        let mut builder = ryft_core::ProgramBuilder::<CpuArray, ReshapeOperation>::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(
                ReshapeOperation::new(
                    ReshapeParameters::new(Shape::new(vec![Dimension::Static(6)])).with_dimensions([1, 0]),
                ),
                Vec::new(),
                vec![input],
                None,
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
        let mut builder = ryft_core::ProgramBuilder::<CpuArray, ReshapeOperation>::new();
        let input = builder.add_input(ArrayType::new(DataType::F32, shape.clone()));
        let output = builder.add_instruction(ReshapeOperation::new(shape), Vec::new(), vec![input], None).unwrap()[0];
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
        let mut builder = ryft_core::ProgramBuilder::<CpuArray, ReshapeOperation>::new();
        let input = builder.add_input(ArrayType::new(DataType::F32, input_shape));
        let output = builder
            .add_instruction(
                ReshapeOperation::new(ReshapeParameters::new(output_shape).with_dimensions([2, 0, 1])),
                Vec::new(),
                vec![input],
                None,
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
            &ReshapeOperation::new(Shape::new(vec![Dimension::Static(4)])),
            &[],
            &[test_vector_type(4)],
            &mut block,
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
                ReshapeOperation::new(
                    ReshapeParameters::new(Shape::new(vec![Dimension::Static(6)])).with_dimensions([1, 0]),
                ),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
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
        let mut builder = ryft_core::ProgramBuilder::<CpuArray, ReshapeOperation>::new();
        let input = builder.add_input(test_vector_type(4));
        let output = builder
            .add_instruction(
                ReshapeOperation::new(
                    ReshapeParameters::new(Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]))
                        .with_output_sharding(output_sharding),
                ),
                Vec::new(),
                vec![input],
                None,
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
        let mut builder = ryft_core::ProgramBuilder::<CpuArray, BroadcastOperation>::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(BroadcastOperation::new(output_type, vec![0]), Vec::new(), vec![input], None)
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
            .add_instruction(BroadcastOperation::new(output_type.clone(), vec![0]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .into_unprojected::<XlaConstant, XlaOperation>()
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
    fn test_composite_lowering_rejects_unresolved_reference_type() {
        use ryft_core::ReferenceType;

        let context = MlirContext::new();
        let location = context.unknown_location();
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        assert!(matches!(
            composite::lower_array_ir_type(&reference_type, &context, location),
            Err(LoweringError::UnresolvedReference { construct }) if construct == reference_type.to_string(),
        ));
    }

    #[test]
    fn test_xla_lowering_rejects_unresolved_reference_state_before_token_threading() {
        use ryft_core::{FreezeReferenceOperation, NewReferenceOperation};

        assert_eq!(token_threaded_effects(Effects::single(Effect::OrderedState)).next(), None);

        // The artifact-wide reference scan catches the reference atoms and intrinsic semantics before token threading,
        // so two representative reference operations are enough to pin the diagnostic.
        let array_type = ArrayType::scalar(DataType::F32);
        let mut builder = crate::experimental::ops::XlaProgramBuilder::new();
        let input = builder.add_input(ArrayIrType::Array(array_type.clone()));
        let reference = builder
            .add_instruction(XlaOperation::NewReference(NewReferenceOperation::new()), Vec::new(), vec![input], None)
            .unwrap()[0];
        let output = builder
            .add_instruction(
                XlaOperation::FreezeReference(FreezeReferenceOperation::new()),
                Vec::new(),
                vec![reference],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // The reference scan runs before the generic state scan at the module entry, so a program whose state comes
        // from reference operations receives the dedicated reference diagnostic; the state error remains behind it
        // for future non-reference state (`token_threaded_effects` above pins that `OrderedState` has no token slot
        // either way).
        assert!(matches!(
            lower_mlir_module_for_program(
                &program,
                &[],
                &vec![array_type.clone()],
                &vec![array_type],
                "main",
                None,
                None,
                None,
            ),
            Err(LoweringError::UnresolvedReference { construct })
                if construct == "program with unresolved references",
        ));
    }

    #[test]
    fn test_reference_state_aliases_use_physical_indices_and_preserve_shardings() {
        use ryft_core::ReferenceSource;

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let state_type = ArrayType::new_static(DataType::F32, [4]);
        let mut builder = crate::experimental::ops::XlaProgramBuilder::new();
        let _zero = builder.add_input(ArrayIrType::Array(zero_type.clone()));
        let mutated = builder.add_input(ArrayIrType::Array(state_type.clone()));
        let _read_only = builder.add_input(ArrayIrType::Array(state_type.clone()));
        let program: FlatXlaProgram = builder.build(vec![mutated], vec![Placeholder; 3], vec![Placeholder]).unwrap();
        let input_types = vec![zero_type.clone(), state_type.clone(), state_type.clone()];
        let output_types = vec![state_type.clone()];
        let argument_shardings = vec![
            Sharding::replicated(mesh.clone(), 1),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
        ];
        let result_shardings = vec![Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()];
        let states = [
            ReferenceStateBinding::new(ReferenceSource::PublicInput { index: 1 }, 1, Some(0)),
            ReferenceStateBinding::new(ReferenceSource::PublicInput { index: 2 }, 2, None),
        ];

        let lowered = lower_mlir_module_for_program_with_reference_state(
            &program,
            &[],
            &input_types,
            &output_types,
            "main",
            Some(argument_shardings.as_slice()),
            Some(result_shardings.as_slice()),
            None,
            &states,
        )
        .unwrap();
        assert_eq!(lowered.signature.input_mapping(), &[None, Some(0), Some(1)]);
        assert_eq!(lowered.signature.output_mapping(), &[Some(0)]);
        let expected_signature = concat!(
            "  func.func @main(",
            "%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>, ",
            "tf.aliasing_output = 0 : i64}, ",
            "%arg1: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) -> ",
            "(tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) {",
        );
        assert_eq!(
            lowered.stable_hlo,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                @SIGNATURE@
                    %c = stablehlo.constant dense<false> : tensor<i1>
                    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<3xi1>
                    return %arg0 : tensor<4xf32>
                  }
                }
            "#}
            .replace("@SIGNATURE@", expected_signature),
        );

        let invalid_state = ReferenceStateBinding::new(ReferenceSource::PublicInput { index: 0 }, 0, None);
        assert!(matches!(
            lower_mlir_module_for_program_with_reference_state(
                &program,
                &[],
                &input_types,
                &output_types,
                "main",
                Some(argument_shardings.as_slice()),
                Some(result_shardings.as_slice()),
                None,
                std::slice::from_ref(&invalid_state),
            ),
            Err(LoweringError::InvalidReferenceStateAbi { message })
                if message == "logical state input 0 is erased from the executable boundary",
        ));

        let asymmetric_sharding_message =
            "reference-state aliases require both argument and result sharding metadata or neither";
        for (argument_shardings, result_shardings) in
            [(Some(argument_shardings.as_slice()), None), (None, Some(result_shardings.as_slice()))]
        {
            assert!(matches!(
                lower_mlir_module_for_program_with_reference_state(
                    &program,
                    &[],
                    &input_types,
                    &output_types,
                    "main",
                    argument_shardings,
                    result_shardings,
                    None,
                    &states,
                ),
                Err(LoweringError::InvalidReferenceStateAbi { message })
                    if message == asymmetric_sharding_message,
            ));
        }

        let mismatched_result_shardings = vec![Sharding::replicated(mesh, 1)];
        assert!(matches!(
            lower_mlir_module_for_program_with_reference_state(
                &program,
                &[],
                &input_types,
                &output_types,
                "main",
                Some(argument_shardings.as_slice()),
                Some(mismatched_result_shardings.as_slice()),
                None,
                &states,
            ),
            Err(LoweringError::InvalidReferenceStateAbi { message })
                if message == "state input 1 and output 0 must use the same sharding",
        ));
    }

    #[test]
    fn test_bounded_dynamic_reference_state_alias_is_rejected_before_lowering() {
        use ryft_core::ReferenceSource;

        let extent = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let state_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 1);
        let mut builder = crate::experimental::ops::XlaProgramBuilder::new();
        let state = builder.add_input(ArrayIrType::Array(state_type.clone()));
        let program: FlatXlaProgram = builder.build(vec![state], vec![Placeholder], vec![Placeholder]).unwrap();
        let reference_state = ReferenceStateBinding::new(ReferenceSource::PublicInput { index: 0 }, 0, Some(0));

        assert!(matches!(
            lower_mlir_module_for_program_with_reference_state(
                &program,
                &[],
                &vec![state_type.clone()],
                &vec![state_type],
                "main",
                Some(std::slice::from_ref(&sharding)),
                Some(std::slice::from_ref(&sharding)),
                None,
                std::slice::from_ref(&reference_state),
            ),
            Err(LoweringError::InvalidReferenceStateAbi { message })
                if message == "state input 0 and output 0 must be static because bounded-dynamic mutation alias \
                               compatibility is unsupported",
        ));
    }

    #[test]
    fn test_entry_alias_attribute_executes_in_place_on_cpu() {
        use std::collections::HashMap;
        use std::sync::Arc;

        use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
        use ryft_pjrt::{
            BufferType, ClientOptions, CpuClientOptions, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram,
            load_cpu_plugin,
        };

        use crate::tests::{values_from_bytes, values_to_bytes};

        let context = MlirContext::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tensor_type = context.tensor_type(context.float32_type(), &[MlirSize::Static(4)], None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let function_block = context.block(&[(tensor_type, location)]);
                {
                    let input = function_block.argument(0).unwrap();
                    let mut function_block_ref = function_block.as_ref();
                    let doubled =
                        function_block_ref.append_operation(stable_hlo::add(input, input, location).unwrap()).unwrap();
                    function_block_ref
                        .append_operation(func::r#return(&[doubled.result(0).unwrap()], location).unwrap())
                        .unwrap();
                }
                let mut function_region = context.region();
                function_region.append_block(function_block).unwrap();
                func::func(
                    "main",
                    func::FuncAttributes {
                        arguments: vec![TypeAndAttributes {
                            r#type: tensor_type.as_ref(),
                            attributes: Some(HashMap::from([
                                ("mhlo.sharding".into(), context.string_attribute("{replicated}").as_ref()),
                                (
                                    "tf.aliasing_output".into(),
                                    context.integer_attribute(context.signless_integer_type(64), 0).as_ref(),
                                ),
                            ])),
                        }],
                        results: vec![tensor_type.into()],
                        ..Default::default()
                    },
                    function_region,
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        let module = module.to_string();
        assert!(
            module.contains("%arg0: tensor<4xf32> {mhlo.sharding = \"{replicated}\", tf.aliasing_output = 0 : i64}",),
            "{module}",
        );

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
                    environment_option_overrides: HashMap::new(),
                    target_config: None,
                    allow_in_place_mlir_modification: false,
                    matrix_unit_operand_precision: Precision::Default as i32,
                },
            )
            .unwrap();
        let device = executable.addressable_devices().unwrap().remove(0);
        let input = client
            .buffer(values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(), BufferType::F32, &[4], None, device, None)
            .unwrap();
        input.ready().unwrap().r#await().unwrap();
        // The input is ready and remains alive in `inputs` until execution completes, so comparing its opaque address
        // with the synchronized output address does not dereference either device pointer.
        let input_pointer = unsafe { input.unsafe_pointer().unwrap() };
        let inputs = [ExecutionInput { buffer: Arc::new(input), donatable: true }];
        let execution = executable
            .execute(
                vec![ExecutionDeviceInputs { inputs: &inputs, ..Default::default() }],
                Vec::new(),
                0,
                None,
                Some(file!()),
                None,
                None,
            )
            .unwrap();
        let mut device_outputs = execution.block_until_ready().unwrap().remove(0);
        let output = device_outputs.outputs.remove(0);
        let output_pointer = unsafe { output.unsafe_pointer().unwrap() };
        // `tf.aliasing_output` is a hint rather than a guarantee (XLA's copy insertion may materialize a fresh
        // output), so this pointer equality is a deliberate canary pinned to the repository's pinned CPU plugin: if a
        // plugin upgrade breaks it while the numeric assertion below still passes, re-evaluate whether the alias hint
        // is still honored rather than assuming a correctness bug.
        assert_eq!(output_pointer, input_pointer);
        let output = output.copy_to_host(None).unwrap().r#await().unwrap();
        assert_eq!(values_from_bytes::<f32>(output.as_slice()), vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_broadcast_explicit_sharding_transition_executes_on_cpu() {
        use std::collections::HashMap;

        use ryft_core::{Device, DeviceMesh};
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
            .add_instruction(BroadcastOperation::new(output_type.clone(), vec![0]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .into_unprojected::<XlaConstant, XlaOperation>()
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
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(array_type.clone());
        let capture = builder.add_constant(XlaConstant::Captured(CaptureReference::new(0, array_type.clone().into())));
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, capture], None).unwrap()[0];
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
    fn test_to_mlir_module_for_program_lowers_immediate_dimension_constants_in_place() {
        use ryft_core::DimensionToScalarOperation;

        // An immediate dimension constant carries its own host-sized extent, and so — unlike a capture reference —
        // it materializes as a scalar `i64` `stablehlo.constant` in the function body instead of consuming a hidden
        // capture argument. That is exactly what makes first-class extents usable where no capture table is
        // reachable, such as inside a `shard_map` manual region.
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap()));
        let mut builder = CompositeXlaProgramBuilder::new();
        let extent = builder.add_constant(XlaConstant::Dimension(DimensionValue::new(extent_type, 4).unwrap()));
        let output = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![extent], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], Vec::new(), vec![Placeholder])
            .unwrap();
        let input_types: [ArrayType; 0] = [];
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &input_types,
            &vec![ArrayType::scalar(DataType::I64)],
            "main",
            None,
            None,
        )
        .unwrap();

        assert!(stablehlo.contains("func.func @main() -> tensor<i64>"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.constant dense<4> : tensor<i64>"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_capture_output_as_a_hidden_argument() {
        let array_type = test_vector_type(4);
        let mut builder = CompositeXlaProgramBuilder::new();
        let output = builder.add_constant(XlaConstant::Captured(CaptureReference::new(0, array_type.clone().into())));
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
            let mut builder = CompositeXlaProgramBuilder::new();
            let output =
                builder.add_constant(XlaConstant::Captured(CaptureReference::new(0, array_type.clone().into())));
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], Vec::new(), vec![Placeholder])
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let true_region = builder.import_program(branch());
        let false_region = builder.import_program(branch());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate],
                None,
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
    fn test_jit_call_capture_prefix_ignores_nested_call_capture_namespaces() {
        // The inner callee establishes its own two-slot capture namespace: its first two inputs are its lifted
        // captures, and its body resolves the constant `Captured(1)` against that prefix. Deriving the OUTER call's
        // capture prefix from capture-constant indices anywhere in the callee arena would conflate the namespaces
        // and reject this program with a false missing-capture error; the operation payload is authoritative.
        let array_type = test_vector_type(4);
        let inner = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let first_capture = builder.add_input(array_type.clone());
            let _second_capture = builder.add_input(array_type.clone());
            let second =
                builder.add_constant(XlaConstant::Captured(CaptureReference::new(1, array_type.clone().into())));
            let output =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![first_capture, second], None).unwrap()[0];
            Arc::new(
                builder
                    .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                    .unwrap(),
            )
        };
        let outer = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let input = builder.add_input(array_type.clone());
            let callee_region = builder.intern_callee(&inner, None).unwrap();
            let output = builder
                .add_instruction(
                    XlaOperation::JitCall(crate::experimental::ops::JitCallOperation::new(2)),
                    vec![callee_region],
                    vec![input, input],
                    None,
                )
                .unwrap()[0];
            Arc::new(
                builder
                    .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                    .unwrap(),
            )
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(array_type.clone());
        let callee_region = builder.intern_callee(&outer, None).unwrap();
        let output = builder
            .add_instruction(
                XlaOperation::JitCall(crate::experimental::ops::JitCallOperation::new(0)),
                vec![callee_region],
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let input_types = vec![array_type.clone()];
        let output_types = vec![array_type];
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();
        assert_eq!(stablehlo.matches("stablehlo.add").count(), 1, "{stablehlo}");
    }

    #[test]
    fn test_plain_elementwise_lowering_normalizes_all_implicit_operands() {
        let program = xla_elementwise_normalization_program();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert_elementwise_operands_are_normalized(&stablehlo);
    }

    #[test]
    fn test_traced_elementwise_lowering_normalizes_all_implicit_operands() {
        let plain_program = xla_elementwise_normalization_program();
        let input_types = plain_program.input_types();
        let output_types = plain_program.output_types();
        let program = plain_program.into_unprojected::<XlaConstant, XlaOperation>().unwrap();
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();
        assert_elementwise_operands_are_normalized(&stablehlo);
    }

    #[test]
    fn test_elementwise_lowering_normalizes_zero_sized_operands() {
        let mut builder = XlaProgramBuilder::new();
        let scalar = builder.add_input(ArrayType::scalar(DataType::F32));
        let empty = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0)])));
        let output = builder.add_instruction(DivOperation::new(), Vec::new(), vec![scalar, empty], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
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
    fn xla_add_self_callee(input_type: ArrayType) -> Arc<FlatXlaProgram> {
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(input_type);
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
        Arc::new(builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap())
    }

    /// Stages one `jit_call` to `callee` (interned as a shared callee root region) over `inputs` in `builder`.
    fn add_xla_jit_call(
        builder: &mut CompositeXlaProgramBuilder,
        callee: &Arc<FlatXlaProgram>,
        inputs: Vec<AtomId>,
    ) -> AtomId {
        let callee_region = builder.intern_callee(callee, None).unwrap();
        builder
            .add_instruction(
                XlaOperation::JitCall(crate::experimental::ops::JitCallOperation::new(0)),
                vec![callee_region],
                inputs,
                None,
            )
            .unwrap()[0]
    }

    /// Lowers an outer program that calls `callees` (one `jit_call` each) and sums the results, returning the
    /// module text. Each callee is `f(x) = x + x`; the outer function is `g(x) = sum_i callee_i(x)`.
    fn lower_two_jit_call_module(callees: Vec<Arc<FlatXlaProgram>>) -> String {
        let array_type = test_vector_type(4);
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(array_type.clone());
        let mut accumulator: Option<AtomId> = None;
        for callee in callees {
            let call_output = add_xla_jit_call(&mut builder, &callee, vec![input]);
            accumulator = Some(match accumulator {
                None => call_output,
                Some(previous) => builder
                    .add_instruction(AddOperation::new(), Vec::new(), vec![previous, call_output], None)
                    .unwrap()[0],
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

    /// Wraps one flat vector program in a replicated manual `shard_map` and lowers the enclosing program.
    fn lower_replicated_shard_map_body(body_program: FlatXlaProgram) -> Result<String, LoweringError> {
        use crate::experimental::operations::ShardMapOperation;
        use crate::experimental::shard_map::FlatTracedShardMap;

        let vector_type = test_vector_type(4);
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
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(vector_type.clone());
        let (operation, body) = ShardMapOperation::from_body(body);
        let body = builder.import_program(body);
        let output =
            builder.add_instruction(XlaOperation::ShardMap(Box::new(operation)), vec![body], vec![input], None)?[0];
        let program =
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])?;
        let input_types = vec![vector_type.clone()];
        let output_types = vec![vector_type];
        to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None)
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
    fn test_jit_call_dedup_does_not_traverse_shard_map_bodies() {
        // Phase 0 boundary pin for the first-class-program-regions plan: `count_jit_calls` intentionally skips
        // shard-map bodies, so a callee that occurs twice inside a `shard_map` body never gets a shared
        // `func.func private @jit_call_*` and both occurrences inline into the `sdy.manual_computation` region.
        let vector_type = test_vector_type(4);
        let callee = xla_add_self_callee(vector_type.clone());
        let body_program = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let input = builder.add_input(vector_type.clone());
            let first = add_xla_jit_call(&mut builder, &callee, vec![input]);
            let second = add_xla_jit_call(&mut builder, &callee, vec![input]);
            let output =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![first, second], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let module = lower_replicated_shard_map_body(body_program).unwrap();

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
    fn test_effectful_shard_map_body_is_rejected() {
        use ryft_core::PrintOperation;

        // Shardy manual-computation boundaries cannot carry StableHLO effect tokens. Reject an effectful body instead
        // of silently creating a private chain that is unordered with respect to effects outside the shard map.
        let vector_type = test_vector_type(4);
        let effectful_body_program = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let input = builder.add_input(vector_type.clone());
            let output =
                builder.add_instruction(PrintOperation::new("body"), Vec::new(), vec![input], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        assert_eq!(
            lower_replicated_shard_map_body(effectful_body_program).unwrap_err().to_string(),
            "effectful shard_map bodies are unsupported because sdy.manual_computation cannot preserve effect \
             ordering across its boundary",
        );
    }

    #[test]
    fn test_custom_jvp_lowering_inlines_only_the_primal_program() {
        use ryft_core::{CustomJvpOperation, PrintOperation};

        // A retained `custom_jvp` call lowers only its primal program and threads its effects onto the enclosing
        // ordered-I/O chain. Nothing from the user-supplied JVP program (marked here by the multiply on the tangent
        // side) reaches the emitted module.
        let vector_type = test_vector_type(4);
        let primal = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let input = builder.add_input(vector_type.clone());
            let printed =
                builder.add_instruction(PrintOperation::new("primal"), Vec::new(), vec![input], None).unwrap()[0];
            let output =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![printed, printed], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let jvp = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let input = builder.add_input(vector_type.clone());
            let tangent = builder.add_input(vector_type.clone());
            let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            let output_tangent =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![tangent, tangent], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![output, output_tangent],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder, Placeholder],
                )
                .unwrap()
        };
        let operation = CustomJvpOperation::new();
        let mut builder = CompositeXlaProgramBuilder::new();
        let primal_region = builder.import_region(primal.entry_region_ref());
        let jvp_region = builder.import_region(jvp.entry_region_ref());
        let input = builder.add_input(vector_type.clone());
        let output = builder
            .add_instruction(XlaOperation::CustomJvp(operation), vec![primal_region, jvp_region], vec![input], None)
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
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1 = stablehlo.custom_call @ryft.print(%arg0, %0) {api_version = 4 : i32, backend_config = {label = "primal"}, has_side_effect = true} : (tensor<4xf32>, !stablehlo.token) -> !stablehlo.token
                    %2 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %2 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_rematerialize_lowering_inlines_primal_effects_on_the_enclosing_token_chain() {
        use ryft_core::{PrintOperation, RematerializeOperation};

        // Rematerialization is a transform boundary, not an execution boundary. Lowering inlines its primal region,
        // so prints immediately before, inside, and after the call must form one ordered-I/O token chain.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let primal = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let input = builder.add_input(scalar_type.clone());
            let output =
                builder.add_instruction(PrintOperation::new("primal"), Vec::new(), vec![input], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let identity = unproject_plain_program(xla_identity_branch(scalar_type.clone()));
        let mut builder = CompositeXlaProgramBuilder::new();
        let primal_region = builder.import_region(primal.entry_region_ref());
        let forward_region = builder.import_region(identity.entry_region_ref());
        let backward_region = builder.import_region(identity.entry_region_ref());
        let tangent_region = builder.import_region(identity.entry_region_ref());
        let input = builder.add_input(scalar_type.clone());
        let before = builder.add_instruction(PrintOperation::new("before"), Vec::new(), vec![input], None).unwrap()[0];
        let rematerialized = builder
            .add_instruction(
                XlaOperation::Rematerialize(RematerializeOperation::new()),
                vec![primal_region, forward_region, backward_region, tangent_region],
                vec![before],
                None,
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(PrintOperation::new("after"), Vec::new(), vec![rematerialized], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &scalar_type, &scalar_type, "main", None, None).unwrap();

        assert_eq!(stablehlo.matches("stablehlo.after_all").count(), 1, "{stablehlo}");
        assert_eq!(stablehlo.matches("@ryft.print").count(), 3, "{stablehlo}");
        let print_lines = stablehlo.lines().filter(|line| line.contains("@ryft.print")).collect::<Vec<_>>();
        assert!(print_lines[0].contains("label = \"before\""), "{stablehlo}");
        assert!(print_lines[1].contains("label = \"primal\""), "{stablehlo}");
        assert!(print_lines[2].contains("label = \"after\""), "{stablehlo}");
        assert!(print_lines[1].contains("%1)"), "{stablehlo}");
        assert!(print_lines[2].contains("%2)"), "{stablehlo}");
    }

    #[test]
    fn test_rematerialize_lowering_rejects_capture_constants_in_its_regions() {
        use ryft_core::RematerializeOperation;

        // Rematerialized regions are traced through fresh-root contexts and can therefore never legally reference
        // the enclosing function's captures. A capture constant smuggled into such a region must fail loudly
        // instead of silently resolving against the enclosing function's capture prefix, which would alias
        // whichever captured value happens to occupy the referenced slot.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let primal = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let input = builder.add_input(scalar_type.clone());
            let captured = builder
                .add_constant(XlaConstant::Captured(CaptureReference::new(0, ArrayIrType::Array(scalar_type.clone()))));
            let output =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, captured], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let identity = unproject_plain_program(xla_identity_branch(scalar_type.clone()));
        let mut builder = CompositeXlaProgramBuilder::new();
        let primal_region = builder.import_region(primal.entry_region_ref());
        let forward_region = builder.import_region(identity.entry_region_ref());
        let backward_region = builder.import_region(identity.entry_region_ref());
        let tangent_region = builder.import_region(identity.entry_region_ref());
        let input = builder.add_input(scalar_type.clone());
        let output = builder
            .add_instruction(
                XlaOperation::Rematerialize(RematerializeOperation::new()),
                vec![primal_region, forward_region, backward_region, tangent_region],
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // The enclosing function's capture prefix has a matching slot #0, so before rematerialized regions were
        // scoped to an empty capture namespace this lowering silently forwarded that unrelated captured value.
        let result = to_mlir_module_for_program(
            &program,
            &[scalar_type.clone()],
            &scalar_type,
            &scalar_type,
            "main",
            None,
            None,
        );
        assert_eq!(result, Err(LoweringError::MissingCapturedConstant { index: 0 }));
    }

    #[test]
    fn test_executable_linear_call_lowering_inlines_only_the_forward_program() {
        use ryft_core::LinearCallOperation;

        let vector_type = test_vector_type(4);
        let forward = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let tangent = builder.add_input(vector_type.clone());
            let residual = builder.add_input(vector_type.clone());
            let output =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![tangent, residual], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![output],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let transpose = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let residual = builder.add_input(vector_type.clone());
            let cotangent = builder.add_input(vector_type.clone());
            let output =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![residual, cotangent], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![output],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let forward = builder.import_region(forward.entry_region_ref());
        let transpose = builder.import_region(transpose.entry_region_ref());
        let tangent = builder.add_input(vector_type.clone());
        let residual = builder.add_input(vector_type.clone());
        let output = builder
            .add_instruction(
                XlaOperation::LinearCall(LinearCallOperation::new(1)),
                vec![forward, transpose],
                vec![tangent, residual],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &[vector_type.clone(), vector_type.clone()],
            &[vector_type],
            "main",
            None,
            None,
        )
        .unwrap();

        assert!(module.contains("stablehlo.multiply"), "{module}");
        assert!(!module.contains("stablehlo.add"), "{module}");
    }

    #[test]
    fn test_transpose_only_linear_call_lowering_is_rejected() {
        use ryft_core::LinearCallOperation;

        // Phase 0 boundary pin for the first-class-program-regions plan: the un-transposed transpose-only linear
        // call carrier is reverse-mode-only and must be transposed away before lowering, so lowering it is rejected.
        let vector_type = test_vector_type(4);
        let backward = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let residual = builder.add_input(vector_type.clone());
            let cotangent = builder.add_input(vector_type.clone());
            let output =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![residual, cotangent], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![output],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let operation = LinearCallOperation::transpose_only(
            1,
            vec![ArrayIrType::Array(vector_type.clone())],
            vec![ArrayIrType::Array(vector_type.clone())],
        );
        let mut builder = CompositeXlaProgramBuilder::new();
        let backward_region = builder.import_region(backward.entry_region_ref());
        let tangent = builder.add_input(vector_type.clone());
        let residual = builder.add_input(vector_type.clone());
        let output = builder
            .add_instruction(XlaOperation::LinearCall(operation), vec![backward_region], vec![tangent, residual], None)
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
                if message == "operation `transpose_only_linear_call` cannot be lowered to StableHLO",
        ));
    }

    /// Builds a flat callee whose body contains a `condition` instruction (`f(p, x) = if p { -x } else { x }`),
    /// making it ineligible for structural `jit_call` deduplication because its nested branch bodies do not render
    /// into the callee's canonical program text.
    fn xla_condition_callee() -> Arc<FlatXlaProgram> {
        let vector_type = test_vector_type(4);
        let mut builder = CompositeXlaProgramBuilder::new();
        let true_region = builder.import_program(unproject_plain_program(xla_neg_branch(vector_type.clone())));
        let false_region = builder.import_program(unproject_plain_program(xla_identity_branch(vector_type.clone())));
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let input = builder.add_input(vector_type);
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
                None,
            )
            .unwrap()[0];
        Arc::new(builder.build(vec![output], vec![Placeholder, Placeholder], vec![Placeholder]).unwrap())
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
        let mut builder = CompositeXlaProgramBuilder::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let input = builder.add_input(vector_type.clone());
        let mut accumulator: Option<AtomId> = None;
        for callee in [first.clone(), first, second.clone(), second] {
            let call_output = add_xla_jit_call(&mut builder, &callee, vec![predicate, input]);
            accumulator = Some(match accumulator {
                None => call_output,
                Some(previous) => builder
                    .add_instruction(AddOperation::new(), Vec::new(), vec![previous, call_output], None)
                    .unwrap()[0],
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
    fn test_to_mlir_module_renders_provenance_locations() {
        let global_input_type = test_vector_type(8);
        let mesh = test_manual_mesh("x", 4);
        let traced: TracedShardMap<ArrayType, ArrayType> = traced_shard_map(
            |x| {
                let context = x.value().context().clone();
                // One instruction staged under nested scopes, one under a fused origin, and one with unknown
                // provenance, covering every provenance shape in one lowered module.
                let scoped = context.with_provenance_scope(ProvenanceScope::new("outer"), || {
                    context.with_provenance_scope(ProvenanceScope::new("inner"), || x.clone() + x.clone())
                });
                let fused = Provenance::fused([
                    Provenance::scope(ProvenanceScope::new("a"), Provenance::unknown()),
                    Provenance::scope(ProvenanceScope::new("b"), Provenance::unknown()),
                ]);
                let product = context.with_provenance_origin(fused, || scoped.clone() * scoped);
                product + x
            },
            global_input_type,
            mesh.clone(),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap(),
        )
        .unwrap();

        // The scoped addition carries the nested name-location chain, the fused multiplication carries a
        // metadata-free fused location, the unknown-provenance addition keeps the base (unknown) location, and
        // module/function scaffolding keeps the base location. Locations render because provenance-carrying modules
        // serialize with debug information enabled (and the non-pretty, parsable form).
        assert_eq!(
            lower_traced_module(&traced, "main").unwrap(),
            indoc! {r#"
                #loc = loc(unknown)
                module {
                  sdy.mesh @mesh = <["x"=4]> loc(#loc)
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>} loc(unknown)) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32> loc(unknown)) {
                      %1 = stablehlo.add %arg1, %arg1 : tensor<2xf32> loc(#loc4)
                      %2 = stablehlo.multiply %1, %1 : tensor<2xf32> loc(#loc5)
                      %3 = stablehlo.add %2, %arg1 : tensor<2xf32> loc(#loc)
                      sdy.return %3 : tensor<2xf32> loc(#loc)
                    } : (tensor<8xf32>) -> tensor<8xf32> loc(#loc)
                    return %0 : tensor<8xf32> loc(#loc)
                  } loc(#loc)
                } loc(#loc)
                #loc1 = loc("inner")
                #loc2 = loc("a")
                #loc3 = loc("b")
                #loc4 = loc("outer"(#loc1))
                #loc5 = loc(fused[#loc2, #loc3])
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
    fn test_to_mlir_module_for_plain_program_lowers_complex_sine_and_cosine() {
        let complex_type = ArrayType::scalar(DataType::C64);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(complex_type);
        let sine = builder
            .add_instruction(ArrayOperation::Sin(SinOperation::new()), Vec::new(), vec![input], None)
            .unwrap()[0];
        let cosine = builder
            .add_instruction(ArrayOperation::Cos(CosOperation::new()), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
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
            .add_instruction(ArrayOperation::Reduce(ReduceOperation::new(axes, kind)), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
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
                    %0 = stablehlo.reduce(%arg0 init: %cst) across dimensions = [1] : (tensor<2x3xbf16>, tensor<bf16>) -> tensor<2xbf16>
                     reducer(%arg1: tensor<bf16>, %arg2: tensor<bf16>)  {
                      %1 = stablehlo.compare GT, %arg1, %arg2, TOTALORDER : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
                      %2 = stablehlo.select %1, %arg1, %arg2 : tensor<i1>, tensor<bf16>
                      %3 = stablehlo.compare NE, %arg1, %arg1, FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
                      %4 = stablehlo.compare NE, %arg2, %arg2, FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
                      %5 = stablehlo.select %4, %arg2, %2 : tensor<i1>, tensor<bf16>
                      %6 = stablehlo.select %3, %arg1, %5 : tensor<i1>, tensor<bf16>
                      stablehlo.return %6 : tensor<bf16>
                    }
                    return %0 : tensor<2xbf16>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_boolean_and_complex_extrema() {
        assert_eq!(
            lowered_reduce_module(DataType::Boolean, ReductionKind::Max, vec![0], vec![0]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<0xi1>) -> tensor<i1> {
                    %c = stablehlo.constant dense<false> : tensor<i1>
                    %0 = stablehlo.reduce(%arg0 init: %c) applies stablehlo.maximum across dimensions = [0] : (tensor<0xi1>, tensor<i1>) -> tensor<i1>
                    return %0 : tensor<i1>
                  }
                }
            "#},
        );

        // StableHLO has no primitive with JAX's complex ordering contract, so the reduction body explicitly compares
        // real components first and imaginary components only when the real components are equal.
        let stablehlo = lowered_reduce_module(DataType::C64, ReductionKind::Max, vec![0], vec![0]).unwrap();
        assert!(stablehlo.contains("tensor<complex<f32>>"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.reduce"), "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.real").count(), 2, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.imag").count(), 2, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.compare").count(), 3, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.select").count(), 2, "{stablehlo}");
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
                ArrayOperation::Reduce(ReduceOperation::new(vec![1], ReductionKind::Sum)),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
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
                ArrayOperation::Reduce(ReduceOperation::new(vec![0], ReductionKind::Mean)),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main"),
            Err(LoweringError::UnsupportedOp { op: "`reduce` mean over dynamically sized axis 0".to_string() }),
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
                    %0 = stablehlo.reduce(%arg0 init: %cst) across dimensions = [0] : (tensor<4xf16>, tensor<f16>) -> tensor<f16>
                     reducer(%arg1: tensor<f16>, %arg2: tensor<f16>)  {
                      %1 = stablehlo.compare LT, %arg1, %arg2, TOTALORDER : (tensor<f16>, tensor<f16>) -> tensor<i1>
                      %2 = stablehlo.select %1, %arg1, %arg2 : tensor<i1>, tensor<f16>
                      %3 = stablehlo.compare NE, %arg1, %arg1, FLOAT : (tensor<f16>, tensor<f16>) -> tensor<i1>
                      %4 = stablehlo.compare NE, %arg2, %arg2, FLOAT : (tensor<f16>, tensor<f16>) -> tensor<i1>
                      %5 = stablehlo.select %4, %arg2, %2 : tensor<i1>, tensor<f16>
                      %6 = stablehlo.select %3, %arg1, %5 : tensor<i1>, tensor<f16>
                      stablehlo.return %6 : tensor<f16>
                    }
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
                    %0 = stablehlo.reduce(%arg0 init: %cst) across dimensions = [0] : (tensor<4xf8E5M2>, tensor<f8E5M2>) -> tensor<f8E5M2>
                     reducer(%arg1: tensor<f8E5M2>, %arg2: tensor<f8E5M2>)  {
                      %1 = stablehlo.compare GT, %arg1, %arg2, TOTALORDER : (tensor<f8E5M2>, tensor<f8E5M2>) -> tensor<i1>
                      %2 = stablehlo.select %1, %arg1, %arg2 : tensor<i1>, tensor<f8E5M2>
                      %3 = stablehlo.compare NE, %arg1, %arg1, FLOAT : (tensor<f8E5M2>, tensor<f8E5M2>) -> tensor<i1>
                      %4 = stablehlo.compare NE, %arg2, %arg2, FLOAT : (tensor<f8E5M2>, tensor<f8E5M2>) -> tensor<i1>
                      %5 = stablehlo.select %4, %arg2, %2 : tensor<i1>, tensor<f8E5M2>
                      %6 = stablehlo.select %3, %arg1, %5 : tensor<i1>, tensor<f8E5M2>
                      stablehlo.return %6 : tensor<f8E5M2>
                    }
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
                    %0 = stablehlo.reduce(%arg0 init: %cst) across dimensions = [0] : (tensor<4xf8E4M3FN>, tensor<f8E4M3FN>) -> tensor<f8E4M3FN>
                     reducer(%arg1: tensor<f8E4M3FN>, %arg2: tensor<f8E4M3FN>)  {
                      %1 = stablehlo.compare GT, %arg1, %arg2, TOTALORDER : (tensor<f8E4M3FN>, tensor<f8E4M3FN>) -> tensor<i1>
                      %2 = stablehlo.select %1, %arg1, %arg2 : tensor<i1>, tensor<f8E4M3FN>
                      %3 = stablehlo.compare NE, %arg1, %arg1, FLOAT : (tensor<f8E4M3FN>, tensor<f8E4M3FN>) -> tensor<i1>
                      %4 = stablehlo.compare NE, %arg2, %arg2, FLOAT : (tensor<f8E4M3FN>, tensor<f8E4M3FN>) -> tensor<i1>
                      %5 = stablehlo.select %4, %arg2, %2 : tensor<i1>, tensor<f8E4M3FN>
                      %6 = stablehlo.select %3, %arg1, %5 : tensor<i1>, tensor<f8E4M3FN>
                      stablehlo.return %6 : tensor<f8E4M3FN>
                    }
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
                    %0 = stablehlo.reduce(%arg0 init: %cst) across dimensions = [0] : (tensor<4xf4E2M1FN>, tensor<f4E2M1FN>) -> tensor<f4E2M1FN>
                     reducer(%arg1: tensor<f4E2M1FN>, %arg2: tensor<f4E2M1FN>)  {
                      %1 = stablehlo.compare LT, %arg1, %arg2, TOTALORDER : (tensor<f4E2M1FN>, tensor<f4E2M1FN>) -> tensor<i1>
                      %2 = stablehlo.select %1, %arg1, %arg2 : tensor<i1>, tensor<f4E2M1FN>
                      %3 = stablehlo.compare NE, %arg1, %arg1, FLOAT : (tensor<f4E2M1FN>, tensor<f4E2M1FN>) -> tensor<i1>
                      %4 = stablehlo.compare NE, %arg2, %arg2, FLOAT : (tensor<f4E2M1FN>, tensor<f4E2M1FN>) -> tensor<i1>
                      %5 = stablehlo.select %4, %arg2, %2 : tensor<i1>, tensor<f4E2M1FN>
                      %6 = stablehlo.select %3, %arg1, %5 : tensor<i1>, tensor<f4E2M1FN>
                      stablehlo.return %6 : tensor<f4E2M1FN>
                    }
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
    fn test_to_mlir_module_for_plain_program_supports_f8e8m0fnu_extrema_but_rejects_sum() {
        assert_eq!(
            lowered_reduce_module(DataType::F8E8M0FNU, ReductionKind::Sum, vec![0], vec![4]).unwrap_err(),
            LoweringError::UnsupportedDataType { data_type: DataType::F8E8M0FNU },
        );
        assert!(lowered_reduce_module(DataType::F8E8M0FNU, ReductionKind::Max, vec![0], vec![0]).is_ok());
        assert!(lowered_reduce_module(DataType::F8E8M0FNU, ReductionKind::Min, vec![0], vec![0]).is_ok());
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_condition_to_stablehlo_if() {
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let input_type = ArrayType::scalar(DataType::F32);
        let mut builder = CompositeXlaProgramBuilder::new();
        let true_region = builder.import_program(unproject_plain_program(xla_neg_branch(input_type.clone())));
        let false_region = builder.import_program(unproject_plain_program(xla_identity_branch(input_type.clone())));
        let predicate = builder.add_input(predicate_type.clone());
        let input = builder.add_input(input_type.clone());
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &(predicate_type, input_type.clone()),
            &input_type,
            "main",
            None,
            None,
        )
        .unwrap();

        assert!(stablehlo.contains("\"stablehlo.if\""), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.negate"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.return"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_differentiated_composite_control_flow() {
        let scalar_boolean = ArrayType::scalar(DataType::Boolean);
        let scalar_f32 = ArrayType::scalar(DataType::F32);
        let vector_f32 = test_vector_type(2);

        let branch = |double: bool| {
            let mut builder = CompositeXlaProgramBuilder::new();
            let operand = builder.add_input(scalar_f32.clone());
            let output = if double {
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![operand, operand], None).unwrap()[0]
            } else {
                operand
            };
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        let mut scan_body_builder = CompositeXlaProgramBuilder::new();
        let carry = scan_body_builder.add_input(scalar_f32.clone());
        let item = scan_body_builder.add_input(scalar_f32.clone());
        let next_carry =
            scan_body_builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, item], None).unwrap()[0];
        let scan_body = scan_body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![next_carry, next_carry],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut while_condition_builder = CompositeXlaProgramBuilder::new();
        let state = while_condition_builder.add_input(scalar_f32.clone());
        let while_predicate = while_condition_builder
            .add_instruction(
                XlaOperation::Array(ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::LessThan))),
                Vec::new(),
                vec![state, state],
                None,
            )
            .unwrap()[0];
        let while_condition = while_condition_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![while_predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut while_body_builder = CompositeXlaProgramBuilder::new();
        let state = while_body_builder.add_input(scalar_f32.clone());
        let next_state = while_body_builder
            .add_instruction(AddOperation::new(), Vec::new(), vec![state, state], None)
            .unwrap()[0];
        let while_body = while_body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![next_state], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = CompositeXlaProgramBuilder::new();
        let true_region = builder.import_region(branch(true).entry_region_ref());
        let false_region = builder.import_region(branch(false).entry_region_ref());
        let scan_region = builder.import_region(scan_body.entry_region_ref());
        let while_condition_region = builder.import_region(while_condition.entry_region_ref());
        let while_body_region = builder.import_region(while_body.entry_region_ref());
        let predicate = builder.add_input(scalar_boolean.clone());
        let operand = builder.add_input(scalar_f32.clone());
        let items = builder.add_input(vector_f32.clone());
        let selected = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, operand],
                None,
            )
            .unwrap()[0];
        let scan_outputs = builder
            .add_instruction(
                XlaOperation::Scan(ScanOperation::new(1, 2)),
                vec![scan_region],
                vec![selected, items],
                None,
            )
            .unwrap()
            .to_vec();
        let final_state = builder
            .add_instruction(
                XlaOperation::While(WhileOperation::new().with_iteration_bound(2).unwrap()),
                vec![while_condition_region, while_body_region],
                vec![scan_outputs[0]],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![final_state, scan_outputs[1]],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();

        // Keep the complete mixed control-flow program behind one compiled-call boundary. Linearization must preserve
        // that boundary while deriving its primal and pullback callees, and lowering must still reach the nested
        // condition and loop operations.
        let mut builder = CompositeXlaProgramBuilder::new();
        let callee = builder.import_region(program.entry_region_ref());
        let predicate = builder.add_input(scalar_boolean.clone());
        let operand = builder.add_input(scalar_f32.clone());
        let items = builder.add_input(vector_f32.clone());
        let outputs = builder
            .add_instruction(
                XlaOperation::JitCall(crate::experimental::ops::JitCallOperation::new(0)),
                vec![callee],
                vec![predicate, operand, items],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();

        let linearization = program.linearize().unwrap();
        let primal = linearization.primal();
        assert_eq!(linearization.residual_count(), 2);
        assert_eq!(primal.instructions().len(), 2);
        let primal_inputs = vec![scalar_boolean, scalar_f32.clone(), vector_f32.clone()];
        let primal_outputs = primal
            .output_types()
            .iter()
            .map(|r#type| match r#type {
                ArrayIrType::Array(r#type) => r#type.clone(),
                ArrayIrType::Dimension(_) => panic!("this fixture has only array residuals"),
                ArrayIrType::Reference(_) => panic!("this fixture has no reference residuals"),
            })
            .collect::<Vec<_>>();
        let primal_stablehlo =
            to_mlir_module_for_program(primal, &[], &primal_inputs, &primal_outputs, "main", None, None).unwrap();
        assert!(primal_stablehlo.contains("\"stablehlo.if\""), "{primal_stablehlo}");
        assert!(primal_stablehlo.matches("stablehlo.while").count() >= 2, "{primal_stablehlo}");

        let pullback = linearization.pullback().unwrap();
        let pullback_inputs = pullback
            .input_types()
            .iter()
            .map(|r#type| match r#type {
                ArrayIrType::Array(r#type) => r#type.clone(),
                ArrayIrType::Dimension(_) => panic!("this fixture has only array residuals"),
                ArrayIrType::Reference(_) => panic!("this fixture has no reference residuals"),
            })
            .collect::<Vec<_>>();
        let pullback_outputs = vec![scalar_f32, vector_f32];
        let pullback_stablehlo =
            to_mlir_module_for_program(&pullback, &[], &pullback_inputs, &pullback_outputs, "main", None, None)
                .unwrap();
        assert_eq!(pullback.instructions().len(), 1);
        assert!(pullback_stablehlo.contains("\"stablehlo.if\""), "{pullback_stablehlo}");
        assert!(pullback_stablehlo.contains("stablehlo.while"), "{pullback_stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_gather() {
        use ryft_core::{
            Dimension, DimensionBounds, DimensionVariable, GatherDimensionNumbers, GatherOperation, Shape,
        };

        // Take whole rows of a [3, 2] matrix at the row indices in a [2, 1] index array: offset axis 1 carries the
        // row (slice sizes [1, 2]); axis 0 is collapsed (start-index driven). Output is [2, 2].
        let operand_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]));
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]));
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        let mut builder = XlaProgramBuilder::new();
        let operand = builder.add_input(operand_type);
        let indices = builder.add_input(indices_type);
        let output = builder
            .add_instruction(ArrayOperation::Gather(operation), Vec::new(), vec![operand, indices], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
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
            .add_instruction(ArrayOperation::Gather(operation), Vec::new(), vec![operand, indices], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
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
        use ryft_core::{Dimension, GatherDimensionNumbers, GatherOperation, GatherScatterMode, Shape};

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
            .add_instruction(ArrayOperation::Gather(operation), Vec::new(), vec![operand, indices], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
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
        use ryft_core::{Dimension, ScatterDimensionNumbers, ScatterOperation, ScatterReductionKind, Shape};

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
            .add_instruction(ArrayOperation::Scatter(operation), Vec::new(), vec![operand, indices, updates], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
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

        // Complex scatter extrema share the explicit lexicographic combiner used by complex reductions.
        let operand_type = ArrayType::new(DataType::C64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]));
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]));
        let updates_type = ArrayType::new(DataType::C64, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![1], vec![0], vec![0]), ScatterReductionKind::Max);
        let mut builder = XlaProgramBuilder::new();
        let operand = builder.add_input(operand_type);
        let indices = builder.add_input(indices_type);
        let updates = builder.add_input(updates_type);
        let output = builder
            .add_instruction(ArrayOperation::Scatter(operation), Vec::new(), vec![operand, indices, updates], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert_eq!(stablehlo.matches("stablehlo.real").count(), 2, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.imag").count(), 2, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.compare").count(), 3, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.select").count(), 2, "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_constant_predicate_condition_to_stablehlo_if() {
        // A condition whose predicate input is fed by a staged constant still lowers to `stablehlo.if`; folding the
        // constant predicate away is the backend's job (StableHLO canonicalization and XLA's conditional
        // simplification), not ryft's.
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let input_type = ArrayType::scalar(DataType::F32);
        let mut builder = CompositeXlaProgramBuilder::new();
        let true_region = builder.import_program(unproject_plain_program(xla_neg_branch(input_type.clone())));
        let false_region = builder.import_program(unproject_plain_program(xla_identity_branch(input_type.clone())));
        let input = builder.add_input(input_type.clone());
        let predicate =
            builder.add_instruction(OneOperation::new(predicate_type), Vec::new(), vec![], None).unwrap()[0];
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &input_type, &input_type, "main", None, None).unwrap();

        assert!(stablehlo.contains("\"stablehlo.if\""), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.constant"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.negate"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_forwards_dimension_extent_through_condition() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let extent_type = DimensionType::new(extent.clone());
        let dynamic_vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let scalar_type = ArrayType::scalar(DataType::F32);
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let output_type = dynamic_vector_type.clone();

        let branch = |negate: bool| {
            let mut builder = CompositeXlaProgramBuilder::new();
            let branch_extent = builder.0.add_input(extent_type.clone().into());
            let scalar = builder.add_input(scalar_type.clone());
            let scalar = if negate {
                builder.add_instruction(NegOperation::new(), Vec::new(), vec![scalar], None).unwrap()[0]
            } else {
                scalar
            };
            let output = builder
                .add_instruction(
                    DynamicBroadcastOperation::new(Vec::new()),
                    Vec::new(),
                    vec![scalar, branch_extent],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![output],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };

        let mut builder = CompositeXlaProgramBuilder::new();
        let true_region = builder.import_region(branch(false).entry_region_ref());
        let false_region = builder.import_region(branch(true).entry_region_ref());
        let vector = builder.add_input(dynamic_vector_type.clone());
        let predicate = builder.add_input(predicate_type.clone());
        let scalar = builder.add_input(scalar_type.clone());
        let extent = builder
            .add_instruction(
                DimensionSizeOperation::new(&dynamic_vector_type, 0).unwrap(),
                Vec::new(),
                vec![vector],
                None,
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, extent, scalar],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &vec![dynamic_vector_type, predicate_type, scalar_type],
            &vec![output_type],
            "main",
            None,
            None,
        )
        .unwrap();

        assert!(stablehlo.contains("stablehlo.get_dimension_size"), "{stablehlo}");
        assert!(stablehlo.contains("\"stablehlo.if\""), "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.set_dimension_size").count(), 3, "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.negate"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_while_to_stablehlo_while() {
        let state_type = ArrayType::scalar(DataType::Boolean);
        let mut builder = CompositeXlaProgramBuilder::new();
        let condition_region = builder.import_program(unproject_plain_program(xla_identity_branch(state_type.clone())));
        let body_region = builder.import_program(unproject_plain_program(xla_identity_branch(state_type.clone())));
        let state = builder.add_input(state_type.clone());
        let output = builder
            .add_instruction(
                XlaOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                vec![state],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &state_type, &state_type, "main", None, None).unwrap();

        assert!(stablehlo.contains("stablehlo.while"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.return"), "{stablehlo}");
        // An unbounded while emits no iteration-counter machinery.
        assert!(!stablehlo.contains("stablehlo.and"), "{stablehlo}");
        assert!(!stablehlo.contains("stablehlo.add"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_threads_effectful_while_condition_once_per_evaluation() {
        use ryft_core::PrintOperation;

        // StableHLO condition regions cannot return tokens. The lowering therefore evaluates an effectful scalar
        // condition once before the loop and once after each body execution, carrying its predicate through the loop
        // state so both prints remain on the enclosing ordered-I/O chain.
        let state_type = ArrayType::scalar(DataType::Boolean);
        let condition = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let state = builder.add_input(state_type.clone());
            let predicate =
                builder.add_instruction(PrintOperation::new("condition"), Vec::new(), vec![state], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![predicate], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_program(unproject_plain_program(xla_identity_branch(state_type.clone())));
        let state = builder.add_input(state_type.clone());
        let output = builder
            .add_instruction(
                XlaOperation::While(WhileOperation::new().with_iteration_bound(1).unwrap()),
                vec![condition_region, body_region],
                vec![state],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &state_type, &state_type, "main", None, None).unwrap();

        assert_eq!(stablehlo.matches("stablehlo.after_all").count(), 1, "{stablehlo}");
        assert_eq!(stablehlo.matches("@ryft.print").count(), 2, "{stablehlo}");
        let condition_region = stablehlo.split_once("cond {").unwrap().1.split_once("} do {").unwrap().0;
        assert!(!condition_region.contains("@ryft.print"), "{stablehlo}");
        let while_header = stablehlo.lines().find(|line| line.contains("stablehlo.while(")).unwrap();
        assert!(while_header.contains("tensor<i1>, !stablehlo.token"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_forwards_loop_carried_dimension_extent() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let extent_type = DimensionType::new(extent.clone());
        let dynamic_vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let scalar_type = ArrayType::scalar(DataType::F32);

        let condition = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let extent = builder.0.add_input(extent_type.clone().into());
            let predicate = builder
                .add_instruction(
                    XlaOperation::Compare(CompareOperation::new(ComparisonDirection::LessThan)),
                    Vec::new(),
                    vec![extent, extent],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![predicate], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let extent = builder.0.add_input(extent_type.into());
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![extent], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        let mut builder = CompositeXlaProgramBuilder::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(body.entry_region_ref());
        let vector = builder.add_input(dynamic_vector_type.clone());
        let scalar = builder.add_input(scalar_type.clone());
        let extent = builder
            .add_instruction(
                DimensionSizeOperation::new(&dynamic_vector_type, 0).unwrap(),
                Vec::new(),
                vec![vector],
                None,
            )
            .unwrap()[0];
        let carried_extent = builder
            .add_instruction(
                XlaOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                vec![extent],
                None,
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![scalar, carried_extent], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &vec![dynamic_vector_type.clone(), scalar_type],
            &vec![dynamic_vector_type],
            "main",
            None,
            None,
        )
        .unwrap();

        assert!(stablehlo.contains("stablehlo.get_dimension_size"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.while"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.compare"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.set_dimension_size"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_bounded_while_with_fused_counter_condition() {
        // A semantic iteration bound threads an internal i64 counter through the `stablehlo.while` state: the
        // condition region conjoins `counter < bound` into the original predicate via `stablehlo.compare` plus
        // `stablehlo.and`, and the body region increments the counter via `stablehlo.add`. The operation's outputs
        // remain the original state elements.
        let state_type = ArrayType::scalar(DataType::Boolean);
        let while_operation = WhileOperation::new().with_iteration_bound(3).unwrap();
        let mut builder = CompositeXlaProgramBuilder::new();
        let condition_region = builder.import_program(unproject_plain_program(xla_identity_branch(state_type.clone())));
        let body_region = builder.import_program(unproject_plain_program(xla_identity_branch(state_type.clone())));
        let state = builder.add_input(state_type.clone());
        let output = builder
            .add_instruction(
                XlaOperation::While(while_operation),
                vec![condition_region, body_region],
                vec![state],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &state_type, &state_type, "main", None, None).unwrap();

        assert!(stablehlo.contains("stablehlo.while"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.compare"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.and"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.add"), "{stablehlo}");
        assert!(stablehlo.contains("tensor<i64>"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_batched_predicate_while_with_masked_state_updates() {
        // A batched (per-item) predicate lowers with the masked semantics owned by the while lowering: the condition
        // region reduces the `tensor<3xi1>` predicate to the scalar loop-continuation decision with a Boolean `or`
        // reduction, and the body region recomputes the per-item predicate on the incoming state and selects per
        // state element between the body's candidate update and the carried state, freezing finished items. The
        // predicate shape equals the state shape here, so no broadcast is needed for the mask.
        use ryft_core::{CompareOperation, OneLikeOperation, ZeroLikeOperation};
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let condition = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let state = builder.add_input(state_type.clone());
            let zero = builder
                .add_instruction(ZeroLikeOperation::<ArrayType>::new(), Vec::new(), vec![state], None)
                .unwrap()[0];
            let predicate = builder
                .add_instruction(
                    XlaOperation::Array(ArrayOperation::Compare(CompareOperation::new(
                        ComparisonDirection::GreaterThan,
                    ))),
                    Vec::new(),
                    vec![state, zero],
                    None,
                )
                .unwrap()[0];
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![predicate], vec![Placeholder], vec![Placeholder])
        }
        .unwrap();
        let body = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let state = builder.add_input(state_type.clone());
            let one = builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
            let next = builder.add_instruction(SubOperation::new(), Vec::new(), vec![state, one], None).unwrap()[0];
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![next], vec![Placeholder], vec![Placeholder])
        }
        .unwrap();
        let mut builder = CompositeXlaProgramBuilder::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(body.entry_region_ref());
        let state = builder.add_input(state_type.clone());
        let output = builder
            .add_instruction(
                XlaOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                vec![state],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &state_type, &state_type, "main", None, None).unwrap();

        assert!(stablehlo.contains("stablehlo.while"), "{stablehlo}");
        // Condition region: `or`-reduce the per-item predicate into the scalar continuation decision.
        assert!(stablehlo.contains("stablehlo.reduce"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.or"), "{stablehlo}");
        // Body region: per-item masked state update under the recomputed predicate.
        assert!(stablehlo.contains("stablehlo.select"), "{stablehlo}");
        assert!(stablehlo.contains("tensor<3xi1>"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_passes_loop_invariant_dimension_through_masked_batched_predicate_while() {
        // A batched-predicate loop may also carry a first-class dimension, which the relaxed `WhileTypeSemantics`
        // contract requires to be loop-invariant. Masking a loop-invariant carry is the identity, so the lowering
        // threads the body's dimension result on directly: only the array carry gets a `stablehlo.select`, while the
        // condition region still `or`-reduces the per-item predicate into the scalar continuation decision.
        use ryft_core::{CompareOperation, OneLikeOperation, ZeroLikeOperation};
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let extent_type = DimensionType::new(extent.clone());
        let dynamic_vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let condition = {
            let mut builder = CompositeXlaProgramBuilder::new();
            builder.0.add_input(extent_type.clone().into());
            let state = builder.add_input(state_type.clone());
            let zero = builder
                .add_instruction(ZeroLikeOperation::<ArrayType>::new(), Vec::new(), vec![state], None)
                .unwrap()[0];
            let predicate = builder
                .add_instruction(
                    XlaOperation::Array(ArrayOperation::Compare(CompareOperation::new(
                        ComparisonDirection::GreaterThan,
                    ))),
                    Vec::new(),
                    vec![state, zero],
                    None,
                )
                .unwrap()[0];
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![predicate],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
        }
        .unwrap();
        let body = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let carried_extent = builder.0.add_input(extent_type.into());
            let state = builder.add_input(state_type.clone());
            let one = builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
            let next = builder.add_instruction(SubOperation::new(), Vec::new(), vec![state, one], None).unwrap()[0];
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![carried_extent, next],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
        }
        .unwrap();
        let mut builder = CompositeXlaProgramBuilder::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(body.entry_region_ref());
        let vector = builder.add_input(dynamic_vector_type.clone());
        let state = builder.add_input(state_type.clone());
        let extent = builder
            .add_instruction(
                DimensionSizeOperation::new(&dynamic_vector_type, 0).unwrap(),
                Vec::new(),
                vec![vector],
                None,
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(
                XlaOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                vec![extent, state],
                None,
            )
            .unwrap()[1];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &vec![dynamic_vector_type, state_type.clone()],
            &state_type,
            "main",
            None,
            None,
        )
        .unwrap();

        // The condition region `or`-reduces the carried `tensor<3xi1>` predicate, the array carry takes one
        // `stablehlo.select` against the carried predicate, and the dimension carry (`%iterArg`) is returned
        // unmasked exactly as the body produced it.
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<8xf32>, %arg1: tensor<3xf64>, %arg2: tensor<i32>) -> tensor<3xf64> {
                    %0 = stablehlo.set_dimension_size %arg0, %arg2, dim = 0 : (tensor<8xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<8>>
                    %1 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xf32, #stablehlo.bounds<8>>) -> tensor<i32>
                    %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i64>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3xf64>
                    %4 = stablehlo.compare GT, %arg1, %3, FLOAT : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xi1>
                    %5:3 = stablehlo.while(%iterArg = %2, %iterArg_0 = %arg1, %iterArg_1 = %4) : tensor<i64>, tensor<3xf64>, tensor<3xi1>
                    cond {
                      %c = stablehlo.constant dense<false> : tensor<i1>
                      %6 = stablehlo.reduce(%iterArg_1 init: %c) applies stablehlo.or across dimensions = [0] : (tensor<3xi1>, tensor<i1>) -> tensor<i1>
                      stablehlo.return %6 : tensor<i1>
                    } do {
                      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
                      %6 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<3xf64>
                      %7 = stablehlo.subtract %iterArg_0, %6 : tensor<3xf64>
                      %8 = stablehlo.select %iterArg_1, %7, %iterArg_0 : tensor<3xi1>, tensor<3xf64>
                      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                      %9 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<3xf64>
                      %10 = stablehlo.compare GT, %8, %9, FLOAT : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xi1>
                      stablehlo.return %iterArg, %8, %10 : tensor<i64>, tensor<3xf64>, tensor<3xi1>
                    }
                    return %5#1 : tensor<3xf64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_scan_to_while() {
        // A primal scan lowers to a `stablehlo.while` over `[counter, carries..., xs..., ys...]`: each iteration
        // reads one slice of the stacked inputs with `stablehlo.dynamic_slice`, inlines the body, and writes the
        // per-iteration outputs into preallocated zero accumulators with `stablehlo.dynamic_update_slice` (the
        // strategy JAX uses for `lax.scan`, which is not an XLA primitive).
        use ryft_core::ScanOperation as CoreScanOperation;

        let scalar_f32 = ArrayType::scalar(DataType::F32);
        let mut body_builder = CompositeXlaProgramBuilder::new();
        let carry = body_builder.add_input(scalar_f32.clone());
        let x = body_builder.add_input(scalar_f32.clone());
        let product = body_builder.add_instruction(MulOperation::new(), Vec::new(), vec![carry, x], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![product, product],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let scan = CoreScanOperation::<XlaConstant>::new(1, 3);

        let mut builder = CompositeXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let init = builder.add_input(scalar_f32.clone());
        let stacked_type = test_vector_type(3);
        let stacked_inputs = builder.add_input(stacked_type.clone());
        let outputs = builder
            .add_instruction(XlaOperation::Scan(scan), vec![body_region], vec![init, stacked_inputs], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                outputs,
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let output_types = (scalar_f32.clone(), stacked_type.clone());
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &(scalar_f32, stacked_type), &output_types, "main", None, None)
                .unwrap();

        assert!(stablehlo.contains("stablehlo.while"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.compare"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.dynamic_slice"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.dynamic_update_slice"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.multiply"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_forwards_scan_carried_dimension_extent() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let extent_type = DimensionType::new(extent.clone());
        let dynamic_vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone())]));

        let body = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let extent = builder.0.add_input(extent_type.into());
            let value = builder.add_input(dynamic_vector_type.clone());
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![extent, value],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder, Placeholder],
                )
                .unwrap()
        };

        let mut builder = CompositeXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let values = builder.add_input(dynamic_vector_type.clone());
        let extent = builder
            .add_instruction(
                DimensionSizeOperation::new(&dynamic_vector_type, 0).unwrap(),
                Vec::new(),
                vec![values],
                None,
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(
                XlaOperation::Scan(ScanOperation::new(2, 3)),
                vec![body_region],
                vec![extent, values],
                None,
            )
            .unwrap()[1];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &vec![dynamic_vector_type.clone()],
            &vec![dynamic_vector_type],
            "main",
            None,
            None,
        )
        .unwrap();

        assert!(stablehlo.contains("stablehlo.get_dimension_size"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.while"), "{stablehlo}");
        assert!(stablehlo.contains("tensor<?xf32, #stablehlo.bounds<8>>"), "{stablehlo}");
        assert!(!stablehlo.contains("dimension_from_scalar"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_dynamic_scan_length_as_scalar_ssa() {
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(9)).unwrap());
        let length_type = DimensionType::new(length.clone());
        let scalar_type = ArrayType::scalar(DataType::F32);
        let stacked_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(length.clone())]));

        let body = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let carry = builder.add_input(scalar_type.clone());
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![carry, carry],
                    vec![Placeholder],
                    vec![Placeholder, Placeholder],
                )
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let carry = builder.add_input(scalar_type.clone());
        let runtime_length = builder.0.add_input(length_type.clone().into());
        let outputs = builder
            .add_instruction(
                XlaOperation::Scan(ScanOperation::new(1, Dimension::Dynamic(length))),
                vec![body_region],
                vec![carry, runtime_length],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                outputs,
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &(scalar_type.clone(), ArrayType::scalar(DataType::I64)),
            &(scalar_type, stacked_type),
            "main",
            None,
            None,
        )
        .unwrap();

        assert!(stablehlo.contains("stablehlo.while"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.compare"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.set_dimension_size"), "{stablehlo}");
        assert!(stablehlo.contains("tensor<?xf32, #stablehlo.bounds<8>>"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_mapped_rng_as_dynamic_scan() {
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_type = DimensionType::new(batch.clone());
        let state_type = ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(2)]));
        let stacked_state_type =
            ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(2)]));
        let bits_type = ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Static(2)]));
        let stacked_bits_type =
            ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(2)]));

        let body = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let state = builder.add_input(state_type);
            let outputs = builder
                .add_instruction(
                    XlaOperation::RngBitGenerator(RngBitGeneratorOperation::new(RandomAlgorithm::ThreeFry, bits_type)),
                    Vec::new(),
                    vec![state],
                    None,
                )
                .unwrap()
                .to_vec();
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder], vec![Placeholder, Placeholder])
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let states = builder.add_input(stacked_state_type.clone());
        let runtime_batch = builder.0.add_input(batch_type.into());
        let outputs = builder
            .add_instruction(
                XlaOperation::Scan(ScanOperation::new(0, Dimension::Dynamic(batch))),
                vec![body_region],
                vec![states, runtime_batch],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                outputs,
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &(stacked_state_type.clone(), ArrayType::scalar(DataType::I64)),
            &(stacked_state_type, stacked_bits_type),
            "main",
            None,
            None,
        )
        .unwrap();

        assert!(stablehlo.contains("stablehlo.while"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.rng_bit_generator"), "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.dynamic_slice"), "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.dynamic_update_slice").count(), 2, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.set_dimension_size").count(), 3, "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_fully_unrolled_scan_without_while() {
        // A scan whose unroll factor equals its length lowers to straight-line operations: no `stablehlo.while` is
        // emitted at all and the body inlines once per iteration (three `stablehlo.multiply` copies for `length = 3`).
        use ryft_core::ScanOperation as CoreScanOperation;

        let scalar_f32 = ArrayType::scalar(DataType::F32);
        let mut body_builder = CompositeXlaProgramBuilder::new();
        let carry = body_builder.add_input(scalar_f32.clone());
        let x = body_builder.add_input(scalar_f32.clone());
        let product = body_builder.add_instruction(MulOperation::new(), Vec::new(), vec![carry, x], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![product, product],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let scan = CoreScanOperation::<XlaConstant>::new(1, 3).with_unroll(3).unwrap();

        let mut builder = CompositeXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let init = builder.add_input(scalar_f32.clone());
        let stacked_type = test_vector_type(3);
        let stacked_inputs = builder.add_input(stacked_type.clone());
        let outputs = builder
            .add_instruction(XlaOperation::Scan(scan), vec![body_region], vec![init, stacked_inputs], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                outputs,
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let output_types = (scalar_f32.clone(), stacked_type.clone());
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &(scalar_f32, stacked_type), &output_types, "main", None, None)
                .unwrap();

        assert!(!stablehlo.contains("stablehlo.while"), "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.multiply").count(), 3, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.dynamic_slice").count(), 3, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.dynamic_update_slice").count(), 3, "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_partially_unrolled_scan() {
        // A scan with `unroll = 2` over `length = 4` keeps the `stablehlo.while` skeleton but runs two body copies
        // per loop trip: the body region contains two `stablehlo.multiply` copies (and one iteration read/write pair
        // per copy) while the counter advances by the unroll factor.
        use ryft_core::ScanOperation as CoreScanOperation;

        let scalar_f32 = ArrayType::scalar(DataType::F32);
        let mut body_builder = CompositeXlaProgramBuilder::new();
        let carry = body_builder.add_input(scalar_f32.clone());
        let x = body_builder.add_input(scalar_f32.clone());
        let product = body_builder.add_instruction(MulOperation::new(), Vec::new(), vec![carry, x], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![product, product],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let scan = CoreScanOperation::<XlaConstant>::new(1, 4).with_unroll(2).unwrap();

        let mut builder = CompositeXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let init = builder.add_input(scalar_f32.clone());
        let stacked_type = test_vector_type(4);
        let stacked_inputs = builder.add_input(stacked_type.clone());
        let outputs = builder
            .add_instruction(XlaOperation::Scan(scan), vec![body_region], vec![init, stacked_inputs], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                outputs,
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let output_types = (scalar_f32.clone(), stacked_type.clone());
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &(scalar_f32, stacked_type), &output_types, "main", None, None)
                .unwrap();

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
        use ryft_core::PrintOperation;

        // Two prints in one flat program: the token chain is created lazily by one zero-operand
        // `stablehlo.after_all` at the first print, the second print consumes the first print's token result, and
        // each print's dataflow output is its forwarded operand (the final add reads `%arg0` and `%0`, not custom
        // call results).
        let array_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(array_type.clone());
        let doubled = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
        let first = builder.add_instruction(PrintOperation::new("first"), Vec::new(), vec![input], None).unwrap()[0];
        let second =
            builder.add_instruction(PrintOperation::new("second"), Vec::new(), vec![doubled], None).unwrap()[0];
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![first, second], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let program = unproject_plain_program(program);
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
        use ryft_core::{PrintOperation, StridedLayout, Tile, TileDimension, TiledLayout};

        // A side-effecting custom call lowers its typed attributes, complete memory-layout lists, buffer alias,
        // and one hidden ordered-I/O token. A second side-effecting call consumes that token even though the first
        // call's array result is unused, while the final pure call remains token-free.
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let column_major_type = array_type.clone().with_layout(Some(TiledLayout::new(vec![0, 1], Vec::new()).into()));
        let mut builder = XlaProgramBuilder::new();
        let left = builder.add_input(column_major_type.clone());
        let right = builder.add_input(array_type.clone());
        let effectful =
            CustomCallOperation::new("ryft.test.scaled_add", vec![array_type.clone(), column_major_type.clone()])
                .with_attribute("scale", 2.0)
                .with_attribute("count", 4i64)
                .with_attribute("flag", true)
                .with_attribute("label", "x")
                .with_input_output_alias(0, 1)
                .unwrap()
                .with_side_effect();
        builder.add_instruction(effectful, Vec::new(), vec![left, right], None).unwrap();
        let printed =
            builder.add_instruction(PrintOperation::new("between"), Vec::new(), vec![right], None).unwrap()[0];
        let chained = builder
            .add_instruction(
                CustomCallOperation::new("ryft.test.record", vec![array_type.clone()]).with_side_effect(),
                Vec::new(),
                vec![printed],
                None,
            )
            .unwrap()[0];
        let pure = CustomCallOperation::new("ryft.test.add_one", vec![array_type.clone()]);
        let output = builder.add_instruction(pure, Vec::new(), vec![chained], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![output],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let program = unproject_plain_program(program);
        let input_types = vec![column_major_type, array_type.clone()];
        let output_types = vec![array_type];
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1:3 = stablehlo.custom_call @ryft.test.scaled_add(%arg0, %arg1, %0) {api_version = 4 : i32, backend_config = {count = 4 : i64, flag = true, label = "x", scale = 2.000000e+00 : f64}, has_side_effect = true, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<2x3xf32>, tensor<2x3xf32>, !stablehlo.token) -> (tensor<2x3xf32>, tensor<2x3xf32>, !stablehlo.token)
                    %2 = stablehlo.custom_call @ryft.print(%arg1, %1#2) {api_version = 4 : i32, backend_config = {label = "between"}, has_side_effect = true} : (tensor<2x3xf32>, !stablehlo.token) -> !stablehlo.token
                    %3:2 = stablehlo.custom_call @ryft.test.record(%arg1, %2) {api_version = 4 : i32, backend_config = {}, has_side_effect = true} : (tensor<2x3xf32>, !stablehlo.token) -> (tensor<2x3xf32>, !stablehlo.token)
                    %4 = stablehlo.custom_call @ryft.test.add_one(%3#0) {api_version = 4 : i32, backend_config = {}} : (tensor<2x3xf32>) -> tensor<2x3xf32>
                    return %4 : tensor<2x3xf32>
                  }
                }
            "#},
        );

        // Byte-strided descriptors cannot be represented by StableHLO's permutation-only custom-call layouts.
        let strided_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
            .with_layout(Some(StridedLayout::new(vec![12, 4]).into()));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(strided_type.clone());
        let output = builder
            .add_instruction(
                CustomCallOperation::new("ryft.test.strided", vec![strided_type.clone()]),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let program = unproject_plain_program(program);
        assert_eq!(
            to_mlir_module_for_program(
                &program,
                &[],
                &vec![strided_type.clone()],
                &vec![strided_type],
                "main",
                None,
                None,
            ),
            Err(LoweringError::UnsupportedOp {
                op: "custom_call with strided array layout `strided{12,4}`".to_string(),
            }),
        );
        let invalid_permutation_type =
            test_matrix_type(2, 3).with_layout(Some(TiledLayout::new(vec![0, 0], Vec::new()).into()));
        assert_eq!(
            lower_custom_call_layout(&invalid_permutation_type),
            Err(LoweringError::UnsupportedOp {
                op: "custom_call with invalid array layout `tiled{0,0}` for rank-2 type \
                     `f32[2, 3][layout=tiled{0,0}]`"
                    .to_string(),
            }),
        );
        let tiled_type = test_matrix_type(2, 3)
            .with_layout(Some(TiledLayout::new(vec![1, 0], vec![Tile::new(vec![TileDimension::Sized(2)])]).into()));
        assert_eq!(
            lower_custom_call_layout(&tiled_type),
            Err(LoweringError::UnsupportedOp {
                op: "custom_call with tiled array layout `tiled{1,0:T(2)}`".to_string(),
            }),
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_dynamic_custom_call_alias() {
        use ryft_core::operations::custom_call::CustomCallOperation;

        // A composite custom call receives its declared extent as trailing scalar SSA metadata, but only the array
        // operand enters the FFI call. An alias continues to refer to that leading array operand and the result is
        // refined with the trailing logical extent after the call.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(dynamic_type.clone());
        let extent = builder
            .add_instruction(DimensionSizeOperation::new(&dynamic_type, 0).unwrap(), Vec::new(), vec![input], None)
            .unwrap()[0];
        let operation = CustomCallOperation::new("ryft.test.dynamic", vec![dynamic_type.clone()])
            .with_input_output_alias(0, 0)
            .unwrap();
        let output = builder.add_instruction(operation, Vec::new(), vec![input, extent], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &vec![dynamic_type.clone()],
            &vec![dynamic_type],
            "main",
            None,
            None,
        )
        .unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<8xf32>, %arg1: tensor<i32>) -> (tensor<?xf32, #stablehlo.bounds<8>>, tensor<i64>) {
                    %0 = stablehlo.set_dimension_size %arg0, %arg1, dim = 0 : (tensor<8xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<8>>
                    %1 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xf32, #stablehlo.bounds<8>>) -> tensor<i32>
                    %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i64>
                    %3 = stablehlo.custom_call @ryft.test.dynamic(%0) {api_version = 4 : i32, backend_config = {}, output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<?xf32, #stablehlo.bounds<8>>) -> tensor<?xf32, #stablehlo.bounds<8>>
                    %4 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
                    %5 = stablehlo.set_dimension_size %3, %4, dim = 0 : (tensor<?xf32, #stablehlo.bounds<8>>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<8>>
                    %6 = stablehlo.get_dimension_size %5, dim = 0 : (tensor<?xf32, #stablehlo.bounds<8>>) -> tensor<i32>
                    %7 = stablehlo.convert %6 : (tensor<i32>) -> tensor<i64>
                    return %5, %7 : tensor<?xf32, #stablehlo.bounds<8>>, tensor<i64>
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
            .add_instruction(SortOperation::new(0, SortDirection::Descending), Vec::new(), vec![keys, indices], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let program = unproject_plain_program(program);
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
            .add_instruction(operation, Vec::new(), vec![primary, secondary, passenger], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(outputs, vec![Placeholder; 3], vec![Placeholder; 3])
            .unwrap();
        let program = unproject_plain_program(program);
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
        use ryft_core::{DotDimensionNumbers, DotOperation};

        // An accumulation-typed dot lowers to a `stablehlo.dot_general` whose result type is the accumulation type
        // (XLA's `preferred_element_type` contract), with the operands kept at their narrow element type.
        let operand_type =
            ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let mut builder = XlaProgramBuilder::new();
        let lhs = builder.add_input(operand_type.clone());
        let rhs = builder.add_input(operand_type.clone());
        let operation = DotOperation::new(DotDimensionNumbers::matmul()).with_accumulation_type(DataType::F32);
        let output = builder.add_instruction(operation, Vec::new(), vec![lhs, rhs], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![output],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let program = unproject_plain_program(program);
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
        use ryft_core::ScaledDotOperation;

        let element_type =
            ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(16)]));
        let scale_type =
            ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let mut builder = XlaProgramBuilder::new();
        let inputs = vec![
            builder.add_input(element_type.clone()),
            builder.add_input(element_type.clone()),
            builder.add_input(scale_type.clone()),
            builder.add_input(scale_type.clone()),
        ];
        let output = builder
            .add_instruction(
                ScaledDotOperation::new(
                    DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()),
                    DataType::F32,
                    true,
                    true,
                ),
                Vec::new(),
                inputs,
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![output],
                vec![Placeholder; 4],
                vec![Placeholder],
            )
            .unwrap();
        let input_types = vec![element_type.clone(), element_type, scale_type.clone(), scale_type];
        (unproject_plain_program(program), input_types, vec![output_type])
    }

    #[test]
    fn test_named_composition_registry_supports_downstream_semantics() {
        // Registration consumes a semantic key and an ordinary typed program, not a Ryft operation enum variant.
        // Consequently a downstream lowerer can use the same sharing seam after performing its own family conversion.
        let boundary_type = ArrayType::scalar(DataType::F32);
        let (_, program) = DomainTracingContext::<XlaDomain<'static>>::trace(
            |inputs: Vec<XlaTracer<'static>>| Ok(inputs),
            vec![ArrayIrType::from(boundary_type.clone())],
        )
        .unwrap();
        let program = program.simplified().unwrap();
        let key = NamedCompositionKey {
            name: "downstream.example",
            version: 1,
            decomposition: "downstream.identity.v1",
            attributes: vec![("enabled", NamedCompositionAttribute::Boolean(true))],
            input_types: vec![boundary_type.clone()],
            output_types: vec![boundary_type],
        };
        let trace_count = Cell::new(0);
        let mut registry = NamedCompositionFunctionMap::default();
        registry
            .register(key.clone(), || {
                trace_count.set(trace_count.get() + 1);
                Ok(program.clone())
            })
            .unwrap();
        registry
            .register(key, || {
                trace_count.set(trace_count.get() + 1);
                Ok(program)
            })
            .unwrap();

        assert_eq!(trace_count.get(), 1);
        assert_eq!(registry.functions.len(), 1);
        assert_eq!(registry.order.len(), 1);
        assert_eq!(registry.functions.values().next().unwrap().symbol, "downstream.example");
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_scaled_dot_composition_without_target_platform() {
        // Target-independent StableHLO contains the exact named composite plus its portable typed decomposition;
        // the eventual XLA target decides whether to replace that boundary.
        let (program, input_types, output_types) = scaled_dot_fixture_program();
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func private @xla.scaled_dot(%arg0: tensor<2x16xf4E2M1FN>, %arg1: tensor<2x16xf4E2M1FN>, %arg2: tensor<2x1xf8E4M3FN>, %arg3: tensor<2x1xf8E4M3FN>) -> tensor<2x2xf32> {
                    %0 = stablehlo.convert %arg0 : (tensor<2x16xf4E2M1FN>) -> tensor<2x16xbf16>
                    %c = stablehlo.constant dense<2> : tensor<i64>
                    %c_0 = stablehlo.constant dense<1> : tensor<i64>
                    %c_1 = stablehlo.constant dense<16> : tensor<i64>
                    %1 = stablehlo.divide %c_1, %c_0 : tensor<i64>
                    %2 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<2x1xf8E4M3FN>) -> tensor<2x1x16xf8E4M3FN>
                    %c_2 = stablehlo.constant dense<2> : tensor<i64>
                    %3 = stablehlo.reshape %2 : (tensor<2x1x16xf8E4M3FN>) -> tensor<2x16xf8E4M3FN>
                    %4 = stablehlo.convert %3 : (tensor<2x16xf8E4M3FN>) -> tensor<2x16xbf16>
                    %5 = stablehlo.multiply %0, %4 : tensor<2x16xbf16>
                    %6 = stablehlo.convert %arg1 : (tensor<2x16xf4E2M1FN>) -> tensor<2x16xbf16>
                    %c_3 = stablehlo.constant dense<2> : tensor<i64>
                    %c_4 = stablehlo.constant dense<1> : tensor<i64>
                    %c_5 = stablehlo.constant dense<16> : tensor<i64>
                    %7 = stablehlo.divide %c_5, %c_4 : tensor<i64>
                    %8 = stablehlo.broadcast_in_dim %arg3, dims = [0, 1] : (tensor<2x1xf8E4M3FN>) -> tensor<2x1x16xf8E4M3FN>
                    %c_6 = stablehlo.constant dense<2> : tensor<i64>
                    %9 = stablehlo.reshape %8 : (tensor<2x1x16xf8E4M3FN>) -> tensor<2x16xf8E4M3FN>
                    %10 = stablehlo.convert %9 : (tensor<2x16xf8E4M3FN>) -> tensor<2x16xbf16>
                    %11 = stablehlo.multiply %6, %10 : tensor<2x16xbf16>
                    %12 = stablehlo.dot_general %5, %11, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x16xbf16>, tensor<2x16xbf16>) -> tensor<2x2xf32>
                    return %12 : tensor<2x2xf32>
                  }
                  func.func @main(%arg0: tensor<2x16xf4E2M1FN>, %arg1: tensor<2x16xf4E2M1FN>, %arg2: tensor<2x1xf8E4M3FN>, %arg3: tensor<2x1xf8E4M3FN>) -> tensor<2x2xf32> {
                    %0 = stablehlo.composite "xla.scaled_dot" %arg0, %arg1, %arg2, %arg3 {composite_attributes = {dimension_numbers = [[[1], [1]], [[], []]], preferred_element_type = f32}, decomposition = @xla.scaled_dot} : (tensor<2x16xf4E2M1FN>, tensor<2x16xf4E2M1FN>, tensor<2x1xf8E4M3FN>, tensor<2x1xf8E4M3FN>) -> tensor<2x2xf32>
                    return %0 : tensor<2x2xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_materializes_absent_scaled_dot_scales() {
        // The public optional-scale API still lowers to JAX's exact four-operand composite. Contracting dummy-scale
        // axes have size one, noncontracting axes retain the corresponding element geometry, and the decomposition
        // ignores the identity operands rather than imposing block-ratio requirements on an absent scale.
        let element_type =
            ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let mut builder = XlaProgramBuilder::new();
        let inputs = vec![builder.add_input(element_type.clone()), builder.add_input(element_type.clone())];
        let output = builder
            .add_instruction(
                ScaledDotOperation::new(
                    DotDimensionNumbers::new(vec![1], vec![0], Vec::new(), Vec::new()),
                    DataType::F32,
                    false,
                    false,
                ),
                Vec::new(),
                inputs,
                None,
            )
            .unwrap()[0];
        let program = unproject_plain_program(
            builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                    vec![output],
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap(),
        );
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &[element_type.clone(), element_type],
            &[output_type],
            "main",
            None,
            None,
        )
        .unwrap();

        assert!(module.contains(
            "func.func private @xla.scaled_dot(%arg0: tensor<2x2xf8E4M3FN>, %arg1: tensor<2x2xf8E4M3FN>, %arg2: tensor<2x1xf8E8M0FNU>, %arg3: tensor<1x2xf8E8M0FNU>)",
        ));
        assert_eq!(module.matches("dense<1.000000e+00> : tensor<f8E8M0FNU>").count(), 2);
        assert!(module.contains("stablehlo.composite \"xla.scaled_dot\" %arg0, %arg1"));
        assert!(!module.contains("func.call @xla.scaled_dot"));
    }

    #[test]
    fn test_lower_mlir_module_for_program_preserves_scaled_dot_composite_on_cuda() {
        // CUDA lowering preserves the exact named-composite boundary and its private semantic fallback. XLA's
        // block-scaling replacement pass, rather than Ryft's StableHLO emitter, decides whether to select a native
        // implementation for the target device.
        let (program, input_types, output_types) = scaled_dot_fixture_program();
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;

        assert!(module.contains("func.func private @xla.scaled_dot"));
        assert!(module.contains("stablehlo.composite \"xla.scaled_dot\" %arg0, %arg1, %arg2, %arg3"));
        assert!(!module.contains("stablehlo.custom_call @__op$block_scaled_dot"));
    }

    #[test]
    fn test_lower_mlir_module_for_program_physicalizes_dynamic_scaled_dot_on_cuda() {
        // CUDA's fused scaled-dot HLO requires a static boundary. Ryft masks the bounded physical operands, keeps the
        // named composite and its decomposition static, and restores the logical row extent on the result.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(5)).unwrap());
        let lhs_type = ArrayType::new(
            DataType::F8E4M3FN,
            Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Static(32)]),
        );
        let rhs_type =
            ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(1), Dimension::Static(32)]));
        let lhs_scale_type = ArrayType::new(
            DataType::F8E8M0FNU,
            Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Static(1)]),
        );
        let rhs_scale_type =
            ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(1), Dimension::Static(1)]));
        let output_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows), Dimension::Static(1)]));
        let input_types = vec![lhs_type, rhs_type, lhs_scale_type, rhs_scale_type];
        let output_types = vec![output_type];
        let mut builder = XlaProgramBuilder::new();
        let inputs = input_types.iter().cloned().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let output = builder
            .add_instruction(
                ScaledDotOperation::new(
                    DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()),
                    DataType::F32,
                    true,
                    true,
                ),
                Vec::new(),
                inputs,
                None,
            )
            .unwrap()[0];
        let program = unproject_plain_program(
            builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                    vec![output],
                    vec![Placeholder; 4],
                    vec![Placeholder],
                )
                .unwrap(),
        );
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;

        assert!(module.contains(
            "func.func private @xla.scaled_dot(%arg0: tensor<4x32xf8E4M3FN>, %arg1: tensor<1x32xf8E4M3FN>, %arg2: tensor<4x1xf8E8M0FNU>, %arg3: tensor<1x1xf8E8M0FNU>) -> tensor<4x1xf32>",
        ));
        assert!(module.contains("stablehlo.composite \"xla.scaled_dot\"",));
        assert!(module.contains(
            ": (tensor<4x32xf8E4M3FN>, tensor<1x32xf8E4M3FN>, tensor<4x1xf8E8M0FNU>, tensor<1x1xf8E8M0FNU>) -> tensor<4x1xf32>",
        ));
        assert!(module.contains("stablehlo.set_dimension_size"));
    }

    #[test]
    fn test_lower_mlir_module_for_program_retains_dynamic_scaled_dot_requirements_on_cuda() {
        // A dynamic contracting block ratio is a runtime semantic requirement. Keep that case on the logical
        // decomposition instead of erasing its dimension checks merely to reach CUDA's static fused boundary.
        let elements = DimensionVariable::new("elements", DimensionBounds::new(32, Some(65)).unwrap());
        let blocks = DimensionVariable::new("blocks", DimensionBounds::new(1, Some(3)).unwrap());
        let element_type =
            ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(1), Dimension::Dynamic(elements)]));
        let scale_type =
            ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(1), Dimension::Dynamic(blocks)]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1), Dimension::Static(1)]));
        let input_types = vec![element_type.clone(), element_type, scale_type.clone(), scale_type];
        let mut builder = XlaProgramBuilder::new();
        let inputs = input_types.iter().cloned().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let output = builder
            .add_instruction(
                ScaledDotOperation::new(
                    DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()),
                    DataType::F32,
                    true,
                    true,
                ),
                Vec::new(),
                inputs,
                None,
            )
            .unwrap()[0];
        let program = unproject_plain_program(
            builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                    vec![output],
                    vec![Placeholder; 4],
                    vec![Placeholder],
                )
                .unwrap(),
        );
        let module = lower_mlir_module_for_program(
            &program,
            &[],
            &input_types,
            &[output_type],
            "main",
            None,
            None,
            Some("cuda"),
        )
        .unwrap()
        .stable_hlo;

        assert!(module.contains("call @xla.scaled_dot"));
        assert!(module.contains("stablehlo.custom_call @ryft.assert"));
        assert!(!module.contains("stablehlo.composite \"xla.scaled_dot\""));
    }

    #[test]
    fn test_lower_mlir_module_for_program_emits_scaled_dot_composite_in_shared_jit_call_callee() {
        // A scaled-dot composite nested inside a shared `jit_call` callee is emitted once, while both outer call sites
        // reuse the same function and XLA retains the opportunity to replace the composite after inlining.
        use ryft_core::ScaledDotOperation;

        let element_type =
            ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(16)]));
        let scale_type =
            ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let mut callee_builder = XlaProgramBuilder::new();
        let callee_inputs = vec![
            callee_builder.add_input(element_type.clone()),
            callee_builder.add_input(element_type.clone()),
            callee_builder.add_input(scale_type.clone()),
            callee_builder.add_input(scale_type.clone()),
        ];
        let callee_output = callee_builder
            .add_instruction(
                ScaledDotOperation::new(
                    DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()),
                    DataType::F32,
                    true,
                    true,
                ),
                Vec::new(),
                callee_inputs,
                None,
            )
            .unwrap()[0];
        let callee = Arc::new(unproject_plain_program(
            callee_builder.build(vec![callee_output], vec![Placeholder; 4], vec![Placeholder]).unwrap(),
        ));

        let mut builder = CompositeXlaProgramBuilder::new();
        let inputs = vec![
            builder.add_input(element_type.clone()),
            builder.add_input(element_type.clone()),
            builder.add_input(scale_type.clone()),
            builder.add_input(scale_type.clone()),
        ];
        let first = add_xla_jit_call(&mut builder, &callee, inputs.clone());
        let second = add_xla_jit_call(&mut builder, &callee, inputs);
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![first, second], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 4], vec![Placeholder])
            .unwrap();
        let input_types = vec![element_type.clone(), element_type, scale_type.clone(), scale_type];
        let module = lower_mlir_module_for_program(
            &program,
            &[],
            &input_types,
            &[output_type],
            "main",
            None,
            None,
            Some("cuda"),
        )
        .unwrap()
        .stable_hlo;

        assert_eq!(module.matches("func.func private @jit_call_0").count(), 1);
        assert_eq!(module.matches("stablehlo.composite \"xla.scaled_dot\"").count(), 1);
        assert_eq!(module.matches("call @jit_call_0").count(), 2);
    }

    /// Builds the attention program shared by the platform-gated lowering fixtures below.
    fn dot_product_attention_fixture_program(
        data_type: DataType,
        head_dimension: usize,
        causal: bool,
    ) -> (XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>, Vec<ArrayType>, Vec<ArrayType>) {
        let operand_type = attention_array_type(data_type, &[1, 4, 2, head_dimension]);
        let mut builder = XlaProgramBuilder::new();
        let inputs = vec![
            builder.add_input(operand_type.clone()),
            builder.add_input(operand_type.clone()),
            builder.add_input(operand_type.clone()),
        ];
        let output = builder
            .add_instruction(
                attention_operation(AttentionConfiguration::new().with_scale(0.125).with_causal(causal)),
                Vec::new(),
                inputs,
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![output],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap();
        let input_types = vec![operand_type.clone(), operand_type.clone(), operand_type.clone()];
        (unproject_plain_program(program), input_types, vec![operand_type])
    }

    #[test]
    fn test_lower_mlir_module_for_program_emits_fmha_softmax_on_cuda() {
        // Qualifying causal BF16 attention uses the cuDNN fMHA ABI and preserves the causal mask.
        let (program, input_types, output_types) = dot_product_attention_fixture_program(DataType::BF16, 8, true);
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;

        assert!(module.contains("stablehlo.custom_call @__cudnn$fmhaSoftmax"));
        assert!(module.contains("CAUSAL"));
        assert!(module.contains("BF16"));
    }
    #[test]
    fn test_lower_mlir_module_for_program_emits_fmha_softmax_without_mask_on_cuda() {
        // The unmasked F16 form uses the same fMHA target with NO_MASK and the F16 element contract.
        let (program, input_types, output_types) = dot_product_attention_fixture_program(DataType::F16, 8, false);
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;

        assert!(module.contains("stablehlo.custom_call @__cudnn$fmhaSoftmax"));
        assert!(module.contains("NO_MASK"));
        assert!(module.contains("F16"));
    }
    #[test]
    fn test_to_mlir_module_for_program_lowers_dot_product_attention_composition_without_target_platform() {
        // Without target information, the entry function calls the one private typed decomposition containing the
        // score dot, causal mask, stable softmax, value dot, and final layout restoration.
        let (program, input_types, output_types) = dot_product_attention_fixture_program(DataType::F32, 8, true);
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(module.matches("func.func private @ryft.dot_product_attention").count(), 1);
        assert_eq!(module.matches("call @ryft.dot_product_attention").count(), 1);
        assert_eq!(module.matches("stablehlo.dot_general").count(), 2);
        assert!(module.contains("stablehlo.compare LE"));
        assert!(module.contains("stablehlo.exponential"));
        assert!(!module.contains("__cudnn$fmha"));
    }

    #[test]
    fn test_lower_mlir_module_for_program_lowers_dot_product_attention_composition_for_f32_on_cuda() {
        // `f32` operands do not qualify for the cuDNN flash-attention call, so even a CUDA target lowers the
        // portable composition.
        let (program, input_types, output_types) = dot_product_attention_fixture_program(DataType::F32, 8, true);
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
        let (program, input_types, output_types) = dot_product_attention_fixture_program(DataType::BF16, 4, true);
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
        let operation = DotProductAttentionOperation::new(
            operation.configuration(),
            AttentionOperandSignature::new(bias, false, sequence_lengths, sequence_lengths),
        );
        let inputs = input_types.iter().map(|input_type| builder.add_input(input_type.clone())).collect::<Vec<_>>();
        let outputs = builder.add_instruction(operation, Vec::new(), inputs, None).unwrap().to_vec();
        let output_count = outputs.len();
        let output_types = builder_output_types(&builder, outputs.as_slice());
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                outputs,
                vec![Placeholder; input_types.len()],
                vec![Placeholder; output_count],
            )
            .unwrap();
        (unproject_plain_program(program), input_types, output_types)
    }

    /// Builds the matching backward fixture program for [`dot_product_attention_extended_fixture_program`]'s
    /// canonical operand order: `(query, key, value[, bias][, query lengths, key/value lengths], output, residual,
    /// output cotangent)`.
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
            data_type,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(4), Dimension::Static(query_heads)]),
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
        input_types.extend([query_type.clone(), activation_type, query_type]);
        let operation = DotProductAttentionBackwardOperation::new(
            operation.configuration(),
            AttentionOperandSignature::new(bias, false, sequence_lengths, sequence_lengths),
        );
        let inputs = input_types.iter().map(|input_type| builder.add_input(input_type.clone())).collect::<Vec<_>>();
        let outputs = builder.add_instruction(operation, Vec::new(), inputs, None).unwrap().to_vec();
        let output_count = outputs.len();
        let output_types = builder_output_types(&builder, outputs.as_slice());
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                outputs,
                vec![Placeholder; input_types.len()],
                vec![Placeholder; output_count],
            )
            .unwrap();
        (unproject_plain_program(program), input_types, output_types)
    }

    #[test]
    fn test_lower_mlir_module_for_program_emits_fmha_scale_bias_softmax_on_cuda() {
        // A broadcastable user bias is expanded to the physical score geometry before entering the ScaleBias ABI.
        let (program, input_types, output_types) = dot_product_attention_extended_fixture_program(
            DataType::BF16,
            2,
            2,
            true,
            false,
            attention_operation(AttentionConfiguration::new().with_scale(0.125).with_causal(true)),
        );
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;

        assert!(module.contains("stablehlo.broadcast_in_dim %arg3"));
        assert!(module.contains("stablehlo.custom_call @__cudnn$fmhaScaleBiasSoftmax"));
        assert!(module.contains("tensor<2x2x4x4xbf16>"));
    }
    #[test]
    fn test_lower_mlir_module_for_program_emits_fmha_training_forward_with_padding_on_cuda() {
        // Training attention carries both length vectors and converts the f32 kernel statistic to the public residual.
        let (program, input_types, output_types) = dot_product_attention_extended_fixture_program(
            DataType::BF16,
            2,
            2,
            false,
            true,
            attention_operation(AttentionConfiguration::new().with_scale(0.125).with_causal(true).with_residual(true)),
        );
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;

        assert!(module.contains("stablehlo.custom_call @__cudnn$fmhaSoftmax"));
        assert!(module.contains("PADDING_CAUSAL"));
        assert!(module.contains("tensor<2x2x4xf32>"));
        assert!(module.contains("stablehlo.convert"));
    }
    #[test]
    fn test_lower_mlir_module_for_program_emits_fmha_dropout_with_sliding_window_on_cuda() {
        // Dropout selects its target and records the deterministic seed and exclusive left-window length.
        let (program, input_types, output_types) = dot_product_attention_extended_fixture_program(
            DataType::BF16,
            2,
            2,
            false,
            false,
            attention_operation(
                AttentionConfiguration::new()
                    .with_scale(0.125)
                    .with_causal(true)
                    .with_local_window((1, 0))
                    .with_implementation(AttentionImplementation::Fused)
                    .with_dropout((0.5, 123)),
            ),
        );
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;

        assert!(module.contains("stablehlo.custom_call @__cudnn$fmhaSoftmaxDropout"));
        assert!(module.contains("dropout_rate"));
        assert!(module.contains("123"));
        assert!(module.contains("sliding_window_length"));
    }
    #[test]
    fn test_lower_mlir_module_for_program_emits_fmha_backward_with_bias_on_cuda() {
        // The fused backward restores operand gradients and sums the score-shaped gradient to the user bias shape.
        let (program, input_types, output_types) = dot_product_attention_backward_fixture_program(
            DataType::BF16,
            2,
            2,
            true,
            false,
            attention_backward_operation(AttentionConfiguration::new().with_scale(0.125).with_causal(true)),
        );
        let module =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"))
                .unwrap()
                .stable_hlo;

        assert!(module.contains("stablehlo.custom_call @__cudnn$fmhaScaleBiasSoftmaxBackward"));
        assert!(module.matches("stablehlo.transpose").count() >= 3);
        assert!(module.contains("stablehlo.reduce"));
        assert!(module.contains("tensor<1x2x4x4xbf16>"));
    }
    #[test]
    fn test_to_mlir_module_for_program_lowers_dot_product_attention_composition_with_extensions() {
        // GQA expansion, broadcast bias, causal/local masking, and residual production all live in one private typed
        // decomposition rather than in the backend adapter.
        let (program, input_types, output_types) = dot_product_attention_extended_fixture_program(
            DataType::F32,
            4,
            2,
            true,
            false,
            attention_operation(
                AttentionConfiguration::new()
                    .with_scale(0.125)
                    .with_causal(true)
                    .with_local_window((1, 0))
                    .with_residual(true),
            ),
        );
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(module.matches("func.func private @ryft.dot_product_attention").count(), 1);
        assert_eq!(module.matches("call @ryft.dot_product_attention").count(), 1);
        assert!(module.contains("stablehlo.broadcast_in_dim"));
        assert!(module.contains("stablehlo.reshape"));
        assert!(module.contains("stablehlo.and"));
        assert!(module.contains("stablehlo.log"));
        assert!(module.contains("-> (tensor<2x4x4x8xf32>, tensor<2x4x4xf32>)"));
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_dot_product_attention_backward_composition() {
        // The private typed backward decomposition recomputes normalized probabilities and emits the query, grouped
        // key/value, and broadcast-bias cotangent reductions.
        let (program, input_types, output_types) = dot_product_attention_backward_fixture_program(
            DataType::F32,
            4,
            2,
            true,
            false,
            attention_backward_operation(AttentionConfiguration::new().with_scale(0.125).with_causal(true)),
        );
        let module =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();

        assert_eq!(module.matches("func.func private @ryft.dot_product_attention_backward").count(), 1);
        assert_eq!(module.matches("call @ryft.dot_product_attention_backward").count(), 1);
        assert!(module.matches("stablehlo.dot_general").count() >= 4);
        assert!(module.contains("stablehlo.exponential"));
        assert!(module.contains("stablehlo.reduce"));
        assert!(module.contains("tensor<1x4x4x4xf32>"));
        assert!(!module.contains("__cudnn$fmha"));
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
            attention_operation(
                AttentionConfiguration::new()
                    .with_scale(0.125)
                    .with_causal(true)
                    .with_implementation(AttentionImplementation::Fused)
                    .with_dropout((0.5, 123)),
            ),
        );
        for platform in [Some("cuda"), None] {
            let result =
                lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, platform);
            assert!(matches!(
                result,
                Err(error) if error.to_string().contains(
                    "`dot_product_attention` dropout is only supported by the fused CUDA lowering",
                ),
            ));
        }
        let (program, input_types, output_types) = dot_product_attention_backward_fixture_program(
            DataType::F32,
            2,
            2,
            false,
            false,
            attention_backward_operation(
                AttentionConfiguration::new()
                    .with_scale(0.125)
                    .with_causal(true)
                    .with_implementation(AttentionImplementation::Fused)
                    .with_dropout((0.5, 123)),
            ),
        );
        let result =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"));
        assert!(matches!(
            result,
            Err(error) if error.to_string().contains(
                "`dot_product_attention_backward` dropout is only supported by the fused CUDA lowering",
            ),
        ));
    }

    #[test]
    fn test_lower_mlir_module_for_program_rejects_padded_fmha_with_a_right_local_window() {
        // The pinned cuDNN ABI supports a nonzero right radius only for the unpadded causal mask kind. Forced fused
        // selection therefore reports this semantic limitation instead of silently dropping the right radius.
        let (program, input_types, output_types) = dot_product_attention_extended_fixture_program(
            DataType::BF16,
            2,
            2,
            false,
            true,
            attention_operation(
                AttentionConfiguration::new()
                    .with_causal(true)
                    .with_local_window((2, 1))
                    .with_implementation(AttentionImplementation::Fused),
            ),
        );

        let result =
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, Some("cuda"));

        assert!(matches!(
            result,
            Err(error) if error.to_string().contains(
                "a nonzero right local-window radius requires causal masking without sequence-length padding",
            ),
        ));
    }

    #[test]
    fn test_lower_mlir_module_for_program_lowers_dynamically_shaped_attention_compositions() {
        // Portable attention obtains every logical extent from the mixed array IR, so bounded-dynamic programs do
        // not need to expose physical padding as sequence-length operands and support the same GQA and window
        // configurations as static programs. Explicit lengths remain an independent semantic mask when provided.
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(4)).unwrap());
        let operand_type = |heads: usize| {
            ArrayType::new(
                DataType::BF16,
                Shape::new(vec![
                    Dimension::Dynamic(batch.clone()),
                    Dimension::Static(4),
                    Dimension::Static(heads),
                    Dimension::Static(8),
                ]),
            )
        };
        let activation_type = ArrayType::new(
            DataType::BF16,
            Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(4), Dimension::Static(2)]),
        );
        let lengths_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(batch.clone())]));
        let build_forward = |key_value_heads: usize,
                             sequence_lengths: bool,
                             operation: DotProductAttentionOperation| {
            let mut input_types = vec![operand_type(2), operand_type(key_value_heads), operand_type(key_value_heads)];
            if sequence_lengths {
                input_types.push(lengths_type.clone());
                input_types.push(lengths_type.clone());
            }
            let operation = DotProductAttentionOperation::new(
                operation.configuration(),
                AttentionOperandSignature::new(false, false, sequence_lengths, sequence_lengths),
            );
            let mut builder = XlaProgramBuilder::new();
            let inputs = input_types.iter().map(|input_type| builder.add_input(input_type.clone())).collect::<Vec<_>>();
            let outputs = builder.add_instruction(operation, Vec::new(), inputs, None).unwrap().to_vec();
            let output_count = outputs.len();
            let output_types = builder_output_types(&builder, outputs.as_slice());
            let program = builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                    outputs,
                    vec![Placeholder; input_types.len()],
                    vec![Placeholder; output_count],
                )
                .unwrap();
            (unproject_plain_program(program), input_types, output_types)
        };
        let build_backward = |key_value_heads: usize,
                              sequence_lengths: bool,
                              operation: DotProductAttentionBackwardOperation| {
            let mut input_types = vec![operand_type(2), operand_type(key_value_heads), operand_type(key_value_heads)];
            if sequence_lengths {
                input_types.push(lengths_type.clone());
                input_types.push(lengths_type.clone());
            }
            input_types.extend([operand_type(2), activation_type.clone(), operand_type(2)]);
            let operation = DotProductAttentionBackwardOperation::new(
                operation.configuration(),
                AttentionOperandSignature::new(false, false, sequence_lengths, sequence_lengths),
            );
            let mut builder = XlaProgramBuilder::new();
            let inputs = input_types.iter().map(|input_type| builder.add_input(input_type.clone())).collect::<Vec<_>>();
            let outputs = builder.add_instruction(operation, Vec::new(), inputs, None).unwrap().to_vec();
            let output_count = outputs.len();
            let output_types = builder_output_types(&builder, outputs.as_slice());
            let program = builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                    outputs,
                    vec![Placeholder; input_types.len()],
                    vec![Placeholder; output_count],
                )
                .unwrap();
            (unproject_plain_program(program), input_types, output_types)
        };

        // Dynamic forward attention without explicit sequence lengths uses the tensor's logical dimensions.
        let (program, input_types, output_types) =
            build_forward(2, false, attention_operation(AttentionConfiguration::new().with_scale(0.125)));
        lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, None).unwrap();

        // Sliding-window masking and grouped-query head expansion are ordinary semantic operations.
        for (key_value_heads, operation) in [
            (
                2,
                attention_operation(
                    AttentionConfiguration::new().with_scale(0.125).with_causal(true).with_local_window((1, 0)),
                ),
            ),
            (1, attention_operation(AttentionConfiguration::new().with_scale(0.125))),
        ] {
            let (program, input_types, output_types) = build_forward(key_value_heads, true, operation);
            lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, None)
                .unwrap();
        }

        // The analytical backward uses the same first-class geometry and mask construction.
        let (program, input_types, output_types) =
            build_backward(2, false, attention_backward_operation(AttentionConfiguration::new().with_scale(0.125)));
        lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, None).unwrap();

        let (program, input_types, output_types) = build_backward(
            2,
            true,
            attention_backward_operation(
                AttentionConfiguration::new().with_scale(0.125).with_causal(true).with_local_window((1, 0)),
            ),
        );
        lower_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None, None).unwrap();
    }

    #[test]
    fn test_lower_mlir_module_for_program_lowers_jax_attention_surface() {
        use ryft_core::operations::attention::AttentionConfiguration;

        // Rank-three attention, an arbitrary Boolean mask, one independently present sequence-length operand, an
        // asymmetric local window, default scaling, and a residual all lower through the typed portable composition.
        let query_type = attention_array_type(DataType::F32, &[2, 1, 2]);
        let key_value_type = attention_array_type(DataType::F32, &[3, 1, 2]);
        let mask_type = attention_array_type(DataType::Boolean, &[1, 2, 3]);
        let lengths_type = attention_array_type(DataType::I32, &[1]);
        let signature = AttentionOperandSignature::new(false, true, true, false);
        let operation = DotProductAttentionOperation::new(
            AttentionConfiguration::new()
                .with_local_window(Some((2, 1)))
                .with_implementation(AttentionImplementation::Portable)
                .with_residual(true),
            signature,
        );
        let input_types = vec![query_type.clone(), key_value_type.clone(), key_value_type, mask_type, lengths_type];
        let mut builder = XlaProgramBuilder::new();
        let inputs = input_types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let outputs = builder.add_instruction(operation, Vec::new(), inputs, None).unwrap().to_vec();
        let output_types = builder_output_types(&builder, outputs.as_slice());
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                outputs,
                vec![Placeholder; input_types.len()],
                vec![Placeholder; 2],
            )
            .unwrap();
        let module = lower_mlir_module_for_program(
            &unproject_plain_program(program),
            &[],
            &input_types,
            &output_types,
            "main",
            None,
            None,
            None,
        )
        .unwrap()
        .stable_hlo;

        assert!(module.contains("func.func private @ryft.dot_product_attention"));
        assert!(module.contains("stablehlo.and"));
        assert!(module.contains("stablehlo.exponential"));
        assert!(!module.contains("__cudnn$fmha"));
        assert_eq!(output_types[0], query_type);
        assert_eq!(output_types[1].shape(), &Shape::new(vec![2.into(), 1.into()]));
    }

    #[test]
    fn test_lower_mlir_module_for_program_fuses_jax_attention_surface_on_cuda() {
        // The cuDNN adapter normalizes TNH inputs, turns an arbitrary Boolean mask into additive bias, and synthesizes
        // the absent key/value length vector while preserving a supported causal left window.
        let query_type = attention_array_type(DataType::F16, &[4, 2, 8]);
        let key_value_type = attention_array_type(DataType::F16, &[4, 2, 8]);
        let mask_type = attention_array_type(DataType::Boolean, &[1, 4, 4]);
        let lengths_type = attention_array_type(DataType::I32, &[1]);
        let residual_type = attention_array_type(DataType::F16, &[4, 2]);
        let signature = AttentionOperandSignature::new(false, true, true, false);
        let configuration = AttentionConfiguration::new()
            .with_causal(true)
            .with_local_window((2, 0))
            .with_implementation(AttentionImplementation::Fused)
            .with_residual(true);
        let forward_input_types = vec![
            query_type.clone(),
            key_value_type.clone(),
            key_value_type.clone(),
            mask_type.clone(),
            lengths_type.clone(),
        ];
        let mut builder = XlaProgramBuilder::new();
        let inputs = forward_input_types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let outputs = builder
            .add_instruction(DotProductAttentionOperation::new(configuration, signature), Vec::new(), inputs, None)
            .unwrap()
            .to_vec();
        let forward_output_types = builder_output_types(&builder, outputs.as_slice());
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                outputs,
                vec![Placeholder; forward_input_types.len()],
                vec![Placeholder; 2],
            )
            .unwrap();
        let forward = lower_mlir_module_for_program(
            &unproject_plain_program(program),
            &[],
            &forward_input_types,
            &forward_output_types,
            "main",
            None,
            None,
            Some("cuda"),
        )
        .unwrap()
        .stable_hlo;

        assert!(forward.contains("stablehlo.custom_call @__cudnn$fmhaScaleBiasSoftmax"));
        assert!(forward.contains("PADDING_CAUSAL"));
        assert!(forward.contains("stablehlo.select"));
        assert!(forward.contains("stablehlo.get_dimension_size %arg1, dim = 0"));
        assert!(forward.contains("tensor<4x2x8xf16>"));
        assert_eq!(forward_output_types, vec![query_type.clone(), residual_type.clone()]);

        // The matching backward consumes the same canonical optional prefix and discards the synthetic mask-bias
        // cotangent while preserving the three differentiable operand cotangents.
        let backward_input_types = forward_input_types
            .iter()
            .cloned()
            .chain([query_type.clone(), residual_type, query_type.clone()])
            .collect::<Vec<_>>();
        let mut builder = XlaProgramBuilder::new();
        let inputs = backward_input_types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let outputs = builder
            .add_instruction(
                DotProductAttentionBackwardOperation::new(configuration, signature),
                Vec::new(),
                inputs,
                None,
            )
            .unwrap()
            .to_vec();
        let backward_output_types = builder_output_types(&builder, outputs.as_slice());
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                outputs,
                vec![Placeholder; backward_input_types.len()],
                vec![Placeholder; 3],
            )
            .unwrap();
        let backward = lower_mlir_module_for_program(
            &unproject_plain_program(program),
            &[],
            &backward_input_types,
            &backward_output_types,
            "main",
            None,
            None,
            Some("cuda"),
        )
        .unwrap()
        .stable_hlo;

        assert!(backward.contains("stablehlo.custom_call @__cudnn$fmhaScaleBiasSoftmaxBackward"));
        assert_eq!(backward_output_types, vec![query_type.clone(), key_value_type.clone(), key_value_type]);
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
            attention_operation(AttentionConfiguration::new().with_scale(0.125).with_causal(true)),
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
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(outputs, vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        let program = unproject_plain_program(program);
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
        use ryft_core::{PrintOperation, ScanOperation as CoreScanOperation};

        // A print inside a scan body makes the lowered `stablehlo.while` carry the effect token as one extra
        // trailing state element: the entry token is created before the loop, both regions receive it as an extra
        // block argument, the body threads it through the print custom call, and the loop's trailing result
        // continues the chain (unused here because the program ends right after the scan).
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut body_builder = CompositeXlaProgramBuilder::new();
        let carry = body_builder.add_input(scalar_f64.clone());
        let x = body_builder.add_input(scalar_f64.clone());
        let printed =
            body_builder.add_instruction(PrintOperation::new("iteration"), Vec::new(), vec![x], None).unwrap()[0];
        let sum = body_builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, printed], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        let scan = CoreScanOperation::<XlaConstant>::new(1, 3);

        let stacked_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = CompositeXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let init = builder.add_input(scalar_f64.clone());
        let stacked_inputs = builder.add_input(stacked_type.clone());
        let output = builder
            .add_instruction(XlaOperation::Scan(scan), vec![body_region], vec![init, stacked_inputs], None)
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
    fn test_to_mlir_module_for_program_threads_both_ordered_effect_classes_through_scan() {
        use ryft_core::{DimensionFromScalarOperation, PrintOperation, ScanOperation as CoreScanOperation};

        // The scan body uses both ordered classes: the checked gateway contributes an assertion and the print
        // contributes ordered I/O. The loop state carries two independent trailing tokens in canonical class order.
        let scalar_type = ArrayType::scalar(DataType::I64);
        let mut body_builder = CompositeXlaProgramBuilder::new();
        let carry = body_builder.add_input(scalar_type.clone());
        let value = body_builder.add_input(scalar_type.clone());
        body_builder
            .add_instruction(
                DimensionFromScalarOperation::new(DimensionVariable::new(
                    "extent",
                    DimensionBounds::new(0, Some(5)).unwrap(),
                )),
                Vec::new(),
                vec![value],
                None,
            )
            .unwrap();
        body_builder
            .add_instruction(PrintOperation::new("iteration"), Vec::new(), vec![value], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![carry], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        let scan = CoreScanOperation::<XlaConstant>::new(1, 3);

        let stacked_type = ArrayType::new(DataType::I64, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = CompositeXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type.clone());
        let stacked = builder.add_input(stacked_type.clone());
        let output = builder
            .add_instruction(XlaOperation::Scan(scan), vec![body_region], vec![initial, stacked], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &vec![scalar_type.clone(), stacked_type],
            &vec![scalar_type],
            "main",
            None,
            None,
        )
        .unwrap();

        assert_eq!(stablehlo.matches("stablehlo.after_all").count(), 2, "{stablehlo}");
        assert_eq!(stablehlo.matches("@ryft.assert").count(), 1, "{stablehlo}");
        assert_eq!(stablehlo.matches("@ryft.print").count(), 1, "{stablehlo}");
        let while_header = stablehlo.lines().find(|line| line.contains("stablehlo.while(")).unwrap();
        assert_eq!(while_header.matches("!stablehlo.token").count(), 2, "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_condition_branch_print_with_token_result() {
        use ryft_core::PrintOperation;

        // A print inside one condition branch makes the lowered `stablehlo.if` return the branch's final effect
        // token as one extra trailing result: both branches capture the entry token implicitly, the effectful branch
        // returns its print custom call's token, and the pure branch returns the entry token unchanged.
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let input_type = ArrayType::scalar(DataType::F64);
        let mut true_builder = CompositeXlaProgramBuilder::new();
        let true_input = true_builder.add_input(input_type.clone());
        let printed = true_builder
            .add_instruction(PrintOperation::new("taken"), Vec::new(), vec![true_input], None)
            .unwrap()[0];
        let negated = true_builder.add_instruction(NegOperation::new(), Vec::new(), vec![printed], None).unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![negated], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = CompositeXlaProgramBuilder::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_program(unproject_plain_program(xla_identity_branch(input_type.clone())));
        let predicate = builder.add_input(predicate_type.clone());
        let input = builder.add_input(input_type.clone());
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
                None,
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
    fn test_to_mlir_module_for_program_threads_branch_effect_class_union_independently() {
        use ryft_core::{DimensionFromScalarOperation, PrintOperation};

        // Each branch uses a different ordered class. StableHLO requires identical branch result signatures, so both
        // branches return the union's two tokens while forwarding the untouched entry token for the unused class.
        let scalar_type = ArrayType::scalar(DataType::I64);
        let true_branch = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let input = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(
                    DimensionFromScalarOperation::new(DimensionVariable::new(
                        "extent",
                        DimensionBounds::new(0, Some(5)).unwrap(),
                    )),
                    Vec::new(),
                    vec![input],
                    None,
                )
                .unwrap();
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![input], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let false_branch = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let input = builder.add_input(scalar_type.clone());
            let output =
                builder.add_instruction(PrintOperation::new("false"), Vec::new(), vec![input], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let input = builder.add_input(scalar_type.clone());
        let output = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &vec![ArrayType::scalar(DataType::Boolean), scalar_type.clone()],
            &vec![scalar_type],
            "main",
            None,
            None,
        )
        .unwrap();

        assert_eq!(stablehlo.matches("stablehlo.after_all").count(), 2, "{stablehlo}");
        assert_eq!(stablehlo.matches("@ryft.assert").count(), 1, "{stablehlo}");
        assert_eq!(stablehlo.matches("@ryft.print").count(), 1, "{stablehlo}");
        let if_line = stablehlo.lines().find(|line| line.contains("\"stablehlo.if\"")).unwrap();
        assert!(if_line.contains(":3 ="), "{stablehlo}");
        assert!(stablehlo.contains("-> (tensor<i64>, !stablehlo.token, !stablehlo.token)"), "{stablehlo}",);
    }

    #[test]
    fn test_to_mlir_module_for_program_separates_assertion_and_io_token_chains() {
        use ryft_core::{DimensionFromScalarOperation, DimensionRequirementOperation, PrintOperation};

        let scalar_type = ArrayType::scalar(DataType::I64);
        let left_variable = DimensionVariable::new("left", DimensionBounds::new(1, Some(9)).unwrap());
        let right_variable = DimensionVariable::new("right", DimensionBounds::new(1, Some(9)).unwrap());
        let left_dimension_type = DimensionType::new(left_variable.clone());
        let right_dimension_type = DimensionType::new(right_variable.clone());

        let mut builder = CompositeXlaProgramBuilder::new();
        let left = builder.add_input(scalar_type.clone());
        let right = builder.add_input(scalar_type.clone());
        let left_dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(left_variable), Vec::new(), vec![left], None)
            .unwrap()[0];
        let right_dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(right_variable), Vec::new(), vec![right], None)
            .unwrap()[0];
        builder
            .add_instruction(
                DimensionRequirementOperation::less_than_or_equal(&left_dimension_type, &right_dimension_type),
                Vec::new(),
                vec![left_dimension, right_dimension],
                None,
            )
            .unwrap();
        builder.add_instruction(PrintOperation::new("value"), Vec::new(), vec![left], None).unwrap();
        builder
            .add_instruction(
                DimensionRequirementOperation::equal(&left_dimension_type, &right_dimension_type),
                Vec::new(),
                vec![left_dimension, right_dimension],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![left], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &vec![scalar_type.clone(), scalar_type.clone()],
            &vec![scalar_type],
            "main",
            None,
            None,
        )
        .unwrap();

        // The two gateway bounds checks and two explicit requirements share the assertion chain. The print receives
        // a separately created token, so it has no artificial dependency on either neighboring assertion.
        assert_eq!(module.matches("stablehlo.after_all").count(), 2, "{module}");
        assert_eq!(module.matches("@ryft.assert").count(), 4, "{module}");
        assert_eq!(module.matches("@ryft.print").count(), 1, "{module}");
        let print = module.lines().find(|line| line.contains("@ryft.print")).unwrap();
        let first_assertion = module.lines().find(|line| line.contains("@ryft.assert")).unwrap();
        let print_token =
            print.split_once("@ryft.print(").unwrap().1.split_once(')').unwrap().0.rsplit_once(", ").unwrap().1;
        let assertion_token = first_assertion
            .split_once("@ryft.assert(")
            .unwrap()
            .1
            .split_once(')')
            .unwrap()
            .0
            .rsplit_once(", ")
            .unwrap()
            .1;
        assert_ne!(print_token, assertion_token, "{module}",);
    }

    #[test]
    fn test_to_mlir_module_for_program_clamps_unproven_dimension_arithmetic_data_paths() {
        use ryft_core::{
            DimensionFromScalarOperation, DimensionOperation, DimensionSubOperation, DimensionToScalarOperation,
        };

        let scalar_type = ArrayType::scalar(DataType::I64);
        let build = |left_bounds: DimensionBounds, right_bounds: DimensionBounds| {
            let left_variable = DimensionVariable::new("left", left_bounds);
            let right_variable = DimensionVariable::new("right", right_bounds);
            let left_type = DimensionType::new(left_variable.clone());
            let right_type = DimensionType::new(right_variable.clone());
            let mut builder = CompositeXlaProgramBuilder::new();
            let left = builder.add_input(scalar_type.clone());
            let right = builder.add_input(scalar_type.clone());
            let left = builder
                .add_instruction(DimensionFromScalarOperation::new(left_variable), Vec::new(), vec![left], None)
                .unwrap()[0];
            let right = builder
                .add_instruction(DimensionFromScalarOperation::new(right_variable), Vec::new(), vec![right], None)
                .unwrap()[0];
            let difference = builder
                .add_instruction(
                    DimensionOperation::Sub(DimensionSubOperation::new(&left_type, &right_type).unwrap()),
                    Vec::new(),
                    vec![left, right],
                    None,
                )
                .unwrap()[0];
            let output =
                builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![difference], None).unwrap()[0];
            let program = builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![output],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap();
            to_mlir_module_for_program(
                &program,
                &[],
                &vec![scalar_type.clone(), scalar_type.clone()],
                &vec![scalar_type.clone()],
                "main",
                None,
                None,
            )
            .unwrap()
        };

        // An unproven subtraction keeps its diagnostic assertion (two gateway checks plus one arithmetic check), and
        // because the assertion custom call has no StableHLO data dependency on the subtraction, the data path is
        // clamped to zero so downstream extent consumers cannot fail inside XLA before the host callback reports the
        // original operands.
        let unproven = build(DimensionBounds::new(1, Some(9)).unwrap(), DimensionBounds::new(1, Some(9)).unwrap());
        assert_eq!(unproven.matches("stablehlo.subtract").count(), 1, "{unproven}");
        assert_eq!(unproven.matches("stablehlo.maximum").count(), 1, "{unproven}");
        assert_eq!(unproven.matches("@ryft.assert").count(), 3, "{unproven}");

        // A bounds-proven subtraction lowers without an arithmetic assertion and without a clamp.
        let proven = build(DimensionBounds::new(5, Some(9)).unwrap(), DimensionBounds::new(1, Some(3)).unwrap());
        assert_eq!(proven.matches("stablehlo.subtract").count(), 1, "{proven}");
        assert_eq!(proven.matches("stablehlo.maximum").count(), 0, "{proven}");
        assert_eq!(proven.matches("@ryft.assert").count(), 2, "{proven}");
    }

    #[test]
    fn test_repeated_effectful_jit_call_callees_inline_and_chain_prints() {
        use ryft_core::PrintOperation;

        // A repeated `jit_call` callee that prints is excluded from function deduplication and inlines at every
        // call site, so both inlined prints chain onto the caller's ordered-I/O token in program order (a shared
        // token-free `func.func` could not preserve that ordering).
        let array_type = test_vector_type(4);
        let mut callee_builder = XlaProgramBuilder::new();
        let callee_input = callee_builder.add_input(array_type.clone());
        let printed = callee_builder
            .add_instruction(PrintOperation::new("callee"), Vec::new(), vec![callee_input], None)
            .unwrap()[0];
        let callee_output = callee_builder
            .add_instruction(AddOperation::new(), Vec::new(), vec![printed, printed], None)
            .unwrap()[0];
        let callee = Arc::new(unproject_plain_program(
            callee_builder.build(vec![callee_output], vec![Placeholder], vec![Placeholder]).unwrap(),
        ));
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
        use ryft_core::{Device, DeviceMesh, Print};
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
            with_captured_prints(|| engine.interpret(&compiled.executable_function(), source).unwrap());

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

        use ryft_core::{PrintOperation, ScanOperation as CoreScanOperation};
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
        let mut body_builder = CompositeXlaProgramBuilder::new();
        let carry = body_builder.add_input(scalar_f64.clone());
        let x = body_builder.add_input(scalar_f64.clone());
        let printed =
            body_builder.add_instruction(PrintOperation::new("iteration"), Vec::new(), vec![x], None).unwrap()[0];
        let sum = body_builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, printed], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        let scan = CoreScanOperation::<XlaConstant>::new(1, 3);

        let stacked_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = CompositeXlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let init = builder.add_input(scalar_f64.clone());
        let stacked_inputs = builder.add_input(stacked_type.clone());
        let output = builder
            .add_instruction(XlaOperation::Scan(scan), vec![body_region], vec![init, stacked_inputs], None)
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

        use ryft_core::{CompareOperation, OneLikeOperation, ZeroLikeOperation};
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
            let mut builder = CompositeXlaProgramBuilder::new();
            let state = builder.add_input(state_type.clone());
            let zero = builder
                .add_instruction(ZeroLikeOperation::<ArrayType>::new(), Vec::new(), vec![state], None)
                .unwrap()[0];
            let predicate = builder
                .add_instruction(
                    XlaOperation::Array(ArrayOperation::Compare(CompareOperation::new(
                        ComparisonDirection::GreaterThan,
                    ))),
                    Vec::new(),
                    vec![state, zero],
                    None,
                )
                .unwrap()[0];
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![predicate], vec![Placeholder], vec![Placeholder])
        }
        .unwrap();
        let body = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let state = builder.add_input(state_type.clone());
            let one = builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
            let next = builder.add_instruction(SubOperation::new(), Vec::new(), vec![state, one], None).unwrap()[0];
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![next], vec![Placeholder], vec![Placeholder])
        }
        .unwrap();
        let mut builder = CompositeXlaProgramBuilder::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(body.entry_region_ref());
        let state = builder.add_input(state_type.clone());
        let output = builder
            .add_instruction(
                XlaOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                vec![state],
                None,
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
            ryft_core::programs::Program<CpuArray, ryft_core::ArrayOperation<CpuArray>, (CpuArray, CpuArray), CpuArray>,
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
            ryft_core::programs::Program<CpuArray, ryft_core::ArrayOperation<CpuArray>, CpuArray, CpuArray>,
        ) = TEST_ARRAY_DOMAIN
            .interpret_and_trace(
                |x| {
                    let context = x.context().clone();
                    Ok(context
                        .differentiate_at(x)
                        .gradient(scalar_quartic_plus_sin)
                        .expect("scalar gradient should succeed"))
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
                None,
            )
            .unwrap()[0];
        let updated = builder
            .add_instruction(UpdateSliceOperation::new(vec![0, 1]), Vec::new(), vec![input, update], None)
            .unwrap()[0];
        let dynamic_sliced = builder
            .add_instruction(DynamicSliceOperation::new(vec![1, 2]), Vec::new(), vec![input, index_0, index_1], None)
            .unwrap()[0];
        let dynamic_updated = builder
            .add_instruction(DynamicUpdateSliceOperation, Vec::new(), vec![input, update, index_0, index_1], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
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
            .add_instruction(DynamicSliceOperation::new(vec![1, 2]), Vec::new(), vec![input, index_0, index_1], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                vec![output],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
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
            .add_instruction(ConcatenateOperation::new(0, 2).unwrap(), Vec::new(), vec![first, second], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, XlaArrayConstant>(vec![joined], vec![Placeholder, Placeholder], Placeholder)
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

        // Dynamic concatenation requires its derived result extent as an explicit operand rather than manufacturing
        // an unstable result identity.
        let dynamic_type =
            ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("rows", None), Dimension::Static(2)]));
        let mut builder = XlaProgramBuilder::new();
        let first = builder.add_input(dynamic_type.clone());
        let second = builder.add_input(dynamic_type);
        assert_eq!(
            builder.add_instruction(ConcatenateOperation::new(0, 2).unwrap(), Vec::new(), vec![first, second], None),
            Err(ProgramError::Type(TypeError::invalid(
                "`concatenate` dynamic axis 0 requires an explicit result-dimension operand".to_string(),
            ))),
        );

        // A mixed concatenate whose static operand types prove its explicit result extent is pure and lowers without
        // assertion callback or dimension-size machinery.
        let first_type = test_matrix_type(1, 2);
        let second_type = test_matrix_type(3, 2);
        let result_extent = DimensionValue::constant(4).unwrap();
        let operation = ConcatenateOperation::<ArrayIrType>::from_input_types(
            0,
            &[first_type.clone().into(), second_type.clone().into(), result_extent.r#type().into_owned().into()],
        )
        .unwrap();
        let mut builder = CompositeXlaProgramBuilder::new();
        let first = builder.add_input(first_type.clone());
        let second = builder.add_input(second_type.clone());
        let result_extent = builder
            .add_instruction(
                DimensionOperation::from(ConstantOperation::new(result_extent)),
                Vec::new(),
                Vec::new(),
                None,
            )
            .unwrap()[0];
        let joined =
            builder.add_instruction(operation, Vec::new(), vec![first, second, result_extent], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![joined],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(program.effects(), Effects::PURE);
        let output_type = <&ArrayType>::try_from(&program.output_types()[0]).unwrap().clone();
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &vec![first_type, second_type],
            &vec![output_type],
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(stablehlo.matches("@ryft.assert").count(), 0, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.get_dimension_size").count(), 0, "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.concatenate").count(), 1, "{stablehlo}");

        // Once that extent is explicit, lowering reads the concrete logical input sizes, checks their sum on the
        // ordered-assertion chain, and passes only the arrays to StableHLO's concatenate operation.
        let first_variable =
            DimensionVariable::new("first", DimensionBounds::new(0, Some(5)).expect("valid test bounds"));
        let second_variable =
            DimensionVariable::new("second", DimensionBounds::new(0, Some(5)).expect("valid test bounds"));
        let first_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(first_variable), Dimension::Static(2)]));
        let second_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(second_variable), Dimension::Static(2)]));
        let first_size_operation = DimensionSizeOperation::new(&first_type, 0).unwrap();
        let second_size_operation = DimensionSizeOperation::new(&second_type, 0).unwrap();
        let add_operation =
            DimensionAddOperation::new(first_size_operation.result_type(), second_size_operation.result_type())
                .unwrap();
        let result_extent_type =
            DimensionType::new(DimensionVariable::new(add_operation.result_name(), add_operation.result_bounds()));
        let concatenate_operation = ConcatenateOperation::<ArrayIrType>::from_input_types(
            0,
            &[first_type.clone().into(), second_type.clone().into(), result_extent_type.into()],
        )
        .unwrap();
        let mut builder = CompositeXlaProgramBuilder::new();
        let first = builder.add_input(first_type.clone());
        let second = builder.add_input(second_type.clone());
        let first_size = builder.add_instruction(first_size_operation, Vec::new(), vec![first], None).unwrap()[0];
        let second_size = builder.add_instruction(second_size_operation, Vec::new(), vec![second], None).unwrap()[0];
        let result_extent = builder
            .add_instruction(DimensionOperation::Add(add_operation), Vec::new(), vec![first_size, second_size], None)
            .unwrap()[0];
        let joined = builder
            .add_instruction(concatenate_operation, Vec::new(), vec![first, second, result_extent], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![joined],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let output_type = <&ArrayType>::try_from(&program.output_types()[0]).unwrap().clone();
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &vec![first_type, second_type],
            &vec![output_type],
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(stablehlo.matches("stablehlo.get_dimension_size").count(), 5, "{stablehlo}");
        assert_eq!(stablehlo.matches("@ryft.assert").count(), 1, "{stablehlo}");
        assert!(stablehlo.contains("stablehlo.concatenate"), "{stablehlo}");

        // Dynamic non-concatenated dimensions need transform residuals rather than a hidden runtime-size lookup in a
        // nominally static slice operation.
        let columns = DimensionVariable::new("columns", DimensionBounds::unbounded());
        let left_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), columns.clone().into()]));
        let right_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), columns.into()]));
        let mut builder = XlaProgramBuilder::new();
        let left = builder.add_input(left_type);
        let right = builder.add_input(right_type);
        let joined = builder
            .add_instruction(ConcatenateOperation::new(0, 2).unwrap(), Vec::new(), vec![left, right], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, XlaArrayConstant>(vec![joined], vec![Placeholder, Placeholder], Placeholder)
            .unwrap();
        assert_eq!(
            program.transpose().unwrap_err().to_string(),
            "`concatenate` transpose requires a static size on axis 1 but operand 0 has size columns",
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
                None,
            )
            .unwrap()[0];
        let padded = builder
            .add_instruction(
                PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
                Vec::new(),
                vec![pad_input, padding_value],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
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
                None,
            )
            .unwrap()[0];
        let mixed = builder
            .add_instruction(
                PadOperation::new(vec![-1], vec![2], vec![2]).unwrap(),
                Vec::new(),
                vec![input, padding_value],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
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
                None
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "`pad` dynamic axis 0 requires an explicit result-dimension operand".to_string(),
            ))),
        );
    }

    #[test]
    fn test_slicing_vjp_pullbacks_lower_to_stablehlo() {
        use ryft_core::{DynamicSlice, Slice};

        // The static slice pullback writes the cotangent into a zero array at the static offsets via the
        // statically indexed update-slice, which lowers to `stablehlo.dynamic_update_slice` with constant indices.
        // The structural-zero destination is emitted as a `ZeroOperation` instruction in the pullback, which lowers
        // through the canonical zero path to a scalar constant broadcast to the array shape. The reverse path
        // stages the pullback over the primal operation family taking `[output_cotangents ++ residuals]`; this slice
        // pullback captures no residuals, so the pullback consumes only the single output cotangent.
        let (_, pullback): (CpuArray, _) = EagerContext::<CpuArray, ArrayOperation<CpuArray>>::new()
            .vjp(|x, ()| Ok(x.slice(&[1], &[3], &[1]).unwrap()), CpuArray::vector(vec![1.0, 2.0, 3.0, 4.0]), ())
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
            .vjp(
                |x, ()| Ok(x.slice(&[1], &[6], &[2]).unwrap()),
                CpuArray::vector(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
                (),
            )
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
                |(x, padding_value), ()| {
                    use ryft_core::Pad;
                    Ok(x.pad(&padding_value, &[1], &[2], &[1]).unwrap())
                },
                (CpuArray::vector(vec![1.0, 2.0, 3.0]), CpuArray::scalar(9.0)),
                (),
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
                |x, ()| {
                    let start =
                        x.context().lift(CpuArray::from_f64s(ArrayType::scalar(DataType::I32), vec![1.0])).unwrap();
                    Ok(x.dynamic_slice(&[start], &[2]).unwrap())
                },
                CpuArray::vector(vec![1.0, 2.0, 3.0, 4.0]),
                (),
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
            .vjp(|inputs, ()| Ok(scalar_bilinear_sin(inputs)), (CpuArray::scalar(2.0), CpuArray::scalar(3.0)), ())
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
            .vjp(|x, ()| function.call(x), CpuArray::scalar(2.0), ())
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
            .vjp(|x, ()| function.call(x), CpuArray::scalar(2.0), ())
            .unwrap();
        let (pullback, _residuals) = pullback.into_parts();
        let stablehlo = to_mlir_module_for_plain_program(&pullback, "main").unwrap();
        assert_eq!(stablehlo, expected, "the reverse pullback is independent of the prevent_cse hint");
    }

    #[test]
    fn test_transfer_to_memory_lowers_to_device_placement_annotations() {
        use ryft_core::TransferToMemory;

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
            let output =
                builder.add_instruction(ZeroOperation::new(output_type), Vec::new(), Vec::new(), None).unwrap()[0];
            let program = builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], Vec::new(), vec![Placeholder])
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
        let output = builder.add_instruction(ZeroOperation::new(output_type), Vec::new(), Vec::new(), None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], Vec::new(), vec![Placeholder])
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
        let output = builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
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
        assert_eq!(signature.physical_input_count(), 3);
        assert_eq!(signature.input_dimensions().len(), 1);
        assert_eq!(signature.input_dimensions()[0].logical_input_index(), 2);
        assert_eq!(signature.input_dimensions()[0].axis(), 0);
        assert_eq!(signature.input_dimensions()[0].physical_input_index(), 2);
        assert_eq!(signature.output_dimensions().len(), 1);
        assert_eq!(signature.output_dimensions()[0].logical_output_index(), 0);
        assert_eq!(signature.output_dimensions()[0].axis(), 0);
        assert_eq!(signature.output_dimensions()[0].physical_output_index(), 1);
    }

    #[test]
    fn test_to_mlir_module_for_program_erases_static_zero_space_boundary() {
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = CompositeXlaProgramBuilder::new();
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
        use ryft_core::PrintOperation;

        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(zero_type.clone());
        let output = builder.add_instruction(PrintOperation::new("zero"), Vec::new(), vec![input], None).unwrap()[0];
        let program: FlatXlaProgram = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        let stablehlo = to_mlir_module_for_program(&program, &[], &zero_type, &zero_type, "main", None, None).unwrap();

        assert!(stablehlo.contains("stablehlo.custom_call @ryft.print"), "{stablehlo}");
        assert!(stablehlo.contains("has_side_effect = true"), "{stablehlo}");
        assert!(stablehlo.contains("return\n"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_preserves_dynamic_zero_space_shape_carrier() {
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![dynamic_dimension("zero", Some(3))]));
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(zero_type.clone());
        let program: FlatXlaProgram = builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();

        let stablehlo = to_mlir_module_for_program(&program, &[], &zero_type, &zero_type, "main", None, None).unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xi1>, %arg1: tensor<i32>) -> (tensor<?xi1, #stablehlo.bounds<2>>, tensor<i64>) {
                    %0 = stablehlo.set_dimension_size %arg0, %arg1, dim = 0 : (tensor<2xi1>, tensor<i32>) -> tensor<?xi1, #stablehlo.bounds<2>>
                    %1 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xi1, #stablehlo.bounds<2>>) -> tensor<i32>
                    %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i64>
                    return %0, %2 : tensor<?xi1, #stablehlo.bounds<2>>, tensor<i64>
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
                .add_instruction(ArrayOperation::Constant(ConstantOperation::new(value)), Vec::new(), Vec::new(), None)
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
            let output = context.fill(&output_type, 2.5f64).unwrap();
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
            let output = context.fill(&output_type, 2.5f64).unwrap();
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
        let literal = CpuArray::from_elements(test_vector_type(2), &[-0.0_f32, 0.0]).unwrap();
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
        let literal = CpuArray::from_elements(test_vector_type(1), &[2.5_f32]).unwrap();
        let mut builder = ProgramBuilder::<CpuArray, ArrayOperation<CpuArray>>::new();
        let output = builder
            .add_instruction(ArrayOperation::Constant(ConstantOperation::new(literal)), Vec::new(), Vec::new(), None)
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
        let boolean = CpuArray::vector(vec![true, false, true]);
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = lower_tensor_type(boolean.r#type().as_ref(), &context, location).unwrap();
        let attribute = boolean.to_dense_elements_attribute(tensor_type, &context).unwrap();
        assert_eq!(
            unsafe { attribute.bool_elements().collect::<Result<Vec<_>, _>>().unwrap() },
            vec![true, false, true],
        );

        // One-bit integers use the same packed MLIR representation as Boolean while retaining their Ryft data type.
        for (literal, expected) in [
            (
                CpuArray::vector(vec![i1::new(-1).unwrap(), i1::new(0).unwrap(), i1::new(-1).unwrap()]),
                vec![true, false, true],
            ),
            (
                CpuArray::vector(vec![u1::new(1).unwrap(), u1::new(0).unwrap(), u1::new(1).unwrap()]),
                vec![true, false, true],
            ),
        ] {
            let tensor_type = lower_tensor_type(literal.r#type().as_ref(), &context, location).unwrap();
            let attribute = literal.to_dense_elements_attribute(tensor_type, &context).unwrap();
            assert_eq!(unsafe { attribute.bool_elements().collect::<Result<Vec<_>, _>>().unwrap() }, expected);
        }

        // Wider sub-byte integers occupy one MLIR raw-buffer byte per element and preserve only their declared bits.
        for (literal, expected) in [
            (CpuArray::vector(vec![i2::new(-2).unwrap(), i2::new(1).unwrap()]), vec![0x02, 0x01]),
            (CpuArray::vector(vec![i4::new(-8).unwrap(), i4::new(7).unwrap()]), vec![0x08, 0x07]),
            (CpuArray::vector(vec![u2::new(0).unwrap(), u2::new(3).unwrap()]), vec![0x00, 0x03]),
            (CpuArray::vector(vec![u4::new(1).unwrap(), u4::new(15).unwrap()]), vec![0x01, 0x0f]),
        ] {
            assert_eq!(test_literal_dense_bytes(&literal, expected.len()), expected);
        }

        // Every byte-aligned integer family preserves signedness, magnitude, and source order without floating-point
        // conversion. The `u64` case deliberately exceeds f64's exact-integer range.
        let integer_cases = vec![
            (CpuArray::vector(vec![-127_i8, 126]), values_to_bytes(&[-127_i8, 126])),
            (CpuArray::vector(vec![-0x1234_i16, 0x2345]), values_to_bytes(&[-0x1234_i16, 0x2345])),
            (CpuArray::vector(vec![-0x1234_567_i32, 0x2345_678]), values_to_bytes(&[-0x1234_567_i32, 0x2345_678])),
            (
                CpuArray::vector(vec![-0x1234_5678_9abc_def_i64, 0x2345_6789_abcd_ef0]),
                values_to_bytes(&[-0x1234_5678_9abc_def_i64, 0x2345_6789_abcd_ef0]),
            ),
            (CpuArray::vector(vec![0x12_u8, 0xfe]), values_to_bytes(&[0x12_u8, 0xfe])),
            (CpuArray::vector(vec![0x1234_u16, 0xfedc]), values_to_bytes(&[0x1234_u16, 0xfedc])),
            (CpuArray::vector(vec![0x1234_5678_u32, 0xfedc_ba98]), values_to_bytes(&[0x1234_5678_u32, 0xfedc_ba98])),
            (
                CpuArray::vector(vec![(1_u64 << 53) + 1, u64::MAX - 1]),
                values_to_bytes(&[(1_u64 << 53) + 1, u64::MAX - 1]),
            ),
        ];
        for (literal, expected) in integer_cases {
            assert_eq!(
                test_literal_dense_bytes(&literal, expected.len()),
                expected,
                "integer literal type {}",
                literal.r#type().as_ref(),
            );
        }

        // Sub-byte and eight-bit floating-point formats retain their exact encodings, including NaN payloads.
        for (data_type, bits, rendered_values) in [
            (DataType::F4E2M1FN, [0x01, 0x0f], "dense<[5.000000e-01, -6.000000e+00]>"),
            (DataType::F6E2M3FN, [0x01, 0x3f], "dense<[1.250000e-01, -7.500000e+00]>"),
            (DataType::F6E3M2FN, [0x02, 0x3e], "dense<[1.250000e-01, -2.400000e+01]>"),
            (DataType::F8E3M4, [0x01, 0xff], "dense<[1.562500e-02, 0xFF]>"),
            (DataType::F8E4M3, [0x02, 0xfe], "dense<[3.906250e-03, 0xFE]>"),
            (DataType::F8E4M3FN, [0x03, 0x7f], "dense<[5.859380e-03, 0x7F]>"),
            (DataType::F8E4M3FNUZ, [0x04, 0x80], "dense<[3.906250e-03, 0x80]>"),
            (DataType::F8E4M3B11FNUZ, [0x05, 0x80], "dense<[6.103520e-04, 0x80]>"),
            (DataType::F8E5M2, [0x06, 0x7f], "dense<[9.155270e-05, 0x7F]>"),
            (DataType::F8E5M2FNUZ, [0x07, 0x80], "dense<[5.340580e-05, 0x80]>"),
            (DataType::F8E8M0FNU, [0x08, 0xff], "dense<[1.504630e-36, 0xFF]>"),
        ] {
            let literal = CpuArray::from_logical_bytes(
                ArrayType::new(data_type, Shape::new(vec![Dimension::Static(bits.len())])),
                &bits,
            )
            .unwrap();
            assert_eq!(test_literal_dense_bytes(&literal, bits.len()), bits, "low-precision literal type {data_type}");
            let mut builder = ProgramBuilder::<CpuArray, ArrayOperation<CpuArray>>::new();
            let output = builder
                .add_instruction(
                    ArrayOperation::Constant(ConstantOperation::new(literal)),
                    Vec::new(),
                    Vec::new(),
                    None,
                )
                .unwrap()[0];
            let program =
                builder.build::<Vec<CpuArray>, Vec<CpuArray>>(vec![output], Vec::new(), vec![Placeholder]).unwrap();
            let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
            assert!(stablehlo.contains(rendered_values), "low-precision literal {data_type}: {stablehlo}");
        }

        // Standard floating-point families preserve signed zero, infinities, and NaN payload bits.
        let bf16_values = [half::bf16::from_bits(0x8000), half::bf16::from_bits(0x7fc1)];
        let f16_values = [half::f16::from_bits(0x8000), half::f16::from_bits(0x7e01)];
        let f32_values = [f32::from_bits(0x8000_0000), f32::INFINITY, f32::from_bits(0x7fc0_1234)];
        let f64_values =
            [f64::from_bits(0x8000_0000_0000_0000), f64::NEG_INFINITY, f64::from_bits(0x7ff8_0000_0000_1234)];
        for (literal, expected) in [
            (CpuArray::vector(bf16_values.to_vec()), values_to_bytes(&bf16_values)),
            (CpuArray::vector(f16_values.to_vec()), values_to_bytes(&f16_values)),
            (CpuArray::vector(f32_values.to_vec()), values_to_bytes(&f32_values)),
            (CpuArray::vector(f64_values.to_vec()), values_to_bytes(&f64_values)),
        ] {
            assert_eq!(
                test_literal_dense_bytes(&literal, expected.len()),
                expected,
                "floating-point literal type {}",
                literal.r#type().as_ref(),
            );
        }

        // Complex storage interleaves independently exact real and imaginary components in source order.
        let c64_components =
            [f32::from_bits(0x8000_0000), f32::from_bits(0x7fc0_1234), f32::INFINITY, f32::NEG_INFINITY];
        let c64 = CpuArray::vector(vec![
            ComplexNumber::new(c64_components[0], c64_components[1]),
            ComplexNumber::new(c64_components[2], c64_components[3]),
        ]);
        assert_eq!(test_literal_dense_bytes(&c64, size_of_val(&c64_components)), values_to_bytes(&c64_components),);
        let c128_components = [
            f64::from_bits(0x8000_0000_0000_0000),
            f64::from_bits(0x7ff8_0000_0000_1234),
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        let c128 = CpuArray::vector(vec![
            ComplexNumber::new(c128_components[0], c128_components[1]),
            ComplexNumber::new(c128_components[2], c128_components[3]),
        ]);
        assert_eq!(test_literal_dense_bytes(&c128, size_of_val(&c128_components)), values_to_bytes(&c128_components),);

        // Explicit physical layouts are traversed in logical row-major order before constructing the literal.
        let layout_type = ArrayType::new(DataType::I16, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]))
            .with_layout(Some(ryft_core::arrays::StridedLayout::new(vec![2, 4]).into()));
        let layout_literal = CpuArray::from_elements(layout_type, &[1_i16, 2, 3, 4]).unwrap();
        assert_eq!(layout_literal.storage_bytes(), values_to_bytes(&[1_i16, 3, 2, 4]));
        assert_eq!(test_literal_dense_bytes(&layout_literal, 8), values_to_bytes(&[1_i16, 2, 3, 4]));

        // Empty tensors remain valid dense constants and carry no raw payload bytes.
        let empty = CpuArray::vector(Vec::<i32>::new());
        let tensor_type = lower_tensor_type(empty.r#type().as_ref(), &context, location).unwrap();
        let attribute = empty.to_dense_elements_attribute(tensor_type, &context).unwrap();
        assert_eq!(attribute.elements_count(), 0);

        // Payload-free logical types never enter raw construction and report the standard structured lowering error.
        let tensor_type = context
            .tensor_type(context.signless_integer_type(1), &[MlirSize::Static(2)], None, location)
            .unwrap();
        for data_type in [DataType::Zero, DataType::Token] {
            let literal =
                CpuArray::from_logical_bytes(ArrayType::new(data_type, Shape::new(vec![Dimension::Static(2)])), &[])
                    .unwrap();
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

        let bf16_values = [half::bf16::from_bits(0x8000), half::bf16::from_bits(0x7fc1)];
        let f16_values = [half::f16::from_bits(0x8000), half::f16::from_bits(0x7e01)];
        let f32_values = [f32::from_bits(0x8000_0000), f32::from_bits(0x7fc0_1234)];
        let f64_values = [f64::from_bits(0x8000_0000_0000_0000), f64::from_bits(0x7ff8_0000_0000_1234)];
        let c64_components = [1.5_f32, -2.0, f32::from_bits(0x8000_0000), f32::from_bits(0x7fc0_1234)];
        let c128_components =
            [1.5_f64, -2.0, f64::from_bits(0x8000_0000_0000_0000), f64::from_bits(0x7ff8_0000_0000_1234)];
        let cases = vec![
            (CpuArray::vector(vec![true, false, true]), vec![1_u8, 0, 1]),
            (CpuArray::vector(vec![-0x1234_i16, 0x2345]), values_to_bytes(&[-0x1234_i16, 0x2345])),
            (
                CpuArray::vector(vec![(1_u64 << 53) + 1, u64::MAX - 1]),
                values_to_bytes(&[(1_u64 << 53) + 1, u64::MAX - 1]),
            ),
            (CpuArray::vector(bf16_values.to_vec()), values_to_bytes(&bf16_values)),
            (CpuArray::vector(f16_values.to_vec()), values_to_bytes(&f16_values)),
            (CpuArray::vector(f32_values.to_vec()), values_to_bytes(&f32_values)),
            (CpuArray::vector(f64_values.to_vec()), values_to_bytes(&f64_values)),
            (
                CpuArray::vector(vec![
                    ComplexNumber::new(c64_components[0], c64_components[1]),
                    ComplexNumber::new(c64_components[2], c64_components[3]),
                ]),
                values_to_bytes(&c64_components),
            ),
            (
                CpuArray::vector(vec![
                    ComplexNumber::new(c128_components[0], c128_components[1]),
                    ComplexNumber::new(c128_components[2], c128_components[3]),
                ]),
                values_to_bytes(&c128_components),
            ),
            (CpuArray::vector(vec![i1::new(-1).unwrap(), i1::new(0).unwrap()]), vec![0x01, 0x00]),
            (CpuArray::vector(vec![i2::new(-2).unwrap(), i2::new(1).unwrap()]), vec![0x02, 0x01]),
            (CpuArray::vector(vec![i4::new(-8).unwrap(), i4::new(7).unwrap()]), vec![0x08, 0x07]),
            (CpuArray::vector(vec![u1::new(0).unwrap(), u1::new(1).unwrap()]), vec![0x00, 0x01]),
            (CpuArray::vector(vec![u2::new(0).unwrap(), u2::new(3).unwrap()]), vec![0x00, 0x03]),
            (CpuArray::vector(vec![u4::new(1).unwrap(), u4::new(15).unwrap()]), vec![0x01, 0x0f]),
            (CpuArray::vector(Vec::<i32>::new()), Vec::new()),
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
                        None,
                    )
                    .unwrap()[0]
            })
            .collect::<Vec<_>>();
        let program = builder
            .build::<Vec<CpuArray>, Vec<CpuArray>>(outputs, Vec::new(), vec![Placeholder; cases.len()])
            .unwrap();
        let module = to_mlir_module_for_plain_program(&program, "main").unwrap();
        for literal in [
            "dense<[true, false, true]> : tensor<3xi1>",
            "dense<[9007199254740993, 18446744073709551614]> : tensor<2xui64>",
            "dense<[-0.000000e+00, 0x7FC1]> : tensor<2xbf16>",
            "dense<[-0.000000e+00, 0x7E01]> : tensor<2xf16>",
            "dense<[-0.000000e+00, 0x7FC01234]> : tensor<2xf32>",
            "dense<[-0.000000e+00, 0x7FF8000000001234]> : tensor<2xf64>",
            "dense<[(1.500000e+00,-2.000000e+00), (-0.000000e+00,0x7FC01234)]> : tensor<2xcomplex<f32>>",
            "dense<[(1.500000e+00,-2.000000e+00), (-0.000000e+00,0x7FF8000000001234)]> : tensor<2xcomplex<f64>>",
            "dense<[true, false]> : tensor<2xi1>",
            "dense<[-2, 1]> : tensor<2xi2>",
            "dense<[-8, 7]> : tensor<2xi4>",
            "dense<[false, true]> : tensor<2xi1>",
            "dense<[0, 3]> : tensor<2xui2>",
            "dense<[1, 15]> : tensor<2xui4>",
            "dense<> : tensor<0xi32>",
        ] {
            assert!(module.contains(literal), "{module}");
        }

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
        use ryft_core::TransferToMemory;

        // The pullback of a transfer moves the cotangent back to the operand's source memory (the default device
        // space here), so it lowers to an `annotate_device_placement` custom call targeting `device`.
        let (_, pullback): (CpuArray, _) = EagerContext::<CpuArray, ArrayOperation<CpuArray>>::new()
            .vjp(|x, ()| Ok(x.transfer_to_memory(Memory::Host { pinned: true })), CpuArray::scalar(2.0), ())
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
                ryft_core::ArrayOperation<CpuArray>,
                (CpuArray, CpuArray),
                (CpuArray, CpuArray),
            >,
        ) = TEST_ARRAY_DOMAIN
            .interpret_and_trace(
                |inputs| {
                    let context = inputs.0.context().clone();
                    Ok(context
                        .differentiate_at(inputs)
                        .gradient(scalar_bilinear_sin)
                        .expect("scalar gradient should succeed"))
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
