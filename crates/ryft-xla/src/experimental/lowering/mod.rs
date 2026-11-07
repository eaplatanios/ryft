use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use ryft_core::{
    Array as CoreArray, ArrayIrType, ArrayOperation, ArrayType, BroadcastOperation, CastOperation, CollectiveAxes,
    ConstrainShardingOperation, DataType, DimensionOperation, DimensionType, DimensionValue, DynamicBroadcastOperation,
    DynamicReshapeOperation, EffectClass, EffectClasses, ExternalReferenceBinding, GatherOperation, Instruction,
    LogicalMesh, Memory, OneLikeOperation, Operation as CoreOperation, OperationPayloadProjection, Parameterized,
    ProgramError, Provenance, RematerializationOptimizationBarrier, ReshapeOperation, ReshardOperation,
    ScatterOperation, Shape, ShardMapBoundary, ShardMapOperation, Sharding, ShardingDimension, Type as RyftType,
    TypeError, Typed, TypedProgram, Value as CoreValue, ZeroLikeOperation,
};
use ryft_mlir::dialects::{func, shardy, stable_hlo};
use ryft_mlir::{
    Attribute, AttributeRef, Block, BlockRef, Context as MlirContext, DenseElementsAttributeRef, Location, LocationRef,
    Module, Operation as MlirOperation, OperationPrintingFlags, Region, Type, TypeAndAttributes, Value as MlirValue,
    ValueRef,
};

use crate::CompilationError;
use crate::domains::runtimes::builtin_requirement;
use crate::domains::{RuntimeRequirement, normalize_runtime_requirements};
use crate::experimental::lowering::compositions::{
    NamedCompositionFunctionMap, collect_named_composition_functions, emit_named_composition_function,
};
use crate::experimental::lowering::regions::{lower_nested_program_inline, lower_program_outputs_with_inputs};
use crate::experimental::lowering::signature::ExecutableSignature;
use crate::experimental::lowering::types::{
    lower_boundary_layout, lower_captured_constant, lower_literal_value, lower_runtime_dimension_size_i64,
    lower_tensor_type, lower_unplaced_constant_output,
};
use crate::experimental::ops::{ArrayConstant, Constant, FlatProgram, Operation, Program, SessionConstant};
use crate::operations::collectives::collective_replica_groups;
use crate::operations::jit_call::{JitCallFunctionMap, collect_jit_call_functions, emit_jit_call_function};
use crate::operations::manipulation::broadcasting::broadcast_changes_explicit_sharding;
use crate::operations::manipulation::conversions::cast_changes_explicit_placement;
use crate::operations::manipulation::memory::memory_placement_kind;
use crate::sharding::{SHARDY_MESH_SYMBOL_NAME, shardy_tensor_sharding_attribute};
use crate::{Error, RaggedDotLoweringStrategy, ToMlir};

#[cfg(test)]
use crate::experimental::lowering::regions::lower_plain_program_outputs;
#[cfg(test)]
use ryft_core::Program as CoreProgram;

/// XLA-owned failures while admitting a program, constructing its StableHLO representation, or verifying it.
///
/// Public lowering entry points return [`Error`], which wraps these failures in [`Error::Lowering`]. Core program
/// errors and native MLIR errors retain their own carriers instead of being wrapped by this enum.
#[derive(Clone, Debug, thiserror::Error)]
pub enum LoweringError {
    /// The caller selected an empty function symbol or one containing whitespace.
    #[error("invalid function name `{function_name}` used during XLA lowering")]
    InvalidFunctionName { function_name: String },

    /// A captured constant has no binding in the capture namespace available to this lowering.
    #[error("missing captured constant #{index} during XLA lowering")]
    MissingCapturedConstant { index: usize },

    /// A session constant appears where no hidden argument of the entry function is reachable (e.g., inside a
    /// `shard_map` body or a shared private function).
    #[error("session constant `{constant}` cannot be referenced in this lowering scope")]
    UnreachableSessionConstant { constant: String },

    /// References reached a lowering boundary that requires their discharge.
    #[error("unresolved reference construct `{construct}` must be discharged before XLA lowering")]
    UnresolvedReference { construct: String },

    /// Mutable state reached a lowering boundary that requires its discharge.
    #[error("unresolved state in `{construct}` must be discharged before XLA lowering")]
    UnresolvedState { construct: String },

    /// The logical reference-state metadata cannot produce a valid physical executable alias.
    #[error("invalid XLA reference-state ABI: {message}")]
    InvalidReferenceStateAbi { message: String },

    /// A shared mesh axis was declared with different sizes by programs lowered into one module.
    #[error("mesh axis `{axis_name}` has size {actual}, but an earlier declaration requires {expected}")]
    ConflictingMeshAxisSize { axis_name: String, expected: usize, actual: usize },

    /// A valid operation requests behavior this backend cannot lower.
    #[error("cannot lower `{operation}`: {reason}")]
    UnsupportedOperation { operation: String, reason: String },

    /// A valid element type cannot be represented in the selected lowering context.
    #[error("cannot lower element type `{data_type}` for {context}")]
    UnsupportedElementType { data_type: DataType, context: String },

    /// The constructed MLIR module failed verification without returning a native error value.
    #[error("constructed MLIR module failed verification during XLA lowering")]
    VerificationFailure,
}

/// Named StableHLO compositions shared by composite operation lowerings (e.g., scaled dot).
pub(crate) mod compositions;
/// Lowering of nested programs and regions inline into an enclosing block.
pub(crate) mod regions;
/// Logical and physical signatures of lowered XLA executables.
pub(crate) mod signature;
/// Shared StableHLO type, constant, and dynamic-dimension lowering helpers.
pub(crate) mod types;

/// Textual StableHLO module paired with the exact executable signature used to emit its entry function.
pub(crate) struct LoweredModule {
    /// Textual StableHLO/Shardy module.
    pub(crate) stable_hlo: String,

    /// Logical-to-physical entry-function signature.
    pub(crate) signature: ExecutableSignature,

    /// Whether execution requires the host runtime assertion handler.
    pub(crate) requires_assertion_handler: bool,

    /// Canonical runtime contracts required by emitted calls and synthesized checks.
    pub(crate) runtime_requirements: Arc<[RuntimeRequirement]>,
}

impl LoweredModule {
    /// Consumes the textual module, executable signature, assertion classification, and runtime requirements.
    #[inline]
    pub(crate) fn into_parts(self) -> (String, ExecutableSignature, bool, Arc<[RuntimeRequirement]>) {
        (self.stable_hlo, self.signature, self.requires_assertion_handler, self.runtime_requirements)
    }
}

/// Per-class StableHLO tokens owned by one lowering scope.
///
/// Ordered assertions and ordered I/O intentionally use independent chains. This preserves program order within
/// each observable class without introducing an ordering relationship between unrelated classes. Unordered I/O has
/// no token slot. Region input guards are the exception: input validation must precede every operation in the
/// invoked region, so their lowering joins the assertion chain into the ordered-I/O chain (refer to
/// `operations::assertions::lower_guard`).
#[derive(Copy, Clone, Default)]
pub(crate) struct EffectTokens<'b, 'c: 'b, 't: 'c> {
    /// Current ordered-assertion token, created lazily by the first runtime assertion in this scope.
    pub(crate) ordered_assertion: Option<ValueRef<'b, 'c, 't>>,

    /// Current ordered-I/O token, created lazily by the first ordered I/O operation in this scope.
    pub(crate) ordered_io: Option<ValueRef<'b, 'c, 't>>,
}

impl<'b, 'c: 'b, 't: 'c> EffectTokens<'b, 'c, 't> {
    /// Returns the current token for `effect_class`. Both ordered I/O classes share the ordered-I/O slot: they form
    /// one ordering domain on any given device and differ only in whether that order also spans devices.
    pub(crate) fn get(&self, effect_class: EffectClass) -> Option<ValueRef<'b, 'c, 't>> {
        match effect_class {
            EffectClass::OrderedState => unreachable!("ordered state effects must be discharged before XLA lowering"),
            EffectClass::OrderedAssertion => self.ordered_assertion,
            EffectClass::OrderedIo | EffectClass::DeviceOrderedIo => self.ordered_io,
            EffectClass::UnorderedIo => None,
        }
    }

    /// Replaces the current token for one ordered effect class.
    pub(crate) fn set(&mut self, effect_class: EffectClass, token: ValueRef<'b, 'c, 't>) {
        match effect_class {
            EffectClass::OrderedState => unreachable!("ordered state effects must be discharged before XLA lowering"),
            EffectClass::OrderedAssertion => self.ordered_assertion = Some(token),
            EffectClass::OrderedIo | EffectClass::DeviceOrderedIo => self.ordered_io = Some(token),
            EffectClass::UnorderedIo => panic!("unordered effects do not have token slots"),
        }
    }
}

/// Returns `true` if any operation in `program`'s complete region arena carries unresolved ordered state.
///
/// This scans every region rather than consulting the program's effect summary because that summary intentionally
/// excludes dormant rule regions. Ordinary XLA cannot preserve state in either computation or rule regions, so both
/// must be rejected before lowering.
pub(crate) fn contains_unresolved_state<ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    program: &Program<ProgramInput, ProgramOutput, O>,
) -> bool
where
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
{
    program.regions().iter().any(|region| {
        region
            .instructions()
            .iter()
            .any(|instruction| instruction.operation().effects().classes().contains(EffectClass::OrderedState))
    })
}

/// Returns `true` if `program` contains a reference-typed atom or a [`CoreOperation`](ryft_core::Operation) whose
/// [effects](ryft_core::Operation::effects) declare a [`ReferenceEffect`](ryft_core::ReferenceEffect) in any region.
///
/// This check is independent from [`contains_unresolved_state`]: a pure reference pass-through or forwarded
/// reference capture can carry reference values without executing a stateful instruction. Both predicates back
/// the pre-compilation state checks and the two direct module-lowering entries; eager binding, dispatch, and staging
/// enforce their corresponding boundary invariants separately.
pub(crate) fn contains_unresolved_references<ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    program: &Program<ProgramInput, ProgramOutput, O>,
) -> bool
where
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
{
    // The atom scan dominates for every well-formed program (reference operations always touch a reference-typed
    // atom); the declaration scan is the independent-verifier arm that still catches a core discharge bug which
    // retyped the boundary atoms but left a reference operation behind.
    program.entry_region_ref().contains_atom_type_in_closure(RyftType::is_reference)
        || program.regions().iter().any(|region| {
            region
                .instructions()
                .iter()
                .any(|instruction| instruction.operation().effects().has_reference_declarations())
        })
}

/// Returns the effect classes that have StableHLO token slots, in canonical token/result order.
///
/// Token slots are an XLA/StableHLO representation decision, so the classification lives here rather than on the core
/// [`EffectClass`] type — but the match is deliberately exhaustive so that adding a class forces an explicit token-slot
/// decision in this backend instead of a silent omission. [`EffectClass::OrderedState`] has no slot: ordinary XLA
/// lowering rejects unresolved state at its module entry boundaries, and no defensive path may accidentally turn state
/// into an ordinary token-threaded effect.
pub(crate) fn token_threaded_effects(effects: EffectClasses) -> impl Iterator<Item = EffectClass> {
    // Each yielded class names one token slot of `EffectTokens`. Both ordered I/O classes share the ordered-I/O slot,
    // so a program mixing them still threads exactly one I/O token, keyed by `OrderedIo`. A scope that threads the
    // assertion token also threads the I/O slot, because an entry guard inside it joins its assertion token into the
    // I/O chain (refer to `lower_guard`), and that dependency must reach the I/O that follows the scope.
    let ordered_io = effects.contains(EffectClass::OrderedIo)
        || effects.contains(EffectClass::DeviceOrderedIo)
        || effects.contains(EffectClass::OrderedAssertion);
    effects
        .contains(EffectClass::OrderedAssertion)
        .then_some(EffectClass::OrderedAssertion)
        .into_iter()
        .chain(ordered_io.then_some(EffectClass::OrderedIo))
}

/// Per-instruction StableHLO lowering state passed to [`LowerableOperation::lower`] while lowering a program of the
/// operation family `O`.
///
/// A lowerer exposes the block that receives the emitted operations, together with the MLIR context and location that
/// those operations use, the attached regions of the instruction being lowered, and the ordered effect chains of the
/// enclosing lowering scope. The lowering entry points construct one lowerer per instruction: they copy the
/// scope-level effect tokens in and read the updated tokens back out after the instruction lowers, so effectful
/// operations thread their tokens through [`Self::effect_token`] and [`Self::set_effect_token`] without knowing the
/// enclosing scope. The lifetimes are those of the MLIR block (`'b`), context (`'c`), and thread pool (`'t`).
pub struct Lowerer<'b, 'c: 'b, 't: 'c, O = Operation> {
    /// Owning block receiving the lowered operations.
    pub(crate) block: BlockRef<'b, 'c, 't>,

    /// MLIR context owning the block and created operations.
    pub(crate) context: &'c MlirContext<'t>,

    /// Shared MLIR location used for emitted operations.
    pub(crate) location: LocationRef<'c, 't>,

    /// Attached regions of the instruction currently being lowered, materialized as flat programs in region-slot order.
    pub(crate) regions: Vec<FlatProgram<O>>,

    /// Shared private functions emitted for deduplicated `jit_call` callees, consulted at `jit_call` lowering sites.
    /// Shared via [`Rc`] so it threads through nested lowering scopes without lifetime entanglement.
    pub(crate) nested_functions: Option<Rc<JitCallFunctionMap<O>>>,

    /// Hidden capture arguments of the function currently being lowered, in capture-table order.
    pub(crate) captured_values: Vec<ValueRef<'b, 'c, 't>>,

    /// Current per-class effect tokens of the lowering scope this lowerer emits into.
    pub(crate) effect_tokens: EffectTokens<'b, 'c, 't>,

    /// Collective lowering state of the lowering scope this lowerer emits into. Refer to the documentation of
    /// [`CollectiveLoweringState`] for more information.
    pub(crate) collective_state: CollectiveLoweringState,
}

impl<'b, 'c: 'b, 't: 'c, O> Lowerer<'b, 'c, 't, O> {
    /// Creates an MLIR lowerer for operations emitted into `block`.
    pub(crate) fn new(
        block: BlockRef<'b, 'c, 't>,
        context: &'c MlirContext<'t>,
        location: LocationRef<'c, 't>,
    ) -> Self {
        Self {
            block,
            context,
            location,
            regions: Vec::new(),
            nested_functions: None,
            captured_values: Vec::new(),
            effect_tokens: EffectTokens::default(),
            collective_state: CollectiveLoweringState::new(),
        }
    }

    /// Attaches the materialized attached regions of the instruction currently being lowered.
    pub(crate) fn with_regions(mut self, regions: Vec<FlatProgram<O>>) -> Self {
        self.regions = regions;
        self
    }

    /// Attaches the shared deduplicated `jit_call` functions consulted while lowering.
    pub(crate) fn with_nested_functions(mut self, nested_functions: Option<Rc<JitCallFunctionMap<O>>>) -> Self {
        self.nested_functions = nested_functions;
        self
    }

    /// Attaches the hidden capture arguments of the function currently being lowered.
    pub(crate) fn with_captured_values(mut self, captured_values: &[ValueRef<'b, 'c, 't>]) -> Self {
        self.captured_values = captured_values.to_vec();
        self
    }

    /// Attaches the current per-class effect tokens of the enclosing lowering scope.
    pub(crate) fn with_effect_tokens(mut self, effect_tokens: EffectTokens<'b, 'c, 't>) -> Self {
        self.effect_tokens = effect_tokens;
        self
    }

    /// Attaches the collective lowering state of the enclosing lowering scope.
    pub(crate) fn with_collective_state(mut self, collective_state: CollectiveLoweringState) -> Self {
        self.collective_state = collective_state;
        self
    }

    /// Declares a runtime contract required by this emitted operation.
    ///
    /// Requirements are shared by every nested lowering scope and retained with the compiled artifact. Runtime
    /// providers are resolved by the invoking session before compilation, loading, or execution.
    ///
    /// # Errors
    ///
    /// Returns an error when this declaration conflicts with an earlier requirement for the same identifier.
    pub fn require_runtime(&mut self, requirement: RuntimeRequirement) -> Result<(), Error> {
        let mut requirements = self.collective_state.runtime_requirements.borrow_mut();
        let mut updated = requirements.clone();
        updated.push(requirement);
        *requirements = normalize_runtime_requirements(updated)?;
        Ok(())
    }

    /// Returns the block that receives the StableHLO operations of the instruction being lowered. [`BlockRef`] is a
    /// copyable handle, so operations appended through the returned handle land in the block of this lowerer.
    pub fn block(&self) -> BlockRef<'b, 'c, 't> {
        self.block
    }

    /// Returns the MLIR context that owns the block of this lowerer and the operations emitted into it.
    pub fn context(&self) -> &'c MlirContext<'t> {
        self.context
    }

    /// Returns the location of the emitted operations, which carries the provenance of the instruction being lowered.
    pub fn location(&self) -> LocationRef<'c, 't> {
        self.location
    }

    /// Returns the attached regions of the instruction being lowered as flat programs, in region-slot order.
    pub fn regions(&self) -> &[FlatProgram<O>] {
        &self.regions
    }

    /// Returns the StableHLO tensor type that represents values of `array_type`.
    ///
    /// # Errors
    ///
    /// Returns [`LoweringError::UnsupportedElementType`] when an element type has no StableHLO tensor representation,
    /// and propagates native construction failures through [`Error::Mlir`].
    pub fn lower_tensor_type(&self, array_type: &ArrayType) -> Result<ryft_mlir::TensorTypeRef<'c, 't>, Error> {
        lower_tensor_type(array_type, self.context, self.location)
    }

    /// Returns the current token of the `effect_class` chain of the enclosing lowering scope, creating that chain with
    /// a zero-operand `stablehlo.after_all` operation the first time the scope uses it. An effectful operation consumes
    /// this token and records the token that it produces through [`Self::set_effect_token`], so that the effects of
    /// each ordered class stay in program order without ordering unrelated classes.
    ///
    /// # Errors
    ///
    /// Returns a [`Error`] when the `stablehlo.after_all` operation cannot be created.
    ///
    /// # Panics
    ///
    /// Panics for [`EffectClass::OrderedState`], which reference discharge removes before lowering, and for
    /// [`EffectClass::UnorderedIo`], which has no token chain.
    pub fn effect_token(&mut self, effect_class: EffectClass) -> Result<ValueRef<'b, 'c, 't>, Error> {
        assert_ne!(effect_class, EffectClass::UnorderedIo, "unordered effects do not have token chains");
        current_or_new_token(effect_class, &mut self.effect_tokens, &mut self.block, self.location)
    }

    /// Records `token` as the current token of the `effect_class` chain of the enclosing lowering scope.
    ///
    /// # Panics
    ///
    /// Panics for [`EffectClass::OrderedState`], which reference discharge removes before lowering, and for
    /// [`EffectClass::UnorderedIo`], which has no token chain.
    pub fn set_effect_token(&mut self, effect_class: EffectClass, token: ValueRef<'b, 'c, 't>) {
        self.effect_tokens.set(effect_class, token);
    }

    /// Lowers the attached region at `index` inline into the block of this lowerer, binding `input_values` to the
    /// region inputs and returning the lowered region outputs. The region shares the capture scope and the effect
    /// chains of the enclosing lowering scope, like the nested-traced regions of `while`, `scan`, `condition`, and
    /// `linear_call`.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the instruction being lowered has no region at `index`, and
    /// propagates the errors of lowering the region.
    pub fn lower_region_inline(
        &mut self,
        index: usize,
        input_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error>
    where
        O: LowerableOperationFamily,
    {
        let Some(region) = self.regions.get(index) else {
            return Err(ProgramError::MalformedProgram(format!(
                "region #{index} requested from an instruction with {} attached regions",
                self.regions.len()
            ))
            .into());
        };
        lower_nested_program_inline(
            region,
            input_values,
            &mut self.block,
            self.context,
            self.location,
            self.captured_values.as_slice(),
            &RematerializationOptimizationBarrier::None,
            self.nested_functions.as_ref(),
            &self.collective_state,
            &mut self.effect_tokens,
        )
    }
}

/// Operations that can be lowered to StableHLO for XLA compilation.
///
/// Each operation type implements this trait at the type universe of its [`CoreOperation`](ryft_core::Operation)
/// contract: homogeneous array operations at [`ArrayType`], array IR operations at [`ArrayIrType`], and first-class
/// dimension operations at [`DimensionType`]. Operation families (e.g., [`ArrayOperation`] and [`Operation`]) implement
/// this trait by dispatching each variant to its payload through
/// [`lower_operation_variants`](crate::lower_operation_variants), which also handles payloads that a family stores as
/// projected or mixed members. A family whose type universe is [`ArrayIrType`] and that implements
/// [`OperationPayloadProjection`] is a [`LowerableOperationFamily`] whose programs [`to_mlir_module_for_program`]
/// lowers, so downstream crates extend lowering by implementing this trait for their own operations and families, just
/// like they extend the transformation traits of `ryft-core`.
///
/// Operations with attached regions lower them through the lowerer (e.g., [`Lowerer::lower_region_inline`]) rather
/// than through bounds on their own family, so the bounds of a family implementation stay finite.
pub trait LowerableOperation: CoreOperation {
    /// Lowers this operation to one or more StableHLO operations emitted into the block of `lowerer`, returning one
    /// MLIR value per output.
    ///
    /// # Parameters
    ///
    ///   - `lowerer`: Lowering state of the instruction applying this operation.
    ///   - `input_values`: Lowered MLIR values of the instruction inputs, one per input.
    ///   - `input_types`: Declared types of the instruction inputs, in input order.
    ///   - `output_types`: Declared types of the instruction outputs, in output order.
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Self::Type],
        output_types: &[Self::Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error>;

    /// Returns the semantic identity of this operation if it is a call of a named composite (refer to
    /// [`NamedComposite`](ryft_core::NamedComposite)), and [`None`] otherwise. Module-level lowering passes use it to
    /// emit one shared function per specialization of a named composite body and to skip the bodies that a backend
    /// implementation replaces. Families forward it to the variants that can hold such calls, and the calls of a family
    /// that does not are lowered without sharing their bodies.
    #[inline]
    fn composite_key(&self) -> Option<&ryft_core::CompositeKey> {
        None
    }
}

/// Operation families whose programs the XLA lowering entry points lower. Such a family lowers at [`ArrayIrType`],
/// and its payloads can be recognized through [`OperationPayloadProjection`] by the module-level lowering passes
/// (e.g., the `jit_call` deduplication and the Shardy mesh declaration passes). Every type that satisfies these bounds
/// is an operation family, so downstream families only need to implement [`LowerableOperation`].
pub trait LowerableOperationFamily: LowerableOperation<Type = ArrayIrType> + OperationPayloadProjection {}

impl<O: LowerableOperation<Type = ArrayIrType> + OperationPayloadProjection> LowerableOperationFamily for O {}

/// Operations whose own type universe is a member universe of an enclosing operation family that lowers them at the
/// family's parent universe `T` (e.g., [`ZeroOperation<ArrayType>`](ryft_core::ZeroOperation) stored in an
/// [`ArrayIrType`] family receives first-class dimension inputs that size its dynamic output axes).
///
/// This is the lowering counterpart of [`MemberOperation`](ryft_core::MemberOperation), and families dispatch their
/// mixed member variants to it through [`lower_operation_variants`](crate::lower_operation_variants).
pub trait MemberLowerableOperation<T: RyftType>: CoreOperation {
    /// Lowers this operation, applied by an instruction of the parent universe `T`, to one or more StableHLO
    /// operations emitted into the block of `lowerer`, returning one MLIR value per output.
    ///
    /// # Parameters
    ///
    ///   - `lowerer`: Lowering state of the instruction applying this operation.
    ///   - `input_values`: Lowered MLIR values of the instruction inputs, one per input.
    ///   - `input_types`: Declared parent-universe types of the instruction inputs, in input order.
    ///   - `output_types`: Declared parent-universe types of the instruction outputs, in output order.
    fn lower_in_parent<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[T],
        output_types: &[T],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error>;
}

/// Lowers a payload that a family stores as a projected member of universe `U` by projecting the instruction's
/// parent-universe types to `U` before delegating to the payload's [`LowerableOperation`] implementation. This function
/// supports the expansion of [`lower_operation_variants`](crate::lower_operation_variants).
#[doc(hidden)]
pub fn lower_projected_operation<'b, 'c: 'b, 't: 'c, U, T, P, O: LowerableOperationFamily>(
    operation: &P,
    lowerer: &mut Lowerer<'b, 'c, 't, O>,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[T],
    output_types: &[T],
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error>
where
    U: RyftType,
    P: LowerableOperation<Type = U>,
    for<'a> &'a U: TryFrom<&'a T, Error: Into<ProgramError>>,
{
    let project = |types: &[T]| {
        types
            .iter()
            .map(|r#type| <&U>::try_from(r#type).cloned().map_err(Into::<ProgramError>::into))
            .collect::<Result<Vec<_>, ProgramError>>()
    };
    let input_types = project(input_types)?;
    let output_types = project(output_types)?;
    operation.lower(lowerer, input_values, input_types.as_slice(), output_types.as_slice())
}

/// Dispatches [`LowerableOperation::lower`] for an operation family to the payload of its active variant.
///
/// The macro takes the family value, the family's type name, the [`Lowerer`], the lowered input values, and the
/// input and output types, followed by every variant of the family with the way its payload lowers:
///
///   - A bare variant lowers its payload at the family's own type universe.
///   - `projected(U)` first projects the instruction types to the member universe `U` (e.g., an `Array` variant that
///     stores an [`ArrayOperation`] in an [`ArrayIrType`] family).
///   - `member` lowers a payload of a member universe at the family's universe through [`MemberLowerableOperation`].
///   - `boxed` lowers a boxed payload at the family's universe.
///
/// The generated `match` is exhaustive, so adding a variant to a family fails to compile until the family states how
/// that variant lowers.
///
/// # Example
///
/// A downstream family that extends [`Operation`] with its own payload:
///
/// ```ignore
/// impl LowerableOperation for MyOperation {
///     fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
///         &self,
///         lowerer: &mut Lowerer<'b, 'c, 't, O>,
///         input_values: &[ValueRef<'b, 'c, 't>],
///         input_types: &[ArrayIrType],
///         output_types: &[ArrayIrType],
///     ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
///         ryft_xla::lower_operation_variants!(
///             self, MyOperation, lowerer, input_values, input_types, output_types;
///             Xla, Custom,
///         )
///     }
/// }
/// ```
#[macro_export]
macro_rules! lower_operation_variants {
    // Public form: `lower_operation_variants!(operation, Family, lowerer, inputs, input_types, output_types; ...)`.
    (
        $operation:expr, $family:ident, $lowerer:expr, $inputs:expr, $input_types:expr, $output_types:expr;
        $($variant:ident $(=> $kind:ident $(($universe:ty))?)?),* $(,)?
    ) => {
        match $operation {
            $(
                $family::$variant(operation) => $crate::lower_operation_variants!(
                    @arm operation, $lowerer, $inputs, $input_types, $output_types $(, $kind $(($universe))?)?
                ),
            )*
        }
    };
    // Payload lowered at the family's universe.
    (@arm $payload:ident, $lowerer:expr, $inputs:expr, $input_types:expr, $output_types:expr) => {
        $crate::LowerableOperation::lower($payload, $lowerer, $inputs, $input_types, $output_types)
    };
    // Projected member payload lowered at its own universe.
    (@arm $payload:ident, $lowerer:expr, $inputs:expr, $input_types:expr, $output_types:expr, projected($universe:ty)) => {
        $crate::experimental::lowering::lower_projected_operation::<$universe, _, _, _>(
            $payload,
            $lowerer,
            $inputs,
            $input_types,
            $output_types,
        )
    };
    // Mixed member payload lowered at the family's universe.
    (@arm $payload:ident, $lowerer:expr, $inputs:expr, $input_types:expr, $output_types:expr, member) => {
        $crate::MemberLowerableOperation::lower_in_parent(
            $payload,
            $lowerer,
            $inputs,
            $input_types,
            $output_types,
        )
    };
    // Boxed payload lowered at the family's universe.
    (@arm $payload:ident, $lowerer:expr, $inputs:expr, $input_types:expr, $output_types:expr, boxed) => {
        $crate::LowerableOperation::lower(
            $payload.as_ref(),
            $lowerer,
            $inputs,
            $input_types,
            $output_types,
        )
    };
}

/// Returns the current StableHLO token for one ordered effect class, creating that class's chain lazily with a
/// zero-operand `stablehlo.after_all` when the lowering scope has not used it yet.
///
/// Each ordered class has an independent chain that preserves program order within that class without imposing an
/// artificial order across classes. The enclosing lowering helpers carry active chains through nested computations;
/// the final tokens are intentionally omitted from the public function results because their custom calls are marked
/// side-effecting and the tokens exist only to encode intra-execution ordering.
pub(crate) fn current_or_new_token<'b, 'c: 'b, 't: 'c>(
    effect_class: EffectClass,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    if let Some(token) = effect_tokens.get(effect_class) {
        return Ok(token);
    }
    let created = block.append_operation(stable_hlo::after_all::<ValueRef, _>(&[], location)?)?;
    let created = created.result(0).expect("stablehlo.after_all should return one result").as_ref();
    effect_tokens.set(effect_class, created);
    Ok(created)
}

impl<V: MlirLowerableValue> LowerableOperation for ArrayOperation<V> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        // This function does not validate the placement of local outputs. `Operation::lower` validates it for every
        // variant, including this projected one, whereas downstream families that embed `ArrayOperation` must reject
        // outputs that shard a local dimension over a bound manual mesh axis themselves (refer to
        // `validate_local_output_placement`).
        lower_operation_variants!(
            self, ArrayOperation, lowerer, input_values, input_types, output_types;
            Zero, ZeroLike, One, OneLike, Constant, Iota, Min, Max, Clamp, Sign, Neg, Add, Sub, SaturatingSub,
            Mul, Div, Rem, Pow, Sqrt, Rsqrt, Abs, Sin, Cos, Tan, Tanh, Atan2, Exp, ExpM1, Log, Ln1p, LogAddExp, Logistic, Erf,
            Floor, Ceil, Round, Not,
            And, Or, Xor, Complex, Real, Imaginary, Conjugate, Reduce, ArgMax, ArgMin, Cumulative, Sort, Dot, RaggedDot,
            ScaledDot, AxisIndex, ParallelReduce, ParallelVary, ParallelCast,
            ParallelAllGather, ParallelSumScatter, ParallelPermute, ParallelBroadcast, ParallelAllToAll,
            ParallelRaggedAllToAll, Compare, Select, Transpose, Reverse, Reshape, Broadcast, Pad, Concatenate, Gather,
            Scatter, Slice, UpdateSlice,
            DynamicSlice, DynamicUpdateSlice, ConvertElementType, ReducePrecision, TransferToMemory, Reshard,
            ConstrainSharding, Condition, While, Scan, StopGradient, Tag, Rematerialize, LinearCall, RngBitGenerator,
            Dropout, Print, Assert, Guard, Cast, CustomCall, Composite, CompositeTranspose,
        )
    }
}

impl LowerableOperation for DimensionOperation<DimensionValue> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[DimensionType],
        output_types: &[DimensionType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        lower_operation_variants!(
            self, DimensionOperation, lowerer, input_values, input_types, output_types;
            Constant, Max, Min, Add, Sub, SaturatingSub, Mul, Div, Rem, Pow,
        )
    }
}

impl LowerableOperation for Operation {
    #[inline]
    fn composite_key(&self) -> Option<&ryft_core::CompositeKey> {
        match self {
            Self::Composite(operation) => operation.key(),
            Self::LiftedComposite(operation) => operation.key(),
            _ => None,
        }
    }

    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        // Every variant, including the projected array operations, has its local output placement validated here
        // exactly once. A nested `shard_map` instead validates its boundary shardings against the enclosing
        // manual axes itself, reporting the more specific `ShardMapError::ShardingNamesEnclosingManualAxis`.
        if !matches!(self, Self::ShardMap(_)) {
            validate_local_output_placement(self.name(), output_types, &lowerer.collective_state.bound_manual_axes)?;
        }
        lower_operation_variants!(
            self, Operation, lowerer, input_values, input_types, output_types;
            Kernel,
            Zero => member,
            DynamicOne => member,
            DynamicIota => member,
            Array => projected(ArrayType),
            Dimension => projected(DimensionType),
            Compare,
            Assert,
            Guard,
            Cast,
            DimensionSize,
            ReferenceNew,
            ReferenceRead,
            ReferenceWrite,
            ReferenceSwap,
            ReferenceAddUpdate,
            ReferenceAtomicAddUpdate,
            ReferenceFreeze,
            DimensionFromScalar,
            DimensionToScalar,
            Reshape,
            Broadcast,
            Concatenate,
            CustomCall => member,
            Pad,
            DynamicSlice,
            RngBitGenerator,
            ParallelAllGather => member,
            ParallelSumScatter => member,
            ParallelAllToAll => member,
            ParallelRaggedAllToAll => member,
            Condition,
            While,
            Scan,
            Composite,
            CompositeTranspose,
            LinearCall,
            Rematerialize,
            LiftedComposite,
            LiftedCompositeTranspose,
            JitCall,
            ShardMap => boxed,
        )
    }
}

/// Rejects array `output_types` of the operation named `operation_name` that shard a local dimension over one of the
/// `bound_manual_axes` of the enclosing manual regions. Manual dimension ownership belongs to region boundaries: local
/// operations may change manual variation through collectives, but must not manufacture a global placement, which
/// lowered sharding constraints would otherwise silently strip (refer to `lower_sharding_constraint`). First-class
/// dimensions and references carry no placement of their own.
fn validate_local_output_placement(
    operation_name: &str,
    output_types: &[ArrayIrType],
    bound_manual_axes: &[String],
) -> Result<(), Error> {
    let assigns_bound_manual_axis = output_types.iter().any(|output_type| {
        let sharding = match output_type {
            ArrayIrType::Array(output_type) => output_type.sharding(),
            ArrayIrType::Dimension(_) | ArrayIrType::Reference(_) => None,
        };
        sharding.into_iter().flat_map(Sharding::dimensions).any(|dimension| match dimension {
            ShardingDimension::Sharded(axes) => axes.iter().any(|axis| bound_manual_axes.contains(axis)),
            ShardingDimension::Replicated | ShardingDimension::Unconstrained => false,
        })
    });
    if assigns_bound_manual_axis {
        return Err(ProgramError::UnsupportedOperation {
            message: format!("`{operation_name}` cannot assign bound manual mesh axes to local output dimensions"),
        }
        .into());
    }
    Ok(())
}

/// Module-scoped lowering state shared across nested regions. StableHLO collectives use the channel counter and the
/// innermost enclosing `sdy.manual_computation` region's [`ShardMapBoundary`], while other lowerings consume immutable
/// target and representation policy stored alongside that shared state.
///
/// The state is created once per lowered module and cloned per instruction lowerer. Module-owned mutable state and
/// registries are shared through [`Rc`], while immutable target and representation policies are copied or cloned.
/// [`enter_manual_region`](Self::enter_manual_region) preserves that module state and policy while replacing the
/// innermost region's [`ShardMapBoundary`].
#[derive(Clone)]
pub(crate) struct CollectiveLoweringState {
    /// Module-scoped counter producing the next collective channel id.
    pub(crate) channel_ids: Rc<Cell<usize>>,

    /// Enclosing manual regions, ordered from outermost to innermost for named-axis resolution.
    manual_regions: Vec<Rc<ShardMapBoundary>>,

    /// Axes bound by every enclosing manual region, excluded from shard-local placement constraints.
    pub(crate) bound_manual_axes: Vec<String>,

    /// PJRT platform name of the compilation target (e.g., `"cuda"` or `"cpu"`), or `None` when the lowering has
    /// no target information. Platform-gated lowerings such as fused attention consult this and
    /// fall back to their portable form when it is absent.
    target_platform: Option<Rc<str>>,

    /// Backend representation used to lower ragged dots in this module.
    ragged_dot_lowering_strategy: RaggedDotLoweringStrategy,

    /// Module-owned private decompositions used by named StableHLO composites.
    pub(crate) named_compositions: Option<Rc<NamedCompositionFunctionMap>>,

    /// Module-scoped flag recording whether any recursive lowering path constructed an instruction-specific
    /// (non-base) location from a non-unknown [`Provenance`]. It selects the module serialization mode after
    /// lowering completes: only a module that actually carries provenance locations is printed with debug
    /// information, so provenance-free modules keep their existing byte-identical StableHLO text and cache keys.
    has_provenance: Rc<Cell<bool>>,

    /// Whether any emitted instruction requires the runtime assertion handler.
    pub(crate) has_assertions: Rc<Cell<bool>>,

    /// Runtime contracts declared by emitted operations across every nested lowering scope.
    runtime_requirements: Rc<RefCell<Vec<RuntimeRequirement>>>,

    /// Distinct [`SessionConstant`]s of the module, in the order of the hidden entry arguments that carry them (refer
    /// to [`collect_session_constants`]). Their values form the suffix of every `captured_values` table that can reach
    /// the entry function's arguments. Scopes that cannot reach them (i.e., `sdy.manual_computation` bodies) carry no
    /// session constants.
    session_constants: Rc<[SessionConstant]>,
}

impl CollectiveLoweringState {
    /// Creates the lowering state for one module, outside any manual region and without target information.
    pub(crate) fn new() -> Self {
        Self {
            channel_ids: Rc::new(Cell::new(1)),
            manual_regions: Vec::new(),
            bound_manual_axes: Vec::new(),
            target_platform: None,
            ragged_dot_lowering_strategy: RaggedDotLoweringStrategy::default(),
            named_compositions: None,
            has_provenance: Rc::new(Cell::new(false)),
            has_assertions: Rc::new(Cell::new(false)),
            runtime_requirements: Rc::new(RefCell::new(Vec::new())),
            session_constants: Rc::from([]),
        }
    }

    /// Returns a copy of this state whose scope reaches the hidden entry arguments carrying `session_constants`.
    pub(crate) fn with_session_constants(mut self, session_constants: Rc<[SessionConstant]>) -> Self {
        self.session_constants = session_constants;
        self
    }

    /// Returns the suffix of `captured_values` that holds the values of the session constants reachable in this scope.
    /// A scope whose `captured_values` table cannot hold them (e.g., a shared private function) has no such values.
    pub(crate) fn session_values<'v, V>(&self, captured_values: &'v [V]) -> &'v [V] {
        captured_values
            .len()
            .checked_sub(self.session_constants.len())
            .map_or(&[], |start| &captured_values[start..])
    }

    /// Returns the value of `constant` in a scope whose captured values are `captured_values`.
    pub(crate) fn lower_session_constant<'b, 'c: 'b, 't: 'c>(
        &self,
        constant: &SessionConstant,
        captured_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<ValueRef<'b, 'c, 't>, Error> {
        let values = self.session_values(captured_values);
        self.session_constants
            .iter()
            .position(|candidate| candidate == constant)
            .filter(|_| values.len() == self.session_constants.len())
            .map(|index| values[index])
            .ok_or_else(|| {
                Error::Lowering(LoweringError::UnreachableSessionConstant { constant: constant.to_string() })
            })
    }

    /// Returns a copy of this state carrying the PJRT platform name of the compilation target.
    pub(crate) fn with_target_platform(mut self, target_platform: Option<&str>) -> Self {
        self.target_platform = target_platform.map(Rc::from);
        self
    }

    /// Returns a copy of this state carrying the grouped-dot lowering strategy.
    pub(crate) fn with_ragged_dot_lowering_strategy(mut self, strategy: RaggedDotLoweringStrategy) -> Self {
        self.ragged_dot_lowering_strategy = strategy;
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

    /// Returns the grouped-dot lowering strategy for this module.
    pub(crate) fn ragged_dot_lowering_strategy(&self) -> RaggedDotLoweringStrategy {
        self.ragged_dot_lowering_strategy
    }

    /// Derives the state used inside one `sdy.manual_computation` region's body. The target platform, grouped-dot
    /// representation, named compositions, provenance state, and module channel allocator are inherited, while the
    /// provided [`ShardMapBoundary`] becomes the innermost manual region.
    pub(crate) fn enter_manual_region(&self, boundary: ShardMapBoundary) -> Self {
        let mut bound_manual_axes = self.bound_manual_axes.clone();
        for axis in boundary.manual_axes() {
            if !bound_manual_axes.contains(axis) {
                bound_manual_axes.push(axis.clone());
            }
        }
        let mut manual_regions = self.manual_regions.clone();
        manual_regions.push(Rc::new(boundary));
        Self {
            channel_ids: self.channel_ids.clone(),
            manual_regions,
            bound_manual_axes,
            target_platform: self.target_platform.clone(),
            ragged_dot_lowering_strategy: self.ragged_dot_lowering_strategy,
            named_compositions: self.named_compositions.clone(),
            has_provenance: self.has_provenance.clone(),
            has_assertions: self.has_assertions.clone(),
            runtime_requirements: self.runtime_requirements.clone(),
            // `sdy.manual_computation` bodies are isolated from above, so they cannot use entry function arguments.
            session_constants: Rc::from([]),
        }
    }

    /// Returns the innermost enclosing manual region's [`ShardMapBoundary`], or `None` outside manual regions.
    pub(crate) fn manual_shard_map(&self) -> Option<&ShardMapBoundary> {
        self.manual_regions.last().map(Rc::as_ref)
    }

    /// Returns the nearest enclosing region binding `axis_name`.
    pub(crate) fn manual_axis_region(&self, axis_name: &str) -> Option<&ShardMapBoundary> {
        self.manual_regions
            .iter()
            .rev()
            .find(|region| region.manual_axes().iter().any(|axis| axis == axis_name))
            .map(Rc::as_ref)
    }

    /// Resolves a collective's actual manual binder and checks any payload mesh against it before communication.
    /// A payload without a mesh is valid only when the owning operation's inference accepts the actual binder.
    pub(crate) fn checked_collective_mesh<A: Into<CollectiveAxes>>(
        &self,
        operation_name: &str,
        axis_name: A,
        axis_size: usize,
        payload_mesh: Option<&LogicalMesh>,
    ) -> Result<&LogicalMesh, Error> {
        // Resolve the binding and recorded size first, preserving the replica-group diagnostics and the nearest
        // enclosing binder when another manual region shadows the axis name. That resolution also requires every named
        // axis to be bound on one mesh.
        let axis_name = axis_name.into();
        collective_replica_groups(self, &axis_name, axis_size, None)?;
        let mesh = self.manual_axis_region(&axis_name.names()[0]).unwrap().mesh();
        if payload_mesh.is_some_and(|payload_mesh| payload_mesh != mesh) {
            return Err(ProgramError::MalformedProgram(format!(
                "`{operation_name}` operation mesh does not match the enclosing manual axis `{axis_name}` mesh",
            ))
            .into());
        }
        Ok(mesh)
    }

    /// Revalidates a collective against the policy of its actual enclosing binder before it emits cross-device
    /// communication. A checked binder passes its mesh to `infer_output_types`, so a mesh-free payload cannot bypass
    /// its variation and pending-sum contract. An unchecked binder requires an ordinary mesh-free payload and passes
    /// `None`, preserving ordinary collective inference and adjoints. In either mode, inference must reproduce the
    /// recorded `output_types`, and the physical mesh and participant count remain checked.
    ///
    /// # Errors
    ///
    /// Returns a [`Error`] if no enclosing manual region binds `axis_name`, if the recorded axis size or a
    /// payload mesh disagrees with that region, if a tracked payload is used in an unchecked binding, if re-inference
    /// fails, or if it produces different output types.
    pub(crate) fn check_manual_collective<T: PartialEq, A: Into<CollectiveAxes>>(
        &self,
        operation_name: &str,
        axis_name: A,
        axis_size: usize,
        payload_mesh: Option<&LogicalMesh>,
        output_types: &[T],
        infer_output_types: impl FnOnce(Option<&LogicalMesh>) -> Result<Vec<T>, TypeError>,
    ) -> Result<(), Error> {
        let axis_name = axis_name.into();
        let mesh = self.checked_collective_mesh(operation_name, &axis_name, axis_size, payload_mesh)?;
        let checked = self.manual_axis_region(&axis_name.names()[0]).unwrap().check_manual_variation();
        if axis_name
            .names()
            .iter()
            .any(|axis_name| self.manual_axis_region(axis_name).unwrap().check_manual_variation() != checked)
        {
            return Err(ProgramError::MalformedProgram(format!(
                "`{operation_name}` axes `{axis_name}` must agree on whether their bindings check manual variation",
            ))
            .into());
        }
        if !checked && payload_mesh.is_some() {
            return Err(ProgramError::MalformedProgram(format!(
                "`{operation_name}` cannot use tracked mesh metadata inside an unchecked manual binding",
            ))
            .into());
        }
        if infer_output_types(checked.then_some(mesh)).map_err(ProgramError::from)? != output_types {
            return Err(ProgramError::MalformedProgram(format!(
                "`{operation_name}` output types do not match the enclosing manual mesh contract",
            ))
            .into());
        }
        Ok(())
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
pub(crate) fn serialize_lowered_module(
    module: &Module<'_, '_>,
    collective_state: &CollectiveLoweringState,
) -> Result<String, Error> {
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

/// Lowers a traced XLA program of any [`LowerableOperationFamily`] to a textual StableHLO/Shardy MLIR module whose
/// entry function is named `function_name`.
///
/// When `argument_shardings` and/or `result_shardings` are provided, the corresponding `sdy.sharding` attribute is
/// attached to each function argument or result, which drives per-device boundary slicing (including uneven splits)
/// in the XLA SPMD partitioner. When they are `None`, the corresponding values carry no sharding attributes, and Shardy
/// propagates their placement from the program.
///
/// # Parameters
///
///   - `program`: Traced program to lower. References and mutable state must have been discharged.
///   - `capture_types`: Types of the hidden capture arguments that precede the program inputs, in capture-table order.
///   - `global_input_types`: Global types of the program inputs.
///   - `global_output_types`: Global types of the program outputs.
///   - `function_name`: Symbol name of the entry function.
///   - `argument_shardings`: Optional shardings of the entry function arguments.
///   - `result_shardings`: Optional shardings of the entry function results.
///
/// # Errors
///
/// Returns [`LoweringError::UnresolvedReference`] or [`LoweringError::UnresolvedState`] for programs with undischarged
/// references or state, [`LoweringError::InvalidFunctionName`] for an invalid `function_name`, and other
/// [`Error`]s for operations or types that cannot be lowered.
pub fn to_mlir_module_for_program<'o, Input, Output, ProgramInput, ProgramOutput, S, O: LowerableOperationFamily>(
    program: &Program<ProgramInput, ProgramOutput, O>,
    capture_types: &[ArrayType],
    global_input_types: &Input,
    global_output_types: &Output,
    function_name: S,
    argument_shardings: Option<&[Sharding]>,
    result_shardings: Option<&[Sharding]>,
) -> Result<String, Error>
where
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
    S: AsRef<str>,
{
    Ok(lower_mlir_module_for_program(
        program,
        capture_types,
        global_input_types,
        global_output_types,
        function_name,
        argument_shardings,
        result_shardings,
        None,
    )?
    .stable_hlo)
}

/// Lowers an arbitrary traced XLA program and returns both its textual module and exact physical entry signature.
pub(crate) fn lower_mlir_module_for_program<
    'o,
    Input,
    Output,
    ProgramInput,
    ProgramOutput,
    S,
    O: LowerableOperationFamily,
>(
    program: &Program<ProgramInput, ProgramOutput, O>,
    capture_types: &[ArrayType],
    global_input_types: &Input,
    global_output_types: &Output,
    function_name: S,
    argument_shardings: Option<&[Sharding]>,
    result_shardings: Option<&[Sharding]>,
    target_platform: Option<&str>,
) -> Result<LoweredModule, Error>
where
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
    S: AsRef<str>,
{
    let argument_count = capture_types.len() + global_input_types.parameter_count();
    lower_mlir_module_for_program_with_effect_boundary(
        program,
        capture_types,
        global_input_types,
        global_output_types,
        function_name,
        &value_shardings(argument_shardings, argument_count),
        &value_shardings(result_shardings, global_output_types.parameter_count()),
        target_platform,
        &[],
        RaggedDotLoweringStrategy::default(),
        &[],
        false,
    )
}

/// Lowers an arbitrary traced XLA program like [`lower_mlir_module_for_program`], but with one optional sharding per
/// entry function argument (captures first) and per result, so that only the values whose sharding is known carry an
/// `sdy.sharding` attribute, while Shardy propagates the placement of the others from the program.
pub(crate) fn lower_mlir_module_for_program_with_value_shardings<
    'o,
    Input,
    Output,
    ProgramInput,
    ProgramOutput,
    S,
    O: LowerableOperationFamily,
>(
    program: &Program<ProgramInput, ProgramOutput, O>,
    capture_types: &[ArrayType],
    global_input_types: &Input,
    global_output_types: &Output,
    function_name: S,
    argument_shardings: &[Option<Sharding>],
    result_shardings: &[Option<Sharding>],
) -> Result<LoweredModule, Error>
where
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
    S: AsRef<str>,
{
    lower_mlir_module_for_program_with_effect_boundary(
        program,
        capture_types,
        global_input_types,
        global_output_types,
        function_name,
        argument_shardings,
        result_shardings,
        None,
        &[],
        RaggedDotLoweringStrategy::default(),
        &[],
        false,
    )
}

/// Lowers a discharged program with logical external reference-state aliases on its entry boundary. The results
/// flagged in `open_results` (one flag per logical output, or none at all) are lowered open over auto mesh axes, so that
/// the compiler chooses their placement over those axes.
///
/// `reference_states` are the discharge artifact's external-state recipes; each must name a distinct logical state
/// input. Each state input must survive the executable boundary. A mutated state must be a static device array and
/// name a hidden output of the identical type and sharding; mutated pairs are recorded as `tf.aliasing_output`
/// input-output aliases, which are non-semantic may-alias hints that merely permit backend buffer reuse. The runtime
/// never donates reference-state inputs, so an alias never authorizes in-place mutation of the caller's state buffer.
/// Read-only state may be finite bounded-dynamic because it has no hidden alias. The validation here owns the
/// physical alias contract; the XLA domain additionally enforces the supported memory, sharding, and dynamic-shape
/// classes before calling it.
pub(crate) fn lower_mlir_module_for_program_with_reference_state<
    'o,
    Input,
    Output,
    ProgramInput,
    ProgramOutput,
    S,
    O: LowerableOperationFamily,
>(
    program: &Program<ProgramInput, ProgramOutput, O>,
    capture_types: &[ArrayType],
    global_input_types: &Input,
    global_output_types: &Output,
    function_name: S,
    argument_shardings: Option<&[Sharding]>,
    result_shardings: Option<&[Sharding]>,
    target_platform: Option<&str>,
    reference_states: &[ExternalReferenceBinding],
    ragged_dot_lowering_strategy: RaggedDotLoweringStrategy,
    open_results: &[bool],
) -> Result<LoweredModule, Error>
where
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
    S: AsRef<str>,
{
    let argument_count = capture_types.len() + global_input_types.parameter_count();
    lower_mlir_module_for_program_with_effect_boundary(
        program,
        capture_types,
        global_input_types,
        global_output_types,
        function_name,
        &value_shardings(argument_shardings, argument_count),
        &value_shardings(result_shardings, global_output_types.parameter_count()),
        target_platform,
        reference_states,
        ragged_dot_lowering_strategy,
        open_results,
        true,
    )
}

/// Returns the distinct [`SessionConstant`]s that `program` references in any of its regions, in the order of their
/// first occurrence in the region arena. Lowering passes their arrays as hidden entry arguments that follow the lifted
/// captures, in this order, and so the caller that assembles the executable arguments must use the same order.
pub(crate) fn collect_session_constants<ProgramInput, ProgramOutput, O: CoreOperation<Type = ArrayIrType>>(
    program: &Program<ProgramInput, ProgramOutput, O>,
) -> Vec<SessionConstant>
where
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
{
    let mut session_constants = Vec::<SessionConstant>::new();
    for region in program.regions().iter() {
        for atom in region.atoms() {
            if let Some(Constant::Session(constant)) = atom.as_constant()
                && !session_constants.contains(constant)
            {
                session_constants.push(constant.clone());
            }
        }
    }
    session_constants
}

/// Expands optional shardings of every value of a signature with `count` values into one optional sharding per value.
/// Provided shardings are kept as they are, so that a count mismatch is still reported when the module is lowered.
fn value_shardings(shardings: Option<&[Sharding]>, count: usize) -> Vec<Option<Sharding>> {
    match shardings {
        Some(shardings) => shardings.iter().cloned().map(Some).collect(),
        None => vec![None; count],
    }
}

/// Lowers the entry boundary, optionally exporting the runtime ordered-I/O chain.
fn lower_mlir_module_for_program_with_effect_boundary<
    'o,
    Input,
    Output,
    ProgramInput,
    ProgramOutput,
    S,
    O: LowerableOperationFamily,
>(
    program: &Program<ProgramInput, ProgramOutput, O>,
    capture_types: &[ArrayType],
    global_input_types: &Input,
    global_output_types: &Output,
    function_name: S,
    argument_shardings: &[Option<Sharding>],
    result_shardings: &[Option<Sharding>],
    target_platform: Option<&str>,
    reference_states: &[ExternalReferenceBinding],
    ragged_dot_lowering_strategy: RaggedDotLoweringStrategy,
    open_results: &[bool],
    export_effects: bool,
) -> Result<LoweredModule, Error>
where
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
    S: AsRef<str>,
{
    if contains_unresolved_references(program) {
        return Err(Error::Lowering(LoweringError::UnresolvedReference { construct: "program".to_string() }));
    }
    if contains_unresolved_state(program) {
        return Err(Error::Lowering(LoweringError::UnresolvedState { construct: "program".to_string() }));
    }
    let function_name = normalize_function_name(function_name.as_ref())?;
    // Session constants are carried by hidden arguments that form the suffix of the capture arguments. The caller
    // declares them through the trailing `capture_types`, in the order of `collect_session_constants`.
    let session_constants: Rc<[SessionConstant]> = collect_session_constants(program).into();
    let session_count = session_constants.len();
    if session_count > 0
        && (capture_types.len() < session_count
            || !capture_types[capture_types.len() - session_count..]
                .iter()
                .eq(session_constants.iter().map(|constant| constant.r#type().into_owned()).collect::<Vec<_>>().iter()))
    {
        return Err(ProgramError::MalformedProgram(format!(
            "the trailing capture types of the entry function must be the types of its {session_count} session \
             constant(s)",
        ))
        .into());
    }
    let global_input_types = global_input_types.parameters().cloned().collect::<Vec<_>>();
    let global_output_types = global_output_types.parameters().cloned().collect::<Vec<_>>();
    let logical_argument_types =
        capture_types.iter().cloned().chain(global_input_types.iter().cloned()).collect::<Vec<_>>();
    let effects = program.effects().classes();
    let ordered_io = effects.contains(EffectClass::OrderedIo) || effects.contains(EffectClass::DeviceOrderedIo);
    let mut signature = ExecutableSignature::new(logical_argument_types.as_slice(), global_output_types.as_slice())
        .with_effects(
            export_effects && ordered_io,
            effects.contains(EffectClass::UnorderedIo),
            effects.contains(EffectClass::OrderedAssertion),
            !effects.contains(EffectClass::OrderedIo),
        );
    if argument_shardings.len() != logical_argument_types.len() {
        return Err(Error::Compilation(CompilationError::ShardingCountMismatch {
            kind: "argument",
            expected: logical_argument_types.len(),
            actual: argument_shardings.len(),
        }));
    }
    if result_shardings.len() != global_output_types.len() {
        return Err(Error::Compilation(CompilationError::ShardingCountMismatch {
            kind: "result",
            expected: global_output_types.len(),
            actual: result_shardings.len(),
        }));
    }
    let mut aliases = HashMap::with_capacity(reference_states.len());
    let mut state_inputs = HashSet::with_capacity(reference_states.len());
    let mut aliased_outputs = HashSet::with_capacity(reference_states.len());
    for state in reference_states {
        let logical_input_index = state.source().flat_input_index(capture_types.len())?;
        if !state_inputs.insert(logical_input_index) {
            return Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                message: format!("logical state input {logical_input_index} appears more than once"),
            }));
        }
        let input_type = logical_argument_types.get(logical_input_index).ok_or_else(|| {
            Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                message: format!("logical input index {logical_input_index} is out of range"),
            })
        })?;
        let physical_input_index = signature.input_mapping()[logical_input_index].ok_or_else(|| {
            Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                message: format!("logical state input {logical_input_index} is erased from the executable boundary"),
            })
        })?;
        if let Some(logical_output_index) = state.output_index() {
            let output_type = global_output_types.get(logical_output_index).ok_or_else(|| {
                Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                    message: format!("logical output index {logical_output_index} is out of range"),
                })
            })?;
            let physical_output_index = signature.output_mapping()[logical_output_index].ok_or_else(|| {
                Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                    message: format!(
                        "logical state output {logical_output_index} is erased from the executable boundary",
                    ),
                })
            })?;
            if input_type.static_shape().is_none() || output_type.static_shape().is_none() {
                return Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                    message: format!(
                        "state input {logical_input_index} and output {logical_output_index} must be static because \
                         bounded-dynamic mutation alias compatibility is unsupported",
                    ),
                }));
            }
            if input_type.memory() != Memory::Device || output_type.memory() != Memory::Device {
                return Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                    message: format!(
                        "state input {logical_input_index} and output {logical_output_index} must use device \
                         memory",
                    ),
                }));
            }
            if input_type != output_type {
                return Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                    message: format!(
                        "state input {logical_input_index} type `{input_type}` is incompatible with output \
                         {logical_output_index} type `{output_type}`",
                    ),
                }));
            }
            match (&argument_shardings[logical_input_index], &result_shardings[logical_output_index]) {
                (Some(argument_sharding), Some(result_sharding)) if argument_sharding != result_sharding => {
                    return Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                        message: format!(
                            "state input {logical_input_index} and output {logical_output_index} must use the same \
                             sharding",
                        ),
                    }));
                }
                (Some(_), None) | (None, Some(_)) => {
                    return Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                        message:
                            "reference-state aliases require both argument and result sharding metadata or neither"
                                .to_string(),
                    }));
                }
                _ => {}
            }
            if !aliased_outputs.insert(physical_output_index) {
                return Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                    message: "reference-state aliases must be injective".to_string(),
                }));
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
    for sharding in argument_shardings.iter().chain(result_shardings).flatten() {
        signature_mesh = Some(match signature_mesh.take() {
            Some(existing_mesh) => merge_logical_meshes(&existing_mesh, sharding.mesh())?,
            None => sharding.mesh().clone(),
        });
    }
    let mesh = collect_nested_sharding_mesh(program, signature_mesh)?;
    if let Some(mesh) = mesh.as_ref() {
        let mesh_operation = mesh.to_mlir(location)?;
        module.body()?.append_operation(mesh_operation)?;
    }

    // Trace each distinct named semantic decomposition once and retain it in the module-scoped lowering state so
    // entry and nested function bodies resolve the same private symbol.
    let named_compositions = Rc::new(collect_named_composition_functions(program, target_platform)?);

    // Module-scoped lowering state, shared between the entry function body and private functions below so channel ids
    // stay unique and target, representation policy, named compositions, and provenance state reach nested bodies.
    let collective_state = CollectiveLoweringState::new()
        .with_target_platform(target_platform)
        .with_ragged_dot_lowering_strategy(ragged_dot_lowering_strategy)
        .with_named_compositions(named_compositions.clone())
        .with_session_constants(session_constants);

    // Deduplicate `jit_call` callees that occur more than once into shared private `func.func`s, so repeated nested
    // programs (identical transformer blocks, or the per-block primal and pullback programs produced by `grad`) lower
    // to one function plus N `func.call`s instead of N inlined copies. The map is empty for modules without repeated
    // calls, in which case every `jit_call` inlines exactly as before.
    let nested_functions = Rc::new(collect_jit_call_functions(program, target_platform)?);
    {
        let mut module_block = module.body()?;
        for key in &named_compositions.order {
            let function = named_compositions.functions.get(key).unwrap();
            emit_named_composition_function(
                &mut module_block,
                function,
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
    let mut physical_argument_tensor_types = physical_argument_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location).map(|r#type| r#type.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    if signature.has_ordered_io() {
        physical_argument_tensor_types.push(context.stable_hlo_token_type()?.as_ref());
    }
    let mut physical_output_tensor_types = physical_output_types
        .iter()
        .map(|array_type| lower_tensor_type(array_type, &context, location).map(|r#type| r#type.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    if signature.has_ordered_io() {
        physical_output_tensor_types.push(context.stable_hlo_token_type()?.as_ref());
    }
    let mut argument_sharding_attributes = signature
        .physical_optional_input_shardings(argument_shardings)
        .iter()
        .map(|sharding| sharding.as_ref().map(|sharding| sharding.to_mlir(location)).transpose())
        .collect::<Result<Vec<_>, _>>()?;
    // Results whose placement over auto mesh axes the compiler chooses are open over those axes, like JAX lowers the
    // shardings of output types without `out_shardings` (refer to `shardy_tensor_sharding_attribute`).
    let physical_open_results =
        if open_results.is_empty() { Vec::new() } else { signature.project_outputs(open_results) };
    let mut result_sharding_attributes = signature
        .physical_optional_output_shardings(result_shardings)
        .iter()
        .enumerate()
        .map(|(index, sharding)| {
            let open = physical_open_results.get(index).copied().unwrap_or(false);
            sharding.as_ref().map(|sharding| shardy_tensor_sharding_attribute(sharding, location, open)).transpose()
        })
        .collect::<Result<Vec<_>, _>>()?;
    // The effect token of a signature side is annotated when any value of that side is, as each device threads its
    // own token.
    if signature.has_ordered_io() && mesh.is_some() {
        let token_sharding = context.shardy_tensor_sharding(
            context.flat_symbol_ref_attribute(SHARDY_MESH_SYMBOL_NAME),
            &[],
            &[],
            &[],
            shardy::ReductionOperation::Sum,
        )?;
        for shardings in [&mut argument_sharding_attributes, &mut result_sharding_attributes] {
            if shardings.iter().any(Option::is_some) {
                shardings.push(Some(token_sharding));
            }
        }
    }
    // Internal layout constraints do not constrain the executable ABI. Declare layouts on the physical boundary,
    // whose types already omit static zeros and include hidden extent scalars; tokens have no array type.
    // Placement belongs to physical array slots, including captures, but never to the appended effect token.
    // Default device placement needs no attribute; host identities still need attributes even without transfers.
    let function_arguments = physical_argument_tensor_types
        .iter()
        .enumerate()
        .map(|(index, tensor_type)| {
            let mut attributes = HashMap::new();
            if let Some(sharding) = argument_sharding_attributes.get(index).and_then(Option::as_ref) {
                attributes.insert("sdy.sharding".into(), sharding.as_ref());
            }
            if let Some(r#type) = physical_argument_types.get(index).filter(|r#type| r#type.memory() != Memory::Device)
            {
                attributes.insert(
                    "mhlo.memory_kind".into(),
                    context.string_attribute(memory_placement_kind(r#type.memory())).as_ref(),
                );
            }
            if let Some(output_index) = aliases.get(&index) {
                attributes.insert(
                    "tf.aliasing_output".into(),
                    context.integer_attribute(context.signless_integer_type(64), *output_index as i64).as_ref(),
                );
            }
            if let Some(array_type) = physical_argument_types.get(index)
                && let Some(layout) = lower_boundary_layout(array_type, "input", &context)?
            {
                attributes.insert("mhlo.layout_mode".into(), layout);
            }
            let attributes = (!attributes.is_empty()).then_some(attributes);
            Ok::<_, Error>(TypeAndAttributes { r#type: tensor_type.as_ref(), attributes })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let function_results = physical_output_tensor_types
        .iter()
        .enumerate()
        .map(|(index, tensor_type)| {
            let mut attributes = HashMap::new();
            if let Some(sharding) = result_sharding_attributes.get(index).and_then(Option::as_ref) {
                attributes.insert("sdy.sharding".into(), sharding.as_ref());
            }
            if let Some(r#type) = physical_output_types.get(index).filter(|r#type| r#type.memory() != Memory::Device) {
                attributes.insert(
                    "mhlo.memory_kind".into(),
                    context.string_attribute(memory_placement_kind(r#type.memory())).as_ref(),
                );
            }
            if let Some(array_type) = physical_output_types.get(index)
                && let Some(layout) = lower_boundary_layout(array_type, "output", &context)?
            {
                attributes.insert("mhlo.layout_mode".into(), layout);
            }
            let attributes = (!attributes.is_empty()).then_some(attributes);
            Ok::<_, Error>(TypeAndAttributes { r#type: tensor_type.as_ref(), attributes })
        })
        .collect::<Result<Vec<_>, _>>()?;

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
            // Session constants are not program inputs, so the lifted form omits the trailing session arguments of the
            // capture prefix from the entry inputs, while `capture_values` keeps them as its suffix for
            // `CollectiveLoweringState::lower_session_constant`.
            debug_assert!(
                program.input_count() + session_count == logical_argument_values.len()
                    || program.input_count() == public_input_values.len(),
            );
            let lifted_input_values: Vec<_>;
            let input_values = if program.input_count() + session_count == logical_argument_values.len() {
                lifted_input_values = capture_values[..capture_values.len() - session_count]
                    .iter()
                    .chain(public_input_values)
                    .copied()
                    .collect();
                lifted_input_values.as_slice()
            } else {
                public_input_values
            };
            let mut effect_tokens = EffectTokens::default();
            if let Some(index) = signature.ordered_io_input_index() {
                effect_tokens.set(EffectClass::OrderedIo, function_block.argument(index).unwrap().as_ref());
            }
            let logical_outputs = lower_program_outputs_with_inputs(
                program,
                capture_values,
                input_values,
                &mut function_block_ref,
                &context,
                location.as_ref(),
                Some(&nested_functions),
                &collective_state,
                &mut effect_tokens,
            )?;
            let mut physical_outputs = signature.project_outputs(logical_outputs.as_slice());
            // XLA writes the runtime-size trailer of a bounded dynamic result only in the `SliceToDynamic` that
            // `DynamicPadder` inserts before each output leaf. When several such leaves carry values that XLA proves
            // equal (the same SSA value, or equal computations merged by CSE), one `SliceToDynamic` survives and copy
            // insertion fills the other result with a plain copy of the bounded payload, which leaves its trailer
            // unset (i.e., the buffer reports zero-sized dynamic dimensions). Routing these results through one
            // optimization barrier makes every leaf a distinct barrier result. XLA expands barriers only after copy
            // insertion, so each leaf keeps its own `SliceToDynamic`. A single dynamic result cannot alias a static
            // one, so the barrier is only needed with two or more of them.
            let mut dynamic_physical_indices = signature
                .output_dimensions()
                .iter()
                .map(|output_dimension| signature.output_mapping()[output_dimension.logical_output_index()].unwrap())
                .collect::<Vec<_>>();
            dynamic_physical_indices.dedup();
            if dynamic_physical_indices.len() > 1 {
                let barrier_inputs =
                    dynamic_physical_indices.iter().map(|index| physical_outputs[*index]).collect::<Vec<_>>();
                let barrier = function_block_ref
                    .append_operation(stable_hlo::optimization_barrier(barrier_inputs.as_slice(), location)?)?;
                for (result_index, physical_index) in dynamic_physical_indices.into_iter().enumerate() {
                    physical_outputs[physical_index] = barrier.result(result_index).unwrap().as_ref();
                }
            }
            for output_dimension in signature.output_dimensions() {
                physical_outputs.push(lower_runtime_dimension_size_i64(
                    logical_outputs[output_dimension.logical_output_index()],
                    output_dimension.axis(),
                    &mut function_block_ref,
                    &context,
                    location.as_ref(),
                )?);
            }
            if signature.has_ordered_io() {
                physical_outputs.push(effect_tokens.get(EffectClass::OrderedIo).unwrap());
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

    remove_unused_constants(&module)?;
    if !module.verify()? {
        return Err(Error::Lowering(LoweringError::VerificationFailure));
    }
    signature.requires_assertion_handler |= collective_state.has_assertions.get();
    let requires_assertion_handler = signature.requires_assertion_handler;
    let mut runtime_requirements = collective_state.runtime_requirements.borrow().clone();
    if requires_assertion_handler {
        runtime_requirements.push(builtin_requirement("ryft.assert"));
    }
    if signature.has_ordered_io() {
        runtime_requirements.push(builtin_requirement("ryft.ordered-io"));
    }
    let runtime_requirements = normalize_runtime_requirements(runtime_requirements)?.into();
    Ok(LoweredModule {
        stable_hlo: serialize_lowered_module(&module, &collective_state)?,
        signature,
        requires_assertion_handler,
        runtime_requirements,
    })
}

/// Value type that can be materialized as a StableHLO dense constant during benchmark lowering.
pub(crate) trait MlirLowerableValue: CoreValue<Type = ArrayType> + 'static {
    /// Builds a dense-elements attribute containing this value.
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, Error>;

    /// Lowers this program constant inside a function body.
    #[inline]
    fn lower_constant_value<'b, 'c: 'b, 't: 'c, B, L>(
        &self,
        _captured_values: &[ValueRef<'b, 'c, 't>],
        block: &mut B,
        context: &'c MlirContext<'t>,
        location: L,
    ) -> Result<ValueRef<'b, 'c, 't>, Error>
    where
        B: Block<'b, 'c, 't>,
        L: Copy + Location<'c, 't>,
    {
        lower_literal_value(self, block, context, location)
    }
}

impl MlirLowerableValue for ArrayConstant {
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, Error> {
        match self {
            Self::Captured(value) => Err(ProgramError::Concretization {
                message: format!(
                    "captured constant #{} cannot be materialized as a literal without capture bindings",
                    value.index()
                ),
            }
            .into()),
            Self::Session(value) => Err(ProgramError::Concretization {
                message: format!("session constant {value} cannot be materialized as a literal"),
            }
            .into()),
            Self::Boolean(value) => CoreArray::scalar(*value)?.to_dense_elements_attribute(tensor_type, context),
        }
    }

    fn lower_constant_value<'b, 'c: 'b, 't: 'c, B, L>(
        &self,
        captured_values: &[ValueRef<'b, 'c, 't>],
        block: &mut B,
        context: &'c MlirContext<'t>,
        location: L,
    ) -> Result<ValueRef<'b, 'c, 't>, Error>
    where
        B: Block<'b, 'c, 't>,
        L: Copy + Location<'c, 't>,
    {
        match self {
            Self::Captured(value) => lower_captured_constant(value, captured_values),
            Self::Session(value) => {
                Err(Error::Lowering(LoweringError::UnreachableSessionConstant { constant: value.to_string() }))
            }
            Self::Boolean(_) => lower_literal_value(self, block, context, location),
        }
    }
}

// [`ArrayType`] is used as the value representation for abstract linear XLA programs. It can type
// program atoms, but it is not a concrete literal; lowering paths that need a real value must
// supply it through captured arguments instead of materializing it from type metadata.
impl MlirLowerableValue for ArrayType {
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        _tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        _context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, Error> {
        Err(ProgramError::Concretization {
            message: format!("abstract XLA value `{self}` cannot be materialized as a StableHLO literal"),
        }
        .into())
    }
}

// Concrete host literal lowering used by [`ConstantOperation`] and MLIR snapshot tooling.
impl MlirLowerableValue for CoreArray {
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, Error> {
        let data_type = self.r#type().data_type();
        if data_type == DataType::Zero {
            return types::lower_f64_scalar_elements_attribute(data_type, tensor_type, 0.0, context);
        }
        if data_type == DataType::Token {
            return Err(Error::Lowering(LoweringError::UnsupportedElementType {
                data_type,
                context: "dense literal materialization".to_string(),
            }));
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
            return Ok(context
                .dense_bool_elements_attribute(
                    tensor_type,
                    bytes.iter().map(|value| value & 1 != 0).collect::<Vec<_>>().as_slice(),
                )?
                .cast::<DenseElementsAttributeRef>()
                .unwrap());
        }

        context.dense_elements_attribute_from_raw_buffer(tensor_type, bytes.as_ref()).map_err(Error::from)
    }
}

/// Lowers a plain traced [`CoreProgram`](ryft_core::Program) to a textual StableHLO MLIR module.
#[cfg(test)]
pub(crate) fn to_mlir_module_for_plain_program<
    V: MlirLowerableValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
    O: LowerableOperation<Type = ArrayType>,
    S: AsRef<str>,
>(
    program: &CoreProgram<V, O, Input, Output>,
    function_name: S,
) -> Result<String, Error> {
    to_mlir_module_for_plain_program_with_ragged_dot_lowering_strategy(
        program,
        function_name,
        RaggedDotLoweringStrategy::default(),
    )
}

/// Lowers a plain traced program with an explicit XLA grouped-dot representation.
#[cfg(test)]
pub(crate) fn to_mlir_module_for_plain_program_with_ragged_dot_lowering_strategy<
    V: MlirLowerableValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
    O: LowerableOperation<Type = ArrayType>,
    S: AsRef<str>,
>(
    program: &CoreProgram<V, O, Input, Output>,
    function_name: S,
    ragged_dot_lowering_strategy: RaggedDotLoweringStrategy,
) -> Result<String, Error> {
    let function_name = normalize_function_name(function_name.as_ref())?;
    let context = MlirContext::new();
    let location = context.unknown_location();
    let module = context.module(location)?;
    // Module-scoped collective lowering state: this entry point lowers one whole module.
    let collective_state =
        CollectiveLoweringState::new().with_ragged_dot_lowering_strategy(ragged_dot_lowering_strategy);
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
                    .zip(program.input_ids())
                    .map(|(tensor_type, atom_id)| {
                        let memory = program.atoms()[atom_id.index()].r#type().memory();
                        let attributes = (memory != Memory::Device).then(|| {
                            HashMap::from([(
                                "mhlo.memory_kind".into(),
                                context.string_attribute(memory_placement_kind(memory)).as_ref(),
                            )])
                        });
                        TypeAndAttributes { r#type: tensor_type.as_ref(), attributes }
                    })
                    .collect(),
                results: output_tensor_types
                    .iter()
                    .zip(program.output_ids())
                    .map(|(tensor_type, atom_id)| {
                        let memory = program.atoms()[atom_id.index()].r#type().memory();
                        let attributes = (memory != Memory::Device).then(|| {
                            HashMap::from([(
                                "mhlo.memory_kind".into(),
                                context.string_attribute(memory_placement_kind(memory)).as_ref(),
                            )])
                        });
                        TypeAndAttributes { r#type: tensor_type.as_ref(), attributes }
                    })
                    .collect(),
                ..Default::default()
            },
            function_region,
            location,
        )?
    })?;

    remove_unused_constants(&module)?;
    if !module.verify()? {
        return Err(Error::Lowering(LoweringError::VerificationFailure));
    }

    serialize_lowered_module(&module, &collective_state)
}

/// Collects meshes referenced by emitted sharding constraints and nested-computation boundaries.
pub(crate) fn collect_nested_sharding_mesh<ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    program: &Program<ProgramInput, ProgramOutput, O>,
    existing: Option<LogicalMesh>,
) -> Result<Option<LogicalMesh>, Error>
where
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
{
    // Attached computations share the canonical region arena. Type metadata alone does not require a mesh:
    // bounded-dynamic replica lowering retains that metadata without emitting Shardy operations.
    let mut mesh = existing;
    for region in program.regions().iter() {
        for instruction in region.instructions() {
            // Payloads are recognized through their projections, so any family that embeds them declares their meshes.
            let operation = instruction.operation();
            let declared_mesh = operation
                .projected_payload::<ShardMapOperation>()
                .map(|operation| operation.boundary().mesh())
                .or_else(|| {
                    operation.projected_payload::<ReshardOperation>().map(|operation| operation.sharding().mesh())
                })
                .or_else(|| {
                    operation
                        .projected_payload::<ConstrainShardingOperation>()
                        .map(|operation| operation.sharding().mesh())
                });
            let incoming = if let Some(mesh) = declared_mesh {
                Some(mesh.clone())
            } else {
                let output_axes = operation
                    .projected_payload::<DynamicBroadcastOperation>()
                    .map(DynamicBroadcastOperation::output_axes)
                    .or_else(|| {
                        operation.projected_payload::<BroadcastOperation>().map(BroadcastOperation::output_axes)
                    });
                let emits_constraint = if let Some(output_axes) = output_axes {
                    let input_type = region.atoms()[instruction.inputs()[0].index()].r#type();
                    let input_type = <&ArrayType>::try_from(input_type.as_ref()).map_err(ProgramError::from)?;
                    let output_type = region.atoms()[instruction.outputs()[0].index()].r#type();
                    let output_type = <&ArrayType>::try_from(output_type.as_ref()).map_err(ProgramError::from)?;
                    broadcast_changes_explicit_sharding(input_type, output_type, output_axes)
                } else if operation.projected_payload::<CastOperation<ArrayIrType>>().is_some()
                    || operation.projected_payload::<CastOperation<ArrayType>>().is_some()
                {
                    // Casts of first-class dimensions carry no placement, and array casts constrain placement only
                    // when the explicit placement changes.
                    let input_type = region.atoms()[instruction.inputs()[0].index()].r#type();
                    let output_type = region.atoms()[instruction.outputs()[0].index()].r#type();
                    match (<&ArrayType>::try_from(input_type.as_ref()), <&ArrayType>::try_from(output_type.as_ref())) {
                        (Ok(input_type), Ok(output_type)) => cast_changes_explicit_placement(input_type, output_type),
                        _ => false,
                    }
                } else {
                    operation
                        .projected_payload::<DynamicReshapeOperation>()
                        .is_some_and(|operation| operation.output_sharding().is_some())
                        || operation
                            .projected_payload::<ReshapeOperation>()
                            .is_some_and(|operation| operation.output_sharding().is_some())
                        || operation
                            .projected_payload::<GatherOperation>()
                            .is_some_and(|operation| operation.output_sharding().is_some())
                        || operation
                            .projected_payload::<ScatterOperation>()
                            .is_some_and(|operation| operation.output_sharding().is_some())
                        || operation.projected_payload::<ZeroLikeOperation<ArrayType>>().is_some()
                        || operation.projected_payload::<OneLikeOperation<ArrayType>>().is_some()
                };
                if !emits_constraint {
                    continue;
                }
                let output_type = region.atoms()[instruction.outputs()[0].index()].r#type();
                let output_type = <&ArrayType>::try_from(output_type.as_ref()).map_err(ProgramError::from)?;
                output_type.sharding().map(|sharding| sharding.mesh().clone())
            };
            if let Some(incoming) = incoming {
                mesh = Some(match mesh.take() {
                    Some(existing_mesh) => merge_logical_meshes(&existing_mesh, &incoming)?,
                    None => incoming,
                });
            }
        }
    }
    Ok(mesh)
}

fn merge_logical_meshes(existing: &LogicalMesh, incoming: &LogicalMesh) -> Result<LogicalMesh, Error> {
    let mut merged_axes = existing.axes().to_vec();
    for incoming_axis in incoming.axes() {
        match existing.axis_size(incoming_axis.name()) {
            Some(existing_size) if existing_size != incoming_axis.size() => {
                return Err(Error::Lowering(LoweringError::ConflictingMeshAxisSize {
                    axis_name: incoming_axis.name().to_string(),
                    expected: existing_size,
                    actual: incoming_axis.size(),
                }));
            }
            Some(_) => {}
            None => merged_axes.push(incoming_axis.clone()),
        }
    }
    LogicalMesh::new(merged_axes).map_err(Error::from)
}

/// Removes every `stablehlo.constant` operation whose result has no uses from `module`, including those in nested
/// regions. Lowering materializes first-class dimension extents eagerly (refer to
/// [`lower_dimension_extent`](crate::experimental::lowering::types::lower_dimension_extent)), while their consumers
/// often read them only for dynamic axes (e.g., the explicit-extent collectives only refine dynamic result axes), so a
/// static extent would otherwise remain in the lowered module as a dead constant. Constants are pure, so removing an
/// unused one never changes the semantics of the module.
pub(crate) fn remove_unused_constants(module: &Module<'_, '_>) -> Result<(), Error> {
    remove_unused_constants_in_block(module.body()?)
}

/// Removes every unused `stablehlo.constant` operation from `block` and from the regions nested in its operations.
fn remove_unused_constants_in_block<'b, 'c: 'b, 't: 'c>(block: BlockRef<'b, 'c, 't>) -> Result<(), Error> {
    let mut unused_constants = Vec::new();
    for operation in block.operations()? {
        let operation = operation?;
        for region in operation.regions() {
            for nested_block in region?.blocks()? {
                remove_unused_constants_in_block(nested_block?)?;
            }
        }
        if operation.name().as_str() == Ok("stablehlo.constant") && operation.result(0)?.uses()?.next().is_none() {
            unused_constants.push(operation);
        }
    }
    for constant in unused_constants {
        // SAFETY: the removed constant has no uses, so no remaining operation references its result, and dropping the
        // detached operation that `remove_operation` returns destroys it.
        unsafe { block.remove_operation(constant)? };
    }
    Ok(())
}

/// Lowers one traced instruction to the corresponding StableHLO operation and returns its result value.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_instruction<'b, 'c: 'b, 't: 'c, ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    program: &Program<ProgramInput, ProgramOutput, O>,
    instruction: &Instruction<O>,
    _atom_values: &[Option<ValueRef<'b, 'c, 't>>],
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    captured_values: &[ValueRef<'b, 'c, 't>],
    nested_functions: Option<&Rc<JitCallFunctionMap<O>>>,
    collective_state: &CollectiveLoweringState,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error>
where
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
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
    let regions = if crate::operations::composites::lowers_without_regions(
        instruction.operation(),
        &input_types,
        &output_types,
        nested_functions,
        collective_state.target_platform(),
    ) {
        Vec::new()
    } else {
        instruction
            .regions()
            .iter()
            .map(|region| program.region_ref(*region).map(|region| region.to_program()))
            .collect::<Result<Vec<_>, _>>()?
    };
    let mut lowerer = Lowerer::new(*block, context, location)
        .with_regions(regions)
        .with_nested_functions(nested_functions.cloned())
        .with_captured_values(captured_values)
        .with_effect_tokens(*effect_tokens)
        .with_collective_state(collective_state.clone());
    let outputs =
        instruction
            .operation()
            .lower(&mut lowerer, input_values, input_types.as_slice(), output_types.as_slice())?;
    if lowerer.effect_tokens.ordered_assertion.is_some() {
        collective_state.has_assertions.set(true);
    }
    *effect_tokens = lowerer.effect_tokens;
    Ok(outputs)
}

/// Normalizes a user-provided MLIR symbol name.
pub(crate) fn normalize_function_name(function_name: &str) -> Result<String, Error> {
    let function_name = function_name.trim();
    let function_name = function_name.strip_prefix('@').unwrap_or(function_name);
    if function_name.is_empty() || function_name.chars().any(char::is_whitespace) {
        return Err(Error::Lowering(LoweringError::InvalidFunctionName { function_name: function_name.to_string() }));
    }
    Ok(function_name.to_string())
}

#[cfg(test)]
pub(crate) mod tests;
