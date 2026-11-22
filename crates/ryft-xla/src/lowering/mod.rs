//! Lowering of staged XLA programs to textual StableHLO/Shardy MLIR modules.
//!
//! [`to_mlir_module_for_program`] is the public entry point. It lowers a program of any [`LowerableOperationFamily`]
//! into one module whose entry function takes the hidden capture arguments followed by the program inputs, and whose
//! function signature optionally carries `sdy.sharding` attributes. The crate-internal entry points that the
//! compilation path uses also return the `ExecutableSignature` that maps the logical program boundary to the
//! physical executable boundary (e.g., erasing static zero-sized leaves and appending hidden extent scalars and effect
//! tokens), together with the runtime requirements of the module. Programs must have their references and mutable
//! state discharged before lowering, and failures that only lowering can detect are reported as [`LoweringError`]s
//! wrapped in [`Error::Lowering`].
//!
//! Each operation lowers itself through [`LowerableOperation::lower`], which receives a [`Lowerer`] that exposes the
//! block receiving the emitted operations, the MLIR context and location, the attached regions of the instruction, and
//! the ordered effect token chains of the enclosing lowering scope. Operation families dispatch to the payloads of
//! their variants through the [`lower_operation_variants`](crate::lower_operation_variants) macro, so downstream
//! crates extend lowering by implementing [`LowerableOperation`] for their own operations and families. One
//! crate-internal `CollectiveLoweringState` per module carries the state that nested regions share, such as the
//! collective channel counter, the enclosing manual regions, the target platform, and the registry of named
//! compositions.
//!
//! The crate-internal submodules own the parts of this pipeline. `admission` rejects data-dependent dynamic programs
//! whose padding could become observable, `compositions` emits the shared private functions of named compositions,
//! `regions` lowers nested programs and regions inline, `signature` defines the executable signature, and `types`
//! lowers types, constants, and dynamic dimensions.

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use ryft_core::{
    Array, ArrayOperation, ArrayType, AtomId, BroadcastLikeOperation, BroadcastOperation, CastOperation,
    CollectiveAxes, CompositeKey, ConstrainShardingOperation, DataType, Device, DeviceMesh, DimensionOperation,
    DimensionType, DimensionValue, DynamicBroadcastOperation, DynamicReshapeOperation, EffectClass, EffectClasses,
    ExternalReferenceBinding, FillLikeOperation, GatherOperation, Instruction, IotaLikeOperation, IotaOperation,
    LogicalMesh, Memory, MeshAxisType, OneLikeOperation, OneOperation, Operation as CoreOperation,
    OperationPayloadProjection, PLACE_OPERATION_NAME, Parameterized, PlaceOperation, PlacementRequest, ProgramError,
    Provenance, RegionId, RegionRef, RematerializationOptimizationBarrier, ReshapeOperation, ReshardOperation,
    ScatterOperation, Shape, ShardMapBoundary, ShardMapOperation, Sharding, ShardingDimension, StopGradientOperation,
    TagOperation, TransferToMemoryOperation, Type as CoreType, TypeError, Typed, TypedProgram, UnreducedKind, Value,
    ZeroLikeOperation, ZeroOperation,
};
use ryft_mlir::dialects::{func, shardy, stable_hlo};
use ryft_mlir::{
    Attribute, AttributeRef, Block, BlockRef, Context as MlirContext, DenseElementsAttributeRef, Location, LocationRef,
    Module, OpRef, Operation as MlirOperation, OperationPrintingFlags, OperationRef, OperationResultRef, Region,
    TensorTypeRef, Type as MlirType, TypeAndAttributes, Value as MlirValue, ValueRef, WalkOrder, WalkResult,
};

use crate::devices::MemoryExtension;
use crate::lowering::compositions::{
    NamedCompositionFunctionMap, collect_named_composition_functions, emit_named_composition_function,
};
use crate::lowering::regions::{lower_nested_program_inline, lower_program_outputs_with_inputs};
use crate::lowering::signature::ExecutableSignature;
use crate::lowering::types::{
    lower_boundary_layout, lower_captured_constant, lower_literal_value, lower_runtime_dimension_size_i64,
    lower_tensor_type, lower_unplaced_constant_output,
};
use crate::operations::collectives::collective_replica_groups;
use crate::operations::jit_call::{JitCallFunctionMap, collect_jit_call_functions, emit_jit_call_function};
use crate::operations::manipulation::broadcasting::broadcast_changes_explicit_sharding;
use crate::operations::manipulation::conversions::cast_changes_explicit_placement;
use crate::operations::manipulation::memory::validate_boundary_memory;
use crate::operations::sharding::constrain_sharding::lower_sharding_constraint;
use crate::sharding::ShardyTensorSharding;
use crate::{
    ArrayConstant, CompilationError, Constant, Error, ExecutionError, FlatProgram, JIT_CALL_OPERATION_NAME,
    JitCallOperation, Operation, Program, RaggedDotLoweringStrategy, RetainedArray, RuntimeRequirement,
    SHARDY_MESH_SYMBOL_NAME, ToMlir, Type, normalize_runtime_requirements,
};

// TODO(eaplatanios): Review this module.

/// Represents errors produced while validating a program for XLA lowering, constructing its StableHLO
/// representation, or verifying the result.
///
/// Public lowering entry points return these errors through [`Error::Lowering`]. Core program and native MLIR errors
/// retain their original types in [`Error::Core`] and [`Error::Mlir`], respectively.
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

    /// A sharding or `shard_map` boundary of the module refers to a mesh whose topology (i.e., axis names and sizes in
    /// order, refer to [`LogicalMesh::has_same_topology`]) differs from the topology of the module mesh. A lowered
    /// module declares one `sdy.mesh`, which assigns device ids to mesh coordinates, while collectives and
    /// `axis_index` compute coordinates from the order of the axes of their own mesh. Meshes that differ in order or
    /// in axes would therefore disagree about which device holds which coordinate, and a union of different meshes
    /// cannot be placed on one device mesh. Meshes that differ only in axis types share the module mesh.
    #[error("mesh `{actual}` does not have the topology of the module mesh `{expected}`")]
    MismatchedMeshTopology { expected: LogicalMesh, actual: LogicalMesh },

    #[error(
        "`{}` cannot move a value to the devices {:?} of mesh `{}` inside a program compiled for the devices {:?} of \
         mesh `{}`",
        PLACE_OPERATION_NAME,
        .target.devices().iter().map(|device| device.id()).collect::<Vec<_>>(),
        .target.logical_mesh(),
        .mesh.devices().iter().map(|device| device.id()).collect::<Vec<_>>(),
        .mesh.logical_mesh(),
    )]
    PlacementOutsideProgramMesh { target: DeviceMesh, mesh: DeviceMesh },

    #[error(
        "`{}` cannot move a value to the device {} inside a program compiled for the devices {:?} of mesh `{}`",
        PLACE_OPERATION_NAME,
        .device.id(),
        .mesh.devices().iter().map(|device| device.id()).collect::<Vec<_>>(),
        .mesh.logical_mesh(),
    )]
    DeviceOutsideProgramMesh { device: Device, mesh: DeviceMesh },

    #[error(
        "`{}` cannot call a function compiled for the devices {:?} of mesh `{}` inside a program compiled for the \
         devices {:?} of mesh `{}`",
        JIT_CALL_OPERATION_NAME,
        .callee.devices().iter().map(|device| device.id()).collect::<Vec<_>>(),
        .callee.logical_mesh(),
        .mesh.devices().iter().map(|device| device.id()).collect::<Vec<_>>(),
        .mesh.logical_mesh(),
    )]
    CallOutsideProgramMesh { callee: DeviceMesh, mesh: DeviceMesh },

    #[error(
        "the sharding of {kind} {index} splits dimension {dimension} of extent {extent} into {shard_count} shards, but \
         XLA requires the number of shards to divide the extent"
    )]
    UnevenSharding { kind: &'static str, index: usize, dimension: usize, extent: usize, shard_count: usize },

    /// A valid operation requests behavior this backend cannot lower.
    #[error("cannot lower `{operation}`: {reason}")]
    UnsupportedOperation { operation: String, reason: String },

    /// A valid element type cannot be represented in the selected lowering context.
    #[error("cannot lower element type `{data_type}` for {context}")]
    UnsupportedElementType { data_type: DataType, context: String },

    #[error("programs compiled for platform `{platform}` cannot take or return values in `{memory}` memory")]
    UnsupportedBoundaryMemory { memory: Memory, platform: String },

    /// The constructed MLIR module failed verification without returning a native error value.
    #[error("constructed MLIR module failed verification during XLA lowering")]
    VerificationFailure,
}

pub(crate) mod admission;
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
    stable_hlo: String,

    /// Signature of the entry function, which maps its logical inputs and outputs to its physical arguments and
    /// results.
    signature: ExecutableSignature,

    /// Whether execution requires the host runtime assertion handler.
    requires_assertion_handler: bool,

    /// Canonical runtime contracts required by emitted calls and synthesized checks.
    runtime_requirements: Arc<[RuntimeRequirement]>,

    /// Module mesh that every sharding and `shard_map` boundary of the module refers to, or `None` when the module
    /// declares no mesh. Compilation checks it against the topology of the device mesh that it targets.
    mesh: Option<LogicalMesh>,
}

impl LoweredModule {
    /// Returns the rendered StableHLO module of this lowering, consuming it.
    #[inline]
    pub(crate) fn into_stable_hlo(self) -> String {
        self.stable_hlo
    }

    /// Returns the signature of the entry function, which maps its logical inputs and outputs to its physical arguments
    /// and results.
    #[inline]
    pub(crate) fn signature(&self) -> &ExecutableSignature {
        &self.signature
    }

    /// Returns the module mesh that every sharding and `shard_map` boundary of the module refers to, or `None` when the
    /// module declares no mesh. Compilation checks it against the topology of the device mesh that it targets.
    #[inline]
    pub(crate) fn mesh(&self) -> Option<&LogicalMesh> {
        self.mesh.as_ref()
    }

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
/// no token slot. Region input guards are the exception, because input validation must precede every operation in the
/// invoked region, so their lowering joins the assertion chain into the ordered I/O chain (refer to
/// [`lower_guard`](crate::operations::assertions::lower_guard)).
#[derive(Copy, Clone, Default)]
pub(crate) struct EffectTokens<'b, 'c: 'b, 't: 'c> {
    /// Current ordered assertion token, created lazily by the first runtime assertion in this scope.
    ordered_assertion: Option<ValueRef<'b, 'c, 't>>,

    /// Current ordered I/O token, created lazily by the first ordered I/O operation in this scope.
    ordered_io: Option<ValueRef<'b, 'c, 't>>,
}

impl<'b, 'c: 'b, 't: 'c> EffectTokens<'b, 'c, 't> {
    /// Returns the current token for `effect_class`. Both ordered I/O classes share the ordered I/O slot, because they
    /// form one ordering domain on any given device and differ only in whether that order also spans devices.
    pub(crate) fn get(&self, effect_class: EffectClass) -> Option<ValueRef<'b, 'c, 't>> {
        match effect_class {
            EffectClass::OrderedState => unreachable!("ordered state effects must be discharged before XLA lowering"),
            EffectClass::OrderedAssertion => self.ordered_assertion,
            EffectClass::OrderedIo | EffectClass::DeviceOrderedIo => self.ordered_io,
            EffectClass::UnorderedIo => None,
        }
    }

    /// Returns the current ordered assertion token, created lazily by the first runtime assertion in this scope.
    #[inline]
    pub(crate) fn ordered_assertion(&self) -> Option<ValueRef<'b, 'c, 't>> {
        self.ordered_assertion
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

/// Returns `true` if any operation in a region of `program` that lowering emits (i.e., its entry region and the
/// executed nested regions of that region) carries unresolved ordered state.
///
/// This scans those regions rather than consulting the effect summary of the program, because ordinary XLA cannot
/// preserve state in any lowered region, including one whose effects the summary of its owner does not expose.
/// Regions that lowering never reaches (e.g., the rule regions of composites or the transpose region of a
/// `linear_call`) are not part of the lowered module, so state in them is never lowered and cannot reject the program.
pub(crate) fn contains_unresolved_state<ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    program: &Program<ProgramInput, ProgramOutput, O>,
) -> bool
where
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
{
    program.entry_region_ref().computation_regions().any(|region| {
        region
            .instructions()
            .iter()
            .any(|instruction| instruction.operation().effects().classes().contains(EffectClass::OrderedState))
    })
}

/// Returns `true` if `program` contains a reference-typed atom or a [`CoreOperation`](ryft_core::Operation) whose
/// [effects](ryft_core::Operation::effects) declare a [`ReferenceEffect`](ryft_core::ReferenceEffect) in a region that
/// lowering emits (i.e., its entry region and the executed nested regions of that region). Regions that lowering never
/// reaches (e.g., the rule regions of composites or the transpose region of a `linear_call`) are not part of the
/// lowered module, so references in them neither need discharge nor reject the program.
///
/// This check is independent from [`contains_unresolved_state`], because a pure reference pass-through or a forwarded
/// reference capture can carry reference values without executing a stateful instruction. Both predicates back the
/// state checks that precede compilation and the two direct module lowering entry points. Eager binding, dispatch, and
/// staging enforce their corresponding boundary invariants separately.
pub(crate) fn contains_unresolved_references<ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    program: &Program<ProgramInput, ProgramOutput, O>,
) -> bool
where
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
{
    // The atom scan dominates for every well-formed program, because reference operations always touch an atom of a
    // reference type. The declaration scan is an independent check that still catches a core discharge bug which
    // retyped the boundary atoms but left a reference operation behind.
    program.entry_region_ref().computation_regions().any(|region| {
        region.atoms().iter().any(|atom| atom.r#type().is_reference())
            || region
                .instructions()
                .iter()
                .any(|instruction| instruction.operation().effects().has_reference_effects())
    })
}

/// Returns the effect classes that have StableHLO token slots, in canonical token and result order.
///
/// Token slots are an XLA/StableHLO representation decision, so the classification lives here rather than on the core
/// [`EffectClass`] type. The classification tests for each ordered class explicitly with [`EffectClasses::contains`],
/// so a newly added [`EffectClass`] gets no token slot here until it is listed. Only [`EffectTokens::get`] and
/// [`EffectTokens::set`] match exhaustively and force an explicit decision for a new class.
/// [`EffectClass::OrderedState`] has no slot, because ordinary XLA lowering rejects unresolved state at its module
/// entry boundaries, and no defensive path may accidentally turn state into an ordinary token-threaded effect.
pub(crate) fn token_threaded_effects(effects: EffectClasses) -> impl Iterator<Item = EffectClass> {
    // Each yielded class names one token slot of `EffectTokens`. Both ordered I/O classes share the ordered I/O slot,
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
/// enclosing lowering scope. The lowering entry points construct one lowerer per instruction. They copy the effect
/// tokens of the scope in and read the updated tokens back out after the instruction lowers, so effectful
/// operations thread their tokens through [`Self::effect_token`] and [`Self::set_effect_token`] without knowing the
/// enclosing scope. The lifetimes are those of the MLIR block (`'b`), context (`'c`), and thread pool (`'t`).
pub struct Lowerer<'b, 'c: 'b, 't: 'c, O = Operation> {
    /// Owning block receiving the lowered operations.
    block: BlockRef<'b, 'c, 't>,

    /// MLIR context owning the block and created operations.
    context: &'c MlirContext<'t>,

    /// Shared MLIR location used for emitted operations.
    location: LocationRef<'c, 't>,

    /// Attached regions of the instruction currently being lowered, materialized as flat programs in region-slot order.
    regions: Vec<FlatProgram<O>>,

    /// Shared private functions emitted for deduplicated `jit_call` callees, consulted at `jit_call` lowering sites.
    /// Shared via [`Rc`] so it threads through nested lowering scopes without lifetime entanglement.
    nested_functions: Option<Rc<JitCallFunctionMap<O>>>,

    /// Hidden capture arguments of the function currently being lowered, in capture-table order.
    captured_values: Vec<ValueRef<'b, 'c, 't>>,

    /// Current per-class effect tokens of the lowering scope this lowerer emits into.
    effect_tokens: EffectTokens<'b, 'c, 't>,

    /// Collective lowering state of the lowering scope this lowerer emits into. Refer to the documentation of
    /// [`CollectiveLoweringState`] for more information.
    collective_state: CollectiveLoweringState,
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

    /// Returns the shared private functions emitted for deduplicated `jit_call` callees, consulted at `jit_call`
    /// lowering sites. Shared via [`Rc`] so it threads through nested lowering scopes without lifetime entanglement.
    #[inline]
    pub(crate) fn nested_functions(&self) -> Option<&Rc<JitCallFunctionMap<O>>> {
        self.nested_functions.as_ref()
    }

    /// Returns the current per-class effect tokens of the lowering scope this lowerer emits into.
    #[inline]
    pub(crate) fn effect_tokens(&self) -> &EffectTokens<'b, 'c, 't> {
        &self.effect_tokens
    }

    /// Returns a mutable reference to the current per-class effect tokens of the lowering scope this lowerer emits into.
    #[inline]
    pub(crate) fn effect_tokens_mut(&mut self) -> &mut EffectTokens<'b, 'c, 't> {
        &mut self.effect_tokens
    }

    /// Returns the collective lowering state of the lowering scope this lowerer emits into. Refer to the documentation
    /// of [`CollectiveLoweringState`] for more information.
    #[inline]
    pub(crate) fn collective_state(&self) -> &CollectiveLoweringState {
        &self.collective_state
    }

    /// Returns the effect tokens of the lowering scope mutably, together with the attached regions, the captured
    /// values, the shared `jit_call` functions, and the collective lowering state of this lowerer, in that order.
    /// Lowering helpers thread the tokens while they read the rest of that state, which the separate accessors cannot
    /// provide, because the mutable borrow of [`Self::effect_tokens_mut`] covers the whole lowerer.
    #[allow(clippy::type_complexity)]
    pub(crate) fn split_effect_tokens(
        &mut self,
    ) -> (
        &mut EffectTokens<'b, 'c, 't>,
        &[FlatProgram<O>],
        &[ValueRef<'b, 'c, 't>],
        Option<&Rc<JitCallFunctionMap<O>>>,
        &CollectiveLoweringState,
    ) {
        (
            &mut self.effect_tokens,
            &self.regions,
            &self.captured_values,
            self.nested_functions.as_ref(),
            &self.collective_state,
        )
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
    /// Returns an [`Error`] when the `stablehlo.after_all` operation cannot be created.
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
/// contract. Homogeneous array operations implement it at [`ArrayType`], array IR operations at [`Type`], and
/// first-class dimension operations at [`DimensionType`]. Operation families (e.g., [`ArrayOperation`] and
/// [`Operation`]) implement this trait by dispatching each variant to its payload through
/// [`lower_operation_variants`](crate::lower_operation_variants), which also handles payloads that a family stores as
/// projected or mixed members. A family whose type universe is [`Type`] and that implements
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
    fn composite_key(&self) -> Option<&CompositeKey> {
        None
    }
}

/// Operation families whose programs the XLA lowering entry points lower. Such a family lowers at [`Type`],
/// and its payloads can be recognized through [`OperationPayloadProjection`] by the module-level lowering passes
/// (e.g., the `jit_call` deduplication and the Shardy mesh declaration passes). Every type that satisfies these bounds
/// is an operation family, so downstream families only need to implement [`LowerableOperation`].
pub trait LowerableOperationFamily: LowerableOperation<Type = Type> + OperationPayloadProjection {}

impl<O: LowerableOperation<Type = Type> + OperationPayloadProjection> LowerableOperationFamily for O {}

/// Operations whose own type universe is a member universe of an enclosing operation family that lowers them at the
/// family's parent universe `T` (e.g., [`ZeroOperation<ArrayType>`](ryft_core::ZeroOperation) stored in an
/// [`Type`] family receives first-class dimension inputs that size its dynamic output axes).
///
/// This is the lowering counterpart of [`MemberOperation`](ryft_core::MemberOperation), and families dispatch their
/// mixed member variants to it through [`lower_operation_variants`](crate::lower_operation_variants).
pub trait MemberLowerableOperation<T: CoreType>: CoreOperation {
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
    U: CoreType,
    P: LowerableOperation<Type = U>,
    for<'o> &'o U: TryFrom<&'o T, Error: Into<ProgramError>>,
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
///     stores an [`ArrayOperation`] in an [`Type`] family).
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
///         input_types: &[Type],
///         output_types: &[Type],
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
        $crate::lowering::lower_projected_operation::<$universe, _, _, _>(
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
/// artificial order across classes. The enclosing lowering helpers carry active chains through nested computations.
/// The final tokens are intentionally omitted from the public function results, because their custom calls are marked
/// as having side effects and the tokens exist only to encode the ordering within one execution.
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
    let created = created.result(0).unwrap().as_ref();
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
            Zero, ZeroLike, One, OneLike, Constant, FillLike, Iota, IotaLike,
            Min, Max, Clamp, Sign, Neg, Add, Sub, SaturatingSub,
            Mul, Div, Rem, Pow, Sqrt, Rsqrt, Abs, Sin, Cos, Tan, Tanh, Atan2, Exp, ExpM1, Log, Ln1p, LogAddExp, Logistic, Erf,
            Floor, Ceil, Round, Not,
            And, Or, Xor, Complex, Real, Imaginary, Conjugate, Reduce, ArgMax, ArgMin, Cumulative, Sort, Dot, RaggedDot,
            ScaledDot, AxisIndex, ParallelReduce, ParallelVary, ParallelCast,
            ParallelAllGather, ParallelSumScatter, ParallelPermute, ParallelBroadcast, ParallelAllToAll,
            ParallelRaggedAllToAll, Compare, Select, Transpose, Reverse, Reshape, Broadcast, BroadcastLike, Pad,
            Concatenate, Gather, Scatter, Slice, UpdateSlice,
            DynamicSlice, DynamicUpdateSlice, ConvertElementType, ReducePrecision, TransferToMemory, Place, Reshard,
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
    fn composite_key(&self) -> Option<&CompositeKey> {
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
        input_types: &[Type],
        output_types: &[Type],
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
    channel_ids: Rc<Cell<usize>>,

    /// Enclosing manual regions, ordered from outermost to innermost for named-axis resolution.
    manual_regions: Vec<Rc<ShardMapBoundary>>,

    /// Axes bound by every enclosing manual region, excluded from shard-local placement constraints.
    bound_manual_axes: Vec<String>,

    /// PJRT platform name of the compilation target (e.g., `"cuda"` or `"cpu"`), or `None` when the lowering has no
    /// target information. Lowerings that depend on the platform, such as fused attention, consult this and fall back
    /// to their portable form when it is absent.
    target_platform: Option<Rc<str>>,

    /// Backend representation used to lower ragged dots in this module.
    ragged_dot_lowering_strategy: RaggedDotLoweringStrategy,

    /// Module-owned private decompositions used by named StableHLO composites.
    named_compositions: Option<Rc<NamedCompositionFunctionMap>>,

    /// Flag of the module recording whether any recursive lowering path constructed a location specific to an
    /// instruction (i.e., not the base location) from a known [`Provenance`]. It selects the module serialization mode
    /// after lowering completes. Only a module that actually carries provenance locations is printed with debug
    /// information, so modules without provenance keep their existing StableHLO text and cache keys byte for byte.
    has_provenance: Rc<Cell<bool>>,

    /// Whether any emitted instruction requires the runtime assertion handler.
    has_assertions: Rc<Cell<bool>>,

    /// Runtime contracts declared by emitted operations across every nested lowering scope.
    runtime_requirements: Rc<RefCell<Vec<RuntimeRequirement>>>,

    /// Distinct [`RetainedArray`]s of the module, in the order of the hidden entry arguments that carry them (refer
    /// to [`collect_session_constants`]). Their values form the suffix of every `captured_values` table that can reach
    /// the entry function's arguments. Scopes that cannot reach them (i.e., `sdy.manual_computation` bodies) carry no
    /// session constants.
    session_constants: Rc<[RetainedArray]>,
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
    pub(crate) fn with_session_constants(mut self, session_constants: Rc<[RetainedArray]>) -> Self {
        self.session_constants = session_constants;
        self
    }

    /// Returns a copy of this state carrying the PJRT platform name of the compilation target.
    pub(crate) fn with_target_platform(mut self, target_platform: Option<&str>) -> Self {
        self.target_platform = target_platform.map(Rc::from);
        self
    }

    /// Returns a copy of this state carrying the [`RaggedDotLoweringStrategy`] of the module.
    pub(crate) fn with_ragged_dot_lowering_strategy(mut self, strategy: RaggedDotLoweringStrategy) -> Self {
        self.ragged_dot_lowering_strategy = strategy;
        self
    }

    /// Returns a copy of this state carrying the registry of the private decompositions of the named compositions of
    /// the module.
    fn with_named_compositions(mut self, named_compositions: Rc<NamedCompositionFunctionMap>) -> Self {
        self.named_compositions = Some(named_compositions);
        self
    }

    /// Returns the PJRT platform name of the compilation target, or `None` when the lowering has no target
    /// information.
    pub(crate) fn target_platform(&self) -> Option<&str> {
        self.target_platform.as_deref()
    }

    /// Returns the ragged dot lowering strategy for this module.
    pub(crate) fn ragged_dot_lowering_strategy(&self) -> RaggedDotLoweringStrategy {
        self.ragged_dot_lowering_strategy
    }

    /// Returns the suffix of `captured_values` that holds the values of the session constants reachable in this scope.
    /// A scope whose `captured_values` table cannot hold them (e.g., a shared private function) has no such values.
    pub(crate) fn session_values<'o, V>(&self, captured_values: &'o [V]) -> &'o [V] {
        captured_values
            .len()
            .checked_sub(self.session_constants.len())
            .map_or(&[], |start| &captured_values[start..])
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

    /// Returns `true` if any recursive lowering path constructed a location specific to an instruction from a known
    /// [`Provenance`].
    pub(crate) fn has_provenance(&self) -> bool {
        self.has_provenance.get()
    }

    /// Returns the axes bound by every enclosing manual region, excluded from shard-local placement constraints.
    #[inline]
    pub(crate) fn bound_manual_axes(&self) -> &[String] {
        &self.bound_manual_axes
    }

    /// Returns the module-owned private decompositions used by named StableHLO composites.
    #[inline]
    pub(crate) fn named_compositions(&self) -> Option<&Rc<NamedCompositionFunctionMap>> {
        self.named_compositions.as_ref()
    }

    /// Returns a fresh module-unique channel id for one channeled collective.
    pub(crate) fn next_channel_id(&self) -> usize {
        let channel_id = self.channel_ids.get();
        self.channel_ids.set(channel_id + 1);
        channel_id
    }

    /// Derives the state used inside the body of one `sdy.manual_computation` region. The target platform, ragged dot
    /// lowering strategy, named compositions, provenance state, and module channel allocator are inherited, while the
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

    /// Returns the value of `constant` in a scope whose captured values are `captured_values`.
    pub(crate) fn lower_session_constant<'b, 'c: 'b, 't: 'c>(
        &self,
        constant: &RetainedArray,
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

    /// Resolves a collective's actual manual binder and checks any payload mesh against it before communication, and
    /// returns the mesh over which the collective's types are expressed. A payload mesh is accepted when it has the
    /// binder mesh's topology and makes the same axes manual (i.e., when [`LogicalMesh::has_same_manual_axes`] holds),
    /// because `auto_axes` and `explicit_axes` retype only the free axes of a manual region, and it is then the
    /// returned mesh. A payload without a mesh yields the binder mesh and is valid only when the owning operation's
    /// inference accepts that binder. Both meshes share their topology, so either one determines axis sizes and replica
    /// groups.
    pub(crate) fn checked_collective_mesh<'o, A: Into<CollectiveAxes>>(
        &'o self,
        operation_name: &str,
        axis_name: A,
        axis_size: usize,
        payload_mesh: Option<&'o LogicalMesh>,
    ) -> Result<&'o LogicalMesh, Error> {
        // Resolve the binding and recorded size first, preserving the replica-group diagnostics and the nearest
        // enclosing binder when another manual region shadows the axis name. That resolution also requires every named
        // axis to be bound on one mesh.
        let axis_name = axis_name.into();
        collective_replica_groups(self, &axis_name, axis_size, None)?;
        let mesh = self.manual_axis_region(&axis_name.names()[0]).unwrap().mesh();
        match payload_mesh {
            Some(payload_mesh) if payload_mesh.has_same_manual_axes(mesh) => Ok(payload_mesh),
            Some(_) => Err(ProgramError::MalformedProgram(format!(
                "`{operation_name}` operation mesh does not match the enclosing manual axis `{axis_name}` mesh",
            ))
            .into()),
            None => Ok(mesh),
        }
    }

    /// Revalidates a collective against the policy of its actual enclosing binder before it emits cross-device
    /// communication. A checked binder passes the mesh that [`Self::checked_collective_mesh`] returns to
    /// `infer_output_types`. That is the payload mesh when the payload records one, because it may retype the free axes
    /// of the binder mesh (e.g., inside `auto_axes`) and the recorded `output_types` are expressed over it. Otherwise
    /// it is the binder mesh, so a mesh-free payload cannot bypass the binder's variation and pending sum contract. An
    /// unchecked binder requires an ordinary mesh-free payload and passes `None`, preserving ordinary collective
    /// inference and adjoints. In either mode, inference must reproduce the recorded `output_types`, and the physical
    /// mesh and participant count remain checked.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if no enclosing manual region binds `axis_name`, if the recorded axis size disagrees with
    /// that region, if a payload mesh differs from that region's mesh in its topology or in the axes it makes manual,
    /// if a tracked payload is used in an unchecked binding, if re-inference fails, or if it produces different output
    /// types.
    pub(crate) fn validate_manual_collective<T: PartialEq, A: Into<CollectiveAxes>>(
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
        let checked = self.manual_axis_region(&axis_name.names()[0]).unwrap().checks_manual_variation();
        if axis_name
            .names()
            .iter()
            .any(|axis_name| self.manual_axis_region(axis_name).unwrap().checks_manual_variation() != checked)
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
}

/// Serializes one lowered module to StableHLO text, printing MLIR debug information (i.e., locations) exactly when
/// the lowering constructed provenance locations specific to instructions. Modules without provenance keep the plain
/// rendering, so their StableHLO text, snapshots, and compilation cache keys remain byte for byte what they were before
/// provenance existed, while modules that carry provenance embed their locations in the text and therefore in the
/// persistent compilation key derived from it. The pretty debug form is documented as unparsable and never used, and
/// the elision thresholds are disabled because the plain rendering performs no elision either.
///
/// Exact backend provenance can reduce the reuse of cached compiled artifacts, because two semantically equal modules
/// with different provenance serialize differently and therefore have different keys. If that cost ever matters, the
/// correct extension is an explicit compilation option that strips provenance before lowering, which makes this helper
/// fall back to the plain rendering. Silently returning an artifact labeled with the provenance of another program is
/// never acceptable.
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
/// attached to each function argument or result, which drives per-device boundary slicing (e.g., for uneven splits) in
/// the XLA SPMD partitioner. When they are `None`, the corresponding values carry no sharding attributes, and Shardy
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
pub fn to_mlir_module_for_program<Input, Output, ProgramInput, ProgramOutput, S, O: LowerableOperationFamily>(
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
        true,
        &[],
        &[],
        RaggedDotLoweringStrategy::default(),
        &[],
        false,
    )
}

/// Lowers an arbitrary traced XLA program like [`lower_mlir_module_for_program`], but with one optional sharding per
/// entry function argument (i.e., captures first) and per result, so that only the values whose sharding is known carry
/// an `sdy.sharding` attribute, while Shardy propagates the placement of the others from the program. The results
/// flagged in `open_results` (i.e., one flag per result, or none at all) are lowered open over auto mesh axes, so that
/// the compiler chooses their placement over those axes (refer to [`entry_open_results`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_mlir_module_for_program_with_value_shardings<
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
    open_results: &[bool],
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
        true,
        &[],
        &[],
        RaggedDotLoweringStrategy::default(),
        open_results,
        false,
    )
}

/// Lowers a discharged program with logical external reference state aliases on its entry boundary. The results
/// flagged in `open_results` (i.e., one flag per logical output, or none at all) are lowered open over auto mesh axes,
/// so that the compiler chooses their placement over those axes.
///
/// `reference_states` are the external state recipes of the discharge artifact, and each must name a distinct logical
/// state input. Each state input must survive the executable boundary. A mutated state must be a static device array
/// and name a hidden output of the identical type and sharding. Mutated pairs are recorded as `tf.aliasing_output`
/// aliases between inputs and outputs, which are non-semantic hints that the buffers may alias and that merely permit
/// backend buffer reuse. The runtime never donates reference state inputs, so an alias never authorizes in-place
/// mutation of the state buffer of the caller. Read-only state may be finite and bounded dynamic because it has no
/// hidden alias. The validation here owns the physical alias contract, and the XLA session additionally enforces the
/// supported memory, sharding, and dynamic shape classes before calling it.
///
/// `donated_inputs` flags the public inputs whose buffers the caller gives up to every execution (i.e., one flag per
/// public input, or none at all). Each donated input is paired with an output that it can share its buffers with and
/// the pair is recorded as the same kind of alias, as JAX does for the donated arguments of a `jit` function. An
/// output can share the buffers of an input when both are static device arrays of the identical type and sharding,
/// and when the output is not open on a mesh of more than one device, because the compiler could then place it
/// differently than the input. Without such an alias the outputs of every execution are written to new buffers,
/// which is correct but forces backends that cache launch state per buffer address (e.g., the command buffers of
/// the XLA GPU backend) to rebuild that state for every execution of a program that updates its inputs in a loop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_mlir_module_for_program_with_reference_state<
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
    supports_shardy: bool,
    reference_states: &[ExternalReferenceBinding],
    donated_inputs: &[bool],
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
        supports_shardy,
        reference_states,
        donated_inputs,
        ragged_dot_lowering_strategy,
        open_results,
        true,
    )
}

/// Returns the distinct [`RetainedArray`]s that `program` references in the regions that lowering emits (i.e., its
/// entry region and the executed nested regions of that region), in the order of their first occurrence in those
/// regions. Lowering passes their arrays as hidden entry arguments that follow the lifted captures, in this order, and
/// so the caller that assembles the executable arguments must use the same order. Regions that lowering never reaches
/// (e.g., the rule regions of composites or the transpose region of a `linear_call`) are skipped, so that an array
/// that only they refer to does not become an argument of the executable.
pub(crate) fn collect_session_constants<ProgramInput, ProgramOutput, O: CoreOperation<Type = Type>>(
    program: &Program<ProgramInput, ProgramOutput, O>,
) -> Vec<RetainedArray>
where
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
{
    let mut session_constants = Vec::new();
    for region in program.entry_region_ref().computation_regions() {
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

/// Returns whether `program` requests the placement of a value explicitly in any of the regions that lowering emits,
/// through a `place` with a target, a `constrain_sharding`, or a `jit_call` whose callee names devices (i.e., carries a
/// sharding override of its boundary or requests a placement itself, refer to [`JitCallOperation::with_boundary`]),
/// whether or not that value reaches a result. These are the analogues of the `device_put` and
/// `with_sharding_constraint` calls and of the shardings of nested `jit` calls that make JAX commit every output of a
/// `jit` function (refer to `get_intermediate_shardings` in `jax._src.dispatch`). Such calls count because their
/// callees cannot show their overrides (e.g., a replicated override over auto mesh axes stages nothing). The callees of
/// calls are among the regions that lowering emits, so a request inside a callee counts too. A `place` without a target
/// leaves its input where it is, so it requests nothing. A `place` with a device target (i.e., a
/// [`PlacementRequest::Device`]) is counted, as an explicit device commits an array eagerly, although JAX leaves a
/// value that a bare device places inside `jit` uncommitted and on the devices that it already occupies. A `place` with
/// an in-place target (i.e., a [`PlacementRequest::InPlace`]) is counted too, because JAX commits every output of a
/// `jit` function that moves a value to a memory space (i.e., `jax.device_put(x, jax.memory.Space.Host)`), whose
/// destinations `jaxpr_transfer_mem_kinds` in `jax._src.interpreters.pxla` collects and `lower_sharding_computation`
/// counts among the reasons to commit. A `reshard` is not counted, because JAX does not count its own `reshard` among
/// these requests either.
///
/// With `includes_memory_requests` unset, an in-place target is not counted, because it names no devices. This is how
/// a compiled function decides whether it names the devices of its mesh (refer to `requests_placement` in
/// `crate::compilation::domains`), since JAX runs a `jit` function whose body only moves a value to another memory
/// (e.g., `jax.device_put(x, jax.memory.Space.Host)`) where its arguments are, directly and nested in another `jit`
/// function alike, and commits its outputs there.
pub(crate) fn has_placement_requests<ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    program: &Program<ProgramInput, ProgramOutput, O>,
    includes_memory_requests: bool,
) -> bool
where
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
{
    has_placement_requests_in_region(program.entry_region_ref(), includes_memory_requests)
}

/// Returns whether `region` requests the placement of a value explicitly in any of the regions that lowering emits for
/// it (refer to [`has_placement_requests`]).
pub(crate) fn has_placement_requests_in_region<V: Value<Type = Type> + Typed, O: LowerableOperationFamily>(
    region: RegionRef<'_, V, O>,
    includes_memory_requests: bool,
) -> bool {
    region.computation_regions().any(|region| {
        region.instructions().iter().any(|instruction| {
            let operation = instruction.operation();
            operation.projected_payload::<PlaceOperation>().is_some_and(|operation| match operation.target() {
                Some(PlacementRequest::InPlace { .. }) => includes_memory_requests,
                target => target.is_some(),
            }) || operation.projected_payload::<ConstrainShardingOperation>().is_some()
                || operation.projected_payload::<JitCallOperation<Type>>().is_some_and(|call| call.mesh().is_some())
        })
    })
}

/// Placement that the entry region of a program gives one of its results, as [`entry_result_placements`] returns it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum EntryResultPlacement {
    /// The result is produced by a `place` whose target has this sharding.
    Target(Sharding),

    /// The result is the entry argument at this index, which it returns where that argument enters the program.
    Argument(usize),
}

/// Returns the placement that `program` itself gives each of its first `output_count` results, or [`None`] for a
/// result that it gives no placement. A result that a `place` in the entry region produces has the placement of its
/// target, which states the placement along every mesh axis, including the auto mesh axes that the result type drops.
/// Only placements in the entry region and in the callees of its `jit_call`s count, because a placement in another
/// nested region (e.g., a branch of a `condition` or the body of a loop) reaches a result only through the result of
/// the operation that owns that region, which the compiler is free to place differently. A result that is an entry
/// argument is placed where that argument enters the program, which the caller knows and which also covers arguments
/// without buffers (i.e., of the zero data type), whose placement the compiler cannot report. A result that returns a
/// [`RetainedArray`] is likewise placed as that array, which enters the program as a hidden argument.
///
/// Operations that return the value of an input together with its placement are followed back to that input, so that a
/// result keeps the placement of a `place` or of an entry argument that precedes them. These are a `place` without a
/// target, which leaves its input where it is, a `place` with a device target, which keeps the sharding of its input
/// on the only device of the program mesh, `transfer_to_memory`, which only changes the memory of its input as a
/// `place` with an in-place target does, and `stop_gradient` and `tag`, which only mark values for program transforms
/// and lower to nothing. A device target that completes a pending reduction over mesh axes of size one
/// gives its result another sharding than its input, so such a result is placed as its type states instead. A
/// `jit_call` always runs its callee, whose results are the results of the call, so a result of a call has the
/// placement that the callee gives the corresponding callee result, and a callee result that returns a callee input is
/// followed back to the corresponding input of the call. The boundary of the callee takes precedence, because the
/// callee body cannot always show it, so a result of the call that a sharding override of a callee output covers is
/// placed as that override, and one that returns a callee input with an effective boundary sharding (i.e., its
/// override or otherwise where the executable of the callee places it, refer to
/// [`JitCallOperation::boundary_input_sharding`]) is placed as that sharding. Lowering constrains that result or
/// operand of the call to that sharding as well, so that the call places it alike whether a program returns it or
/// consumes it (refer to the lowering of [`JitCallOperation`]). Other operations that return a value equal to an input
/// are not followed, because they change its placement (e.g., `reshard` and `constrain_sharding`) or because the
/// compiler is free to place their results differently (e.g., the `optimization_barrier` of rematerialization).
pub(crate) fn entry_result_placements<ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    program: &Program<ProgramInput, ProgramOutput, O>,
    output_count: usize,
) -> Vec<Option<EntryResultPlacement>>
where
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
{
    let mut callee_placements = HashMap::new();
    region_result_placements(program.entry_region_ref(), &mut callee_placements)[..output_count].to_vec()
}

/// Returns the sharding with which the entry function of a program annotates a value at its boundary (i.e., an
/// argument or a result) whose runtime array is placed as `sharding`. This is `sharding` with its unconstrained
/// dimensions replicated. A runtime array stores such a dimension replicated, because its placement reads an
/// unconstrained dimension as replicated, so an annotation that left the dimension open would let the compiler shard
/// it and exchange buffers of other shapes than those of the array (e.g., expect `[2, 2]` argument buffers where an
/// array placed as `[{"x"}, {?}]` supplies `[2, 4]` buffers).
pub(crate) fn entry_boundary_sharding(sharding: &Sharding) -> Result<Sharding, Error> {
    if !sharding.dimensions().contains(&ShardingDimension::Unconstrained) {
        return Ok(sharding.clone());
    }
    let dimensions = sharding
        .dimensions()
        .iter()
        .map(|dimension| match dimension {
            ShardingDimension::Unconstrained => ShardingDimension::Replicated,
            dimension => dimension.clone(),
        })
        .collect();
    Ok(sharding.with_dimensions(dimensions)?)
}

/// Returns the sharding with which the entry function of a program annotates each of its results, whose types are
/// `output_types`, and which describes the buffers that the results are returned in. These are the `result_shardings`
/// overrides when they are provided, and otherwise the placement that the program gives each result (i.e., the
/// corresponding element of `placements`, as [`entry_result_placements`] returns them, where an entry argument is
/// placed as the corresponding element of `argument_shardings`), or the sharding of the result type when the program
/// gives it none (e.g., when that argument has no sharding). A result without either has no sharding. Compilation for
/// a [`Session`](crate::Session) and the StableHLO export of a [`TracedProgram`](crate::TracedProgram) both choose the
/// shardings of results with this function, so that an exported module places its results as the executable that a
/// session compiles for it does.
///
/// An override, a `place` target, an argument placement, or a result type may leave dimensions unconstrained, but the
/// runtime placement of an array reads such a dimension as replicated, and the result type records it exactly, so it
/// cannot record the placement that the compiler would choose for it. These dimensions are therefore replicated here
/// (refer to [`entry_boundary_sharding`]), whatever operation produces the result (e.g., arithmetic whose result type
/// inherits an unconstrained dimension from the target of a `place` that precedes it), so that the executable returns
/// the buffers that the runtime placement of the result describes, while the dimensions that they constrain keep their
/// placement. Provided overrides are not checked against `output_types`, so that a count mismatch is still reported
/// when the module is lowered.
pub(crate) fn entry_result_shardings(
    output_types: &[ArrayType],
    result_shardings: Option<&[Sharding]>,
    placements: &[Option<EntryResultPlacement>],
    argument_shardings: &[Option<Sharding>],
) -> Result<Vec<Option<Sharding>>, Error> {
    match result_shardings {
        Some(shardings) => shardings.iter().map(|sharding| entry_boundary_sharding(sharding).map(Some)).collect(),
        None => output_types
            .iter()
            .zip(placements)
            .map(|(output_type, placement)| {
                let placement = match placement {
                    Some(EntryResultPlacement::Target(sharding)) => Some(sharding),
                    Some(EntryResultPlacement::Argument(index)) => {
                        argument_shardings.get(*index).and_then(Option::as_ref)
                    }
                    None => None,
                };
                placement.or(output_type.sharding()).map(entry_boundary_sharding).transpose()
            })
            .collect(),
    }
}

/// Returns whether the entry function of a program leaves each of its results, whose types are `output_types` and
/// whose annotations are `result_annotations` (i.e., the shardings that [`entry_result_shardings`] returns for
/// `result_shardings` and `placements`), open over the auto mesh axes of its annotation, so that the compiler chooses
/// the placement of the result along those axes, like JAX lowers the results of a `jit` function without
/// `out_shardings`. Compilation for a [`Session`](crate::Session) and the StableHLO export of a
/// [`TracedProgram`](crate::TracedProgram) both decide this with this function, so that an exported module leaves
/// open the results whose placement the compiler chooses for the executable that a session compiles for it.
///
/// A result is open when it has an annotation over a mesh with an auto axis larger than one (i.e., one along which the
/// compiler has a choice), and when nothing else fixes its placement. A result sharding override, a placement that the
/// program gives the result (i.e., that a `place` produces it or that it returns an argument), reduction state or
/// manual variation in the annotation, which the compiler must not change, and unconstrained dimensions in the result
/// type, which the runtime placement of the result must record as its type does and read as replicated, all fix it.
/// Results without an annotation are never open, because they have no annotation that could be.
///
/// Export lowers a program without a device mesh, so it can only decide what the program and its signature state.
/// It cannot know whether compilation runs the program partitioned, because a program with bounded dynamic shapes
/// runs as fully replicated replicas whose results are never open, nor which placements the compiler then chooses for
/// the open results, which only the compiled executable reports. It also cannot know the runtime placements of
/// arguments without signature shardings, whose types do not record auto mesh axes, so that their annotations and
/// those of the results that return them are left to Shardy.
pub(crate) fn entry_open_results(
    output_types: &[ArrayType],
    result_shardings: Option<&[Sharding]>,
    placements: &[Option<EntryResultPlacement>],
    result_annotations: &[Option<Sharding>],
) -> Vec<bool> {
    output_types
        .iter()
        .zip(placements)
        .zip(result_annotations)
        .map(|((output_type, placement), annotation)| {
            result_shardings.is_none()
                && placement.is_none()
                && annotation.as_ref().is_some_and(|sharding| {
                    has_untracked_auto_axes(sharding)
                        && sharding.unreduced_axes().is_empty()
                        && sharding.reduced_axes().is_empty()
                        && sharding.varying_manual_axes().is_empty()
                })
                && !output_type
                    .sharding()
                    .is_some_and(|sharding| sharding.dimensions().contains(&ShardingDimension::Unconstrained))
        })
        .collect()
}

/// Returns whether the mesh of `sharding` has an auto axis larger than one (i.e., one along which the compiler has a
/// choice), along which an array type never records a placement (refer to [`ArrayType::with_sharding`]), so that a
/// value with the type of `sharding` may be placed along it in more than one way.
pub(crate) fn has_untracked_auto_axes(sharding: &Sharding) -> bool {
    sharding.mesh().axes().iter().any(|axis| axis.r#type() == MeshAxisType::Auto && axis.size() > 1)
}

/// Value type that lowering can materialize as a StableHLO dense constant or bind to a captured argument (e.g., the
/// values of [`ArrayOperation`] constants).
pub(crate) trait MlirLowerableValue: 'static + Value<Type = ArrayType> {
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
        captured_values: &[ValueRef<'b, 'c, 't>],
        block: &mut B,
        context: &'c MlirContext<'t>,
        location: L,
    ) -> Result<ValueRef<'b, 'c, 't>, Error>
    where
        B: Block<'b, 'c, 't>,
        L: Copy + Location<'c, 't>,
    {
        let _ = captured_values;
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
                message: format!("session constant `{value}` cannot be materialized as a literal"),
            }
            .into()),
            Self::Boolean(value) => Array::scalar(*value)?.to_dense_elements_attribute(tensor_type, context),
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

// `ArrayType` is used as the value representation for abstract linear XLA programs. It can type program atoms, but it
// is not a concrete literal, so lowering paths that need a real value must supply it through captured arguments instead
// of materializing it from type metadata.
impl MlirLowerableValue for ArrayType {
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, Error> {
        let _ = (tensor_type, context);
        Err(ProgramError::Concretization {
            message: format!("abstract XLA value `{self}` cannot be materialized as a StableHLO literal"),
        }
        .into())
    }
}

// Concrete host literal lowering used by [`ConstantOperation`] and MLIR snapshot tooling.
impl MlirLowerableValue for Array {
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

/// Collects the module mesh from the meshes referenced by emitted sharding constraints and nested-computation
/// boundaries, starting from `existing`, and rejects every mesh whose topology differs from it with
/// [`LoweringError::MismatchedMeshTopology`]. The constraints include those of the boundary overrides of `jit_call`s,
/// whose shardings no type records. A constraint that this function does not recognize but that lowering emits over
/// the sharding of a type still finds its mesh, because the module declares the mesh of those types whenever the
/// lowered module refers to a mesh that this function did not find (refer to `declare_constrained_mesh`).
pub(crate) fn collect_nested_sharding_mesh<ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    program: &Program<ProgramInput, ProgramOutput, O>,
    existing: Option<LogicalMesh>,
) -> Result<Option<LogicalMesh>, Error>
where
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
{
    // Only the computation regions that lowering emits contribute, so dormant rule regions (e.g., the transpose region
    // of a `linear_call`) cannot bring other meshes into the module. Type metadata alone does not require a mesh,
    // because bounded dynamic replica lowering retains that metadata without emitting Shardy operations.
    let mut mesh = existing;
    for region in program.entry_region_ref().computation_regions() {
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
                })
                .or_else(|| {
                    // A device target names no mesh. It lowers to a sharding constraint only when it completes a
                    // reduction, over the mesh of its result type, which `declare_constrained_mesh` declares.
                    operation
                        .projected_payload::<PlaceOperation>()
                        .and_then(PlaceOperation::target)
                        .and_then(PlacementRequest::placement)
                        .map(|target| target.sharding().mesh())
                })
                .or_else(|| {
                    // A call constrains its operands and results that the boundary of its callee overrides.
                    operation
                        .projected_payload::<JitCallOperation<Type>>()
                        .and_then(|call| call.boundary_shardings().next())
                        .map(Sharding::mesh)
                });
            let incoming = if let Some(mesh) = declared_mesh {
                Some(mesh.clone())
            } else {
                let output_axes = operation
                    .projected_payload::<DynamicBroadcastOperation>()
                    .map(DynamicBroadcastOperation::output_axes)
                    .or_else(|| {
                        operation.projected_payload::<BroadcastOperation>().map(BroadcastOperation::output_axes)
                    })
                    .or_else(|| {
                        operation.projected_payload::<BroadcastLikeOperation>().map(BroadcastLikeOperation::output_axes)
                    });
                let emits_constraint = if let Some(output_axes) = output_axes {
                    let input_type = region.atoms()[instruction.inputs()[0].index()].r#type();
                    let input_type = <&ArrayType>::try_from(input_type.as_ref()).map_err(ProgramError::from)?;
                    let output_type = region.atoms()[instruction.outputs()[0].index()].r#type();
                    let output_type = <&ArrayType>::try_from(output_type.as_ref()).map_err(ProgramError::from)?;
                    broadcast_changes_explicit_sharding(input_type, output_type, output_axes)
                } else if operation.projected_payload::<CastOperation<Type>>().is_some()
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
                        || operation.projected_payload::<ZeroOperation<ArrayType>>().is_some()
                        || operation.projected_payload::<OneOperation<ArrayType>>().is_some()
                        || operation.projected_payload::<IotaOperation<ArrayType>>().is_some()
                        || operation.projected_payload::<ZeroLikeOperation<ArrayType>>().is_some()
                        || operation.projected_payload::<OneLikeOperation<ArrayType>>().is_some()
                        || operation.projected_payload::<FillLikeOperation<ArrayType>>().is_some()
                        || operation.projected_payload::<IotaLikeOperation<ArrayType>>().is_some()
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
                    Some(existing_mesh) => merge_logical_meshes(existing_mesh, &incoming)?,
                    None => incoming,
                });
            }
        }
    }
    Ok(mesh)
}

/// Removes every `stablehlo.constant` operation whose result has no uses from `module`, including those in nested
/// regions. Lowering materializes first-class dimension extents eagerly (refer to
/// [`lower_dimension_extent`](crate::lowering::types::lower_dimension_extent)), while their consumers
/// often read them only for dynamic axes (e.g., the explicit-extent collectives only refine dynamic result axes), so a
/// static extent would otherwise remain in the lowered module as a dead constant. Constants are pure, so removing an
/// unused one never changes the semantics of the module.
pub(crate) fn remove_unused_constants(module: &Module<'_, '_>) -> Result<(), Error> {
    remove_unused_constants_in_block(module.body()?)
}

/// Removes every Shardy operation from `module`, including those in nested regions, for a plugin that cannot compile
/// with Shardy. Each `sdy.sharding_constraint` is replaced by its operand and the `sdy.mesh` declaration is erased.
/// Ryft runs programs on such targets on a single device. Every sharding then holds the whole array, so removing
/// these constraints preserves the result.
///
/// # Errors
///
/// Returns [`LoweringError::UnsupportedOperation`] for any other Shardy operation (e.g., the `sdy.manual_computation`
/// of a `shard_map`), which has no equivalent without the dialect.
pub(crate) fn remove_shardy_operations(module: &Module<'_, '_>) -> Result<(), Error> {
    remove_shardy_operations_in_block(module.body()?)
}

/// Lowers one traced instruction to the corresponding StableHLO operations through its [`LowerableOperation`]
/// implementation and returns its output values. The ordered effect tokens of the enclosing scope flow through
/// `effect_tokens`, and an instruction that uses the assertion token marks the module as requiring the runtime
/// assertion handler.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_instruction<'b, 'c: 'b, 't: 'c, ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    program: &Program<ProgramInput, ProgramOutput, O>,
    instruction: &Instruction<O>,
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
    let outputs = lower_instruction_operation(
        instruction.operation(),
        &mut lowerer,
        input_values,
        input_types.as_slice(),
        output_types.as_slice(),
    )?;
    if lowerer.effect_tokens.ordered_assertion.is_some() {
        collective_state.has_assertions.set(true);
    }
    *effect_tokens = lowerer.effect_tokens;
    Ok(outputs)
}

/// Lowers `operation`, applied by one instruction to `input_values` of `input_types` with outputs of `output_types`,
/// with `lowerer`, and constrains every emitted value that carries a pending reduction over a mesh axis that no
/// enclosing manual region binds. Every path that lowers an instruction (i.e., [`lower_instruction`] and the inline
/// lowering of nested regions) goes through this function, which owns the following rule.
///
/// Shardy rejects an operation whose result drops an unreduced axis of its operands, unless a blessed operation (e.g.,
/// `sdy.sharding_constraint`) performs that transition, and its propagation does not carry unreduced axes through most
/// operations (e.g., `stablehlo.add` or `stablehlo.broadcast_in_dim`). JAX constrains the result of every operation
/// on a mesh with explicit axes for that reason. Here, the values that the lowering of an instruction emits are
/// constrained in two steps:
///
///   - Intermediate values, with [`constrain_unreduced_intermediates`]. The reduction state of each emitted value
///     follows the dataflow of the emitted operations from the inputs of the instruction, and is never copied from its
///     output types, because a value that is replicated in fact (e.g., a constant, an iota, or a broadcast of a
///     replicated operand) and declared unreduced would be counted once per device when the reduction completes.
///   - Outputs, with [`constrain_unreduced_outputs`], which constrains each output to the tracked sharding of its type.
///
/// Only an instruction with an input that carries a pending reduction can emit an intermediate value with one, so any
/// other instruction pays one scan of its input types and of its output types, without allocating or walking the
/// emitted operations. The constraints are emitted for every target, and lowering removes them, with every other
/// Shardy operation, for a target without the Shardy dialect (refer to [`remove_shardy_operations`]).
pub(crate) fn lower_instruction_operation<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
    operation: &O,
    lowerer: &mut Lowerer<'b, 'c, 't, O>,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[Type],
    output_types: &[Type],
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    let mut block = lowerer.block();
    let location = lowerer.location();
    let bound_manual_axes = lowerer.collective_state().bound_manual_axes();
    let has_pending_reduction = input_types.iter().any(|input_type| match input_type {
        Type::Array(input_type) => input_type.unreduced_axes().iter().any(|axis| !bound_manual_axes.contains(axis)),
        Type::Dimension(_) | Type::Reference(_) => false,
    });
    // The lowering only appends operations to the block, so the operations that follow an anchor appended before the
    // lowering are those that the lowering emits into this block, which are found without walking the operations of
    // the block that precede them. The anchor is a `stablehlo.after_all` without operands, whose result nothing uses,
    // and it is removed once they are constrained.
    let anchor = if has_pending_reduction {
        Some(block.append_operation(stable_hlo::after_all::<ValueRef<'b, 'c, 't>, _>(&[], location)?)?)
    } else {
        None
    };
    let mut outputs = operation.lower(lowerer, input_values, input_types, output_types)?;
    let collective_state = lowerer.collective_state();
    if let Some(anchor) = anchor {
        constrain_unreduced_intermediates(
            anchor,
            input_values,
            input_types,
            &outputs,
            output_types,
            collective_state,
            &mut block,
            location,
        )?;
        // SAFETY: nothing uses the result of the anchor, so removing it from the block and dropping the detached
        // operation that `remove_operation` returns destroys it without leaving a dangling use.
        unsafe { block.remove_operation(anchor)? };
    }
    constrain_unreduced_outputs(&mut outputs, output_types, collective_state, &mut block, location)?;
    Ok(outputs)
}

/// Pending reduction of a value: the mesh axes, excluding those that an enclosing manual region binds, along which the
/// value holds one contribution per device, and the kind of the reduction that combines those contributions.
#[derive(Clone)]
struct PendingReduction {
    axes: BTreeSet<String>,
    kind: UnreducedKind,
}

impl PendingReduction {
    /// Returns the pending reduction of a value of type `r#type`, if it has one over an axis that is not one of the
    /// `bound_manual_axes`.
    fn of_type(r#type: &Type, bound_manual_axes: &[String]) -> Option<Self> {
        let Type::Array(r#type) = r#type else {
            return None;
        };
        let axes = r#type
            .unreduced_axes()
            .iter()
            .filter(|axis| !bound_manual_axes.contains(axis))
            .cloned()
            .collect::<BTreeSet<_>>();
        (!axes.is_empty()).then(|| Self { axes, kind: r#type.unreduced_kind() })
    }

    /// Returns the pending reduction that a `sdy.sharding_constraint` operation declares for its result, if any.
    fn of_constraint<'o, 'c: 'o, 't: 'c>(operation: OperationRef<'o, 'c, 't>) -> Result<Option<Self>, Error> {
        let Some(constraint) = (unsafe { operation.cast::<shardy::ShardingConstraintOperationRef>() }) else {
            return Ok(None);
        };
        let sharding = shardy::ShardingConstraintOperation::sharding(&constraint)?;
        let axes = sharding
            .unreduced_axes()
            .iter()
            .map(|axis| axis.name().as_str().map(str::to_string))
            .collect::<Result<BTreeSet<_>, _>>()
            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
        let kind = match sharding.reduction_operation()? {
            shardy::ReductionOperation::Sum => UnreducedKind::Sum,
            shardy::ReductionOperation::Max => UnreducedKind::Max,
            shardy::ReductionOperation::Min => UnreducedKind::Min,
        };
        Ok((!axes.is_empty()).then_some(Self { axes, kind }))
    }

    /// Returns the sharding of rank `rank` over `mesh` that declares this pending reduction for a value whose
    /// dimensions are partitioned as `dimensions`, if known, and are otherwise replicated. Replication is also the
    /// fallback when `dimensions` partitions a dimension along an axis of the pending reduction.
    fn sharding(
        &self,
        mesh: &LogicalMesh,
        rank: usize,
        dimensions: Option<&[ShardingDimension]>,
    ) -> Result<Sharding, Error> {
        let declare = |sharding: Sharding| {
            sharding
                .with_unreduced_axes(self.axes.iter().cloned())
                .and_then(|sharding| sharding.with_unreduced_kind(self.kind))
        };
        let partitioned = dimensions
            .filter(|dimensions| dimensions.len() == rank)
            .and_then(|dimensions| Sharding::new(mesh.clone(), dimensions.to_vec()).and_then(declare).ok());
        match partitioned {
            Some(sharding) => Ok(sharding),
            None => declare(Sharding::replicated(mesh.clone(), rank))
                .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() }.into()),
        }
    }
}

/// Names of the elementwise StableHLO operations whose results are partitioned like their operands of the same shape,
/// along which [`constrain_unreduced_intermediates`] carries the dimension shardings of the values that it constrains.
const ELEMENTWISE_OPERATIONS: &[&str] = &[
    "stablehlo.abs",
    "stablehlo.add",
    "stablehlo.and",
    "stablehlo.bitcast_convert",
    "stablehlo.clamp",
    "stablehlo.compare",
    "stablehlo.convert",
    "stablehlo.divide",
    "stablehlo.maximum",
    "stablehlo.minimum",
    "stablehlo.multiply",
    "stablehlo.negate",
    "stablehlo.not",
    "stablehlo.or",
    "stablehlo.select",
    "stablehlo.subtract",
    "stablehlo.xor",
];

/// Returns the shape of `value` when it is a ranked tensor, where [`None`] marks a dynamic dimension.
fn tensor_shape(value: ValueRef<'_, '_, '_>) -> Result<Option<Vec<Option<usize>>>, Error> {
    Ok(value
        .r#type()?
        .cast::<TensorTypeRef>()
        .map(|r#type| r#type.dimensions().map(|size| size.value()).collect()))
}

/// Returns the dimension shardings of `sharding`, with its unconstrained dimensions closed as replicated, which is how
/// a runtime array stores them and how a constraint that must state its pending reduction can declare them.
fn closed_dimensions(sharding: &Sharding) -> Rc<[ShardingDimension]> {
    sharding
        .dimensions()
        .iter()
        .map(|dimension| match dimension {
            ShardingDimension::Unconstrained => ShardingDimension::Replicated,
            dimension => dimension.clone(),
        })
        .collect()
}

/// Constrains the intermediate values that the lowering of one instruction emitted into `block` after `anchor` and that
/// carry a pending reduction, for [`lower_instruction_operation`]. The `outputs` of the instruction, of types
/// `output_types`, are left to [`constrain_unreduced_outputs`].
///
/// The pending reductions follow the dataflow of the emitted operations, in their order in `block`:
///
///   - The `input_values` of the instruction carry the pending reductions of their `input_types`.
///   - A `sdy.sharding_constraint` result carries the pending reduction that the constraint declares (e.g., none for a
///     constraint that completes the reduction). The results of the other Shardy operations, of `func.call`, and of
///     `stablehlo.case` and `stablehlo.if` carry none here, because their shardings are declared by the operation or
///     by the callee, and the values computed in their regions or callees are constrained by the lowering of the
///     instructions there.
///   - Result `i` of `stablehlo.while` carries the pending reduction of its operand `i`, which is how Shardy relates
///     the two, and the body of the loop is constrained by the lowering of its own instructions.
///   - Every result of any other operation (e.g., `stablehlo.compare`, `stablehlo.convert`, `stablehlo.reduce`, or a
///     `stablehlo.broadcast_in_dim` of a scalar) that has an operand with a pending reduction carries the pending
///     reduction of its operands, because each device computes the operation on its own contribution. Shardy requires
///     such a result to keep the unreduced axes of its operands with the same kind, whatever its element type (e.g.,
///     the Boolean result of a comparison of maximum-unreduced values is maximum-unreduced, and the `f32` copy of a
///     `bf16` sum-unreduced value is sum-unreduced), and all results of an operation with several results to have the
///     same unreduced axes. The declaration does not move any data, so it does not change the values. The reduction
///     is only completed where an output type of the instruction or a placement says so, and the core type rules make
///     each such output type the reduction state of the values that the operation computes. Operands without a
///     pending reduction (e.g., a constant, or an operand that the core rules require to be reduced, like the
///     denominator of a `div`) do not contribute to it. The core rules reject operands that carry pending
///     reductions of different kinds, which no single Shardy sharding can declare, so such an operation is rejected
///     with [`LoweringError::UnsupportedOperation`]. Operands that are unreduced over different axes (e.g., the
///     factors of a `mul`) give a result that is unreduced over the union of those axes, like the core rules.
///
/// Each constrained value is a `sdy.sharding_constraint` placed right after the operation that produces it, which
/// declares its pending reduction, and it replaces every later use of that value. A value that a
/// `sdy.sharding_constraint` already uses (e.g., one that the lowering of an instruction of a nested region inlined
/// into the same block constrained) is left as it is. Operations nested in the regions of emitted operations are not
/// visited, because their region arguments carry no sharding of their own (e.g., the scalar arguments of a
/// `stablehlo.reduce` body).
///
/// The constraint partitions the dimensions of the value as an input of the instruction with a pending reduction or an
/// output of the instruction partitions its own, where the value is known to be partitioned alike, so that it does not
/// gather a value that the instruction leaves split along another axis. These dimension shardings follow the
/// [`ELEMENTWISE_OPERATIONS`] whose operands and results have the same shape: forward from the inputs to the results,
/// and backward from the outputs to the operands, which covers the intermediates of a reduction that an elementwise
/// operation turns into an output (e.g., the sum of a mean before its division). Every other operation may permute or
/// drop dimensions, so its results take only the dimension shardings that reach them backward from an output, and an
/// intermediate that neither reaches is replicated in every dimension. Shardy rejects an unconstrained dimension in
/// such a constraint, because its propagation then drops the unreduced axes at the consumer, so the dimensions are
/// always closed.
#[allow(clippy::too_many_arguments)]
fn constrain_unreduced_intermediates<'b, 'c: 'b, 't: 'c>(
    anchor: OperationRef<'b, 'c, 't>,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[Type],
    outputs: &[ValueRef<'b, 'c, 't>],
    output_types: &[Type],
    collective_state: &CollectiveLoweringState,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<(), Error> {
    let bound_manual_axes = collective_state.bound_manual_axes();
    let mut pending_reductions = input_values
        .iter()
        .zip(input_types)
        .filter_map(|(value, r#type)| Some((*value, PendingReduction::of_type(r#type, bound_manual_axes)?)))
        .collect::<Vec<_>>();
    // Every value of a lowered program is placed over the same mesh, which is the one of any input with a pending
    // reduction.
    let Some(mesh) = input_types.iter().zip(input_values).find_map(|(input_type, value)| match input_type {
        Type::Array(input_type) if pending_reductions.iter().any(|(candidate, _)| candidate == value) => {
            input_type.sharding().map(Sharding::mesh)
        }
        _ => None,
    }) else {
        return Ok(());
    };
    // The emitted operations are collected before any constraint is inserted, so that the constraints are not visited.
    let operations = block.operations_after(&anchor)?.collect::<Result<Vec<_>, _>>()?;
    let operation_name = |operation: &OperationRef<'b, 'c, 't>| -> Result<String, Error> {
        let name = operation.name();
        let name = name.as_str().map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
        Ok(name.to_string())
    };
    let find = |value: ValueRef<'b, 'c, 't>, dimensions: &[(ValueRef<'b, 'c, 't>, Rc<[ShardingDimension]>)]| {
        dimensions
            .iter()
            .find(|(candidate, _)| *candidate == value)
            .map(|(_, dimensions)| Rc::clone(dimensions))
    };

    // The dimension shardings of the inputs with a pending reduction, which flow forward, and those of the outputs,
    // which flow backward through the elementwise operations, before any constraint replaces a use.
    let array_dimensions = |values: &[ValueRef<'b, 'c, 't>], types: &[Type], pending: bool| {
        values
            .iter()
            .zip(types)
            .filter_map(|(value, r#type)| match r#type {
                Type::Array(r#type)
                    if !pending || pending_reductions.iter().any(|(candidate, _)| candidate == value) =>
                {
                    r#type.sharding().map(|sharding| (*value, closed_dimensions(sharding)))
                }
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let mut forward_dimensions = array_dimensions(input_values, input_types, true);
    let mut backward_dimensions = array_dimensions(outputs, output_types, false);
    for operation in operations.iter().rev() {
        if operation.result_count() != 1 || !ELEMENTWISE_OPERATIONS.contains(&operation_name(operation)?.as_str()) {
            continue;
        }
        let result = operation.result(0)?.as_ref();
        let Some(dimensions) = find(result, &backward_dimensions) else { continue };
        let shape = tensor_shape(result)?;
        for operand in operation.operand_values() {
            let operand = operand?;
            if find(operand, &backward_dimensions).is_none() && tensor_shape(operand)? == shape {
                backward_dimensions.push((operand, Rc::clone(&dimensions)));
            }
        }
    }

    // The values of a nested region that the lowering inlines into the same block are constrained by the lowering of
    // the instructions of that region, which this walk visits again.
    let is_constrained = |value: ValueRef<'b, 'c, 't>| -> Result<bool, Error> {
        for operand in value.uses()? {
            if operand?.operation()?.name().as_str() == Ok("sdy.sharding_constraint") {
                return Ok(true);
            }
        }
        Ok(false)
    };
    let pending_reduction_of =
        |value: ValueRef<'b, 'c, 't>, pending_reductions: &[(ValueRef<'b, 'c, 't>, PendingReduction)]| {
            pending_reductions
                .iter()
                .find(|(candidate, _)| *candidate == value)
                .map(|(_, reduction)| reduction.clone())
        };
    for operation in operations {
        let name = operation_name(&operation)?;
        let name = name.as_str();
        let result_reductions = if name == "sdy.sharding_constraint" {
            vec![PendingReduction::of_constraint(operation)?]
        } else if name.starts_with("sdy.") || matches!(name, "func.call" | "stablehlo.case" | "stablehlo.if") {
            continue;
        } else if name == "stablehlo.while" {
            operation
                .operand_values()
                .map(|operand| Ok(pending_reduction_of(operand?, &pending_reductions)))
                .collect::<Result<Vec<_>, Error>>()?
        } else {
            let mut merged = None::<PendingReduction>;
            for operand in operation.operand_values() {
                let Some(reduction) = pending_reduction_of(operand?, &pending_reductions) else {
                    continue;
                };
                match &mut merged {
                    None => merged = Some(reduction),
                    Some(merged) if merged.kind == reduction.kind => merged.axes.extend(reduction.axes),
                    Some(merged) => {
                        return Err(Error::Lowering(LoweringError::UnsupportedOperation {
                            operation: name.to_string(),
                            reason: format!(
                                "operands with pending reductions of different kinds `{}` and `{}`",
                                merged.kind, reduction.kind,
                            ),
                        }));
                    }
                }
            }
            if merged.is_none() {
                continue;
            }
            vec![merged; operation.result_count()]
        };
        // An elementwise result is partitioned like its operands of its shape, and result `i` of a loop like its
        // operand `i`.
        let elementwise = ELEMENTWISE_OPERATIONS.contains(&name);
        let constrains = name != "sdy.sharding_constraint";
        let mut anchor = operation;
        for (index, reduction) in result_reductions.into_iter().enumerate() {
            let Some(reduction) = reduction else {
                continue;
            };
            let result = operation.result(index)?.as_ref();
            let mut dimensions = None;
            if elementwise {
                let shape = tensor_shape(result)?;
                for operand in operation.operand_values() {
                    let operand = operand?;
                    if let Some(operand_dimensions) = find(operand, &forward_dimensions)
                        && tensor_shape(operand)? == shape
                    {
                        dimensions = Some(operand_dimensions);
                        break;
                    }
                }
            } else if name == "stablehlo.while" {
                dimensions = find(operation.operand_value(index)?, &forward_dimensions);
            }
            let dimensions = dimensions.or_else(|| find(result, &backward_dimensions));
            let rank = result.r#type()?.cast::<TensorTypeRef>().map(|r#type| r#type.rank());
            let (Some(rank), true) = (rank, constrains && !outputs.contains(&result) && !is_constrained(result)?)
            else {
                if let Some(dimensions) = dimensions {
                    forward_dimensions.push((result, dimensions));
                }
                pending_reductions.push((result, reduction));
                continue;
            };
            let sharding = reduction.sharding(mesh, rank, dimensions.as_deref())?;
            let constrained = lower_sharding_constraint(&[result], &sharding, bound_manual_axes, block, location)?[0];
            let constraint = constrained.cast::<OperationResultRef>().unwrap().operation()?;
            // The constraint is appended to the block, and moved right after the operation that produces its operand
            // (or after the constraint of the previous result), so that it dominates every later use of that operand.
            anchor = unsafe { constraint.move_after(&anchor) };
            result.replace_uses_except(constrained, &[&anchor]);
            if let Some(dimensions) = dimensions {
                forward_dimensions.push((constrained, dimensions));
            }
            pending_reductions.push((constrained, reduction));
        }
    }
    Ok(())
}

/// Constrains every lowered instruction output in `outputs` whose type in `output_types` is unreduced over a mesh axis
/// that no enclosing manual region binds to its tracked sharding, for [`lower_instruction_operation`]. An output that
/// a Shardy operation produces (e.g., the `sdy.sharding_constraint` of a placement transition or the
/// `sdy.manual_computation` of a `shard_map`) already declares its sharding and is left as it is.
fn constrain_unreduced_outputs<'b, 'c: 'b, 't: 'c>(
    outputs: &mut [ValueRef<'b, 'c, 't>],
    output_types: &[Type],
    collective_state: &CollectiveLoweringState,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<(), Error> {
    let bound_manual_axes = collective_state.bound_manual_axes();
    let is_declared = |output: &ValueRef<'b, 'c, 't>| {
        output
            .cast::<OperationResultRef>()
            .and_then(|result| result.operation().ok())
            .is_some_and(|operation| operation.name().as_str().is_ok_and(|name| name.starts_with("sdy.")))
    };
    for (output, output_type) in outputs.iter_mut().zip(output_types) {
        if let Type::Array(output_type) = output_type
            && let Some(sharding) = output_type.sharding()
            && sharding.unreduced_axes().iter().any(|axis| !bound_manual_axes.contains(axis))
            && !is_declared(output)
        {
            *output = lower_sharding_constraint(&[*output], sharding, bound_manual_axes, block, location)?[0];
        }
    }
    Ok(())
}

/// Normalizes an MLIR symbol name provided by the user by trimming surrounding whitespace and one leading `@`.
///
/// # Errors
///
/// Returns [`LoweringError::InvalidFunctionName`] when the normalized name is empty or contains whitespace.
pub(crate) fn normalize_function_name(function_name: &str) -> Result<String, Error> {
    let function_name = function_name.trim();
    let function_name = function_name.strip_prefix('@').unwrap_or(function_name);
    if function_name.is_empty() || function_name.chars().any(char::is_whitespace) {
        return Err(Error::Lowering(LoweringError::InvalidFunctionName { function_name: function_name.to_string() }));
    }
    Ok(function_name.to_string())
}

/// Rejects array `output_types` of the operation named `operation_name` that shard a local dimension over one of the
/// `bound_manual_axes` of the enclosing manual regions. Manual dimension ownership belongs to region boundaries. Local
/// operations may change manual variation through collectives, but must not manufacture a global placement, which
/// lowered sharding constraints would otherwise silently strip (refer to
/// [`lower_sharding_constraint`](crate::operations::sharding::constrain_sharding::lower_sharding_constraint)).
/// First-class dimensions and references carry no placement of their own.
fn validate_local_output_placement(
    operation_name: &str,
    output_types: &[Type],
    bound_manual_axes: &[String],
) -> Result<(), Error> {
    let assigns_bound_manual_axis = output_types.iter().any(|output_type| {
        let sharding = match output_type {
            Type::Array(output_type) => output_type.sharding(),
            Type::Dimension(_) | Type::Reference(_) => None,
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

/// Expands optional shardings of every value of a signature with `count` values into one optional sharding per value.
/// Provided shardings are kept as they are, so that a count mismatch is still reported when the module is lowered.
fn value_shardings(shardings: Option<&[Sharding]>, count: usize) -> Vec<Option<Sharding>> {
    match shardings {
        Some(shardings) => shardings.iter().cloned().map(Some).collect(),
        None => vec![None; count],
    }
}

/// Lowers `program` to a [`LoweredModule`] for the lowering entry points above, validating its reference state
/// aliases and entry boundary. When `export_effects` is set and the program has ordered I/O, the entry function also
/// exports the runtime ordered I/O token chain as an extra argument and result.
#[allow(clippy::too_many_arguments)]
fn lower_mlir_module_for_program_with_effect_boundary<
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
    supports_shardy: bool,
    reference_states: &[ExternalReferenceBinding],
    donated_inputs: &[bool],
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
    let session_constants: Rc<[RetainedArray]> = collect_session_constants(program).into();
    let session_count = session_constants.len();
    if session_count > 0
        && (capture_types.len() < session_count
            || !capture_types[capture_types.len() - session_count..].iter().eq(session_constants
                .iter()
                .map(|constant| constant.r#type().into_owned())
                .collect::<Vec<_>>()
                .iter()))
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
            // Distinct logical inputs, which `state_inputs` enforces above, map to distinct physical inputs, so this
            // insertion can never displace an earlier alias.
            aliases.insert(physical_input_index, physical_output_index);
        }
    }
    if !donated_inputs.is_empty() && donated_inputs.len() != global_input_types.len() {
        return Err(Error::Execution(ExecutionError::DonationFlagCountMismatch {
            expected_count: global_input_types.len(),
            actual_count: donated_inputs.len(),
        }));
    }
    // Each donated input is paired with the first output that can share its buffers and that no earlier input was
    // paired with, as JAX pairs the donated arguments of a `jit` function with its results. The pairs are the same
    // hints as the reference state aliases above, so an input that finds no output simply stays unpaired.
    for (input_index, _) in donated_inputs.iter().enumerate().filter(|(_, donated)| **donated) {
        let logical_input_index = capture_types.len() + input_index;
        let input_type = &logical_argument_types[logical_input_index];
        let Some(physical_input_index) = signature.input_mapping()[logical_input_index] else { continue };
        if state_inputs.contains(&logical_input_index)
            || input_type.static_shape().is_none()
            || input_type.memory() != Memory::Device
        {
            continue;
        }
        let input_sharding = &argument_shardings[logical_input_index];
        // The compiler places an open output itself, and only on a single device is that placement certain to be the
        // placement of the input.
        let has_single_device = input_sharding
            .as_ref()
            .is_none_or(|sharding| sharding.mesh().axes().iter().map(|axis| axis.size()).product::<usize>() == 1);
        let physical_output_index =
            global_output_types.iter().enumerate().find_map(|(logical_output_index, output_type)| {
                let physical_output_index = signature.output_mapping()[logical_output_index]?;
                let is_open = open_results.get(logical_output_index).copied().unwrap_or(false);
                (output_type == input_type
                    && &result_shardings[logical_output_index] == input_sharding
                    && (has_single_device || !is_open)
                    && !aliased_outputs.contains(&physical_output_index))
                .then_some(physical_output_index)
            });
        if let Some(physical_output_index) = physical_output_index {
            aliased_outputs.insert(physical_output_index);
            aliases.insert(physical_input_index, physical_output_index);
        }
    }
    let physical_argument_types = signature.physical_input_types(logical_argument_types.as_slice());
    let physical_output_types = signature.physical_output_types(global_output_types.as_slice());

    let context = MlirContext::new();
    let location = context.unknown_location();
    let module = context.module(location)?;

    // Emit `sdy.mesh` declarations for any sharding referenced either by inner operations or by the optional signature
    // shardings, so the `func.func` attributes can refer to `@mesh`.
    let mut signature_mesh = None;
    for sharding in argument_shardings.iter().chain(result_shardings).flatten() {
        signature_mesh = Some(match signature_mesh.take() {
            Some(existing_mesh) => merge_logical_meshes(existing_mesh, sharding.mesh())?,
            None => sharding.mesh().clone(),
        });
    }
    let mesh = collect_nested_sharding_mesh(program, signature_mesh)?;
    if let Some(mesh) = mesh.as_ref() {
        let mesh_operation = mesh.to_mlir(location)?;
        module.body()?.append_operation(mesh_operation)?;
    }

    // Trace each distinct named semantic decomposition once and retain it in the lowering state of the module, so that
    // the entry and nested function bodies resolve the same private symbol.
    let named_compositions = Rc::new(collect_named_composition_functions(program, target_platform)?);

    // Module-scoped lowering state, shared between the entry function body and private functions below so channel ids
    // stay unique and target, representation policy, named compositions, and provenance state reach nested bodies.
    let collective_state = CollectiveLoweringState::new()
        .with_target_platform(target_platform)
        .with_ragged_dot_lowering_strategy(ragged_dot_lowering_strategy)
        .with_named_compositions(named_compositions.clone())
        .with_session_constants(session_constants);

    // Deduplicate `jit_call` callees that occur more than once into shared private `func.func`s, so repeated nested
    // programs (e.g., identical transformer blocks or the per-block primal and pullback programs of `grad`) lower
    // to one function plus N `func.call`s instead of N inlined copies. The map is empty for modules without repeated
    // calls, in which case every `jit_call` inlines exactly as before.
    let nested_functions = Rc::new(collect_jit_call_functions(program, target_platform)?);
    {
        let mut module_block = module.body()?;
        for key in named_compositions.order() {
            let function = named_compositions.functions().get(key).unwrap();
            emit_named_composition_function(
                &mut module_block,
                function,
                &collective_state,
                &context,
                location.as_ref(),
            )?;
        }
        for key in nested_functions.order() {
            let function = nested_functions.functions().get(key).unwrap();
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
    // shardings of output types without `out_shardings` (refer to `ShardyTensorSharding`).
    let physical_open_results =
        if open_results.is_empty() { Vec::new() } else { signature.project_outputs(open_results) };
    let mut result_sharding_attributes = signature
        .physical_optional_output_shardings(result_shardings)
        .iter()
        .enumerate()
        .map(|(index, sharding)| {
            let open = physical_open_results.get(index).copied().unwrap_or(false);
            sharding
                .as_ref()
                .map(|sharding| ShardyTensorSharding::new(sharding, open).to_mlir(location))
                .transpose()
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
    // Internal layout constraints do not constrain the executable ABI. Declare layouts on the physical boundary, whose
    // types already omit static zeros and include hidden extent scalars, while tokens have no array type. Placement
    // belongs to physical array slots, including captures, but never to the appended effect token. Default device
    // placement needs no attribute, but host identities still need attributes even without transfers.
    let function_arguments = physical_argument_tensor_types
        .iter()
        .enumerate()
        .map(|(index, tensor_type)| {
            let mut attributes = HashMap::new();
            if supports_shardy && let Some(sharding) = argument_sharding_attributes.get(index).and_then(Option::as_ref)
            {
                attributes.insert("sdy.sharding".into(), sharding.as_ref());
            }
            if let Some(r#type) = physical_argument_types.get(index).filter(|r#type| r#type.memory() != Memory::Device)
            {
                validate_boundary_memory(r#type.memory(), target_platform)?;
                attributes.insert(
                    "mhlo.memory_kind".into(),
                    context.string_attribute(r#type.memory().placement_kind()).as_ref(),
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
            if supports_shardy && let Some(sharding) = result_sharding_attributes.get(index).and_then(Option::as_ref) {
                attributes.insert("sdy.sharding".into(), sharding.as_ref());
            }
            if let Some(r#type) = physical_output_types.get(index).filter(|r#type| r#type.memory() != Memory::Device) {
                validate_boundary_memory(r#type.memory(), target_platform)?;
                attributes.insert(
                    "mhlo.memory_kind".into(),
                    context.string_attribute(r#type.memory().placement_kind()).as_ref(),
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
                        let mut value = function_block.argument(physical_index).unwrap().as_ref();
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
                            let size =
                                function_block.argument(input_dimension.physical_input_index()).unwrap().as_ref();
                            let operation = function_block_ref.append_operation(stable_hlo::set_dimension_size(
                                value,
                                size,
                                lower_tensor_type(&refined_type, &context, location)?,
                                input_dimension.axis(),
                                location,
                            )?)?;
                            value = operation.result(0).unwrap().as_ref();
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
                    .unwrap(),
                };
                debug_assert_eq!(value.r#type()?, tensor_type.as_ref());
                logical_argument_values.push(value);
            }
            let (capture_values, public_input_values) = logical_argument_values.split_at(capture_types.len());
            // Ordinary module callers lower an unlifted program whose entry inputs are only the public arguments.
            // The compilation domain instead supplies the program with lifted captures, so that core reference
            // discharge and lowering observe the same arena. In that form the complete logical argument list is also
            // the entry input list, while the leading prefix still serves attached `CaptureReference` constants. No
            // third boundary form exists, so the entry input count must match one of those two shapes exactly.
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
            // equal (e.g., the same SSA value or equal computations merged by CSE), one `SliceToDynamic` survives and
            // copy insertion fills the other result with a plain copy of the bounded payload, which leaves its trailer
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
    let mesh = declare_constrained_mesh(&module, program, mesh, location.as_ref())?;
    if !supports_shardy {
        remove_shardy_operations(&module)?;
    }

    remove_unused_constants(&module)?;
    if !module.verify()? {
        return Err(Error::Lowering(LoweringError::VerificationFailure));
    }
    if collective_state.has_assertions.get() {
        signature.require_assertion_handler();
    }
    let requires_assertion_handler = signature.requires_assertion_handler();
    let mut runtime_requirements = collective_state.runtime_requirements.borrow().clone();
    if requires_assertion_handler {
        runtime_requirements.push(RuntimeRequirement::builtin("ryft.assert"));
    }
    if signature.has_ordered_io() {
        runtime_requirements.push(RuntimeRequirement::builtin("ryft.ordered-io"));
    }
    let runtime_requirements = normalize_runtime_requirements(runtime_requirements)?.into();
    Ok(LoweredModule {
        stable_hlo: serialize_lowered_module(&module, &collective_state)?,
        signature,
        requires_assertion_handler,
        runtime_requirements,
        mesh,
    })
}

/// Returns the placement that `region` gives each of its results, which [`entry_result_placements`] describes, with
/// [`EntryResultPlacement::Argument`] naming an input of `region`. A result that returns a [`RetainedArray`] has the
/// placement of that array, which enters the executable as a hidden argument with that placement, and which has no
/// buffer whose placement the compiler could report when it has the zero data type. The placements of the callee
/// regions of `jit_call`s are computed once per callee and memoized in `callee_placements`, because repeated calls
/// share their callee region. The inputs of `region` and the producer of each of its values, together with the
/// position of that value among the outputs of its producer, are indexed once, so that following a result back to an
/// input costs one lookup per step rather than a scan of the inputs or of the outputs of an instruction.
fn region_result_placements<O: LowerableOperationFamily>(
    region: RegionRef<'_, Constant, O>,
    callee_placements: &mut HashMap<RegionId, Rc<[Option<EntryResultPlacement>]>>,
) -> Rc<[Option<EntryResultPlacement>]> {
    let mut producers = vec![None; region.atoms().len()];
    for (instruction_index, instruction) in region.instructions().iter().enumerate() {
        for (output_index, output) in instruction.outputs().iter().enumerate() {
            producers[output.index()] = Some((instruction_index, output_index));
        }
    }
    // A value that is several inputs is placed as the first of them, so later positions are visited first.
    let mut input_positions = vec![None; region.atoms().len()];
    for (position, input) in region.input_ids().iter().enumerate().rev() {
        input_positions[input.index()] = Some(position);
    }
    let mut result_placement = |output_id: AtomId| {
        let mut value_id = output_id;
        loop {
            let Some((instruction_index, output_index)) = producers[value_id.index()] else {
                if let Some(Constant::Session(constant)) = region.atoms()[value_id.index()].as_constant() {
                    return Some(EntryResultPlacement::Target(constant.sharding().clone()));
                }
                return input_positions[value_id.index()].map(EntryResultPlacement::Argument);
            };
            let instruction = &region.instructions()[instruction_index];
            let operation = instruction.operation();
            let input_index = if let Some(operation) = operation.projected_payload::<PlaceOperation>() {
                match operation.target() {
                    Some(PlacementRequest::Resolved(target)) => {
                        return Some(EntryResultPlacement::Target(target.sharding().clone()));
                    }
                    Some(PlacementRequest::Device { .. }) => {
                        // A device target completes a pending reduction over mesh axes of size one, so its result can
                        // have another sharding than its input, which is then the placement of the result.
                        let sharding = |atom: AtomId| match region.atoms()[atom.index()].r#type().as_ref() {
                            Type::Array(array_type) => array_type.sharding().cloned(),
                            _ => None,
                        };
                        let output_sharding = sharding(instruction.outputs()[output_index]);
                        if output_sharding != sharding(*instruction.inputs().first()?) {
                            return output_sharding.map(EntryResultPlacement::Target);
                        }
                        0
                    }
                    // An in-place target leaves its input on the devices where it lives with its sharding, so the
                    // result is placed as its input is.
                    Some(PlacementRequest::InPlace { .. }) | None => 0,
                }
            } else if operation.projected_payload::<StopGradientOperation<ArrayType>>().is_some()
                || operation.projected_payload::<TagOperation<ArrayType>>().is_some()
                || operation.projected_payload::<TransferToMemoryOperation>().is_some()
            {
                // These operations return each of their inputs as the output at the same position, and a memory
                // transfer keeps the devices and the sharding of its input like an in-place `place` target does.
                output_index
            } else if let Some(call) = operation.projected_payload::<JitCallOperation<Type>>() {
                // The inputs of the callee receive the inputs of the call positionally, and the results of the call
                // are the results of the callee. A sharding override of the boundary of the callee places the result
                // or the forwarded input that it covers, as an execution of the callee does, which its body cannot
                // always show (e.g., a replicated override over auto mesh axes stages nothing).
                if let Some(sharding) = call.output_sharding(output_index) {
                    return Some(EntryResultPlacement::Target(sharding.clone()));
                }
                let callee = *instruction.regions().first()?;
                let placements = match callee_placements.get(&callee) {
                    Some(placements) => Rc::clone(placements),
                    None => {
                        let placements = region_result_placements(region.with_id(callee).ok()?, callee_placements);
                        callee_placements.insert(callee, Rc::clone(&placements));
                        placements
                    }
                };
                match placements.get(output_index)?.as_ref()? {
                    EntryResultPlacement::Target(sharding) => {
                        return Some(EntryResultPlacement::Target(sharding.clone()));
                    }
                    EntryResultPlacement::Argument(index) => {
                        let actual = region.atoms()[instruction.inputs().get(*index)?.index()].r#type();
                        let callee = region.with_id(callee).ok()?;
                        let declared = callee.input_types();
                        match (actual.as_ref(), declared.get(*index)?) {
                            (Type::Array(actual), declared) => {
                                match call.boundary_input_sharding(*index, actual, declared) {
                                    Some(sharding) => return Some(EntryResultPlacement::Target(sharding.into_owned())),
                                    None => *index,
                                }
                            }
                            _ => *index,
                        }
                    }
                }
            } else {
                return None;
            };
            value_id = *instruction.inputs().get(input_index)?;
        }
    };
    region.output_ids().iter().map(|&output_id| result_placement(output_id)).collect()
}

/// Returns the module mesh of `module`, which is the lowering of `program`, after declaring it when `mesh`, the mesh
/// that the signature and `collect_nested_sharding_mesh` found before the program was lowered, is [`None`] but the
/// lowered operations refer to a mesh anyway. This makes the declaration follow what lowering emitted rather than a
/// list of the operations that emit sharding constraints, which a new constraint (e.g., one that a lowering rule emits
/// for the sharding of a result type) would otherwise have to join. Every Shardy operation other than the declaration
/// itself refers to the module mesh, and the declared mesh is the one shared by the shardings of the array types of
/// the regions that lowering emits, which every constraint over the sharding of a type refers to. A constraint over a
/// sharding that no type carries must be found by `collect_nested_sharding_mesh` instead, and verification of the
/// module reports it otherwise.
///
/// # Errors
///
/// Returns [`LoweringError::MismatchedMeshTopology`] when the shardings of those types disagree on the topology.
fn declare_constrained_mesh<'c, 't, ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    module: &Module<'c, 't>,
    program: &Program<ProgramInput, ProgramOutput, O>,
    mesh: Option<LogicalMesh>,
    location: LocationRef<'c, 't>,
) -> Result<Option<LogicalMesh>, Error>
where
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
{
    if mesh.is_some() {
        return Ok(mesh);
    }
    let mut refers_to_mesh = false;
    module.as_operation()?.walk(WalkOrder::PreOrder, |operation| match operation.name().as_str() {
        Ok(name) if name.starts_with("sdy.") && name != "sdy.mesh" => {
            refers_to_mesh = true;
            WalkResult::Interrupt
        }
        _ => WalkResult::Advance,
    });
    if !refers_to_mesh {
        return Ok(None);
    }
    let mut mesh = None;
    for region in program.entry_region_ref().computation_regions() {
        for atom in region.atoms() {
            if let Type::Array(array_type) = atom.r#type().as_ref()
                && let Some(sharding) = array_type.sharding()
            {
                mesh = Some(match mesh.take() {
                    Some(existing_mesh) => merge_logical_meshes(existing_mesh, sharding.mesh())?,
                    None => sharding.mesh().clone(),
                });
            }
        }
    }
    if let Some(mesh) = &mesh {
        module.body()?.insert_operation(mesh.to_mlir(location)?, 0)?;
    }
    Ok(mesh)
}

/// Validates that `incoming` has the topology of the module mesh `existing`, which it returns. The module mesh keeps
/// the axis types of the first mesh that declared it, because the emitted `sdy.mesh` records only axis names and sizes.
///
/// # Errors
///
/// Returns [`LoweringError::MismatchedMeshTopology`] when the axis names or sizes of the two meshes differ, including
/// in order.
fn merge_logical_meshes(existing: LogicalMesh, incoming: &LogicalMesh) -> Result<LogicalMesh, Error> {
    if !existing.has_same_topology(incoming) {
        return Err(Error::Lowering(LoweringError::MismatchedMeshTopology {
            expected: existing,
            actual: incoming.clone(),
        }));
    }
    Ok(existing)
}

/// Removes every Shardy operation from `block` and from the regions nested in its operations, as
/// [`remove_shardy_operations`] describes.
fn remove_shardy_operations_in_block<'b, 'c: 'b, 't: 'c>(block: BlockRef<'b, 'c, 't>) -> Result<(), Error> {
    let mut shardy_operations = Vec::new();
    for operation in block.operations()? {
        let operation = operation?;
        // An operation is classified before its regions are visited, so that an unsupported operation is reported
        // rather than the Shardy terminator of its body.
        let name = operation.name();
        match name.as_str() {
            Ok("sdy.sharding_constraint") => {
                operation.result(0)?.replace_uses(operation.operand_value(0)?);
                shardy_operations.push(operation);
            }
            Ok("sdy.mesh") => shardy_operations.push(operation),
            Ok(name) if name.starts_with("sdy.") => {
                return Err(Error::Lowering(LoweringError::UnsupportedOperation {
                    operation: name.to_string(),
                    reason: "the target plugin cannot compile with Shardy".to_string(),
                }));
            }
            _ => {}
        }
        for region in operation.regions() {
            for nested_block in region?.blocks()? {
                remove_shardy_operations_in_block(nested_block?)?;
            }
        }
    }
    for operation in shardy_operations {
        // SAFETY: the uses of a removed constraint were replaced by its operand and a mesh declaration has no results,
        // so no remaining operation references a result of the removed operation, and dropping the detached operation
        // that `remove_operation` returns destroys it.
        unsafe { block.remove_operation(operation)? };
    }
    Ok(())
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

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, Array, ArrayOperations, AtomId, CaptureReference, CastOperation, Complex, ConditionOperation,
        ConstantOperation, Context, Cos, Differentiate, Dimension, DimensionBounds, DimensionToScalarOperation,
        DimensionType, DimensionValue, DimensionVariable, DomainTracingContext, Dot, DotDimensionNumbers, EagerContext,
        EffectClasses, Error as CoreError, Layout, LinearCallOperation, Max, MeshAxis, MeshAxisType, MulOperation,
        NegOperation, OneLike, ParallelReduce, ParameterizedFamily, Place, Placeholder, Placement, PrintOperation,
        Program, ProgramBuilder as CoreProgramBuilder, ProjectedContext, ProvenanceScope, Reduce, ReduceOperation,
        ReductionKind, ReferenceFreezeOperation, ReferenceNewOperation, ReferenceReadOperation, ReferenceSource,
        ReferenceType, ReferenceWriteOperation, Reshape, ReshapeOperation, ReverseModeDifferentiate, ScanOptions,
        ShardMap, ShardMapOperation, ShardMapOptions, Sin, StopGradient, StridedLayout, Tag, TiledLayout,
        TransferToMemory, Transpose, TypedProgram, UnreducedKind, ValueProjection, ZeroLike, i1, i2, i4, scan,
        shard_map, u1, u2, u4,
    };
    use ryft_kernels::{KernelBoundaryContract, KernelParameterAccess, validate_kernel_body};
    use ryft_mlir::dialects::builtin::attributes::DenseElementsAttribute;
    use ryft_mlir::{
        Attribute, DialectHandle, ElementsAttribute, Operation as MlirOperation, Region, Size as MlirSize,
    };
    use ryft_pjrt::protos::{OptionOverride, OptionValue, ShapeChecks};
    use ryft_pjrt::{
        BufferType, ClientOptions, CpuClientOptions, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram,
        load_cpu_plugin,
    };

    use crate::lowering::LoweringError;
    use crate::lowering::regions::tests::lower_plain_program_outputs;
    use crate::operations::assertions::ensure_assertion_handler_registered;
    use crate::sessions::tests::{domain_mesh, f32_scalar, two_device_mesh};
    use crate::tests::{cpu_client, replicated_array, test_compilation_options, values_from_bytes, values_to_bytes};
    use crate::{
        ArrayTracer, CompileTracer, CompiledFunction, FlatProgram, FromPjrt, JitCallOperation, Options,
        PlacementOptions, ProgramBuilder as ArrayIrProgramBuilder, Session, TracedProgram, compile, trace,
        trace_with_context,
    };

    use super::*;

    /// Builder of homogeneous array programs that operation tests across the crate use to stage fixtures for the test-only
    /// plain program lowering (i.e., [`to_mlir_module_for_plain_program`]) and for [`unproject_plain_program`].
    pub(crate) type ProgramBuilder = CoreProgramBuilder<ArrayConstant, ArrayOperation<ArrayConstant>>;

    /// Homogeneous array program that [`ProgramBuilder`] builds, which operation tests across the crate lower through the
    /// test-only plain program lowering or lift into the array IR through [`unproject_plain_program`].
    pub(crate) type PlainProgram =
        Program<ArrayConstant, ArrayOperation<ArrayConstant>, Vec<ArrayConstant>, Vec<ArrayConstant>>;

    /// Lowers a plain traced [`Program`] to a textual StableHLO MLIR module.
    pub(crate) fn to_mlir_module_for_plain_program<
        V: MlirLowerableValue,
        Input: Parameterized<V>,
        Output: Parameterized<V>,
        O: LowerableOperation<Type = ArrayType>,
        S: AsRef<str>,
    >(
        program: &Program<V, O, Input, Output>,
        function_name: S,
    ) -> Result<String, Error> {
        to_mlir_module_for_plain_program_with_ragged_dot_lowering_strategy(
            program,
            function_name,
            RaggedDotLoweringStrategy::default(),
        )
    }

    /// Lowers a plain traced program like [`to_mlir_module_for_plain_program`], but with an explicit
    /// [`RaggedDotLoweringStrategy`].
    pub(crate) fn to_mlir_module_for_plain_program_with_ragged_dot_lowering_strategy<
        V: MlirLowerableValue,
        Input: Parameterized<V>,
        Output: Parameterized<V>,
        O: LowerableOperation<Type = ArrayType>,
        S: AsRef<str>,
    >(
        program: &Program<V, O, Input, Output>,
        function_name: S,
        ragged_dot_lowering_strategy: RaggedDotLoweringStrategy,
    ) -> Result<String, Error> {
        let function_name = normalize_function_name(function_name.as_ref())?;
        let context = MlirContext::new();
        let location = context.unknown_location();
        let module = context.module(location)?;
        // This entry point lowers one whole module, so the collective lowering state is scoped to that module.
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
                    Some(existing_mesh) => merge_logical_meshes(existing_mesh, sharding.mesh())?,
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
                lower_tensor_type(input_atom.r#type().as_ref(), &context, location)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_tensor_types = program
            .output_ids()
            .iter()
            .map(|atom_id| {
                let output_atom = &program.atoms()[atom_id.index()];
                lower_tensor_type(output_atom.r#type().as_ref(), &context, location)
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
                                    context.string_attribute(memory.placement_kind()).as_ref(),
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
                                    context.string_attribute(memory.placement_kind()).as_ref(),
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

    /// Logical mesh with one manual axis named `axis_name` of size `axis_size`.
    pub(crate) fn test_manual_mesh(axis_name: &str, axis_size: usize) -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::new(axis_name, axis_size, MeshAxisType::Manual).unwrap()]).unwrap()
    }

    /// Static `f32` vector type of `length` elements.
    pub(crate) fn test_vector_type(length: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(length)]))
    }

    /// Static `f32` matrix type with `rows` rows and `columns` columns.
    pub(crate) fn test_matrix_type(rows: usize, columns: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(rows), Dimension::Static(columns)]))
    }

    /// Global array type of `data_type` with the static `global_shape`, placed by `sharding`.
    pub(crate) fn static_sharded_array_type(
        data_type: DataType,
        global_shape: &[usize],
        sharding: Sharding,
    ) -> ArrayType {
        ArrayType::new(data_type, Shape::new(global_shape.iter().copied().map(Dimension::Static).collect()))
            .with_sharding(sharding)
            .unwrap()
    }

    /// Sharding over `mesh` with the provided dimension shardings and unreduced axes.
    pub(crate) fn test_sharding(
        mesh: &LogicalMesh,
        dimensions: Vec<ShardingDimension>,
        unreduced_axes: Vec<String>,
    ) -> Sharding {
        Sharding::new(mesh.clone(), dimensions).unwrap().with_unreduced_axes(unreduced_axes).unwrap()
    }

    /// Traces `function` as the body of one top-level `shard_map` (refer to [`ShardMap`]) over global inputs of
    /// `global_input_types` placed by `input_shardings` without their auto axes (i.e., as the boundary places them), and
    /// lowers the traced program to a module whose `main` function signature carries the shardings of its global input and
    /// output types, which are the boundary shardings.
    pub(crate) fn shard_map_module<F, Input, Output>(
        function: F,
        global_input_types: Input,
        mesh: LogicalMesh,
        input_shardings: Input::To<Sharding>,
        output_shardings: Output::To<Sharding>,
        options: ShardMapOptions,
    ) -> Result<String, Error>
    where
        Input: Parameterized<ArrayType>,
        Input::Family: ParameterizedFamily<Constant> + ParameterizedFamily<ArrayTracer> + ParameterizedFamily<Sharding>,
        Output: Parameterized<ArrayType>,
        Output::Family:
            ParameterizedFamily<Constant> + ParameterizedFamily<ArrayTracer> + ParameterizedFamily<Sharding>,
        F: FnOnce(
            &ProjectedContext<DomainTracingContext<Session>, ArrayType>,
            Input::To<ArrayTracer>,
        ) -> Result<Output::To<ArrayTracer>, ProgramError>,
    {
        let input_structure = global_input_types.parameter_structure();
        let output_structure = output_shardings.parameter_structure();
        let input_shardings = input_shardings.into_parameters().collect::<Vec<_>>();
        let output_shardings = output_shardings.into_parameters().collect::<Vec<_>>();
        let global_input_types = Input::from_parameters(
            input_structure.clone(),
            global_input_types
                .into_parameters()
                .zip(&input_shardings)
                .map(|(r#type, sharding)| r#type.with_sharding(sharding.without_auto_axes()))
                .collect::<Result<Vec<_>, _>>()?,
        )?;
        let traced: TracedProgram<Input, Output> = trace_with_context(
            |context, inputs| {
                let outputs = context.shard_map(
                    |body_context, local_inputs: Vec<ArrayTracer>| {
                        let local_inputs = Input::To::<ArrayTracer>::from_parameters(input_structure, local_inputs)?;
                        Ok(function(body_context, local_inputs)?.into_parameters().collect::<Vec<_>>())
                    },
                    inputs.into_parameters().collect::<Vec<_>>(),
                    mesh,
                    input_shardings,
                    output_shardings,
                    options,
                )?;
                Ok(Output::To::<ArrayTracer>::from_parameters(output_structure, outputs)?)
            },
            global_input_types,
        )?;
        traced.to_mlir_module("main")
    }

    /// Dynamic dimension of a fresh non-negative variable named `name`, bounded above by `exclusive_upper_bound` when it
    /// is provided and unbounded otherwise.
    pub(crate) fn dynamic_dimension(name: &str, exclusive_upper_bound: Option<usize>) -> Dimension {
        DimensionVariable::new(name, DimensionBounds::non_negative(exclusive_upper_bound).unwrap()).into()
    }

    /// One physical value of the mixed executable ABI used by the reshape execution tests. It is either `f64` or `i32`
    /// array storage, with its physical dimensions on input and the row-major elements inside its logical dimensions on
    /// output, or one `i64` dimension, which on output also carries the hidden runtime extents that follow the logical
    /// outputs.
    #[derive(Clone, Debug, PartialEq)]
    pub(crate) enum MixedValue {
        /// `f64` storage of one array together with its dimensions.
        Array(Vec<f64>, Vec<u64>),

        /// `i32` storage of one array together with its dimensions (e.g., explicit group sizes or transfer lengths).
        Integers(Vec<i32>, Vec<u64>),

        /// One `i64` first-class dimension or hidden runtime extent.
        Dimension(i64),
    }

    /// Compiles one array IR program for the mixed executable ABI and executes it once on `client`. Each logical input
    /// receives the matching entry of `inputs` (i.e., array storage sized to the allocation bound, or one dimension), and
    /// every bounded dynamic input axis then receives its runtime extent from `extents` in boundary order.
    pub(crate) fn execute_mixed_program(
        client: &'static ryft_pjrt::Client<'static>,
        program: &FlatProgram,
        inputs: &[MixedValue],
        extents: &[i32],
    ) -> Result<Vec<MixedValue>, Error> {
        // First-class dimensions cross the executable boundary as `i64` scalars.
        let boundary_type = |r#type: &Type| match r#type {
            Type::Array(r#type) => r#type.clone(),
            Type::Dimension(_) => ArrayType::scalar(DataType::I64),
            Type::Reference(_) => panic!("reshape execution fixtures have no reference values"),
        };
        let input_types = program.input_types().iter().map(boundary_type).collect::<Vec<_>>();
        let output_types = program.output_types().iter().map(boundary_type).collect::<Vec<_>>();
        let module = to_mlir_module_for_program(program, &[], &input_types, &output_types, "main", None, None).unwrap();
        // Lowering may synthesize runtime checks (e.g., entry guards of mixed region entries), so register their handler.
        ensure_assertion_handler_registered(client)?;
        let mut options = test_compilation_options(client, None);
        if client.platform_name()?.eq_ignore_ascii_case("cuda") {
            // CUDA's DynamicPadder can reject mismatched inputs before the Ryft assertions these fixtures exercise.
            options.environment_option_overrides.insert(
                "xla_gpu_shape_checks".to_string(),
                OptionOverride { value: Some(OptionValue::IntField(ShapeChecks::Ignore as i64)) },
            );
        }
        let executable = client.compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &options)?;
        let device = executable.addressable_devices()?.remove(0);
        let mut buffers = Vec::with_capacity(inputs.len() + extents.len());
        for input in inputs {
            buffers.push(match input {
                MixedValue::Array(values, dimensions) => client.buffer(
                    values_to_bytes(values).as_slice(),
                    BufferType::F64,
                    dimensions.as_slice(),
                    None,
                    device.clone(),
                    None,
                )?,
                MixedValue::Integers(values, dimensions) => client.buffer(
                    values_to_bytes(values).as_slice(),
                    BufferType::I32,
                    dimensions.as_slice(),
                    None,
                    device.clone(),
                    None,
                )?,
                MixedValue::Dimension(value) => client.buffer(
                    values_to_bytes(&[*value]).as_slice(),
                    BufferType::I64,
                    [],
                    None,
                    device.clone(),
                    None,
                )?,
            });
        }
        for extent in extents {
            buffers.push(client.buffer(
                values_to_bytes(&[*extent]).as_slice(),
                BufferType::I32,
                [],
                None,
                device.clone(),
                None,
            )?);
        }
        let inputs = buffers
            .into_iter()
            .map(|buffer| ExecutionInput { buffer: Arc::new(buffer), donatable: false })
            .collect::<Vec<_>>();
        let outputs = executable
            .execute(
                vec![ExecutionDeviceInputs { inputs: &inputs, ..Default::default() }],
                Vec::new(),
                0,
                None,
                Some(file!()),
                None,
                None,
                &[],
            )?
            .block_until_ready()?
            .remove(0)
            .outputs;
        outputs
            .iter()
            .enumerate()
            .map(|(index, output)| {
                let bytes = output
                    .copy_to_host(Some(ryft_pjrt::Layout::dense_major_to_minor(output.dimensions()?.len())))?
                    .r#await()?;
                let data_type = output_types.get(index).map(|output_type| output_type.data_type());
                if data_type.is_none_or(|data_type| data_type != DataType::F64 && data_type != DataType::I32) {
                    // Dimension outputs and the trailing hidden runtime extents are `i64` scalars.
                    return Ok(MixedValue::Dimension(values_from_bytes::<i64>(bytes.as_slice())[0]));
                }
                // A populated dynamic result is copied back as its compact logical elements, while an empty logical
                // result may still carry its backing allocation and trailing size metadata, which are ignored.
                let logical = output.unpadded_dimensions()?.to_vec();
                let element_count = logical.iter().product::<u64>() as usize;
                if data_type == Some(DataType::I32) {
                    let elements = values_from_bytes::<i32>(&bytes[..element_count * size_of::<i32>()]);
                    return Ok(MixedValue::Integers(elements, logical));
                }
                let elements = values_from_bytes::<f64>(&bytes[..element_count * size_of::<f64>()]);
                Ok(MixedValue::Array(elements, logical))
            })
            .collect()
    }

    /// Plain program that returns its single input of `input_type` unchanged.
    pub(crate) fn identity_branch(input_type: ArrayType) -> PlainProgram {
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(input_type);
        builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Plain program that returns the negation of its single input of `input_type`.
    pub(crate) fn neg_branch(input_type: ArrayType) -> PlainProgram {
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(input_type);
        let output = builder.add_instruction(NegOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Lifts an array-only test fixture into the production array IR XLA program representation.
    pub(crate) fn unproject_plain_program(program: PlainProgram) -> FlatProgram {
        program.into_unprojected::<Constant, Operation>().unwrap()
    }

    /// Stages one `jit_call` to `callee`, which is interned as a shared callee root region, over `inputs` in `builder`.
    pub(crate) fn add_jit_call(
        builder: &mut ArrayIrProgramBuilder,
        callee: &Arc<FlatProgram>,
        inputs: Vec<AtomId>,
    ) -> AtomId {
        let callee_region = builder.intern_callee(callee, None).unwrap();
        builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee_region], inputs, None)
            .unwrap()[0]
    }

    /// Wraps one flat vector program in a replicated manual `shard_map` and lowers the enclosing program.
    pub(crate) fn lower_replicated_shard_map_body(
        body_program: FlatProgram,
        device_count: usize,
    ) -> Result<String, Error> {
        let vector_type = test_vector_type(4);
        let mesh = test_manual_mesh("x", device_count);
        let sharding = Sharding::replicated(mesh.clone(), 1);
        let operation = ShardMapOperation::from_program(
            &body_program,
            vec![vector_type.clone().into()],
            ShardMapBoundary::new(mesh, vec![sharding.clone()], vec![sharding], vec!["x".to_string()]).unwrap(),
        )
        .unwrap();
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(vector_type.clone().into());
        let body = builder.import_program(body_program);
        let output =
            builder.add_instruction(Operation::ShardMap(Box::new(operation)), vec![body], vec![input], None)?[0];
        let program =
            builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])?;
        let input_types = vec![vector_type.clone()];
        let output_types = vec![vector_type];
        to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None)
    }

    /// Builds a one-instruction program applying the given single-operand operation to an input of the given data
    /// type and dimensions, and returns the result of lowering it to a rendered StableHLO module.
    pub(crate) fn lowered_unary_module(
        operation: ArrayOperation<ArrayConstant>,
        data_type: DataType,
        dimensions: Vec<usize>,
    ) -> Result<String, Error> {
        let input_type = ArrayType::new(data_type, Shape::new(dimensions.into_iter().map(Dimension::Static).collect()));
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(input_type);
        let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        to_mlir_module_for_plain_program(&program, "main")
    }

    /// Builds a one-instruction program reducing an input of the given data type and dimensions, and returns the
    /// result of lowering it to a rendered StableHLO module.
    pub(crate) fn lowered_reduce_module(
        data_type: DataType,
        kind: ReductionKind,
        axes: Vec<usize>,
        dimensions: Vec<usize>,
    ) -> Result<String, Error> {
        lowered_unary_module(ArrayOperation::Reduce(ReduceOperation::new(axes, kind)), data_type, dimensions)
    }

    /// Operation that [`lower_unreduced_application`] applies both to values with reduction state, which it stages and
    /// executes on two devices, and to the values without reduction state of each device contribution.
    pub(crate) trait UnreducedApplication {
        /// Applies this operation to `inputs`.
        fn apply<V: ArrayOperations>(&self, inputs: &[V]) -> Result<V, ProgramError>;
    }

    /// Input of [`lower_unreduced_application`], with its type over a mesh with the axis `x` of size two and the bytes
    /// of the contribution of each of the two devices along that axis.
    pub(crate) struct UnreducedInput {
        pub(crate) r#type: ArrayType,
        pub(crate) contributions: [Vec<u8>; 2],
    }

    impl UnreducedInput {
        /// Creates an input of `data_type` and `dimensions` over the explicit mesh axis `x` of size two, whose sharding
        /// is replicated and unreduced over `x` with `kind` when `kind` is present, and whose devices contribute the
        /// `values` of `contributions`.
        pub(crate) fn new<T: Copy>(
            data_type: DataType,
            dimensions: &[usize],
            kind: Option<UnreducedKind>,
            contributions: [&[T]; 2],
        ) -> Self {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
            let mut sharding = Sharding::replicated(mesh, dimensions.len());
            if let Some(kind) = kind {
                sharding = sharding.with_unreduced_axes(["x"]).unwrap().with_unreduced_kind(kind).unwrap();
            }
            let r#type = ArrayType::new_static(data_type, dimensions.to_vec()).with_sharding(sharding).unwrap();
            Self { r#type, contributions: contributions.map(values_to_bytes) }
        }

        /// Returns this input with its sharding reduced over `x`, which every device must then contribute equally.
        pub(crate) fn reduced(mut self) -> Self {
            let sharding = self.r#type.sharding().unwrap().clone().with_reduced_axes(["x"]).unwrap();
            self.r#type = self.r#type.with_sharding(sharding).unwrap();
            self
        }
    }

    /// Lowering and executions of [`lower_unreduced_application`].
    pub(crate) struct UnreducedLowering {
        /// Lowered module of the operation at the top level of a compiled function.
        pub(crate) module: String,

        /// Output type and shard bytes of each execution, one line per context.
        pub(crate) outcomes: String,

        /// Expected [`Self::outcomes`], with the type that staging infers and, for each device, the bytes of the
        /// operation applied to the device contributions without reduction state.
        pub(crate) expected_outcomes: String,
    }

    /// Stages `application` of `inputs` on a two-device CPU mesh at the top level of a compiled function, in the body of
    /// a `scan`, and in a called compiled function, executes each of them, and applies `application` to the values
    /// without reduction state of the contributions of each device on one device, so that callers compare each device
    /// shard with the same operation on plain values. Returns the error message of staging, lowering, compilation, or
    /// execution when one fails (e.g., when the core type rule of the operation rejects the inputs).
    pub(crate) fn lower_unreduced_application<A: UnreducedApplication>(
        application: &A,
        inputs: &[UnreducedInput],
    ) -> Result<UnreducedLowering, String> {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let pjrt_devices = client.addressable_devices().unwrap();
        let devices = pjrt_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let logical_mesh = inputs[0].r#type.sharding().unwrap().mesh().clone();
        let mesh = DeviceMesh::new(logical_mesh, devices.clone()).unwrap();
        let single_device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![devices[0]],
        )
        .unwrap();
        let types = inputs.iter().map(|input| input.r#type.clone()).collect::<Vec<_>>();
        let error = RefCell::new(None::<String>);
        let apply = |inputs: Vec<CompileTracer>| match application.apply(&inputs) {
            Ok(output) => output,
            Err(staging_error) => {
                error.borrow_mut().get_or_insert(staging_error.to_string());
                inputs[0].clone()
            }
        };
        let staged = crate::stage::<_, Vec<ArrayType>, ArrayType>(apply, types.clone(), Options::new(mesh.clone()));
        if let Some(error) = error.borrow_mut().take() {
            return Err(error);
        }
        let lowered = staged.map_err(|error| error.to_string())?.into_inner().lower(&session);
        let module = lowered.map_err(|error| error.to_string())?.lowered_program().stable_hlo().to_string();
        let top_level: CompiledFunction<Vec<ArrayType>, ArrayType> =
            compile(apply, types.clone(), &session, mesh.clone()).map_err(|error| error.to_string())?;
        let output_type = <&ArrayType>::try_from(&top_level.output_types()[0]).unwrap().clone();
        let scanned: CompiledFunction<Vec<ArrayType>, ArrayType> = compile(
            |inputs: Vec<CompileTracer>| {
                let carry = inputs.into_iter().map(CompileTracer::into_value).collect::<Vec<_>>();
                let (_, outputs) = scan(
                    |_, carry: Vec<_>, ()| {
                        let inputs = carry
                            .iter()
                            .map(|input| ValueProjection::<ArrayType>::into_projected(input.clone()))
                            .collect::<Result<Vec<_>, _>>()?;
                        let output = application.apply(&inputs)?;
                        Ok((carry, output.into_value()))
                    },
                    carry,
                    (),
                    ScanOptions::new().with_iteration_count(1usize),
                )
                .unwrap();
                let outputs = ValueProjection::<ArrayType>::into_projected(outputs).unwrap();
                outputs.reshape(output_type.shape().clone()).unwrap()
            },
            types.clone(),
            &session,
            mesh.clone(),
        )
        .map_err(|error| error.to_string())?;
        let called: CompiledFunction<Vec<ArrayType>, ArrayType> =
            compile(|inputs| top_level.call(inputs).unwrap(), types, &session, mesh.clone())
                .map_err(|error| error.to_string())?;
        let arrays = inputs
            .iter()
            .map(|input| {
                let shards =
                    devices.iter().zip(&input.contributions).map(|(device, bytes)| (device.id(), bytes.as_slice()));
                crate::Array::from_host_shards(&session, input.r#type.clone(), mesh.clone(), None, shards)
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let expected_shards = (0..2)
            .map(|device| {
                let plain = inputs
                    .iter()
                    .map(|input| {
                        let dimensions = input.r#type.static_shape().unwrap().dimensions().to_vec();
                        let bytes = input.contributions[device].as_slice();
                        replicated_array(&session, &single_device_mesh, input.r#type.data_type(), &dimensions, bytes)
                    })
                    .collect::<Vec<_>>();
                application.apply(&plain).unwrap().copy_to_host_bytes().unwrap()
            })
            .collect::<Vec<_>>();
        let mut outcomes = Vec::new();
        let mut expected_outcomes = Vec::new();
        for (context, function) in [("top level", &top_level), ("scan body", &scanned), ("called function", &called)] {
            let output = function.executable().execute(&session, arrays.clone()).map_err(|error| error.to_string())?;
            let shards = output.shards().iter().map(|shard| shard.copy_to_host_bytes().unwrap()).collect::<Vec<_>>();
            outcomes.push(format!("{context}: {} {shards:?}", output.r#type()));
            expected_outcomes.push(format!("{context}: {output_type} {expected_shards:?}"));
        }
        Ok(UnreducedLowering { module, outcomes: outcomes.join("\n"), expected_outcomes: expected_outcomes.join("\n") })
    }

    /// Computes `f(x, y) = x * y + sin(x)` for the pair `inputs` of `(x, y)`.
    fn scalar_bilinear_sin<T>(inputs: (T, T)) -> T
    where
        T: Clone + Sin + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    {
        inputs.0.clone() * inputs.1 + inputs.0.sin().unwrap()
    }

    /// Computes `f(x) = x^4 + sin(x)`.
    fn scalar_quartic_plus_sin<T>(x: T) -> T
    where
        T: Clone + Sin + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    {
        x.clone() * x.clone() * x.clone() * x.clone() + x.sin().unwrap()
    }

    /// Eager context over reference backend arrays that stages the scalar programs of the plain program lowering tests.
    static TEST_ARRAY_DOMAIN: EagerContext<Array, ArrayOperation<Array>> =
        EagerContext::<Array, ArrayOperation<Array>>::new();

    /// Copies the exact raw storage bytes from the MLIR dense attribute built for `literal`.
    fn literal_dense_bytes(literal: &Array, byte_count: usize) -> Vec<u8> {
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = lower_tensor_type(literal.r#type().as_ref(), &context, location).unwrap();
        let attribute = literal.to_dense_elements_attribute(tensor_type, &context).unwrap();
        unsafe { std::slice::from_raw_parts(attribute.raw_data().cast::<u8>(), byte_count).to_vec() }
    }

    /// Builds a region over two inputs of `scalar_type` (i.e., a residual and a linear input) whose output is the
    /// product of its inputs, and that also runs the instructions that `extend` adds to it when `extended` is set.
    fn scalar_linear_region(
        scalar_type: &ArrayType,
        extended: bool,
        extend: impl Fn(&mut ArrayIrProgramBuilder, AtomId, AtomId) -> AtomId,
    ) -> FlatProgram {
        let scalar_type = Type::Array(scalar_type.clone());
        let mut builder = ArrayIrProgramBuilder::new();
        let residual = builder.add_input(scalar_type.clone());
        let linear = builder.add_input(scalar_type);
        let mut output =
            builder.add_instruction(MulOperation::new(), Vec::new(), vec![residual, linear], None).unwrap()[0];
        if extended {
            output = extend(&mut builder, residual, output);
        }
        builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap()
    }

    /// Builds a program over the two inputs of the `forward` region that calls a `linear_call` of the `forward` and
    /// `transpose` regions, the second of which is a dormant rule region that lowering never emits.
    fn linear_call_program(forward: &FlatProgram, transpose: &FlatProgram) -> FlatProgram {
        let scalar_type = forward.input_types()[0].clone();
        let mut builder = ArrayIrProgramBuilder::new();
        let forward = builder.import_region(forward.entry_region_ref());
        let transpose = builder.import_region(transpose.entry_region_ref());
        let residual = builder.add_input(scalar_type.clone());
        let linear = builder.add_input(scalar_type);
        let output = builder
            .add_instruction(
                Operation::LinearCall(LinearCallOperation::new(1)),
                vec![forward, transpose],
                vec![residual, linear],
                None,
            )
            .unwrap()[0];
        builder.build(vec![output], vec![Placeholder; 2], vec![Placeholder]).unwrap()
    }

    /// Extends a [`scalar_linear_region`] with a reference that is allocated with its `value`, overwritten with its
    /// `residual`, and frozen into its output.
    fn add_reference_state(builder: &mut ArrayIrProgramBuilder, residual: AtomId, value: AtomId) -> AtomId {
        let reference = builder
            .add_instruction(Operation::ReferenceNew(ReferenceNewOperation::new()), Vec::new(), vec![value], None)
            .unwrap()[0];
        builder
            .add_instruction(
                Operation::ReferenceWrite(ReferenceWriteOperation::new()),
                Vec::new(),
                vec![reference, residual],
                None,
            )
            .unwrap();
        builder
            .add_instruction(
                Operation::ReferenceFreeze(ReferenceFreezeOperation::new()),
                Vec::new(),
                vec![reference],
                None,
            )
            .unwrap()[0]
    }

    #[test]
    fn test_contains_unresolved_state_ignores_dormant_regions() {
        let stateful =
            |extended| scalar_linear_region(&ArrayType::scalar(DataType::F32), extended, add_reference_state);

        // State in the dormant transpose region of a `linear_call` is never lowered, unlike state in its forward region.
        assert!(!contains_unresolved_state(&linear_call_program(&stateful(false), &stateful(true))));
        assert!(contains_unresolved_state(&linear_call_program(&stateful(true), &stateful(false))));
    }

    #[test]
    fn test_contains_unresolved_references_ignores_dormant_regions() {
        let stateful =
            |extended| scalar_linear_region(&ArrayType::scalar(DataType::F32), extended, add_reference_state);

        // References in the dormant transpose region of a `linear_call` are never lowered, unlike references in its
        // forward region.
        assert!(!contains_unresolved_references(&linear_call_program(&stateful(false), &stateful(true))));
        assert!(contains_unresolved_references(&linear_call_program(&stateful(true), &stateful(false))));
    }

    #[test]
    fn test_collective_lowering_state_next_channel_id() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));

        // Two manual regions in one module each emit a channeled `all_reduce`. The channel allocator of the module
        // hands out distinct handles (i.e., 1 and 2), which XLA requires across a module.
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = mesh.clone();
                let sharding = sharding.clone();
                move |x: ArrayTracer| {
                    let first = shard_map(
                        |local_x: ArrayTracer| local_x.parallel_reduce(ReductionKind::Sum, "x"),
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                        ShardMapOptions::new(),
                    )
                    .unwrap();
                    shard_map(
                        |local_x: ArrayTracer| local_x.parallel_reduce(ReductionKind::Sum, "x"),
                        first,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                        ShardMapOptions::new(),
                    )
                    .unwrap()
                }
            },
            global_input_type,
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %2 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %3 : tensor<f32>
                      }) : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %2 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %2 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %3 : tensor<f32>
                      }) : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %2 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %1 : tensor<8xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_collective_lowering_state_checked_collective_mesh() {
        let outer_mesh = test_manual_mesh("x", 2);
        let inner_mesh = test_manual_mesh("x", 3);
        let outer = ShardMapBoundary::new(outer_mesh.clone(), Vec::new(), Vec::new(), Vec::new()).unwrap();
        let inner = ShardMapBoundary::new(inner_mesh.clone(), Vec::new(), Vec::new(), Vec::new()).unwrap();
        let state = CollectiveLoweringState::new().enter_manual_region(outer).enter_manual_region(inner);
        assert_eq!(state.checked_collective_mesh("parallel_permute", "x", 3, None).unwrap(), &inner_mesh);
        assert!(matches!(
            state.checked_collective_mesh("parallel_permute", "x", 3, Some(&outer_mesh)),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::MalformedProgram(message))
                    if *message == "`parallel_permute` operation mesh does not match the enclosing manual axis `x` mesh",
            ),
        ));
        assert!(matches!(
            state.checked_collective_mesh("parallel_permute", "x", 2, Some(&outer_mesh)),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::MalformedProgram(message))
                    if *message == "collective over axis `x` records size 2, but the enclosing mesh axis has size 3",
            ),
        ));

        // A payload mesh that retypes only a free axis of the binder mesh (e.g., inside `auto_axes`) is accepted and
        // returned, while a payload mesh that retypes a manual axis or has another topology is still rejected.
        let binder_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let binder = ShardMapBoundary::new(binder_mesh.clone(), Vec::new(), Vec::new(), Vec::new()).unwrap();
        let state = CollectiveLoweringState::new().enter_manual_region(binder);
        let free_axis_retyped_mesh = binder_mesh.with_axis_types(["y"], MeshAxisType::Auto).unwrap();
        let manual_axis_retyped_mesh = binder_mesh.with_axis_types(["x"], MeshAxisType::Explicit).unwrap();
        let other_topology_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 4, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        assert_eq!(state.checked_collective_mesh("parallel_permute", "x", 2, None).unwrap(), &binder_mesh);
        assert_eq!(
            state.checked_collective_mesh("parallel_permute", "x", 2, Some(&free_axis_retyped_mesh)).unwrap(),
            &free_axis_retyped_mesh,
        );
        for payload_mesh in [&manual_axis_retyped_mesh, &other_topology_mesh] {
            assert!(matches!(
                state.checked_collective_mesh("parallel_permute", "x", 2, Some(payload_mesh)),
                Err(Error::Core(core_error)) if matches!(
                    core_error.as_ref(),
                    CoreError::Program(ProgramError::MalformedProgram(message))
                        if *message == "`parallel_permute` operation mesh does not match the enclosing manual axis `x` mesh",
                ),
            ));
        }
    }

    #[test]
    fn test_collective_lowering_state_validate_manual_collective_uses_owner_policy() {
        let mesh = test_manual_mesh("x", 2);
        let boundary = ShardMapBoundary::new(mesh.clone(), Vec::new(), Vec::new(), Vec::new()).unwrap();
        let checked = CollectiveLoweringState::new().enter_manual_region(boundary.clone());
        checked
            .validate_manual_collective("parallel_permute", "x", 2, None, &[1usize], |inference_mesh| {
                assert_eq!(inference_mesh, Some(&mesh));
                Ok(vec![1])
            })
            .unwrap();
        let unchecked = CollectiveLoweringState::new().enter_manual_region(boundary.with_check_manual_variation(false));
        unchecked
            .validate_manual_collective("parallel_permute", "x", 2, None, &[1usize], |inference_mesh| {
                assert_eq!(inference_mesh, None);
                Ok(vec![1])
            })
            .unwrap();
    }

    #[test]
    fn test_collective_lowering_state_validate_manual_collective_uses_each_axis_owner() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let outer = ShardMapBoundary::new(mesh.clone(), Vec::new(), Vec::new(), vec!["x".to_string()]).unwrap();
        let inner = ShardMapBoundary::new(mesh.clone(), Vec::new(), Vec::new(), vec!["y".to_string()]).unwrap();
        let state = CollectiveLoweringState::new()
            .enter_manual_region(outer.clone())
            .enter_manual_region(inner.clone().with_check_manual_variation(false));
        state
            .validate_manual_collective("parallel_permute", "x", 2, None, &[1usize], |inference_mesh| {
                assert_eq!(inference_mesh, Some(&mesh));
                Ok(vec![1])
            })
            .unwrap();
        state
            .validate_manual_collective("parallel_permute", "y", 2, None, &[1usize], |inference_mesh| {
                assert_eq!(inference_mesh, None);
                Ok(vec![1])
            })
            .unwrap();

        let state = CollectiveLoweringState::new()
            .enter_manual_region(outer.with_check_manual_variation(false))
            .enter_manual_region(inner);
        state
            .validate_manual_collective("parallel_permute", "x", 2, None, &[1usize], |inference_mesh| {
                assert_eq!(inference_mesh, None);
                Ok(vec![1])
            })
            .unwrap();
        state
            .validate_manual_collective("parallel_permute", "y", 2, None, &[1usize], |inference_mesh| {
                assert_eq!(inference_mesh, Some(&mesh));
                Ok(vec![1])
            })
            .unwrap();
    }

    #[test]
    fn test_collective_lowering_state_validate_manual_collective_rejects_tracked_unchecked_payload() {
        let mesh = test_manual_mesh("x", 2);
        let boundary = ShardMapBoundary::new(mesh.clone(), Vec::new(), Vec::new(), Vec::new())
            .unwrap()
            .with_check_manual_variation(false);
        let state = CollectiveLoweringState::new().enter_manual_region(boundary);
        let result = state.validate_manual_collective("parallel_permute", "x", 2, Some(&mesh), &[1usize], |_| {
            panic!("tracked payload must be rejected before ordinary inference");
        });
        assert!(matches!(
            result,
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::MalformedProgram(message))
                    if *message == "`parallel_permute` cannot use tracked mesh metadata inside an unchecked manual binding",
            ),
        ));
    }

    #[test]
    fn test_collective_lowering_state_validate_manual_collective_retains_unchecked_validation() {
        let mesh = test_manual_mesh("x", 2);
        let boundary = ShardMapBoundary::new(mesh.clone(), Vec::new(), Vec::new(), Vec::new())
            .unwrap()
            .with_check_manual_variation(false);
        let state = CollectiveLoweringState::new().enter_manual_region(boundary);
        let wrong_size = state.validate_manual_collective("parallel_permute", "x", 3, None, &[1usize], |_| {
            panic!("axis size must be validated before ordinary inference");
        });
        assert!(matches!(
            wrong_size,
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::MalformedProgram(message))
                    if *message == "collective over axis `x` records size 3, but the enclosing mesh axis has size 2",
            ),
        ));

        let other_mesh = test_manual_mesh("x", 3);
        let wrong_mesh =
            state.validate_manual_collective("parallel_permute", "x", 2, Some(&other_mesh), &[1usize], |_| {
                panic!("payload mesh must be validated before ordinary inference");
            });
        assert!(matches!(
            wrong_mesh,
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::MalformedProgram(message))
                    if *message == "`parallel_permute` operation mesh does not match the enclosing manual axis `x` mesh",
            ),
        ));

        let wrong_outputs =
            state.validate_manual_collective("parallel_permute", "x", 2, None, &[1usize], |_| Ok(vec![2usize]));
        assert!(matches!(
            wrong_outputs,
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::MalformedProgram(message))
                    if *message == "`parallel_permute` output types do not match the enclosing manual mesh contract",
            ),
        ));
    }

    #[test]
    fn test_to_mlir_module_for_program() {
        let global_input_type = test_vector_type(8);
        let mesh = test_manual_mesh("x", 4);
        let module = shard_map_module::<_, ArrayType, ArrayType>(
            |_, x| Ok(x.clone() + x),
            global_input_type,
            mesh.clone(),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap(),
            ShardMapOptions::new(),
        )
        .unwrap();

        assert_eq!(
            module,
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
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_renders_provenance_locations() {
        let global_input_type = test_vector_type(8);
        let mesh = test_manual_mesh("x", 4);
        let module = shard_map_module::<_, ArrayType, ArrayType>(
            |_, x| {
                let context = x.value().context().clone();
                // One instruction staged under nested scopes, one under a fused origin, and one with unknown
                // provenance, covering every provenance shape in one lowered module.
                let scoped = context.invoke_with_provenance_scope(ProvenanceScope::new("outer"), || {
                    context.invoke_with_provenance_scope(ProvenanceScope::new("inner"), || x.clone() + x.clone())
                });
                let fused = Provenance::fused([
                    Provenance::scope(ProvenanceScope::new("a"), Provenance::unknown()),
                    Provenance::scope(ProvenanceScope::new("b"), Provenance::unknown()),
                ]);
                let product = context.invoke_with_provenance_origin(fused, || scoped.clone() * scoped);
                Ok(product + x)
            },
            global_input_type,
            mesh.clone(),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap(),
            ShardMapOptions::new(),
        )
        .unwrap();

        // The scoped addition carries the nested chain of name locations, the fused multiplication carries a fused location
        // without metadata, the addition with unknown provenance keeps the unknown base location, and the module and
        // function scaffolding keeps the base location. Locations render because modules that carry provenance serialize
        // with debug information enabled, in the parsable form that is not pretty printed.
        assert_eq!(
            module,
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
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_renders_constants_and_supported_ops() {
        let global_input_type = test_matrix_type(4, 4);
        let mesh = test_manual_mesh("x", 2);
        let module = shard_map_module::<_, ArrayType, ArrayType>(
            |_, x| {
                let product = x.transpose(vec![1, 0])?.dot(&x, &DotDimensionNumbers::matmul())?;
                let waveform = (-product).cos()?.sin()?;
                Ok((waveform.clone() * waveform.one_like()?) + waveform.zero_like()?)
            },
            global_input_type,
            mesh.clone(),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap(),
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap(),
            ShardMapOptions::new(),
        )
        .unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}, {}]>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x"} (%arg1: tensor<2x4xf32>) {
                      %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<2x4xf32>) -> tensor<4x2xf32>
                      %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
                      %3 = stablehlo.negate %2 : tensor<4x4xf32>
                      %4 = stablehlo.cosine %3 : tensor<4x4xf32>
                      %5 = stablehlo.sine %4 : tensor<4x4xf32>
                      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                      %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x4xf32>
                      %7 = sdy.sharding_constraint %6 <@mesh, [{}, {}]> : tensor<4x4xf32>
                      %8 = stablehlo.multiply %5, %7 : tensor<4x4xf32>
                      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                      %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4x4xf32>
                      %10 = sdy.sharding_constraint %9 <@mesh, [{}, {}]> : tensor<4x4xf32>
                      %11 = stablehlo.add %8, %10 : tensor<4x4xf32>
                      sdy.return %11 : tensor<4x4xf32>
                    } : (tensor<4x4xf32>) -> tensor<8x4xf32>
                    return %0 : tensor<8x4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_preserves_boundary_layouts() {
        let array_type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])));
        let zero_type = ArrayType::new_static(DataType::Zero, [2, 3]);
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(array_type.clone().into());
        let zero = builder.add_input(zero_type.clone().into());
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![zero, input], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            to_mlir_module_for_program(
                &program,
                &[],
                &vec![array_type.clone(), zero_type.clone()],
                &vec![zero_type, array_type],
                "main",
                None,
                None,
            )
            .unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<2x3xf32> {mhlo.layout_mode = \"{0,1}\"}) \
                      -> (tensor<2x3xf32> {mhlo.layout_mode = \"{0,1}\"}) {
                    %c = stablehlo.constant dense<false> : tensor<i1>
                    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<2x3xi1>
                    return %arg0 : tensor<2x3xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_captures_as_hidden_arguments() {
        let array_type = test_vector_type(4);
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(array_type.clone().into());
        let capture = builder.add_constant(Constant::Captured(CaptureReference::new(0, array_type.clone().into())));
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, capture], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
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

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg1, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_immediate_dimension_constants_in_place() {
        // An immediate dimension constant carries its own extent from the host, so unlike a capture reference, it
        // materializes as a scalar `i64` `stablehlo.constant` in the function body instead of consuming a hidden capture
        // argument. That is exactly what makes first-class extents usable where no capture table is reachable, such as
        // inside a `shard_map` manual region.
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let mut builder = ArrayIrProgramBuilder::new();
        let extent = builder.add_constant(Constant::Dimension(DimensionValue::new(extent_type, 4).unwrap()));
        let output = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![extent], None).unwrap()[0];
        let program =
            builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], Vec::new(), vec![Placeholder]).unwrap();
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

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main() -> tensor<i64> {
                    %c = stablehlo.constant dense<4> : tensor<i64>
                    return %c : tensor<i64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_lowers_capture_output_as_a_hidden_argument() {
        let array_type = test_vector_type(4);
        let mut builder = ArrayIrProgramBuilder::new();
        let output = builder.add_constant(Constant::Captured(CaptureReference::new(0, array_type.clone().into())));
        let program =
            builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], Vec::new(), vec![Placeholder]).unwrap();
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

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    return %arg0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_forwards_capture_into_nested_regions() {
        let array_type = test_vector_type(4);
        let branch = || {
            let mut builder = ArrayIrProgramBuilder::new();
            let output = builder.add_constant(Constant::Captured(CaptureReference::new(0, array_type.clone().into())));
            builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], Vec::new(), vec![Placeholder]).unwrap()
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let true_region = builder.import_program(branch());
        let false_region = builder.import_program(branch());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let output = builder
            .add_instruction(
                Operation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
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

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<i1>) -> tensor<4xf32> {
                    %0 = "stablehlo.if"(%arg1) ({
                      stablehlo.return %arg0 : tensor<4xf32>
                    }, {
                      stablehlo.return %arg0 : tensor<4xf32>
                    }) : (tensor<i1>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_erases_static_zero_space_boundary() {
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(zero_type.clone().into());
        let program: FlatProgram = builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();

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
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(zero_type.clone().into());
        let output = builder
            .add_instruction(PrintOperation::<ArrayType>::new("zero"), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program: FlatProgram = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        let stablehlo = to_mlir_module_for_program(&program, &[], &zero_type, &zero_type, "main", None, None).unwrap();

        assert_eq!(
            stablehlo,
            indoc! {"
                module {
                  func.func @main() {
                    %c = stablehlo.constant dense<false> : tensor<i1>
                    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<3xi1>
                    %1 = stablehlo.after_all  : !stablehlo.token
                    %2 = stablehlo.custom_call @ryft.print(%0, %1) {api_version = 4 : i32, \
                        backend_config = {label = \"zero\", logical_type = \"zero\"}, has_side_effect = true, \
                        operand_layouts = [dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], \
                        result_layouts = [dense<> : tensor<0xindex>]} : (tensor<3xi1>, !stablehlo.token) \
                        -> !stablehlo.token
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_preserves_dynamic_zero_space_shape_carrier() {
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![dynamic_dimension("zero", Some(3))]));
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(zero_type.clone().into());
        let program: FlatProgram = builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();

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
    fn test_to_mlir_module_for_program_isolates_bounded_dynamic_results_behind_one_barrier() {
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("count", Some(4))]));
        let static_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let mut builder = ArrayIrProgramBuilder::new();
        let dynamic = builder.add_input(dynamic_type.clone().into());
        let r#static = builder.add_input(static_type.clone().into());
        let program: FlatProgram =
            builder.build(vec![dynamic, r#static, dynamic], vec![Placeholder; 2], vec![Placeholder; 3]).unwrap();

        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &vec![dynamic_type.clone(), static_type.clone()],
            &vec![dynamic_type.clone(), static_type, dynamic_type],
            "main",
            None,
            None,
        )
        .unwrap();

        // Both dynamic results pass through the barrier, while the static result and the hidden extents bypass it.
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<2xf32>, %arg2: tensor<i32>) -> (tensor<?xf32, #stablehlo.bounds<3>>, tensor<2xf32>, tensor<?xf32, #stablehlo.bounds<3>>, tensor<i64>, tensor<i64>) {
                    %0 = stablehlo.set_dimension_size %arg0, %arg2, dim = 0 : (tensor<3xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<3>>
                    %1:2 = stablehlo.optimization_barrier %0, %0 : tensor<?xf32, #stablehlo.bounds<3>>, tensor<?xf32, #stablehlo.bounds<3>>
                    %2 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xf32, #stablehlo.bounds<3>>) -> tensor<i32>
                    %3 = stablehlo.convert %2 : (tensor<i32>) -> tensor<i64>
                    %4 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xf32, #stablehlo.bounds<3>>) -> tensor<i32>
                    %5 = stablehlo.convert %4 : (tensor<i32>) -> tensor<i64>
                    return %1#0, %arg1, %1#1, %3, %5 : tensor<?xf32, #stablehlo.bounds<3>>, tensor<2xf32>, tensor<?xf32, #stablehlo.bounds<3>>, tensor<i64>, tensor<i64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_declares_the_mesh_of_completed_device_reductions() {
        // A device target that completes a pending reduction over a mesh axis of size one constrains its result over
        // the mesh of its type, which neither the signature nor a payload names, so the module declares the mesh
        // because the lowered constraint refers to it.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let unreduced = Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let input_type = test_vector_type(4).with_sharding(unreduced).unwrap();
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let output = builder
            .add_instruction(PlaceOperation::new(Device::new(0, 0), None), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let output_types = program
            .output_types()
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            to_mlir_module_for_program(&program, &[], &vec![input_type], &output_types, "main", None, None).unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{}], replicated={"x"}> : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_program_constrains_unreduced_outputs() {
        // Shardy rejects an operation whose result does not declare the unreduced axes of its inputs, so every output
        // that is unreduced over an axis that no manual region binds is constrained, also where its placement does not
        // change (e.g., a sum, and a broadcast that gives it a batch dimension).
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = test_vector_type(4)
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let output_type = ArrayType::new_static(DataType::F32, [2, 4])
            .with_sharding(Sharding::replicated(mesh, 2).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let mut builder = ArrayIrProgramBuilder::new();
        let left = builder.add_input(input_type.clone().into());
        let right = builder.add_input(input_type.clone().into());
        let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let output = builder
            .add_instruction(BroadcastOperation::new(output_type.clone(), vec![1]), Vec::new(), vec![sum], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            to_mlir_module_for_program(
                &program,
                &[],
                &vec![input_type.clone(), input_type],
                &vec![output_type],
                "main",
                None,
                None,
            )
            .unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<2x4xf32> {
                    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{}], unreduced={"x"}> : tensor<4xf32>
                    %2 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<4xf32>) -> tensor<2x4xf32>
                    %3 = sdy.sharding_constraint %2 <@mesh, [{}, {}], unreduced={"x"}> : tensor<2x4xf32>
                    return %3 : tensor<2x4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_instruction_constrains_unreduced_intermediates() {
        // The mean of a `bf16` vector converts it to `f32`, reduces it, divides it by the element count, and converts it
        // back.
        struct Mean;

        impl UnreducedApplication for Mean {
            fn apply<V: ArrayOperations>(&self, inputs: &[V]) -> Result<V, ProgramError> {
                inputs[0].reduce_mean(&[0])
            }
        }

        // The `bf16` bits of 1, 2, 3, and 4, and of -1, 0.5, 4, and 2.
        let input = UnreducedInput::new(
            DataType::BF16,
            &[4],
            Some(UnreducedKind::Sum),
            [&[0x3f80u16, 0x4000, 0x4040, 0x4080], &[0xbf80u16, 0x3f00, 0x4080, 0x4000]],
        );
        let lowering = lower_unreduced_application(&Mean, &[input]).unwrap();
        assert_eq!(lowering.outcomes, lowering.expected_outcomes);
        assert_eq!(
            lowering.module,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}], unreduced={"x"}>}) -> (tensor<bf16> {sdy.sharding = #sdy.sharding<@mesh, [], unreduced={"x"}>}) {
                    %0 = stablehlo.convert %arg0 : (tensor<4xbf16>) -> tensor<4xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{}], unreduced={"x"}> : tensor<4xf32>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
                    %3 = sdy.sharding_constraint %2 <@mesh, [], unreduced={"x"}> : tensor<f32>
                    %cst_0 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
                    %4 = stablehlo.divide %3, %cst_0 : tensor<f32>
                    %5 = sdy.sharding_constraint %4 <@mesh, [], unreduced={"x"}> : tensor<f32>
                    %6 = stablehlo.convert %5 : (tensor<f32>) -> tensor<bf16>
                    %7 = sdy.sharding_constraint %6 <@mesh, [], unreduced={"x"}> : tensor<bf16>
                    return %7 : tensor<bf16>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_instruction_constrains_unreduced_intermediates_with_dimension_shardings() {
        // The intermediates of an instruction whose inputs carry a pending reduction and are split along another axis
        // keep that split, which flows forward through the elementwise comparisons and selections of a maximum, and
        // backward from the output through the division of a mean to the sum that it divides, so that none of them is
        // gathered.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let split = Sharding::new(mesh, vec![ShardingDimension::sharded(["y"]), ShardingDimension::Replicated])
            .unwrap()
            .with_unreduced_axes(["x"])
            .unwrap();
        let lower = |kind: UnreducedKind| {
            let sharding = split.clone().with_unreduced_kind(kind).unwrap();
            let input_type = ArrayType::new_static(DataType::F32, [8, 4]).with_sharding(sharding).unwrap();
            let traced: TracedProgram<(ArrayType, ArrayType), ArrayType> = trace(
                |(left, right): (ArrayTracer, ArrayTracer)| match kind {
                    UnreducedKind::Max => left.max(&right).unwrap(),
                    _ => left.reduce_mean(&[1]).unwrap() - right.reduce_mean(&[1]).unwrap(),
                },
                (input_type.clone(), input_type),
            )
            .unwrap();
            traced.to_mlir_module("main").unwrap()
        };
        assert_eq!(
            lower(UnreducedKind::Max),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2, "y"=2]>
                  func.func @main(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced=max{"x"}>}, %arg1: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced=max{"x"}>}) -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced=max{"x"}>}) {
                    %0 = stablehlo.compare GT, %arg0, %arg1, FLOAT : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xi1>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{"y"}, {}], unreduced=max{"x"}> : tensor<8x4xi1>
                    %2 = stablehlo.select %1, %arg0, %arg1 : tensor<8x4xi1>, tensor<8x4xf32>
                    %3 = sdy.sharding_constraint %2 <@mesh, [{"y"}, {}], unreduced=max{"x"}> : tensor<8x4xf32>
                    %4 = stablehlo.negate %arg0 : tensor<8x4xf32>
                    %5 = sdy.sharding_constraint %4 <@mesh, [{"y"}, {}], unreduced=max{"x"}> : tensor<8x4xf32>
                    %6 = stablehlo.compare EQ, %arg0, %arg1, FLOAT : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xi1>
                    %7 = sdy.sharding_constraint %6 <@mesh, [{"y"}, {}], unreduced=max{"x"}> : tensor<8x4xi1>
                    %8 = stablehlo.compare EQ, %arg0, %5, FLOAT : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xi1>
                    %9 = sdy.sharding_constraint %8 <@mesh, [{"y"}, {}], unreduced=max{"x"}> : tensor<8x4xi1>
                    %10 = stablehlo.and %7, %9 : tensor<8x4xi1>
                    %11 = sdy.sharding_constraint %10 <@mesh, [{"y"}, {}], unreduced=max{"x"}> : tensor<8x4xi1>
                    %12 = stablehlo.add %arg0, %arg1 : tensor<8x4xf32>
                    %13 = sdy.sharding_constraint %12 <@mesh, [{"y"}, {}], unreduced=max{"x"}> : tensor<8x4xf32>
                    %14 = stablehlo.select %11, %13, %3 : tensor<8x4xi1>, tensor<8x4xf32>
                    %15 = sdy.sharding_constraint %14 <@mesh, [{"y"}, {}], unreduced=max{"x"}> : tensor<8x4xf32>
                    %16 = stablehlo.compare NE, %arg0, %arg0, FLOAT : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xi1>
                    %17 = sdy.sharding_constraint %16 <@mesh, [{"y"}, {}], unreduced=max{"x"}> : tensor<8x4xi1>
                    %18 = stablehlo.compare NE, %arg1, %arg1, FLOAT : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xi1>
                    %19 = sdy.sharding_constraint %18 <@mesh, [{"y"}, {}], unreduced=max{"x"}> : tensor<8x4xi1>
                    %20 = stablehlo.select %19, %arg1, %15 : tensor<8x4xi1>, tensor<8x4xf32>
                    %21 = sdy.sharding_constraint %20 <@mesh, [{"y"}, {}], unreduced=max{"x"}> : tensor<8x4xf32>
                    %22 = stablehlo.select %17, %arg0, %21 : tensor<8x4xi1>, tensor<8x4xf32>
                    %23 = sdy.sharding_constraint %22 <@mesh, [{"y"}, {}], unreduced=max{"x"}> : tensor<8x4xf32>
                    return %23 : tensor<8x4xf32>
                  }
                }
            "#},
        );
        assert_eq!(
            lower(UnreducedKind::Sum),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2, "y"=2]>
                  func.func @main(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced={"x"}>}, %arg1: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced={"x"}>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}], unreduced={"x"}>}) {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{"y"}], unreduced={"x"}> : tensor<8xf32>
                    %cst_0 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
                    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
                    %3 = stablehlo.divide %1, %2 : tensor<8xf32>
                    %4 = sdy.sharding_constraint %3 <@mesh, [{"y"}], unreduced={"x"}> : tensor<8xf32>
                    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %5 = stablehlo.reduce(%arg1 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
                    %6 = sdy.sharding_constraint %5 <@mesh, [{"y"}], unreduced={"x"}> : tensor<8xf32>
                    %cst_2 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
                    %7 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8xf32>
                    %8 = stablehlo.divide %6, %7 : tensor<8xf32>
                    %9 = sdy.sharding_constraint %8 <@mesh, [{"y"}], unreduced={"x"}> : tensor<8xf32>
                    %10 = stablehlo.subtract %4, %9 : tensor<8xf32>
                    %11 = sdy.sharding_constraint %10 <@mesh, [{"y"}], unreduced={"x"}> : tensor<8xf32>
                    return %11 : tensor<8xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_mlir_module_for_program_omits_shardy_for_targets_without_the_dialect() {
        // A device target that completes a pending reduction over a mesh axis of size one constrains its result, and
        // the signature shardings annotate the entry function. A target whose plugin lacks the Shardy dialect gets
        // neither, nor the mesh that both refer to.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let input_type = test_vector_type(4).with_sharding(unreduced.clone()).unwrap();
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let output = builder
            .add_instruction(PlaceOperation::new(Device::new(0, 0), None), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let output_types = program
            .output_types()
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned().unwrap())
            .collect::<Vec<_>>();
        let lower = |supports_shardy| {
            lower_mlir_module_for_program_with_reference_state(
                &program,
                &[],
                &vec![input_type.clone()],
                &output_types,
                "main",
                Some(std::slice::from_ref(&unreduced)),
                Some(std::slice::from_ref(&replicated)),
                Some("cpu"),
                supports_shardy,
                &[],
                &[],
                RaggedDotLoweringStrategy::default(),
                &[],
            )
            .unwrap()
        };
        let lowered = lower(true);
        assert_eq!(lowered.mesh(), Some(&mesh));
        assert_eq!(
            lowered.into_stable_hlo(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], unreduced={"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], replicated={"x"}>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{}], replicated={"x"}> : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
        let lowered = lower(false);
        assert_eq!(lowered.mesh(), Some(&mesh));
        assert_eq!(
            lowered.into_stable_hlo(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    return %arg0 : tensor<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_remove_shardy_operations() {
        // Constraints are replaced by their operands in nested regions as well, and the mesh declaration is erased.
        let context = MlirContext::new();
        context.load_dialect(DialectHandle::func().unwrap()).unwrap();
        context.load_dialect(DialectHandle::stable_hlo().unwrap()).unwrap();
        context.load_dialect(DialectHandle::shardy().unwrap()).unwrap();
        let module = context
            .parse_module(indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<i32>) -> tensor<4xf32> {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{}], replicated={"x"}> : tensor<4xf32>
                    %1 = "stablehlo.case"(%arg1) ({
                      %2 = sdy.sharding_constraint %0 <@mesh, [{"x"}]> : tensor<4xf32>
                      %3 = stablehlo.negate %2 : tensor<4xf32>
                      stablehlo.return %3 : tensor<4xf32>
                    }, {
                      stablehlo.return %0 : tensor<4xf32>
                    }) : (tensor<i32>) -> tensor<4xf32>
                    return %1 : tensor<4xf32>
                  }
                }
            "#})
            .unwrap();
        remove_shardy_operations(&module).unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<i32>) -> tensor<4xf32> {
                    %0 = "stablehlo.case"(%arg1) ({
                      %1 = stablehlo.negate %arg0 : tensor<4xf32>
                      stablehlo.return %1 : tensor<4xf32>
                    }, {
                      stablehlo.return %arg0 : tensor<4xf32>
                    }) : (tensor<i32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );

        // A manual computation has no equivalent without the dialect.
        let body = unproject_plain_program(neg_branch(test_vector_type(4)));
        let module = context.parse_module(lower_replicated_shard_map_body(body, 1).unwrap()).unwrap();
        assert!(matches!(
            remove_shardy_operations(&module),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { operation, reason }))
                if operation == "sdy.manual_computation"
                    && reason == "the target plugin cannot compile with Shardy",
        ));
    }

    #[test]
    fn test_lower_mlir_module_for_program_rejects_unresolved_reference_state() {
        assert_eq!(token_threaded_effects(EffectClasses::single(EffectClass::OrderedState)).next(), None);

        // The reference scan over the whole artifact catches the reference atoms and intrinsic semantics before token
        // threading, so two representative reference operations are enough to pin the diagnostic.
        let array_type = ArrayType::scalar(DataType::F32);
        let mut builder = crate::operations::ProgramBuilder::new();
        let input = builder.add_input(Type::Array(array_type.clone()));
        let reference = builder
            .add_instruction(Operation::ReferenceNew(ReferenceNewOperation::new()), Vec::new(), vec![input], None)
            .unwrap()[0];
        builder
            .add_instruction(
                Operation::ReferenceWrite(ReferenceWriteOperation::new()),
                Vec::new(),
                vec![reference, input],
                None,
            )
            .unwrap();
        let output = builder
            .add_instruction(
                Operation::ReferenceFreeze(ReferenceFreezeOperation::new()),
                Vec::new(),
                vec![reference],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // The reference scan runs before the generic state scan at the module entry, so a program whose state comes
        // from reference operations receives the dedicated reference diagnostic. The state error remains behind it for
        // future state that does not come from references, and `token_threaded_effects` above pins that `OrderedState`
        // has no token slot either way.
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
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct })) if construct == "program",
        ));
    }

    #[test]
    fn test_lower_mlir_module_for_program_rejects_preserved_reference_entries() {
        let mut builder = ArrayIrProgramBuilder::new();
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference = builder.add_input(Type::Reference(ReferenceType::new(scalar_type.clone())));
        let snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let contract = KernelBoundaryContract::new(vec![Some(KernelParameterAccess::ReadOnly)]);
        validate_kernel_body(program.entry_region_ref(), &contract).unwrap();
        assert!(matches!(
            lower_mlir_module_for_program(
                &program,
                &[],
                &vec![scalar_type.clone()],
                &vec![scalar_type],
                "main",
                None,
                None,
                None,
            ),
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct })) if construct == "program",
        ));
    }

    #[test]
    fn test_lower_mlir_module_for_program_with_reference_state() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let state_type = ArrayType::new_static(DataType::F32, [4]);
        let mut builder = crate::operations::ProgramBuilder::new();
        let _zero = builder.add_input(Type::Array(zero_type.clone()));
        let mutated = builder.add_input(Type::Array(state_type.clone()));
        let _read_only = builder.add_input(Type::Array(state_type.clone()));
        let program: FlatProgram = builder.build(vec![mutated], vec![Placeholder; 3], vec![Placeholder]).unwrap();
        let input_types = vec![zero_type.clone(), state_type.clone(), state_type.clone()];
        let output_types = vec![state_type.clone()];
        let argument_shardings = vec![
            Sharding::replicated(mesh.clone(), 1),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
        ];
        let result_shardings = vec![Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()];
        let states = [
            ExternalReferenceBinding::new(ReferenceSource::Input { index: 1 }, Some(0)),
            ExternalReferenceBinding::new(ReferenceSource::Input { index: 2 }, None),
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
            true,
            &states,
            &[],
            RaggedDotLoweringStrategy::default(),
            &[],
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

        let invalid_state = ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, None);
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
                true,
                std::slice::from_ref(&invalid_state),
                &[],
                RaggedDotLoweringStrategy::default(),
                &[],
            ),
            Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi { message }))
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
                    true,
                    &states,
                    &[],
                    RaggedDotLoweringStrategy::default(),
                    &[],
                ),
                Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi { message }))
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
                true,
                &states,
                &[],
                RaggedDotLoweringStrategy::default(),
                &[],
            ),
            Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi { message }))
                if message == "state input 1 and output 0 must use the same sharding",
        ));
    }

    #[test]
    fn test_lower_mlir_module_for_program_with_reference_state_rejects_bounded_dynamic_aliases() {
        let extent = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let state_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 1);
        let mut builder = crate::operations::ProgramBuilder::new();
        let state = builder.add_input(Type::Array(state_type.clone()));
        let program: FlatProgram = builder.build(vec![state], vec![Placeholder], vec![Placeholder]).unwrap();
        let reference_state = ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(0));

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
                true,
                std::slice::from_ref(&reference_state),
                &[],
                RaggedDotLoweringStrategy::default(),
                &[],
            ),
            Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi { message }))
                if message == "state input 0 and output 0 must be static because bounded-dynamic mutation alias \
                               compatibility is unsupported",
        ));
    }

    #[test]
    fn test_lower_mlir_module_for_program_with_reference_state_aliases_donated_inputs() {
        let vector_type = ArrayType::new_static(DataType::F32, [4]);
        let pair_type = ArrayType::new_static(DataType::F32, [2]);
        let mut builder = crate::operations::ProgramBuilder::new();
        let first = builder.add_input(Type::Array(vector_type.clone()));
        let second = builder.add_input(Type::Array(vector_type.clone()));
        let third = builder.add_input(Type::Array(pair_type.clone()));
        let program: FlatProgram =
            builder.build(vec![second, first, third], vec![Placeholder; 3], vec![Placeholder; 3]).unwrap();
        let input_types = vec![vector_type.clone(), vector_type.clone(), pair_type.clone()];
        let output_types = vec![vector_type.clone(), vector_type.clone(), pair_type.clone()];
        let lower = |shardings: Option<&[Sharding]>, donated_inputs: &[bool], open_results: &[bool]| {
            lower_mlir_module_for_program_with_reference_state(
                &program,
                &[],
                &input_types,
                &output_types,
                "main",
                shardings,
                shardings,
                None,
                true,
                &[],
                donated_inputs,
                RaggedDotLoweringStrategy::default(),
                open_results,
            )
        };
        let signature = |lowered: LoweredModule| {
            let line = lowered.stable_hlo.lines().find(|line| line.contains("func.func @main")).unwrap();
            line.trim().to_string()
        };

        // Each donated input is paired with the first unpaired output of its type, and other inputs with none.
        assert_eq!(
            signature(lower(None, &[true, true, false], &[]).unwrap()),
            concat!(
                "func.func @main(%arg0: tensor<4xf32> {tf.aliasing_output = 0 : i64}, ",
                "%arg1: tensor<4xf32> {tf.aliasing_output = 1 : i64}, %arg2: tensor<2xf32>) -> ",
                "(tensor<4xf32>, tensor<4xf32>, tensor<2xf32>) {",
            ),
        );
        assert_eq!(
            signature(lower(None, &[false, true, true], &[]).unwrap()),
            concat!(
                "func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32> {tf.aliasing_output = 0 : i64}, ",
                "%arg2: tensor<2xf32> {tf.aliasing_output = 2 : i64}) -> ",
                "(tensor<4xf32>, tensor<4xf32>, tensor<2xf32>) {",
            ),
        );
        assert!(!signature(lower(None, &[], &[]).unwrap()).contains("tf.aliasing_output"));

        // An open output is only paired on a single device, where the compiler cannot place it differently.
        let shardings = |device_count: usize| {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", device_count, MeshAxisType::Auto).unwrap()]).unwrap();
            vec![Sharding::replicated(mesh, 1); 3]
        };
        let open_results = [true, false, false];
        let single_device = signature(lower(Some(&shardings(1)), &[true, false, false], &open_results).unwrap());
        assert!(single_device.contains("tf.aliasing_output = 0 : i64"));
        let two_devices = signature(lower(Some(&shardings(2)), &[true, false, false], &open_results).unwrap());
        assert!(two_devices.contains("tf.aliasing_output = 1 : i64"));

        assert!(matches!(
            lower(None, &[true], &[]),
            Err(Error::Execution(ExecutionError::DonationFlagCountMismatch { expected_count: 3, actual_count: 1 })),
        ));
    }

    #[test]
    fn test_entry_alias_attribute_executes_in_place_on_cpu() {
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
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32> {mhlo.sharding = "{replicated}", tf.aliasing_output = 0 : i64}) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );

        let client = cpu_client(1);
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_compilation_options(client, None))
            .unwrap();
        let device = executable.addressable_devices().unwrap().remove(0);
        let input = client
            .buffer(values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(), BufferType::F32, [4], None, device, None)
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
                &[],
            )
            .unwrap();
        let mut device_outputs = execution.block_until_ready().unwrap().remove(0);
        let output = device_outputs.outputs.remove(0);
        let output_pointer = unsafe { output.unsafe_pointer().unwrap() };
        // `tf.aliasing_output` is a hint rather than a guarantee, because the copy insertion of XLA may materialize a fresh
        // output. This pointer equality is therefore a deliberate canary pinned to the CPU plugin version that the
        // repository pins. If a plugin upgrade breaks it while the numeric assertion below still passes, re-evaluate
        // whether the alias hint is still honored rather than assuming a correctness bug.
        assert_eq!(output_pointer, input_pointer);
        let output = output.copy_to_host(None).unwrap().r#await().unwrap();
        assert_eq!(values_from_bytes::<f32>(output.as_slice()), vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_mlir_lowerable_value_to_dense_elements_attribute() {
        // Boolean values retain their logical order through MLIR's bit-packed dense representation.
        let boolean = Array::vector(vec![true, false, true]).unwrap();
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
                Array::vector(vec![i1::new(-1).unwrap(), i1::new(0).unwrap(), i1::new(-1).unwrap()]).unwrap(),
                vec![true, false, true],
            ),
            (
                Array::vector(vec![u1::new(1).unwrap(), u1::new(0).unwrap(), u1::new(1).unwrap()]).unwrap(),
                vec![true, false, true],
            ),
        ] {
            let tensor_type = lower_tensor_type(literal.r#type().as_ref(), &context, location).unwrap();
            let attribute = literal.to_dense_elements_attribute(tensor_type, &context).unwrap();
            assert_eq!(unsafe { attribute.bool_elements().collect::<Result<Vec<_>, _>>().unwrap() }, expected);
        }

        // Wider sub-byte integers occupy one MLIR raw-buffer byte per element and preserve only their declared bits.
        for (literal, expected) in [
            (Array::vector(vec![i2::new(-2).unwrap(), i2::new(1).unwrap()]).unwrap(), vec![0x02, 0x01]),
            (Array::vector(vec![i4::new(-8).unwrap(), i4::new(7).unwrap()]).unwrap(), vec![0x08, 0x07]),
            (Array::vector(vec![u2::new(0).unwrap(), u2::new(3).unwrap()]).unwrap(), vec![0x00, 0x03]),
            (Array::vector(vec![u4::new(1).unwrap(), u4::new(15).unwrap()]).unwrap(), vec![0x01, 0x0f]),
        ] {
            assert_eq!(literal_dense_bytes(&literal, expected.len()), expected);
        }

        // Every byte-aligned integer family preserves signedness, magnitude, and source order without floating-point
        // conversion. The `u64` case deliberately exceeds f64's exact-integer range.
        let integer_cases = vec![
            (Array::vector(vec![-127i8, 126]).unwrap(), values_to_bytes(&[-127i8, 126])),
            (Array::vector(vec![-0x1234i16, 0x2345]).unwrap(), values_to_bytes(&[-0x1234i16, 0x2345])),
            (
                Array::vector(vec![-0x0123_4567i32, 0x0234_5678]).unwrap(),
                values_to_bytes(&[-0x0123_4567i32, 0x0234_5678]),
            ),
            (
                Array::vector(vec![-0x0123_4567_89ab_cdefi64, 0x0234_5678_9abc_def0]).unwrap(),
                values_to_bytes(&[-0x0123_4567_89ab_cdefi64, 0x0234_5678_9abc_def0]),
            ),
            (Array::vector(vec![0x12u8, 0xfe]).unwrap(), values_to_bytes(&[0x12u8, 0xfe])),
            (Array::vector(vec![0x1234u16, 0xfedc]).unwrap(), values_to_bytes(&[0x1234u16, 0xfedc])),
            (
                Array::vector(vec![0x1234_5678u32, 0xfedc_ba98]).unwrap(),
                values_to_bytes(&[0x1234_5678u32, 0xfedc_ba98]),
            ),
            (
                Array::vector(vec![(1u64 << 53) + 1, u64::MAX - 1]).unwrap(),
                values_to_bytes(&[(1u64 << 53) + 1, u64::MAX - 1]),
            ),
        ];
        for (literal, expected) in integer_cases {
            assert_eq!(
                literal_dense_bytes(&literal, expected.len()),
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
            let literal = Array::from_logical_bytes(
                ArrayType::new(data_type, Shape::new(vec![Dimension::Static(bits.len())])),
                &bits,
            )
            .unwrap();
            assert_eq!(literal_dense_bytes(&literal, bits.len()), bits, "low-precision literal type {data_type}");
            let mut builder = CoreProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let output = builder
                .add_instruction(
                    ArrayOperation::Constant(ConstantOperation::new(literal)),
                    Vec::new(),
                    Vec::new(),
                    None,
                )
                .unwrap()[0];
            let program = builder.build::<Vec<Array>, Vec<Array>>(vec![output], Vec::new(), vec![Placeholder]).unwrap();
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
            (Array::vector(bf16_values.to_vec()).unwrap(), values_to_bytes(&bf16_values)),
            (Array::vector(f16_values.to_vec()).unwrap(), values_to_bytes(&f16_values)),
            (Array::vector(f32_values.to_vec()).unwrap(), values_to_bytes(&f32_values)),
            (Array::vector(f64_values.to_vec()).unwrap(), values_to_bytes(&f64_values)),
        ] {
            assert_eq!(
                literal_dense_bytes(&literal, expected.len()),
                expected,
                "floating-point literal type {}",
                literal.r#type().as_ref(),
            );
        }

        // Complex storage interleaves independently exact real and imaginary components in source order.
        let c64_components =
            [f32::from_bits(0x8000_0000), f32::from_bits(0x7fc0_1234), f32::INFINITY, f32::NEG_INFINITY];
        let c64 = Array::vector(vec![
            Complex::new(c64_components[0], c64_components[1]),
            Complex::new(c64_components[2], c64_components[3]),
        ])
        .unwrap();
        assert_eq!(literal_dense_bytes(&c64, size_of_val(&c64_components)), values_to_bytes(&c64_components));
        let c128_components = [
            f64::from_bits(0x8000_0000_0000_0000),
            f64::from_bits(0x7ff8_0000_0000_1234),
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        let c128 = Array::vector(vec![
            Complex::new(c128_components[0], c128_components[1]),
            Complex::new(c128_components[2], c128_components[3]),
        ])
        .unwrap();
        assert_eq!(literal_dense_bytes(&c128, size_of_val(&c128_components)), values_to_bytes(&c128_components));

        // Explicit physical layouts are traversed in logical row-major order before constructing the literal.
        let layout_type = ArrayType::new(DataType::I16, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]))
            .with_layout(Some(StridedLayout::new(vec![2, 4]).into()));
        let layout_literal = Array::from_elements(layout_type, &[1i16, 2, 3, 4]).unwrap();
        assert_eq!(layout_literal.storage_bytes(), values_to_bytes(&[1i16, 3, 2, 4]));
        assert_eq!(literal_dense_bytes(&layout_literal, 8), values_to_bytes(&[1i16, 2, 3, 4]));

        // Empty tensors remain valid dense constants and carry no raw payload bytes.
        let empty = Array::vector(Vec::<i32>::new()).unwrap();
        let tensor_type = lower_tensor_type(empty.r#type().as_ref(), &context, location).unwrap();
        let attribute = empty.to_dense_elements_attribute(tensor_type, &context).unwrap();
        assert_eq!(attribute.elements_count(), 0);

        // Zero literals use the established false i1 physical carrier without introducing logical payload bytes.
        let tensor_type = context
            .tensor_type(context.signless_integer_type(1), &[MlirSize::Static(2)], None, location)
            .unwrap();
        let zero = Array::from_logical_bytes(ArrayType::new_static(DataType::Zero, [2]), &[]).unwrap();
        assert_eq!(
            zero.to_dense_elements_attribute(tensor_type, &context).unwrap().to_string(),
            "dense<false> : tensor<2xi1>",
        );

        // Tokens remain effect values and have no dense literal representation.
        let token = Array::from_logical_bytes(ArrayType::new_static(DataType::Token, [2]), &[]).unwrap();
        assert!(matches!(
            token.to_dense_elements_attribute(tensor_type, &context),
            Err(Error::Lowering(LoweringError::UnsupportedElementType { data_type: DataType::Token, .. })),
        ));
    }

    #[test]
    fn test_mlir_lowerable_value_to_dense_elements_attribute_rejects_symbolic_values() {
        let context = MlirContext::new();
        let array_type = ArrayType::scalar(DataType::F32);
        let tensor_type = lower_tensor_type(&array_type, &context, context.unknown_location()).unwrap();
        assert!(matches!(
            array_type.to_dense_elements_attribute(tensor_type, &context),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::Concretization { message })
                    if *message == "abstract XLA value `f32[]` cannot be materialized as a StableHLO literal",
            ),
        ));
        let captured = ArrayConstant::Captured(CaptureReference::new(3, array_type));
        assert!(matches!(
            captured.to_dense_elements_attribute(tensor_type, &context),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::Concretization { message })
                    if *message == "captured constant #3 cannot be materialized as a literal without capture bindings",
            ),
        ));
    }

    #[test]
    fn test_to_mlir_module_for_plain_program() {
        let (_, compiled): (Array, Program<Array, ArrayOperation<Array>, (Array, Array), Array>) = TEST_ARRAY_DOMAIN
            .interpret_and_trace(
                |inputs| Ok(scalar_bilinear_sin(inputs)),
                (Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()),
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
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_gradients() {
        let (_, compiled): (Array, Program<Array, ArrayOperation<Array>, Array, Array>) = TEST_ARRAY_DOMAIN
            .interpret_and_trace(
                |x| {
                    let context = x.context().clone();
                    Ok(context.differentiate_at(x).gradient(scalar_quartic_plus_sin).unwrap())
                },
                Array::scalar(2.0).unwrap(),
            )
            .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&compiled, "main").unwrap();

        // The gradient computes one cosine for the derivative of `sin(x)`, expands the `x^4` chain rule into
        // multiplications, accumulates the cotangent contributions with additions, and contains no sine.
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>) -> tensor<f64> {
                    %0 = stablehlo.cosine %arg0 : tensor<f64>
                    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
                    %1 = stablehlo.multiply %0, %cst : tensor<f64>
                    %2 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
                    %3 = stablehlo.multiply %2, %arg0 : tensor<f64>
                    %4 = stablehlo.multiply %3, %cst : tensor<f64>
                    %5 = stablehlo.add %1, %4 : tensor<f64>
                    %6 = stablehlo.multiply %arg0, %cst : tensor<f64>
                    %7 = stablehlo.multiply %2, %6 : tensor<f64>
                    %8 = stablehlo.add %5, %7 : tensor<f64>
                    %9 = stablehlo.multiply %arg0, %6 : tensor<f64>
                    %10 = stablehlo.multiply %arg0, %9 : tensor<f64>
                    %11 = stablehlo.add %8, %10 : tensor<f64>
                    %12 = stablehlo.multiply %arg0, %9 : tensor<f64>
                    %13 = stablehlo.add %11, %12 : tensor<f64>
                    return %13 : tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_standalone_pullbacks() {
        // Standalone pullback over the primal operation family, produced by the reverse path that is aware of partitions.
        // The reverse program of `f(x, y) = x * y + sin(x)` consumes `[output_cotangent ++ residuals]`, where the residuals
        // are the factors `cos(x)`, `x`, and `y` at the linearization point. It lowers the backward pass, weighted by those
        // residuals, as `stablehlo.multiply`s of the cotangent against the residual inputs rather than baking the primal
        // point in as constants. This is the analogue of the standalone `vjp_fn` of JAX, with the residuals threaded as
        // explicit arguments.
        let (_, pullback): (Array, _) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .vjp(
                |inputs, ()| Ok(scalar_bilinear_sin(inputs)),
                (Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()),
                (),
            )
            .unwrap();
        let (pullback, _residuals) = pullback.into_transposed_parts().unwrap();
        let pullback = pullback.into_program();

        let stablehlo = to_mlir_module_for_plain_program(&pullback, "main").unwrap();
        // The pullback takes the output cotangent plus three residual inputs and returns one cotangent output for each of
        // `x` and `y`. It multiplies the cotangent by the residual inputs rather than baking the primal point in as
        // constants, so the lowering consists of multiplications and additions with no `stablehlo.constant`.
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
                    %0 = stablehlo.multiply %arg3, %arg0 : tensor<f64>
                    %1 = stablehlo.multiply %arg2, %arg0 : tensor<f64>
                    %2 = stablehlo.multiply %arg1, %arg0 : tensor<f64>
                    %3 = stablehlo.add %0, %2 : tensor<f64>
                    return %3, %1 : tensor<f64>, tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_staged_gradients() {
        // The gradient of `f` staged for JIT compilation stays symbolic, like `jit(grad(f))` in JAX. It uses the traced
        // gradient path, which traces through the VJP and its pullback.
        let (_, compiled): ((Array, Array), Program<Array, ArrayOperation<Array>, (Array, Array), (Array, Array)>) =
            TEST_ARRAY_DOMAIN
                .interpret_and_trace(
                    |inputs| {
                        let context = inputs.0.context().clone();
                        Ok(context.differentiate_at(inputs).gradient(scalar_bilinear_sin).unwrap())
                    },
                    (Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()),
                )
                .unwrap();

        let stablehlo = to_mlir_module_for_plain_program(&compiled, "main").unwrap();
        // `cos(x)` is computed symbolically from `%arg0` rather than baked in as a constant, both inputs are referenced, and
        // the gradient contains no sine, because the derivative of sine is cosine.
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
                    %0 = stablehlo.cosine %arg0 : tensor<f64>
                    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
                    %1 = stablehlo.multiply %0, %cst : tensor<f64>
                    %2 = stablehlo.multiply %arg1, %cst : tensor<f64>
                    %3 = stablehlo.add %1, %2 : tensor<f64>
                    %4 = stablehlo.multiply %arg0, %cst : tensor<f64>
                    return %3, %4 : tensor<f64>, tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_entry_result_placements_follow_operations_that_keep_the_placement_of_their_input() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let split = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let vector_type = ArrayType::new_static(DataType::F32, [4]);
        let input = crate::Array::from_host_buffer(
            &session,
            vector_type.clone(),
            mesh.clone(),
            Some(replicated),
            values_to_bytes(&[1f32, 2.0, 3.0, 4.0]),
        )
        .unwrap();
        let target = Placement::new(mesh.clone(), split.clone()).unwrap();

        // Every operation that returns its input together with its placement keeps the split that a preceding `place`
        // gives its input along an auto mesh axis, which the type of the result does not record, so that the result
        // of the program is split as well (refer to `entry_result_placements`).
        let operations: [(&str, fn(CompileTracer) -> CompileTracer); 5] = [
            ("place without a target", |x| x.place(&None::<Placement>, None).unwrap()),
            ("place with an in-place target", |x| x.place(&Memory::Host { pinned: true }, None).unwrap()),
            ("transfer_to_memory", |x| x.transfer_to_memory(Memory::Host { pinned: true }).unwrap()),
            ("stop_gradient", |x| x.stop_gradient().unwrap()),
            ("tag", |x| x.tag("tagged").unwrap()),
        ];
        for (name, operation) in operations {
            let function: CompiledFunction<ArrayType, ArrayType> = compile(
                |x: CompileTracer| operation(x.place(&target, None).unwrap()),
                vector_type.clone(),
                &session,
                mesh.clone(),
            )
            .unwrap();
            let output = function.executable().execute(&session, input.clone()).unwrap();
            assert_eq!((name, output.sharding()), (name, &split));
        }
    }

    #[test]
    fn test_collect_session_constants_assigns_the_same_arguments_on_every_process() {
        // Two single-device clients stand for two processes. Each independently places its default scalar on global
        // device 0 twice. Only the owner can reuse the original device buffer.
        let clients = [0, 1].map(|process| {
            let options = CpuClientOptions { device_count: Some(1), process_id: Some(process) };
            let client = load_cpu_plugin().unwrap().client(ClientOptions::CPU(options)).unwrap();
            let client: &'static ryft_pjrt::Client<'static> = Box::leak(Box::new(client));
            client
        });
        let first_device = Device::from_pjrt(&clients[0].addressable_devices().unwrap()[0]).unwrap();
        let modules = clients.map(|client| {
            let session = Session::builder().client(client).build().unwrap();
            let scalar = session.place(4f32, PlacementOptions::defaults()).unwrap();
            let first = session.place(scalar.clone(), PlacementOptions::to(first_device)).unwrap();
            let second = session.place(scalar.clone(), PlacementOptions::to(first_device)).unwrap();
            if client.process_index().unwrap() == 0 {
                let buffer = scalar.shards()[0].buffer().unwrap();
                assert!(Arc::ptr_eq(first.shards()[0].buffer().unwrap(), buffer));
                assert!(Arc::ptr_eq(second.shards()[0].buffer().unwrap(), buffer));
            } else {
                assert_eq!(first.addressable_shards().count(), 0);
                assert_eq!(second.addressable_shards().count(), 0);
            }
            let mut builder = ArrayIrProgramBuilder::new();
            let first_clone = first.clone();
            let outputs = [&scalar, &first, &second, &first_clone]
                .map(|array| builder.add_constant(Constant::Session(RetainedArray::new(array))))
                .to_vec();
            let program = builder.build::<Vec<Constant>, Vec<Constant>>(outputs, vec![], vec![Placeholder; 4]).unwrap();
            let constants = collect_session_constants(&program);
            assert_eq!(constants, [&scalar, &first, &second].map(RetainedArray::new));
            let captures = constants.iter().map(|constant| constant.r#type().into_owned()).collect::<Vec<_>>();
            let output_types = vec![scalar.r#type().into_owned(); 4];
            to_mlir_module_for_program(&program, &captures, &(), &output_types, "main", None, None).unwrap()
        });

        assert_eq!(modules[0], modules[1]);
        assert_eq!(
            modules[0],
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) {
                    return %arg0, %arg1, %arg2, %arg1 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_collect_session_constants_ignores_dormant_regions() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let constant = RetainedArray::new(&f32_scalar(&session, &domain_mesh(client, "x", 1), 2.0));
        let scalar_type = constant.r#type().into_owned();
        let with_constant = |extended| {
            scalar_linear_region(&scalar_type, extended, |builder, _, value| {
                let constant = builder.add_constant(Constant::Session(constant.clone()));
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![value, constant], None).unwrap()[0]
            })
        };

        // A session constant that only the dormant transpose region of a `linear_call` refers to is never lowered, so
        // it does not become a hidden argument of the entry function, unlike one that its forward region refers to.
        assert_eq!(
            collect_session_constants(&linear_call_program(&with_constant(false), &with_constant(true))),
            vec![],
        );
        assert_eq!(
            collect_session_constants(&linear_call_program(&with_constant(true), &with_constant(false))),
            vec![constant],
        );
    }

    #[test]
    fn test_collect_nested_sharding_mesh() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 2);
        let input_type = test_vector_type(4);
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(input_type.clone());
        builder
            .add_instruction(
                ReshapeOperation::new(vec![4], vec![2, 2]).with_output_sharding(sharding),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![], vec![Placeholder], vec![])
            .unwrap()
            .into_unprojected::<Constant, Operation>()
            .unwrap();
        assert_eq!(collect_nested_sharding_mesh(&program, None).unwrap(), Some(mesh.clone()));
        // The mesh exists only on an internal value. Lowering must still declare the symbol referenced by its
        // reshape constraint, even though neither the signature nor an explicit sharding operation supplies it.
        to_mlir_module_for_program(
            &program,
            &[],
            &vec![input_type.clone()],
            &Vec::<ArrayType>::new(),
            "main",
            None,
            None,
        )
        .unwrap();
        let mut builder = ProgramBuilder::new();
        builder.add_input(input_type.clone().with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap());
        let metadata_only = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![], vec![Placeholder], vec![])
            .unwrap()
            .into_unprojected::<Constant, Operation>()
            .unwrap();
        assert_eq!(collect_nested_sharding_mesh(&metadata_only, None).unwrap(), None);
        let incompatible = LogicalMesh::new(vec![MeshAxis::new("x", 3, MeshAxisType::Explicit).unwrap()]).unwrap();
        assert!(matches!(
            collect_nested_sharding_mesh(&program, Some(incompatible.clone())),
            Err(Error::Lowering(LoweringError::MismatchedMeshTopology { expected, actual }))
                if expected == incompatible && actual == mesh,
        ));
        assert!(matches!(
            to_mlir_module_for_program(
                &program,
                &[],
                &vec![input_type],
                &Vec::<ArrayType>::new(),
                "main",
                Some(&[Sharding::replicated(incompatible.clone(), 1)]),
                None,
            ),
            Err(Error::Lowering(LoweringError::MismatchedMeshTopology { expected, actual }))
                if expected == incompatible && actual == mesh,
        ));

        // Cast placement can be the module's only mesh use, including when its result is internal.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = test_vector_type(4);
        let output_type = input_type.clone().with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap();
        for operation in [
            CastOperation::<Type>::checked(output_type.clone().into(), vec![input_type.clone().into()])
                .unwrap()
                .into(),
            Operation::Array(ArrayOperation::Cast(
                CastOperation::checked(output_type, vec![input_type.clone()]).unwrap(),
            )),
        ] {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(input_type.clone().into());
            builder.add_instruction(operation, vec![], vec![input], None).unwrap();
            let program = builder.build::<Vec<Constant>, Vec<Constant>>(vec![], vec![Placeholder], vec![]).unwrap();
            assert_eq!(collect_nested_sharding_mesh(&program, None).unwrap(), Some(mesh.clone()));
            assert_eq!(
                to_mlir_module_for_program(
                    &program,
                    &[],
                    &vec![input_type.clone()],
                    &Vec::<ArrayType>::new(),
                    "main",
                    None,
                    None,
                )
                .unwrap(),
                indoc! {r#"
                    module {
                      sdy.mesh @mesh = <["x"=2]>
                      func.func @main(%arg0: tensor<4xf32>) {
                        %0 = sdy.sharding_constraint %arg0 <@mesh, [{}], replicated={"x"}> : tensor<4xf32>
                        return
                      }
                    }
                "#},
            );
        }
    }

    #[test]
    fn test_collect_nested_sharding_mesh_requires_one_topology() {
        let mesh_2x2 = |names: [&str; 2], r#type: MeshAxisType| {
            LogicalMesh::new(names.map(|name| MeshAxis::new(name, 2, r#type).unwrap()).to_vec()).unwrap()
        };
        let mesh = mesh_2x2(["x", "y"], MeshAxisType::Explicit);
        let input_type = test_vector_type(4);
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(input_type.clone());
        builder
            .add_instruction(
                ReshapeOperation::new(vec![4], vec![2, 2]).with_output_sharding(Sharding::replicated(mesh.clone(), 2)),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![], vec![Placeholder], vec![])
            .unwrap()
            .into_unprojected::<Constant, Operation>()
            .unwrap();

        // Device ids follow the order of the axes of the module mesh, so a reordered mesh is another topology.
        let reordered = mesh_2x2(["y", "x"], MeshAxisType::Explicit);
        assert!(matches!(
            collect_nested_sharding_mesh(&program, Some(reordered.clone())),
            Err(Error::Lowering(LoweringError::MismatchedMeshTopology { expected, actual }))
                if expected == reordered && actual == mesh,
        ));
        assert!(matches!(
            to_mlir_module_for_program(
                &program,
                &[],
                &vec![input_type.clone()],
                &Vec::<ArrayType>::new(),
                "main",
                Some(&[Sharding::replicated(reordered.clone(), 1)]),
                None,
            ),
            Err(Error::Lowering(LoweringError::MismatchedMeshTopology { expected, actual }))
                if expected == reordered && actual == mesh,
        ));
        assert_eq!(
            LoweringError::MismatchedMeshTopology { expected: reordered, actual: mesh.clone() }.to_string(),
            "mesh `['x'=2:explicit, 'y'=2:explicit]` does not have the topology of the module mesh \
             `['y'=2:explicit, 'x'=2:explicit]`",
        );

        // Meshes with other axes are not merged into a union.
        let other_axes = mesh_2x2(["x", "z"], MeshAxisType::Explicit);
        assert!(matches!(
            collect_nested_sharding_mesh(&program, Some(other_axes.clone())),
            Err(Error::Lowering(LoweringError::MismatchedMeshTopology { expected, actual }))
                if expected == other_axes && actual == mesh,
        ));

        // Meshes that differ only in axis types share the module mesh, which keeps the axis types of its first mesh.
        let manual = mesh_2x2(["x", "y"], MeshAxisType::Manual);
        assert_eq!(collect_nested_sharding_mesh(&program, Some(manual.clone())).unwrap(), Some(manual.clone()));
        assert_eq!(
            to_mlir_module_for_program(
                &program,
                &[],
                &vec![input_type],
                &Vec::<ArrayType>::new(),
                "main",
                Some(&[Sharding::replicated(manual, 1)]),
                None,
            )
            .unwrap(),
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2, \"y\"=2]>
                  func.func @main(%arg0: tensor<4xf32> \
                      {sdy.sharding = #sdy.sharding<@mesh, [{}], replicated={\"x\", \"y\"}>}) {
                    %0 = stablehlo.reshape %arg0 : (tensor<4xf32>) -> tensor<2x2xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{}, {}], replicated={\"x\", \"y\"}> : tensor<2x2xf32>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_collect_nested_sharding_mesh_includes_call_boundaries() {
        // A call constrains its result to the override of the boundary of its callee, which no type records because the
        // override is over an auto mesh axis, so the module declares the mesh of the override even when it is lowered
        // without signature shardings, both for a static program and for one with a bounded dynamic value. The call
        // also constrains its operand replicated over that mesh, as an execution of its callee, whose input type has no
        // sharding, places its argument (refer to `JitCallOperation::boundary_input_sharding`).
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let device_mesh = DeviceMesh::new(mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let lower = |array_type: ArrayType| {
            let callee = {
                let mut builder = ArrayIrProgramBuilder::new();
                let input = builder.add_input(array_type.clone().into());
                Arc::new(
                    builder
                        .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
                        .unwrap(),
                )
            };
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(array_type.clone().into());
            let callee_region = builder.intern_callee(&callee, None).unwrap();
            let call =
                JitCallOperation::new(0).with_boundary(Some(&device_mesh), Vec::new(), vec![Some(replicated.clone())]);
            let output =
                builder.add_instruction(Operation::JitCall(call), vec![callee_region], vec![input], None).unwrap()[0];
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap();
            assert_eq!(collect_nested_sharding_mesh(&program, None).unwrap(), Some(mesh.clone()));
            let types = vec![array_type];
            to_mlir_module_for_program(&program, &[], &types, &types, "main", None, None).unwrap()
        };
        assert_eq!(
            lower(test_vector_type(4)),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{}]> : tensor<4xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{}]> : tensor<4xf32>
                    return %1 : tensor<4xf32>
                  }
                }
            "#},
        );
        let length = DimensionVariable::new("length", DimensionBounds::new(1, Some(4)).unwrap());
        assert_eq!(
            lower(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(length)]))),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<i32>) -> (tensor<?xf32, #stablehlo.bounds<3>>, tensor<i64>) {
                    %0 = stablehlo.set_dimension_size %arg0, %arg1, dim = 0 : (tensor<3xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<3>>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{}]> : tensor<?xf32, #stablehlo.bounds<3>>
                    %2 = sdy.sharding_constraint %1 <@mesh, [{}]> : tensor<?xf32, #stablehlo.bounds<3>>
                    %3 = stablehlo.get_dimension_size %2, dim = 0 : (tensor<?xf32, #stablehlo.bounds<3>>) -> tensor<i32>
                    %4 = stablehlo.convert %3 : (tensor<i32>) -> tensor<i64>
                    return %2, %4 : tensor<?xf32, #stablehlo.bounds<3>>, tensor<i64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_normalize_function_name() {
        assert_eq!(normalize_function_name(" main ").unwrap(), "main");
        assert_eq!(normalize_function_name(" @main ").unwrap(), "main");
        assert!(matches!(
            normalize_function_name(" @ "),
            Err(Error::Lowering(LoweringError::InvalidFunctionName { function_name })) if function_name.is_empty(),
        ));
        assert!(matches!(
            normalize_function_name("main function"),
            Err(Error::Lowering(LoweringError::InvalidFunctionName { function_name })) if function_name == "main function",
        ));
    }

    #[test]
    fn test_validate_local_output_placement() {
        // An array IR `Operation::Cast` (i.e., a `CastOperation<Type>` rather than a projected
        // `ArrayOperation::Cast`) lowers without passing through `ArrayOperation::lower`, so only the validation of the
        // family in `Operation::lower` sees its output placement. Its first output requests a mesh that redefines the bound
        // manual axis `x` as explicit and shards a local dimension over it, which only the region boundary may do.
        let vector_type = test_vector_type(4);
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let explicit_type = vector_type
            .clone()
            .with_sharding(Sharding::new(explicit_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let cast = |output_type: &ArrayType, input_type: &ArrayType| {
            Operation::Cast(
                CastOperation::unchecked(output_type.clone().into(), vec![input_type.clone().into()]).unwrap(),
            )
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(vector_type.clone().into());
        let placed =
            builder.add_instruction(cast(&explicit_type, &vector_type), Vec::new(), vec![input], None).unwrap()[0];
        let output =
            builder.add_instruction(cast(&vector_type, &explicit_type), Vec::new(), vec![placed], None).unwrap()[0];
        let body = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            lower_replicated_shard_map_body(body, 2),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::UnsupportedOperation { message })
                    if *message == "`cast` cannot assign bound manual mesh axes to local output dimensions",
            ),
        ));
    }
}
