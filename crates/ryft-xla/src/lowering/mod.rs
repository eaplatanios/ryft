//! Lowering of staged XLA programs to StableHLO/Shardy MLIR modules.
//!
//! # Exporting Programs
//!
//! [`ProgramLoweringExtension`] lowers a staged program over any [`ProgramLoweringOperation`].
//! [`ProgramLoweringExtension::to_rendered_mlir`] renders the module according to its [`ProgramLoweringOptions`]. The
//! default options read the entry signature from the global input and output types of the program, pass the arrays that
//! the program imports as hidden inputs that precede its public inputs, and prune the program first. Other options
//! replace those types, attach `sdy.sharding` attributes to the arguments and results of the entry function, or lower
//! the program as it is.
//! [`ProgramLoweringExtension::to_mlir`] returns the [`MlirProgram`], which pairs the module with what executing it
//! requires. Programs must have their references and mutable state discharged before lowering. Failures that only
//! lowering can detect are reported as [`LoweringError`]s wrapped in [`Error::Lowering`].
//!
//! # Lowering Custom Operations
//!
//! Each operation lowers itself through [`LowerableOperation::lower`], which receives a [`Lowerer`] that exposes the
//! block receiving the emitted operations, the MLIR context and location, the regions of the instruction, and the
//! ordered effect token chains of the enclosing lowering scope. Operation families dispatch to the payloads of their
//! variants through the [`lower_operation_variants`](crate::lower_operation_variants) macro, so downstream crates
//! extend lowering by implementing [`LowerableOperation`] for their own operations and families. An operation whose
//! types are a member of the types of its family (e.g., an operation over array types in a family over [`Type`])
//! implements [`MemberLowerableOperation`] instead, which lowers it with the types of the family. The remaining
//! public functions of this module lower types, constants, and dynamic dimensions, for use in such implementations.
//!
//! # Options
//!
//! [`RaggedDotLoweringStrategy`] selects how a compilation lowers ragged dots, and an operation states through
//! [`LowerableOperation::validate_data_derived_dimensions`] whether it can consume dimensions that are derived from
//! runtime data.

use std::rc::Rc;
use std::sync::Arc;

use ryft_core::{
    ArrayType, CALL_OPERATION_NAME, CompositeKey, ConstrainShardingOperation, DataType, Device, DeviceMesh,
    EagerContext, EffectClass, EffectClasses, FlatProgram, LogicalMesh, Memory, Operation as CoreOperation,
    OperationPayloadProjection, PLACE_OPERATION_NAME, Parameterized, Program, ProgramError, RegionRef, Sharding,
    ShardingDimension, Type as CoreType,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Value as MlirValue, ValueRef,
};

use crate::lowering::contexts::ProgramLoweringContext;
use crate::lowering::programs::ProgramInlineLoweringExtension;
use crate::lowering::reductions::{
    constrain_unreduced_intermediates, constrain_unreduced_outputs, validate_declared_pending_reduction,
    validate_pending_reductions_across_regions,
};
use crate::operations::call::CallFunctionMap;
use crate::operations::composites::NamedCompositionKey;
use crate::{Constant, Error, Operation, RuntimeRequirement, Session, Type};

// TODO(eaplatanios): Review this module.

/// Represents errors produced while validating a program for XLA lowering, constructing its StableHLO
/// representation, or verifying the result.
///
/// Public lowering entry points return these errors through [`Error::Lowering`]. Core program and native MLIR errors
/// retain their original types in [`Error::Core`] and [`Error::Mlir`], respectively.
#[derive(Clone, Debug, PartialEq, Eq, Hash, thiserror::Error)]
pub enum LoweringError {
    /// The caller selected an empty function symbol or one containing whitespace.
    #[error("invalid function name `{function_name}` used during XLA lowering")]
    InvalidFunctionName { function_name: String },

    /// A captured constant has no binding in the capture namespace available to this lowering.
    #[error("missing captured constant #{index} during XLA lowering")]
    MissingCapturedConstant { index: usize },

    /// A session constant appears where no hidden input of the entry function is reachable (e.g., inside a
    /// `shard_map` body or a shared private function).
    #[error("session constant `{constant}` cannot be referenced in this lowering scope")]
    UnreachableSessionConstant { constant: String },

    /// References reached a lowering boundary that requires their discharge.
    #[error("unresolved reference construct `{construct}` must be discharged before XLA lowering")]
    UnresolvedReference { construct: String },

    /// Mutable state reached a lowering boundary that requires its discharge.
    #[error("unresolved state in `{construct}` must be discharged before XLA lowering")]
    UnresolvedState { construct: String },

    /// The logical reference-state metadata cannot produce a valid executable buffer alias.
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
        "`{}` cannot move a value to the devices [{}] of mesh `{}` inside a program compiled for the devices [{}] of \
         mesh `{}`",
        PLACE_OPERATION_NAME,
        .target.devices().iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
        .target.logical_mesh(),
        .mesh.devices().iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
        .mesh.logical_mesh(),
    )]
    PlacementOutsideProgramMesh { target: DeviceMesh, mesh: DeviceMesh },

    #[error(
        "`{}` cannot move a value to the device {} inside a program compiled for the devices [{}] of mesh `{}`",
        PLACE_OPERATION_NAME,
        .device,
        .mesh.devices().iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
        .mesh.logical_mesh(),
    )]
    DeviceOutsideProgramMesh { device: Device, mesh: DeviceMesh },

    #[error(
        "`{}` cannot call a function compiled for the devices [{}] of mesh `{}` inside a program compiled for the \
         devices [{}] of mesh `{}`",
        CALL_OPERATION_NAME,
        .callee.devices().iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
        .callee.logical_mesh(),
        .mesh.devices().iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
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

/// XLA representation used when lowering ragged dots. Set it for a compilation with
/// [`Options::with_ragged_dot_lowering_strategy`](crate::Options::with_ragged_dot_lowering_strategy) or for a direct
/// lowering with [`ProgramLoweringOptions::with_ragged_dot_lowering_strategy`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum RaggedDotLoweringStrategy {
    /// Expand ragged dots into masking plus ordinary dense generalized dots. This is the default used by
    /// [`Options::new`](crate::Options::new).
    #[default]
    Decomposition,

    /// Emit dedicated CHLO `ragged_dot` instructions.
    Instruction,
}

/// Program lowering context for nested regions and their enclosing module.
pub(crate) mod contexts;
/// Decisions about which regions, captures, and session constants of a program lowering emits.
pub(crate) mod emissions;
/// Options, entry points, and placement analysis for lowering whole programs.
pub(crate) mod programs;
/// Lowering and validation of pending reductions.
pub(crate) mod reductions;
/// Lowering of logical meshes, shardings, and sharding constraints to Shardy.
pub(crate) mod sharding;
/// Shared StableHLO type, constant, and dynamic-dimension lowering helpers.
pub(crate) mod types;
/// Lowering of the values of programs (i.e., their constants) to StableHLO values.
pub(crate) mod values;

pub use programs::{
    EntryShardingOverride, MlirProgram, MlirProgramBoundaryMapping, MlirProgramBoundaryMappingInputDimension,
    MlirProgramBoundaryMappingOutputDimension, ProgramLoweringExtension, ProgramLoweringOptions,
};
pub use sharding::{
    LogicalMeshExtension, OpenDimensions, SHARDY_MESH_SYMBOL_NAME, ShardingExtension, UnreducedKindExtension,
};
pub use types::{
    ArrayTypeLoweringExtension, DataTypeLoweringExtension, MemoryLoweringExtension, TypeLoweringExtension,
    dimension_extent_i32, dimension_extent_i64, lower_constant_output, lower_explicit_shape, lower_f64_constant_splat,
    lower_runtime_dimension_size_i64, lower_signed_one_bit_to_mlir, lower_static_index_constants,
    lower_u64_scalar_constant, lower_unplaced_constant_output, physical_bound_type, stable_hlo_dynamic_dimension_bound,
    static_dimensions,
};

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
pub struct Lowerer<'b, 'c: 'b, 't: 'c, O: CoreOperation<Type = Type> = Operation> {
    /// Owning block receiving the lowered operations.
    block: BlockRef<'b, 'c, 't>,

    /// MLIR context owning the block and created operations.
    context: &'c MlirContext<'t>,

    /// Shared MLIR location used for emitted operations.
    location: LocationRef<'c, 't>,

    /// Attached regions of the instruction currently being lowered, materialized as flat programs in region-slot order.
    regions: Vec<FlatProgram<EagerContext<Constant, O>>>,

    /// Shared private functions emitted for deduplicated `call` callees, consulted at `call` lowering sites.
    /// Shared via [`Rc`] so it threads through nested lowering scopes without lifetime entanglement.
    nested_functions: Option<Rc<CallFunctionMap<O>>>,

    /// Hidden capture inputs of the function currently being lowered, in capture-table order.
    captured_values: Vec<ValueRef<'b, 'c, 't>>,

    /// Current per-class effect tokens of the lowering scope this lowerer emits into.
    effect_tokens: EffectTokens<'b, 'c, 't>,

    /// Program lowering context for the enclosing scope.
    lowering_context: ProgramLoweringContext,
}

impl<'b, 'c: 'b, 't: 'c, O: CoreOperation<Type = Type>> Lowerer<'b, 'c, 't, O> {
    /// Creates a lowerer that emits operations into `block` in the given program lowering context.
    pub(crate) fn new(
        block: BlockRef<'b, 'c, 't>,
        context: &'c MlirContext<'t>,
        location: LocationRef<'c, 't>,
        lowering_context: ProgramLoweringContext,
    ) -> Self {
        Self {
            block,
            context,
            location,
            regions: Vec::new(),
            nested_functions: None,
            captured_values: Vec::new(),
            effect_tokens: EffectTokens::default(),
            lowering_context,
        }
    }

    /// Attaches the materialized attached regions of the instruction currently being lowered.
    pub(crate) fn with_regions(mut self, regions: Vec<FlatProgram<EagerContext<Constant, O>>>) -> Self {
        self.regions = regions;
        self
    }

    /// Attaches the shared deduplicated `call` functions consulted while lowering.
    pub(crate) fn with_nested_functions(mut self, nested_functions: Option<Rc<CallFunctionMap<O>>>) -> Self {
        self.nested_functions = nested_functions;
        self
    }

    /// Attaches the hidden capture inputs of the function currently being lowered.
    pub(crate) fn with_captured_values(mut self, captured_values: &[ValueRef<'b, 'c, 't>]) -> Self {
        self.captured_values = captured_values.to_vec();
        self
    }

    /// Attaches the current per-class effect tokens of the enclosing lowering scope.
    pub(crate) fn with_effect_tokens(mut self, effect_tokens: EffectTokens<'b, 'c, 't>) -> Self {
        self.effect_tokens = effect_tokens;
        self
    }

    /// Returns the shared private functions emitted for deduplicated `call` callees, consulted at `call`
    /// lowering sites. Shared via [`Rc`] so it threads through nested lowering scopes without lifetime entanglement.
    #[inline]
    pub(crate) fn nested_functions(&self) -> Option<&Rc<CallFunctionMap<O>>> {
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

    /// Returns the program lowering context for the enclosing scope.
    #[inline]
    pub(crate) fn lowering_context(&self) -> &ProgramLoweringContext {
        &self.lowering_context
    }

    /// Returns the effect tokens of the lowering scope mutably, together with the attached regions, the captured
    /// values, the shared `call` functions, and the program lowering context of this lowerer, in that order.
    /// Lowering helpers thread the tokens while they read the rest of that state, which the separate accessors cannot
    /// provide, because the mutable borrow of [`Self::effect_tokens_mut`] covers the whole lowerer.
    #[allow(clippy::type_complexity)]
    pub(crate) fn split_effect_tokens(
        &mut self,
    ) -> (
        &mut EffectTokens<'b, 'c, 't>,
        &[FlatProgram<EagerContext<Constant, O>>],
        &[ValueRef<'b, 'c, 't>],
        Option<&Rc<CallFunctionMap<O>>>,
        &ProgramLoweringContext,
    ) {
        (
            &mut self.effect_tokens,
            &self.regions,
            &self.captured_values,
            self.nested_functions.as_ref(),
            &self.lowering_context,
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
    pub fn regions(&self) -> &[FlatProgram<EagerContext<Constant, O>>] {
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
        self.lowering_context.require_runtime(requirement)
    }

    /// Declares `mesh` as the mesh that this emitted operation refers to (e.g., through the Shardy attribute of a
    /// [`Sharding`] that it lowers with [`ShardingExtension`]).
    ///
    /// A module declares one mesh, under [`SHARDY_MESH_SYMBOL_NAME`], which every nested lowering scope shares, and it
    /// declares that mesh only when an emitted operation refers to it. An operation that lowers a sharding itself must
    /// therefore declare the mesh of that sharding here.
    ///
    /// # Errors
    ///
    /// Returns [`LoweringError::MismatchedMeshTopology`] when the axis names or sizes of `mesh` differ from those of
    /// the mesh that the module already refers to, including in order.
    pub fn require_mesh(&mut self, mesh: &LogicalMesh) -> Result<(), Error> {
        self.lowering_context.require_mesh(mesh)
    }

    /// Returns the symbol of the private function of the module that holds the decomposition of the named
    /// composition `key`, and declares that this emitted operation refers to that function (e.g., as the
    /// decomposition of a `stablehlo.composite` that it emits, or as the callee of a `func.call`).
    ///
    /// A module holds one such function for each key, which every nested lowering scope shares, and it holds the
    /// function only when an emitted operation requests it here. The first request for a key calls `program`, which
    /// returns the decomposition as a program whose inputs and outputs are those of the boundary of `key`, and every
    /// later request returns the same symbol without calling it. The symbol is the name of the composite of `key`,
    /// with a numeric suffix when another key of the module has that name already.
    ///
    /// The decomposition is lowered outside any manual region, into a function without an effect token or hidden
    /// inputs. It must therefore have no effects when it is lowered and import no array, which this function does
    /// not check.
    ///
    /// # Errors
    ///
    /// Returns the error of `program` when it fails, in which case nothing is recorded for `key`.
    pub fn require_named_composition<F: FnOnce() -> Result<FlatProgram<Session>, Error>>(
        &mut self,
        key: NamedCompositionKey,
        program: F,
    ) -> Result<Rc<str>, Error> {
        self.lowering_context.require_named_composition(key, program)
    }

    /// Returns the StableHLO tensor type that represents values of `array_type`.
    ///
    /// # Errors
    ///
    /// Returns [`LoweringError::UnsupportedElementType`] when an element type has no StableHLO tensor representation,
    /// and propagates native construction failures through [`Error::Mlir`].
    pub fn lower_tensor_type(&self, array_type: &ArrayType) -> Result<ryft_mlir::TensorTypeRef<'c, 't>, Error> {
        array_type.lower(self.context, self.location)
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
        O: ProgramLoweringOperation,
    {
        let Some(region) = self.regions.get(index) else {
            return Err(ProgramError::MalformedProgram(format!(
                "region #{index} requested from an instruction with {} attached regions",
                self.regions.len()
            ))
            .into());
        };
        region.lower_inline(
            self.captured_values.as_slice(),
            input_values,
            self.nested_functions.as_ref(),
            &self.lowering_context,
            &mut self.effect_tokens,
            self.block,
            self.location,
        )
    }
}

/// Operations that can be lowered to StableHLO for XLA compilation.
///
/// In [`Self::lower`], `Self` is the operation being lowered. Its [`CoreOperation`](ryft_core::Operation) contract
/// determines `Self::Type`, the type of its input and output descriptors. For example, an array operation receives
/// [`ArrayType`] descriptors, and a first-class dimension operation receives [`DimensionType`](ryft_core::DimensionType)
/// descriptors. `O` is the enclosing program's operation family, which [`Lowerer`] uses to lower attached regions.
/// Both kinds of operation can be lowered inside a program whose family uses [`Type`].
///
/// Families such as [`ArrayOperation`](ryft_core::ArrayOperation) and [`Operation`] implement this trait by dispatching
/// each variant to its payload through [`lower_operation_variants`](crate::lower_operation_variants), which also handles
/// projected and mixed members. Downstream families that meet the bounds of [`ProgramLoweringOperation`] can lower whole
/// programs through [`ProgramLoweringExtension`].
///
/// Operations lower attached regions through methods such as [`Lowerer::lower_region_inline`]. This keeps the bounds of
/// a family's lowering implementation from depending recursively on its own lowering rules.
pub trait LowerableOperation: CoreOperation {
    /// Lowers this operation to one or more StableHLO operations emitted into the block of `lowerer`, returning one
    /// MLIR value per output.
    ///
    /// # Parameters
    ///
    ///   - `lowerer`: Lowering state of the instruction applying this operation, with access to attached regions of `O`.
    ///   - `input_values`: Lowered MLIR values of the instruction inputs, one per input.
    ///   - `input_types`: Declared types of the instruction inputs, in input order.
    ///   - `output_types`: Declared types of the instruction outputs, in output order.
    fn lower<'b, 'c: 'b, 't: 'c, O: ProgramLoweringOperation>(
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

    /// Returns `true` if a backend implementation replaces the calls of the named composite `key` of this family at
    /// the boundary with the provided types when they are lowered for `target_platform`. The lowering of such a call
    /// emits none of its regions, so the module-level lowering passes skip them. It returns `false` unless a family
    /// overrides it, in which case the body of every named composite call is lowered.
    #[inline]
    fn has_composite_implementation(
        key: &CompositeKey,
        input_types: &[Type],
        output_types: &[Type],
        target_platform: Option<&str>,
    ) -> bool {
        let _ = (key, input_types, output_types, target_platform);
        false
    }

    /// Validates that this operation can consume a dimension that is derived from runtime data (i.e., one that
    /// `dimension_from_scalar` produces, or that dimension arithmetic derives from such a dimension). A value with
    /// such a dimension is bounded dynamic, so an operation can consume it only if its lowering keeps the physical
    /// padding of that value unobservable. Compilation admits a program that derives dimensions from data only if
    /// every operation that consumes one passes this validation. Families answer for each of their variants, and
    /// [`Operation`] does so with exhaustive matches, so that adding an operation to it does not compile until it
    /// is stated whether that operation passes.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::UnsupportedOperation`], which names this operation and the reason, when this
    /// operation cannot consume such a dimension. It returns that error unless a family overrides this function, so
    /// that an operation whose padding behavior nobody has stated is rejected and never assumed to be safe.
    #[inline]
    fn validate_data_derived_dimensions(&self) -> Result<(), Error> {
        Err(unsupported_data_derived_dimensions(
            self.name(),
            "its operation family does not state how it keeps the padding of data-derived dimensions unobservable",
        ))
    }

    /// Returns `program`, a pruned program of this family, without the mesh that each of its `call`s records for the
    /// devices that its callee names when that callee names none anymore once it is lowered for `target_platform`
    /// (e.g., because pruning dropped the only request that named them). Lowering prepares every program that it
    /// prunes with this function. It returns `program` as it is unless a family overrides it, in which case such a
    /// call keeps its mesh.
    #[inline]
    fn without_stale_call_meshes<Input, Output>(
        program: Program<Constant, Self, Input, Output>,
        target_platform: Option<&str>,
    ) -> Result<Program<Constant, Self, Input, Output>, ProgramError>
    where
        Self: CoreOperation<Type = Type> + Sized,
        Input: Parameterized<Constant>,
        Output: Parameterized<Constant>,
    {
        let _ = target_platform;
        Ok(program)
    }

    /// Returns, for each input of `callee`, the callee of a `call` of this family whose first `capture_count`
    /// inputs are its lifted captures, whether it is a capture that the callee never reads when it is lowered for
    /// `target_platform`, or [`None`] when every input counts as read. The lowering of the call passes a value that
    /// requires no buffer in place of such a capture, and the function that makes the call reads the input only
    /// where the callee reads it. Only a family knows the regions of its callees, so this function is asked of the
    /// family. It returns [`None`] unless a family overrides it, and [`Operation`] answers from the regions that its
    /// lowering emits.
    #[inline]
    fn unread_captures(
        callee: RegionRef<'_, Constant, Self>,
        capture_count: usize,
        target_platform: Option<&str>,
    ) -> Option<Arc<[bool]>>
    where
        Self: CoreOperation<Type = Type>,
    {
        let _ = (callee, capture_count, target_platform);
        None
    }
}

/// Enclosing operation families for programs lowered by [`ProgramLoweringExtension`].
///
/// This is the bound on `O` in [`LowerableOperation::lower`]. The family lowers program instructions at [`Type`]
/// and exposes its payloads through [`OperationPayloadProjection`] so that lowering passes can recognize operations
/// inside the family, for example when deduplicating calls or deciding which regions to emit.
///
/// The blanket implementation covers every type that meets these bounds, including downstream operation families.
pub trait ProgramLoweringOperation: LowerableOperation<Type = Type> + OperationPayloadProjection {}

impl<O: LowerableOperation<Type = Type> + OperationPayloadProjection> ProgramLoweringOperation for O {}

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
    fn lower_in_parent<'b, 'c: 'b, 't: 'c, O: ProgramLoweringOperation>(
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
pub fn lower_projected_operation<'b, 'c: 'b, 't: 'c, U, T, P, O: ProgramLoweringOperation>(
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
///     stores an [`ArrayOperation`](ryft_core::ArrayOperation) in an [`Type`] family).
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
///     fn lower<'b, 'c: 'b, 't: 'c, O: ProgramLoweringOperation>(
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

/// Lowers `operation`, applied by one instruction to `input_values` of `input_types` with outputs of `output_types`,
/// with `lowerer`, and constrains every emitted value that carries a pending reduction over a mesh axis that no
/// enclosing manual region binds. [`ProgramInlineLoweringExtension::lower_inline`] applies these constraints to the
/// instructions of both entry functions and nested regions.
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
///   - Outputs, with [`constrain_unreduced_outputs`], which constrains each output to the tracked sharding of its type,
///     with the placement along auto mesh axes that dataflow gives it (refer to [`constrain_unreduced_intermediates`]).
///
/// Only an instruction with an input that carries a pending reduction can emit an intermediate value with one, so any
/// other instruction pays one scan of its input types and of its output types, without allocating or walking the
/// emitted operations. Lowering emits the constraints for every target and removes them with
/// [`ryft_mlir::Module::remove_shardy_operations`] when the target lacks the Shardy dialect.
fn lower_instruction_operation<'b, 'c: 'b, 't: 'c, O: ProgramLoweringOperation>(
    operation: &O,
    lowerer: &mut Lowerer<'b, 'c, 't, O>,
    input_values: &[ValueRef<'b, 'c, 't>],
    input_types: &[Type],
    output_types: &[Type],
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    let mut block = lowerer.block();
    let location = lowerer.location();
    let bound_manual_axes = lowerer.lowering_context().bound_manual_axes();
    if let Some(constraint) = operation.projected_payload::<ConstrainShardingOperation>()
        && let Some(Type::Array(input_type)) = input_types.first()
    {
        let declared = constraint.lowered_sharding(input_type).map_err(ProgramError::from)?;
        let carried = input_type.sharding();
        let data_type = input_type.data_type();
        validate_declared_pending_reduction(&declared, carried, data_type, bound_manual_axes, "constrain_sharding")?;
    }
    let has_pending_reduction = input_types.iter().any(|input_type| match input_type {
        Type::Array(input_type) => input_type.unreduced_axes().iter().any(|axis| !bound_manual_axes.contains(axis)),
        Type::Dimension(_) | Type::Reference(_) => false,
    });
    validate_pending_reductions_across_regions(operation.name(), input_types, output_types, bound_manual_axes)?;
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
    let lowering_context = lowerer.lowering_context();
    let mut output_dimensions = Vec::new();
    if let Some(anchor) = anchor {
        output_dimensions = constrain_unreduced_intermediates(
            anchor,
            input_values,
            input_types,
            &outputs,
            output_types,
            lowering_context,
            &mut block,
            location,
        )?;
        // SAFETY: nothing uses the result of the anchor, so removing it from the block and dropping the detached
        // operation that `remove_operation` returns destroys it without leaving a dangling use.
        unsafe { block.remove_operation(anchor)? };
    }
    constrain_unreduced_outputs(
        &mut outputs,
        output_types,
        &output_dimensions,
        lowering_context,
        &mut block,
        location,
    )?;
    Ok(outputs)
}

/// Returns the error with which [`LowerableOperation::validate_data_derived_dimensions`] rejects the operation named
/// `operation_name`, which cannot consume a dimension that is derived from runtime data because of `reason`.
pub(crate) fn unsupported_data_derived_dimensions(operation_name: &str, reason: &str) -> Error {
    ProgramError::UnsupportedOperation {
        message: format!(
            "operation `{operation_name}` cannot consume data-derived dimensions during XLA lowering because {reason}",
        ),
    }
    .into()
}

/// Rejects array `output_types` of the operation named `operation_name` that shard a local dimension over one of the
/// `bound_manual_axes` of the enclosing manual regions. Manual dimension ownership belongs to region boundaries. Local
/// operations may change manual variation through collectives, but must not manufacture a global placement, which
/// lowered sharding constraints would otherwise silently strip (refer to
/// `ShardingExtension::lower_constraint`).
/// First-class dimensions and references carry no placement of their own.
pub(crate) fn validate_local_output_placement(
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

#[cfg(test)]
pub(crate) mod tests {
    use std::any::{Any, TypeId};
    use std::cell::RefCell;
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayOperation, ArrayOperations, ArrayType, AtomId, CallOperation, CastOperation, DataType, Device, DeviceMesh,
        Dimension, DimensionBounds, DimensionVariable, DomainTracingContext, ErasedOperation, Error as CoreError,
        LinearCallOperation, LogicalMesh, MeshAxis, MeshAxisType, MulOperation, NegOperation,
        Operation as CoreOperation, OperationPayloadProjection, Parameterized, ParameterizedFamily, Placeholder,
        Program, ProgramBuilder as CoreProgramBuilder, ProgramError, ProjectedContext, ReduceOperation, ReductionKind,
        RegionInterface, Reshape, ScanOptions, Shape, ShardMap, ShardMapBoundary, ShardMapOperation, ShardMapOptions,
        Sharding, ShardingDimension, TypeError, Typed, TypedProgram, UnreducedKind, ValueProjection, scan,
    };

    use ryft_mlir::ValueRef;
    use ryft_pjrt::protos::{OptionOverride, OptionValue, ShapeChecks};
    use ryft_pjrt::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram};

    use crate::operations::assertions::ensure_assertion_handler_registered;
    use crate::tests::{cpu_client, replicated_array, test_compilation_options, values_from_bytes, values_to_bytes};
    use crate::{
        ArrayConstant, ArrayTracer, CompileTracer, CompiledFunction, Constant, Error, FromPjrt, Operation, Options,
        ProgramLoweringExtension, Session, Type, compile, trace_with_context,
    };

    use super::*;

    /// Builds homogeneous array fixtures that [`unproject_plain_program`] lifts into the XLA program representation.
    pub(crate) type ProgramBuilder = CoreProgramBuilder<ArrayConstant, ArrayOperation<ArrayConstant>>;

    /// Homogeneous array program built by [`ProgramBuilder`].
    pub(crate) type PlainProgram =
        Program<ArrayConstant, ArrayOperation<ArrayConstant>, Vec<ArrayConstant>, Vec<ArrayConstant>>;

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
        Input::Family: ParameterizedFamily<Constant>
            + ParameterizedFamily<Type>
            + ParameterizedFamily<ArrayTracer>
            + ParameterizedFamily<Sharding>,
        Output: Parameterized<ArrayType>,
        Output::Family: ParameterizedFamily<Constant>
            + ParameterizedFamily<Type>
            + ParameterizedFamily<ArrayTracer>
            + ParameterizedFamily<Sharding>,
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
        let traced = trace_with_context::<_, Input, Output::To<ArrayTracer>>(
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
        traced.to_rendered_mlir("main", &ProgramLoweringOptions::new())
    }

    /// Dynamic dimension of a fresh non-negative variable named `name`, bounded above by `exclusive_upper_bound` when it
    /// is provided and unbounded otherwise.
    pub(crate) fn dynamic_dimension(name: &str, exclusive_upper_bound: Option<usize>) -> Dimension {
        DimensionVariable::new(name, DimensionBounds::non_negative(exclusive_upper_bound).unwrap()).into()
    }

    /// One buffer of the mixed-type executable ABI used by the reshape execution tests. It is either `f64` or `i32`
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
        program: &FlatProgram<Session>,
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
        let module = program
            .to_rendered_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_input_types(&input_types)
                    .with_output_types(&output_types)
                    .without_boundary_shardings(),
            )
            .unwrap();
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
                // A populated dynamic output is copied back as its compact logical elements, while an empty logical
                // output may still carry its backing allocation and trailing size metadata, which are ignored.
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
    pub(crate) fn unproject_plain_program(program: PlainProgram) -> FlatProgram<Session> {
        program.into_unprojected::<Constant, Operation>().unwrap()
    }

    /// Stages one `call` to `callee`, which is interned as a shared callee root region, over `inputs` in `builder`.
    pub(crate) fn add_call(
        builder: &mut CoreProgramBuilder<Constant, Operation>,
        callee: &Arc<FlatProgram<Session>>,
        inputs: Vec<AtomId>,
    ) -> AtomId {
        let callee_region = builder.intern_callee(callee, None).unwrap();
        builder
            .add_instruction(Operation::Call(CallOperation::new(0)), vec![callee_region], inputs, None)
            .unwrap()[0]
    }

    /// Wraps one flat vector program in a replicated manual `shard_map` and lowers the enclosing program.
    pub(crate) fn lower_replicated_shard_map_body(
        body_program: FlatProgram<Session>,
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
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(vector_type.clone().into());
        let body = builder.import_program(body_program);
        let output =
            builder.add_instruction(Operation::ShardMap(Box::new(operation)), vec![body], vec![input], None)?[0];
        let program =
            builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])?;
        let input_types = vec![vector_type.clone()];
        let output_types = vec![vector_type];
        program.to_rendered_mlir(
            "main",
            &ProgramLoweringOptions::new()
                .without_pruning()
                .with_input_types(&input_types)
                .with_output_types(&output_types)
                .without_boundary_shardings(),
        )
    }

    /// Builds a one-instruction program applying the given single-input operation to an input of the given data
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
        unproject_plain_program(program).to_rendered_mlir("main", &ProgramLoweringOptions::new().without_pruning())
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

    /// Builds a region over two inputs of `scalar_type` (i.e., a residual and a linear input) whose output is the
    /// product of its inputs, and that also runs the instructions that `extend` adds to it when `extended` is set.
    pub(crate) fn scalar_linear_region(
        scalar_type: &ArrayType,
        extended: bool,
        extend: impl Fn(&mut CoreProgramBuilder<Constant, Operation>, AtomId, AtomId) -> AtomId,
    ) -> FlatProgram<Session> {
        let scalar_type = Type::Array(scalar_type.clone());
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
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
    pub(crate) fn linear_call_program(
        forward: &FlatProgram<Session>,
        transpose: &FlatProgram<Session>,
    ) -> FlatProgram<Session> {
        let scalar_type = forward.input_types()[0].clone();
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
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

    /// Operation family whose operations borrow their configuration, which lowering admits for any lifetime of the
    /// borrow. Each operation returns the inputs at the positions that its configuration lists.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct BorrowingOperation<'p> {
        positions: &'p [usize],
    }

    impl CoreOperation for BorrowingOperation<'_> {
        type Type = Type;

        fn name(&self) -> &'static str {
            "borrowing"
        }

        fn infer_output_types(
            &self,
            input_types: &[Type],
            region_interfaces: &[RegionInterface<Type>],
        ) -> Result<Vec<Type>, TypeError> {
            let _ = region_interfaces;
            Ok(self.positions.iter().map(|position| input_types[*position].clone()).collect())
        }
    }

    impl OperationPayloadProjection for BorrowingOperation<'_> {
        fn project_payload(&self, payload: TypeId) -> Option<&dyn Any> {
            let _ = payload;
            None
        }

        fn from_payload(payload: ErasedOperation) -> Result<Self, ErasedOperation> {
            Err(payload)
        }
    }

    impl LowerableOperation for BorrowingOperation<'_> {
        fn lower<'b, 'c: 'b, 't: 'c, O: ProgramLoweringOperation>(
            &self,
            lowerer: &mut Lowerer<'b, 'c, 't, O>,
            input_values: &[ValueRef<'b, 'c, 't>],
            input_types: &[Type],
            output_types: &[Type],
        ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
            let _ = (lowerer, input_types, output_types);
            Ok(self.positions.iter().map(|position| input_values[*position]).collect())
        }
    }

    #[test]
    fn test_lowerable_operation_family_admits_families_that_borrow() {
        // A family whose operations borrow their configuration is a family that lowering lowers, because what lowering
        // must know of the calls of a family it asks of that family (refer to `LowerableOperation::unread_captures`)
        // rather than recovering it from a program that lives for as long as the whole program does.
        let r#type = ArrayType::new_static(DataType::F32, [3]);
        let positions = vec![1, 0];
        let mut builder = CoreProgramBuilder::<Constant, BorrowingOperation<'_>>::new();
        let first = builder.add_input(r#type.clone().into());
        let second = builder.add_input(r#type.clone().into());
        let operation = BorrowingOperation { positions: &positions };
        let outputs = builder.add_instruction(operation, Vec::new(), vec![first, second], None).unwrap().to_vec();
        let program =
            builder.build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2]);
        let types = vec![r#type.clone(), r#type];
        let module = program
            .unwrap()
            .to_rendered_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_input_types(&types)
                    .with_output_types(&types)
                    .without_boundary_shardings(),
            )
            .unwrap();
        assert_eq!(
            module,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>) {
                    return %arg1, %arg0 : tensor<3xf32>, tensor<3xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_lowerable_operation_validate_data_derived_dimensions() {
        // A family that does not state whether its operations keep the padding of data-derived dimensions
        // unobservable is not assumed to be safe.
        let operation = BorrowingOperation { positions: &[0] };
        assert_eq!(
            operation.validate_data_derived_dimensions(),
            Err(Error::from(ProgramError::UnsupportedOperation {
                message: "operation `borrowing` cannot consume data-derived dimensions during XLA lowering because \
                          its operation family does not state how it keeps the padding of data-derived dimensions \
                          unobservable"
                    .to_string(),
            })),
        );
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
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
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
