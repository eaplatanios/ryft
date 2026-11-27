use std::borrow::Cow;
use std::collections::HashMap;
use std::rc::Rc;

use ryft_core::{
    ArrayType, AtomId, BroadcastOperation, CallOperation, DeviceMesh, ExternalReferenceBinding, LogicalMesh,
    MeshAxisType, Parameterized, PlaceOperation, PlacementRequest, Program, ProgramError, RegionId,
    RegionPlacementRequests, RegionRef, RequestedPlacement, Sharding, ShardingDimension, StopGradientOperation,
    TagOperation, TransferToMemoryOperation, Type as CoreType, Typed, Value,
};

use crate::lowering::emissions::{collect_session_constants, omits_regions};
use crate::lowering::modules::{LoweredModule, render_devices};
use crate::lowering::{LowerableOperationFamily, LoweringError, RaggedDotLoweringStrategy};
use crate::{Constant, Error, RetainedArray, Type};

/// Options for lowering a staged [`Program`] with [`LowerableProgram`]. By default, the entry signature is read from
/// the program. The global types of its inputs and outputs are those of the program, and the arrays that the program
/// imports (i.e., its session constants) enter the entry function as hidden arguments that precede the inputs. The
/// arguments carry the sharding of each of their global types that has one, with the hidden arguments placed as
/// their arrays are. The results carry the placement that the program gives them (i.e., the target of a
/// [`place`](PlaceOperation) that produces them, including over auto mesh axes, or the sharding of the argument that
/// they return, also through the callee of a `call`) or otherwise the sharding of their global types. Arguments and
/// results replicate the unconstrained dimensions of their shardings, which their runtime placements read as
/// replicated, and results whose placement over auto mesh axes nothing fixes are left open over those axes, so that the
/// compiler chooses it. A value without a sharding is left unannotated, so that Shardy propagates its placement from
/// the program.
///
/// The program is pruned before it is lowered, which drops the dead outputs of instructions that carry regions together
/// with the body computation that only they use (e.g., the dead primal outputs of a forward mode `shard_map` whose
/// residuals feed a gradient), and is then simplified. A `shard_map` whose ordinary outputs are all dead but whose body
/// has observable effects (e.g., prints with the [`DeviceOrderedIo`](ryft_core::EffectClass::DeviceOrderedIo) effect)
/// is kept with no ordinary outputs and lowers to a manual computation that only threads its effect token.
#[derive(Clone, Debug)]
pub struct ProgramLoweringOptions {
    /// Types of the hidden capture arguments, or [`None`] for the types of the session constants of the program.
    capture_types: Option<Vec<ArrayType>>,

    /// Global types of the inputs, or [`None`] for those of the program.
    input_types: Option<Vec<ArrayType>>,

    /// Global types of the outputs, or [`None`] for those of the program.
    output_types: Option<Vec<ArrayType>>,

    /// Shardings of the inputs that replace those of their global types.
    input_shardings: Option<Vec<Sharding>>,

    /// Shardings of the outputs that replace those that the program and their global types give them.
    output_shardings: Option<Vec<Sharding>>,

    /// Shardings of the entry function arguments that are attached as they are, in place of derived ones.
    exact_input_shardings: Option<EntryShardingOverride>,

    /// Shardings of the entry function results that are attached as they are, in place of derived ones.
    exact_output_shardings: Option<EntryShardingOverride>,

    /// Whether the program is pruned and simplified before it is lowered.
    prune: bool,

    /// Platform that the program is lowered for.
    target_platform: Option<String>,

    /// Whether the target supports the Shardy dialect.
    supports_shardy: bool,

    /// External reference state recipes whose aliases the entry boundary records.
    reference_bindings: Vec<ExternalReferenceBinding>,

    /// Public inputs whose buffers the caller gives up to every execution.
    donated_inputs: Vec<bool>,

    /// Public inputs that the program never reads.
    unread_inputs: Vec<bool>,

    /// Whether the entry function exports the runtime ordered I/O token chain as an extra argument and result.
    export_effects: bool,

    /// Representation of ragged dots.
    ragged_dot_lowering_strategy: RaggedDotLoweringStrategy,

    /// Results that are lowered open over auto mesh axes, or [`None`] to use the output sharding defaults.
    open_outputs: Option<Vec<bool>>,
}

impl ProgramLoweringOptions {
    /// Creates the default [`ProgramLoweringOptions`], which the documentation of this type describes.
    #[inline]
    pub fn new() -> Self {
        Self {
            capture_types: None,
            input_types: None,
            output_types: None,
            input_shardings: None,
            output_shardings: None,
            exact_input_shardings: None,
            exact_output_shardings: None,
            prune: true,
            target_platform: None,
            supports_shardy: true,
            reference_bindings: Vec::new(),
            donated_inputs: Vec::new(),
            unread_inputs: Vec::new(),
            export_effects: false,
            ragged_dot_lowering_strategy: RaggedDotLoweringStrategy::default(),
            open_outputs: None,
        }
    }

    /// Returns these options with `capture_types` as the types of the hidden capture arguments that precede the inputs,
    /// in the order of the capture table, in place of the types of the session constants of the program. The types must
    /// end with the types of those session constants.
    #[inline]
    pub fn with_capture_types(mut self, capture_types: Vec<ArrayType>) -> Self {
        self.capture_types = Some(capture_types);
        self
    }

    /// Returns these options with `input_types` as the global types of the inputs, in place of those of the program.
    #[inline]
    pub fn with_input_types<Input: Parameterized<ArrayType>>(mut self, input_types: &Input) -> Self {
        self.input_types = Some(input_types.parameters().cloned().collect());
        self
    }

    /// Returns these options with `output_types` as the global types of the outputs, in place of those of the program.
    #[inline]
    pub fn with_output_types<Output: Parameterized<ArrayType>>(mut self, output_types: &Output) -> Self {
        self.output_types = Some(output_types.parameters().cloned().collect());
        self
    }

    /// Returns these options with `input_shardings` as the shardings of the inputs, one per input, in place of those
    /// of their global types. This is what the XLA Single Program Multiple Data (SPMD) partitioner reads to drive
    /// boundary slicing of the buffers of each device, including for shapes whose dimensions are not divisible by
    /// the partition count (e.g., shape `[5]` on 2 partitions producing `[3]` + `[2]`).
    #[inline]
    pub fn with_input_shardings(mut self, input_shardings: Vec<Sharding>) -> Self {
        self.input_shardings = Some(input_shardings);
        self
    }

    /// Returns these options with `output_shardings` as the shardings of the outputs, one per output, in place of
    /// those that the program and their global types give them.
    #[inline]
    pub fn with_output_shardings(mut self, output_shardings: Vec<Sharding>) -> Self {
        self.output_shardings = Some(output_shardings);
        self
    }

    /// Returns these options with `input_shardings` attached to the entry function arguments as they are (i.e.,
    /// one optional sharding per argument, with the hidden capture arguments first), in place of the shardings that
    /// the default options derive. An argument without a sharding is left unannotated.
    #[inline]
    pub fn with_exact_input_shardings(mut self, input_shardings: Vec<Option<Sharding>>) -> Self {
        self.exact_input_shardings = Some(EntryShardingOverride::Values(input_shardings));
        self
    }

    /// Returns these options with `output_shardings` attached to the entry function results as they are (i.e., one
    /// optional sharding per result), in place of the shardings that the default options derive. A result without a
    /// sharding is left unannotated. Results are closed over auto mesh axes by default. Use
    /// [`Self::with_open_outputs`] to choose which annotated results to leave open.
    #[inline]
    pub fn with_exact_output_shardings(mut self, output_shardings: Vec<Option<Sharding>>) -> Self {
        self.exact_output_shardings = Some(EntryShardingOverride::Values(output_shardings));
        self
    }

    /// Returns these options with no sharding attached to any entry function argument or result, so that Shardy
    /// propagates every placement from the program.
    #[inline]
    pub fn without_boundary_shardings(mut self) -> Self {
        self.exact_input_shardings = Some(EntryShardingOverride::Unannotated);
        self.exact_output_shardings = Some(EntryShardingOverride::Unannotated);
        self
    }

    /// Returns these options with the program lowered as it is, without pruning or simplifying it first.
    #[inline]
    pub fn without_pruning(mut self) -> Self {
        self.prune = false;
        self
    }

    /// Returns these options with the module lowered for `target_platform`.
    ///
    /// Use the case-sensitive PJRT name returned by [`ryft_pjrt::Client::platform_name`] (e.g., `"cpu"`, `"cuda"`, or
    /// `"rocm"`). [`None`] uses the default lowering rules without selecting a platform. The name selects
    /// platform-specific lowering rules. For lowering that also takes a client's detected capabilities into account,
    /// use a [`Session`](crate::Session).
    #[inline]
    pub fn with_target_platform(mut self, target_platform: Option<&str>) -> Self {
        self.target_platform = target_platform.map(str::to_string);
        self
    }

    /// Returns these options with `supports_shardy` as whether the target supports the Shardy dialect.
    /// Lowering for a target without it removes the Shardy operations from the module.
    #[inline]
    pub(crate) fn with_shardy_support(mut self, supports_shardy: bool) -> Self {
        self.supports_shardy = supports_shardy;
        self
    }

    /// Returns these options with the logical external reference state aliases of a discharged program recorded on its
    /// entry boundary, and with the entry function exporting its ordered I/O token chain.
    ///
    /// `reference_bindings` are the external state recipes of the discharge artifact, and each must name a distinct
    /// logical state input. Each state input must survive the executable boundary. A mutated state must be a static
    /// device array and name a hidden output of the identical type and sharding. Mutated pairs are recorded as
    /// `tf.aliasing_output` aliases between inputs and outputs, which are non-semantic hints that the buffers may alias
    /// and that merely permit backend buffer reuse. The runtime never donates reference state inputs, so an alias never
    /// authorizes in-place mutation of the state buffer of the caller. Read-only state may be finite and bounded
    /// dynamic because it has no hidden alias. The validation here owns the physical alias contract, and the XLA
    /// session additionally enforces the supported memory, sharding, and dynamic shape classes before calling it.
    ///
    /// `donated_inputs` flags the public inputs whose buffers the caller gives up to every execution (i.e., one flag
    /// per public input, or none at all). Each donated input is paired with an output that it can share its buffers
    /// with and the pair is recorded as the same kind of alias, as JAX does for the donated arguments of a `jit`
    /// function. An output can share the buffers of an input when both are static device arrays of the identical type
    /// and sharding, or, for an input without such an output, when both have shardings under which every device holds
    /// shards of the same type (e.g., a donated input with a pending sum and an output that completes it, as in the
    /// compiled reshard of a donated array), and in both cases only when the output is not open on a mesh of more than
    /// one device, because the compiler could then place it differently than the input. Without such an alias the
    /// outputs of every execution are written to new buffers, which is correct but forces backends that cache launch
    /// state per buffer address (e.g., the command buffers of the XLA GPU backend) to rebuild that state for every
    /// execution of a program that updates its inputs in a loop.
    ///
    /// `unread_inputs` flags the public inputs that the program never reads (i.e., one flag per public input, or none
    /// at all), which the caller states because it cannot be read from `program` (e.g., an input of the program of an
    /// eager call that the call passes as a capture which its callee never reads). Such an input with a static shape
    /// has no executable argument, like a capture that no emitted region reads.
    #[inline]
    pub(crate) fn with_reference_bindings(
        mut self,
        reference_bindings: Vec<ExternalReferenceBinding>,
        donated_inputs: Vec<bool>,
        unread_inputs: Vec<bool>,
    ) -> Self {
        self.reference_bindings = reference_bindings;
        self.donated_inputs = donated_inputs;
        self.unread_inputs = unread_inputs;
        self.export_effects = true;
        self
    }

    /// Returns these options with `strategy` as the representation of ragged dots.
    #[inline]
    pub fn with_ragged_dot_lowering_strategy(mut self, strategy: RaggedDotLoweringStrategy) -> Self {
        self.ragged_dot_lowering_strategy = strategy;
        self
    }

    /// Returns these options with the annotated results flagged in `open_outputs` lowered open over auto mesh axes,
    /// so that the compiler chooses their placement over those axes. Supply one flag per logical result in result
    /// order, or an empty vector to close every annotated result. Lowering returns
    /// [`CompilationError::InvalidOptions`](crate::CompilationError::InvalidOptions) for any other length.
    #[inline]
    pub fn with_open_outputs(mut self, open_outputs: Vec<bool>) -> Self {
        self.open_outputs = Some(open_outputs);
        self
    }

    /// Returns the hidden capture argument type overrides, or [`None`] to use the types of the session constants.
    #[inline]
    pub fn capture_types(&self) -> Option<&[ArrayType]> {
        self.capture_types.as_deref()
    }

    /// Returns the global input type overrides, or [`None`] to use the program's input types.
    #[inline]
    pub fn input_types(&self) -> Option<&[ArrayType]> {
        self.input_types.as_deref()
    }

    /// Returns the global output type overrides, or [`None`] to use the program's output types.
    #[inline]
    pub fn output_types(&self) -> Option<&[ArrayType]> {
        self.output_types.as_deref()
    }

    /// Returns the public input sharding overrides used to derive entry argument annotations, if supplied.
    #[inline]
    pub fn input_shardings(&self) -> Option<&[Sharding]> {
        self.input_shardings.as_deref()
    }

    /// Returns the output sharding overrides used to derive entry result annotations, if supplied.
    #[inline]
    pub fn output_shardings(&self) -> Option<&[Sharding]> {
        self.output_shardings.as_deref()
    }

    /// Returns the exact entry argument sharding override, if supplied, including any hidden captures first.
    #[inline]
    pub fn exact_input_shardings(&self) -> Option<&EntryShardingOverride> {
        self.exact_input_shardings.as_ref()
    }

    /// Returns the exact entry result sharding override, if supplied.
    #[inline]
    pub fn exact_output_shardings(&self) -> Option<&EntryShardingOverride> {
        self.exact_output_shardings.as_ref()
    }

    /// Returns whether lowering prunes and simplifies the program first.
    #[inline]
    pub fn prunes(&self) -> bool {
        self.prune
    }

    /// Returns the case-sensitive PJRT platform name used for lowering, if supplied.
    #[inline]
    pub fn target_platform(&self) -> Option<&str> {
        self.target_platform.as_deref()
    }

    /// Returns whether the target supports the Shardy dialect.
    #[inline]
    pub(crate) fn supports_shardy(&self) -> bool {
        self.supports_shardy
    }

    /// Returns the external reference state recipes whose aliases the entry boundary records.
    #[inline]
    pub(crate) fn reference_bindings(&self) -> &[ExternalReferenceBinding] {
        &self.reference_bindings
    }

    /// Returns the flags of the public inputs whose buffers the caller gives up to every execution.
    #[inline]
    pub(crate) fn donated_inputs(&self) -> &[bool] {
        &self.donated_inputs
    }

    /// Returns the flags of the public inputs that the program never reads.
    #[inline]
    pub(crate) fn unread_inputs(&self) -> &[bool] {
        &self.unread_inputs
    }

    /// Returns whether the entry function exports the runtime ordered I/O token chain as an extra argument and result.
    #[inline]
    pub(crate) fn exports_effects(&self) -> bool {
        self.export_effects
    }

    /// Returns the representation of ragged dots.
    #[inline]
    pub fn ragged_dot_lowering_strategy(&self) -> RaggedDotLoweringStrategy {
        self.ragged_dot_lowering_strategy
    }

    /// Returns the result flags for openness over auto mesh axes, or [`None`] to use the output sharding defaults.
    /// An empty slice closes every annotated result.
    #[inline]
    pub fn open_outputs(&self) -> Option<&[bool]> {
        self.open_outputs.as_deref()
    }
}

impl Default for ProgramLoweringOptions {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Exact entry boundary sharding overrides reported by [`ProgramLoweringOptions`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EntryShardingOverride {
    /// Leave every value unannotated so that Shardy propagates placements from the program.
    Unannotated,

    /// Attach the supplied shardings in signature order and leave values with [`None`] unannotated.
    Values(Vec<Option<Sharding>>),
}

/// Lowering of staged [`Program`]s of the XLA backend, over any [`LowerableOperationFamily`], to StableHLO/Shardy MLIR
/// modules. References and mutable state must have been discharged from a program before it is lowered.
pub trait LowerableProgram {
    /// Renders a full StableHLO/Shardy MLIR module for this program with the default [`ProgramLoweringOptions`].
    ///
    /// # Parameters
    ///
    ///   - `function_name`: Symbol name to use for the outer `func.func` operation.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::lower`].
    #[inline]
    fn to_mlir_module<S: AsRef<str>>(&self, function_name: S) -> Result<String, Error> {
        self.to_mlir_module_with_options(function_name, &ProgramLoweringOptions::new())
    }

    /// Renders a full StableHLO/Shardy MLIR module for this program according to `options`.
    ///
    /// # Parameters
    ///
    ///   - `function_name`: Symbol name to use for the outer `func.func` operation.
    ///   - `options`: Options that determine the entry signature and how the program is prepared.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::lower`].
    #[inline]
    fn to_mlir_module_with_options<S: AsRef<str>>(
        &self,
        function_name: S,
        options: &ProgramLoweringOptions,
    ) -> Result<String, Error> {
        Ok(self.lower(function_name, options)?.into_stable_hlo())
    }

    /// Lowers this program according to `options` and returns its [`LoweredModule`], which pairs the textual module
    /// with what executing it requires.
    ///
    /// # Parameters
    ///
    ///   - `function_name`: Symbol name to use for the outer `func.func` operation.
    ///   - `options`: Options that determine the entry signature and how the program is prepared.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Core`] when an input or output of the program whose type the options do not provide is not an
    /// array, when pruning or simplifying the program fails, or when the program imports an array whose type names a
    /// mesh of another topology than the meshes of the signature, because a plain trace keeps the type of an array that
    /// it imports (a function compiled through a [`Session`](crate::Session), whose captures follow its call, receives
    /// it where it runs instead). Lowering failures use the corresponding [`Error`] variants (e.g.,
    /// [`LoweringError::UnresolvedReference`] or [`LoweringError::UnresolvedState`] for undischarged references or
    /// state, [`LoweringError::InvalidFunctionName`] for an invalid function name, mismatched signature shardings,
    /// or operations that the backend cannot lower).
    fn lower<S: AsRef<str>>(&self, function_name: S, options: &ProgramLoweringOptions) -> Result<LoweredModule, Error>;
}

impl<O, Input, Output> LowerableProgram for Program<Constant, O, Input, Output>
where
    O: LowerableOperationFamily,
    Input: Parameterized<Constant>,
    Output: Parameterized<Constant>,
{
    fn lower<S: AsRef<str>>(&self, function_name: S, options: &ProgramLoweringOptions) -> Result<LoweredModule, Error> {
        let target_platform = options.target_platform();
        let pruned;
        let program = if options.prune {
            pruned = self.pruned_for_lowering(target_platform)?;
            &pruned
        } else {
            self
        };

        // Only arrays cross the boundary of an executable, so the types that the program provides must be arrays.
        let as_array_types = |types: Vec<Type>| {
            types
                .iter()
                .map(|r#type| <&ArrayType>::try_from(r#type).cloned().map_err(ProgramError::from))
                .collect::<Result<Vec<_>, _>>()
        };
        let input_types = match &options.input_types {
            Some(input_types) => input_types.clone(),
            None => as_array_types(self.entry_region_ref().input_types())?,
        };
        let output_types = match &options.output_types {
            Some(output_types) => output_types.clone(),
            None => as_array_types(self.entry_region_ref().output_types())?,
        };

        // Arrays that session constants refer to enter the module as hidden arguments that precede the inputs,
        // which are annotated as their runtime arrays are placed, as compilation for a `Session` passes them.
        let session_constants = match &options.capture_types {
            Some(_) => None,
            None => Some(collect_session_constants(program, target_platform)),
        };
        let capture_types = match (&options.capture_types, &session_constants) {
            (Some(capture_types), _) => capture_types.clone(),
            (None, Some(constants)) => constants.iter().map(|constant| constant.r#type().into_owned()).collect(),
            (None, None) => Vec::new(),
        };
        let exact_arguments = match &options.exact_input_shardings {
            Some(EntryShardingOverride::Values(shardings)) => Some(shardings.clone()),
            Some(EntryShardingOverride::Unannotated) => Some(vec![None; capture_types.len() + input_types.len()]),
            None => None,
        };
        let input_shardings: Vec<Option<Sharding>> = match (&exact_arguments, &options.input_shardings) {
            (Some(shardings), _) => shardings.iter().skip(capture_types.len()).cloned().collect(),
            (None, Some(shardings)) => shardings.iter().cloned().map(Some).collect(),
            (None, None) => input_types.iter().map(|r#type| r#type.sharding().cloned()).collect(),
        };

        // A plain trace preserves the types of the arrays that enter it (i.e., `CapturePlacement::Preserve`), so an
        // array that it imports keeps its own type, which names the mesh of the array, and a signature over a mesh of
        // another topology cannot hold it. This is reported with its cause and the way forward, rather than as the
        // disagreement of the meshes that lowering would find.
        if let Some(session_constants) = &session_constants {
            let mesh = input_shardings
                .iter()
                .flatten()
                .chain(options.output_shardings.iter().flatten())
                .chain(output_types.iter().filter_map(ArrayType::sharding))
                .map(Sharding::mesh)
                .next();
            if let Some(mesh) = mesh {
                for constant in session_constants {
                    let declared_type = constant.r#type();
                    if let Some(sharding) = declared_type.sharding()
                        && !sharding.mesh().has_same_topology(mesh)
                    {
                        return Err(ProgramError::InvalidArgument {
                            message: format!(
                                "the traced program imports an array of type `{declared_type}`, whose mesh does not \
                                 have the topology of the mesh `{mesh}` of the signature of the module, because a \
                                 plain `trace` keeps the type of an array that it imports; stage or compile the \
                                 function for a device mesh over `{mesh}` through a `Session` instead (e.g., with \
                                 `stage` or `compile`), which receives the array on that mesh",
                            ),
                        }
                        .into());
                    }
                }
            }
        }

        let argument_shardings = match exact_arguments {
            Some(shardings) => shardings,
            None => {
                let capture_shardings: Vec<Option<Sharding>> = match &session_constants {
                    Some(constants) => constants.iter().map(|constant| Some(constant.sharding().clone())).collect(),
                    None => capture_types.iter().map(|r#type| r#type.sharding().cloned()).collect(),
                };
                capture_shardings
                    .iter()
                    .chain(&input_shardings)
                    .map(|sharding| {
                        sharding
                            .as_ref()
                            .map(ShardingLoweringExtension::with_unconstrained_dimensions_replicated)
                            .transpose()
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
        };

        let (output_shardings, open_outputs) = match &options.exact_output_shardings {
            Some(EntryShardingOverride::Values(shardings)) => {
                (shardings.clone(), options.open_outputs.clone().unwrap_or_default())
            }
            Some(EntryShardingOverride::Unannotated) => {
                (vec![None; output_types.len()], options.open_outputs.clone().unwrap_or_default())
            }
            None => {
                let explicit = options.output_shardings.as_deref();
                let sources =
                    program.output_sharding_sources(output_types.len(), |constant| constant.sharding().clone());
                let shardings = entry_output_shardings(&output_types, explicit, &sources, &input_shardings)?;
                let open_outputs = match &options.open_outputs {
                    Some(open_outputs) => open_outputs.clone(),
                    None => entry_outputs_with_compiler_chosen_placement(&output_types, explicit, &sources, &shardings),
                };
                (shardings, open_outputs)
            }
        };

        LoweredModule::new(
            program,
            function_name.as_ref(),
            &capture_types,
            &input_types,
            &output_types,
            &argument_shardings,
            &output_shardings,
            &open_outputs,
            options,
        )
    }
}

/// Prepares XLA programs for lowering and analyzes their entry placements.
pub(crate) trait ProgramLoweringExtension: Sized {
    /// Operation family of the program.
    type Operation: LowerableOperationFamily;

    /// Returns this program as lowering emits it, without the unused outputs of its instructions that carry regions,
    /// together with the computations that only those outputs use (i.e., like [`Program::into_pruned`]), and without
    /// every other instruction that neither an output nor an observable effect needs (i.e., like
    /// [`Program::into_simplified`]). Every entry point that lowers a program (i.e., the compilation of a
    /// [`Session`](crate::Session) and the StableHLO export of traced programs) lowers this program, and
    /// [`RegionPlacementRequests::requests`] analyzes it, so that the validation of placements, the lowering itself,
    /// commitment, and the devices that a program names all see the same instructions. Whether a call still names
    /// devices is read for `target_platform`, so a callee that names devices only inside a body that lowering for
    /// that platform replaces names none.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Program::into_pruned`] and [`Program::into_simplified`].
    fn pruned_for_lowering(&self, target_platform: Option<&str>) -> Result<Self, ProgramError>;

    /// Returns whether this program requests the placement of a value explicitly in any of the regions that lowering
    /// emits, through a `place` with a target, a `constrain_sharding`, or a `call` whose callee names devices (i.e.,
    /// carries a sharding override of its boundary or requests a placement itself), whether or not that value reaches
    /// a result. Only a request of the program that lowering emits (i.e., [`Self::pruned_for_lowering`]) counts, so a
    /// wrapper that drops the only result of a callee that places a value requests nothing. Such calls count because
    /// their callees cannot show their overrides (e.g., a replicated override over auto mesh axes stages nothing).
    /// The callees of calls are among the regions that lowering emits, so a request inside a callee counts too.
    /// A `place` without a target leaves its input where it is, so it requests nothing. A `place` with a device
    /// target (i.e., a [`PlacementRequest::Device`]) is counted, as an explicit device commits an array eagerly.
    /// A `place` with an in-place target (i.e., a [`PlacementRequest::InPlace`]) is counted too. A `reshard` is
    /// not counted.
    ///
    /// With `includes_memory_requests` unset, an in-place target is not counted, because it names no devices. This is
    /// how a compiled function decides whether it names the devices of its mesh, and commits its outputs there.
    ///
    /// The program is analyzed for `target_platform`, so a request inside a body that lowering for that platform
    /// replaces is not counted, and with [`None`] no region is replaced.
    fn has_placement_requests(&self, target_platform: Option<&str>, includes_memory_requests: bool) -> bool
    where
        Self::Operation: 'static;

    /// Returns where the sharding of each of the first `output_count` outputs of this program comes from (refer to
    /// [`OutputShardingSource`]), or [`None`] for an output whose sharding the program does not determine. An output
    /// that a `place` in the entry region produces has the placement of its target, which states the placement along
    /// every mesh axis, including the auto mesh axes that the output type drops. A bufferless output of `condition`,
    /// `scan`, or `while` uses its declared sharding or an entry output override, because the compiler has no output
    /// buffer from which to report a placement along auto axes. An output that is an entry argument is placed where
    /// that argument enters the program. An output that returns a [`RetainedArray`] is placed as the hidden argument
    /// that supplies it, whose sharding `constant_sharding` returns.
    ///
    /// Operations that return the value of an input together with its placement are followed back to that input, so
    /// that an output keeps the placement of a `place` or of an entry argument that precedes them. These are a `place`
    /// without a target, which leaves its input where it is, a `place` with a device target, which keeps the sharding
    /// of its input on the only device of the program mesh, `transfer_to_memory`, which only changes the memory of its
    /// input as a `place` with an in-place target does, `stop_gradient` and `tag`, which only mark values for program
    /// transforms, and identity `broadcast` operations that forget a compatible global sharding on an explicit mesh.
    /// A device target that completes a pending reduction over mesh axes of size one gives its output another sharding
    /// than its input, so such an output is placed as its type states instead. A `call` always runs its callee,
    /// whose outputs are the outputs of the call, so an output of a call has the placement that the callee gives the
    /// corresponding callee output, and a callee output that returns a callee input is followed back to the
    /// corresponding input of the call. The boundary of the callee takes precedence, because the callee body cannot
    /// always show it, so an output of the call that a sharding override of a callee output covers is placed as that
    /// override, and one that returns a callee input with an effective boundary sharding (i.e., its override or
    /// otherwise where the executable of the callee places it, refer to [`CallOperation::boundary_input_sharding`])
    /// is placed as that sharding. Lowering constrains that output or operand of the call to that sharding as well,
    /// so that the call places it alike whether a program returns it or consumes it (refer to the lowering of
    /// [`CallOperation`]). Other operations that return a value equal to an input are not followed, because they
    /// change its placement (e.g., `reshard` and `constrain_sharding`) or because the compiler is free to place their
    /// outputs differently (e.g., the `optimization_barrier` of rematerialization).
    fn output_sharding_sources<F: Fn(&RetainedArray) -> Sharding>(
        &self,
        output_count: usize,
        constant_sharding: F,
    ) -> Vec<Option<OutputShardingSource>>;
}

impl<O, Input, Output> ProgramLoweringExtension for Program<Constant, O, Input, Output>
where
    O: LowerableOperationFamily,
    Input: Parameterized<Constant>,
    Output: Parameterized<Constant>,
{
    type Operation = O;

    #[inline]
    fn pruned_for_lowering(&self, target_platform: Option<&str>) -> Result<Self, ProgramError> {
        O::without_stale_call_meshes(self.clone().into_pruned()?.into_simplified()?, target_platform)
    }

    #[inline]
    fn has_placement_requests(&self, target_platform: Option<&str>, includes_memory_requests: bool) -> bool
    where
        Self::Operation: 'static,
    {
        RegionPlacementRequests::for_lowering(self.entry_region_ref(), target_platform)
            .has_requests(includes_memory_requests)
    }

    #[inline]
    fn output_sharding_sources<F: Fn(&RetainedArray) -> Sharding>(
        &self,
        output_count: usize,
        constant_sharding: F,
    ) -> Vec<Option<OutputShardingSource>> {
        /// Returns the source of the sharding of each output of `region`, which
        /// [`ProgramLoweringExtension::output_sharding_sources`] describes, with [`OutputShardingSource::Input`]
        /// naming an input of `region`. An output that returns a [`RetainedArray`] has the placement of that array,
        /// which enters the executable as a hidden argument with that placement, and which has no buffer whose
        /// placement the compiler could report when it has the zero data type. The placements of the callee regions
        /// of `call`s are computed once per callee and memoized in `callee_sources`, because repeated calls share their
        /// callee region. The inputs of `region` and the producer of each of its values, together with the position of
        /// that value among the outputs of its producer, are indexed once, so that following an output back to an input
        /// costs one lookup per step rather than a scan of the inputs or of the outputs of an instruction.
        fn region_output_sharding_sources<O: LowerableOperationFamily, F: Fn(&RetainedArray) -> Sharding>(
            region: RegionRef<'_, Constant, O>,
            constant_sharding: &F,
            callee_sources: &mut HashMap<RegionId, Rc<[Option<OutputShardingSource>]>>,
        ) -> Rc<[Option<OutputShardingSource>]> {
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

            let mut output_source = |output_id: AtomId| {
                let mut value_id = output_id;
                loop {
                    let Some((instruction_index, output_index)) = producers[value_id.index()] else {
                        if let Some(Constant::Session(constant)) = region.atoms()[value_id.index()].as_constant() {
                            return Some(OutputShardingSource::Determined(constant_sharding(constant)));
                        }

                        return input_positions[value_id.index()].map(OutputShardingSource::Input);
                    };

                    let instruction = &region.instructions()[instruction_index];
                    let operation = instruction.operation();
                    let input_index = if let Some(operation) = operation.projected_payload::<PlaceOperation>() {
                        match operation.target() {
                            Some(PlacementRequest::Resolved(target)) => {
                                return Some(OutputShardingSource::Determined(target.sharding().clone()));
                            }
                            Some(PlacementRequest::Device { .. }) => {
                                // A device target completes a pending reduction over mesh axes of size one, so its
                                // result can have another sharding than its input, which is then the placement of the
                                // result.
                                let sharding = |atom: AtomId| match region.atoms()[atom.index()].r#type().as_ref() {
                                    Type::Array(array_type) => array_type.sharding().cloned(),
                                    _ => None,
                                };

                                let output_sharding = sharding(instruction.outputs()[output_index]);
                                if output_sharding != sharding(*instruction.inputs().first()?) {
                                    return output_sharding.map(OutputShardingSource::Determined);
                                }

                                0
                            }
                            Some(PlacementRequest::InPlace { .. }) | None => {
                                // An in-place target leaves its input on the devices where it lives with its sharding,
                                // so the result is placed as its input is.
                                0
                            }
                        }
                    } else if operation.projected_payload::<StopGradientOperation<ArrayType>>().is_some()
                        || operation.projected_payload::<TagOperation<ArrayType>>().is_some()
                        || operation.projected_payload::<TransferToMemoryOperation>().is_some()
                    {
                        // These operations return each of their inputs as the output at the same position, and a memory
                        // transfer keeps the devices and the sharding of its input like an in-place `place` target
                        // does.
                        output_index
                    } else if let Some(broadcast) = operation.projected_payload::<BroadcastOperation>() {
                        let source = region.atoms()[instruction.inputs().first()?.index()].r#type();
                        let Type::Array(source) = source.as_ref() else { return None };
                        let target = broadcast.output_type();

                        // A transpose can forget tracked sharding to match an untracked cotangent type. Keep the split
                        // chosen by an explicit placement. An auto mesh still leaves this choice to the partitioner.
                        let explicit_mesh = source.sharding().is_some_and(|sharding| {
                            sharding.mesh().axes().iter().all(|axis| axis.r#type() == MeshAxisType::Explicit)
                        });
                        if !explicit_mesh
                            || target.sharding().is_some()
                            || source.shape() != target.shape()
                            || !broadcast.output_axes().iter().copied().eq(0..source.rank())
                            || !target.is_refined_by(source)
                        {
                            return None;
                        }

                        0
                    } else {
                        let call = operation.projected_payload::<CallOperation<Type>>()?;

                        // The inputs of the callee receive the inputs of the call positionally, and the results of the
                        // call are the results of the callee. A sharding override of the boundary of the callee places
                        // the result or the forwarded input that it covers, as an execution of the callee does, which
                        // its body cannot always show (e.g., a replicated override over auto mesh axes stages nothing).
                        if let Some(sharding) = call.output_sharding(output_index) {
                            return Some(OutputShardingSource::Determined(sharding.clone()));
                        }

                        let callee = *instruction.regions().first()?;
                        let sources = match callee_sources.get(&callee) {
                            Some(sources) => Rc::clone(sources),
                            None => {
                                let sources = region_output_sharding_sources(
                                    region.with_id(callee).ok()?,
                                    constant_sharding,
                                    callee_sources,
                                );
                                callee_sources.insert(callee, Rc::clone(&sources));
                                sources
                            }
                        };

                        match sources.get(output_index)?.as_ref()? {
                            OutputShardingSource::Determined(sharding) => {
                                return Some(OutputShardingSource::Determined(sharding.clone()));
                            }
                            OutputShardingSource::Input(index) => {
                                let actual = region.atoms()[instruction.inputs().get(*index)?.index()].r#type();
                                let callee = region.with_id(callee).ok()?;
                                let declared = callee.input_types();
                                match (actual.as_ref(), declared.get(*index)?) {
                                    (Type::Array(actual), declared) => {
                                        match call.boundary_input_sharding(*index, actual, declared) {
                                            Some(sharding) => {
                                                return Some(OutputShardingSource::Determined(sharding.into_owned()));
                                            }
                                            None => *index,
                                        }
                                    }
                                    _ => *index,
                                }
                            }
                        }
                    };

                    value_id = *instruction.inputs().get(input_index)?;
                }
            };

            region.output_ids().iter().map(|&output_id| output_source(output_id)).collect()
        }

        let mut callee_sources = HashMap::new();
        region_output_sharding_sources(self.entry_region_ref(), &constant_sharding, &mut callee_sources)[..output_count]
            .to_vec()
    }
}

// TODO(eaplatanios): Should we merge the two functions of this trait into `ShardingExtension`?
/// Converts core shardings to the form in which runtime arrays store them.
pub(crate) trait ShardingLoweringExtension {
    /// Returns this sharding with its unconstrained dimensions replicated (refer to
    /// [`Self::dimensions_with_unconstrained_replicated`]), which is the sharding of a runtime array that is
    /// placed with this sharding. A runtime array stores such a dimension replicated, because its placement reads
    /// an unconstrained dimension as replicated. The entry function of a program therefore annotates a value at its
    /// boundary (i.e., an argument or an output) with this sharding, because an annotation that left the dimension
    /// open would let the compiler shard it and exchange buffers of other shapes than those of the array (e.g., expect
    /// `[2, 2]` argument buffers where an array placed as `[{"x"}, {?}]` supplies `[2, 4]` buffers).
    fn with_unconstrained_dimensions_replicated(&self) -> Result<Sharding, Error>;

    /// Returns the dimension shardings of this sharding with its unconstrained dimensions replicated, which is how a
    /// runtime array stores them and how a constraint that must state its pending reduction can declare them.
    fn dimensions_with_unconstrained_replicated(&self) -> Rc<[ShardingDimension]>;
}

impl ShardingLoweringExtension for Sharding {
    #[inline]
    fn with_unconstrained_dimensions_replicated(&self) -> Result<Sharding, Error> {
        if !self.dimensions().contains(&ShardingDimension::Unconstrained) {
            return Ok(self.clone());
        }
        Ok(self.with_dimensions(self.dimensions_with_unconstrained_replicated().to_vec())?)
    }

    #[inline]
    fn dimensions_with_unconstrained_replicated(&self) -> Rc<[ShardingDimension]> {
        self.dimensions()
            .iter()
            .map(|dimension| match dimension {
                ShardingDimension::Unconstrained => ShardingDimension::Replicated,
                dimension => dimension.clone(),
            })
            .collect()
    }
}

/// Derives the [`RegionPlacementRequests`] of the program that lowering emits for a region, and the devices that its
/// placement requests name.
pub(crate) trait RegionPlacementRequestsExtension: Sized {
    /// Returns the [`RegionPlacementRequests`] of the program that lowering emits for `region` and `target_platform`,
    /// every output of `region` being used (refer to [`ProgramLoweringExtension::pruned_for_lowering`]). For a target
    /// platform, the regions of the instructions whose lowering for that platform omits them are not analyzed (refer
    /// to [`RegionPlacementRequests::for_target`]), and [`None`] analyzes the program for no target (refer to
    /// [`RegionPlacementRequests::new`]). The summary is derived once for each region and platform and
    /// retained with the region.
    fn for_lowering<V, O>(region: RegionRef<'_, V, O>, target_platform: Option<&str>) -> Self
    where
        V: 'static + Value<Type = Type>,
        O: 'static + LowerableOperationFamily;

    /// Returns the devices that the placement requests of the program name (refer to
    /// [`RegionPlacementRequests::requests`]), as the mesh of a request that names devices together with the name
    /// of the operation that makes it, or [`None`] when no request names devices. A device target is resolved as
    /// [`RequestedPlacement`] documents, and a request over a named mesh of the same devices takes precedence over the
    /// canonical mesh (refer to [`merge_requested_placement`]), so that the result does not depend on the order of the
    /// requests. The caller merges the requests of other regions with [`merge_requested_placement`].
    ///
    /// # Errors
    ///
    /// Returns [`LoweringError::UnsupportedOperation`] when two requests name different devices, which no single mesh
    /// runs.
    fn requested_mesh(&self) -> Result<Option<(DeviceMesh, &'static str)>, Error>;
}

impl RegionPlacementRequestsExtension for RegionPlacementRequests {
    #[inline]
    fn for_lowering<V, O>(region: RegionRef<'_, V, O>, target_platform: Option<&str>) -> Self
    where
        V: 'static + Value<Type = Type>,
        O: 'static + LowerableOperationFamily,
    {
        match target_platform {
            None => Self::new(region),
            Some(target_platform) => Self::for_target(region, target_platform, |region, instruction| {
                omits_regions(region, instruction, Some(target_platform))
            }),
        }
    }

    #[inline]
    fn requested_mesh(&self) -> Result<Option<(DeviceMesh, &'static str)>, Error> {
        let mut requested = None;
        for (request, operation) in self.requests() {
            if let RequestedPlacement::Devices(mesh) = request {
                merge_requested_placement(&mut requested, Cow::Borrowed(mesh), operation)?;
            }
        }
        Ok(requested.map(|(mesh, operation)| (mesh.into_owned(), operation)))
    }
}

/// Source of the sharding of one output of a program, as [`ProgramLoweringExtension::output_sharding_sources`]
/// returns it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum OutputShardingSource {
    /// The program determines the sharding of the output.
    Determined(Sharding),

    /// The output inherits the sharding of the region input at this index.
    Input(usize),
}

/// Merges `mesh`, the devices that a placement request of the operation named `operation` names, into `requested`, the
/// request that names devices among those seen before it, if any, together with the name of its operation. A request
/// over the canonical mesh of one device (i.e., [`LogicalMesh::single_device`], over which a device target whose input
/// tracks no mesh is resolved, refer to [`RegionPlacementRequestsExtension::requested_mesh`]) gives way to a request
/// over another logical mesh of the same device, whichever comes first, so that a bare device and a named mesh of that
/// device select the named mesh in either order. Requests over two named meshes of the same devices keep the first.
///
/// # Errors
///
/// Returns [`LoweringError::UnsupportedOperation`] when `mesh` names other devices than `requested`, which no single
/// mesh runs.
pub(crate) fn merge_requested_placement<'r>(
    requested: &mut Option<(Cow<'r, DeviceMesh>, &'static str)>,
    mesh: Cow<'r, DeviceMesh>,
    operation: &'static str,
) -> Result<(), Error> {
    match requested {
        None => *requested = Some((mesh, operation)),
        Some((first, _)) if first.devices() == mesh.devices() => {
            let canonical = LogicalMesh::single_device();
            if first.logical_mesh() == &canonical && mesh.logical_mesh() != &canonical {
                *requested = Some((mesh, operation));
            }
        }
        Some((first, _)) => {
            return Err(Error::Lowering(LoweringError::UnsupportedOperation {
                operation: operation.to_string(),
                reason: format!(
                    "it requests a placement onto the devices {}, while another placement request of the same \
                     program names the devices {}, and the two do not form one mesh",
                    render_devices(mesh.devices()),
                    render_devices(first.devices()),
                ),
            }));
        }
    }
    Ok(())
}

/// Returns the sharding with which the entry function of a program annotates each of its outputs, whose types are
/// `output_types`, and which describes the buffers that the outputs are returned in. These are the `output_shardings`
/// overrides when they are provided, and otherwise the placement that the program gives each output (i.e., the
/// corresponding element of `sharding_sources`, as [`ProgramLoweringExtension::output_sharding_sources`] returns them,
/// where an entry argument is placed as the corresponding element of `argument_shardings`), or the sharding of the
/// output type when the program gives it none (e.g., when that argument has no sharding). An output without either
/// has no sharding. Compilation for a [`Session`](crate::Session) and the StableHLO export of a
/// [`TracedProgram`](crate::TracedProgram) both choose the shardings of outputs with this function,
/// so that an exported module places its outputs as the executable that a session compiles for it does.
///
/// An override, a `place` target, an argument placement, or an output type may leave dimensions unconstrained, but the
/// runtime placement of an array reads such a dimension as replicated, and the output type records it exactly, so it
/// cannot record the placement that the compiler would choose for it. These dimensions are therefore replicated here
/// (refer to [`ShardingLoweringExtension::with_unconstrained_dimensions_replicated`]), whatever operation produces the
/// output (e.g., arithmetic whose output type inherits an unconstrained dimension from the target of a `place` that
/// precedes it), so that the executable returns the buffers that the runtime placement of the output describes, while
/// the dimensions that they constrain keep their placement. Provided overrides are not checked against `output_types`,
/// so that a count mismatch is still reported when the module is lowered.
///
/// A placement that the program gives an output never keeps reduction state that the output type does not record (i.e.,
/// a pending reduction or a completed-reduction marker over auto mesh axes). No value of a program carries such state:
/// an argument whose input sharding override states a pending reduction is reduced where it enters the program (refer
/// to the lowering of entry arguments, which constrains it to its placement with only the unreduced axes that its type
/// records), and a marker states nothing about the buffers, so an output that returns that argument, directly, through
/// operations that keep its placement, or through a call whose boundary states the state, returns the completed value.
/// Annotating the output with the placement of the argument as it arrives would instead make the executable introduce
/// the pending reduction again, and return an array whose runtime sharding states a reduction that its type does not
/// record and that every consumer of the array rejects, or whose marker makes every consumer move it again.
pub(crate) fn entry_output_shardings(
    output_types: &[ArrayType],
    output_shardings: Option<&[Sharding]>,
    sharding_sources: &[Option<OutputShardingSource>],
    argument_shardings: &[Option<Sharding>],
) -> Result<Vec<Option<Sharding>>, Error> {
    match output_shardings {
        Some(shardings) => shardings
            .iter()
            .map(|sharding| sharding.with_unconstrained_dimensions_replicated().map(Some))
            .collect(),
        None => output_types
            .iter()
            .zip(sharding_sources)
            .map(|(output_type, source)| {
                let stated = match source {
                    Some(OutputShardingSource::Determined(sharding)) => Some(sharding),
                    Some(OutputShardingSource::Input(index)) => argument_shardings.get(*index).and_then(Option::as_ref),
                    None => None,
                };
                let (unreduced_axes, reduced_axes) = (output_type.unreduced_axes(), output_type.reduced_axes());
                match stated {
                    Some(sharding)
                        if sharding.unreduced_axes().iter().any(|axis| !unreduced_axes.contains(axis))
                            || sharding.reduced_axes().iter().any(|axis| !reduced_axes.contains(axis)) =>
                    {
                        let unreduced = sharding.clone().with_unreduced_axes(unreduced_axes.iter().cloned())?;
                        unreduced
                            .with_reduced_axes(reduced_axes.iter().cloned())?
                            .with_unconstrained_dimensions_replicated()
                            .map(Some)
                    }
                    stated => stated
                        .or(output_type.sharding())
                        .map(ShardingLoweringExtension::with_unconstrained_dimensions_replicated)
                        .transpose(),
                }
            })
            .collect(),
    }
}

/// Returns whether the entry function of a program leaves each of its outputs, whose types are `output_types`
/// and whose annotations are `output_annotations` (i.e., the shardings that [`entry_output_shardings`] returns for
/// `output_shardings` and `sharding_sources`), open over the auto mesh axes of its annotation, so that the compiler
/// chooses the placement of the output along those axes, like JAX lowers the outputs of a `jit` function without
/// `out_shardings`. Compilation for a [`Session`](crate::Session) and the StableHLO export of a
/// [`TracedProgram`](crate::TracedProgram) both decide this with this function, so that an exported module
/// leaves open the outputs whose placement the compiler chooses for the executable that a session compiles for it.
///
/// An output is open when it has an annotation over a mesh with an auto axis larger than one (i.e., one along which the
/// compiler has a choice), and when nothing else fixes its placement. An output sharding override, a placement that the
/// program gives the output (i.e., that a `place` produces it or that it returns an argument), reduction state or
/// manual variation in the annotation, which the compiler must not change, and unconstrained dimensions in the output
/// type, which the runtime placement of the output must record as its type does and read as replicated, all fix it.
/// Outputs without an annotation are never open, because they have no annotation that could be.
///
/// Export lowers a program without a device mesh, so it can only decide what the program and its signature state.
/// It cannot know whether compilation runs the program partitioned, because a program with bounded dynamic shapes
/// runs as fully replicated replicas whose outputs are never open, nor which placements the compiler then chooses
/// for the open outputs, which only the compiled executable reports. It also cannot know the runtime placements of
/// arguments without signature shardings, whose types do not record auto mesh axes, so that their annotations and
/// those of the outputs that return them are left to Shardy.
pub(crate) fn entry_outputs_with_compiler_chosen_placement(
    output_types: &[ArrayType],
    output_shardings: Option<&[Sharding]>,
    sharding_sources: &[Option<OutputShardingSource>],
    output_annotations: &[Option<Sharding>],
) -> Vec<bool> {
    output_types
        .iter()
        .zip(sharding_sources)
        .zip(output_annotations)
        .map(|((output_type, source), annotation)| {
            output_shardings.is_none()
                && source.is_none()
                && annotation.as_ref().is_some_and(|sharding| {
                    sharding.mesh().has_untracked_auto_axes()
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayType, CALL_OPERATION_NAME, CompositeKey, CompositeOperation, ConstrainSharding, CustomRuleDefinition,
        CustomRuleRegistration, DataType, Device, DeviceMesh, DimensionValue, DomainTracingContext, EffectClass,
        FlatProgram, ImportingContext, LogicalMesh, Memory, MeshAxis, MeshAxisType, NamedComposite,
        PLACE_OPERATION_NAME, Place, PlaceOperation, Placeholder, Placement, PlacementRequest, Print, ProgramBuilder,
        Reshard, ScanOperation, ShardMapOptions, Sharding, ShardingDimension, Sin, StopGradient, Tag, TransferToMemory,
        TypeError, Typed, ValueProjection, shard_map,
    };
    use ryft_pjrt::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram};

    use crate::lowering::tests::{test_sharding, test_vector_type};
    use crate::sessions::tests::two_device_mesh;
    use crate::tests::{cpu_client, test_compilation_options, values_from_bytes, values_to_bytes};
    use crate::{
        Array, ArrayTracer, CompilationError, CompileTracer, CompiledFunction, Constant, Operation, Options, Session,
        StagedFunction, ThroughSession, TracedProgram, Tracer, Type, Value, compile, ensure_print_handler_registered,
        stage, trace, trace_with_context, with_captured_prints,
    };

    use super::*;

    /// Traces a program over a sharded `f64[4]` input that invokes a `shard_map` over the manual mesh of two devices,
    /// whose body prints its local shard with the [`EffectClass::DeviceOrderedIo`] effect and whose only output is
    /// dead, and that returns its input unchanged.
    fn traced_effectful_dead_shard_map() -> (TracedProgram<ArrayType, ArrayType>, Sharding) {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced = trace(
            {
                let sharding = sharding.clone();
                move |x: ArrayTracer| {
                    let _printed = shard_map(
                        |local_x: ArrayTracer| local_x.print_with_effect_class("body", EffectClass::DeviceOrderedIo),
                        x.clone(),
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                        ShardMapOptions::new(),
                    )
                    .unwrap();
                    x
                }
            },
            ArrayType::new_static(DataType::F64, [4]),
        )
        .unwrap();
        (traced, sharding)
    }

    #[test]
    fn test_program_lowering_options_exact_sharding_accessors() {
        let options = ProgramLoweringOptions::new();
        assert_eq!(options.exact_input_shardings(), None);
        assert_eq!(options.exact_output_shardings(), None);

        let options = options.without_boundary_shardings();
        assert_eq!(options.exact_input_shardings(), Some(&EntryShardingOverride::Unannotated));
        assert_eq!(options.exact_output_shardings(), Some(&EntryShardingOverride::Unannotated));

        let shardings = vec![Some(Sharding::replicated(LogicalMesh::single_device(), 0)), None];
        let options = options.with_exact_input_shardings(shardings.clone()).with_exact_output_shardings(vec![None]);
        assert_eq!(options.exact_input_shardings(), Some(&EntryShardingOverride::Values(shardings)));
        assert_eq!(options.exact_output_shardings(), Some(&EntryShardingOverride::Values(vec![None])));
    }

    #[test]
    fn test_program_to_mlir_module() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]);
        let traced: TracedProgram<ArrayType, ArrayType> =
            trace(move |x: ArrayTracer| x.sin().unwrap().reshard(&sharding).unwrap(), test_vector_type(8)).unwrap();
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> \
                        (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) {
                    %0 = stablehlo.sine %arg0 : tensor<8xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{\"x\"}]> : tensor<8xf32>
                    return %1 : tensor<8xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_program_to_mlir_module_merges_tracked_placement_into_sharding_constraint() {
        // The input is tracked as sharded over the explicit axis and the constraint places it over the auto axis, so
        // the emitted constraint must carry both placements rather than the constraint alone, which would contradict
        // the tracked type by marking the explicit axis replicated. The result is not the result of a `place`, so its
        // placement along the auto axis is left to the compiler, as compilation leaves it.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let tracked = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]);
        let constraint = test_sharding(&mesh, vec![ShardingDimension::sharded(["a"])], vec![]);
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            move |x: ArrayTracer| x.constrain_sharding(&constraint).unwrap(),
            test_vector_type(8).with_sharding(tracked).unwrap(),
        )
        .unwrap();
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2, \"a\"=2]>
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) -> \
                        (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\", ?}]>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{\"x\", \"a\"}]> : tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_program_to_mlir_module_annotates_placed_results() {
        // A result that a `place` produces, also through an operation that returns its input unchanged, is annotated
        // with the target of that `place`, including its split along the auto axis that the result type drops, as
        // compilation places it.
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(0, 0), Device::new(1, 0)],
        )
        .unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let target = Placement::new(mesh, sharding).unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> =
            trace(move |x: ArrayTracer| x.place(&target, None).unwrap().stop_gradient().unwrap(), test_vector_type(8))
                .unwrap();
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<8xf32>) -> \
                        (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{\"x\"}]> : tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_program_to_mlir_module_passes_imported_arrays_as_arguments() {
        // An array that the traced function imports becomes a session constant, which enters the module as a hidden
        // argument that precedes the inputs and is annotated as the array is placed, as compilation passes it.
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let array = Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 4f32.to_ne_bytes())
            .unwrap();
        let device = array.mesh().devices()[0];
        let traced = trace_with_context::<_, (), ArrayTracer>(
            |context, ()| {
                let imported =
                    ImportingContext::<Session, ThroughSession>::import(context, Value::Array(array.clone()))?;
                ValueProjection::<ArrayType>::into_projected(imported)?.place(&device, None)
            },
            (),
        )
        .unwrap();
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"device\"=1]>
                  func.func @main(%arg0: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}) -> \
                        (tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}) {
                    return %arg0 : tensor<f32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_program_to_mlir_module_with_options() {
        // Explicit shardings are attached to the arguments and the results of the function. The global types have no
        // sharding here, so without explicit shardings the signature is not annotated, as `to_mlir_module` renders it.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]);
        let traced: TracedProgram<ArrayType, ArrayType> =
            trace(|x: ArrayTracer| x.sin().unwrap(), test_vector_type(8)).unwrap();
        let shardings = vec![sharding.clone()];
        assert_eq!(
            traced
                .to_mlir_module_with_options(
                    "main",
                    &ProgramLoweringOptions::new()
                        .with_input_shardings(shardings.clone())
                        .with_output_shardings(shardings.clone())
                )
                .unwrap(),
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=4]>
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) -> \
                        (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) {
                    %0 = stablehlo.sine %arg0 : tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "},
        );
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    %0 = stablehlo.sine %arg0 : tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "},
        );
        assert_eq!(traced.to_mlir_module("main").unwrap(), traced.to_mlir_module("main").unwrap());
    }

    #[test]
    fn test_program_to_mlir_module_with_options_rejects_wrong_open_output_flag_count() {
        let traced = trace(|input: ArrayTracer| (input.clone(), input), test_vector_type(4)).unwrap();
        for (flags, reason) in [
            (vec![true], "expected 2 open output flags or an empty list, but got 1"),
            (vec![true, false, true], "expected 2 open output flags or an empty list, but got 3"),
        ] {
            assert_eq!(
                traced.to_mlir_module_with_options("main", &ProgramLoweringOptions::new().with_open_outputs(flags)),
                Err(Error::Compilation(CompilationError::InvalidOptions { reason: reason.to_string() })),
            );
        }
    }

    #[test]
    fn test_program_to_mlir_module_with_options_rejects_boundaries_that_are_not_arrays() {
        // The export reads array types at the boundary of the program, so a program whose input is a dimension is
        // rejected before anything is lowered.
        let dimension_type = Type::Dimension(DimensionValue::constant(3).unwrap().r#type().into_owned());
        let (_, program) =
            DomainTracingContext::<Session>::trace(|inputs: Vec<Tracer>| Ok(inputs), vec![dimension_type]).unwrap();
        assert_eq!(
            program.to_mlir_module("main"),
            Err(Error::from(ProgramError::from(TypeError::invalid("expected array type but got dimension type")))),
        );
    }

    #[test]
    fn test_program_to_mlir_module_with_options_rejects_imports_over_other_topologies() {
        // A plain trace keeps the type of an array that it imports, so a signature over a mesh of another topology
        // cannot hold an array over two devices, and the error names the cause and the way forward.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 1);
        let array_type = test_vector_type(4);
        let array = Array::from_host_buffer(
            &session,
            array_type.clone().with_sharding(sharding.clone()).unwrap(),
            None,
            Some(sharding),
            values_to_bytes(&[0f32, 1., 2., 3.]),
        )
        .unwrap();
        let traced = trace_with_context::<_, ArrayType, ArrayTracer>(
            |context, _| {
                let imported =
                    ImportingContext::<Session, ThroughSession>::import(context, Value::Array(array.clone()))?;
                Ok(ValueProjection::<ArrayType>::into_projected(imported)?)
            },
            array_type,
        )
        .unwrap();
        let signature = Sharding::replicated(LogicalMesh::single_device(), 1);
        let signatures = vec![signature.clone()];
        assert_eq!(
            traced.to_mlir_module_with_options(
                "main",
                &ProgramLoweringOptions::new()
                    .with_input_shardings(signatures.clone())
                    .with_output_shardings(signatures.clone())
            ),
            Err(Error::from(ProgramError::InvalidArgument {
                message: "the traced program imports an array of type \
                          `f32[4][sharding={mesh<['x'=2:auto]>, [{}]}]`, whose mesh does not have the topology of the \
                          mesh `['device'=1:auto]` of the signature of the module, because a plain `trace` keeps the \
                          type of an array that it imports; stage or compile the function for a device mesh over \
                          `['device'=1:auto]` through a `Session` instead (e.g., with `stage` or `compile`), which \
                          receives the array on that mesh"
                    .to_string(),
            })),
        );
    }

    #[test]
    fn test_program_to_mlir_module_with_options_closes_unconstrained_result_placements() {
        // The runtime placement of a result reads an unconstrained dimension of the target of the `place` that produces
        // it, or of a result sharding override, as replicated, so the result is annotated with that dimension closed,
        // as compilation annotates it, while the constraint inside the function keeps it open.
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![
                MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
                MeshAxis::new("r", 2, MeshAxisType::Auto).unwrap(),
            ])
            .unwrap(),
            (0..4).map(|id| Device::new(id, 0)).collect(),
        )
        .unwrap();
        let input_sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["r"])],
        )
        .unwrap();
        let open_sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::unconstrained()],
        )
        .unwrap();
        let target = Placement::new(mesh, open_sharding.clone()).unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> =
            trace(move |x: ArrayTracer| x.place(&target, None).unwrap(), ArrayType::new_static(DataType::I32, [4, 4]))
                .unwrap();
        let expected = indoc! {"
            module {
              sdy.mesh @mesh = <[\"x\"=2, \"r\"=2]>
              func.func @main(%arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}, {\"r\"}]>}) -> \
                    (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}, {}]>}) {
                %0 = sdy.sharding_constraint %arg0 <@mesh, [{\"x\"}, {?}]> : tensor<4x4xi32>
                return %0 : tensor<4x4xi32>
              }
            }
        "};
        assert_eq!(
            traced
                .to_mlir_module_with_options(
                    "main",
                    &ProgramLoweringOptions::new()
                        .with_input_shardings((std::slice::from_ref(&input_sharding)).to_vec())
                )
                .unwrap(),
            expected,
        );
        assert_eq!(
            traced
                .to_mlir_module_with_options(
                    "main",
                    &ProgramLoweringOptions::new()
                        .with_input_shardings((std::slice::from_ref(&input_sharding)).to_vec())
                        .with_output_shardings((std::slice::from_ref(&open_sharding)).to_vec())
                )
                .unwrap(),
            expected,
        );
    }

    #[test]
    fn test_program_to_mlir_module_with_options_closes_unconstrained_result_types() {
        // A result that is computed from the result of a `place` inherits the unconstrained dimension of its target in
        // its type, which its runtime placement reads as replicated as well, so it is annotated with that dimension
        // closed, as compilation annotates it.
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![
                MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
                MeshAxis::new("r", 2, MeshAxisType::Auto).unwrap(),
            ])
            .unwrap(),
            (0..4).map(|id| Device::new(id, 0)).collect(),
        )
        .unwrap();
        let input_sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let open_sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::unconstrained()],
        )
        .unwrap();
        let target = Placement::new(mesh, open_sharding).unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            move |x: ArrayTracer| {
                let placed = x.place(&target, None).unwrap();
                placed.clone() + placed
            },
            ArrayType::new_static(DataType::I32, [4, 4]),
        )
        .unwrap();
        assert_eq!(
            traced
                .to_mlir_module_with_options(
                    "main",
                    &ProgramLoweringOptions::new()
                        .with_input_shardings((std::slice::from_ref(&input_sharding)).to_vec())
                )
                .unwrap(),
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2, \"r\"=2]>
                  func.func @main(%arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}, {}]>}) -> \
                        (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{\"x\"}, {?}]> : tensor<4x4xi32>
                    %1 = stablehlo.add %0, %0 : tensor<4x4xi32>
                    return %1 : tensor<4x4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_program_to_mlir_module_with_options_leaves_results_open_over_auto_axes() {
        // A result that is computed from the result of a `place`, rather than produced by it, has a type that drops the
        // split of the target along the auto axis `x`, so the compiler chooses its placement along that axis, and the
        // result is annotated open over it, as compilation annotates it.
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(0, 0), Device::new(1, 0)],
        )
        .unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let target = Placement::new(mesh, sharded).unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            move |x: ArrayTracer| {
                let placed = x.place(&target, None).unwrap();
                placed.clone() + placed
            },
            test_vector_type(4),
        )
        .unwrap();
        let options = ProgramLoweringOptions::new().with_input_shardings(vec![replicated]);
        let open = indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) -> \
                        (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}]>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{\"x\"}]> : tensor<4xf32>
                    %1 = stablehlo.add %0, %0 : tensor<4xf32>
                    return %1 : tensor<4xf32>
                  }
                }
            "};
        assert_eq!(traced.to_mlir_module_with_options("main", &options).unwrap(), open);
        assert_eq!(
            traced.to_mlir_module_with_options("main", &options.clone().with_open_outputs(vec![true])).unwrap(),
            open,
        );
        assert_eq!(
            traced.to_mlir_module_with_options("main", &options.with_open_outputs(Vec::new())).unwrap(),
            open.replace("[{?}]", "[{}]"),
        );
    }

    #[test]
    fn test_program_to_mlir_module_with_options_keeps_effectful_dead_shard_maps() {
        // Pruning keeps a `shard_map` whose ordinary results are all dead when its body has observable effects, with no
        // ordinary results, so the lowered manual computation only threads the effect token of its body.
        let (traced, sharding) = traced_effectful_dead_shard_map();
        assert_eq!(
            traced
                .to_mlir_module_with_options(
                    "main",
                    &ProgramLoweringOptions::new()
                        .with_input_shardings((std::slice::from_ref(&sharding)).to_vec())
                        .with_output_shardings((std::slice::from_ref(&sharding)).to_vec())
                )
                .unwrap(),
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<4xf64> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) -> \
                        (tensor<4xf64> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1 = sdy.manual_computation(%arg0, %0) in_shardings=[<@mesh, [{\"x\"}]>, <@mesh, []>] \
                        out_shardings=[<@mesh, []>] manual_axes={\"x\"} \
                        (%arg1: tensor<2xf64>, %arg2: !stablehlo.token) {
                      %2 = stablehlo.custom_call @ryft.print(%arg1, %arg2) {api_version = 4 : i32, \
                          backend_config = {label = \"body\"}, has_side_effect = true, \
                          operand_layouts = [dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], \
                          result_layouts = [dense<> : tensor<0xindex>]} : (tensor<2xf64>, !stablehlo.token) \
                          -> !stablehlo.token
                      sdy.return %2 : !stablehlo.token
                    } : (tensor<4xf64>, !stablehlo.token) -> !stablehlo.token
                    return %arg0 : tensor<4xf64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_program_to_mlir_module_with_options_executes_effectful_dead_shard_maps_on_cpu() {
        let (traced, sharding) = traced_effectful_dead_shard_map();
        let module = traced
            .to_mlir_module_with_options(
                "main",
                &ProgramLoweringOptions::new()
                    .with_input_shardings((std::slice::from_ref(&sharding)).to_vec())
                    .with_output_shardings((std::slice::from_ref(&sharding)).to_vec()),
            )
            .unwrap();
        let client = cpu_client(2);
        ensure_print_handler_registered(client).unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_compilation_options(client, Some(2)))
            .unwrap();
        let shards = [vec![1.5f64, 2.5], vec![3.5f64, 4.5]];
        let inputs = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .zip(&shards)
            .map(|(device, shard)| {
                let bytes = values_to_bytes(shard.as_slice());
                vec![ExecutionInput {
                    buffer: Arc::new(
                        client.buffer(bytes.as_slice(), BufferType::F64, [2u64], None, device.clone(), None).unwrap(),
                    ),
                    donatable: false,
                }]
            })
            .collect::<Vec<_>>();
        let (outputs, mut lines) = with_captured_prints(|| {
            executable
                .execute(
                    inputs.iter().map(|inputs| ExecutionDeviceInputs { inputs, ..Default::default() }).collect(),
                    Vec::new(),
                    0,
                    None,
                    Some(file!()),
                    None,
                    None,
                    &[],
                )
                .unwrap()
                .block_until_ready()
                .unwrap()
        });

        // Every device returns its own shard and prints it exactly once, although nothing uses the `shard_map` output.
        assert_eq!(outputs.len(), 2);
        for (output, shard) in outputs.iter().zip(&shards) {
            assert_eq!(output.outputs.len(), 1);
            let bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(&values_from_bytes::<f64>(&bytes), shard);
        }
        lines.sort();
        assert_eq!(lines, vec!["body: [1.5, 2.5]".to_string(), "body: [3.5, 4.5]".to_string()]);
    }

    #[test]
    fn test_program_pruned_for_lowering() {
        // The callee `f(x) = (x, place(x, CPU 1))` names devices only through the `place` that computes its second
        // output, and its call records their mesh. A program that returns both outputs keeps that mesh. A program that
        // drops the second output loses the `place` to pruning, so its call is lowered as one that names no devices.
        let vector_type = test_vector_type(4);
        let device = Device::new(1, 0);
        let mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![device]).unwrap();
        let callee = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(vector_type.clone().into());
            let operation = PlaceOperation::new(PlacementRequest::from(device), None);
            let placed = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
            Arc::new(builder.build(vec![input, placed], vec![Placeholder], vec![Placeholder; 2]).unwrap())
        };
        let program = |output_indices: &[usize]| {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(vector_type.clone().into());
            let regions = vec![builder.intern_callee(&callee, None).unwrap()];
            let call = Operation::Call(CallOperation::new(0).with_boundary(Some(&mesh), Vec::new(), Vec::new()));
            let outputs = builder.add_instruction(call, regions, vec![input], None).unwrap().to_vec();
            let outputs = output_indices.iter().map(|&index| outputs[index]).collect::<Vec<_>>();
            let output_count = outputs.len();
            builder
                .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder; output_count])
                .unwrap()
        };
        assert_eq!(
            program(&[0, 1]).pruned_for_lowering(None).unwrap().to_string(),
            indoc! {"
                lambda %0:f32[4] .
                let %1:f32[4], %2:f32[4] = call [devices=[1]] %0 [
                    callee={
                        lambda %0:f32[4] .
                        let %1:f32[4] = place [target=device(1)] %0
                        in (%0, %1)
                    },
                ]
                in (%1, %2)"},
        );
        assert_eq!(
            program(&[0]).pruned_for_lowering(None).unwrap().to_string(),
            indoc! {"
                lambda %0:f32[4] .
                let %1:f32[4] = call %0 [
                    callee={
                        lambda %0:f32[4] .
                        in (%0)
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_program_has_placement_requests() {
        let sharding = |axis_type: MeshAxisType| {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap()
        };
        let (auto_sharding, explicit_sharding) = (sharding(MeshAxisType::Auto), sharding(MeshAxisType::Explicit));
        let device = Device::new(0, 0);
        let has_requests = |traced: TracedProgram<ArrayType, ArrayType>| {
            (traced.has_placement_requests(None, true), traced.has_placement_requests(None, false))
        };

        // A `place` with a device target and a `constrain_sharding` always count.
        let placed = trace(|x: ArrayTracer| x.place(&device, None).unwrap(), test_vector_type(4)).unwrap();
        assert_eq!(has_requests(placed), (true, true));
        let constrained =
            trace(|x: ArrayTracer| x.constrain_sharding(&auto_sharding).unwrap(), test_vector_type(4)).unwrap();
        assert_eq!(has_requests(constrained), (true, true));

        // A `place` with an in-place target names no devices, so it counts only among the memory requests.
        let moved = trace(|x: ArrayTracer| x.place(&Memory::Host { pinned: true }, None).unwrap(), test_vector_type(4))
            .unwrap();
        assert_eq!(has_requests(moved), (true, false));

        // A `place` without a target, a `reshard`, and operations that place nothing request no placement.
        let kept = trace(|x: ArrayTracer| x.place(&None::<Placement>, None).unwrap(), test_vector_type(4)).unwrap();
        assert_eq!(has_requests(kept), (false, false));
        let resharded = trace(|x: ArrayTracer| x.reshard(&explicit_sharding).unwrap(), test_vector_type(4)).unwrap();
        assert_eq!(has_requests(resharded), (false, false));
        let computed = trace(|x: ArrayTracer| x.sin().unwrap(), test_vector_type(4)).unwrap();
        assert_eq!(has_requests(computed), (false, false));
    }

    #[test]
    fn test_program_output_sharding_sources() {
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(0, 0), Device::new(1, 0)],
        )
        .unwrap();
        let split = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let vector_type = test_vector_type(4);

        // An output that a `place` produces, also through an operation that returns its input unchanged, has the
        // placement of the target, an output that returns an argument is placed where that argument enters, and the
        // program determines no sharding for an output that it computes.
        let target = Placement::new(mesh.clone(), split.clone()).unwrap();
        let traced: TracedProgram<ArrayType, (ArrayType, ArrayType, ArrayType)> = trace(
            move |x: ArrayTracer| {
                let placed = x.place(&target, None).unwrap().stop_gradient().unwrap();
                let computed = x.sin().unwrap();
                (placed, x, computed)
            },
            vector_type.clone(),
        )
        .unwrap();
        assert_eq!(
            traced.output_sharding_sources(3, |constant| constant.sharding().clone()),
            vec![Some(OutputShardingSource::Determined(split.clone())), Some(OutputShardingSource::Input(0)), None],
        );
        assert_eq!(
            traced.output_sharding_sources(1, |constant| constant.sharding().clone()),
            vec![Some(OutputShardingSource::Determined(split.clone()))],
        );

        // The boundary of the callee `f(x) = (x, x)` takes precedence over its body. An output of a call with a
        // sharding override has that override, and an output that returns a callee input is placed where the executable
        // of the callee places that input, which is replicated over the mesh that the call names. The same output of a
        // call that names no devices is followed back to the input of the call.
        let callee = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(vector_type.clone().into());
            Arc::new(builder.build(vec![input, input], vec![Placeholder], vec![Placeholder; 2]).unwrap())
        };
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(vector_type.into());
        let regions = vec![builder.intern_callee(&callee, None).unwrap()];
        let call = CallOperation::new(0).with_boundary(Some(&mesh), Vec::new(), vec![Some(split.clone())]);
        let placed =
            builder.add_instruction(Operation::Call(call), regions.clone(), vec![input], None).unwrap().to_vec();
        let call = Operation::Call(CallOperation::new(0));
        let followed = builder.add_instruction(call, regions, vec![input], None).unwrap()[1];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![placed[0], placed[1], followed],
                vec![Placeholder],
                vec![Placeholder; 3],
            )
            .unwrap();
        assert_eq!(
            program.output_sharding_sources(3, |constant| constant.sharding().clone()),
            vec![
                Some(OutputShardingSource::Determined(split)),
                Some(OutputShardingSource::Determined(replicated)),
                Some(OutputShardingSource::Input(0)),
            ],
        );
    }

    #[test]
    fn test_program_output_sharding_sources_follow_operations_that_keep_the_placement_of_their_input() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let split = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let vector_type = test_vector_type(4);
        let input = Array::from_host_buffer(
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
        // of the program is split as well (refer to `ProgramLoweringExtension::output_sharding_sources`).
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
    fn test_sharding_with_unconstrained_dimensions_replicated() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded = |second: ShardingDimension| {
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), second]).unwrap()
        };
        assert_eq!(
            sharded(ShardingDimension::unconstrained()).with_unconstrained_dimensions_replicated(),
            Ok(sharded(ShardingDimension::replicated())),
        );

        // A sharding without unconstrained dimensions is returned as it is.
        let constrained = sharded(ShardingDimension::replicated());
        assert_eq!(constrained.with_unconstrained_dimensions_replicated(), Ok(constrained.clone()));
    }

    #[test]
    fn test_sharding_dimensions_with_unconstrained_replicated() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let dimensions = vec![
            ShardingDimension::sharded(["x"]),
            ShardingDimension::unconstrained(),
            ShardingDimension::replicated(),
        ];
        let sharding = Sharding::new(mesh, dimensions).unwrap();
        assert_eq!(
            sharding.dimensions_with_unconstrained_replicated().to_vec(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated(), ShardingDimension::replicated()],
        );
    }

    #[test]
    fn test_region_placement_requests_for_lowering() {
        // A `place` in the body of a named composite call is a request where lowering emits the body, and not for a
        // target whose backend implementation replaces the whole call (i.e., `ryft.tests.xla_named_square` version 1
        // on CUDA, refer to `omits_regions`), so that the eager placement of the enclosing operation does not read it.
        // The requests are retained for each platform separately.
        let vector_type = test_vector_type(4);
        let other = LogicalMesh::new(vec![MeshAxis::new("other", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other_devices = DeviceMesh::new(other.clone(), vec![Device::new(2, 0), Device::new(3, 0)]).unwrap();
        let placement = Placement::new(other_devices.clone(), Sharding::replicated(other, 1)).unwrap();
        let body = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(vector_type.clone().into());
            let operation = PlaceOperation::new(PlacementRequest::from(placement), None);
            let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
            Arc::new(builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap())
        };
        let key = CompositeKey::new("ryft.tests.xla_named_square".to_string(), 1);
        let named = NamedComposite::new(key, |types: &[Type]| Ok(types.to_vec()), |inputs| Ok(vec![inputs[0].clone()]));
        let rules = CustomRuleRegistration::new(CustomRuleDefinition::new("placing").with_named_composite(named));
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(vector_type.into());
        let operation = Operation::Composite(CompositeOperation::new(rules.reference()));
        let regions = vec![builder.intern_callee(&body, None).unwrap()];
        let output = builder.add_instruction(operation, regions, vec![input], None).unwrap()[0];
        let program = builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder]);
        let program = program.unwrap();
        let requests = |target_platform| {
            RegionPlacementRequests::for_lowering(program.entry_region_ref(), target_platform)
                .requests()
                .to_vec()
        };
        let placed = vec![(RequestedPlacement::Devices(other_devices), PLACE_OPERATION_NAME)];
        assert_eq!(requests(None), placed);
        assert_eq!(requests(Some("cuda")), vec![]);
        assert_eq!(requests(Some("cpu")), placed);
        assert_eq!(requests(Some("cuda")), vec![]);
    }

    #[test]
    fn test_region_placement_requests_for_lowering_of_nested_calls() {
        // `f(x) = place(x, CPU 0)` wrapped in 40 functions that each call the previous one. The call of each wrapper
        // names devices because the request in the innermost callee does, which the analysis of a wrapper reads from
        // the facts of its callee that it derives in the same pass over the pruned program, rather than by analyzing
        // the callee of every nested call again, which took time exponential in the depth of the nesting (i.e., about
        // 4.5 seconds for 20 wrappers in a debug build), and would not finish for these 40. Staging nests as deeply as
        // the functions do, so the test runs on a thread with a larger stack.
        let depth = 40;
        let requests = std::thread::Builder::new()
            .stack_size(256 << 20)
            .spawn(move || {
                let device = Device::new(0, 0);
                let mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![device]).unwrap();
                let scalar_type = ArrayType::scalar(DataType::F32);
                let options = Options::new(mesh.clone());
                let placing = |x: CompileTracer| x.place(&device, None).unwrap();
                let mut function: StagedFunction<ArrayType, ArrayType> =
                    stage(placing, scalar_type.clone(), options.clone()).unwrap();
                for _ in 0..depth {
                    function =
                        stage(|x: CompileTracer| function.call(x).unwrap(), scalar_type.clone(), options.clone())
                            .unwrap();
                }
                let program = function.source_program().program();
                let requested =
                    RegionPlacementRequests::for_lowering(program.entry_region_ref(), None).requested_mesh();
                assert_eq!(requested.unwrap().map(|(requested, _)| requested), Some(mesh.clone()));
                assert_eq!(program.pruned_for_lowering(None).unwrap().to_string(), program.to_string());
                (mesh, RegionPlacementRequests::for_lowering(program.entry_region_ref(), None).requests().to_vec())
            })
            .unwrap()
            .join()
            .unwrap();
        let (mesh, requests) = requests;
        assert_eq!(requests.len(), depth + 1);
        assert_eq!(requests[0], (RequestedPlacement::Devices(mesh.clone()), CALL_OPERATION_NAME));
        assert_eq!(requests[depth], (RequestedPlacement::Devices(mesh), PLACE_OPERATION_NAME));
        assert!(requests[1..depth].iter().all(|request| *request == requests[0]));
    }

    #[test]
    fn test_region_placement_requests_for_lowering_skips_the_bodies_of_scans_without_iterations() {
        // A `place` in the body of a scan is a request when the length of the scan admits an iteration, and not on any
        // target when its length is zero, because lowering then emits nothing of its body (refer to `omits_regions`).
        let vector_type = test_vector_type(4);
        let elsewhere = Device::new(1, 0);
        let body = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(vector_type.clone().into());
            let operation = PlaceOperation::new(PlacementRequest::from(elsewhere), None);
            let output = builder.add_instruction(operation, Vec::new(), vec![carry], None).unwrap()[0];
            Arc::new(builder.build(vec![output], vec![Placeholder; 2], vec![Placeholder]).unwrap())
        };
        let program = |length: usize| {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(vector_type.clone().into());
            let regions = vec![builder.intern_callee(&body, None).unwrap()];
            let scan = ScanOperation::<Type>::new(1, length);
            let output = builder.add_instruction(scan, regions, vec![input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let requests = |length, target_platform| {
            RegionPlacementRequests::for_lowering(program(length).entry_region_ref(), target_platform)
                .requests()
                .to_vec()
        };
        let device_mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![elsewhere]).unwrap();
        let placed = vec![(RequestedPlacement::Devices(device_mesh), PLACE_OPERATION_NAME)];
        assert_eq!(requests(0, None), vec![]);
        assert_eq!(requests(0, Some("cpu")), vec![]);
        assert_eq!(requests(1, None), placed);
        assert_eq!(requests(1, Some("cpu")), placed);
    }

    #[test]
    fn test_region_placement_requests_requested_mesh() {
        let vector_type = test_vector_type(4);
        let first = Device::new(0, 0);
        let second = Device::new(1, 0);
        let device_mesh = |device: Device| DeviceMesh::new(LogicalMesh::single_device(), vec![device]).unwrap();
        let program = |targets: Vec<PlacementRequest>| {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let mut value = builder.add_input(vector_type.clone().into());
            for target in targets {
                value = builder
                    .add_instruction(PlaceOperation::new(target, None), Vec::new(), vec![value], None)
                    .unwrap()[0];
            }
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![value], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let requested = |program: &FlatProgram<Session>| {
            RegionPlacementRequests::for_lowering(program.entry_region_ref(), None)
                .requested_mesh()
                .map(|requested| requested.map(|(mesh, _)| mesh))
        };
        let memory = PlacementRequest::InPlace { memory: Some(Memory::Device), layout: None };

        // A request that only moves a value to another memory names no devices, and requests that name the same
        // devices form one mesh, which a device target names as the canonical mesh of that device.
        assert_eq!(requested(&program(vec![memory.clone()])), Ok(None));
        assert_eq!(
            requested(&program(vec![memory, PlacementRequest::from(first), PlacementRequest::from(first)])),
            Ok(Some(device_mesh(first))),
        );

        // Requests that name different devices do not form one mesh.
        assert_eq!(
            requested(&program(vec![PlacementRequest::from(first), PlacementRequest::from(second)])),
            Err(Error::Lowering(LoweringError::UnsupportedOperation {
                operation: PLACE_OPERATION_NAME.to_string(),
                reason: "it requests a placement onto the devices [1], while another placement request of the same \
                         program names the devices [0], and the two do not form one mesh"
                    .to_string(),
            })),
        );

        // A device target and a named mesh of the same device select the named mesh in either order, and a device
        // target of an input that is tracked over a mesh of one device keeps that logical mesh, as the type rule of
        // `place` does.
        let named = LogicalMesh::new(vec![MeshAxis::new("named", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let named_mesh = DeviceMesh::new(named.clone(), vec![second]).unwrap();
        let named_target =
            PlacementRequest::from(Placement::new(named_mesh.clone(), Sharding::replicated(named.clone(), 1)).unwrap());
        for targets in [
            vec![named_target.clone(), PlacementRequest::from(second)],
            vec![PlacementRequest::from(second), named_target.clone()],
        ] {
            assert_eq!(requested(&program(targets)), Ok(Some(named_mesh.clone())));
        }
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let named_type = vector_type.clone().with_sharding(Sharding::replicated(named, 1)).unwrap();
        let input = builder.add_input(named_type.into());
        let output = builder
            .add_instruction(PlaceOperation::new(PlacementRequest::from(second), None), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(requested(&program), Ok(Some(named_mesh)));
    }

    #[test]
    fn test_merge_requested_placement() {
        let canonical = DeviceMesh::new(LogicalMesh::single_device(), vec![Device::new(0, 0)]).unwrap();
        let named = |name: &str| {
            let mesh = LogicalMesh::new(vec![MeshAxis::new(name, 1, MeshAxisType::Explicit).unwrap()]).unwrap();
            DeviceMesh::new(mesh, vec![Device::new(0, 0)]).unwrap()
        };
        let (first_named, second_named) = (named("first"), named("second"));
        let elsewhere = DeviceMesh::new(LogicalMesh::single_device(), vec![Device::new(1, 0)]).unwrap();

        // The first request that names devices is recorded with the name of its operation.
        let mut requested = None;
        assert_eq!(merge_requested_placement(&mut requested, Cow::Borrowed(&canonical), PLACE_OPERATION_NAME), Ok(()));
        assert_eq!(requested, Some((Cow::Borrowed(&canonical), PLACE_OPERATION_NAME)));

        // A named mesh of the same device replaces the canonical mesh and is kept when the canonical mesh or another
        // named mesh of that device follows it.
        assert_eq!(merge_requested_placement(&mut requested, Cow::Borrowed(&first_named), CALL_OPERATION_NAME), Ok(()));
        assert_eq!(requested, Some((Cow::Borrowed(&first_named), CALL_OPERATION_NAME)));
        assert_eq!(merge_requested_placement(&mut requested, Cow::Borrowed(&canonical), PLACE_OPERATION_NAME), Ok(()));
        assert_eq!(
            merge_requested_placement(&mut requested, Cow::Borrowed(&second_named), PLACE_OPERATION_NAME),
            Ok(()),
        );
        assert_eq!(requested, Some((Cow::Borrowed(&first_named), CALL_OPERATION_NAME)));

        // A request that names other devices does not form one mesh with the recorded request, which is kept.
        assert_eq!(
            merge_requested_placement(&mut requested, Cow::Borrowed(&elsewhere), PLACE_OPERATION_NAME),
            Err(Error::Lowering(LoweringError::UnsupportedOperation {
                operation: PLACE_OPERATION_NAME.to_string(),
                reason: "it requests a placement onto the devices [1], while another placement request of the same \
                         program names the devices [0], and the two do not form one mesh"
                    .to_string(),
            })),
        );
        assert_eq!(requested, Some((Cow::Borrowed(&first_named), CALL_OPERATION_NAME)));
    }

    #[test]
    fn test_entry_output_shardings() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded = |dimension: ShardingDimension| Sharding::new(mesh.clone(), vec![dimension]).unwrap();
        let split = sharded(ShardingDimension::sharded(["x"]));
        let replicated = sharded(ShardingDimension::replicated());
        let open = sharded(ShardingDimension::unconstrained());
        let plain_type = test_vector_type(4);
        let split_type = plain_type.clone().with_sharding(split.clone()).unwrap();

        // Output sharding overrides are used whatever the program and the output types state, with their unconstrained
        // dimensions replicated.
        assert_eq!(
            entry_output_shardings(
                std::slice::from_ref(&split_type),
                Some(std::slice::from_ref(&open)),
                &[Some(OutputShardingSource::Determined(split.clone()))],
                &[],
            ),
            Ok(vec![Some(replicated.clone())]),
        );

        // Without overrides, an output has the placement that the program determines for it or that of the argument
        // that it returns, with its unconstrained dimensions replicated, and otherwise the sharding of its type, if
        // any.
        assert_eq!(
            entry_output_shardings(
                &[plain_type.clone(), plain_type.clone(), plain_type.clone(), split_type, plain_type.clone()],
                None,
                &[
                    Some(OutputShardingSource::Determined(split.clone())),
                    Some(OutputShardingSource::Input(0)),
                    Some(OutputShardingSource::Input(1)),
                    None,
                    None,
                ],
                &[Some(open), None],
            ),
            Ok(vec![Some(split.clone()), Some(replicated.clone()), None, Some(split), None]),
        );

        // A placement that the program gives an output keeps only the reduction state that the output type records.
        let pending = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let pending_type = plain_type.clone().with_sharding(pending.clone()).unwrap();
        assert_eq!(
            entry_output_shardings(
                &[plain_type, pending_type],
                None,
                &[
                    Some(OutputShardingSource::Determined(pending.clone())),
                    Some(OutputShardingSource::Determined(pending.clone())),
                ],
                &[],
            ),
            Ok(vec![Some(replicated), Some(pending)]),
        );
    }

    #[test]
    fn test_entry_outputs_with_compiler_chosen_placement() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let annotation = Sharding::replicated(mesh.clone(), 1);
        let plain_type = test_vector_type(4);
        let open_type = plain_type
            .clone()
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::unconstrained()]).unwrap())
            .unwrap();
        let open = |output_type: &ArrayType,
                    output_shardings: Option<&[Sharding]>,
                    source: Option<OutputShardingSource>,
                    annotation: Option<Sharding>| {
            entry_outputs_with_compiler_chosen_placement(
                std::slice::from_ref(output_type),
                output_shardings,
                &[source],
                &[annotation],
            )
        };

        // An output is open when its annotation is over a mesh with an auto axis along which the compiler has a choice
        // and nothing fixes its placement.
        assert_eq!(open(&plain_type, None, None, Some(annotation.clone())), vec![true]);
        assert_eq!(open(&plain_type, None, None, Some(Sharding::replicated(explicit_mesh, 1))), vec![false]);
        assert_eq!(open(&plain_type, None, None, None), vec![false]);

        // An output sharding override and a placement that the program gives the output fix it.
        let overrides = Some(std::slice::from_ref(&annotation));
        assert_eq!(open(&plain_type, overrides, None, Some(annotation.clone())), vec![false]);
        let source = Some(OutputShardingSource::Input(0));
        assert_eq!(open(&plain_type, None, source, Some(annotation.clone())), vec![false]);

        // Reduction state and manual variation in the annotation fix it, and so does an unconstrained dimension in the
        // output type, which the runtime placement of the output reads as replicated.
        let unreduced = annotation.clone().with_unreduced_axes(["a"]).unwrap();
        assert_eq!(open(&plain_type, None, None, Some(unreduced)), vec![false]);
        let reduced = annotation.clone().with_reduced_axes(["a"]).unwrap();
        assert_eq!(open(&plain_type, None, None, Some(reduced)), vec![false]);
        let varying = annotation.clone().with_varying_manual_axes(["m"]).unwrap();
        assert_eq!(open(&plain_type, None, None, Some(varying)), vec![false]);
        assert_eq!(open(&open_type, None, None, Some(annotation)), vec![false]);
    }
}
