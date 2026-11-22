use std::any::TypeId;
use std::borrow::Cow;
use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use ryft_core::macros::check_count;
use ryft_core::{
    ArrayBatch, ArrayBatchingPolicy, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayMemberType, ArrayType, AtomId,
    AttachedRegions, BatchAxis, BatchableOperation, BatchedProgram, BatchingContext, BatchingDriver, BatchingError,
    CaptureConstant, CastOperation, CompositeKey, Concretizable, ConstrainShardingOperation, Context,
    CotangentAccumulator, CotangentDestinations, Device, DeviceMesh, DifferentiableOperation, DifferentiableType,
    DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationError, DifferentiationPolicy,
    InputRegionProvenance, InterpretableOperation, InterpretationDriver, LogicalMesh, MaybeZero,
    Operation as CoreOperation, OperationBoundaryPruning, OperationFormatter, OutputRegionProvenance,
    ParallelReduceOperation, Parameterized, PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationOutput,
    PartialEvaluationValue, PartialValue, PartiallyEvaluatableOperation, PartitionedProgram, PlaceOperation, Placement,
    Program as CoreProgram, ProgramBatchingOutputAxesPolicy, ProgramError, ReductionKind, ReferenceDischargeContext,
    ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeValue, ReferenceDischargeableOperation,
    Region as CoreRegion, RegionDataFlow, RegionId, RegionInterface, RegionLiveness, RegionRef, RegionRole, RegionSlot,
    RematerializationOptimizationBarrier, ReshardOperation, ResidualDemandSplit, ResidualInputSource,
    ResidualPolicyReference, ShardMapOperation, Sharding, ShardingDimension, StagedCallOperation, TangentActivity,
    Tracer, TracingContext, Transform, TransformArtifact, TransposableOperation, TransposedProgramCotangent,
    TranspositionContext, TranspositionDriver, Type as CoreType, TypeError, TypeIdentityRenaming, TypeRefinements,
    Typed, TypedProgram, UnreducedKind, Value as CoreValue, ValueProjection, Zero,
};
use ryft_mlir::dialects::func;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Region, SymbolVisibility,
    Type as MlirType, TypeAndAttributes, Value as MlirValue, ValueAndAttributes, ValueRef,
};

use crate::compilation::domains::{commits_public_outputs, declared_input_sharding};
use crate::execution::validate_committed_input_devices;
use crate::lowering::regions::lower_nested_program_inline;
use crate::lowering::types::lower_array_ir_type;
use crate::lowering::{
    CollectiveLoweringState, EffectTokens, collect_session_constants, entry_boundary_sharding,
    has_placement_requests_in_region, has_untracked_auto_axes,
};
use crate::operations::composites::NamedCompositeImplementation;
use crate::operations::control_flow::lowered_effect_classes;
use crate::operations::manipulation::conversions::lower_cast_shape;
use crate::operations::{FlatProgram, Operation, Program};
use crate::{
    Array, Constant, Error, LowerableOperation, LowerableOperationFamily, Lowerer, LoweringError, Session, Type, Value,
};

/// Structural identity of a flat callee program attached behind a `jit_call`, used to deduplicate repeated nested
/// programs into shared private `func.func`s at lowering time.
///
/// Eligible programs (refer to [`supports_structural_deduplication`]) are keyed by their canonical rendering plus their
/// complete flat input/output signature. Type inference is deterministic and attached regions (i.e., control-flow
/// bodies, composite programs, and nested `jit_call` callees) render contextually inside their instructions, so two
/// programs that render identically with equal boundary types compute the same function and may share one emitted
/// function. This holds even when they are distinct staged programs produced by separate transform passes (e.g., the
/// per-block primal and pullback programs of `grad(jit(f))` over repeated blocks).
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct JitCallProgramKey {
    /// Canonical [`Program`](ryft_core::Program) rendering (i.e., operation names, all bracketed attributes, and
    /// attached region bodies).
    rendered: String,

    /// Flat input types, which together with the rendering pin the full callee signature.
    input_types: Vec<Type>,

    /// Flat output types, completing the callee signature when output-only placement metadata is not visible
    /// in the rendered body.
    output_types: Vec<Type>,

    /// Payload of every call in the callee and in the regions that it reaches, in the order of a depth-first walk of
    /// its instructions, compared by the equality of [`JitCallOperation`]. Each of these calls lowers its boundary
    /// inside the shared function, so everything that equality compares (e.g., a mesh, an override, a kept input
    /// placement, or a derived input boundary) must distinguish callees, including what the rendering of a call does
    /// not show (refer to [`JitCallOperation`]).
    calls: Vec<JitCallOperation<Type>>,
}

/// One deduplicated callee emitted as a shared private `func.func`.
pub(crate) struct JitCallFunction<O = Operation> {
    /// Symbol name of the emitted private function.
    symbol: String,

    /// Representative callee program for this key (materialized from its callee region), lowered once as the
    /// function body.
    program: FlatProgram<O>,

    /// Flat input types of the callee, also the emitted function's argument types.
    input_types: Vec<Type>,

    /// Flat output types of the callee, also the emitted function's result types.
    output_types: Vec<Type>,
}

impl<O> JitCallFunction<O> {
    /// Returns the symbol name of the emitted private function.
    #[inline]
    pub(crate) fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Returns the flat input types of the callee, also the emitted function's argument types.
    #[inline]
    pub(crate) fn input_types(&self) -> &[Type] {
        &self.input_types
    }

    /// Returns the flat output types of the callee, also the emitted function's result types.
    #[inline]
    pub(crate) fn output_types(&self) -> &[Type] {
        &self.output_types
    }
}

/// Shared private functions emitted for `jit_call` callees that occur more than once in a module.
///
/// Built once by [`collect_jit_call_functions`] before a module is lowered and threaded read-only through the
/// lowering pass. At each `jit_call` lowering site, a callee whose key is present is emitted as a `func.call` to the
/// shared function instead of being inlined, and absent callees inline.
pub(crate) struct JitCallFunctionMap<O = Operation> {
    /// Shared functions keyed by callee identity.
    functions: HashMap<JitCallProgramKey, JitCallFunction<O>>,

    /// Keys in first-occurrence order, so emitted symbol names and module layout are deterministic.
    order: Vec<JitCallProgramKey>,

    /// Shared functions that implement the bodies of named composite calls, indexed by the calls' semantic identity
    /// and then by their boundary, so that lowering a call finds its function without rendering or hashing its body.
    named: HashMap<CompositeKey, Vec<NamedFunction>>,
}

impl<O> Default for JitCallFunctionMap<O> {
    fn default() -> Self {
        Self { functions: HashMap::new(), order: Vec::new(), named: HashMap::new() }
    }
}

impl<O: LowerableOperationFamily> JitCallFunctionMap<O> {
    /// Returns the shared function for `program`, if one was emitted for its identity.
    pub(crate) fn get(&self, program: &FlatProgram<O>) -> Option<&JitCallFunction<O>> {
        self.functions.get(&jit_call_program_key(program)?)
    }

    /// Returns the shared function that implements the body of the named composite calls of `key` at the provided
    /// boundary, if one was emitted for them.
    ///
    /// The index assumes that named composite calls with equal keys and equal boundaries carry equal bodies, which
    /// the contract of [`NamedComposite`](ryft_core::NamedComposite) requires but cannot verify (and which does not
    /// hold across definitions with equal keys under different named axes, or in different operation families).
    /// [`collect_jit_call_functions`] enforces it by leaving out of the index any key and boundary whose calls carry
    /// different bodies, so those calls look up their shared function by their own body (refer to [`Self::get`]).
    pub(crate) fn named(
        &self,
        key: &CompositeKey,
        input_types: &[Type],
        output_types: &[Type],
    ) -> Option<&NamedFunction> {
        self.named
            .get(key)?
            .iter()
            .find(|function| function.input_types == input_types && function.output_types == output_types)
    }

    /// Returns the shared functions keyed by callee identity.
    #[inline]
    pub(crate) fn functions(&self) -> &HashMap<JitCallProgramKey, JitCallFunction<O>> {
        &self.functions
    }

    /// Returns the keys in first-occurrence order, so emitted symbol names and module layout are deterministic.
    #[inline]
    pub(crate) fn order(&self) -> &[JitCallProgramKey] {
        &self.order
    }
}

/// Shared function that implements the body of the named composite calls with one boundary.
pub(crate) struct NamedFunction {
    /// Input types of the calls.
    input_types: Vec<Type>,

    /// Output types of the calls, which are also the results of the function.
    output_types: Vec<Type>,

    /// Symbol of the function.
    symbol: String,
}

impl NamedFunction {
    /// Returns the output types of the calls, which are also the results of the function.
    #[inline]
    pub(crate) fn output_types(&self) -> &[Type] {
        &self.output_types
    }

    /// Returns the symbol of the function.
    #[inline]
    pub(crate) fn symbol(&self) -> &str {
        &self.symbol
    }
}

/// Builds the [`JitCallFunctionMap`] for a module by emitting a shared private function for every `jit_call` callee
/// that occurs at least twice (per [`JitCallProgramKey`] identity), and for every named composite body that no
/// backend implementation replaces on `target_platform` and that occurs at least twice. Single-occurrence callees are
/// left to inline, so modules without repeated calls lower exactly as before. A shared function that named composite
/// calls reach is named after the name and version of the first such definition (e.g., `ryft.example.v1`, with a
/// numeric suffix for other specializations), even when a `jit_call` also reaches it, and no `jit_call` or
/// named-composition function name can match such a name. A named composite boundary is indexed only when it equals
/// the boundary of its shared function, so that the emitted call signature matches the function.
///
/// # Errors
///
/// Returns an error when the effects that lowering a callee introduces cannot be determined (refer to
/// [`lowered_effect_classes`]).
pub(crate) fn collect_jit_call_functions<Input, Output, O: LowerableOperationFamily>(
    program: &Program<Input, Output, O>,
    target_platform: Option<&str>,
) -> Result<JitCallFunctionMap<O>, Error>
where
    Input: Parameterized<Constant>,
    Output: Parameterized<Constant>,
{
    let mut counts: HashMap<JitCallProgramKey, (usize, FlatProgram<O>)> = HashMap::new();
    let mut order: Vec<JitCallProgramKey> = Vec::new();
    let mut memo: HashMap<RegionId, Option<JitCallProgramKey>> = HashMap::new();
    let mut named = HashMap::new();
    let mut names = HashMap::new();
    count_jit_calls(program, target_platform, &mut counts, &mut order, &mut memo, &mut named, &mut names)?;

    let mut map = JitCallFunctionMap::default();
    let mut name_counts = HashMap::<String, usize>::new();
    for key in order {
        let (count, program) = counts.remove(&key).unwrap();
        // Effectful callees always inline (even when repeated) so their effectful instructions chain onto the
        // caller's effect token in program order, which a shared token-free function could not preserve. That
        // includes the entry guards that lowering synthesizes for the callee's nested control flow. Callees that
        // reference session constants also inline, because only the entry function receives their hidden arguments.
        if count < 2 || !lowered_effect_classes(&program)?.is_empty() || !collect_session_constants(&program).is_empty()
        {
            continue;
        }
        let symbol = match names.get(&key) {
            Some(name) => {
                let count = name_counts.entry(name.clone()).or_default();
                *count += 1;
                if *count == 1 { name.clone() } else { format!("{name}_{}", *count - 1) }
            }
            None => format!("jit_call_{}", map.order.len()),
        };
        let input_types = program.input_types().into_owned();
        let output_types = program.output_types().into_owned();
        map.functions.insert(key.clone(), JitCallFunction { symbol, program, input_types, output_types });
        map.order.push(key);
    }
    for (named_key, key) in named {
        let Some(function) = key.and_then(|key| map.functions.get(&key)) else {
            continue;
        };
        if function.input_types != named_key.input_types || function.output_types != named_key.output_types {
            continue;
        }
        let NamedFunctionKey { key: composite_key, input_types, output_types } = named_key;
        let symbol = function.symbol.clone();
        map.named
            .entry(composite_key)
            .or_default()
            .push(NamedFunction { input_types, output_types, symbol });
    }
    Ok(map)
}

/// Emits the shared private `func.func` for one deduplicated callee into `module_block`.
///
/// The body is lowered with `nested_functions` in scope so that any repeated `jit_call`s inside this callee also
/// lower to `func.call`s. Calls between shared functions are resolved by symbol, so emission order does not matter. The
/// body is also lowered with the module's `collective_state`, so module-scoped lowering state (i.e., the shared channel
/// id counter and the target platform that gates platform-specific fast paths such as fused attention) reaches the
/// callee body exactly like it reaches inlined callees. The mesh and manual region state intentionally resets per
/// function.
pub(crate) fn emit_jit_call_function<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
    module_block: &mut BlockRef<'b, 'c, 't>,
    function: &JitCallFunction<O>,
    nested_functions: &Rc<JitCallFunctionMap<O>>,
    collective_state: &CollectiveLoweringState,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), Error> {
    let argument_tensor_types = function
        .input_types
        .iter()
        .map(|r#type| lower_array_ir_type(r#type, context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let result_tensor_types = function
        .output_types
        .iter()
        .map(|r#type| lower_array_ir_type(r#type, context, location))
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
        // Deduplicated callees are pure by construction (`collect_jit_call_functions` skips programs whose lowering
        // has effects, including synthesized entry guards), so the shared function body never creates an effect token.
        let mut effect_tokens = EffectTokens::default();
        let outputs = lower_nested_program_inline(
            &function.program,
            input_values.as_slice(),
            &mut function_block_ref,
            context,
            location,
            &[],
            &RematerializationOptimizationBarrier::None,
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

/// Calls the shared private `func.func` named `symbol` with the provided operands, which must match its inputs, and
/// returns its results, typed by `output_types`.
pub(crate) fn lower_shared_function_call<'b, 'c: 'b, 't: 'c>(
    symbol: &str,
    output_types: &[Type],
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    let result_tensor_types = output_types
        .iter()
        .map(|r#type| lower_array_ir_type(r#type, context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let operation = block.append_operation(func::call(
        symbol,
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
    Ok((0..output_types.len()).map(|index| operation.result(index).unwrap().as_ref()).collect())
}

/// Canonical operation name for [`JitCallOperation`].
pub const JIT_CALL_OPERATION_NAME: &str = "jit_call";

/// Staged call to a flat XLA program. The callee program is not part of this payload. It is a shared callee root
/// [`Region`](ryft_core::Region) attached to the [`Instruction`](ryft_core::Instruction) applying the operation (the
/// single `["callee"]` slot), interned by [`Arc`] identity when the call is staged through the
/// [`AttachedRegions`] passed to [`Context::bind`], so repeated calls staged from one
/// function handle share one callee root and remain identity-comparable for call-site deduplication at lowering. The
/// `T` parameter fixes the callee boundary's type universe, allowing the reusable homogeneous-array batching form and
/// the executable array IR form to remain distinct payload types with one [`CoreOperation`](ryft_core::Operation)
/// contract each. The retained `capture_count` names the callee's exact leading lifted-capture input prefix. It
/// participates in operation equality and callee deduplication identity and scopes reference capture resolution during
/// analysis.
///
/// [`CoreOperation::region_data_flow`](ryft_core::Operation::region_data_flow) declares ordinary callee data flow, in
/// which callee inputs receive instruction inputs positionally and instruction outputs originate from the corresponding
/// callee outputs. Capture-prefix retention remains an independent boundary-pruning requirement.
///
/// Declare callee-local result identities through [`with_callee_type_signature`](Self::with_callee_type_signature)
/// before binding the call. The operation allocates caller-owned result names once, so repeated inference retains them
/// while separate calls receive independent local result identities.
///
/// # Calls of compiled functions
///
/// A call stages the callee's source in its caller. When an input or output has a sharding override, or the body
/// requests placement through a `place` whose target names devices, a `constrain_sharding`, or a nested call that names
/// devices, the call uses the callee's mesh. Otherwise it runs on the devices of its caller or public arguments, also
/// when its body only moves a value to another memory, as JAX runs such a `jit` function where its arguments are.
/// Lifted captures follow those devices, even when the original array is committed. Executing the retained executable
/// directly always uses the mesh it was compiled for.
///
/// A call therefore carries the boundary of its callee as that function was compiled (refer to `with_boundary`), which
/// is the sharding override of each of its inputs and outputs, if any, together with the device mesh that the function
/// names, if any. Neither can be read from the callee body where the call is used (e.g., a replicated output
/// override over auto mesh axes stages nothing, and a callee inlined into a caller on other devices no longer shows
/// where it was compiled to run). A call of a `jit` dispatcher additionally records that it keeps each input without
/// an override placed as it is (refer to `with_kept_input_placements`). Each entry point honors the boundary as
/// follows:
///
///   - Lowering constrains the operands and the results that have overrides where the call executes, as an execution
///     of the function annotates its arguments and results, and mesh discovery declares the mesh of these constraints
///     (refer to `collect_nested_sharding_mesh`). Compiling a caller for other devices than those of a call that names
///     devices, or for the replicas of a bounded dynamic program while an override splits a value, fails with an exact
///     error (refer to `validate_placements`), as JAX rejects a nested `jit` with an incompatible device assignment.
///   - Eager dispatch and eager interpretation require committed public arguments to be on the physical devices of
///     a call that names devices. They move captures and uncommitted arguments to those devices and apply the input
///     and output sharding overrides. Eager dispatch places every other input as the executable of the callee
///     does (refer to `boundary_input_sharding`), which is as the callee input type records for a function compiled
///     for its input types (replicated for a type without a sharding), and as the input is for a call of a `jit`
///     dispatcher, whose executables are specialized for the placements of their inputs.
///   - The type rule of the call accepts every input whose type refines the callee input type up to the placement that
///     its sharding describes, so every context that binds a call accepts the same inputs (refer to
///     `validate_call_input_types`). It rejects an input whose unreduced or reduced axes differ from those of the
///     callee input type, which an execution would reshard where it can, because a call does not reshard its inputs.
///     An input of a `shard_map` body may vary along manual mesh axes that the callee input type does not declare,
///     and every output of the call then varies along them too.
///   - A call commits its results when it names devices, when its callee requests any placement (including one that
///     only moves a value to another memory), runs on multiple devices, or has a committed public input.
///   - Program transforms keep the overrides of every input and output that the derived calls keep, together with the
///     device mesh and the kept input placements of the call (e.g., the known and residual calls of a partial
///     evaluation split, or a batched call). A call that a transform creates (e.g., the primal and tangent calls of a
///     linearization, the known and residual calls of a split, or a transposed call) also keeps the effective
///     boundary of every input of the original call that it receives, and places no input that the transform adds
///     (e.g., a residual), because its declared input types are only traced types (refer to
///     `with_derived_boundary_shardings`). Partial evaluation treats a call as forwarding one of its inputs only when
///     the call names no devices and does not place that input at its boundary, so that the call could not move it
///     (refer to `forward_jit_call_inputs`).
///
/// The boundary takes part in equality and hashing, but a call of a function that names no devices and that is not a
/// `jit` dispatcher hashes and renders exactly as it did before boundaries existed. A call that a transform derives
/// renders the boundary that it keeps only when that boundary places an input, so the rendering of a call does not
/// identify it, and the structural deduplication of callees compares the calls that they contain by equality.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JitCallOperation<T: CoreType> {
    /// Number of leading callee inputs that form its lifted lexical capture prefix.
    capture_count: usize,

    /// Type universe of the callee boundary.
    marker: PhantomData<fn() -> T>,

    /// Caller-owned result identities and the formal input scope used to instantiate forwarded names.
    output_types: Option<CallTypeBindings<T>>,

    /// Sharding override of each callee input, by position, or [`None`] when no callee input has one.
    input_shardings: Option<Arc<[Option<Sharding>]>>,

    /// Sharding override of each callee output, by position, or [`None`] when no callee output has one.
    output_shardings: Option<Arc<[Option<Sharding>]>>,

    /// Device mesh that the callee was compiled for, on which its sharding overrides are placed, when the callee names
    /// devices (refer to [`Self::with_boundary`]), or [`None`] when it does not.
    mesh: Option<DeviceMesh>,

    /// Whether the call leaves each input without a sharding override placed as it is, as the executables of a `jit`
    /// dispatcher are specialized for (refer to [`Self::with_kept_input_placements`]).
    keeps_input_placements: bool,

    /// Effective boundary sharding of each callee input without a sharding override, by position, for a call that a
    /// program transform derives from another call (refer to `with_derived_boundary_shardings`), or [`None`] for a
    /// call of a compiled function, whose declared input types define it (refer to [`Self::boundary_input_sharding`]).
    derived_input_shardings: Option<Arc<[Option<Sharding>]>>,
}

impl<T: CoreType + Hash> Hash for JitCallOperation<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // The fields that existed before the boundary shardings hash as the derived implementation did, and an absent
        // boundary writes nothing, so that the keys of programs whose calls have none stay unchanged.
        self.capture_count.hash(state);
        self.marker.hash(state);
        self.output_types.hash(state);
        if let Some(shardings) = &self.input_shardings {
            shardings.hash(state);
        }
        if let Some(shardings) = &self.output_shardings {
            shardings.hash(state);
        }
        if let Some(mesh) = &self.mesh {
            mesh.hash(state);
        }
        if self.keeps_input_placements {
            self.keeps_input_placements.hash(state);
        }
        if let Some(shardings) = &self.derived_input_shardings {
            shardings.hash(state);
        }
    }
}

impl<T: CoreType> JitCallOperation<T> {
    /// Creates a staged `jit_call` operation. The flat callee program is supplied as a shared region attachment to
    /// [`Context::bind`], and `capture_count` identifies its exact leading lifted-capture prefix.
    #[inline]
    pub(crate) fn new(capture_count: usize) -> Self {
        Self {
            capture_count,
            marker: PhantomData,
            output_types: None,
            input_shardings: None,
            output_shardings: None,
            mesh: None,
            keeps_input_placements: false,
            derived_input_shardings: None,
        }
    }

    /// Returns this call with the boundary of its callee, made of `mesh`, the device mesh that the callee was compiled
    /// for when it names devices, and of the sharding override of each callee input and output, by position, where
    /// [`None`] marks a position without one. A callee names devices when it has a sharding override or its program
    /// requests a placement that names devices (refer to `has_placement_requests`), so `mesh` must be present whenever
    /// an override is.
    /// A missing override is implicit past the end of either list. A list without overrides is stored as absent.
    pub(crate) fn with_boundary(
        mut self,
        mesh: Option<&DeviceMesh>,
        input_shardings: Vec<Option<Sharding>>,
        output_shardings: Vec<Option<Sharding>>,
    ) -> Self {
        let normalize = |shardings: Vec<Option<Sharding>>| {
            shardings.iter().any(Option::is_some).then(|| Arc::<[Option<Sharding>]>::from(shardings))
        };
        self.input_shardings = normalize(input_shardings);
        self.output_shardings = normalize(output_shardings);
        debug_assert!(mesh.is_some() || !self.has_boundary_shardings(), "sharding overrides require a mesh");
        self.mesh = mesh.cloned();
        self
    }

    /// Returns this call marked as a call of a `jit` dispatcher, whose executables are specialized for the placement of
    /// each of their inputs without a sharding override, including over the auto mesh axes that input types do not
    /// record (refer to [`CompilationDomain::dispatch_signature`](ryft_core::CompilationDomain::dispatch_signature)).
    /// Such a call leaves those inputs placed as they are, like a JAX `jit` call without input shardings, so that an
    /// eager call of a dispatcher places them as an execution of the dispatcher does. A call that is not marked is a
    /// call of a function compiled ahead of time for its input types, whose executable places such an input as its
    /// type records, and an eager call places it the same way.
    pub(crate) fn with_kept_input_placements(mut self) -> Self {
        self.keeps_input_placements = true;
        self
    }

    /// Instantiates every callee-local result identity freshly for this call. Input identities continue to name the
    /// forwarded quantities. Call this after constructing the callee and before binding a call with local results.
    /// Repeated inference allocates no identities and returns the same result types.
    ///
    /// # Errors
    ///
    /// Returns a type error for an oversized capture prefix or a type that cannot rename its local identities.
    pub fn with_callee_type_signature(mut self, input_types: &[T], output_types: &[T]) -> Result<Self, TypeError> {
        self.validate_capture_prefix(input_types.len())?;
        self.output_types = CallTypeBindings::fresh(input_types, output_types)?;
        Ok(self)
    }

    /// Instantiates a sealed callee's result names, using its cached signature to recognize identity-free boundaries
    /// without cloning their input and output types. An identity-bearing callee follows the ordinary signature path.
    fn with_callee_region<V: CoreValue<Type = T>, O: CoreOperation<Type = T>>(
        mut self,
        callee: RegionRef<'_, V, O>,
    ) -> Result<Self, TypeError> {
        self.validate_capture_prefix(callee.input_ids().len())?;
        if callee.type_identity_signature().identities().is_empty() {
            // A closed signature without identities proves that `CallTypeBindings::fresh` would return `None` before
            // attempting any renaming. Preserve that exact payload, including when replacing an older signature.
            self.output_types = None;
            return Ok(self);
        }
        self.with_callee_type_signature(&callee.input_types(), &callee.output_types())
    }

    /// Retains caller-owned result identities after a callee boundary rewrite, including reference discharge.
    fn with_result_type_signature(
        mut self,
        input_types: &[T],
        callee_outputs: &[T],
        outputs: Vec<T>,
    ) -> Result<Self, TypeError> {
        self.output_types = CallTypeBindings::with_outputs(input_types, callee_outputs, outputs)?;
        Ok(self)
    }

    /// Returns the exact number of leading callee inputs that form the lifted-capture prefix.
    #[inline]
    pub(crate) fn capture_count(&self) -> usize {
        self.capture_count
    }

    /// Returns the sharding override of callee input `index`, if the callee has one.
    #[inline]
    pub(crate) fn input_sharding(&self, index: usize) -> Option<&Sharding> {
        self.input_shardings.as_deref().and_then(|shardings| shardings.get(index)?.as_ref())
    }

    /// Returns the sharding override of callee output `index`, if the callee has one.
    #[inline]
    pub(crate) fn output_sharding(&self, index: usize) -> Option<&Sharding> {
        self.output_shardings.as_deref().and_then(|shardings| shardings.get(index)?.as_ref())
    }

    /// Returns every sharding override of the boundary of the callee, inputs first.
    pub(crate) fn boundary_shardings(&self) -> impl Iterator<Item = &Sharding> {
        let input_shardings = self.input_shardings.iter().flat_map(|shardings| shardings.iter());
        let output_shardings = self.output_shardings.iter().flat_map(|shardings| shardings.iter());
        input_shardings.chain(output_shardings).flatten()
    }

    /// Returns the device mesh that the callee was compiled for, on which its sharding overrides are placed, or
    /// [`None`] when the callee names no devices (refer to [`Self::with_boundary`]).
    #[inline]
    pub(crate) fn mesh(&self) -> Option<&DeviceMesh> {
        self.mesh.as_ref()
    }

    /// Returns whether this call leaves each input without a sharding override placed as it is (refer to
    /// [`Self::with_kept_input_placements`]).
    #[inline]
    pub(crate) fn keeps_input_placements(&self) -> bool {
        self.keeps_input_placements
    }

    /// Returns whether the callee has any boundary sharding override.
    #[inline]
    fn has_boundary_shardings(&self) -> bool {
        self.input_shardings.is_some() || self.output_shardings.is_some()
    }

    /// Returns the input overrides at `indices` and the output overrides at `output_indices`, where [`None`] selects
    /// a position without one, for a call that is derived from this one (e.g., by a program transform).
    fn derived_boundary_shardings(
        &self,
        input_indices: impl IntoIterator<Item = Option<usize>>,
        output_indices: impl IntoIterator<Item = Option<usize>>,
    ) -> (Vec<Option<Sharding>>, Vec<Option<Sharding>>) {
        let inputs = input_indices.into_iter().map(|index| index.and_then(|index| self.input_sharding(index)).cloned());
        let outputs =
            output_indices.into_iter().map(|index| index.and_then(|index| self.output_sharding(index)).cloned());
        (inputs.collect(), outputs.collect())
    }

    /// Returns `derived` with the boundary shardings of this call at the positions that `input_indices` and
    /// `output_indices` select (refer to `derived_boundary_shardings`), with the mesh of this call, if any, because
    /// the derived call runs a part of the callee of this call, which an execution runs on the devices that it names,
    /// and with the placement of its inputs that this call keeps (refer to [`Self::with_kept_input_placements`]).
    ///
    /// `boundary` is the effective boundary sharding of each input of this call without an override (refer to
    /// `effective_input_boundary`), which the derived call receives as its own at the selected positions, so that a
    /// call that a program transform creates places the inputs of the original call as that call does, and leaves
    /// the inputs that the transform adds (e.g., residuals, tangents of inactive inputs, or cotangent references)
    /// where they are, as the declared types of its callee, which are only traced types, say nothing about where an
    /// execution of the original function places them. [`None`] keeps the kind of this call, for a derived call
    /// whose callee keeps the declared input types of the callee of this call (i.e., a pruned call).
    fn with_derived_boundary_shardings<U: CoreType>(
        &self,
        derived: JitCallOperation<U>,
        boundary: Option<&[Option<Sharding>]>,
        input_indices: impl IntoIterator<Item = Option<usize>>,
        output_indices: impl IntoIterator<Item = Option<usize>>,
    ) -> JitCallOperation<U> {
        let input_indices = input_indices.into_iter().collect::<Vec<_>>();
        let boundary = boundary.or(self.derived_input_shardings.as_deref());
        let derived_input_shardings = boundary.map(|boundary| {
            input_indices.iter().map(|index| index.and_then(|index| boundary.get(index)?.clone())).collect()
        });
        let derived = JitCallOperation {
            keeps_input_placements: self.keeps_input_placements,
            derived_input_shardings,
            ..derived
        };
        let Some(mesh) = &self.mesh else {
            return derived;
        };
        let (inputs, outputs) = self.derived_boundary_shardings(input_indices, output_indices);
        derived.with_boundary(Some(mesh), inputs, outputs)
    }

    /// Returns the derived boundary sharding of callee input `index` (refer to `with_derived_boundary_shardings`).
    #[inline]
    fn derived_input_sharding(&self, index: usize) -> Option<&Sharding> {
        self.derived_input_shardings.as_deref().and_then(|shardings| shardings.get(index)?.as_ref())
    }

    /// Returns `known_call` and `residual_call`, the calls of the two sides of a partial evaluation split of this call,
    /// with the boundary shardings of the inputs and outputs of this call that they receive and return, as JAX keeps
    /// the shardings of the known and unknown inputs and outputs of a split `pjit`. Residual edges have none.
    ///
    /// # Parameters
    ///
    ///   - `boundary`: Effective boundary sharding of each input of this call (refer to `effective_input_boundary`).
    ///   - `known_input_indices`: Input of this call that each known callee input receives.
    ///   - `residual_inputs`: Source of each residual callee input.
    ///   - `outputs`: Source of each output of this call among the outputs of the known or the residual callee.
    ///   - `known_output_count`: Number of outputs of the known callee, including its residual edges.
    ///   - `residual_output_count`: Number of outputs of the residual callee.
    fn with_split_boundary_shardings(
        &self,
        known_call: Self,
        residual_call: Self,
        boundary: &[Option<Sharding>],
        known_input_indices: &[usize],
        residual_inputs: &[ResidualInputSource],
        outputs: &[PartialEvaluationOutput<usize>],
        known_output_count: usize,
        residual_output_count: usize,
    ) -> (Self, Self) {
        let mut known_outputs = vec![None; known_output_count];
        let mut residual_outputs = vec![None; residual_output_count];
        for (index, output) in outputs.iter().enumerate() {
            let position = match *output {
                PartialEvaluationOutput::Known(position) => known_outputs.get_mut(position),
                PartialEvaluationOutput::Unknown(position) => residual_outputs.get_mut(position),
            };
            if let Some(position) = position {
                *position = Some(index);
            }
        }
        let residual_input_indices = residual_inputs.iter().map(|source| match *source {
            ResidualInputSource::UnknownInput(index) | ResidualInputSource::KnownInput(index) => Some(index),
            ResidualInputSource::KnownOutput(_) | ResidualInputSource::ResidualEdge(_) => None,
        });
        let known_call = self.with_derived_boundary_shardings(
            known_call,
            Some(boundary),
            known_input_indices.iter().copied().map(Some),
            known_outputs,
        );
        let residual_call = self.with_derived_boundary_shardings(
            residual_call,
            Some(boundary),
            residual_input_indices,
            residual_outputs,
        );
        (known_call, residual_call)
    }

    /// Checks that the lexical capture namespace is a valid prefix of this callee's actual boundary.
    fn validate_capture_prefix(&self, input_count: usize) -> Result<(), TypeError> {
        if self.capture_count > input_count {
            return Err(TypeError::invalid(format!(
                "`jit_call` lifted capture prefix {} exceeds its {input_count} callee inputs",
                self.capture_count,
            )));
        }
        Ok(())
    }

    /// Rebinds forwarded result names from the retained formal input scope to this callee instantiation.
    fn instantiated_output_types(&self, inputs: &[T]) -> Result<Option<Vec<T>>, TypeError> {
        self.output_types.as_ref().map(|types| types.instantiated(inputs)).transpose()
    }

    /// Validates exact local-result instantiation and returns the callee-to-caller result correspondence.
    fn output_renaming(
        &self,
        inputs: &[T],
        callee_outputs: &[T],
    ) -> Result<TypeIdentityRenaming<T::Identity>, TypeError> {
        if let Some(types) = &self.output_types {
            return types.renaming(inputs, callee_outputs);
        }
        if callee_outputs
            .iter()
            .flat_map(CoreType::identities)
            .any(|(_, identity)| !inputs.iter().flat_map(CoreType::identities).any(|(_, input)| input == identity))
        {
            return Err(TypeError::invalid(
                "`jit_call` with callee-local result identities requires `with_callee_type_signature`",
            ));
        }
        Ok(TypeIdentityRenaming::new())
    }
}

impl<T: ArrayMemberType> JitCallOperation<T> {
    /// Returns the effective boundary sharding of callee input `index`, an input of type `actual` for a callee input of
    /// type `declared`, which is where an execution of the callee places that input. This is the sharding override of
    /// the input when the callee has one, which replaces its declared sharding. Otherwise, for a call of a function
    /// compiled for its input types, it is where the executable of that function places the input (refer to
    /// [`declared_input_sharding`]), which is the sharding of the declared input type, including over the auto mesh
    /// axes that the type does not record (e.g., a declared replicated input stays replicated over them), and for a
    /// declared input type without a sharding, replicated over the mesh of the call, which is the mesh that the callee
    /// names, if any, and otherwise the mesh of `actual`, on whose devices the call runs. A call that a program
    /// transform derives from another call instead has the effective boundary of the original call for the inputs that
    /// it receives from it and none for the inputs that the transform adds (refer to
    /// `with_derived_boundary_shardings`), because the declared types of its callee are only traced types.
    ///
    /// There is none for an input of the lifted capture prefix, for an input that a call of a `jit` dispatcher keeps
    /// placed as it is (refer to [`Self::with_kept_input_placements`]), for an input that varies along manual mesh
    /// axes that the declared type does not declare, which the declared sharding does not describe, for a declared
    /// input type without a sharding when the mesh of the call is unknown (i.e., `actual` has no sharding), and for a
    /// placement over a single device, which places nothing that the devices of the call do not. Eager calls, region
    /// replay, call lowering, the result placements of compiled programs, and the admission of bounded dynamic callers
    /// all consult this function, so that they agree on where a call places its inputs.
    pub(crate) fn boundary_input_sharding<'s>(
        &'s self,
        index: usize,
        actual: &ArrayType,
        declared: &'s T,
    ) -> Option<Cow<'s, Sharding>> {
        if index < self.capture_count {
            return None;
        }
        if let Some(sharding) = self.input_sharding(index) {
            return Some(Cow::Borrowed(sharding));
        }
        if self.keeps_input_placements {
            return None;
        }
        if self.derived_input_shardings.is_some() {
            return self.derived_input_sharding(index).map(Cow::Borrowed);
        }
        let Ok(declared) = declared.as_array_type() else { return None };
        if !call_input_variation(declared, actual).is_some_and(|axes| axes.is_empty()) {
            return None;
        }
        let sharding = match declared.sharding() {
            Some(sharding) => Cow::Borrowed(sharding),
            None => {
                let mesh = match &self.mesh {
                    Some(mesh) => mesh.logical_mesh(),
                    None => actual.sharding()?.mesh(),
                };
                Cow::Owned(declared_input_sharding(declared, mesh))
            }
        };
        (sharding.mesh().device_count() > 1).then_some(sharding)
    }

    /// Returns the effective boundary sharding of each callee input without a sharding override (refer to
    /// [`Self::boundary_input_sharding`]), for a call of a callee whose declared input types are `declared` with inputs
    /// of types `actual`, which a call that a program transform derives from this call receives (refer to
    /// `with_derived_boundary_shardings`).
    fn effective_input_boundary<'t>(
        &self,
        declared: &[T],
        actual: impl IntoIterator<Item = Cow<'t, T>>,
    ) -> Vec<Option<Sharding>>
    where
        T: 't,
    {
        declared
            .iter()
            .zip(actual)
            .enumerate()
            .map(|(index, (declared, actual))| match actual.as_array_type() {
                Ok(actual) if self.input_sharding(index).is_none() => {
                    self.boundary_input_sharding(index, actual, declared).map(Cow::into_owned)
                }
                _ => None,
            })
            .collect()
    }
}

impl JitCallOperation<Type> {
    /// Prepares the inputs of an eager call, including calls replayed by a region interpreter. An explicit boundary
    /// checks the devices of committed public arguments before placing inputs. Otherwise the public arguments
    /// select the devices and the lifted captures follow them, as closed-over constants do in JAX.
    ///
    /// # Errors
    ///
    /// Returns the error from selecting the mesh or placing an argument or capture on it.
    pub(crate) fn eager_inputs<'o>(
        &self,
        session: &Session,
        inputs: &'o [Value],
        declared: &[Type],
    ) -> Result<Cow<'o, [Value]>, Error> {
        if let Some(mesh) = &self.mesh {
            for (index, input) in inputs.iter().skip(self.capture_count).enumerate() {
                if let Value::Array(array) = input {
                    validate_committed_input_devices(array, index, mesh)?;
                }
            }
            // Check the values' types before an override changes their sharding, including any reduction state.
            let types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
            validate_call_input_types(JIT_CALL_OPERATION_NAME, declared, &types)?;
            let placed = inputs
                .iter()
                .enumerate()
                .map(|(index, input)| match input {
                    Value::Array(array) => {
                        self.boundary_input(session, mesh, index, array.clone(), &declared[index]).map(Value::Array)
                    }
                    _ => Ok(input.clone()),
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(Cow::Owned(placed));
        }

        // Committed public arguments fix the devices of the call, so they must agree on them with one another, as
        // JAX rejects a `jit` call whose committed arguments are on different devices ("Received incompatible devices
        // for jitted computation").
        let mut committed =
            inputs.iter().skip(self.capture_count).enumerate().filter_map(|(index, input)| match input {
                Value::Array(array) if array.is_committed() => Some((index, array)),
                _ => None,
            });
        if let Some((first_index, first)) = committed.next() {
            for (index, array) in committed {
                if array.mesh().devices() != first.mesh().devices() {
                    let devices = |array: &Array| array.mesh().devices().iter().map(Device::id).collect::<Vec<_>>();
                    return Err(ProgramError::InvalidArgument {
                        message: format!(
                            "public argument #{index} is committed to devices {:?}, but public argument \
                             #{first_index} is committed to devices {:?}",
                            devices(array),
                            devices(first),
                        ),
                    }
                    .into());
                }
            }
        }
        let mut prepared = self.capture_inputs(session, inputs)?;
        for (index, declared) in declared.iter().enumerate().skip(self.capture_count) {
            let Some(Value::Array(array)) = prepared.get(index) else { continue };
            if let Some(sharding) = self.boundary_input_sharding(index, &array.r#type(), declared)
                && array.sharding() != sharding.as_ref()
            {
                let mesh = DeviceMesh::new(sharding.mesh().clone(), array.mesh().devices().to_vec())?;
                let placed = array.clone().reshard_on_devices(session, &mesh, &sharding, false)?;
                prepared.to_mut()[index] = Value::Array(placed);
            }
        }
        Ok(prepared)
    }

    /// Returns the effective boundary sharding of callee input `index` (refer to [`Self::boundary_input_sharding`])
    /// when the call can move an input of type `actual` to it, which is when it is a sharding override, when it
    /// differs from the sharding of `actual`, or when its mesh has auto axes, along which `actual` cannot show where
    /// the input is placed. Call lowering constrains exactly these inputs, and partial evaluation forwards an input
    /// through a call only when it is not one of them.
    pub(crate) fn boundary_input_placement<'s>(
        &'s self,
        index: usize,
        actual: &ArrayType,
        declared: &'s Type,
    ) -> Option<Cow<'s, Sharding>> {
        let sharding = self.boundary_input_sharding(index, actual, declared)?;
        let moves = self.input_sharding(index).is_some()
            || actual.sharding() != Some(sharding.as_ref())
            || has_untracked_auto_axes(&sharding);
        moves.then_some(sharding)
    }

    /// Returns the placement that the executable of an eager call must give callee input `index`, which receives
    /// `array` for a callee input of type `declared`. This is the effective boundary sharding of the input when the
    /// call has one (refer to [`Self::boundary_input_sharding`]). Otherwise, for a call that a program transform
    /// created (refer to `with_derived_boundary_shardings`) or that keeps its input placements (refer to
    /// [`Self::with_kept_input_placements`]), it is the runtime sharding of `array`, because such a call leaves that
    /// input where it is and its executable must not move it to the placement that its traced input type would
    /// give it. It is [`None`] for any other input, whose placement the executable chooses by its own rule.
    pub(crate) fn eager_input_placement(&self, index: usize, array: &Array, declared: &Type) -> Option<Sharding> {
        if let Some(sharding) = self.boundary_input_sharding(index, &array.r#type(), declared) {
            return Some(sharding.into_owned());
        }
        (self.derived_input_shardings.is_some() || self.keeps_input_placements).then(|| array.sharding().clone())
    }

    /// Returns whether the caller chose the result devices locally. An explicit call boundary names devices for
    /// every process. An unpinned call inherits the device assignment of its public arguments or the session default.
    pub(crate) fn results_are_locally_assigned(&self, session: &Session, inputs: &[Value]) -> bool {
        let operands = inputs.iter().skip(self.capture_count).filter_map(|input| match input {
            Value::Array(array) => Some(array),
            Value::Dimension(_) | Value::Reference(_) => None,
        });
        self.mesh.is_none() && session.assigns_placement_locally(operands, &[])
    }

    /// Places an input on the boundary's devices after all committed public arguments have passed the device check,
    /// as its effective boundary sharding requests (refer to [`Self::boundary_input_sharding`]). An input without one
    /// keeps its sharding when the mesh topologies agree and is replicated when they differ. The input keeps its
    /// commitment.
    ///
    /// # Errors
    ///
    /// Returns the error of moving the input (e.g., for an input of another client).
    fn boundary_input(
        &self,
        session: &Session,
        mesh: &DeviceMesh,
        index: usize,
        array: Array,
        declared: &Type,
    ) -> Result<Array, Error> {
        let sharding = match self.boundary_input_sharding(index, &array.r#type(), declared) {
            Some(sharding) => sharding.into_owned(),
            None if array.sharding().mesh().has_same_topology(mesh.logical_mesh()) => array.sharding().clone(),
            None => Sharding::replicated(mesh.logical_mesh().clone(), array.r#type().shape().rank()),
        };
        if array.mesh() == mesh && array.sharding() == &sharding {
            return Ok(array);
        }
        array.reshard_on_devices(session, mesh, &sharding, false)
    }

    /// Returns the device mesh on which `session` executes this call eagerly, with public arguments `inputs`, which is
    /// the mesh of the call when its callee names devices. Otherwise the call runs where its public array arguments
    /// are. A committed public argument selects its devices, and uncommitted public arguments on the same devices keep
    /// them. With no public array argument, or with uncommitted arguments on different devices, the session default
    /// supplies them. Eager dispatch moves the lifted captures to this mesh before it chooses where to run the call
    /// (refer to `capture_inputs`), and region replay runs the operations of the callee on it, so that the two agree
    /// (e.g., on where an operation of the callee without array inputs, such as a `fill`, places its result).
    ///
    /// # Errors
    ///
    /// Returns the error of selecting the default devices of `session`.
    fn execution_mesh(&self, session: &Session, inputs: &[Value]) -> Result<DeviceMesh, Error> {
        if let Some(mesh) = &self.mesh {
            return Ok(mesh.clone());
        }
        let operands = inputs.iter().skip(self.capture_count).filter_map(|input| match input {
            Value::Array(array) => Some(array),
            Value::Dimension(_) | Value::Reference(_) => None,
        });
        Ok(match operands.clone().find(|array| array.is_committed()) {
            Some(committed) => committed.mesh().clone(),
            None => match operands.clone().next() {
                Some(first) if operands.clone().all(|other| other.mesh().devices() == first.mesh().devices()) => {
                    first.mesh().clone()
                }
                _ => session.device_mesh_for(&[], None)?,
            },
        })
    }

    /// Returns `inputs`, the inputs of this call that `session` executes eagerly, with the arrays of the lifted capture
    /// prefix moved to where the call runs (refer to `execution_mesh`) when the callee names no devices. The capture
    /// keeps its logical mesh and sharding when the device count matches, and is replicated otherwise. The temporary
    /// handle contributes no commitment to the call.
    ///
    /// # Errors
    ///
    /// Returns the error of moving a capture (e.g., for a capture of another client).
    fn capture_inputs<'o>(&self, session: &Session, inputs: &'o [Value]) -> Result<Cow<'o, [Value]>, Error> {
        if self.capture_count == 0 || inputs.len() < self.capture_count {
            return Ok(Cow::Borrowed(inputs));
        }
        let target = self.execution_mesh(session, inputs)?;
        let mut prepared = Cow::Borrowed(inputs);
        for (index, input) in inputs[..self.capture_count].iter().enumerate() {
            let Value::Array(array) = input else { continue };
            // The callee can name the capture's logical mesh in its types while running on the caller's devices.
            let logical_mesh = array.sharding().mesh();
            let (mesh, sharding) = if logical_mesh.has_same_topology(target.logical_mesh()) {
                (target.clone(), array.sharding().clone())
            } else if logical_mesh.device_count() == target.device_count() {
                (DeviceMesh::new(logical_mesh.clone(), target.devices().to_vec())?, array.sharding().clone())
            } else {
                (target.clone(), Sharding::replicated(target.logical_mesh().clone(), array.r#type().shape().rank()))
            };
            let capture = if array.mesh() == &mesh && array.sharding() == &sharding {
                if !array.is_committed() {
                    continue;
                }
                array.clone()
            } else {
                array.clone().reshard_on_devices(session, &mesh, &sharding, false)?
            };
            prepared.to_mut()[index] = Value::Array(capture.with_committed(false));
        }
        // Eager dispatch compiles the callee on its first array operand's logical mesh. Region replay executes
        // individual operations, whose type rules see the runtime shardings. Give replicated arguments the same
        // logical mesh on those devices, so replay can combine them with the capture just as compiled execution can.
        let callee_mesh = prepared[..self.capture_count].iter().find_map(|input| match input {
            Value::Array(array) => Some(array.mesh().clone()),
            _ => None,
        });
        if let Some(mesh) = callee_mesh {
            for (index, input) in inputs.iter().enumerate().skip(self.capture_count) {
                if let Value::Array(array) = input
                    && array.mesh().devices() == mesh.devices()
                    && array.mesh().logical_mesh() != mesh.logical_mesh()
                    && array.sharding().is_replicated()
                {
                    let sharding = Sharding::replicated(mesh.logical_mesh().clone(), array.sharding().rank());
                    prepared.to_mut()[index] =
                        Value::Array(array.clone().reshard_on_devices(session, &mesh, &sharding, false)?);
                }
            }
        }
        Ok(prepared)
    }

    /// Returns `array`, the result `index` of this call that `session` interprets eagerly, placed on the devices of
    /// the mesh of the call as the override of that result requests, like an execution of the callee places it, or
    /// `array` unchanged when the result has no override.
    ///
    /// # Errors
    ///
    /// Returns the error of placing the result.
    fn boundary_output(&self, session: &Session, index: usize, array: Array) -> Result<Array, Error> {
        match (&self.mesh, self.output_sharding(index)) {
            (Some(mesh), Some(sharding)) if array.mesh() != mesh || array.sharding() != sharding => {
                array.reshard_on_devices(session, mesh, sharding, false)
            }
            _ => Ok(array),
        }
    }
}

impl StagedCallOperation<Constant> for Operation {
    #[inline]
    fn staged_call(capture_count: usize) -> Self {
        Self::JitCall(JitCallOperation::new(capture_count))
    }

    fn with_call_types(self, input_types: &[Type], output_types: &[Type]) -> Result<Self, TypeError> {
        match self {
            Self::JitCall(call) => Ok(Self::JitCall(call.with_callee_type_signature(input_types, output_types)?)),
            operation => Ok(operation),
        }
    }
}

impl<T: ArrayMemberType + From<ArrayType>> CoreOperation for JitCallOperation<T> {
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        JIT_CALL_OPERATION_NAME
    }

    fn defined_type_identities(&self) -> Cow<'_, [T::Identity]> {
        self.output_types.as_ref().map_or(Cow::Borrowed(&[]), |types| Cow::Owned(types.definitions()))
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<T::Identity>) -> Result<Self, TypeError> {
        let mut renamed = self.clone();
        renamed.output_types = self.output_types.as_ref().map(|types| types.renamed(renaming)).transpose()?;
        Ok(renamed)
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        const { &[RegionSlot::computation("callee")] }
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        if region_interfaces.len() != 1 {
            return Err(TypeError::invalid(format!(
                "`{JIT_CALL_OPERATION_NAME}` expects 1 attached callee region but got {}",
                region_interfaces.len(),
            )));
        }
        let callee_interface = &region_interfaces[0];
        self.validate_capture_prefix(callee_interface.input_types().len())?;
        let variation = validate_call_input_types(self.name(), callee_interface.input_types(), input_types)?;
        self.output_renaming(callee_interface.input_types(), callee_interface.output_types())?;
        let output_types = self
            .instantiated_output_types(callee_interface.input_types())?
            .unwrap_or_else(|| callee_interface.output_types().to_vec());
        if callee_interface.input_types() == input_types {
            return Ok(output_types);
        }

        // Inputs that refine the callee input types (e.g., with a static extent for a dynamic one) refine the outputs
        // that name the same identities, which the identities of the inputs no longer define. A sharding that the
        // callee input type does not have establishes no such fact, except that the outputs vary along the manual
        // mesh axes along which the inputs vary beyond the callee input types.
        let output_types = match T::Refinements::new(callee_interface.input_types(), input_types) {
            Ok(refinements) => output_types
                .iter()
                .map(|r#type| refinements.refine_declared(r#type, &[]))
                .collect::<Result<_, _>>()?,
            Err(_) => output_types,
        };
        match variation {
            Some(variation) => variation.apply(self.name(), output_types),
            None => Ok(output_types),
        }
    }

    fn infer_region_input_types(
        &self,
        input_types: &[T],
        interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<Option<Vec<T>>>, TypeError> {
        check_count!("region", interfaces, 1, TypeError);
        let declared = interfaces[0].input_types();
        if declared == input_types {
            return Ok(vec![None]);
        }

        // Inputs that instantiate identities of the callee rename them in the callee, whose inputs otherwise keep their
        // declared types, because the call accepts inputs whose types only refine them (refer to
        // `validate_call_input_types`), as an execution of the callee accepts its arguments.
        match T::derive_identity_renaming(declared, input_types) {
            Ok(renaming) if !renaming.is_identity() => {
                let renamed =
                    declared.iter().map(|r#type| r#type.rename_identities(&renaming)).collect::<Result<Vec<_>, _>>()?;
                validate_call_input_types(self.name(), &renamed, input_types)?;
                Ok(vec![Some(renamed)])
            }
            _ => {
                validate_call_input_types(self.name(), declared, input_types)?;
                Ok(vec![None])
            }
        }
    }

    #[inline]
    fn input_region_provenance(&self, region_index: usize, input_index: usize) -> InputRegionProvenance {
        if region_index == 0 {
            InputRegionProvenance::Input { index: input_index }
        } else {
            InputRegionProvenance::None
        }
    }

    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        vec![OutputRegionProvenance { region_index: 0, output_index }]
    }

    fn region_data_flow(&self) -> RegionDataFlow<'_> {
        RegionDataFlow::Provenance
    }

    fn prune_boundary(
        &self,
        input_types: &[Self::Type],
        used_outputs: &[bool],
        regions: &mut dyn RegionLiveness,
    ) -> Result<Option<OperationBoundaryPruning<Self>>, ProgramError> {
        // Operands map onto the callee inputs one for one, so the operands that the callee does not use are dropped
        // together with its unused outputs (as in JAX's `dce_jaxpr_pjit_rule`). The lifted capture prefix is always
        // kept, because the captures in the callee refer to it by absolute position.
        let mut kept_inputs = regions.used_region_inputs(0, used_outputs)?;
        kept_inputs.iter_mut().take(self.capture_count).for_each(|kept| *kept = true);
        check_count!("input", kept_inputs, input_types.len(), ProgramError);
        let mut operation = self.clone();
        operation.output_types =
            self.output_types.as_ref().map(|types| types.pruned(&kept_inputs, used_outputs)).transpose()?;
        // The boundary shardings follow the kept callee inputs and outputs.
        let kept_input_indices = kept_inputs.iter().enumerate().filter_map(|(index, kept)| kept.then_some(Some(index)));
        let kept_output_indices =
            used_outputs.iter().enumerate().filter_map(|(index, used)| used.then_some(Some(index)));
        let operation = self.with_derived_boundary_shardings(
            JitCallOperation { input_shardings: None, output_shardings: None, mesh: None, ..operation },
            None,
            kept_input_indices,
            kept_output_indices,
        );
        Ok(Some(OperationBoundaryPruning { operation, kept_inputs, kept_outputs: used_outputs.to_vec() }))
    }

    #[inline]
    fn region_capture_input_count(&self, region_index: usize) -> Option<usize> {
        (region_index == 0).then_some(self.capture_count)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        let operation = OperationFormatter::new(formatter, indentation, JIT_CALL_OPERATION_NAME)?;
        let derived_input_shardings =
            self.derived_input_shardings.as_deref().filter(|shardings| shardings.iter().any(Option::is_some));
        if self.capture_count == 0
            && self.output_types.is_none()
            && self.mesh.is_none()
            && !self.keeps_input_placements
            && derived_input_shardings.is_none()
        {
            Ok(())
        } else {
            // Boundary shardings and the devices that the callee names render only when present, so that calls of
            // callees that name no devices render as before. A position without an override renders as `_`. The
            // effective boundary that a derived call keeps renders only when it places an input.
            let render_shardings = |shardings: &[Option<Sharding>]| {
                shardings
                    .iter()
                    .map(|sharding| sharding.as_ref().map_or_else(|| "_".to_string(), Sharding::to_string))
                    .collect::<Vec<_>>()
            };
            operation.bracketed(|operation| {
                if self.capture_count != 0 {
                    operation.field("capture_count", self.capture_count)?;
                }
                if let Some(outputs) = &self.output_types {
                    operation.list("output_types", outputs.outputs())?;
                }
                if let Some(shardings) = &self.input_shardings {
                    operation.list("input_shardings", render_shardings(shardings))?;
                }
                if let Some(shardings) = &self.output_shardings {
                    operation.list("output_shardings", render_shardings(shardings))?;
                }
                if let Some(shardings) = derived_input_shardings {
                    operation.list("derived_input_shardings", render_shardings(shardings))?;
                }
                if let Some(mesh) = &self.mesh {
                    operation.list("devices", mesh.devices().iter().map(|device| device.id()))?;
                }
                if self.keeps_input_placements {
                    operation.field("keeps_input_placements", true)?;
                }
                Ok(())
            })
        }
    }
}

// A `jit_call` operation forwards its operands onto its callee's inputs one for one and reports the callee's outputs as
// its own, which is the positionally forwarding shape the shared structured rewrite serves with no leading operands.
// The callee's lifted capture prefix is unaffected, because threaded state is appended after every declared operand,
// so the prefix keeps its positions. Instantiated result types belong to the source callee signature, which discharge
// changes by turning discharged references into arrays and appending state inputs and outputs. The rebuilt call
// therefore keeps the same caller-owned result identities but derives its result metadata from the rebuilt callee
// signature.
impl<T, C, P> ReferenceDischargeableOperation<C, P> for JitCallOperation<T>
where
    T: ArrayMemberType + From<ArrayType> + From<P::Referent>,
    C: Context<Type = T, Operation: From<JitCallOperation<T>>>,
    P: ReferenceDischargePolicy<C>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        let result_renaming = match &self.output_types {
            Some(_) => {
                let callee = driver.region(0)?;
                Some(self.output_renaming(&callee.input_types(), &callee.output_types())?)
            }
            None => None,
        };
        context.discharge_positional_region_operation(self, driver, inputs, 0, |_, regions| {
            let Some(result_renaming) = result_renaming else { return Ok(self.clone()) };
            check_count!("region", regions, 1, TypeError);
            let input_types = regions[0].input_types();
            let callee_outputs = regions[0].output_types();
            let outputs = callee_outputs
                .iter()
                .map(|r#type| r#type.rename_identities(&result_renaming))
                .collect::<Result<Vec<_>, _>>()?;
            let operation = Self { output_types: None, ..self.clone() };
            Ok(operation.with_result_type_signature(&input_types, &callee_outputs, outputs)?)
        })
    }
}

// Online partial evaluation rule for a staged `jit_call` operation, which is ryft's analogue of JAX's call partial
// evaluation rules. It splits the callee against the caller's knownness while preserving the `jit_call` boundary on
// both sides.
//
// The split fires whenever the knownness of the call inputs is mixed (i.e., some inputs are known and some are not).
// All-known and all-unknown calls preserve the original boundary, and their callee is bound as a shared program (an
// interned callee, or a copy retained by the region transform cache of the callee), so that repeated calls of one
// callee keep sharing one region on either side. Under a residual policy, the outputs of a pure all-known call that
// only forward call inputs resolve to those inputs, and a residual planner that demands an output of such a call whose
// callee saves interior values asks this operation to split the call on residual demand (refer to
// `PartiallyEvaluatableOperation::split_on_residual_demand`). That split binds a known call that also returns those
// values and a replay call that recomputes the demanded outputs from them, as JAX's `_pjit_partial_eval_custom` does.
// Whether the mixed call's known inputs are concrete values of an eager known-side context or tracers into a live outer
// trace does not matter, as in JAX's call partial evaluation rules, which always bind the known call. The known side is
// bound through the default fold-or-residualize policy, which compiles and executes it under an eager context (e.g.,
// [`Session`](crate::Session)) and stages it into the outer program under a staging context. The `shard_map` rule
// follows the same policy.
//
// When a split fires, the callee is split through the shared
// [`PartitionedProgram`](ryft_core::partial::PartitionedProgram) machinery. The known side is bound into the
// enclosing known-side context as a `jit_call` over the known call inputs that it uses, and the residual side is
// emitted as the residual `jit_call` over the surviving unknown call inputs plus the known-side call's residual-edge
// outputs. Residual edges that would merely repeat a known call input or a known call output are
// [forwarded](ryft_core::partial::PartitionedProgram::forward_residuals), so the residual call consumes that input or
// output directly and the known-side call does not return the same value twice. For a pure callee (i.e., one without
// effects, references, captures, or nominal type identities), both kinds of split are retained by the region transform
// cache of the callee and bound as shared regions (refer to `split_jit_call`), as JAX caches its call partial
// evaluation per callee, knownness, and policy, so repeated and nested calls of one callee reuse one split instead of
// partitioning and copying the callee at every call site. A callee with references or captures keeps its boundary,
// and a callee with nominal type identities is split per call when its inputs are mixed. Effectful mixed calls also
// share their splits, keyed by the driver's repeated-residual mode because that mode determines effect placement.
// Effectful and identity-bearing all-known calls remain whole on residual demand.
impl<V, C> PartiallyEvaluatableOperation<C> for JitCallOperation<Type>
where
    V: PartialEq
        + CoreValue<Type = Type>
        + ValueProjection<ArrayType, Projected: CoreValue<Type = ArrayType>>
        + CaptureConstant
        + Concretizable<bool>,
    C: Context<Type = Type, Constant = V, Operation = Operation<V>>,
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        let all_known = inputs.iter().all(PartialEvaluationValue::is_known);
        let callee = driver.region(0)?;
        if all_known || inputs.iter().all(PartialEvaluationValue::is_unknown) {
            // An all-known call folds as a whole, unless its callee holds work that must not fold under this context
            // (e.g., a copy that every execution of a specialized program must make), in which case the default rule
            // keeps the whole call residual.
            let outputs = fold_or_residualize_jit_call(self, context, driver, inputs)?;
            if all_known && context.residual_policy().is_some() && !keeps_jit_call_boundary(callee, inputs) {
                return Ok(forward_jit_call_inputs(self, callee, inputs, outputs));
            }
            return Ok(outputs);
        }
        // Retain mixed splits without references, captures, or nominal identities. Unlike all-known replay,
        // partitioning a mixed effectful call preserves its effects and shares only the structural split.
        if can_cache_jit_call_split(callee, inputs) {
            let input_known = inputs.iter().map(PartialEvaluationValue::is_known).collect::<Vec<_>>();
            let partition =
                |callee: RegionRef<'_, V, Operation<V>>| driver.partition_program(context, callee, &input_known);
            let kind = JitCallSplitKind::Knownness(input_known.clone());
            return split_jit_call(self, context, driver, inputs, kind, &partition);
        }

        // Split the callee of any other mixed call per call through the shared online boundary machinery, bind the
        // known side into the enclosing known-side context wrapped in a fresh `jit_call` over the original known call
        // inputs, emit the residual side as the residual `jit_call`, and reassemble the original output order. These
        // splits are not retained because the identity-bearing boundary needs call-local type identities.
        //
        // A reference operand, a reference output, or a reference access anywhere in the callee closure keeps the call
        // whole. Splitting would spread the accesses to one root across two compiled calls whose known side runs
        // first, and a known reference-typed feeder of the residual side would then mean that both sides access the
        // same root, the same reordering hazard that keeps a `scan` body whole. The default rule preserves effect order
        // by placing the whole call on one side instead. An allocation that nothing accesses is dead intrinsic state
        // that neither side retains, so it does not pin the boundary, but an allocation that escapes through a
        // reference output is live state whose later accesses are unknown to the callee, so it does.
        let accesses_references = inputs.iter().any(|input| input.r#type().is_reference())
            || callee.output_types().iter().any(CoreType::is_reference)
            || callee
                .instructions_in_closure()
                .any(|(_, instruction)| instruction.operation().effects().has_reference_accesses());
        if accesses_references {
            return fold_or_residualize_jit_call(self, context, driver, inputs);
        }
        // Partitioning does not remap the absolute indices of capture constants retained in attached regions, so a
        // split boundary would leave them naming a compacted or absent capture prefix, which fails lowering at best and
        // silently aliases an unrelated leading operand at worst. Preserve the original call boundary for any callee
        // whose closure still holds a capture constant.
        if callee.contains_atom_in_closure(|atom| {
            atom.as_constant().is_some_and(|constant| constant.capture_index().is_some())
        }) {
            return fold_or_residualize_jit_call(self, context, driver, inputs);
        }
        let input_known = inputs.iter().map(PartialEvaluationValue::is_known).collect::<Vec<bool>>();
        let partition = driver.partition_program(context, callee, input_known.as_slice())?;
        // A trivial partition, whose known program contains no instructions, hoists no work because its known side
        // can only forward known inputs as residual edges. Keep the original boundary and let the default materialize
        // those knowns directly as residual feeders.
        if partition.known_program().instructions().is_empty() {
            return fold_or_residualize_jit_call(self, context, driver, inputs);
        }
        // Feed the residual call directly from the known call inputs and known call outputs that residual edges would
        // merely repeat (JAX's `in_fwd` and `out_fwd`), so that the known-side call returns each value once.
        let partition = partition.forward_residuals()?;
        // Known inputs keep callee source order, so the known-side callee's leading inputs are exactly the known
        // members of the original lifted-capture prefix that it still uses. The residual callee's inputs are call
        // inputs, known call outputs, and residual edges, which never form a capture prefix. The guard above already
        // preserved the boundary of any callee retaining attached region capture constants, so neither derived callee
        // can hold one.
        let known_capture_count =
            partition.known_input_indices().iter().filter(|index| **index < self.capture_count()).count();
        let known_call = JitCallOperation::new(known_capture_count).with_callee_type_signature(
            &partition.known_program().input_types(),
            &partition.known_program().output_types(),
        )?;
        let residual_call = JitCallOperation::new(0).with_callee_type_signature(
            &partition.residual_program().input_types(),
            &partition.residual_program().output_types(),
        )?;
        let boundary =
            self.effective_input_boundary(&callee.input_types(), inputs.iter().map(PartialEvaluationValue::r#type));
        let (known_call, residual_call) = self.with_split_boundary_shardings(
            known_call,
            residual_call,
            &boundary,
            partition.known_input_indices(),
            partition.residual_inputs(),
            partition.outputs(),
            partition.known_program().output_count(),
            partition.residual_program().output_count(),
        );
        context.inline_partitioned_program(
            partition,
            Some(callee),
            inputs,
            |known_program| (Operation::JitCall(known_call), vec![known_program]),
            |residual_program| (Operation::JitCall(residual_call), vec![residual_program]),
        )
    }

    fn split_on_residual_demand<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
        demanded_outputs: &[bool],
    ) -> Result<Option<ResidualDemandSplit<C::Value>>, ProgramError>
    where
        C::Operation: PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>,
    {
        let callee = driver.region(0)?;
        if !inputs.iter().all(PartialEvaluationValue::is_known)
            || context.residual_policy().is_none()
            || keeps_jit_call_boundary(callee, inputs)
        {
            return Ok(None);
        }
        let partition =
            |callee: RegionRef<'_, V, Operation<V>>| context.partition_residual_demand(callee, demanded_outputs);
        let kind = JitCallSplitKind::ResidualDemand(demanded_outputs.to_vec());
        let mut outputs = split_jit_call(self, context, driver, inputs, kind, &partition)?;
        let replayed_outputs = outputs.split_off(callee.output_types().len());
        Ok(Some(ResidualDemandSplit::new(outputs, replayed_outputs)?))
    }
}

// Batching rule for [`JitCallOperation`]. The callee region is rebatched over the mapped input axes through
// [`BatchingDriver::batch_program`], and the batched call is bound through `context.parent()` with the batched callee
// attached again. An eager client-backed parent (e.g., [`Session`](crate::Session)) compiles and executes the batched
// call immediately, a staging parent stages it into the enclosing program, and a differentiation parent dispatches it
// through its own `jit_call` JVP rule, which is what serves `vmap` nested inside `gradient` and `linearize` closures.
impl<C> BatchableOperation<C, ArrayIrBatchingPolicy> for JitCallOperation<Type>
where
    C: Context<Type = Type>,
    C::Operation: From<Self>,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<Vec<ArrayIrBatch<C::Value>>, BatchingError> {
        let axes = inputs.iter().map(ArrayIrBatch::batch_axis).collect::<Vec<_>>();
        // Only an all-replicated call at an anonymous level can bind its original callee unchanged. A named level must
        // rebatch the callee even then, because the callee may read that level by name (e.g., through `axis_index`),
        // and binding it unchanged would let that read escape the level that binds it.
        if axes.iter().all(BatchAxis::is_replicated) && context.axis_name().is_none() {
            return Ok(context
                .parent()
                .bind(
                    self.clone(),
                    vec![driver.region(0)?.to_program()],
                    &inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>(),
                )?
                .into_iter()
                .map(ArrayIrBatch::replicated)
                .collect::<Vec<_>>());
        }
        // The canonical array IR transform threads one leading dimension through the callee as its first input and
        // first output. Treat the input as an additional explicit capture, so that the original capture order remains,
        // and drop the bookkeeping output from the callee itself rather than after the call. The output only forwards
        // the extent input, and an eager parent compiles the call as a standalone program, whose results must be
        // arrays.
        let original = driver.region(0)?;
        let boundary = self.effective_input_boundary(
            &original.input_types(),
            inputs.iter().map(|input| Cow::Owned(input.unbatched_type())),
        );
        let (callee, axes) = driver
            .batch_program(context, original, &axes, ProgramBatchingOutputAxesPolicy::Natural)?
            .into_parts();
        let kept_outputs = (1..callee.output_count()).collect::<Vec<_>>();
        let callee = callee.with_outputs(&kept_outputs)?;
        let mut values = vec![context.axis_extent().clone()];
        values.extend(inputs.iter().map(|input| input.value().clone()));
        let operation = Self::new(self.capture_count() + 1).with_callee_region(callee.entry_region_ref())?;
        let array_type = |r#type: &Type| match r#type {
            Type::Array(r#type) => Some(r#type.clone()),
            _ => None,
        };
        // The batched callee is created by this transform, so its declared input types are only traced types, and the
        // batched call is a derived call in every case, which places each input at the effective boundary of the
        // original call for its items (refer to `with_derived_boundary_shardings`). The boundary shardings gain the
        // batch dimension of each batched value (refer to `batched_boundary_sharding`), and the leading extent input
        // has none.
        let derived_input_shardings = Some(
            std::iter::once(None)
                .chain(inputs.iter().zip(&boundary).map(|(input, sharding)| {
                    let packed = array_type(&input.value().r#type());
                    batched_boundary_sharding(sharding.as_ref(), &input.batch_axis(), packed.as_ref())
                }))
                .collect(),
        );
        let operation =
            Self { keeps_input_placements: self.keeps_input_placements, derived_input_shardings, ..operation };
        let operation = if let Some(mesh) = self.mesh() {
            let input_shardings = std::iter::once(None)
                .chain(inputs.iter().enumerate().map(|(index, input)| {
                    let packed = array_type(&input.value().r#type());
                    batched_boundary_sharding(self.input_sharding(index), &input.batch_axis(), packed.as_ref())
                }))
                .collect();
            let output_types = callee.output_types();
            let output_shardings = axes
                .iter()
                .enumerate()
                .map(|(index, axis)| {
                    let packed = output_types.get(index).and_then(array_type);
                    batched_boundary_sharding(self.output_sharding(index), axis, packed.as_ref())
                })
                .collect();
            operation.with_boundary(Some(mesh), input_shardings, output_shardings)
        } else {
            operation
        };
        let outputs = context.parent().bind(operation, vec![callee], &values)?;
        check_count!("output", outputs, axes.len(), ProgramError);
        outputs
            .into_iter()
            .zip(axes)
            .map(|(value, axis)| ArrayIrBatch::new(value, axis))
            .collect::<Result<Vec<_>, _>>()
    }
}

impl<C> BatchableOperation<C, ArrayBatchingPolicy> for JitCallOperation<ArrayType>
where
    C: Context<Type = ArrayType>,
    C::Operation: From<JitCallOperation<ArrayType>>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy>,
        driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        let physical_inputs = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
        // Rebatch the callee when inputs are mapped or when it can address the batching level by name. Only an
        // all-replicated call at an anonymous level can bind its original callee unchanged.
        //
        // Unlike the forward mode and transpose rules below, this rebatching is deliberately not served from the
        // callee region's retained transform cache. The batched program depends on live `BatchingContext` state (i.e.,
        // the batch extent, the axis name, and the surrounding nesting), so caching it against the callee region and
        // its input axes alone would risk exactly the unsound key reuse that the cache is designed to avoid. Caching it
        // needs a complete key and a measured benefit, which `.tasks/plan_compilation_transform_composition.md` gates
        // (refer to its gate G8), through a batching policy hook returning `Option<key>` where `None` keeps a policy
        // uncached, so that soundness stays opt-in per policy.
        let boundary = self.effective_input_boundary(
            &driver.region(0)?.input_types(),
            inputs.iter().map(|input| Cow::Owned(input.unbatched_type())),
        );
        let batch_callee = ArrayBatch::common_batch_size(inputs)?.is_some() || context.axis_name().is_some();
        let (batched_callee, output_axes) = if batch_callee {
            let input_batch_axes = inputs
                .iter()
                .map(|input| BatchAxis::from_optional_position(input.batch_axis_position()))
                .collect::<Vec<_>>();
            let (batched_callee, output_axes) = driver
                .batch_program(
                    context,
                    driver.region(0)?,
                    input_batch_axes.as_slice(),
                    ProgramBatchingOutputAxesPolicy::Natural,
                )?
                .into_parts();
            // The core program batching pass returns canonical nonnegative physical positions.
            let output_axes = output_axes
                .iter()
                .map(|axis| axis.axis().map(|axis| usize::try_from(axis.value()).unwrap()))
                .collect::<Vec<_>>();
            (batched_callee.into_simplified()?, output_axes)
        } else {
            let callee = driver.region(0)?;
            let output_axes = vec![None; callee.output_types().len()];
            (callee.to_program(), output_axes)
        };
        let operation = Self::new(self.capture_count()).with_callee_region(batched_callee.entry_region_ref())?;
        // The batched call is a derived call in every case, as in the batching rule of the array IR form, and its
        // boundary shardings gain the batch dimension of each mapped value at its batch axis (refer to
        // `batched_boundary_sharding`).
        let derived_input_shardings = Some(
            inputs
                .iter()
                .zip(&boundary)
                .map(|(input, sharding)| {
                    let axis = BatchAxis::from_optional_position(input.batch_axis_position());
                    let packed = input.value().r#type();
                    batched_boundary_sharding(sharding.as_ref(), &axis, Some(packed.as_ref()))
                })
                .collect(),
        );
        let operation =
            Self { keeps_input_placements: self.keeps_input_placements, derived_input_shardings, ..operation };
        let operation = if let Some(mesh) = self.mesh() {
            let input_shardings = inputs
                .iter()
                .enumerate()
                .map(|(index, input)| {
                    let axis = BatchAxis::from_optional_position(input.batch_axis_position());
                    let packed = input.value().r#type();
                    batched_boundary_sharding(self.input_sharding(index), &axis, Some(packed.as_ref()))
                })
                .collect();
            let output_types = batched_callee.output_types();
            let output_shardings = output_axes
                .iter()
                .enumerate()
                .map(|(index, axis)| {
                    let axis = BatchAxis::from_optional_position(*axis);
                    batched_boundary_sharding(self.output_sharding(index), &axis, output_types.get(index))
                })
                .collect();
            operation.with_boundary(Some(mesh), input_shardings, output_shardings)
        } else {
            operation
        };
        let outputs = context.parent().bind(
            operation,
            AttachedRegions::Callees(&[Arc::new(batched_callee)]),
            &physical_inputs,
        )?;
        outputs
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| ArrayBatch::new(output, BatchAxis::from_optional_position(axis)))
            .collect::<Result<Vec<_>, _>>()
    }
}

// Capture-free forward mode rule (i.e., JVP rule) for [`JitCallOperation`]. When the primal and tangent contexts are
// shared (e.g., for `jvp` and for the fused program that a `rematerialize` rule partitions), the rule binds one fused
// `jit_call` over the callee's forward mode program, projected onto the primal outputs and the live output tangents,
// as JAX's `_pjit_jvp` does. An enclosing partial evaluation then splits that call with its own residual policy.
// Otherwise, it binds a primal `jit_call` and a tangent `jit_call` as ordinary XLA [`Operation`] values through the
// active context. A staging context stages both calls over its shared builder, while an eager context (e.g., a
// client-backed [`Session`](crate::Session)) compiles and executes them immediately.
//
// Both forms realize the identity `jvp(jit(f)) = jit(jvp f)`. Rather than capturing the primal inputs as residual
// factors and staging a linear `jit_call`, the separate form keeps the compilation boundary and threads every residual
// as a plain primal operand edge between two `jit_call`s, so no symbolic capture is ever introduced. The enclosing
// partial evaluation split then discovers the residual operand edges structurally, exactly as it does for the
// condition and rematerialize rules.
//
// For separate contexts, the rule linearizes the callee program capture-free through
// [`Program::linearize`](ryft_core::Program::linearize) under the operands' activity mask (an operand with a live
// tangent is active, while the capture prefix, an operand with a structurally zero tangent, and a plumbing reference
// operand that receives no tangent reference are not, as under JAX's `which_nz`), giving a primal sub-program
// `inputs -> [outputs..., residuals...]` and a tangent sub-program
// `[active_input_tangents..., residuals...] -> [output_tangents...]` together with the residual count. Tangents for
// zero differential spaces are omitted from both compact boundaries, and the tangent sub-program is projected onto the
// output tangents that depend on an active input tangent (refer to
// [`Program::output_dependence`](ryft_core::Program::output_dependence) and
// [`Program::with_outputs`](ryft_core::Program::with_outputs)), since every other output tangent is a structural zero
// (JAX's `which_nz_out`). When no output tangent is live and the projected tangent sub-program has no observable
// effects, the rule binds only the source call and pairs every output with a structurally zero tangent. Otherwise, it:
//
//   1. Wraps the primal sub-program in a fresh `jit_call` and stages it over the operand primals, recovering the
//      primal outputs followed by the residual values (program variables produced by the staged primal call).
//   2. Wraps the projected tangent sub-program in a fresh `jit_call` and stages it over the active operand tangents
//      followed by those residual values, recovering the live output tangents.
//   3. Pairs each primal output tracer with its live tangent output tracer, or with a structural zero for an output
//      without a live tangent, into a [`DifferentiationDual`].
//
// The callee program is materialized from the instruction's callee region in the context's constant universe `V`
// (concretely [`Constant`] for staged XLA programs), so the split halves ride the fresh primal and tangent calls
// as shared callee regions. Preserving both `jit_call` boundaries keeps the callee body out of the caller's program,
// so forward mode over a `jit_call` operation stays compiled rather than inlined.
//
// # Parameters
//
//   - `context`: Active evaluation or staging context used to bind the differentiated calls.
//   - `driver`: Call-scoped access to the attached callee region.
//   - `inputs`: Primal and tangent values for the call operands.
impl<C, V> DifferentiableOperation<C> for JitCallOperation<Type>
where
    C: Context<Type = Type, Constant = V, Operation = Operation<V>> + Zero<C::Value>,
    V: PartialEq
        + CoreValue<Type = Type>
        + ValueProjection<ArrayType, Projected: CoreValue<Type = ArrayType>>
        + CaptureConstant
        + Concretizable<bool>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let callee = driver.region(0)?;
        let input_types = callee.input_types();
        let output_types = callee.output_types();
        let output_count = output_types.len();
        check_count!("input", inputs, input_types.len(), ProgramError);

        // Linearize the callee through the instruction-scoped driver under the call's activity mask. The callee's
        // leading lifted-capture inputs are inactive, numeric and reference alike, because a capture is
        // nondifferentiated plumbing that receives no tangent input. The remaining inputs take the operand duals'
        // activity. An operand with a live tangent is active, while an operand with a structurally zero tangent is
        // inactive like a missing one (i.e., JAX's `which_nz`), so it gets no tangent callee slot instead of being
        // materialized as a zero operand. A plumbing reference operand, whose structurally zero tangent is a missing
        // tangent reference, is therefore inactive too. The primal sub-program produces `[outputs..., residuals...]`
        // and the tangent sub-program consumes `[active(input_tangents)..., residuals...]`. The residual count is the
        // number of trailing outputs of the primal sub-program beyond the original callee outputs.
        let capture_count = self.capture_count();
        let activity = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| index >= capture_count && input.is_tangent_active() && !input.tangent().is_zero())
            .collect::<Vec<_>>();
        let input_indices = activity
            .iter()
            .enumerate()
            .filter_map(|(index, &active)| active.then_some(index))
            .collect::<Vec<_>>();
        // A shared primal/tangent destination binds one fused call over the callee's forward mode program, projected
        // onto its primal outputs and live output tangents (JAX's `_pjit_jvp`). An enclosing partial evaluation then
        // splits that call with its own residual policy, instead of reusing a linearization boundary chosen here.
        if std::ptr::eq(context.primal(), context.tangent()) {
            self.validate_capture_prefix(input_types.len())?;
            let activity = TangentActivity::new(driver, callee, &activity)?;
            let output_activity = activity.output_tangent_liveness();
            let live_tangent_slots = activity.live_tangent_slots(output_activity);
            let program = activity.jvp_program();
            let mut operands = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            if live_tangent_slots.is_empty() && !program.effects().is_retained_when_unused() {
                let callee = shared_jit_call_callee(callee)?;
                return context
                    .primal()
                    .bind(Operation::JitCall(self.clone()), AttachedRegions::Callees(&[callee]), &operands)?
                    .into_iter()
                    .map(DifferentiationDual::new_with_zero_tangent)
                    .collect();
            }
            let outputs = (0..output_count).chain(live_tangent_slots.iter().map(|&slot| output_count + slot));
            let program = if outputs.clone().eq(0..program.output_count()) {
                program.clone()
            } else {
                Arc::new(program.as_ref().clone().with_outputs(&outputs.collect::<Vec<_>>())?)
            };
            operands.extend(
                activity.input_indices().iter().map(|&index| inputs[index].tangent().as_value().unwrap().clone()),
            );

            // The primal operands come first, so the lifted-capture prefix of the source call carries over unchanged.
            // Each tangent has the boundary sharding of its primal, as JAX's `_pjit_jvp` gives each tangent the
            // sharding of its primal.
            let live_outputs = (0..output_count).filter(|&index| output_activity[index]);
            let boundary =
                self.effective_input_boundary(&input_types, inputs.iter().map(|input| input.primal().r#type()));
            let operation = Operation::JitCall(self.with_derived_boundary_shardings(
                JitCallOperation::new(capture_count).with_callee_region(program.entry_region_ref())?,
                Some(&boundary),
                (0..input_types.len()).chain(activity.input_indices().iter().copied()).map(Some),
                (0..output_count).chain(live_outputs).map(Some),
            ));
            let mut outputs = context.primal().bind(operation, AttachedRegions::Callees(&[program]), &operands)?;
            check_count!("output", outputs, output_count + live_tangent_slots.len(), ProgramError);
            let mut tangents = outputs.split_off(output_count).into_iter();
            return outputs
                .into_iter()
                .zip(output_activity)
                .map(|(primal, &live)| match live {
                    true => DifferentiationDual::new(primal, tangents.next().unwrap()),
                    false => DifferentiationDual::new_with_zero_tangent(primal),
                })
                .collect::<Result<Vec<_>, _>>();
        }
        let linearization = driver.linearize_program(callee, &input_indices, Some(&[]))?;

        // An output tangent that depends on no active input tangent (e.g., the tangent of a constant output or of an
        // output computed only from inactive inputs) is a structural zero (JAX's `which_nz_out`), so the tangent callee
        // is projected onto the live output tangents, and every other output is paired with a structurally zero
        // tangent. When no output tangent is live and the tangent callee has no observable effects, no tangent call is
        // bound, and the source callee is bound as the primal call, since nothing would consume its residuals.
        //
        // The tangent program's leading inputs are compact active tangents, not the original primal input indices,
        // and its trailing residual inputs are held independent during dependence analysis. Linearization has
        // validated both boundaries, so subtracting the residual count cannot underflow.
        let tangent_program = linearization.tangent();
        let tangent_input_count = tangent_program.input_count() - linearization.residual_count();
        let tangent_slots = linearization.tangent_output_slots().to_vec();
        let mut dependence =
            tangent_program.output_dependence(&(0..tangent_input_count).collect::<Vec<_>>())?.into_iter();

        // Map compact tangent output dependence back to the original callee's output order. Outputs without tangent
        // slots consume no dependence entry, and trailing primal residual outputs are outside this boundary entirely.
        let output_activity = tangent_slots[..output_count]
            .iter()
            .map(|&slot| slot && dependence.next().unwrap())
            .collect::<Vec<_>>();
        let live_tangent_slots = (0..output_count)
            .filter(|&index| tangent_slots[index])
            .enumerate()
            .filter_map(|(slot, index)| output_activity[index].then_some(slot))
            .collect::<Vec<_>>();

        // Output projection retains the complete tangent/residual input boundary and observable effects. Keep the
        // shared callee handle when every slot is live, preserving the `Arc` identity used for callee interning.
        let tangent_program = if live_tangent_slots.len() == tangent_program.output_count() {
            tangent_program.clone()
        } else {
            Arc::new(tangent_program.as_ref().clone().with_outputs(&live_tangent_slots)?)
        };
        let primal_operands = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        if !output_activity.contains(&true) && !tangent_program.effects().is_retained_when_unused() {
            let callee = shared_jit_call_callee(callee)?;
            return context
                .primal()
                .bind(Operation::JitCall(self.clone()), AttachedRegions::Callees(&[callee]), &primal_operands)?
                .into_iter()
                .map(DifferentiationDual::new_with_zero_tangent)
                .collect();
        }
        let (primal_program, ..) = linearization.into_parts();

        // Wrap the primal sub-program in a fresh `jit_call` and bind it over the operand primals, recovering the
        // primal outputs followed by the residual values. The shared sub-program handles are attached directly, so
        // repeated binds of one derived callee intern by `Arc` identity instead of copying it again.
        // The primal sub-program preserves the callee's inputs in order, so the original lifted-capture prefix
        // survives verbatim and its length carries over.
        // The primal call has the boundary shardings of this call, and its trailing residuals have none.
        let boundary =
            self.effective_input_boundary(&input_types, primal_operands.iter().map(|operand| operand.r#type()));
        let primal_call = Operation::JitCall(self.with_derived_boundary_shardings(
            JitCallOperation::new(self.capture_count()).with_callee_region(primal_program.entry_region_ref())?,
            Some(&boundary),
            (0..input_types.len()).map(Some),
            (0..output_count).map(Some),
        ));
        let mut primal_call_outputs =
            context.primal().bind(primal_call, AttachedRegions::Callees(&[primal_program]), &primal_operands)?;
        if primal_call_outputs.len() < output_count {
            return Err(ProgramError::MalformedProgram(format!(
                "`jit_call` primal program produced {} outputs which is fewer than its {output_count} primal \
                 output(s)",
                primal_call_outputs.len(),
            ))
            .into());
        }
        let residuals = primal_call_outputs.split_off(output_count);
        let primal_outputs = primal_call_outputs;

        // Wrap the tangent sub-program in a fresh `jit_call` and bind it over only the active operand tangents followed
        // by the residual values. Structurally zero tangents are inactive, so every active operand tangent is a live
        // value and materializing it only unwraps it, while inactive operands (the capture prefix, structurally zero
        // tangents, zero-space operands, and plumbing references) have no compact callee boundary slot.
        let mut tangent_operands = inputs
            .iter()
            .zip(&activity)
            .filter(|(_, active)| **active)
            .map(|(input, _)| input.tangent().clone().materialize(context.tangent()))
            .collect::<Result<Vec<_>, _>>()?;
        tangent_operands.extend(
            residuals.into_iter().map(|value| context.primal_to_tangent(value)).collect::<Result<Vec<_>, _>>()?,
        );
        // The tangent callee's inputs are live input tangents followed by residuals and never a lifted-capture prefix.
        // Attached region capture constants cannot appear in it because fresh-root region traces reject bodies that
        // register captures (`ProgramError::DiscardedCaptures`), and a stale reference that slipped in anyway would
        // fail lowering loudly against the tangent callee's empty capture prefix instead of aliasing an unrelated
        // value.
        // Each active input tangent and each live output tangent has the boundary sharding of its primal, and the
        // trailing residual inputs have none.
        let active_inputs = activity.iter().enumerate().filter_map(|(index, active)| active.then_some(Some(index)));
        let live_outputs =
            output_activity.iter().enumerate().filter_map(|(index, active)| active.then_some(Some(index)));
        let tangent_call = Operation::JitCall(self.with_derived_boundary_shardings(
            JitCallOperation::new(0).with_callee_region(tangent_program.entry_region_ref())?,
            Some(&boundary),
            active_inputs,
            live_outputs,
        ));
        let tangent_outputs =
            context
                .tangent()
                .bind(tangent_call, AttachedRegions::Callees(&[tangent_program]), &tangent_operands)?;
        let tangent_output_count = output_activity.iter().filter(|&&active| active).count();
        check_count!("output", tangent_outputs, tangent_output_count, ProgramError);

        let mut tangent_outputs = tangent_outputs.into_iter();
        primal_outputs
            .into_iter()
            .zip(output_activity)
            .map(|(primal, active)| {
                if !active {
                    DifferentiationDual::new_with_zero_tangent(primal)
                } else {
                    let tangent = tangent_outputs.next().unwrap();
                    let refined = primal.r#type().tangent()?;
                    let tangent = refined_derived_value(context.tangent(), self.mesh(), refined, tangent)?;
                    DifferentiationDual::new(primal, tangent)
                }
            })
            .collect::<Result<Vec<_>, _>>()
    }
}

/// Transposes a tangent [`JitCallOperation`] while retaining its compilation boundary.
///
/// The [`TranspositionDriver`] transposes the attached callee under the operands' linearity and cotangent-destination
/// masks and the structural zeros of its output cotangents. Known operands supply residual or captured values and may
/// appear anywhere in the input boundary. The transposed call consumes the ordinary output cotangents that are not
/// structural zeros, live cotangent references, and known operands, in that order. A structural zero output cotangent
/// seeds nothing, so it has no input in the transposed call and is not materialized at the call boundary, although a
/// dense backward rule inside the callee may still materialize the zero that it consumes. Reference outputs forward
/// input roots and therefore have no separate output cotangent slot.
///
/// Returned cotangents follow the original input order. Linear value inputs receive call results, reference inputs
/// accumulate into their supplied cotangent destinations, and known or ignored inputs receive structural zeros.
///
/// # Parameters
///
///   - `operation`: Primal tangent `jit_call` staged into the tangent program.
///   - `context`: Active transpose tracing context the pullback is staged into.
///   - `driver`: Instruction-scoped access to the attached callee region and its recursive transposition machinery.
///   - `inputs`: Per-operand [`PartialValue`] knowledge, mirroring the callee's inputs one to one. The
///     [`Unknown`](PartialValue::Unknown) entries are the input tangents, and the [`Known`](PartialValue::Known)
///     entries carry the residual and captured-constant-tangent tracers the pullback reads.
///   - `outputs`: Symbolic cotangents for the tangent call's outputs, one per callee output. A
///     [`Zero`](MaybeZero::Zero) entry has no input in the transposed call.
///   - `cotangents`: Cotangent destinations of the operands (refer to the documentation of
///     [`TranspositionContext::cotangent_destinations`]). The callee is transposed with their destination kinds, so a
///     live (`Reference`-kind) reference operand's cotangent reference is an operand of the transposed call, which
///     accumulates into it in place and returns it by identity, while a dead (`Ignore`-kind) reference operand has no
///     slot in the transposed callee at all.
pub fn transpose_primal_jit_call<
    V: CoreValue<Type = Type> + ValueProjection<ArrayType, Projected: CoreValue<Type = ArrayType>>,
    D: TranspositionDriver<V, Operation<V>>,
>(
    operation: &JitCallOperation<Type>,
    context: &mut TracingContext<V, Operation<V>>,
    driver: &D,
    inputs: &[PartialValue<Tracer<TracingContext<V, Operation<V>>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, Operation<V>>>>],
    cotangents: &CotangentDestinations<Tracer<TracingContext<V, Operation<V>>>>,
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, Operation<V>>>>>, ProgramError> {
    // A `jit_call` operation with no live output cotangents and no live reference operand is a zero linear map, so
    // every operand cotangent is zero. A live reference operand keeps the call live, because its accumulated state
    // cotangent flows through the transposed callee even when no ordinary output cotangent does. A callee with deferred
    // work or observable rule effects also keeps it live.
    check_count!("input", cotangents.kinds(), inputs.len(), ProgramError);
    if outputs.iter().all(MaybeZero::is_zero)
        && !cotangents.has_reference_state_destinations()
        && !driver.region(0)?.must_transpose()
    {
        return inputs
            .iter()
            .map(|input| input.r#type().cotangent().map(MaybeZero::Zero).map_err(ProgramError::from))
            .collect();
    }

    // Each operand maps to one callee input, independently linear (an input tangent) or known (a residual value or a
    // captured-constant tangent). The linear operands need not lead, because a captured compiled function threads its
    // captured prefix as known leading operands. The dispatch guarantees a `Known` operand carries its pullback value,
    // so each known tracer is read directly in callee-input order.
    let operand_linear = inputs.iter().map(PartialValue::is_unknown).collect::<Vec<_>>();
    let callee = driver.region(0)?;
    check_count!("input", operand_linear, callee.input_types().len(), ProgramError);

    // Transpose the callee with respect to its linear inputs and with its structural zero output cotangents flagged,
    // through the region's retained transform cache so that a callee shared by several outer programs is transposed
    // once per linearity and seed mask. A live reference-typed linear input is transposed with a `Reference`
    // destination and a dead one with an `Ignore` destination, so the transposed callee maps
    // `[live_non_reference_output_cotangents..., cotangent_references..., known_input_values...]` to
    // `[linear_input_cotangents...]`, in callee-input order, where a live reference input's cotangent is its cotangent
    // reference itself and a dead reference input has no cotangent slot. The flags follow the callee's complete output
    // order, and flags of reference outputs are normalized away because those outputs never own a cotangent slot.
    let (input_indices, destination_kinds): (Vec<_>, Vec<_>) = operand_linear
        .iter()
        .zip(cotangents.kinds())
        .enumerate()
        .filter_map(|(index, (linear, kind))| linear.then_some((index, *kind)))
        .unzip();
    let output_types = callee.output_types();
    check_count!("output", outputs, output_types.len(), ProgramError);
    let zero_output_cotangents = outputs.iter().map(MaybeZero::is_zero).collect::<Vec<_>>();
    let transposed_callee = driver.transpose_program_with_zero_seeds(
        callee,
        &input_indices,
        &[],
        &zero_output_cotangents,
        &destination_kinds,
    )?;

    // Stage a fresh `jit_call` over the transposed callee on `[live_outputs..., cotangent_references...,
    // known_input_values...]`. Its outputs are the linear-input cotangents. A reference output forwards a callee input
    // root whose state cotangent lives in that operand's cotangent reference, so it owns no cotangent slot.
    let mut operands = Vec::with_capacity(output_types.len() + inputs.len());
    operands.extend(
        outputs
            .iter()
            .zip(output_types.iter())
            .filter(|(_, output_type)| !output_type.is_reference())
            .filter_map(|(cotangent, _)| cotangent.as_value())
            .cloned(),
    );
    operands.extend(cotangents.references().iter().cloned());
    operands.extend(inputs.iter().filter_map(PartialValue::as_known).cloned());
    // The transposed callee's inputs are live output cotangents, cotangent references, and known input values, and
    // never a lifted-capture prefix. Attached region capture constants cannot appear in it because region bodies are
    // traced through fresh-root contexts.
    // Each output cotangent has the boundary sharding of its output and each known input value that of its input, and
    // the cotangent references have none, as JAX's `_pjit_transpose` gives the cotangents the shardings of their
    // primals. Each returned linear input cotangent has the sharding of its input, whichever other inputs have a
    // structurally zero or absent cotangent, which returns nothing (refer to `TransposedProgramCotangent`).
    let mut transposed_call =
        JitCallOperation::new(0).with_callee_region(transposed_callee.program().entry_region_ref())?;
    transposed_call.keeps_input_placements = operation.keeps_input_placements();
    // The known input values keep the effective boundary of the call (refer to `with_derived_boundary_shardings`),
    // which, in the tangent call of a linearization, places the inputs of the original call that it receives and
    // leaves its residuals where they are.
    let boundary = operation.effective_input_boundary(&callee.input_types(), inputs.iter().map(PartialValue::r#type));
    let output_cotangent_count = outputs
        .iter()
        .zip(output_types.iter())
        .filter(|(cotangent, output_type)| !output_type.is_reference() && cotangent.as_value().is_some())
        .count();
    transposed_call.derived_input_shardings = Some(
        std::iter::repeat_n(None, output_cotangent_count + cotangents.references().len())
            .chain(
                inputs
                    .iter()
                    .zip(boundary)
                    .filter(|(input, _)| input.as_known().is_some())
                    .map(|(_, sharding)| sharding),
            )
            .collect(),
    );
    if let Some(mesh) = operation.mesh() {
        let output_cotangent_shardings = outputs
            .iter()
            .zip(output_types.iter())
            .enumerate()
            .filter(|(_, (cotangent, output_type))| !output_type.is_reference() && cotangent.as_value().is_some())
            .map(|(index, _)| operation.output_sharding(index).cloned());
        let reference_shardings = cotangents.references().iter().map(|_| None);
        let known_input_shardings = inputs
            .iter()
            .enumerate()
            .filter(|(_, input)| input.as_known().is_some())
            .map(|(index, _)| operation.input_sharding(index).cloned());
        let input_shardings =
            output_cotangent_shardings.chain(reference_shardings).chain(known_input_shardings).collect();
        let output_shardings = input_indices
            .iter()
            .zip(transposed_callee.cotangents())
            .filter(|(_, cotangent)| **cotangent == TransposedProgramCotangent::Output)
            .map(|(&index, _)| operation.input_sharding(index).cloned())
            .collect();
        transposed_call = transposed_call.with_boundary(Some(mesh), input_shardings, output_shardings);
    }
    let transposed_call = Operation::JitCall(transposed_call);
    let input_cotangents = context.bind(
        transposed_call,
        AttachedRegions::Callees(&[transposed_callee.program().clone()]),
        operands.as_slice(),
    )?;
    let cotangent_types = input_indices
        .iter()
        .map(|&index| inputs[index].r#type().cotangent().map_err(ProgramError::from))
        .collect::<Result<Vec<_>, _>>()?;
    let mut input_cotangents = transposed_callee.assemble_cotangents(&cotangent_types, input_cotangents)?.into_iter();

    // Reassemble selected cotangents from the call's compact output boundary. Known and ignored inputs carry
    // structural zeros. Reference-state outputs only thread the caller-owned destination, so they do not contribute
    // a value cotangent to the enclosing context.
    inputs
        .iter()
        .map(|input| {
            let cotangent = if input.is_unknown() { input_cotangents.next().unwrap() } else { None };
            match cotangent {
                Some(MaybeZero::Value(cotangent)) if !input.r#type().is_reference() => {
                    let refined = input.r#type().cotangent().map_err(ProgramError::from)?;
                    let cotangent = summed_input_cotangent(context, &refined, cotangent)?;
                    return Ok(MaybeZero::Value(refined_derived_value(context, operation.mesh(), refined, cotangent)?));
                }
                Some(cotangent) if !input.r#type().is_reference() => return Ok(cotangent),
                _ => {}
            }
            input.r#type().cotangent().map(MaybeZero::Zero).map_err(ProgramError::from)
        })
        .collect()
}

// Transpose rule for a primal tangent [`JitCallOperation`], forwarding to [`transpose_primal_jit_call`] with the
// cotangent references of its linear reference-typed operands resolved (and allocated on first use) through the
// enclosing [`TranspositionContext`]. The callee transposition happens on the concretely [`Constant`]-keyed
// [`FlatProgram`], so the recursion is resolved once at definition time and instantiating this implementation
// introduces no recursive [`TransposableOperation`] obligation on [`Operation`].
impl<V> TransposableOperation<V, Operation<V>> for JitCallOperation<Type>
where
    V: CoreValue<Type = Type> + ValueProjection<ArrayType, Projected: CoreValue<Type = ArrayType>>,
{
    fn transpose<D: TranspositionDriver<V, Operation<V>>>(
        &self,
        context: &mut TranspositionContext<V, Operation<V>>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, Operation<V>>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, Operation<V>>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        check_count!("output", outputs, driver.region(0)?.output_types().len(), ProgramError);
        check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
        let contributions =
            (|| -> Result<Vec<MaybeZero<Tracer<TracingContext<V, Operation<V>>>>>, DifferentiationError> {
                let cotangents = context.cotangent_destinations(driver, inputs, accumulators)?;
                transpose_primal_jit_call(self, context, driver, inputs, outputs, &cotangents)
                    .map_err(DifferentiationError::from)
            })()?;
        check_count!("input", contributions, accumulators.len(), ProgramError);
        for (accumulator, contribution) in accumulators.iter().zip(contributions) {
            accumulator.accumulate(context, contribution)?;
        }
        Ok(())
    }
}

// Eager interpretation of a staged `jit_call` operation over concrete [`Array`](crate::Array)s. Top-level flat programs
// containing `jit_call` instructions (e.g., the pullbacks produced by eager `vjp` and `grad`) replay through
// [`Program::interpret_in_context`](ryft_core::Program::interpret_in_context), whose bind channel hands the callee
// region to [`Context::bind`] so the call is compiled whole through this session's dispatch cache and executed on its
// PJRT client. This mirrors JAX dispatching a `jit`-compiled function called from eager code straight to the compiled
// executable. This rule covers the remaining path, a `jit_call` nested inside another region being interpreted, by
// entering the active interpreter again on the callee region, which dispatches the callee's operations one by one
// through this same session. Both entry points prepare the call arguments before running the callee. Region replay
// also applies output overrides and the call's commitment to the results of the individually dispatched operations.
impl InterpretableOperation<Session> for JitCallOperation<Type> {
    fn interpret<D: InterpretationDriver<Session>>(
        &self,
        context: &Session,
        driver: &D,
        inputs: &[Value],
    ) -> Result<Vec<Value>, ProgramError> {
        let callee = driver.region(0)?;
        let renaming = self.output_renaming(&callee.input_types(), &callee.output_types())?;
        let invalid_argument = |error: Error| ProgramError::InvalidArgument { message: error.to_string() };
        let locally_assigned = self.results_are_locally_assigned(context, inputs);
        let mesh = self.execution_mesh(context, inputs).map_err(invalid_argument)?;
        let inputs = self
            .eager_inputs(context, inputs, callee.interface().input_types())
            .map_err(invalid_argument)?
            .into_owned();
        // The results are committed exactly as an execution of the call commits them, which is when a public argument
        // is committed or when the compiled program commits its public outputs (refer to `commits_public_outputs`),
        // and not as the replayed operations commit their own results on the pinned session below.
        let requests_placement = self.mesh.is_some() || has_placement_requests_in_region(callee, true);
        let committed = commits_public_outputs(&mesh, requests_placement)
            || inputs.iter().any(|input| matches!(input, Value::Array(array) if array.is_committed()));
        // The callee runs on the devices on which an eager execution of the call runs (refer to `execution_mesh`), so
        // the operations of the callee that take no array input (e.g., a `fill`) are replayed there rather than on the
        // default mesh of the session, as an execution of the callee places them.
        let session = context.with_device_mesh(mesh);
        driver
            .interpret_region(&session, 0, inputs)?
            .iter()
            .enumerate()
            .map(|(index, value)| match value.rename_type_identities(&renaming)? {
                Value::Array(array) => {
                    let array = self.boundary_output(context, index, array).map_err(invalid_argument)?;
                    Ok(Value::Array(array.with_committed(committed).with_locally_assigned(locally_assigned)))
                }
                value => Ok(value),
            })
            .collect()
    }
}

// A call of a compiled function places the values that cross the boundary of its callee as the effective boundary
// shardings of that boundary request (i.e., its sharding overrides, refer to `JitCallOperation::with_boundary`, and the
// placements of the inputs without one, refer to `JitCallOperation::boundary_input_sharding`), as an execution
// of the function places its arguments and results. The overrides are applied at the call site, as constraints on the
// operands and on the results of the call, rather than as annotations of the callee, because the callee may be inlined
// or shared with calls that carry other overrides. Each constraint is the sharding with which the function annotates
// its own boundary (refer to `entry_boundary_sharding`), and it is closed over auto mesh axes, so that a replicated
// override stops the compiler from splitting the value, like the replicated result of an execution. An effective input
// sharding without an override that the operand type already has is not constrained again, unless the mesh has auto
// axes, along which the operand type cannot show where the operand is (refer to
// `JitCallOperation::boundary_input_placement`).
impl LowerableOperation for JitCallOperation<Type> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Type],
        output_types: &[Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        self.validate_region_count(lowerer.regions().len())?;
        let input_values = input_values
            .iter()
            .zip(input_types)
            .enumerate()
            .map(|(index, (&value, ty))| {
                let Type::Array(actual) = ty else { return Ok(value) };
                let callee = lowerer.regions()[0].entry_region_ref();
                let declared = &callee.interface().input_types()[index];
                match self.boundary_input_placement(index, actual, declared).map(Cow::into_owned) {
                    Some(sharding) => lower_boundary_constraint(lowerer, value, ty, &sharding),
                    None => Ok(value),
                }
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let output_values = {
            let (mut block, context, location) = (lowerer.block(), lowerer.context(), lowerer.location());
            let (effect_tokens, regions, captured_values, nested_functions, collective_state) =
                lowerer.split_effect_tokens();
            let [callee] = regions else { unreachable!("validated jit_call region count") };

            // An operand whose type refines the callee input type (e.g., with a static extent for a bounded dynamic
            // one) enters the callee at the type of the callee input, and a result leaves it at the refined type of
            // the call (refer to `validate_call_input_types`). Neither conversion checks anything, because the call
            // has accepted its operands and the callee computes its results from them. Only shapes are converted. A
            // difference in sharding alone (i.e., in placement, or in manual variation that a call in a `shard_map`
            // body passes on to its results) lowers to nothing, because the call accepts only operands whose values
            // the callee input types describe.
            let callee_input_types = callee.input_types();
            let input_values = input_values
                .iter()
                .zip(input_types.iter().zip(callee_input_types.iter()))
                .map(|(&value, (r#type, declared))| match r#type == declared {
                    true => Ok(value),
                    false => lower_cast_shape(value, r#type, declared, &mut block, context, location),
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let output_values = lower_jit_call(
                callee,
                self.capture_count(),
                &input_values,
                captured_values,
                &mut block,
                context,
                location,
                nested_functions,
                collective_state,
                effect_tokens,
            )?;
            output_values
                .into_iter()
                .zip(callee.output_types().iter().zip(output_types))
                .map(|(value, (declared, r#type))| match declared == r#type {
                    true => Ok(value),
                    false => lower_cast_shape(value, declared, r#type, &mut block, context, location),
                })
                .collect::<Result<Vec<_>, Error>>()?
        };
        output_values
            .into_iter()
            .zip(output_types)
            .enumerate()
            .map(|(index, (value, r#type))| match self.output_sharding(index) {
                Some(sharding) => lower_boundary_constraint(lowerer, value, r#type, sharding),
                None => Ok(value),
            })
            .collect()
    }
}

/// Semantic identity and boundary of the named composite calls that one shared function implements.
#[derive(Clone, PartialEq, Eq, Hash)]
struct NamedFunctionKey {
    /// Semantic identity of the calls.
    key: CompositeKey,

    /// Input types of the calls.
    input_types: Vec<Type>,

    /// Output types of the calls.
    output_types: Vec<Type>,
}

/// Named composite calls found by [`count_jit_calls`], grouped by the interned body region, the key, and the boundary
/// that they share.
#[derive(Default)]
struct NamedRegionCalls {
    /// Positions in [`Self::calls`] of the groups of each body region, one per key and boundary that call it.
    index: HashMap<RegionId, Vec<usize>>,

    /// Body regions in first-occurrence order.
    calls: Vec<NamedRegionCall>,
}

/// Named composite calls that share one interned body region, one key, and one boundary.
struct NamedRegionCall {
    /// Body region of the calls.
    region: RegionId,

    /// Semantic identity of the calls.
    composite: CompositeKey,

    /// Input types of the calls.
    input_types: Vec<Type>,

    /// Output types of the calls.
    output_types: Vec<Type>,

    /// Number of calls.
    count: usize,
}

/// Formal callee input types and caller result types that bind [`TypeIdentity`](ryft_core::TypeIdentity)s across one
/// sealed call boundary. The bindings are encoded in these two type signatures, and the stored metadata contains no
/// runtime values.
///
/// An output identity present in the callee inputs is forwarded from the corresponding caller input. An identity
/// defined inside the callee receives one fresh caller identity shared by every corresponding result occurrence.
/// Separate calls receive independent local identities, because their results can have different extents even when
/// their input shapes agree. [`Self::fresh`] allocates those identities once during call construction, before
/// interpretation, so repeated type inference returns the same identities. Calls whose results only forward input
/// identities retain no instance of this metadata.
///
/// The formal input types allow [`Self::instantiated`] to specialize forwarded dimensions to actual inputs without
/// replacing the call's local result identities. When a transformation already owns the caller result names,
/// [`Self::with_outputs`] preserves them and validates their correspondence to the callee's output types.
/// [`Self::renaming`] recovers that correspondence from the current callee output types to the stored caller types.
///
/// For example, suppose a callee accepts a dimension with identity `n` and returns three dimensions with identities
/// `n`, `m`, and `m`, where `m` is defined inside the callee. The first result forwards `n`, and the other two share
/// one fresh caller identity. Another call receives a different identity for `m`, even though its diagnostic name can
/// remain `m`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CallTypeBindings<T: CoreType> {
    /// Formal input namespace used when instantiating forwarded result identities.
    inputs: Arc<[T]>,

    /// Caller result types, with callee-local identities already freshly bound.
    outputs: Arc<[T]>,
}

impl<T: CoreType> CallTypeBindings<T> {
    /// Freshens callee-local result identities, returning no metadata when all result identities are forwarded.
    ///
    /// # Errors
    ///
    /// Returns a type error when a result type cannot preserve its structure under identity renaming.
    fn fresh(inputs: &[T], outputs: &[T]) -> Result<Option<Self>, TypeError> {
        if !outputs
            .iter()
            .flat_map(CoreType::identities)
            .any(|(_, identity)| !inputs.iter().flat_map(CoreType::identities).any(|(_, input)| input == identity))
        {
            return Ok(None);
        }
        let mut unavailable = inputs
            .iter()
            .chain(outputs)
            .flat_map(CoreType::identities)
            .map(|(_, identity)| identity.clone())
            .collect::<Vec<_>>();
        let mut renaming = TypeIdentityRenaming::new();
        for (_, identity) in outputs.iter().flat_map(CoreType::identities) {
            if renaming.target(identity).is_none()
                && !inputs.iter().flat_map(CoreType::identities).any(|(_, input)| input == identity)
            {
                unavailable.push(renaming.insert_fresh(identity.clone(), &unavailable)?);
            }
        }
        let outputs =
            outputs.iter().map(|r#type| r#type.rename_identities(&renaming)).collect::<Result<Vec<_>, _>>()?;
        Ok(Some(Self { inputs: inputs.into(), outputs: outputs.into() }))
    }

    /// Records caller result types whose local identities have already been chosen, validating their exact
    /// correspondence to the callee's output types. Unlike [`Self::fresh`], this function allocates no identities.
    /// Forwarded identities must stay unchanged, and callee-local identities must have fresh caller counterparts.
    /// Returns no metadata when both output signatures agree and only forward input identities.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: Callee input types. Both output signatures use this input scope for forwarded identities.
    ///   - `callee_outputs`: Output types of the attached callee, using its local result identities.
    ///   - `outputs`: Result types retained by the caller, using the existing caller-owned result identities.
    ///
    /// # Errors
    ///
    /// Returns a type error for changed result structure, captured local names, or renamed forwarded identities.
    fn with_outputs(inputs: &[T], callee_outputs: &[T], outputs: Vec<T>) -> Result<Option<Self>, TypeError> {
        if callee_outputs == outputs {
            if callee_outputs
                .iter()
                .flat_map(CoreType::identities)
                .any(|(_, identity)| !inputs.iter().flat_map(CoreType::identities).any(|(_, input)| input == identity))
            {
                return Err(TypeError::invalid("call local result declarations must use fresh identities"));
            }
            return Ok(None);
        }
        let signature = Self { inputs: inputs.into(), outputs: outputs.into() };
        signature.renaming(inputs, callee_outputs)?;
        Ok(Some(signature))
    }

    /// Returns the formal result types retained by the caller.
    fn outputs(&self) -> &[T] {
        &self.outputs
    }

    /// Returns the caller-owned identities defined by this call, once each in result occurrence order.
    fn definitions(&self) -> Vec<T::Identity> {
        let mut definitions = Vec::new();
        for (_, identity) in self.outputs.iter().flat_map(CoreType::identities) {
            if !self.inputs.iter().flat_map(CoreType::identities).any(|(_, input)| input == identity)
                && !definitions.contains(identity)
            {
                definitions.push(identity.clone());
            }
        }
        definitions
    }

    /// Returns caller result types with forwarded names and ordinary shape refinements instantiated at `inputs`.
    /// Local result identities remain symbolic even when the input facts determine an exact extent.
    ///
    /// # Errors
    ///
    /// Returns the input-scope correspondence or type-renaming error when the call signature is incompatible.
    fn instantiated(&self, inputs: &[T]) -> Result<Vec<T>, TypeError> {
        if self.inputs.as_ref() == inputs {
            return Ok(self.outputs.to_vec());
        }
        let renaming = T::derive_identity_renaming(self.inputs.as_ref(), inputs)?;
        let refinements = T::Refinements::from_local_declarations(self.inputs.iter(), inputs)?;
        let local = self.definitions();
        self.outputs
            .iter()
            .map(|r#type| refinements.refine_declared(r#type, &local)?.rename_identities(&renaming))
            .collect()
    }

    /// Returns the validated renaming from the current callee output types to this call's instantiated result types.
    /// The source types are supplied in `callee_outputs`, and the target types come from [`Self::instantiated`]. For
    /// the type-level example, this maps the callee's local `m` to the caller's fresh identity while preserving
    /// forwarded `n`. Caller result types alone cannot recover the callee's original local identities.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: Input types of the current callee instantiation, used to instantiate forwarded result identities.
    ///   - `callee_outputs`: Output types of that callee instantiation, before renaming its local identities into the
    ///     caller's scope.
    ///
    /// # Errors
    ///
    /// Returns a type error when local names are not fresh, forwarded names change, or result structure differs.
    fn renaming(&self, inputs: &[T], callee_outputs: &[T]) -> Result<TypeIdentityRenaming<T::Identity>, TypeError> {
        let outputs = self.instantiated(inputs)?;
        let renaming = T::derive_identity_renaming(callee_outputs, &outputs)?;
        for (_, identity) in callee_outputs.iter().flat_map(CoreType::identities) {
            let target = renaming.target(identity).unwrap_or(identity);
            let forwarded = inputs.iter().flat_map(CoreType::identities).any(|(_, input)| input == identity);
            if forwarded && target != identity {
                return Err(TypeError::invalid("call result declarations must preserve forwarded input identities"));
            }
            if !forwarded
                && (target == identity
                    || inputs
                        .iter()
                        .chain(callee_outputs)
                        .flat_map(CoreType::identities)
                        .any(|(_, original)| original == target))
            {
                return Err(TypeError::invalid("call local result declarations must use fresh identities"));
            }
        }
        Ok(renaming)
    }

    /// Renames caller-owned definitions while retaining the callee's formal input namespace.
    ///
    /// # Errors
    ///
    /// Returns an identity conflict or type-renaming error.
    fn renamed(&self, renaming: &TypeIdentityRenaming<T::Identity>) -> Result<Self, TypeError> {
        let mut owned = TypeIdentityRenaming::new();
        for (source, target) in renaming.replacements() {
            if !self.inputs.iter().flat_map(CoreType::identities).any(|(_, input)| input == source) {
                owned.insert(source.clone(), target.clone())?;
            }
        }
        let outputs =
            self.outputs.iter().map(|r#type| r#type.rename_identities(&owned)).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { inputs: self.inputs.clone(), outputs: outputs.into() })
    }

    /// Retains the indicated input and result positions after ordinary call-boundary pruning.
    ///
    /// # Errors
    ///
    /// Returns a type error when a positional mask has the wrong arity.
    fn pruned(&self, inputs: &[bool], outputs: &[bool]) -> Result<Self, TypeError> {
        if inputs.len() != self.inputs.len() || outputs.len() != self.outputs.len() {
            return Err(TypeError::invalid("call result declaration pruning masks have incompatible arity"));
        }
        Ok(Self {
            inputs: self
                .inputs
                .iter()
                .zip(inputs)
                .filter_map(|(r#type, kept)| kept.then_some(r#type.clone()))
                .collect(),
            outputs: self
                .outputs
                .iter()
                .zip(outputs)
                .filter_map(|(r#type, kept)| kept.then_some(r#type.clone()))
                .collect(),
        })
    }
}

/// [`Region`](CoreRegion) [`Transform`] marker for the shared copy of a `jit_call` callee that
/// [`shared_jit_call_callee`] retains for a callee region that is not interned as a shared region (e.g., the entry
/// region of a shared callee program that a rule binds directly, such as the primal or tangent callee of a linearized
/// call). Binding that copy instead of a fresh copy of the region lets every call of one callee share one region.
struct JitCallCalleeTransform;

impl<V: CoreValue<Type = Type>, O: CoreOperation<Type = Type>> Transform<CoreRegion<V, O>> for JitCallCalleeTransform {
    type Arguments = ();
    type Artifact = TransformArtifact<V, O, ()>;

    const DEFAULT_CACHE_CAPACITY: usize = 1;
}

/// [`Region`](CoreRegion) [`Transform`] marker for the splits of a `jit_call` callee that [`split_jit_call`] binds.
/// Such a split partitions the callee in fresh traces, where every input is a fresh known or unknown input, so it
/// depends on no value of a live context and is a structural function of the callee and of its
/// [`JitCallSplitArguments`]. Every call of one callee with the same arguments therefore shares one split, as JAX's
/// call partial evaluation rules share one split per callee, knownness, and policy through their caches.
struct JitCallSplitTransform;

impl<V: CoreValue<Type = Type>, O: CoreOperation<Type = Type>> Transform<CoreRegion<V, O>> for JitCallSplitTransform {
    type Arguments = JitCallSplitArguments;
    type Artifact = TransformArtifact<V, O, Arc<JitCallSplit>>;

    const DEFAULT_CACHE_CAPACITY: usize = 8;
}

/// Kind of a retained [`JitCallSplitTransform`] split.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum JitCallSplitKind {
    /// Split of a call whose inputs are partially known, marked in input order (refer to
    /// [`PartialEvaluationDriver::partition_program`]).
    Knownness(Vec<bool>),

    /// Split of an all-known call on residual demand for the outputs of the callee that the residual planner demands,
    /// marked in output order (refer to [`PartialEvaluationContext::partition_residual_demand`]).
    ResidualDemand(Vec<bool>),
}

/// Argument key for one retained [`JitCallSplitTransform`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct JitCallSplitArguments {
    /// Kind of the split.
    kind: JitCallSplitKind,

    /// Residual policy that places the residuals of the split and of the splits nested within it, compared by
    /// definition identity, or [`None`] when partial evaluation has no residual policy.
    policy: Option<ResidualPolicyReference<Type>>,

    /// Type of the rounding with which the placement of the residuals of the split also rounds the values that the
    /// known callee saves (refer to [`PartialEvaluationContext::residual_rounding_type`]), which stages rounding
    /// operations in the known callee, or [`None`] when the placement keeps those values unrounded.
    residual_rounding_type: Option<TypeId>,

    /// Whether effectful operations with known inputs may fold into the known callee (refer to
    /// [`PartialEvaluationContext::allow_effect_folding`]).
    allow_effect_folding: bool,

    /// Whether operations whose results must have storage of their own for every execution may fold into the known
    /// callee (refer to [`PartialEvaluationContext::allow_copy_folding`]), which program specialization disallows.
    allow_copy_folding: bool,

    /// Whether recursive partitioning analyzes repeated residual invocations, which changes effect placement (refer
    /// to [`PartialEvaluationDriver::partitions_for_repeated_residual_calls`]). Pure and residual-demand splits use
    /// `false` because their partitions do not depend on this mode.
    repeated_residual: bool,
}

/// Boundary wiring of a retained [`JitCallSplitTransform`] artifact, whose programs are the known callee and the
/// residual callee, in that order. The known callee returns the known outputs of the original callee followed by the
/// residual edges that the residual callee reads from it, and the residual callee returns the remaining outputs (i.e.,
/// the demanded outputs for a split on residual demand). Refer to [`PartitionedProgram`]
/// for more information on this layout.
#[derive(Clone, Debug, PartialEq)]
struct JitCallSplit {
    /// Index of the call input feeding each input of the known callee, in order.
    known_input_indices: Vec<usize>,

    /// Source of each input of the residual callee, which is a call input, an output of the known callee, or a
    /// residual edge.
    residual_inputs: Vec<ResidualInputSource>,

    /// Source of each output of the call, in output order, which is a known output of the known callee or an output of
    /// the residual callee.
    outputs: Vec<PartialEvaluationOutput<usize>>,
}

/// Returns whether `program` may be deduplicated by structural identity.
///
/// A program is eligible only when the canonical rendering captures the full semantics needed to safely share a private
/// function. Although immediate constant payloads are rendered, a [`CaptureReference`](ryft_core::CaptureReference)
/// identifies a slot and type rather than the runtime value in the surrounding capture table, so programs with
/// constants remain ineligible. `shard_map` also remains outside this deduplication path because its lowering owns a
/// separate outlined boundary.
fn supports_structural_deduplication<O: LowerableOperationFamily>(program: &FlatProgram<O>) -> bool {
    program.regions().iter().all(|region| {
        region.atoms().iter().all(|atom| !atom.is_constant())
            && region
                .instructions()
                .iter()
                .all(|instruction| instruction.operation().projected_payload::<ShardMapOperation>().is_none())
    })
}

/// Returns whether a borrowed rooted region may be deduplicated by structural identity.
fn supports_structural_deduplication_of_region<O: LowerableOperationFamily>(
    region: RegionRef<'_, Constant, O>,
) -> bool {
    fn walk<O: LowerableOperationFamily>(region: RegionRef<'_, Constant, O>, visited: &mut HashSet<RegionId>) -> bool {
        if !visited.insert(region.id()) {
            return true;
        }
        region.atoms().iter().all(|atom| !atom.is_constant())
            && region.instructions().iter().all(|instruction| {
                instruction.operation().projected_payload::<ShardMapOperation>().is_none()
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
/// structural deduplication (refer to [`supports_structural_deduplication`]) and therefore always inline. The key is
/// generic over the program's constant universe so that lookups from the value-generic lowering path type-check,
/// although the map itself is only ever populated from the staged [`Constant`]-keyed pipeline.
fn jit_call_program_key<O: LowerableOperationFamily>(program: &FlatProgram<O>) -> Option<JitCallProgramKey> {
    supports_structural_deduplication(program).then(|| structural_key(program))
}

/// Computes the deduplication key for a borrowed callee region, materializing only after borrowed eligibility checks.
fn jit_call_region_key<O: LowerableOperationFamily>(region: RegionRef<'_, Constant, O>) -> Option<JitCallProgramKey> {
    supports_structural_deduplication_of_region(region).then(|| structural_key(&region.to_program()))
}

/// Returns the deduplication key of `program`, an eligible callee (refer to [`JitCallProgramKey`]).
fn structural_key<O: LowerableOperationFamily>(program: &FlatProgram<O>) -> JitCallProgramKey {
    fn walk<O: LowerableOperationFamily>(
        region: RegionRef<'_, Constant, O>,
        visited: &mut HashSet<RegionId>,
        calls: &mut Vec<JitCallOperation<Type>>,
    ) {
        if !visited.insert(region.id()) {
            return;
        }
        for instruction in region.instructions() {
            if let Some(call) = instruction.operation().projected_payload::<JitCallOperation<Type>>() {
                calls.push(call.clone());
            }
            for &nested in instruction.regions() {
                if let Ok(nested) = RegionRef::new(region.arena(), nested) {
                    walk(nested, visited, calls);
                }
            }
        }
    }

    let mut calls = Vec::new();
    walk(program.entry_region_ref(), &mut HashSet::new(), &mut calls);
    JitCallProgramKey {
        rendered: program.to_string(),
        input_types: program.input_types().into_owned(),
        output_types: program.output_types().into_owned(),
        calls,
    }
}

/// Counts `jit_call` callee occurrences in `program`, covering nested computations at every depth. Attached regions
/// (i.e., control-flow bodies, composite programs, and callee bodies) all live in the program's one canonical region
/// arena, so the walk descends region edges without materializing nested programs. `shard_map` body regions are
/// intentionally skipped, because their `jit_call`s lower with shard-local types and always inline.
///
/// The bodies of named composite calls count like callees (refer to
/// [`NamedComposite`](ryft_core::NamedComposite)), unless a backend implementation replaces the call on
/// `target_platform`, so that each specialization lowers to one shared function. Their calls share one interned body
/// region per specialization, so each body is rendered once. `named` records the semantic identity and boundary
/// of the calls that each counted body implements, and `names` the name of their definition.
///
/// `counts` accumulates the occurrence count and a representative materialized program per identity, `order` records
/// keys in first-occurrence order, and `memo` caches the (possibly expensive) key computation per callee root
/// [`RegionId`]. Interned shared callees repeat their root region id, so each shared callee is rendered once.
fn count_jit_calls<Input, Output, O: LowerableOperationFamily>(
    program: &Program<Input, Output, O>,
    target_platform: Option<&str>,
    counts: &mut HashMap<JitCallProgramKey, (usize, FlatProgram<O>)>,
    order: &mut Vec<JitCallProgramKey>,
    memo: &mut HashMap<RegionId, Option<JitCallProgramKey>>,
    named: &mut HashMap<NamedFunctionKey, Option<JitCallProgramKey>>,
    names: &mut HashMap<JitCallProgramKey, String>,
) -> Result<(), Error>
where
    Input: Parameterized<Constant>,
    Output: Parameterized<Constant>,
{
    #[allow(clippy::too_many_arguments)]
    fn walk<Input, Output, O: LowerableOperationFamily>(
        program: &Program<Input, Output, O>,
        region: RegionId,
        target_platform: Option<&str>,
        counts: &mut HashMap<JitCallProgramKey, (usize, FlatProgram<O>)>,
        order: &mut Vec<JitCallProgramKey>,
        memo: &mut HashMap<RegionId, Option<JitCallProgramKey>>,
        named_calls: &mut NamedRegionCalls,
        visited: &mut HashSet<RegionId>,
    ) -> Result<(), Error>
    where
        Input: Parameterized<Constant>,
        Output: Parameterized<Constant>,
    {
        // Interned shared callees repeat their root region id. Each region's instructions are walked once while
        // every referencing `jit_call` instruction still contributes one occurrence to the counts below.
        if !visited.insert(region) {
            return Ok(());
        }
        let region_ref = program.region_ref(region).unwrap();
        for instruction in region_ref.instructions() {
            if instruction.operation().projected_payload::<ShardMapOperation>().is_some() {
                continue;
            }
            if let Some(key) = instruction.operation().composite_key() {
                let types = |atoms: &[AtomId]| {
                    atoms.iter().map(|atom| region_ref.atoms()[atom.index()].r#type().into_owned()).collect::<Vec<_>>()
                };
                if NamedCompositeImplementation::select(
                    key,
                    &types(instruction.inputs()),
                    &types(instruction.outputs()),
                    target_platform,
                )
                .is_some()
                {
                    continue;
                }
            }
            for (index, &nested) in instruction.regions().iter().enumerate() {
                if instruction.operation().region_role(index) == Some(RegionRole::Computation) {
                    walk(program, nested, target_platform, counts, order, memo, named_calls, visited)?;
                }
            }

            // Calls that share an interned body region, a key, and a boundary share a function. Calls of different
            // keys (e.g., of different versions) or boundaries are grouped separately because their definitions may
            // require different backend implementations or emitted signatures.
            if let Some(composite_key) = instruction.operation().composite_key() {
                let Some(&body_region) = instruction.regions().first() else {
                    continue;
                };
                let atom_type = |atom: &AtomId| region_ref.atoms()[atom.index()].r#type();
                let has_boundary = |call: &NamedRegionCall| {
                    call.input_types.len() == instruction.inputs().len()
                        && call.output_types.len() == instruction.outputs().len()
                        && instruction
                            .inputs()
                            .iter()
                            .zip(&call.input_types)
                            .all(|(atom, r#type)| *atom_type(atom) == *r#type)
                        && instruction
                            .outputs()
                            .iter()
                            .zip(&call.output_types)
                            .all(|(atom, r#type)| *atom_type(atom) == *r#type)
                };
                let grouped = named_calls.index.get(&body_region).and_then(|indices| {
                    indices.iter().copied().find(|index| {
                        let call = &named_calls.calls[*index];
                        &call.composite == composite_key && has_boundary(call)
                    })
                });
                if let Some(index) = grouped {
                    named_calls.calls[index].count += 1;
                    continue;
                }
                let input_types =
                    instruction.inputs().iter().map(|atom| atom_type(atom).into_owned()).collect::<Vec<_>>();
                let output_types =
                    instruction.outputs().iter().map(|atom| atom_type(atom).into_owned()).collect::<Vec<_>>();
                named_calls.index.entry(body_region).or_default().push(named_calls.calls.len());
                named_calls.calls.push(NamedRegionCall {
                    region: body_region,
                    composite: composite_key.clone(),
                    input_types,
                    output_types,
                    count: 1,
                });
                continue;
            }
            if instruction.operation().projected_payload::<JitCallOperation<Type>>().is_some() {
                let Some(&callee_region) = instruction.regions().first() else {
                    continue;
                };
                let key = memo
                    .entry(callee_region)
                    .or_insert_with(|| {
                        let callee = program.region_ref(callee_region).unwrap();
                        jit_call_region_key(callee)
                    })
                    .clone();
                let Some(key) = key else {
                    continue;
                };
                let entry = counts.entry(key.clone()).or_insert_with(|| {
                    order.push(key.clone());
                    (0, program.region_ref(callee_region).unwrap().to_program())
                });
                entry.0 += 1;
            }
        }
        Ok(())
    }
    let mut visited = HashSet::new();
    let mut named_calls = NamedRegionCalls::default();
    walk(program, program.entry(), target_platform, counts, order, memo, &mut named_calls, &mut visited)?;

    // Each named composite body region is rendered and counted once, with the number of calls that share it.
    for call in named_calls.calls {
        let named_key =
            NamedFunctionKey { key: call.composite, input_types: call.input_types, output_types: call.output_types };
        let key = memo
            .entry(call.region)
            .or_insert_with(|| jit_call_region_key(program.region_ref(call.region).unwrap()))
            .clone();
        // A body that structural deduplication cannot key (e.g., one with constants) has no shared function, and it
        // marks its key and boundary as conflicting, so that its calls never resolve to another body's function.
        let Some(key) = key else {
            named.insert(named_key, None);
            continue;
        };
        let entry = counts.entry(key.clone()).or_insert_with(|| {
            order.push(key.clone());
            (0, program.region_ref(call.region).unwrap().to_program())
        });
        entry.0 += call.count;
        names
            .entry(key.clone())
            .or_insert_with(|| format!("{}.v{}", named_key.key.name(), named_key.key.version()));

        // Calls with equal keys and boundaries but different bodies must not share an index entry, because lowering
        // would then call one body's function for the other's calls. Such a conflict is recorded as `None` and left
        // out of the index.
        match named.entry(named_key) {
            Entry::Vacant(entry) => {
                entry.insert(Some(key));
            }
            Entry::Occupied(mut entry) => {
                if entry.get().as_ref() != Some(&key) {
                    entry.insert(None);
                }
            }
        }
    }
    Ok(())
}

/// Returns `value`, a tangent or cotangent that a derived call of a callee returns at a type of the callee boundary, at
/// `refined`, the type that the value has in the caller (e.g., the tangent type of a primal output, or the cotangent
/// type of a primal input). A call accepts inputs whose types refine the callee input types (refer to
/// `validate_call_input_types`), so a primal that a partial evaluation computes from such inputs (e.g., an eager array,
/// whose type records its sharding) can refine the type of the callee boundary, while the derived call that computes
/// the tangent or the cotangent of that primal returns it at the type of the callee boundary. The value has the shape
/// and placement of the primal, so an unchecked cast in `context` states its refined type. A value that a call whose
/// callee names devices on `mesh` returns lives on those devices, so it is cast only to a type over a mesh of the same
/// topology. A cotangent of an input over the canonical mesh of one device (i.e., [`LogicalMesh::single_device`]),
/// which is where an uncommitted array lives, that a callee on a mesh of one device returns is instead placed onto the
/// canonical mesh over the device of that mesh, so that it combines with the other cotangents of that input, and lives
/// where JAX returns such a cotangent (i.e., on the devices of the call). Every other value is returned unchanged, because values over
/// different logical meshes do not combine, and a callee mesh of several devices has no such placement.
///
/// A value whose type differs from `refined` in its sharding only in dimensions that one of them leaves unconstrained
/// (i.e., `{?}`) and the other replicates (refer to [`replicates_unconstrained_dimensions`]) is cast to `refined` as
/// well (e.g., the cotangent of an input that a call takes through an unconstrained input override). This is an
/// unchecked cast rather than a `reshard`, because the value is already placed as `refined` states. An eager array
/// stores an unconstrained dimension replicated (refer to `entry_boundary_sharding` in `lowering/mod.rs`), so its
/// buffers are those of `refined`. Inside a compiled program, the cast emits no sharding constraint (an unconstrained
/// and a replicated dimension do not conflict over explicit mesh axes, and types record no placement over auto mesh
/// axes), and the entry boundary of the program replicates an unconstrained dimension of any value that it exchanges.
///
/// A value whose type tracks a sharding, where `refined` tracks none, is cast to `refined` as well when a call accepts
/// an input of type `refined` for a callee input of the type of the value (refer to [`is_untracked_refinement`]), as
/// the cotangent of an input that a staged caller of untracked type passes to a callee with an input sharding override
/// is. This mirrors the forward direction, where the call takes that input as it is (refer to `call_input_variation`)
/// and its lowering constrains the operand to the override, so the derivative returns to the type of the caller input.
/// The cast only stops tracking the placement of the value, which a type without a sharding states nothing about, and
/// keeps its values and buffers.
///
/// A value whose type differs from `refined` only in the placement that its sharding tracks over a mesh of the same
/// topology (e.g., the cotangent of a replicated argument of a callee that declares that input split, which the call
/// places at its effective boundary, refer to [`JitCallOperation::boundary_input_sharding`]) is resharded to `refined`,
/// so that it combines with the other cotangents of that argument, as JAX returns the cotangent of a `jit` argument
/// with the sharding of that argument.
fn refined_derived_value<C, V>(
    context: &C,
    mesh: Option<&DeviceMesh>,
    refined: Type,
    value: C::Value,
) -> Result<C::Value, ProgramError>
where
    C: Context<Type = Type, Operation = Operation<V>>,
    V: CoreValue<Type = Type> + ValueProjection<ArrayType, Projected: CoreValue<Type = ArrayType>>,
{
    let declared = value.r#type().into_owned();
    let moves_devices = match (mesh, &refined) {
        (Some(mesh), Type::Array(refined)) => {
            refined.sharding().is_some_and(|sharding| !sharding.mesh().has_same_topology(mesh.logical_mesh()))
        }
        _ => false,
    };
    if moves_devices
        && let (Some(mesh), Type::Array(refined)) = (mesh, &refined)
        && let [device] = mesh.devices()
        && refined.sharding().is_some_and(|sharding| sharding.mesh() == &LogicalMesh::single_device())
    {
        let canonical_mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![*device]).map_err(TypeError::from)?;
        let sharding = Sharding::replicated(canonical_mesh.logical_mesh().clone(), refined.rank());
        let target = Placement::new(canonical_mesh, sharding).map_err(TypeError::from)?.with_memory(refined.memory());
        return Ok(context.bind(PlaceOperation::new(target, None), Vec::new(), &[value])?.remove(0));
    }
    if let (Type::Array(declared), Type::Array(refined)) = (&declared, &refined)
        && let (Some(declared_sharding), Some(refined_sharding)) = (declared.sharding(), refined.sharding())
        && !moves_devices
        && declared_sharding != refined_sharding
        && !replicates_unconstrained_dimensions(declared_sharding, refined_sharding)
        && declared_sharding.mesh().has_same_topology(refined_sharding.mesh())
        && declared_sharding.unreduced_axes() == refined_sharding.unreduced_axes()
        && declared_sharding.reduced_axes() == refined_sharding.reduced_axes()
        && declared_sharding.varying_manual_axes() == refined_sharding.varying_manual_axes()
        && declared.clone().with_sharding(refined_sharding.clone()).is_ok_and(|declared| &declared == refined)
    {
        let reshard = ReshardOperation::new(refined_sharding.clone());
        return Ok(context.bind(reshard, Vec::new(), &[value])?.remove(0));
    }
    if moves_devices
        || refined.is_interchangeable_with(&declared)
        || !(declared.is_refined_by(&refined)
            || is_refined_up_to_unconstrained_dimensions(&declared, &refined)?
            || is_untracked_refinement(&declared, &refined))
    {
        return Ok(value);
    }
    let cast = CastOperation::unchecked(refined, vec![declared])?;
    Ok(context.bind(cast, Vec::new(), &[value])?.remove(0))
}

/// Returns whether `declared` is refined by `refined` once the sharding of `refined` is replaced by that of
/// `declared`, where both are array types whose shardings differ only in dimensions that one of them leaves
/// unconstrained and the other replicates (refer to [`replicates_unconstrained_dimensions`]).
fn is_refined_up_to_unconstrained_dimensions(declared: &Type, refined: &Type) -> Result<bool, TypeError> {
    let (Type::Array(declared), Type::Array(refined)) = (declared, refined) else {
        return Ok(false);
    };
    let (Some(declared_sharding), Some(refined_sharding)) = (declared.sharding(), refined.sharding()) else {
        return Ok(false);
    };
    if !replicates_unconstrained_dimensions(declared_sharding, refined_sharding) {
        return Ok(false);
    }
    Ok(declared.is_refined_by(&refined.clone().with_sharding(declared_sharding.clone())?))
}

/// Returns whether `refined`, an array type that tracks no sharding, is a type that a call accepts for a callee input
/// of type `declared` without varying along more manual mesh axes (refer to `call_input_variation`), where `declared`
/// tracks a sharding. A call accepts it exactly when `declared` has the shape that `refined` refines and a sharding
/// without unreduced or reduced axes and without manual variation, so that a value of type `declared` is a value of
/// type `refined` whose placement is no longer tracked. A sharding that partitions a dimension along manual mesh axes
/// is not accepted, because a cast cannot change placement over manual mesh axes, which inside a manual region also
/// determines the shape of the local value.
fn is_untracked_refinement(declared: &Type, refined: &Type) -> bool {
    let (Type::Array(declared), Type::Array(refined)) = (declared, refined) else {
        return false;
    };
    refined.sharding().is_none()
        && declared.sharding().is_some_and(|sharding| {
            sharding.dimensions().iter().all(|dimension| dimension.manual_axes(sharding.mesh()).is_empty())
        })
        && call_input_variation(declared, refined).is_some()
}

/// Returns whether the distinct shardings `left` and `right` place an array alike, because they have the same logical
/// mesh, rank, reduction state, and manual variation, and each of their dimensions is either equal in both or
/// unconstrained (i.e., `{?}`) in one and replicated in the other. A runtime array stores an unconstrained dimension
/// replicated, so arrays placed as either sharding have the same buffers.
fn replicates_unconstrained_dimensions(left: &Sharding, right: &Sharding) -> bool {
    left != right
        && left.rank() == right.rank()
        && left.dimensions().iter().zip(right.dimensions()).all(|dimensions| match dimensions {
            (ShardingDimension::Unconstrained, ShardingDimension::Replicated)
            | (ShardingDimension::Replicated, ShardingDimension::Unconstrained) => true,
            (left, right) => left == right,
        })
        && left.with_dimensions(right.dimensions().to_vec()).is_ok_and(|left| &left == right)
}

/// Returns `cotangent`, the cotangent that the transposition of a call returns for an input whose cotangent type is
/// `refined`, summed over the manual mesh axes along which it varies beyond `refined`. A call whose inputs vary along
/// manual mesh axes beyond the callee input types runs its callee as if every input varied along all of them (refer
/// to `CallVariation`), as a JAX `jit` function is retraced with `pvary` inserted on the inputs that vary along fewer
/// axes. Each shard along such an axis then holds one contribution to the cotangent of such an input, and their sum
/// is its cotangent, which a mesh-form `parallel_reduce` computes as the transposition of `parallel_vary` does. Every
/// other cotangent, including every cotangent of a call without that variation, is returned unchanged.
fn summed_input_cotangent<C, V>(context: &C, refined: &Type, cotangent: C::Value) -> Result<C::Value, ProgramError>
where
    C: Context<Type = Type, Operation = Operation<V>>,
    V: CoreValue<Type = Type> + ValueProjection<ArrayType, Projected: CoreValue<Type = ArrayType>>,
{
    let (mesh, axes) = match (refined, cotangent.r#type().as_ref()) {
        (Type::Array(refined), Type::Array(actual)) if let Some(sharding) = actual.sharding() => {
            let declared = refined.sharding().map(Sharding::varying_manual_axes);
            let axes = sharding
                .varying_manual_axes()
                .iter()
                .filter(|axis| !declared.is_some_and(|declared| declared.contains(*axis)))
                .cloned()
                .collect::<Vec<_>>();
            if axes.is_empty() {
                return Ok(cotangent);
            }
            (sharding.mesh().clone(), axes)
        }
        _ => return Ok(cotangent),
    };
    axes.into_iter().try_fold(cotangent, |cotangent, axis| {
        let reduction = ParallelReduceOperation::new(ReductionKind::Sum, axis).with_mesh(mesh.clone());
        Ok(context.bind(reduction, Vec::new(), &[cotangent])?.remove(0))
    })
}

/// Returns `sharding`, the boundary sharding of each item of a value that batching maps along `axis`, with the batch
/// dimension inserted at that axis, `sharding` itself for an unmapped value, and [`None`] without a sharding or when
/// the batch dimension cannot be inserted. The per-item override states nothing about the batch dimension, because an
/// explicit placement under batching applies to each batch item, so the batch dimension keeps the placement that
/// `packed`, the type of the batched value, records for it (e.g., split along a mesh axis by the batch specification),
/// and is replicated when that type records none (e.g., over auto mesh axes, which types never record).
fn batched_boundary_sharding(
    sharding: Option<&Sharding>,
    axis: &BatchAxis,
    packed: Option<&ArrayType>,
) -> Option<Sharding> {
    let sharding = sharding?;
    let Some(axis) = axis.axis() else {
        return Some(sharding.clone());
    };
    let rank = sharding.dimensions().len() as i128 + 1;
    let position = if axis.value() < 0 { axis.value() + rank } else { axis.value() };
    let position = usize::try_from(position).ok().filter(|&position| (position as i128) < rank)?;
    let batch_dimension = packed
        .and_then(ArrayType::sharding)
        .filter(|packed| packed.rank() as i128 == rank && packed.mesh().has_same_topology(sharding.mesh()))
        .map_or(ShardingDimension::Replicated, |packed| packed.dimensions()[position].clone());
    let mut dimensions = sharding.dimensions().to_vec();
    dimensions.insert(position, batch_dimension);
    sharding.with_dimensions(dimensions).ok()
}

/// Manual mesh axes along which inputs of a call vary beyond what the callee input types declare (e.g., a value of a
/// `shard_map` body that varies along the manual axes of the map, passed to a function compiled for a plain type),
/// along which every output of the call then varies too (refer to `validate_call_input_types`). The callee runs as if
/// every input varied along all of them, so the transposition of the call sums the cotangent of an input that varies
/// along fewer of them over the others (refer to `summed_input_cotangent`).
struct CallVariation {
    /// Logical mesh of the manual axes, which is the mesh of the sharding of the varying inputs.
    mesh: LogicalMesh,

    /// Manual mesh axes along which the inputs vary beyond their callee input types.
    axes: BTreeSet<String>,
}

impl CallVariation {
    /// Returns `output_types`, the output types of a call named `operation_name`, each varying along the axes of this
    /// variation as well. An output without a sharding takes a replicated sharding over the mesh of this variation,
    /// and an output with a sharding over another mesh is rejected.
    fn apply<T: ArrayMemberType + From<ArrayType>>(
        &self,
        operation_name: &'static str,
        output_types: Vec<T>,
    ) -> Result<Vec<T>, TypeError> {
        output_types
            .into_iter()
            .enumerate()
            .map(|(index, r#type)| {
                let Ok(array_type) = r#type.as_array_type() else {
                    return Ok(r#type);
                };
                let mut sharding = match array_type.sharding() {
                    None => Sharding::replicated(self.mesh.clone(), array_type.rank()),
                    Some(sharding) if sharding.mesh() == &self.mesh => sharding.clone(),
                    Some(_) => {
                        return Err(TypeError::invalid(format!(
                            "`{operation_name}` output #{index} has type {array_type}, which cannot vary along the \
                             manual mesh axes {:?} of mesh `{}` along which its inputs vary",
                            self.axes, self.mesh,
                        )));
                    }
                };
                sharding.extend_varying_manual_axes(self.axes.iter().cloned())?;
                Ok(T::from(array_type.clone().with_sharding(sharding)?))
            })
            .collect()
    }
}

/// Returns the manual mesh axes along which an input of type `actual` varies beyond `declared`, the type of the callee
/// input that receives it, when a call accepts that input, or [`None`] when it rejects it. The type of the input must
/// refine the declared type up to the placement that their shardings describe, which a call may change because it
/// does not change the values. What a sharding states about the values must agree, because a call, unlike an
/// execution (refer to `validate_input_type`), does not reshard its inputs. These are the unreduced and reduced axes,
/// with the kind of reduction, where a declared type without a sharding describes a fully reduced value. The input may
/// vary along more manual mesh axes than the declared type, which a value of a `shard_map` body does where the callee
/// declares none, as a JAX `jit` function called on such a value is retraced at its varying type, and the outputs of
/// the call then vary along those axes too (refer to `CallVariation`).
fn call_input_variation(declared: &ArrayType, actual: &ArrayType) -> Option<BTreeSet<String>> {
    fn components(
        sharding: Option<&Sharding>,
    ) -> (&BTreeSet<String>, Option<UnreducedKind>, &BTreeSet<String>, &BTreeSet<String>) {
        let no_axes = const { &BTreeSet::new() };
        match sharding {
            Some(sharding) => (
                sharding.unreduced_axes(),
                (!sharding.unreduced_axes().is_empty()).then(|| sharding.unreduced_kind()),
                sharding.reduced_axes(),
                sharding.varying_manual_axes(),
            ),
            None => (no_axes, None, no_axes, no_axes),
        }
    }
    let (declared_unreduced, declared_kind, declared_reduced, declared_variation) = components(declared.sharding());
    let (actual_unreduced, actual_kind, actual_reduced, actual_variation) = components(actual.sharding());
    if (declared_unreduced, declared_kind, declared_reduced) != (actual_unreduced, actual_kind, actual_reduced)
        || !declared_variation.is_subset(actual_variation)
    {
        return None;
    }
    let declared_without_sharding = declared.clone().with_sharding(None).ok()?;
    let actual_without_sharding = actual.clone().with_sharding(None).ok()?;
    declared_without_sharding
        .is_refined_by(&actual_without_sharding)
        .then(|| actual_variation.difference(declared_variation).cloned().collect())
}

/// Validates that a call accepts `input_types` as the inputs of a callee whose input types are `expected_types`, in
/// count and position, and reports the first input that it rejects as a type error naming `operation_name`. A call
/// accepts an input of exactly the callee input type and every input that `call_input_variation` accepts. Returns the
/// manual mesh axes along which the inputs vary beyond the callee input types, if any, along which the outputs of the
/// call vary too. This is the only owner of that rule, so that every context that binds a call (e.g., eager dispatch,
/// staging, and the transforms of a call) accepts the same inputs and infers the same outputs. An execution of the
/// callee accepts a reduced argument as well (refer to `validate_input_type`), because it reshards its arguments.
fn validate_call_input_types<T: ArrayMemberType>(
    operation_name: &'static str,
    expected_types: &[T],
    input_types: &[T],
) -> Result<Option<CallVariation>, TypeError> {
    if expected_types.len() != input_types.len() {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` expected {} input(s) but got {}",
            expected_types.len(),
            input_types.len(),
        )));
    }
    let mut variation = None::<CallVariation>;
    for (index, (expected, actual)) in expected_types.iter().zip(input_types).enumerate() {
        if expected == actual {
            continue;
        }
        let accepted = match (expected.as_array_type(), actual.as_array_type()) {
            (Ok(expected), Ok(actual)) => call_input_variation(expected, actual).map(|axes| (actual, axes)),
            _ => None,
        };
        let Some((actual, axes)) = accepted else {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` input #{index} expected {expected} but got {actual}",
            )));
        };
        let Some(mesh) = actual.sharding().map(Sharding::mesh).filter(|_| !axes.is_empty()) else {
            continue;
        };
        match &mut variation {
            None => variation = Some(CallVariation { mesh: mesh.clone(), axes }),
            Some(variation) if &variation.mesh == mesh => variation.axes.extend(axes),
            Some(variation) => {
                return Err(TypeError::invalid(format!(
                    "`{operation_name}` input #{index} varies along manual mesh axes of mesh `{mesh}`, but an earlier \
                     input varies along those of mesh `{}`",
                    variation.mesh,
                )));
            }
        }
    }
    Ok(variation)
}

/// Returns the callee of a `jit_call` bound over `callee` as a shared program, which is the program that `callee` was
/// interned from, if any, and otherwise a copy of `callee` that the region transform cache of `callee` retains (refer
/// to [`JitCallCalleeTransform`]).
fn shared_jit_call_callee<V: CoreValue<Type = Type>, O: CoreOperation<Type = Type>>(
    callee: RegionRef<'_, V, O>,
) -> Result<Arc<CoreProgram<V, O, Vec<V>, Vec<V>>>, ProgramError> {
    if let Some(callee) = callee.interned_callee() {
        return Ok(callee.clone());
    }
    let artifact = callee.transform::<JitCallCalleeTransform, _, ProgramError>((), |callee, _| {
        Ok(TransformArtifact::new(vec![Arc::new(callee.to_program())], ()))
    })?;
    Ok(artifact.programs()[0].clone())
}

/// Folds or residualizes `operation` as one whole call (refer to [`PartialEvaluationContext::fold_or_residualize`]).
/// The callee is bound as a shared program (refer to [`shared_jit_call_callee`]), so that repeated calls of one callee
/// keep sharing one region in the known-side context and in the residual program.
fn fold_or_residualize_jit_call<V, C, D>(
    operation: &JitCallOperation<Type>,
    context: &PartialEvaluationContext<C>,
    driver: &D,
    inputs: &[PartialEvaluationValue<C::Value>],
) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>
where
    V: CoreValue<Type = Type> + ValueProjection<ArrayType, Projected: CoreValue<Type = ArrayType>> + CaptureConstant,
    C: Context<Type = Type, Constant = V, Operation = Operation<V>>,
    D: PartialEvaluationDriver<C>,
{
    let callee = shared_jit_call_callee(driver.region(0)?)?;
    context.fold_or_residualize(
        Operation::JitCall(operation.clone()),
        AttachedRegions::Callees(std::slice::from_ref(&callee)),
        inputs,
    )
}

/// Returns whether an all-known call of `callee` with `inputs` keeps its boundary under a residual policy, which is the
/// case when `callee` has effects, references, nominal type identities, or capture constants. Splitting such a call on
/// residual demand, or forwarding its outputs, would move effects, reference accesses, nominally scoped types, or
/// captures that name its capture prefix across the boundary that validates them.
fn keeps_jit_call_boundary<V: CoreValue<Type = Type> + CaptureConstant, O, W: CoreValue>(
    callee: RegionRef<'_, V, O>,
    inputs: &[PartialEvaluationValue<W>],
) -> bool
where
    O: CoreOperation<Type = Type>,
{
    !callee.effects().classes().is_empty() || !can_cache_jit_call_split(callee, inputs)
}

/// Returns whether a mixed call can retain its structural partition. Effects are permitted because its cache key
/// includes their partition configuration, whereas references, captures, and nominal identities require the original
/// boundary or call-local identities. All-known replay additionally excludes effects (refer to
/// [`keeps_jit_call_boundary`]).
fn can_cache_jit_call_split<V: CoreValue<Type = Type> + CaptureConstant, O, W: CoreValue>(
    callee: RegionRef<'_, V, O>,
    inputs: &[PartialEvaluationValue<W>],
) -> bool
where
    O: CoreOperation<Type = Type>,
{
    !inputs
        .iter()
        .any(|input| input.r#type().is_reference() || input.r#type().identities().next().is_some())
        && !callee.contains_atom_type_in_closure(|r#type| r#type.is_reference() || r#type.identities().next().is_some())
        && !callee.contains_atom_in_closure(|atom| {
            atom.as_constant().is_some_and(|constant| constant.capture_index().is_some())
        })
}

/// Returns `outputs`, the outputs of `operation`, a call of `callee` with `inputs`, with every output that `callee`
/// returns by forwarding one of its inputs replaced by that call input, like the input forwarding of JAX's `pjit`
/// staging rule. An all-known call that partial evaluation keeps whole under a residual policy uses this, so that the
/// residual planner sees such an output as the call input that it is, rather than as an output of the call that it
/// would save separately (e.g., a primal input that the primal call of a linearized callee returns as a residual).
/// Logical types do not record placements along auto mesh axes nor devices, so a call that names devices forwards
/// nothing, because it can move and place every value that crosses it (refer to [`JitCallOperation`]), unlike its
/// input, and any other call forwards no input that it places at its boundary (refer to
/// [`JitCallOperation::boundary_input_placement`]). That includes an input whose type records no sharding of a call of
/// a function compiled for an input type without a sharding, which is neither a call that a transform derived nor a
/// call of a `jit` dispatcher: the executable of such a function places that input replicated over the mesh where the
/// call runs, which the type of the input does not show, so the call can move it.
fn forward_jit_call_inputs<V: CoreValue<Type = Type>, O: CoreOperation<Type = Type>, W: CoreValue<Type = Type>>(
    operation: &JitCallOperation<Type>,
    callee: RegionRef<'_, V, O>,
    inputs: &[PartialEvaluationValue<W>],
    outputs: Vec<PartialEvaluationValue<W>>,
) -> Vec<PartialEvaluationValue<W>> {
    if operation.mesh().is_some() {
        return outputs;
    }
    let declared = callee.input_types();
    outputs
        .into_iter()
        .zip(callee.output_ids())
        .map(|(output, atom)| {
            let Some(position) = callee.input_ids().iter().position(|input| input == atom) else { return output };
            let input = &inputs[position];
            let placed = match input.r#type().as_ref() {
                Type::Array(actual) => {
                    let unknown_placement = position >= operation.capture_count()
                        && !operation.keeps_input_placements()
                        && operation.derived_input_shardings.is_none()
                        && operation.input_sharding(position).is_none()
                        && actual.sharding().is_none()
                        && matches!(&declared[position], Type::Array(declared) if declared.sharding().is_none());
                    unknown_placement
                        || operation.boundary_input_placement(position, actual, &declared[position]).is_some()
                }
                _ => false,
            };
            if !placed && input.r#type() == output.r#type() { input.clone() } else { output }
        })
        .collect()
}

/// Splits `operation` into a known call, which binds a known callee in the known-side context, and a residual call,
/// which is emitted into the residual program, and returns the outputs of the call in output order.
///
///   - A split by [`Knownness`](JitCallSplitKind::Knownness) splits a call whose inputs are partially known. The known
///     callee computes the known outputs together with the residual edges that the residual callee needs, which the
///     residual policy of `context`, if any, places. When the known callee contains no instructions, the split hoists
///     no work, and the call is kept whole.
///   - A split on [`ResidualDemand`](JitCallSplitKind::ResidualDemand) splits an all-known call (refer to
///     [`PartiallyEvaluatableOperation::split_on_residual_demand`]). The known callee computes every output of the
///     callee together with the interior values that the residual policy of `context` saves, and the residual callee
///     replays the demanded outputs from those values.
///
/// The residual callee reads the call inputs and the known outputs that its residual edges would merely repeat
/// directly (JAX's `in_fwd` and `out_fwd`), so that the known callee returns each value once and a residual planner
/// plans those inputs like any other demanded value. The two callees are retained by the region transform cache of the
/// callee (refer to [`JitCallSplitTransform`]) and bound as shared regions. Every call of one callee, including the
/// calls nested in the callees of an enclosing split, therefore shares one known callee and one residual callee for
/// each kind of split and each policy, so the programs grow with the number of calls rather than with the size of
/// their callees, and each callee is partitioned once rather than once per call. Calls whose partial evaluations round
/// the saved values (refer to [`PartialEvaluationContext::residual_rounding_type`]) share a split of their own, whose
/// known callee rounds the values that it saves right after their producers, before its own known work consumes them.
/// Mixed splits may contain effects, with the driver's repeated-residual mode included in the key. All-known splits
/// exclude effects, and both kinds exclude references, captures, and nominal identities (refer to
/// [`keeps_jit_call_boundary`] and [`can_cache_jit_call_split`]).
///
/// `partition` partitions the callee for `kind`, which keys the retained split. Callers supply it because a split on
/// residual demand partitions through [`PartialEvaluationContext::partition_residual_demand`], whose bound only the
/// caller's [`PartiallyEvaluatableOperation::split_on_residual_demand`] states.
fn split_jit_call<V, C, D>(
    operation: &JitCallOperation<Type>,
    context: &PartialEvaluationContext<C>,
    driver: &D,
    inputs: &[PartialEvaluationValue<C::Value>],
    kind: JitCallSplitKind,
    partition: &dyn Fn(RegionRef<'_, V, Operation<V>>) -> Result<PartitionedProgram<V, Operation<V>>, ProgramError>,
) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>
where
    V: CoreValue<Type = Type> + ValueProjection<ArrayType, Projected: CoreValue<Type = ArrayType>> + CaptureConstant,
    C: Context<Type = Type, Constant = V, Operation = Operation<V>>,
    D: PartialEvaluationDriver<C>,
{
    let callee = driver.region(0)?;
    let keeps_trivial_splits_whole = matches!(kind, JitCallSplitKind::Knownness(_));
    let arguments = JitCallSplitArguments {
        kind,
        policy: context.residual_policy().cloned(),
        residual_rounding_type: context.residual_rounding_type(),
        allow_effect_folding: context.allow_effect_folding(),
        allow_copy_folding: context.allow_copy_folding(),
        repeated_residual: keeps_trivial_splits_whole
            && !callee.effects().classes().is_empty()
            && driver.partitions_for_repeated_residual_calls(),
    };
    let artifact = callee.transform::<JitCallSplitTransform, _, ProgramError>(arguments, |callee, _| {
        // Partitioning stages both callees in fresh traces, so the partition depends only on the callee and the
        // arguments, including whether mixed splits use repeated-residual allocation and effect-ordering analysis.
        let (known_program, residual_program, known_input_indices, residual_inputs, outputs) =
            partition(callee)?.forward_residuals()?.into_parts();
        Ok(TransformArtifact::new(
            vec![Arc::new(known_program), Arc::new(residual_program)],
            Arc::new(JitCallSplit { known_input_indices, residual_inputs, outputs }),
        ))
    })?;
    let [known_callee, residual_callee] = artifact.programs() else {
        unreachable!("`jit_call` splits retain a known callee and a residual callee");
    };
    if keeps_trivial_splits_whole && known_callee.instructions().is_empty() {
        return fold_or_residualize_jit_call(operation, context, driver, inputs);
    }
    let split = artifact.metadata();

    // The effective boundary of the call follows the types of its inputs before they are guarded.
    let boundary =
        operation.effective_input_boundary(&callee.input_types(), inputs.iter().map(PartialEvaluationValue::r#type));

    // Partitioning seeds both callees at the declared input types of the callee, so neither checks a relationship that
    // the entry of the callee checks between a known and an unknown input. Both calls therefore consume guarded inputs.
    let inputs = context.guard_inputs(&callee.input_types(), inputs.to_vec(), false)?;

    // The known callee keeps the members of the lifted capture prefix that it uses as its leading inputs, in order.
    // The residual callee reads no capture prefix, and neither callee holds a capture constant.
    let known_capture_count =
        split.known_input_indices.iter().filter(|index| **index < operation.capture_count()).count();
    let known_call = JitCallOperation::new(known_capture_count).with_callee_region(known_callee.entry_region_ref())?;
    let residual_call = JitCallOperation::new(0).with_callee_region(residual_callee.entry_region_ref())?;
    let (known_call, residual_call) = operation.with_split_boundary_shardings(
        known_call,
        residual_call,
        &boundary,
        &split.known_input_indices,
        &split.residual_inputs,
        &split.outputs,
        known_callee.output_count(),
        residual_callee.output_count(),
    );
    let known_inputs = split.known_input_indices.iter().map(|index| inputs[*index].clone()).collect::<Vec<_>>();
    let known_outputs = context.fold_or_residualize(
        Operation::JitCall(known_call.clone()),
        AttachedRegions::Callees(std::slice::from_ref(known_callee)),
        &known_inputs,
    )?;
    let known_outputs = match context.residual_policy() {
        Some(_) if known_callee.effects().classes().is_empty() => {
            forward_jit_call_inputs(&known_call, known_callee.entry_region_ref(), &known_inputs, known_outputs)
        }
        _ => known_outputs,
    };

    // The residual edges trail the known outputs among the outputs of the known call.
    let known_output_count = split.outputs.iter().filter(|output| output.is_known()).count();
    let residual_inputs = split
        .residual_inputs
        .iter()
        .map(|source| match *source {
            ResidualInputSource::UnknownInput(index) | ResidualInputSource::KnownInput(index) => inputs[index].clone(),
            ResidualInputSource::KnownOutput(index) => known_outputs[index].clone(),
            ResidualInputSource::ResidualEdge(edge) => known_outputs[known_output_count + edge].clone(),
        })
        .collect::<Vec<_>>();
    let residual_outputs = context.residualize(
        Operation::JitCall(residual_call),
        AttachedRegions::Callees(std::slice::from_ref(residual_callee)),
        &residual_inputs,
    )?;
    Ok(split
        .outputs
        .iter()
        .map(|output| match *output {
            PartialEvaluationOutput::Known(index) => known_outputs[index].clone(),
            PartialEvaluationOutput::Unknown(index) => residual_outputs[index].clone(),
        })
        .collect())
}

/// Lowers one `jit_call` to either a `func.call` of a shared private function (when its callee was deduplicated) or
/// an inlined copy of the callee body (otherwise).
///
/// `input_values` are the lowered call operands in callee-input order. `capture_count` is the operation payload's
/// exact leading lifted-capture prefix length. Deriving it from capture-constant indices instead would conflate the
/// independent capture namespaces of nested calls inside the callee arena. `caller_captured_values` are the captured
/// values of the calling scope, whose session-constant suffix the inlined callee inherits (refer to
/// [`CollectiveLoweringState::session_values`]).
#[allow(clippy::too_many_arguments)]
fn lower_jit_call<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
    program: &FlatProgram<O>,
    capture_count: usize,
    input_values: &[ValueRef<'b, 'c, 't>],
    caller_captured_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    nested_functions: Option<&Rc<JitCallFunctionMap<O>>>,
    collective_state: &CollectiveLoweringState,
    effect_tokens: &mut EffectTokens<'b, 'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    // Only pure callees are ever deduplicated (`collect_jit_call_functions` skips effectful programs), so the shared
    // function path below never interacts with the caller's effect tokens. Effectful callees always take the inline
    // path, which threads the caller's per-class tokens through the callee body in program order.
    if let Some(map) = nested_functions
        && let Some(function) = map.get(program)
    {
        // The `jit_call` operation's type inference already pins its operands to the callee input types, so a
        // matching arity is the only guard needed before emitting the symbol call. Anything else inlines.
        if input_values.len() == function.input_types.len() {
            return lower_shared_function_call(
                &function.symbol,
                &function.output_types,
                input_values,
                block,
                context,
                location,
            );
        }
    }
    if input_values.len() < capture_count {
        // The provided inputs cover the leading captures, so the first capture without a value is the next one.
        return Err(Error::Lowering(LoweringError::MissingCapturedConstant { index: input_values.len() }));
    }
    // The callee's capture references name its own leading inputs, while session constants remain module-scoped.
    let mut captured_values = input_values[..capture_count].to_vec();
    captured_values.extend_from_slice(collective_state.session_values(caller_captured_values));
    lower_nested_program_inline(
        program,
        input_values,
        block,
        context,
        location,
        captured_values.as_slice(),
        &RematerializationOptimizationBarrier::None,
        nested_functions,
        collective_state,
        effect_tokens,
    )
}

/// Constrains `value`, an operand or a result of type `r#type` of a `jit_call`, to `sharding`, the effective boundary
/// sharding of the corresponding position of the boundary of its callee (i.e., its override, or the declared sharding
/// of an input without one, refer to [`JitCallOperation::boundary_input_sharding`]), and returns the constrained value.
/// The constraint is the sharding with which the called function annotates that position of its own boundary (refer to
/// [`entry_boundary_sharding`]), closed over auto mesh axes. Values that are not arrays are returned unchanged.
fn lower_boundary_constraint<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
    lowerer: &mut Lowerer<'b, 'c, 't, O>,
    value: ValueRef<'b, 'c, 't>,
    r#type: &Type,
    sharding: &Sharding,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    let Type::Array(array_type) = r#type else {
        return Ok(value);
    };
    // The override states the placement along every mesh axis, so it is lowered for a type that tracks no placement,
    // with which a constraint is not merged (refer to `ConstrainShardingOperation::lowered_sharding`).
    let untracked_type = array_type.clone().with_sharding(None)?;
    let constraint = ConstrainShardingOperation::new(entry_boundary_sharding(sharding)?);
    let types = std::slice::from_ref(&untracked_type);
    Ok(constraint.lower(lowerer, &[value], types, types)?.remove(0))
}

#[cfg(test)]
mod tests {
    use std::hash::DefaultHasher;
    use std::sync::{Arc, Mutex};

    use indoc::{formatdoc, indoc};
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, Array as CoreArray, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayOperation, ArrayType,
        AssertOperation, AtomId, AttachedRegions, AxisIndexOperation, BatchableOperation, BatchingContext,
        BroadcastOperation, CaptureReference, CompareOperation, ComparisonDirection, CompositeJvpRule,
        CompositeOperation, CompositeTransposeOperation, ConditionOperation, Context, ConvertElementTypeOperation,
        CosOperation, CotangentDestinationKind, CotangentDestinations, CustomCallOperation, CustomRuleDefinition,
        CustomRuleRegistration, DataType, Device, DifferentiableType, DifferentiationError, Dimension, DimensionBounds,
        DimensionFromScalarOperation, DimensionToScalarOperation, DimensionType, DimensionValue, DimensionVariable,
        DotDimensionNumbers, DotOperation, DotsSavable, DynamicBroadcastOperation, EagerInterpretationDriver,
        EffectClass, EffectClasses, Fill, InputRegionProvenance, IotaOperation, Linearization, LogicalMesh, MaybeZero,
        Memory, MeshAxis, MeshAxisType, MulOperation, NamedComposite, NegOperation, NoStorage, NothingSavable,
        Operation as CoreOperation, OutputRegionProvenance, PartialEvaluation, PartialEvaluationContext,
        PartialEvaluationInput, PartialEvaluationOutput, PartialValue, Place, PlaceOperation, Placeholder,
        PrintOperation, ProgramBuilder as CoreProgramBuilder, ProgramError, RecursiveBatchingDriver, Reduce,
        ReferenceAddUpdateOperation, ReferenceFreezeOperation, ReferenceNewOperation, ReferenceReadOperation,
        ReferenceSource, ReferenceSwapOperation, ReferenceType, RegionDataFlow, RegionDriver, RegionInterface,
        RegionRef, RematerializationPolicyFn, RematerializeOperation, ResidualCandidate, ResidualDecision,
        ResidualPolicy, ResidualPolicyError, ResidualPolicyReference, ResidualRejection, SaveOnlyTheseNames,
        ScanOperation, Shape, Sharding, ShardingDimension, Sin, SinOperation, StagingContext, TagOperation, Tracer,
        TracingContext, TransposeOperation, TransposedProgram, TransposedProgramCotangent, TranspositionDriver,
        TypeError, TypeIdentity, Typed, TypedProgram, ValueProjection, ZeroLikeOperation, ZeroOperation,
    };
    use ryft_macros::Operation;
    use ryft_pjrt::{ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram};

    use crate::lowering::lower_mlir_module_for_program;
    use crate::lowering::tests::{
        MixedValue, ProgramBuilder, add_jit_call, execute_mixed_program, identity_branch,
        lower_replicated_shard_map_body, neg_branch, test_vector_type, unproject_plain_program,
    };
    use crate::operations::debugging::{ensure_print_handler_registered, with_captured_prints};
    use crate::operations::{Operation, Program, ProgramBuilder as ArrayIrProgramBuilder};
    use crate::sessions::tests::{domain_mesh, read_first_shard, two_device_mesh};
    use crate::tests::{
        cpu_client, execution_client, test_compilation_options, values_from_bytes, values_to_bytes, vector_type,
    };
    use crate::{
        Array, ArrayConstant, CompileTracer, CompiledFunction, Constant, CustomCallOperationExtension, Error, FromPjrt,
        Options, RuntimeRequirement, StagedFunction, ToPjrt, Type, Value, compile, compile_with_options, stage,
        to_mlir_module_for_program,
    };

    use super::*;

    /// Test-only driver that exposes one source callee and returns a predetermined transpose for it.
    struct TestTranspositionDriver {
        /// Source callee exposed to the `jit_call` transpose rule.
        source: Program<Vec<Constant>, Vec<Constant>>,

        /// Predetermined transposed callee returned by the recursive request.
        transposed: Program<Vec<Constant>, Vec<Constant>>,
    }

    impl RegionDriver<Constant, Operation> for TestTranspositionDriver {
        fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, Constant, Operation>>
        where
            Constant: 'r,
            Operation: 'r,
        {
            std::iter::once(self.source.entry_region_ref())
        }
    }

    impl TranspositionDriver<Constant, Operation> for TestTranspositionDriver {
        fn transpose_program(
            &self,
            region: RegionRef<'_, Constant, Operation>,
            input_indices: &[usize],
            materialize_zeros: &[bool],
            destination_kinds: &[CotangentDestinationKind],
        ) -> Result<TransposedProgram<Constant, Operation>, DifferentiationError> {
            let _ = materialize_zeros;
            // The predetermined program must have the boundary of the requested region's transposition, with one input
            // per non-reference output, per `Reference` destination, and per known input. The default zero-seed
            // request projects the region onto its unflagged outputs before calling this function, so this also pins
            // the structural zero seed mask that the rule requests.
            let input_types = region.input_types();
            let output_count = region.output_types().iter().filter(|r#type| !r#type.is_reference()).count();
            let reference_count =
                destination_kinds.iter().filter(|&&kind| kind == CotangentDestinationKind::Reference).count();
            assert_eq!(
                self.transposed.input_types().len(),
                output_count + reference_count + input_types.len() - input_indices.len(),
            );
            let cotangents = input_indices
                .iter()
                .enumerate()
                .map(|(index, &input_index)| {
                    match destination_kinds.get(index).copied().unwrap_or(CotangentDestinationKind::Return) {
                        CotangentDestinationKind::Return => TransposedProgramCotangent::Output,
                        CotangentDestinationKind::Reference if input_types[input_index].is_reference() => {
                            TransposedProgramCotangent::Output
                        }
                        CotangentDestinationKind::Reference | CotangentDestinationKind::Ignore => {
                            TransposedProgramCotangent::Absent
                        }
                    }
                })
                .collect();
            Ok(TransposedProgram::new(Arc::new(self.transposed.clone()), cotangents)?)
        }
    }

    /// Compiles and executes an array-only `f32` fixture using its declared unsharded program signature.
    fn execute_f32_program(
        client: &'static ryft_pjrt::Client<'static>,
        program: &Program<Vec<Constant>, Vec<Constant>>,
        values: Vec<Vec<f32>>,
    ) -> Vec<Vec<f32>> {
        for r#type in program.input_types().iter() {
            assert!(matches!(r#type, Type::Array(r#type) if r#type.data_type() == DataType::F32));
        }
        let values = values.iter().map(|values| values_to_bytes(values)).collect();
        execute_array_program(client, program, values)
            .iter()
            .map(|bytes| values_from_bytes::<f32>(bytes))
            .collect()
    }

    /// Builds a mixed-call fixture with one print in each partition, i.e., `known = print(k * k)` and
    /// `residual = print(x * known)`. Its two scalar inputs are the known coefficient and the unknown value.
    fn effectful_mixed_callee(effect_class: EffectClass) -> Arc<FlatProgram> {
        let mut builder = ArrayIrProgramBuilder::new();
        let scalar_type: Type = ArrayType::scalar(DataType::F32).into();
        let known = builder.add_input(scalar_type.clone());
        let unknown = builder.add_input(scalar_type);
        let squared = builder.add_instruction(MulOperation::new(), Vec::new(), vec![known, known], None).unwrap()[0];
        let coefficient = builder
            .add_instruction(
                PrintOperation::<ArrayType>::new("known").with_effect_class(effect_class),
                Vec::new(),
                vec![squared],
                None,
            )
            .unwrap()[0];
        let product =
            builder.add_instruction(MulOperation::new(), Vec::new(), vec![unknown, coefficient], None).unwrap()[0];
        let output = builder
            .add_instruction(
                PrintOperation::<ArrayType>::new("residual").with_effect_class(effect_class),
                Vec::new(),
                vec![product],
                None,
            )
            .unwrap()[0];
        Arc::new(
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap(),
        )
    }

    /// Compiles and executes an array-only fixture using its declared unsharded program signature, with each input
    /// and output given as the native-endian bytes of its elements.
    fn execute_array_program(
        client: &'static ryft_pjrt::Client<'static>,
        program: &Program<Vec<Constant>, Vec<Constant>>,
        values: Vec<Vec<u8>>,
    ) -> Vec<Vec<u8>> {
        let array_type = |r#type: &Type| match r#type {
            Type::Array(r#type) => r#type.clone(),
            _ => panic!("expected an array signature"),
        };
        let input_types = program.input_types().iter().map(array_type).collect::<Vec<_>>();
        let output_types = program.output_types().iter().map(array_type).collect::<Vec<_>>();
        let module = to_mlir_module_for_program(program, &[], &input_types, &output_types, "main", None, None).unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_compilation_options(client, None))
            .unwrap();
        let device = executable.addressable_devices().unwrap().remove(0);
        assert_eq!(values.len(), program.input_count());
        let inputs = values
            .into_iter()
            .zip(program.input_types().into_owned())
            .map(|(values, r#type)| {
                let Type::Array(r#type) = r#type else {
                    panic!("expected an array input");
                };
                let dimensions =
                    r#type.static_shape().unwrap().dimensions().iter().map(|value| *value as u64).collect::<Vec<_>>();
                let buffer = client
                    .buffer(values.as_slice(), r#type.data_type().to_pjrt(), &dimensions, None, device.clone(), None)
                    .unwrap();
                ExecutionInput { buffer: Arc::new(buffer), donatable: false }
            })
            .collect::<Vec<_>>();
        executable
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
            .unwrap()
            .block_until_ready()
            .unwrap()
            .remove(0)
            .outputs
            .iter()
            .map(|output| output.copy_to_host(None).unwrap().r#await().unwrap())
            .collect()
    }

    /// Returns the `f32[2, 2]` matrix type of the residual demand workloads.
    fn residual_demand_matrix_type() -> Type {
        Type::from(ArrayType::new_static(DataType::F32, [2, 2]))
    }

    /// Stages `sin(a · b) ⊙ cos(a · b) + a` for `a = left` and `b = right` into `builder` and returns its output.
    fn add_residual_demand_leaf(builder: &mut ArrayIrProgramBuilder, left: AtomId, right: AtomId) -> AtomId {
        let dot = DotOperation::new(DotDimensionNumbers::new(vec![1], vec![0], vec![], vec![]));
        let product = builder.add_instruction(dot, Vec::new(), vec![left, right], None).unwrap()[0];
        let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![product], None).unwrap()[0];
        let cosine = builder.add_instruction(CosOperation::new(), Vec::new(), vec![product], None).unwrap()[0];
        let scaled = builder.add_instruction(MulOperation::new(), Vec::new(), vec![sine, cosine], None).unwrap()[0];
        builder.add_instruction(AddOperation::new(), Vec::new(), vec![scaled, left], None).unwrap()[0]
    }

    /// Stages the nested diamond `g_k(a, b) = g_{k-1}(a, b) + g_{k-1}(sin a, b)` of `depth` levels over
    /// `g_0(a, b) = sin(a · b) ⊙ cos(a · b) + a` into `builder` with every level inlined, and returns its output. This
    /// is the call-free reference of [`residual_demand_diamond_callee`].
    fn add_inlined_residual_demand_diamond(
        builder: &mut ArrayIrProgramBuilder,
        depth: usize,
        left: AtomId,
        right: AtomId,
    ) -> AtomId {
        if depth == 0 {
            return add_residual_demand_leaf(builder, left, right);
        }
        let first = add_inlined_residual_demand_diamond(builder, depth - 1, left, right);
        let shifted = builder.add_instruction(SinOperation::new(), Vec::new(), vec![left], None).unwrap()[0];
        let second = add_inlined_residual_demand_diamond(builder, depth - 1, shifted, right);
        builder.add_instruction(AddOperation::new(), Vec::new(), vec![first, second], None).unwrap()[0]
    }

    /// Returns `(a, b) ↦ sin(a · b) ⊙ cos(a · b) + a`, the shared callee of the residual demand workloads, whose matrix
    /// product `DotsSavable` saves while it recomputes the rest of the callee.
    fn residual_demand_leaf_callee() -> Arc<FlatProgram> {
        let mut builder = ArrayIrProgramBuilder::new();
        let left = builder.add_input(residual_demand_matrix_type());
        let right = builder.add_input(residual_demand_matrix_type());
        let output = add_residual_demand_leaf(&mut builder, left, right);
        Arc::new(
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap(),
        )
    }

    /// Returns the nested diamond `g_k(a, b) = g_{k-1}(a, b) + g_{k-1}(sin a, b)` of `depth` levels over the shared
    /// residual demand callee `g_0`, where each level calls the previous level twice through one shared callee.
    fn residual_demand_diamond_callee(depth: usize) -> Arc<FlatProgram> {
        let mut callee = residual_demand_leaf_callee();
        for _ in 0..depth {
            let mut builder = ArrayIrProgramBuilder::new();
            let left = builder.add_input(residual_demand_matrix_type());
            let right = builder.add_input(residual_demand_matrix_type());
            let first = add_jit_call(&mut builder, &callee, vec![left, right]);
            let shifted = builder.add_instruction(SinOperation::new(), Vec::new(), vec![left], None).unwrap()[0];
            let second = add_jit_call(&mut builder, &callee, vec![shifted, right]);
            let output =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![first, second], None).unwrap()[0];
            callee = Arc::new(
                builder
                    .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                    .unwrap(),
            );
        }
        callee
    }

    /// Returns `(x, w, b) ↦ x ⊙ y`, where `body` stages the calls that compute `y` from `w` and `b`, and, when `policy`
    /// is provided, wraps that function in a rematerialization under `policy`. Differentiating only `x` makes `y` the
    /// only value that the tangent needs, so every call stays all-known, while differentiating `w` as well makes every
    /// call a differentiated call whose tangent call mixes unknown tangents with known residuals.
    fn residual_demand_program(
        policy: Option<ResidualPolicyReference<Type>>,
        body: impl Fn(&mut ArrayIrProgramBuilder, AtomId, AtomId) -> AtomId,
    ) -> FlatProgram {
        let mut builder = ArrayIrProgramBuilder::new();
        let inputs = (0..3).map(|_| builder.add_input(residual_demand_matrix_type())).collect::<Vec<_>>();
        let called = body(&mut builder, inputs[1], inputs[2]);
        let output =
            builder.add_instruction(MulOperation::new(), Vec::new(), vec![inputs[0], called], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let Some(policy) = policy else {
            return program;
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let inputs = (0..3).map(|_| builder.add_input(residual_demand_matrix_type())).collect::<Vec<_>>();
        let body = builder.import_program(program);
        let output = builder.add_instruction(RematerializeOperation::new(policy), vec![body], inputs, None).unwrap()[0];
        builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap()
    }

    /// Returns the residual count of the linearization of `program` with respect to the inputs that `input_indices`
    /// selects, followed by the numbers of instructions in the complete region closures of its primal and tangent
    /// programs, which count every shared callee once.
    fn residual_demand_statistics(program: &FlatProgram, input_indices: &[usize]) -> [usize; 3] {
        let linearization = program.linearize_with_respect_to(input_indices).unwrap();
        let closure_size = |program: &FlatProgram| program.entry_region_ref().instructions_in_closure().count();
        [linearization.residual_count(), closure_size(linearization.primal()), closure_size(linearization.tangent())]
    }

    /// Asserts that the linearizations of `actual` and `expected` with respect to the inputs that `input_indices`
    /// selects (at most two) compute the same primal output, tangent, and cotangents on fixed inputs, up to the
    /// rounding differences that recomputing transcendental functions in differently fused XLA kernels introduces.
    fn assert_residual_demand_linearizations_match(
        client: &'static ryft_pjrt::Client<'static>,
        actual: &FlatProgram,
        expected: &FlatProgram,
        input_indices: &[usize],
    ) {
        let execute = |program: &FlatProgram| {
            let linearization = program.linearize_with_respect_to(input_indices).unwrap();
            let inputs =
                vec![vec![0.5f32, -1.0, 0.25, 2.0], vec![0.3f32, 0.1, -0.2, 0.4], vec![0.7f32, -0.6, 0.2, 0.9]];
            let mut primal = execute_f32_program(client, linearization.primal(), inputs);
            let value = primal.remove(0);
            let input_tangents = [vec![1.0f32, 0.5, -0.5, 0.25], vec![-0.25f32, 0.75, 0.5, -1.0]];
            let tangent_inputs =
                input_tangents[..input_indices.len()].iter().cloned().chain(primal.clone()).collect::<Vec<_>>();
            let tangent = execute_f32_program(client, linearization.tangent(), tangent_inputs);
            let pullback_inputs = std::iter::once(vec![-1.5f32, 0.75, 2.0, 1.0]).chain(primal).collect();
            let pullback =
                execute_f32_program(client, &linearization.pullback().unwrap().into_program(), pullback_inputs);
            [vec![value], tangent, pullback].concat().concat()
        };
        let actual = execute(actual);
        let expected = execute(expected);
        assert_eq!(actual.len(), expected.len());
        assert!(
            actual
                .iter()
                .zip(&expected)
                .all(|(actual, expected)| (actual - expected).abs() <= 1e-5 * expected.abs().max(1.0)),
            "{actual:?} != {expected:?}",
        );
    }

    /// Builds the flat callee `f(x) = x + x` over a vector type, returned behind an [`Arc`] so callers control
    /// whether `jit_call` sites share one program (i.e., by pointer identity) or use structurally identical distinct
    /// programs.
    fn add_self_callee(input_type: ArrayType) -> Arc<FlatProgram> {
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(input_type.into());
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
        Arc::new(builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap())
    }

    /// Lowers an outer program that calls each of `callees` through one `jit_call` and sums the results, returning
    /// the module text. Each callee is `f(x) = x + x`, and the outer function is `g(x) = sum_i callee_i(x)`.
    fn lower_jit_call_sum_module(callees: Vec<Arc<FlatProgram>>) -> String {
        let array_type = test_vector_type(4);
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(array_type.clone().into());
        let mut accumulator: Option<AtomId> = None;
        for callee in callees {
            let call_output = add_jit_call(&mut builder, &callee, vec![input]);
            accumulator = Some(match accumulator {
                None => call_output,
                Some(previous) => builder
                    .add_instruction(AddOperation::new(), Vec::new(), vec![previous, call_output], None)
                    .unwrap()[0],
            });
        }
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![accumulator.unwrap()], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let input_types = vec![array_type.clone()];
        let output_types = vec![array_type];
        to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap()
    }

    /// Builds the flat callee `f(p, x) = if p { -x } else { x }`, whose body contains a `condition` instruction. Its
    /// nested branch bodies render contextually into the callee's canonical program text, so separately constructed
    /// copies of this callee remain eligible for structural `jit_call` deduplication.
    fn condition_callee() -> Arc<FlatProgram> {
        let vector_type = test_vector_type(4);
        let mut builder = ArrayIrProgramBuilder::new();
        let true_region = builder.import_program(unproject_plain_program(neg_branch(vector_type.clone())));
        let false_region = builder.import_program(unproject_plain_program(identity_branch(vector_type.clone())));
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(vector_type.into());
        let output = builder
            .add_instruction(
                Operation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
                None,
            )
            .unwrap()[0];
        Arc::new(builder.build(vec![output], vec![Placeholder, Placeholder], vec![Placeholder]).unwrap())
    }

    #[test]
    fn test_collect_jit_call_functions() {
        let callee = add_self_callee(test_vector_type(4));
        let module = lower_jit_call_sum_module(vec![callee.clone(), callee]);

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
    fn test_collect_jit_call_functions_shares_structurally_identical_programs() {
        // Two distinct programs (i.e., separate `Arc`s) with identical structure must still deduplicate into the same
        // single shared function as the pointer-identical case above. This is the shape produced when a transform
        // such as `grad` linearizes each of several identical blocks into its own staged program.
        let module =
            lower_jit_call_sum_module(vec![add_self_callee(test_vector_type(4)), add_self_callee(test_vector_type(4))]);

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
    fn test_collect_jit_call_functions_distinguishes_callees_whose_calls_differ() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let device_mesh = DeviceMesh::new(mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)]).unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let array_type = test_vector_type(4).with_sharding(replicated.clone()).unwrap();
        let inner = {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(array_type.clone().into());
            Arc::new(
                builder
                    .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
                    .unwrap(),
            )
        };
        let outer = |call: &JitCallOperation<Type>| {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(array_type.clone().into());
            let region = builder.intern_callee(&inner, None).unwrap();
            let output =
                builder.add_instruction(Operation::JitCall(call.clone()), vec![region], vec![input], None).unwrap()[0];
            Arc::new(
                builder
                    .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                    .unwrap(),
            )
        };
        let program = |first: &JitCallOperation<Type>, second: &JitCallOperation<Type>| {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(array_type.clone().into());
            let first = add_jit_call(&mut builder, &outer(first), vec![input]);
            let second = add_jit_call(&mut builder, &outer(second), vec![input]);
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![first, second],
                    vec![Placeholder],
                    vec![Placeholder, Placeholder],
                )
                .unwrap()
        };
        let base = JitCallOperation::<Type>::new(0);

        // Two structurally identical callees whose calls differ in any field of the call that lowering uses share no
        // function, so that each call lowers its own boundary. Only the shared innermost callee gets a function. The
        // destructuring names every field of the call, so that a field added later must be classified here.
        let JitCallOperation {
            capture_count: _,
            marker: _,
            output_types: _,
            input_shardings: _,
            output_shardings: _,
            mesh: _,
            keeps_input_placements: _,
            derived_input_shardings: _,
        } = &base;
        let derived = |shardings: Vec<Option<Sharding>>| JitCallOperation::<Type> {
            derived_input_shardings: Some(Arc::from(shardings)),
            ..JitCallOperation::new(0)
        };
        let variants = [
            JitCallOperation::new(1),
            JitCallOperation::new(0).with_boundary(Some(&device_mesh), vec![Some(sharded.clone())], Vec::new()),
            JitCallOperation::new(0).with_boundary(Some(&device_mesh), Vec::new(), vec![Some(replicated.clone())]),
            JitCallOperation::new(0).with_boundary(Some(&device_mesh), Vec::new(), Vec::new()),
            JitCallOperation::new(0).with_kept_input_placements(),
            derived(vec![None]),
            derived(vec![Some(sharded)]),
        ];
        let function_count =
            |program: &FlatProgram| collect_jit_call_functions(program, None).unwrap().functions().len();
        assert_eq!(function_count(&program(&base, &base)), 2);
        for variant in &variants {
            assert_eq!(function_count(&program(&base, variant)), 1, "{variant:?}");
        }

        // A call that a transform derives without a boundary for its input renders as a call of a compiled function,
        // which constrains its input to the replicated sharding of the declared type, but the two lower differently.
        let input_types = vec![array_type.clone()];
        let output_types = vec![array_type.clone(), array_type.clone()];
        let module = to_mlir_module_for_program(
            &program(&base, &derived(vec![None])),
            &[],
            &input_types,
            &output_types,
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                }
            "#},
        );
    }

    #[test]
    fn test_collect_jit_call_functions_inlines_single_calls() {
        // A single-occurrence callee stays below the dedup threshold and inlines, so no shared function is emitted
        // and the callee body appears directly in `@main`.
        let module = lower_jit_call_sum_module(vec![add_self_callee(test_vector_type(4))]);

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
    fn test_collect_jit_call_functions_excludes_dormant_regions() {
        let array_type = test_vector_type(4);
        let callee = |identifier| {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(array_type.clone().into());
            let operation = CustomCallOperation::new(identifier, vec![array_type.clone()])
                .with_runtime_requirements([RuntimeRequirement::builtin(identifier)])
                .unwrap();
            let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
            Arc::new(
                builder
                    .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                    .unwrap(),
            )
        };
        let active = callee("test.active");
        let dormant = callee("test.dormant");
        let primal = {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(array_type.clone().into());
            let first = add_jit_call(&mut builder, &active, vec![input]);
            let second = add_jit_call(&mut builder, &active, vec![input]);
            let output =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![first, second], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let jvp = {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(array_type.clone().into());
            let tangent = builder.add_input(array_type.clone().into());
            let first = add_jit_call(&mut builder, &dormant, vec![tangent]);
            let second = add_jit_call(&mut builder, &dormant, vec![tangent]);
            let output_tangent =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![first, second], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![input, output_tangent],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(array_type.clone().into());
        let primal_region = builder.import_region(primal.entry_region_ref());
        let jvp = builder.import_region(jvp.entry_region_ref());
        let output = builder
            .add_instruction(
                Operation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, false)),
                vec![primal_region, jvp],
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let lowered = lower_mlir_module_for_program(
            &program,
            &[],
            &vec![array_type.clone()],
            &vec![array_type.clone()],
            "main",
            None,
            None,
            None,
        )
        .unwrap();

        let (stable_hlo, _, _, runtime_requirements) = lowered.into_parts();
        assert_eq!(runtime_requirements.as_ref(), &[RuntimeRequirement::builtin("test.active")]);
        assert_eq!(
            stable_hlo,
            indoc! {r#"
                module {
                  func.func private @jit_call_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.custom_call @test.active(%arg0) {api_version = 4 : i32, backend_config = {}, operand_layouts = [dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>]} : (tensor<4xf32>) -> tensor<4xf32>
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

        // Backend replacement also makes a source region dormant, so helper discovery must not emit its callees.
        let named = NamedComposite::new(
            CompositeKey::new("ryft.tests.xla_named_square", 1),
            |types: &[Type]| Ok(types.to_vec()),
            |inputs| Ok(vec![inputs[0].clone() + inputs[0].clone()]),
        );
        let rules: CustomRuleRegistration<Constant, Operation> =
            CustomRuleRegistration::new(CustomRuleDefinition::new("replaced").with_named_composite(named));
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(array_type.clone().into());
        let body = builder.import_region(primal.entry_region_ref());
        let output = builder
            .add_instruction(
                Operation::Composite(CompositeOperation::new(rules.reference())),
                vec![body],
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let lowered = lower_mlir_module_for_program(
            &program,
            &[],
            &vec![array_type.clone()],
            &vec![array_type],
            "main",
            None,
            None,
            Some("cuda"),
        )
        .unwrap();

        let (stable_hlo, _, _, runtime_requirements) = lowered.into_parts();
        assert!(runtime_requirements.is_empty());
        assert_eq!(
            stable_hlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_collect_jit_call_functions_skips_shard_map_bodies() {
        // `count_jit_calls` intentionally skips `shard_map` bodies, so a callee that occurs twice inside a `shard_map`
        // body never gets a shared `func.func private @jit_call_*` and both occurrences inline into the
        // `sdy.manual_computation` region.
        let vector_type = test_vector_type(4);
        let callee = add_self_callee(vector_type.clone());
        let body_program = {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(vector_type.clone().into());
            let first = add_jit_call(&mut builder, &callee, vec![input]);
            let second = add_jit_call(&mut builder, &callee, vec![input]);
            let output =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![first, second], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let module = lower_replicated_shard_map_body(body_program, 1).unwrap();

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
    fn test_collect_jit_call_functions_shares_programs_with_nested_bodies() {
        // With nested computations attached as regions, a callee's `condition` bodies render contextually inside its
        // canonical rendering, so two structurally identical but separately constructed callees share one structural
        // identity and merge into a single emitted function, exactly like the body-free callees of
        // `test_collect_jit_call_functions_shares_structurally_identical_programs`.
        let first = condition_callee();
        let second = condition_callee();
        let vector_type = test_vector_type(4);
        let mut builder = ArrayIrProgramBuilder::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(vector_type.clone().into());
        let mut accumulator: Option<AtomId> = None;
        for callee in [first.clone(), first, second.clone(), second] {
            let call_output = add_jit_call(&mut builder, &callee, vec![predicate, input]);
            accumulator = Some(match accumulator {
                None => call_output,
                Some(previous) => builder
                    .add_instruction(AddOperation::new(), Vec::new(), vec![previous, call_output], None)
                    .unwrap()[0],
            });
        }
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(
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
    fn test_collect_jit_call_functions_inlines_effectful_callees() {
        // A repeated `jit_call` callee that prints is excluded from function deduplication and inlines at every
        // call site, so both inlined prints chain onto the caller's ordered I/O token in program order, which a shared
        // token-free `func.func` could not preserve.
        let array_type = test_vector_type(4);
        let mut callee_builder = ProgramBuilder::new();
        let callee_input = callee_builder.add_input(array_type.clone());
        let printed = callee_builder
            .add_instruction(PrintOperation::<ArrayType>::new("callee"), Vec::new(), vec![callee_input], None)
            .unwrap()[0];
        let callee_output = callee_builder
            .add_instruction(AddOperation::new(), Vec::new(), vec![printed, printed], None)
            .unwrap()[0];
        let callee = Arc::new(unproject_plain_program(
            callee_builder.build(vec![callee_output], vec![Placeholder], vec![Placeholder]).unwrap(),
        ));
        let module = lower_jit_call_sum_module(vec![callee.clone(), callee]);

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1 = stablehlo.custom_call @ryft.print(%arg0, %0) {api_version = 4 : i32, backend_config = {label = "callee"}, has_side_effect = true, operand_layouts = [dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], result_layouts = [dense<> : tensor<0xindex>]} : (tensor<4xf32>, !stablehlo.token) -> !stablehlo.token
                    %2 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    %3 = stablehlo.custom_call @ryft.print(%arg0, %1) {api_version = 4 : i32, backend_config = {label = "callee"}, has_side_effect = true, operand_layouts = [dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], result_layouts = [dense<> : tensor<0xindex>]} : (tensor<4xf32>, !stablehlo.token) -> !stablehlo.token
                    %4 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    %5 = stablehlo.add %2, %4 : tensor<4xf32>
                    return %5 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_collect_jit_call_functions_inlines_callees_with_synthesized_guards() {
        // The repeated callee is pure in the source program, but its inner conditional's mixed entry (a symbolic
        // `dimension<m>` and an exact `f64[2]` entering `[dimension<n>, f64[n]]`) gains an entry guard during lowering.
        // A shared token-free function could not order that guard, so the callee inlines at both call sites, and the
        // print after them consumes a token that the guards produce.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let m = DimensionType::new("m", DimensionBounds::positive(Some(8)).unwrap());
        let array_type = ArrayType::new_static(DataType::F64, [2]);
        let branch = || {
            let mut builder = ArrayIrProgramBuilder::new();
            builder.add_input(n.clone().into());
            let array =
                builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![n.simplified_dimension()])).into());
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![array], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let true_region = builder.import_region(branch().entry_region_ref());
            let false_region = builder.import_region(branch().entry_region_ref());
            let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
            let dimension = builder.add_input(m.clone().into());
            let array = builder.add_input(array_type.clone().into());
            let selected = builder
                .add_instruction(
                    Operation::Condition(ConditionOperation::new()),
                    vec![true_region, false_region],
                    vec![predicate, dimension, array],
                    None,
                )
                .unwrap()[0];
            Arc::new(
                builder
                    .build::<Vec<Constant>, Vec<Constant>>(vec![selected], vec![Placeholder; 3], vec![Placeholder])
                    .unwrap(),
            )
        };
        assert!(callee.effects().classes().is_empty());
        let mut builder = ArrayIrProgramBuilder::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let dimension = builder.add_input(m.clone().into());
        let array = builder.add_input(array_type.clone().into());
        add_jit_call(&mut builder, &callee, vec![predicate, dimension, array]);
        let selected = add_jit_call(&mut builder, &callee, vec![predicate, dimension, array]);
        let printed = builder
            .add_instruction(PrintOperation::<ArrayType>::new("after"), Vec::new(), vec![selected], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![printed], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &[ArrayType::scalar(DataType::Boolean), ArrayType::scalar(DataType::I64), array_type.clone()],
            &[array_type],
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<2xf64>) -> tensor<2xf64> {
                    %c = stablehlo.constant dense<2> : tensor<i32>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %0 = stablehlo.pad %arg2, %cst, low = [0], high = [5], interior = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<7xf64>
                    %1 = stablehlo.set_dimension_size %0, %c, dim = 0 : (tensor<7xf64>, tensor<i32>) -> tensor<?xf64, #stablehlo.bounds<7>>
                    %c_0 = stablehlo.constant dense<2> : tensor<i64>
                    %2 = stablehlo.compare EQ, %c_0, %arg1, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                    %3 = stablehlo.after_all  : !stablehlo.token
                    %4 = stablehlo.custom_call @ryft.assert(%2, %arg1, %c_0, %3) {api_version = 4 : i32, backend_config = {actor = "guard", kind = "generic", label_0 = "expected", label_1 = "actual", label_count = "2", message = "input dimension `n` must have one extent at the entry of `condition`"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                    %5 = "stablehlo.if"(%arg0) ({
                      %c_4 = stablehlo.constant dense<7> : tensor<i32>
                      %13 = stablehlo.set_dimension_size %1, %c_4, dim = 0 : (tensor<?xf64, #stablehlo.bounds<7>>, tensor<i32>) -> tensor<7xf64>
                      %14 = stablehlo.slice %13 [0:2] : (tensor<7xf64>) -> tensor<2xf64>
                      stablehlo.return %14 : tensor<2xf64>
                    }, {
                      %c_4 = stablehlo.constant dense<7> : tensor<i32>
                      %13 = stablehlo.set_dimension_size %1, %c_4, dim = 0 : (tensor<?xf64, #stablehlo.bounds<7>>, tensor<i32>) -> tensor<7xf64>
                      %14 = stablehlo.slice %13 [0:2] : (tensor<7xf64>) -> tensor<2xf64>
                      stablehlo.return %14 : tensor<2xf64>
                    }) : (tensor<i1>) -> tensor<2xf64>
                    %c_1 = stablehlo.constant dense<2> : tensor<i32>
                    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %6 = stablehlo.pad %arg2, %cst_2, low = [0], high = [5], interior = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<7xf64>
                    %7 = stablehlo.set_dimension_size %6, %c_1, dim = 0 : (tensor<7xf64>, tensor<i32>) -> tensor<?xf64, #stablehlo.bounds<7>>
                    %c_3 = stablehlo.constant dense<2> : tensor<i64>
                    %8 = stablehlo.compare EQ, %c_3, %arg1, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                    %9 = stablehlo.custom_call @ryft.assert(%8, %arg1, %c_3, %4) {api_version = 4 : i32, backend_config = {actor = "guard", kind = "generic", label_0 = "expected", label_1 = "actual", label_count = "2", message = "input dimension `n` must have one extent at the entry of `condition`"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                    %10 = stablehlo.after_all %4, %9 : !stablehlo.token
                    %11 = "stablehlo.if"(%arg0) ({
                      %c_4 = stablehlo.constant dense<7> : tensor<i32>
                      %13 = stablehlo.set_dimension_size %7, %c_4, dim = 0 : (tensor<?xf64, #stablehlo.bounds<7>>, tensor<i32>) -> tensor<7xf64>
                      %14 = stablehlo.slice %13 [0:2] : (tensor<7xf64>) -> tensor<2xf64>
                      stablehlo.return %14 : tensor<2xf64>
                    }, {
                      %c_4 = stablehlo.constant dense<7> : tensor<i32>
                      %13 = stablehlo.set_dimension_size %7, %c_4, dim = 0 : (tensor<?xf64, #stablehlo.bounds<7>>, tensor<i32>) -> tensor<7xf64>
                      %14 = stablehlo.slice %13 [0:2] : (tensor<7xf64>) -> tensor<2xf64>
                      stablehlo.return %14 : tensor<2xf64>
                    }) : (tensor<i1>) -> tensor<2xf64>
                    %12 = stablehlo.custom_call @ryft.print(%11, %10) {api_version = 4 : i32, backend_config = {label = "after"}, has_side_effect = true, operand_layouts = [dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], result_layouts = [dense<> : tensor<0xindex>]} : (tensor<2xf64>, !stablehlo.token) -> !stablehlo.token
                    return %11 : tensor<2xf64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_collect_jit_call_functions_inlines_callees_with_extent_validations() {
        // The repeated callee is pure in the source program, but its `zero[n, n]` constructor consumes a symbolic
        // `dimension<n>` and an exact `dimension<2>`, so type inference refines it to `f64[2, 2]` and lowering
        // validates that `n = 2` at runtime. A shared token-free function could not order those validations, so the
        // callee inlines at both call sites, and the explicit assertion after them continues their assertion chain.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let symbolic = builder.add_input(n.clone().into());
            let exact = builder.add_input(DimensionType::from(Dimension::Static(2)).into());
            let zero = builder
                .add_instruction(
                    Operation::Zero(ZeroOperation::new(ArrayType::new(
                        DataType::F64,
                        Shape::new(vec![n.simplified_dimension(), n.simplified_dimension()]),
                    ))),
                    Vec::new(),
                    vec![symbolic, exact],
                    None,
                )
                .unwrap()[0];
            Arc::new(
                builder
                    .build::<Vec<Constant>, Vec<Constant>>(vec![zero], vec![Placeholder; 2], vec![Placeholder])
                    .unwrap(),
            )
        };
        assert!(callee.effects().classes().is_empty());
        let mut builder = ArrayIrProgramBuilder::new();
        let symbolic = builder.add_input(n.clone().into());
        let exact = builder.add_input(DimensionType::from(Dimension::Static(2)).into());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        add_jit_call(&mut builder, &callee, vec![symbolic, exact]);
        let zero = add_jit_call(&mut builder, &callee, vec![symbolic, exact]);
        builder
            .add_instruction(AssertOperation::<ArrayType>::new("required predicate"), Vec::new(), vec![predicate], None)
            .unwrap();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![zero], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let output_type = ArrayType::new_static(DataType::F64, [2, 2]);
        let module = to_mlir_module_for_program(
            &program,
            &[],
            &[ArrayType::scalar(DataType::I64), ArrayType::scalar(DataType::I64), ArrayType::scalar(DataType::Boolean)],
            &[output_type],
            "main",
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<i1>) -> tensor<2x2xf64> {
                    %c = stablehlo.constant dense<2> : tensor<i64>
                    %0 = stablehlo.compare EQ, %arg0, %c, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                    %1 = stablehlo.after_all  : !stablehlo.token
                    %2 = stablehlo.custom_call @ryft.assert(%0, %arg0, %1) {api_version = 4 : i32, backend_config = {actor = "zero", kind = "generic", label_0 = "n", label_count = "1", message = "input dimension `n` expected 2"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x2xf64>
                    %c_0 = stablehlo.constant dense<2> : tensor<i64>
                    %4 = stablehlo.compare EQ, %arg0, %c_0, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
                    %5 = stablehlo.custom_call @ryft.assert(%4, %arg0, %2) {api_version = 4 : i32, backend_config = {actor = "zero", kind = "generic", label_0 = "n", label_count = "1", message = "input dimension `n` expected 2"}, has_side_effect = true} : (tensor<i1>, tensor<i64>, !stablehlo.token) -> !stablehlo.token
                    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
                    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<2x2xf64>
                    %7 = stablehlo.custom_call @ryft.assert(%arg2, %5) {api_version = 4 : i32, backend_config = {actor = "assert", kind = "generic", label_count = "0", message = "required predicate"}, has_side_effect = true} : (tensor<i1>, !stablehlo.token) -> !stablehlo.token
                    return %6 : tensor<2x2xf64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_jit_call_operation_with_boundary() {
        let hash = |operation: &JitCallOperation<Type>| {
            let mut hasher = DefaultHasher::new();
            operation.hash(&mut hasher);
            hasher.finish()
        };
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let device_mesh = DeviceMesh::new(mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)]).unwrap();
        let reversed_mesh = DeviceMesh::new(mesh.clone(), vec![Device::new(1, 0), Device::new(0, 0)]).unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh, 1);

        // A call of a callee that names no devices, including one whose lists hold no override, equals and hashes like a
        // call whose fields hash as the derived implementation did before boundaries existed.
        let plain = JitCallOperation::<Type>::new(1);
        let mut expected = DefaultHasher::new();
        1usize.hash(&mut expected);
        PhantomData::<fn() -> Type>.hash(&mut expected);
        None::<CallTypeBindings<Type>>.hash(&mut expected);
        assert_eq!(hash(&plain), expected.finish());
        let empty = JitCallOperation::<Type>::new(1).with_boundary(None, vec![None, None], vec![None]);
        assert_eq!(empty, plain);
        assert_eq!(empty.mesh(), None);
        assert_eq!(hash(&empty), hash(&plain));

        // A callee that names devices without overrides (e.g., because its body places a value) keeps its mesh, and a
        // call of a `jit` dispatcher records that it keeps the placements of its inputs. Both take part in equality and
        // hashing.
        let pinned = JitCallOperation::<Type>::new(1).with_boundary(Some(&device_mesh), vec![None, None], vec![None]);
        assert_ne!(pinned, plain);
        assert_ne!(hash(&pinned), hash(&plain));
        assert_eq!(pinned.mesh(), Some(&device_mesh));
        assert_eq!(pinned.boundary_shardings().count(), 0);
        let dispatched = JitCallOperation::<Type>::new(1).with_kept_input_placements();
        assert_ne!(dispatched, plain);
        assert_ne!(hash(&dispatched), hash(&plain));
        assert!(dispatched.keeps_input_placements());
        assert!(!plain.keeps_input_placements());

        // Overrides and the devices of their mesh take part in equality and hashing, and render only when present.
        let placed = JitCallOperation::<Type>::new(1).with_boundary(
            Some(&device_mesh),
            vec![None, Some(sharded.clone())],
            vec![Some(replicated.clone())],
        );
        let moved = JitCallOperation::<Type>::new(1).with_boundary(
            Some(&reversed_mesh),
            vec![None, Some(sharded.clone())],
            vec![Some(replicated.clone())],
        );
        assert_ne!(placed, plain);
        assert_ne!(hash(&placed), hash(&plain));
        assert_ne!(placed, moved);
        assert_ne!(hash(&placed), hash(&moved));
        assert_eq!(placed.mesh(), Some(&device_mesh));
        assert_eq!(placed.input_sharding(0), None);
        assert_eq!(placed.input_sharding(1), Some(&sharded));
        assert_eq!(placed.input_sharding(2), None);
        assert_eq!(placed.output_sharding(0), Some(&replicated));
        assert_eq!(placed.boundary_shardings().collect::<Vec<_>>(), vec![&sharded, &replicated]);
        let input_type: Type = ArrayType::new_static(DataType::F32, [4]).into();
        let mut builder = ArrayIrProgramBuilder::new();
        builder.add_input(input_type.clone());
        let input = builder.add_input(input_type.clone());
        let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let callee = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![sine], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let render = |operation: JitCallOperation<Type>| {
            let mut builder = ArrayIrProgramBuilder::new();
            let callee_region = builder.import_region(callee.entry_region_ref());
            let inputs = vec![builder.add_input(input_type.clone()), builder.add_input(input_type.clone())];
            let output =
                builder.add_instruction(Operation::JitCall(operation), vec![callee_region], inputs, None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
                .to_string()
        };
        assert_eq!(
            render(plain),
            indoc! {"
                lambda %0:f32[4], %1:f32[4] .
                let %2:f32[4] = jit_call [capture_count=1] %0 %1 [
                    callee={
                        lambda %0:f32[4], %1:f32[4] .
                        let %2:f32[4] = sin %1
                        in (%2)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(
            render(placed),
            indoc! {"
                lambda %0:f32[4], %1:f32[4] .
                let %2:f32[4] = jit_call [
                    capture_count=1,
                    input_shardings=[_, {mesh<['x'=2:auto]>, [{'x'}]}],
                    output_shardings=[{mesh<['x'=2:auto]>, [{}]}],
                    devices=[0, 1],
                ] %0 %1 [
                    callee={
                        lambda %0:f32[4], %1:f32[4] .
                        let %2:f32[4] = sin %1
                        in (%2)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(
            render(pinned.with_kept_input_placements()),
            indoc! {"
                lambda %0:f32[4], %1:f32[4] .
                let %2:f32[4] = jit_call [capture_count=1, devices=[0, 1], keeps_input_placements=true] %0 %1 [
                    callee={
                        lambda %0:f32[4], %1:f32[4] .
                        let %2:f32[4] = sin %1
                        in (%2)
                    },
                ]
                in (%2)"},
        );
    }

    #[test]
    fn test_jit_call_operation_with_callee_region() {
        let input_type: Type = ArrayType::scalar(DataType::F32).into();
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(input_type.clone());
        let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let callee = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![sine], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let operation = JitCallOperation::new(1).with_callee_region(callee.entry_region_ref()).unwrap();
        assert_eq!(
            operation,
            JitCallOperation::new(1)
                .with_callee_type_signature(&callee.input_types(), &callee.output_types())
                .unwrap(),
        );
        let mut builder = ArrayIrProgramBuilder::new();
        let callee_region = builder.import_region(callee.entry_region_ref());
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(Operation::JitCall(operation), vec![callee_region], vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = jit_call [capture_count=1] %0 [
                    callee={
                        lambda %0:f32[] .
                        let %1:f32[] = sin %0
                        in (%1)
                    },
                ]
                in (%1)"},
        );
        assert_eq!(
            JitCallOperation::new(2).with_callee_region(callee.entry_region_ref()),
            Err(TypeError::invalid("`jit_call` lifted capture prefix 2 exceeds its 1 callee inputs")),
        );

        // Generated callees can introduce local identities even when the source that was transformed had none.
        // The sealed signature records those definitions, so each new call still receives a fresh caller result.
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(ArrayType::scalar(DataType::I64).into());
        let local = DimensionType::new("local", DimensionBounds::non_negative(Some(8)).unwrap());
        let output = builder
            .add_instruction(DimensionFromScalarOperation::new(local), Vec::new(), vec![input], None)
            .unwrap()[0];
        let identity_callee = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let first = JitCallOperation::new(0).with_callee_region(identity_callee.entry_region_ref()).unwrap();
        let second = JitCallOperation::new(0).with_callee_region(identity_callee.entry_region_ref()).unwrap();
        let original = JitCallOperation::new(0)
            .with_callee_type_signature(&identity_callee.input_types(), &identity_callee.output_types())
            .unwrap();
        assert_eq!(first.defined_type_identities().len(), 1);
        assert_ne!(first.defined_type_identities(), second.defined_type_identities());
        assert_ne!(
            first.defined_type_identities().as_ref(),
            identity_callee.type_identity_signature().internal_identities()
        );
        let interface = identity_callee.entry_region_ref().interface();
        let first_outputs =
            first.infer_output_types(&identity_callee.input_types(), std::slice::from_ref(interface)).unwrap();
        let repeated_outputs =
            first.infer_output_types(&identity_callee.input_types(), std::slice::from_ref(interface)).unwrap();
        assert_eq!(first_outputs, repeated_outputs);
        assert_ne!(first_outputs, identity_callee.output_types().into_owned());
        assert_eq!(
            first.infer_output_types(&identity_callee.input_types(), std::slice::from_ref(interface)).unwrap()[0]
                .to_string(),
            original
                .infer_output_types(&identity_callee.input_types(), std::slice::from_ref(identity_callee.interface()))
                .unwrap()[0]
                .to_string(),
        );
        // Replacing a previous identity-bearing declaration with an identity-free callee clears that declaration,
        // exactly as the public signature constructor does.
        assert_eq!(
            first.with_callee_region(callee.entry_region_ref()).unwrap(),
            JitCallOperation::new(0)
                .with_callee_type_signature(&callee.input_types(), &callee.output_types())
                .unwrap(),
        );
    }

    #[test]
    fn test_jit_call_operation_dense_result_bindings() {
        let bounds = DimensionBounds::non_negative(Some(8)).unwrap();
        let input = DimensionType::new("input", bounds);
        let local = DimensionType::new("local", bounds);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![input.dimension().clone()]));
        let output_type =
            ArrayType::new(DataType::F32, Shape::new(vec![input.dimension().clone(), local.dimension().clone()]));
        let output_types = vec![output_type.clone(), output_type];
        let operation = JitCallOperation::<ArrayType>::new(0)
            .with_callee_type_signature(std::slice::from_ref(&input_type), &output_types)
            .unwrap();
        let other = JitCallOperation::<ArrayType>::new(0)
            .with_callee_type_signature(std::slice::from_ref(&input_type), &output_types)
            .unwrap();
        let interface = RegionInterface::new(vec![input_type.clone()], output_types.clone(), EffectClasses::NONE);
        let outputs = operation
            .infer_output_types(std::slice::from_ref(&input_type), std::slice::from_ref(&interface))
            .unwrap();
        assert_eq!(outputs[0], outputs[1]);
        assert_ne!(outputs, output_types);
        assert_eq!(
            operation
                .infer_output_types(std::slice::from_ref(&input_type), std::slice::from_ref(&interface))
                .unwrap(),
            outputs
        );
        assert_ne!(other.infer_output_types(std::slice::from_ref(&input_type), &[interface]).unwrap(), outputs);
        let actual = ArrayType::new_static(DataType::F32, [2]);
        let callee_outputs = output_types
            .iter()
            .map(|output| {
                ArrayType::new(DataType::F32, Shape::new(vec![2usize.into(), output.shape().dimensions()[1].clone()]))
            })
            .collect::<Vec<_>>();
        let actual_interface = RegionInterface::new(vec![actual.clone()], callee_outputs, EffectClasses::NONE);
        let instantiated = operation.infer_output_types(&[actual], &[actual_interface]).unwrap();
        assert_eq!(instantiated[0].shape().dimensions()[0], 2usize.into());
        assert_eq!(instantiated[0].shape().dimensions()[1], outputs[0].shape().dimensions()[1]);
        assert_eq!(instantiated[0], instantiated[1]);
        assert_eq!(operation.defined_type_identities().len(), 1);
        let retained = JitCallOperation::<ArrayType>::new(0)
            .with_result_type_signature(std::slice::from_ref(&input_type), &output_types, outputs.clone())
            .unwrap();
        assert_eq!(retained.instantiated_output_types(&[input_type]).unwrap(), Some(outputs));
    }

    #[test]
    fn test_jit_call_supports_array_ir_region_boundaries() {
        let dimension_type = Type::Dimension(DimensionType::new("size", DimensionBounds::positive(Some(9)).unwrap()));
        let array_type = Type::Array(vector_type());
        let interface = RegionInterface::new(
            vec![dimension_type.clone()],
            vec![array_type.clone(), dimension_type.clone()],
            EffectClasses::NONE,
        );
        let operation = JitCallOperation::new(0);

        assert_eq!(operation.name(), JIT_CALL_OPERATION_NAME);
        assert_eq!(
            operation.infer_output_types(&[], &[]),
            Err(TypeError::invalid("`jit_call` expects 1 attached callee region but got 0")),
        );
        assert_eq!(
            operation
                .infer_output_types(std::slice::from_ref(&dimension_type), std::slice::from_ref(&interface))
                .unwrap(),
            vec![array_type.clone(), dimension_type.clone()],
        );

        // The call operands must match the callee boundary, which must cover the lifted capture prefix.
        assert_eq!(
            JitCallOperation::new(2)
                .infer_output_types(std::slice::from_ref(&dimension_type), std::slice::from_ref(&interface)),
            Err(TypeError::invalid("`jit_call` lifted capture prefix 2 exceeds its 1 callee inputs")),
        );
        assert_eq!(
            operation.infer_output_types(&[], std::slice::from_ref(&interface)),
            Err(TypeError::invalid("`jit_call` expected 1 input(s) but got 0")),
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&array_type), std::slice::from_ref(&interface)),
            Err(TypeError::invalid(format!("`jit_call` input #0 expected {dimension_type} but got {array_type}"))),
        );

        // A callee that defines its own result identities needs caller-owned result names.
        let local_interface = RegionInterface::new(Vec::new(), vec![dimension_type], EffectClasses::NONE);
        assert_eq!(
            operation.infer_output_types(&[], &[local_interface]),
            Err(TypeError::invalid(
                "`jit_call` with callee-local result identities requires `with_callee_type_signature`",
            )),
        );
    }

    #[test]
    fn test_jit_call_operation_region_provenance() {
        // A `jit_call` operation forwards its operands to the callee positionally, so region provenance, capture
        // counts, and output provenance are all index-preserving for the single callee region and absent for any other
        // region.
        let operation = JitCallOperation::<Type>::new(2);
        assert_eq!(operation.input_region_provenance(0, 3), InputRegionProvenance::Input { index: 3 });
        assert_eq!(operation.input_region_provenance(1, 3), InputRegionProvenance::None);
        assert_eq!(operation.region_capture_input_count(0), Some(2));
        assert_eq!(operation.region_capture_input_count(1), None);
        assert_eq!(
            std::fmt::from_fn(|formatter| operation.render(formatter, 0)).to_string(),
            "jit_call [capture_count=2]",
        );
        assert_eq!(
            operation.output_region_provenance(2),
            vec![OutputRegionProvenance { region_index: 0, output_index: 2 }],
        );
    }

    #[test]
    fn test_jit_call_operation_region_data_flow() {
        let operation = JitCallOperation::<Type>::new(2);
        assert!(matches!(operation.region_data_flow(), RegionDataFlow::Provenance));
        let dispatched = Operation::<Constant>::from(operation);
        assert!(matches!(dispatched.region_data_flow(), RegionDataFlow::Provenance));
    }

    #[test]
    fn test_jit_call_operation_prune_boundary() {
        // The callee maps its lifted capture `c` and its operands `(x, y)` to `(cos(x), c * y)`. Using only the first
        // output drops the operand `y` and the second output, while the capture prefix is kept although nothing uses
        // it.
        let r#type = Type::Array(vector_type());
        let callee = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let captured = builder.add_input(r#type.clone());
            let x = builder.add_input(r#type.clone());
            let y = builder.add_input(r#type.clone());
            let cosine = builder.add_instruction(CosOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
            let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![captured, y], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![cosine, product],
                    vec![Placeholder; 3],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let inputs = (0..3).map(|_| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let outputs = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(1)), vec![callee], inputs, None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![outputs[0]], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.into_pruned().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4], %2:f64[4] .
                let %3:f64[4] = jit_call [capture_count=1] %0 %1 [
                    callee={
                        lambda %0:f64[4], %1:f64[4] .
                        let %2:f64[4] = cos %1
                        in (%2)
                    },
                ]
                in (%3)"},
        );
    }

    #[test]
    fn test_jit_call_reference_discharge_threads_nested_callee_state_into_the_caller() {
        // Two nested call levels each swap a distinct reference. Discharge must lift both states to the caller's
        // boundary in public-input order, mark both as mutated with their own final-state output slot, and leave no
        // reference-typed atom anywhere in the resulting program.
        let reference_type = ReferenceType::new(vector_type());
        let inner_callee = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            let replacement = builder.add_input(vector_type().into());
            let snapshot = builder
                .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let callee = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let inner_callee = builder.import_region(inner_callee.entry_region_ref());
            let first_reference = builder.add_input(reference_type.clone().into());
            let first_replacement = builder.add_input(vector_type().into());
            let second_reference = builder.add_input(reference_type.clone().into());
            let second_replacement = builder.add_input(vector_type().into());
            let first_snapshot = builder
                .add_instruction(
                    Operation::JitCall(JitCallOperation::new(0)),
                    vec![inner_callee],
                    vec![first_reference, first_replacement],
                    None,
                )
                .unwrap()[0];
            let second_snapshot = builder
                .add_instruction(
                    Operation::JitCall(JitCallOperation::new(0)),
                    vec![inner_callee],
                    vec![second_reference, second_replacement],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![first_snapshot, second_snapshot],
                    vec![Placeholder; 4],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };

        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let first_reference = builder.add_input(reference_type.clone().into());
        let first_replacement = builder.add_input(vector_type().into());
        let second_reference = builder.add_input(reference_type.into());
        let second_replacement = builder.add_input(vector_type().into());
        let snapshots = builder
            .add_instruction(
                Operation::JitCall(JitCallOperation::new(0)),
                vec![callee],
                vec![first_reference, first_replacement, second_reference, second_replacement],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(snapshots, vec![Placeholder; 4], vec![Placeholder; 2])
            .unwrap();

        let discharged = program.discharge_references(0).unwrap();
        assert_eq!(discharged.output_count(), 2);
        assert_eq!(discharged.program().output_count(), 4);
        assert_eq!(discharged.external_reference_bindings().len(), 2);
        assert_eq!(discharged.external_reference_bindings()[0].source(), ReferenceSource::Input { index: 0 });
        assert!(discharged.external_reference_bindings()[0].is_mutated());
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), Some(2));
        assert_eq!(discharged.external_reference_bindings()[1].source(), ReferenceSource::Input { index: 2 });
        assert!(discharged.external_reference_bindings()[1].is_mutated());
        assert_eq!(discharged.external_reference_bindings()[1].output_index(), Some(3));
    }

    #[test]
    fn test_jit_call_reference_discharge_resolves_nested_callee_lexical_capture_scope() {
        // A branch region inside the callee reaches its reference through a capture constant, which must resolve
        // against the callee's own capture scope rather than the entry program's lifted prefix. The accessed root is
        // the caller's second public input, and only that mutated state gains a final-state output.
        let reference_type = ReferenceType::new(vector_type());
        let branch = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let replacement = builder.add_input(vector_type().into());
            let reference = builder
                .add_constant(Constant::Captured(CaptureReference::new(0, Type::Reference(reference_type.clone()))));
            let snapshot = builder
                .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let callee = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let branch = builder.import_region(branch.entry_region_ref());
            builder.add_input(reference_type.clone().into());
            let replacement = builder.add_input(vector_type().into());
            let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
            let snapshot = builder
                .add_instruction(ConditionOperation::new(), vec![branch, branch], vec![predicate, replacement], None)
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![snapshot], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };

        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        builder.add_input(reference_type.clone().into());
        let reference = builder.add_input(reference_type.into());
        let replacement = builder.add_input(vector_type().into());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let snapshot = builder
            .add_instruction(
                Operation::JitCall(JitCallOperation::new(1)),
                vec![callee],
                vec![reference, replacement, predicate],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![snapshot], vec![Placeholder; 4], vec![Placeholder])
            .unwrap();

        let discharged = program.discharge_references_in_capture_lifted_program(0).unwrap();
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(discharged.program().output_count(), 2);
        assert_eq!(discharged.external_reference_bindings().len(), 2);
        assert_eq!(discharged.external_reference_bindings()[0].source(), ReferenceSource::Input { index: 0 });
        assert!(!discharged.external_reference_bindings()[0].is_mutated());
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), None);
        assert_eq!(discharged.external_reference_bindings()[1].source(), ReferenceSource::Input { index: 1 });
        assert!(discharged.external_reference_bindings()[1].is_mutated());
        assert_eq!(discharged.external_reference_bindings()[1].output_index(), Some(1));
    }

    #[test]
    fn test_jit_call_reference_discharge_preserves_forwarded_root_identity() {
        let reference_type = ReferenceType::new(vector_type());
        let mut callee_builder = CoreProgramBuilder::<Constant, Operation>::new();
        let reference = callee_builder.add_input(reference_type.clone().into());
        let callee = callee_builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        let forwarded = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee], vec![reference], None)
            .unwrap()[0];
        let value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![forwarded], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // The callee only forwards its reference input, so discharge must keep the forwarded value bound to the same
        // root as the caller's public reference input instead of synthesizing a second state slot for it. Reading
        // through the forwarded result is not a mutation, so no final-state output is appended.
        let discharged = program.discharge_references(0).unwrap();
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(discharged.program().output_count(), 1);
        assert_eq!(discharged.external_reference_bindings().len(), 1);
        assert_eq!(discharged.external_reference_bindings()[0].source(), ReferenceSource::Input { index: 0 });
        assert!(!discharged.external_reference_bindings()[0].is_mutated());
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), None);
    }

    #[test]
    fn test_jit_call_reference_discharge_widens_its_callee_with_the_final_state() {
        // `jit_call` forwards its operands onto its callee one for one and reports the callee's outputs as its own, so
        // the shared positional rewrite threads state through it. The callee gains the entering state after its
        // declared inputs and publishes the final state after its declared outputs, while the call site keeps its own
        // operand and output contract. The callee mutates the root and returns only the previous snapshot, so the
        // final state is exactly what the call site would otherwise have lost.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut callee_builder = CoreProgramBuilder::<Constant, Operation>::new();
        let callee_reference = callee_builder.add_input(reference_type.into());
        let callee_update = callee_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let previous = callee_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![callee_reference, callee_update], None)
            .unwrap()[0];
        let callee = callee_builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![previous], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let callee = builder.import_program(callee);
        let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let update = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let root = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let previous = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee], vec![root, update], None)
            .unwrap()[0];
        let frozen = builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![root], None).unwrap()[0];
        let source = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![previous, frozen], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        // The callee's local root is allocated, mutated, and consumed inside the call, so widening the boundary
        // leaves no external state behind and the rewritten call is the entry region's only instruction.
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.external_reference_bindings(), &[]);
        assert_eq!(discharged.program().output_types().len(), 2);
        assert_eq!(discharged.program().entry_region_ref().instructions().len(), 1);
    }

    #[test]
    fn test_jit_call_reference_discharge_instantiates_callee_local_results() {
        // The callee mutates a reference and returns a dimension that it defines itself, so the call instantiates a
        // fresh result identity. Discharge widens the callee with the reference state, which must not leave the
        // call's result metadata pinned to the narrower pre-discharge callee signature.
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let logical_mesh = mesh.logical_mesh().clone();
        let session = Session::builder().client(client).build().unwrap();
        let value_type = ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(logical_mesh.clone(), 0));
        let value_type = value_type.unwrap();
        let extent_type = ArrayType::scalar(DataType::I64).with_sharding(Sharding::replicated(logical_mesh, 0));
        let extent_type = extent_type.unwrap();
        let rows = DimensionType::from(DimensionVariable::new("rows", DimensionBounds::non_negative(Some(8)).unwrap()));

        let mut callee_builder = CoreProgramBuilder::<Constant, Operation>::new();
        let reference = callee_builder.add_input(ReferenceType::new(value_type.clone()).into());
        let update = callee_builder.add_input(value_type.clone().into());
        let extent = callee_builder.add_input(extent_type.clone().into());
        let previous = callee_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap()[0];
        let dimension = callee_builder
            .add_instruction(DimensionFromScalarOperation::new(rows.clone()), Vec::new(), vec![extent], None)
            .unwrap()[0];
        let callee = callee_builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![previous, dimension],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();
        let operation = JitCallOperation::new(0)
            .with_callee_type_signature(&callee.input_types(), &callee.output_types())
            .unwrap();

        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let callee = builder.import_program(callee);
        let initial = builder.add_input(value_type.clone().into());
        let update = builder.add_input(value_type.clone().into());
        let extent = builder.add_input(extent_type.clone().into());
        let root = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let outputs = builder
            .add_instruction(Operation::JitCall(operation), vec![callee], vec![root, update, extent], None)
            .unwrap()
            .to_vec();
        let frozen = builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![root], None).unwrap()[0];
        let extent =
            builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![outputs[1]], None).unwrap()[0];
        let source = builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![outputs[0], extent, frozen],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();

        let program = source.discharge_references(0).unwrap().into_program_without_external_references().unwrap();
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda %0:{value_type}, %1:{value_type}, %2:{extent_type} .
                let %3:{value_type}, %4:dimension<rows ∈ [0, 8)>, %5:{value_type} = jit_call [
                    output_types=[{value_type}, dimension<rows ∈ [0, 8)>, {value_type}],
                ] %0 %1 %2 [
                    callee={{
                        lambda %0:{value_type}, %1:{value_type}, %2:{extent_type} .
                        let %3:dimension<rows ∈ [0, 8)> = dimension_from_scalar [bounds=[0, 8)] %2
                        in (%0, %3, %1)
                    }},
                ]
                    %6:i64[] = dimension_to_scalar %4
                in (%3, %6, %5)"},
        );
        // The call owns a fresh result identity, distinct from the callee-local identity it instantiates.
        let call = &program.entry_region_ref().instructions()[0];
        assert_ne!(program.atoms()[call.outputs()[1].index()].r#type().as_ref(), &Type::Dimension(rows));

        // Execute the discharged program on CPU, with the call staged inside one compiled function.
        let compiled: CompiledFunction<(ArrayType, ArrayType, ArrayType), (ArrayType, ArrayType, ArrayType)> =
            crate::compile(
                |(initial, update, extent)| {
                    let initial = initial.into_value();
                    let inputs = vec![initial.clone(), update.into_value(), extent.into_value()];
                    let outputs = program.interpret_in_context(initial.context(), inputs).unwrap();
                    let mut outputs =
                        outputs.into_iter().map(|output| ValueProjection::<ArrayType>::into_projected(output).unwrap());
                    (outputs.next().unwrap(), outputs.next().unwrap(), outputs.next().unwrap())
                },
                (value_type.clone(), value_type.clone(), extent_type.clone()),
                &session,
                mesh.clone(),
            )
            .unwrap();
        let array = |r#type: &ArrayType, bytes: Vec<u8>| {
            Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), None, bytes).unwrap()
        };
        let (previous, extent, state) = compiled
            .executable()
            .execute(
                &session,
                (
                    array(&value_type, values_to_bytes(&[3f32])),
                    array(&value_type, values_to_bytes(&[4f32])),
                    array(&extent_type, values_to_bytes(&[5i64])),
                ),
            )
            .unwrap();
        let read = |array: &Array| {
            array
                .addressable_shards()
                .next()
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap()
        };
        assert_eq!(values_from_bytes::<f32>(&read(&previous)), vec![3f32]);
        assert_eq!(values_from_bytes::<i64>(&read(&extent)), vec![5i64]);
        assert_eq!(values_from_bytes::<f32>(&read(&state)), vec![4f32]);
    }

    #[test]
    fn test_jit_call_residual_candidates_classify_through_callee_provenance() {
        // `jit_call` reports positional output-region provenance, so a residual produced by a computed callee output
        // is classified through the callee to that output's own producer instead of to the opaque `jit_call`. The
        // program squares its known input `x` inside the callee and multiplies the result of the call by its unknown
        // input `y` outside of it, so the residual program needs the result of the call.
        let scalar_type = Type::from(ArrayType::scalar(DataType::F64));
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(scalar_type.clone());
            let squared =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![squared], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let callee = builder.import_program(callee);
        let x = builder.add_input(scalar_type.clone());
        let y = builder.add_input(scalar_type);
        let called = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee], vec![x], None)
            .unwrap()[0];
        let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![called, y], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let names = Arc::new(Mutex::new(Vec::new()));
        let recorded = names.clone();
        let policy = RematerializationPolicyFn::new::<Type>(move |candidate| {
            let names = candidate.producers().iter().map(|producer| producer.name().to_string());
            recorded.lock().unwrap().extend(names);
            Ok::<_, ResidualRejection>(ResidualDecision::<NoStorage>::Save)
        });
        program
            .partition(&[true, false])
            .unwrap()
            .with_residual_policy(&ResidualPolicyReference::new(policy))
            .unwrap();
        assert_eq!(names.lock().unwrap().clone(), vec!["mul".to_string()]);
    }

    #[test]
    fn test_jit_call_partial_evaluation_keeps_reference_bearing_callees_whole() {
        // The callee `f(r, x) = { a = read(r); r += x; a * read(r) }` reads the reference before and after mutating it.
        // Splitting it against a known `r` would hoist the first read into the outer program and leave the mutation
        // and the second read behind the residual call, so the call stays whole instead.
        let scalar_type = Type::Array(ArrayType::scalar(DataType::F32));
        let reference_type = Type::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let reference = builder.add_input(reference_type.clone());
            let update = builder.add_input(scalar_type.clone());
            let before = builder
                .add_instruction(
                    Operation::ReferenceRead(ReferenceReadOperation::new()),
                    Vec::new(),
                    vec![reference],
                    None,
                )
                .unwrap()[0];
            builder
                .add_instruction(
                    Operation::ReferenceAddUpdate(ReferenceAddUpdateOperation::new()),
                    Vec::new(),
                    vec![reference, update],
                    None,
                )
                .unwrap();
            let after = builder
                .add_instruction(
                    Operation::ReferenceRead(ReferenceReadOperation::new()),
                    Vec::new(),
                    vec![reference],
                    None,
                )
                .unwrap()[0];
            let output =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![before, after], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let reference = builder.add_input(reference_type.clone());
        let update = builder.add_input(scalar_type.clone());
        let output = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee], vec![reference, update], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let outer = TracingContext::<Constant, Operation>::new();
        let known = outer.input(reference_type);
        let evaluation = program
            .partially_evaluate_in_context(&outer, &[PartialValue::Known(known), PartialValue::Unknown(scalar_type)])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:ref<f32[]> .
                let %2:f32[] = jit_call %1 %0 [
                    callee={
                        lambda %0:ref<f32[]>, %1:f32[] .
                        let %2:f32[] = reference_read %0
                            () = reference_add_update %0 %1
                            %3:f32[] = reference_read %0
                            %4:f32[] = mul %2 %3
                        in (%4)
                    },
                ]
                in (%2)"},
        );
        assert!(outer.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_jit_call_partial_evaluation_keeps_escaping_allocation_callees_whole() {
        // The callee `f(k, x) = (reference_new(k), k * x)` allocates a reference from the known input and returns it
        // beside a runtime-dependent product. The allocation is computable from known inputs alone, but it escapes
        // through a reference output whose later accesses the callee cannot see, so it is live state rather than dead
        // intrinsic state, and the call stays whole instead of hoisting the allocation into the known side.
        let r#type = Type::Array(vector_type());
        let callee = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let known_input = builder.add_input(r#type.clone());
            let runtime_input = builder.add_input(r#type.clone());
            let reference =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![known_input], None).unwrap()[0];
            let product = builder
                .add_instruction(MulOperation::new(), Vec::new(), vec![known_input, runtime_input], None)
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![reference, product],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let known_input = builder.add_input(r#type.clone());
        let runtime_input = builder.add_input(r#type.clone());
        let callee_region = builder.intern_callee(&Arc::new(callee), None).unwrap();
        let outputs = builder
            .add_instruction(
                Operation::JitCall(JitCallOperation::new(0)),
                vec![callee_region],
                vec![known_input, runtime_input],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let outer = TracingContext::<Constant, Operation>::new();
        let known = outer.input(r#type.clone());
        let evaluation = program
            .partially_evaluate_in_context(&outer, &[PartialValue::Known(known), PartialValue::Unknown(r#type)])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4] .
                let %2:ref<f64[4]>, %3:f64[4] = jit_call %1 %0 [
                    callee={
                        lambda %0:f64[4], %1:f64[4] .
                        let %2:ref<f64[4]> = reference_new %0
                            %3:f64[4] = mul %0 %1
                        in (%2, %3)
                    },
                ]
                in (%2, %3)"},
        );
        assert!(outer.builder().borrow().instructions().is_empty());
    }

    // Online partial evaluation of a mixed `jit_call` against a live outer trace, which is the second recorded consumer
    // of parent-context-polymorphic partial evaluation. The known half of the callee, including a callee literal that
    // it consumes, is wrapped again as a known-side `jit_call` staged into the outer program over the symbolic known
    // input. The unknown half stays behind a residual `jit_call` whose literal is rebuilt inline, and the known→unknown
    // residual edge flows from the known-side call's outputs into the residual call's inputs.
    #[test]
    fn test_jit_call_partial_evaluation_splits_callee_against_a_live_outer_trace() {
        let r#type = Type::Array(vector_type());

        // Callee `f(a, x) = (a + a, x * x, (a + a) * x)` over a known `a` and an unknown `x`. Capture constants are
        // deliberately absent, because a capture-bearing callee preserves its boundary instead of splitting, which
        // `test_jit_call_partial_evaluation_preserves_boundary_for_capture_bearing_callees` pins.
        let callee = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let known_input = builder.add_input(r#type.clone());
            let runtime_input = builder.add_input(r#type.clone());
            let shifted = builder
                .add_instruction(AddOperation::new(), Vec::new(), vec![known_input, known_input], None)
                .unwrap()[0];
            let scaled = builder
                .add_instruction(MulOperation::new(), Vec::new(), vec![runtime_input, runtime_input], None)
                .unwrap()[0];
            let product = builder
                .add_instruction(MulOperation::new(), Vec::new(), vec![shifted, runtime_input], None)
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![shifted, scaled, product],
                    vec![Placeholder; 2],
                    vec![Placeholder; 3],
                )
                .unwrap()
        };

        // Enclosing program staging one call to the callee over `[a, x]`.
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let known_input = builder.add_input(r#type.clone());
        let runtime_input = builder.add_input(r#type.clone());
        let callee_region = builder.intern_callee(&Arc::new(callee), None).unwrap();
        let call = Operation::JitCall(JitCallOperation::new(0));
        let outputs = builder
            .add_instruction(call, vec![callee_region], vec![known_input, runtime_input], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 3])
            .unwrap();

        let outer = TracingContext::<Constant, Operation>::new();
        let known = outer.input(r#type.clone());
        let evaluation = program
            .partially_evaluate_in_context(&outer, &[PartialValue::Known(known), PartialValue::Unknown(r#type.clone())])
            .unwrap();

        // The known half lands in the outer program as one known-side `jit_call` over the symbolic known input, which
        // produces the fully known callee output once, and the unknown half stays behind one residual `jit_call` over
        // the unknown input and the residual edge fed from that output.
        let known_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Constant>, Vec<Constant>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        assert_eq!(
            known_program.to_string(),
            indoc! {"
                lambda %0:f64[4] .
                let %1:f64[4] = jit_call %0 [
                    callee={
                        lambda %0:f64[4] .
                        let %1:f64[4] = add %0 %0
                        in (%1)
                    },
                ]
                in ()"},
        );
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4] .
                let %2:f64[4], %3:f64[4] = jit_call %0 %1 [
                    callee={
                        lambda %0:f64[4], %1:f64[4] .
                        let %2:f64[4] = mul %0 %0
                            %3:f64[4] = mul %1 %0
                        in (%2, %3)
                    },
                ]
                in (%2, %3)"},
        );

        // In the boundary descriptors, the unknown enclosing input feeds the residual call, the residual edge is a
        // known feeder naming the known-side call's staged output, and the outputs reassemble in original order.
        let known_output = known_program.instructions()[0].outputs()[0];
        assert!(matches!(
            evaluation.inputs(),
            [PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(value)]
                if value.atom_id().ok() == Some(known_output),
        ));
        assert!(matches!(
            evaluation.outputs(),
            [PartialEvaluationOutput::Known(value), PartialEvaluationOutput::Unknown(0), PartialEvaluationOutput::Unknown(1)]
                if value.atom_id().ok() == Some(known_output),
        ));
    }

    #[test]
    fn test_jit_call_partial_evaluation_splits_concrete_known_inputs() {
        // Partial evaluation splits a mixed `jit_call` whose known inputs are concrete arrays of the eager XLA session
        // exactly as it splits one whose known inputs are symbolic. The known-side call compiles and executes
        // immediately, so the output `a + a` is known and concrete, and the residual call keeps only `(a + a) * x`,
        // fed by that known output (JAX's `out_fwd`).
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let logical_mesh = mesh.logical_mesh().clone();
        let session = Session::builder().client(client).build().unwrap();
        let array_type = vector_type().with_sharding(Sharding::replicated(logical_mesh, 1)).unwrap();
        let r#type = Type::Array(array_type.clone());

        // Callee `f(a, x) = (a + a, (a + a) * x)` over a known `a` and an unknown `x`.
        let callee = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let a = builder.add_input(r#type.clone());
            let x = builder.add_input(r#type.clone());
            let doubled = builder.add_instruction(AddOperation::new(), Vec::new(), vec![a, a], None).unwrap()[0];
            let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![doubled, x], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![doubled, product],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let a = builder.add_input(r#type.clone());
        let x = builder.add_input(r#type.clone());
        let callee_region = builder.intern_callee(&Arc::new(callee), None).unwrap();
        let outputs = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee_region], vec![a, x], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let array = |values: &[f64]| {
            let bytes = values_to_bytes::<f64>(values);
            Value::Array(Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, bytes).unwrap())
        };
        let read = |value: &Value| {
            let Value::Array(array) = value else {
                panic!("expected an array value");
            };
            let buffer = array.addressable_shards().next().unwrap().buffer().unwrap();
            values_from_bytes::<f64>(buffer.copy_to_host(None).unwrap().r#await().unwrap().as_slice())
        };
        let evaluation = program
            .partially_evaluate_in_context(
                &session,
                &[PartialValue::Known(array(&[1.0, 2.0, 3.0, 4.0])), PartialValue::Unknown(r#type.clone())],
            )
            .unwrap();
        let [PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(residual)] = evaluation.inputs() else {
            panic!("expected the unknown input followed by the forwarded known output");
        };
        let [PartialEvaluationOutput::Known(doubled), PartialEvaluationOutput::Unknown(0)] = evaluation.outputs()
        else {
            panic!("expected one known output followed by one unknown output");
        };
        assert_eq!(read(doubled), vec![2.0, 4.0, 6.0, 8.0]);
        assert_eq!(read(residual), vec![2.0, 4.0, 6.0, 8.0]);
        assert_eq!(
            evaluation.program().to_string(),
            formatdoc! {"
                lambda %0:{array_type}, %1:{array_type} .
                let %2:{array_type} = jit_call %0 %1 [
                    callee={{
                        lambda %0:{array_type}, %1:{array_type} .
                        let %2:{array_type} = mul %1 %0
                        in (%2)
                    }},
                ]
                in (%2)"},
        );
        let outputs = evaluation.interpret(&session, &[array(&[5.0, 6.0, 7.0, 8.0])]).unwrap();
        assert_eq!(
            outputs.iter().map(read).collect::<Vec<_>>(),
            vec![vec![2.0, 4.0, 6.0, 8.0], vec![10.0, 24.0, 42.0, 64.0]],
        );
    }

    #[test]
    fn test_jit_call_partial_evaluation_keeps_forced_copies_under_specialization() {
        // Specializing a program never folds a copy that every execution must make (i.e., a `place` with `may_alias`
        // disabled), whether the call that holds it is all known or split, so two executions of the specialized
        // program return distinct buffers. A call without such a copy still folds.
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = vector_type().with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap();
        let r#type = Type::Array(array_type.clone());
        let array = |values: &[f64]| {
            let bytes = values_to_bytes::<f64>(values);
            Value::Array(Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, bytes).unwrap())
        };
        let buffer = |value: &Value| {
            let Value::Array(array) = value else {
                panic!("expected an array value");
            };
            Arc::clone(array.addressable_shards().next().unwrap().buffer().unwrap())
        };
        let read = |value: &Value| {
            values_from_bytes::<f64>(buffer(value).copy_to_host(None).unwrap().r#await().unwrap().as_slice())
        };

        // Program `(a, x) -> call(a, x)` whose callee returns `(copy(a), -x)` when `mixed` is set and `copy(a)` or
        // `-a` otherwise.
        let program = |copy: bool, mixed: bool| {
            let callee = {
                let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
                let a = builder.add_input(r#type.clone());
                let x = builder.add_input(r#type.clone());
                let first = if copy {
                    let place = PlaceOperation::new(None, None).with_may_alias(false);
                    builder.add_instruction(place, Vec::new(), vec![a], None).unwrap()[0]
                } else {
                    builder.add_instruction(NegOperation::new(), Vec::new(), vec![a], None).unwrap()[0]
                };
                let mut outputs = vec![first];
                if mixed {
                    outputs.push(builder.add_instruction(NegOperation::new(), Vec::new(), vec![x], None).unwrap()[0]);
                }
                let count = outputs.len();
                builder.build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder; count])
            }
            .unwrap();
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let a = builder.add_input(r#type.clone());
            let x = builder.add_input(r#type.clone());
            let callee_region = builder.intern_callee(&Arc::new(callee), None).unwrap();
            let call = Operation::JitCall(JitCallOperation::new(0));
            let outputs = builder.add_instruction(call, vec![callee_region], vec![a, x], None).unwrap().to_vec();
            let count = outputs.len();
            builder
                .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder; count])
                .unwrap()
        };
        let known = array(&[1.0, 2.0, 3.0, 4.0]);

        // An all-known call whose callee forces a copy stays residual as a whole.
        let copying = program(true, false);
        let evaluation = copying
            .partially_evaluate_in_context(
                &session,
                &[PartialValue::Known(known.clone()), PartialValue::Known(known.clone())],
            )
            .unwrap();
        assert!(evaluation.outputs().iter().all(PartialEvaluationOutput::is_unknown));
        let first = evaluation.interpret(&session, &[]).unwrap();
        let second = evaluation.interpret(&session, &[]).unwrap();
        assert_eq!(read(&first[0]), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(read(&second[0]), vec![1.0, 2.0, 3.0, 4.0]);
        assert!(!Arc::ptr_eq(&buffer(&first[0]), &buffer(&second[0])));
        assert!(!Arc::ptr_eq(&buffer(&first[0]), &buffer(&known)));

        // An all-known call without a forced copy folds.
        let negating = program(false, false);
        let evaluation = negating
            .partially_evaluate_in_context(&session, &[PartialValue::Known(known.clone()), PartialValue::Known(known)])
            .unwrap();
        let [PartialEvaluationOutput::Known(negated)] = evaluation.outputs() else {
            panic!("expected one known output");
        };
        assert_eq!(read(negated), vec![-1.0, -2.0, -3.0, -4.0]);
    }

    #[test]
    fn test_jit_call_partial_evaluation_does_not_reuse_splits_that_fold_forced_copies() {
        // The split of a mixed call is cached per callee and partial evaluation configuration, so a split that a
        // partition which folds copies computed is not reused by a specialization, which must not fold them.
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array_type = vector_type().with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap();
        let r#type = Type::Array(array_type.clone());
        let array = |values: &[f64]| {
            let bytes = values_to_bytes::<f64>(values);
            Value::Array(Array::from_host_buffer(&session, array_type.clone(), mesh.clone(), None, bytes).unwrap())
        };
        let buffer = |value: &Value| {
            let Value::Array(array) = value else {
                panic!("expected an array value");
            };
            Arc::clone(array.addressable_shards().next().unwrap().buffer().unwrap())
        };
        let read = |value: &Value| {
            values_from_bytes::<f64>(buffer(value).copy_to_host(None).unwrap().r#await().unwrap().as_slice())
        };

        // Program `(a, x) -> call(a, x)` whose callee returns `(copy(a), -x)` when `mixed` is set and `copy(a)` or
        // `-a` otherwise.
        let program = |copy: bool, mixed: bool| {
            let callee = {
                let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
                let a = builder.add_input(r#type.clone());
                let x = builder.add_input(r#type.clone());
                let first = if copy {
                    let place = PlaceOperation::new(None, None).with_may_alias(false);
                    builder.add_instruction(place, Vec::new(), vec![a], None).unwrap()[0]
                } else {
                    builder.add_instruction(NegOperation::new(), Vec::new(), vec![a], None).unwrap()[0]
                };
                let mut outputs = vec![first];
                if mixed {
                    outputs.push(builder.add_instruction(NegOperation::new(), Vec::new(), vec![x], None).unwrap()[0]);
                }
                let count = outputs.len();
                builder.build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder; count])
            }
            .unwrap();
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let a = builder.add_input(r#type.clone());
            let x = builder.add_input(r#type.clone());
            let callee_region = builder.intern_callee(&Arc::new(callee), None).unwrap();
            let call = Operation::JitCall(JitCallOperation::new(0));
            let outputs = builder.add_instruction(call, vec![callee_region], vec![a, x], None).unwrap().to_vec();
            let count = outputs.len();
            builder
                .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder; count])
                .unwrap()
        };
        let known = array(&[1.0, 2.0, 3.0, 4.0]);
        let unknown = array(&[5.0, 6.0, 7.0, 8.0]);

        // A mixed call whose split was cached by a partition that folds copies does not reuse that split.
        let mixed = program(true, true);
        mixed.partition(&[true, false]).unwrap();
        let evaluation = mixed
            .partially_evaluate_in_context(
                &session,
                &[PartialValue::Known(known.clone()), PartialValue::Unknown(r#type.clone())],
            )
            .unwrap();
        assert!(evaluation.outputs().iter().all(PartialEvaluationOutput::is_unknown));
        let first = evaluation.interpret(&session, std::slice::from_ref(&unknown)).unwrap();
        let second = evaluation.interpret(&session, std::slice::from_ref(&unknown)).unwrap();
        assert_eq!(read(&first[0]), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(read(&first[1]), vec![-5.0, -6.0, -7.0, -8.0]);
        assert!(!Arc::ptr_eq(&buffer(&first[0]), &buffer(&second[0])));
        assert!(!Arc::ptr_eq(&buffer(&first[0]), &buffer(&known)));
    }

    #[test]
    fn test_jit_call_partial_evaluation_forwards_known_output_residuals() {
        // When the callee `f(a, x) = (sin(a), sin(a) * x)` is partitioned with `a` known and `x` unknown, the residual
        // call needs `sin(a)`, which is also a known output, so the known call returns it once and the residual call is
        // fed from that output (JAX's `out_fwd`). The outer partition repeats the known call's output as its own
        // residual edge, because `Program::partition` keeps one edge per residual feeder.
        let r#type = Type::Array(vector_type());
        let callee = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let a = builder.add_input(r#type.clone());
            let x = builder.add_input(r#type.clone());
            let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![a], None).unwrap()[0];
            let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![sine, x], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![sine, product], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let a = builder.add_input(r#type.clone());
        let x = builder.add_input(r#type);
        let callee_region = builder.intern_callee(&Arc::new(callee), None).unwrap();
        let outputs = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee_region], vec![a, x], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.partition(&[true, false]).unwrap().to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0)],
                    outputs=[Known(0), Unknown(0)],
                ]
                known={
                    lambda %0:f64[4] .
                    let %1:f64[4] = jit_call %0 [
                        callee={
                            lambda %0:f64[4] .
                            let %1:f64[4] = sin %0
                            in (%1)
                        },
                    ]
                    in (%1, %1)
                }
                residual={
                    lambda %0:f64[4], %1:f64[4] .
                    let %2:f64[4] = jit_call %0 %1 [
                        callee={
                            lambda %0:f64[4], %1:f64[4] .
                            let %2:f64[4] = mul %1 %0
                            in (%2)
                        },
                    ]
                    in (%2)
                }"},
        );
    }

    #[test]
    fn test_jit_call_partial_evaluation_preserves_boundary_for_capture_bearing_callees() {
        // Partitioning does not remap the absolute indices of retained capture constants, so a mixed call whose
        // callee still holds one must keep its original boundary instead of splitting into derived callees whose
        // compacted or absent capture prefixes those indices would misname.
        let r#type = Type::Array(vector_type());
        let callee = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let known_input = builder.add_input(r#type.clone());
            let runtime_input = builder.add_input(r#type.clone());
            let captured = builder.add_constant(Constant::Captured(CaptureReference::new(0, r#type.clone())));
            let shifted =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![known_input, captured], None).unwrap()[0];
            let product = builder
                .add_instruction(MulOperation::new(), Vec::new(), vec![shifted, runtime_input], None)
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![shifted, product],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let known_input = builder.add_input(r#type.clone());
        let runtime_input = builder.add_input(r#type.clone());
        let callee_region = builder.intern_callee(&Arc::new(callee), None).unwrap();
        let outputs = builder
            .add_instruction(
                Operation::JitCall(JitCallOperation::new(0)),
                vec![callee_region],
                vec![known_input, runtime_input],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let outer = TracingContext::<Constant, Operation>::new();
        let known = outer.input(r#type.clone());
        let evaluation = program
            .partially_evaluate_in_context(&outer, &[PartialValue::Known(known), PartialValue::Unknown(r#type)])
            .unwrap();

        // No known-side call is hoisted into the outer program, and the residual program retains the original call
        // with its callee intact, including its capture constant.
        assert!(outer.builder().borrow().instructions().is_empty());
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4] .
                let %2:f64[4], %3:f64[4] = jit_call %1 %0 [
                    callee={
                        lambda %0:f64[4], %1:f64[4] .
                        let %2:f64[4] = const capture#0:f64[4]
                            %3:f64[4] = add %0 %2
                            %4:f64[4] = mul %3 %1
                        in (%3, %4)
                    },
                ]
                in (%2, %3)"},
        );
    }

    #[test]
    fn test_jit_call_partial_evaluation_partitions_a_callee_with_a_dead_allocation() {
        let r#type = Type::Array(vector_type());
        let callee = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let known_input = builder.add_input(r#type.clone());
            let runtime_input = builder.add_input(r#type.clone());
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![known_input], None).unwrap();
            let doubled = builder
                .add_instruction(AddOperation::new(), Vec::new(), vec![known_input, known_input], None)
                .unwrap()[0];
            let product = builder
                .add_instruction(MulOperation::new(), Vec::new(), vec![known_input, runtime_input], None)
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![doubled, product],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };

        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let known_input = builder.add_input(r#type.clone());
        let runtime_input = builder.add_input(r#type.clone());
        let callee_region = builder.intern_callee(&Arc::new(callee), None).unwrap();
        let outputs = builder
            .add_instruction(
                Operation::JitCall(JitCallOperation::new(0)),
                vec![callee_region],
                vec![known_input, runtime_input],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        // The callee partitions like any other. Its pure prefix over the known input folds into the known side, which
        // is staged into the outer program, while the runtime-dependent product stays in the residual callee, and the
        // allocation, which nothing observes, is a dead intrinsic state operation that neither side retains.
        let outer = TracingContext::<Constant, Operation>::new();
        let known = outer.input(r#type.clone());
        let evaluation = program
            .partially_evaluate_in_context(&outer, &[PartialValue::Known(known), PartialValue::Unknown(r#type)])
            .unwrap();
        let known_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Constant>, Vec<Constant>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        assert_eq!(
            known_program.to_string(),
            indoc! {"
                lambda %0:f64[4] .
                let %1:f64[4] = jit_call %0 [
                    callee={
                        lambda %0:f64[4] .
                        let %1:f64[4] = add %0 %0
                        in (%1)
                    },
                ]
                in ()"},
        );
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4] .
                let %2:f64[4] = jit_call %0 %1 [
                    callee={
                        lambda %0:f64[4], %1:f64[4] .
                        let %2:f64[4] = mul %1 %0
                        in (%2)
                    },
                ]
                in (%2)"},
        );
    }

    #[test]
    fn test_jit_call_partial_evaluation_places_callee_residuals_with_the_context_policy() {
        struct SaveDots;

        impl ResidualPolicy<Type> for SaveDots {
            type Storage = NoStorage;

            fn name(&self) -> &str {
                "save_dots"
            }

            fn classify(
                &self,
                candidate: &ResidualCandidate<'_, Type>,
            ) -> Result<ResidualDecision<NoStorage>, ResidualRejection> {
                Ok(match candidate.producers().iter().any(|producer| producer.payload::<DotOperation>().is_some()) {
                    true => ResidualDecision::Save,
                    false => ResidualDecision::Recompute,
                })
            }
        }

        // The callee computes `cos(dot(x, x)) * t`. Partitioning the caller with `x` known splits the call, and the
        // policy then plans the callee partition, in which the known call returns the dot product instead of its
        // cosine and the residual call recomputes the cosine.
        let callee = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let x = builder.add_input(Type::Array(vector_type()));
            let t = builder.add_input(Type::Array(ArrayType::scalar(DataType::F64)));
            let dot = ArrayOperation::<ArrayConstant>::from(DotOperation::new(DotDimensionNumbers::new(
                vec![0],
                vec![0],
                vec![],
                vec![],
            )));
            let product = builder.add_instruction(dot, Vec::new(), vec![x, x], None).unwrap()[0];
            let cosine = ArrayOperation::<ArrayConstant>::from(CosOperation::<ArrayType>::new());
            let cosine = builder.add_instruction(cosine, Vec::new(), vec![product], None).unwrap()[0];
            let tangent = ArrayOperation::<ArrayConstant>::from(MulOperation::<ArrayType>::new());
            let tangent = builder.add_instruction(tangent, Vec::new(), vec![cosine, t], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![tangent], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let x = builder.add_input(Type::Array(vector_type()));
        let t = builder.add_input(Type::Array(ArrayType::scalar(DataType::F64)));
        let callee_region = builder.intern_callee(&Arc::new(callee), None).unwrap();
        let output = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee_region], vec![x, t], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let partition = program
            .partition_with_residual_policy(&[true, false], &ResidualPolicyReference::new(SaveDots))
            .unwrap();
        assert_eq!(
            partition.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:f64[4] .
                    let %1:f64[] = jit_call %0 [
                        callee={
                            lambda %0:f64[4] .
                            let %1:f64[] = dot [
                                dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            ] %0 %0
                            in (%1)
                        },
                    ]
                    in (%1)
                }
                residual={
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = jit_call %0 %1 [
                        callee={
                            lambda %0:f64[], %1:f64[] .
                            let %2:f64[] = cos %1
                                %3:f64[] = mul %2 %0
                            in (%3)
                        },
                    ]
                    in (%2)
                }"},
        );
    }

    #[test]
    fn test_jit_call_partial_evaluation_places_all_known_callee_intermediates() {
        // Only the expanded activation is needed by the tangent. Its all-known call must expose the compact tagged
        // scale to the policy, saving 4 KiB instead of 256 KiB, so the call is split on residual demand into a known
        // call that also returns the scale and a replay call that recomputes only the broadcast.
        let compact_type = ArrayType::new_static(DataType::F32, [1024]);
        let expanded_type = ArrayType::new_static(DataType::F32, [64, 1024]);
        let mut builder = ArrayIrProgramBuilder::new();
        let weights = builder.add_input(compact_type.clone().into());
        let scale = builder.add_instruction(SinOperation::new(), Vec::new(), vec![weights], None).unwrap()[0];
        let scale = builder
            .add_instruction(TagOperation::<ArrayType>::new("scale"), Vec::new(), vec![scale], None)
            .unwrap()[0];
        let expanded = builder
            .add_instruction(BroadcastOperation::new(expanded_type.clone(), vec![1]), Vec::new(), vec![scale], None)
            .unwrap()[0];
        let callee = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![expanded], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(expanded_type.clone().into());
        let weights = builder.add_input(compact_type.clone().into());
        let callee = builder.import_program(callee);
        let scale = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee], vec![weights], None)
            .unwrap()[0];
        let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, scale], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // Ordinary partial evaluation preserves the original compiled-call boundary and expanded residual.
        let ordinary = program.partition(&[false, true]).unwrap();
        assert_eq!(ordinary.known_program().output_types(), vec![Type::from(expanded_type.clone())]);
        assert_eq!(
            ordinary.to_string(),
            indoc! {"
                partition [
                    known_inputs=[1],
                    residual_inputs=[UnknownInput(0), ResidualEdge(0)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:f32[1024] .
                    let %1:f32[64, 1024] = jit_call %0 [
                        callee={
                            lambda %0:f32[1024] .
                            let %1:f32[1024] = sin %0
                                %2:f32[1024] = tag [key=scale] %1
                                %3:f32[64, 1024] = broadcast [output_type=f32[64, 1024], output_axes=[1]] %2
                            in (%3)
                        },
                    ]
                    in (%1)
                }
                residual={
                    lambda %0:f32[64, 1024], %1:f32[64, 1024] .
                    let %2:f32[64, 1024] = mul %0 %1
                    in (%2)
                }"},
        );
        assert_eq!(expanded_type.element_count().unwrap().unwrap() * size_of::<f32>(), 262_144);

        let policy = ResidualPolicyReference::new(SaveOnlyTheseNames::new(["scale"]));
        let placed = program.partition_with_residual_policy(&[false, true], &policy).unwrap();
        assert_eq!(placed.known_program().output_types(), vec![Type::from(compact_type.clone())]);
        assert_eq!(compact_type.element_count().unwrap().unwrap() * size_of::<f32>(), 4096);
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[1],
                    residual_inputs=[UnknownInput(0), ResidualEdge(0)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:f32[1024] .
                    let %1:f32[1024] = jit_call %0 [
                        callee={
                            lambda %0:f32[1024] .
                            let %1:f32[1024] = sin %0
                                %2:f32[1024] = tag [key=scale] %1
                            in (%2)
                        },
                    ]
                    in (%1)
                }
                residual={
                    lambda %0:f32[64, 1024], %1:f32[1024] .
                    let %2:f32[64, 1024] = jit_call %1 [
                        callee={
                            lambda %0:f32[1024] .
                            let %1:f32[64, 1024] = broadcast [output_type=f32[64, 1024], output_axes=[1]] %0
                            in (%1)
                        },
                    ]
                        %3:f32[64, 1024] = mul %0 %2
                    in (%3)
                }"},
        );

        // The same cut survives the rematerialization wrapper's linearization, where the original output remains a
        // known primal result and the expanded scale is needed only by the derivative with respect to the activation.
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(expanded_type.clone().into());
        let weights = builder.add_input(compact_type.clone().into());
        let plain_linearization = program.linearize_with_respect_to(&[0]).unwrap();
        let body = builder.import_program(program);
        let output = builder
            .add_instruction(RematerializeOperation::new(policy), vec![body], vec![input, weights], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let linearization = program.linearize_with_respect_to(&[0]).unwrap();
        assert_eq!(linearization.residual_count(), 1);
        assert_eq!(
            linearization.primal().output_types(),
            vec![Type::from(expanded_type.clone()), Type::from(compact_type.clone())],
        );
        assert_eq!(
            format!("{}\n{}", linearization.primal(), linearization.tangent()),
            indoc! {"
                lambda %0:f32[64, 1024], %1:f32[1024] .
                let %2:f32[64, 1024], %3:f32[1024] = jit_call %1 [
                    callee={
                        lambda %0:f32[1024] .
                        let %1:f32[1024] = sin %0
                            %2:f32[1024] = tag [key=scale] %1
                            %3:f32[64, 1024] = broadcast [output_type=f32[64, 1024], output_axes=[1]] %2
                        in (%3, %2)
                    },
                ]
                    %4:f32[64, 1024] = mul %0 %2
                in (%4, %3)
                lambda %0:f32[64, 1024], %1:f32[1024] .
                let %2:f32[64, 1024] = rematerialize [policy=\"save_only_these_names\", differentiated=true] %0 %1 [
                    body={
                        lambda %0:f32[64, 1024], %1:f32[1024] .
                        let %2:f32[64, 1024] = jit_call %1 [
                            callee={
                                lambda %0:f32[1024] .
                                let %1:f32[64, 1024] = broadcast [output_type=f32[64, 1024], output_axes=[1]] %0
                                in (%1)
                            },
                        ]
                            %3:f32[64, 1024] = mul %2 %0
                        in (%3)
                    },
                ]
                in (%2)"},
        );

        let client = execution_client();
        let inputs = vec![vec![2f32; 64 * 1024], (0..1024).map(|index| 0.1f32 + index as f32 * 1e-4).collect()];
        let direction = vec![0.75f32; 64 * 1024];
        let cotangent = vec![1.5f32; 64 * 1024];
        let evaluate = |linearization: &Linearization<Constant, Operation>| {
            let mut primal = execute_f32_program(client, linearization.primal(), inputs.clone());
            let value = primal.remove(0);
            let tangent_inputs = std::iter::once(direction.clone()).chain(primal.clone()).collect();
            let tangent = execute_f32_program(client, linearization.tangent(), tangent_inputs);
            let pullback_inputs = std::iter::once(cotangent.clone()).chain(primal).collect();
            let pullback =
                execute_f32_program(client, &linearization.pullback().unwrap().into_program(), pullback_inputs);
            vec![value, tangent[0].clone(), pullback[0].clone()]
        };
        let actual = evaluate(&linearization);
        assert_eq!(actual, evaluate(&plain_linearization));
        assert!(actual.iter().flatten().all(|value| value.is_finite() && *value != 0.0));
    }

    #[test]
    fn test_jit_call_partial_evaluation_ignores_known_outputs_without_residual_demand() {
        let scalar_type = Type::from(ArrayType::scalar(DataType::F32));
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(scalar_type.clone());
        let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let needed = builder
            .add_instruction(TagOperation::<ArrayType>::new("needed"), Vec::new(), vec![sine], None)
            .unwrap()[0];
        let cosine = builder.add_instruction(CosOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let unused = builder
            .add_instruction(TagOperation::<ArrayType>::new("unused"), Vec::new(), vec![cosine], None)
            .unwrap()[0];
        let callee = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![needed, unused], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        let mut builder = ArrayIrProgramBuilder::new();
        let known = builder.add_input(scalar_type.clone());
        let unknown = builder.add_input(scalar_type.clone());
        let callee = builder.import_program(callee);
        let called = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee], vec![known], None)
            .unwrap()
            .to_vec();
        let first =
            builder.add_instruction(MulOperation::new(), Vec::new(), vec![called[0], unknown], None).unwrap()[0];
        let second =
            builder.add_instruction(MulOperation::new(), Vec::new(), vec![called[1], unknown], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![first, second, called[1]],
                vec![Placeholder; 2],
                vec![Placeholder; 3],
            )
            .unwrap();
        let policy =
            ResidualPolicyReference::new(RematerializationPolicyFn::new::<Type>(|candidate| {
                if candidate.producers().iter().any(|producer| {
                    producer.payload::<TagOperation<ArrayType>>().is_some_and(|tag| tag.key() == "unused")
                }) {
                    Err(ResidualRejection::new("the unused result must not cross the residual boundary"))
                } else {
                    Ok(ResidualDecision::<NoStorage>::Save)
                }
            }));

        // The second call output remains a primal result. Its presence must not classify it as a residual. The call
        // keeps its boundary, and the policy saves its first output, which it classifies through the callee.
        let primal_only = program.clone().with_outputs(&[0, 2]).unwrap();
        let placed = primal_only.partition_with_residual_policy(&[true, false], &policy).unwrap();
        assert_eq!(placed.known_program().output_types(), vec![scalar_type.clone(), scalar_type]);
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0)],
                    outputs=[Unknown(0), Known(0)],
                ]
                known={
                    lambda %0:f32[] .
                    let %1:f32[], %2:f32[] = jit_call %0 [
                        callee={
                            lambda %0:f32[] .
                            let %1:f32[] = sin %0
                                %2:f32[] = tag [key=needed] %1
                                %3:f32[] = cos %0
                                %4:f32[] = tag [key=unused] %3
                            in (%2, %4)
                        },
                    ]
                    in (%2, %1)
                }
                residual={
                    lambda %0:f32[], %1:f32[] .
                    let %2:f32[] = mul %1 %0
                    in (%2)
                }"},
        );

        // Requiring the same result in residual work must propagate the original rejection instead.
        assert_eq!(
            program
                .with_outputs(&[0, 1])
                .unwrap()
                .partition_with_residual_policy(&[true, false], &policy)
                .map(|_| ())
                .map_err(ResidualPolicyError::from),
            Err(ResidualPolicyError::Rejected {
                policy: policy.name().to_owned(),
                rejection: ResidualRejection::new("the unused result must not cross the residual boundary"),
            }),
        );
    }

    #[test]
    fn test_jit_call_partial_evaluation_preserves_guarded_known_callees() {
        let policy = ResidualPolicyReference::new(NothingSavable);
        let check_boundary = |callee: Program<Vec<Constant>, Vec<Constant>>, capture_count: usize| {
            let mut builder = ArrayIrProgramBuilder::new();
            let inputs = callee
                .input_types()
                .into_owned()
                .into_iter()
                .map(|r#type| builder.add_input(r#type))
                .collect::<Vec<_>>();
            let input_count = inputs.len();
            let callee = builder.import_program(callee);
            let outputs = builder
                .add_instruction(Operation::JitCall(JitCallOperation::new(capture_count)), vec![callee], inputs, None)
                .unwrap()
                .to_vec();
            let output_count = outputs.len();
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    outputs,
                    vec![Placeholder; input_count],
                    vec![Placeholder; output_count],
                )
                .unwrap();
            let partition = program.partition_with_residual_policy(&vec![true; input_count], &policy).unwrap();
            assert_eq!(partition.known_program().to_string(), program.to_string());
            assert_eq!(partition.residual_program().instructions().len(), 0);
        };

        // Nominal dimensions remain scoped to the call, even when the callee is otherwise pure.
        let dimension = DimensionVariable::new("length", DimensionBounds::new(1, Some(8)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(dimension)]));
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(dynamic_type.into());
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        check_boundary(
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap(),
            0,
        );

        // An ordered assertion must keep the same owning call and execution order.
        let predicate_type = Type::from(ArrayType::scalar(DataType::Boolean));
        let mut builder = ArrayIrProgramBuilder::new();
        let predicate = builder.add_input(predicate_type.clone());
        builder
            .add_instruction(AssertOperation::<ArrayType>::new("required predicate"), Vec::new(), vec![predicate], None)
            .unwrap();
        check_boundary(
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![predicate], vec![Placeholder], vec![Placeholder])
                .unwrap(),
            0,
        );

        // Reference handles and retained capture constants keep their existing boundary semantics too.
        let mut builder = ArrayIrProgramBuilder::new();
        let reference = builder.add_input(Type::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        check_boundary(
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap(),
            0,
        );
        let mut builder = ArrayIrProgramBuilder::new();
        builder.add_input(predicate_type.clone());
        let captured = builder.add_constant(Constant::Captured(CaptureReference::new(0, predicate_type.clone())));
        check_boundary(
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![captured], vec![Placeholder], vec![Placeholder])
                .unwrap(),
            1,
        );

        // A known value without a staged parent atom must not trigger per-operation replay. The same guard keeps
        // concrete eager calls whole. Using a literal in a live trace makes the retained call directly observable.
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(predicate_type);
        let callee = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let outer = TracingContext::<Constant, Operation>::new();
        let context = PartialEvaluationContext::new(outer.clone()).with_residual_policy(&policy);
        let input = context.lift(Constant::Boolean(true)).unwrap();
        context.bind(Operation::JitCall(JitCallOperation::new(0)), vec![callee], &[input]).unwrap();
        let known_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Constant>, Vec<Constant>>(Vec::new(), Vec::new(), Vec::new())
            .unwrap();
        assert_eq!(
            known_program.to_string(),
            indoc! {"
                lambda  .
                let %0:bool[] = const true
                    %1:bool[] = jit_call %0 [
                        callee={
                            lambda %0:bool[] .
                            in (%0)
                        },
                    ]
                in ()"},
        );
    }

    #[test]
    fn test_jit_call_partial_evaluation_does_not_forward_inputs_across_call_boundaries() {
        // The callee `f(a, x) = (a, sin(a) * x)` returns its first input, which the boundary of the compiled function
        // that it comes from replicates. Logical types do not record that placement, so neither an all-known call of
        // it nor the known call of its split may be replaced by its input under a residual policy.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let device_mesh = DeviceMesh::new(mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)]).unwrap();
        let replicated = Sharding::replicated(mesh, 1);
        let r#type: Type = ArrayType::new_static(DataType::F32, [4]).into();
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let a = builder.add_input(r#type.clone());
            let x = builder.add_input(r#type.clone());
            let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![a], None).unwrap()[0];
            let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![sine, x], None).unwrap()[0];
            Arc::new(
                builder
                    .build::<Vec<Constant>, Vec<Constant>>(vec![a, product], vec![Placeholder; 2], vec![Placeholder; 2])
                    .unwrap(),
            )
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let a = builder.add_input(r#type.clone());
        let x = builder.add_input(r#type.clone());
        let callee_region = builder.intern_callee(&callee, None).unwrap();
        let call = JitCallOperation::new(0).with_boundary(Some(&device_mesh), Vec::new(), vec![Some(replicated), None]);
        let outputs = builder
            .add_instruction(Operation::JitCall(call), vec![callee_region], vec![a, x], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let policy = ResidualPolicyReference::new(NothingSavable);
        assert_eq!(
            program.partition_with_residual_policy(&[true, true], &policy).unwrap().to_string(),
            indoc! {"
                partition [known_inputs=[0, 1], residual_inputs=[], outputs=[Known(0), Known(1)]]
                known={
                    lambda %0:f32[4], %1:f32[4] .
                    let %2:f32[4], %3:f32[4] = jit_call [output_shardings=[{mesh<['x'=2:auto]>, [{}]}, _], devices=[0, 1]] %0 %1 [
                        callee={
                            lambda %0:f32[4], %1:f32[4] .
                            let %2:f32[4] = sin %0
                                %3:f32[4] = mul %2 %1
                            in (%0, %3)
                        },
                    ]
                    in (%2, %3)
                }
                residual={
                    lambda  .
                    in ()
                }"},
        );
        assert_eq!(
            program.partition_with_residual_policy(&[true, false], &policy).unwrap().to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0)],
                    outputs=[Unknown(0), Unknown(1)],
                ]
                known={
                    lambda %0:f32[4] .
                    in (%0)
                }
                residual={
                    lambda %0:f32[4], %1:f32[4] .
                    let %2:f32[4], %3:f32[4] = jit_call [output_shardings=[{mesh<['x'=2:auto]>, [{}]}, _], devices=[0, 1]] %1 %0 [
                        callee={
                            lambda %0:f32[4], %1:f32[4] .
                            let %2:f32[4] = sin %0
                                %3:f32[4] = mul %2 %1
                            in (%0, %3)
                        },
                    ]
                    in (%2, %3)
                }"},
        );
    }

    #[test]
    fn test_jit_call_partial_evaluation_keeps_effectful_all_known_calls_whole_on_residual_demand() {
        let scalar_type: Type = ArrayType::scalar(DataType::F32).into();
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(scalar_type.clone());
        builder
            .add_instruction(PrintOperation::<ArrayType>::new("known"), Vec::new(), vec![input], None)
            .unwrap();
        let callee = Arc::new(
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
                .unwrap(),
        );
        let mut builder = ArrayIrProgramBuilder::new();
        let known = builder.add_input(scalar_type.clone());
        let unknown = builder.add_input(scalar_type);
        let forwarded = add_jit_call(&mut builder, &callee, vec![known]);
        let output =
            builder.add_instruction(MulOperation::new(), Vec::new(), vec![forwarded, unknown], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let partition = program
            .partition_with_residual_policy(&[true, false], &ResidualPolicyReference::new(NothingSavable))
            .unwrap();
        assert_eq!(
            partition.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:f32[] .
                    let %1:f32[] = jit_call %0 [
                        callee={
                            lambda %0:f32[] .
                            let %1:f32[] = print [label=known] %0
                            in (%0)
                        },
                    ]
                    in (%1)
                }
                residual={
                    lambda %0:f32[], %1:f32[] .
                    let %2:f32[] = mul %1 %0
                    in (%2)
                }"},
        );
    }

    #[test]
    fn test_jit_call_partial_evaluation_splits_repeated_all_known_calls_on_residual_demand() {
        let callee = residual_demand_leaf_callee();
        let repeated = |count: usize, policy: Option<ResidualPolicyReference<Type>>| {
            residual_demand_program(policy, |builder, weights, bias| {
                (0..count).fold(weights, |hidden, _| add_jit_call(builder, &callee, vec![hidden, bias]))
            })
        };

        // Every call is split into a known call that also returns its matrix product and a replay call that recomputes
        // its output from that product. All calls share one known callee and one replay callee, so the residual count
        // and the programs grow by one per call (i.e., the saved product and the call itself) rather than by the size
        // of the inlined callee, which made them grow by five and four instructions per call.
        let policy = ResidualPolicyReference::new(DotsSavable);
        assert_eq!(
            [1, 2, 4, 8].map(|count| residual_demand_statistics(&repeated(count, Some(policy.clone())), &[0])),
            [[2, 7, 7], [3, 8, 8], [5, 10, 10], [9, 14, 14]],
        );
        let linearization = repeated(2, Some(policy.clone())).linearize_with_respect_to(&[0]).unwrap();
        assert_eq!(
            format!("{}\n{}", linearization.primal(), linearization.tangent()),
            indoc! {"
                lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2] .
                let %3:f32[2, 2], %4:f32[2, 2] = jit_call %1 %2 [
                    callee=^0={
                        lambda %0:f32[2, 2], %1:f32[2, 2] .
                        let %2:f32[2, 2] = dot [
                            dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        ] %0 %1
                            %3:f32[2, 2] = sin %2
                            %4:f32[2, 2] = cos %2
                            %5:f32[2, 2] = mul %3 %4
                            %6:f32[2, 2] = add %5 %0
                        in (%6, %2)
                    },
                ]
                    %5:f32[2, 2], %6:f32[2, 2] = jit_call %3 %2 [
                        callee=^0,
                    ]
                    %7:f32[2, 2] = mul %0 %5
                in (%7, %1, %4, %6)
                lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2], %3:f32[2, 2] .
                let %4:f32[2, 2] = rematerialize [policy=\"dots_savable\", differentiated=true] %0 %1 %2 %3 [
                    body={
                        lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2], %3:f32[2, 2] .
                        let %4:f32[2, 2] = jit_call %1 %2 [
                            callee=^0={
                                lambda %0:f32[2, 2], %1:f32[2, 2] .
                                let %2:f32[2, 2] = sin %1
                                    %3:f32[2, 2] = cos %1
                                    %4:f32[2, 2] = mul %2 %3
                                    %5:f32[2, 2] = add %4 %0
                                in (%5)
                            },
                        ]
                            %5:f32[2, 2] = jit_call %4 %3 [
                                callee=^0,
                            ]
                            %6:f32[2, 2] = mul %5 %0
                        in (%6)
                    },
                ]
                in (%4)"},
        );

        // Rematerialization changes neither the primal value nor the derivatives.
        let client = execution_client();
        assert_residual_demand_linearizations_match(client, &repeated(4, Some(policy)), &repeated(4, None), &[0]);
    }

    #[test]
    fn test_jit_call_partial_evaluation_splits_nested_all_known_calls_on_residual_demand() {
        let nested = |depth: usize, policy: Option<ResidualPolicyReference<Type>>| {
            let callee = residual_demand_diamond_callee(depth);
            residual_demand_program(policy, |builder, weights, bias| {
                add_jit_call(builder, &callee, vec![weights, bias])
            })
        };

        // The split of each level contains the shared splits of the level below it, so the programs grow by four
        // instructions per level (i.e., one known or replay call, the second call that shares its callee, and the
        // shifted input and sum of the level) while the saved matrix products double, as in JAX. Inlining the calls
        // instead doubled the programs with every level.
        let policy = ResidualPolicyReference::new(DotsSavable);
        assert_eq!(
            [1, 2, 3, 4].map(|depth| residual_demand_statistics(&nested(depth, Some(policy.clone())), &[0])),
            [[3, 11, 11], [5, 15, 15], [9, 19, 19], [17, 23, 23]],
        );
        let linearization = nested(1, Some(policy.clone())).linearize_with_respect_to(&[0]).unwrap();
        assert_eq!(
            format!("{}\n{}", linearization.primal(), linearization.tangent()),
            indoc! {"
                lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2] .
                let %3:f32[2, 2], %4:f32[2, 2], %5:f32[2, 2] = jit_call %1 %2 [
                    callee={
                        lambda %0:f32[2, 2], %1:f32[2, 2] .
                        let %2:f32[2, 2], %3:f32[2, 2] = jit_call %0 %1 [
                            callee=^0={
                                lambda %0:f32[2, 2], %1:f32[2, 2] .
                                let %2:f32[2, 2] = dot [
                                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                                ] %0 %1
                                    %3:f32[2, 2] = sin %2
                                    %4:f32[2, 2] = cos %2
                                    %5:f32[2, 2] = mul %3 %4
                                    %6:f32[2, 2] = add %5 %0
                                in (%6, %2)
                            },
                        ]
                            %4:f32[2, 2] = sin %0
                            %5:f32[2, 2], %6:f32[2, 2] = jit_call %4 %1 [
                                callee=^0,
                            ]
                            %7:f32[2, 2] = add %2 %5
                        in (%7, %3, %6)
                    },
                ]
                    %6:f32[2, 2] = mul %0 %3
                in (%6, %1, %4, %5)
                lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2], %3:f32[2, 2] .
                let %4:f32[2, 2] = rematerialize [policy=\"dots_savable\", differentiated=true] %0 %1 %2 %3 [
                    body={
                        lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2], %3:f32[2, 2] .
                        let %4:f32[2, 2] = jit_call %1 %2 %3 [
                            callee={
                                lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2] .
                                let %3:f32[2, 2] = jit_call %0 %1 [
                                    callee=^0={
                                        lambda %0:f32[2, 2], %1:f32[2, 2] .
                                        let %2:f32[2, 2] = sin %1
                                            %3:f32[2, 2] = cos %1
                                            %4:f32[2, 2] = mul %2 %3
                                            %5:f32[2, 2] = add %4 %0
                                        in (%5)
                                    },
                                ]
                                    %4:f32[2, 2] = sin %0
                                    %5:f32[2, 2] = jit_call %4 %2 [
                                        callee=^0,
                                    ]
                                    %6:f32[2, 2] = add %3 %5
                                in (%6)
                            },
                        ]
                            %5:f32[2, 2] = mul %4 %0
                        in (%5)
                    },
                ]
                in (%4)"},
        );

        let client = execution_client();
        assert_residual_demand_linearizations_match(client, &nested(3, Some(policy)), &nested(3, None), &[0]);
    }

    #[test]
    fn test_jit_call_partial_evaluation_replays_all_known_calls_without_saved_interior_values() {
        let callee = residual_demand_leaf_callee();
        let repeated = |count: usize, policy: Option<ResidualPolicyReference<Type>>| {
            residual_demand_program(policy, |builder, weights, bias| {
                (0..count).fold(weights, |hidden, _| add_jit_call(builder, &callee, vec![hidden, bias]))
            })
        };

        // A policy that saves nothing splits no call. The tangent replays every call whole from the saved weights and
        // bias, which it saves no matter how many calls there are, and the calls keep sharing one callee.
        let policy = ResidualPolicyReference::new(NothingSavable);
        assert_eq!(
            [1, 2, 4, 8].map(|count| residual_demand_statistics(&repeated(count, Some(policy.clone())), &[0])),
            [[2, 7, 8], [2, 8, 9], [2, 10, 11], [2, 14, 15]],
        );
        let linearization = repeated(2, Some(policy.clone())).linearize_with_respect_to(&[0]).unwrap();
        assert_eq!(
            format!("{}\n{}", linearization.primal(), linearization.tangent()),
            indoc! {"
                lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2] .
                let %3:f32[2, 2] = jit_call %1 %2 [
                    callee=^0={
                        lambda %0:f32[2, 2], %1:f32[2, 2] .
                        let %2:f32[2, 2] = dot [
                            dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        ] %0 %1
                            %3:f32[2, 2] = sin %2
                            %4:f32[2, 2] = cos %2
                            %5:f32[2, 2] = mul %3 %4
                            %6:f32[2, 2] = add %5 %0
                        in (%6)
                    },
                ]
                    %4:f32[2, 2] = jit_call %3 %2 [
                        callee=^0,
                    ]
                    %5:f32[2, 2] = mul %0 %4
                in (%5, %1, %2)
                lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2] .
                let %3:f32[2, 2] = rematerialize [differentiated=true] %0 %1 %2 [
                    body={
                        lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2] .
                        let %3:f32[2, 2] = jit_call %1 %2 [
                            callee=^0={
                                lambda %0:f32[2, 2], %1:f32[2, 2] .
                                let %2:f32[2, 2] = dot [
                                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                                ] %0 %1
                                    %3:f32[2, 2] = sin %2
                                    %4:f32[2, 2] = cos %2
                                    %5:f32[2, 2] = mul %3 %4
                                    %6:f32[2, 2] = add %5 %0
                                in (%6)
                            },
                        ]
                            %4:f32[2, 2] = jit_call %3 %2 [
                                callee=^0,
                            ]
                            %5:f32[2, 2] = mul %4 %0
                        in (%5)
                    },
                ]
                in (%3)"},
        );

        let client = execution_client();
        assert_residual_demand_linearizations_match(client, &repeated(4, Some(policy)), &repeated(4, None), &[0]);
    }

    #[test]
    fn test_jit_call_partial_evaluation_saves_guarded_known_callee_outputs_on_residual_demand() {
        // Each callee computes a matrix product that `DotsSavable` saves, so replaying its call would recompute a saved
        // value. An ordered assertion and a capture constant keep the boundary of the call, so the residual planner
        // saves the demanded output of the call instead of splitting it.
        let matrix_type = residual_demand_matrix_type();
        let predicate_type = Type::from(ArrayType::scalar(DataType::Boolean));
        let dot = DotOperation::new(DotDimensionNumbers::new(vec![1], vec![0], vec![], vec![]));
        let check_saved_output = |callee: FlatProgram, capture_count: usize| {
            let mut builder = ArrayIrProgramBuilder::new();
            let inputs = callee
                .input_types()
                .into_owned()
                .into_iter()
                .map(|r#type| builder.add_input(r#type))
                .collect::<Vec<_>>();
            let tangent = builder.add_input(matrix_type.clone());
            let input_count = inputs.len() + 1;
            let callee = builder.import_program(callee);
            let called = builder
                .add_instruction(Operation::JitCall(JitCallOperation::new(capture_count)), vec![callee], inputs, None)
                .unwrap()[0];
            let output =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![called, tangent], None).unwrap()[0];
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; input_count], vec![Placeholder])
                .unwrap();
            let input_known = (0..input_count).map(|index| index + 1 < input_count).collect::<Vec<_>>();
            let partition = program
                .partition_with_residual_policy(&input_known, &ResidualPolicyReference::new(DotsSavable))
                .unwrap();
            assert_eq!(partition.known_program().instructions().len(), 1);
            assert!(matches!(partition.known_program().instructions()[0].operation(), Operation::JitCall(_)));
            assert_eq!(partition.known_program().output_ids(), partition.known_program().instructions()[0].outputs());
        };

        let mut builder = ArrayIrProgramBuilder::new();
        let predicate = builder.add_input(predicate_type.clone());
        let input = builder.add_input(matrix_type.clone());
        builder
            .add_instruction(AssertOperation::<ArrayType>::new("required predicate"), Vec::new(), vec![predicate], None)
            .unwrap();
        let product = builder.add_instruction(dot.clone(), Vec::new(), vec![input, input], None).unwrap()[0];
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![product], None).unwrap()[0];
        check_saved_output(
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap(),
            0,
        );

        let mut builder = ArrayIrProgramBuilder::new();
        builder.add_input(matrix_type.clone());
        let input = builder.add_input(matrix_type.clone());
        let captured = builder.add_constant(Constant::Captured(CaptureReference::new(0, matrix_type.clone())));
        let product = builder.add_instruction(dot, Vec::new(), vec![input, captured], None).unwrap()[0];
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![product], None).unwrap()[0];
        check_saved_output(
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap(),
            1,
        );
    }

    #[test]
    fn test_jit_call_partial_evaluation_rounds_values_saved_on_residual_demand() {
        // The function `(x, s) ↦ w · s` calls `w = f32(t)` for `t = tag(bf16(x) · bf16(x), "t")`. Rematerializing it
        // under a policy that saves `t` and differentiating it with respect to `s` splits the all-known call on
        // residual demand into a known call that saves `t` and computes `w` from it and a replay call that recomputes
        // `w` from the saved `t`. XLA computes the `bf16` product in `f32`, so the known callee rounds `t` right after
        // its producer, before computing `w` from it, and the output and its derivative with respect to `s` (both `w`
        // for `s = 1`) agree. Without that rounding, the output used `t` at `f32` precision (1.21343994 for `x = 1.1`)
        // while the derivative used the saved `bf16` value (1.2109375).
        let scalar_type = Type::from(ArrayType::scalar(DataType::F32));
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let x = builder.add_input(scalar_type.clone());
            let to_bf16 = ConvertElementTypeOperation::<ArrayType>::new(DataType::BF16, false);
            let x = builder.add_instruction(to_bf16, Vec::new(), vec![x], None).unwrap()[0];
            let t = builder.add_instruction(MulOperation::new(), Vec::new(), vec![x, x], None).unwrap()[0];
            let t = builder.add_instruction(TagOperation::<ArrayType>::new("t"), Vec::new(), vec![t], None).unwrap()[0];
            let to_f32 = ConvertElementTypeOperation::<ArrayType>::new(DataType::F32, false);
            let w = builder.add_instruction(to_f32, Vec::new(), vec![t], None).unwrap()[0];
            Arc::new(
                builder
                    .build::<Vec<Constant>, Vec<Constant>>(vec![w], vec![Placeholder], vec![Placeholder])
                    .unwrap(),
            )
        };
        let body = |returns_w: bool| {
            let mut builder = ArrayIrProgramBuilder::new();
            let x = builder.add_input(scalar_type.clone());
            let s = builder.add_input(scalar_type.clone());
            let w = add_jit_call(&mut builder, &callee, vec![x]);
            let y = builder.add_instruction(MulOperation::new(), Vec::new(), vec![w, s], None).unwrap()[0];
            let outputs = if returns_w { vec![w, y] } else { vec![y] };
            let output_count = outputs.len();
            builder
                .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder; output_count])
                .unwrap()
        };
        let policy = ResidualPolicyReference::new(SaveOnlyTheseNames::new(["t"]));

        // Placing the residuals of `(x, s) ↦ (w, w · s)` with `x` known, with the same policy, and without rounding,
        // splits the call in the same way, but its known callee consumes the saved `t` unrounded. The
        // rematerialization below must not reuse that split of the shared callee.
        let partition = body(true).partition_with_residual_policy(&[true, false], &policy).unwrap();
        assert_eq!(
            partition.known_program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[], %2:bf16[] = jit_call %0 [
                    callee={
                        lambda %0:f32[] .
                        let %1:bf16[] = convert_element_type [data_type=bf16] %0
                            %2:bf16[] = mul %1 %1
                            %3:bf16[] = tag [key=t] %2
                            %4:f32[] = convert_element_type [data_type=f32] %3
                        in (%4, %3)
                    },
                ]
                in (%1, %2)"},
        );

        let mut builder = ArrayIrProgramBuilder::new();
        let inputs = vec![builder.add_input(scalar_type.clone()), builder.add_input(scalar_type.clone())];
        let body = builder.import_program(body(false));
        let output = builder.add_instruction(RematerializeOperation::new(policy), vec![body], inputs, None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let linearization = program.linearize_with_respect_to(&[1]).unwrap();
        assert_eq!(
            format!("{}\n{}", linearization.primal(), linearization.tangent()),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[], %3:bf16[] = jit_call %0 [
                    callee={
                        lambda %0:f32[] .
                        let %1:bf16[] = convert_element_type [data_type=bf16] %0
                            %2:bf16[] = mul %1 %1
                            %3:bf16[] = tag [key=t] %2
                            %4:bf16[] = reduce_precision [exponent_bits=8, mantissa_bits=7] %3
                            %5:f32[] = convert_element_type [data_type=f32] %4
                        in (%5, %4)
                    },
                ]
                    %4:f32[] = mul %2 %1
                in (%4, %3)
                lambda %0:f32[], %1:bf16[] .
                let %2:f32[] = rematerialize [policy=\"save_only_these_names\", differentiated=true] %0 %1 [
                    body={
                        lambda %0:f32[], %1:bf16[] .
                        let %2:f32[] = jit_call %1 [
                            callee={
                                lambda %0:bf16[] .
                                let %1:f32[] = convert_element_type [data_type=f32] %0
                                in (%1)
                            },
                        ]
                            %3:f32[] = mul %2 %0
                        in (%3)
                    },
                ]
                in (%2)"},
        );

        let client = execution_client();
        let inputs = vec![values_to_bytes(&[1.1f32]), values_to_bytes(&[1f32])];
        let mut primal = execute_array_program(client, linearization.primal(), inputs);
        let value = values_from_bytes::<f32>(&primal.remove(0));
        let pullback = linearization.pullback().unwrap();
        let pullback_inputs = std::iter::once(values_to_bytes(&[1f32])).chain(primal).collect();
        let gradient = execute_array_program(client, pullback.program(), pullback_inputs);
        assert_eq!(value, vec![1.2109375f32]);
        assert_eq!(values_from_bytes::<f32>(&gradient[0]), vec![1.2109375f32]);
    }

    #[test]
    fn test_jit_call_split_transform_shares_boundary_metadata() {
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let source = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let arguments = JitCallSplitArguments {
            kind: JitCallSplitKind::Knownness(vec![true]),
            policy: None,
            residual_rounding_type: None,
            allow_effect_folding: true,
            allow_copy_folding: true,
            repeated_residual: false,
        };
        let derive = |region: RegionRef<'_, Constant, Operation>, _: &JitCallSplitArguments| {
            let residual = ArrayIrProgramBuilder::new().build::<Vec<Constant>, Vec<Constant>>(
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )?;
            Ok::<_, ProgramError>(TransformArtifact::new(
                vec![Arc::new(region.to_program()), Arc::new(residual)],
                Arc::new(JitCallSplit {
                    known_input_indices: vec![0],
                    residual_inputs: Vec::new(),
                    outputs: vec![PartialEvaluationOutput::Known(0)],
                }),
            ))
        };
        let first = source
            .entry_region_ref()
            .transform::<JitCallSplitTransform, _, _>(arguments.clone(), derive)
            .unwrap();
        let repeated = source.entry_region_ref().transform::<JitCallSplitTransform, _, _>(arguments, derive).unwrap();
        assert!(Arc::ptr_eq(first.metadata(), repeated.metadata()));
        assert!(Arc::ptr_eq(&first.programs()[0], &repeated.programs()[0]));
        assert!(Arc::ptr_eq(&first.programs()[1], &repeated.programs()[1]));
        assert_eq!(first.metadata().known_input_indices, vec![0]);
        assert_eq!(first.metadata().outputs, vec![PartialEvaluationOutput::Known(0)]);
        assert_eq!(
            first.programs()[0].to_string(),
            indoc! {"
                lambda %0:f32[] .
                in (%0)"},
        );
        assert_eq!(
            first.programs()[1].to_string(),
            indoc! {"
                lambda  .
                in ()"},
        );
    }

    #[test]
    fn test_jit_call_partial_evaluation_retains_splits_of_mixed_calls() {
        // Callee `g(k, x) = x ⊙ sin(k · k) + k` over a known `k` and an unknown `x`, whose known part computes
        // `sin(k · k)` and whose residual part computes the rest. The repeated workload chains `h ↦ g(k, h)` from `x`,
        // and the nested workload is the diamond `g_d(k, x) = g_{d-1}(k, x) + g_{d-1}(k, sin x)` with `g_0 = g`.
        let matrix_type = residual_demand_matrix_type();
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let known = builder.add_input(matrix_type.clone());
            let unknown = builder.add_input(matrix_type.clone());
            let dot = DotOperation::new(DotDimensionNumbers::new(vec![1], vec![0], vec![], vec![]));
            let product = builder.add_instruction(dot, Vec::new(), vec![known, known], None).unwrap()[0];
            let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![product], None).unwrap()[0];
            let scaled =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![unknown, sine], None).unwrap()[0];
            let output =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![scaled, known], None).unwrap()[0];
            Arc::new(
                builder
                    .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                    .unwrap(),
            )
        };
        let program = |body: &dyn Fn(&mut ArrayIrProgramBuilder, AtomId, AtomId) -> AtomId| {
            let mut builder = ArrayIrProgramBuilder::new();
            let known = builder.add_input(matrix_type.clone());
            let unknown = builder.add_input(matrix_type.clone());
            let output = body(&mut builder, known, unknown);
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let repeated = |count: usize| {
            program(&|builder, known, unknown| {
                (0..count).fold(unknown, |hidden, _| add_jit_call(builder, &callee, vec![known, hidden]))
            })
        };
        let nested = |depth: usize| {
            let mut nested_callee = callee.clone();
            for _ in 0..depth {
                let mut builder = ArrayIrProgramBuilder::new();
                let known = builder.add_input(matrix_type.clone());
                let unknown = builder.add_input(matrix_type.clone());
                let first = add_jit_call(&mut builder, &nested_callee, vec![known, unknown]);
                let shifted = builder.add_instruction(SinOperation::new(), Vec::new(), vec![unknown], None).unwrap()[0];
                let second = add_jit_call(&mut builder, &nested_callee, vec![known, shifted]);
                let output =
                    builder.add_instruction(AddOperation::new(), Vec::new(), vec![first, second], None).unwrap()[0];
                nested_callee = Arc::new(
                    builder
                        .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                        .unwrap(),
                );
            }
            program(&|builder, known, unknown| add_jit_call(builder, &nested_callee, vec![known, unknown]))
        };
        let closure_sizes = |program: &FlatProgram| {
            let partition = program.partition(&[true, false]).unwrap();
            let closure_size = |program: &FlatProgram| program.entry_region_ref().instructions_in_closure().count();
            [closure_size(partition.known_program()), closure_size(partition.residual_program())]
        };

        // Every call shares one known callee and one residual callee, so both programs grow by one call per call site.
        // Each nesting level adds its two known calls to the known program and its two residual calls, the shifted
        // input, and the sum to the residual program, rather than copying the callees of the level below for each
        // call, which made both programs grow exponentially with the nesting depth. The outer partition returns `k` as
        // a residual edge of its own, because `Program::partition` keeps one edge per residual feeder.
        assert_eq!([1, 2, 4, 8].map(|count| closure_sizes(&repeated(count))), [[3, 3], [4, 4], [6, 6], [10, 10]]);
        assert_eq!([1, 2, 3, 4].map(|depth| closure_sizes(&nested(depth))), [[5, 7], [7, 11], [9, 15], [11, 19]]);
        let partition = repeated(2).partition(&[true, false]).unwrap();
        assert_eq!(
            format!("{}\n{}", partition.known_program(), partition.residual_program()),
            indoc! {"
                lambda %0:f32[2, 2] .
                let %1:f32[2, 2] = jit_call %0 [
                    callee=^0={
                        lambda %0:f32[2, 2] .
                        let %1:f32[2, 2] = dot [
                            dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        ] %0 %0
                            %2:f32[2, 2] = sin %1
                        in (%2)
                    },
                ]
                    %2:f32[2, 2] = jit_call %0 [
                        callee=^0,
                    ]
                in (%1, %0, %2)
                lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2], %3:f32[2, 2] .
                let %4:f32[2, 2] = jit_call %0 %1 %2 [
                    callee=^0={
                        lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2] .
                        let %3:f32[2, 2] = mul %0 %1
                            %4:f32[2, 2] = add %3 %2
                        in (%4)
                    },
                ]
                    %5:f32[2, 2] = jit_call %4 %3 %2 [
                        callee=^0,
                    ]
                in (%5)"},
        );

        // The splits are retained by the region transform cache of the shared callee, so partitioning the same
        // workload again binds the very same known callee instead of partitioning the callee again.
        let first_callee = |program: &FlatProgram| {
            let region = program.instructions()[0].regions()[0];
            program.region_ref(region).unwrap().interned_callee().unwrap().clone()
        };
        let other_partition = repeated(4).partition(&[true, false]).unwrap();
        assert!(Arc::ptr_eq(&first_callee(partition.known_program()), &first_callee(other_partition.known_program())));

        // Executing the known program and then the residual program reproduces the outputs of the original program.
        let client = execution_client();
        let inputs = vec![vec![0.5f32, -1.0, 0.25, 2.0], vec![0.3f32, 0.1, -0.2, 0.4]];
        for program in [repeated(4), nested(3)] {
            let partition = program.partition(&[true, false]).unwrap();
            let known_inputs = partition.known_input_indices().iter().map(|index| inputs[*index].clone()).collect();
            let known_outputs = execute_f32_program(client, partition.known_program(), known_inputs);
            let known_output_count = partition.outputs().iter().filter(|output| output.is_known()).count();
            let residual_inputs = partition
                .residual_inputs()
                .iter()
                .map(|source| match *source {
                    ResidualInputSource::UnknownInput(index) | ResidualInputSource::KnownInput(index) => {
                        inputs[index].clone()
                    }
                    ResidualInputSource::KnownOutput(index) => known_outputs[index].clone(),
                    ResidualInputSource::ResidualEdge(edge) => known_outputs[known_output_count + edge].clone(),
                })
                .collect();
            let actual = execute_f32_program(client, partition.residual_program(), residual_inputs);
            let expected = execute_f32_program(client, &program, inputs.clone());
            assert_eq!(partition.outputs(), &[PartialEvaluationOutput::Unknown(0)]);
            assert_eq!(actual.len(), expected.len());
            assert!(
                actual[0]
                    .iter()
                    .zip(&expected[0])
                    .all(|(actual, expected)| (actual - expected).abs() <= 1e-5 * expected.abs().max(1.0)),
                "{actual:?} != {expected:?}",
            );
        }
    }

    #[test]
    fn test_jit_call_partial_evaluation_retains_splits_of_effectful_mixed_calls() {
        let callee = effectful_mixed_callee(EffectClass::UnorderedIo);
        let program = |callee: &Arc<FlatProgram>, count: usize| {
            let mut builder = ArrayIrProgramBuilder::new();
            let scalar_type: Type = ArrayType::scalar(DataType::F32).into();
            let known = builder.add_input(scalar_type.clone());
            let unknown = builder.add_input(scalar_type);
            let output = (0..count).fold(unknown, |value, _| add_jit_call(&mut builder, callee, vec![known, value]));
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let statistics = |program: &FlatProgram| {
            let partition = program.partition(&[true, false]).unwrap();
            let count = |program: &FlatProgram| {
                let region = program.entry_region_ref();
                [region.instructions_in_closure().count(), region.region_ids_in_closure().len()]
            };
            [count(partition.known_program()), count(partition.residual_program())]
        };

        // These counts precede lowering and count each shared callee once. Unordered effects let every known call
        // remain known, so sharing changes representation without changing the number or placement of runtime prints.
        assert_eq!(
            [1, 2, 4, 16].map(|count| statistics(&program(&callee, count))),
            [[[3, 2], [3, 2]], [[4, 2], [4, 2]], [[6, 2], [6, 2]], [[18, 2], [18, 2]]],
        );
        let partition = program(&callee, 1).partition(&[true, false]).unwrap();
        let other = program(&callee, 16).partition(&[true, false]).unwrap();
        let first_callee = |program: &FlatProgram| {
            program
                .region_ref(program.instructions()[0].regions()[0])
                .unwrap()
                .interned_callee()
                .unwrap()
                .clone()
        };
        assert!(Arc::ptr_eq(&first_callee(partition.known_program()), &first_callee(other.known_program())));
        assert!(Arc::ptr_eq(&first_callee(partition.residual_program()), &first_callee(other.residual_program())));

        let mut nested = callee;
        let mut counts = Vec::new();
        for _ in 0..6 {
            let mut builder = ArrayIrProgramBuilder::new();
            let scalar_type: Type = ArrayType::scalar(DataType::F32).into();
            let known = builder.add_input(scalar_type.clone());
            let unknown = builder.add_input(scalar_type);
            let first = add_jit_call(&mut builder, &nested, vec![known, unknown]);
            let second = add_jit_call(&mut builder, &nested, vec![known, unknown]);
            let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![first, second], None).unwrap()[0];
            nested = Arc::new(
                builder
                    .build::<Vec<Constant>, Vec<Constant>>(vec![sum], vec![Placeholder; 2], vec![Placeholder])
                    .unwrap(),
            );
            counts.push(statistics(&program(&nested, 1)));
        }
        assert_eq!(
            counts,
            vec![
                [[5, 3], [6, 3]],
                [[7, 4], [9, 4]],
                [[9, 5], [12, 5]],
                [[11, 6], [15, 6]],
                [[13, 7], [18, 7]],
                [[15, 8], [21, 8]],
            ],
        );
    }

    #[test]
    fn test_jit_call_partial_evaluation_caches_repeated_residual_modes_separately() {
        /// Supplies only the source callee and partition mode, while the test provides the partition closure directly.
        struct SplitCacheDriver {
            /// Callee shared across both partition modes.
            callee: Arc<FlatProgram>,

            /// Mode to include in the retained split's key.
            repeated_residual: bool,
        }

        impl RegionDriver<Constant, Operation> for SplitCacheDriver {
            fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, Constant, Operation>>
            where
                Constant: 'r,
                Operation: 'r,
            {
                std::iter::once(self.callee.entry_region_ref())
            }
        }

        impl PartialEvaluationDriver<TracingContext<Constant, Operation>> for SplitCacheDriver {
            fn partially_evaluate_region(
                &self,
                _: &PartialEvaluationContext<TracingContext<Constant, Operation>>,
                _: usize,
                _: Vec<PartialEvaluationValue<Tracer<TracingContext<Constant, Operation>>>>,
            ) -> Result<Vec<PartialEvaluationValue<Tracer<TracingContext<Constant, Operation>>>>, ProgramError>
            {
                unreachable!("the split cache fixture does not recursively evaluate regions")
            }

            fn partially_evaluate_program(
                &self,
                _: &PartialEvaluationContext<TracingContext<Constant, Operation>>,
                _: RegionRef<'_, Constant, Operation>,
                _: &[PartialValue<Tracer<TracingContext<Constant, Operation>>>],
            ) -> Result<PartialEvaluation<TracingContext<Constant, Operation>>, ProgramError> {
                unreachable!("the split cache fixture does not recursively evaluate programs")
            }

            fn partition_program(
                &self,
                _: &PartialEvaluationContext<TracingContext<Constant, Operation>>,
                _: RegionRef<'_, Constant, Operation>,
                _: &[bool],
            ) -> Result<PartitionedProgram<Constant, Operation>, ProgramError> {
                unreachable!("the split cache fixture supplies the partition closure directly")
            }

            fn partitions_for_repeated_residual_calls(&self) -> bool {
                self.repeated_residual
            }
        }

        let callee = effectful_mixed_callee(EffectClass::UnorderedIo);
        let scalar_type: Type = ArrayType::scalar(DataType::F32).into();
        let mut builder = ArrayIrProgramBuilder::new();
        let known = builder.add_input(scalar_type.clone());
        let squared = builder.add_instruction(MulOperation::new(), Vec::new(), vec![known, known], None).unwrap()[0];
        let ordinary_known = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![squared], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ArrayIrProgramBuilder::new();
        let unknown = builder.add_input(scalar_type.clone());
        let coefficient = builder.add_input(scalar_type.clone());
        let printed = builder
            .add_instruction(
                PrintOperation::<ArrayType>::new("known").with_effect_class(EffectClass::UnorderedIo),
                Vec::new(),
                vec![coefficient],
                None,
            )
            .unwrap()[0];
        let product =
            builder.add_instruction(MulOperation::new(), Vec::new(), vec![unknown, printed], None).unwrap()[0];
        let output = builder
            .add_instruction(
                PrintOperation::<ArrayType>::new("residual").with_effect_class(EffectClass::UnorderedIo),
                Vec::new(),
                vec![product],
                None,
            )
            .unwrap()[0];
        let ordinary_residual = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let partitions = std::cell::Cell::new(0);
        let mut emitted_counts = Vec::new();
        for repeated_residual in [false, false, true, true] {
            let driver = SplitCacheDriver { callee: callee.clone(), repeated_residual };
            let outer = TracingContext::<Constant, Operation>::new();
            let known = outer.input(scalar_type.clone());
            let context = PartialEvaluationContext::new(outer).with_allow_effect_folding(false);
            let inputs = [context.known_value(known), context.unknown_input(scalar_type.clone(), 1)];
            let partition = |callee: RegionRef<'_, Constant, Operation>| {
                partitions.set(partitions.get() + 1);
                if repeated_residual {
                    // Repeated-call partitioning permits known effects in its fresh analysis despite the caller's
                    // effect folding restriction. Emission still defers those effects through the active context.
                    callee.partition(&[true, false])
                } else {
                    // Ordinary specialization inherits that restriction, so only the pure square remains known.
                    PartitionedProgram::from_parts(
                        ordinary_known.clone(),
                        ordinary_residual.clone(),
                        2,
                        vec![0],
                        vec![ResidualInputSource::UnknownInput(1), ResidualInputSource::ResidualEdge(0)],
                        vec![PartialEvaluationOutput::Unknown(0)],
                    )
                }
            };
            let outputs = split_jit_call(
                &JitCallOperation::new(0),
                &context,
                &driver,
                &inputs,
                JitCallSplitKind::Knownness(vec![true, false]),
                &partition,
            )
            .unwrap();
            let known_count = context.parent().builder().borrow().instructions().len();
            drop(inputs);
            let evaluation = context.into_evaluation(outputs).unwrap();
            emitted_counts.push([known_count, evaluation.program().instructions().len()]);
        }
        assert_eq!(emitted_counts, vec![[1, 1], [1, 1], [0, 2], [0, 2]]);
        // Debug builds re-derive every cache hit to validate deterministic derivation.
        assert_eq!(partitions.get(), if cfg!(debug_assertions) { 4 } else { 2 });
    }

    #[test]
    fn test_jit_call_partial_evaluation_cached_effectful_splits_preserve_order() {
        let callee = effectful_mixed_callee(EffectClass::OrderedIo);
        let program = |prefix: bool| {
            let mut builder = ArrayIrProgramBuilder::new();
            let scalar_type: Type = ArrayType::scalar(DataType::F32).into();
            let known = builder.add_input(scalar_type.clone());
            let unknown = builder.add_input(scalar_type);
            if prefix {
                builder
                    .add_instruction(PrintOperation::<ArrayType>::new("before"), Vec::new(), vec![unknown], None)
                    .unwrap();
            }
            let output = add_jit_call(&mut builder, &callee, vec![known, unknown]);
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let policy = ResidualPolicyReference::new(NothingSavable);
        let warm = program(false).partition_with_residual_policy(&[true, false], &policy).unwrap();
        let partition = program(true).partition_with_residual_policy(&[true, false], &policy).unwrap();
        assert!(partition.known_program().instructions().is_empty());
        let known_callee =
            warm.known_program().region_ref(warm.known_program().instructions()[0].regions()[0]).unwrap();
        let cached_callee = partition
            .residual_program()
            .region_ref(partition.residual_program().instructions()[1].regions()[0])
            .unwrap();
        assert!(Arc::ptr_eq(known_callee.interned_callee().unwrap(), cached_callee.interned_callee().unwrap()));

        let client = execution_client();
        ensure_print_handler_registered(client).unwrap();
        let (known, lines) =
            with_captured_prints(|| execute_f32_program(client, warm.known_program(), vec![vec![2.0]]));
        assert_eq!(lines, vec!["known: 4.0"]);
        let known_output_count = warm.outputs().iter().filter(|output| output.is_known()).count();
        for unknown in [3.0, 5.0] {
            let product = unknown * 4.0;
            let inputs = warm
                .residual_inputs()
                .iter()
                .map(|source| match *source {
                    ResidualInputSource::UnknownInput(1) => vec![unknown],
                    ResidualInputSource::KnownInput(0) => vec![2.0],
                    ResidualInputSource::KnownOutput(index) => known[index].clone(),
                    ResidualInputSource::ResidualEdge(index) => known[known_output_count + index].clone(),
                    source => panic!("unexpected residual input {source:?}"),
                })
                .collect();
            let (outputs, lines) =
                with_captured_prints(|| execute_f32_program(client, warm.residual_program(), inputs));
            assert_eq!(outputs, vec![vec![product]]);
            assert_eq!(lines, vec![format!("residual: {product:?}")]);

            let inputs = partition
                .residual_inputs()
                .iter()
                .map(|source| match *source {
                    ResidualInputSource::UnknownInput(1) => vec![unknown],
                    ResidualInputSource::KnownInput(0) | ResidualInputSource::ResidualEdge(_) => vec![2.0],
                    source => panic!("unexpected residual input {source:?}"),
                })
                .collect();
            let (outputs, lines) =
                with_captured_prints(|| execute_f32_program(client, partition.residual_program(), inputs));
            assert_eq!(outputs, vec![vec![product]]);
            assert_eq!(
                lines,
                vec![format!("before: {unknown:?}"), "known: 4.0".to_string(), format!("residual: {product:?}")],
            );
        }
    }

    #[test]
    fn test_jit_call_partial_evaluation_cached_effectful_splits_keep_outputless_residuals() {
        let scalar_type: Type = ArrayType::scalar(DataType::F32).into();
        let mut builder = ArrayIrProgramBuilder::new();
        let known = builder.add_input(scalar_type.clone());
        let unknown = builder.add_input(scalar_type.clone());
        let squared = builder.add_instruction(MulOperation::new(), Vec::new(), vec![known, known], None).unwrap()[0];
        builder
            .add_instruction(PrintOperation::<ArrayType>::new("residual"), Vec::new(), vec![unknown], None)
            .unwrap();
        let callee = Arc::new(
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![squared], vec![Placeholder; 2], vec![Placeholder])
                .unwrap(),
        );
        let mut builder = ArrayIrProgramBuilder::new();
        let known = builder.add_input(scalar_type.clone());
        let unknown = builder.add_input(scalar_type);
        let output = add_jit_call(&mut builder, &callee, vec![known, unknown]);
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let warm = program.partition(&[true, false]).unwrap();
        let partition = program.partition(&[true, false]).unwrap();
        assert_eq!(partition.outputs(), &[PartialEvaluationOutput::Known(0)]);
        assert_eq!(partition.residual_program().output_count(), 0);
        let client = execution_client();
        ensure_print_handler_registered(client).unwrap();
        assert_eq!(execute_f32_program(client, partition.known_program(), vec![vec![2.0]]), vec![vec![4.0]]);
        for unknown in [3.0, 5.0] {
            let (outputs, lines) =
                with_captured_prints(|| execute_f32_program(client, partition.residual_program(), vec![vec![unknown]]));
            assert_eq!(outputs, Vec::<Vec<f32>>::new());
            assert_eq!(lines, vec![format!("residual: {unknown:?}")]);
        }
        assert_eq!(warm.residual_program().to_string(), partition.residual_program().to_string());
    }

    #[test]
    fn test_jit_call_partial_evaluation_shares_callees_of_repeated_differentiated_calls() {
        let callee = residual_demand_leaf_callee();
        let repeated = |count: usize, policy: Option<ResidualPolicyReference<Type>>| {
            residual_demand_program(policy, |builder, weights, bias| {
                (0..count).fold(weights, |hidden, _| add_jit_call(builder, &callee, vec![hidden, bias]))
            })
        };
        let inlined = |count: usize| {
            residual_demand_program(None, |builder, weights, bias| {
                (0..count).fold(weights, |hidden, _| add_residual_demand_leaf(builder, hidden, bias))
            })
        };

        // Differentiating `w` as well as `x` differentiates every call. Linearization binds one primal call and one
        // tangent call of the shared linearized callee per call site, and the tangent call mixes unknown tangents with
        // known residuals. Every call shares one primal callee and one tangent callee, so the programs grow by one call
        // per call site, while the residuals grow with the number of calls as in JAX. Copying the callees per call
        // site made the programs grow by eight and nine instructions per call.
        assert_eq!(
            [1, 2, 4, 8].map(|count| residual_demand_statistics(&repeated(count, None), &[0, 1])),
            [[7, 9, 12], [12, 10, 13], [22, 12, 15], [42, 16, 19]],
        );
        let policy = ResidualPolicyReference::new(DotsSavable);
        assert_eq!(
            [1, 2, 4, 8].map(|count| residual_demand_statistics(&repeated(count, Some(policy.clone())), &[0, 1])),
            [[4, 7, 22], [5, 8, 24], [7, 10, 28], [11, 14, 36]],
        );
        let linearization = repeated(2, None).linearize_with_respect_to(&[0, 1]).unwrap();
        assert_eq!(
            format!("{}\n{}", linearization.primal(), linearization.tangent()),
            indoc! {"
                lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2] .
                let %3:f32[2, 2], %4:f32[2, 2], %5:f32[2, 2], %6:f32[2, 2], %7:f32[2, 2], %8:f32[2, 2] = jit_call %1 %2 [
                    callee=^0={
                        lambda %0:f32[2, 2], %1:f32[2, 2] .
                        let %2:f32[2, 2] = dot [
                            dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        ] %0 %1
                            %3:f32[2, 2] = sin %2
                            %4:f32[2, 2] = cos %2
                            %5:f32[2, 2] = mul %3 %4
                            %6:f32[2, 2] = add %5 %0
                            %7:f32[2, 2] = cos %2
                            %8:f32[2, 2] = sin %2
                        in (%6, %1, %7, %8, %4, %3)
                    },
                ]
                    %9:f32[2, 2], %10:f32[2, 2], %11:f32[2, 2], %12:f32[2, 2], %13:f32[2, 2], %14:f32[2, 2] = jit_call %3 %2 [
                        callee=^0,
                    ]
                    %15:f32[2, 2] = mul %0 %9
                in (%15, %4, %5, %6, %7, %8, %10, %11, %12, %13, %14, %9, %0)
                lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2], %3:f32[2, 2], %4:f32[2, 2], %5:f32[2, 2], %6:f32[2, 2], %7:f32[2, 2], %8:f32[2, 2], %9:f32[2, 2], %10:f32[2, 2], %11:f32[2, 2], %12:f32[2, 2], %13:f32[2, 2] .
                let %14:f32[2, 2] = mul %12 %0
                    %15:f32[2, 2] = jit_call %1 %2 %3 %4 %5 %6 [
                        callee=^0={
                            lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2], %3:f32[2, 2], %4:f32[2, 2], %5:f32[2, 2] .
                            let %6:f32[2, 2] = dot [
                                dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            ] %0 %1
                                %7:f32[2, 2] = mul %2 %6
                                %8:f32[2, 2] = mul %4 %7
                                %9:f32[2, 2] = mul %3 %6
                                %10:f32[2, 2] = neg %9
                                %11:f32[2, 2] = mul %5 %10
                                %12:f32[2, 2] = add %8 %11
                                %13:f32[2, 2] = add %12 %0
                            in (%13)
                        },
                    ]
                    %16:f32[2, 2] = jit_call %15 %7 %8 %9 %10 %11 [
                        callee=^0,
                    ]
                    %17:f32[2, 2] = mul %13 %16
                    %18:f32[2, 2] = add %14 %17
                in (%18)"},
        );

        // Neither sharing nor rematerialization changes the primal value or the derivatives, which match those of the
        // same function with every call inlined.
        let client = execution_client();
        assert_residual_demand_linearizations_match(client, &repeated(4, None), &inlined(4), &[0, 1]);
        assert_residual_demand_linearizations_match(client, &repeated(4, Some(policy)), &inlined(4), &[0, 1]);
    }

    #[test]
    fn test_jit_call_partial_evaluation_shares_callees_of_nested_differentiated_calls() {
        let nested = |depth: usize, policy: Option<ResidualPolicyReference<Type>>| {
            let callee = residual_demand_diamond_callee(depth);
            residual_demand_program(policy, |builder, weights, bias| {
                add_jit_call(builder, &callee, vec![weights, bias])
            })
        };
        let inlined = |depth: usize| {
            residual_demand_program(None, |builder, weights, bias| {
                add_inlined_residual_demand_diamond(builder, depth, weights, bias)
            })
        };

        // The primal and tangent callees of each level call the shared callees of the level below them twice, so the
        // programs grow by a few instructions per level, while the residuals double with every level as in JAX.
        // Copying the callees per call site instead doubled the programs with every level.
        assert_eq!(
            [1, 2, 3, 4].map(|depth| residual_demand_statistics(&nested(depth, None), &[0, 1])),
            [[13, 14, 16], [25, 19, 20], [49, 24, 24], [97, 29, 28]],
        );
        let policy = ResidualPolicyReference::new(DotsSavable);
        assert_eq!(
            [1, 2, 3, 4].map(|depth| residual_demand_statistics(&nested(depth, Some(policy.clone())), &[0, 1])),
            [[5, 11, 31], [7, 15, 41], [11, 19, 51], [19, 23, 61]],
        );

        let client = execution_client();
        assert_residual_demand_linearizations_match(client, &nested(3, None), &inlined(3), &[0, 1]);
        assert_residual_demand_linearizations_match(client, &nested(3, Some(policy)), &inlined(3), &[0, 1]);
    }

    #[test]
    fn test_jit_call_batching() {
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let callee = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let callees = [Arc::new(callee)];
        let regions = AttachedRegions::Callees(&callees);
        let driver = RecursiveBatchingDriver::new(&regions);
        let vector = Type::Array(ArrayType::new_static(DataType::F32, [3]));
        let (_, program) = TracingContext::<Constant, Operation>::trace(
            |inputs: Vec<Tracer<TracingContext<Constant, Operation>>>| {
                let parent = inputs[0].context();
                let extent = parent.constant(Constant::Dimension(DimensionValue::constant(3).unwrap()));
                let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent.clone(), extent);
                let outputs = JitCallOperation::<Type>::new(0).batch(
                    &context,
                    &driver,
                    &[ArrayIrBatch::new(inputs[0].clone(), Some(0))?],
                )?;
                Ok::<_, ProgramError>(outputs.into_iter().map(ArrayIrBatch::into_value).collect::<Vec<_>>())
            },
            vec![vector.clone()],
        )
        .unwrap();
        assert_eq!(program.input_types(), vec![vector.clone()]);
        assert_eq!(program.output_types(), vec![vector]);
        let Operation::JitCall(call) = program.instructions()[0].operation() else {
            panic!("expected jit_call");
        };
        assert_eq!(call.capture_count(), 1);

        // The batched callee receives the extent as its leading input, and it does not return the extent, because that
        // bookkeeping output would only forward the input and an eager parent cannot return it from a compiled call.
        let body = program.region_ref(program.instructions()[0].regions()[0]).unwrap();
        assert_eq!(body.input_types().len(), 2);
        assert_eq!(body.output_ids(), &[body.input_ids()[1]]);
    }

    #[test]
    fn test_jit_call_batching_places_inputs_at_the_boundary_of_the_original_call() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let item = ArrayType::new_static(DataType::F32, [4]);
        let batch_split =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated])
                .unwrap();
        let batch_type =
            Type::Array(ArrayType::new_static(DataType::F32, [2, 4]).with_sharding(batch_split.clone()).unwrap());
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(item.into());
            let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let callees = [Arc::new(callee)];
        let regions = AttachedRegions::Callees(&callees);
        let driver = RecursiveBatchingDriver::new(&regions);
        let batched = |call: JitCallOperation<Type>| {
            let (_, program) = TracingContext::<Constant, Operation>::trace(
                |inputs: Vec<Tracer<TracingContext<Constant, Operation>>>| {
                    let parent = inputs[0].context();
                    let extent = parent.constant(Constant::Dimension(DimensionValue::constant(2).unwrap()));
                    let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent.clone(), extent);
                    let outputs = call.batch(&context, &driver, &[ArrayIrBatch::new(inputs[0].clone(), Some(0))?])?;
                    Ok::<_, ProgramError>(outputs.into_iter().map(ArrayIrBatch::into_value).collect::<Vec<_>>())
                },
                vec![batch_type.clone()],
            )
            .unwrap();
            let Operation::JitCall(call) = program.instructions()[0].operation() else {
                panic!("expected jit_call");
            };
            call.derived_input_shardings.as_deref().map(<[_]>::to_vec)
        };

        // The batched callee is created by batching, so the batched call is a derived call, which places each item
        // where the original call places it, with the batch dimension placed as the batched value is, rather than as
        // the traced types of the batched callee say. The callee declares an item type without a sharding, which the
        // original call places replicated over the mesh where it runs, so the batch stays split. A derived original
        // call keeps its own boundary, and a call of a `jit` dispatcher places nothing. The leading extent input has
        // no boundary.
        let derived = |shardings: Vec<Option<Sharding>>| JitCallOperation::<Type> {
            derived_input_shardings: Some(Arc::from(shardings)),
            ..JitCallOperation::new(0)
        };
        assert_eq!(batched(JitCallOperation::new(0)), Some(vec![None, Some(batch_split.clone())]));
        assert_eq!(
            batched(derived(vec![Some(Sharding::replicated(mesh.clone(), 1))])),
            Some(vec![None, Some(batch_split)])
        );
        assert_eq!(batched(derived(vec![None])), Some(vec![None, None]));
        assert_eq!(batched(JitCallOperation::new(0).with_kept_input_placements()), Some(vec![None, None]));
    }

    #[test]
    fn test_jit_call_batching_homogeneous_named_axis() {
        /// Array-only family for exercising the homogeneous `jit_call` batching rule without array IR carriers.
        #[derive(Clone, Debug, Operation)]
        #[ryft(crate = "ryft_core", type(ArrayType), constant(CoreArray), dispatch(batching))]
        enum NamedAxisCallOperation {
            AxisIndex(AxisIndexOperation),
            Iota(IotaOperation<ArrayType>),
            Broadcast(BroadcastOperation),
            Transpose(TransposeOperation),
            Zero(ZeroOperation<ArrayType>),
            Call(JitCallOperation<ArrayType>),
        }

        for replicated in [false, true] {
            let mut builder = CoreProgramBuilder::<CoreArray, NamedAxisCallOperation>::new();
            let input = replicated.then(|| builder.add_input(ArrayType::scalar(DataType::F32)));
            let index =
                builder.add_instruction(AxisIndexOperation::new("items".to_owned()), vec![], vec![], None).unwrap()[0];
            let outputs = std::iter::once(index).chain(input).collect::<Vec<_>>();
            let callee = builder
                .build::<Vec<CoreArray>, Vec<CoreArray>>(
                    outputs,
                    vec![Placeholder; usize::from(replicated)],
                    vec![Placeholder; 1 + usize::from(replicated)],
                )
                .unwrap();
            let callees = [Arc::new(callee)];
            let regions = AttachedRegions::Callees(&callees);
            let driver = RecursiveBatchingDriver::new(&regions);
            let (_, program) = TracingContext::<CoreArray, NamedAxisCallOperation>::trace_with_context_and_named_axes(
                |parent, inputs: Vec<Tracer<TracingContext<CoreArray, NamedAxisCallOperation>>>| {
                    let context = BatchingContext::<_, ArrayBatchingPolicy>::new(parent.clone(), 3)
                        .with_axis_name("items".to_owned());
                    let inputs = inputs.into_iter().map(ArrayBatch::replicated).collect::<Vec<_>>();
                    let outputs = JitCallOperation::<ArrayType>::new(0).batch(&context, &driver, &inputs)?;
                    assert_eq!(outputs[0].batch_axis_position(), Some(0));
                    if replicated {
                        assert_eq!(outputs[1].batch_axis_position(), None);
                    }
                    Ok::<_, ProgramError>(outputs.into_iter().map(ArrayBatch::into_value).collect::<Vec<_>>())
                },
                vec![ArrayType::scalar(DataType::F32); usize::from(replicated)],
                vec![],
            )
            .unwrap();
            let expected = if replicated {
                indoc! {"
                    lambda %0:f32[] .
                    let %1:u64[3], %2:f32[] = jit_call %0 [
                        callee={
                            lambda %0:f32[] .
                            let %1:u64[3] = iota [type=u64[3], dimension=0]
                            in (%1, %0)
                        },
                    ]
                    in (%1, %2)"}
            } else {
                indoc! {"
                    lambda  .
                    let %0:u64[3] = jit_call [
                        callee={
                            lambda  .
                            let %0:u64[3] = iota [type=u64[3], dimension=0]
                            in (%0)
                        },
                    ]
                    in (%0)"}
            };
            assert_eq!(program.to_string(), expected);
            assert_eq!(program.output_types()[0], ArrayType::new_static(DataType::U64, [3]));
        }
    }

    #[test]
    fn test_jit_call_differentiation_omits_zero_space_boundary_tangents() {
        let dimension_type = Type::Dimension(DimensionType::new("size", DimensionBounds::positive(Some(9)).unwrap()));
        let array_type = Type::Array(ArrayType::scalar(DataType::F64));

        let mut callee_builder = ArrayIrProgramBuilder::new();
        let dimension = callee_builder.add_input(dimension_type.clone());
        let array = callee_builder.add_input(array_type.clone());
        let callee = callee_builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![dimension, array], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let mut builder = ArrayIrProgramBuilder::new();
        let dimension = builder.add_input(dimension_type.clone());
        let array = builder.add_input(array_type.clone());
        let callee = builder.import_region(callee.entry_region_ref());
        let outputs = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee], vec![dimension, array], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_types(), vec![dimension_type.clone(), array_type.clone(), array_type.clone()]);
        assert_eq!(jvp.output_types(), vec![dimension_type, array_type.clone(), array_type]);
    }

    #[test]
    fn test_jit_call_differentiation_keeps_capture_prefix_inactive() {
        // The callee's leading input is its lifted capture (`capture_count = 1`). The call's activity mask starts
        // inactive over that prefix, numeric captures included, so the fused callee consumes the primal operands,
        // capture prefix first, followed only by the live operand tangent.
        let r#type = Type::Array(vector_type());
        let callee = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let captured = builder.add_input(r#type.clone());
            let input = builder.add_input(r#type.clone());
            let product =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![captured, input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![product], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let captured = builder.add_input(r#type.clone());
        let input = builder.add_input(r#type);
        let output = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(1)), vec![callee], vec![captured, input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4], %2:f64[4], %3:f64[4] .
                let %4:f64[4], %5:f64[4] = jit_call [capture_count=1] %0 %1 %3 [
                    callee={
                        lambda %0:f64[4], %1:f64[4], %2:f64[4] .
                        let %3:f64[4] = mul %0 %1
                            %4:f64[4] = mul %0 %2
                        in (%3, %4)
                    },
                ]
                in (%4, %5)"},
        );
    }

    #[test]
    fn test_jit_call_differentiation_threads_plumbing_and_active_references() {
        // The callee `f(captured, r, x) = { r += x; read(r) * read(captured) }` behind `jit_call[capture_count=1]`
        // reads a captured plumbing reference and mutates an active reference operand.
        let scalar_type = Type::Array(ArrayType::scalar(DataType::F32));
        let reference_type = Type::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let captured = builder.add_input(reference_type.clone());
            let reference = builder.add_input(reference_type.clone());
            let update = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(
                    Operation::ReferenceAddUpdate(ReferenceAddUpdateOperation::new()),
                    Vec::new(),
                    vec![reference, update],
                    None,
                )
                .unwrap();
            let state = builder
                .add_instruction(
                    Operation::ReferenceRead(ReferenceReadOperation::new()),
                    Vec::new(),
                    vec![reference],
                    None,
                )
                .unwrap()[0];
            let scale = builder
                .add_instruction(
                    Operation::ReferenceRead(ReferenceReadOperation::new()),
                    Vec::new(),
                    vec![captured],
                    None,
                )
                .unwrap()[0];
            let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![state, scale], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let captured = builder.add_input(reference_type.clone());
        let reference = builder.add_input(reference_type.clone());
        let update = builder.add_input(scalar_type.clone());
        let output = builder
            .add_instruction(
                Operation::JitCall(JitCallOperation::new(1)),
                vec![callee],
                vec![captured, reference, update],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        // The plumbing capture is inactive and the active reference receives a tangent reference input, so the fused
        // program and its one fused call consume `[captured, r, x, ṫr, ẋ]`, keeping the capture prefix verbatim.
        let jvp = program.entry_region_ref().jvp(&[1, 2]).unwrap();
        assert_eq!(
            jvp.input_types(),
            vec![
                reference_type.clone(),
                reference_type.clone(),
                scalar_type.clone(),
                reference_type.clone(),
                scalar_type.clone(),
            ],
        );
        assert_eq!(jvp.output_types(), vec![scalar_type.clone(), scalar_type.clone()]);
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:ref<f32[]>, %1:ref<f32[]>, %2:f32[], %3:ref<f32[]>, %4:f32[] .
                let %5:f32[], %6:f32[] = jit_call [capture_count=1] %0 %1 %2 %3 %4 [
                    callee={
                        lambda %0:ref<f32[]>, %1:ref<f32[]>, %2:f32[], %3:ref<f32[]>, %4:f32[] .
                        let () = reference_add_update %1 %2
                            () = reference_add_update %3 %4
                            %5:f32[] = reference_read %1
                            %6:f32[] = reference_read %3
                            %7:f32[] = reference_read %0
                            %8:f32[] = mul %5 %7
                            %9:f32[] = mul %7 %6
                        in (%8, %9)
                    },
                ]
                in (%5, %6)"},
        );
    }

    #[test]
    fn test_jit_call_differentiation_omits_inactive_reference_output_tangents() {
        let reference_type = Type::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let scalar_type = Type::Array(ArrayType::scalar(DataType::F32));
        let mut callee_builder = ArrayIrProgramBuilder::new();
        let reference = callee_builder.add_input(reference_type.clone());
        let value = callee_builder.add_input(scalar_type.clone());
        let doubled =
            callee_builder.add_instruction(AddOperation::new(), Vec::new(), vec![value, value], None).unwrap()[0];
        let callee = callee_builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![reference, doubled], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let mut builder = ArrayIrProgramBuilder::new();
        let reference = builder.add_input(reference_type.clone());
        let value = builder.add_input(scalar_type.clone());
        let callee = builder.import_region(callee.entry_region_ref());
        let outputs = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(1)), vec![callee], vec![reference, value], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let jvp = program.entry_region_ref().jvp(&[1]).unwrap();
        assert_eq!(jvp.input_types(), vec![reference_type.clone(), scalar_type.clone(), scalar_type.clone()]);
        assert_eq!(jvp.output_types(), vec![reference_type, scalar_type.clone(), scalar_type]);
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:ref<f32[]>, %1:f32[], %2:f32[] .
                let %3:ref<f32[]>, %4:f32[], %5:f32[] = jit_call [capture_count=1] %0 %1 %2 [
                    callee={
                        lambda %0:ref<f32[]>, %1:f32[], %2:f32[] .
                        let %3:f32[] = add %1 %1
                            %4:f32[] = add %2 %2
                        in (%0, %3, %4)
                    },
                ]
                in (%3, %4, %5)"},
        );
    }

    #[test]
    fn test_jit_call_differentiation_treats_structurally_zero_tangents_as_inactive() {
        // When `f(x, y) = (sin(x), cos(y))` is differentiated with respect to `x` only, the structurally zero tangent
        // of `y` gets no fused callee slot, and the tangent of `cos(y)`, which then depends on no tangent input, is a
        // structural zero that the fused call does not return.
        let r#type = Type::Array(vector_type());
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let x = builder.add_input(r#type.clone());
            let y = builder.add_input(r#type.clone());
            let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
            let cosine = builder.add_instruction(CosOperation::new(), Vec::new(), vec![y], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![sine, cosine], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let inputs = (0..2).map(|_| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let outputs = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee], inputs, None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.entry_region_ref().jvp(&[0]).unwrap().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4], %2:f64[4] .
                let %3:f64[4], %4:f64[4], %5:f64[4] = jit_call %0 %1 %2 [
                    callee={
                        lambda %0:f64[4], %1:f64[4], %2:f64[4] .
                        let %3:f64[4] = sin %0
                            %4:f64[4] = cos %1
                            %5:f64[4] = cos %0
                            %6:f64[4] = mul %5 %2
                        in (%3, %4, %6)
                    },
                ]
                    %6:f64[4] = zero [type=f64[4]]
                in (%3, %4, %5, %6)"},
        );
    }

    #[test]
    fn test_jit_call_differentiation_omits_independent_output_tangents() {
        // In `f(x) = (sin(x), 0)`, the tangent of the constant output depends on no tangent input, so it is a
        // structural zero that the fused call does not return.
        let r#type = Type::Array(vector_type());
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let x = builder.add_input(r#type.clone());
            let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
            let zero =
                builder.add_instruction(ZeroOperation::new(vector_type()), Vec::new(), Vec::new(), None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![sine, zero], vec![Placeholder], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let input = builder.add_input(r#type);
        let outputs = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee], vec![input], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4] .
                let %2:f64[4], %3:f64[4], %4:f64[4] = jit_call %0 %1 [
                    callee={
                        lambda %0:f64[4], %1:f64[4] .
                        let %2:f64[4] = sin %0
                            %3:f64[4] = zero [type=f64[4]]
                            %4:f64[4] = cos %0
                            %5:f64[4] = mul %4 %1
                        in (%2, %3, %5)
                    },
                ]
                    %5:f64[4] = zero [type=f64[4]]
                in (%2, %3, %4, %5)"},
        );
    }

    #[test]
    fn test_jit_call_differentiation_preserves_live_outputs_around_inactive_tangent() {
        // `f(x) = (sin(x), 0, x)` has an inactive middle tangent slot. Projecting that slot must leave both live
        // tangents aligned with their original primal outputs rather than pairing the final tangent with the zero.
        let r#type = Type::Array(vector_type());
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(r#type.clone());
            let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
            let zero =
                builder.add_instruction(ZeroOperation::new(vector_type()), Vec::new(), Vec::new(), None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![sine, zero, input], vec![Placeholder], vec![Placeholder; 3])
                .unwrap()
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let input = builder.add_input(r#type.clone());
        let outputs = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee], vec![input], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder; 3])
            .unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.output_types(), vec![r#type; 6]);
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4] .
                let %2:f64[4], %3:f64[4], %4:f64[4], %5:f64[4], %6:f64[4] = jit_call %0 %1 [
                    callee={
                        lambda %0:f64[4], %1:f64[4] .
                        let %2:f64[4] = sin %0
                            %3:f64[4] = zero [type=f64[4]]
                            %4:f64[4] = cos %0
                            %5:f64[4] = mul %4 %1
                        in (%2, %3, %0, %5, %1)
                    },
                ]
                    %7:f64[4] = zero [type=f64[4]]
                in (%2, %3, %4, %5, %7, %6)"},
        );
    }

    #[test]
    fn test_jit_call_differentiation_skips_tangent_calls_without_live_output_tangents() {
        // When `f(x, y) = sin(x)` is differentiated with respect to `y` only, no output tangent depends on the tangent
        // of `y` and the tangent callee has no effects, so only the source call is bound and the output tangent is a
        // structural zero.
        let r#type = Type::Array(vector_type());
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let x = builder.add_input(r#type.clone());
            builder.add_input(r#type.clone());
            let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![sine], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let inputs = (0..2).map(|_| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let output = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee], inputs, None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.entry_region_ref().jvp(&[1]).unwrap().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4], %2:f64[4] .
                let %3:f64[4] = jit_call %0 %1 [
                    callee={
                        lambda %0:f64[4], %1:f64[4] .
                        let %2:f64[4] = sin %0
                        in (%2)
                    },
                ]
                    %4:f64[4] = zero [type=f64[4]]
                in (%3, %4)"},
        );
    }

    #[test]
    fn test_jit_call_differentiation_preserves_local_result_shape_relationships() {
        // `f(x, size) = broadcast(x, [dimension_from_scalar(size)])` is linear in `x` and returns an array whose extent
        // the callee defines itself. The JVP rule binds separate primal and tangent calls, so the output tangent shares
        // the primal output's extent only if that extent reaches the tangent call through a residual.
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharding = Sharding::replicated(logical_mesh, 0);
        let value_type = Type::Array(ArrayType::scalar(DataType::F32).with_sharding(sharding.clone()).unwrap());
        let size_type = Type::Array(ArrayType::scalar(DataType::I64).with_sharding(sharding).unwrap());
        let rows = DimensionType::from(DimensionVariable::new("rows", DimensionBounds::non_negative(Some(8)).unwrap()));
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let value = builder.add_input(value_type.clone());
            let size = builder.add_input(size_type.clone());
            let dimension = builder
                .add_instruction(DimensionFromScalarOperation::new(rows), Vec::new(), vec![size], None)
                .unwrap()[0];
            let broadcast = builder
                .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![value, dimension], None)
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![broadcast], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let operation = JitCallOperation::new(0)
            .with_callee_type_signature(&callee.input_types(), &callee.output_types())
            .unwrap();
        let mut builder = ArrayIrProgramBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let inputs = vec![builder.add_input(value_type), builder.add_input(size_type)];
        let output = builder.add_instruction(Operation::JitCall(operation), vec![callee], inputs, None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let outputs = program.jvp().unwrap().output_types().into_owned();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[1], outputs[0].tangent().unwrap());
    }

    #[test]
    fn test_jit_call_operation_linearize_preserves_call_boundaries() {
        // Driving a `jit_call` program through the forward path without captures (i.e., `Program::linearize`, which
        // dispatches the `jvp` rule of `JitCallOperation`) realizes `jvp(jit(f)) = jit(jvp f)`, because the split keeps
        // both compilation boundaries instead of inlining the callee.
        //
        // For `f = |x| x.sin()` wrapped in an outer `jit_call`, the primal half of the split is itself a single
        // `jit_call` that produces the primal output plus one residual, and its tangent half is a single `jit_call`
        // that consumes that residual. Neither half inlines `Sin` or `Cos`. Rewrapping the two halves as a compiled
        // `(x, dx) -> (primal, tangent)` function reproduces the result of `CompiledFunction::jvp`, which is
        // `(sin(x), cos(x) * dx)`, within `1e-5`.

        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();

        // Inner `jit(sin)` staged as one `jit_call` inside an outer program, so the outer source program holds exactly
        // one `jit_call` instruction for the replay to differentiate.
        let inner: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let inner_for_outer = inner.clone();
        let outer: StagedFunction<ArrayType, ArrayType> =
            stage(move |x| inner_for_outer.call(x).unwrap(), input_type.clone(), Options::new(mesh.clone())).unwrap();
        let outer_program = outer.source_program().program();
        assert_eq!(
            outer_program.to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                let %1:f32[][sharding={mesh<['x'=1:auto]>, []}] = jit_call %0 [
                    callee={
                        lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                        let %1:f32[][sharding={mesh<['x'=1:auto]>, []}] = sin %0
                        in (%1)
                    },
                ]
                in (%1)"},
        );

        // Linearize the outer `jit_call` program, flattened first because `Program::linearize` is defined on the
        // canonical flat form. This drives the replay through the `jvp` rule of `JitCallOperation`, which rewraps the
        // split callee into a primal `jit_call` (outputs followed by residuals) and a tangent `jit_call` (input
        // tangents followed by residuals).
        let linearization = outer_program.to_flat_program().linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1, "the pushforward of `sin` saves `cos(x)` as its only residual");

        // Both halves must keep the callee behind a `jit_call` boundary rather than inlining `sin`/`cos`.
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                let %1:f32[][sharding={mesh<['x'=1:auto]>, []}], %2:f32[][sharding={mesh<['x'=1:auto]>, []}] = jit_call %0 [
                    callee={
                        lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                        let %1:f32[][sharding={mesh<['x'=1:auto]>, []}] = sin %0
                            %2:f32[][sharding={mesh<['x'=1:auto]>, []}] = cos %0
                        in (%1, %2)
                    },
                ]
                in (%1, %2)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}], %1:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                let %2:f32[][sharding={mesh<['x'=1:auto]>, []}] = jit_call %0 %1 [
                    callee={
                        lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}], %1:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                        let %2:f32[][sharding={mesh<['x'=1:auto]>, []}] = mul %1 %0
                        in (%2)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(
            linearization.primal().output_types().len(),
            1 + linearization.residual_count(),
            "primal half should produce the primal output followed by the residuals",
        );
        assert_eq!(
            linearization.tangent().input_types().len(),
            1 + linearization.residual_count(),
            "tangent half should consume the input tangent followed by the residuals",
        );

        // Rewrap the two halves as a compiled `(x, dx) -> (primal, tangent)` function, which exercises the real XLA
        // lowering and execution of both `jit_call` boundaries.
        let (primal_half, tangent_half, residual_count, _) = linearization.into_parts();
        let jvp_compiled: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> = compile(
            move |(primal_input, tangent_input)| {
                let context = primal_input.value().context().clone();
                let mut primal_outputs = context
                    .stage_operation(
                        Operation::JitCall(JitCallOperation::new(0)),
                        AttachedRegions::Callees(std::slice::from_ref(&primal_half)),
                        &[primal_input.into_value()],
                    )
                    .unwrap();
                let residuals = primal_outputs.split_off(1);
                assert_eq!(residuals.len(), residual_count, "primal half residual count should match linearization");
                let primal_output = primal_outputs.remove(0);
                let mut tangent_inputs = vec![tangent_input.into_value()];
                tangent_inputs.extend(residuals);
                let tangent_output = context
                    .stage_operation(
                        Operation::JitCall(JitCallOperation::new(0)),
                        AttachedRegions::Callees(std::slice::from_ref(&tangent_half)),
                        tangent_inputs.as_slice(),
                    )
                    .unwrap()
                    .remove(0);
                (
                    ValueProjection::<ArrayType>::into_projected(primal_output).unwrap(),
                    ValueProjection::<ArrayType>::into_projected(tangent_output).unwrap(),
                )
            },
            (input_type.clone(), input_type.clone()),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        for &(primal, tangent) in &[(0.0f32, 1.0f32), (0.25, 2.0), (0.5, -0.5), (1.0, 0.7)] {
            let primal_array = Array::from_host_buffer(
                &engine,
                input_type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>([primal].as_slice()).as_slice(),
            )
            .unwrap();
            let tangent_array = Array::from_host_buffer(
                &engine,
                input_type.clone(),
                mesh.clone(),
                None,
                values_to_bytes::<f32>([tangent].as_slice()).as_slice(),
            )
            .unwrap();
            let (jvp_primal, jvp_tangent) =
                jvp_compiled.executable().execute(&engine, (primal_array.clone(), tangent_array.clone())).unwrap();
            let jvp_primal_value = read_first_shard::<f32>(client, &jvp_primal)[0];
            let jvp_tangent_value = read_first_shard::<f32>(client, &jvp_tangent)[0];
            let expected_primal = primal.sin();
            let expected_tangent = primal.cos() * tangent;

            assert!(
                (jvp_primal_value - expected_primal).abs() < 1e-5,
                "primal at (primal={primal}, tangent={tangent}): expected ~{expected_primal}, \
                 got {jvp_primal_value}",
            );
            assert!(
                (jvp_tangent_value - expected_tangent).abs() < 1e-5,
                "tangent at (primal={primal}, tangent={tangent}): expected ~{expected_tangent}, \
                 got {jvp_tangent_value}",
            );
        }
    }

    #[test]
    fn test_jit_call_transposition_ignores_dead_reference_operands() {
        // The linear callee `ẏ = { ṫr += ẋ; ẋ }` only stores into its reference operand and forwards `x`. Under an
        // `Ignore` destination for the reference no later instruction accumulated into its root and the callee never
        // reads it, so the callee is transposed with an `Ignore` destination as well. The transposed call consumes
        // `[ȳ]` only, returns `x̄` only, and neither it nor its callee stages a cotangent reference.
        let scalar_type = Type::Array(ArrayType::scalar(DataType::F32));
        let reference_type = Type::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let reference = builder.add_input(reference_type.clone());
            let update = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(
                    Operation::ReferenceAddUpdate(ReferenceAddUpdateOperation::new()),
                    Vec::new(),
                    vec![reference, update],
                    None,
                )
                .unwrap();
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![update], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let callee_region = builder.import_region(callee.entry_region_ref());
        let reference = builder.add_input(reference_type.clone());
        let update = builder.add_input(scalar_type.clone());
        let output = builder
            .add_instruction(
                Operation::JitCall(JitCallOperation::new(0)),
                vec![callee_region],
                vec![reference, update],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let destination_kinds = [CotangentDestinationKind::Ignore, CotangentDestinationKind::Return];
        let transposed = program.transpose_with_respect_to(&[0, 1], &destination_kinds).unwrap();
        assert_eq!(transposed.input_types(), vec![scalar_type.clone()]);
        assert_eq!(transposed.output_types(), vec![scalar_type.clone()]);
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = jit_call %0 [
                    callee={
                        lambda %0:f32[] .
                        in (%0)
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_jit_call_transposition_threads_cotangent_references() {
        // The linear callee `ẏ = { ṫr += ẋ; read(ṫr) }` accumulates into its reference operand and reads it back.
        let scalar_type = Type::Array(ArrayType::scalar(DataType::F32));
        let reference_type = Type::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let reference = builder.add_input(reference_type.clone());
            let update = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(
                    Operation::ReferenceAddUpdate(ReferenceAddUpdateOperation::new()),
                    Vec::new(),
                    vec![reference, update],
                    None,
                )
                .unwrap();
            let output = builder
                .add_instruction(
                    Operation::ReferenceRead(ReferenceReadOperation::new()),
                    Vec::new(),
                    vec![reference],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let callee_region = builder.import_region(callee.entry_region_ref());
        let reference = builder.add_input(reference_type.clone());
        let update = builder.add_input(scalar_type.clone());
        let output = builder
            .add_instruction(
                Operation::JitCall(JitCallOperation::new(0)),
                vec![callee_region],
                vec![reference, update],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // Transposed under a `Reference` destination for the reference operand, the pullback consumes `[ȳ, r̄]` and
        // returns the cotangent reference by identity ahead of `x̄`. The reference operand's cotangent lives in that
        // destination, which the transposed call accumulates into in place, so the pullback forwards its own
        // destination input rather than the call's forwarded reference output, while `x̄` is the call's second output.
        // The transposed callee is exactly the transposition of the inlined callee under the same destinations.
        let destination_kinds = [CotangentDestinationKind::Reference, CotangentDestinationKind::Return];
        let transposed = program.transpose_with_respect_to(&[0, 1], &destination_kinds).unwrap();
        assert_eq!(transposed.input_types(), vec![scalar_type.clone(), reference_type.clone()]);
        assert_eq!(transposed.output_types(), vec![reference_type.clone(), scalar_type]);
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[], %1:ref<f32[]> .
                let %2:ref<f32[]>, %3:f32[] = jit_call %0 %1 [
                    callee={
                        lambda %0:f32[], %1:ref<f32[]> .
                        let () = reference_add_update %1 %0
                            %2:f32[] = reference_read %1
                        in (%1, %2)
                    },
                ]
                in (%1, %3)"},
        );
        let call = &transposed.instructions()[0];
        let transposed_callee = transposed.region_ref(call.regions()[0]).unwrap().to_program();
        let inlined = callee.transpose_with_respect_to(&[0, 1], &destination_kinds).unwrap();
        assert_eq!(transposed_callee.to_string(), inlined.to_string());
    }

    #[test]
    fn test_jit_call_transposition_uses_cotangent_descriptors_for_zero_cotangents() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let tangent_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        let expected = Type::Array(tangent_type.cotangent().unwrap());
        let tangent_type = Type::Array(tangent_type);
        // Zero cotangents skip a pure callee only after its effects have been inspected. Supply the real
        // identity-region boundary rather than an empty driver that cannot answer that query.
        let mut source_builder = ArrayIrProgramBuilder::new();
        let input = source_builder.add_input(tangent_type.clone());
        let source = source_builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut transpose_builder = ArrayIrProgramBuilder::new();
        let input = transpose_builder.add_input(expected.clone());
        let transposed = transpose_builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let driver = TestTranspositionDriver { source, transposed };
        let mut context = TracingContext::<Constant, Operation>::new();
        let cotangents = transpose_primal_jit_call(
            &JitCallOperation::new(0),
            &mut context,
            &driver,
            &[PartialValue::Unknown(tangent_type.clone())],
            &[MaybeZero::Zero(tangent_type.clone())],
            &CotangentDestinations::without_references([true]),
        )
        .unwrap();
        assert!(matches!(&cotangents[..], [MaybeZero::Zero(actual)] if actual == &expected));

        let known = context.input(tangent_type.clone());
        let cotangents = transpose_primal_jit_call(
            &JitCallOperation::new(0),
            &mut context,
            &driver,
            &[PartialValue::Known(known)],
            &[MaybeZero::Zero(tangent_type)],
            &CotangentDestinations::without_references([true]),
        )
        .unwrap();
        assert!(matches!(&cotangents[..], [MaybeZero::Zero(actual)] if actual == &expected));
    }

    #[test]
    fn test_jit_call_transposition_omits_dead_dynamic_seeds() {
        // The callee `(x, z) ↦ (x, z)` returns both of its dynamically shaped inputs, and only `z` is live. The dead
        // output's seed is a structural zero of type `f64[dead]`, whose extent no operand of the call defines. The
        // transposed call has no input for that seed, so it needs no extent for it. It consumes the live seed alone,
        // `x̄` stays a structural zero, and `z̄` keeps the `live` identity of the caller's cotangent type.
        let dead = DimensionVariable::new("dead", DimensionBounds::new(1, Some(5)).unwrap());
        let live = DimensionVariable::new("live", DimensionBounds::new(1, Some(5)).unwrap());
        let dead_type = Type::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(dead)])));
        let live_type = Type::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(live)])));
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let dead = builder.add_input(dead_type.clone());
            let live = builder.add_input(live_type.clone());
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![dead, live], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let dead = builder.add_input(dead_type);
        let live = builder.add_input(live_type.clone());
        let outputs = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee], vec![dead, live], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![outputs[1]], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let transposed = program.entry_region_ref().transpose(&[0, 1], &[], &[]).unwrap();
        assert_eq!(transposed.cotangents(), &[TransposedProgramCotangent::Zero, TransposedProgramCotangent::Output]);
        let pullback = transposed.into_program();
        assert_eq!(pullback.output_types(), vec![live_type]);
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[live] .
                let %1:f64[live] = jit_call %0 [
                    callee={
                        lambda %0:f64[live] .
                        in (%0)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(
            execute_mixed_program(
                execution_client(),
                &pullback,
                &[MixedValue::Array(vec![1.0, 2.0, 3.0, 0.0], vec![4])],
                &[3],
            )
            .unwrap(),
            vec![MixedValue::Array(vec![1.0, 2.0, 3.0], vec![3]), MixedValue::Dimension(3)],
        );
    }

    #[test]
    fn test_jit_call_transposition_omits_dead_seeds_of_repeated_outputs() {
        // The callee `x ↦ (x, x)` returns its input twice. With only the second output live, the first output's seed
        // is a structural zero, so the transposed call consumes the second seed alone, which is `x̄`. With both outputs
        // live, the transposed callee receives both seeds and adds them.
        let scalar_type = Type::Array(ArrayType::scalar(DataType::F64));
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(scalar_type.clone());
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![input, input], vec![Placeholder], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let input = builder.add_input(scalar_type);
        let outputs = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee], vec![input], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        let client = execution_client();

        // Projecting onto the second output keeps the two-output call, so its first seed reaches the call's transpose
        // rule as a structural zero rather than being pruned away beforehand.
        let projected = program.clone().with_outputs(&[1]).unwrap();
        assert_eq!(
            projected.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[], %2:f64[] = jit_call %0 [
                    callee={
                        lambda %0:f64[] .
                        in (%0, %0)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        let transposed = projected.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = jit_call %0 [
                    callee={
                        lambda %0:f64[] .
                        in (%0)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(
            execute_mixed_program(client, &transposed, &[MixedValue::Array(vec![2.0], vec![])], &[]).unwrap(),
            vec![MixedValue::Array(vec![2.0], vec![])],
        );

        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = jit_call %0 %1 [
                    callee={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = add %0 %1
                        in (%2)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            execute_mixed_program(
                client,
                &transposed,
                &[MixedValue::Array(vec![2.0], vec![]), MixedValue::Array(vec![5.0], vec![])],
                &[],
            )
            .unwrap(),
            vec![MixedValue::Array(vec![7.0], vec![])],
        );
    }

    #[test]
    fn test_jit_call_transposition_runs_backward_effects_without_live_seeds() {
        // The callee maps its input through a reverse mode carrier whose backward rule prints its seed and returns a
        // zero. With the call's output dead, its only seed is a structural zero, yet the backward effect must still
        // run. The transposed call has no inputs, and its callee prints a materialized zero seed.
        let scalar_type = Type::Array(ArrayType::scalar(DataType::F64));
        let backward = {
            let mut builder = ArrayIrProgramBuilder::new();
            let seed = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(PrintOperation::<ArrayType>::new("backward"), Vec::new(), vec![seed], None)
                .unwrap();
            let zero = builder
                .add_instruction(ZeroOperation::new(ArrayType::scalar(DataType::F64)), Vec::new(), Vec::new(), None)
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![zero], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let backward = builder.import_region(backward.entry_region_ref());
            let input = builder.add_input(scalar_type.clone());
            let output = builder
                .add_instruction(
                    CompositeTransposeOperation::<Constant, Operation>::from_backward_region(
                        0,
                        vec![scalar_type.clone()],
                        vec![scalar_type.clone()],
                    ),
                    vec![backward],
                    vec![input],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let input = builder.add_input(scalar_type);
        builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee], vec![input], None)
            .unwrap();
        let program = builder.build::<Vec<Constant>, Vec<Constant>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let transposed = program.entry_region_ref().transpose(&[0], &[], &[]).unwrap();
        assert_eq!(transposed.cotangents(), &[TransposedProgramCotangent::Zero]);
        let pullback = transposed.into_program();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda  .
                let () = jit_call [
                    callee={
                        lambda  .
                        let %0:f64[] = zero [type=f64[]]
                            %1:f64[] = print [label=backward] %0
                            %2:f64[] = zero [type=f64[]]
                        in ()
                    },
                ]
                in ()"},
        );
        let client = execution_client();
        ensure_print_handler_registered(client).unwrap();
        let (outputs, lines) = with_captured_prints(|| execute_mixed_program(client, &pullback, &[], &[]));
        assert_eq!(outputs.unwrap(), Vec::new(),);
        assert_eq!(lines, vec!["backward: 0.0"]);
    }

    #[test]
    fn test_jit_call_transposition_omits_dead_seeds_beside_cotangent_references() {
        // The callee `ẏ = { ṫr += ẋ; (read(ṫr), ẋ) }` accumulates into its reference operand, reads it back, and
        // forwards `x`. Only the read is live, so the forwarded output's seed is a structural zero. Under a `Reference`
        // destination the transposed call consumes `[ȳ₀, r̄]` with no input for the dead seed, accumulates into `r̄` in
        // place, and returns `x̄` read back from it.
        let scalar_type = Type::Array(ArrayType::scalar(DataType::F64));
        let reference_type = Type::Reference(ReferenceType::new(ArrayType::scalar(DataType::F64)));
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let reference = builder.add_input(reference_type.clone());
            let update = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(
                    Operation::ReferenceAddUpdate(ReferenceAddUpdateOperation::new()),
                    Vec::new(),
                    vec![reference, update],
                    None,
                )
                .unwrap();
            let read = builder
                .add_instruction(
                    Operation::ReferenceRead(ReferenceReadOperation::new()),
                    Vec::new(),
                    vec![reference],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![read, update], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let reference = builder.add_input(reference_type.clone());
        let update = builder.add_input(scalar_type.clone());
        let outputs = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee], vec![reference, update], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![outputs[0]], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let destination_kinds = [CotangentDestinationKind::Reference, CotangentDestinationKind::Return];
        let transposed = program.transpose_with_respect_to(&[0, 1], &destination_kinds).unwrap();
        assert_eq!(transposed.input_types(), vec![scalar_type.clone(), reference_type.clone()]);
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[], %1:ref<f64[]> .
                let %2:ref<f64[]>, %3:f64[] = jit_call %0 %1 [
                    callee={
                        lambda %0:f64[], %1:ref<f64[]> .
                        let () = reference_add_update %1 %0
                            %2:f64[] = reference_read %1
                        in (%1, %2)
                    },
                ]
                in (%1, %3)
            "}
            .trim_end(),
        );

        // With both outputs dead, every seed is a structural zero, but the live reference destination keeps the call
        // live, so the transposed call consumes `r̄` alone and returns `x̄` read back from it.
        let transposed =
            program.with_outputs(&[]).unwrap().transpose_with_respect_to(&[0, 1], &destination_kinds).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:ref<f64[]> .
                let %1:ref<f64[]>, %2:f64[] = jit_call %0 [
                    callee={
                        lambda %0:ref<f64[]> .
                        let %1:f64[] = reference_read %0
                        in (%0, %1)
                    },
                ]
                in (%0, %2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_jit_call_transposition_keeps_dead_seeds_structural_in_nested_scans() {
        // The callee runs the scan `(c, x) ↦ (c + x, c)` over three items and returns its final carry and its stacked
        // outputs, of which only the final carry is live. The dead stack's seed reaches the scan inside the callee as a
        // structural zero, so neither the transposed call nor its reversed scan receives a zero stack.
        let scalar_type = Type::Array(ArrayType::scalar(DataType::F64));
        let stack_type = Type::Array(ArrayType::new_static(DataType::F64, [3]));
        let body = {
            let mut builder = ArrayIrProgramBuilder::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(scalar_type.clone());
            let item = builder.add_input(scalar_type.clone());
            let next = builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, item], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![next, carry], vec![Placeholder; 3], vec![Placeholder; 2])
                .unwrap()
        };
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let body = builder.import_region(body.entry_region_ref());
            let carry = builder.add_input(scalar_type.clone());
            let items = builder.add_input(stack_type.clone());
            let outputs = builder
                .add_instruction(Operation::Scan(ScanOperation::new(1, 3)), vec![body], vec![carry, items], None)
                .unwrap()
                .to_vec();
            builder
                .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let carry = builder.add_input(scalar_type);
        let items = builder.add_input(stack_type);
        let outputs = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee], vec![carry, items], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![outputs[0]], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[], %2:f64[3] = jit_call %0 [
                    callee={
                        lambda %0:f64[] .
                        let %1:f64[], %2:f64[3] = scan [carry_count=1, iteration_count=3, reverse=true] %0 [
                            body={
                                lambda %0:i64[], %1:f64[] .
                                in (%1, %1)
                            },
                        ]
                        in (%1, %2)
                    },
                ]
                in (%1, %2)
            "}
            .trim_end(),
        );
        assert_eq!(
            execute_mixed_program(execution_client(), &transposed, &[MixedValue::Array(vec![2.0], vec![])], &[])
                .unwrap(),
            vec![MixedValue::Array(vec![2.0], vec![]), MixedValue::Array(vec![2.0, 2.0, 2.0], vec![3])],
        );
    }

    #[test]
    fn test_jit_call_transposition_keeps_dead_seeds_structural_in_nested_conditions() {
        // The callee selects on `c > 0` between the branches `x ↦ (x, x + x)` and `x ↦ (x + x, x)` and only the first
        // output is live. The dead output's seed reaches the condition inside the callee as a structural zero, so each
        // transposed branch consumes the live seed alone, giving `x̄ = ȳ₀` when `c > 0` and `x̄ = 2ȳ₀` otherwise.
        let scalar_type = Type::Array(ArrayType::scalar(DataType::F64));
        let branch = |doubled_output: usize| {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(scalar_type.clone());
            let doubled =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            let outputs = if doubled_output == 0 { vec![doubled, input] } else { vec![input, doubled] };
            builder
                .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder; 2])
                .unwrap()
        };
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let true_branch = builder.import_region(branch(1).entry_region_ref());
            let false_branch = builder.import_region(branch(0).entry_region_ref());
            let selector = builder.add_input(scalar_type.clone());
            let input = builder.add_input(scalar_type.clone());
            let zero = builder
                .add_instruction(ZeroLikeOperation::<ArrayType>::new(), Vec::new(), vec![selector], None)
                .unwrap()[0];
            let predicate = builder
                .add_instruction(
                    Operation::Array(ArrayOperation::Compare(CompareOperation::<ArrayType>::new(
                        ComparisonDirection::GreaterThan,
                    ))),
                    Vec::new(),
                    vec![selector, zero],
                    None,
                )
                .unwrap()[0];
            let outputs = builder
                .add_instruction(
                    Operation::Condition(ConditionOperation::new()),
                    vec![true_branch, false_branch],
                    vec![predicate, input],
                    None,
                )
                .unwrap()
                .to_vec();
            builder
                .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let selector = builder.add_input(scalar_type.clone());
        let input = builder.add_input(scalar_type);
        let outputs = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee], vec![selector, input], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![outputs[0]], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[1], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = jit_call %0 %1 [
                    callee={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = zero_like %1
                            %3:bool[] = compare [direction=GreaterThan] %1 %2
                            %4:f64[] = condition %3 %0 [
                                true={
                                    lambda %0:f64[] .
                                    in (%0)
                                },
                                false={
                                    lambda %0:f64[] .
                                    let %1:f64[] = add %0 %0
                                    in (%1)
                                },
                            ]
                        in (%4)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        let client = execution_client();
        for (selector, expected) in [(1.0, 3.0), (-1.0, 6.0)] {
            assert_eq!(
                execute_mixed_program(
                    client,
                    &transposed,
                    &[MixedValue::Array(vec![3.0], vec![]), MixedValue::Array(vec![selector], vec![])],
                    &[],
                )
                .unwrap(),
                vec![MixedValue::Array(vec![expected], vec![])],
            );
        }
    }

    #[test]
    fn test_jit_call_transposition_keeps_dead_seeds_structural_across_rematerializations() {
        // A rematerialization inside a `jit_call` callee and a `jit_call` operation inside a rematerialization both map
        // `x` to `(x, x + x)`, and only `x + x` is live. In both nestings the dead output's seed stays a structural
        // zero across both boundaries. The outer transposed region consumes the live seed alone, the inner one receives
        // it alone, and `x̄ = ȳ₁ + ȳ₁`.
        let scalar_type = Type::Array(ArrayType::scalar(DataType::F64));
        let doubling = {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(scalar_type.clone());
            let doubled =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![input, doubled], vec![Placeholder], vec![Placeholder; 2])
                .unwrap()
        };
        let rematerialize = || RematerializeOperation::new(ResidualPolicyReference::new(NothingSavable));
        let nest = |outer: Operation, inner: Operation| {
            let inner_program = {
                let mut builder = ArrayIrProgramBuilder::new();
                let region = builder.import_region(doubling.entry_region_ref());
                let input = builder.add_input(scalar_type.clone());
                let outputs = builder.add_instruction(inner, vec![region], vec![input], None).unwrap().to_vec();
                builder
                    .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder; 2])
                    .unwrap()
            };
            let mut builder = ArrayIrProgramBuilder::new();
            let region = builder.import_region(inner_program.entry_region_ref());
            let input = builder.add_input(scalar_type.clone());
            let outputs = builder.add_instruction(outer, vec![region], vec![input], None).unwrap().to_vec();
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![outputs[1]], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let client = execution_client();

        let program = nest(Operation::JitCall(JitCallOperation::new(0)), rematerialize().into());
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = jit_call %0 [
                    callee={
                        lambda %0:f64[] .
                        let %1:f64[] = rematerialize %0 [
                            body={
                                lambda %0:f64[] .
                                let %1:f64[] = add %0 %0
                                in (%1)
                            },
                        ]
                        in (%1)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(
            execute_mixed_program(client, &transposed, &[MixedValue::Array(vec![2.0], vec![])], &[]).unwrap(),
            vec![MixedValue::Array(vec![4.0], vec![])],
        );

        let program = nest(rematerialize().into(), Operation::JitCall(JitCallOperation::new(0)));
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = rematerialize %0 [
                    body={
                        lambda %0:f64[] .
                        let %1:f64[] = jit_call %0 [
                            callee={
                                lambda %0:f64[] .
                                let %1:f64[] = add %0 %0
                                in (%1)
                            },
                        ]
                        in (%1)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(
            execute_mixed_program(client, &transposed, &[MixedValue::Array(vec![2.0], vec![])], &[]).unwrap(),
            vec![MixedValue::Array(vec![4.0], vec![])],
        );
    }

    #[test]
    fn test_jit_call_transposition_omits_disconnected_symbolic_cotangent() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let scalar_type = ArrayType::scalar(DataType::F32);
        let mut callee_builder = ArrayIrProgramBuilder::new();
        callee_builder.add_input(dynamic_type.clone().into());
        let live = callee_builder.add_input(scalar_type.clone().into());
        let callee = callee_builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![live], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ArrayIrProgramBuilder::new();
        let disconnected = builder.add_input(dynamic_type.into());
        let live = builder.add_input(scalar_type.into());
        let callee = builder.import_region(callee.entry_region_ref());
        let outputs = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee], vec![disconnected, live], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 0);
        let pullback = linearization.pullback().unwrap();
        assert_eq!(pullback.cotangents(), &[TransposedProgramCotangent::Zero, TransposedProgramCotangent::Output]);
        assert_eq!(
            pullback.program().to_string(),
            indoc! {"
            lambda %0:f32[] .
            let %1:f32[] = jit_call %0 [
                callee={
                    lambda %0:f32[] .
                    in (%0)
                },
            ]
            in (%1)
        "}
            .trim_end()
        );
    }

    #[test]
    fn test_jit_call_transposition_omits_zero_space_seeds_of_mixed_outputs() {
        // The predicate output's cotangent is a zero space that arrives as a structural zero, so the transposed callee
        // has no input for it and the transposed call consumes the value output's seed alone.
        let value_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let predicate_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(4)]));
        let value_program_type = Type::Array(value_type.clone());
        let predicate_program_type = Type::Array(predicate_type.clone());
        let source = {
            let mut builder = ArrayIrProgramBuilder::new();
            let value = builder.add_input(value_program_type.clone());
            let predicate =
                builder.add_constant(Constant::Captured(CaptureReference::new(0, predicate_program_type.clone())));
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![value, predicate], vec![Placeholder], vec![Placeholder; 2])
                .unwrap()
        };
        let transposed = {
            let mut builder = ArrayIrProgramBuilder::new();
            let value_cotangent = builder.add_input(value_program_type.clone());
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![value_cotangent], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let driver = TestTranspositionDriver { source, transposed };
        let mut context = TracingContext::<Constant, Operation>::new();
        let value_cotangent = context.input(value_program_type.clone());

        let contributions = transpose_primal_jit_call(
            &JitCallOperation::new(0),
            &mut context,
            &driver,
            &[PartialValue::Unknown(value_program_type.clone())],
            &[
                MaybeZero::Value(value_cotangent.clone()),
                MaybeZero::Zero(Type::Array(predicate_type.cotangent().unwrap())),
            ],
            &CotangentDestinations::without_references([true]),
        )
        .unwrap();

        assert!(
            matches!(&contributions[..], [MaybeZero::Value(value)] if value.r#type().as_ref() == &value_program_type)
        );
        let builder = context.builder().borrow();
        let [call] = builder.instructions() else {
            panic!("expected the transposed call to be the only staged instruction");
        };
        assert!(matches!(call.operation(), Operation::JitCall(_)));
        assert_eq!(call.inputs(), &[value_cotangent.atom_id().unwrap()]);
    }

    #[test]
    fn test_jit_call_operation_transpose_keeps_the_input_overrides_of_returned_cotangents() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let split = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let vector_type = ArrayType::new_static(DataType::F32, [4]);
        let types = (vector_type.clone(), vector_type.clone());
        let options = Options::new(mesh.clone()).with_input_shardings(vec![split.clone(), replicated.clone()]);
        let inner: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> =
            compile_with_options(|(x, y)| (x, y), types.clone(), &session, options).unwrap();
        let input = Array::from_host_buffer(
            &session,
            vector_type,
            mesh.clone(),
            Some(replicated.clone()),
            values_to_bytes(&[1f32, 2.0, 3.0, 4.0]),
        )
        .unwrap();

        // The cotangent of the first input of the call keeps the split of its input override, as JAX keeps the input
        // sharding of a `jit` function on the cotangent of that input, whether the second input has a cotangent or a
        // structurally zero one, which the transposed callee does not return.
        for sums_both_outputs in [false, true] {
            let outer: CompiledFunction<(ArrayType, ArrayType), ArrayType> = compile(
                |(x, y): (CompileTracer, CompileTracer)| {
                    let (x, y) = inner.call((x, y)).unwrap();
                    let output = if sums_both_outputs { x + y } else { x };
                    output.reduce_sum(&[0], None).unwrap()
                },
                types.clone(),
                &session,
                mesh.clone(),
            )
            .unwrap();
            let gradient = outer.gradient(&session).unwrap();
            let (first, _) = gradient.executable().execute(&session, (input.clone(), input.clone())).unwrap();
            assert_eq!(first.sharding(), &split);
            assert_eq!(first.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1.0; 4]);
        }
    }

    #[test]
    fn test_jit_call_operation_interpret() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let session = Session::builder().client(client).build().unwrap();
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let shapes = |array: &Array| array.shards().iter().map(|shard| shard.shape().to_string()).collect::<Vec<_>>();
        let values = |array: &Array| {
            array
                .shards()
                .iter()
                .map(|shard| values_from_bytes::<i32>(&shard.copy_to_host_bytes().unwrap()))
                .collect::<Vec<_>>()
        };
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::I32, [4]),
            mesh.clone(),
            Some(sharded.clone()),
            values_to_bytes(&[1i32, 2, 3, 4]),
        )
        .unwrap();
        // The callee declares no sharding for its input, so its executable places that input replicated over the mesh
        // of the call (refer to `JitCallOperation::boundary_input_sharding`).
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(Type::Array(ArrayType::new_static(DataType::I32, [4])));
            Arc::new(
                builder
                    .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
                    .unwrap(),
            )
        };
        let regions = AttachedRegions::Callees(std::slice::from_ref(&callee));
        let driver = EagerInterpretationDriver::new(&regions);
        let interpret = |call: JitCallOperation<Type>| {
            let outputs = call.interpret(&session, &driver, &[Value::Array(input.clone())]).unwrap();
            let [Value::Array(output)] = outputs.as_slice() else {
                panic!("expected one array output");
            };
            output.block_until_ready().unwrap();
            output.clone()
        };

        // Without boundary shardings, the identity callee returns its input replicated, as an execution of a function
        // compiled for that input type places its argument, while a call of a `jit` dispatcher returns its input where
        // it is, as the executables of the dispatcher do.
        let output = interpret(JitCallOperation::new(0));
        assert_eq!(output.sharding(), &replicated);
        assert_eq!(shapes(&output), vec!["[4]", "[4]"]);
        assert_eq!(values(&output), vec![vec![1, 2, 3, 4]; 2]);
        let output = interpret(JitCallOperation::new(0).with_kept_input_placements());
        assert_eq!(output.sharding(), &sharded);
        assert_eq!(shapes(&output), vec!["[2]", "[2]"]);
        assert_eq!(values(&output), vec![vec![1, 2], vec![3, 4]]);

        // A replicated output override places the result of the callee, and a replicated input override places the
        // input that the callee returns, as executions of the called function place their results and arguments.
        for call in [
            JitCallOperation::new(0).with_boundary(Some(&mesh), Vec::new(), vec![Some(replicated.clone())]),
            JitCallOperation::new(0).with_boundary(Some(&mesh), vec![Some(replicated.clone())], Vec::new()),
        ] {
            let output = interpret(call);
            assert_eq!(output.sharding(), &replicated);
            assert!(output.is_committed());
            assert_eq!(shapes(&output), vec!["[4]", "[4]"]);
            assert_eq!(values(&output), vec![vec![1, 2, 3, 4]; 2]);
        }

        // A call whose boundary is on another single device moves an uncommitted input there, as an execution of its
        // callee moves its argument, and commits the result, as that execution does, although the input already has
        // the sharding that the override requests.
        let second_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![mesh.devices()[1]],
        )
        .unwrap();
        let second_replicated = Sharding::replicated(second_mesh.logical_mesh().clone(), 1);
        let vector_type = ArrayType::new_static(DataType::I32, [4]);
        let uncommitted =
            Array::from_host_buffer(&session, vector_type.clone(), None, None, values_to_bytes(&[1i32, 2, 3, 4]))
                .unwrap();
        assert!(!uncommitted.is_committed());
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(Type::Array(vector_type));
            Arc::new(
                builder
                    .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
                    .unwrap(),
            )
        };
        let regions = AttachedRegions::Callees(std::slice::from_ref(&callee));
        let call = JitCallOperation::new(0).with_boundary(
            Some(&second_mesh),
            Vec::new(),
            vec![Some(second_replicated.clone())],
        );
        let outputs = call
            .interpret(&session, &EagerInterpretationDriver::new(&regions), &[Value::Array(uncommitted)])
            .unwrap();
        let [Value::Array(output)] = outputs.as_slice() else {
            panic!("expected one array output");
        };
        output.block_until_ready().unwrap();
        assert_eq!(output.mesh(), &second_mesh);
        assert_eq!(output.sharding(), &second_replicated);
        assert!(output.is_committed());
        assert_eq!(shapes(output), vec!["[4]"]);
        assert_eq!(values(output), vec![vec![1, 2, 3, 4]]);
    }

    #[test]
    fn test_jit_call_operation_eager_input_placement() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let session = Session::builder().client(client).build().unwrap();
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let vector_type = ArrayType::new_static(DataType::I32, [4]);
        let array = Array::from_host_buffer(
            &session,
            vector_type.clone(),
            mesh.clone(),
            Some(sharded.clone()),
            values_to_bytes(&[1i32, 2, 3, 4]),
        )
        .unwrap();
        let declared = Type::Array(vector_type);
        let derived = |shardings: Vec<Option<Sharding>>| JitCallOperation::<Type> {
            derived_input_shardings: Some(Arc::from(shardings)),
            ..JitCallOperation::new(0)
        };

        // A call of a function compiled for an input type without a sharding places the input replicated, and an
        // override places it as it says. A call of a `jit` dispatcher and a call that a transform created without a
        // boundary for the input keep the input where it is, and a derived call with a boundary places it there.
        let placement = |call: &JitCallOperation<Type>, index| call.eager_input_placement(index, &array, &declared);
        assert_eq!(placement(&JitCallOperation::new(0), 0), Some(replicated.clone()));
        assert_eq!(
            placement(&JitCallOperation::new(0).with_boundary(Some(&mesh), vec![Some(sharded.clone())], Vec::new()), 0),
            Some(sharded.clone()),
        );
        assert_eq!(placement(&JitCallOperation::new(0).with_kept_input_placements(), 0), Some(sharded.clone()));
        assert_eq!(placement(&derived(vec![None]), 0), Some(sharded.clone()));
        assert_eq!(placement(&derived(vec![Some(replicated.clone())]), 0), Some(replicated));

        // A lifted capture of a call of a compiled function is placed by the executable's own rule.
        assert_eq!(placement(&JitCallOperation::new(1), 0), None);
    }

    #[test]
    fn test_jit_call_operation_interpret_runs_the_callee_on_the_devices_of_the_call() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let second_device = Device::from_pjrt(&client.addressable_devices().unwrap()[1]).unwrap();
        let second_mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![second_device]).unwrap();
        let vector_type = ArrayType::new_static(DataType::F32, [4]);
        let upload = |mesh: Option<&DeviceMesh>| {
            Array::from_host_buffer(&session, vector_type.clone(), mesh.cloned(), None, values_to_bytes(&[0f32; 4]))
                .unwrap()
        };
        let interpret = |function: &CompiledFunction<_, _>, inputs: &[Value]| {
            let caller: CompiledFunction<ArrayType, (ArrayType, ArrayType)> =
                compile(|x| function.call(x).unwrap(), vector_type.clone(), &session, second_mesh.clone()).unwrap();
            let program = caller.source_program().program();
            let region = program.entry_region_ref();
            let call = region
                .instructions()
                .iter()
                .find(|instruction| matches!(instruction.operation(), Operation::JitCall(_)))
                .unwrap();
            let Operation::JitCall(operation) = call.operation() else { unreachable!("selected a `jit_call`") };
            let callee = Arc::new(region.with_id(call.regions()[0]).unwrap().to_program());
            let regions = AttachedRegions::Callees(std::slice::from_ref(&callee));
            operation.interpret(&session, &EagerInterpretationDriver::new(&regions), inputs)
        };

        // A callee compiled for the second device that places its input there and fills a value that depends on no
        // input returns both on that device, committed, when the call is replayed operation by operation, as an
        // execution of the callee returns them, rather than filling the value on the default device of the session.
        let function: CompiledFunction<ArrayType, (ArrayType, ArrayType)> = compile(
            |x: CompileTracer| (x.place(&second_device, None).unwrap(), x.domain().fill(&vector_type, 17f32).unwrap()),
            vector_type.clone(),
            &session,
            second_mesh.clone(),
        )
        .unwrap();
        let input = upload(None);
        let executed = function.executable().execute(&session, input.clone()).unwrap();
        let outputs = interpret(&function, &[Value::Array(input)]).unwrap();
        let placements = outputs
            .iter()
            .map(|output| match output {
                Value::Array(array) => (array.mesh().devices().to_vec(), array.is_committed()),
                _ => panic!("expected an array"),
            })
            .collect::<Vec<_>>();
        assert_eq!(placements, vec![(vec![second_device], true); 2]);
        assert_eq!(
            [executed.0.mesh().devices(), executed.1.mesh().devices()],
            [std::slice::from_ref(&second_device); 2],
        );

        // A callee that only moves its input to another memory names no devices, so its call runs where its public
        // argument is and commits both results, because the callee requests a placement. Replay fills the value there
        // too, rather than on the default device of the session, and commits both results whatever the commitment of
        // the argument, as an execution of the callee on those devices does.
        let first_device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let first_mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![first_device]).unwrap();
        let memory: CompiledFunction<ArrayType, (ArrayType, ArrayType)> = compile(
            |x: CompileTracer| (x.place(&Memory::Device, None).unwrap(), x.domain().fill(&vector_type, 17f32).unwrap()),
            vector_type.clone(),
            &session,
            first_mesh.clone(),
        )
        .unwrap();
        let executed = memory.executable().execute(&session, upload(None)).unwrap();
        assert_eq!(
            [
                (executed.0.mesh().devices(), executed.0.is_committed()),
                (executed.1.mesh().devices(), executed.1.is_committed())
            ],
            [(std::slice::from_ref(&first_device), true); 2],
        );
        for (input, device) in [(upload(Some(&second_mesh)), second_device), (upload(None), first_device)] {
            let outputs = interpret(&memory, &[Value::Array(input)]).unwrap();
            let placements = outputs
                .iter()
                .map(|output| match output {
                    Value::Array(array) => (array.mesh().devices().to_vec(), array.is_committed()),
                    _ => panic!("expected an array"),
                })
                .collect::<Vec<_>>();
            assert_eq!(placements, vec![(vec![device], true); 2]);
        }

        // A call of a callee that names no devices rejects committed arguments on different devices from one another
        // before it runs its callee, as JAX rejects such a `jit` call ("Received incompatible devices for jitted
        // computation").
        let pair: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> =
            compile(|(x, y)| (x, y), (vector_type.clone(), vector_type.clone()), &session, second_mesh.clone())
                .unwrap();
        let first_mesh = DeviceMesh::new(
            LogicalMesh::single_device(),
            vec![Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap()],
        )
        .unwrap();
        let (first, second) = (upload(Some(&first_mesh)), upload(Some(&second_mesh)));
        let caller: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> = compile(
            |(x, y)| pair.call((x, y)).unwrap(),
            (vector_type.clone(), vector_type.clone()),
            &session,
            second_mesh.clone(),
        )
        .unwrap();
        let program = caller.source_program().program();
        let region = program.entry_region_ref();
        let call = region.instructions().iter().find(|i| matches!(i.operation(), Operation::JitCall(_))).unwrap();
        let Operation::JitCall(operation) = call.operation() else { unreachable!("selected a `jit_call`") };
        let callee = Arc::new(region.with_id(call.regions()[0]).unwrap().to_program());
        let regions = AttachedRegions::Callees(std::slice::from_ref(&callee));
        assert_eq!(
            operation
                .interpret(
                    &session,
                    &EagerInterpretationDriver::new(&regions),
                    &[Value::Array(first), Value::Array(second)]
                )
                .unwrap_err(),
            ProgramError::InvalidArgument {
                message: Error::from(ProgramError::InvalidArgument {
                    message: format!(
                        "public argument #1 is committed to devices [{}], but public argument #0 is committed to \
                         devices [{}]",
                        second_device.id(),
                        first_mesh.devices()[0].id(),
                    ),
                })
                .to_string(),
            },
        );
    }

    #[test]
    fn test_jit_call_operation_lower_constrains_boundary_shardings() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let unconstrained = Sharding::new(mesh.clone(), vec![ShardingDimension::Unconstrained]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let device_mesh = DeviceMesh::new(mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)]).unwrap();
        let array_type = test_vector_type(4).with_sharding(replicated.clone()).unwrap();
        let callee = {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(array_type.clone().into());
            let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
            Arc::new(
                builder
                    .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                    .unwrap(),
            )
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(array_type.clone().into());
        let callee_region = builder.intern_callee(&callee, None).unwrap();
        let call = JitCallOperation::new(0).with_boundary(
            Some(&device_mesh),
            vec![Some(sharded.clone())],
            vec![Some(replicated.clone())],
        );
        let called =
            builder.add_instruction(Operation::JitCall(call), vec![callee_region], vec![input], None).unwrap()[0];
        let call = JitCallOperation::new(0).with_boundary(Some(&device_mesh), Vec::new(), vec![Some(unconstrained)]);
        let output =
            builder.add_instruction(Operation::JitCall(call), vec![callee_region], vec![called], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // Each operand and result of a call that its callee boundary overrides is constrained to the override where
        // the call executes, closed over auto mesh axes and with unconstrained dimensions replicated, as an execution
        // of the called function annotates its own boundary. An operand without an input override is constrained to
        // the replicated sharding that the callee declares for it, because its type cannot show its placement along
        // the auto axis (refer to `JitCallOperation::boundary_input_sharding`). The shared callee itself carries no
        // override.
        let input_types = vec![array_type.clone()];
        let output_types = vec![array_type];
        let shardings = [replicated];
        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &input_types,
            &output_types,
            "main",
            Some(&shardings),
            Some(&shardings),
        )
        .unwrap();
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func private @jit_call_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.sine %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{"x"}]> : tensor<4xf32>
                    %1 = call @jit_call_0(%0) : (tensor<4xf32>) -> tensor<4xf32>
                    %2 = sdy.sharding_constraint %1 <@mesh, [{}]> : tensor<4xf32>
                    %3 = sdy.sharding_constraint %2 <@mesh, [{}]> : tensor<4xf32>
                    %4 = call @jit_call_0(%3) : (tensor<4xf32>) -> tensor<4xf32>
                    %5 = sdy.sharding_constraint %4 <@mesh, [{}]> : tensor<4xf32>
                    return %5 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_jit_call_operation_lower_ignores_nested_call_capture_namespaces() {
        // The inner callee establishes its own two-slot capture namespace. Its first two inputs are its lifted
        // captures, and its body resolves the constant `Captured(1)` against that prefix. Deriving the OUTER call's
        // capture prefix from capture-constant indices anywhere in the callee arena would conflate the namespaces
        // and reject this program with a false missing-capture error, because the operation payload is authoritative.
        let array_type = test_vector_type(4);
        let inner = {
            let mut builder = ArrayIrProgramBuilder::new();
            let first_capture = builder.add_input(array_type.clone().into());
            let _second_capture = builder.add_input(array_type.clone().into());
            let second = builder.add_constant(Constant::Captured(CaptureReference::new(1, array_type.clone().into())));
            let output =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![first_capture, second], None).unwrap()[0];
            Arc::new(
                builder
                    .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                    .unwrap(),
            )
        };
        let outer = {
            let mut builder = ArrayIrProgramBuilder::new();
            let input = builder.add_input(array_type.clone().into());
            let callee_region = builder.intern_callee(&inner, None).unwrap();
            let output = builder
                .add_instruction(
                    Operation::JitCall(JitCallOperation::new(2)),
                    vec![callee_region],
                    vec![input, input],
                    None,
                )
                .unwrap()[0];
            Arc::new(
                builder
                    .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                    .unwrap(),
            )
        };
        let mut builder = ArrayIrProgramBuilder::new();
        let input = builder.add_input(array_type.clone().into());
        let callee_region = builder.intern_callee(&outer, None).unwrap();
        let output = builder
            .add_instruction(Operation::JitCall(JitCallOperation::new(0)), vec![callee_region], vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let input_types = vec![array_type.clone()];
        let output_types = vec![array_type];
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();
        assert_eq!(
            stablehlo,
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
    fn test_call_type_bindings() {
        let bounds = DimensionBounds::new(0, Some(4)).unwrap();
        let input = Type::from(DimensionType::new("input", bounds));
        let local = Type::from(DimensionType::new("local", bounds));
        let inputs = vec![input.clone()];
        let outputs = vec![input.clone(), local.clone(), local.clone()];
        assert_eq!(CallTypeBindings::fresh(&inputs, &inputs).unwrap(), None);

        // One local family is exported once even when several results share it. Separate calls allocate separate
        // families, while forwarded input names stay shared.
        let first = CallTypeBindings::fresh(&inputs, &outputs).unwrap().unwrap();
        let second = CallTypeBindings::fresh(&inputs, &outputs).unwrap().unwrap();
        assert_eq!(first.outputs()[0], input);
        assert_eq!(first.outputs()[1], first.outputs()[2]);
        assert_ne!(first.outputs()[1], local);
        assert_ne!(first.outputs()[1], second.outputs()[1]);
        assert_eq!(first.definitions().len(), 1);
        assert_eq!(first.instantiated(&inputs).unwrap(), first.outputs());
        let renaming = first.renaming(&inputs, &outputs).unwrap();
        assert_eq!(local.rename_identities(&renaming).unwrap(), first.outputs()[1]);
        assert_eq!(input.rename_identities(&renaming).unwrap(), input);

        // Instantiating a forwarded input in another namespace must not rename the call's local result.
        let actual = Type::from(DimensionType::new("actual", bounds));
        let instantiated = first.instantiated(std::slice::from_ref(&actual)).unwrap();
        assert_eq!(instantiated[0], actual);
        assert_eq!(&instantiated[1..], &first.outputs()[1..]);
        let retained = first.pruned(&[true], &[true, false, true]).unwrap();
        assert_eq!(retained.outputs(), &[input.clone(), first.outputs()[2].clone()]);
        assert_eq!(retained.definitions(), first.definitions());
        assert_eq!(
            first.pruned(&[], &[true; 3]),
            Err(TypeError::invalid("call result declaration pruning masks have incompatible arity")),
        );

        // A declaration cannot export the callee's private name verbatim or rename a forwarded input.
        assert_eq!(
            CallTypeBindings::with_outputs(&inputs, &outputs, outputs.clone()),
            Err(TypeError::invalid("call local result declarations must use fresh identities")),
        );
        let other = Type::from(DimensionType::new("other", bounds));
        assert_eq!(
            CallTypeBindings::with_outputs(&inputs, &inputs, vec![other]),
            Err(TypeError::invalid("call result declarations must preserve forwarded input identities")),
        );
        assert_eq!(
            CallTypeBindings::with_outputs(&inputs, &outputs, first.outputs().to_vec()),
            Ok(Some(first.clone()))
        );
    }

    #[test]
    fn test_call_type_bindings_instantiated_dense_static_input() {
        let bounds = DimensionBounds::non_negative(Some(8)).unwrap();
        let input = DimensionType::new("input", bounds);
        let local = DimensionType::new("local", bounds);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![input.dimension().clone()]));
        let output_type =
            ArrayType::new(DataType::F32, Shape::new(vec![input.dimension().clone(), local.dimension().clone()]));
        let outputs = [output_type.clone(), output_type];
        let first = CallTypeBindings::fresh(std::slice::from_ref(&input_type), &outputs).unwrap().unwrap();
        let second = CallTypeBindings::fresh(std::slice::from_ref(&input_type), &outputs).unwrap().unwrap();
        assert_ne!(first.outputs(), second.outputs());
        assert_eq!(first.outputs()[0], first.outputs()[1]);
        let actual_input = ArrayType::new_static(DataType::F32, [2]);
        let actual = first.instantiated(std::slice::from_ref(&actual_input)).unwrap();
        assert_eq!(actual[0].shape().dimensions()[0], 2usize.into());
        assert_eq!(actual[0].shape().dimensions()[1], first.outputs()[0].shape().dimensions()[1]);
        assert_eq!(actual[0], actual[1]);
        let callee_outputs = outputs
            .iter()
            .map(|output| {
                ArrayType::new(DataType::F32, Shape::new(vec![2usize.into(), output.shape().dimensions()[1].clone()]))
            })
            .collect::<Vec<_>>();
        let renaming = first.renaming(std::slice::from_ref(&actual_input), &callee_outputs).unwrap();
        assert_eq!(callee_outputs[0].rename_identities(&renaming).unwrap(), actual[0]);
        assert_eq!(first.definitions().len(), 1);
        let retained =
            CallTypeBindings::with_outputs(std::slice::from_ref(&input_type), &outputs, first.outputs().to_vec())
                .unwrap()
                .unwrap();
        assert_eq!(retained.instantiated(&[actual_input]).unwrap(), actual);
        let mut renamed = TypeIdentityRenaming::new();
        let local = first.definitions()[0].fresh();
        renamed.insert(first.definitions()[0].clone(), local.clone()).unwrap();
        let rebound = first.renamed(&renamed).unwrap();
        assert_eq!(rebound.definitions(), vec![local]);
        assert_ne!(rebound.outputs(), first.outputs());
    }

    #[test]
    fn test_replicates_unconstrained_dimensions() {
        // Shardings place an array alike when each dimension is equal or unconstrained in one and replicated in the
        // other, over the same mesh and with the same reduction state and manual variation.
        let axes = |axis_type: MeshAxisType| {
            vec![MeshAxis::new("x", 2, axis_type).unwrap(), MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap()]
        };
        let mesh = LogicalMesh::new(axes(MeshAxisType::Explicit)).unwrap();
        let sharding =
            |mesh: &LogicalMesh, dimensions: Vec<ShardingDimension>| Sharding::new(mesh.clone(), dimensions).unwrap();
        let (replicated, unconstrained, split) =
            (ShardingDimension::Replicated, ShardingDimension::Unconstrained, ShardingDimension::sharded(["x"]));
        let open = sharding(&mesh, vec![unconstrained.clone(), split.clone()]);
        let closed = sharding(&mesh, vec![replicated.clone(), split.clone()]);
        assert!(replicates_unconstrained_dimensions(&open, &closed));
        assert!(replicates_unconstrained_dimensions(&closed, &open));
        assert!(replicates_unconstrained_dimensions(
            &sharding(&mesh, vec![unconstrained.clone(), replicated.clone()]),
            &sharding(&mesh, vec![replicated.clone(), unconstrained.clone()]),
        ));

        // Equal shardings, an unconstrained dimension that the other sharding splits, another rank, another mesh,
        // another reduction state, and another manual variation are not.
        assert!(!replicates_unconstrained_dimensions(&open, &open));
        assert!(!replicates_unconstrained_dimensions(
            &sharding(&mesh, vec![unconstrained.clone(), replicated.clone()]),
            &sharding(&mesh, vec![split.clone(), replicated.clone()]),
        ));
        assert!(!replicates_unconstrained_dimensions(&sharding(&mesh, vec![unconstrained.clone()]), &closed));
        let auto_mesh = LogicalMesh::new(axes(MeshAxisType::Auto)).unwrap();
        assert!(!replicates_unconstrained_dimensions(
            &open,
            &sharding(&auto_mesh, vec![replicated.clone(), split.clone()])
        ));
        let open = sharding(&mesh, vec![unconstrained, replicated.clone()]);
        let closed = sharding(&mesh, vec![replicated.clone(), replicated]);
        assert!(!replicates_unconstrained_dimensions(&open, &closed.clone().with_unreduced_axes(["y"]).unwrap()));
        assert!(!replicates_unconstrained_dimensions(&open, &closed.clone().with_reduced_axes(["y"]).unwrap()));
        assert!(!replicates_unconstrained_dimensions(&open, &closed.with_varying_manual_axes(["y"]).unwrap()));
    }

    #[test]
    fn test_is_untracked_refinement() {
        // A type without a sharding is an untracked refinement of a type of the same shape whose sharding has no
        // reduction state and no manual variation, whether that sharding replicates, splits over explicit axes, or
        // leaves a dimension unconstrained.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let untracked = Type::Array(test_vector_type(4));
        let typed = |dimension: ShardingDimension| {
            let sharding = Sharding::new(mesh.clone(), vec![dimension]).unwrap();
            test_vector_type(4).with_sharding(sharding).unwrap()
        };
        for dimension in
            [ShardingDimension::Replicated, ShardingDimension::Unconstrained, ShardingDimension::sharded(["x"])]
        {
            assert!(is_untracked_refinement(&Type::Array(typed(dimension)), &untracked));
        }

        // A tracked or missing sharding of either side, another shape, reduction state, manual variation, and a split
        // over manual axes are not.
        let replicated = typed(ShardingDimension::Replicated);
        let sharding = replicated.sharding().unwrap().clone();
        assert!(!is_untracked_refinement(&Type::Array(replicated.clone()), &Type::Array(replicated.clone())));
        assert!(!is_untracked_refinement(&untracked, &untracked));
        assert!(!is_untracked_refinement(&Type::Array(replicated.clone()), &Type::Array(test_vector_type(3))));
        for sharding in [
            sharding.clone().with_unreduced_axes(["y"]).unwrap(),
            sharding.clone().with_reduced_axes(["y"]).unwrap(),
            sharding.with_varying_manual_axes(["y"]).unwrap(),
        ] {
            let declared = test_vector_type(4).with_sharding(sharding).unwrap();
            assert!(!is_untracked_refinement(&Type::Array(declared), &untracked));
        }
        assert!(!is_untracked_refinement(&Type::Array(typed(ShardingDimension::sharded(["y"]))), &untracked));
    }
}
