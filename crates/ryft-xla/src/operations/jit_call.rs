use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use ryft_core::macros::check_count;
use ryft_core::{
    ArrayBatch, ArrayBatchingPolicy, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrType, ArrayType, BatchAxis,
    BatchableOperation, BatchedOutputs, BatchedProgram, BatchingContext, BatchingDriver, BatchingError,
    CalleeRegionDriver, CaptureConstant, CompiledCallOperation, Concretizable, Context, CotangentDestinationKind,
    CotangentDestinations, DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver,
    DifferentiationDual, DifferentiationError, DifferentiationPolicy, Dimension, InputRegionProvenance,
    InterpretableOperation, InterpretationDriver, MaybeZero, Operation, OperationBoundaryPruning, OperationFormatter,
    OperationProvider, OutputRegionProvenance, Parameterized, PartialEvaluationContext, PartialEvaluationDriver,
    PartialEvaluationValue, PartialValue, PartiallyEvaluatableOperation, ProgramBatchingOutputAxesPolicy, ProgramError,
    ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeValue,
    ReferenceDischargeableOperation, RegionDataFlow, RegionId, RegionInterface, RegionLiveness, RegionRef, RegionSlot,
    RematerializationOptimizationBarrier, ShardMapOperation, StagingContext, Tracer, TracingContext,
    TransposableOperation, TranspositionContext, TranspositionDriver, Type, TypeError, Typed, Value, ValueProjection,
    ValueResolution, Zero, ZeroOperation, discharge_positional_region_operation,
};
use ryft_mlir::dialects::func;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Region, SymbolVisibility,
    Type as MlirType, TypeAndAttributes, Value as MlirValue, ValueAndAttributes, ValueRef,
};

use crate::experimental::lowering::regions::lower_nested_program_inline;
use crate::experimental::lowering::types::lower_array_ir_type;
use crate::experimental::lowering::{CollectiveLoweringState, EffectTokens};
use crate::experimental::ops::{FlatXlaProgram, XlaConstant, XlaOperation, XlaProgram};
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaDomain, XlaLowerer, XlaValue};

/// Structural identity of a flat callee program attached behind a `jit_call`, used to deduplicate repeated nested
/// programs into shared private `func.func`s at lowering time.
///
/// Eligible programs (see [`supports_structural_dedup`]) are keyed by their canonical rendering plus their complete
/// flat input/output signature: type inference is deterministic and attached regions (control-flow bodies,
/// custom-function programs, and nested `jit_call` callees) render contextually inside their instructions, so two
/// programs that render identically with equal boundary types compute the same function and may share one emitted
/// function — even when they are distinct staged programs produced by separate transform passes (for example the
/// per-block primal and pullback programs of `grad(jit(f))` over repeated blocks).
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct JitCallProgramKey {
    /// Canonical [`Program`](ryft_core::Program) rendering (operation names, all bracketed attributes, and attached
    /// region bodies).
    rendered: String,

    /// Flat input types, which together with the rendering pin the full callee signature.
    pub(crate) input_types: Vec<ArrayIrType>,

    /// Flat output types, completing the callee signature when output-only placement metadata is not visible
    /// in the rendered body.
    pub(crate) output_types: Vec<ArrayIrType>,
}

/// Returns whether `program` may be deduplicated by structural identity.
///
/// A program is eligible only when the canonical rendering captures the full semantics needed to safely share a private
/// function. Although immediate constant payloads are rendered, a [`CaptureReference`](ryft_core::CaptureReference)
/// identifies a slot and type rather than the runtime value in the surrounding capture table, so programs with
/// constants remain ineligible. `shard_map` also remains outside this deduplication path because its lowering owns a
/// separate outlined boundary.
fn supports_structural_dedup<O: LowerableOperationFamily>(program: &FlatXlaProgram<O>) -> bool {
    program.regions().iter().all(|region| {
        region.atoms().iter().all(|atom| !atom.is_constant())
            && region
                .instructions()
                .iter()
                .all(|instruction| instruction.operation().projected_payload::<ShardMapOperation>().is_none())
    })
}

/// Returns whether a borrowed rooted region may be deduplicated by structural identity.
fn supports_structural_dedup_region<O: LowerableOperationFamily>(region: RegionRef<'_, XlaConstant, O>) -> bool {
    fn walk<O: LowerableOperationFamily>(
        region: RegionRef<'_, XlaConstant, O>,
        visited: &mut HashSet<RegionId>,
    ) -> bool {
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
/// structural deduplication (see [`supports_structural_dedup`]) and therefore always inline. The key is generic over
/// the program's constant universe so that lookups from the value-generic lowering path type-check; the map itself is
/// only ever populated from the staged [`XlaConstant`]-keyed pipeline.
fn jit_call_program_key<O: LowerableOperationFamily>(program: &FlatXlaProgram<O>) -> Option<JitCallProgramKey> {
    supports_structural_dedup(program).then(|| JitCallProgramKey {
        rendered: program.to_string(),
        input_types: program.input_types(),
        output_types: program.output_types(),
    })
}

/// Computes the deduplication key for a borrowed callee region, materializing only after borrowed eligibility checks.
fn jit_call_region_key<O: LowerableOperationFamily>(
    region: RegionRef<'_, XlaConstant, O>,
) -> Option<JitCallProgramKey> {
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
pub(crate) struct JitCallFunction<O = XlaOperation> {
    /// Symbol name of the emitted private function.
    pub(crate) symbol: String,

    /// Representative callee program for this key (materialized from its callee region), lowered once as the
    /// function body.
    pub(crate) program: FlatXlaProgram<O>,

    /// Flat input types of the callee, also the emitted function's argument types.
    pub(crate) input_types: Vec<ArrayIrType>,

    /// Flat output types of the callee, also the emitted function's result types.
    pub(crate) output_types: Vec<ArrayIrType>,
}

/// Shared private functions emitted for `jit_call` callees that occur more than once in a module.
///
/// Built once by [`collect_jit_call_functions`] before a module is lowered and threaded read-only through the
/// lowering pass. At each `jit_call` lowering site, a callee whose key is present is emitted as a `func.call` to the
/// shared function instead of being inlined; absent callees inline as before.
pub(crate) struct JitCallFunctionMap<O = XlaOperation> {
    /// Shared functions keyed by callee identity.
    pub(crate) functions: HashMap<JitCallProgramKey, JitCallFunction<O>>,

    /// Keys in first-occurrence order, so emitted symbol names and module layout are deterministic.
    pub(crate) order: Vec<JitCallProgramKey>,
}

impl<O> Default for JitCallFunctionMap<O> {
    fn default() -> Self {
        Self { functions: HashMap::new(), order: Vec::new() }
    }
}

impl<O: LowerableOperationFamily> JitCallFunctionMap<O> {
    /// Returns the shared function for `program`, if one was emitted for its identity.
    pub(crate) fn get(&self, program: &FlatXlaProgram<O>) -> Option<&JitCallFunction<O>> {
        self.functions.get(&jit_call_program_key(program)?)
    }
}

/// Counts `jit_call` callee occurrences in `program`, covering nested computations at every depth: attached regions
/// (control-flow bodies, custom-function programs, and callee bodies) all live in the program's one canonical
/// region arena, so the walk descends region edges without materializing nested programs. Shard-map body regions are
/// intentionally skipped: their `jit_call`s lower with shard-local types and always inline.
///
/// `counts` accumulates the occurrence count and a representative materialized program per identity, `order` records
/// keys in first-occurrence order, and `memo` caches the (possibly expensive) key computation per callee root
/// [`RegionId`] — interned shared callees repeat their root region id, so each shared callee is rendered once.
fn count_jit_calls<Input, Output, O: LowerableOperationFamily>(
    program: &XlaProgram<Input, Output, O>,
    counts: &mut HashMap<JitCallProgramKey, (usize, FlatXlaProgram<O>)>,
    order: &mut Vec<JitCallProgramKey>,
    memo: &mut HashMap<RegionId, Option<JitCallProgramKey>>,
) where
    Input: Parameterized<XlaConstant>,
    Output: Parameterized<XlaConstant>,
{
    fn walk<Input, Output, O: LowerableOperationFamily>(
        program: &XlaProgram<Input, Output, O>,
        region: RegionId,
        counts: &mut HashMap<JitCallProgramKey, (usize, FlatXlaProgram<O>)>,
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
            if instruction.operation().projected_payload::<ShardMapOperation>().is_some() {
                continue;
            }
            for &nested in instruction.regions() {
                walk(program, nested, counts, order, memo, visited);
            }
            if instruction.operation().projected_payload::<JitCallOperation<ArrayIrType>>().is_some() {
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
pub(crate) fn collect_jit_call_functions<Input, Output, O: LowerableOperationFamily>(
    program: &XlaProgram<Input, Output, O>,
) -> JitCallFunctionMap<O>
where
    Input: Parameterized<XlaConstant>,
    Output: Parameterized<XlaConstant>,
{
    let mut counts: HashMap<JitCallProgramKey, (usize, FlatXlaProgram<O>)> = HashMap::new();
    let mut order: Vec<JitCallProgramKey> = Vec::new();
    let mut memo: HashMap<RegionId, Option<JitCallProgramKey>> = HashMap::new();
    count_jit_calls(program, &mut counts, &mut order, &mut memo);

    let mut map = JitCallFunctionMap::default();
    for key in order {
        let (count, program) = counts.remove(&key).expect("every ordered key was counted");
        // Effectful callees always inline (even when repeated) so their effectful instructions chain onto the
        // caller's effect token in program order; a shared token-free function could not preserve that ordering.
        if count < 2 || !program.effects().classes().is_empty() {
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
pub(crate) fn emit_jit_call_function<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
    module_block: &mut BlockRef<'b, 'c, 't>,
    function: &JitCallFunction<O>,
    nested_functions: &Rc<JitCallFunctionMap<O>>,
    collective_state: &CollectiveLoweringState,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), LoweringError> {
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

/// Lowers one `jit_call` to either a `func.call` of a shared private function (when its callee was deduplicated) or
/// an inlined copy of the callee body (otherwise).
///
/// `input_values` are the lowered call operands in callee-input order. `capture_count` is the operation payload's
/// exact leading lifted-capture prefix length; deriving it from capture-constant indices instead would conflate the
/// independent capture namespaces of nested calls inside the callee arena.
pub(crate) fn lower_jit_call<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
    program: &FlatXlaProgram<O>,
    capture_count: usize,
    input_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    nested_functions: Option<&Rc<JitCallFunctionMap<O>>>,
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
                    .map(|r#type| lower_array_ir_type(r#type, context, location))
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
        &RematerializationOptimizationBarrier::None,
        nested_functions,
        collective_state,
        effect_tokens,
    )
}

/// Canonical operation name for [`JitCallOperation`].
pub const JIT_CALL_OPERATION_NAME: &str = "jit_call";

/// Staged call to a flat jitted XLA program. The callee program is not part of this payload: it is a shared
/// callee root [`Region`](ryft_core::Region) attached to the [`Instruction`](ryft_core::Instruction) applying the
/// operation (the single `["callee"]` slot), interned by [`Arc`] identity when the call is staged through the
/// [`BindingRegionDriver`](ryft_core::BindingRegionDriver) passed to [`Context::bind`], so repeated calls staged from
/// one function handle share one callee root and remain identity-comparable for call-site deduplication at lowering.
/// The `T` parameter fixes the callee boundary's type universe, allowing the reusable homogeneous-array batching form
/// and the executable composite array IR form to remain distinct payload types with one [`Operation`] contract each.
/// The retained `capture_count` names the callee's exact leading lifted-capture input prefix; it participates in
/// operation equality and callee-deduplication identity and scopes reference-capture resolution during analysis.
///
/// [`Operation::region_data_flow`] declares ordinary callee data flow: callee inputs receive instruction inputs
/// positionally and instruction outputs originate from the corresponding callee outputs. Capture-prefix retention
/// remains an independent boundary-pruning requirement.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct JitCallOperation<T: Type> {
    /// Number of leading callee inputs that form its lifted lexical capture prefix.
    capture_count: usize,

    /// Type universe of the callee boundary.
    marker: PhantomData<fn() -> T>,
}

impl<T: Type> Copy for JitCallOperation<T> {}

impl<T: Type> JitCallOperation<T> {
    /// Creates a staged jitted-call operation. The flat callee program is supplied as a shared region attachment to
    /// [`Context::bind`], and `capture_count` identifies its exact leading lifted-capture prefix.
    #[inline]
    pub(crate) fn new(capture_count: usize) -> Self {
        Self { capture_count, marker: PhantomData }
    }

    /// Returns the exact number of leading callee inputs that form the lifted-capture prefix.
    #[inline]
    pub(crate) fn capture_count(&self) -> usize {
        self.capture_count
    }
}

impl CompiledCallOperation<XlaConstant> for XlaOperation {
    #[inline]
    fn compiled_call(capture_count: usize) -> Self {
        Self::JitCall(JitCallOperation::new(capture_count))
    }
}

/// Bridges canonical internal physical positions to the public batching declaration.
fn batch_axis_from_position(axis: Option<usize>) -> BatchAxis {
    BatchAxis::from_optional_position(axis)
}

/// Recovers a canonical physical position returned by the core program batching pass.
fn batch_axis_position(axis: &BatchAxis) -> Option<usize> {
    axis.axis()
        .map(|axis| usize::try_from(axis.value()).expect("program batching returns canonical nonnegative axes"))
}

fn ensure_call_input_types<T: Type>(
    operation_name: &'static str,
    expected_types: &[T],
    input_types: &[T],
) -> Result<(), TypeError> {
    if expected_types.len() != input_types.len() {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` expected {} input(s) but got {}",
            expected_types.len(),
            input_types.len(),
        )));
    }
    for (index, (expected, actual)) in expected_types.iter().zip(input_types).enumerate() {
        if expected != actual {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` input #{index} expected {expected} but got {actual}",
            )));
        }
    }
    Ok(())
}

impl<T: Type> Operation for JitCallOperation<T> {
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        JIT_CALL_OPERATION_NAME
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
                "{} expects 1 attached callee region but got {}",
                JIT_CALL_OPERATION_NAME,
                region_interfaces.len()
            )));
        }
        let callee_interface = &region_interfaces[0];
        ensure_call_input_types(self.name(), callee_interface.input_types(), input_types)?;
        Ok(callee_interface.output_types().to_vec())
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
        input_count: usize,
        used_outputs: &[bool],
        regions: &mut dyn RegionLiveness,
    ) -> Result<Option<OperationBoundaryPruning<Self>>, ProgramError> {
        // Operands map onto the callee inputs one for one, so the operands that the callee does not use are dropped
        // together with its unused outputs (as in JAX's `dce_jaxpr_pjit_rule`). The lifted capture prefix is always
        // kept, because the captures in the callee refer to it by absolute position.
        let mut kept_inputs = regions.used_region_inputs(0, used_outputs)?;
        kept_inputs.iter_mut().take(self.capture_count).for_each(|kept| *kept = true);
        check_count!("input", kept_inputs, input_count, ProgramError);
        Ok(Some(OperationBoundaryPruning { operation: *self, kept_inputs, kept_outputs: used_outputs.to_vec() }))
    }

    #[inline]
    fn region_capture_input_count(&self, region_index: usize) -> Option<usize> {
        (region_index == 0).then_some(self.capture_count)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        let operation = OperationFormatter::new(formatter, indentation, JIT_CALL_OPERATION_NAME)?;
        if self.capture_count == 0 {
            Ok(())
        } else {
            operation.bracketed(|operation| operation.field("capture_count", self.capture_count))
        }
    }
}

// A jitted call forwards its operands onto its callee's inputs one for one and reports the callee's outputs as its
// own, which is the positionally forwarding shape the shared structured rewrite serves with no leading operands. The
// callee's lifted capture prefix is unaffected: threaded state is appended after every declared operand, so the prefix
// keeps its positions.
impl<T, C, P> ReferenceDischargeableOperation<C, P> for JitCallOperation<T>
where
    T: Type,
    JitCallOperation<T>: Operation<Type = C::Type>,
    C: Context<Operation: From<JitCallOperation<T>>>,
    P: ReferenceDischargePolicy<C>,
    C::Type: From<P::Referent>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        discharge_positional_region_operation(self, context, driver, inputs, 0, |_| self.clone())
    }
}

// Online partial-evaluation rule for a staged jitted call — ryft's analogue of JAX's call partial-evaluation
// rules: it splits the callee against the caller's known-ness while preserving the `jit_call` boundary on both
// sides.
//
// The split fires whenever the known-ness of the call inputs is mixed (i.e., some inputs are known and some are not).
// All-unknown calls preserve the original boundary. All-known calls normally do too, except that a residual policy
// can inline a pure callee into a live known-side trace. This exposes its intermediate producers to demand-driven
// residual placement, so a policy can save a compact intermediate rather than the callee's expanded output. Eager
// calls and callees with references, captures, or nominal type identities keep their boundary. Whether the mixed
// call's known inputs are concrete values of an eager known-side context or tracers into a live
// outer trace does not matter, as in JAX's call partial-evaluation rules, which always bind the known call: the known
// side is bound through the default fold-or-residualize policy, which compiles and executes it under an eager context
// (e.g., [`XlaDomain`](crate::XlaDomain)) and stages it into the outer program under a staging context. The `shard_map`
// rule follows the same policy.
//
// When the split fires, the callee is split through the shared
// [`PartitionedProgram`](ryft_core::partial::PartitionedProgram) machinery: the known side is bound into the
// enclosing known-side context wrapped in a fresh `jit_call` over the known call inputs that it uses, and the residual
// side is emitted as the residual `jit_call` over the surviving unknown call inputs plus the known-side call's
// residual-edge outputs. Residual edges that would merely repeat a known call input or a known call output are
// [forwarded](ryft_core::partial::PartitionedProgram::forward_residuals), so the residual call consumes that input or
// output directly and the known-side call does not return the same value twice.
impl<V, C> PartiallyEvaluatableOperation<C> for JitCallOperation<ArrayIrType>
where
    V: PartialEq
        + Value<Type = ArrayIrType>
        + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + CaptureConstant
        + Concretizable<bool>,
    C: Context<Type = ArrayIrType, Constant = V, Operation = XlaOperation<V>>,
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        let all_known = inputs.iter().all(PartialEvaluationValue::is_known);
        // Inlining an all-known callee exposes its producers to the policy only when they can be staged into the
        // known-side trace. Keep eager calls whole instead of replacing one compiled call with per-operation dispatch.
        let inline_known = all_known
            && context.residual_policy().is_some()
            && inputs.iter().any(|input| {
                input
                    .as_known()
                    .is_some_and(|value| matches!(context.parent().resolve(value), ValueResolution::Staged(_)))
            });
        if (all_known && !inline_known) || inputs.iter().all(PartialEvaluationValue::is_unknown) {
            return context.fold_or_residualize(
                XlaOperation::JitCall(*self),
                driver.regions().map(|region| region.to_program()).collect(),
                inputs,
            );
        }
        // Split the callee through the shared online boundary machinery, bind the known side into the enclosing
        // known-side context wrapped in a fresh `jit_call` over the original known call inputs, emit the residual
        // side as the residual `jit_call`, and reassemble the original output order.
        let callee = driver.region(0)?;
        // A reference operand, a reference output, or a reference access anywhere in the callee closure keeps the call
        // whole. Splitting would spread the accesses to one root across two compiled calls whose known side runs
        // first, and a known reference-typed feeder of the residual side would then mean that both sides access the
        // same root, the same reordering hazard that keeps a `scan` body whole. The default
        // rule preserves effect order by placing the whole call on one side instead. An allocation that nothing
        // accesses is dead intrinsic state
        // that neither side retains, so it does not pin the boundary, but an allocation that escapes through a
        // reference output is live state whose later accesses are unknown to the callee, so it does.
        let accesses_references = inputs.iter().any(|input| input.r#type().is_reference())
            || callee.output_types().iter().any(Type::is_reference)
            || callee
                .instructions_in_closure()
                .any(|(_, instruction)| instruction.operation().effects().has_accesses());
        if accesses_references {
            return context.fold_or_residualize(XlaOperation::JitCall(*self), vec![callee.to_program()], inputs);
        }
        // Partitioning does not remap the absolute indices of capture constants retained in attached regions, so a
        // split boundary would leave them naming a compacted or absent capture prefix — failing lowering at best and
        // silently aliasing an unrelated leading operand at worst. Preserve the original call boundary for any callee
        // whose closure still holds a capture constant.
        if callee.contains_atom_in_closure(|atom| {
            atom.as_constant().is_some_and(|constant| constant.capture_index().is_some())
        }) {
            return context.fold_or_residualize(XlaOperation::JitCall(*self), vec![callee.to_program()], inputs);
        }
        if all_known {
            // Keep effectful work and nominally scoped types behind their validated call boundary. The pure,
            // identity-free case can replay into this context directly; placement still happens only when residual
            // work actually demands a value, so unused outputs cannot trigger policy decisions or rejections.
            if !callee.effects().classes().is_empty()
                || inputs.iter().any(|input| input.r#type().identities().next().is_some())
                || callee.contains_atom_type_in_closure(|r#type| r#type.identities().next().is_some())
            {
                return context.fold_or_residualize(XlaOperation::JitCall(*self), vec![callee.to_program()], inputs);
            }
            return driver.partially_evaluate_region(context, 0, inputs.to_vec());
        }
        let input_known = inputs.iter().map(PartialEvaluationValue::is_known).collect::<Vec<bool>>();
        let partition = driver.partition_program(context, callee, input_known.as_slice())?;
        // A trivial partition — one whose known program contains no instructions — hoists no work (its known side
        // can only forward known inputs as residual edges), so keep the original boundary and let the default
        // materialize those knowns directly as residual feeders.
        if partition.known_program().instructions().is_empty() {
            return context.fold_or_residualize(XlaOperation::JitCall(*self), vec![callee.to_program()], inputs);
        }
        // Feed the residual call directly from the known call inputs and known call outputs that residual edges would
        // merely repeat (JAX's `in_fwd` and `out_fwd`), so that the known-side call returns each value once.
        let partition = partition.forward_residuals()?;
        // Known inputs keep callee source order, so the known-side callee's leading inputs are exactly the known
        // members of the original lifted-capture prefix that it still uses; the residual callee's inputs are call
        // inputs, known call outputs, and residual edges, which never form a capture prefix. The guard above already
        // preserved the boundary of any callee retaining attached-region capture constants, so neither derived callee
        // can hold one.
        let known_capture_count =
            partition.known_input_indices().iter().filter(|index| **index < self.capture_count()).count();
        context.inline_partitioned_program(
            partition,
            inputs,
            |known_program| (XlaOperation::JitCall(JitCallOperation::new(known_capture_count)), vec![known_program]),
            |residual_program| (XlaOperation::JitCall(JitCallOperation::new(0)), vec![residual_program]),
        )
    }
}

// Batching rule for [`JitCallOperation`]: the callee region is rebatched over the mapped input axes (via
// [`BatchingDriver::batch_program`]) and the batched call is bound through `context.parent()` with the
// batched callee re-attached. An eager
// client-backed parent (e.g., [`XlaDomain`](crate::XlaDomain)) compiles and executes the batched call immediately, a
// staging parent stages it into the enclosing trace, and a differentiation parent dispatches it through its own
// `jit_call` JVP rule — which is what serves `vmap` nested inside `gradient`/`linearize` closures.
impl<C> BatchableOperation<C, ArrayIrBatchingPolicy> for JitCallOperation<ArrayIrType>
where
    C: Context<Type = ArrayIrType>,
    C::Operation: From<Self>,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatchingPolicy>, BatchingError> {
        let axes = inputs.iter().map(ArrayIrBatch::batch_axis).collect::<Vec<_>>();
        if axes.iter().all(BatchAxis::is_replicated) {
            return Ok(context
                .parent()
                .bind(
                    *self,
                    vec![driver.region(0)?.to_program()],
                    &inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>(),
                )?
                .into_iter()
                .map(ArrayIrBatch::replicated)
                .collect::<Vec<_>>()
                .into());
        }
        // The canonical composite transform threads one leading dimension through the callee. Treat it as an
        // additional explicit capture and remove only its bookkeeping output; the original capture order remains.
        let (callee, axes) = driver
            .batch_program(context, driver.region(0)?, &axes, ProgramBatchingOutputAxesPolicy::Natural)?
            .into_parts();
        let mut values = vec![context.axis_extent().clone()];
        values.extend(inputs.iter().map(|input| input.value().clone()));
        let mut outputs = context.parent().bind(Self::new(self.capture_count() + 1), vec![callee], &values)?;
        check_count!("output", outputs, axes.len() + 1, ProgramError);
        outputs.remove(0);
        Ok(outputs
            .into_iter()
            .zip(axes)
            .map(|(value, axis)| ArrayIrBatch::new(value, axis))
            .collect::<Result<Vec<_>, _>>()?
            .into())
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
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy>, BatchingError> {
        let physical_inputs = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
        // Rebatch the callee region over the mapped input axes when any input carries the batch axis; an
        // all-replicated call binds its original callee unchanged.
        //
        // Unlike the forward-mode and transpose rules below, this rebatching is deliberately not served from the
        // callee region's retained transform cache: the batched program depends on live `BatchingContext` state (the
        // batch extent, the axis name, and the surrounding nesting) whose complete key surface is not yet explicit,
        // so caching it against the callee region and its input axes alone would risk exactly the unsound key reuse
        // that cache is designed to avoid. The decided follow-up is to specify that key material when composite
        // batching lands behind the `CompiledXlaFunction::batch` stub, which has to define it anyway: batching then
        // joins the region transform cache behind a gate measurement, through a batching-policy hook returning
        // `Option<key>` where `None` keeps a policy uncached, so soundness stays opt-in per policy. See
        // `.tasks/plan_general_transform_caching_analysis.md`.
        let (batched_callee, output_axes) = match ArrayBatch::common_batch_size(inputs)? {
            Some(_) => {
                let input_batch_axes = inputs
                    .iter()
                    .map(|input| batch_axis_from_position(input.batch_axis_position()))
                    .collect::<Vec<_>>();
                let (batched_callee, output_axes) = driver
                    .batch_program(
                        context,
                        driver.region(0)?,
                        input_batch_axes.as_slice(),
                        ProgramBatchingOutputAxesPolicy::Natural,
                    )?
                    .into_parts();
                let output_axes = output_axes.iter().map(batch_axis_position).collect::<Vec<_>>();
                (batched_callee.into_simplified()?, output_axes)
            }
            None => {
                let callee = driver.region(0)?;
                let output_axes = vec![None; callee.output_types().len()];
                (callee.to_program(), output_axes)
            }
        };
        let outputs =
            context
                .parent()
                .bind(*self, CalleeRegionDriver::new(&[Arc::new(batched_callee)]), &physical_inputs)?;
        Ok(outputs
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| ArrayBatch::new(output, batch_axis_from_position(axis)))
            .collect::<Result<Vec<_>, _>>()?
            .into())
    }
}

// Capture-free forward-mode (JVP) rule for [`JitCallOperation`], binding a primal `jit_call` and a tangent
// `jit_call` as ordinary XLA-enum operations through the active context: a staging context stages both calls over
// its shared builder, while an eager context (e.g. a client-backed [`XlaDomain`](crate::XlaDomain)) compiles and
// executes them immediately, which is what powers top-level `jvp` over concrete arrays.
//
// This realizes the identity `jvp(jit(f)) = jit(jvp f)`: rather than capturing the primal inputs as residual factors
// and staging a linear `jit_call`, the rule keeps the compilation boundary and threads every residual as a plain
// primal operand edge between two `jit_call`s, so no symbolic capture is ever introduced. The enclosing
// partial-evaluation split then discovers the residual operand edges structurally, exactly as it does for the
// condition and rematerialize rules.
//
// The rule linearizes the callee program capture-free through
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
// (concretely [`XlaConstant`] for staged XLA programs), so the split halves ride the fresh primal and tangent calls
// as shared callee regions. Preserving both `jit_call` boundaries keeps the callee body out of the caller's program,
// so forward mode over a jitted call stays compiled rather than inlined.
//
// # Parameters
//
//   - `context`: Active evaluation or staging context used to bind the differentiated calls.
//   - `driver`: Call-scoped access to the attached callee region.
//   - `inputs`: Primal and tangent values for the call operands.
impl<C, V> DifferentiableOperation<C> for JitCallOperation<ArrayIrType>
where
    C: Context<Type = ArrayIrType, Constant = V, Operation = XlaOperation<V>> + Zero<C::Value>,
    V: PartialEq
        + Value<Type = ArrayIrType>
        + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
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
        // nondifferentiated plumbing that receives no tangent input; the remaining inputs take the operand duals'
        // activity: an operand with a live tangent is active, while an operand with a structurally zero tangent is
        // inactive like a missing one (JAX's `which_nz`), so it gets no tangent callee slot instead of being
        // materialized as a zero operand. A plumbing reference operand (whose structurally zero tangent is a missing
        // tangent reference) is therefore inactive too. The primal sub-program produces `[outputs..., residuals...]`
        // and the tangent sub-program consumes `[active(input_tangents)..., residuals...]`; the residual count is the
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
        let linearization = driver.linearize_program(callee, &input_indices)?;

        // An output tangent that depends on no active input tangent (e.g., the tangent of a constant output or of an
        // output computed only from inactive inputs) is a structural zero (JAX's `which_nz_out`), so the tangent callee
        // is projected onto the live output tangents, and every other output is paired with a structurally zero
        // tangent. When no output tangent is live and the tangent callee has no observable effects, no tangent call is
        // bound, and the source callee is bound as the primal call, since nothing would consume its residuals.
        //
        // The tangent program's leading inputs are compact active tangents, not the original primal input indices;
        // trailing residual inputs are held independent during dependence analysis. Linearization has validated both
        // boundaries, so subtracting the residual count cannot underflow.
        let tangent_program = linearization.tangent();
        let tangent_input_count = tangent_program.input_count() - linearization.residual_count();
        let tangent_slots = linearization.primal().entry_region_ref().tangent_output_mask(&input_indices)?;
        let mut dependence =
            tangent_program.output_dependence(&(0..tangent_input_count).collect::<Vec<_>>())?.into_iter();

        // Map compact tangent output dependence back to the original callee's output order. Outputs without tangent
        // slots consume no dependence entry; trailing primal residual outputs are outside this boundary entirely.
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
            Arc::new(tangent_program.with_outputs(&live_tangent_slots)?)
        };
        let primal_operands = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        if !output_activity.contains(&true) && !tangent_program.effects().is_retained_when_unused() {
            let callee = Arc::new(callee.to_program());
            return context
                .primal()
                .bind(XlaOperation::JitCall(*self), CalleeRegionDriver::new(&[callee]), &primal_operands)?
                .into_iter()
                .map(DifferentiationDual::new_with_zero_tangent)
                .collect();
        }
        let (primal_program, _, _) = linearization.into_parts();

        // Wrap the primal sub-program in a fresh `jit_call` and bind it over the operand primals, recovering the
        // primal outputs followed by the residual values. The shared sub-program handles are attached directly, so
        // repeated binds of one derived callee intern by `Arc` identity instead of copying it again.
        // The primal sub-program preserves the callee's inputs in order, so the original lifted-capture prefix
        // survives verbatim and its length carries over.
        let primal_call = XlaOperation::JitCall(JitCallOperation::new(self.capture_count()));
        let mut primal_call_outputs =
            context.primal().bind(primal_call, CalleeRegionDriver::new(&[primal_program]), &primal_operands)?;
        if primal_call_outputs.len() < output_count {
            return Err(ProgramError::MalformedProgram(format!(
                "jit_call primal program produced {} outputs which is fewer than its {output_count} primal \
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
        // The tangent callee's inputs are live input tangents followed by residuals — never a lifted-capture prefix.
        // Attached-region capture constants cannot appear in it because fresh-root region traces reject bodies that
        // register captures (`ProgramError::DiscardedCaptures`), and a stale reference that slipped in anyway would
        // fail lowering loudly against the tangent callee's empty capture prefix instead of aliasing an unrelated
        // value.
        let tangent_call = XlaOperation::JitCall(JitCallOperation::new(0));
        let tangent_outputs =
            context
                .tangent()
                .bind(tangent_call, CalleeRegionDriver::new(&[tangent_program]), &tangent_operands)?;
        let tangent_output_count = output_activity.iter().filter(|&&active| active).count();
        check_count!("output", tangent_outputs, tangent_output_count, ProgramError);

        let mut tangent_outputs = tangent_outputs.into_iter();
        Ok(primal_outputs
            .into_iter()
            .zip(output_activity)
            .map(|(primal, active)| {
                if !active {
                    DifferentiationDual::new_with_zero_tangent(primal)
                } else {
                    DifferentiationDual::new(primal, tangent_outputs.next().unwrap())
                }
            })
            .collect::<Result<Vec<_>, _>>()?)
    }
}

/// Returns a tracer for `cotangent`, materializing a structural zero through the canonical XLA representation when a
/// higher-order transpose call requires an ordinary SSA operand. Dynamic array zeros read their extent operands from
/// the known dimension inputs of the differentiated call; they are never reconstructed from type metadata.
pub(crate) fn materialize_transpose_cotangent<
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
>(
    context: &TracingContext<V, XlaOperation<V>>,
    cotangent: &MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>,
    output_type: &ArrayIrType,
    inputs: &[PartialValue<Tracer<TracingContext<V, XlaOperation<V>>>>],
) -> Result<Tracer<TracingContext<V, XlaOperation<V>>>, ProgramError> {
    if let MaybeZero::Value(cotangent) = cotangent {
        return Ok(cotangent.clone());
    }

    let (operation, operands) = match output_type {
        ArrayIrType::Array(array_type)
            if array_type.shape().dimensions().iter().any(|dimension| matches!(dimension, Dimension::Dynamic(_))) =>
        {
            // The generic input-free provider cannot construct a dynamically shaped array zero. Resolve each dynamic
            // axis from the known dimension inputs and pass those extents through the mixed zero operation instead.
            let operands = array_type
                .shape()
                .dimensions()
                .iter()
                .filter_map(Dimension::variable)
                .map(|variable| {
                    inputs
                        .iter()
                        .filter_map(PartialValue::as_known)
                        .find(|input| {
                            matches!(
                                input.r#type().as_ref(),
                                ArrayIrType::Dimension(r#type) if r#type.variable() == Some(variable)
                            )
                        })
                        .cloned()
                        .ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "cannot materialize dynamic transpose cotangent of type {output_type} because no known \
                                 dimension input defines `{variable}`",
                            ))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            (XlaOperation::from(ZeroOperation::new(array_type.clone())), operands)
        }
        _ => (XlaOperation::provide(ZeroOperation::new(output_type.clone()), &[])?, Vec::new()),
    };
    let mut outputs = context.stage_operation(operation, Vec::new(), operands.as_slice())?;
    check_count!("output", outputs, 1, ProgramError);
    Ok(outputs.remove(0))
}

/// Transposes a tangent [`JitCallOperation`] while retaining its compilation boundary.
///
/// The [`TranspositionDriver`] transposes the attached callee under the operands' linearity and cotangent-destination
/// masks. Known operands supply residual or captured values and may appear anywhere in the input boundary. The
/// transposed call consumes ordinary output cotangents, live cotangent references, and known operands, in that order.
/// Reference outputs forward input roots and therefore have no separate output-cotangent slot.
///
/// Returned cotangents follow the original input order: linear value inputs receive call results, reference inputs
/// accumulate into their supplied cotangent destinations, and known or ignored inputs receive structural zeros.
///
/// # Parameters
///
///   - `operation`: Primal tangent `jit_call` staged into the tangent program.
///   - `context`: Active transpose tracing context the pullback is staged into.
///   - `driver`: Instruction-scoped access to the attached callee region and its recursive transposition machinery.
///   - `inputs`: Per-operand [`PartialValue`] knowledge, mirroring the callee's inputs one-to-one. The
///     [`Unknown`](PartialValue::Unknown) entries are the input tangents; the [`Known`](PartialValue::Known) entries
///     carry the residual and captured-constant-tangent tracers the pullback reads.
///   - `outputs`: Symbolic cotangents for the tangent call's outputs.
///   - `cotangents`: Cotangent destinations of the operands (refer to the documentation of
///     [`TranspositionContext::cotangent_destinations`]). The callee is transposed with their destination kinds, so a
///     live (`Reference`-kind) reference operand's cotangent reference is an operand of the transposed call, which
///     accumulates into it in place and returns it by identity, while a dead (`Ignore`-kind) reference operand has no
///     slot in the transposed callee at all.
pub fn transpose_primal_jit_call<
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    D: TranspositionDriver<V, XlaOperation<V>>,
>(
    _operation: &JitCallOperation<ArrayIrType>,
    context: &mut TracingContext<V, XlaOperation<V>>,
    driver: &D,
    inputs: &[PartialValue<Tracer<TracingContext<V, XlaOperation<V>>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>],
    cotangents: &CotangentDestinations<Tracer<TracingContext<V, XlaOperation<V>>>>,
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>>, ProgramError> {
    // A jitted call with no live output cotangents and no live reference operand is a zero linear map, so every
    // operand cotangent is zero. A live reference operand keeps the call live, because its accumulated state cotangent
    // flows through the transposed callee even when no ordinary output cotangent does. A callee with deferred work or
    // observable rule effects also keeps it live.
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
    // captured-constant tangent). The linear operands need not lead: a captured compiled function threads its captured
    // prefix as known leading operands. The dispatch guarantees a `Known` operand carries its pullback value, so each
    // known tracer is read directly in callee-input order.
    let operand_linear = inputs.iter().map(PartialValue::is_unknown).collect::<Vec<_>>();
    let callee = driver.region(0)?;
    check_count!("input", operand_linear, callee.input_types().len(), ProgramError);

    // Transpose the callee with respect to its linear inputs, through the region's retained transform cache so that a
    // callee shared by several outer programs is transposed once per linearity mask. A live reference-typed linear
    // input is transposed with a `Reference` destination and a dead one with an `Ignore` destination, so the transposed
    // callee maps `[non_reference_output_cotangents..., cotangent_references..., known_input_values...]` to
    // `[linear_input_cotangents...]`, in callee-input order, where a live reference input's cotangent is its cotangent
    // reference itself and a dead reference input has no cotangent slot.
    let (input_indices, destination_kinds): (Vec<_>, Vec<_>) = operand_linear
        .iter()
        .zip(cotangents.kinds())
        .enumerate()
        .filter_map(|(index, (linear, kind))| linear.then_some((index, *kind)))
        .unzip();
    let transposed_callee = driver.transpose_program(callee, &input_indices, &destination_kinds)?;

    // Stage the output cotangents, materializing a typed zero for each structurally zero cotangent, then stage a fresh
    // `jit_call` over the transposed callee on `[outputs..., cotangent_references..., known_input_values...]`. Its
    // outputs are the linear-input cotangents.
    let output_types = callee.output_types();
    check_count!("output", outputs, output_types.len(), ProgramError);
    let mut operands = Vec::with_capacity(output_types.len() + inputs.len());
    for (cotangent, output_type) in outputs.iter().zip(output_types.iter()) {
        // A reference output forwards a callee input root whose state cotangent lives in that operand's cotangent
        // reference, so it owns no cotangent slot.
        if output_type.is_reference() {
            continue;
        }
        let output_cotangent_type = output_type.cotangent()?;
        operands.push(materialize_transpose_cotangent(context, cotangent, &output_cotangent_type, inputs)?);
    }
    operands.extend(cotangents.references().iter().cloned());
    operands.extend(inputs.iter().filter_map(PartialValue::as_known).cloned());
    // The transposed callee's inputs are output cotangents, cotangent references, and known input values — never a
    // lifted-capture prefix. Attached-region capture constants cannot appear in it because region bodies are traced
    // through fresh-root contexts.
    let transposed_call = XlaOperation::JitCall(JitCallOperation::new(0));
    let input_cotangents =
        context.bind(transposed_call, CalleeRegionDriver::new(&[transposed_callee]), operands.as_slice())?;
    let output_count = (0..inputs.len())
        .filter(|&index| operand_linear[index] && cotangents.returns_cotangent(index))
        .count();
    check_count!("output", input_cotangents, output_count, ProgramError);

    // Reassemble one cotangent per operand: the known operands carry structural zeros, while the linear input tangents
    // receive the transposed call's outputs in callee-input order. A live reference tangent's output is its cotangent
    // reference, whose contents were accumulated in place, and a dead one has no output, so every reference operand
    // receives a structural zero.
    let mut input_cotangents = input_cotangents.into_iter();
    operand_linear
        .iter()
        .zip(inputs)
        .enumerate()
        .map(|(index, (&linear, input))| match cotangents.kind(index) {
            CotangentDestinationKind::Return if linear => Ok(MaybeZero::Value(input_cotangents.next().unwrap())),
            CotangentDestinationKind::Reference => {
                if cotangents.is_reference_input(index) {
                    input_cotangents.next();
                }
                input.r#type().cotangent().map(MaybeZero::Zero).map_err(ProgramError::from)
            }
            CotangentDestinationKind::Return | CotangentDestinationKind::Ignore => {
                input.r#type().cotangent().map(MaybeZero::Zero).map_err(ProgramError::from)
            }
        })
        .collect()
}

// Transpose rule for a primal tangent [`JitCallOperation`], forwarding to [`transpose_primal_jit_call`] with the
// cotangent references of its linear reference-typed operands resolved (and allocated on first use) through the
// enclosing [`TranspositionContext`]. The callee transposition happens on the concretely [`XlaConstant`]-keyed
// [`FlatXlaProgram`], so the recursion is resolved once at definition time and instantiating this implementation
// introduces no recursive [`TransposableOperation`] obligation on [`XlaOperation`].
impl<V> TransposableOperation<V, XlaOperation<V>> for JitCallOperation<ArrayIrType>
where
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    fn transpose<D: TranspositionDriver<V, XlaOperation<V>>>(
        &self,
        context: &mut TranspositionContext<V, XlaOperation<V>>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, XlaOperation<V>>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>],
        accumulators: &[ryft_core::CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        check_count!("output", outputs, driver.region(0)?.output_types().len(), ProgramError);
        check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
        let contributions =
            (|| -> Result<Vec<MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>>, DifferentiationError> {
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

// Eager interpretation of a staged jitted call over concrete [`XlaArray`]s. Top-level flat programs containing
// `jit_call` instructions (for example the pullbacks produced by eager `vjp`/`grad`) replay through
// [`Program::interpret_in_context`](ryft_core::Program::interpret_in_context), whose bind channel hands the callee
// region to [`Context::bind`] so the call is compiled whole through this domain's dispatch cache and executed on its
// PJRT client — mirroring JAX dispatching a jitted function called from eager code straight to the compiled
// executable. This rule covers the remaining path — a `jit_call` nested inside another region being interpreted —
// by re-entering the active interpreter on the callee region, which dispatches the callee's operations one by one
// through this same domain.
impl<'c> InterpretableOperation<XlaDomain<'c>> for JitCallOperation<ArrayIrType> {
    fn interpret<D: InterpretationDriver<XlaDomain<'c>>>(
        &self,
        context: &XlaDomain<'c>,
        driver: &D,
        inputs: &[XlaValue<'c>],
    ) -> Result<Vec<XlaValue<'c>>, ProgramError> {
        driver.interpret_region(context, 0, inputs.to_vec())
    }
}

impl LowerableOperation for JitCallOperation<ArrayIrType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayIrType],
        _output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let [callee] = lowerer.regions.as_slice() else {
            return Err(LoweringError::UnsupportedOp {
                op: format!("jit_call expected 1 attached region but got {}", lowerer.regions.len()),
            });
        };
        lower_jit_call(
            callee,
            self.capture_count(),
            input_values,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
            lowerer.nested_functions.as_ref(),
            &lowerer.collective_state,
            &mut lowerer.effect_tokens,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::{formatdoc, indoc};
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, ArrayIrType, ArrayOperation, ArrayType, AssertOperation, AtomId, CaptureReference,
        ConditionOperation, Context, CosOperation, CotangentDestinationKind, CotangentDestinations, DataType,
        DifferentiableType, DifferentiationError, Dimension, DimensionBounds, DimensionType, DimensionValue,
        DimensionVariable, DotDimensionNumbers, DotOperation, EffectClasses, InputRegionProvenance, LogicalMesh,
        MaybeZero, MeshAxis, MeshAxisType, MulOperation, NoStorage, Operation, OutputRegionProvenance, PartialValue,
        Placeholder, ProgramBuilder, ProgramError, ReferenceAddUpdateOperation, ReferenceFreezeOperation,
        ReferenceNewOperation, ReferenceReadOperation, ReferenceSource, ReferenceSwapOperation, ReferenceType,
        RegionDataFlow, RegionDriver, RegionInterface, RegionRef, RematerializeOperation, ResidualCandidate,
        ResidualDecision, ResidualPolicy, ResidualPolicyReference, ResidualRejection, Shape, Sharding,
        ShardingDimension, SinOperation, StagingContext, TagOperation, Tracer, TracingContext, TranspositionDriver,
        TypeError, Typed, Value, ZeroOperation,
    };

    use crate::experimental::lowering::tests::{
        XlaProgramBuilder, add_xla_jit_call, lower_replicated_shard_map_body, test_vector_type,
        unproject_plain_program, xla_identity_branch, xla_neg_branch,
    };
    use crate::experimental::ops::tests::vector_type;
    use crate::experimental::ops::{
        XlaArrayConstant, XlaConstant, XlaOperation, XlaProgram, XlaProgramBuilder as CompositeXlaProgramBuilder,
    };
    use crate::{XlaArray, to_mlir_module_for_program};

    use super::*;

    /// Test-only driver that exposes one source callee and returns a predetermined transpose for it.
    struct TestTranspositionDriver {
        /// Source callee exposed to the JIT-call transpose rule.
        source: XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>,

        /// Predetermined transposed callee returned by the recursive request.
        transposed: XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>,
    }

    impl RegionDriver<XlaConstant, XlaOperation> for TestTranspositionDriver {
        fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, XlaConstant, XlaOperation>>
        where
            XlaConstant: 'r,
            XlaOperation: 'r,
        {
            std::iter::once(self.source.entry_region_ref())
        }
    }

    impl TranspositionDriver<XlaConstant, XlaOperation> for TestTranspositionDriver {
        fn transpose_program(
            &self,
            _region: RegionRef<'_, XlaConstant, XlaOperation>,
            _input_indices: &[usize],
            _destination_kinds: &[CotangentDestinationKind],
        ) -> Result<Arc<XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>>, DifferentiationError> {
            Ok(Arc::new(self.transposed.clone()))
        }
    }

    /// Compiles and executes an array-only `f32` fixture using its declared unsharded program signature.
    fn execute_f32_program(
        client: &ryft_pjrt::Client<'_>,
        program: &XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>,
        values: Vec<Vec<f32>>,
    ) -> Vec<Vec<f32>> {
        use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions};
        use ryft_pjrt::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram};

        use crate::tests::{values_from_bytes, values_to_bytes};
        use crate::to_mlir_module_for_program;

        let array_type = |r#type: &ArrayIrType| match r#type {
            ArrayIrType::Array(r#type) => r#type.clone(),
            _ => panic!("expected an array signature"),
        };
        let input_types = program.input_types().iter().map(array_type).collect::<Vec<_>>();
        let output_types = program.output_types().iter().map(array_type).collect::<Vec<_>>();
        let module = to_mlir_module_for_program(program, &[], &input_types, &output_types, "main", None, None).unwrap();
        let executable = client
            .compile(
                &PjrtProgram::Mlir { bytecode: module.into_bytes() },
                &CompilationOptions {
                    executable_build_options: Some(ExecutableCompilationOptions {
                        device_ordinal: -1,
                        replica_count: 1,
                        partition_count: 1,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )
            .unwrap();
        let device = executable.addressable_devices().unwrap().remove(0);
        assert_eq!(values.len(), program.input_count());
        let inputs = values
            .into_iter()
            .zip(program.input_types())
            .map(|(values, r#type)| {
                let ArrayIrType::Array(r#type) = r#type else {
                    panic!("expected an array input");
                };
                assert_eq!(r#type.data_type(), DataType::F32);
                let dimensions =
                    r#type.static_shape().unwrap().dimensions().iter().map(|value| *value as u64).collect::<Vec<_>>();
                let buffer = client
                    .buffer(
                        values_to_bytes(&values).as_slice(),
                        BufferType::F32,
                        &dimensions,
                        None,
                        device.clone(),
                        None,
                    )
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
            )
            .unwrap()
            .block_until_ready()
            .unwrap()
            .remove(0)
            .outputs
            .iter()
            .map(|output| values_from_bytes::<f32>(&output.copy_to_host(None).unwrap().r#await().unwrap()))
            .collect()
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
            let first_capture = builder.add_input(array_type.clone().into());
            let _second_capture = builder.add_input(array_type.clone().into());
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
            let input = builder.add_input(array_type.clone().into());
            let callee_region = builder.intern_callee(&inner, None).unwrap();
            let output = builder
                .add_instruction(
                    XlaOperation::JitCall(JitCallOperation::new(2)),
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
        let input = builder.add_input(array_type.clone().into());
        let callee_region = builder.intern_callee(&outer, None).unwrap();
        let output = builder
            .add_instruction(XlaOperation::JitCall(JitCallOperation::new(0)), vec![callee_region], vec![input], None)
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

    /// Builds the flat callee `f(x) = x + x` over a vector type, returned behind an [`Rc`] so callers control
    /// whether `jit_call` sites share one program (pointer identity) or use structurally-identical distinct programs.
    fn xla_add_self_callee(input_type: ArrayType) -> Arc<FlatXlaProgram> {
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(input_type.into());
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
        Arc::new(builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap())
    }

    /// Lowers an outer program that calls `callees` (one `jit_call` each) and sums the results, returning the
    /// module text. Each callee is `f(x) = x + x`; the outer function is `g(x) = sum_i callee_i(x)`.
    fn lower_two_jit_call_module(callees: Vec<Arc<FlatXlaProgram>>) -> String {
        let array_type = test_vector_type(4);
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(array_type.clone().into());
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
            let input = builder.add_input(vector_type.clone().into());
            let first = add_xla_jit_call(&mut builder, &callee, vec![input]);
            let second = add_xla_jit_call(&mut builder, &callee, vec![input]);
            let output =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![first, second], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
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

    /// Builds a flat callee whose body contains a `condition` instruction (`f(p, x) = if p { -x } else { x }`),
    /// making it ineligible for structural `jit_call` deduplication because its nested branch bodies do not render
    /// into the callee's canonical program text.
    fn xla_condition_callee() -> Arc<FlatXlaProgram> {
        let vector_type = test_vector_type(4);
        let mut builder = CompositeXlaProgramBuilder::new();
        let true_region = builder.import_program(unproject_plain_program(xla_neg_branch(vector_type.clone())));
        let false_region = builder.import_program(unproject_plain_program(xla_identity_branch(vector_type.clone())));
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(vector_type.into());
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
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(vector_type.clone().into());
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
    fn test_repeated_effectful_jit_call_callees_inline_and_chain_prints() {
        use ryft_core::PrintOperation;

        // A repeated `jit_call` callee that prints is excluded from function deduplication and inlines at every
        // call site, so both inlined prints chain onto the caller's ordered-I/O token in program order (a shared
        // token-free `func.func` could not preserve that ordering).
        let array_type = test_vector_type(4);
        let mut callee_builder = XlaProgramBuilder::new();
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
    fn test_jit_call_reference_provenance_is_positional() {
        // A jitted call forwards its operands to the callee positionally, so region provenance, capture counts, and
        // output provenance are all index-preserving for the single callee region and absent for any other region.
        let operation = JitCallOperation::<ArrayIrType>::new(2);
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
    fn test_jit_call_region_data_flow() {
        let operation = JitCallOperation::<ArrayIrType>::new(2);
        assert!(matches!(operation.region_data_flow(), RegionDataFlow::Provenance));
        let dispatched = XlaOperation::<XlaConstant>::from(operation);
        assert!(matches!(dispatched.region_data_flow(), RegionDataFlow::Provenance));
    }

    #[test]
    fn test_jit_call_boundary_pruning() {
        // The callee maps its lifted capture `c` and its operands `(x, y)` to `(cos(x), c * y)`. Using only the first
        // output drops the operand `y` and the second output, while the capture prefix is kept although nothing uses
        // it.
        let r#type = ArrayIrType::Array(vector_type());
        let callee = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let captured = builder.add_input(r#type.clone());
            let x = builder.add_input(r#type.clone());
            let y = builder.add_input(r#type.clone());
            let cosine = builder.add_instruction(CosOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
            let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![captured, y], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![cosine, product],
                    vec![Placeholder; 3],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let inputs = (0..3).map(|_| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let outputs = builder
            .add_instruction(XlaOperation::JitCall(JitCallOperation::new(1)), vec![callee], inputs, None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![outputs[0]], vec![Placeholder; 3], vec![Placeholder])
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
    fn test_jit_call_residual_candidates_classify_through_callee_provenance() {
        use std::sync::{Arc, Mutex};

        use ryft_core::RematerializationPolicyFn;

        // `jit_call` reports positional output-region provenance, so a residual produced by a computed callee output
        // is classified through the callee to that output's own producer instead of to the opaque `jit_call`. The
        // program squares its known input `x` inside the callee and multiplies the result of the call by its unknown
        // input `y` outside of it, so the residual program needs the result of the call.
        let scalar_type = ArrayIrType::from(ArrayType::scalar(DataType::F64));
        let callee = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let input = builder.add_input(scalar_type.clone());
            let squared =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![squared], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let callee = builder.import_program(callee);
        let x = builder.add_input(scalar_type.clone());
        let y = builder.add_input(scalar_type);
        let called = builder
            .add_instruction(XlaOperation::JitCall(JitCallOperation::new(0)), vec![callee], vec![x], None)
            .unwrap()[0];
        let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![called, y], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let names = Arc::new(Mutex::new(Vec::new()));
        let recorded = names.clone();
        let policy = RematerializationPolicyFn::new::<ArrayIrType>(move |candidate| {
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
    fn test_jit_call_supports_composite_region_boundaries() {
        let dimension_type =
            ArrayIrType::Dimension(DimensionType::new("size", DimensionBounds::positive(Some(9)).unwrap()));
        let array_type = ArrayIrType::Array(vector_type());
        let interface = RegionInterface::new(
            vec![dimension_type.clone()],
            vec![array_type.clone(), dimension_type.clone()],
            EffectClasses::NONE,
        );
        let operation = JitCallOperation::new(0);

        assert_eq!(operation.name(), JIT_CALL_OPERATION_NAME);
        assert_eq!(
            operation.infer_output_types(&[], &[]),
            Err(TypeError::invalid(format!("{} expects 1 attached callee region but got 0", JIT_CALL_OPERATION_NAME,))),
        );
        assert_eq!(
            operation
                .infer_output_types(std::slice::from_ref(&dimension_type), std::slice::from_ref(&interface))
                .unwrap(),
            vec![array_type, dimension_type],
        );
    }

    #[test]
    fn test_jit_call_operation_batch() {
        use ryft_core::{
            ArrayIrBatch, ArrayIrBatchingPolicy, BatchableOperation, BatchingContext, CalleeRegionDriver,
            RecursiveBatchingDriver,
        };

        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let callee = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let callees = [Arc::new(callee)];
        let regions = CalleeRegionDriver::new(&callees);
        let driver = RecursiveBatchingDriver::new(&regions);
        let vector = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3]));
        let (_, program) = TracingContext::<XlaConstant, XlaOperation>::trace(
            |inputs: Vec<Tracer<TracingContext<XlaConstant, XlaOperation>>>| {
                let parent = inputs[0].context();
                let extent = parent.constant(XlaConstant::Dimension(DimensionValue::constant(3).unwrap()));
                let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent.clone(), extent);
                let (outputs, _) = JitCallOperation::<ArrayIrType>::new(0)
                    .batch(&context, &driver, &[ArrayIrBatch::new(inputs[0].clone(), Some(0))?])?
                    .into_parts();
                Ok::<_, ProgramError>(outputs.into_iter().map(ArrayIrBatch::into_value).collect::<Vec<_>>())
            },
            vec![vector.clone()],
        )
        .unwrap();
        assert_eq!(program.input_types(), vec![vector.clone()]);
        assert_eq!(program.output_types(), vec![vector]);
        let XlaOperation::JitCall(call) = program.instructions()[0].operation() else {
            panic!("expected jit_call");
        };
        assert_eq!(call.capture_count(), 1);
        let body = program.region_ref(program.instructions()[0].regions()[0]).unwrap();
        assert_eq!(body.input_types().len(), 2);
        assert_eq!(body.output_ids()[0], body.input_ids()[0]);
    }

    #[test]
    fn test_jit_call_jvp_omits_zero_space_boundary_tangents() {
        let dimension_type =
            ArrayIrType::Dimension(DimensionType::new("size", DimensionBounds::positive(Some(9)).unwrap()));
        let array_type = ArrayIrType::Array(ArrayType::scalar(DataType::F64));

        let mut callee_builder = CompositeXlaProgramBuilder::new();
        let dimension = callee_builder.add_input(dimension_type.clone());
        let array = callee_builder.add_input(array_type.clone());
        let callee = callee_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![dimension, array],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut builder = CompositeXlaProgramBuilder::new();
        let dimension = builder.add_input(dimension_type.clone());
        let array = builder.add_input(array_type.clone());
        let callee = builder.import_region(callee.entry_region_ref());
        let outputs = builder
            .add_instruction(
                XlaOperation::JitCall(JitCallOperation::new(0)),
                vec![callee],
                vec![dimension, array],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_types(), vec![dimension_type.clone(), array_type.clone(), array_type.clone()]);
        assert_eq!(jvp.output_types(), vec![dimension_type, array_type.clone(), array_type]);
    }

    #[test]
    fn test_jit_call_jvp_keeps_capture_prefix_inactive() {
        // The callee's leading input is its lifted capture (`capture_count = 1`). The call's activity mask starts
        // inactive over that prefix, numeric captures included, so the tangent callee consumes only the live operand
        // tangent followed by the residuals, and the primal callee keeps the capture prefix verbatim.
        let r#type = ArrayIrType::Array(vector_type());
        let callee = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let captured = builder.add_input(r#type.clone());
            let input = builder.add_input(r#type.clone());
            let product =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![captured, input], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![product], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let captured = builder.add_input(r#type.clone());
        let input = builder.add_input(r#type);
        let output = builder
            .add_instruction(XlaOperation::JitCall(JitCallOperation::new(1)), vec![callee], vec![captured, input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let jvp = program.jvp().unwrap();
        let calls = jvp
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), XlaOperation::JitCall(_)))
            .collect::<Vec<_>>();
        assert_eq!(calls.len(), 2);
        let primal_callee = jvp.region_ref(calls[0].regions()[0]).unwrap();
        assert!(matches!(calls[0].operation(), XlaOperation::JitCall(operation) if operation.capture_count() == 1));
        assert_eq!(primal_callee.input_types().len(), 2);
        let residual_count = primal_callee.output_types().len() - 1;
        assert!(residual_count >= 1);
        let tangent_callee = jvp.region_ref(calls[1].regions()[0]).unwrap();
        assert!(matches!(calls[1].operation(), XlaOperation::JitCall(operation) if operation.capture_count() == 0));
        assert_eq!(tangent_callee.input_types().len(), 1 + residual_count);
        assert_eq!(calls[1].inputs().len(), 1 + residual_count);
    }

    #[test]
    fn test_jit_call_jvp_threads_plumbing_and_active_references() {
        // The callee reads a captured plumbing reference and mutates an active reference operand:
        // `f(captured, r, x) = { r += x; read(r) * read(captured) }` behind `jit_call[capture_count=1]`.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let callee = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let captured = builder.add_input(reference_type.clone());
            let reference = builder.add_input(reference_type.clone());
            let update = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(
                    XlaOperation::ReferenceAddUpdate(ReferenceAddUpdateOperation::new()),
                    Vec::new(),
                    vec![reference, update],
                    None,
                )
                .unwrap();
            let state = builder
                .add_instruction(
                    XlaOperation::ReferenceRead(ReferenceReadOperation::new()),
                    Vec::new(),
                    vec![reference],
                    None,
                )
                .unwrap()[0];
            let scale = builder
                .add_instruction(
                    XlaOperation::ReferenceRead(ReferenceReadOperation::new()),
                    Vec::new(),
                    vec![captured],
                    None,
                )
                .unwrap()[0];
            let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![state, scale], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let captured = builder.add_input(reference_type.clone());
        let reference = builder.add_input(reference_type.clone());
        let update = builder.add_input(scalar_type.clone());
        let output = builder
            .add_instruction(
                XlaOperation::JitCall(JitCallOperation::new(1)),
                vec![callee],
                vec![captured, reference, update],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        // The plumbing capture is inactive and the active reference receives a tangent reference input, so the fused
        // program consumes `[captured, r, x, ṫr, ẋ]`: the primal call keeps the capture prefix verbatim, while the
        // tangent call consumes the tangent reference and the tangent value followed by the residuals.
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
        let calls = jvp
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), XlaOperation::JitCall(_)))
            .collect::<Vec<_>>();
        assert_eq!(calls.len(), 2);
        assert!(matches!(calls[0].operation(), XlaOperation::JitCall(operation) if operation.capture_count() == 1));
        assert_eq!(&calls[0].inputs()[..3], &jvp.input_ids()[..3]);
        let primal_callee = jvp.region_ref(calls[0].regions()[0]).unwrap();
        assert_eq!(
            primal_callee.input_types(),
            vec![reference_type.clone(), reference_type.clone(), scalar_type.clone()]
        );
        let residual_count = primal_callee.output_types().len() - 1;
        assert!(matches!(calls[1].operation(), XlaOperation::JitCall(operation) if operation.capture_count() == 0));
        assert_eq!(&calls[1].inputs()[..2], &jvp.input_ids()[3..]);
        assert_eq!(calls[1].inputs().len(), 2 + residual_count);
        let tangent_callee = jvp.region_ref(calls[1].regions()[0]).unwrap();
        assert_eq!(&tangent_callee.input_types()[..2], &[reference_type, scalar_type]);
    }

    #[test]
    fn test_jit_call_jvp_omits_inactive_reference_output_tangents() {
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let mut callee_builder = CompositeXlaProgramBuilder::new();
        let reference = callee_builder.add_input(reference_type.clone());
        let value = callee_builder.add_input(scalar_type.clone());
        let doubled =
            callee_builder.add_instruction(AddOperation::new(), Vec::new(), vec![value, value], None).unwrap()[0];
        let callee = callee_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![reference, doubled],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut builder = CompositeXlaProgramBuilder::new();
        let reference = builder.add_input(reference_type.clone());
        let value = builder.add_input(scalar_type.clone());
        let callee = builder.import_region(callee.entry_region_ref());
        let outputs = builder
            .add_instruction(
                XlaOperation::JitCall(JitCallOperation::new(1)),
                vec![callee],
                vec![reference, value],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let jvp = program.entry_region_ref().jvp(&[1]).unwrap();
        assert_eq!(jvp.input_types(), vec![reference_type.clone(), scalar_type.clone(), scalar_type.clone()]);
        assert_eq!(jvp.output_types(), vec![reference_type, scalar_type.clone(), scalar_type]);
        let calls = jvp
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), XlaOperation::JitCall(_)))
            .collect::<Vec<_>>();
        assert_eq!(calls.len(), 2);
        assert_eq!(jvp.region_ref(calls[1].regions()[0]).unwrap().output_types().len(), 1);
    }

    #[test]
    fn test_jit_call_jvp_treats_structurally_zero_tangents_as_inactive() {
        // `f(x, y) = (sin(x), cos(y))` differentiated with respect to `x` only: the structurally zero tangent of `y`
        // gets no tangent callee slot, and the tangent of `cos(y)`, which then depends on no tangent input, is a
        // structural zero that the tangent call does not return.
        let r#type = ArrayIrType::Array(vector_type());
        let callee = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let x = builder.add_input(r#type.clone());
            let y = builder.add_input(r#type.clone());
            let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
            let cosine = builder.add_instruction(CosOperation::new(), Vec::new(), vec![y], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![sine, cosine],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let inputs = (0..2).map(|_| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let outputs = builder
            .add_instruction(XlaOperation::JitCall(JitCallOperation::new(0)), vec![callee], inputs, None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.entry_region_ref().jvp(&[0]).unwrap().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4], %2:f64[4] .
                let %3:f64[4], %4:f64[4], %5:f64[4] = jit_call %0 %1 [
                    callee={
                        lambda %0:f64[4], %1:f64[4] .
                        let %2:f64[4] = sin %0
                            %3:f64[4] = cos %1
                            %4:f64[4] = cos %0
                        in (%2, %3, %4)
                    },
                ]
                    %6:f64[4] = jit_call %2 %5 [
                        callee={
                            lambda %0:f64[4], %1:f64[4] .
                            let %2:f64[4] = mul %1 %0
                            in (%2)
                        },
                    ]
                    %7:f64[4] = zero [type=f64[4]]
                in (%3, %4, %6, %7)"},
        );
    }

    #[test]
    fn test_jit_call_jvp_omits_independent_output_tangents() {
        // `f(x) = (sin(x), 0)`: the tangent of the constant output depends on no tangent input, so it is a structural
        // zero that the tangent call does not return.
        let r#type = ArrayIrType::Array(vector_type());
        let callee = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let x = builder.add_input(r#type.clone());
            let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
            let zero =
                builder.add_instruction(ZeroOperation::new(vector_type()), Vec::new(), Vec::new(), None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sine, zero], vec![Placeholder], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let input = builder.add_input(r#type);
        let outputs = builder
            .add_instruction(XlaOperation::JitCall(JitCallOperation::new(0)), vec![callee], vec![input], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4] .
                let %2:f64[4], %3:f64[4], %4:f64[4] = jit_call %0 [
                    callee={
                        lambda %0:f64[4] .
                        let %1:f64[4] = sin %0
                            %2:f64[4] = zero [type=f64[4]]
                            %3:f64[4] = cos %0
                        in (%1, %2, %3)
                    },
                ]
                    %5:f64[4] = jit_call %1 %4 [
                        callee={
                            lambda %0:f64[4], %1:f64[4] .
                            let %2:f64[4] = mul %1 %0
                            in (%2)
                        },
                    ]
                    %6:f64[4] = zero [type=f64[4]]
                in (%2, %3, %5, %6)"},
        );
    }

    #[test]
    fn test_jit_call_jvp_preserves_live_outputs_around_inactive_tangent() {
        // `f(x) = (sin(x), 0, x)` has an inactive middle tangent slot. Projecting that slot must leave both live
        // tangents aligned with their original primal outputs rather than pairing the final tangent with the zero.
        let r#type = ArrayIrType::Array(vector_type());
        let callee = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let input = builder.add_input(r#type.clone());
            let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
            let zero =
                builder.add_instruction(ZeroOperation::new(vector_type()), Vec::new(), Vec::new(), None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![sine, zero, input],
                    vec![Placeholder],
                    vec![Placeholder; 3],
                )
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let input = builder.add_input(r#type.clone());
        let outputs = builder
            .add_instruction(XlaOperation::JitCall(JitCallOperation::new(0)), vec![callee], vec![input], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder], vec![Placeholder; 3])
            .unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.output_types(), vec![r#type; 6]);
        let calls = jvp
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), XlaOperation::JitCall(_)))
            .collect::<Vec<_>>();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].outputs().len(), 2);
        assert_eq!(jvp.output_ids()[3], calls[1].outputs()[0]);
        assert_eq!(jvp.output_ids()[5], calls[1].outputs()[1]);
        assert!(jvp.instructions().iter().any(|instruction| {
            instruction.outputs().contains(&jvp.output_ids()[4]) && instruction.operation().is_zero(0)
        }));
    }

    #[test]
    fn test_jit_call_jvp_skips_tangent_calls_without_live_output_tangents() {
        // `f(x, y) = sin(x)` differentiated with respect to `y` only: no output tangent depends on the tangent of `y`
        // and the tangent callee has no effects, so only the source call is bound and the output tangent is a
        // structural zero.
        let r#type = ArrayIrType::Array(vector_type());
        let callee = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let x = builder.add_input(r#type.clone());
            builder.add_input(r#type.clone());
            let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sine], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let inputs = (0..2).map(|_| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let output = builder
            .add_instruction(XlaOperation::JitCall(JitCallOperation::new(0)), vec![callee], inputs, None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
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
    fn test_jit_call_partial_evaluation_keeps_reference_bearing_callees_whole() {
        // The callee reads the reference before and after mutating it: `f(r, x) = { a = read(r); r += x; a * read(r) }`
        // Splitting it against a known `r` would hoist the first read into the outer trace and leave the mutation and
        // the second read behind the residual call, so the call stays whole instead.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let callee = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let reference = builder.add_input(reference_type.clone());
            let update = builder.add_input(scalar_type.clone());
            let before = builder
                .add_instruction(
                    XlaOperation::ReferenceRead(ReferenceReadOperation::new()),
                    Vec::new(),
                    vec![reference],
                    None,
                )
                .unwrap()[0];
            builder
                .add_instruction(
                    XlaOperation::ReferenceAddUpdate(ReferenceAddUpdateOperation::new()),
                    Vec::new(),
                    vec![reference, update],
                    None,
                )
                .unwrap();
            let after = builder
                .add_instruction(
                    XlaOperation::ReferenceRead(ReferenceReadOperation::new()),
                    Vec::new(),
                    vec![reference],
                    None,
                )
                .unwrap()[0];
            let output =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![before, after], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let reference = builder.add_input(reference_type.clone());
        let update = builder.add_input(scalar_type.clone());
        let output = builder
            .add_instruction(
                XlaOperation::JitCall(JitCallOperation::new(0)),
                vec![callee],
                vec![reference, update],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let outer = TracingContext::<XlaConstant, XlaOperation>::new();
        let known = outer.input(reference_type);
        let evaluation = program
            .partially_evaluate_in_context(&outer, &[PartialValue::Known(known), PartialValue::Unknown(scalar_type)])
            .unwrap();
        let residual = evaluation.program();
        assert_eq!(residual.instructions().len(), 1);
        assert!(matches!(residual.instructions()[0].operation(), XlaOperation::JitCall(_)));
        assert_eq!(residual.instructions()[0].inputs().len(), 2);
        let residual_names = residual
            .entry_region_ref()
            .instructions_in_closure()
            .map(|(_, instruction)| instruction.operation().name())
            .collect::<Vec<_>>();
        assert_eq!(residual_names, vec!["jit_call", "reference_read", "reference_add_update", "reference_read", "mul"]);
        assert!(outer.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_jit_call_partial_evaluation_keeps_escaping_allocation_callees_whole() {
        // The callee allocates a reference from the known input and returns it beside a runtime-dependent product:
        // `f(k, x) = (reference_new(k), k * x)`. The allocation is computable from known inputs alone, but it escapes
        // through a reference output whose later accesses the callee cannot see, so it is live state rather than dead
        // intrinsic state, and the call stays whole instead of hoisting the allocation into the known side.
        let r#type = ArrayIrType::Array(vector_type());
        let callee = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let known_input = builder.add_input(r#type.clone());
            let runtime_input = builder.add_input(r#type.clone());
            let reference =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![known_input], None).unwrap()[0];
            let product = builder
                .add_instruction(MulOperation::new(), Vec::new(), vec![known_input, runtime_input], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![reference, product],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let known_input = builder.add_input(r#type.clone());
        let runtime_input = builder.add_input(r#type.clone());
        let callee_region = builder.intern_callee(&Arc::new(callee), None).unwrap();
        let outputs = builder
            .add_instruction(
                XlaOperation::JitCall(JitCallOperation::new(0)),
                vec![callee_region],
                vec![known_input, runtime_input],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let outer = TracingContext::<XlaConstant, XlaOperation>::new();
        let known = outer.input(r#type.clone());
        let evaluation = program
            .partially_evaluate_in_context(&outer, &[PartialValue::Known(known), PartialValue::Unknown(r#type)])
            .unwrap();
        let residual = evaluation.program();
        assert_eq!(residual.instructions().len(), 1);
        assert!(matches!(residual.instructions()[0].operation(), XlaOperation::JitCall(_)));
        let residual_names = residual
            .entry_region_ref()
            .instructions_in_closure()
            .map(|(_, instruction)| instruction.operation().name())
            .collect::<Vec<_>>();
        assert_eq!(residual_names, vec!["jit_call", "reference_new", "mul"]);
        assert!(outer.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_jit_call_transposition_ignores_dead_reference_operands() {
        // The linear callee only stores into its reference operand and forwards `x`: `ẏ = { ṫr += ẋ; ẋ }`. Under an
        // `Ignore` destination for the reference no later instruction accumulated into its root and the callee never
        // reads it, so the callee is transposed with an `Ignore` destination as well: the transposed call consumes
        // `[ȳ]` only, returns `x̄` only, and neither it nor its callee stages a cotangent reference.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let callee = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let reference = builder.add_input(reference_type.clone());
            let update = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(
                    XlaOperation::ReferenceAddUpdate(ReferenceAddUpdateOperation::new()),
                    Vec::new(),
                    vec![reference, update],
                    None,
                )
                .unwrap();
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![update], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let callee_region = builder.import_region(callee.entry_region_ref());
        let reference = builder.add_input(reference_type.clone());
        let update = builder.add_input(scalar_type.clone());
        let output = builder
            .add_instruction(
                XlaOperation::JitCall(JitCallOperation::new(0)),
                vec![callee_region],
                vec![reference, update],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let destination_kinds = [CotangentDestinationKind::Ignore, CotangentDestinationKind::Return];
        let transposed = program.transpose_with_respect_to(&[0, 1], &destination_kinds).unwrap();
        assert_eq!(transposed.input_types(), vec![scalar_type.clone()]);
        assert_eq!(transposed.output_types(), vec![scalar_type.clone()]);
        assert_eq!(transposed.instructions().len(), 1);
        let call = &transposed.instructions()[0];
        assert!(matches!(call.operation(), XlaOperation::JitCall(_)));
        assert_eq!(call.inputs(), transposed.input_ids());
        assert_eq!(call.outputs(), transposed.output_ids());
        let transposed_callee = transposed.region_ref(call.regions()[0]).unwrap().to_program();
        assert_eq!(transposed_callee.input_types(), vec![scalar_type.clone()]);
        assert_eq!(transposed_callee.output_types(), vec![scalar_type]);
        assert!(transposed_callee.instructions().is_empty());
        let transposed_names = transposed
            .entry_region_ref()
            .instructions_in_closure()
            .map(|(_, instruction)| instruction.operation().name())
            .collect::<Vec<_>>();
        assert_eq!(transposed_names, vec!["jit_call"]);
    }

    #[test]
    fn test_jit_call_transposition_threads_cotangent_references() {
        // The linear callee accumulates into its reference operand and reads it back: `ẏ = { ṫr += ẋ; read(ṫr) }`.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let callee = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let reference = builder.add_input(reference_type.clone());
            let update = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(
                    XlaOperation::ReferenceAddUpdate(ReferenceAddUpdateOperation::new()),
                    Vec::new(),
                    vec![reference, update],
                    None,
                )
                .unwrap();
            let output = builder
                .add_instruction(
                    XlaOperation::ReferenceRead(ReferenceReadOperation::new()),
                    Vec::new(),
                    vec![reference],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = CompositeXlaProgramBuilder::new();
        let callee_region = builder.import_region(callee.entry_region_ref());
        let reference = builder.add_input(reference_type.clone());
        let update = builder.add_input(scalar_type.clone());
        let output = builder
            .add_instruction(
                XlaOperation::JitCall(JitCallOperation::new(0)),
                vec![callee_region],
                vec![reference, update],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
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
        assert_eq!(transposed.instructions().len(), 1);
        let call = &transposed.instructions()[0];
        assert!(matches!(call.operation(), XlaOperation::JitCall(operation) if operation.capture_count() == 0));
        assert_eq!(call.inputs(), transposed.input_ids());
        assert_eq!(call.outputs().len(), 2);
        assert_eq!(transposed.atoms()[call.outputs()[0].index()].r#type().as_ref(), &reference_type);
        assert_eq!(transposed.output_ids(), &[transposed.input_ids()[1], call.outputs()[1]]);
        let transposed_callee = transposed.region_ref(call.regions()[0]).unwrap().to_program();
        let inlined = callee.transpose_with_respect_to(&[0, 1], &destination_kinds).unwrap();
        assert_eq!(transposed_callee.to_string(), inlined.to_string());
    }

    #[test]
    fn test_jit_call_zero_transpose_uses_cotangent_descriptors() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let tangent_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        let expected = ArrayIrType::Array(tangent_type.cotangent().unwrap());
        let tangent_type = ArrayIrType::Array(tangent_type);
        // Zero cotangents skip a pure callee only after its effects have been inspected. Supply the real
        // identity-region boundary rather than an empty driver that cannot answer that query.
        let mut source_builder = CompositeXlaProgramBuilder::new();
        let input = source_builder.add_input(tangent_type.clone());
        let source = source_builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut transpose_builder = CompositeXlaProgramBuilder::new();
        let input = transpose_builder.add_input(expected.clone());
        let transposed = transpose_builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let driver = TestTranspositionDriver { source, transposed };
        let mut context = TracingContext::<XlaConstant, XlaOperation>::new();
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
    fn test_jit_call_dynamic_zero_transpose_uses_known_extent_operand() {
        let extent_type = DimensionType::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let output_type =
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![extent_type.dimension().clone()])));
        let context = TracingContext::<XlaConstant, XlaOperation>::new();
        let extent = context.input(extent_type.into());

        let zero = materialize_transpose_cotangent(
            &context,
            &MaybeZero::Zero(output_type.clone()),
            &output_type,
            &[PartialValue::Known(extent)],
        )
        .unwrap();

        let builder = context.builder().borrow();
        let [instruction] = builder.instructions() else {
            panic!("expected one dynamic zero instruction");
        };
        assert!(matches!(instruction.operation(), XlaOperation::Zero(_)));
        assert_eq!(instruction.inputs(), &[ryft_core::AtomId::new(0)]);
        assert_eq!(zero.atom_id().unwrap(), instruction.outputs()[0]);
    }

    #[test]
    fn test_jit_call_mixed_output_transpose_materializes_zero_space_values() {
        let value_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let predicate_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(4)]));
        let value_program_type = ArrayIrType::Array(value_type.clone());
        let predicate_program_type = ArrayIrType::Array(predicate_type.clone());
        let source = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let value = builder.add_input(value_program_type.clone());
            let predicate =
                builder.add_constant(XlaConstant::Captured(CaptureReference::new(0, predicate_program_type.clone())));
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![value, predicate],
                    vec![Placeholder],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let transposed = {
            let mut builder = CompositeXlaProgramBuilder::new();
            let value_cotangent = builder.add_input(value_program_type.clone());
            let _predicate_cotangent = builder.add_input(ArrayIrType::Array(predicate_type.cotangent().unwrap()));
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![value_cotangent],
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let driver = TestTranspositionDriver { source, transposed };
        let mut context = TracingContext::<XlaConstant, XlaOperation>::new();
        let value_cotangent = context.input(value_program_type.clone());

        let contributions = transpose_primal_jit_call(
            &JitCallOperation::new(0),
            &mut context,
            &driver,
            &[PartialValue::Unknown(value_program_type.clone())],
            &[
                MaybeZero::Value(value_cotangent),
                MaybeZero::Zero(ArrayIrType::Array(predicate_type.cotangent().unwrap())),
            ],
            &CotangentDestinations::without_references([true]),
        )
        .unwrap();

        assert!(
            matches!(&contributions[..], [MaybeZero::Value(value)] if value.r#type().as_ref() == &value_program_type)
        );
    }

    /// Online partial evaluation of a mixed `jit_call` against a live outer trace — the second recorded consumer of
    /// parent-context-polymorphic partial evaluation. The known half of the callee (including a callee literal it
    /// consumes) is rewrapped as a known-side `jit_call` staged into the outer program over the symbolic known
    /// input; the unknown half stays behind a residual `jit_call` whose literal is rebuilt inline; and the
    /// known→unknown residual edge flows from the known-side call's outputs into the residual call's inputs.
    #[test]
    fn test_jit_call_online_partial_evaluation_splits_callee_against_a_live_outer_trace() {
        use ryft_core::{PartialEvaluationInput, PartialEvaluationOutput, StagingContext, TracingContext};

        let r#type = ArrayIrType::Array(vector_type());

        // Callee `f(a, x) = (a + a, x * x, (a + a) * x)` over a known `a` and an unknown `x`. Capture constants are
        // deliberately absent: a capture-bearing callee preserves its boundary instead of splitting, which
        // `test_jit_call_partial_evaluation_preserves_boundary_for_capture_bearing_callees` pins.
        let callee = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
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
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![shifted, scaled, product],
                    vec![Placeholder; 2],
                    vec![Placeholder; 3],
                )
                .unwrap()
        };

        // Enclosing program staging one call to the callee over `[a, x]`.
        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let known_input = builder.add_input(r#type.clone());
        let runtime_input = builder.add_input(r#type.clone());
        let callee_region = builder.intern_callee(&Arc::new(callee), None).unwrap();
        let call = XlaOperation::JitCall(JitCallOperation::new(0));
        let outputs = builder
            .add_instruction(call, vec![callee_region], vec![known_input, runtime_input], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 3])
            .unwrap();

        let outer = TracingContext::<XlaConstant, XlaOperation>::new();
        let known = outer.input(r#type.clone());
        let evaluation = program
            .partially_evaluate_in_context(&outer, &[PartialValue::Known(known), PartialValue::Unknown(r#type.clone())])
            .unwrap();

        // The known half landed in the outer program as one known-side `jit_call` over the symbolic known input,
        // producing the fully known callee output once: the residual edge, which is the same folded value, is fed from
        // that output.
        {
            let outer_builder = outer.builder().borrow();
            assert_eq!(outer_builder.instructions().len(), 1);
            let known_instruction = &outer_builder.instructions()[0];
            assert!(
                matches!(known_instruction.operation(), XlaOperation::JitCall(_)),
                "expected the outer program to contain the known-side jit_call",
            );
            let known_callee = outer_builder.region_ref(known_instruction.regions()[0]).unwrap().to_program();
            assert_eq!(known_callee.input_ids().len(), 1);
            assert_eq!(known_callee.output_ids().len(), 1);
            assert_eq!(known_callee.instructions().len(), 1);
            assert!(matches!(known_callee.instructions()[0].operation(), XlaOperation::Array(ArrayOperation::Add(_)),));
        }

        // The unknown half stayed behind one residual `jit_call` over the unknown input plus the residual edge.
        assert_eq!(evaluation.program().instructions().len(), 1);
        let residual_instruction = &evaluation.program().instructions()[0];
        assert!(
            matches!(residual_instruction.operation(), XlaOperation::JitCall(_)),
            "expected the residual program to contain the residual jit_call",
        );
        let residual_callee = evaluation.program().region_ref(residual_instruction.regions()[0]).unwrap().to_program();
        assert_eq!(residual_callee.input_ids().len(), 2);
        assert_eq!(residual_callee.instructions().len(), 2);

        // The boundary descriptors: the unknown enclosing input feeds the residual call, the residual edge is a
        // known feeder naming the known-side call's staged output, and the outputs reassemble in original order.
        assert_eq!(evaluation.inputs().len(), 2);
        assert!(matches!(&evaluation.inputs()[0], PartialEvaluationInput::Unknown(1)));
        assert!(matches!(&evaluation.inputs()[1], PartialEvaluationInput::Known(value) if value.atom_id().is_ok()));
        assert_eq!(evaluation.outputs().len(), 3);
        assert!(matches!(&evaluation.outputs()[0], PartialEvaluationOutput::Known(value) if value.atom_id().is_ok()));
        assert!(matches!(&evaluation.outputs()[1], PartialEvaluationOutput::Unknown(0)));
        assert!(matches!(&evaluation.outputs()[2], PartialEvaluationOutput::Unknown(1)));
    }

    #[test]
    fn test_jit_call_partial_evaluation_splits_concrete_known_inputs() {
        use ryft_core::{Device, DeviceMesh, PartialEvaluationInput, PartialEvaluationOutput};

        use crate::tests::{execution_client, values_from_bytes, values_to_bytes};
        use crate::{FromPjrt, XlaSession, XlaValue};

        // Partial evaluation splits a mixed `jit_call` whose known inputs are concrete arrays of the eager XLA domain
        // exactly as it splits one whose known inputs are symbolic: the known-side call compiles and executes
        // immediately, so the output `a + a` is known and concrete, and the residual call keeps only `(a + a) * x`,
        // fed by that known output (JAX's `out_fwd`).
        let client = execution_client();
        let device = Device::from_pjrt(client.addressable_devices().unwrap().remove(0)).unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![device]).unwrap();
        let domain = XlaSession::new(&client).domain();
        let array_type = vector_type().with_sharding(Sharding::replicated(logical_mesh, 1)).unwrap();
        let r#type = ArrayIrType::Array(array_type.clone());

        // Callee `f(a, x) = (a + a, (a + a) * x)` over a known `a` and an unknown `x`.
        let callee = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let a = builder.add_input(r#type.clone());
            let x = builder.add_input(r#type.clone());
            let doubled = builder.add_instruction(AddOperation::new(), Vec::new(), vec![a, a], None).unwrap()[0];
            let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![doubled, x], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![doubled, product],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let a = builder.add_input(r#type.clone());
        let x = builder.add_input(r#type.clone());
        let callee_region = builder.intern_callee(&Arc::new(callee), None).unwrap();
        let outputs = builder
            .add_instruction(XlaOperation::JitCall(JitCallOperation::new(0)), vec![callee_region], vec![a, x], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let array = |values: &[f64]| {
            let bytes = values_to_bytes::<f64>(values);
            XlaValue::Array(XlaArray::from_host_buffer(&domain, array_type.clone(), mesh.clone(), bytes).unwrap())
        };
        let read = |value: &XlaValue<'_>| {
            let XlaValue::Array(array) = value else {
                panic!("expected an array value");
            };
            let buffer = array.addressable_shards().next().unwrap().buffer().unwrap();
            values_from_bytes::<f64>(buffer.copy_to_host(None).unwrap().r#await().unwrap().as_slice())
        };
        let evaluation = program
            .partially_evaluate_in_context(
                &domain,
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
        let outputs = evaluation.interpret(&domain, &[array(&[5.0, 6.0, 7.0, 8.0])]).unwrap();
        assert_eq!(
            outputs.iter().map(read).collect::<Vec<_>>(),
            vec![vec![2.0, 4.0, 6.0, 8.0], vec![10.0, 24.0, 42.0, 64.0]],
        );
    }

    #[test]
    fn test_jit_call_partial_evaluation_forwards_known_output_residuals() {
        // The callee `f(a, x) = (sin(a), sin(a) * x)` partitioned with `a` known and `x` unknown: the residual call
        // needs `sin(a)`, which is also a known output, so the known call returns it once and the residual call is fed
        // from that output (JAX's `out_fwd`). The outer partition repeats the known call's output as its own residual
        // edge, because `Program::partition` keeps one edge per residual feeder.
        let r#type = ArrayIrType::Array(vector_type());
        let callee = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let a = builder.add_input(r#type.clone());
            let x = builder.add_input(r#type.clone());
            let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![a], None).unwrap()[0];
            let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![sine, x], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![sine, product],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let a = builder.add_input(r#type.clone());
        let x = builder.add_input(r#type);
        let callee_region = builder.intern_callee(&Arc::new(callee), None).unwrap();
        let outputs = builder
            .add_instruction(XlaOperation::JitCall(JitCallOperation::new(0)), vec![callee_region], vec![a, x], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
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
        let r#type = ArrayIrType::Array(vector_type());
        let callee = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let known_input = builder.add_input(r#type.clone());
            let runtime_input = builder.add_input(r#type.clone());
            let captured = builder.add_constant(XlaConstant::Captured(CaptureReference::new(0, r#type.clone())));
            let shifted =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![known_input, captured], None).unwrap()[0];
            let product = builder
                .add_instruction(MulOperation::new(), Vec::new(), vec![shifted, runtime_input], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![shifted, product],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let known_input = builder.add_input(r#type.clone());
        let runtime_input = builder.add_input(r#type.clone());
        let callee_region = builder.intern_callee(&Arc::new(callee), None).unwrap();
        let outputs = builder
            .add_instruction(
                XlaOperation::JitCall(JitCallOperation::new(0)),
                vec![callee_region],
                vec![known_input, runtime_input],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let outer = TracingContext::<XlaConstant, XlaOperation>::new();
        let known = outer.input(r#type.clone());
        let evaluation = program
            .partially_evaluate_in_context(&outer, &[PartialValue::Known(known), PartialValue::Unknown(r#type)])
            .unwrap();

        // No known-side call was hoisted into the outer trace, and the residual program retains the original call
        // with its callee — capture constant included — intact.
        assert!(outer.builder().borrow().instructions().is_empty());
        assert_eq!(evaluation.program().instructions().len(), 1);
        let residual_instruction = &evaluation.program().instructions()[0];
        assert!(matches!(residual_instruction.operation(), XlaOperation::JitCall(_)));
        let residual_callee = evaluation.program().region_ref(residual_instruction.regions()[0]).unwrap().to_program();
        assert_eq!(residual_callee.instructions().len(), 2);
        assert!(
            residual_callee
                .atoms()
                .iter()
                .any(|atom| atom.as_constant().is_some_and(|constant| constant.capture_index().is_some())),
        );
    }

    #[test]
    fn test_jit_call_partial_evaluation_partitions_a_callee_with_a_dead_allocation() {
        let r#type = ArrayIrType::Array(vector_type());
        let callee = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
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
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![doubled, product],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };

        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let known_input = builder.add_input(r#type.clone());
        let runtime_input = builder.add_input(r#type.clone());
        let callee_region = builder.intern_callee(&Arc::new(callee), None).unwrap();
        let outputs = builder
            .add_instruction(
                XlaOperation::JitCall(JitCallOperation::new(0)),
                vec![callee_region],
                vec![known_input, runtime_input],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        // The callee partitions like any other: its pure prefix over the known input folds into the known side
        // (staged into the outer trace) while the runtime-dependent product stays in the residual callee, and the
        // allocation, which nothing observes, is a dead intrinsic-state operation that neither side retains.
        let outer = TracingContext::<XlaConstant, XlaOperation>::new();
        let known = outer.input(r#type.clone());
        let evaluation = program
            .partially_evaluate_in_context(&outer, &[PartialValue::Known(known), PartialValue::Unknown(r#type)])
            .unwrap();
        let residual_names = evaluation
            .program()
            .entry_region_ref()
            .instructions_in_closure()
            .map(|(_, instruction)| instruction.operation().name())
            .collect::<Vec<_>>();
        assert!(residual_names.contains(&"mul"), "{residual_names:?}");
        assert!(!residual_names.contains(&"add"), "{residual_names:?}");
        assert!(!residual_names.contains(&"reference_new"), "{residual_names:?}");
        let known_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        let known_names = known_program
            .entry_region_ref()
            .instructions_in_closure()
            .map(|(_, instruction)| instruction.operation().name())
            .collect::<Vec<_>>();
        assert!(known_names.contains(&"add"), "{known_names:?}");
        assert!(!known_names.contains(&"reference_new"), "{known_names:?}");
    }

    #[test]
    fn test_jit_call_partial_evaluation_places_callee_residuals_with_the_context_policy() {
        struct SaveDots;

        impl ResidualPolicy<ArrayIrType> for SaveDots {
            type Storage = NoStorage;

            fn name(&self) -> &str {
                "save_dots"
            }

            fn classify(
                &self,
                candidate: &ResidualCandidate<'_, ArrayIrType>,
            ) -> Result<ResidualDecision<NoStorage>, ResidualRejection> {
                Ok(match candidate.producers().iter().any(|producer| producer.payload::<DotOperation>().is_some()) {
                    true => ResidualDecision::Save,
                    false => ResidualDecision::Recompute,
                })
            }
        }

        // The callee computes `cos(dot(x, x)) * t`. Partitioning the caller with `x` known splits the call, and the
        // policy then plans the callee partition: the known call returns the dot product instead of its cosine, and
        // the residual call recomputes the cosine.
        let callee = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let x = builder.add_input(ArrayIrType::Array(vector_type()));
            let t = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
            let dot = ArrayOperation::<XlaArrayConstant>::from(DotOperation::new(DotDimensionNumbers::new(
                vec![0],
                vec![0],
                vec![],
                vec![],
            )));
            let product = builder.add_instruction(dot, Vec::new(), vec![x, x], None).unwrap()[0];
            let cosine = ArrayOperation::<XlaArrayConstant>::from(CosOperation::<ArrayType>::new());
            let cosine = builder.add_instruction(cosine, Vec::new(), vec![product], None).unwrap()[0];
            let tangent = ArrayOperation::<XlaArrayConstant>::from(MulOperation::<ArrayType>::new());
            let tangent = builder.add_instruction(tangent, Vec::new(), vec![cosine, t], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![tangent], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let x = builder.add_input(ArrayIrType::Array(vector_type()));
        let t = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
        let callee_region = builder.intern_callee(&Arc::new(callee), None).unwrap();
        let output = builder
            .add_instruction(XlaOperation::JitCall(JitCallOperation::new(0)), vec![callee_region], vec![x, t], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
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
        use ryft_core::{BroadcastOperation, Linearization, SaveOnlyTheseNames};

        use crate::tests::execution_client;

        // Only the expanded activation is needed by the tangent. Its all-known call must expose the compact tagged
        // scale to the policy, saving 4 KiB instead of 256 KiB and replaying only the broadcast.
        let compact_type = ArrayType::new_static(DataType::F32, [1024]);
        let expanded_type = ArrayType::new_static(DataType::F32, [64, 1024]);
        let mut builder = CompositeXlaProgramBuilder::new();
        let weights = builder.add_input(compact_type.clone().into());
        let scale = builder.add_instruction(SinOperation::new(), Vec::new(), vec![weights], None).unwrap()[0];
        let scale = builder
            .add_instruction(TagOperation::<ArrayType>::new("scale"), Vec::new(), vec![scale], None)
            .unwrap()[0];
        let expanded = builder
            .add_instruction(BroadcastOperation::new(expanded_type.clone(), vec![1]), Vec::new(), vec![scale], None)
            .unwrap()[0];
        let callee = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![expanded], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(expanded_type.clone().into());
        let weights = builder.add_input(compact_type.clone().into());
        let callee = builder.import_program(callee);
        let scale = builder
            .add_instruction(XlaOperation::JitCall(JitCallOperation::new(0)), vec![callee], vec![weights], None)
            .unwrap()[0];
        let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, scale], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // Ordinary partial evaluation preserves the original compiled-call boundary and expanded residual.
        let ordinary = program.partition(&[false, true]).unwrap();
        assert_eq!(ordinary.known_program().output_types(), vec![ArrayIrType::from(expanded_type.clone())]);
        assert!(matches!(ordinary.known_program().instructions()[0].operation(), XlaOperation::JitCall(_)));
        assert_eq!(expanded_type.element_count().unwrap().unwrap() * size_of::<f32>(), 262_144);

        let policy = ResidualPolicyReference::new(SaveOnlyTheseNames::new(["scale"]));
        let placed = program.partition_with_residual_policy(&[false, true], &policy).unwrap();
        assert_eq!(placed.known_program().output_types(), vec![ArrayIrType::from(compact_type.clone())]);
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
                    let %1:f32[1024] = sin %0
                        %2:f32[1024] = tag [key=scale] %1
                    in (%2)
                }
                residual={
                    lambda %0:f32[64, 1024], %1:f32[1024] .
                    let %2:f32[64, 1024] = broadcast [output_type=f32[64, 1024], output_axes=[1]] %1
                        %3:f32[64, 1024] = mul %0 %2
                    in (%3)
                }"},
        );

        // The same cut survives the rematerialization wrapper's linearization, where the original output remains a
        // known primal result and the expanded scale is needed only by the derivative with respect to the activation.
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(expanded_type.clone().into());
        let weights = builder.add_input(compact_type.clone().into());
        let plain_linearization = program.linearize_with_respect_to(&[0]).unwrap();
        let body = builder.import_program(program);
        let output = builder
            .add_instruction(RematerializeOperation::new(policy), vec![body], vec![input, weights], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let linearization = program.linearize_with_respect_to(&[0]).unwrap();
        assert_eq!(linearization.residual_count(), 1);
        assert_eq!(
            linearization.primal().output_types(),
            vec![ArrayIrType::from(expanded_type.clone()), ArrayIrType::from(compact_type.clone())],
        );
        assert_eq!(
            format!("{}\n{}", linearization.primal(), linearization.tangent()),
            indoc! {"
                lambda %0:f32[64, 1024], %1:f32[1024] .
                let %2:f32[1024] = sin %1
                    %3:f32[1024] = tag [key=scale] %2
                    %4:f32[64, 1024] = broadcast [output_type=f32[64, 1024], output_axes=[1]] %3
                    %5:f32[64, 1024] = mul %0 %4
                in (%5, %3)
                lambda %0:f32[64, 1024], %1:f32[1024] .
                let %2:f32[64, 1024] = rematerialize [policy=\"save_only_these_names\", differentiated=true] %0 %1 [
                    body={
                        lambda %0:f32[64, 1024], %1:f32[1024] .
                        let %2:f32[64, 1024] = broadcast [output_type=f32[64, 1024], output_axes=[1]] %1
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
        let evaluate = |linearization: &Linearization<XlaConstant, XlaOperation>| {
            let mut primal = execute_f32_program(&client, linearization.primal(), inputs.clone());
            let value = primal.remove(0);
            let tangent_inputs = std::iter::once(direction.clone()).chain(primal.clone()).collect();
            let tangent = execute_f32_program(&client, linearization.tangent(), tangent_inputs);
            let pullback_inputs = std::iter::once(cotangent.clone()).chain(primal).collect();
            let pullback = execute_f32_program(&client, &linearization.pullback().unwrap(), pullback_inputs);
            vec![value, tangent[0].clone(), pullback[0].clone()]
        };
        let actual = evaluate(&linearization);
        assert_eq!(actual, evaluate(&plain_linearization));
        assert!(actual.iter().flatten().all(|value| value.is_finite() && *value != 0.0));
    }

    #[test]
    fn test_jit_call_partial_evaluation_ignores_known_outputs_without_residual_demand() {
        use ryft_core::{RematerializationPolicyFn, ResidualPolicyError};

        let scalar_type = ArrayIrType::from(ArrayType::scalar(DataType::F32));
        let mut builder = CompositeXlaProgramBuilder::new();
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
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![needed, unused], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        let mut builder = CompositeXlaProgramBuilder::new();
        let known = builder.add_input(scalar_type.clone());
        let unknown = builder.add_input(scalar_type.clone());
        let callee = builder.import_program(callee);
        let called = builder
            .add_instruction(XlaOperation::JitCall(JitCallOperation::new(0)), vec![callee], vec![known], None)
            .unwrap()
            .to_vec();
        let first =
            builder.add_instruction(MulOperation::new(), Vec::new(), vec![called[0], unknown], None).unwrap()[0];
        let second =
            builder.add_instruction(MulOperation::new(), Vec::new(), vec![called[1], unknown], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![first, second, called[1]],
                vec![Placeholder; 2],
                vec![Placeholder; 3],
            )
            .unwrap();
        let policy =
            ResidualPolicyReference::new(RematerializationPolicyFn::new::<ArrayIrType>(|candidate| {
                if candidate.producers().iter().any(|producer| {
                    producer.payload::<TagOperation<ArrayType>>().is_some_and(|tag| tag.key() == "unused")
                }) {
                    Err(ResidualRejection::new("the unused result must not cross the residual boundary"))
                } else {
                    Ok(ResidualDecision::<NoStorage>::Save)
                }
            }));

        // The second call output remains a primal result. Its presence must not classify it as a residual.
        let primal_only = program.with_outputs(&[0, 2]).unwrap();
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
                    let %1:f32[] = cos %0
                        %2:f32[] = tag [key=unused] %1
                        %3:f32[] = sin %0
                        %4:f32[] = tag [key=needed] %3
                    in (%2, %4)
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
        use ryft_core::{NothingSavable, PartialEvaluationContext};

        let policy = ResidualPolicyReference::new(NothingSavable);
        let check_boundary = |callee: XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>, capture_count: usize| {
            let mut builder = CompositeXlaProgramBuilder::new();
            let inputs = callee.input_types().into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
            let input_count = inputs.len();
            let callee = builder.import_program(callee);
            let outputs = builder
                .add_instruction(
                    XlaOperation::JitCall(JitCallOperation::new(capture_count)),
                    vec![callee],
                    inputs,
                    None,
                )
                .unwrap()
                .to_vec();
            let output_count = outputs.len();
            let program = builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
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
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(dynamic_type.into());
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        check_boundary(
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap(),
            0,
        );

        // An ordered assertion must keep the same owning call and execution order.
        let predicate_type = ArrayIrType::from(ArrayType::scalar(DataType::Boolean));
        let mut builder = CompositeXlaProgramBuilder::new();
        let predicate = builder.add_input(predicate_type.clone());
        builder
            .add_instruction(AssertOperation::<ArrayType>::new("required predicate"), Vec::new(), vec![predicate], None)
            .unwrap();
        check_boundary(
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![predicate], vec![Placeholder], vec![Placeholder])
                .unwrap(),
            0,
        );

        // Reference handles and retained capture constants keep their existing boundary semantics too.
        let mut builder = CompositeXlaProgramBuilder::new();
        let reference = builder.add_input(ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        check_boundary(
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap(),
            0,
        );
        let mut builder = CompositeXlaProgramBuilder::new();
        builder.add_input(predicate_type.clone());
        let captured = builder.add_constant(XlaConstant::Captured(CaptureReference::new(0, predicate_type.clone())));
        check_boundary(
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![captured], vec![Placeholder], vec![Placeholder])
                .unwrap(),
            1,
        );

        // A known value without a staged parent atom must not trigger per-operation replay. The same guard keeps
        // concrete eager calls whole; using a literal in a live trace makes the retained call directly observable.
        let mut builder = CompositeXlaProgramBuilder::new();
        let input = builder.add_input(predicate_type);
        let callee = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let outer = TracingContext::<XlaConstant, XlaOperation>::new();
        let context = PartialEvaluationContext::new(outer.clone()).with_residual_policy(&policy);
        let input = context.lift(XlaConstant::Boolean(true)).unwrap();
        context.bind(XlaOperation::JitCall(JitCallOperation::new(0)), vec![callee], &[input]).unwrap();
        assert_eq!(outer.builder().borrow().instructions().len(), 1);
        assert!(matches!(outer.builder().borrow().instructions()[0].operation(), XlaOperation::JitCall(_)));
    }

    #[test]
    fn test_nested_jit_call_reference_discharge_threads_callee_state_into_the_caller() {
        // Two nested call levels each swap a distinct reference. Discharge must lift both states to the caller's
        // boundary in public-input order, mark both as mutated with their own final-state output slot, and leave no
        // reference-typed atom anywhere in the resulting program.
        let reference_type = ReferenceType::new(vector_type());
        let inner_callee = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            let replacement = builder.add_input(vector_type().into());
            let snapshot = builder
                .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let callee = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let inner_callee = builder.import_region(inner_callee.entry_region_ref());
            let first_reference = builder.add_input(reference_type.clone().into());
            let first_replacement = builder.add_input(vector_type().into());
            let second_reference = builder.add_input(reference_type.clone().into());
            let second_replacement = builder.add_input(vector_type().into());
            let first_snapshot = builder
                .add_instruction(
                    XlaOperation::JitCall(JitCallOperation::new(0)),
                    vec![inner_callee],
                    vec![first_reference, first_replacement],
                    None,
                )
                .unwrap()[0];
            let second_snapshot = builder
                .add_instruction(
                    XlaOperation::JitCall(JitCallOperation::new(0)),
                    vec![inner_callee],
                    vec![second_reference, second_replacement],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![first_snapshot, second_snapshot],
                    vec![Placeholder; 4],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };

        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let first_reference = builder.add_input(reference_type.clone().into());
        let first_replacement = builder.add_input(vector_type().into());
        let second_reference = builder.add_input(reference_type.into());
        let second_replacement = builder.add_input(vector_type().into());
        let snapshots = builder
            .add_instruction(
                XlaOperation::JitCall(JitCallOperation::new(0)),
                vec![callee],
                vec![first_reference, first_replacement, second_reference, second_replacement],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(snapshots, vec![Placeholder; 4], vec![Placeholder; 2])
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
    fn test_nested_jit_call_reference_discharge_resolves_callee_lexical_capture_scope() {
        // A branch region inside the callee reaches its reference through a capture constant, which must resolve
        // against the callee's own capture scope rather than the entry program's lifted prefix: the accessed root is
        // the caller's second public input, and only that mutated state gains a final-state output.
        let reference_type = ReferenceType::new(vector_type());
        let branch = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let replacement = builder.add_input(vector_type().into());
            let reference = builder.add_constant(XlaConstant::Captured(CaptureReference::new(
                0,
                ArrayIrType::Reference(reference_type.clone()),
            )));
            let snapshot = builder
                .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let callee = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let branch = builder.import_region(branch.entry_region_ref());
            builder.add_input(reference_type.clone().into());
            let replacement = builder.add_input(vector_type().into());
            let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
            let snapshot = builder
                .add_instruction(ConditionOperation::new(), vec![branch, branch], vec![predicate, replacement], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![snapshot], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };

        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        builder.add_input(reference_type.clone().into());
        let reference = builder.add_input(reference_type.into());
        let replacement = builder.add_input(vector_type().into());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let snapshot = builder
            .add_instruction(
                XlaOperation::JitCall(JitCallOperation::new(1)),
                vec![callee],
                vec![reference, replacement, predicate],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![snapshot], vec![Placeholder; 4], vec![Placeholder])
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
        let mut callee_builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let reference = callee_builder.add_input(reference_type.clone().into());
        let callee = callee_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        let forwarded = builder
            .add_instruction(XlaOperation::JitCall(JitCallOperation::new(0)), vec![callee], vec![reference], None)
            .unwrap()[0];
        let value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![forwarded], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![value], vec![Placeholder], vec![Placeholder])
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
        // the shared positional rewrite threads state through it: the callee gains the entering state after its
        // declared inputs and publishes the final state after its declared outputs, while the call site keeps its own
        // operand and output contract. The callee mutates the root and returns only the previous snapshot, so the
        // final state is exactly what the call site would otherwise have lost.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut callee_builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let callee_reference = callee_builder.add_input(reference_type.into());
        let callee_update = callee_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let previous = callee_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![callee_reference, callee_update], None)
            .unwrap()[0];
        let callee = callee_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![previous], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let callee = builder.import_program(callee);
        let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let update = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let root = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let previous = builder
            .add_instruction(XlaOperation::JitCall(JitCallOperation::new(0)), vec![callee], vec![root, update], None)
            .unwrap()[0];
        let frozen = builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![root], None).unwrap()[0];
        let source = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![previous, frozen],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        // The callee's local root is allocated, mutated, and consumed inside the call, so widening the boundary
        // leaves no external state behind and the rewritten call is the entry region's only instruction.
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.external_reference_bindings(), &[]);
        assert_eq!(discharged.program().output_types().len(), 2);
        assert_eq!(discharged.program().entry_region_ref().instructions().len(), 1);
    }
}
