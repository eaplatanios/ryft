//! Lowering and eager application of [`CallOperation`], the staged call of a flat program, whose payload, boundary,
//! type rule, and transform rules `ryft-core` owns (refer to the documentation of [`CallOperation`] for the boundary
//! of a call and how program transforms keep it). A call whose callee names no devices runs where
//! [Where Compiled Functions Run](crate::compilation::functions#where-compiled-functions-run) states, which is that an
//! application by a session reads the mesh of its function (refer to [`CallOperation::function_mesh`]) and that a
//! nested call follows its caller. Each entry point of this crate honors the boundary of a call as follows:
//!
//!   - Lowering constrains the inputs and the outputs that have overrides where the call executes, as an execution of
//!     the function annotates its inputs and outputs, and the module declares the mesh of these constraints (refer to
//!     `ProgramLoweringContext::require_mesh`). Compiling a caller for other devices than those of
//!     a call that names devices, or for the replicas of a bounded dynamic program while an override splits a value,
//!     fails with an exact error (refer to `validate_placements`), as JAX rejects a nested `jit` with an incompatible
//!     device assignment. Lowering emits each callee inline or, when it occurs more than once, as a call of one shared
//!     private `func.func` (refer to `plan_call_functions`).
//!   - Eager dispatch and eager interpretation take one placement decision for each application (refer to
//!     `CallEagerExtension::eager_decision`), which admits the inputs as an execution does (their client, their
//!     committed devices, and their reduction state), selects the devices of the call, moves captures and uncommitted
//!     inputs to them, applies the input and output sharding overrides, and states whether the inputs commit the
//!     outputs and whether this process alone chose their devices. Every other input is placed as the executable of the
//!     callee places it (refer to [`CallOperation::boundary_input_sharding`]), which is as the callee input type
//!     records for a function compiled for its input types (replicated for a type without a sharding), and as the input
//!     is for a call of a `jit` dispatcher, whose executables are specialized for the placements of their inputs.
//!   - A call commits its outputs when it names devices, when its callee requests any placement (including one that
//!     only moves a value to another memory), runs on multiple devices, has a committed public input, or is applied by
//!     a session on the named mesh of its function (refer to `commits_public_outputs`).

use std::borrow::Cow;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use ryft_core::{
    AtomId, AttachedRegions, CALL_OPERATION_NAME, CallOperation, CompositeKey, Device, DeviceMesh, EagerContext,
    FlatProgram, InterpretableOperation, InterpretationDriver, Operation as CoreOperation, Parameterized, Program,
    ProgramError, RegionId, RegionPlacementRequests, RegionRef, RegionRole, ShardMapOperation, Sharding, Typed,
    TypedProgram,
};
use ryft_mlir::dialects::func;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Type as MlirType,
    TypeAndAttributes, Value as MlirValue, ValueAndAttributes, ValueRef,
};

use crate::compilation::domains::{commits_public_outputs, relocated_input_type};
use crate::execution::{validate_committed_input_devices, validate_input_clients, validate_input_reduction};
use crate::lowering::EffectTokens;
use crate::lowering::contexts::{PrivateFunctionBody, ProgramLoweringContext};
use crate::lowering::programs::{
    CallTraversal, ProgramInlineLoweringExtension, ProgramSessionConstants, RegionLoweringExtension,
    RegionPlacementRequestsExtension, UnreadLiftedCaptures,
};
use crate::lowering::types::{ArrayTypeLoweringExtension, TypeLoweringExtension};
use crate::operations::control_flow::lowered_effect_classes;
use crate::operations::manipulation::conversions::TypeCastLoweringExtension;
use crate::operations::sharding::constrain_sharding::lower_declared_sharding_constraint;
use crate::operations::{Operation, cached_unread_lifted_captures};
use crate::{
    Array, Constant, Error, ExecutionError, LowerableOperation, Lowerer, LoweringError, ProgramLoweringOperation,
    Session, Type, Value,
};

// TODO(eaplatanios): Review this module.

/// Structural identity of a flat callee program attached behind a `call`, used to deduplicate repeated nested
/// programs into shared private `func.func`s at lowering time.
///
/// Eligible programs (refer to [`supports_structural_deduplication`]) are keyed by their canonical rendering plus their
/// complete flat input/output signature. Type inference is deterministic and attached regions (i.e., control-flow
/// bodies, composite programs, and nested `call` callees) render contextually inside their instructions, so two
/// programs that render identically with equal boundary types compute the same function and may share one emitted
/// function. This holds even when they are distinct staged programs produced by separate transform passes (e.g., the
/// per-block primal and pullback programs of `grad(jit(f))` over repeated blocks).
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct CallProgramKey {
    /// Canonical [`Program`](ryft_core::Program) rendering (i.e., operation names, all bracketed attributes, and
    /// attached region bodies).
    rendered: String,

    /// Flat input types, which together with the rendering pin the full callee signature.
    input_types: Vec<Type>,

    /// Flat output types, completing the callee signature when output-only placement metadata is not visible
    /// in the rendered body.
    output_types: Vec<Type>,

    /// Payload of every call in the callee and in the regions that it reaches, in the order of a depth-first walk of
    /// its instructions, compared by the equality of [`CallOperation`]. Each of these calls lowers its boundary
    /// inside the shared function, so everything that equality compares (e.g., a mesh, an override, a kept input
    /// placement, or a derived input boundary) must distinguish callees, including what the rendering of a call does
    /// not show (refer to [`CallOperation`]).
    calls: Vec<CallOperation<Type>>,

    /// Number of leading inputs of the callee that are its lifted captures, which its captured constants refer to by
    /// position, or zero when the callee has no captured constant. The rendering shows the position of a captured
    /// constant, and this count states which inputs those positions name. Calls that lift different numbers of
    /// captures therefore still share the function of a callee that reads none of them through a captured constant.
    capture_count: usize,

    /// For each session constant of the callee, in the order of its regions and of their atoms, the position of the
    /// first constant that refers to the same array. The rendering shows only the type of a session constant, and a
    /// shared function receives the arrays as inputs (refer to [`lower_shared_call`]), so callees that read different
    /// arrays share a function as long as they repeat their arrays in the same pattern.
    session_constants: Vec<usize>,
}

/// Plans the shared private functions of the `call`s of `program`, which is the program that a module is lowered from,
/// by registering a function with `lowering_context` (refer to [`ProgramLoweringContext::require_function`]) for
/// every `call` callee that occurs at least twice (per [`CallProgramKey`] identity), and for every named composite
/// body that no backend implementation replaces on the target platform and that occurs at least twice.
/// Single-occurrence callees are left to inline, so modules without repeated calls lower without private functions.
/// At each `call` lowering site, a callee whose function is registered is emitted as a `func.call` of that function
/// instead of being inlined (refer to [`lower_shared_call`]).
///
/// A function that named composite calls reach is named after the name and version of the first such definition (e.g.,
/// `ryft.example.v1`, with a numeric suffix for other specializations), even when a `call` also reaches it, and the
/// other functions are named `callee_N`, where `N` counts the functions that were planned before them. The calls of a
/// named composite also find their function by their key and boundary (refer to [`NamedFunctionKey`]), but only when
/// that boundary equals the boundary of the function, so that the emitted call signature matches the function.
///
/// # Errors
///
/// Returns an error when the effects that lowering a callee introduces cannot be determined (refer to
/// [`lowered_effect_classes`]).
pub(crate) fn plan_call_functions<Input, Output, O: ProgramLoweringOperation>(
    program: &Program<Constant, O, Input, Output>,
    lowering_context: &ProgramLoweringContext,
) -> Result<(), Error>
where
    Input: Parameterized<Constant>,
    Output: Parameterized<Constant>,
{
    let target_platform = lowering_context.target_platform();
    let mut counts = HashMap::new();
    let mut order: Vec<CallProgramKey> = Vec::new();
    let mut memo: HashMap<(RegionId, usize), Option<CallProgramKey>> = HashMap::new();
    let mut named = HashMap::new();
    let mut names = HashMap::new();
    count_calls(program, target_platform, &mut counts, &mut order, &mut memo, &mut named, &mut names)?;

    let mut function_count = 0;
    let mut boundaries = HashMap::<CallProgramKey, (Vec<Type>, Vec<Type>)>::new();
    for key in order {
        let (count, region, program) = counts.remove(&key).unwrap();
        // Effectful callees always inline (even when repeated) so their effectful instructions chain onto the
        // caller's effect token in program order, which a shared token-free function could not preserve. That
        // includes the entry guards that lowering synthesizes for the callee's nested control flow.
        if count < 2 || !lowered_effect_classes(&program)?.is_empty() {
            continue;
        }
        let name = match names.get(&key) {
            Some(name) => Cow::Borrowed(name.as_str()),
            None => Cow::Owned(format!("callee_{}", function_count)),
        };
        let body = PrivateFunctionBody::Region { region, capture_count: key.capture_count };
        lowering_context.require_function(key.clone(), &name, || Ok(body))?;
        function_count += 1;

        // A function that receives session constants is found by the body of each of its calls, which states the
        // arrays that the call passes, and never by a key and boundary alone.
        if key.session_constants.is_empty() {
            boundaries.insert(key, (program.input_types().into_owned(), program.output_types().into_owned()));
        }
    }
    for (named_key, key) in named {
        let Some((key, (input_types, output_types))) =
            key.and_then(|key| boundaries.get(&key).map(|boundary| (key, boundary)))
        else {
            continue;
        };
        if *input_types == named_key.input_types && *output_types == named_key.output_types {
            lowering_context.alias_function(named_key, &key);
        }
    }
    Ok(())
}

/// Lowers a call of `callee`, whose first `capture_count` inputs are its lifted captures, to a `func.call` of the
/// shared private function that the module of `lowering_context` holds for it, and returns the results of that call.
/// Returns [`None`] without emitting anything when the module plans no function for the identity of `callee` (refer
/// to [`plan_call_functions`]), when the scope of `lowering_context` inlines its calls, or when `input_values` are
/// not the inputs of `callee`.
///
/// The function takes the inputs of `callee` followed by the session constants that `callee` reads, so the call
/// passes `input_values` followed by the values that the calling scope, whose captured values are
/// `caller_captured_values`, binds for those constants.
///
/// # Errors
///
/// Returns [`LoweringError::UnboundSessionConstant`] when the calling scope binds no value for a session constant
/// that `callee` reads.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_shared_call<'b, 'c: 'b, 't: 'c, O: ProgramLoweringOperation>(
    callee: &FlatProgram<EagerContext<Constant, O>>,
    capture_count: usize,
    input_values: &[ValueRef<'b, 'c, 't>],
    caller_captured_values: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
    lowering_context: &ProgramLoweringContext,
) -> Result<Option<Vec<ValueRef<'b, 'c, 't>>>, Error> {
    // The `call` operation's type inference already pins its inputs to the callee input types, so a matching arity
    // is the only guard needed before emitting the symbol call. Anything else inlines.
    if input_values.len() != callee.input_count() {
        return Ok(None);
    }
    let function = call_program_key(callee, capture_count).and_then(|key| lowering_context.shared_function(&key));
    let Some(function) = function else {
        return Ok(None);
    };
    let target_platform = lowering_context.target_platform();
    let regions = callee.entry_region_ref().emitted_regions(target_platform, CallTraversal::Enter);
    let session_constants = ProgramSessionConstants::from_regions(regions, target_platform);
    let mut arguments = input_values.to_vec();
    for constant in session_constants.inputs() {
        arguments.push(lowering_context.lower_session_constant(constant, caller_captured_values)?);
    }
    let output_types = callee.output_types();
    lower_shared_function_call(function.symbol(), &output_types, &arguments, block, context, location).map(Some)
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
    let result_tensor_types =
        output_types.iter().map(|r#type| r#type.lower(context, location)).collect::<Result<Vec<_>, _>>()?;
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

/// Placement decision of one eager application of a call (i.e., an eager call of a compiled function, which
/// `Session::bind` applies), which [`CallEagerExtension::eager_decision`] makes once from the call, its callee, and the
/// actual inputs. It states the facts that the types of the inputs omit, and the application consumes it instead of
/// deriving any of them again, so that it agrees with an execution of the callee. Partial evaluation, which sees types
/// only, forwards an input through a call only where this decision returns it unchanged whatever the inputs are.
pub(crate) struct EagerCallDecision<'o> {
    /// Inputs as the application receives them, which are admitted (i.e., they belong to the client of the session,
    /// their committed public inputs agree on their devices, and none carries a pending reduction that its boundary
    /// cannot receive) and placed on [`Self::mesh`] as an execution of the callee places them.
    inputs: Cow<'o, [Value]>,

    /// Devices on which the application runs, borrowed from the input that selected them when one did. An eager call
    /// compiles its callee for these devices.
    mesh: Cow<'o, DeviceMesh>,

    /// Whether every output of the application is committed, which is when a public input is committed or when the
    /// callee commits its public outputs on [`Self::mesh`] (i.e., when the call requests a placement, runs on more than
    /// one device, or runs on the named mesh of its function, refer to [`commits_public_outputs`]), as the executable
    /// of the callee commits them.
    committed: bool,

    /// Whether this process alone chose [`Self::mesh`] (refer to `Session::assigns_placement_locally`), which every
    /// output of the application states (refer to [`Array::is_locally_assigned`]).
    locally_assigned: bool,

    /// For each input, whether it is a capture that an executable of the callee has no physical input for (refer to
    /// `unread_lifted_captures`), or [`None`] when the callee has no such capture. The application neither admits,
    /// moves, nor requires the buffers of such a capture, as an execution of the callee does not, so [`Self::inputs`]
    /// holds it as it arrived.
    unread_captures: Option<Arc<[bool]>>,
}

impl<'o> EagerCallDecision<'o> {
    /// Returns the inputs as the application receives them (refer to [`EagerCallDecision`]).
    #[inline]
    pub(crate) fn inputs(&self) -> &[Value] {
        &self.inputs
    }

    /// Returns the devices on which the application runs.
    #[inline]
    pub(crate) fn mesh(&self) -> &DeviceMesh {
        &self.mesh
    }

    /// Returns whether every output of the application is committed.
    #[inline]
    pub(crate) fn committed(&self) -> bool {
        self.committed
    }

    /// Returns whether this process alone chose the devices of the application.
    #[inline]
    pub(crate) fn locally_assigned(&self) -> bool {
        self.locally_assigned
    }

    /// Returns, for each input, whether it is a capture that an executable of the callee has no physical input for, or
    /// [`None`] when the callee has no such capture (refer to [`EagerCallDecision`]).
    #[inline]
    pub(crate) fn unread_captures(&self) -> Option<&[bool]> {
        self.unread_captures.as_deref()
    }
}

/// Returns the array of input `index` of an eager application of a call, or [`None`] when the input is not an array or
/// is a capture that `unread_captures` flags (refer to [`EagerCallDecision::unread_captures`]), of which the
/// application requires nothing.
fn read_array<'v>(unread_captures: Option<&[bool]>, index: usize, input: &'v Value) -> Option<&'v Array> {
    match input {
        Value::Array(array) if !unread_captures.is_some_and(|unread| unread.get(index) == Some(&true)) => Some(array),
        _ => None,
    }
}

/// Eager application of a [`CallOperation`] by a [`Session`], which `Session::bind` applies to every call that a
/// session binds, and which the interpretation of a call on a session reaches through the bind channel of its driver.
/// The application takes one placement decision (refer to [`Self::eager_decision`]) and executes the compiled callee
/// with the inputs as that decision places them, so that an eager call agrees with an execution of the function that it
/// calls. Refer to the documentation of this module for how the application honors the boundary of a call.
pub(crate) trait CallEagerExtension {
    /// Returns the placement decision of an eager application of this call of `callee` on `inputs`, executed by
    /// `session` (refer to [`EagerCallDecision`]). This is the one place that decides where an eager call runs, which
    /// inputs it admits, where they move, whether its outputs are committed, and whether its devices were chosen by
    /// this process alone.
    ///
    ///   - Every input, captures included, must belong to the client of `session`, which an execution requires too
    ///     (refer to `validate_input_clients`).
    ///   - A call whose callee names devices runs on them. Its committed public inputs must already be on them, and
    ///     every input is placed there as its effective boundary sharding requests (refer to
    ///     [`CallOperation::boundary_input_sharding`]), or keeps its sharding when the mesh topologies agree and is
    ///     replicated otherwise. Its devices are named for every process, so they are not locally assigned.
    ///   - A call runs on the mesh of its function (refer to [`CallOperation::with_function_mesh`]) in the same way
    ///     when that mesh is named (refer to [`LogicalMesh::is_named`](ryft_core::LogicalMesh::is_named)), as JAX runs
    ///     a `jit` function under `set_mesh`, and commits its outputs. On the anonymous mesh of one device, it does so
    ///     only when no public input is committed, as JAX runs a `jit` function under `default_device`, with
    ///     uncommitted outputs, and a committed public input decides as below. This decision is taken only for the call
    ///     that the session applies. A call nested in its callee, or in a region of another eager operation (e.g.,
    ///     `condition`), is compiled into the computation of that call or operation and runs where it runs, as a nested
    ///     call runs where its compiled caller runs.
    ///   - Any other call runs where its public inputs are, as JAX runs a `jit` function without shardings. Committed
    ///     public inputs must agree on their devices, as JAX rejects a `jit` call whose committed arguments are on
    ///     different devices ("Received incompatible devices for jitted computation"), and they select the devices.
    ///     Without one, uncommitted public inputs on the same devices keep them, and otherwise the session default
    ///     supplies them, which, for a call without public array inputs, is where the session places a value without
    ///     array inputs over the logical mesh of the outputs of the callee (refer to `Session::device_mesh_for`), as
    ///     for any other eager operation without array inputs. The lifted captures move there, as closed-over constants
    ///     do in JAX (refer to `capture_inputs`), and so do the uncommitted public inputs that live elsewhere, through
    ///     the one rule by which uncommitted inputs follow committed ones (refer to
    ///     `Session::place_uncommitted_inputs`). Each input with an effective boundary sharding is then placed as it
    ///     requests, as is an input on one device whose callee input declares another logical mesh of one device (refer
    ///     to [`CallOperation::eager_boundary_input_sharding`]).
    ///
    /// The inputs are admitted before anything moves, and whether or not anything moves: an input that carries a
    /// pending reduction that the executable of the callee cannot receive is rejected wherever it is (refer to
    /// `validate_input_reduction`), because a reshard never completes a pending reduction, so an eager call rejects
    /// what an execution rejects, whatever placement the decision chooses. The outputs are committed when a public
    /// input is committed or when the callee commits its public outputs on the selected devices (refer to
    /// [`CallOperation::has_placement_requests`] and [`commits_public_outputs`]), which includes every application that
    /// runs on the named mesh of its function, as JAX commits every output of a `jit` function under `set_mesh`, and
    /// excludes an application that runs where its inputs are, whatever logical mesh they record.
    ///
    /// # Errors
    ///
    /// Returns the error of admitting an input, of selecting the default devices, or of moving an input.
    fn eager_decision<'o>(
        &self,
        session: &Session,
        inputs: &'o [Value],
        callee: RegionRef<'_, Constant, Operation>,
    ) -> Result<EagerCallDecision<'o>, Error>;

    /// Returns the placement that the executable of an eager call must give callee input `index`, which receives
    /// `array` for a callee input of type `declared`. This is the effective boundary sharding of the input when the
    /// call has one (refer to [`CallOperation::boundary_input_sharding`]). Otherwise, for a call that a program
    /// transform created (refer to [`CallOperation::derived_input_shardings`]) or that keeps its input placements
    /// (refer to [`CallOperation::with_kept_input_placements`]), it is the runtime sharding of `array`, because such a
    /// call leaves that input where it is and its executable must not move it to the placement that its traced input
    /// type would give it. A call of a `jit` dispatcher keeps no relocatable input where it is (refer to
    /// `relocated_input_type`), because the executions of the dispatcher do not specialize on its placement either. It is
    /// [`None`] for any other input, whose placement the executable chooses by its own rule.
    fn eager_input_placement(&self, index: usize, array: &Array, declared: &Type) -> Option<Sharding>;
}

impl CallEagerExtension for CallOperation<Type> {
    fn eager_decision<'o>(
        &self,
        session: &Session,
        inputs: &'o [Value],
        callee: RegionRef<'_, Constant, Operation>,
    ) -> Result<EagerCallDecision<'o>, Error> {
        fn array(input: &Value) -> Option<&Array> {
            match input {
                Value::Array(array) => Some(array),
                Value::Dimension(_) | Value::Reference(_) => None,
            }
        }
        // A capture that the executable of the callee has no physical input for is neither admitted, moved, nor
        // required to hold buffers here, as an execution of the callee requires nothing of it. A callee without
        // captures pays one comparison, and one with captures a look-up.
        let target_platform = session.target()?.platform_name();
        let callee_requests = || RegionPlacementRequests::for_lowering(callee, Some(target_platform));
        let unread_captures = match self.capture_count() {
            0 => None,
            capture_count => cached_unread_lifted_captures(callee, capture_count, Some(target_platform)),
        };
        let unread = unread_captures.as_deref();
        let read_arrays = inputs.iter().enumerate().map(|(index, input)| read_array(unread, index, input));
        validate_input_clients(session, read_arrays, self.capture_count())?;
        let interface = callee.interface();
        let declared = interface.input_types();
        // Admission is decided here, once, from the inputs as they arrive and before any of them moves, whether or not
        // the application moves it, so that an eager call, which runs a program of its own rather than the executable
        // of the callee, still rejects what that executable rejects. A pending reduction that a type records is judged
        // by the type rule of the call first, as when the call is staged, and the common case of inputs without one
        // costs one branch for each input.
        let pending = |(index, input): (usize, &Value)| {
            read_array(unread, index, input).is_some_and(|array| !array.sharding().unreduced_axes().is_empty())
        };
        if inputs.iter().enumerate().any(pending) {
            let types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
            self.validate_input_types(declared, &types)?;
            for (index, (input, declared)) in inputs.iter().zip(declared.iter()).enumerate() {
                if let Some(array) = read_array(unread, index, input) {
                    validate_input_admission(self, index, array, declared)?;
                }
            }
        }
        let public_arrays = inputs.iter().skip(self.capture_count()).filter_map(array);
        // The mesh commits the outputs for being named only when the application runs on it because the function was
        // compiled for it, and not when it is where the inputs of the application are (refer to
        // `commits_public_outputs`).
        let committed = |mesh: &DeviceMesh, function_mesh: bool| {
            public_arrays.clone().any(Array::is_committed)
                || commits_public_outputs(mesh, function_mesh, self.has_placement_requests(callee_requests))
        };
        // The call records its mesh when it is staged, for no target, so whether it still names devices is read for the
        // platform of this session, as the pruning of a compiled caller reads it (refer to `Self::names_devices`).
        let names_devices = || callee_requests().has_requests(false);
        let function_assignment = || {
            let mesh = self.function_mesh()?;
            if mesh.logical_mesh().is_named() {
                Some((mesh, false))
            } else if !public_arrays.clone().any(Array::is_committed) {
                Some((mesh, session.uses_local_default_device(mesh)))
            } else {
                None
            }
        };
        let assignment = self.names_devices(names_devices).map(|mesh| (mesh, false)).or_else(function_assignment);
        if let Some((mesh, locally_assigned)) = assignment {
            for (index, input) in inputs.iter().skip(self.capture_count()).enumerate() {
                if let Value::Array(array) = input {
                    validate_committed_input_devices(array, index, mesh)?;
                }
            }
            // Check the values' types before an override changes their sharding, including any reduction state.
            let types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
            self.validate_input_types(declared, &types)?;
            let placed = inputs
                .iter()
                .enumerate()
                .map(|(index, input)| match read_array(unread, index, input) {
                    Some(array) => {
                        boundary_input(self, session, mesh, index, array.clone(), &declared[index]).map(Value::Array)
                    }
                    None => Ok(input.clone()),
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(EagerCallDecision {
                inputs: Cow::Owned(placed),
                mesh: Cow::Owned(mesh.clone()),
                committed: committed(mesh, true),
                locally_assigned,
                unread_captures,
            });
        }

        let mut committed_inputs =
            inputs.iter().skip(self.capture_count()).enumerate().filter_map(|(index, input)| match input {
                Value::Array(array) if array.is_committed() => Some((index, array)),
                _ => None,
            });
        let first_committed = committed_inputs.next();
        if let Some((first_index, first)) = first_committed {
            for (index, array) in committed_inputs {
                if array.mesh().devices() != first.mesh().devices() {
                    // Each device is rendered by its `Display` implementation, which names its process when it is not
                    // process zero.
                    let devices = |array: &Array| {
                        let devices = array.mesh().devices().iter().map(Device::to_string).collect::<Vec<_>>();
                        format!("[{}]", devices.join(", "))
                    };
                    return Err(ProgramError::InvalidArgument {
                        message: format!(
                            "public input #{index} is committed to devices {}, but public input #{first_index} \
                             is committed to devices {}",
                            devices(array),
                            devices(first),
                        ),
                    }
                    .into());
                }
            }
        }
        let mesh = match first_committed {
            Some((_, committed)) => Cow::Borrowed(committed.mesh()),
            None => match public_arrays.clone().next() {
                Some(first) if public_arrays.clone().all(|other| other.mesh().devices() == first.mesh().devices()) => {
                    Cow::Borrowed(first.mesh())
                }
                Some(_) => Cow::Owned(session.device_mesh_for(&[], None)?),
                None => {
                    let logical_mesh = interface.output_types().iter().find_map(|r#type| match r#type {
                        Type::Array(r#type) => r#type.sharding().map(Sharding::mesh),
                        _ => None,
                    });
                    Cow::Owned(session.device_mesh_for(&[], logical_mesh)?)
                }
            },
        };
        let mut prepared = capture_inputs(self, session, inputs, unread, &mesh)?;

        // Uncommitted public inputs that live elsewhere follow the committed ones, or move to the default devices,
        // exactly as the inputs of any other eager operation do, so that every input of the callee is on the devices
        // where it runs, including an input that the callee returns as it is. Arrays produced by the same program, or
        // cloned from one array, share their mesh allocation, so the common case of inputs on one mesh only compares
        // pointers, as the inputs of other eager operations do (refer to `Session::place_uncommitted_inputs`).
        let shares_mesh = |mesh: &DeviceMesh, other: &DeviceMesh| std::ptr::eq(mesh, other) || mesh == other;
        let read_arrays = || prepared.iter().enumerate().filter_map(|(index, input)| read_array(unread, index, input));
        let mut prepared_arrays = read_arrays();
        let first_mesh = prepared_arrays.next().map(Array::mesh);
        let moves = match first_mesh {
            Some(first_mesh) if !prepared_arrays.all(|other| shares_mesh(other.mesh(), first_mesh)) => {
                let arrays = read_arrays().cloned().collect::<Vec<_>>();
                let moved = session.place_uncommitted_inputs(&arrays, &[], None)?;
                let moves = moved.is_some();
                let arrays = match moved {
                    Some(moved) => {
                        let mut replacements = moved.iter();
                        for (index, input) in prepared.to_mut().iter_mut().enumerate() {
                            if read_array(unread, index, input).is_some()
                                && let Value::Array(array) = input
                            {
                                *array = replacements.next().expect("one replacement for each array input").clone();
                            }
                        }
                        moved
                    }
                    None => arrays,
                };
                session.validate_input_placement(&arrays, &[])?;
                moves
            }
            _ => false,
        };
        let public_arrays = prepared.iter().skip(self.capture_count()).filter_map(array);
        let locally_assigned = session.assigns_placement_locally(public_arrays, &[], false, moves);

        for (index, declared) in declared.iter().enumerate().skip(self.capture_count()) {
            let Some(Value::Array(array)) = prepared.get(index) else { continue };
            if let Some(sharding) = self.eager_boundary_input_sharding(index, &array.r#type(), declared)
                && array.sharding() != sharding.as_ref()
            {
                let mesh = DeviceMesh::new(sharding.mesh().clone(), array.mesh().devices().to_vec())?;
                let placed = array.clone().reshard_on_devices(session, &mesh, &sharding, false)?;
                prepared.to_mut()[index] = Value::Array(placed);
            }
        }
        let committed = committed(&mesh, false);
        Ok(EagerCallDecision { inputs: prepared, mesh, committed, locally_assigned, unread_captures })
    }

    fn eager_input_placement(&self, index: usize, array: &Array, declared: &Type) -> Option<Sharding> {
        if let Some(sharding) = self.boundary_input_sharding(index, &array.r#type(), declared) {
            return Some(sharding.into_owned());
        }
        let keeps_placement = self.keeps_input_placements() && relocated_input_type(array, &array.r#type()).is_none();
        (self.derived_input_shardings().is_some() || keeps_placement).then(|| array.sharding().clone())
    }
}

/// Validates that `array`, the input at `index` of an eager application of `call` for a callee input of type
/// `declared`, carries no pending reduction that the executable of the callee cannot receive, by the rule with which an
/// execution admits its inputs (refer to [`validate_input_reduction`]). That executable receives the input at its eager
/// placement (refer to [`CallEagerExtension::eager_input_placement`]), or else as `declared` places it (refer to
/// [`CallOperation::declared_input_sharding`]). An input override of the callee is the input sharding of that
/// executable as stated, including a pending reduction over auto mesh axes that it states, which the executable
/// completes where the input enters, as a direct execution of the callee does. Any other placement is received without
/// the pending reductions over auto mesh axes, which no type records (refer to [`Sharding::without_auto_axes`]), so
/// such an input is rejected. An input without a pending reduction, which is the common case, costs one branch.
///
/// # Errors
///
/// Returns [`ExecutionError::UnreducedAutoAxisInput`] or [`ExecutionError::UnreducedAutoAxisCapture`] for such an
/// input.
fn validate_input_admission(
    call: &CallOperation<Type>,
    index: usize,
    array: &Array,
    declared: &Type,
) -> Result<(), ExecutionError> {
    if array.sharding().unreduced_axes().is_empty() {
        return Ok(());
    }
    let expected = match (call.eager_input_placement(index, array, declared), declared) {
        (Some(placement), _) if call.input_sharding(index).is_some() => placement,
        (Some(placement), _) => placement.without_auto_axes(),
        (None, Type::Array(declared)) => CallOperation::declared_input_sharding(declared, array.sharding().mesh()),
        (None, Type::Dimension(_) | Type::Reference(_)) => return Ok(()),
    };
    validate_input_reduction(array, &expected, index, call.capture_count())
}

/// Places an input of `call` on the boundary's devices after all committed public inputs have passed the device check,
/// as its effective boundary sharding requests (refer to [`CallOperation::boundary_input_sharding`]). An input without
/// one keeps its sharding when the mesh topologies agree and is replicated when they differ. The input keeps its
/// commitment.
///
/// # Errors
///
/// Returns the error of moving the input (e.g., for an input of another client).
fn boundary_input(
    call: &CallOperation<Type>,
    session: &Session,
    mesh: &DeviceMesh,
    index: usize,
    array: Array,
    declared: &Type,
) -> Result<Array, Error> {
    let sharding = match call.boundary_input_sharding(index, &array.r#type(), declared) {
        Some(sharding) => sharding.into_owned(),
        None if array.sharding().mesh().has_same_topology(mesh.logical_mesh()) => array.sharding().clone(),
        None => Sharding::replicated(mesh.logical_mesh().clone(), array.r#type().shape().rank()),
    };
    if array.mesh() == mesh && array.sharding() == &sharding {
        return Ok(array);
    }
    array.reshard_on_devices(session, mesh, &sharding, false)
}

/// Returns `inputs`, the inputs of `call` that `session` executes eagerly on `target` when its callee names no devices
/// (refer to [`CallEagerExtension::eager_decision`]), with the arrays of the lifted capture prefix moved to
/// `target`. A capture keeps its logical mesh and sharding when the device count matches, and is replicated otherwise.
/// The temporary handle contributes no commitment to the call. A capture that `unread_captures` flags stays as it is
/// (refer to [`EagerCallDecision::unread_captures`]).
///
/// # Errors
///
/// Returns the error of moving a capture.
fn capture_inputs<'o>(
    call: &CallOperation<Type>,
    session: &Session,
    inputs: &'o [Value],
    unread_captures: Option<&[bool]>,
    target: &DeviceMesh,
) -> Result<Cow<'o, [Value]>, Error> {
    if call.capture_count() == 0 || inputs.len() < call.capture_count() {
        return Ok(Cow::Borrowed(inputs));
    }
    let mut prepared = Cow::Borrowed(inputs);
    for (index, input) in inputs[..call.capture_count()].iter().enumerate() {
        let Some(array) = read_array(unread_captures, index, input) else { continue };
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
    // Eager dispatch compiles the callee on the logical mesh of its first array input, which is a capture here. A
    // replicated public input on the devices of the captures is given that logical mesh too, which moves no data, so
    // that the array inputs of the call name the logical mesh on which its callee runs.
    let callee_mesh = prepared[..call.capture_count()]
        .iter()
        .enumerate()
        .find_map(|(index, input)| read_array(unread_captures, index, input).map(|array| array.mesh().clone()));
    if let Some(mesh) = callee_mesh {
        for (index, input) in inputs.iter().enumerate().skip(call.capture_count()) {
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

// Eager interpretation of a staged `call` operation over concrete [`Array`](crate::Array)s. Every route that
// runs a call on a session reaches `Session::bind`, which compiles the callee whole through the eager dispatch cache of
// the session and executes it on its PJRT client, as JAX dispatches a `jit` function called from eager code straight to
// its executable. Interpreting a program on a session (e.g., the pullbacks of eager `vjp` and `grad`, partial
// evaluation with an eager known side, or the body of a composite whose inputs refine its traced types) binds each of
// its instructions, nested calls included, through `Context::bind`. This rule only runs when a caller applies the
// operation directly, and it hands the call to the bind channel of the driver, so that it runs exactly as a call that
// the session binds itself (i.e., with the same placement decision, compiled program, and outputs). The callee is
// copied into an owned program for that bind, because the driver only lends a view of it.
impl InterpretableOperation<Session> for CallOperation<Type> {
    fn interpret<D: InterpretationDriver<Session>>(
        &self,
        context: &Session,
        driver: &D,
        inputs: &[Value],
    ) -> Result<Vec<Value>, ProgramError> {
        let regions = AttachedRegions::Owned(driver.region_programs());
        driver.bind(context, Operation::Call(self.clone()), regions, inputs)
    }
}

// A call of a compiled function places the values that cross the boundary of its callee as the effective boundary
// shardings of that boundary request (i.e., its sharding overrides, refer to `CallOperation::with_boundary`, and the
// placements of the inputs without one, refer to `CallOperation::boundary_input_sharding`), as an execution of the
// function places its inputs and outputs. The overrides are applied at the call site, as constraints on the inputs and
// on the outputs of the call, rather than as annotations of the callee, because the callee may be inlined or shared
// with calls that carry other overrides. Each constraint is the sharding with which the function annotates its own
// boundary (refer to `Sharding::with_unconstrained_dimensions_replicated`), and it is closed over auto mesh axes, so
// that a replicated override stops the compiler from splitting the value, like the replicated output of an execution.
// An effective input sharding without an override that the input type already has is not constrained again, unless the
// mesh has auto axes, along which the input type cannot show where the input is (refer to
// `CallOperation::boundary_input_sharding_constraint`).
impl LowerableOperation for CallOperation<Type> {
    fn lower<'b, 'c: 'b, 't: 'c, O: ProgramLoweringOperation>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Type],
        output_types: &[Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        self.validate_region_count(lowerer.regions().len())?;
        // A capture that the callee never reads receives a bufferless value, just like an unread entry argument.
        // Both capture analysis and session constant partitioning use the same decision about those inputs.
        let unread = match self.capture_count() {
            0 => None,
            capture_count => lowerer.regions()[0].entry_region_ref().unread_lifted_captures(
                capture_count,
                lowerer.lowering_context().target_platform(),
                &mut UnreadLiftedCaptures::new(),
            ),
        };
        let input_values = input_values
            .iter()
            .zip(input_types)
            .enumerate()
            .map(|(index, (&value, ty))| {
                let Type::Array(actual) = ty else { return Ok(value) };
                if unread.as_deref().is_some_and(|unread| unread.get(index).copied().unwrap_or(false)) {
                    return actual.lower_i64_splat(0, &mut lowerer.block(), lowerer.location());
                }
                let callee = lowerer.regions()[0].entry_region_ref();
                let declared = &callee.interface().input_types()[index];
                // The callee receives the input as its entry function holds an argument that arrives at the boundary
                // sharding, with the reduction state over auto mesh axes that an override may state completed (refer to
                // `Sharding::without_auto_reduction_axes`). An input is a value of the caller, which never carries such
                // a state, because the entry of the outermost program completes it or admission rejects it, so the
                // constraint neither introduces nor completes a reduction over auto mesh axes, of any kind. Only the
                // outputs of a call can still introduce one (refer to `lower_boundary_constraint`).
                let Some(sharding) =
                    self.boundary_input_sharding_constraint(index, actual, declared).map(Cow::into_owned)
                else {
                    return Ok(value);
                };
                lower_boundary_constraint(lowerer, value, ty, &sharding.without_auto_reduction_axes())
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let output_values = {
            let (mut block, context, location) = (lowerer.block(), lowerer.context(), lowerer.location());
            let (effect_tokens, regions, captured_values, lowering_context) = lowerer.split_effect_tokens();
            let [callee] = regions else { unreachable!("validated call region count") };

            // An input whose type refines the callee input type (e.g., with a static extent for a bounded dynamic one)
            // enters the callee at the type of the callee input, and an output leaves it at the refined type of the
            // call (refer to `CallOperation::validate_input_types`). Neither conversion checks anything, because the
            // call has accepted its inputs and the callee computes its outputs from them. Only shapes are converted. A
            // difference in sharding alone (i.e., in placement, or in manual variation that a call in a `shard_map`
            // body passes on to its outputs) lowers to nothing, because the call accepts only inputs whose values the
            // callee input types describe.
            let callee_input_types = callee.input_types();
            let input_values = input_values
                .iter()
                .zip(input_types.iter().zip(callee_input_types.iter()))
                .map(|(&value, (r#type, declared))| match r#type == declared {
                    true => Ok(value),
                    false => r#type.lower_cast_shape(value, declared, &mut block, context, location),
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let output_values = callee.lower_call(
                self.capture_count(),
                &input_values,
                captured_values,
                &mut block,
                context,
                location,
                lowering_context,
                effect_tokens,
            )?;
            output_values
                .into_iter()
                .zip(callee.output_types().iter().zip(output_types))
                .map(|(value, (declared, r#type))| match declared == r#type {
                    true => Ok(value),
                    false => declared.lower_cast_shape(value, r#type, &mut block, context, location),
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

/// Semantic identity and boundary of the named composite calls that one shared function implements, by which those
/// calls find their function without rendering or hashing their body.
///
/// The lookup assumes that named composite calls with equal keys and equal boundaries carry equal bodies, which the
/// contract of [`NamedComposite`](ryft_core::NamedComposite) requires but cannot verify (and which does not hold
/// across definitions with equal keys under different named axes, or in different operation families).
/// [`plan_call_functions`] enforces it by registering no function under a key and boundary whose calls carry
/// different bodies, so those calls look up their shared function by their own body (refer to
/// [`lower_shared_call`]).
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct NamedFunctionKey {
    /// Semantic identity of the calls.
    key: CompositeKey,

    /// Input types of the calls.
    input_types: Vec<Type>,

    /// Output types of the calls.
    output_types: Vec<Type>,
}

impl NamedFunctionKey {
    /// Creates the key of the named composite calls of `key` at the boundary with the provided types.
    #[inline]
    pub(crate) fn new(key: CompositeKey, input_types: Vec<Type>, output_types: Vec<Type>) -> Self {
        Self { key, input_types, output_types }
    }
}

/// Named composite calls found by [`count_calls`], grouped by the interned body region, the key, and the boundary
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

/// Returns whether `program` may be deduplicated by structural identity.
///
/// A program is eligible only when its [`CallProgramKey`] captures the full semantics needed to safely share a private
/// function. That holds for programs with constants. The payloads of immediate constants are rendered. A
/// [`CaptureReference`](ryft_core::CaptureReference) renders its position among the lifted captures of the callee,
/// which are leading inputs of the shared function. A session constant is an input of the shared function as well
/// (refer to [`lower_shared_call`]). `shard_map` remains outside this deduplication path because its lowering owns a
/// separate outlined boundary.
fn supports_structural_deduplication<O: ProgramLoweringOperation>(
    program: &FlatProgram<EagerContext<Constant, O>>,
) -> bool {
    program.regions().iter().all(|region| {
        region
            .instructions()
            .iter()
            .all(|instruction| instruction.operation().projected_payload::<ShardMapOperation>().is_none())
    })
}

/// Returns whether a borrowed rooted region may be deduplicated by structural identity.
fn supports_structural_deduplication_of_region<O: ProgramLoweringOperation>(
    region: RegionRef<'_, Constant, O>,
) -> bool {
    fn walk<O: ProgramLoweringOperation>(region: RegionRef<'_, Constant, O>, visited: &mut HashSet<RegionId>) -> bool {
        if !visited.insert(region.id()) {
            return true;
        }
        region.instructions().iter().all(|instruction| {
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

/// Computes the deduplication key for a flat callee program whose first `capture_count` inputs are its lifted
/// captures, or [`None`] for programs that are not eligible for structural deduplication (refer to
/// [`supports_structural_deduplication`]) and therefore always inline.
fn call_program_key<O: ProgramLoweringOperation>(
    program: &FlatProgram<EagerContext<Constant, O>>,
    capture_count: usize,
) -> Option<CallProgramKey> {
    supports_structural_deduplication(program).then(|| structural_key(program, capture_count))
}

/// Computes the deduplication key for a borrowed callee region whose first `capture_count` inputs are its lifted
/// captures, materializing only after borrowed eligibility checks.
fn call_region_key<O: ProgramLoweringOperation>(
    region: RegionRef<'_, Constant, O>,
    capture_count: usize,
) -> Option<CallProgramKey> {
    supports_structural_deduplication_of_region(region).then(|| structural_key(&region.to_program(), capture_count))
}

/// Returns the deduplication key of `program`, an eligible callee whose first `capture_count` inputs are its lifted
/// captures (refer to [`CallProgramKey`]).
fn structural_key<O: ProgramLoweringOperation>(
    program: &FlatProgram<EagerContext<Constant, O>>,
    capture_count: usize,
) -> CallProgramKey {
    fn walk<O: ProgramLoweringOperation>(
        region: RegionRef<'_, Constant, O>,
        visited: &mut HashSet<RegionId>,
        calls: &mut Vec<CallOperation<Type>>,
    ) {
        if !visited.insert(region.id()) {
            return;
        }
        for instruction in region.instructions() {
            if let Some(call) = instruction.operation().projected_payload::<CallOperation<Type>>() {
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

    let mut has_captured_constants = false;
    let mut first_positions = HashMap::new();
    let mut session_constants = Vec::new();
    for region in program.regions() {
        for atom in region.atoms() {
            match atom.as_constant() {
                Some(Constant::Captured(_)) => has_captured_constants = true,
                Some(Constant::Session(constant)) => {
                    let position = session_constants.len();
                    session_constants.push(*first_positions.entry(constant.identity_key()).or_insert(position));
                }
                _ => {}
            }
        }
    }

    CallProgramKey {
        rendered: program.to_string(),
        input_types: program.input_types().into_owned(),
        output_types: program.output_types().into_owned(),
        calls,
        capture_count: if has_captured_constants { capture_count } else { 0 },
        session_constants,
    }
}

/// Counts `call` callee occurrences in `program`, covering nested computations at every depth. Attached regions
/// (i.e., control-flow bodies, composite programs, and callee bodies) all live in the program's one canonical region
/// arena, so the walk descends region edges without materializing nested programs. `shard_map` body regions are
/// intentionally skipped, because their `call`s lower with shard-local types and always inline.
///
/// The bodies of named composite calls count like callees (refer to
/// [`NamedComposite`](ryft_core::NamedComposite)), unless a backend implementation replaces the call on
/// `target_platform`, so that each specialization lowers to one shared function. Their calls share one interned body
/// region per specialization, so each body is rendered once. `named` records the semantic identity and boundary
/// of the calls that each counted body implements, and `names` the name of their definition.
///
/// `counts` accumulates the occurrence count and a representative callee region, with its materialized program, per
/// identity, `order` records
/// keys in first-occurrence order, and `memo` caches the (possibly expensive) key computation per callee root
/// [`RegionId`]. Interned shared callees repeat their root region id, so each shared callee is rendered once.
fn count_calls<Input, Output, O: ProgramLoweringOperation>(
    program: &Program<Constant, O, Input, Output>,
    target_platform: Option<&str>,
    counts: &mut HashMap<CallProgramKey, (usize, RegionId, FlatProgram<EagerContext<Constant, O>>)>,
    order: &mut Vec<CallProgramKey>,
    memo: &mut HashMap<(RegionId, usize), Option<CallProgramKey>>,
    named: &mut HashMap<NamedFunctionKey, Option<CallProgramKey>>,
    names: &mut HashMap<CallProgramKey, String>,
) -> Result<(), Error>
where
    Input: Parameterized<Constant>,
    Output: Parameterized<Constant>,
{
    #[allow(clippy::too_many_arguments)]
    fn walk<Input, Output, O: ProgramLoweringOperation>(
        program: &Program<Constant, O, Input, Output>,
        region: RegionId,
        target_platform: Option<&str>,
        counts: &mut HashMap<CallProgramKey, (usize, RegionId, FlatProgram<EagerContext<Constant, O>>)>,
        order: &mut Vec<CallProgramKey>,
        memo: &mut HashMap<(RegionId, usize), Option<CallProgramKey>>,
        named_calls: &mut NamedRegionCalls,
        visited: &mut HashSet<RegionId>,
    ) -> Result<(), Error>
    where
        Input: Parameterized<Constant>,
        Output: Parameterized<Constant>,
    {
        // Interned shared callees repeat their root region id. Each region's instructions are walked once while
        // every referencing `call` instruction still contributes one occurrence to the counts below.
        if !visited.insert(region) {
            return Ok(());
        }
        let region_ref = program.region_ref(region).unwrap();
        for instruction in region_ref.instructions() {
            if instruction.operation().projected_payload::<ShardMapOperation>().is_some() {
                continue;
            }
            if instruction
                .operation()
                .omits_regions(target_platform, || region_ref.instruction_boundary_types(instruction))
            {
                continue;
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
                let (input_types, output_types) = region_ref.instruction_boundary_types(instruction);
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
            if let Some(call) = instruction.operation().projected_payload::<CallOperation<Type>>() {
                let Some(&callee_region) = instruction.regions().first() else {
                    continue;
                };
                let capture_count = call.capture_count();
                let key = memo
                    .entry((callee_region, capture_count))
                    .or_insert_with(|| {
                        let callee = program.region_ref(callee_region).unwrap();
                        call_region_key(callee, capture_count)
                    })
                    .clone();
                let Some(key) = key else {
                    continue;
                };
                let entry = counts.entry(key.clone()).or_insert_with(|| {
                    order.push(key.clone());
                    (0, callee_region, program.region_ref(callee_region).unwrap().to_program())
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
        // The body of a named composite call has no captures (refer to the lowering of `CompositeOperation`).
        let key = memo
            .entry((call.region, 0))
            .or_insert_with(|| call_region_key(program.region_ref(call.region).unwrap(), 0))
            .clone();
        // A body that structural deduplication cannot key (e.g., one with constants) has no shared function, and it
        // marks its key and boundary as conflicting, so that its calls never resolve to another body's function.
        let Some(key) = key else {
            named.insert(named_key, None);
            continue;
        };
        let entry = counts.entry(key.clone()).or_insert_with(|| {
            order.push(key.clone());
            (0, call.region, program.region_ref(call.region).unwrap().to_program())
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

/// Lowering of the callee program of a `call`.
trait CallProgramLoweringExtension<O: ProgramLoweringOperation> {
    /// Lowers one `call` to either a `func.call` of a shared private function (when its callee was deduplicated) or
    /// an inlined copy of the callee body (otherwise).
    ///
    /// `input_values` are the lowered call inputs in callee-input order. `capture_count` is the operation payload's
    /// exact leading lifted-capture prefix length. Deriving it from capture-constant indices instead would conflate the
    /// independent capture namespaces of nested calls inside the callee arena. `caller_captured_values` are the
    /// captured values of the calling scope, whose session-constant suffix the inlined callee inherits (refer to
    /// [`ProgramLoweringContext::session_constant_values`]).
    fn lower_call<'b, 'c: 'b, 't: 'c>(
        &self,
        capture_count: usize,
        input_values: &[ValueRef<'b, 'c, 't>],
        caller_captured_values: &[ValueRef<'b, 'c, 't>],
        block: &mut BlockRef<'b, 'c, 't>,
        context: &'c MlirContext<'t>,
        location: LocationRef<'c, 't>,
        lowering_context: &ProgramLoweringContext,
        effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error>;
}

impl<O: ProgramLoweringOperation> CallProgramLoweringExtension<O> for FlatProgram<EagerContext<Constant, O>> {
    #[allow(clippy::too_many_arguments)]
    fn lower_call<'b, 'c: 'b, 't: 'c>(
        &self,
        capture_count: usize,
        input_values: &[ValueRef<'b, 'c, 't>],
        caller_captured_values: &[ValueRef<'b, 'c, 't>],
        block: &mut BlockRef<'b, 'c, 't>,
        context: &'c MlirContext<'t>,
        location: LocationRef<'c, 't>,
        lowering_context: &ProgramLoweringContext,
        effect_tokens: &mut EffectTokens<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        // Only pure callees are ever deduplicated (`plan_call_functions` skips effectful programs), so the shared
        // function path below never interacts with the caller's effect tokens. Effectful callees always take the inline
        // path, which threads the caller's per-class tokens through the callee body in program order.
        if let Some(outputs) = lower_shared_call(
            self,
            capture_count,
            input_values,
            caller_captured_values,
            block,
            context,
            location,
            lowering_context,
        )? {
            return Ok(outputs);
        }
        if input_values.len() < capture_count {
            // The provided inputs cover the leading captures, so the first capture without a value is the next one.
            return Err(Error::Lowering(LoweringError::MissingCapturedConstant { index: input_values.len() }));
        }
        // The callee binds its own leading inputs as captures and inherits the caller's session constant bindings.
        let mut captured_values = input_values[..capture_count].to_vec();
        captured_values.extend_from_slice(lowering_context.session_constant_values(caller_captured_values));
        self.lower_inline(captured_values.as_slice(), input_values, lowering_context, effect_tokens, *block, location)
    }
}

/// Constrains `value`, an input or an output of type `r#type` of a `call`, to `sharding`, the effective boundary
/// sharding of the corresponding position of the boundary of its callee (i.e., its override, or the declared sharding
/// of an input without one, refer to [`CallOperation::boundary_input_sharding`]), and returns the constrained value.
/// The constraint is the sharding with which the called function annotates that position of its own boundary (refer to
/// [`Sharding::with_unconstrained_dimensions_replicated`]), closed over auto mesh axes. Values that
/// are not arrays are returned unchanged.
///
/// # Errors
///
/// Returns [`LoweringError::UnsupportedOperation`] when the boundary sharding declares a pending maximum or minimum
/// that `r#type` does not record (refer to [`lower_declared_sharding_constraint`]), which XLA cannot introduce.
fn lower_boundary_constraint<'b, 'c: 'b, 't: 'c, O: ProgramLoweringOperation>(
    lowerer: &Lowerer<'b, 'c, 't, O>,
    value: ValueRef<'b, 'c, 't>,
    r#type: &Type,
    sharding: &Sharding,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    let Type::Array(array_type) = r#type else {
        return Ok(value);
    };
    // The boundary states the whole placement. Its reduction transition starts from the value's original type.
    let declared = sharding.with_unconstrained_dimensions_replicated();
    lower_declared_sharding_constraint(lowerer, value, array_type, &declared, CALL_OPERATION_NAME)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, ArrayType, AssertOperation, AtomId, AttachedRegions, CaptureReference, CompositeJvpRule,
        CompositeOperation, CompositeTransposeOperation, ConditionOperation, Context, ConvertElementTypeOperation,
        CustomCallOperation, CustomRuleDefinition, CustomRuleRegistration, DataType, Device, Differentiate, Dimension,
        DimensionBounds, DimensionType, DotDimensionNumbers, DotOperation, DotsSavable, EagerInterpretationDriver,
        EffectClass, Fill, ImportingContext, IotaOperation, Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType,
        MulOperation, NamedComposite, NothingSavable, Operation as CoreOperation, PartialEvaluationContext,
        PartialEvaluationValue, PartialTracer, Place, Placeholder, PrintOperation,
        ProgramBuilder as CoreProgramBuilder, ProgramError, Reduce, ReferenceNewOperation, ReferenceReadOperation,
        RegionDriver, RegionRef, RematerializeOperation, ResidualInputSource, ResidualPolicyReference,
        SaveOnlyTheseNames, ScanOptions, Shape, Sharding, ShardingDimension, Sin, SinOperation, StagingContext,
        TagOperation, ThroughCapture, TiledLayout, Typed, TypedProgram, UnreducedKind, Value as CoreValue,
        ValueProjection, ZeroOperation, scan,
    };
    use ryft_pjrt::{
        BufferSpecification, BufferType, ExecutionDeviceInputs, ExecutionFence, ExecutionInput, Program as PjrtProgram,
    };

    use crate::lowering::tests::{
        ProgramBuilder, add_call, execute_mixed_program, identity_branch, lower_replicated_shard_map_body, neg_branch,
        test_vector_type, unproject_plain_program,
    };
    use crate::operations::debugging::{ensure_print_handler_registered, with_captured_prints};
    use crate::sessions::tests::{domain_mesh, f32_scalar, read_first_shard, read_sharded_f32_array, two_device_mesh};
    use crate::tests::{cpu_client, execution_client, test_compilation_options, values_from_bytes, values_to_bytes};
    use crate::{
        Array, CompileTracer, CompiledFunction, Constant, CustomCallOperationExtension, Error, FromPjrt, Options,
        ProgramLoweringExtension, ProgramLoweringOptions, RetainedArray, RuntimeRequirement, StagedFunction, ToPjrt,
        Type, Value, compile, compile_with_options, stage,
    };

    use super::*;

    /// Compiles and executes an array-only `f32` fixture using its declared unsharded program signature.
    fn execute_f32_program(
        client: &'static ryft_pjrt::Client<'static>,
        program: &FlatProgram<Session>,
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

    /// Compiles and executes an array-only fixture using its declared unsharded program signature, with each input
    /// and output given as the native-endian bytes of its elements.
    fn execute_array_program(
        client: &'static ryft_pjrt::Client<'static>,
        program: &FlatProgram<Session>,
        values: Vec<Vec<u8>>,
    ) -> Vec<Vec<u8>> {
        let array_type = |r#type: &Type| match r#type {
            Type::Array(r#type) => r#type.clone(),
            _ => panic!("expected an array signature"),
        };
        let input_types = program.input_types().iter().map(array_type).collect::<Vec<_>>();
        let output_types = program.output_types().iter().map(array_type).collect::<Vec<_>>();
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

    /// Builds the flat callee `f(x) = x + x` over a vector type, returned behind an [`Arc`] so callers control
    /// whether `call` sites share one program (i.e., by pointer identity) or use structurally identical distinct
    /// programs.
    fn add_self_callee(input_type: ArrayType) -> Arc<FlatProgram<Session>> {
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(input_type.into());
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
        Arc::new(builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap())
    }

    /// Lowers an outer program that calls each of `callees` through one `call` and sums the outputs, returning
    /// the module text. Each callee is `f(x) = x + x`, and the outer function is `g(x) = sum_i callee_i(x)`.
    fn lower_call_sum_module(callees: Vec<Arc<FlatProgram<Session>>>) -> String {
        let array_type = test_vector_type(4);
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(array_type.clone().into());
        let mut accumulator: Option<AtomId> = None;
        for callee in callees {
            let call_output = add_call(&mut builder, &callee, vec![input]);
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
        program
            .to_rendered_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_input_types(&input_types)
                    .with_output_types(&output_types)
                    .without_boundary_shardings(),
            )
            .unwrap()
    }

    #[test]
    fn test_plan_call_functions() {
        let callee = add_self_callee(test_vector_type(4));
        let module = lower_call_sum_module(vec![callee.clone(), callee]);

        // Both calls of the one shared program collapse to a single private `func.func` plus two `func.call`s.
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func private @callee_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = call @callee_0(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %1 = call @callee_0(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %2 = stablehlo.add %0, %1 : tensor<4xf32>
                    return %2 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_plan_call_functions_shares_structurally_identical_programs() {
        // Two distinct programs (i.e., separate `Arc`s) with identical structure must still deduplicate into the same
        // single shared function as the pointer-identical case above. This is the shape produced when a transform
        // such as `grad` linearizes each of several identical blocks into its own staged program.
        let module =
            lower_call_sum_module(vec![add_self_callee(test_vector_type(4)), add_self_callee(test_vector_type(4))]);

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func private @callee_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = call @callee_0(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %1 = call @callee_0(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %2 = stablehlo.add %0, %1 : tensor<4xf32>
                    return %2 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_plan_call_functions_shares_callees_with_constants() {
        // An immediate constant is part of the rendering of its callee, and a captured constant names a leading input
        // of its callee, so a repeated callee that reads either is shared like any other.
        let vector_type = test_vector_type(4);
        let callee = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            builder.add_input(vector_type.clone().into());
            let input = builder.add_input(vector_type.clone().into());
            let capture =
                builder.add_constant(Constant::Captured(CaptureReference::new(0, vector_type.clone().into())));
            builder.add_constant(Constant::Boolean(true));
            let output =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, capture], None).unwrap()[0];
            let callee =
                builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder]);
            Arc::new(callee.unwrap())
        };
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let capture = builder.add_input(vector_type.clone().into());
        let mut value = builder.add_input(vector_type.into());
        for _ in 0..2 {
            let region = builder.intern_callee(&callee, None).unwrap();
            let call = Operation::Call(CallOperation::new(1));
            value = builder.add_instruction(call, vec![region], vec![capture, value], None).unwrap()[0];
        }
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_rendered_mlir("main", &ProgramLoweringOptions::new().without_pruning()).unwrap(),
            indoc! {r#"
                module {
                  func.func private @callee_0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg1, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = call @callee_0(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
                    %1 = call @callee_0(%arg0, %0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
                    return %1 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_plan_call_functions_passes_session_constants_to_shared_functions() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let [first, second] = [1.0, 2.0].map(|value| RetainedArray::new(&f32_scalar(&session, &mesh, value)));
        let scalar_type = Type::Array(first.r#type().into_owned());
        let options = ProgramLoweringOptions::new().without_pruning().without_boundary_shardings();
        let add_constants = |constants: &[&RetainedArray]| {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let mut value = builder.add_input(scalar_type.clone());
            for constant in constants {
                let constant = builder.add_constant(Constant::Session((*constant).clone()));
                value =
                    builder.add_instruction(AddOperation::new(), Vec::new(), vec![value, constant], None).unwrap()[0];
            }
            let callee =
                builder.build::<Vec<Constant>, Vec<Constant>>(vec![value], vec![Placeholder], vec![Placeholder]);
            Arc::new(callee.unwrap())
        };
        let chain = |callees: &[&Arc<FlatProgram<Session>>]| {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let mut value = builder.add_input(scalar_type.clone());
            for callee in callees {
                value = add_call(&mut builder, callee, vec![value]);
            }
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![value], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        // Structurally identical callees that read different arrays share one function, which receives the array of
        // each call as its last input. The entry function receives both arrays as hidden inputs and passes them on.
        let program = chain(&[&add_constants(&[&first]), &add_constants(&[&second])]);
        assert_eq!(
            program.to_rendered_mlir("main", &options).unwrap(),
            indoc! {r#"
                module {
                  func.func private @callee_0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
                    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
                    return %0 : tensor<f32>
                  }
                  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
                    %0 = call @callee_0(%arg2, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
                    %1 = call @callee_0(%0, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
                    return %1 : tensor<f32>
                  }
                }
            "#},
        );

        // A shared function receives the arrays that the functions which it calls read, and passes them on.
        let inner = add_constants(&[&first]);
        let outer = Arc::new(chain(&[&inner, &inner]));
        assert_eq!(
            chain(&[&outer, &outer]).to_rendered_mlir("main", &options).unwrap(),
            indoc! {r#"
                module {
                  func.func private @callee_0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
                    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
                    return %0 : tensor<f32>
                  }
                  func.func private @callee_1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
                    %0 = call @callee_0(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
                    %1 = call @callee_0(%0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
                    return %1 : tensor<f32>
                  }
                  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
                    %0 = call @callee_1(%arg1, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
                    %1 = call @callee_1(%0, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
                    return %1 : tensor<f32>
                  }
                }
            "#},
        );

        // Callees share a function only when they repeat their arrays in the same pattern, because a function has
        // one input for each distinct array.
        let function_count = |program: &FlatProgram<Session>| {
            let lowering_context = ProgramLoweringContext::new();
            plan_call_functions(program, &lowering_context).unwrap();
            (0..).take_while(|index| lowering_context.function(*index).is_some()).count()
        };
        let (repeated, distinct) = (add_constants(&[&first, &first]), add_constants(&[&first, &second]));
        assert_eq!(function_count(&chain(&[&repeated, &add_constants(&[&second, &second])])), 1);
        assert_eq!(function_count(&chain(&[&distinct, &add_constants(&[&second, &first])])), 1);
        assert_eq!(function_count(&chain(&[&repeated, &distinct])), 0);
    }

    #[test]
    fn test_plan_call_functions_distinguishes_callees_whose_calls_differ() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let device_mesh = DeviceMesh::new(mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)]).unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let array_type = test_vector_type(4).with_sharding(replicated.clone()).unwrap();
        let inner = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(array_type.clone().into());
            Arc::new(
                builder
                    .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
                    .unwrap(),
            )
        };
        let outer = |call: &CallOperation<Type>| {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(array_type.clone().into());
            let region = builder.intern_callee(&inner, None).unwrap();
            let output =
                builder.add_instruction(Operation::Call(call.clone()), vec![region], vec![input], None).unwrap()[0];
            Arc::new(
                builder
                    .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                    .unwrap(),
            )
        };
        let program = |first: &CallOperation<Type>, second: &CallOperation<Type>| {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(array_type.clone().into());
            let first = add_call(&mut builder, &outer(first), vec![input]);
            let second = add_call(&mut builder, &outer(second), vec![input]);
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![first, second],
                    vec![Placeholder],
                    vec![Placeholder, Placeholder],
                )
                .unwrap()
        };
        let base = CallOperation::<Type>::new(0);

        // Two structurally identical callees whose calls differ in any field of the call that lowering uses share no
        // function, so that each call lowers its own boundary. Only the shared innermost callee gets a function.
        // Lowering never reads the mesh of the called function, which only an application by a session reads.
        let derived = |shardings| CallOperation::<Type>::new(0).with_derived_input_shardings(shardings);
        let variants = [
            CallOperation::new(1),
            CallOperation::new(0).with_boundary(Some(&device_mesh), vec![Some(sharded.clone())], Vec::new()),
            CallOperation::new(0).with_boundary(Some(&device_mesh), Vec::new(), vec![Some(replicated.clone())]),
            CallOperation::new(0).with_boundary(Some(&device_mesh), Vec::new(), Vec::new()),
            CallOperation::new(0).with_kept_input_placements(),
            derived(vec![None]),
            derived(vec![Some(sharded)]),
        ];
        let function_count = |program: &FlatProgram<Session>| {
            let lowering_context = ProgramLoweringContext::new();
            plan_call_functions(program, &lowering_context).unwrap();
            (0..).take_while(|index| lowering_context.function(*index).is_some()).count()
        };
        assert_eq!(function_count(&program(&base, &base)), 2);
        for variant in &variants {
            assert_eq!(function_count(&program(&base, variant)), 1, "{variant:?}");
        }

        // A call that a transform derives without a boundary for its input renders as a call of a compiled function,
        // which constrains its input to the replicated sharding of the declared type, but the two lower differently.
        let input_types = vec![array_type.clone()];
        let output_types = vec![array_type.clone(), array_type.clone()];
        let module = (program(&base, &derived(vec![None])))
            .to_rendered_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .with_input_types(&input_types)
                    .with_output_types(&output_types)
                    .without_boundary_shardings(),
            )
            .unwrap();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func private @callee_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    return %arg0 : tensor<4xf32>
                  }
                  func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{}]> : tensor<4xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{}]> : tensor<4xf32>
                    %2 = call @callee_0(%1) : (tensor<4xf32>) -> tensor<4xf32>
                    %3 = sdy.sharding_constraint %arg0 <@mesh, [{}]> : tensor<4xf32>
                    %4 = call @callee_0(%3) : (tensor<4xf32>) -> tensor<4xf32>
                    return %2, %4 : tensor<4xf32>, tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_plan_call_functions_inlines_single_calls() {
        // A single-occurrence callee stays below the dedup threshold and inlines, so no shared function is emitted
        // and the callee body appears directly in `@main`.
        let module = lower_call_sum_module(vec![add_self_callee(test_vector_type(4))]);

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
    fn test_plan_call_functions_excludes_dormant_regions() {
        let array_type = test_vector_type(4);
        let callee = |identifier| {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
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
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(array_type.clone().into());
            let first = add_call(&mut builder, &active, vec![input]);
            let second = add_call(&mut builder, &active, vec![input]);
            let output =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![first, second], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let jvp = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(array_type.clone().into());
            let tangent = builder.add_input(array_type.clone().into());
            let first = add_call(&mut builder, &dormant, vec![tangent]);
            let second = add_call(&mut builder, &dormant, vec![tangent]);
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
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
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
        let lowered = program.to_mlir("main", &ProgramLoweringOptions::new()).unwrap();

        let (stable_hlo, _, runtime_requirements) = lowered.into_parts();
        assert_eq!(runtime_requirements.as_ref(), &[RuntimeRequirement::builtin("test.active")]);
        assert_eq!(
            stable_hlo,
            indoc! {r#"
                module {
                  func.func private @callee_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.custom_call @test.active(%arg0) {api_version = 4 : i32, backend_config = {}, operand_layouts = [dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>]} : (tensor<4xf32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = call @callee_0(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %1 = call @callee_0(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
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
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
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
        let lowered =
            program.to_mlir("main", &ProgramLoweringOptions::new().with_target_platform(Some("cuda"))).unwrap();

        let (stable_hlo, _, runtime_requirements) = lowered.into_parts();
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
    fn test_plan_call_functions_skips_shard_map_bodies() {
        // `count_calls` intentionally skips `shard_map` bodies, so a callee that occurs twice inside a `shard_map`
        // body never gets a shared `func.func private @callee_*` and both occurrences inline into the
        // `sdy.manual_computation` region.
        let vector_type = test_vector_type(4);
        let callee = add_self_callee(vector_type.clone());
        let body_program = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(vector_type.clone().into());
            let first = add_call(&mut builder, &callee, vec![input]);
            let second = add_call(&mut builder, &callee, vec![input]);
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
    fn test_plan_call_functions_shares_programs_with_nested_bodies() {
        /// Builds the flat callee `f(p, x) = if p { -x } else { x }`, whose body contains a `condition` instruction.
        /// Its nested branch bodies render contextually into the callee's canonical program text, so separately
        /// constructed copies of this callee remain eligible for structural `call` deduplication.
        fn condition_callee() -> Arc<FlatProgram<Session>> {
            let vector_type = test_vector_type(4);
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
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

        // With nested computations attached as regions, a callee's `condition` bodies render contextually inside its
        // canonical rendering, so two structurally identical but separately constructed callees share one structural
        // identity and merge into a single emitted function, exactly like the body-free callees of
        // `test_plan_call_functions_shares_structurally_identical_programs`.
        let first = condition_callee();
        let second = condition_callee();
        let vector_type = test_vector_type(4);
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(vector_type.clone().into());
        let mut accumulator: Option<AtomId> = None;
        for callee in [first.clone(), first, second.clone(), second] {
            let call_output = add_call(&mut builder, &callee, vec![predicate, input]);
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
        let module = program.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func private @callee_0(%arg0: tensor<i1>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = "stablehlo.if"(%arg0) ({
                      %1 = stablehlo.negate %arg1 : tensor<4xf32>
                      stablehlo.return %1 : tensor<4xf32>
                    }, {
                      stablehlo.return %arg1 : tensor<4xf32>
                    }) : (tensor<i1>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func @main(%arg0: tensor<i1>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = call @callee_0(%arg0, %arg1) : (tensor<i1>, tensor<4xf32>) -> tensor<4xf32>
                    %1 = call @callee_0(%arg0, %arg1) : (tensor<i1>, tensor<4xf32>) -> tensor<4xf32>
                    %2 = stablehlo.add %0, %1 : tensor<4xf32>
                    %3 = call @callee_0(%arg0, %arg1) : (tensor<i1>, tensor<4xf32>) -> tensor<4xf32>
                    %4 = stablehlo.add %2, %3 : tensor<4xf32>
                    %5 = call @callee_0(%arg0, %arg1) : (tensor<i1>, tensor<4xf32>) -> tensor<4xf32>
                    %6 = stablehlo.add %4, %5 : tensor<4xf32>
                    return %6 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_plan_call_functions_inlines_effectful_callees() {
        // A repeated `call` callee that prints is excluded from function deduplication and inlines at every
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
        let module = lower_call_sum_module(vec![callee.clone(), callee]);

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
    fn test_plan_call_functions_inlines_callees_with_synthesized_guards() {
        // The repeated callee is pure in the source program, but its inner conditional's mixed entry (a symbolic
        // `dimension<m>` and an exact `f64[2]` entering `[dimension<n>, f64[n]]`) gains an entry guard during lowering.
        // A shared token-free function could not order that guard, so the callee inlines at both call sites, and the
        // print after them consumes a token that the guards produce.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let m = DimensionType::new("m", DimensionBounds::positive(Some(8)).unwrap());
        let array_type = ArrayType::new_static(DataType::F64, [2]);
        let branch = || {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            builder.add_input(n.clone().into());
            let array =
                builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![n.simplified_dimension()])).into());
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![array], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let callee = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
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
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let dimension = builder.add_input(m.clone().into());
        let array = builder.add_input(array_type.clone().into());
        add_call(&mut builder, &callee, vec![predicate, dimension, array]);
        let selected = add_call(&mut builder, &callee, vec![predicate, dimension, array]);
        let printed = builder
            .add_instruction(PrintOperation::<ArrayType>::new("after"), Vec::new(), vec![selected], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![printed], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let module = program
            .to_rendered_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_input_types(&[
                        ArrayType::scalar(DataType::Boolean),
                        ArrayType::scalar(DataType::I64),
                        array_type.clone(),
                    ])
                    .with_output_types(&[array_type]),
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
    fn test_plan_call_functions_inlines_callees_with_extent_validations() {
        // The repeated callee is pure in the source program, but its `zero[n, n]` constructor consumes a symbolic
        // `dimension<n>` and an exact `dimension<2>`, so type inference refines it to `f64[2, 2]` and lowering
        // validates that `n = 2` at runtime. A shared token-free function could not order those validations, so the
        // callee inlines at both call sites, and the explicit assertion after them continues their assertion chain.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let callee = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
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
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let symbolic = builder.add_input(n.clone().into());
        let exact = builder.add_input(DimensionType::from(Dimension::Static(2)).into());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        add_call(&mut builder, &callee, vec![symbolic, exact]);
        let zero = add_call(&mut builder, &callee, vec![symbolic, exact]);
        builder
            .add_instruction(AssertOperation::<ArrayType>::new("required predicate"), Vec::new(), vec![predicate], None)
            .unwrap();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![zero], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let output_type = ArrayType::new_static(DataType::F64, [2, 2]);
        let module = program
            .to_rendered_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_input_types(&[
                        ArrayType::scalar(DataType::I64),
                        ArrayType::scalar(DataType::I64),
                        ArrayType::scalar(DataType::Boolean),
                    ])
                    .with_output_types(&[output_type]),
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
    fn test_call_operation_eager_decision_rejects_reduction_state_over_auto_axes() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let session = Session::builder().client(client).build().unwrap();
        let scalar_type = ArrayType::scalar(DataType::F32);
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input = |sharding: &Sharding| {
            let buffers = client
                .addressable_devices()
                .unwrap()
                .into_iter()
                .zip([1.0f32, 2.0])
                .map(|(device, value)| {
                    client.buffer(value.to_ne_bytes().as_slice(), BufferType::F32, [], None, device, None).unwrap()
                })
                .collect();
            Array::from_addressable_buffers(
                &session,
                scalar_type.clone(),
                mesh.clone(),
                Some(sharding.clone()),
                buffers,
            )
            .unwrap()
        };
        let plain: CompiledFunction<ArrayType, ArrayType> =
            compile(|x: CompileTracer| x, scalar_type.clone(), &session, mesh.clone()).unwrap();
        let overridden: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
            |x: CompileTracer| x,
            scalar_type.clone(),
            &session,
            Options::new(mesh.clone()).with_input_shardings(vec![replicated.clone()]),
        )
        .unwrap();

        // An eager call places its input at the boundary of its callee, which expects it reduced, and a reshard never
        // completes a pending sum over an auto axis, so the call rejects the input as an execution of the callee does,
        // whether or not the callee names devices. A reduced input passes.
        for function in [&plain, &overridden] {
            assert_eq!(
                function
                    .call(input(&replicated.clone().with_unreduced_axes(["x"]).unwrap()))
                    .unwrap_err()
                    .to_string(),
                "public input #0 is unreduced along the mesh axes [\"x\"], which the executable cannot receive \
                 because a staged type does not record a pending reduction over auto mesh axes; reduce the input \
                 (e.g., with `Array::to_placement` onto a sharding without unreduced axes), or place it over explicit \
                 mesh axes, before passing it to a compiled function or an eager operation",
            );
            assert!(function.call(input(&replicated)).is_ok());
        }
    }

    #[test]
    fn test_call_operation_eager_decision_gives_inputs_the_declared_logical_mesh() {
        // A compiled identity declares its input replicated over the mesh `q`. An input replicated over another
        // logical mesh of the same devices (e.g., `p`, or the default mesh of an upload) only needs the logical mesh of
        // the declared input, which a direct execution and a call both give it, on one device as on two.
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let values = [1f32, 2., 3., 4.];
        let vector_type = ArrayType::new_static(DataType::F32, [4]);
        for device_count in [1, 2] {
            let mesh = |name: &str, axis_type| {
                let axes = vec![MeshAxis::new(name, device_count, axis_type).unwrap()];
                DeviceMesh::new(LogicalMesh::new(axes).unwrap(), devices[..device_count].to_vec()).unwrap()
            };
            for declared_axis_type in [MeshAxisType::Explicit, MeshAxisType::Auto] {
                let target = mesh("q", declared_axis_type);
                let replicated = Sharding::replicated(target.logical_mesh().clone(), 1);
                let declared = vector_type.clone().with_sharding(replicated).unwrap();
                let identity: CompiledFunction<ArrayType, ArrayType> =
                    compile(|x: CompileTracer| x, declared.clone(), &session, target).unwrap();
                let sources = [
                    ("q", declared_axis_type),
                    ("p", declared_axis_type),
                    ("p", MeshAxisType::Explicit),
                    ("device", MeshAxisType::Auto),
                ];
                for (name, axis_type) in sources {
                    let input = Array::from_host_buffer(
                        &session,
                        vector_type.clone(),
                        Some(mesh(name, axis_type)),
                        None,
                        values_to_bytes(&values),
                    )
                    .unwrap();
                    let observe = |output: Array| {
                        (output.r#type().into_owned(), values_from_bytes::<f32>(&output.copy_to_host_bytes().unwrap()))
                    };
                    let executed = identity.executable().execute(&session, input.clone()).map(observe);
                    let called = identity.call(input).map(observe);
                    let expected = Ok((declared.clone(), values.to_vec()));
                    assert_eq!(
                        (executed.map_err(|error| error.to_string()), called.map_err(|error| error.to_string())),
                        (expected.clone(), expected),
                        "{device_count} {declared_axis_type:?} {name} {axis_type:?}",
                    );
                }
            }
        }
    }

    #[test]
    fn test_call_operation_eager_decision_without_array_inputs() {
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let observe = |outputs: Result<Vec<Value>, ProgramError>| {
            let outputs = outputs.unwrap();
            let [Value::Array(output)] = outputs.as_slice() else {
                panic!("expected one array output");
            };
            let values = output.copy_to_host().unwrap().elements::<i32>().unwrap();
            (output.mesh().clone(), output.is_committed(), output.is_locally_assigned(), values)
        };

        // A call of a callee without inputs that names no devices runs where the session places a value without array
        // inputs over the logical mesh of its outputs (refer to `Session::device_mesh_for`), which is the first device
        // of the client for a named mesh of one device and the first two devices for a mesh of two devices, and the
        // outputs are committed exactly when they live on more than one device. An eager call and the interpretation
        // rule of the call run it on the same devices and give its outputs the same facts.
        for (axis, size) in [("a", 1), ("x", 2)] {
            let logical_mesh =
                LogicalMesh::new(vec![MeshAxis::new(axis, size, MeshAxisType::Explicit).unwrap()]).unwrap();
            let r#type = ArrayType::new_static(DataType::I32, [2])
                .with_sharding(Sharding::replicated(logical_mesh.clone(), 1))
                .unwrap();
            let callee = {
                let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
                let iota = IotaOperation::<ArrayType>::new(r#type, 0).unwrap();
                let output = builder.add_instruction(iota, Vec::new(), Vec::new(), None).unwrap()[0];
                Arc::new(
                    builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], Vec::new(), vec![Placeholder]).unwrap(),
                )
            };
            let regions = AttachedRegions::Callees(std::slice::from_ref(&callee));
            let call = CallOperation::new(0);
            let mesh = DeviceMesh::new(logical_mesh, devices[..size].to_vec()).unwrap();
            let expected = (mesh, size > 1, true, vec![0, 1]);
            let bound = session.bind(Operation::Call(call.clone()), AttachedRegions::Callees(&[callee.clone()]), &[]);
            assert_eq!(observe(bound), expected);
            assert_eq!(observe(call.interpret(&session, &EagerInterpretationDriver::new(&regions), &[])), expected);
        }
    }

    #[test]
    fn test_call_operation_eager_decision_runs_known_calls_on_the_function_mesh() {
        // A call of a function compiled for a named mesh of the second device runs there and commits its output when
        // an uncommitted input on the first device is applied directly, when partial evaluation applies the call on
        // the eager known side (with and without a residual policy), and when it applies the known call of a split.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("q", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![devices[1]],
        )
        .unwrap();
        let scalar_type = ArrayType::scalar(DataType::F32).replicated(&mesh).unwrap();
        let input = Array::from_host_buffer(&session, scalar_type.clone(), None, None, 4f32.to_ne_bytes()).unwrap();
        assert!(!input.is_committed());
        assert_ne!(input.mesh().devices(), mesh.devices());

        let function: CompiledFunction<ArrayType, ArrayType> =
            compile(|input: CompileTracer| input, scalar_type.clone(), &session, mesh.clone()).unwrap();
        let observe =
            |output: &Array| (output.mesh().devices().to_vec(), output.is_committed(), read_sharded_f32_array(output));
        let expected = (mesh.devices().to_vec(), true, vec![4.0]);
        assert_eq!(observe(&function.call(input.clone()).unwrap()), expected);

        let policy = ResidualPolicyReference::new(NothingSavable);
        let context = PartialEvaluationContext::new(session.clone());
        for context in [context.clone(), context.with_residual_policy(&policy)] {
            let input = ValueProjection::<ArrayType>::into_projected(PartialTracer::new(
                context.clone(),
                PartialEvaluationValue::known(Value::Array(input.clone())),
            ))
            .unwrap();
            let output = function.call_in_context(&context, input).unwrap().into_value().into_value().unwrap();
            let Some(Value::Array(output)) = output.as_known() else { panic!("expected a known array") };
            assert_eq!(observe(output), expected);
        }

        // A nontrivial split has separate calls for its known and residual outputs. The known call runs on the
        // function's named mesh even though the unknown input will only enter the residual call.
        let split = compile::<_, (ArrayType, ArrayType), (ArrayType, ArrayType)>(
            |(left, right): (CompileTracer, CompileTracer)| (left.clone() + left, right.clone() + right),
            (scalar_type.clone(), scalar_type.clone()),
            &session,
            mesh.clone(),
        )
        .unwrap();
        let context = PartialEvaluationContext::new(session);
        let partial =
            |value| ValueProjection::<ArrayType>::into_projected(PartialTracer::new(context.clone(), value)).unwrap();
        let known = partial(PartialEvaluationValue::known(Value::Array(input)));
        let unknown = partial(context.unknown_input(scalar_type.into(), 1));
        let (known, unknown) = split.call_in_context(&context, (known, unknown)).unwrap();
        let known = known.into_value().into_value().unwrap();
        let Some(Value::Array(known)) = known.as_known() else { panic!("expected the known output of the split") };
        assert_eq!(observe(known), (mesh.devices().to_vec(), true, vec![8.0]));
        assert!(unknown.into_value().into_value().unwrap().is_unknown());
    }

    #[test]
    fn test_call_operation_eager_input_placement() {
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
        let derived = |shardings| CallOperation::<Type>::new(0).with_derived_input_shardings(shardings);

        // A call of a function compiled for an input type without a sharding places the input replicated, and an
        // override places it as it says. A call of a `jit` dispatcher and a call that a transform created without a
        // boundary for the input keep the input where it is, and a derived call with a boundary places it there.
        let placement = |call: &CallOperation<Type>, index| call.eager_input_placement(index, &array, &declared);
        assert_eq!(placement(&CallOperation::new(0), 0), Some(replicated.clone()));
        assert_eq!(
            placement(&CallOperation::new(0).with_boundary(Some(&mesh), vec![Some(sharded.clone())], Vec::new()), 0),
            Some(sharded.clone()),
        );
        assert_eq!(placement(&CallOperation::new(0).with_kept_input_placements(), 0), Some(sharded.clone()));
        assert_eq!(placement(&derived(vec![None]), 0), Some(sharded.clone()));
        assert_eq!(placement(&derived(vec![Some(replicated.clone())]), 0), Some(replicated));

        // A lifted capture of a call of a compiled function is placed by the executable's own rule.
        assert_eq!(placement(&CallOperation::new(1), 0), None);
    }

    #[test]
    fn test_call_operation_interpret() {
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
        // of the call (refer to `CallOperation::boundary_input_sharding`).
        let callee = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(Type::Array(ArrayType::new_static(DataType::I32, [4])));
            Arc::new(
                builder
                    .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
                    .unwrap(),
            )
        };
        let regions = AttachedRegions::Callees(std::slice::from_ref(&callee));
        let driver = EagerInterpretationDriver::new(&regions);
        let interpret = |call: CallOperation<Type>| {
            let outputs = call.interpret(&session, &driver, &[Value::Array(input.clone())]).unwrap();
            let [Value::Array(output)] = outputs.as_slice() else {
                panic!("expected one array output");
            };
            output.block_until_ready().unwrap();
            output.clone()
        };

        // Without boundary shardings, the identity callee returns its input replicated, as an execution of a function
        // compiled for that input type places its input, while a call of a `jit` dispatcher returns its input where
        // it is, as the executables of the dispatcher do.
        let output = interpret(CallOperation::new(0));
        assert_eq!(output.sharding(), &replicated);
        assert_eq!(shapes(&output), vec!["[4]", "[4]"]);
        assert_eq!(values(&output), vec![vec![1, 2, 3, 4]; 2]);
        let output = interpret(CallOperation::new(0).with_kept_input_placements());
        assert_eq!(output.sharding(), &sharded);
        assert_eq!(shapes(&output), vec!["[2]", "[2]"]);
        assert_eq!(values(&output), vec![vec![1, 2], vec![3, 4]]);

        // A replicated output override places the output of the callee, and a replicated input override places the
        // input that the callee returns, as executions of the called function place their outputs and inputs.
        for call in [
            CallOperation::new(0).with_boundary(Some(&mesh), Vec::new(), vec![Some(replicated.clone())]),
            CallOperation::new(0).with_boundary(Some(&mesh), vec![Some(replicated.clone())], Vec::new()),
        ] {
            let output = interpret(call);
            assert_eq!(output.sharding(), &replicated);
            assert!(output.is_committed());
            assert_eq!(shapes(&output), vec!["[4]", "[4]"]);
            assert_eq!(values(&output), vec![vec![1, 2, 3, 4]; 2]);
        }

        // A call whose boundary is on another single device moves an uncommitted input there, as an execution of its
        // callee moves its input, and commits the output, as that execution does, although the input already has
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
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(Type::Array(vector_type));
            Arc::new(
                builder
                    .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
                    .unwrap(),
            )
        };
        let regions = AttachedRegions::Callees(std::slice::from_ref(&callee));
        let call =
            CallOperation::new(0).with_boundary(Some(&second_mesh), Vec::new(), vec![Some(second_replicated.clone())]);
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
    fn test_call_operation_interpret_receives_inputs_and_returns_outputs_as_an_execution() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let session = Session::builder().client(client).build().unwrap();
        let matrix_type = ArrayType::new_static(DataType::F32, [4, 2]);
        let values = (1..=8).map(|value| value as f32).collect::<Vec<_>>();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 2);
        let split = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated],
        )
        .unwrap();
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        type Observation = Result<(usize, Option<Layout>, Vec<f32>), String>;
        let observe = |array: Array| -> Observation {
            let values = array.copy_to_host_bytes().map_err(|error| error.to_string())?;
            Ok((array.addressable_shards().count(), array.layout().cloned(), values_from_bytes::<f32>(&values)))
        };

        // An identity whose input override is the sharding of its input is applied directly, through a call, and
        // through the interpretation rule of its call in a caller, to an input that holds the buffers of only some of
        // its local devices. The rule supplies a missing buffer from a local replica of the same contribution, rejects
        // an input with a contribution that no local replica holds with the error of the execution, and returns an
        // output without the layout of its input, which the declaration of the output omits, as an execution of the
        // callee does.
        for sharding in [&replicated, &split] {
            let options = Options::new(mesh.clone()).with_input_shardings(vec![sharding.clone()]);
            let callee: CompiledFunction<ArrayType, ArrayType> =
                compile_with_options(|x: CompileTracer| x, matrix_type.clone(), &session, options).unwrap();
            let caller: StagedFunction<ArrayType, ArrayType> =
                stage(|x| callee.call(x).unwrap(), matrix_type.clone(), Options::new(mesh.clone())).unwrap();
            let program = caller.source_program().to_program_with_lifted_captures().unwrap();
            let region = program.entry_region_ref();
            let call = region
                .instructions()
                .iter()
                .find(|instruction| matches!(instruction.operation(), Operation::Call(_)))
                .unwrap();
            let Operation::Call(operation) = call.operation() else { unreachable!("selected a `call`") };
            let nested = Arc::new(region.with_id(call.regions()[0]).unwrap().to_program());
            let regions = AttachedRegions::Callees(std::slice::from_ref(&nested));
            for layout in [None, Some(column_major.clone())] {
                let input_type = matrix_type.clone().with_layout(layout.clone());
                let input = Array::from_host_buffer(
                    &session,
                    input_type.clone(),
                    mesh.clone(),
                    Some(sharding.clone()),
                    values_to_bytes(&values),
                )
                .unwrap();
                for buffer_count in [2, 1, 0] {
                    let buffers = input
                        .addressable_shards()
                        .take(buffer_count)
                        .map(|shard| {
                            let buffer = shard.buffer().unwrap();
                            buffer.copy_to_memory(buffer.memory().unwrap()).unwrap()
                        })
                        .collect();
                    let partial = Array::from_addressable_buffers(
                        &session,
                        input_type.clone(),
                        mesh.clone(),
                        Some(sharding.clone()),
                        buffers,
                    )
                    .unwrap();
                    let mut inputs = caller.source_program().captures().to_vec();
                    inputs.push(Value::Array(partial.clone()));
                    let executed = callee.executable().execute(&session, partial.clone());
                    let executed = executed.map_err(|error| error.to_string()).and_then(observe);
                    let called = callee.call(partial).map_err(|error| error.to_string()).and_then(observe);
                    let interpreted = operation
                        .interpret(&session, &EagerInterpretationDriver::new(&regions), &inputs)
                        .map_err(|error| error.to_string())
                        .and_then(|mut outputs| match outputs.remove(0) {
                            Value::Array(array) => observe(array),
                            _ => panic!("expected an array"),
                        });
                    let expected: Observation = match (sharding == &split, buffer_count) {
                        (_, 0) => Err("input array #0 has no addressable shard for device 0".to_string()),
                        (true, 1) => Err("input array #0 has no addressable shard for device 1".to_string()),
                        _ => Ok((2, None, values.clone())),
                    };
                    assert_eq!(
                        (executed, called, interpreted),
                        (expected.clone(), expected.clone(), expected),
                        "{sharding} {layout:?} {buffer_count}",
                    );
                }
            }
        }
    }

    #[test]
    fn test_call_operation_interpret_binds_the_captures_that_nested_regions_read() {
        // The body of a scan in the callee of a call reads a capture of the callee through a capture constant (refer to
        // `used_captures`), which the lowering of the whole callee binds to the input that the call receives
        // for the capture. The interpretation rule of the call returns what `.call` and an execution of the callee
        // return, on the same devices and with the same commitment, also for a capture that lives on another mesh of
        // the same devices in the opposite order.
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        let mesh = two_device_mesh(session.client(), MeshAxisType::Explicit);
        let reversed_mesh =
            DeviceMesh::new(mesh.logical_mesh().clone(), mesh.devices().iter().rev().copied().collect()).unwrap();
        let vector_type = ArrayType::new_static(DataType::F32, [4]);
        let observe = |output: Array| {
            let values = values_from_bytes::<f32>(&output.copy_to_host_bytes().unwrap());
            (output.r#type().into_owned(), output.mesh().devices().to_vec(), output.is_committed(), values)
        };
        for (capture_mesh, length) in [(&mesh, 1), (&mesh, 3), (&reversed_mesh, 1)] {
            let split = vec![ShardingDimension::sharded(["x"])];
            let sharding = Some(Sharding::new(capture_mesh.logical_mesh().clone(), split).unwrap());
            let bytes = values_to_bytes(&[1f32, 2.0, 3.0, 4.0]);
            let capture =
                Array::from_host_buffer(&session, vector_type.clone(), capture_mesh.clone(), sharding, bytes).unwrap();
            let input_type = ArrayType::new_static(DataType::F32, [length, 4]);
            let function: CompiledFunction<ArrayType, ArrayType> = compile(
                |stack: CompileTracer| {
                    let scanned = scan(
                        |_, (), slice| {
                            let capture = ImportingContext::<Session, ThroughCapture>::import(
                                slice.context(),
                                Value::Array(capture.clone()),
                            )?;
                            let slice = ValueProjection::<ArrayType>::into_projected(slice)?;
                            let capture = ValueProjection::<ArrayType>::into_projected(capture)?;
                            Ok(((), (slice + capture).into_value()))
                        },
                        (),
                        stack.into_value(),
                        ScanOptions::new(),
                    );
                    ValueProjection::<ArrayType>::into_projected(scanned.unwrap().1).unwrap()
                },
                input_type.clone(),
                &session,
                mesh.clone(),
            )
            .unwrap();
            let bytes = values_to_bytes(&vec![0f32; 4 * length]);
            let input = Array::from_host_buffer(&session, input_type.clone(), mesh.clone(), None, bytes).unwrap();
            let executed = function.executable().execute(&session, input.clone()).map(observe);
            let called = function.call(input.clone()).map(observe);
            let caller: StagedFunction<ArrayType, ArrayType> =
                stage(|x| function.call(x).unwrap(), input_type, Options::new(mesh.clone())).unwrap();
            let program = caller.source_program().to_program_with_lifted_captures().unwrap();
            let mut inputs = caller.source_program().captures().to_vec();
            inputs.push(Value::Array(input));
            let call = program
                .instructions()
                .iter()
                .find(|instruction| matches!(instruction.operation(), Operation::Call(_)))
                .unwrap();
            let Operation::Call(operation) = call.operation() else { unreachable!("selected a `call`") };
            let callee = Arc::new(program.entry_region_ref().with_id(call.regions()[0]).unwrap().to_program());
            let regions = AttachedRegions::Callees(std::slice::from_ref(&callee));
            let interpreted =
                operation
                    .interpret(&session, &EagerInterpretationDriver::new(&regions), &inputs)
                    .map(|mut outputs| match outputs.remove(0) {
                        Value::Array(output) => observe(output),
                        _ => panic!("expected an array"),
                    });
            let executed = executed.map_err(|error| error.to_string());
            let observed = executed.as_ref().map(|(_, devices, committed, values)| (devices, *committed, values));
            let values = [1f32, 2.0, 3.0, 4.0].repeat(length);
            assert_eq!(observed, Ok((&mesh.devices().to_vec(), true, &values)), "{length}");
            assert_eq!(
                (called.map_err(|error| error.to_string()), interpreted.map_err(|error| error.to_string())),
                (executed.clone(), executed),
                "{length} {:?}",
                capture_mesh.devices(),
            );
        }
    }

    #[test]
    fn test_call_operation_interpret_binds_the_call_through_the_supplied_driver() {
        /// Test-only interpretation driver that rejects every operation that it is asked to bind, naming the operation
        /// and the number of its regions, and interprets regions as the eager driver that it wraps interprets them.
        struct RejectingBindDriver<'r>(EagerInterpretationDriver<'r, Constant, Operation>);

        impl RegionDriver<Constant, Operation> for RejectingBindDriver<'_> {
            fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, Constant, Operation>>
            where
                Constant: 'r,
                Operation: 'r,
            {
                self.0.regions()
            }
        }

        impl InterpretationDriver<Session> for RejectingBindDriver<'_> {
            fn interpret_region(
                &self,
                context: &Session,
                index: usize,
                inputs: Vec<Value>,
            ) -> Result<Vec<Value>, ProgramError> {
                self.0.interpret_region(context, index, inputs)
            }

            fn bind(
                &self,
                context: &Session,
                operation: Operation,
                regions: AttachedRegions<'_, Constant, Operation>,
                inputs: &[Value],
            ) -> Result<Vec<Value>, ProgramError> {
                let _ = (context, inputs);
                let message =
                    format!("the driver rejects `{}` with region count {}", operation.name(), regions.region_count());
                Err(ProgramError::InvalidArgument { message })
            }
        }

        // The interpretation rule of a call binds the call, with its whole callee as its one region, through the driver
        // that it receives, which here rejects it, whether or not the body of a scan in the callee reads a capture of
        // the callee.
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        let mesh = two_device_mesh(session.client(), MeshAxisType::Explicit);
        let split = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let bytes = values_to_bytes(&[1f32, 2.0, 3.0, 4.0]);
        let vector_type = ArrayType::new_static(DataType::F32, [4]);
        let capture = Array::from_host_buffer(&session, vector_type, mesh.clone(), Some(split), bytes).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [1, 4]);
        for reads_capture in [false, true] {
            let function: CompiledFunction<ArrayType, ArrayType> = compile(
                |stack: CompileTracer| {
                    if !reads_capture {
                        return stack;
                    }
                    let scanned = scan(
                        |_, (), slice| {
                            let capture = ImportingContext::<Session, ThroughCapture>::import(
                                slice.context(),
                                Value::Array(capture.clone()),
                            )?;
                            let slice = ValueProjection::<ArrayType>::into_projected(slice)?;
                            let capture = ValueProjection::<ArrayType>::into_projected(capture)?;
                            Ok(((), (slice + capture).into_value()))
                        },
                        (),
                        stack.into_value(),
                        ScanOptions::new(),
                    );
                    ValueProjection::<ArrayType>::into_projected(scanned.unwrap().1).unwrap()
                },
                input_type.clone(),
                &session,
                mesh.clone(),
            )
            .unwrap();
            let caller: StagedFunction<ArrayType, ArrayType> =
                stage(|x| function.call(x).unwrap(), input_type.clone(), Options::new(mesh.clone())).unwrap();
            let program = caller.source_program().to_program_with_lifted_captures().unwrap();
            let call = program
                .instructions()
                .iter()
                .find(|instruction| matches!(instruction.operation(), Operation::Call(_)))
                .unwrap();
            let Operation::Call(operation) = call.operation() else { unreachable!("selected a `call`") };
            let callee = Arc::new(program.entry_region_ref().with_id(call.regions()[0]).unwrap().to_program());
            let regions = AttachedRegions::Callees(std::slice::from_ref(&callee));
            let bytes = values_to_bytes(&[0f32; 4]);
            let input = Array::from_host_buffer(&session, input_type.clone(), mesh.clone(), None, bytes).unwrap();
            let mut inputs = caller.source_program().captures().to_vec();
            inputs.push(Value::Array(input));
            let driver = RejectingBindDriver(EagerInterpretationDriver::new(&regions));
            let bound = operation.interpret(&session, &driver, &inputs).map_err(|error| error.to_string());
            let expected = Err("the driver rejects `call` with region count 1".to_string());
            assert_eq!(
                (inputs.len(), bound.map(|outputs| outputs.len())),
                (if reads_capture { 2 } else { 1 }, expected),
            );
        }
    }

    #[test]
    fn test_call_operation_interpret_rejects_callees_with_reference_state() {
        // A callee that allocates a reference and reads it carries reference state that the program of an eager call
        // cannot lower, so the interpretation rule of the call rejects it with the error of an eager call.
        let session = Session::builder().client(cpu_client(1)).build().unwrap();
        let scalar_type = ArrayType::scalar(DataType::F32);
        let callee = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(Type::Array(scalar_type.clone()));
            let operation = Operation::ReferenceNew(ReferenceNewOperation::new());
            let reference = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
            let operation = Operation::ReferenceRead(ReferenceReadOperation::new());
            let value = builder.add_instruction(operation, Vec::new(), vec![reference], None).unwrap()[0];
            Arc::new(
                builder
                    .build::<Vec<Constant>, Vec<Constant>>(vec![value], vec![Placeholder], vec![Placeholder])
                    .unwrap(),
            )
        };
        let regions = AttachedRegions::Callees(std::slice::from_ref(&callee));
        let input = Array::from_host_buffer(&session, scalar_type, None, None, 2f32.to_ne_bytes()).unwrap();
        let inputs = [Value::Array(input)];
        let call = CallOperation::new(0);
        let expected = "`call` carries reference state that XLA eager execution cannot lower; discharge references \
                        before lowering";
        let bound = session.bind(Operation::Call(call.clone()), AttachedRegions::Callees(&[callee.clone()]), &inputs);
        assert!(matches!(bound, Err(ProgramError::UnsupportedOperation { message }) if message == expected));
        let interpreted = call.interpret(&session, &EagerInterpretationDriver::new(&regions), &inputs);
        assert!(matches!(interpreted, Err(ProgramError::UnsupportedOperation { message }) if message == expected));
    }

    #[test]
    fn test_call_operation_interpret_fails_every_output_of_a_call_with_a_failed_input() {
        // A compiled identity of two inputs receives a failed first input and a healthy second one. Every output of an
        // execution carries the completion of the whole execution, so reading the healthy second output fails with the
        // failure of the first input on direct execution, on `.call`, and through the interpretation rule of a call of
        // the function. The first input is either of the zero data type, without buffers, with a failed producer, or an
        // ordinary `f32` produced by an execution whose input upload failed.
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let device = client.addressable_devices().unwrap()[0].clone();
        let (zero_type, scalar_type) = (ArrayType::scalar(DataType::Zero), ArrayType::scalar(DataType::F32));
        let (event, promise) = client.event(()).unwrap();
        promise.set(Some(ryft_pjrt::Error::aborted("producer failed"))).unwrap();
        let zero = Array::from_host_buffer(&session, zero_type.clone(), mesh.clone(), None, [])
            .unwrap()
            .with_execution_fence(ExecutionFence::new(vec![event]));
        let specification = BufferSpecification::new(BufferType::F32, Vec::<u64>::new());
        let manager = client
            .host_to_device_transfer_manager(vec![specification], device.default_memory().unwrap())
            .unwrap();
        let buffer = manager.retrieve_buffer(0).unwrap();
        let pending =
            Array::from_addressable_buffers(&session, scalar_type.clone(), mesh.clone(), None, vec![buffer]).unwrap();
        let double: CompiledFunction<ArrayType, ArrayType> =
            compile(|x: CompileTracer| x.clone() + x, scalar_type.clone(), &session, mesh.clone()).unwrap();
        let failed = double.executable().execute_async(&session, pending).unwrap().into_output();
        manager.set_error(0, ryft_pjrt::Error::aborted("upload failed")).unwrap();
        let scalar =
            Array::from_host_buffer(&session, scalar_type.clone(), mesh.clone(), None, 4f32.to_ne_bytes()).unwrap();
        // XLA prefixes the failure of an input with "Error dispatching computation: " once for each dispatch that reads
        // it, and whether a dispatch reads the failed input or the failure of the dispatch that produced it depends on
        // which of the two completes first, so the prefixes of XLA are removed and the failure itself is compared.
        let observe = |result: Result<Array, String>| {
            result
                .and_then(|result| result.copy_to_host_bytes().map_err(|error| error.to_string()))
                .map(|bytes| values_from_bytes::<f32>(&bytes))
                .map_err(|error| error.rsplit("Error dispatching computation: ").next().unwrap().to_string())
        };
        let mut observations = Vec::new();
        for (failed, failed_type) in [(zero, zero_type), (failed, scalar_type.clone())] {
            let types = (failed_type, scalar_type.clone());
            let identity = |inputs: (CompileTracer, CompileTracer)| inputs;
            let function: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> =
                compile(identity, types.clone(), &session, mesh.clone()).unwrap();
            let inputs = (failed.clone(), scalar.clone());
            let executed = function.executable().execute(&session, inputs.clone());
            let executed = observe(executed.map(|(_, output)| output).map_err(|error| error.to_string()));
            let called = observe(function.call(inputs).map(|(_, output)| output).map_err(|error| error.to_string()));
            let caller: StagedFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> =
                stage(|inputs| function.call(inputs).unwrap(), types, Options::new(mesh.clone())).unwrap();
            let program = caller.source_program().to_program_with_lifted_captures().unwrap();
            let call = program
                .instructions()
                .iter()
                .find(|instruction| matches!(instruction.operation(), Operation::Call(_)))
                .unwrap();
            let Operation::Call(operation) = call.operation() else { unreachable!("selected a `call`") };
            let callee = Arc::new(program.entry_region_ref().with_id(call.regions()[0]).unwrap().to_program());
            let regions = AttachedRegions::Callees(std::slice::from_ref(&callee));
            let inputs = [Value::Array(failed), Value::Array(scalar.clone())];
            let interpreted = operation
                .interpret(&session, &EagerInterpretationDriver::new(&regions), &inputs)
                .map_err(|error| error.to_string())
                .map(|mut outputs| match outputs.remove(1) {
                    Value::Array(output) => output,
                    _ => panic!("expected an array"),
                });
            observations.push((executed, called, observe(interpreted)));
        }
        let (zero_failure, upload_failure) = (Err("producer failed".to_string()), Err("upload failed".to_string()));
        assert_eq!(
            observations,
            vec![
                (zero_failure.clone(), zero_failure.clone(), zero_failure),
                (upload_failure.clone(), upload_failure.clone(), upload_failure),
            ],
        );
    }

    #[test]
    fn test_call_operation_interpret_runs_the_callee_on_the_devices_of_the_call() {
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
                .find(|instruction| matches!(instruction.operation(), Operation::Call(_)))
                .unwrap();
            let Operation::Call(operation) = call.operation() else { unreachable!("selected a `call`") };
            let callee = Arc::new(region.with_id(call.regions()[0]).unwrap().to_program());
            let regions = AttachedRegions::Callees(std::slice::from_ref(&callee));
            operation.interpret(&session, &EagerInterpretationDriver::new(&regions), inputs)
        };

        // A callee compiled for the second device that places its input there and fills a value that depends on no
        // input returns both on that device, committed, when the call is interpreted, as an execution of the callee
        // returns them, rather than filling the value on the default device of the session.
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
        // input is and commits both outputs, because the callee requests a placement. The interpreted call fills the
        // value there too, rather than on the default device of the session, and commits both outputs whatever the
        // commitment of the input, as an execution of the callee on those devices does.
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

        // A call of a callee that names no devices rejects committed inputs on different devices from one another
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
        let call = region.instructions().iter().find(|i| matches!(i.operation(), Operation::Call(_))).unwrap();
        let Operation::Call(operation) = call.operation() else { unreachable!("selected a `call`") };
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
                        "public input #1 is committed to devices [{}], but public input #0 is committed to devices \
                         [{}]",
                        second_device.id(),
                        first_mesh.devices()[0].id(),
                    ),
                })
                .to_string(),
            },
        );
    }

    #[test]
    fn test_call_operation_lower_constrains_boundary_shardings() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let unconstrained = Sharding::new(mesh.clone(), vec![ShardingDimension::Unconstrained]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let device_mesh = DeviceMesh::new(mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)]).unwrap();
        let array_type = test_vector_type(4).with_sharding(replicated.clone()).unwrap();
        let callee = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(array_type.clone().into());
            let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
            Arc::new(
                builder
                    .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                    .unwrap(),
            )
        };
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(array_type.clone().into());
        let callee_region = builder.intern_callee(&callee, None).unwrap();
        let call = CallOperation::new(0).with_boundary(
            Some(&device_mesh),
            vec![Some(sharded.clone())],
            vec![Some(replicated.clone())],
        );
        let called = builder.add_instruction(Operation::Call(call), vec![callee_region], vec![input], None).unwrap()[0];
        let call = CallOperation::new(0).with_boundary(Some(&device_mesh), Vec::new(), vec![Some(unconstrained)]);
        let output =
            builder.add_instruction(Operation::Call(call), vec![callee_region], vec![called], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // Each input and output of a call that its callee boundary overrides is constrained to the override where the
        // call executes, closed over auto mesh axes and with unconstrained dimensions replicated, as an execution of
        // the called function annotates its own boundary. An input without an input override is constrained to
        // the replicated sharding that the callee declares for it, because its type cannot show its placement along
        // the auto axis (refer to `CallOperation::boundary_input_sharding`). The shared callee itself carries no
        // override.
        let input_types = vec![array_type.clone()];
        let output_types = vec![array_type];
        let shardings = [replicated];
        let stablehlo = program
            .to_rendered_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_input_types(&input_types)
                    .with_output_types(&output_types)
                    .without_boundary_shardings()
                    .with_exact_input_shardings(shardings.iter().cloned().map(Some).collect())
                    .with_exact_output_shardings(shardings.iter().cloned().map(Some).collect()),
            )
            .unwrap();
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func private @callee_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.sine %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{"x"}]> : tensor<4xf32>
                    %1 = call @callee_0(%0) : (tensor<4xf32>) -> tensor<4xf32>
                    %2 = sdy.sharding_constraint %1 <@mesh, [{}]> : tensor<4xf32>
                    %3 = sdy.sharding_constraint %2 <@mesh, [{}]> : tensor<4xf32>
                    %4 = call @callee_0(%3) : (tensor<4xf32>) -> tensor<4xf32>
                    %5 = sdy.sharding_constraint %4 <@mesh, [{}]> : tensor<4xf32>
                    return %5 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_call_operation_lower_constrains_returned_cotangents_to_input_overrides() {
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
        // structurally zero one, which the transposed callee does not return. No type records a split over auto mesh
        // axes, so only the runtime sharding of the executed gradient shows it.
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
        }
    }

    #[test]
    fn test_call_operation_lower_places_cotangents_at_the_cotangent_shardings_of_overrides() {
        // A function that moves its scalar input to device memory and returns it, compiled with an inferred boundary,
        // has the same derivatives when that boundary is stated as an input override, an output override, or both,
        // for every reduction state of its input. A tangent has the sharding of its primal, while the cotangent of a
        // value with a pending sum has a completed one and the reverse (refer to `Sharding::cotangent`), so the
        // transposed call places each cotangent that crosses the boundary at the cotangent sharding of the override of
        // its primal, and keeps each known primal value at the override itself. The inferred boundary of an input is
        // the sharding of its type, which records reduction state over explicit axes only, so over auto axes only the
        // runtime shardings and values that the session-level derivative paths return show it.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let describe = |array: Array| -> Result<String, Error> {
            let values = array.copy_to_host()?.elements::<f32>()?;
            Ok(format!("{} {values:?}", array.sharding()))
        };
        let columns = ["JVP", "eager VJP", "gradient", "gradient of a caller", "second order gradient"];
        let mut outcomes = Vec::new();
        for axis_type in [MeshAxisType::Auto, MeshAxisType::Explicit] {
            let mesh = two_device_mesh(client, axis_type);
            let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 0);
            let pending = replicated.clone().with_unreduced_axes(["x"]).unwrap();
            let states = [
                ("no state", replicated.clone()),
                ("pending sum", pending.clone()),
                ("completed sum", replicated.clone().with_reduced_axes(["x"]).unwrap()),
                ("pending maximum", pending.with_unreduced_kind(UnreducedKind::Max).unwrap()),
            ];
            for (state, sharding) in states {
                let r#type = ArrayType::scalar(DataType::F32).with_sharding(sharding.without_auto_axes()).unwrap();
                let inferred = r#type.sharding().unwrap().clone();
                let shards =
                    mesh.devices().iter().map(|device| (device.id(), Arc::<[u8]>::from(1f32.to_ne_bytes().as_slice())));
                let input =
                    Array::from_host_shards(&session, r#type.clone(), mesh.clone(), Some(sharding), shards).unwrap();
                let derivatives = |options: Options| {
                    let function: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
                        |x: CompileTracer| x.place(&Memory::Device, None).unwrap(),
                        r#type.clone(),
                        &session,
                        options,
                    )
                    .unwrap();
                    let caller = || -> Result<CompiledFunction<ArrayType, ArrayType>, Error> {
                        compile(|x| function.call(x).unwrap(), r#type.clone(), &session, mesh.clone())
                    };
                    [
                        function
                            .jvp(&session)
                            .and_then(|jvp| jvp.executable().execute(&session, (input.clone(), input.clone())))
                            .and_then(|(_, tangent)| describe(tangent)),
                        session
                            .differentiate_at(Value::Array(input.clone()))
                            .vjp(|x| Ok(function.call(ValueProjection::<ArrayType>::into_projected(x)?)?.into_value()))
                            .and_then(|(output, pullback)| Ok(pullback.apply_materialized(output)?))
                            .map_err(Error::from)
                            .and_then(|cotangent| describe(ValueProjection::<ArrayType>::into_projected(cotangent)?)),
                        function
                            .gradient(&session)
                            .and_then(|gradient| gradient.executable().execute(&session, input.clone()))
                            .and_then(describe),
                        caller()
                            .and_then(|caller| caller.gradient(&session))
                            .and_then(|gradient| gradient.executable().execute(&session, input.clone()))
                            .and_then(describe),
                        caller()
                            .and_then(|caller| caller.gradient(&session))
                            .and_then(|gradient| gradient.gradient(&session))
                            .and_then(|gradient| gradient.executable().execute(&session, input.clone()))
                            .and_then(describe),
                    ]
                    .map(|derivative| derivative.map_err(|error| error.to_string()))
                };
                let options = Options::new(mesh.clone());
                let references = derivatives(options.clone());
                let boundaries = [
                    ("input override", options.clone().with_input_shardings(vec![inferred.clone()])),
                    ("output override", options.clone().with_output_shardings(vec![inferred.clone()])),
                    (
                        "input and output overrides",
                        options.with_input_shardings(vec![inferred.clone()]).with_output_shardings(vec![inferred]),
                    ),
                ];
                for (boundary, options) in boundaries {
                    for ((column, derivative), reference) in columns.iter().zip(derivatives(options)).zip(&references) {
                        let outcome = match derivative == *reference {
                            true => "agrees".to_string(),
                            false => format!("has {derivative:?} instead of {reference:?}"),
                        };
                        outcomes.push(format!("{axis_type:?} {state}, {boundary}, {column}: {outcome}"));
                    }
                }
            }
        }
        let mut expected = Vec::new();
        for axis_type in [MeshAxisType::Auto, MeshAxisType::Explicit] {
            for state in ["no state", "pending sum", "completed sum", "pending maximum"] {
                for boundary in ["input override", "output override", "input and output overrides"] {
                    for column in columns {
                        expected.push(format!("{axis_type:?} {state}, {boundary}, {column}: agrees"));
                    }
                }
            }
        }
        assert_eq!(outcomes.join("\n"), expected.join("\n"));
    }

    #[test]
    fn test_call_operation_lower_rejects_boundaries_that_introduce_pending_extrema() {
        // XLA aborts the process when it exports a sharding that introduces a pending maximum or minimum, so the
        // boundary of a nested call rejects one that its output does not carry, as the result shardings of an entry
        // function do. The caller uses the output of the call, so the override does not reach its own result shardings.
        // An input override introduces nothing: the callee holds its input as it has entered, with the pending
        // reduction over auto mesh axes that the override states completed (refer to
        // `Sharding::without_auto_reduction_axes`), and the input of the call is a completed value of the caller, so
        // the call lowers and constrains the input replicated. Lowering alone is checked here, so that a missing
        // rejection cannot abort the test process.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let vector_type = ArrayType::new_static(DataType::F32, [4]);
        let pending = |mesh: &DeviceMesh, kind: UnreducedKind| {
            Sharding::replicated(mesh.logical_mesh().clone(), 1)
                .with_unreduced_axes(["x"])
                .unwrap()
                .with_unreduced_kind(kind)
                .unwrap()
        };
        let lower = |inner: StagedFunction<ArrayType, ArrayType>, options: Options, adds: bool| {
            let outer: StagedFunction<ArrayType, ArrayType> = stage(
                move |x: CompileTracer| {
                    let y = inner.call(x).unwrap();
                    if adds { y.clone() + y } else { y }
                },
                vector_type.clone(),
                options,
            )
            .unwrap();
            outer.into_inner().lower(&session).map(|lowered| lowered.lowered_program().stable_hlo().to_string())
        };

        // A type never records reduction state over an auto mesh axis, so no value carries one there.
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        for kind in [UnreducedKind::Max, UnreducedKind::Min] {
            for overrides_output in [true, false] {
                let options = Options::new(mesh.clone());
                let options = match overrides_output {
                    true => options.with_output_shardings(vec![pending(&mesh, kind)]),
                    false => options.with_input_shardings(vec![pending(&mesh, kind)]),
                };
                let inner = stage(|x| x, vector_type.clone(), options).unwrap();
                let options = Options::new(mesh.clone()).with_input_shardings(vec![replicated.clone()]);
                let lowered = lower(inner, options, true).map_err(|error| error.to_string());
                match overrides_output {
                    true => assert_eq!(
                        lowered,
                        Err(format!(
                            "cannot lower `call`: the sharding `{{mesh<['x'=2:auto]>, [{{}}], unreduced={{'x'}}, \
                             unreduced_kind={kind}}}` declares a pending `{kind}` reduction over mesh axes (i.e., \
                             `x`) along which the value does not carry one, and XLA can only introduce a pending \
                             `sum` reduction (types never record reduction state over auto mesh axes, so a pending \
                             `{kind}` reduction over them is never carried)",
                        )),
                    ),
                    false => assert_eq!(
                        lowered.as_deref(),
                        Ok(indoc! {r#"
                            module {
                              sdy.mesh @mesh = <["x"=2]>
                              func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}]>}) {
                                %0 = sdy.sharding_constraint %arg0 <@mesh, [{}]> : tensor<4xf32>
                                %1 = stablehlo.add %0, %0 : tensor<4xf32>
                                return %1 : tensor<4xf32>
                              }
                            }
                        "#}),
                    ),
                }
            }
        }

        // A boundary that keeps the pending maximum that its input and its output carry over an explicit mesh axis
        // introduces nothing (the caller returns the output as it is, because `add` takes no pending maximum).
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let maximum = pending(&mesh, UnreducedKind::Max);
        let typed = vector_type.clone().with_sharding(maximum.clone()).unwrap();
        let options = Options::new(mesh.clone()).with_output_shardings(vec![maximum.clone()]);
        let inner = stage(|x| x, typed, options).unwrap();
        let options = Options::new(mesh.clone()).with_input_shardings(vec![maximum]);
        assert!(lower(inner, options, false).is_ok());
    }

    #[test]
    fn test_call_operation_lower_ignores_nested_call_capture_namespaces() {
        // The inner callee establishes its own two-slot capture namespace. Its first two inputs are its lifted
        // captures, and its body resolves the constant `Captured(1)` against that prefix. Deriving the OUTER call's
        // capture prefix from capture-constant indices anywhere in the callee arena would conflate the namespaces
        // and reject this program with a false missing-capture error, because the operation payload is authoritative.
        let array_type = test_vector_type(4);
        let inner = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
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
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(array_type.clone().into());
            let callee_region = builder.intern_callee(&inner, None).unwrap();
            let output = builder
                .add_instruction(Operation::Call(CallOperation::new(2)), vec![callee_region], vec![input, input], None)
                .unwrap()[0];
            Arc::new(
                builder
                    .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                    .unwrap(),
            )
        };
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(array_type.clone().into());
        let callee_region = builder.intern_callee(&outer, None).unwrap();
        let output = builder
            .add_instruction(Operation::Call(CallOperation::new(0)), vec![callee_region], vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let stablehlo = program.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap();
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
    fn test_call_operation_lower_executes_the_halves_of_a_linearized_call() {
        // The linearization of a program that holds one `call` of `f = |x| x.sin()` keeps a primal `call`, which
        // returns the primal output followed by the residuals, and a tangent `call`, which consumes the input tangent
        // followed by the residuals. Staging both calls into a compiled `(x, dx) -> (primal, tangent)` function and
        // executing it on XLA returns `(sin(x), cos(x) * dx)` within `1e-5`, which only the lowering and execution of
        // the two call boundaries show.
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let engine = Session::builder().client(client).build().unwrap();

        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let input_type = ArrayType::new(DataType::F32, Shape::new(Vec::new())).with_sharding(sharding).unwrap();
        let inner: CompiledFunction<ArrayType, ArrayType> =
            compile(|x| x.sin().unwrap(), input_type.clone(), &engine, mesh.clone()).unwrap();
        let inner_for_outer = inner.clone();
        let outer: StagedFunction<ArrayType, ArrayType> =
            stage(move |x| inner_for_outer.call(x).unwrap(), input_type.clone(), Options::new(mesh.clone())).unwrap();
        let linearization = outer.source_program().program().to_flat_program().linearize().unwrap();
        let (primal_half, tangent_half, _, _) = linearization.into_parts();
        let jvp_compiled: CompiledFunction<(ArrayType, ArrayType), (ArrayType, ArrayType)> = compile(
            move |(primal_input, tangent_input)| {
                let context = primal_input.value().context().clone();
                let mut primal_outputs = context
                    .stage_operation(
                        Operation::Call(CallOperation::new(0)),
                        AttachedRegions::Callees(std::slice::from_ref(&primal_half)),
                        &[primal_input.into_value()],
                    )
                    .unwrap();
                let residuals = primal_outputs.split_off(1);
                let primal_output = primal_outputs.remove(0);
                let mut tangent_inputs = vec![tangent_input.into_value()];
                tangent_inputs.extend(residuals);
                let tangent_output = context
                    .stage_operation(
                        Operation::Call(CallOperation::new(0)),
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
    fn test_call_operation_lower_rounds_values_saved_on_residual_demand() {
        // The function `(x, s) ↦ w · s` calls `w = f32(t)` for `t = tag(bf16(x) · bf16(x), "t")`, rematerialized under
        // a policy that saves `t`. Its linearization with respect to `s` splits the call into a known call that saves
        // `t` and a replay call that computes `w` from the saved `t`, and the known callee rounds `t` right after its
        // producer, so the output and its derivative with respect to `s` (both `w` for `s = 1`) agree. XLA computes the
        // `bf16` product in `f32`, so only an execution through XLA shows that rounding. Without it, the output would
        // use `t` at `f32` precision (1.21343994 for `x = 1.1`) while the derivative used the saved `bf16` value
        // (1.2109375).
        let scalar_type = Type::from(ArrayType::scalar(DataType::F32));
        let callee = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
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
        let body = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let x = builder.add_input(scalar_type.clone());
            let s = builder.add_input(scalar_type.clone());
            let w = add_call(&mut builder, &callee, vec![x]);
            let y = builder.add_instruction(MulOperation::new(), Vec::new(), vec![w, s], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![y], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let policy = ResidualPolicyReference::new(SaveOnlyTheseNames::new(["t"]));
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let inputs = vec![builder.add_input(scalar_type.clone()), builder.add_input(scalar_type)];
        let body = builder.import_program(body);
        let output = builder.add_instruction(RematerializeOperation::new(policy), vec![body], inputs, None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let linearization = program.linearize_with_respect_to(&[1]).unwrap();

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
    fn test_call_operation_lower_orders_the_effects_of_split_callees() {
        // The callee `(k, x) ↦ print(x · print(k · k))` has an ordered print in each partition. Executing the known
        // program of its split prints the known print, and executing its residual program prints the residual print.
        // A prefix print of the unknown input keeps the known print in the residual program, which then prints all
        // three in program order. Print order is observable only through the print handler of an XLA execution.
        let scalar_type: Type = ArrayType::scalar(DataType::F32).into();
        let callee = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let known = builder.add_input(scalar_type.clone());
            let unknown = builder.add_input(scalar_type.clone());
            let squared =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![known, known], None).unwrap()[0];
            let print = PrintOperation::<ArrayType>::new("known").with_effect_class(EffectClass::OrderedIo);
            let coefficient = builder.add_instruction(print, Vec::new(), vec![squared], None).unwrap()[0];
            let product =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![unknown, coefficient], None).unwrap()[0];
            let print = PrintOperation::<ArrayType>::new("residual").with_effect_class(EffectClass::OrderedIo);
            let output = builder.add_instruction(print, Vec::new(), vec![product], None).unwrap()[0];
            Arc::new(
                builder
                    .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                    .unwrap(),
            )
        };
        let program = |prefix: bool| {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
            let known = builder.add_input(scalar_type.clone());
            let unknown = builder.add_input(scalar_type.clone());
            if prefix {
                builder
                    .add_instruction(PrintOperation::<ArrayType>::new("before"), Vec::new(), vec![unknown], None)
                    .unwrap();
            }
            let output = add_call(&mut builder, &callee, vec![known, unknown]);
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let policy = ResidualPolicyReference::new(NothingSavable);
        let warm = program(false).partition_with_residual_policy(&[true, false], &policy).unwrap();
        let partition = program(true).partition_with_residual_policy(&[true, false], &policy).unwrap();

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
    fn test_call_operation_lower_runs_the_effects_of_outputless_callees() {
        // The split of a call whose callee prints its unknown input has a residual program without outputs, and each
        // execution of that program runs the print. Prints are observable only through the print handler of an XLA
        // execution.
        let scalar_type: Type = ArrayType::scalar(DataType::F32).into();
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
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
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let known = builder.add_input(scalar_type.clone());
        let unknown = builder.add_input(scalar_type);
        let output = add_call(&mut builder, &callee, vec![known, unknown]);
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let partition = program.partition(&[true, false]).unwrap();
        let client = execution_client();
        ensure_print_handler_registered(client).unwrap();
        for unknown in [3.0, 5.0] {
            let (_, lines) =
                with_captured_prints(|| execute_f32_program(client, partition.residual_program(), vec![vec![unknown]]));
            assert_eq!(lines, vec![format!("residual: {unknown:?}")]);
        }
    }

    #[test]
    fn test_call_operation_lower_runs_the_backward_effects_of_transposed_callees() {
        // The callee maps its input through a reverse mode carrier whose backward rule prints its seed and returns a
        // zero. With the output of the call dead, its only seed is a structural zero, and executing the pullback still
        // runs the backward print on a materialized zero seed. Prints are observable only through the print handler of
        // an XLA execution.
        let scalar_type = Type::Array(ArrayType::scalar(DataType::F64));
        let backward = {
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
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
            let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
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
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let input = builder.add_input(scalar_type);
        builder
            .add_instruction(Operation::Call(CallOperation::new(0)), vec![callee], vec![input], None)
            .unwrap();
        let program = builder.build::<Vec<Constant>, Vec<Constant>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let pullback = program.entry_region_ref().transpose(&[0], &[], &[]).unwrap().into_program();
        let client = execution_client();
        ensure_print_handler_registered(client).unwrap();
        let (_, lines) = with_captured_prints(|| execute_mixed_program(client, &pullback, &[], &[]).unwrap());
        assert_eq!(lines, vec!["backward: 0.0"]);
    }

    #[test]
    fn test_call_operation_partially_evaluate_keeps_callees_with_capture_constants_whole() {
        // A call whose callee returns a retained capture constant keeps its boundary when every input is known, so the
        // known program is the whole program and the residual program is empty. Only the constants of this crate can
        // retain a capture (the array values of `ryft-core` report no capture index), so this case is tested here.
        let predicate_type = Type::from(ArrayType::scalar(DataType::Boolean));
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        builder.add_input(predicate_type.clone());
        let captured = builder.add_constant(Constant::Captured(CaptureReference::new(0, predicate_type.clone())));
        let callee = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![captured], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(predicate_type);
        let callee = builder.import_program(callee);
        let outputs = builder
            .add_instruction(Operation::Call(CallOperation::new(1)), vec![callee], vec![input], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder])
            .unwrap();
        let partition = program
            .partition_with_residual_policy(&[true], &ResidualPolicyReference::new(NothingSavable))
            .unwrap();
        assert_eq!(partition.known_program().to_string(), program.to_string());
        assert_eq!(partition.residual_program().instructions().len(), 0);
    }

    #[test]
    fn test_call_operation_split_on_residual_demand_saves_outputs_of_callees_with_capture_constants() {
        // The callee computes a matrix product that `DotsSavable` saves, so replaying its call would recompute a saved
        // value. A capture constant keeps the boundary of the call, so the residual planner saves the demanded output
        // of the call instead of splitting it. Only the constants of this crate can retain a capture (the array values
        // of `ryft-core` report no capture index), so this case is tested here.
        let matrix_type = Type::from(ArrayType::new_static(DataType::F32, [2, 2]));
        let dot = DotOperation::new(DotDimensionNumbers::new(vec![1], vec![0], vec![], vec![]));
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        builder.add_input(matrix_type.clone());
        let input = builder.add_input(matrix_type.clone());
        let captured = builder.add_constant(Constant::Captured(CaptureReference::new(0, matrix_type.clone())));
        let product = builder.add_instruction(dot, Vec::new(), vec![input, captured], None).unwrap()[0];
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![product], None).unwrap()[0];
        let callee = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let inputs = vec![builder.add_input(matrix_type.clone()), builder.add_input(matrix_type.clone())];
        let tangent = builder.add_input(matrix_type);
        let callee = builder.import_program(callee);
        let called =
            builder.add_instruction(Operation::Call(CallOperation::new(1)), vec![callee], inputs, None).unwrap()[0];
        let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![called, tangent], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let partition = program
            .partition_with_residual_policy(&[true, true, false], &ResidualPolicyReference::new(DotsSavable))
            .unwrap();
        assert_eq!(partition.known_program().instructions().len(), 1);
        assert!(matches!(partition.known_program().instructions()[0].operation(), Operation::Call(_)));
        assert_eq!(partition.known_program().output_ids(), partition.known_program().instructions()[0].outputs());
    }
}
