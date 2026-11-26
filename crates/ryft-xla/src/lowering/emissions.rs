//! Decisions about what lowering emits of a program (i.e., which regions it emits for a target platform, and which
//! captures and session constants those regions read).

use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use ryft_core::{
    ArrayType, AtomId, Instruction, Parameterized, Program, RegionRef, RegionRole, ScanOperation, Typed, TypedProgram,
    Value,
};

use crate::lowering::LowerableOperationFamily;
use crate::lowering::types::stable_hlo_dynamic_dimension_bound;
use crate::operations::call::CallFunctionMap;
use crate::{CallOperation, Constant, RetainedArray, Type};

// TODO(eaplatanios): Review this module.

/// Returns `true` if the length of `scan` admits an iteration, which is false for a static length of zero and for a
/// dynamic length whose bounds admit only zero. The lowering of a scan that admits no iteration emits nothing of its
/// body, and this is the one decision of that, which the walks that follow the regions that lowering emits read
/// (refer to `omits_regions`).
pub(crate) fn admits_iterations(scan: &ScanOperation<Type>) -> bool {
    let length = scan.iteration_count();
    length.value() != Some(0) && stable_hlo_dynamic_dimension_bound(length) != Some(0)
}

/// Returns `true` if lowering `operation` at the provided boundary never reads its attached regions, because it is a
/// named composite call, which `LowerableOperation::composite_key` identifies, that a backend implementation replaces
/// or whose body lowers to a shared function. The lowering loops skip materializing the regions of such instructions,
/// which would otherwise copy the body per call.
pub(crate) fn lowers_without_regions<O: LowerableOperationFamily>(
    operation: &O,
    input_types: &[Type],
    output_types: &[Type],
    nested_functions: Option<&Rc<CallFunctionMap<O>>>,
    target_platform: Option<&str>,
) -> bool {
    let Some(key) = operation.composite_key() else {
        return false;
    };
    O::has_composite_implementation(key, input_types, output_types, target_platform)
        || nested_functions.is_some_and(|functions| functions.named(key, input_types, output_types).is_some())
}

/// Returns `true` if lowering `instruction` of `region` for `target_platform` emits none of its regions, which holds
/// for a named composite call whose backend implementation replaces the whole call for that target (refer to
/// `LowerableOperation::has_composite_implementation`) and, on every target, for a scan whose length admits no iteration (refer
/// to [`admits_iterations`], which its lowering reads). This is the one test of whether lowering for a target emits the
/// regions of an instruction, which the walks that follow the regions that lowering emits read: the collector of the
/// functions that calls share (refer to `collect_call_functions`), the traversal of the regions that lowering emits
/// (refer to `emitted_regions`, which compilation admission also reads), the summaries of the placement requests and of
/// the session constants of a region, and the validation of the placements of a compiled program (refer to
/// `validate_placements`), so that a placement request or a captured array inside a region that lowering omits is never
/// validated, placed, or passed. The eager application of an operation asks the same question of that operation (refer
/// to [`omits_operation_regions`]).
pub(crate) fn omits_regions<V: Value<Type = Type>, O: LowerableOperationFamily>(
    region: RegionRef<'_, V, O>,
    instruction: &Instruction<O>,
    target_platform: Option<&str>,
) -> bool {
    let types = |atoms: &[AtomId]| {
        atoms.iter().map(|atom| region.atoms()[atom.index()].r#type().into_owned()).collect::<Vec<_>>()
    };
    omits_operation_regions(
        instruction.operation(),
        || (types(instruction.inputs()), types(instruction.outputs())),
        target_platform,
    )
}

/// Returns `true` if lowering `operation` for `target_platform` emits none of its regions. This is the part of
/// [`omits_regions`] that reads only the operation, its input and output types, which `boundary_types` returns when a
/// named composite call needs them, and the platform, so that the eager application of an operation (refer to
/// `Session::bind`) asks the same question of the operation that it binds as the walks of the regions that lowering
/// emits ask of each instruction.
pub(crate) fn omits_operation_regions<O: LowerableOperationFamily>(
    operation: &O,
    boundary_types: impl FnOnce() -> (Vec<Type>, Vec<Type>),
    target_platform: Option<&str>,
) -> bool {
    if operation.projected_payload::<ScanOperation<Type>>().is_some_and(|scan| !admits_iterations(scan)) {
        return true;
    }
    let Some(key) = operation.composite_key() else {
        return false;
    };
    let (input_types, output_types) = boundary_types();
    O::has_composite_implementation(key, &input_types, &output_types, target_platform)
}

/// Returns `entry` and the computation regions below it that lowering emits for `target_platform`, visiting shared
/// regions once, in the order of [`RegionRef::computation_regions`]. Dormant rule regions (e.g., the transpose region
/// of a `linear_call`) and the regions of the instructions whose lowering omits them (refer to [`omits_regions`]) are
/// skipped, so that a region reachable only through them is never visited. This is the one traversal of the regions
/// that lowering emits, which the collection of the module mesh and of the session constants and the selection of the
/// partitioner read, and it visits the same regions in the same order as [`RegionRef::computation_regions`] when no
/// instruction omits its regions.
pub(crate) fn emitted_regions<'r, V: Value<Type = Type>, O: LowerableOperationFamily>(
    entry: RegionRef<'r, V, O>,
    target_platform: Option<&'r str>,
) -> impl Iterator<Item = RegionRef<'r, V, O>> {
    emitted_regions_entering_calls(entry, target_platform, true)
}

/// Returns the regions of [`emitted_regions`], without the callees of the `call`s among them and the regions below
/// those callees unless `enters_calls` is set. Without them, these are the regions of one capture scope: a constant of
/// one of them that names capture `k` names capture `k` of `entry`, whereas the same constant in a callee names its own
/// input `k` (refer to `lower_call`), which the call receives as its operand `k`.
pub(crate) fn emitted_regions_entering_calls<'r, V: Value<Type = Type>, O: LowerableOperationFamily>(
    entry: RegionRef<'r, V, O>,
    target_platform: Option<&'r str>,
    enters_calls: bool,
) -> impl Iterator<Item = RegionRef<'r, V, O>> {
    let mut visited = vec![false; entry.arena().len()];
    let mut pending = vec![entry.id()];
    std::iter::from_fn(move || {
        while let Some(id) = pending.pop() {
            if std::mem::replace(&mut visited[id.index()], true) {
                continue;
            }
            // Every pending identifier is the entry or a region of an instruction of the same arena.
            let region = entry.with_id(id).unwrap();
            for instruction in region.instructions() {
                if !omits_regions(region, instruction, target_platform)
                    && (enters_calls || instruction.operation().projected_payload::<CallOperation<Type>>().is_none())
                {
                    pending.extend(instruction.regions().iter().copied().enumerate().filter_map(|(index, id)| {
                        (instruction.operation().region_role(index) == Some(RegionRole::Computation)).then_some(id)
                    }));
                }
            }
            return Some(region);
        }
        None
    })
}

/// Returns, for each of the first `capture_count` captures of the program whose entry region is `entry`, whether a
/// region that lowering emits for `target_platform` reads it (refer to [`emitted_regions`]), as an operand of one of
/// its instructions or as one of its outputs. Every region names a capture through a constant whose
/// [`Value::capture_index`] is its position, except the entry region of a program with lifted captures (i.e., when
/// `lifted` is set), which names capture `k` through its input `k`. A capture that no emitted region reads (e.g., one
/// that only the body of a scan that admits no iteration reads) is neither admitted when a function is compiled nor
/// passed to its executable, whose signature erases it (refer to `lower_mlir_module_for_program_with_effect_boundary`),
/// so that admission and lowering read this one decision. A capture that a region only passes on to a call as a
/// capture which the callee never reads is not read either (refer to [`visit_emitted_reads`]). The regions of the
/// callees of calls are not visited, because their constants name the captures of those callees, which a callee reads
/// only through the operands of its call (refer to [`emitted_regions_entering_calls`]).
pub(crate) fn emitted_capture_reads<'r, O: LowerableOperationFamily>(
    entry: RegionRef<'r, Constant, O>,
    capture_count: usize,
    lifted: bool,
    target_platform: Option<&'r str>,
) -> Vec<bool> {
    let mut is_read = vec![false; capture_count];
    let entry_id = entry.id();
    for region in emitted_regions_entering_calls(entry, target_platform, false) {
        let mut lifted_captures = HashMap::new();
        if lifted && region.id() == entry_id {
            let inputs = region.input_ids().iter().take(capture_count);
            lifted_captures.extend(inputs.enumerate().map(|(index, id)| (*id, index)));
        }
        visit_emitted_reads(region, target_platform, |id, is_live| {
            if !is_live {
                return;
            }
            let index = lifted_captures.get(&id).copied().or_else(|| {
                region.atoms().get(id.index()).and_then(|atom| atom.as_constant()).and_then(Value::capture_index)
            });
            if let Some(index) = index.filter(|index| *index < capture_count) {
                is_read[index] = true;
            }
        });
    }
    is_read
}

/// Calls `visit` with every atom that `region` reads, as an operand of one of its instructions or as one of its
/// outputs, and with whether that read is live. Every read is live except an operand that a `call` passes as a
/// lifted capture which its callee never reads (refer to [`unread_call_captures`]), which the lowering of the call
/// replaces with a value that requires no buffer. This is the one rule by which the captures of a program (refer to
/// [`emitted_capture_reads`]) and its session constants (refer to [`partition_session_constants`]) are read, so that
/// the admission of a function, the signature of its executable, its session constants, and the lowering of its calls
/// agree. A call without captures costs one comparison.
fn visit_emitted_reads<O: LowerableOperationFamily>(
    region: RegionRef<'_, Constant, O>,
    target_platform: Option<&str>,
    mut visit: impl FnMut(AtomId, bool),
) {
    for instruction in region.instructions() {
        let unread = instruction
            .operation()
            .projected_payload::<CallOperation<Type>>()
            .filter(|call| call.capture_count() != 0)
            .and_then(|call| {
                let callee = region.with_id(*instruction.regions().first()?).ok()?;
                unread_call_captures(callee, call.capture_count(), target_platform)
            });
        for (index, id) in instruction.inputs().iter().enumerate() {
            visit(*id, !unread.as_deref().is_some_and(|unread| unread.get(index).copied().unwrap_or(false)));
        }
    }
    for id in region.output_ids() {
        visit(*id, true);
    }
}

/// Returns, for each input of `callee`, the callee of a call whose first `capture_count` operands are its lifted
/// captures, whether it is a capture that the callee never reads, as the family of the call answers it (refer to
/// `LowerableOperation::unread_captures`, which `Operation` answers with `unread_lifted_captures`), or [`None`]
/// when every capture counts as read.
pub(crate) fn unread_call_captures<O: LowerableOperationFamily>(
    callee: RegionRef<'_, Constant, O>,
    capture_count: usize,
    target_platform: Option<&str>,
) -> Option<Arc<[bool]>> {
    O::unread_captures(callee, capture_count, target_platform)
}

/// Returns, for each logical argument of the entry function of `program`, whose types are `argument_types` and whose
/// first `capture_count` arguments are captures that end with the arguments of its `session_count` session constants,
/// whether the executable has no argument for it because nothing reads it. That is a capture that no region which
/// lowering emits for `target_platform` reads (refer to [`emitted_capture_reads`]) and a public input that
/// `unread_inputs` flags, which the caller states because it cannot be read from `program` (refer to
/// `Options::with_unread_inputs`), of a static shape and backing no reference state (i.e., not at one of
/// `reference_state_inputs`). This is the one liveness decision of an executable boundary: the admission and the
/// placement of its arguments when a function is compiled, the signature of its executable, which records it (refer to
/// `ExecutableSignature::is_unread_input`), and every execution and persisted copy of it read it, and none of them
/// requires anything of such an argument.
pub(crate) fn unread_entry_arguments<ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    program: &Program<Constant, O, ProgramInput, ProgramOutput>,
    argument_types: &[ArrayType],
    capture_count: usize,
    session_count: usize,
    unread_inputs: &[bool],
    reference_state_inputs: &[usize],
    target_platform: Option<&str>,
) -> Vec<bool>
where
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
{
    let lifted = program.input_count() + session_count == argument_types.len();
    let source_capture_count = capture_count.saturating_sub(session_count);
    let capture_reads =
        emitted_capture_reads(program.entry_region_ref(), source_capture_count, lifted, target_platform);
    argument_types
        .iter()
        .enumerate()
        .map(|(index, r#type)| {
            let is_read = match index.checked_sub(capture_count) {
                Some(public_index) => !unread_inputs.get(public_index).copied().unwrap_or(false),
                None => capture_reads.get(index).copied().unwrap_or(true),
            };
            !is_read && r#type.static_shape().is_some() && !reference_state_inputs.contains(&index)
        })
        .collect()
}

/// Returns the distinct [`RetainedArray`]s that `program` references in the regions that lowering emits for
/// `target_platform` (i.e., its entry region and the executed nested regions of that region, refer to
/// `emitted_regions`), in the order of their first occurrence in those regions. Lowering passes their arrays as hidden
/// entry arguments that follow the lifted captures, in this order, and so the caller that assembles the executable
/// arguments must use the same order and the same target as that lowering. Regions that lowering never reaches (e.g.,
/// the rule regions of composites, the transpose region of a `linear_call`, or a body that the implementation of a
/// named composite replaces for the target) are skipped, so that an array that only they refer to does not become an
/// argument of the executable. So is an array that the emitted regions only pass on to calls as captures which their
/// callees never read (refer to [`partition_session_constants`]).
pub(crate) fn collect_session_constants<ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    program: &Program<Constant, O, ProgramInput, ProgramOutput>,
    target_platform: Option<&str>,
) -> Vec<RetainedArray>
where
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
{
    partition_session_constants(program.entry_region_ref(), target_platform).0
}

/// Returns the distinct [`RetainedArray`]s that the regions which lowering emits for `target_platform` below `entry`
/// reference, in the order of their first occurrence in those regions, split into the session constants of the program
/// (refer to [`collect_session_constants`]) and the arrays that those regions only pass on to calls as captures which
/// their callees never read (refer to [`visit_emitted_reads`]). The latter require nothing of an execution, so they have
/// no executable argument, and lowering binds each of them to a value that requires no buffer (i.e., the value with
/// which the entry function binds an erased argument), which the calls receive in its place.
pub(crate) fn partition_session_constants<O: LowerableOperationFamily>(
    entry: RegionRef<'_, Constant, O>,
    target_platform: Option<&str>,
) -> (Vec<RetainedArray>, Vec<RetainedArray>) {
    partition_region_session_constants(emitted_regions(entry, target_platform), target_platform)
}

/// Returns the distinct [`RetainedArray`]s that `regions` reference, in the order of their first occurrence in them,
/// split into those that they read and those that they only pass on to calls as captures which the callees never read
/// (refer to [`partition_session_constants`] and `entry_session_constants`).
pub(crate) fn partition_region_session_constants<'r, O: LowerableOperationFamily + 'r>(
    regions: impl Iterator<Item = RegionRef<'r, Constant, O>>,
    target_platform: Option<&str>,
) -> (Vec<RetainedArray>, Vec<RetainedArray>) {
    // The set indexes the constants seen so far by their identity, so that collecting distinct constants costs one hash
    // for each occurrence rather than a comparison with every constant collected before it.
    let mut seen = HashSet::new();
    let mut session_constants = Vec::new();
    let mut live = HashSet::new();
    let mut passed_on = HashSet::new();
    for region in regions {
        let mut has_session_constants = false;
        for atom in region.atoms() {
            if let Some(constant) = atom.as_constant().and_then(|constant| session_constant(constant)) {
                has_session_constants = true;
                if seen.insert(constant.clone()) {
                    session_constants.push(constant.clone());
                }
            }
        }
        if has_session_constants {
            visit_emitted_reads(region, target_platform, |id, is_live| {
                let atom = region.atoms().get(id.index());
                if let Some(constant) = atom.and_then(|atom| atom.as_constant()).and_then(session_constant) {
                    if is_live {
                        live.insert(constant.clone())
                    } else {
                        passed_on.insert(constant.clone())
                    };
                }
            });
        }
    }
    if passed_on.iter().all(|constant| live.contains(constant)) {
        return (session_constants, Vec::new());
    }
    session_constants
        .into_iter()
        .partition(|constant| live.contains(constant) || !passed_on.contains(constant))
}

/// Returns the array that `constant` retains if it is a session constant (i.e., a [`Constant::Session`]), and [`None`]
/// for any other constant.
fn session_constant(constant: &Constant) -> Option<&RetainedArray> {
    match constant {
        Constant::Session(constant) => Some(constant),
        _ => None,
    }
}

#[cfg(test)]
mod tests {

    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, CompositeKey, CompositeOperation, CustomRuleDefinition, CustomRuleRegistration, Device,
        LogicalMesh, MeshAxis, MeshAxisType, NamedComposite, Placeholder, ProgramBuilder, Sharding, Typed,
    };

    use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};

    use crate::{Constant, FromPjrt, Operation, PlacementOptions, RetainedArray, Session, Type};

    use crate::lowering::modules::to_mlir_module_for_program;

    use crate::lowering::tests::{linear_call_program, scalar_linear_region};

    use crate::sessions::tests::{domain_mesh, f32_scalar};

    use crate::tests::{cpu_client, peer_devices, session_with_remote_devices};

    use super::*;

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
            let session = session_with_remote_devices(client, peer_devices(client, &clients));
            let scalar = session.place(4f32, PlacementOptions::default()).unwrap();
            let first = session.place(scalar.clone(), PlacementOptions::to(first_device)).unwrap();
            let second = session.place(scalar.clone(), PlacementOptions::to(first_device)).unwrap();
            let copy = session.place(first.clone(), PlacementOptions::default().with_may_alias(false)).unwrap();
            if client.process_index().unwrap() == 0 {
                let buffer = scalar.shards()[0].buffer().unwrap();
                assert!(Arc::ptr_eq(first.shards()[0].buffer().unwrap(), buffer));
                assert!(Arc::ptr_eq(second.shards()[0].buffer().unwrap(), buffer));
            } else {
                assert_eq!(first.addressable_shards().count(), 0);
                assert_eq!(second.addressable_shards().count(), 0);
            }
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let first_clone = first.clone();
            let outputs = [&scalar, &first, &second, &first_clone, &copy]
                .map(|array| builder.add_constant(Constant::Session(RetainedArray::new(array))))
                .to_vec();
            let program = builder.build::<Vec<Constant>, Vec<Constant>>(outputs, vec![], vec![Placeholder; 5]).unwrap();
            let constants = collect_session_constants(&program, None);
            assert_eq!(constants, [&scalar, &first, &second, &copy].map(RetainedArray::new));

            // One array declared with one type twice is one constant, and declared with two types (e.g., by a function
            // compiled for a mesh of another topology, refer to `imported_constant`) two, on the process that holds its
            // buffers and on the one that holds none alike.
            let other_mesh = LogicalMesh::new(vec![MeshAxis::new("other", 1, MeshAxisType::Auto).unwrap()]).unwrap();
            let other_type = first.r#type().into_owned().with_sharding(Sharding::replicated(other_mesh, 0)).unwrap();
            let declared = RetainedArray::new(&first).with_declared_type(other_type.clone());
            assert_eq!(declared, RetainedArray::new(&first).with_declared_type(other_type));
            assert_ne!(declared, RetainedArray::new(&first));
            let mut declared_builder = ProgramBuilder::<Constant, Operation>::new();
            let declared_outputs = [declared.clone(), RetainedArray::new(&first), declared.clone()]
                .map(|constant| declared_builder.add_constant(Constant::Session(constant)))
                .to_vec();
            let declared_program = declared_builder
                .build::<Vec<Constant>, Vec<Constant>>(declared_outputs, vec![], vec![Placeholder; 3])
                .unwrap();
            assert_eq!(collect_session_constants(&declared_program, None), vec![declared, RetainedArray::new(&first)]);

            let captures = constants.iter().map(|constant| constant.r#type().into_owned()).collect::<Vec<_>>();
            let output_types = vec![scalar.r#type().into_owned(); 5];
            to_mlir_module_for_program(&program, &captures, &(), &output_types, "main", None, None).unwrap()
        });

        assert_eq!(modules[0], modules[1]);
        assert_eq!(
            modules[0],
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) {
                    return %arg0, %arg1, %arg2, %arg1, %arg3 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
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
            collect_session_constants(&linear_call_program(&with_constant(false), &with_constant(true)), None),
            vec![],
        );
        assert_eq!(
            collect_session_constants(&linear_call_program(&with_constant(true), &with_constant(false)), None),
            vec![constant],
        );
    }

    #[test]
    fn test_collect_session_constants_skips_bodies_that_the_target_replaces() {
        // A session constant that only the body of a named composite call refers to becomes a hidden argument where
        // lowering emits the body, and not on a target whose backend implementation replaces the whole call
        // (`ryft.tests.xla_named_square` version 1 on CUDA, refer to `omits_regions`), which never lowers the body,
        // as the collection of the module mesh and the validation of placements skip it.
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let constant = RetainedArray::new(&f32_scalar(&session, &domain_mesh(client, "x", 1), 2.0));
        let scalar_type = Type::Array(constant.r#type().into_owned());
        let body = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(scalar_type.clone());
            let constant = builder.add_constant(Constant::Session(constant.clone()));
            let output =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, constant], None).unwrap()[0];
            Arc::new(builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap())
        };
        let key = CompositeKey::new("ryft.tests.xla_named_square".to_string(), 1);
        let named = NamedComposite::new(key, |types: &[Type]| Ok(types.to_vec()), |inputs| Ok(vec![inputs[0].clone()]));
        let rules = CustomRuleRegistration::new(CustomRuleDefinition::new("adding").with_named_composite(named));
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(scalar_type);
        let operation = Operation::Composite(CompositeOperation::new(rules.reference()));
        let regions = vec![builder.intern_callee(&body, None).unwrap()];
        let output = builder.add_instruction(operation, regions, vec![input], None).unwrap()[0];
        let program = builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder]);
        let program = program.unwrap();
        assert_eq!(collect_session_constants(&program, Some("cuda")), vec![]);
        assert_eq!(collect_session_constants(&program, Some("cpu")), vec![constant]);
    }
}
