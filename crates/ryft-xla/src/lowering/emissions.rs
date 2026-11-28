//! Decisions about what lowering emits of a program (i.e., which regions it emits for a target platform, and which
//! captures and session constants those regions read).

use std::collections::{HashMap, HashSet};

use ryft_core::{
    ArrayType, Atom, AtomId, CallOperation, Instruction, Parameterized, Program, RegionRef, RegionRole, Typed,
    TypedProgram, Value,
};

use crate::lowering::ProgramLoweringOperation;
use crate::{Constant, RetainedArray, Type};

// TODO(eaplatanios): Review this module.

/// Whether a traversal follows a [`CallOperation`] into its callee.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum CallTraversal {
    /// Visit the computations that execute across call boundaries.
    Enter,

    /// Stay in the current capture scope. A callee's capture indices refer to its own inputs.
    Skip,
}

/// Queries the instruction boundaries and computation regions that XLA lowers from a [`RegionRef`].
pub(crate) trait RegionLoweringExtension<O: ProgramLoweringOperation>: Sized {
    /// Returns the input and output types of an instruction in this region, in operand and result order.
    fn instruction_boundary_types(self, instruction: &Instruction<O>) -> (Vec<Type>, Vec<Type>);

    /// Visits this region and its emitted computation descendants once each, in the order of
    /// [`RegionRef::computation_regions`]. Dormant rules and bodies omitted by
    /// [`ProgramLoweringOperation::omits_regions`] are skipped. Use [`CallTraversal::Enter`] for whole-program
    /// analyses and [`CallTraversal::Skip`] when resolving captures in this region's scope.
    fn emitted_regions(self, target_platform: Option<&str>, calls: CallTraversal) -> impl Iterator<Item = Self>;
}

impl<V: Value<Type = Type>, O: ProgramLoweringOperation> RegionLoweringExtension<O> for RegionRef<'_, V, O> {
    fn instruction_boundary_types(self, instruction: &Instruction<O>) -> (Vec<Type>, Vec<Type>) {
        let types =
            |atoms: &[AtomId]| atoms.iter().map(|atom| self.atoms()[atom.index()].r#type().into_owned()).collect();
        (types(instruction.inputs()), types(instruction.outputs()))
    }

    fn emitted_regions(self, target_platform: Option<&str>, calls: CallTraversal) -> impl Iterator<Item = Self> {
        let mut visited = vec![false; self.arena().len()];
        let mut pending = vec![self.id()];
        std::iter::from_fn(move || {
            while let Some(id) = pending.pop() {
                if std::mem::replace(&mut visited[id.index()], true) {
                    continue;
                }
                // Every pending identifier is this region or an attachment in the same arena.
                let region = self.with_id(id).unwrap();
                for instruction in region.instructions() {
                    let operation = instruction.operation();
                    if operation.omits_regions(target_platform, || region.instruction_boundary_types(instruction))
                        || (calls == CallTraversal::Skip
                            && operation.projected_payload::<CallOperation<Type>>().is_some())
                    {
                        continue;
                    }
                    pending.extend(instruction.regions().iter().copied().enumerate().filter_map(|(index, id)| {
                        (operation.region_role(index) == Some(RegionRole::Computation)).then_some(id)
                    }));
                }
                return Some(region);
            }
            None
        })
    }
}

/// Returns, for each of the first `capture_count` captures of the program whose entry region is `entry`, whether a
/// region that lowering emits for `target_platform` reads it (refer to [`RegionLoweringExtension::emitted_regions`]),
/// as an input of one of its instructions or as one of its outputs. Every region names a capture through a constant
/// whose [`Value::capture_index`] is its position, except the entry region of a program with lifted captures (i.e.,
/// when `lifted` is set), which names capture `k` through its input `k`. A capture that no emitted region reads (e.g.,
/// one that only the body of a scan with zero iterations reads) is neither admitted when a function is compiled nor
/// passed to its executable, whose signature erases it (refer to `ProgramLoweringExtension::to_mlir`), so that
/// admission and lowering read this one decision. A capture that a region only passes on to a call as a capture which
/// the callee never reads is not read either (refer to [`visit_emitted_reads`]). The regions of the callees of calls
/// are not visited, because their constants name the captures of those callees, which a callee reads only through the
/// inputs of its call (refer to [`CallTraversal::Skip`]).
pub(crate) fn emitted_capture_reads<'r, O: ProgramLoweringOperation>(
    entry: RegionRef<'r, Constant, O>,
    capture_count: usize,
    lifted: bool,
    target_platform: Option<&'r str>,
) -> Vec<bool> {
    let mut is_read = vec![false; capture_count];
    let entry_id = entry.id();
    for region in entry.emitted_regions(target_platform, CallTraversal::Skip) {
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

/// Calls `visit` with every atom that `region` reads, as an input of one of its instructions or as one of its
/// outputs, and with whether that read is live. Every read is live except an input that a `call` passes as a
/// lifted capture that its callee never reads (see [`unread_captures`](super::LowerableOperation::unread_captures)).
/// Call lowering replaces that input with a value that requires no buffer. Both [`emitted_capture_reads`] and
/// [`partition_region_session_constants`] use this rule so that admission, the executable signature, and the lowering
/// of calls agree on which arrays need buffers. A call without captures costs one comparison.
fn visit_emitted_reads<O: ProgramLoweringOperation>(
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
                O::unread_captures(callee, call.capture_count(), target_platform)
            });
        for (index, id) in instruction.inputs().iter().enumerate() {
            visit(*id, !unread.as_deref().is_some_and(|unread| unread.get(index).copied().unwrap_or(false)));
        }
    }
    for id in region.output_ids() {
        visit(*id, true);
    }
}

/// Returns, for each logical input of the entry function of `program`, whose types are `logical_input_types` and whose
/// first `capture_count` inputs are captures that end with the inputs of its `session_count` session constants, whether
/// its executable slot is omitted because nothing reads it. This covers captures unread by any region emitted for
/// `target_platform` (refer to [`emitted_capture_reads`]) and public inputs that
/// `unread_inputs` flags, which the caller states because it cannot be read from `program` (refer to
/// `Options::with_unread_inputs`), of a static shape and backing no reference state (i.e., not at one of
/// `reference_state_inputs`). Input admission and placement use this liveness decision during compilation. The
/// executable boundary mapping records it for execution and persistence, so an unread input requires no buffer
/// (refer to [`MlirProgramBoundaryMapping::is_unread_input`](super::MlirProgramBoundaryMapping::is_unread_input)).
pub(crate) fn unread_entry_inputs<ProgramInput, ProgramOutput, O: ProgramLoweringOperation>(
    program: &Program<Constant, O, ProgramInput, ProgramOutput>,
    logical_input_types: &[ArrayType],
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
    let lifted = program.input_count() + session_count == logical_input_types.len();
    let source_capture_count = capture_count.saturating_sub(session_count);
    let capture_reads =
        emitted_capture_reads(program.entry_region_ref(), source_capture_count, lifted, target_platform);
    logical_input_types
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

/// Partitions the distinct [`RetainedArray`]s in `regions`, preserving their order of first occurrence within each
/// group. The second group contains arrays passed only as captures to callees that never read them on
/// `target_platform` (refer to [`visit_emitted_reads`]). The first group contains all other session constants.
///
/// For a whole program, callers pass its [`RegionLoweringExtension::emitted_regions`] for the same target. Arrays in
/// the first group become hidden entry inputs after the lifted captures, in the order returned. Lowering binds arrays
/// in the second group to bufferless values and passes those values to the calls.
pub(crate) fn partition_region_session_constants<'r, O: ProgramLoweringOperation + 'r>(
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
            if let Atom::Constant(Constant::Session(constant)) = atom {
                has_session_constants = true;
                if seen.insert(constant.clone()) {
                    session_constants.push(constant.clone());
                }
            }
        }
        if has_session_constants {
            visit_emitted_reads(region, target_platform, |id, is_live| {
                if let Some(Atom::Constant(Constant::Session(constant))) = region.atoms().get(id.index()) {
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

    use crate::lowering::programs::{ProgramLoweringExtension, ProgramLoweringOptions};

    use crate::lowering::tests::{
        add_call, linear_call_program, neg_branch, scalar_linear_region, test_vector_type, unproject_plain_program,
    };

    use crate::sessions::tests::{domain_mesh, f32_scalar};

    use crate::tests::{cpu_client, peer_devices, session_with_remote_devices};

    use super::*;

    #[test]
    fn test_region_ref_emitted_regions() {
        let vector = test_vector_type(4);
        let callee = Arc::new(unproject_plain_program(neg_branch(vector.clone())));
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(vector.into());
        let first = add_call(&mut builder, &callee, vec![input]);
        let second = add_call(&mut builder, &callee, vec![first]);
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![second], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let entry = program.entry_region_ref();
        let shared = entry.instructions()[0].regions()[0];
        assert_eq!(entry.instructions()[1].regions(), &[shared]);
        assert_eq!(
            entry.emitted_regions(None, CallTraversal::Enter).map(RegionRef::id).collect::<Vec<_>>(),
            vec![entry.id(), shared],
        );
        assert_eq!(
            entry.emitted_regions(None, CallTraversal::Skip).map(RegionRef::id).collect::<Vec<_>>(),
            vec![entry.id()],
        );
    }

    #[test]
    fn test_partition_region_session_constants_assigns_the_same_inputs_on_every_process() {
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
            let regions = program.entry_region_ref().emitted_regions(None, CallTraversal::Enter);
            let (constants, passed_on) = partition_region_session_constants(regions, None);
            assert_eq!(constants, [&scalar, &first, &second, &copy].map(RetainedArray::new));
            assert!(passed_on.is_empty());

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
            let regions = declared_program.entry_region_ref().emitted_regions(None, CallTraversal::Enter);
            assert_eq!(
                partition_region_session_constants(regions, None),
                (vec![declared, RetainedArray::new(&first)], vec![]),
            );

            let captures = constants.iter().map(|constant| constant.r#type().into_owned()).collect::<Vec<_>>();
            let output_types = vec![scalar.r#type().into_owned(); 5];
            program
                .to_rendered_mlir(
                    "main",
                    &ProgramLoweringOptions::new()
                        .without_pruning()
                        .with_capture_types(captures.to_vec())
                        .with_input_types(&())
                        .with_output_types(&output_types)
                        .without_boundary_shardings(),
                )
                .unwrap()
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
    fn test_partition_region_session_constants_ignores_dormant_regions() {
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
        // it does not become a hidden input of the entry function, unlike one that its forward region refers to.
        let plain_region = with_constant(false);
        let constant_region = with_constant(true);
        let dormant = linear_call_program(&plain_region, &constant_region);
        let regions = dormant.entry_region_ref().emitted_regions(None, CallTraversal::Enter);
        assert_eq!(partition_region_session_constants(regions, None), (vec![], vec![]));

        let live = linear_call_program(&constant_region, &plain_region);
        let regions = live.entry_region_ref().emitted_regions(None, CallTraversal::Enter);
        assert_eq!(partition_region_session_constants(regions, None), (vec![constant], vec![]));
    }

    #[test]
    fn test_partition_region_session_constants_skips_bodies_that_the_target_replaces() {
        // A session constant that only the body of a named composite call refers to becomes a hidden input where
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
        for (platform, expected) in [("cuda", vec![]), ("cpu", vec![constant])] {
            let regions = program.entry_region_ref().emitted_regions(Some(platform), CallTraversal::Enter);
            assert_eq!(partition_region_session_constants(regions, Some(platform)), (expected, vec![]));
        }
    }
}
