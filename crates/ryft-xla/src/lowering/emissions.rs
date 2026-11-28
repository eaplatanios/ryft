//! Partitioning of session constants into hidden entry inputs and arrays forwarded only as unread captures.

use std::collections::HashSet;

use ryft_core::{Atom, RegionRef};

use crate::lowering::ProgramLoweringOperation;
use crate::lowering::programs::RegionLoweringExtension;
use crate::{Constant, RetainedArray};

// TODO(eaplatanios): Review this module.

/// Partitions the distinct [`RetainedArray`]s in `regions`, preserving their order of first occurrence within each
/// group. The second group contains arrays passed only as captures to callees that never read them on
/// `target_platform` (refer to [`RegionLoweringExtension::visit_atom_reads`]). The first group contains all other
/// session constants.
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
            region.visit_atom_reads(target_platform, |id, is_live| {
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

    use crate::lowering::programs::{CallTraversal, ProgramLoweringExtension, ProgramLoweringOptions};

    use crate::lowering::tests::{linear_call_program, scalar_linear_region};

    use crate::sessions::tests::{domain_mesh, f32_scalar};

    use crate::tests::{cpu_client, peer_devices, session_with_remote_devices};

    use super::*;

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
