use ryft_core::macros::check_count;
use ryft_core::{ArrayType, PLACE_OPERATION_NAME, PlaceOperation};
use ryft_mlir::dialects::shardy;
use ryft_mlir::{Block, Operation as MlirOperation, Value as MlirValue, ValueRef};

use crate::operations::manipulation::conversions::lower_layout_conversion;
use crate::operations::manipulation::memory::lower_transfer_to_memory;
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer, LoweringError, ToMlir};

impl LowerableOperation for PlaceOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        // A compiled program runs on the devices of one mesh, so inside it a placement can only redistribute a value
        // over those devices and change its storage. Compilation rejects targets on any other devices before lowering
        // (refer to `LoweringError::PlacementOutsideProgramMesh`), because the lowering only knows the logical mesh,
        // and it also rejects targets that are not replicated in bounded dynamic programs, whose replicas ignore
        // sharding constraints.
        // What remains lowers to a sharding constraint for the target sharding, followed by a layout constraint when
        // the layout changes and by a memory annotation when the memory changes. Unlike the sharding of a type (e.g.,
        // that of `reshard`), a target states the placement along every mesh axis, including auto axes, so the
        // constraint is closed over the dimensions that the target specifies, like that of `constrain_sharding`.
        // The layout is constrained first, so that the conversion runs before the value leaves device memory. JAX
        // instead lowers `device_put` inside `jit` to nothing, unless its target names a memory kind.
        // The copy policy of the operation (i.e., `PlaceOperation::may_alias`) is ignored, because a compiled program
        // has no observable buffer identity. Its outputs are buffers that its execution allocates (or donated inputs,
        // which no other handle can observe), so a value inside it never shares storage with any array that the caller
        // holds, and a forced copy would only add work.
        check_count!("input", input_values, 1, ProgramError);
        check_count!("input", input_types, 1, ProgramError);
        check_count!("output", output_types, 1, ProgramError);

        // A placement without a target leaves its input where it is, so it lowers to nothing, also inside a manual
        // region. Its source only matters to transposition, and its copy policy is ignored like that of any other
        // placement, because a compiled program has no observable buffer identity.
        let Some(target) = self.target() else {
            return Ok(vec![input_values[0]]);
        };

        // The values of a manual region live on the devices of that region. Type inference rejects inputs that are
        // tracked over manual mesh axes, but a value that is created inside the region (e.g., a constant) has no
        // tracked sharding, and constraining it over the whole mesh there would contradict the region.
        if !lowerer.collective_state().bound_manual_axes().is_empty() {
            return Err(LoweringError::UnsupportedOperation {
                operation: PLACE_OPERATION_NAME.to_string(),
                reason: "a value inside a manual region cannot be moved to another placement".to_string(),
            }
            .into());
        }
        let (input_type, output_type) = (&input_types[0], &output_types[0]);
        let location = lowerer.location();
        let sharding = target.sharding().to_mlir(location)?;
        let mut value = lowerer
            .block()
            .append_operation(shardy::sharding_constraint(input_values[0], sharding, location)?)?
            .result(0)
            .unwrap()
            .as_ref();
        if output_type.layout().is_some() && output_type.layout() != input_type.layout() {
            value = lower_layout_conversion(value, output_type, lowerer)?;
        }
        if output_type.memory() != input_type.memory() {
            value = lower_transfer_to_memory(
                output_type.memory(),
                &[value],
                &mut lowerer.block(),
                lowerer.context(),
                lowerer.location(),
            )?[0];
        }
        Ok(vec![value])
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        BatchAxis, DataType, Device, DeviceMesh, DimensionBounds, DimensionVariable, Layout, LogicalMesh, Memory,
        MeshAxis, MeshAxisType, Place, PlaceDispatch, Placement, Reduce, ReductionKind, Shape, Sharding,
        ShardingDimension, TiledLayout, Typed, Value as CoreValue, batch, differentiate_at,
    };
    use ryft_pjrt::Client;

    use crate::sessions::tests::{f32_vector, two_device_mesh};
    use crate::tests::{cpu_client, values_from_bytes, values_to_bytes};
    use crate::{
        Array, ArrayTracer, CompileTracer, CompiledFunction, FromPjrt, Options, Session, TracedProgram, compile,
        compile_with_options, trace,
    };

    use super::*;

    /// Returns two meshes with the explicit axis `x` of size 1, over the first and the second addressable device of
    /// `client`, respectively.
    fn single_device_meshes(client: &Client<'_>) -> (DeviceMesh, DeviceMesh) {
        let devices = client.addressable_devices().unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = |index: usize| {
            DeviceMesh::new(logical_mesh.clone(), vec![Device::from_pjrt(&devices[index]).unwrap()]).unwrap()
        };
        (mesh(0), mesh(1))
    }

    /// Returns the `f32` values that the devices of `array` hold, in the device order of its mesh.
    fn shard_values(array: &Array) -> Vec<Vec<f32>> {
        array
            .shards()
            .iter()
            .map(|shard| values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()))
            .collect()
    }

    #[test]
    fn test_place_operation_lower() {
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![Device::new(0, 0), Device::new(1, 0)],
        )
        .unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4, 2])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2))
            .unwrap();
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();

        // A placement that keeps the storage of its input lowers to a sharding constraint alone.
        let target = Placement::new(mesh.clone(), sharding.clone()).unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> =
            trace(move |input: ArrayTracer| input.place(&target, None).unwrap(), input_type.clone()).unwrap();
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], replicated={"x"}>}) -> (tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{"x"}, {}]> : tensor<4x2xf32>
                    return %0 : tensor<4x2xf32>
                  }
                }
            "#},
        );

        // A placement that changes the layout and the memory of its input also constrains the layout and then
        // annotates the memory.
        let target = Placement::new(mesh, sharding)
            .unwrap()
            .with_memory(Memory::Host { pinned: true })
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])));
        let traced: TracedProgram<ArrayType, ArrayType> =
            trace(move |input: ArrayTracer| input.place(&target, None).unwrap(), input_type).unwrap();
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], replicated={"x"}>}) -> (tensor<4x2xf32> {mhlo.layout_mode = "{0,1}", mhlo.memory_kind = "pinned_host", sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{"x"}, {}]> : tensor<4x2xf32>
                    %1 = stablehlo.custom_call @LayoutConstraint(%0) {operand_layouts = [dense<[0, 1]> : tensor<2xindex>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<4x2xf32>) -> tensor<4x2xf32>
                    %2 = stablehlo.custom_call @annotate_device_placement(%1) {backend_config = "", has_side_effect = true, mhlo.frontend_attributes = {_xla_buffer_placement = "pinned_host"}} : (tensor<4x2xf32>) -> tensor<4x2xf32>
                    return %2 : tensor<4x2xf32>
                  }
                }
            "#},
        );

        // A target states the placement along auto mesh axes as well, which types do not track, so the constraint is
        // closed over the dimensions of the target and the compiler cannot shard them along further auto axes.
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(0, 0), Device::new(1, 0)],
        )
        .unwrap();
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let target = Placement::new(mesh, sharding).unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            move |input: ArrayTracer| input.place(&target, None).unwrap(),
            ArrayType::new_static(DataType::F32, [4, 2]),
        )
        .unwrap();
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4x2xf32>) -> (tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{"x"}, {}]> : tensor<4x2xf32>
                    return %0 : tensor<4x2xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_place_operation_lower_ignores_may_alias() {
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![Device::new(0, 0), Device::new(1, 0)],
        )
        .unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let target = Placement::new(mesh, sharding).unwrap();
        let place = |may_alias: bool| -> TracedProgram<ArrayType, ArrayType> {
            let target = target.clone();
            trace(
                move |input: ArrayTracer| {
                    <ArrayTracer as CoreValue>::Dispatch::place(&input, Some(&target), None, may_alias).unwrap()
                },
                input_type.clone(),
            )
            .unwrap()
        };

        // A forced copy is staged, but a compiled program has no observable buffer identity, so it lowers to the same
        // module as a placement that may alias its input.
        let copying = place(false);
        assert_eq!(
            copying.program().to_string(),
            indoc! {r#"
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = place [
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                    may_alias=false,
                ] %0
                in (%1)
            "#}
            .trim_end(),
        );
        assert_eq!(copying.to_mlir_module("main").unwrap(), place(true).to_mlir_module("main").unwrap());
        assert_eq!(
            copying.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], replicated={"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{"x"}]> : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_place_operation_lower_without_target() {
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![Device::new(0, 0), Device::new(1, 0)],
        )
        .unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let source = Placement::new(mesh, input_type.sharding().unwrap().clone()).unwrap();
        let place = |source: Option<Placement>, may_alias: bool| -> TracedProgram<ArrayType, ArrayType> {
            trace(
                move |input: ArrayTracer| {
                    <ArrayTracer as CoreValue>::Dispatch::place(&input, None, source.clone(), may_alias).unwrap()
                },
                input_type.clone(),
            )
            .unwrap()
        };

        // A placement without a target leaves its input where it is, so it lowers to nothing, whatever its source and
        // copy policy are.
        let traced = place(Some(source.clone()), false);
        assert_eq!(
            traced.program().to_string(),
            indoc! {r#"
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                    may_alias=false,
                ] %0
                in (%1)
            "#}
            .trim_end(),
        );
        assert_eq!(traced.to_mlir_module("main").unwrap(), place(None, true).to_mlir_module("main").unwrap());
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    return %arg0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_place_operation_lower_rejects_other_device_meshes() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let input_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let place_error = |target: Placement| {
            let compiled: Result<CompiledFunction<ArrayType, ArrayType>, _> =
                compile(move |input| input.place(&target, None).unwrap(), input_type.clone(), &session, mesh.clone());
            compiled.err().unwrap()
        };

        // The same devices in another order are another device mesh.
        let reversed =
            DeviceMesh::new(mesh.logical_mesh().clone(), mesh.devices().iter().rev().copied().collect()).unwrap();
        let error = place_error(
            Placement::new(reversed.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap(),
        );
        assert!(matches!(
            &error,
            Error::Lowering(LoweringError::PlacementOutsideProgramMesh { target, mesh: program_mesh })
                if target == &reversed && program_mesh == &mesh,
        ));
        let device_ids = |mesh: &DeviceMesh| mesh.devices().iter().map(|device| device.id()).collect::<Vec<_>>();
        assert_eq!(
            error.to_string(),
            format!(
                "`place` cannot move a value to the devices {:?} of mesh `{}` inside a program compiled for the \
                 devices {:?} of mesh `{}`",
                device_ids(&reversed),
                reversed.logical_mesh(),
                device_ids(&mesh),
                mesh.logical_mesh(),
            ),
        );

        // A target on another topology is reported the same way, rather than as a mismatch of the module mesh.
        let (single_device_mesh, _) = single_device_meshes(client);
        let error = place_error(
            Placement::new(
                single_device_mesh.clone(),
                Sharding::replicated(single_device_mesh.logical_mesh().clone(), 1),
            )
            .unwrap(),
        );
        assert!(matches!(
            &error,
            Error::Lowering(LoweringError::PlacementOutsideProgramMesh { target, mesh: program_mesh })
                if target == &single_device_mesh && program_mesh == &mesh,
        ));
    }

    #[test]
    fn test_place_operation_lower_rejects_splits_in_bounded_dynamic_programs() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let extent = DimensionVariable::new("n", DimensionBounds::new(1, Some(8)).unwrap());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.into()]));
        let place = |sharding: Sharding| -> Result<CompiledFunction<ArrayType, ArrayType>, Error> {
            let target = Placement::new(mesh.clone(), sharding).unwrap();
            compile(move |input| input.place(&target, None).unwrap(), input_type.clone(), &session, mesh.clone())
        };

        // Bounded dynamic programs run one fully replicated replica per device, because the Shardy partitioner rejects
        // dynamic tensors, so a target that splits a dimension would be ignored and is rejected instead.
        let error = place(Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .err()
            .unwrap();
        assert!(matches!(
            &error,
            Error::Lowering(LoweringError::UnsupportedOperation { operation, .. }) if operation == PLACE_OPERATION_NAME,
        ));
        assert_eq!(
            error.to_string(),
            format!(
                "cannot lower `{PLACE_OPERATION_NAME}`: bounded-dynamic multi-device programs currently require fully \
                 replicated placement targets because Shardy rejects dynamic tensors",
            ),
        );

        // A replicated target is what every replica computes anyway.
        assert!(place(Sharding::replicated(mesh.logical_mesh().clone(), 1)).is_ok());
    }

    #[test]
    fn test_place_operation_lower_executes_on_the_program_mesh() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let input_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let target = Placement::new(mesh.clone(), sharding.clone()).unwrap();
        let compiled: CompiledFunction<ArrayType, ArrayType> =
            compile(move |input| input.place(&target, None).unwrap(), input_type.clone(), &session, mesh.clone())
                .unwrap();
        let input = Array::from_host_buffer(
            &session,
            input_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        let output = compiled.executable().execute(&session, input).unwrap();
        assert_eq!(output.r#type().as_ref(), &input_type.with_sharding(sharding.clone()).unwrap());
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(output.sharding(), &sharding);
        assert_eq!(shard_values(&output), vec![vec![1., 2.], vec![3., 4.]]);
    }

    #[test]
    fn test_place_operation_lower_executes_on_auto_meshes() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let input_type = ArrayType::new_static(DataType::F32, [4]);
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        // Types record no placement along auto mesh axes, so the placement of the input is an override.
        let place = |input_sharding: &Sharding, target_sharding: &Sharding| {
            let target = Placement::new(mesh.clone(), target_sharding.clone()).unwrap();
            let compiled: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
                move |input| input.place(&target, None).unwrap(),
                input_type.clone(),
                &session,
                Options::new(mesh.clone()).with_input_shardings(vec![input_sharding.clone()]),
            )
            .unwrap();
            let input = Array::from_host_buffer(
                &session,
                input_type.clone(),
                mesh.clone(),
                Some(input_sharding.clone()),
                values_to_bytes(&[1f32, 2., 3., 4.]),
            )
            .unwrap();
            compiled.executable().execute(&session, input).unwrap()
        };

        // A replicated target replicates an input that is split along `x`.
        let output = place(&sharded, &replicated);
        assert_eq!(output.sharding(), &replicated);
        assert_eq!(shard_values(&output), vec![vec![1., 2., 3., 4.], vec![1., 2., 3., 4.]]);

        // A target that is split along `x` splits a replicated input.
        let output = place(&replicated, &sharded);
        assert_eq!(output.sharding(), &sharded);
        assert_eq!(shard_values(&output), vec![vec![1., 2.], vec![3., 4.]]);
    }

    #[test]
    fn test_place_operation_lower_executes_auto_splits_with_reduction_state() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("r", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let devices = client.addressable_devices().unwrap();
        let devices = devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices).unwrap();
        let replicated = Sharding::replicated(logical_mesh.clone(), 1);
        let sharded = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let place = |input_sharding: Sharding, target_sharding: Sharding| {
            let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(input_sharding.clone()).unwrap();
            let target = Placement::new(mesh.clone(), target_sharding).unwrap();
            let compiled: CompiledFunction<ArrayType, ArrayType> =
                compile(move |input| input.place(&target, None).unwrap(), input_type.clone(), &session, mesh.clone())
                    .unwrap();
            let input = Array::from_host_buffer(
                &session,
                input_type,
                mesh.clone(),
                Some(input_sharding),
                values_to_bytes(&[1f32, 2., 3., 4.]),
            )
            .unwrap();
            compiled.executable().execute(&session, input).unwrap()
        };

        // The type of the result drops the split along the auto axis `x`, and the reduction state along the explicit
        // axis `r` keeps the compiler from choosing the placement of the result, so the result is placed as the target
        // requests.
        let reduced = sharded.clone().with_reduced_axes(["r"]).unwrap();
        let output = place(replicated.clone().with_reduced_axes(["r"]).unwrap(), reduced.clone());
        assert_eq!(output.sharding(), &reduced);
        assert_eq!(shard_values(&output), vec![vec![1., 2.], vec![1., 2.], vec![3., 4.], vec![3., 4.]]);
        let unreduced = sharded.with_unreduced_axes(["r"]).unwrap();
        let output = place(replicated.with_unreduced_axes(["r"]).unwrap(), unreduced.clone());
        assert_eq!(output.sharding(), &unreduced);
        assert_eq!(shard_values(&output).iter().map(Vec::len).collect::<Vec<_>>(), vec![2, 2, 2, 2]);
        assert_eq!(output.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1., 2., 3., 4.]);
    }

    #[test]
    fn test_place_operation_lower_executes_auto_splits_with_unconstrained_dimensions() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let input_type = ArrayType::new_static(DataType::F32, [2, 2]);
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 2);
        let sharded = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::unconstrained()],
        )
        .unwrap();

        // The result keeps the split of its first dimension even though an unconstrained dimension leaves the rest of
        // its placement to the compiler.
        let target = Placement::new(mesh.clone(), sharded.clone()).unwrap();
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
            move |input| input.place(&target, None).unwrap(),
            input_type.clone(),
            &session,
            Options::new(mesh.clone()).with_input_shardings(vec![replicated.clone()]),
        )
        .unwrap();
        let input = Array::from_host_buffer(
            &session,
            input_type,
            mesh.clone(),
            Some(replicated),
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        let output = compiled.executable().execute(&session, input).unwrap();
        assert_eq!(output.sharding(), &sharded);
        assert_eq!(shard_values(&output), vec![vec![1., 2.], vec![3., 4.]]);
    }

    #[test]
    fn test_place_operation_lower_executes_auto_splits_of_zero_arrays() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let input_type = ArrayType::new_static(DataType::Zero, [4]);
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        // A result of the zero data type has no buffer whose placement the compiler could report, so its placement
        // comes from the target alone.
        let target = Placement::new(mesh.clone(), sharded.clone()).unwrap();
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
            move |input| input.place(&target, None).unwrap(),
            input_type.clone(),
            &session,
            Options::new(mesh.clone()).with_input_shardings(vec![replicated.clone()]),
        )
        .unwrap();
        let input =
            Array::from_host_buffer(&session, input_type.clone(), mesh.clone(), Some(replicated.clone()), []).unwrap();
        let output = compiled.executable().execute(&session, input.clone()).unwrap();
        assert_eq!(output.sharding(), &sharded);

        // A placement without a target leaves the placed value where it is, so the result keeps the target placement.
        let target = Placement::new(mesh.clone(), sharded.clone()).unwrap();
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
            move |input: CompileTracer| {
                let placed = input.place(&target, None).unwrap();
                <CompileTracer as CoreValue>::Dispatch::place(&placed, None, None, true).unwrap()
            },
            input_type,
            &session,
            Options::new(mesh).with_input_shardings(vec![replicated]),
        )
        .unwrap();
        let output = compiled.executable().execute(&session, input).unwrap();
        assert_eq!(output.sharding(), &sharded);
    }

    #[test]
    fn test_place_operation_eager_jvp() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let (source_mesh, target_mesh) = single_device_meshes(client);
        let target =
            Placement::new(target_mesh.clone(), Sharding::replicated(target_mesh.logical_mesh().clone(), 1)).unwrap();
        let primal = f32_vector(&session, &source_mesh, &[1., 2., 3.]);
        let tangent = f32_vector(&session, &source_mesh, &[1., 0., -1.]);

        // The primal and the tangent both move to the device of the target mesh.
        let (value, tangent): (Array, Array) =
            differentiate_at(primal).jvp(tangent, |x| x.place(&target, None)).unwrap();
        assert_eq!(value.mesh(), &target_mesh);
        assert_eq!(tangent.mesh(), &target_mesh);
        assert_eq!(shard_values(&value), vec![vec![1., 2., 3.]]);
        assert_eq!(shard_values(&tangent), vec![vec![1., 0., -1.]]);
    }

    #[test]
    fn test_place_operation_eager_vjp() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let (source_mesh, target_mesh) = single_device_meshes(client);
        let target =
            Placement::new(target_mesh.clone(), Sharding::replicated(target_mesh.logical_mesh().clone(), 1)).unwrap();
        let primal = f32_vector(&session, &source_mesh, &[1., 2., 3.]);

        // The pullback moves the cotangent from the target device back to the source device and mesh.
        let (value, pullback) = differentiate_at(primal.clone()).vjp(|x| x.place(&target, None)).unwrap();
        assert_eq!(value.mesh(), &target_mesh);
        assert_eq!(shard_values(&value), vec![vec![1., 2., 3.]]);
        let cotangent = f32_vector(&session, &target_mesh, &[4., 5., 6.]);
        let input_cotangent = pullback.apply_materialized(cotangent).unwrap();
        assert_eq!(input_cotangent.mesh(), &source_mesh);
        assert_eq!(input_cotangent.r#type(), primal.r#type());
        assert_eq!(shard_values(&input_cotangent), vec![vec![4., 5., 6.]]);

        // The cotangent of a sum that is computed on the target device is broadcast there and then comes back on the
        // source device.
        let (value, pullback) = differentiate_at(primal.clone())
            .vjp(|x| x.place(&target, None)?.reduce(&[0], ReductionKind::Sum))
            .unwrap();
        assert_eq!(value.mesh(), &target_mesh);
        assert_eq!(shard_values(&value), vec![vec![6.]]);
        let cotangent = Array::from_host_buffer(
            &session,
            ArrayType::scalar(DataType::F32),
            target_mesh,
            None,
            values_to_bytes(&[2f32]),
        )
        .unwrap();
        let input_cotangent = pullback.apply_materialized(cotangent).unwrap();
        assert_eq!(input_cotangent.mesh(), &source_mesh);
        assert_eq!(input_cotangent.r#type(), primal.r#type());
        assert_eq!(shard_values(&input_cotangent), vec![vec![2., 2., 2.]]);
    }

    #[test]
    fn test_place_operation_eager_gradient() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let (source_mesh, target_mesh) = single_device_meshes(client);
        let target =
            Placement::new(target_mesh.clone(), Sharding::replicated(target_mesh.logical_mesh().clone(), 1)).unwrap();
        let primal = f32_vector(&session, &source_mesh, &[1., 2., 3.]);

        // The seed of a gradient is created on the mesh of the session and not on the target device, where the output
        // lives. The transposed placement moves the cotangent from wherever it actually lives, so the gradient still
        // comes back on the source device and mesh.
        let gradient: Array = differentiate_at(primal.clone())
            .gradient(|x| x.place(&target, None)?.reduce(&[0], ReductionKind::Sum))
            .unwrap();
        assert_eq!(gradient.mesh(), &source_mesh);
        assert_eq!(gradient.r#type(), primal.r#type());
        assert_eq!(shard_values(&gradient), vec![vec![1., 1., 1.]]);
    }

    #[test]
    fn test_place_operation_eager_batch() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let (source_mesh, target_mesh) = single_device_meshes(client);
        let target =
            Placement::new(target_mesh.clone(), Sharding::replicated(target_mesh.logical_mesh().clone(), 1)).unwrap();
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2, 3]),
            source_mesh,
            None,
            values_to_bytes(&[1f32, 2., 3., 4., 5., 6.]),
        )
        .unwrap();

        // Each batch item is placed on the target device, so the packed result lives there.
        let output: Array =
            batch(|x| x.place(&target, None), input, BatchAxis::new(0), BatchAxis::new(0), None).unwrap();
        assert_eq!(output.mesh(), &target_mesh);
        assert_eq!(output.r#type().shape(), ArrayType::new_static(DataType::F32, [2, 3]).shape());
        assert_eq!(shard_values(&output), vec![vec![1., 2., 3., 4., 5., 6.]]);
    }

    #[test]
    fn test_place_operation_eager_batch_gradient() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let (source_mesh, target_mesh) = single_device_meshes(client);
        let target =
            Placement::new(target_mesh.clone(), Sharding::replicated(target_mesh.logical_mesh().clone(), 1)).unwrap();
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2, 3]),
            source_mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2., 3., 4., 5., 6.]),
        )
        .unwrap();

        // A batched value reports the placement of its batch items, so the gradient of each item comes back on the
        // source device without an explicit source.
        let gradient: Array = batch(
            |x| Ok(differentiate_at(x).gradient(|x| x.place(&target, None)?.reduce(&[0], ReductionKind::Sum))?),
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(gradient.mesh(), &source_mesh);
        assert_eq!(shard_values(&gradient), vec![vec![1., 1., 1., 1., 1., 1.]]);
    }
}
