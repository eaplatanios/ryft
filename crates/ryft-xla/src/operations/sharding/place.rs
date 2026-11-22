use ryft_core::macros::check_count;
use ryft_core::{ArrayType, PLACE_OPERATION_NAME, PlaceOperation, PlacementRequest};
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

        // Transposition turns the source of a placement into the target of the transposed placement, which moves the
        // cotangent there, so a source that names devices (i.e., a placement or a device) inside a manual region would
        // move a value of that region in reverse mode, which the target rule below rejects. A program whose derivative
        // could not be compiled is rejected when it is compiled itself, so such a source is rejected here, unless the
        // transposition leaves the cotangent where it lives (i.e., the placement moves nothing and its input varies
        // along the manual axes of the region or carries reduction state over them, refer to
        // `PlaceOperation::transposes_in_place`). An in-place source names only the storage of the value, so it is
        // lowered there like anywhere else.
        let is_inside_manual_region = !lowerer.collective_state().bound_manual_axes().is_empty();
        if is_inside_manual_region
            && self.source().is_some_and(PlacementRequest::moves)
            && !self.transposes_in_place(&input_types[0])
        {
            let reason = "the source of a value inside a manual region cannot name devices unless the placement moves \
                          nothing and the value varies along the manual axes of the region or carries reduction state \
                          over them, because transposition would move the cotangent of that value to them";
            return Err(LoweringError::UnsupportedOperation {
                operation: PLACE_OPERATION_NAME.to_string(),
                reason: reason.to_string(),
            }
            .into());
        }

        // A placement without a target leaves its input where it is, so it lowers to nothing, also inside a manual
        // region. Its source only matters to transposition, and its copy policy is ignored like that of any other
        // placement, because a compiled program has no observable buffer identity.
        let Some(target) = self.target() else {
            return Ok(vec![input_values[0]]);
        };

        // The values of a manual region live on the devices of that region. Type inference rejects inputs that are
        // tracked over manual mesh axes, but a value that is created inside the region (e.g., a constant) has no
        // tracked sharding, and constraining it over the whole mesh there would contradict the region. A device target
        // is rejected there as well, because it moves the value onto one device, which a value of the region cannot
        // leave, even when the program mesh consists of that device alone. This rule exists only because those targets
        // move values, so an in-place target, which leaves the value on the devices of the region and changes only its
        // storage, is lowered there like anywhere else.
        if target.moves() && is_inside_manual_region {
            return Err(LoweringError::UnsupportedOperation {
                operation: PLACE_OPERATION_NAME.to_string(),
                reason: "a value inside a manual region cannot be moved to another placement".to_string(),
            }
            .into());
        }
        // A device target is the only device of the program mesh (refer to `validate_placements`), and its output keeps
        // the sharding of its input, so it lowers to no sharding constraint and changes the storage only, unless its
        // input carries reduction state over mesh axes of size one, which the device target completes (refer to
        // `PlacementRequest`). The output sharding is then constrained without that state, so that the compiler
        // completes the pending reduction, which involves a single device and moves no data. An in-place target keeps
        // the sharding of its input exactly, so it lowers to its layout and memory only, as JAX lowers a `device_put`
        // onto a memory space to a memory annotation alone.
        let (input_type, output_type) = (&input_types[0], &output_types[0]);
        let mut value = input_values[0];
        let sharding = match target {
            PlacementRequest::Resolved(target) => Some(target.sharding()),
            PlacementRequest::Device { .. } => {
                output_type.sharding().filter(|&sharding| input_type.sharding() != Some(sharding))
            }
            PlacementRequest::InPlace { .. } => None,
        };
        if let Some(sharding) = sharding {
            let location = lowerer.location();
            let sharding = sharding.to_mlir(location)?;
            value = lowerer
                .block()
                .append_operation(shardy::sharding_constraint(value, sharding, location)?)?
                .result(0)
                .unwrap()
                .as_ref();
        }
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
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        BatchAxis, DataType, Device, DeviceMesh, DimensionBounds, DimensionVariable, Layout, LogicalMesh, Memory,
        MeshAxis, MeshAxisType, NegOperation, PartialValue, Place, PlaceDispatch, Placeholder, Placement, ProgramError,
        ProjectedValue, Reduce, ReductionKind, Shape, ShardMapOptions, Sharding, ShardingDimension, StopGradient,
        TiledLayout, Typed, Value as CoreValue, ValueProjection, batch, differentiate_at, shard_map,
    };
    use ryft_pjrt::Client;

    use crate::sessions::tests::{f32_vector, two_device_mesh};
    use crate::tests::{cpu_client, values_from_bytes, values_to_bytes};
    use crate::{
        Array, ArrayTracer, CompileTracer, CompiledFunction, Constant, FromPjrt, Options,
        PlacementOptions, ProgramBuilder, Session, TracedProgram, Type, Value, compile, compile_with_options, trace,
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

    /// Places `value` on `target` through the `PlaceDispatch` primitive of its value type, without a source and with
    /// the default copy policy, which is how a device target reaches values that `Session::place` resolves otherwise.
    fn place_on<V: CoreValue<Type = ArrayType, Dispatch: PlaceDispatch<ArrayType, V>>>(
        value: &V,
        target: &PlacementRequest,
    ) -> Result<V, ProgramError> {
        V::Dispatch::place(value, Some(target), None, true)
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
        // closed over the dimensions of the target and the compiler cannot shard them along further auto axes. The
        // result is annotated with the target, as compilation places it, rather than with its type.
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
                  func.func @main(%arg0: tensor<4x2xf32>) -> (tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
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
            let target = PlacementRequest::Resolved(target.clone());
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
                    <ArrayTracer as CoreValue>::Dispatch::place(
                        &input,
                        None,
                        source.clone().map(PlacementRequest::from),
                        may_alias,
                    )
                    .unwrap()
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
    fn test_place_operation_lower_device_target() {
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![Device::new(0, 0)],
        )
        .unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4, 2])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2))
            .unwrap();
        let place = |target: PlacementRequest| -> TracedProgram<ArrayType, ArrayType> {
            trace(move |input: ArrayTracer| place_on(&input, &target).unwrap(), input_type.clone()).unwrap()
        };

        // A device target keeps the sharding of its input, so it lowers to no sharding constraint, and it lowers to
        // nothing at all when it keeps the storage of its input.
        let device = Device::new(0, 0);
        assert_eq!(
            place(PlacementRequest::Device { device, memory: Some(Memory::Device), layout: None })
                .to_mlir_module("main")
                .unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                  func.func @main(%arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], replicated={"x"}>}) -> (tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], replicated={"x"}>}) {
                    return %arg0 : tensor<4x2xf32>
                  }
                }
            "#},
        );

        // A device target that changes the layout and the memory of its input constrains the layout and then
        // annotates the memory.
        let target = PlacementRequest::Device {
            device,
            memory: Some(Memory::Host { pinned: true }),
            layout: Some(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]))),
        };
        assert_eq!(
            place(target).to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                  func.func @main(%arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], replicated={"x"}>}) -> (tensor<4x2xf32> {mhlo.layout_mode = "{0,1}", mhlo.memory_kind = "pinned_host", sdy.sharding = #sdy.sharding<@mesh, [{}, {}], replicated={"x"}>}) {
                    %0 = stablehlo.custom_call @LayoutConstraint(%arg0) {operand_layouts = [dense<[0, 1]> : tensor<2xindex>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<4x2xf32>) -> tensor<4x2xf32>
                    %1 = stablehlo.custom_call @annotate_device_placement(%0) {backend_config = "", has_side_effect = true, mhlo.frontend_attributes = {_xla_buffer_placement = "pinned_host"}} : (tensor<4x2xf32>) -> tensor<4x2xf32>
                    return %1 : tensor<4x2xf32>
                  }
                }
            "#},
        );

        // A device target completes a pending reduction along an axis of size one, so its output is constrained to the
        // sharding of its type, without that reduction, and the annotation of the result is that sharding too.
        let unreduced_type = ArrayType::new_static(DataType::F32, [4, 2])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> =
            trace(move |input: ArrayTracer| place_on(&input, &PlacementRequest::from(device)).unwrap(), unreduced_type)
                .unwrap();
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                  func.func @main(%arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x"}>}) -> (tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], replicated={"x"}>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {}], replicated={"x"}> : tensor<4x2xf32>
                    return %0 : tensor<4x2xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_place_operation_lower_in_place_target() {
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![Device::new(0, 0), Device::new(1, 0)],
        )
        .unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4, 2])
            .with_sharding(
                Sharding::new(
                    mesh.logical_mesh().clone(),
                    vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
                )
                .unwrap(),
            )
            .unwrap();

        // An in-place target keeps the sharding of its input, so it lowers to no sharding constraint, only to a layout
        // constraint and a memory annotation.
        let target = PlacementRequest::InPlace {
            memory: Some(Memory::Host { pinned: true }),
            layout: Some(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]))),
        };
        let traced: TracedProgram<ArrayType, ArrayType> =
            trace(move |input: ArrayTracer| place_on(&input, &target).unwrap(), input_type).unwrap();
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x2xf32> {mhlo.layout_mode = "{0,1}", mhlo.memory_kind = "pinned_host", sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
                    %0 = stablehlo.custom_call @LayoutConstraint(%arg0) {operand_layouts = [dense<[0, 1]> : tensor<2xindex>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<4x2xf32>) -> tensor<4x2xf32>
                    %1 = stablehlo.custom_call @annotate_device_placement(%0) {backend_config = "", has_side_effect = true, mhlo.frontend_attributes = {_xla_buffer_placement = "pinned_host"}} : (tensor<4x2xf32>) -> tensor<4x2xf32>
                    return %1 : tensor<4x2xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_place_operation_lower_in_place_target_inside_manual_regions() {
        // A value inside a manual region cannot be moved to another placement, but an in-place target leaves it on the
        // devices of the region and changes only its storage, so it lowers there like anywhere else. The result of the
        // region keeps the memory that its body selects, so the function returns it in that memory.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let target = PlacementRequest::from(Memory::Host { pinned: true });
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            move |x: ArrayTracer| {
                shard_map(
                    |local: ArrayTracer| place_on(&local, &target),
                    x,
                    mesh.clone(),
                    sharding.clone(),
                    sharding.clone(),
                    ShardMapOptions::new(),
                )
                .unwrap()
            },
            ArrayType::new_static(DataType::F32, [4]),
        )
        .unwrap();
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32> {mhlo.memory_kind = "pinned_host", sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %2 = stablehlo.custom_call @annotate_device_placement(%arg1) {backend_config = "", has_side_effect = true, mhlo.frontend_attributes = {_xla_buffer_placement = "pinned_host"}} : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %2 : tensor<2xf32>
                    } : (tensor<4xf32>) -> tensor<4xf32>
                    %1 = stablehlo.custom_call @annotate_device_placement(%0) {backend_config = "", has_side_effect = true, mhlo.frontend_attributes = {_xla_buffer_placement = "pinned_host"}} : (tensor<4xf32>) -> tensor<4xf32>
                    return %1 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_place_operation_lower_rejects_sources_that_name_devices_inside_manual_regions() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client.addressable_devices().unwrap();
        let devices = devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let error = format!(
            "cannot lower `{PLACE_OPERATION_NAME}`: the source of a value inside a manual region cannot name devices \
             unless the placement moves nothing and the value varies along the manual axes of the region or carries \
             reduction state over them, because transposition would move the cotangent of that value to them",
        );
        let moved_error = format!(
            "cannot lower `{PLACE_OPERATION_NAME}`: a value inside a manual region cannot be moved to another \
             placement",
        );

        // Places the invariant value of a `shard_map` body over the first `count` devices with `target` and `source`,
        // compiled forward, compiled in reverse mode, and eagerly, and returns whether each of them succeeded.
        let place = |count: usize, target: Option<PlacementRequest>, source: Option<PlacementRequest>| {
            let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", count, MeshAxisType::Explicit).unwrap()]);
            let program_mesh = DeviceMesh::new(explicit_mesh.unwrap(), devices[..count].to_vec()).unwrap();
            let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("x", count, MeshAxisType::Manual).unwrap()]).unwrap();
            let specification = Sharding::replicated(manual_mesh.clone(), 1);
            let r#type = ArrayType::new_static(DataType::F32, [4])
                .with_sharding(Sharding::replicated(program_mesh.logical_mesh().clone(), 1))
                .unwrap();
            let forward: Result<CompiledFunction<ArrayType, ArrayType>, _> = compile(
                |x: CompileTracer| {
                    let options = ShardMapOptions::new();
                    let body = |local: ProjectedValue<ArrayType, _>| local.place(&target, &source);
                    shard_map(body, x, manual_mesh.clone(), specification.clone(), specification.clone(), options)
                        .unwrap()
                },
                r#type.clone(),
                &session,
                program_mesh.clone(),
            );
            let seed_type = match target.as_ref().and_then(PlacementRequest::explicit_memory) {
                Some(memory) => r#type.clone().with_memory(memory),
                None => r#type.clone(),
            };
            let reverse: Result<CompiledFunction<(ArrayType, ArrayType), ArrayType>, _> = compile(
                |(x, seed): (CompileTracer, CompileTracer)| {
                    let (_, pullback) = differentiate_at(x.into_value())
                        .vjp(|x| {
                            let x = ValueProjection::<ArrayType>::into_projected(x)?;
                            let options = ShardMapOptions::new();
                            let output = shard_map(
                                |local: ProjectedValue<ArrayType, _>| local.place(&target, &source),
                                x,
                                manual_mesh.clone(),
                                specification.clone(),
                                specification.clone(),
                                options,
                            )
                            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
                            Ok(output.into_value())
                        })
                        .unwrap();
                    let cotangent = pullback.apply_materialized(seed.into_value()).unwrap();
                    ValueProjection::<ArrayType>::into_projected(cotangent).unwrap()
                },
                (r#type.clone(), seed_type),
                &session,
                program_mesh.clone(),
            );
            let input = Array::from_host_buffer(
                &session,
                r#type.clone(),
                program_mesh,
                None,
                values_to_bytes(&[1f32, 2., 3., 4.]),
            )
            .unwrap();
            let body = |local: ProjectedValue<ArrayType, _>| local.place(&target, &source);
            let eager =
                shard_map(body, input, manual_mesh, specification.clone(), specification, ShardMapOptions::new());
            [
                forward.map(|_| ()).map_err(|error| error.to_string()),
                reverse.map(|_| ()).map_err(|error| error.to_string()),
                eager.map(|_| ()).map_err(|error| error.to_string()),
            ]
        };

        // A source that names devices (i.e., the placement of the value over the devices of the region, or the sole
        // device of a region of one device) would become the target of the transposed placement of this invariant
        // value, which would move its cotangent, so it is rejected forward and eagerly, whatever the target is. The
        // program of reverse mode keeps only the transposed placement, whose target is rejected. The same sources of a
        // value that varies along the manual axis are accepted on every path (refer to
        // `test_session_place_agrees_across_paths_for_manual_regions_and_host_values`).
        let pinned = Some(PlacementRequest::from(Memory::Host { pinned: true }));
        for count in [1, 2] {
            let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("x", count, MeshAxisType::Manual).unwrap()]).unwrap();
            let manual_devices = DeviceMesh::new(manual_mesh.clone(), devices[..count].to_vec()).unwrap();
            let resolved = Placement::new(manual_devices, Sharding::replicated(manual_mesh, 1)).unwrap();
            let mut sources = vec![PlacementRequest::Resolved(resolved)];
            if count == 1 {
                sources.push(PlacementRequest::from(devices[0]));
            }
            for source in sources {
                for target in [pinned.clone(), None] {
                    assert_eq!(
                        place(count, target.clone(), Some(source.clone())),
                        [Err(error.clone()), Err(moved_error.clone()), Err(error.clone())],
                    );
                }
            }

            // An in-place source names only the storage of the value, and no source names nothing, so both compile
            // forward and in reverse mode, and they execute eagerly.
            for source in [Some(PlacementRequest::from(Memory::Device)), None] {
                for target in [pinned.clone(), None] {
                    assert_eq!(place(count, target, source.clone()), [Ok(()), Ok(()), Ok(())]);
                }
            }
        }
    }

    #[test]
    fn test_place_operation_lower_executes_in_place_targets() {
        // A compiled function moves its result to the memory of an in-place target on the devices and with the
        // distribution of its input.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap();
        let pinned = Memory::Host { pinned: true };
        let compiled: CompiledFunction<ArrayType, ArrayType> =
            compile(move |input| input.place(&pinned, None).unwrap(), input_type.clone(), &session, mesh.clone())
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
        assert_eq!(output.r#type().as_ref(), &input_type.with_memory(pinned));
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(output.sharding(), &sharding);
        for shard in output.shards() {
            assert_eq!(shard.buffer().unwrap().memory().unwrap().kind().unwrap(), "pinned_host");
        }
        assert_eq!(shard_values(&output), vec![vec![1., 2.], vec![3., 4.]]);
        assert!(output.is_committed());
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
    fn test_place_operation_lower_executes_auto_splits_with_unconstrained_dimensions_over_other_auto_axes() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("r", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let devices = client.addressable_devices().unwrap();
        let devices = devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices).unwrap();
        let input_type = ArrayType::new_static(DataType::I32, [4, 4]);
        let input_sharding = Sharding::new(
            logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["r"])],
        )
        .unwrap();
        let target_sharding =
            Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::unconstrained()])
                .unwrap();

        // The compiler would keep the split of the input along `r` on the unconstrained dimension, but the runtime
        // placement of the result reads that dimension as replicated, so the executable returns the result with the
        // split along `x` that the target requires and the unconstrained dimension replicated.
        let target = Placement::new(mesh.clone(), target_sharding.clone()).unwrap();
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
            move |input| input.place(&target, None).unwrap(),
            input_type.clone(),
            &session,
            Options::new(mesh.clone()).with_input_shardings(vec![input_sharding.clone()]),
        )
        .unwrap();
        let values = (0..16).collect::<Vec<i32>>();
        let input =
            Array::from_host_buffer(&session, input_type, mesh, Some(input_sharding.clone()), values_to_bytes(&values))
                .unwrap();
        let output = compiled.executable().execute(&session, input).unwrap();
        assert_eq!(output.sharding(), &target_sharding);
        assert_eq!(
            output
                .shards()
                .iter()
                .map(|shard| values_from_bytes::<i32>(&shard.copy_to_host_bytes().unwrap()))
                .collect::<Vec<_>>(),
            vec![
                (0..8).collect::<Vec<_>>(),
                (0..8).collect::<Vec<_>>(),
                (8..16).collect::<Vec<_>>(),
                (8..16).collect::<Vec<_>>(),
            ],
        );
        assert_eq!(output.copy_to_host().unwrap().elements::<i32>().unwrap(), values);
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
    fn test_place_operation_lower_executes_auto_splits_through_identity_operations() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();

        // An operation that returns its input unchanged (i.e., `stop_gradient`) keeps the placement of a result of the
        // zero data type, which has no buffer whose placement the compiler could report.
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let input_type = ArrayType::new_static(DataType::Zero, [4]);
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let target = Placement::new(mesh.clone(), sharded.clone()).unwrap();
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
            move |input: CompileTracer| input.place(&target, None).unwrap().stop_gradient().unwrap(),
            input_type.clone(),
            &session,
            Options::new(mesh.clone()).with_input_shardings(vec![replicated.clone()]),
        )
        .unwrap();
        let input = Array::from_host_buffer(&session, input_type, mesh, Some(replicated), []).unwrap();
        let output = compiled.executable().execute(&session, input).unwrap();
        assert_eq!(output.sharding(), &sharded);

        // It also keeps the split along an auto axis of a result whose reduction state along an explicit axis keeps
        // the compiler from choosing its placement.
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("r", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let devices = client.addressable_devices().unwrap();
        let devices = devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices).unwrap();
        let input_sharding = Sharding::replicated(logical_mesh.clone(), 1).with_reduced_axes(["r"]).unwrap();
        let reduced = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"])])
            .unwrap()
            .with_reduced_axes(["r"])
            .unwrap();
        let input_type = ArrayType::new_static(DataType::I32, [4]).with_sharding(input_sharding.clone()).unwrap();
        let target = Placement::new(mesh.clone(), reduced.clone()).unwrap();
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile(
            move |input: CompileTracer| input.place(&target, None).unwrap().stop_gradient().unwrap(),
            input_type.clone(),
            &session,
            mesh.clone(),
        )
        .unwrap();
        let input = Array::from_host_buffer(
            &session,
            input_type,
            mesh,
            Some(input_sharding),
            values_to_bytes(&[1i32, 2, 3, 4]),
        )
        .unwrap();
        let output = compiled.executable().execute(&session, input).unwrap();
        assert_eq!(output.sharding(), &reduced);
        assert_eq!(output.copy_to_host().unwrap().elements::<i32>().unwrap(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_place_operation_eager_partial_evaluation_may_alias() {
        // The XLA operation family holds the placement through its projected array member. Specializing a program on a
        // known input of a placement that forces a copy keeps that placement residual, so that two executions of the
        // specialized program return distinct buffers, while the negation of the unknown input stays residual too.
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap()],
        )
        .unwrap();
        let input = f32_vector(&session, &mesh, &[1., 2.]);
        let r#type = Type::Array(input.r#type().into_owned());
        let mut builder = ProgramBuilder::new();
        let known = builder.add_input(r#type.clone());
        let unknown = builder.add_input(r#type.clone());
        let copy = builder
            .add_instruction(PlaceOperation::new(None, None).with_may_alias(false), Vec::new(), vec![known], None)
            .unwrap()[0];
        let negation = builder.add_instruction(NegOperation::new(), Vec::new(), vec![unknown], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![copy, negation], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let input = Value::Array(input);
        let evaluation = program
            .partially_evaluate_in_context(
                &session,
                &[PartialValue::Known(input.clone()), PartialValue::Unknown(r#type)],
            )
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}], \
                    %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %2:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [may_alias=false] %1
                    %3:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = neg %0
                in (%2, %3)"
            },
        );
        assert_eq!(evaluation.outputs().iter().map(|output| output.is_known()).collect::<Vec<_>>(), vec![false, false]);
        let first = evaluation.interpret(&session, std::slice::from_ref(&input)).unwrap();
        let second = evaluation.interpret(&session, std::slice::from_ref(&input)).unwrap();
        let (Value::Array(first), Value::Array(second), Value::Array(input)) = (&first[0], &second[0], &input) else {
            unreachable!("placements of arrays return arrays");
        };
        assert_eq!(shard_values(first), vec![vec![1., 2.]]);
        assert_eq!(shard_values(second), vec![vec![1., 2.]]);
        let buffer = |array: &Array| array.shards()[0].buffer().unwrap().clone();
        assert!(!Arc::ptr_eq(&buffer(first), &buffer(second)));
        assert!(!Arc::ptr_eq(&buffer(first), &buffer(input)));
    }

    #[test]
    fn test_place_operation_eager_device_target() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let (first_mesh, second_mesh) = single_device_meshes(client);
        let (first_device, second_device) = (first_mesh.devices()[0], second_mesh.devices()[0]);
        let input = f32_vector(&session, &first_mesh, &[1., 2.]);
        let traced = |target: PlacementRequest, input: &Array| -> TracedProgram<ArrayType, ArrayType> {
            trace(move |input: ArrayTracer| place_on(&input, &target).unwrap(), input.r#type().into_owned()).unwrap()
        };
        let interpret = |traced: TracedProgram<ArrayType, ArrayType>, input: &Array| {
            let Value::Array(output) =
                traced.program().interpret_in_context(&session, Value::Array(input.clone())).unwrap()
            else {
                panic!("placing an array must produce an array");
            };
            output
        };

        // A staged device target resolves against the array that it executes on, as a bare device destination of
        // `Session::place` does. The device of the array keeps the array on its own mesh.
        let target = PlacementRequest::Device { device: first_device, memory: Some(Memory::Device), layout: None };
        let traced_program = traced(target, &input);
        assert_eq!(
            traced_program.program().to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [target=device(0, memory=Device)] %0
                in (%1)"
            },
        );
        let output = interpret(traced_program, &input);
        assert_eq!(output.mesh(), &first_mesh);
        assert_eq!(output.sharding(), input.sharding());
        assert_eq!(output.r#type().memory(), Memory::Device);
        assert!(output.is_committed());
        assert_eq!(shard_values(&output), vec![vec![1., 2.]]);

        // Another device moves the array onto that device with its logical mesh and sharding, which is what the type
        // of the staged result records, and in the memory that the target names.
        let memory = Memory::Host { pinned: true };
        let target = PlacementRequest::Device { device: second_device, memory: Some(memory), layout: None };
        let output = interpret(traced(target, &input), &input);
        assert_eq!(output.mesh(), &second_mesh);
        assert_eq!(output.sharding(), input.sharding());
        assert_eq!(output.r#type().memory(), memory);
        assert_eq!(output.shards()[0].buffer().unwrap().memory().unwrap().kind().unwrap(), "pinned_host");
        assert!(output.is_committed());
        assert_eq!(shard_values(&output), vec![vec![1., 2.]]);

        // That is how a direct placement of the array on the device places it, so both report the same mesh, sharding,
        // and type, and so they do for an array with a pending reduction along `x`, which has size one and which the
        // device completes.
        let direct = session
            .place(input.clone(), PlacementOptions::to(PlacementRequest::Device { device: second_device, memory: Some(memory), layout: None }))
            .unwrap();
        assert_eq!(
            (output.mesh(), output.sharding(), output.r#type()),
            (direct.mesh(), direct.sharding(), direct.r#type())
        );
        let unreduced = Array::from_host_buffer(
            &session,
            input
                .r#type()
                .into_owned()
                .with_sharding(input.sharding().clone().with_unreduced_axes(["x"]).unwrap())
                .unwrap(),
            first_mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();
        let output = interpret(traced(PlacementRequest::from(second_device), &unreduced), &unreduced);
        let direct = session.place(unreduced, PlacementOptions::to(second_device)).unwrap();
        assert_eq!(
            (output.mesh(), output.sharding(), output.r#type()),
            (direct.mesh(), direct.sharding(), direct.r#type())
        );
        assert_eq!((output.mesh(), output.sharding()), (&second_mesh, input.sharding()));
        assert_eq!(shard_values(&output), vec![vec![1., 2.]]);

        // An array on more devices cannot keep its sharding on one device, so it is gathered onto the target device,
        // replicated over the canonical mesh of one device. That is the type that the staged result records, so the
        // replay of the staged program on such an array gathers it as a direct placement of the array does.
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let sharded = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding).unwrap(),
            mesh,
            None,
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        let traced_program = traced(PlacementRequest::from(second_device), &sharded);
        assert_eq!(
            traced_program.program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] .
                let %1:f32[4][sharding={mesh<['device'=1:auto]>, [{}]}] = place [target=device(1)] %0
                in (%1)"
            },
        );
        let output = interpret(traced_program, &sharded);
        let direct = session.place(sharded, PlacementOptions::to(second_device)).unwrap();
        let gathered = Sharding::replicated(LogicalMesh::single_device(), 1);
        assert_eq!((output.mesh().devices(), output.sharding()), ([second_device].as_slice(), &gathered));
        assert_eq!(
            (output.mesh(), output.sharding(), output.r#type(), output.is_committed()),
            (direct.mesh(), direct.sharding(), direct.r#type(), direct.is_committed()),
        );
        assert_eq!(shard_values(&output), vec![vec![1., 2., 3., 4.]]);
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
    fn test_place_operation_eager_vjp_device_target() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let (source_mesh, target_mesh) = single_device_meshes(client);
        let device = target_mesh.devices()[0];
        let primal = f32_vector(&session, &source_mesh, &[1., 2., 3.]);

        // The primal moves onto the device, and the source of the placement is where the primal concretely lives, so
        // the pullback moves the cotangent back to the source mesh.
        let target = PlacementRequest::from(device);
        let (value, pullback) = differentiate_at(primal.clone()).vjp(|x| place_on(&x, &target)).unwrap();
        assert_eq!(value.mesh().devices(), [device]);
        assert_eq!(shard_values(&value), vec![vec![1., 2., 3.]]);
        let cotangent = Array::from_host_buffer(
            &session,
            value.r#type().into_owned(),
            value.mesh().clone(),
            None,
            values_to_bytes(&[4f32, 5., 6.]),
        )
        .unwrap();
        let input_cotangent = pullback.apply_materialized(cotangent).unwrap();
        assert_eq!(input_cotangent.mesh(), &source_mesh);
        assert_eq!(input_cotangent.r#type(), primal.r#type());
        assert_eq!(shard_values(&input_cotangent), vec![vec![4., 5., 6.]]);
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
    fn test_place_operation_eager_in_place_target() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let pinned = Memory::Host { pinned: true };
        let primal = f32_vector(&session, &mesh, &[1., 2., 3.]);
        let tangent = f32_vector(&session, &mesh, &[1., 0., -1.]);
        let memory_kinds = |array: &Array| {
            array
                .shards()
                .iter()
                .map(|shard| shard.buffer().unwrap().memory().unwrap().kind().unwrap().to_string())
                .collect::<Vec<_>>()
        };

        // The primal and the tangent both move to the memory on the devices where they live.
        let (value, tangent): (Array, Array) =
            differentiate_at(primal.clone()).jvp(tangent, |x| x.place(&pinned, None)).unwrap();
        for (output, values) in [(&value, vec![1., 2., 3.]), (&tangent, vec![1., 0., -1.])] {
            assert_eq!(output.mesh(), &mesh);
            assert_eq!(output.sharding(), primal.sharding());
            assert_eq!(output.r#type().as_ref(), &primal.r#type().into_owned().with_memory(pinned));
            assert_eq!(memory_kinds(output), vec!["pinned_host"; 2]);
            assert_eq!(shard_values(output), vec![values.clone(), values]);
        }

        // The pullback returns the cotangent to the memory of the input, on the same devices.
        let (value, pullback) = differentiate_at(primal.clone()).vjp(|x| x.place(&pinned, None)).unwrap();
        assert_eq!(value.r#type().memory(), pinned);
        let input_cotangent = pullback.apply_materialized(value).unwrap();
        assert_eq!(input_cotangent.mesh(), &mesh);
        assert_eq!(input_cotangent.r#type(), primal.r#type());
        assert_eq!(memory_kinds(&input_cotangent), memory_kinds(&primal));
        assert_eq!(shard_values(&input_cotangent), vec![vec![1., 2., 3.]; 2]);
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
    fn test_place_operation_eager_batch_device_target() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let (source_mesh, target_mesh) = single_device_meshes(client);
        let device = target_mesh.devices()[0];
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2, 3]),
            source_mesh,
            None,
            values_to_bytes(&[1f32, 2., 3., 4., 5., 6.]),
        )
        .unwrap();

        // Batching passes the device target through unchanged, so the packed result moves onto the device.
        let target = PlacementRequest::from(device);
        let output: Array =
            batch(|x| place_on(&x, &target), input, BatchAxis::new(0), BatchAxis::new(0), None).unwrap();
        assert_eq!(output.mesh().devices(), [device]);
        assert!(output.is_committed());
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

    #[test]
    fn test_place_operation_eager_batch_vjp_across_device_sets() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let source_mesh = DeviceMesh::new(logical_mesh.clone(), devices[..2].to_vec()).unwrap();
        let target_mesh = DeviceMesh::new(logical_mesh.clone(), devices[2..].to_vec()).unwrap();
        let sharding = Sharding::new(
            logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2, 3]),
            source_mesh.clone(),
            Some(sharding.clone()),
            values_to_bytes(&[1f32, 2., 3., 4., 5., 6.]),
        )
        .unwrap();
        let target = Placement::new(target_mesh.clone(), Sharding::replicated(logical_mesh.clone(), 1)).unwrap();

        // The batch is split along `x` on the first two devices, so each batch item lives on one of them. The
        // placement that a batch item reports records that split, and the pullback restores it, so a cotangent that
        // differs for each item comes back on the first two devices with the batch still split along `x`. The primal
        // of each item is replicated on the last two devices, as its explicit target says.
        let (value, cotangent): (Array, Array) = batch(
            |x| {
                let (value, pullback) = differentiate_at(x).vjp(|x| x.place(&target, None))?;
                let cotangent = pullback.apply_materialized(value.clone())?;
                Ok((value, cotangent))
            },
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(value.mesh(), &target_mesh);
        assert_eq!(value.sharding(), &Sharding::replicated(logical_mesh, 2));
        assert_eq!(cotangent.mesh(), &source_mesh);
        assert_eq!(cotangent.sharding(), &sharding);
        assert_eq!(shard_values(&cotangent), vec![vec![1., 2., 3.], vec![4., 5., 6.]]);
    }

    #[test]
    fn test_place_operation_eager_batch_gradient_across_device_sets() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();

        // The batch is split along `x` on the first two devices, so each of them holds one `[1, 2]` batch item, and
        // the gradient of each item comes back split in the same way, whatever the type of `x`, rather than as the
        // whole batch on each of those devices.
        for axis_type in [MeshAxisType::Auto, MeshAxisType::Explicit] {
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let source_mesh = DeviceMesh::new(logical_mesh.clone(), devices[..2].to_vec()).unwrap();
            let target_mesh = DeviceMesh::new(logical_mesh.clone(), devices[2..].to_vec()).unwrap();
            let sharding = Sharding::new(
                logical_mesh.clone(),
                vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
            )
            .unwrap();
            let input = Array::from_host_buffer(
                &session,
                ArrayType::new_static(DataType::F32, [2, 2]),
                source_mesh.clone(),
                Some(sharding.clone()),
                values_to_bytes(&[1f32, 2., 3., 4.]),
            )
            .unwrap();
            let target = Placement::new(target_mesh, Sharding::replicated(logical_mesh, 1)).unwrap();
            let gradient: Array = batch(
                |x| Ok(differentiate_at(x).gradient(|x| x.place(&target, None)?.reduce(&[0], ReductionKind::Sum))?),
                input,
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
            assert_eq!(gradient.mesh(), &source_mesh);
            assert_eq!(gradient.sharding(), &sharding);
            assert_eq!(
                gradient
                    .shards()
                    .iter()
                    .map(|shard| shard.buffer().unwrap().dimensions().unwrap().to_vec())
                    .collect::<Vec<_>>(),
                vec![vec![1, 2], vec![1, 2]],
            );
            assert_eq!(shard_values(&gradient), vec![vec![1., 1.], vec![1., 1.]]);
            assert!(gradient.is_committed());
        }
    }

    #[test]
    fn test_place_operation_eager_batch_gradient_across_differently_named_meshes() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();

        // The batch is split along the explicit axis `x` of the mesh of the first two devices, and each batch item is
        // moved to the mesh of the other two devices, whose only axis is the explicit axis `y`. The cotangent of the
        // sum is the same for every batch item and lives on the mesh of `y`, so it gains its batch dimension there
        // replicated, and the transposed placement moves it back split along `x`.
        let source_logical_mesh =
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let target_logical_mesh =
            LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let source_mesh = DeviceMesh::new(source_logical_mesh.clone(), devices[..2].to_vec()).unwrap();
        let target_mesh = DeviceMesh::new(target_logical_mesh.clone(), devices[2..].to_vec()).unwrap();
        let sharding = Sharding::new(
            source_logical_mesh,
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2, 2]),
            source_mesh.clone(),
            Some(sharding.clone()),
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        let target = Placement::new(target_mesh, Sharding::replicated(target_logical_mesh, 1)).unwrap();
        let gradient: Array = batch(
            |x| Ok(differentiate_at(x).gradient(|x| x.place(&target, None)?.reduce(&[0], ReductionKind::Sum))?),
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(gradient.mesh(), &source_mesh);
        assert_eq!(gradient.sharding(), &sharding);
        assert_eq!(
            gradient
                .shards()
                .iter()
                .map(|shard| shard.buffer().unwrap().dimensions().unwrap().to_vec())
                .collect::<Vec<_>>(),
            vec![vec![1, 2], vec![1, 2]],
        );
        assert_eq!(shard_values(&gradient), vec![vec![1., 1.], vec![1., 1.]]);
        assert!(gradient.is_committed());
    }
}
