use ryft_core::macros::check_count;
use ryft_core::{
    ArrayType, PLACE_OPERATION_NAME, PlaceDispatch, PlaceOperation, PlacementRequest, ProgramError, ValueDomainDispatch,
};
use ryft_mlir::dialects::shardy;
use ryft_mlir::{Block, Operation as MlirOperation, Value as MlirValue, ValueRef};

use crate::arrays::placement::place_operation_bindings;
use crate::lowering::sharding::{OpenDimensions, ShardingExtension};
use crate::lowering::types::MemoryLoweringExtension;
use crate::operations::manipulation::conversions::ArrayTypeLayoutConversionLoweringExtension;
use crate::{Array, Error, LowerableOperation, LowerableOperationFamily, Lowerer, LoweringError};

impl PlaceDispatch<ArrayType, Array> for ValueDomainDispatch {
    fn place_operation(input: &Array, operation: PlaceOperation) -> Result<Array, ProgramError> {
        Self::place_operations(vec![(input.clone(), operation)]).map(|mut outputs| outputs.remove(0))
    }

    fn place_operations(bindings: Vec<(Array, PlaceOperation)>) -> Result<Vec<Array>, ProgramError> {
        let bindings = bindings.into_iter().map(|(input, operation)| (input.session().clone(), input, operation));
        place_operation_bindings(bindings.collect())
    }
}

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
        // sharding constraints. What remains lowers to a sharding constraint for the target sharding, followed by a
        // layout constraint when the layout changes and by a memory annotation when the memory changes. Unlike the
        // sharding of a type (e.g., that of `reshard`), a target states the placement along every mesh axis, including
        // auto axes, so the constraint is closed over the dimensions that the target specifies, like that of
        // `constrain_sharding`. The layout is constrained first, so that the conversion runs before the value leaves
        // device memory. JAX 0.12.0.1 returns the inputs from its CPU primitive lowering. Its GPU and TPU lowerings can
        // annotate sharding and memory for a requested memory kind. Ryft rejects changes of reduction state before
        // lowering. The copy policy of the operation (i.e., `PlaceOperation::may_alias`) is ignored, so the lowering
        // emits no copy instruction even for a forced copy, because a compiled program has no observable buffer
        // identity. A forced copy is still honored where it can be observed. Every public output of an execution of the
        // program is written to a new buffer that the execution allocates, unless it is paired with a donated input
        // whose buffers the caller gave up and no other handle can observe (refer to the donation pairing of
        // `ProgramLoweringOptions::with_reference_states`), so no public output ever shares
        // storage with an array that the caller holds. An eager placement is not lowered here, and it applies the copy
        // policy to its output itself (refer to `Session::bind` and `Session::place`).
        check_count!("input", input_values, 1, ProgramError);
        check_count!("input", input_types, 1, ProgramError);
        check_count!("output", output_types, 1, ProgramError);

        // A source only states where the input lives, and the forward program moves nothing to it, so it is lowered to
        // nothing, also when it names devices inside a manual region or outside the program mesh, as JAX elides the
        // source of a `device_put` inside `jit` and `shard_map`. Transposition turns such a source into the target of
        // the transposed placement, which moves the cotangent there, so the program of reverse mode is rejected by the
        // target rule below when that target leaves the region, and by the program mesh check when it leaves the mesh
        // (refer to `LoweringError::PlacementOutsideProgramMesh`), when that program is compiled itself.
        let is_inside_manual_region = !lowerer.lowering_state().bound_manual_axes().is_empty();

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
            let sharding = sharding.lower(OpenDimensions::None, location)?;
            value = lowerer
                .block()
                .append_operation(shardy::sharding_constraint(value, sharding, location)?)?
                .result(0)
                .unwrap()
                .as_ref();
        }
        if output_type.layout().is_some() && output_type.layout() != input_type.layout() {
            value = output_type.lower_layout_conversion(value, lowerer)?;
        }
        if output_type.memory() != input_type.memory() {
            value = output_type.memory().lower_transfer(
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
        Add, BatchAxis, Context, DataType, Device, DeviceMesh, DifferentiableType, Differentiate, DifferentiationError,
        DimensionBounds, DimensionVariable, EmptyRegionDriver, InterpretableOperation, Layout, LogicalMesh, Memory,
        MeshAxis, MeshAxisType, Neg, NegOperation, Operation as CoreOperation, PartialTracer, PartialValue, Place,
        PlaceDispatch, Placed, Placeholder, Placement, ProgramBuilder, ProgramError, ProjectedValue, Reduce,
        ReductionKind, Shape, ShardMapOptions, Sharding, ShardingDimension, Sin, StopGradient, TiledLayout, TypeError,
        Typed, Value as CoreValue, ValueProjection, batch, differentiate_at, shard_map,
    };
    use ryft_pjrt::Client;

    use crate::operations::Operation;
    use crate::sessions::tests::{f32_vector, two_device_mesh};
    use crate::tests::{cpu_client, native_buffer_addresses, shard_values, values_from_bytes, values_to_bytes};
    use crate::{
        Array, ArrayTracer, CompileTracer, CompiledFunction, Constant, FromPjrt, Options, PlacementOptions,
        ProgramLoweringExtension, ProgramLoweringOptions, Session, StatefulCompileTracer, TracedProgram, Type, Value,
        compile, compile_statefully, compile_with_options, trace,
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

    /// Places `value` on `target` through the `PlaceDispatch` primitive of its value type, without a source and with
    /// the default copy policy, which is how a device target reaches values that `Session::place` resolves otherwise.
    fn place_on<V: CoreValue<Type = ArrayType, Dispatch: PlaceDispatch<ArrayType, V>>>(
        value: &V,
        target: &PlacementRequest,
    ) -> Result<V, ProgramError> {
        V::Dispatch::place(value, Some(target), None, true)
    }

    /// Uploads `values` as an `f32` array of shape `shape` without a mesh or a sharding, so the array lives on the
    /// first device of `session` uncommitted, which makes it relocatable.
    fn upload(session: &Session, shape: &[usize], values: &[f32]) -> Array {
        let r#type = ArrayType::new_static(DataType::F32, shape.to_vec());
        Array::from_host_buffer(session, r#type, None, None, values_to_bytes(values)).unwrap()
    }

    /// Returns the primal `[1, 2]`, committed to `device` and replicated over the explicit axis `other` of size 1, and
    /// the tangent `[3, 4]`, uploaded with [`upload`], for the JVP tests of placements whose primal lives on a named
    /// mesh.
    fn named_primal_and_tangent(session: &Session, device: Device) -> (Array, Array) {
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("other", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![device]).unwrap();
        let placement = Placement::new(mesh, Sharding::replicated(logical_mesh, 1)).unwrap();
        let primal = upload(session, &[2], &[1., 2.]).place(&placement, None).unwrap();
        (primal, upload(session, &[2], &[3., 4.]))
    }

    /// Returns the devices on which `array` lives, its `f32` elements, and whether it is committed, which is what the
    /// eager transform tests of placements observe of an output.
    fn observe_placement(array: Array) -> (Vec<Device>, Vec<f32>, bool) {
        let elements = array.copy_to_host().unwrap().elements::<f32>().unwrap();
        (array.mesh().devices().to_vec(), elements, array.is_committed())
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
            traced.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap(),
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
            traced.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap(),
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
        // output is annotated with the target, as compilation places it, rather than with its type.
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
            traced.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap(),
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
            copying.to_string(),
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
        assert_eq!(
            copying.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap(),
            place(true).to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap()
        );
        assert_eq!(
            copying.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap(),
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
            traced.to_string(),
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
        assert_eq!(
            traced.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap(),
            place(None, true).to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap()
        );
        assert_eq!(
            traced.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap(),
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
                .to_rendered_mlir("main", &ProgramLoweringOptions::new())
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
            place(target).to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap(),
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
        // sharding of its type, without that reduction, and the annotation of the output is that sharding too.
        let unreduced_type = ArrayType::new_static(DataType::F32, [4, 2])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> =
            trace(move |input: ArrayTracer| place_on(&input, &PlacementRequest::from(device)).unwrap(), unreduced_type)
                .unwrap();
        assert_eq!(
            traced.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap(),
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
            traced.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap(),
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
        // devices of the region and changes only its storage, so it lowers there like anywhere else. The output of the
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
            traced.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap(),
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
    fn test_place_operation_lower_admits_sources_that_name_devices_inside_manual_regions() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client.addressable_devices().unwrap();
        let devices = devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
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

        // A source that names devices (i.e., the placement of the value over the devices of the region, the sole device
        // of a region of one device, or a device outside the program mesh) moves nothing forward, so it compiles
        // forward and executes eagerly, whatever the target is, as pinned JAX compiles and runs a `device_put` with a
        // `src` outside a `shard_map`. The program of reverse mode keeps only the transposed placement, whose target is
        // that source, and is rejected for it, as pinned JAX rejects the gradient. The forward and eager paths never
        // reject such a source because of that transposition. The same sources of a value that varies along the manual
        // axis are accepted on every path (refer to
        // `test_session_place_agrees_across_paths_for_sources_inside_shard_map_bodies`).
        let pinned = Some(PlacementRequest::from(Memory::Host { pinned: true }));
        for count in [1, 2] {
            let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("x", count, MeshAxisType::Manual).unwrap()]).unwrap();
            let manual_devices = DeviceMesh::new(manual_mesh.clone(), devices[..count].to_vec()).unwrap();
            let resolved = Placement::new(manual_devices, Sharding::replicated(manual_mesh, 1)).unwrap();
            let mut sources = vec![PlacementRequest::Resolved(resolved)];
            if count == 1 {
                sources.push(PlacementRequest::from(devices[0]));
                sources.push(PlacementRequest::from(devices[1]));
            }
            // The transposed placement of a source on the devices of the region is rejected for moving a value inside
            // it, and that of a source outside the program mesh for naming devices that the program does not run on.
            let outside_error = format!(
                "`{PLACE_OPERATION_NAME}` cannot move a value to the devices [1] of mesh `['x'=1:manual]` inside a \
                 program compiled for the devices [0] of mesh `['x'=1:explicit]`",
            );
            for source in sources {
                let reverse_error =
                    if source == PlacementRequest::from(devices[1]) { &outside_error } else { &moved_error };
                for target in [pinned.clone(), None] {
                    assert_eq!(
                        place(count, target.clone(), Some(source.clone())),
                        [Ok(()), Err(reverse_error.clone()), Ok(())],
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
        // A compiled function moves its output to the memory of an in-place target on the devices and with the
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
    fn test_place_operation_lower_executes_device_targets_of_stateful_functions() {
        // A stateful function traces values of the array IR type universe, which `Session::place` does not accept, so
        // it stages a device target through the `PlaceDispatch` primitive of `Place` directly, and the compiled
        // function moves its output to the memory of that target on the device.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let (mesh, _) = single_device_meshes(client);
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let memory = Memory::Host { pinned: true };
        let target = PlacementRequest::Device { device: mesh.devices()[0], memory: Some(memory), layout: None };
        let compiled = compile_statefully::<_, Type, Type>(
            |input| Ok(<StatefulCompileTracer as CoreValue>::Dispatch::place(&input, Some(&target), None, true)?),
            Type::Array(input_type.clone()),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let input =
            Array::from_host_buffer(&session, input_type, mesh.clone(), None, values_to_bytes(&[1f32, 2.])).unwrap();
        let Value::Array(output) = compiled.execute_statefully_and_wait(&session, Value::Array(input)).unwrap() else {
            panic!("placing an array must produce an array");
        };
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(output.r#type().memory(), memory);
        assert!(output.is_committed());
        assert_eq!(shard_values(&output), vec![vec![1., 2.]]);
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

        // So is a target that leaves a dimension unconstrained, which an array stores replicated, as the unconstrained
        // sharding override of a call is.
        let unconstrained = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::Unconstrained]).unwrap();
        assert_eq!(
            place(unconstrained).unwrap().source_program().program().to_string(),
            indoc! {"
                lambda %0:f32[n] .
                let %1:f32[n][sharding={mesh<['x'=2:auto]>, [{?}]}] = place [
                    target={mesh<['x'=2:auto]>, devices=[0, 1], sharding={mesh<['x'=2:auto]>, [{?}]}},
                ] %0
                in (%1)"
            },
        );
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

        // The type of the output drops the split along the auto axis `x`, and the reduction state along the explicit
        // axis `r` keeps the compiler from choosing the placement of the output, so the output is placed as the target
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

        // The output keeps the split of its first dimension even though an unconstrained dimension leaves the rest of
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
        // placement of the output reads that dimension as replicated, so the executable returns the output with the
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

        // An output of the zero data type has no buffer whose placement the compiler could report, so its placement
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

        // A placement without a target leaves the placed value where it is, so the output keeps the target placement.
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

        // An operation that returns its input unchanged (i.e., `stop_gradient`) keeps the placement of an output of the
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

        // It also keeps the split along an auto axis of an output whose reduction state along an explicit axis keeps
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
            let Value::Array(output) = traced.interpret_in_context(&session, Value::Array(input.clone())).unwrap()
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
            traced_program.to_string(),
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
        // of the staged output records, and in the memory that the target names.
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
            .place(
                input.clone(),
                PlacementOptions::to(PlacementRequest::Device {
                    device: second_device,
                    memory: Some(memory),
                    layout: None,
                }),
            )
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
        // replicated over the anonymous mesh of one device at runtime. The staged type carries no sharding. Replaying
        // the staged program gathers the input as a direct placement of the array does.
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
            traced_program.to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] .
                let %1:f32[4] = place [target=device(1)] %0
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
    fn test_place_operation_eager_admits_relocatable_sources() {
        // A targetless placement of an uncommitted upload on the first device with a source on a named mesh of the
        // second device is admitted, because the upload may be placed there first, and leaves the upload where it is,
        // uncommitted. Eager placement, linearization, the fused JVP, batching, and the direct interpretation of the
        // operation all admit it by the same rule, as pinned JAX admits `device_put` with such a source eagerly, under
        // `jvp`, and under `vmap`.
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("other", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let source = Placement::new(mesh, Sharding::replicated(logical_mesh, 1)).unwrap();
        let options = || PlacementOptions::default().with_source(source.clone());
        let first = devices[0];

        assert_eq!(
            session
                .place(upload(&session, &[2], &[3., 4.]), options())
                .map(observe_placement)
                .map_err(|error| error.to_string()),
            Ok((vec![first], vec![3., 4.], false)),
        );
        assert_eq!(
            upload(&session, &[2], &[3., 4.])
                .place(&Option::<Placement>::None, &source)
                .map(observe_placement)
                .map_err(|error| error.to_string()),
            Ok((vec![first], vec![3., 4.], false)),
        );
        let (primal, pushforward) = differentiate_at(upload(&session, &[2], &[3., 4.]))
            .linearize(|x| x.place(&Option::<Placement>::None, &source))
            .unwrap();
        assert_eq!(observe_placement(primal), (vec![first], vec![3., 4.], false));
        assert_eq!(
            observe_placement(pushforward.apply(upload(&session, &[2], &[5., 6.])).unwrap()),
            (vec![first], vec![5., 6.], false),
        );
        let (primal, tangent) = differentiate_at(upload(&session, &[2], &[3., 4.]))
            .jvp(upload(&session, &[2], &[5., 6.]), |x| x.place(&Option::<Placement>::None, &source))
            .unwrap();
        assert_eq!(
            (observe_placement(primal), observe_placement(tangent)),
            ((vec![first], vec![3., 4.], false), (vec![first], vec![5., 6.], false)),
        );
        let batched = batch(
            |x| x.place(&Option::<Placement>::None, &source),
            upload(&session, &[2, 2], &[3., 4., 5., 6.]),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        );
        assert_eq!(
            batched.map(observe_placement).map_err(|error| error.to_string()),
            Ok((vec![first], vec![3., 4., 5., 6.], false)),
        );
        let input = upload(&session, &[2], &[3., 4.]);
        let operation = PlaceOperation::new(None, source.clone());
        let interpreted = InterpretableOperation::interpret(&operation, &input.domain(), &EmptyRegionDriver, &[input])
            .map(|outputs| outputs.into_iter().map(observe_placement).collect::<Vec<_>>())
            .map_err(|error| error.to_string());
        assert_eq!(interpreted, Ok(vec![(vec![first], vec![3., 4.], false)]));

        // The same source is rejected for an array committed to the first device, through `Session::place`,
        // linearization, and `Place`.
        let committed = upload(&session, &[2], &[3., 4.]).to_placement(&session, devices[0]).unwrap();
        let rejection = format!(
            "`{PLACE_OPERATION_NAME}` source sharding `{}` does not match the input sharding `{}`",
            source.sharding(),
            committed.sharding(),
        );
        assert_eq!(
            session.place(committed.clone(), options()).map(|_| ()).map_err(|error| error.to_string()),
            Err(rejection.clone()),
        );
        assert_eq!(
            differentiate_at(committed.clone())
                .linearize(|x| x.place(&Option::<Placement>::None, &source))
                .map(|_| ())
                .map_err(|error| error.to_string()),
            Err(rejection.clone()),
        );
        assert_eq!(
            committed.place(&Option::<Placement>::None, &source).map(|_| ()).map_err(|error| error.to_string()),
            Err(rejection),
        );
    }

    #[test]
    fn test_place_operation_eager_partial_evaluation_may_alias() {
        // The XLA operation family holds the placement through its projected array member. Specializing a program on a
        // known input of a placement that forces a copy keeps that placement residual, so that two executions of the
        // specialized program return distinct buffers, while the negation of the unknown input stays residual too.
        let session = Session::builder().client(cpu_client(1)).build().unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            session.addressable_devices().unwrap(),
        )
        .unwrap();
        let input = f32_vector(&session, &mesh, &[1., 2.]);
        let r#type = Type::Array(input.r#type().into_owned());
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
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
    fn test_place_operation_eager_partial_evaluation_uses_declared_reduction_state() {
        let session = Session::builder().client(cpu_client(1)).build().unwrap();
        let device = session.addressable_devices().unwrap()[0];
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![device]).unwrap();
        let session = session.with_device_mesh(mesh);
        let sharding = Sharding::replicated(logical_mesh, 0).with_reduced_axes(["x"]).unwrap();
        let input = Array::from_host_buffer(
            &session,
            ArrayType::scalar(DataType::F32),
            None,
            Some(sharding.clone()),
            values_to_bytes(&[2.0f32]),
        )
        .unwrap();
        assert_eq!(input.sharding(), &sharding);
        assert_eq!(input.r#type().sharding(), Some(&sharding.without_auto_axes()));
        let target = input.placement().unwrap();
        let eager = input.place(&target, None).unwrap();
        assert_eq!(eager.placement(), Some(target.clone()));
        assert_eq!(shard_values(&eager), vec![vec![2.0f32]]);

        // The shared fold validates the staged type before it can execute a known placement. That type omits the
        // physical reduction over the automatic axis, so the resolved target cannot restate that reduction here.
        let error = ProgramError::Type(TypeError::invalid(
            "`place` cannot change reduction state; use a reduction (or the corresponding collective) instead",
        ));
        let context = ryft_core::PartialEvaluationContext::new(CoreValue::domain(&input));
        let known = PartialTracer::new(context.clone(), context.known_value(input.clone()));
        assert_eq!(known.place(&target, None).unwrap_err(), error);

        let context = ryft_core::PartialEvaluationContext::new(session);
        let known = PartialTracer::new(context.clone(), context.known_value(Value::Array(input)));
        assert_eq!(known.place(&target, None).unwrap_err(), error);
    }

    #[test]
    fn test_place_operation_eager_partial_evaluation_admits_relocatable_known_values() {
        // A known partial value keeps the relocatability of the uncommitted upload that it carries, so a targetless
        // placement with a source on a named mesh of the second device admits it through `Place`, direct
        // interpretation, and batching, and leaves it where it is, uncommitted, as it does for the upload itself. An
        // unknown partial value with a declared sharding names a residual program variable, which is staged,
        // so it is validated strictly against that declaration.
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("other", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let source = Placement::new(mesh, Sharding::replicated(logical_mesh, 1)).unwrap();
        let known_array = |value: PartialTracer<ryft_core::ProjectedContext<Session, ArrayType>>| {
            value.value().unwrap().as_known().unwrap().clone()
        };
        let array = upload(&session, &[2], &[1., 2.]);
        let context = ryft_core::PartialEvaluationContext::new(CoreValue::domain(&array));
        let known = PartialTracer::new(context.clone(), context.known_value(array.clone()));
        let packed = PartialTracer::new(context.clone(), context.known_value(upload(&session, &[1, 2], &[1., 2.])));
        let operation = PlaceOperation::new(None, source.clone());
        let outcomes = vec![
            known
                .place(&Option::<Placement>::None, &source)
                .map(known_array)
                .map(observe_placement)
                .map_err(|error| error.to_string()),
            InterpretableOperation::interpret(&operation, &context, &EmptyRegionDriver, &[known.clone()])
                .map(|mut outputs| observe_placement(known_array(outputs.remove(0))))
                .map_err(|error| error.to_string()),
            batch(
                |item| item.place(&Option::<Placement>::None, &source),
                packed,
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .map(known_array)
            .map(observe_placement)
            .map_err(|error| error.to_string()),
        ];
        let accepted = Ok((vec![devices[0]], vec![1., 2.], false));
        assert_eq!(outcomes, vec![accepted.clone(), accepted.clone(), accepted]);

        let context = ryft_core::PartialEvaluationContext::new(CoreValue::domain(&array));
        let initial = LogicalMesh::new(vec![MeshAxis::new("initial", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let declaration = array.r#type().into_owned().with_sharding(Sharding::replicated(initial, 1)).unwrap();
        let unknown = PartialTracer::new(context.clone(), context.unknown_input(declaration.clone(), 0));
        let placed = unknown
            .place(&Option::<Placement>::None, &source)
            .map_err(|error| error.to_string())
            .and_then(|value| value.value().map(|_| ()).map_err(|error| error.to_string()));
        assert_eq!(
            placed,
            Err(format!(
                "`{PLACE_OPERATION_NAME}` source sharding `{}` does not match the input type `{}`",
                source.sharding(),
                declaration,
            )),
        );
    }

    #[test]
    fn test_place_operation_eager_partial_evaluation_folds_and_residualizes_relocatable_values() {
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        let second = session.addressable_devices().unwrap()[1];
        let named_mesh =
            |axis| LogicalMesh::new(vec![MeshAxis::new(axis, 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let initial = Sharding::replicated(named_mesh("first"), 1);
        let source_mesh = named_mesh("second");
        let source = Placement::new(
            DeviceMesh::new(source_mesh.clone(), vec![second]).unwrap(),
            Sharding::replicated(source_mesh, 1),
        )
        .unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let values = [1f32, 2.];
        let anonymous = upload(&session, &[2], &values);
        let named = Array::from_host_buffer(&session, r#type, None, Some(initial), values_to_bytes(&values)).unwrap();
        let copy = PlaceOperation::new(None, source).with_may_alias(false);
        assert!(!named.is_committed());
        assert!(matches!(
            copy.infer_output_types(&[named.r#type().into_owned()], &[]),
            Err(TypeError::Invalid { message })
                if message == "`place` source sharding `{mesh<['second'=1:explicit]>, [{}]}` does not match the input \
                               type `f32[2][sharding={mesh<['first'=1:explicit]>, [{}]}]`",
        ));
        let residualized = indoc! {"
            lambda %0:f32[2] .
            let %1:f32[2] = place [
                source={mesh<['second'=1:explicit]>, devices=[1], sharding={mesh<['second'=1:explicit]>, [{}]}},
                may_alias=false,
            ] %0
            in (%1)"
        };

        // The type rule rejects the source for the named sharding, but both uncommitted uploads are relocatable, so
        // partial evaluation admits it. A placement that may alias folds to the known input. A copy stays residual when
        // copy folding is disabled, and each execution of the residual program returns a new buffer.
        for input in [anonymous, named] {
            let domain = CoreValue::domain(&input);
            let context = ryft_core::PartialEvaluationContext::new(domain.clone());
            let known = PartialTracer::new(context.clone(), context.known_value(input.clone()));
            let [output] = InterpretableOperation::interpret(
                &copy.clone().with_may_alias(true),
                &context,
                &EmptyRegionDriver,
                &[known],
            )
            .unwrap()
            .try_into()
            .unwrap();
            assert_eq!(output.value().unwrap().declared_type(), None);
            assert_eq!(output.r#type(), input.r#type());
            assert_eq!(output.value().unwrap().as_known().unwrap().placement(), input.placement());

            let context = ryft_core::PartialEvaluationContext::new(domain.clone()).with_allow_copy_folding(false);
            let known = PartialTracer::new(context.clone(), context.known_value(input.clone()));
            let [output] = InterpretableOperation::interpret(&copy, &context, &EmptyRegionDriver, &[known])
                .unwrap()
                .try_into()
                .unwrap();
            let residual = output.value().unwrap().clone();
            assert!(residual.is_unknown());
            drop(output);
            let evaluation = context.into_evaluation(vec![residual]).unwrap();
            assert_eq!(evaluation.program().to_string(), residualized);

            let first_copy = evaluation.interpret(&domain, &[]).unwrap().remove(0);
            let second_copy = evaluation.interpret(&domain, &[]).unwrap().remove(0);
            assert_eq!(first_copy.mesh(), input.mesh());
            assert_eq!(second_copy.mesh(), input.mesh());
            assert_eq!(shard_values(&first_copy), vec![values.to_vec()]);
            assert_eq!(shard_values(&second_copy), vec![values.to_vec()]);
            let input_addresses = native_buffer_addresses(&input);
            let first_addresses = native_buffer_addresses(&first_copy);
            let second_addresses = native_buffer_addresses(&second_copy);
            assert_ne!(first_addresses, input_addresses);
            assert_ne!(second_addresses, input_addresses);
            assert_ne!(first_addresses, second_addresses);

            // The same rule runs through the projected array member of the full session family.
            let context = ryft_core::PartialEvaluationContext::new(session.clone()).with_allow_copy_folding(false);
            let known = PartialTracer::new(context.clone(), context.known_value(Value::Array(input.clone())));
            let [output] = context.bind(copy.clone(), Vec::new(), &[known]).unwrap().try_into().unwrap();
            let residual = output.value().unwrap().clone();
            assert!(residual.is_unknown());
            drop(output);
            let evaluation = context.into_evaluation(vec![residual]).unwrap();
            assert_eq!(evaluation.program().to_string(), residualized);
            let output = evaluation.interpret(&session, &[]).unwrap().remove(0);
            let output = ValueProjection::<ArrayType>::into_projected(output).unwrap();
            assert_eq!(output.mesh(), input.mesh());
            assert_eq!(shard_values(&output), vec![values.to_vec()]);
            assert_ne!(native_buffer_addresses(&output), input_addresses);
        }
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
    fn test_place_operation_eager_jvp_admits_relocatable_tangents() {
        let session = Session::builder().client(cpu_client(3)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let (primal, tangent) = named_primal_and_tangent(&session, devices[1]);
        let aligned = tangent.place(&primal.placement().unwrap(), None).unwrap();
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let laid_out = aligned
            .place(&PlacementRequest::InPlace { memory: None, layout: Some(layout.clone()) }, None)
            .unwrap();
        assert_eq!(laid_out.r#type().layout(), Some(&layout));

        let target = devices[2];
        let (_, pushforward) = differentiate_at(primal.clone()).linearize(|x| x.place(&target, None)).unwrap();
        for tangent in [tangent, aligned, laid_out] {
            let (value, derivative) =
                differentiate_at(primal.clone()).jvp(tangent.clone(), |x| x.place(&target, None)).unwrap();
            assert_eq!(observe_placement(value), (vec![target], vec![1., 2.], true));
            assert_eq!(observe_placement(derivative), (vec![target], vec![3., 4.], true));
            assert_eq!(observe_placement(pushforward.apply(tangent).unwrap()), (vec![target], vec![3., 4.], true));
        }
    }

    #[test]
    fn test_place_operation_eager_jvp_moves_committed_inputs_to_an_explicit_device() {
        let session = Session::builder().client(cpu_client(3)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let primal = upload(&session, &[2], &[1., 2.]).place(&devices[1], None).unwrap();
        let tangent = upload(&session, &[2], &[3., 4.]).place(&devices[0], None).unwrap();
        let target = devices[2];
        let (value, derivative) =
            differentiate_at(primal.clone()).jvp(tangent.clone(), |x| x.place(&target, None)).unwrap();
        let (_, pushforward) = differentiate_at(primal).linearize(|x| x.place(&target, None)).unwrap();
        assert_eq!(observe_placement(value), (vec![target], vec![1., 2.], true));
        assert_eq!(observe_placement(derivative), (vec![target], vec![3., 4.], true));
        assert_eq!(observe_placement(pushforward.apply(tangent).unwrap()), (vec![target], vec![3., 4.], true));
    }

    #[test]
    fn test_place_operation_eager_jvp_on_explicit_device_feeds_elementwise_arithmetic() {
        let session = Session::builder().client(cpu_client(3)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let (primal, tangent) = named_primal_and_tangent(&session, devices[1]);
        let target = devices[2];
        let (value, derivative) = differentiate_at(primal)
            .jvp(tangent, |x| {
                let placed = x.place(&target, None)?;
                placed.add(&placed)
            })
            .unwrap();
        assert_eq!(observe_placement(value), (vec![target], vec![2., 4.], true));
        assert_eq!(observe_placement(derivative), (vec![target], vec![6., 8.], true));
    }

    #[test]
    fn test_place_operation_eager_jvp_preserves_devices_for_memory_targets() {
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let primal = upload(&session, &[2], &[1., 2.]).place(&devices[1], None).unwrap();
        let tangent = upload(&session, &[2], &[3., 4.]);
        let committed = tangent.place(&devices[0], None).unwrap();
        let pinned = Memory::Host { pinned: true };
        let (_, pushforward) = differentiate_at(primal.clone()).linearize(|x| x.place(&pinned, None)).unwrap();

        for tangent in [tangent, committed] {
            let (value, derivative) =
                differentiate_at(primal.clone()).jvp(tangent.clone(), |x| x.place(&pinned, None)).unwrap();
            let saved = pushforward.apply(tangent).unwrap();
            assert_eq!(value.r#type().memory(), pinned);
            assert_eq!(observe_placement(value), (vec![devices[1]], vec![1., 2.], true));
            for result in [derivative, saved] {
                assert_eq!(result.r#type().memory(), pinned);
                assert_eq!(observe_placement(result), (vec![devices[0]], vec![3., 4.], true));
            }
        }
    }

    #[test]
    fn test_place_operation_eager_jvp_keeps_memory_placements_for_independent_arithmetic() {
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let primal = upload(&session, &[2], &[1., 2.]).place(&devices[1], None).unwrap();
        let tangent = upload(&session, &[2], &[3., 4.]);
        let (value, derivative) = differentiate_at(primal.clone())
            .jvp(tangent.clone(), |x| {
                let placed = x.place(&Memory::Device, None)?;
                placed.add(&placed)
            })
            .unwrap();
        assert_eq!(observe_placement(value), (vec![devices[1]], vec![2., 4.], true));
        assert_eq!(observe_placement(derivative), (vec![devices[0]], vec![6., 8.], true));
        let (_, pushforward) = differentiate_at(primal)
            .linearize(|x| {
                let placed = x.place(&Memory::Device, None)?;
                placed.add(&placed)
            })
            .unwrap();
        assert_eq!(observe_placement(pushforward.apply(tangent).unwrap()), (vec![devices[0]], vec![6., 8.], true));
    }

    #[test]
    fn test_place_operation_eager_jvp_rejects_incompatible_devices_after_memory_placement() {
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let (primal, tangent) = named_primal_and_tangent(&session, devices[1]);
        let (_, pushforward) = differentiate_at(primal).linearize(|x| x.place(&Memory::Device, None)?.sin()).unwrap();
        assert!(matches!(
            pushforward.apply(tangent),
            Err(ProgramError::InvalidArgument { message })
                if message == "received incompatible devices for eager XLA execution: input #1 is placed on \
                               devices [0] but input #0 is placed on devices [1]",
        ));
    }

    #[test]
    fn test_place_operation_eager_jvp_rejects_anonymous_tangent_outputs_for_named_memory_placement() {
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let (primal, tangent) = named_primal_and_tangent(&session, devices[1]);
        let pinned = Memory::Host { pinned: true };
        let (_, pushforward) = differentiate_at(primal.clone()).linearize(|x| x.place(&pinned, None)).unwrap();
        assert!(matches!(
            differentiate_at(primal).jvp(tangent.clone(), |x| x.place(&pinned, None)),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "tangent type f32[2]@Host[Pinned] does not match type \
                               f32[2][sharding={mesh<['other'=1:explicit]>, [{}]}]@Host[Pinned] required by \
                               primal type f32[2][sharding={mesh<['other'=1:explicit]>, [{}]}]@Host[Pinned]",
        ));
        assert!(matches!(
            pushforward.apply(tangent),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "encountered output type f32[2]@Host[Pinned] which is incompatible with the region's \
                               declared type f32[2][sharding={mesh<['other'=1:explicit]>, [{}]}]@Host[Pinned]",
        ));
    }

    #[test]
    fn test_place_operation_eager_jvp_device_target_uses_default_memory() {
        let session = Session::builder().client(cpu_client(3)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let primal = upload(&session, &[2], &[1., 2.]).place(&devices[1], None).unwrap();
        let tangent = upload(&session, &[2], &[3., 4.]);
        let pinned = Memory::Host { pinned: true };
        let primal = primal.place(&pinned, None).unwrap();
        let tangent = tangent.place(&pinned, None).unwrap();
        let target = devices[2];
        let (value, derivative) =
            differentiate_at(primal.clone()).jvp(tangent.clone(), |x| x.place(&target, None)).unwrap();
        let (_, pushforward) = differentiate_at(primal).linearize(|x| x.place(&target, None)).unwrap();
        let saved = pushforward.apply(tangent).unwrap();
        for output in [value, derivative, saved] {
            assert_eq!(output.r#type().memory(), Memory::Device);
            assert_eq!(output.mesh().devices(), &[target]);
            assert!(output.is_committed());
        }
    }

    #[test]
    fn test_place_operation_eager_linearize_keeps_saved_return_placement() {
        // A saved linearization of a placement from the first device onto the third records the first device as its
        // return placement. Rebinding that linear map at a tangent on the second device (i.e., pulling it back or
        // linearizing it again) keeps that return placement, so every cotangent goes back to the first device, as
        // pinned JAX returns it there in all three cases.
        let session = Session::builder().client(cpu_client(3)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let uploaded = upload(&session, &[2], &[1., 2.]);
        let x = uploaded.to_placement(&session, devices[0]).unwrap();
        let t = uploaded.to_placement(&session, devices[1]).unwrap();
        let (primal, linear) = differentiate_at(x).linearize(|x| x.place(&devices[2], None)).unwrap();
        let returned = (vec![devices[0]], vec![1., 2.], true);
        let transposed = linear.program().transpose().unwrap();
        let direct = transposed.interpret_in_context(&uploaded.domain(), vec![primal.clone()]).unwrap();
        assert_eq!(direct.into_iter().map(observe_placement).collect::<Vec<_>>(), vec![returned.clone()]);

        let (_, pullback) = differentiate_at(t.clone()).vjp(|z| linear.apply_in_context(&z.domain(), z)).unwrap();
        assert_eq!(observe_placement(pullback.apply_materialized(primal).unwrap()), returned);

        let (result, second) = differentiate_at(t).linearize(|z| linear.apply_in_context(&z.domain(), z)).unwrap();
        assert_eq!(observe_placement(result), (vec![devices[2]], vec![1., 2.], true));
        assert_eq!(
            second.program().to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = place [
                    inferred_source={mesh<['device'=1:auto]>, devices=[0], \
                        sharding={mesh<['device'=1:auto]>, [{}]}, memory=Device},
                    target=device(2),
                ] %0
                in (%1)"
            },
        );
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
    fn test_place_operation_eager_vjp_accepts_relocatable_seeds() {
        // The VJP of placing an uncommitted upload on the first device onto a named mesh of the second device accepts
        // an uncommitted seed on the first device, which may be placed as the cotangent type states, as well as a seed
        // on that mesh, through an `Array` and through its `Value::Array` representation, and returns the cotangent
        // to the first device, as JAX does. A seed committed to the first device has the same anonymous abstract
        // sharding. The pullback accepts it and replays the transposed placement on its actual devices (refer to
        // `test_place_operation_eager_vjp_uses_seed_on_its_own_devices`).
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("q", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let target = Placement::new(mesh.clone(), Sharding::replicated(logical_mesh.clone(), 1)).unwrap();
        let seeds = [
            upload(&session, &[2], &[5., 6.]),
            Array::from_host_buffer(
                &session,
                ArrayType::new_static(DataType::F32, [2]),
                mesh,
                Some(Sharding::replicated(logical_mesh, 1)),
                values_to_bytes(&[5f32, 6.]),
            )
            .unwrap(),
            upload(&session, &[2], &[5., 6.]).to_placement(&session, devices[0]).unwrap(),
        ];
        let mut outcomes = Vec::new();
        for seed in seeds {
            let through_array = differentiate_at(upload(&session, &[2], &[1., 2.]))
                .vjp(|x| x.place(&target, None))
                .map_err(|error| error.to_string())
                .and_then(|(_, pullback)| pullback.apply_materialized(seed.clone()).map_err(|error| error.to_string()))
                .map(observe_placement);
            let through_value = session
                .differentiate_at(Value::Array(upload(&session, &[2], &[1., 2.])))
                .vjp(|x| x.place(&target, None))
                .map_err(|error| error.to_string())
                .and_then(|(_, pullback)| {
                    let cotangent =
                        pullback.apply_materialized(Value::Array(seed)).map_err(|error| error.to_string())?;
                    ValueProjection::<ArrayType>::into_projected(cotangent).map_err(|error| error.to_string())
                })
                .map(observe_placement);
            outcomes.push(vec![through_array, through_value]);
        }
        // The transposed placement has a target (where the primal lived), so its output is committed to it.
        let accepted = Ok((vec![devices[0]], vec![5., 6.], true));
        assert_eq!(outcomes, vec![vec![accepted.clone(), accepted.clone()]; 3]);
    }

    #[test]
    fn test_place_operation_eager_vjp_uses_seed_on_its_own_devices() {
        let session = Session::builder().client(cpu_client(4)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let split = |name: &str, mesh_devices: &[Device]| {
            let axis = MeshAxis::new(name, mesh_devices.len(), MeshAxisType::Explicit).unwrap();
            let logical_mesh = LogicalMesh::new(vec![axis]).unwrap();
            let sharding = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded([name])]).unwrap();
            Placement::new(DeviceMesh::new(logical_mesh, mesh_devices.to_vec()).unwrap(), sharding).unwrap()
        };
        let target = split("q", &devices[..2]);
        let upload_to = |shape: &[usize], values: &[f32], placement: PlacementRequest| {
            upload(&session, shape, values).to_placement(&session, placement).unwrap()
        };
        let x = upload_to(&[4], &[1., 2., 3., 4.], devices[0].into());
        let [bare_seed, matching_seed, mismatched_seed] = [
            upload_to(&[4], &[5., 6., 7., 8.], devices[2].into()),
            upload_to(&[4], &[5., 6., 7., 8.], split("q", &devices[2..]).into()),
            upload_to(&[4], &[5., 6., 7., 8.], split("r", &devices[2..]).into()),
        ];
        let source = devices[0];

        // Exercise both the array operation family and its wrapper in the session operation family.
        macro_rules! check_pullbacks {
            ($differentiate:expr, $lift:expr, $into_array:expr) => {{
                let differentiate = $differentiate;
                let lift = $lift;
                let into_array = $into_array;
                let (output, pullback) = differentiate().vjp(|x| x.place(&target, &source)).unwrap();
                let output = into_array(output);
                assert_eq!(output.mesh().devices(), &devices[..2]);
                assert_eq!(output.sharding(), target.sharding());

                // An anonymous seed and the same logical sharding on other devices both pass the boundary.
                for seed in [&bare_seed, &matching_seed] {
                    let gradient =
                        pullback.apply_materialized(lift(seed.clone())).map(into_array).map(observe_placement);
                    assert_eq!(gradient, Ok((vec![source], vec![5., 6., 7., 8.], true)));
                }
                let gradient =
                    pullback.apply_materialized(lift(mismatched_seed.clone())).map(into_array).map(observe_placement);
                assert_eq!(
                    gradient,
                    Err(ProgramError::MalformedProgram(format!(
                        "pullback cotangent 0 has type {} but its primal boundary requires cotangent type {}",
                        mismatched_seed.r#type(),
                        output.r#type().cotangent().unwrap(),
                    ))),
                );

                // Negation runs where the anonymous seed lives and the placement sends the output to the source.
                let (_, pullback) = differentiate().vjp(|x| x.place(&target, &source)?.neg()).unwrap();
                let gradient =
                    pullback.apply_materialized(lift(bare_seed.clone())).map(into_array).map(observe_placement);
                assert_eq!(gradient, Ok((vec![source], vec![-5., -6., -7., -8.], true)));

                // Sine's transpose multiplies the seed by the cosine evaluated on the forward output devices.
                let (_, pullback) = differentiate().vjp(|x| x.place(&target, &source)?.sin()).unwrap();
                let gradient =
                    pullback.apply_materialized(lift(bare_seed.clone())).map(into_array).map(observe_placement);
                assert_eq!(
                    gradient,
                    Err(ProgramError::InvalidArgument {
                        message: "received incompatible devices for eager XLA execution: input #1 is placed on \
                                  devices [2] but input #0 is placed on devices [0, 1]"
                            .to_string(),
                    }),
                );
            }};
        }
        check_pullbacks!(|| differentiate_at(x.clone()), std::convert::identity, std::convert::identity);
        check_pullbacks!(|| session.differentiate_at(Value::Array(x.clone())), Value::Array, |value| {
            ValueProjection::<ArrayType>::into_projected(value).unwrap()
        });

        // Under `vmap`, the seed is the same for every batch item, and the gradient of every item returns to the first
        // device.
        let packed = upload_to(&[2, 4], &[1., 2., 3., 4., 5., 6., 7., 8.], devices[0].into());
        let batched: Result<Array, _> = batch(
            |(x, seed)| {
                let (_, pullback) = differentiate_at(x).vjp(|x| x.place(&target, None))?;
                pullback.apply_materialized(seed)
            },
            (packed, bare_seed.clone()),
            (BatchAxis::new(0), BatchAxis::replicated()),
            BatchAxis::new(0),
            None,
        );
        assert_eq!(
            batched.map(observe_placement).map_err(|error| error.to_string()),
            Ok((vec![devices[0]], vec![5., 6., 7., 8., 5., 6., 7., 8.], true)),
        );

        // Transposing the pullback again places the cotangent of the inner seed on the forward target,
        // including when that seed lived on a single device outside the mesh.
        let cotangents = [&matching_seed, &bare_seed].map(|inner_seed| -> Result<_, String> {
            let (_, pullback) = differentiate_at((x.clone(), inner_seed.clone()))
                .vjp(|(x, cotangent)| {
                    let (_, pullback) = differentiate_at(x).vjp(|x| x.place(&target, None))?;
                    pullback.apply_materialized(cotangent)
                })
                .map_err(|error| error.to_string())?;
            let seed = upload_to(&[4], &[9., 10., 11., 12.], devices[3].into());
            let cotangents = pullback.apply_materialized(seed).map_err(|error| error.to_string())?;
            assert_eq!(cotangents.1.sharding(), target.sharding());
            assert_eq!(shard_values(&cotangents.1), vec![vec![9., 10.], vec![11., 12.]]);
            Ok(observe_placement(cotangents.1))
        });
        assert_eq!(
            cotangents,
            [
                Ok((devices[..2].to_vec(), vec![9., 10., 11., 12.], true)),
                Ok((devices[..2].to_vec(), vec![9., 10., 11., 12.], true)),
            ],
        );
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

        // Each batch item is placed on the target device, so the packed output lives there.
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

        // Batching passes the device target through unchanged, so the packed output moves onto the device.
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
        let session = Session::builder().client(cpu_client(4)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
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
        let session = Session::builder().client(cpu_client(4)).build().unwrap();
        let devices = session.addressable_devices().unwrap();

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
        let session = Session::builder().client(cpu_client(4)).build().unwrap();
        let devices = session.addressable_devices().unwrap();

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

    #[test]
    fn test_place_operation_eager_batch_admits_relocatable_sources_under_nested_transforms() {
        // A targetless placement of an uncommitted upload on the first device with a source on a named mesh of the
        // second device is admitted under two nested batching levels, batching around a JVP, and a JVP around
        // batching, as it is eagerly (refer to `test_place_operation_eager_admits_relocatable_sources`),
        // because every transform value projects the relocatable type of the value that it wraps. Each leaves the
        // upload where it is, uncommitted, as pinned JAX does.
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("other", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let source = Placement::new(mesh, Sharding::replicated(logical_mesh, 1)).unwrap();
        let first = devices[0];

        let nested = batch(
            |outer| {
                Ok(batch(
                    |item| item.place(&Option::<Placement>::None, &source),
                    outer,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )?)
            },
            upload(&session, &[2, 1, 2], &[3., 4., 5., 6.]),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        );
        assert_eq!(
            nested.map(observe_placement).map_err(|error| error.to_string()),
            Ok((vec![first], vec![3., 4., 5., 6.], false)),
        );
        let batched_jvp = batch(
            |outer| {
                let (_, tangent) = differentiate_at(outer.clone())
                    .jvp(outer, |item| item.place(&Option::<Placement>::None, &source))?;
                Ok(tangent)
            },
            upload(&session, &[2, 2], &[3., 4., 5., 6.]),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        );
        assert_eq!(
            batched_jvp.map(observe_placement).map_err(|error| error.to_string()),
            Ok((vec![first], vec![3., 4., 5., 6.], false)),
        );
        let jvp_of_batch = differentiate_at(upload(&session, &[2, 2], &[1., 2., 3., 4.]))
            .jvp(upload(&session, &[2, 2], &[5., 6., 7., 8.]), |x| {
                Ok(batch(
                    |item| item.place(&Option::<Placement>::None, &source),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )?)
            })
            .map(|(primal, tangent)| (observe_placement(primal), observe_placement(tangent)))
            .map_err(|error| error.to_string());
        assert_eq!(
            jvp_of_batch,
            Ok(((vec![first], vec![1., 2., 3., 4.], false), (vec![first], vec![5., 6., 7., 8.], false))),
        );

        // The same three compositions through the `Value::Array` representation of the uploads, whose batching levels
        // (refer to `ArrayIrBatchingPolicy`) and array views (i.e., `ProjectedValue`s) project the relocatable type of
        // an array member, give the same results.
        let observe_value =
            |value: Value| observe_placement(ValueProjection::<ArrayType>::into_projected(value).unwrap());
        let nested = batch(
            |outer| {
                Ok(batch(
                    |item| item.place(&Option::<Placement>::None, &source),
                    outer,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )?)
            },
            Value::Array(upload(&session, &[2, 1, 2], &[3., 4., 5., 6.])),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        );
        let batched_jvp = batch(
            |outer| {
                let (_, tangent) = differentiate_at(outer.clone())
                    .jvp(outer, |item| item.place(&Option::<Placement>::None, &source))?;
                Ok(tangent)
            },
            Value::Array(upload(&session, &[2, 2], &[3., 4., 5., 6.])),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        );
        let jvp_of_batch = differentiate_at(Value::Array(upload(&session, &[2, 2], &[1., 2., 3., 4.]))).jvp(
            Value::Array(upload(&session, &[2, 2], &[5., 6., 7., 8.])),
            |x| {
                Ok(batch(
                    |item| item.place(&Option::<Placement>::None, &source),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )?)
            },
        );
        let outcomes = vec![
            nested.map(|value| vec![observe_value(value)]).map_err(|error| error.to_string()),
            batched_jvp.map(|value| vec![observe_value(value)]).map_err(|error| error.to_string()),
            jvp_of_batch
                .map(|(primal, tangent)| vec![observe_value(primal), observe_value(tangent)])
                .map_err(|error| error.to_string()),
        ];
        assert_eq!(
            outcomes,
            vec![
                Ok(vec![(vec![first], vec![3., 4., 5., 6.], false)]),
                Ok(vec![(vec![first], vec![3., 4., 5., 6.], false)]),
                Ok(vec![(vec![first], vec![1., 2., 3., 4.], false), (vec![first], vec![5., 6., 7., 8.], false)]),
            ],
        );
    }
}
