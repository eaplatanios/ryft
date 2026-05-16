use std::collections::HashMap;

use indoc::indoc;
use pretty_assertions::assert_eq;
use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
use ryft_pjrt::{BufferType, ClientOptions, CpuClientOptions, Program, load_cpu_plugin};

use ryft_core::sharding::{Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
use ryft_core::types::data_types::DataType;
use ryft_core::types::{ArrayType, Shape, Size, StaticShape};

use crate::tests::{logical_mesh_2x2, values_from_bytes, values_to_bytes};
use crate::{Array, CompilationContext, Error, FromPjrt, ToMlir};

use super::*;

fn test_spmd_compilation_options(partition_count: usize) -> CompilationOptions {
    CompilationOptions {
        argument_layouts: Vec::new(),
        parameter_is_tupled_arguments: false,
        executable_build_options: Some(ExecutableCompilationOptions {
            device_ordinal: -1,
            replica_count: 1,
            partition_count: partition_count as i64,
            use_spmd_partitioning: true,
            use_shardy_partitioner: true,
            ..Default::default()
        }),
        compile_portable_executable: false,
        profile_version: 0,
        serialized_multi_slice_configuration: Vec::new(),
        environment_option_overrides: HashMap::new(),
        target_config: None,
        allow_in_place_mlir_modification: false,
        matrix_unit_operand_precision: Precision::Default as i32,
    }
}

fn test_shape(dimensions: &[usize]) -> StaticShape {
    StaticShape::new(dimensions.to_vec())
}

#[test]
fn test_array_new_requires_sharding_without_single_buffer() {
    let mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
        vec![Device::new(0, 1)],
    )
    .unwrap();
    let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, None).unwrap();

    assert!(matches!(Array::from_addressable_buffers(array_type, mesh, Vec::new()), Err(Error::MissingSharding),));
}

#[test]
fn test_array_new_accepts_unsharded_type_with_single_buffer() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
    let devices = client
        .addressable_devices()
        .unwrap()
        .into_iter()
        .map(|device| Device::from_pjrt(device).unwrap())
        .collect::<Vec<_>>();
    let mesh =
        DeviceMesh::new(LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(), devices)
            .unwrap();
    let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)]), None, None).unwrap();
    let device = client.addressable_devices().unwrap().into_iter().next().unwrap();
    let buffer = client
        .buffer(values_to_bytes::<f32>(&[1.0, 2.0]).as_slice(), BufferType::F32, [2u64], None, device, None)
        .unwrap();

    let array = Array::from_addressable_buffers(array_type, mesh, vec![buffer]).unwrap();

    assert_eq!(array.shape(), StaticShape::new(vec![2]));
    assert_eq!(array.data_type(), DataType::F32);
    assert_eq!(array.sharding().mesh().axes()[0].size(), 2);
    assert_eq!(array.sharding().dimensions(), [ShardingDimension::replicated()].as_slice());
    assert_eq!(array.shards().len(), 2);
    assert_eq!(array.addressable_shards().count(), 1);
    assert!(array.shards().iter().all(|shard| shard.shape() == test_shape(&[2])));
}

#[test]
fn test_array_new_rejects_dynamic_shape() {
    let mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
        vec![Device::new(0, 1)],
    )
    .unwrap();
    let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
    let array_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Dynamic(Some(10))]), None, Some(sharding)).unwrap();

    assert!(matches!(
        Array::from_addressable_buffers(array_type, mesh, Vec::new()),
        Err(Error::DynamicShape { shape }) if shape == Shape::new(vec![Size::Dynamic(Some(10))]),
    ));
}

#[test]
fn test_device_put_visualizes_uneven_1d_partitioning() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
    let client_devices = client.addressable_devices().unwrap();
    let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
    let mesh =
        DeviceMesh::new(LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(), devices)
            .unwrap();
    let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    let values = [0.0f32, 1.0, 2.0, 3.0, 4.0];
    let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(5)]), None, Some(sharding)).unwrap();

    let array =
        Array::from_host_buffer(&client, r#type, mesh.clone(), values_to_bytes::<f32>(values.as_slice()).as_slice())
            .unwrap();

    assert_eq!(array.addressable_shards().count(), 2);
    assert!(array.shards().iter().all(|shard| shard.buffer().is_some()));
    assert_eq!(
        array.sharding().visualize().unwrap().render(false),
        indoc! {"
                ┌─────┬─────┐
                │  0  │  1  │
                └─────┴─────┘
            "}
        .trim_end()
        .to_string()
    );
}

#[test]
fn test_device_put_visualizes_2d_partitioning() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) })).unwrap();
    let client_devices = client.addressable_devices().unwrap();
    let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
    let mesh = DeviceMesh::new(logical_mesh_2x2(), devices).unwrap();
    let sharding = Sharding::new(
        mesh.logical_mesh().clone(),
        vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
    )
    .unwrap();
    let values = (0..48).map(|value| value as f32).collect::<Vec<_>>();
    let r#type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8), Size::Static(6)]), None, Some(sharding))
            .unwrap();

    let array =
        Array::from_host_buffer(&client, r#type, mesh.clone(), values_to_bytes::<f32>(values.as_slice()).as_slice())
            .unwrap();

    assert_eq!(array.addressable_shards().count(), 4);
    assert!(array.shards().iter().all(|shard| shard.buffer().is_some()));
    assert_eq!(
        array.sharding().visualize().unwrap().render(false),
        indoc! {"
                ┌─────┬─────┐
                │     │     │
                │  0  │  1  │
                │     │     │
                ├─────┼─────┤
                │     │     │
                │  2  │  3  │
                │     │     │
                └─────┴─────┘
            "}
        .trim_end()
        .to_string()
    );
    let first_shard_bytes = array
        .device_shard(client_devices[0].id().unwrap())
        .unwrap()
        .buffer()
        .unwrap()
        .copy_to_host(None)
        .unwrap()
        .r#await()
        .unwrap();
    assert_eq!(first_shard_bytes.as_slice().len() % size_of::<f32>(), 0);
    assert_eq!(
        values_from_bytes::<f32>(first_shard_bytes.as_slice()),
        vec![0.0, 1.0, 2.0, 6.0, 7.0, 8.0, 12.0, 13.0, 14.0, 18.0, 19.0, 20.0]
    );
    let second_shard_bytes = array
        .device_shard(client_devices[1].id().unwrap())
        .unwrap()
        .buffer()
        .unwrap()
        .copy_to_host(None)
        .unwrap()
        .r#await()
        .unwrap();
    assert_eq!(second_shard_bytes.as_slice().len() % size_of::<f32>(), 0);
    assert_eq!(
        values_from_bytes::<f32>(second_shard_bytes.as_slice()),
        vec![3.0, 4.0, 5.0, 9.0, 10.0, 11.0, 15.0, 16.0, 17.0, 21.0, 22.0, 23.0]
    );
    let third_shard_bytes = array
        .device_shard(client_devices[2].id().unwrap())
        .unwrap()
        .buffer()
        .unwrap()
        .copy_to_host(None)
        .unwrap()
        .r#await()
        .unwrap();
    assert_eq!(third_shard_bytes.as_slice().len() % size_of::<f32>(), 0);
    assert_eq!(
        values_from_bytes::<f32>(third_shard_bytes.as_slice()),
        vec![24.0, 25.0, 26.0, 30.0, 31.0, 32.0, 36.0, 37.0, 38.0, 42.0, 43.0, 44.0]
    );
    let fourth_shard_bytes = array
        .device_shard(client_devices[3].id().unwrap())
        .unwrap()
        .buffer()
        .unwrap()
        .copy_to_host(None)
        .unwrap()
        .r#await()
        .unwrap();
    assert_eq!(fourth_shard_bytes.as_slice().len() % size_of::<f32>(), 0);
    assert_eq!(
        values_from_bytes::<f32>(fourth_shard_bytes.as_slice()),
        vec![27.0, 28.0, 29.0, 33.0, 34.0, 35.0, 39.0, 40.0, 41.0, 45.0, 46.0, 47.0]
    );
}

#[test]
fn test_array_put_reshards_fully_addressable_array() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
    let client_devices = client.addressable_devices().unwrap();
    let source_mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
        vec![Device::new(client_devices[0].id().unwrap(), client_devices[0].process_index().unwrap())],
    )
    .unwrap();
    let source_sharding = Sharding::replicated(source_mesh.logical_mesh().clone(), 1);
    // Use an evenly-divisible (4 elements / 2 devices) split. Uneven splits (e.g. 5/2) are not
    // currently supported by XLA's CPU SPMD partitioner — the per-device output buffer would
    // need different shapes ([3] + [2]), but the partitioner returns the full unsliced buffer
    // and ryft's descriptor model expects exact logical per-device shapes. See M9 in the plan
    // for the limitation; revisit if a future XLA release or backend handles this.
    let source_values = [0.0f32, 1.0, 2.0, 3.0];
    let source_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]), None, Some(source_sharding)).unwrap();
    let source_array = Array::from_host_buffer(
        &client,
        source_type,
        source_mesh,
        values_to_bytes::<f32>(source_values.as_slice()).as_slice(),
    )
    .unwrap();

    let target_devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
    let target_mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
        target_devices,
    )
    .unwrap();
    let target_sharding =
        Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

    let context = CompilationContext::new(&client);
    let moved_array = source_array
        .to(
            &context,
            crate::arrays_v0::DevicePutTarget::Placement { mesh: target_mesh.clone(), sharding: target_sharding },
            false,
        )
        .unwrap();

    assert_eq!(moved_array.addressable_shards().count(), 2);
    assert_eq!(
        moved_array.sharding().visualize().unwrap().render(false),
        indoc! {"
                ┌─────┬─────┐
                │  0  │  1  │
                └─────┴─────┘
            "}
        .trim_end()
        .to_string()
    );
    let first_shard_bytes = moved_array
        .device_shard(client_devices[0].id().unwrap())
        .unwrap()
        .buffer()
        .unwrap()
        .copy_to_host(None)
        .unwrap()
        .r#await()
        .unwrap();
    let second_shard_bytes = moved_array
        .device_shard(client_devices[1].id().unwrap())
        .unwrap()
        .buffer()
        .unwrap()
        .copy_to_host(None)
        .unwrap()
        .r#await()
        .unwrap();
    assert_eq!(first_shard_bytes.as_slice().len() % size_of::<f32>(), 0);
    assert_eq!(values_from_bytes::<f32>(first_shard_bytes.as_slice()), vec![0.0, 1.0]);
    assert_eq!(second_shard_bytes.as_slice().len() % size_of::<f32>(), 0);
    assert_eq!(values_from_bytes::<f32>(second_shard_bytes.as_slice()), vec![2.0, 3.0]);
}

#[test]
fn test_array_put_copies_matching_local_shards_without_full_source_addressability() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
    let local_device = client.addressable_devices().unwrap().remove(0);
    let local_device_id = local_device.id().unwrap();
    let remote_device_id = local_device_id + 1;
    let mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
        vec![Device::new(local_device_id, local_device.process_index().unwrap()), Device::new(remote_device_id, 1)],
    )
    .unwrap();
    let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    let local_source_buffer = client
        .buffer(values_to_bytes::<f32>(&[0.0, 1.0]).as_slice(), BufferType::F32, [2u64], None, local_device, None)
        .unwrap();
    let source_array_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]), None, Some(sharding.clone())).unwrap();
    let source_array =
        Array::from_addressable_buffers(source_array_type, mesh.clone(), vec![local_source_buffer]).unwrap();

    let context = CompilationContext::new(&client);
    let copied_array = source_array
        .to(&context, crate::arrays_v0::DevicePutTarget::Placement { mesh: mesh.clone(), sharding }, false)
        .unwrap();
    let expected_visualization =
        format!("┌─────┬─────┐\n│{:^5}│{:^5}│\n└─────┴─────┘", local_device_id, remote_device_id);

    assert_eq!(copied_array.addressable_shards().count(), 1);
    assert_eq!(copied_array.sharding().visualize().unwrap().render(false), expected_visualization);
    let copied_shard_bytes = copied_array
        .device_shard(local_device_id)
        .unwrap()
        .buffer()
        .unwrap()
        .copy_to_host(None)
        .unwrap()
        .r#await()
        .unwrap();
    assert_eq!(copied_shard_bytes.as_slice().len() % size_of::<f32>(), 0);
    assert_eq!(values_from_bytes::<f32>(copied_shard_bytes.as_slice()), vec![0.0, 1.0]);
    assert!(copied_array.device_shard(remote_device_id).unwrap().buffer().is_none());
}

#[test]
fn test_plan_exact_shard_put_uses_cross_host_send_and_receive_for_remote_exact_moves() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
    let local_device = client.addressable_devices().unwrap().remove(0);
    let local_device_id = local_device.id().unwrap();
    let remote_device_id = local_device_id + 1;
    let source_mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
        vec![Device::new(local_device_id, local_device.process_index().unwrap()), Device::new(remote_device_id, 1)],
    )
    .unwrap();
    let source_sharding =
        Sharding::new(source_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    let local_source_buffer = client
        .buffer(values_to_bytes::<f32>(&[0.0, 1.0]).as_slice(), BufferType::F32, [2u64], None, local_device, None)
        .unwrap();
    let source_array_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]), None, Some(source_sharding)).unwrap();
    let source_array =
        Array::from_addressable_buffers(source_array_type, source_mesh, vec![local_source_buffer]).unwrap();
    let target_mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
        vec![Device::new(remote_device_id, 1), Device::new(local_device_id, client.process_index().unwrap())],
    )
    .unwrap();
    let target_sharding =
        Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

    let plan = plan_exact_shard_put(
        &source_array,
        client.process_index().unwrap(),
        &source_array.shape(),
        &target_mesh,
        &target_sharding,
    )
    .unwrap();

    assert_eq!(
        plan,
        Some(ExactShardPutPlan::new(
            Vec::new(),
            vec![CrossHostShardSendPlan::new(0, local_device_id, 0, remote_device_id, 0)],
            vec![CrossHostShardReceivePlan::new(1, remote_device_id, 1, local_device_id, test_shape(&[2]), 3,)],
        ))
    );
}

#[test]
fn test_array_put_rejects_non_addressable_source_shards() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
    let source_mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
        vec![Device::new(0, 0), Device::new(1, 1)],
    )
    .unwrap();
    let source_sharding =
        Sharding::new(source_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    let source_array_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]), None, Some(source_sharding)).unwrap();
    let source_array = Array::from_addressable_buffers(source_array_type, source_mesh, Vec::new()).unwrap();
    let target_mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("y", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
        vec![Device::new(0, 0)],
    )
    .unwrap();
    let target_sharding = Sharding::replicated(target_mesh.logical_mesh().clone(), 1);

    let context = CompilationContext::new(&client);
    assert!(matches!(
        source_array.to(
            &context,
            crate::arrays_v0::DevicePutTarget::Placement { mesh: target_mesh, sharding: target_sharding },
            false
        ),
        Err(ArrayError::MissingAddressableShardForMove { shard_index: 0, device_id: 0 }),
    ));
}

#[test]
fn test_device_put_broadcasts_root_placement_over_array_tuple() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
    let client_devices = client.addressable_devices().unwrap();
    let source_mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
        vec![Device::new(client_devices[0].id().unwrap(), client_devices[0].process_index().unwrap())],
    )
    .unwrap();
    let source_sharding = Sharding::replicated(source_mesh.logical_mesh().clone(), 1);
    let first_source_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]), None, Some(source_sharding.clone())).unwrap();
    let first_source_array = Array::from_host_buffer(
        &client,
        first_source_type,
        source_mesh.clone(),
        values_to_bytes::<f32>(&[0.0, 1.0, 2.0, 3.0]).as_slice(),
    )
    .unwrap();
    let second_source_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]), None, Some(source_sharding)).unwrap();
    let second_source_array = Array::from_host_buffer(
        &client,
        second_source_type,
        source_mesh,
        values_to_bytes::<f32>(&[10.0, 11.0, 12.0, 13.0]).as_slice(),
    )
    .unwrap();

    let target_mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
        client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
    )
    .unwrap();
    let target_sharding =
        Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

    let context = CompilationContext::new(&client);
    let moved_arrays = device_put(
        &context,
        (first_source_array, second_source_array),
        DevicePutOptions::new(
            Some(DevicePutTarget::Placement { mesh: target_mesh.clone(), sharding: target_sharding.clone() }),
            Option::<DevicePutTarget>::None,
            Option::<bool>::None,
            Option::<Option<bool>>::None,
        ),
    )
    .unwrap();

    assert_eq!(
        moved_arrays.0.sharding().visualize().unwrap().render(false),
        indoc! {"
                ┌─────┬─────┐
                │  0  │  1  │
                └─────┴─────┘
            "}
        .trim_end()
        .to_string()
    );
    assert_eq!(
        moved_arrays.1.sharding().visualize().unwrap().render(false),
        indoc! {"
                ┌─────┬─────┐
                │  0  │  1  │
                └─────┴─────┘
            "}
        .trim_end()
        .to_string()
    );
    let first_moved_shard_bytes = moved_arrays
        .0
        .device_shard(client_devices[1].id().unwrap())
        .unwrap()
        .buffer()
        .unwrap()
        .copy_to_host(None)
        .unwrap()
        .r#await()
        .unwrap();
    assert_eq!(first_moved_shard_bytes.as_slice().len() % size_of::<f32>(), 0);
    assert_eq!(values_from_bytes::<f32>(first_moved_shard_bytes.as_slice()), vec![2.0, 3.0]);
    let second_moved_shard_bytes = moved_arrays
        .1
        .device_shard(client_devices[0].id().unwrap())
        .unwrap()
        .buffer()
        .unwrap()
        .copy_to_host(None)
        .unwrap()
        .r#await()
        .unwrap();
    assert_eq!(second_moved_shard_bytes.as_slice().len() % size_of::<f32>(), 0);
    assert_eq!(values_from_bytes::<f32>(second_moved_shard_bytes.as_slice()), vec![10.0, 11.0]);
}

#[test]
fn test_device_put_preserves_partially_addressable_array_when_device_is_absent() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
    let local_device = client.addressable_devices().unwrap().remove(0);
    let local_device_id = local_device.id().unwrap();
    let remote_device_id = local_device_id + 1;
    let mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
        vec![Device::new(local_device_id, local_device.process_index().unwrap()), Device::new(remote_device_id, 1)],
    )
    .unwrap();
    let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    let local_source_buffer = client
        .buffer(values_to_bytes::<f32>(&[0.0, 1.0]).as_slice(), BufferType::F32, [2u64], None, local_device, None)
        .unwrap();
    let source_array_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]), None, Some(sharding.clone())).unwrap();
    let source_array =
        Array::from_addressable_buffers(source_array_type, mesh.clone(), vec![local_source_buffer]).unwrap();

    let context = CompilationContext::new(&client);
    let copied_array = device_put(&context, source_array, DevicePutOptions::defaults()).unwrap();
    let expected_visualization =
        format!("┌─────┬─────┐\n│{:^5}│{:^5}│\n└─────┴─────┘", local_device_id, remote_device_id);

    assert_eq!(copied_array.addressable_shards().count(), 1);
    assert_eq!(copied_array.sharding().visualize().unwrap().render(false), expected_visualization);
    let copied_shard_bytes = copied_array
        .device_shard(local_device_id)
        .unwrap()
        .buffer()
        .unwrap()
        .copy_to_host(None)
        .unwrap()
        .r#await()
        .unwrap();
    assert_eq!(copied_shard_bytes.as_slice().len() % size_of::<f32>(), 0);
    assert_eq!(values_from_bytes::<f32>(copied_shard_bytes.as_slice()), vec![0.0, 1.0]);
    assert!(copied_array.device_shard(remote_device_id).unwrap().buffer().is_none());
}

#[test]
fn test_array_to_device_preserves_same_partially_addressable_placement() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
    let local_device = client.addressable_devices().unwrap().remove(0);
    let local_device_id = local_device.id().unwrap();
    let remote_device_id = local_device_id + 1;
    let mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
        vec![Device::new(local_device_id, local_device.process_index().unwrap()), Device::new(remote_device_id, 1)],
    )
    .unwrap();
    let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    let local_source_buffer = client
        .buffer(values_to_bytes::<f32>(&[0.0, 1.0]).as_slice(), BufferType::F32, [2u64], None, local_device, None)
        .unwrap();
    let source_array_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]), None, Some(sharding.clone())).unwrap();
    let source_array =
        Array::from_addressable_buffers(source_array_type, mesh.clone(), vec![local_source_buffer]).unwrap();

    let context = CompilationContext::new(&client);
    let copied_array = source_array
        .to(&context, DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharding.clone() }, true)
        .unwrap();
    let expected_visualization =
        format!("┌─────┬─────┐\n│{:^5}│{:^5}│\n└─────┴─────┘", local_device_id, remote_device_id);

    assert_eq!(copied_array.addressable_shards().count(), 1);
    assert_eq!(copied_array.sharding().visualize().unwrap().render(false), expected_visualization);
    let copied_shard_bytes = copied_array
        .device_shard(local_device_id)
        .unwrap()
        .buffer()
        .unwrap()
        .copy_to_host(None)
        .unwrap()
        .r#await()
        .unwrap();
    assert_eq!(copied_shard_bytes.as_slice().len() % size_of::<f32>(), 0);
    assert_eq!(values_from_bytes::<f32>(copied_shard_bytes.as_slice()), vec![0.0, 1.0]);
    assert!(copied_array.device_shard(remote_device_id).unwrap().buffer().is_none());
}

#[test]
fn test_device_put_rejects_mismatched_src_for_array_leaf() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
    let client_device = client.addressable_devices().unwrap().remove(0);
    let source_device = Device::from_pjrt(client_device).unwrap();
    let source_mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
        vec![source_device],
    )
    .unwrap();
    let source_sharding = Sharding::replicated(source_mesh.logical_mesh().clone(), 1);
    let source_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)]), None, Some(source_sharding.clone())).unwrap();
    let source_array = Array::from_host_buffer(
        &client,
        source_type,
        source_mesh.clone(),
        values_to_bytes::<f32>(&[0.0, 1.0]).as_slice(),
    )
    .unwrap();
    let expected_src = DevicePutTarget::device(Device::new(source_device.id() + 1, 0)).resolve(1).unwrap();
    let actual_src = (source_mesh, source_sharding);

    let context = CompilationContext::new(&client);
    assert!(matches!(
        device_put(
            &context,
            source_array,
            DevicePutOptions::new(
                Option::<DevicePutTarget>::None,
                Some(DevicePutTarget::device(Device::new(source_device.id() + 1, 0))),
                Option::<bool>::None,
                Option::<Option<bool>>::None,
            ),
        ),
        Err(ArrayError::SourcePlacementMismatch {
            expected_mesh: reported_expected_mesh,
            expected_sharding: reported_expected_sharding,
            actual_mesh: reported_actual_mesh,
            actual_sharding: reported_actual_sharding,
        }) if &reported_expected_mesh == &expected_src.0
            && &reported_expected_sharding == &expected_src.1
            && &reported_actual_mesh == &actual_src.0
            && &reported_actual_sharding == &actual_src.1,
    ));
}

#[test]
fn test_array_driven_shardy_jit_sharded_matmul_on_cpu() {
    // Use the same 8-device CPU setup as `ryft_pjrt` tests.
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(8) }))
        .expect("failed to create 8-device CPU client");
    let client_devices = client.addressable_devices().unwrap();
    assert_eq!(client_devices.len(), 8);

    // Build mesh used for runtime arrays. In a JIT setting, we derive StableHLO Shardy
    // annotations directly from these arrays.
    let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
    let mesh =
        DeviceMesh::new(LogicalMesh::new(vec![MeshAxis::new("x", 8, MeshAxisType::Auto).unwrap()]).unwrap(), devices)
            .unwrap();

    let lhs_sharding = Sharding::new(
        mesh.logical_mesh().clone(),
        vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
    )
    .unwrap();
    let rhs_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 2);

    // Global lhs matrix is 8x4, split by rows across 8 devices (each shard is 1x4).
    // Row i is [i, i+1, i+2, i+3].
    let lhs_buffers = client_devices
        .iter()
        .enumerate()
        .map(|(row_index, device)| {
            let row = row_index as f32;
            client
                .buffer(
                    values_to_bytes::<f32>(&[row, row + 1.0, row + 2.0, row + 3.0]).as_slice(),
                    BufferType::F32,
                    [1u64, 4u64],
                    None,
                    device.clone(),
                    None,
                )
                .unwrap()
        })
        .collect::<Vec<_>>();

    // Global rhs matrix is replicated on each device.
    // [[1, 2], [0, 1], [1, 0], [2, 1]]
    let rhs_values = [1.0f32, 2.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0];
    let rhs_buffers = client_devices
        .iter()
        .map(|device| {
            client
                .buffer(
                    values_to_bytes::<f32>(rhs_values.as_slice()).as_slice(),
                    BufferType::F32,
                    [4u64, 2u64],
                    None,
                    device.clone(),
                    None,
                )
                .unwrap()
        })
        .collect::<Vec<_>>();

    let lhs_array_type = ArrayType::new(
        DataType::F32,
        Shape::new(vec![Size::Static(8), Size::Static(4)]),
        None,
        Some(lhs_sharding.clone()),
    )
    .unwrap();
    let rhs_array_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4), Size::Static(2)]), None, Some(rhs_sharding))
            .unwrap();
    let lhs_array = Array::from_addressable_buffers(lhs_array_type, mesh.clone(), lhs_buffers).unwrap();
    let rhs_array = Array::from_addressable_buffers(rhs_array_type, mesh.clone(), rhs_buffers).unwrap();

    assert_eq!(lhs_array.data_type(), DataType::F32);
    assert_eq!(rhs_array.data_type(), DataType::F32);
    assert_eq!(lhs_array.addressable_shards().count(), 8);
    assert!(lhs_array.shards().iter().all(|shard| shard.buffer().is_some()));

    // Derive Shardy attributes from runtime arrays (JIT-style).
    let context = ryft_mlir::Context::new();
    let mesh_module = context.module(context.unknown_location()).unwrap();
    let mesh_operation = mesh_module
        .body()
        .unwrap()
        .append_operation(lhs_array.sharding().mesh().to_mlir(context.unknown_location()).unwrap())
        .unwrap()
        .to_string();
    let lhs_sharding_attribute = lhs_array.to_shardy_tensor_sharding_attribute().unwrap();
    let rhs_sharding_attribute = rhs_array.to_shardy_tensor_sharding_attribute().unwrap();
    let output_sharding_attribute = lhs_array.to_shardy_tensor_sharding_attribute().unwrap();

    assert_eq!(mesh_operation, "sdy.mesh @mesh = <[\"x\"=8]>");
    assert_eq!(lhs_sharding_attribute, "#sdy.sharding<@mesh, [{\"x\"}, {}]>");
    assert_eq!(rhs_sharding_attribute, "#sdy.sharding<@mesh, [{}, {}]>");

    let mlir_program = format!(
        r#"
                module {{
                    {mesh_operation}
                    func.func @main(
                        %arg0: tensor<8x4xf32> {{sdy.sharding = {lhs_sharding_attribute}}},
                        %arg1: tensor<4x2xf32> {{sdy.sharding = {rhs_sharding_attribute}}}
                    ) -> (tensor<8x2xf32> {{sdy.sharding = {output_sharding_attribute}}}) {{
                        %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [] x [], contracting_dims = [1] x [0]
                            : (tensor<8x4xf32>, tensor<4x2xf32>) -> tensor<8x2xf32>
                        return %0 : tensor<8x2xf32>
                    }}
                }}
            "#
    );
    let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
    let executable = client.compile(&program, &test_spmd_compilation_options(8)).unwrap();

    let execution_devices = executable.addressable_devices().unwrap();
    assert_eq!(execution_devices.len(), 8);
    let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
    let row_start_by_device = execution_device_ids
        .iter()
        .map(|device_id| {
            let row_start = lhs_array.device_shard(*device_id).unwrap().slice()[0].start;
            (*device_id, row_start)
        })
        .collect::<HashMap<_, _>>();

    let execute_arguments =
        Array::into_execute_arguments(vec![lhs_array, rhs_array], execution_device_ids.as_slice()).unwrap();
    let outputs = executable
        .execute(execute_arguments.as_execution_device_inputs(), 0, None, Some(file!()), None, None)
        .unwrap();

    // Validate each output shard: row r should be [4r + 8, 4r + 4].
    assert_eq!(outputs.len(), execution_device_ids.len());
    for (output, device_id) in outputs.into_iter().zip(execution_device_ids.iter().copied()) {
        output.done.r#await().unwrap();
        assert_eq!(output.outputs.len(), 1);
        let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
        let values: [f32; 2] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
        let row = *row_start_by_device.get(&device_id).unwrap() as f32;
        assert_eq!(values[0], 4.0 * row + 8.0);
        assert_eq!(values[1], 4.0 * row + 4.0);
    }
}

fn four_device_mesh_x(client: &ryft_pjrt::Client<'_>) -> DeviceMesh {
    let devices = client
        .addressable_devices()
        .unwrap()
        .iter()
        .map(|device| Device::from_pjrt(device).unwrap())
        .collect::<Vec<_>>();
    DeviceMesh::new(LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Auto).unwrap()]).unwrap(), devices)
        .unwrap()
}

#[test]
fn test_compiled_reshard_replicated_to_sharded_on_same_mesh() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) })).unwrap();
    let mesh = four_device_mesh_x(&client);
    let context = CompilationContext::new(&client);

    let values = [10.0f32, 11.0, 12.0, 13.0];
    let replicated_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
    let source_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(values.len())]), None, Some(replicated_sharding))
            .unwrap();
    let source_array =
        Array::from_host_buffer(&client, source_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
            .unwrap();

    let sharded_target = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    let resharded = source_array
        .to(
            &context,
            crate::arrays_v0::DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharded_target },
            false,
        )
        .unwrap();

    assert_eq!(resharded.addressable_shards().count(), 4);
    let device_ids =
        client.addressable_devices().unwrap().iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
    for (shard_index, device_id) in device_ids.iter().copied().enumerate() {
        let shard_bytes = resharded
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let shard_values = values_from_bytes::<f32>(shard_bytes.as_slice());
        assert_eq!(shard_values, vec![values[shard_index]]);
    }
}

#[test]
fn test_compiled_reshard_sharded_to_replicated_on_same_mesh() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) })).unwrap();
    let mesh = four_device_mesh_x(&client);
    let context = CompilationContext::new(&client);

    let values = [20.0f32, 21.0, 22.0, 23.0];
    let sharded_sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    let source_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(values.len())]), None, Some(sharded_sharding))
            .unwrap();
    let source_array =
        Array::from_host_buffer(&client, source_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
            .unwrap();

    let replicated_target = Sharding::replicated(mesh.logical_mesh().clone(), 1);
    let resharded = source_array
        .to(
            &context,
            crate::arrays_v0::DevicePutTarget::Placement { mesh: mesh.clone(), sharding: replicated_target },
            false,
        )
        .unwrap();

    let device_ids =
        client.addressable_devices().unwrap().iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
    for device_id in device_ids {
        let shard_bytes = resharded
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let shard_values = values_from_bytes::<f32>(shard_bytes.as_slice());
        assert_eq!(shard_values, values.to_vec());
    }
}

#[test]
fn test_compiled_reshard_sharded_to_differently_sharded_on_same_mesh() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) })).unwrap();
    let devices = client
        .addressable_devices()
        .unwrap()
        .iter()
        .map(|device| Device::from_pjrt(device).unwrap())
        .collect::<Vec<_>>();
    let mesh = DeviceMesh::new(logical_mesh_2x2(), devices).unwrap();
    let context = CompilationContext::new(&client);

    let row_values = [
        100.0f32, 101.0, 102.0, 103.0, 110.0, 111.0, 112.0, 113.0, 120.0, 121.0, 122.0, 123.0, 130.0, 131.0, 132.0,
        133.0,
    ];

    let sharded_along_x = Sharding::new(
        mesh.logical_mesh().clone(),
        vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
    )
    .unwrap();
    let source_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4), Size::Static(4)]), None, Some(sharded_along_x))
            .unwrap();
    let source_array =
        Array::from_host_buffer(&client, source_type, mesh.clone(), values_to_bytes::<f32>(&row_values).as_slice())
            .unwrap();

    let sharded_along_y = Sharding::new(
        mesh.logical_mesh().clone(),
        vec![ShardingDimension::replicated(), ShardingDimension::sharded(["y"])],
    )
    .unwrap();
    let resharded = source_array
        .to(
            &context,
            crate::arrays_v0::DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharded_along_y },
            false,
        )
        .unwrap();
    assert_eq!(resharded.addressable_shards().count(), 4);

    for shard in resharded.addressable_shards() {
        let shard_buffer = shard.buffer().unwrap();
        let shard_bytes = shard_buffer.copy_to_host(None).unwrap().r#await().unwrap();
        let shard_values = values_from_bytes::<f32>(shard_bytes.as_slice());
        let slice = shard.descriptor().slice();
        let row_range = slice[0].clone();
        let col_range = slice[1].clone();
        let mut expected = Vec::new();
        for row in row_range.clone() {
            for column in col_range.clone() {
                expected.push(row_values[row * 4 + column]);
            }
        }
        assert_eq!(shard_values, expected, "shard slice {:?}", slice);
    }
}

#[test]
fn test_compiled_reshard_cross_mesh_replicated_source_to_sharded_destination() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) })).unwrap();
    let client_devices = client.addressable_devices().unwrap();

    // Source lives on a 1-device sub-mesh.
    let source_device_id = client_devices[0].id().unwrap();
    let source_mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
        vec![Device::new(source_device_id, client_devices[0].process_index().unwrap())],
    )
    .unwrap();
    let values = [10.0f32, 11.0, 12.0, 13.0];
    let source_type = ArrayType::new(
        DataType::F32,
        Shape::new(vec![Size::Static(values.len())]),
        None,
        Some(Sharding::replicated(source_mesh.logical_mesh().clone(), 1)),
    )
    .unwrap();
    let source_array =
        Array::from_host_buffer(&client, source_type, source_mesh, values_to_bytes::<f32>(&values).as_slice()).unwrap();

    // Reshard onto the full 4-device mesh, sharded along "x".
    let target_mesh = four_device_mesh_x(&client);
    let target_sharding =
        Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    let context = CompilationContext::new(&client);
    let resharded = source_array
        .to(
            &context,
            crate::arrays_v0::DevicePutTarget::Placement { mesh: target_mesh.clone(), sharding: target_sharding },
            false,
        )
        .unwrap();

    assert_eq!(resharded.addressable_shards().count(), 4);
    let device_ids = client_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
    for (shard_index, device_id) in device_ids.iter().copied().enumerate() {
        let shard_bytes = resharded
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let shard_values = values_from_bytes::<f32>(shard_bytes.as_slice());
        assert_eq!(shard_values, vec![values[shard_index]]);
    }
}

#[test]
fn test_to_device_donates_source_and_returns_independently_readable_output() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) })).unwrap();
    let mesh = four_device_mesh_x(&client);
    let context = CompilationContext::new(&client);

    let values = [300.0f32, 301.0, 302.0, 303.0];
    let replicated_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
    let source_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(values.len())]), None, Some(replicated_sharding))
            .unwrap();
    let source_array =
        Array::from_host_buffer(&client, source_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
            .unwrap();

    // to_device consumes self and donates the source's input buffers to the compiled SPMD
    // reshard. PJRT may reuse those buffers' memory for the output; the donated source must not
    // be readable after this call (it has been consumed), but the returned array's output
    // buffers must be independently addressable on each device.
    let target_sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    let moved = source_array
        .to(&context, DevicePutTarget::Placement { mesh: mesh.clone(), sharding: target_sharding }, true)
        .unwrap();

    let device_ids =
        client.addressable_devices().unwrap().iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
    for (shard_index, device_id) in device_ids.iter().copied().enumerate() {
        let shard_bytes = moved
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(values_from_bytes::<f32>(shard_bytes.as_slice()), vec![values[shard_index]]);
    }
}

/// Validates that the second reshard with the same structural signature hits the cache and
/// **skips** the trace + lower work — the structural cache key (input/output shardings, mesh,
/// shape, dtype) matches without ever materializing the MLIR text. Mirrors how JAX caches
/// `jit` invocations on abstract value signatures.
#[test]
#[ignore = "timing-sensitive; runs locally to validate the structural cache hit"]
fn bench_compiled_reshard_cache_hit_avoids_trace_and_lower() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) })).unwrap();
    let mesh = four_device_mesh_x(&client);
    let context = CompilationContext::new(&client);
    let values = (0..4096).map(|index| index as f32).collect::<Vec<_>>();
    let source_type = ArrayType::new(
        DataType::F32,
        Shape::new(vec![Size::Static(values.len())]),
        None,
        Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
    )
    .unwrap();
    let source_array =
        Array::from_host_buffer(&client, source_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
            .unwrap();
    let target_sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

    let cold_start = std::time::Instant::now();
    let _ = source_array
        .clone()
        .to(
            &context,
            crate::arrays_v0::DevicePutTarget::Placement { mesh: mesh.clone(), sharding: target_sharding.clone() },
            false,
        )
        .unwrap();
    let cold_elapsed = cold_start.elapsed();

    let warm_start = std::time::Instant::now();
    let _ = source_array
        .to(
            &context,
            crate::arrays_v0::DevicePutTarget::Placement { mesh: mesh.clone(), sharding: target_sharding },
            false,
        )
        .unwrap();
    let warm_elapsed = warm_start.elapsed();

    eprintln!("cold reshard (trace+lower+compile+execute): {cold_elapsed:?}");
    eprintln!("warm reshard (structural cache hit + execute, no trace/lower): {warm_elapsed:?}");
    assert_eq!(context.cache_size(), 1);
}

#[test]
fn test_compilation_context_preserves_custom_base_options() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) })).unwrap();
    let mesh = four_device_mesh_x(&client);

    // Construct contexts with two distinct base option templates. Different `Debug`
    // representations should produce different cache keys, so the same MLIR program compiles
    // separately under each context.
    let mut custom_options = CompilationOptions::default();
    custom_options.matrix_unit_operand_precision = Precision::Highest as i32;
    let default_context = CompilationContext::new(&client);
    let custom_context = CompilationContext::with_options(&client, custom_options);

    let values = [200.0f32, 201.0, 202.0, 203.0];
    let replicated_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
    let source_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(values.len())]), None, Some(replicated_sharding))
            .unwrap();

    // Reshard once per context with identical inputs. Each context compiles independently.
    let sharded_target = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    let source_for_default =
        Array::from_host_buffer(&client, source_type.clone(), mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
            .unwrap();
    let _ = source_for_default
        .to(
            &default_context,
            crate::arrays_v0::DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharded_target.clone() },
            false,
        )
        .unwrap();
    let source_for_custom =
        Array::from_host_buffer(&client, source_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
            .unwrap();
    let _ = source_for_custom
        .to(&custom_context, crate::arrays_v0::DevicePutTarget::Placement { mesh, sharding: sharded_target }, false)
        .unwrap();

    assert_eq!(default_context.cache_size(), 1);
    assert_eq!(custom_context.cache_size(), 1);
    assert_eq!(
        custom_context.base_options().matrix_unit_operand_precision,
        Precision::Highest as i32,
        "custom CompilationContext should retain the requested base options",
    );
}

#[test]
fn test_to_placement_rejects_non_addressable_destination_device() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
    let local_device = client.addressable_devices().unwrap().remove(0);
    let local_device_id = local_device.id().unwrap();
    let remote_device_id = local_device_id + 1;
    let context = CompilationContext::new(&client);

    // Source array on the local device, replicated on a 1-device sub-mesh.
    let source_mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
        vec![Device::new(local_device_id, local_device.process_index().unwrap())],
    )
    .unwrap();
    let values = [100.0f32, 101.0, 102.0, 103.0];
    let source_type = ArrayType::new(
        DataType::F32,
        Shape::new(vec![Size::Static(values.len())]),
        None,
        Some(Sharding::replicated(source_mesh.logical_mesh().clone(), 1)),
    )
    .unwrap();
    let source_array =
        Array::from_host_buffer(&client, source_type, source_mesh, values_to_bytes::<f32>(&values).as_slice()).unwrap();

    // Destination mesh contains a device on a remote process (process_index 1) that is not
    // addressable from the current client. The compiled cross-mesh path surfaces this as a typed
    // error rather than silently failing inside PJRT.
    let target_mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
        vec![Device::new(local_device_id, local_device.process_index().unwrap()), Device::new(remote_device_id, 1)],
    )
    .unwrap();
    let target_sharding = Sharding::replicated(target_mesh.logical_mesh().clone(), 1);

    let result = source_array.to(
        &context,
        crate::arrays_v0::DevicePutTarget::Placement { mesh: target_mesh, sharding: target_sharding },
        false,
    );
    assert!(
        matches!(
            result,
            Err(ArrayError::NonAddressableDestinationDevice {
                device_id,
                process_index: 1,
            }) if device_id == remote_device_id
        ),
        "expected NonAddressableDestinationDevice, got {result:?}"
    );
}

#[test]
fn test_compiled_reshard_with_explicit_mesh_axes() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) })).unwrap();
    let devices = client
        .addressable_devices()
        .unwrap()
        .iter()
        .map(|device| Device::from_pjrt(device).unwrap())
        .collect::<Vec<_>>();
    // Build a 4-device mesh whose axis type is `Explicit` (sharding propagation is user-declared
    // rather than inferred). The compiled reshard path should accept this just like `Auto`.
    let explicit_mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap()]).unwrap(),
        devices,
    )
    .unwrap();
    let context = CompilationContext::new(&client);

    let values = [400.0f32, 401.0, 402.0, 403.0];
    let replicated_sharding = Sharding::replicated(explicit_mesh.logical_mesh().clone(), 1);
    let source_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(values.len())]), None, Some(replicated_sharding))
            .unwrap();
    let source_array = Array::from_host_buffer(
        &client,
        source_type,
        explicit_mesh.clone(),
        values_to_bytes::<f32>(&values).as_slice(),
    )
    .unwrap();

    let sharded_target =
        Sharding::new(explicit_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    let resharded = source_array
        .to(
            &context,
            crate::arrays_v0::DevicePutTarget::Placement { mesh: explicit_mesh, sharding: sharded_target },
            false,
        )
        .unwrap();
    assert_eq!(resharded.addressable_shards().count(), 4);

    let device_ids =
        client.addressable_devices().unwrap().iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
    for (shard_index, device_id) in device_ids.iter().copied().enumerate() {
        let shard_bytes = resharded
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(values_from_bytes::<f32>(shard_bytes.as_slice()), vec![values[shard_index]]);
    }
}

#[test]
fn test_to_with_manual_mesh_axes_uses_host_fallback() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) })).unwrap();
    let devices = client
        .addressable_devices()
        .unwrap()
        .iter()
        .map(|device| Device::from_pjrt(device).unwrap())
        .collect::<Vec<_>>();
    let manual_mesh =
        DeviceMesh::new(LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(), devices)
            .unwrap();
    let context = CompilationContext::new(&client);

    let values = [40.0f32, 41.0, 42.0, 43.0];
    let replicated_sharding = Sharding::replicated(manual_mesh.logical_mesh().clone(), 1);
    let source_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(values.len())]), None, Some(replicated_sharding))
            .unwrap();
    let source_array =
        Array::from_host_buffer(&client, source_type, manual_mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
            .unwrap();

    // Manual mesh axes cannot be planned by the SPMD partitioner — the compiled path declines.
    // `Array::to` falls through to the host materialization fallback (full source is addressable,
    // destination is on the same single process) and successfully reshapes the data to the
    // requested sharding.
    let sharded_target =
        Sharding::new(manual_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    let resharded = source_array
        .to(
            &context,
            crate::arrays_v0::DevicePutTarget::Placement {
                mesh: manual_mesh.clone(),
                sharding: sharded_target.clone(),
            },
            false,
        )
        .expect("host fallback should satisfy the reshard when both endpoints are local");
    assert_eq!(resharded.sharding(), &sharded_target);
    assert_eq!(resharded.addressable_shards().count(), 4);

    // Each device-local shard now owns one element of the input vector.
    for (index, device) in client.addressable_devices().unwrap().iter().enumerate() {
        let shard_bytes = resharded
            .device_shard(device.id().unwrap())
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(values_from_bytes::<f32>(shard_bytes.as_slice()), vec![values[index]]);
    }

    // No executable was compiled — the host fallback bypasses the compile cache.
    assert_eq!(context.cache_size(), 0, "host fallback should not populate the executable cache");
}

fn two_device_sub_mesh_x(client: &ryft_pjrt::Client<'_>) -> DeviceMesh {
    let client_devices = client.addressable_devices().unwrap();
    let devices = client_devices.iter().take(2).map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
    DeviceMesh::new(LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(), devices)
        .unwrap()
}

#[test]
fn test_compiled_reshard_cross_mesh_sharded_source_to_replicated_destination() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) })).unwrap();
    let context = CompilationContext::new(&client);

    // Source: sharded along "x" on a 2-device sub-mesh (devices 0 and 1 each hold half the data).
    let source_mesh = two_device_sub_mesh_x(&client);
    let source_sharding =
        Sharding::new(source_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    let values = [70.0f32, 71.0, 72.0, 73.0];
    let source_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(values.len())]), None, Some(source_sharding))
            .unwrap();
    let source_array =
        Array::from_host_buffer(&client, source_type, source_mesh, values_to_bytes::<f32>(&values).as_slice()).unwrap();

    // Destination: replicated on the full 4-device mesh. The sharded source first all-gathers on
    // src_mesh, broadcasts onto dst_mesh, and (since the intermediate sharding matches the
    // requested replicated dst_sharding) no further reshard is needed.
    let target_mesh = four_device_mesh_x(&client);
    let target_sharding = Sharding::replicated(target_mesh.logical_mesh().clone(), 1);
    let resharded = source_array
        .to(
            &context,
            crate::arrays_v0::DevicePutTarget::Placement { mesh: target_mesh.clone(), sharding: target_sharding },
            false,
        )
        .unwrap();

    assert_eq!(resharded.addressable_shards().count(), 4);
    let device_ids =
        client.addressable_devices().unwrap().iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
    for device_id in device_ids {
        let shard_bytes = resharded
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(values_from_bytes::<f32>(shard_bytes.as_slice()), values.to_vec());
    }
}

#[test]
fn test_compiled_reshard_cross_mesh_sharded_source_to_sharded_destination() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) })).unwrap();
    let context = CompilationContext::new(&client);

    // Source: sharded along "x" on a 2-device sub-mesh.
    let source_mesh = two_device_sub_mesh_x(&client);
    let source_sharding =
        Sharding::new(source_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    let values = [80.0f32, 81.0, 82.0, 83.0];
    let source_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(values.len())]), None, Some(source_sharding))
            .unwrap();
    let source_array =
        Array::from_host_buffer(&client, source_type, source_mesh, values_to_bytes::<f32>(&values).as_slice()).unwrap();

    // Destination: sharded along "x" on the full 4-device mesh. Each destination shard holds
    // exactly one element of the global array.
    let target_mesh = four_device_mesh_x(&client);
    let target_sharding =
        Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    let resharded = source_array
        .to(
            &context,
            crate::arrays_v0::DevicePutTarget::Placement { mesh: target_mesh.clone(), sharding: target_sharding },
            false,
        )
        .unwrap();

    assert_eq!(resharded.addressable_shards().count(), 4);
    let device_ids =
        client.addressable_devices().unwrap().iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
    for (shard_index, device_id) in device_ids.iter().copied().enumerate() {
        let shard_bytes = resharded
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(values_from_bytes::<f32>(shard_bytes.as_slice()), vec![values[shard_index]]);
    }
}

#[test]
fn test_compiled_reshard_cross_mesh_sharded_source_compiles_two_executables() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) })).unwrap();
    let context = CompilationContext::new(&client);

    let source_mesh = two_device_sub_mesh_x(&client);
    let source_sharding =
        Sharding::new(source_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    let values = [90.0f32, 91.0, 92.0, 93.0];
    let source_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(values.len())]), None, Some(source_sharding))
            .unwrap();
    let source_array =
        Array::from_host_buffer(&client, source_type, source_mesh, values_to_bytes::<f32>(&values).as_slice()).unwrap();

    let target_mesh = four_device_mesh_x(&client);
    let target_sharding =
        Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    assert_eq!(context.cache_size(), 0);
    let _ = source_array
        .to(
            &context,
            crate::arrays_v0::DevicePutTarget::Placement { mesh: target_mesh, sharding: target_sharding },
            false,
        )
        .unwrap();

    // Sub-case B compiles two executables: an all-gather on src_mesh and a replicated->sharded
    // reshard on dst_mesh.
    assert_eq!(context.cache_size(), 2);
}

#[test]
fn test_fast_path_replicated_cross_mesh_to_replicated_destination() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) })).unwrap();
    let client_devices = client.addressable_devices().unwrap();

    // Source: replicated on a 1-device sub-mesh.
    let source_device_id = client_devices[0].id().unwrap();
    let source_mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
        vec![Device::new(source_device_id, client_devices[0].process_index().unwrap())],
    )
    .unwrap();
    let values = [60.0f32, 61.0, 62.0, 63.0];
    let source_type = ArrayType::new(
        DataType::F32,
        Shape::new(vec![Size::Static(values.len())]),
        None,
        Some(Sharding::replicated(source_mesh.logical_mesh().clone(), 1)),
    )
    .unwrap();
    let source_array =
        Array::from_host_buffer(&client, source_type, source_mesh, values_to_bytes::<f32>(&values).as_slice()).unwrap();

    // Target: replicated on the full 4-device mesh. The fast path matches every destination shard
    // to the source's single full-array shard. With the bitcast branch removed, copy_to_device
    // produces independent buffers on each device and copy_to_host works on every one of them.
    let target_mesh = four_device_mesh_x(&client);
    let target_sharding = Sharding::replicated(target_mesh.logical_mesh().clone(), 1);
    let context = CompilationContext::new(&client);
    let resharded = source_array
        .to(
            &context,
            crate::arrays_v0::DevicePutTarget::Placement { mesh: target_mesh.clone(), sharding: target_sharding },
            false,
        )
        .unwrap();

    assert_eq!(resharded.addressable_shards().count(), 4);
    for device in client_devices.iter() {
        let shard_bytes = resharded
            .device_shard(device.id().unwrap())
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(values_from_bytes::<f32>(shard_bytes.as_slice()), values.to_vec());
    }
    // The fast path satisfies this request without compiling any SPMD program.
    assert_eq!(context.cache_size(), 0);
}

#[test]
fn test_compiled_reshard_caches_executable_across_calls() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) })).unwrap();
    let mesh = four_device_mesh_x(&client);
    let context = CompilationContext::new(&client);

    let values = [30.0f32, 31.0, 32.0, 33.0];
    let replicated_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
    let source_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(values.len())]), None, Some(replicated_sharding))
            .unwrap();
    let source_array =
        Array::from_host_buffer(&client, source_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
            .unwrap();

    let sharded_target = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    assert_eq!(context.cache_size(), 0);
    let first = source_array
        .clone()
        .to(
            &context,
            crate::arrays_v0::DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharded_target.clone() },
            false,
        )
        .unwrap();
    let after_first = context.cache_size();
    let second = source_array
        .to(
            &context,
            crate::arrays_v0::DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharded_target },
            false,
        )
        .unwrap();
    let after_second = context.cache_size();

    assert_eq!(first.addressable_shards().count(), 4);
    assert_eq!(second.addressable_shards().count(), 4);
    assert_eq!(after_first, 1, "first reshard should populate the executable cache");
    assert_eq!(after_second, 1, "second reshard with identical MLIR should reuse the cached executable");
}

#[test]
fn test_compilation_context_lru_evicts_oldest_entry() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) })).unwrap();
    let devices = client
        .addressable_devices()
        .unwrap()
        .iter()
        .map(|device| Device::from_pjrt(device).unwrap())
        .collect::<Vec<_>>();
    let mesh = DeviceMesh::new(logical_mesh_2x2(), devices).unwrap();
    // Capacity = 2 so that the third distinct reshard evicts the first.
    let context = CompilationContext::with_capacity(&client, 2);

    // Replicated source on the 2x2 mesh. Three reshards to three different non-replicated
    // shardings all force the compiled path (the fast path returns `None` because no destination
    // shard matches a source shard exactly).
    let values = (0..16).map(|index| index as f32).collect::<Vec<_>>();
    let replicated_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 2);
    let source_type = ArrayType::new(
        DataType::F32,
        Shape::new(vec![Size::Static(4), Size::Static(4)]),
        None,
        Some(replicated_sharding),
    )
    .unwrap();
    let make_source = || {
        Array::from_host_buffer(&client, source_type.clone(), mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
            .unwrap()
    };
    let sharded_along_x = Sharding::new(
        mesh.logical_mesh().clone(),
        vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
    )
    .unwrap();
    let sharded_along_y = Sharding::new(
        mesh.logical_mesh().clone(),
        vec![ShardingDimension::replicated(), ShardingDimension::sharded(["y"])],
    )
    .unwrap();
    let sharded_along_both = Sharding::new(
        mesh.logical_mesh().clone(),
        vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
    )
    .unwrap();

    assert_eq!(context.cache_size(), 0);
    let _ = make_source()
        .to(
            &context,
            crate::arrays_v0::DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharded_along_x },
            false,
        )
        .unwrap();
    assert_eq!(context.cache_size(), 1);
    let _ = make_source()
        .to(
            &context,
            crate::arrays_v0::DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharded_along_y },
            false,
        )
        .unwrap();
    assert_eq!(context.cache_size(), 2);
    let _ = make_source()
        .to(
            &context,
            crate::arrays_v0::DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharded_along_both },
            false,
        )
        .unwrap();
    assert_eq!(context.cache_size(), 2, "third distinct reshard should evict the LRU entry");
}

#[test]
fn test_compilation_context_disk_cache_warm_starts_a_fresh_context() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) })).unwrap();
    let mesh = four_device_mesh_x(&client);
    let cache_dir = tempfile::tempdir().unwrap();

    let values = [200.0f32, 201.0, 202.0, 203.0];
    let replicated_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
    let source_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(values.len())]), None, Some(replicated_sharding))
            .unwrap();
    let target_sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

    // First context: cold compile, disk cache picks up the serialized executable.
    let context_one = CompilationContext::with_disk_cache(&client, cache_dir.path()).unwrap();
    let source =
        Array::from_host_buffer(&client, source_type.clone(), mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
            .unwrap();
    let _ = source
        .to(
            &context_one,
            crate::arrays_v0::DevicePutTarget::Placement { mesh: mesh.clone(), sharding: target_sharding.clone() },
            false,
        )
        .unwrap();
    assert_eq!(context_one.cache_size(), 1, "first reshard should populate the in-memory cache");
    drop(context_one);

    // Second context starts with an empty in-memory cache but loads from the disk cache.
    let context_two = CompilationContext::with_disk_cache(&client, cache_dir.path()).unwrap();
    assert_eq!(context_two.cache_size(), 0, "fresh context starts empty in-memory");
    let source_two =
        Array::from_host_buffer(&client, source_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
            .unwrap();
    let _ = source_two
        .to(&context_two, crate::arrays_v0::DevicePutTarget::Placement { mesh, sharding: target_sharding }, false)
        .unwrap();
    assert_eq!(
        context_two.cache_size(),
        1,
        "second context should populate its in-memory cache via the disk hit (no recompile expected)",
    );
}

#[test]
fn test_compilation_context_clear_cache() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) })).unwrap();
    let mesh = four_device_mesh_x(&client);
    let context = CompilationContext::new(&client);

    let values = [80.0f32, 81.0, 82.0, 83.0];
    let replicated_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
    let source_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(values.len())]), None, Some(replicated_sharding))
            .unwrap();
    let source_array =
        Array::from_host_buffer(&client, source_type, mesh.clone(), values_to_bytes::<f32>(&values).as_slice())
            .unwrap();
    let target_sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

    let _ = source_array
        .to(&context, crate::arrays_v0::DevicePutTarget::Placement { mesh, sharding: target_sharding }, false)
        .unwrap();
    assert_eq!(context.cache_size(), 1);
    context.clear_cache();
    assert_eq!(context.cache_size(), 0);
}
