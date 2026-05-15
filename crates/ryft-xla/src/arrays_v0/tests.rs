use std::collections::HashMap;

use indoc::indoc;
use pretty_assertions::assert_eq;
use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
use ryft_pjrt::{BufferType, ClientOptions, CpuClientOptions, Program, load_cpu_plugin};

use ryft_core::Typed;
use ryft_core::sharding::{Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
use ryft_core::types::data_types::DataType;
use ryft_core::types::{ArrayType, Shape, Size, StaticShape};

use crate::tests::logical_mesh_2x2;
use crate::{Array, Error, ToMlir};

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

fn f32_values_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn two_f32s_from_bytes(bytes: &[u8]) -> [f32; 2] {
    assert_eq!(bytes.len(), 2 * size_of::<f32>());
    let first = f32::from_ne_bytes(bytes[..size_of::<f32>()].try_into().unwrap());
    let second = f32::from_ne_bytes(bytes[size_of::<f32>()..].try_into().unwrap());
    [first, second]
}

fn f32_values_from_bytes(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % size_of::<f32>(), 0);
    bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect()
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
        .map(|device| Device::new(device.id().unwrap(), device.process_index().unwrap()))
        .collect::<Vec<_>>();
    let mesh =
        DeviceMesh::new(LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(), devices)
            .unwrap();
    let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)]), None, None).unwrap();
    let device = client.addressable_devices().unwrap().into_iter().next().unwrap();
    let buffer = client
        .buffer(f32_values_to_bytes(&[1.0, 2.0]).as_slice(), BufferType::F32, [2u64], None, device, None)
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
fn test_array_shape_returns_static_dimensions() {
    let mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
        vec![Device::new(0, 1)],
    )
    .unwrap();
    let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
    let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(7)]), None, Some(sharding)).unwrap();

    let array = Array::from_addressable_buffers(array_type.clone(), mesh.clone(), Vec::new()).unwrap();

    assert_eq!(array.r#type().as_ref(), &array_type);
    assert_eq!(array.shape(), StaticShape::new(vec![7]));
    assert_eq!(array.shards().len(), 1);
    assert_eq!(array.addressable_shards().count(), 0);
    assert_eq!(array.shards()[0].shape(), test_shape(&[7]));
    assert!(!array.shards()[0].buffer().is_some());
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
    let devices = client_devices
        .iter()
        .map(|device| Device::new(device.id().unwrap(), device.process_index().unwrap()))
        .collect::<Vec<_>>();
    let mesh =
        DeviceMesh::new(LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(), devices)
            .unwrap();
    let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    let values = [0.0f32, 1.0, 2.0, 3.0, 4.0];
    let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(5)]), None, Some(sharding)).unwrap();

    let array =
        Array::from_host_buffer(&client, r#type, mesh.clone(), f32_values_to_bytes(values.as_slice()).as_slice())
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
    let devices = client_devices
        .iter()
        .map(|device| Device::new(device.id().unwrap(), device.process_index().unwrap()))
        .collect::<Vec<_>>();
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
        Array::from_host_buffer(&client, r#type, mesh.clone(), f32_values_to_bytes(values.as_slice()).as_slice())
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
    assert_eq!(
        f32_values_from_bytes(
            array
                .device_shard(client_devices[0].id().unwrap())
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap()
                .as_slice()
        ),
        vec![0.0, 1.0, 2.0, 6.0, 7.0, 8.0, 12.0, 13.0, 14.0, 18.0, 19.0, 20.0]
    );
    assert_eq!(
        f32_values_from_bytes(
            array
                .device_shard(client_devices[1].id().unwrap())
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap()
                .as_slice()
        ),
        vec![3.0, 4.0, 5.0, 9.0, 10.0, 11.0, 15.0, 16.0, 17.0, 21.0, 22.0, 23.0]
    );
    assert_eq!(
        f32_values_from_bytes(
            array
                .device_shard(client_devices[2].id().unwrap())
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap()
                .as_slice()
        ),
        vec![24.0, 25.0, 26.0, 30.0, 31.0, 32.0, 36.0, 37.0, 38.0, 42.0, 43.0, 44.0]
    );
    assert_eq!(
        f32_values_from_bytes(
            array
                .device_shard(client_devices[3].id().unwrap())
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap()
                .as_slice()
        ),
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
    let source_values = [0.0f32, 1.0, 2.0, 3.0, 4.0];
    let source_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(5)]), None, Some(source_sharding)).unwrap();
    let source_array = Array::from_host_buffer(
        &client,
        source_type,
        source_mesh,
        f32_values_to_bytes(source_values.as_slice()).as_slice(),
    )
    .unwrap();

    let target_devices = client_devices
        .iter()
        .map(|device| Device::new(device.id().unwrap(), device.process_index().unwrap()))
        .collect::<Vec<_>>();
    let target_mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
        target_devices,
    )
    .unwrap();
    let target_sharding =
        Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

    let moved_array = source_array.to_placement(&client, target_mesh.clone(), target_sharding).unwrap();

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
    let first_shard_bytes = moved_array.device_shard(client_devices[0].id().unwrap()).unwrap();
    let second_shard_bytes = moved_array.device_shard(client_devices[1].id().unwrap()).unwrap();
    assert_eq!(
        f32_values_from_bytes(
            first_shard_bytes.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap().as_slice()
        ),
        vec![0.0, 1.0, 2.0]
    );
    assert_eq!(
        f32_values_from_bytes(
            second_shard_bytes.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap().as_slice()
        ),
        vec![3.0, 4.0]
    );
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
        .buffer(f32_values_to_bytes(&[0.0, 1.0]).as_slice(), BufferType::F32, [2u64], None, local_device, None)
        .unwrap();
    let source_array_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]), None, Some(sharding.clone())).unwrap();
    let source_array =
        Array::from_addressable_buffers(source_array_type, mesh.clone(), vec![local_source_buffer]).unwrap();

    let copied_array = source_array.to_placement(&client, mesh.clone(), sharding).unwrap();
    let expected_visualization =
        format!("┌─────┬─────┐\n│{:^5}│{:^5}│\n└─────┴─────┘", local_device_id, remote_device_id);

    assert_eq!(copied_array.addressable_shards().count(), 1);
    assert_eq!(copied_array.sharding().visualize().unwrap().render(false), expected_visualization);
    assert_eq!(
        f32_values_from_bytes(
            copied_array
                .device_shard(local_device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap()
                .as_slice()
        ),
        vec![0.0, 1.0]
    );
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
        .buffer(f32_values_to_bytes(&[0.0, 1.0]).as_slice(), BufferType::F32, [2u64], None, local_device, None)
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

    assert!(matches!(
        source_array.to_placement(&client, target_mesh, target_sharding),
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
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(5)]), None, Some(source_sharding.clone())).unwrap();
    let first_source_array = Array::from_host_buffer(
        &client,
        first_source_type,
        source_mesh.clone(),
        f32_values_to_bytes(&[0.0, 1.0, 2.0, 3.0, 4.0]).as_slice(),
    )
    .unwrap();
    let second_source_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(5)]), None, Some(source_sharding)).unwrap();
    let second_source_array = Array::from_host_buffer(
        &client,
        second_source_type,
        source_mesh,
        f32_values_to_bytes(&[10.0, 11.0, 12.0, 13.0, 14.0]).as_slice(),
    )
    .unwrap();

    let target_mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
        client_devices
            .iter()
            .map(|device| Device::new(device.id().unwrap(), device.process_index().unwrap()))
            .collect(),
    )
    .unwrap();
    let target_sharding =
        Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

    let moved_arrays = device_put(
        &client,
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
    assert_eq!(
        f32_values_from_bytes(
            moved_arrays
                .0
                .device_shard(client_devices[1].id().unwrap())
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap()
                .as_slice()
        ),
        vec![3.0, 4.0]
    );
    assert_eq!(
        f32_values_from_bytes(
            moved_arrays
                .1
                .device_shard(client_devices[0].id().unwrap())
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap()
                .as_slice()
        ),
        vec![10.0, 11.0, 12.0]
    );
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
        .buffer(f32_values_to_bytes(&[0.0, 1.0]).as_slice(), BufferType::F32, [2u64], None, local_device, None)
        .unwrap();
    let source_array_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]), None, Some(sharding.clone())).unwrap();
    let source_array =
        Array::from_addressable_buffers(source_array_type, mesh.clone(), vec![local_source_buffer]).unwrap();

    let copied_array = device_put(&client, source_array, DevicePutOptions::defaults()).unwrap();
    let expected_visualization =
        format!("┌─────┬─────┐\n│{:^5}│{:^5}│\n└─────┴─────┘", local_device_id, remote_device_id);

    assert_eq!(copied_array.addressable_shards().count(), 1);
    assert_eq!(copied_array.sharding().visualize().unwrap().render(false), expected_visualization);
    assert_eq!(
        f32_values_from_bytes(
            copied_array
                .device_shard(local_device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap()
                .as_slice()
        ),
        vec![0.0, 1.0]
    );
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
        .buffer(f32_values_to_bytes(&[0.0, 1.0]).as_slice(), BufferType::F32, [2u64], None, local_device, None)
        .unwrap();
    let source_array_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]), None, Some(sharding.clone())).unwrap();
    let source_array =
        Array::from_addressable_buffers(source_array_type, mesh.clone(), vec![local_source_buffer]).unwrap();

    let copied_array = source_array
        .to_device(&client, DevicePutTarget::Placement { mesh: mesh.clone(), sharding: sharding.clone() })
        .unwrap();
    let expected_visualization =
        format!("┌─────┬─────┐\n│{:^5}│{:^5}│\n└─────┴─────┘", local_device_id, remote_device_id);

    assert_eq!(copied_array.addressable_shards().count(), 1);
    assert_eq!(copied_array.sharding().visualize().unwrap().render(false), expected_visualization);
    assert_eq!(
        f32_values_from_bytes(
            copied_array
                .device_shard(local_device_id)
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap()
                .as_slice()
        ),
        vec![0.0, 1.0]
    );
    assert!(copied_array.device_shard(remote_device_id).unwrap().buffer().is_none());
}

#[test]
fn test_device_put_rejects_mismatched_src_for_array_leaf() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
    let client_device = client.addressable_devices().unwrap().remove(0);
    let source_device = Device::new(client_device.id().unwrap(), client_device.process_index().unwrap());
    let source_mesh = DeviceMesh::new(
        LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
        vec![source_device],
    )
    .unwrap();
    let source_sharding = Sharding::replicated(source_mesh.logical_mesh().clone(), 1);
    let source_type =
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)]), None, Some(source_sharding.clone())).unwrap();
    let source_array =
        Array::from_host_buffer(&client, source_type, source_mesh.clone(), f32_values_to_bytes(&[0.0, 1.0]).as_slice())
            .unwrap();
    let expected_src = DevicePutTarget::device(Device::new(source_device.id() + 1, 0)).resolve(1).unwrap();
    let actual_src = (source_mesh, source_sharding);

    assert!(matches!(
        device_put(
            &client,
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
    let devices = client_devices
        .iter()
        .map(|device| Device::new(device.id().unwrap(), device.process_index().unwrap()))
        .collect::<Vec<_>>();
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
                    f32_values_to_bytes(&[row, row + 1.0, row + 2.0, row + 3.0]).as_slice(),
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
                    f32_values_to_bytes(rhs_values.as_slice()).as_slice(),
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
        let values = two_f32s_from_bytes(output_bytes.as_slice());
        let row = *row_start_by_device.get(&device_id).unwrap() as f32;
        assert_eq!(values[0], 4.0 * row + 8.0);
        assert_eq!(values[1], 4.0 * row + 4.0);
    }
}
