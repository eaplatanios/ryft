//! Checks that downstream callers can construct reference-bearing shard maps through the public API.

use pretty_assertions::assert_eq;

use ryft_core::{
    ArrayIrType, ArrayReference, ArrayType, BatchAxis, Context, DataType, Device, DeviceMesh, LogicalMesh, MeshAxis,
    MeshAxisType, Placeholder, ProgramBuilder, ReferenceAddUpdateOperation, ReferenceReadOperation, ReferenceType,
    ShardMap, ShardMapOperation, Sharding, ShardingDimension, Value, batch,
};
use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};
use ryft_xla::experimental::ops::{XlaConstant, XlaOperation};
use ryft_xla::{FromPjrt, XlaArray, XlaOptions, XlaReference, XlaSession, XlaValue, compile_statefully};

/// Returns the `f32` contents of the shard of `array` on each device of its mesh, in mesh order.
fn device_shards(array: &XlaArray<'_>) -> Vec<Vec<f32>> {
    array
        .mesh()
        .devices()
        .iter()
        .map(|device| {
            let shard = array.device_shard(device.id()).unwrap().buffer().unwrap().copy_to_host(None).unwrap();
            let bytes = shard.r#await().unwrap();
            bytes.chunks_exact(4).map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap())).collect()
        })
        .collect()
}

#[test]
fn test_shard_map_operation_from_program() {
    let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
    let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    let local = ArrayType::new_static(DataType::F32, [2])
        .with_sharding(sharding.clone().with_varying_manual_axes(["x"]).unwrap())
        .unwrap();
    let global = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap();
    let reference_type = ArrayIrType::Reference(ReferenceType::new(global.clone()));
    let mut body = ProgramBuilder::<XlaConstant, XlaOperation>::new();
    let reference = body.add_input(ReferenceType::new(local).into());
    let value = body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
    let body = body
        .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![value, reference], vec![Placeholder], vec![Placeholder; 2])
        .unwrap();
    let operation = ShardMapOperation::from_program(
        &body,
        vec![reference_type.clone()],
        ShardMap::new(mesh, vec![sharding.clone()], vec![sharding.clone(), sharding], vec!["x".to_string()]).unwrap(),
    )
    .unwrap();
    let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
    let input = builder.add_input(reference_type.clone());
    let body = builder.import_program(body);
    let outputs = builder
        .add_instruction(XlaOperation::ShardMap(Box::new(operation)), vec![body], vec![input], None)
        .unwrap();
    let outputs = outputs.to_vec();
    let program = builder
        .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder], vec![Placeholder; 2])
        .unwrap();
    assert_eq!(program.output_types(), vec![ArrayIrType::Array(global.clone()), reference_type.clone()]);
    assert_eq!(program.jvp().unwrap().input_types(), vec![reference_type.clone(), reference_type.clone()]);
    let inactive = program.entry_region_ref().jvp(&[]).unwrap();
    assert_eq!(inactive.input_types(), vec![reference_type.clone()]);
    assert_eq!(
        inactive.output_types(),
        vec![ArrayIrType::Array(global.clone()), reference_type, ArrayIrType::Array(global)]
    );
}

#[test]
fn test_batched_shard_map_mutates_mapped_reference_items_on_cpu() {
    // The boundary is `[r: ref<f32[4]>, x: f32[4]] -> f32[4]`, everything sharded along `x`, and the local body
    // performs `add_update(r_local, x_local); read(r_local)`. Batching it over three items that are mapped along the
    // leading axis of both inputs inserts an unpartitioned leading dimension into the referent, so every device owns
    // the `f32[3, 2]` shard of all three items and mutates exactly that shard.
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
        .unwrap();
    let devices = client.addressable_devices().unwrap();
    let devices = devices.iter().take(2).map(|device| Device::from_pjrt(device).unwrap()).collect();
    let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
    let mesh = DeviceMesh::new(logical_mesh.clone(), devices).unwrap();
    let domain = XlaSession::new(&client).domain();
    let sharded = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    let item_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap();
    let shard_map =
        ShardMap::new(logical_mesh.clone(), vec![sharded.clone(); 2], vec![sharded], vec!["x".to_string()]).unwrap();
    let body = {
        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let local_type = shard_map.local_input_type(0, &item_type).unwrap();
        let reference = builder.add_input(ArrayIrType::Reference(ReferenceType::new(local_type.clone())));
        let update = builder.add_input(ArrayIrType::Array(local_type));
        builder.add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None).unwrap();
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap()
    };
    let operation = ShardMapOperation::from_program(
        &body,
        vec![ArrayIrType::Reference(ReferenceType::new(item_type.clone())), ArrayIrType::Array(item_type)],
        shard_map,
    )
    .unwrap();
    let operation = XlaOperation::ShardMap(Box::new(operation));
    let batched_sharding = Sharding::new(
        logical_mesh,
        vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
    )
    .unwrap();
    let batched_type = ArrayType::new_static(DataType::F32, [3, 4]).with_sharding(batched_sharding.clone()).unwrap();
    let compiled = compile_statefully::<_, (ArrayIrType, ArrayIrType), ArrayIrType>(
        |(reference, update)| {
            Ok(batch(
                |(reference, update)| {
                    let context = reference.domain();
                    let mut outputs = context.bind(operation, vec![body], &[reference, update])?;
                    Ok(outputs.pop().unwrap())
                },
                (reference, update),
                (BatchAxis::new(0), BatchAxis::new(0)),
                BatchAxis::new(0),
                None,
            )?)
        },
        (ArrayIrType::Reference(ReferenceType::new(batched_type.clone())), ArrayIrType::Array(batched_type.clone())),
        &domain,
        XlaOptions::new(mesh.clone()),
    )
    .unwrap();

    // Item `i` of the reference holds `[10·i + 1, 10·i + 2, 10·i + 3, 10·i + 4]` and item `i` of the update holds
    // `100·(i + 1)` everywhere.
    let to_bytes = |values: &[f32]| values.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>();
    let reference_values =
        (0..3).flat_map(|item| (1..=4).map(move |column| (10 * item + column) as f32)).collect::<Vec<_>>();
    let update_values = (0..3).flat_map(|item| [(100 * (item + 1)) as f32; 4]).collect::<Vec<_>>();
    let reference = ArrayReference::new(
        XlaArray::from_host_buffer(&domain, batched_type.clone(), mesh.clone(), to_bytes(&reference_values)).unwrap(),
    );
    let update = XlaArray::from_host_buffer(&domain, batched_type, mesh, to_bytes(&update_values)).unwrap();
    let XlaValue::Array(output) = compiled
        .call_statefully(
            &domain,
            (XlaValue::Reference(XlaReference::new(reference.clone(), domain.clone())), XlaValue::Array(update)),
        )
        .unwrap()
    else {
        panic!("the batched shard-map output must be an array")
    };

    // Device 0 owns columns 0 and 1 and device 1 owns columns 2 and 3 of every item, so each device's shard holds the
    // updated values of its own columns of all three items, both in the read output and in the committed referent.
    let expected = vec![
        vec![101.0, 102.0, 211.0, 212.0, 321.0, 322.0],
        vec![103.0, 104.0, 213.0, 214.0, 323.0, 324.0],
    ];
    assert_eq!(output.sharding(), &batched_sharding);
    assert_eq!(device_shards(&output), expected);
    let committed = reference.read().unwrap();
    assert_eq!(committed.sharding(), &batched_sharding);
    assert_eq!(device_shards(&committed), expected);
}
