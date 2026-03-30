use ryft::core::sharding::{DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, MeshDevice, ShardingDimension};
use ryft::core::xla::Sharding;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = DeviceMesh::new(
        LogicalMesh::new(vec![
            MeshAxis::new("data", 2, MeshAxisType::Auto)?,
            MeshAxis::new("replica", 2, MeshAxisType::Auto)?,
            MeshAxis::new("model", 3, MeshAxisType::Auto)?,
        ])?,
        (0..12).map(|device_id| MeshDevice::new(device_id, usize::from(device_id >= 6))).collect(),
    )?;
    let sharding = Sharding::new(
        mesh.logical_mesh.clone(),
        vec![ShardingDimension::sharded(["data"]), ShardingDimension::sharded(["model"])],
        vec![],
        vec![],
        vec![],
    )?;
    let global_shape = [48, 96];
    let visualization = sharding.visualize(&global_shape, &mesh, true)?;

    println!("Colored sharding visualization example");
    println!("global shape: {global_shape:?}");
    println!("mesh axes: data=2, replica=2, model=3");
    println!("sharding: {sharding}");
    println!("note: the `replica` mesh axis is left replicated, so each rendered shard groups two devices");
    println!();
    println!("{visualization}");

    Ok(())
}
