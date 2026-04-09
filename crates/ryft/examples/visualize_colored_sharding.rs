use ryft::core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, ShardingDimension};
use ryft::core::xla::Sharding;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = LogicalMesh::new(vec![
        MeshAxis::new("data", 2, MeshAxisType::Auto)?,
        MeshAxis::new("replica", 2, MeshAxisType::Auto)?,
        MeshAxis::new("model", 3, MeshAxisType::Auto)?,
    ])?;
    let sharding = Sharding::new(
        mesh,
        vec![ShardingDimension::sharded(["data"]), ShardingDimension::sharded(["model"])],
    )?;
    let visualization = sharding.visualize()?.render(true);

    println!("Colored sharding visualization example");
    println!("mesh axes: data=2, replica=2, model=3");
    println!("sharding: {sharding}");
    println!("note: the `replica` mesh axis is left replicated, so each rendered shard groups two devices");
    println!();
    println!("{visualization}");

    Ok(())
}
