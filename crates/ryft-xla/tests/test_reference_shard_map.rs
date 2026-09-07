//! Checks that downstream callers can construct reference-bearing shard maps through the public API.

use pretty_assertions::assert_eq;

use ryft_core::{
    ArrayIrType, ArrayType, DataType, LogicalMesh, MeshAxis, MeshAxisType, Placeholder, ProgramBuilder,
    ReferenceReadOperation, ReferenceType, Sharding, ShardingDimension,
};
use ryft_xla::experimental::operations::ShardMapOperation;
use ryft_xla::experimental::ops::{XlaConstant, XlaOperation};

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
        mesh,
        vec![sharding.clone()],
        vec![sharding.clone(), sharding],
        vec!["x".to_string()],
        true,
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
