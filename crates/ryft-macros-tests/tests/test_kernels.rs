//! Kernel macro integration through a renamed core-only import and canonical tracing.

extern crate ryft_core as core_alias;

use pretty_assertions::assert_eq;

use core_alias::kernels::{KernelDefinition, KernelError, KernelOperation};
use core_alias::{
    Array, ArrayIrType, ArrayIrValue, ArrayType, Context, DataType, ProjectedValue, ReferenceRead, ReferenceWrite,
    TracingContext,
};

/// Adds two vectors through the generated kernel callable.
#[core_alias::kernels::kernel(crate = "::core_alias", requires = left.shape()[0] == right.shape()[0])]
fn add(
    #[input(data_type = F32, rank = 1)] left: &Array,
    #[input(data_type = F32, rank = 1)] right: &Array,
    #[output(data_type = F32, shape = [left.shape()[0]])] output: &mut Array,
) {
    let first = left.load();
    let second = right.load();
    output.store(first + second);
}

/// Multiplies two matrices using a whole-array kernel body.
#[core_alias::kernels::kernel(crate = "::core_alias", requires = left.shape()[1] == right.shape()[0])]
fn matmul(
    #[input(data_type = F32, rank = 2)] left: &Array,
    #[input(data_type = F32, rank = 2)] right: &Array,
    #[output(data_type = F32, shape = [left.shape()[0], right.shape()[1]])] output: &mut Array,
) {
    output.store(left.load().dot(right.load()));
}

/// Constructs a vector without input parameters.
#[core_alias::kernels::kernel(crate = "::core_alias")]
fn zero(#[output(data_type = F32, shape = [2])] output: &mut Array) {
    let value = zeros::<f32>([2]);
    output.store(value);
}

/// Reduces a vector through the canonical sum operation.
#[core_alias::kernels::kernel(crate = "::core_alias")]
fn sum(#[input(data_type = F32, rank = 1)] input: &Array, #[output(data_type = F32, shape = [])] output: &mut Array) {
    output.store(input.load().sum([0]));
}

#[test]
fn test_kernel_sum() {
    let input = Array::vector(vec![1.0f32, -2.0, 4.0]).unwrap();
    assert_eq!(sum(&input), Ok(Array::scalar(3.0f32).unwrap()));
    assert_eq!(sum(&Array::vector(Vec::<f32>::new()).unwrap()), Ok(Array::scalar(0.0f32).unwrap()));
    let definition = sum::definition(&ArrayType::new_static(DataType::F32, [3])).unwrap();
    let reductions = definition
        .body()
        .instructions()
        .iter()
        .filter_map(|instruction| match instruction.operation() {
            KernelOperation::Portable(core_alias::ArrayIrOperation::Array(core_alias::ArrayOperation::Reduce(
                operation,
            ))) => Some(operation),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(reductions.len(), 1);
    assert_eq!(reductions[0].axes(), &[0]);
    assert_eq!(reductions[0].kind(), core_alias::ReductionKind::Sum);
}

#[test]
fn test_kernel_callable() {
    let left = Array::vector(vec![1.0f32, 2.0]).unwrap();
    let right = Array::vector(vec![3.0f32, 5.0]).unwrap();
    assert_eq!(add(&left, &right), Ok(Array::vector(vec![4.0f32, 7.0]).unwrap()));
    assert_eq!(zero::<Array>(), Ok(Array::vector(vec![0.0f32, 0.0]).unwrap()));
}

#[test]
fn test_kernel_definition() {
    let r#type = ArrayType::new_static(DataType::F32, [2, 2]);
    let generated = matmul::definition(&r#type, &r#type).unwrap();
    let explicit: KernelDefinition = KernelDefinition::trace(generated.operation().clone(), |(references, _)| {
        references[2].write(&core_alias::kernels::dot(&references[0].read()?, &references[1].read()?)?)
    })
    .unwrap();
    assert_eq!(generated.semantic_key().unwrap(), explicit.semantic_key().unwrap());
    let left = Array::from_elements(r#type.clone(), &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let right = Array::from_elements(r#type.clone(), &[5.0f32, 6.0, 7.0, 8.0]).unwrap();
    assert_eq!(matmul(&left, &right), Ok(Array::from_elements(r#type, &[19.0f32, 22.0, 43.0, 50.0]).unwrap()));
}

#[test]
fn test_kernel_callable_stages() {
    let r#type = ArrayType::new_static(DataType::F32, [2]);
    let (_, program) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
        |inputs: Vec<_>| {
            let left = ProjectedValue::new(inputs[0].clone(), r#type.clone());
            let right = ProjectedValue::new(inputs[1].clone(), r#type.clone());
            Ok(vec![add(&left, &right)?.into_value()])
        },
        vec![ArrayIrType::Array(r#type.clone()); 2],
    )
    .unwrap();
    assert!(matches!(program.instructions()[0].operation(), KernelOperation::Call(_)));
    assert_eq!(
        program.interpret(vec![
            ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0]).unwrap()),
            ArrayIrValue::Array(Array::vector(vec![3.0f32, 5.0]).unwrap())
        ]),
        Ok(vec![ArrayIrValue::Array(Array::vector(vec![4.0f32, 7.0]).unwrap())])
    );
}

#[test]
fn test_kernel_definition_checks_rank_before_shape() {
    let error = matmul::definition(&ArrayType::scalar(DataType::F32), &ArrayType::new_static(DataType::F32, [2, 2]))
        .unwrap_err();
    assert!(matches!(error, KernelError::Type(_)));
    assert_eq!(error.to_string(), "kernel input `left` requires data type `F32` and rank `2`");
}

#[test]
fn test_kernel_definition_checks_full_reduction_dimension() {
    let error = matmul::definition(
        &ArrayType::new_static(DataType::F32, [2, 3]),
        &ArrayType::new_static(DataType::F32, [4, 2]),
    )
    .unwrap_err();
    assert!(matches!(error, KernelError::Type(_)));
    assert_eq!(error.to_string(), "kernel shape requirement is not satisfied");
}

#[test]
fn test_kernel_diagnostics() {
    trybuild::TestCases::new().compile_fail("tests/kernels/*.rs");
}

/// Carries an accumulator through a shape-bounded loop.
#[core_alias::kernels::kernel(crate = "::core_alias")]
fn accumulate(
    #[input(data_type = F32, rank = 1)] input: &Array,
    #[output(data_type = F32, shape = [input.shape()[0]])] output: &mut Array,
) {
    let mut accumulator = zeros::<f32>([input.shape()[0]]);
    for depth in 0..input.shape()[0].div_ceil(1) {
        accumulator += input.load();
    }
    output.store(accumulator);
}

/// Selects additive or subtractive updates inside a staged loop.
#[core_alias::kernels::kernel(crate = "::core_alias")]
fn conditional_accumulate(
    #[input(data_type = Boolean, rank = 0)] flag: &Array,
    #[input(data_type = F32, rank = 1)] input: &Array,
    #[output(data_type = F32, shape = [input.shape()[0]])] output: &mut Array,
) {
    let mut accumulator = zeros::<f32>([input.shape()[0]]);
    for depth in 0..3 {
        if flag.load() {
            accumulator += input.load();
        } else {
            accumulator -= input.load();
        }
    }
    output.store(accumulator);
}

#[test]
fn test_kernel_loop_carried_values() {
    let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
    assert_eq!(accumulate(&input), Ok(Array::vector(vec![2.0f32, 4.0]).unwrap()));
    let empty = Array::vector(Vec::<f32>::new()).unwrap();
    assert_eq!(accumulate(&empty), Ok(empty));
    let definition = accumulate::definition(&ArrayType::new_static(DataType::F32, [1000])).unwrap();
    assert_eq!(
        definition
            .body()
            .instructions()
            .iter()
            .filter(|instruction| matches!(
                instruction.operation(),
                KernelOperation::Portable(core_alias::ArrayIrOperation::While(_))
            ))
            .count(),
        1
    );
}

#[test]
fn test_kernel_value_dependent_branches() {
    let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
    assert_eq!(
        conditional_accumulate(&Array::scalar(true).unwrap(), &input),
        Ok(Array::vector(vec![3.0f32, 6.0]).unwrap())
    );
    assert_eq!(
        conditional_accumulate(&Array::scalar(false).unwrap(), &input),
        Ok(Array::vector(vec![-3.0f32, -6.0]).unwrap())
    );
}

#[test]
fn test_kernel_loop_definition_matches_explicit_staging() {
    let generated = accumulate::definition(&ArrayType::new_static(DataType::F32, [2])).unwrap();
    let explicit: KernelDefinition = KernelDefinition::trace(generated.operation().clone(), |(references, _)| {
        let context = references[0].context();
        let accumulator = core_alias::kernels::zeros::<f32, _>(context, [2])?;
        let values = core_alias::kernels::for_loop(
            context,
            0..2,
            vec![references[0].clone(), references[1].clone(), accumulator],
            |_index, mut values| {
                let input = values[0].read()?;
                let accumulator = values[0]
                    .context()
                    .bind(
                        core_alias::ArrayIrOperation::from(core_alias::ArrayOperation::Add(
                            core_alias::AddOperation::new(),
                        )),
                        vec![],
                        &[values[2].clone(), input],
                    )?
                    .remove(0);
                values[2] = accumulator;
                Ok(values)
            },
        )?;
        references[1].write(&values[2])
    })
    .unwrap();
    assert_eq!(generated.semantic_key().unwrap(), explicit.semantic_key().unwrap());
}

#[test]
fn test_kernel_matmul_small_shapes() {
    for (rows, depth, columns) in [(0, 0, 0), (0, 2, 3), (2, 0, 3), (2, 3, 0), (1, 1, 1), (2, 3, 4), (3, 2, 5)] {
        let left_values = (0..rows * depth).map(|index| (index % 7) as f32 - 3.0).collect::<Vec<_>>();
        let right_values = (0..depth * columns).map(|index| (index % 5) as f32 - 2.0).collect::<Vec<_>>();
        let mut expected = vec![0.0f32; rows * columns];
        for row in 0..rows {
            for column in 0..columns {
                for inner in 0..depth {
                    expected[row * columns + column] +=
                        left_values[row * depth + inner] * right_values[inner * columns + column];
                }
            }
        }
        let left = Array::from_elements(ArrayType::new_static(DataType::F32, [rows, depth]), &left_values).unwrap();
        let right =
            Array::from_elements(ArrayType::new_static(DataType::F32, [depth, columns]), &right_values).unwrap();
        assert_eq!(
            matmul(&left, &right),
            Ok(Array::from_elements(ArrayType::new_static(DataType::F32, [rows, columns]), &expected).unwrap())
        );
    }
}

/// Multiplies matrices with explicitly padded tile loads.
#[core_alias::kernels::kernel(crate = "::core_alias", requires = left.shape()[1] == right.shape()[0])]
fn tiled_matmul(
    #[input(data_type = F32, rank = 2)] left: &Array,
    #[input(data_type = F32, rank = 2)] right: &Array,
    #[output(data_type = F32, shape = [left.shape()[0], right.shape()[1]], tile = [32, 32], boundary = masked)]
    output: &mut Array,
) {
    let [row, column] = output.tile_index();
    let left_tiles = left.tiles([32, 32]).pad(0.0);
    let right_tiles = right.tiles([32, 32]).pad(0.0);
    let mut accumulator = zeros::<f32>([32, 32]);
    for depth in 0..left.shape()[1].div_ceil(32) {
        let left_tile = left_tiles.load([row, depth]);
        let right_tile = right_tiles.load([depth, column]);
        accumulator += left_tile.dot(right_tile);
    }
    output.store(accumulator);
}

#[test]
fn test_kernel_tiled_matmul() {
    for (rows, depth, columns) in
        [(0, 0, 0), (0, 2, 3), (2, 0, 3), (2, 3, 0), (1, 1, 1), (2, 3, 4), (3, 2, 5), (1, 33, 2), (33, 35, 34)]
    {
        let left_values = (0..rows * depth).map(|index| (index % 7) as f32 - 3.0).collect::<Vec<_>>();
        let right_values = (0..depth * columns).map(|index| (index % 5) as f32 - 2.0).collect::<Vec<_>>();
        let mut expected = vec![0.0f32; rows * columns];
        for row in 0..rows {
            for column in 0..columns {
                for inner in 0..depth {
                    expected[row * columns + column] +=
                        left_values[row * depth + inner] * right_values[inner * columns + column];
                }
            }
        }
        let left = Array::from_elements(ArrayType::new_static(DataType::F32, [rows, depth]), &left_values).unwrap();
        let right =
            Array::from_elements(ArrayType::new_static(DataType::F32, [depth, columns]), &right_values).unwrap();
        assert_eq!(
            tiled_matmul(&left, &right),
            Ok(Array::from_elements(ArrayType::new_static(DataType::F32, [rows, columns]), &expected).unwrap())
        );
    }
}

#[test]
fn test_kernel_tiled_definition_matches_explicit_staging() {
    let generated = tiled_matmul::definition(
        &ArrayType::new_static(DataType::F32, [3, 5]),
        &ArrayType::new_static(DataType::F32, [5, 4]),
    )
    .unwrap();
    let explicit: KernelDefinition =
        KernelDefinition::trace(generated.operation().clone(), |(references, coordinates)| {
            let context = references[0].context();
            let accumulator = core_alias::kernels::zeros::<f32, _>(context, [32, 32])?;
            let state = vec![
                references[0].clone(),
                references[1].clone(),
                references[2].clone(),
                coordinates[0].clone(),
                coordinates[1].clone(),
                accumulator,
            ];
            let values = core_alias::kernels::for_loop(context, 0..1, state, |depth, mut values| {
                let context = depth.context();
                let left = core_alias::kernels::tile_load(
                    context,
                    &values[0],
                    vec![32, 32],
                    &[values[3].clone(), depth.clone()],
                    0.0f32,
                )?;
                let right = core_alias::kernels::tile_load(
                    context,
                    &values[1],
                    vec![32, 32],
                    &[depth.clone(), values[4].clone()],
                    0.0f32,
                )?;
                let product = core_alias::kernels::dot(&left, &right)?;
                values[5] = context
                    .bind(
                        core_alias::ArrayIrOperation::from(core_alias::ArrayOperation::Add(
                            core_alias::AddOperation::new(),
                        )),
                        vec![],
                        &[values[5].clone(), product],
                    )?
                    .remove(0);
                Ok(values)
            })?;
            core_alias::kernels::tile_store(context, &references[2], &values[5])
        })
        .unwrap();
    assert_eq!(generated.semantic_key().unwrap(), explicit.semantic_key().unwrap());
}

const LOCATED_SOURCE_LINE: u32 = line!() + 7;
/// Checks source locations on a kernel store.
#[core_alias::kernels::kernel(crate = "::core_alias")]
fn located(
    #[input(data_type = F32, rank = 1)] input: &Array,
    #[output(data_type = F32, shape = [input.shape()[0]])] output: &mut Array,
) {
    output.store(input.load());
}

#[test]
fn test_kernel_source_provenance() {
    let definition = located::definition(&ArrayType::new_static(DataType::F32, [2])).unwrap();
    let write = definition.body().instructions().last().unwrap();
    let (scope, origin) = write.provenance().as_scope().unwrap();
    assert_eq!(scope.name(), format!("{}:{}:5", file!(), LOCATED_SOURCE_LINE));
    assert!(origin.is_unknown());
    let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
    assert_eq!(located(&input), Ok(input));
}
