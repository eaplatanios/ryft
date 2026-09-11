//! Allocation regression tests for reference array kernels.
//!
//! Direct typed kernels must allocate output storage without adding a payload-sized intermediate. These tests use a
//! counting global allocator in a dedicated integration-test binary, keeping its counters and serialized measurement
//! state independent of the array cloning and mixed-IR projection contracts.

use ryft_core::operations::random::{RandomAlgorithm, RngBitGenerator};
use ryft_core::{Add, Array, ArrayType, DataType, Dimension, EagerContext, Fill, Iota, Shape, Sin};

include!("support/allocation_measurement.rs");

#[test]
fn test_reference_elementwise_kernels_allocate_only_one_payload_buffer() {
    let small_unary = measure_allocations(|| Array::vector(vec![1.0f32]), |array| array.sin().unwrap());
    let large_unary = measure_allocations(
        || Array::vector((0..4096).map(|value| value as f32).collect()),
        |array| array.sin().unwrap(),
    );
    assert_eq!(large_unary.allocation_count, small_unary.allocation_count);
    assert_eq!(large_unary.allocated_byte_count - small_unary.allocated_byte_count, (4096 - 1) * size_of::<f32>());

    let small_binary = measure_allocations(
        || (Array::vector(vec![1.0f32]), Array::vector(vec![2.0f32])),
        |(left, right)| left.add(&right).unwrap(),
    );
    let large_binary = measure_allocations(
        || {
            (
                Array::vector((0..4096).map(|value| value as f32).collect()),
                Array::vector((0..4096).map(|value| value as f32).collect()),
            )
        },
        |(left, right)| left.add(&right).unwrap(),
    );
    assert_eq!(large_binary.allocation_count, small_binary.allocation_count);
    assert_eq!(large_binary.allocated_byte_count - small_binary.allocated_byte_count, (4096 - 1) * size_of::<f32>());
}

#[test]
fn test_reference_constructor_kernels_allocate_only_one_payload_buffer() {
    let small_fill = measure_allocations(
        || ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)])),
        |r#type| EagerContext::<Array>::new().fill(&r#type, 2.5f32).unwrap(),
    );
    let large_fill = measure_allocations(
        || ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4096)])),
        |r#type| EagerContext::<Array>::new().fill(&r#type, 2.5f32).unwrap(),
    );
    assert_eq!(large_fill.allocation_count, small_fill.allocation_count);
    assert_eq!(large_fill.allocated_byte_count - small_fill.allocated_byte_count, (4096 - 1) * size_of::<f32>());

    let small_iota = measure_allocations(
        || ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Static(1)])),
        |r#type| EagerContext::<Array>::new().iota(&r#type, 0).unwrap(),
    );
    let large_iota = measure_allocations(
        || ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Static(4096)])),
        |r#type| EagerContext::<Array>::new().iota(&r#type, 0).unwrap(),
    );
    assert_eq!(large_iota.allocation_count, small_iota.allocation_count);
    assert_eq!(large_iota.allocated_byte_count - small_iota.allocated_byte_count, (4096 - 1) * size_of::<u32>());

    // Narrow random outputs retain the generated U32 words and construct their output storage directly, without a
    // third payload-sized narrowing buffer.
    let small_random = measure_allocations(
        || (Array::vector(vec![42u64, 7]), ArrayType::new(DataType::U16, Shape::new(vec![Dimension::Static(1)]))),
        |(state, r#type)| state.rng_bit_generator(RandomAlgorithm::ThreeFry, &r#type).unwrap(),
    );
    let large_random = measure_allocations(
        || (Array::vector(vec![42u64, 7]), ArrayType::new(DataType::U16, Shape::new(vec![Dimension::Static(4096)]))),
        |(state, r#type)| state.rng_bit_generator(RandomAlgorithm::ThreeFry, &r#type).unwrap(),
    );
    assert_eq!(large_random.allocation_count, small_random.allocation_count);
    assert_eq!(
        large_random.allocated_byte_count - small_random.allocated_byte_count,
        (4096 - 1) * (size_of::<u32>() + size_of::<u16>()),
    );
}
