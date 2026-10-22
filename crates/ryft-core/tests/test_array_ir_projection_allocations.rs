//! Allocation regression tests for reference arrays and projecting members out of [`ArrayIrValue`].
//!
//! The reference [`Array`] backend shares its immutable physical byte storage, so cloning an array must not copy or
//! allocate storage proportional to its payload. Projecting ordinary arrays, first-class runtime dimensions, or
//! ragged aggregates out of [`ArrayIrValue`] must add no allocation at all. These tests use a counting allocator to pin
//! those contracts. This dedicated integration-test binary isolates its allocator and serialized measurement state
//! from unrelated tests, including the separate kernel-allocation regression tests.

use ryft_core::{
    Array, ArrayIrType, ArrayIrValue, ArrayType, DataType, Dimension, DimensionBounds, DimensionType, DimensionValue,
    DimensionVariable, RaggedDimension, RaggedType, RaggedValue, RaggedValueKind, Type, Typed, ValueProjection,
};

include!("support/allocation_measurement.rs");

/// Constructs a large stored reference array outside the measured interval.
fn stored_array() -> ArrayIrValue<Array> {
    ArrayIrValue::Array(Array::vector((0..4096).map(|value| value as f32).collect()).unwrap())
}

/// Constructs a stored first-class runtime dimension outside the measured interval.
fn stored_dimension() -> ArrayIrValue<Array> {
    let variable = DimensionVariable::new("extent", DimensionBounds::positive(Some(9)).unwrap());
    ArrayIrValue::Dimension(DimensionValue::new(DimensionType::from(variable), 4).unwrap())
}

/// Constructs a ragged aggregate outside the measured interval.
fn stored_ragged_value() -> ArrayIrValue<Array> {
    let storage = Array::matrix(2, 3, vec![1.0f32; 6]).unwrap();
    let dimension = DimensionVariable::new("extent", DimensionBounds::non_negative(Some(4)).unwrap());
    let r#type = RaggedType::new(
        storage.r#type().into_owned(),
        RaggedValueKind::Array,
        vec![RaggedDimension::new(1, dimension, vec![0])],
    )
    .unwrap();
    RaggedValue::new(r#type, storage, vec![Array::vector(vec![1i64, 3]).unwrap()]).unwrap().into()
}

#[test]
fn test_large_array_clone_does_not_allocate_payload_storage() {
    let small_statistics = measure_allocations(
        || Array::vector(vec![0.0_f32]).unwrap(),
        |array| {
            let payload = array.storage_bytes().as_ptr();
            let cloned = black_box(&array).clone();
            assert_eq!(cloned.storage_bytes().as_ptr(), payload);
            cloned
        },
    );
    let large_statistics = measure_allocations(
        || Array::vector((0..4096).map(|value| value as f32).collect()).unwrap(),
        |array| {
            let payload = array.storage_bytes().as_ptr();
            let cloned = black_box(&array).clone();
            assert_eq!(cloned.storage_bytes().as_ptr(), payload);
            cloned
        },
    );
    let payload_byte_count = 4096 * size_of::<f32>();
    assert_eq!(large_statistics, small_statistics);
    assert!(large_statistics.allocated_byte_count < payload_byte_count);
    assert!(large_statistics.largest_allocation_byte_count < payload_byte_count);
}

#[test]
fn test_borrowed_array_ir_projection_does_not_allocate() {
    let statistics = measure_allocations(stored_array, |stored| {
        for _ in 0..1_000 {
            let projected = <ArrayIrValue<Array> as ValueProjection<ArrayType>>::projected(black_box(&stored)).unwrap();
            black_box(projected.storage_bytes().as_ptr());
        }
    });
    assert_eq!(statistics, AllocationStatistics::default());
}

#[test]
fn test_consuming_array_ir_projection_does_not_allocate() {
    let statistics = measure_allocations(stored_array, |stored| {
        <ArrayIrValue<Array> as ValueProjection<ArrayType>>::into_projected(stored).unwrap()
    });
    assert_eq!(statistics, AllocationStatistics::default());
}

#[test]
fn test_borrowed_and_consuming_dimension_ir_projection_does_not_allocate() {
    // A borrowed dimension projection hands back a reference into the stored member, so repeating it many times must
    // stay at exactly zero allocations rather than merely at a small constant.
    let borrowed = measure_allocations(stored_dimension, |stored| {
        for _ in 0..1_000 {
            let projected =
                <ArrayIrValue<Array> as ValueProjection<DimensionType>>::projected(black_box(&stored)).unwrap();
            black_box(projected.extent());
        }
    });
    assert_eq!(borrowed, AllocationStatistics::default());

    // The consuming projection transfers ownership of the stored `DimensionValue`, whose declared type shares one
    // reference-counted `DimensionVariable` payload, so no type metadata is copied either. Note that the *projected
    // binding* path (`ProjectedContext::bind`) does clone type metadata by design; that clone is a refcount bump on
    // the variable payload and is not measured here.
    let consuming = measure_allocations(stored_dimension, |stored| {
        <ArrayIrValue<Array> as ValueProjection<DimensionType>>::into_projected(stored).unwrap()
    });
    assert_eq!(consuming, AllocationStatistics::default());
}

#[test]
fn test_borrowed_and_consuming_ragged_ir_projection_does_not_allocate() {
    let borrowed = measure_allocations(stored_ragged_value, |stored| {
        for _ in 0..1_000 {
            let value = ValueProjection::<RaggedType>::projected(black_box(&stored)).unwrap();
            black_box((value.storage().storage_bytes().as_ptr(), value.extents()[0].storage_bytes().as_ptr()));
        }
    });
    assert_eq!(borrowed, AllocationStatistics::default());
    let consuming = measure_allocations(stored_ragged_value, |stored| {
        ValueProjection::<RaggedType>::into_projected(stored).unwrap()
    });
    assert_eq!(consuming, AllocationStatistics::default());
}

#[test]
fn test_ragged_ir_clone_shares_storage_and_extent_metadata() {
    let statistics = measure_allocations(stored_ragged_value, |stored| {
        let cloned = black_box(&stored).clone();
        let original = ValueProjection::<RaggedType>::projected(&stored).unwrap();
        let projected = ValueProjection::<RaggedType>::projected(&cloned).unwrap();
        assert_eq!(original.storage().storage_bytes().as_ptr(), projected.storage().storage_bytes().as_ptr());
        assert_eq!(original.extents().as_ptr(), projected.extents().as_ptr());
        cloned
    });
    assert_eq!(statistics, AllocationStatistics::default());
}

#[test]
fn test_static_array_ir_identity_renaming_does_not_allocate() {
    let statistics = measure_allocations(
        || {
            vec![
                ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2, 3])),
                ArrayIrType::Dimension(DimensionType::from(Dimension::Static(3))),
            ]
        },
        |types| ArrayIrType::derive_identity_renaming(black_box(&types), black_box(&types)).unwrap(),
    );
    assert_eq!(statistics, AllocationStatistics::default());
}

#[test]
fn test_static_array_ir_scope_identity_renaming_does_not_allocate() {
    let statistics = measure_allocations(
        || {
            vec![
                ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2, 3])),
                ArrayIrType::Dimension(DimensionType::from(Dimension::Static(3))),
            ]
        },
        |types| {
            let declared = black_box(&types).iter();
            let actual = black_box(&types).iter();
            ArrayIrType::derive_scope_identity_renaming(declared, actual).unwrap()
        },
    );
    assert_eq!(statistics, AllocationStatistics::default());
}
