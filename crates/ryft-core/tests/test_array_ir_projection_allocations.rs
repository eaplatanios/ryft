//! Allocation regression tests for reference arrays and projecting array members out of [`ArrayIrValue`].
//!
//! The reference [`Array`] backend shares its immutable physical byte storage, so cloning an array must not copy or
//! allocate storage proportional to its payload. Direct typed kernels must allocate output storage without adding a
//! payload-sized intermediate, and projecting an array member out of [`ArrayIrValue`] must add no allocation at all.
//! These tests use a counting global allocator to pin those contracts. They live in a dedicated integration-test
//! binary so its global allocator and serialized measurement state cannot affect unrelated tests.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ryft_core::backends::arrays::Array;
use ryft_core::contexts::EagerContext;
use ryft_core::operations::constants::{Fill, Iota};
use ryft_core::operations::math::{Add, Sin};
use ryft_core::{ArrayIrValue, ArrayType, DataType, Dimension, Shape, ValueProjection};

/// Allocator that counts allocations made by this integration-test binary.
struct CountingAllocator;

/// Number of allocations made since the latest measurement reset.
static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Total number of bytes requested across allocations since the latest measurement reset.
static ALLOCATED_BYTE_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Largest allocation requested since the latest measurement reset.
static LARGEST_ALLOCATION_BYTE_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Serializes measurements because the allocation counter is global to this test binary.
static MEASUREMENT_LOCK: Mutex<()> = Mutex::new(());

thread_local! {
    /// Whether allocations on the current test thread belong to the active measurement.
    static COUNT_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
}

/// Records one allocation of `byte_count` bytes when the current thread is inside the measured interval.
fn record_allocation(byte_count: usize) {
    if COUNT_ALLOCATIONS.get() {
        ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTE_COUNT.fetch_add(byte_count, Ordering::Relaxed);
        LARGEST_ALLOCATION_BYTE_COUNT.fetch_max(byte_count, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation(new_size);
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

/// Allocation activity measured during one operation.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct AllocationStatistics {
    /// Number of allocation requests.
    allocation_count: usize,

    /// Total number of bytes requested across all allocations.
    allocated_byte_count: usize,

    /// Number of bytes requested by the largest single allocation.
    largest_allocation_byte_count: usize,
}

/// Runs `setup` outside the counted interval, then returns the allocations performed by `operation`.
fn measure_allocations<S, T>(setup: impl FnOnce() -> S, operation: impl FnOnce(S) -> T) -> AllocationStatistics {
    let _guard = MEASUREMENT_LOCK.lock().expect("allocation measurement mutex is poisoned");
    let state = setup();
    ALLOCATION_COUNT.store(0, Ordering::Relaxed);
    ALLOCATED_BYTE_COUNT.store(0, Ordering::Relaxed);
    LARGEST_ALLOCATION_BYTE_COUNT.store(0, Ordering::Relaxed);
    COUNT_ALLOCATIONS.set(true);
    let result = operation(state);
    COUNT_ALLOCATIONS.set(false);
    black_box(&result);
    AllocationStatistics {
        allocation_count: ALLOCATION_COUNT.load(Ordering::Relaxed),
        allocated_byte_count: ALLOCATED_BYTE_COUNT.load(Ordering::Relaxed),
        largest_allocation_byte_count: LARGEST_ALLOCATION_BYTE_COUNT.load(Ordering::Relaxed),
    }
}

/// Constructs a large stored reference array outside the measured interval.
fn stored_array() -> ArrayIrValue<Array> {
    ArrayIrValue::Array(Array::vector((0..4096).map(|value| value as f32).collect()))
}

#[test]
fn test_large_array_clone_does_not_allocate_payload_storage() {
    let small_statistics = measure_allocations(
        || Array::vector(vec![0.0_f32]),
        |array| {
            let payload = array.storage_bytes().as_ptr();
            let cloned = black_box(&array).clone();
            assert_eq!(cloned.storage_bytes().as_ptr(), payload);
            cloned
        },
    );
    let large_statistics = measure_allocations(
        || Array::vector((0..4096).map(|value| value as f32).collect()),
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
