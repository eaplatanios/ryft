//! Allocation regression tests for projecting array members out of [`ArrayProgramValue`].
//!
//! The reference [`Array`] backend owns its scalar payload in a [`Vec`], so cloning an array during projection would
//! copy the complete payload and allocate on every operation. These tests use a counting global allocator to prove
//! that both borrowed projection and consuming ownership transfer remain allocation-free. They live in a dedicated
//! integration-test binary so its global allocator and serialized measurement state cannot affect unrelated tests.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ryft_core::backends::arrays::Array;
use ryft_core::{ArrayProgramValue, ArrayType, ValueProjection};

/// Allocator that counts allocations made by this integration-test binary.
struct CountingAllocator;

/// Number of allocations made since the latest measurement reset.
static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Serializes measurements because the allocation counter is global to this test binary.
static MEASUREMENT_LOCK: Mutex<()> = Mutex::new(());

thread_local! {
    /// Whether allocations on the current test thread belong to the active measurement.
    static COUNT_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
}

/// Records one allocation when the current thread is inside the measured interval.
fn record_allocation() {
    if COUNT_ALLOCATIONS.get() {
        ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation();
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

/// Runs `setup` outside the counted interval, then returns the number of allocations performed by `operation`.
fn measure_allocations<S, T>(setup: impl FnOnce() -> S, operation: impl FnOnce(S) -> T) -> usize {
    let _guard = MEASUREMENT_LOCK.lock().expect("allocation measurement mutex is poisoned");
    let state = setup();
    ALLOCATION_COUNT.store(0, Ordering::Relaxed);
    COUNT_ALLOCATIONS.set(true);
    let result = operation(state);
    COUNT_ALLOCATIONS.set(false);
    black_box(&result);
    ALLOCATION_COUNT.load(Ordering::Relaxed)
}

/// Constructs a large stored reference array outside the measured interval.
fn stored_array() -> ArrayProgramValue<Array> {
    ArrayProgramValue::Array(Array::vector((0..4096).map(|value| value as f32).collect()))
}

#[test]
fn test_borrowed_array_program_projection_does_not_allocate() {
    let allocations = measure_allocations(stored_array, |stored| {
        for _ in 0..1_000 {
            let projected =
                <ArrayProgramValue<Array> as ValueProjection<ArrayType>>::projected(black_box(&stored)).unwrap();
            black_box(projected.values().as_ptr());
        }
    });
    assert_eq!(allocations, 0);
}

#[test]
fn test_consuming_array_program_projection_does_not_allocate() {
    let allocations = measure_allocations(stored_array, |stored| {
        <ArrayProgramValue<Array> as ValueProjection<ArrayType>>::into_projected(stored).unwrap()
    });
    assert_eq!(allocations, 0);
}
