// Shared source included at the root of each allocation-test binary so every binary owns its allocator and counters.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

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
