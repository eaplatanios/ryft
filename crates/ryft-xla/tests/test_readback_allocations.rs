//! Measures the host payload allocations of [`Array::copy_to_host_bytes`] through its public API. A dedicated
//! test binary keeps the counting allocator scoped to these measurements, as in `ryft-core`'s allocation tests.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use ryft_core::{
    ArrayType, DataType, Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension,
};
use ryft_pjrt::{ClientOptions, CpuClientOptions};
use ryft_xla::{Array, FromPjrt, Platform, Session};

/// Shape of the measured `u8` array. Each sharded tile holds half the payload bytes.
const SHAPE: [usize; 2] = [64, 1024];

/// Byte count of the dense host representation of the measured array.
const OUTPUT_BYTES: usize = SHAPE[0] * SHAPE[1];

thread_local! {
    /// Counts of tile-sized and output-sized allocations during readback on this thread.
    static PAYLOAD_ALLOCATIONS: Cell<Option<(usize, usize)>> = const { Cell::new(None) };
}

/// System allocator that counts allocations large enough to hold a tile during the measured readback.
struct CountingAllocator;

impl CountingAllocator {
    /// Counts one request of `size` bytes if readback is being measured on the current thread.
    fn record(size: usize) {
        if size >= OUTPUT_BYTES / 2 {
            let _ = PAYLOAD_ALLOCATIONS.try_with(|counts| {
                if let Some((mut tile_allocations, mut output_allocations)) = counts.get() {
                    if size >= OUTPUT_BYTES {
                        output_allocations = output_allocations.saturating_add(1);
                    } else {
                        tile_allocations = tile_allocations.saturating_add(1);
                    }
                    counts.set(Some((tile_allocations, output_allocations)));
                }
            });
        }
    }
}

// All allocations and deallocations use System with the original layouts. Updating the thread-local counters
// needs no allocation and cannot unwind.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::record(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        Self::record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        Self::record(size);
        unsafe { System.realloc(pointer, layout, size) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn test_array_copy_to_host_bytes_payload_allocations() {
    let client = Platform::Cpu
        .client(Some(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() })))
        .unwrap();
    let session = Session::builder().client(client).build().unwrap();
    let logical_mesh = LogicalMesh::new(vec![
        MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
        MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
    ])
    .unwrap();
    let devices = client.addressable_devices().unwrap();
    let mesh = DeviceMesh::new(
        logical_mesh.clone(),
        devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
    )
    .unwrap();
    let bytes = (0..OUTPUT_BYTES).map(|index| (index % 251) as u8).collect::<Vec<_>>();
    let rows = vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()];
    let subrow_shape = [1, SHAPE[0], SHAPE[1]];
    let subrows =
        vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"]), ShardingDimension::replicated()];
    let columns = vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])];
    let replicated = vec![ShardingDimension::replicated(), ShardingDimension::replicated()];

    // A tile can copy directly into the output when it covers whole rows or a partial slice followed by whole trailing
    // dimensions, so those layouts allocate only the output buffer. Readback reads one replica of each distinct tile,
    // and the column layout has two distinct tiles, each replicated along `y`, so it allocates one buffer for each of
    // those two tiles and a separate output buffer for the merge.
    for (shape, dimensions, expected_allocations) in [
        (SHAPE.as_slice(), rows, (0, 1)),
        (subrow_shape.as_slice(), subrows, (0, 1)),
        (SHAPE.as_slice(), columns, (2, 1)),
        (SHAPE.as_slice(), replicated, (0, 1)),
    ] {
        let sharding = Sharding::new(logical_mesh.clone(), dimensions).unwrap();
        let r#type = ArrayType::new_static(DataType::U8, shape).with_sharding(sharding.clone()).unwrap();
        let array = Array::from_host_buffer(&session, r#type, mesh.clone(), None, bytes.as_slice()).unwrap();
        array.block_until_ready().unwrap();

        PAYLOAD_ALLOCATIONS.set(Some((0, 0)));
        let result = array.copy_to_host_bytes();
        let allocations = PAYLOAD_ALLOCATIONS.take().unwrap();

        assert_eq!(allocations, expected_allocations, "readback allocations for {sharding}");
        assert_eq!(result.unwrap(), bytes);
    }
}
