//! Process-wide telemetry counters for `ryft-xla` runtime resources.
//!
//! Counters in this module are best-effort instrumentation that mirrors a small subset of JAX's
//! `jax.live_buffers()` / `jax.live_arrays()` debug surface. They are intended for live
//! monitoring (memory leak hunting, allocation hotspot detection) and not for correctness
//! checks: the values are eventually consistent under concurrent writers and may be off by a
//! small amount during rapid construction / destruction.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Cumulative count of [`Array`](crate::Array) constructions (including [`Clone::clone`]) over
/// the lifetime of the process. Monotonically non-decreasing.
static CONSTRUCTED_ARRAY_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Cumulative count of [`Array`](crate::Array) drops over the lifetime of the process.
/// Monotonically non-decreasing.
static DROPPED_ARRAY_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Returns the cumulative number of [`Array`](crate::Array) handles constructed by the process so
/// far, including handles created via [`Clone::clone`]. This counter is monotonically
/// non-decreasing, which makes it suitable for allocation hotspot detection: the difference
/// between two reads bounds from below the number of constructions that happened between them.
#[inline]
pub fn constructed_array_count() -> usize {
    CONSTRUCTED_ARRAY_COUNT.load(Ordering::Relaxed)
}

/// Returns the cumulative number of [`Array`](crate::Array) handles dropped by the process so
/// far. This counter is monotonically non-decreasing, which makes it suitable for deallocation
/// tracking: the difference between two reads bounds from below the number of drops that
/// happened between them.
#[inline]
pub fn dropped_array_count() -> usize {
    DROPPED_ARRAY_COUNT.load(Ordering::Relaxed)
}

/// Returns the current number of live [`Array`](crate::Array) handles, computed as the number of
/// handles constructed so far minus the number dropped so far. Mirrors a subset of
/// `jax.live_arrays()`'s functionality (count rather than full descriptor list). Cloned `Array`s
/// share their underlying device buffers via `Arc`, but each clone is its own handle and is
/// counted separately — the count therefore tracks the number of live `Array` *handles*, not the
/// number of distinct device buffers.
///
/// This is best-effort telemetry: under concurrent construction and destruction the value is
/// eventually consistent rather than a snapshot of any specific moment, and the subtraction
/// saturates at zero if the two underlying counters are read on either side of a concurrent
/// construction.
#[inline]
pub fn live_array_count() -> usize {
    constructed_array_count().saturating_sub(dropped_array_count())
}

/// Increments [`CONSTRUCTED_ARRAY_COUNT`]. Called by every [`Array`](crate::Array) constructor
/// and every [`Clone::clone`] implementation.
#[inline]
pub(crate) fn array_constructed() {
    CONSTRUCTED_ARRAY_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Increments [`DROPPED_ARRAY_COUNT`]. Called by [`Array`](crate::Array)'s [`Drop`]
/// implementation.
#[inline]
pub(crate) fn array_dropped() {
    DROPPED_ARRAY_COUNT.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Array;
    use ryft_core::{
        ArrayType, DataType, Device, DeviceMesh, Dimension, LogicalMesh, MeshAxis, MeshAxisType, Shape, Sharding,
    };

    /// The telemetry hooks advance their respective cumulative counters. Because both counters
    /// are monotonically non-decreasing, concurrent `Array` activity from other tests can only
    /// push them further up, so the lower-bound assertions below are deterministic under
    /// parallel test execution.
    #[test]
    fn test_array_constructed_and_dropped_advance_cumulative_counters() {
        let constructed_baseline = constructed_array_count();
        for _ in 0..1000 {
            array_constructed();
        }
        let constructed_after = constructed_array_count();
        assert!(
            constructed_after >= constructed_baseline + 1000,
            "1000 `array_constructed` calls should advance the constructed counter by at least 1000 \
             (after={constructed_after}, baseline={constructed_baseline})",
        );

        let dropped_baseline = dropped_array_count();
        for _ in 0..1000 {
            array_dropped();
        }
        let dropped_after = dropped_array_count();
        assert!(
            dropped_after >= dropped_baseline + 1000,
            "1000 `array_dropped` calls should advance the dropped counter by at least 1000 \
             (after={dropped_after}, baseline={dropped_baseline})",
        );
    }

    /// Constructing and cloning an `Array` must advance the cumulative constructed counter, and
    /// dropping the resulting handles must advance the cumulative dropped counter. The
    /// assertions are exact lower bounds on monotonic counters, so they hold no matter how many
    /// `Array`s other tests construct or drop concurrently.
    #[test]
    fn test_live_array_count_tracks_array_construction_and_drop() {
        let shape = Shape::new(vec![Dimension::Static(2)]);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let device_mesh = DeviceMesh::new(logical_mesh, vec![Device::new(0, 1)]).unwrap();
        let sharding = Sharding::replicated(device_mesh.logical_mesh().clone(), 1);
        let array_type = ArrayType::new(DataType::F32, shape).with_sharding(sharding).unwrap();

        let constructed_baseline = constructed_array_count();
        let arrays: Vec<Array<'_>> = (0..200)
            .map(|_| {
                Array::from_addressable_buffers(None, array_type.clone(), device_mesh.clone(), Vec::new()).unwrap()
            })
            .collect();
        let after_construct = constructed_array_count();
        assert!(
            after_construct >= constructed_baseline + 200,
            "200 `Array` constructions should advance the constructed counter by at least 200 \
             (after={after_construct}, baseline={constructed_baseline})",
        );

        let clones: Vec<Array<'_>> = arrays.iter().cloned().collect();
        let after_clone = constructed_array_count();
        assert!(
            after_clone >= after_construct + 200,
            "200 `Array` clones should advance the constructed counter by at least another 200 \
             (after_clone={after_clone}, after_construct={after_construct})",
        );

        let dropped_baseline = dropped_array_count();
        drop(clones);
        drop(arrays);
        let after_drop = dropped_array_count();
        assert!(
            after_drop >= dropped_baseline + 400,
            "dropping 400 `Array` handles should advance the dropped counter by at least 400 \
             (after={after_drop}, baseline={dropped_baseline})",
        );
    }
}
