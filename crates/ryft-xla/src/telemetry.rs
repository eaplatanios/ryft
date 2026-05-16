//! Process-wide telemetry counters for `ryft-xla` runtime resources.
//!
//! Counters in this module are best-effort instrumentation that mirrors a small subset of JAX's
//! `jax.live_buffers()` / `jax.live_arrays()` debug surface. They are intended for live
//! monitoring (memory leak hunting, allocation hotspot detection) and not for correctness
//! checks: the values are eventually consistent under concurrent writers and may be off by a
//! small amount during rapid construction / destruction.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Global count of live [`Array`](crate::Array) instances. Each construction (including
/// [`Clone::clone`]) increments the counter; each drop decrements it. Cloned `Array`s share
/// their underlying device buffers via `Arc`, but each clone is its own handle and is counted
/// separately — the counter therefore tracks the number of live `Array` *handles*, not the
/// number of distinct device buffers.
///
/// Use [`live_array_count`] to read the current value.
static LIVE_ARRAY_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Returns the current number of live [`Array`](crate::Array) handles. Mirrors a subset of
/// `jax.live_arrays()`'s functionality (count rather than full descriptor list).
///
/// This is best-effort telemetry: under concurrent construction and destruction the value is
/// eventually consistent rather than a snapshot of any specific moment.
#[inline]
pub fn live_array_count() -> usize {
    LIVE_ARRAY_COUNT.load(Ordering::Relaxed)
}

/// Increments [`LIVE_ARRAY_COUNT`]. Called by every [`Array`](crate::Array) constructor and
/// every [`Clone::clone`] implementation.
#[inline]
pub(crate) fn array_constructed() {
    LIVE_ARRAY_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Decrements [`LIVE_ARRAY_COUNT`]. Called by [`Array`](crate::Array)'s [`Drop`] implementation.
#[inline]
pub(crate) fn array_dropped() {
    LIVE_ARRAY_COUNT.fetch_sub(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use ryft_core::{ArrayType, DataType, Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, Shape, Sharding, Size};
    use crate::Array;
    use super::*;

    /// Bulk-balanced operations bring the counter back to its starting baseline within the noise
    /// from other tests' parallel `Array` work. The 1000-cycle volume dominates that noise so
    /// the deltas are observable even under concurrent test execution.
    #[test]
    fn test_array_constructed_and_dropped_balance_under_bulk_operations() {
        let baseline = live_array_count();
        for _ in 0..1000 {
            array_constructed();
        }
        let peak = live_array_count();
        assert!(
            peak >= baseline + 900,
            "1000 constructs should drive the counter up by ~1000 (peak={peak}, baseline={baseline})",
        );
        for _ in 0..1000 {
            array_dropped();
        }
        let after = live_array_count();
        let drift = (after as i64) - (baseline as i64);
        assert!(
            drift.abs() < 200,
            "balanced 1000 constructs + 1000 drops should return within noise of baseline \
             (after={after}, baseline={baseline}, drift={drift})",
        );
    }
    
    /// Constructing and cloning an `Array` must increment the live-array telemetry counter, and
    /// dropping the resulting handles must decrement it. Uses bulk operations so the observed
    /// deltas dominate the noise from other parallel tests creating arrays.
    #[test]
    fn test_live_array_count_tracks_array_construction_and_drop() {
        let shape = Shape::new(vec![Size::Static(2)]);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let device_mesh = DeviceMesh::new(logical_mesh, vec![Device::new(0, 1)]).unwrap();
        let sharding = Sharding::replicated(device_mesh.logical_mesh().clone(), 1);
        let array_type = ArrayType::new(DataType::F32, shape, None, Some(sharding)).unwrap();

        let baseline = live_array_count();
        let arrays: Vec<Array<'_>> = (0..200)
            .map(|_| Array::from_addressable_buffers(array_type.clone(), device_mesh.clone(), Vec::new()).unwrap())
            .collect();
        let after_construct = live_array_count();
        assert!(
            after_construct >= baseline + 180,
            "200 Array constructions should drive the live count up by ~200 (after={after_construct}, baseline={baseline})",
        );

        let clones: Vec<Array<'_>> = arrays.iter().cloned().collect();
        let after_clone = live_array_count();
        assert!(
            after_clone >= after_construct + 180,
            "200 Array clones should drive the live count up by another ~200 (after_clone={after_clone}, after_construct={after_construct})",
        );

        drop(clones);
        drop(arrays);
        let after_drop = live_array_count();
        let drift = (after_drop as i64) - (baseline as i64);
        assert!(
            drift.abs() < 200,
            "after dropping 400 Arrays the live count should return within noise of the baseline (after={after_drop}, baseline={baseline}, drift={drift})",
        );
    }
}
