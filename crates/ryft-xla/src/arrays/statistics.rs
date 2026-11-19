use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ryft_core::{ArrayType, DeviceId, DeviceMesh, Sharding, Typed};

use crate::arrays::shards::ArrayShard;
use crate::{Error, ToPjrt};

/// Snapshot of the logical array counts of a [`Session`](crate::Session), which
/// [`Session::array_statistics`](crate::Session::array_statistics) returns for diagnosing array churn and retention.
///
/// The unit that is counted is the logical array, which is the storage that an [`Array`](crate::Array) and all of
/// its clones share. Cloning and dropping array handles is therefore free of bookkeeping, and only creating a logical
/// array or dropping its last handle updates the counts. This matches `jax.live_arrays()`, which tracks one entry per
/// array object rather than one per reference to it. Like the entries of JAX, two logical arrays may refer to the same
/// device buffers (e.g., an array and the array that [`RetainedArray::to_array`](crate::RetainedArray::to_array)
/// creates from it), so the number of live arrays can exceed the number of distinct buffers.
///
/// The counts say nothing about the bytes that the arrays hold or about bytes that the allocator
/// of a device holds for other reasons (e.g., temporaries of a running program). A session built with
/// [`SessionBuilder::live_array_tracking`](crate::SessionBuilder::live_array_tracking) describes each
/// of its live arrays as a [`LiveArray`] and totals their device bytes through
/// [`Session::live_array_bytes_by_device`](crate::Session::live_array_bytes_by_device), and
/// [`Session::device_memory_statistics`](crate::Session::device_memory_statistics) reports
/// the allocator statistics for the PJRT plugins that support it.
///
/// Each count is read independently and without blocking writers. [`Self::live_arrays`] is a single counter rather than
/// a difference of two, so every read returns a value that the count actually had. While arrays are being created and
/// dropped concurrently, the two counts of one snapshot may be from slightly different moments.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ArrayStatistics {
    /// Number of logical arrays that were created through the session and whose storage is still alive. The storage of
    /// an array stays alive for as long as any of its clones exists.
    pub live_arrays: u64,

    /// Number of logical arrays that were created through the session so far, including the ones that have since been
    /// dropped. The difference between two reads is the number of arrays created between them.
    pub created_arrays: u64,
}

/// Description of one live logical array of a [`Session`](crate::Session), which
/// [`Session::live_arrays`](crate::Session::live_arrays) returns for sessions that were built with
/// [`SessionBuilder::live_array_tracking`](crate::SessionBuilder::live_array_tracking). It records the type and
/// placement of the array and the logical sizes of the device buffers that the current process holds for it, as
/// they were when the array was created. Its [`Typed`] type is the type of the array.
///
/// A description corresponds to an element of `jax.live_arrays()`, except that it describes the array instead of being
/// the array. It is metadata alone. It owns no array storage and no device buffers, and it never obtains a reference
/// to either, so tracking and inspecting the live arrays affects neither when buffers are released nor whether an
/// array can donate its buffers to an execution.
#[derive(Clone, Debug)]
pub struct LiveArray {
    /// [`ArrayType`] of the array, shared with it.
    r#type: Arc<ArrayType>,

    /// Runtime placement of the array over all mesh axes, shared with it.
    sharding: Arc<Sharding>,

    /// Concrete [`DeviceMesh`] that the array is placed on, shared with it.
    mesh: Arc<DeviceMesh>,

    /// Device buffers that the current process holds for the array, one per shard that has a buffer.
    buffers: Vec<LiveArrayBuffer>,
}

impl LiveArray {
    /// Returns the runtime placement of the array over all mesh axes, including the auto mesh axes that its type omits.
    #[inline]
    pub fn sharding(&self) -> &Sharding {
        &self.sharding
    }

    /// Returns the concrete [`DeviceMesh`] that the array is placed on.
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        &self.mesh
    }

    /// Returns the number of logical payload bytes in the device buffers that the current process holds for the array.
    /// Unlike [`Array::size_in_bytes`](crate::Array::size_in_bytes), which is the size of the one logical value, this
    /// counts every local buffer, so an array that is replicated over two local devices occupies twice its logical
    /// size. Shards of other processes and bufferless [`DataType::Zero`](ryft_core::DataType::Zero) shards occupy no
    /// bytes. Buffers that the array shares with other live arrays are counted in full, and
    /// [`Session::live_array_bytes_by_device`](crate::Session::live_array_bytes_by_device)
    /// is what counts each buffer once.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Pjrt`] if the element type of the array has no dense storage size and
    /// [`Error::SizeLimitExceeded`] if the total does not fit in a [`usize`].
    #[inline]
    pub fn addressable_size_in_bytes(&self) -> Result<usize, Error> {
        self.buffers.iter().try_fold(0usize, |total, buffer| {
            total.checked_add(self.buffer_size_in_bytes(buffer)?).ok_or_else(|| self.size_limit_exceeded())
        })
    }

    /// Returns the number of logical payload bytes of `buffer`, which must be one of the buffers of this array.
    fn buffer_size_in_bytes(&self, buffer: &LiveArrayBuffer) -> Result<usize, Error> {
        let element_size_in_bytes = self.r#type.data_type().to_pjrt().element_size_in_bytes()?;
        buffer.element_count.checked_mul(element_size_in_bytes).ok_or_else(|| self.size_limit_exceeded())
    }

    /// Returns the error for buffer sizes of this array that do not fit in a [`usize`].
    fn size_limit_exceeded(&self) -> Error {
        Error::SizeLimitExceeded {
            message: format!(
                "device buffer size for live array of type `{}` exceeds the maximum allowed size of {}",
                self.r#type,
                usize::MAX,
            ),
        }
    }
}

impl Typed for LiveArray {
    type Type = ArrayType;

    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayType> {
        Cow::Borrowed(self.r#type.as_ref())
    }
}

/// Information about a device buffer that the current process holds for a [`LiveArray`].
#[derive(Copy, Clone, Debug)]
struct LiveArrayBuffer {
    /// [`DeviceId`] of the device that holds the buffer.
    device_id: DeviceId,

    /// Address of the buffer when the array was registered, which identifies the buffer among the live arrays without
    /// referring to it. It is never turned back into a reference.
    identity: usize,

    /// Number of elements in the shard that the buffer stores.
    element_count: usize,
}

/// Statistics tracker for the logical arrays that are created through one [`Session`](crate::Session), which all
/// handles of the session share. [`Self::register`] counts a new logical array and returns the [`ArrayRegistration`]
/// that its storage holds, and dropping that registration counts the array as released. The counters live next to the
/// reference count of the [`Arc`] that holds the tracker, so registering an array writes to a single cache line that
/// only the threads of one session share.
pub(crate) struct ArrayTracker {
    /// Number of registered logical arrays whose registrations are still alive.
    live: AtomicU64,

    /// Number of logical arrays registered so far, which is also the index of the next registration.
    created: AtomicU64,

    /// Descriptions of the live arrays keyed by registration index, or [`None`] when the session does not track them.
    /// Indices increase with every registration, so the map iterates in creation order.
    live_arrays: Option<Mutex<BTreeMap<u64, LiveArray>>>,
}

impl ArrayTracker {
    /// Creates an [`ArrayTracker`] with no registered arrays. With `track_live_arrays`, it also keeps a description of
    /// every live array for [`Self::live_arrays`] and [`Self::live_array_bytes_by_device`], which costs a lock and an
    /// allocation per registration.
    pub(crate) fn new(track_live_arrays: bool) -> Self {
        Self {
            live: AtomicU64::new(0),
            created: AtomicU64::new(0),
            live_arrays: track_live_arrays.then(|| Mutex::new(BTreeMap::new())),
        }
    }

    /// Returns a snapshot of the counts of this tracker.
    pub(crate) fn statistics(&self) -> ArrayStatistics {
        ArrayStatistics {
            live_arrays: self.live.load(Ordering::Relaxed),
            created_arrays: self.created.load(Ordering::Relaxed),
        }
    }

    /// Returns the descriptions of the live arrays in creation order, or [`None`] if this tracker does not keep them.
    pub(crate) fn live_arrays(&self) -> Option<Vec<LiveArray>> {
        let live_arrays = self.live_arrays.as_ref()?.lock().expect("live array registry mutex should not be poisoned");
        Some(live_arrays.values().cloned().collect())
    }

    /// Returns the number of logical payload bytes that the live arrays hold on each device of the current process,
    /// keyed by [`DeviceId`], or [`None`] if this tracker does not keep descriptions of the live arrays. A device
    /// buffer that several live arrays share is counted once, and devices on which no live array holds a buffer are
    /// absent from the result.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Pjrt`] if the element type of a live array has no dense storage size and
    /// [`Error::SizeLimitExceeded`] if a total does not fit in a [`usize`].
    pub(crate) fn live_array_bytes_by_device(&self) -> Result<Option<HashMap<DeviceId, usize>>, Error> {
        let Some(live_arrays) = &self.live_arrays else {
            return Ok(None);
        };
        let live_arrays = live_arrays.lock().expect("live array registry mutex should not be poisoned");
        let mut counted_buffers = HashSet::new();
        let mut sizes = HashMap::new();
        for array in live_arrays.values() {
            for buffer in &array.buffers {
                if counted_buffers.insert(buffer.identity) {
                    let size: &mut usize = sizes.entry(buffer.device_id).or_default();
                    *size = size
                        .checked_add(array.buffer_size_in_bytes(buffer)?)
                        .ok_or_else(|| array.size_limit_exceeded())?;
                }
            }
        }
        Ok(Some(sizes))
    }

    /// Counts a new logical array with the provided type, placement, and shards, and returns the [`ArrayRegistration`]
    /// that its storage must hold for as long as it is alive.
    ///
    /// A tracker that keeps descriptions of the live arrays records one from the arguments. The description shares the
    /// metadata of the array, and it identifies each device buffer by its address without referring to it. An address
    /// stays unique among the live descriptions because a buffer outlives the registration of every array that holds
    /// it.
    pub(crate) fn register(
        self: &Arc<Self>,
        r#type: &Arc<ArrayType>,
        sharding: &Arc<Sharding>,
        mesh: &Arc<DeviceMesh>,
        shards: &[ArrayShard],
    ) -> ArrayRegistration {
        // The counters order nothing and are only read for diagnostics, so relaxed updates suffice.
        let index = self.created.fetch_add(1, Ordering::Relaxed);
        self.live.fetch_add(1, Ordering::Relaxed);
        if let Some(live_arrays) = &self.live_arrays {
            let buffers = shards
                .iter()
                .filter_map(|shard| {
                    shard.buffer().map(|buffer| LiveArrayBuffer {
                        device_id: shard.device().id(),
                        identity: Arc::as_ptr(buffer) as usize,
                        element_count: shard.slice().iter().map(|slice| slice.len()).product(),
                    })
                })
                .collect();
            let array = LiveArray {
                r#type: Arc::clone(r#type),
                sharding: Arc::clone(sharding),
                mesh: Arc::clone(mesh),
                buffers,
            };
            live_arrays.lock().expect("live array registry mutex should not be poisoned").insert(index, array);
        }
        ArrayRegistration { tracker: Arc::clone(self), index }
    }
}

/// Registration of one logical array with the [`ArrayTracker`] of the session that created it. The storage of the array
/// owns it, so the array stops counting as live exactly when its storage is dropped, which is when the last of its
/// clones is dropped.
pub(crate) struct ArrayRegistration {
    /// [`ArrayTracker`] that counted the array.
    tracker: Arc<ArrayTracker>,

    /// Index that [`ArrayTracker::register`] assigned to the array, which keys its description.
    index: u64,
}

impl Drop for ArrayRegistration {
    fn drop(&mut self) {
        if let Some(live_arrays) = &self.tracker.live_arrays {
            live_arrays.lock().expect("live array registry mutex should not be poisoned").remove(&self.index);
        }
        self.tracker.live.fetch_sub(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::DataType;

    use crate::sessions::tests::domain_mesh;
    use crate::tests::{assert_send_sync, cpu_client, execution_client, replicated_array, values_to_bytes};
    use crate::{Array, RetainedArray, Session};

    use super::*;

    #[test]
    fn test_live_array_sharding() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).live_array_tracking(true).build().unwrap();
        let array = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.0]));
        let live_arrays = session.live_arrays().unwrap();
        assert_eq!(live_arrays[0].sharding(), array.sharding());
    }

    #[test]
    fn test_live_array_mesh() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).live_array_tracking(true).build().unwrap();
        let _array = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.0]));
        let live_arrays = session.live_arrays().unwrap();
        assert_eq!(live_arrays[0].mesh(), &mesh);
    }

    #[test]
    fn test_live_array_addressable_size_in_bytes() {
        let client = cpu_client(2);
        let mesh = domain_mesh(client, "x", 2);
        let session = Session::builder().client(client).live_array_tracking(true).build().unwrap();

        // An array that is replicated over two local devices holds its three `f64` elements once on each device, while
        // its logical size counts them once.
        let replicated = replicated_array(&session, &mesh, DataType::F64, &[3], &values_to_bytes(&[1f64, 2.0, 3.0]));
        let live_arrays = session.live_arrays().unwrap();
        assert_eq!(replicated.size_in_bytes().unwrap(), 24);
        assert_eq!(live_arrays[0].addressable_size_in_bytes().unwrap(), 48);

        // A zero-space array has no device buffers.
        let zero_type = replicated.r#type().into_owned().with_data_type(DataType::Zero);
        let _zero = Array::zero(&session, zero_type, None, mesh.clone()).unwrap();
        let live_arrays = session.live_arrays().unwrap();
        assert_eq!(live_arrays[1].addressable_size_in_bytes().unwrap(), 0);
    }

    #[test]
    fn test_live_array_type() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).live_array_tracking(true).build().unwrap();
        let array = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.0]));
        let live_arrays = session.live_arrays().unwrap();
        assert_eq!(live_arrays[0].r#type(), array.r#type());
    }

    #[test]
    fn test_array_tracker_is_send_and_sync() {
        assert_send_sync::<ArrayTracker>();
        assert_send_sync::<ArrayRegistration>();
    }

    #[test]
    fn test_array_tracker_statistics() {
        let tracker = Arc::new(ArrayTracker::new(false));
        assert_eq!(tracker.statistics(), ArrayStatistics { live_arrays: 0, created_arrays: 0 });
        assert_eq!(tracker.statistics(), ArrayStatistics::default());
    }

    #[test]
    fn test_array_tracker_live_arrays() {
        // Only a tracker that was asked to keep descriptions of the live arrays returns them.
        assert!(ArrayTracker::new(false).live_arrays().is_none());
        assert!(ArrayTracker::new(true).live_arrays().unwrap().is_empty());

        // Live arrays are described in creation order, and an array leaves the snapshot when its last clone drops.
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).live_array_tracking(true).build().unwrap();
        let tracker = session.array_tracker();
        let first = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.0]));
        let second = replicated_array(&session, &mesh, DataType::I32, &[3], &values_to_bytes(&[1i32, 2, 3]));
        let clone = first.clone();
        let live_arrays = tracker.live_arrays().unwrap();
        assert_eq!(live_arrays.len(), 2);
        assert_eq!(live_arrays[0].r#type(), first.r#type());
        assert_eq!(live_arrays[1].r#type(), second.r#type());
        drop(first);
        assert_eq!(tracker.live_arrays().unwrap().len(), 2);
        drop(clone);
        let live_arrays = tracker.live_arrays().unwrap();
        assert_eq!(live_arrays.len(), 1);
        assert_eq!(live_arrays[0].r#type(), second.r#type());

        // A snapshot is unaffected by arrays that are dropped after it was taken, and it keeps none of them alive.
        drop(second);
        assert_eq!(live_arrays.len(), 1);
        assert_eq!(tracker.statistics().live_arrays, 0);
    }

    #[test]
    fn test_array_tracker_live_array_bytes_by_device() {
        // Only a tracker that keeps descriptions of the live arrays can total their bytes.
        assert_eq!(ArrayTracker::new(false).live_array_bytes_by_device().unwrap(), None);
        assert_eq!(ArrayTracker::new(true).live_array_bytes_by_device().unwrap(), Some(HashMap::new()));

        // Two arrays that are replicated over both devices hold their payloads on each of them.
        let client = cpu_client(2);
        let mesh = domain_mesh(client, "x", 2);
        let devices = mesh.devices().iter().map(|device| device.id()).collect::<Vec<_>>();
        let session = Session::builder().client(client).live_array_tracking(true).build().unwrap();
        let tracker = session.array_tracker();
        let vector = replicated_array(&session, &mesh, DataType::F64, &[3], &values_to_bytes(&[1f64, 2.0, 3.0]));
        let scalar = replicated_array(&session, &mesh, DataType::F32, &[], &values_to_bytes(&[1f32]));
        assert_eq!(
            tracker.live_array_bytes_by_device().unwrap(),
            Some(HashMap::from([(devices[0], 28), (devices[1], 28)])),
        );

        // An array that is created from a retained array is another logical array over the same device buffers, so it
        // adds a description without adding bytes.
        let restored = RetainedArray::new(&vector).to_array(&session).unwrap();
        let live_arrays = tracker.live_arrays().unwrap();
        assert_eq!(live_arrays.len(), 3);
        assert_eq!(live_arrays[2].addressable_size_in_bytes().unwrap(), 48);
        assert_eq!(
            tracker.live_array_bytes_by_device().unwrap(),
            Some(HashMap::from([(devices[0], 28), (devices[1], 28)])),
        );

        // The shared buffers stay counted until the last array that holds them is dropped.
        drop(vector);
        assert_eq!(
            tracker.live_array_bytes_by_device().unwrap(),
            Some(HashMap::from([(devices[0], 28), (devices[1], 28)])),
        );
        drop(restored);
        assert_eq!(
            tracker.live_array_bytes_by_device().unwrap(),
            Some(HashMap::from([(devices[0], 4), (devices[1], 4)])),
        );
        drop(scalar);
        assert_eq!(tracker.live_array_bytes_by_device().unwrap(), Some(HashMap::new()));
    }

    #[test]
    fn test_array_tracker_register() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let array = replicated_array(&session, &mesh, DataType::F32, &[2], &values_to_bytes(&[1f32, 2.0]));
        let r#type = Arc::new(array.r#type().into_owned());
        let sharding = Arc::new(array.sharding().clone());
        let mesh = Arc::new(mesh);

        // Every registration counts one created and one live array, and dropping it counts the array as released.
        let tracker = Arc::new(ArrayTracker::new(false));
        let first = tracker.register(&r#type, &sharding, &mesh, array.shards());
        assert_eq!(tracker.statistics(), ArrayStatistics { live_arrays: 1, created_arrays: 1 });
        let second = tracker.register(&r#type, &sharding, &mesh, array.shards());
        assert_eq!(tracker.statistics(), ArrayStatistics { live_arrays: 2, created_arrays: 2 });
        drop(first);
        assert_eq!(tracker.statistics(), ArrayStatistics { live_arrays: 1, created_arrays: 2 });
        drop(second);
        assert_eq!(tracker.statistics(), ArrayStatistics { live_arrays: 0, created_arrays: 2 });
        assert!(tracker.live_arrays().is_none());

        // A tracker that keeps descriptions records one per registration and removes it with the registration. The
        // registrations own the tracker, and the descriptions refer to neither the storage nor the buffers.
        let tracker = Arc::new(ArrayTracker::new(true));
        let registration = tracker.register(&r#type, &sharding, &mesh, array.shards());
        let live_arrays = tracker.live_arrays().unwrap();
        assert_eq!(live_arrays.len(), 1);
        assert_eq!(live_arrays[0].addressable_size_in_bytes().unwrap(), 8);
        assert_eq!(Arc::strong_count(&tracker), 2);
        assert!(array.has_unique_shard_buffers());
        drop(registration);
        assert!(tracker.live_arrays().unwrap().is_empty());
        assert_eq!(tracker.statistics(), ArrayStatistics { live_arrays: 0, created_arrays: 1 });
        assert_eq!(Arc::strong_count(&tracker), 1);
    }
}
