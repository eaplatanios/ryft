//! Leaves on XLA devices, which is available with the `xla` feature.
//!
//! [`XlaLeafHandler`] is the [`LeafHandler`] for parameter structures whose leaves are [`ryft_xla::Array`]s. With it in
//! a [`Context`](crate::contexts::Context), a [`ParameterizedHandler`](crate::handlers::ParameterizedHandler) loads
//! every leaf of a checkpoint directly into device buffers, with the placement that the caller selects for it, and
//! reads the leaves of a structure back from their devices when it saves them. Handlers convert one leaf at a time, so
//! a load holds at most one leaf in host memory while the rest of the structure already lives on its devices, and a
//! save does the same in the other direction. This is the counterpart of the per leaf sharding of the abstract arrays
//! that Orbax loads into. The handler borrows its session, so it saves and loads synchronously only.
//!
//! [`XlaProcessBarrier`] synchronizes the processes of a distributed XLA runtime while they save one checkpoint
//! together (refer to [`MultiprocessingOptions`](crate::multiprocessing::MultiprocessingOptions)), through the key
//! value store of the coordination service of that runtime. Every process counts how often it has waited at each
//! key, and the processes meet at equal counts. A save that fails in one process can therefore leave the counts of
//! the processes apart, so the job should end after a failed save instead of retrying it (refer to the
//! [`multiprocessing`](crate::multiprocessing) module).

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::sync::Mutex;
use std::time::Duration;

use ryft_core::{Array as HostArray, ArrayType, DeviceMesh, ParameterPath, Sharding, Typed};
use ryft_pjrt::KeyValueStore;
use ryft_xla::{Array, DistributedRuntime, Session};

use crate::errors::CheckpointError;
use crate::handlers::LeafHandler;
use crate::multiprocessing::ProcessBarrier;

// TODO(eaplatanios): Review this module.

/// Function that selects the runtime [`Sharding`] of a loaded leaf from its [`ParameterPath`] and the type of its
/// host array, or [`None`] for a leaf that every device of the mesh holds completely.
type ShardingSelector<'s> = dyn Fn(&ParameterPath, &ArrayType) -> Option<Sharding> + 's;

/// [`LeafHandler`] for structures whose leaves are [`ryft_xla::Array`]s on the devices of one [`DeviceMesh`].
/// Loaded leaves are uploaded with [`Array::from_host_buffer`], replicated over the mesh unless a sharding is selected
/// for them with [`Self::with_shardings`], and saved leaves are read back with [`Array::copy_to_host`], which is a
/// synchronization boundary for the leaf. Leaves are stored as dense host arrays without their placement, so a
/// checkpoint can be loaded onto another mesh or backend.
pub struct XlaLeafHandler<'s> {
    /// Session whose devices hold the leaves.
    session: &'s Session,

    /// Mesh that loaded leaves are placed on.
    mesh: DeviceMesh,

    /// Selector of the runtime sharding of every loaded leaf.
    shardings: Box<ShardingSelector<'s>>,
}

impl<'s> XlaLeafHandler<'s> {
    /// Creates a new [`XlaLeafHandler`] that loads every leaf replicated over `mesh`.
    ///
    /// # Parameters
    ///
    ///   - `session`: Session whose devices hold the leaves.
    ///   - `mesh`: Mesh that loaded leaves are placed on.
    #[inline]
    pub fn new(session: &'s Session, mesh: DeviceMesh) -> Self {
        Self { session, mesh, shardings: Box::new(|_, _| None) }
    }

    /// Returns this handler with `shardings` selecting the runtime [`Sharding`] over the mesh of every loaded leaf
    /// from its [`ParameterPath`] and the type of its host array (e.g., to split the leading dimension of large
    /// projections across devices). A leaf for which it returns [`None`] is replicated.
    #[inline]
    pub fn with_shardings<F: Fn(&ParameterPath, &ArrayType) -> Option<Sharding> + 's>(mut self, shardings: F) -> Self {
        self.shardings = Box::new(shardings);
        self
    }

    /// Returns the session whose devices hold the leaves.
    #[inline]
    pub fn session(&self) -> &'s Session {
        self.session
    }

    /// Returns the mesh that loaded leaves are placed on.
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        &self.mesh
    }
}

impl LeafHandler for XlaLeafHandler<'_> {
    type Leaf = Array;

    #[inline]
    fn r#type(&self, path: &ParameterPath, leaf: &Array) -> Result<ArrayType, CheckpointError> {
        let _ = path;
        Ok(ArrayType::new(leaf.data_type(), leaf.shape().into()))
    }

    fn serialize<'l>(&self, path: &ParameterPath, leaf: &'l Array) -> Result<Cow<'l, HostArray>, CheckpointError> {
        let array = leaf.copy_to_host().map_err(|error| leaf_error(path, error))?;
        Ok(Cow::Owned(array))
    }

    fn deserialize(&self, path: &ParameterPath, array: HostArray) -> Result<Array, CheckpointError> {
        let host_type = array.r#type();
        let sharding = (self.shardings)(path, &host_type);
        let r#type = host_type.as_ref().clone().replicated(&self.mesh).map_err(|error| leaf_error(path, error))?;
        Array::from_host_buffer(self.session, r#type, self.mesh.clone(), sharding, array.storage_bytes())
            .map_err(|error| leaf_error(path, error))
    }
}

/// Timeout of an [`XlaProcessBarrier`] wait for which the caller provides none, because the key value store of the
/// distributed runtime only waits for a bounded time.
const DEFAULT_BARRIER_TIMEOUT: Duration = Duration::from_secs(24 * 60 * 60);

/// [`ProcessBarrier`] across the processes of a [`DistributedRuntime`], which uses the key value store of the
/// coordination service of that runtime. A process that reaches the barrier announces itself under a key that names
/// the barrier, the round, and its process index, and then waits until every process of the runtime has announced
/// itself for the same round. The store never overwrites a key, so every wait on the same barrier key is a new round,
/// which this barrier counts per key. Processes therefore stay synchronized as long as each of them waits on the
/// same keys the same number of times, which is the case for processes that save the same checkpoints.
pub struct XlaProcessBarrier {
    /// Runtime whose key value store the processes announce themselves in.
    runtime: DistributedRuntime,

    /// Index of this process.
    process_index: usize,

    /// Number of processes of the runtime.
    process_count: usize,

    /// Number of completed waits per barrier key.
    rounds: Mutex<HashMap<String, u64>>,
}

impl XlaProcessBarrier {
    /// Creates a new [`XlaProcessBarrier`].
    ///
    /// # Parameters
    ///
    ///   - `runtime`: Distributed runtime that every participating process has initialized.
    ///   - `process_index`: Index of this process in the runtime.
    ///   - `process_count`: Number of processes of the runtime.
    #[inline]
    pub fn new(runtime: DistributedRuntime, process_index: usize, process_count: usize) -> Self {
        Self { runtime, process_index, process_count, rounds: Mutex::new(HashMap::new()) }
    }

    /// Returns the index of this process.
    #[inline]
    pub fn process_index(&self) -> usize {
        self.process_index
    }

    /// Returns the number of processes of the runtime.
    #[inline]
    pub fn process_count(&self) -> usize {
        self.process_count
    }
}

impl Debug for XlaProcessBarrier {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("XlaProcessBarrier")
            .field("process_index", &self.process_index)
            .field("process_count", &self.process_count)
            .finish_non_exhaustive()
    }
}

impl ProcessBarrier for XlaProcessBarrier {
    fn wait(&self, key: &str, timeout: Option<Duration>) -> Result<(), CheckpointError> {
        let barrier_error = |message: String| CheckpointError::Barrier { key: key.to_string(), message };
        let round = {
            let mut rounds =
                self.rounds.lock().map_err(|_| barrier_error("the barrier lock was poisoned".to_string()))?;
            let round = rounds.entry(key.to_string()).or_insert(0);
            *round += 1;
            *round
        };
        let store = self.runtime.key_value_store();
        let announcement = |process_index: usize| format!("ryft-checkpoints/barrier/{key}/{round}/{process_index}");
        store
            .put(announcement(self.process_index).as_bytes(), b"reached")
            .map_err(|error| barrier_error(error.to_string()))?;
        let timeout = timeout.unwrap_or(DEFAULT_BARRIER_TIMEOUT);
        for process_index in 0..self.process_count {
            store.get(announcement(process_index).as_bytes(), timeout).map_err(|error| {
                barrier_error(format!("process {process_index} did not reach the barrier: {error}"))
            })?;
        }
        Ok(())
    }
}

/// Returns the [`CheckpointError`] for a failed conversion of the leaf at `path`.
fn leaf_error<E: std::fmt::Display>(path: &ParameterPath, error: E) -> CheckpointError {
    CheckpointError::Leaf { path: path.to_string(), message: error.to_string() }
}

#[cfg(all(test, feature = "safetensors"))]
mod tests {
    use std::net::TcpListener;
    use std::sync::Arc;

    use pretty_assertions::assert_eq;
    use ryft_core::{DataType, MeshAxis, MeshAxisType, Parameterized};
    use ryft_pjrt::load_cpu_plugin;
    use ryft_xla::Platform;

    use crate::contexts::Context;
    use crate::loading::load;
    use crate::multiprocessing::MultiprocessingOptions;
    use crate::safetensors::SafetensorsFormat;
    use crate::saving::{SaveOptions, save};

    use super::*;

    #[test]
    fn test_xla_leaf_handler() {
        let session = Session::builder().platform(Platform::Cpu).build().unwrap();
        let mesh = session.mesh([MeshAxis::new("device", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let matrix = HostArray::matrix(2, 2, vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let vector = HostArray::vector(vec![5i32, 6, 7]).unwrap();
        let host = (matrix.clone(), vec![vector.clone()]);

        // Loaded leaves are uploaded with the sharding that is selected for their path, and saved leaves are read back
        // from the device, so saving the device structure and loading it reproduces the host arrays.
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 2);
        let leaf_handler = XlaLeafHandler::new(&session, mesh.clone()).with_shardings({
            let sharding = sharding.clone();
            move |path, r#type| (path.to_string() == "$.0" && r#type.rank() == 2).then(|| sharding.clone())
        });
        assert_eq!(leaf_handler.mesh(), &mesh);
        let device = host.clone().try_map_named_parameters(|path, leaf| leaf_handler.deserialize(path, leaf)).unwrap();
        assert_eq!(device.0.sharding(), &sharding);
        let context = Context::new(leaf_handler, Arc::new(SafetensorsFormat));
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("checkpoint");
        save(&path, &device, &SaveOptions::default(), &context).unwrap();
        let abstract_state =
            (ArrayType::new_static(DataType::F32, [2, 2]), vec![ArrayType::new_static(DataType::I32, [3])]);
        let loaded = load(&path, abstract_state, &context).unwrap();
        assert_eq!(loaded.0.copy_to_host().unwrap(), matrix);
        assert_eq!(loaded.1[0].copy_to_host().unwrap(), vector);
        assert_eq!(loaded.0.sharding(), &sharding);

        // A leaf whose type the abstract state does not match is rejected before anything is uploaded.
        let mismatched = (ArrayType::new_static(DataType::F32, [4]), vec![ArrayType::new_static(DataType::I32, [3])]);
        assert!(load(&path, mismatched, &context).is_err());
    }

    #[test]
    fn test_xla_process_barrier() {
        // A runtime with one process is its own coordinator, so its barrier returns as soon as the process has
        // announced itself, and every wait on the same key is a new round. The port is assigned by the operating
        // system, and the test has nothing to check on a machine without a loopback interface.
        let Ok(listener) = TcpListener::bind("127.0.0.1:0") else {
            return;
        };
        let address = format!("127.0.0.1:{}", listener.local_addr().unwrap().port());
        drop(listener);
        let plugin = load_cpu_plugin().unwrap();
        let runtime = DistributedRuntime::initialize(&plugin, &address, 1, 0).unwrap();
        let barrier = XlaProcessBarrier::new(runtime.clone(), 0, 1);
        assert_eq!((barrier.process_index(), barrier.process_count()), (0, 1));
        assert_eq!(barrier.wait("save_start", Some(Duration::from_secs(10))), Ok(()));
        assert_eq!(barrier.wait("save_start", Some(Duration::from_secs(10))), Ok(()));

        // A barrier that expects a process that never arrives fails after its timeout instead of waiting forever.
        let barrier = XlaProcessBarrier::new(runtime, 0, 2);
        assert!(matches!(
            barrier.wait("commit", Some(Duration::from_millis(200))),
            Err(CheckpointError::Barrier { key, message })
                if key == "commit" && message.starts_with("process 1 did not reach the barrier"),
        ));

        // The barrier plugs into the multi-process options of a context.
        let options = MultiprocessingOptions::new(0, 1).unwrap().with_barrier(Arc::new(barrier));
        assert!(options.is_primary());
    }
}
