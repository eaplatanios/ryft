//! Multi-process integration tests for the compiled cross-host reshard path.
//!
//! These tests verify that the compiled cross-mesh path of [`Array::to_placement`](ryft_xla::Array::to_placement)
//! correctly uses the PJRT cross-host transfers extension when the destination spans multiple processes.
//!
//! The PJRT CPU plugin shipped with this workspace does not expose the cross-host transfers extension, so the tests are
//! `#[ignore]`d by default and serve as a template for backends that do (e.g., GPU or TPU backends). To run them
//! locally against a plugin that exposes the extension:
//!
//! 1. Make `ryft-xla` use a PJRT plugin whose `cross_host_transfers_extension()` returns
//!    `Ok(_)`.
//! 2. Run `cargo test -p ryft-xla --test cross_host_reshard -- --ignored`.
//!
//! Each test spawns its own test binary, which it locates with [`std::env::current_exe`], as peer processes with the
//! `RYFT_CROSS_HOST_WORKER_RANK` environment variable set to their rank. A test that observes this variable runs as a
//! worker, and otherwise it runs as the coordinator.
//!
//! The harness exists primarily as a working template. When a future XLA backend exposes the extension on the
//! developer's machine, implementing the worker and pointing PJRT at the right plugin should be sufficient to execute
//! the scenarios end to end.

use std::env;
use std::process::Command;
use std::time::Duration;

/// Environment variable that holds the rank of a worker process.
const WORKER_RANK_VARIABLE: &str = "RYFT_CROSS_HOST_WORKER_RANK";

/// Duration after which the coordinator kills a worker that has not exited.
const TIMEOUT: Duration = Duration::from_secs(30);

/// Returns the rank of this process if it was spawned as a worker.
fn run_as_worker() -> Option<i32> {
    env::var(WORKER_RANK_VARIABLE).ok().and_then(|rank| rank.parse::<i32>().ok())
}

/// Entry point reached when this test binary is executed again with `RYFT_CROSS_HOST_WORKER_RANK` set, which returns
/// the exit status of the worker. In the real flow, each worker would:
///
///   1. Connect to a [`ryft_pjrt::DistributedRuntimeClient`] keyed by `rank`.
///   2. Construct a PJRT [`Client`](ryft_pjrt::Client) with the distributed configuration.
///   3. Build an [`Array`](ryft_xla::Array) holding only its locally addressable shards.
///   4. Call [`Array::to_placement`](ryft_xla::Array::to_placement) with a multi-process target mesh.
///   5. Verify the resulting addressable shards with `copy_to_host`.
///   6. Exit with status 0 on success and a non-zero status on failure.
fn worker_main(rank: i32) -> i32 {
    eprintln!("[worker {rank}] cross-host reshard worker — not implemented for the CPU plugin");
    // This is a placeholder, and a real worker would do the steps above. It returns a non-zero status so that any
    // accidental run on a backend without the cross-host extension produces a visible failure rather than a silent
    // pass.
    1
}

/// Spawns `world_size` workers by executing this test binary again with each rank in [`WORKER_RANK_VARIABLE`].
/// Every worker runs only the ignored test named `test_name`, which is the test that spawns it, so that it reaches
/// [`worker_main`]. Without these arguments, the test harness of the worker would skip every ignored test and exit
/// successfully without doing any work.
fn spawn_workers(world_size: i32, test_name: &str) -> std::io::Result<Vec<std::process::Child>> {
    let self_path = env::current_exe()?;
    (0..world_size)
        .map(|rank| {
            Command::new(&self_path)
                .args(["--ignored", "--exact", test_name])
                .env(WORKER_RANK_VARIABLE, rank.to_string())
                .spawn()
        })
        .collect()
}

/// Waits for every worker, killing those that do not exit within [`TIMEOUT`], and returns their exit statuses.
fn await_workers(mut children: Vec<std::process::Child>) -> Vec<std::process::ExitStatus> {
    let mut statuses = Vec::with_capacity(children.len());
    for child in children.iter_mut() {
        child.wait_timeout_or_kill(TIMEOUT).expect("worker wait failed");
    }
    for mut child in children {
        statuses.push(child.wait().expect("worker wait failed"));
    }
    statuses
}

/// Tiny `Child::wait_timeout` polyfill that doesn't pull in an extra dependency. Tests using it are `#[ignore]`d, so
/// the loop overhead doesn't matter.
trait WaitTimeoutOrKill {
    /// Polls this process until it exits or `timeout` elapses, and kills it in the latter case.
    fn wait_timeout_or_kill(&mut self, timeout: Duration) -> std::io::Result<()>;
}

impl WaitTimeoutOrKill for std::process::Child {
    fn wait_timeout_or_kill(&mut self, timeout: Duration) -> std::io::Result<()> {
        let start = std::time::Instant::now();
        while start.elapsed() < timeout {
            if self.try_wait()?.is_some() {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        self.kill()
    }
}

#[test]
#[ignore = "requires a PJRT plugin that exposes the cross-host transfers extension"]
fn test_cross_host_replicated_broadcast() {
    // Worker mode short-circuits to the worker entry point. The coordinator, which runs without the environment
    // variable, spawns two workers and asserts that all of them exit successfully.
    if let Some(rank) = run_as_worker() {
        std::process::exit(worker_main(rank));
    }

    let children = spawn_workers(2, "test_cross_host_replicated_broadcast").expect("spawn workers");
    let statuses = await_workers(children);
    for (rank, status) in statuses.iter().enumerate() {
        assert!(status.success(), "worker {rank} exited with {status:?}");
    }
}

#[test]
#[ignore = "requires a PJRT plugin that exposes the cross-host transfers extension"]
fn test_cross_host_sharded_reshard() {
    if let Some(rank) = run_as_worker() {
        std::process::exit(worker_main(rank));
    }

    let children = spawn_workers(2, "test_cross_host_sharded_reshard").expect("spawn workers");
    let statuses = await_workers(children);
    for (rank, status) in statuses.iter().enumerate() {
        assert!(status.success(), "worker {rank} exited with {status:?}");
    }
}
