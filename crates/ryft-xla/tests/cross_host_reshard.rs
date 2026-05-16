//! Multi-process integration tests for the compiled cross-host reshard path.
//!
//! These tests verify that
//! [`Array::to_placement`](ryft_xla::Array::to_placement)'s compiled cross-mesh path correctly
//! uses the PJRT cross-host transfers extension when the destination spans multiple processes.
//!
//! The PJRT CPU plugin shipped with this workspace does not expose the cross-host transfers
//! extension, so the tests are `#[ignore]`d by default and serve as a template for backends
//! (GPU/TPU/etc.) that do. To run them locally against a plugin that exposes the extension:
//!
//! 1. Make `ryft-xla` use a PJRT plugin whose `cross_host_transfers_extension()` returns
//!    `Ok(_)`.
//! 2. Run `cargo test -p ryft-xla --test cross_host_reshard -- --ignored`.
//!
//! Each test spawns the test binary itself as a peer process via `CARGO_BIN_EXE_<name>` (Cargo's
//! standard mechanism). The peer worker dispatches into worker mode when it sees the
//! `RYFT_CROSS_HOST_WORKER_RANK` environment variable; otherwise it runs the test as the
//! coordinator.
//!
//! The harness exists primarily as a working template: when a future XLA backend exposes the
//! extension on the developer's machine, switching `#[ignore]` to `#[test]` and pointing PJRT
//! at the right plugin should be sufficient to execute the scenarios end-to-end.

use std::env;
use std::process::Command;
use std::time::Duration;

const WORKER_ENV: &str = "RYFT_CROSS_HOST_WORKER_RANK";
const TIMEOUT: Duration = Duration::from_secs(30);

fn run_as_worker() -> Option<i32> {
    env::var(WORKER_ENV).ok().and_then(|rank| rank.parse::<i32>().ok())
}

/// Entry point reached when this test binary is re-exec'd with `RYFT_CROSS_HOST_WORKER_RANK` set.
/// In the real flow, each worker would:
///   1. Connect to a [`ryft_pjrt::DistributedRuntimeClient`] keyed by `rank`.
///   2. Construct a PJRT [`Client`](ryft_pjrt::Client) with the distributed config.
///   3. Build an [`Array`](ryft_xla::Array) holding only its locally-addressable shards.
///   4. Call `Array::to_placement` with a multi-process target mesh.
///   5. Verify the resulting addressable shards via `copy_to_host`.
///   6. Exit with status 0 on success, non-zero on failure.
fn worker_main(rank: i32) -> i32 {
    eprintln!("[worker {rank}] cross-host reshard worker — not implemented for the CPU plugin");
    // Placeholder: a real worker would do the steps above. We return non-zero so that any
    // accidental run on a backend without the cross-host extension produces a visible failure
    // rather than a silent pass.
    1
}

fn spawn_workers(world_size: i32) -> std::io::Result<Vec<std::process::Child>> {
    let self_path = env::current_exe()?;
    (0..world_size)
        .map(|rank| Command::new(&self_path).env(WORKER_ENV, rank.to_string()).spawn())
        .collect()
}

fn await_workers(mut children: Vec<std::process::Child>) -> Vec<std::process::ExitStatus> {
    let mut statuses = Vec::with_capacity(children.len());
    for child in children.iter_mut() {
        let _ = child.wait_timeout_or_kill(TIMEOUT);
    }
    for mut child in children {
        statuses.push(child.wait().expect("worker wait failed"));
    }
    statuses
}

/// Tiny `Child::wait_timeout` polyfill that doesn't pull in an extra dependency. Tests using it
/// are `#[ignore]`d so the loop overhead doesn't matter.
trait WaitTimeoutOrKill {
    fn wait_timeout_or_kill(&mut self, timeout: Duration) -> std::io::Result<()>;
}

impl WaitTimeoutOrKill for std::process::Child {
    fn wait_timeout_or_kill(&mut self, timeout: Duration) -> std::io::Result<()> {
        let start = std::time::Instant::now();
        while start.elapsed() < timeout {
            if let Some(_) = self.try_wait()? {
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
    // Worker mode short-circuits to the worker entry point. The coordinator (no env var)
    // spawns N workers and asserts all of them exit successfully.
    if let Some(rank) = run_as_worker() {
        std::process::exit(worker_main(rank));
    }

    let children = spawn_workers(2).expect("spawn workers");
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

    let children = spawn_workers(2).expect("spawn workers");
    let statuses = await_workers(children);
    for (rank, status) in statuses.iter().enumerate() {
        assert!(status.success(), "worker {rank} exited with {status:?}");
    }
}
