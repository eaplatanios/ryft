//! Backend-neutral exchange of serialized compilation artifacts among the processes of one distributed run.

// TODO(eaplatanios): Review this module.

use std::time::Duration;

use thiserror::Error;

/// Failure reported by a [`CompilationArtifactExchange`] implementation.
#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum CompilationExchangeError {
    /// The exchange deadline elapsed before an artifact became available.
    #[error("compilation artifact exchange timed out")]
    TimedOut,

    /// Participating processes disagreed about the launch, compilation, backend, or topology identity.
    #[error("compilation artifact exchange is incompatible: {message}")]
    Incompatible {
        /// Human-readable lowercase description of the incompatibility.
        message: String,
    },

    /// The exchange transport or coordination protocol failed.
    #[error("compilation artifact exchange failed: {message}")]
    Failed {
        /// Human-readable lowercase description of the failure.
        message: String,
    },
}

/// Exchanges serialized compiled programs between processes participating in one distributed run.
///
/// Core treats keys and artifacts as opaque bytes. Backends must encode the complete compilation identity in keys,
/// using identical bytes for matching compilations across processes. They must serialize every piece of invocation
/// metadata and reject artifacts that are incompatible with the local runtime. Process zero is the leader that
/// compiles and publishes, and every other process is a follower that receives.
pub trait CompilationArtifactExchange: Send + Sync {
    /// Returns the zero-based index of this process.
    fn process_index(&self) -> usize;

    /// Returns the number of processes participating in the exchange.
    fn process_count(&self) -> usize;

    /// Verifies that this process is compatible with the leader for `key`, before the leader compiles the program or a
    /// follower waits for its artifact.
    ///
    /// A process calls this function only for compilations that miss its in-memory cache. Processes may therefore call
    /// it for different keys and in different orders, skip a key that another process compiles, or call it again for a
    /// key whose program they evicted. Implementations must address each round by `key` rather than by call order, and
    /// the leader must not wait for followers. Implementations should compare `key`, the process count, and any launch
    /// or backend compatibility identity that the transport owns. The default validates only the local process
    /// coordinates so that simple transports remain usable.
    fn preflight(&self, _key: &[u8], _timeout: Duration) -> Result<(), CompilationExchangeError> {
        if self.process_count() == 0 {
            return Err(CompilationExchangeError::Incompatible {
                message: "exchange process count must be greater than zero".to_string(),
            });
        }
        if self.process_index() >= self.process_count() {
            return Err(CompilationExchangeError::Incompatible {
                message: "exchange process index is outside the configured process count".to_string(),
            });
        }
        Ok(())
    }

    /// Publishes `artifact` for `key` so that follower processes can receive it.
    ///
    /// Publication must not wait for followers to receive the artifact, because a follower may never call
    /// [`Self::receive`] for `key`.
    fn publish(&self, key: &[u8], artifact: &[u8]) -> Result<(), CompilationExchangeError>;

    /// Publishes a terminal leader failure for `key` so that followers do not wait until their deadline.
    ///
    /// The leader calls this function whenever it cannot publish an artifact for `key`, which includes failed
    /// compilations as well as compiled programs that cannot be serialized. Transports that cannot represent terminal
    /// failures may keep the default no-op behavior, in which case followers follow their configured timeout policy.
    #[inline]
    fn publish_failure(&self, _key: &[u8], _message: &str) -> Result<(), CompilationExchangeError> {
        Ok(())
    }

    /// Waits up to `timeout` for an artifact published for `key`.
    ///
    /// `Ok(None)` means that no artifact became available before the deadline. Transport and protocol failures,
    /// including terminal failures published by the leader, are returned explicitly.
    ///
    /// A follower calls this function only when it cannot restore the artifact from a cheaper tier, so a follower whose
    /// persistent cache already holds the artifact never receives it. Implementations must therefore not require every
    /// follower to receive each published artifact, which rules out transports that implement [`Self::publish`] and
    /// this function as a collective operation such as a broadcast in which every process must take part.
    fn receive(&self, key: &[u8], timeout: Duration) -> Result<Option<Vec<u8>>, CompilationExchangeError>;
}

/// Policy controlling the distributed exchange of compiled artifacts.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum CompilationArtifactExchangePolicy {
    /// Never publish or receive compilation artifacts.
    #[default]
    Disabled,

    /// Prefer an exchanged artifact and optionally compile locally when the exchange is unavailable or fails.
    PreferSharing {
        /// Maximum time that followers wait for the leader.
        timeout: Duration,

        /// Whether a follower may compile locally after an exchange miss or failure.
        fallback_to_local_compile: bool,
    },

    /// Require a compatible exchanged artifact and fail instead of compiling on follower processes.
    RequireSharing {
        /// Maximum time that followers wait for the leader.
        timeout: Duration,
    },
}

impl CompilationArtifactExchangePolicy {
    /// Returns the follower deadline, or [`None`] when this policy disables the exchange.
    #[inline]
    pub(crate) fn timeout(self) -> Option<Duration> {
        match self {
            Self::Disabled => None,
            Self::PreferSharing { timeout, .. } | Self::RequireSharing { timeout } => Some(timeout),
        }
    }

    /// Returns whether a process may compile locally after the exchange could not supply or publish an artifact.
    #[inline]
    pub(crate) fn permits_local_fallback(self) -> bool {
        match self {
            Self::Disabled => true,
            Self::PreferSharing { fallback_to_local_compile, .. } => fallback_to_local_compile,
            Self::RequireSharing { .. } => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compilation_exchange_error_display() {
        assert_eq!(CompilationExchangeError::TimedOut.to_string(), "compilation artifact exchange timed out");
        assert_eq!(
            CompilationExchangeError::Incompatible { message: "launch identities differ".to_string() }.to_string(),
            "compilation artifact exchange is incompatible: launch identities differ",
        );
        assert_eq!(
            CompilationExchangeError::Failed { message: "connection reset".to_string() }.to_string(),
            "compilation artifact exchange failed: connection reset",
        );
    }

    #[test]
    fn test_compilation_artifact_exchange_preflight() {
        struct TestExchange {
            process_index: usize,
            process_count: usize,
        }

        impl CompilationArtifactExchange for TestExchange {
            fn process_index(&self) -> usize {
                self.process_index
            }

            fn process_count(&self) -> usize {
                self.process_count
            }

            fn publish(&self, _key: &[u8], _artifact: &[u8]) -> Result<(), CompilationExchangeError> {
                Ok(())
            }

            fn receive(&self, _key: &[u8], _timeout: Duration) -> Result<Option<Vec<u8>>, CompilationExchangeError> {
                Ok(None)
            }
        }

        let timeout = Duration::from_secs(1);
        assert_eq!(TestExchange { process_index: 1, process_count: 2 }.preflight(b"key", timeout), Ok(()));
        assert_eq!(
            TestExchange { process_index: 0, process_count: 0 }.preflight(b"key", timeout),
            Err(CompilationExchangeError::Incompatible {
                message: "exchange process count must be greater than zero".to_string(),
            }),
        );
        assert_eq!(
            TestExchange { process_index: 2, process_count: 2 }.preflight(b"key", timeout),
            Err(CompilationExchangeError::Incompatible {
                message: "exchange process index is outside the configured process count".to_string(),
            }),
        );
    }

    #[test]
    fn test_compilation_artifact_exchange_policy_timeout() {
        let timeout = Duration::from_millis(10);
        assert_eq!(CompilationArtifactExchangePolicy::Disabled.timeout(), None);
        assert_eq!(
            CompilationArtifactExchangePolicy::PreferSharing { timeout, fallback_to_local_compile: false }.timeout(),
            Some(timeout),
        );
        assert_eq!(CompilationArtifactExchangePolicy::RequireSharing { timeout }.timeout(), Some(timeout));
    }

    #[test]
    fn test_compilation_artifact_exchange_policy_permits_local_fallback() {
        let timeout = Duration::from_millis(10);
        assert!(CompilationArtifactExchangePolicy::Disabled.permits_local_fallback());
        assert!(
            CompilationArtifactExchangePolicy::PreferSharing { timeout, fallback_to_local_compile: true }
                .permits_local_fallback()
        );
        assert!(
            !CompilationArtifactExchangePolicy::PreferSharing { timeout, fallback_to_local_compile: false }
                .permits_local_fallback()
        );
        assert!(!CompilationArtifactExchangePolicy::RequireSharing { timeout }.permits_local_fallback());
    }
}
