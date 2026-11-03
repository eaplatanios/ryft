use std::time::Duration;

use thiserror::Error;

/// Failure reported by a [`CompilationArtifactExchange`] implementation.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum CompilationArtifactExchangeError {
    #[error("compilation artifact exchange timed out")]
    TimedOut,

    #[error("compilation artifact exchange is incompatible: {message}")]
    Incompatible { message: String },

    #[error("compilation artifact exchange failed: {message}")]
    Failed { message: String },
}

/// Exchanges serialized compiled programs between processes participating in a distributed job. Core treats keys and
/// artifacts as opaque bytes. Backends must encode the complete compilation identity in keys, using identical bytes
/// for matching compilations across processes. They must serialize every piece of invocation metadata and reject
/// artifacts that are incompatible with the local runtime. Process zero is the leader that compiles and publishes,
/// and every other process is a follower that receives.
pub trait CompilationArtifactExchange: Send + Sync {
    /// Returns the zero-based index of this process.
    fn process_index(&self) -> usize;

    /// Returns the number of processes participating in the exchange.
    fn process_count(&self) -> usize;

    /// Validates this process's compatibility with the leader for `key` before the leader compiles the program or a
    /// follower waits for its artifact. Followers may wait up to `timeout` for the leader's compatibility announcement.
    /// A process calls this function only for compilations that miss its in-memory cache. Processes may therefore call
    /// it for different keys and in different orders, skip a key that another process compiles, or call it again for a
    /// key whose program they evicted. Implementations must address each round by `key` rather than by call order, and
    /// the leader must not wait for followers. Implementations should compare `key`, the process count, and any launch
    /// or backend compatibility identity that the transport owns. The default validates only the local process
    /// coordinates so that simple transports remain usable.
    fn validate_compatibility(&self, key: &[u8], timeout: Duration) -> Result<(), CompilationArtifactExchangeError> {
        let _ = key;
        let _ = timeout;
        if self.process_count() == 0 {
            return Err(CompilationArtifactExchangeError::Incompatible {
                message: "exchange process count must be greater than zero".to_string(),
            });
        }
        if self.process_index() >= self.process_count() {
            return Err(CompilationArtifactExchangeError::Incompatible {
                message: "exchange process index is outside the configured process count".to_string(),
            });
        }
        Ok(())
    }

    /// Publishes `artifact` for `key` so that follower processes can receive it. Publication must not wait for
    /// followers to receive the artifact, because a follower may never call [`Self::receive`] for `key`.
    fn publish(&self, key: &[u8], artifact: &[u8]) -> Result<(), CompilationArtifactExchangeError>;

    /// Publishes a terminal leader failure for `key` so that followers do not wait until their deadline. The leader
    /// calls this function whenever it cannot publish an artifact for `key`, which includes failed compilations as well
    /// as compiled programs that cannot be serialized. Transports that cannot represent terminal failures may keep the
    /// default no-op behavior, in which case followers follow their configured timeout policy.
    #[inline]
    fn publish_failure(&self, key: &[u8], message: &str) -> Result<(), CompilationArtifactExchangeError> {
        let _ = key;
        let _ = message;
        Ok(())
    }

    /// Waits up to `timeout` to receive an artifact published for `key`. `Ok(None)` means that no artifact became
    /// available before the deadline. Transport and protocol failures, including terminal failures published by the
    /// leader, are returned explicitly. A follower calls this function only when it cannot restore the artifact from
    /// a cheaper tier, so a follower whose persistent cache already holds the artifact never receives it.
    /// Implementations must therefore not require every follower to receive each published artifact, which rules out
    /// transports that implement [`Self::publish`] and this function as a collective operation such as a broadcast in
    /// which every process must take part.
    fn receive(&self, key: &[u8], timeout: Duration) -> Result<Option<Vec<u8>>, CompilationArtifactExchangeError>;
}

/// Policy controlling the distributed exchange of compiled artifacts.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum CompilationArtifactExchangePolicy {
    /// Never publish or receive compilation artifacts.
    #[default]
    Disabled,

    /// Prefer an exchanged artifact and allow local compilation when the exchange is unavailable or fails.
    /// Leaders may keep their compiled programs when serialization or publication fails.
    PreferSharing {
        /// Maximum time that followers wait for the leader.
        timeout: Duration,
    },

    /// Require a compatible exchanged artifact on followers and fail on exchange errors, including serialization
    /// or publication failures on the leader.
    RequireSharing {
        /// Maximum time that followers wait for the leader.
        timeout: Duration,
    },
}

impl CompilationArtifactExchangePolicy {
    /// Returns the follower deadline, or [`None`] when this policy disables the exchange.
    #[inline]
    pub fn timeout(self) -> Option<Duration> {
        match self {
            Self::Disabled => None,
            Self::PreferSharing { timeout } | Self::RequireSharing { timeout } => Some(timeout),
        }
    }

    /// Returns whether a process may compile locally after the exchange could not supply or publish an artifact.
    #[inline]
    pub fn permits_local_fallback(self) -> bool {
        match self {
            Self::Disabled | Self::PreferSharing { .. } => true,
            Self::RequireSharing { .. } => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    /// Exchange with fixed process coordinates that implements only the required functions, so tests observe the
    /// default implementations of the optional ones.
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

        fn publish(&self, _key: &[u8], _artifact: &[u8]) -> Result<(), CompilationArtifactExchangeError> {
            Ok(())
        }

        fn receive(
            &self,
            _key: &[u8],
            _timeout: Duration,
        ) -> Result<Option<Vec<u8>>, CompilationArtifactExchangeError> {
            Ok(None)
        }
    }

    #[test]
    fn test_compilation_artifact_exchange_error_display() {
        for (error, message) in [
            (CompilationArtifactExchangeError::TimedOut, "compilation artifact exchange timed out"),
            (
                CompilationArtifactExchangeError::Incompatible { message: "launch identities differ".to_string() },
                "compilation artifact exchange is incompatible: launch identities differ",
            ),
            (
                CompilationArtifactExchangeError::Failed { message: "connection reset".to_string() },
                "compilation artifact exchange failed: connection reset",
            ),
        ] {
            let diagnostic: &dyn std::error::Error = &error;
            assert_eq!(diagnostic.to_string(), message);
            assert!(diagnostic.source().is_none());
        }
    }

    #[test]
    fn test_compilation_artifact_exchange_validate_compatibility() {
        // The default checks only the process coordinates, so it accepts any key and timeout.
        assert_eq!(
            TestExchange { process_index: 0, process_count: 1 }.validate_compatibility(b"", Duration::ZERO),
            Ok(())
        );
        assert_eq!(
            TestExchange { process_index: 1, process_count: 2 }.validate_compatibility(b"key", Duration::from_secs(1)),
            Ok(()),
        );

        // An empty exchange reports its process count before the index that it cannot contain.
        assert_eq!(
            TestExchange { process_index: 0, process_count: 0 }.validate_compatibility(b"key", Duration::from_secs(1)),
            Err(CompilationArtifactExchangeError::Incompatible {
                message: "exchange process count must be greater than zero".to_string(),
            }),
        );
        assert_eq!(
            TestExchange { process_index: 2, process_count: 2 }.validate_compatibility(b"key", Duration::from_secs(1)),
            Err(CompilationArtifactExchangeError::Incompatible {
                message: "exchange process index is outside the configured process count".to_string(),
            }),
        );
    }

    #[test]
    fn test_compilation_artifact_exchange_publish_failure() {
        // Transports that cannot represent terminal failures keep the default, which accepts every notification.
        let exchange = TestExchange { process_index: 0, process_count: 2 };
        assert_eq!(exchange.publish_failure(b"key", "backend compilation failed"), Ok(()));
    }

    #[test]
    fn test_compilation_artifact_exchange_policy_default() {
        assert_eq!(CompilationArtifactExchangePolicy::default(), CompilationArtifactExchangePolicy::Disabled);
    }

    #[test]
    fn test_compilation_artifact_exchange_policy_timeout() {
        let timeout = Duration::from_millis(10);
        assert_eq!(CompilationArtifactExchangePolicy::Disabled.timeout(), None);
        assert_eq!(CompilationArtifactExchangePolicy::PreferSharing { timeout }.timeout(), Some(timeout));
        assert_eq!(CompilationArtifactExchangePolicy::RequireSharing { timeout }.timeout(), Some(timeout));
    }

    #[test]
    fn test_compilation_artifact_exchange_policy_permits_local_fallback() {
        let timeout = Duration::from_millis(10);
        assert!(CompilationArtifactExchangePolicy::Disabled.permits_local_fallback());
        assert!(CompilationArtifactExchangePolicy::PreferSharing { timeout }.permits_local_fallback());
        assert!(!CompilationArtifactExchangePolicy::RequireSharing { timeout }.permits_local_fallback());
    }
}
