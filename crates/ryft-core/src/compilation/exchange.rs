//! Backend-neutral exchange of serialized compilation artifacts.

use std::fmt::{Display, Formatter};
use std::time::Duration;

/// Exchanges serialized compiled programs between processes participating in one distributed run.
///
/// Core treats keys and artifacts as opaque bytes. Backends remain responsible for producing a complete stable key,
/// serializing every piece of invocation metadata, and rejecting artifacts that are incompatible with the local
/// runtime.
pub trait CompilationArtifactExchange: Send + Sync {
    /// Returns the zero-based index of this process.
    fn process_index(&self) -> usize;

    /// Returns the number of processes participating in the exchange.
    fn process_count(&self) -> usize;

    /// Verifies that every process is participating in the same compilation round.
    ///
    /// Implementations should compare `key`, process count, and any transport-owned launch or backend compatibility
    /// identity before allowing the producer to compile or followers to wait for an artifact. The default validates
    /// only the local process coordinates so simple transports remain usable.
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

    /// Publishes `artifact` for `key` so follower processes can receive it.
    fn publish(&self, key: &[u8], artifact: &[u8]) -> Result<(), CompilationExchangeError>;

    /// Publishes a terminal producer failure so followers do not wait until their deadline.
    ///
    /// Transports that cannot represent terminal failures may keep the default no-op behavior; followers then follow
    /// their configured timeout policy.
    #[inline]
    fn publish_failure(&self, _key: &[u8], _message: &str) -> Result<(), CompilationExchangeError> {
        Ok(())
    }

    /// Waits up to `timeout` for an artifact published for `key`.
    ///
    /// `Ok(None)` means no artifact became available before the deadline. Transport and protocol failures are returned
    /// explicitly.
    fn receive(&self, key: &[u8], timeout: Duration) -> Result<Option<Vec<u8>>, CompilationExchangeError>;
}

/// Policy controlling distributed compiled-artifact exchange.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum CompilationArtifactExchangePolicy {
    /// Never publish or receive compilation artifacts.
    #[default]
    Disabled,

    /// Prefer an exchanged artifact and optionally compile locally when exchange is unavailable or fails.
    PreferSharing {
        /// Maximum time followers wait for the producer.
        timeout: Duration,

        /// Whether a follower may compile locally after an exchange miss or failure.
        fallback_to_local_compile: bool,
    },

    /// Require a compatible exchanged artifact and fail instead of compiling on follower processes.
    RequireSharing {
        /// Maximum time followers wait for the producer.
        timeout: Duration,
    },
}

impl CompilationArtifactExchangePolicy {
    #[inline]
    pub(crate) fn timeout(self) -> Option<Duration> {
        match self {
            Self::Disabled => None,
            Self::PreferSharing { timeout, .. } | Self::RequireSharing { timeout } => Some(timeout),
        }
    }

    #[inline]
    pub(crate) fn permits_local_fallback(self) -> bool {
        match self {
            Self::Disabled => true,
            Self::PreferSharing { fallback_to_local_compile, .. } => fallback_to_local_compile,
            Self::RequireSharing { .. } => false,
        }
    }
}

/// Failure reported by a compiled-artifact exchange implementation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompilationExchangeError {
    /// The exchange deadline elapsed before an artifact became available.
    TimedOut,

    /// Participating processes disagreed about the launch, compilation, backend, or topology identity.
    Incompatible {
        /// Human-readable lower-case description of the incompatibility.
        message: String,
    },

    /// The exchange transport or coordination protocol failed.
    Failed {
        /// Human-readable lower-case description of the failure.
        message: String,
    },
}

impl Display for CompilationExchangeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TimedOut => write!(formatter, "compilation artifact exchange timed out"),
            Self::Incompatible { message } => {
                write!(formatter, "compilation artifact exchange is incompatible: {message}")
            }
            Self::Failed { message } => write!(formatter, "compilation artifact exchange failed: {message}"),
        }
    }
}

impl std::error::Error for CompilationExchangeError {}
