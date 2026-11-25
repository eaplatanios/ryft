//! Results of operations that run on background threads, mirroring the `AsyncResponse` of the Orbax v1 API.
//!
//! [`save_async`](crate::saving::save_async) returns an [`AsyncResponse`] as soon as the save runs on its own thread,
//! so that training can continue while the checkpoint is written. [`AsyncResponse::result`] waits for the result,
//! optionally for a bounded time. Unlike Orbax, which copies device arrays to the host before it returns and only
//! writes them in the background, the conversion of leaves to host arrays also runs on the background thread, so the
//! saved state must be [`Send`] and should own its data (e.g., host arrays, which share their storage cheaply when
//! cloned).

use std::any::Any;
use std::sync::mpsc::{Receiver, RecvTimeoutError, sync_channel};
use std::thread::{Builder, JoinHandle};
use std::time::Duration;

use crate::errors::CheckpointError;

// TODO(eaplatanios): Review this module.

/// Result of an operation that runs on a background thread, which mirrors the `AsyncResponse` of Orbax. Dropping a
/// response waits for its operation, so that a save is never cut short when the process exits, and an error of that
/// operation is then lost unless [`Self::result`] returned it first.
#[derive(Debug)]
pub struct AsyncResponse<T> {
    /// Receiver of the result of the operation, which the background thread sends exactly once.
    receiver: Receiver<Result<T, CheckpointError>>,

    /// Background thread of the operation, until its result was returned.
    thread: Option<JoinHandle<()>>,
}

impl<T: 'static + Send> AsyncResponse<T> {
    /// Starts `operation` on a new background thread named `name` and returns its response.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::Io`] with `path` if the thread cannot be started.
    pub(crate) fn spawn<F: 'static + Send + FnOnce() -> Result<T, CheckpointError>>(
        name: &str,
        path: &str,
        operation: F,
    ) -> Result<Self, CheckpointError> {
        let (sender, receiver) = sync_channel(1);
        let thread = Builder::new()
            .name(name.to_string())
            .spawn(move || {
                // The receiver may be gone when the response was dropped, and then the result has no caller to go to.
                sender.send(operation()).ok();
            })
            .map_err(|error| CheckpointError::Io { path: path.to_string(), message: error.to_string() })?;
        Ok(Self { receiver, thread: Some(thread) })
    }
}

impl<T> AsyncResponse<T> {
    /// Waits for the operation to finish and returns its result, which mirrors `AsyncResponse.result` of Orbax. With
    /// a `timeout`, waits at most that long and returns [`CheckpointError::Timeout`] if the operation is still running,
    /// in which case the result can be asked for again. The result of the operation is returned once, and later calls
    /// return [`CheckpointError::InvalidArgument`].
    ///
    /// # Errors
    ///
    /// Returns the error of the operation, [`CheckpointError::Panic`] if its thread panicked,
    /// [`CheckpointError::Timeout`] if it did not finish within `timeout`, and [`CheckpointError::InvalidArgument`] if
    /// its result was already returned.
    pub fn result(&mut self, timeout: Option<Duration>) -> Result<T, CheckpointError> {
        if self.thread.is_none() {
            return Err(CheckpointError::InvalidArgument {
                message: "the result of this asynchronous operation was already returned".to_string(),
            });
        }
        let received = match timeout {
            Some(timeout) => self.receiver.recv_timeout(timeout),
            None => self.receiver.recv().map_err(|_| RecvTimeoutError::Disconnected),
        };
        match received {
            Ok(result) => {
                // The thread sent its result as its last action, so joining it does not wait for anything else.
                self.thread.take().unwrap().join().ok();
                result
            }
            Err(RecvTimeoutError::Timeout) => Err(CheckpointError::Timeout {
                message: format!("the asynchronous operation did not finish within {timeout:?}"),
            }),
            Err(RecvTimeoutError::Disconnected) => {
                let message = match self.thread.take().unwrap().join() {
                    Ok(()) => "the background thread ended without a result".to_string(),
                    Err(payload) => panic_message(payload),
                };
                Err(CheckpointError::Panic { message })
            }
        }
    }
}

impl<T> Drop for AsyncResponse<T> {
    fn drop(&mut self) {
        // Joining keeps the operation from being cut short when the process exits. Its result has no caller to go to.
        if let Some(thread) = self.thread.take() {
            thread.join().ok();
        }
    }
}

/// Returns the message of the panic whose payload is `payload`.
pub(crate) fn panic_message(payload: Box<dyn Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|message| message.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic payload".to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_async_response_result() {
        let mut response = AsyncResponse::spawn("test", "path", || Ok(3)).unwrap();
        assert_eq!(response.result(None), Ok(3));
        assert_eq!(
            response.result(None),
            Err(CheckpointError::InvalidArgument {
                message: "the result of this asynchronous operation was already returned".to_string(),
            }),
        );

        // Errors and panics of the operation are returned once.
        let mut failing = AsyncResponse::<()>::spawn("test", "path", || {
            Err(CheckpointError::InvalidArgument { message: "failed".to_string() })
        })
        .unwrap();
        assert_eq!(failing.result(None), Err(CheckpointError::InvalidArgument { message: "failed".to_string() }));
        let mut panicking = AsyncResponse::<()>::spawn("test", "path", || panic!("broken")).unwrap();
        assert_eq!(panicking.result(None), Err(CheckpointError::Panic { message: "broken".to_string() }));
    }

    #[test]
    fn test_async_response_result_timeout() {
        // An operation that is still running times out, and its result can be asked for again.
        let barrier = Arc::new(Barrier::new(2));
        let mut response = {
            let barrier = barrier.clone();
            AsyncResponse::spawn("test", "path", move || {
                barrier.wait();
                Ok("done")
            })
            .unwrap()
        };
        assert!(matches!(
            response.result(Some(Duration::from_millis(10))),
            Err(CheckpointError::Timeout { message })
                if message.starts_with("the asynchronous operation did not finish"),
        ));
        barrier.wait();
        assert_eq!(response.result(Some(Duration::from_secs(60))), Ok("done"));
    }
}
