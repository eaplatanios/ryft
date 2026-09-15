//! Explicitly borrowed primary-context streams and pointer-only launch frames.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::Error;

/// Non-null device allocation borrowed from the embedding runtime.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RocmDevicePointer<'o> {
    /// Native device address.
    pointer: NonNull<c_void>,

    /// Lifetime of the external allocation owner.
    owner: PhantomData<&'o c_void>,
}

impl<'o> RocmDevicePointer<'o> {
    /// Borrows an externally owned HIP device address.
    ///
    /// # Safety
    /// `pointer` must remain allocated on the launch device through asynchronous execution, including when the host
    /// launch frame is dropped. Its bounds, permissions and aliases must satisfy the kernel. Raw addresses are
    /// exposed for interoperability with embedding runtimes that own HIP buffers.
    pub unsafe fn from_raw(pointer: *mut c_void) -> Result<Self, Error> {
        Ok(Self {
            pointer: NonNull::new(pointer).ok_or_else(|| Error::invalid_argument("rocm device pointer is null"))?,
            owner: PhantomData,
        })
    }

    /// Returns the borrowed address for native pointer argument packing.
    pub(crate) fn as_raw(self) -> *mut c_void {
        self.pointer.as_ptr()
    }
}

/// Explicit non-default HIP stream owned by its device's primary context.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RocmStream<'o> {
    /// Native stream handle.
    pointer: NonNull<c_void>,

    /// Lifetime of the external stream owner.
    owner: PhantomData<&'o c_void>,
}

impl<'o> RocmStream<'o> {
    /// Borrows a primary-context HIP stream from an embedding runtime.
    ///
    /// # Safety
    /// The handle must be a valid explicit HIP stream associated with the device's primary context and remain alive
    /// through all submitted work. The caller owns ordering and completion. This function exposes the native handle
    /// because framework integrations supply their own streams; null and special default stream handles are rejected.
    pub unsafe fn from_raw(pointer: *mut c_void) -> Result<Self, Error> {
        if (pointer as usize) <= 2 {
            return Err(Error::invalid_argument("rocm stream must be an explicit non-default stream"));
        }
        Ok(Self { pointer: NonNull::new(pointer).unwrap(), owner: PhantomData })
    }

    /// Returns the borrowed native stream handle.
    pub(crate) fn as_raw(self) -> *mut c_void {
        self.pointer.as_ptr()
    }
}

/// Pointer values copied into one asynchronous native submission.
#[derive(Clone, Debug)]
pub struct RocmKernelLaunch<'o> {
    /// Stream receiving the invocation.
    stream: RocmStream<'o>,

    /// Physical arguments in compiler-established order.
    arguments: Vec<RocmDevicePointer<'o>>,
}

impl<'o> RocmKernelLaunch<'o> {
    /// Creates a pointer-only frame; artifact-dependent arity is checked before submission.
    pub fn new(stream: RocmStream<'o>, arguments: impl IntoIterator<Item = RocmDevicePointer<'o>>) -> Self {
        Self { stream, arguments: arguments.into_iter().collect() }
    }

    /// Returns the borrowed invocation stream.
    pub fn stream(&self) -> RocmStream<'o> {
        self.stream
    }

    /// Returns physical arguments in their native order.
    pub fn arguments(&self) -> &[RocmDevicePointer<'o>] {
        &self.arguments
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rocm_device_pointer_from_raw() {
        let pointer = std::ptr::without_provenance_mut(16);
        assert_eq!(unsafe { RocmDevicePointer::from_raw(pointer) }.unwrap().as_raw(), pointer);
        assert_eq!(
            unsafe { RocmDevicePointer::from_raw(std::ptr::null_mut()) },
            Err(Error::invalid_argument("rocm device pointer is null")),
        );
    }

    #[test]
    fn test_rocm_stream_from_raw() {
        for address in 0..=2 {
            assert_eq!(
                unsafe { RocmStream::from_raw(std::ptr::without_provenance_mut(address)) },
                Err(Error::invalid_argument("rocm stream must be an explicit non-default stream")),
            );
        }
        let pointer = std::ptr::without_provenance_mut(16);
        assert_eq!(unsafe { RocmStream::from_raw(pointer) }.unwrap().as_raw(), pointer);
    }

    #[test]
    fn test_rocm_kernel_launch_new() {
        let stream = unsafe { RocmStream::from_raw(std::ptr::without_provenance_mut(16)) }.unwrap();
        let argument = unsafe { RocmDevicePointer::from_raw(std::ptr::without_provenance_mut(32)) }.unwrap();
        let launch = RocmKernelLaunch::new(stream, [argument]);
        assert_eq!(launch.stream(), stream);
        assert_eq!(launch.arguments(), &[argument]);
    }

    #[test]
    fn test_rocm_kernel_launch_stream() {
        let stream = unsafe { RocmStream::from_raw(std::ptr::without_provenance_mut(16)) }.unwrap();
        assert_eq!(RocmKernelLaunch::new(stream, []).stream(), stream);
    }

    #[test]
    fn test_rocm_kernel_launch_arguments() {
        let stream = unsafe { RocmStream::from_raw(std::ptr::without_provenance_mut(16)) }.unwrap();
        let first = unsafe { RocmDevicePointer::from_raw(std::ptr::without_provenance_mut(32)) }.unwrap();
        let second = unsafe { RocmDevicePointer::from_raw(std::ptr::without_provenance_mut(64)) }.unwrap();
        assert_eq!(RocmKernelLaunch::new(stream, [first, second]).arguments(), &[first, second]);
    }
}
