pub use ryft_cuda::{CudaError, CudaKernelArgument, CudaKernelLaunch, CudaKernelLauncher, CudaStream, CudaVersion};

use crate::clients::Client;
use crate::extensions::ffi::{FfiBuffer, FfiExecutionContext};

impl Client<'_> {
    /// Returns the CUDA version reported by this [`Client`].
    pub fn cuda_version(&self) -> Result<CudaVersion, CudaError> {
        let platform_name = self.platform_name().map_err(|error| {
            CudaError::integration(format!("failed to query the PJRT client platform name: {error}"))
        })?;
        let platform_version = self.platform_version().map_err(|error| {
            CudaError::integration(format!("failed to query the PJRT client platform version: {error}"))
        })?;
        if !platform_name.eq_ignore_ascii_case("cuda") {
            return Err(CudaError::integration(format!(
                "CUDA kernel launchers require a CUDA PJRT client, but the provided client uses platform \
                 `{platform_name}`",
            )));
        }
        let encoded = platform_version
            .strip_prefix("cuda ")
            .and_then(|version| version.parse::<u32>().ok())
            .ok_or_else(|| {
                CudaError::integration(format!(
                    "invalid CUDA PJRT platform version `{platform_version}`; expected `cuda <encoded-version>`",
                ))
            })?;
        CudaVersion::from_encoded(encoded)
    }

    /// Creates a new [`CudaKernelLauncher`] configured for this [`Client`].
    #[inline]
    pub fn cuda_kernel_launcher(&self) -> Result<CudaKernelLauncher, CudaError> {
        CudaKernelLauncher::new(self.cuda_version()?)
    }
}

impl<'o> FfiBuffer<'o> {
    /// Creates a [`CudaKernelArgument`] borrowing this [`FfiBuffer`].
    ///
    /// # Safety
    ///
    /// The buffer must belong to a CUDA XLA FFI invocation, and its device allocation must remain live for `'o`.
    /// XLA FFI exposes the allocation as a raw address, so this adapter is provided for interoperability with
    /// [`CudaKernelLauncher`].
    #[inline]
    pub unsafe fn cuda_kernel_argument(&self) -> Result<CudaKernelArgument<'o>, CudaError> {
        let pointer = unsafe { CudaDevicePointer::from_raw(self.data()) }?;
        Ok(CudaKernelArgument::device_pointer(pointer))
    }
}

impl<'o> FfiExecutionContext<'o> {
    /// Creates a [`CudaKernelLaunch`] borrowing the stream of the current XLA FFI invocation.
    ///
    /// # Safety
    ///
    /// This execution context must belong to a CUDA XLA FFI invocation, and its stream and CUDA context must remain
    /// live for `'o`. XLA FFI exposes the stream as a raw handle, so this adapter is provided for interoperability
    /// with [`CudaKernelLauncher`].
    #[inline]
    pub unsafe fn cuda_kernel_launch<A: Into<Box<[CudaKernelArgument<'o>]>>>(
        &self,
        arguments: A,
    ) -> Result<CudaKernelLaunch<'o>, CudaError> {
        let stream = self
            .stream()
            .map_err(|error| CudaError::integration(format!("failed to get the XLA FFI CUDA stream: {error}")))?;
        let stream = unsafe { CudaStream::from_raw(stream) }?;
        Ok(CudaKernelLaunch::new(stream, arguments))
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use crate::extensions::ffi::buffers::ffi;
    use crate::tests::{TestPlatform, test_for_each_platform};

    use super::*;

    #[test]
    fn test_client_cuda_version() {
        test_for_each_platform!(|_plugin, client, platform| {
            match platform {
                TestPlatform::Cuda12 => assert_eq!(client.cuda_version().unwrap().major(), 12),
                TestPlatform::Cuda13 => assert_eq!(client.cuda_version().unwrap().major(), 13),
                _ => assert!(matches!(client.cuda_version(), Err(CudaError::Integration { .. }))),
            }
        });
    }

    #[test]
    fn test_client_cuda_kernel_launcher() {
        test_for_each_platform!(|_plugin, client, platform| {
            match platform {
                TestPlatform::Cuda12 | TestPlatform::Cuda13 => {
                    let mut launcher = client.cuda_kernel_launcher().unwrap();
                    unsafe { launcher.shutdown() }.unwrap();
                }
                _ => assert!(matches!(client.cuda_kernel_launcher(), Err(CudaError::Integration { .. }),)),
            }
        });
    }

    #[test]
    fn test_ffi_buffer_cuda_kernel_argument() {
        let raw_buffer = ffi::XLA_FFI_Buffer {
            struct_size: size_of::<ffi::XLA_FFI_Buffer>(),
            extension_start: std::ptr::null_mut(),
            data_type: ffi::XLA_FFI_DataType_S32,
            data: 0x1234usize as *mut c_void,
            rank: 0,
            dimensions: std::ptr::null(),
        };
        let buffer = unsafe { FfiBuffer::from_c_api(&raw_buffer) }.unwrap();
        assert!(matches!(unsafe { buffer.cuda_kernel_argument() }, Ok(CudaKernelArgument::DevicePointer(_)),));
    }
}
