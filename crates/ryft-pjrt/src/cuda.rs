//! Adapts CUDA PJRT clients and XLA FFI resources for launching precompiled kernels with [`ryft_cuda`].
//!
//! This module is available with the `cuda-12` or `cuda-13` feature. [`Client::cuda_version`] reads the client's
//! required CUDA version, and [`Client::cuda_kernel_launcher`] creates a [`CudaKernelLauncher`] using a compatible
//! system driver. Artifact validation, argument packing, module caching, and CUDA driver calls remain in `ryft_cuda`.
//!
//! Inside an XLA FFI handler, [`FfiBuffer::cuda_kernel_argument`] borrows a buffer's device address and
//! [`FfiExecutionContext::cuda_kernel_launch`] combines ordered arguments with the invocation's CUDA stream. The
//! resulting [`CudaKernelLaunch`] can be submitted with [`CudaKernelLauncher::launch`] and a separately supplied
//! [`CudaKernelArtifact`](ryft_cuda::CudaKernelArtifact). Buffer shapes, strides, and scalar parameters must be
//! supplied explicitly when the artifact's Application Binary Interface (ABI) requires them; a buffer argument
//! contributes only its device address.
//!
//! # Ownership and Completion
//!
//! These adapters borrow resources owned by XLA/PJRT. They neither allocate device memory nor take ownership of
//! buffers, streams, or CUDA contexts. Constructing a launch frame does not submit work, and successful submission
//! does not establish device completion. The enclosing runtime must preserve input readiness, output dependencies,
//! and resource lifetimes through completion of the enqueued work. Rust borrows alone do not enforce these
//! asynchronous obligations.
//!
//! Keep the launcher available for reuse, then call [`CudaKernelLauncher::shutdown`] while the CUDA contexts it has
//! used are still alive and its cleanup requirements can be satisfied. Launcher construction does not retain the
//! [`Client`] or bind the launcher to one device. Each launch must satisfy the launcher's current-context, stream,
//! allocation, access, and graph-capture requirements.
//!
//! # Errors
//!
//! The adapters use [`ryft_cuda::Error`] so construction and submission share the launcher's error type. Failures
//! querying PJRT/XLA metadata become [`Error::Integration`]; invalid device pointers, unsupported stream handles, and
//! version encodings retain the errors produced by `ryft_cuda`. No adapter can establish that an arbitrary raw address
//! belongs to a live CUDA allocation or that an FFI context actually represents a CUDA invocation.

pub use ryft_cuda::{CudaKernelArgument, CudaKernelLaunch, CudaKernelLauncher, CudaStream, CudaVersion};

use ryft_cuda::{CudaDevicePointer, Error};

use crate::clients::Client;
use crate::extensions::ffi::{FfiBuffer, FfiExecutionContext};

impl Client<'_> {
    /// Returns the [`CudaVersion`] reported by this [`Client`]'s PJRT plugin. This is the client's required CUDA
    /// version and not a query of the installed driver or the device's compute capability. The platform name must be
    /// `cuda` (case-insensitive), and its version must have the form `cuda <encoded-version>`, where CUDA encodes
    /// versions as `1000 * major + 10 * minor`.
    ///
    /// Returns [`Error::Integration`] for a non-CUDA client, failed platform queries, or an unparseable version
    /// string. Parsed values are validated by [`CudaVersion::from_encoded`], which requires a supported encoding.
    pub fn cuda_version(&self) -> Result<CudaVersion, Error> {
        let platform_name = self
            .platform_name()
            .map_err(|error| Error::integration(format!("failed to query the PJRT client platform name: {error}")))?;

        // Reject other platforms before interpreting their unrelated version metadata as a CUDA requirement.
        if !platform_name.eq_ignore_ascii_case("cuda") {
            return Err(Error::integration(format!(
                "CUDA kernel launchers require a CUDA PJRT client, but the provided client uses platform \
                 `{platform_name}`",
            )));
        }

        let platform_version = self.platform_version().map_err(|error| {
            Error::integration(format!("failed to query the PJRT client platform version: {error}"))
        })?;

        // The plugin reports CUDA's integer encoding; the launcher owns its supported version validation.
        let encoded_version = platform_version
            .strip_prefix("cuda ")
            .and_then(|version| version.parse::<u32>().ok())
            .ok_or_else(|| {
                Error::integration(format!(
                    "invalid CUDA PJRT platform version `{platform_version}`; expected `cuda <encoded-version>`",
                ))
            })?;

        CudaVersion::from_encoded(encoded_version)
    }

    /// Creates a [`CudaKernelLauncher`] using the CUDA version reported by this [`Client`]. Loads a compatible system
    /// CUDA driver and creates an empty module cache with the launcher's default limits. The launcher does not retain
    /// the client or select a device; CUDA contexts and streams are supplied at launch time. Reuse the launcher across
    /// invocations and shut it down before destroying contexts whose modules it caches.
    ///
    /// Propagates errors from [`Self::cuda_version`] and [`CudaKernelLauncher::new`], including an unavailable or
    /// incompatible driver. No kernel is compiled, loaded, or submitted by this function.
    #[inline]
    pub fn cuda_kernel_launcher(&self) -> Result<CudaKernelLauncher, Error> {
        CudaKernelLauncher::new(self.cuda_version()?)
    }
}

impl<'o> FfiBuffer<'o> {
    /// Creates a [`CudaKernelArgument::DevicePointer`] borrowing this [`FfiBuffer`]'s device address. Does not copy
    /// data, extend the allocation's lifetime, or add shape/stride arguments. Returns [`Error::InvalidArgument`] when
    /// the data address is null, including for an empty buffer with a null address. The adapter cannot check allocation
    /// ownership, bounds, or access permissions from the raw address.
    ///
    /// # Safety
    ///
    /// The buffer must belong to a CUDA XLA FFI invocation. Its device allocation must remain live for `'o` and through
    /// completion of every enqueued kernel that accesses it. The caller must satisfy the memory access, aliasing,
    /// Application Binary Interface (ABI), and ordering requirements of [`CudaKernelLauncher::launch`].
    ///
    /// This function exposes XLA's raw device address to the launcher for interoperability. It is unsafe because
    /// the FFI buffer does not encode the CUDA allocation's ownership or asynchronous lifetime in its Rust type.
    #[inline]
    pub unsafe fn cuda_kernel_argument(&self) -> Result<CudaKernelArgument<'o>, Error> {
        // Wrap the borrowed address without copying or retaining the runtime-owned allocation.
        let pointer = unsafe { CudaDevicePointer::from_raw(self.data()) }?;
        Ok(CudaKernelArgument::DevicePointer(pointer))
    }
}

impl<'o> FfiExecutionContext<'o> {
    /// Creates a [`CudaKernelLaunch`] borrowing this [`FfiExecutionContext`]'s CUDA stream. The frame owns the ordered
    /// argument list and scalar values, while device addresses and the stream remain borrowed. This function only
    /// constructs the frame; it does not validate it against an artifact or submit work. Returns [`Error::Integration`]
    /// if stream discovery fails and [`Error::InvalidArgument`] for a null or unsupported default stream handle.
    ///
    /// # Parameters
    ///
    ///   - `arguments`: Flattened [`CudaKernelArgument`]s in the order required by the artifact's Application Binary
    ///     Interface (ABI). Include explicit scalar, shape, or stride arguments where required; device pointers must
    ///     satisfy their allocation/access contracts.
    ///
    /// # Safety
    ///
    /// This context must belong to a CUDA XLA FFI invocation. Its stream, owning CUDA context, and referenced
    /// allocations must remain live for `'o` and through completion of the enqueued work. The owning CUDA context
    /// must be current when required by [`CudaKernelLauncher::launch`], and the caller must satisfy that function's
    /// access, ordering, and graph capture requirements.
    ///
    /// This function exposes the runtime-owned raw stream to the launcher for interoperability. It is unsafe because
    /// the FFI handle does not prove that it is a CUDA stream or retain the external resources until GPU completion.
    #[inline]
    pub unsafe fn cuda_kernel_launch<A: Into<Box<[CudaKernelArgument<'o>]>>>(
        &self,
        arguments: A,
    ) -> Result<CudaKernelLaunch<'o>, Error> {
        // Use the invocation's stream so submission participates in the runtime's existing dependency ordering.
        let stream = self
            .stream()
            .map_err(|error| Error::integration(format!("failed to get the XLA FFI CUDA stream: {error}")))?;

        // Reject null/default handles without taking ownership of the external stream or its CUDA context.
        let stream = unsafe { CudaStream::from_raw(stream) }?;
        Ok(CudaKernelLaunch::new(stream, arguments))
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use pretty_assertions::assert_eq;
    use ryft_cuda::{CudaKernelParameterType, CudaScalarValue};

    use crate::extensions::ffi::FfiApi;
    use crate::extensions::ffi::buffers::ffi;
    use crate::extensions::ffi::context::ffi::XLA_FFI_Stream_Get_Args;
    use crate::extensions::ffi::errors::ffi::XLA_FFI_Error;
    use crate::extensions::ffi::tests::test_ffi_api;
    use crate::tests::{TestPlatform, test_for_each_platform};

    use super::*;

    /// Reads the synthetic stream handle stored behind a test execution context.
    unsafe extern "C" fn stream_get(arguments: *mut XLA_FFI_Stream_Get_Args) -> *mut XLA_FFI_Error {
        let arguments = unsafe { &mut *arguments };
        // Only this callback reads the synthetic context, which points to a live local stream handle variable.
        arguments.stream = unsafe { *arguments.context.cast::<*mut c_void>() };
        std::ptr::null_mut()
    }

    #[test]
    fn test_client_cuda_version() {
        test_for_each_platform!(|_plugin, client, platform| {
            match platform {
                TestPlatform::Cuda12 | TestPlatform::Cuda13 => {
                    let version = client.cuda_version().unwrap();
                    let expected_major = match platform {
                        TestPlatform::Cuda12 => 12,
                        TestPlatform::Cuda13 => 13,
                        _ => unreachable!(),
                    };
                    assert_eq!(version.major(), expected_major);
                    assert_eq!(client.platform_version().unwrap(), format!("cuda {}", version.encoded()));
                }
                _ => assert!(matches!(
                    client.cuda_version(),
                    Err(Error::Integration { message, .. })
                        if message == format!(
                            "CUDA kernel launchers require a CUDA PJRT client, but the provided client \
                             uses platform `{}`",
                            client.platform_name().unwrap(),
                        ),
                )),
            }
        });
    }

    #[test]
    fn test_client_cuda_kernel_launcher() {
        test_for_each_platform!(|_plugin, client, platform| {
            match platform {
                TestPlatform::Cuda12 | TestPlatform::Cuda13 => {
                    let mut launcher = client.cuda_kernel_launcher().unwrap();
                    assert_eq!(launcher.cuda_version(), client.cuda_version().unwrap());
                    // No work has been submitted, and the client still owns any external CUDA contexts.
                    assert_eq!(unsafe { launcher.shutdown() }, Ok(()));
                }
                _ => assert!(matches!(
                    client.cuda_kernel_launcher(),
                    Err(Error::Integration { message, .. })
                        if message == format!(
                            "CUDA kernel launchers require a CUDA PJRT client, but the provided client \
                             uses platform `{}`",
                            client.platform_name().unwrap(),
                        ),
                )),
            }
        });
    }

    #[test]
    fn test_ffi_buffer_cuda_kernel_argument() {
        // This descriptor exercises address adaptation only; no device pointer is dereferenced or submitted to CUDA.
        let mut raw_buffer = ffi::XLA_FFI_Buffer {
            struct_size: size_of::<ffi::XLA_FFI_Buffer>(),
            extension_start: std::ptr::null_mut(),
            data_type: ffi::XLA_FFI_DataType_S32,
            data: 0x1234usize as *mut c_void,
            rank: 0,
            dimensions: std::ptr::null(),
        };
        let buffer = unsafe { FfiBuffer::from_c_api(&raw_buffer) }.unwrap();
        assert_eq!(
            unsafe { buffer.cuda_kernel_argument() },
            Ok(CudaKernelArgument::DevicePointer(unsafe { CudaDevicePointer::from_raw(raw_buffer.data).unwrap() })),
        );
        assert_eq!(unsafe { buffer.cuda_kernel_argument() }.unwrap().r#type(), CudaKernelParameterType::DevicePointer);

        // An FFI descriptor can carry a null address, but the launch ABI requires non-null pointer arguments.
        raw_buffer.data = std::ptr::null_mut();
        let buffer = unsafe { FfiBuffer::from_c_api(&raw_buffer) }.unwrap();
        assert!(matches!(
            unsafe { buffer.cuda_kernel_argument() },
            Err(Error::InvalidArgument { message, .. }) if message == "CUDA kernel device pointer is a null pointer",
        ));
    }

    #[test]
    fn test_ffi_execution_context_cuda_kernel_launch() {
        // Preserve the runtime API for error handling, overriding only stream discovery in this local table.
        let runtime_api = test_ffi_api();
        let mut table = unsafe { std::ptr::read(runtime_api.to_c_api()) };
        table.XLA_FFI_Stream_Get = Some(stream_get);
        let api = unsafe { FfiApi::from_c_api(&table) }.unwrap();

        // The callback borrows this local handle; these frames are never submitted to a CUDA driver.
        let mut stream = 0x40usize as *mut c_void;
        let context_handle = (&mut stream as *mut *mut c_void).cast();
        let context = unsafe { FfiExecutionContext::from_c_api(context_handle, api) }.unwrap();
        let arguments = [
            CudaKernelArgument::Scalar(CudaScalarValue::U32(42)),
            CudaKernelArgument::Scalar(CudaScalarValue::I64(-7)),
        ];
        assert_eq!(context.stream(), Ok(stream));
        assert_eq!(unsafe { context.cuda_kernel_launch(arguments) }.unwrap().arguments(), &arguments);
        assert_eq!(unsafe { context.cuda_kernel_launch([]) }.unwrap().arguments(), &[]);

        // Invalid stream handles must fail during adaptation before a frame could reach the launcher.
        stream = std::ptr::null_mut();
        assert_eq!(context.stream(), Ok(stream));
        assert!(matches!(
            unsafe { context.cuda_kernel_launch(arguments) },
            Err(Error::InvalidArgument { message, .. }) if message == "CUDA stream handle is a null pointer",
        ));
        for default_stream in [1usize, 2] {
            stream = default_stream as *mut c_void;
            assert_eq!(context.stream(), Ok(stream));
            assert!(matches!(
                unsafe { context.cuda_kernel_launch(arguments) },
                Err(Error::InvalidArgument { message, .. }) if message == "CUDA default stream handles are unsupported",
            ));
        }

        // A missing FFI entry point remains an integration error instead of being mistaken for a null stream.
        table.XLA_FFI_Stream_Get = None;
        let api = unsafe { FfiApi::from_c_api(&table) }.unwrap();
        let context = unsafe { FfiExecutionContext::from_c_api(context_handle, api) }.unwrap();
        assert!(matches!(
            unsafe { context.cuda_kernel_launch(arguments) },
            Err(Error::Integration { message, .. })
                if message == format!(
                    "failed to get the XLA FFI CUDA stream: \
                     `XLA_FFI_Stream_Get` is not implemented in the loaded XLA FFI API (version {})",
                    api.version(),
                ),
        ));
    }
}
