//! Borrowed CUDA resources, typed execution arguments, and native argument packing.

// TODO(eaplatanios): Review this.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::{CudaKernelArtifact, CudaKernelParameterType, CudaScalarType, Error};

/// Scalar value supplied in one CUDA kernel launch frame.
///
/// Low-precision values carry their exact storage bits so launch packing never performs a numeric conversion or
/// canonicalizes NaN encodings.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum CudaScalarValue {
    /// Signed 8-bit integer value.
    I8(i8),

    /// Signed 16-bit integer value.
    I16(i16),

    /// Signed 32-bit integer value.
    I32(i32),

    /// Signed 64-bit integer value.
    I64(i64),

    /// Unsigned 8-bit integer value.
    U8(u8),

    /// Unsigned 16-bit integer value.
    U16(u16),

    /// Unsigned 32-bit integer value.
    U32(u32),

    /// Unsigned 64-bit integer value.
    U64(u64),

    /// Exact storage byte of a [`CudaScalarType::F4E2M1FN`] value.
    F4E2M1FN(u8),

    /// Exact storage byte of a [`CudaScalarType::F6E2M3FN`] value.
    F6E2M3FN(u8),

    /// Exact storage byte of a [`CudaScalarType::F6E3M2FN`] value.
    F6E3M2FN(u8),

    /// Exact storage bits of a [`CudaScalarType::F8E3M4`] value.
    F8E3M4(u8),

    /// Exact storage bits of a [`CudaScalarType::F8E4M3`] value.
    F8E4M3(u8),

    /// Exact storage bits of a [`CudaScalarType::F8E4M3FN`] value.
    F8E4M3FN(u8),

    /// Exact storage bits of a [`CudaScalarType::F8E4M3FNUZ`] value.
    F8E4M3FNUZ(u8),

    /// Exact storage bits of a [`CudaScalarType::F8E4M3B11FNUZ`] value.
    F8E4M3B11FNUZ(u8),

    /// Exact storage bits of a [`CudaScalarType::F8E5M2`] value.
    F8E5M2(u8),

    /// Exact storage bits of a [`CudaScalarType::F8E5M2FNUZ`] value.
    F8E5M2FNUZ(u8),

    /// Exact storage bits of a [`CudaScalarType::F8E8M0FNU`] value.
    F8E8M0FNU(u8),

    /// Exact storage bits of a [`CudaScalarType::BF16`] value.
    BF16(u16),

    /// Exact storage bits of a [`CudaScalarType::F16`] value.
    F16(u16),

    /// IEEE 32-bit floating-point value.
    F32(f32),

    /// IEEE 64-bit floating-point value.
    F64(f64),
}

impl CudaScalarValue {
    /// Returns the immutable ABI type represented by this scalar value.
    pub fn r#type(self) -> CudaScalarType {
        match self {
            Self::I8(_) => CudaScalarType::I8,
            Self::I16(_) => CudaScalarType::I16,
            Self::I32(_) => CudaScalarType::I32,
            Self::I64(_) => CudaScalarType::I64,
            Self::U8(_) => CudaScalarType::U8,
            Self::U16(_) => CudaScalarType::U16,
            Self::U32(_) => CudaScalarType::U32,
            Self::U64(_) => CudaScalarType::U64,
            Self::F4E2M1FN(_) => CudaScalarType::F4E2M1FN,
            Self::F6E2M3FN(_) => CudaScalarType::F6E2M3FN,
            Self::F6E3M2FN(_) => CudaScalarType::F6E3M2FN,
            Self::F8E3M4(_) => CudaScalarType::F8E3M4,
            Self::F8E4M3(_) => CudaScalarType::F8E4M3,
            Self::F8E4M3FN(_) => CudaScalarType::F8E4M3FN,
            Self::F8E4M3FNUZ(_) => CudaScalarType::F8E4M3FNUZ,
            Self::F8E4M3B11FNUZ(_) => CudaScalarType::F8E4M3B11FNUZ,
            Self::F8E5M2(_) => CudaScalarType::F8E5M2,
            Self::F8E5M2FNUZ(_) => CudaScalarType::F8E5M2FNUZ,
            Self::F8E8M0FNU(_) => CudaScalarType::F8E8M0FNU,
            Self::BF16(_) => CudaScalarType::BF16,
            Self::F16(_) => CudaScalarType::F16,
            Self::F32(_) => CudaScalarType::F32,
            Self::F64(_) => CudaScalarType::F64,
        }
    }
}

/// Borrowed, non-null CUDA device pointer supplied by an external runtime.
///
/// Optional null pointer parameters are not represented by this API.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CudaDevicePointer<'o> {
    /// Non-null CUDA device address.
    pointer: NonNull<c_void>,

    /// Tracks the lifetime of the external allocation that owns the address.
    owner: PhantomData<&'o c_void>,
}

impl<'o> CudaDevicePointer<'o> {
    /// Borrows a raw CUDA device address owned by an external runtime.
    ///
    /// # Safety
    ///
    /// `pointer` must be a valid CUDA device address for `'o` and for every launch that receives the returned value.
    /// The allocation must remain valid through asynchronous execution, beyond the lifetime of the host launch frame.
    /// Access bounds, permissions, aliasing, and stream ordering must satisfy the safety contract of
    /// [`CudaKernelLauncher::launch`](crate::CudaKernelLauncher::launch). This function is exposed because CUDA
    /// framework integrations represent device memory using raw addresses.
    pub unsafe fn from_raw(pointer: *mut c_void) -> Result<Self, Error> {
        let pointer = NonNull::new(pointer)
            .ok_or_else(|| Error::invalid_argument("CUDA kernel device pointer is a null pointer"))?;
        Ok(Self { pointer, owner: PhantomData })
    }

    /// Returns the borrowed CUDA device address for argument packing.
    pub(crate) fn as_raw(self) -> *mut c_void {
        self.pointer.as_ptr()
    }
}

/// One per-execution CUDA kernel argument.
///
/// Only non-null device addresses and the listed scalar types are supported; by-value aggregates require a separate
/// ABI.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum CudaKernelArgument<'o> {
    /// Device address borrowed from an external runtime.
    DevicePointer(CudaDevicePointer<'o>),

    /// Host scalar copied into the launch frame.
    Scalar(CudaScalarValue),
}

impl CudaKernelArgument<'_> {
    /// Returns the immutable ABI type represented by this argument.
    pub fn r#type(self) -> CudaKernelParameterType {
        match self {
            Self::DevicePointer(_) => CudaKernelParameterType::DevicePointer,
            Self::Scalar(value) => CudaKernelParameterType::Scalar(value.r#type()),
        }
    }
}

/// Borrowed, explicitly created CUDA stream supplied by an external runtime.
///
/// Null/default, legacy-default, and per-thread-default stream handles are unsupported.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CudaStream<'o> {
    /// Non-null CUDA stream handle.
    handle: NonNull<c_void>,

    /// Tracks the lifetime of the external runtime that owns the stream.
    owner: PhantomData<&'o c_void>,
}

impl<'o> CudaStream<'o> {
    /// Borrows a raw CUDA stream handle owned by an external runtime.
    ///
    /// # Safety
    ///
    /// `handle` must identify a live CUDA stream for `'o`. Its owning CUDA context must remain live and current while
    /// the stream is used. This function is exposed because CUDA framework integrations represent streams as raw
    /// handles.
    pub unsafe fn from_raw(handle: *mut c_void) -> Result<Self, Error> {
        if matches!(handle.addr(), 1 | 2) {
            return Err(Error::invalid_argument("CUDA default stream handles are unsupported"));
        }
        let handle =
            NonNull::new(handle).ok_or_else(|| Error::invalid_argument("CUDA stream handle is a null pointer"))?;
        Ok(Self { handle, owner: PhantomData })
    }

    /// Returns the borrowed CUDA stream handle for driver calls.
    pub(crate) fn as_raw(self) -> *mut c_void {
        self.handle.as_ptr()
    }
}

/// Per-execution CUDA kernel launch state.
///
/// This frame owns host parameter storage, but it does not retain external device allocations until GPU completion.
/// [`CudaKernelLauncher::launch`](crate::CudaKernelLauncher::launch) enqueues asynchronously; callers retain resources
/// and arrange synchronization before reading results or releasing allocations.
pub struct CudaKernelLaunch<'o> {
    /// Externally owned CUDA stream on which the launch must be enqueued.
    stream: CudaStream<'o>,

    /// Per-execution device pointers and scalars in flattened ABI order.
    arguments: Box<[CudaKernelArgument<'o>]>,
}

impl<'o> CudaKernelLaunch<'o> {
    /// Creates a launch frame from an externally owned CUDA stream and its ordered arguments.
    pub fn new<A: Into<Box<[CudaKernelArgument<'o>]>>>(stream: CudaStream<'o>, arguments: A) -> Self {
        Self { stream, arguments: arguments.into() }
    }

    /// Returns the per-execution kernel arguments.
    pub fn arguments(&self) -> &[CudaKernelArgument<'o>] {
        self.arguments.as_ref()
    }

    /// Returns the externally owned CUDA stream stored in this launch frame.
    pub(super) fn stream(&self) -> *mut c_void {
        self.stream.as_raw()
    }
}

/// Validates one launch frame against the immutable ABI declared by its artifact.
pub(super) fn validate_launch(artifact: &CudaKernelArtifact, launch: &CudaKernelLaunch<'_>) -> Result<(), Error> {
    let expected = artifact.abi().parameters();
    if expected.len() != launch.arguments.len() {
        return Err(Error::invalid_argument(format!(
            "CUDA kernel `{}` expects {} parameters but the launch frame contains {}",
            artifact.symbol(),
            expected.len(),
            launch.arguments.len(),
        )));
    }
    for (index, (expected, argument)) in expected.iter().zip(launch.arguments.iter()).enumerate() {
        let actual = argument.r#type();
        if *expected != actual {
            return Err(Error::invalid_argument(format!(
                "CUDA kernel `{}` parameter {index} expects `{expected:?}` but received `{actual:?}`",
                artifact.symbol(),
            )));
        }
    }
    Ok(())
}

/// Properly aligned native parameter storage borrowed only for the duration of a driver enqueue call.
pub(super) enum CudaKernelArgumentStorage {
    DevicePointer(*mut c_void),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F4E2M1FN(u8),
    F6E2M3FN(u8),
    F6E3M2FN(u8),
    F8E3M4(u8),
    F8E4M3(u8),
    F8E4M3FN(u8),
    F8E4M3FNUZ(u8),
    F8E4M3B11FNUZ(u8),
    F8E5M2(u8),
    F8E5M2FNUZ(u8),
    F8E8M0FNU(u8),
    BF16(u16),
    F16(u16),
    F32(f32),
    F64(f64),
}

impl CudaKernelArgumentStorage {
    /// Packs one typed argument into the storage representation expected by `cuLaunchKernel`.
    pub(super) fn from_argument(argument: &CudaKernelArgument<'_>) -> Self {
        match argument {
            CudaKernelArgument::DevicePointer(pointer) => Self::DevicePointer(pointer.as_raw()),
            CudaKernelArgument::Scalar(CudaScalarValue::I8(value)) => Self::I8(*value),
            CudaKernelArgument::Scalar(CudaScalarValue::I16(value)) => Self::I16(*value),
            CudaKernelArgument::Scalar(CudaScalarValue::I32(value)) => Self::I32(*value),
            CudaKernelArgument::Scalar(CudaScalarValue::I64(value)) => Self::I64(*value),
            CudaKernelArgument::Scalar(CudaScalarValue::U8(value)) => Self::U8(*value),
            CudaKernelArgument::Scalar(CudaScalarValue::U16(value)) => Self::U16(*value),
            CudaKernelArgument::Scalar(CudaScalarValue::U32(value)) => Self::U32(*value),
            CudaKernelArgument::Scalar(CudaScalarValue::U64(value)) => Self::U64(*value),
            CudaKernelArgument::Scalar(CudaScalarValue::F4E2M1FN(bits)) => Self::F4E2M1FN(*bits),
            CudaKernelArgument::Scalar(CudaScalarValue::F6E2M3FN(bits)) => Self::F6E2M3FN(*bits),
            CudaKernelArgument::Scalar(CudaScalarValue::F6E3M2FN(bits)) => Self::F6E3M2FN(*bits),
            CudaKernelArgument::Scalar(CudaScalarValue::F8E3M4(bits)) => Self::F8E3M4(*bits),
            CudaKernelArgument::Scalar(CudaScalarValue::F8E4M3(bits)) => Self::F8E4M3(*bits),
            CudaKernelArgument::Scalar(CudaScalarValue::F8E4M3FN(bits)) => Self::F8E4M3FN(*bits),
            CudaKernelArgument::Scalar(CudaScalarValue::F8E4M3FNUZ(bits)) => Self::F8E4M3FNUZ(*bits),
            CudaKernelArgument::Scalar(CudaScalarValue::F8E4M3B11FNUZ(bits)) => Self::F8E4M3B11FNUZ(*bits),
            CudaKernelArgument::Scalar(CudaScalarValue::F8E5M2(bits)) => Self::F8E5M2(*bits),
            CudaKernelArgument::Scalar(CudaScalarValue::F8E5M2FNUZ(bits)) => Self::F8E5M2FNUZ(*bits),
            CudaKernelArgument::Scalar(CudaScalarValue::F8E8M0FNU(bits)) => Self::F8E8M0FNU(*bits),
            CudaKernelArgument::Scalar(CudaScalarValue::BF16(bits)) => Self::BF16(*bits),
            CudaKernelArgument::Scalar(CudaScalarValue::F16(bits)) => Self::F16(*bits),
            CudaKernelArgument::Scalar(CudaScalarValue::F32(value)) => Self::F32(*value),
            CudaKernelArgument::Scalar(CudaScalarValue::F64(value)) => Self::F64(*value),
        }
    }

    /// Returns a mutable pointer to the packed value for the CUDA Driver API argument vector.
    pub(super) fn as_mut_ptr(&mut self) -> *mut c_void {
        match self {
            Self::DevicePointer(value) => (value as *mut *mut c_void).cast(),
            Self::I8(value) => (value as *mut i8).cast(),
            Self::I16(value) => (value as *mut i16).cast(),
            Self::I32(value) => (value as *mut i32).cast(),
            Self::I64(value) => (value as *mut i64).cast(),
            Self::U8(value) => (value as *mut u8).cast(),
            Self::U16(value) => (value as *mut u16).cast(),
            Self::U32(value) => (value as *mut u32).cast(),
            Self::U64(value) => (value as *mut u64).cast(),
            Self::F4E2M1FN(bits) => (bits as *mut u8).cast(),
            Self::F6E2M3FN(bits) => (bits as *mut u8).cast(),
            Self::F6E3M2FN(bits) => (bits as *mut u8).cast(),
            Self::F8E3M4(bits) => (bits as *mut u8).cast(),
            Self::F8E4M3(bits) => (bits as *mut u8).cast(),
            Self::F8E4M3FN(bits) => (bits as *mut u8).cast(),
            Self::F8E4M3FNUZ(bits) => (bits as *mut u8).cast(),
            Self::F8E4M3B11FNUZ(bits) => (bits as *mut u8).cast(),
            Self::F8E5M2(bits) => (bits as *mut u8).cast(),
            Self::F8E5M2FNUZ(bits) => (bits as *mut u8).cast(),
            Self::F8E8M0FNU(bits) => (bits as *mut u8).cast(),
            Self::BF16(bits) => (bits as *mut u16).cast(),
            Self::F16(bits) => (bits as *mut u16).cast(),
            Self::F32(value) => (value as *mut f32).cast(),
            Self::F64(value) => (value as *mut f64).cast(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;

    use crate::tests::test_artifact;

    use super::*;

    /// Supplies a non-default stream identity for metadata-only tests without invoking CUDA.
    fn test_stream() -> CudaStream<'static> {
        CudaStream { handle: NonNull::<u64>::dangling().cast(), owner: PhantomData }
    }

    #[test]
    fn test_cuda_scalar_value_type() {
        for (value, expected) in [
            (CudaScalarValue::I8(1), CudaScalarType::I8),
            (CudaScalarValue::I16(1), CudaScalarType::I16),
            (CudaScalarValue::I32(1), CudaScalarType::I32),
            (CudaScalarValue::I64(1), CudaScalarType::I64),
            (CudaScalarValue::U8(1), CudaScalarType::U8),
            (CudaScalarValue::U16(1), CudaScalarType::U16),
            (CudaScalarValue::U32(1), CudaScalarType::U32),
            (CudaScalarValue::U64(1), CudaScalarType::U64),
            (CudaScalarValue::F4E2M1FN(1), CudaScalarType::F4E2M1FN),
            (CudaScalarValue::F6E2M3FN(1), CudaScalarType::F6E2M3FN),
            (CudaScalarValue::F6E3M2FN(1), CudaScalarType::F6E3M2FN),
            (CudaScalarValue::F8E3M4(1), CudaScalarType::F8E3M4),
            (CudaScalarValue::F8E4M3(1), CudaScalarType::F8E4M3),
            (CudaScalarValue::F8E4M3FN(1), CudaScalarType::F8E4M3FN),
            (CudaScalarValue::F8E4M3FNUZ(1), CudaScalarType::F8E4M3FNUZ),
            (CudaScalarValue::F8E4M3B11FNUZ(1), CudaScalarType::F8E4M3B11FNUZ),
            (CudaScalarValue::F8E5M2(1), CudaScalarType::F8E5M2),
            (CudaScalarValue::F8E5M2FNUZ(1), CudaScalarType::F8E5M2FNUZ),
            (CudaScalarValue::F8E8M0FNU(1), CudaScalarType::F8E8M0FNU),
            (CudaScalarValue::BF16(1), CudaScalarType::BF16),
            (CudaScalarValue::F16(1), CudaScalarType::F16),
            (CudaScalarValue::F32(1.0), CudaScalarType::F32),
            (CudaScalarValue::F64(1.0), CudaScalarType::F64),
        ] {
            assert_eq!(value.r#type(), expected);
        }
    }

    #[test]
    fn test_cuda_device_pointer_from_raw() {
        let mut allocation = 0u64;
        let address = (&mut allocation as *mut u64).cast();
        // Construction only wraps the address; these tests never submit the host allocation to CUDA.
        let pointer = unsafe { CudaDevicePointer::from_raw(address) }.unwrap();
        assert_eq!(pointer.as_raw(), address);
        assert!(matches!(
            unsafe { CudaDevicePointer::from_raw(std::ptr::null_mut()) },
            Err(Error::InvalidArgument { message, .. })
                if message == "CUDA kernel device pointer is a null pointer",
        ));
    }

    #[test]
    fn test_cuda_device_pointer_equality_hash_debug() {
        let pointer = CudaDevicePointer { pointer: NonNull::<u64>::dangling().cast(), owner: PhantomData };
        let other = CudaDevicePointer { pointer: NonNull::<u128>::dangling().cast(), owner: PhantomData };
        assert_eq!(pointer, pointer);
        assert_ne!(pointer, other);
        assert_eq!(HashMap::from([(pointer, 7)]).get(&pointer), Some(&7));
        assert_eq!(
            format!("{pointer:?}"),
            format!("CudaDevicePointer {{ pointer: {:?}, owner: PhantomData<&core::ffi::c_void> }}", pointer.as_raw()),
        );
    }

    #[test]
    fn test_cuda_kernel_argument_type() {
        let pointer = CudaDevicePointer { pointer: NonNull::dangling(), owner: PhantomData };
        assert_eq!(CudaKernelArgument::DevicePointer(pointer).r#type(), CudaKernelParameterType::DevicePointer);
        assert_eq!(
            CudaKernelArgument::Scalar(CudaScalarValue::I32(-1)).r#type(),
            CudaKernelParameterType::Scalar(CudaScalarType::I32),
        );
        let argument = CudaKernelArgument::Scalar(CudaScalarValue::I32(-1));
        assert_eq!(argument, argument);
        assert_ne!(argument, CudaKernelArgument::Scalar(CudaScalarValue::I32(1)));
        assert_eq!(format!("{argument:?}"), "Scalar(I32(-1))");
    }

    #[test]
    fn test_cuda_stream_from_raw() {
        assert_eq!(unsafe { CudaStream::from_raw(test_stream().as_raw()) }, Ok(test_stream()));
        assert!(matches!(
            unsafe { CudaStream::from_raw(std::ptr::null_mut()) },
            Err(Error::InvalidArgument { message, .. })
                if message == "CUDA stream handle is a null pointer",
        ));
        for address in [1, 2] {
            assert!(matches!(
                unsafe { CudaStream::from_raw(std::ptr::without_provenance_mut(address)) },
                Err(Error::InvalidArgument { message, .. })
                    if message == "CUDA default stream handles are unsupported",
            ));
        }
    }

    #[test]
    fn test_cuda_stream_equality_hash_debug() {
        let stream = test_stream();
        let other = CudaStream { handle: NonNull::<u128>::dangling().cast(), owner: PhantomData };
        assert_eq!(stream, stream);
        assert_ne!(stream, other);
        assert_eq!(HashMap::from([(stream, 7)]).get(&stream), Some(&7));
        assert_eq!(
            format!("{stream:?}"),
            format!("CudaStream {{ handle: {:?}, owner: PhantomData<&core::ffi::c_void> }}", stream.as_raw()),
        );
    }

    #[test]
    fn test_cuda_kernel_launch_new() {
        let arguments = vec![CudaKernelArgument::Scalar(CudaScalarValue::I32(-7))];
        let launch = CudaKernelLaunch::new(test_stream(), arguments.clone());
        assert_eq!(launch.stream, test_stream());
        assert_eq!(launch.arguments.as_ref(), arguments);
    }

    #[test]
    fn test_cuda_kernel_launch_arguments() {
        let arguments = vec![
            CudaKernelArgument::Scalar(CudaScalarValue::U32(7)),
            CudaKernelArgument::Scalar(CudaScalarValue::F64(1.5)),
        ];
        assert_eq!(CudaKernelLaunch::new(test_stream(), arguments.clone()).arguments(), arguments);
        assert_eq!(CudaKernelLaunch::new(test_stream(), Vec::new()).arguments(), &[]);
    }

    #[test]
    fn test_validate_launch() {
        let artifact = test_artifact(vec![CudaKernelParameterType::Scalar(CudaScalarType::I32)]);
        let launch = CudaKernelLaunch::new(test_stream(), vec![CudaKernelArgument::Scalar(CudaScalarValue::I32(-1))]);
        assert_eq!(validate_launch(&artifact, &launch), Ok(()));
        assert_eq!(
            validate_launch(&test_artifact(Vec::new()), &CudaKernelLaunch::new(test_stream(), Vec::new())),
            Ok(()),
        );
    }

    #[test]
    fn test_validate_launch_parameter_count() {
        let artifact = test_artifact(vec![CudaKernelParameterType::Scalar(CudaScalarType::I32)]);
        assert!(matches!(
            validate_launch(&artifact, &CudaKernelLaunch::new(test_stream(), Vec::new())),
            Err(Error::InvalidArgument { message, .. })
                if message == "CUDA kernel `test_kernel` expects 1 parameters but the launch frame contains 0",
        ));
    }

    #[test]
    fn test_validate_launch_parameter_type() {
        let artifact = test_artifact(vec![CudaKernelParameterType::Scalar(CudaScalarType::I32)]);
        let launch = CudaKernelLaunch::new(test_stream(), vec![CudaKernelArgument::Scalar(CudaScalarValue::U32(1))]);
        assert!(matches!(
            validate_launch(&artifact, &launch),
            Err(Error::InvalidArgument { message, .. })
                if message == "CUDA kernel `test_kernel` parameter 0 expects `Scalar(I32)` but received `Scalar(U32)`",
        ));
        let artifact = test_artifact(vec![CudaKernelParameterType::Scalar(CudaScalarType::F16)]);
        let launch =
            CudaKernelLaunch::new(test_stream(), vec![CudaKernelArgument::Scalar(CudaScalarValue::BF16(0x3f80))]);
        assert!(matches!(
            validate_launch(&artifact, &launch),
            Err(Error::InvalidArgument { message, .. })
                if message == "CUDA kernel `test_kernel` parameter 0 expects `Scalar(F16)` but received `Scalar(BF16)`",
        ));
    }

    #[test]
    fn test_cuda_kernel_argument_storage_from_argument() {
        let pointer = CudaDevicePointer { pointer: NonNull::dangling(), owner: PhantomData };
        assert!(matches!(
            CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::DevicePointer(pointer)),
            CudaKernelArgumentStorage::DevicePointer(address) if address == pointer.as_raw(),
        ));
        let argument = CudaKernelArgument::Scalar(CudaScalarValue::I8(-7));
        assert!(matches!(
            CudaKernelArgumentStorage::from_argument(&argument),
            CudaKernelArgumentStorage::I8(value) if value == -7,
        ));
        let argument = CudaKernelArgument::Scalar(CudaScalarValue::I16(-7));
        assert!(matches!(
            CudaKernelArgumentStorage::from_argument(&argument),
            CudaKernelArgumentStorage::I16(value) if value == -7,
        ));
        let argument = CudaKernelArgument::Scalar(CudaScalarValue::I32(-7));
        assert!(matches!(
            CudaKernelArgumentStorage::from_argument(&argument),
            CudaKernelArgumentStorage::I32(value) if value == -7,
        ));
        let argument = CudaKernelArgument::Scalar(CudaScalarValue::I64(-7));
        assert!(matches!(
            CudaKernelArgumentStorage::from_argument(&argument),
            CudaKernelArgumentStorage::I64(value) if value == -7,
        ));
        let argument = CudaKernelArgument::Scalar(CudaScalarValue::U8(0x81));
        assert!(matches!(
            CudaKernelArgumentStorage::from_argument(&argument),
            CudaKernelArgumentStorage::U8(value) if value == 0x81,
        ));
        let argument = CudaKernelArgument::Scalar(CudaScalarValue::U16(0x81));
        assert!(matches!(
            CudaKernelArgumentStorage::from_argument(&argument),
            CudaKernelArgumentStorage::U16(value) if value == 0x81,
        ));
        let argument = CudaKernelArgument::Scalar(CudaScalarValue::U32(0x81));
        assert!(matches!(
            CudaKernelArgumentStorage::from_argument(&argument),
            CudaKernelArgumentStorage::U32(value) if value == 0x81,
        ));
        let argument = CudaKernelArgument::Scalar(CudaScalarValue::U64(0x81));
        assert!(matches!(
            CudaKernelArgumentStorage::from_argument(&argument),
            CudaKernelArgumentStorage::U64(value) if value == 0x81,
        ));
        let argument = CudaKernelArgument::Scalar(CudaScalarValue::F4E2M1FN(0x81));
        assert!(matches!(
            CudaKernelArgumentStorage::from_argument(&argument),
            CudaKernelArgumentStorage::F4E2M1FN(value) if value == 0x81,
        ));
        let argument = CudaKernelArgument::Scalar(CudaScalarValue::F6E2M3FN(0x81));
        assert!(matches!(
            CudaKernelArgumentStorage::from_argument(&argument),
            CudaKernelArgumentStorage::F6E2M3FN(value) if value == 0x81,
        ));
        let argument = CudaKernelArgument::Scalar(CudaScalarValue::F6E3M2FN(0x81));
        assert!(matches!(
            CudaKernelArgumentStorage::from_argument(&argument),
            CudaKernelArgumentStorage::F6E3M2FN(value) if value == 0x81,
        ));
        let argument = CudaKernelArgument::Scalar(CudaScalarValue::F8E3M4(0x81));
        assert!(matches!(
            CudaKernelArgumentStorage::from_argument(&argument),
            CudaKernelArgumentStorage::F8E3M4(value) if value == 0x81,
        ));
        let argument = CudaKernelArgument::Scalar(CudaScalarValue::F8E4M3(0x81));
        assert!(matches!(
            CudaKernelArgumentStorage::from_argument(&argument),
            CudaKernelArgumentStorage::F8E4M3(value) if value == 0x81,
        ));
        let argument = CudaKernelArgument::Scalar(CudaScalarValue::F8E4M3FN(0x81));
        assert!(matches!(
            CudaKernelArgumentStorage::from_argument(&argument),
            CudaKernelArgumentStorage::F8E4M3FN(value) if value == 0x81,
        ));
        let argument = CudaKernelArgument::Scalar(CudaScalarValue::F8E4M3FNUZ(0x81));
        assert!(matches!(
            CudaKernelArgumentStorage::from_argument(&argument),
            CudaKernelArgumentStorage::F8E4M3FNUZ(value) if value == 0x81,
        ));
        let argument = CudaKernelArgument::Scalar(CudaScalarValue::F8E4M3B11FNUZ(0x81));
        assert!(matches!(
            CudaKernelArgumentStorage::from_argument(&argument),
            CudaKernelArgumentStorage::F8E4M3B11FNUZ(value) if value == 0x81,
        ));
        let argument = CudaKernelArgument::Scalar(CudaScalarValue::F8E5M2(0x81));
        assert!(matches!(
            CudaKernelArgumentStorage::from_argument(&argument),
            CudaKernelArgumentStorage::F8E5M2(value) if value == 0x81,
        ));
        let argument = CudaKernelArgument::Scalar(CudaScalarValue::F8E5M2FNUZ(0x81));
        assert!(matches!(
            CudaKernelArgumentStorage::from_argument(&argument),
            CudaKernelArgumentStorage::F8E5M2FNUZ(value) if value == 0x81,
        ));
        let argument = CudaKernelArgument::Scalar(CudaScalarValue::F8E8M0FNU(0x81));
        assert!(matches!(
            CudaKernelArgumentStorage::from_argument(&argument),
            CudaKernelArgumentStorage::F8E8M0FNU(value) if value == 0x81,
        ));
        let argument = CudaKernelArgument::Scalar(CudaScalarValue::BF16(0x7fc1));
        assert!(matches!(
            CudaKernelArgumentStorage::from_argument(&argument),
            CudaKernelArgumentStorage::BF16(value) if value == 0x7fc1,
        ));
        let argument = CudaKernelArgument::Scalar(CudaScalarValue::F16(0x7fc1));
        assert!(matches!(
            CudaKernelArgumentStorage::from_argument(&argument),
            CudaKernelArgumentStorage::F16(value) if value == 0x7fc1,
        ));
        let argument = CudaKernelArgument::Scalar(CudaScalarValue::F32(1.5));
        assert!(matches!(
            CudaKernelArgumentStorage::from_argument(&argument),
            CudaKernelArgumentStorage::F32(value) if value == 1.5,
        ));
        let argument = CudaKernelArgument::Scalar(CudaScalarValue::F64(1.5));
        assert!(matches!(
            CudaKernelArgumentStorage::from_argument(&argument),
            CudaKernelArgumentStorage::F64(value) if value == 1.5,
        ));
    }

    #[test]
    fn test_cuda_kernel_argument_storage_as_mut_ptr() {
        let pointer = CudaDevicePointer { pointer: NonNull::dangling(), owner: PhantomData };
        let mut storage = CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::DevicePointer(pointer));
        assert_eq!(unsafe { *storage.as_mut_ptr().cast::<*mut c_void>() }, pointer.as_raw());
        let mut storage =
            CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::Scalar(CudaScalarValue::I8(-7)));
        assert_eq!(storage.as_mut_ptr().addr() % std::mem::align_of::<i8>(), 0);
        assert_eq!(unsafe { *storage.as_mut_ptr().cast::<i8>() }, -7);
        let mut storage =
            CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::Scalar(CudaScalarValue::I16(-7)));
        assert_eq!(storage.as_mut_ptr().addr() % std::mem::align_of::<i16>(), 0);
        assert_eq!(unsafe { *storage.as_mut_ptr().cast::<i16>() }, -7);
        let mut storage =
            CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::Scalar(CudaScalarValue::I32(-7)));
        assert_eq!(storage.as_mut_ptr().addr() % std::mem::align_of::<i32>(), 0);
        assert_eq!(unsafe { *storage.as_mut_ptr().cast::<i32>() }, -7);
        let mut storage =
            CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::Scalar(CudaScalarValue::I64(-7)));
        assert_eq!(storage.as_mut_ptr().addr() % std::mem::align_of::<i64>(), 0);
        assert_eq!(unsafe { *storage.as_mut_ptr().cast::<i64>() }, -7);
        let mut storage =
            CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::Scalar(CudaScalarValue::U8(0x81)));
        assert_eq!(storage.as_mut_ptr().addr() % std::mem::align_of::<u8>(), 0);
        assert_eq!(unsafe { *storage.as_mut_ptr().cast::<u8>() }, 0x81);
        let mut storage =
            CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::Scalar(CudaScalarValue::U16(0x81)));
        assert_eq!(storage.as_mut_ptr().addr() % std::mem::align_of::<u16>(), 0);
        assert_eq!(unsafe { *storage.as_mut_ptr().cast::<u16>() }, 0x81);
        let mut storage =
            CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::Scalar(CudaScalarValue::U32(0x81)));
        assert_eq!(storage.as_mut_ptr().addr() % std::mem::align_of::<u32>(), 0);
        assert_eq!(unsafe { *storage.as_mut_ptr().cast::<u32>() }, 0x81);
        let mut storage =
            CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::Scalar(CudaScalarValue::U64(0x81)));
        assert_eq!(storage.as_mut_ptr().addr() % std::mem::align_of::<u64>(), 0);
        assert_eq!(unsafe { *storage.as_mut_ptr().cast::<u64>() }, 0x81);
        let mut storage =
            CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::Scalar(CudaScalarValue::F4E2M1FN(0x81)));
        assert_eq!(storage.as_mut_ptr().addr() % std::mem::align_of::<u8>(), 0);
        assert_eq!(unsafe { *storage.as_mut_ptr().cast::<u8>() }, 0x81);
        let mut storage =
            CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::Scalar(CudaScalarValue::F6E2M3FN(0x81)));
        assert_eq!(storage.as_mut_ptr().addr() % std::mem::align_of::<u8>(), 0);
        assert_eq!(unsafe { *storage.as_mut_ptr().cast::<u8>() }, 0x81);
        let mut storage =
            CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::Scalar(CudaScalarValue::F6E3M2FN(0x81)));
        assert_eq!(storage.as_mut_ptr().addr() % std::mem::align_of::<u8>(), 0);
        assert_eq!(unsafe { *storage.as_mut_ptr().cast::<u8>() }, 0x81);
        let mut storage =
            CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::Scalar(CudaScalarValue::F8E3M4(0x81)));
        assert_eq!(storage.as_mut_ptr().addr() % std::mem::align_of::<u8>(), 0);
        assert_eq!(unsafe { *storage.as_mut_ptr().cast::<u8>() }, 0x81);
        let mut storage =
            CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::Scalar(CudaScalarValue::F8E4M3(0x81)));
        assert_eq!(storage.as_mut_ptr().addr() % std::mem::align_of::<u8>(), 0);
        assert_eq!(unsafe { *storage.as_mut_ptr().cast::<u8>() }, 0x81);
        let mut storage =
            CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::Scalar(CudaScalarValue::F8E4M3FN(0x81)));
        assert_eq!(storage.as_mut_ptr().addr() % std::mem::align_of::<u8>(), 0);
        assert_eq!(unsafe { *storage.as_mut_ptr().cast::<u8>() }, 0x81);
        let mut storage =
            CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::Scalar(CudaScalarValue::F8E4M3FNUZ(0x81)));
        assert_eq!(storage.as_mut_ptr().addr() % std::mem::align_of::<u8>(), 0);
        assert_eq!(unsafe { *storage.as_mut_ptr().cast::<u8>() }, 0x81);
        let mut storage =
            CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::Scalar(CudaScalarValue::F8E4M3B11FNUZ(0x81)));
        assert_eq!(storage.as_mut_ptr().addr() % std::mem::align_of::<u8>(), 0);
        assert_eq!(unsafe { *storage.as_mut_ptr().cast::<u8>() }, 0x81);
        let mut storage =
            CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::Scalar(CudaScalarValue::F8E5M2(0x81)));
        assert_eq!(storage.as_mut_ptr().addr() % std::mem::align_of::<u8>(), 0);
        assert_eq!(unsafe { *storage.as_mut_ptr().cast::<u8>() }, 0x81);
        let mut storage =
            CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::Scalar(CudaScalarValue::F8E5M2FNUZ(0x81)));
        assert_eq!(storage.as_mut_ptr().addr() % std::mem::align_of::<u8>(), 0);
        assert_eq!(unsafe { *storage.as_mut_ptr().cast::<u8>() }, 0x81);
        let mut storage =
            CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::Scalar(CudaScalarValue::F8E8M0FNU(0x81)));
        assert_eq!(storage.as_mut_ptr().addr() % std::mem::align_of::<u8>(), 0);
        assert_eq!(unsafe { *storage.as_mut_ptr().cast::<u8>() }, 0x81);
        let mut storage =
            CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::Scalar(CudaScalarValue::BF16(0x7fc1)));
        assert_eq!(storage.as_mut_ptr().addr() % std::mem::align_of::<u16>(), 0);
        assert_eq!(unsafe { *storage.as_mut_ptr().cast::<u16>() }, 0x7fc1);
        let mut storage =
            CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::Scalar(CudaScalarValue::F16(0x7fc1)));
        assert_eq!(storage.as_mut_ptr().addr() % std::mem::align_of::<u16>(), 0);
        assert_eq!(unsafe { *storage.as_mut_ptr().cast::<u16>() }, 0x7fc1);
        let mut storage = CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::Scalar(CudaScalarValue::F32(
            f32::from_bits(0x7fc12345),
        )));
        assert_eq!(storage.as_mut_ptr().addr() % std::mem::align_of::<u32>(), 0);
        assert_eq!(unsafe { *storage.as_mut_ptr().cast::<u32>() }, 0x7fc12345);
        let mut storage = CudaKernelArgumentStorage::from_argument(&CudaKernelArgument::Scalar(CudaScalarValue::F64(
            f64::from_bits(0x7ff8123456789abc),
        )));
        assert_eq!(storage.as_mut_ptr().addr() % std::mem::align_of::<u64>(), 0);
        assert_eq!(unsafe { *storage.as_mut_ptr().cast::<u64>() }, 0x7ff8123456789abc);
    }
}
