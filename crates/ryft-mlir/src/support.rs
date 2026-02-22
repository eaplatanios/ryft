use std::backtrace::Backtrace;
use std::fmt::{Debug, Display};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::marker::PhantomData;
use std::path::Path;

use thiserror::Error;

use ryft_xla_sys::bindings::{
    MlirLlvmThreadPool, MlirLogicalResult, MlirStringRef, MlirTypeID, MlirTypeIDAllocator, mlirLlvmThreadPoolCreate,
    mlirLlvmThreadPoolDestroy, mlirStringRefCreateFromCString, mlirStringRefEqual, mlirTypeIDAllocatorAllocateTypeID,
    mlirTypeIDAllocatorCreate, mlirTypeIDAllocatorDestroy, mlirTypeIDCreate, mlirTypeIDEqual, mlirTypeIDHashValue,
};

/// Pointer to a sized fragment of a string that is not necessarily null-terminated. Note that [`StringRef`] does not
/// own the underlying string and that is why it has a lifetime parameter that is tied to the owner of that string.
#[derive(Copy, Clone)]
pub struct StringRef<'o> {
    /// Handle that represents this [`StringRef`] in the MLIR C API.
    handle: MlirStringRef,

    /// [`PhantomData`] used to track the lifetime of the owner of the underlying string.
    owner: PhantomData<&'o ()>,
}

impl<'o> StringRef<'o> {
    /// Constructs a new [`StringRef`] from the provided [`MlirStringRef`] handle
    /// that came from a function in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn from_c_api(handle: MlirStringRef) -> Self {
        Self { handle, owner: PhantomData }
    }

    /// Returns the [`MlirStringRef`] that corresponds to this [`StringRef`]
    /// and which can be passed to functions in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn to_c_api(&self) -> MlirStringRef {
        self.handle
    }

    /// Returns the underlying bytes of this [`StringRef`].
    pub fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.handle.data as *mut u8, self.handle.length) }
    }

    /// Returns an [`str`] slice representation of the underlying string.
    pub fn as_str(&self) -> Result<&'o str, std::str::Utf8Error> {
        self.try_into()
    }
}

impl Display for StringRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str().unwrap_or("<non-utf-8 string>"))
    }
}

impl Debug for StringRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StringRef[{}]", self.to_string())
    }
}

impl PartialEq for StringRef<'_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirStringRefEqual(self.handle, other.handle) }
    }
}

impl Eq for StringRef<'_> {}

impl Hash for StringRef<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        unsafe { std::slice::from_raw_parts(self.handle.data as *mut u8, self.handle.length).hash(state) }
    }
}

impl<'o> From<&'o str> for StringRef<'o> {
    fn from(value: &'o str) -> Self {
        unsafe { Self::from_c_api(MlirStringRef { data: value.as_bytes().as_ptr() as *const _, length: value.len() }) }
    }
}

impl<'o> From<&'o std::ffi::CStr> for StringRef<'o> {
    fn from(value: &'o std::ffi::CStr) -> Self {
        unsafe { Self::from_c_api(mlirStringRefCreateFromCString(value.to_bytes_with_nul().as_ptr() as *const _)) }
    }
}

impl<'o> TryFrom<&StringRef<'o>> for &'o str {
    type Error = std::str::Utf8Error;

    fn try_from(value: &StringRef<'o>) -> Result<Self, Self::Error> {
        unsafe {
            let bytes = std::slice::from_raw_parts(value.handle.data as *mut u8, value.handle.length);
            std::str::from_utf8(if bytes[bytes.len() - 1] == 0 { &bytes[..bytes.len() - 1] } else { bytes })
        }
    }
}

impl<'o> TryFrom<StringRef<'o>> for &'o str {
    type Error = std::str::Utf8Error;

    fn try_from(value: StringRef<'o>) -> Result<Self, Self::Error> {
        (&value).try_into()
    }
}

impl<'p> From<&'p Path> for StringRef<'p> {
    fn from(value: &'p Path) -> Self {
        value.to_str().expect("non-UTF-8 paths cannot be converted to MLIR `StringRef`s").into()
    }
}

/// [`MlirStringCallback`](ryft_xla_sys::bindings::MlirStringCallback) that can be passed to MLIR functions so that they
/// can return a string by writing it to a [`std::fmt::Formatter`]. The provided `data` pointer must point to a tuple
/// containing a [`std::fmt::Formatter`] and a [`std::fmt::Result`]. Note that if something goes wrong while writing to
/// the [`std::fmt::Formatter`], the [`std::fmt::Result`] stored in `data` will be set to an [`Err`] instance containing
/// the error.
///
/// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
/// safe and should not be necessary outside of this library. However, it is still supported via making functions
/// like this one public so that users of this library can extend it with yet unsupported features that the
/// underlying MLIR C API supports.
pub unsafe extern "C" fn write_to_formatter_callback(string_ref: MlirStringRef, data: *mut std::ffi::c_void) {
    unsafe {
        let (formatter, result) = &mut *(data as *mut (&mut std::fmt::Formatter, std::fmt::Result));
        if let Ok(_) = result {
            *result = StringRef::from_c_api(string_ref)
                .as_str()
                .map_err(|_| std::fmt::Error)
                .and_then(|result| write!(formatter, "{}", result));
        }
    }
}

/// [`MlirStringCallback`](ryft_xla_sys::bindings::MlirStringCallback) that can be passed to MLIR functions so that they
/// can return a string by writing it to a [`String`]. The provided `data` pointer must point to a tuple containing a
/// [`String`] and a [`Result<(), std::str::Utf8Error>`](Result). Note that if something goes wrong while writing to the
/// string, the [`Result`] stored in `data` will be set to an [`Err`] instance containing the error.
///
/// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
/// safe and should not be necessary outside of this library. However, it is still supported via making functions
/// like this one public so that users of this library can extend it with yet unsupported features that the
/// underlying MLIR C API supports.
pub unsafe extern "C" fn write_to_string_callback(string_ref: MlirStringRef, data: *mut std::ffi::c_void) {
    unsafe {
        let (string, result) = &mut *(data as *mut (String, Result<(), std::str::Utf8Error>));
        if let Ok(_) = result {
            *result = StringRef::from_c_api(string_ref).as_str().map(|result| string.push_str(result));
        }
    }
}

/// [`MlirStringCallback`](ryft_xla_sys::bindings::MlirStringCallback) that can be passed to MLIR functions so that they
/// can return a string by writing it to a bytes buffer. The provided `data` pointer must point to a [`Vec<u8>`].
///
/// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
/// safe and should not be necessary outside of this library. However, it is still supported via making functions
/// like this one public so that users of this library can extend it with yet unsupported features that the
/// underlying MLIR C API supports.
pub unsafe extern "C" fn write_to_bytes_callback(string_ref: MlirStringRef, data: *mut std::ffi::c_void) {
    unsafe {
        let bytes = &mut *(data as *mut Vec<u8>);
        bytes.extend_from_slice(StringRef::from_c_api(string_ref).bytes());
    }
}

/// Logical result value (essentially a boolean with named states). The LLVM convention for using boolean values
/// to designate success or failure of an operation is a moving target and so MLIR opted for an explicit class.
/// Instances of [`LogicalResult`] must only be inspected using the associated [`LogicalResult::is_success`]
/// and [`LogicalResult::is_failure`] functions.
#[derive(Copy, Clone)]
pub struct LogicalResult {
    /// Handle that represents this [`LogicalResult`] in the MLIR C API.
    handle: MlirLogicalResult,
}

impl LogicalResult {
    /// Constructs a new [`LogicalResult`] from the provided [`MlirLogicalResult`] handle
    /// that came from a function in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn from_c_api(handle: MlirLogicalResult) -> Self {
        Self { handle }
    }

    /// Returns the [`MlirLogicalResult`] that corresponds to this [`LogicalResult`]
    /// and which can be passed to functions in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn to_c_api(&self) -> MlirLogicalResult {
        self.handle
    }

    /// Creates a new [`LogicalResult`] that represents a "success" result.
    pub fn success() -> Self {
        Self { handle: MlirLogicalResult { value: 1 } }
    }

    /// Creates a new [`LogicalResult`] that represents a "failure" result.
    pub fn failure() -> Self {
        Self { handle: MlirLogicalResult { value: 0 } }
    }

    /// Returns `true` if this [`LogicalResult`] represents a "success" result.
    pub fn is_success(&self) -> bool {
        self.handle.value != 0
    }

    /// Returns `true` if this [`LogicalResult`] represents a "failure" result.
    pub fn is_failure(&self) -> bool {
        self.handle.value == 0
    }
}

impl Display for LogicalResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", if self.is_success() { "success" } else { "failure" })
    }
}

impl Debug for LogicalResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LogicalResult[{}]", self.to_string())
    }
}

impl PartialEq for LogicalResult {
    fn eq(&self, other: &Self) -> bool {
        self.is_success() == other.is_success()
    }
}

impl Eq for LogicalResult {}

impl From<bool> for LogicalResult {
    fn from(value: bool) -> Self {
        if value { Self::success() } else { Self::failure() }
    }
}

impl From<LogicalResult> for bool {
    fn from(value: LogicalResult) -> bool {
        value.is_success()
    }
}

/// LLVM thread pool allowing asynchronous parallel execution. This is mainly used in the context of MLIR
/// [`Context`](crate::Context)s. Refer to the documentation of [`Context`](crate::Context) for more information
/// on how to use it.
#[derive(Debug)]
pub struct ThreadPool {
    /// Handle that represents this [`ThreadPool`] in the MLIR C API.
    handle: MlirLlvmThreadPool,
}

impl ThreadPool {
    /// Creates a new LLVM [`ThreadPool`].
    pub fn new() -> Self {
        Self { handle: unsafe { mlirLlvmThreadPoolCreate() } }
    }

    /// Constructs a new [`ThreadPool`] from the provided [`MlirLlvmThreadPool`] handle
    /// that came from a function in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn from_c_api(handle: MlirLlvmThreadPool) -> Self {
        Self { handle }
    }

    /// Returns the [`MlirLlvmThreadPool`] that corresponds to this [`ThreadPool`]
    /// and which can be passed to functions in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn to_c_api(&self) -> MlirLlvmThreadPool {
        self.handle
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        unsafe { mlirLlvmThreadPoolDestroy(self.handle) }
    }
}

/// Non-owning reference to a [`ThreadPool`].
#[derive(Copy, Clone, Debug)]
pub struct ThreadPoolRef<'o> {
    /// Handle that represents this [`ThreadPoolRef`] in the MLIR C API.
    handle: MlirLlvmThreadPool,

    /// [`PhantomData`] used to track the lifetime of the owner of the underlying thread pool.
    owner: PhantomData<&'o ()>,
}

impl ThreadPoolRef<'_> {
    /// Constructs a new [`ThreadPoolRef`] from the provided [`MlirLlvmThreadPool`] handle
    /// that came from a function in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn from_c_api(handle: MlirLlvmThreadPool) -> Self {
        Self { handle, owner: Default::default() }
    }

    /// Returns the [`MlirLlvmThreadPool`] that corresponds to this [`ThreadPoolRef`]
    /// and which can be passed to functions in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn to_c_api(&self) -> MlirLlvmThreadPool {
        self.handle
    }
}

impl<'o> From<&'o ThreadPool> for ThreadPoolRef<'o> {
    fn from(value: &'o ThreadPool) -> Self {
        Self { handle: value.handle, owner: PhantomData }
    }
}

#[derive(Error, Debug)]
pub enum TypeIdError {
    #[error("type ID reference data must be 8-byte aligned")]
    AlignmentError { backtrace: String },
}

/// [`TypeId`]s provide efficient and unique identifiers for specific C++ types (not to be conflated with MLIR
/// [`Type`](crate::Type)s; e.g., multiple non-equal instances of MLIR [`Type`](crate::Type)s might have the same
/// [`TypeId`]). This allows for a C++ type to be compared, hashed, and stored in an opaque context.
pub struct TypeId<'allocator> {
    /// Handle that represents this [`TypeId`] in the MLIR C API.
    handle: MlirTypeID,

    /// [`PhantomData`] used to track the lifetime of the [`TypeIdAllocator`] that owns this [`TypeId`].
    owner: PhantomData<&'allocator TypeIdAllocator>,
}

impl TypeId<'_> {
    /// Constructs a new [`TypeId`] from the provided [`MlirTypeID`] handle
    /// that came from a function in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn from_c_api(handle: MlirTypeID) -> Option<Self> {
        if handle.ptr.is_null() { None } else { Some(Self { handle, owner: PhantomData }) }
    }

    /// Returns the [`MlirTypeID`] that corresponds to this [`TypeId`]
    /// and which can be passed to functions in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn to_c_api(&self) -> MlirTypeID {
        self.handle
    }

    /// Creates a new [`TypeId`] using the provided reference data. Note that the reference data must be
    /// 8-byte aligned. If it is not, then this function will return a [`TypeIdError`].
    pub fn create<T>(reference: &T) -> Result<Self, TypeIdError> {
        let reference = reference as *const _ as *const u8;
        if reference.align_offset(8) != 0 {
            Err(TypeIdError::AlignmentError { backtrace: Backtrace::capture().to_string() })
        } else {
            let reference = reference as *const _ as *const std::ffi::c_void;
            Ok(unsafe { Self::from_c_api(mlirTypeIDCreate(reference)).unwrap() })
        }
    }
}

impl PartialEq for TypeId<'_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirTypeIDEqual(self.handle, other.handle) }
    }
}

impl Eq for TypeId<'_> {}

impl Hash for TypeId<'_> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        unsafe { mlirTypeIDHashValue(self.handle).hash(hasher) }
    }
}

impl Debug for TypeId<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        let hash = hasher.finish();
        write!(formatter, "TypeId[{hash}]")
    }
}

/// [`TypeIdAllocator`]s provide a way to create new [`TypeId`]s dynamically at runtime.
pub struct TypeIdAllocator {
    /// Handle that represents this [`TypeIdAllocator`] in the MLIR C API.
    handle: MlirTypeIDAllocator,
}

impl TypeIdAllocator {
    /// Creates a new [`TypeIdAllocator`].
    pub fn new() -> Self {
        Self { handle: unsafe { mlirTypeIDAllocatorCreate() } }
    }

    /// Allocates a new [`TypeId`] that is ensured to be unique for the lifetime of this [`TypeIdAllocator`].
    pub fn allocate(&self) -> TypeId<'_> {
        unsafe { TypeId::from_c_api(mlirTypeIDAllocatorAllocateTypeID(self.handle)).unwrap() }
    }
}

impl Drop for TypeIdAllocator {
    fn drop(&mut self) {
        // Deallocates both the type ID allocator and also all of its allocated [`TypeId`]s.
        unsafe { mlirTypeIDAllocatorDestroy(self.handle) }
    }
}

impl Default for TypeIdAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::ffi::CString;
    use std::path::Path;

    use ryft_xla_sys::bindings::{MlirStringRef, MlirTypeID};

    use super::*;

    #[test]
    fn test_string_ref() {
        // Test construction and comparison.
        assert_eq!(StringRef::from("foo"), StringRef::from("foo"));
        assert_ne!(StringRef::from("foo"), StringRef::from("bar"));
        assert_eq!(StringRef::from("foo").as_str().unwrap(), "foo");

        // Test display and debug.
        assert_eq!(format!("{}", StringRef::from("test")), "test");
        assert_eq!(format!("{:?}", StringRef::from("test")), "StringRef[test]");

        // Test conversions.
        assert_eq!(StringRef::from("hello").bytes(), b"hello");
        assert_eq!(StringRef::from(Path::new("/tmp/test.txt")).as_str().unwrap(), "/tmp/test.txt");

        // Test C string conversions.
        let c_string = CString::new("test").unwrap();
        assert_eq!(StringRef::from(c_string.as_c_str()).as_str().unwrap(), "test");
        let bytes = c_string.as_c_str().to_bytes_with_nul();
        assert_eq!(
            unsafe { StringRef::from_c_api(MlirStringRef { data: bytes.as_ptr() as *const _, length: bytes.len() }) }
                .as_str()
                .unwrap(),
            "test"
        );

        // Test hashing.
        let string_ref_0 = StringRef::from("test");
        let string_ref_1 = StringRef::from("test");
        let string_ref_2 = StringRef::from("other");
        let mut map = HashMap::new();
        map.insert(string_ref_0, 1);
        map.insert(string_ref_1, 2);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&string_ref_0), Some(&2));
        assert_eq!(map.get(&string_ref_2), None);
    }

    #[test]
    fn test_write_to_string_callback() {
        let string = "test string for callback";
        let string_ref = StringRef::from(string);
        let mut data = (String::new(), Ok::<(), std::str::Utf8Error>(()));
        unsafe { write_to_string_callback(string_ref.to_c_api(), &mut data as *mut _ as *mut std::ffi::c_void) };
        assert!(data.1.is_ok());
        assert_eq!(data.0, string);
    }

    #[test]
    fn test_write_to_bytes_callback() {
        let string = "test bytes";
        let string_ref = StringRef::from(string);
        let mut data = Vec::<u8>::new();
        unsafe { write_to_bytes_callback(string_ref.to_c_api(), &mut data as *mut _ as *mut std::ffi::c_void) };
        assert_eq!(data, string.as_bytes());
    }

    #[test]
    fn test_logical_result() {
        // Test construction and comparison.
        let success = LogicalResult::success();
        assert!(success.is_success());
        assert!(!success.is_failure());
        assert_eq!(LogicalResult::from(true), success);
        assert_eq!(bool::from(success), true);
        let failure = LogicalResult::failure();
        assert!(failure.is_failure());
        assert!(!failure.is_success());
        assert_eq!(failure, LogicalResult::from(false));
        assert_eq!(bool::from(failure), false);

        // Test display and debug.
        assert_eq!(format!("{}", success), "success");
        assert_eq!(format!("{:?}", failure), "LogicalResult[failure]");
    }

    #[test]
    fn test_thread_pool() {
        let pool = ThreadPool::new();
        let pool_ref = ThreadPoolRef::from(&pool);
        let _ = pool_ref.clone();
        let handle = unsafe { pool.to_c_api() };
        std::mem::forget(pool);
        let _ = unsafe { ThreadPool::from_c_api(handle) };
    }

    #[test]
    fn test_type_id() {
        // Test with good data.
        #[repr(align(8))]
        struct AlignedData {
            _value: u64,
        }

        let data_0 = AlignedData { _value: 42 };
        let data_1 = AlignedData { _value: 42 };
        let type_id_0 = TypeId::create(&data_0);
        let type_id_1 = TypeId::create(&data_1);
        assert!(type_id_0.is_ok());
        assert!(type_id_1.is_ok());

        // Test debug.
        assert!(format!("{:?}", TypeId::create(&data_0).unwrap()).contains("TypeId"));

        // Test equality.
        let type_id_0 = type_id_0.unwrap();
        let type_id_1 = type_id_1.unwrap();
        assert_eq!(type_id_0, type_id_0);
        assert_eq!(TypeId::create(&data_0).ok(), Some(type_id_0));
        let type_id_0 = TypeId::create(&data_0).unwrap();
        assert_ne!(type_id_0, type_id_1);
        assert_ne!(type_id_1, type_id_0);
        assert_eq!(Some(type_id_1), TypeId::create(&data_1).ok());

        // Test hashing.
        let type_id_0 = TypeId::create(&data_0).unwrap();
        let type_id_1 = TypeId::create(&data_1).unwrap();
        let mut map = HashMap::new();
        map.insert(unsafe { TypeId::from_c_api(type_id_0.to_c_api()).unwrap() }, "first");
        map.insert(TypeId::create(&data_0).unwrap(), "second");
        map.insert(type_id_1, "third");
        assert_eq!(map.len(), 2);

        // Test with bad data.
        #[repr(packed)]
        struct UnalignedData {
            _value_0: bool,
            _value_1: u8,
        }

        let type_id = TypeId::create(&UnalignedData { _value_0: false, _value_1: 42 });

        #[cfg(unix)]
        assert!(type_id.is_err());

        #[cfg(windows)]
        assert!(type_id.is_ok());

        // Test null pointer edge case.
        let bad_handle = MlirTypeID { ptr: std::ptr::null_mut() };
        let type_id = unsafe { TypeId::from_c_api(bad_handle) };
        assert!(type_id.is_none());
    }

    #[test]
    fn test_type_id_allocator() {
        let allocator_0 = TypeIdAllocator::default();
        let allocator_1 = TypeIdAllocator::default();
        let type_id_0 = allocator_0.allocate();
        let type_id_1 = allocator_0.allocate();
        let type_id_2 = allocator_0.allocate();
        let type_id_3 = allocator_1.allocate();
        assert_eq!(type_id_0, type_id_0);
        assert_ne!(type_id_0, type_id_1);
        assert_ne!(type_id_0, type_id_2);
        assert_ne!(type_id_1, type_id_2);
        assert_eq!(type_id_2, type_id_2);
        assert_ne!(type_id_2, type_id_3);
    }
}
