use std::cell::{Ref, RefCell, RefMut};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;

use ryft_xla_sys::bindings::{
    MlirContext, mlirContextCreate, mlirContextCreateWithThreading, mlirContextDestroy,
    mlirContextEnableMultithreading, mlirContextEqual, mlirContextGetNumThreads, mlirContextGetThreadPool,
    mlirContextSetThreadPool,
};

use crate::ThreadPoolRef;

/// Thread pool being used by a [`Context`]. This enum allows us to use safe abstractions for both [`Context`]s that
/// own an underlying [`ThreadPoolRef`] and for [`Context`]s that use a shared one.
#[derive(Debug)]
pub(crate) enum ContextThreadPool<'t> {
    /// Multi-threading is disabled for the corresponding [`Context`].
    None,

    /// Multi-threading is enabled for the corresponding [`Context`] and it owns its own [`ThreadPoolRef`].
    Owned,

    /// Multi-threading is enabled for the corresponding [`Context`] and it uses a borrowed [`ThreadPoolRef`].
    Borrowed(ThreadPoolRef<'t>),
}

/// Represents the threading mode of a [`Context`] (i.e., whether threading is enabled or disabled for that context).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Threading {
    Enabled,
    Disabled,
}

/// [`Context`] is the [top-level object](https://mlir.llvm.org/doxygen/classmlir_1_1MLIRContext.html) for a collection
/// of MLIR operations. It holds immortal uniqued objects like [`Type`](crate::Type)s, [`Location`](crate::Location)s,
/// [`Dialect`](crate::Dialect)s, as well as the tables used to unique them.
///
/// [`Context`]s wrap some multithreading facilities, and in particular by default they will each implicitly create a
/// new thread pool. This can be undesirable if multiple [`Context`]s exist at the same time or if a process will be
/// long-lived and will create and destroy multiple [`Context`]s. To control better thread spawning, an externally
/// owned [`ThreadPoolRef`] can be injected in each [`Context`]. For example:
///
/// ```
/// # #[derive(Copy, Clone)]
/// # struct ThreadPool;
/// # impl ThreadPool { fn new() -> Self { ThreadPool } }
/// #
/// # struct Context;
/// # impl Context {
/// #   fn new_without_multi_threading() -> Self { Context }
/// #   fn set_thread_pool(&mut self, thread_pool: ThreadPool) {}
/// # }
/// #
/// # let requests = vec![0, 1, 2, 3, 4, 5];
/// # fn process_request(request: usize, context: Context) {}
/// #
/// let thread_pool = ThreadPool::new();
/// for request in requests {
///   let mut context = Context::new_without_multi_threading();
///   context.set_thread_pool(thread_pool);
///   process_request(request, context);
/// }
/// ```
///
/// Note that [`Context`] uses interior mutability internally (via [`RefCell`]) to allow for thread-safe access
/// to the underlying MLIR context.
#[derive(Debug)]
pub struct Context<'t> {
    /// [`RefCell`] holding the handle that represents this [`Context`] in the MLIR C API. In MLIR contexts are shared
    /// globally during the construction of a program/module. We use interior mutability in order to model that safely
    /// in Rust.
    pub(crate) handle: RefCell<MlirContext>,

    /// [`ContextThreadPool`] used by this [`Context`].
    pub(crate) thread_pool: ContextThreadPool<'t>,
}

impl<'t> Context<'t> {
    /// Borrows the underlying [`MlirContext`] handle immutably. The returned handle can be used when interacting
    /// with the MLIR C API.
    pub fn borrow(&self) -> Ref<'_, MlirContext> {
        self.handle.borrow()
    }

    /// Borrows the underlying [`MlirContext`] handle mutably. The returned handle can be used when interacting
    /// with the MLIR C API.
    pub fn borrow_mut(&self) -> RefMut<'_, MlirContext> {
        self.handle.borrow_mut()
    }

    /// Creates a new MLIR [`Context`] with the default [`Threading`] option (i.e., [`Threading::Enabled`]).
    pub fn new() -> Self {
        Self { handle: RefCell::new(unsafe { mlirContextCreate() }), thread_pool: ContextThreadPool::Owned }
    }

    /// Creates a new MLIR [`Context`] with [`Threading`] disabled.
    pub fn new_without_multi_threading() -> Self {
        Self {
            handle: RefCell::new(unsafe { mlirContextCreateWithThreading(false) }),
            thread_pool: ContextThreadPool::None,
        }
    }

    /// Consumes this [`Context`] and returns a new one replacing it, using the provided [`ThreadPoolRef`]. Note that
    /// the new [`Context`] is effectively the original/provided [`Context`] but with its [`ThreadPoolRef`] replaced.
    pub fn with_thread_pool(self, thread_pool: ThreadPoolRef) -> Context {
        unsafe {
            // Forget the original [`Context`] instance since we are going to be re-using its handle.
            let mut manually_dropped_self = ManuallyDrop::new(self);
            std::ptr::drop_in_place(&mut manually_dropped_self.thread_pool);
            mlirContextSetThreadPool(*manually_dropped_self.handle.borrow_mut(), thread_pool.to_c_api());
            Context {
                handle: std::ptr::read(&manually_dropped_self.handle),
                thread_pool: ContextThreadPool::Borrowed(thread_pool),
            }
        }
    }

    /// Returns a reference to the [`ThreadPoolRef`] associated with this [`Context`],
    /// or [`None`] if multithreading is disabled.
    pub fn thread_pool(&self) -> Option<ThreadPoolRef<'t>> {
        match self.thread_pool {
            ContextThreadPool::None => None,
            ContextThreadPool::Owned => {
                Some(unsafe { ThreadPoolRef::from_c_api(mlirContextGetThreadPool(*self.handle.borrow())) })
            }
            ContextThreadPool::Borrowed(thread_pool) => Some(thread_pool),
        }
    }

    /// Returns the number of threads in the thread pool associated with this [`Context`]. If multithreading
    /// is disabled, then this function will return `1`.
    pub fn thread_count(&self) -> usize {
        unsafe { mlirContextGetNumThreads(*self.handle.borrow()) as usize }
    }

    /// Enables multithreading for this [`Context`].
    pub fn enable_multi_threading(&self) {
        unsafe { mlirContextEnableMultithreading(*self.handle.borrow_mut(), true) }
    }

    /// Disables multithreading for this [`Context`].
    pub fn disable_multi_threading(&self) {
        unsafe { mlirContextEnableMultithreading(*self.handle.borrow_mut(), false) }
    }

    /// Returns a reference to this [`Context`].
    pub fn as_ref<'c>(&'c self) -> ContextRef<'c, 't> {
        self.into()
    }
}

impl Drop for Context<'_> {
    fn drop(&mut self) {
        unsafe { mlirContextDestroy(*self.handle.borrow_mut()) };
    }
}

impl Default for Context<'static> {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for Context<'_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirContextEqual(*self.handle.borrow(), *other.handle.borrow()) }
    }
}

impl<'c, 't> PartialEq<ContextRef<'c, 't>> for Context<'t> {
    fn eq(&self, &other: &ContextRef<'c, 't>) -> bool {
        unsafe { mlirContextEqual(*self.handle.borrow(), other.handle) }
    }
}

impl<'t> Eq for Context<'t> {}

/// Immutable reference to a [`Context`] allowing borrowed access from types that are owned by a [`Context`].
#[derive(Copy, Clone, Debug)]
pub struct ContextRef<'c, 't> {
    /// Handle that represents this [`ContextRef`] in the MLIR C API.
    handle: MlirContext,

    /// [`PhantomData`] used to track the lifetime of the underlying [`Context`].
    owner: PhantomData<&'c Context<'t>>,
}

impl<'c, 't> ContextRef<'c, 't> {
    /// Constructs a new [`ContextRef`] from the provided [`MlirContext`] handle
    /// that came from a function in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn from_c_api(handle: MlirContext) -> Self {
        Self { handle, owner: PhantomData }
    }

    /// Returns the [`MlirContext`] that corresponds to this [`ContextRef`]
    /// and which can be passed to functions in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn to_c_api(&self) -> MlirContext {
        self.handle
    }

    /// Refer to [`Context::thread_count`] for information on this function.
    pub fn thread_count(&self) -> usize {
        unsafe { mlirContextGetNumThreads(self.handle) as usize }
    }
}

impl<'c, 't> PartialEq for ContextRef<'c, 't> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirContextEqual(self.handle, other.handle) }
    }
}

impl<'c, 't> PartialEq<Context<'t>> for ContextRef<'c, 't> {
    fn eq(&self, other: &Context<'t>) -> bool {
        unsafe { mlirContextEqual(self.handle, *other.handle.borrow()) }
    }
}

impl<'c, 't> Eq for ContextRef<'c, 't> {}

impl<'c, 't> From<&'c Context<'t>> for ContextRef<'c, 't> {
    fn from(value: &'c Context<'t>) -> Self {
        Self { handle: *value.handle.borrow(), owner: PhantomData }
    }
}

/// Helper trait that is semantically equivalent to [`From`] in the presence of a [`Context`]. This is useful
/// for types that can be constructed from other types, but only in the presence of a [`Context`] instance
/// (e.g., [`Attribute`](crate::Attribute)s, [`Type`](crate::Type)s, etc.).
pub trait FromWithContext<'c, 't, T> {
    fn from_with_context(value: T, context: &'c Context<'t>) -> Self;
}

impl<'c, 't, T> FromWithContext<'c, 't, T> for T {
    fn from_with_context(value: T, _context: &'c Context<'t>) -> Self {
        value
    }
}

/// Helper trait that is to [`FromWithContext`] what [`Into`] is to [`From`].
pub trait IntoWithContext<'c, 't, T> {
    fn into_with_context(self, context: &'c Context<'t>) -> T;
}

impl<'c, 't, T, V: FromWithContext<'c, 't, T>> IntoWithContext<'c, 't, V> for T {
    fn into_with_context(self, context: &'c Context<'t>) -> V {
        V::from_with_context(self, context)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::ThreadPool;

    use super::*;

    #[test]
    fn test_context() {
        // With multi-threading enabled (the default), the thread count should be at least 1.
        let context = Context::default();
        assert!(context.thread_count() > 1);
        assert!(context.thread_pool().is_some());

        // With multi-threading disabled, the thread count should be 1.
        let context = Context::new_without_multi_threading();
        assert!(context.thread_pool().is_none());
        assert_eq!(context.thread_count(), 1);

        // It should go up after enabling multi-threading.
        context.enable_multi_threading();
        assert!(context.thread_count() > 1);
        assert!(context.as_ref().thread_count() > 1);

        // It should go back down after disabling multi-threading.
        context.disable_multi_threading();
        assert_eq!(context.thread_count(), 1);

        // Using an external and shared thread pool works.
        let thread_pool = ThreadPool::new();
        let context = Context::new_without_multi_threading();
        let context = context.with_thread_pool(ThreadPoolRef::from(&thread_pool));
        assert!(context.thread_pool().is_some());
        assert!(context.thread_count() > 1);
    }

    #[test]
    fn test_context_equality() {
        let context_0 = Context::new();
        let context_1 = Context::new();
        assert_eq!(context_0, context_0);
        assert_ne!(context_0, context_1);
        assert_ne!(context_1, context_0);
        assert_eq!(context_1, context_1);
    }

    #[test]
    fn test_context_ref() {
        let context_0 = Context::new();
        let context_1 = Context::new();
        let context_0_ref = ContextRef::from(&context_0);
        let context_1_ref = ContextRef::from(&context_1);
        assert_eq!(context_0.as_ref(), context_0_ref);
        assert_eq!(context_0, context_0_ref);
        assert_eq!(context_0_ref, context_0);
        assert_eq!(context_0_ref, context_0_ref);
        assert_ne!(context_0, context_1_ref);
        assert_ne!(context_0_ref, context_1);
        assert_ne!(context_0_ref, context_1_ref);
        assert_ne!(context_1, context_0_ref);
        assert_ne!(context_1_ref, context_0);
        assert_ne!(context_1_ref, context_0_ref);
        assert_eq!(context_1, context_1_ref);
        assert_eq!(context_1_ref, context_1);
        assert_eq!(context_1_ref, context_1_ref);

        // C API round-trip check.
        let context_0_ref_handle = unsafe { context_0_ref.to_c_api() };
        let reconstructed_context_0_ref = unsafe { ContextRef::from_c_api(context_0_ref_handle) };
        assert_eq!(reconstructed_context_0_ref, context_0_ref);
    }
}
