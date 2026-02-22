use std::cell::RefCell;
use std::rc::Rc;

use ryft_xla_sys::bindings::{
    MlirDialectRegistry, mlirContextAppendDialectRegistry, mlirContextCreateWithRegistry,
    mlirContextGetAllowUnregisteredDialects, mlirContextGetNumLoadedDialects, mlirContextGetNumRegisteredDialects,
    mlirContextGetOrLoadDialect, mlirContextIsRegisteredOperation, mlirContextLoadAllAvailableDialects,
    mlirContextSetAllowUnregisteredDialects, mlirDialectHandleInsertDialect, mlirDialectHandleLoadDialect,
    mlirDialectHandleRegisterDialect, mlirDialectRegistryCreate, mlirDialectRegistryDestroy, mlirRegisterAllDialects,
};

use crate::{Context, ContextRef, ContextThreadPool, Dialect, DialectHandle, StringRef, Threading};

/// MLIR [`DialectRegistry`]s map [`Dialect`] namespaces to constructors for the matching [`Dialect`]s. This allows for
/// decoupling the list of dialects that are "available" from the list of dialects that have already been loaded in a
/// [`Context`]. The MLIR parser, in particular, will lazily load [`Dialect`]s in the [`Context`] as operations are
/// encountered.
#[derive(Clone, Debug)]
pub struct DialectRegistry {
    /// Reference-counted pointer to the handle that represents this [`DialectRegistry`] in the MLIR C API.
    handle: Rc<RefCell<MlirDialectRegistry>>,
}

impl DialectRegistry {
    /// Creates a new (empty) [`DialectRegistry`].
    pub fn new() -> Self {
        Self { handle: Rc::new(RefCell::new(unsafe { mlirDialectRegistryCreate() })) }
    }

    /// Creates a new [`DialectRegistry`] that contains all built-in/upstream (from MLIR) dialects and extensions.
    pub fn new_with_all_built_in_dialects() -> Self {
        let registry = Self::new();
        registry.insert_all_built_in_dialects();
        registry
    }

    /// Inserts the [`Dialect`] that corresponds to the provided [`DialectHandle`] into this [`DialectRegistry`].
    pub fn insert(&self, dialect: DialectHandle) {
        unsafe { mlirDialectHandleInsertDialect(dialect.to_c_api(), *self.handle.borrow_mut()) }
    }

    /// Inserts all built-in/upstream (from MLIR) dialects and extensions to this [`DialectRegistry`].
    pub fn insert_all_built_in_dialects(&self) {
        unsafe { mlirRegisterAllDialects(*self.handle.borrow_mut()) }
    }
}

impl Default for DialectRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for DialectRegistry {
    fn drop(&mut self) {
        // Only destroy the underlying dialect registry when this is wrapper holds the last reference to it.
        if let Some(handle) = Rc::get_mut(&mut self.handle) {
            let handle = handle.borrow_mut();
            if !handle.ptr.is_null() {
                unsafe { mlirDialectRegistryDestroy(*handle) };
            }
        }
    }
}

impl<'t> Context<'t> {
    /// Creates a new MLIR [`Context`] with all the [`Dialect`]s registered in the provided [`DialectRegistry`]
    /// preloaded and with the provided [`Threading`] option.
    pub fn new_with_registry(registry: &DialectRegistry, threading: Threading) -> Self {
        let threading_enabled = threading == Threading::Enabled;
        Self {
            handle: RefCell::new(unsafe {
                mlirContextCreateWithRegistry(*registry.handle.borrow(), threading_enabled)
            }),
            thread_pool: if threading_enabled { ContextThreadPool::Owned } else { ContextThreadPool::None },
        }
    }

    /// Registers the [`Dialect`] that corresponds to the provided [`DialectHandle`] in this [`Context`].
    ///
    /// Refer to the [official MLIR documentation](https://mlir.llvm.org/getting_started/Faq/#registered-loaded-dependent-whats-up-with-dialects-management)
    /// for information on the differences between [`Context::register_dialect`] and [`Context::load_dialect`].
    pub fn register_dialect<'c>(&self, dialect: DialectHandle<'c, 't>)
    where
        Self: 'c,
    {
        unsafe { mlirDialectHandleRegisterDialect(dialect.to_c_api(), *self.handle.borrow_mut()) }
    }

    /// Appends the contents of the provided [`DialectRegistry`] (i.e., all of the [`Dialect`]s that it contains)
    /// to the [`DialectRegistry`] that is associated with this [`Context`].
    ///
    /// Refer to the [official MLIR documentation](https://mlir.llvm.org/getting_started/Faq/#registered-loaded-dependent-whats-up-with-dialects-management)
    /// for information on the differences between [`Context::register_dialect`] and [`Context::load_dialect`].
    pub fn register_dialects(&self, dialect_registry: &DialectRegistry) {
        unsafe { mlirContextAppendDialectRegistry(*self.handle.borrow_mut(), *dialect_registry.handle.borrow()) }
    }

    /// Loads the [`Dialect`] that corresponds to the provided [`DialectHandle`] and which has already been
    /// registered in this [`Context`]. Note that if the corresponding [`Dialect`] is already loaded, then this
    /// function will simply return the already loaded instance. If the [`Dialect`] is not registered with this
    /// [`Context`] then this function will return [`None`]. You must use [`Context::register_dialect`] to register
    /// [`Dialect`]s in this [`Context`] before you can load them.
    ///
    /// Refer to the [official MLIR documentation](https://mlir.llvm.org/getting_started/Faq/#registered-loaded-dependent-whats-up-with-dialects-management)
    /// for information on the differences between [`Context::register_dialect`] and [`Context::load_dialect`].
    pub fn load_dialect<'c>(&self, dialect: DialectHandle<'c, 't>) -> Option<Dialect<'c, 't>>
    where
        Self: 'c,
    {
        unsafe { Dialect::from_c_api(mlirDialectHandleLoadDialect(dialect.to_c_api(), *self.handle.borrow_mut())) }
    }

    /// Gets or loads the [`Dialect`] with the provided name that has already been registered in this [`Context`].
    /// Note that if the corresponding [`Dialect`] is already loaded, then this function will simply return the already
    /// loaded instance. If the [`Dialect`] is not registered with this [`Context`] then this function will return
    /// [`None`]. You must use [`Context::register_dialect`] to register [`Dialect`]s in this [`Context`] before you can
    /// load them.
    ///
    /// Refer to the [official MLIR documentation](https://mlir.llvm.org/getting_started/Faq/#registered-loaded-dependent-whats-up-with-dialects-management)
    /// for information on the differences between [`Context::register_dialect`] and [`Context::load_dialect`].
    pub fn load_dialect_by_name<'c, 'n>(&self, name: &'n str) -> Option<Dialect<'c, 't>>
    where
        Self: 'c,
    {
        unsafe {
            Dialect::from_c_api(mlirContextGetOrLoadDialect(
                *self.handle.borrow_mut(),
                StringRef::from(name).to_c_api(),
            ))
        }
    }

    /// Returns a number of registered [`Dialect`]s in this [`Context`].
    /// A registered [`Dialect`] will be loaded if needed by the MLIR parser.
    pub fn registered_dialect_count(&self) -> usize {
        unsafe { mlirContextGetNumRegisteredDialects(*self.handle.borrow()).cast_unsigned() }
    }

    /// Returns a number of loaded [`Dialect`]s by this [`Context`].
    pub fn loaded_dialect_count(&self) -> usize {
        unsafe { mlirContextGetNumLoadedDialects(*self.handle.borrow()).cast_unsigned() }
    }

    /// Eagerly loads all available [`Dialect`]s that have been registered with this [`Context`], making them
    /// available for use in IR construction.
    pub fn load_all_available_dialects(&self) {
        unsafe { mlirContextLoadAllAvailableDialects(*self.handle.borrow_mut()) }
    }

    /// Returns `true` if this [`Context`] allows unregistered [`Dialect`]s.
    pub fn allows_unregistered_dialects(&self) -> bool {
        unsafe { mlirContextGetAllowUnregisteredDialects(*self.handle.borrow()) }
    }

    /// Configures this [`Context`] to allow unregistered [`Dialect`]s.
    pub fn allow_unregistered_dialects(&self) {
        unsafe { mlirContextSetAllowUnregisteredDialects(*self.handle.borrow_mut(), true) }
    }

    /// Configures this [`Context`] to disallow unregistered [`Dialect`]s.
    pub fn disallow_unregistered_dialects(&self) {
        unsafe { mlirContextSetAllowUnregisteredDialects(*self.handle.borrow_mut(), false) }
    }

    /// Returns `true` if there is an operation registered in this [`Context`] that has the provided (fully-qualified)
    /// name (i.e., `<dialect_name>.<operation_name>` like `arith.add`, for example). This function will return `true`
    /// if the corresponding [`Dialect`] is loaded in this [`Context`] and an operation with that name is registered
    /// within that [`Dialect`].
    pub fn is_registered<S: AsRef<str>>(&self, operation_name: S) -> bool {
        unsafe {
            mlirContextIsRegisteredOperation(*self.handle.borrow(), StringRef::from(operation_name.as_ref()).to_c_api())
        }
    }
}

impl ContextRef<'_, '_> {
    /// Refer to [`Context::registered_dialect_count`] for information on this function.
    pub fn registered_dialect_count(&self) -> usize {
        unsafe { mlirContextGetNumRegisteredDialects(self.to_c_api()).cast_unsigned() }
    }

    /// Refer to [`Context::loaded_dialect_count`] for information on this function.
    pub fn loaded_dialect_count(&self) -> usize {
        unsafe { mlirContextGetNumLoadedDialects(self.to_c_api()).cast_unsigned() }
    }

    /// Refer to [`Context::allows_unregistered_dialects`] for information on this function.
    pub fn allows_unregistered_dialects(&self) -> bool {
        unsafe { mlirContextGetAllowUnregisteredDialects(self.to_c_api()) }
    }

    /// Refer to [`Context::is_registered`] for information on this function.
    pub fn is_registered<S: AsRef<str>>(&self, operation_name: S) -> bool {
        unsafe {
            mlirContextIsRegisteredOperation(self.to_c_api(), StringRef::from(operation_name.as_ref()).to_c_api())
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_dialect_registry_insert() {
        let registry = DialectRegistry::default();

        // Insert multiple dialects without crashing.
        registry.insert(DialectHandle::gpu());
        registry.insert(DialectHandle::linalg());

        // Verify that we can create a context with this registry.
        let _ = Context::new_with_registry(&registry, Threading::Disabled);
    }

    #[test]
    fn test_dialect_registry_insert_all_built_in_dialects() {
        let registry = DialectRegistry::new();
        registry.insert_all_built_in_dialects();

        // Verify that we can create a context with this registry.
        let context = Context::new_with_registry(&registry, Threading::Disabled);
        assert!(context.registered_dialect_count() > 0);
    }

    #[test]
    fn test_dialect_registry_new_with_all_built_in_dialects() {
        let registry = DialectRegistry::new_with_all_built_in_dialects();

        // Verify that we can create a context with this registry.
        let context = Context::new_with_registry(&registry, Threading::Disabled);
        assert!(context.registered_dialect_count() > 0);
    }

    #[test]
    fn test_dialect_registry_clone() {
        let registry = DialectRegistry::new();
        registry.insert(DialectHandle::gpu());
        let _ = Context::new_with_registry(&registry, Threading::Disabled);
        let _ = Context::new_with_registry(&registry.clone(), Threading::Disabled);
    }

    #[test]
    fn test_context_register_dialect() {
        let context = Context::new();
        let initial_dialect_count = context.registered_dialect_count();
        context.register_dialect(DialectHandle::gpu());
        assert_eq!(context.registered_dialect_count(), initial_dialect_count + 1);
        context.register_dialect(DialectHandle::gpu());
        assert_eq!(context.registered_dialect_count(), initial_dialect_count + 1);
        context.register_dialect(DialectHandle::linalg());
        assert_eq!(context.registered_dialect_count(), initial_dialect_count + 2);
    }

    #[test]
    fn test_context_register_dialects() {
        let context = Context::new();
        let initial_dialect_count = context.registered_dialect_count();
        let registry = DialectRegistry::new();
        registry.insert(DialectHandle::gpu());
        registry.insert(DialectHandle::linalg());
        registry.insert(DialectHandle::sparse_tensor());
        context.register_dialects(&registry);
        assert_eq!(context.registered_dialect_count(), initial_dialect_count + 3);
        context.register_dialects(&registry);
        assert_eq!(context.as_ref().registered_dialect_count(), initial_dialect_count + 3);
    }

    #[test]
    fn test_context_load_dialect() {
        let context = Context::new();
        let dialect_0 = DialectHandle::gpu();
        let dialect_1 = DialectHandle::linalg();
        context.register_dialect(dialect_0);
        let dialect_2 = context.load_dialect(dialect_0);
        assert!(dialect_2.is_some());
        assert_eq!(dialect_2.unwrap().namespace().unwrap(), "gpu");
        assert_eq!(context.loaded_dialect_count(), 3);
        let dialect_3 = context.load_dialect(dialect_1);
        assert!(dialect_3.is_some());
        assert_eq!(dialect_3.unwrap().namespace().unwrap(), "linalg");
        assert_eq!(context.loaded_dialect_count(), 10);
        let dialect_4 = context.load_dialect(dialect_0);
        assert_eq!(dialect_2, dialect_4);
        assert_eq!(context.loaded_dialect_count(), 10);
    }

    #[test]
    fn test_context_load_dialect_by_name() {
        let context = Context::new();

        // Try to load an unregistered dialect by name (should return [`None`]).
        assert_eq!(context.load_dialect_by_name("gpu"), None);

        // Register and then load the `gpu` dialect by name.
        context.register_dialect(DialectHandle::gpu());
        let dialect_0 = context.load_dialect_by_name("gpu");
        assert!(dialect_0.is_some());
        assert_eq!(dialect_0.unwrap().namespace().unwrap(), "gpu");
        let dialect_1 = context.load_dialect_by_name("gpu");
        assert_eq!(dialect_0, dialect_1);
        assert_eq!(context.loaded_dialect_count(), 3);
    }

    #[test]
    fn test_context_load_all_available_dialects() {
        let context = Context::new();
        let initial_dialect_count = context.loaded_dialect_count();
        context.register_dialect(DialectHandle::gpu());
        context.register_dialect(DialectHandle::linalg());
        context.register_dialect(DialectHandle::sparse_tensor());
        assert_eq!(context.loaded_dialect_count(), initial_dialect_count);
        context.load_all_available_dialects();
        assert!(context.as_ref().loaded_dialect_count() > initial_dialect_count);
    }

    #[test]
    fn test_context_allows_unregistered_dialects() {
        let context = Context::new();
        assert_eq!(context.allows_unregistered_dialects(), false);
        context.allow_unregistered_dialects();
        assert_eq!(context.as_ref().allows_unregistered_dialects(), true);
        context.disallow_unregistered_dialects();
        assert_eq!(context.allows_unregistered_dialects(), false);
    }

    #[test]
    fn test_context_is_registered() {
        let context = Context::new();
        context.register_dialect(DialectHandle::func());
        context.load_dialect(DialectHandle::func());
        assert_eq!(context.is_registered("func.func"), true);
        assert_eq!(context.as_ref().is_registered("func.return"), true);
        assert_eq!(context.is_registered("nonexistent.op"), false);
    }
}
