use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::path::Path;

use ryft_xla_sys::bindings::{
    MlirModule, mlirLoadIRDLDialects, mlirModuleCreateEmpty, mlirModuleCreateParse, mlirModuleCreateParseFromFile,
    mlirModuleDestroy, mlirModuleEqual, mlirModuleFromOperation, mlirModuleGetBody, mlirModuleGetOperation,
    mlirModuleHashValue,
};

use crate::dialects::builtin::{DetachedModuleOperation, ModuleOperationRef};
use crate::{BlockRef, Context, Location, Operation, StringRef};

/// [`Module`]s in MLIR represent top-level [`Operation`]s (i.e., they are instances of a built-in operation type).
/// A [`Module`] contains a single [`Region`](crate::Region) which itself contains a single [`Block`](crate::Block).
/// That [`Block`](crate::Block) can contain any number of [`Operation`]s and does not have a
/// [`Block::terminator`](crate::Block::terminator). [`Operation`]s within this [`Region`](crate::Region) cannot
/// implicitly capture [`Value`](crate::Value)s defined outside the [`Module`]. [`Module`]s can also have an optional
/// name which can be used to refer to them from within other [`Operation`]s.
///
/// Note that this struct is basically a wrapper over a [`ModuleOperation`](crate::ModuleOperation) that is provided
/// by the MLIR C API for convenience.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#builtinmodule-moduleop)
/// for more information.
pub struct Module<'c, 't> {
    /// Handle that represents this [`Module`] in the MLIR C API.
    handle: MlirModule,

    /// [`Context`] associated with this [`Module`].
    context: &'c Context<'t>,
}

impl<'c, 't> Module<'c, 't> {
    /// Constructs a new [`Module`] from the provided [`MlirModule`] that came from a function in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn from_c_api(handle: MlirModule, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() { None } else { Some(Self { handle, context }) }
    }

    /// Returns the [`MlirModule`] that corresponds to this [`Module`] and which can be passed to functions
    /// in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn to_c_api(&self) -> MlirModule {
        self.handle
    }

    /// Returns a reference to the [`Context`] that this [`Module`] is associated with.
    pub fn context(&self) -> &'c Context<'t> {
        self.context
    }

    /// Returns a reference to the [`Block`](crate::Block) that represents the body of this [`Module`]
    /// (i.e., the only [`Block`](crate::Block) it contains).
    pub fn body<'m>(&'m self) -> BlockRef<'m, 'c, 't> {
        unsafe { BlockRef::from_c_api(mlirModuleGetBody(self.handle), self.context).unwrap() }
    }

    /// Returns a [`ModuleOperationRef`] that refers to this [`Module`].
    pub fn as_operation<'m>(&'m self) -> ModuleOperationRef<'m, 'c, 't> {
        unsafe { ModuleOperationRef::from_c_api(mlirModuleGetOperation(self.handle), self.context).unwrap() }
    }

    /// Verifies this [`Module`] (as in, checks if it is well-defined) and returns `true` if the verification passes.
    pub fn verify(&self) -> bool {
        self.as_operation().verify()
    }

    /// Loads all [IRDL](https://mlir.llvm.org/docs/Dialects/IRDL) [`Dialect`](crate::Dialect)s in this [`Module`],
    /// registering the dialects in the [`Context`] associated with this module.
    pub fn load_irdl_dialects(&self) -> bool {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context.borrow_mut();
        unsafe { mlirLoadIRDLDialects(self.to_c_api()).value != 0 }
    }
}

impl<'c, 't: 'c> PartialEq<Module<'c, 't>> for Module<'c, 't> {
    fn eq(&self, other: &Module<'c, 't>) -> bool {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context.borrow();
        // Note that this function only checks for whether the two module handles point to the same underlying
        // module. It does not perform a deep comparison of the contents of these modules.
        unsafe { mlirModuleEqual(self.handle, other.to_c_api()) }
    }
}

impl Eq for Module<'_, '_> {}

impl Hash for Module<'_, '_> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        unsafe { mlirModuleHashValue(self.handle).hash(hasher) }
    }
}

impl Display for Module<'_, '_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.as_operation(), f)
    }
}

impl Debug for Module<'_, '_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.as_operation(), f)
    }
}

impl Drop for Module<'_, '_> {
    fn drop(&mut self) {
        unsafe { mlirModuleDestroy(self.handle) }
    }
}

impl<'c, 't> From<DetachedModuleOperation<'c, 't>> for Module<'c, 't> {
    fn from(value: DetachedModuleOperation<'c, 't>) -> Self {
        unsafe {
            let module = Module::from_c_api(mlirModuleFromOperation(value.to_c_api()), value.context()).unwrap();
            std::mem::forget(value);
            module
        }
    }
}

impl<'c, 't> From<Module<'c, 't>> for DetachedModuleOperation<'c, 't> {
    fn from(value: Module<'c, 't>) -> Self {
        unsafe {
            let operation = DetachedModuleOperation::from_c_api(mlirModuleGetOperation(value.handle), value.context());
            std::mem::forget(value);
            operation.unwrap()
        }
    }
}

impl<'m, 'c, 't> From<&'m Module<'c, 't>> for ModuleOperationRef<'m, 'c, 't> {
    fn from(value: &'m Module<'c, 't>) -> Self {
        value.as_operation()
    }
}

impl<'t> Context<'t> {
    /// Creates a new (empty) [`Module`] at the specified [`Location`].
    pub fn module<'c, L: Location<'c, 't>>(&'c self, location: L) -> Module<'c, 't> {
        unsafe { Module::from_c_api(mlirModuleCreateEmpty(location.to_c_api()), self).unwrap() }
    }

    /// Parses a [`Module`] from the provided string representation. Returns [`None`] if MLIR fails to parse
    /// the provided string into a [`Module`] (this function will also emit diagnostics if that happens).
    pub fn parse_module<'c, S: AsRef<str>>(&'c self, source: S) -> Option<Module<'c, 't>> {
        unsafe {
            let source = std::ffi::CString::new(source.as_ref()).unwrap();
            Module::from_c_api(
                mlirModuleCreateParse(*self.handle.borrow_mut(), StringRef::from(source.as_c_str()).to_c_api()),
                self,
            )
        }
    }

    /// Parses a [`Module`] from the the contents of the file at the specified [`Path`]. Returns [`None`] if MLIR fails
    /// to parse the provided string into a [`Module`] (this function will also emit diagnostics if that happens).
    pub fn parse_module_from_file<'c>(&'c self, path: &Path) -> Option<Module<'c, 't>> {
        unsafe {
            let path = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
            Module::from_c_api(
                mlirModuleCreateParseFromFile(*self.handle.borrow_mut(), StringRef::from(path.as_c_str()).to_c_api()),
                self,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::{builtin, func};
    use crate::{Block, Context, DialectHandle, OneRegion, Operation, Region, Symbol, SymbolVisibility, ValueRef};

    use super::*;

    #[test]
    fn test_module() {
        let context = Context::new();
        context.load_dialect(DialectHandle::func());
        let location = context.unknown_location();

        // Modules with no operations.
        let module_0 = context.module(context.file_location("foo", 42, 42));
        assert!(module_0.verify());
        assert_eq!(module_0.context(), &context);
        assert_eq!(module_0.body().operations().count(), 0);
        assert_eq!(module_0.as_operation().name().as_str().unwrap(), "builtin.module");

        // Module with one operation.
        let mut block = context.block_with_no_arguments();
        block.append_operation(func::r#return::<ValueRef, _>(&[], location));
        let function = func::func("test_function", func::FuncAttributes::default(), block.into(), location);
        module_0.body().append_operation(function);
        assert!(module_0.verify());
        assert_eq!(module_0.body().operations().count(), 1);
    }

    #[test]
    fn test_module_equality_and_hashing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::func());
        let module_0 = context.module(context.file_location("foo", 42, 42));
        let module_1 = context.module(context.file_location("foo", 42, 42));
        assert_eq!(module_0, module_0);
        assert_ne!(module_0, module_1);
        assert_ne!(module_1, module_0);
        assert_eq!(module_1, module_1);

        let mut map = HashMap::new();
        map.insert(&module_0, "module_0");
        map.insert(&module_1, "module_1");
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&module_0), Some(&"module_0"));
        assert_eq!(map.get(&module_1), Some(&"module_1"));
    }

    #[test]
    fn test_module_display_and_debug() {
        let context = Context::new();
        context.load_dialect(DialectHandle::func());
        let module = context.module(context.file_location("foo", 42, 42));
        assert_eq!(format!("{}", module), "module {\n}\n");
        assert_eq!(format!("{:?}", module), "ModuleOperationRef[module {\n}\n]");
    }

    #[test]
    fn test_module_casting() {
        let context = Context::new();
        context.load_dialect(DialectHandle::func());
        let module = context.module(context.file_location("foo", 42, 42));
        assert_eq!(module.as_operation().body_region().blocks().next(), Some(module.body()));
        assert_eq!(module.as_operation(), module.as_operation().as_ref());
    }

    #[test]
    fn test_module_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::func());

        // Parse a good module.
        let module = context.parse_module(indoc! {"
            module {
              func.func @test() {
                func.return
              }
            }
        "});
        assert!(module.is_some());
        let module = module.unwrap();
        assert!(module.verify());
        assert_eq!(module.body().operations().count(), 1);

        // Trying parsing a bad module.
        let module = context.parse_module("module{");
        assert!(module.is_none());
    }

    #[test]
    fn test_module_parsing_from_file() {
        let context = Context::new();
        context.load_dialect(DialectHandle::func());

        // Create a temporary file with our MLIR code.
        let module_path = std::env::temp_dir().join("test_module.mlir");
        std::fs::write(
            &module_path,
            indoc! {"
                module {
                  func.func @test() {
                    func.return
                  }
                }
            "},
        )
        .unwrap();
        let module = context.parse_module_from_file(&module_path);
        assert!(module.is_some());
        let module = module.unwrap();
        assert!(module.verify());
        assert_eq!(module.body().operations().count(), 1);

        // Clean up our temporary file.
        std::fs::remove_file(module_path).unwrap();

        // Try parsing from a bad path.
        let module = context.parse_module_from_file(Path::new("/nonexistent/path/to/file.mlir"));
        assert!(module.is_none());
    }

    #[test]
    fn test_module_from_operation() {
        let context = Context::new();
        let location = context.unknown_location();

        let mut region = context.region();
        region.append_block(context.block_with_no_arguments());
        let module = Module::from(builtin::module(region, location).unwrap());
        let operation = ModuleOperationRef::from(&module);

        assert!(operation.verify());
        assert_eq!(operation.symbol_name(), None);
        assert_eq!(operation.symbol_visibility(), SymbolVisibility::Public);
        assert_eq!(operation.to_string(), "module {\n}\n");

        let mut region = context.region();
        region.append_block(context.block_with_no_arguments());
        let named_module = builtin::named_module("test_module", SymbolVisibility::Private, region, location).unwrap();
        let module = Module::from(named_module);
        let operation = ModuleOperationRef::from(&module);

        assert!(operation.verify());
        assert_eq!(operation.symbol_name().map(|name| name.as_str().unwrap()), Some("test_module"));
        assert_eq!(operation.symbol_visibility(), SymbolVisibility::Private);
        assert_eq!(operation.to_string(), "module @test_module attributes {sym_visibility = \"private\"} {\n}\n")
    }

    #[test]
    fn test_module_to_operation() {
        let context = Context::new();
        let module = context.module(context.unknown_location());
        let operation = DetachedModuleOperation::from(module);
        assert!(operation.verify());
        assert_eq!(operation.name().as_str().unwrap(), "builtin.module");
    }

    #[test]
    fn test_module_set_attribute() {
        let context = Context::new();
        let module = context.module(context.unknown_location());
        ModuleOperationRef::from(&module).set_attribute("sym_name", context.string_attribute("foo"));
        assert!(ModuleOperationRef::from(&module).verify());

        let attribute = ModuleOperationRef::from(&module).attribute("sym_name");
        assert!(attribute.is_some());
        assert_eq!(attribute.unwrap().to_string(), "\"foo\"");
    }

    #[test]
    fn test_module_load_irdl_dialects() {
        // We intentionally try to register multiple times to ensure that the operation is idempotent.
        let context = Context::new();
        let module = context.module(context.unknown_location());
        for _ in 0..1000 {
            assert!(module.load_irdl_dialects());
        }
    }
}
