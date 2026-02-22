use std::cell::RefCell;
use std::fmt::Display;

use ryft_xla_sys::bindings::{
    MlirSymbolTable, mlirSymbolTableDestroy, mlirSymbolTableErase, mlirSymbolTableInsert, mlirSymbolTableLookup,
};

use crate::{Attribute, StringAttributeRef, StringRef};

use super::{Operation, OperationRef};

/// Symbol tables are a fundamental mechanism in MLIR for managing and organizing named entities (i.e., symbols) within
/// the IR structure. They provide a way to define, reference, and manage symbols across different scopes in an MLIR
/// program.
///
/// # What is a Symbol Table?
///
/// A symbol table is an [`Operation`] that can contain other operations with the
/// [`SymbolTable`](crate::operations::traits::SymbolTable) trait (i.e., operations that can define symbols). In MLIR,
/// symbols are named entities that can be references from other parts of the IR. The symbol table acts as a scope that
/// owns these symbol definitions.
///
/// [`Operation`]s that implement the [`SymbolTable`](crate::operations::traits::SymbolTable) trait can contain
/// symbols. A built-in MLIR example is [`ModuleOperation`](crate::ModuleOperation), but custom operations can also be
/// symbol tables. Furthermore, symbol tables can be *nested*, creating hierarchical scoping rules similar to
/// traditional programming languages. [`Operation`]s that define symbols in [`SymbolTable`]s have two attributes
/// describing those symbols:
///   - `sym_name`: [`StringAttributeRef`] defining the symbol name.
///   - `sym_visibility`: [`SymbolVisibilityAttributeRef`](crate::SymbolVisibilityAttributeRef)
///     defining the [`SymbolVisibility`].
///
/// # How are Symbol Tables Used?
///
/// *Symbol Definition:* When you define a function, global variable, or other named entity, it becomes a symbol in the
/// nearest enclosing symbol table. For example, in the following MLIR code:
///
/// ```mlir
/// module {
///   func.func @my_function(%arg0: i32) -> i32 {
///     return %arg0 : i32
///   }
///   memref.global "private" @global_var : memref<10xi32>
/// }
/// ```
///
/// `my_function` is declared as a symbol in the enclosing MLIR module.
///
/// *Symbol References:* Other operations can reference these symbols using
/// [`SymbolReferenceAttributeRef`](crate::SymbolRefAttributeRef)s. For example, in the following MLIR code:
///
/// ```mlir
/// func.func @caller() {
///   %0 = arith.constant 42 : i32
///   %1 = func.call @my_function(%0) : (i32) -> i32
///   %2 = memref.get_global @global_var : memref<10xi32>
///   return
/// }
/// ```
///
/// the `caller` function implementation references the `my_function` symbol.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/#symbol-table)
/// for more information.
pub struct SymbolTable<'o, 'c, 't> {
    /// Handle that represents this [`SymbolTable`] in the MLIR C API.
    handle: RefCell<MlirSymbolTable>,

    /// Reference to the [`Operation`] that owns this [`SymbolTable`].
    operation: OperationRef<'o, 'c, 't>,
}

impl Drop for SymbolTable<'_, '_, '_> {
    fn drop(&mut self) {
        let handle = self.handle.borrow_mut();
        if !handle.ptr.is_null() {
            unsafe { mlirSymbolTableDestroy(*handle) }
        }
    }
}

impl<'o, 'c, 't: 'c> SymbolTable<'o, 'c, 't> {
    /// Constructs a new [`SymbolTable`] from the provided [`MlirSymbolTable`]
    /// that came from a function in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn from_c_api<O: Operation<'o, 'c, 't>>(handle: MlirSymbolTable, operation: &O) -> Option<Self> {
        if handle.ptr.is_null() {
            None
        } else {
            Some(Self { handle: RefCell::new(handle), operation: operation.as_ref() })
        }
    }

    /// Returns the [`MlirSymbolTable`] that corresponds to this [`SymbolTable`] and which can be passed to functions
    /// in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn to_c_api(&self) -> &RefCell<MlirSymbolTable> {
        &self.handle
    }

    /// Looks up a symbol with the provided `name` in this [`SymbolTable`] and returns a reference to the [`Operation`]
    /// that corresponds to that symbol. If the symbol cannot be found, then this function will return [`None`].
    pub fn lookup<'r, S: AsRef<str>>(&self, name: S) -> Option<OperationRef<'r, 'c, 't>>
    where
        'o: 'r,
    {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.operation.context().borrow();
        unsafe {
            OperationRef::from_c_api(
                mlirSymbolTableLookup(*self.handle.borrow(), StringRef::from(name.as_ref()).to_c_api()),
                self.operation.context(),
            )
        }
    }

    /// Inserts the provided [`Operation`] into this [`SymbolTable`], and renames it as necessary to avoid collisions.
    /// Returns the name of the symbol after insertion, if successful. Note that this does not move the operation
    /// itself into the [`Block`](crate::Block) of the operation that this [`SymbolTable`] is associated with;
    /// this should be done separately.
    pub fn insert<'p, O: Operation<'p, 'c, 't>>(&self, operation: &O) -> StringRef<'c>
    where
        'c: 'p,
    {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.operation.context().borrow_mut();
        unsafe {
            StringAttributeRef::from_c_api(
                mlirSymbolTableInsert(*self.handle.borrow_mut(), operation.to_c_api()),
                self.operation.context(),
            )
            .unwrap()
            .string()
        }
    }

    /// Removes the provided [`Operation`] from this [`SymbolTable`] and erases it.
    pub fn erase<'p, O: Operation<'p, 'c, 't>>(&self, operation: O)
    where
        'c: 'p,
    {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.operation.context().borrow_mut();
        unsafe { mlirSymbolTableErase(*self.handle.borrow_mut(), operation.to_c_api()) }
        std::mem::forget(operation);
    }
}

/// Represents the types of visibility that an MLIR symbol may have.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Default)]
pub enum SymbolVisibility {
    /// Represents symbols that are public and may be referenced anywhere internal or external to the visible
    /// references in the IR.
    #[default]
    Public,

    /// Represents symbols that are private and may only be referenced by
    /// [`SymbolRefAttributeRef`](crate::SymbolRefAttributeRef)s that are local to the [`Operation`]s
    /// within the current [`SymbolTable`].
    Private,

    /// Represents symbols that are visible to the current IR, which may include [`Operation`]s in [`SymbolTable`]s
    /// above the one that owns the current symbol. [`SymbolVisibility::Nested`] allows for referencing a symbol
    /// outside of its own [`SymbolTable`], while retaining the ability to observe all of its uses.
    Nested,
}


impl Display for SymbolVisibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Public => write!(f, "public"),
            Self::Private => write!(f, "private"),
            Self::Nested => write!(f, "nested"),
        }
    }
}

impl<'s> TryFrom<&'s str> for SymbolVisibility {
    type Error = String;

    fn try_from(value: &'s str) -> Result<Self, Self::Error> {
        match value {
            "public" => Ok(Self::Public),
            "private" => Ok(Self::Private),
            "nested" => Ok(Self::Nested),
            _ => Err(format!("'{value}' is not a valid MLIR symbol visibility")),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::ManuallyDrop;

    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Block, Context, DialectHandle, SymbolTable, ValueRef};

    use super::*;

    #[test]
    fn test_symbol_visibility() {
        assert_eq!(SymbolVisibility::default(), SymbolVisibility::Public);
        assert_eq!(SymbolVisibility::Public.to_string(), "public");
        assert_eq!(SymbolVisibility::Private.to_string(), "private");
        assert_eq!(SymbolVisibility::Nested.to_string(), "nested");
        assert_eq!(SymbolVisibility::try_from("public"), Ok(SymbolVisibility::Public));
        assert_eq!(SymbolVisibility::try_from("private"), Ok(SymbolVisibility::Private));
        assert_eq!(SymbolVisibility::try_from("nested"), Ok(SymbolVisibility::Nested));
        assert!(SymbolVisibility::try_from("invalid").is_err());
    }

    #[test]
    fn test_symbol_table() {
        let context = Context::new();
        context.load_dialect(DialectHandle::func());

        // Create a module that contains a function named `test_function`.
        let location = context.unknown_location();
        let module = context.module(location);
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            block.append_operation(func::r#return::<ValueRef, _>(&[], location));
            func::func("test_function", func::FuncAttributes::default(), block.into(), location)
        });
        let module = module.as_operation();

        // Look up a symbol that exists in the test module.
        let symbol_table = module.new_symbol_table();
        let function = symbol_table.lookup("test_function");
        assert!(function.is_some());
        assert_eq!(function.unwrap().name().as_str().unwrap(), "func.func");

        // Look up a non-existent symbol.
        let non_existent = symbol_table.lookup("non_existent");
        assert!(non_existent.is_none());

        // Attempt to insert a symbol with the same name as the function that is already there.
        let block = context.block_with_no_arguments();
        let op = func::func("test_function", func::FuncAttributes::default(), block.into(), location);
        let name = symbol_table.insert(&op);
        assert_eq!(name.as_str().unwrap(), "test_function_0");
        let function = symbol_table.lookup("test_function_0");
        assert!(function.is_some());
        assert_eq!(function.unwrap().name().as_str().unwrap(), "func.func");

        // Remove the `test_function_0` symbol.
        symbol_table.erase(op);
        let non_existent = symbol_table.lookup("test_function_0");
        assert!(non_existent.is_none());
    }

    #[test]
    fn test_symbol_table_c_api() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let symbol_table = module.as_operation().new_symbol_table();
        let symbol_table = ManuallyDrop::new(symbol_table);
        let handle = unsafe { *symbol_table.to_c_api().borrow() };
        let symbol_table = unsafe { super::SymbolTable::from_c_api(handle, &symbol_table.operation) };
        assert!(symbol_table.is_some());
        let non_existent = symbol_table.unwrap().lookup("non_existent");
        assert!(non_existent.is_none());
    }
}
