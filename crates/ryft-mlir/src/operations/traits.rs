use std::collections::HashMap;

use ryft_xla_sys::bindings::{
    MlirOperation, mlirSymbolTableCreate, mlirSymbolTableReplaceAllSymbolUses, mlirSymbolTableWalkSymbolTables,
};

use crate::{
    ArrayAttributeRef, Attribute, AttributeRef, BlockRef, Context, DictionaryAttributeRef, FlatSymbolRefAttributeRef,
    FromWithContext, FunctionTypeRef, IntoWithContext, LogicalResult, OperationBuilder, Region, RegionRef,
    StringAttributeRef, StringRef, SymbolVisibility, SymbolVisibilityAttributeRef, Type, TypeAndAttributes,
    TypeAttributeRef, TypeRef, Value, ValueRef,
};

use super::{Operation, OperationRef};

/// Trait that represents [`Operation`]s that define a new scope for the purposes of polyhedral optimization and the
/// affine dialect in particular. Any [SSA](https://en.wikipedia.org/wiki/Static_single-assignment_form) values with
/// [`IndexTypeRef`](crate::IndexTypeRef) that either dominate such operations, or are defined at the top-level of such
/// operations, or appear as region arguments for such operations automatically become valid symbols for the polyhedral
/// scope defined by that [`Operation`]. As a result, such SSA values could be used as the operands or index operands of
/// various affine dialect operations like `affine.for`, `affine.load`, and `affine.store`. The polyhedral scope defined
/// by an operation with this trait includes all operations in its region excluding operations that are nested inside
/// other operations that themselves have this trait.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Traits/#affinescope) and
/// [codebase](https://mlir.llvm.org/doxygen/classmlir_1_1OpTrait_1_1AffineScope.html) for more information.
pub trait AffineScope<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

/// Trait that represents [`Operation`]s that are always speculatively executable.
///
/// Refer to the [official MLIR codebase](
/// https://mlir.llvm.org/doxygen/structmlir_1_1OpTrait_1_1AlwaysSpeculatableImplTrait.html) for more information.
pub trait AlwaysSpeculatable<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

/// Trait that represents [`Region`]-holding [`Operation`]s that define a new scope for automatic allocations
/// (i.e., allocations that are freed when control is transferred back from the operation's region). Any operations
/// performing such allocations (e.g., for `memref.alloca`) will have their allocations automatically freed at their
/// closest enclosing operation with this trait.
///
/// Refer to the [official MLIR codebase](
/// https://mlir.llvm.org/doxygen/classmlir_1_1OpTrait_1_1AutomaticAllocationScope.html) for more information.
pub trait AutomaticAllocationScope<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

/// Name of the [`Attribute`] that is used to store [`Callee`]s in [`Call`]s where it is a [`Callee::Symbol`].
pub const CALLEE_ATTRIBUTE: &'static str = "callee";

/// Callee in a [`Call`] [`Operation`].
pub enum Callee<'o, 'c, 't> {
    /// Symbol that corresponds to a [`Callable`].
    Symbol(FlatSymbolRefAttributeRef<'c, 't>),

    /// [`Value`] that corresponds to a [`Region`] of a lambda-like [`Operation`].
    Value(ValueRef<'o, 'c, 't>),
}

/// Trait that represents [`Operation`]s which transfer control from one sub-routine to another. These operations may
/// be traditional direct calls (e.g., `call @foo`) or indirect calls to other operations (e.g., `call_indirect %foo`).
///
/// Note that any operation that implements this trait must *not* also implement [`Callable`].
///
/// Refer to `CallOpInterface` in the [official MLIR documentation](
/// https://mlir.llvm.org/docs/Interfaces/#callinterfaces) for more information.
pub trait Call<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the callee of this [`Call`] (e.g., a symbol that corresponds to a [`Function`]).
    fn callee(&self) -> Callee<'o, 'c, 't>;
}

/// Trait that represents [`Operation`]s that are effectively potential sub-routines, and may be targets for [`Call`]
/// [`Operation`]s. These operations may be traditional functional operations (e.g., `func @foo(...)`), as well as
/// function-producing operations (e.g., `%foo = dialect.create_function(...)`). These operations may only contain a
/// single [`Region`], or subroutine.
///
/// Refer to `CallableOpInterface` in the [official MLIR documentation](
/// https://mlir.llvm.org/docs/Interfaces/#callinterfaces) for more information.
pub trait Callable<'o, 'c: 'o, 't: 'c>: HasCallableArgumentAndResultAttributes<'o, 'c, 't> {
    /// Returns `true` if this [`Callable`] is external (i.e., if it has no body).
    fn is_external_callable(&self) -> bool {
        self.callable_region().is_none()
    }

    /// Returns the [`Region`] on this [`Callable`] that is callable or [`None`] if this is an _external_ callable
    /// object (e.g., an external function).
    fn callable_region(&self) -> Option<RegionRef<'o, 'c, 't>> {
        self.region(0)
    }
}

/// Trait that represents [`Operation`]s that behave like constants. These are non-side effecting [`Operation`]s
/// that can always be folded to [`Attribute`] values.
pub trait ConstantLike<'o, 'c: 'o, 't: 'c>: OneResult<'o, 'c, 't> + ZeroOperands<'o, 'c, 't> {}

/// Name of the [`Attribute`] that is used to store [`FunctionTypeRef`]s in [`Function`]s.
pub const FUNCTION_TYPE_ATTRIBUTE: &'static str = "function_type";

/// Trait that represents [`Operation`]s that behave like functions. In particular, these operations:
///
///   - are [`Symbol`]s,
///   - must have a single [`Region`] which may be comprised of multiple [`Block`](crate::Block)s that correspond to the
///     function body (i.e., they are [`Callable`]). When this [`Region`] is empty, the [`Operation`] corresponds to an
///     _external_ function. When this [`Region`] is not empty, the leading arguments of the first
///     [`Block`](crate::Block) in this [`Region`] are treated as the function arguments.
///
/// Refer to `FunctionOpInterface` in the [official MLIR documentation](
/// https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/FunctionInterfaces.td#L24)
/// for more information.
pub trait Function<'o, 'c: 'o, 't: 'c>: Symbol<'o, 'c, 't> + Callable<'o, 'c, 't> {
    /// Returns the symbol name of this [`Callable`], which can be used to refer to it from other [`Operation`]s.
    fn function_name(&self) -> StringRef<'c> {
        self.symbol_name().unwrap()
    }

    /// Returns the [`Type`] of this [`Function`].
    fn function_type(&self) -> FunctionTypeRef<'c, 't> {
        self.attribute(FUNCTION_TYPE_ATTRIBUTE)
            .unwrap()
            .cast::<TypeAttributeRef>()
            .unwrap()
            .r#type()
            .cast()
            .unwrap()
    }

    /// Returns the number of arguments of this [`Function`].
    fn function_argument_count(&self) -> usize {
        self.function_type().input_count()
    }

    /// Returns [`Vec`] that contains the [`Type`]s of the arguments of this [`Function`].
    fn function_argument_types(&self) -> Vec<TypeRef<'c, 't>> {
        self.function_type().inputs().collect()
    }

    /// Returns the [`Type`] of the argument at the `index`-pth position in the arguments list of this [`Function`].
    ///
    /// Note that this function will panic if the provided index is out of bounds.
    fn function_argument_type(&self, index: usize) -> TypeRef<'c, 't> {
        self.function_type().input(index)
    }

    /// Returns the number of results of this [`Function`].
    fn function_result_count(&self) -> usize {
        self.function_type().output_count()
    }

    /// Returns [`Vec`] that contains the [`Type`]s of the results of this [`Function`].
    fn function_result_types(&self) -> Vec<TypeRef<'c, 't>> {
        self.function_type().outputs().collect()
    }

    /// Returns the [`Type`] of the argument at the `index`-pth position in the results list of this [`Function`].
    ///
    /// Note that this function will panic if the provided index is out of bounds.
    fn function_result_type(&self, index: usize) -> TypeRef<'c, 't> {
        self.function_type().output(index)
    }

    /// Convenient helper that returns the combined results of [`Function::function_argument_types`] and
    /// [`Function::callable_argument_attributes`](HasCallableArgumentAndResultAttributes::callable_argument_attributes)
    /// combined in a single [`Vec`].
    fn function_argument_types_and_attributes<'r>(&'r self) -> Vec<TypeAndAttributes<'o, 't, 'c>> {
        let argument_types = self.function_argument_types();
        let argument_attributes = self.callable_argument_attributes();
        argument_types
            .into_iter()
            .zip(argument_attributes.into_iter())
            .map(|(r#type, attributes)| {
                if attributes.is_empty() {
                    TypeAndAttributes { r#type, attributes: None }
                } else {
                    TypeAndAttributes { r#type, attributes: Some(attributes) }
                }
            })
            .collect()
    }

    /// Convenient helper that returns the combined results of [`Function::function_result_types`] and
    /// [`Function::callable_result_attributes`](HasCallableArgumentAndResultAttributes::callable_result_attributes)
    /// combined in a single [`Vec`].
    fn function_result_types_and_attributes<'r>(&'r self) -> Vec<TypeAndAttributes<'o, 't, 'c>> {
        let result_types = self.function_result_types();
        let result_attributes = self.callable_result_attributes();
        result_types
            .into_iter()
            .zip(result_attributes.into_iter())
            .map(|(r#type, attributes)| {
                if attributes.is_empty() {
                    TypeAndAttributes { r#type, attributes: None }
                } else {
                    TypeAndAttributes { r#type, attributes: Some(attributes) }
                }
            })
            .collect()
    }

    /// Returns `true` if this is an _external_ [`Function`].
    fn is_external_function(&self) -> bool {
        self.function_body().is_empty()
    }

    /// Returns the body of this [`Function`].
    fn function_body(&self) -> RegionRef<'o, 'c, 't> {
        self.callable_region().unwrap()
    }
}

/// Name of the [`Attribute`] that is used to store argument attributes in [`HasCallableArgumentAndResultAttributes`].
pub const ARGUMENT_ATTRIBUTES_ATTRIBUTE: &'static str = "arg_attrs";

/// Name of the [`Attribute`] that is used to store result attributes in [`HasCallableArgumentAndResultAttributes`].
pub const RESULT_ATTRIBUTES_ATTRIBUTE: &'static str = "res_attrs";

/// Trait that represents [`Operation`]s that have _callable_ argument and result [`Attribute`]s.
///
/// Refer to `ArgAndResultAttrsOpInterface` in the [official MLIR codebase](
/// https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/CallInterfaces.td#L22)
/// for more information.
pub trait HasCallableArgumentAndResultAttributes<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the [`Attribute`]s attached to the arguments of this [`Operation`]. Specifically, when such attributes
    /// exist, this function returns a [`Vec`] with one element per argument. Each element is a [`HashMap`] that maps
    /// attribute names to their values for the corresponding argument of this operation.
    fn callable_argument_attributes(&self) -> Vec<HashMap<StringRef<'c>, AttributeRef<'c, 't>>> {
        self.attribute(ARGUMENT_ATTRIBUTES_ATTRIBUTE)
            .map(|attribute| {
                attribute
                    .cast::<ArrayAttributeRef>()
                    .unwrap()
                    .elements()
                    .map(|element| element.cast::<DictionaryAttributeRef>().unwrap().into())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns the [`Attribute`]s attached to the results of this [`Operation`]. Specifically, when such attributes
    /// exist, this function returns a [`Vec`] with one element per result. Each element is a [`HashMap`] that maps
    /// attribute names to their values for the corresponding result of this operation.
    fn callable_result_attributes(&self) -> Vec<HashMap<StringRef<'c>, AttributeRef<'c, 't>>> {
        self.attribute(RESULT_ATTRIBUTES_ATTRIBUTE)
            .map(|attribute| {
                attribute
                    .cast::<ArrayAttributeRef>()
                    .unwrap()
                    .elements()
                    .map(|element| element.cast::<DictionaryAttributeRef>().unwrap().into())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Adds the provided argument [`Attribute`]s to the provided [`OperationBuilder`], assuming that it is building
    /// an [`Operation`] with the [`HasCallableArgumentAndResultAttributes`] trait.
    fn add_callable_argument_attributes<'a, 's: 'a>(
        builder: OperationBuilder<'c, 't>,
        argument_attributes: impl Iterator<Item = &'a Option<HashMap<StringRef<'s>, AttributeRef<'c, 't>>>>,
    ) -> OperationBuilder<'c, 't>
    where
        'c: 'a,
    {
        let attribute: ArrayAttributeRef<'c, 't> = argument_attributes
            .map(|attributes| {
                DictionaryAttributeRef::from_with_context(
                    attributes.as_ref().unwrap_or(&HashMap::new()),
                    builder.context(),
                )
            })
            .collect::<Vec<_>>()
            .as_slice()
            .into_with_context(builder.context());
        builder.add_attribute(ARGUMENT_ATTRIBUTES_ATTRIBUTE, attribute)
    }

    /// Adds the provided result [`Attribute`]s to the provided [`OperationBuilder`], assuming that it is building
    /// an [`Operation`] with the [`HasCallableArgumentAndResultAttributes`] trait.
    fn add_callable_result_attributes<'a, 's: 'a>(
        builder: OperationBuilder<'c, 't>,
        result_attributes: impl Iterator<Item = &'a Option<HashMap<StringRef<'s>, AttributeRef<'c, 't>>>>,
    ) -> OperationBuilder<'c, 't>
    where
        'c: 'a,
    {
        let attribute: ArrayAttributeRef<'c, 't> = result_attributes
            .map(|attributes| {
                DictionaryAttributeRef::from_with_context(
                    attributes.as_ref().unwrap_or(&HashMap::new()),
                    builder.context(),
                )
            })
            .collect::<Vec<_>>()
            .as_slice()
            .into_with_context(builder.context());
        builder.add_attribute(RESULT_ATTRIBUTES_ATTRIBUTE, attribute)
    }
}

/// Trait that represents [`Operation`]s that only contain "graph" [`Region`]s (i.e., regions that, in contrast to
/// [SSACFG](https://mlir.llvm.org/docs/LangRef/#control-flow-and-ssacfg-regions) regions, do not require the
/// [SSA](https://en.wikipedia.org/wiki/Static_single-assignment_form)-dominance property to hold).
pub trait HasOnlyGraphRegion<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

/// Trait which signals that the [`Region`]s of an [`Operation`] are known to be "isolated from above" (i.e., that they
/// do not capture, or reference, [SSA](https://en.wikipedia.org/wiki/Static_single-assignment_form) values defined above
/// the region scope).
///
/// This means that the following is invalid if `foo.region_op` is [`IsolatedFromAbove`]:
///
/// ```mlir
/// %result = arith.constant 10 : i32
/// foo.region_op {
///   foo.yield %result : i32
/// }
/// ```
///
/// This trait is an important structural property of the IR, and enables operations to have [`Pass`](crate::Pass)es
/// scheduled under them.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Traits/#isolatedfromabove)
/// for more information.
pub trait IsolatedFromAbove<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

/// Trait that represents [`Operation`]s that are known to be
/// [terminators](https://mlir.llvm.org/docs/LangRef/#control-flow-and-ssacfg-regions).
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Traits/#terminator)
/// for more information.
pub trait IsTerminator<'o, 'c: 'o, 't: 'c>: SingleBlockRegions<'o, 'c, 't> {}

/// Trait that represents [`Operation`]s that can be safely normalized in the presence of
/// [`MemRefTypeRef`](crate::MemRefTypeRef) values with non-identity maps.
pub trait MemRefsNormalizable<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

/// Trait that represents [`Operation`]s which have no effect on memory but which may have undefined behavior.
pub trait NoMemoryEffect<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

/// Trait that represents [`Operation`]s whose [`Region`]s do not have any arguments.
pub trait NoRegionArguments<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

/// Trait that represents [`Operation`]s that do not require their [`Region`]s to have a
/// [terminator](https://mlir.llvm.org/docs/LangRef/#control-flow-and-ssacfg-regions) [`Block`](crate::Block).
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Traits/#terminator)
/// for more information.
pub trait NoTerminator<'o, 'c: 'o, 't: 'c>: SingleBlockRegions<'o, 'c, 't> {}

/// Trait that represents [`Operation`]s that have a single operand.
pub trait OneOperand<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the input of this [`Operation`].
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }
}

/// Trait that represents [`Operation`]s that contain a single [`Region`].
pub trait OneRegion<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the _body_ region of this [`Operation`], if it has one.
    fn body_region(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }
}

/// Trait that represents [`Operation`]s that has a single result/output [`Value`].
pub trait OneResult<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the single result/output of this [`Operation`].
    ///
    /// Note that this function is called `output` and not `result` in order to avoid
    /// a name collision with [`Operation::result`].
    fn output(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the [`Type`] of the single result/output of this [`Operation`].
    ///
    /// Note that this function is called `output_type` and not `result_type` in order to avoid
    /// a name collision with [`Operation::result_type`].
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap().cast().unwrap()
    }
}

/// Trait that represents [`Operation`]s that are always speculatively executable and which have no memory effects.
/// Such operations are always legal to hoist or sink.
pub trait Pure<'o, 'c: 'o, 't: 'c>: AlwaysSpeculatable<'o, 'c, 't> + NoMemoryEffect<'o, 'c, 't> {}

/// Trait that represents [`Operation`]s that behave like return statements from functions.
pub trait ReturnLike<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

/// Trait that represents [`Operation`]s that contain [`Region`]s that are either empty or contain a single
/// [`Block`](crate::Block).
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Traits/#single-block-region)
/// for more information.
pub trait SingleBlockRegions<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

/// Trait that represents [`Operation`]s that contain a single [`Block`](crate::Block) (and which we refer to as the _body_
/// of the [`Operation`]).
pub trait SingleBlock<'o, 'c: 'o, 't: 'c>:
    Operation<'o, 'c, 't> + OneRegion<'o, 'c, 't> + SingleBlockRegions<'o, 'c, 't>
{
    /// Returns a reference to the [`Block`](crate::Block) that represents the body of this [`Operation`]
    /// (i.e., the only [`Block`](crate::Block) it contains).
    fn body(&self) -> BlockRef<'o, 'c, 't> {
        self.body_region().blocks().next().unwrap()
    }
}

/// Name of the [`Attribute`] that is used to store symbol names that are compatible with
/// [`SymbolTable`](crate::operations::symbol_table::SymbolTable)s.
pub const SYMBOL_NAME_ATTRIBUTE: &'static str = "sym_name";

/// Name of the [`Attribute`] that is used to store symbol visibilities that are compatible with
/// [`SymbolTable`](crate::operations::symbol_table::SymbolTable)s.
pub const SYMBOL_VISIBILITY_ATTRIBUTE: &'static str = "sym_visibility";

/// Trait that represents [`Operation`]s that define symbols and which can be referred to from other [`Operation`]s
/// using those symbols. Refer to the documentation of [`SymbolTable`] for more information on symbols in MLIR.
///
/// Refer to `SymbolOpInterface` in the [official MLIR codebase](
/// https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/SymbolInterfaces.td#L23) for more information.
pub trait Symbol<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the symbol name of this [`Operation`], which can be used to refer to it from other [`Operation`]s.
    /// Note that if the name is optional and missing, this function will return [`None`].
    fn symbol_name(&self) -> Option<StringRef<'c>> {
        self.attribute(SYMBOL_NAME_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<StringAttributeRef<'c, 't>>().map(|attribute| attribute.string()))
    }

    /// Returns the [`SymbolVisibility`] of this [`Operation`]. If not specified, then [`SymbolVisibility::default()`]
    /// will be returned (which is [`SymbolVisibility::Public`]).
    fn symbol_visibility(&self) -> SymbolVisibility {
        self.attribute(SYMBOL_VISIBILITY_ATTRIBUTE)
            .and_then(|attribute| {
                attribute
                    .cast::<SymbolVisibilityAttributeRef<'c, 't>>()
                    .and_then(|attribute| attribute.visibility())
            })
            .unwrap_or(SymbolVisibility::default())
    }

    /// Returns `true` if this [`Symbol`]'s visibility is [`SymbolVisibility::Public`].
    fn is_public(&self) -> bool {
        self.symbol_visibility() == SymbolVisibility::Public
    }

    /// Returns `true` if this [`Symbol`]'s visibility is [`SymbolVisibility::Private`].
    fn is_private(&self) -> bool {
        self.symbol_visibility() == SymbolVisibility::Private
    }

    /// Returns `true` if this [`Symbol`]'s visibility is [`SymbolVisibility::Nested`].
    fn is_nested(&self) -> bool {
        self.symbol_visibility() == SymbolVisibility::Nested
    }

    /// Replaces all uses of this [`Symbol`] with `new_symbol` in the provided [`Operation`]
    /// by using [`SymbolTable::replace_symbol`].
    fn replace_all_uses<'r, S: Into<StringRef<'c>>, O: SymbolTable<'r, 'c, 't>>(
        &self,
        new_symbol: S,
        operation: O,
    ) -> LogicalResult
    where
        'c: 'r,
    {
        self.symbol_name()
            .map(|old_symbol| operation.replace_symbol(old_symbol, new_symbol))
            .unwrap_or(LogicalResult::failure())
    }
}

/// Trait that represents [`Operation`]s which define [`SymbolTable`](crate::operations::symbol_table::SymbolTable)s.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Traits/#symboltable)
/// for more information.
pub trait SymbolTable<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Creates a new [`SymbolTable`] for this operation.
    fn new_symbol_table(&self) -> super::symbol_table::SymbolTable<'o, 'c, 't> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe { super::symbol_table::SymbolTable::from_c_api(mlirSymbolTableCreate(self.to_c_api()), self).unwrap() }
    }

    /// Replaces all uses of `old_symbol` with `new_symbol` in this [`Operation`] and returns a [`LogicalResult`]
    /// that represents whether the replacement was successful. Note that this function will not traverse into nested
    /// [`SymbolTable`]s and it will fail atomically if there are any unknown operations that may potentially have the
    /// MLIR [`SymbolTable`] trait.
    fn replace_symbol<'old, 'new, Old: Into<StringRef<'old>>, New: Into<StringRef<'new>>>(
        &self,
        old_symbol: Old,
        new_symbol: New,
    ) -> LogicalResult {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe {
            LogicalResult::from_c_api(mlirSymbolTableReplaceAllSymbolUses(
                old_symbol.into().to_c_api(),
                new_symbol.into().to_c_api(),
                self.to_c_api(),
            ))
        }
    }

    /// Performs a walk over all [`SymbolTable`]s nested within this [`Operation`] (i.e., itself and all of its nested
    /// [`SymbolTable`]s), invoking `callback` on each operation it visits, passing it that operation along with a
    /// boolean signifying if the symbols within that symbol table can be treated as if all their uses within the IR
    /// are visible to the caller. `all_symbol_uses_visible` identifies whether all the symbol uses within `self`
    /// are visible.
    ///
    /// Note that this function does not support callbacks that mutate the associated [`Context`] and if such callbacks
    /// are used, they will result in runtime panics.
    fn walk_symbol_tables<F: FnMut(OperationRef<'o, 'c, 't>, bool)>(
        &self,
        all_symbol_uses_visible: bool,
        mut callback: F,
    ) {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();

        unsafe extern "C" fn c_api_callback<'o, 'c: 'o, 't: 'c, F: FnMut(OperationRef<'o, 'c, 't>, bool)>(
            operation: MlirOperation,
            all_symbol_uses_visible: bool,
            data: *mut std::ffi::c_void,
        ) {
            unsafe {
                let data = data as *mut (&mut F, &'c Context<'t>);
                let (ref mut callback, ref context) = *data;
                let operation = OperationRef::from_c_api(operation, context).unwrap();
                (callback)(operation, all_symbol_uses_visible)
            }
        }

        unsafe {
            mlirSymbolTableWalkSymbolTables(
                self.to_c_api(),
                all_symbol_uses_visible,
                Some(c_api_callback::<'o, 'c, 't, F>),
                &mut (&mut callback, self.context()) as *mut _ as *mut _,
            );
        }
    }
}

/// Trait that represents [`Operation`]s that have no operands (i.e., inputs).
pub trait ZeroOperands<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

/// Trait that represents [`Operation`]s that contain no [`Region`]s.
pub trait ZeroRegions<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

/// Trait that represents [`Operation`]s that are known to have no [`Block::successors`](crate::Block::successors).
pub trait ZeroSuccessors<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::{func, stable_hlo};
    use crate::{Block, Context, DialectHandle, Region, Size};

    use super::*;

    #[test]
    fn test_callable() {
        let context = Context::new();
        context.load_dialect(DialectHandle::func());
        let location = context.unknown_location();
        let function = func::func("external_function", func::FuncAttributes::default(), context.region(), location);
        assert!(!function.is_external_callable());
    }

    #[test]
    fn test_function() {
        let context = Context::new();
        let location = context.unknown_location();
        let f32_type = context.float32_type().as_ref();
        let f64_type = context.float64_type().as_ref();
        let function = func::func(
            "custom_function",
            func::FuncAttributes {
                visibility: SymbolVisibility::Private,
                arguments: vec![
                    f64_type.into(),
                    TypeAndAttributes {
                        r#type: f64_type.as_ref(),
                        attributes: Some(HashMap::from([
                            ("custom.4".into(), context.unit_attribute().as_ref()),
                            ("custom.2".into(), context.string_attribute("are").as_ref()),
                        ])),
                    },
                    f64_type.into(),
                ],
                results: vec![
                    TypeAndAttributes {
                        r#type: f64_type.as_ref(),
                        attributes: Some(HashMap::from([(
                            "custom.yes".into(),
                            context.boolean_attribute(false).as_ref(),
                        )])),
                    },
                    f32_type.into(),
                ],
                no_inline: true,
                llvm_emit_c_interface: true,
                other_attributes: HashMap::from([
                    ("custom.custom_1", context.unit_attribute().as_ref()),
                    ("custom.custom_2", context.string_attribute("42").as_ref()),
                ]),
            },
            context.region(),
            location,
        );
        assert_eq!(function.symbol_name().map(|name| name.as_str()), Some(Ok("custom_function")));
        assert_eq!(function.symbol_visibility(), SymbolVisibility::Private);
        assert!(!function.is_public());
        assert!(function.is_private());
        assert!(!function.is_nested());
        assert_eq!(function.function_name().as_str(), Ok("custom_function"));
        assert_eq!(
            function.function_type(),
            context.function_type(&[f64_type, f64_type, f64_type], &[f64_type, f32_type])
        );
        assert_eq!(function.function_argument_count(), 3);
        assert_eq!(function.function_argument_types(), vec![f64_type, f64_type, f64_type]);
        assert_eq!(function.function_argument_type(1), f64_type);
        assert_eq!(function.function_result_count(), 2);
        assert_eq!(function.function_result_types(), vec![f64_type, f32_type]);
        assert_eq!(function.function_result_type(1), f32_type);
        assert_eq!(
            function.function_argument_types_and_attributes(),
            vec![
                TypeAndAttributes::from(f64_type),
                TypeAndAttributes {
                    r#type: f64_type.as_ref(),
                    attributes: Some(HashMap::from([
                        ("custom.4".into(), context.unit_attribute().as_ref()),
                        ("custom.2".into(), context.string_attribute("are").as_ref()),
                    ])),
                },
                TypeAndAttributes::from(f64_type),
            ],
        );
        assert_eq!(
            function.function_result_types_and_attributes(),
            vec![
                TypeAndAttributes {
                    r#type: f64_type.as_ref(),
                    attributes: Some(HashMap::from([
                        ("custom.yes".into(), context.boolean_attribute(false).as_ref(),)
                    ])),
                },
                TypeAndAttributes::from(f32_type),
            ],
        );
        assert!(function.is_external_function());
    }

    #[test]
    fn test_one_operand() {
        let context = Context::new();
        let location = context.unknown_location();
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(3)], None, location).unwrap();
        let block = context.block(&[(tensor_type, location)]);
        let input = block.argument(0).unwrap();
        let op = stable_hlo::sign(input, location);
        assert_eq!(op.operand_count(), 1);
        let _ = op.input();
    }

    #[test]
    fn test_one_region() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let module = context.module(location);
        let module = module.as_operation();
        assert_eq!(module.region_count(), 1);
        assert!(module.region(0).is_some());
        assert!(module.region(1).is_none());
        assert_eq!(module.body_region().blocks().count(), 1);
    }

    #[test]
    fn test_one_result() {
        let context = Context::new();
        let location = context.unknown_location();
        let f64_type = context.float64_type();
        let i64_type = context.signless_integer_type(64);
        let function_type = context.function_type(&[f64_type], &[i64_type]);
        let op = func::constant("test_constant", function_type, location);
        assert_eq!(op.result_count(), 1);
        let _ = op.output();
        assert_eq!(op.output_type(), function_type);
    }

    #[test]
    #[allow(deprecated)]
    fn test_single_block() {
        let context = Context::new();
        let location = context.unknown_location();
        let i32_type = context.signless_integer_type(32);
        let tensor_type = context.tensor_type(i32_type, &[], None, location).unwrap();
        let mut block = context.block(&[(tensor_type, location)]);
        block.append_operation(stable_hlo::r#return(&[block.argument(0).unwrap()], location));
        let mut region = context.region();
        region.append_block(block);
        let block = context.block(&[(tensor_type, location)]);
        let op = stable_hlo::map(&[block.argument(0).unwrap()], &[], region, location);
        let _ = op.body();
    }

    #[test]
    fn test_symbol_table() {
        let context = Context::new();
        context.load_dialect(DialectHandle::func());
        let location = context.unknown_location();
        let module = context.module(location);
        let function_0_op = func::func(
            "function_0",
            func::FuncAttributes {
                results: vec![context.none_type().into()],
                visibility: SymbolVisibility::Private,
                ..func::FuncAttributes::default()
            },
            context.region(),
            location,
        );
        let function_1_op = func::func(
            "function_1",
            func::FuncAttributes {
                results: vec![context.none_type().into()],
                visibility: SymbolVisibility::Private,
                ..func::FuncAttributes::default()
            },
            context.region(),
            location,
        );
        module.body().append_operation(function_0_op);
        module.body().append_operation(function_1_op.clone());
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let call_op = block.append_operation(func::call(
                "function_0",
                func::CallProperties { results: vec![context.none_type().into()], ..func::CallProperties::default() },
                location,
            ));
            block.append_operation(func::r#return(&[call_op.result(0).unwrap()], location));
            func::func(
                "function_2",
                func::FuncAttributes { results: vec![context.none_type().into()], ..func::FuncAttributes::default() },
                block.into(),
                location,
            )
        });
        let module = module.as_operation();
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func private @function_0() -> none
                  func.func private @function_1() -> none
                  func.func @function_2() -> none {
                    %0 = call @function_0() : () -> none
                    return %0 : none
                  }
                }
            "},
        );

        // Verify what symbols exist.
        let symbol_table = module.new_symbol_table();
        assert!(symbol_table.lookup("function_0").is_some());
        assert!(symbol_table.lookup("function_1").is_some());
        assert!(symbol_table.lookup("function_2").is_some());
        assert!(symbol_table.lookup("function_3").is_none());

        // Try to do a replacement.
        assert!(module.replace_symbol("function_0", "function_1").is_success());
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func private @function_0() -> none
                  func.func private @function_1() -> none
                  func.func @function_2() -> none {
                    %0 = call @function_1() : () -> none
                    return %0 : none
                  }
                }
            "},
        );

        // Try to do another replacement.
        assert!(function_1_op.replace_all_uses("function_0", module).is_success());
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func private @function_0() -> none
                  func.func private @function_1() -> none
                  func.func @function_2() -> none {
                    %0 = call @function_0() : () -> none
                    return %0 : none
                  }
                }
            "},
        );

        // Checking that walking the symbol table works.
        let mut names = Vec::new();
        module.walk_symbol_tables(false, |op, _| {
            names.push(op.name().as_str().unwrap().to_string());
        });
        assert_eq!(names, vec!["builtin.module".to_string()]);
    }
}
