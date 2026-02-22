use std::path::Path;

use ryft_xla_sys::bindings::{
    MlirExecutionEngine, mlirExecutionEngineCreate, mlirExecutionEngineDestroy, mlirExecutionEngineDumpToObjectFile,
    mlirExecutionEngineInitialize, mlirExecutionEngineInvokePacked, mlirExecutionEngineLookup,
    mlirExecutionEngineLookupPacked, mlirExecutionEngineRegisterSymbol,
};

use crate::{LogicalResult, Module, StringRef};

/// Represents an MLIR compiler optimization level. These levels determine how much effort the compiler puts into
/// optimizing for runtime code execution speed while trading off compilation time and code size growth for it.
#[repr(i32)]
#[derive(Default)]
pub enum OptimizationLevel {
    /// Results in disabling as many compiler optimizations as possible but not all possible optimizations
    /// (e.g., inlining certain functions may be required for correctness in certain cases).
    O0 = 0,

    /// Results in just enabling quick optimizations that do not negatively impact "debuggability". This level is tuned
    /// to produce a result from the optimizer as quickly as possible and to avoid destroying "debuggability". This
    /// tends to be useful during software development where the compiled code may be immediately executed as part of
    /// testing. As a consequence, where possible, we would like to produce efficient-to-execute code, but not when that
    /// significantly slows down compilation or when it would prevent even basic debugging of the resulting binary. As
    /// an example, complex loop transformations such as versioning, vectorization, or fusion do not make sense here due
    /// to the degree to which the executed code differs from the source code, and the high compilation time cost.
    O1 = 1,

    /// Results in optimizations aimed at as fast execution as possible without triggering significant incremental
    /// compilation time costs or code size growth. The key idea is that optimizations at this level should "pay for
    /// themselves". So, if an optimization increases compile time by 5% or increases code size by 5% for a particular
    /// benchmark, that benchmark should also be one which sees a 5% runtime improvement. If the compile time or code
    /// size penalties happen on average across a diverse range of MLIR users' benchmarks, then the improvements should
    /// as well. Also, no matter what, the compilation time needs to not grow superlinearly with the size of input to
    /// MLIR so that users can control the runtime of the optimizer in this mode. This is expected to be a good default
    /// optimization level for the vast majority of users.
    #[default]
    O2 = 2,

    /// Results in optimizations aimed at as fast execution as possible with no constraints. This mode is significantly
    /// more aggressive in trading off compile time and code size for execution time improvements. The core idea is that
    /// this mode should include any optimization that helps execution time on balance across a diverse collection of
    /// benchmarks, even if it increases code size or compile time for some benchmarks without corresponding
    /// improvements in execution time. Despite being willing to trade more compile time off to get improved execution
    /// time, this mode still tries to avoid superlinear growth in order to make even significantly slower compile times
    /// at least scale reasonably. This does not preclude very substantial constant factor costs though.
    O3 = 3,
}


/// Just-In-Time (JIT) compilation-backed [`ExecutionEngine`] for MLIR (or rather LLVM specifically to be precise: more
/// on this in the example below). This engine takes [`Module`]s and assumes that their IR can be converted to LLVM IR.
/// For each function in the [`Module`] it creates a wrapper function with the following interface:
///
/// ```ignore
/// extern "C" fn _mlir_function_name(arguments_and_result: *mut *mut std::ffi::c_void) -> std::ffi::c_void;
/// ```
///
/// where `function_name` is replaced with the name of that function in the [`Module`]. The arguments are interpreted
/// as a list of pointers to the actual arguments of the function, followed by a pointer to its result. This allows the
/// [`ExecutionEngine`] to provide the caller with a generic function pointer that can be used to invoke the
/// JIT-compiled function (via [`InitializedExecutionEngine::invoke_function`]).
///
/// Note that in order for MLIR functions to be "invocable" by [`ExecutionEngine`]s, they must have been tagged with
/// the `llvm.emit_c_interface` attribute.
///
/// # Examples
///
/// The following is an example for how to construct and use an [`ExecutionEngine`], including, among other things,
/// the use of an external LLVM function to show how you can register an implementation for it:
///
/// ```rust
/// use ryft_mlir::dialects::builtin;
/// use ryft_mlir::{Context, DialectRegistry, ExecutionEngine, OptimizationLevel};
///
/// let context = Context::new();
/// context.register_dialects(&DialectRegistry::new_with_all_built_in_dialects());
/// context.load_all_available_dialects();
/// context.register_all_llvm_translations();
///
/// // Construct a [`Module`] by parsing some MLIR code.
/// let module = context
///     .parse_module(
///         r#"
///         module {
///             llvm.func @callback(i32) -> i32
///
///             func.func @example(%argument : i32) -> i32 attributes { llvm.emit_c_interface } {
///                 %callback_result = llvm.call @callback(%argument) : (i32) -> i32
///                 %result = arith.addi %argument, %callback_result : i32
///                 return %result : i32
///             }
///         }
///         "#,
///     ).unwrap();
///
/// // Convert the MLIR into LLVM IR to prevent undefined behavior when we invoke [`ExecutionEngine::initialize`].
/// let mut pass_manager = context.pass_manager();
/// pass_manager.add_pass(builtin::passes::create_conversion_to_llvm_pass());
/// assert!(pass_manager.run(&module.as_operation()).is_success());
///
/// // Construct a new [`ExecutionEngine`] for our [`Module`]. Note that we are setting `enable_object_dump` to `true`
/// // here so that we can illustrate the use of [`InitializedExecutionEngine::dump_to_object_file`] later on, but you
/// // will want to set that argument to `false` if you do not need to produce object files.
/// let mut engine = ExecutionEngine::for_module(&module, OptimizationLevel::O2, &[], true, false);
///
/// // Most of the following operations are unsafe and so we wrap everything that follows in an `unsafe` block.
/// unsafe {
///     // Register a Rust implementation for the `@callback` function. Note that the name and argument names of the
///     // Rust function here do not matter. However, the signature of the function does matter and if it is not
///     // consistent with the signature of `@callback` in the MLIR module, that can result in undefined behavior.
///     extern "C" fn callback(argument: i32) -> i32 {
///         argument * 5 + 1
///     }
///
///     engine.register_function("callback", callback as *mut _);
///
///     // Initialize our [`ExecutionEngine`]. This function can result in undefined behavior if the provided module
///     // uses anything other than LLVM IR.
///     let engine = engine.initialize();
///
///     // Invoke `@example`.
///     let mut argument = 42;
///     let mut result = -1;
///     let mut argument_and_result = [&mut argument as *mut i32 as *mut _, &mut result as *mut i32 as *mut _];
///     assert!(engine.invoke_function("example", argument_and_result.as_mut_ptr()).is_success());
///     assert_eq!(argument, 42);
///     assert_eq!(result, 253);
///
///     // Obtain function pointers in Rust and invoke them manually. Note that the type signatures in the `transmute`
///     // invocations must match exactly the underlying LLVM function type signatures. Mismatched type signatures
///     // result in undefined behavior.
///     let example_fn = std::mem::transmute::<_, extern "C" fn(i32) -> i32>(engine.get_function("example").unwrap());
///     assert_eq!(example_fn(42), 253);
///
///     // You can also obtain function pointers to registered external functions.
///     let callback_fn = std::mem::transmute::<_, extern "C" fn(i32) -> i32>(engine.get_function("callback").unwrap());
///     assert_eq!(callback_fn(42), 211);
///
///     // Dump the compiled object into a file.
///     let object_file = tempfile::NamedTempFile::new().unwrap();
///     let object_file_path = object_file.path();
///     engine.dump_to_object_file(&object_file_path);
///     assert!(object_file_path.exists());
///     assert!(std::fs::metadata(object_file_path).unwrap().len() > 0);
/// }
/// ```
pub struct ExecutionEngine {
    /// Handle that represents this [`ExecutionEngine`] in the MLIR C API.
    handle: MlirExecutionEngine,
}

impl ExecutionEngine {
    /// Creates a new [`ExecutionEngine`] for the provided [`Module`].
    ///
    /// To use an [`ExecutionEngine`] after constructing it, you must register any necessary function implementations
    /// using [`ExecutionEngine::register_function`] and then call [`ExecutionEngine::initialize`].
    ///
    /// # Parameters
    ///
    ///   * `module` - [`Module`] for which to construct an [`ExecutionEngine`]. This module is expected to be
    ///     "translatable" to LLVM IR (i.e., it must only contain operations in dialects that implement
    ///     [`LLVMTranslationDialectInterface`](https://mlir.llvm.org/doxygen/classmlir_1_1LLVMTranslationDialectInterface.html).
    ///     You can use [`create_conversion_to_llvm_pass`](crate::dialects::builtin::passes::create_conversion_to_llvm_pass)
    ///     to create a [`Pass`](crate::Pass) that translates non-LLVM IR [`Module`]s to [`Module`]s supported by
    ///     [`ExecutionEngine`]s using a [`PassManager`](crate::PassManager). Note that this function does not take
    ///     ownership of this [`Module`] and the lifetime of the resulting [`ExecutionEngine`] is independent of the
    ///     lifetime of this [`Module`].
    ///   * `optimization_level` - [`OptimizationLevel`] to use when compiling the provided [`Module`].
    ///   * `shared_library_paths` - [`Path`]s to shared libraries that will be loaded into the [`ExecutionEngine`].
    ///   * `enable_object_dump` - If `true`, then the compiler will cache the object generated for the provided
    ///     [`Module`]. The contents of the cache can then be dumped to a file via
    ///     [`InitializedExecutionEngine::dump_to_object_file`].
    ///   * `enable_pic` - If `true`, then the compiler will build
    ///     [Position-Independent Code (PIC)](https://en.wikipedia.org/wiki/Position-independent_code).
    pub fn for_module<'c, 't>(
        module: &Module<'c, 't>,
        optimization_level: OptimizationLevel,
        shared_library_paths: &[&Path],
        enable_object_dump: bool,
        enable_pic: bool,
    ) -> Self {
        Self {
            handle: unsafe {
                mlirExecutionEngineCreate(
                    module.to_c_api(),
                    optimization_level as i32,
                    shared_library_paths.len() as i32,
                    shared_library_paths
                        .iter()
                        .map(|&path| StringRef::from(path).to_c_api())
                        .collect::<Vec<_>>()
                        .as_ptr(),
                    enable_object_dump,
                    enable_pic,
                )
            },
        }
    }

    /// Registers a function implementation with this [`ExecutionEngine`]. This is used to register implementations of
    /// external LLVM functions. Refer to the documentation string of [`ExecutionEngine`] for an example of how to
    /// use this function.
    ///
    /// # Safety
    ///
    /// If you use a function name that does not exist in the underlying [`Module`] or if you pass a function
    /// implementation that has a different signature or calling convention than what is expected, then this function
    /// will result in undefined behavior.
    ///
    /// # Parameters
    ///
    ///   * `name` - Name of the function for which to register an implementation.
    ///   * `implementation` - Implementation of the function to use. This must be a pointer to an `extern "C"` function
    ///     whose signature matches the signature of the corresponding declaration in the underlying [`Module`].
    pub unsafe fn register_function<'s, S: Into<StringRef<'s>>>(
        &mut self,
        function: S,
        implementation: *mut std::ffi::c_void,
    ) {
        unsafe { mlirExecutionEngineRegisterSymbol(self.handle, function.into().to_c_api(), implementation) }
    }

    /// Initializes this [`ExecutionEngine`] returning an [`InitializedExecutionEngine`]. This includes running any
    /// global constructors specified by `llvm.mlir.global_ctors` (e.g., any kernel binaries compiled from `gpu.module`s
    /// will be loaded during this initialization phase). You must make sure that all symbols in the [`Module`] are
    /// resolvable before calling this function. This means that you must make sure to have included any relevant shared
    /// libraries or to have called [`ExecutionEngine::register_function`] as needed, before calling this function.
    ///
    /// # Safety
    ///
    /// If there are any unresolvable symbols or if the provided [`Module`] is not using LLVM IR, then this function
    /// will result in undefined behavior.
    pub unsafe fn initialize(self) -> InitializedExecutionEngine {
        unsafe {
            let handle = self.handle;
            mlirExecutionEngineInitialize(handle);
            std::mem::forget(self);
            InitializedExecutionEngine { handle }
        }
    }
}

impl Drop for ExecutionEngine {
    fn drop(&mut self) {
        unsafe { mlirExecutionEngineDestroy(self.handle) }
    }
}

/// Initialized [`ExecutionEngine`]. Refer to [`ExecutionEngine::initialize`] for information on what this means.
pub struct InitializedExecutionEngine {
    /// Handle that represents this [`InitializedExecutionEngine`] in the MLIR C API.
    handle: MlirExecutionEngine,
}

impl InitializedExecutionEngine {
    /// Looks up the provided function name in the compiled [`Module`] and, if found, returns an opaque pointer that
    /// can be used to invoke that function. Refer to the documentation string of [`ExecutionEngine`] for an example
    /// of how to use this function.
    ///
    /// Note that this function supports looking up functions both using their original names (e.g., `"example"` in the
    /// example from the documentation string of [`ExecutionEngine`]) and their wrapper function names
    /// (e.g., `"_mlir_example"` for that same example).
    ///
    /// # Parameters
    ///
    ///   * `name` - Name of the function to look up.
    pub unsafe fn get_function<'s, S: Into<StringRef<'s>>>(&self, name: S) -> Option<*mut std::ffi::c_void> {
        unsafe {
            let name: StringRef<'s> = name.into();
            let symbol = mlirExecutionEngineLookup(self.handle, name.to_c_api());
            if symbol.is_null() {
                let symbol = mlirExecutionEngineLookupPacked(self.handle, name.to_c_api());
                if symbol.is_null() { None } else { Some(symbol) }
            } else {
                Some(symbol)
            }
        }
    }

    /// Invokes a function in the underlying [`Module`] and returns a [`LogicalResult`] indicating whether the function
    /// invocation was successful. The underlying function must have been tagged with the `llvm.emit_c_interface`
    /// attribute. Refer to the documentation string of [`ExecutionEngine`] for an example of how to use this function.
    ///
    /// # Safety
    ///
    /// This function modifies memory locations pointed by the `arguments_and_result` argument. If those pointers are
    /// invalid or misaligned in that they do not match what is specified in the function declaration in the underlying
    /// [`Module`], calling this function will result in undefined behavior.
    ///
    /// # Parameters
    ///
    ///   * `name` - Name of the function to invoke.
    ///   * `arguments_and_result` - Pointer that is interpreted as a list of pointers to the actual arguments that will
    ///     be passed to the underlying function, followed by a pointer to its result. The pointed result value will be
    ///     mutated by this invocation and will hold the invocation result after this function returns.
    pub unsafe fn invoke_function<'s, S: Into<StringRef<'s>>>(
        &self,
        name: S,
        arguments_and_result: *mut *mut std::ffi::c_void,
    ) -> LogicalResult {
        unsafe {
            LogicalResult::from_c_api(mlirExecutionEngineInvokePacked(
                self.handle,
                name.into().to_c_api(),
                arguments_and_result,
            ))
        }
    }

    /// Dumps the object that has been generated internally by this [`InitializedExecutionEngine`] to a file at the
    /// specified [`Path`]. Note that if `enable_object_dump` was set to `false` when constructing the
    /// [`ExecutionEngine`] that resulted in this [`InitializedExecutionEngine`] using [`ExecutionEngine::for_module`],
    /// then the resulting file will be empty.
    pub fn dump_to_object_file(&self, path: &Path) {
        unsafe { mlirExecutionEngineDumpToObjectFile(self.handle, StringRef::from(path).to_c_api()) }
    }
}

impl Drop for InitializedExecutionEngine {
    fn drop(&mut self) {
        unsafe { mlirExecutionEngineDestroy(self.handle) }
    }
}

#[cfg(test)]
mod tests {
    use crate::dialects::builtin;
    use crate::{Context, DialectRegistry};

    use super::*;

    fn test_module<'c, 't>(context: &'c Context<'t>, include_callback: bool) -> Module<'c, 't> {
        context.register_dialects(&DialectRegistry::new_with_all_built_in_dialects());
        context.load_all_available_dialects();
        context.register_all_llvm_translations();
        let module = context
            .parse_module(if include_callback {
                r#"
                module {
                    llvm.func @callback(i32) -> i32

                    func.func @example(%argument : i32) -> i32 attributes { llvm.emit_c_interface } {
                        %callback_result = llvm.call @callback(%argument) : (i32) -> i32
                        %result = arith.addi %argument, %callback_result : i32
                        return %result : i32
                    }
                }
                "#
            } else {
                r#"
                module {
                    func.func @example(%argument : i32) -> i32 attributes { llvm.emit_c_interface } {
                        %result = arith.addi %argument, %argument : i32
                        return %result : i32
                    }
                }
                "#
            })
            .unwrap();

        // Convert to LLVM-compatible MLIR so that the resulting [`Module`] is supposed by [`ExecutionEngine`]s.
        let mut pass_manager = context.pass_manager();
        pass_manager.add_pass(builtin::passes::create_conversion_to_llvm_pass());
        assert!(pass_manager.run(&module.as_operation()).is_success());

        module
    }

    #[test]
    fn test_execution_engine_end_to_end_without_callback_function() {
        let context = Context::new();
        let module = test_module(&context, false);
        let engine = ExecutionEngine::for_module(&module, OptimizationLevel::default(), &[], true, false);
        unsafe {
            let engine = engine.initialize();

            // Verify that we can directly invoke the `@example` function using the [`ExecutionEngine`].
            let mut argument = 42;
            let mut result = -1;
            let mut argument_and_result = [&mut argument as *mut i32 as *mut _, &mut result as *mut i32 as *mut _];
            assert!(engine.invoke_function("example", argument_and_result.as_mut_ptr()).is_success());
            assert_eq!(argument, 42);
            assert_eq!(result, 84);

            // Verify that we can obtain a pointer to the `@example` function and invoke it.
            let example_fn = engine.get_function("example").unwrap();
            let example_fn = std::mem::transmute::<_, extern "C" fn(i32) -> i32>(example_fn);
            assert_eq!(example_fn(42), 84);

            // Verify that we can dump the compiled object into a file.
            let object_file = tempfile::NamedTempFile::new().unwrap();
            let object_file_path = object_file.path();
            engine.dump_to_object_file(&object_file_path);
            assert!(object_file_path.exists());
            assert!(std::fs::metadata(object_file_path).unwrap().len() > 0);
        }
    }

    #[test]
    fn test_execution_engine_end_to_end_with_callback_function() {
        let context = Context::new();
        let module = test_module(&context, true);
        let mut engine = ExecutionEngine::for_module(&module, OptimizationLevel::O2, &[], true, false);
        unsafe {
            extern "C" fn callback(argument: i32) -> i32 {
                argument * 5 + 1
            }

            engine.register_function("callback", callback as *mut _);

            let engine = engine.initialize();

            // Verify that we can directly invoke the `@example` function using the [`ExecutionEngine`].
            let mut argument = 42;
            let mut result = -1;
            let mut argument_and_result = [&mut argument as *mut i32 as *mut _, &mut result as *mut i32 as *mut _];
            assert!(engine.invoke_function("example", argument_and_result.as_mut_ptr()).is_success());
            assert_eq!(argument, 42);
            assert_eq!(result, 253);

            // Verify that we can obtain a pointer to the `@example` function and invoke it.
            let example_fn = engine.get_function("example").unwrap();
            let example_fn = std::mem::transmute::<_, extern "C" fn(i32) -> i32>(example_fn);
            assert_eq!(example_fn(42), 253);

            // Verify that we can obtain a pointer to the `@callback` function and invoke it.
            let callback_fn = engine.get_function("callback").unwrap();
            let callback_fn = std::mem::transmute::<_, extern "C" fn(i32) -> i32>(callback_fn);
            assert_eq!(callback_fn(42), 211);

            // Verify that we can dump the compiled object into a file.
            let object_file = tempfile::NamedTempFile::new().unwrap();
            let object_file_path = object_file.path();
            engine.dump_to_object_file(&object_file_path);
            assert!(object_file_path.exists());
            assert!(std::fs::metadata(object_file_path).unwrap().len() > 0);
        }
    }

    #[test]
    fn test_execution_engine_get_wrapped_function() {
        let context = Context::new();
        let module = test_module(&context, false);
        let engine = ExecutionEngine::for_module(&module, OptimizationLevel::O2, &[], false, false);
        unsafe {
            let engine = engine.initialize();
            assert!(engine.get_function("_mlir_example").is_some());
        }
    }

    #[test]
    fn test_execution_engine_get_invalid_function() {
        let context = Context::new();
        let module = test_module(&context, false);
        let engine = ExecutionEngine::for_module(&module, OptimizationLevel::O2, &[], false, false);
        unsafe {
            let engine = engine.initialize();
            assert!(engine.get_function("invalid").is_none());
        }
    }

    #[test]
    fn test_execution_engine_with_disabled_object_dumping() {
        let context = Context::new();
        let module = test_module(&context, false);
        let engine = ExecutionEngine::for_module(&module, OptimizationLevel::O2, &[], false, false);
        unsafe {
            let engine = engine.initialize();
            let object_file = tempfile::NamedTempFile::new().unwrap();
            let object_file_path = object_file.path();
            engine.dump_to_object_file(&object_file_path);
            assert!(object_file_path.exists());
            assert_eq!(std::fs::metadata(object_file_path).unwrap().len(), 0);
        }
    }

    #[test]
    fn test_execution_engine_drop() {
        let context = Context::new();
        let module = test_module(&context, false);
        let engine = ExecutionEngine::for_module(&module, OptimizationLevel::O2, &[], false, false);
        drop(engine);
    }
}
