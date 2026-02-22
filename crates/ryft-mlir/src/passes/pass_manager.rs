use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::path::PathBuf;

use ryft_xla_sys::bindings::{
    MlirOpPassManager, MlirPassManager, MlirStringRef, mlirOpPassManagerAddOwnedPass, mlirOpPassManagerAddPipeline,
    mlirOpPassManagerGetNestedUnder, mlirPassManagerAddOwnedPass, mlirPassManagerCreate,
    mlirPassManagerCreateOnOperation, mlirPassManagerDestroy, mlirPassManagerEnableIRPrinting,
    mlirPassManagerEnableStatistics, mlirPassManagerEnableTiming, mlirPassManagerEnableVerifier,
    mlirPassManagerGetAsOpPassManager, mlirPassManagerGetNestedUnder, mlirPassManagerRunOnOp, mlirPrintPassPipeline,
};

use crate::support::write_to_formatter_callback;
use crate::{Context, LogicalResult, Operation, OperationPrintingFlags, StringRef};

use super::Pass;

/// Controls settings for printing IR during [`PassManager::run`] transformations.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PassIrPrintingOptions {
    /// If `true`, the IR will be printed before all [`Pass`]es.
    pub print_before_all_passes: bool,

    /// If `true`, the IR will be printed after all [`Pass`]es.
    pub print_after_all_passes: bool,

    /// If `true`, then module IR will be printed even for non-module [`Pass`]es.
    pub print_module_scope: bool,

    /// If `true`, then IR will only be printed after all [`Pass`]es that resulted in changes.
    pub print_only_on_change: bool,

    /// If `true`, then IR will only be printed after all [`Pass`]es that resulted in failures.
    pub print_only_on_failure: bool,

    /// [`OperationPrintingFlags`] that control how the way in which the IR should be printed.
    pub flags: OperationPrintingFlags,

    /// Optional [`PathBuf`] to a directory in which to write the IR.
    /// If not provided, then the IR will be printed in the standard error stream.
    pub path: Option<PathBuf>,
}

impl Default for PassIrPrintingOptions {
    fn default() -> Self {
        Self {
            print_before_all_passes: true,
            print_after_all_passes: true,
            print_module_scope: true,
            print_only_on_change: true,
            print_only_on_failure: false,
            flags: OperationPrintingFlags::default(),
            path: None,
        }
    }
}

/// Enum describing different display modes for information dumped by [`PassManager`]s.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PassDisplayMode {
    /// Results are displayed in a list sorted by total, with each pass/analysis instance aggregated
    /// into one unique result.
    List = 0,

    /// Results are displayed in a nested pipeline view that mirrors the internal pass pipeline
    /// that is being executed in the pass manager.
    Pipeline = 1,
}

/// Represents an MLIR pass manager that can be used to configure and schedule a [`Pass`] pipeline. This pass manager
/// acts as the top-level entry point for MLIR pass pipelines and contains various configurations used for the entire
/// pass pipeline. [`OperationPassManager`]s can be used to schedule passes to run at specific levels of nesting and
/// can be constructed using [`PassManager::nest`] and [`OperationPassManager::nest`].
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/PassManagement/#pass-manager)
/// for more information.
pub struct PassManager<'c, 't: 'c> {
    /// Handle that represents this [`PassManager`] in the MLIR C API.
    handle: MlirPassManager,

    /// [`Context`] associated with this [`PassManager`].
    context: &'c Context<'t>,
}

impl<'c, 't: 'c> PassManager<'c, 't> {
    /// Constructs a new [`PassManager`] of this type from the provided handle that came
    /// from a function in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    unsafe fn from_c_api(handle: MlirPassManager, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() { None } else { Some(Self { handle, context }) }
    }

    /// Returns a reference to the [`Context`] that is associated with this [`PassManager`].
    pub fn context(&self) -> &'c Context<'t> {
        self.context
    }

    /// Casts this [`PassManager`] to an [`OperationPassManager`].
    pub fn as_operation_pass_manager<'p>(&'p self) -> OperationPassManager<'p, 'c, 't> {
        unsafe {
            OperationPassManager::from_c_api(mlirPassManagerGetAsOpPassManager(self.handle), self.context).unwrap()
        }
    }

    /// Enables printing the IR (potentially) before and after [`Pass`]es during calls to [`PassManager::run`],
    /// based on the provided [`PassIrPrintingOptions`]. Returns `true` if successful and `false` otherwise.
    /// The only case where this function can fail is when the [`Context`] associated with this [`PassManager`]
    /// uses multi-threading. That is because IR printing is only supported for single-threaded [`Context`]s.
    pub fn enable_ir_printing(&mut self, options: &PassIrPrintingOptions) -> bool {
        if self.context.thread_count() > 1 {
            false
        } else {
            // The following context borrow ensures that access to the underlying MLIR data structures is done safely
            // from Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to
            // MLIR internals that we have when working with the MLIR C API.
            let _guard = self.context().borrow_mut();
            unsafe {
                let flags_handle = options.flags.handle();
                mlirPassManagerEnableIRPrinting(
                    self.handle,
                    options.print_before_all_passes,
                    options.print_after_all_passes,
                    options.print_module_scope,
                    options.print_only_on_change,
                    options.print_only_on_failure,
                    flags_handle.to_c_api(),
                    options
                        .path
                        .as_ref()
                        .map(|path| StringRef::from(path.display().to_string().as_str()).to_c_api())
                        .unwrap_or(MlirStringRef { data: std::ptr::null(), length: 0 }),
                );
                true
            }
        }
    }

    /// Enables verification of the IR after each [`Pass`] during calls to [`PassManager::run`].
    pub fn enable_verification(&mut self) {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe { mlirPassManagerEnableVerifier(self.handle, true) }
    }

    /// Disables verification of the IR after each [`Pass`] during calls to [`PassManager::run`].
    pub fn disable_verification(&mut self) {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe { mlirPassManagerEnableVerifier(self.handle, false) }
    }

    /// Enables instrumentation to time the execution of [`Pass`]es and the computation of analyses
    /// during calls to [`PassManager::run`].
    pub fn enable_timing(&mut self) {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe { mlirPassManagerEnableTiming(self.handle) }
    }

    /// Enables the dumping of statistics for each [`Pass`] after running.
    pub fn enable_statistics(&mut self, mode: PassDisplayMode) {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe { mlirPassManagerEnableStatistics(self.handle, mode as u32) }
    }

    /// Nests an [`OperationPassManager`] under this top-level [`PassManager`] and returns it. The nested
    /// [`OperationPassManager`] will only run on [`Operation`]s matching the provided name.
    pub fn nest<'p, 's, S: Into<StringRef<'s>>>(&self, operation_name: S) -> OperationPassManager<'p, 'c, 't> {
        unsafe {
            OperationPassManager::from_c_api(
                mlirPassManagerGetNestedUnder(self.handle, operation_name.into().to_c_api()),
                self.context,
            )
            .unwrap()
        }
    }

    /// Adds the provided [`Pass`] to this [`PassManager`].
    pub fn add_pass(&mut self, pass: Pass) {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe {
            mlirPassManagerAddOwnedPass(self.handle, pass.to_c_api());
        }
    }

    /// Runs all the [`Pass`]es that have been added to this [`PassManager`] on the provided [`Operation`],
    /// returning a [`LogicalResult`] that represents whether the passes succeeded or not.
    pub fn run<'o, O: Operation<'o, 'c, 't>>(&self, operation: &O) -> LogicalResult
    where
        'c: 'o,
    {
        // Note that we cannot use a borrow guard on `self.context` here because that would prevent any
        // [`ExternalPass`]es from modifying the context during this run.
        unsafe { LogicalResult::from_c_api(mlirPassManagerRunOnOp(self.handle, operation.to_c_api())) }
    }
}

impl Drop for PassManager<'_, '_> {
    fn drop(&mut self) {
        unsafe { mlirPassManagerDestroy(self.handle) }
    }
}

impl<'t> Context<'t> {
    /// Creates a new top-level [`PassManager`] that is associated with this [`Context`], using the default anchor.
    pub fn pass_manager<'c>(&'c self) -> PassManager<'c, 't> {
        unsafe {
            PassManager::from_c_api(
                mlirPassManagerCreate(
                    // The following context borrow ensures that access to the underlying MLIR data structures is done
                    // safely from Rust. It is maybe more conservative than would be ideal, but that is due to the
                    // limited exposure to MLIR internals that we have when working with the MLIR C API.
                    *self.handle.borrow(),
                ),
                self,
            )
            .unwrap()
        }
    }

    /// Creates a new top-level [`PassManager`] that is associated with this [`Context`],
    /// anchored on the specified [`Operation`].
    pub fn pass_manager_on_operation<'c, 's, S: Into<StringRef<'s>>>(
        &'c self,
        anchor_operation: S,
    ) -> PassManager<'c, 't> {
        unsafe {
            PassManager::from_c_api(
                mlirPassManagerCreateOnOperation(
                    // The following context borrow ensures that access to the underlying MLIR data structures is done
                    // safely from Rust. It is maybe more conservative than would be ideal, but that is due to the
                    // limited exposure to MLIR internals that we have when working with the MLIR C API.
                    *self.handle.borrow(),
                    anchor_operation.into().to_c_api(),
                ),
                self,
            )
            .unwrap()
        }
    }
}

/// Represents an MLIR pass manager that runs [`Pass`]es on either a specific [`Operation`] type, or any isolated
/// operation. This pass manager can not be run on an operation directly. It must be run as part of a top-level
/// [`PassManager`] (e.g. when constructed via [`PassManager::nest`] or [`OperationPassManager::nest`]).
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/PassManagement/#pass-manager)
/// for more information.
pub struct OperationPassManager<'p, 'c: 'p, 't: 'c> {
    /// Handle that represents this [`OperationPassManager`] in the MLIR C API.
    handle: MlirOpPassManager,

    /// [`Context`] associated with this [`OperationPassManager`] reference.
    context: &'c Context<'t>,

    /// [`PhantomData`] used to track the lifetime of the [`PassManager`] or [`OperationPassManager`]
    /// that owns this [`OperationPassManager`].
    owner: PhantomData<&'p ()>,
}

impl<'p, 'c: 'p, 't: 'c> OperationPassManager<'p, 'c, 't> {
    /// Constructs a new [`OperationPassManager`] of this type from the provided handle that came
    /// from a function in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn from_c_api(handle: MlirOpPassManager, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() { None } else { Some(Self { handle, context, owner: PhantomData }) }
    }

    /// Returns the [`MlirOpPassManager`] that corresponds to this [`OperationPassManager`] and which can be passed
    /// to functions in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn to_c_api(&self) -> MlirOpPassManager {
        self.handle
    }

    /// Returns a reference to the [`Context`] that is associated with this [`OperationPassManager`].
    pub fn context(&self) -> &'c Context<'t> {
        self.context
    }

    /// Nests an [`OperationPassManager`] under this [`OperationPassManager`] and returns it. The nested
    /// [`OperationPassManager`] will only run on [`Operation`]s matching the provided name.
    pub fn nest<'s, S: Into<StringRef<'s>>>(&self, operation_name: S) -> OperationPassManager<'p, 'c, 't> {
        unsafe {
            OperationPassManager::from_c_api(
                mlirOpPassManagerGetNestedUnder(self.handle, operation_name.into().to_c_api()),
                self.context,
            )
            .unwrap()
        }
    }

    /// Adds the provided [`Pass`] to this [`OperationPassManager`].
    pub fn add_pass(&mut self, pass: Pass) {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe {
            mlirOpPassManagerAddOwnedPass(self.handle, pass.to_c_api());
        }
    }

    /// Parses the provided sequence of MLIR pass pipeline elements and adds them to this [`OperationPassManager`].
    pub fn add_pass_pipeline<'s, S: Into<StringRef<'s>>>(&mut self, pipeline_elements: S) -> Result<(), String> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        let mut error_message = None;
        let result = unsafe {
            LogicalResult::from_c_api(mlirOpPassManagerAddPipeline(
                self.handle,
                pipeline_elements.into().to_c_api(),
                Some(c_api_pass_pipeline_parse_error_callback),
                &mut error_message as *mut _ as *mut _,
            ))
        };

        if result.is_success() {
            Ok(())
        } else {
            Err(error_message.unwrap_or("failed to parse pass pipeline elements".into()))
        }
    }

    /// Parses the provided MLIR pass pipeline elements and assigns it to this [`OperationPassManager`].
    pub fn parse_pass_pipeline<'s, S: Into<StringRef<'s>>>(&mut self, pipeline: S) -> Result<(), String> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        let mut error_message = None;
        let result = unsafe {
            LogicalResult::from_c_api(mlirOpPassManagerAddPipeline(
                self.handle,
                pipeline.into().to_c_api(),
                Some(c_api_pass_pipeline_parse_error_callback),
                &mut error_message as *mut _ as *mut _,
            ))
        };

        if result.is_success() { Ok(()) } else { Err(error_message.unwrap_or("failed to parse pass pipeline".into())) }
    }
}

impl Display for OperationPassManager<'_, '_, '_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut data = (formatter, Ok(()));
        unsafe {
            mlirPrintPassPipeline(
                self.to_c_api(),
                Some(write_to_formatter_callback),
                &mut data as *mut _ as *mut std::ffi::c_void,
            );
        }
        data.1
    }
}

impl Debug for OperationPassManager<'_, '_, '_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "OperationPassManager[{self}]")
    }
}

unsafe extern "C" fn c_api_pass_pipeline_parse_error_callback(string: MlirStringRef, data: *mut std::ffi::c_void) {
    unsafe {
        let string = StringRef::from_c_api(string);
        let data = &mut *(data as *mut Option<String>);
        if let Some(message) = data {
            message.extend(string.as_str())
        } else {
            *data = string.as_str().map(String::from).ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::builtin;
    use crate::{Context, DialectRegistry};

    use super::*;

    #[test]
    fn test_pass_manager() {
        builtin::register_transforms_print_op_stats_pass();
        let context = Context::new();
        context.register_dialects(&DialectRegistry::new_with_all_built_in_dialects());
        context.load_all_available_dialects();
        context.register_all_llvm_translations();
        let location = context.unknown_location();
        let mut pass_manager = context.pass_manager_on_operation("builtin.module");
        pass_manager.enable_verification();
        pass_manager.disable_verification();
        pass_manager.enable_verification();
        pass_manager.enable_timing();
        pass_manager.enable_statistics(PassDisplayMode::Pipeline);
        pass_manager.add_pass(builtin::create_conversion_func_to_llvm_pass());
        assert!(!pass_manager.enable_ir_printing(&PassIrPrintingOptions::default()));
        context.disable_multi_threading();
        assert!(pass_manager.enable_ir_printing(&PassIrPrintingOptions {
            path: Some(PathBuf::default()),
            ..PassIrPrintingOptions::default()
        }));
        let mut op_pass_manager = pass_manager.as_operation_pass_manager();
        assert_eq!(
            format!("{}", op_pass_manager),
            "builtin.module(convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false})",
        );
        assert_eq!(
            format!("{:?}", op_pass_manager),
            "OperationPassManager[builtin.module(convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false})]",
        );
        assert!(op_pass_manager.add_pass_pipeline("func.func(print-op-stats{json=false})").is_ok());
        assert!(op_pass_manager.add_pass_pipeline("func.func(").is_err());

        // Run on a top-level module.
        let module = context.module(location);
        assert!(pass_manager.run(&module.as_operation()).is_success());

        // Run on a nested function.
        let module = context
            .parse_module(indoc! {"
                func.func @foo(%arg0 : i32) -> i32 {
                    %res = arith.addi %arg0, %arg0 : i32
                    return %res : i32
                }

                module {
                    func.func @bar(%arg0 : f32) -> f32 {
                        %res = arith.addf %arg0, %arg0 : f32
                        return %res : f32
                    }
                }"})
            .unwrap();
        let pass_manager = context.pass_manager();
        let mut nested_manager = pass_manager.nest("func.func");
        nested_manager.add_pass(builtin::create_transforms_print_op_stats_pass());
        assert_eq!(format!("{}", nested_manager), "func.func(print-op-stats{json=false})");
        assert_eq!(format!("{:?}", nested_manager), "OperationPassManager[func.func(print-op-stats{json=false})]");
        assert!(pass_manager.run(&module.as_operation()).is_success());
        let pass_manager = context.pass_manager();
        pass_manager
            .nest("builtin.module")
            .nest("func.func")
            .add_pass(builtin::create_transforms_print_op_stats_pass());
        assert!(pass_manager.run(&module.as_operation()).is_success());

        // Test a couple C API edge cases.
        let bad_pass_manager_handle = MlirPassManager { ptr: std::ptr::null_mut() };
        let bad_op_pass_manager_handle = MlirOpPassManager { ptr: std::ptr::null_mut() };
        assert!(unsafe { PassManager::from_c_api(bad_pass_manager_handle, &context) }.is_none());
        assert!(unsafe { OperationPassManager::from_c_api(bad_op_pass_manager_handle, &context) }.is_none());
    }

    #[test]
    fn test_pass_manager_parse_pass_pipeline() {
        builtin::register_transforms_print_op_stats_pass();
        let context = Context::new();

        // Test using a good pipeline.
        let pass_manager = context.pass_manager();
        let mut op_pass_manager = pass_manager.as_operation_pass_manager();
        assert!(
            op_pass_manager
                .parse_pass_pipeline(
                    "builtin.module(func.func(print-op-stats{json=false}),func.func(print-op-stats{json=false}))",
                )
                .is_ok(),
        );
        assert_eq!(
            op_pass_manager.to_string(),
            "any(builtin.module(func.func(print-op-stats{json=false}),func.func(print-op-stats{json=false})))",
        );

        // Test using a bad pipeline.
        let result = op_pass_manager.parse_pass_pipeline("func.func(nonexistent-pass)");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("does not refer to a registered pass"));
    }
}
