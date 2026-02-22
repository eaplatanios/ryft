use ryft_xla_sys::bindings::{
    MlirAsmState, MlirBytecodeWriterConfig, MlirOpPrintingFlags, mlirAsmStateCreateForOperation,
    mlirAsmStateCreateForValue, mlirAsmStateDestroy, mlirBytecodeWriterConfigCreate,
    mlirBytecodeWriterConfigDesiredEmitVersion, mlirBytecodeWriterConfigDestroy, mlirOpPrintingFlagsAssumeVerified,
    mlirOpPrintingFlagsCreate, mlirOpPrintingFlagsDestroy, mlirOpPrintingFlagsElideLargeElementsAttrs,
    mlirOpPrintingFlagsElideLargeResourceString, mlirOpPrintingFlagsEnableDebugInfo,
    mlirOpPrintingFlagsPrintGenericOpForm, mlirOpPrintingFlagsPrintNameLocAsPrefix, mlirOpPrintingFlagsSkipRegions,
    mlirOpPrintingFlagsUseLocalScope,
};

use crate::Value;

use super::Operation;

/// Controls settings for writing bytecode for [`Operation`]s.
#[derive(Copy, Clone, Debug, Default)]
pub struct BytecodeWriterConfiguration {
    /// Version of bytecode to write.
    pub version: Option<u64>,
}

impl BytecodeWriterConfiguration {
    /// Constructs a new [`BytecodeWriterConfiguration`] instance using the default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the [`BytecodeWriterConfigurationHandle`] that corresponds to this [`BytecodeWriterConfiguration`].
    pub(crate) unsafe fn handle(&self) -> BytecodeWriterConfigurationHandle {
        unsafe {
            let handle = mlirBytecodeWriterConfigCreate();
            if let Some(version) = self.version {
                mlirBytecodeWriterConfigDesiredEmitVersion(handle, version.cast_signed());
            }
            BytecodeWriterConfigurationHandle { handle }
        }
    }
}

/// Internal wrapper of the MLIR C API representation of [`BytecodeWriterConfiguration`] to make sure that the underlying
/// instance is properly freed after being used.
#[derive(Clone)]
#[repr(transparent)]
pub(crate) struct BytecodeWriterConfigurationHandle {
    /// Handle that represents this [`BytecodeWriterConfigurationHandle`] in the MLIR C API.
    handle: MlirBytecodeWriterConfig,
}

impl BytecodeWriterConfigurationHandle {
    /// Returns the [`MlirBytecodeWriterConfig`] that corresponds to this [`BytecodeWriterConfigurationHandle`]
    /// and which can be passed to functions in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub(crate) unsafe fn to_c_api(&self) -> MlirBytecodeWriterConfig {
        self.handle
    }
}

impl Drop for BytecodeWriterConfigurationHandle {
    fn drop(&mut self) {
        unsafe { mlirBytecodeWriterConfigDestroy(self.handle) }
    }
}

/// Controls settings for printing [`Operation`]s and [`Value`]s.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OperationPrintingFlags {
    /// Enables the elision of large [`ElementsAttribute`](crate::ElementsAttribute)s by printing a lexically valid but
    /// otherwise meaningless form instead of the actual element data. The value of this field is used to configure what
    /// is considered to be a "large" [`ElementsAttribute`](crate::ElementsAttribute) by providing an upper limit to the
    /// number of elements it contains. If [`None`], no elision is performed.
    pub elements_attribute_size_threshold: Option<usize>,

    /// Enables the elision of large resource strings (used in
    /// [`DenseResourceElementsAttributeRef`](crate::DenseResourceElementsAttributeRef)s) by omitting them from the
    /// `dialect_resources` section. The value of this field is used to configure what is considered to be a "large"
    /// resource string by providing an upper limit to its size/length. If [`None`], no elision is performed.
    pub resource_string_size_threshold: Option<usize>,

    /// If `true`, enables the printing of debug information.
    pub enable_debug_information: bool,

    /// Enables "pretty" printing of debug information (only applicable when `enable_debug_information` is `true`).
    /// The "prettified" format is more human readable but the Intermediate Representation (IR) that is generated in
    /// this "prettified" format **will not be parsable**.
    pub pretty_print_debug_information: bool,

    /// If `true`, results in always printing operations in the generic form.
    pub use_generic_op_form: bool,

    /// If `true`, results in printing [`NamedLocationRef`](crate::NamedLocationRef)s as prefixes to Single Static
    /// Assignment (SSA) IDs, when such [`NamedLocationRef`](crate::NamedLocationRef)s are provided/available.
    pub use_name_location_as_prefix: bool,

    /// If `true`, results in using the local scope when printing operations. This allows for using the printer
    /// in a more localized and thread-safe setting, but may not necessarily be identical to what the Intermediate
    /// Representation (IR) will look like when dumping full modules.
    pub use_local_scope: bool,

    /// If `true`, results in not verifying operations (i.e., assuming them already verified)
    /// when using custom operation printers.
    pub assume_verified_operations: bool,

    /// If `true`, results in skipping the printing of regions.
    pub skip_regions: bool,
}

impl OperationPrintingFlags {
    /// Returns the [`OperationPrintingFlagsHandle`] that corresponds to this [`OperationPrintingFlags`].
    pub(crate) unsafe fn handle(&self) -> OperationPrintingFlagsHandle {
        unsafe {
            let handle = OperationPrintingFlagsHandle { handle: mlirOpPrintingFlagsCreate() };

            if let Some(elements_attribute_size_threshold) = self.elements_attribute_size_threshold {
                mlirOpPrintingFlagsElideLargeElementsAttrs(
                    handle.handle,
                    elements_attribute_size_threshold.cast_signed(),
                )
            }

            if let Some(resource_string_size_threshold) = self.resource_string_size_threshold {
                mlirOpPrintingFlagsElideLargeResourceString(handle.handle, resource_string_size_threshold.cast_signed())
            }

            mlirOpPrintingFlagsEnableDebugInfo(
                handle.handle,
                self.enable_debug_information,
                self.pretty_print_debug_information,
            );

            if self.use_generic_op_form {
                mlirOpPrintingFlagsPrintGenericOpForm(handle.handle)
            }

            if self.use_name_location_as_prefix {
                mlirOpPrintingFlagsPrintNameLocAsPrefix(handle.handle)
            }

            if self.use_local_scope {
                mlirOpPrintingFlagsUseLocalScope(handle.handle)
            }

            if self.assume_verified_operations {
                mlirOpPrintingFlagsAssumeVerified(handle.handle)
            }

            if self.skip_regions {
                mlirOpPrintingFlagsSkipRegions(handle.handle)
            }

            handle
        }
    }
}

impl Default for OperationPrintingFlags {
    fn default() -> Self {
        Self {
            elements_attribute_size_threshold: Some(16),
            resource_string_size_threshold: Some(64),
            enable_debug_information: false,
            pretty_print_debug_information: true,
            use_generic_op_form: false,
            use_name_location_as_prefix: false,
            use_local_scope: false,
            assume_verified_operations: false,
            skip_regions: false,
        }
    }
}

/// Internal wrapper of the MLIR C API representation of [`OperationPrintingFlags`] to make sure that the underlying
/// instance is properly freed after being used.
#[derive(Clone)]
#[repr(transparent)]
pub(crate) struct OperationPrintingFlagsHandle {
    /// Handle that represents this [`OperationPrintingFlagsHandle`] in the MLIR C API.
    handle: MlirOpPrintingFlags,
}

impl OperationPrintingFlagsHandle {
    /// Returns the [`MlirOpPrintingFlags`] that corresponds to this [`OperationPrintingFlagsHandle`] instance
    /// and which can be passed to functions in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub(crate) unsafe fn to_c_api(&self) -> MlirOpPrintingFlags {
        self.handle
    }
}

impl Drop for OperationPrintingFlagsHandle {
    fn drop(&mut self) {
        unsafe { mlirOpPrintingFlagsDestroy(self.handle) }
    }
}

/// State that is used for generating MLIR assembly.
#[derive(Clone)]
#[repr(transparent)]
pub struct AsmState {
    /// Handle that represents this [`AsmState`] in the MLIR C API.
    handle: MlirAsmState,
}

impl AsmState {
    /// Creates a new [`AsmState`] that is to be used for generating assembly for the provided [`Operation`].
    pub fn for_operation<'o, 'c: 'o, 't: 'c, O: Operation<'o, 'c, 't>>(
        operation: &O,
        flags: OperationPrintingFlags,
    ) -> Self {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = operation.context().borrow();
        unsafe {
            let flags_handle = flags.handle();
            Self { handle: mlirAsmStateCreateForOperation(operation.to_c_api(), flags_handle.to_c_api()) }
        }
    }

    /// Creates a new [`AsmState`] that is to be used for generating assembly for the provided [`Value`].
    pub fn for_value<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>>(value: V, flags: OperationPrintingFlags) -> Self {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = value.context().borrow();
        unsafe {
            let flags_handle = flags.handle();
            Self { handle: mlirAsmStateCreateForValue(value.to_c_api(), flags_handle.to_c_api()) }
        }
    }

    /// Returns the [`MlirAsmState`] that corresponds to this [`AsmState`]
    /// and which can be passed to functions in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn to_c_api(&self) -> MlirAsmState {
        self.handle
    }
}

impl Drop for AsmState {
    fn drop(&mut self) {
        unsafe { mlirAsmStateDestroy(self.handle) }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Block, Context, OperationBuilder};

    use super::*;

    #[test]
    fn test_bytecode_writer_configuration() {
        assert_eq!(BytecodeWriterConfiguration::new().version, None);
        assert_eq!(BytecodeWriterConfiguration { version: Some(5) }.version, Some(5));

        // Check that we can construct a C API handle for a [`BytecodeWriterConfiguration`] without crashing.
        let configuration = BytecodeWriterConfiguration { version: Some(42) };
        let handle = unsafe { configuration.handle() };
        let _ = unsafe { handle.to_c_api() };
    }

    #[test]
    fn test_operation_printing_flags() {
        let flags = OperationPrintingFlags::default();
        assert_eq!(flags.elements_attribute_size_threshold, Some(16));
        assert_eq!(flags.resource_string_size_threshold, Some(64));
        assert_eq!(flags.enable_debug_information, false);
        assert_eq!(flags.pretty_print_debug_information, true);
        assert_eq!(flags.use_generic_op_form, false);
        assert_eq!(flags.use_name_location_as_prefix, false);
        assert_eq!(flags.use_local_scope, false);
        assert_eq!(flags.assume_verified_operations, false);
        assert_eq!(flags.skip_regions, false);

        // Check that we can construct a C API handle for an [`OperationPrintingFlags`] without crashing.
        let flags = OperationPrintingFlags {
            elements_attribute_size_threshold: None,
            resource_string_size_threshold: None,
            use_name_location_as_prefix: true,
            assume_verified_operations: true,
            skip_regions: true,
            ..OperationPrintingFlags::default()
        };
        let handle = unsafe { flags.handle() };
        let _ = unsafe { handle.to_c_api() };
    }

    #[test]
    fn test_asm_state() {
        let context = Context::new();
        context.allow_unregistered_dialects();

        let location = context.unknown_location();
        let index_type = context.index_type();
        let block = context.block(&[(index_type, location)]);
        let region = context.region();
        let operation = OperationBuilder::new("test.op", location).add_region(region).build().unwrap();

        // Check that we can create [`AsmState`] instances without crashing.
        let _ = AsmState::for_operation(&operation, OperationPrintingFlags::default());
        let _ = AsmState::for_value(block.argument(0).unwrap(), OperationPrintingFlags::default());

        // Check that we can construct a C API handle for an [`AsmState`] without crashing.
        let state = AsmState::for_operation(&operation, OperationPrintingFlags::default());
        let _ = unsafe { state.to_c_api() };
    }
}
