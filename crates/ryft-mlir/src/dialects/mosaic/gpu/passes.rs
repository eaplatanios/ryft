use std::sync::OnceLock;

use ryft_xla_sys::mlir::dialects::mosaic::gpu::mlirMosaicGpuRegisterSerdePass;

use crate::{
    Attribute, Context, Error, GLOBAL_REGISTRATION_MUTEX, IntegerAttributeRef, Module, Operation, PassManager,
};

/// Name of the [`Module`] attribute in which the `mosaic_gpu-serde` pass records the bytecode schema version.
pub const MOSAIC_GPU_SERDE_VERSION_ATTRIBUTE: &str = "stable_mosaic_gpu.version";

/// Registers the pinned Mosaic GPU serialization pass with the global registry. The serialization pass
/// converts operations to a versioned representation in the unregistered `stable_mosaic_gpu` namespace. Enable
/// [`Context::allow_unregistered_dialects`] before running it. Deserialization restores registered operations and
/// removes the schema-version attribute. Pipeline construction alone does not transform a module.
pub fn register_mosaic_gpu_serde_pass() {
    // Use `OnceLock` to ensure that `register_mosaic_gpu_serde_pass` is called at most once.
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    INITIALIZED.get_or_init(|| {
        // Registration mutates MLIR's process-wide registry; retain the guard through the native call.
        let _guard = GLOBAL_REGISTRATION_MUTEX.lock().expect("MLIR global registration mutex poisoned");
        unsafe { mlirMosaicGpuRegisterSerdePass() };
    });
}

/// Returns the Mosaic GPU serde version recorded in the `stable_mosaic_gpu.version` attribute of `module`, or [`None`]
/// if the attribute is absent. Returns an error when the attribute is not an integer. This function reads the recorded
/// version without checking whether it is supported by the current runtime.
pub fn mosaic_gpu_serde_version(module: &Module<'_, '_>) -> Result<Option<i64>, Error> {
    match module.as_operation()?.attribute(MOSAIC_GPU_SERDE_VERSION_ATTRIBUTE)? {
        None => Ok(None),
        Some(attribute) => attribute
            .cast::<IntegerAttributeRef>()
            .map(|attribute| Some(attribute.signless_value()))
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "invalid `{MOSAIC_GPU_SERDE_VERSION_ATTRIBUTE}` module attribute; expected an integer attribute",
                ))
            }),
    }
}

/// Returns a textual `builtin.module` pipeline containing the Mosaic GPU serialization pass.
///
/// `serialize` selects serialization or deserialization. `target_version` selects the export schema when serializing;
/// omitting it uses the latest schema supported by the pass. The option does not select the input schema during
/// deserialization, which reads the module's version attribute instead.
///
/// The outer `builtin.module` names the pass-manager anchor. Use [`mosaic_gpu_serde_pass_manager`] to run the pass on
/// the supplied module itself. Passing this text to
/// [`OperationPassManager::parse_pass_pipeline`](crate::OperationPassManager::parse_pass_pipeline) instead nests the
/// pass under the receiving manager, so it visits nested modules.
pub fn mosaic_gpu_serde_pipeline(serialize: bool, target_version: Option<i32>) -> String {
    format!("builtin.module({})", mosaic_gpu_serde_pass_element(serialize, target_version))
}

/// Registers the serialization pass and constructs a [`PassManager`] anchored on `builtin.module`.
///
/// The manager runs [`mosaic_gpu_serde_pipeline`] on the module passed to [`PassManager::run`]. Serialization requires
/// [`Context::allow_unregistered_dialects`] and writes [`MOSAIC_GPU_SERDE_VERSION_ATTRIBUTE`]. Deserialization reads
/// that attribute to select the input schema, then removes it.
///
/// An omitted `target_version` uses the latest supported schema. MLIR prints the unset option as `target-version=0`,
/// but explicitly supplying zero is different: it requests schema zero. The pass checks version support when it runs.
pub fn mosaic_gpu_serde_pass_manager<'c, 't>(
    context: &'c Context<'t>,
    serialize: bool,
    target_version: Option<i32>,
) -> Result<PassManager<'c, 't>, Error> {
    register_mosaic_gpu_serde_pass();
    let manager = context.pass_manager_on_operation("builtin.module")?;
    manager
        .as_operation_pass_manager()?
        .add_pass_pipeline(mosaic_gpu_serde_pass_element(serialize, target_version).as_str())
        .map_err(Error::invalid_argument)?;
    Ok(manager)
}

/// Renders the pass options without an anchor so the pass can be added directly to a module-anchored manager.
fn mosaic_gpu_serde_pass_element(serialize: bool, target_version: Option<i32>) -> String {
    match target_version {
        Some(target_version) => format!("mosaic_gpu-serde{{serialize={serialize} target-version={target_version}}}"),
        None => format!("mosaic_gpu-serde{{serialize={serialize}}}"),
    }
}

#[cfg(test)]
mod tests {
    use indoc::formatdoc;
    use pretty_assertions::assert_eq;

    use ryft_xla_sys::mlir::dialects::mosaic::gpu::MOSAIC_GPU_SERDE_VERSION;

    use super::*;

    #[test]
    fn test_register_mosaic_gpu_serde_pass() {
        // Verify that repeated global registration is safe.
        register_mosaic_gpu_serde_pass();
        register_mosaic_gpu_serde_pass();

        let context = Context::new();
        let manager = context.pass_manager().unwrap();
        let mut manager = manager.as_operation_pass_manager().unwrap();
        assert_eq!(manager.add_pass_pipeline("mosaic_gpu-serde{serialize=true}"), Ok(()));
    }

    #[test]
    fn test_mosaic_gpu_serde_version() {
        let context = Context::new();
        let module = context.module(context.unknown_location()).unwrap();
        assert_eq!(mosaic_gpu_serde_version(&module), Ok(None));
        module.as_operation().unwrap().set_attribute(
            MOSAIC_GPU_SERDE_VERSION_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), 3),
        );
        assert_eq!(mosaic_gpu_serde_version(&module), Ok(Some(3)));
        module
            .as_operation()
            .unwrap()
            .set_attribute(MOSAIC_GPU_SERDE_VERSION_ATTRIBUTE, context.string_attribute("three"));
        assert!(matches!(mosaic_gpu_serde_version(&module),
            Err(Error::InvalidArgument { message, .. })
                if message == "invalid `stable_mosaic_gpu.version` module attribute; expected an integer attribute",
        ));
    }

    #[test]
    fn test_mosaic_gpu_serde_pipeline() {
        assert_eq!(mosaic_gpu_serde_pipeline(true, None), "builtin.module(mosaic_gpu-serde{serialize=true})");
        assert_eq!(mosaic_gpu_serde_pipeline(false, None), "builtin.module(mosaic_gpu-serde{serialize=false})");
        assert_eq!(
            mosaic_gpu_serde_pipeline(true, Some(3)),
            "builtin.module(mosaic_gpu-serde{serialize=true target-version=3})",
        );
        assert_eq!(
            mosaic_gpu_serde_pipeline(false, Some(0)),
            "builtin.module(mosaic_gpu-serde{serialize=false target-version=0})",
        );
    }

    #[test]
    fn test_mosaic_gpu_serde_pass_manager() {
        // The constructed pass manager prints back exactly as the pinned textual pipeline.
        let context = Context::new();
        let manager = mosaic_gpu_serde_pass_manager(&context, true, Some(MOSAIC_GPU_SERDE_VERSION)).unwrap();
        assert_eq!(
            manager.as_operation_pass_manager().unwrap().to_string(),
            mosaic_gpu_serde_pipeline(true, Some(MOSAIC_GPU_SERDE_VERSION)),
        );
        // MLIR prints the unspecified target version as the pass default, which the pass resolves when it runs.
        let manager = mosaic_gpu_serde_pass_manager(&context, false, None).unwrap();
        assert_eq!(
            manager.as_operation_pass_manager().unwrap().to_string(),
            "builtin.module(mosaic_gpu-serde{serialize=false target-version=0})",
        );

        // Run both directions on the root module, including the required context setting.
        let module = context.module(context.unknown_location()).unwrap();
        let serializer = mosaic_gpu_serde_pass_manager(&context, true, None).unwrap();
        assert!(serializer.run(&module.as_operation().unwrap()).is_failure());
        assert_eq!(mosaic_gpu_serde_version(&module), Ok(None));
        context.allow_unregistered_dialects();
        assert!(serializer.run(&module.as_operation().unwrap()).is_success());
        assert!(module.verify().unwrap());
        assert_eq!(mosaic_gpu_serde_version(&module), Ok(Some(i64::from(MOSAIC_GPU_SERDE_VERSION))));
        assert_eq!(
            module.to_string(),
            formatdoc!(
                "
                module attributes {{stable_mosaic_gpu.version = {MOSAIC_GPU_SERDE_VERSION} : i64}} {{
                }}
                ",
            ),
        );
        assert!(manager.run(&module.as_operation().unwrap()).is_success());
        assert_eq!(mosaic_gpu_serde_version(&module), Ok(None));

        // Pinning a version newer than the supported schema is rejected when the pipeline runs.
        let module = context.module(context.unknown_location()).unwrap();
        let manager = mosaic_gpu_serde_pass_manager(&context, true, Some(MOSAIC_GPU_SERDE_VERSION + 1)).unwrap();
        assert!(manager.run(&module.as_operation().unwrap()).is_failure());
        assert_eq!(mosaic_gpu_serde_version(&module), Ok(None));
    }
}
