use std::sync::OnceLock;

use ryft_xla_sys::bindings::{mlirRegisterAllSdyPassesAndPipelines, mlirRegisterAllXlaSdyPassesAndPipelines};

use crate::{Block, BlockRef, Error, GLOBAL_REGISTRATION_MUTEX, Module, Operation, Region, Value};

/// Registers all compiler passes and pipelines of the Shardy [`Dialect`](crate::Dialect).
pub fn register_shardy_passes_and_pipelines() {
    // Use `OnceLock` to ensure that `register_shardy_passes_and_pipelines` is called at most once.
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    INITIALIZED.get_or_init(|| unsafe {
        let _guard = GLOBAL_REGISTRATION_MUTEX.lock();
        mlirRegisterAllSdyPassesAndPipelines()
    });
}

/// Registers all compiler passes and pipelines of the XLA Shardy [`Dialect`](crate::Dialect).
pub fn register_xla_shardy_passes_and_pipelines() {
    // Use `OnceLock` to ensure that `register_xla_shardy_passes_and_pipelines` is called at most once.
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    INITIALIZED.get_or_init(|| unsafe {
        let _guard = GLOBAL_REGISTRATION_MUTEX.lock();
        mlirRegisterAllXlaSdyPassesAndPipelines()
    });
}

impl Module<'_, '_> {
    /// Removes every Shardy operation from this [`Module`], including those in nested regions, for a compiler that
    /// does not support the Shardy dialect. Each `sdy.sharding_constraint` is replaced by its input and each `sdy.mesh`
    /// declaration is erased. This preserves the result of a module that runs on a single device, where every sharding
    /// holds the whole value. Attributes of other operations that refer to an erased mesh are not changed.
    ///
    /// This function borrows the module mutably so that no reference to a removed operation can outlive the call.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] for any other Shardy operation (e.g., `sdy.manual_computation`), which has
    /// no equivalent without the dialect. The operations visited before it have already been removed.
    #[inline]
    pub fn remove_shardy_operations(&mut self) -> Result<(), Error> {
        /// Removes every Shardy operation from `block` and from the regions nested in its operations, as
        /// [`Module::remove_shardy_operations`] describes.
        fn remove_shardy_operations_in_block<'b, 'c: 'b, 't: 'c>(block: BlockRef<'b, 'c, 't>) -> Result<(), Error> {
            let mut shardy_operations = Vec::new();
            for operation in block.operations()? {
                let operation = operation?;
                let name = operation.name();
                match name.as_str() {
                    Ok("sdy.sharding_constraint") => {
                        operation.result(0)?.replace_uses(operation.operand_value(0)?);
                        shardy_operations.push(operation);
                    }
                    Ok("sdy.mesh") => shardy_operations.push(operation),
                    Ok(name) if name.starts_with("sdy.") => {
                        return Err(Error::invalid_argument(format!(
                            "Shardy operation `{name}` has no equivalent without the Shardy dialect",
                        )));
                    }
                    _ => {}
                }
                for region in operation.regions() {
                    for nested_block in region?.blocks()? {
                        remove_shardy_operations_in_block(nested_block?)?;
                    }
                }
            }

            for operation in shardy_operations {
                // The uses of a constraint were replaced above and a mesh declaration has no results,
                // so no value refers to a result of these operations after they are removed.
                unsafe { block.remove_operation(operation)? };
            }

            Ok(())
        }

        remove_shardy_operations_in_block(self.body()?)
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle};

    use super::*;

    #[test]
    fn test_register_shardy_passes_and_pipelines() {
        // Verify that there are no segmentation faults, even when we try to register multiple times.
        register_shardy_passes_and_pipelines();
        register_shardy_passes_and_pipelines();
        register_xla_shardy_passes_and_pipelines();
        register_xla_shardy_passes_and_pipelines();
    }

    #[test]
    fn test_module_remove_shardy_operations() {
        // Constraints are replaced by their inputs in nested regions as well, and the mesh declaration is erased.
        let context = Context::new();
        context.load_dialect(DialectHandle::func().unwrap()).unwrap();
        context.load_dialect(DialectHandle::stable_hlo().unwrap()).unwrap();
        context.load_dialect(DialectHandle::shardy().unwrap()).unwrap();
        let mut module = context
            .parse_module(indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<i32>) -> tensor<4xf32> {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{}], replicated={"x"}> : tensor<4xf32>
                    %1 = "stablehlo.case"(%arg1) ({
                      %2 = sdy.sharding_constraint %0 <@mesh, [{"x"}]> : tensor<4xf32>
                      %3 = stablehlo.negate %2 : tensor<4xf32>
                      stablehlo.return %3 : tensor<4xf32>
                    }, {
                      stablehlo.return %0 : tensor<4xf32>
                    }) : (tensor<i32>) -> tensor<4xf32>
                    return %1 : tensor<4xf32>
                  }
                }
            "#})
            .unwrap();
        module.remove_shardy_operations().unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<i32>) -> tensor<4xf32> {
                    %0 = "stablehlo.case"(%arg1) ({
                      %1 = stablehlo.negate %arg0 : tensor<4xf32>
                      stablehlo.return %1 : tensor<4xf32>
                    }, {
                      stablehlo.return %arg0 : tensor<4xf32>
                    }) : (tensor<i32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );

        // A manual computation has no equivalent without the dialect.
        let mut module = context
            .parse_module(indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
                      sdy.return %arg1 : tensor<4xf32>
                    } : (tensor<4xf32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#})
            .unwrap();
        assert!(matches!(
            module.remove_shardy_operations(),
            Err(Error::InvalidArgument { message, .. })
                if message == "Shardy operation `sdy.manual_computation` has no equivalent without the Shardy dialect",
        ));
    }
}
