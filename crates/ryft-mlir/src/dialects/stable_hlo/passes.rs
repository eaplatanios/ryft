use std::sync::OnceLock;

use ryft_xla_sys::bindings::mlirRegisterAllStablehloPasses;

use crate::{Block, BlockRef, Error, GLOBAL_REGISTRATION_MUTEX, Module, Operation, Region, Value};

/// Registers all compiler passes of the StableHLO [`Dialect`](crate::Dialect).
pub fn register_stable_hlo_passes() {
    // Use `OnceLock` to ensure that `register_stable_hlo_passes` is called at most once.
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    INITIALIZED.get_or_init(|| unsafe {
        let _guard = GLOBAL_REGISTRATION_MUTEX.lock();
        mlirRegisterAllStablehloPasses()
    });
}

impl Module<'_, '_> {
    /// Removes every `stablehlo.constant` operation whose result has no uses from this [`Module`], including those in
    /// nested regions. Constants are pure, so removing an unused one never changes the semantics of the module.
    ///
    /// This function borrows the module mutably so that no reference to a removed operation can outlive the call.
    pub fn remove_unused_stable_hlo_constants(&mut self) -> Result<(), Error> {
        /// Removes every unused `stablehlo.constant` operation from `block` and from the regions nested
        /// in its operations.
        fn remove_unused_constants_in_block<'b, 'c: 'b, 't: 'c>(block: BlockRef<'b, 'c, 't>) -> Result<(), Error> {
            let mut unused_constants = Vec::new();
            for operation in block.operations()? {
                let operation = operation?;
                for region in operation.regions() {
                    for nested_block in region?.blocks()? {
                        remove_unused_constants_in_block(nested_block?)?;
                    }
                }

                if operation.name().as_str() == Ok("stablehlo.constant")
                    && operation.result(0)?.uses()?.next().is_none()
                {
                    unused_constants.push(operation);
                }
            }

            for constant in unused_constants {
                // The constant has no uses, so no value refers to its result after it is removed.
                unsafe { block.remove_operation(constant)? };
            }

            Ok(())
        }

        remove_unused_constants_in_block(self.body()?)
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle};

    use super::*;

    #[test]
    fn test_register_stable_hlo_passes() {
        // Verify that there are no segmentation faults, even when we try to register multiple times.
        register_stable_hlo_passes();
        register_stable_hlo_passes();
    }

    #[test]
    fn test_module_remove_unused_stable_hlo_constants() {
        // Unused constants are removed from nested regions as well, and constants with uses are kept.
        let context = Context::new();
        context.load_dialect(DialectHandle::func().unwrap()).unwrap();
        context.load_dialect(DialectHandle::stable_hlo().unwrap()).unwrap();
        let mut module = context
            .parse_module(indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<i32>) -> tensor<f32> {
                    %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                    %1 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
                    %2 = "stablehlo.case"(%arg0) ({
                      %3 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
                      stablehlo.return %1 : tensor<f32>
                    }, {
                      stablehlo.return %1 : tensor<f32>
                    }) : (tensor<i32>) -> tensor<f32>
                    return %2 : tensor<f32>
                  }
                }
            "#})
            .unwrap();
        module.remove_unused_stable_hlo_constants().unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<i32>) -> tensor<f32> {
                    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
                    %0 = "stablehlo.case"(%arg0) ({
                      stablehlo.return %cst : tensor<f32>
                    }, {
                      stablehlo.return %cst : tensor<f32>
                    }) : (tensor<i32>) -> tensor<f32>
                    return %0 : tensor<f32>
                  }
                }
            "#},
        );
    }
}
