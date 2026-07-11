//! Backend-specific options carried through the backend-neutral compilation pipeline.

use std::fmt::Debug;

use super::contexts::CompilationDomain;

/// Backend-specific options passed through the common compilation lifecycle.
///
/// The core pipeline keeps the payload opaque and hands it to the [`CompilationDomain`] when lowering, deriving cache
/// identity, and compiling. This wrapper provides one consistent public boundary without requiring backend option
/// types to implement unrelated traits. Its payload is private so callers cannot couple to the wrapper's layout;
/// [`options`](Self::options) borrows it and [`into_options`](Self::into_options) recovers ownership.
pub struct CompilationOptions<D: CompilationDomain> {
    /// Backend-specific options. See [`CompilationDomain::Options`] for the contract.
    options: D::Options,
}

impl<D: CompilationDomain<Options: Clone>> Clone for CompilationOptions<D> {
    fn clone(&self) -> Self {
        Self { options: self.options.clone() }
    }
}

impl<D: CompilationDomain<Options: Debug>> Debug for CompilationOptions<D> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("CompilationOptions").field("options", &self.options).finish()
    }
}

impl<D: CompilationDomain> CompilationOptions<D> {
    /// Creates [`CompilationOptions`] carrying the provided backend `options`.
    #[inline]
    pub fn new(options: D::Options) -> Self {
        Self { options }
    }

    /// Returns the backend-specific options carried by this [`CompilationOptions`].
    #[inline]
    pub fn options(&self) -> &D::Options {
        &self.options
    }

    /// Consumes this [`CompilationOptions`] and returns its backend-specific options.
    #[inline]
    pub fn into_options(self) -> D::Options {
        self.options
    }
}

impl<D: CompilationDomain<Options: Default>> Default for CompilationOptions<D> {
    #[inline]
    fn default() -> Self {
        Self { options: D::Options::default() }
    }
}

#[cfg(test)]
mod tests {
    use crate::backends::scalars::Scalar;
    use crate::contexts::Domain;
    use crate::operations::scalars::ScalarOperation;
    use crate::programs::{Program, ProgramError};
    use crate::types::DataType;

    use super::*;

    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    struct TestOptions {
        optimization_level: usize,
    }

    #[derive(Clone)]
    struct TestCompilationDomain;

    impl Domain for TestCompilationDomain {
        type Type = DataType;
        type Value = Scalar;
        type Constant = Scalar;
        type Operation = ScalarOperation<Scalar>;
    }

    impl CompilationDomain for TestCompilationDomain {
        type LoweredProgram = Vec<DataType>;
        type CompiledProgram = Vec<DataType>;
        type Options = TestOptions;
        type Error = ProgramError;
        type CacheKey = ();

        fn lower(
            &self,
            _program: &Program<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>>,
            _capture_count: usize,
            _options: &TestOptions,
        ) -> Result<Vec<DataType>, ProgramError> {
            Ok(Vec::new())
        }

        fn lowered_output_types<'a>(&self, program: &'a Vec<DataType>) -> &'a [DataType] {
            program
        }

        fn compilation_key(&self, _program: &Vec<DataType>, _options: &TestOptions) -> Result<(), ProgramError> {
            Ok(())
        }

        fn compile(&self, program: &Vec<DataType>, _options: &TestOptions) -> Result<Vec<DataType>, ProgramError> {
            Ok(program.clone())
        }

        fn compiled_output_types<'a>(&self, program: &'a Vec<DataType>) -> &'a [DataType] {
            program
        }

        fn execute(&self, _program: &Vec<DataType>, inputs: Vec<Scalar>) -> Result<Vec<Scalar>, ProgramError> {
            Ok(inputs)
        }
    }

    #[test]
    fn test_compilation_options() {
        let options = CompilationOptions::<TestCompilationDomain>::new(TestOptions { optimization_level: 3 });
        assert_eq!(options.options(), &TestOptions { optimization_level: 3 });
        assert_eq!(options.clone().into_options(), TestOptions { optimization_level: 3 });
        assert_eq!(format!("{options:?}"), "CompilationOptions { options: TestOptions { optimization_level: 3 } }");

        let default = CompilationOptions::<TestCompilationDomain>::default();
        assert_eq!(default.options(), &TestOptions::default());
    }
}
