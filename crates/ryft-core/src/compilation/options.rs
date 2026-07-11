//! Backend-specific options carried through the backend-neutral compilation pipeline.

use std::fmt::Debug;

use super::contexts::CompilationDomain;

/// Backend-specific options passed through the common compilation lifecycle.
///
/// The core pipeline keeps the payload opaque and hands it to the [`CompilationDomain`] before staging. The staged
/// artifact retains the same options for lowering, cache identity, and compilation. This wrapper provides one
/// consistent public boundary while the domain owns any option-sensitive signature normalization. Its payload is
/// private so callers cannot couple to the wrapper's layout;
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
    use crate::backends::scalars::ScalarOperation;
    use crate::compilation::{CaptureReference, StagedFunction};
    use crate::contexts::Domain;
    use crate::programs::ProgramError;
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
        type Constant = CaptureReference<DataType>;
        type Operation = ScalarOperation<Scalar>;
    }

    impl CompilationDomain for TestCompilationDomain {
        type LoweredProgram = Vec<DataType>;
        type CompiledProgram = Vec<DataType>;
        type Options = TestOptions;
        type Error = ProgramError;

        fn stage<Request>(
            &self,
            request: Request,
        ) -> Result<StagedFunction<Self, Request::Input, Request::Output>, Self::Error>
        where
            Request: crate::compilation::function::StageRequest<Self>,
        {
            request.trace(|_, output_types| Ok(output_types))
        }

        fn lower<Request>(
            &self,
            staged: Request,
        ) -> Result<crate::compilation::LoweredFunction<Self, Request::Input, Request::Output>, Self::Error>
        where
            Request: crate::compilation::function::LoweringRequest<Self>,
        {
            let output_types = staged.staged().output_types().to_vec();
            Ok(staged.into_lowered(output_types.clone(), output_types))
        }

        fn compile<Request>(
            &self,
            lowered: Request,
        ) -> Result<crate::compilation::CompiledFunction<Self, Request::Input, Request::Output>, Self::Error>
        where
            Request: crate::compilation::function::CompileRequest<Self>,
        {
            let output_types = lowered.lowered().output_types().to_vec();
            Ok(lowered.into_compiled(std::sync::Arc::new(output_types.clone()), output_types))
        }

        fn call<Request>(&self, request: Request) -> Result<Request::RuntimeOutput, Self::Error>
        where
            Request: crate::compilation::function::CallRequest<Self>,
        {
            let executable = request.executable().clone();
            Request::reconstruct(&executable, Vec::new())
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
