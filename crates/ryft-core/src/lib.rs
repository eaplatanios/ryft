pub mod batching;
pub mod broadcasting;
pub mod compilation;
pub mod contexts;
pub mod differentiation;
pub mod domains;
pub mod errors;
pub mod macros;
pub mod operations;
pub mod parameters;
pub mod programs;
pub mod scalars;
pub mod sharding;
pub mod tracing;
pub mod tracing_v2;
pub mod types;
pub mod utilities;

#[cfg(any(test, feature = "test-utilities"))]
pub mod tests;

// TODO(eaplatanios): Make all of the following more specific.
pub use batching::BatchingError;
pub use broadcasting::{Broadcastable, BroadcastingError};
pub use compilation::*;
pub use contexts::{Context, EagerContext, ProvidesContext, StagingContext};
pub use differentiation::*;
pub use domains::{AbstractDomain, Domain};
pub use errors::{CustomError, Error};
pub use operations::*;
pub use parameters::{
    ArrayParameterizedFamily, BTreeMapParameterizedFamily, HashMapParameterizedFamily, Parameter, ParameterError,
    ParameterParameterizedFamily, ParameterPath, ParameterPathSegment, Parameterized, ParameterizedFamily,
    PathPrefixedParameterIterator, PhantomDataParameterizedFamily, Placeholder, VecParameterizedFamily,
};
pub use programs::{Atom, AtomId, Instruction, Program, ProgramBuilder, ProgramError, ProgramLiveSets, Value};
pub use scalars::{LinearScalarDomain, ScalarDomain};
pub use sharding::*;
pub use tracing::{
    AbstractTracer, AbstractTracingContext, DomainTracer, Tracer, TracerState, TracingContext, infer_output_type,
    interpret_and_trace, trace,
};
pub use tracing_v2::batch;
pub use types::*;
