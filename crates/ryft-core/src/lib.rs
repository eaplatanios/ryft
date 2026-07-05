// Derive macros emitted by `ryft-macros` use the public `ryft::...` facade path by default. This `self`-alias
// lets those same generated paths resolve when those macros are used inside `ryft-core` itself.
extern crate self as ryft;

pub mod axes;
pub mod batching;
pub mod broadcasting;
pub mod compilation;
pub mod contexts;
pub mod differentiation;
pub mod effects;
pub mod errors;
pub mod interpretation;
pub mod macros;
pub mod operations;
pub mod parameters;
pub mod partial;
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
pub use axes::{AxisError, AxisIndex, NamedAxes, NamedAxis};
pub use batching::{
    ArrayBatch, BatchAxis, BatchAxisSpecification, BatchableOperation, BatchableProgramOperation, BatchingError,
    ProgramBatchingOutputAxesPolicy,
};
pub use broadcasting::{Broadcastable, BroadcastingError};
pub use compilation::*;
pub use contexts::{Context, Domain, EagerContext, StagingContext, ValueResolution};
pub use differentiation::*;
pub use effects::{Effect, Effects};
pub use errors::{CustomError, Error};
pub use interpretation::{InterpretableOperation, InterpretableProgramOperation};
pub use operations::*;
pub use parameters::{
    ArrayParameterizedFamily, BTreeMapParameterizedFamily, HashMapParameterizedFamily, Parameter, ParameterError,
    ParameterParameterizedFamily, ParameterPath, ParameterPathSegment, Parameterized, ParameterizedFamily,
    PathPrefixedParameterIterator, PhantomDataParameterizedFamily, Placeholder, VecParameterizedFamily,
};
pub use partial::{
    PartialEvaluation, PartialEvaluationInput, PartialEvaluationOutput, PartialEvaluationValue, PartialEvaluator,
    PartialValue, PartialValueMaterialization, PartiallyEvaluatableOperation, PartiallyEvaluatableProgramOperation,
    PartitionedProgram,
};
pub use programs::{
    Atom, AtomId, Instruction, MaybeZero, Program, ProgramBuilder, ProgramError, ProgramLiveSets, Value,
};
pub use scalars::{Scalar, ScalarDomain};
pub use sharding::*;
pub use tracing::{DomainTracer, DomainTracingContext, NestedTracer, Tracer, TracerState, TracingContext};
pub use tracing_v2::batching::{Batch, BatchingContext, BatchingMeta, BatchingTracer, batch_program};
pub use tracing_v2::differentiation::{
    DifferentiableOperation, DifferentiableProgramOperation, JvpTracer, Linearization, materialize,
};
pub use tracing_v2::operations::custom_derivatives::{CustomJvpOperation, CustomVjpOperation};
pub use types::*;
