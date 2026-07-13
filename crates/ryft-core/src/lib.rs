// Derive macros emitted by `ryft-macros` use the public `ryft::...` facade path by default. This `self`-alias
// lets those same generated paths resolve when those macros are used inside `ryft-core` itself.
extern crate self as ryft;

pub mod axes;
pub mod backends;
pub mod batching;
pub mod broadcasting;
pub mod captures;
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
pub mod sharding;
pub mod tests;
pub mod tracing;
pub mod tracing_v2;
pub mod types;
pub mod utilities;

// TODO(eaplatanios): Make all of the following more specific.
pub use axes::{AxisError, AxisIndex, NamedAxes, NamedAxis};
pub use backends::*;
pub use batching::{
    ArrayBatch, Batch, BatchAxis, BatchAxisSpecification, BatchableOperation, BatchableProgramOperation,
    BatchingContext, BatchingError, BatchingTracer, InterpretableBatchableOperation, ProgramBatchingOutputAxesPolicy,
    batch,
};
pub use broadcasting::{Broadcastable, BroadcastingError};
pub use captures::{CaptureReference, CapturingContext, ClosedProgram};
pub use compilation::*;
pub use contexts::{Context, Domain, EagerContext, StagingContext, ValueResolution};
pub use differentiation::*;
pub use effects::{Effect, Effects};
pub use errors::{CustomError, Error, MaybeFallible};
pub use interpretation::{InterpretableOperation, InterpretableProgramOperation};
pub use operations::*;
pub use parameters::{
    ArrayParameterizedFamily, BTreeMapParameterizedFamily, HashMapParameterizedFamily, Parameter, ParameterError,
    ParameterParameterizedFamily, ParameterPath, ParameterPathSegment, Parameterized, ParameterizedFamily,
    PathPrefixedParameterIterator, PhantomDataParameterizedFamily, Placeholder, VecParameterizedFamily,
};
pub use partial::{
    PartialEvaluation, PartialEvaluationContext, PartialEvaluationInput, PartialEvaluationOutput,
    PartialEvaluationValue, PartialTracer, PartialValue, PartialValueMaterialization, PartiallyEvaluatableOperation,
    PartiallyEvaluatableProgramOperation, PartitionedProgram,
};
pub use programs::{
    Atom, AtomId, FlatProgram, Instruction, InstructionId, InstructionRef, MaybeZero, Program, ProgramBuilder,
    ProgramError, ProgramLiveSets, Region, RegionId, Value, ValueId,
};
pub use sharding::*;
pub use tracing::{
    DomainTracer, DomainTracingContext, NestedTracer, Trace, Tracer, TracerState, TracingContext, infer_output_type,
    trace,
};
pub use tracing_v2::operations::custom_derivatives::{CustomJvpOperation, CustomVjpOperation};
pub use types::*;
