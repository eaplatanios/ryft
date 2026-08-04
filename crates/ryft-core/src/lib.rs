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
pub mod errors;
pub mod interpretation;
pub mod macros;
pub mod operations;
pub mod parameters;
pub mod partial;
pub mod programs;
pub mod sharding;
pub mod tracing;
pub mod tracing_v2;
pub mod types;
pub mod utilities;

// TODO(eaplatanios): Make all of the following more specific.
pub use axes::{AXIS_INDEX_OPERATION_NAME, Axes, Axis, AxisError, AxisIndex, AxisIndexOperation, NamedAxes, NamedAxis};
// Both `backends` and `types` currently expose public `arrays` and `dimensions` modules. P9 replaces these temporary
// glob exports with an explicit facade; suppress the known ambiguity until that dedicated API increment.
#[allow(ambiguous_glob_reexports)]
pub use backends::*;
pub use batching::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, Batch, BatchAxis, BatchAxisSpecification, BatchableOperation,
    BatchableType, BatchedProgram, BatchingContext, BatchingDriver, BatchingEntrypointPolicy, BatchingError,
    BatchingPolicy, BatchingPolicyProjection, BatchingTracer, DimensionSource, InterpretableBatchableOperation,
    ProgramBatchingOutputAxesPolicy, RecursiveBatchingPolicy, StaticArrayBatchingPolicy, batch,
    batch_projected_operation,
};
pub use broadcasting::{Broadcastable, BroadcastingError};
pub use captures::{CaptureReference, CapturingContext, ClosedProgram};
pub use compilation::*;
pub use contexts::{Context, Domain, EagerContext, ProjectedContext, StagingContext, ValueResolution};
pub use differentiation::*;
pub use errors::{CustomError, Error, MaybeFallible};
pub use interpretation::{InterpretableOperation, InterpretationDriver};
pub use operations::*;
pub use parameters::{
    ArrayParameterizedFamily, BTreeMapParameterizedFamily, HashMapParameterizedFamily, Parameter, ParameterError,
    ParameterParameterizedFamily, ParameterPath, ParameterPathSegment, Parameterized, ParameterizedFamily,
    PathPrefixedParameterIterator, PhantomDataParameterizedFamily, Placeholder, VecParameterizedFamily,
};
pub use partial::{
    PartialEvaluation, PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationInput,
    PartialEvaluationOutput, PartialEvaluationValue, PartialTracer, PartialValue, PartialValueMaterialization,
    PartiallyEvaluatableOperation, PartitionedProgram,
};
pub use programs::*;
pub use sharding::*;
pub use tracing::{
    DomainTracer, DomainTracingContext, NestedTracer, NestedTracingContext, Trace, Tracer, TracerState, TracingContext,
    infer_output_type, trace,
};
pub use tracing_v2::custom_derivatives::{CustomJvpOperation, CustomVjpOperation};
pub use tracing_v2::rematerialization::RematerializeOperation;
pub use types::*;

#[cfg(test)]
pub(crate) mod tests {
    use crate::macros::check_count;
    use crate::programs::effects::{Effect, Effects};
    use crate::programs::operations::Operation;
    use crate::programs::regions::{RegionInterface, RegionSlot};
    use crate::programs::types::TypeError;
    use crate::types::DataType;

    /// Test [`Operation`] with declared attached-region slots, used to exercise the [`Region`](crate::Region) machinery
    /// (i.e., construction, interning and sharing, interface derivation, validation, effects propagation, rendering,
    /// splicing, and rebuild paths) in isolation. Production region-carrying operations (e.g., the control-flow family)
    /// impose their own type invariants such as Boolean predicates and congruent branch outputs, whereas this fixture
    /// declares arbitrary region slot names with a trivial inference rule, so machinery tests stay three-line fixtures
    /// whose failures cannot be masked by control-flow inference.
    #[derive(Clone, Debug, PartialEq)]
    pub enum TestRegionOperation {
        /// Region-free binary addition stand-in used inside region bodies.
        Add,

        /// Region-free unary identity stand-in with an observable ordered-IO effect.
        Effectful,

        /// Region-carrying operation declaring its region slots. Its inferred output types are the first attached
        /// region's output types, which pins that region interfaces are derived and delivered during inference.
        WithRegions(&'static [RegionSlot]),
    }

    impl Operation for TestRegionOperation {
        type Type = DataType;

        fn name(&self) -> &'static str {
            match self {
                Self::Add => "add",
                Self::Effectful => "effectful",
                Self::WithRegions(_) => "with_regions",
            }
        }

        fn region_slots(&self) -> &'static [RegionSlot] {
            match self {
                Self::Add | Self::Effectful => &[],
                Self::WithRegions(slots) => slots,
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[DataType],
            region_interfaces: &[RegionInterface<DataType>],
        ) -> Result<Vec<DataType>, TypeError> {
            match self {
                Self::Add => {
                    check_count!("input", input_types, 2, TypeError);
                    Ok(vec![input_types[0]])
                }
                Self::Effectful => {
                    check_count!("input", input_types, 1, TypeError);
                    Ok(vec![input_types[0]])
                }
                Self::WithRegions(names) => {
                    check_count!("input", input_types, 1, TypeError);
                    if region_interfaces.len() != names.len() {
                        return Err(TypeError::invalid(format!(
                            "expected {} region interfaces but got {}",
                            names.len(),
                            region_interfaces.len(),
                        )));
                    }
                    Ok(region_interfaces[0].output_types().to_vec())
                }
            }
        }

        fn effects(&self) -> Effects {
            match self {
                Self::Add | Self::WithRegions(_) => Effects::PURE,
                Self::Effectful => Effects::single(Effect::OrderedIo),
            }
        }
    }
}
