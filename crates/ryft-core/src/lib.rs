// Derive macros emitted by `ryft-macros` use the public `ryft::...` facade path by default. This `self`-alias
// lets those same generated paths resolve when those macros are used inside `ryft-core` itself.
extern crate self as ryft;

pub mod arrays;
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
pub mod tracing;
pub mod tracing_v2;
pub mod utilities;

// TODO(eaplatanios): Make all of the following more specific.
// `arrays` shares its public `sharding` and `types` submodule names with `operations` and `programs` respectively, so
// these glob exports make those crate-level names ambiguous. P9 replaces these temporary glob exports with an explicit
// facade; suppress the known ambiguity until that dedicated API increment.
#[allow(ambiguous_glob_reexports)]
pub use arrays::*;
pub use axes::{AXIS_INDEX_OPERATION_NAME, Axes, Axis, AxisError, AxisIndex, AxisIndexOperation, NamedAxes, NamedAxis};
// Both `backends` and `operations` expose a public `dimensions` module, so these glob exports make the crate-level
// `dimensions` name ambiguous. P9 replaces these temporary glob exports with an explicit facade; suppress the known
// ambiguity until that dedicated API increment.
#[allow(ambiguous_glob_reexports)]
pub use backends::*;
pub use batching::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, ArrayIrBatch, ArrayIrBatching, Batch, BatchAxis,
    BatchAxisSpecification, BatchableOperation, BatchableType, BatchedProgram, BatchingContext, BatchingDriver,
    BatchingEntrypointPolicy, BatchingError, BatchingPolicy, BatchingPolicyProjection, BatchingTracer,
    BoundaryPreservingBatchedProgram, DimensionSource, InterpretableBatchableOperation, MemberBatchableOperation,
    ProgramBatchingOutputAxesPolicy, RecursiveBatchingPolicy, ReplicatedDimensionBatchingPolicy,
    StaticArrayBatchingPolicy, batch, batch_projected_operation,
};
pub use broadcasting::{Broadcastable, BroadcastingError};
pub use captures::{CaptureReference, CapturingContext, ClosedProgram};
pub use compilation::*;
pub use contexts::{Context, Domain, EagerContext, ProjectedContext, StagingContext, ValueResolution};
// Both `differentiation` and `programs` expose a public `types` module, so these glob exports make the crate-level
// `types` name ambiguous. P9 replaces these temporary glob exports with an explicit facade; suppress the known
// ambiguity until that dedicated API increment.
#[allow(ambiguous_glob_reexports)]
pub use differentiation::*;
pub use errors::{CustomError, Error, MaybeFallible};
pub use interpretation::{
    InterpretableOperation, InterpretationDriver, MemberInterpretableOperation, interpret_projected_operation,
};
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
pub use tracing::{
    DomainTracer, DomainTracingContext, NestedTracer, NestedTracingContext, Trace, Tracer, TracerState, TracingContext,
    infer_output_type, trace,
};
pub use tracing_v2::custom_derivatives::{CustomJvpOperation, CustomVjpOperation};
pub use tracing_v2::rematerialization::RematerializeOperation;

#[cfg(test)]
pub(crate) mod tests {
    use std::cell::Cell;

    use crate::arrays::ArrayType;
    use crate::batching::{
        BatchAxis, BatchingContext, BatchingDriver, BatchingError, ProgramBatchingOutputAxesPolicy,
        RecursiveBatchingDriver, RecursiveBatchingPolicy,
    };
    use crate::contexts::Context;
    use crate::macros::check_count;
    use crate::programs::effects::{Effect, Effects};
    use crate::programs::operations::Operation;
    use crate::programs::programs::Program;
    use crate::programs::regions::{RegionDriver, RegionInterface, RegionRef, RegionSlot};
    use crate::programs::types::TypeError;
    use crate::programs::values::Value;

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
        type Type = ArrayType;

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
            input_types: &[ArrayType],
            region_interfaces: &[RegionInterface<ArrayType>],
        ) -> Result<Vec<ArrayType>, TypeError> {
            match self {
                Self::Add => {
                    check_count!("input", input_types, 2, TypeError);
                    Ok(vec![input_types[0].clone()])
                }
                Self::Effectful => {
                    check_count!("input", input_types, 1, TypeError);
                    Ok(vec![input_types[0].clone()])
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

    /// [`BatchingDriver`] that counts the structural [`batch_program`](BatchingDriver::batch_program) requests a
    /// region-carrying batching rule makes, delegating every request to the ordinary [`RecursiveBatchingDriver`] over
    /// the same regions. Region-carrying rules discover their nested programs' natural output axes before instantiating
    /// them at reconciled targets, and reuse a discovery program when its axes already match. This fixture lets rule
    /// tests pin how many structural passes such a rule actually performs, which a program rendering alone cannot
    /// observe.
    pub(crate) struct CountingBatchingDriver<'r, V: Value, O: Operation<Type = V::Type>> {
        /// [`Region`]s attached to the operation application under test, in operation-defined order.
        regions: &'r Vec<Program<V, O, Vec<V>, Vec<V>>>,

        /// Number of structural program-batching requests observed so far.
        batch_program_calls: Cell<usize>,
    }

    impl<'r, V: Value, O: Operation<Type = V::Type>> CountingBatchingDriver<'r, V, O> {
        /// Creates a new [`CountingBatchingDriver`] over the provided attached regions.
        pub(crate) fn new(regions: &'r Vec<Program<V, O, Vec<V>, Vec<V>>>) -> Self {
            Self { regions, batch_program_calls: Cell::new(0) }
        }

        /// Returns the number of structural program-batching requests observed so far.
        pub(crate) fn batch_program_calls(&self) -> usize {
            self.batch_program_calls.get()
        }
    }

    impl<V: Value, O: Operation<Type = V::Type>> RegionDriver<V, O> for CountingBatchingDriver<'_, V, O> {
        fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, V, O>>
        where
            V: 'r,
            O: 'r,
        {
            self.regions.regions()
        }
    }

    impl<C: Context, P: RecursiveBatchingPolicy<C>> BatchingDriver<C, P>
        for CountingBatchingDriver<'_, C::Constant, C::Operation>
    {
        fn batch_region(
            &self,
            context: &BatchingContext<C, P>,
            index: usize,
            inputs: Vec<P::Batch>,
        ) -> Result<Vec<P::Batch>, BatchingError> {
            RecursiveBatchingDriver::new(self.regions).batch_region(context, index, inputs)
        }

        fn batch_program(
            &self,
            context: &BatchingContext<C, P>,
            region: RegionRef<'_, C::Constant, C::Operation>,
            input_axes: &[BatchAxis],
            output_axes_policy: ProgramBatchingOutputAxesPolicy,
        ) -> Result<P::BatchedProgram, BatchingError> {
            self.batch_program_calls.set(self.batch_program_calls.get() + 1);
            RecursiveBatchingDriver::new(self.regions).batch_program(context, region, input_axes, output_axes_policy)
        }
    }
}
