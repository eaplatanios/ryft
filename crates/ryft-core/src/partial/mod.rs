//! Partially evaluates [`Program`]s into work available in a known-side [`Context`] and work deferred to a residual
//! program.
//!
//! Partial evaluation is a transform boundary. Each input is classified as a concrete or symbolic value available
//! to the parent context, or as an unknown value represented only by its [`Type`]. Operations whose results can be
//! established from known inputs bind through the parent context. Work that depends on an unknown value is recorded in
//! a residual [`ProgramBuilder`], together with the minimum boundary needed to run it later. Finalization returns the
//! residual program plus descriptors that reconnect its inputs and outputs to the original program. Refer to the
//! documentation of [`PartialEvaluationContext`] for a rendered diagram of this split and to the documentation of
//! [`PartitionedProgram`] for the corresponding two-program wiring.
//!
//! Partial evaluation is both a public specialization transform and infrastructure for other transforms. In
//! linearization, for example, primals are known, tangents are unknown, and the residual tangent program becomes the
//! reusable linear computation.
//!
//! # Choosing an Entry Point
//!
//!   - [`Program::partially_evaluate`] is the eager specialization entry point for a flat program. It executes known
//!     work immediately and returns a [`PartialEvaluation`] carrying concrete known values and a residual program.
//!   - [`Program::partially_evaluate_in_context`] performs the same split relative to an explicit known-side context.
//!     With a staging context, known work is appended to an enclosing program instead of executed. The matching
//!     [`RegionRef::partially_evaluate_in_context`] method applies the transform to a borrowed sealed region without
//!     first materializing it as a standalone program.
//!   - [`Program::partition`] and [`RegionRef::partition`] reify both sides of the split as a [`PartitionedProgram`]:
//!     a known program, a residual program, and positional wiring between them.
//!   - [`PartialEvaluation::interpret`] supplies the surviving unknown inputs, runs the residual program in the same
//!     context family, and reconstructs the original outputs in their original order.
//!
//! # Known Work and Residual Work
//!
//! _Known_ means available in the parent context; it does not necessarily mean host-concrete. An eager parent executes
//! an all-known operation immediately. A staging parent binds the same operation into its enclosing program, making
//! the resulting tracer known to this partial-evaluation level. Mixed or unknown operations are offered to their
//! [`PartiallyEvaluatableOperation`] rule and ordinarily emitted into the residual program.
//!
//! Operation-owned rules may make a more precise split. A condition with a concretizable known predicate can inline
//! only its selected branch, for example. If a known value cannot be resolved or concretized through the parent
//! context, the rule must preserve it conservatively rather than inspect unavailable runtime data.
//!
//! # Values and Residual Materialization
//!
//! [`PartialValue`] carries only semantic classification: [`Known`](PartialValue::Known) contains a parent-context
//! value available now, while [`Unknown`](PartialValue::Unknown) carries the type of a future value.
//! [`PartialEvaluationValue`] adds a shared [`PartialValueMaterialization`] slot describing how that logical value
//! crosses into residual work. A known value may become a residual input or an inline residual constant. An unknown
//! value is already a residual variable. The first residual consumer assigns an atom, and every clone reuses it.
//! Staged-identity deduplication additionally merges distinct known values that name the same outer-program atom.
//!
//! Literal constants remain constants in the residual program. Known variables needed by residual work become
//! [`Known`](PartialEvaluationInput::Known) feeders, while original unknown inputs become
//! [`Unknown`](PartialEvaluationInput::Unknown) feeders. This distinction keeps runtime values out of staged constant
//! payloads while avoiding duplicate boundary inputs.
//!
//! # Results and Wiring
//!
//! [`PartialEvaluation`] owns one residual program. Its [`PartialEvaluationInput`] sequence is ordered like that
//! program's inputs and carries either a known feeder value or an original unknown-input index. Its
//! [`PartialEvaluationOutput`] sequence is ordered like the original outputs and carries either a folded value or a
//! residual-output index. [`PartialEvaluation::interpret`] follows those two mappings to replay and reassemble.
//!
//! [`PartitionedProgram`] expresses the same split without retaining parent-context values. It replaces feeder and
//! output values with positions, yielding a known program whose trailing outputs are residual edges and a residual
//! program that consumes those edges together with the original unknown inputs, each named by a
//! [`ResidualInputSource`]. Boundary operations whose two halves become separate operations, and rematerialized calls,
//! [forward](PartitionedProgram::forward_residuals) the edges that merely repeat original known inputs or fully known
//! outputs, so that the residual program consumes those values directly. The result is still a
//! [`PartitionedProgram`], whose residual inputs then also name those known inputs and outputs.
//!
//! # Identity, Concretization, and Failure Propagation
//!
//! [`PartialTracer`] equality is logical transform identity—two live tracers compare equal only when they share one
//! materialization slot, not when their eventual payloads are equal. This conservative identity is used by fixed-point
//! and passthrough analyses. Host control flow can inspect a known tracer only when its parent context resolves it to
//! a constant supporting the requested concretization; unknown and opaque values remain residual.
//!
//! Binding failures are deferred through poisoned [`PartialTracer`]s so infallible operator syntax can continue to
//! construct the surrounding closure. The context retains the first failure even when outputs are discarded or absent,
//! skips subsequent operations, and reports the original [`ProgramError`] at the partial-evaluation boundary. Escaped
//! context or value clones keep shared builders alive and are rejected during finalization with
//! [`ProgramError::EscapedProgramBuilder`].
//!
//! # Control Flow, Effects, and Recursion
//!
//! Higher-order rules receive a [`PartialEvaluationDriver`] for recursively transforming attached regions. A rule may
//! inline selected nested work. Uninlined mixed work remains attached to a residual operation. Effectful operations
//! fold when their inputs are known, subject to the ordering rules on [`PartialEvaluationContext`]: once an ordered
//! operation residualizes, every later ordered operation residualizes, including accesses of other references and
//! other effect classes. This preserves observable failures
//! and synchronization before later I/O and mutations. Splitting separately invoked programs requires source-level
//! validation of reference dependencies and allocation lifetimes. Live references cross the
//! boundary by identity as [`Known`](PartialEvaluationInput::Known) reference feeders (refer to
//! [`PartialEvaluation::known_reference_inputs`] for more information), reference-typed constants embed inline,
//! and a [`ReferencePlacement`] decides whether known reference operations execute or stage under an eager parent.
//! Speculative fixed-point probes never execute effectful bodies.
//!
//! # Extending Partial Evaluation
//!
//! Implement [`PartiallyEvaluatableOperation`] for operation payloads. Most operations use the default
//! [`PartialEvaluationContext::fold_or_residualize`] policy; control flow, loops, scans, and other higher-order
//! operations may override it to preserve more known work. Use the supplied context and driver to materialize values,
//! residualize operations, and recurse into regions rather than constructing boundary atoms independently. Rules that
//! inspect known payloads must first establish [`Constant`](ValueResolution::Constant) resolution and fall back
//! conservatively when it is unavailable.

#[cfg(doc)]
use crate::contexts::{Context, ValueResolution};
#[cfg(doc)]
use crate::programs::{Program, ProgramBuilder, ProgramError, RegionRef, Type};

pub mod contexts;
pub mod evaluations;
pub mod operations;
pub mod partitions;
pub mod residuals;
pub mod values;

pub use contexts::{PartialEvaluationContext, PartialTracer, ReferencePlacement};
pub use evaluations::PartialEvaluation;
pub use operations::{PartialEvaluationDriver, PartiallyEvaluatableOperation};
pub use partitions::{PartitionMetadata, PartitionedProgram};
pub use residuals::{
    ErasedResidualStorage, NativeResidualPolicies, NoStorage, ProjectionFallbackCandidate, ResidualCandidate,
    ResidualDecision, ResidualPolicy, ResidualPolicyError, ResidualPolicyReference, ResidualProducer,
    ResidualRejection, ResidualStorage,
};
pub use values::{
    PartialEvaluationInput, PartialEvaluationOutput, PartialEvaluationValue, PartialValue, PartialValueMaterialization,
    ResidualInputSource,
};

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashSet;

    use crate::arrays::{Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayType, DataType};
    use crate::captures::CaptureReference;
    use crate::contexts::StagingContext;
    use crate::operations::{ConditionOperation, ReferenceReadOperation, ReferenceWriteOperation};
    use crate::parameters::Placeholder;
    use crate::partial::contexts::{PartialEvaluationContext, ReferencePlacement};
    use crate::partial::values::PartialEvaluationValue;
    use crate::programs::{InstructionId, Program, ProgramBuilder, ReferenceAnalysis, ReferenceType, RegionRef};
    use crate::tracing::TracingContext;

    pub(super) type TestValue = ArrayIrValue<Array>;
    pub(super) type TestOperation = ArrayIrOperation<Array>;
    pub(super) type TestCapture = CaptureReference<ArrayIrType>;

    /// Builds a nested residual read followed by a write to an independently supplied reference.
    pub(super) fn reference_ordering_program() -> Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>> {
        let reference_type: ArrayIrType = ReferenceType::new(ArrayType::scalar(DataType::F32)).into();
        let mut branch = ProgramBuilder::<TestValue, TestOperation>::new();
        let source = branch.add_input(reference_type.clone());
        branch.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![source], None).unwrap();
        let branch = branch.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let destination = builder.add_input(reference_type.clone());
        let source = builder.add_input(reference_type);
        let predicate = builder.add_constant(TestValue::Array(Array::scalar(true).unwrap()));
        let zero = builder.add_constant(TestValue::Array(Array::scalar(0.0_f32).unwrap()));
        let true_branch = builder.import_program(branch.clone());
        let false_branch = builder.import_program(branch);
        builder
            .add_instruction(ConditionOperation::new(), vec![true_branch, false_branch], vec![predicate, source], None)
            .unwrap();
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![destination, zero], None)
            .unwrap();
        builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
            .unwrap()
    }

    /// Replays a region with optional instruction observation and returns both emitted programs for comparison.
    pub(super) fn replay_reference_ordering_program(
        region: RegionRef<'_, TestValue, TestOperation>,
        observations: Option<&RefCell<Vec<InstructionId>>>,
        reference_analysis: Option<&ReferenceAnalysis>,
    ) -> (
        Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>,
        Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>,
    ) {
        let parent = TracingContext::<TestValue, TestOperation>::new();
        let context = PartialEvaluationContext::new(parent.clone()).with_reference_placement(ReferencePlacement::Stage);
        let inputs = vec![
            PartialEvaluationValue::known_input(parent.input(region.input_types()[0].clone())),
            context.unknown_input(region.input_types()[1].clone(), 1),
        ];
        let outputs = context.inline_region(region, inputs, &HashSet::new(), observations, reference_analysis).unwrap();
        assert!(outputs.is_empty());
        let residual = context.into_evaluation(outputs).unwrap().program;
        let known = parent
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        (known, residual)
    }
}
