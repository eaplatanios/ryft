//! Conditional execution through [`ConditionOperation`], which evaluates one of two attached [`Region`](crate::Region)s
//! according to a scalar Boolean predicate, and through the [`Condition`] capability, which traces two branch functions
//! over structured inputs and stages a condition that selects between them.
//!
//! A condition application takes the predicate followed by one input per branch input. Both branches receive the
//! same inputs and must return outputs of the same types after the facts that those inputs establish refine them.
//! [`ConditionType`] defines which types may be predicates and which output types are valid under each predicate in a
//! type universe. For example, an array output must record every manual mesh axis that its predicate varies over.
//! Each reference output must denote the same reference in both branches, and under a varying predicate a branch may
//! only mutate references whose referents vary over the predicate's axes. Every transformation rule validates this
//! complete contract before it selects, rewrites, or stages either branch, so acceptance depends neither on the
//! transformation nor on the value of the predicate. The rules are:
//!
//!   - **Interpretation:** Only the selected branch runs.
//!   - **Reference Discharge:** Both branches receive one shared state boundary, and only the references that either
//!     branch mutates publish their final states.
//!   - **Partial Evaluation:** A concrete known predicate inlines the selected branch, a known symbolic predicate
//!     splits the condition into a known and a residual condition that per-branch residual edges connect, and an
//!     unknown predicate keeps the condition whole.
//!   - **Batching:** A replicated predicate keeps one condition over the batched branches, while a batch-varying
//!     predicate evaluates both pure branches and selects their outputs per batch item.
//!   - **Differentiation:** The predicate carries no tangent. Forward mode stages one fused condition, separate-context
//!     linearization joins the linearizations of both branches over one shared residual boundary, and transposition
//!     selects between the transposed branches with the same predicate.
//!
//! The XLA backend lowers a condition to StableHLO [`if`](https://openxla.org/stablehlo/spec#if) operation.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::fmt::Display;
use std::marker::PhantomData;
use std::sync::Arc;

use ryft_macros::capability;

use crate::arrays::{
    ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrType,
    ArrayType, DimensionValue, ShardableType, Sharding,
};
use crate::batching::{
    BatchableOperation, BatchedProgram, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
    ProgramBatchingOutputAxesPolicy, batch_projected_operation,
};
use crate::contexts::{Context, ContextProjection, Domain, StagingContext};
use crate::differentiation::{
    CotangentAccumulator, CotangentDestinationKind, DifferentiableOperation, DifferentiableType,
    DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationError, DifferentiationPolicy,
    ResidualZeroProvider, TangentActivity, TransposableOperation, TransposedProgramCotangent, TranspositionContext,
    TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, check_types};
use crate::operations::Capability;
use crate::operations::arithmetic::AddOperation;
use crate::operations::collectives::parallel_vary::{ManualVariationAlignment, PARALLEL_VARY_OPERATION_NAME};
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::constants::zero::ZeroOperation;
use crate::operations::control_flow::select::{Select, SelectOperation};
use crate::operations::control_flow::{
    RegionReferenceSource, refine_output_types, validate_output_identities, validate_region_input,
};
use crate::operations::differentiation::stop_gradient::StopGradient;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::broadcasting::{
    Broadcast, BroadcastOperation, DynamicBroadcast, DynamicBroadcastOperation,
};
use crate::operations::manipulation::transposition::{Transpose, TransposeOperation};
use crate::operations::references::ReferenceNewOperation;
use crate::parameters::{Parameterized, ParameterizedFamily, Placeholder};
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationOutput, PartialEvaluationValue, PartialValue,
    PartiallyEvaluatableOperation, PartitionedProgram, ResidualInputSource,
};
use crate::programs::{
    AtomId, CalleeRegionDriver, Concretizable, EmptyRegionDriver, InputRegionProvenance, MaybeZero, Operation,
    OperationBoundaryPruning, OperationPayloadProjection, OperationProjection, OperationProvider,
    OutputRegionProvenance, Program, ProgramBuilder, ProgramError, ReferenceDischargeContext, ReferenceDischargeDriver,
    ReferenceDischargePolicy, ReferenceDischargeValue, ReferenceDischargeableOperation, ReferenceMemberType,
    ReferenceRoot, RegionDataFlow, RegionDriver, RegionInterface, RegionLiveness, RegionRef, RegionSlot, Type,
    TypeError, TypeIdentityPosition, TypeRefinements, Typed, Value, ValueProjection,
    discharge_positional_region_operation,
};
use crate::tracing::{NestedTracingContext, Tracer, TracingContext};

/// Type-family predicate and output semantics for [`ConditionOperation`].
///
/// Implementations define which types represent valid predicates and which output types are valid in a condition
/// controlled by a given predicate. These rules depend on the type family; agreement between the two branches' output
/// types alone does not establish that the selected result is valid under that predicate.
///
/// # Array Type Families
///
/// For example, [`ArrayType`] accepts rank-zero Boolean predicates, while [`ArrayIrType`] accepts only its rank-zero
/// Boolean array member. A first-class dimension describes an array extent rather than Boolean data, despite its scalar
/// runtime representation, and a reference is a mutable state handle rather than a predicate value.
///
/// A manual region is a [`ShardMapOperation`](crate::ShardMapOperation) body. It runs per device on local shards, with
/// explicit collectives for communication along the operation's selected mesh axes (its active manual mesh axes). A
/// scalar predicate may differ between devices along those axes and let them take different branches. Every condition
/// output must therefore record variation over each axis that the predicate varies over (refer to
/// [`Sharding::varying_manual_axes`] and [`validate_condition_output`](Self::validate_condition_output)). An invariant
/// predicate keeps every device on the same branch. The [`Condition::condition`] function aligns the whole condition
/// boundary with a varying predicate, varying the inputs before the branches and the branch outputs at their ends.
/// A condition staged directly whose inputs do not vary over the predicate's axes still evaluates and partially
/// evaluates, but its branches compute invariant-typed values from those inputs although devices take the branches
/// differently. Splitting it around its known work (i.e., partial evaluation with a symbolic known predicate and
/// separate-context linearization) would turn such values into invariant-typed outputs of a known condition that
/// differ across devices, so partial evaluation keeps the condition whole and separate-context linearization fails.
/// Transposing it with respect to such an input fails as well, because the input's cotangent differs across devices and
/// its mesh sum would run inside the branches. As for a `while` loop with a varying predicate, keeping collectives out
/// of branches that devices may take differently is the program's responsibility.
pub trait ConditionType: ShardableType {
    /// Returns whether this type represents a valid scalar Boolean predicate for [`ConditionOperation`].
    fn is_condition_predicate(&self) -> bool;

    /// Validates the type family's restrictions on returning a value of this type from a condition whose predicate
    /// has type `predicate`. [`ConditionOperation`] validates predicate admission and branch signature agreement
    /// separately; matching branch output types alone need not make their selected result valid under the predicate.
    ///
    /// For example, [`ArrayType`] outputs must record variation over every manual mesh axis that the predicate varies
    /// over, because devices may take different branches. [`ArrayIrType`] applies the same check to reference referents
    /// and rejects first-class dimension outputs under a varying predicate because they cannot record that variation.
    /// Type families without additional output restrictions can return `Ok(())`.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if this output type is invalid under `predicate` according to the type family's rules.
    fn validate_condition_output(&self, predicate: &Self) -> Result<(), TypeError>;
}

impl ConditionType for ArrayType {
    #[inline]
    fn is_condition_predicate(&self) -> bool {
        self.is_scalar() && self.data_type().is_boolean()
    }

    fn validate_condition_output(&self, predicate: &Self) -> Result<(), TypeError> {
        let Some(predicate_axes) = predicate.sharding().map(Sharding::varying_manual_axes) else {
            return Ok(());
        };
        let output_axes = self.sharding().map(Sharding::varying_manual_axes);
        if predicate_axes.iter().all(|axis| output_axes.is_some_and(|output_axes| output_axes.contains(axis))) {
            return Ok(());
        }
        Err(TypeError::invalid(format!(
            "`{CONDITION_OPERATION_NAME}` output `{self}` must vary over every manual axis that the \
             predicate `{predicate}` varies over, because devices may take different branches; \
             insert `{PARALLEL_VARY_OPERATION_NAME}` on the branch outputs",
        )))
    }
}

impl ConditionType for ArrayIrType {
    #[inline]
    fn is_condition_predicate(&self) -> bool {
        matches!(self, Self::Array(r#type) if r#type.is_condition_predicate())
    }

    fn validate_condition_output(&self, predicate: &Self) -> Result<(), TypeError> {
        match (self, predicate) {
            (Self::Array(output), Self::Array(predicate)) => output.validate_condition_output(predicate),
            (Self::Reference(output), Self::Array(predicate)) => output.referent().validate_condition_output(predicate),
            (Self::Dimension(_), Self::Array(predicate)) if predicate.is_varying() => Err(TypeError::invalid(format!(
                "`{CONDITION_OPERATION_NAME}` output `{self}` cannot record manual variation, so it cannot be \
                 produced under the varying predicate `{predicate}`",
            ))),
            (Self::Dimension(_), Self::Array(_)) | (_, Self::Dimension(_) | Self::Reference(_)) => Ok(()),
        }
    }
}

/// Canonical operation name for [`ConditionOperation`].
pub const CONDITION_OPERATION_NAME: &str = "condition";

/// [`Operation`] that evaluates one of its two attached branch [`Region`](crate::Region)s depending on a Boolean
/// predicate supplied as the first operation input (a scalar Boolean); the remaining operation inputs are forwarded
/// to the selected branch.
///
/// The branch computations are not part of this payload; they are [`Region`](crate::Region)s attached to the
/// [`Instruction`](crate::Instruction) applying the operation, in the [`region_slots`](Operation::region_slots)
/// order `["true", "false"]`, and semantic rules reach them through their driver-granted region access. Conditions
/// with owned branches supply the two branch [`Program`]s through the region driver passed to [`Context::bind`]. The
/// [`Condition::condition`] function stages a condition from two branch functions instead, and aligns its inputs and
/// branch outputs with a predicate that varies over manual mesh axes (refer to [`ConditionType`]).
///
/// A predicate that is already known while _building_ a program is naturally expressed with a plain Rust `if` that
/// chooses which operations to stage, so no `condition` operation is needed for it. A predicate that is staged as a
/// constant remains a conditional during ordinary staging, and the XLA backend, for example, can fold its
/// `stablehlo.if` operation away via [canonicalization](https://openxla.org/stablehlo/generated/stablehlo_passes)
/// and XLA's conditional simplification. Partial evaluation specializes a concrete known predicate by inlining the
/// selected branch; a known symbolic predicate retains conditional execution of both the known and residual branch
/// programs.
///
/// Batching a mapped predicate evaluates both pure branches and selects their outputs per item. The primal computations
/// of inactive branches still execute and can fail, but gradient propagation into their inputs is stopped before branch
/// evaluation so inactive non-finite derivatives do not contaminate the selected derivative. The selected branch must
/// still have a defined derivative.
///
/// Array IR staging with a mapped predicate also rejects branch-local type identities. Replaying both branches into one
/// parent trace loses their separate definition scopes, and freshening those definitions alone would leave any
/// dependent array shapes needing a runtime join. Dimension identities established by branch inputs remain supported.
///
/// Each reference output must denote the same reference (i.e., the same instruction input or captured reference)
/// in bot  branches. Forward-mode differentiation relies on this as well, because it gives both branches the same
/// reference-output tangent slots: a reference tangent that only one branch carried could not be padded in its peer
/// without inventing allocation state.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConditionOperation<T: Type> {
    /// [`PhantomData`] marker tying this [`Operation`] to the [`Type`] universe in which it is valid.
    marker: PhantomData<fn() -> T>,
}

impl<T: Type> ConditionOperation<T> {
    /// Creates a new [`ConditionOperation`]. The two branch [`Program`]s are supplied separately as the operation's
    /// attached regions via the region driver passed to [`Context::bind`]. [`Operation::infer_output_types`] validates
    /// that the branch interfaces agree and that the predicate input is a scalar Boolean.
    pub const fn new() -> Self {
        Self { marker: PhantomData }
    }
}

impl<T: ConditionType> ConditionOperation<T> {
    /// Validates that the two branches with interfaces `region_interfaces` declare the same input types. Both branches
    /// receive the same inputs, and staging can specialize a branch to the actual input types, which could otherwise
    /// make two different declarations equal (refer to [`Operation::validate_region_interfaces`]).
    fn validate_branch_input_types(region_interfaces: &[RegionInterface<T>]) -> Result<(), TypeError> {
        check_count!("region", region_interfaces, 2, TypeError);
        check_types!(@same, format!("`{CONDITION_OPERATION_NAME}` branch input"), [
            region_interfaces[0].input_types(),
            region_interfaces[1].input_types(),
        ]);
        Ok(())
    }

    /// Validates predicate admission before a rule or capability uses its sharding to align other values.
    fn validate_predicate_type(predicate_type: &T) -> Result<(), TypeError> {
        if predicate_type.is_condition_predicate() {
            return Ok(());
        }
        Err(TypeError::invalid(format!(
            "`{CONDITION_OPERATION_NAME}` predicate type must be a scalar boolean, but got `{predicate_type}`",
        )))
    }

    /// Validates the application of this condition to inputs of the types `input_types` (i.e., the predicate followed
    /// by one input per branch input) with the `true` and `false` `branches`, and returns its output types. Every rule
    /// other than reference discharge validates this complete contract before it selects, rewrites, or stages either
    /// branch, so that whether an application is accepted depends neither on the transform that reaches it nor on the
    /// branch that its predicate selects. The contract consists of the boundary that [`Operation::infer_output_types`]
    /// validates and of the reference contracts that only the branch bodies reveal (refer to
    /// [`validate_reference_output_roots`](Self::validate_reference_output_roots) and
    /// [`validate_reference_mutations`](Self::validate_reference_mutations)). Reference discharge validates the same
    /// reference contracts directly, because it exempts the references that it preserves from the mutation rule.
    fn validate_application<V: Value<Type = T>, O: Operation<Type = T>>(
        &self,
        input_types: &[T],
        branches: [RegionRef<'_, V, O>; 2],
    ) -> Result<Vec<T>, ProgramError> {
        let output_types = self.infer_output_types(input_types, &branches.map(|branch| branch.interface()))?;
        self.validate_reference_output_roots(branches)?;
        self.validate_reference_mutations(&input_types[0], branches, |_| Ok(true))?;
        Ok(output_types)
    }

    /// Validates that every reference output of the `true` and `false` `branches` denotes the same reference whichever
    /// branch runs (i.e., the same instruction input or the same captured reference in both branches). Reference
    /// outputs keep their declared types, so type inference cannot check this: which reference a branch returns is
    /// only visible in its body. Every consumer of the condition still relies on it, because the caller keeps using
    /// the handle that the output denotes and reference discharge publishes one allocation per output. A reference
    /// that a branch allocates cannot leave that branch at all.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] when the branches disagree on which outputs are references (the signature mismatch that
    /// type inference reports, so that the capability, which validates the branches before binding them, never reports
    /// it as an internal error) or naming the first reference output whose branches denote different references or
    /// return references that they allocate, and [`ProgramError::MalformedProgram`] when the reference analysis of
    /// a branch reports a root that does not belong to that branch.
    fn validate_reference_output_roots<V: Value<Type = T>, O: Operation<Type = T>>(
        &self,
        branches: [RegionRef<'_, V, O>; 2],
    ) -> Result<(), ProgramError> {
        // Declared output types only agree after the input facts refine them, which inference checks. Here only the
        // reference kind of each output must agree, and a disagreement is that same signature mismatch rather than an
        // internal error, so that the capability, which validates the branches before binding them, reports it as one.
        let (true_output_types, false_output_types) = (branches[0].output_types(), branches[1].output_types());
        if true_output_types.len() != false_output_types.len()
            || true_output_types
                .iter()
                .zip(&false_output_types)
                .any(|(true_type, false_type)| true_type.is_reference() != false_type.is_reference())
        {
            check_types!(@same, format!("`{CONDITION_OPERATION_NAME}` branch output"), [
                true_output_types,
                false_output_types,
            ]);
        }

        if !true_output_types.iter().any(Type::is_reference) {
            return Ok(());
        }

        let [true_analysis, false_analysis] = [0, 1].map(|index| {
            branches[index].reference_analysis_with_configuration(self.region_capture_input_count(index), true, &[])
        });
        let (true_analysis, false_analysis) = (true_analysis?, false_analysis?);
        for (index, (true_root, false_root)) in
            true_analysis.output_roots().iter().zip(false_analysis.output_roots()).enumerate()
        {
            let (true_source, false_source) = match (true_root, false_root) {
                (None, None) => continue,
                (Some(true_root), Some(false_root)) => (
                    RegionReferenceSource::from_root(CONDITION_OPERATION_NAME, branches[0], *true_root)?,
                    RegionReferenceSource::from_root(CONDITION_OPERATION_NAME, branches[1], *false_root)?,
                ),
                _ => {
                    return Err(ProgramError::MalformedProgram(format!(
                        "`{CONDITION_OPERATION_NAME}` branches disagree on whether output {index} is a reference",
                    )));
                }
            };
            if true_source != false_source || true_source == RegionReferenceSource::Allocation {
                return Err(TypeError::invalid(format!(
                    "`{CONDITION_OPERATION_NAME}` reference output {index} must denote the same reference in both \
                     branches, but the `true` branch returns {true_source} and the `false` branch returns \
                     {false_source}",
                ))
                .into());
            }
        }
        Ok(())
    }

    /// Validates that the `true` and `false` `branches` mutate no reference whose referent does not vary over
    /// every manual axis that the predicate of type `predicate_type` varies over. Such a predicate lets devices take
    /// different branches, so a branch that mutates a reference input or a captured reference updates it on only some
    /// of those devices, and its final state must vary over those axes exactly like a condition output (refer to
    /// [`ConditionType::validate_condition_output`]). Reference analysis rather than the interface types of the
    /// branches decides which references a branch mutates, and the analysis is conservative: a mutation anywhere in a
    /// branch counts, including inside nested regions that may not run. A `while` loop needs no such rule, because its
    /// type inference already requires every reference state to vary over the axes of its predicate.
    ///
    /// # Parameters
    ///
    ///   - `predicate_type`: Type of the condition predicate.
    ///   - `branches`: The `true` and `false` branches of the condition.
    ///   - `is_published`: Returns whether the final state of a mutated reference is at stake. Staging and
    ///     interpretation check every mutated reference, while partial reference discharge exempts the references
    ///     that it preserves, because it publishes no final state for them.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] naming the branch and the first mutated reference that violates the rule, and
    /// [`ProgramError::MalformedProgram`] when the reference analysis of a branch reports a root that does not
    /// belong to that branch.
    fn validate_reference_mutations<V: Value<Type = T>, O: Operation<Type = T>>(
        &self,
        predicate_type: &T,
        branches: [RegionRef<'_, V, O>; 2],
        mut is_published: impl FnMut(RegionReferenceSource) -> Result<bool, ProgramError>,
    ) -> Result<(), ProgramError> {
        if !predicate_type.is_varying() {
            return Ok(());
        }

        for (index, (slot, branch)) in ["true", "false"].into_iter().zip(branches).enumerate() {
            // Branches inherit lexical captures unless the operation declares a fresh capture prefix. Analyze
            // that open scope directly instead of treating inherited captures as an empty explicit scope.
            let reference_analysis =
                branch.reference_analysis_with_configuration(self.region_capture_input_count(index), true, &[])?;
            let input_types = branch.input_types();
            for root in reference_analysis.roots().filter(|root| reference_analysis.is_mutated(*root)) {
                let reference_type = match root {
                    ReferenceRoot::RegionInput { region, input_index } if region == branch.id() => {
                        input_types[input_index].clone()
                    }
                    ReferenceRoot::Constant { value } => {
                        branch.with_id(value.region())?.atoms()[value.atom().index()].r#type().into_owned()
                    }
                    ReferenceRoot::RegionInput { .. } | ReferenceRoot::Allocation { .. } => {
                        // Allocations and the inputs of nested regions are local to the branch,
                        // so their states never cross the condition boundary.
                        continue;
                    }
                };
                let source = RegionReferenceSource::from_root(CONDITION_OPERATION_NAME, branch, root)?;
                if reference_type.validate_condition_output(predicate_type).is_err() && is_published(source)? {
                    return Err(TypeError::invalid(format!(
                        "`{CONDITION_OPERATION_NAME}` `{slot}` branch mutates {source} of type `{reference_type}`, \
                         whose referent does not vary over every manual axis that the predicate `{predicate_type}` \
                         varies over, so devices that take different branches would leave it different across \
                         devices; vary the referent over those axes",
                    ))
                    .into());
                }
            }
        }

        Ok(())
    }
}

impl<T: Type> Copy for ConditionOperation<T> {}

impl<T: Type> Default for ConditionOperation<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Type> Display for ConditionOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(CONDITION_OPERATION_NAME)
    }
}

impl<T: ConditionType> Operation for ConditionOperation<T> {
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        CONDITION_OPERATION_NAME
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        const { &[RegionSlot::computation("true"), RegionSlot::computation("false")] }
    }

    #[inline]
    fn validate_region_interfaces(&self, region_interfaces: &[RegionInterface<T>]) -> Result<(), ProgramError> {
        self.validate_region_count(region_interfaces.len())?;
        Ok(Self::validate_branch_input_types(region_interfaces)?)
    }

    fn infer_region_input_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<Option<Vec<T>>>, TypeError> {
        check_count!("region", region_interfaces, 2, TypeError);
        if input_types.is_empty() {
            return Err(TypeError::invalid(format!(
                "`{CONDITION_OPERATION_NAME}` expects at least one input but got 0",
            )));
        }

        // Replay can refine shapes, but cannot repair mismatched metadata or supply a missing variation contract, so
        // specialize only when both refined branch signatures agree and their outputs satisfy the predicate's manual
        // variation requirements. Compare inputs after refinement because these interfaces may be partially reconciled
        // (one branch already specialized and its peer not yet), and refine reference outputs too, so that specializing
        // the first branch does not prevent specializing its peer. `Operation::validate_region_interfaces` separately
        // requires the original branches to declare the same input types.
        let refined_signatures = region_interfaces
            .iter()
            .map(|interface| {
                // A pure identity substitution renames the branch identities to those of the caller. A valid mixed
                // refinement admits none and keeps the declared identities, while an invalid signature fails the
                // refinement below, so the derivation error itself carries no additional information here.
                let renaming = T::derive_identity_renaming(interface.input_types(), &input_types[1..]).ok();
                let rename = |types: &[T]| match &renaming {
                    Some(renaming) => types.iter().map(|r#type| r#type.rename_identities(renaming)).collect(),
                    None => Ok(types.to_vec()),
                };
                let (declared_inputs, declared_outputs) =
                    (rename(interface.input_types())?, rename(interface.output_types())?);
                let refinements = T::Refinements::new(&declared_inputs, &input_types[1..])?;
                let symbolic = declared_outputs
                    .iter()
                    .flat_map(Type::identities)
                    .filter(|&(position, _)| position == TypeIdentityPosition::Definition)
                    .map(|(_, identity)| identity.clone())
                    .collect::<Vec<_>>();
                let refine = |types: &[T], symbolic: &[T::Identity]| {
                    types
                        .iter()
                        .map(|r#type| refinements.refine_declared(r#type, symbolic))
                        .collect::<Result<Vec<_>, _>>()
                };
                Ok::<_, TypeError>((refine(&declared_inputs, &[])?, refine(&declared_outputs, &symbolic)?))
            })
            .collect::<Result<Vec<_>, _>>();

        if !refined_signatures.is_ok_and(|signatures| {
            signatures[0] == signatures[1]
                && signatures[0].1.iter().all(|output| output.validate_condition_output(&input_types[0]).is_ok())
        }) {
            return Ok(vec![None, None]);
        }

        if region_interfaces.iter().all(|interface| interface.input_types() == &input_types[1..]) {
            return Ok(vec![None, None]);
        }

        let branch_input_types = input_types[1..].to_vec();
        Ok(vec![Some(branch_input_types.clone()), Some(branch_input_types)])
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        // Both branches must declare the same input types, the predicate must be a scalar Boolean, and every
        // input must refine its declared branch input type (a reference input refines it by its referent, refer to
        // `validate_region_input`). The declared branch outputs must agree after applying the facts that the inputs
        // establish (e.g., `f64[extent]` and `f64[3]` agree for an `f64[3]` input, and a `ref<f64[extent]>` output
        // becomes `ref<f64[3]>` for a `ref<f64[3]>` input), and every output must satisfy the predicate's manual
        // variation requirements (refer to `ConditionType::validate_condition_output`).
        Self::validate_branch_input_types(region_interfaces)?;
        let (true_interface, false_interface) = (&region_interfaces[0], &region_interfaces[1]);
        let branch_input_types = true_interface.input_types();
        check_count!("input", input_types, branch_input_types.len() + 1, TypeError);
        Self::validate_predicate_type(&input_types[0])?;
        for (index, (declared, actual)) in branch_input_types.iter().zip(&input_types[1..]).enumerate() {
            validate_region_input(CONDITION_OPERATION_NAME, index + 1, actual, declared, "branch input type")?;
        }

        // Which input a branch forwards to a reference output is only visible in its body, so reference outputs are
        // refined by the input facts like value outputs rather than taking the type of one input. Output-defined
        // identities stay symbolic, rather than inheriting facts about the entering values. Compare the two refined
        // output signatures before checking their variation.
        let [true_output_types, false_output_types] = [true_interface, false_interface].map(|interface| {
            refine_output_types(branch_input_types, &input_types[1..], interface.output_types(), |_| None)
        });
        let (true_output_types, false_output_types) = (true_output_types?, false_output_types?);
        check_types!(@same, format!("`{CONDITION_OPERATION_NAME}` branch output"), [
            true_output_types,
            false_output_types,
        ]);
        for output_type in &true_output_types {
            output_type.validate_condition_output(&input_types[0])?;
        }
        validate_output_identities(CONDITION_OPERATION_NAME, input_types, &true_output_types)?;
        Ok(true_output_types)
    }

    #[inline]
    fn input_region_provenance(&self, region_index: usize, input_index: usize) -> InputRegionProvenance {
        if region_index < 2 {
            InputRegionProvenance::Input { index: input_index + 1 }
        } else {
            InputRegionProvenance::None
        }
    }

    #[inline]
    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        vec![
            OutputRegionProvenance { region_index: 0, output_index },
            OutputRegionProvenance { region_index: 1, output_index },
        ]
    }

    #[inline]
    fn region_data_flow(&self) -> RegionDataFlow<'_> {
        RegionDataFlow::Provenance
    }

    fn prune_boundary(
        &self,
        input_types: &[Self::Type],
        used_outputs: &[bool],
        regions: &mut dyn RegionLiveness,
    ) -> Result<Option<OperationBoundaryPruning<Self>>, ProgramError> {
        // Both branches keep one shared boundary, whose region inputs the instruction inputs after the predicate supply
        // positionally, so an instruction input is kept when either branch uses the region input it supplies (as in
        // JAX's `_cond_dce_rule`). The predicate is always kept.
        let mut used_inputs = vec![false; input_types.len().saturating_sub(1)];
        for region_index in 0..2 {
            let branch_inputs = regions.used_region_inputs(region_index, used_outputs)?;
            for (used, branch_used) in used_inputs.iter_mut().zip(branch_inputs) {
                *used |= branch_used;
            }
        }
        Ok(Some(OperationBoundaryPruning {
            operation: *self,
            kept_inputs: std::iter::once(true).chain(used_inputs).collect(),
            kept_outputs: used_outputs.to_vec(),
        }))
    }
}

impl<
    C: Context<Type: ConditionType + From<P::Referent>, Operation: From<ConditionOperation<C::Type>>>,
    P: ReferenceDischargePolicy<C>,
> ReferenceDischargeableOperation<C, P> for ConditionOperation<C::Type>
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        // A condition's branches mirror its input list after the leading predicate, and its results are each branch's
        // own outputs, which is exactly the positionally forwarding shape the shared structured rewrite serves. Both
        // branches therefore receive one shared state boundary: every root either branch touches enters, and only the
        // roots one of them mutates are published back, so a condition whose branches merely read keeps its source
        // boundary unchanged. The rewrite needs every reference output to denote one allocation whichever branch runs,
        // and a final state that a branch publishes under a predicate that varies over manual mesh axes to vary over
        // those axes as well. Both are validated before the branches are rewritten (refer to
        // `ConditionOperation::validate_reference_output_roots` and
        // `ConditionOperation::validate_reference_mutations`), so that a violation is reported as such instead of
        // failing the rebuilt condition with a diagnostic about one of its added outputs. A reference that partial
        // discharge preserves keeps its handle and publishes no final state, so its mutations are not checked here.
        let branches = driver.condition_branches(inputs.len())?;
        self.validate_reference_output_roots(branches)?;
        let predicate_type = inputs[0].try_as_value("a condition predicate")?.r#type().into_owned();
        self.validate_reference_mutations(&predicate_type, branches, |source| {
            let allocation = match source {
                RegionReferenceSource::Input(index) => {
                    // Branch input `index` is condition input `index + 1`, after the predicate.
                    context.boundary_allocation(&inputs[index + 1])?
                }
                RegionReferenceSource::Capture(index) => context.captures().allocations().get(index).copied().flatten(),
                RegionReferenceSource::Constant(_) | RegionReferenceSource::Allocation => None,
            };
            allocation.map_or(Ok(true), |allocation| context.is_allocation_discharged(allocation))
        })?;
        discharge_positional_region_operation(self, context, driver, inputs, 1, |_, _| Ok(*self))
    }
}

impl<C: Domain<Type: ConditionType, Value: Concretizable<bool>>> InterpretableOperation<C>
    for ConditionOperation<C::Type>
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // Interpretation validates the complete application before selecting a branch, including the unselected peer's
        // signature and reference contracts, because direct eager binding does not pass through program construction.
        // Only the selected branch runs (i.e., region `0` for `true` and region `1` for `false`). Eager region entry
        // then validates the complete runtime input signature, including shared dimensions and concrete reference
        // referents, and compares outputs using those same refinements. The `shard_map` emulator uses this same rule
        // with values holding one value per device, but its lockstep region replay validates neither the region entry
        // nor the region outputs, so only the validation of this rule applies there.
        let branches = [driver.region(0)?, driver.region(1)?];
        let expected = branches[0].input_ids().len() + 1;
        if inputs.len() != expected {
            return Err(ProgramError::InvalidInputCount { expected, actual: inputs.len() });
        }
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.validate_application(&input_types, branches)?;
        let branch = if inputs[0].concretize()? { 0 } else { 1 };
        driver.interpret_region(context, branch, inputs[1..].to_vec())
    }
}

impl<
    C: Context<Type: ConditionType, Constant: Concretizable<bool>, Operation = O>,
    O: Operation<Type = C::Type>
        + OperationProvider<C::Type, ZeroOperation<C::Type>, Operation = O>
        + From<ConditionOperation<C::Type>>,
> PartiallyEvaluatableOperation<C> for ConditionOperation<C::Type>
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // With a `Known` predicate that the known-side context can `Context::resolve` to a `Constant` payload whose
        // value can be concretized as a Boolean, it selects the taken branch and always inlines it over the remaining
        // inputs, so the condition disappears from the residual program. Inlining enters the branch through the shared
        // guarded region entry, which stages a guard for every relationship between its inputs that their actual types
        // leave unproven (e.g., a computed `dimension<2>` next to an unknown `f32[n]`, both entering at `n`). The guard
        // stays in the residual program when one of its members is unknown and keeps later ordered known effects of the
        // inlined body behind that check, so the residual program accepts and rejects the same inputs that the branch
        // entry would. A known predicate that is _not_ concretizable (under a staging known-side context, a genuine
        // `Tracer` into the outer program) cannot select a branch at partial evaluation time. The condition is instead
        // split after guarding its inputs through the same entry contract into a _known_ condition bound in the
        // enclosing known-side context and a _residual_ condition over the unknown work, connected by per-branch
        // residual edges. Keeping known branch work behind the known condition avoids staging it speculatively for
        // both branches.
        //
        // With an `Unknown` predicate the condition stays whole. Even pure known work must remain behind its branch:
        // evaluating an untaken branch speculatively can fail or never terminate, for example when it contains an
        // unbounded `while` whose condition is always true. Residualizing the whole conditional also preserves
        // reference placement, deferred work, and the ordering of effects with later operations.

        // Specializing a concrete predicate or splitting symbolic branches rewrites the original condition. Validate
        // the complete application first, including direct context binding and conditions that return no values, so
        // that a concrete predicate cannot skip the contracts of the branch that it does not select.
        let branches = driver.condition_branches(inputs.len())?;
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.validate_application(&input_types, branches)?;

        // A known predicate selects a branch only when it resolves to a program constant: under a staging known-side
        // context "known" means known to the outer program, and a genuine tracer carries no boolean to branch on. A
        // known-but-symbolic predicate (or a program constant payload that exposes no concrete boolean, such as an
        // abstract backend capture reference) keeps the conditional on both sides of the split instead. The rule
        // requests all nested-computation work through its region access (region `0` is the `true` branch and region
        // `1` the `false` branch), which keeps its bounds free of the operation family's own semantic traits.
        let selected_branch = inputs[0]
            .as_known()
            .and_then(|predicate| context.parent().resolve(predicate).into_constant())
            .and_then(|predicate| predicate.concretize().ok())
            .map(|predicate| if predicate { 0 } else { 1 });
        if let Some(index) = selected_branch {
            return driver.partially_evaluate_region(context, index, inputs[1..].to_vec());
        }

        let mut inputs = Cow::Borrowed(inputs);
        if inputs[0].is_known() && !inputs.iter().all(PartialEvaluationValue::is_known) {
            // The split replaces both branch entries, and each branch is partitioned at its declared input types,
            // so neither half would check a relationship that the entry checks between a known and an unknown input
            // (e.g., a known `f32[3]` next to an unknown `f32[n]`, both entering at `n`). Guard the inputs through the
            // shared entry contract first, exactly as inlining a selected branch does. The split binds two conditions
            // that join both branches' partitions rather than the programs of one partition, so it guards here instead
            // of binding through `PartialEvaluationContext::inline_partitioned_program`. Both branches declare the
            // same input types, so one guard covers both entries, and the guarded inputs determine which inputs are
            // known. A condition that stays whole receives the guarded inputs as well.
            let [true_branch, false_branch] = branches;
            let branch_input_types = true_branch.input_types();
            let mut guarded_inputs = vec![inputs[0].clone()];
            guarded_inputs.extend(context.guard_inputs(&branch_input_types, inputs[1..].to_vec(), false)?);
            let input_known = guarded_inputs[1..].iter().map(PartialEvaluationValue::is_known).collect::<Vec<_>>();

            // Partition each branch in its own fresh known-side context, whose inputs stand in for the known inputs,
            // so no branch work is staged speculatively in the caller. The driver supplies the fresh-trace semantics.
            // These partitions are not cached: they depend on the evaluation's effect-folding and residual-placement
            // configuration, which the branch and knownness mask alone do not determine.
            let true_partition = driver.partition_program(context, true_branch, &input_known)?;
            let false_partition = driver.partition_program(context, false_branch, &input_known)?;

            // An output is known only if both branches fold it. A value that only one branch folds crosses to the
            // residual condition as an edge. Both known branches return the shared signature `[known outputs, true
            // edges, false edges]`, and both residual branches receive `[unknown inputs, true edges, false edges]`,
            // reading only their own edges. The peer's unused edge slots need placeholders of the same types.
            if let Ok(outputs) = reconstruct_partitioned_condition(
                guarded_inputs[0].r#type().as_ref(),
                &branch_input_types,
                true_branch.output_types().len(),
                &guarded_inputs,
                &input_known,
                [true_partition, false_partition],
                |builder: &mut ProgramBuilder<_, _>, edge_type: &C::Type, _known_inputs: &[AtomId]| {
                    // Only identity-free types that admit zero can supply a placeholder. Inventing a reference or
                    // a dimension identity here would change the residual program's contract, so reject the split.
                    if edge_type.identities().next().is_some() || edge_type.validate_zero().is_err() {
                        return Ok(None);
                    }
                    let zeros = builder.add_instruction(
                        O::provide(ZeroOperation::new(edge_type.clone()), &[])?,
                        Vec::new(),
                        Vec::new(),
                        None,
                    )?;
                    check_count!("output", zeros, 1, ProgramError);
                    Ok(Some(zeros[0]))
                },
                |operation, programs, inputs: &[_]| context.fold_or_residualize(operation, programs, inputs),
                |operation, programs, inputs: &[_]| context.residualize(operation, programs, inputs),
            )? {
                return Ok(outputs);
            }

            // Reconstruction rejects unsupported splits before binding either half. Preserve the entry guards when
            // falling back to the whole condition, so it still checks the relationships of the original branch entry.
            inputs = Cow::Owned(guarded_inputs);
        }

        context.fold_or_residualize(O::from(*self), driver.region_programs(), &inputs)
    }
}

impl<
    O: Operation<Type = ArrayType>
        + From<TransposeOperation>
        + From<BroadcastOperation>
        + From<SelectOperation<ArrayType>>
        + From<ConditionOperation<ArrayType>>,
    C: Context<
            Type = ArrayType,
            Value: Broadcast + Transpose + Select + StopGradient + ManualVariationAlignment<ArrayType>,
            Operation = O,
        >,
    P: ArrayExtentBatchingPolicy<C>,
> BatchableOperation<C, ArrayBatchingPolicy<P>> for ConditionOperation<ArrayType>
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        // A replicated predicate binds one condition whose items all take the same branch. A mapped predicate evaluates
        // both pure branches, gates every input, and selects every output. Every per-item primitive re-enters this
        // operation family's batching rules in the active context, so the rewrite composes for eager and staging
        // parents alike.

        // Validate the original logical application after input refinement, before either lowering rewrites the
        // branches, so that a staging parent cannot skip it. Per-item types remove the mapped axis while retaining
        // input refinements.
        let branches = driver.condition_branches(inputs.len())?;
        let input_types = inputs.iter().map(ArrayBatch::unbatched_type).collect::<Vec<_>>();
        self.validate_application(&input_types, branches)?;

        let (predicate, branch_inputs) = (&inputs[0], &inputs[1..]);
        if predicate.batch_axis().is_replicated() {
            return context.batch_condition_with_replicated_predicate(
                driver,
                branches,
                predicate.value(),
                branch_inputs,
            );
        }

        // Both branches run for a mapped predicate, including work that an item does not select. Effects and
        // deferred transformation obligations cannot be selected per item, so both branches must be pure.
        if branches.iter().any(|branch| !branch.effects().is_pure()) {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "cannot batch a `{CONDITION_OPERATION_NAME}` with a batch-varying predicate and branches that have \
                     effects or deferred work because neither observable effects nor transformation obligations can be \
                     selected per batch item",
                ),
            });
        }

        context.batch_condition_with_mapped_predicate(
            driver,
            predicate,
            branch_inputs,
            |input| Ok(Some(ArrayBatch::new(input.value().stop_gradient()?, input.batch_axis())?)),
            |inputs| {
                let mut selected = SelectOperation::<ArrayType>::new().batch(context, &EmptyRegionDriver, inputs)?;
                check_count!("output", selected, 1, ProgramError);
                Ok(selected.remove(0))
            },
            |_, true_output, false_output, select| select(&true_output, &false_output),
        )
    }
}

impl<
    C: ContextProjection<
            ArrayType,
            Type = ArrayIrType,
            Value: DimensionSize
                       + DynamicBroadcast
                       + ValueProjection<
                ArrayType,
                Projected: Broadcast
                               + Select
                               + StopGradient
                               + Transpose
                               + ManualVariationAlignment<ArrayType>
                               + Value<Type = ArrayType>,
            >,
            Operation: OperationPayloadProjection
                           + OperationProjection<
                ArrayType,
                Projected: From<BroadcastOperation> + From<SelectOperation<ArrayType>> + From<TransposeOperation>,
            > + From<DynamicBroadcastOperation>
                           + From<ConditionOperation<ArrayIrType>>
                           + From<ConstantOperation<DimensionValue>>
                           + From<DimensionSizeOperation>,
        >,
> BatchableOperation<C, ArrayIrBatchingPolicy> for ConditionOperation<ArrayIrType>
{
    fn batch<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<Vec<ArrayIrBatch<C::Value>>, BatchingError> {
        // A replicated predicate binds one condition whose batched branches explicitly thread the mapped extent.
        // A batch-varying predicate evaluates both branches and selects their outputs per item, as for homogeneous
        // arrays,except that only array inputs are gated and only array outputs are selected. Dimension and reference
        // inputs pass through unchanged, a dimension output is replicated and equal in both branches, and a reference
        // output must forward the same branch input in both branches. The branches must not access or allocate
        // references, and a staging parent rejects branch-local type identities before either branch is replayed.

        // Validate the complete per-item application before structural batching or per-item selection rewrites it,
        // so that a staging parent cannot skip it.
        let branches = driver.condition_branches(inputs.len())?;
        let input_types = inputs.iter().map(ArrayIrBatch::unbatched_type).collect::<Vec<_>>();
        self.validate_application(&input_types, branches)?;
        let (predicate, branch_inputs) = (&inputs[0], &inputs[1..]);
        if predicate.batch_axis().is_replicated() {
            return context.batch_condition_with_replicated_predicate(
                driver,
                branches,
                predicate.value(),
                branch_inputs,
            );
        }

        // Both branches run, so even a read in the inactive branch would be ordered against writes in the selected
        // branch, and allocating a reference is itself an effect. Reject accesses and allocations before generic
        // purity; forwarding an untouched reference remains valid and its output roots are checked when merging.
        let [true_analysis, false_analysis] = branches.map(|branch| branch.reference_analysis(0));
        let analyses = [true_analysis.map_err(ProgramError::from)?, false_analysis.map_err(ProgramError::from)?];
        for analysis in &analyses {
            let allocates = analysis.roots().any(|root| matches!(root, ReferenceRoot::Allocation { .. }));
            if !analysis.access_modes().is_empty() || allocates {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "cannot batch a `{CONDITION_OPERATION_NAME}` with a batch-varying predicate whose branches \
                         access references because select lowering runs both branches and reference effects cannot \
                         be selected per batch item",
                    ),
                });
            }
        }

        // Check other effects and deferred work only after reference accesses, so the reference diagnostic names
        // the cause when both restrictions apply. Selecting outputs cannot suppress work in an inactive branch.
        if branches.iter().any(|branch| !branch.effects().is_pure()) {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "cannot batch a `{CONDITION_OPERATION_NAME}` with a batch-varying predicate and branches that have \
                     effects or deferred work because neither observable effects nor transformation obligations can be \
                     selected per batch item",
                ),
            });
        }

        // Staging both branches into one trace would collide their local identities across branches or invocations.
        // Freshening those identities would instead give dependent arrays unrelated shapes at selection. Identities
        // established by the inputs keep their existing authority; eager execution has concrete output dimensions.
        if !context.parent().is_eager() {
            for (slot, branch) in ["true", "false"].into_iter().zip(branches) {
                if let Some(identity) = branch.type_identity_signature().internal_identities().first() {
                    return Err(BatchingError::UnsupportedOperation {
                        message: format!(
                            "cannot stage a `{CONDITION_OPERATION_NAME}` with a batch-varying predicate whose `{slot}` \
                             branch defines branch-local type identity `{identity}` because select lowering cannot \
                             preserve branch-local type identity scopes",
                        ),
                    });
                }
            }
        }

        let branch_ids = branches.map(|branch| branch.id());
        context.batch_condition_with_mapped_predicate(
            driver,
            predicate,
            branch_inputs,
            |input| {
                // Array inputs keep their primal values in both branches but gradients enter only the selected one.
                // Dimensions and untouched references describe shared structural state and pass through unchanged.
                if !matches!(input.unbatched_type(), ArrayIrType::Array(_)) {
                    return Ok(None);
                }
                let stopped = ValueProjection::<ArrayType>::into_projected(input.value().clone())?.stop_gradient()?;
                Ok(Some(ArrayIrBatch::new(ValueProjection::<ArrayType>::from_projected(stopped), input.batch_axis())?))
            },
            |inputs| {
                let mut selected = batch_projected_operation(context, &SelectOperation::<ArrayType>::new(), inputs)?;
                check_count!("output", selected, 1, ProgramError);
                Ok(selected.remove(0))
            },
            |index, true_output, false_output, select| match true_output.unbatched_type() {
                ArrayIrType::Array(_) => {
                    <&ArrayType>::try_from(&false_output.unbatched_type())?;
                    select(&true_output, &false_output)
                }
                ArrayIrType::Dimension(_) => {
                    // Refined branch signatures already agree. Staged dimensions can name only input-established
                    // identities, and eager dimension types fix their exact values, so these replicated dimensions
                    // are equal without a runtime comparison.
                    true_output.validate_replicated_dimension()?;
                    false_output.validate_replicated_dimension()?;
                    Ok(true_output)
                }
                ArrayIrType::Reference(_) => {
                    // References cannot be selected per item. Forward the original input with its referent's batch
                    // axis only when both branches return that input. Check the root's region as well as its position
                    // so an input of a foreign region cannot index this condition's inputs.
                    let [true_root, false_root] =
                        [0, 1].map(|branch| analyses[branch].output_roots().get(index).copied().flatten());
                    match (true_root, false_root) {
                        (
                            Some(ReferenceRoot::RegionInput { region: true_region, input_index: true_input }),
                            Some(ReferenceRoot::RegionInput { region: false_region, input_index: false_input }),
                        ) if true_region == branch_ids[0]
                            && false_region == branch_ids[1]
                            && true_input == false_input
                            && true_input < branch_inputs.len() =>
                        {
                            Ok(branch_inputs[true_input].clone())
                        }
                        _ => Err(BatchingError::UnsupportedOperation {
                            message: format!(
                                "cannot batch a `{CONDITION_OPERATION_NAME}` with a batch-varying predicate whose \
                                 reference output {index} does not forward the same region input in both branches",
                            ),
                        }),
                    }
                }
            },
        )
    }
}

impl<
    C: Context<
            Type: ConditionType + DifferentiableType,
            Operation: ResidualZeroProvider<C::Type, Operation = C::Operation> + From<ConditionOperation<C::Type>>,
        >,
> DifferentiableOperation<C> for ConditionOperation<C::Type>
where
    TracingContext<C::Constant, C::Operation>:
        StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The predicate is the first input and carries no tangent as Boolean predicates have no tangent space. The rule
        // analyzes which branch inputs and outputs carry live tangents and differentiates each branch through its
        // instruction-scoped differentiation driver. A shared primal/tangent context stages one fused condition.
        // Separate contexts stage a primal condition that records the selected branch's residuals and a tangent
        // condition that consumes them, joining the two branch linearizations over one shared residual boundary as the
        // known/residual split of partial evaluation does. Placeholders for the peer branch's residual slots use live
        // input extents or forward a known input of the identical type, and a residual that admits no placeholder makes
        // separate-context linearization fail.

        // Direct differentiated binds can reach this rule before the original operation's output inference.
        // Validate the complete primal application before analyzing tangent slots or deriving either branch program;
        // specializing the attached regions alone validates only their input interfaces.
        let branches = driver.condition_branches(inputs.len())?;
        let input_types = inputs.iter().map(|input| input.primal().r#type().into_owned()).collect::<Vec<_>>();
        let output_count = self.validate_application(&input_types, branches)?.len();

        // Both branches receive the same live input tangents. Equal primal signatures and reference roots give them the
        // same tangent slots, including reference tangents, which cannot be padded with invented allocation state.
        let input_tangent_liveness = inputs[1..].iter().map(|input| !input.tangent().is_zero()).collect::<Vec<_>>();
        let branch_activities = [
            TangentActivity::new(driver, branches[0], &input_tangent_liveness)?,
            TangentActivity::new(driver, branches[1], &input_tangent_liveness)?,
        ];

        // An output tangent is live when it depends on an active input in either branch.
        // Both derivative paths use this shared layout, leaving inactive outputs as structural zeros.
        let output_activity = branch_activities[0]
            .output_tangent_liveness()
            .iter()
            .zip(branch_activities[1].output_tangent_liveness())
            .map(|(&left, &right)| left || right)
            .collect::<Vec<_>>();
        let live_tangent_slots = branch_activities[0].live_tangent_slots(&output_activity);
        let mut condition_inputs = vec![inputs[0].primal().clone()];
        condition_inputs.extend(inputs[1..].iter().map(|input| input.primal().clone()));
        condition_inputs.extend(inputs[1..].iter().filter_map(|input| input.tangent().as_value().cloned()));
        let outputs = if std::ptr::eq(context.primal(), context.tangent()) {
            // One condition returns both primals and live tangents, selecting the same branch for both. Keep a cached
            // branch handle when all of its outputs are already needed; otherwise project to the shared live slots.
            let jvp_outputs = (0..output_count)
                .chain(live_tangent_slots.iter().map(|&slot| output_count + slot))
                .collect::<Vec<_>>();
            let branches = branch_activities
                .iter()
                .map(|activity| {
                    let program = activity.jvp_program();
                    if jvp_outputs.iter().copied().eq(0..program.output_count()) {
                        Ok(program.clone())
                    } else {
                        program.with_outputs(&jvp_outputs).map(Arc::new)
                    }
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            context
                .primal()
                .bind(ConditionOperation::new(), CalleeRegionDriver::new(&branches), &condition_inputs)?
        } else {
            // Separate contexts require a primal condition that records the selected residuals and a tangent condition
            // that consumes them. Treat the primals as known and the live input tangents as unknown in both partitions.
            let primal_input_count = branches[0].input_types().len();
            let input_indices = branch_activities[0].input_indices();
            let live_input_count = input_indices.len();
            let mut partitions = Vec::with_capacity(2);
            for branch in branches {
                let (primal, tangent, residual_count) = driver.linearize_program(branch, input_indices)?.into_parts();
                partitions.push(PartitionedProgram::from_parts(
                    Arc::unwrap_or_clone(primal),
                    tangent.with_outputs(&live_tangent_slots)?,
                    primal_input_count + live_input_count,
                    (0..primal_input_count).collect(),
                    (primal_input_count..primal_input_count + live_input_count)
                        .map(ResidualInputSource::UnknownInput)
                        .chain((0..residual_count).map(ResidualInputSource::ResidualEdge))
                        .collect(),
                    (0..output_count)
                        .map(PartialEvaluationOutput::Known)
                        .chain((0..live_tangent_slots.len()).map(PartialEvaluationOutput::Unknown))
                        .collect(),
                )?);
            }

            // The boundary of the joined branches is the branch primals followed by the live branch tangents, whose
            // types the tangent program of either linearization declares first.
            let mut branch_input_types = branches[0].input_types();
            branch_input_types
                .extend(partitions[0].residual_program().input_types().into_iter().take(live_input_count));
            let input_known = (0..primal_input_count + live_input_count)
                .map(|index| index < primal_input_count)
                .collect::<Vec<_>>();
            let false_partition = partitions.pop().unwrap();
            let true_partition = partitions.pop().unwrap();
            let bind_residual = |operation, programs, inputs: &[C::Value]| {
                // The residual condition receives the predicate, the live tangents, and the residuals. The
                // predicate and the residuals are primal values, which enter the tangent context explicitly.
                let tangent_inputs = inputs
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        if index == 0 || index > live_input_count {
                            context.primal_to_tangent(value.clone()).map_err(ProgramError::from)
                        } else {
                            Ok(value.clone())
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                context.tangent().bind(operation, programs, &tangent_inputs)
            };

            // Join both branch linearizations over one residual signature. Each known branch supplies placeholders
            // for the peer's unused edges; a reference residual, invalid variation, or missing placeholder rejects
            // the join.
            let outputs = reconstruct_partitioned_condition(
                &input_types[0],
                &branch_input_types,
                output_count + live_tangent_slots.len(),
                &condition_inputs,
                &input_known,
                [true_partition, false_partition],
                |builder: &mut ProgramBuilder<C::Constant, C::Operation>,
                 edge_type: &C::Type,
                 known_inputs: &[AtomId]| {
                    // This value fills an unused residual slot of the peer branch. Its type and runtime identities
                    // must still match: construct a zero using live extents when possible, or forward/capture a
                    // known value instead.
                    let placeholder_context = TracingContext::<C::Constant, C::Operation>::new();
                    let sources = known_inputs
                        .iter()
                        .map(|input| placeholder_context.input(builder.atoms()[input.index()].r#type().into_owned()))
                        .collect::<Vec<_>>();

                    // Non-zeroable types have no typed zero. Type-only zeros also cannot supply reference-position
                    // identities such as the runtime extent of an array; the residual-zero protocol must capture
                    // those from known inputs.
                    let zero_is_constructible = edge_type.validate_zero().is_ok()
                        && (!C::Operation::zero_residual_types(edge_type).is_empty()
                            || edge_type.identities().all(|(position, _)| position != TypeIdentityPosition::Reference));
                    let placeholder = if zero_is_constructible {
                        C::Operation::materialize_zero_from_residual_sources(
                            &placeholder_context,
                            MaybeZero::Zero(edge_type.clone()),
                            &sources,
                        )?
                    } else if let Some(source) = sources.iter().find(|source| source.r#type().as_ref() == edge_type) {
                        source.clone()
                    } else {
                        // An input of another type may carry the required identity, for example an array carrying
                        // an extent.
                        let mut captured = None;
                        for source in &sources {
                            if let Some(value) =
                                C::Operation::capture_zero_residual_value(&placeholder_context, source, edge_type)?
                            {
                                captured = Some(value);
                                break;
                            }
                        }

                        let Some(captured) = captured else {
                            return Ok(None);
                        };

                        if captured.r#type().as_ref() != edge_type {
                            return Err(ProgramError::MalformedProgram(format!(
                                "`{}` residual placeholder has type `{}` but expected `{}`",
                                CONDITION_OPERATION_NAME,
                                captured.r#type().as_ref(),
                                edge_type,
                            )));
                        }

                        captured
                    };

                    // Build in a detached trace and splice into this known branch with its original inputs,
                    // preserving identities.
                    let placeholder_program =
                        placeholder_context.builder().borrow().clone().build::<Vec<C::Constant>, Vec<C::Constant>>(
                            vec![placeholder.atom_id()?],
                            vec![Placeholder; known_inputs.len()],
                            vec![Placeholder],
                        )?;
                    let placeholders = builder.splice_program(&placeholder_program, known_inputs)?;
                    check_count!("output", placeholders, 1, ProgramError);
                    Ok(Some(placeholders[0]))
                },
                |operation, programs, inputs: &[C::Value]| context.primal().bind(operation, programs, inputs),
                bind_residual,
            )?;

            outputs.map_err(|rejection| -> DifferentiationError {
                ProgramError::UnsupportedOperation {
                    message: format!(
                        "`{CONDITION_OPERATION_NAME}` linearization cannot join the residuals of its branches, \
                         because {rejection}",
                    ),
                }
                .into()
            })?
        };

        // Reconstruct structural zeros from the actual primal outputs so runtime type identities remain intact.
        check_count!("output", outputs, output_count + live_tangent_slots.len(), ProgramError);
        let mut primal_outputs = outputs;
        let tangent_outputs = primal_outputs.split_off(output_count);
        DifferentiationDual::from_live_tangents(primal_outputs, tangent_outputs, &output_activity)
    }
}

impl<
    T: ConditionType + DifferentiableType + ReferenceMemberType,
    V: Value<Type = T>,
    O: Operation<Type = T>
        + ResidualZeroProvider<T, Operation = O>
        + OperationProvider<T, AddOperation<T>, Operation = O>
        + OperationProvider<T, ReferenceNewOperation<T::Referent, T>, Operation = O>
        + From<ConditionOperation<T>>,
> TransposableOperation<V, O> for ConditionOperation<T>
where
    TracingContext<V, O>: StagingContext<Type = T, Constant = V, Operation = O>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        // The predicate and the per-branch residuals ride as ordinary known inputs, and the branch recursion happens
        // through the instruction-scoped driver's transposition requests, so instantiating this implementation for a
        // closed operation enum introduces no recursive `TransposableOperation` obligation on `O`. Reference inputs
        // accumulate through the enclosing context's cotangent references, which are resolved (and allocated on first
        // use when their state cotangent is live) before the shared rule passes them into both transposed branches.
        // A universe without reference types (e.g., `ArrayType`, whose referent family is `NoReferent`) has no
        // cotangent references, so every needed cotangent is returned as a value.
        let branch = driver.region(0)?;
        check_count!("input", inputs, 1 + branch.input_types().len(), ProgramError);
        check_count!("output", outputs, branch.output_types().len(), ProgramError);
        check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);

        // Resolve shared cotangent references before transposing either branch (the selected branch accumulates into
        // those references in place, while ignored destinations own no slot in the transposed boundary).
        let cotangents = context.cotangent_destinations(driver, inputs, accumulators)?;
        check_count!("input", cotangents.kinds(), inputs.len(), ProgramError);
        let branches = [driver.region(0)?, driver.region(1)?];
        let contributions = if cotangents.can_skip_transposition(outputs, branches) {
            // A zero map contributes structural zeros, but live reference state, effects, or deferred work keep the
            // transpose live even with zero output seeds. Accumulate the zero contributions below as usual.
            inputs
                .iter()
                .map(|input| Ok(MaybeZero::Zero(input.r#type().cotangent()?)))
                .collect::<Result<Vec<_>, ProgramError>>()?
        } else {
            // Linear branch inputs can occur at any boundary position. Preserve source order separately for the linear
            // branch inputs and the known residual inputs, as the branch transposition driver does. The predicate is
            // input `0` and the residuals are the known branch inputs; both are known values read from the pullback.
            // The dispatch guarantees that a `Known` input carries its pullback value, so each tracer is read directly.
            let branch_input_indices = inputs[1..]
                .iter()
                .enumerate()
                .filter_map(|(index, input)| input.is_unknown().then_some(index))
                .collect::<Vec<_>>();
            let mut known_inputs =
                inputs.iter().enumerate().filter(|(index, input)| *index == 0 || !input.is_unknown()).map(
                    |(index, input)| {
                        input.as_known().cloned().ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "`{CONDITION_OPERATION_NAME}` transpose input {index} has no known value",
                            ))
                        })
                    },
                );
            let predicate = known_inputs.next().unwrap()?;
            let residuals = known_inputs.collect::<Result<Vec<_>, _>>()?;

            // Structural-zero output cotangents stay symbolic in both transposed branches,
            // which receive no input for them.
            let branch_destination_kinds =
                branch_input_indices.iter().map(|&index| cotangents.kind(index + 1)).collect::<Vec<_>>();
            let zero_outputs = outputs.iter().map(MaybeZero::is_zero).collect::<Vec<_>>();

            // Use retained region transforms and initially omit every structural-zero input cotangent. Output seeds
            // that are structural zeros remain symbolic and own no input of either transposed branch.
            let mut transposed_branches = [
                driver.transpose_program_with_zero_seeds(
                    branches[0],
                    &branch_input_indices,
                    &[],
                    &zero_outputs,
                    &branch_destination_kinds,
                )?,
                driver.transpose_program_with_zero_seeds(
                    branches[1],
                    &branch_input_indices,
                    &[],
                    &zero_outputs,
                    &branch_destination_kinds,
                )?,
            ];

            // Inspect the derived cotangent metadata before inserting zeros. This keeps branch-transposition errors
            // ahead of variation errors, and variation errors ahead of zero-materialization failures.
            let predicate_type = predicate.r#type();
            if predicate_type.is_varying() {
                // Devices may take different branches. A cotangent crossing this boundary must already vary with the
                // predicate: summing it for an invariant input would put a collective inside diverging control flow.
                // Both-zero cotangents cross no value boundary and impose no additional variation requirement.
                let output_cotangent_count = branches[0]
                    .output_types()
                    .iter()
                    .zip(outputs)
                    .filter(|(output_type, cotangent)| !output_type.is_reference() && !cotangent.is_zero())
                    .count();
                let mut cotangent_reference_input = output_cotangent_count;
                for (position, index) in branch_input_indices.iter().map(|&index| index + 1).enumerate() {
                    // Returned values cross when either branch needs them. Reference-state cotangents retain
                    // state effects; an ordinary gradient buffer crosses only when a branch actually mutates
                    // its corresponding input.
                    let crosses_boundary = match cotangents.kind(index) {
                        CotangentDestinationKind::Return => transposed_branches
                            .iter()
                            .any(|branch| branch.cotangents()[position] == TransposedProgramCotangent::Output),
                        CotangentDestinationKind::Reference if cotangents.is_reference_input(index) => true,
                        CotangentDestinationKind::Reference => {
                            let mut mutated = false;
                            for branch in &transposed_branches {
                                let region = branch.program().entry_region_ref();
                                let analysis = region
                                    .reference_analysis_with_configuration(None, true, &[])
                                    .map_err(ProgramError::from)?;
                                let root = ReferenceRoot::RegionInput {
                                    region: region.id(),
                                    input_index: cotangent_reference_input,
                                };
                                mutated |= analysis.is_mutated(root);
                            }
                            mutated
                        }
                        CotangentDestinationKind::Ignore => false,
                    };

                    if cotangents.kind(index) == CotangentDestinationKind::Reference {
                        cotangent_reference_input += 1;
                    }

                    let input_type = inputs[index].r#type();
                    if crosses_boundary
                        && input_type.cotangent()?.validate_condition_output(predicate_type.as_ref()).is_err()
                    {
                        return Err(ProgramError::UnsupportedOperation {
                            message: format!(
                                "`{CONDITION_OPERATION_NAME}` transposition cannot produce the cotangent of input \
                                 {index} of type `{input_type}`, which does not vary over every manual axis that the \
                                 predicate `{predicate_type}` varies over; vary the input before the condition",
                            ),
                        }
                        .into());
                    }
                }
            }

            // Only after variation validation, materialize a zero inside its branch when the peer returns a value.
            // Compute both masks before changing either branch. Cotangents zero on both sides stay absent, and the
            // resulting metadata must agree before the branches can share a condition output signature.
            let materialize_zeros = [0, 1].map(|index| {
                transposed_branches[index]
                    .cotangents()
                    .iter()
                    .zip(transposed_branches[1 - index].cotangents())
                    .map(|(slot, peer)| {
                        *slot == TransposedProgramCotangent::Zero && *peer == TransposedProgramCotangent::Output
                    })
                    .collect::<Vec<_>>()
            });
            for (index, materialize) in materialize_zeros.iter().enumerate() {
                if materialize.contains(&true) {
                    transposed_branches[index] = driver.transpose_program_with_zero_seeds(
                        branches[index],
                        &branch_input_indices,
                        materialize,
                        &zero_outputs,
                        &branch_destination_kinds,
                    )?;
                }
            }

            if transposed_branches[0].cotangents() != transposed_branches[1].cotangents() {
                return Err(ProgramError::MalformedProgram(format!(
                    "`{CONDITION_OPERATION_NAME}` branch transposition returned inconsistent cotangent metadata after \
                     materializing one-sided zeros",
                ))
                .into());
            }

            // The selected transpose receives `[predicate, live non-reference output cotangents, cotangent references,
            // residuals]`. Reference outputs use the supplying input's cotangent reference; zero outputs own no input.
            let output_types = branches[0].output_types();
            let mut transposed_inputs = Vec::with_capacity(1 + output_types.len() + residuals.len());
            transposed_inputs.push(predicate);
            transposed_inputs.extend(
                outputs
                    .iter()
                    .zip(&output_types)
                    .filter(|(_, output_type)| !output_type.is_reference())
                    .filter_map(|(cotangent, _)| cotangent.as_value().cloned()),
            );
            transposed_inputs.extend(cotangents.references().iter().cloned());
            transposed_inputs.extend(residuals);

            // The shared transposed-branch handles are attached directly, so repeated binds of one transposed branch
            // intern by `Arc` identity instead of copying it again.
            let branch_cotangents = context.bind(
                O::from(ConditionOperation::new()),
                CalleeRegionDriver::new(&[
                    transposed_branches[0].program().clone(),
                    transposed_branches[1].program().clone(),
                ]),
                transposed_inputs.as_slice(),
            )?;
            let branch_cotangent_types = branch_input_indices
                .iter()
                .map(|&index| inputs[index + 1].r#type().cotangent())
                .collect::<Result<Vec<_>, _>>()?;
            let branch_cotangents =
                transposed_branches[0].assemble_cotangents(&branch_cotangent_types, branch_cotangents)?;

            // Predicate and residual inputs contribute nothing. Returned branch cotangents preserve both-zero slots
            // as structural zeros; reference destinations have already accumulated in place and need no further
            // contribution.
            let mut input_cotangents = inputs
                .iter()
                .map(|input| Ok(MaybeZero::Zero(input.r#type().cotangent()?)))
                .collect::<Result<Vec<_>, ProgramError>>()?;
            for (&index, cotangent) in branch_input_indices.iter().zip(branch_cotangents) {
                if cotangents.kind(index + 1) == CotangentDestinationKind::Return {
                    input_cotangents[index + 1] = cotangent.unwrap();
                }
            }

            input_cotangents
        };

        check_count!("input", contributions, accumulators.len(), ProgramError);
        for (accumulator, contribution) in accumulators.iter().zip(contributions) {
            accumulator.accumulate(context, contribution)?;
        }

        Ok(())
    }
}

/// Represents the ability to stage a [`ConditionOperation`] that applies one of two branch functions to structured
/// inputs according to a scalar Boolean predicate, returning the outputs of the selected branch. This is the Ryft
/// value-level analogue of JAX's [`lax.cond`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html). It
/// binds the condition in the [`domain`](Value::domain) of its values (i.e., the live trace of a staged value or
/// the operation-executing eager domain of a concrete value). Specifically, each function is traced into a branch
/// [`Program`] through a [`NestedTracingContext`] over the predicate's domain, and both branches must return values
/// of the same types. An output that both branches forward from the same input equals that input whichever branch runs,
/// so, as in JAX's `cond`, it is pruned from the branches and returned as the input itself instead of through the
/// condition.
///
/// Conditions that appear within _manual regions_ (i.e., inside the body of a
/// [`ShardMapOperation`](crate::ShardMapOperation)) receive special treatment. Each device runs that body on its local
/// shards, with explicit collectives for communication along the operation's selected mesh axes (i.e., its active
/// manual mesh axes). A predicate that varies over those axes lets devices take different branches. This function
/// aligns the condition boundary with the predicate. It varies every input over each manual axis of the predicate that
/// the input does not already vary over before tracing the branches, and varies every branch output in the same way at
/// the end of its branch (refer to [`ManualVariationAlignment::align_manual_variation_to`]). The branches therefore
/// compute on varying values, so their outputs and partial-evaluation residuals are typed as the per-device values that
/// they are, and the gradient of an input that was invariant is summed across devices by the transpose of its boundary
/// `parallel_vary`, outside of the branches. An output that both branches forward from the same invariant input keeps
/// its invariance because it is returned as that input, while an output that only one branch forwards varies, because
/// the other branch can produce a different value. With an invariant predicate, or outside of manual regions, no
/// alignment is staged. The functions cannot use values of the enclosing trace other than through `inputs` (only
/// constants can be closed over), so every differentiable value crosses the aligned boundary, while constants
/// carry no tangent and so their alignment inside a branch never transposes into a collective.
///
/// Inputs and outputs may be leaves, tuples, vectors, maps, or custom [`Parameterized`] structures. Both branch outputs
/// must have equal parameter structures, including static metadata: a runtime predicate selects value leaves, while the
/// Rust structure is determined during tracing. The input and output structures may differ, and neither needs to
/// implement `Clone`. Nested branch programs retain flat boundaries for positional replay.
///
/// The universe parameter `T` defaults to the `Capability` universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and array IR values implement it for [`ArrayIrType`].
///
/// # Example
///
/// ```rust
/// # use ryft_core::{Array, Condition, ProgramError};
/// let predicate = Array::scalar(true)?;
/// let output = predicate.condition(
///     |(first, second)| Ok(first + second),
///     |(first, second)| Ok(first * second),
///     (Array::scalar(2.0f32)?, Array::scalar(3.0f32)?),
/// )?;
/// assert_eq!(output, Array::scalar(5.0f32)?);
/// # Ok::<(), ProgramError>(())
/// ```
#[capability]
pub trait Condition<T = <Self as Capability>::Universe>: Capability + Value<Domain: Context> {
    /// Applies `true_fn` or `false_fn` to `inputs` according to this scalar Boolean predicate.
    ///
    /// # Parameters
    ///
    ///   - `true_fn`: Function traced for the `true` branch over the input family's nested tracers.
    ///   - `false_fn`: Function traced for the `false` branch over the same input family.
    ///   - `inputs`: Structured values received by both branches, including their static fields.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidArgument`] if branch output structures or static metadata differ. Also returns
    /// errors from manual-variation alignment, reconstruction, tracing, or binding (including invalid predicates and
    /// mismatched branch output types).
    fn condition<
        Input: Parameterized<Self, Family: ParameterizedFamily<Tracer<NestedTracingContext<Self::Domain>>>>,
        Output: Parameterized<
                Tracer<NestedTracingContext<Self::Domain>>,
                Family: ParameterizedFamily<Self>,
                ParameterStructure: PartialEq,
            >,
        TrueFn: FnOnce(Input::To<Tracer<NestedTracingContext<Self::Domain>>>) -> Result<Output, ProgramError>,
        FalseFn: FnOnce(Input::To<Tracer<NestedTracingContext<Self::Domain>>>) -> Result<Output, ProgramError>,
    >(
        &self,
        true_fn: TrueFn,
        false_fn: FalseFn,
        inputs: Input,
    ) -> Result<Output::To<Self>, ProgramError>;
}

impl<
    T,
    V: Value<Type: ConditionType, Domain: Context<Operation: From<ConditionOperation<V::Type>>>>
        + Capability<Universe = T>
        + ManualVariationAlignment,
> Condition<T> for V
where
    Tracer<NestedTracingContext<V::Domain>>: ManualVariationAlignment,
{
    fn condition<
        Input: Parameterized<Self, Family: ParameterizedFamily<Tracer<NestedTracingContext<Self::Domain>>>>,
        Output: Parameterized<
                Tracer<NestedTracingContext<Self::Domain>>,
                Family: ParameterizedFamily<Self>,
                ParameterStructure: PartialEq,
            >,
        TrueFn: FnOnce(Input::To<Tracer<NestedTracingContext<Self::Domain>>>) -> Result<Output, ProgramError>,
        FalseFn: FnOnce(Input::To<Tracer<NestedTracingContext<Self::Domain>>>) -> Result<Output, ProgramError>,
    >(
        &self,
        true_fn: TrueFn,
        false_fn: FalseFn,
        inputs: Input,
    ) -> Result<Output::To<Self>, ProgramError> {
        // Reject invalid predicates before either branch function is traced.
        ConditionOperation::<V::Type>::validate_predicate_type(self.r#type().as_ref())?;

        // Preserve the input structure and its static fields separately from the runtime leaves that become region
        // inputs. The original leaves also supply any outputs that both branches forward unchanged.
        let input_structure = inputs.parameter_structure();
        let inputs = inputs.into_parameters().collect::<Vec<_>>();
        let context = self.domain();

        // Vary inputs over the predicate's manual axes before tracing the branches. The transpose can then sum an
        // invariant input's cotangents outside branches that different devices may take differently.
        let predicate_sharding = self.r#type().sharding().cloned();
        let aligned_inputs = match &predicate_sharding {
            Some(sharding) => Self::align_manual_variation_to(&inputs, sharding)?,
            None => inputs.clone(),
        };
        let input_types = aligned_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();

        // Align branch outputs before they leave the nested trace: even an invariant branch result may be selected
        // differently across devices. Preserve the output structure and static fields around its aligned leaves.
        let align_outputs = |outputs: Output| {
            let structure = outputs.parameter_structure();
            let outputs = outputs.into_parameters().collect::<Vec<_>>();
            let outputs = match &predicate_sharding {
                Some(sharding) => Tracer::align_manual_variation_to(&outputs, sharding)?,
                None => outputs,
            };
            Ok::<_, ProgramError>(Output::from_parameters(structure, outputs)?)
        };

        // Trace both functions in separate nested contexts over the predicate's domain, rebuilding their structured
        // arguments from the same input structure and aligned types. Eager and staged values share this tracing path.
        let (true_structure, true_branch) = NestedTracingContext::trace(
            context.clone(),
            |inputs| {
                let inputs = Input::To::<Tracer<NestedTracingContext<Self::Domain>>>::from_parameters(
                    input_structure.clone(),
                    inputs,
                )?;
                true_fn(inputs).and_then(align_outputs)
            },
            input_types.clone(),
        )?;
        let (false_structure, false_branch) = NestedTracingContext::trace(
            context.clone(),
            |inputs| {
                let inputs =
                    Input::To::<Tracer<NestedTracingContext<Self::Domain>>>::from_parameters(input_structure, inputs)?;
                false_fn(inputs).and_then(align_outputs)
            },
            input_types,
        )?;

        // A runtime predicate selects value leaves; the Rust structure and static metadata must already agree when
        // tracing finishes so either branch's values can reconstruct the same output structure.
        if true_structure != false_structure {
            return Err(ProgramError::InvalidArgument {
                message: format!(
                    "`{CONDITION_OPERATION_NAME}` branches must return the same parameter structure and static \
                     metadata",
                ),
            });
        }

        // An output that both branches forward from the same input is returned as the unaligned input, so it keeps the
        // variation of that input, and the branches stop carrying it. Matching output structures ensure that both
        // branches have the same number of output leaves.
        let input_index = |branch: &Program<_, _, Vec<_>, Vec<_>>, output: usize| {
            branch.input_ids().iter().position(|input| *input == branch.output_ids()[output])
        };
        let forwarded_inputs = (0..true_branch.output_ids().len())
            .map(|output| match (input_index(&true_branch, output), input_index(&false_branch, output)) {
                (Some(true_input), Some(false_input)) if true_input == false_input => Some(true_input),
                _ => None,
            })
            .collect::<Vec<_>>();
        let kept_outputs = (0..true_branch.output_ids().len())
            .filter(|&output| forwarded_inputs.get(output).copied().flatten().is_none())
            .collect::<Vec<_>>();
        let branches = if kept_outputs.len() == true_branch.output_ids().len() {
            vec![true_branch, false_branch]
        } else {
            vec![true_branch.with_outputs(&kept_outputs)?, false_branch.with_outputs(&kept_outputs)?]
        };

        // Which references the branches return and mutate is only visible in their bodies, which binding does not
        // inspect when it stages the condition, so validate the complete application here.
        let mut condition_inputs = Vec::with_capacity(aligned_inputs.len() + 1);
        condition_inputs.push(self.clone());
        condition_inputs.extend(aligned_inputs);
        let condition = ConditionOperation::<Self::Type>::new();
        let condition_input_types =
            condition_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        condition.validate_application(
            &condition_input_types,
            [branches[0].entry_region_ref(), branches[1].entry_region_ref()],
        )?;

        // Bind the validated branch programs in the predicate's domain, which stages or executes the condition
        // according to that domain. With no forwarded outputs, its results already have the original leaf order.
        let outputs = context.bind(condition, branches, &condition_inputs)?;
        if forwarded_inputs.iter().all(Option::is_none) {
            return Ok(Output::To::<Self>::from_parameters(true_structure, outputs)?);
        }

        // Reinsert pruned leaves at their original positions using the unaligned inputs, preserving their variation.
        // Each remaining slot consumes one bound output, as guaranteed by the validated branch signatures.
        let mut outputs = outputs.into_iter();
        let outputs = forwarded_inputs
            .into_iter()
            .map(|input| match input {
                Some(input) => inputs[input].clone(),
                None => outputs.next().unwrap(),
            })
            .collect::<Vec<_>>();
        Ok(Output::To::<Self>::from_parameters(true_structure, outputs)?)
    }
}

/// Condition-specific branch lookup for a [`RegionDriver`].
trait ConditionRegionDriver<V: Value, O: Operation<Type = V::Type>>: RegionDriver<V, O> {
    /// Returns the `true` and `false` branches (i.e., regions `0` and `1`) attached to this condition application,
    /// checking that `input_count` counts the predicate followed by one input per `true` branch input.
    /// [`ConditionOperation`] validates the branch signatures separately.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] if either branch is missing, or [`ProgramError::InvalidInputCount`]
    /// if `input_count` does not match the `true` branch's input count plus the predicate.
    fn condition_branches(&self, input_count: usize) -> Result<[RegionRef<'_, V, O>; 2], ProgramError> {
        let branches = [self.region(0)?, self.region(1)?];
        let expected = branches[0].input_ids().len() + 1;
        if input_count != expected {
            return Err(ProgramError::InvalidInputCount { expected, actual: input_count });
        }
        Ok(branches)
    }
}

impl<V: Value, O: Operation<Type = V::Type>, D: ?Sized + RegionDriver<V, O>> ConditionRegionDriver<V, O> for D {}

impl<C: Context<Type: ConditionType>, P: BatchingPolicy<C>> BatchingContext<C, P> {
    /// Batches an application of a condition whose `predicate` is replicated, so that every batch item takes the
    /// same branch, for any [`BatchingPolicy`]. Both `branches` are batched at the batch axes of `branch_inputs` with
    /// natural output axes to discover which outputs each branch maps, and the two answers are joined into one output
    /// layout, which takes the `true` branch's axis when both branches map an output. Each branch is then instantiated
    /// at the joined axes and adapted to an ordinary region boundary, which keeps a discovery program whose axes
    /// already equal the joined ones. Finally, one condition over the batched branches is bound into the parent context
    /// over the unbatched predicate, the policy's boundary inputs, and the batched branch inputs. A staging parent
    /// therefore keeps one `condition` whose selected branch runs the whole batch, while an eager parent concretizes
    /// the predicate and interprets the selected batched branch.
    fn batch_condition_with_replicated_predicate<D: BatchingDriver<C, P>>(
        &self,
        driver: &D,
        branches: [RegionRef<'_, C::Constant, C::Operation>; 2],
        predicate: &C::Value,
        branch_inputs: &[P::Batch],
    ) -> Result<Vec<P::Batch>, BatchingError>
    where
        C::Operation: From<ConditionOperation<C::Type>>,
    {
        // Discover each branch's natural output axes using the physical batch axes of its inputs. This reveals which
        // outputs vary per item before either branch is forced into the shared condition boundary.
        let branch_input_axes = branch_inputs.iter().map(P::batch_axis).collect::<Vec<_>>();
        let [true_program, false_program] = branches.map(|branch| {
            driver.batch_program(self, branch, &branch_input_axes, ProgramBatchingOutputAxesPolicy::Natural)
        });
        let (true_program, false_program) = (true_program?, false_program?);
        check_count!("output", false_program.output_axes(), true_program.output_axes().len(), ProgramError);

        // Both branches must agree on each output's physical batch axis. Prefer the true branch's mapped axis, falling
        // back to the false branch's axis. An output remains replicated only when both branches produce it that way.
        let output_axes = true_program
            .output_axes()
            .iter()
            .zip(false_program.output_axes())
            .map(|(true_axis, false_axis)| if true_axis.is_replicated() { *false_axis } else { *true_axis })
            .collect::<Vec<_>>();

        // Align both branches to that layout, broadcasting replicated outputs or moving mapped axes as needed. Reuse
        // discovery programs that already fit, and adapt away policy bookkeeping outputs before attaching the regions.
        let mut batched_branches = Vec::with_capacity(2);
        for (branch, program) in branches.into_iter().zip([true_program, false_program]) {
            batched_branches.push(self.align_and_adapt_batched_program_outputs(
                driver,
                branch,
                &branch_input_axes,
                program,
                &output_axes,
            )?);
        }

        // Bind one condition in the parent domain so the selected branch handles the whole batch. Adapted regions keep
        // policy-owned leading inputs (e.g., a dynamic batch extent), which must precede the ordinary branch inputs.
        let mut condition_inputs = vec![predicate.clone()];
        condition_inputs.extend(P::boundary_inputs(self.axis_extent()));
        condition_inputs.extend(branch_inputs.iter().map(|input| P::value(input).clone()));
        let outputs = self.parent().bind(ConditionOperation::new(), batched_branches, &condition_inputs)?;

        // The joined axes describe these packed outputs whichever branch ran, so they also determine the batch
        // carriers returned to the enclosing batching context.
        check_count!("output", outputs, output_axes.len(), ProgramError);
        outputs.into_iter().zip(output_axes).map(|(output, axis)| P::batch(output, axis)).collect()
    }

    /// Batches an application of a condition whose `predicate` is batch-varying, for any [`BatchingPolicy`], by
    /// evaluating both branches and selecting their outputs per item. The caller must have already validated that
    /// the branches can be evaluated together (i.e., both must be pure, have no deferred work, and satisfy the type
    /// family's reference and identity constraints).
    ///
    /// Each branch is batched through `driver.batch_region` over gated inputs: an input for which `stop_gradient_fn`
    /// returns its gradient-stopped copy enters each branch as a per-item selection between itself and that copy, so
    /// primal values are unchanged but an inactive branch contributes no derivatives, and in particular no non-finite
    /// ones, while an input for which it returns `None` passes through unchanged. `merge_fn` then combines output
    /// `index` of the two branches, given the per-item selection.
    ///
    /// Each selection binds the predicate and its two candidates through `bind_select_fn`, after aligning their manual
    /// variation, because [`Context::bind`] does not align it. The candidates are aligned with each other and with the
    /// predicate, and the predicate is aligned with them, staging `parallel_vary` on the Boolean predicate, which
    /// carries no tangent, or on a candidate that varies less, whose transition then owns the adjoint. The predicate
    /// is varied once for each distinct variation that the selections require, and later selections reuse it. Values
    /// without manual variation pass through unchanged.
    fn batch_condition_with_mapped_predicate<
        D: BatchingDriver<C, P>,
        StopGradientFn: Fn(&P::Batch) -> Result<Option<P::Batch>, BatchingError>,
        BindSelectFn: Fn(&[P::Batch]) -> Result<P::Batch, BatchingError>,
        MergeFn: Fn(
            usize,
            P::Batch,
            P::Batch,
            &dyn Fn(&P::Batch, &P::Batch) -> Result<P::Batch, BatchingError>,
        ) -> Result<P::Batch, BatchingError>,
    >(
        &self,
        driver: &D,
        predicate: &P::Batch,
        branch_inputs: &[P::Batch],
        stop_gradient_fn: StopGradientFn,
        bind_select_fn: BindSelectFn,
        merge_fn: MergeFn,
    ) -> Result<Vec<P::Batch>, BatchingError>
    where
        C::Value: ManualVariationAlignment<C::Type>,
    {
        // Prepare gradient-stopped copies once for both branches. The family callback leaves structural inputs without
        // a stopped copy, so their shared state passes through unchanged.
        let stopped_inputs = branch_inputs.iter().map(stop_gradient_fn).collect::<Result<Vec<_>, _>>()?;

        // Compare each selection's required manual variation with the predicate's original variation,
        // so only candidates that introduce additional manual axes require a widened predicate.
        let predicate_type = P::value(predicate).r#type().into_owned();
        let predicate_sharding = predicate_type.sharding();
        let varies_beyond_predicate = |sharding: &Sharding| {
            let predicate_axes = predicate_sharding.map(Sharding::varying_manual_axes);
            sharding
                .varying_manual_axes()
                .iter()
                .any(|axis| !predicate_axes.is_some_and(|axes| axes.contains(axis)))
        };

        // Share each widened predicate across input gates and output merges. Interior mutability keeps `select`
        // callable through the shared `Fn` callback passed to `merge_fn` while allowing it to populate this cache.
        let varied_predicates = RefCell::new(Vec::<(BTreeSet<String>, C::Value)>::new());
        let select = |on_true: &P::Batch, on_false: &P::Batch| -> Result<P::Batch, BatchingError> {
            // Aligning candidates stages any needed `parallel_vary`, whose transpose performs the required device sums.
            // Both candidates must vary over every manual axis present in either candidate or the predicate.
            let candidates = [P::value(on_true).clone(), P::value(on_false).clone()];
            let mut candidates = C::Value::align_manual_variation(&candidates)?;
            if let Some(sharding) = predicate_sharding {
                candidates = C::Value::align_manual_variation_to(&candidates, sharding)?;
            }

            // The predicate must have the same manual variation as the aligned candidates, so the first candidate
            // supplies its target axes. Widening this Boolean predicate introduces no tangent.
            let candidate_type = candidates[0].r#type().into_owned();
            let predicate_value = match candidate_type.sharding() {
                Some(target) if varies_beyond_predicate(target) => {
                    let mut varied_predicates = varied_predicates.borrow_mut();
                    match varied_predicates.iter().find(|(axes, _)| axes == target.varying_manual_axes()) {
                        Some((_, varied)) => varied.clone(),
                        None => {
                            let varied =
                                C::Value::align_manual_variation_to(&[P::value(predicate).clone()], target)?.remove(0);
                            varied_predicates.push((target.varying_manual_axes().clone(), varied.clone()));
                            varied
                        }
                    }
                }
                _ => P::value(predicate).clone(),
            };

            // Variation alignment preserves the physical batch axes. Rebuild the carriers with those axes and let
            // the caller's select rule reconcile them and broadcast the predicate over any per-item dimensions.
            let mut candidates = candidates.into_iter();
            bind_select_fn(&[
                P::batch(predicate_value, P::batch_axis(predicate))?,
                P::batch(candidates.next().unwrap(), P::batch_axis(on_true))?,
                P::batch(candidates.next().unwrap(), P::batch_axis(on_false))?,
            ])
        };

        // Each branch sees its original primal inputs, but gradients enter only where the predicate selects it.
        // Selecting outputs alone can pass zero cotangents through inactive non-finite derivatives and produce NaNs.
        // Structural inputs without a stopped copy pass through unchanged.
        let true_inputs = branch_inputs
            .iter()
            .zip(&stopped_inputs)
            .map(|(input, stopped)| match stopped {
                Some(stopped) => select(input, stopped),
                None => Ok(input.clone()),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let true_outputs = driver.batch_region(self, 0, true_inputs)?;

        // Reverse each gate for the false branch, retaining gradients only for items whose predicate is `false`.
        let false_inputs = branch_inputs
            .iter()
            .zip(&stopped_inputs)
            .map(|(input, stopped)| match stopped {
                Some(stopped) => select(stopped, input),
                None => Ok(input.clone()),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let false_outputs = driver.batch_region(self, 1, false_inputs)?;

        // Merge corresponding outputs after both pure branches have run. The callback selects array results and
        // handles structural results, whose rules may depend on output position (e.g., reference identity validation).
        check_count!("output", false_outputs, true_outputs.len(), ProgramError);
        true_outputs
            .into_iter()
            .zip(false_outputs)
            .enumerate()
            .map(|(index, (true_output, false_output))| merge_fn(index, true_output, false_output, &select))
            .collect()
    }
}

/// Contains the known and residual programs of one condition branch, together with the values passed between them.
/// Specifically, this struct stores the [`PartitionedProgram`] produced by partial evaluation or linearization
/// alongside a list of residual edges. The list contains the partition's own edges, followed by outputs known in this
/// branch but residual in the other branch. Those outputs must still be returned by the residual condition, so their
/// known values become extra inputs to the residual branch. [`reconstruct_partitioned_condition`] creates one instance
/// per branch and combines their edges into a shared boundary. The known condition produces those edges; each residual
/// branch reads only its own branch's edges. Known branches fill unused slots for the other branch's edges with
/// placeholders.
struct PartitionedConditionBranch<V: Value, O: Operation<Type = V::Type>> {
    /// Original branch partition, including its known and residual programs and their boundary wiring.
    partition: PartitionedProgram<V, O>,

    /// Type and known-program output index of each value carried into the residual condition. Entries follow the
    /// partition's residual-edge order, then output order for values known only in this branch.
    edges: Vec<(V::Type, usize)>,
}

impl<V: Value, O: Operation<Type = V::Type>> PartitionedConditionBranch<V, O> {
    /// Creates a new [`PartitionedConditionBranch`] by validating `partition` against its input knownness mask
    /// `input_known` and preparing the residual edges for the condition's output knownness mask `output_known`.
    /// An output is known to the condition only when it is known in both branches. A result known only in this branch
    /// must be carried through an extra residual edge so the residual branch can return it.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the partition forwards residual inputs,
    /// selects other known inputs than `input_known`, or violates the edge layout that
    /// [`PartitionedProgram::validate_canonical_residual_edges`] validates, and [`ProgramError::InvalidOutputCount`]
    /// when it reports a different number of outputs than `output_known`.
    fn from_partition(
        partition: PartitionedProgram<V, O>,
        input_known: &[bool],
        output_known: &[bool],
    ) -> Result<Self, ProgramError> {
        // Each branch passes its known feeders through its own residual edges, which the split assembles into one
        // residual signature. Forwarded wiring would require reconciling branch-specific sources instead, so branch
        // partitions must keep their residual edges.
        if partition.has_forwarded_residual_inputs() {
            return Err(ProgramError::MalformedProgram(format!(
                "`{CONDITION_OPERATION_NAME}` branch partition forwards residual inputs, but splitting branches \
                 requires residual edges",
            )));
        }
        partition.validate_canonical_residual_edges()?;
        let mut edges = partition.residual_edges();
        let known_program_output_types = partition.known_program().output_types();
        check_count!("output", partition.outputs(), output_known.len(), ProgramError);
        partition.validate_known_input_indices(input_known)?;
        for (output, &known) in partition.outputs().iter().zip(output_known) {
            if !known && let PartialEvaluationOutput::Known(output) = output {
                edges.push((known_program_output_types[*output].clone(), *output));
            }
        }
        Ok(Self { partition, edges })
    }

    /// Builds the branch [`Program`] of the known condition that corresponds to this branch. Its inputs are the known
    /// condition inputs, and its outputs are the condition's known outputs (i.e., those that `output_known` selects)
    /// followed by the residual edges of the `true` branch and then those of the `false` branch, where `is_true_branch`
    /// says which of the two this branch is. The slots of the edges of `peer` hold placeholders staged by
    /// `build_placeholder_fn` into this branch's detached builder using its known input atoms. They are dead outputs,
    /// because the residual branch that the same predicate selects reads only this branch's edges. Returns a rejection
    /// message in the inner result when `build_placeholder_fn` cannot construct a placeholder for one of the edges of
    /// `peer`.
    fn known_branch<
        BuildPlaceholderFn: FnMut(&mut ProgramBuilder<V, O>, &V::Type, &[AtomId]) -> Result<Option<AtomId>, ProgramError>,
    >(
        &self,
        peer: &Self,
        is_true_branch: bool,
        output_known: &[bool],
        build_placeholder_fn: &mut BuildPlaceholderFn,
    ) -> Result<Result<Program<V, O, Vec<V>, Vec<V>>, String>, ProgramError> {
        // Replay the known program in a detached builder over matching known inputs. Placeholder failure may reject
        // the split, so this work must not reach the caller's context until both known branches are ready.
        let mut builder = ProgramBuilder::<V, O>::new();
        let known_program = self.partition.known_program();
        let known_inputs = known_program
            .input_types()
            .into_iter()
            .map(|input_type| builder.add_input(input_type))
            .collect::<Vec<_>>();
        let known_outputs = builder.splice_program(known_program, known_inputs.as_slice())?;

        // Publish only outputs known in both branches. Values known only in this branch must instead cross the split
        // as residual edges so the residual condition can return them alongside the peer's unknown results.
        let mut output_atoms = Vec::new();
        for (index, output) in self.partition.outputs().iter().enumerate() {
            if output_known[index] {
                match output {
                    PartialEvaluationOutput::Known(output) => output_atoms.push(known_outputs[*output]),
                    PartialEvaluationOutput::Unknown(_) => {
                        return Err(ProgramError::MalformedProgram(format!(
                            "`{CONDITION_OPERATION_NAME}` knownness split lost a known output"
                        )));
                    }
                }
            }
        }

        // Reserve slots for the peer's edges so both known branches have identical signatures. These slots are dead
        // for this predicate choice: the residual condition selects this branch again and reads only its own edges.
        // If a typed placeholder cannot be built, reject the split before binding any work in the caller's context.
        let mut placeholder_atoms = Vec::with_capacity(peer.edges.len());
        for (edge_type, _) in &peer.edges {
            let Some(placeholder) = build_placeholder_fn(&mut builder, edge_type, &known_inputs)? else {
                return Ok(Err(format!(
                    "no known input determines a placeholder of type `{edge_type}` for a residual of the other branch",
                )));
            };
            placeholder_atoms.push(placeholder);
        }

        // Append real residual values and peer placeholders in a fixed `true`-then-`false` branch order.
        // The edge list includes both original residual feeders and outputs that are known only in this branch.
        let edge_atoms = self.edges.iter().map(|(_, output)| known_outputs[*output]).collect::<Vec<_>>();
        if is_true_branch {
            output_atoms.extend(edge_atoms);
            output_atoms.extend(placeholder_atoms);
        } else {
            output_atoms.extend(placeholder_atoms);
            output_atoms.extend(edge_atoms);
        }

        // Seal and simplify the detached program, removing unused pure computations while preserving effects.
        // The inner result accepts the split; builder and simplification errors propagate through the outer result.
        let output_count = output_atoms.len();
        builder
            .build::<Vec<V>, Vec<V>>(
                output_atoms,
                vec![Placeholder; known_inputs.len()],
                vec![Placeholder; output_count],
            )?
            .into_simplified()
            .map(Ok)
    }

    /// Builds the branch [`Program`] of the residual condition that corresponds to this branch. Its inputs are the
    /// unknown branch inputs (i.e., the `branch_input_types` that `input_known` does not select), followed by the
    /// residual edges of the `true` branch and then those of the `false` branch, with the types from each branch's
    /// `edges` in that order, where `is_true_branch` says which of the two this branch is. Its outputs are the
    /// condition outputs that `output_known` leaves residual. The branch reads only its own edges, and returns
    /// an output that it folded from that output's instantiated edge.
    fn residual_branch(
        &self,
        is_true_branch: bool,
        branch_input_types: &[V::Type],
        input_known: &[bool],
        output_known: &[bool],
        edges: [&[(V::Type, usize)]; 2],
    ) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError> {
        // Declare only unknown branch inputs, but retain an entry at every original input position. This keeps the
        // partition's input indices valid while known inputs have no slot in the residual branch signature.
        let mut builder = ProgramBuilder::<V, O>::new();
        let unknown_input_atoms = branch_input_types
            .iter()
            .zip(input_known)
            .map(|(input_type, &known)| (!known).then(|| builder.add_input(input_type.clone())))
            .collect::<Vec<_>>();

        // Both residual branches accept the shared edge boundary in `true`-then-`false` order. Keep the peer's slots
        // in the signature even though this branch reads only its own edges; the peer's slots hold placeholders here.
        let [true_edge_atoms, false_edge_atoms] = edges
            .map(|edges| edges.iter().map(|(edge_type, _)| builder.add_input(edge_type.clone())).collect::<Vec<_>>());
        let own_edge_atoms = if is_true_branch { &true_edge_atoms } else { &false_edge_atoms };

        // Canonical partition wiring consumes the original edges in order; appended known outputs follow them.
        // `from_partition` validated that layout, so one cursor can serve the original program and then extra outputs.
        let mut edge_atoms = own_edge_atoms.iter().copied();

        // Rebuild the partition's residual input order from original unknown inputs and this branch's edges. Direct
        // known feeders are invalid here: all known values must cross the condition split through its residual edges.
        let mut spliced_inputs = Vec::with_capacity(self.partition.residual_inputs().len());
        for input in self.partition.residual_inputs() {
            match input {
                ResidualInputSource::UnknownInput(index) => {
                    spliced_inputs.push(unknown_input_atoms.get(*index).copied().flatten().ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "`{CONDITION_OPERATION_NAME}` knownness split saw a residual feeder for a known input",
                        ))
                    })?);
                }
                ResidualInputSource::ResidualEdge(_) => spliced_inputs.push(edge_atoms.next().unwrap()),
                ResidualInputSource::KnownInput(_) | ResidualInputSource::KnownOutput(_) => {
                    return Err(ProgramError::MalformedProgram(format!(
                        "`{CONDITION_OPERATION_NAME}` knownness split saw a forwarded residual input",
                    )));
                }
            }
        }
        let spliced_outputs = builder.splice_program(self.partition.residual_program(), &spliced_inputs)?;

        // Restore only the condition's residual outputs, in their original order. Results computed by the residual
        // program come from its replay; results known only in this branch come from the cursor's appended edge values.
        let mut output_atoms = Vec::new();
        for (output, &known) in self.partition.outputs().iter().zip(output_known) {
            if known {
                continue;
            }
            match output {
                PartialEvaluationOutput::Unknown(spliced) => output_atoms.push(spliced_outputs[*spliced]),
                PartialEvaluationOutput::Known(_) => {
                    let edge = edge_atoms.next().ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "`{CONDITION_OPERATION_NAME}` knownness split lost an instantiated output edge",
                        ))
                    })?;
                    output_atoms.push(edge);
                }
            }
        }

        // Seal the region over the compact unknown inputs and both edge sets, including the unused peer slots.
        // Retain residual effects and deferred work even when there are no residual output values.
        let input_count = unknown_input_atoms.iter().flatten().count() + true_edge_atoms.len() + false_edge_atoms.len();
        let output_count = output_atoms.len();
        builder.build::<Vec<V>, Vec<V>>(output_atoms, vec![Placeholder; input_count], vec![Placeholder; output_count])
    }
}

/// Rebuilds both halves of a conditional over one shared residual signature (i.e., a known condition bound through
/// `bind_known_fn` and a residual condition bound through `bind_residual_fn`) and returns the outputs of the original
/// condition in order. Unit-returning splits retain residual effects and deferred work without binding an empty known
/// condition.
///
/// Returns a rejection message in the inner result when the split would expose a reference, mistype a residual edge,
/// fold nothing, or require a placeholder that `build_placeholder_fn` cannot construct. Partial evaluation then keeps
/// the original condition intact, while separate-context linearization, which has no whole-condition fallback, reports
/// the message. Program errors propagate through the outer result. Both known branches are constructed before either
/// condition is bound, so a rejection never leaves partially staged work behind.
///
/// # Parameters
///
///   - `predicate_type`: Type of the condition predicate, whose varying manual axes every residual edge must vary over.
///   - `branch_input_types`: Types of the branch region inputs (i.e., the condition inputs after the predicate).
///   - `output_count`: Number of outputs of the original condition, which both partitions must report.
///   - `inputs`: Condition inputs, starting with the predicate, in the caller's value representation.
///   - `input_known`: Knownness mask over the branch inputs, which both partitions were split with.
///   - `partitions`: Partitions of the `true` and `false` branches into their known and residual programs.
///   - `build_placeholder_fn`: Stages a placeholder of the given edge type for a residual edge of the peer branch into
///     a detached known-branch builder, given that branch's known input atoms. Returning `None` rejects the split with
///     a message identifying the missing placeholder's type.
///   - `bind_known_fn`: Binds the known condition over the predicate and the known inputs into the caller's known-side
///     context and returns its outputs: the known outputs followed by both branches' residual edges.
///   - `bind_residual_fn`: Binds the residual condition over the predicate, the unknown inputs, and both branches'
///     residual edges, and returns the residual outputs.
#[allow(clippy::too_many_arguments)]
fn reconstruct_partitioned_condition<
    V: Value<Type: ConditionType>,
    O: Operation<Type = V::Type> + From<ConditionOperation<V::Type>>,
    Input: Clone,
    BuildPlaceholderFn: FnMut(&mut ProgramBuilder<V, O>, &V::Type, &[AtomId]) -> Result<Option<AtomId>, ProgramError>,
    BindKnownFn: FnMut(O, Vec<Program<V, O, Vec<V>, Vec<V>>>, &[Input]) -> Result<Vec<Input>, ProgramError>,
    BindResidualFn: FnMut(O, Vec<Program<V, O, Vec<V>, Vec<V>>>, &[Input]) -> Result<Vec<Input>, ProgramError>,
>(
    predicate_type: &V::Type,
    branch_input_types: &[V::Type],
    output_count: usize,
    inputs: &[Input],
    input_known: &[bool],
    partitions: [PartitionedProgram<V, O>; 2],
    mut build_placeholder_fn: BuildPlaceholderFn,
    mut bind_known_fn: BindKnownFn,
    mut bind_residual_fn: BindResidualFn,
) -> Result<Result<Vec<Input>, String>, ProgramError> {
    let [true_partition, false_partition] = partitions;
    let (predicate, branch_inputs) = (&inputs[0], &inputs[1..]);

    // Reject reference-typed known feeders and executable reference constants. A reference feeder has no typed zero
    // for the other branch's edge slot, and splitting branches must not expose a reference through either an edge or
    // a captured constant without accounting for its identity.
    if [&true_partition, &false_partition]
        .into_iter()
        .any(|partition| partition.has_known_reference_inputs_or_constants())
    {
        return Ok(Err("a branch would pass a known reference as a residual".to_string()));
    }

    // An output is known only when both branches folded it.
    let output_known = (0..output_count)
        .map(|index| {
            [&true_partition, &false_partition]
                .into_iter()
                .all(|partition| matches!(partition.outputs().get(index), Some(PartialEvaluationOutput::Known(_))))
        })
        .collect::<Vec<bool>>();
    let true_split = PartitionedConditionBranch::from_partition(true_partition, input_known, &output_known)?;
    let false_split = PartitionedConditionBranch::from_partition(false_partition, input_known, &output_known)?;
    let splits = [&true_split, &false_split];

    // Every residual edge is an output of the known condition. Devices whose predicates differ take different branches
    // and produce different edge values, so each edge must vary over the predicate's manual axes. An edge that does not
    // (e.g., a residual computed only from invariant inputs) cannot be retyped without changing the residual program
    // that consumes it, so the original condition is kept instead of being mistyped.
    if let Some((edge_type, _)) = splits
        .iter()
        .flat_map(|split| &split.edges)
        .find(|(edge_type, _)| edge_type.validate_condition_output(predicate_type).is_err())
    {
        return Ok(Err(format!(
            "a residual of type `{edge_type}` does not vary over every manual axis that the predicate \
             `{predicate_type}` varies over",
        )));
    }

    // An empty known side (i.e., no outputs, edges, effects, or deferred work on either branch) folds nothing,
    // and neither does a known side whose branches only forward known inputs into their edges or compute values that
    // depend on no input (e.g., a structural zero), because the residual condition could receive those inputs directly
    // and compute those values itself instead of selecting them through a known condition. Retain a value-returning
    // condition unchanged in both cases. In particular, the residual condition of separate-context linearization
    // reaches this rule again when its tangent context is a partial-evaluation context, and must not be split a second
    // time. Unit-returning branches still need their residual effects reconstructed, without binding an empty known
    // condition.
    let known_side_is_inert =
        !output_known.contains(&true) && splits.iter().all(|split| split.partition.known_program().effects().is_pure());
    let known_side_is_empty = known_side_is_inert && splits.iter().all(|split| split.edges.is_empty());
    let known_side_only_forwards = known_side_is_inert
        && splits.iter().all(|split| {
            split
                .partition
                .known_program()
                .entry_region_ref()
                .instructions()
                .iter()
                .all(|instruction| instruction.inputs().is_empty() && instruction.regions().is_empty())
        });
    if (known_side_is_empty || known_side_only_forwards) && output_count != 0 {
        return Ok(Err("its known side would fold no work".to_string()));
    }

    // Build both known branches before binding either condition, so that an unsupported placeholder can retain
    // the original condition without staging any work in the caller's context. The caller decides how to construct
    // placeholders: partial evaluation accepts only identity-free typed zeros, while differentiation can obtain runtime
    // extents from the known branch inputs or forward a known input of the identical type. A unit-returning condition
    // with no known work still needs its residual branch effects, but binding an empty pure known condition serves no
    // purpose.
    let known_output_count = output_known.iter().filter(|&&known| known).count();
    let edge_count = true_split.edges.len() + false_split.edges.len();
    let known_outputs = if known_side_is_empty {
        Vec::new()
    } else {
        let known_true = match true_split.known_branch(&false_split, true, &output_known, &mut build_placeholder_fn)? {
            Ok(branch) => branch,
            Err(rejection) => return Ok(Err(rejection)),
        };
        let known_false =
            match false_split.known_branch(&true_split, false, &output_known, &mut build_placeholder_fn)? {
                Ok(branch) => branch,
                Err(rejection) => return Ok(Err(rejection)),
            };
        let mut known_condition_inputs = vec![predicate.clone()];
        known_condition_inputs
            .extend(branch_inputs.iter().zip(input_known).filter(|(_, known)| **known).map(|(input, _)| input.clone()));
        let known_outputs = bind_known_fn(
            O::from(ConditionOperation::new()),
            vec![known_true, known_false],
            known_condition_inputs.as_slice(),
        )?;
        check_count!("output", known_outputs, known_output_count + edge_count, ProgramError);
        known_outputs
    };

    // A branch residual with effects or deferred work must survive even when the condition returns no residual values,
    // because simplification retains that work and dropping the condition would discard it. The known condition returns
    // the `true` branch's edges and then the `false` branch's edges after its known outputs, which is the order in
    // which the residual condition receives them after the unknown inputs.
    let needs_residual_condition = output_known.contains(&false)
        || splits.iter().any(|split| !split.partition.residual_program().effects().is_pure());
    let residual_outputs = if needs_residual_condition {
        let edges = [true_split.edges.as_slice(), false_split.edges.as_slice()];
        let residual_branches = [(&true_split, true), (&false_split, false)]
            .into_iter()
            .map(|(split, is_true_branch)| {
                split.residual_branch(is_true_branch, branch_input_types, input_known, &output_known, edges)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut residual_condition_inputs = vec![predicate.clone()];
        residual_condition_inputs.extend(
            branch_inputs.iter().zip(input_known).filter(|(_, known)| !**known).map(|(input, _)| input.clone()),
        );
        residual_condition_inputs.extend(known_outputs[known_output_count..].iter().cloned());
        bind_residual_fn(O::from(ConditionOperation::new()), residual_branches, residual_condition_inputs.as_slice())?
    } else {
        Vec::new()
    };
    check_count!("output", residual_outputs, output_count - known_output_count, ProgramError);

    // Reassemble the original output order from the two sides.
    let mut known_values = known_outputs.into_iter();
    let mut residual_values = residual_outputs.into_iter();
    Ok(Ok(output_known
        .iter()
        .map(|&known| if known { known_values.next() } else { residual_values.next() }.unwrap())
        .collect()))
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Arc;

    use indoc::{formatdoc, indoc};
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayBatch, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayOperation, ArrayReference,
        ArrayReferenceTransform, ArrayReferenceTransformIndex, DataType, Dimension, DimensionBounds, DimensionError,
        DimensionOperation, DimensionType, DimensionValue, DimensionVariable, LogicalMesh, MeshAxis, MeshAxisType,
        Shape, Sharding, ShardingDimension,
    };
    use crate::axes::NamedAxis;
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer, RecursiveBatchingDriver, batch};
    use crate::captures::{CapturingContext, ClosedProgram};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::reverse::tests::{run_transposed_with_destinations, transposition_statistics};
    use crate::differentiation::{
        ForwardModeDifferentiate, LinearizationTracer, ReverseModeDifferentiate, TransposedProgram, differentiate_at,
    };
    use crate::kernels::{KernelOperation, NoKernelExtension};
    use crate::macros::{check_gradient, check_operation_batching};
    use crate::operations::GuardOperation;
    use crate::operations::arithmetic::{AddOperation, DivOperation, MulOperation, NegOperation, SqrtOperation};
    use crate::operations::assertions::{AssertOperation, AssertionError};
    use crate::operations::collectives::parallel_vary::ParallelVaryOperation;
    use crate::operations::comparisons::{CompareOperation, ComparisonDirection};
    use crate::operations::composites::CompositeTransposeOperation;
    use crate::operations::constants::zero::ZeroOperation;
    use crate::operations::constants::zero_like::ZeroLikeOperation;
    use crate::operations::control_flow::scan::ScanOperation;
    use crate::operations::control_flow::tests::{
        CountingBatchingDriver, DischargeCapture, DischargeCaptureOperation, TestIrOperation, TestIrValue, array,
        dimension, resolve_captures,
    };
    use crate::operations::control_flow::r#while::WhileOperation;
    use crate::operations::debugging::PrintOperation;
    use crate::operations::differentiation::stop_gradient::StopGradientOperation;
    use crate::operations::dimensions::DimensionAddOperation;
    use crate::operations::dimensions::dimension_from_scalar::{DimensionFromScalar, DimensionFromScalarOperation};
    use crate::operations::manipulation::conversions::Cast;
    use crate::operations::references::{
        ReferenceAddUpdateOperation, ReferenceFreezeOperation, ReferenceNewOperation, ReferenceRead,
        ReferenceReadOperation, ReferenceSwapOperation, ReferenceWriteOperation,
    };
    use crate::operations::trigonometric::{Sin, SinOperation};
    use crate::parameters::{Parameter, Placeholder};
    use crate::partial::{
        PartialEvaluation, PartialEvaluationInput, PartialEvaluationOutput, PartialTracer, PartialValue,
        ReferencePlacement,
    };
    use crate::programs::{
        EffectClasses, ExternalReferenceBinding, ProgramBuilder, ReferenceAccessOperation, ReferenceAnalysisError,
        ReferenceDischargeResult, ReferenceDischargeTarget, ReferenceSource, ReferenceType, ReferenceView,
        RegionDriver, RegionRef, TypeError,
    };
    use crate::tracing::{DomainTracingContext, NestedTracingContext, Trace, Tracer, TracingContext};

    use super::*;

    /// Eager context over the array IR test universe.
    type TestEagerContext = EagerContext<TestIrValue, TestIrOperation>;

    /// Staged value of a branch traced over the array IR test universe.
    type TestTracer = Tracer<TracingContext<TestIrValue, TestIrOperation>>;

    /// Staged value of a branch traced over the array universe.
    type ArrayTracer = Tracer<TracingContext<Array, ArrayOperation<Array>>>;

    /// Builds a unit-returning branch that fails if execution reaches it, for primitive validation-order tests.
    fn failing_unit_branch() -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let predicate = builder.add_constant(Array::scalar(false).unwrap());
        builder
            .add_instruction(
                AssertOperation::new("branch executed before validation"),
                Vec::new(),
                vec![predicate],
                None,
            )
            .unwrap();
        builder.build(Vec::new(), Vec::new(), Vec::new()).unwrap()
    }

    /// Builds the canonical array IR test program whose whole-array state crosses a [`ConditionOperation`] boundary,
    /// shared by tests comparing direct reference transforms with explicit discharge. The program takes a Boolean
    /// predicate and an `f32[]` initial value, allocates one local reference from that initial value, and passes the
    /// reference into a condition whose branches access it with unequal modes. The `true` branch accumulates `1.0` and
    /// reads the reference, while the `false` branch swaps in `9.0` and yields the replaced value. Its two outputs are
    /// the condition's snapshot followed by the frozen final state, so a discharged program must thread identical state
    /// through both branches and keep both public outputs interpretable. On `[true, 4.0]` the outputs are `[5.0, 5.0]`,
    /// and on `[false, 4.0]` they are `[4.0, 9.0]`.
    fn state_condition_program() -> Program<TestIrValue, TestIrOperation, Vec<TestIrValue>, Vec<TestIrValue>> {
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());

        let mut true_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = true_builder.add_input(reference_type.clone().into());
        let update = true_builder.add_constant(TestIrValue::Array(Array::scalar(1.0f32).unwrap()));
        true_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let snapshot = true_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut false_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = false_builder.add_input(reference_type.into());
        let replacement = false_builder.add_constant(TestIrValue::Array(Array::scalar(9.0f32).unwrap()));
        let snapshot = false_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap()[0];
        let false_branch = false_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let initial = builder.add_input(ArrayIrType::Array(scalar_type));
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let snapshot = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![snapshot, frozen],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap()
    }

    /// Builds the `true` and `false` branches of a condition over two `ref<f32[]>` inputs that return the reference
    /// inputs at the positions in `true_outputs` and `false_outputs`, respectively.
    fn reference_forwarding_branches(
        true_outputs: &[usize],
        false_outputs: &[usize],
    ) -> [Program<TestIrValue, TestIrOperation, Vec<TestIrValue>, Vec<TestIrValue>>; 2] {
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        [true_outputs, false_outputs].map(|output_indices| {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let inputs = [builder.add_input(reference_type.clone()), builder.add_input(reference_type.clone())];
            let outputs = output_indices.iter().map(|&index| inputs[index]).collect::<Vec<_>>();
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                    outputs,
                    vec![Placeholder; 2],
                    vec![Placeholder; output_indices.len()],
                )
                .unwrap()
        })
    }

    /// Builds a program that applies a condition with the [`reference_forwarding_branches`] for `true_outputs` and
    /// `false_outputs` to a predicate and two `ref<f32[]>` inputs, and returns the condition outputs.
    fn reference_forwarding_condition_program(
        true_outputs: &[usize],
        false_outputs: &[usize],
    ) -> Program<TestIrValue, TestIrOperation, Vec<TestIrValue>, Vec<TestIrValue>> {
        let reference_type = ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let first = builder.add_input(reference_type.clone());
        let second = builder.add_input(reference_type);
        let branches = reference_forwarding_branches(true_outputs, false_outputs)
            .map(|branch| builder.import_program(branch))
            .to_vec();
        let outputs = builder
            .add_instruction(ConditionOperation::new(), branches, vec![predicate, first, second], None)
            .unwrap()
            .to_vec();
        builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                outputs,
                vec![Placeholder; 3],
                vec![Placeholder; true_outputs.len()],
            )
            .unwrap()
    }

    /// Builds the `true` and `false` branches of a condition over two `ref<f32[]>` inputs that return different
    /// reference inputs, so the reference that the condition output denotes depends on the predicate.
    fn branch_dependent_reference_branches()
    -> [Program<TestIrValue, TestIrOperation, Vec<TestIrValue>, Vec<TestIrValue>>; 2] {
        reference_forwarding_branches(&[0], &[1])
    }

    /// Builds a program that applies a condition with the [`branch_dependent_reference_branches`] to a predicate and
    /// two references, and reads the reference that the condition returns.
    fn branch_dependent_reference_program() -> Program<TestIrValue, TestIrOperation, Vec<TestIrValue>, Vec<TestIrValue>>
    {
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let branches = branch_dependent_reference_branches().map(|branch| builder.import_program(branch)).to_vec();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let first = builder.add_input(reference_type.clone());
        let second = builder.add_input(reference_type);
        let selected = builder
            .add_instruction(ConditionOperation::<ArrayIrType>::new(), branches, vec![predicate, first, second], None)
            .unwrap()[0];
        let value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![selected], None).unwrap()[0];
        builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![value], vec![Placeholder; 3], vec![Placeholder])
            .unwrap()
    }

    /// Builds a program that applies a condition, under the varying predicate of [`manual_mutation_types`], to one
    /// reference whose referent varies along `x` when `varying` is set, and to one varying `f32[2]` value that the
    /// condition returns. Only the `true` branch mutates the reference (refer to [`manual_mutation_branch`]).
    fn manual_mutation_program(
        varying: bool,
    ) -> Program<TestIrValue, TestIrOperation, Vec<TestIrValue>, Vec<TestIrValue>> {
        let (referent_type, predicate_type) = manual_mutation_types(varying);
        let (value_type, _) = manual_mutation_types(true);
        let reference_type = ArrayIrType::Reference(ReferenceType::new(referent_type));
        let branch = |mutate: bool| {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let reference = builder.add_input(reference_type.clone());
            let value = builder.add_input(value_type.clone().into());
            manual_mutation_branch(&mut builder, reference, mutate);
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let branches = vec![builder.import_program(branch(true)), builder.import_program(branch(false))];
        let predicate = builder.add_input(predicate_type.into());
        let reference = builder.add_input(reference_type);
        let value = builder.add_input(value_type.into());
        let output = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                branches,
                vec![predicate, reference, value],
                None,
            )
            .unwrap()[0];
        builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap()
    }

    /// Builds a single-input flat program that maps its scalar `f64` input through `operation`.
    fn scalar_branch(
        operation: ArrayOperation<Array>,
    ) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let inputs = if matches!(operation, ArrayOperation::Add(_)) { vec![input, input] } else { vec![input] };
        let output = builder.add_instruction(operation, Vec::new(), inputs, None).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds branches whose declared outputs differ and whose input refinements make them agree at extent three.
    fn refinable_dynamic_branches() -> [Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>; 2] {
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(input_type.clone());
        let true_branch = builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        builder.add_input(input_type);
        let output = builder.add_constant(Array::vector(vec![2.0f64; 3]).unwrap());
        let false_branch = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        [true_branch, false_branch]
    }

    /// Builds identity branches that declare `f64[extent]` and `f64[3]` inputs, respectively, which an `f64[3]` input
    /// refines to the same type even though the branches declare different input types.
    fn mismatched_input_branches() -> [Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>; 2] {
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        [dynamic_type, ArrayType::new_static(DataType::F64, [3])].map(|input_type| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(input_type);
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
        })
    }

    /// Builds a scalar branch that returns whether its input is greater than zero.
    fn boolean_branch() -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let zero = builder.add_constant(Array::scalar(0.0).unwrap());
        let output = builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                Vec::new(),
                vec![input, zero],
                None,
            )
            .unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a single-input branch that scales its scalar input by `factor`.
    fn scalar_scale_branch(factor: f64) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let factor = builder.add_constant(Array::scalar(factor).unwrap());
        let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, factor], None).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a single-input branch that scales a vector input by `factor`.
    fn vector_scale_branch(size: usize, factor: f64) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(size)])));
        let factor = builder.add_constant(Array::scalar(factor).unwrap());
        let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, factor], None).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a vector-input branch that returns a replicated constant vector.
    fn constant_vector_branch(values: Vec<f64>) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(values.len())])));
        let output = builder.add_constant(Array::vector(values).unwrap());
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Returns the predicate `input > 0` for a scalar `f64` input, bound in the domain of `input`.
    fn positive_predicate<V: Value<Type = ArrayType>>(input: &V) -> Result<V, ProgramError>
    where
        V::Domain: Context<Type = ArrayType, Constant = Array, Operation = ArrayOperation<Array>>,
    {
        let context = input.domain();
        let zero = context.lift(Array::scalar(0.0).unwrap())?;
        let mut predicates = context.bind(
            ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::GreaterThan)),
            Vec::new(),
            &[input.clone(), zero],
        )?;
        Ok(predicates.remove(0))
    }

    /// Applies a condition with the `true` and `false` `branches` to `input` under the [`positive_predicate`] of
    /// `input`, retaining both attached regions during replay.
    fn stage_runtime_predicate_condition<V: Value<Type = ArrayType>>(
        input: V,
        branches: [Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>; 2],
    ) -> Result<V, ProgramError>
    where
        V::Domain: Context<Type = ArrayType, Constant = Array, Operation = ArrayOperation<Array>>,
    {
        let predicate = positive_predicate(&input)?;
        let mut outputs = input.domain().bind(
            ArrayOperation::Condition(ConditionOperation::new()),
            branches.to_vec(),
            &[predicate, input],
        )?;
        Ok(outputs.remove(0))
    }

    /// Builds a scalar condition with a square root in its positive branch and the identity otherwise.
    fn square_root_or_identity_condition_program() -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut identity_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let identity_input = identity_builder.add_input(scalar_type.clone());
        let identity_branch = identity_builder
            .build::<Vec<Array>, Vec<Array>>(vec![identity_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(scalar_type);
        let zero = builder.add_constant(Array::scalar(0f64).unwrap());
        let predicate = builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                Vec::new(),
                vec![input, zero],
                None,
            )
            .unwrap()[0];
        let regions = vec![
            builder.import_program(scalar_branch(ArrayOperation::Sqrt(SqrtOperation::new()))),
            builder.import_program(identity_branch),
        ];
        let output =
            builder.add_instruction(ConditionOperation::new(), regions, vec![predicate, input], None).unwrap()[0];
        builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds an array IR branch that forwards its `dimension_type` extent input and scales its scalar `f64` input by
    /// `factor`.
    fn scale_branch(
        dimension_type: DimensionType,
        factor: f64,
    ) -> Program<TestIrValue, TestIrOperation, Vec<TestIrValue>, Vec<TestIrValue>> {
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let extent = builder.add_input(ArrayIrType::Dimension(dimension_type));
        let input = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
        let factor = builder.add_constant(array(Array::scalar(factor).unwrap()));
        let output = builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(MulOperation::new())),
                Vec::new(),
                vec![input, factor],
                None,
            )
            .unwrap()[0];
        builder.build(vec![extent, output], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap()
    }

    /// Builds an array IR condition over `[predicate, extent, input]`, where `input` is an `f64` vector whose dynamic
    /// size is `extent`. The `true` branch squares the input and the `false` branch doubles it. Returns the program
    /// together with the extent's dimension type and the input type.
    fn dynamic_extent_condition_program()
    -> (Program<TestIrValue, TestIrOperation, Vec<TestIrValue>, Vec<TestIrValue>>, DimensionType, ArrayType) {
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![extent_type.dimension().clone()]));
        let branch = |squares| {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            builder.add_input(extent_type.clone().into());
            let input = builder.add_input(input_type.clone().into());
            let operation = if squares {
                TestIrOperation::Array(ArrayOperation::Mul(MulOperation::new()))
            } else {
                TestIrOperation::Array(ArrayOperation::Add(AddOperation::new()))
            };
            let output = builder.add_instruction(operation, Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let extent = builder.add_input(extent_type.clone().into());
        let input = builder.add_input(input_type.clone().into());
        let true_region = builder.import_program(branch(true));
        let false_region = builder.import_program(branch(false));
        let output = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_region, false_region],
                vec![predicate, extent, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        (program, extent_type, input_type)
    }

    /// Builds a program whose `condition` selects between the unspecialized branches `x * x + y` and `x + y` over
    /// `[x: f64[rows], y: f64[rows]]` while the program feeds it static `f64[3]` inputs, so the condition's output is
    /// refined by its inputs rather than by re-typed regions.
    fn refined_vector_condition_program() -> Program<TestIrValue, TestIrOperation, Vec<TestIrValue>, Vec<TestIrValue>> {
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type = ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(rows)])));
        let branch = |squares| {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let input = builder.add_input(vector_type.clone());
            let other_input = builder.add_input(vector_type.clone());
            let input = if squares {
                builder
                    .add_instruction(
                        TestIrOperation::Array(ArrayOperation::Mul(MulOperation::new())),
                        Vec::new(),
                        vec![input, input],
                        None,
                    )
                    .unwrap()[0]
            } else {
                input
            };
            let output = builder
                .add_instruction(
                    TestIrOperation::Array(ArrayOperation::Add(AddOperation::new())),
                    Vec::new(),
                    vec![input, other_input],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(ArrayType::new_static(DataType::F64, [3]).into());
        let other_input = builder.add_input(ArrayType::new_static(DataType::F64, [3]).into());
        let true_region = builder.import_program(branch(true));
        let false_region = builder.import_program(branch(false));
        let output = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_region, false_region],
                vec![predicate, input, other_input],
                None,
            )
            .unwrap()[0];
        builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap()
    }

    /// Returns the `f32[2]` type replicated on a two-device manual mesh over `x`, varying along `x` when `varying` is
    /// set, together with the `bool[]` predicate type on that mesh that varies along `x`.
    fn manual_mutation_types(varying: bool) -> (ArrayType, ArrayType) {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(varying.then_some("x"));
        let referent_type = ArrayType::new_static(DataType::F32, [2]).with_sharding(sharding.unwrap()).unwrap();
        let predicate_sharding = Sharding::replicated(mesh, 0).with_varying_manual_axes(["x"]).unwrap();
        let predicate_type = ArrayType::scalar(DataType::Boolean).with_sharding(predicate_sharding).unwrap();
        (referent_type, predicate_type)
    }

    /// Returns a branch with no outputs that doubles the referent of `reference` in place when `mutate` is set and
    /// only reads it otherwise.
    fn manual_mutation_branch<V: Value<Type = ArrayIrType>, O: Operation<Type = ArrayIrType>>(
        builder: &mut ProgramBuilder<V, O>,
        reference: AtomId,
        mutate: bool,
    ) where
        O: From<ReferenceReadOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>>
            + From<ReferenceWriteOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>>
            + From<AddOperation<ArrayType>>,
    {
        let value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        if mutate {
            let operation = AddOperation::<ArrayType>::new();
            let doubled = builder.add_instruction(operation, Vec::new(), vec![value, value], None).unwrap()[0];
            builder
                .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, doubled], None)
                .unwrap();
        }
    }

    /// Builds the two identical branches of [`dynamic_zero_condition_program`], each of which maps its
    /// `dimension<extent>` input to a zero `f64[extent]` vector of that extent.
    fn dynamic_zero_condition_branches(
        extent_type: &DimensionType,
    ) -> [Program<TestIrValue, TestIrOperation, Vec<TestIrValue>, Vec<TestIrValue>>; 2] {
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![extent_type.dimension().clone()]));
        [(), ()].map(|()| {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let extent = builder.add_input(extent_type.clone().into());
            let output = builder
                .add_instruction(ZeroOperation::new(output_type.clone()), Vec::new(), vec![extent], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        })
    }

    /// Builds a condition over `[predicate, extent]` whose [`dynamic_zero_condition_branches`] both return a zero
    /// `f64[extent]` vector. Both inputs have zero tangent spaces, but the dynamic floating-point output does not.
    fn dynamic_zero_condition_program(
        extent_type: &DimensionType,
    ) -> Program<TestIrValue, TestIrOperation, Vec<TestIrValue>, Vec<TestIrValue>> {
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let extent = builder.add_input(extent_type.clone().into());
        let regions = dynamic_zero_condition_branches(extent_type)
            .each_ref()
            .map(|branch| builder.import_region(branch.entry_region_ref()))
            .to_vec();
        let output =
            builder.add_instruction(ConditionOperation::new(), regions, vec![predicate, extent], None).unwrap()[0];
        builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap()
    }

    /// Returns the two-device manual mesh over the `devices` axis.
    fn devices_manual_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap()
    }

    /// Returns the `bool[]` predicate type that varies over the `devices` axis of the [`devices_manual_mesh`], together
    /// with the `f32[]` types on that mesh that are invariant and that vary over `devices`, respectively.
    fn devices_varying_predicate_types() -> (ArrayType, ArrayType, ArrayType) {
        let mesh = devices_manual_mesh();
        let varying_sharding = Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["devices"]).unwrap();
        let predicate_type = ArrayType::scalar(DataType::Boolean).with_sharding(varying_sharding.clone()).unwrap();
        let invariant_type = ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(mesh, 0)).unwrap();
        let varying_type = ArrayType::scalar(DataType::F32).with_sharding(varying_sharding).unwrap();
        (predicate_type, invariant_type, varying_type)
    }

    /// Finalizes a closure-driven partial evaluation in `context` whose original outputs are `outputs`. The `inputs`
    /// and `outputs` tracers are dropped first because finalization requires `context` to hold the only handle on its
    /// accumulated residual state.
    fn finish_partial_evaluation<C: Context>(
        context: PartialEvaluationContext<C>,
        inputs: impl IntoIterator<Item = PartialTracer<C>>,
        outputs: Vec<PartialTracer<C>>,
    ) -> PartialEvaluation<C> {
        let output_values = outputs.iter().map(|output| output.value().unwrap().clone()).collect();
        drop(inputs);
        drop(outputs);
        context.into_evaluation(output_values).unwrap()
    }

    /// Builds the known-side program that `outer` staged while partially evaluating a program with `input_count`
    /// inputs into `evaluation`. Its outputs are the known residual feeders of `evaluation`, in residual input order,
    /// followed by the known outputs of `evaluation`, in output order.
    fn known_side_program<V: Value, O>(
        outer: &TracingContext<V, O>,
        evaluation: &PartialEvaluation<TracingContext<V, O>>,
        input_count: usize,
    ) -> Program<V, O, Vec<V>, Vec<V>>
    where
        O: Operation<Type = V::Type> + OperationProvider<V::Type, GuardOperation<V::Type>, Operation = O> + Clone,
    {
        let output_ids = evaluation
            .inputs()
            .iter()
            .filter_map(|input| match input {
                PartialEvaluationInput::Known(value) => Some(value.atom_id().unwrap()),
                PartialEvaluationInput::Unknown(_) => None,
            })
            .chain(evaluation.outputs().iter().filter_map(|output| match output {
                PartialEvaluationOutput::Known(value) => Some(value.atom_id().unwrap()),
                PartialEvaluationOutput::Unknown(_) => None,
            }))
            .collect::<Vec<_>>();
        let output_count = output_ids.len();
        outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<V>, Vec<V>>(output_ids, vec![Placeholder; input_count], vec![Placeholder; output_count])
            .unwrap()
    }

    /// Composes the [`known_side_program`] of a split partial evaluation with its residual program at `arguments`,
    /// the inputs of the original program, of which the first `known_input_count` are the known-side program's
    /// inputs, and reassembles the original program's outputs in their original order.
    fn interpret_known_side_and_residual<V: Value, O>(
        known_program: &Program<V, O, Vec<V>, Vec<V>>,
        evaluation: &PartialEvaluation<TracingContext<V, O>>,
        arguments: Vec<V>,
        known_input_count: usize,
    ) -> Result<Vec<V>, ProgramError>
    where
        O: Operation<Type = V::Type>
            + InterpretableOperation<EagerContext<V, O>>
            + OperationProvider<V::Type, GuardOperation<V::Type>, Operation = O>,
        EagerContext<V, O>: Domain<Type = V::Type, Value = V, Constant = V, Operation = O>,
    {
        let mut known_outputs = known_program.interpret(arguments[..known_input_count].to_vec())?.into_iter();
        let residual_arguments = evaluation
            .inputs()
            .iter()
            .map(|input| match input {
                PartialEvaluationInput::Unknown(index) => arguments[*index].clone(),
                PartialEvaluationInput::Known(_) => known_outputs.next().unwrap(),
            })
            .collect::<Vec<_>>();
        let residual_outputs = evaluation.program().interpret(residual_arguments)?;
        let outputs = evaluation
            .outputs()
            .iter()
            .map(|output| match output {
                PartialEvaluationOutput::Known(_) => known_outputs.next().unwrap(),
                PartialEvaluationOutput::Unknown(index) => residual_outputs[*index].clone(),
            })
            .collect();
        assert!(known_outputs.next().is_none());
        Ok(outputs)
    }

    #[test]
    fn test_condition_type_is_condition_predicate() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let invariant =
            ArrayType::scalar(DataType::Boolean).with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
        let varying = ArrayType::scalar(DataType::Boolean)
            .with_sharding(Sharding::replicated(mesh, 0).with_varying_manual_axes(["devices"]).unwrap())
            .unwrap();
        assert!(ArrayType::scalar(DataType::Boolean).is_condition_predicate());
        assert!(invariant.is_condition_predicate());
        assert!(varying.is_condition_predicate());
        assert!(ArrayIrType::Array(varying).is_condition_predicate());
        assert!(!ArrayType::scalar(DataType::F32).is_condition_predicate());
        assert!(!ArrayType::new_static(DataType::Boolean, [2]).is_condition_predicate());
        assert!(
            !ArrayIrType::Dimension(DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap()))
                .is_condition_predicate(),
        );
        assert!(!ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::Boolean))).is_condition_predicate());
    }

    #[test]
    fn test_condition_type_validate_condition_output() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let invariant =
            ArrayType::scalar(DataType::Boolean).with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
        let varying = ArrayType::scalar(DataType::Boolean)
            .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["devices"]).unwrap())
            .unwrap();

        // Outputs must vary over every manual axis that the predicate varies over.
        let varying_output = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["devices"]).unwrap())
            .unwrap();
        let invariant_output = ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(mesh, 0)).unwrap();
        assert_eq!(varying_output.validate_condition_output(&varying), Ok(()));
        assert_eq!(invariant_output.validate_condition_output(&invariant), Ok(()));
        assert_eq!(ArrayType::scalar(DataType::F32).validate_condition_output(&invariant), Ok(()));
        assert_eq!(
            invariant_output.validate_condition_output(&varying),
            Err(TypeError::invalid(
                "`condition` output `f32[][sharding={mesh<['devices'=2:manual]>, []}]` must vary over every manual \
                 axis that the predicate \
                 `bool[][sharding={mesh<['devices'=2:manual]>, [], varying_manual={'devices'}}]` varies over, because \
                 devices may take different branches; insert `parallel_vary` on the branch outputs",
            )),
        );
        assert_eq!(
            ArrayIrType::Array(varying_output).validate_condition_output(&ArrayIrType::Array(varying.clone())),
            Ok(()),
        );
        assert_eq!(
            ArrayIrType::Dimension(DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap()))
                .validate_condition_output(&ArrayIrType::Array(varying)),
            Err(TypeError::invalid(
                "`condition` output `dimension<extent ∈ [1, 8)>` cannot record manual variation, so it cannot be \
                 produced under the varying predicate \
                 `bool[][sharding={mesh<['devices'=2:manual]>, [], varying_manual={'devices'}}]`",
            )),
        );
    }

    #[test]
    fn test_condition() {
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let branch_input_type = ArrayType::scalar(DataType::F64);
        let operation = ConditionOperation::<ArrayType>::new();
        let true_branch = scalar_branch(ArrayOperation::Add(AddOperation::new()));
        let false_branch = scalar_branch(ArrayOperation::ZeroLike(ZeroLikeOperation::new()));

        // Operation identity, declared region slots, and payload-free rendering.
        assert_eq!(operation.name(), CONDITION_OPERATION_NAME);
        assert_eq!(operation.region_slots(), &[RegionSlot::computation("true"), RegionSlot::computation("false")]);
        assert_eq!(format!("{operation}"), "condition");
        assert_eq!(
            format!("{operation:?}"),
            "ConditionOperation { marker: PhantomData<fn() -> ryft_core::arrays::types::arrays::ArrayType> }",
        );
        assert_eq!(operation, operation);
        assert_eq!(operation, ConditionOperation::default());
        let identities = HashMap::from([(operation, "condition")]);
        assert_eq!(identities.get(&ConditionOperation::new()), Some(&"condition"));

        // Program rendering shows the attached branch regions at the instruction with their declared slot names.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let program_predicate = builder.add_input(predicate_type);
        let program_branch_input = builder.add_input(branch_input_type);
        let program_output = builder
            .add_instruction(
                operation,
                vec![true_region, false_region],
                vec![program_predicate, program_branch_input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![program_output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[] .
                let %2:f64[] = condition %0 %1 [
                    true={
                        lambda %0:f64[] .
                        let %1:f64[] = add %0 %0
                        in (%1)
                    },
                    false={
                        lambda %0:f64[] .
                        let %1:f64[] = zero_like %0
                        in (%1)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_condition_type_inference() {
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let branch_input_type = ArrayType::scalar(DataType::F64);
        let operation = ConditionOperation::<ArrayType>::new();
        let true_branch = scalar_branch(ArrayOperation::Add(AddOperation::new()));
        let false_branch = scalar_branch(ArrayOperation::ZeroLike(ZeroLikeOperation::new()));
        let interfaces = vec![true_branch.interface(), false_branch.interface()];

        // Type inference validates the branch interfaces, the predicate, and the input types, and returns the
        // branch output types.
        assert_eq!(
            operation
                .infer_region_input_types(&[predicate_type.clone(), branch_input_type.clone()], interfaces.as_slice()),
            Ok(vec![None, None]),
        );
        assert_eq!(
            operation.infer_region_input_types(&[], interfaces.as_slice()),
            Err(TypeError::invalid("`condition` expects at least one input but got 0")),
        );
        assert_eq!(
            operation.infer_output_types(&[predicate_type.clone(), branch_input_type.clone()], interfaces.as_slice()),
            Ok(vec![branch_input_type.clone()]),
        );
        assert_eq!(
            operation.infer_output_types(&[predicate_type.clone(), branch_input_type.clone()], &[]),
            Err(TypeError::invalid("expected 2 regions but got 0")),
        );
        assert_eq!(
            operation.infer_output_types(&[], interfaces.as_slice()),
            Err(TypeError::invalid("expected 2 inputs but got 0")),
        );
        assert_eq!(
            operation
                .infer_output_types(&[branch_input_type.clone(), branch_input_type.clone()], interfaces.as_slice()),
            Err(TypeError::invalid("`condition` predicate type must be a scalar boolean, but got `f64[]`")),
        );
        assert_eq!(
            operation.infer_output_types(
                &[ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(2)])), branch_input_type.clone()],
                interfaces.as_slice(),
            ),
            Err(TypeError::invalid("`condition` predicate type must be a scalar boolean, but got `bool[2]`")),
        );
        assert_eq!(
            operation.infer_output_types(
                &[predicate_type.clone(), ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))],
                interfaces.as_slice(),
            ),
            Err(TypeError::invalid(
                "`condition` input 1 has type `f64[2]`, which does not refine its branch input type `f64[]`",
            )),
        );

        // Branches must agree on their input signatures even if both produce the same output type.
        let mismatched_interface = RegionInterface::new(
            vec![ArrayType::new_static(DataType::F64, [2])],
            vec![branch_input_type.clone()],
            EffectClasses::NONE,
        );
        assert_eq!(
            operation.infer_output_types(
                &[predicate_type.clone(), branch_input_type.clone()],
                &[true_branch.interface(), mismatched_interface],
            ),
            Err(TypeError::invalid(
                "`condition` branch input type signature mismatch: expected [f64[]] but got [f64[2]]",
            )),
        );

        // Inference rejects branch interfaces with mismatched output signatures.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let zero = builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let boolean_output = builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                Vec::new(),
                vec![input, zero],
                None,
            )
            .unwrap()[0];
        let boolean_branch = builder
            .build::<Vec<Array>, Vec<Array>>(vec![boolean_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            operation.infer_output_types(
                &[predicate_type.clone(), branch_input_type.clone()],
                &[true_branch.interface(), boolean_branch.interface()],
            ),
            Err(TypeError::invalid(
                "`condition` branch output type signature mismatch: expected [f64[]] but got [bool[]]",
            )),
        );

        // Array IR universes infer the same contract over mixed dimension and array members, and reject a dimension
        // predicate.
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let dimension_type = ArrayIrType::Dimension(DimensionType::from(extent.clone()));
        let array_type =
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)])));
        let branch_inputs = vec![dimension_type.clone(), array_type.clone()];
        let branch_outputs = vec![array_type.clone(), dimension_type.clone()];
        let branch_interface = RegionInterface::new(branch_inputs.clone(), branch_outputs.clone(), EffectClasses::NONE);
        let operation = ConditionOperation::<ArrayIrType>::new();
        let mut input_types = vec![ArrayIrType::Array(ArrayType::scalar(DataType::Boolean))];
        input_types.extend(branch_inputs);

        assert_eq!(
            operation.infer_output_types(input_types.as_slice(), &[branch_interface.clone(), branch_interface]),
            Ok(branch_outputs),
        );
        assert_eq!(
            operation.infer_output_types(
                &[dimension_type],
                &[
                    RegionInterface::new(Vec::new(), Vec::new(), EffectClasses::NONE),
                    RegionInterface::new(Vec::new(), Vec::new(), EffectClasses::NONE),
                ],
            ),
            Err(TypeError::invalid(
                "`condition` predicate type must be a scalar boolean, but got `dimension<extent ∈ [1, 8)>`",
            )),
        );

        // Inference dispatches through boxed, closed-enum, and kernel operation wrappers exactly like through the bare
        // operation. The derive tests own this delegation; this block keeps one representative check per wrapper.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let identity_branch =
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let operation = ArrayOperation::<Array>::Condition(ConditionOperation::new());
        assert_eq!(
            operation.infer_output_types(
                &[ArrayType::scalar(DataType::Boolean), ArrayType::scalar(DataType::F64)],
                &[identity_branch.interface(), identity_branch.interface()],
            ),
            Ok(vec![ArrayType::scalar(DataType::F64)]),
        );
        let branches = refinable_dynamic_branches();
        let interfaces = branches.each_ref().map(Program::interface);
        let array_ir_branches =
            branches.map(|branch| branch.into_unprojected::<TestIrValue, TestIrOperation>().unwrap());
        let array_ir_interfaces = array_ir_branches.each_ref().map(Program::interface);
        let refined_input_types = [ArrayType::scalar(DataType::Boolean), ArrayType::new_static(DataType::F64, [3])];
        let refined_array_ir_input_types = refined_input_types.clone().map(ArrayIrType::Array);
        let mismatched_input_types = [ArrayType::scalar(DataType::Boolean), ArrayType::new_static(DataType::F64, [4])];
        let mismatched_array_ir_input_types = mismatched_input_types.clone().map(ArrayIrType::Array);
        let mismatch =
            TypeError::invalid("`condition` branch output type signature mismatch: expected [f64[4]] but got [f64[3]]");
        assert_eq!(
            Box::new(ConditionOperation::<ArrayType>::new()).infer_output_types(&refined_input_types, &interfaces),
            Ok(vec![ArrayType::new_static(DataType::F64, [3])]),
        );
        assert_eq!(
            Box::new(ConditionOperation::<ArrayType>::new()).infer_output_types(&mismatched_input_types, &interfaces),
            Err(mismatch.clone()),
        );
        assert_eq!(
            ArrayOperation::<Array>::Condition(ConditionOperation::new())
                .infer_output_types(&refined_input_types, &interfaces),
            Ok(vec![ArrayType::new_static(DataType::F64, [3])]),
        );
        assert_eq!(
            ArrayOperation::<Array>::Condition(ConditionOperation::new())
                .infer_output_types(&mismatched_input_types, &interfaces),
            Err(mismatch.clone()),
        );
        assert_eq!(
            TestIrOperation::Condition(ConditionOperation::new())
                .infer_output_types(&refined_array_ir_input_types, &array_ir_interfaces),
            Ok(vec![ArrayIrType::Array(ArrayType::new_static(DataType::F64, [3]))]),
        );
        assert_eq!(
            TestIrOperation::Condition(ConditionOperation::new())
                .infer_output_types(&mismatched_array_ir_input_types, &array_ir_interfaces),
            Err(mismatch.clone()),
        );
        assert_eq!(
            KernelOperation::<NoKernelExtension>::Portable(TestIrOperation::Condition(ConditionOperation::new()))
                .infer_output_types(&refined_array_ir_input_types, &array_ir_interfaces),
            Ok(vec![ArrayIrType::Array(ArrayType::new_static(DataType::F64, [3]))]),
        );
        assert_eq!(
            KernelOperation::<NoKernelExtension>::Portable(TestIrOperation::Condition(ConditionOperation::new()))
                .infer_output_types(&mismatched_array_ir_input_types, &array_ir_interfaces),
            Err(mismatch),
        );
    }

    #[test]
    fn test_condition_type_inference_refines_outputs() {
        let branches = refinable_dynamic_branches();
        assert_eq!(
            branches[0].to_string(),
            indoc! {"
            lambda %0:f64[extent] .
            in (%0)"},
        );
        assert_eq!(
            branches[1].to_string(),
            indoc! {"
            lambda %0:f64[extent] .
            let %1:f64[3] = const [2.0, 2.0, 2.0]
            in (%1)"},
        );
        let interfaces = branches.each_ref().map(Program::interface);

        // An `f64[3]` input refines the `true` branch's output to the `false` branch's `f64[3]`, so the refined
        // branch outputs agree. An `f64[4]` input refines it to `f64[4]`, which the `false` branch does not produce.
        assert_eq!(
            ConditionOperation::<ArrayType>::new().infer_output_types(
                &[ArrayType::scalar(DataType::Boolean), ArrayType::new_static(DataType::F64, [3])],
                &interfaces,
            ),
            Ok(vec![ArrayType::new_static(DataType::F64, [3])]),
        );
        assert_eq!(
            ConditionOperation::<ArrayType>::new().infer_output_types(
                &[ArrayType::scalar(DataType::Boolean), ArrayType::new_static(DataType::F64, [4])],
                &interfaces,
            ),
            Err(TypeError::invalid(
                "`condition` branch output type signature mismatch: expected [f64[4]] but got [f64[3]]",
            )),
        );
    }

    #[test]
    fn test_condition_type_inference_agrees_across_builder_eager_and_tracing() {
        // Builder construction, tracing, and eager binding refine the branch outputs by the actual inputs in the same
        // way: an `f64[3]` input makes the refined branch outputs agree on `f64[3]`, and an `f64[4]` input makes them
        // disagree, which every entry point rejects with the same error.
        let branches = refinable_dynamic_branches();
        let refined_type = ArrayType::new_static(DataType::F64, [3]);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let regions = branches.each_ref().map(|branch| builder.import_region(branch.entry_region_ref()));
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let input = builder.add_input(refined_type.clone());
        let outputs = builder
            .add_instruction(ConditionOperation::new(), regions.to_vec(), vec![predicate, input], None)
            .unwrap()
            .to_vec();
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], vec![Placeholder]).unwrap();
        assert_eq!(program.output_types(), vec![refined_type.clone()]);
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:bool[], %1:f64[3] .
            let %2:f64[3] = condition %0 %1 [
                true={
                    lambda %0:f64[extent] .
                    in (%0)
                },
                false={
                    lambda %0:f64[extent] .
                    let %1:f64[3] = const [2.0, 2.0, 2.0]
                    in (%1)
                },
            ]
            in (%2)"},
        );
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let predicate = context.input(ArrayType::scalar(DataType::Boolean));
        let input = context.input(refined_type.clone());
        let traced = context.bind(ConditionOperation::new(), branches.clone(), &[predicate, input]).unwrap();
        assert_eq!(traced[0].r#type().as_ref(), &refined_type);
        let input = Array::vector(vec![1.0f64; 3]).unwrap();
        assert_eq!(
            EagerContext::<Array, ArrayOperation<Array>>::new().bind(
                ConditionOperation::new(),
                branches.clone(),
                &[Array::scalar(false).unwrap(), input.clone()],
            ),
            Ok(vec![Array::vector(vec![2.0f64; 3]).unwrap()]),
        );
        assert_eq!(
            EagerContext::<Array, ArrayOperation<Array>>::new().bind(
                ConditionOperation::new(),
                branches.clone(),
                &[Array::scalar(true).unwrap(), input.clone()],
            ),
            Ok(vec![input]),
        );

        let expected_error =
            TypeError::invalid("`condition` branch output type signature mismatch: expected [f64[4]] but got [f64[3]]");
        let mismatched_type = ArrayType::new_static(DataType::F64, [4]);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let regions = branches.each_ref().map(|branch| builder.import_region(branch.entry_region_ref()));
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let input = builder.add_input(mismatched_type.clone());
        assert_eq!(
            builder
                .add_instruction(ConditionOperation::new(), regions.to_vec(), vec![predicate, input], None)
                .map(|outputs| outputs.to_vec()),
            Err(ProgramError::Type(expected_error.clone())),
        );
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let predicate = context.input(ArrayType::scalar(DataType::Boolean));
        let input = context.input(mismatched_type);
        assert_eq!(
            context.bind(ConditionOperation::new(), branches.clone(), &[predicate, input]).map(|_| ()),
            Err(ProgramError::Type(expected_error.clone())),
        );
        assert!(context.builder().borrow().instructions().is_empty());
        let input = Array::vector(vec![1.0f64; 4]).unwrap();
        assert_eq!(
            EagerContext::<Array, ArrayOperation<Array>>::new().bind(
                ConditionOperation::new(),
                branches.clone(),
                &[Array::scalar(false).unwrap(), input.clone()],
            ),
            Err(ProgramError::Type(expected_error.clone())),
        );
        assert_eq!(
            EagerContext::<Array, ArrayOperation<Array>>::new().bind(
                ConditionOperation::new(),
                branches,
                &[Array::scalar(true).unwrap(), input],
            ),
            Err(ProgramError::Type(expected_error)),
        );
    }

    #[test]
    fn test_condition_type_inference_refines_input_types() {
        // Branch inputs only need to refine the branch input types, so actual types that carry metadata the branch
        // input types leave unspecified (e.g., the normalized shardings of concrete backend array types) are accepted,
        // and the outputs do not inherit that metadata.
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let branch_input_type = ArrayType::scalar(DataType::F64);
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharded_type = branch_input_type.clone().with_sharding(Sharding::replicated(mesh, 0)).unwrap();
        let interfaces = vec![
            scalar_branch(ArrayOperation::Add(AddOperation::new())).interface(),
            scalar_branch(ArrayOperation::ZeroLike(ZeroLikeOperation::new())).interface(),
        ];
        assert_eq!(
            ConditionOperation::<ArrayType>::new()
                .infer_output_types(&[predicate_type.clone(), sharded_type.clone()], interfaces.as_slice()),
            Ok(vec![branch_input_type.clone()]),
        );
        assert_eq!(
            ConditionOperation::<ArrayType>::new()
                .infer_region_input_types(&[predicate_type, sharded_type.clone()], interfaces.as_slice()),
            Ok(vec![Some(vec![sharded_type.clone()]), Some(vec![sharded_type])]),
        );

        // A static extent within the bounds of a dynamic branch input dimension refines it, and the outputs take the
        // extent that the inputs establish for that dimension's identity. Inputs that share the identity share its
        // extent, so this holds even when another input still carries the identity dynamically.
        let predicate_type = ArrayIrType::Array(ArrayType::scalar(DataType::Boolean));
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type =
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)])));
        let static_type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [5]));
        let interface = RegionInterface::new(
            vec![vector_type.clone(), vector_type.clone()],
            vec![vector_type.clone()],
            EffectClasses::NONE,
        );
        let interfaces = vec![interface.clone(), interface];
        assert_eq!(
            ConditionOperation::<ArrayIrType>::new().infer_output_types(
                &[predicate_type.clone(), vector_type.clone(), static_type.clone()],
                interfaces.as_slice(),
            ),
            Ok(vec![static_type.clone()]),
        );
        assert_eq!(
            ConditionOperation::<ArrayIrType>::new().infer_output_types(
                &[
                    predicate_type.clone(),
                    ArrayIrType::Array(ArrayType::new_static(DataType::F32, [9])),
                    static_type.clone(),
                ],
                interfaces.as_slice(),
            ),
            Err(TypeError::invalid(
                "`condition` input 1 has type `f32[9]`, which does not refine its branch input type `f32[extent]`",
            )),
        );
        assert_eq!(
            ConditionOperation::<ArrayIrType>::new().infer_output_types(
                &[predicate_type.clone(), static_type.clone(), static_type.clone()],
                interfaces.as_slice(),
            ),
            Ok(vec![static_type.clone()]),
        );

        // A reference input refines its branch input type by its referent, and its referent's facts refine the
        // reference outputs like the value outputs.
        let ArrayIrType::Array(referent_type) = &vector_type else { unreachable!() };
        let reference_type = ArrayIrType::Reference(ReferenceType::new(referent_type.clone()));
        let refined_reference_type =
            ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [5])));
        let interface = RegionInterface::new(
            vec![reference_type.clone(), vector_type.clone()],
            vec![reference_type.clone(), vector_type.clone()],
            EffectClasses::NONE,
        );
        assert_eq!(
            ConditionOperation::<ArrayIrType>::new().infer_output_types(
                &[predicate_type.clone(), refined_reference_type.clone(), vector_type.clone()],
                &[interface.clone(), interface.clone()],
            ),
            Ok(vec![refined_reference_type.clone(), static_type.clone()]),
        );
        assert_eq!(
            ConditionOperation::<ArrayIrType>::new().infer_output_types(
                &[
                    predicate_type,
                    ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [9]))),
                    vector_type,
                ],
                &[interface.clone(), interface],
            ),
            Err(TypeError::invalid(
                "`condition` input 1 has type `ref<f32[9]>`, which does not refine its branch input type \
                 `ref<f32[extent]>`",
            )),
        );
    }

    #[test]
    fn test_condition_type_inference_refines_reference_inputs_by_their_referents() {
        // The `true` branch writes its vector input into its reference input and both branches return the reference.
        // A `ref<f32[5]>` input refines the declared `ref<f32[extent]>` branch input, and program construction,
        // tracing, and eager interpretation all type the reference output as `ref<f32[5]>`.
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(vector_type.clone()));
        let branch = |write: bool| {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let reference = builder.add_input(reference_type.clone());
            let value = builder.add_input(vector_type.clone().into());
            if write {
                builder
                    .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, value], None)
                    .unwrap();
            }
            builder.build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![reference],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
        };
        let branches = [branch(true).unwrap(), branch(false).unwrap()];
        let predicate_type = ArrayIrType::Array(ArrayType::scalar(DataType::Boolean));
        let value_type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [5]));
        let refined_reference_type =
            ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [5])));

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let regions = branches.each_ref().map(|branch| builder.import_region(branch.entry_region_ref()));
        let predicate = builder.add_input(predicate_type.clone());
        let reference = builder.add_input(refined_reference_type.clone());
        let value = builder.add_input(value_type.clone());
        let outputs = builder
            .add_instruction(ConditionOperation::new(), regions.to_vec(), vec![predicate, reference, value], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        assert_eq!(program.output_types(), vec![refined_reference_type.clone()]);
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:bool[], %1:ref<f32[5]>, %2:f32[5] .
            let %3:ref<f32[5]> = condition %0 %1 %2 [
                true={
                    lambda %0:ref<f32[extent]>, %1:f32[extent] .
                    let () = reference_write %0 %1
                    in (%0)
                },
                false={
                    lambda %0:ref<f32[extent]>, %1:f32[extent] .
                    in (%0)
                },
            ]
            in (%3)"},
        );

        let context = TracingContext::<TestIrValue, TestIrOperation>::new();
        let inputs =
            [context.input(predicate_type), context.input(refined_reference_type.clone()), context.input(value_type)];
        let traced = context.bind(ConditionOperation::new(), branches.clone(), &inputs).unwrap();
        let output_ids = traced.iter().map(|output| output.atom_id().unwrap()).collect();
        let traced_program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(output_ids, vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        assert_eq!(traced_program.output_types(), vec![refined_reference_type.clone()]);
        assert_eq!(
            traced_program.to_string(),
            indoc! {"
            lambda %0:bool[], %1:ref<f32[5]>, %2:f32[5] .
            let %3:ref<f32[5]> = condition %0 %1 %2 [
                true={
                    lambda %0:ref<f32[5]>, %1:f32[5] .
                    let () = reference_write %0 %1
                    in (%0)
                },
                false={
                    lambda %0:ref<f32[5]>, %1:f32[5] .
                    in (%0)
                },
            ]
            in (%3)"},
        );

        for predicate in [false, true] {
            let initial = Array::vector(vec![0.0f32; 5]).unwrap();
            let written = Array::vector(vec![1.0f32; 5]).unwrap();
            let reference = ArrayReference::new(initial.clone());
            let context = TestEagerContext::new();
            let outputs = context
                .bind(
                    ConditionOperation::new(),
                    branches.clone(),
                    &[
                        array(Array::scalar(predicate).unwrap()),
                        TestIrValue::Reference(reference.clone()),
                        array(written.clone()),
                    ],
                )
                .unwrap();
            assert_eq!(outputs, vec![TestIrValue::Reference(reference.clone())]);
            assert_eq!(outputs[0].r#type().as_ref(), &refined_reference_type);
            assert_eq!(reference.read(), Ok(if predicate { written } else { initial }));
        }
    }

    #[test]
    fn test_condition_type_inference_rejects_outputs_invariant_under_varying_predicates() {
        // An identity branch declares an invariant `f64[]` input and output. A varying input refines the branch input,
        // but the output keeps the declared invariant type, so under a predicate that varies over `devices` tracing
        // rejects the condition before staging it, because devices may take different branches.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let predicate_type = ArrayType::scalar(DataType::Boolean)
            .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["devices"]).unwrap())
            .unwrap();
        let declared_type = ArrayType::scalar(DataType::F64);
        let varying_type = declared_type
            .clone()
            .with_sharding(Sharding::replicated(mesh, 0).with_varying_manual_axes(["devices"]).unwrap())
            .unwrap();
        assert!(declared_type.is_refined_by(&varying_type));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(declared_type.clone());
        let branch =
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            branch.to_string(),
            indoc! {"
            lambda %0:f64[] .
            in (%0)"},
        );
        let expected = TypeError::invalid(
            "`condition` output `f64[]` must vary over every manual axis that the predicate \
             `bool[][sharding={mesh<['devices'=2:manual]>, [], varying_manual={'devices'}}]` varies over, because \
             devices may take different branches; insert `parallel_vary` on the branch outputs",
        );
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let predicate = context.input(predicate_type);
        let input = context.input(varying_type);
        assert_eq!(
            context
                .bind(ConditionOperation::new(), vec![branch.clone(), branch], &[predicate, input])
                .map(|_| ()),
            Err(ProgramError::Type(expected)),
        );
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_condition_type_inference_rejects_branch_outputs_that_differ_in_metadata() {
        // The `true` branch returns a constant that varies over `devices`, while the `false` branch returns its
        // invariant `f64[]` input. Refining the branches by a varying input does not carry its sharding metadata into
        // the `false` branch output, so the outputs differ in that metadata alone. Inference, tracing, and eager
        // binding all reject the branches, and region input inference declines to specialize them.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let declared_type = ArrayType::scalar(DataType::F64);
        let varying_type = declared_type
            .clone()
            .with_sharding(Sharding::replicated(mesh, 0).with_varying_manual_axes(["devices"]).unwrap())
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        builder.add_input(declared_type.clone());
        let output = builder.add_constant(Array::from_elements::<f64>(varying_type.clone(), &[2.0]).unwrap());
        let true_branch =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(declared_type);
        let false_branch =
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let branches = [true_branch, false_branch];
        let interfaces = branches.each_ref().map(Program::interface);
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let input_types = [predicate_type.clone(), varying_type.clone()];
        let operation = ConditionOperation::new();
        let expected = TypeError::invalid(format!(
            "`condition` branch output type signature mismatch: expected [{varying_type}] but got [f64[]]",
        ));
        assert_eq!(operation.infer_output_types(&input_types, &interfaces), Err(expected.clone()));
        assert_eq!(operation.infer_region_input_types(&input_types, &interfaces), Ok(vec![None, None]));
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let predicate = context.input(predicate_type);
        let input = context.input(varying_type.clone());
        assert_eq!(
            context.bind(operation, branches.clone(), &[predicate, input]).map(|_| ()),
            Err(ProgramError::Type(expected.clone())),
        );
        assert!(context.builder().borrow().instructions().is_empty());
        for predicate in [false, true] {
            let context = EagerContext::<Array, ArrayOperation<Array>>::new();
            let input = Array::from_elements::<f64>(varying_type.clone(), &[1.0]).unwrap();
            assert_eq!(
                context.bind(operation, branches.clone(), &[Array::scalar(predicate).unwrap(), input]),
                Err(ProgramError::Type(expected.clone())),
            );
        }
    }

    #[test]
    fn test_condition_type_inference_rejects_branch_local_output_identities() {
        // The branches broadcast a scalar to a length that they compute from a scalar input without returning it, so
        // the output type refers to an identity that only exists inside each branch. Refining the outputs by the input
        // facts cannot remove that identity, so type inference rejects the output.
        let length = DimensionVariable::new("length", DimensionBounds::positive(Some(4)).unwrap());
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let count = builder.add_input(ArrayType::scalar(DataType::U64).into());
        let value = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let length = builder
            .add_instruction(
                TestIrOperation::from(DimensionFromScalarOperation::new(length.into())),
                Vec::new(),
                vec![count],
                None,
            )
            .unwrap()[0];
        let broadcast = builder
            .add_instruction(
                TestIrOperation::from(DynamicBroadcastOperation::new(Vec::new())),
                Vec::new(),
                vec![value, length],
                None,
            )
            .unwrap()[0];
        let branch = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![broadcast], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            branch.to_string(),
            indoc! {"
                lambda %0:u64[], %1:f32[] .
                let %2:dimension<length ∈ [1, 4)> = dimension_from_scalar [bounds=[1, 4)] %0
                    %3:f32[length] = broadcast [output_axes=[]] %1 %2
                in (%3)"},
        );
        let input_types = [
            ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)),
            ArrayIrType::Array(ArrayType::scalar(DataType::U64)),
            ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
        ];
        let expected = TypeError::invalid(
            "`condition` output 0 has type `f32[length]`, which refers to the identity `length` that no input carries \
             and no output defines",
        );
        assert_eq!(
            ConditionOperation::<ArrayIrType>::new()
                .infer_output_types(&input_types, &[branch.interface(), branch.interface()]),
            Err(expected.clone()),
        );
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let branches = vec![builder.import_program(branch.clone()), builder.import_program(branch)];
        let inputs = input_types.iter().map(|input_type| builder.add_input(input_type.clone())).collect::<Vec<_>>();
        assert_eq!(
            builder
                .add_instruction(TestIrOperation::Condition(ConditionOperation::new()), branches, inputs, None)
                .map(|_| ()),
            Err(ProgramError::Type(expected)),
        );
    }

    #[test]
    fn test_condition_validate_region_interfaces() {
        let branches = mismatched_input_branches();
        let interfaces = branches.each_ref().map(Program::interface);
        let expected = Err(ProgramError::Type(TypeError::invalid(
            "`condition` branch input type signature mismatch: expected [f64[extent]] but got [f64[3]]",
        )));
        assert_eq!(ConditionOperation::<ArrayType>::new().validate_region_interfaces(&interfaces), expected);
        assert_eq!(
            ConditionOperation::<ArrayType>::new().validate_region_interfaces(&[interfaces[0].clone()]),
            Err(ProgramError::MalformedProgram(
                "operation `condition` declares 2 region slots but 1 regions were attached".to_string(),
            )),
        );
        let same_interfaces = [interfaces[0].clone(), interfaces[0].clone()];
        assert_eq!(ConditionOperation::<ArrayType>::new().validate_region_interfaces(&same_interfaces), Ok(()));

        // Every entry point rejects branches that declare different input types, even when refinement by the actual
        // inputs would make them agree (an `f64[3]` input refines `f64[extent]` to `f64[3]`). Builder construction,
        // type inference, and eager binding all reject such branches, and so does staging, which validates the
        // original branch interfaces before it can specialize the dynamic branch to the actual input types and make
        // both declared signatures equal.
        let operation = ConditionOperation::new();
        let expected_error = TypeError::invalid(
            "`condition` branch input type signature mismatch: expected [f64[extent]] but got [f64[3]]",
        );
        for extent in [3, 4] {
            let input_types = [ArrayType::scalar(DataType::Boolean), ArrayType::new_static(DataType::F64, [extent])];
            assert_eq!(operation.infer_output_types(&input_types, &interfaces), Err(expected_error.clone()));

            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let regions = branches.each_ref().map(|branch| builder.import_region(branch.entry_region_ref()));
            let predicate = builder.add_input(input_types[0].clone());
            let input = builder.add_input(input_types[1].clone());
            assert_eq!(
                builder.add_instruction(operation, regions.to_vec(), vec![predicate, input], None).map(|_| ()),
                Err(ProgramError::Type(expected_error.clone())),
            );

            let context = EagerContext::<Array, ArrayOperation<Array>>::new();
            let input = Array::vector(vec![1.0f64; extent]).unwrap();
            assert_eq!(
                context.bind(operation, branches.clone(), &[Array::scalar(true).unwrap(), input]),
                Err(ProgramError::Type(expected_error.clone())),
            );

            let context = TracingContext::<Array, ArrayOperation<Array>>::new();
            let predicate = context.input(input_types[0].clone());
            let input = context.input(input_types[1].clone());
            assert_eq!(
                context.bind(operation, branches.clone(), &[predicate, input]).map(|_| ()),
                Err(ProgramError::Type(expected_error.clone())),
            );
            assert!(context.builder().borrow().instructions().is_empty());
        }

        // Validation dispatches through boxed, closed-enum, and kernel operation wrappers exactly like through the bare
        // operation. The derive tests own this delegation; this block keeps one representative check per wrapper.
        assert_eq!(Box::new(ConditionOperation::<ArrayType>::new()).validate_region_interfaces(&interfaces), expected);
        assert_eq!(
            ArrayOperation::<Array>::Condition(ConditionOperation::new()).validate_region_interfaces(&interfaces),
            expected,
        );
        let array_ir_branches =
            branches.map(|branch| branch.into_unprojected::<TestIrValue, TestIrOperation>().unwrap());
        let interfaces = array_ir_branches.each_ref().map(Program::interface);
        assert_eq!(
            TestIrOperation::Condition(ConditionOperation::new()).validate_region_interfaces(&interfaces),
            expected,
        );
        assert_eq!(
            KernelOperation::<NoKernelExtension>::Portable(TestIrOperation::Condition(ConditionOperation::new()))
                .validate_region_interfaces(&interfaces),
            expected,
        );
    }

    #[test]
    fn test_condition_region_data_flow() {
        let operation = ConditionOperation::<ArrayType>::new();
        assert!(matches!(operation.region_data_flow(), RegionDataFlow::Provenance));
        assert_eq!(operation.input_region_provenance(0, 1), InputRegionProvenance::Input { index: 2 });
        assert_eq!(operation.input_region_provenance(1, 1), InputRegionProvenance::Input { index: 2 });
        assert_eq!(operation.input_region_provenance(0, 2), InputRegionProvenance::Input { index: 3 });
        assert_eq!(operation.input_region_provenance(1, 2), InputRegionProvenance::Input { index: 3 });
        assert_eq!(operation.input_region_provenance(2, 0), InputRegionProvenance::None);
        assert_eq!(
            operation.output_region_provenance(0),
            vec![
                OutputRegionProvenance { region_index: 0, output_index: 0 },
                OutputRegionProvenance { region_index: 1, output_index: 0 },
            ],
        );
        assert_eq!(
            operation.output_region_provenance(1),
            vec![
                OutputRegionProvenance { region_index: 0, output_index: 1 },
                OutputRegionProvenance { region_index: 1, output_index: 1 },
            ],
        );
    }

    #[test]
    fn test_condition_boundary_pruning() {
        // The true branch maps `(a, b, c)` to `(sin(a), sin(c))` and the false branch maps it to `(sin(b), sin(c))`.
        // Using only the first output keeps the inputs that either branch needs for it, `a` and `b`, together with the
        // predicate, and drops `c` and the second output from both branches.
        let scalar = ArrayType::scalar(DataType::F64);
        let branch = |first: usize| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let inputs = (0..3).map(|_| builder.add_input(scalar.clone())).collect::<Vec<_>>();
            let outputs = [inputs[first], inputs[2]]
                .map(|input| builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0]);
            builder
                .build::<Vec<Array>, Vec<Array>>(outputs.to_vec(), vec![Placeholder; 3], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let inputs = (0..3).map(|_| builder.add_input(scalar.clone())).collect::<Vec<_>>();
        let true_branch = builder.import_program(branch(0));
        let false_branch = builder.import_program(branch(1));
        let outputs = builder
            .add_instruction(
                ConditionOperation::<ArrayType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, inputs[0], inputs[1], inputs[2]],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![outputs[0]], vec![Placeholder; 4], vec![Placeholder])
            .unwrap();
        let pruned = program.clone().into_pruned().unwrap();
        assert_eq!(
            pruned.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[], %2:f64[], %3:f64[] .
                let %4:f64[] = condition %0 %1 %2 [
                    true={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = sin %0
                        in (%2)
                    },
                    false={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = sin %1
                        in (%2)
                    },
                ]
                in (%4)"},
        );
        let inputs = vec![
            Array::scalar(false).unwrap(),
            Array::scalar(0.5f64).unwrap(),
            Array::scalar(1.5f64).unwrap(),
            Array::scalar(2.5f64).unwrap(),
        ];
        assert_eq!(pruned.interpret(inputs.clone()), Ok(vec![Array::scalar(1.5f64.sin()).unwrap()]));
        assert_eq!(program.interpret(inputs), Ok(vec![Array::scalar(1.5f64.sin()).unwrap()]));
    }

    #[test]
    fn test_condition_boundary_pruning_keeps_inputs_that_refine_kept_outputs() {
        // Neither branch reads its second or third input. When the second input is the static `f64[3]`, it is the only
        // input that fixes `rows = 3`, which refines the output to `f64[3]`, so dropping it would change the type of
        // the kept output. The pruning that the condition proposes is then rejected as a whole and the instruction
        // keeps its boundary. When every input carries the same symbolic `rows`, the enclosing boundary already
        // validates their shared extent, so both unread inputs are pruned. When the read input is the static `f64[3]`
        // and an unread input carries the symbolic `rows`, only the branch boundary checks that its runtime extent is
        // 3, so the unread inputs remain as validation peers of the read one.
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type = ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(rows)])));
        let static_type = ArrayIrType::Array(ArrayType::new_static(DataType::F64, [3]));
        let branch = {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let first = builder.add_input(vector_type.clone());
            builder.add_input(vector_type.clone());
            builder.add_input(vector_type.clone());
            let output = builder
                .add_instruction(
                    TestIrOperation::Array(ArrayOperation::from(SinOperation::new())),
                    Vec::new(),
                    vec![first],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };
        let program = |first_type: ArrayIrType, second_type: ArrayIrType| {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
            let first = builder.add_input(first_type);
            let second = builder.add_input(second_type);
            let third = builder.add_input(vector_type.clone());
            let true_branch = builder.import_program(branch.clone());
            let false_branch = builder.import_program(branch.clone());
            let output = builder
                .add_instruction(
                    TestIrOperation::Condition(ConditionOperation::new()),
                    vec![true_branch, false_branch],
                    vec![predicate, first, second, third],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 4], vec![Placeholder])
                .unwrap()
        };

        let refined = program(vector_type.clone(), static_type.clone());
        assert_eq!(refined.output_types(), vec![static_type.clone()]);
        let pruned = refined.clone().into_pruned().unwrap();
        assert_eq!(
            pruned.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[rows], %2:f64[3], %3:f64[rows] .
                let %4:f64[3] = condition %0 %1 %2 %3 [
                    true={
                        lambda %0:f64[rows], %1:f64[rows], %2:f64[rows] .
                        let %3:f64[rows] = sin %0
                        in (%3)
                    },
                    false={
                        lambda %0:f64[rows], %1:f64[rows], %2:f64[rows] .
                        let %3:f64[rows] = sin %0
                        in (%3)
                    },
                ]
                in (%4)"},
        );
        assert_eq!(pruned.output_types(), vec![static_type.clone()]);

        let unrefined = program(vector_type.clone(), vector_type.clone());
        let pruned = unrefined.clone().into_pruned().unwrap();
        assert_eq!(pruned.output_types(), vec![vector_type.clone()]);
        assert_eq!(
            pruned.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[rows], %2:f64[rows], %3:f64[rows] .
                let %4:f64[rows] = condition %0 %1 [
                    true={
                        lambda %0:f64[rows] .
                        let %1:f64[rows] = sin %0
                        in (%1)
                    },
                    false={
                        lambda %0:f64[rows] .
                        let %1:f64[rows] = sin %0
                        in (%1)
                    },
                ]
                in (%4)"},
        );
        let matching = array(Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap());
        let inputs = vec![array(Array::scalar(true).unwrap()), matching.clone(), matching.clone(), matching];
        assert_eq!(pruned.interpret(inputs.clone()), unrefined.interpret(inputs.clone()));
        let mut mismatching = inputs;
        mismatching[2] = array(Array::vector(vec![4.0f64, 5.0]).unwrap());
        let original_error = unrefined.interpret(mismatching.clone());
        assert!(matches!(
            &original_error,
            Err(ProgramError::Type(error))
                if error.downcast_custom::<DimensionError>() == Some(&DimensionError::InputDimensionMismatch {
                    dimension: "rows".to_string(),
                    expected: 3,
                    actual: 2,
                }),
        ));
        assert_eq!(pruned.interpret(mismatching), original_error);

        let statically_read = program(static_type.clone(), vector_type.clone());
        let pruned = statically_read.clone().into_pruned().unwrap();
        assert_eq!(
            pruned.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[3], %2:f64[rows], %3:f64[rows] .
                let %4:f64[3] = condition %0 %1 %2 %3 [
                    true={
                        lambda %0:f64[rows], %1:f64[rows], %2:f64[rows] .
                        let %3:f64[rows] = sin %0
                        in (%3)
                    },
                    false={
                        lambda %0:f64[rows], %1:f64[rows], %2:f64[rows] .
                        let %3:f64[rows] = sin %0
                        in (%3)
                    },
                ]
                in (%4)"},
        );
        let exact = array(Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap());
        let short = array(Array::vector(vec![4.0f64, 5.0]).unwrap());
        let inputs = vec![array(Array::scalar(true).unwrap()), exact.clone(), short.clone(), short];
        let original_error = statically_read.interpret(inputs.clone());
        assert!(matches!(
            &original_error,
            Err(ProgramError::Type(error))
                if error.downcast_custom::<DimensionError>() == Some(&DimensionError::InputDimensionMismatch {
                    dimension: "rows".to_string(),
                    expected: 3,
                    actual: 2,
                }),
        ));
        assert_eq!(pruned.interpret(inputs), original_error);
    }

    #[test]
    fn test_condition_reference_discharge() {
        let scalar_type = ArrayType::scalar(DataType::F32);
        let boolean_type = ArrayType::scalar(DataType::Boolean);
        let reference_type = ReferenceType::new(scalar_type.clone());

        // Read-only pruning. Both branches merely read the root, so the shared state boundary enters both branches but
        // publishes nothing back: the rebuilt branches gain no appended output, the discharged condition keeps the
        // source output boundary exactly, and the entry root receives no hidden final-state output.
        let mut true_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = true_builder.add_input(reference_type.clone().into());
        let snapshot = true_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let doubled = true_builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![snapshot, snapshot], None)
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut false_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = false_builder.add_input(reference_type.clone().into());
        let snapshot = false_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let false_branch = false_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_branch = builder.import_program(true_branch);
        let false_branch = builder.import_program(false_branch);
        let predicate = builder.add_input(boolean_type.clone().into());
        let reference = builder.add_input(reference_type.clone().into());
        let value = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[] .
                let %2:f32[] = condition %0 %1 [
                    true={
                        lambda %0:f32[] .
                        let %1:f32[] = add %0 %0
                        in (%1)
                    },
                    false={
                        lambda %0:f32[] .
                        in (%0)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Input { index: 1 }, None)],
        );

        // A closure that only reads an external allocation needs the state to enter both branches, but the allocation's
        // value never changes, so no branch gains a final-state result and the parent condition keeps exactly its
        // public outputs instead of carrying a dead state output. Either predicate returns the entering state.
        let make_branch = || {
            let mut branch_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let reference = branch_builder.add_input(reference_type.clone().into());
            let snapshot = branch_builder
                .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
                .unwrap()[0];
            branch_builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_branch = builder.import_region(make_branch().entry_region_ref());
        let false_branch = builder.import_region(make_branch().entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let reference = builder.add_input(reference_type.clone().into());
        let snapshot = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[] .
                let %2:f32[] = condition %0 %1 [
                    true={
                        lambda %0:f32[] .
                        in (%0)
                    },
                    false={
                        lambda %0:f32[] .
                        in (%0)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Input { index: 1 }, None)],
        );
        assert_eq!(
            discharged.program().interpret(vec![
                TestIrValue::Array(Array::scalar(true).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(4.0).unwrap()),
            ]),
            Ok(vec![TestIrValue::Array(Array::scalar::<f32>(4.0).unwrap())]),
        );
        assert_eq!(
            discharged.program().interpret(vec![
                TestIrValue::Array(Array::scalar(false).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(4.0).unwrap()),
            ]),
            Ok(vec![TestIrValue::Array(Array::scalar::<f32>(4.0).unwrap())]),
        );

        // A root one branch mutates is published instead. Exactly one output is appended, and both branches receive
        // the identical widened boundary even though only the true branch writes: the reading branch republishes the
        // state it received, which is what keeps the two branch interfaces agreeing after the rewrite.
        let mut true_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = true_builder.add_input(reference_type.clone().into());
        let replacement = true_builder.add_constant(TestIrValue::Array(Array::scalar(7.0f32).unwrap()));
        let snapshot = true_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut false_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = false_builder.add_input(reference_type.clone().into());
        let snapshot = false_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let false_branch = false_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_branch = builder.import_program(true_branch);
        let false_branch = builder.import_program(false_branch);
        let predicate = builder.add_input(boolean_type.clone().into());
        let reference = builder.add_input(reference_type.clone().into());
        let value = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[] .
                let %2:f32[], %3:f32[] = condition %0 %1 [
                    true={
                        lambda %0:f32[] .
                        let %1:f32[] = const 7.0
                        in (%0, %1)
                    },
                    false={
                        lambda %0:f32[] .
                        in (%0, %0)
                    },
                ]
                in (%2, %3)"},
        );
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Input { index: 1 }, Some(1))],
        );

        // A branch that writes a replacement input and then reads it, beside a branch that only reads, also receives
        // the identical widened boundary.
        let mut true_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = true_builder.add_input(reference_type.clone().into());
        let replacement = true_builder.add_input(ArrayType::scalar(DataType::F32).into());
        true_builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap();
        let snapshot = true_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut false_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = false_builder.add_input(reference_type.clone().into());
        false_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let snapshot = false_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let false_branch = false_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let reference = builder.add_input(reference_type.clone().into());
        let replacement = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let snapshot = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference, replacement],
                None,
            )
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![snapshot], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        // Both branches receive the entering state and return their own final state after the source output, so the
        // writing branch returns its replacement while the reading branch returns the state unchanged.
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[], %2:f32[] .
                let %3:f32[], %4:f32[] = condition %0 %1 %2 [
                    true={
                        lambda %0:f32[], %1:f32[] .
                        in (%1, %1)
                    },
                    false={
                        lambda %0:f32[], %1:f32[] .
                        in (%0, %0)
                    },
                ]
                in (%3, %4)"},
        );
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Input { index: 1 }, Some(1))],
        );

        // The true branch writes and then reads, so both the public snapshot and final state are the replacement.
        assert_eq!(
            discharged.program().interpret(vec![
                TestIrValue::Array(Array::scalar(true).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(10.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(7.0).unwrap()),
            ]),
            Ok(vec![
                TestIrValue::Array(Array::scalar::<f32>(7.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(7.0).unwrap()),
            ]),
        );

        // The false branch only reads, so the entering state is both the snapshot and the final state.
        assert_eq!(
            discharged.program().interpret(vec![
                TestIrValue::Array(Array::scalar(false).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(10.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(7.0).unwrap()),
            ]),
            Ok(vec![
                TestIrValue::Array(Array::scalar::<f32>(10.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(10.0).unwrap()),
            ]),
        );

        // Pruning is decided per root rather than for the operation as a whole. Two roots enter as instruction inputs
        // and both branches reach both of them, but only the second is ever written, so only the second gains an
        // appended output while the first keeps crossing the boundary as a read-only state.
        let mut true_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let first = true_builder.add_input(reference_type.clone().into());
        let second = true_builder.add_input(reference_type.clone().into());
        let update = true_builder.add_constant(TestIrValue::Array(Array::scalar(1.0f32).unwrap()));
        let snapshot =
            true_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![first], None).unwrap()[0];
        true_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![second, update], None)
            .unwrap();
        let true_branch = true_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut false_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let first = false_builder.add_input(reference_type.clone().into());
        let second = false_builder.add_input(reference_type.clone().into());
        false_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![first], None).unwrap();
        let snapshot = false_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![second], None)
            .unwrap()[0];
        let false_branch = false_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_branch = builder.import_program(true_branch);
        let false_branch = builder.import_program(false_branch);
        let predicate = builder.add_input(boolean_type.into());
        let first = builder.add_input(reference_type.clone().into());
        let second = builder.add_input(reference_type.into());
        let value = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, first, second],
                None,
            )
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![value], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[], %2:f32[] .
                let %3:f32[], %4:f32[] = condition %0 %1 %2 [
                    true={
                        lambda %0:f32[], %1:f32[] .
                        let %2:f32[] = const 1.0
                            %3:f32[] = add %1 %2
                        in (%0, %3)
                    },
                    false={
                        lambda %0:f32[], %1:f32[] .
                        in (%1, %1)
                    },
                ]
                in (%3, %4)"},
        );
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[
                ExternalReferenceBinding::new(ReferenceSource::Input { index: 1 }, None),
                ExternalReferenceBinding::new(ReferenceSource::Input { index: 2 }, Some(1)),
            ],
        );

        // The true branch reports the first root's state and accumulates into the second; the false branch reports the
        // second root's state and writes nothing, so its appended final state is the value that entered.
        let inputs = vec![
            TestIrValue::Array(Array::scalar(true).unwrap()),
            TestIrValue::Array(Array::scalar(10.0f32).unwrap()),
            TestIrValue::Array(Array::scalar(20.0f32).unwrap()),
        ];
        assert_eq!(
            discharged.program().interpret(inputs),
            Ok(vec![
                TestIrValue::Array(Array::scalar(10.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(21.0f32).unwrap()),
            ]),
        );
        let inputs = vec![
            TestIrValue::Array(Array::scalar(false).unwrap()),
            TestIrValue::Array(Array::scalar(10.0f32).unwrap()),
            TestIrValue::Array(Array::scalar(20.0f32).unwrap()),
        ];
        assert_eq!(
            discharged.program().interpret(inputs),
            Ok(vec![
                TestIrValue::Array(Array::scalar(20.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(20.0f32).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_condition_reference_discharge_threads_preserved_allocations_through_branches() {
        // A condition's shared state boundary carries both kinds of allocation: the selected one crosses as immutable
        // state and is widened with a published successor, while the preserved one crosses as the reference it already
        // is, at its own declared input position, and is read inside each branch exactly as the source read it.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let branch = |accumulates: bool| {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let pipeline = builder.add_input(reference_type.clone().into());
            let kernel = builder.add_input(reference_type.clone().into());
            let observed =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![kernel], None).unwrap()[0];
            if accumulates {
                builder
                    .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![pipeline, observed], None)
                    .unwrap();
            }
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![observed], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let true_branch = branch(true);
        let false_branch = branch(false);

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let pipeline_initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let kernel_initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let pipeline = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![pipeline_initial], None)
            .unwrap()[0];
        let kernel = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![kernel_initial], None)
            .unwrap()[0];
        let observed = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, pipeline, kernel],
                None,
            )
            .unwrap()[0];
        let pipeline_final =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![pipeline], None).unwrap()[0];
        let kernel_final =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![kernel], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![observed, pipeline_final, kernel_final],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();

        let targets = source.reference_discharge_targets(0).unwrap();
        let discharged = source.clone().partially_discharge_references(0, &targets[..1]).unwrap();
        assert_eq!(discharged.output_count(), 3);
        assert_eq!(discharged.external_reference_bindings(), &[]);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[], %2:f32[] .
                let %3:ref<f32[]> = reference_new %2
                    %4:f32[], %5:f32[] = condition %0 %1 %3 [
                        true={
                            lambda %0:f32[], %1:ref<f32[]> .
                            let %2:f32[] = reference_read %1
                                %3:f32[] = add %0 %2
                            in (%2, %3)
                        },
                        false={
                            lambda %0:f32[], %1:ref<f32[]> .
                            let %2:f32[] = reference_read %1
                            in (%2, %0)
                        },
                    ]
                    %6:f32[] = reference_freeze %3
                in (%4, %5, %6)"},
        );

        // Eager reference semantics stay the oracle on both sides of the rewrite.
        for (predicate, expected) in [(true, 13.0f32), (false, 10.0)] {
            let inputs = vec![
                TestIrValue::Array(Array::scalar(predicate).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(10.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(3.0).unwrap()),
            ];
            let outputs = vec![
                TestIrValue::Array(Array::scalar::<f32>(3.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(expected).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(3.0).unwrap()),
            ];
            assert_eq!(source.clone().interpret(inputs.clone()), Ok(outputs.clone()));
            assert_eq!(discharged.program().interpret(inputs), Ok(outputs));
        }
    }

    #[test]
    fn test_condition_reference_discharge_threads_a_preserved_allocation_through_nested_structured_boundaries() {
        // A rebuilt region is discharged against its own isolated environment, so a preserved reference crossing two
        // boundaries is bound as a preserved reference of the outer fork and then threaded again into the inner one.
        // The reference therefore reaches the innermost access as the caller's own, and the discharged reference beside
        // it is widened independently at each level.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let inner = |accumulates: bool| {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let pipeline = builder.add_input(reference_type.clone().into());
            let kernel = builder.add_input(reference_type.clone().into());
            let observed =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![kernel], None).unwrap()[0];
            if accumulates {
                builder
                    .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![pipeline, observed], None)
                    .unwrap();
            }
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![observed], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let inner_true = inner(true);
        let inner_false = inner(false);

        let mut outer_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let inner_true = outer_builder.import_region(inner_true.entry_region_ref());
        let inner_false = outer_builder.import_region(inner_false.entry_region_ref());
        let predicate = outer_builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let pipeline = outer_builder.add_input(reference_type.clone().into());
        let kernel = outer_builder.add_input(reference_type.clone().into());
        let observed = outer_builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![inner_true, inner_false],
                vec![predicate, pipeline, kernel],
                None,
            )
            .unwrap()[0];
        let outer = outer_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![observed], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let outer_true = builder.import_region(outer.entry_region_ref());
        let outer_false = builder.import_region(outer.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let pipeline_initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let kernel_initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let pipeline = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![pipeline_initial], None)
            .unwrap()[0];
        let kernel = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![kernel_initial], None)
            .unwrap()[0];
        let observed = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![outer_true, outer_false],
                vec![predicate, predicate, pipeline, kernel],
                None,
            )
            .unwrap()[0];
        let pipeline_final =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![pipeline], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![observed, pipeline_final],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();

        let targets = source.reference_discharge_targets(0).unwrap();
        let discharged = source.clone().partially_discharge_references(0, &targets[..1]).unwrap();
        assert_eq!(discharged.output_count(), 2);
        assert_eq!(discharged.external_reference_bindings(), &[]);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[], %2:f32[] .
                let %3:ref<f32[]> = reference_new %2
                    %4:f32[], %5:f32[] = condition %0 %0 %1 %3 [
                        true={
                            lambda %0:bool[], %1:f32[], %2:ref<f32[]> .
                            let %3:f32[], %4:f32[] = condition %0 %1 %2 [
                                true={
                                    lambda %0:f32[], %1:ref<f32[]> .
                                    let %2:f32[] = reference_read %1
                                        %3:f32[] = add %0 %2
                                    in (%2, %3)
                                },
                                false={
                                    lambda %0:f32[], %1:ref<f32[]> .
                                    let %2:f32[] = reference_read %1
                                    in (%2, %0)
                                },
                            ]
                            in (%3, %4)
                        },
                        false={
                            lambda %0:bool[], %1:f32[], %2:ref<f32[]> .
                            let %3:f32[], %4:f32[] = condition %0 %1 %2 [
                                true={
                                    lambda %0:f32[], %1:ref<f32[]> .
                                    let %2:f32[] = reference_read %1
                                        %3:f32[] = add %0 %2
                                    in (%2, %3)
                                },
                                false={
                                    lambda %0:f32[], %1:ref<f32[]> .
                                    let %2:f32[] = reference_read %1
                                    in (%2, %0)
                                },
                            ]
                            in (%3, %4)
                        },
                    ]
                in (%4, %5)"},
        );

        let inputs = vec![
            TestIrValue::Array(Array::scalar(true).unwrap()),
            TestIrValue::Array(Array::scalar::<f32>(10.0).unwrap()),
            TestIrValue::Array(Array::scalar::<f32>(3.0).unwrap()),
        ];
        let outputs = vec![
            TestIrValue::Array(Array::scalar::<f32>(3.0).unwrap()),
            TestIrValue::Array(Array::scalar::<f32>(13.0).unwrap()),
        ];
        assert_eq!(source.interpret(inputs.clone()), Ok(outputs.clone()));
        assert_eq!(discharged.program().interpret(inputs), Ok(outputs));
    }

    #[test]
    fn test_condition_reference_discharge_selecting_nothing_is_identity() {
        // Preserving every allocation is the opposite extreme from full discharge, and it must be the identity: every
        // access, every transform, and every structured boundary replays exactly as the source declared it. This is the
        // sharpest statement of what "preserved" means, and it holds through a condition's attached regions.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut branch_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = branch_builder.add_input(reference_type.clone().into());
        let replacement = branch_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let previous = branch_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap()[0];
        let branch = branch_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![previous], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_branch = builder.import_region(branch.entry_region_ref());
        let false_branch = builder.import_region(branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let replacement = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let allocation =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let previous = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, allocation, replacement],
                None,
            )
            .unwrap()[0];
        let frozen = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![allocation], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![previous, frozen],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();

        assert_eq!(
            source.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[], %2:f32[] .
                let %3:ref<f32[]> = reference_new %1
                    %4:f32[] = condition %0 %3 %2 [
                        true={
                            lambda %0:ref<f32[]>, %1:f32[] .
                            let %2:f32[] = reference_swap %0 %1
                            in (%2)
                        },
                        false={
                            lambda %0:ref<f32[]>, %1:f32[] .
                            let %2:f32[] = reference_swap %0 %1
                            in (%2)
                        },
                    ]
                    %5:f32[] = reference_freeze %3
                in (%4, %5)"},
        );
        let preserved = source.clone().partially_discharge_references(0, &[]).unwrap();
        assert_eq!(preserved.output_count(), 2);
        assert_eq!(preserved.external_reference_bindings(), &[]);
        assert_eq!(preserved.program().to_string(), source.to_string());

        // Selecting the one target instead is full discharge, which is the other extreme of the same rewrite.
        let targets = source.reference_discharge_targets(0).unwrap();
        let discharged = ReferenceDischargeResult::try_from(
            source.clone().partially_discharge_references(0, targets.as_slice()).unwrap(),
        )
        .unwrap();
        assert_eq!(discharged.program().to_string(), source.discharge_references(0).unwrap().program().to_string());
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[], %2:f32[] .
                let %3:f32[], %4:f32[] = condition %0 %1 %2 [
                    true={
                        lambda %0:f32[], %1:f32[] .
                        in (%0, %1)
                    },
                    false={
                        lambda %0:f32[], %1:f32[] .
                        in (%0, %1)
                    },
                ]
                in (%3, %4)"},
        );
    }

    #[test]
    fn test_condition_reference_discharge_orders_multiple_allocations_by_parent_boundary() {
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut true_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let first = true_builder.add_input(reference_type.clone().into());
        let second = true_builder.add_input(reference_type.clone().into());
        true_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let second_replacement = true_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let first_snapshot =
            true_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![first], None).unwrap()[0];
        let second_snapshot = true_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![second, second_replacement], None)
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![first_snapshot, second_snapshot],
                vec![Placeholder; 4],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut false_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let first = false_builder.add_input(reference_type.clone().into());
        let second = false_builder.add_input(reference_type.clone().into());
        let first_replacement = false_builder.add_input(ArrayType::scalar(DataType::F32).into());
        false_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let first_snapshot = false_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![first, first_replacement], None)
            .unwrap()[0];
        let second_snapshot = false_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![second], None)
            .unwrap()[0];
        let false_branch = false_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![first_snapshot, second_snapshot],
                vec![Placeholder; 4],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let first = builder.add_input(reference_type.clone().into());
        let second = builder.add_input(reference_type.into());
        let first_replacement = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let second_replacement = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let outputs = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, first, second, first_replacement, second_replacement],
                None,
            )
            .unwrap()
            .to_vec();
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 5], vec![Placeholder; 2])
            .unwrap();

        // Both branches write a different allocation, so both allocations cross the boundary; the appended final-state
        // outputs follow parent entry-boundary order rather than the order in which either branch happens to access
        // them.
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[], %2:f32[], %3:f32[], %4:f32[] .
                let %5:f32[], %6:f32[], %7:f32[], %8:f32[] = condition %0 %1 %2 %3 %4 [
                    true={
                        lambda %0:f32[], %1:f32[], %2:f32[], %3:f32[] .
                        in (%0, %1, %0, %3)
                    },
                    false={
                        lambda %0:f32[], %1:f32[], %2:f32[], %3:f32[] .
                        in (%0, %1, %2, %1)
                    },
                ]
                in (%5, %6, %7, %8)"},
        );
        assert_eq!(discharged.output_count(), 2);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[
                ExternalReferenceBinding::new(ReferenceSource::Input { index: 1 }, Some(2)),
                ExternalReferenceBinding::new(ReferenceSource::Input { index: 2 }, Some(3)),
            ],
        );

        // The true branch swaps only the second allocation, leaving the first allocation's final state at its entering
        // value.
        let inputs = vec![
            TestIrValue::Array(Array::scalar(true).unwrap()),
            TestIrValue::Array(Array::scalar::<f32>(10.0).unwrap()),
            TestIrValue::Array(Array::scalar::<f32>(20.0).unwrap()),
            TestIrValue::Array(Array::scalar::<f32>(11.0).unwrap()),
            TestIrValue::Array(Array::scalar::<f32>(22.0).unwrap()),
        ];
        assert_eq!(
            discharged.program().interpret(inputs),
            Ok(vec![
                TestIrValue::Array(Array::scalar::<f32>(10.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(20.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(10.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(22.0).unwrap()),
            ]),
        );

        // The false branch swaps only the first allocation, which mirrors the same contract on the other position.
        let inputs = vec![
            TestIrValue::Array(Array::scalar(false).unwrap()),
            TestIrValue::Array(Array::scalar::<f32>(10.0).unwrap()),
            TestIrValue::Array(Array::scalar::<f32>(20.0).unwrap()),
            TestIrValue::Array(Array::scalar::<f32>(11.0).unwrap()),
            TestIrValue::Array(Array::scalar::<f32>(22.0).unwrap()),
        ];
        assert_eq!(
            discharged.program().interpret(inputs),
            Ok(vec![
                TestIrValue::Array(Array::scalar::<f32>(10.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(20.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(11.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(20.0).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_condition_reference_discharge_isolates_its_branches() {
        // Both branches accumulate a different amount into the same allocation and return the state they observe. If
        // either branch's staging leaked into the other's, the second branch would start from the first's successor
        // state.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let branch = |amount: f32| {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            let update = builder.add_constant(TestIrValue::Array(Array::scalar::<f32>(amount).unwrap()));
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                .unwrap();
            let snapshot =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_branch = builder.import_program(branch(1.0));
        let false_branch = builder.import_program(branch(10.0));
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let allocation =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let snapshot = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, allocation],
                None,
            )
            .unwrap()[0];

        // The condition's outputs are bound in the *parent*, so a later parent instruction consumes them directly. A
        // value stamped with a branch's own destination builder would be rejected here instead of staged.
        let doubled = builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![snapshot, snapshot], None)
            .unwrap()[0];
        let frozen = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![allocation], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![doubled, frozen],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let discharged = source.clone().discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[] .
                let %2:f32[], %3:f32[] = condition %0 %1 [
                    true={
                        lambda %0:f32[] .
                        let %1:f32[] = const 1.0
                            %2:f32[] = add %0 %1
                        in (%2, %2)
                    },
                    false={
                        lambda %0:f32[] .
                        let %1:f32[] = const 10.0
                            %2:f32[] = add %0 %1
                        in (%2, %2)
                    },
                ]
                    %4:f32[] = add %2 %2
                in (%4, %3)"},
        );
        for (predicate, expected) in [
            (
                true,
                vec![
                    TestIrValue::Array(Array::scalar::<f32>(6.0).unwrap()),
                    TestIrValue::Array(Array::scalar::<f32>(3.0).unwrap()),
                ],
            ),
            (
                false,
                vec![
                    TestIrValue::Array(Array::scalar::<f32>(24.0).unwrap()),
                    TestIrValue::Array(Array::scalar::<f32>(12.0).unwrap()),
                ],
            ),
        ] {
            let inputs = vec![
                TestIrValue::Array(Array::scalar(predicate).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(2.0).unwrap()),
            ];
            assert_eq!(source.clone().interpret(inputs.clone()), Ok(expected.clone()));
            assert_eq!(discharged.program().interpret(inputs), Ok(expected));
        }

        // The branches of [`state_condition_program`] mutate the shared allocation with unequal accesses, so the eager
        // reference interpreter and the discharged program must agree on the branch snapshot as well as on the state
        // observed after the condition.
        let source = state_condition_program();
        let discharged = source.clone().discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[] .
                let %2:f32[], %3:f32[] = condition %0 %1 [
                    true={
                        lambda %0:f32[] .
                        let %1:f32[] = const 1.0
                            %2:f32[] = add %0 %1
                        in (%2, %2)
                    },
                    false={
                        lambda %0:f32[] .
                        let %1:f32[] = const 9.0
                        in (%0, %1)
                    },
                ]
                in (%2, %3)"},
        );
        assert_eq!(discharged.external_reference_bindings(), &[]);
        for (predicate, expected) in [
            (
                true,
                vec![
                    TestIrValue::Array(Array::scalar::<f32>(5.0).unwrap()),
                    TestIrValue::Array(Array::scalar::<f32>(5.0).unwrap()),
                ],
            ),
            (
                false,
                vec![
                    TestIrValue::Array(Array::scalar::<f32>(4.0).unwrap()),
                    TestIrValue::Array(Array::scalar::<f32>(9.0).unwrap()),
                ],
            ),
        ] {
            let inputs = vec![
                TestIrValue::Array(Array::scalar(predicate).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(4.0).unwrap()),
            ];
            let eager = source.clone().interpret(inputs.clone()).unwrap();
            assert_eq!(eager, expected);
            assert_eq!(discharged.program().interpret(inputs), Ok(eager));
        }
    }

    #[test]
    fn test_condition_reference_discharge_rejects_a_branch_local_allocation_that_escapes() {
        // Both branches allocate an allocation of their own and return it, so the condition's output denotes a
        // reference its caller never threaded in. Merging that output would hand the caller a handle into an
        // environment that no longer exists, so the rewrite rejects it, and eager interpretation rejects it as well.
        let branch = || {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
            let allocation =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![allocation], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_branch = builder.import_program(branch());
        let false_branch = builder.import_program(branch());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let escaped = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, initial],
                None,
            )
            .unwrap()[0];
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![escaped], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![frozen], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let expected = Err(ProgramError::Type(TypeError::invalid(
            "`condition` reference output 0 must denote the same reference in both branches, but the `true` branch \
             returns a reference allocated in the region and the `false` branch returns a reference allocated in the \
             region",
        )));
        assert_eq!(source.clone().discharge_references(0).map(|_| ()), expected);
        assert_eq!(
            source
                .interpret(vec![array(Array::scalar(true).unwrap()), array(Array::scalar(1.0f32).unwrap())])
                .map(|_| ()),
            expected,
        );
    }

    #[test]
    fn test_condition_reference_discharge_rejects_branch_dependent_reference_outputs() {
        // Each branch returns a different reference input, so the reference that the output denotes depends on the
        // predicate. Discharge and eager interpretation reject the condition with the same error, whichever branch
        // the predicate selects.
        let source = branch_dependent_reference_program();
        assert_eq!(
            source.to_string(),
            indoc! {"
                lambda %0:bool[], %1:ref<f32[]>, %2:ref<f32[]> .
                let %3:ref<f32[]> = condition %0 %1 %2 [
                    true={
                        lambda %0:ref<f32[]>, %1:ref<f32[]> .
                        in (%0)
                    },
                    false={
                        lambda %0:ref<f32[]>, %1:ref<f32[]> .
                        in (%1)
                    },
                ]
                    %4:f32[] = reference_read %3
                in (%4)"},
        );
        let expected = Err(ProgramError::Type(TypeError::invalid(
            "`condition` reference output 0 must denote the same reference in both branches, but the `true` branch \
             returns region input 0 and the `false` branch returns region input 1",
        )));
        assert_eq!(source.clone().discharge_references(0).map(|_| ()), expected);
        for predicate in [true, false] {
            assert_eq!(
                source
                    .interpret(vec![
                        array(Array::scalar(predicate).unwrap()),
                        TestIrValue::Reference(ArrayReference::new(Array::scalar(1.0f32).unwrap())),
                        TestIrValue::Reference(ArrayReference::new(Array::scalar(2.0f32).unwrap())),
                    ])
                    .map(|_| ()),
                expected,
            );
        }

        // Branches that return different captured references are rejected in the same way, and the diagnostic names
        // each capture.
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let branch = |capture: usize| {
            let mut builder = ProgramBuilder::<DischargeCapture, DischargeCaptureOperation>::new();
            let reference = builder.add_constant(DischargeCapture::new(capture, reference_type.clone()));
            builder
                .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(vec![reference], Vec::new(), vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<DischargeCapture, DischargeCaptureOperation>::new();
        let branches = vec![
            builder.import_region(branch(0).entry_region_ref()),
            builder.import_region(branch(1).entry_region_ref()),
        ];
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let selected = builder
            .add_instruction(ConditionOperation::<ArrayIrType>::new(), branches, vec![predicate], None)
            .unwrap()[0];
        let value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![selected], None).unwrap()[0];
        let program = builder
            .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let captures = [1.0f32, 2.0]
            .map(|value| TestIrValue::Reference(ArrayReference::new(Array::scalar(value).unwrap())))
            .to_vec();
        assert_eq!(
            ClosedProgram::new(program, captures).unwrap().discharge_references().map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "`condition` reference output 0 must denote the same reference in both branches, but the `true` branch \
                 returns captured reference 0 and the `false` branch returns captured reference 1",
            ))),
        );
    }

    #[test]
    fn test_condition_reference_discharge_captures_a_lazy_view_root_and_dynamic_binding() {
        // Branches traced as nested regions read a captured root through lazy views. The views are not program values,
        // so the regions capture only the root and the dynamic index, and each access re-applies its own transforms.
        let context = TracingContext::<DischargeCapture, TestIrOperation, TestIrValue>::new();
        let predicate = context.input(ArrayType::scalar(DataType::Boolean).into());
        let offset = context.input(ArrayType::scalar(DataType::F32).into());
        let root = TestIrValue::Reference(ArrayReference::new(Array::vector(vec![10f32, 20., 30.]).unwrap()));
        let index = TestIrValue::Array(Array::scalar(2i32).unwrap());
        let (_, true_branch) = NestedTracingContext::trace(
            context.clone(),
            |inputs: Vec<Tracer<_>>| {
                let branch = inputs[0].context().clone();
                let root = StagingContext::constant(&branch, branch.capture(root.clone())?);
                let index = StagingContext::constant(&branch, branch.capture(index.clone())?);
                let element = ReferenceView::new(root)?.dynamic_index(0, &index)?.read()?;
                Ok(branch.bind(AddOperation::<ArrayType>::new(), Vec::new(), &[element, inputs[0].clone()])?.remove(0))
            },
            vec![ArrayType::scalar(DataType::F32).into()],
        )
        .unwrap();
        let (_, false_branch) = NestedTracingContext::trace(
            context.clone(),
            |inputs: Vec<Tracer<_>>| Ok(inputs[0].clone()),
            vec![ArrayType::scalar(DataType::F32).into()],
        )
        .unwrap();
        assert_eq!(
            true_branch.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:ref<f32[3]> = const capture#0:ref<f32[3]>
                    %2:i32[] = const capture#1:i32[]
                    %3:f32[] = reference_read [transforms=[index(axis=0, index=dynamic)]] %1 %2
                    %4:f32[] = add %3 %0
                in (%4)"},
        );
        let access = &true_branch.instructions()[0];
        assert_eq!(access.operation().reference_access_descriptor(0).unwrap().bindings(), 1..2);
        assert_eq!(
            access.operation().reference_access_descriptor(0).unwrap().transforms(),
            &[ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }],
        );
        let output = context
            .bind(ConditionOperation::<ArrayIrType>::new(), vec![true_branch, false_branch], &[predicate, offset])
            .unwrap()
            .remove(0);
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let captures = context.captures().borrow().clone();
        assert_eq!(captures, vec![root, index.clone()]);

        // Discharge threads the captured root's state as an array input while the index capture stays an ordinary
        // value, and executing the result selects the element the captured index names when the branch runs.
        let closed = ClosedProgram::new(program, captures.clone()).unwrap();
        let discharged = closed.discharge_references().unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[3], %1:i32[], %2:bool[], %3:f32[] .
                let %4:f32[] = condition %2 %3 %0 [
                    true={
                        lambda %0:f32[], %1:f32[3] .
                        let %2:i32[] = const capture#1:i32[]
                            %3:f32[1] = dynamic_slice [sizes=[1]] %1 %2
                            %4:f32[] = reshape [shape=[]] %3
                            %5:f32[] = add %4 %0
                        in (%5)
                    },
                    false={
                        lambda %0:f32[], %1:f32[3] .
                        in (%0)
                    },
                ]
                in (%4)"},
        );
        assert_eq!(
            discharged.program().input_types(),
            vec![
                ArrayType::new_static(DataType::F32, [3]).into(),
                ArrayType::scalar(DataType::I32).into(),
                ArrayType::scalar(DataType::Boolean).into(),
                ArrayType::scalar(DataType::F32).into(),
            ],
        );
        assert!(!discharged.program().entry_region_ref().contains_reference_accesses_in_closure());
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, None)],
        );
        let executable = resolve_captures(discharged.program(), &captures);
        assert_eq!(
            executable.to_string(),
            indoc! {"
                lambda %0:f32[3], %1:i32[], %2:bool[], %3:f32[] .
                let %4:f32[] = condition %2 %3 %0 [
                    true={
                        lambda %0:f32[], %1:f32[3] .
                        let %2:i32[] = const 2
                            %3:f32[1] = dynamic_slice [sizes=[1]] %1 %2
                            %4:f32[] = reshape [shape=[]] %3
                            %5:f32[] = add %4 %0
                        in (%5)
                    },
                    false={
                        lambda %0:f32[], %1:f32[3] .
                        in (%0)
                    },
                ]
                in (%4)"},
        );
        let state = TestIrValue::Array(Array::vector(vec![10f32, 20., 30.]).unwrap());
        assert_eq!(
            executable.interpret(vec![
                state.clone(),
                index.clone(),
                TestIrValue::Array(Array::scalar(true).unwrap()),
                TestIrValue::Array(Array::scalar(0.5f32).unwrap()),
            ]),
            Ok(vec![TestIrValue::Array(Array::scalar(30.5f32).unwrap())]),
        );
        assert_eq!(
            executable.interpret(vec![
                state,
                index,
                TestIrValue::Array(Array::scalar(false).unwrap()),
                TestIrValue::Array(Array::scalar(0.5f32).unwrap()),
            ]),
            Ok(vec![TestIrValue::Array(Array::scalar(0.5f32).unwrap())]),
        );
    }

    #[test]
    fn test_condition_reference_discharge_resolves_reference_captures_inside_branches() {
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut branch_builder = ProgramBuilder::<DischargeCapture, DischargeCaptureOperation>::new();
        let reference = branch_builder.add_constant(DischargeCapture::new(0, reference_type.into()));
        let value = branch_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let branch = branch_builder
            .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(vec![value], Vec::new(), vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<DischargeCapture, DischargeCaptureOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let value = builder
            .add_instruction(ConditionOperation::<ArrayIrType>::new(), vec![branch, branch], vec![predicate], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let reference = ArrayReference::new(Array::scalar(4.0f32).unwrap());
        let closed = ClosedProgram::new(program, vec![TestIrValue::Reference(reference)]).unwrap();

        let discharged = closed.discharge_references().unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:bool[] .
                let %2:f32[] = condition %1 %0 [
                    true={
                        lambda %0:f32[] .
                        in (%0)
                    },
                    false={
                        lambda %0:f32[] .
                        in (%0)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, None)],
        );
        // `ExternalReferenceBinding` serialization is owned by `programs`; no test in this module owns it.
        assert_eq!(
            serde_json::to_string(discharged.external_reference_bindings()).unwrap(),
            r#"[{"source":{"capture":{"index":0}},"output_index":null}]"#,
        );
        assert_eq!(
            discharged.program().input_types(),
            vec![ArrayType::scalar(DataType::F32).into(), ArrayType::scalar(DataType::Boolean).into()],
        );
    }

    #[test]
    fn test_condition_reference_discharge_rejects_mutations_under_varying_predicates() {
        // A predicate that varies along `x` lets the devices take different branches, so a branch that mutates a
        // reference input is accepted only when the referent varies along `x` as well.
        let program = |varying: bool| {
            let (referent_type, predicate_type) = manual_mutation_types(varying);
            let reference_type = ArrayIrType::Reference(ReferenceType::new(referent_type));
            let branch = |mutate: bool| {
                let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
                let reference = builder.add_input(reference_type.clone());
                manual_mutation_branch(&mut builder, reference, mutate);
                builder
                    .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder], Vec::new())
                    .unwrap()
            };
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let branches = vec![builder.import_program(branch(true)), builder.import_program(branch(false))];
            let predicate = builder.add_input(predicate_type.into());
            let reference = builder.add_input(reference_type.clone());
            builder
                .add_instruction(ConditionOperation::<ArrayIrType>::new(), branches, vec![predicate, reference], None)
                .unwrap();
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
                .unwrap()
        };
        let discharged = program(true).discharge_references(0).unwrap();
        let (referent_type, predicate_type) = manual_mutation_types(true);
        assert_eq!(discharged.output_count(), 0);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Input { index: 1 }, Some(0))],
        );
        assert_eq!(
            discharged.program().input_types(),
            vec![predicate_type.clone().into(), referent_type.clone().into()],
        );
        assert_eq!(discharged.program().output_types(), vec![referent_type.clone().into()]);
        for (predicate, expected_state) in [(true, [2.0f32, 4.0]), (false, [1.0f32, 2.0])] {
            assert_eq!(
                discharged.program().interpret(vec![
                    array(Array::from_elements(predicate_type.clone(), &[predicate]).unwrap()),
                    array(Array::from_elements(referent_type.clone(), &[1.0f32, 2.0]).unwrap()),
                ]),
                Ok(vec![array(Array::from_elements(referent_type.clone(), &expected_state).unwrap())]),
            );
        }
        assert_eq!(
            program(false).discharge_references(0).map(|_| ()),
            Err(TypeError::invalid(
                "`condition` `true` branch mutates region input 0 of type \
                 `ref<f32[2][sharding={mesh<['x'=2:manual]>, [{}]}]>`, whose referent does not vary over every manual \
                 axis that the predicate `bool[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]` varies \
                 over, so devices that take different branches would leave it different across devices; vary the \
                 referent over those axes",
            )
            .into()),
        );

        // Branches that capture a reference instead of receiving it as an input are subject to the same requirement,
        // because discharge publishes the captured reference's final state back exactly like that of an input.
        let closed_program = |varying: bool| {
            let (referent_type, predicate_type) = manual_mutation_types(varying);
            let reference_type = ArrayIrType::Reference(ReferenceType::new(referent_type.clone()));
            let branch = |mutate: bool| {
                let mut builder = ProgramBuilder::<DischargeCapture, DischargeCaptureOperation>::new();
                let reference = builder.add_constant(DischargeCapture::new(0, reference_type.clone()));
                manual_mutation_branch(&mut builder, reference, mutate);
                builder
                    .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(Vec::new(), Vec::new(), Vec::new())
                    .unwrap()
            };
            let mut builder = ProgramBuilder::<DischargeCapture, DischargeCaptureOperation>::new();
            let branches = vec![
                builder.import_region(branch(true).entry_region_ref()),
                builder.import_region(branch(false).entry_region_ref()),
            ];
            let predicate = builder.add_input(predicate_type.into());
            builder
                .add_instruction(ConditionOperation::<ArrayIrType>::new(), branches, vec![predicate], None)
                .unwrap();
            let program = builder
                .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(Vec::new(), vec![Placeholder], Vec::new())
                .unwrap();
            let referent = Array::from_elements(referent_type, &[1.0f32, 2.0]).unwrap();
            ClosedProgram::new(program, vec![TestIrValue::Reference(ArrayReference::new(referent))]).unwrap()
        };
        let discharged = closed_program(true).discharge_references().unwrap();
        let (referent_type, predicate_type) = manual_mutation_types(true);
        assert_eq!(discharged.capture_count(), 1);
        assert_eq!(discharged.output_count(), 0);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, Some(0))],
        );
        assert_eq!(
            discharged.program().input_types(),
            vec![referent_type.clone().into(), predicate_type.clone().into()],
        );
        assert_eq!(discharged.program().output_types(), vec![referent_type.clone().into()]);
        assert_eq!(
            closed_program(false).discharge_references().map(|_| ()),
            Err(TypeError::invalid(
                "`condition` `true` branch mutates captured reference 0 of type \
                 `ref<f32[2][sharding={mesh<['x'=2:manual]>, [{}]}]>`, whose referent does not vary over every manual \
                 axis that the predicate `bool[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]` varies \
                 over, so devices that take different branches would leave it different across devices; vary the \
                 referent over those axes",
            )
            .into()),
        );
    }

    #[test]
    fn test_condition_reference_discharge_validates_mutations_in_either_branch_under_varying_predicates() {
        // Only the `false` branch mutates the invariant reference input 1, which suffices for discharge and eager
        // interpretation to reject the condition whichever branch the predicate selects. Reading that reference is
        // fine, and partial discharge that preserves it publishes no final state for it, so both are accepted.
        let (invariant_referent_type, predicate_type) = manual_mutation_types(false);
        let (varying_referent_type, _) = manual_mutation_types(true);
        let invariant_type = ArrayIrType::Reference(ReferenceType::new(invariant_referent_type.clone()));
        let varying_type = ArrayIrType::Reference(ReferenceType::new(varying_referent_type.clone()));
        let program = |mutate_invariant: bool| {
            let branch = |mutate: bool| {
                let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
                let invariant = builder.add_input(invariant_type.clone());
                let varying = builder.add_input(varying_type.clone());
                manual_mutation_branch(&mut builder, invariant, mutate && mutate_invariant);
                manual_mutation_branch(&mut builder, varying, mutate);
                builder
                    .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
                    .unwrap()
            };
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let branches = vec![builder.import_program(branch(false)), builder.import_program(branch(true))];
            let predicate = builder.add_input(predicate_type.clone().into());
            let invariant = builder.add_input(invariant_type.clone());
            let varying = builder.add_input(varying_type.clone());
            builder
                .add_instruction(
                    ConditionOperation::<ArrayIrType>::new(),
                    branches,
                    vec![predicate, invariant, varying],
                    None,
                )
                .unwrap();
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder; 3], Vec::new())
                .unwrap()
        };
        let interpret = |program: &Program<TestIrValue, TestIrOperation, Vec<TestIrValue>, Vec<TestIrValue>>,
                         predicate: bool| {
            let invariant =
                ArrayReference::new(Array::from_elements(invariant_referent_type.clone(), &[1.0f32, 2.0]).unwrap());
            let varying =
                ArrayReference::new(Array::from_elements(varying_referent_type.clone(), &[3.0f32, 4.0]).unwrap());
            program.interpret(vec![
                array(Array::from_elements(predicate_type.clone(), &[predicate]).unwrap()),
                TestIrValue::Reference(invariant.clone()),
                TestIrValue::Reference(varying.clone()),
            ])?;
            Ok::<_, ProgramError>((invariant.read().unwrap(), varying.read().unwrap()))
        };
        let expected = Err(ProgramError::Type(TypeError::invalid(
            "`condition` `false` branch mutates region input 0 of type \
             `ref<f32[2][sharding={mesh<['x'=2:manual]>, [{}]}]>`, whose referent does not vary over every manual axis \
             that the predicate `bool[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]` varies over, so \
             devices that take different branches would leave it different across devices; vary the referent over \
             those axes",
        )));
        let mutating = program(true);
        assert_eq!(mutating.clone().discharge_references(0).map(|_| ()), expected);
        assert_eq!(interpret(&mutating, true).map(|_| ()), expected);
        assert_eq!(interpret(&mutating, false).map(|_| ()), expected);

        // A read-only invariant reference is accepted.
        let reading = program(false);
        assert_eq!(
            interpret(&reading, false),
            Ok((
                Array::from_elements(invariant_referent_type.clone(), &[1.0f32, 2.0]).unwrap(),
                Array::from_elements(varying_referent_type.clone(), &[6.0f32, 8.0]).unwrap(),
            )),
        );
        assert_eq!(reading.discharge_references(0).map(|discharged| discharged.output_count()), Ok(0));

        // Partial discharge of the varying reference alone preserves the mutated invariant reference.
        let target = ReferenceDischargeTarget::External(ReferenceSource::Input { index: 2 });
        let discharged = mutating.partially_discharge_references(0, &[target]).unwrap();
        let predicate = "bool[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]";
        let invariant = "f32[2][sharding={mesh<['x'=2:manual]>, [{}]}]";
        let varying = "f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]";
        assert_eq!(
            discharged.program().to_string(),
            formatdoc! {"
                lambda %0:{predicate}, %1:ref<{invariant}>, %2:{varying} .
                let %3:{varying} = condition %0 %1 %2 [
                    true={{
                        lambda %0:ref<{invariant}>, %1:{varying} .
                        let %2:{invariant} = reference_read %0
                        in (%1)
                    }},
                    false={{
                        lambda %0:ref<{invariant}>, %1:{varying} .
                        let %2:{invariant} = reference_read %0
                            %3:{invariant} = add %2 %2
                            () = reference_write %0 %3
                            %4:{varying} = add %1 %1
                        in (%4)
                    }},
                ]
                in (%3)"},
        );
    }

    #[test]
    fn test_condition_reference_discharge_lowers_static_index_transforms_inside_branches() {
        // The `true` branch accumulates into element 1 of a local `f32[3]` allocation through a static index transform,
        // and the `false` branch leaves it untouched. Discharge lowers the transformed update inside the branch region
        // to a `slice`/`update_slice` pair over the threaded state, and both predicates compute the eager results.
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]));
        let reference_type = ReferenceType::new(vector_type.clone());
        let true_branch = {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            let update = builder.add_constant(TestIrValue::Array(Array::scalar::<f32>(1.0).unwrap()));
            builder
                .add_instruction(
                    ReferenceAddUpdateOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                        axis: 0,
                        index: ArrayReferenceTransformIndex::Static(1),
                    }]),
                    Vec::new(),
                    vec![reference, update],
                    None,
                )
                .unwrap();
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let false_branch = {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let reference = builder.add_input(reference_type.into());
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(vector_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let reference = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let true_inputs = vec![
            TestIrValue::Array(Array::scalar(true).unwrap()),
            TestIrValue::Array(Array::vector::<f32>(vec![1.0, 2.0, 3.0]).unwrap()),
        ];
        let false_inputs = vec![
            TestIrValue::Array(Array::scalar(false).unwrap()),
            TestIrValue::Array(Array::vector::<f32>(vec![1.0, 2.0, 3.0]).unwrap()),
        ];
        assert_eq!(
            source.clone().interpret(true_inputs.clone()),
            Ok(vec![TestIrValue::Array(Array::vector::<f32>(vec![1.0, 3.0, 3.0]).unwrap())]),
        );
        assert_eq!(
            source.clone().interpret(false_inputs.clone()),
            Ok(vec![TestIrValue::Array(Array::vector::<f32>(vec![1.0, 2.0, 3.0]).unwrap())]),
        );

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().interpret(true_inputs),
            Ok(vec![TestIrValue::Array(Array::vector::<f32>(vec![1.0, 3.0, 3.0]).unwrap())]),
        );
        assert_eq!(
            discharged.program().interpret(false_inputs),
            Ok(vec![TestIrValue::Array(Array::vector::<f32>(vec![1.0, 2.0, 3.0]).unwrap())]),
        );
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[3] .
                let %2:f32[3] = condition %0 %1 [
                    true={
                        lambda %0:f32[3] .
                        let %1:f32[] = const 1.0
                            %2:f32[1] = slice [start_indices=[1], limits=[2]] %0
                            %3:f32[] = reshape [shape=[]] %2
                            %4:f32[] = add %3 %1
                            %5:f32[1] = reshape [shape=[1]] %4
                            %6:f32[3] = update_slice [start_indices=[1]] %0 %5
                        in (%6)
                    },
                    false={
                        lambda %0:f32[3] .
                        in (%0)
                    },
                ]
                in (%2)"},
        );
    }

    #[test]
    fn test_condition_interpretation() {
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let operation = ConditionOperation::<ArrayType>::new();
        let true_branch = scalar_branch(ArrayOperation::Add(AddOperation::new()));
        let false_branch = scalar_branch(ArrayOperation::ZeroLike(ZeroLikeOperation::new()));

        // Eager binding interprets the predicate-selected branch through detached region access, and interpretation
        // without a predicate input is rejected.
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let predicate = |value: bool| Array::from_elements::<bool>(predicate_type.clone(), &[value]).unwrap();
        let outputs = context
            .bind(
                operation,
                vec![true_branch.clone(), false_branch.clone()],
                &[predicate(true), Array::scalar(4.0).unwrap()],
            )
            .unwrap();
        assert_eq!(outputs, vec![Array::scalar(8.0f64).unwrap()]);
        let outputs = context
            .bind(
                operation,
                vec![true_branch.clone(), false_branch.clone()],
                &[predicate(false), Array::scalar(4.0).unwrap()],
            )
            .unwrap();
        assert_eq!(outputs, vec![Array::scalar(0.0f64).unwrap()]);
        assert_eq!(
            context.bind(operation, vec![true_branch.clone(), false_branch.clone()], &[]),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );
    }

    #[test]
    fn test_condition_interpretation_skips_effects_of_the_unselected_branch() {
        // The `true` branch adds `1.0` to the reference, while the `false` branch adds `10.0` and then fails an
        // assertion. Interpretation executes only the selected branch, so the `true` predicate observes neither the
        // update nor the failure of the `false` branch, and the `false` predicate applies its update before failing.
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let branch = |update: f32, fails: bool| {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let reference = builder.add_input(reference_type.clone());
            let update = builder.add_constant(array(Array::scalar(update).unwrap()));
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                .unwrap();
            if fails {
                let holds = builder.add_constant(array(Array::scalar(false).unwrap()));
                builder
                    .add_instruction(
                        TestIrOperation::Array(ArrayOperation::Assert(AssertOperation::new(
                            "unselected branch executed",
                        ))),
                        Vec::new(),
                        vec![holds],
                        None,
                    )
                    .unwrap();
            }
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder], Vec::new())
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_region = builder.import_program(branch(1.0, false));
        let false_region = builder.import_program(branch(10.0, true));
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let reference = builder.add_input(reference_type);
        builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_region, false_region],
                vec![predicate, reference],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:ref<f32[]> .
                let () = condition %0 %1 [
                    true={
                        lambda %0:ref<f32[]> .
                        let %1:f32[] = const 1.0
                            () = reference_add_update %0 %1
                        in ()
                    },
                    false={
                        lambda %0:ref<f32[]> .
                        let %1:f32[] = const 10.0
                            %2:bool[] = const false
                            () = reference_add_update %0 %1
                            () = assert [message=\"unselected branch executed\", labels=[]] %2
                        in ()
                    },
                ]
                in ()"},
        );
        let reference = ArrayReference::new(Array::scalar(2.0f32).unwrap());
        assert_eq!(
            program.interpret(vec![array(Array::scalar(true).unwrap()), TestIrValue::Reference(reference.clone())]),
            Ok(Vec::new()),
        );
        assert_eq!(reference.read(), Ok(Array::scalar(3.0f32).unwrap()));
        let error = program
            .interpret(vec![array(Array::scalar(false).unwrap()), TestIrValue::Reference(reference.clone())])
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "unselected branch executed".to_string(),
                observations: Vec::new()
            }),
        );
        assert_eq!(reference.read(), Ok(Array::scalar(13.0f32).unwrap()));
    }

    #[test]
    fn test_condition_interpretation_validates_direct_branch_contracts() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let operation = ConditionOperation::<ArrayType>::new();
        let scalar_branch = scalar_branch(ArrayOperation::Add(AddOperation::new()));
        let boolean_branch = boolean_branch();
        for predicate in [true, false] {
            assert_eq!(
                context.bind(
                    operation,
                    vec![scalar_branch.clone(), boolean_branch.clone()],
                    &[Array::scalar(predicate).unwrap(), Array::scalar(4.0f64).unwrap()],
                ),
                Err(TypeError::invalid(
                    "`condition` branch output type signature mismatch: expected [f64[]] but got [bool[]]",
                )
                .into()),
            );
        }
        assert_eq!(
            context.bind(
                operation,
                vec![scalar_branch, vector_scale_branch(2, 2.0)],
                &[Array::scalar(true).unwrap(), Array::scalar(4.0f64).unwrap()],
            ),
            Err(TypeError::invalid(
                "`condition` branch input type signature mismatch: expected [f64[]] but got [f64[2]]",
            )
            .into()),
        );

        // The selected branch would fail its assertion if run; its empty output boundary must still reject a peer
        // that returns a value before executing either branch.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let output = builder.add_constant(Array::scalar(1.0f64).unwrap());
        let value_branch = builder.build(vec![output], Vec::new(), vec![Placeholder]).unwrap();
        assert_eq!(
            context.bind(operation, vec![failing_unit_branch(), value_branch], &[Array::scalar(true).unwrap()]),
            Err(TypeError::invalid("`condition` branch output type signature mismatch: expected [] but got [f64[]]")
                .into()),
        );

        let unit_branch = ProgramBuilder::<Array, ArrayOperation<Array>>::new()
            .build::<Vec<Array>, Vec<Array>>(Vec::new(), Vec::new(), Vec::new())
            .unwrap();
        assert_eq!(
            context.bind(operation, vec![unit_branch.clone(), unit_branch.clone()], &[Array::scalar(true).unwrap()]),
            Ok(Vec::new()),
        );
        assert_eq!(
            context.bind(operation, vec![unit_branch.clone(), unit_branch], &[Array::vector(vec![true]).unwrap()]),
            Err(TypeError::invalid("`condition` predicate type must be a scalar boolean, but got `bool[1]`").into()),
        );
    }

    #[test]
    fn test_condition_interpretation_validates_runtime_reference_extents() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(8)).unwrap());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = builder.add_input(ReferenceType::new(array_type.clone()).into());
        let value = builder.add_input(array_type.into());
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, value], None)
            .unwrap();
        let branch = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            branch.to_string(),
            indoc! {"
                lambda %0:ref<f32[extent]>, %1:f32[extent] .
                let () = reference_write %0 %1
                in (%0)"},
        );
        let context = TestEagerContext::new();
        let reference = ArrayReference::new(Array::vector(vec![1f32, 2.0, 3.0]).unwrap());
        assert_eq!(
            context.bind(
                ConditionOperation::new(),
                vec![branch.clone(), branch.clone()],
                &[
                    array(Array::scalar(true).unwrap()),
                    TestIrValue::Reference(reference.clone()),
                    array(Array::vector(vec![4f32, 5.0, 6.0]).unwrap()),
                ],
            ),
            Ok(vec![TestIrValue::Reference(reference.clone())]),
        );
        assert_eq!(reference.read(), Ok(Array::vector(vec![4f32, 5.0, 6.0]).unwrap()));

        // A reference and an ordinary value sharing one formal extent must agree before the selected write executes.
        let result = context.bind(
            ConditionOperation::new(),
            vec![branch.clone(), branch],
            &[
                array(Array::scalar(false).unwrap()),
                TestIrValue::Reference(reference.clone()),
                array(Array::vector(vec![7f32, 8.0]).unwrap()),
            ],
        );
        assert!(matches!(
            result,
            Err(ProgramError::Type(error))
                if error.downcast_custom::<DimensionError>() == Some(&DimensionError::InputDimensionMismatch {
                    dimension: "extent".to_string(), expected: 3, actual: 2,
                }),
        ));
        assert_eq!(reference.read(), Ok(Array::vector(vec![4f32, 5.0, 6.0]).unwrap()));
    }

    #[test]
    fn test_condition_interpretation_validates_dimension_payloads() {
        // A first-class dimension of extent 2 contradicts the `f32[3]` array, because the branches declare both inputs
        // with the identity `n`. Entering a branch must reject that pair before any of its work executes, in direct
        // binding and in program replay, whether the branch returns its array or writes it to a reference.
        let n_type = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![n_type.dimension().clone()]));
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        builder.add_input(n_type.clone().into());
        let value = builder.add_input(vector_type.clone().into());
        let returning_branch = builder.build(vec![value], vec![Placeholder; 2], vec![Placeholder]).unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        builder.add_input(n_type.clone().into());
        let reference = builder.add_input(ReferenceType::new(vector_type.clone()).into());
        let value = builder.add_input(vector_type.into());
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, value], None)
            .unwrap();
        let writing_branch = builder.build(Vec::new(), vec![Placeholder; 3], Vec::new()).unwrap();

        let context = TestEagerContext::new();
        let predicate = array(Array::scalar(true).unwrap());
        let vector = array(Array::vector(vec![1f32, 2.0, 3.0]).unwrap());
        let reference = ArrayReference::new(Array::vector(vec![4f32, 5.0, 6.0]).unwrap());
        for (branch, inputs) in [
            (returning_branch, vec![predicate.clone(), dimension(&n_type, 2), vector.clone()]),
            (writing_branch, vec![predicate, dimension(&n_type, 2), TestIrValue::Reference(reference.clone()), vector]),
        ] {
            let direct = context.bind(ConditionOperation::new(), vec![branch.clone(), branch.clone()], &inputs);
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let regions = vec![builder.import_program(branch.clone()), builder.import_program(branch.clone())];
            let input_atoms = std::iter::once(ArrayType::scalar(DataType::Boolean).into())
                .chain(branch.input_types())
                .map(|r#type| builder.add_input(r#type))
                .collect::<Vec<_>>();
            let outputs = builder
                .add_instruction(TestIrOperation::Condition(ConditionOperation::new()), regions, input_atoms, None)
                .unwrap()
                .to_vec();
            let output_count = outputs.len();
            let replayed = builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                    outputs,
                    vec![Placeholder; inputs.len()],
                    vec![Placeholder; output_count],
                )
                .unwrap()
                .interpret(inputs.clone());
            for result in [direct, replayed] {
                assert!(
                    matches!(
                        &result,
                        Err(ProgramError::Type(error))
                            if error.downcast_custom::<DimensionError>()
                                == Some(&DimensionError::InputDimensionMismatch {
                                    dimension: "n".to_string(), expected: 2, actual: 3,
                                }),
                    ),
                    "{result:?}",
                );
            }
            assert_eq!(reference.read(), Ok(Array::vector(vec![4f32, 5.0, 6.0]).unwrap()));
        }
    }

    #[test]
    fn test_condition_interpretation_keeps_branch_local_allocations_inside_the_branch() {
        // Only the taken branch allocates and reads its local reference, and that allocation never leaves the branch,
        // so both predicates interpret to the input value.
        let array_type = ArrayType::new_static(DataType::F32, [2]);

        let mut true_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_input = true_builder.add_input(array_type.clone().into());
        let true_reference = true_builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![true_input], None)
            .unwrap()[0];
        let true_output = true_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![true_reference], None)
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![true_output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut false_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let false_input = false_builder.add_input(array_type.clone().into());
        let false_branch = false_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![false_input], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(array_type.clone().into());
        let output = builder
            .add_instruction(
                TestIrOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let value = TestIrValue::Array(Array::vector(vec![2.0f32, 4.0]).unwrap());
        assert_eq!(
            program.interpret(vec![TestIrValue::Array(Array::scalar(true).unwrap()), value.clone()]),
            Ok(vec![value.clone()]),
        );
        assert_eq!(
            program.interpret(vec![TestIrValue::Array(Array::scalar(false).unwrap()), value.clone()]),
            Ok(vec![value]),
        );
    }

    #[test]
    fn test_condition_interpretation_passes_caller_references_into_branches() {
        // A reference allocated before the condition enters each branch as an input, so both branches read the
        // caller's allocation, through both program interpretation and region interpretation in an explicit context.
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let reference_type = ReferenceType::new(array_type.clone());
        let build_branch = || {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            let output =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let true_branch = build_branch();
        let false_branch = build_branch();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(array_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let output = builder
            .add_instruction(
                TestIrOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let context = TestEagerContext::new();
        let value = TestIrValue::Array(Array::vector(vec![2.0f32, 4.0]).unwrap());
        for predicate in [true, false] {
            assert_eq!(
                program.interpret(vec![TestIrValue::Array(Array::scalar(predicate).unwrap()), value.clone()]),
                Ok(vec![value.clone()]),
            );
            assert_eq!(
                program.entry_region_ref().interpret_in_context(
                    &context,
                    vec![TestIrValue::Array(Array::scalar(predicate).unwrap()), value.clone()],
                ),
                Ok(vec![value.clone()]),
            );
        }
    }

    #[test]
    fn test_condition_interpretation_forwards_dimension_outputs() {
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let input = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
        let regions = vec![
            builder.import_region(scale_branch(extent_type.clone(), 2.0).entry_region_ref()),
            builder.import_region(scale_branch(extent_type.clone(), 3.0).entry_region_ref()),
        ];
        let outputs = builder
            .add_instruction(
                TestIrOperation::Condition(ConditionOperation::new()),
                regions,
                vec![predicate, extent, input],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder.build(outputs, vec![Placeholder; 3], vec![Placeholder; 2]).unwrap();

        // A dimension carried through a condition is an ordinary structural value: it appears in both branch interfaces
        // and in the array IR output signature exactly like the array beside it.
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:dimension<extent ∈ [1, 8)>, %2:f64[] .
                let %3:dimension<extent ∈ [1, 8)>, %4:f64[] = condition %0 %1 %2 [
                    true={
                        lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[] .
                        let %2:f64[] = const 2.0
                            %3:f64[] = mul %1 %2
                        in (%0, %3)
                    },
                    false={
                        lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[] .
                        let %2:f64[] = const 3.0
                            %3:f64[] = mul %1 %2
                        in (%0, %3)
                    },
                ]
                in (%3, %4)"},
        );

        // Eager interpretation selects one branch per predicate value and forwards the same dimension either way.
        let boolean =
            |value: bool| array(Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[value]).unwrap());
        assert_eq!(
            program.interpret(vec![boolean(true), dimension(&extent_type, 4), array(Array::scalar(5.0).unwrap())]),
            Ok(vec![dimension(&extent_type, 4), array(Array::scalar(10.0).unwrap())]),
        );
        assert_eq!(
            program.interpret(vec![boolean(false), dimension(&extent_type, 4), array(Array::scalar(5.0).unwrap())]),
            Ok(vec![dimension(&extent_type, 4), array(Array::scalar(15.0).unwrap())]),
        );

        // Relocating the array IR program imports both branch regions unchanged, so it renders and executes exactly
        // like its source. Program splicing is owned by `programs`; no test in this module owns it.
        let mut relocated_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let relocated_inputs = vec![
            relocated_builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean))),
            relocated_builder.add_input(ArrayIrType::Dimension(extent_type.clone())),
            relocated_builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64))),
        ];
        let relocated_outputs = relocated_builder.splice_program(&program, &relocated_inputs).unwrap();
        let relocated = relocated_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(relocated_outputs, vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(relocated.to_string(), program.to_string());
        assert_eq!(
            relocated.interpret(vec![boolean(true), dimension(&extent_type, 4), array(Array::scalar(5.0).unwrap())]),
            Ok(vec![dimension(&extent_type, 4), array(Array::scalar(10.0).unwrap())]),
        );
    }

    #[test]
    fn test_condition_interpretation_preserves_mixed_dimension_constraints() {
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![extent_type.dimension().clone()]));

        // A symbolic first-class dimension and a static array sharing its extent are a valid refinement that admits
        // no structural identity substitution. Staging keeps both branches declared instead of replaying them, which
        // would erase the relationship between the two inputs, and the refined output still comes from inference.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        builder.add_input(extent_type.clone().into());
        let vector = builder.add_input(vector_type.clone().into());
        let identity_branch = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![vector], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let context = TracingContext::<TestIrValue, TestIrOperation>::new();
        let inputs = [
            context.input(ArrayType::scalar(DataType::Boolean).into()),
            context.input(extent_type.clone().into()),
            context.input(ArrayType::new_static(DataType::F32, [3]).into()),
        ];
        let outputs = context
            .bind(ConditionOperation::new(), vec![identity_branch.clone(), identity_branch], &inputs)
            .unwrap();
        assert_eq!(outputs[0].r#type().into_owned(), ArrayType::new_static(DataType::F32, [3]).into());
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![outputs[0].atom_id().unwrap()],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:dimension<extent ∈ [1, 8)>, %2:f32[3] .
                let %3:f32[3] = condition %0 %1 %2 [
                    true={
                        lambda %0:dimension<extent ∈ [1, 8)>, %1:f32[extent] .
                        in (%1)
                    },
                    false={
                        lambda %0:dimension<extent ∈ [1, 8)>, %1:f32[extent] .
                        in (%1)
                    },
                ]
                in (%3)"},
        );
        assert_eq!(
            program.interpret(vec![
                array(Array::scalar(true).unwrap()),
                dimension(&extent_type, 3),
                array(Array::vector(vec![1f32, 2.0, 3.0]).unwrap()),
            ]),
            Ok(vec![array(Array::vector(vec![1f32, 2.0, 3.0]).unwrap())]),
        );

        // A branch that squares its vector input keeps the same contract under a renamed dimension identity, and its
        // retained declared branches validate the dimension extent at runtime entry.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        builder.add_input(extent_type.clone().into());
        let input = builder.add_input(vector_type.into());
        let output = builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::Mul(MulOperation::new())),
                Vec::new(),
                vec![input, input],
                None,
            )
            .unwrap()[0];
        let branch = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let expected_rendering = |caller_extent: &str| {
            formatdoc! {"
                lambda %0:bool[], %1:dimension<{caller_extent} ∈ [1, 8)>, %2:f32[3] .
                let %3:f32[3] = condition %0 %1 %2 [
                    true={{
                        lambda %0:dimension<extent ∈ [1, 8)>, %1:f32[extent] .
                        let %2:f32[extent] = mul %1 %1
                        in (%2)
                    }},
                    false={{
                        lambda %0:dimension<extent ∈ [1, 8)>, %1:f32[extent] .
                        let %2:f32[extent] = mul %1 %1
                        in (%2)
                    }},
                ]
                in (%3)"}
        };
        let input = array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let expected = vec![array(Array::vector(vec![1.0f32, 4.0, 9.0]).unwrap())];
        let is_extent_mismatch = |result: &Result<Vec<TestIrValue>, ProgramError>| {
            matches!(
                result,
                Err(ProgramError::Type(error))
                    if error.downcast_custom::<DimensionError>()
                        == Some(&DimensionError::InputDimensionMismatch {
                            dimension: "extent".to_string(),
                            expected: 4,
                            actual: 3,
                        }),
            )
        };
        let eager = TestEagerContext::new();

        // The primitive accepts a broad dimension together with a concrete array, but one structural identity
        // renaming cannot preserve their shared extent. Retain the declared branches, including after a caller's
        // dimension definition changes identity, while ordinary output inference still produces a concrete shape.
        let context = TracingContext::<TestIrValue, TestIrOperation>::new();
        let inputs = [
            context.input(ArrayType::scalar(DataType::Boolean).into()),
            context.input(extent_type.clone().into()),
            context.input(ArrayType::new_static(DataType::F32, [3]).into()),
        ];
        let outputs = context.bind(ConditionOperation::new(), vec![branch.clone(), branch.clone()], &inputs).unwrap();
        let output_ids = outputs.iter().map(|output| output.atom_id().unwrap()).collect();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(output_ids, vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        assert_eq!(program.to_string(), expected_rendering("extent"));

        // The retained declared branches validate the exact type of the dimension value at runtime entry, so the
        // application succeeds only when its extent agrees with the array's extent.
        for predicate in [true, false] {
            let inputs = vec![array(Array::scalar(predicate).unwrap()), dimension(&extent_type, 3), input.clone()];
            assert_eq!(
                eager.bind(ConditionOperation::new(), vec![branch.clone(), branch.clone()], &inputs),
                Ok(expected.clone()),
            );
            assert_eq!(program.interpret(inputs), Ok(expected.clone()));
        }

        // A conflicting extent is rejected before either branch runs, in direct binding and in program replay alike,
        // because the static type of the dimension value binds the branch identity to its extent.
        let inputs = vec![array(Array::scalar(true).unwrap()), dimension(&extent_type, 4), input.clone()];
        assert!(is_extent_mismatch(&eager.bind(
            ConditionOperation::new(),
            vec![branch.clone(), branch.clone()],
            &inputs,
        )));
        assert!(is_extent_mismatch(&program.interpret(inputs)));

        // A caller dimension with a renamed identity keeps the same declared branches and runtime validation.
        let renamed_extent_type = DimensionType::new("renamed", extent_type.bounds());
        let context = TracingContext::<TestIrValue, TestIrOperation>::new();
        let inputs = [
            context.input(ArrayType::scalar(DataType::Boolean).into()),
            context.input(renamed_extent_type.clone().into()),
            context.input(ArrayType::new_static(DataType::F32, [3]).into()),
        ];
        let outputs = context.bind(ConditionOperation::new(), vec![branch.clone(), branch.clone()], &inputs).unwrap();
        let output_ids = outputs.iter().map(|output| output.atom_id().unwrap()).collect();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(output_ids, vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        assert_eq!(program.to_string(), expected_rendering("renamed"));
        for predicate in [true, false] {
            let inputs =
                vec![array(Array::scalar(predicate).unwrap()), dimension(&renamed_extent_type, 3), input.clone()];
            assert_eq!(
                eager.bind(ConditionOperation::new(), vec![branch.clone(), branch.clone()], &inputs),
                Ok(expected.clone()),
            );
            assert_eq!(program.interpret(inputs), Ok(expected.clone()));
        }
        let inputs = vec![array(Array::scalar(true).unwrap()), dimension(&renamed_extent_type, 4), input];
        assert!(is_extent_mismatch(&eager.bind(ConditionOperation::new(), vec![branch.clone(), branch], &inputs)));
        assert!(is_extent_mismatch(&program.interpret(inputs)));
    }

    #[test]
    fn test_condition_partial_evaluation() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let branch = |operation: ArrayOperation<Array>| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let known = builder.add_input(scalar_type.clone());
            let unknown = builder.add_input(scalar_type.clone());
            let arguments =
                if matches!(operation, ArrayOperation::Add(_)) { vec![unknown, unknown] } else { vec![unknown] };
            let output = builder.add_instruction(operation, Vec::new(), arguments, None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![known, output], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let known = builder.add_input(scalar_type.clone());
        let unknown = builder.add_input(scalar_type.clone());
        let true_region = builder.import_program(branch(ArrayOperation::Add(AddOperation::new())));
        let false_region = builder.import_program(branch(ArrayOperation::ZeroLike(ZeroLikeOperation::new())));
        let outputs = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_region, false_region],
                vec![predicate, known, unknown],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();

        // Concrete predicates inline only the selected branch, while the first output remains known.
        let true_evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(true).unwrap()),
                PartialValue::Known(Array::scalar(3.0).unwrap()),
                PartialValue::Unknown(scalar_type.clone()),
            ])
            .unwrap();
        assert_eq!(
            true_evaluation.outputs,
            vec![PartialEvaluationOutput::Known(Array::scalar(3.0).unwrap()), PartialEvaluationOutput::Unknown(0)],
        );
        assert_eq!(
            true_evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = add %0 %0
                in (%1)"},
        );
        assert_eq!(
            true_evaluation.program.interpret(vec![Array::scalar(4.0).unwrap()]),
            Ok(vec![Array::scalar(8.0).unwrap()]),
        );
        let false_evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(false).unwrap()),
                PartialValue::Known(Array::scalar(3.0).unwrap()),
                PartialValue::Unknown(scalar_type.clone()),
            ])
            .unwrap();
        assert_eq!(false_evaluation.outputs, true_evaluation.outputs);
        assert_eq!(
            false_evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = zero_like %0
                in (%1)"},
        );
        assert_eq!(
            false_evaluation.program.interpret(vec![Array::scalar(4.0).unwrap()]),
            Ok(vec![Array::scalar(0.0).unwrap()]),
        );

        // A symbolic known predicate retains one conditional on each side of the split, with the common known
        // output produced by the known side and the arithmetic result produced by the residual side.
        let outer = TracingContext::<Array, ArrayOperation<Array>>::new();
        let symbolic_predicate = outer.input(ArrayType::scalar(DataType::Boolean));
        let symbolic_known = outer.input(scalar_type.clone());
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(symbolic_predicate),
                    PartialValue::Known(symbolic_known),
                    PartialValue::Unknown(scalar_type),
                ],
            )
            .unwrap();
        let [PartialEvaluationOutput::Known(known_output), PartialEvaluationOutput::Unknown(0)] =
            evaluation.outputs.as_slice()
        else {
            panic!("the conditional split did not retain its known and residual outputs");
        };
        let known_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(
                vec![known_output.atom_id().unwrap()],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            known_program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[] .
                let %2:f64[] = condition %0 %1 [
                    true={
                        lambda %0:f64[] .
                        in (%0)
                    },
                    false={
                        lambda %0:f64[] .
                        in (%0)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:bool[] .
                let %2:f64[] = condition %1 %0 [
                    true={
                        lambda %0:f64[] .
                        let %1:f64[] = add %0 %0
                        in (%1)
                    },
                    false={
                        lambda %0:f64[] .
                        let %1:f64[] = zero_like %0
                        in (%1)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(
            known_program.interpret(vec![Array::scalar(true).unwrap(), Array::scalar(3.0).unwrap()]),
            Ok(vec![Array::scalar(3.0).unwrap()]),
        );
        assert_eq!(
            evaluation.program.interpret(vec![Array::scalar(4.0).unwrap(), Array::scalar(true).unwrap()]),
            Ok(vec![Array::scalar(8.0).unwrap()]),
        );
        assert_eq!(
            evaluation.program.interpret(vec![Array::scalar(4.0).unwrap(), Array::scalar(false).unwrap()]),
            Ok(vec![Array::scalar(0.0).unwrap()]),
        );
    }

    #[test]
    fn test_condition_partial_evaluation_folds_symbolic_predicates_with_known_inputs() {
        // A symbolic known predicate cannot select a branch, but when every branch input is known as well the whole
        // condition belongs to the known side: it is bound once in the outer trace and leaves an empty residual.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let true_region = builder.import_program(scalar_scale_branch(2.0));
        let false_region = builder.import_program(scalar_branch(ArrayOperation::Sin(SinOperation::new())));
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(ConditionOperation::new(), vec![true_region, false_region], vec![predicate, input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let outer = TracingContext::<Array, ArrayOperation<Array>>::new();
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(outer.input(ArrayType::scalar(DataType::Boolean))),
                    PartialValue::Known(outer.input(ArrayType::scalar(DataType::F64))),
                ],
            )
            .unwrap();
        let [PartialEvaluationOutput::Known(known_output)] = evaluation.outputs.as_slice() else {
            panic!("the condition did not fold to the known side");
        };
        let known_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(
                vec![known_output.atom_id().unwrap()],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            known_program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[] .
                let %2:f64[] = condition %0 %1 [
                    true={
                        lambda %0:f64[] .
                        let %1:f64[] = const 2.0
                            %2:f64[] = mul %0 %1
                        in (%2)
                    },
                    false={
                        lambda %0:f64[] .
                        let %1:f64[] = sin %0
                        in (%1)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda  .
                in ()"},
        );
        assert_eq!(
            known_program.interpret(vec![Array::scalar(true).unwrap(), Array::scalar(3.0f64).unwrap()]),
            Ok(vec![Array::scalar(6.0f64).unwrap()]),
        );
        assert_eq!(
            known_program.interpret(vec![Array::scalar(false).unwrap(), Array::scalar(3.0f64).unwrap()]),
            Ok(vec![Array::scalar(3.0f64.sin()).unwrap()]),
        );
    }

    #[test]
    fn test_condition_partial_evaluation_compares_refined_branch_outputs() {
        // A concrete predicate selects a branch only after the branch outputs, refined by the actual inputs, agree: an
        // `f64[3]` input folds the selected identity or constant branch, while an `f64[4]` input is rejected for either
        // predicate. An unknown predicate keeps both branches, specialized to the refined `f64[3]` input.
        let input = Array::vector(vec![1.0f64; 3]).unwrap();
        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let predicate = context.lift(Array::scalar(false).unwrap()).unwrap();
        let input_tracer = context.lift(input.clone()).unwrap();
        let outputs = context
            .bind(ConditionOperation::new(), refinable_dynamic_branches(), &[predicate, input_tracer])
            .unwrap();
        let PartialValue::Known(output) = outputs[0].value().unwrap().value() else {
            panic!("expected a known output")
        };
        assert_eq!(output, &Array::vector(vec![2.0f64; 3]).unwrap());
        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let predicate = context.lift(Array::scalar(true).unwrap()).unwrap();
        let input_tracer = context.lift(input.clone()).unwrap();
        let outputs = context
            .bind(ConditionOperation::new(), refinable_dynamic_branches(), &[predicate, input_tracer])
            .unwrap();
        let PartialValue::Known(output) = outputs[0].value().unwrap().value() else {
            panic!("expected a known output")
        };
        assert_eq!(output, &input);

        for predicate in [false, true] {
            let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
            let predicate = context.lift(Array::scalar(predicate).unwrap()).unwrap();
            let input = context.lift(Array::vector(vec![1.0f64; 4]).unwrap()).unwrap();
            assert_eq!(
                context
                    .bind(ConditionOperation::new(), refinable_dynamic_branches(), &[predicate, input])
                    .map(|_| ()),
                Err(TypeError::invalid(
                    "`condition` branch output type signature mismatch: expected [f64[4]] but got [f64[3]]",
                )
                .into()),
            );
        }

        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let predicate =
            PartialTracer::new(context.clone(), context.unknown_input(ArrayType::scalar(DataType::Boolean), 0));
        let input =
            PartialTracer::new(context.clone(), context.unknown_input(ArrayType::new_static(DataType::F64, [3]), 1));
        let outputs =
            context.bind(ConditionOperation::new(), refinable_dynamic_branches(), &[predicate, input]).unwrap();
        let evaluation = finish_partial_evaluation(context, Vec::new(), outputs);
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
            lambda %0:bool[], %1:f64[3] .
            let %2:f64[3] = condition %0 %1 [
                true={
                    lambda %0:f64[3] .
                    in (%0)
                },
                false={
                    lambda %0:f64[3] .
                    let %1:f64[3] = const [2.0, 2.0, 2.0]
                    in (%1)
                },
            ]
            in (%2)"},
        );
    }

    #[test]
    fn test_condition_partial_evaluation_validates_direct_branch_contracts() {
        let operation = ConditionOperation::<ArrayType>::new();
        let scalar_branch = scalar_branch(ArrayOperation::Add(AddOperation::new()));
        let boolean_branch = boolean_branch();
        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let predicate = context.lift(Array::scalar(true).unwrap()).unwrap();
        let input = context.lift(Array::scalar(4.0f64).unwrap()).unwrap();
        assert_eq!(
            context.bind(operation, vec![scalar_branch.clone(), boolean_branch.clone()], &[predicate, input]),
            Err(TypeError::invalid(
                "`condition` branch output type signature mismatch: expected [f64[]] but got [bool[]]",
            )
            .into()),
        );

        // Symbolic known predicates and unknown branch inputs must reject the original boundary before partitioning.
        let outer = TracingContext::<Array, ArrayOperation<Array>>::new();
        let context = PartialEvaluationContext::new(outer.clone());
        let predicate = PartialTracer::new(
            context.clone(),
            PartialEvaluationValue::known_input(outer.input(ArrayType::scalar(DataType::Boolean))),
        );
        let input = PartialTracer::new(context.clone(), context.unknown_input(ArrayType::scalar(DataType::F64), 0));
        assert!(matches!(
            context.bind(operation, vec![scalar_branch, boolean_branch], &[predicate, input]),
            Err(ProgramError::Type(error))
                if error == TypeError::invalid(
                    "`condition` branch output type signature mismatch: expected [f64[]] but got [bool[]]",
                ),
        ));
        assert!(outer.builder().borrow().instructions().is_empty());

        // Empty and unit-result boundaries still validate predicate presence and shape before specialization.
        let unit_branch = ProgramBuilder::<Array, ArrayOperation<Array>>::new()
            .build::<Vec<Array>, Vec<Array>>(Vec::new(), Vec::new(), Vec::new())
            .unwrap();
        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        assert_eq!(
            context.bind(operation, vec![unit_branch.clone(), unit_branch.clone()], &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let predicate = context.lift(Array::vector(vec![true]).unwrap()).unwrap();
        assert_eq!(
            context.bind(operation, vec![unit_branch.clone(), unit_branch.clone()], &[predicate]),
            Err(TypeError::invalid("`condition` predicate type must be a scalar boolean, but got `bool[1]`").into()),
        );
        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let predicate = context.lift(Array::scalar(1.0f32).unwrap()).unwrap();
        assert_eq!(
            context.bind(operation, vec![unit_branch.clone(), unit_branch], &[predicate]),
            Err(TypeError::invalid("`condition` predicate type must be a scalar boolean, but got `f32[]`").into()),
        );

        // An invalid peer must be diagnosed before a selected zero-output branch executes its failing assertion.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let output = builder.add_constant(Array::scalar(1.0f64).unwrap());
        let value_branch = builder.build(vec![output], Vec::new(), vec![Placeholder]).unwrap();
        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let predicate = context.lift(Array::scalar(true).unwrap()).unwrap();
        assert_eq!(
            context.bind(operation, vec![failing_unit_branch(), value_branch], &[predicate]),
            Err(TypeError::invalid("`condition` branch output type signature mismatch: expected [] but got [f64[]]")
                .into()),
        );
    }

    #[test]
    fn test_condition_partial_evaluation_rejects_branch_dependent_reference_outputs() {
        // Partial evaluation validates the reference contracts of both branches before it selects one for a concrete
        // predicate or splits them around a symbolic one, so it rejects exactly the conditions that interpretation and
        // discharge reject, whichever branch the predicate would select.
        let program = branch_dependent_reference_program();
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let expected = Err(ProgramError::Type(TypeError::invalid(
            "`condition` reference output 0 must denote the same reference in both branches, but the `true` branch \
             returns region input 0 and the `false` branch returns region input 1",
        )));
        for predicate in [true, false] {
            let evaluation = program.partially_evaluate(&[
                PartialValue::Known(array(Array::scalar(predicate).unwrap())),
                PartialValue::Unknown(reference_type.clone()),
                PartialValue::Unknown(reference_type.clone()),
            ]);
            assert_eq!(evaluation.map(|_| ()), expected);
        }
        let outer = TracingContext::<TestIrValue, TestIrOperation>::new();
        let evaluation = program.partially_evaluate_in_context(
            &outer,
            &[
                PartialValue::Known(outer.input(ArrayType::scalar(DataType::Boolean).into())),
                PartialValue::Unknown(reference_type.clone()),
                PartialValue::Unknown(reference_type),
            ],
        );
        assert_eq!(evaluation.map(|_| ()), expected);
        assert!(outer.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_condition_partial_evaluation_rejects_mutations_under_varying_predicates() {
        // Partial evaluation validates the mutation rule of both branches under a varying predicate in the same way,
        // whichever branch a concrete predicate would select: only the `true` branch mutates the reference, whose
        // referent does not vary over the predicate's axis.
        let program = manual_mutation_program(false);
        let (referent_type, predicate_type) = manual_mutation_types(false);
        let (value_type, _) = manual_mutation_types(true);
        let reference_type = ArrayIrType::Reference(ReferenceType::new(referent_type));
        let expected = Err(ProgramError::Type(TypeError::invalid(
            "`condition` `true` branch mutates region input 0 of type \
             `ref<f32[2][sharding={mesh<['x'=2:manual]>, [{}]}]>`, whose referent does not vary over every manual \
             axis that the predicate `bool[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]` varies \
             over, so devices that take different branches would leave it different across devices; vary the \
             referent over those axes",
        )));
        for predicate in [true, false] {
            let evaluation = program.partially_evaluate(&[
                PartialValue::Known(array(Array::from_elements(predicate_type.clone(), &[predicate]).unwrap())),
                PartialValue::Unknown(reference_type.clone()),
                PartialValue::Unknown(value_type.clone().into()),
            ]);
            assert_eq!(evaluation.map(|_| ()), expected);
        }
        let outer = TracingContext::<TestIrValue, TestIrOperation>::new();
        let evaluation = program.partially_evaluate_in_context(
            &outer,
            &[
                PartialValue::Known(outer.input(predicate_type.into())),
                PartialValue::Unknown(reference_type),
                PartialValue::Unknown(value_type.into()),
            ],
        );
        assert_eq!(evaluation.map(|_| ()), expected);
        assert!(outer.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_condition_partial_evaluation_refines_outputs_from_live_reference_facts() {
        // The `true` branch reads its first `ref<f64[extent]>` input and the `false` branch returns a constant
        // `f64[3]`. Under a concrete predicate and executed reference placement, the live handles' referent facts
        // refine the branch outputs exactly as eager binding does: a three-element referent makes the branches agree,
        // a four-element referent makes them disagree, and handles with conflicting extents are rejected. No branch
        // writes, so every handle keeps its initial referent.
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let reference_type =
            ReferenceType::new(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)])));
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = builder.add_input(reference_type.clone().into());
        builder.add_input(reference_type.clone().into());
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let true_branch = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        builder.add_input(reference_type.clone().into());
        builder.add_input(reference_type.into());
        let output = builder.add_constant(array(Array::vector(vec![2.0f64; 3]).unwrap()));
        let false_branch = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let branches = [true_branch, false_branch];
        let expected_error =
            TypeError::invalid("`condition` branch output type signature mismatch: expected [f64[4]] but got [f64[3]]");
        let initial = Array::vector(vec![1.0f64; 3]).unwrap();
        for (predicate, expected) in [(false, Array::vector(vec![2.0f64; 3]).unwrap()), (true, initial.clone())] {
            let reference = ArrayReference::new(initial.clone());
            let expected = array(expected);
            let eager = TestEagerContext::new();
            let values = [
                array(Array::scalar(predicate).unwrap()),
                TestIrValue::Reference(reference.clone()),
                TestIrValue::Reference(reference.clone()),
            ];
            assert_eq!(eager.bind(ConditionOperation::new(), branches.clone(), &values), Ok(vec![expected.clone()]));
            let context = PartialEvaluationContext::new(eager).with_reference_placement(ReferencePlacement::Execute);
            let inputs = values.map(|value| context.lift(value).unwrap());
            let outputs = context.bind(ConditionOperation::new(), branches.clone(), &inputs).unwrap();
            assert_eq!(outputs[0].value().unwrap().as_known(), Some(&expected));
            assert_eq!(reference.read(), Ok(initial.clone()));
        }

        let initial = Array::vector(vec![1.0f64; 4]).unwrap();
        for predicate in [false, true] {
            let reference = ArrayReference::new(initial.clone());
            let eager = TestEagerContext::new();
            let values = [
                array(Array::scalar(predicate).unwrap()),
                TestIrValue::Reference(reference.clone()),
                TestIrValue::Reference(reference.clone()),
            ];
            assert_eq!(
                eager.bind(ConditionOperation::new(), branches.clone(), &values),
                Err(ProgramError::Type(expected_error.clone())),
            );
            let context = PartialEvaluationContext::new(eager).with_reference_placement(ReferencePlacement::Execute);
            let inputs = values.map(|value| context.lift(value).unwrap());
            assert_eq!(
                context.bind(ConditionOperation::new(), branches.clone(), &inputs).map(|_| ()),
                Err(ProgramError::Type(expected_error.clone())),
            );
            assert_eq!(reference.read(), Ok(initial.clone()));
        }
        let context = PartialEvaluationContext::new(TestEagerContext::new());
        let first = ArrayReference::new(Array::vector(vec![1.0f64; 3]).unwrap());
        let second = ArrayReference::new(Array::vector(vec![1.0f64; 4]).unwrap());
        let inputs = [
            context.lift(array(Array::scalar(false).unwrap())).unwrap(),
            context.lift(TestIrValue::Reference(first.clone())).unwrap(),
            context.lift(TestIrValue::Reference(second.clone())).unwrap(),
        ];
        assert!(matches!(
            context.bind(ConditionOperation::new(), branches, &inputs),
            Err(ProgramError::Type(error))
                if error.downcast_custom::<DimensionError>() == Some(&DimensionError::InputDimensionMismatch {
                    dimension: "extent".to_string(),
                    expected: 3,
                    actual: 4,
                }),
        ));
        assert_eq!(first.read(), Ok(Array::vector(vec![1.0f64; 3]).unwrap()));
        assert_eq!(second.read(), Ok(Array::vector(vec![1.0f64; 4]).unwrap()));
    }

    #[test]
    fn test_condition_partial_evaluation_refines_known_eager_reference_shape() {
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(vector_type));
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let destination = builder.add_input(reference_type.clone());
        let source = builder.add_input(reference_type.clone());
        let value = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![source], None).unwrap()[0];
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![destination, value], None)
            .unwrap();
        let branch = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![destination], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            branch.to_string(),
            indoc! {"
                lambda %0:ref<f32[extent]>, %1:ref<f32[extent]> .
                let %2:f32[extent] = reference_read %1
                    () = reference_write %0 %2
                in (%0)"},
        );

        // Known eager handles have concrete referent types that refine the shared symbolic extent. The selected
        // branch executes once, writes the caller's destination, and returns that same handle by identity.
        let context = PartialEvaluationContext::new(TestEagerContext::new())
            .with_reference_placement(ReferencePlacement::Execute);
        let destination = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let source = ArrayReference::new(Array::vector(vec![4.0f32, 5.0, 6.0]).unwrap());
        let inputs = [
            context.lift(array(Array::scalar(true).unwrap())).unwrap(),
            context.lift(TestIrValue::Reference(destination.clone())).unwrap(),
            context.lift(TestIrValue::Reference(source)).unwrap(),
        ];
        let outputs = context.bind(ConditionOperation::new(), vec![branch.clone(), branch.clone()], &inputs).unwrap();
        assert_eq!(outputs[0].value().unwrap().as_known(), Some(&TestIrValue::Reference(destination.clone())));
        assert_eq!(destination.read(), Ok(Array::vector(vec![4.0f32, 5.0, 6.0]).unwrap()));

        // Each handle independently satisfies its declared bounds, but their common identity requires equal extents.
        // Reject that conflict before executing the selected write.
        let context = PartialEvaluationContext::new(TestEagerContext::new())
            .with_reference_placement(ReferencePlacement::Execute);
        let destination = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let source = ArrayReference::new(Array::vector(vec![4.0f32, 5.0]).unwrap());
        let inputs = [
            context.lift(array(Array::scalar(true).unwrap())).unwrap(),
            context.lift(TestIrValue::Reference(destination.clone())).unwrap(),
            context.lift(TestIrValue::Reference(source)).unwrap(),
        ];
        assert!(matches!(
            context.bind(ConditionOperation::new(), vec![branch.clone(), branch.clone()], &inputs),
            Err(ProgramError::Type(error))
                if error.downcast_custom::<DimensionError>() == Some(&DimensionError::InputDimensionMismatch {
                    dimension: "extent".to_string(),
                    expected: 3,
                    actual: 2,
                }),
        ));
        assert_eq!(destination.read(), Ok(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()));

        // An unknown reference describes a staged allocation, which refines the primitive boundary by its referent
        // exactly like a known handle does, so its static extent fixes the shared symbolic extent of the residual
        // condition.
        let context = PartialEvaluationContext::new(TestEagerContext::new());
        let static_reference_type =
            ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])));
        let inputs = [
            PartialTracer::new(context.clone(), context.unknown_input(ArrayType::scalar(DataType::Boolean).into(), 0)),
            PartialTracer::new(context.clone(), context.unknown_input(static_reference_type.clone(), 1)),
            PartialTracer::new(context.clone(), context.unknown_input(static_reference_type, 2)),
        ];
        let outputs = context.bind(ConditionOperation::new(), vec![branch.clone(), branch], &inputs).unwrap();
        let evaluation = finish_partial_evaluation(context, inputs, outputs);
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
            lambda %0:bool[], %1:ref<f32[3]>, %2:ref<f32[3]> .
            let %3:ref<f32[3]> = condition %0 %1 %2 [
                true={
                    lambda %0:ref<f32[3]>, %1:ref<f32[3]> .
                    let %2:f32[3] = reference_read %1
                        () = reference_write %0 %2
                    in (%0)
                },
                false={
                    lambda %0:ref<f32[3]>, %1:ref<f32[3]> .
                    let %2:f32[3] = reference_read %1
                        () = reference_write %0 %2
                    in (%0)
                },
            ]
            in (%3)"},
        );
    }

    #[test]
    fn test_condition_partial_evaluation_preserves_mixed_dimension_reference_contracts() {
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::new(
            DataType::F32,
            Shape::new(vec![extent_type.dimension().clone()]),
        )));
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        builder.add_input(extent_type.clone().into());
        let reference = builder.add_input(reference_type);
        let value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, value], None)
            .unwrap();
        let branch = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        for placement in [ReferencePlacement::Stage, ReferencePlacement::Execute] {
            let reference = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
            let context = PartialEvaluationContext::new(TestEagerContext::new()).with_reference_placement(placement);
            let inputs = [
                PartialTracer::new(
                    context.clone(),
                    context.unknown_input(ArrayType::scalar(DataType::Boolean).into(), 0),
                ),
                PartialTracer::new(context.clone(), PartialEvaluationValue::known_input(dimension(&extent_type, 3))),
                PartialTracer::new(
                    context.clone(),
                    PartialEvaluationValue::known_input(TestIrValue::Reference(reference.clone())),
                ),
            ];

            // The known first-class dimension is typed exactly, like the concrete handle, so the residual condition
            // specializes both branches consistently to them without executing either branch's additive write.
            let outputs =
                context.bind(ConditionOperation::new(), vec![branch.clone(), branch.clone()], &inputs).unwrap();
            let evaluation = finish_partial_evaluation(context, inputs, outputs);
            assert_eq!(
                evaluation.program().to_string(),
                indoc! {"
                    lambda %0:bool[], %1:dimension<3>, %2:ref<f32[3]> .
                    let %3:ref<f32[3]> = condition %0 %1 %2 [
                        true={
                            lambda %0:dimension<3>, %1:ref<f32[3]> .
                            let %2:f32[3] = reference_read %1
                                () = reference_add_update %1 %2
                            in (%1)
                        },
                        false={
                            lambda %0:dimension<3>, %1:ref<f32[3]> .
                            let %2:f32[3] = reference_read %1
                                () = reference_add_update %1 %2
                            in (%1)
                        },
                    ]
                    in (%3)"},
            );
            assert_eq!(reference.read(), Ok(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()));

            // Exact dimension metadata conflicts with the referent extent even before residual emission. An unknown
            // predicate never permits either branch's additive write to execute during this failed preparation.
            let context = PartialEvaluationContext::new(TestEagerContext::new()).with_reference_placement(placement);
            let inputs = [
                PartialTracer::new(
                    context.clone(),
                    context.unknown_input(ArrayType::scalar(DataType::Boolean).into(), 0),
                ),
                PartialTracer::new(
                    context.clone(),
                    PartialEvaluationValue::known_input(TestIrValue::Dimension(DimensionValue::constant(4).unwrap())),
                ),
                PartialTracer::new(
                    context.clone(),
                    PartialEvaluationValue::known_input(TestIrValue::Reference(reference.clone())),
                ),
            ];
            assert!(matches!(
                context.bind(ConditionOperation::new(), vec![branch.clone(), branch.clone()], &inputs),
                Err(ProgramError::Type(error))
                    if error.downcast_custom::<DimensionError>() == Some(&DimensionError::InputDimensionMismatch {
                        dimension: "extent".to_string(),
                        expected: 4,
                        actual: 3,
                    }),
            ));
            assert_eq!(reference.read(), Ok(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()));
        }
    }

    #[test]
    fn test_condition_partial_evaluation_specializes_residual_known_eager_reference_shape() {
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(extent)]),
        )));
        let mut true_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let destination = true_builder.add_input(reference_type.clone());
        let source = true_builder.add_input(reference_type.clone());
        let value =
            true_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![source], None).unwrap()[0];
        true_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![destination, value], None)
            .unwrap();
        let true_branch = true_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![destination], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut false_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let destination = false_builder.add_input(reference_type.clone());
        let source = false_builder.add_input(reference_type);
        let value = false_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![destination], None)
            .unwrap()[0];
        false_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![source, value], None)
            .unwrap();
        let false_branch = false_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![destination], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        for placement in [ReferencePlacement::Stage, ReferencePlacement::Execute] {
            let context = PartialEvaluationContext::new(TestEagerContext::new()).with_reference_placement(placement);
            let destination = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
            let source = ArrayReference::new(Array::vector(vec![4.0f32, 5.0, 6.0]).unwrap());
            let inputs = [
                PartialTracer::new(
                    context.clone(),
                    context.unknown_input(ArrayType::scalar(DataType::Boolean).into(), 0),
                ),
                PartialTracer::new(
                    context.clone(),
                    PartialEvaluationValue::known_input(TestIrValue::Reference(destination.clone())),
                ),
                PartialTracer::new(
                    context.clone(),
                    PartialEvaluationValue::known_input(TestIrValue::Reference(source.clone())),
                ),
            ];
            let outputs = context
                .bind(ConditionOperation::new(), vec![true_branch.clone(), false_branch.clone()], &inputs)
                .unwrap();
            let evaluation = finish_partial_evaluation(context, inputs, outputs);
            assert_eq!(evaluation.known_reference_inputs().collect::<Vec<_>>(), vec![1, 2]);
            assert_eq!(
                evaluation.program().to_string(),
                indoc! {"
                    lambda %0:bool[], %1:ref<f32[3]>, %2:ref<f32[3]> .
                    let %3:ref<f32[3]> = condition %0 %1 %2 [
                        true={
                            lambda %0:ref<f32[3]>, %1:ref<f32[3]> .
                            let %2:f32[3] = reference_read %1
                                () = reference_add_update %0 %2
                            in (%0)
                        },
                        false={
                            lambda %0:ref<f32[3]>, %1:ref<f32[3]> .
                            let %2:f32[3] = reference_read %0
                                () = reference_add_update %1 %2
                            in (%0)
                        },
                    ]
                    in (%3)"},
            );

            // Discharge requires value-only public outputs. Retain the conditional effects while omitting its
            // forwarded handle; specialization and discharge must leave both live allocations untouched.
            let discharged = evaluation.program().with_outputs(&[]).unwrap().discharge_references(0).unwrap();
            assert_eq!(
                discharged.external_reference_bindings(),
                &[
                    ExternalReferenceBinding::new(ReferenceSource::Input { index: 1 }, Some(0)),
                    ExternalReferenceBinding::new(ReferenceSource::Input { index: 2 }, Some(1)),
                ],
            );
            assert_eq!(destination.read(), Ok(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()));
            assert_eq!(source.read(), Ok(Array::vector(vec![4.0f32, 5.0, 6.0]).unwrap()));

            // Additive writes expose both duplicate execution and speculative execution of the peer branch. The
            // second replay observes the first replay's destination through the original known handle.
            let eager = TestEagerContext::new();
            assert_eq!(
                evaluation.interpret(&eager, &[array(Array::scalar(true).unwrap())]),
                Ok(vec![TestIrValue::Reference(destination.clone())]),
            );
            assert_eq!(destination.read(), Ok(Array::vector(vec![5.0f32, 7.0, 9.0]).unwrap()));
            assert_eq!(source.read(), Ok(Array::vector(vec![4.0f32, 5.0, 6.0]).unwrap()));
            assert_eq!(
                evaluation.interpret(&eager, &[array(Array::scalar(false).unwrap())]),
                Ok(vec![TestIrValue::Reference(destination.clone())]),
            );
            assert_eq!(destination.read(), Ok(Array::vector(vec![5.0f32, 7.0, 9.0]).unwrap()));
            assert_eq!(source.read(), Ok(Array::vector(vec![9.0f32, 12.0, 15.0]).unwrap()));

            // Specialization must not weaken the shared extent constraint, even with an unknown predicate and
            // handles whose individual concrete shapes satisfy the declared bounds.
            let context = PartialEvaluationContext::new(TestEagerContext::new()).with_reference_placement(placement);
            let destination = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
            let source = ArrayReference::new(Array::vector(vec![4.0f32, 5.0]).unwrap());
            let inputs = [
                PartialTracer::new(
                    context.clone(),
                    context.unknown_input(ArrayType::scalar(DataType::Boolean).into(), 0),
                ),
                PartialTracer::new(
                    context.clone(),
                    PartialEvaluationValue::known_input(TestIrValue::Reference(destination.clone())),
                ),
                PartialTracer::new(
                    context.clone(),
                    PartialEvaluationValue::known_input(TestIrValue::Reference(source.clone())),
                ),
            ];
            assert!(matches!(
                context.bind(ConditionOperation::new(), vec![true_branch.clone(), false_branch.clone()], &inputs),
                Err(ProgramError::Type(error))
                    if error.downcast_custom::<DimensionError>() == Some(&DimensionError::InputDimensionMismatch {
                        dimension: "extent".to_string(),
                        expected: 3,
                        actual: 2,
                    }),
            ));
            assert_eq!(destination.read(), Ok(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()));
            assert_eq!(source.read(), Ok(Array::vector(vec![4.0f32, 5.0]).unwrap()));
        }
    }

    #[test]
    fn test_condition_partial_evaluation_inlines_selected_branch_with_entry_relationships() {
        // A known predicate selects the branch, which is inlined through its guarded entry: a relationship the input
        // types prove needs no guard, and an unproven one stays as a residual guard. Both branch inputs share `n`, and
        // the branch reads only the unknown second input.
        let n = DimensionVariable::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(n)])).into();
        let branch = {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            builder.add_input(vector_type.clone());
            let input = builder.add_input(vector_type.clone());
            let output = builder
                .add_instruction(
                    TestIrOperation::Array(ArrayOperation::from(MulOperation::new())),
                    Vec::new(),
                    vec![input, input],
                    None,
                )
                .unwrap()[0];
            builder.build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
        }
        .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let first = builder.add_input(vector_type.clone());
        let second = builder.add_input(vector_type.clone());
        let true_branch = builder.import_program(branch.clone());
        let false_branch = builder.import_program(branch);
        let output = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, first, second],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        // A known symbolic `f32[n]` next to the unknown `f32[n]` proves their relationship through their types, so
        // the selected branch is inlined.
        let outer = TracingContext::<TestIrValue, TestIrOperation>::new();
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(outer.lift(array(Array::scalar(true).unwrap())).unwrap()),
                    PartialValue::Known(outer.input(vector_type.clone())),
                    PartialValue::Unknown(vector_type.clone()),
                ],
            )
            .unwrap();
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f32[n], %1:f32[n] .
                let %2:f32[n] = mul %0 %0
                in (%2)"},
        );

        // A known exact `f32[3]` root input proves that the unknown root input is an `f32[3]`. The residual entry keeps
        // the supplied `f32[n]`, and its entry guard checks the relationship and narrows the unknown input, so the
        // selected branch is inlined too.
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(array(Array::scalar(true).unwrap())),
                PartialValue::Known(array(Array::vector(vec![1f32; 3]).unwrap())),
                PartialValue::Unknown(vector_type.clone()),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {r#"
                lambda %0:f32[n], %1:f32[3] .
                let %2:f32[3], %3:f32[3] = guard [boundary="partial evaluation", declared_types=[f32[n], f32[n]]] %1 %0
                    %4:f32[3] = mul %3 %3
                in (%4)"#},
        );
        let context = TestEagerContext::new();
        assert_eq!(
            evaluation.interpret(&context, &[array(Array::vector(vec![2f32; 3]).unwrap())]),
            Ok(vec![array(Array::vector(vec![4f32; 3]).unwrap())]),
        );
        assert_eq!(
            evaluation.interpret(&context, &[array(Array::vector(vec![2f32; 4]).unwrap())]),
            program.interpret(vec![
                array(Array::scalar(true).unwrap()),
                array(Array::vector(vec![1f32; 3]).unwrap()),
                array(Array::vector(vec![2f32; 4]).unwrap()),
            ]),
        );
        assert!(matches!(
            evaluation.interpret(&context, &[array(Array::vector(vec![2f32; 4]).unwrap())]),
            Err(ProgramError::Type(error))
                if error.downcast_custom::<DimensionError>() == Some(&DimensionError::InputDimensionMismatch {
                    dimension: "n".to_string(),
                    expected: 3,
                    actual: 4,
                }),
        ));
    }

    #[test]
    fn test_condition_partial_evaluation_inlines_selected_branch_for_computed_extents() {
        // The extent `1 + 1` is computed inside the program, so no root entry establishes it, and the array that the
        // branch reads is a computed unknown whose residual type stays symbolic. The guarded entry of the inlined
        // selected branch checks that the array's runtime extent is 2 and narrows the array before the branch uses it.
        let n = DimensionVariable::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let dimension_type = DimensionType::from(n.clone());
        let vector_type: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(n)])).into();
        let branch = {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            builder.add_input(dimension_type.clone().into());
            let input = builder.add_input(vector_type.clone());
            let output = builder
                .add_instruction(
                    TestIrOperation::Array(ArrayOperation::from(MulOperation::new())),
                    Vec::new(),
                    vec![input, input],
                    None,
                )
                .unwrap()[0];
            builder.build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
        }
        .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let input = builder.add_input(vector_type.clone());
        let doubled = builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(AddOperation::new())),
                Vec::new(),
                vec![input, input],
                None,
            )
            .unwrap()[0];
        let one_type = DimensionType::from(Dimension::Static(1));
        let one = builder.add_constant(TestIrValue::Dimension(DimensionValue::new(one_type.clone(), 1).unwrap()));
        let extent = builder
            .add_instruction(
                DimensionAddOperation::new(&one_type, &one_type).unwrap(),
                Vec::new(),
                vec![one, one],
                None,
            )
            .unwrap()[0];
        let predicate = builder.add_constant(array(Array::scalar(true).unwrap()));
        let true_branch = builder.import_program(branch.clone());
        let false_branch = builder.import_program(branch);
        let output = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, extent, doubled],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let evaluation = program.partially_evaluate(&[PartialValue::Unknown(vector_type.clone())]).unwrap();
        let guard = r#"guard [boundary="partial evaluation", declared_types=[dimension<n ∈ [1, 8)>, f32[n]]]"#;
        assert_eq!(
            evaluation.program.to_string(),
            formatdoc! {"
                lambda %0:f32[n] .
                let %1:dimension<2> = const 2
                    %2:f32[n] = add %0 %0
                    %3:dimension<2>, %4:f32[2] = {guard} %1 %2
                    %5:f32[2] = mul %4 %4
                in (%5)"},
        );
        let context = TestEagerContext::new();
        assert_eq!(
            evaluation.interpret(&context, &[array(Array::vector(vec![1f32; 2]).unwrap())]),
            Ok(vec![array(Array::vector(vec![4f32; 2]).unwrap())]),
        );
        let mismatch = Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "n".to_string(),
            expected: 2,
            actual: 3,
        })));
        assert_eq!(program.interpret(vec![array(Array::vector(vec![1f32; 3]).unwrap())]), mismatch);
        assert_eq!(evaluation.interpret(&context, &[array(Array::vector(vec![1f32; 3]).unwrap())]), mismatch);

        // When the selected branch writes the array through a reference whose type keeps `n`, the guarded entry checks
        // the computed extent against both and retains `n` instead of narrowing the array, and the residual program
        // rejects a mismatching array before the inlined write, exactly like the source program.
        let ArrayIrType::Array(array_type) = &vector_type else { unreachable!() };
        let reference_type: ArrayIrType = ReferenceType::new(array_type.clone()).into();
        let branch = {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            builder.add_input(dimension_type.into());
            let input = builder.add_input(vector_type.clone());
            let reference = builder.add_input(reference_type.clone());
            builder
                .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, input], None)
                .unwrap();
            builder.build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder; 3], Vec::new())
        }
        .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let input = builder.add_input(vector_type.clone());
        let reference = builder.add_input(reference_type.clone());
        let doubled = builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(AddOperation::new())),
                Vec::new(),
                vec![input, input],
                None,
            )
            .unwrap()[0];
        let one = builder.add_constant(TestIrValue::Dimension(DimensionValue::new(one_type.clone(), 1).unwrap()));
        let extent = builder
            .add_instruction(
                DimensionAddOperation::new(&one_type, &one_type).unwrap(),
                Vec::new(),
                vec![one, one],
                None,
            )
            .unwrap()[0];
        let predicate = builder.add_constant(array(Array::scalar(true).unwrap()));
        let true_branch = builder.import_program(branch.clone());
        let false_branch = builder.import_program(branch);
        builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, extent, doubled, reference],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
            .unwrap();
        let evaluation = program
            .partially_evaluate(&[PartialValue::Unknown(vector_type), PartialValue::Unknown(reference_type)])
            .unwrap();
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {r#"
                lambda %0:f32[n], %1:ref<f32[n]> .
                let %2:dimension<2> = const 2
                    %3:f32[n] = add %0 %0
                    %4:dimension<n ∈ [1, 8)>, %5:f32[n] = guard [
                        boundary="partial evaluation",
                        declared_types=[dimension<n ∈ [1, 8)>, f32[n], ref<f32[n]>],
                        retained_identities=[n],
                    ] %2 %3 %1
                    () = reference_write %1 %5
                in ()"#},
        );
        let reference = ArrayReference::new(Array::vector(vec![1f32; 2]).unwrap());
        let inputs = vec![array(Array::vector(vec![4f32; 2]).unwrap()), TestIrValue::Reference(reference.clone())];
        assert_eq!(evaluation.interpret(&context, &inputs), Ok(Vec::new()));
        assert_eq!(reference.read(), Ok(Array::vector(vec![8f32; 2]).unwrap()));
        let reference = ArrayReference::new(Array::vector(vec![1f32; 3]).unwrap());
        let inputs = vec![array(Array::vector(vec![4f32; 3]).unwrap()), TestIrValue::Reference(reference.clone())];
        assert_eq!(program.interpret(inputs.clone()), mismatch);
        assert_eq!(evaluation.interpret(&context, &inputs), mismatch);
        assert_eq!(reference.read(), Ok(Array::vector(vec![1f32; 3]).unwrap()));
    }

    #[test]
    fn test_condition_partial_evaluation_inlines_selected_branch_with_known_inputs() {
        // The branch squares an array, writes the square through a reference, and returns it. All three of its inputs
        // share the symbolic `extent` identity.
        let extent = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.dimension().clone()]));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(vector_type.clone()));
        let branch = {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            builder.add_input(extent.clone().into());
            let input = builder.add_input(vector_type.into());
            let reference = builder.add_input(reference_type);
            let output = builder
                .add_instruction(
                    TestIrOperation::Array(ArrayOperation::from(MulOperation::new())),
                    Vec::new(),
                    vec![input, input],
                    None,
                )
                .unwrap()[0];
            builder
                .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, output], None)
                .unwrap();
            builder.build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
        }
        .unwrap();

        // Under an eager parent, the exact types of the known inputs prove every entry relationship, so the selected
        // branch is inlined without a guard and folds entirely on the known side: its write through the known
        // reference executes during partial evaluation, and the residual program keeps no `condition`, `guard`, or
        // reference instruction.
        let context = PartialEvaluationContext::new(TestEagerContext::new())
            .with_reference_placement(ReferencePlacement::Execute);
        let reference = ArrayReference::new(Array::vector(vec![0f32; 2]).unwrap());
        let inputs = [
            context.lift(array(Array::scalar(true).unwrap())).unwrap(),
            context.lift(dimension(&extent, 2)).unwrap(),
            context.lift(array(Array::vector(vec![2f32, 3.0]).unwrap())).unwrap(),
            context.lift(TestIrValue::Reference(reference.clone())).unwrap(),
        ];
        let outputs = context.bind(ConditionOperation::new(), vec![branch.clone(), branch.clone()], &inputs).unwrap();
        let evaluation = finish_partial_evaluation(context, inputs, outputs);
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda  .
                in ()"},
        );
        assert_eq!(
            evaluation.outputs(),
            [PartialEvaluationOutput::Known(array(Array::vector(vec![4f32, 9.0]).unwrap()))],
        );
        assert_eq!(reference.read(), Ok(Array::vector(vec![4f32, 9.0]).unwrap()));

        // Known inputs that violate the shared `extent` relationship are rejected during partial evaluation, before
        // the selected write executes, instead of being deferred to a residual entry guard.
        let context = PartialEvaluationContext::new(TestEagerContext::new())
            .with_reference_placement(ReferencePlacement::Execute);
        let reference = ArrayReference::new(Array::vector(vec![0f32; 2]).unwrap());
        let inputs = [
            context.lift(array(Array::scalar(true).unwrap())).unwrap(),
            context.lift(dimension(&extent, 2)).unwrap(),
            context.lift(array(Array::vector(vec![2f32, 3.0, 4.0]).unwrap())).unwrap(),
            context.lift(TestIrValue::Reference(reference.clone())).unwrap(),
        ];
        assert_eq!(
            context.bind(ConditionOperation::new(), vec![branch.clone(), branch], &inputs).map(|_| ()),
            Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "extent".to_string(),
                expected: 2,
                actual: 3,
            }))),
        );
        assert_eq!(reference.read(), Ok(Array::vector(vec![0f32; 2]).unwrap()));
    }

    #[test]
    fn test_condition_partial_evaluation_shares_known_references_with_inlined_branches() {
        // A known live reference that enters an inlined selected branch reaches the residual program once, at its exact
        // type, whether the handle is also used by a later instruction, by a second condition, or by an earlier
        // instruction. The guarded entry of each inlined branch checks the unknown array against the reference before
        // its write, and the predicate folds away.
        let dimension = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![dimension.dimension().clone()]));
        let reference_type = ReferenceType::new(array_type.clone());
        let mut branch = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        branch.add_input(dimension.clone().into());
        let array = branch.add_input(array_type.clone().into());
        let reference = branch.add_input(reference_type.clone().into());
        branch
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, array], None)
            .unwrap();
        let branch = branch
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder; 3], Vec::new())
            .unwrap();

        // The source program runs the condition over all four inputs and then reads the reference.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let extent = builder.add_input(dimension.clone().into());
        let array = builder.add_input(array_type.clone().into());
        let reference = builder.add_input(reference_type.into());
        let true_branch = builder.import_program(branch.clone());
        let false_branch = builder.import_program(branch.clone());
        builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_branch, false_branch],
                vec![predicate, extent, array, reference],
                None,
            )
            .unwrap();
        let read =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![read], vec![Placeholder; 4], vec![Placeholder])
            .unwrap();

        // Bind the same instructions directly with a known predicate, extent, and reference, and an unknown array.
        let context = TestEagerContext::new();
        let reference = ArrayReference::new(Array::vector(vec![1f32; 2]).unwrap());
        let predicate = TestIrValue::Array(Array::scalar(true).unwrap());
        let extent = TestIrValue::Dimension(DimensionValue::new(dimension.clone(), 2).unwrap());
        let direct = PartialEvaluationContext::new(context.clone()).with_reference_placement(ReferencePlacement::Stage);
        let known_reference =
            PartialTracer::new(direct.clone(), direct.known_value(TestIrValue::Reference(reference.clone())));
        let inputs = vec![
            PartialTracer::new(direct.clone(), direct.known_value(predicate.clone())),
            PartialTracer::new(direct.clone(), direct.known_value(extent.clone())),
            PartialTracer::new(direct.clone(), direct.unknown_input(array_type.clone().into(), 0)),
            known_reference.clone(),
        ];
        assert_eq!(
            direct.bind(ConditionOperation::new(), vec![branch.clone(), branch.clone()], &inputs),
            Ok(Vec::new()),
        );
        let read = direct.bind(ReferenceReadOperation::new(), Vec::new(), &[known_reference]).unwrap();
        let evaluation = finish_partial_evaluation(direct, inputs, read);
        assert_eq!(evaluation.known_reference_inputs().collect::<Vec<_>>(), vec![1]);
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {r#"
                lambda %0:f32[extent], %1:ref<f32[2]> .
                let %2:dimension<2> = const 2
                    %3:dimension<2>, %4:f32[2] = guard [
                        boundary="partial evaluation",
                        declared_types=[dimension<extent ∈ [1, 8)>, f32[extent], ref<f32[extent]>],
                    ] %2 %0 %1
                    () = reference_write %1 %4
                    %5:f32[2] = reference_read %1
                in (%5)"#},
        );

        // The residual program rejects a mismatching array before writing and otherwise writes and reads the handle
        // exactly like the source program does with an equal handle.
        let mismatching = TestIrValue::Array(Array::vector(vec![4f32; 3]).unwrap());
        let source_reference = ArrayReference::new(Array::vector(vec![1f32; 2]).unwrap());
        let mismatch = Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "extent".to_string(),
            expected: 2,
            actual: 3,
        })));
        assert_eq!(
            program.interpret(vec![
                predicate.clone(),
                extent.clone(),
                mismatching.clone(),
                TestIrValue::Reference(source_reference.clone()),
            ]),
            mismatch,
        );
        assert_eq!(evaluation.interpret(&context, std::slice::from_ref(&mismatching)), mismatch);
        assert_eq!(reference.read(), source_reference.read());
        assert_eq!(reference.read().unwrap(), Array::vector(vec![1f32; 2]).unwrap());
        let matching = TestIrValue::Array(Array::vector(vec![4f32; 2]).unwrap());
        assert_eq!(
            evaluation.interpret(&context, std::slice::from_ref(&matching)),
            program.interpret(vec![predicate, extent, matching, TestIrValue::Reference(source_reference.clone())]),
        );
        assert_eq!(reference.read(), source_reference.read());
        assert_eq!(reference.read().unwrap(), Array::vector(vec![4f32; 2]).unwrap());

        // A known live reference that enters two inlined selected branches reaches the residual program once, at its
        // exact type, and the residual program behaves like eagerly binding both conditions.
        let condition_inputs = |direct: &PartialEvaluationContext<_>, unknown: &PartialTracer<_>, reference| {
            vec![
                PartialTracer::new(
                    direct.clone(),
                    direct.known_value(TestIrValue::Array(Array::scalar(true).unwrap())),
                ),
                PartialTracer::new(
                    direct.clone(),
                    direct.known_value(TestIrValue::Dimension(DimensionValue::new(dimension.clone(), 2).unwrap())),
                ),
                unknown.clone(),
                reference,
            ]
        };
        let eager_inputs = |array: &Array, reference: &ArrayReference<Array>| {
            vec![
                TestIrValue::Array(Array::scalar(true).unwrap()),
                TestIrValue::Dimension(DimensionValue::new(dimension.clone(), 2).unwrap()),
                TestIrValue::Array(array.clone()),
                TestIrValue::Reference(reference.clone()),
            ]
        };
        let reference = ArrayReference::new(Array::vector(vec![1f32; 2]).unwrap());
        let direct = PartialEvaluationContext::new(context.clone()).with_reference_placement(ReferencePlacement::Stage);
        let unknown = PartialTracer::new(direct.clone(), direct.unknown_input(array_type.clone().into(), 0));
        let known_reference =
            PartialTracer::new(direct.clone(), direct.known_value(TestIrValue::Reference(reference.clone())));
        for reference in [known_reference.clone(), known_reference] {
            let inputs = condition_inputs(&direct, &unknown, reference);
            let outputs = direct.bind(ConditionOperation::new(), vec![branch.clone(), branch.clone()], &inputs);
            assert_eq!(outputs, Ok(Vec::new()));
        }
        drop(unknown);
        let evaluation = direct.into_evaluation(Vec::new()).unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {r#"
                lambda %0:f32[extent], %1:ref<f32[2]> .
                let %2:dimension<2> = const 2
                    %3:dimension<2>, %4:f32[2] = guard [
                        boundary="partial evaluation",
                        declared_types=[dimension<extent ∈ [1, 8)>, f32[extent], ref<f32[extent]>],
                    ] %2 %0 %1
                    %5:dimension<2> = const 2
                    () = reference_write %1 %4
                    %6:dimension<2>, %7:f32[2] = guard [
                        boundary="partial evaluation",
                        declared_types=[dimension<extent ∈ [1, 8)>, f32[extent], ref<f32[extent]>],
                    ] %5 %0 %1
                    () = reference_write %1 %7
                in ()"#},
        );
        let mismatching = Array::vector(vec![4f32; 3]).unwrap();
        let source_reference = ArrayReference::new(Array::vector(vec![1f32; 2]).unwrap());
        assert_eq!(
            context.bind(
                ConditionOperation::new(),
                vec![branch.clone(), branch.clone()],
                &eager_inputs(&mismatching, &source_reference),
            ),
            mismatch,
        );
        assert_eq!(evaluation.interpret(&context, &[TestIrValue::Array(mismatching)]), mismatch);
        assert_eq!(reference.read(), source_reference.read());
        let matching = Array::vector(vec![4f32; 2]).unwrap();
        for _ in 0..2 {
            let outputs = context.bind(
                ConditionOperation::new(),
                vec![branch.clone(), branch.clone()],
                &eager_inputs(&matching, &source_reference),
            );
            assert_eq!(outputs, Ok(Vec::new()));
        }
        assert_eq!(evaluation.interpret(&context, &[TestIrValue::Array(matching.clone())]), Ok(Vec::new()));
        assert_eq!(reference.read(), source_reference.read());
        assert_eq!(reference.read().unwrap(), Array::vector(vec![4f32; 2]).unwrap());

        // An earlier use that materializes the handle at its exact type shares that residual input with the inlined
        // selected branch, instead of giving the residual program two aliasing reference inputs.
        let reference = ArrayReference::new(Array::vector(vec![1f32; 2]).unwrap());
        let direct = PartialEvaluationContext::new(context.clone()).with_reference_placement(ReferencePlacement::Stage);
        let unknown = PartialTracer::new(direct.clone(), direct.unknown_input(array_type.into(), 0));
        let known_reference =
            PartialTracer::new(direct.clone(), direct.known_value(TestIrValue::Reference(reference.clone())));
        let read = direct.bind(ReferenceReadOperation::new(), Vec::new(), &[known_reference.clone()]).unwrap();
        let inputs = condition_inputs(&direct, &unknown, known_reference);
        assert_eq!(
            direct.bind(ConditionOperation::new(), vec![branch.clone(), branch.clone()], &inputs),
            Ok(Vec::new()),
        );
        let output = read[0].value().unwrap().clone();
        drop((inputs, read, unknown));
        let evaluation = direct.into_evaluation(vec![output]).unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {r#"
                lambda %0:f32[extent], %1:ref<f32[2]> .
                let %2:f32[2] = reference_read %1
                    %3:dimension<2> = const 2
                    %4:dimension<2>, %5:f32[2] = guard [
                        boundary="partial evaluation",
                        declared_types=[dimension<extent ∈ [1, 8)>, f32[extent], ref<f32[extent]>],
                    ] %3 %0 %1
                    () = reference_write %1 %5
                in (%2)"#},
        );
        let mismatching = Array::vector(vec![4f32; 3]).unwrap();
        let source_reference = ArrayReference::new(Array::vector(vec![1f32; 2]).unwrap());
        let source_read = context.bind(
            ReferenceReadOperation::new(),
            Vec::new(),
            &[TestIrValue::Reference(source_reference.clone())],
        );
        assert_eq!(
            context.bind(
                ConditionOperation::new(),
                vec![branch.clone(), branch.clone()],
                &eager_inputs(&mismatching, &source_reference),
            ),
            mismatch,
        );
        assert_eq!(evaluation.interpret(&context, &[TestIrValue::Array(mismatching)]), mismatch);
        assert_eq!(reference.read(), source_reference.read());
        let matching = Array::vector(vec![4f32; 2]).unwrap();
        assert_eq!(
            context.bind(
                ConditionOperation::new(),
                vec![branch.clone(), branch],
                &eager_inputs(&matching, &source_reference),
            ),
            Ok(Vec::new()),
        );
        assert_eq!(evaluation.interpret(&context, &[TestIrValue::Array(matching)]), source_read);
        assert_eq!(reference.read(), source_reference.read());
    }

    #[test]
    fn test_condition_partial_evaluation_instantiates_asymmetric_known_outputs() {
        // The true branch returns `(sin(known), sin(known) * unknown, known * known, sin(unknown))`, and the false
        // branch returns `(unknown, known * unknown, unknown, unknown)`. The true branch's two instantiated outputs
        // follow its original `sin(known)` feeder edge and are interleaved with unknown outputs in the result.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let mut true_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let known = true_builder.add_input(scalar_type.clone());
        let unknown = true_builder.add_input(scalar_type.clone());
        let [known_sine, unknown_sine] = [known, unknown]
            .map(|input| true_builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0]);
        let scaled = true_builder
            .add_instruction(MulOperation::new(), Vec::new(), vec![known_sine, unknown], None)
            .unwrap()[0];
        let squared =
            true_builder.add_instruction(MulOperation::new(), Vec::new(), vec![known, known], None).unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<Array>, Vec<Array>>(
                vec![known_sine, scaled, squared, unknown_sine],
                vec![Placeholder; 2],
                vec![Placeholder; 4],
            )
            .unwrap();
        let mut false_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let known = false_builder.add_input(scalar_type.clone());
        let unknown = false_builder.add_input(scalar_type.clone());
        let product =
            false_builder.add_instruction(MulOperation::new(), Vec::new(), vec![known, unknown], None).unwrap()[0];
        let false_branch = false_builder
            .build::<Vec<Array>, Vec<Array>>(
                vec![unknown, product, unknown, unknown],
                vec![Placeholder; 2],
                vec![Placeholder; 4],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let predicate = builder.add_input(predicate_type.clone());
        let known = builder.add_input(scalar_type.clone());
        let unknown = builder.add_input(scalar_type.clone());
        let true_region = builder.import_program(true_branch);
        let false_region = builder.import_program(false_branch);
        let outputs = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_region, false_region],
                vec![predicate, known, unknown],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 3], vec![Placeholder; 4])
            .unwrap();

        let outer = TracingContext::<Array, ArrayOperation<Array>>::new();
        let symbolic_predicate = outer.input(predicate_type);
        let symbolic_known = outer.input(scalar_type.clone());
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(symbolic_predicate),
                    PartialValue::Known(symbolic_known),
                    PartialValue::Unknown(scalar_type),
                ],
            )
            .unwrap();
        assert_eq!(
            evaluation.outputs,
            vec![
                PartialEvaluationOutput::Unknown(0),
                PartialEvaluationOutput::Unknown(1),
                PartialEvaluationOutput::Unknown(2),
                PartialEvaluationOutput::Unknown(3),
            ],
        );
        let known_program = known_side_program(&outer, &evaluation, 2);
        assert_eq!(
            known_program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[] .
                let %2:f64[], %3:f64[], %4:f64[], %5:f64[] = condition %0 %1 [
                    true={
                        lambda %0:f64[] .
                        let %1:f64[] = sin %0
                            %2:f64[] = mul %0 %0
                            %3:f64[] = zero [type=f64[]]
                        in (%1, %1, %2, %3)
                    },
                    false={
                        lambda %0:f64[] .
                        let %1:f64[] = zero [type=f64[]]
                            %2:f64[] = zero [type=f64[]]
                            %3:f64[] = zero [type=f64[]]
                        in (%1, %2, %3, %0)
                    },
                ]
                in (%0, %2, %3, %4, %5)"},
        );
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:bool[], %2:f64[], %3:f64[], %4:f64[], %5:f64[] .
                let %6:f64[], %7:f64[], %8:f64[], %9:f64[] = condition %1 %0 %2 %3 %4 %5 [
                    true={
                        lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[] .
                        let %5:f64[] = mul %1 %0
                            %6:f64[] = sin %0
                        in (%2, %5, %3, %6)
                    },
                    false={
                        lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[] .
                        let %5:f64[] = mul %4 %0
                        in (%0, %5, %0, %0)
                    },
                ]
                in (%6, %7, %8, %9)"},
        );

        // Compose the two programs through the reported residual boundary for each runtime predicate. The inactive
        // branch's placeholder edge must never replace the selected branch's real known value.
        for (predicate, expected) in
            [(true, [3.0f64.sin(), 3.0f64.sin() * 4.0, 9.0, 4.0f64.sin()]), (false, [4.0, 12.0, 4.0, 4.0])]
        {
            let arguments =
                vec![Array::scalar(predicate).unwrap(), Array::scalar(3.0f64).unwrap(), Array::scalar(4.0f64).unwrap()];
            let expected = expected.into_iter().map(|value| Array::scalar(value).unwrap()).collect::<Vec<_>>();
            assert_eq!(program.interpret(arguments.clone()), Ok(expected.clone()));
            assert_eq!(interpret_known_side_and_residual(&known_program, &evaluation, arguments, 2), Ok(expected));
        }
    }

    #[test]
    fn test_condition_partial_evaluation_guards_entry_relationships_of_split_branches() {
        // Both branches declare `[f32[n], f32[n]]` and return `(x + x, y)`. A symbolic predicate splits the condition
        // between the known `x: f32[3]` and the unknown `y: f32[n]`, so neither half alone sees that the branch entry
        // requires `n = 3`. The residual program must check that relationship like the source condition does, and
        // its output keeps the `f32[3]` type that the source condition infers.
        let n = DimensionVariable::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(n)])).into();
        let static_type: ArrayIrType = ArrayType::new_static(DataType::F32, [3]).into();
        let predicate_type: ArrayIrType = ArrayType::scalar(DataType::Boolean).into();
        let branch = {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let x = builder.add_input(vector_type.clone());
            let y = builder.add_input(vector_type.clone());
            let doubled = builder
                .add_instruction(
                    TestIrOperation::Array(ArrayOperation::from(AddOperation::new())),
                    Vec::new(),
                    vec![x, x],
                    None,
                )
                .unwrap()[0];
            builder.build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![doubled, y],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
        }
        .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let predicate = builder.add_input(predicate_type.clone());
        let x = builder.add_input(static_type.clone());
        let y = builder.add_input(vector_type.clone());
        let true_branch = builder.import_program(branch.clone());
        let false_branch = builder.import_program(branch);
        let outputs = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, x, y],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(program.output_types(), vec![static_type.clone(), static_type.clone()]);

        let outer = TracingContext::<TestIrValue, TestIrOperation>::new();
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(outer.input(predicate_type)),
                    PartialValue::Known(outer.input(static_type.clone())),
                    PartialValue::Unknown(vector_type),
                ],
            )
            .unwrap();
        let known_program = known_side_program(&outer, &evaluation, 2);
        assert_eq!(
            known_program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[3] .
                let %2:f32[3] = condition %0 %1 [
                    true={
                        lambda %0:f32[3] .
                        let %1:f32[3] = add %0 %0
                        in (%1)
                    },
                    false={
                        lambda %0:f32[3] .
                        let %1:f32[3] = add %0 %0
                        in (%1)
                    },
                ]
                in (%1, %0, %2)"},
        );
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {r#"
                lambda %0:f32[n], %1:f32[3], %2:bool[] .
                let %3:f32[3], %4:f32[3] = guard [boundary="partial evaluation", declared_types=[f32[n], f32[n]]] %1 %0
                    %5:f32[3] = condition %2 %4 [
                        true={
                            lambda %0:f32[3] .
                            in (%0)
                        },
                        false={
                            lambda %0:f32[3] .
                            in (%0)
                        },
                    ]
                in (%5)"#},
        );
        assert_eq!(evaluation.program.output_types(), vec![static_type.clone()]);

        // Composing the known and residual programs accepts a `y` of the length that `x` fixes, and rejects any other
        // length exactly as the source condition does.
        let compose = |y_length: usize| {
            let arguments = vec![
                array(Array::scalar(true).unwrap()),
                array(Array::vector(vec![1f32; 3]).unwrap()),
                array(Array::vector(vec![2f32; y_length]).unwrap()),
            ];
            interpret_known_side_and_residual(&known_program, &evaluation, arguments, 2)
        };
        let source = |y_length: usize| {
            program.interpret(vec![
                array(Array::scalar(true).unwrap()),
                array(Array::vector(vec![1f32; 3]).unwrap()),
                array(Array::vector(vec![2f32; y_length]).unwrap()),
            ])
        };
        assert_eq!(
            compose(3),
            Ok(vec![array(Array::vector(vec![2f32; 3]).unwrap()), array(Array::vector(vec![2f32; 3]).unwrap())]),
        );
        assert_eq!(compose(3), source(3));
        assert!(matches!(
            source(4),
            Err(ProgramError::Type(error))
                if error.downcast_custom::<DimensionError>() == Some(&DimensionError::InputDimensionMismatch {
                    dimension: "n".to_string(),
                    expected: 3,
                    actual: 4,
                }),
        ));
        assert_eq!(compose(4), source(4));
    }

    #[test]
    fn test_condition_partial_evaluation_rejects_forwarded_branch_partitions() {
        // `(a, x) ↦ a · x` with `a` known saves `a`, which forwarding then feeds to the residual branch directly.
        // Splitting assembles one residual signature from the residual edges of both branches, so it rejects the
        // forwarded partitions before binding anything.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let a = builder.add_input(scalar_type.clone());
        let x = builder.add_input(scalar_type.clone());
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![a, x], None).unwrap()[0];
        let branch = builder
            .build::<Vec<Array>, Vec<Array>>(vec![product], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let partition = || branch.partition(&[true, false]).unwrap().forward_residuals().unwrap();
        assert_eq!(
            partition().residual_inputs(),
            &[ResidualInputSource::UnknownInput(1), ResidualInputSource::KnownInput(0)],
        );
        assert_eq!(
            reconstruct_partitioned_condition(
                &ArrayType::scalar(DataType::Boolean),
                &[scalar_type.clone(), scalar_type],
                1,
                &[Array::scalar(true).unwrap(), Array::scalar(3.0).unwrap(), Array::scalar(4.0).unwrap()],
                &[true, false],
                [partition(), partition()],
                |_: &mut ProgramBuilder<Array, ArrayOperation<Array>>, _: &ArrayType, _: &[_]| {
                    panic!("a rejected split builds no placeholders")
                },
                |_, _, _: &[Array]| panic!("a rejected split binds no known condition"),
                |_, _, _: &[Array]| panic!("a rejected split binds no residual condition"),
            ),
            Err(ProgramError::MalformedProgram(
                "`condition` branch partition forwards residual inputs, but splitting branches requires residual edges"
                    .to_string(),
            )),
        );
    }

    #[test]
    fn test_condition_partial_evaluation_keeps_unselected_nonterminating_work_lazy() {
        // The guard fails immediately if specialization probes either branch, so a regression cannot hang this test
        // by interpreting the always-true loop before the unknown predicate selects its safe peer.
        /// Region driver that rejects speculative branch evaluation.
        struct BranchProbeGuard<'o> {
            /// Original branches whose execution must remain conditional.
            branches: &'o Vec<Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>>,
        }

        impl RegionDriver<Array, ArrayOperation<Array>> for BranchProbeGuard<'_> {
            fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, Array, ArrayOperation<Array>>>
            where
                Array: 'r,
                ArrayOperation<Array>: 'r,
            {
                self.branches.regions()
            }
        }

        impl PartialEvaluationDriver<EagerContext<Array, ArrayOperation<Array>>> for BranchProbeGuard<'_> {
            fn partially_evaluate_region(
                &self,
                _context: &PartialEvaluationContext<EagerContext<Array, ArrayOperation<Array>>>,
                _index: usize,
                _inputs: Vec<PartialEvaluationValue<Array>>,
            ) -> Result<Vec<PartialEvaluationValue<Array>>, ProgramError> {
                panic!("an unknown predicate cannot select a branch")
            }

            fn partially_evaluate_program(
                &self,
                _context: &PartialEvaluationContext<EagerContext<Array, ArrayOperation<Array>>>,
                _region: RegionRef<'_, Array, ArrayOperation<Array>>,
                _knowledge: &[PartialValue<Array>],
            ) -> Result<PartialEvaluation<EagerContext<Array, ArrayOperation<Array>>>, ProgramError> {
                panic!("an unknown predicate cannot probe branch execution")
            }

            fn partition_program(
                &self,
                _context: &PartialEvaluationContext<EagerContext<Array, ArrayOperation<Array>>>,
                _region: RegionRef<'_, Array, ArrayOperation<Array>>,
                _input_known: &[bool],
            ) -> Result<PartitionedProgram<Array, ArrayOperation<Array>>, ProgramError> {
                panic!("an unknown predicate cannot partition branch execution")
            }
        }

        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        condition_builder.add_input(scalar_type.clone());
        let predicate = condition_builder.add_constant(Array::scalar(true).unwrap());
        let loop_condition = condition_builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let state = body_builder.add_input(scalar_type.clone());
        let loop_body = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![state], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut branch_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let state = branch_builder.add_input(scalar_type);
        let condition = branch_builder.import_program(loop_condition);
        let body = branch_builder.import_program(loop_body);
        let state = branch_builder
            .add_instruction(WhileOperation::new(), vec![condition, body], vec![state], None)
            .unwrap()[0];
        let nonterminating_branch = branch_builder
            .build::<Vec<Array>, Vec<Array>>(vec![state], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(nonterminating_branch.effects().classes().is_empty());
        assert!(!nonterminating_branch.effects().has_deferred_work());
        let branches = vec![nonterminating_branch, scalar_scale_branch(2.0)];
        let driver = BranchProbeGuard { branches: &branches };
        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let inputs = vec![
            context.unknown_input(ArrayType::scalar(DataType::Boolean), 0),
            PartialEvaluationValue::known_input(Array::scalar(3.0).unwrap()),
        ];
        let outputs = ConditionOperation::new().partially_evaluate(&context, &driver, &inputs).unwrap();
        let evaluation = context.into_evaluation(outputs).unwrap();
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[] .
                let %2:f64[] = condition %0 %1 [
                    true={
                        lambda %0:f64[] .
                        let %1:f64[] = while %0 [
                            condition={
                                lambda %0:f64[] .
                                let %1:bool[] = const true
                                in (%1)
                            },
                            body={
                                lambda %0:f64[] .
                                in (%0)
                            },
                        ]
                        in (%1)
                    },
                    false={
                        lambda %0:f64[] .
                        let %1:f64[] = const 2.0
                            %2:f64[] = mul %0 %1
                        in (%2)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(
            evaluation.interpret(&EagerContext::new(), &[Array::scalar(false).unwrap()]),
            Ok(vec![Array::scalar(6.0).unwrap()]),
        );
    }

    #[test]
    fn test_condition_partial_evaluation_keeps_constant_branch_work_behind_unknown_predicates() {
        // Even literal branch knowledge must not cause constant-only arithmetic to be staged in the outer context:
        // both branch computations belong behind the unknown predicate.
        let branch = |value| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let constant = builder.add_constant(Array::scalar(value).unwrap());
            let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![constant], None).unwrap()[0];
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], Vec::new(), vec![Placeholder]).unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let predicate = builder.add_input(predicate_type.clone());
        let true_region = builder.import_program(branch(0.5f64));
        let false_region = builder.import_program(branch(1.5f64));
        let output = builder
            .add_instruction(ConditionOperation::new(), vec![true_region, false_region], vec![predicate], None)
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let outer = TracingContext::<Array, ArrayOperation<Array>>::new();
        let evaluation =
            program.partially_evaluate_in_context(&outer, &[PartialValue::Unknown(predicate_type)]).unwrap();
        assert!(outer.builder().borrow().instructions().is_empty());
        let outer_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(Vec::new(), Vec::new(), Vec::new())
            .unwrap();
        assert_eq!(
            outer_program.to_string(),
            indoc! {"
            lambda  .
            in ()"},
        );
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:bool[] .
                let %1:f64[] = condition %0 [
                    true={
                        lambda  .
                        let %0:f64[] = const 0.5
                            %1:f64[] = sin %0
                        in (%1)
                    },
                    false={
                        lambda  .
                        let %0:f64[] = const 1.5
                            %1:f64[] = sin %0
                        in (%1)
                    },
                ]
                in (%1)"},
        );
        assert_eq!(
            evaluation.program.interpret(vec![Array::scalar(true).unwrap()]),
            Ok(vec![Array::scalar(0.5f64.sin()).unwrap()]),
        );
        assert_eq!(
            evaluation.program.interpret(vec![Array::scalar(false).unwrap()]),
            Ok(vec![Array::scalar(1.5f64.sin()).unwrap()]),
        );
    }

    #[test]
    fn test_condition_partial_evaluation_preserves_zero_output_residual_work() {
        // A known-symbolic predicate splits known branch results from residual branch work without dropping an
        // effectful residual condition whose branches only print, even though that condition has no data outputs.
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let branch_input_type = ArrayType::scalar(DataType::F64);
        let branch = |label| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(branch_input_type.clone());
            builder.add_instruction(PrintOperation::new(label), Vec::new(), vec![input], None).unwrap();
            let output = builder.add_constant(Array::scalar(1.0).unwrap());
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let true_branch = branch("true");
        let false_branch = branch("false");
        assert!(true_branch.partition(&[false]).unwrap().residual_program().effects().classes().is_ordered());
        assert!(false_branch.partition(&[false]).unwrap().residual_program().effects().classes().is_ordered());

        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(predicate_type.clone());
        let branch_input = builder.add_input(branch_input_type.clone());
        let output = builder
            .add_instruction(
                ArrayOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, branch_input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let outer = TracingContext::<Array, ArrayOperation<Array>>::new();
        let symbolic_predicate = outer.input(predicate_type);
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(symbolic_predicate), PartialValue::Unknown(branch_input_type)],
            )
            .unwrap();

        assert!(matches!(evaluation.outputs.as_slice(), [PartialEvaluationOutput::Known(_)]));
        assert!(evaluation.program.effects().classes().is_ordered());
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:bool[] .
                let () = condition %1 %0 [
                    true={
                        lambda %0:f64[] .
                        let %1:f64[] = print [label=true] %0
                        in ()
                    },
                    false={
                        lambda %0:f64[] .
                        let %1:f64[] = print [label=false] %0
                        in ()
                    },
                ]
                in ()"},
        );

        // A selected backward carrier has no ordinary outputs or execution effects. Its dormant backward rule
        // updates the supplied reference only when transposition resolves the obligation in the selected branch.
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F64)));
        let predicate_type = ArrayIrType::Array(ArrayType::scalar(DataType::Boolean));
        let mut backward_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = backward_builder.add_input(reference_type.clone());
        let update = backward_builder.add_constant(array(Array::scalar(1.0f64).unwrap()));
        backward_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let backward = backward_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        let mut true_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = true_builder.add_input(reference_type.clone());
        let backward_region = true_builder.import_program(backward);
        true_builder
            .add_instruction(
                TestIrOperation::CompositeTranspose(CompositeTransposeOperation::from_backward_region(
                    1,
                    Vec::new(),
                    Vec::new(),
                )),
                vec![backward_region],
                vec![reference],
                None,
            )
            .unwrap();
        let true_branch = true_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        assert!(true_branch.effects().classes().is_empty());
        assert!(true_branch.effects().has_deferred_work());
        let mut false_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        false_builder.add_input(reference_type.clone());
        let false_branch = false_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let predicate = builder.add_input(predicate_type.clone());
        let reference = builder.add_input(reference_type.clone());
        let true_region = builder.import_program(true_branch);
        let false_region = builder.import_program(false_branch);
        builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_region, false_region],
                vec![predicate, reference],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
            .unwrap();

        // Mixed knowledge takes the branch-partitioning path. The known side has no work, and the residual side
        // must retain the unit-returning condition solely for its deferred obligation.
        let outer = TracingContext::<TestIrValue, TestIrOperation>::new();
        let symbolic_predicate = outer.input(predicate_type);
        let predicate_id = symbolic_predicate.atom_id().unwrap();
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(symbolic_predicate), PartialValue::Unknown(reference_type)],
            )
            .unwrap();
        assert_eq!(evaluation.outputs, Vec::new());
        assert!(evaluation.program.effects().classes().is_empty());
        assert!(evaluation.program.effects().has_deferred_work());
        let known_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate_id], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            known_program.to_string(),
            indoc! {"
                lambda %0:bool[] .
                in (%0)"},
        );
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:ref<f64[]>, %1:bool[] .
                let () = condition %1 %0 [
                    true={
                        lambda %0:ref<f64[]> .
                        let () = composite_transpose [leading_input_count=1] %0 [
                            backward={
                                lambda %0:ref<f64[]> .
                                let %1:f64[] = const 1.0
                                    () = reference_add_update %0 %1
                                in ()
                            },
                        ]
                        in ()
                    },
                    false={
                        lambda %0:ref<f64[]> .
                        in ()
                    },
                ]
                in ()"},
        );

        let resolved = evaluation.program.transpose_with_respect_to(&[], &[]).unwrap();
        assert!(!resolved.effects().has_deferred_work());
        assert_eq!(
            resolved.to_string(),
            indoc! {"
                lambda %0:ref<f64[]>, %1:bool[] .
                let () = condition %1 %0 [
                    true={
                        lambda %0:ref<f64[]> .
                        let %1:f64[] = const 1.0
                            () = reference_add_update %0 %1
                        in ()
                    },
                    false={
                        lambda %0:ref<f64[]> .
                        in ()
                    },
                ]
                in ()"},
        );
        for (predicate, expected) in [(false, 4.0f64), (true, 5.0)] {
            let reference = ArrayReference::new(Array::scalar(4.0f64).unwrap());
            let mut known_outputs = known_program.interpret(vec![array(Array::scalar(predicate).unwrap())]).unwrap();
            assert_eq!(
                resolved.interpret(vec![TestIrValue::Reference(reference.clone()), known_outputs.remove(0)]),
                Ok(Vec::new()),
            );
            assert!(known_outputs.is_empty());
            assert_eq!(reference.read(), Ok(Array::scalar(expected).unwrap()));
        }
    }

    #[test]
    fn test_condition_partial_evaluation_keeps_known_side_ordered_effects() {
        // `f(p, k, x) = if p { print(k); x + x } else { x }` with a symbolic known `p`, a known `k`, and an unknown
        // `x`. The known side of the `true` branch prints, so the split must stage that ordered effect exactly once.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let branch = |prints: bool| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let known = builder.add_input(scalar_type.clone());
            let input = builder.add_input(scalar_type.clone());
            let output = if prints {
                builder
                    .add_instruction(ArrayOperation::Print(PrintOperation::new("known")), Vec::new(), vec![known], None)
                    .unwrap();
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0]
            } else {
                input
            };
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let true_region = builder.import_program(branch(true));
        let false_region = builder.import_program(branch(false));
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let known = builder.add_input(scalar_type.clone());
        let input = builder.add_input(scalar_type.clone());
        let output = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_region, false_region],
                vec![predicate, known, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let outer = TracingContext::<Array, ArrayOperation<Array>>::new();
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(outer.input(ArrayType::scalar(DataType::Boolean))),
                    PartialValue::Known(outer.input(scalar_type.clone())),
                    PartialValue::Unknown(scalar_type),
                ],
            )
            .unwrap();
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Unknown(0)]);
        assert!(matches!(
            evaluation.inputs.as_slice(),
            [PartialEvaluationInput::Unknown(2), PartialEvaluationInput::Known(_)],
        ));
        let known_program = known_side_program(&outer, &evaluation, 2);
        assert_eq!(
            known_program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[] .
                let () = condition %0 %1 [
                    true={
                        lambda %0:f64[] .
                        let %1:f64[] = print [label=known] %0
                        in ()
                    },
                    false={
                        lambda %0:f64[] .
                        in ()
                    },
                ]
                in (%0)"},
        );
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:bool[] .
                let %2:f64[] = condition %1 %0 [
                    true={
                        lambda %0:f64[] .
                        let %1:f64[] = add %0 %0
                        in (%1)
                    },
                    false={
                        lambda %0:f64[] .
                        in (%0)
                    },
                ]
                in (%2)"},
        );

        // The known side prints only under a `true` predicate and forwards the predicate to the residual condition.
        for (predicate, expected) in [(true, 8.0f64), (false, 4.0)] {
            let known_outputs = known_program
                .interpret(vec![Array::scalar(predicate).unwrap(), Array::scalar(1.0f64).unwrap()])
                .unwrap();
            assert_eq!(known_outputs, vec![Array::scalar(predicate).unwrap()]);
            assert_eq!(
                evaluation
                    .program
                    .interpret(vec![Array::scalar(4.0f64).unwrap(), Array::scalar(predicate).unwrap()]),
                Ok(vec![Array::scalar(expected).unwrap()]),
            );
        }
    }

    #[test]
    fn test_condition_partial_evaluation_residualizes_whole_when_a_branch_shares_a_reference_root() {
        // A branch whose knownness split would leave one reference root reachable from both sides (here a known
        // reference that the residual side writes through a residual edge) keeps the conditional whole instead of
        // hoisting the known branch work ahead of it, so that the accesses stay in program order and no typed zero is
        // needed for the reference-typed edge slot of the other branch.
        // `f(p, r, x) = if p { write(r, x); x } else { x }`.
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let branch = |writes: bool| {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            let input = builder.add_input(scalar_type.clone().into());
            if writes {
                builder
                    .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, input], None)
                    .unwrap();
            }
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![input], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_region = builder.import_program(branch(true));
        let false_region = builder.import_program(branch(false));
        let predicate = builder.add_input(predicate_type.clone().into());
        let reference = builder.add_input(reference_type.clone().into());
        let input = builder.add_input(scalar_type.clone().into());
        let output = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_region, false_region],
                vec![predicate, reference, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        // A symbolic known predicate cannot select a branch, and the true branch's partition would need the known
        // reference as a residual edge, so the whole condition residualizes with the reference as a residual reference
        // and nothing is hoisted into the outer program.
        let outer = TracingContext::<TestIrValue, TestIrOperation>::new();
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(outer.input(predicate_type.into())),
                    PartialValue::Known(outer.input(reference_type.into())),
                    PartialValue::Unknown(scalar_type.into()),
                ],
            )
            .unwrap();
        assert!(outer.builder().borrow().instructions().is_empty());
        assert!(matches!(evaluation.outputs(), [PartialEvaluationOutput::Unknown(0)]));
        assert_eq!(evaluation.known_reference_inputs().collect::<Vec<_>>(), vec![2]);
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:bool[], %2:ref<f32[]> .
                let %3:f32[] = condition %1 %2 %0 [
                    true={
                        lambda %0:ref<f32[]>, %1:f32[] .
                        let () = reference_write %0 %1
                        in (%1)
                    },
                    false={
                        lambda %0:ref<f32[]>, %1:f32[] .
                        in (%1)
                    },
                ]
                in (%3)"},
        );
    }

    #[test]
    fn test_condition_partial_evaluation_residualizes_whole_for_an_invariant_edge_under_a_varying_predicate() {
        // `f(p, w, y) = if p { parallel_vary(sin(w) * y) } else { parallel_vary(y) }` inside a manual region, with a
        // predicate that varies over `devices` and invariant `w` and `y`. Splitting it around a symbolic known `w`
        // would make the invariant `sin(w)` an output of the known condition, but devices whose predicates differ take
        // different branches and so produce different values for that output. The whole condition residualizes
        // instead and nothing is hoisted into the outer program.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let predicate_type = ArrayIrType::Array(
            ArrayType::scalar(DataType::Boolean)
                .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["devices"]).unwrap())
                .unwrap(),
        );
        let invariant_type =
            ArrayIrType::Array(ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(mesh, 0)).unwrap());
        let branch = |multiplies: bool| {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let weight = builder.add_input(invariant_type.clone());
            let input = builder.add_input(invariant_type.clone());
            let value = if multiplies {
                let sine = builder
                    .add_instruction(
                        TestIrOperation::Array(ArrayOperation::Sin(SinOperation::new())),
                        Vec::new(),
                        vec![weight],
                        None,
                    )
                    .unwrap()[0];
                builder
                    .add_instruction(
                        TestIrOperation::Array(ArrayOperation::Mul(MulOperation::new())),
                        Vec::new(),
                        vec![sine, input],
                        None,
                    )
                    .unwrap()[0]
            } else {
                input
            };
            let output = builder
                .add_instruction(
                    TestIrOperation::Array(ArrayOperation::ParallelVary(ParallelVaryOperation::new(
                        "devices".to_string(),
                    ))),
                    Vec::new(),
                    vec![value],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_region = builder.import_program(branch(true));
        let false_region = builder.import_program(branch(false));
        let predicate = builder.add_input(predicate_type.clone());
        let weight = builder.add_input(invariant_type.clone());
        let input = builder.add_input(invariant_type.clone());
        let output = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_region, false_region],
                vec![predicate, weight, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        let outer = TracingContext::<TestIrValue, TestIrOperation>::new();
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(outer.input(predicate_type)),
                    PartialValue::Known(outer.input(invariant_type.clone())),
                    PartialValue::Unknown(invariant_type),
                ],
            )
            .unwrap();
        assert!(outer.builder().borrow().instructions().is_empty());
        let invariant = "f32[][sharding={mesh<['devices'=2:manual]>, []}]";
        let varying = "f32[][sharding={mesh<['devices'=2:manual]>, [], varying_manual={'devices'}}]";
        let predicate = "bool[][sharding={mesh<['devices'=2:manual]>, [], varying_manual={'devices'}}]";
        assert_eq!(
            evaluation.program().to_string(),
            formatdoc! {"
                lambda %0:{invariant}, %1:{predicate}, %2:{invariant} .
                let %3:{varying} = condition %1 %2 %0 [
                    true={{
                        lambda %0:{invariant}, %1:{invariant} .
                        let %2:{invariant} = sin %0
                            %3:{invariant} = mul %2 %1
                            %4:{varying} = parallel_vary [axis_name=\"devices\"] %3
                        in (%4)
                    }},
                    false={{
                        lambda %0:{invariant}, %1:{invariant} .
                        let %2:{varying} = parallel_vary [axis_name=\"devices\"] %1
                        in (%2)
                    }},
                ]
                in (%3)"},
        );
    }

    #[test]
    fn test_condition_partial_evaluation_splits_varying_edges_under_a_varying_predicate() {
        // `f(p, w, y) = if p { sin(w) * parallel_vary(y) } else { parallel_vary(y) }` inside a manual region, with a
        // predicate and a weight `w` that vary over `devices`. Unlike an invariant edge, the varying `sin(w)` may
        // differ across devices, so the condition splits around the symbolic known `w`.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying_sharding = Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["devices"]).unwrap();
        let predicate_array_type =
            ArrayType::scalar(DataType::Boolean).with_sharding(varying_sharding.clone()).unwrap();
        let varying_array_type = ArrayType::scalar(DataType::F32).with_sharding(varying_sharding).unwrap();
        let predicate_type = ArrayIrType::Array(predicate_array_type.clone());
        let varying_type = ArrayIrType::Array(varying_array_type.clone());
        let invariant_type =
            ArrayIrType::Array(ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(mesh, 0)).unwrap());
        let branch = |multiplies: bool| {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let weight = builder.add_input(varying_type.clone());
            let input = builder.add_input(invariant_type.clone());
            let varied = builder
                .add_instruction(
                    TestIrOperation::Array(ArrayOperation::ParallelVary(ParallelVaryOperation::new(
                        "devices".to_string(),
                    ))),
                    Vec::new(),
                    vec![input],
                    None,
                )
                .unwrap()[0];
            let output = if multiplies {
                let sine = builder
                    .add_instruction(
                        TestIrOperation::Array(ArrayOperation::Sin(SinOperation::new())),
                        Vec::new(),
                        vec![weight],
                        None,
                    )
                    .unwrap()[0];
                builder
                    .add_instruction(
                        TestIrOperation::Array(ArrayOperation::Mul(MulOperation::new())),
                        Vec::new(),
                        vec![sine, varied],
                        None,
                    )
                    .unwrap()[0]
            } else {
                varied
            };
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_region = builder.import_program(branch(true));
        let false_region = builder.import_program(branch(false));
        let predicate = builder.add_input(predicate_type.clone());
        let weight = builder.add_input(varying_type.clone());
        let input = builder.add_input(invariant_type.clone());
        let output = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_region, false_region],
                vec![predicate, weight, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        let outer = TracingContext::<TestIrValue, TestIrOperation>::new();
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(outer.input(predicate_type)),
                    PartialValue::Known(outer.input(varying_type)),
                    PartialValue::Unknown(invariant_type),
                ],
            )
            .unwrap();
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Unknown(0)]);
        assert!(matches!(
            evaluation.inputs.as_slice(),
            [PartialEvaluationInput::Unknown(2), PartialEvaluationInput::Known(_), PartialEvaluationInput::Known(_)],
        ));
        let known_program = known_side_program(&outer, &evaluation, 2);
        let invariant = "f32[][sharding={mesh<['devices'=2:manual]>, []}]";
        let varying = "f32[][sharding={mesh<['devices'=2:manual]>, [], varying_manual={'devices'}}]";
        let predicate = "bool[][sharding={mesh<['devices'=2:manual]>, [], varying_manual={'devices'}}]";
        assert_eq!(
            known_program.to_string(),
            formatdoc! {"
                lambda %0:{predicate}, %1:{varying} .
                let %2:{varying} = condition %0 %1 [
                    true={{
                        lambda %0:{varying} .
                        let %1:{varying} = sin %0
                        in (%1)
                    }},
                    false={{
                        lambda %0:{varying} .
                        let %1:{varying} = zero [
                            type={varying},
                        ]
                        in (%1)
                    }},
                ]
                in (%0, %2)"},
        );
        assert_eq!(
            evaluation.program.to_string(),
            formatdoc! {"
                lambda %0:{invariant}, %1:{predicate}, %2:{varying} .
                let %3:{varying} = condition %1 %0 %2 [
                    true={{
                        lambda %0:{invariant}, %1:{varying} .
                        let %2:{varying} = parallel_vary [axis_name=\"devices\"] %0
                            %3:{varying} = mul %1 %2
                        in (%3)
                    }},
                    false={{
                        lambda %0:{invariant}, %1:{varying} .
                        let %2:{varying} = parallel_vary [axis_name=\"devices\"] %0
                        in (%2)
                    }},
                ]
                in (%3)"},
        );

        // The known side computes the varying edge eagerly (the residual's `parallel_vary` needs an active manual axis,
        // so only the known side is replayed here): `sin(w)` under a `true` predicate and a zero otherwise.
        for (predicate, edge) in [(true, 0.5f32.sin()), (false, 0.0)] {
            let predicate = array(Array::from_elements(predicate_array_type.clone(), &[predicate]).unwrap());
            let weight = array(Array::from_elements(varying_array_type.clone(), &[0.5f32]).unwrap());
            assert_eq!(
                known_program.interpret(vec![predicate.clone(), weight]),
                Ok(vec![predicate, array(Array::from_elements(varying_array_type.clone(), &[edge]).unwrap())]),
            );
        }
    }

    #[test]
    fn test_condition_partial_evaluation_retains_integer_division_by_zero() {
        // An unknown predicate retains known integer division behind its selected branch, including division by zero.
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let branch_input_type = ArrayType::scalar(DataType::I32);
        let divide_branch = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(branch_input_type.clone());
            let one = builder.add_constant(Array::from_elements::<i32>(branch_input_type.clone(), &[1]).unwrap());
            let output = builder
                .add_instruction(ArrayOperation::Div(DivOperation::new()), Vec::new(), vec![one, input], None)
                .unwrap()[0];
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let identity_branch = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(branch_input_type.clone());
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let true_region = builder.import_region(divide_branch.entry_region_ref());
        let false_region = builder.import_region(identity_branch.entry_region_ref());
        let predicate = builder.add_input(predicate_type.clone());
        let branch_input = builder.add_input(branch_input_type.clone());
        let output = builder
            .add_instruction(
                ArrayOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, branch_input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // Known branch inputs remain residual feeders because the unknown predicate has not selected a branch.
        let knowledge = vec![
            PartialValue::Unknown(predicate_type),
            PartialValue::Known(Array::from_elements::<i32>(branch_input_type.clone(), &[0]).unwrap()),
        ];
        let evaluation = program.partially_evaluate(knowledge.as_slice()).unwrap();
        assert!(matches!(evaluation.outputs.as_slice(), [PartialEvaluationOutput::Unknown(0)]));
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:i32[] .
                let %2:i32[] = condition %0 %1 [
                    true={
                        lambda %0:i32[] .
                        let %1:i32[] = const 1
                            %2:i32[] = div %1 %0
                        in (%2)
                    },
                    false={
                        lambda %0:i32[] .
                        in (%0)
                    },
                ]
                in (%2)"},
        );

        // A false predicate selects the identity branch's zero, while a true predicate evaluates division to `-1`.
        let outputs = evaluation
            .interpret(
                &EagerContext::new(),
                &[Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[false]).unwrap()],
            )
            .unwrap();
        assert_eq!(outputs[0].elements::<i32>(), Ok(vec![0]));
        assert_eq!(
            evaluation.program.interpret(vec![Array::scalar(true).unwrap(), Array::scalar(0i32).unwrap()]),
            Ok(vec![Array::scalar(-1i32).unwrap()]),
        );
    }

    #[test]
    fn test_condition_partial_evaluation_specializes_residual_branches() {
        // Concrete predicates inline the selected branch; unknown predicates retain the condition and specialize both
        // residual branches to the concrete input shapes. Both paths preserve the refined output and its values.
        let program = refined_vector_condition_program();
        let vector_type = ArrayIrType::Array(ArrayType::new_static(DataType::F64, [3]));
        assert_eq!(program.output_types(), vec![vector_type.clone()]);
        let arguments = [
            array(Array::scalar(true).unwrap()),
            array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap()),
            array(Array::vector(vec![10.0, 20.0, 30.0]).unwrap()),
        ];
        let expected = vec![array(Array::vector(vec![11.0, 24.0, 39.0]).unwrap())];
        for (partial_inputs, unknown_arguments, expected_program) in [
            (
                vec![
                    PartialValue::Known(arguments[0].clone()),
                    PartialValue::Known(arguments[1].clone()),
                    PartialValue::Unknown(vector_type.clone()),
                ],
                vec![arguments[2].clone()],
                indoc! {"
                    lambda %0:f64[3], %1:f64[3] .
                    let %2:f64[3] = add %1 %0
                    in (%2)"},
            ),
            (
                vec![
                    PartialValue::Unknown(ArrayType::scalar(DataType::Boolean).into()),
                    PartialValue::Known(arguments[1].clone()),
                    PartialValue::Unknown(vector_type.clone()),
                ],
                vec![arguments[0].clone(), arguments[2].clone()],
                indoc! {"
                    lambda %0:bool[], %1:f64[3], %2:f64[3] .
                    let %3:f64[3] = condition %0 %2 %1 [
                        true={
                            lambda %0:f64[3], %1:f64[3] .
                            let %2:f64[3] = mul %0 %0
                                %3:f64[3] = add %2 %1
                            in (%3)
                        },
                        false={
                            lambda %0:f64[3], %1:f64[3] .
                            let %2:f64[3] = add %0 %1
                            in (%2)
                        },
                    ]
                    in (%3)"},
            ),
        ] {
            let evaluation = program.partially_evaluate(&partial_inputs).unwrap();
            assert_eq!(evaluation.program.to_string(), expected_program);
            assert_eq!(evaluation.program.output_types(), vec![vector_type.clone()]);
            assert_eq!(evaluation.interpret(&TestEagerContext::new(), &unknown_arguments), Ok(expected.clone()));
        }

        // A symbolic known predicate cannot select a branch. Partitioning the original symbolic branches produces an
        // identity-bearing residual edge `x * x`, for which no peer-branch placeholder is invented. The condition
        // remains whole; emission specializes its residual branches without hoisting arithmetic into the known side.
        let outer = TracingContext::<TestIrValue, TestIrOperation>::new();
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(outer.input(ArrayType::scalar(DataType::Boolean).into())),
                    PartialValue::Known(outer.input(vector_type.clone())),
                    PartialValue::Unknown(vector_type.clone()),
                ],
            )
            .unwrap();
        assert!(outer.builder().borrow().instructions().is_empty());
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:bool[], %2:f64[3] .
                let %3:f64[3] = condition %1 %2 %0 [
                    true={
                        lambda %0:f64[3], %1:f64[3] .
                        let %2:f64[3] = mul %0 %0
                            %3:f64[3] = add %2 %1
                        in (%3)
                    },
                    false={
                        lambda %0:f64[3], %1:f64[3] .
                        let %2:f64[3] = add %0 %1
                        in (%2)
                    },
                ]
                in (%3)"},
        );
        assert_eq!(evaluation.program.output_types(), vec![vector_type]);
    }

    #[test]
    fn test_condition_partial_evaluation_keeps_condition_whole_when_a_split_folds_no_work() {
        // A symbolic known predicate cannot select a branch, and both branches compute only on the unknown `x`, so
        // their known sides would merely forward the known extent to the residual condition. Such a split folds no
        // work, so the whole condition is retained, without staging either branch's arithmetic in the outer known
        // context.
        let (program, extent_type, input_type) = dynamic_extent_condition_program();
        let outer = TracingContext::<TestIrValue, TestIrOperation>::new();
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(outer.input(ArrayType::scalar(DataType::Boolean).into())),
                    PartialValue::Known(outer.input(extent_type.into())),
                    PartialValue::Unknown(input_type.into()),
                ],
            )
            .unwrap();
        assert!(outer.builder().borrow().instructions().is_empty());
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[extent], %1:dimension<extent ∈ [1, 8)>, %2:bool[] .
                let %3:f64[extent] = condition %2 %1 %0 [
                    true={
                        lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[extent] .
                        let %2:f64[extent] = mul %1 %1
                        in (%2)
                    },
                    false={
                        lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[extent] .
                        let %2:f64[extent] = add %1 %1
                        in (%2)
                    },
                ]
                in (%3)"},
        );
    }

    #[test]
    fn test_condition_partial_evaluation_keeps_dynamic_zero_outputs_whole_under_symbolic_predicates() {
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let program = dynamic_zero_condition_program(&extent_type);

        // A known symbolic predicate cannot select a branch during partial evaluation. Because the dynamic output
        // edge refers to the extent identity, the condition remains whole instead of fabricating an opposite-branch
        // placeholder with arbitrary extents.
        let outer = TracingContext::<TestIrValue, TestIrOperation>::new();
        let symbolic_predicate = outer.input(ArrayType::scalar(DataType::Boolean).into());
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(symbolic_predicate), PartialValue::Unknown(extent_type.clone().into())],
            )
            .unwrap();
        assert!(matches!(evaluation.outputs.as_slice(), [PartialEvaluationOutput::Unknown(0)]));
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:dimension<extent ∈ [1, 8)>, %1:bool[] .
                let %2:f64[extent] = condition %1 %0 [
                    true={
                        lambda %0:dimension<extent ∈ [1, 8)> .
                        let %1:f64[extent] = zero [type=f64[extent]] %0
                        in (%1)
                    },
                    false={
                        lambda %0:dimension<extent ∈ [1, 8)> .
                        let %1:f64[extent] = zero [type=f64[extent]] %0
                        in (%1)
                    },
                ]
                in (%2)"},
        );
    }

    #[test]
    fn test_condition_partial_evaluation_refines_dynamic_branches_with_known_inputs() {
        // `f(p, extent, x, y) = if p { x * x } else { y + y }` over `f64[extent]` vectors. An unknown predicate retains
        // both branches while the known common extent refines the unknown peer and both branch signatures.
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![extent_type.dimension().clone()]));
        let branch = |squares: bool| {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            builder.add_input(extent_type.clone().into());
            let left_input = builder.add_input(vector_type.clone().into());
            let right_input = builder.add_input(vector_type.clone().into());
            let (operation, input) = if squares {
                (TestIrOperation::Array(ArrayOperation::Mul(MulOperation::new())), left_input)
            } else {
                (TestIrOperation::Array(ArrayOperation::Add(AddOperation::new())), right_input)
            };
            let output = builder.add_instruction(operation, Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let extent = builder.add_input(extent_type.clone().into());
        let left_input = builder.add_input(vector_type.clone().into());
        let right_input = builder.add_input(vector_type.clone().into());
        let true_region = builder.import_program(branch(true));
        let false_region = builder.import_program(branch(false));
        let output = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_region, false_region],
                vec![predicate, extent, left_input, right_input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 4], vec![Placeholder])
            .unwrap();

        let left = array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());
        let right = array(Array::vector(vec![10.0, 20.0, 30.0]).unwrap());
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(ArrayType::scalar(DataType::Boolean).into()),
                PartialValue::Known(dimension(&extent_type, 3)),
                PartialValue::Known(left.clone()),
                PartialValue::Unknown(vector_type.clone().into()),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {r#"
            lambda %0:bool[], %1:f64[extent], %2:dimension<3>, %3:f64[3] .
            let %4:dimension<3>, %5:f64[3], %6:f64[3] = guard [
                boundary="partial evaluation",
                declared_types=[dimension<extent ∈ [1, 8)>, f64[extent], f64[extent]],
            ] %2 %3 %1
                %7:f64[3] = condition %0 %2 %3 %6 [
                    true={
                        lambda %0:dimension<3>, %1:f64[3], %2:f64[3] .
                        let %3:f64[3] = mul %1 %1
                        in (%3)
                    },
                    false={
                        lambda %0:dimension<3>, %1:f64[3], %2:f64[3] .
                        let %3:f64[3] = add %2 %2
                        in (%3)
                    },
                ]
            in (%7)"#},
        );
        for (predicate, expected) in [(true, vec![1.0, 4.0, 9.0]), (false, vec![20.0, 40.0, 60.0])] {
            let arguments =
                [array(Array::scalar(predicate).unwrap()), dimension(&extent_type, 3), left.clone(), right.clone()];
            assert_eq!(
                program.interpret(arguments.to_vec()),
                Ok(vec![array(Array::vector(expected.clone()).unwrap())]),
            );
            assert_eq!(
                evaluation.interpret(&TestEagerContext::new(), &[arguments[0].clone(), arguments[3].clone()]),
                Ok(vec![array(Array::vector(expected).unwrap())]),
            );
        }

        // A dimension payload that conflicts with the known concrete array is rejected before any conditional branch is
        // emitted or interpreted, whether its descriptor is broad or a singleton.
        let expected_error = ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "extent".to_string(),
            expected: 4,
            actual: 3,
        }));
        for extent in [dimension(&extent_type, 4), TestIrValue::Dimension(DimensionValue::constant(4).unwrap())] {
            assert_eq!(
                program.interpret(vec![
                    array(Array::scalar(true).unwrap()),
                    extent.clone(),
                    left.clone(),
                    right.clone(),
                ]),
                Err(expected_error.clone()),
            );
            assert_eq!(
                program
                    .partially_evaluate(&[
                        PartialValue::Unknown(ArrayType::scalar(DataType::Boolean).into()),
                        PartialValue::Known(extent),
                        PartialValue::Known(left.clone()),
                        PartialValue::Unknown(vector_type.clone().into()),
                    ])
                    .map(|_| ()),
                Err(expected_error.clone()),
            );
        }
    }

    #[test]
    fn test_condition_batching() {
        // A replicated *abstract* condition predicate under trace-time batching cannot be concretized to pick one
        // branch, so the staged batching rule batches both branch programs at the batch axes of the non-predicate
        // inputs and stages exactly one `condition` operation over them, with the unbatched predicate passed through.
        // Interpreting the staged batched program with both concrete predicate values scales every item by 2 when the
        // predicate is true and by 3 when it is false.
        let parent = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = parent.builder().clone();
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let branch_input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let predicate_atom = builder.borrow_mut().add_input(predicate_type.clone());
        let branch_input_atom = builder.borrow_mut().add_input(branch_input_type);
        let predicate_tracer = parent.tracer(predicate_atom, None);
        let branch_input_tracer = parent.tracer(branch_input_atom, None);
        let output = batch(
            |(predicate, input)| {
                let condition_regions = vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)];
                let condition = ConditionOperation::new();
                let operation = ArrayOperation::Condition(condition);
                let outputs =
                    input.context().bind(operation, condition_regions, &[predicate.clone(), input.clone()])?;
                Ok(outputs.into_iter().next().unwrap())
            },
            (predicate_tracer, branch_input_tracer),
            (BatchAxis::replicated(), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        let output_atom = output.atom_id().unwrap();
        let program = builder
            .borrow()
            .clone()
            .build::<(Array, Array), Array>(vec![output_atom], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[3] .
                let %2:f64[3] = condition %0 %1 [
                    true={
                        lambda %0:f64[3] .
                        let %1:f64[] = const 2.0
                            %2:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %1
                            %3:f64[3] = mul %0 %2
                        in (%3)
                    },
                    false={
                        lambda %0:f64[3] .
                        let %1:f64[] = const 3.0
                            %2:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %1
                            %3:f64[3] = mul %0 %2
                        in (%3)
                    },
                ]
                in (%2)"},
        );
        let truthy = Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[true]).unwrap();
        let falsy = Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[false]).unwrap();
        let branch_input = Array::vector(vec![1.0, 4.0, 9.0]).unwrap();
        assert_eq!(program.interpret((truthy, branch_input.clone())), Ok(Array::vector(vec![2.0, 8.0, 18.0]).unwrap()));
        assert_eq!(program.interpret((falsy, branch_input)), Ok(Array::vector(vec![3.0, 12.0, 27.0]).unwrap()));
    }

    #[test]
    fn test_condition_batching_compares_refined_branch_outputs() {
        // The batched branches compare their outputs after refinement by the batched inputs, whether the predicate is
        // replicated (and stays a condition) or mapped (and becomes a `select`): an `f64[2, 3]` input makes the
        // identity and constant branches agree, while an `f64[2, 4]` input is rejected before any branch is batched.
        for (predicate_type, predicate_axis, expected_program) in [
            (
                ArrayType::scalar(DataType::Boolean),
                BatchAxis::replicated(),
                indoc! {"
                    lambda %0:bool[], %1:f64[2, 3] .
                    let %2:f64[2, 3] = condition %0 %1 [
                        true={
                            lambda %0:f64[2, 3] .
                            in (%0)
                        },
                        false={
                            lambda %0:f64[2, 3] .
                            let %1:f64[3] = const [2.0, 2.0, 2.0]
                                %2:f64[2, 3] = broadcast [output_type=f64[2, 3], output_axes=[1]] %1
                            in (%2)
                        },
                    ]
                    in (%2)"},
            ),
            (
                ArrayType::new_static(DataType::Boolean, [2]),
                BatchAxis::new(0),
                indoc! {"
                    lambda %0:bool[2], %1:f64[2, 3] .
                    let %2:f64[2, 3] = stop_gradient %1
                        %3:bool[2, 3] = broadcast [output_type=bool[2, 3], output_axes=[0]] %0
                        %4:f64[2, 3] = select %3 %1 %2
                        %5:bool[2, 3] = broadcast [output_type=bool[2, 3], output_axes=[0]] %0
                        %6:f64[2, 3] = select %5 %2 %1
                        %7:f64[3] = const [2.0, 2.0, 2.0]
                        %8:bool[2, 3] = broadcast [output_type=bool[2, 3], output_axes=[0]] %0
                        %9:f64[2, 3] = broadcast [output_type=f64[2, 3], output_axes=[1]] %7
                        %10:f64[2, 3] = select %8 %4 %9
                    in (%10)"},
            ),
        ] {
            let parent = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
            let builder = parent.builder().clone();
            let predicate = parent.input(predicate_type);
            let input = parent.input(ArrayType::new_static(DataType::F64, [2, 3]));
            let context = BatchingContext::new(parent, 2);
            let regions = refinable_dynamic_branches().to_vec();
            let driver = CountingBatchingDriver::new(&regions);
            let output = ConditionOperation::new()
                .batch(
                    &context,
                    &driver,
                    &[
                        ArrayBatch::new(predicate, predicate_axis).unwrap(),
                        ArrayBatch::new(input, BatchAxis::new(0)).unwrap(),
                    ],
                )
                .unwrap();
            let program = builder
                .borrow()
                .clone()
                .build::<Vec<Array>, Vec<Array>>(
                    output.iter().map(|output| output.value().atom_id().unwrap()).collect(),
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap();
            assert_eq!(program.to_string(), expected_program);
            assert_eq!(output.iter().map(ArrayBatch::batch_axis).collect::<Vec<_>>(), vec![BatchAxis::new(0)]);
        }

        for (predicate_type, predicate_axis) in [
            (ArrayType::scalar(DataType::Boolean), BatchAxis::replicated()),
            (ArrayType::new_static(DataType::Boolean, [2]), BatchAxis::new(0)),
        ] {
            let parent = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
            let predicate = parent.input(predicate_type);
            let input = parent.input(ArrayType::new_static(DataType::F64, [2, 4]));
            let context = BatchingContext::new(parent, 2);
            let regions = refinable_dynamic_branches().to_vec();
            let driver = CountingBatchingDriver::new(&regions);
            let output = ConditionOperation::new().batch(
                &context,
                &driver,
                &[
                    ArrayBatch::new(predicate, predicate_axis).unwrap(),
                    ArrayBatch::new(input, BatchAxis::new(0)).unwrap(),
                ],
            );
            assert_eq!(
                output.unwrap_err(),
                BatchingError::Program(ProgramError::Type(TypeError::invalid(
                    "`condition` branch output type signature mismatch: expected [f64[4]] but got [f64[3]]",
                ))),
            );
            assert_eq!(driver.batch_program_calls(), 0);
        }
    }

    #[test]
    fn test_condition_batching_array_ir_compares_refined_branch_outputs() {
        // The array IR batching rule compares refined branch outputs like the array rule does, with the batch extent
        // threaded through the batched branches as a first-class dimension.
        for (predicate_type, predicate_axis, expected_program) in [
            (
                ArrayType::scalar(DataType::Boolean),
                BatchAxis::replicated(),
                indoc! {"
                    lambda %0:bool[], %1:f64[2, 3] .
                    let %2:dimension<2> = const 2
                        %3:f64[2, 3] = condition %0 %2 %1 [
                            true={
                                lambda %0:dimension<2>, %1:f64[2, 3] .
                                in (%1)
                            },
                            false={
                                lambda %0:dimension<2>, %1:f64[2, 3] .
                                let %2:f64[3] = const [2.0, 2.0, 2.0]
                                    %3:dimension<3> = constant [value=3]
                                    %4:f64[2, 3] = broadcast [output_axes=[1]] %2 %0 %3
                                in (%4)
                            },
                        ]
                    in (%3)"},
            ),
            (
                ArrayType::new_static(DataType::Boolean, [2]),
                BatchAxis::new(0),
                indoc! {"
                    lambda %0:bool[2], %1:f64[2, 3] .
                    let %2:dimension<2> = const 2
                        %3:f64[2, 3] = stop_gradient %1
                        %4:dimension<3> = constant [value=3]
                        %5:bool[2, 3] = broadcast [output_axes=[0]] %0 %2 %4
                        %6:f64[2, 3] = select %5 %1 %3
                        %7:dimension<3> = constant [value=3]
                        %8:bool[2, 3] = broadcast [output_axes=[0]] %0 %2 %7
                        %9:f64[2, 3] = select %8 %3 %1
                        %10:f64[3] = const [2.0, 2.0, 2.0]
                        %11:dimension<3> = constant [value=3]
                        %12:bool[2, 3] = broadcast [output_axes=[0]] %0 %2 %11
                        %13:dimension<3> = constant [value=3]
                        %14:f64[2, 3] = broadcast [output_axes=[1]] %10 %2 %13
                        %15:f64[2, 3] = select %12 %6 %14
                    in (%15)"},
            ),
        ] {
            let parent = DomainTracingContext::<TestEagerContext>::new();
            let builder = parent.builder().clone();
            let predicate = parent.input(ArrayIrType::Array(predicate_type));
            let input = parent.input(ArrayType::new_static(DataType::F64, [2, 3]).into());
            let batch_extent = parent.lift(TestIrValue::Dimension(DimensionValue::constant(2).unwrap())).unwrap();
            let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent, batch_extent);
            let regions = refinable_dynamic_branches()
                .map(|branch| branch.into_unprojected::<TestIrValue, TestIrOperation>().unwrap())
                .to_vec();
            let driver = CountingBatchingDriver::new(&regions);
            let output = ConditionOperation::new()
                .batch(
                    &context,
                    &driver,
                    &[
                        ArrayIrBatch::new(predicate, predicate_axis).unwrap(),
                        ArrayIrBatch::new(input, BatchAxis::new(0)).unwrap(),
                    ],
                )
                .unwrap();
            let program = builder
                .borrow()
                .clone()
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                    output.iter().map(|output| output.value().atom_id().unwrap()).collect(),
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap();
            assert_eq!(program.to_string(), expected_program);
            assert_eq!(output.iter().map(ArrayIrBatch::batch_axis).collect::<Vec<_>>(), vec![BatchAxis::new(0)]);
        }

        for (predicate_type, predicate_axis) in [
            (ArrayType::scalar(DataType::Boolean), BatchAxis::replicated()),
            (ArrayType::new_static(DataType::Boolean, [2]), BatchAxis::new(0)),
        ] {
            let parent = DomainTracingContext::<TestEagerContext>::new();
            let predicate = parent.input(ArrayIrType::Array(predicate_type));
            let input = parent.input(ArrayType::new_static(DataType::F64, [2, 4]).into());
            let batch_extent = parent.lift(TestIrValue::Dimension(DimensionValue::constant(2).unwrap())).unwrap();
            let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent, batch_extent);
            let regions = refinable_dynamic_branches()
                .map(|branch| branch.into_unprojected::<TestIrValue, TestIrOperation>().unwrap())
                .to_vec();
            let driver = CountingBatchingDriver::new(&regions);
            let output = ConditionOperation::new().batch(
                &context,
                &driver,
                &[
                    ArrayIrBatch::new(predicate, predicate_axis).unwrap(),
                    ArrayIrBatch::new(input, BatchAxis::new(0)).unwrap(),
                ],
            );
            assert_eq!(
                output.unwrap_err(),
                BatchingError::Program(ProgramError::Type(TypeError::invalid(
                    "`condition` branch output type signature mismatch: expected [f64[4]] but got [f64[3]]",
                ))),
            );
            assert_eq!(driver.batch_program_calls(), 0);
        }
    }

    #[test]
    fn test_condition_batching_preserves_mapped_axis_sharding() {
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let batched_sharding =
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap()
                    .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                    .unwrap();
            let batched_type =
                ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
                    .with_sharding(batched_sharding)
                    .unwrap();
            let branch_input = ArrayBatch::new(
                Array::from_elements::<f64>(batched_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
                BatchAxis::new(0),
            )
            .unwrap();
            let unbatched_type = branch_input.unbatched_type();
            let (_, branch) =
                EagerContext::<Array, ArrayOperation<Array>>::trace(|inputs: Vec<_>| Ok(inputs), vec![unbatched_type])
                    .unwrap();
            let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
                .with_axis_sharding(ShardingDimension::sharded(["x"]));
            let predicate = ArrayBatch::replicated(
                Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[true]).unwrap(),
            );

            let outputs = context
                .bind(
                    ArrayOperation::Condition(ConditionOperation::new()),
                    vec![branch.clone(), branch],
                    &[
                        BatchingTracer::new(context.clone(), predicate),
                        BatchingTracer::new(context.clone(), branch_input),
                    ],
                )
                .unwrap();

            assert_eq!(
                outputs.into_iter().map(BatchingTracer::into_batch).collect::<Vec<_>>(),
                vec![
                    ArrayBatch::new(
                        Array::from_elements::<f64>(batched_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
                        BatchAxis::new(0),
                    )
                    .unwrap(),
                ],
            );
        }
    }

    #[test]
    fn test_condition_batching_reuses_naturally_aligned_branch_programs() {
        // The replicated-predicate rule discovers each branch's natural output axes before instantiating both branches
        // at the joined layout. `AlignEachTo` stages axis movement only where a natural axis differs from a mapped
        // target, so a branch whose discovered axes already equal the joined targets keeps its discovery program and
        // the rule performs one structural pass for it instead of two.
        let packed_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let truthy = Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[true]).unwrap();
        let falsy = Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[false]).unwrap();
        let branch_input_values =
            Array::from_elements::<f64>(packed_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Both branches scale the batched input per batch item, so both discover axis 0 and the joined layout equals
        // each branch's discovered layout: the rule batches each branch exactly once.
        let parent = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = parent.builder().clone();
        let predicate_atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::Boolean));
        let branch_input_atom = builder.borrow_mut().add_input(packed_type.clone());
        let predicate = parent.tracer(predicate_atom, None);
        let branch_input = parent.tracer(branch_input_atom, None);
        let context = BatchingContext::new(parent, 2);
        let inputs = vec![ArrayBatch::replicated(predicate), ArrayBatch::new(branch_input, BatchAxis::new(0)).unwrap()];
        let regions = vec![vector_scale_branch(3, 2.0), vector_scale_branch(3, 3.0)];
        let driver = CountingBatchingDriver::new(&regions);
        let outputs = ConditionOperation::new().batch(&context, &driver, inputs.as_slice()).unwrap();
        assert_eq!(driver.batch_program_calls(), 2);
        assert_eq!(outputs.iter().map(ArrayBatch::batch_axis).collect::<Vec<_>>(), vec![BatchAxis::new(0)]);
        let program = builder
            .borrow()
            .clone()
            .build::<(Array, Array), Array>(
                outputs.iter().map(|output| output.value().atom_id().unwrap()).collect(),
                (Placeholder, Placeholder),
                Placeholder,
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[2, 3] .
                let %2:f64[2, 3] = condition %0 %1 [
                    true={
                        lambda %0:f64[2, 3] .
                        let %1:f64[] = const 2.0
                            %2:f64[2, 3] = broadcast [output_type=f64[2, 3], output_axes=[]] %1
                            %3:f64[2, 3] = mul %0 %2
                        in (%3)
                    },
                    false={
                        lambda %0:f64[2, 3] .
                        let %1:f64[] = const 3.0
                            %2:f64[2, 3] = broadcast [output_type=f64[2, 3], output_axes=[]] %1
                            %3:f64[2, 3] = mul %0 %2
                        in (%3)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(
            program.interpret((truthy.clone(), branch_input_values.clone())),
            Ok(Array::matrix(2, 3, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]).unwrap()),
        );
        assert_eq!(
            program.interpret((falsy.clone(), branch_input_values.clone())),
            Ok(Array::matrix(2, 3, vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0]).unwrap()),
        );

        // A replicated false-branch output disagrees with the joined layout, so only that branch is re-batched to
        // broadcast its output across the batch: three structural passes in total.
        let parent = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = parent.builder().clone();
        let predicate_atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::Boolean));
        let branch_input_atom = builder.borrow_mut().add_input(packed_type.clone());
        let predicate = parent.tracer(predicate_atom, None);
        let branch_input = parent.tracer(branch_input_atom, None);
        let context = BatchingContext::new(parent, 2);
        let inputs = vec![ArrayBatch::replicated(predicate), ArrayBatch::new(branch_input, BatchAxis::new(0)).unwrap()];
        let regions = vec![vector_scale_branch(3, 2.0), constant_vector_branch(vec![10.0, 20.0, 30.0])];
        let driver = CountingBatchingDriver::new(&regions);
        let outputs = ConditionOperation::new().batch(&context, &driver, inputs.as_slice()).unwrap();
        assert_eq!(driver.batch_program_calls(), 3);
        assert_eq!(outputs.iter().map(ArrayBatch::batch_axis).collect::<Vec<_>>(), vec![BatchAxis::new(0)]);
        let program = builder
            .borrow()
            .clone()
            .build::<(Array, Array), Array>(
                outputs.iter().map(|output| output.value().atom_id().unwrap()).collect(),
                (Placeholder, Placeholder),
                Placeholder,
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[2, 3] .
                let %2:f64[2, 3] = condition %0 %1 [
                    true={
                        lambda %0:f64[2, 3] .
                        let %1:f64[] = const 2.0
                            %2:f64[2, 3] = broadcast [output_type=f64[2, 3], output_axes=[]] %1
                            %3:f64[2, 3] = mul %0 %2
                        in (%3)
                    },
                    false={
                        lambda %0:f64[2, 3] .
                        let %1:f64[3] = const [10.0, 20.0, 30.0]
                            %2:f64[2, 3] = broadcast [output_type=f64[2, 3], output_axes=[1]] %1
                        in (%2)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(
            program.interpret((truthy, branch_input_values.clone())),
            Ok(Array::matrix(2, 3, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]).unwrap()),
        );

        // The re-batched false branch broadcasts its replicated constant across the batch, so every batch item
        // receives the same constant vector.
        assert_eq!(
            program.interpret((falsy, branch_input_values)),
            Ok(Array::matrix(2, 3, vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0]).unwrap()),
        );

        // The two branches of a staged batched condition may disagree on their natural output batch axes: here the true
        // branch scales the batched input per batch item (axis 0) while the false branch returns a replicated constant
        // (no batch axis). The staged rule normalizes the false branch by appending a broadcast at its tail, so the
        // staged condition stays well-typed and both predicate values interpret correctly per batch item.
        let mut constant_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        constant_builder.add_input(ArrayType::scalar(DataType::F64));
        let constant_output = constant_builder.add_constant(Array::scalar(7.0).unwrap());
        let constant_branch = constant_builder
            .build::<Vec<Array>, Vec<Array>>(vec![constant_output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let parent = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = parent.builder().clone();
        let predicate_atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::Boolean));
        let branch_input_atom = builder
            .borrow_mut()
            .add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])));
        let predicate_tracer = parent.tracer(predicate_atom, None);
        let branch_input_tracer = parent.tracer(branch_input_atom, None);
        let output = batch(
            |(predicate, input)| {
                let condition_regions = vec![scalar_scale_branch(2.0), constant_branch];
                let condition = ConditionOperation::new();
                let operation = ArrayOperation::Condition(condition);
                let outputs =
                    input.context().bind(operation, condition_regions, &[predicate.clone(), input.clone()])?;
                Ok(outputs.into_iter().next().unwrap())
            },
            (predicate_tracer, branch_input_tracer),
            (BatchAxis::replicated(), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        let output_atom = output.atom_id().unwrap();
        let program = builder
            .borrow()
            .clone()
            .build::<(Array, Array), Array>(vec![output_atom], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[3] .
                let %2:f64[3] = condition %0 %1 [
                    true={
                        lambda %0:f64[3] .
                        let %1:f64[] = const 2.0
                            %2:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %1
                            %3:f64[3] = mul %0 %2
                        in (%3)
                    },
                    false={
                        lambda %0:f64[3] .
                        let %1:f64[] = const 7.0
                            %2:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %1
                        in (%2)
                    },
                ]
                in (%2)"},
        );
        let truthy = Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[true]).unwrap();
        let falsy = Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[false]).unwrap();
        let branch_input = Array::vector(vec![1.0, 4.0, 9.0]).unwrap();
        assert_eq!(program.interpret((truthy, branch_input.clone())), Ok(Array::vector(vec![2.0, 8.0, 18.0]).unwrap()));
        assert_eq!(program.interpret((falsy, branch_input)), Ok(Array::vector(vec![7.0, 7.0, 7.0]).unwrap()));
    }

    #[test]
    fn test_condition_batching_replicated_predicate_with_mismatched_mapped_axes() {
        // A replicated predicate stages one condition whose branches receive inputs mapped along different physical
        // axes. The rule aligns both inputs before batching the branches, so the per-item `x + y` and `x * y` pair
        // up the same batch items.
        let branch = |operation: ArrayOperation<Array>| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let x = builder.add_input(ArrayType::new_static(DataType::F64, [3]));
            let y = builder.add_input(ArrayType::new_static(DataType::F64, [3]));
            let output = builder.add_instruction(operation, Vec::new(), vec![x, y], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let parent = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = parent.builder().clone();
        let predicate = parent.input(ArrayType::scalar(DataType::Boolean));
        let x = parent.input(ArrayType::new_static(DataType::F64, [2, 3]));
        let y = parent.input(ArrayType::new_static(DataType::F64, [3, 2]));
        let context = BatchingContext::new(parent, 2);
        let regions =
            vec![branch(ArrayOperation::Add(AddOperation::new())), branch(ArrayOperation::Mul(MulOperation::new()))];
        let driver = CountingBatchingDriver::new(&regions);
        let outputs = ConditionOperation::new()
            .batch(
                &context,
                &driver,
                &[
                    ArrayBatch::replicated(predicate),
                    ArrayBatch::new(x, BatchAxis::new(0)).unwrap(),
                    ArrayBatch::new(y, BatchAxis::new(1)).unwrap(),
                ],
            )
            .unwrap();
        assert_eq!(outputs.iter().map(ArrayBatch::batch_axis).collect::<Vec<_>>(), vec![BatchAxis::new(0)]);
        let program = builder
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(
                outputs.iter().map(|output| output.value().atom_id().unwrap()).collect(),
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[2, 3], %2:f64[3, 2] .
                let %3:f64[2, 3] = condition %0 %1 %2 [
                    true={
                        lambda %0:f64[2, 3], %1:f64[3, 2] .
                        let %2:f64[2, 3] = transpose [permutation=[1, 0]] %1
                            %3:f64[2, 3] = add %0 %2
                        in (%3)
                    },
                    false={
                        lambda %0:f64[2, 3], %1:f64[3, 2] .
                        let %2:f64[2, 3] = transpose [permutation=[1, 0]] %1
                            %3:f64[2, 3] = mul %0 %2
                        in (%3)
                    },
                ]
                in (%3)"},
        );
        let x = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array::matrix(3, 2, vec![10.0, 40.0, 20.0, 50.0, 30.0, 60.0]).unwrap();
        assert_eq!(
            program.interpret(vec![Array::scalar(true).unwrap(), x.clone(), y.clone()]),
            Ok(vec![Array::matrix(2, 3, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]).unwrap()]),
        );
        assert_eq!(
            program.interpret(vec![Array::scalar(false).unwrap(), x, y]),
            Ok(vec![Array::matrix(2, 3, vec![10.0, 40.0, 90.0, 160.0, 250.0, 360.0]).unwrap()]),
        );
    }

    #[test]
    fn test_condition_batching_replicated_predicate_with_effectful_branches() {
        // Unlike a batch-varying predicate, a replicated predicate selects one branch for the whole batch, so an
        // effectful branch batches into one staged condition that runs its effect once per execution.
        let branch = |label: &str, factor: f64| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(ArrayType::scalar(DataType::F64));
            builder
                .add_instruction(ArrayOperation::Print(PrintOperation::new(label)), Vec::new(), vec![input], None)
                .unwrap();
            let factor = builder.add_constant(Array::scalar(factor).unwrap());
            let output =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, factor], None).unwrap()[0];
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let parent = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = parent.builder().clone();
        let predicate = parent.input(ArrayType::scalar(DataType::Boolean));
        let input = parent.input(ArrayType::new_static(DataType::F64, [3]));
        let context = BatchingContext::new(parent, 3);
        let regions = vec![branch("true", 2.0), branch("false", 3.0)];
        let driver = CountingBatchingDriver::new(&regions);
        let outputs = ConditionOperation::new()
            .batch(
                &context,
                &driver,
                &[ArrayBatch::replicated(predicate), ArrayBatch::new(input, BatchAxis::new(0)).unwrap()],
            )
            .unwrap();
        assert_eq!(outputs.iter().map(ArrayBatch::batch_axis).collect::<Vec<_>>(), vec![BatchAxis::new(0)]);
        let program = builder
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(
                outputs.iter().map(|output| output.value().atom_id().unwrap()).collect(),
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[3] .
                let %2:f64[3] = condition %0 %1 [
                    true={
                        lambda %0:f64[3] .
                        let %1:f64[3] = print [label=true] %0
                            %2:f64[] = const 2.0
                            %3:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %2
                            %4:f64[3] = mul %0 %3
                        in (%4)
                    },
                    false={
                        lambda %0:f64[3] .
                        let %1:f64[3] = print [label=false] %0
                            %2:f64[] = const 3.0
                            %3:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %2
                            %4:f64[3] = mul %0 %3
                        in (%4)
                    },
                ]
                in (%2)"},
        );
        assert!(program.effects().classes().is_ordered());
        let input = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(
            program.interpret(vec![Array::scalar(true).unwrap(), input.clone()]),
            Ok(vec![Array::vector(vec![2.0, 4.0, 6.0]).unwrap()]),
        );
        assert_eq!(
            program.interpret(vec![Array::scalar(false).unwrap(), input]),
            Ok(vec![Array::vector(vec![3.0, 6.0, 9.0]).unwrap()]),
        );
    }

    #[test]
    fn test_condition_batching_mapped_predicate() {
        // A batch-varying predicate cannot select one branch for the whole batch, so batching runs both pure branches
        // and merges their outputs per batch item through the `Select` batching rule.
        let regions = vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)];
        check_operation_batching!(
            @exact,
            context = EagerContext::<Array, ArrayOperation<Array>>::new(),
            driver = &RecursiveBatchingDriver::new(&regions),
            operation = ConditionOperation::<ArrayType>::new(),
            axis_size = 3,
            axis_sharding = ShardingDimension::Replicated,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![true, false, true]).unwrap()),
                    (@mapped(axis = 0), Array::vector(vec![1.0, 4.0, 9.0]).unwrap()),
                ],
                outputs = [(@mapped(axis = 0), Array::vector(vec![2.0, 12.0, 18.0]).unwrap())],
            }],
        );

        // Non-scalar outputs select whole batch items. The batch size differs from the per-item vector length in the
        // first case, so the Boolean `[2]` predicate must become `[2, 1]` before selecting between the `[2, 3]` branch
        // values. In the second case the predicate's physical axis is 0 while the input's physical batch axis is 1, so
        // selection must keep columns as batch items while reconciling those axes, and it returns them along axis 0.
        let regions = vec![vector_scale_branch(3, 2.0), vector_scale_branch(3, 3.0)];
        check_operation_batching!(
            @exact,
            context = EagerContext::<Array, ArrayOperation<Array>>::new(),
            driver = &RecursiveBatchingDriver::new(&regions),
            operation = ConditionOperation::<ArrayType>::new(),
            axis_size = 2,
            axis_sharding = ShardingDimension::Replicated,
            cases = [
                {
                    inputs = [
                        (@mapped(axis = 0), Array::vector(vec![true, false]).unwrap()),
                        (@mapped(axis = 0), Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
                    ],
                    outputs = [
                        (@mapped(axis = 0), Array::matrix(2, 3, vec![2.0, 4.0, 6.0, 12.0, 15.0, 18.0]).unwrap()),
                    ],
                },
                {
                    inputs = [
                        (@mapped(axis = 0), Array::vector(vec![true, false]).unwrap()),
                        (@mapped(axis = 1), Array::matrix(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap()),
                    ],
                    outputs = [
                        (@mapped(axis = 0), Array::matrix(2, 3, vec![2.0, 4.0, 6.0, 12.0, 15.0, 18.0]).unwrap()),
                    ],
                },
            ],
        );

        // Equal batch and item sizes must still select rows (i.e., batch items) rather than columns.
        let regions = vec![vector_scale_branch(2, 2.0), vector_scale_branch(2, 3.0)];
        check_operation_batching!(
            @exact,
            context = EagerContext::<Array, ArrayOperation<Array>>::new(),
            driver = &RecursiveBatchingDriver::new(&regions),
            operation = ConditionOperation::<ArrayType>::new(),
            axis_size = 2,
            axis_sharding = ShardingDimension::Replicated,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![true, false]).unwrap()),
                    (@mapped(axis = 0), Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap()),
                ],
                outputs = [(@mapped(axis = 0), Array::matrix(2, 2, vec![2.0, 4.0, 9.0, 12.0]).unwrap())],
            }],
        );

        // A replicated output of one branch is broadcast across the batch before it is joined with the mapped output
        // of the other branch.
        let regions = vec![constant_vector_branch(vec![10.0, 20.0, 30.0]), vector_scale_branch(3, 3.0)];
        check_operation_batching!(
            @exact,
            context = EagerContext::<Array, ArrayOperation<Array>>::new(),
            driver = &RecursiveBatchingDriver::new(&regions),
            operation = ConditionOperation::<ArrayType>::new(),
            axis_size = 2,
            axis_sharding = ShardingDimension::Replicated,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![true, false]).unwrap()),
                    (@mapped(axis = 0), Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
                ],
                outputs = [
                    (@mapped(axis = 0), Array::matrix(2, 3, vec![10.0, 20.0, 30.0, 12.0, 15.0, 18.0]).unwrap()),
                ],
            }],
        );
    }

    #[test]
    fn test_condition_batching_aligns_manual_variation_for_batch_varying_predicates() {
        // Inside a manual region, branch values may vary over manual axes while the predicate stays invariant. The
        // selections that gate the branch inputs and merge the branch outputs per batch item first give the Boolean
        // predicate, which carries no tangent, the variation of the branch values through one `parallel_vary`, which
        // every selection that requires the same variation shares.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying_sharding = |rank| Sharding::replicated(mesh.clone(), rank).with_varying_manual_axes(["x"]).unwrap();
        let predicate_type = ArrayType::new_static(DataType::Boolean, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1))
            .unwrap();
        let value_type = ArrayType::new_static(DataType::F64, [2]).with_sharding(varying_sharding(1)).unwrap();
        let branch_type = ArrayType::scalar(DataType::F64).with_sharding(varying_sharding(0)).unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(branch_type.clone());
        let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let sine_branch =
            builder.build::<Vec<Array>, Vec<Array>>(vec![sine], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(branch_type);
        let identity_branch =
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |inputs: Vec<ArrayTracer>| {
                let context = BatchingContext::<_, ArrayBatchingPolicy>::new(inputs[0].domain(), 2);
                let predicate =
                    BatchingTracer::new(context.clone(), ArrayBatch::new(inputs[0].clone(), BatchAxis::new(0))?);
                let value =
                    BatchingTracer::new(context.clone(), ArrayBatch::new(inputs[1].clone(), BatchAxis::new(0))?);
                let outputs = context.bind(
                    ArrayOperation::Condition(ConditionOperation::new()),
                    vec![sine_branch, identity_branch],
                    &[predicate, value],
                )?;
                Ok(outputs.into_iter().next().unwrap().into_batch().into_value())
            },
            vec![predicate_type, value_type.clone()],
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh })],
        )
        .unwrap();
        assert_eq!(output_type, value_type);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[2][sharding={mesh<['x'=2:manual]>, [{}]}], \
                    %1:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                let %2:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = stop_gradient %1
                    %3:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %0
                    %4:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %3 %1 %2
                    %5:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sin %4
                    %6:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %3 %2 %1
                    %7:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %3 %5 %6
                in (%7)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_condition_batching_rejects_batch_varying_predicates_with_effectful_branches() {
        // Effectful branches cannot be batched under a batch-varying predicate: both branches would run for the whole
        // batch and their observable effects cannot be selected per batch item.
        let effectful_branch = |label| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(ArrayType::scalar(DataType::F64));
            builder
                .add_instruction(ArrayOperation::Print(PrintOperation::new(label)), Vec::new(), vec![input], None)
                .unwrap();
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let result: Result<Array, BatchingError> = batch(
            |(predicate, input)| {
                let outputs = input.context().bind(
                    ArrayOperation::Condition(ConditionOperation::new()),
                    vec![effectful_branch("true"), effectful_branch("false")],
                    &[predicate.clone(), input.clone()],
                )?;
                Ok(outputs.into_iter().next().unwrap())
            },
            (Array::vector(vec![true, false]).unwrap(), Array::vector(vec![1.0, 2.0]).unwrap()),
            (BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        );
        assert_eq!(
            result,
            Err(BatchingError::UnsupportedOperation {
                message: "cannot batch a `condition` with a batch-varying predicate and branches that have effects or \
                          deferred work because neither observable effects nor transformation obligations can be \
                          selected per batch item"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_condition_batching_rejects_incompatible_branch_output_signatures() {
        // Both regions are well-formed individually. Selection must not broadcast their unequal result signatures
        // into an apparently valid mapped result before enforcing the condition's primitive contract.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let scalar = builder.add_constant(Array::scalar(1.0f64).unwrap());
        let true_branch = builder.build::<Vec<Array>, Vec<Array>>(vec![scalar], Vec::new(), vec![Placeholder]).unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let vector = builder.add_constant(Array::vector(vec![2.0f64, 3.0]).unwrap());
        let false_branch =
            builder.build::<Vec<Array>, Vec<Array>>(vec![vector], Vec::new(), vec![Placeholder]).unwrap();
        assert_eq!(
            true_branch.to_string(),
            indoc! {"
            lambda  .
            let %0:f64[] = const 1.0
            in (%0)"},
        );
        assert_eq!(
            false_branch.to_string(),
            indoc! {"
            lambda  .
            let %0:f64[2] = const [2.0, 3.0]
            in (%0)"},
        );
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);
        let predicate = BatchingTracer::new(
            context.clone(),
            ArrayBatch::new(Array::vector(vec![true, false]).unwrap(), BatchAxis::new(0)).unwrap(),
        );
        let error = context
            .bind(ArrayOperation::Condition(ConditionOperation::new()), vec![true_branch, false_branch], &[predicate])
            .unwrap_err();
        assert_eq!(
            error,
            ProgramError::Type(TypeError::invalid(
                "`condition` branch output type signature mismatch: expected [f64[]] but got [f64[2]]",
            )),
        );
    }

    #[test]
    fn test_condition_batching_validates_input_count() {
        // Batching reports a missing predicate or an extra input as the input-count error of the other rules.
        let unit_branch = ProgramBuilder::<Array, ArrayOperation<Array>>::new()
            .build::<Vec<Array>, Vec<Array>>(Vec::new(), Vec::new(), Vec::new())
            .unwrap();
        let regions = vec![unit_branch.clone(), unit_branch];
        let driver = CountingBatchingDriver::new(&regions);
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);
        assert_eq!(
            ConditionOperation::<ArrayType>::new().batch(&context, &driver, &[]),
            Err(BatchingError::Program(ProgramError::InvalidInputCount { expected: 1, actual: 0 })),
        );
        assert_eq!(
            ConditionOperation::<ArrayType>::new().batch(
                &context,
                &driver,
                &[
                    ArrayBatch::replicated(Array::scalar(true).unwrap()),
                    ArrayBatch::replicated(Array::scalar(1.0f64).unwrap()),
                ],
            ),
            Err(BatchingError::Program(ProgramError::InvalidInputCount { expected: 1, actual: 2 })),
        );
    }

    #[test]
    fn test_condition_batching_after_differentiation_with_a_mapped_predicate() {
        // Batching the forward-mode program of a condition along a batch-varying predicate selects the primal and
        // tangent outputs of each batch item's branch.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let true_region = builder.import_program(scalar_scale_branch(2.0));
        let false_region = builder.import_program(scalar_branch(ArrayOperation::Sin(SinOperation::new())));
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(ConditionOperation::new(), vec![true_region, false_region], vec![predicate, input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[], %2:f64[] .
                let %3:f64[], %4:f64[] = condition %0 %1 %2 [
                    true={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = const 2.0
                            %3:f64[] = mul %0 %2
                            %4:f64[] = mul %2 %1
                        in (%3, %4)
                    },
                    false={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = sin %0
                            %3:f64[] = cos %0
                            %4:f64[] = mul %3 %1
                        in (%2, %4)
                    },
                ]
                in (%3, %4)"},
        );
        let (batched, _) = jvp
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:bool[2], %1:f64[2], %2:f64[2] .
                let %3:f64[2] = stop_gradient %1
                    %4:f64[2] = select %0 %1 %3
                    %5:f64[] = const 2.0
                    %6:f64[2] = broadcast [output_type=f64[2], output_axes=[]] %5
                    %7:f64[2] = mul %4 %6
                    %8:f64[2] = select %0 %3 %1
                    %9:f64[2] = sin %8
                    %10:f64[2] = select %0 %7 %9
                    %11:f64[2] = broadcast [output_type=f64[2], output_axes=[]] %5
                    %12:f64[2] = stop_gradient %2
                    %13:f64[2] = select %0 %2 %12
                    %14:f64[2] = mul %11 %13
                    %15:f64[2] = cos %8
                    %16:f64[2] = select %0 %12 %2
                    %17:f64[2] = mul %15 %16
                    %18:f64[2] = select %0 %14 %17
                in (%10, %18)"},
        );
        assert_eq!(
            batched.interpret(vec![
                Array::vector(vec![true, false]).unwrap(),
                Array::vector(vec![1.0, 2.0]).unwrap(),
                Array::vector(vec![3.0, 3.0]).unwrap(),
            ]),
            Ok(vec![
                Array::vector(vec![2.0, 2.0f64.sin()]).unwrap(),
                Array::vector(vec![6.0, 3.0 * 2.0f64.cos()]).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_condition_batching_array_ir_preserves_eager_reference_refinements() {
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let referent_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = builder.add_input(ReferenceType::new(referent_type).into());
        let snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let branch = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            branch.to_string(),
            indoc! {"
            lambda %0:ref<f32[extent]> .
            let %1:f32[extent] = reference_read %0
            in (%1)"},
        );
        let values = Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let reference = ArrayReference::new(values.clone());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            TestEagerContext::new(),
            TestIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let inputs = [
            ArrayIrBatch::replicated(array(Array::scalar(true).unwrap())),
            ArrayIrBatch::new(TestIrValue::Reference(reference.clone()), BatchAxis::new(0)).unwrap(),
        ];
        let regions = vec![branch.clone(), branch];
        let driver = CountingBatchingDriver::new(&regions);
        assert_eq!(
            ConditionOperation::new().batch(&context, &driver, &inputs),
            Ok(vec![ArrayIrBatch::new(array(values.clone()), BatchAxis::new(0)).unwrap()]),
        );
        assert_eq!(reference.read().unwrap(), values);
    }

    #[test]
    fn test_condition_batching_array_ir_rejects_mapped_shape_computation() {
        // Computing a first-class output extent from mapped scalar data cannot produce a rectangular array.
        // The ordered scalar-to-dimension gateway makes the branch effectful, so mapped selection rejects it before
        // either branch is transformed.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            TestEagerContext::new(),
            TestIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let predicate = ArrayIrBatch::new(array(Array::vector(vec![true, false]).unwrap()), BatchAxis::new(0)).unwrap();
        let computed_branch = |use_second: bool| {
            let (_, program) = TracingContext::<TestIrValue, TestIrOperation>::trace(
                |(value, first, second): (TestTracer, TestTracer, TestTracer)| {
                    let extent = if use_second { second } else { first }.to_dimension(length.clone())?;
                    let broadcast = value.dynamic_broadcast_to(std::slice::from_ref(&extent))?;
                    Ok((extent, broadcast))
                },
                (
                    ArrayIrType::from(ArrayType::scalar(DataType::F64)),
                    ArrayIrType::from(ArrayType::scalar(DataType::I64)),
                    ArrayIrType::from(ArrayType::scalar(DataType::I64)),
                ),
            )
            .unwrap();
            program.into_flat_program()
        };
        let regions = vec![computed_branch(false), computed_branch(true)];
        let driver = CountingBatchingDriver::new(&regions);
        let inputs = [
            predicate,
            ArrayIrBatch::new(array(Array::vector(vec![5.0f64, 7.0]).unwrap()), BatchAxis::new(0)).unwrap(),
            ArrayIrBatch::new(array(Array::vector(vec![1i64, 3]).unwrap()), BatchAxis::new(0)).unwrap(),
            ArrayIrBatch::new(array(Array::vector(vec![2i64, 0]).unwrap()), BatchAxis::new(0)).unwrap(),
        ];
        assert_eq!(
            ConditionOperation::new().batch(&context, &driver, &inputs).unwrap_err(),
            BatchingError::UnsupportedOperation {
                message: "cannot batch a `condition` with a batch-varying predicate and branches that have effects or \
                          deferred work because neither observable effects nor transformation obligations can be \
                          selected per batch item"
                    .to_string(),
            },
        );
    }

    #[test]
    fn test_condition_batching_array_ir_rejects_incompatible_branch_output_signatures() {
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let scalar = builder.add_constant(array(Array::scalar(1.0f64).unwrap()));
        let true_branch = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![scalar], Vec::new(), vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let vector = builder.add_constant(array(Array::vector(vec![2.0f64, 3.0]).unwrap()));
        let false_branch = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![vector], Vec::new(), vec![Placeholder])
            .unwrap();
        assert_eq!(
            true_branch.to_string(),
            indoc! {"
            lambda  .
            let %0:f64[] = const 1.0
            in (%0)"},
        );
        assert_eq!(
            false_branch.to_string(),
            indoc! {"
            lambda  .
            let %0:f64[2] = const [2.0, 3.0]
            in (%0)"},
        );
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            TestEagerContext::new(),
            TestIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let predicate = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::new(array(Array::vector(vec![true, false]).unwrap()), BatchAxis::new(0)).unwrap(),
        );
        let error = context
            .bind(TestIrOperation::Condition(ConditionOperation::new()), vec![true_branch, false_branch], &[predicate])
            .unwrap_err();
        assert_eq!(
            error,
            ProgramError::Type(TypeError::invalid(
                "`condition` branch output type signature mismatch: expected [f64[]] but got [f64[2]]",
            )),
        );
    }

    #[test]
    fn test_condition_batching_array_ir_keeps_dimension_outputs_replicated_under_mapped_predicates() {
        // Under a batch-varying predicate, dimension results of the two branches stay replicated, so they must agree.
        // Dimension constants are typed exactly, so branches that return different extents have different output
        // types, and the condition rejects them by their signatures before batching selects anything.
        let output_dimension_type = DimensionType::new("result", DimensionBounds::new(1, Some(9)).unwrap());
        let branch = |extent: usize| {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let extent = builder.add_constant(TestIrValue::Dimension(
                DimensionValue::new(output_dimension_type.clone(), extent).unwrap(),
            ));
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![extent], Vec::new(), vec![Placeholder])
                .unwrap()
        };
        let true_branch = branch(2);
        let false_branch = branch(3);
        let expected = TypeError::invalid(
            "`condition` branch output type signature mismatch: expected [dimension<2>] but got [dimension<3>]",
        );
        assert_eq!(
            ConditionOperation::<ArrayIrType>::new().infer_output_types(
                &[ArrayType::scalar(DataType::Boolean).into()],
                &[true_branch.interface(), false_branch.interface()],
            ),
            Err(expected.clone()),
        );
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            TestEagerContext::new(),
            TestIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let predicate = array(Array::vector(vec![true, false]).unwrap());
        let error = context
            .bind(
                TestIrOperation::Condition(ConditionOperation::new()),
                vec![true_branch, false_branch],
                &[BatchingTracer::new(context.clone(), ArrayIrBatch::new(predicate, BatchAxis::new(0)).unwrap())],
            )
            .unwrap_err();
        assert_eq!(error, ProgramError::Type(expected));

        // Branches that return the same exact dimension agree, and the replicated dimension is the batched result.
        let expected = TestIrValue::Dimension(DimensionValue::new(output_dimension_type, 2).unwrap());
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let extent = builder.add_constant(expected.clone());
        let branch = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![extent], Vec::new(), vec![Placeholder])
            .unwrap();
        assert_eq!(
            ConditionOperation::<ArrayIrType>::new().infer_output_types(
                &[ArrayType::scalar(DataType::Boolean).into()],
                &[branch.interface(), branch.interface()],
            ),
            Ok(vec![expected.r#type().into_owned()]),
        );
        let outputs = context
            .bind(
                TestIrOperation::Condition(ConditionOperation::new()),
                vec![branch.clone(), branch],
                &[BatchingTracer::new(
                    context.clone(),
                    ArrayIrBatch::new(array(Array::vector(vec![true, false]).unwrap()), BatchAxis::new(0)).unwrap(),
                )],
            )
            .unwrap();
        assert_eq!(
            outputs.into_iter().map(BatchingTracer::into_batch).collect::<Vec<_>>(),
            vec![ArrayIrBatch::replicated(expected)],
        );

        // A dimension established by the branch input keeps one parent identity when both branches forward it.
        // Equality follows from that shared identity, so no runtime assertion or branch-local definition is needed.
        let dimension_type = DimensionType::new("length", DimensionBounds::new(0, Some(9)).unwrap());
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let input = builder.add_input(dimension_type.clone().into());
        let branch = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            ConditionOperation::<ArrayIrType>::new().infer_output_types(
                &[ArrayType::scalar(DataType::Boolean).into(), dimension_type.clone().into()],
                &[branch.interface(), branch.interface()],
            ),
            Ok(vec![dimension_type.clone().into()]),
        );
        let batch_type = DimensionType::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let trace = TracingContext::<TestIrValue, TestIrOperation>::new();
        let batch_extent = trace.input(batch_type.clone().into());
        let predicate =
            trace.input(ArrayType::new(DataType::Boolean, Shape::new(vec![batch_type.dimension().clone()])).into());
        let dimension = trace.input(dimension_type.clone().into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent);
        let outputs = context
            .bind(
                TestIrOperation::Condition(ConditionOperation::new()),
                vec![branch.clone(), branch],
                &[
                    BatchingTracer::new(context.clone(), ArrayIrBatch::new(predicate, BatchAxis::new(0)).unwrap()),
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(dimension.clone())),
                ],
            )
            .unwrap()
            .into_iter()
            .map(BatchingTracer::into_batch)
            .collect::<Vec<_>>();
        assert_eq!(outputs, vec![ArrayIrBatch::replicated(dimension.clone())]);
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![outputs[0].value().atom_id().unwrap()],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:bool[batch], %2:dimension<length ∈ [0, 9)> .
                in (%2)"},
        );
        let dimension = TestIrValue::Dimension(DimensionValue::new(dimension_type, 4).unwrap());
        assert_eq!(
            program.interpret(vec![
                TestIrValue::Dimension(DimensionValue::new(batch_type, 2).unwrap()),
                array(Array::vector(vec![true, false]).unwrap()),
                dimension.clone(),
            ]),
            Ok(vec![dimension]),
        );
    }

    #[test]
    fn test_condition_batching_array_ir_rejects_staged_branch_local_type_identities() {
        // The branches are valid and agree on their result value. Flattening their definitions into the parent would
        // still duplicate one nominal identity, so rejection must precede any replay or parent-trace mutation.
        let input_type = DimensionType::new("n", DimensionBounds::new(1, Some(8)).unwrap());
        let one = DimensionValue::constant(1).unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let input = builder.add_input(input_type.clone().into());
        let one_atom = builder.add_constant(TestIrValue::Dimension(one.clone()));
        let result = builder
            .add_instruction(
                DimensionOperation::Add(DimensionAddOperation::new(&input_type, one.r#type().as_ref()).unwrap()),
                Vec::new(),
                vec![input, one_atom],
                None,
            )
            .unwrap()[0];
        let branch = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![result], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let trace = TracingContext::<TestIrValue, TestIrOperation>::new();
        let batch_dimension = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_extent = trace.input(DimensionType::from(batch_dimension.clone()).into());
        let predicate = trace
            .input(ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Dynamic(batch_dimension)])).into());
        let extent = trace.input(input_type.into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent);
        let error = context
            .bind(
                TestIrOperation::Condition(ConditionOperation::new()),
                vec![branch.clone(), branch],
                &[
                    BatchingTracer::new(context.clone(), ArrayIrBatch::new(predicate, BatchAxis::new(0)).unwrap()),
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(extent)),
                ],
            )
            .unwrap_err();
        assert!(matches!(
            error.downcast_custom::<BatchingError>(),
            Some(BatchingError::UnsupportedOperation { message })
                if message == "cannot stage a `condition` with a batch-varying predicate whose `true` branch defines \
                               branch-local type identity `n + 1` because select lowering cannot preserve branch-local \
                               type identity scopes",
        ));
        let unchanged = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder; 3], Vec::new())
            .unwrap();
        assert_eq!(
            unchanged.to_string(),
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:bool[batch], %2:dimension<n ∈ [1, 8)> .
                in ()"},
        );
    }

    #[test]
    fn test_condition_batching_array_ir_refines_outputs_of_unspecialized_branches() {
        // Batching a condition whose refined output comes from unspecialized branches keeps its batched output refined,
        // under replicated and mapped predicates alike.
        let matrix = |values: &[f64]| {
            array(Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2, 3]), values).unwrap())
        };
        let input = matrix(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let other_input = matrix(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        let extent = DimensionValue::constant(2).unwrap();
        for (predicate_axis, predicate, expected, expected_program) in [
            (
                BatchAxis::replicated(),
                array(Array::scalar(true).unwrap()),
                matrix(&[11.0, 24.0, 39.0, 56.0, 75.0, 96.0]),
                indoc! {"
                    lambda %0:dimension<2>, %1:bool[], %2:f64[2, 3], %3:f64[2, 3] .
                    let %4:f64[2, 3] = condition %1 %0 %2 %3 [
                        true={
                            lambda %0:dimension<2>, %1:f64[2, 3], %2:f64[2, 3] .
                            let %3:f64[2, 3] = mul %1 %1
                                %4:f64[2, 3] = add %3 %2
                            in (%4)
                        },
                        false={
                            lambda %0:dimension<2>, %1:f64[2, 3], %2:f64[2, 3] .
                            let %3:f64[2, 3] = add %1 %2
                            in (%3)
                        },
                    ]
                    in (%0, %4)"},
            ),
            (
                BatchAxis::new(0),
                array(Array::vector(vec![true, false]).unwrap()),
                matrix(&[11.0, 24.0, 39.0, 44.0, 55.0, 66.0]),
                indoc! {"
                    lambda %0:dimension<2>, %1:bool[2], %2:f64[2, 3], %3:f64[2, 3] .
                    let %4:dimension<3> = constant [value=3]
                        %5:bool[2, 3] = broadcast [output_axes=[0]] %1 %0 %4
                        %6:dimension<3> = constant [value=3]
                        %7:bool[2, 3] = broadcast [output_axes=[0]] %1 %0 %6
                        %8:f64[2, 3] = stop_gradient %2
                        %9:f64[2, 3] = select %7 %2 %8
                        %10:f64[2, 3] = mul %9 %9
                        %11:dimension<3> = constant [value=3]
                        %12:bool[2, 3] = broadcast [output_axes=[0]] %1 %0 %11
                        %13:f64[2, 3] = stop_gradient %3
                        %14:f64[2, 3] = select %12 %3 %13
                        %15:f64[2, 3] = add %10 %14
                        %16:dimension<3> = constant [value=3]
                        %17:bool[2, 3] = broadcast [output_axes=[0]] %1 %0 %16
                        %18:f64[2, 3] = select %17 %8 %2
                        %19:dimension<3> = constant [value=3]
                        %20:bool[2, 3] = broadcast [output_axes=[0]] %1 %0 %19
                        %21:f64[2, 3] = select %20 %13 %3
                        %22:f64[2, 3] = add %18 %21
                        %23:f64[2, 3] = select %5 %15 %22
                    in (%0, %23)"},
            ),
        ] {
            let batched = refined_vector_condition_program()
                .batched_with_threaded_extent(
                    extent.r#type().into_owned(),
                    ShardingDimension::Replicated,
                    &[predicate_axis, BatchAxis::new(0), BatchAxis::new(0)],
                    ProgramBatchingOutputAxesPolicy::Natural,
                )
                .unwrap()
                .into_parts()
                .0;
            assert_eq!(batched.to_string(), expected_program);
            assert_eq!(batched.output_types()[1], ArrayIrType::Array(ArrayType::new_static(DataType::F64, [2, 3])));
            assert_eq!(
                batched.interpret(vec![
                    TestIrValue::Dimension(extent.clone()),
                    predicate,
                    input.clone(),
                    other_input.clone(),
                ]),
                Ok(vec![TestIrValue::Dimension(extent.clone()), expected]),
            );
        }
    }

    #[test]
    fn test_condition_batching_array_ir_threads_reference_carries() {
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let mut true_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = true_builder.add_input(reference_type.clone().into());
        let current = true_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        true_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, current], None)
            .unwrap();
        let snapshot = true_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut false_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = false_builder.add_input(reference_type.into());
        let snapshot = false_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let false_branch = false_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_branch = builder.import_program(true_branch);
        let false_branch = builder.import_program(false_branch);
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(scalar_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let snapshot = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![snapshot, frozen],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        // A replicated predicate keeps one structural condition through which the mapped reference carry flows like an
        // array carry: the taken branch doubles every item in place and its snapshot comes back at the referent's axis,
        // so batching the stateful program directly agrees with batching its discharged counterpart for either branch.
        let axis_extent = DimensionValue::constant(2).unwrap();
        let extent_type = axis_extent.r#type().into_owned();
        let direct = source
            .batched_with_threaded_extent(
                extent_type.clone(),
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        let discharged = source
            .discharge_references(0)
            .unwrap()
            .into_program_without_external_references()
            .unwrap()
            .batched_with_threaded_extent(
                extent_type,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(direct.output_axes(), &[BatchAxis::new(0), BatchAxis::new(0)]);
        assert_eq!(discharged.output_axes(), direct.output_axes());
        let (direct, _) = direct.into_parts();
        let (discharged, _) = discharged.into_parts();
        assert_eq!(
            discharged.to_string(),
            indoc! {"
                lambda %0:dimension<2>, %1:bool[], %2:f32[2] .
                let %3:f32[2], %4:f32[2] = condition %1 %0 %2 [
                    true={
                        lambda %0:dimension<2>, %1:f32[2] .
                        let %2:f32[2] = add %1 %1
                        in (%2, %2)
                    },
                    false={
                        lambda %0:dimension<2>, %1:f32[2] .
                        in (%1, %1)
                    },
                ]
                in (%0, %3, %4)"},
        );
        assert_eq!(
            direct.to_string(),
            indoc! {"
                lambda %0:dimension<2>, %1:bool[], %2:f32[2] .
                let %3:ref<f32[2]> = reference_new %2
                    %4:f32[2] = condition %1 %0 %3 [
                        true={
                            lambda %0:dimension<2>, %1:ref<f32[2]> .
                            let %2:f32[2] = reference_read %1
                                () = reference_add_update %1 %2
                                %3:f32[2] = reference_read %1
                            in (%3)
                        },
                        false={
                            lambda %0:dimension<2>, %1:ref<f32[2]> .
                            let %2:f32[2] = reference_read %1
                            in (%2)
                        },
                    ]
                    %5:f32[2] = reference_freeze %3
                in (%0, %4, %5)"},
        );

        // Interpreting both batched programs yields the same snapshot and final state for either predicate.
        for (predicate, expected) in [(true, vec![2.0f32, 4.0]), (false, vec![1.0f32, 2.0])] {
            let inputs = vec![
                TestIrValue::Dimension(axis_extent.clone()),
                TestIrValue::Array(Array::scalar(predicate).unwrap()),
                TestIrValue::Array(Array::vector(vec![1.0f32, 2.0]).unwrap()),
            ];
            let expected = vec![
                TestIrValue::Dimension(axis_extent.clone()),
                TestIrValue::Array(Array::vector(expected.clone()).unwrap()),
                TestIrValue::Array(Array::vector(expected).unwrap()),
            ];
            assert_eq!(direct.interpret(inputs.clone()), Ok(expected.clone()));
            assert_eq!(discharged.interpret(inputs), Ok(expected));
        }
    }

    #[test]
    fn test_condition_batching_array_ir_rejects_branch_dependent_reference_outputs() {
        // A replicated predicate binds one condition into the parent context. A staging parent does not interpret that
        // condition, so batching validates the reference contracts of both branches itself and rejects the condition
        // that interpretation and discharge reject.
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let parent = DomainTracingContext::<TestEagerContext>::new();
        let builder = parent.builder().clone();
        let predicate = parent.input(ArrayType::scalar(DataType::Boolean).into());
        let first = parent.input(reference_type.clone());
        let second = parent.input(reference_type);
        let batch_extent = parent.lift(TestIrValue::Dimension(DimensionValue::constant(2).unwrap())).unwrap();
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent, batch_extent);
        let regions = branch_dependent_reference_branches().to_vec();
        let driver = CountingBatchingDriver::new(&regions);
        assert_eq!(
            ConditionOperation::<ArrayIrType>::new()
                .batch(
                    &context,
                    &driver,
                    &[
                        ArrayIrBatch::replicated(predicate),
                        ArrayIrBatch::replicated(first),
                        ArrayIrBatch::replicated(second),
                    ],
                )
                .map(|_| ()),
            Err(BatchingError::Program(ProgramError::Type(TypeError::invalid(
                "`condition` reference output 0 must denote the same reference in both branches, but the `true` \
                 branch returns region input 0 and the `false` branch returns region input 1",
            )))),
        );
        assert_eq!(driver.batch_program_calls(), 0);
        assert!(builder.borrow().instructions().is_empty());
    }

    #[test]
    fn test_condition_batching_array_ir_validates_reference_accesses_under_mapped_predicates() {
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut branch_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = branch_builder.add_input(reference_type.into());
        let snapshot = branch_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let branch = branch_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // Select lowering runs both branches, so even a read-only access in one branch is rejected ahead of the general
        // purity check, naming the reference access as the cause.
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            TestEagerContext::new(),
            TestIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let reference = ArrayReference::new(Array::vector(vec![1.0f32, 2.0]).unwrap());
        let error = context
            .bind(
                TestIrOperation::Condition(ConditionOperation::new()),
                vec![branch.clone(), branch],
                &[
                    BatchingTracer::new(
                        context.clone(),
                        ArrayIrBatch::new(
                            TestIrValue::Array(Array::vector(vec![true, false]).unwrap()),
                            BatchAxis::new(0),
                        )
                        .unwrap(),
                    ),
                    BatchingTracer::new(
                        context.clone(),
                        ArrayIrBatch::new(TestIrValue::Reference(reference), BatchAxis::new(0)).unwrap(),
                    ),
                ],
            )
            .unwrap_err();
        assert!(matches!(
            error.downcast_custom::<BatchingError>(),
            Some(BatchingError::UnsupportedOperation { message })
                if message == "cannot batch a `condition` with a batch-varying predicate whose branches access \
                               references because select lowering runs both branches and reference effects cannot be \
                               selected per batch item",
        ));

        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let mut true_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = true_builder.add_input(reference_type.clone().into());
        let value = true_builder.add_input(scalar_type.clone().into());
        let doubled = true_builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![value, value], None)
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![reference, doubled],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut false_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = false_builder.add_input(reference_type.into());
        let value = false_builder.add_input(scalar_type.into());
        let false_branch = false_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![reference, value],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        // A branch that merely forwards a reference it never accesses is fine under a batch-varying predicate: the
        // reference output is never selected but passes through as the region input both branches forward, while the
        // array output is selected per item.
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            TestEagerContext::new(),
            TestIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let reference = ArrayReference::new(Array::vector(vec![1.0f32, 2.0]).unwrap());
        let outputs = context
            .bind(
                TestIrOperation::Condition(ConditionOperation::new()),
                vec![true_branch, false_branch],
                &[
                    BatchingTracer::new(
                        context.clone(),
                        ArrayIrBatch::new(
                            TestIrValue::Array(Array::vector(vec![true, false]).unwrap()),
                            BatchAxis::new(0),
                        )
                        .unwrap(),
                    ),
                    BatchingTracer::new(
                        context.clone(),
                        ArrayIrBatch::new(TestIrValue::Reference(reference.clone()), BatchAxis::new(0)).unwrap(),
                    ),
                    BatchingTracer::new(
                        context.clone(),
                        ArrayIrBatch::new(
                            TestIrValue::Array(Array::vector(vec![3.0f32, 4.0]).unwrap()),
                            BatchAxis::new(0),
                        )
                        .unwrap(),
                    ),
                ],
            )
            .unwrap();
        assert_eq!(
            outputs.into_iter().map(BatchingTracer::into_batch).collect::<Vec<_>>(),
            vec![
                ArrayIrBatch::new(TestIrValue::Reference(reference.clone()), BatchAxis::new(0)).unwrap(),
                ArrayIrBatch::new(array(Array::vector(vec![6.0f32, 4.0]).unwrap()), BatchAxis::new(0)).unwrap(),
            ],
        );
        assert_eq!(reference.read(), Ok(Array::vector(vec![1.0f32, 2.0]).unwrap()));
    }

    #[test]
    fn test_condition_batching_array_ir_forwards_refined_eager_references_under_mapped_predicates() {
        // The branches declare `ref<f32[extent]>` and `f32[extent]` inputs, and the eager reference's `f32[3]` referent
        // refines its declared type. A batch-varying predicate only forwards the untouched reference, so it accepts
        // that refinement like eager interpretation does and selects the array outputs per item.
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let reference_type = ReferenceType::new(vector_type.clone());
        let mut true_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = true_builder.add_input(reference_type.clone().into());
        let value = true_builder.add_input(vector_type.clone().into());
        let doubled = true_builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![value, value], None)
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![reference, doubled],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut false_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = false_builder.add_input(reference_type.into());
        let value = false_builder.add_input(vector_type.into());
        let false_branch = false_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![reference, value],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            TestEagerContext::new(),
            TestIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let reference = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let values = Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let outputs = context
            .bind(
                TestIrOperation::Condition(ConditionOperation::new()),
                vec![true_branch, false_branch],
                &[
                    BatchingTracer::new(
                        context.clone(),
                        ArrayIrBatch::new(
                            TestIrValue::Array(Array::vector(vec![true, false]).unwrap()),
                            BatchAxis::new(0),
                        )
                        .unwrap(),
                    ),
                    BatchingTracer::new(
                        context.clone(),
                        ArrayIrBatch::new(TestIrValue::Reference(reference.clone()), BatchAxis::replicated()).unwrap(),
                    ),
                    BatchingTracer::new(
                        context.clone(),
                        ArrayIrBatch::new(TestIrValue::Array(values), BatchAxis::new(0)).unwrap(),
                    ),
                ],
            )
            .unwrap();
        assert_eq!(
            outputs.into_iter().map(BatchingTracer::into_batch).collect::<Vec<_>>(),
            vec![
                ArrayIrBatch::replicated(TestIrValue::Reference(reference.clone())),
                ArrayIrBatch::new(
                    array(Array::matrix(2, 3, vec![2.0f32, 4.0, 6.0, 4.0, 5.0, 6.0]).unwrap()),
                    BatchAxis::new(0),
                )
                .unwrap(),
            ],
        );
        assert_eq!(reference.read(), Ok(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()));
    }

    #[test]
    fn test_condition_batching_array_ir_aligns_manual_variation() {
        // Array IR batching gives the invariant Boolean predicate the variation of the array branch values in the same
        // way as array batching, once for both the gating and the merging selections.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying_sharding = |rank| Sharding::replicated(mesh.clone(), rank).with_varying_manual_axes(["x"]).unwrap();
        let predicate_type = ArrayType::new_static(DataType::Boolean, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1))
            .unwrap();
        let value_type = ArrayType::new_static(DataType::F64, [2]).with_sharding(varying_sharding(1)).unwrap();
        let branch_type = ArrayType::scalar(DataType::F64).with_sharding(varying_sharding(0)).unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let input = builder.add_input(branch_type.clone().into());
        let sine = builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(SinOperation::new())),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let sine_branch = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![sine], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let input = builder.add_input(branch_type.into());
        let identity_branch = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let (output_type, program) = TracingContext::<TestIrValue, TestIrOperation>::trace_with_named_axes(
            |inputs: Vec<TestTracer>| {
                let parent = inputs[0].domain();
                let extent = parent.constant(TestIrValue::Dimension(DimensionValue::constant(2).unwrap()));
                let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent, extent);
                let predicate =
                    BatchingTracer::new(context.clone(), ArrayIrBatch::new(inputs[0].clone(), BatchAxis::new(0))?);
                let value =
                    BatchingTracer::new(context.clone(), ArrayIrBatch::new(inputs[1].clone(), BatchAxis::new(0))?);
                let outputs = context.bind(
                    TestIrOperation::Condition(ConditionOperation::new()),
                    vec![sine_branch, identity_branch],
                    &[predicate, value],
                )?;
                Ok(outputs.into_iter().next().unwrap().into_batch().into_value())
            },
            vec![ArrayIrType::Array(predicate_type), ArrayIrType::Array(value_type.clone())],
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh })],
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::Array(value_type));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[2][sharding={mesh<['x'=2:manual]>, [{}]}], \
                    %1:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                let %2:dimension<2> = const 2
                    %3:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = stop_gradient %1
                    %4:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %0
                    %5:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %4 %1 %3
                    %6:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sin %5
                    %7:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %4 %3 %1
                    %8:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %4 %6 %7
                in (%8)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_condition_batching_array_ir_after_reference_discharge() {
        // Discharge across a condition: the mapped predicate turns the condition into a select over both discharged
        // branch states while the batched program stays pure and reference-free.
        let program = state_condition_program();

        let axis_extent = DimensionValue::constant(2).unwrap();
        let batched = program
            .discharge_references(0)
            .unwrap()
            .into_program_without_external_references()
            .unwrap()
            .batched_with_threaded_extent(
                axis_extent.r#type().into_owned(),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(batched.output_axes(), &[BatchAxis::new(0), BatchAxis::new(0)]);
        let (batched, _) = batched.into_parts();
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:dimension<2>, %1:bool[2], %2:f32[2] .
                let %3:f32[2] = stop_gradient %2
                    %4:f32[2] = select %1 %2 %3
                    %5:f32[] = const 1.0
                    %6:f32[2] = broadcast [output_axes=[]] %5 %0
                    %7:f32[2] = add %4 %6
                    %8:f32[2] = select %1 %3 %2
                    %9:f32[2] = select %1 %7 %8
                    %10:f32[] = const 9.0
                    %11:f32[2] = broadcast [output_axes=[]] %10 %0
                    %12:f32[2] = select %1 %7 %11
                in (%0, %9, %12)"},
        );
        assert!(batched.effects().classes().is_empty());
        assert!(batched.regions().iter().flat_map(|region| region.atoms()).all(|atom| !atom.r#type().is_reference()));

        // Mixed predicates pin that each batch item selects its own branch's state: the accumulating true branch
        // yields `4 + 1` for both outputs, while the overwriting false branch yields the pre-swap `7` snapshot and
        // the replacement `9` as the final state.
        assert_eq!(
            batched
                .interpret(vec![
                    TestIrValue::Dimension(axis_extent.clone()),
                    TestIrValue::Array(Array::vector(vec![true, false]).unwrap()),
                    TestIrValue::Array(Array::vector(vec![4.0f32, 7.0]).unwrap()),
                ])
                .unwrap(),
            vec![
                TestIrValue::Dimension(axis_extent),
                TestIrValue::Array(Array::vector(vec![5.0f32, 7.0]).unwrap()),
                TestIrValue::Array(Array::vector(vec![5.0f32, 9.0]).unwrap()),
            ],
        );
    }

    #[test]
    fn test_condition_differentiation() {
        // Finite differences stay away from the predicate boundary and check the derivative of each selected branch.
        check_gradient!(
            |input| stage_runtime_predicate_condition(input, [scalar_scale_branch(2.0), scalar_scale_branch(3.0)]),
            at = Array::scalar(4.0f64).unwrap(),
            step = 1e-6,
            tolerance = 1e-6,
        );
        check_gradient!(
            |input| stage_runtime_predicate_condition(input, [scalar_scale_branch(2.0), scalar_scale_branch(3.0)]),
            at = Array::scalar(-4.0f64).unwrap(),
            step = 1e-6,
            tolerance = 1e-6,
        );
    }

    #[test]
    fn test_condition_differentiation_linearization_replays_the_selected_branch() {
        // Linearization replays the selected branch, and its pushforward applies that branch's derivative.
        for (predicate, expected_value, expected_tangent) in
            [(true, 1.4, 3.0), (false, 0.7f64.sin(), 1.5 * 0.7f64.cos())]
        {
            let (value, pushforward) = differentiate_at(Array::scalar(0.7).unwrap())
                .linearize(move |input| {
                    let predicate = input.context().lift(
                        Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[predicate]).unwrap(),
                    )?;
                    let mut outputs = input.context().bind(
                        ArrayOperation::Condition(ConditionOperation::new()),
                        vec![
                            scalar_branch(ArrayOperation::Add(AddOperation::new())),
                            scalar_branch(ArrayOperation::Sin(SinOperation::new())),
                        ],
                        &[predicate, input.clone()],
                    )?;
                    Ok(outputs.remove(0))
                })
                .unwrap();
            assert_eq!(value, Array::scalar(expected_value).unwrap());
            assert_eq!(pushforward.apply(Array::scalar(1.5).unwrap()), Ok(Array::scalar(expected_tangent).unwrap()));
        }
    }

    #[test]
    fn test_condition_differentiation_jacobians_replay_the_selected_branch() {
        // Dense forward and reverse Jacobians replay the branch the runtime predicate selects.
        let forward = differentiate_at(Array::scalar(4.0).unwrap())
            .jacobian_forward(|input| {
                stage_runtime_predicate_condition(input, [scalar_scale_branch(2.0), scalar_scale_branch(3.0)])
            })
            .unwrap();
        let reverse = differentiate_at(Array::scalar(4.0).unwrap())
            .jacobian_reverse(|input| {
                stage_runtime_predicate_condition(input, [scalar_scale_branch(2.0), scalar_scale_branch(3.0)])
            })
            .unwrap();
        assert_eq!(forward.into_values(), vec![Array::scalar(2.0f64).unwrap()]);
        assert_eq!(reverse.into_values(), vec![Array::scalar(2.0f64).unwrap()]);

        let forward = differentiate_at(Array::scalar(-4.0).unwrap())
            .jacobian_forward(|input| {
                stage_runtime_predicate_condition(input, [scalar_scale_branch(2.0), scalar_scale_branch(3.0)])
            })
            .unwrap();
        let reverse = differentiate_at(Array::scalar(-4.0).unwrap())
            .jacobian_reverse(|input| {
                stage_runtime_predicate_condition(input, [scalar_scale_branch(2.0), scalar_scale_branch(3.0)])
            })
            .unwrap();
        assert_eq!(forward.into_values(), vec![Array::scalar(3.0f64).unwrap()]);
        assert_eq!(reverse.into_values(), vec![Array::scalar(3.0f64).unwrap()]);
    }

    #[test]
    fn test_condition_differentiation_pullback_selects_the_branch_cotangent() {
        // The pullback of a condition selects the cotangent of the runtime branch, and the predicate gets none.
        let (output, pullback) = differentiate_at((
            Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[true]).unwrap(),
            Array::scalar(4.0).unwrap(),
        ))
        .vjp(|(predicate, branch_input)| {
            let mut outputs = predicate.context().bind(
                ArrayOperation::Condition(ConditionOperation::new()),
                vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)],
                &[predicate.clone(), branch_input],
            )?;
            Ok(outputs.remove(0))
        })
        .unwrap();
        let cotangents = pullback.apply(Array::scalar(5.0).unwrap()).unwrap();
        assert_eq!(output, Array::scalar(8.0f64).unwrap());
        assert_eq!(
            cotangents,
            (MaybeZero::Zero(ArrayType::scalar(DataType::Zero)), MaybeZero::Value(Array::scalar(10.0f64).unwrap())),
        );

        let (output, pullback) = differentiate_at((
            Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[false]).unwrap(),
            Array::scalar(4.0).unwrap(),
        ))
        .vjp(|(predicate, branch_input)| {
            let mut outputs = predicate.context().bind(
                ArrayOperation::Condition(ConditionOperation::new()),
                vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)],
                &[predicate.clone(), branch_input],
            )?;
            Ok(outputs.remove(0))
        })
        .unwrap();
        let cotangents = pullback.apply(Array::scalar(5.0).unwrap()).unwrap();
        assert_eq!(output, Array::scalar(12.0f64).unwrap());
        assert_eq!(
            cotangents,
            (MaybeZero::Zero(ArrayType::scalar(DataType::Zero)), MaybeZero::Value(Array::scalar(15.0f64).unwrap())),
        );
    }

    #[test]
    fn test_condition_differentiation_validates_direct_branch_output_signatures() {
        let true_branch = scalar_branch(ArrayOperation::Add(AddOperation::new()));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        builder.add_input(ArrayType::scalar(DataType::F64));
        let false_branch = builder.build::<Vec<Array>, Vec<Array>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        assert_eq!(
            true_branch.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = add %0 %0
                in (%1)"},
        );
        assert_eq!(
            false_branch.to_string(),
            indoc! {"
                lambda %0:f64[] .
                in ()"},
        );

        // Each standalone branch is valid, but the condition's output signatures disagree. A live tangent dispatches
        // directly to its JVP rule, which must validate the primitive boundary before joining compact tangent outputs.
        let result = differentiate_at(Array::scalar(2f64).unwrap()).jvp(Array::scalar(1f64).unwrap(), |input| {
            let context = input.context().clone();
            let predicate = context.lift(Array::scalar(true).unwrap())?;
            context.bind(ConditionOperation::new(), vec![true_branch, false_branch], &[predicate, input])
        });
        assert!(matches!(
            result,
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "`condition` branch output type signature mismatch: expected [f64[]] but got []",
        ));

        // Forward-mode differentiation specializes attached regions before applying its rule, so it also rejects
        // branches that declare different input types even when the actual input refines both to the same type.
        let branches = mismatched_input_branches();
        let expected_error = TypeError::invalid(
            "`condition` branch input type signature mismatch: expected [f64[extent]] but got [f64[3]]",
        );
        for extent in [3, 4] {
            let input = Array::vector(vec![1.0f64; extent]).unwrap();
            let result = differentiate_at(input.clone()).jvp(input, |input| {
                let context = input.context().clone();
                let predicate = context.lift(Array::scalar(true).unwrap())?;
                Ok(context.bind(ConditionOperation::new(), branches.clone(), &[predicate, input])?.remove(0))
            });
            assert!(matches!(
                result,
                Err(DifferentiationError::Program(ProgramError::Type(error))) if error == expected_error,
            ));
        }
    }

    #[test]
    fn test_condition_differentiation_compares_refined_branch_outputs() {
        // The JVP rule compares the branch outputs after refining them by the actual inputs: an `f64[3]` input makes
        // the identity and constant branches agree, while an `f64[4]` input makes them disagree for either predicate.
        let input = Array::vector(vec![1.0f64; 3]).unwrap();
        let tangent = Array::vector(vec![1.0f64; 3]).unwrap();
        assert_eq!(
            differentiate_at(input.clone()).jvp(tangent.clone(), |input| {
                let context = input.context().clone();
                let predicate = context.lift(Array::scalar(false).unwrap())?;
                Ok(context
                    .bind(ConditionOperation::new(), refinable_dynamic_branches(), &[predicate, input])?
                    .remove(0))
            }),
            Ok((Array::vector(vec![2.0f64; 3]).unwrap(), Array::vector(vec![0.0f64; 3]).unwrap())),
        );
        assert_eq!(
            differentiate_at(input.clone()).jvp(tangent.clone(), |input| {
                let context = input.context().clone();
                let predicate = context.lift(Array::scalar(true).unwrap())?;
                Ok(context
                    .bind(ConditionOperation::new(), refinable_dynamic_branches(), &[predicate, input])?
                    .remove(0))
            }),
            Ok((input, tangent)),
        );

        let input = Array::vector(vec![1.0f64; 4]).unwrap();
        let tangent = Array::vector(vec![1.0f64; 4]).unwrap();
        for predicate in [false, true] {
            let result = differentiate_at(input.clone()).jvp(tangent.clone(), |input| {
                let context = input.context().clone();
                let predicate = context.lift(Array::scalar(predicate).unwrap())?;
                Ok(context
                    .bind(ConditionOperation::new(), refinable_dynamic_branches(), &[predicate, input])?
                    .remove(0))
            });
            assert!(matches!(
                result,
                Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                    if message
                        == "`condition` branch output type signature mismatch: expected [f64[4]] but got [f64[3]]",
            ));
        }
    }

    #[test]
    fn test_condition_differentiation_accepts_eager_reference_refinements() {
        // Identical branches forward a `ref<f32[n]>` input. Eager execution accepts a live `ref<f32[3]>` for it, as a
        // refinement of its symbolic allocation type, and so must linearization, whose primal context is eager.
        let n_type = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![n_type.dimension().clone()]));
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        builder.add_input(n_type.clone().into());
        let reference = builder.add_input(ReferenceType::new(vector_type).into());
        let branch = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let reference = TestIrValue::Reference(ArrayReference::new(Array::vector(vec![1f32, 2.0, 3.0]).unwrap()));
        let inputs = vec![array(Array::scalar(true).unwrap()), dimension(&n_type, 3), reference.clone()];
        assert_eq!(
            TestEagerContext::new().bind(ConditionOperation::new(), vec![branch.clone(), branch.clone()], &inputs),
            Ok(vec![reference.clone()]),
        );
        let (outputs, _) = differentiate_at(inputs)
            .linearize(|inputs: Vec<LinearizationTracer<TestEagerContext>>| {
                inputs[0].context().bind(
                    TestIrOperation::Condition(ConditionOperation::new()),
                    vec![branch.clone(), branch.clone()],
                    &inputs,
                )
            })
            .unwrap();
        assert_eq!(outputs, vec![reference]);
    }

    #[test]
    fn test_condition_differentiation_linearization_accepts_unit_returning_branches() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let branch = || {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            builder.add_input(scalar_type.clone());
            builder.build::<Vec<Array>, Vec<Array>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let input = builder.add_input(scalar_type.clone());
        let true_region = builder.import_program(branch());
        let false_region = builder.import_program(branch());
        builder
            .add_instruction(ConditionOperation::new(), vec![true_region, false_region], vec![predicate, input], None)
            .unwrap();
        let program = builder.build::<Vec<Array>, Vec<Array>>(Vec::new(), vec![Placeholder; 2], Vec::new()).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[] .
                let () = condition %0 %1 [
                    true={
                        lambda %0:f64[] .
                        in ()
                    },
                    false={
                        lambda %0:f64[] .
                        in ()
                    },
                ]
                in ()"},
        );

        // The active numeric input makes linearization replay the operation through its separate-context JVP rule,
        // even though neither branch returns any values. No primal or tangent result is required in this case.
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 0);
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[] .
                in ()"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[] .
                in ()"},
        );
        for predicate in [true, false] {
            assert_eq!(
                linearization
                    .primal()
                    .interpret(vec![Array::scalar(predicate).unwrap(), Array::scalar(3.0).unwrap()]),
                Ok(vec![]),
            );
        }
        assert_eq!(linearization.tangent().interpret(vec![Array::scalar(1.0).unwrap()]), Ok(vec![]));
    }

    #[test]
    fn test_condition_differentiation_keeps_reference_only_reset_tangents_structural() {
        // A reference makes dependence conservative, but both branches reset the numeric output to a constant, so its
        // tangent remains a structural zero when linearizing with respect to the reference only.
        let vector_type = ArrayType::new_static(DataType::F32, [3]);
        let reference_type = ReferenceType::new(vector_type.clone());
        let branch = || {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            builder.add_input(vector_type.clone().into());
            let reference = builder.add_input(reference_type.clone().into());
            let zero = builder
                .add_instruction(ZeroOperation::new(vector_type.clone()), Vec::new(), Vec::new(), None)
                .unwrap()[0];
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                    vec![zero, reference],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(vector_type.clone().into());
        let reference = builder.add_input(reference_type.clone().into());
        let true_branch = builder.import_program(branch());
        let false_branch = builder.import_program(branch());
        let outputs = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, input, reference],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        let linearization = program.linearize_with_respect_to(&[2]).unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[3], %2:ref<f32[3]> .
                let %3:f32[3], %4:ref<f32[3]> = condition %0 %1 %2 [
                    true={
                        lambda %0:f32[3], %1:ref<f32[3]> .
                        let %2:f32[3] = zero [type=f32[3]]
                        in (%2, %1)
                    },
                    false={
                        lambda %0:f32[3], %1:ref<f32[3]> .
                        let %2:f32[3] = zero [type=f32[3]]
                        in (%2, %1)
                    },
                ]
                in (%3, %4, %0)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:ref<f32[3]>, %1:bool[] .
                let %2:f32[3] = zero [type=f32[3]]
                    %3:ref<f32[3]> = condition %1 %0 [
                        true={
                            lambda %0:ref<f32[3]> .
                            in (%0)
                        },
                        false={
                            lambda %0:ref<f32[3]> .
                            in (%0)
                        },
                    ]
                in (%2, %3)"},
        );
    }

    #[test]
    fn test_condition_differentiation_linearization_preserves_dynamic_residual_extents() {
        let (program, extent_type, _) = dynamic_extent_condition_program();

        // Separate primal and tangent contexts require the untaken branch's residual slots to retain the same live
        // extent. Only the selected branch consumes its residuals, so extent-aware zero placeholders are sufficient.
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:bool[], %1:dimension<extent ∈ [1, 8)>, %2:f64[extent] .
                let %3:f64[extent], %4:f64[extent] = condition %0 %1 %2 [
                    true={
                        lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[extent] .
                        let %2:f64[extent] = mul %1 %1
                        in (%2, %1)
                    },
                    false={
                        lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[extent] .
                        let %2:f64[extent] = add %1 %1
                            %3:f64[extent] = zero [type=f64[extent]] %0
                        in (%2, %3)
                    },
                ]
                in (%3, %0, %4)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[extent], %1:bool[], %2:f64[extent] .
                let %3:f64[extent] = condition %1 %0 %2 [
                    true={
                        lambda %0:f64[extent], %1:f64[extent] .
                        let %2:f64[extent] = mul %1 %0
                            %3:f64[extent] = mul %1 %0
                            %4:f64[extent] = add %2 %3
                        in (%4)
                    },
                    false={
                        lambda %0:f64[extent], %1:f64[extent] .
                        let %2:f64[extent] = add %0 %0
                        in (%2)
                    },
                ]
                in (%3)"},
        );
        assert_eq!(
            linearization.pullback().unwrap().program().to_string(),
            indoc! {"
                lambda %0:f64[extent], %1:bool[], %2:f64[extent] .
                let %3:f64[extent] = condition %1 %0 %2 [
                    true={
                        lambda %0:f64[extent], %1:f64[extent] .
                        let %2:f64[extent] = mul %1 %0
                            %3:f64[extent] = mul %1 %0
                            %4:f64[extent] = add %2 %3
                        in (%4)
                    },
                    false={
                        lambda %0:f64[extent], %1:f64[extent] .
                        let %2:f64[extent] = add %0 %0
                        in (%2)
                    },
                ]
                in (%3)"},
        );
        let pullback = linearization.pullback().unwrap().into_program();
        for (predicate, expected_value, expected_gradient) in
            [(true, vec![4.0f64, 9.0, 16.0], vec![4.0f64, 6.0, 8.0]), (false, vec![4.0f64, 6.0, 8.0], vec![2.0f64; 3])]
        {
            let mut outputs = linearization
                .primal()
                .interpret(vec![
                    array(Array::scalar(predicate).unwrap()),
                    dimension(&extent_type, 3),
                    array(Array::vector(vec![2.0f64, 3.0, 4.0]).unwrap()),
                ])
                .unwrap();
            let residuals = outputs.split_off(1);
            assert_eq!(outputs, vec![array(Array::vector(expected_value).unwrap())]);
            let mut pullback_inputs = vec![array(Array::vector(vec![1.0f64; 3]).unwrap())];
            pullback_inputs.extend(residuals);
            assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![array(Array::vector(expected_gradient).unwrap())]));
        }
    }

    #[test]
    fn test_condition_differentiation_linearization_forwards_symbolic_residual_placeholders() {
        // Only the `true` branch retains its `f32[n]` input as a residual, so the `false` branch needs a placeholder
        // for that edge. Homogeneous array operations can only construct zeros from types, which cannot supply the
        // runtime extent of `n`, so the placeholder forwards the `false` branch input of the identical type instead.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![n.dimension().clone()]));
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(predicate, input): (ArrayTracer, ArrayTracer)| {
                predicate.condition(|input| Ok(input.clone() * input), |input| Ok(input), input)
            },
            (ArrayType::scalar(DataType::Boolean), vector_type),
        )
        .unwrap();
        let linearization = program.into_flat_program().linearize_with_respect_to(&[1]).unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[n] .
                let %2:f32[n], %3:f32[n] = condition %0 %1 [
                    true={
                        lambda %0:f32[n] .
                        let %1:f32[n] = mul %0 %0
                        in (%1, %0)
                    },
                    false={
                        lambda %0:f32[n] .
                        in (%0, %0)
                    },
                ]
                in (%2, %0, %3)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[n], %1:bool[], %2:f32[n] .
                let %3:f32[n] = condition %1 %0 %2 [
                    true={
                        lambda %0:f32[n], %1:f32[n] .
                        let %2:f32[n] = mul %1 %0
                            %3:f32[n] = mul %1 %0
                            %4:f32[n] = add %2 %3
                        in (%4)
                    },
                    false={
                        lambda %0:f32[n], %1:f32[n] .
                        in (%0)
                    },
                ]
                in (%3)"},
        );
        let pullback = linearization.pullback().unwrap().into_program();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[n], %1:bool[], %2:f32[n] .
                let %3:f32[n] = condition %1 %0 %2 [
                    true={
                        lambda %0:f32[n], %1:f32[n] .
                        let %2:f32[n] = mul %1 %0
                            %3:f32[n] = mul %1 %0
                            %4:f32[n] = add %2 %3
                        in (%4)
                    },
                    false={
                        lambda %0:f32[n], %1:f32[n] .
                        in (%0)
                    },
                ]
                in (%3)"},
        );

        // Both branches differentiate at every extent of `n`. When the `false` branch runs, the pullback never consumes
        // the placeholder that it forwarded.
        for (predicate, input, expected_value, expected_gradient) in [
            (true, vec![2f32, 3.0], vec![4f32, 9.0], vec![4f32, 6.0]),
            (false, vec![2f32, 3.0], vec![2f32, 3.0], vec![1f32, 1.0]),
            (true, vec![2f32, 3.0, 4.0], vec![4f32, 9.0, 16.0], vec![4f32, 6.0, 8.0]),
            (false, vec![2f32, 3.0, 4.0], vec![2f32, 3.0, 4.0], vec![1f32, 1.0, 1.0]),
        ] {
            let extent = input.len();
            let mut outputs = linearization
                .primal()
                .interpret(vec![Array::scalar(predicate).unwrap(), Array::vector(input).unwrap()])
                .unwrap();
            let residuals = outputs.split_off(1);
            assert_eq!(outputs, vec![Array::vector(expected_value).unwrap()]);
            let mut pullback_inputs = vec![Array::vector(vec![1f32; extent]).unwrap()];
            pullback_inputs.extend(residuals);
            assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![Array::vector(expected_gradient).unwrap()]));
        }
    }

    #[test]
    fn test_condition_differentiation_linearization_rejects_invariant_residuals_under_varying_predicates() {
        // The `true` branch varies `sin(w)` of an invariant weight `w` before using it, so its linearization saves the
        // invariant residual `cos(w)`. Every residual is an output of the primal condition, whose outputs must vary
        // over the predicate's manual axes, so separate-context linearization reports that residual.
        let (predicate_type, invariant_type, varying_type) = devices_varying_predicate_types();
        let branch = |uses_weight: bool| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let weight = builder.add_input(invariant_type.clone());
            let input = builder.add_input(varying_type.clone());
            let output = if uses_weight {
                let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![weight], None).unwrap()[0];
                let varied = builder
                    .add_instruction(ParallelVaryOperation::new("devices".to_string()), Vec::new(), vec![sine], None)
                    .unwrap()[0];
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![varied, input], None).unwrap()[0]
            } else {
                input
            };
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let predicate = builder.add_input(predicate_type.clone());
        let weight = builder.add_input(invariant_type.clone());
        let input = builder.add_input(varying_type.clone());
        let true_region = builder.import_program(branch(true));
        let false_region = builder.import_program(branch(false));
        let output = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_region, false_region],
                vec![predicate, weight, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda %0:{predicate_type}, %1:{invariant_type}, %2:{varying_type} .
                let %3:{varying_type} = condition %0 %1 %2 [
                    true={{
                        lambda %0:{invariant_type}, %1:{varying_type} .
                        let %2:{invariant_type} = sin %0
                            %3:{varying_type} = parallel_vary [axis_name=\"devices\"] %2
                            %4:{varying_type} = mul %3 %1
                        in (%4)
                    }},
                    false={{
                        lambda %0:{invariant_type}, %1:{varying_type} .
                        in (%1)
                    }},
                ]
                in (%3)"},
        );
        assert_eq!(
            program.linearize_with_respect_to(&[1]).map(|_| ()),
            Err(DifferentiationError::from(ProgramError::UnsupportedOperation {
                message: format!(
                    "`condition` linearization cannot join the residuals of its branches, because a residual of type \
                     `{invariant_type}` does not vary over every manual axis that the predicate `{predicate_type}` \
                     varies over",
                ),
            })),
        );
    }

    #[test]
    fn test_condition_differentiation_passes_plumbing_references_into_branches() {
        // Both branches read the reference they receive at input position 1, after a numeric input, and add it to
        // that numeric input: `f(p, x, r) = x + read(r)`.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let branch = || {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let value = builder.add_input(scalar_type.clone().into());
            let reference = builder.add_input(reference_type.clone().into());
            let current =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            let output = builder
                .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![value, current], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_branch = builder.import_region(branch().entry_region_ref());
        let false_branch = builder.import_region(branch().entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let value = builder.add_input(scalar_type.clone().into());
        let reference = builder.add_input(reference_type.into());
        let output = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_branch, false_branch],
                vec![predicate, value, reference],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        // With the reference inactive it reaches the branches as plumbing at a non-prefix position: the branches
        // receive no tangent input for it (`[x, r, ẋ]`), the read's tangent is zero, and the program still
        // differentiates with `ẏ = ẋ`. The fused program is spliced behind local allocations of the reference inputs so
        // that the interpreted program owns the state that its branches read.
        let jvp = program.entry_region_ref().jvp(&[1]).unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[], %2:ref<f32[]>, %3:f32[] .
                let %4:f32[], %5:f32[] = condition %0 %1 %2 %3 [
                    true={
                        lambda %0:f32[], %1:ref<f32[]>, %2:f32[] .
                        let %3:f32[] = reference_read %1
                            %4:f32[] = add %0 %3
                        in (%4, %2)
                    },
                    false={
                        lambda %0:f32[], %1:ref<f32[]>, %2:f32[] .
                        let %3:f32[] = reference_read %1
                            %4:f32[] = add %0 %3
                        in (%4, %2)
                    },
                ]
                in (%4, %5)"},
        );
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let value = builder.add_input(scalar_type.clone().into());
        let state = builder.add_input(scalar_type.clone().into());
        let tangent = builder.add_input(scalar_type.clone().into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
        let outputs = builder.splice_program(&jvp, &[predicate, value, reference, tangent]).unwrap();
        let runnable = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 4], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            runnable.interpret(vec![
                TestIrValue::Array(Array::scalar(true).unwrap()),
                TestIrValue::Array(Array::scalar(2.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(5.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(3.0f32).unwrap()),
            ]),
            Ok(vec![
                TestIrValue::Array(Array::scalar(7.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(3.0f32).unwrap()),
            ]),
        );

        // With the reference active the branches receive its tangent reference too (`[x, r, ẋ, ṛ]`) and the read's
        // tangent is the referenced tangent: `ẏ = ẋ + read(ṛ)`.
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[], %2:ref<f32[]>, %3:f32[], %4:ref<f32[]> .
                let %5:f32[], %6:f32[] = condition %0 %1 %2 %3 %4 [
                    true={
                        lambda %0:f32[], %1:ref<f32[]>, %2:f32[], %3:ref<f32[]> .
                        let %4:f32[] = reference_read %1
                            %5:f32[] = reference_read %3
                            %6:f32[] = add %0 %4
                            %7:f32[] = add %2 %5
                        in (%6, %7)
                    },
                    false={
                        lambda %0:f32[], %1:ref<f32[]>, %2:f32[], %3:ref<f32[]> .
                        let %4:f32[] = reference_read %1
                            %5:f32[] = reference_read %3
                            %6:f32[] = add %0 %4
                            %7:f32[] = add %2 %5
                        in (%6, %7)
                    },
                ]
                in (%5, %6)"},
        );
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let value = builder.add_input(scalar_type.clone().into());
        let state = builder.add_input(scalar_type.clone().into());
        let tangent = builder.add_input(scalar_type.clone().into());
        let state_tangent = builder.add_input(scalar_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
        let reference_tangent = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![state_tangent], None)
            .unwrap()[0];
        let outputs = builder.splice_program(&jvp, &[predicate, value, reference, tangent, reference_tangent]).unwrap();
        let runnable = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 5], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            runnable.interpret(vec![
                TestIrValue::Array(Array::scalar(false).unwrap()),
                TestIrValue::Array(Array::scalar(2.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(5.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(3.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(0.5f32).unwrap()),
            ]),
            Ok(vec![
                TestIrValue::Array(Array::scalar(7.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(3.5f32).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_condition_differentiation_forwards_inactive_reference_outputs() {
        // Both branches forward the reference they receive, and the program reads through the conditional's result,
        // so the public output is numeric while the branch outputs are references.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let branch = || {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_branch = builder.import_region(branch().entry_region_ref());
        let false_branch = builder.import_region(branch().entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let reference = builder.add_input(reference_type.into());
        let forwarded = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![forwarded], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // An inactive reference is forwarded through the branch without a tangent slot. Reading it produces a
        // numeric structural zero, which is materialized at the outer output boundary.
        let inactive_jvp = program.entry_region_ref().jvp(&[]).unwrap();
        assert_eq!(
            inactive_jvp.to_string(),
            indoc! {"
                lambda %0:bool[], %1:ref<f32[]> .
                let %2:ref<f32[]> = condition %0 %1 [
                    true={
                        lambda %0:ref<f32[]> .
                        in (%0)
                    },
                    false={
                        lambda %0:ref<f32[]> .
                        in (%0)
                    },
                ]
                    %3:f32[] = reference_read %2
                    %4:f32[] = zero [type=f32[]]
                in (%3, %4)"},
        );
        assert_eq!(
            inactive_jvp.interpret(vec![
                TestIrValue::Array(Array::scalar(true).unwrap()),
                TestIrValue::Reference(ArrayReference::new(Array::scalar(5.0f32).unwrap())),
            ]),
            Ok(vec![
                TestIrValue::Array(Array::scalar(5.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(0.0f32).unwrap()),
            ]),
        );

        // An active reference is forwarded together with its tangent reference, and the read pairs them back up. The
        // fused program is spliced behind local allocations of the reference inputs so that the interpreted program
        // owns the state that it reads.
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:bool[], %1:ref<f32[]>, %2:ref<f32[]> .
                let %3:ref<f32[]>, %4:ref<f32[]> = condition %0 %1 %2 [
                    true={
                        lambda %0:ref<f32[]>, %1:ref<f32[]> .
                        in (%0, %1)
                    },
                    false={
                        lambda %0:ref<f32[]>, %1:ref<f32[]> .
                        in (%0, %1)
                    },
                ]
                    %5:f32[] = reference_read %3
                    %6:f32[] = reference_read %4
                in (%5, %6)"},
        );
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let state = builder.add_input(scalar_type.clone().into());
        let state_tangent = builder.add_input(scalar_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
        let reference_tangent = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![state_tangent], None)
            .unwrap()[0];
        let outputs = builder.splice_program(&jvp, &[predicate, reference, reference_tangent]).unwrap();
        let runnable = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            runnable.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[], %2:f32[] .
                let %3:ref<f32[]> = reference_new %1
                    %4:ref<f32[]> = reference_new %2
                    %5:ref<f32[]>, %6:ref<f32[]> = condition %0 %3 %4 [
                        true={
                            lambda %0:ref<f32[]>, %1:ref<f32[]> .
                            in (%0, %1)
                        },
                        false={
                            lambda %0:ref<f32[]>, %1:ref<f32[]> .
                            in (%0, %1)
                        },
                    ]
                    %7:f32[] = reference_read %5
                    %8:f32[] = reference_read %6
                in (%7, %8)"},
        );
        assert_eq!(
            runnable.interpret(vec![
                TestIrValue::Array(Array::scalar(true).unwrap()),
                TestIrValue::Array(Array::scalar(5.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(0.5f32).unwrap()),
            ]),
            Ok(vec![
                TestIrValue::Array(Array::scalar(5.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(0.5f32).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_condition_differentiation_rejects_branch_dependent_reference_roots() {
        // Matching reference types do not imply matching reference roots. These distinct roots already fail
        // reference analysis, and differentiation rejects them by their roots whichever inputs are active, without
        // manufacturing a tangent reference for an inactive root or joining compact branch outputs that carry tangents
        // of different references.
        let program = reference_forwarding_condition_program(&[0], &[1]);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:ref<f32[]>, %2:ref<f32[]> .
                let %3:ref<f32[]> = condition %0 %1 %2 [
                    true={
                        lambda %0:ref<f32[]>, %1:ref<f32[]> .
                        in (%0)
                    },
                    false={
                        lambda %0:ref<f32[]>, %1:ref<f32[]> .
                        in (%1)
                    },
                ]
                in (%3)"},
        );
        assert!(matches!(
            program.reference_analysis(0),
            Err(ReferenceAnalysisError::ReferenceRootMismatch { output_index: 0, region_index: 1, .. }),
        ));
        let root_mismatch = |output: usize, true_input: usize, false_input: usize| {
            DifferentiationError::Program(ProgramError::Type(TypeError::invalid(format!(
                "`condition` reference output {output} must denote the same reference in both branches, but the \
                 `true` branch returns region input {true_input} and the `false` branch returns region input \
                 {false_input}",
            ))))
        };
        for input_indices in [&[1][..], &[2], &[1, 2]] {
            assert_eq!(program.jvp_with_respect_to(input_indices).map(|_| ()), Err(root_mismatch(0, 0, 1)));
        }
        let reversed = reference_forwarding_condition_program(&[1], &[0]);
        for input_indices in [&[1][..], &[2], &[1, 2]] {
            assert_eq!(reversed.jvp_with_respect_to(input_indices).map(|_| ()), Err(root_mismatch(0, 1, 0)));
        }

        // Both branches can have one compact tangent output while assigning it to different primal output positions.
        // Comparing only their tangent counts would silently attach the active reference tangent to the wrong output.
        let swapped = reference_forwarding_condition_program(&[0, 1], &[1, 0]);
        assert_eq!(
            swapped.to_string(),
            indoc! {"
                lambda %0:bool[], %1:ref<f32[]>, %2:ref<f32[]> .
                let %3:ref<f32[]>, %4:ref<f32[]> = condition %0 %1 %2 [
                    true={
                        lambda %0:ref<f32[]>, %1:ref<f32[]> .
                        in (%0, %1)
                    },
                    false={
                        lambda %0:ref<f32[]>, %1:ref<f32[]> .
                        in (%1, %0)
                    },
                ]
                in (%3, %4)"},
        );
        for input_indices in [&[1][..], &[2], &[1, 2]] {
            assert_eq!(swapped.jvp_with_respect_to(input_indices).map(|_| ()), Err(root_mismatch(0, 0, 1)));
        }
        let reversed_swapped = reference_forwarding_condition_program(&[1, 0], &[0, 1]);
        assert_eq!(reversed_swapped.jvp_with_respect_to(&[1]).map(|_| ()), Err(root_mismatch(0, 1, 0)));
    }

    #[test]
    fn test_condition_differentiation_rejects_mutations_under_varying_predicates() {
        // Only the `true` branch mutates the reference input, whose referent does not vary over the predicate's axis.
        // Differentiation validates the mutations of both branches before deriving either branch program, so it rejects
        // the condition that interpretation and discharge reject.
        let program = manual_mutation_program(false);
        let predicate = "bool[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]";
        let invariant = "f32[2][sharding={mesh<['x'=2:manual]>, [{}]}]";
        let varying = "f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]";
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda %0:{predicate}, %1:ref<{invariant}>, %2:{varying} .
                let %3:{varying} = condition %0 %1 %2 [
                    true={{
                        lambda %0:ref<{invariant}>, %1:{varying} .
                        let %2:{invariant} = reference_read %0
                            %3:{invariant} = add %2 %2
                            () = reference_write %0 %3
                        in (%1)
                    }},
                    false={{
                        lambda %0:ref<{invariant}>, %1:{varying} .
                        let %2:{invariant} = reference_read %0
                        in (%1)
                    }},
                ]
                in (%3)"},
        );
        let expected = Err(DifferentiationError::Program(ProgramError::Type(TypeError::invalid(
            "`condition` `true` branch mutates region input 0 of type \
             `ref<f32[2][sharding={mesh<['x'=2:manual]>, [{}]}]>`, whose referent does not vary over every manual \
             axis that the predicate `bool[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]` varies \
             over, so devices that take different branches would leave it different across devices; vary the \
             referent over those axes",
        ))));
        assert_eq!(program.jvp_with_respect_to(&[2]).map(|_| ()), expected);
        assert_eq!(program.linearize_with_respect_to(&[2]).map(|_| ()), expected);

        // A referent that varies along `x` may be mutated under the same predicate.
        assert_eq!(
            manual_mutation_program(true).jvp_with_respect_to(&[2]).unwrap().to_string(),
            formatdoc! {"
                lambda %0:{predicate}, %1:ref<{varying}>, %2:{varying}, %3:{varying} .
                let %4:{varying}, %5:{varying} = condition %0 %1 %2 %3 [
                    true={{
                        lambda %0:ref<{varying}>, %1:{varying}, %2:{varying} .
                        let %3:{varying} = reference_read %0
                            %4:{varying} = add %3 %3
                            () = reference_write %0 %4
                        in (%1, %2)
                    }},
                    false={{
                        lambda %0:ref<{varying}>, %1:{varying}, %2:{varying} .
                        let %3:{varying} = reference_read %0
                        in (%1, %2)
                    }},
                ]
                in (%4, %5)"},
        );
    }

    #[test]
    fn test_condition_differentiation_after_local_reference_discharge() {
        let source = state_condition_program();

        // Forward mode, linearization, and transposition all consume the discharged program, so every derived
        // program must be pure and reference-free even though the source threads state through both branches.
        let jvp = source
            .clone()
            .discharge_references(0)
            .unwrap()
            .into_program_without_external_references()
            .unwrap()
            .jvp()
            .unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[], %2:f32[] .
                let %3:f32[], %4:f32[], %5:f32[], %6:f32[] = condition %0 %1 %2 [
                    true={
                        lambda %0:f32[], %1:f32[] .
                        let %2:f32[] = const 1.0
                            %3:f32[] = add %0 %2
                        in (%3, %3, %1, %1)
                    },
                    false={
                        lambda %0:f32[], %1:f32[] .
                        let %2:f32[] = const 9.0
                            %3:f32[] = zero [type=f32[]]
                        in (%0, %2, %1, %3)
                    },
                ]
                in (%3, %4, %5, %6)"},
        );
        let linearization = source
            .discharge_references(0)
            .unwrap()
            .into_program_without_external_references()
            .unwrap()
            .linearize()
            .unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[] .
                let %2:f32[], %3:f32[] = condition %0 %1 [
                    true={
                        lambda %0:f32[] .
                        let %1:f32[] = const 1.0
                            %2:f32[] = add %0 %1
                        in (%2, %2)
                    },
                    false={
                        lambda %0:f32[] .
                        let %1:f32[] = const 9.0
                        in (%0, %1)
                    },
                ]
                in (%2, %3, %0)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[], %1:bool[] .
                let %2:f32[], %3:f32[] = condition %1 %0 [
                    true={
                        lambda %0:f32[] .
                        in (%0, %0)
                    },
                    false={
                        lambda %0:f32[] .
                        let %1:f32[] = zero [type=f32[]]
                        in (%0, %1)
                    },
                ]
                in (%2, %3)"},
        );
        assert_eq!(
            linearization.pullback().unwrap().program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[], %2:bool[] .
                let %3:f32[] = condition %2 %0 %1 [
                    true={
                        lambda %0:f32[], %1:f32[] .
                        let %2:f32[] = add %0 %1
                        in (%2)
                    },
                    false={
                        lambda %0:f32[], %1:f32[] .
                        in (%0)
                    },
                ]
                in (%3)"},
        );
        let pullback = linearization.pullback().unwrap().into_program();
        for program in [&jvp, linearization.primal(), linearization.tangent(), &pullback] {
            assert!(!program.entry_region_ref().contains_atom_type_in_closure(Type::is_reference));
            assert!(program.effects().classes().is_empty());
        }

        // The true branch accumulates the input, so both public outputs remain differentiable.
        let predicate = TestIrValue::Array(Array::scalar(true).unwrap());
        let initial = TestIrValue::Array(Array::scalar(4.0f32).unwrap());
        assert_eq!(
            jvp.interpret(vec![predicate.clone(), initial.clone(), TestIrValue::Array(Array::scalar(2.0f32).unwrap())]),
            Ok(vec![
                TestIrValue::Array(Array::scalar(5.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(5.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(2.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(2.0f32).unwrap()),
            ]),
        );
        let primal_outputs = linearization.primal().interpret(vec![predicate, initial]).unwrap();
        assert_eq!(
            primal_outputs[..2],
            [TestIrValue::Array(Array::scalar(5.0f32).unwrap()), TestIrValue::Array(Array::scalar(5.0f32).unwrap())],
        );
        let mut pullback_inputs = vec![
            TestIrValue::Array(Array::scalar(2.0f32).unwrap()),
            TestIrValue::Array(Array::scalar(3.0f32).unwrap()),
        ];
        pullback_inputs.extend_from_slice(&primal_outputs[2..]);
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![TestIrValue::Array(Array::scalar(5.0f32).unwrap())]));

        // The false branch replaces the state with a constant, so the frozen output has zero tangent and contributes
        // no cotangent to the input.
        let predicate = TestIrValue::Array(Array::scalar(false).unwrap());
        let initial = TestIrValue::Array(Array::scalar(4.0f32).unwrap());
        assert_eq!(
            jvp.interpret(vec![predicate.clone(), initial.clone(), TestIrValue::Array(Array::scalar(2.0f32).unwrap())]),
            Ok(vec![
                TestIrValue::Array(Array::scalar(4.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(9.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(2.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(0.0f32).unwrap()),
            ]),
        );
        let primal_outputs = linearization.primal().interpret(vec![predicate, initial]).unwrap();
        assert_eq!(
            primal_outputs[..2],
            [TestIrValue::Array(Array::scalar(4.0f32).unwrap()), TestIrValue::Array(Array::scalar(9.0f32).unwrap())],
        );
        let mut pullback_inputs = vec![
            TestIrValue::Array(Array::scalar(2.0f32).unwrap()),
            TestIrValue::Array(Array::scalar(3.0f32).unwrap()),
        ];
        pullback_inputs.extend_from_slice(&primal_outputs[2..]);
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![TestIrValue::Array(Array::scalar(2.0f32).unwrap())]));
    }

    #[test]
    fn test_condition_differentiation_keeps_tangents_of_inactive_outputs_structural() {
        // The true branch maps `(a, b, c)` to `(sin(a), sin(c))` and the false branch maps it to `(sin(b), sin(c))`.
        // Only `a` has a live tangent, so the first output's tangent is live because the true branch depends on it,
        // while the second output depends on `a` in neither branch and keeps a structural-zero tangent. The branch
        // derivatives therefore take one tangent input and return one tangent output, and linearization succeeds.
        let scalar = ArrayType::scalar(DataType::F64);
        let branch = |first: usize| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let inputs = (0..3).map(|_| builder.add_input(scalar.clone())).collect::<Vec<_>>();
            let outputs = [inputs[first], inputs[2]]
                .map(|input| builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0]);
            builder
                .build::<Vec<Array>, Vec<Array>>(outputs.to_vec(), vec![Placeholder; 3], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let inputs = (0..3).map(|_| builder.add_input(scalar.clone())).collect::<Vec<_>>();
        let true_branch = builder.import_program(branch(0));
        let false_branch = builder.import_program(branch(1));
        let outputs = builder
            .add_instruction(
                ConditionOperation::<ArrayType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, inputs[0], inputs[1], inputs[2]],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 4], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.jvp_with_respect_to(&[1]).unwrap().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[], %2:f64[], %3:f64[], %4:f64[] .
                let %5:f64[], %6:f64[], %7:f64[] = condition %0 %1 %2 %3 %4 [
                    true={
                        lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = sin %0
                            %5:f64[] = sin %2
                            %6:f64[] = cos %0
                            %7:f64[] = mul %6 %3
                        in (%4, %5, %7)
                    },
                    false={
                        lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = sin %1
                            %5:f64[] = sin %2
                            %6:f64[] = zero [type=f64[]]
                        in (%4, %5, %6)
                    },
                ]
                    %8:f64[] = zero [type=f64[]]
                in (%5, %6, %7, %8)
            "}
            .trim_end(),
        );

        let linearization = program.linearize_with_respect_to(&[1]).unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[], %2:f64[], %3:f64[] .
                let %4:f64[], %5:f64[], %6:f64[] = condition %0 %1 %2 %3 [
                    true={
                        lambda %0:f64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = sin %0
                            %4:f64[] = sin %2
                            %5:f64[] = cos %0
                        in (%3, %4, %5)
                    },
                    false={
                        lambda %0:f64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = sin %1
                            %4:f64[] = sin %2
                            %5:f64[] = zero [type=f64[]]
                        in (%3, %4, %5)
                    },
                ]
                in (%4, %5, %0, %6)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[], %1:bool[], %2:f64[] .
                let %3:f64[] = condition %1 %0 %2 [
                    true={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = mul %1 %0
                        in (%2)
                    },
                    false={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = zero [type=f64[]]
                        in (%2)
                    },
                ]
                    %4:f64[] = zero [type=f64[]]
                in (%3, %4)"},
        );
        for (predicate, expected_tangent) in [(true, 0.5f64.cos()), (false, 0.0)] {
            let mut primal_outputs = linearization
                .primal()
                .interpret(vec![
                    Array::scalar(predicate).unwrap(),
                    Array::scalar(0.5f64).unwrap(),
                    Array::scalar(1.5f64).unwrap(),
                    Array::scalar(2.5f64).unwrap(),
                ])
                .unwrap();
            let residuals = primal_outputs.split_off(2);
            let mut tangent_inputs = vec![Array::scalar(1f64).unwrap()];
            tangent_inputs.extend(residuals);
            assert_eq!(
                linearization.tangent().interpret(tangent_inputs),
                Ok(vec![Array::scalar(expected_tangent).unwrap(), Array::scalar(0f64).unwrap()]),
            );
        }
    }

    #[test]
    fn test_condition_differentiation_jvp_preserves_zero_space_output_tangents() {
        let (primal, tangent) = differentiate_at(Array::scalar(2.0).unwrap())
            .jvp(Array::scalar(3.0).unwrap(), |input| {
                let predicate = input
                    .context()
                    .lift(Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[true]).unwrap())?;
                let mut outputs = input.context().bind(
                    ArrayOperation::Condition(ConditionOperation::new()),
                    vec![boolean_branch(), boolean_branch()],
                    &[predicate, input.clone()],
                )?;
                Ok(outputs.remove(0))
            })
            .unwrap();
        assert_eq!(primal, Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[true]).unwrap());
        assert_eq!(tangent, Array::new(ArrayType::scalar(DataType::Zero), Vec::new()).unwrap());
    }

    #[test]
    fn test_condition_differentiation_reuses_shared_branch_transforms() {
        // The `condition` differentiation rules reach their branches through the per-`Region` transform cache, so
        // several programs attaching one shared pair of branches derive each branch's transposition once per linearity
        // mask, while staging exactly the programs the uncached path stages from independently built copies of the
        // same branches.
        /// Builds a program that applies a condition over the provided branches followed by `epilogue` sines, so that
        /// programs sharing one pair of branches still have distinct derived programs.
        fn conditional_program(
            true_branch: &Arc<Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>>,
            false_branch: &Arc<Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>>,
            epilogue: usize,
        ) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
            let branch_input = builder.add_input(ArrayType::scalar(DataType::F64));
            let regions = vec![
                builder.intern_callee(true_branch, None).unwrap(),
                builder.intern_callee(false_branch, None).unwrap(),
            ];
            let mut value = builder
                .add_instruction(
                    ArrayOperation::Condition(ConditionOperation::new()),
                    regions,
                    vec![predicate, branch_input],
                    None,
                )
                .unwrap()[0];
            for _ in 0..epilogue {
                value = builder.add_instruction(SinOperation::new(), Vec::new(), vec![value], None).unwrap()[0];
            }
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![value], vec![Placeholder, Placeholder], vec![Placeholder])
                .unwrap()
        }

        let true_branch = Arc::new(scalar_scale_branch(2.0));
        let false_branch = Arc::new(scalar_scale_branch(3.0));
        let first = conditional_program(&true_branch, &false_branch, 1).linearize().unwrap();
        let second = conditional_program(&true_branch, &false_branch, 2).linearize().unwrap();
        assert_ne!(first.tangent().to_string(), second.tangent().to_string());
        assert_eq!(
            first.primal().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[] .
                let %2:f64[] = condition %0 %1 [
                    true={
                        lambda %0:f64[] .
                        let %1:f64[] = const 2.0
                            %2:f64[] = mul %0 %1
                        in (%2)
                    },
                    false={
                        lambda %0:f64[] .
                        let %1:f64[] = const 3.0
                            %2:f64[] = mul %0 %1
                        in (%2)
                    },
                ]
                    %3:f64[] = sin %2
                    %4:f64[] = cos %2
                in (%3, %0, %4)"},
        );
        assert_eq!(
            first.tangent().to_string(),
            indoc! {"
                lambda %0:f64[], %1:bool[], %2:f64[] .
                let %3:f64[] = condition %1 %0 [
                    true={
                        lambda %0:f64[] .
                        let %1:f64[] = const 2.0
                            %2:f64[] = mul %1 %0
                        in (%2)
                    },
                    false={
                        lambda %0:f64[] .
                        let %1:f64[] = const 3.0
                            %2:f64[] = mul %1 %0
                        in (%2)
                    },
                ]
                    %4:f64[] = mul %2 %3
                in (%4)"},
        );
        assert_eq!(
            second.tangent().to_string(),
            indoc! {"
                lambda %0:f64[], %1:bool[], %2:f64[], %3:f64[] .
                let %4:f64[] = condition %1 %0 [
                    true={
                        lambda %0:f64[] .
                        let %1:f64[] = const 2.0
                            %2:f64[] = mul %1 %0
                        in (%2)
                    },
                    false={
                        lambda %0:f64[] .
                        let %1:f64[] = const 3.0
                            %2:f64[] = mul %1 %0
                        in (%2)
                    },
                ]
                    %5:f64[] = mul %2 %4
                    %6:f64[] = mul %3 %5
                in (%6)"},
        );

        // Independently built copies of the same branches share no retained transforms, so they exercise the uncached
        // path and pin that caching changed nothing about what is staged.
        let uncached = conditional_program(&Arc::new(scalar_scale_branch(2.0)), &Arc::new(scalar_scale_branch(3.0)), 1)
            .linearize()
            .unwrap();
        assert_eq!(first.primal().to_string(), uncached.primal().to_string());
        assert_eq!(first.tangent().to_string(), uncached.tangent().to_string());
        assert_eq!(first.residual_count(), uncached.residual_count());

        // Transposing the tangent program twice transposes its condition's branches once: the second pass is served
        // from the branch regions' retained transpositions and produces the identical pullback.
        // Build a fresh outer transpose on both calls so each reaches the nested region's cache. These static
        // tangent types need no residual dimension mappings for zeros; trailing residual inputs remain known.
        let tangent_input_indices = (0..first.tangent().input_ids().len() - first.residual_count()).collect::<Vec<_>>();
        let pullback = first
            .tangent()
            .entry_region_ref()
            .transpose(&tangent_input_indices, &vec![true; tangent_input_indices.len()], &[])
            .unwrap()
            .into_program();
        let repeated = first
            .tangent()
            .entry_region_ref()
            .transpose(&tangent_input_indices, &vec![true; tangent_input_indices.len()], &[])
            .unwrap()
            .into_program();
        assert_eq!(pullback.to_string(), repeated.to_string());
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[], %1:bool[], %2:f64[] .
                let %3:f64[] = mul %2 %0
                    %4:f64[] = condition %1 %3 [
                        true={
                            lambda %0:f64[] .
                            let %1:f64[] = const 2.0
                                %2:f64[] = mul %1 %0
                            in (%2)
                        },
                        false={
                            lambda %0:f64[] .
                            let %1:f64[] = const 3.0
                                %2:f64[] = mul %1 %0
                            in (%2)
                        },
                    ]
                in (%4)"},
        );
        let tangent_condition = first
            .tangent()
            .instructions()
            .iter()
            .find(|instruction| matches!(instruction.operation(), ArrayOperation::Condition(_)))
            .unwrap();
        for region in tangent_condition.regions() {
            let statistics = transposition_statistics(first.tangent().region_ref(*region).unwrap()).unwrap();
            assert_eq!((statistics.productions, statistics.hits), (1, 1));
        }
        assert_eq!(
            pullback.to_string(),
            uncached
                .tangent()
                .entry_region_ref()
                .transpose(
                    &(0..uncached.tangent().input_ids().len() - uncached.residual_count()).collect::<Vec<_>>(),
                    &vec![true; uncached.tangent().input_ids().len() - uncached.residual_count()],
                    &[],
                )
                .unwrap()
                .program()
                .to_string(),
        );
    }

    #[test]
    fn test_condition_differentiation_refines_outputs_of_unspecialized_branches() {
        // Differentiating a condition whose refined output comes from unspecialized branches keeps the primal, tangent,
        // and cotangent types refined. The true branch computes `x * x + y`.
        let program = refined_vector_condition_program();
        let predicate = array(Array::scalar(true).unwrap());
        let input = array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());
        let other_input = array(Array::vector(vec![10.0, 20.0, 30.0]).unwrap());

        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[3], %2:f64[3], %3:f64[3], %4:f64[3] .
                let %5:f64[3], %6:f64[3] = condition %0 %1 %2 %3 %4 [
                    true={
                        lambda %0:f64[3], %1:f64[3], %2:f64[3], %3:f64[3] .
                        let %4:f64[3] = mul %0 %0
                            %5:f64[3] = mul %0 %2
                            %6:f64[3] = mul %0 %2
                            %7:f64[3] = add %5 %6
                            %8:f64[3] = add %4 %1
                            %9:f64[3] = add %7 %3
                        in (%8, %9)
                    },
                    false={
                        lambda %0:f64[3], %1:f64[3], %2:f64[3], %3:f64[3] .
                        let %4:f64[3] = add %0 %1
                            %5:f64[3] = add %2 %3
                        in (%4, %5)
                    },
                ]
                in (%5, %6)"},
        );
        assert!(jvp.output_types().iter().all(|r#type| r#type.identities().next().is_none()));
        assert_eq!(
            jvp.interpret(vec![
                predicate.clone(),
                input.clone(),
                other_input.clone(),
                array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap()),
                array(Array::vector(vec![0.0, 0.0, 1.0]).unwrap()),
            ]),
            Ok(vec![
                array(Array::vector(vec![11.0, 24.0, 39.0]).unwrap()),
                array(Array::vector(vec![2.0, 4.0, 7.0]).unwrap()),
            ]),
        );

        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[3], %2:f64[3] .
                let %3:f64[3], %4:f64[3] = condition %0 %1 %2 [
                    true={
                        lambda %0:f64[3], %1:f64[3] .
                        let %2:f64[3] = mul %0 %0
                            %3:f64[3] = add %2 %1
                        in (%3, %0)
                    },
                    false={
                        lambda %0:f64[3], %1:f64[3] .
                        let %2:f64[3] = add %0 %1
                            %3:f64[3] = zero [type=f64[3]]
                        in (%2, %3)
                    },
                ]
                in (%3, %0, %4)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3], %2:bool[], %3:f64[3] .
                let %4:f64[3] = condition %2 %0 %1 %3 [
                    true={
                        lambda %0:f64[3], %1:f64[3], %2:f64[3] .
                        let %3:f64[3] = mul %2 %0
                            %4:f64[3] = mul %2 %0
                            %5:f64[3] = add %3 %4
                            %6:f64[3] = add %5 %1
                        in (%6)
                    },
                    false={
                        lambda %0:f64[3], %1:f64[3], %2:f64[3] .
                        let %3:f64[3] = add %0 %1
                        in (%3)
                    },
                ]
                in (%4)"},
        );
        assert_eq!(
            linearization.pullback().unwrap().program().to_string(),
            indoc! {"
                lambda %0:f64[3], %1:bool[], %2:f64[3] .
                let %3:f64[3], %4:f64[3] = condition %1 %0 %2 [
                    true={
                        lambda %0:f64[3], %1:f64[3] .
                        let %2:f64[3] = mul %1 %0
                            %3:f64[3] = mul %1 %0
                            %4:f64[3] = add %2 %3
                        in (%4, %0)
                    },
                    false={
                        lambda %0:f64[3], %1:f64[3] .
                        in (%0, %0)
                    },
                ]
                in (%3, %4)"},
        );
        let mut primal_outputs = linearization.primal().interpret(vec![predicate, input, other_input]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let pullback = linearization.pullback().unwrap().into_program();
        assert!(pullback.output_types().iter().all(|r#type| r#type.identities().next().is_none()));
        let mut pullback_inputs = vec![array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            pullback.interpret(pullback_inputs),
            Ok(vec![
                array(Array::vector(vec![2.0, 4.0, 6.0]).unwrap()),
                array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_condition_differentiation_preserves_dimension_outputs() {
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let input = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
        let true_branch = scale_branch(extent_type.clone(), 2.0);
        let false_branch = scale_branch(extent_type.clone(), 3.0);
        let regions = vec![
            builder.import_region(true_branch.entry_region_ref()),
            builder.import_region(false_branch.entry_region_ref()),
        ];
        let outputs = builder
            .add_instruction(
                TestIrOperation::Condition(ConditionOperation::new()),
                regions,
                vec![predicate, extent, input],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder.build(outputs, vec![Placeholder; 3], vec![Placeholder; 2]).unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:bool[], %1:dimension<extent ∈ [1, 8)>, %2:f64[], %3:f64[] .
                let %4:dimension<extent ∈ [1, 8)>, %5:f64[], %6:f64[] = condition %0 %1 %2 %3 [
                    true={
                        lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[], %2:f64[] .
                        let %3:f64[] = const 2.0
                            %4:f64[] = mul %1 %3
                            %5:f64[] = mul %3 %2
                        in (%0, %4, %5)
                    },
                    false={
                        lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[], %2:f64[] .
                        let %3:f64[] = const 3.0
                            %4:f64[] = mul %1 %3
                            %5:f64[] = mul %3 %2
                        in (%0, %4, %5)
                    },
                ]
                in (%4, %5, %6)"},
        );
        assert_eq!(jvp.input_count(), 4);
        assert_eq!(jvp.output_count(), 3);
        assert_eq!(
            jvp.interpret(vec![
                array(Array::scalar(true).unwrap()),
                dimension(&extent_type, 4),
                array(Array::scalar(5.0).unwrap()),
                array(Array::scalar(7.0).unwrap()),
            ]),
            Ok(vec![
                dimension(&extent_type, 4),
                array(Array::scalar(10.0).unwrap()),
                array(Array::scalar(14.0).unwrap()),
            ]),
        );

        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:bool[], %1:dimension<extent ∈ [1, 8)>, %2:f64[] .
                let %3:dimension<extent ∈ [1, 8)>, %4:f64[] = condition %0 %1 %2 [
                    true={
                        lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[] .
                        let %2:f64[] = const 2.0
                            %3:f64[] = mul %1 %2
                        in (%0, %3)
                    },
                    false={
                        lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[] .
                        let %2:f64[] = const 3.0
                            %3:f64[] = mul %1 %2
                        in (%0, %3)
                    },
                ]
                in (%3, %4, %0)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[], %1:bool[] .
                let %2:f64[] = condition %1 %0 [
                    true={
                        lambda %0:f64[] .
                        let %1:f64[] = const 2.0
                            %2:f64[] = mul %1 %0
                        in (%2)
                    },
                    false={
                        lambda %0:f64[] .
                        let %1:f64[] = const 3.0
                            %2:f64[] = mul %1 %0
                        in (%2)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(
            linearization.pullback().unwrap().program().to_string(),
            indoc! {"
                lambda %0:f64[], %1:bool[] .
                let %2:f64[] = condition %1 %0 [
                    true={
                        lambda %0:f64[] .
                        let %1:f64[] = const 2.0
                            %2:f64[] = mul %1 %0
                        in (%2)
                    },
                    false={
                        lambda %0:f64[] .
                        let %1:f64[] = const 3.0
                            %2:f64[] = mul %1 %0
                        in (%2)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(linearization.residual_count(), 1);
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                array(Array::scalar(true).unwrap()),
                dimension(&extent_type, 4),
                array(Array::scalar(5.0).unwrap()),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(2);
        let mut pullback_inputs = vec![array(Array::scalar(1.0).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().program().interpret(pullback_inputs),
            Ok(vec![array(Array::scalar(2.0).unwrap())]),
        );
    }

    #[test]
    fn test_condition_differentiation_jvp_materializes_dynamic_zero_output_tangent() {
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![extent_type.dimension().clone()]));
        let program = dynamic_zero_condition_program(&extent_type);

        // Both inputs have zero tangent spaces, but the dynamic floating-point result does not. Its zero tangent must
        // therefore consume the selected primal result's explicit runtime extent instead of using a nullary zero.
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:bool[], %1:dimension<extent ∈ [1, 8)> .
                let %2:f64[extent] = condition %0 %1 [
                    true={
                        lambda %0:dimension<extent ∈ [1, 8)> .
                        let %1:f64[extent] = zero [type=f64[extent]] %0
                        in (%1)
                    },
                    false={
                        lambda %0:dimension<extent ∈ [1, 8)> .
                        let %1:f64[extent] = zero [type=f64[extent]] %0
                        in (%1)
                    },
                ]
                    %3:dimension<extent ∈ [1, 8)> = dimension_size [axis=0] %2
                    %4:f64[extent] = zero [type=f64[extent]] %3
                in (%2, %4)"},
        );
        assert_eq!(jvp.input_count(), 2);
        assert_eq!(jvp.output_count(), 2);
        assert_eq!(
            jvp.interpret(vec![
                array(Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[true]).unwrap()),
                dimension(&extent_type, 3),
            ]),
            Ok(vec![array(Array::vector(vec![0.0f64; 3]).unwrap()), array(Array::vector(vec![0.0f64; 3]).unwrap())]),
        );

        // Eager direct JVP derives the concrete output tangent extent from the selected primal result at the public
        // boundary.
        let (primal, tangent) = differentiate_at(vec![
            array(Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[true]).unwrap()),
            dimension(&extent_type, 3),
        ])
        .jvp(
            vec![
                array(Array::new(ArrayType::scalar(DataType::Zero), Vec::new()).unwrap()),
                array(Array::new(ArrayType::scalar(DataType::Zero), Vec::new()).unwrap()),
            ],
            |inputs| {
                let context = inputs[0].context().clone();
                context.bind(
                    ConditionOperation::new(),
                    dynamic_zero_condition_branches(&extent_type).to_vec(),
                    inputs.as_slice(),
                )
            },
        )
        .unwrap();
        assert_eq!(primal, vec![array(Array::vector(vec![0.0f64; 3]).unwrap())]);
        assert_eq!(tangent, vec![array(Array::vector(vec![0.0f64; 3]).unwrap())]);

        // Direct transform dispatch must make the same decision before it has a staged instruction whose result type
        // it can inspect. The condition rule retains the selected branch's extent and constructs the tangent there.
        let context = TracingContext::<TestIrValue, TestIrOperation>::new();
        let predicate = context.input(ArrayType::scalar(DataType::Boolean).into());
        let extent = context.input(extent_type.clone().into());
        let predicate_tangent = context.input(ArrayType::scalar(DataType::Zero).into());
        let extent_tangent = context.input(ArrayType::scalar(DataType::Zero).into());
        let (_, tangent) = context
            .jvp(
                |inputs, ()| {
                    let context = inputs[0].context().clone();
                    Ok(context
                        .bind(
                            ConditionOperation::new(),
                            dynamic_zero_condition_branches(&extent_type).to_vec(),
                            inputs.as_slice(),
                        )?
                        .remove(0))
                },
                vec![predicate, extent],
                vec![predicate_tangent, extent_tangent],
                (),
            )
            .unwrap();
        assert_eq!(tangent.r#type().as_ref(), &ArrayIrType::Array(output_type.clone()));
    }

    #[test]
    fn test_condition_differentiation_linearization_materializes_dynamic_zero_output_tangent() {
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![extent_type.dimension().clone()]));
        let program = dynamic_zero_condition_program(&extent_type);

        // Split program linearization stages the same extent read on the primal side and forces the shaped zero into
        // the tangent program, rather than folding it into an affine known tangent.
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:bool[], %1:dimension<extent ∈ [1, 8)> .
                let %2:f64[extent] = condition %0 %1 [
                    true={
                        lambda %0:dimension<extent ∈ [1, 8)> .
                        let %1:f64[extent] = zero [type=f64[extent]] %0
                        in (%1)
                    },
                    false={
                        lambda %0:dimension<extent ∈ [1, 8)> .
                        let %1:f64[extent] = zero [type=f64[extent]] %0
                        in (%1)
                    },
                ]
                    %3:dimension<extent ∈ [1, 8)> = dimension_size [axis=0] %2
                in (%2, %3)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:dimension<extent ∈ [1, 8)> .
                let %1:f64[extent] = zero [type=f64[extent]] %0
                in (%1)"},
        );
        assert_eq!(linearization.residual_count(), 1);
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                array(Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[true]).unwrap()),
                dimension(&extent_type, 3),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        assert_eq!(
            linearization.tangent().interpret(residuals),
            Ok(vec![array(Array::vector(vec![0.0f64; 3]).unwrap())]),
        );

        // Reusable linearization follows the same ordinary region rule and closes over the dynamic result shape;
        // applying its null linear map therefore reconstructs the shaped tangent without a type-only zero.
        let context = TracingContext::<TestIrValue, TestIrOperation>::new();
        let predicate = context.input(ArrayType::scalar(DataType::Boolean).into());
        let extent = context.input(extent_type.clone().into());
        let (_, pushforward) = context
            .linearize(
                |inputs, ()| {
                    let context = inputs[0].context().clone();
                    Ok(context
                        .bind(
                            ConditionOperation::new(),
                            dynamic_zero_condition_branches(&extent_type).to_vec(),
                            inputs.as_slice(),
                        )?
                        .remove(0))
                },
                vec![predicate, extent],
                (),
            )
            .unwrap();
        let predicate_tangent = context.input(ArrayType::scalar(DataType::Zero).into());
        let extent_tangent = context.input(ArrayType::scalar(DataType::Zero).into());
        assert_eq!(
            pushforward.apply(vec![predicate_tangent, extent_tangent]).unwrap().r#type().as_ref(),
            &ArrayIrType::Array(output_type),
        );
    }

    #[test]
    fn test_condition_differentiation_jvp_receives_projection_materialized_zero_tangents() {
        // The `stop_gradient` projection owns the shaped-zero materialization; no test in this module owns it.
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let array_type = ArrayType::new(DataType::F64, Shape::new(vec![extent_type.dimension().clone()]));
        let branch = || {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
            let left = builder.add_input(ArrayIrType::Array(array_type.clone()));
            let right = builder.add_input(ArrayIrType::Array(array_type.clone()));
            let sum = builder
                .add_instruction(
                    TestIrOperation::Array(ArrayOperation::from(AddOperation::new())),
                    Vec::new(),
                    vec![left, right],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                    vec![extent, sum],
                    vec![Placeholder; 3],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let left = builder.add_input(ArrayIrType::Array(array_type.clone()));
        let right = builder.add_input(ArrayIrType::Array(array_type.clone()));

        // In the array IR family, the projection of `stop_gradient` materializes the severed tangent of the last
        // instruction input as a zero shaped by its primal's runtime extent, so that tangent reaches the condition rule
        // as a live value rather than as a structural zero.
        let severed = builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(StopGradientOperation::<ArrayType>::new())),
                Vec::new(),
                vec![right],
                None,
            )
            .unwrap()[0];
        let regions = vec![
            builder.import_region(branch().entry_region_ref()),
            builder.import_region(branch().entry_region_ref()),
        ];
        let outputs = builder
            .add_instruction(
                TestIrOperation::Condition(ConditionOperation::new()),
                regions,
                vec![predicate, extent, left, severed],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![outputs[1]], vec![Placeholder; 4], vec![Placeholder])
            .unwrap();

        // The projection, not the condition rule, reads the severed primal's runtime extent to construct the dynamic
        // zero, and the condition differentiates that materialized tangent like any other live tangent.
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:bool[], %1:dimension<extent ∈ [1, 8)>, %2:f64[extent], %3:f64[extent], \
                    %4:f64[extent], %5:f64[extent] .
                let %6:f64[extent] = stop_gradient %3
                    %7:dimension<extent ∈ [1, 8)> = dimension_size [axis=0] %6
                    %8:f64[extent] = zero [type=f64[extent]] %7
                    %9:dimension<extent ∈ [1, 8)>, %10:f64[extent], %11:f64[extent] = condition %0 %1 %2 %6 %4 %8 [
                        true={
                            lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[extent], %2:f64[extent], \
                                %3:f64[extent], %4:f64[extent] .
                            let %5:f64[extent] = add %1 %2
                                %6:f64[extent] = add %3 %4
                            in (%0, %5, %6)
                        },
                        false={
                            lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[extent], %2:f64[extent], \
                                %3:f64[extent], %4:f64[extent] .
                            let %5:f64[extent] = add %1 %2
                                %6:f64[extent] = add %3 %4
                            in (%0, %5, %6)
                        },
                    ]
                in (%10, %11)
            "}
            .trim_end(),
        );
        assert_eq!(
            jvp.interpret(vec![
                array(Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[true]).unwrap()),
                dimension(&extent_type, 3),
                array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap()),
                array(Array::vector(vec![10.0, 20.0, 30.0]).unwrap()),
                array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap()),
                array(Array::vector(vec![5.0, 5.0, 5.0]).unwrap()),
            ]),
            Ok(vec![
                array(Array::vector(vec![11.0, 22.0, 33.0]).unwrap()),
                array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_condition_differentiation_after_batching_blocks_inactive_non_finite_derivatives() {
        // The inactive square root produces NaN at -1 and an infinite derivative at zero. Input gradient barriers
        // must discard both contributions, while preserving the selected identity derivative for those items.
        let (batched, _) = square_root_or_identity_condition_program()
            .batched(3, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        let linearization = batched.linearize().unwrap();
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[] = const 0.0
                    %2:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %1
                    %3:bool[3] = compare [direction=GreaterThan] %0 %2
                    %4:f64[3] = stop_gradient %0
                    %5:f64[3] = select %3 %0 %4
                    %6:f64[3] = sqrt %5
                    %7:f64[3] = select %3 %4 %0
                    %8:f64[3] = select %3 %6 %7
                in (%8)"},
        );
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[] = const 0.0
                    %2:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %1
                    %3:bool[3] = compare [direction=GreaterThan] %0 %2
                    %4:f64[3] = stop_gradient %0
                    %5:f64[3] = select %3 %0 %4
                    %6:f64[3] = sqrt %5
                    %7:f64[3] = select %3 %4 %0
                    %8:f64[3] = select %3 %6 %7
                    %9:f64[3] = zero [type=f64[3]]
                    %10:f64[3] = add %6 %6
                    %11:f64[3] = zero [type=f64[3]]
                in (%8, %3, %9, %10, %11)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[3], %1:bool[3], %2:f64[3], %3:f64[3], %4:f64[3] .
                let %5:f64[3] = select %1 %0 %2
                    %6:f64[3] = div %5 %3
                    %7:f64[3] = select %1 %4 %0
                    %8:f64[3] = select %1 %6 %7
                in (%8)"},
        );
        assert_eq!(
            linearization.pullback().unwrap().program().to_string(),
            indoc! {"
                lambda %0:f64[3], %1:bool[3], %2:f64[3], %3:f64[3], %4:f64[3] .
                let %5:f64[3] = zero [type=f64[3]]
                    %6:f64[3] = select %1 %0 %5
                    %7:f64[3] = select %1 %5 %0
                    %8:f64[3] = zero [type=f64[3]]
                    %9:f64[3] = select %1 %8 %7
                    %10:f64[3] = div %6 %3
                    %11:f64[3] = zero [type=f64[3]]
                    %12:f64[3] = select %1 %10 %11
                    %13:f64[3] = add %9 %12
                in (%13)"},
        );

        let mut primal_outputs =
            linearization.primal().interpret(vec![Array::vector(vec![4.0f64, -1.0, 0.0]).unwrap()]).unwrap();
        let residuals = primal_outputs.split_off(1);
        assert_eq!(primal_outputs, vec![Array::vector(vec![2.0f64, -1.0, 0.0]).unwrap()]);
        let mut pullback_inputs = vec![Array::vector(vec![1.0f64, 1.0, 1.0]).unwrap()];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().program().interpret(pullback_inputs),
            Ok(vec![Array::vector(vec![0.25f64, 1.0, 1.0]).unwrap()]),
        );

        // Array IR batching threads the batch extent as a dimension input and blocks the same contributions.
        let program = square_root_or_identity_condition_program()
            .into_unprojected::<TestIrValue, TestIrOperation>()
            .unwrap();
        let extent = DimensionValue::constant(3).unwrap();
        let (batched, _) = program
            .batched_with_threaded_extent(
                extent.r#type().into_owned(),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        let linearization = batched.linearize().unwrap();
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:dimension<3>, %1:f64[3] .
                let %2:f64[] = const 0.0
                    %3:f64[3] = broadcast [output_axes=[]] %2 %0
                    %4:bool[3] = compare [direction=GreaterThan] %1 %3
                    %5:f64[3] = stop_gradient %1
                    %6:f64[3] = select %4 %1 %5
                    %7:f64[3] = sqrt %6
                    %8:f64[3] = select %4 %5 %1
                    %9:f64[3] = select %4 %7 %8
                in (%0, %9)"},
        );
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:dimension<3>, %1:f64[3] .
                let %2:f64[] = const 0.0
                    %3:f64[3] = broadcast [output_axes=[]] %2 %0
                    %4:bool[3] = compare [direction=GreaterThan] %1 %3
                    %5:f64[3] = stop_gradient %1
                    %6:f64[3] = select %4 %1 %5
                    %7:f64[3] = sqrt %6
                    %8:f64[3] = select %4 %5 %1
                    %9:f64[3] = select %4 %7 %8
                    %10:f64[3] = zero [type=f64[3]]
                    %11:f64[3] = add %7 %7
                    %12:f64[3] = zero [type=f64[3]]
                in (%0, %9, %4, %10, %11, %12)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[3], %1:bool[3], %2:f64[3], %3:f64[3], %4:f64[3] .
                let %5:f64[3] = select %1 %0 %2
                    %6:f64[3] = div %5 %3
                    %7:f64[3] = select %1 %4 %0
                    %8:f64[3] = select %1 %6 %7
                in (%8)"},
        );
        assert_eq!(
            linearization.pullback().unwrap().program().to_string(),
            indoc! {"
                lambda %0:f64[3], %1:bool[3], %2:f64[3], %3:f64[3], %4:f64[3] .
                let %5:f64[3] = zero [type=f64[3]]
                    %6:f64[3] = select %1 %0 %5
                    %7:f64[3] = select %1 %5 %0
                    %8:f64[3] = zero [type=f64[3]]
                    %9:f64[3] = select %1 %8 %7
                    %10:f64[3] = div %6 %3
                    %11:f64[3] = zero [type=f64[3]]
                    %12:f64[3] = select %1 %10 %11
                    %13:f64[3] = add %9 %12
                in (%13)"},
        );

        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                TestIrValue::Dimension(extent.clone()),
                array(Array::vector(vec![4.0f64, -1.0, 0.0]).unwrap()),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(2);
        assert_eq!(
            primal_outputs,
            vec![TestIrValue::Dimension(extent), array(Array::vector(vec![2.0f64, -1.0, 0.0]).unwrap())],
        );
        let mut pullback_inputs = vec![array(Array::vector(vec![1.0f64, 1.0, 1.0]).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().program().interpret(pullback_inputs),
            Ok(vec![array(Array::vector(vec![0.25f64, 1.0, 1.0]).unwrap())]),
        );
    }

    #[test]
    fn test_condition_differentiation_after_batching_with_a_replicated_predicate() {
        // Batching with a replicated predicate keeps one condition whose branches are batched, and differentiating the
        // batched program differentiates those batched branches behind the same predicate.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let true_region = builder.import_program(scalar_scale_branch(2.0));
        let false_region = builder.import_program(scalar_branch(ArrayOperation::Sin(SinOperation::new())));
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(ConditionOperation::new(), vec![true_region, false_region], vec![predicate, input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let (batched, _) = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[2] .
                let %2:f64[2] = condition %0 %1 [
                    true={
                        lambda %0:f64[2] .
                        let %1:f64[] = const 2.0
                            %2:f64[2] = broadcast [output_type=f64[2], output_axes=[]] %1
                            %3:f64[2] = mul %0 %2
                        in (%3)
                    },
                    false={
                        lambda %0:f64[2] .
                        let %1:f64[2] = sin %0
                        in (%1)
                    },
                ]
                in (%2)"},
        );
        let jvp = batched.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[2], %2:f64[2] .
                let %3:f64[2], %4:f64[2] = condition %0 %1 %2 [
                    true={
                        lambda %0:f64[2], %1:f64[2] .
                        let %2:f64[] = const 2.0
                            %3:f64[2] = broadcast [output_type=f64[2], output_axes=[]] %2
                            %4:f64[2] = mul %0 %3
                            %5:f64[2] = mul %3 %1
                        in (%4, %5)
                    },
                    false={
                        lambda %0:f64[2], %1:f64[2] .
                        let %2:f64[2] = sin %0
                            %3:f64[2] = cos %0
                            %4:f64[2] = mul %3 %1
                        in (%2, %4)
                    },
                ]
                in (%3, %4)"},
        );
        assert_eq!(
            jvp.interpret(vec![
                Array::scalar(true).unwrap(),
                Array::vector(vec![1.0, 2.0]).unwrap(),
                Array::vector(vec![3.0, 3.0]).unwrap(),
            ]),
            Ok(vec![Array::vector(vec![2.0, 4.0]).unwrap(), Array::vector(vec![6.0, 6.0]).unwrap()]),
        );
        assert_eq!(
            jvp.interpret(vec![
                Array::scalar(false).unwrap(),
                Array::vector(vec![1.0, 2.0]).unwrap(),
                Array::vector(vec![3.0, 3.0]).unwrap(),
            ]),
            Ok(vec![
                Array::vector(vec![1.0f64.sin(), 2.0f64.sin()]).unwrap(),
                Array::vector(vec![3.0 * 1.0f64.cos(), 3.0 * 2.0f64.cos()]).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_condition_differentiation_nested_selects_the_runtime_branch() {
        // Higher-order differentiation selects the runtime branch at every nesting level.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let square = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
        let true_branch =
            builder.build::<Vec<Array>, Vec<Array>>(vec![square], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let square = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
        let cube = builder.add_instruction(MulOperation::new(), Vec::new(), vec![square, input], None).unwrap()[0];
        let false_branch =
            builder.build::<Vec<Array>, Vec<Array>>(vec![cube], vec![Placeholder], vec![Placeholder]).unwrap();
        let branches = [true_branch, false_branch];

        // Away from the predicate boundary, the positive branch has derivatives (2x, 2), while the negative branch
        // has derivatives (3x², 6x). Both nesting orders must retain the branch choice and the derivative of its
        // captured primal residuals; the forward-over-reverse tangent also checks a non-unit direction.
        for (input, expected_gradient, expected_second_derivative) in [(2f64, 4f64, 2f64), (-2f64, 12f64, -12f64)] {
            let (gradient, second_derivative) = differentiate_at(Array::scalar(input).unwrap())
                .value_and_gradient(|input| {
                    differentiate_at(input)
                        .gradient(|input| stage_runtime_predicate_condition(input, branches.clone()))
                        .map_err(ProgramError::from)
                })
                .unwrap();
            assert_eq!(gradient, Array::scalar(expected_gradient).unwrap());
            assert_eq!(second_derivative, Array::scalar(expected_second_derivative).unwrap());

            let (gradient, tangent) = differentiate_at(Array::scalar(input).unwrap())
                .jvp(Array::scalar(3f64).unwrap(), |input| {
                    differentiate_at(input)
                        .gradient(|input| stage_runtime_predicate_condition(input, branches.clone()))
                        .map_err(ProgramError::from)
                })
                .unwrap();
            assert_eq!(gradient, Array::scalar(expected_gradient).unwrap());
            assert_eq!(tangent, Array::scalar(3.0 * expected_second_derivative).unwrap());

            check_gradient!(
                |input| differentiate_at(input)
                    .gradient(|input| stage_runtime_predicate_condition(input, branches.clone()))
                    .map_err(ProgramError::from),
                at = Array::scalar(input).unwrap(),
                step = 1e-6,
                tolerance = 1e-6,
            );
        }

        // The ignored vector is selected by the inner gradient but disconnected in both branches. The live scalar's
        // gradient still differentiates through the selected nonlinear branch in the enclosing forward pass.
        for (input, expected_gradient, expected_tangent) in [(2.0f64, 4.0f64, 6.0f64), (-2.0f64, 12.0f64, -36.0f64)] {
            let (gradient, tangent) = differentiate_at(Array::scalar(input).unwrap())
                .jvp(Array::scalar(3.0f64).unwrap(), |input| {
                    let ignored = input.context().lift(Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap())?;
                    let (_, gradient) = differentiate_at((ignored, input)).gradient(|(ignored, input)| {
                        let predicate = positive_predicate(&input)?;
                        predicate.condition(
                            |(_, value)| Ok(value.clone() * value),
                            |(_, value)| Ok(value.clone() * value.clone() * value),
                            (ignored, input),
                        )
                    })?;
                    Ok(gradient)
                })
                .unwrap();
            assert_eq!(gradient, Array::scalar(expected_gradient).unwrap());
            assert_eq!(tangent, Array::scalar(expected_tangent).unwrap());
        }

        // Reverse over reverse differentiates the same inner gradient a second time.
        for (input, expected) in [(2.0f64, 2.0f64), (-2.0f64, -12.0f64)] {
            let second = differentiate_at(Array::scalar(input).unwrap())
                .gradient(|input| {
                    let ignored = input.context().lift(Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap())?;
                    let (_, gradient) = differentiate_at((ignored, input)).gradient(|(ignored, input)| {
                        let predicate = positive_predicate(&input)?;
                        predicate.condition(
                            |(_, value)| Ok(value.clone() * value),
                            |(_, value)| Ok(value.clone() * value.clone() * value),
                            (ignored, input),
                        )
                    })?;
                    Ok(gradient)
                })
                .unwrap();
            assert_eq!(second, Array::scalar(expected).unwrap());
        }
    }

    #[test]
    fn test_condition_differentiation_batches_pullbacks_with_structural_zeros() {
        for (predicate, expected) in [(true, vec![4.0f64, 8.0, 12.0]), (false, vec![1.0f64, 2.0, 3.0])] {
            let cotangents = batch(
                |seed| {
                    let context = seed.context();
                    let ignored = context.lift(Array::vector(vec![7.0f64; 2]).unwrap())?;
                    let live = context.lift(Array::scalar(2.0f64).unwrap())?;
                    let predicate = context.lift(Array::scalar(predicate).unwrap())?;
                    let (_, pullback) = context.vjp(
                        |(ignored, live), predicate| {
                            predicate.condition(
                                |(_, live)| Ok(live.clone() * live),
                                |(_, live)| Ok(live),
                                (ignored, live),
                            )
                        },
                        (ignored, live),
                        predicate,
                    )?;
                    let (ignored, live) = pullback.apply(seed)?;
                    assert_eq!(ignored, MaybeZero::Zero(ArrayType::new_static(DataType::F64, [2])));
                    Ok(live.as_value().unwrap().clone())
                },
                Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
            assert_eq!(cotangents, Array::vector(expected).unwrap());
        }
    }

    #[test]
    fn test_condition_transposition() {
        // Builds a scalar condition over `[predicate, branch_inputs...]` whose branches sum the products of the
        // `(scale, linear)` branch input pairs in `pairs` and scale that sum by `1.0` in the `true` branch and by
        // `2.0` in the `false` branch, so that known scale inputs and linear inputs can be interleaved in any order.
        let interleaved_product_condition_program = |pairs: &[(usize, usize)]| {
            let input_count = pairs.len() * 2;
            let branch = |factor: f64| {
                let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
                let inputs =
                    (0..input_count).map(|_| builder.add_input(ArrayType::scalar(DataType::F64))).collect::<Vec<_>>();
                let products = pairs
                    .iter()
                    .map(|&(scale, linear)| {
                        builder
                            .add_instruction(MulOperation::new(), Vec::new(), vec![inputs[scale], inputs[linear]], None)
                            .unwrap()[0]
                    })
                    .collect::<Vec<_>>();
                let mut output = products[0];
                for product in &products[1..] {
                    output = builder
                        .add_instruction(AddOperation::new(), Vec::new(), vec![output, *product], None)
                        .unwrap()[0];
                }
                let factor = builder.add_constant(Array::scalar(factor).unwrap());
                let output =
                    builder.add_instruction(MulOperation::new(), Vec::new(), vec![output, factor], None).unwrap()[0];
                builder
                    .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; input_count], vec![Placeholder])
                    .unwrap()
            };
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let true_branch = builder.import_program(branch(1.0));
            let false_branch = builder.import_program(branch(2.0));
            let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
            let mut inputs = vec![predicate];
            inputs.extend((0..input_count).map(|_| builder.add_input(ArrayType::scalar(DataType::F64))));
            let outputs = builder
                .add_instruction(ConditionOperation::new(), vec![true_branch, false_branch], inputs, None)
                .unwrap()
                .to_vec();
            builder
                .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; input_count + 1], vec![Placeholder])
                .unwrap()
        };

        // A known scale precedes the linear input.
        let transposed = interleaved_product_condition_program(&[(0, 1)]).transpose_with_respect_to(&[2], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[], %1:bool[], %2:f64[] .
                let %3:f64[] = condition %1 %0 %2 [
                    true={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = const 1.0
                            %3:f64[] = mul %2 %0
                            %4:f64[] = mul %1 %3
                        in (%4)
                    },
                    false={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = const 2.0
                            %3:f64[] = mul %2 %0
                            %4:f64[] = mul %1 %3
                        in (%4)
                    },
                ]
                in (%3)"},
        );
        assert_eq!(
            transposed.interpret(vec![
                Array::scalar(3.0f64).unwrap(),
                Array::scalar(true).unwrap(),
                Array::scalar(5.0f64).unwrap(),
            ]),
            Ok(vec![Array::scalar(15.0f64).unwrap()]),
        );
        assert_eq!(
            transposed.interpret(vec![
                Array::scalar(3.0f64).unwrap(),
                Array::scalar(false).unwrap(),
                Array::scalar(5.0f64).unwrap(),
            ]),
            Ok(vec![Array::scalar(30.0f64).unwrap()]),
        );

        // Known scales and linear inputs alternate. The cotangents are requested in reverse input order to check that
        // the branch's source order is reassembled correctly.
        let transposed = interleaved_product_condition_program(&[(1, 0), (3, 2)])
            .transpose_with_respect_to(&[3, 1], &[])
            .unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[], %1:bool[], %2:f64[], %3:f64[] .
                let %4:f64[], %5:f64[] = condition %1 %0 %2 %3 [
                    true={
                        lambda %0:f64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = const 1.0
                            %4:f64[] = mul %3 %0
                            %5:f64[] = mul %2 %4
                            %6:f64[] = mul %1 %4
                        in (%6, %5)
                    },
                    false={
                        lambda %0:f64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = const 2.0
                            %4:f64[] = mul %3 %0
                            %5:f64[] = mul %2 %4
                            %6:f64[] = mul %1 %4
                        in (%6, %5)
                    },
                ]
                in (%5, %4)"},
        );
        assert_eq!(
            transposed.interpret(vec![
                Array::scalar(3.0f64).unwrap(),
                Array::scalar(true).unwrap(),
                Array::scalar(2.0f64).unwrap(),
                Array::scalar(5.0f64).unwrap(),
            ]),
            Ok(vec![Array::scalar(15.0f64).unwrap(), Array::scalar(6.0f64).unwrap()]),
        );
        assert_eq!(
            transposed.interpret(vec![
                Array::scalar(3.0f64).unwrap(),
                Array::scalar(false).unwrap(),
                Array::scalar(2.0f64).unwrap(),
                Array::scalar(5.0f64).unwrap(),
            ]),
            Ok(vec![Array::scalar(30.0f64).unwrap(), Array::scalar(12.0f64).unwrap()]),
        );
    }

    #[test]
    fn test_condition_transposition_omits_cotangents_disconnected_in_both_branches() {
        let extent = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.simplified_dimension()]));
        let scalar_type = ArrayType::scalar(DataType::F32);
        let branch = |negate| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            builder.add_input(array_type.clone());
            let value = builder.add_input(scalar_type.clone());
            let output = if negate {
                builder.add_instruction(NegOperation::new(), Vec::new(), vec![value], None).unwrap()[0]
            } else {
                value
            };
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let ignored = builder.add_input(array_type.clone());
        let value = builder.add_input(scalar_type.clone());
        let branches = vec![builder.import_program(branch(false)), builder.import_program(branch(true))];
        let output = builder
            .add_instruction(ConditionOperation::new(), branches, vec![predicate, ignored, value], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let transposed = program.entry_region_ref().transpose(&[1, 2], &[], &[]).unwrap();
        assert_eq!(transposed.cotangents(), &[TransposedProgramCotangent::Zero, TransposedProgramCotangent::Output]);
        assert_eq!(
            transposed.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:bool[] .
                let %2:f32[] = condition %1 %0 [
                    true={
                        lambda %0:f32[] .
                        in (%0)
                    },
                    false={
                        lambda %0:f32[] .
                        let %1:f32[] = neg %0
                        in (%1)
                    },
                ]
                in (%2)"},
        );
        for (predicate, expected) in [(true, 3.0f32), (false, -3.0f32)] {
            let outputs = transposed
                .program()
                .interpret(vec![Array::scalar(3.0f32).unwrap(), Array::scalar(predicate).unwrap()])
                .unwrap();
            assert_eq!(
                transposed.assemble_cotangents(&[array_type.clone(), scalar_type.clone()], outputs),
                Ok(vec![
                    Some(MaybeZero::Zero(array_type.clone())),
                    Some(MaybeZero::Value(Array::scalar(expected).unwrap())),
                ]),
            );
        }
    }

    #[test]
    fn test_condition_transposition_with_a_transpose_program_only_driver() {
        /// A downstream driver that implements only `transpose_program`, so that the condition rule requests the
        /// transposition of its branches through the default `transpose_program_with_zero_seeds`.
        struct Driver([Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>; 2]);

        impl RegionDriver<Array, ArrayOperation<Array>> for Driver {
            fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, Array, ArrayOperation<Array>>>
            where
                Array: 'r,
                ArrayOperation<Array>: 'r,
            {
                self.0.iter().map(Program::entry_region_ref)
            }
        }

        impl TranspositionDriver<Array, ArrayOperation<Array>> for Driver {
            fn transpose_program(
                &self,
                region: RegionRef<'_, Array, ArrayOperation<Array>>,
                input_indices: &[usize],
                materialize_zeros: &[bool],
                destination_kinds: &[CotangentDestinationKind],
            ) -> Result<TransposedProgram<Array, ArrayOperation<Array>>, DifferentiationError> {
                region.transpose(input_indices, materialize_zeros, destination_kinds)
            }
        }

        // The branches return their input, negated in the `false` branch, followed by a second output that is dead,
        // so the condition rule flags its cotangent as a structural zero. Program transposition, whose recursive
        // driver serves that request symbolically, and the rule invoked directly with the custom driver give the same
        // cotangents.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let branch = |negate| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let value = builder.add_input(scalar_type.clone());
            let output = if negate {
                builder.add_instruction(NegOperation::new(), Vec::new(), vec![value], None).unwrap()[0]
            } else {
                value
            };
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![output, value], vec![Placeholder], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let value = builder.add_input(scalar_type.clone());
        let branches = vec![builder.import_program(branch(false)), builder.import_program(branch(true))];
        let output =
            builder.add_instruction(ConditionOperation::new(), branches, vec![predicate, value], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let recursive = program.entry_region_ref().transpose(&[1], &[], &[]).unwrap();
        let (_, custom) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |tracers: Vec<Tracer<TracingContext<Array, ArrayOperation<Array>>>>| {
                let mut context = TranspositionContext::new(tracers[0].context().clone());
                let inputs = [PartialValue::Known(tracers[1].clone()), PartialValue::Unknown(scalar_type.clone())];
                let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
                ConditionOperation::<ArrayType>::new()
                    .transpose(
                        &mut context,
                        &Driver([branch(false), branch(true)]),
                        &inputs,
                        &[MaybeZero::Value(tracers[0].clone()), MaybeZero::Zero(scalar_type.clone())],
                        &accumulators,
                    )
                    .unwrap();
                Ok(vec![context.take_cotangents(&accumulators).unwrap()[1].as_value().unwrap().clone()])
            },
            vec![scalar_type.clone(), ArrayType::scalar(DataType::Boolean)],
        )
        .unwrap();
        let rendering = indoc! {"
            lambda %0:f32[], %1:bool[] .
            let %2:f32[] = condition %1 %0 [
                true={
                    lambda %0:f32[] .
                    in (%0)
                },
                false={
                    lambda %0:f32[] .
                    let %1:f32[] = neg %0
                    in (%1)
                },
            ]
            in (%2)"};
        assert_eq!(recursive.program().to_string(), rendering);
        assert_eq!(custom.to_string(), rendering);
        for (predicate, expected) in [(true, 3.0f32), (false, -3.0f32)] {
            let inputs = vec![Array::scalar(3.0f32).unwrap(), Array::scalar(predicate).unwrap()];
            assert_eq!(recursive.program().interpret(inputs.clone()), Ok(vec![Array::scalar(expected).unwrap()]));
            assert_eq!(custom.interpret(inputs), Ok(vec![Array::scalar(expected).unwrap()]));
        }
    }

    #[test]
    fn test_condition_transposition_materializes_one_sided_zeros() {
        let scalar_type = ArrayType::scalar(DataType::F32);
        let branch = |selected: usize| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let values = [builder.add_input(scalar_type.clone()), builder.add_input(scalar_type.clone())];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![values[selected]], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let first = builder.add_input(scalar_type.clone());
        let second = builder.add_input(scalar_type.clone());
        let branches = vec![builder.import_program(branch(0)), builder.import_program(branch(1))];
        let output = builder
            .add_instruction(ConditionOperation::new(), branches, vec![predicate, first, second], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let transposed = program.entry_region_ref().transpose(&[1, 2], &[], &[]).unwrap();
        assert_eq!(transposed.cotangents(), &[TransposedProgramCotangent::Output; 2]);
        assert_eq!(
            transposed.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:bool[] .
                let %2:f32[], %3:f32[] = condition %1 %0 [
                    true={
                        lambda %0:f32[] .
                        let %1:f32[] = zero [type=f32[]]
                        in (%0, %1)
                    },
                    false={
                        lambda %0:f32[] .
                        let %1:f32[] = zero [type=f32[]]
                        in (%1, %0)
                    },
                ]
                in (%2, %3)"},
        );
        assert_eq!(
            transposed.program().interpret(vec![Array::scalar(5.0f32).unwrap(), Array::scalar(true).unwrap()]),
            Ok(vec![Array::scalar(5.0f32).unwrap(), Array::scalar(0.0f32).unwrap()]),
        );
        assert_eq!(
            transposed.program().interpret(vec![Array::scalar(5.0f32).unwrap(), Array::scalar(false).unwrap()]),
            Ok(vec![Array::scalar(0.0f32).unwrap(), Array::scalar(5.0f32).unwrap()]),
        );
    }

    #[test]
    fn test_condition_transposition_materializes_one_sided_cast_zeros_from_extent_residuals() {
        // The `true` branch narrows `x: f32[n]` to `f32[2]` with a checked cast, while the `false` branch returns `y`.
        // Over the array IR universe, linearization saves one `dimension<n>` extent residual, which both branches read
        // from `x` with `dimension_size`, for the `true` branch's tangent cast, and the disconnected `x` needs no
        // separate residual. Condition transposition materializes each one-sided zero from that shared extent residual
        // when its type needs runtime extents.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![n.dimension().clone()]));
        let pair_type = ArrayType::new_static(DataType::F32, [2]);
        let cast_type = ArrayIrType::from(pair_type.clone());
        let (_, program) = TracingContext::<TestIrValue, TestIrOperation>::trace(
            |(predicate, x, y): (TestTracer, TestTracer, TestTracer)| {
                predicate.condition(|(x, _)| x.cast(cast_type.clone(), &[]), |(_, y)| Ok(y), (x, y))
            },
            (
                ArrayIrType::from(ArrayType::scalar(DataType::Boolean)),
                ArrayIrType::from(vector_type),
                ArrayIrType::from(pair_type),
            ),
        )
        .unwrap();
        let linearization = program.into_flat_program().linearize_with_respect_to(&[1, 2]).unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[n], %2:f32[2] .
                let %3:f32[2], %4:dimension<n ∈ [1, 8)> = condition %0 %1 %2 [
                    true={
                        lambda %0:f32[n], %1:f32[2] .
                        let %2:f32[2] = cast [type=f32[2], inputs=[f32[n]], checked=true] %0
                            %3:dimension<n ∈ [1, 8)> = dimension_size [axis=0] %0
                        in (%2, %3)
                    },
                    false={
                        lambda %0:f32[n], %1:f32[2] .
                        let %2:dimension<n ∈ [1, 8)> = dimension_size [axis=0] %0
                        in (%1, %2)
                    },
                ]
                in (%3, %0, %4)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[n], %1:f32[2], %2:bool[], %3:dimension<n ∈ [1, 8)> .
                let %4:f32[2] = condition %2 %0 %1 %3 [
                    true={
                        lambda %0:f32[n], %1:f32[2], %2:dimension<n ∈ [1, 8)> .
                        let %3:f32[2] = cast [type=f32[2], inputs=[f32[n], dimension<n ∈ [1, 8)>]] %0 %2
                        in (%3)
                    },
                    false={
                        lambda %0:f32[n], %1:f32[2], %2:dimension<n ∈ [1, 8)> .
                        in (%1)
                    },
                ]
                in (%4)"},
        );
        let pullback = linearization.pullback().unwrap().into_program();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:bool[], %2:dimension<n ∈ [1, 8)> .
                let %3:f32[n], %4:f32[2] = condition %1 %0 %2 [
                    true={
                        lambda %0:f32[2], %1:dimension<n ∈ [1, 8)> .
                        let %2:f32[n] = cast [type=f32[n], inputs=[f32[2], dimension<n ∈ [1, 8)>]] %0 %1
                            %3:f32[2] = zero [type=f32[2]]
                        in (%2, %3)
                    },
                    false={
                        lambda %0:f32[2], %1:dimension<n ∈ [1, 8)> .
                        let %2:f32[n] = zero [type=f32[n]] %1
                        in (%2, %0)
                    },
                ]
                in (%3, %4)"},
        );

        // At `n = 2` either branch differentiates. At `n = 3` the `true` branch rejects its checked cast, while the
        // `false` branch never evaluates it and returns a zero cotangent with the runtime extent of `x`.
        let y = array(Array::vector(vec![5f32, 6.0]).unwrap());
        let cotangent = array(Array::vector(vec![1f32, 2.0]).unwrap());
        for (predicate, x, expected_value, expected_gradients) in [
            (true, vec![2f32, 3.0], vec![2f32, 3.0], [vec![1f32, 2.0], vec![0f32, 0.0]]),
            (false, vec![2f32, 3.0], vec![5f32, 6.0], [vec![0f32, 0.0], vec![1f32, 2.0]]),
            (false, vec![2f32, 3.0, 4.0], vec![5f32, 6.0], [vec![0f32, 0.0, 0.0], vec![1f32, 2.0]]),
        ] {
            let mut outputs = linearization
                .primal()
                .interpret(vec![array(Array::scalar(predicate).unwrap()), array(Array::vector(x).unwrap()), y.clone()])
                .unwrap();
            let residuals = outputs.split_off(1);
            assert_eq!(outputs, vec![array(Array::vector(expected_value).unwrap())]);
            let mut pullback_inputs = vec![cotangent.clone()];
            pullback_inputs.extend(residuals);
            assert_eq!(
                pullback.interpret(pullback_inputs),
                Ok(expected_gradients.map(|gradient| array(Array::vector(gradient).unwrap())).to_vec()),
            );
        }
        assert!(matches!(
            linearization.primal().interpret(vec![
                array(Array::scalar(true).unwrap()),
                array(Array::vector(vec![2f32, 3.0, 4.0]).unwrap()),
                y,
            ]),
            Err(ProgramError::Type(TypeError::Invalid { message, .. }))
                if message == "`cast` cannot convert type `f32[3]` to unrelated type `f32[2]`",
        ));
    }

    #[test]
    fn test_condition_transposition_materializes_one_sided_cast_zeros_from_forwarded_inputs() {
        // The same checked-cast condition over the array universe, which has no first-class dimensions, saves `x`
        // itself as the residual that carries the runtime extent, and the `false` branch materializes its one-sided
        // zero cotangent for `x` with `zero_like`.
        let extent = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.simplified_dimension()]));
        let pair_type = ArrayType::new_static(DataType::F32, [2]);
        let cast_type = pair_type.clone();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(predicate, x, y): (ArrayTracer, ArrayTracer, ArrayTracer)| {
                predicate.condition(|(x, _)| x.cast(cast_type.clone(), &[]), |(_, y)| Ok(y), (x, y))
            },
            (ArrayType::scalar(DataType::Boolean), vector_type, pair_type),
        )
        .unwrap();
        let linearization = program.into_flat_program().linearize_with_respect_to(&[1, 2]).unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[extent], %2:f32[2] .
                let %3:f32[2], %4:f32[extent] = condition %0 %1 %2 [
                    true={
                        lambda %0:f32[extent], %1:f32[2] .
                        let %2:f32[2] = cast [type=f32[2], inputs=[f32[extent]], checked=true] %0
                        in (%2, %0)
                    },
                    false={
                        lambda %0:f32[extent], %1:f32[2] .
                        in (%1, %0)
                    },
                ]
                in (%3, %0, %4)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[extent], %1:f32[2], %2:bool[], %3:f32[extent] .
                let %4:f32[2] = condition %2 %0 %1 %3 [
                    true={
                        lambda %0:f32[extent], %1:f32[2], %2:f32[extent] .
                        let %3:f32[2] = cast [type=f32[2], inputs=[f32[extent], f32[extent]]] %0 %2
                        in (%3)
                    },
                    false={
                        lambda %0:f32[extent], %1:f32[2], %2:f32[extent] .
                        in (%1)
                    },
                ]
                in (%4)"},
        );
        assert_eq!(linearization.residual_count(), 2);
        let pullback = linearization.pullback().unwrap();
        assert_eq!(pullback.cotangents(), &[TransposedProgramCotangent::Output; 2]);
        assert_eq!(
            pullback.program().to_string(),
            indoc! {"
                lambda %0:f32[2], %1:bool[], %2:f32[extent] .
                let %3:f32[extent], %4:f32[2] = condition %1 %0 %2 [
                    true={
                        lambda %0:f32[2], %1:f32[extent] .
                        let %2:f32[extent] = cast [type=f32[extent], inputs=[f32[2], f32[extent]]] %0 %1
                            %3:f32[2] = zero [type=f32[2]]
                        in (%2, %3)
                    },
                    false={
                        lambda %0:f32[2], %1:f32[extent] .
                        let %2:f32[extent] = zero_like %1
                        in (%2, %0)
                    },
                ]
                in (%3, %4)"},
        );
        for (predicate, x, expected_value, expected) in [
            (true, vec![2.0f32, 3.0], vec![2.0f32, 3.0], [vec![1.0f32, 2.0], vec![0.0f32, 0.0]]),
            (false, vec![2.0f32, 3.0], vec![5.0f32, 6.0], [vec![0.0f32, 0.0], vec![1.0f32, 2.0]]),
            (false, vec![2.0f32, 3.0, 4.0], vec![5.0f32, 6.0], [vec![0.0f32, 0.0, 0.0], vec![1.0f32, 2.0]]),
        ] {
            let mut outputs = linearization
                .primal()
                .interpret(vec![
                    Array::scalar(predicate).unwrap(),
                    Array::vector(x).unwrap(),
                    Array::vector(vec![5.0f32, 6.0]).unwrap(),
                ])
                .unwrap();
            let mut inputs = vec![Array::vector(vec![1.0f32, 2.0]).unwrap()];
            inputs.extend(outputs.split_off(1));
            assert_eq!(outputs, vec![Array::vector(expected_value).unwrap()]);
            assert_eq!(
                pullback.program().interpret(inputs),
                Ok(expected.map(|values| Array::vector(values).unwrap()).to_vec()),
            );
        }
        assert!(matches!(
            linearization.primal().interpret(vec![
                Array::scalar(true).unwrap(),
                Array::vector(vec![2.0f32, 3.0, 4.0]).unwrap(),
                Array::vector(vec![5.0f32, 6.0]).unwrap(),
            ]),
            Err(ProgramError::Type(TypeError::Invalid { message, .. }))
                if message == "`cast` cannot convert type `f32[3]` to unrelated type `f32[2]`",
        ));
    }

    #[test]
    fn test_condition_transposition_keeps_dead_dynamic_output_cotangents_symbolic() {
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let array_type = ArrayType::new(DataType::F64, Shape::new(vec![extent_type.dimension().clone()]));
        let branch = || {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
            let input = builder.add_input(ArrayIrType::Array(array_type.clone()));
            let doubled = builder
                .add_instruction(
                    TestIrOperation::Array(ArrayOperation::from(AddOperation::new())),
                    Vec::new(),
                    vec![input, input],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                    vec![extent, doubled, input],
                    vec![Placeholder; 2],
                    vec![Placeholder; 3],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let input = builder.add_input(ArrayIrType::Array(array_type.clone()));
        let regions = vec![
            builder.import_region(branch().entry_region_ref()),
            builder.import_region(branch().entry_region_ref()),
        ];
        let outputs = builder
            .add_instruction(
                TestIrOperation::Condition(ConditionOperation::new()),
                regions,
                vec![predicate, extent, input],
                None,
            )
            .unwrap()
            .to_vec();

        // Keeping only the doubled output leaves the third branch output dead, so its dynamic cotangent reaches the
        // transposed condition as a structural zero that no type-only constructor can build. It stays symbolic: both
        // branches are transposed without a seed for it, so the pullback neither materializes it nor adds it.
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![outputs[1]], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:bool[], %1:dimension<extent ∈ [1, 8)>, %2:f64[extent] .
                let %3:dimension<extent ∈ [1, 8)>, %4:f64[extent], %5:f64[extent] = condition %0 %1 %2 [
                    true={
                        lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[extent] .
                        let %2:f64[extent] = add %1 %1
                        in (%0, %2, %1)
                    },
                    false={
                        lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[extent] .
                        let %2:f64[extent] = add %1 %1
                        in (%0, %2, %1)
                    },
                ]
                in (%4, %0)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[extent], %1:bool[] .
                let %2:f64[extent], %3:f64[extent] = condition %1 %0 [
                    true={
                        lambda %0:f64[extent] .
                        let %1:f64[extent] = add %0 %0
                        in (%1, %0)
                    },
                    false={
                        lambda %0:f64[extent] .
                        let %1:f64[extent] = add %0 %0
                        in (%1, %0)
                    },
                ]
                in (%2)"},
        );

        let pullback = linearization.pullback().unwrap().into_program();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[extent], %1:bool[] .
                let %2:f64[extent] = condition %1 %0 [
                    true={
                        lambda %0:f64[extent] .
                        let %1:f64[extent] = add %0 %0
                        in (%1)
                    },
                    false={
                        lambda %0:f64[extent] .
                        let %1:f64[extent] = add %0 %0
                        in (%1)
                    },
                ]
                in (%2)"}
            .trim_end(),
        );
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                array(Array::scalar(true).unwrap()),
                dimension(&extent_type, 3),
                array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap()),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        let mut pullback_inputs = vec![array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![array(Array::vector(vec![2.0, 2.0, 2.0]).unwrap())]));
    }

    #[test]
    fn test_condition_transposition_omits_structural_zeros_through_nested_scan() {
        let scalar = ArrayType::scalar(DataType::F64);
        let vector = ArrayType::new_static(DataType::F64, [3]);
        let branch = || {
            let mut body = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            body.add_input(ArrayType::scalar(DataType::I64));
            let carry = body.add_input(scalar.clone());
            body.add_input(scalar.clone());
            let next = body.add_instruction(AddOperation::new(), Vec::new(), vec![carry, carry], None).unwrap()[0];
            let body =
                body.build::<Vec<Array>, Vec<Array>>(vec![next], vec![Placeholder; 3], vec![Placeholder]).unwrap();
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let ignored = builder.add_input(vector.clone());
            let carry = builder.add_input(scalar.clone());
            let body = builder.import_program(body);
            let output =
                builder.add_instruction(ScanOperation::new(1, 3), vec![body], vec![carry, ignored], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let ignored = builder.add_input(vector.clone());
        let live = builder.add_input(scalar.clone());
        let branches = vec![builder.import_program(branch()), builder.import_program(branch())];
        let output = builder
            .add_instruction(ConditionOperation::new(), branches, vec![predicate, ignored, live], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let transposed = program.entry_region_ref().transpose(&[1, 2], &[], &[]).unwrap();
        assert_eq!(transposed.cotangents(), &[TransposedProgramCotangent::Zero, TransposedProgramCotangent::Output]);
        assert_eq!(
            transposed.program().to_string(),
            indoc! {"
                lambda %0:f64[], %1:bool[] .
                let %2:f64[] = condition %1 %0 [
                    true={
                        lambda %0:f64[] .
                        let %1:f64[] = scan [carry_count=1, iteration_count=3, reverse=true] %0 [
                            body={
                                lambda %0:i64[], %1:f64[] .
                                let %2:f64[] = add %1 %1
                                in (%2)
                            },
                        ]
                        in (%1)
                    },
                    false={
                        lambda %0:f64[] .
                        let %1:f64[] = scan [carry_count=1, iteration_count=3, reverse=true] %0 [
                            body={
                                lambda %0:i64[], %1:f64[] .
                                let %2:f64[] = add %1 %1
                                in (%2)
                            },
                        ]
                        in (%1)
                    },
                ]
                in (%2)"},
        );
        for predicate in [true, false] {
            assert_eq!(
                transposed
                    .program()
                    .interpret(vec![Array::scalar(2.0f64).unwrap(), Array::scalar(predicate).unwrap()]),
                Ok(vec![Array::scalar(16.0f64).unwrap()]),
            );
        }
    }

    #[test]
    fn test_condition_transposition_preserves_zero_result_backward_effects() {
        let scalar = ArrayType::scalar(DataType::F64);
        let branch = |label| {
            let mut backward = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let seed = backward.add_input(scalar.clone());
            backward.add_instruction(PrintOperation::new(label), Vec::new(), vec![seed], None).unwrap();
            let zero =
                backward.add_instruction(ZeroOperation::new(scalar.clone()), Vec::new(), Vec::new(), None).unwrap()[0];
            let backward =
                backward.build::<Vec<Array>, Vec<Array>>(vec![zero], vec![Placeholder], vec![Placeholder]).unwrap();
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let tangent = builder.add_input(scalar.clone());
            let backward = builder.import_program(backward);
            let carrier = CompositeTransposeOperation::<Array, ArrayOperation<Array>>::from_backward_region(
                0,
                vec![scalar.clone()],
                vec![scalar.clone()],
            );
            let output = builder.add_instruction(carrier, vec![backward], vec![tangent], None).unwrap()[0];
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let tangent = builder.add_input(scalar.clone());
        let branches =
            vec![builder.import_program(branch("true backward")), builder.import_program(branch("false backward"))];
        let output = builder
            .add_instruction(ConditionOperation::new(), branches, vec![predicate, tangent], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let transposed = program.entry_region_ref().transpose(&[1], &[], &[]).unwrap();
        assert_eq!(transposed.cotangents(), &[TransposedProgramCotangent::Zero]);
        assert!(transposed.program().effects().classes().is_ordered());
        assert!(transposed.program().output_ids().is_empty());
        assert_eq!(
            transposed.program().to_string(),
            indoc! {"
                lambda %0:f64[], %1:bool[] .
                let () = condition %1 %0 [
                    true={
                        lambda %0:f64[] .
                        let %1:f64[] = print [label=true backward] %0
                            %2:f64[] = zero [type=f64[]]
                        in ()
                    },
                    false={
                        lambda %0:f64[] .
                        let %1:f64[] = print [label=false backward] %0
                            %2:f64[] = zero [type=f64[]]
                        in ()
                    },
                ]
                in ()"},
        );
        let simplified = transposed.program().simplified().unwrap();
        assert_eq!(
            simplified.to_string(),
            indoc! {"
                lambda %0:f64[], %1:bool[] .
                let () = condition %1 %0 [
                    true={
                        lambda %0:f64[] .
                        let %1:f64[] = print [label=true backward] %0
                        in ()
                    },
                    false={
                        lambda %0:f64[] .
                        let %1:f64[] = print [label=false backward] %0
                        in ()
                    },
                ]
                in ()"},
        );
        // Standard-error capture is unavailable here. The retained ordered print and successful execution of both
        // selected branches pin this path; backend tests additionally capture the actual callback output.
        for program in [transposed.program().as_ref(), &simplified] {
            assert!(program.effects().classes().is_ordered());
            assert!(program.output_ids().is_empty());
            for predicate in [true, false] {
                assert_eq!(
                    program.interpret(vec![Array::scalar(3.0f64).unwrap(), Array::scalar(predicate).unwrap()]),
                    Ok(Vec::new()),
                );
            }
        }
    }

    #[test]
    fn test_condition_transposition_reference_input_destinations() {
        // Both branches receive the cotangent reference of the reference input: the taken branch's transpose acts on it
        // in place (`add_update` reads the destination into `x̄`, `write` swaps a zero into it).
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let true_branch = {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
            let value = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, value], None)
                .unwrap();
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
                .unwrap()
        };
        let false_branch = {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
            let value = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, value], None)
                .unwrap();
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let value = builder.add_input(scalar_type.clone());
        builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference, value],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder; 3], Vec::<Placeholder>::new())
            .unwrap();

        // The predicate is a known parameter of the linear map, so the transposed program consumes `[r̄, p]`.
        let transposed = program.transpose_with_respect_to(&[1, 2], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:ref<f32[]>, %1:bool[] .
                let %2:ref<f32[]>, %3:f32[] = condition %1 %0 [
                    true={
                        lambda %0:ref<f32[]> .
                        let %1:f32[] = reference_read %0
                        in (%0, %1)
                    },
                    false={
                        lambda %0:ref<f32[]> .
                        let %1:f32[] = zero [type=f32[]]
                            %2:f32[] = reference_swap %0 %1
                        in (%0, %2)
                    },
                ]
                in (%0, %3)"},
        );
        assert_eq!(
            transposed.input_types(),
            vec![
                ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))),
                ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)),
            ],
        );
        assert_eq!(
            transposed.output_types(),
            vec![ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))), scalar_type],
        );
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![],
                vec![Array::scalar(5.0f32).unwrap()],
                vec![Array::scalar(true).unwrap()],
            ),
            vec![Array::scalar(5.0f32).unwrap(), Array::scalar(5.0f32).unwrap()],
        );
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![],
                vec![Array::scalar(5.0f32).unwrap()],
                vec![Array::scalar(false).unwrap()],
            ),
            vec![Array::scalar(5.0f32).unwrap(), Array::scalar(0.0f32).unwrap()],
        );
    }

    #[test]
    fn test_condition_transposition_keeps_cotangents_reached_only_from_dead_outputs_symbolic() {
        // Branches `[x, z] -> [x, 2 · z]` and `[x, z] -> [x, 3 · z]`, of which only the first output is used, so the
        // second output's cotangent is a structural zero. Both branches are transposed without a seed for it: the
        // transposed condition receives no zero input, and `z`, which only that output reaches, keeps a
        // structural-zero cotangent instead of a branch-selected zero.
        let branch = |factor: f64| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(ArrayType::scalar(DataType::F64));
            let scaled = builder.add_input(ArrayType::scalar(DataType::F64));
            let factor = builder.add_constant(Array::scalar(factor).unwrap());
            let product =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![factor, scaled], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![input, product], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let true_branch = builder.import_program(branch(2.0));
        let false_branch = builder.import_program(branch(3.0));
        let inputs =
            [ArrayType::scalar(DataType::Boolean), ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)]
                .map(|r#type| builder.add_input(r#type))
                .to_vec();
        let output = builder
            .add_instruction(ConditionOperation::new(), vec![true_branch, false_branch], inputs, None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let transposed = program.entry_region_ref().transpose(&[1, 2], &[], &[]).unwrap();
        assert_eq!(transposed.cotangents(), &[TransposedProgramCotangent::Output, TransposedProgramCotangent::Zero]);
        assert_eq!(
            transposed.program().to_string(),
            indoc! {"
                lambda %0:f64[], %1:bool[] .
                let %2:f64[] = condition %1 %0 [
                    true={
                        lambda %0:f64[] .
                        in (%0)
                    },
                    false={
                        lambda %0:f64[] .
                        in (%0)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(
            transposed.program().interpret(vec![Array::scalar(5.0f64).unwrap(), Array::scalar(false).unwrap()]),
            Ok(vec![Array::scalar(5.0f64).unwrap()]),
        );
    }

    #[test]
    fn test_condition_transposition_keeps_dead_dynamic_outputs_of_reference_inputs_symbolic() {
        // The condition's snapshot output is dead, but its reference state cotangent is live and keeps the transposed
        // condition live. The dead snapshot's cotangent stays a structural zero, so the pullback needs no zero seed,
        // whose dynamic extent only the reference supplies, and the untaken branch leaves the state cotangent alone.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(8)).unwrap());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let reference_type = ArrayIrType::from(ReferenceType::new(array_type));
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = builder.add_input(reference_type.clone());
        let snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, snapshot], None)
            .unwrap();
        let true_branch = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = builder.add_input(reference_type.clone());
        let snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let false_branch = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_branch = builder.import_program(true_branch);
        let false_branch = builder.import_program(false_branch);
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let reference = builder.add_input(reference_type);
        builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[1], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:ref<f32[extent]>, %1:bool[] .
                let %2:ref<f32[extent]> = condition %1 %0 [
                    true={
                        lambda %0:ref<f32[extent]> .
                        let %1:f32[extent] = reference_read %0
                            () = reference_add_update %0 %1
                        in (%0)
                    },
                    false={
                        lambda %0:ref<f32[extent]> .
                        in (%0)
                    },
                ]
                in (%0)"},
        );
        let destination = ArrayReference::new(Array::vector(vec![3f32, 5.0, 7.0]).unwrap());
        assert_eq!(
            transposed
                .interpret(vec![TestIrValue::Reference(destination.clone()), array(Array::scalar(true).unwrap())]),
            Ok(vec![TestIrValue::Reference(destination.clone())]),
        );
        assert_eq!(destination.read(), Ok(Array::vector(vec![6f32, 10.0, 14.0]).unwrap()));
        let destination = ArrayReference::new(Array::vector(vec![3f32, 5.0]).unwrap());
        assert_eq!(
            transposed
                .interpret(vec![TestIrValue::Reference(destination.clone()), array(Array::scalar(false).unwrap())]),
            Ok(vec![TestIrValue::Reference(destination.clone())]),
        );
        assert_eq!(destination.read(), Ok(Array::vector(vec![3f32, 5.0]).unwrap()));
    }

    #[test]
    fn test_condition_transposition_write_only_reference_input_destinations() {
        // Both branches only store into the reference input (`write` when taken, `add_update` otherwise) and forward
        // `x` as the live output. Under an `Ignore` destination for the reference no later instruction accumulated into
        // its root and neither branch reads it, so its state cotangent is provably zero: the branches are transposed
        // with an `Ignore` destination as well and the pullback stages no cotangent reference at all instead of
        // allocating, zeroing, and freezing a dead accumulator around the transposed condition.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let predicate_type = ArrayIrType::Array(ArrayType::scalar(DataType::Boolean));
        let true_branch = {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
            let value = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, value], None)
                .unwrap();
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let false_branch = {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
            let value = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, value], None)
                .unwrap();
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(predicate_type.clone());
        let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let value = builder.add_input(scalar_type.clone());
        let output = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference, value],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let transposed = program
            .transpose_with_respect_to(&[1, 2], &[CotangentDestinationKind::Ignore, CotangentDestinationKind::Return])
            .unwrap();
        assert_eq!(transposed.input_types(), vec![scalar_type.clone(), predicate_type.clone()]);
        assert_eq!(transposed.output_types(), vec![scalar_type.clone()]);
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[], %1:bool[] .
                let %2:f32[] = condition %1 %0 [
                    true={
                        lambda %0:f32[] .
                        in (%0)
                    },
                    false={
                        lambda %0:f32[] .
                        in (%0)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![Array::scalar(3.0f32).unwrap()],
                vec![],
                vec![Array::scalar(true).unwrap()],
            ),
            vec![Array::scalar(3.0f32).unwrap()],
        );

        // Under a `Reference` destination the reference input's state cotangent is live, so both branches receive the
        // cotangent reference and their stores transpose against it.
        let transposed = program.transpose_with_respect_to(&[1, 2], &[]).unwrap();
        assert_eq!(
            transposed.input_types(),
            vec![
                scalar_type.clone(),
                ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))),
                predicate_type,
            ],
        );
        assert_eq!(
            transposed.output_types(),
            vec![ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))), scalar_type],
        );
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[], %1:ref<f32[]>, %2:bool[] .
                let %3:ref<f32[]>, %4:f32[] = condition %2 %0 %1 [
                    true={
                        lambda %0:f32[], %1:ref<f32[]> .
                        let %2:f32[] = zero [type=f32[]]
                            %3:f32[] = reference_swap %1 %2
                            %4:f32[] = add %0 %3
                        in (%1, %4)
                    },
                    false={
                        lambda %0:f32[], %1:ref<f32[]> .
                        let %2:f32[] = reference_read %1
                            %3:f32[] = add %0 %2
                        in (%1, %3)
                    },
                ]
                in (%1, %4)"},
        );
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![Array::scalar(3.0f32).unwrap()],
                vec![Array::scalar(5.0f32).unwrap()],
                vec![Array::scalar(true).unwrap()],
            ),
            vec![Array::scalar(8.0f32).unwrap(), Array::scalar(0.0f32).unwrap()],
        );
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![Array::scalar(3.0f32).unwrap()],
                vec![Array::scalar(5.0f32).unwrap()],
                vec![Array::scalar(false).unwrap()],
            ),
            vec![Array::scalar(8.0f32).unwrap(), Array::scalar(5.0f32).unwrap()],
        );
    }

    #[test]
    fn test_condition_transposition_reference_access_with_enclosing_binding() {
        // Each branch accesses the reference root through a dynamic index that the enclosing region computes. The
        // transposed branches apply the same transforms to the root's cotangent reference, so the index reaches them as
        // an ordinary known input recomputed in the enclosing region: `add_update(r[i], x)` transposes into
        // `x̄ = read(r̄[i])`, and `write(r[i], x)` additionally clears `r̄[i]`.
        let vector_reference_type = ArrayIrType::from(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let index_type = ArrayIrType::Array(ArrayType::scalar(DataType::I32));
        let element_transforms =
            vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        let true_branch = {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let reference = builder.add_input(vector_reference_type.clone());
            let value = builder.add_input(scalar_type.clone());
            let index = builder.add_input(index_type.clone());
            builder
                .add_instruction(
                    ReferenceAddUpdateOperation::new().with_transforms(element_transforms.clone()),
                    Vec::new(),
                    vec![reference, value, index],
                    None,
                )
                .unwrap();
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder; 3], Vec::new())
                .unwrap()
        };
        let false_branch = {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let reference = builder.add_input(vector_reference_type.clone());
            let value = builder.add_input(scalar_type.clone());
            let index = builder.add_input(index_type.clone());
            builder
                .add_instruction(
                    ReferenceWriteOperation::new().with_transforms(element_transforms),
                    Vec::new(),
                    vec![reference, value, index],
                    None,
                )
                .unwrap();
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder; 3], Vec::new())
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let reference = builder.add_input(vector_reference_type.clone());
        let value = builder.add_input(scalar_type.clone());
        let offset = builder.add_input(index_type);
        let one = builder.add_constant(TestIrValue::Array(Array::scalar(1i32).unwrap()));
        let index = builder.add_instruction(AddOperation::new(), Vec::new(), vec![offset, one], None).unwrap()[0];
        builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference, value, index],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder; 4], Vec::<Placeholder>::new())
            .unwrap();

        let transposed = program.transpose_with_respect_to(&[1, 2], &[]).unwrap();
        assert_eq!(transposed.output_types(), vec![vector_reference_type, scalar_type]);
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:ref<f32[3]>, %1:bool[], %2:i32[] .
                let %3:i32[] = const 1
                    %4:i32[] = add %2 %3
                    %5:ref<f32[3]>, %6:f32[] = condition %1 %0 %4 [
                        true={
                            lambda %0:ref<f32[3]>, %1:i32[] .
                            let %2:f32[] = reference_read [transforms=[index(axis=0, index=dynamic)]] %0 %1
                            in (%0, %2)
                        },
                        false={
                            lambda %0:ref<f32[3]>, %1:i32[] .
                            let %2:f32[] = zero [type=f32[]]
                                %3:f32[] = reference_swap [transforms=[index(axis=0, index=dynamic)]] %0 %2 %1
                            in (%0, %3)
                        },
                    ]
                in (%0, %6)"},
        );
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![],
                vec![Array::vector(vec![1f32, 2.0, 3.0]).unwrap()],
                vec![Array::scalar(true).unwrap(), Array::scalar(0i32).unwrap()],
            ),
            vec![Array::scalar(2f32).unwrap(), Array::vector(vec![1f32, 2.0, 3.0]).unwrap()],
        );
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![],
                vec![Array::vector(vec![1f32, 2.0, 3.0]).unwrap()],
                vec![Array::scalar(false).unwrap(), Array::scalar(1i32).unwrap()],
            ),
            vec![Array::scalar(3f32).unwrap(), Array::vector(vec![1f32, 2., 0.]).unwrap()],
        );
    }

    #[test]
    fn test_condition_transposition_preserves_shared_gradient_buffers() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let mut branch = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let left = branch.add_input(scalar_type.clone());
        let right = branch.add_input(scalar_type.clone());
        let sum = branch.add_instruction(AddOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let branch = branch
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![sum], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let branch = builder.import_program(branch);
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(scalar_type.clone());
        let output = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![branch, branch],
                vec![predicate, input, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[1], &[CotangentDestinationKind::Reference]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[], %1:ref<f32[]>, %2:bool[] .
                let () = condition %2 %0 %1 %1 [
                    true=^0={
                        lambda %0:f32[], %1:ref<f32[]>, %2:ref<f32[]> .
                        let () = reference_add_update %1 %0
                            () = reference_add_update %2 %0
                        in ()
                    },
                    false=^0,
                ]
                in ()"},
        );

        // Both nested instruction input positions share one caller-owned buffer. Test through a local allocation so
        // reference discharge must preserve that internal alias rather than treating the two branch inputs as
        // independent state.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(scalar_type.clone());
        let seed = builder.add_input(scalar_type);
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        assert!(builder.splice_program(&transposed, &[seed, reference, predicate]).unwrap().is_empty());
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let staged = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        assert_eq!(
            staged.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[], %2:bool[] .
                let %3:ref<f32[]> = reference_new %0
                    () = condition %2 %1 %3 %3 [
                        true=^0={
                            lambda %0:f32[], %1:ref<f32[]>, %2:ref<f32[]> .
                            let () = reference_add_update %1 %0
                                () = reference_add_update %2 %0
                            in ()
                        },
                        false=^0,
                    ]
                    %4:f32[] = reference_freeze %3
                in (%4)"},
        );

        let inputs = vec![
            Array::scalar(5.0f32).unwrap().into(),
            Array::scalar(3.0f32).unwrap().into(),
            Array::scalar(true).unwrap().into(),
        ];
        let expected = vec![Array::scalar(11.0f32).unwrap().into()];
        assert_eq!(staged.interpret(inputs.clone()).unwrap(), expected);
        let discharged = staged.discharge_references(0).unwrap().into_program_without_external_references().unwrap();
        assert_eq!(
            discharged.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[], %2:bool[] .
                let %3:f32[] = condition %2 %1 %0 %0 [
                    true={
                        lambda %0:f32[], %1:f32[], %2:f32[] .
                        let %3:f32[] = add %1 %0
                            %4:f32[] = add %3 %0
                        in (%4)
                    },
                    false={
                        lambda %0:f32[], %1:f32[], %2:f32[] .
                        let %3:f32[] = add %1 %0
                            %4:f32[] = add %3 %0
                        in (%4)
                    },
                ]
                in (%3)"},
        );

        assert_eq!(discharged.interpret(inputs).unwrap(), expected);
    }

    #[test]
    fn test_condition_transposition_allows_disconnected_invariant_inputs_under_varying_predicates() {
        let (predicate_type, invariant_type, varying_type) = devices_varying_predicate_types();
        let branch = |negate| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            builder.add_input(invariant_type.clone());
            let value = builder.add_input(varying_type.clone());
            let output = if negate {
                builder.add_instruction(NegOperation::new(), Vec::new(), vec![value], None).unwrap()[0]
            } else {
                value
            };
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let predicate = builder.add_input(predicate_type.clone());
        let ignored = builder.add_input(invariant_type.clone());
        let value = builder.add_input(varying_type.clone());
        let branches = vec![builder.import_program(branch(false)), builder.import_program(branch(true))];
        let output = builder
            .add_instruction(ConditionOperation::new(), branches, vec![predicate, ignored, value], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let transposed = program.entry_region_ref().transpose(&[1, 2], &[], &[]).unwrap();
        assert_eq!(transposed.cotangents(), &[TransposedProgramCotangent::Zero, TransposedProgramCotangent::Output]);
        assert_eq!(
            transposed.program().to_string(),
            formatdoc! {"
                lambda %0:{varying_type}, %1:{predicate_type} .
                let %2:{varying_type} = condition %1 %0 [
                    true={{
                        lambda %0:{varying_type} .
                        in (%0)
                    }},
                    false={{
                        lambda %0:{varying_type} .
                        let %1:{varying_type} = neg %0
                        in (%1)
                    }},
                ]
                in (%2)"},
        );
    }

    #[test]
    fn test_condition_transposition_rejects_invariant_inputs_under_varying_predicates() {
        // A condition staged directly with an invariant weight under a predicate that varies over `devices` linearizes,
        // because its branches vary the weight before using it. The cotangent of the weight, however, differs across
        // devices that take different branches, and summing it would place collectives inside the branches, so
        // transposition rejects the condition with a diagnostic instead of a mistyped transposed condition.
        let (predicate_type, invariant_type, varying_type) = devices_varying_predicate_types();
        let branch = |multiplies: bool| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let weight = builder.add_input(invariant_type.clone());
            let input = builder.add_input(varying_type.clone());
            let varied = builder
                .add_instruction(ParallelVaryOperation::new("devices".to_string()), Vec::new(), vec![weight], None)
                .unwrap()[0];
            let output = if multiplies {
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![varied, input], None).unwrap()[0]
            } else {
                varied
            };
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let predicate = builder.add_input(predicate_type.clone());
        let weight = builder.add_input(invariant_type.clone());
        let input = builder.add_input(varying_type.clone());
        let true_region = builder.import_program(branch(true));
        let false_region = builder.import_program(branch(false));
        let output = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_region, false_region],
                vec![predicate, weight, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let linearization = program.linearize_with_respect_to(&[1]).unwrap();
        assert_eq!(
            linearization.pullback().err(),
            Some(DifferentiationError::from(ProgramError::UnsupportedOperation {
                message: format!(
                    "`condition` transposition cannot produce the cotangent of input 1 of type `{invariant_type}`, \
                     which does not vary over every manual axis that the predicate `{predicate_type}` varies over; \
                     vary the input before the condition",
                ),
            })),
        );
    }

    #[test]
    fn test_condition_transposition_gradient_buffer_destinations_under_varying_predicates() {
        // A gradient buffer of an invariant input crosses the condition boundary only when a transposed branch
        // accumulates into it. A disconnected input's buffer is untouched by both branches and is accepted like a
        // disconnected returned cotangent, while a connected input's buffer would receive a cotangent that differs
        // across the devices that take different branches.
        let (predicate_type, invariant_type, varying_type) = devices_varying_predicate_types();
        let condition_program = |uses_weight: bool| {
            let branch = |doubles: bool| {
                let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
                let weight = builder.add_input(invariant_type.clone().into());
                let value = builder.add_input(varying_type.clone().into());
                let mut output = if doubles {
                    builder
                        .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![value, value], None)
                        .unwrap()[0]
                } else {
                    value
                };
                if uses_weight {
                    let varied = builder
                        .add_instruction(
                            ParallelVaryOperation::new("devices".to_string()),
                            Vec::new(),
                            vec![weight],
                            None,
                        )
                        .unwrap()[0];
                    output = builder
                        .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![varied, output], None)
                        .unwrap()[0];
                }
                builder
                    .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                    .unwrap()
            };
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let predicate = builder.add_input(predicate_type.clone().into());
            let weight = builder.add_input(invariant_type.clone().into());
            let value = builder.add_input(varying_type.clone().into());
            let branches = vec![builder.import_program(branch(false)), builder.import_program(branch(true))];
            let output = builder
                .add_instruction(ConditionOperation::new(), branches, vec![predicate, weight, value], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };
        let destination_kinds = [CotangentDestinationKind::Reference, CotangentDestinationKind::Return];

        let transposed =
            condition_program(false).entry_region_ref().transpose(&[1, 2], &[], &destination_kinds).unwrap();
        assert_eq!(transposed.cotangents(), &[TransposedProgramCotangent::Absent, TransposedProgramCotangent::Output]);
        assert_eq!(
            transposed.program().to_string(),
            formatdoc! {"
                lambda %0:{varying_type}, %1:ref<{invariant_type}>, %2:{predicate_type} .
                let %3:{varying_type} = condition %2 %0 %1 [
                    true={{
                        lambda %0:{varying_type}, %1:ref<{invariant_type}> .
                        in (%0)
                    }},
                    false={{
                        lambda %0:{varying_type}, %1:ref<{invariant_type}> .
                        let %2:{varying_type} = add %0 %0
                        in (%2)
                    }},
                ]
                in (%3)"},
        );

        assert_eq!(
            condition_program(true).entry_region_ref().transpose(&[1, 2], &[], &destination_kinds).map(|_| ()),
            Err(DifferentiationError::from(ProgramError::UnsupportedOperation {
                message: format!(
                    "`condition` transposition cannot produce the cotangent of input 1 of type `{invariant_type}`, \
                     which does not vary over every manual axis that the predicate `{predicate_type}` varies over; \
                     vary the input before the condition",
                ),
            })),
        );
    }

    #[test]
    fn test_condition_condition() {
        // Outside of manual regions, `condition` traces both functions into branches over the provided inputs and
        // stages one condition without any manual-variation alignment, both in a trace and eagerly.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<Tracer<_>>| {
                inputs[0].condition(
                    |inputs| Ok(vec![inputs[0].clone() * inputs[0].clone()]),
                    |inputs| Ok(vec![inputs[0].clone() + inputs[0].clone()]),
                    vec![inputs[1].clone()],
                )
            },
            vec![ArrayType::scalar(DataType::Boolean), ArrayType::scalar(DataType::F64)],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[] .
                let %2:f64[] = condition %0 %1 [
                    true={
                        lambda %0:f64[] .
                        let %1:f64[] = mul %0 %0
                        in (%1)
                    },
                    false={
                        lambda %0:f64[] .
                        let %1:f64[] = add %0 %0
                        in (%1)
                    },
                ]
                in (%2)"},
        );
        for (predicate, expected) in [(true, 9.0), (false, 6.0)] {
            assert_eq!(
                program.interpret(vec![Array::scalar(predicate).unwrap(), Array::scalar(3.0).unwrap()]),
                Ok(vec![Array::scalar(expected).unwrap()]),
            );
            assert_eq!(
                Array::scalar(predicate).unwrap().condition(
                    |inputs| Ok(vec![inputs[0].clone() * inputs[0].clone()]),
                    |inputs| Ok(vec![inputs[0].clone() + inputs[0].clone()]),
                    vec![Array::scalar(3.0).unwrap()],
                ),
                Ok(vec![Array::scalar(expected).unwrap()]),
            );
        }
    }

    #[test]
    fn test_condition_condition_structured() {
        let (output_types, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(predicate, first, second): (Tracer<_>, Tracer<_>, Tracer<_>)| {
                predicate.condition(
                    |(first, second)| Ok((first.clone() * second, vec![first])),
                    |(first, second)| Ok((first + second.clone(), vec![second])),
                    (first, second),
                )
            },
            (ArrayType::scalar(DataType::Boolean), ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)),
        )
        .unwrap();
        assert_eq!(output_types, (ArrayType::scalar(DataType::F64), vec![ArrayType::scalar(DataType::F64)]));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[], %2:f64[] .
                let %3:f64[], %4:f64[] = condition %0 %1 %2 [
                    true={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = mul %0 %1
                        in (%2, %0)
                    },
                    false={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = add %0 %1
                        in (%2, %1)
                    },
                ]
                in (%3, %4)"},
        );
        assert_eq!(
            program.interpret((
                Array::scalar(true).unwrap(),
                Array::scalar(3.0f64).unwrap(),
                Array::scalar(4.0f64).unwrap(),
            )),
            Ok((Array::scalar(12.0f64).unwrap(), vec![Array::scalar(3.0f64).unwrap()])),
        );
        assert_eq!(
            program.interpret((
                Array::scalar(false).unwrap(),
                Array::scalar(3.0f64).unwrap(),
                Array::scalar(4.0f64).unwrap(),
            )),
            Ok((Array::scalar(7.0f64).unwrap(), vec![Array::scalar(4.0f64).unwrap()])),
        );

        /// Inputs and outputs whose static fields must survive tracing and reconstruction.
        #[derive(ryft_macros::Parameterized)]
        struct Payload<P: Parameter> {
            /// Runtime leaf selected by the condition.
            value: P,

            /// Static metadata shared by both branches.
            label: &'static str,
        }

        // Only the structure skeleton is cloned or compared; payloads intentionally do not implement Clone.
        impl Clone for Payload<Placeholder> {
            fn clone(&self) -> Self {
                Self { value: self.value, label: self.label }
            }
        }

        impl PartialEq for Payload<Placeholder> {
            fn eq(&self, other: &Self) -> bool {
                self.label == other.label
            }
        }

        /// A Condition bound alone provides the authoring capability to ordinary generic value callers.
        fn choose<V: Condition>(predicate: &V, input: V) -> Result<V, ProgramError> {
            predicate.condition(Ok, Ok, input)
        }

        // Eager values keep the static fields of structured inputs and outputs, and branches must agree on them.
        let predicate = Array::scalar(true).unwrap();
        let input = Array::scalar(3.0f32).unwrap();
        assert_eq!(choose(&predicate, input.clone()), Ok(input.clone()));
        let output = predicate
            .condition(
                |input| Ok(Payload { value: input.value.clone() + input.value, label: input.label }),
                Ok,
                Payload { value: input.clone(), label: "shared" },
            )
            .unwrap();
        assert_eq!(output.value, Array::scalar(6.0f32).unwrap());
        assert_eq!(output.label, "shared");

        let result = predicate.condition(
            |input| Ok(Payload { value: input.value, label: "true" }),
            |input| Ok(Payload { value: input.value, label: "false" }),
            Payload { value: input, label: "input" },
        );
        assert!(matches!(
            result,
            Err(ProgramError::InvalidArgument { message, .. })
                if message == "`condition` branches must return the same parameter structure and static metadata",
        ));

        // A structural error is rejected before the enclosing trace binds a condition instruction.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(predicate, input): (Tracer<_>, Tracer<_>)| {
                let result = predicate.condition(
                    |input| Ok(Payload { value: input.value, label: "true" }),
                    |input| Ok(Payload { value: input.value, label: "false" }),
                    Payload { value: input.clone(), label: "input" },
                );
                assert!(matches!(
                    result,
                    Err(ProgramError::InvalidArgument { message, .. })
                        if message
                            == "`condition` branches must return the same parameter structure and static metadata",
                ));
                Ok(input)
            },
            (ArrayType::scalar(DataType::Boolean), ArrayType::scalar(DataType::F32)),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:bool[], %1:f32[] .
            in (%1)"},
        );

        assert_eq!(predicate.condition(|()| Ok(()), |()| Ok(()), ()), Ok(()));
        assert_eq!(predicate.condition(Ok, Ok, Vec::<Array>::new()), Ok(Vec::<Array>::new()));

        // Array IR values carry arrays, dimensions, and references through the same structure.
        let predicate = TestIrValue::Array(Array::scalar(true).unwrap());
        let dimension = TestIrValue::Dimension(DimensionValue::constant(3).unwrap());
        let reference = TestIrValue::Reference(ArrayReference::new(Array::scalar(4.0f32).unwrap()));
        assert_eq!(
            predicate.condition(
                |(input, dimension, reference)| Ok((input.clone() + input, dimension, reference)),
                Ok,
                (TestIrValue::Array(Array::scalar(3.0f32).unwrap()), dimension.clone(), reference.clone()),
            ),
            Ok((TestIrValue::Array(Array::scalar(6.0f32).unwrap()), dimension, reference)),
        );

        // Branches must return the same parameter paths.
        let predicate = Array::scalar(true).unwrap();
        let input = Array::scalar(3.0f32).unwrap();
        let result =
            predicate.condition(|input| Ok(vec![input]), |input| Ok(vec![input.clone(), input]), input.clone());
        assert!(matches!(
            result,
            Err(ProgramError::InvalidArgument { message, .. })
                if message == "`condition` branches must return the same parameter structure and static metadata",
        ));
        let result = predicate.condition(
            |input| Ok(BTreeMap::from([("left", input)])),
            |input| Ok(BTreeMap::from([("right", input)])),
            input,
        );
        assert!(matches!(
            result,
            Err(ProgramError::InvalidArgument { message, .. })
                if message == "`condition` branches must return the same parameter structure and static metadata",
        ));
    }

    #[test]
    fn test_condition_condition_rejects_invalid_predicates_and_branch_signatures() {
        let input = Array::scalar(3.0f32).unwrap();
        assert_eq!(
            Array::scalar(1.0f32).unwrap().condition(Ok, Ok, input),
            Err(TypeError::invalid("`condition` predicate type must be a scalar boolean, but got `f32[]`").into()),
        );
        for predicate in [true, false] {
            assert_eq!(
                Array::scalar(predicate).unwrap().condition(
                    Ok,
                    |input| Ok(StagingContext::constant(input.context(), Array::scalar(true).unwrap())),
                    Array::scalar(3.0f32).unwrap(),
                ),
                Err(TypeError::invalid(
                    "`condition` branch output type signature mismatch: expected [f32[]] but got [bool[]]",
                )
                .into()),
            );
        }
        let result = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(predicate, input): (Tracer<_>, Tracer<_>)| predicate.condition(Ok, Ok, input),
            (ArrayType::scalar(DataType::F32), ArrayType::scalar(DataType::F32)),
        );
        assert!(matches!(
            result,
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`condition` predicate type must be a scalar boolean, but got `f32[]`",
        ));
        let result = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(predicate, input): (Tracer<_>, Tracer<_>)| {
                predicate.condition(
                    Ok,
                    |input| Ok(StagingContext::constant(input.context(), Array::scalar(true).unwrap())),
                    input,
                )
            },
            (ArrayType::scalar(DataType::Boolean), ArrayType::scalar(DataType::F32)),
        );
        assert!(matches!(
            result,
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message
                    == "`condition` branch output type signature mismatch: expected [f32[]] but got [bool[]]",
        ));

        // A reference returned by only one branch is the same signature mismatch, reported before the branches are
        // bound, rather than an internal error from the reference checks that staging performs first.
        let result = TracingContext::<TestIrValue, TestIrOperation>::trace(
            |inputs: Vec<Tracer<_>>| {
                inputs[0].condition(
                    Ok,
                    |reference| {
                        let read = ReferenceReadOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
                        Ok(reference
                            .context()
                            .bind(TestIrOperation::from(read), Vec::new(), &[reference.clone()])?
                            .remove(0))
                    },
                    inputs[1].clone(),
                )
            },
            vec![
                ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)),
                ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))),
            ],
        );
        assert!(matches!(
            result,
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message
                    == "`condition` branch output type signature mismatch: expected [ref<f32[]>] but got [f32[]]",
        ));
    }

    #[test]
    fn test_condition_condition_rejects_reference_predicate_before_alignment() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let referent = ArrayType::scalar(DataType::Boolean)
            .with_sharding(Sharding::replicated(mesh, 0).with_varying_manual_axes(["devices"]).unwrap())
            .unwrap();
        let predicate_type = ArrayIrType::Reference(ReferenceType::new(referent));
        let context = TracingContext::<TestIrValue, TestIrOperation>::new();
        let predicate = context.input(predicate_type.clone());
        let input = context.input(ArrayType::scalar(DataType::F32).into());
        let traced = Cell::new(false);
        let result = predicate.condition(
            |input| {
                traced.set(true);
                Ok(input)
            },
            |input| {
                traced.set(true);
                Ok(input)
            },
            input,
        );
        assert_eq!(
            result.map(|_| ()),
            Err(TypeError::invalid(format!(
                "`condition` predicate type must be a scalar boolean, but got `{predicate_type}`",
            ))
            .into()),
        );
        assert!(!traced.get());
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_condition_condition_kernel_operations() {
        // `condition` serves the kernel operation family that the `#[kernel]` macro lowers Rust `if` statements to,
        // tracing each branch function exactly once and selecting its result at runtime.
        let true_calls = Cell::new(0);
        let false_calls = Cell::new(0);
        let (_, program) = TracingContext::<TestIrValue, KernelOperation>::trace(
            |inputs: Vec<Tracer<_>>| {
                inputs[0].condition(
                    |inputs| {
                        true_calls.set(true_calls.get() + 1);
                        inputs[0].context().bind(
                            TestIrOperation::from(ArrayOperation::Add(AddOperation::new())),
                            Vec::new(),
                            &[inputs[0].clone(), inputs[0].clone()],
                        )
                    },
                    |inputs| {
                        false_calls.set(false_calls.get() + 1);
                        inputs[0].context().bind(
                            TestIrOperation::from(ArrayOperation::Neg(NegOperation::new())),
                            Vec::new(),
                            &inputs,
                        )
                    },
                    vec![inputs[1].clone()],
                )
            },
            vec![
                ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)),
                ArrayIrType::Array(ArrayType::scalar(DataType::I64)),
            ],
        )
        .unwrap();
        assert_eq!((true_calls.get(), false_calls.get()), (1, 1));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:i64[] .
                let %2:i64[] = condition %0 %1 [
                    true={
                        lambda %0:i64[] .
                        let %1:i64[] = add %0 %0
                        in (%1)
                    },
                    false={
                        lambda %0:i64[] .
                        let %1:i64[] = neg %0
                        in (%1)
                    },
                ]
                in (%2)"},
        );
        for (predicate, expected) in [(true, 6i64), (false, -3i64)] {
            assert_eq!(
                program.interpret(vec![
                    TestIrValue::Array(Array::scalar(predicate).unwrap()),
                    TestIrValue::Array(Array::scalar(3i64).unwrap()),
                ]),
                Ok(vec![TestIrValue::Array(Array::scalar(expected).unwrap())]),
            );
        }
    }

    #[test]
    fn test_condition_condition_aligns_manual_variation() {
        // With a predicate that varies over `devices`, `condition` varies the invariant input before the branches and
        // the constant output of the `false` branch at its end, so every branch value is typed as the per-device value
        // that it is, while the input that already varies passes through unchanged.
        let (predicate_type, invariant_type, varying_type) = devices_varying_predicate_types();
        let named_axes =
            vec![("devices".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: devices_manual_mesh() })];
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |inputs: Vec<Tracer<_>>| {
                inputs[0].condition(
                    |inputs| Ok(vec![inputs[0].clone() * inputs[1].clone()]),
                    |inputs| Ok(vec![StagingContext::constant(inputs[0].context(), Array::scalar(1.0f32).unwrap())]),
                    vec![inputs[1].clone(), inputs[2].clone()],
                )
            },
            vec![predicate_type.clone(), invariant_type.clone(), varying_type.clone()],
            named_axes.clone(),
        )
        .unwrap();
        let predicate = "bool[][sharding={mesh<['devices'=2:manual]>, [], varying_manual={'devices'}}]";
        let invariant = "f32[][sharding={mesh<['devices'=2:manual]>, []}]";
        let varying = "f32[][sharding={mesh<['devices'=2:manual]>, [], varying_manual={'devices'}}]";
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda %0:{predicate}, %1:{invariant}, %2:{varying} .
                let %3:{varying} = parallel_vary [axis_name=\"devices\"] %1
                    %4:{varying} = condition %0 %3 %2 [
                        true={{
                            lambda %0:{varying}, %1:{varying} .
                            let %2:{varying} = mul %0 %1
                            in (%2)
                        }},
                        false={{
                            lambda %0:{varying}, %1:{varying} .
                            let %2:f32[] = const 1.0
                                %3:{invariant} = broadcast [output_type={invariant}, output_axes=[]] %2
                                %4:{varying} = parallel_vary [axis_name=\"devices\"] %3
                            in (%4)
                        }},
                    ]
                in (%4)"},
        );

        // An invariant predicate keeps every device on the same branch, so nothing is aligned.
        let invariant_predicate_type = ArrayType::scalar(DataType::Boolean)
            .with_sharding(invariant_type.sharding().unwrap().clone())
            .unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |inputs: Vec<Tracer<_>>| {
                inputs[0].condition(
                    |inputs| Ok(vec![inputs[0].clone() * inputs[0].clone()]),
                    |inputs| Ok(vec![inputs[0].clone() + inputs[0].clone()]),
                    vec![inputs[1].clone()],
                )
            },
            vec![invariant_predicate_type, invariant_type],
            named_axes,
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda %0:bool[][sharding={{mesh<['devices'=2:manual]>, []}}], %1:{invariant} .
                let %2:{invariant} = condition %0 %1 [
                    true={{
                        lambda %0:{invariant} .
                        let %1:{invariant} = mul %0 %0
                        in (%1)
                    }},
                    false={{
                        lambda %0:{invariant} .
                        let %1:{invariant} = add %0 %0
                        in (%1)
                    }},
                ]
                in (%2)"},
        );
    }

    #[test]
    fn test_condition_condition_rejects_mutated_references_under_varying_predicates() {
        // Staging inspects the traced branches before binding them: a mutation of the invariant reference in only the
        // `false` branch is rejected, while reading it is accepted.
        let (invariant_referent_type, predicate_type) = manual_mutation_types(false);
        let (varying_referent_type, _) = manual_mutation_types(true);
        let mesh = predicate_type.sharding().unwrap().mesh().clone();
        let named_axes = vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh })];
        let input_types = vec![
            ArrayIrType::Array(predicate_type),
            ArrayIrType::Reference(ReferenceType::new(invariant_referent_type)),
            ArrayIrType::Reference(ReferenceType::new(varying_referent_type)),
        ];
        let trace = |mutate_invariant: bool| {
            TracingContext::<TestIrValue, TestIrOperation>::trace_with_named_axes(
                |inputs: Vec<Tracer<_>>| {
                    let access =
                        |reference: &Tracer<NestedTracingContext<TracingContext<TestIrValue, TestIrOperation>>>,
                         mutate: bool| {
                            let context = reference.context();
                            let read = ReferenceReadOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
                            let value = context.bind(TestIrOperation::from(read), Vec::new(), &[reference.clone()])?;
                            if mutate {
                                let write =
                                    ReferenceWriteOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
                                context.bind(
                                    TestIrOperation::from(write),
                                    Vec::new(),
                                    &[reference.clone(), value[0].clone()],
                                )?;
                            }
                            Ok::<_, ProgramError>(())
                        };
                    inputs[0].condition(
                        |inputs| {
                            access(&inputs[0], false)?;
                            access(&inputs[1], false)
                        },
                        |inputs| {
                            access(&inputs[0], mutate_invariant)?;
                            access(&inputs[1], true)
                        },
                        vec![inputs[1].clone(), inputs[2].clone()],
                    )
                },
                input_types.clone(),
                named_axes.clone(),
            )
            .map(|(_, program)| program)
        };
        assert_eq!(
            trace(true).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "`condition` `false` branch mutates region input 0 of type \
                 `ref<f32[2][sharding={mesh<['x'=2:manual]>, [{}]}]>`, whose referent does not vary over every manual \
                 axis that the predicate `bool[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]` varies \
                 over, so devices that take different branches would leave it different across devices; vary the \
                 referent over those axes",
            ))),
        );
        let predicate = "bool[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]";
        let invariant = "f32[2][sharding={mesh<['x'=2:manual]>, [{}]}]";
        let varying = "f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]";
        assert_eq!(
            trace(false).unwrap().to_string(),
            formatdoc! {"
                lambda %0:{predicate}, %1:ref<{invariant}>, %2:ref<{varying}> .
                let () = condition %0 %1 %2 [
                    true={{
                        lambda %0:ref<{invariant}>, %1:ref<{varying}> .
                        let %2:{invariant} = reference_read %0
                            %3:{varying} = reference_read %1
                        in ()
                    }},
                    false={{
                        lambda %0:ref<{invariant}>, %1:ref<{varying}> .
                        let %2:{invariant} = reference_read %0
                            %3:{varying} = reference_read %1
                            () = reference_write %1 %3
                        in ()
                    }},
                ]
                in ()"},
        );
    }

    #[test]
    fn test_condition_condition_returns_forwarded_inputs_directly() {
        // Both branches forward the invariant weight, which therefore equals that weight on every device whichever
        // branch runs, so it is pruned from the branches and returned as the unaligned input with its invariant type.
        // The other output is computed differently by the branches and is returned by the condition as a varying value.
        let (predicate_type, invariant_type, varying_type) = devices_varying_predicate_types();
        let (output_types, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |inputs: Vec<Tracer<_>>| {
                inputs[0].condition(
                    |inputs| Ok(vec![inputs[0].clone(), inputs[0].clone() * inputs[1].clone()]),
                    |inputs| Ok(vec![inputs[0].clone(), inputs[1].clone()]),
                    vec![inputs[1].clone(), inputs[2].clone()],
                )
            },
            vec![predicate_type, invariant_type.clone(), varying_type.clone()],
            vec![("devices".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: devices_manual_mesh() })],
        )
        .unwrap();
        assert_eq!(output_types, vec![invariant_type, varying_type]);
        let predicate = "bool[][sharding={mesh<['devices'=2:manual]>, [], varying_manual={'devices'}}]";
        let invariant = "f32[][sharding={mesh<['devices'=2:manual]>, []}]";
        let varying = "f32[][sharding={mesh<['devices'=2:manual]>, [], varying_manual={'devices'}}]";
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda %0:{predicate}, %1:{invariant}, %2:{varying} .
                let %3:{varying} = parallel_vary [axis_name=\"devices\"] %1
                    %4:{varying} = condition %0 %3 %2 [
                        true={{
                            lambda %0:{varying}, %1:{varying} .
                            let %2:{varying} = mul %0 %1
                            in (%2)
                        }},
                        false={{
                            lambda %0:{varying}, %1:{varying} .
                            in (%1)
                        }},
                    ]
                in (%1, %4)"},
        );
    }

    #[test]
    fn test_condition_condition_differentiates_invariant_inputs_outside_of_branches() {
        // Under a predicate that varies over `devices`, the condition that `condition` stages computes on the varied
        // weight, so linearization splits it with varying residuals, and the pullback sums the weight's per-device
        // cotangent across devices after the transposed condition, through the transpose of the boundary
        // `parallel_vary`, rather than inside the branches.
        let (predicate_type, invariant_type, varying_type) = devices_varying_predicate_types();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |inputs: Vec<Tracer<_>>| {
                inputs[0].condition(
                    |inputs| Ok(vec![inputs[0].clone() * inputs[1].clone()]),
                    |inputs| Ok(vec![inputs[0].clone()]),
                    vec![inputs[1].clone(), inputs[2].clone()],
                )
            },
            vec![predicate_type, invariant_type, varying_type],
            vec![("devices".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: devices_manual_mesh() })],
        )
        .unwrap();
        let linearization = program.linearize_with_respect_to(&[1]).unwrap();
        let predicate = "bool[][sharding={mesh<['devices'=2:manual]>, [], varying_manual={'devices'}}]";
        let invariant = "f32[][sharding={mesh<['devices'=2:manual]>, []}]";
        let varying = "f32[][sharding={mesh<['devices'=2:manual]>, [], varying_manual={'devices'}}]";
        assert_eq!(
            linearization.primal().to_string(),
            formatdoc! {"
                lambda %0:{predicate}, %1:{invariant}, %2:{varying} .
                let %3:{varying} = parallel_vary [axis_name=\"devices\"] %1
                    %4:{varying}, %5:{varying} = condition %0 %3 %2 [
                        true={{
                            lambda %0:{varying}, %1:{varying} .
                            let %2:{varying} = mul %0 %1
                            in (%2, %1)
                        }},
                        false={{
                            lambda %0:{varying}, %1:{varying} .
                            let %2:{varying} = zero [
                                type={varying},
                            ]
                            in (%0, %2)
                        }},
                    ]
                in (%4, %0, %5)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            formatdoc! {"
                lambda %0:{invariant}, %1:{predicate}, %2:{varying} .
                let %3:{varying} = parallel_vary [axis_name=\"devices\"] %0
                    %4:{varying} = condition %1 %3 %2 [
                        true={{
                            lambda %0:{varying}, %1:{varying} .
                            let %2:{varying} = mul %1 %0
                            in (%2)
                        }},
                        false={{
                            lambda %0:{varying}, %1:{varying} .
                            in (%0)
                        }},
                    ]
                in (%4)"},
        );
        assert_eq!(
            linearization.pullback().unwrap().program().to_string(),
            formatdoc! {"
                lambda %0:{varying}, %1:{predicate}, %2:{varying} .
                let %3:{varying} = condition %1 %0 %2 [
                    true={{
                        lambda %0:{varying}, %1:{varying} .
                        let %2:{varying} = mul %1 %0
                        in (%2)
                    }},
                    false={{
                        lambda %0:{varying}, %1:{varying} .
                        in (%0)
                    }},
                ]
                    %4:{invariant} = parallel_reduce [kind=sum, axis_name=\"devices\", mesh=['devices'=2:manual]] %3
                in (%4)"},
        );
    }

    #[test]
    fn test_condition_condition_linearizes_input_independent_residuals_under_varying_predicates() {
        // Under a predicate that varies over `devices`, the `true` branch scales its input by `sin(2.0)`, which depends
        // on no input. Its linearization saves that factor as a residual of the primal condition.
        let (predicate_type, _, varying_type) = devices_varying_predicate_types();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |inputs: Vec<Tracer<_>>| {
                inputs[0].condition(
                    |inputs| {
                        let factor = StagingContext::constant(inputs[0].context(), Array::scalar(2.0f32).unwrap());
                        Ok(vec![inputs[0].clone() * factor.sin()?])
                    },
                    |inputs| Ok(vec![inputs[0].clone()]),
                    vec![inputs[1].clone()],
                )
            },
            vec![predicate_type, varying_type],
            vec![("devices".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: devices_manual_mesh() })],
        )
        .unwrap();
        let linearization = program.linearize_with_respect_to(&[1]).unwrap();
        let predicate = "bool[][sharding={mesh<['devices'=2:manual]>, [], varying_manual={'devices'}}]";
        let invariant = "f32[][sharding={mesh<['devices'=2:manual]>, []}]";
        let varying = "f32[][sharding={mesh<['devices'=2:manual]>, [], varying_manual={'devices'}}]";
        // The factor is computed as an invariant value, but multiplying it by the varying input aligns it first, so the
        // residual that linearization saves is the varied factor, typed as the per-device value that it is, and the
        // split succeeds.
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda %0:{predicate}, %1:{varying} .
                let %2:{varying} = condition %0 %1 [
                    true={{
                        lambda %0:{varying} .
                        let %1:f32[] = const 2.0
                            %2:f32[] = sin %1
                            %3:{invariant} = broadcast [output_type={invariant}, output_axes=[]] %2
                            %4:{varying} = parallel_vary [axis_name=\"devices\"] %3
                            %5:{varying} = mul %0 %4
                        in (%5)
                    }},
                    false={{
                        lambda %0:{varying} .
                        in (%0)
                    }},
                ]
                in (%2)"},
        );
        assert_eq!(
            linearization.primal().to_string(),
            formatdoc! {"
                lambda %0:{predicate}, %1:{varying} .
                let %2:{varying}, %3:{varying} = condition %0 %1 [
                    true={{
                        lambda %0:{varying} .
                        let %1:f32[] = const 2.0
                            %2:f32[] = sin %1
                            %3:{invariant} = broadcast [output_type={invariant}, output_axes=[]] %2
                            %4:{varying} = parallel_vary [axis_name=\"devices\"] %3
                            %5:{varying} = mul %0 %4
                        in (%5, %4)
                    }},
                    false={{
                        lambda %0:{varying} .
                        let %1:{varying} = zero [
                            type={varying},
                        ]
                        in (%0, %1)
                    }},
                ]
                in (%2, %0, %3)"},
        );
        let tangent = formatdoc! {"
            lambda %0:{varying}, %1:{predicate}, %2:{varying} .
            let %3:{varying} = condition %1 %0 %2 [
                true={{
                    lambda %0:{varying}, %1:{varying} .
                    let %2:{varying} = mul %1 %0
                    in (%2)
                }},
                false={{
                    lambda %0:{varying}, %1:{varying} .
                    in (%0)
                }},
            ]
            in (%3)"};
        assert_eq!(linearization.tangent().to_string(), tangent);
        assert_eq!(linearization.pullback().unwrap().program().to_string(), tangent);
    }
}
