//! The [`CompositeOperation`] that calls a primal program with custom derivative rules. The operation holds its rules
//! either as attached rule regions or as retained rule sets (refer to [`CustomRuleSource`]), and reverse mode replaces
//! a call with reverse-mode rules with a [`CompositeTransposeOperation`] carrier.

// TODO(eaplatanios): Review this module.

use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use crate::axes::Axis;
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedProgram, BatchingContext, BatchingDriver, BatchingError, BatchingLevelExtent,
    BatchingPolicy, ProgramBatchingOutputAxesPolicy,
};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{
    CotangentAccumulator, CotangentBatchingPolicy, CotangentDestinationKind, DifferentiableOperation,
    DifferentiableType, DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    DifferentiationPolicy, ResidualZeroProvider, TransposableOperation, TranspositionContext, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, check_types};
use crate::operations::arithmetic::AddOperation;
use crate::operations::composites::operations::transpose::{CompositeTransposeOperation, CompositeTransposeRules};
use crate::operations::composites::rules::{
    CustomRuleBatching, CustomRuleBatchingLevel, CustomRuleBatchingRuleKey, CustomRuleDerivationKey,
    CustomRuleDerivationKind, CustomRuleReference, CustomRuleSource, CustomRuleSpecializationKey,
    CustomRuleSpecializer, CustomRuleTracer, derive_bijective_identity_renaming,
};
use crate::operations::composites::{
    replay_custom_jvp_rule, retained_carrier_batching, validate_composite_discharge_inputs,
    validate_composite_reference_boundary, validate_composite_replay, validate_deferred_forward_rule,
    validate_non_differentiated_count,
};
use crate::operations::constants::zero::Zero;
use crate::parameters::{Parameter, Placeholder};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    EffectsSummary, InputRegionProvenance, MaybeZero, Operation, OperationBoundaryPruning, OperationFormatter,
    OperationProvider, OutputRegionProvenance, ProgramBuilder, ProgramError, ReferenceDischargeContext,
    ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeValue, ReferenceDischargeableOperation,
    RegionDataFlow, RegionInterface, RegionLiveness, RegionSlot, Type, TypeError, TypeIdentityRenaming, Typed, Value,
    discharge_local_reference_operation,
};
use crate::tracing::TracingContext;

/// Canonical operation name for [`CompositeOperation`].
pub const COMPOSITE_OPERATION_NAME: &str = "composite";

/// Forward-mode (i.e., Jacobian-Vector Product or JVP) rule of a [`CompositeOperation`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum CompositeJvpRule {
    /// No forward-mode rule. Forward mode is rejected, and reverse mode requires a reverse-mode rule.
    #[default]
    Absent,

    /// Explicit forward-mode rule `(p, x, ẋ) → (y, ẏ)`, which is a `"jvp"` rule region of a call with attached rules
    /// and a retained callback of a call with retained rules.
    Explicit,

    /// Forward-mode rule derived by differentiating the primal region, which is an explicit registration choice
    /// rather than a fallback for a missing rule.
    Primal,
}

/// Derivative rules of a [`CompositeOperation`], in one of its two representations.
#[derive(Clone, PartialEq, Eq, Hash)]
enum CompositeRules<T, S> {
    /// Rule programs attached as rule regions that follow the primal region.
    Attached {
        /// Forward-mode rule, whose explicit form is a `"jvp"` rule region.
        jvp_rule: CompositeJvpRule,

        /// Whether the call has `"forward"` and `"backward"` reverse-mode rule regions.
        has_vjp_rule: bool,
    },

    /// Rules retained by a source and traced lazily, together with the call state that their specializations depend
    /// on.
    Retained {
        /// Source of the retained rules.
        rules: S,

        /// Batching applied to the call after it was staged, or [`None`] for an unbatched call.
        batching: Option<CustomRuleBatching<T>>,

        /// Whether the call's reference state was discharged, which discharges its rule programs as well (refer to
        /// [`CustomRuleDefinition::with_reference_discharge`](crate::CustomRuleDefinition::with_reference_discharge)).
        discharged: bool,

        /// Whether each output of the rules is an output of the call, or [`None`] when the call returns every output
        /// of its rules. Pruning the unused outputs of a call (refer to [`Operation::prune_boundary`]) prunes them
        /// from its primal region and records them here, because its rules are traced for the complete output
        /// structure: the rule specializations of the call project the kept outputs, and its reverse-mode carrier
        /// keeps every output of the forward rule, whose pruned outputs then receive structural-zero seeds.
        kept_outputs: Option<Vec<bool>>,
    },
}

/// Higher-order [`Operation`] that calls a primal program with custom derivative rules. It is the operation that
/// [`Composite`](crate::Composite) functions stage, and it is the one boundary through which every transform treats a
/// function whose derivative is not the derivative of its implementation (e.g., a fused or foreign kernel, or a
/// numerically stabilized derivative). The primal program is attached as the computation region `"primal"`, and
/// ordinary execution, lowering, and partial evaluation use it exclusively. A call has any of the following rules:
///
///   - a forward-mode rule (refer to [`CompositeJvpRule`]), which is either an explicit Jacobian-Vector Product (JVP)
///     rule or the derivative of the primal region, and
///   - reverse-mode forward and backward (i.e., Vector-Jacobian Product or VJP) rules.
///
/// # Representations
///
/// The rules have one of two representations, which differ only in where the rule programs come from and in how those
/// programs follow batching and reference discharge:
///
///   - **Attached rules** ([`Self::from_rule_regions`]) are programs supplied as the operation's attached rule regions
///     (i.e., via the [`RegionDriver`](crate::RegionDriver) passed to [`Context::bind`]), in the region order
///     `["primal", "jvp", "forward", "backward"]`, where the rule regions that a call does not have are omitted. They
///     are used for pre-traced rule programs (e.g., by kernel staging), and they follow their call through every
///     program transform and family conversion like any other region.
///   - **Retained rules** ([`Self::new`]) are callbacks of a [`CustomRuleSource`] (e.g., a registered
///     [`CustomRuleDefinition`](crate::CustomRuleDefinition)), so an ordinary call traces only its primal and never
///     declares the effects of its rules. Each rule is traced on the first request of each static specialization and
///     cached by the definition's [`CustomRuleRegistration`](crate::CustomRuleRegistration).
///
/// # Interfaces
///
/// Writing the leading [`non_differentiated_count`](Self::non_differentiated_count) inputs as `p`, the remaining
/// _differentiated_ inputs as `x`, the primal outputs as `y`, the forward residuals as `r`, and tangents and cotangents
/// using a dot and an overbar, respectively, the rule interfaces are:
///
///   - **Primal:**   `(p, x) → y`,
///   - **JVP:**      `(p, x, ẋ) → (y, ẏ)`, with one tangent per differentiated input and one tangent per primal output,
///   - **Forward:**  `(p, x) → (y, r)`, with arbitrarily many residuals following the primal outputs, and
///   - **Backward:** `(p, r, ȳ) → x̄`, with one cotangent per primal output and one cotangent per differentiated input.
///
/// [`Operation::infer_output_types`] validates that attached rule regions realize exactly these interfaces, that only
/// `p` contains references, and that no output is a reference. Retained rules are validated when they are first traced.
/// A reference-typed residual must additionally have the type of a distinct plumbing input in `p`, because the forward
/// rule may hand a plumbing reference to the backward rule only by forwarding that input by identity (which
/// [`Composite::call`](crate::Composite::call) checks on the traced forward program). Keeping `p` explicit while
/// omitting its tangent and cotangent from the rules distinguishes an input that parameterizes the rules from an
/// ordinary input whose derivative merely happens to be zero. This is the same input split that
/// [`LinearCallOperation`](crate::LinearCallOperation) draws with its residual count. Batching is a canonical producer
/// of such inputs: a batching policy that threads batching state through a structurally batched region's boundary
/// (e.g., the array IR universe's first-class mapped extent) reintroduces that state as additional leading
/// non-differentiated inputs of the batched call.
///
/// # Rule Selection
///
/// The rules that a call has determine how each transform differentiates it:
///
/// | Rules                           | Forward mode (`jvp`)           | Reverse mode (`jvp_for_transpose`)   |
/// | ------------------------------- | ------------------------------ | ------------------------------------ |
/// | None                            | Rejected                       | Rejected                             |
/// | JVP rule (explicit or primal)   | Replays or derives the JVP     | Transposes the linearized JVP        |
/// | VJP rules                       | Rejected                       | Replays forward and stages backward  |
/// | JVP rule and VJP rules          | Replays or derives the JVP     | Replays forward and stages backward  |
///
/// A missing rule never silently falls back to differentiating the primal region, because that would replace a
/// deliberately custom derivative with the derivative of the primal implementation. Differentiating the primal region
/// is instead an explicit registration choice (i.e., [`CompositeJvpRule::Primal`]). Reverse mode replays the forward
/// rule and stages a [`CompositeTransposeOperation`] carrier for the backward rule, in the same representation as the
/// call. Retained rules may instead defer their forward rule to the backward pass (refer to
/// [`CustomRuleSource::defers_forward`], rendered as `deferred_forward=true`), in which case reverse mode replays the
/// primal region and stages a carrier over the call's inputs, whose transposition replays the forward rule to recover
/// the residuals before it applies the backward rule.
///
/// # Batching
///
/// Batching a call with attached rules batches every attached region contract while retaining the call and its rules,
/// reconciling the batch axes of the logical outputs across the primal and rule regions. Batching a call with retained
/// rules batches its primal region and records the batching level, but traces no derivative rule. Its batched outputs
/// use the definition's declared layout (refer to
/// [`CustomRuleDefinition::with_batched_output_axes`](crate::CustomRuleDefinition::with_batched_output_axes)), which
/// maps every output at axis 0 by default. A definition's custom batching rule (refer to
/// [`CustomRuleDefinition::with_batching_rule`](crate::CustomRuleDefinition::with_batching_rule)) takes precedence over
/// structurally batching the primal region: its traced program becomes the batched call's primal region, and it
/// declares the batch axes of the outputs. Differentiating a batched call with retained rules traces the rule at the
/// unbatched types, sharing that specialization with unbatched calls, and batches the traced program once per recorded
/// level (refer to
/// [`CustomRuleDefinition::with_batching_support`](crate::CustomRuleDefinition::with_batching_support)). A
/// derivative whose natural layout is incompatible with the declared layout is therefore rejected when it is first
/// requested, not when the call is batched. In both representations, any boundary inputs of the batching policy (e.g.,
/// a first-class batch extent) become leading non-differentiated inputs. A definition that batches its calls
/// sequentially (refer to
/// [`CustomRuleDefinition::with_sequential_batching`](crate::CustomRuleDefinition::with_sequential_batching)) without a
/// custom batching rule is the exception: batching replaces the call with a scan over the batch items whose body
/// applies the unbatched call, so the batched call's derivatives are those of the scan.
///
/// A custom batching rule also stays on the path of forward-mode derivatives that are batched after they are taken.
/// Forward-mode differentiation of an unbatched call with retained rules whose forward-mode rule is derived from its
/// primal stages a derived call (with rules that [`CustomRuleSpecializer::derived`] derives) instead of inlining the
/// derivative of the primal region, and batching that call applies the derivative of the batching rule. The derived
/// call's primal region is that derivative: in fused contexts, it computes the outputs and their tangents, while
/// partitioned contexts compute the outputs with the call itself and stage a pushforward that computes only the
/// tangents (for a pure primal region, since the pushforward recomputes the primal; other primals are linearized
/// inline, so that their effects run once). A pushforward is linear in its tangent inputs, so it transposes by
/// transposing its primal region inline (e.g., when a linearization is transposed). Reverse mode inlines the derivative
/// instead of staging a derived call, because that transposition would drop the batching rule anyway.
///
/// # Dead Code Elimination
///
/// Pruning the unused outputs of a call (refer to [`Program::into_pruned`](crate::Program::into_pruned)) removes them
/// from its primal region while keeping every input, because the rules may use inputs that the pruned primal does not,
/// which is how JAX's dead code elimination treats its custom derivative calls. A call with retained rules records
/// which outputs of its rules it keeps (rendered as `kept_outputs`), because its rules are traced for the complete
/// output structure: its forward-mode and batching rules are projected onto the kept outputs, and its reverse-mode
/// carrier keeps every output of the forward rule, whose pruned outputs receive structural-zero seeds. A call with
/// attached rule regions, whose boundaries mirror the complete primal boundary, stays whole, and so does a call whose
/// forward-mode rule is derived from its primal while its rules have a custom batching rule, because the derived calls
/// that its derivatives stage apply that rule to every output.
///
/// Equality and hashing of calls with retained rules use the identity of the shared definition (i.e.,
/// [`Arc::ptr_eq`](std::sync::Arc::ptr_eq)), while rendering prints its human-readable name.
pub struct CompositeOperation<V: Typed + Parameter, O, S = CustomRuleReference<V, O>> {
    /// Number of leading inputs that parameterize the call without being differentiated, including the boundary inputs
    /// that the batching levels of a call with retained rules prepended to its inputs.
    non_differentiated_count: usize,

    /// Derivative rules of the call.
    rules: CompositeRules<V::Type, S>,

    /// Marker for the operation family, which only the rule source and the rule regions name.
    marker: PhantomData<fn() -> O>,
}

impl<V: Typed<Type: DifferentiableType + Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>>
    CompositeOperation<V, O, S>
{
    /// Creates a new call of the retained rules of `rules`, whose inputs are all differentiated.
    #[inline]
    pub fn new(rules: S) -> Self {
        Self {
            non_differentiated_count: 0,
            rules: CompositeRules::Retained { rules, batching: None, discharged: false, kept_outputs: None },
            marker: PhantomData,
        }
    }

    /// Creates a new call with attached rule programs, whose inputs are all differentiated. An explicit forward-mode
    /// rule (i.e., [`CompositeJvpRule::Explicit`]) is a `"jvp"` rule region that follows the primal region, and
    /// reverse-mode rules are `"forward"` and `"backward"` rule regions that follow the primal region and the optional
    /// `"jvp"` rule region.
    #[inline]
    pub fn from_rule_regions(jvp_rule: CompositeJvpRule, has_vjp_rule: bool) -> Self {
        Self {
            non_differentiated_count: 0,
            rules: CompositeRules::Attached { jvp_rule, has_vjp_rule },
            marker: PhantomData,
        }
    }

    /// Returns this call with the provided number of leading non-differentiated inputs. Refer to the documentation of
    /// [`CompositeOperation`] for the impact of this property on the rule interfaces. For a batched call with
    /// retained rules, this count includes the boundary inputs that its batching levels prepended to its inputs.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] when the call is a batched call with retained rules and `non_differentiated_count` is
    /// smaller than the number of those boundary inputs, which are always non-differentiated.
    pub fn with_non_differentiated_count(mut self, non_differentiated_count: usize) -> Result<Self, TypeError> {
        if let CompositeRules::Retained { rules, batching: Some(batching), .. } = &self.rules {
            let boundary_input_count = batching.boundary_input_count();
            if non_differentiated_count < boundary_input_count {
                return Err(TypeError::invalid(format!(
                    "batched {} must have at least {boundary_input_count} \
                     non-differentiated inputs, which are the boundary inputs that its batching levels prepended to \
                     its inputs, but got {non_differentiated_count}",
                    {
                        let name = rules.name();
                        if name == COMPOSITE_OPERATION_NAME {
                            format!("`{COMPOSITE_OPERATION_NAME}`")
                        } else {
                            format!("`{COMPOSITE_OPERATION_NAME}` `{name}`")
                        }
                    },
                )));
            }
        }
        self.non_differentiated_count = non_differentiated_count;
        Ok(self)
    }

    /// Returns the number of leading inputs that parameterize this call without being differentiated.
    #[inline]
    pub fn non_differentiated_count(&self) -> usize {
        self.non_differentiated_count
    }

    /// Returns the source of the retained rules of this call, or [`None`] when its rules are attached.
    #[inline]
    pub fn rules(&self) -> Option<&S> {
        match &self.rules {
            CompositeRules::Attached { .. } => None,
            CompositeRules::Retained { rules, .. } => Some(rules),
        }
    }

    /// Returns the forward-mode rule of this call.
    pub fn jvp_rule(&self) -> CompositeJvpRule {
        match &self.rules {
            CompositeRules::Attached { jvp_rule, .. } => *jvp_rule,
            CompositeRules::Retained { rules, .. } if rules.has_jvp() => CompositeJvpRule::Explicit,
            CompositeRules::Retained { rules, .. } if rules.derives_jvp_from_primal() => CompositeJvpRule::Primal,
            CompositeRules::Retained { .. } => CompositeJvpRule::Absent,
        }
    }

    /// Returns whether this call has reverse-mode forward and backward rules.
    pub fn has_vjp_rule(&self) -> bool {
        match &self.rules {
            CompositeRules::Attached { has_vjp_rule, .. } => *has_vjp_rule,
            CompositeRules::Retained { rules, .. } => rules.has_vjp(),
        }
    }

    /// Converts this call into the family `(V2, O2)` with the rule source that `rules_fn` derives from its current
    /// source. This is how a call follows its program into a family that contains its family (e.g., with
    /// [`LiftedCustomRules::Member`](crate::LiftedCustomRules::Member) when an array program is converted into an
    /// array IR program). A call with attached rules carries no source, so `rules_fn` is only invoked for retained
    /// rules. Refer to [`Self::into_attached_family`] for converting calls with attached rules into a family whose
    /// source type the caller chooses independently.
    pub fn into_family<V2, O2, S2, F>(self, rules_fn: F) -> CompositeOperation<V2, O2, S2>
    where
        V2: Typed<Type: DifferentiableType + Eq + Hash + From<V::Type>> + Parameter,
        F: FnOnce(S) -> S2,
    {
        CompositeOperation {
            non_differentiated_count: self.non_differentiated_count,
            rules: match self.rules {
                CompositeRules::Attached { jvp_rule, has_vjp_rule } => {
                    CompositeRules::Attached { jvp_rule, has_vjp_rule }
                }
                CompositeRules::Retained { rules, batching, discharged, kept_outputs } => CompositeRules::Retained {
                    rules: rules_fn(rules),
                    batching: batching.map(CustomRuleBatching::map_types),
                    discharged,
                    kept_outputs,
                },
            },
            marker: PhantomData,
        }
    }

    /// Converts this call into the family `(V2, O2)` with any rule source type when its rules are attached, because
    /// attached rules are regions that follow their call independently of any source. Returns this call unchanged when
    /// its rules are retained, which [`Self::into_family`] converts instead. Family conversions use this function to
    /// normalize every call with attached rules into the family's native variant.
    pub fn into_attached_family<V2, O2, S2>(self) -> Result<CompositeOperation<V2, O2, S2>, Self>
    where
        V2: Typed + Parameter,
    {
        match self.rules {
            CompositeRules::Attached { jvp_rule, has_vjp_rule } => Ok(CompositeOperation {
                non_differentiated_count: self.non_differentiated_count,
                rules: CompositeRules::Attached { jvp_rule, has_vjp_rule },
                marker: PhantomData,
            }),
            CompositeRules::Retained { .. } => Err(self),
        }
    }

    /// Returns a description of this call for diagnostics, which names the definition of retained rules unless it has
    /// the default name of [`Composite`](crate::Composite) functions.
    fn description(&self) -> String {
        match &self.rules {
            CompositeRules::Retained { rules, .. } if rules.name() != COMPOSITE_OPERATION_NAME => {
                format!("a `{COMPOSITE_OPERATION_NAME}` call of `{}`", rules.name())
            }
            _ => format!("a `{COMPOSITE_OPERATION_NAME}` call"),
        }
    }

    /// Returns the index of the `"jvp"` rule region, if this call has attached rules with one.
    #[inline]
    fn jvp_region_index(&self) -> Option<usize> {
        matches!(self.rules, CompositeRules::Attached { jvp_rule: CompositeJvpRule::Explicit, .. }).then_some(1)
    }

    /// Returns the indices of the `"forward"` and `"backward"` rule regions, if this call has attached rules with
    /// them.
    #[inline]
    fn vjp_region_indices(&self) -> Option<(usize, usize)> {
        let CompositeRules::Attached { has_vjp_rule: true, .. } = self.rules else {
            return None;
        };
        let forward_region_index = 1 + usize::from(self.jvp_region_index().is_some());
        Some((forward_region_index, forward_region_index + 1))
    }

    /// Splits the provided input `values` into the leading non-differentiated group and the trailing differentiated
    /// group, based on the value of [`Self::non_differentiated_count`].
    #[inline]
    fn split_inputs<'v, T>(&self, values: &'v [T]) -> Result<(&'v [T], &'v [T]), TypeError> {
        validate_non_differentiated_count(self.name(), self.non_differentiated_count, values.len())?;
        Ok(values.split_at(self.non_differentiated_count))
    }

    /// Validates the composite contract over the attached region interfaces (refer to the documentation of
    /// [`CompositeOperation`] for that contract) and returns the primal interface.
    fn validated_interfaces<'i>(
        &self,
        region_interfaces: &'i [RegionInterface<V::Type>],
    ) -> Result<&'i RegionInterface<V::Type>, TypeError> {
        check_count!("region", region_interfaces, self.region_slots().len(), TypeError);
        let primal_interface = &region_interfaces[0];
        let input_types = primal_interface.input_types();
        let output_types = primal_interface.output_types();
        let (non_differentiated_types, differentiated_types) = self.split_inputs(input_types)?;
        validate_composite_reference_boundary(
            COMPOSITE_OPERATION_NAME,
            self.non_differentiated_count,
            input_types,
            output_types,
        )?;

        if let Some(jvp_region_index) = self.jvp_region_index() {
            let jvp_interface = &region_interfaces[jvp_region_index];
            let mut expected_jvp_input_types = input_types.to_vec();
            expected_jvp_input_types.extend(
                differentiated_types
                    .iter()
                    .map(DifferentiableType::tangent)
                    .collect::<Result<Vec<_>, DifferentiationError>>()?,
            );
            check_types!(@same, format!("`{COMPOSITE_OPERATION_NAME}` JVP rule input"), [
                &expected_jvp_input_types,
                jvp_interface.input_types(),
            ]);
            let mut expected_jvp_output_types = output_types.to_vec();
            expected_jvp_output_types.extend(
                output_types
                    .iter()
                    .map(DifferentiableType::tangent)
                    .collect::<Result<Vec<_>, DifferentiationError>>()?,
            );
            check_types!(@same, format!("`{COMPOSITE_OPERATION_NAME}` JVP rule output"), [
                &expected_jvp_output_types,
                jvp_interface.output_types(),
            ]);
        }

        if let Some((forward_region_index, backward_region_index)) = self.vjp_region_indices() {
            let forward_interface = &region_interfaces[forward_region_index];
            let backward_interface = &region_interfaces[backward_region_index];
            check_types!(@same, format!("`{COMPOSITE_OPERATION_NAME}` forward rule input"), [
                input_types,
                forward_interface.input_types(),
            ]);
            let forward_output_types = forward_interface.output_types();
            if forward_output_types.len() < output_types.len() {
                return Err(TypeError::invalid(format!(
                    "`{}` forward rule must produce at least the {} primal output(s) but produced {} value(s)",
                    COMPOSITE_OPERATION_NAME,
                    output_types.len(),
                    forward_output_types.len(),
                )));
            }
            check_types!(@same, format!("`{COMPOSITE_OPERATION_NAME}` forward rule output"), [
                output_types,
                &forward_output_types[..output_types.len()],
            ]);

            // Reference residuals must have types compatible with distinct leading non-differentiated inputs. This is
            // only a necessary interface constraint: two inputs can have the same reference type. Tracing the forward
            // rule of a `Composite` checks the actual forwarded input identities; raw program producers must ensure
            // that same contract when attaching their forward regions.
            let residual_types = &forward_output_types[output_types.len()..];
            let mut forwarded = vec![false; non_differentiated_types.len()];
            for (index, residual_type) in residual_types.iter().enumerate().filter(|(_, r#type)| r#type.is_reference())
            {
                let available = non_differentiated_types
                    .iter()
                    .zip(forwarded.iter_mut())
                    .find(|(r#type, forwarded)| *r#type == residual_type && !**forwarded);
                match available {
                    Some((_, forwarded)) => *forwarded = true,
                    None if non_differentiated_types.contains(residual_type) => {
                        return Err(TypeError::invalid(format!(
                            "`{COMPOSITE_OPERATION_NAME}` forward rule returns residual {index} of reference \
                             type `{residual_type}`, but every leading non-differentiated input of that type is \
                             already forwarded by an earlier residual",
                        )));
                    }
                    None => {
                        return Err(TypeError::invalid(format!(
                            "`{COMPOSITE_OPERATION_NAME}` forward rule returns residual {index} of reference \
                             type `{residual_type}`, which matches none of its leading non-differentiated inputs",
                        )));
                    }
                }
            }

            let output_cotangent_types = output_types
                .iter()
                .map(DifferentiableType::cotangent)
                .collect::<Result<Vec<_>, DifferentiationError>>()?;
            let expected_backward_input_types: Vec<V::Type> = non_differentiated_types
                .iter()
                .chain(residual_types)
                .cloned()
                .chain(output_cotangent_types)
                .collect();
            check_types!(@same, format!("`{COMPOSITE_OPERATION_NAME}` backward rule input"), [
                &expected_backward_input_types,
                backward_interface.input_types(),
            ]);
            let expected_backward_output_types = differentiated_types
                .iter()
                .map(DifferentiableType::cotangent)
                .collect::<Result<Vec<_>, DifferentiationError>>()?;
            check_types!(@same, format!("`{COMPOSITE_OPERATION_NAME}` backward rule output"), [
                &expected_backward_output_types,
                backward_interface.output_types(),
            ]);
        }
        Ok(primal_interface)
    }

    /// Returns the specialization key of this call with retained rules, given the current types of its inputs and
    /// outputs and which outputs of its rules it keeps (refer to the `kept_outputs` field of its retained rules).
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when a batched call has fewer non-differentiated inputs than its batching levels
    /// recorded boundary inputs, which type inference rejects before the call is staged.
    fn specialization_key(
        &self,
        rules: &S,
        batching: &Option<CustomRuleBatching<V::Type>>,
        discharged: bool,
        kept_outputs: &Option<Vec<bool>>,
        input_types: Vec<V::Type>,
        output_types: Vec<V::Type>,
    ) -> Result<CustomRuleSpecializationKey<V::Type>, ProgramError> {
        Ok(match batching {
            None => CustomRuleSpecializationKey {
                input_types,
                output_types,
                kept_outputs: kept_outputs.clone(),
                non_differentiated_count: self.non_differentiated_count,
                tangent_activity: Vec::new(),
                levels: Vec::new(),
                discharged,
            },
            Some(batching) => CustomRuleSpecializationKey {
                input_types: batching.input_types.clone(),
                output_types: batching.output_types.clone(),
                kept_outputs: kept_outputs.clone(),
                non_differentiated_count: self
                    .non_differentiated_count
                    .checked_sub(batching.boundary_input_count())
                    .ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "batched {} has fewer non-differentiated inputs \
                             than boundary inputs",
                            {
                                let name = rules.name();
                                if name == COMPOSITE_OPERATION_NAME {
                                    format!("`{COMPOSITE_OPERATION_NAME}`")
                                } else {
                                    format!("`{COMPOSITE_OPERATION_NAME}` `{name}`")
                                }
                            },
                        ))
                    })?,
                tangent_activity: Vec::new(),
                levels: batching.levels.clone(),
                discharged,
            },
        })
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> Clone for CompositeOperation<V, O, S> {
    fn clone(&self) -> Self {
        Self { non_differentiated_count: self.non_differentiated_count, rules: self.rules.clone(), marker: PhantomData }
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> Debug for CompositeOperation<V, O, S> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Retained rules are identified by the name of their definition, whose own rendering would include its rules.
        // The deferral of a forward rule is a property of the rules that is shown only where it applies.
        let mut debug = formatter.debug_struct("CompositeOperation");
        match &self.rules {
            CompositeRules::Attached { jvp_rule, has_vjp_rule } => debug
                .field("non_differentiated_count", &self.non_differentiated_count)
                .field("jvp_rule", jvp_rule)
                .field("has_vjp_rule", has_vjp_rule),
            CompositeRules::Retained { rules, batching, discharged, kept_outputs } => {
                debug
                    .field("name", &rules.name())
                    .field("non_differentiated_count", &self.non_differentiated_count)
                    .field("batching", batching)
                    .field("discharged", discharged)
                    .field("kept_outputs", kept_outputs);
                if rules.defers_forward() {
                    debug.field("deferred_forward", &true);
                }
                &mut debug
            }
        }
        .finish()
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> PartialEq for CompositeOperation<V, O, S> {
    fn eq(&self, other: &Self) -> bool {
        self.non_differentiated_count == other.non_differentiated_count && self.rules == other.rules
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> Eq for CompositeOperation<V, O, S> {}

impl<V: Typed<Type: Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> Hash for CompositeOperation<V, O, S> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.non_differentiated_count.hash(state);
        self.rules.hash(state);
    }
}

impl<V: Typed<Type: DifferentiableType + Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> Display
    for CompositeOperation<V, O, S>
{
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<V: Typed<Type: DifferentiableType + Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> Operation
    for CompositeOperation<V, O, S>
{
    type Type = V::Type;

    #[inline]
    fn name(&self) -> &'static str {
        COMPOSITE_OPERATION_NAME
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        const PRIMAL: RegionSlot = RegionSlot::computation("primal");
        const JVP: RegionSlot = RegionSlot::rule("jvp");
        const FORWARD: RegionSlot = RegionSlot::rule("forward");
        const BACKWARD: RegionSlot = RegionSlot::rule("backward");
        match (self.jvp_region_index().is_some(), self.vjp_region_indices().is_some()) {
            (false, false) => &[PRIMAL],
            (true, false) => &[PRIMAL, JVP],
            (false, true) => &[PRIMAL, FORWARD, BACKWARD],
            (true, true) => &[PRIMAL, JVP, FORWARD, BACKWARD],
        }
    }

    fn infer_region_input_types(
        &self,
        input_types: &[V::Type],
        region_interfaces: &[RegionInterface<V::Type>],
    ) -> Result<Vec<Option<Vec<V::Type>>>, TypeError> {
        // The primal region receives every input of the call at its own position.
        check_count!("region", region_interfaces, self.region_slots().len(), TypeError);
        let (non_differentiated_types, differentiated_types) = self.split_inputs(input_types)?;
        let mut region_input_types = vec![Some(input_types.to_vec())];
        if self.jvp_region_index().is_some() {
            let mut jvp_input_types = input_types.to_vec();
            jvp_input_types.extend(
                differentiated_types
                    .iter()
                    .map(DifferentiableType::tangent)
                    .collect::<Result<Vec<_>, DifferentiationError>>()?,
            );
            region_input_types.push(Some(jvp_input_types));
        }
        if let Some((forward_region_index, _)) = self.vjp_region_indices() {
            // The primal and forward regions were traced independently, so each boundary owns its own formal
            // identities. Derive each region's caller-specific renaming from its input boundary before using its
            // outputs to construct the backward region's input signature.
            let primal_interface = &region_interfaces[0];
            let forward_interface = &region_interfaces[forward_region_index];
            let primal_renaming = V::Type::derive_identity_renaming(primal_interface.input_types(), input_types)?;
            let primal_output_types = primal_interface
                .output_types()
                .iter()
                .map(|r#type| r#type.rename_identities(&primal_renaming))
                .collect::<Result<Vec<_>, _>>()?;
            let forward_renaming = V::Type::derive_identity_renaming(forward_interface.input_types(), input_types)?;
            let forward_output_types = forward_interface
                .output_types()
                .iter()
                .map(|r#type| r#type.rename_identities(&forward_renaming))
                .collect::<Result<Vec<_>, _>>()?;
            if forward_output_types.len() < primal_output_types.len() {
                return Err(TypeError::invalid(format!(
                    "`{}` forward rule must produce at least the {} primal output(s) but produced {} value(s)",
                    COMPOSITE_OPERATION_NAME,
                    primal_output_types.len(),
                    forward_output_types.len(),
                )));
            }
            let mut backward_input_types = non_differentiated_types.to_vec();
            backward_input_types.extend_from_slice(&forward_output_types[primal_output_types.len()..]);
            backward_input_types.extend(
                primal_output_types
                    .iter()
                    .map(DifferentiableType::cotangent)
                    .collect::<Result<Vec<_>, DifferentiationError>>()?,
            );
            region_input_types.push(Some(input_types.to_vec()));
            region_input_types.push(Some(backward_input_types));
        }
        Ok(region_input_types)
    }

    fn infer_output_types(
        &self,
        input_types: &[V::Type],
        region_interfaces: &[RegionInterface<V::Type>],
    ) -> Result<Vec<V::Type>, TypeError> {
        match &self.rules {
            // Output inference is a standalone validation entry point, so it must validate the complete
            // post-instantiation region contract even when `infer_region_input_types` was not called first.
            CompositeRules::Attached { .. } => {
                let primal_interface = self.validated_interfaces(region_interfaces)?;
                check_types!(@same, format!("`{COMPOSITE_OPERATION_NAME}` input"), [
                    primal_interface.input_types(),
                    input_types,
                ]);
                Ok(primal_interface.output_types().to_vec())
            }
            CompositeRules::Retained { rules, batching, .. } => {
                check_count!("region", region_interfaces, 1, TypeError);
                validate_non_differentiated_count(
                    COMPOSITE_OPERATION_NAME,
                    self.non_differentiated_count,
                    input_types.len(),
                )?;

                // A batched call's inputs start with the boundary inputs of its batching levels, followed by the inputs
                // of the unbatched call (and `with_non_differentiated_count` keeps those boundary inputs
                // non-differentiated).
                if let Some(batching) = batching {
                    let boundary_input_count = batching.boundary_input_count();
                    if input_types.len() != boundary_input_count + batching.input_types.len() {
                        return Err(TypeError::invalid(format!(
                            "batched {} has {} inputs but its batching levels \
                             record {} boundary inputs and {} unbatched inputs",
                            {
                                let name = rules.name();
                                if name == COMPOSITE_OPERATION_NAME {
                                    format!("`{COMPOSITE_OPERATION_NAME}`")
                                } else {
                                    format!("`{COMPOSITE_OPERATION_NAME}` `{name}`")
                                }
                            },
                            input_types.len(),
                            boundary_input_count,
                            batching.input_types.len(),
                        )));
                    }
                }
                check_types!(@same, format!("`{COMPOSITE_OPERATION_NAME}` input"), [
                    region_interfaces[0].input_types(),
                    input_types,
                ]);
                validate_composite_reference_boundary(
                    COMPOSITE_OPERATION_NAME,
                    self.non_differentiated_count,
                    input_types,
                    region_interfaces[0].output_types(),
                )?;
                Ok(region_interfaces[0].output_types().to_vec())
            }
        }
    }

    #[inline]
    fn input_region_provenance(&self, region_index: usize, input_index: usize) -> InputRegionProvenance {
        // The primal computation region receives every input at its own position. Attached rule regions are dormant
        // rules that reference analysis does not enter, so they declare no provenance.
        if region_index == 0 {
            InputRegionProvenance::Input { index: input_index }
        } else {
            InputRegionProvenance::None
        }
    }

    #[inline]
    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        // Interpretation replays the primal region, whose outputs are the call's outputs one for one.
        vec![OutputRegionProvenance { region_index: 0, output_index }]
    }

    #[inline]
    fn region_data_flow(&self) -> RegionDataFlow<'_> {
        RegionDataFlow::Provenance
    }

    fn prune_boundary(
        &self,
        input_types: &[V::Type],
        used_outputs: &[bool],
        _regions: &mut dyn RegionLiveness,
    ) -> Result<Option<OperationBoundaryPruning<Self>>, ProgramError> {
        // Like JAX's dead code elimination of `custom_jvp_call` and `custom_vjp_call`, the unused outputs of a call are
        // pruned from its primal region while every input is kept, because its rules may use inputs that the pruned
        // primal does not. A call whose outputs are all used, or all unused (i.e., one that is kept for its effects),
        // stays whole.
        if used_outputs.iter().all(|used| *used) || !used_outputs.contains(&true) {
            return Ok(None);
        }
        let rules = match &self.rules {
            // Attached rule regions mirror the complete output boundary of the primal region, which the region
            // provenance of the call cannot prune together with it, so only a call without rule regions is pruned (its
            // forward-mode rule, if any, is derived from its pruned primal region).
            CompositeRules::Attached { jvp_rule, has_vjp_rule } => {
                if *jvp_rule == CompositeJvpRule::Explicit || *has_vjp_rule {
                    return Ok(None);
                }
                self.rules.clone()
            }

            // The derived calls with which a forward-mode rule derived from the primal keeps a custom batching rule on
            // the path of batched derivatives (refer to `CustomRuleSpecializer::derived`) apply that rule to every
            // output, so such calls stay whole.
            CompositeRules::Retained { rules, .. } if rules.derives_jvp_from_primal() && rules.has_batching_rule() => {
                return Ok(None);
            }

            // The rules are traced for the complete output structure, so the call records which of their outputs it
            // keeps, composing this pruning with any earlier one, and its batching record, which describes the call's
            // own outputs, keeps the used ones.
            CompositeRules::Retained { rules, batching, discharged, kept_outputs } => {
                let mut used = used_outputs.iter().copied();
                let kept_outputs = match kept_outputs {
                    None => used_outputs.to_vec(),
                    Some(kept_outputs) => kept_outputs
                        .iter()
                        .map(|kept| match kept {
                            true => used.next().ok_or_else(|| {
                                ProgramError::MalformedProgram(format!(
                                    "{} keeps more rule outputs than it has outputs",
                                    {
                                        let name = rules.name();
                                        if name == COMPOSITE_OPERATION_NAME {
                                            format!("`{COMPOSITE_OPERATION_NAME}`")
                                        } else {
                                            format!("`{COMPOSITE_OPERATION_NAME}` `{name}`")
                                        }
                                    },
                                ))
                            }),
                            false => Ok(false),
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                };
                /// Returns the `values` of the outputs that `used_outputs` marks as used.
                fn keep_used<T: Clone>(values: &[T], used_outputs: &[bool]) -> Result<Vec<T>, ProgramError> {
                    check_count!("output", values, used_outputs.len(), ProgramError);
                    Ok(values
                        .iter()
                        .zip(used_outputs)
                        .filter_map(|(value, used)| used.then(|| value.clone()))
                        .collect())
                }
                let batching = match batching {
                    None => None,
                    Some(batching) => Some(CustomRuleBatching {
                        input_types: batching.input_types.clone(),
                        output_types: keep_used(&batching.output_types, used_outputs)?,
                        levels: batching
                            .levels
                            .iter()
                            .map(|level| {
                                let output_axes = keep_used(&level.output_axes, used_outputs)?;
                                Ok(CustomRuleBatchingLevel { output_axes, ..level.clone() })
                            })
                            .collect::<Result<Vec<_>, ProgramError>>()?,
                    }),
                };
                CompositeRules::Retained {
                    rules: rules.clone(),
                    batching,
                    discharged: *discharged,
                    kept_outputs: Some(kept_outputs),
                }
            }
        };
        let operation = Self { non_differentiated_count: self.non_differentiated_count, rules, marker: PhantomData };
        Ok(Some(OperationBoundaryPruning {
            operation,
            kept_inputs: vec![true; input_types.len()],
            kept_outputs: used_outputs.to_vec(),
        }))
    }

    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<V::Type as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        // Attached rules store no types, while retained rules store the types of their batching levels.
        let rules = match &self.rules {
            CompositeRules::Attached { jvp_rule, has_vjp_rule } => {
                CompositeRules::Attached { jvp_rule: *jvp_rule, has_vjp_rule: *has_vjp_rule }
            }
            CompositeRules::Retained { rules, batching, discharged, kept_outputs } => CompositeRules::Retained {
                rules: rules.clone(),
                batching: batching.as_ref().map(|batching| batching.rename_identities(renaming)).transpose()?,
                discharged: *discharged,
                kept_outputs: kept_outputs.clone(),
            },
        };
        Ok(Self { non_differentiated_count: self.non_differentiated_count, rules, marker: PhantomData })
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        // Attached rule regions render under their slot names, so their fields appear only for the non-differentiated
        // split and for a forward-mode rule derived from the primal region, both exactly where they exist. Retained
        // rules render the name of their definition and the state of the call that their specializations depend on.
        let operation = OperationFormatter::new(formatter, indentation, COMPOSITE_OPERATION_NAME)?;
        match &self.rules {
            CompositeRules::Attached { jvp_rule, .. } => {
                if self.non_differentiated_count == 0 && *jvp_rule != CompositeJvpRule::Primal {
                    return Ok(());
                }
                operation.bracketed(|operation| {
                    if self.non_differentiated_count > 0 {
                        operation.field("non_differentiated_count", self.non_differentiated_count)?;
                    }
                    if *jvp_rule == CompositeJvpRule::Primal {
                        operation.field("jvp_from_primal", true)?;
                    }
                    Ok(())
                })
            }
            CompositeRules::Retained { rules, batching, discharged, kept_outputs } => {
                operation.bracketed(|operation| {
                    operation.field("name", format_args!("{:?}", rules.name()))?;
                    if self.non_differentiated_count > 0 {
                        operation.field("non_differentiated_count", self.non_differentiated_count)?;
                    }
                    if let Some(batching) = batching {
                        operation.field("batching", batching)?;
                    }
                    if *discharged {
                        operation.field("discharged", true)?;
                    }
                    if let Some(kept_outputs) = kept_outputs {
                        operation.field("kept_outputs", format_args!("{kept_outputs:?}"))?;
                    }
                    if rules.defers_forward() {
                        operation.field("deferred_forward", true)?;
                    }
                    Ok(())
                })
            }
        }
    }
}

impl<C, P, V, O, S> ReferenceDischargeableOperation<C, P> for CompositeOperation<V, O, S>
where
    C: Context<Operation: From<CompositeOperation<V, O, S>>>,
    P: ReferenceDischargePolicy<C>,
    V: Typed<Type: DifferentiableType + Eq + Hash> + Parameter,
    S: CustomRuleSource<V, O>,
    CompositeOperation<V, O, S>: Operation<Type = C::Type>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        // Local reference lifecycles discharge inside each region while all user-declared numeric boundaries stay
        // intact. External reference inputs still require explicit state threading that these derivative interfaces do
        // not supply. Attached rule regions discharge with the primal region now, while a call with retained rules
        // records that its lazily traced rule programs must be discharged as well once they are traced.
        let name = match &self.rules {
            CompositeRules::Attached { .. } => COMPOSITE_OPERATION_NAME,
            CompositeRules::Retained { rules, .. } => rules.name(),
        };
        validate_composite_discharge_inputs(
            &if name == COMPOSITE_OPERATION_NAME {
                format!("`{COMPOSITE_OPERATION_NAME}`")
            } else {
                format!("`{COMPOSITE_OPERATION_NAME}` `{name}`")
            },
            inputs,
        )?;
        match &self.rules {
            CompositeRules::Attached { .. } => discharge_local_reference_operation(self, context, driver, inputs),
            CompositeRules::Retained { rules, batching, kept_outputs, .. } => {
                let discharged = Self {
                    non_differentiated_count: self.non_differentiated_count,
                    rules: CompositeRules::Retained {
                        rules: rules.clone(),
                        batching: batching.clone(),
                        discharged: true,
                        kept_outputs: kept_outputs.clone(),
                    },
                    marker: PhantomData,
                };
                discharge_local_reference_operation(&discharged, context, driver, inputs)
            }
        }
    }
}

// Interpretation replays only the primal region, so it applies in every context, including contexts of families that
// embed this call's family (e.g., kernel bodies).
impl<
    C: Domain<Type = V::Type>,
    V: Typed<Type: DifferentiableType + Eq + Hash> + Parameter,
    O,
    S: CustomRuleSource<V, O>,
> InterpretableOperation<C> for CompositeOperation<V, O, S>
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // An ordinary call computes only `f(p, x) = y`. Replaying a rule here would also compute derivative work (e.g.,
        // `ẏ` or the residuals `r`) and would charge every non-differentiated execution for it, so interpretation
        // delegates solely to the primal region at slot 0.
        driver.interpret_region(context, 0, inputs.to_vec())
    }
}

// The default partial-evaluation rule is the desired one here: it interprets the primal region when every input is
// known and otherwise residualizes the complete call, so that its rules remain available to later differentiation.
impl<C: Context<Type: DifferentiableType + Eq + Hash>, S: CustomRuleSource<C::Constant, C::Operation>>
    PartiallyEvaluatableOperation<C> for CompositeOperation<C::Constant, C::Operation, S>
where
    C::Operation: From<CompositeOperation<C::Constant, C::Operation, S>>,
{
}

impl<C, P, S> BatchableOperation<C, P> for CompositeOperation<C::Constant, C::Operation, S>
where
    C: Context<Type: DifferentiableType + Eq + Hash, Operation: From<CompositeOperation<C::Constant, C::Operation, S>>>,
    P: CotangentBatchingPolicy<C>,
    S: CustomRuleSpecializer<C::Constant, C::Operation>,
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<Vec<P::Batch>, BatchingError> {
        // A call whose rules batch it sequentially, without a custom batching rule, becomes a scan over its batch
        // items.
        if let CompositeRules::Retained { rules, .. } = &self.rules
            && !rules.has_batching_rule()
            && rules.has_sequential_batching()
        {
            return batch_composite_sequentially(self, rules, context, driver, inputs);
        }

        // Structurally batch the attached regions and rule programs over the packed inputs and their batch axes.
        let input_axes = inputs.iter().map(P::batch_axis).collect::<Vec<_>>();
        let primal_region = driver.region(0)?;
        let boundary_inputs = P::boundary_inputs(context.axis_extent());
        let (operation, regions, output_axes) = match &self.rules {
            CompositeRules::Attached { .. } => {
                // Batch every attached region contract while retaining the call and its rules:
                //
                //   - Primal:   (p, x)    → y
                //   - JVP:      (p, x, ẋ) → (y_jvp, ẏ)
                //   - Forward:  (p, x)    → (y_fwd, r)
                //   - Backward: (p, r, ȳ) → x̄.
                //
                // Each `ẋ` follows the batch axis of its corresponding `x`, while `p` has no tangent counterpart. The
                // ordinary primal, the JVP primal and tangent halves, and the primal prefix of the forward rule may
                // independently choose replicated or mapped representations for the same logical output. Reconcile
                // them to one wrapper axis per output, preferring the first mapped axis so that reconciliation never
                // discards batch variation, and align every batched region to it. Residuals are internal edges between
                // the forward and backward rules, so they keep the axes that the forward rule naturally produces. The
                // backward rule is batched on those exact axes and each `x̄` is aligned with its corresponding
                // differentiated input `x`: when a replicated `x` receives a mapped cotangent, the batching policy sums
                // that mapped axis, which is the transpose of broadcasting `x` across the batch. A forward-mode rule
                // derived from the primal region has no region of its own, so batching the primal region also batches
                // it.
                let (non_differentiated_axes, differentiated_axes) = self.split_inputs(input_axes.as_slice())?;
                let naturally_batched_primal = driver.batch_program(
                    context,
                    primal_region,
                    input_axes.as_slice(),
                    ProgramBatchingOutputAxesPolicy::Natural,
                )?;
                let primal_output_axes = naturally_batched_primal.output_axes().to_vec();
                let output_count = primal_output_axes.len();
                let mut candidate_output_axes = primal_output_axes.iter().map(|axis| vec![*axis]).collect::<Vec<_>>();

                // The JVP region consumes `(primals..., differentiated_tangents...)`, and a tangent has the same
                // packed batch axis as its corresponding primal input.
                let jvp_input_axes = input_axes.iter().chain(differentiated_axes).copied().collect::<Vec<_>>();
                let naturally_batched_jvp = match self.jvp_region_index() {
                    Some(jvp_region_index) => {
                        let batched = driver.batch_program(
                            context,
                            driver.region(jvp_region_index)?,
                            jvp_input_axes.as_slice(),
                            ProgramBatchingOutputAxesPolicy::Natural,
                        )?;
                        let jvp_output_axes = batched.output_axes();
                        check_count!("output", jvp_output_axes, 2 * output_count, ProgramError);
                        for (index, candidates) in candidate_output_axes.iter_mut().enumerate() {
                            candidates.push(jvp_output_axes[index]);
                            candidates.push(jvp_output_axes[output_count + index]);
                        }
                        Some(batched)
                    }
                    None => None,
                };

                let naturally_batched_forward = match self.vjp_region_indices() {
                    Some((forward_region_index, _)) => {
                        let batched = driver.batch_program(
                            context,
                            driver.region(forward_region_index)?,
                            input_axes.as_slice(),
                            ProgramBatchingOutputAxesPolicy::Natural,
                        )?;
                        if batched.output_axes().len() < output_count {
                            return Err(ProgramError::MalformedProgram(format!(
                                "batched `{}` forward rule produced {} outputs which is fewer than its {} primal \
                                 outputs",
                                COMPOSITE_OPERATION_NAME,
                                batched.output_axes().len(),
                                output_count,
                            ))
                            .into());
                        }
                        for (candidates, axis) in candidate_output_axes.iter_mut().zip(batched.output_axes()) {
                            candidates.push(*axis);
                        }
                        Some(batched)
                    }
                    None => None,
                };

                let output_axes = candidate_output_axes
                    .into_iter()
                    .map(|candidates| candidates.into_iter().find(|axis| !axis.is_replicated()).unwrap_or_default())
                    .collect::<Vec<_>>();
                let mut regions = vec![context.align_and_adapt_batched_program_outputs(
                    driver,
                    primal_region,
                    input_axes.as_slice(),
                    naturally_batched_primal,
                    output_axes.as_slice(),
                )?];
                if let (Some(jvp_region_index), Some(naturally_batched_jvp)) =
                    (self.jvp_region_index(), naturally_batched_jvp)
                {
                    let jvp_required_output_axes = output_axes.iter().chain(&output_axes).copied().collect::<Vec<_>>();
                    regions.push(context.align_and_adapt_batched_program_outputs(
                        driver,
                        driver.region(jvp_region_index)?,
                        jvp_input_axes.as_slice(),
                        naturally_batched_jvp,
                        jvp_required_output_axes.as_slice(),
                    )?);
                }
                if let (Some((forward_region_index, backward_region_index)), Some(naturally_batched_forward)) =
                    (self.vjp_region_indices(), naturally_batched_forward)
                {
                    let residual_axes = naturally_batched_forward.output_axes()[output_count..].to_vec();
                    let forward_required_output_axes =
                        output_axes.iter().chain(&residual_axes).copied().collect::<Vec<_>>();
                    regions.push(context.align_and_adapt_batched_program_outputs(
                        driver,
                        driver.region(forward_region_index)?,
                        input_axes.as_slice(),
                        naturally_batched_forward,
                        forward_required_output_axes.as_slice(),
                    )?);

                    // The backward rule maps `(non_differentiated..., residuals..., output_cotangents...)` to the
                    // differentiated inputs' cotangents. Align mapped results to their primal input positions while
                    // they are live; adaptation then sums the only non-structural mismatch, namely a mapped cotangent
                    // corresponding to a replicated primal input.
                    let backward_input_axes = non_differentiated_axes
                        .iter()
                        .chain(&residual_axes)
                        .chain(&output_axes)
                        .copied()
                        .collect::<Vec<_>>();
                    let batched_backward = driver.batch_program(
                        context,
                        driver.region(backward_region_index)?,
                        backward_input_axes.as_slice(),
                        ProgramBatchingOutputAxesPolicy::AlignEachTo(differentiated_axes.to_vec()),
                    )?;
                    let (backward, backward_output_axes) = P::adapt_batched_program(
                        batched_backward,
                        Some(differentiated_axes),
                        P::sum_mapped_cotangents,
                    )?
                    .into_parts();
                    if backward_output_axes.as_slice() != differentiated_axes {
                        return Err(BatchingError::MisalignedBatchAxes {
                            message: format!(
                                "batched `{COMPOSITE_OPERATION_NAME}` backward rule output axes \
                                 {backward_output_axes:?} do not match its differentiated input axes \
                                 {differentiated_axes:?}",
                            ),
                        });
                    }
                    regions.push(backward);
                }
                let operation = Self {
                    non_differentiated_count: self.non_differentiated_count + boundary_inputs.len(),
                    ..self.clone()
                };
                (operation, regions, output_axes)
            }
            CompositeRules::Retained { rules, batching, discharged, kept_outputs } => {
                // Batch the primal now and record the level, so that the derivative rules are batched only when they
                // are first traced. A custom batching rule takes precedence over structurally batching the primal
                // region, and it declares the batch axes of the outputs. Otherwise, the declared layout is required
                // exactly, so an incompatible primal is rejected here. Boundary inputs reach the primal and every
                // batched rule program as leading non-differentiated inputs. A call that keeps only some outputs of its
                // rules keeps only those of the batching rule and of the declared layout.
                let level = driver.batching_level(context)?;
                let output_count = primal_region.output_types().len();
                let rule_output_count = kept_outputs.as_ref().map_or(output_count, Vec::len);
                let is_kept = |index: usize| kept_outputs.as_ref().is_none_or(|kept_outputs| kept_outputs[index]);
                let (primal, output_axes) = if rules.has_batching_rule() {
                    let key = CustomRuleBatchingRuleKey {
                        level: level.clone(),
                        boundary_input_types: boundary_inputs.iter().map(|value| value.r#type().into_owned()).collect(),
                        input_types: inputs.iter().map(|input| P::value(input).r#type().into_owned()).collect(),
                        unbatched_input_types: primal_region.input_types(),
                        input_axes: input_axes.clone(),
                        discharged: *discharged,
                    };
                    let specialization = rules.batching_rule_specialization(key).map_err(ProgramError::from)?;
                    check_count!("output", specialization.output_axes, rule_output_count, ProgramError);
                    match kept_outputs {
                        None => (specialization.program.clone(), specialization.output_axes.clone()),
                        Some(_) => {
                            let kept_indices =
                                (0..rule_output_count).filter(|&index| is_kept(index)).collect::<Vec<_>>();
                            let output_axes =
                                kept_indices.iter().map(|&index| specialization.output_axes[index]).collect();
                            (specialization.program.with_outputs(&kept_indices)?, output_axes)
                        }
                    }
                } else {
                    let output_axes = rules
                        .batched_call_output_axes(rule_output_count)?
                        .into_iter()
                        .enumerate()
                        .filter_map(|(index, axis)| is_kept(index).then_some(axis))
                        .collect::<Vec<_>>();
                    let batched_primal = driver.batch_program(
                        context,
                        primal_region,
                        input_axes.as_slice(),
                        ProgramBatchingOutputAxesPolicy::AlignEachTo(output_axes.clone()),
                    )?;
                    let primal = context.align_and_adapt_batched_program_outputs(
                        driver,
                        primal_region,
                        input_axes.as_slice(),
                        batched_primal,
                        output_axes.as_slice(),
                    )?;
                    (primal, output_axes)
                };
                let mut batching = batching.clone().unwrap_or_else(|| CustomRuleBatching {
                    input_types: primal_region.input_types(),
                    output_types: primal_region.output_types(),
                    levels: Vec::new(),
                });
                batching.levels.push(CustomRuleBatchingLevel {
                    level,
                    boundary_input_count: boundary_inputs.len(),
                    input_axes,
                    output_axes: output_axes.clone(),
                });
                let operation = Self {
                    non_differentiated_count: self.non_differentiated_count + boundary_inputs.len(),
                    rules: CompositeRules::Retained {
                        rules: rules.clone(),
                        batching: Some(batching),
                        discharged: *discharged,
                        kept_outputs: kept_outputs.clone(),
                    },
                    marker: PhantomData,
                };
                (operation, vec![primal], output_axes)
            }
        };
        let mut packed_inputs = boundary_inputs;
        packed_inputs.extend(inputs.iter().map(P::value).cloned());
        let outputs = context.parent().bind(operation, regions, packed_inputs.as_slice())?;
        check_count!("output", outputs, output_axes.len(), ProgramError);
        let outputs = outputs
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| P::batch(output, axis))
            .collect::<Result<Vec<_>, _>>()?;

        // A custom batching rule declares its output batch axes itself, so each output unbatched along its axis must
        // have the type of the corresponding primal output, up to a bijective renaming of the type identities across
        // the call's boundary (e.g., of an extent that the rule computes in its own trace).
        if let CompositeRules::Retained { rules, .. } = &self.rules
            && rules.has_batching_rule()
        {
            let boundary = |inputs: Vec<C::Type>, outputs: Vec<C::Type>| -> Vec<C::Type> {
                inputs.into_iter().chain(outputs).collect()
            };
            let expected_output_types = primal_region.output_types();
            let output_types = outputs.iter().map(|output| P::unbatched_type(output).into_owned()).collect::<Vec<_>>();
            let expected = boundary(primal_region.input_types(), expected_output_types.clone());
            let actual = boundary(
                inputs.iter().map(|input| P::unbatched_type(input).into_owned()).collect(),
                output_types.clone(),
            );
            if derive_bijective_identity_renaming(&actual, &expected).is_none() {
                let render = |types: &[C::Type]| types.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ");
                let (expected, actual) = (render(&expected_output_types), render(&output_types));
                let reason = if expected == actual { " with different type identities" } else { "" };
                return Err(TypeError::invalid(format!(
                    "{} batching rule output type signature mismatch: expected per-item output types [{expected}] \
                     but got [{actual}]{reason}",
                    {
                        let name = rules.name();
                        if name == COMPOSITE_OPERATION_NAME {
                            format!("`{COMPOSITE_OPERATION_NAME}`")
                        } else {
                            format!("`{COMPOSITE_OPERATION_NAME}` `{name}`")
                        }
                    },
                ))
                .into());
            }
        }
        Ok(outputs)
    }
}

impl<C, S> DifferentiableOperation<C> for CompositeOperation<C::Constant, C::Operation, S>
where
    TracingContext<C::Constant, C::Operation>:
        StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    C: Context<
            Type: DifferentiableType + Eq + Hash,
            Operation: ResidualZeroProvider<C::Type, Operation = C::Operation>
                           + From<CompositeOperation<C::Constant, C::Operation, S>>
                           + From<CompositeTransposeOperation<C::Constant, C::Operation, S>>,
        > + Zero<C::Value>,
    S: CustomRuleSpecializer<C::Constant, C::Operation>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        validate_non_differentiated_count(COMPOSITE_OPERATION_NAME, self.non_differentiated_count, inputs.len())?;
        match (self.jvp_rule(), &self.rules) {
            // Apply the explicit pushforward directly instead of differentiating the primal region.
            (CompositeJvpRule::Explicit, CompositeRules::Attached { .. }) => replay_custom_jvp_rule(
                COMPOSITE_OPERATION_NAME,
                self.non_differentiated_count,
                context,
                driver,
                driver.region(1)?,
                inputs,
                None,
            ),

            // A retained rule is specialized for the tangent activity of the differentiated inputs, so structural zero
            // tangents stay symbolic inside the specialization.
            (CompositeJvpRule::Explicit, CompositeRules::Retained { rules, batching, discharged, kept_outputs }) => {
                let input_types = inputs.iter().map(|input| input.primal().r#type().into_owned()).collect::<Vec<_>>();
                let tangent_activity = inputs[self.non_differentiated_count..]
                    .iter()
                    .map(|input| !input.tangent().is_zero())
                    .collect::<Vec<_>>();
                let key = CustomRuleSpecializationKey {
                    tangent_activity: tangent_activity.clone(),
                    ..self.specialization_key(
                        rules,
                        batching,
                        *discharged,
                        kept_outputs,
                        input_types,
                        driver.region(0)?.output_types(),
                    )?
                };
                let specialization = rules.jvp_specialization(key)?;
                replay_custom_jvp_rule(
                    COMPOSITE_OPERATION_NAME,
                    self.non_differentiated_count,
                    context,
                    driver,
                    specialization.program.entry_region_ref(),
                    inputs,
                    Some(tangent_activity.as_slice()),
                )
            }
            // An unbatched call whose rules have a custom batching rule or batch it sequentially stages a derived call,
            // which keeps that rule, or the sequential batching, on the path of derivatives that are batched after they
            // are taken (refer to `CustomRuleSpecializer::derived`). A batched call already computes its primal with
            // its batching rule, whose derivative is inlined.
            (CompositeJvpRule::Primal, CompositeRules::Retained { rules, batching: None, .. })
                if rules.has_batching_rule() || rules.has_sequential_batching() =>
            {
                differentiate_primal_region(
                    COMPOSITE_OPERATION_NAME,
                    self.non_differentiated_count,
                    context,
                    driver,
                    inputs,
                    Some(self),
                )
            }
            (CompositeJvpRule::Primal, _) => differentiate_primal_region(
                COMPOSITE_OPERATION_NAME,
                self.non_differentiated_count,
                context,
                driver,
                inputs,
                None,
            ),
            (CompositeJvpRule::Absent, _) => {
                Err(missing_custom_jvp_rule_error(&self.description(), self.has_vjp_rule()))
            }
        }
    }

    fn jvp_for_transpose<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // Without reverse-mode rules, reverse mode transposes the linearized forward-mode rule. A forward-mode rule
        // derived from the primal is inlined, because transposing the derived calls that forward mode stages transposes
        // their primal regions inline, which would drop the batching rule that they exist to keep.
        if !self.has_vjp_rule() {
            if self.jvp_rule() == CompositeJvpRule::Primal {
                return differentiate_primal_region(
                    COMPOSITE_OPERATION_NAME,
                    self.non_differentiated_count,
                    context,
                    driver,
                    inputs,
                    None,
                );
            }
            return self.jvp(context, driver, inputs);
        }

        // Reverse-mode rules specify the pullback `ȳ ↦ x̄`, not the pushforward `ẋ ↦ ẏ`. Replay:
        //
        //   forward(p, x) = (y, r)
        //
        // to recover the primal outputs and residuals, then stage an opaque carrier representing the unknown map:
        //
        //   L_(p,r): ẋ ↦ ẏ.
        //
        // The carrier knows only how to transpose that map: its transpose applies `backward(p, r, ȳ) = x̄`. Passing
        // `p` and `r` as the carrier's leading known inputs keeps the path capture-free and exposes every dependency
        // as an ordinary Single Static Assignment (SSA) edge. The forward and backward rules bypass ordinary
        // differentiation dispatch: the forward rule is replayed directly and the backward rule is retained for later
        // transposition, so the replayed inputs are validated as defense in depth (refer to the documentation of
        // `validate_composite_replay`).
        let (non_differentiated_inputs, differentiated_inputs) = self.split_inputs(inputs)?;
        let primal_region = driver.region(0)?;
        let primal_output_types = primal_region.output_types();
        let output_count = primal_output_types.len();
        validate_composite_replay(
            COMPOSITE_OPERATION_NAME,
            self.non_differentiated_count,
            context.primal(),
            inputs,
            primal_output_types.as_slice(),
        )?;
        check_count!("input", inputs, primal_region.input_types().len(), ProgramError);
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let mut leading_values =
            non_differentiated_inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();

        // Structural-zero seeds are materialized from the known leading inputs, which need not carry the runtime
        // extents of the output tangent types (e.g., the extents of a dynamically shaped output that the residuals do
        // not mention), so those extents are read from the primal outputs and appended to the leading inputs as hidden
        // seed extents, which the backward rule never receives. Equal extent types name the same runtime quantity, so
        // each one is passed once.
        let capture_seed_extents = |outputs: &[C::Value],
                                    extent_types: &[C::Type],
                                    leading_values: &mut Vec<C::Value>|
         -> Result<Vec<C::Type>, ProgramError> {
            let mut seed_extent_types = Vec::new();
            for (output, r#type) in outputs.iter().zip(extent_types) {
                for extent_type in C::Operation::zero_residual_types(r#type) {
                    if seed_extent_types.contains(&extent_type) {
                        continue;
                    }
                    let value = C::Operation::capture_zero_residual_value(context.primal(), output, &extent_type)?
                        .ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "seed extent of type `{extent_type}` cannot be captured from output of type `{}`",
                                output.r#type().as_ref(),
                            ))
                        })?;
                    leading_values.push(value);
                    seed_extent_types.push(extent_type);
                }
            }
            Ok(seed_extent_types)
        };
        let (outputs, seed_extent_count, carrier_rules, carrier_regions, kept_outputs) = match &self.rules {
            CompositeRules::Attached { .. } => {
                // Replay the forward region on the dual primals, recovering the primal outputs followed by the
                // residuals. The carrier transposes by replaying the backward region, whose own inputs are exactly the
                // leading known inputs followed by the output cotangents.
                let (forward_region_index, backward_region_index) = self.vjp_region_indices().unwrap();
                let mut outputs =
                    driver.region(forward_region_index)?.interpret_in_context(context.primal(), primal_inputs)?;
                if outputs.len() < output_count {
                    return Err(ProgramError::MalformedProgram(format!(
                        "`{}` forward rule produced {} outputs which is fewer than its {} primal output(s)",
                        COMPOSITE_OPERATION_NAME,
                        outputs.len(),
                        output_count,
                    ))
                    .into());
                }
                leading_values.extend(outputs.split_off(output_count));
                let extent_types =
                    outputs.iter().map(|output| output.r#type().tangent()).collect::<Result<Vec<_>, _>>()?;
                let seed_extent_types = capture_seed_extents(&outputs, &extent_types, &mut leading_values)?;
                (
                    outputs,
                    seed_extent_types.len(),
                    CompositeTransposeRules::Attached,
                    vec![driver.region(backward_region_index)?.to_program()],
                    None,
                )
            }
            // A call that defers its forward rule replays its primal region instead (i.e., its batched primal region
            // when it is batched), and its carrier receives the call's inputs in place of the residuals, from which it
            // replays the forward rule when it is transposed. The seed extents are read from the primal outputs as
            // usual. A call that keeps only some outputs of its rules is differentiated without deferral (below),
            // because only the forward rule computes the outputs that it does not keep, whose seeds and runtime
            // extents the backward rule needs.
            CompositeRules::Retained { rules, batching, discharged, kept_outputs: None } if rules.defers_forward() => {
                let input_types = inputs.iter().map(|input| input.primal().r#type().into_owned()).collect::<Vec<_>>();
                let key =
                    self.specialization_key(rules, batching, *discharged, &None, input_types, primal_output_types)?;

                // The forward rule is traced here, on its first request, to reject an effectful forward rule before
                // anything is staged, and the carrier's transposition replays the cached specialization.
                let specialization = rules.forward_specialization(key.clone())?;
                validate_deferred_forward_rule(rules.name(), specialization.program.effects())?;
                let outputs = primal_region.interpret_in_context(context.primal(), primal_inputs)?;
                leading_values.extend(differentiated_inputs.iter().map(|input| input.primal().clone()));
                let extent_types = match batching {
                    None => outputs.iter().map(|output| output.r#type().tangent()).collect::<Result<Vec<_>, _>>()?,
                    Some(batching) => {
                        batching.output_types.iter().map(DifferentiableType::tangent).collect::<Result<Vec<_>, _>>()?
                    }
                };
                let seed_extent_types = capture_seed_extents(&outputs, &extent_types, &mut leading_values)?;
                (
                    outputs,
                    seed_extent_types.len(),
                    CompositeTransposeRules::Retained {
                        rules: rules.clone(),
                        batching: None,
                        invokes_rule_directly: false,
                        discharged: *discharged,
                        deferred_forward: Some(key),
                    },
                    Vec::new(),
                    None,
                )
            }
            CompositeRules::Retained { rules, batching, discharged, kept_outputs } => {
                // Replay the forward rule specialization on the dual primals, returning every output of the rule
                // followed by the residuals. A call that keeps only some of the rule's outputs still replays them all,
                // and its carrier receives every output, so that the backward rule receives a seed for each of them
                // (i.e., a structural zero for each output that the call does not keep).
                let input_types = inputs.iter().map(|input| input.primal().r#type().into_owned()).collect::<Vec<_>>();
                let key = self.specialization_key(
                    rules,
                    batching,
                    *discharged,
                    kept_outputs,
                    input_types,
                    primal_output_types,
                )?;
                let rule_output_count = key.rule_output_count()?;
                let specialization = rules.forward_specialization(key.clone())?;
                let mut outputs = specialization.program.interpret_in_context(context.primal(), primal_inputs)?;
                leading_values.extend(outputs.split_off(rule_output_count));

                // A batched call stages a batched carrier, whose types are those of the unbatched forward rule.
                let unbatched =
                    match batching {
                        None => None,
                        Some(_) => Some(rules.forward_specialization(CustomRuleSpecializationKey {
                            levels: Vec::new(),
                            ..key.clone()
                        })?),
                    };

                // The seed extents of a batched call are those of its unbatched output tangent types, read by identity
                // from its outputs. The outputs that a batched call does not keep have no recorded unbatched types, so
                // those of its unbatched forward rule are used instead.
                let rule_output_types = match (batching, &unbatched, kept_outputs) {
                    (Some(batching), _, None) => batching.output_types.clone(),
                    (Some(_), Some(unbatched), Some(_)) => {
                        unbatched.program.output_types()[..rule_output_count].to_vec()
                    }
                    _ => outputs.iter().map(|output| output.r#type().into_owned()).collect(),
                };
                let extent_types =
                    rule_output_types.iter().map(DifferentiableType::tangent).collect::<Result<Vec<_>, _>>()?;
                let seed_extent_types = capture_seed_extents(&outputs, &extent_types, &mut leading_values)?;
                let seed_extent_count = seed_extent_types.len();

                let carrier_batching = match batching {
                    None => None,
                    Some(_) => Some(retained_carrier_batching(
                        rules,
                        &key,
                        rule_output_count,
                        &seed_extent_types,
                        extent_types,
                    )?),
                };
                (
                    outputs,
                    seed_extent_count,
                    CompositeTransposeRules::Retained {
                        rules: rules.clone(),
                        batching: carrier_batching,
                        invokes_rule_directly: false,
                        discharged: *discharged,
                        deferred_forward: None,
                    },
                    Vec::new(),
                    kept_outputs.clone(),
                )
            }
        };

        // Stage the carrier over `[leading_values..., differentiated_tangents...]`, producing the output tangents. The
        // carrier takes every differentiated input tangent as a real input, so structural zeros are materialized
        // against their own primal, which names every runtime quantity that a reference-bearing tangent type omits.
        let leading_input_count = leading_values.len();
        let mut carrier_inputs = leading_values
            .into_iter()
            .map(|value| context.primal_to_tangent(value))
            .collect::<Result<Vec<_>, _>>()?;
        let mut input_tangent_types = Vec::with_capacity(differentiated_inputs.len());
        for input in differentiated_inputs {
            input_tangent_types.push(input.primal().r#type().tangent()?);
            let source = context.primal_to_tangent(input.primal().clone())?;
            carrier_inputs.push(C::Operation::materialize_zero_from_residual_sources(
                context.tangent(),
                input.tangent().clone(),
                std::iter::once(&source),
            )?);
        }
        let carrier = CompositeTransposeOperation {
            leading_input_count,
            seed_extent_count,
            input_tangent_types,
            output_tangent_types: outputs
                .iter()
                .map(|output| output.r#type().tangent())
                .collect::<Result<Vec<_>, _>>()?,
            rules: carrier_rules,
            marker: PhantomData,
        };
        let output_tangents = context.tangent().bind(carrier, carrier_regions, &carrier_inputs)?;
        check_count!("output", output_tangents, outputs.len(), ProgramError);

        // Only the outputs that the call keeps are returned, while the tangents of the others are left unused, so that
        // transposing the carrier seeds them with structural zeros.
        outputs
            .into_iter()
            .zip(output_tangents)
            .enumerate()
            .filter(|(index, _)| kept_outputs.as_ref().is_none_or(|kept_outputs| kept_outputs[*index]))
            .map(|(_, (primal, tangent))| DifferentiationDual::new(primal, tangent))
            .collect::<Result<Vec<_>, _>>()
    }
}

impl<V, O, S> TransposableOperation<V, O> for CompositeOperation<V, O, S>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: Value<Type: DifferentiableType + Eq + Hash>,
    O: Operation<Type = V::Type> + OperationProvider<V::Type, AddOperation<V::Type>, Operation = O>,
    S: CustomRuleSource<V, O>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<CustomRuleTracer<V, O>>],
        outputs: &[MaybeZero<CustomRuleTracer<V, O>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        // Reverse mode linearizes a call before transposing it: a forward-mode rule replaces `f(p, x)` with the
        // ordinary primitive program computing the linear map `ẋ ↦ (∂f/∂x)(p, x) · ẋ`, and reverse-mode rules replace
        // it with a `CompositeTransposeOperation` carrier, whose transpose evaluates `backward(p, r, ȳ) = x̄`. A call
        // that reaches transposition is therefore one that is linear in its unknown inputs, such as the pushforward
        // call that a partitioned differentiation context stages for a call whose forward-mode rule is derived from its
        // primal (refer to `differentiate_primal_region`). Like JAX's transposition of a `custom_jvp_call` through its
        // call jaxpr, the primal region is transposed with respect to the unknown inputs and replayed inline, so a call
        // whose primal is not linear in them is rejected by the transpose rules of the primal's own operations.
        let primal = driver.region(0)?;
        check_count!("input", inputs, primal.input_types().len(), ProgramError);
        check_count!("output", outputs, primal.output_types().len(), ProgramError);
        check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);

        // Without live seeds or requested cotangents, the call is a zero linear map unless its primal region carries
        // deferred work or has observable effects, which must run even with structurally zero cotangents.
        if (outputs.iter().all(MaybeZero::is_zero) || !accumulators.iter().any(CotangentAccumulator::is_needed))
            && !primal.must_transpose()
        {
            return Ok(());
        }

        // The unknown inputs are differentiated inputs, which are never references (refer to
        // `validate_composite_reference_boundary`), so the cotangent of each one is either ignored or returned to its
        // accumulator, which adds it to its destination. Structural-zero seeds stay symbolic: the transposed primal
        // receives only the live seeds, followed by the known inputs.
        let (linear_input_indices, destination_kinds): (Vec<_>, Vec<_>) = inputs
            .iter()
            .zip(accumulators)
            .enumerate()
            .filter(|(_, (input, _))| input.is_unknown())
            .map(|(index, (_, accumulator))| match accumulator.is_needed() {
                true => (index, CotangentDestinationKind::Return),
                false => (index, CotangentDestinationKind::Ignore),
            })
            .unzip();
        let zero_seeds = outputs.iter().map(MaybeZero::is_zero).collect::<Vec<_>>();
        let transposed = driver.transpose_program_with_zero_seeds(
            primal,
            &linear_input_indices,
            &[],
            &zero_seeds,
            &destination_kinds,
        )?;
        let mut transposed_inputs = outputs.iter().filter_map(MaybeZero::as_value).cloned().collect::<Vec<_>>();
        transposed_inputs.extend(inputs.iter().filter_map(PartialValue::as_known).cloned());
        let cotangents = transposed.program().interpret_in_context(&**context, transposed_inputs)?;
        let cotangent_types = linear_input_indices
            .iter()
            .map(|&index| inputs[index].r#type().cotangent())
            .collect::<Result<Vec<_>, _>>()?;
        let cotangents = transposed.assemble_cotangents(&cotangent_types, cotangents)?;
        for (&index, cotangent) in linear_input_indices.iter().zip(cotangents) {
            if let Some(cotangent) = cotangent {
                accumulators[index].accumulate(context, cotangent)?;
            }
        }
        Ok(())
    }
}

/// Batches the composite call `call`, whose retained rules `rules` batch it sequentially (refer to
/// [`CustomRuleDefinition::with_sequential_batching`](crate::CustomRuleDefinition::with_sequential_batching)), by
/// looping over the batch items with a scan whose body applies the unbatched call to one item. The mapped inputs are
/// aligned to axis 0 and become the scan's stacked inputs, the replicated inputs become invariant carries, and every
/// output is mapped at axis 0. The body applies the call with its own primal region, so the call's rules keep governing
/// the derivatives of each item, and batching the scan again (e.g., at an enclosing batching level) batches the calls
/// in its body sequentially in turn. A call whose inputs are all replicated is the same for every item, so it is
/// applied once instead.
///
/// # Parameters
///
///   - `call`: Unbatched call to batch.
///   - `rules`: Retained rules of `call`, which construct the scan.
///   - `context`: Batching context in which the call is batched.
///   - `driver`: Driver of the call's primal region.
///   - `inputs`: Batched inputs of the call.
///
/// # Errors
///
/// Returns a [`BatchingError`] when the batch extent is not static, and propagates the errors of the scan construction,
/// of the axis alignment, and of staging the scan.
fn batch_composite_sequentially<C, P, D, S>(
    call: &CompositeOperation<C::Constant, C::Operation, S>,
    rules: &S,
    context: &BatchingContext<C, P>,
    driver: &D,
    inputs: &[P::Batch],
) -> Result<Vec<P::Batch>, BatchingError>
where
    C: Context<Type: DifferentiableType + Eq + Hash, Operation: From<CompositeOperation<C::Constant, C::Operation, S>>>,
    P: BatchingPolicy<C>,
    D: BatchingDriver<C, P>,
    S: CustomRuleSpecializer<C::Constant, C::Operation>,
{
    let primal_region = driver.region(0)?;
    let level = driver.batching_level(context)?;
    let BatchingLevelExtent::Static(length) = level.extent() else {
        return Err(BatchingError::UnsupportedOperation {
            message: {
                let name = rules.name();
                let description = if name == COMPOSITE_OPERATION_NAME {
                    format!("`{COMPOSITE_OPERATION_NAME}`")
                } else {
                    format!("`{COMPOSITE_OPERATION_NAME}` `{name}`")
                };
                format!("{description} batches its calls sequentially, which requires a static batch extent")
            },
        });
    };
    if inputs.iter().all(|input| P::batch_axis(input).is_replicated()) {
        let values = inputs.iter().map(P::value).cloned().collect::<Vec<_>>();
        let outputs = context.parent().bind(call.clone(), vec![primal_region.to_program()], values.as_slice())?;
        return Ok(outputs.into_iter().map(P::replicated).collect());
    }

    // The body receives the slice index first, then the carries, then the slices, and it applies the call to them in
    // the call's own input order. It returns the carries unchanged, followed by the call's outputs, which the scan
    // stacks along the batch axis. Body inputs omit the batch axis, while the packed scan inputs retain it.
    let (carry_indices, stacked_indices): (Vec<_>, Vec<_>) =
        (0..inputs.len()).partition(|&index| P::batch_axis(&inputs[index]).is_replicated());
    let output_count = primal_region.output_types().len();
    let (scan, index_type) = rules
        .sequential_batching_scan(*length, carry_indices.len(), context.axis_sharding(), output_count)
        .map_err(ProgramError::from)?;
    let mut builder = ProgramBuilder::<C::Constant, C::Operation>::new();
    let primal = builder.import_program(primal_region.to_program());
    builder.add_input(index_type);
    let mut call_inputs = vec![None; inputs.len()];
    let mut packed_inputs = Vec::with_capacity(inputs.len());
    for &index in carry_indices.iter().chain(&stacked_indices) {
        let input = match P::batch_axis(&inputs[index]).is_replicated() {
            true => inputs[index].clone(),
            false => driver.align_batch_axis(context, inputs[index].clone(), Axis::from(0))?,
        };
        call_inputs[index] = Some(builder.add_input(P::unbatched_type(&input).into_owned()));
        packed_inputs.push(P::value(&input).clone());
    }

    // The partition visits every input exactly once, so every input slot is populated.
    let carries = carry_indices.iter().map(|&index| call_inputs[index].unwrap()).collect::<Vec<_>>();
    let call_inputs = call_inputs.into_iter().map(Option::unwrap).collect::<Vec<_>>();
    let outputs = builder.add_instruction(call.clone(), vec![primal], call_inputs, None)?.to_vec();
    let body = builder.build::<Vec<C::Constant>, Vec<C::Constant>>(
        carries.iter().copied().chain(outputs).collect(),
        vec![Placeholder; 1 + inputs.len()],
        vec![Placeholder; carries.len() + output_count],
    )?;
    let mut outputs = context.parent().bind(scan, vec![body], packed_inputs.as_slice())?;
    check_count!("output", outputs, carries.len() + output_count, ProgramError);
    outputs.drain(..carries.len());
    outputs.into_iter().map(|output| P::batch(output, BatchAxis::new(0))).collect()
}

/// Differentiates the primal region at slot 0 of a composite call in forward mode, which is the forward-mode rule of a
/// call whose JVP is derived from its primal (i.e., [`CompositeJvpRule::Primal`]). The derivative is taken with respect
/// to the differentiated inputs with active tangents, while the leading `non_differentiated_count` inputs, which must
/// have zero tangents, parameterize the derivative.
///
/// The derivative is inlined unless `call` is provided, in which case it is staged as a derived call of the rules
/// that [`CustomRuleSpecializer::derived`] derives from the call's rules (refer to the documentation of the derivation
/// for why): a fused context stages one call that computes the outputs and their tangents, while a partitioned context
/// computes the outputs with the call itself and stages a pushforward call on its tangent side.
///
/// # Parameters
///
///   - `name`: Operation name used in diagnostics.
///   - `non_differentiated_count`: Number of leading inputs that parameterize the call without being differentiated.
///   - `context`: Differentiation context in which the call is differentiated.
///   - `driver`: Driver of the call's regions.
///   - `inputs`: Dual inputs of the call.
///   - `call`: Unbatched call with retained rules whose derivative is staged as a derived call, or [`None`] to
///     inline the derivative.
fn differentiate_primal_region<C, P, D, S>(
    name: &str,
    non_differentiated_count: usize,
    context: &DifferentiationContext<C, P>,
    driver: &D,
    inputs: &[DifferentiationDual<C::Value>],
    call: Option<&CompositeOperation<C::Constant, C::Operation, S>>,
) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>
where
    TracingContext<C::Constant, C::Operation>:
        StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    C: Context<
            Type: DifferentiableType + Eq + Hash,
            Operation: ResidualZeroProvider<C::Type, Operation = C::Operation>
                           + From<CompositeOperation<C::Constant, C::Operation, S>>,
        > + Zero<C::Value>,
    P: DifferentiationPolicy<C>,
    D: DifferentiationDriver<C>,
    S: CustomRuleSpecializer<C::Constant, C::Operation>,
{
    let primal_region = driver.region(0)?;
    let output_types = primal_region.output_types();
    let output_count = output_types.len();
    validate_composite_replay(name, non_differentiated_count, context.primal(), inputs, output_types.as_slice())?;
    check_count!("input", inputs, primal_region.input_types().len(), ProgramError);

    // Differentiate with respect to the differentiated inputs whose tangents are live. Inactive inputs (e.g.,
    // structural zeros and zero-space inputs) receive no tangent input. The derivative has a tangent for every output
    // in `output_activity`, which also keys the derived rules, but only the outputs whose derivative depends on a live
    // tangent input carry a live tangent. The others are structural zeros, so linearization never sees a tangent that
    // depends on no tangent input. Their liveness comes from the derivative program that produces the tangents, so
    // that a custom JVP rule may give an output a tangent that the derivative of the primal would not.
    let input_indices = inputs
        .iter()
        .enumerate()
        .skip(non_differentiated_count)
        .filter_map(|(index, input)| (input.is_tangent_active() && !input.tangent().is_zero()).then_some(index))
        .collect::<Vec<_>>();
    let output_activity = primal_region.tangent_output_mask(&input_indices)?;
    let jvp_program = driver.jvp_program(primal_region, &input_indices)?;
    let tangent_inputs = (inputs.len()..jvp_program.input_count()).collect::<Vec<_>>();
    let mut output_depends = jvp_program.output_dependence(&tangent_inputs)?.into_iter().skip(output_count);
    let output_liveness = output_activity
        .iter()
        .map(|&has_slot| has_slot && output_depends.next().unwrap())
        .collect::<Vec<_>>();
    let primals = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
    let mut tangents = Vec::with_capacity(input_indices.len());
    for &index in &input_indices {
        let source = context.primal_to_tangent(inputs[index].primal().clone())?;
        tangents.push(C::Operation::materialize_zero_from_residual_sources(
            context.tangent(),
            inputs[index].tangent().clone(),
            std::iter::once(&source),
        )?);
    }

    // A derived call has the rules derived from the call's rules for this activity pattern, and the derivative of the
    // call's primal region (or its tangent half) as its own primal region. Without active tangents, there is no
    // derivative to batch, so it is inlined.
    let call = call.filter(|_| !input_indices.is_empty());
    let derived_call = |kind: CustomRuleDerivationKind| match call {
        Some(CompositeOperation { rules: CompositeRules::Retained { rules, discharged, .. }, .. }) => {
            let rules = rules.derived(CustomRuleDerivationKey {
                kind,
                input_count: inputs.len(),
                active_input_indices: input_indices.clone(),
                output_tangent_mask: output_activity.clone(),
            })?;
            Ok::<_, DifferentiationError>(Some(CompositeOperation {
                non_differentiated_count,
                rules: CompositeRules::Retained { rules, batching: None, discharged: *discharged, kept_outputs: None },
                marker: PhantomData,
            }))
        }
        _ => Ok(None),
    };

    // Pairs the tangents of one derivative path with the outputs that `slots` flags, in output order, leaving the other
    // outputs without a tangent.
    let distribute = |tangents: Vec<C::Value>, slots: &[bool]| {
        check_count!("output", tangents, slots.iter().filter(|&&has_slot| has_slot).count(), ProgramError);
        let mut tangents = tangents.into_iter();
        Ok::<_, ProgramError>(slots.iter().map(|&has_slot| has_slot.then(|| tangents.next().unwrap())).collect())
    };

    // A fused differentiation context stages primal and tangent work in the same context, so the fused JVP program
    // replays there directly. Otherwise, the linearized primal replays in the primal context and its tangent map
    // replays in the tangent context over the residuals. Each path pairs its tangents with the outputs that it gives a
    // tangent slot. The derivative program's slots follow `output_activity`, and the linearization reports its own.
    let (outputs, output_tangents): (Vec<_>, Vec<Option<C::Value>>) =
        if std::ptr::eq(context.primal(), context.tangent()) {
            let program = jvp_program;
            let mut values = primals;
            values.extend(tangents);
            let mut outputs = match derived_call(CustomRuleDerivationKind::Jvp)? {
                Some(operation) => context.primal().bind(operation, vec![(*program).clone()], values.as_slice())?,
                None => program.interpret_in_context(context.primal(), values)?,
            };
            let output_tangents = outputs.split_off(output_count);
            (outputs, distribute(output_tangents, &output_activity)?)
        } else if let Some(call) = call.filter(|_| primal_region.effects() == EffectsSummary::PURE) {
            // The known side computes the outputs with the call itself, and the tangent side stages a pushforward call
            // that recomputes the primal internally, because the call is opaque to partial evaluation. Recomputing is only
            // valid for a pure primal, because the effects of any other primal (e.g., reference updates, or reference reads that
            // later effects would change) must run once, when the known side runs, so it is linearized inline instead.
            let pushforward =
                jvp_program.with_outputs(&(output_count..jvp_program.output_count()).collect::<Vec<_>>())?;
            let outputs = context.primal().bind(call.clone(), vec![primal_region.to_program()], primals.as_slice())?;
            let mut tangent_inputs =
                primals.into_iter().map(|primal| context.primal_to_tangent(primal)).collect::<Result<Vec<_>, _>>()?;
            tangent_inputs.extend(tangents);
            let operation = derived_call(CustomRuleDerivationKind::Pushforward)?.unwrap();
            let output_tangents = context.tangent().bind(operation, vec![pushforward], tangent_inputs.as_slice())?;
            (outputs, distribute(output_tangents, &output_activity)?)
        } else {
            // A structural zero output tangent depends on no input tangent, so no live output loses its slot. The
            // position check rejects a linearization that would silently turn a live tangent into a zero.
            let linearization = driver.linearize_program(primal_region, &input_indices, Some(&[]))?;
            linearization.tangent_output_positions(&output_liveness)?;
            let mut outputs = linearization.primal().interpret_in_context(context.primal(), primals)?;
            let residuals = outputs.split_off(output_count);
            let mut tangent_inputs = tangents;
            for residual in residuals {
                tangent_inputs.push(context.primal_to_tangent(residual)?);
            }
            let output_tangents = linearization.tangent().interpret_in_context(context.tangent(), tangent_inputs)?;
            (outputs, distribute(output_tangents, linearization.tangent_output_slots())?)
        };
    check_count!("output", outputs, output_count, ProgramError);

    // The tangents of outputs that are not live are unused zeros, which dead code elimination removes.
    outputs
        .into_iter()
        .zip(output_tangents)
        .zip(output_liveness)
        .map(|((primal, tangent), live)| match tangent.filter(|_| live) {
            Some(tangent) => DifferentiationDual::new(primal, MaybeZero::Value(tangent)),
            None => DifferentiationDual::new_with_zero_tangent(primal),
        })
        .collect::<Result<Vec<_>, _>>()
}

/// Returns the error that forward-mode differentiation of a composite call without a forward-mode rule reports, where
/// `call` describes the call (e.g., ``a `composite` call``). A call with reverse-mode rules supports only reverse-mode
/// differentiation, while a call without any rule is not differentiable. A missing rule never falls back to
/// differentiating the primal.
pub(super) fn missing_custom_jvp_rule_error(call: &str, has_vjp_rule: bool) -> DifferentiationError {
    ProgramError::UnsupportedOperation {
        message: if has_vjp_rule {
            format!(
                "cannot apply forward-mode differentiation to {call} that has only reverse-mode rules; it supports \
                 only reverse-mode differentiation (e.g., `vjp`, `value_and_gradient`, or `jacobian_reverse`)",
            )
        } else {
            format!("cannot differentiate {call} that has no derivative rule")
        },
    }
    .into()
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayBatch, ArrayBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation,
        ArrayReference, ArrayReferenceTransform, ArrayReferenceTransformIndex, ArrayType, DataType, Dimension,
        DimensionBounds, DimensionType, DimensionValue, DimensionVariable, Shape, ShardingDimension,
    };
    use crate::batching::{
        BatchAxis, BatchingContext, BatchingLevelExtent, ProgramBatchingOutputAxesPolicy, RecursiveBatchingDriver,
        batch,
    };
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::forward::JvpPartitionTransform;
    use crate::differentiation::{
        CotangentDestination, CotangentDestinationKind, CotangentSeed, DifferentiationRule, EverythingSavable,
        ForwardModeDifferentiate, ReverseModeDifferentiate, TransposedProgramCotangent, differentiate_at,
        rematerialize,
    };
    use crate::operations::arithmetic::{AddOperation, MulOperation};
    use crate::operations::assertions::{AssertOperation, AssertionError};
    use crate::operations::collectives::axis_index::AxisIndexOperation;
    use crate::operations::comparisons::{CompareOperation, ComparisonDirection};
    use crate::operations::composites::builders::composite;
    use crate::operations::composites::rules::{
        CustomRuleBackwardSpecializationKey, CustomRuleRegistration, LiftedCustomRules,
    };
    use crate::operations::composites::tests::{
        ArrayContext, ConvertedProgram, EagerArrayIrContext, IrDefinition, IrTracer, MemberDefinition,
        ReferenceRuleDifferentiationDriver, RuleCounters, TestContext, TestDefinition, TestRegistration,
        batched_program, call_jvp, composite_call_program, cube_definition, cube_program, custom_rule_program,
        ir_custom_rule_program, member_cube_definition,
    };
    use crate::operations::control_flow::condition::ConditionOperation;
    use crate::operations::control_flow::scan::ScanOperation;
    use crate::operations::debugging::Print;
    use crate::operations::differentiation::linear_call::tests::scalar_multiply_program;
    use crate::operations::dimensions::dimension_from_scalar::DimensionFromScalarOperation;
    use crate::operations::reductions::{Reduce, ReduceOperation, ReductionKind};
    use crate::operations::references::{
        ReferenceAddUpdate, ReferenceAddUpdateOperation, ReferenceFreezeOperation, ReferenceNew, ReferenceNewOperation,
        ReferenceRead, ReferenceReadOperation,
    };
    use crate::operations::trigonometric::{Cos, CosOperation, Sin, SinOperation};
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationOutput, PartialValue};
    use crate::programs::{
        AttachedRegions, EffectClass, EffectClasses, FlatProgram, InputRegionProvenance, MaybeZero,
        OutputRegionProvenance, ProgramBuilder, ReferenceType, RegionInterface, RegionRole, RegionSlot,
    };
    use crate::tests::hash_of;
    use crate::tracing::{DomainTracer, Trace};

    use super::*;

    /// Composite call in the array operation family.
    type ArrayComposite = CompositeOperation<Array, ArrayOperation<Array>>;

    /// Composite call in the array IR operation family.
    type ArrayIrComposite = CompositeOperation<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Builds a reference-free program representing the identity function over `r#type`.
    fn array_ir_identity_program(r#type: &ArrayIrType) -> FlatProgram<EagerArrayIrContext> {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(r#type.clone());
        builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![input],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap()
    }

    /// Pairs an identity primal program with a custom JVP program that allocates and reads local reference state.
    fn custom_jvp_regions_with_reference_state(r#type: &ArrayIrType) -> Vec<FlatProgram<EagerArrayIrContext>> {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(r#type.clone());
        let tangent = builder.add_input(r#type.clone());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let jvp_program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output, tangent],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        vec![array_ir_identity_program(r#type), jvp_program]
    }

    /// Builds a custom-derivative rule whose nested custom JVP closure contains local reference state.
    fn nested_composite_state_program(
        scalar_type: &ArrayIrType,
        include_tangent_output: bool,
    ) -> FlatProgram<EagerArrayIrContext> {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let regions = custom_jvp_regions_with_reference_state(scalar_type)
            .iter()
            .map(|region| builder.import_region(region.entry_region_ref()))
            .collect::<Vec<_>>();
        let input = builder.add_input(scalar_type.clone());
        let tangent = include_tangent_output.then(|| builder.add_input(scalar_type.clone()));
        let output = builder
            .add_instruction(
                ArrayIrComposite::from_rule_regions(CompositeJvpRule::Explicit, false),
                regions,
                vec![input],
                None,
            )
            .unwrap()[0];
        let mut outputs = vec![output];
        if let Some(tangent) = tangent {
            outputs.push(tangent);
        }
        let input_count = usize::from(include_tangent_output) + 1;
        let output_count = outputs.len();
        builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                outputs,
                vec![Placeholder; input_count],
                vec![Placeholder; output_count],
            )
            .unwrap()
    }

    /// Builds `f(x) = sin(x)` over one input of the provided type.
    fn sin_program(r#type: &ArrayType) -> FlatProgram<ArrayContext> {
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(r#type.clone());
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds the custom rule `jvp(x, ẋ) = (sin(x), 2 * cos(x) * ẋ)`, whose tangent deliberately differs from the
    /// mathematical derivative so that tests can prove that the custom rule is used.
    fn doubled_sin_jvp_program(r#type: &ArrayType) -> FlatProgram<ArrayContext> {
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(r#type.clone());
        let tangent = builder.add_input(r#type.clone());
        let y = builder.add_instruction(SinOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
        let cosine = builder.add_instruction(CosOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
        let two = builder.add_constant(Array::scalar(2.0).unwrap());
        let scaled = builder.add_instruction(MulOperation::new(), Vec::new(), vec![two, cosine], None).unwrap()[0];
        let output_tangent =
            builder.add_instruction(MulOperation::new(), Vec::new(), vec![scaled, tangent], None).unwrap()[0];
        builder.build(vec![y, output_tangent], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap()
    }

    /// Builds the square function and its fused rule, with a tangent accumulator allocated before the required primal
    /// output. The coefficient is pure known work even though the accumulator must be fresh for every pushforward
    /// call. The rule reads the accumulator through [`ReferenceFreezeOperation`] when `consume` is `true` and through
    /// [`ReferenceReadOperation`] otherwise.
    fn stateful_square_regions(consume: bool) -> Vec<FlatProgram<EagerArrayIrContext>> {
        let scalar: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        let mut primal = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = primal.add_input(scalar.clone());
        let output = primal
            .add_instruction(ArrayOperation::Mul(MulOperation::new()), Vec::new(), vec![input, input], None)
            .unwrap()[0];
        let primal = primal.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut rule = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = rule.add_input(scalar.clone());
        let tangent = rule.add_input(scalar);
        let coefficient = rule.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
        let zero = rule.add_constant(ArrayIrValue::Array(Array::scalar(0.0f32).unwrap()));
        let accumulator = rule.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![zero], None).unwrap()[0];
        let update = rule
            .add_instruction(ArrayOperation::Mul(MulOperation::new()), Vec::new(), vec![coefficient, tangent], None)
            .unwrap()[0];
        rule.add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![accumulator, update], None)
            .unwrap();
        let read = if consume {
            ArrayIrOperation::from(ReferenceFreezeOperation::new())
        } else {
            ArrayIrOperation::from(ReferenceReadOperation::new())
        };
        let derivative = rule.add_instruction(read, Vec::new(), vec![accumulator], None).unwrap()[0];
        let output = rule
            .add_instruction(ArrayOperation::Mul(MulOperation::new()), Vec::new(), vec![input, input], None)
            .unwrap()[0];
        let rule = rule.build(vec![output, derivative], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();
        vec![primal, rule]
    }

    /// Builds the plumbing-reference primal `f(counters..., x) = { counters += x; x }` and the JVP rule
    /// `jvp(counters..., x, ẋ) = { counters += x; (x, ẋ) }` over `counter_count` leading `ref<f32[]>` counters and
    /// an `f32[]` input, so that replaying either region is observable through the counters.
    fn counting_custom_jvp_regions(counter_count: usize) -> Vec<FlatProgram<EagerArrayIrContext>> {
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let mut primal = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let counters = (0..counter_count).map(|_| primal.add_input(reference_type.clone())).collect::<Vec<_>>();
        let x = primal.add_input(scalar_type.clone());
        for counter in counters {
            primal
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![counter, x], None)
                .unwrap();
        }
        let primal = primal
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![x],
                vec![Placeholder; counter_count + 1],
                vec![Placeholder],
            )
            .unwrap();
        let mut jvp = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let counters = (0..counter_count).map(|_| jvp.add_input(reference_type.clone())).collect::<Vec<_>>();
        let x = jvp.add_input(scalar_type.clone());
        let tangent = jvp.add_input(scalar_type);
        for counter in counters {
            jvp.add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![counter, x], None).unwrap();
        }
        let jvp = jvp
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![x, tangent],
                vec![Placeholder; counter_count + 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        vec![primal, jvp]
    }

    /// Checks that discharging local state preserves the declared custom derivative for the chosen read behavior.
    fn check_custom_jvp_reference_discharge(consume: bool) {
        let mut regions = stateful_square_regions(consume);
        let mut rule = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let inputs = regions[1].input_types().iter().map(|r#type| rule.add_input(r#type.clone())).collect::<Vec<_>>();
        let outputs = rule.splice_program(&regions[1], &inputs).unwrap();
        let doubled =
            rule.add_instruction(AddOperation::new(), Vec::new(), vec![outputs[1], outputs[1]], None).unwrap()[0];
        regions[1] = rule.build(vec![outputs[0], doubled], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();
        let program = composite_call_program(
            ArrayIrComposite::from_rule_regions(CompositeJvpRule::Explicit, false),
            regions,
            vec![ArrayType::scalar(DataType::F32).into()],
        );

        // Local rule state is discharged inside the rule region, while the declared JVP rule remains attached and
        // active. Its tangent is deliberately doubled, so differentiating the primal instead cannot pass.
        let discharged = program.discharge_references(0).unwrap().into_program_without_external_references().unwrap();
        assert_eq!(
            discharged.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = composite %0 [
                    primal={
                        lambda %0:f32[] .
                        let %1:f32[] = mul %0 %0
                        in (%1)
                    },
                    jvp={
                        lambda %0:f32[], %1:f32[] .
                        let %2:f32[] = const 0.0
                            %3:f32[] = add %0 %0
                            %4:f32[] = mul %3 %1
                            %5:f32[] = add %2 %4
                            %6:f32[] = mul %0 %0
                            %7:f32[] = add %5 %5
                        in (%6, %7)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert!(!discharged.entry_region_ref().contains_references_in_closure());
        assert_eq!(
            discharged.interpret(vec![ArrayIrValue::Array(Array::scalar(3.0f32).unwrap())]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(9.0f32).unwrap())]),
        );
        assert_eq!(
            discharged.jvp().unwrap().interpret(vec![
                ArrayIrValue::Array(Array::scalar(3.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::scalar(9.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(12.0f32).unwrap()),
            ]),
        );
    }

    /// Checks coefficient hoisting and independent repeated pushforwards with a read or consuming freeze, optionally
    /// for a call that also has reverse-mode rules, which forward-mode linearization must never consult.
    fn check_custom_jvp_linearization_fresh_tangent_state(consume: bool, with_vjp_rule: bool) {
        let endpoint = if consume { "reference_freeze" } else { "reference_read" };
        let operation = ArrayIrComposite::from_rule_regions(CompositeJvpRule::Explicit, with_vjp_rule);
        let mut regions = stateful_square_regions(consume);
        if with_vjp_rule {
            let scalar: ArrayIrType = ArrayType::scalar(DataType::F32).into();
            let mut forward = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let x = forward.add_input(scalar.clone());
            let square = forward
                .add_instruction(ArrayOperation::Mul(MulOperation::new()), Vec::new(), vec![x, x], None)
                .unwrap()[0];
            regions.push(forward.build(vec![square], vec![Placeholder], vec![Placeholder]).unwrap());
            let mut backward = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let seed = backward.add_input(scalar);
            regions.push(backward.build(vec![seed], vec![Placeholder], vec![Placeholder]).unwrap());
        }
        let program = composite_call_program(operation, regions, vec![ArrayType::scalar(DataType::F32).into()]);

        // Linearization hoists the pure coefficient into the primal program as a residual, while the tangent
        // program allocates a fresh accumulator on every application.
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = mul %0 %0
                    %2:f32[] = add %0 %0
                in (%1, %2)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[] = const 0.0
                    %3:ref<f32[]> = reference_new %2
                    %4:f32[] = mul %1 %0
                    () = reference_add_update %3 %4
                    %5:f32[] = reference_read %3
                in (%5)
            "}
            .trim_end()
            .replace("reference_read", endpoint),
        );
        let primals =
            linearization.primal().interpret(vec![ArrayIrValue::Array(Array::scalar(3.0f32).unwrap())]).unwrap();
        assert_eq!(
            primals,
            vec![
                ArrayIrValue::Array(Array::scalar(9.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(6.0f32).unwrap()),
            ],
        );
        assert_eq!(
            linearization
                .tangent()
                .interpret(vec![ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()), primals[1].clone()]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(12.0f32).unwrap())]),
        );
        assert_eq!(
            linearization
                .tangent()
                .interpret(vec![ArrayIrValue::Array(Array::scalar(5.0f32).unwrap()), primals[1].clone()]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(30.0f32).unwrap())]),
        );
        assert_eq!(
            linearization
                .tangent()
                .interpret(vec![ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()), primals[1].clone()]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(12.0f32).unwrap())]),
        );
    }

    /// Checks nested partitioning when a scan updates an aliased root directly or through its iteration index.
    fn check_custom_jvp_linearization_aliased_reference_carries(scanned_view: bool) {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let referent_type =
            if scanned_view { ArrayType::new_static(DataType::F32, [2]) } else { ArrayType::scalar(DataType::F32) };
        let reference_type = ArrayIrType::Reference(ReferenceType::new(referent_type));
        let initial =
            if scanned_view { Array::vector(vec![0.0f32; 2]).unwrap() } else { Array::scalar(0.0f32).unwrap() };
        let expected_state =
            if scanned_view { Array::vector(vec![3.0f32; 2]).unwrap() } else { Array::scalar(6.0f32).unwrap() };
        let mut branch = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let reference = branch.add_input(reference_type.clone());
        let branch = branch
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![reference],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        // The body receives the same whole-root reference through a known carry and an unknown input. The
        // unknown input is either another carry or a stacked reference indexed by the body's iteration input.
        // Each iteration updates through the unknown input, then reads the root through the known carry.
        let mut body = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let index = body.add_input(ArrayType::scalar(DataType::I64).into());
        let known_reference = body.add_input(reference_type.clone());
        let increment = body.add_input(scalar_type.clone());
        let unknown_root = body.add_input(reference_type.clone());
        let (transforms, inputs) = if scanned_view {
            (
                vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }],
                vec![unknown_root, increment, index],
            )
        } else {
            (Vec::new(), vec![unknown_root, increment])
        };
        body.add_instruction(ReferenceAddUpdateOperation::new().with_transforms(transforms), Vec::new(), inputs, None)
            .unwrap();
        let read = body
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![known_reference], None)
            .unwrap()[0];
        let read = if scanned_view {
            body.add_instruction(
                ArrayOperation::Reduce(ReduceOperation::new(vec![0], ReductionKind::Sum)),
                Vec::new(),
                vec![read],
                None,
            )
            .unwrap()[0]
        } else {
            read
        };
        let body_outputs =
            if scanned_view { vec![known_reference, read] } else { vec![known_reference, read, unknown_root] };
        let carry_count = body_outputs.len();
        let body = body
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                body_outputs,
                vec![Placeholder; 4],
                vec![Placeholder; carry_count],
            )
            .unwrap();

        let mut rule = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let reference = rule.add_input(reference_type.clone());
        let primal = rule.add_input(scalar_type.clone());
        let tangent = rule.add_input(scalar_type.clone());
        let zero = rule.add_constant(ArrayIrValue::Array(Array::scalar(0.0f32).unwrap()));
        let predicate = rule
            .add_instruction(
                ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::GreaterThan)),
                Vec::new(),
                vec![tangent, zero],
                None,
            )
            .unwrap()[0];
        let branch = rule.import_program(branch);
        let alias = rule
            .add_instruction(ConditionOperation::new(), vec![branch, branch], vec![predicate, reference], None)
            .unwrap()[0];
        let body = rule.import_program(body);
        let outputs = rule
            .add_instruction(ScanOperation::new(carry_count, 2), vec![body], vec![reference, tangent, alias], None)
            .unwrap()
            .to_vec();
        let rule = rule
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![primal, outputs[1]],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();
        let reference = ArrayReference::new(initial.clone());
        assert_eq!(
            rule.interpret(vec![
                ArrayIrValue::Reference(reference.clone()),
                ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(3.0f32).unwrap()),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(6.0f32).unwrap()),
            ]),
        );
        assert_eq!(reference.read(), Ok(expected_state.clone()));

        // The condition's predicate is tangent-dependent, making its forwarded reference unknown even though
        // canonical provenance still ties it to the known reference. Nested partitioning must retain that alias.
        let mut primal = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        primal.add_input(reference_type.clone());
        let input = primal.add_input(scalar_type.clone());
        let primal = primal
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![input],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let program = composite_call_program(
            ArrayIrComposite::from_rule_regions(CompositeJvpRule::Explicit, false)
                .with_non_differentiated_count(1)
                .unwrap(),
            vec![primal, rule],
            vec![reference_type, scalar_type],
        );
        let linearization = program.entry_region_ref().linearize(&[1]).unwrap();
        let reference = ArrayReference::new(initial);
        let mut primals = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Reference(reference.clone()),
                ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
            ])
            .unwrap();
        assert_eq!(primals.remove(0), ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()));
        let mut tangents = vec![ArrayIrValue::Array(Array::scalar(3.0f32).unwrap())];
        tangents.extend(primals);
        assert_eq!(
            linearization.tangent().interpret(tangents),
            Ok(vec![ArrayIrValue::Array(Array::scalar(6.0f32).unwrap())]),
        );
        assert_eq!(reference.read(), Ok(expected_state));
    }

    /// Tracer of the array IR universe used by the plumbing-reference tests.
    type ArrayIrTracer = DomainTracer<EagerArrayIrContext>;

    /// Error message of the forward-mode rejection of calls that have only reverse-mode rules.
    const FORWARD_MODE_REJECTION: &str = "cannot apply forward-mode differentiation to a `composite` call that \
                                          has only reverse-mode rules; it supports only reverse-mode differentiation \
                                          (e.g., `vjp`, `value_and_gradient`, or `jacobian_reverse`)";

    /// Builds the forward rule `forward(x) = (sin(x), cos(x))`, with the cosine as the residual.
    fn sin_forward_program(r#type: &ArrayType) -> FlatProgram<ArrayContext> {
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(r#type.clone());
        let y = builder.add_instruction(SinOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
        let residual = builder.add_instruction(CosOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
        builder.build(vec![y, residual], vec![Placeholder], vec![Placeholder; 2]).unwrap()
    }

    /// Builds the custom rule `backward(residual, cotangent) = 3 * residual * cotangent`, whose result deliberately
    /// differs from the mathematical gradient so that tests can prove that the custom rule is used.
    fn tripled_sin_backward_program(r#type: &ArrayType) -> FlatProgram<ArrayContext> {
        let mut builder = ProgramBuilder::new();
        let residual = builder.add_input(r#type.clone());
        let cotangent = builder.add_input(r#type.clone());
        let three = builder.add_constant(Array::scalar(3.0).unwrap());
        let scaled = builder.add_instruction(MulOperation::new(), Vec::new(), vec![three, residual], None).unwrap()[0];
        let gradient =
            builder.add_instruction(MulOperation::new(), Vec::new(), vec![scaled, cotangent], None).unwrap()[0];
        builder.build(vec![gradient], vec![Placeholder; 2], vec![Placeholder]).unwrap()
    }

    /// Returns the attached regions of a call of `operation` over `f(x) = sin(x)` on `f64[]` values, in its region
    /// order. The rule regions deliberately differ from the true derivative `cos(x)`: the JVP rule doubles it and the
    /// reverse-mode rules triple it, so tests can prove which rule each transform selects.
    fn sin_regions(operation: &ArrayComposite) -> Vec<FlatProgram<ArrayContext>> {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut regions = vec![sin_program(&scalar_type)];
        if operation.jvp_rule() == CompositeJvpRule::Explicit {
            regions.push(doubled_sin_jvp_program(&scalar_type));
        }
        if operation.has_vjp_rule() {
            regions.push(sin_forward_program(&scalar_type));
            regions.push(tripled_sin_backward_program(&scalar_type));
        }
        regions
    }

    /// Builds the reverse-mode rules `forward(counter, x) = (x², 2x)` and `backward(counter, r, ȳ) = r · ȳ` over a
    /// leading `ref<f32[]>` counter and an `f32[]` input, where the backward rule also adds one to the counter so that
    /// every execution of it is observable.
    fn counting_square_vjp_regions() -> Vec<FlatProgram<EagerArrayIrContext>> {
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let mut forward = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        forward.add_input(reference_type.clone());
        let x = forward.add_input(scalar_type.clone());
        let square = forward
            .add_instruction(ArrayOperation::Mul(MulOperation::new()), Vec::new(), vec![x, x], None)
            .unwrap()[0];
        let residual = forward.add_instruction(AddOperation::new(), Vec::new(), vec![x, x], None).unwrap()[0];
        let forward = forward.build(vec![square, residual], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();
        let mut backward = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let counter = backward.add_input(reference_type);
        let residual = backward.add_input(scalar_type.clone());
        let seed = backward.add_input(scalar_type);
        let one = backward.add_constant(ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()));
        backward
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![counter, one], None)
            .unwrap();
        let cotangent = backward
            .add_instruction(ArrayOperation::Mul(MulOperation::new()), Vec::new(), vec![residual, seed], None)
            .unwrap()[0];
        let backward = backward.build(vec![cotangent], vec![Placeholder; 3], vec![Placeholder]).unwrap();
        vec![forward, backward]
    }

    /// Builds a reference-capable region that forwards the specified inputs unchanged, preserving their order.
    fn forwarded_inputs_program(
        input_types: Vec<ArrayIrType>,
        output_positions: Vec<usize>,
    ) -> FlatProgram<EagerArrayIrContext> {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let inputs = input_types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let outputs = output_positions.iter().map(|position| inputs[*position]).collect::<Vec<_>>();
        builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                outputs,
                vec![Placeholder; input_types.len()],
                vec![Placeholder; output_positions.len()],
            )
            .unwrap()
    }

    // Fixtures of the tests of calls and carriers with retained rules.

    /// Multiplies two array IR tracers, whose values have no operator sugar.
    fn ir_multiply(left: &IrTracer, right: &IrTracer) -> Result<IrTracer, ProgramError> {
        let operation = ArrayIrOperation::from(ArrayOperation::<Array>::Mul(MulOperation::new()));
        Ok(left.context().bind(operation, Vec::new(), &[left.clone(), right.clone()])?.remove(0))
    }

    /// Builds the array IR primal `f(counter, x) = x²` over a leading `ref<f32[]>` counter, which it ignores.
    fn ir_counter_square_program() -> ConvertedProgram {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        builder.add_input(ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let x = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let operation = ArrayIrOperation::from(ArrayOperation::<Array>::Mul(MulOperation::new()));
        let square = builder.add_instruction(operation, Vec::new(), vec![x, x], None).unwrap()[0];
        builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![square],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap()
    }

    /// Builds the member-family program `f(x) = x³` over one input of the provided type.
    fn member_cube_program(r#type: &ArrayType) -> FlatProgram<EagerContext<Array, ArrayOperation<Array>>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let x = builder.add_input(r#type.clone());
        let square = builder
            .add_instruction(ArrayOperation::Mul(MulOperation::new()), Vec::new(), vec![x, x], None)
            .unwrap()[0];
        let cube = builder
            .add_instruction(ArrayOperation::Mul(MulOperation::new()), Vec::new(), vec![square, x], None)
            .unwrap()[0];
        builder.build::<Vec<Array>, Vec<Array>>(vec![cube], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Stages one call of the member-family `definition` over an input of type `r#type`.
    fn member_custom_rule_program(
        definition: &CustomRuleRegistration<Array, ArrayOperation<Array>>,
        r#type: ArrayType,
    ) -> FlatProgram<EagerContext<Array, ArrayOperation<Array>>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let primal = builder.import_program(member_cube_program(&r#type));
        let input = builder.add_input(r#type);
        let operation = ArrayOperation::Composite(CompositeOperation::new(definition.reference()));
        let output = builder.add_instruction(operation, vec![primal], vec![input], None).unwrap()[0];
        builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Returns the outputs of one call of `definition` at `x` and the input cotangent for the output seed `seed`.
    fn call_vjp(
        definition: &TestRegistration,
        x: Array,
        seed: Array,
    ) -> Result<(Vec<Array>, Array), DifferentiationError> {
        let r#type = x.r#type().into_owned();
        let (outputs, pullback) = TestContext::new().vjp(
            |x, ()| {
                let operation = CompositeOperation::new(definition.reference());
                x.context().bind(operation, vec![cube_program(&r#type)], &[x.clone()])
            },
            x,
            (),
        )?;
        Ok((outputs, pullback.apply_materialized(vec![seed])?))
    }

    #[test]
    fn test_composite() {
        let operation = ArrayComposite::from_rule_regions(CompositeJvpRule::Absent, false);
        assert_eq!(operation.name(), COMPOSITE_OPERATION_NAME);
        assert_eq!(operation.non_differentiated_count(), 0);
        assert_eq!(operation.rules(), None);
        assert_eq!(operation.jvp_rule(), CompositeJvpRule::Absent);
        assert!(!operation.has_vjp_rule());
        assert_eq!(format!("{operation}"), "composite");
        assert_eq!(operation.region_slots(), &[RegionSlot::computation("primal")]);

        // Rule regions follow the primal region in the order `["jvp", "forward", "backward"]`, omitting the ones that a
        // call does not have, and every one of them is a dormant rule region without input provenance.
        let combined = ArrayComposite::from_rule_regions(CompositeJvpRule::Explicit, true);
        assert_eq!(combined.jvp_rule(), CompositeJvpRule::Explicit);
        assert!(combined.has_vjp_rule());
        assert_eq!(
            combined.region_slots(),
            &[
                RegionSlot::computation("primal"),
                RegionSlot::rule("jvp"),
                RegionSlot::rule("forward"),
                RegionSlot::rule("backward"),
            ],
        );
        assert_eq!(combined.region_role(0), Some(RegionRole::Computation));
        assert_eq!(combined.region_role(3), Some(RegionRole::Rule));
        assert_eq!(combined.input_region_provenance(0, 0), InputRegionProvenance::Input { index: 0 });
        assert_eq!(combined.input_region_provenance(3, 0), InputRegionProvenance::None);
        assert_eq!(format!("{combined}"), "composite");

        // A forward-mode rule derived from the primal region adds no region, so its rendering records it.
        let derived = ArrayComposite::from_rule_regions(CompositeJvpRule::Primal, false);
        assert_eq!(derived.region_slots(), &[RegionSlot::computation("primal")]);
        assert_eq!(format!("{derived}"), "composite [jvp_from_primal=true]");
        let derived = ArrayComposite::from_rule_regions(CompositeJvpRule::Primal, true);
        assert_eq!(
            format!("{}", derived.clone().with_non_differentiated_count(1).unwrap()),
            "composite [non_differentiated_count=1, jvp_from_primal=true]",
        );
        assert_eq!(
            derived.region_slots(),
            &[RegionSlot::computation("primal"), RegionSlot::rule("forward"), RegionSlot::rule("backward")],
        );
    }

    #[test]
    fn test_composite_jvp_rule() {
        let operation = ArrayComposite::from_rule_regions(CompositeJvpRule::Explicit, false);
        assert_eq!(operation.name(), COMPOSITE_OPERATION_NAME);
        assert_eq!(operation.non_differentiated_count(), 0);
        assert_eq!(operation.jvp_rule(), CompositeJvpRule::Explicit);
        assert!(!operation.has_vjp_rule());
        assert_eq!(format!("{operation}"), "composite");
        assert_eq!(
            format!("{operation:?}"),
            "CompositeOperation { non_differentiated_count: 0, jvp_rule: Explicit, has_vjp_rule: false }",
        );

        // The primal program is a computation region and the user JVP program is a dormant rule region, in that order.
        assert_eq!(operation.region_slots(), &[RegionSlot::computation("primal"), RegionSlot::rule("jvp")]);
        assert_eq!(operation.region_role(0), Some(RegionRole::Computation));
        assert_eq!(operation.region_role(1), Some(RegionRole::Rule));

        // The primal region receives every input at its own position and its outputs are the call's outputs, while
        // the dormant rule region declares no input provenance.
        assert_eq!(operation.input_region_provenance(0, 1), InputRegionProvenance::Input { index: 1 });
        assert_eq!(operation.input_region_provenance(1, 1), InputRegionProvenance::None);
        assert_eq!(
            operation.output_region_provenance(0),
            vec![OutputRegionProvenance { region_index: 0, output_index: 0 }],
        );
    }

    #[test]
    fn test_composite_vjp_rule() {
        let operation = ArrayComposite::from_rule_regions(CompositeJvpRule::Absent, true);
        assert_eq!(operation.name(), COMPOSITE_OPERATION_NAME);
        assert_eq!(operation.non_differentiated_count(), 0);
        assert_eq!(operation.jvp_rule(), CompositeJvpRule::Absent);
        assert!(operation.has_vjp_rule());
        assert_eq!(format!("{operation}"), "composite");
        assert_eq!(
            format!("{operation:?}"),
            "CompositeOperation { non_differentiated_count: 0, jvp_rule: Absent, has_vjp_rule: true }",
        );

        // The primal program is a computation region followed by the dormant forward and backward rule regions.
        assert_eq!(
            operation.region_slots(),
            &[RegionSlot::computation("primal"), RegionSlot::rule("forward"), RegionSlot::rule("backward")],
        );
        assert_eq!(operation.region_role(0), Some(RegionRole::Computation));
        assert_eq!(operation.region_role(1), Some(RegionRole::Rule));
        assert_eq!(operation.region_role(2), Some(RegionRole::Rule));

        // The primal region receives every input at its own position and its outputs are the call's outputs, while
        // the dormant rule regions declare no input provenance.
        assert_eq!(operation.input_region_provenance(0, 1), InputRegionProvenance::Input { index: 1 });
        assert_eq!(operation.input_region_provenance(1, 1), InputRegionProvenance::None);
        assert_eq!(operation.input_region_provenance(2, 1), InputRegionProvenance::None);
        assert_eq!(
            operation.output_region_provenance(0),
            vec![OutputRegionProvenance { region_index: 0, output_index: 0 }],
        );
    }

    #[test]
    fn test_composite_with_non_differentiated_count() {
        let operation = ArrayComposite::from_rule_regions(CompositeJvpRule::Explicit, false)
            .with_non_differentiated_count(1)
            .unwrap();
        assert_eq!(operation.non_differentiated_count(), 1);
        assert_eq!(format!("{operation}"), "composite [non_differentiated_count=1]");

        // Both regions receive the non-differentiated input at its own position, but the JVP region receives a tangent
        // only for the differentiated input.
        let parameter_type = ArrayType::new_static(DataType::F64, [2]);
        let scalar_type = ArrayType::scalar(DataType::F64);
        let input_types = vec![parameter_type.clone(), scalar_type.clone()];
        let primal_interface =
            RegionInterface::new(input_types.clone(), vec![scalar_type.clone()], EffectClasses::NONE);
        let jvp_input_types = vec![parameter_type, scalar_type.clone(), scalar_type.clone()];
        let jvp_interface = RegionInterface::new(
            jvp_input_types.clone(),
            vec![scalar_type.clone(), scalar_type.clone()],
            EffectClasses::NONE,
        );
        assert_eq!(
            operation.infer_region_input_types(&input_types, &[primal_interface.clone(), jvp_interface.clone()]),
            Ok(vec![Some(input_types.clone()), Some(jvp_input_types)]),
        );
        assert_eq!(
            operation.infer_output_types(&input_types, &[primal_interface, jvp_interface]),
            Ok(vec![scalar_type]),
        );
    }

    #[test]
    fn test_composite_with_non_differentiated_count_vjp_rule() {
        let operation = ArrayComposite::from_rule_regions(CompositeJvpRule::Absent, true)
            .with_non_differentiated_count(1)
            .unwrap();
        assert_eq!(operation.non_differentiated_count(), 1);
        assert_eq!(format!("{operation}"), "composite [non_differentiated_count=1]");

        // The primal and forward regions receive the non-differentiated input at its own position, and the backward
        // region receives it ahead of the residuals but produces no cotangent for it.
        let parameter_type = ArrayType::new_static(DataType::F64, [2]);
        let scalar_type = ArrayType::scalar(DataType::F64);
        let input_types = vec![parameter_type.clone(), scalar_type.clone()];
        let backward_input_types = vec![parameter_type, scalar_type.clone(), scalar_type.clone()];
        let interfaces = [
            RegionInterface::new(input_types.clone(), vec![scalar_type.clone()], EffectClasses::NONE),
            RegionInterface::new(
                input_types.clone(),
                vec![scalar_type.clone(), scalar_type.clone()],
                EffectClasses::NONE,
            ),
            RegionInterface::new(backward_input_types.clone(), vec![scalar_type.clone()], EffectClasses::NONE),
        ];
        assert_eq!(
            operation.infer_region_input_types(&input_types, &interfaces),
            Ok(vec![Some(input_types.clone()), Some(input_types.clone()), Some(backward_input_types)]),
        );
        assert_eq!(operation.infer_output_types(&input_types, &interfaces), Ok(vec![scalar_type]));
    }

    #[test]
    fn test_composite_retained_rules() {
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, true));
        let operation = CompositeOperation::new(definition.reference());
        assert_eq!(operation, CompositeOperation::new(definition.reference()));
        assert_eq!(hash_of(&operation), hash_of(&CompositeOperation::new(definition.reference())));
        assert_ne!(operation, operation.clone().with_non_differentiated_count(1).unwrap());

        // Identity is the definition allocation, so an equally named definition built from the same closure types is a
        // different rule set, even though it renders identically.
        let twin =
            CompositeOperation::new(CustomRuleRegistration::new(cube_definition(&counters, true, true)).reference());
        assert_ne!(operation, twin);
        assert_eq!(operation.to_string(), twin.to_string());
        assert_eq!(operation.to_string(), "composite [name=\"cube\"]");
        assert_eq!(
            operation.clone().with_non_differentiated_count(1).unwrap().to_string(),
            "composite [name=\"cube\", non_differentiated_count=1]",
        );
        assert_eq!(operation.rules().unwrap().name(), "cube");
        assert_eq!(
            format!("{:?}", operation),
            "CompositeOperation { name: \"cube\", non_differentiated_count: 0, batching: None, discharged: \
             false, kept_outputs: None }",
        );
    }

    #[test]
    fn test_composite_into_family() {
        // A call with retained rules follows its program into a containing family with the source that the conversion
        // derives from its rules, and it keeps its other state.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(member_cube_definition(&counters));
        let operation = ArrayComposite::new(definition.reference()).with_non_differentiated_count(1).unwrap();
        let converted: CompositeOperation<
            ArrayIrValue<Array>,
            ArrayIrOperation<Array>,
            LiftedCustomRules<Array, ArrayOperation<Array>>,
        > = operation.clone().into_family(LiftedCustomRules::Member);
        assert_eq!(converted.non_differentiated_count(), 1);
        assert_eq!(converted.rules(), Some(&LiftedCustomRules::Member(definition.reference())));
        assert_eq!(converted.jvp_rule(), CompositeJvpRule::Explicit);
        assert!(converted.has_vjp_rule());
        assert_eq!(converted.to_string(), operation.to_string());
        assert_eq!(
            format!("{converted:?}"),
            "CompositeOperation { name: \"cube\", non_differentiated_count: 1, batching: None, discharged: false, \
             kept_outputs: None }",
        );

        // A call with attached rules carries no source, so the conversion never derives one.
        let attached = || {
            ArrayComposite::from_rule_regions(CompositeJvpRule::Explicit, true)
                .with_non_differentiated_count(1)
                .unwrap()
        };
        let converted: ArrayIrComposite = attached().into_family(|_| unreachable!("attached rules carry no source"));
        assert_eq!(
            converted,
            ArrayIrComposite::from_rule_regions(CompositeJvpRule::Explicit, true)
                .with_non_differentiated_count(1)
                .unwrap(),
        );
        assert_eq!(converted.to_string(), attached().to_string());
    }

    #[test]
    fn test_composite_into_attached_family() {
        // A call with attached rules converts into any family with any source type, while a call with retained rules is
        // returned unchanged.
        let attached = ArrayComposite::from_rule_regions(CompositeJvpRule::Primal, true);
        assert_eq!(
            attached.with_non_differentiated_count(1).unwrap().into_attached_family::<
                ArrayIrValue<Array>,
                ArrayIrOperation<Array>,
                CustomRuleReference<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
            >(),
            Ok(ArrayIrComposite::from_rule_regions(CompositeJvpRule::Primal, true)
                .with_non_differentiated_count(1)
                .unwrap()),
        );
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(member_cube_definition(&counters));
        let retained = ArrayComposite::new(definition.reference());
        assert_eq!(
            retained.clone().into_attached_family::<
                ArrayIrValue<Array>,
                ArrayIrOperation<Array>,
                CustomRuleReference<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
            >(),
            Err(retained),
        );
    }

    #[test]
    fn test_composite_type_inference() {
        // A call with both a JVP rule region and reverse-mode rule regions validates every rule interface against the
        // primal interface.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let primal = RegionInterface::new(vec![scalar_type.clone()], vec![scalar_type.clone()], EffectClasses::NONE);
        let jvp = RegionInterface::new(
            vec![scalar_type.clone(), scalar_type.clone()],
            vec![scalar_type.clone(), scalar_type.clone()],
            EffectClasses::NONE,
        );
        let forward = RegionInterface::new(
            vec![scalar_type.clone()],
            vec![scalar_type.clone(), scalar_type.clone()],
            EffectClasses::NONE,
        );
        let backward = RegionInterface::new(
            vec![scalar_type.clone(), scalar_type.clone()],
            vec![scalar_type.clone()],
            EffectClasses::NONE,
        );
        let combined = ArrayComposite::from_rule_regions(CompositeJvpRule::Explicit, true);
        let interfaces = vec![primal.clone(), jvp.clone(), forward.clone(), backward];
        assert_eq!(
            combined.infer_region_input_types(&[scalar_type.clone()], &interfaces),
            Ok(vec![
                Some(vec![scalar_type.clone()]),
                Some(vec![scalar_type.clone(), scalar_type.clone()]),
                Some(vec![scalar_type.clone()]),
                Some(vec![scalar_type.clone(), scalar_type.clone()]),
            ]),
        );
        assert_eq!(combined.infer_output_types(&[scalar_type.clone()], &interfaces), Ok(vec![scalar_type.clone()]));
        assert_eq!(
            combined.infer_output_types(&[scalar_type.clone()], &[primal.clone(), jvp, forward]),
            Err(TypeError::invalid("expected 4 regions but got 3")),
        );

        // A forward-mode rule derived from the primal region has no region to validate.
        let derived = ArrayComposite::from_rule_regions(CompositeJvpRule::Primal, false);
        assert_eq!(derived.infer_output_types(&[scalar_type.clone()], &[primal]), Ok(vec![scalar_type]));
    }

    #[test]
    fn test_composite_type_inference_jvp_rule() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let vector_type = ArrayType::new_static(DataType::F64, [2]);
        let operation = ArrayComposite::from_rule_regions(CompositeJvpRule::Explicit, false);
        let primal_interface = sin_program(&scalar_type).interface();
        let jvp_interface = doubled_sin_jvp_program(&scalar_type).interface();

        // The primal region receives the call inputs, the JVP region receives `(inputs..., input_tangents...)`, and a
        // rule that satisfies the interface contract makes the call produce the primal outputs.
        assert_eq!(
            operation.infer_region_input_types(
                std::slice::from_ref(&scalar_type),
                &[primal_interface.clone(), jvp_interface.clone()],
            ),
            Ok(vec![Some(vec![scalar_type.clone()]), Some(vec![scalar_type.clone(), scalar_type.clone()])]),
        );
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&scalar_type),
                &[primal_interface.clone(), jvp_interface.clone()],
            ),
            Ok(vec![scalar_type.clone()]),
        );

        // Inference maps the primal boundary through _differential_ types rather than requiring the tangents to reuse
        // the primal storage type, so an `f8e8m0fnu` primal pairs with an `f32` tangent.
        let primal_type = ArrayType::scalar(DataType::F8E8M0FNU);
        let tangent_type = ArrayType::scalar(DataType::F32);
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&primal_type),
                &[
                    RegionInterface::new(vec![primal_type.clone()], vec![primal_type.clone()], EffectClasses::NONE),
                    RegionInterface::new(
                        vec![primal_type.clone(), tangent_type.clone()],
                        vec![primal_type.clone(), tangent_type],
                        EffectClasses::NONE,
                    ),
                ],
            ),
            Ok(vec![primal_type]),
        );

        // The JVP interface must be `(inputs..., input_tangents...) → (outputs..., output_tangents...)`, so a
        // primal-shaped rule signature is rejected.
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&scalar_type),
                &[primal_interface.clone(), primal_interface.clone()],
            ),
            Err(TypeError::invalid(
                "`composite` JVP rule input type signature mismatch: expected [f64[], f64[]] but got [f64[]]"
                    .to_string(),
            )),
        );
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&scalar_type),
                &[
                    primal_interface.clone(),
                    RegionInterface::new(
                        vec![scalar_type.clone(), scalar_type.clone()],
                        vec![scalar_type.clone()],
                        EffectClasses::NONE,
                    ),
                ],
            ),
            Err(TypeError::invalid(
                "`composite` JVP rule output type signature mismatch: expected [f64[], f64[]] but got [f64[]]"
                    .to_string(),
            )),
        );

        // The call inputs must match the primal region inputs.
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&vector_type),
                &[primal_interface.clone(), jvp_interface.clone()],
            ),
            Err(TypeError::invalid(
                "`composite` input type signature mismatch: expected [f64[]] but got [f64[2]]".to_string(),
            )),
        );

        // Output inference independently rejects an invalid non-differentiated count, even when region input
        // inference has not run first.
        assert_eq!(
            operation.clone().with_non_differentiated_count(2).unwrap().infer_output_types(
                std::slice::from_ref(&scalar_type),
                &[primal_interface.clone(), jvp_interface.clone()],
            ),
            Err(TypeError::invalid("`composite` non-differentiated input count 2 exceeds input count 1".to_string())),
        );

        // The call carries exactly two regions and at most as many non-differentiated inputs as it has inputs.
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&scalar_type), std::slice::from_ref(&primal_interface)),
            Err(TypeError::invalid("expected 2 regions but got 1".to_string())),
        );
        assert_eq!(
            operation
                .clone()
                .with_non_differentiated_count(2)
                .unwrap()
                .infer_region_input_types(std::slice::from_ref(&scalar_type), &[primal_interface, jvp_interface]),
            Err(TypeError::invalid("`composite` non-differentiated input count 2 exceeds input count 1".to_string())),
        );
    }

    #[test]
    fn test_composite_type_inference_jvp_rule_references() {
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let input_types = vec![reference_type.clone(), scalar_type.clone()];
        let primal_interface =
            RegionInterface::new(input_types.clone(), vec![scalar_type.clone()], EffectClasses::NONE);

        // A reference input in the leading non-differentiated segment is plumbing that both regions receive at its own
        // position, so the rule interface carries a tangent only for the differentiated input.
        let jvp_interface = RegionInterface::new(
            vec![reference_type.clone(), scalar_type.clone(), scalar_type.clone()],
            vec![scalar_type.clone(), scalar_type.clone()],
            EffectClasses::NONE,
        );
        assert_eq!(
            ArrayIrComposite::from_rule_regions(CompositeJvpRule::Explicit, false)
                .with_non_differentiated_count(1)
                .unwrap()
                .infer_output_types(&input_types, &[primal_interface.clone(), jvp_interface]),
            Ok(vec![scalar_type.clone()]),
        );

        // The same input in the differentiated segment would need a tangent reference that the rule cannot define, so
        // it is rejected even when the rule interface declares one.
        let active_jvp_interface = RegionInterface::new(
            vec![reference_type.clone(), scalar_type.clone(), reference_type.clone(), scalar_type.clone()],
            vec![scalar_type.clone(), scalar_type.clone()],
            EffectClasses::NONE,
        );
        assert_eq!(
            ArrayIrComposite::from_rule_regions(CompositeJvpRule::Explicit, false)
                .infer_output_types(&input_types, &[primal_interface, active_jvp_interface]),
            Err(TypeError::invalid(
                "`composite` accepts reference inputs only in its leading non-differentiated segment; move \
                 input 0 of type `ref<f32[]>` before the differentiated inputs"
                    .to_string(),
            )),
        );

        // No output may be a reference, not even a forwarded plumbing input, because the rule would then have to
        // produce its tangent reference.
        let forwarding_primal_interface =
            RegionInterface::new(input_types.clone(), vec![reference_type.clone()], EffectClasses::NONE);
        let forwarding_jvp_interface = RegionInterface::new(
            vec![reference_type.clone(), scalar_type.clone(), scalar_type],
            vec![reference_type.clone(), reference_type],
            EffectClasses::NONE,
        );
        assert_eq!(
            ArrayIrComposite::from_rule_regions(CompositeJvpRule::Explicit, false)
                .with_non_differentiated_count(1)
                .unwrap()
                .infer_output_types(&input_types, &[forwarding_primal_interface, forwarding_jvp_interface]),
            Err(TypeError::invalid(
                "`composite` cannot return a reference, but output 0 has type `ref<f32[]>`".to_string(),
            )),
        );
    }

    #[test]
    fn test_composite_type_inference_vjp_rule() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let vector_type = ArrayType::new_static(DataType::F64, [2]);
        let operation = ArrayComposite::from_rule_regions(CompositeJvpRule::Absent, true);
        let primal_interface = sin_program(&scalar_type).interface();
        let forward_interface = sin_forward_program(&scalar_type).interface();
        let backward_interface = tripled_sin_backward_program(&scalar_type).interface();

        // The primal and forward regions receive the call inputs, the backward region receives the trailing forward
        // residuals followed by one cotangent per primal output, and rules that satisfy the interface contract make the
        // call produce the primal outputs.
        assert_eq!(
            operation.infer_region_input_types(
                std::slice::from_ref(&scalar_type),
                &[primal_interface.clone(), forward_interface.clone(), backward_interface.clone()],
            ),
            Ok(vec![
                Some(vec![scalar_type.clone()]),
                Some(vec![scalar_type.clone()]),
                Some(vec![scalar_type.clone(), scalar_type.clone()]),
            ]),
        );
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&scalar_type),
                &[primal_interface.clone(), forward_interface.clone(), backward_interface.clone()],
            ),
            Ok(vec![scalar_type.clone()]),
        );

        // Inference maps the primal boundary through _differential_ types, so the cotangent boundary of an
        // `f8e8m0fnu` primal is `f32` while the residual keeps its own storage type.
        let primal_type = ArrayType::scalar(DataType::F8E8M0FNU);
        let cotangent_type = ArrayType::scalar(DataType::F32);
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&primal_type),
                &[
                    RegionInterface::new(vec![primal_type.clone()], vec![primal_type.clone()], EffectClasses::NONE),
                    RegionInterface::new(
                        vec![primal_type.clone()],
                        vec![primal_type.clone(), scalar_type.clone()],
                        EffectClasses::NONE,
                    ),
                    RegionInterface::new(
                        vec![scalar_type.clone(), cotangent_type.clone()],
                        vec![cotangent_type],
                        EffectClasses::NONE,
                    ),
                ],
            ),
            Ok(vec![primal_type]),
        );

        // The forward interface must be `inputs... → (outputs..., residuals...)`.
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&scalar_type),
                &[
                    primal_interface.clone(),
                    RegionInterface::new(vec![vector_type.clone()], vec![scalar_type.clone()], EffectClasses::NONE),
                    backward_interface.clone(),
                ],
            ),
            Err(TypeError::invalid(
                "`composite` forward rule input type signature mismatch: expected [f64[]] but got [f64[2]]".to_string(),
            )),
        );
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&scalar_type),
                &[
                    primal_interface.clone(),
                    RegionInterface::new(vec![scalar_type.clone()], Vec::new(), EffectClasses::NONE),
                    backward_interface.clone(),
                ],
            ),
            Err(TypeError::invalid(
                "`composite` forward rule must produce at least the 1 primal output(s) but produced 0 value(s)"
                    .to_string(),
            )),
        );

        // A forward rule with enough outputs must still match the types of the leading primal outputs.
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&scalar_type),
                &[
                    primal_interface.clone(),
                    RegionInterface::new(
                        vec![scalar_type.clone()],
                        vec![vector_type.clone(), scalar_type.clone()],
                        EffectClasses::NONE
                    ),
                    backward_interface.clone(),
                ],
            ),
            Err(TypeError::invalid(
                "`composite` forward rule output type signature mismatch: expected [f64[]] but got [f64[2]]"
                    .to_string()
            )),
        );

        // The backward interface must be `(residuals..., output_cotangents...) → input_cotangents...`, so a
        // primal-shaped rule signature is rejected.
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&scalar_type),
                &[primal_interface.clone(), forward_interface.clone(), primal_interface.clone()],
            ),
            Err(TypeError::invalid(
                "`composite` backward rule input type signature mismatch: expected [f64[], f64[]] but got \
                 [f64[]]"
                    .to_string(),
            )),
        );
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&scalar_type),
                &[
                    primal_interface.clone(),
                    forward_interface.clone(),
                    RegionInterface::new(
                        vec![scalar_type.clone(), scalar_type.clone()],
                        Vec::new(),
                        EffectClasses::NONE,
                    ),
                ],
            ),
            Err(TypeError::invalid(
                "`composite` backward rule output type signature mismatch: expected [f64[]] but got []".to_string(),
            )),
        );

        // The call inputs must match the primal region inputs.
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&vector_type),
                &[primal_interface.clone(), forward_interface.clone(), backward_interface.clone()],
            ),
            Err(TypeError::invalid(
                "`composite` input type signature mismatch: expected [f64[]] but got [f64[2]]".to_string(),
            )),
        );

        // Standalone output inference must reject an excessive non-differentiated count as well.
        assert_eq!(
            operation.clone().with_non_differentiated_count(2).unwrap().infer_output_types(
                std::slice::from_ref(&scalar_type),
                &[primal_interface.clone(), forward_interface.clone(), backward_interface.clone()],
            ),
            Err(TypeError::invalid("`composite` non-differentiated input count 2 exceeds input count 1".to_string())),
        );

        // The call carries exactly three regions and at most as many non-differentiated inputs as it has inputs.
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&scalar_type), std::slice::from_ref(&primal_interface)),
            Err(TypeError::invalid("expected 3 regions but got 1".to_string())),
        );
        assert_eq!(
            operation.clone().with_non_differentiated_count(2).unwrap().infer_region_input_types(
                std::slice::from_ref(&scalar_type),
                &[primal_interface, forward_interface, backward_interface],
            ),
            Err(TypeError::invalid("`composite` non-differentiated input count 2 exceeds input count 1".to_string())),
        );
    }

    #[test]
    fn test_composite_type_inference_vjp_rule_references() {
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let input_types = vec![reference_type.clone(), scalar_type.clone()];
        let primal_interface =
            RegionInterface::new(input_types.clone(), vec![scalar_type.clone()], EffectClasses::NONE);
        let forward_interface = RegionInterface::new(
            input_types.clone(),
            vec![scalar_type.clone(), reference_type.clone()],
            EffectClasses::NONE,
        );
        let backward_interface = RegionInterface::new(
            vec![reference_type.clone(), reference_type.clone(), scalar_type.clone()],
            vec![scalar_type.clone()],
            EffectClasses::NONE,
        );
        let interfaces = [primal_interface.clone(), forward_interface.clone(), backward_interface.clone()];

        // The stash-gradients boundary: a leading plumbing reference that the forward rule hands to the backward rule
        // as a residual of the same reference type is accepted.
        let operation = ArrayIrComposite::from_rule_regions(CompositeJvpRule::Absent, true)
            .with_non_differentiated_count(1)
            .unwrap();
        assert_eq!(operation.infer_output_types(&input_types, &interfaces), Ok(vec![scalar_type.clone()]));
        assert_eq!(
            operation.infer_region_input_types(&input_types, &interfaces),
            Ok(vec![
                Some(input_types.clone()),
                Some(input_types.clone()),
                Some(vec![reference_type.clone(), reference_type.clone(), scalar_type.clone()]),
            ]),
        );

        // A reference-typed residual that matches no plumbing input cannot be a forwarded plumbing reference.
        let other_reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F64)));
        let allocating_forward_interface = RegionInterface::new(
            input_types.clone(),
            vec![scalar_type.clone(), other_reference_type.clone()],
            EffectClasses::NONE,
        );
        let allocating_backward_interface = RegionInterface::new(
            vec![reference_type.clone(), other_reference_type, scalar_type.clone()],
            vec![scalar_type.clone()],
            EffectClasses::NONE,
        );
        assert_eq!(
            operation.infer_output_types(
                &input_types,
                &[primal_interface.clone(), allocating_forward_interface, allocating_backward_interface],
            ),
            Err(TypeError::invalid(
                "`composite` forward rule returns residual 0 of reference type `ref<f64[]>`, which matches \
                 none of its leading non-differentiated inputs"
                    .to_string(),
            )),
        );

        // Each leading non-differentiated reference input may supply at most one residual. A second residual of
        // the same reference type exceeds that input's allowance, even though its type matches.
        let duplicating_forward_interface = RegionInterface::new(
            input_types.clone(),
            vec![scalar_type.clone(), reference_type.clone(), reference_type.clone()],
            EffectClasses::NONE,
        );
        let duplicating_backward_interface = RegionInterface::new(
            vec![reference_type.clone(), reference_type.clone(), reference_type.clone(), scalar_type.clone()],
            vec![scalar_type.clone()],
            EffectClasses::NONE,
        );
        assert_eq!(
            operation.infer_output_types(
                &input_types,
                &[primal_interface.clone(), duplicating_forward_interface, duplicating_backward_interface],
            ),
            Err(TypeError::invalid(
                "`composite` forward rule returns residual 1 of reference type `ref<f32[]>`, but every leading \
                 non-differentiated input of that type is already forwarded by an earlier residual"
                    .to_string(),
            )),
        );

        // A reference input in the differentiated segment would need tangent and cotangent references that the rules
        // cannot define, so it is rejected even when the rule interfaces declare them.
        let active_backward_interface = RegionInterface::new(
            vec![reference_type.clone(), scalar_type.clone()],
            vec![reference_type.clone(), scalar_type.clone()],
            EffectClasses::NONE,
        );
        assert_eq!(
            ArrayIrComposite::from_rule_regions(CompositeJvpRule::Absent, true)
                .infer_output_types(&input_types, &[primal_interface, forward_interface, active_backward_interface]),
            Err(TypeError::invalid(
                "`composite` accepts reference inputs only in its leading non-differentiated segment; move \
                 input 0 of type `ref<f32[]>` before the differentiated inputs"
                    .to_string(),
            )),
        );

        // No output may be a reference, not even a forwarded plumbing input, because the backward rule would then have
        // to consume its cotangent reference.
        let forwarding_primal_interface =
            RegionInterface::new(input_types.clone(), vec![reference_type.clone()], EffectClasses::NONE);
        let forwarding_forward_interface =
            RegionInterface::new(input_types.clone(), vec![reference_type.clone()], EffectClasses::NONE);
        let forwarding_backward_interface =
            RegionInterface::new(vec![reference_type.clone(), reference_type], vec![scalar_type], EffectClasses::NONE);
        assert_eq!(
            operation.infer_output_types(
                &input_types,
                &[forwarding_primal_interface, forwarding_forward_interface, forwarding_backward_interface],
            ),
            Err(TypeError::invalid(
                "`composite` cannot return a reference, but output 0 has type `ref<f32[]>`".to_string(),
            )),
        );
    }

    #[test]
    fn test_composite_type_inference_retained_rules() {
        // A call returns the outputs of its primal region, whose inputs are the call's inputs.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, false));
        let scalar_type = ArrayType::scalar(DataType::F64);
        let region = RegionInterface::new(vec![scalar_type.clone()], vec![scalar_type.clone()], EffectClasses::NONE);
        let operation = CompositeOperation::new(definition.reference());
        assert_eq!(
            operation.infer_output_types(&[scalar_type.clone()], std::slice::from_ref(&region)),
            Ok(vec![scalar_type.clone()]),
        );
        assert_eq!(
            operation.infer_output_types(&[scalar_type.clone()], &[]),
            Err(TypeError::invalid("expected 1 region but got 0")),
        );
        assert_eq!(
            Operation::infer_output_types(
                &operation,
                &[ArrayType::scalar(DataType::F32)],
                std::slice::from_ref(&region),
            ),
            Err(TypeError::invalid("`composite` input type signature mismatch: expected [f64[]] but got [f32[]]")),
        );
        assert_eq!(
            Operation::infer_output_types(
                &operation.clone().with_non_differentiated_count(2).unwrap(),
                &[scalar_type.clone()],
                std::slice::from_ref(&region),
            ),
            Err(TypeError::invalid("`composite` non-differentiated input count 2 exceeds input count 1")),
        );

        // A batched call's leading inputs are the boundary inputs of its batching levels, which a call that is
        // reconfigured after it was batched must still declare as non-differentiated.
        let definition = CustomRuleRegistration::new(
            IrDefinition::new("identity")
                .with_jvp(|primals, tangents| Ok((primals.to_vec(), tangents.to_vec())))
                .with_batching_support(),
        );
        let input_type: ArrayIrType = scalar_type.into();
        let mut primal = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = primal.add_input(input_type.clone());
        let primal = primal
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![input],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let extent = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5)).unwrap());
        let batched = ir_custom_rule_program(&definition, 0, primal)
            .batched_with_threaded_extent(
                DimensionType::from(extent),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        let ArrayIrOperation::Composite(batched_operation) = batched.instructions()[0].operation().clone() else {
            panic!("batching should stage a custom rule call");
        };
        assert!(matches!(
            batched.map_operations(|operation| {
                Ok(match operation {
                    ArrayIrOperation::Composite(operation) => {
                        ArrayIrOperation::Composite(operation.clone().with_non_differentiated_count(0)?)
                    }
                    operation => operation.clone(),
                })
            }),
            Err(ProgramError::Type(error)) if error == TypeError::invalid(
                "batched `composite` `identity` must have at least 1 non-differentiated inputs, which are the \
                 boundary inputs that its batching levels prepended to its inputs, but got 0",
            ),
        ));

        // The batched call cannot be staged without the boundary input of its batching level.
        let batched_type: ArrayIrType = batched.input_types()[1].clone();
        let batched_region =
            RegionInterface::new(vec![batched_type.clone()], vec![batched_type.clone()], EffectClasses::NONE);
        assert_eq!(
            batched_operation.infer_output_types(&[batched_type], std::slice::from_ref(&batched_region)),
            Err(TypeError::invalid(
                "batched `composite` `identity` has 1 inputs but its batching levels record 1 boundary \
                 inputs and 1 unbatched inputs",
            )),
        );
    }

    #[test]
    fn test_composite_type_inference_retained_rules_reference_boundary() {
        // The reference contract is validated when a call is staged: references are accepted only in the leading
        // non-differentiated segment.
        let definition = CustomRuleRegistration::new(IrDefinition::new("counter"));
        let program = ir_counter_square_program();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let primal = builder.import_program(program.clone());
        let inputs = program.input_types().into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        assert!(matches!(
            builder.add_instruction(
                ArrayIrOperation::Composite(CompositeOperation::new(definition.reference())),
                vec![primal],
                inputs,
                None,
            ),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`composite` accepts reference inputs only in its leading non-differentiated segment; \
                               move input 0 of type `ref<f32[]>` before the differentiated inputs",
        ));
        let staged = ir_custom_rule_program(&definition, 1, program);
        assert_eq!(
            staged.to_string(),
            indoc! {"
                lambda %0:ref<f32[]>, %1:f32[] .
                let %2:f32[] = composite [name=\"counter\", non_differentiated_count=1] %0 %1 [
                    primal={
                        lambda %0:ref<f32[]>, %1:f32[] .
                        let %2:f32[] = mul %1 %1
                        in (%2)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_composite_region_data_flow() {
        let operation = ArrayComposite::from_rule_regions(CompositeJvpRule::Explicit, true);
        assert!(matches!(operation.region_data_flow(), RegionDataFlow::Provenance));
        assert_eq!(operation.input_region_provenance(0, 1), InputRegionProvenance::Input { index: 1 });
        assert_eq!(operation.input_region_provenance(1, 1), InputRegionProvenance::None);
        assert_eq!(
            operation.output_region_provenance(1),
            vec![OutputRegionProvenance { region_index: 0, output_index: 1 }],
        );
    }

    #[test]
    fn test_composite_prune_boundary() {
        // A call of `x ↦ (sin(x), cos(x))` with explicit forward- and reverse-mode rules whose second output is
        // unused drops that output from its primal region while keeping every input, and it records which outputs of
        // its rules it keeps, so its rules keep their complete output structure and its derivatives match those of the
        // unpruned call, whether it is batched before or after it is pruned.
        type Tracer = DomainTracer<ArrayContext>;
        let function = composite(|x: Tracer| Ok((x.sin()?, x.cos()?)))
            .with_jvp(|x: Tracer, tangent: Tracer| {
                let tangents = (x.cos()? * tangent.clone(), -(x.sin()? * tangent));
                Ok(((x.sin()?, x.cos()?), tangents))
            })
            .with_vjp(
                |x: Tracer| Ok(((x.sin()?, x.cos()?), x)),
                |x: Tracer, (sine_seed, cosine_seed): (Tracer, Tracer)| {
                    Ok(x.cos()? * sine_seed - x.sin()? * cosine_seed)
                },
            )
            .with_name("sine_cosine");
        let (_, program) = ArrayContext::trace(|x| Ok(function.call(x)?.0), ArrayType::scalar(DataType::F64)).unwrap();
        let program = program.into_flat_program();
        let pruned = program.clone().into_pruned().unwrap();
        assert_eq!(
            pruned.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = composite [name=\"sine_cosine\", kept_outputs=[true, false]] %0 [
                    primal={
                        lambda %0:f64[] .
                        let %1:f64[] = sin %0
                        in (%1)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );

        // Forward mode projects the kept outputs of the JVP rule, and reverse mode keeps every output of the forward
        // rule, seeding the pruned one with a structural zero.
        let reverse = |program: &FlatProgram<ArrayContext>, x: Array, seed: Array| {
            let linearization = program
                .entry_region_ref()
                .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose, None)
                .unwrap();
            let pullback = linearization.tangent().transpose_with_respect_to(&[0], &[]).unwrap();
            let mut pullback_inputs = vec![seed];
            pullback_inputs.extend(linearization.primal().interpret(vec![x]).unwrap().into_iter().skip(1));
            pullback.interpret(pullback_inputs).unwrap()
        };
        let (x, one) = (Array::scalar(0.7).unwrap(), Array::scalar(1.0).unwrap());
        assert_eq!(
            pruned.jvp().unwrap().interpret(vec![x.clone(), one.clone()]),
            Ok(vec![Array::scalar(0.7f64.sin()).unwrap(), Array::scalar(0.7f64.cos()).unwrap()]),
        );
        assert_eq!(
            pruned.jvp().unwrap().interpret(vec![x.clone(), one.clone()]),
            program.jvp().unwrap().interpret(vec![x.clone(), one.clone()])
        );
        assert_eq!(reverse(&pruned, x.clone(), one.clone()), vec![Array::scalar(0.7f64.cos()).unwrap()]);
        assert_eq!(reverse(&pruned, x.clone(), one.clone()), reverse(&program, x, one));

        // The batching record of a call pruned after it was batched keeps the used outputs, and a call batched after
        // it was pruned projects its rules' outputs at every level.
        let batch_program = |program: &FlatProgram<ArrayContext>| {
            program
                .batched(
                    3,
                    ShardingDimension::Replicated,
                    &[BatchAxis::new(0)],
                    ProgramBatchingOutputAxesPolicy::Natural,
                )
                .unwrap()
                .into_parts()
                .0
        };
        let x = Array::vector(vec![0.5, 1.0, 1.5]).unwrap();
        let ones = Array::vector(vec![1.0; 3]).unwrap();
        let expected_jvp = batch_program(&program).jvp().unwrap().interpret(vec![x.clone(), ones.clone()]).unwrap();
        let expected_reverse = reverse(&batch_program(&program), x.clone(), ones.clone());
        assert_eq!(expected_reverse, vec![Array::vector(vec![0.5f64.cos(), 1.0f64.cos(), 1.5f64.cos()]).unwrap()]);
        for batched in [batch_program(&pruned), batch_program(&program).into_pruned().unwrap()] {
            assert_eq!(
                batched.to_string(),
                indoc! {"
                    lambda %0:f64[3] .
                    let %1:f64[3] = composite [
                        name=\"sine_cosine\",
                        batching=[(extent=3, input_axes=[axis 0], output_axes=[axis 0])],
                        kept_outputs=[true, false],
                    ] %0 [
                        primal={
                            lambda %0:f64[3] .
                            let %1:f64[3] = sin %0
                            in (%1)
                        },
                    ]
                    in (%1)
                "}
                .trim_end(),
            );
            assert_eq!(batched.jvp().unwrap().interpret(vec![x.clone(), ones.clone()]), Ok(expected_jvp.clone()));
            assert_eq!(reverse(&batched, x.clone(), ones.clone()), expected_reverse);
        }
    }

    #[test]
    fn test_composite_prune_boundary_kept_calls() {
        // Attached rule regions mirror the complete output boundary of the primal region, so a call with a JVP rule
        // region stays whole, while a call without rule regions is pruned like any other call. Derived calls apply a
        // custom batching rule to every output, so a call whose forward-mode rule is derived from its primal and that
        // has a custom batching rule stays whole as well.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let attached_program = |jvp_rule: CompositeJvpRule| -> FlatProgram<ArrayContext> {
            let mut primal = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let x = primal.add_input(scalar_type.clone());
            let sine = primal.add_instruction(SinOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
            let primal = primal.build::<Vec<Array>, Vec<Array>>(vec![sine, x], vec![Placeholder], vec![Placeholder; 2]);
            let mut regions = vec![primal.unwrap()];
            if jvp_rule == CompositeJvpRule::Explicit {
                let mut jvp = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
                let x = jvp.add_input(scalar_type.clone());
                let tangent = jvp.add_input(scalar_type.clone());
                let sine = jvp.add_instruction(SinOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
                let cosine = jvp.add_instruction(CosOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
                let sine_tangent =
                    jvp.add_instruction(MulOperation::new(), Vec::new(), vec![cosine, tangent], None).unwrap()[0];
                let jvp = jvp.build::<Vec<Array>, Vec<Array>>(
                    vec![sine, x, sine_tangent, tangent],
                    vec![Placeholder; 2],
                    vec![Placeholder; 4],
                );
                regions.push(jvp.unwrap());
            }
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let regions = regions.into_iter().map(|region| builder.import_program(region)).collect::<Vec<_>>();
            let x = builder.add_input(scalar_type.clone());
            let operation = ArrayComposite::from_rule_regions(jvp_rule, false);
            let output = builder.add_instruction(operation, regions, vec![x], None).unwrap()[0];
            builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let program = attached_program(CompositeJvpRule::Explicit);
        assert_eq!(program.clone().into_pruned().unwrap().to_string(), program.to_string());
        assert_eq!(
            attached_program(CompositeJvpRule::Primal).into_pruned().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = composite [jvp_from_primal=true] %0 [
                    primal={
                        lambda %0:f64[] .
                        let %1:f64[] = sin %0
                        in (%1)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );

        type Tracer = DomainTracer<ArrayContext>;
        let function = composite(|x: Tracer| Ok((x.sin()?, x.cos()?))).with_batching(
            |_: BatchingLevelExtent<Tracer>, x: Tracer, axis: BatchAxis| Ok(((x.sin()?, x.cos()?), (axis, axis))),
        );
        let (_, program) = ArrayContext::trace(|x| Ok(function.call(x)?.0), scalar_type.clone()).unwrap();
        let program = program.into_flat_program();
        assert_eq!(program.clone().into_pruned().unwrap().to_string(), program.to_string());
    }

    #[test]
    fn test_composite_reference_discharge_jvp_rule() {
        // Discharge preserves both a read of the accumulator and a consuming freeze.
        check_custom_jvp_reference_discharge(false);
        check_custom_jvp_reference_discharge(true);
    }

    #[test]
    fn test_composite_reference_discharge_vjp_rule() {
        // The primal is the identity, but the backward rule deliberately returns three times its cotangent. A local
        // lifecycle inside that dormant rule must disappear during discharge without replacing the custom derivative.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let identity = {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let input = builder.add_input(scalar_type.clone());
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![input],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let mut backward = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let cotangent = backward.add_input(scalar_type.clone());
        let three = backward.add_constant(ArrayIrValue::Array(Array::scalar(3.0f32).unwrap()));
        let scaled = backward
            .add_instruction(ArrayOperation::from(MulOperation::new()), Vec::new(), vec![three, cotangent], None)
            .unwrap()[0];
        let reference =
            backward.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![scaled], None).unwrap()[0];
        let output = backward
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let backward = backward
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let program = composite_call_program(
            ArrayIrComposite::from_rule_regions(CompositeJvpRule::Absent, true),
            vec![identity.clone(), identity, backward],
            vec![scalar_type],
        )
        .discharge_references(0)
        .unwrap()
        .into_program_without_external_references()
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = composite %0 [
                    primal={
                        lambda %0:f32[] .
                        in (%0)
                    },
                    forward={
                        lambda %0:f32[] .
                        in (%0)
                    },
                    backward={
                        lambda %0:f32[] .
                        let %1:f32[] = const 3.0
                            %2:f32[] = mul %1 %0
                        in (%2)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert!(!program.entry_region_ref().contains_references_in_closure());
        let input = ArrayIrValue::Array(Array::scalar(5.0f32).unwrap());
        assert_eq!(program.interpret(vec![input.clone()]), Ok(vec![input.clone()]));
        let linearization = program
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose, None)
            .unwrap();
        let mut primal_outputs = linearization.primal().interpret(vec![input]).unwrap();
        let mut cotangents = vec![ArrayIrValue::Array(Array::scalar(1.0f32).unwrap())];
        cotangents.extend(primal_outputs.split_off(1));
        assert_eq!(
            linearization.pullback().unwrap().program().interpret(cotangents),
            Ok(vec![ArrayIrValue::Array(Array::scalar(3.0f32).unwrap())]),
        );
    }

    #[test]
    fn test_composite_reference_discharge_vjp_rule_rejects_external_references() {
        // A custom-VJP call threads a plumbing reference into its dormant forward and backward rules, whose
        // reference-typed inputs are bound by the transform that instantiates them and therefore declare no input
        // provenance. Summarizing a condition branch containing such a call skips those rules exactly as the reference
        // analysis does, so discharging the program reaches the call's own discharge rule, which reports that a
        // caller reference cannot cross the composite boundary, instead of failing on undeclared provenance.
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));

        let branch = composite_call_program(
            ArrayIrComposite::from_rule_regions(CompositeJvpRule::Absent, true)
                .with_non_differentiated_count(1)
                .unwrap(),
            vec![
                forwarded_inputs_program(vec![reference_type.clone(), scalar_type.clone()], vec![1]),
                forwarded_inputs_program(vec![reference_type.clone(), scalar_type.clone()], vec![1, 0]),
                forwarded_inputs_program(
                    vec![reference_type.clone(), reference_type.clone(), scalar_type.clone()],
                    vec![2],
                ),
            ],
            vec![reference_type.clone(), scalar_type.clone()],
        );
        let program = composite_call_program(
            ConditionOperation::new(),
            vec![branch.clone(), branch],
            vec![ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)), reference_type, scalar_type],
        );
        assert!(matches!(
            program.discharge_references(0),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "`composite` cannot discharge the reference state of its input 0, because its rules \
                    receive non-differentiated reference inputs by identity and cannot thread their state \
                    explicitly; pass reference-free inputs or discharge external references first",
        ));
    }

    #[test]
    fn test_composite_reference_discharge_retained_rules() {
        // The primal `f(x) = 2x` and the JVP rule `(x, ẋ) ↦ (2x, 2ẋ)` both keep local reference state. Discharging the
        // call discharges its primal region and records that its rule programs must be discharged when traced.
        let primal = {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let x = builder.add_input(ArrayType::scalar(DataType::F32).into());
            let state = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![state, x], None)
                .unwrap();
            let output =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![output],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let definition = |discharge: bool| {
            let definition = IrDefinition::new("doubled").with_jvp(|primals, tangents| {
                let output = primals[0].reference_new()?;
                output.add_update(&primals[0])?;
                let tangent = tangents[0].reference_new()?;
                tangent.add_update(&tangents[0])?;
                Ok((vec![output.read()?], vec![tangent.read()?]))
            });
            CustomRuleRegistration::new(if discharge { definition.with_reference_discharge() } else { definition })
        };
        let x = ArrayIrValue::Array(Array::scalar(3.0f32).unwrap());
        let one = ArrayIrValue::Array(Array::scalar(1.0f32).unwrap());
        let supported = definition(true);
        let discharged = ir_custom_rule_program(&supported, 0, primal.clone())
            .discharge_references(0)
            .unwrap()
            .into_program_without_external_references()
            .unwrap();
        assert_eq!(
            discharged.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = composite [name=\"doubled\", discharged=true] %0 [
                    primal={
                        lambda %0:f32[] .
                        let %1:f32[] = add %0 %0
                        in (%1)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        let jvp = discharged.jvp().unwrap();
        assert!(jvp.regions().iter().flat_map(|region| region.atoms()).all(|atom| !atom.r#type().is_reference()));
        assert_eq!(
            jvp.interpret(vec![x.clone(), one.clone()]),
            Ok(vec![
                ArrayIrValue::Array(Array::scalar(6.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(2.0f32).unwrap())
            ]),
        );

        // Without discharge support, a discharged call whose rule programs keep reference state is not differentiable.
        let unsupported = definition(false);
        let discharged = ir_custom_rule_program(&unsupported, 0, primal)
            .discharge_references(0)
            .unwrap()
            .into_program_without_external_references()
            .unwrap();
        assert!(matches!(
            discharged.jvp(),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`composite` `doubled` cannot differentiate a call whose reference state was \
                               discharged, because its rule programs contain reference state and its definition has \
                               no reference discharge support",
        ));
    }

    #[test]
    fn test_composite_reference_discharge_retained_rules_rejects_reference_inputs() {
        // A call's rules receive its non-differentiated reference inputs by identity and cannot thread their state
        // explicitly, so discharge rejects a call with a reference input and names that limitation.
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let definition = CustomRuleRegistration::new(IrDefinition::new("plumbing"));
        let program = ir_custom_rule_program(
            &definition,
            1,
            forwarded_inputs_program(vec![reference_type, scalar_type], vec![1]),
        );
        assert!(matches!(
            program.discharge_references(0),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "`composite` `plumbing` cannot discharge the reference state of its input 0, because \
                    its rules receive non-differentiated reference inputs by identity and cannot thread their state \
                    explicitly; pass reference-free inputs or discharge external references first",
        ));
    }

    #[test]
    fn test_composite_interpretation() {
        // Interpretation replays only the primal region of a call with every kind of rule, both eagerly and in staged
        // programs.
        let operation = ArrayComposite::from_rule_regions(CompositeJvpRule::Explicit, true);
        let regions = sin_regions(&operation);
        let operation = ArrayOperation::Composite(operation);
        assert_eq!(
            ArrayContext::new().bind(operation.clone(), regions.clone(), &[Array::scalar(2.0).unwrap()]),
            Ok(vec![Array::scalar(2.0f64.sin()).unwrap()]),
        );
        let program = composite_call_program(operation, regions, vec![ArrayType::scalar(DataType::F64)]);
        assert_eq!(
            program.interpret(vec![Array::scalar(2.0).unwrap()]),
            Ok(vec![Array::scalar(2.0f64.sin()).unwrap()])
        );
    }

    #[test]
    fn test_composite_interpretation_jvp_rule() {
        // Interpretation replays the primal region only, so an un-differentiated call never pays for the tangent
        // computation of the JVP region.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let operation =
            ArrayOperation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, false));
        let regions = vec![sin_program(&scalar_type), doubled_sin_jvp_program(&scalar_type)];
        assert_eq!(
            ArrayContext::new().bind(operation, regions, &[Array::scalar(2.0).unwrap()]),
            Ok(vec![Array::scalar(2.0f64.sin()).unwrap()]),
        );
    }

    #[test]
    fn test_composite_interpretation_vjp_rule() {
        // Assertions in both derivative rules make accidental execution observable: interpreting the primal must
        // succeed without running either dormant rule.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let operation =
            ArrayOperation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Absent, true));
        let mut regions = vec![
            sin_program(&scalar_type),
            sin_forward_program(&scalar_type),
            tripled_sin_backward_program(&scalar_type),
        ];
        // Keep each rule's valid interface while making execution of its first instruction fail.
        for region in &mut regions[1..] {
            let mut builder = ProgramBuilder::new();
            let inputs =
                region.input_types().iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
            let predicate = builder.add_constant(Array::scalar(false).unwrap());
            builder
                .add_instruction(
                    AssertOperation::new("dormant derivative rule executed"),
                    Vec::new(),
                    vec![predicate],
                    None,
                )
                .unwrap();
            let outputs = builder.splice_program(region, &inputs).unwrap();
            let output_count = outputs.len();
            *region = builder.build(outputs, vec![Placeholder; inputs.len()], vec![Placeholder; output_count]).unwrap();
        }
        assert_eq!(
            ArrayContext::new().bind(operation, regions, &[Array::scalar(2.0).unwrap()]),
            Ok(vec![Array::scalar(2.0f64.sin()).unwrap()]),
        );
    }

    #[test]
    fn test_composite_interpretation_retained_rules() {
        // An ordinary call computes the primal without invoking any retained rule.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, true));
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = composite [name=\"cube\"] %0 [
                    primal={
                        lambda %0:f64[] .
                        let %1:f64[] = mul %0 %0
                            %2:f64[] = mul %1 %0
                        in (%2)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(program.interpret(vec![Array::scalar(2f64).unwrap()]), Ok(vec![Array::scalar(8f64).unwrap()]));
        assert_eq!(counters.counts(), (0, 0, 0));
    }

    #[test]
    fn test_composite_partial_evaluation() {
        // A call with every kind of rule folds by interpreting its primal region when its inputs are known, and it is
        // residualized unchanged, together with all of its rule regions, when an input is unknown.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let operation = ArrayComposite::from_rule_regions(CompositeJvpRule::Explicit, true);
        let regions = sin_regions(&operation);
        let program = composite_call_program(ArrayOperation::Composite(operation), regions, vec![scalar_type.clone()]);
        let evaluation = program.partially_evaluate(&[PartialValue::Known(Array::scalar(2.0).unwrap())]).unwrap();
        assert!(matches!(
            &evaluation.outputs[0],
            PartialEvaluationOutput::Known(output) if output == &Array::scalar(2.0f64.sin()).unwrap(),
        ));
        let evaluation = program.partially_evaluate(&[PartialValue::Unknown(scalar_type)]).unwrap();
        assert!(matches!(evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = composite %0 [
                    primal={
                        lambda %0:f64[] .
                        let %1:f64[] = sin %0
                        in (%1)
                    },
                    jvp={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = sin %0
                            %3:f64[] = const 2.0
                            %4:f64[] = cos %0
                            %5:f64[] = mul %3 %4
                            %6:f64[] = mul %5 %1
                        in (%2, %6)
                    },
                    forward={
                        lambda %0:f64[] .
                        let %1:f64[] = sin %0
                            %2:f64[] = cos %0
                        in (%1, %2)
                    },
                    backward={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = const 3.0
                            %3:f64[] = mul %2 %0
                            %4:f64[] = mul %3 %1
                        in (%4)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_composite_partial_evaluation_jvp_rule() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let operation =
            ArrayOperation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, false));
        let regions = vec![sin_program(&scalar_type), doubled_sin_jvp_program(&scalar_type)];
        let program = composite_call_program(operation, regions, vec![scalar_type.clone()]);

        // A call whose inputs are all known folds by interpreting its primal region.
        let evaluation = program.partially_evaluate(&[PartialValue::Known(Array::scalar(2.0).unwrap())]).unwrap();
        assert!(matches!(
            &evaluation.outputs[0],
            PartialEvaluationOutput::Known(output) if output == &Array::scalar(2.0f64.sin()).unwrap(),
        ));

        // A call with an unknown input residualizes unchanged instead of inlining its primal region, which keeps the
        // custom rule attached to the residual program.
        let evaluation = program.partially_evaluate(&[PartialValue::Unknown(scalar_type)]).unwrap();
        assert!(matches!(evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = composite %0 [
                    primal={
                        lambda %0:f64[] .
                        let %1:f64[] = sin %0
                        in (%1)
                    },
                    jvp={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = sin %0
                            %3:f64[] = const 2.0
                            %4:f64[] = cos %0
                            %5:f64[] = mul %3 %4
                            %6:f64[] = mul %5 %1
                        in (%2, %6)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(
            evaluation
                .program
                .jvp()
                .unwrap()
                .interpret(vec![Array::scalar(2.0).unwrap(), Array::scalar(1.0).unwrap()]),
            Ok(vec![Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(2.0 * 2.0f64.cos()).unwrap()]),
        );
    }

    #[test]
    fn test_composite_partial_evaluation_specializes_known_reference_input() {
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(extent)]),
        )));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let mut primal_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let reference = primal_builder.add_input(reference_type.clone());
        let input = primal_builder.add_input(scalar_type.clone());
        let value = primal_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        primal_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, value], None)
            .unwrap();
        let primal = primal_builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![input],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let operation = ArrayIrComposite::from_rule_regions(CompositeJvpRule::Absent, false)
            .with_non_differentiated_count(1)
            .unwrap();
        let program = composite_call_program(operation, vec![primal], vec![reference_type, scalar_type.clone()]);

        // A concrete leading reference refines the primal region without executing its effect during preparation.
        let live = ArrayReference::new(Array::vector(vec![1f32, 2., 3.]).unwrap());
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(ArrayIrValue::Reference(live.clone())),
                PartialValue::Unknown(scalar_type),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:ref<f32[3]> .
                let %2:f32[] = composite [non_differentiated_count=1] %1 %0 [
                    primal={
                        lambda %0:ref<f32[3]>, %1:f32[] .
                        let %2:f32[3] = reference_read %0
                            () = reference_add_update %0 %2
                        in (%1)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(live.read(), Ok(Array::vector(vec![1f32, 2., 3.]).unwrap()));
        assert_eq!(evaluation.known_reference_inputs().collect::<Vec<_>>(), vec![1]);
        assert_eq!(
            evaluation.interpret(&EagerArrayIrContext::new(), &[ArrayIrValue::Array(Array::scalar(7f32).unwrap())],),
            Ok(vec![ArrayIrValue::Array(Array::scalar(7f32).unwrap())]),
        );
        assert_eq!(live.read(), Ok(Array::vector(vec![2f32, 4., 6.]).unwrap()));
    }

    #[test]
    fn test_composite_partial_evaluation_vjp_rule() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let operation =
            ArrayOperation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Absent, true));
        let regions = vec![
            sin_program(&scalar_type),
            sin_forward_program(&scalar_type),
            tripled_sin_backward_program(&scalar_type),
        ];
        let program = composite_call_program(operation, regions, vec![scalar_type.clone()]);

        // A call whose inputs are all known folds by interpreting its primal region.
        let evaluation = program.partially_evaluate(&[PartialValue::Known(Array::scalar(2.0).unwrap())]).unwrap();
        assert!(matches!(
            &evaluation.outputs[0],
            PartialEvaluationOutput::Known(output) if output == &Array::scalar(2.0f64.sin()).unwrap(),
        ));

        // A call with an unknown input residualizes unchanged instead of inlining its primal region, which keeps the
        // custom rules attached to the residual program.
        let evaluation = program.partially_evaluate(&[PartialValue::Unknown(scalar_type)]).unwrap();
        assert!(matches!(evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = composite %0 [
                    primal={
                        lambda %0:f64[] .
                        let %1:f64[] = sin %0
                        in (%1)
                    },
                    forward={
                        lambda %0:f64[] .
                        let %1:f64[] = sin %0
                            %2:f64[] = cos %0
                        in (%1, %2)
                    },
                    backward={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = const 3.0
                            %3:f64[] = mul %2 %0
                            %4:f64[] = mul %3 %1
                        in (%4)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        let linearization = evaluation
            .program
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose, None)
            .unwrap();
        let mut outputs = linearization.primal().interpret(vec![Array::scalar(2.0).unwrap()]).unwrap();
        assert_eq!(outputs.remove(0), Array::scalar(2.0f64.sin()).unwrap());
        let mut cotangents = vec![Array::scalar(1.0).unwrap()];
        cotangents.extend(outputs);
        assert_eq!(
            linearization.pullback().unwrap().program().interpret(cotangents),
            Ok(vec![Array::scalar(3.0 * 2.0f64.cos()).unwrap()]),
        );
    }

    #[test]
    fn test_composite_batching() {
        // Batching a call that has every kind of rule batches all of its regions and preserves both rules, so forward
        // mode through the batched call still uses the doubled JVP rule and reverse mode still uses the tripled
        // reverse-mode rules.
        let operation = CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, true);
        let regions = sin_regions(&operation);
        let (value, tangent) = differentiate_at(Array::vector(vec![0.5, 1.0]).unwrap())
            .jvp(Array::vector(vec![1.0, 1.0]).unwrap(), |x| {
                batch(
                    |item| {
                        let operation = ArrayOperation::Composite(operation.clone());
                        Ok(item.context().bind(operation, regions.clone(), &[item.clone()])?.remove(0))
                    },
                    x,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )
                .map_err(ProgramError::from)
            })
            .unwrap();
        assert_eq!(value, Array::vector(vec![0.5f64.sin(), 1.0f64.sin()]).unwrap());
        assert_eq!(tangent, Array::vector(vec![2.0 * 0.5f64.cos(), 2.0 * 1.0f64.cos()]).unwrap());
        let (value, gradient) = differentiate_at(Array::vector(vec![0.5, 1.0]).unwrap())
            .value_and_gradient(|x| {
                let mapped = batch(
                    |item| {
                        let operation = ArrayOperation::Composite(operation.clone());
                        Ok(item.context().bind(operation, regions.clone(), &[item.clone()])?.remove(0))
                    },
                    x,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )
                .unwrap();
                mapped.reduce(&[0], ReductionKind::Sum).unwrap()
            })
            .unwrap();
        assert_eq!(value, Array::scalar(0.5f64.sin() + 1.0f64.sin()).unwrap());
        assert_eq!(gradient, Array::vector(vec![3.0 * 0.5f64.cos(), 3.0 * 1.0f64.cos()]).unwrap());
    }

    #[test]
    fn test_composite_batching_jvp_rule() {
        let output: Array = batch(
            |x| {
                let scalar_type = ArrayType::scalar(DataType::F64);
                let operation =
                    ArrayOperation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, false));
                let regions = vec![sin_program(&scalar_type), doubled_sin_jvp_program(&scalar_type)];
                Ok(x.context().bind(operation, regions, &[x.clone()])?.remove(0))
            },
            Array::vector(vec![0.5, 1.0, 1.5]).unwrap(),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(output, Array::vector(vec![0.5f64.sin(), 1.0f64.sin(), 1.5f64.sin()]).unwrap());
    }

    #[test]
    fn test_composite_batching_jvp_rule_natural_output_axes() {
        let vector_type = ArrayType::new_static(DataType::F64, [3]);

        // The primal has one mapped identity output, one naturally replicated constant output, and one replicated
        // constant whose JVP tangent is mapped. The third pair forces reconciliation to mapped without forcing the
        // independently replicated second pair to acquire a batch axis.
        let primal = {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(vector_type.clone());
            let replicated = builder.add_constant(Array::vector(vec![4.0, 5.0, 6.0]).unwrap());
            let reconciled = builder.add_constant(Array::vector(vec![7.0, 8.0, 9.0]).unwrap());
            builder.build(vec![input, replicated, reconciled], vec![Placeholder], vec![Placeholder; 3]).unwrap()
        };
        let jvp = {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(vector_type.clone());
            let tangent = builder.add_input(vector_type.clone());
            let replicated = builder.add_constant(Array::vector(vec![4.0, 5.0, 6.0]).unwrap());
            let reconciled = builder.add_constant(Array::vector(vec![7.0, 8.0, 9.0]).unwrap());
            let zero = builder.add_constant(Array::vector(vec![0.0, 0.0, 0.0]).unwrap());
            builder
                .build(
                    vec![input, replicated, reconciled, tangent, zero, tangent],
                    vec![Placeholder; 2],
                    vec![Placeholder; 6],
                )
                .unwrap()
        };
        let program: FlatProgram<ArrayContext> = composite_call_program(
            ArrayOperation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, false)),
            vec![primal, jvp],
            vec![vector_type],
        );

        // Mapping the input at packed axis 1 preserves that position for varying outputs. The independent constant
        // remains replicated, while the third primal constant is broadcast only because its corresponding tangent
        // varies at axis 1. None of the attached regions transposes that natural axis to a wrapper-wide convention.
        let (batched, output_axes) = program
            .batched(2, ShardingDimension::Replicated, &[BatchAxis::new(1)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(1), BatchAxis::replicated(), BatchAxis::new(1)]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f64[3, 2] .
                let %1:f64[3, 2], %2:f64[3], %3:f64[3, 2] = composite %0 [
                    primal={
                        lambda %0:f64[3, 2] .
                        let %1:f64[3] = const [4.0, 5.0, 6.0]
                            %2:f64[3] = const [7.0, 8.0, 9.0]
                            %3:f64[3, 2] = broadcast [output_type=f64[3, 2], output_axes=[0]] %2
                        in (%0, %1, %3)
                    },
                    jvp={
                        lambda %0:f64[3, 2], %1:f64[3, 2] .
                        let %2:f64[3] = const [4.0, 5.0, 6.0]
                            %3:f64[3] = const [7.0, 8.0, 9.0]
                            %4:f64[3, 2] = broadcast [output_type=f64[3, 2], output_axes=[0]] %3
                            %5:f64[3] = const [0.0, 0.0, 0.0]
                        in (%0, %2, %4, %1, %5, %1)
                    },
                ]
                in (%1, %2, %3)
            "}
            .trim_end(),
        );

        let input = Array::matrix(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]).unwrap();
        assert_eq!(
            batched.interpret(vec![input.clone()]),
            Ok(vec![
                input,
                Array::vector(vec![4.0, 5.0, 6.0]).unwrap(),
                Array::matrix(3, 2, vec![7.0, 7.0, 8.0, 8.0, 9.0, 9.0]).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_composite_batching_jvp_rule_replicated_input_custom_cotangent_reduction() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let primal = {
            let mut builder = ProgramBuilder::new();
            builder.add_input(scalar_type.clone());
            let parameter = builder.add_input(scalar_type.clone());
            builder.build(vec![parameter], vec![Placeholder; 2], vec![Placeholder]).unwrap()
        };
        let jvp = {
            let mut builder = ProgramBuilder::new();
            let coefficient = builder.add_input(scalar_type.clone());
            let parameter = builder.add_input(scalar_type.clone());
            builder.add_input(scalar_type.clone());
            let tangent = builder.add_input(scalar_type.clone());
            let output_tangent =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![coefficient, tangent], None).unwrap()[0];
            builder.build(vec![parameter, output_tangent], vec![Placeholder; 4], vec![Placeholder; 2]).unwrap()
        };
        let regions = vec![primal, jvp];
        let program = composite_call_program(
            ArrayOperation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, false)),
            regions.clone(),
            vec![scalar_type.clone(), scalar_type],
        );
        let (batched, output_axes) = program
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();

        // The primal ignores its mapped input. Only the custom tangent forces the shared output boundary to be
        // mapped, so a lazy factory cannot recover this natural layout from the primal program alone.
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            batched.jvp().unwrap().interpret(vec![
                Array::vector(vec![2.0, 3.0, 5.0]).unwrap(),
                Array::scalar(7.0).unwrap(),
                Array::vector(vec![0.0, 0.0, 0.0]).unwrap(),
                Array::scalar(2.0).unwrap(),
            ]),
            Ok(vec![Array::vector(vec![7.0, 7.0, 7.0]).unwrap(), Array::vector(vec![4.0, 6.0, 10.0]).unwrap()]),
        );

        // Batching before reverse differentiation retains the custom rule and sums contributions to the shared
        // scalar input. Differentiating the primal body instead would incorrectly produce a scalar cotangent of 3.
        let (value, pullback) =
            differentiate_at((Array::vector(vec![2.0, 3.0, 5.0]).unwrap(), Array::scalar(7.0).unwrap()))
                .vjp(|inputs| {
                    batch(
                        |(coefficient, parameter)| {
                            Ok(coefficient
                                .context()
                                .bind(
                                    ArrayOperation::Composite(CompositeOperation::from_rule_regions(
                                        CompositeJvpRule::Explicit,
                                        false,
                                    )),
                                    regions.clone(),
                                    &[coefficient.clone(), parameter],
                                )?
                                .remove(0))
                        },
                        inputs,
                        (BatchAxis::new(0), BatchAxis::replicated()),
                        BatchAxis::new(0),
                        None,
                    )
                    .map_err(ProgramError::from)
                })
                .unwrap();
        assert_eq!(value, Array::vector(vec![7.0, 7.0, 7.0]).unwrap());
        assert_eq!(
            pullback.apply(Array::vector(vec![1.0, 2.0, 4.0]).unwrap()),
            Ok((
                MaybeZero::Zero(ArrayType::new_static(DataType::F64, [3])),
                MaybeZero::Value(Array::scalar(28.0).unwrap())
            )),
        );
    }

    #[test]
    fn test_composite_batching_jvp_rule_named_axis_outputs() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let primal = {
            let mut builder = ProgramBuilder::new();
            builder.add_input(scalar_type.clone());
            let index = builder
                .add_instruction(AxisIndexOperation::new("items".to_string()), Vec::new(), Vec::new(), None)
                .unwrap()[0];
            builder.build(vec![index], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let jvp = {
            let mut builder = ProgramBuilder::new();
            builder.add_input(scalar_type.clone());
            builder.add_input(scalar_type);
            let index = builder
                .add_instruction(AxisIndexOperation::new("items".to_string()), Vec::new(), Vec::new(), None)
                .unwrap()[0];
            let tangent = builder.add_constant(Array::new(ArrayType::scalar(DataType::Zero), Vec::new()).unwrap());
            builder.build(vec![index, tangent], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap()
        };
        let regions = AttachedRegions::from(vec![primal, jvp]);
        let driver = RecursiveBatchingDriver::new(&regions);
        let context =
            BatchingContext::<_, ArrayBatchingPolicy>::new(ArrayContext::new(), 3).with_axis_name("items".to_string());

        // No input carries a mapped axis, but `axis_index("items")` observes the active transform inside both regions
        // and naturally produces a mapped output. Batching must therefore inspect the regions rather than assuming
        // that all-replicated call inputs imply all-replicated call outputs.
        let outputs = CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, false)
            .batch(&context, &driver, &[ArrayBatch::replicated(Array::scalar(1.0).unwrap())])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &Array::vector(vec![0u64, 1, 2]).unwrap());
    }

    #[test]
    fn test_composite_batching_jvp_rule_preserves_composite() {
        // Differentiating through a batched custom call must still use the deliberately doubled custom rule, because
        // batching preserves the call around batched regions instead of inlining its primal region.
        let (value, gradient) = differentiate_at(Array::vector(vec![0.5, 1.0]).unwrap())
            .value_and_gradient(|x| {
                let mapped = batch(
                    |item| {
                        let scalar_type = ArrayType::scalar(DataType::F64);
                        let operation = ArrayOperation::Composite(CompositeOperation::from_rule_regions(
                            CompositeJvpRule::Explicit,
                            false,
                        ));
                        let regions = vec![sin_program(&scalar_type), doubled_sin_jvp_program(&scalar_type)];
                        Ok(item.context().bind(operation, regions, &[item.clone()])?.remove(0))
                    },
                    x,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )
                .unwrap();
                mapped.reduce(&[0], ReductionKind::Sum).unwrap()
            })
            .unwrap();
        assert_eq!(value, Array::scalar(0.5f64.sin() + 1.0f64.sin()).unwrap());
        assert_eq!(gradient, Array::vector(vec![2.0 * 0.5f64.cos(), 2.0 * 1.0f64.cos()]).unwrap());
    }

    #[test]
    fn test_composite_batching_vjp_rule() {
        let output: Array = batch(
            |x| {
                let scalar_type = ArrayType::scalar(DataType::F64);
                let operation =
                    ArrayOperation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Absent, true));
                let regions = vec![
                    sin_program(&scalar_type),
                    sin_forward_program(&scalar_type),
                    tripled_sin_backward_program(&scalar_type),
                ];
                Ok(x.context().bind(operation, regions, &[x.clone()])?.remove(0))
            },
            Array::vector(vec![0.5, 1.0, 1.5]).unwrap(),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(output, Array::vector(vec![0.5f64.sin(), 1.0f64.sin(), 1.5f64.sin()]).unwrap());
    }

    #[test]
    fn test_composite_batching_vjp_rule_residual_axes_and_replicated_cotangents() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let primal = {
            let mut builder = ProgramBuilder::new();
            let x = builder.add_input(scalar_type.clone());
            let y = builder.add_input(scalar_type.clone());
            let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![x, y], None).unwrap()[0];
            builder.build(vec![output], vec![Placeholder; 2], vec![Placeholder]).unwrap()
        };
        let forward = {
            let mut builder = ProgramBuilder::new();
            let x = builder.add_input(scalar_type.clone());
            let y = builder.add_input(scalar_type.clone());
            let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![x, y], None).unwrap()[0];
            builder.build(vec![output, x, y], vec![Placeholder; 2], vec![Placeholder; 3]).unwrap()
        };
        let backward = {
            let mut builder = ProgramBuilder::new();
            let x = builder.add_input(scalar_type.clone());
            let y = builder.add_input(scalar_type.clone());
            let cotangent = builder.add_input(scalar_type.clone());
            let x_cotangent =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![y, cotangent], None).unwrap()[0];
            let y_cotangent =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![x, cotangent], None).unwrap()[0];
            builder.build(vec![x_cotangent, y_cotangent], vec![Placeholder; 3], vec![Placeholder; 2]).unwrap()
        };
        let program: FlatProgram<ArrayContext> = composite_call_program(
            ArrayOperation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Absent, true)),
            vec![primal, forward, backward],
            vec![scalar_type.clone(), scalar_type.clone()],
        );

        // `x` varies across the batch while `y` is shared, so the forward residuals carry axes `(0, None)`. The
        // backward rule naturally produces both cotangents mapped, but the cotangent of `y` must be summed back to the
        // replicated position rather than leaking a mapped axis through the call boundary.
        let (batched, output_axes) = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f64[2], %1:f64[] .
                let %2:f64[2] = composite %0 %1 [
                    primal={
                        lambda %0:f64[2], %1:f64[] .
                        let %2:f64[2] = broadcast [output_type=f64[2], output_axes=[]] %1
                            %3:f64[2] = mul %0 %2
                        in (%3)
                    },
                    forward={
                        lambda %0:f64[2], %1:f64[] .
                        let %2:f64[2] = broadcast [output_type=f64[2], output_axes=[]] %1
                            %3:f64[2] = mul %0 %2
                        in (%3, %0, %1)
                    },
                    backward={
                        lambda %0:f64[2], %1:f64[], %2:f64[2] .
                        let %3:f64[2] = broadcast [output_type=f64[2], output_axes=[]] %1
                            %4:f64[2] = mul %3 %2
                            %5:f64[2] = mul %0 %2
                            %6:f64[] = reduce [kind=sum, axes=[0]] %5
                        in (%4, %6)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            batched.interpret(vec![Array::vector(vec![2.0, 3.0]).unwrap(), Array::scalar(5.0).unwrap()]),
            Ok(vec![Array::vector(vec![10.0, 15.0]).unwrap()]),
        );

        // Unequal output cotangents verify the actual reduction, not just the presence of a reduce instruction.
        let linearization = batched
            .entry_region_ref()
            .linearize_shared_for_rule(&[0, 1], DifferentiationRule::JvpForTranspose, None)
            .unwrap();
        let mut outputs = linearization
            .primal()
            .interpret(vec![Array::vector(vec![2.0, 3.0]).unwrap(), Array::scalar(5.0).unwrap()])
            .unwrap();
        assert_eq!(outputs.remove(0), Array::vector(vec![10.0, 15.0]).unwrap());
        let mut cotangents = vec![Array::vector(vec![7.0, 11.0]).unwrap()];
        cotangents.extend(outputs);
        assert_eq!(
            linearization.pullback().unwrap().program().interpret(cotangents),
            Ok(vec![Array::vector(vec![35.0, 55.0]).unwrap(), Array::scalar(47.0).unwrap()]),
        );
    }

    #[test]
    fn test_composite_batching_vjp_rule_preserves_composite() {
        // Differentiating through a batched custom call must still use the deliberately tripled custom backward rule,
        // because batching preserves the call around batched regions instead of inlining its primal region.
        let (value, gradient) = differentiate_at(Array::vector(vec![0.5, 1.0]).unwrap())
            .value_and_gradient(|x| {
                let mapped = batch(
                    |item| {
                        let scalar_type = ArrayType::scalar(DataType::F64);
                        let operation = ArrayOperation::Composite(CompositeOperation::from_rule_regions(
                            CompositeJvpRule::Absent,
                            true,
                        ));
                        let regions = vec![
                            sin_program(&scalar_type),
                            sin_forward_program(&scalar_type),
                            tripled_sin_backward_program(&scalar_type),
                        ];
                        Ok(item.context().bind(operation, regions, &[item.clone()])?.remove(0))
                    },
                    x,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )
                .unwrap();
                mapped.reduce(&[0], ReductionKind::Sum).unwrap()
            })
            .unwrap();
        assert_eq!(value, Array::scalar(0.5f64.sin() + 1.0f64.sin()).unwrap());
        assert_eq!(gradient, Array::vector(vec![3.0 * 0.5f64.cos(), 3.0 * 1.0f64.cos()]).unwrap());
    }

    #[test]
    fn test_composite_batching_retained_rules() {
        // Batching a call batches its primal and records the level without invoking any rule.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, true));
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        let batched = batched_program(&program, 3, &[BatchAxis::new(0)]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[3] = composite [name=\"cube\", batching=[(extent=3, input_axes=[axis 0], output_axes=[axis 0])]] %0 [
                    primal={
                        lambda %0:f64[3] .
                        let %1:f64[3] = mul %0 %0
                            %2:f64[3] = mul %1 %0
                        in (%2)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        let x = Array::vector(vec![1f64, 2f64, 3f64]).unwrap();
        assert_eq!(batched.interpret(vec![x]), Ok(vec![Array::vector(vec![1f64, 8f64, 27f64]).unwrap()]));
        assert_eq!(counters.counts(), (0, 0, 0));
    }

    #[test]
    fn test_composite_batching_retained_rules_custom_batching_rule() {
        // A custom batching rule takes precedence over structurally batching the primal region. This rule computes
        // `x · (x · x)` rather than the primal's `(x · x) · x`, which shows that it batched the call, and it declares
        // the batch axes of its outputs. It is traced once per level and input signature.
        let invocations = Arc::new(AtomicUsize::new(0));
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, false).with_batching_rule({
            let invocations = invocations.clone();
            move |_, _, inputs, input_axes| {
                invocations.fetch_add(1, Ordering::SeqCst);
                let x = inputs[0].clone();
                Ok((vec![x.clone() * (x.clone() * x)], input_axes.to_vec()))
            }
        }));
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        let batched = batched_program(&program, 3, &[BatchAxis::new(0)]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[3] = composite [name=\"cube\", batching=[(extent=3, input_axes=[axis 0], output_axes=[axis 0])]] %0 [
                    primal={
                        lambda %0:f64[3] .
                        let %1:f64[3] = mul %0 %0
                            %2:f64[3] = mul %0 %1
                        in (%2)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        let x = Array::vector(vec![1f64, 2f64, 3f64]).unwrap();
        assert_eq!(batched.interpret(vec![x.clone()]), Ok(vec![Array::vector(vec![1f64, 8f64, 27f64]).unwrap()]));
        batched_program(&program, 3, &[BatchAxis::new(0)]);
        assert_eq!(invocations.load(Ordering::SeqCst), 1);

        // Differentiating the batched call traces the JVP rule at the unbatched types and batches it structurally,
        // aligned to the batch axes that the rule declared (the rule computes `ẏ = x² ẋ`).
        let jvp = batched.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3] .
                let %2:f64[3] = mul %0 %0
                    %3:f64[3] = mul %2 %0
                    %4:f64[3] = mul %2 %1
                in (%3, %4)
            "}
            .trim_end(),
        );
        assert_eq!(
            jvp.interpret(vec![x, Array::vector(vec![1f64; 3]).unwrap()]),
            Ok(vec![Array::vector(vec![1f64, 8f64, 27f64]).unwrap(), Array::vector(vec![1f64, 4f64, 9f64]).unwrap()]),
        );
        assert_eq!(counters.counts(), (1, 0, 0));
    }

    #[test]
    fn test_composite_batching_retained_rules_custom_batching_rule_output_types() {
        // Unbatching each output of a custom batching rule along its declared batch axis must give the primal's output
        // type. This rule returns its mapped input as an output that it declares replicated, whose per-item type is
        // therefore `f64[3]` rather than the primal's `f64[]`.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(
            cube_definition(&counters, true, false)
                .with_batching_rule(|_, _, inputs, _| Ok((vec![inputs[0].clone()], vec![BatchAxis::replicated()]))),
        );
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        let output_axes_policy = ProgramBatchingOutputAxesPolicy::Natural;
        assert!(matches!(
            program.batched(3, ShardingDimension::Replicated, &[BatchAxis::new(0)], output_axes_policy),
            Err(BatchingError::Type(TypeError::Invalid { message }))
                if message == "`composite` `cube` batching rule output type signature mismatch: expected per-item \
                    output types [f64[]] but got [f64[3]]",
        ));
    }

    #[test]
    fn test_composite_batching_retained_rules_custom_batching_rule_reference_discharge() {
        // Batching a discharged call with a custom batching rule makes the rule's program the batched call's primal
        // region, so a rule program that allocates local reference state is discharged as well. The primal and the
        // rule both compute `2x` through local reference state, which discharging the program discharges in the call.
        let primal = {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let x = builder.add_input(ArrayType::scalar(DataType::F32).into());
            let state = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![state, x], None)
                .unwrap();
            let output =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![output],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let definition = CustomRuleRegistration::new(
            IrDefinition::new("stateful")
                .with_batching_support()
                .with_batching_rule(|_, _, inputs, input_axes| {
                    let state = inputs[0].reference_new()?;
                    state.add_update(&inputs[0])?;
                    Ok((vec![state.read()?], input_axes.to_vec()))
                })
                .with_reference_discharge(),
        );
        let program = ir_custom_rule_program(&definition, 0, primal)
            .discharge_references(0)
            .unwrap()
            .into_program_without_external_references()
            .unwrap();
        let (batched, output_axes) = program
            .batched_with_threaded_extent(
                DimensionType::new("n", DimensionBounds::new(1, Some(8)).unwrap()),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert!(!batched.entry_region_ref().contains_references_in_closure());
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:dimension<n ∈ [1, 8)>, %1:f32[n] .
                let %2:f32[n] = composite [
                    name=\"stateful\",
                    non_differentiated_count=1,
                    batching=[(extent=dimension<n ∈ [1, 8)>, input_axes=[axis 0], output_axes=[axis 0])],
                    discharged=true,
                ] %0 %1 [
                    primal={
                        lambda %0:dimension<n ∈ [1, 8)>, %1:f32[n] .
                        let %2:f32[n] = add %1 %1
                        in (%2)
                    },
                ]
                in (%0, %2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_composite_batching_retained_rules_differentiation() {
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, true));
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        let batched = batched_program(&program, 3, &[BatchAxis::new(0)]);
        let x = Array::vector(vec![1f64, 2f64, 3f64]).unwrap();
        let ones = Array::vector(vec![1f64; 3]).unwrap();

        // Forward mode traces the rule once at the unbatched types and batches that trace, so the unbatched and batched
        // specializations share one rule invocation.
        let jvp = batched.jvp().unwrap();
        assert_eq!(counters.counts(), (1, 0, 0));
        assert_eq!(definition.caches().jvp_specializations.len(), 2);
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3] .
                let %2:f64[3] = mul %0 %0
                    %3:f64[3] = mul %2 %0
                    %4:f64[3] = mul %2 %1
                in (%3, %4)
            "}
            .trim_end(),
        );
        assert_eq!(
            jvp.interpret(vec![x.clone(), ones.clone()]),
            Ok(vec![Array::vector(vec![1f64, 8f64, 27f64]).unwrap(), Array::vector(vec![1f64, 4f64, 9f64]).unwrap()]),
        );

        // Reverse mode batches the forward rule, stages a batched carrier, and batches the backward specialization.
        let reverse = batched
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose, None)
            .unwrap();
        assert_eq!(
            reverse.tangent().to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3] .
                let %2:f64[3] = composite_transpose [
                    name=\"cube\",
                    leading_input_count=1,
                    batching=[(extent=3, input_axes=[axis 0, axis 0], output_axes=[axis 0])],
                ] %1 %0
                in (%2)
            "}
            .trim_end(),
        );
        let residuals = reverse.primal().interpret(vec![x]).unwrap();
        let pullback = reverse.tangent().transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3] .
                let %2:f64[3] = mul %1 %0
                    %3:f64[3] = add %2 %2
                in (%3)
            "}
            .trim_end(),
        );
        assert_eq!(
            pullback.interpret(vec![ones, residuals[1].clone()]),
            Ok(vec![Array::vector(vec![2f64, 8f64, 18f64]).unwrap()]),
        );
        assert_eq!(counters.counts(), (1, 1, 1));
    }

    #[test]
    fn test_composite_batching_retained_rules_deferred_forward() {
        // Reverse mode of a batched call that defers its forward rule replays the batched primal and stages a carrier
        // over the batched input, whose transposition batches the forward rule traced at the unbatched types to recover
        // the batched residual `x²`, from which the batched backward rule computes `2 x² ȳ`.
        let counters = Arc::new(RuleCounters::default());
        let definition =
            CustomRuleRegistration::new(cube_definition(&counters, false, true).with_deferred_forward_rule());
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        let batched = batched_program(&program, 3, &[BatchAxis::new(0)]);
        let x = Array::vector(vec![1f64, 2f64, 3f64]).unwrap();
        let ones = Array::vector(vec![1f64; 3]).unwrap();
        let reverse = batched
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose, None)
            .unwrap();
        assert_eq!(
            reverse.primal().to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[3] = mul %0 %0
                    %2:f64[3] = mul %1 %0
                in (%2, %0)
            "}
            .trim_end(),
        );
        assert_eq!(
            reverse.tangent().to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3] .
                let %2:f64[3] = composite_transpose [name=\"cube\", leading_input_count=1, deferred_forward=true] %1 %0
                in (%2)
            "}
            .trim_end(),
        );
        let residuals = reverse.primal().interpret(vec![x.clone()]).unwrap();
        let pullback = reverse.tangent().transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3] .
                let %2:f64[3] = mul %1 %1
                    %3:f64[3] = mul %2 %0
                    %4:f64[3] = add %3 %3
                in (%4)
            "}
            .trim_end(),
        );
        assert_eq!(
            pullback.interpret(vec![ones, residuals[1].clone()]),
            Ok(vec![Array::vector(vec![2f64, 8f64, 18f64]).unwrap()]),
        );
        assert_eq!(counters.counts(), (0, 1, 1));
    }

    #[test]
    fn test_composite_batching_retained_rules_output_layout() {
        // `f(c, p) = p` with the rule `ḟ = c ṗ`, where `c` is non-differentiated: batching a mapped `c` and a
        // replicated `p` leaves the primal replicated, while the derivative is mapped.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let primal = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            builder.add_input(scalar_type.clone());
            let parameter = builder.add_input(scalar_type.clone());
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![parameter], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let definition = TestDefinition::new("scaled_parameter")
            .with_jvp(|primals, tangents| {
                Ok((vec![primals[1].clone()], vec![primals[0].clone() * tangents[0].clone()]))
            })
            .with_batching_support();
        let program = |definition: TestDefinition| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let primal = builder.import_program(primal.clone());
            let coefficient = builder.add_input(scalar_type.clone());
            let parameter = builder.add_input(scalar_type.clone());
            let operation = CompositeOperation::new(CustomRuleRegistration::new(definition).reference())
                .with_non_differentiated_count(1)
                .unwrap();
            let output =
                builder.add_instruction(operation, vec![primal], vec![coefficient, parameter], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let coefficient = Array::vector(vec![2f64, 3f64, 5f64]).unwrap();
        let parameter = Array::scalar(7f64).unwrap();
        let parameter_tangent = Array::scalar(2f64).unwrap();

        // By default, the output is mapped (broadcasting the replicated primal), so the mapped derivative fits.
        let batched = batched_program(&program(definition), 3, &[BatchAxis::new(0), BatchAxis::replicated()]);
        let arguments = vec![coefficient.clone(), parameter.clone(), parameter_tangent.clone()];
        assert_eq!(
            batched.entry_region_ref().jvp(&[1]).unwrap().interpret(arguments.clone()),
            Ok(vec![Array::vector(vec![7f64; 3]).unwrap(), Array::vector(vec![4f64, 6f64, 10f64]).unwrap()]),
        );

        // A declared replicated output avoids the broadcast, but the mapped derivative violates it, which is reported
        // when the rule is first traced at that level rather than when the call is batched.
        let definition = TestDefinition::new("scaled_parameter")
            .with_jvp(|primals, tangents| {
                Ok((vec![primals[1].clone()], vec![primals[0].clone() * tangents[0].clone()]))
            })
            .with_batching_support()
            .with_batched_output_axes(vec![BatchAxis::replicated()]);
        let batched = batched_program(&program(definition), 3, &[BatchAxis::new(0), BatchAxis::replicated()]);
        assert_eq!(batched.interpret(vec![coefficient.clone(), parameter.clone()]), Ok(vec![parameter.clone()]));
        assert!(matches!(
            batched.entry_region_ref().jvp(&[1]),
            Err(DifferentiationError::Program(error))
                if error.downcast_custom::<BatchingError>() == Some(&BatchingError::InvalidBatchMetadata {
                    message: "a custom rule output is mapped along axis 0 but its batched layout requires it to be \
                              replicated"
                        .to_string(),
                }),
        ));

        // A declared replicated output is valid when the derivative agrees with it. Here, the second output is `p`
        // itself, whose tangent `ṗ` stays replicated, while the first output is mapped.
        let definition = TestDefinition::new("scaled_parameter_pair")
            .with_jvp(|primals, tangents| {
                let outputs = vec![primals[1].clone(), primals[1].clone()];
                Ok((outputs, vec![primals[0].clone() * tangents[0].clone(), tangents[0].clone()]))
            })
            .with_batching_support()
            .with_batched_output_axes(vec![BatchAxis::new(0), BatchAxis::replicated()]);
        let pair_primal = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            builder.add_input(scalar_type.clone());
            let parameter = builder.add_input(scalar_type.clone());
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![parameter, parameter], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let pair_program = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let primal = builder.import_program(pair_primal);
            let coefficient = builder.add_input(scalar_type.clone());
            let parameter = builder.add_input(scalar_type.clone());
            let operation = CompositeOperation::new(CustomRuleRegistration::new(definition).reference())
                .with_non_differentiated_count(1)
                .unwrap();
            let outputs = builder
                .add_instruction(operation, vec![primal], vec![coefficient, parameter], None)
                .unwrap()
                .to_vec();
            builder
                .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let batched = batched_program(&pair_program, 3, &[BatchAxis::new(0), BatchAxis::replicated()]);
        assert_eq!(
            batched.entry_region_ref().jvp(&[1]).unwrap().interpret(arguments.clone()),
            Ok(vec![
                Array::vector(vec![7f64; 3]).unwrap(),
                parameter.clone(),
                Array::vector(vec![4f64, 6f64, 10f64]).unwrap(),
                Array::scalar(2f64).unwrap(),
            ]),
        );

        // A declared replicated output whose primal is mapped is rejected when the call is batched.
        let definition = TestDefinition::new("scaled_parameter")
            .with_batching_support()
            .with_batched_output_axes(vec![BatchAxis::replicated()]);
        assert!(matches!(
            program(definition).batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            ),
            Err(BatchingError::MisalignedBatchAxes { message })
                if message == "batched region output axes [axis 0] do not match the target output axes [replicated]",
        ));
    }

    #[test]
    fn test_composite_batching_retained_rules_nested() {
        // Two batching levels reuse one trace of the rule, and each level has its own specialization.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, false));
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        let batched = batched_program(&batched_program(&program, 3, &[BatchAxis::new(0)]), 2, &[BatchAxis::new(0)]);
        let x = Array::matrix(2, 3, vec![1f64, 2f64, 3f64, 4f64, 5f64, 6f64]).unwrap();
        let ones = Array::matrix(2, 3, vec![1f64; 6]).unwrap();
        assert_eq!(
            batched.jvp().unwrap().interpret(vec![x, ones]),
            Ok(vec![
                Array::matrix(2, 3, vec![1f64, 8f64, 27f64, 64f64, 125f64, 216f64]).unwrap(),
                Array::matrix(2, 3, vec![1f64, 4f64, 9f64, 16f64, 25f64, 36f64]).unwrap(),
            ]),
        );
        assert_eq!(counters.counts(), (1, 0, 0));
        assert_eq!(
            definition
                .caches()
                .jvp_specializations
                .keys()
                .iter()
                .map(|key| key.levels.len())
                .collect::<Vec<_>>(),
            vec![2, 1, 0],
        );
    }

    #[test]
    fn test_composite_batching_retained_rules_without_batcher() {
        // Without batching support, a batched call still executes its primal, but cannot be differentiated.
        let definition = TestDefinition::new("cube").with_jvp(|primals, tangents| {
            let square = primals[0].clone() * primals[0].clone();
            Ok((vec![square.clone() * primals[0].clone()], vec![square * tangents[0].clone()]))
        });
        let definition = CustomRuleRegistration::new(definition);
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        let batched = batched_program(&program, 3, &[BatchAxis::new(0)]);
        assert_eq!(
            batched.interpret(vec![Array::vector(vec![1f64, 2f64, 3f64]).unwrap()]),
            Ok(vec![Array::vector(vec![1f64, 8f64, 27f64]).unwrap()]),
        );
        assert!(matches!(
            batched.jvp(),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`composite` `cube` cannot differentiate a batched call because its definition has no \
                               batching support (refer to `CustomRuleDefinition::with_batching_support`)",
        ));
    }

    #[test]
    fn test_composite_batching_retained_rules_renaming() {
        // Renaming a batched call renames its recorded unbatched types, so specializations use the renamed boundary.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, false));
        let original = DimensionVariable::new("original", DimensionBounds::new(2, Some(6)).unwrap());
        let relocated = DimensionVariable::new("relocated", DimensionBounds::new(2, Some(6)).unwrap());
        let original_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(original.clone())]));
        let relocated_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(relocated.clone())]));
        let batched = batched_program(&custom_rule_program(&definition, original_type), 3, &[BatchAxis::new(0)]);
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(original, relocated).unwrap();
        let renamed = batched.rename_type_identities(&renaming).unwrap();
        assert_eq!(
            renamed.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[3, relocated], %1:f64[3, relocated] .
                let %2:f64[3, relocated] = mul %0 %0
                    %3:f64[3, relocated] = mul %2 %0
                    %4:f64[3, relocated] = mul %2 %1
                in (%3, %4)
            "}
            .trim_end(),
        );
        assert!(
            definition
                .caches()
                .jvp_specializations
                .keys()
                .iter()
                .all(|key| key.input_types == vec![relocated_type.clone()]),
        );
    }

    #[test]
    fn test_composite_differentiation() {
        // The rule-selection contract: every layout of rules selects exactly the rule that each mode requires, and a
        // missing rule never falls back to differentiating the primal region. The rule regions of the sine fixture
        // double (JVP rule) or triple (reverse-mode rules) the true derivative `cos(x)`, which a derived JVP computes.
        let no_rule = "cannot differentiate a `composite` call that has no derivative rule";
        let (absent, explicit, primal) =
            (CompositeJvpRule::Absent, CompositeJvpRule::Explicit, CompositeJvpRule::Primal);
        let x = 0.5f64;
        for (jvp_rule, has_vjp_rule, tangent, gradient) in [
            (absent, false, Err(no_rule), Err(no_rule)),
            (primal, false, Ok(x.cos()), Ok(x.cos())),
            (explicit, false, Ok(2.0 * x.cos()), Ok(2.0 * x.cos())),
            (absent, true, Err(FORWARD_MODE_REJECTION), Ok(3.0 * x.cos())),
            (explicit, true, Ok(2.0 * x.cos()), Ok(3.0 * x.cos())),
            (primal, true, Ok(x.cos()), Ok(3.0 * x.cos())),
        ] {
            let operation = ArrayComposite::from_rule_regions(jvp_rule, has_vjp_rule);
            let regions = sin_regions(&operation);
            let result = differentiate_at(Array::scalar(x).unwrap()).jvp(Array::scalar(1f64).unwrap(), |x| {
                Ok(x.context()
                    .bind(ArrayOperation::Composite(operation.clone()), regions.clone(), &[x.clone()])?
                    .remove(0))
            });
            match tangent {
                Ok(tangent) => {
                    assert_eq!(result, Ok((Array::scalar(x.sin()).unwrap(), Array::scalar(tangent).unwrap())));
                }
                Err(expected) => assert!(matches!(
                    result,
                    Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                        if message == expected,
                )),
            }
            let result = differentiate_at(Array::scalar(x).unwrap())
                .vjp(|x| {
                    Ok(x.context()
                        .bind(ArrayOperation::Composite(operation.clone()), regions.clone(), &[x.clone()])?
                        .remove(0))
                })
                .and_then(|(value, pullback)| Ok((value, pullback.apply_materialized(Array::scalar(1f64).unwrap())?)));
            match gradient {
                Ok(gradient) => {
                    assert_eq!(result, Ok((Array::scalar(x.sin()).unwrap(), Array::scalar(gradient).unwrap())));
                }
                Err(expected) => assert!(matches!(
                    result,
                    Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                        if message == expected,
                )),
            }
        }

        // Higher-order derivatives compose with both a derived JVP, whose second derivative is `-sin(x)`, and with
        // forward differentiation of reverse-mode rules, whose tripled backward rule makes it `-3 sin(x)`.
        for (jvp_rule, has_vjp_rule, expected) in [(primal, false, -x.sin()), (explicit, true, -3.0 * x.sin())] {
            let operation = ArrayComposite::from_rule_regions(jvp_rule, has_vjp_rule);
            let regions = sin_regions(&operation);
            let (_, second_derivative) = differentiate_at(Array::scalar(x).unwrap())
                .jvp(Array::scalar(1f64).unwrap(), |x| {
                    Ok(differentiate_at(x)
                        .gradient(|y| {
                            let operation = ArrayOperation::Composite(operation.clone());
                            y.context().bind(operation, regions.clone(), &[y.clone()]).unwrap().remove(0)
                        })
                        .unwrap())
                })
                .unwrap();
            assert_eq!(second_derivative, Array::scalar(expected).unwrap());
        }
    }

    #[test]
    fn test_composite_differentiation_keeps_tangents_of_inactive_outputs_structural() {
        // The primal maps `(x, y)` to `(sin(x), sin(y))` and only `x` has a live tangent, so the second output depends
        // on no live tangent and keeps a structural-zero tangent, which lets linearization succeed.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let primal = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let inputs = (0..2).map(|_| builder.add_input(scalar_type.clone())).collect::<Vec<_>>();
            let outputs = inputs
                .iter()
                .map(|&input| builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0])
                .collect::<Vec<_>>();
            builder
                .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let operation = CompositeOperation::from_rule_regions(CompositeJvpRule::Primal, false);
        let program = composite_call_program(
            ArrayOperation::Composite(operation),
            vec![primal],
            vec![scalar_type.clone(), scalar_type],
        );
        let linearization = program.linearize_with_respect_to(&[0]).unwrap();
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![Array::scalar(0.7f64).unwrap(), Array::scalar(1.1f64).unwrap()])
            .unwrap();
        let residuals = primal_outputs.split_off(2);
        assert_eq!(primal_outputs, vec![Array::scalar(0.7f64.sin()).unwrap(), Array::scalar(1.1f64.sin()).unwrap()]);
        let mut tangent_inputs = vec![Array::scalar(2f64).unwrap()];
        tangent_inputs.extend(residuals);
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![Array::scalar(2.0 * 0.7f64.cos()).unwrap(), Array::scalar(0f64).unwrap()]),
        );
    }

    #[test]
    fn test_composite_differentiation_jvp_from_primal_linearization() {
        // Reusable linearization partitions a derived JVP like the primal region's own derivative: the primal program
        // saves `cos(x)` and the tangent program applies it.
        let operation = CompositeOperation::from_rule_regions(CompositeJvpRule::Primal, false);
        let program = composite_call_program(
            ArrayOperation::Composite(operation.clone()),
            sin_regions(&operation),
            vec![ArrayType::scalar(DataType::F64)],
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        let primals = linearization.primal().interpret(vec![Array::scalar(0.7f64).unwrap()]).unwrap();
        assert_eq!(primals, vec![Array::scalar(0.7f64.sin()).unwrap(), Array::scalar(0.7f64.cos()).unwrap()]);
        assert_eq!(
            linearization.tangent().interpret(vec![Array::scalar(2f64).unwrap(), primals[1].clone()]),
            Ok(vec![Array::scalar(2.0 * 0.7f64.cos()).unwrap()]),
        );
    }

    #[test]
    fn test_composite_differentiation_jvp_from_primal_non_differentiated_inputs() {
        // A derived JVP differentiates only with respect to the differentiated inputs, while the leading
        // non-differentiated input `p` of `f(p, x) = p * sin(x)` parameterizes the derivative `p * cos(x)`.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let primal = {
            let mut builder = ProgramBuilder::new();
            let parameter = builder.add_input(scalar_type.clone());
            let x = builder.add_input(scalar_type.clone());
            let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
            let output =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![parameter, sine], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let operation = ArrayOperation::Composite(
            CompositeOperation::from_rule_regions(CompositeJvpRule::Primal, false)
                .with_non_differentiated_count(1)
                .unwrap(),
        );
        let (value, tangent) = differentiate_at(Array::scalar(0.5f64).unwrap())
            .jvp(Array::scalar(1f64).unwrap(), |x| {
                let parameter = x.context().lift(Array::scalar(3f64).unwrap())?;
                Ok(x.context().bind(operation.clone(), vec![primal.clone()], &[parameter, x.clone()])?.remove(0))
            })
            .unwrap();
        assert_eq!(value, Array::scalar(3.0 * 0.5f64.sin()).unwrap());
        assert_eq!(tangent, Array::scalar(3.0 * 0.5f64.cos()).unwrap());
        let (value, gradient) = differentiate_at(Array::scalar(0.5f64).unwrap())
            .value_and_gradient(|x| {
                let parameter = x.context().lift(Array::scalar(3f64).unwrap()).unwrap();
                x.context()
                    .bind(operation.clone(), vec![primal.clone()], &[parameter, x.clone()])
                    .unwrap()
                    .remove(0)
            })
            .unwrap();
        assert_eq!(value, Array::scalar(3.0 * 0.5f64.sin()).unwrap());
        assert_eq!(gradient, Array::scalar(3.0 * 0.5f64.cos()).unwrap());
    }

    #[test]
    fn test_composite_differentiation_jvp_rule() {
        // The custom rule doubles the true derivative, which proves that it governs forward-mode differentiation.
        let (primal, tangent) = differentiate_at(Array::scalar(2.0).unwrap())
            .jvp(Array::scalar(1.0).unwrap(), |x| {
                let scalar_type = ArrayType::scalar(DataType::F64);
                let operation =
                    ArrayOperation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, false));
                let regions = vec![sin_program(&scalar_type), doubled_sin_jvp_program(&scalar_type)];
                Ok(x.context().bind(operation, regions, &[x.clone()])?.remove(0))
            })
            .unwrap();
        assert_eq!(primal, Array::scalar(2.0f64.sin()).unwrap());
        assert_eq!(tangent, Array::scalar(2.0 * 2.0f64.cos()).unwrap());

        // Reverse mode transposes the linearized custom rule, so the doubled derivative carries over.
        let (value, gradient) = differentiate_at(Array::scalar(3.0).unwrap())
            .value_and_gradient(|x| {
                let scalar_type = ArrayType::scalar(DataType::F64);
                let operation =
                    ArrayOperation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, false));
                let regions = vec![sin_program(&scalar_type), doubled_sin_jvp_program(&scalar_type)];
                x.context().bind(operation, regions, &[x.clone()]).unwrap().remove(0)
            })
            .unwrap();
        assert_eq!(value, Array::scalar(3.0f64.sin()).unwrap());
        assert_eq!(gradient, Array::scalar(2.0 * 3.0f64.cos()).unwrap());
    }

    #[test]
    fn test_composite_differentiation_jvp_rule_second_order() {
        // The JVP rule replays the user program as plain primitive operations, so the gradient program that it
        // produces is itself differentiable. The doubled rule makes the first derivative `2 cos(x)`, and so the second
        // derivative is `-2 sin(x)`.
        let (gradient, second_derivative) = differentiate_at(Array::scalar(0.7).unwrap())
            .value_and_gradient(|x| {
                differentiate_at(x)
                    .gradient(|y| {
                        let scalar_type = ArrayType::scalar(DataType::F64);
                        let operation = ArrayOperation::Composite(CompositeOperation::from_rule_regions(
                            CompositeJvpRule::Explicit,
                            false,
                        ));
                        let regions = vec![sin_program(&scalar_type), doubled_sin_jvp_program(&scalar_type)];
                        y.context().bind(operation, regions, &[y.clone()]).unwrap().remove(0)
                    })
                    .unwrap()
            })
            .unwrap();
        assert_eq!(gradient, Array::scalar(2.0 * 0.7f64.cos()).unwrap());
        assert_eq!(second_derivative, Array::scalar(-2.0 * 0.7f64.sin()).unwrap());
    }

    #[test]
    fn test_composite_differentiation_jvp_rule_forward_over_reverse() {
        // Forward differentiation of the gradient differentiates the ordinary operations that the JVP rule replayed
        // into the pullback, so the doubled rule yields the second derivative `-2 sin(x)`.
        let (gradient, second_derivative) = differentiate_at(Array::scalar(0.7).unwrap())
            .jvp(Array::scalar(1.0).unwrap(), |x| {
                Ok(differentiate_at(x)
                    .gradient(|y| {
                        let scalar_type = ArrayType::scalar(DataType::F64);
                        let operation = ArrayOperation::Composite(CompositeOperation::from_rule_regions(
                            CompositeJvpRule::Explicit,
                            false,
                        ));
                        let regions = vec![sin_program(&scalar_type), doubled_sin_jvp_program(&scalar_type)];
                        y.context().bind(operation, regions, &[y.clone()]).unwrap().remove(0)
                    })
                    .unwrap())
            })
            .unwrap();
        assert_eq!(gradient, Array::scalar(2.0 * 0.7f64.cos()).unwrap());
        assert_eq!(second_derivative, Array::scalar(-2.0 * 0.7f64.sin()).unwrap());
    }

    #[test]
    fn test_composite_differentiation_jvp_rule_reverse_over_forward() {
        // Reverse differentiation of the directional derivative differentiates the rule's replayed coefficient.
        let (tangent, second_derivative) = differentiate_at(Array::scalar(0.7).unwrap())
            .value_and_gradient(|x| {
                let direction = x.context().lift(Array::scalar(1.0).unwrap()).unwrap();
                differentiate_at(x)
                    .jvp(direction, |y| {
                        let scalar_type = ArrayType::scalar(DataType::F64);
                        let operation = ArrayOperation::Composite(CompositeOperation::from_rule_regions(
                            CompositeJvpRule::Explicit,
                            false,
                        ));
                        let regions = vec![sin_program(&scalar_type), doubled_sin_jvp_program(&scalar_type)];
                        Ok(y.context().bind(operation, regions, &[y.clone()])?.remove(0))
                    })
                    .unwrap()
                    .1
            })
            .unwrap();
        assert_eq!(tangent, Array::scalar(2.0 * 0.7f64.cos()).unwrap());
        assert_eq!(second_derivative, Array::scalar(-2.0 * 0.7f64.sin()).unwrap());
    }

    #[test]
    fn test_composite_differentiation_jvp_rule_forward_over_forward() {
        let (tangent, second_derivative) = differentiate_at(Array::scalar(0.7).unwrap())
            .jvp(Array::scalar(1.0).unwrap(), |x| {
                let direction = x.context().lift(Array::scalar(1.0).unwrap())?;
                Ok(differentiate_at(x)
                    .jvp(direction, |y| {
                        let scalar_type = ArrayType::scalar(DataType::F64);
                        let operation = ArrayOperation::Composite(CompositeOperation::from_rule_regions(
                            CompositeJvpRule::Explicit,
                            false,
                        ));
                        let regions = vec![sin_program(&scalar_type), doubled_sin_jvp_program(&scalar_type)];
                        Ok(y.context().bind(operation, regions, &[y.clone()])?.remove(0))
                    })
                    .unwrap()
                    .1)
            })
            .unwrap();
        assert_eq!(tangent, Array::scalar(2.0 * 0.7f64.cos()).unwrap());
        assert_eq!(second_derivative, Array::scalar(-2.0 * 0.7f64.sin()).unwrap());
    }

    #[test]
    fn test_composite_differentiation_jvp_rule_pullback() {
        // The pullback of a JVP-only rule is linear in its seed, so its derivative with respect to the seed is the
        // custom coefficient `2 cos(x)`.
        let (cotangent, seed_derivative) = differentiate_at(Array::scalar(3.0).unwrap())
            .jvp(Array::scalar(1.0).unwrap(), |seed| {
                let input = seed.context().lift(Array::scalar(0.7).unwrap())?;
                let (_, pullback) = differentiate_at(input)
                    .vjp(|y| {
                        let scalar_type = ArrayType::scalar(DataType::F64);
                        let operation = ArrayOperation::Composite(CompositeOperation::from_rule_regions(
                            CompositeJvpRule::Explicit,
                            false,
                        ));
                        let regions = vec![sin_program(&scalar_type), doubled_sin_jvp_program(&scalar_type)];
                        Ok(y.context().bind(operation, regions, &[y.clone()])?.remove(0))
                    })
                    .unwrap();
                pullback.apply_materialized(seed)
            })
            .unwrap();
        assert_eq!(cotangent, Array::scalar(2.0 * 0.7f64.cos() * 3.0).unwrap());
        assert_eq!(seed_derivative, Array::scalar(2.0 * 0.7f64.cos()).unwrap());

        // Transposing the pullback again transposes its ordinary linear operations and restores the tangent map.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let primal = builder.import_program(sin_program(&scalar_type));
        let jvp = builder.import_program(doubled_sin_jvp_program(&scalar_type));
        let input = builder.add_input(scalar_type);
        let output = builder
            .add_instruction(
                CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, false),
                vec![primal, jvp],
                vec![input],
                None,
            )
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let linearization = program.linearize().unwrap();
        let residuals = linearization.primal().interpret(vec![Array::scalar(0.7).unwrap()]).unwrap();
        let pullback = linearization.tangent().transpose_with_respect_to(&[0], &[]).unwrap();
        let retransposed = pullback.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            retransposed.interpret(vec![Array::scalar(1.0).unwrap(), residuals[1].clone()]),
            linearization.tangent().interpret(vec![Array::scalar(1.0).unwrap(), residuals[1].clone()]),
        );
    }

    #[test]
    fn test_composite_differentiation_jvp_rule_local_reference_state() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let regions = custom_jvp_regions_with_reference_state(&scalar_type);

        // A custom derivative rule may allocate and use local reference state: the rule is replayed directly when it
        // consumes the active input, so its state executes like any other primitive operation of the identity rule.
        assert_eq!(
            differentiate_at(ArrayIrValue::Array(Array::scalar(1.0f32).unwrap())).jvp(
                ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
                |input| {
                    let operation = ArrayIrOperation::Composite(CompositeOperation::from_rule_regions(
                        CompositeJvpRule::Explicit,
                        false,
                    ));
                    Ok(input.context().bind(operation, regions.clone(), std::slice::from_ref(&input))?.remove(0))
                },
            ),
            Ok((
                ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
            )),
        );

        // A lifted input has a structural zero tangent. An assertion in the rule makes replay observable:
        // returning the primal with a zero tangent without running the rule would incorrectly succeed.
        let mut rule = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = rule.add_input(scalar_type.clone());
        let tangent = rule.add_input(scalar_type);
        let predicate = rule.add_constant(ArrayIrValue::Array(Array::scalar(false).unwrap()));
        rule.add_instruction(AssertOperation::new("zero-tangent rule was replayed"), Vec::new(), vec![predicate], None)
            .unwrap();
        let outputs = rule.splice_program(&regions[1], &[input, tangent]).unwrap();
        let rule = rule.build(outputs, vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();
        let error = differentiate_at(ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()))
            .jvp(ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()), |input| {
                let lifted = input.context().lift(ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()))?;
                let operation = ArrayIrOperation::Composite(CompositeOperation::from_rule_regions(
                    CompositeJvpRule::Explicit,
                    false,
                ));
                Ok(input.context().bind(operation, vec![regions[0].clone(), rule], &[lifted])?.remove(0))
            })
            .unwrap_err();
        let DifferentiationError::Program(error) = error else {
            panic!("expected a program error, got {error:?}");
        };
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "zero-tangent rule was replayed".to_owned(),
                observations: vec![]
            }),
        );
    }

    #[test]
    fn test_composite_differentiation_jvp_rule_staged_local_reference_state() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let program = composite_call_program(
            ArrayIrOperation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, false)),
            custom_jvp_regions_with_reference_state(&scalar_type),
            vec![scalar_type],
        );

        // The entry region is pure because the state lives in the dormant rule region. Forward mode replays that rule
        // when it fires on the live tangent, so the fused program stages the rule's local allocation and read, which
        // execute like any other primitive operations of the identity rule.
        assert!(program.effects().classes().is_empty());
        let jvp = program.jvp().unwrap();
        assert!(jvp.entry_region_ref().contains_effect_in_closure(EffectClass::OrderedState));
        let inputs = vec![
            ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
            ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()),
        ];
        assert_eq!(jvp.interpret(inputs.clone()), Ok(inputs));
    }

    #[test]
    fn test_composite_differentiation_jvp_rule_nested_local_reference_state() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let input = DifferentiationDual::new(
            ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
            ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
        )
        .unwrap();

        // A rule region may allocate and use local reference state inside a dormant nested rule: the rule is replayed
        // directly (the driver makes recursive differentiation an assertion failure), so its state executes like any
        // other primitive operation and the identity rule yields the identity dual.
        let jvp = nested_composite_state_program(&scalar_type, true);
        assert!(jvp.entry_region_ref().contains_effect_in_closure(EffectClass::OrderedState));
        let driver =
            ReferenceRuleDifferentiationDriver { programs: vec![array_ir_identity_program(&scalar_type), jvp] };
        let outputs = ArrayIrComposite::from_rule_regions(CompositeJvpRule::Explicit, false)
            .jvp(&DifferentiationContext::fused(EagerArrayIrContext::new()), &driver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()));
        assert!(matches!(
            outputs[0].tangent(),
            MaybeZero::Value(ArrayIrValue::Array(tangent)) if tangent == &Array::scalar(1.0f32).unwrap(),
        ));
    }

    #[test]
    fn test_composite_differentiation_jvp_rule_plumbing_references() {
        let context = DifferentiationContext::fused(EagerArrayIrContext::new());
        let driver = ReferenceRuleDifferentiationDriver { programs: counting_custom_jvp_regions(1) };

        // The plumbing counter reaches the replayed rule as the same reference and is mutated by it. Its live tangent
        // reference is left untouched, because the rule declares no derivative through the state it denotes.
        let counter = ArrayReference::new(Array::scalar(0.0f32).unwrap());
        let counter_tangent = ArrayReference::new(Array::scalar(0.0f32).unwrap());
        let inputs = [
            DifferentiationDual::new(
                ArrayIrValue::Reference(counter.clone()),
                ArrayIrValue::Reference(counter_tangent.clone()),
            )
            .unwrap(),
            DifferentiationDual::new(
                ArrayIrValue::Array(Array::scalar(3.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
            )
            .unwrap(),
        ];
        let outputs = ArrayIrComposite::from_rule_regions(CompositeJvpRule::Explicit, false)
            .with_non_differentiated_count(1)
            .unwrap()
            .jvp(&context, &driver, &inputs)
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &ArrayIrValue::Array(Array::scalar(3.0f32).unwrap()));
        assert!(matches!(
            outputs[0].tangent(),
            MaybeZero::Value(ArrayIrValue::Array(tangent)) if tangent == &Array::scalar(1.0f32).unwrap(),
        ));
        assert_eq!(counter.read(), Ok(Array::scalar(3.0f32).unwrap()));
        assert_eq!(counter_tangent.read(), Ok(Array::scalar(0.0f32).unwrap()));

        // A reference input that is also bound at another input position is rejected before either rule region runs.
        let aliased = [
            inputs[0].clone(),
            DifferentiationDual::new_with_zero_tangent(ArrayIrValue::Reference(counter.clone())).unwrap(),
            inputs[1].clone(),
        ];
        let aliasing_driver = ReferenceRuleDifferentiationDriver { programs: counting_custom_jvp_regions(2) };
        assert!(matches!(
            ArrayIrComposite::from_rule_regions(CompositeJvpRule::Explicit, false)
                .with_non_differentiated_count(2)
                .unwrap()
                .jvp(&context, &aliasing_driver, &aliased),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "input 1 and input 0 bind the same reference allocation",
        ));
        assert_eq!(counter.read(), Ok(Array::scalar(3.0f32).unwrap()));

        // The same reference in the differentiated segment is rejected by the replayed rule as well.
        assert!(matches!(
            ArrayIrComposite::from_rule_regions(CompositeJvpRule::Explicit, false)
                .jvp(&context, &driver, &inputs),
            Err(DifferentiationError::Program(ProgramError::Type(error)))
                if error == TypeError::invalid(
                    "`composite` accepts reference inputs only in its leading non-differentiated segment; \
                     move input 0 of type `ref<f32[]>` before the differentiated inputs"
                        .to_string(),
                ),
        ));
    }

    #[test]
    fn test_composite_differentiation_jvp_rule_dormant_assertions() {
        let scalar_type = ArrayType::scalar(DataType::F32);
        let mut primal = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = primal.add_input(scalar_type.clone());
        let primal = primal.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut rule = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = rule.add_input(scalar_type.clone());
        let tangent = rule.add_input(scalar_type.clone());
        let zero = rule.add_constant(Array::scalar(0.0f32).unwrap());
        let predicate = rule
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                Vec::new(),
                vec![input, zero],
                None,
            )
            .unwrap()[0];
        rule.add_instruction(
            AssertOperation::new("custom derivative requires positive input").with_labels(vec!["input".to_owned()]),
            Vec::new(),
            vec![predicate, input],
            None,
        )
        .unwrap();
        let rule = rule
            .build::<Vec<Array>, Vec<Array>>(vec![input, tangent], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let program = composite_call_program(
            CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, false),
            vec![primal, rule],
            vec![scalar_type],
        );

        // Dormant derivative rules contribute no execution effects to the primal call and do not run when it is
        // interpreted.
        assert_eq!(program.effects().classes(), EffectClasses::NONE);
        assert_eq!(program.interpret(vec![Array::scalar(-1.0f32).unwrap()]), Ok(vec![Array::scalar(-1.0f32).unwrap()]));

        // Differentiation replays the rule, which stages its assertion into the differentiated program.
        let differentiated = program.jvp().unwrap();
        assert_eq!(differentiated.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        assert_eq!(
            differentiated.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[] = const 0.0
                    %3:bool[] = compare [direction=GreaterThan] %0 %2
                    () = assert [message=\"custom derivative requires positive input\", labels=[\"input\"]] %3 %0
                in (%0, %1)
            "}
            .trim_end(),
        );
        assert_eq!(
            differentiated.interpret(vec![Array::scalar(1.0f32).unwrap(), Array::scalar(2.0f32).unwrap()]),
            Ok(vec![Array::scalar(1.0f32).unwrap(), Array::scalar(2.0f32).unwrap()]),
        );
        let error = differentiated
            .interpret(vec![Array::scalar(-1.0f32).unwrap(), Array::scalar(2.0f32).unwrap()])
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "custom derivative requires positive input".to_owned(),
                observations: vec![("input".to_owned(), "-1".to_owned())],
            }),
        );
    }

    #[test]
    fn test_composite_differentiation_jvp_rule_linearization_fresh_tangent_state() {
        // Repeated calls must neither reuse accumulated state nor access an accumulator consumed by an earlier call.
        check_custom_jvp_linearization_fresh_tangent_state(false, false);
        check_custom_jvp_linearization_fresh_tangent_state(true, false);
    }

    #[test]
    fn test_composite_differentiation_jvp_rule_linearization_stateful_scan() {
        let vector: ArrayIrType = ArrayType::new_static(DataType::F32, [3]).into();
        let regions = stateful_square_regions(false)
            .into_iter()
            .map(|body| {
                // Inline the existing rule into the explicit scan body, retaining its instructions and effects.
                let mut indexed_body = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
                indexed_body.add_input(ArrayType::scalar(DataType::I64).into());
                let body_inputs =
                    body.input_types().iter().map(|r#type| indexed_body.add_input(r#type.clone())).collect::<Vec<_>>();
                let outputs = indexed_body.splice_program(&body, &body_inputs).unwrap();
                let output_count = outputs.len();
                let body = indexed_body
                    .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                        outputs,
                        vec![Placeholder; 1 + body_inputs.len()],
                        vec![Placeholder; output_count],
                    )
                    .unwrap();
                composite_call_program(ScanOperation::new(0, 3), vec![body], vec![vector.clone(); body_inputs.len()])
            })
            .collect::<Vec<_>>();
        let program = composite_call_program(
            ArrayIrComposite::from_rule_regions(CompositeJvpRule::Explicit, false),
            regions,
            vec![vector],
        );

        // The fused rule is partitioned inside the scan: the known coefficient becomes a stacked residual, while the
        // tangent scan keeps the accumulator lifecycle.
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f32[3] .
                let %1:f32[3], %2:f32[3] = scan [carry_count=0, iteration_count=3, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:f32[] .
                        let %2:f32[] = mul %1 %1
                            %3:f32[] = add %1 %1
                        in (%2, %3)
                    },
                ]
                in (%1, %2)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[3], %1:f32[3] .
                let %2:f32[3] = scan [carry_count=0, iteration_count=3, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f32[], %2:f32[] .
                        let %3:f32[] = const 0.0
                            %4:ref<f32[]> = reference_new %3
                            %5:f32[] = mul %2 %1
                            () = reference_add_update %4 %5
                            %6:f32[] = reference_read %4
                        in (%6)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        let primals = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap())])
            .unwrap();
        assert_eq!(
            primals,
            vec![
                ArrayIrValue::Array(Array::vector(vec![1.0f32, 4.0, 9.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![2.0f32, 4.0, 6.0]).unwrap()),
            ],
        );
        assert_eq!(
            linearization
                .tangent()
                .interpret(vec![ArrayIrValue::Array(Array::vector(vec![2.0f32; 3]).unwrap()), primals[1].clone()]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![4.0f32, 8.0, 12.0]).unwrap())]),
        );
        assert_eq!(
            linearization
                .tangent()
                .interpret(vec![ArrayIrValue::Array(Array::vector(vec![5.0f32; 3]).unwrap()), primals[1].clone()]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![10.0f32, 20.0, 30.0]).unwrap())]),
        );
        assert_eq!(
            linearization
                .tangent()
                .interpret(vec![ArrayIrValue::Array(Array::vector(vec![2.0f32; 3]).unwrap()), primals[1].clone()]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![4.0f32, 8.0, 12.0]).unwrap())]),
        );
    }

    #[test]
    fn test_composite_differentiation_jvp_rule_linearization_nested_aliased_reference_carries() {
        // Exercise whole-root carry updates and indexed updates of a stacked root independently.
        check_custom_jvp_linearization_aliased_reference_carries(false);
        check_custom_jvp_linearization_aliased_reference_carries(true);
    }

    #[test]
    fn test_composite_differentiation_jvp_rule_zero_tangent_outputs() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(scalar_type.clone());
        builder.add_input(scalar_type.clone());
        let zero = builder.add_constant(Array::scalar(0.0).unwrap());
        let rule = builder.build(vec![input, zero], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();
        let mut primal = ProgramBuilder::new();
        let input = primal.add_input(scalar_type.clone());
        let primal = primal.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let operation =
            ArrayOperation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, false));
        let regions = vec![primal, rule];
        let program = composite_call_program(operation.clone(), regions.clone(), vec![scalar_type]);

        // A constant zero is a valid linear tangent map. Program linearization must preserve its output arity even
        // though it needs neither the primal input nor any tangent input to produce the result.
        let linearization = program.linearize().unwrap();
        let inputs = vec![Array::scalar(7.0).unwrap(); linearization.tangent().input_ids().len()];
        assert_eq!(linearization.residual_count(), 0);
        assert_eq!(linearization.tangent().interpret(inputs), Ok(vec![Array::scalar(0.0).unwrap()]));

        // Value linearization and reverse mode use the same custom rule boundary.
        let (_, pushforward) = differentiate_at(Array::scalar(2.0).unwrap())
            .linearize(|input| {
                Ok(input.context().bind(operation.clone(), regions.clone(), &[input.clone()])?.remove(0))
            })
            .unwrap();
        assert_eq!(pushforward.apply(Array::scalar(7.0).unwrap()), Ok(Array::scalar(0.0).unwrap()));
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap())
                .gradient(|input| { input.context().bind(operation, regions, &[input.clone()]).unwrap().remove(0) }),
            Ok(Array::scalar(0.0).unwrap()),
        );
    }

    #[test]
    fn test_composite_differentiation_jvp_rule_rejects_known_tangent_outputs() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let operation =
            ArrayOperation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, false));

        // A constant non-zero tangent ignores the input tangent and is not a linear map.
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(scalar_type.clone());
        builder.add_input(scalar_type.clone());
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let tangent = builder.add_constant(Array::scalar(1.0).unwrap());
        let rule = builder.build(vec![output, tangent], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();
        let regions = vec![sin_program(&scalar_type), rule];
        let expected = "linearization produced a known tangent output; differentiation rules must represent \
                        input-independent zero tangents structurally";

        // Program-level linearization rejects the malformed rule rather than silently replacing its constant tangent
        // with zero.
        let program = composite_call_program(operation.clone(), regions.clone(), vec![scalar_type.clone()]);
        assert!(matches!(
            program.linearize(),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message))) if message == expected,
        ));

        // Value-level linearization enforces the same rule contract before exposing a reusable pushforward.
        assert!(matches!(
            differentiate_at(Array::scalar(2.0).unwrap())
                .linearize(|input| Ok(input.context().bind(operation, regions, &[input.clone()])?.remove(0))),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message))) if message == expected,
        ));
    }

    #[test]
    fn test_composite_differentiation_combined_rules_fresh_tangent_state() {
        // Adding reverse-mode rules does not change forward-mode linearization: the tangent program still allocates
        // fresh JVP rule state on every application, exactly as for a call that has only a JVP rule.
        check_custom_jvp_linearization_fresh_tangent_state(false, true);
        check_custom_jvp_linearization_fresh_tangent_state(true, true);
    }

    #[test]
    fn test_composite_differentiation_combined_rules_repeated_pullback_effects() {
        // A call over a plumbing counter with a pure JVP rule and an effectful backward rule. Forward mode uses only
        // the JVP rule, so it never executes the backward rule, while every application of a pullback executes the
        // backward rule, and with it its effect, exactly once.
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let mut primal = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        primal.add_input(reference_type.clone());
        let x = primal.add_input(scalar_type.clone());
        let square = primal
            .add_instruction(ArrayOperation::Mul(MulOperation::new()), Vec::new(), vec![x, x], None)
            .unwrap()[0];
        let primal = primal.build(vec![square], vec![Placeholder; 2], vec![Placeholder]).unwrap();
        let mut jvp = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        jvp.add_input(reference_type);
        let x = jvp.add_input(scalar_type.clone());
        let tangent = jvp.add_input(scalar_type);
        let square =
            jvp.add_instruction(ArrayOperation::Mul(MulOperation::new()), Vec::new(), vec![x, x], None).unwrap()[0];
        let coefficient = jvp.add_instruction(AddOperation::new(), Vec::new(), vec![x, x], None).unwrap()[0];
        let output_tangent = jvp
            .add_instruction(ArrayOperation::Mul(MulOperation::new()), Vec::new(), vec![coefficient, tangent], None)
            .unwrap()[0];
        let jvp = jvp.build(vec![square, output_tangent], vec![Placeholder; 3], vec![Placeholder; 2]).unwrap();
        let mut regions = vec![primal, jvp];
        regions.extend(counting_square_vjp_regions());
        let operation = ArrayIrOperation::Composite(
            CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, true)
                .with_non_differentiated_count(1)
                .unwrap(),
        );
        let counter = ArrayReference::new(Array::scalar(0.0f32).unwrap());
        let counter_tangent = ArrayReference::new(Array::scalar(0.0f32).unwrap());
        let inputs = (ArrayIrValue::Reference(counter.clone()), ArrayIrValue::Array(Array::scalar(3.0f32).unwrap()));
        let tangents =
            (ArrayIrValue::Reference(counter_tangent.clone()), ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()));
        assert_eq!(
            differentiate_at(inputs.clone()).jvp(tangents, |(counter, x)| {
                Ok(x.context().bind(operation.clone(), regions.clone(), &[counter, x.clone()])?.remove(0))
            }),
            Ok((
                ArrayIrValue::Array(Array::scalar(9.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(6.0f32).unwrap()),
            )),
        );
        assert_eq!(counter.read(), Ok(Array::scalar(0.0f32).unwrap()));
        let (value, pullback) = differentiate_at(inputs)
            .vjp(|(counter, x)| {
                Ok(x.context().bind(operation.clone(), regions.clone(), &[counter, x.clone()])?.remove(0))
            })
            .unwrap();
        assert_eq!(value, ArrayIrValue::Array(Array::scalar(9.0f32).unwrap()));
        assert_eq!(counter.read(), Ok(Array::scalar(0.0f32).unwrap()));
        for application in 1..=3 {
            assert_eq!(
                pullback.apply_with_destinations(
                    CotangentSeed::Value(ArrayIrValue::Array(Array::scalar(1.0f32).unwrap())),
                    (CotangentDestination::Ignore, CotangentDestination::Return),
                ),
                Ok((None, Some(MaybeZero::Value(ArrayIrValue::Array(Array::scalar(6.0f32).unwrap()))))),
            );
            assert_eq!(counter.read(), Ok(Array::scalar(application as f32).unwrap()));
        }
    }

    #[test]
    fn test_composite_differentiation_vjp_rule() {
        // The custom backward rule triples the true gradient, which proves that it governs reverse-mode
        // differentiation.
        let (value, gradient) = differentiate_at(Array::scalar(2.0).unwrap())
            .value_and_gradient(|x| {
                let scalar_type = ArrayType::scalar(DataType::F64);
                let operation =
                    ArrayOperation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Absent, true));
                let regions = vec![
                    sin_program(&scalar_type),
                    sin_forward_program(&scalar_type),
                    tripled_sin_backward_program(&scalar_type),
                ];
                x.context().bind(operation, regions, &[x.clone()]).unwrap().remove(0)
            })
            .unwrap();
        assert_eq!(value, Array::scalar(2.0f64.sin()).unwrap());
        assert_eq!(gradient, Array::scalar(3.0 * 2.0f64.cos()).unwrap());
    }

    #[test]
    fn test_composite_differentiation_vjp_rule_second_order() {
        // The backward program is inlined into the pullback. Its tripled cosine is differentiable with respect to
        // the primal through the saved residual, giving -3 sin(x), while preserving the custom first derivative.
        let (gradient, second_derivative) = differentiate_at(Array::scalar(0.7).unwrap())
            .value_and_gradient(|input| {
                differentiate_at(input)
                    .gradient(|input| {
                        let scalar_type = ArrayType::scalar(DataType::F64);
                        input
                            .context()
                            .bind(
                                ArrayOperation::Composite(CompositeOperation::from_rule_regions(
                                    CompositeJvpRule::Absent,
                                    true,
                                )),
                                vec![
                                    sin_program(&scalar_type),
                                    sin_forward_program(&scalar_type),
                                    tripled_sin_backward_program(&scalar_type),
                                ],
                                &[input.clone()],
                            )
                            .unwrap()
                            .remove(0)
                    })
                    .unwrap()
            })
            .unwrap();
        assert_eq!(gradient, Array::scalar(3.0 * 0.7f64.cos()).unwrap());
        assert_eq!(second_derivative, Array::scalar(-3.0 * 0.7f64.sin()).unwrap());

        // Forward differentiation of the resulting gradient also runs the backward program's ordinary operations;
        // this does not require executing the original call's tangent carrier.
        let (gradient, second_derivative) = differentiate_at(Array::scalar(0.7).unwrap())
            .jvp(Array::scalar(1.0).unwrap(), |input| {
                Ok(differentiate_at(input)
                    .gradient(|input| {
                        let scalar_type = ArrayType::scalar(DataType::F64);
                        input
                            .context()
                            .bind(
                                ArrayOperation::Composite(CompositeOperation::from_rule_regions(
                                    CompositeJvpRule::Absent,
                                    true,
                                )),
                                vec![
                                    sin_program(&scalar_type),
                                    sin_forward_program(&scalar_type),
                                    tripled_sin_backward_program(&scalar_type),
                                ],
                                &[input.clone()],
                            )
                            .unwrap()
                            .remove(0)
                    })
                    .unwrap())
            })
            .unwrap();
        assert_eq!(gradient, Array::scalar(3.0 * 0.7f64.cos()).unwrap());
        assert_eq!(second_derivative, Array::scalar(-3.0 * 0.7f64.sin()).unwrap());
    }

    #[test]
    fn test_composite_differentiation_vjp_rule_dead_output_extents() {
        // The attached identity rules of `(x: f32[n], y: f32[m]) ↦ (x, y)` save no residuals, so when the call's output
        // `x` is dead, only the seed extents that the carrier captures from the primal outputs name the extent `n` of
        // its zero seed. The backward rule never receives those extents.
        type Builder = ProgramBuilder<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let types = ["n", "m"].map(|name| {
            let dimension = DimensionVariable::new(name, DimensionBounds::new(1, Some(8)).unwrap());
            ArrayIrType::from(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(dimension)])))
        });

        // The identity rule optionally asserts that `1` is a valid extent, which gives it an effect.
        let identity = |effectful: bool| {
            let mut builder = Builder::new();
            let inputs = types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
            if effectful {
                let extent = builder.add_constant(Array::scalar(1i32).unwrap().into());
                let variable = DimensionVariable::new("extent", DimensionBounds::new(0, None).unwrap());
                builder
                    .add_instruction(DimensionFromScalarOperation::new(variable.into()), Vec::new(), vec![extent], None)
                    .unwrap();
            }
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    inputs,
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };

        // The program returns the call's output `y`, or its own input `x` when every output of the call is dead.
        let program = |uses_call: bool| {
            let mut builder = Builder::new();
            let primal = builder.import_program(identity(false));
            let forward = builder.import_program(identity(false));
            let backward = builder.import_program(identity(!uses_call));
            let inputs = types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
            let outputs = builder
                .add_instruction(
                    ArrayIrComposite::from_rule_regions(CompositeJvpRule::Absent, true),
                    vec![primal, forward, backward],
                    inputs.clone(),
                    None,
                )
                .unwrap()
                .to_vec();
            let output = if uses_call { outputs[1] } else { inputs[0] };
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![output],
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let linearize = |uses_call: bool| {
            let linearization = program(uses_call)
                .entry_region_ref()
                .linearize_shared_for_rule(&[0, 1], DifferentiationRule::JvpForTranspose, None)
                .unwrap();
            let pullback = linearization.tangent().transpose_with_respect_to(&[0, 1], &[]).unwrap();
            (linearization.tangent().to_string(), pullback)
        };
        let extent = |name: &str, extent: usize| {
            let variable = DimensionVariable::new(name, DimensionBounds::new(1, Some(8)).unwrap());
            ArrayIrValue::Dimension(DimensionValue::new(DimensionType::from(variable), extent).unwrap())
        };

        let (tangent, used) = linearize(true);
        assert_eq!(
            tangent,
            indoc! {"
                lambda %0:f32[n], %1:f32[m], %2:dimension<n ∈ [1, 8)>, %3:dimension<m ∈ [1, 8)> .
                let %4:f32[n], %5:f32[m] = composite_transpose [leading_input_count=2, seed_extent_count=2] %2 %3 %0 %1 [
                    backward={
                        lambda %0:f32[n], %1:f32[m] .
                        in (%0, %1)
                    },
                ]
                in (%5)
            "}
            .trim_end(),
        );
        // The carrier materializes the zero seed of `x` for the backward rule, which forwards it. The cotangent of `x`
        // therefore remains a structural zero, which the dense transposition materializes again at its boundary.
        assert_eq!(
            used.to_string(),
            indoc! {"
                lambda %0:f32[m], %1:dimension<n ∈ [1, 8)>, %2:dimension<m ∈ [1, 8)> .
                let %3:f32[n] = zero [type=f32[n]] %1
                    %4:f32[n] = zero [type=f32[n]] %1
                in (%4, %0)
            "}
            .trim_end(),
        );
        assert_eq!(
            used.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()),
                extent("n", 2),
                extent("m", 3),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![0.0f32; 2]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()),
            ]),
        );

        // An effectful backward rule runs even when every seed is dead, so both zero seeds need the seed extents. The
        // rule forwards both seeds, so neither contributes a cotangent: `x` receives only the cotangent of its direct
        // use, and the dense transposition materializes the structural zero cotangent of `y` from the seed extent `m`.
        let (_, dead) = linearize(false);
        assert_eq!(
            dead.to_string(),
            indoc! {"
                lambda %0:f32[n], %1:dimension<n ∈ [1, 8)>, %2:dimension<m ∈ [1, 8)> .
                let %3:f32[n] = zero [type=f32[n]] %1
                    %4:f32[m] = zero [type=f32[m]] %2
                    %5:i32[] = const 1
                    %6:dimension<extent ∈ [0, ∞)> = dimension_from_scalar [bounds=[0, ∞)] %5
                    %7:f32[m] = zero [type=f32[m]] %2
                in (%0, %7)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_composite_differentiation_vjp_rule_forwarded_zero_seed() {
        // The backward rule of `(x, y) ↦ (x, y)` returns its seeds as the cotangents of `x` and `y`. When the call's
        // output `x` is dead, its seed is a structural zero that the carrier materializes only because the backward
        // rule takes a value. The cotangent of `x` forwards that seed, so it remains a structural zero instead of
        // crossing the pullback boundary as a materialized value.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let identity = || {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let inputs = vec![builder.add_input(scalar_type.clone()), builder.add_input(scalar_type.clone())];
            builder.build::<Vec<Array>, Vec<Array>>(inputs, vec![Placeholder; 2], vec![Placeholder; 2]).unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let primal = builder.import_program(identity());
        let forward = builder.import_program(identity());
        let backward = builder.import_program(identity());
        let inputs = vec![builder.add_input(scalar_type.clone()), builder.add_input(scalar_type.clone())];
        let outputs = builder
            .add_instruction(
                ArrayOperation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Absent, true)),
                vec![primal, forward, backward],
                inputs,
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![outputs[1]], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let pullback = program
            .entry_region_ref()
            .linearize_shared_for_rule(&[0, 1], DifferentiationRule::JvpForTranspose, None)
            .unwrap()
            .pullback()
            .unwrap();

        // The seed that the carrier materializes for the backward rule stays behind as a dead instruction,
        // but no zero crosses the pullback boundary.
        assert_eq!(pullback.cotangents(), &[TransposedProgramCotangent::Zero, TransposedProgramCotangent::Output]);
        assert_eq!(
            pullback.program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = zero [type=f32[]]
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_composite_differentiation_vjp_rule_nonlinear_backward_seed() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let identity =
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
        let square =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        // A custom backward rule may be nonlinear in its seed. Transposition replays its body inline, so further
        // differentiation computes d(seed²)/d(seed) = 6 at seed 3 instead of treating the rule as a linear map.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let primal = builder.import_region(identity.entry_region_ref());
        let forward = builder.import_region(identity.entry_region_ref());
        let backward = builder.import_region(square.entry_region_ref());
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(
                CompositeOperation::from_rule_regions(CompositeJvpRule::Absent, true),
                vec![primal, forward, backward],
                vec![input],
                None,
            )
            .unwrap()[0];
        let custom =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let linearization = custom
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose, None)
            .unwrap();
        assert_eq!(linearization.residual_count(), 0);
        let custom_pullback = linearization.tangent().transpose().unwrap();
        assert_eq!(custom_pullback.interpret(vec![Array::scalar(3.0).unwrap()]), Ok(vec![Array::scalar(9.0).unwrap()]));
        assert_eq!(
            custom_pullback
                .jvp()
                .unwrap()
                .interpret(vec![Array::scalar(3.0).unwrap(), Array::scalar(1.0).unwrap()]),
            Ok(vec![Array::scalar(9.0).unwrap(), Array::scalar(6.0).unwrap()]),
        );

        // Transposing the pullback again transposes the backward program's ordinary operations, which requires the
        // backward program to be linear in its seed. The seed-non-linear rule is therefore rejected.
        assert!(matches!(
            custom_pullback.transpose(),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message
                    == "operation `mul` does not support transposition for input pattern [left = linear, \
                        right = linear]",
        ));
    }

    #[test]
    fn test_composite_differentiation_vjp_rule_re_transposition() {
        // A seed-linear backward program can be transposed again. The pullback contains the backward program's ordinary
        // operations rather than the carrier, so the result is the forward map `ẋ ↦ 3 cos(x) ẋ` of the
        // custom rule.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let primal = builder.import_program(sin_program(&scalar_type));
        let forward = builder.import_program(sin_forward_program(&scalar_type));
        let backward = builder.import_program(tripled_sin_backward_program(&scalar_type));
        let input = builder.add_input(scalar_type);
        let output = builder
            .add_instruction(
                CompositeOperation::from_rule_regions(CompositeJvpRule::Absent, true),
                vec![primal, forward, backward],
                vec![input],
                None,
            )
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let linearization = program
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose, None)
            .unwrap();
        assert_eq!(linearization.residual_count(), 1);
        let residuals = linearization.primal().interpret(vec![Array::scalar(0.7).unwrap()]).unwrap();
        let pullback = linearization.tangent().transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.interpret(vec![Array::scalar(1.0).unwrap(), residuals[1].clone()]),
            Ok(vec![Array::scalar(3.0 * 0.7f64.cos()).unwrap()]),
        );
        let re_transposed = pullback.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            re_transposed.interpret(vec![Array::scalar(1.0).unwrap(), residuals[1].clone()]),
            Ok(vec![Array::scalar(3.0 * 0.7f64.cos()).unwrap()]),
        );
    }

    #[test]
    fn test_composite_differentiation_vjp_rule_rejects_forward_mode() {
        // A custom VJP supplies no executable tangent rule. Forward mode rejects before the forward rule executes,
        // with the same custom-VJP diagnostic whether evaluating immediately or constructing staged derivatives.
        assert!(matches!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| {
                let scalar_type = ArrayType::scalar(DataType::F64);
                let operation =
                    ArrayOperation::Composite(ArrayComposite::from_rule_regions(CompositeJvpRule::Absent, true));
                let regions = vec![
                    sin_program(&scalar_type),
                    sin_forward_program(&scalar_type),
                    tripled_sin_backward_program(&scalar_type),
                ];
                Ok(x.context().bind(operation, regions, &[x.clone()])?.remove(0))
            }),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == FORWARD_MODE_REJECTION,
        ));
    }

    #[test]
    fn test_composite_differentiation_vjp_rule_rejects_reverse_over_forward() {
        // Reverse differentiation of a directional derivative first applies forward mode to the custom VJP call,
        // which is rejected with the same diagnostic as immediate forward mode.
        assert!(matches!(
            differentiate_at(Array::scalar(2.0).unwrap()).vjp(|x| {
                let direction = x.context().lift(Array::scalar(1.0).unwrap())?;
                Ok(differentiate_at(x)
                    .jvp(direction, |y| {
                        let scalar_type = ArrayType::scalar(DataType::F64);
                        let operation = ArrayOperation::Composite(ArrayComposite::from_rule_regions(
                            CompositeJvpRule::Absent,
                            true,
                        ));
                        let regions = vec![
                            sin_program(&scalar_type),
                            sin_forward_program(&scalar_type),
                            tripled_sin_backward_program(&scalar_type),
                        ];
                        Ok(y.context().bind(operation, regions, &[y.clone()])?.remove(0))
                    })?
                    .1)
            }),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == FORWARD_MODE_REJECTION,
        ));
    }

    #[test]
    fn test_composite_differentiation_vjp_rule_rejects_forward_before_rule_execution() {
        let function = composite(|(counter, input): (ArrayIrTracer, ArrayIrTracer)| {
            counter.add_update(&input)?;
            Ok(input)
        })
        .with_vjp(
            |(counter, input)| {
                counter.add_update(&input)?;
                Ok((input, counter))
            },
            |counter, cotangent| Ok((counter, cotangent)),
        )
        .with_non_differentiated_count(1);
        let rejection = "cannot apply forward-mode differentiation to a `composite` call that \
                         has only reverse-mode rules; it supports only reverse-mode differentiation (e.g., `vjp`, \
                         `value_and_gradient`, or `jacobian_reverse`)";
        let counter = ArrayReference::new(Array::scalar(0.0f32).unwrap());
        let counter_tangent = ArrayReference::new(Array::scalar(0.0f32).unwrap());
        let inputs = (ArrayIrValue::Reference(counter.clone()), ArrayIrValue::Array(Array::scalar(3.0f32).unwrap()));
        let tangents =
            (ArrayIrValue::Reference(counter_tangent.clone()), ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()));
        assert!(matches!(
            differentiate_at(inputs.clone()).jvp(tangents, |inputs| function.call(inputs)),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == rejection,
        ));
        assert!(matches!(
            differentiate_at(inputs.clone()).linearize(|inputs| function.call(inputs)),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == rejection,
        ));
        assert_eq!(counter.read(), Ok(Array::scalar(0.0f32).unwrap()));
        assert_eq!(counter_tangent.read(), Ok(Array::scalar(0.0f32).unwrap()));

        // Building executable forward derivatives of an already staged call rejects immediately too: neither
        // request returns an artifact that defers the error until its reverse-mode carrier is executed.
        let (_, program) = EagerArrayIrContext::trace(
            |inputs| function.call(inputs),
            (
                ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))),
                ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
            ),
        )
        .unwrap();
        assert!(matches!(
            program.entry_region_ref().jvp(&[1]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == rejection,
        ));
        assert!(matches!(
            program.entry_region_ref().linearize(&[1]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == rejection,
        ));
        assert_eq!(counter.read(), Ok(Array::scalar(0.0f32).unwrap()));

        // Constructing the VJP still executes the forward rule's state update exactly once. Reusing the pullback
        // executes only the identity backward rule and preserves the saved caller-owned reference.
        let (value, pullback) = differentiate_at(inputs).vjp(|inputs| function.call(inputs)).unwrap();
        assert_eq!(value, ArrayIrValue::Array(Array::scalar(3.0f32).unwrap()));
        assert_eq!(counter.read(), Ok(Array::scalar(3.0f32).unwrap()));
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(ArrayIrValue::Array(Array::scalar(2.0f32).unwrap())),
                (CotangentDestination::Ignore, CotangentDestination::Return),
            ),
            Ok((None, Some(MaybeZero::Value(ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()))))),
        );
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(ArrayIrValue::Array(Array::scalar(5.0f32).unwrap())),
                (CotangentDestination::Ignore, CotangentDestination::Return),
            ),
            Ok((None, Some(MaybeZero::Value(ArrayIrValue::Array(Array::scalar(5.0f32).unwrap()))))),
        );
        assert_eq!(counter.read(), Ok(Array::scalar(3.0f32).unwrap()));
    }

    #[test]
    fn test_composite_differentiation_vjp_rule_nested_local_reference_state() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let identity = {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let input = builder.add_input(scalar_type.clone());
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![input],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };

        // The JVP-for-transpose rule replays a forward region with local state inside a dormant nested custom rule.
        // It preserves the nested primal semantics without differentiating the forward region itself.
        let forward = nested_composite_state_program(&scalar_type, false);
        assert!(forward.entry_region_ref().contains_effect_in_closure(EffectClass::OrderedState));
        let program = composite_call_program(
            ArrayIrComposite::from_rule_regions(CompositeJvpRule::Absent, true),
            vec![identity.clone(), forward, identity],
            vec![scalar_type],
        );
        let linearization = program
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose, None)
            .unwrap();
        let mut outputs =
            linearization.primal().interpret(vec![ArrayIrValue::Array(Array::scalar(1.0f32).unwrap())]).unwrap();
        let mut inputs = vec![ArrayIrValue::Array(Array::scalar(2.0f32).unwrap())];
        inputs.extend(outputs.split_off(1));
        assert_eq!(outputs, vec![ArrayIrValue::Array(Array::scalar(1.0f32).unwrap())]);
        assert_eq!(
            linearization.pullback().unwrap().program().interpret(inputs),
            Ok(vec![ArrayIrValue::Array(Array::scalar(2.0f32).unwrap())]),
        );
    }

    #[test]
    fn test_composite_differentiation_vjp_rule_reverse_jacobian() {
        // `jacobian_reverse` interprets the pullback with batch-stacked cotangent bases, which exercises the batched
        // replay of the custom backward program. The Jacobian of elementwise `sin` with the tripled rule is the
        // diagonal matrix `diag(3 * cos(x))`.
        let vector_type = ArrayType::new_static(DataType::F64, [2]);
        let jacobian = differentiate_at(Array::vector(vec![0.5, 1.0]).unwrap())
            .jacobian_reverse(|x| {
                let operation =
                    ArrayOperation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Absent, true));
                let regions = vec![
                    sin_program(&vector_type),
                    sin_forward_program(&vector_type),
                    tripled_sin_backward_program(&vector_type),
                ];
                Ok(x.context().bind(operation, regions, &[x.clone()])?.remove(0))
            })
            .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 1);
        assert_eq!(
            blocks[0].value(),
            &Array::matrix(2, 2, vec![3.0 * 0.5f64.cos(), 0.0, 0.0, 3.0 * 1.0f64.cos()]).unwrap(),
        );
    }

    #[test]
    fn test_composite_differentiation_retained_rules() {
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, true));
        let x = Array::scalar(2f64).unwrap();

        // Forward mode traces and replays the JVP rule once per specialization, never the VJP rules.
        for _ in 0..2 {
            assert_eq!(
                call_jvp(&definition, x.clone(), Array::scalar(1f64).unwrap()),
                Ok((vec![Array::scalar(8f64).unwrap()], vec![Array::scalar(4f64).unwrap()])),
            );
        }
        assert_eq!(counters.counts(), (1, 0, 0));

        // Reverse mode traces the forward rule and specializes the backward rule once, never the JVP rule.
        for seed in [1f64, 3f64] {
            assert_eq!(
                call_vjp(&definition, x.clone(), Array::scalar(seed).unwrap()),
                Ok((vec![Array::scalar(8f64).unwrap()], Array::scalar(8f64 * seed).unwrap())),
            );
        }
        assert_eq!(counters.counts(), (1, 1, 1));
    }

    #[test]
    fn test_composite_differentiation_retained_rules_rule_selection() {
        /// Forward-mode rule of one registration row.
        #[derive(Copy, Clone, Debug)]
        enum ForwardRule {
            /// No forward-mode rule.
            Absent,

            /// The deliberately wrong JVP rule `ẏ = x² ẋ`.
            Explicit,

            /// The JVP derived from the primal, `ẏ = 3 x² ẋ`.
            Derived,
        }

        // The registration acceptance contract over the production member family: every registration row selects
        // exactly the rule that each mode requires, and a missing rule never falls back to differentiating the
        // primal. The reverse-mode rules compute the deliberately wrong gradient `x̄ = 2 x² ȳ`.
        let registration = |forward: ForwardRule, reverse: bool| {
            let mut definition = MemberDefinition::new("cube");
            definition = match forward {
                ForwardRule::Absent => definition,
                ForwardRule::Explicit => definition.with_jvp(|primals, tangents| {
                    let square = primals[0].clone() * primals[0].clone();
                    Ok((vec![square.clone() * primals[0].clone()], vec![square * tangents[0].clone()]))
                }),
                ForwardRule::Derived => definition.with_jvp_from_primal(),
            };
            if reverse {
                definition = definition.with_vjp(
                    |primals| {
                        let square = primals[0].clone() * primals[0].clone();
                        Ok((vec![square.clone() * primals[0].clone()], vec![square]))
                    },
                    |leading_inputs, seeds| {
                        let contribution = leading_inputs[0].clone() * seeds[0].clone();
                        Ok(vec![contribution.clone() + contribution])
                    },
                );
            }
            CustomRuleRegistration::new(definition)
        };
        let no_rule = "cannot differentiate a `composite` call of `cube` that has no derivative rule";
        let forward_rejection = "cannot apply forward-mode differentiation to a `composite` call of `cube` \
                                 that has only reverse-mode rules; it supports only reverse-mode differentiation \
                                 (e.g., `vjp`, `value_and_gradient`, or `jacobian_reverse`)";
        let x = 2f64;
        for (forward, reverse, tangent, gradient) in [
            (ForwardRule::Absent, false, Err(no_rule), Err(no_rule)),
            (ForwardRule::Derived, false, Ok(3.0 * x * x), Ok(3.0 * x * x)),
            (ForwardRule::Explicit, false, Ok(x * x), Ok(x * x)),
            (ForwardRule::Absent, true, Err(forward_rejection), Ok(2.0 * x * x)),
            (ForwardRule::Explicit, true, Ok(x * x), Ok(2.0 * x * x)),
            (ForwardRule::Derived, true, Ok(3.0 * x * x), Ok(2.0 * x * x)),
        ] {
            let definition = registration(forward, reverse);
            let scalar_type = ArrayType::scalar(DataType::F64);
            let result = crate::differentiation::differentiate_at(Array::scalar(x).unwrap()).jvp(
                Array::scalar(1f64).unwrap(),
                |input| {
                    let operation = ArrayOperation::Composite(CompositeOperation::new(definition.reference()));
                    Ok(input
                        .context()
                        .bind(operation, vec![member_cube_program(&scalar_type)], &[input.clone()])?
                        .remove(0))
                },
            );
            match tangent {
                Ok(tangent) => assert_eq!(
                    result,
                    Ok((Array::scalar(x * x * x).unwrap(), Array::scalar(tangent).unwrap())),
                    "{forward:?} {reverse}",
                ),
                Err(expected) => assert!(
                    matches!(
                        result,
                        Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                            if message == expected,
                    ),
                    "{forward:?} {reverse}",
                ),
            }
            let result = crate::differentiation::differentiate_at(Array::scalar(x).unwrap())
                .vjp(|input| {
                    let operation = ArrayOperation::Composite(CompositeOperation::new(definition.reference()));
                    Ok(input
                        .context()
                        .bind(operation, vec![member_cube_program(&scalar_type)], &[input.clone()])?
                        .remove(0))
                })
                .and_then(|(value, pullback)| Ok((value, pullback.apply_materialized(Array::scalar(1f64).unwrap())?)));
            match gradient {
                Ok(gradient) => assert_eq!(
                    result,
                    Ok((Array::scalar(x * x * x).unwrap(), Array::scalar(gradient).unwrap())),
                    "{forward:?} {reverse}",
                ),
                Err(expected) => assert!(
                    matches!(
                        result,
                        Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                            if message == expected,
                    ),
                    "{forward:?} {reverse}",
                ),
            }
        }

        // Higher-order differentiation differentiates the selected rule programs rather than the primal: the derived
        // JVP has the true second derivative `6 x`, while forward-over-reverse through the backward rule `2 x² ȳ` gives
        // `4 x`.
        for (forward, reverse, expected) in
            [(ForwardRule::Derived, false, 6.0 * x), (ForwardRule::Explicit, true, 4.0 * x)]
        {
            let definition = registration(forward, reverse);
            let scalar_type = ArrayType::scalar(DataType::F64);
            let (_, second_derivative) = crate::differentiation::differentiate_at(Array::scalar(x).unwrap())
                .jvp(Array::scalar(1f64).unwrap(), |input| {
                    Ok(crate::differentiation::differentiate_at(input)
                        .gradient(|input| {
                            let operation = ArrayOperation::Composite(CompositeOperation::new(definition.reference()));
                            input
                                .context()
                                .bind(operation, vec![member_cube_program(&scalar_type)], &[input.clone()])
                                .unwrap()
                                .remove(0)
                        })
                        .unwrap())
                })
                .unwrap();
            assert_eq!(second_derivative, Array::scalar(expected).unwrap(), "{forward:?} {reverse}");
        }

        // A call without any rule still executes its primal.
        let definition = registration(ForwardRule::Absent, false);
        let program = member_custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        assert_eq!(program.interpret(vec![Array::scalar(x).unwrap()]), Ok(vec![Array::scalar(x * x * x).unwrap()]));
    }

    #[test]
    fn test_composite_differentiation_retained_rules_error_timing() {
        let x = Array::scalar(2f64).unwrap();
        let one = Array::scalar(1f64).unwrap();
        let failure = || ProgramError::InvalidArgument { message: "rule failure".to_string() };

        // A failing JVP rule surfaces at the first forward-mode request, never during ordinary execution or reverse
        // mode.
        let counters = Arc::new(RuleCounters::default());
        let definition =
            CustomRuleRegistration::new(cube_definition(&counters, false, true).with_jvp(move |_, _| Err(failure())));
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        assert_eq!(program.interpret(vec![x.clone()]), Ok(vec![Array::scalar(8f64).unwrap()]));
        assert_eq!(
            call_vjp(&definition, x.clone(), one.clone()),
            Ok((vec![Array::scalar(8f64).unwrap()], Array::scalar(8f64).unwrap())),
        );
        assert_eq!(call_jvp(&definition, x.clone(), one.clone()), Err(DifferentiationError::Program(failure())));

        // A JVP rule with the wrong result structure surfaces at the same stage with a type error.
        let definition = CustomRuleRegistration::new(
            TestDefinition::new("cube").with_jvp(|primals, _| Ok((primals.to_vec(), vec![]))),
        );
        assert!(matches!(
            call_jvp(&definition, x.clone(), one.clone()),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "`composite` `cube` JVP rule returned 1 outputs and 0 output tangents but the primal \
                               has 1 outputs",
        ));

        // A failing forward rule surfaces at the first reverse-mode request, while a failing backward rule surfaces
        // only when the first pullback application transposes the carrier.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(
            cube_definition(&counters, true, false).with_vjp(move |_| Err(failure()), |_, _| unreachable!()),
        );
        let jvp = Ok((vec![Array::scalar(8f64).unwrap()], vec![Array::scalar(4f64).unwrap()]));
        assert_eq!(call_jvp(&definition, x.clone(), one.clone()), jvp);
        assert_eq!(call_vjp(&definition, x.clone(), one.clone()), Err(DifferentiationError::Program(failure())));
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, false).with_vjp(
            |primals| Ok((vec![primals[0].clone() * primals[0].clone() * primals[0].clone()], vec![])),
            move |_, _| Err(failure()),
        ));
        assert_eq!(call_jvp(&definition, x.clone(), one.clone()), jvp);
        let r#type = x.r#type().into_owned();
        let (_, pullback) = TestContext::new()
            .vjp(
                |x, ()| {
                    let operation = CompositeOperation::new(definition.reference());
                    x.context().bind(operation, vec![cube_program(&r#type)], &[x.clone()])
                },
                x,
                (),
            )
            .unwrap();
        assert!(matches!(
            pullback.apply(vec![one]),
            Err(ProgramError::InvalidArgument { message }) if message == "rule failure",
        ));
    }

    #[test]
    fn test_composite_differentiation_retained_rules_rule_result_partitions() {
        // Each rule result partition is validated on its own, so a rule that moves a primal output into its other
        // partition is rejected even though the flattened result types match the expected ones exactly.
        let x = Array::scalar(2f64).unwrap();
        let one = Array::scalar(1f64).unwrap();
        let definition = CustomRuleRegistration::new(TestDefinition::new("cube").with_jvp(|primals, tangents| {
            let square = primals[0].clone() * primals[0].clone();
            Ok((vec![], vec![square.clone() * primals[0].clone(), square * tangents[0].clone()]))
        }));
        assert!(matches!(
            call_jvp(&definition, x.clone(), one.clone()),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "`composite` `cube` JVP rule returned 0 outputs and 2 output tangents but the primal \
                               has 1 outputs",
        ));

        // Correctly partitioned results still have their types validated.
        let definition = CustomRuleRegistration::new(TestDefinition::new("cube").with_jvp(|primals, tangents| {
            let tangent = tangents[0].context().lift(Array::scalar(1f32).unwrap())?;
            Ok((primals.to_vec(), vec![tangent]))
        }));
        assert!(matches!(
            call_jvp(&definition, x.clone(), one.clone()),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "`composite` `cube` JVP rule output type signature mismatch: expected \
                               [f64[], f64[]] but got [f64[], f32[]]",
        ));
        let definition = CustomRuleRegistration::new(TestDefinition::new("cube").with_vjp(
            |primals| Ok((vec![], vec![primals[0].clone() * primals[0].clone() * primals[0].clone()])),
            |_, _| unreachable!(),
        ));
        assert!(matches!(
            call_vjp(&definition, x, one),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "`composite` `cube` forward rule returned 0 outputs but the primal has 1",
        ));
    }

    #[test]
    fn test_composite_differentiation_retained_rules_structural_zero_tangents() {
        // Forward-mode specializations receive only the active input tangents, and each activity pattern is a separate
        // specialization. The pair rule forwards each input tangent to the corresponding output.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let pair_program = |definition: &TestRegistration| {
            let mut primal = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let inputs = vec![primal.add_input(scalar_type.clone()), primal.add_input(scalar_type.clone())];
            let primal =
                primal.build::<Vec<Array>, Vec<Array>>(inputs, vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let primal = builder.import_program(primal);
            let inputs = vec![builder.add_input(scalar_type.clone()), builder.add_input(scalar_type.clone())];
            let outputs = builder
                .add_instruction(CompositeOperation::new(definition.reference()), vec![primal], inputs, None)
                .unwrap()
                .to_vec();
            builder
                .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };

        // A rule that receives materialized zeros sees zeros that its specialization stages itself, so an output
        // tangent that forwards an inactive input tangent remains a structural zero.
        let definition = TestRegistration::new(
            TestDefinition::new("pair").with_jvp(|primals, tangents| Ok((primals.to_vec(), tangents.to_vec()))),
        );
        let program = pair_program(&definition);
        assert_eq!(
            program.jvp_with_respect_to(&[0]).unwrap().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[] .
                let %3:f64[] = zero [type=f64[]]
                    %4:f64[] = zero [type=f64[]]
                in (%0, %1, %2, %4)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                in (%0, %1, %2, %3)
            "}
            .trim_end(),
        );
        assert_eq!(definition.caches().jvp_specializations.len(), 2);

        // A rule that receives structural zeros sees them as `MaybeZero::Zero` leaves and may return them as the
        // tangents of its outputs, which the specialization materializes at its boundary.
        let observed = Arc::new(Mutex::new(Vec::new()));
        let definition = TestRegistration::new(TestDefinition::new("pair").with_symbolic_zero_jvp({
            let observed = observed.clone();
            move |primals, tangents| {
                observed.lock().unwrap().push(tangents.iter().map(MaybeZero::is_zero).collect::<Vec<_>>());
                Ok((primals.to_vec(), tangents.to_vec()))
            }
        }));
        let program = pair_program(&definition);
        assert_eq!(
            program.jvp_with_respect_to(&[1]).unwrap().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[] .
                let %3:f64[] = zero [type=f64[]]
                    %4:f64[] = zero [type=f64[]]
                in (%0, %1, %4, %2)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                in (%0, %1, %2, %3)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.jvp_with_respect_to(&[1]).unwrap().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[] .
                let %3:f64[] = zero [type=f64[]]
                    %4:f64[] = zero [type=f64[]]
                in (%0, %1, %4, %2)
            "}
            .trim_end(),
        );
        assert_eq!(*observed.lock().unwrap(), vec![vec![true, false], vec![false, false]]);
    }

    #[test]
    fn test_composite_differentiation_retained_rules_retained_partition() {
        // Reverse mode through a transposed JVP rule partitions the rule's specialization into its known and tangent
        // parts. The specialization's region retains that partition, so repeated eager requests partition it once.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, false));
        for _ in 0..3 {
            assert_eq!(
                call_vjp(&definition, Array::scalar(2f64).unwrap(), Array::scalar(1f64).unwrap()),
                Ok((vec![Array::scalar(8f64).unwrap()], Array::scalar(4f64).unwrap())),
            );
        }
        let scalar_type = ArrayType::scalar(DataType::F64);
        let specialization = definition
            .reference()
            .jvp_specialization(CustomRuleSpecializationKey {
                input_types: vec![scalar_type.clone()],
                output_types: vec![scalar_type],
                kept_outputs: None,
                non_differentiated_count: 0,
                tangent_activity: vec![true],
                levels: Vec::new(),
                discharged: false,
            })
            .unwrap();
        let statistics =
            specialization.program.entry_region_ref().transform_statistics::<JvpPartitionTransform>().unwrap();
        assert_eq!((statistics.productions, statistics.hits), (1, 2));
        assert_eq!(counters.counts(), (1, 0, 0));
    }

    #[test]
    fn test_composite_differentiation_retained_rules_linearization() {
        // Forward-mode linearization replays the traced JVP rule, while reverse-mode linearization stages the carrier
        // over the forward rule's residual and the input tangent.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, true));
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        let forward = program.linearize().unwrap();
        let reverse = program
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose, None)
            .unwrap();
        assert_eq!(counters.counts(), (1, 1, 0));
        assert_eq!(
            forward.tangent().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %0
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            reverse.primal().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = mul %0 %0
                    %2:f64[] = mul %1 %0
                in (%2, %1)
            "}
            .trim_end(),
        );
        assert_eq!(
            reverse.tangent().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = composite_transpose [name=\"cube\", leading_input_count=1] %1 %0
                in (%2)
            "}
            .trim_end(),
        );

        // Transposition replaces the carrier with the destination-specialized backward rule.
        let pullback = reverse.tangent().transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(counters.counts(), (1, 1, 1));
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %0
                    %3:f64[] = add %2 %2
                in (%3)
            "}
            .trim_end(),
        );

        // The specialized pullback consists of ordinary operations, so differentiating it again needs no rule and
        // invokes none. Its tangent at the seed `ȳ` and the residual `s` is `2 (ȳ δs + s δȳ)`.
        let pullback_linearization = pullback.linearize().unwrap();
        assert_eq!(counters.counts(), (1, 1, 1));
        assert_eq!(
            pullback_linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                let %4:f64[] = mul %2 %1
                    %5:f64[] = mul %3 %0
                    %6:f64[] = add %4 %5
                    %7:f64[] = add %6 %6
                in (%7)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_composite_differentiation_retained_rules_deferred_forward() {
        // A call whose rules defer their forward rule replays its primal on the known side and stages a carrier over
        // its own input `x`, whose transposition replays the forward rule for the residual `x²` before applying the
        // backward rule. The deliberately wrong gradient `2 x² ȳ` of the backward rule is that of the call without
        // deferral.
        let counters = Arc::new(RuleCounters::default());
        let definition =
            CustomRuleRegistration::new(cube_definition(&counters, false, true).with_deferred_forward_rule());
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = composite [name=\"cube\", deferred_forward=true] %0 [
                    primal={
                        lambda %0:f64[] .
                        let %1:f64[] = mul %0 %0
                            %2:f64[] = mul %1 %0
                        in (%2)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );

        // The forward rule is traced while the carrier is staged, which validates that it is pure.
        let reverse = program
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose, None)
            .unwrap();
        assert_eq!(counters.counts(), (0, 1, 0));
        assert_eq!(
            reverse.primal().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = mul %0 %0
                    %2:f64[] = mul %1 %0
                in (%2, %0)
            "}
            .trim_end(),
        );
        assert_eq!(
            reverse.tangent().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = composite_transpose [name=\"cube\", leading_input_count=1, deferred_forward=true] %1 %0
                in (%2)
            "}
            .trim_end(),
        );
        let pullback = reverse.tangent().transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(counters.counts(), (0, 1, 1));
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %1
                    %3:f64[] = mul %2 %0
                    %4:f64[] = add %3 %3
                in (%4)
            "}
            .trim_end(),
        );
        for seed in [1f64, 3f64] {
            assert_eq!(
                call_vjp(&definition, Array::scalar(2f64).unwrap(), Array::scalar(seed).unwrap()),
                Ok((vec![Array::scalar(8f64).unwrap()], Array::scalar(8f64 * seed).unwrap())),
            );
        }
        assert_eq!(counters.counts(), (0, 1, 1));
    }

    #[test]
    fn test_composite_differentiation_retained_rules_deferred_forward_rematerialization() {
        // Saving everything that a rematerialized body computes saves the forward rule's residual `cos(x)` of a call
        // that does not defer its forward rule, but only the call's input `x` of one that does, while the gradient is
        // the same.
        type Tracer = DomainTracer<ArrayContext>;
        let sine = || {
            composite(|x: Tracer| Ok(x.sin()?))
                .with_vjp(|x: Tracer| Ok((x.sin()?, x.cos()?)), |cosine, cotangent| Ok(cosine * cotangent))
        };
        let x = Array::scalar(0.5f64).unwrap();
        let seed = Array::scalar(1f64).unwrap();
        let gradient = Array::scalar(0.5f64.cos()).unwrap();
        let eager = sine();
        let function = rematerialize(move |x: Tracer| eager.call(x)).with_policy(EverythingSavable);
        let (value, pullback) = differentiate_at(x.clone()).vjp(|x| function.call(x)).unwrap();
        assert_eq!(value, Array::scalar(0.5f64.sin()).unwrap());
        assert_eq!(pullback.residuals(), &[Array::scalar(0.5f64.cos()).unwrap()]);
        assert_eq!(pullback.apply_materialized(seed.clone()), Ok(gradient.clone()));
        let deferred = sine().with_deferred_forward();
        let function = rematerialize(move |x: Tracer| deferred.call(x)).with_policy(EverythingSavable);
        let (value, pullback) = differentiate_at(x.clone()).vjp(|x| function.call(x)).unwrap();
        assert_eq!(value, Array::scalar(0.5f64.sin()).unwrap());
        assert_eq!(pullback.residuals(), std::slice::from_ref(&x));
        assert_eq!(pullback.apply_materialized(seed), Ok(gradient));
    }

    #[test]
    fn test_composite_differentiation_retained_rules_deferred_forward_rejects_effectful_forward_rule() {
        // A forward rule that prints would print in the backward pass, out of order, so reverse mode rejects it before
        // staging anything, while ordinary execution of the call, which runs only the primal, is unaffected.
        type Tracer = DomainTracer<ArrayContext>;
        let function = composite(|x: Tracer| Ok(x.sin()?))
            .with_name("sine")
            .with_vjp(|x: Tracer| Ok((x.sin()?, x.print("x")?.cos()?)), |cosine, cotangent| Ok(cosine * cotangent))
            .with_deferred_forward();
        let x = Array::scalar(0.5f64).unwrap();
        assert_eq!(function.call(x.clone()), Ok(Array::scalar(0.5f64.sin()).unwrap()));
        assert!(matches!(
            differentiate_at(x).gradient(|x| function.call(x)),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`composite` `sine` defers its forward rule to the backward pass (refer to \
                               `with_deferred_forward`), which requires a pure forward rule, but its forward rule has \
                               effects, which would run out of order in the backward pass",
        ));
    }

    #[test]
    fn test_composite_differentiation_retained_rules_deferred_forward_second_order() {
        // The pullback replays the forward and backward rules as ordinary operations, so differentiating the gradient
        // `cos(x) ȳ` again, in reverse or in forward mode, differentiates the recomputed residual `cos(x)`.
        type Tracer = DomainTracer<ArrayContext>;
        let sine = composite(|x: Tracer| Ok(x.sin()?))
            .with_vjp(|x: Tracer| Ok((x.sin()?, x.cos()?)), |cosine, cotangent| Ok(cosine * cotangent))
            .with_deferred_forward();
        let x = Array::scalar(0.7f64).unwrap();
        assert_eq!(
            differentiate_at(x.clone()).value_and_gradient(|x| differentiate_at(x).gradient(|x| sine.call(x)).unwrap()),
            Ok((Array::scalar(0.7f64.cos()).unwrap(), Array::scalar(-0.7f64.sin()).unwrap())),
        );
        assert_eq!(
            differentiate_at(x)
                .jvp(Array::scalar(1f64).unwrap(), |x| Ok(differentiate_at(x).gradient(|x| sine.call(x)).unwrap())),
            Ok((Array::scalar(0.7f64.cos()).unwrap(), Array::scalar(-0.7f64.sin()).unwrap())),
        );
    }

    #[test]
    fn test_composite_differentiation_retained_rules_reference_residuals() {
        // A reference residual forwards a leading non-differentiated input by identity, and the backward rule receives
        // it as plumbing. The rules compute the deliberately wrong gradient `x̄ = 3 x ȳ` of `f(counter, x) = x²`.
        let registration = |forward: fn(&[IrTracer]) -> Result<(Vec<IrTracer>, Vec<IrTracer>), ProgramError>| {
            CustomRuleRegistration::new(IrDefinition::new("scaled").with_vjp(forward, |leading_inputs, seeds| {
                let doubled = ir_multiply(&leading_inputs[2], &seeds[0])?;
                let three = seeds[0].context().lift(ArrayIrValue::Array(Array::scalar(3.0f32)?))?;
                Ok(vec![ir_multiply(&three, &doubled)?])
            }))
        };
        let vjp = |definition: &CustomRuleRegistration<ArrayIrValue<Array>, ArrayIrOperation<Array>>| {
            let counter = ArrayReference::new(Array::scalar(0.0f32).unwrap());
            let (value, pullback) = differentiate_at((
                ArrayIrValue::Reference(counter),
                ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()),
            ))
            .vjp(|(counter, x)| {
                let operation = ArrayIrOperation::Composite(
                    CompositeOperation::new(definition.reference()).with_non_differentiated_count(1).unwrap(),
                );
                Ok(x.context().bind(operation, vec![ir_counter_square_program()], &[counter, x.clone()])?.remove(0))
            })?;
            let (_, cotangent) = pullback.apply_with_destinations(
                CotangentSeed::Value(ArrayIrValue::Array(Array::scalar(1.0f32).unwrap())),
                (CotangentDestination::Ignore, CotangentDestination::Return),
            )?;
            Ok::<_, DifferentiationError>((value, cotangent))
        };
        let definition = registration(|primals| {
            Ok((vec![ir_multiply(&primals[1], &primals[1])?], vec![primals[0].clone(), primals[1].clone()]))
        });
        assert_eq!(
            vjp(&definition),
            Ok((
                ArrayIrValue::Array(Array::scalar(4.0f32).unwrap()),
                Some(MaybeZero::Value(ArrayIrValue::Array(Array::scalar(6.0f32).unwrap()))),
            )),
        );

        // A residual reference that does not forward a leading input, or that forwards an input already forwarded by an
        // earlier residual, is rejected when the forward rule is first traced.
        let definition = registration(|primals| {
            Ok((vec![ir_multiply(&primals[1], &primals[1])?], vec![primals[1].reference_new()?, primals[1].clone()]))
        });
        assert!(matches!(
            vjp(&definition),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "`composite` `scaled` forward rule returns residual 0 of reference type `ref<f32[]>` \
                               that is not a leading non-differentiated input forwarded by identity",
        ));
        let definition = registration(|primals| {
            Ok((vec![ir_multiply(&primals[1], &primals[1])?], vec![primals[0].clone(), primals[0].clone()]))
        });
        assert!(matches!(
            vjp(&definition),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "`composite` `scaled` forward rule returns residual 1 of reference type `ref<f32[]>` \
                               from an input already forwarded by an earlier residual",
        ));
    }

    #[test]
    fn test_composite_differentiation_retained_rules_specialization_after_program_changes() {
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, true));
        let original = DimensionVariable::new("original", DimensionBounds::new(2, Some(6)).unwrap());
        let relocated = DimensionVariable::new("relocated", DimensionBounds::new(2, Some(6)).unwrap());
        let original_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(original.clone())]));
        let relocated_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(relocated.clone())]));
        let static_type = ArrayType::new_static(DataType::F64, [3]);
        let program = custom_rule_program(&definition, original_type);

        // The first specialization happens only after the staged program was renamed, so every rule sees the renamed
        // boundary.
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(original, relocated).unwrap();
        let renamed = program.clone().rename_type_identities(&renaming).unwrap();
        let forward = renamed.linearize().unwrap();
        let reverse = renamed
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose, None)
            .unwrap();
        let pullback = reverse.tangent().transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            forward.tangent().to_string(),
            indoc! {"
                lambda %0:f64[relocated], %1:f64[relocated] .
                let %2:f64[relocated] = mul %1 %0
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[relocated], %1:f64[relocated] .
                let %2:f64[relocated] = mul %1 %0
                    %3:f64[relocated] = add %2 %2
                in (%3)
            "}
            .trim_end(),
        );

        // Specializing the boundary to a static extent replays the call at the refined type.
        let specialized = program.specialize(&[static_type.clone()]).unwrap().linearize().unwrap();
        assert_eq!(
            specialized.tangent().to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3] .
                let %2:f64[3] = mul %1 %0
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(counters.counts(), (2, 1, 1));
        assert_eq!(
            definition.caches().jvp_specializations.keys(),
            vec![
                CustomRuleSpecializationKey {
                    input_types: vec![static_type.clone()],
                    output_types: vec![static_type],
                    kept_outputs: None,
                    non_differentiated_count: 0,
                    tangent_activity: vec![true],
                    levels: Vec::new(),
                    discharged: false,
                },
                CustomRuleSpecializationKey {
                    input_types: vec![relocated_type.clone()],
                    output_types: vec![relocated_type.clone()],
                    kept_outputs: None,
                    non_differentiated_count: 0,
                    tangent_activity: vec![true],
                    levels: Vec::new(),
                    discharged: false,
                },
            ],
        );
        assert_eq!(
            definition.caches().backward_specializations.keys(),
            vec![CustomRuleBackwardSpecializationKey {
                leading_input_types: vec![relocated_type.clone()],
                seed_extent_count: 0,
                input_tangent_types: vec![relocated_type.clone()],
                output_tangent_types: vec![relocated_type.clone()],
                seed_types: vec![Some(relocated_type)],
                destination_kinds: vec![CotangentDestinationKind::Return],
                levels: Vec::new(),
                discharged: false,
            }],
        );
    }

    #[test]
    fn test_composite_differentiation_retained_rules_specialization_output_signature() {
        // Two calls of one definition with identical input types but different primal outputs must not share a
        // specialization: after the caches are warmed by a one-output primal, a two-output primal is validated afresh
        // and rejected, rather than replaying the cached programs against the wrong output boundary.
        let definition = CustomRuleRegistration::new(
            TestDefinition::new("forward_input")
                .with_jvp(|primals, tangents| Ok((primals.to_vec(), tangents.to_vec())))
                .with_vjp(|primals| Ok((primals.to_vec(), vec![])), |_, seeds| Ok(seeds.to_vec())),
        );
        let r#type = ArrayType::scalar(DataType::F64);
        let primal = |output_count: usize| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(r#type.clone());
            builder
                .build(vec![input; output_count], vec![Placeholder], vec![Placeholder; output_count])
                .unwrap()
        };
        let jvp = |primal: FlatProgram<TestContext>| {
            TestContext::new().jvp(
                |x, ()| {
                    x.context().bind(
                        CompositeOperation::new(definition.reference()),
                        vec![primal.clone()],
                        &[x.clone()],
                    )
                },
                Array::scalar(2f64).unwrap(),
                Array::scalar(1f64).unwrap(),
                (),
            )
        };
        let vjp = |primal: FlatProgram<TestContext>| {
            TestContext::new()
                .vjp(
                    |x, ()| {
                        x.context().bind(
                            CompositeOperation::new(definition.reference()),
                            vec![primal.clone()],
                            &[x.clone()],
                        )
                    },
                    Array::scalar(2f64).unwrap(),
                    (),
                )
                .map(|(outputs, _)| outputs)
        };
        assert_eq!(jvp(primal(1)), Ok((vec![Array::scalar(2f64).unwrap()], vec![Array::scalar(1f64).unwrap()])));
        assert_eq!(vjp(primal(1)), Ok(vec![Array::scalar(2f64).unwrap()]));
        assert!(matches!(
            jvp(primal(2)),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "`composite` `forward_input` JVP rule returned 1 outputs and 1 output tangents but \
                               the primal has 2 outputs",
        ));
        assert!(matches!(
            vjp(primal(2)),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "`composite` `forward_input` forward rule returned 1 outputs but the primal has 2",
        ));
        assert_eq!(definition.caches().jvp_specializations.keys().len(), 1);
        assert_eq!(definition.caches().forward_specializations.keys().len(), 1);
    }

    #[test]
    fn test_composite_transposition() {
        // A call that is linear in its unknown inputs transposes its primal region with respect to them, and the
        // transposed region receives the live seeds followed by the known inputs.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let operation = ArrayOperation::Composite(ArrayComposite::from_rule_regions(CompositeJvpRule::Absent, false));
        let program =
            composite_call_program(operation, vec![scalar_multiply_program()], vec![scalar_type.clone(), scalar_type]);
        let transposed = program.transpose_with_respect_to(&[1], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %0
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            transposed.interpret(vec![Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()]),
            Ok(vec![Array::scalar(6.0).unwrap()]),
        );
    }

    #[test]
    fn test_composite_transposition_omits_structural_zero_seeds() {
        // The primal `(p, x) ↦ (p · x, x + x)` is linear in `x`. When only its first output is live, the second
        // output's seed is a structural zero: the transposed primal receives no input for it and stages none of the
        // work that only that seed reaches. With both outputs live, the second seed contributes `ȳ₁ + ȳ₁`.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let primal = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let scale = builder.add_input(scalar_type.clone());
            let linear = builder.add_input(scalar_type.clone());
            let product =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![linear, scale], None).unwrap()[0];
            let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![linear, linear], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![product, sum], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let operation = ArrayOperation::Composite(ArrayComposite::from_rule_regions(CompositeJvpRule::Absent, false));
        let program = composite_call_program(operation, vec![primal], vec![scalar_type.clone(), scalar_type]);
        let scalar = |value: f64| Array::scalar(value).unwrap();

        // Projecting the program onto the first output keeps the two-output call, so its second output's seed reaches
        // the call's transpose rule as a structural zero rather than being pruned away beforehand.
        let projected = program.with_outputs(&[0]).unwrap();
        assert_eq!(
            projected.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[], %3:f64[] = composite %0 %1 [
                    primal={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = mul %1 %0
                            %3:f64[] = add %1 %1
                        in (%2, %3)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        let transposed = projected.transpose_with_respect_to(&[1], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %0
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(transposed.interpret(vec![scalar(2.0), scalar(3.0)]), Ok(vec![scalar(6.0)]));

        let transposed = program.transpose_with_respect_to(&[1], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[] .
                let %3:f64[] = add %1 %1
                    %4:f64[] = mul %2 %0
                    %5:f64[] = add %3 %4
                in (%5)
            "}
            .trim_end(),
        );
        assert_eq!(transposed.interpret(vec![scalar(2.0), scalar(5.0), scalar(3.0)]), Ok(vec![scalar(16.0)]));
    }

    #[test]
    fn test_composite_transposition_jvp_rule() {
        // Differentiation replaces the call with its replayed rule before transposition, so a direct transpose of an
        // un-linearized call transposes its primal region, whose nonlinear `sin` rejects it.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let operation =
            ArrayOperation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, false));
        let regions = vec![sin_program(&scalar_type), doubled_sin_jvp_program(&scalar_type)];
        let program = composite_call_program(operation, regions, vec![scalar_type]);
        assert!(matches!(
            program.transpose_with_respect_to(&[0], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `sin` is not transposable",
        ));
    }

    #[test]
    fn test_composite_transposition_vjp_rule() {
        // Differentiation replaces the call with its reverse-mode carrier before transposition, so a direct transpose
        // of an un-linearized call transposes its primal region, whose nonlinear `sin` rejects it.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let operation =
            ArrayOperation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Absent, true));
        let regions = vec![
            sin_program(&scalar_type),
            sin_forward_program(&scalar_type),
            tripled_sin_backward_program(&scalar_type),
        ];
        let program = composite_call_program(operation, regions, vec![scalar_type]);
        assert!(matches!(
            program.transpose_with_respect_to(&[0], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `sin` is not transposable",
        ));
    }

    #[test]
    fn test_composite_transposition_pushforward() {
        // Linearizing a call whose forward-mode rule is derived from its primal and whose rules have a custom batching
        // rule stages a pushforward call on the tangent side, which is linear in its tangent inputs, so transposing the
        // linearization transposes its primal region with respect to them.
        type Tracer = DomainTracer<ArrayContext>;
        let function = composite(|(x, y): (Tracer, Tracer)| Ok(x * y)).with_batching(
            |_: BatchingLevelExtent<Tracer>, (x, y): (Tracer, Tracer), (x_axis, y_axis): (BatchAxis, BatchAxis)| {
                Ok((x * y, if x_axis.is_replicated() { y_axis } else { x_axis }))
            },
        );
        let scalar_type = ArrayType::scalar(DataType::F64);
        let (_, program) =
            ArrayContext::trace(|inputs| function.call(inputs), (scalar_type.clone(), scalar_type)).unwrap();
        let linearization = program.into_flat_program().linearize().unwrap();
        let pullback = linearization.tangent().transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[] .
                let %3:f64[] = mul %1 %0
                    %4:f64[] = mul %2 %0
                in (%4, %3)
            "}
            .trim_end(),
        );
        let scalar = |value: f64| Array::scalar(value).unwrap();
        assert_eq!(pullback.interpret(vec![scalar(1.0), scalar(2.0), scalar(3.0)]), Ok(vec![scalar(3.0), scalar(2.0)]));
    }

    #[test]
    fn test_composite_lifting_retained_rules() {
        // A definition written against the member family reaches array IR programs by converting a member program. The
        // converted call keeps the member definition, which specializes its rules at the projected member types, so
        // every family that the call reaches shares the member definition's traces.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(member_cube_definition(&counters));
        let member = member_custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        let converted: ConvertedProgram = member.clone().into_unprojected().unwrap();
        assert_eq!(
            converted.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = composite [name=\"cube\"] %0 [
                    primal={
                        lambda %0:f64[] .
                        let %1:f64[] = mul %0 %0
                            %2:f64[] = mul %1 %0
                        in (%2)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );

        // The rendering does not show that the call is lifted, which only its operation variant records.
        assert!(matches!(converted.instructions()[0].operation(), ArrayIrOperation::LiftedComposite(_)));
        assert_eq!(counters.counts(), (0, 0, 0));

        // Forward mode uses the converted JVP rule, whose program contains only array IR operations.
        let x = ArrayIrValue::Array(Array::scalar(2f64).unwrap());
        let one = ArrayIrValue::Array(Array::scalar(1f64).unwrap());
        let linearization = converted.linearize().unwrap();
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %0
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            converted.jvp().unwrap().interpret(vec![x.clone(), one.clone()]),
            Ok(vec![
                ArrayIrValue::Array(Array::scalar(8f64).unwrap()),
                ArrayIrValue::Array(Array::scalar(4f64).unwrap())
            ]),
        );
        assert_eq!(counters.counts(), (1, 0, 0));

        // Reverse mode uses the converted reverse-mode rules.
        let reverse = converted
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose, None)
            .unwrap();
        let pullback = reverse.tangent().transpose_with_respect_to(&[0], &[]).unwrap();
        let primals = reverse.primal().interpret(vec![x.clone()]).unwrap();
        let mut pullback_inputs = vec![one.clone()];
        pullback_inputs.extend_from_slice(&primals[1..]);
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![ArrayIrValue::Array(Array::scalar(8f64).unwrap())]));
        assert_eq!(counters.counts(), (1, 1, 1));

        // Converting the same member program again yields an equal call that reuses the member definition's traces.
        let again: ConvertedProgram = member.clone().into_unprojected().unwrap();
        let (
            ArrayIrOperation::LiftedComposite(again_operation),
            ArrayIrOperation::LiftedComposite(converted_operation),
        ) = (again.instructions()[0].operation(), converted.instructions()[0].operation())
        else {
            unreachable!()
        };
        assert_eq!(again_operation, converted_operation);
        assert_eq!(again.linearize().unwrap().tangent().to_string(), linearization.tangent().to_string());
        assert_eq!(counters.counts(), (1, 1, 1));

        // A member call batched at a static level before the conversion is differentiated at that level.
        let batched: ConvertedProgram = member
            .batched(3, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts()
            .0
            .into_unprojected()
            .unwrap();
        assert_eq!(
            batched.jvp().unwrap().interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1f64, 2f64, 3f64]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![1f64; 3]).unwrap()),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![1f64, 8f64, 27f64]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![1f64, 4f64, 9f64]).unwrap()),
            ]),
        );
        assert_eq!(counters.counts(), (1, 1, 1));

        // Converted calls follow the ownership of their member definition: the handle owns the caches, and the programs
        // keep the definition alive.
        let (retained, retained_caches) =
            (Arc::downgrade(&definition.definition()), Arc::downgrade(&definition.caches()));
        drop(definition);
        assert!(retained_caches.upgrade().is_none());
        drop((member, converted, again, batched, linearization, reverse, pullback));
        assert!(retained.upgrade().is_none());
    }
}
