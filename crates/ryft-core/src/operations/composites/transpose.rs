//! The [`CompositeTransposeOperation`] carrier that reverse mode stages for the reverse-mode rules of a
//! [`CompositeOperation`]. The carrier holds its rules either as attached rule regions or as retained rule sets (refer
//! to [`CustomRuleSource`]), and its transposition applies the backward rule.

use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::LazyLock;

use crate::arrays::{ArrayIrOperation, ArrayOperation, ArrayType};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError, ProgramBatchingOutputAxesPolicy,
};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{
    CotangentAccumulator, CotangentBatchingPolicy, CotangentDestinationKind, DifferentiableOperation,
    DifferentiableType, DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    DifferentiationPolicy, ResidualZeroProvider, TransposableOperation, TransposedProgramCotangent,
    TranspositionContext, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, check_types};
use crate::operations::arithmetic::AddOperation;
use crate::operations::composites::call::COMPOSITE_OPERATION_NAME;
use crate::operations::composites::rules::{
    CustomRuleBackwardSpecializationKey, CustomRuleBatching, CustomRuleBatchingLevel, CustomRuleReference,
    CustomRuleSource, CustomRuleSpecializationKey, CustomRuleSpecializer, CustomRuleTracer, CustomVjpBackward,
    LiftedCustomRules, boundary_input_count, validate_rule_output_types,
};
use crate::operations::composites::{
    describe_composite_transpose, retained_carrier_batching, validate_composite_discharge_inputs,
    validate_deferred_forward_rule,
};
use crate::parameters::{Parameter, Placeholder};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    Effects, MaybeZero, Operation, OperationFormatter, OperationProvider, ProgramBuilder, ProgramError,
    ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeValue,
    ReferenceDischargeableOperation, RegionInterface, RegionSlot, Type, TypeError, TypeIdentityRenaming,
    TypeRefinements, Typed, Value, discharge_local_reference_operation, discharge_reference_free_operation,
};
use crate::tracing::TracingContext;

#[cfg(doc)]
use crate::operations::composites::call::CompositeOperation;

/// Canonical operation name for [`CompositeTransposeOperation`].
pub const COMPOSITE_TRANSPOSE_OPERATION_NAME: &str = "composite_transpose";

/// Effects of every [`CompositeTransposeOperation`] with retained rules, which carries an unresolved backward rule.
static COMPOSITE_TRANSPOSE_EFFECTS: LazyLock<Effects> = LazyLock::new(|| Effects::empty().clone().with_deferred_work());

/// Backward rule of a [`CompositeTransposeOperation`], in the representation of the call that staged it.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) enum CompositeTransposeRules<T, S> {
    /// Backward rule attached as the `"backward"` rule region.
    Attached,

    /// Backward rule retained by a source and specialized lazily, together with the carrier state that its
    /// specializations depend on.
    Retained {
        /// Source of the backward rule.
        rules: S,

        /// Batching applied to the carrier (or to the call from which it was staged), or [`None`] when it is
        /// unbatched.
        batching: Option<CustomRuleBatching<T>>,

        /// Whether transposition invokes the backward rule directly instead of replaying a cached specialization. Only
        /// the source programs from which specializations are derived set it, and their carriers are never batched.
        invokes_rule_directly: bool,

        /// Whether the carrier's reference state (or that of the call from which it was staged) was discharged, which
        /// discharges its backward rule programs as well.
        discharged: bool,

        /// Specialization key of the forward rule of the call from which the carrier was staged when that call defers
        /// its forward rule to the backward pass (refer to [`CustomRuleSource::defers_forward`]), or [`None`] when the
        /// carrier's leading inputs already contain the residuals. A deferred carrier's leading
        /// inputs are the call's inputs (including the boundary inputs of its batching levels, which the key records),
        /// followed by the seed extents, and it records no batching of its own.
        deferred_forward: Option<CustomRuleSpecializationKey<T>>,
    },
}

/// Selected reverse-mode carrier of a [`CompositeOperation`] with reverse-mode rules, which its
/// [`jvp_for_transpose`](DifferentiableOperation::jvp_for_transpose) rule stages in place of the call. It represents
/// the linear map `ẋ ↦ ẏ` from the differentiated inputs' tangents to the output tangents, whose transpose `ȳ ↦ x̄` is
/// the call's backward rule. This is the role of the `custom_lin` primitive of [JAX's automatic
/// differentiation](https://github.com/jax-ml/jax/blob/main/jax/_src/interpreters/ad.py), which has no rendered
/// documentation page and is therefore linked at its source. The map has no forward program: reverse-mode rules supply
/// only its transpose, and deriving the map itself would differentiate the primal, which the custom rules exist to
/// avoid. The carrier therefore rejects execution, lowering, and forward mode, and it exists only to be transposed. A
/// call without reverse-mode rules stages no carrier, because reverse mode transposes the linearization of its
/// forward-mode rule instead.
///
/// Its inputs are the known leading inputs (i.e., any batching boundary inputs, the non-differentiated inputs, the
/// residuals, and the seed extents) followed by the differentiated inputs' tangents. The seed extents name the runtime
/// quantities of the output tangent types (e.g., the extents of dynamically shaped outputs) that structural-zero seeds
/// need when no other leading input names them, and the backward rule never receives them. The carrier of a call whose
/// retained rules defer their forward rule to the backward pass (refer to [`CustomRuleSource::defers_forward`],
/// rendered as `deferred_forward=true`) receives every input of the call in place of the non-differentiated inputs and
/// the residuals, and its transposition first replays the forward rule on them to recover the residuals.
///
/// A [`LinearCallOperation`](crate::LinearCallOperation), by contrast, carries both an executable forward map and its
/// transpose: it executes, lowers, differentiates, and batches through its forward region, and it transposes by
/// swapping its two regions into another linear call. Keeping maps without a forward program in this carrier spares
/// every consumer of linear calls a missing-forward case, and it keeps the machinery that only retained backward rules
/// need (i.e., a rule source, deferred work, and caller-buffer destinations) in one operation.
///
/// The backward rule has the representation of the call that staged the carrier:
///
///   - **Attached** ([`Self::from_backward_region`]): the backward program is the carrier's `"backward"` rule region,
///     `(leading, ȳ) → x̄` over the leading inputs other than the seed extents, which transposition replays inline into
///     the pullback, since a user-supplied backward program has no linearity contract of its own that would let it be
///     transposed again. The region's own effects decide whether transposition must run it when no cotangent is
///     requested. It is a deferred rule region (refer to
///     [`RegionRole::DeferredRule`](crate::RegionRole::DeferredRule)), so a backward region with effects or deferred
///     work makes the carrier carry deferred work, which keeps simplification from removing an unused carrier before
///     transposition runs those effects.
///   - **Retained** ([`Self::new`]): the backward rule is specialized per [`CustomRuleBackwardSpecializationKey`] by
///     transposing a source program that invokes the rule directly, cached in the definition, and replayed with the
///     actual known inputs, seeds, and caller-buffer destinations. The runtime buffer identities are invocation inputs
///     and never part of a specialization. The carrier declares deferred work (refer to the
///     [Deferred Work](crate::Effects#deferred-work) section of [`Effects`]), so that shared transform boundaries
///     retain it until transposition replaces it with the specialized backward program.
///
/// # Batching
///
/// A carrier with an attached backward rule is preserved unchanged when every input is replicated at an unnamed
/// batching level, which no attached region can observe, and it is otherwise rejected, because it has no forward
/// program to determine the batch axes of its outputs. A carrier with a retained backward rule that was staged from a
/// batched call, or that is batched itself (e.g., when a linearization is batched), records its batching levels as the
/// call does, mapping every output at axis 0 when it is batched itself. Transposition derives the unbatched backward
/// specialization and batches it once per level. Seeds, caller buffers, and leading inputs keep their recorded batch
/// axes, and each returned cotangent is aligned to its input, summing the per-item cotangents of a replicated input. A
/// carrier with a deferred forward rule records the batching levels of the call from which it was staged instead, and
/// it cannot be batched itself, because its transposition would have to batch the replayed forward rule as well.
pub struct CompositeTransposeOperation<V: Typed + Parameter, O, S = CustomRuleReference<V, O>> {
    /// Number of leading known inputs (i.e., boundary inputs, non-differentiated inputs, residuals, and seed extents).
    pub(super) leading_input_count: usize,

    /// Number of trailing leading inputs that carry the runtime extents of the output tangent types (e.g., the extents
    /// of dynamically shaped outputs), from which structural-zero seeds are materialized when no other leading input
    /// carries them. The backward rule never receives them.
    pub(super) seed_extent_count: usize,

    /// Tangent types of the differentiated inputs.
    pub(super) input_tangent_types: Vec<V::Type>,

    /// Tangent types of the outputs.
    pub(super) output_tangent_types: Vec<V::Type>,

    /// Backward rule of the carrier.
    pub(super) rules: CompositeTransposeRules<V::Type, S>,

    /// Marker for the operation family, which only the rule source and the rule region name.
    pub(super) marker: PhantomData<fn() -> O>,
}

impl<V: Typed<Type: DifferentiableType + Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>>
    CompositeTransposeOperation<V, O, S>
{
    /// Creates a new [`CompositeTransposeOperation`] that applies the retained backward rule of the provided rules to
    /// an unbatched, undischarged call without seed extents. This is the carrier that the
    /// [`jvp_for_transpose`](DifferentiableOperation::jvp_for_transpose) rule of a [`CompositeOperation`] with retained
    /// rules stages for such a call, whose dead outputs need no runtime quantities beyond those that the leading inputs
    /// name. Carriers with seed extents, batching levels, or discharged reference state are staged only by the
    /// transforms that produce them.
    ///
    /// # Parameters
    ///
    ///   - `rules`: Source of the backward rule.
    ///   - `leading_input_count`: Number of leading known inputs (i.e., the non-differentiated inputs followed by the
    ///     residuals).
    ///   - `input_tangent_types`: Tangent types of the differentiated inputs, which follow the leading inputs.
    ///   - `output_tangent_types`: Tangent types of the outputs.
    pub fn new(
        rules: S,
        leading_input_count: usize,
        input_tangent_types: Vec<V::Type>,
        output_tangent_types: Vec<V::Type>,
    ) -> Self {
        Self {
            leading_input_count,
            seed_extent_count: 0,
            input_tangent_types,
            output_tangent_types,
            rules: CompositeTransposeRules::Retained {
                rules,
                batching: None,
                invokes_rule_directly: false,
                discharged: false,
                deferred_forward: None,
            },
            marker: PhantomData,
        }
    }

    /// Creates a new [`CompositeTransposeOperation`] whose backward rule is its attached `"backward"` rule region,
    /// exactly as the [`jvp_for_transpose`](DifferentiableOperation::jvp_for_transpose) rule of a
    /// [`CompositeOperation`] with attached rules stages it, except that it has no seed extents. The region maps
    /// the leading inputs followed by one cotangent per output to one cotangent per differentiated input. When the zero
    /// seeds of dead outputs need runtime quantities (e.g., dynamic extents) that no leading input names, the region
    /// can receive them as additional leading inputs.
    ///
    /// # Parameters
    ///
    ///   - `leading_input_count`: Number of leading known inputs (i.e., the non-differentiated inputs followed by the
    ///     residuals).
    ///   - `input_tangent_types`: Tangent types of the differentiated inputs, which follow the leading inputs.
    ///   - `output_tangent_types`: Tangent types of the outputs.
    pub fn from_backward_region(
        leading_input_count: usize,
        input_tangent_types: Vec<V::Type>,
        output_tangent_types: Vec<V::Type>,
    ) -> Self {
        Self {
            leading_input_count,
            seed_extent_count: 0,
            input_tangent_types,
            output_tangent_types,
            rules: CompositeTransposeRules::Attached,
            marker: PhantomData,
        }
    }

    /// Creates the source carrier from which the backward specializations of `rules` are derived, whose
    /// transposition invokes the retained backward rule directly.
    pub(super) fn direct(
        rules: S,
        leading_input_count: usize,
        seed_extent_count: usize,
        input_tangent_types: Vec<V::Type>,
        output_tangent_types: Vec<V::Type>,
        discharged: bool,
    ) -> Self {
        Self {
            leading_input_count,
            seed_extent_count,
            input_tangent_types,
            output_tangent_types,
            rules: CompositeTransposeRules::Retained {
                rules,
                batching: None,
                invokes_rule_directly: true,
                discharged,
                deferred_forward: None,
            },
            marker: PhantomData,
        }
    }

    /// Returns the number of leading known inputs of this carrier (i.e., any batching boundary inputs, the
    /// non-differentiated inputs, the residuals, and the seed extents), which includes its
    /// [`seed_extent_count`](Self::seed_extent_count) trailing seed extents.
    #[inline]
    pub fn leading_input_count(&self) -> usize {
        self.leading_input_count
    }

    /// Returns the number of trailing leading inputs of this carrier that carry the runtime extents of its output
    /// tangent types, from which structural-zero seeds are materialized and which the backward rule never receives.
    #[inline]
    pub fn seed_extent_count(&self) -> usize {
        self.seed_extent_count
    }

    /// Returns the tangent types of the differentiated inputs of this carrier.
    #[inline]
    pub fn input_tangent_types(&self) -> &[V::Type] {
        &self.input_tangent_types
    }

    /// Returns the tangent types of the outputs of this carrier.
    #[inline]
    pub fn output_tangent_types(&self) -> &[V::Type] {
        &self.output_tangent_types
    }

    /// Returns the source of the retained backward rule of this carrier, or [`None`] when its backward rule is
    /// attached.
    #[inline]
    pub fn rules(&self) -> Option<&S> {
        match &self.rules {
            CompositeTransposeRules::Attached => None,
            CompositeTransposeRules::Retained { rules, .. } => Some(rules),
        }
    }

    /// Converts this carrier into the family `(V2, O2)` with the rule source that `rules_fn` derives from its current
    /// source (refer to [`CompositeOperation::into_family`]).
    pub fn into_family<V2, O2, S2, F>(self, rules_fn: F) -> CompositeTransposeOperation<V2, O2, S2>
    where
        V2: Typed<Type: DifferentiableType + Eq + Hash + From<V::Type>> + Parameter,
        F: FnOnce(S) -> S2,
    {
        let map_types = |types: Vec<V::Type>| types.into_iter().map(V2::Type::from).collect::<Vec<_>>();
        CompositeTransposeOperation {
            leading_input_count: self.leading_input_count,
            seed_extent_count: self.seed_extent_count,
            input_tangent_types: map_types(self.input_tangent_types),
            output_tangent_types: map_types(self.output_tangent_types),
            rules: match self.rules {
                CompositeTransposeRules::Attached => CompositeTransposeRules::Attached,
                CompositeTransposeRules::Retained {
                    rules,
                    batching,
                    invokes_rule_directly,
                    discharged,
                    deferred_forward,
                } => CompositeTransposeRules::Retained {
                    rules: rules_fn(rules),
                    batching: batching.map(CustomRuleBatching::map_types),
                    invokes_rule_directly,
                    discharged,
                    deferred_forward: deferred_forward.map(CustomRuleSpecializationKey::map_types),
                },
            },
            marker: PhantomData,
        }
    }

    /// Converts this carrier into the family `(V2, O2)` with any rule source type when its backward rule is attached,
    /// or returns it unchanged when its backward rule is retained (refer to
    /// [`CompositeOperation::into_attached_family`]).
    pub fn into_attached_family<V2, O2, S2>(self) -> Result<CompositeTransposeOperation<V2, O2, S2>, Self>
    where
        V2: Typed<Type: From<V::Type>> + Parameter,
    {
        match self.rules {
            CompositeTransposeRules::Attached => {
                let map_types = |types: Vec<V::Type>| types.into_iter().map(V2::Type::from).collect::<Vec<_>>();
                Ok(CompositeTransposeOperation {
                    leading_input_count: self.leading_input_count,
                    seed_extent_count: self.seed_extent_count,
                    input_tangent_types: map_types(self.input_tangent_types),
                    output_tangent_types: map_types(self.output_tangent_types),
                    rules: CompositeTransposeRules::Attached,
                    marker: PhantomData,
                })
            }
            CompositeTransposeRules::Retained { .. } => Err(self),
        }
    }

    /// Returns a description of this carrier for diagnostics, which names the definition of a retained backward rule.
    fn description(&self) -> String {
        match &self.rules {
            CompositeTransposeRules::Attached => format!("`{COMPOSITE_TRANSPOSE_OPERATION_NAME}`"),
            CompositeTransposeRules::Retained { rules, .. } => describe_composite_transpose(rules.name()),
        }
    }

    /// Returns the input types of an attached backward region: the types of the leading inputs among `input_types`,
    /// except for the seed extents, followed by the cotangent types of the outputs.
    fn backward_input_types(&self, input_types: &[V::Type]) -> Result<Vec<V::Type>, TypeError> {
        if self.leading_input_count > input_types.len() {
            return Err(TypeError::invalid(format!(
                "`{}` leading input count {} exceeds input count {}",
                COMPOSITE_TRANSPOSE_OPERATION_NAME,
                self.leading_input_count,
                input_types.len(),
            )));
        }
        let Some(rule_leading_input_count) = self.leading_input_count.checked_sub(self.seed_extent_count) else {
            return Err(TypeError::invalid(format!(
                "`{}` seed extent count {} exceeds leading input count {}",
                COMPOSITE_TRANSPOSE_OPERATION_NAME, self.seed_extent_count, self.leading_input_count,
            )));
        };
        let mut backward_input_types = input_types[..rule_leading_input_count].to_vec();
        backward_input_types.extend(
            self.output_tangent_types.iter().map(DifferentiableType::cotangent).collect::<Result<Vec<_>, _>>()?,
        );
        Ok(backward_input_types)
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> Clone
    for CompositeTransposeOperation<V, O, S>
{
    fn clone(&self) -> Self {
        Self {
            leading_input_count: self.leading_input_count,
            seed_extent_count: self.seed_extent_count,
            input_tangent_types: self.input_tangent_types.clone(),
            output_tangent_types: self.output_tangent_types.clone(),
            rules: self.rules.clone(),
            marker: PhantomData,
        }
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> Debug
    for CompositeTransposeOperation<V, O, S>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // A retained backward rule is identified by the name of its definition, whose own rendering would include its
        // rules.
        let mut debug = formatter.debug_struct("CompositeTransposeOperation");
        if let CompositeTransposeRules::Retained { rules, .. } = &self.rules {
            debug.field("name", &rules.name());
        }
        debug
            .field("leading_input_count", &self.leading_input_count)
            .field("seed_extent_count", &self.seed_extent_count)
            .field("input_tangent_types", &self.input_tangent_types)
            .field("output_tangent_types", &self.output_tangent_types);
        if let CompositeTransposeRules::Retained {
            batching, invokes_rule_directly, discharged, deferred_forward, ..
        } = &self.rules
        {
            debug
                .field("batching", batching)
                .field("invokes_rule_directly", invokes_rule_directly)
                .field("discharged", discharged);
            if let Some(key) = deferred_forward {
                debug.field("deferred_forward", key);
            }
        }
        debug.finish()
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> PartialEq
    for CompositeTransposeOperation<V, O, S>
{
    fn eq(&self, other: &Self) -> bool {
        self.leading_input_count == other.leading_input_count
            && self.seed_extent_count == other.seed_extent_count
            && self.input_tangent_types == other.input_tangent_types
            && self.output_tangent_types == other.output_tangent_types
            && self.rules == other.rules
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> Eq for CompositeTransposeOperation<V, O, S> {}

impl<V: Typed<Type: Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> Hash
    for CompositeTransposeOperation<V, O, S>
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.leading_input_count.hash(state);
        self.seed_extent_count.hash(state);
        self.input_tangent_types.hash(state);
        self.output_tangent_types.hash(state);
        self.rules.hash(state);
    }
}

impl<V: Typed<Type: DifferentiableType + Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> Display
    for CompositeTransposeOperation<V, O, S>
{
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<V: Typed<Type: DifferentiableType + Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> Operation
    for CompositeTransposeOperation<V, O, S>
{
    type Type = V::Type;

    #[inline]
    fn name(&self) -> &'static str {
        COMPOSITE_TRANSPOSE_OPERATION_NAME
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        match self.rules {
            CompositeTransposeRules::Attached => const { &[RegionSlot::deferred_rule("backward")] },
            CompositeTransposeRules::Retained { .. } => &[],
        }
    }

    fn infer_region_input_types(
        &self,
        input_types: &[V::Type],
        region_interfaces: &[RegionInterface<V::Type>],
    ) -> Result<Vec<Option<Vec<V::Type>>>, TypeError> {
        // The attached backward region receives the leading inputs followed by one cotangent per output.
        check_count!("region", region_interfaces, self.region_slots().len(), TypeError);
        match self.rules {
            CompositeTransposeRules::Attached => Ok(vec![Some(self.backward_input_types(input_types)?)]),
            CompositeTransposeRules::Retained { .. } => Ok(Vec::new()),
        }
    }

    fn infer_output_types(
        &self,
        input_types: &[V::Type],
        region_interfaces: &[RegionInterface<V::Type>],
    ) -> Result<Vec<V::Type>, TypeError> {
        check_count!("region", region_interfaces, self.region_slots().len(), TypeError);
        check_count!("input", input_types, self.leading_input_count + self.input_tangent_types.len(), TypeError);
        check_types!(@same, format!("`{COMPOSITE_TRANSPOSE_OPERATION_NAME}` tangent input"), [
            self.input_tangent_types.as_slice(),
            &input_types[self.leading_input_count..],
        ]);
        if let CompositeTransposeRules::Attached = self.rules {
            let backward = &region_interfaces[0];
            check_types!(@same, format!("`{COMPOSITE_TRANSPOSE_OPERATION_NAME}` backward rule input"), [
                &self.backward_input_types(input_types)?,
                backward.input_types(),
            ]);

            // Explicit dimension inputs retain their nominal identities even when their bounds prove an exact extent.
            // Consequently, specializing `linear: f32[n]` to `f32[2]` can leave the backward region's
            // `zero[n](extent)` output typed as `f32[n]`, with `n` now constrained to exactly 2. Require refinement in
            // both directions to accept these equivalent representations without accepting a merely broader result.
            // In particular, `f32[n]` with non-singleton bounds, a different extent, or unrelated non-exact identities
            // still fail.
            let cotangent_types =
                self.input_tangent_types.iter().map(DifferentiableType::cotangent).collect::<Result<Vec<_>, _>>()?;
            if cotangent_types != backward.output_types()
                && (<V::Type as Type>::Refinements::new(cotangent_types.iter(), backward.output_types().iter())
                    .is_err()
                    || <V::Type as Type>::Refinements::new(backward.output_types().iter(), cotangent_types.iter())
                        .is_err())
            {
                check_types!(@same, format!("`{COMPOSITE_TRANSPOSE_OPERATION_NAME}` backward rule output"), [
                    &cotangent_types,
                    backward.output_types(),
                ]);
            }
        }
        Ok(self.output_tangent_types.clone())
    }

    #[inline]
    fn effects(&self) -> Cow<'_, Effects> {
        // A retained backward rule is an unresolved obligation of the carrier itself. An attached backward region is a
        // deferred rule region instead (refer to `region_slots`), which makes the carrier carry deferred work exactly
        // when the region has effects or deferred work of its own.
        match self.rules {
            CompositeTransposeRules::Attached => Cow::Borrowed(Effects::empty()),
            CompositeTransposeRules::Retained { .. } => Cow::Borrowed(&COMPOSITE_TRANSPOSE_EFFECTS),
        }
    }

    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<V::Type as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        let rename = |types: &[V::Type]| {
            types.iter().map(|r#type| r#type.rename_identities(renaming)).collect::<Result<Vec<_>, _>>()
        };
        let rules = match &self.rules {
            CompositeTransposeRules::Attached => CompositeTransposeRules::Attached,
            CompositeTransposeRules::Retained {
                rules,
                batching,
                invokes_rule_directly,
                discharged,
                deferred_forward,
            } => CompositeTransposeRules::Retained {
                rules: rules.clone(),
                batching: batching.as_ref().map(|batching| batching.rename_identities(renaming)).transpose()?,
                invokes_rule_directly: *invokes_rule_directly,
                discharged: *discharged,
                deferred_forward: deferred_forward.as_ref().map(|key| key.rename_identities(renaming)).transpose()?,
            },
        };
        Ok(Self {
            leading_input_count: self.leading_input_count,
            seed_extent_count: self.seed_extent_count,
            input_tangent_types: rename(&self.input_tangent_types)?,
            output_tangent_types: rename(&self.output_tangent_types)?,
            rules,
            marker: PhantomData,
        })
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        // The tangent types coincide with the types of the carrier's tangent inputs and outputs, so they are not
        // rendered.
        let operation = OperationFormatter::new(formatter, indentation, COMPOSITE_TRANSPOSE_OPERATION_NAME)?;
        operation.bracketed(|operation| {
            if let CompositeTransposeRules::Retained { rules, .. } = &self.rules {
                operation.field("name", format_args!("{:?}", rules.name()))?;
            }
            operation.field("leading_input_count", self.leading_input_count)?;
            if self.seed_extent_count != 0 {
                operation.field("seed_extent_count", self.seed_extent_count)?;
            }
            match &self.rules {
                CompositeTransposeRules::Attached => Ok(()),
                CompositeTransposeRules::Retained {
                    batching,
                    invokes_rule_directly,
                    discharged,
                    deferred_forward,
                    ..
                } => {
                    if let Some(batching) = batching {
                        operation.field("batching", batching)?;
                    }
                    if *invokes_rule_directly {
                        operation.field("direct", true)?;
                    }
                    if *discharged {
                        operation.field("discharged", true)?;
                    }
                    if deferred_forward.is_some() {
                        operation.field("deferred_forward", true)?;
                    }
                    Ok(())
                }
            }
        })
    }
}

impl<C, P, V, O, S> ReferenceDischargeableOperation<C, P> for CompositeTransposeOperation<V, O, S>
where
    C: Context<Operation: From<CompositeTransposeOperation<V, O, S>>>,
    P: ReferenceDischargePolicy<C>,
    V: Typed<Type: DifferentiableType + Eq + Hash> + Parameter,
    S: CustomRuleSource<V, O>,
    CompositeTransposeOperation<V, O, S>: Operation<Type = C::Type>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        // An attached backward region discharges its local reference state now. A carrier with a retained backward
        // rule holds no reference state of its own, and its rebound form records that its lazily traced backward rule
        // programs (and its deferred forward rule programs, if any) must be discharged once they are traced.
        let name = match &self.rules {
            CompositeTransposeRules::Attached => COMPOSITE_OPERATION_NAME,
            CompositeTransposeRules::Retained { rules, .. } => rules.name(),
        };
        validate_composite_discharge_inputs(&describe_composite_transpose(name), inputs)?;
        match &self.rules {
            CompositeTransposeRules::Attached => discharge_local_reference_operation(self, context, driver, inputs),
            CompositeTransposeRules::Retained { rules, batching, invokes_rule_directly, deferred_forward, .. } => {
                let discharged = Self {
                    rules: CompositeTransposeRules::Retained {
                        rules: rules.clone(),
                        batching: batching.clone(),
                        invokes_rule_directly: *invokes_rule_directly,
                        discharged: true,
                        deferred_forward: deferred_forward
                            .as_ref()
                            .map(|key| CustomRuleSpecializationKey { discharged: true, ..key.clone() }),
                    },
                    ..self.clone()
                };
                discharge_reference_free_operation(&discharged, context, driver, inputs)
            }
        }
    }
}

impl<
    C: Domain<Type = V::Type>,
    V: Typed<Type: DifferentiableType + Eq + Hash> + Parameter,
    O,
    S: CustomRuleSource<V, O>,
> InterpretableOperation<C> for CompositeTransposeOperation<V, O, S>
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        _inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "{} has no forward program to execute; it supports only reverse-mode differentiation (e.g., `vjp`, \
                 `value_and_gradient`, or `jacobian_reverse`)",
                self.description(),
            ),
        })
    }
}

impl<C: Context<Type: DifferentiableType + Eq + Hash>, S: CustomRuleSource<C::Constant, C::Operation>>
    PartiallyEvaluatableOperation<C> for CompositeTransposeOperation<C::Constant, C::Operation, S>
where
    C::Operation: From<CompositeTransposeOperation<C::Constant, C::Operation, S>>,
{
    // A carrier always has an unknown tangent input when it is staged, and a retained backward rule's deferred work is
    // always residualized by the shared folding boundary, so the default rule suffices.
}

impl<C, P, S> BatchableOperation<C, P> for CompositeTransposeOperation<C::Constant, C::Operation, S>
where
    C: Context<
            Type: DifferentiableType + Eq + Hash,
            Operation: From<CompositeTransposeOperation<C::Constant, C::Operation, S>>,
        >,
    P: CotangentBatchingPolicy<C>,
    S: CustomRuleSource<C::Constant, C::Operation>,
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<Vec<P::Batch>, BatchingError> {
        check_count!("input", inputs, self.leading_input_count + self.input_tangent_types.len(), ProgramError);
        let input_axes = inputs.iter().map(P::batch_axis).collect::<Vec<_>>();
        let (rules, batching, discharged) = match &self.rules {
            CompositeTransposeRules::Attached => {
                // A completely replicated carrier at an unnamed batching level needs no structural region rewrite, and
                // keeping it avoids manufacturing a batch axis that its backward region does not observe. What makes
                // this shortcut sound is the level being unnamed: a region's value can vary per batch item with no
                // mapped input only by addressing the level by name (e.g., an `axis_index` or a collective over the
                // level's axis). Any other carrier is rejected, because no forward program determines the batch axes of
                // its outputs.
                if input_axes.iter().all(BatchAxis::is_replicated) && context.axis_name().is_none() {
                    let outputs = context.parent().bind(
                        self.clone(),
                        driver.region_programs(),
                        inputs.iter().map(P::value).cloned().collect::<Vec<_>>().as_slice(),
                    )?;
                    return Ok(outputs.into_iter().map(P::replicated).collect::<Vec<_>>());
                }
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "a `{COMPOSITE_TRANSPOSE_OPERATION_NAME}` with an attached backward rule cannot be \
                         batched structurally, because it has no forward program to determine the batch axes of its \
                         outputs; it is preserved unchanged only when every input is replicated at an unnamed \
                         batching level",
                    ),
                });
            }
            CompositeTransposeRules::Retained { invokes_rule_directly: true, rules, .. } => {
                return Err(ProgramError::MalformedProgram(format!(
                    "{} source carriers are never batched",
                    describe_composite_transpose(rules.name()),
                ))
                .into());
            }

            // Transposing a deferred carrier replays the forward rule at the batching levels of the call from which it
            // was staged, which its forward key records, while a level added to the carrier itself would also have to
            // batch that replay over the carrier's leading inputs, whose batch axes no forward key describes.
            CompositeTransposeRules::Retained { deferred_forward: Some(_), rules, .. } => {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "{} defers its forward rule to the backward pass (refer to `with_deferred_forward`), so it \
                         cannot be batched after it is differentiated; batch the call before differentiating it, or \
                         transpose the derivative before batching it",
                        describe_composite_transpose(rules.name()),
                    ),
                });
            }
            CompositeTransposeRules::Retained { rules, batching, discharged, .. } => (rules, batching, *discharged),
        };

        // Without a forward program, the natural output layout is unknown, so every output is mapped at axis 0. The
        // batched output types are those of an identity program over the current output types at that layout.
        let output_axes = vec![BatchAxis::new(0); self.output_tangent_types.len()];
        let mut builder = ProgramBuilder::<C::Constant, C::Operation>::new();
        let identity_inputs =
            self.output_tangent_types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let identity = builder.build::<Vec<C::Constant>, Vec<C::Constant>>(
            identity_inputs,
            vec![Placeholder; output_axes.len()],
            vec![Placeholder; output_axes.len()],
        )?;
        let batched_identity = driver.batch_program(
            context,
            identity.entry_region_ref(),
            output_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::AlignEachTo(output_axes.clone()),
        )?;
        let batched_identity = context.align_and_adapt_batched_program_outputs(
            driver,
            identity.entry_region_ref(),
            output_axes.as_slice(),
            batched_identity,
            output_axes.as_slice(),
        )?;

        let boundary_inputs = P::boundary_inputs(context.axis_extent());
        let mut batching = batching.clone().unwrap_or_else(|| CustomRuleBatching {
            input_types: inputs.iter().map(|input| P::unbatched_type(input).into_owned()).collect(),
            output_types: self.output_tangent_types.clone(),
            levels: Vec::new(),
        });
        batching.levels.push(CustomRuleBatchingLevel {
            level: driver.batching_level(context)?,
            boundary_input_count: boundary_inputs.len(),
            input_axes,
            output_axes: output_axes.clone(),
        });
        let carrier = Self {
            leading_input_count: self.leading_input_count + boundary_inputs.len(),
            seed_extent_count: self.seed_extent_count,
            input_tangent_types: inputs[self.leading_input_count..]
                .iter()
                .map(|input| P::value(input).r#type().into_owned())
                .collect(),
            output_tangent_types: batched_identity.output_types(),
            rules: CompositeTransposeRules::Retained {
                rules: rules.clone(),
                batching: Some(batching),
                invokes_rule_directly: false,
                discharged,
                deferred_forward: None,
            },
            marker: PhantomData,
        };
        let mut packed_inputs = boundary_inputs;
        packed_inputs.extend(inputs.iter().map(P::value).cloned());
        let outputs = context.parent().bind(carrier, Vec::new(), packed_inputs.as_slice())?;
        check_count!("output", outputs, output_axes.len(), ProgramError);
        Ok(outputs
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| P::batch(output, axis))
            .collect::<Result<Vec<_>, _>>()?)
    }
}

impl<C: Context<Type: DifferentiableType + Eq + Hash>, S: CustomRuleSource<C::Constant, C::Operation>>
    DifferentiableOperation<C> for CompositeTransposeOperation<C::Constant, C::Operation, S>
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        _context: &DifferentiationContext<C, P>,
        _driver: &D,
        _inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "{} has no forward-mode (i.e., JVP) rule; it supports only reverse-mode differentiation",
                self.description(),
            ),
        }
        .into())
    }
}

impl<V, O, S> TransposableOperation<V, O> for CompositeTransposeOperation<V, O, S>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: Value<Type: DifferentiableType + Eq + Hash>,
    O: Operation<Type = V::Type>
        + From<CompositeTransposeOperation<V, O>>
        + ResidualZeroProvider<V::Type, Operation = O>
        + OperationProvider<V::Type, AddOperation<V::Type>, Operation = O>,
    S: CustomRuleSpecializer<V, O>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<CustomRuleTracer<V, O>>],
        outputs: &[MaybeZero<CustomRuleTracer<V, O>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        check_count!("input", inputs, self.leading_input_count + self.input_tangent_types.len(), ProgramError);
        check_count!("output", outputs, self.output_tangent_types.len(), ProgramError);
        check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
        let mut leading_values = Vec::with_capacity(self.leading_input_count);
        for (index, input) in inputs[..self.leading_input_count].iter().enumerate() {
            leading_values.push(input.as_known().cloned().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "`{COMPOSITE_TRANSPOSE_OPERATION_NAME}` leading input {index} is not known during \
                     transposition",
                ))
            })?);
        }
        let seed_extent_count = self.seed_extent_count;
        let (rules, batching, invokes_rule_directly, discharged, deferred_forward) = match &self.rules {
            CompositeTransposeRules::Attached => {
                // Unrequested gradients can omit a pure backward program, but its observable effects must still
                // execute and its deferred work must still be staged even when all seeds are structural zeros or none
                // of the differentiated inputs requests a cotangent. The backward program is selected here, so its own
                // summary decides; deferred work that exists only in its dormant alternatives (i.e., nested rule
                // regions) creates no obligation.
                let backward = driver.region(0)?;
                if (outputs.iter().all(MaybeZero::is_zero) || !accumulators.iter().any(CotangentAccumulator::is_needed))
                    && !backward.effects().is_retained_when_unused()
                {
                    return Ok(());
                }

                // Classify each backward-region output as structurally zero by inspecting its producing instruction,
                // so zero cotangents stay symbolic instead of accumulating. An output that forwards a seed input is
                // zero exactly when that seed is a structural zero, which this carrier materializes below only because
                // the backward region needs a value. Outputs that forward a leading input or a constant conservatively
                // classify as non-zero.
                let seed_input_ids = &backward.input_ids()[self.leading_input_count - seed_extent_count..];
                let output_is_zero = backward
                    .output_ids()
                    .iter()
                    .map(|output| {
                        if let Some(seed_index) = seed_input_ids.iter().position(|input| input == output) {
                            return outputs[seed_index].is_zero();
                        }
                        backward
                            .instructions()
                            .iter()
                            .find_map(|instruction| {
                                instruction
                                    .outputs()
                                    .iter()
                                    .position(|candidate| candidate == output)
                                    .map(|output_index| instruction.operation().is_zero(output_index))
                            })
                            .unwrap_or(false)
                    })
                    .collect::<Vec<_>>();

                // A dead output's structural-zero seed still becomes a real input of the backward region. Its type
                // alone cannot construct it when it references runtime identities, but the live seeds and the leading
                // inputs, whose seed extents name every runtime quantity of the output tangent types, do, so the zero
                // is assembled from them one identity at a time before falling back to the nullary zero that every
                // identity-free type keeps. The backward region never receives the seed extents.
                let seeds = outputs
                    .iter()
                    .cloned()
                    .map(|output| {
                        O::materialize_zero_from_residual_sources(
                            &**context,
                            output,
                            outputs.iter().filter_map(MaybeZero::as_value).chain(&leading_values),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let mut backward_inputs = leading_values;
                backward_inputs.truncate(self.leading_input_count - seed_extent_count);
                backward_inputs.extend(seeds);

                // A user-supplied backward program has no linearity contract of its own, so it cannot be transposed
                // again and is replayed inline into the pullback.
                let cotangents = backward.interpret_in_context(&**context, backward_inputs)?;
                check_count!("output", cotangents, self.input_tangent_types.len(), ProgramError);

                // Leading inputs are known and contribute nothing. Preserve structural zeros reported by the
                // backward program, including zeros whose runtime extents belong to the caller's primal inputs.
                return inputs[self.leading_input_count..]
                    .iter()
                    .zip(cotangents)
                    .zip(output_is_zero)
                    .zip(&accumulators[self.leading_input_count..])
                    .filter(|(((input, _), is_zero), _)| input.is_unknown() && !is_zero)
                    .try_for_each(|(((_, cotangent), _), accumulator)| {
                        accumulator.accumulate(context, MaybeZero::Value(cotangent))
                    });
            }
            CompositeTransposeRules::Retained {
                rules,
                batching,
                invokes_rule_directly,
                discharged,
                deferred_forward,
            } => (rules, batching, *invokes_rule_directly, *discharged, deferred_forward),
        };

        // Without leading inputs, the backward rule is a linear function of the seeds alone, so all-zero seeds give
        // structurally zero cotangents without tracing the rule, which could not even recover its context from its
        // inputs (as for an unused output of a call without residuals or non-differentiated inputs).
        if self.leading_input_count == 0 && outputs.iter().all(MaybeZero::is_zero) {
            return Ok(());
        }
        if invokes_rule_directly {
            let Some(backward) = rules.native_definition().and_then(|definition| definition.backward.as_ref()) else {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!("{} has no backward rule", describe_composite_transpose(rules.name())),
                }
                .into());
            };

            // The seed extents are used only to materialize zero seeds, so the rule sees neither those inputs nor their
            // accumulators.
            let rule_leading_input_count = self.leading_input_count - seed_extent_count;
            let rule_leading_values = &leading_values[..rule_leading_input_count];
            let cotangents = match backward {
                CustomVjpBackward::Accumulating(rule) => {
                    if seed_extent_count == 0 {
                        return Ok(rule.apply(context, inputs, outputs, accumulators)?);
                    }
                    let inputs = inputs[..rule_leading_input_count]
                        .iter()
                        .chain(&inputs[self.leading_input_count..])
                        .cloned()
                        .collect::<Vec<_>>();
                    let accumulators = accumulators[..rule_leading_input_count]
                        .iter()
                        .chain(&accumulators[self.leading_input_count..])
                        .cloned()
                        .collect::<Vec<_>>();
                    return Ok(rule.apply(context, &inputs, outputs, &accumulators)?);
                }

                // A pure rule receives the known leading inputs and materialized seeds, and its cotangents are added
                // to each differentiated input's destination, which also discards the cotangents of ignored inputs.
                // A zero seed whose type names runtime extents reads them from the leading inputs, including the seed
                // extents.
                CustomVjpBackward::Pure(rule) => {
                    let seeds = outputs
                        .iter()
                        .map(|seed| {
                            O::materialize_zero_from_residual_sources(&**context, seed.clone(), &leading_values)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    rule.apply(rule_leading_values, &seeds)?.into_iter().map(MaybeZero::Value).collect::<Vec<_>>()
                }

                // A rule that receives structural-zero seeds may also return structural-zero cotangents, which
                // contribute nothing to their destinations, so they stay structural zeros for the transposition.
                CustomVjpBackward::PureWithSymbolicZeros(rule) => rule.apply(rule_leading_values, outputs)?,
            };

            // Each cotangent has the cotangent type of its differentiated input, which may differ from the input's
            // tangent type (e.g., sharded arrays swap their reduced and unreduced axes). Structural zeros are checked
            // too, so that a misplaced zero cannot pass as the cotangent of another input.
            let cotangent_types =
                cotangents.iter().map(|cotangent| cotangent.r#type().into_owned()).collect::<Vec<_>>();
            let expected_cotangent_types =
                self.input_tangent_types.iter().map(DifferentiableType::cotangent).collect::<Result<Vec<_>, _>>()?;
            validate_rule_output_types(
                rules.name(),
                "backward",
                expected_cotangent_types.as_slice(),
                cotangent_types.as_slice(),
            )?;
            for (accumulator, cotangent) in accumulators[self.leading_input_count..].iter().zip(cotangents) {
                accumulator.accumulate(context, cotangent)?;
            }
            return Ok(());
        }

        // The leading inputs of a deferred carrier are the inputs of its call rather than the call's residuals, so the
        // forward rule is replayed on them here, in the backward pass, to recover the residuals. The backward rule is
        // then applied exactly as for the carrier that the call would have staged without deferring its forward rule:
        // its leading inputs become the call's boundary and non-differentiated inputs, the residuals, and the seed
        // extents, and its batching levels are derived from those that the call recorded in the forward key.
        let deferred_batching;
        let batching = match deferred_forward {
            None => batching,
            Some(key) => {
                let call_input_count = self.leading_input_count - seed_extent_count;
                let call_boundary_input_count = boundary_input_count(&key.levels);
                if call_input_count != call_boundary_input_count + key.input_types.len() {
                    return Err(ProgramError::MalformedProgram(format!(
                        "{} has {call_input_count} call inputs but its deferred forward rule expects {} boundary \
                         inputs and {} call inputs",
                        describe_composite_transpose(rules.name()),
                        call_boundary_input_count,
                        key.input_types.len(),
                    ))
                    .into());
                }
                let rule_output_count = key.rule_output_count()?;
                let forward = rules.forward_specialization(key.clone())?;
                validate_deferred_forward_rule(rules.name(), forward.program.effects())?;

                // Only the residuals are replayed, because the primal outputs of the forward rule were already computed
                // by the primal and the pullback would otherwise keep their dead computation.
                let residual_indices = (rule_output_count..forward.program.output_ids().len()).collect::<Vec<_>>();
                let residuals = forward
                    .program
                    .with_outputs(&residual_indices)?
                    .interpret_in_context(&**context, leading_values[..call_input_count].to_vec())?;
                let seed_extents = leading_values.split_off(call_input_count);
                leading_values.truncate(call_boundary_input_count + key.non_differentiated_count);
                leading_values.extend(residuals);
                leading_values.extend(seed_extents);
                deferred_batching = match key.levels.is_empty() {
                    true => None,
                    false => {
                        let seed_extent_types = leading_values[leading_values.len() - seed_extent_count..]
                            .iter()
                            .map(|value| value.r#type().into_owned())
                            .collect::<Vec<_>>();
                        let extent_types =
                            key.output_types.iter().map(DifferentiableType::tangent).collect::<Result<Vec<_>, _>>()?;
                        Some(retained_carrier_batching(
                            rules,
                            key,
                            rule_output_count,
                            &seed_extent_types,
                            extent_types,
                        )?)
                    }
                };
                &deferred_batching
            }
        };

        // Key the specialization by static information only: the unbatched leading input types, the carrier's tangent
        // signature, which seeds are live, where each differentiated input's cotangent goes, and the batching levels.
        // Caller buffers stay invocation inputs.
        let mut references = Vec::new();
        let mut destination_kinds = Vec::with_capacity(self.input_tangent_types.len());
        for accumulator in &accumulators[self.leading_input_count..] {
            let reference = accumulator.reference(context)?;
            destination_kinds.push(if !accumulator.is_needed() {
                CotangentDestinationKind::Ignore
            } else if reference.is_some() {
                CotangentDestinationKind::Reference
            } else {
                CotangentDestinationKind::Return
            });
            references.extend(reference);
        }

        // Rules that cannot write caller buffers return those cotangents instead, and the accumulators, which know
        // their destinations, add the returned values to the buffers.
        if !rules.supports_reference_destinations() {
            references.clear();
            for kind in &mut destination_kinds {
                if *kind == CotangentDestinationKind::Reference {
                    *kind = CotangentDestinationKind::Return;
                }
            }
        }
        let (key, boundary_input_count) = match batching {
            None => (
                CustomRuleBackwardSpecializationKey {
                    leading_input_types: leading_values.iter().map(|value| value.r#type().into_owned()).collect(),
                    seed_extent_count,
                    input_tangent_types: self.input_tangent_types.clone(),
                    output_tangent_types: self.output_tangent_types.clone(),
                    seed_types: outputs
                        .iter()
                        .map(|output| output.as_value().map(|seed| seed.r#type().into_owned()))
                        .collect(),
                    destination_kinds: destination_kinds.clone(),
                    levels: Vec::new(),
                    discharged,
                },
                0,
            ),
            Some(batching) => {
                let leading_input_count = batching.input_types.len() - self.input_tangent_types.len();
                let (leading_input_types, input_tangent_types) = batching.input_types.split_at(leading_input_count);
                (
                    CustomRuleBackwardSpecializationKey {
                        leading_input_types: leading_input_types.to_vec(),
                        seed_extent_count,
                        input_tangent_types: input_tangent_types.to_vec(),
                        output_tangent_types: batching.output_types.clone(),
                        seed_types: outputs
                            .iter()
                            .zip(&batching.output_types)
                            .map(|(output, r#type)| output.as_value().map(|_| r#type.clone()))
                            .collect(),
                        destination_kinds: destination_kinds.clone(),
                        levels: batching.levels.clone(),
                        discharged,
                    },
                    batching.boundary_input_count(),
                )
            }
        };
        let specialization = rules.backward_specialization(key)?;

        // The specialized program consumes the boundary inputs, then the live seeds, then the caller buffers, then the
        // remaining known leading inputs, and returns the cotangents of the differentiated inputs with returned
        // destinations, except for the structural-zero cotangents, which contribute nothing to their destinations.
        let mut leading_values = leading_values.into_iter();
        let mut arguments = leading_values.by_ref().take(boundary_input_count).collect::<Vec<_>>();
        arguments.extend(outputs.iter().filter_map(MaybeZero::as_value).cloned());
        arguments.extend(references);
        arguments.extend(leading_values);
        check_count!("cotangent", specialization.cotangents, self.input_tangent_types.len(), TypeError);
        let mut returned = specialization.program.interpret_in_context(&**context, arguments)?.into_iter();
        for (accumulator, cotangent) in accumulators[self.leading_input_count..].iter().zip(&specialization.cotangents)
        {
            if *cotangent == TransposedProgramCotangent::Output {
                let cotangent = returned.next().ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "`{COMPOSITE_TRANSPOSE_OPERATION_NAME}` specialization returned too few cotangents",
                    ))
                })?;
                accumulator.accumulate(context, MaybeZero::Value(cotangent))?;
            }
        }
        Ok(())
    }
}

impl<A: Value<Type = ArrayType>> From<CompositeTransposeOperation<A, ArrayOperation<A>>> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: CompositeTransposeOperation<A, ArrayOperation<A>>) -> Self {
        // Attached backward regions use the native carrier; retained rules preserve their original member source.
        match operation.into_attached_family() {
            Ok(operation) => Self::CompositeTranspose(operation),
            Err(operation) => Self::LiftedCompositeTranspose(operation.into_family(LiftedCustomRules::Member)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReference, ArrayReferenceTransform,
        ArraySliceAxis, ArrayType, DataType, Dimension, DimensionBounds, DimensionType, DimensionValue,
        DimensionVariable, LogicalMesh, MeshAxis, MeshAxisType, Shape, Sharding, ShardingDimension,
    };
    use crate::batching::{BatchAxis, BatchedProgram, ProgramBatchingOutputAxesPolicy};
    use crate::contexts::{Context, EagerContext, StagingContext};
    use crate::differentiation::{
        CotangentDestinationKind, DifferentiationRule, TransposedProgram, TransposedProgramCotangent,
        TranspositionDriver,
    };
    use crate::operations::arithmetic::MulOperation;
    use crate::operations::composites::call::{CompositeJvpRule, CompositeOperation};
    use crate::operations::composites::rules::{CustomRuleRegistration, LiftedCustomRules};
    use crate::operations::composites::tests::{
        ArrayContext, ConvertedProgram, IrDefinition, IrTracer, RuleCounters, batched_program, cube_definition,
        custom_rule_program, ir_custom_rule_program, member_cube_definition,
    };
    use crate::operations::constants::zero::ZeroOperation;
    use crate::operations::constants::zero_like::ZeroLikeOperation;
    use crate::operations::control_flow::condition::ConditionOperation;
    use crate::operations::differentiation::linear_call::tests::scalar_multiply_program;
    use crate::operations::dimensions::dimension_from_scalar::DimensionFromScalarOperation;
    use crate::operations::manipulation::padding::PadOperation;
    use crate::operations::manipulation::slicing::Slice;
    use crate::operations::references::ReferenceAddUpdateOperation;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationOutput, PartialEvaluationValue, PartialValue};
    use crate::programs::{
        AtomId, EffectClasses, FlatProgram, MaybeZero, Program, ProgramBuilder, RegionDataFlow, RegionDriver,
        RegionInterface, RegionRef, RegionSlot, ValueProjection,
    };
    use crate::tracing::{Trace, TracingContext};

    use super::*;

    /// Reverse-mode carrier of a composite call in the array operation family.
    type ArrayCompositeTranspose = CompositeTransposeOperation<Array, ArrayOperation<Array>>;

    /// Reverse-mode carrier of a composite call in the array IR operation family.
    type ArrayIrCompositeTranspose = CompositeTransposeOperation<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Test-only transposition driver exposing one attached region, used to invoke the transposition rule of a carrier
    /// with an attached backward rule directly on the backward region that it replays.
    struct TestTranspositionDriver<'r> {
        /// Transpose region exposed by this driver.
        region: RegionRef<'r, Array, ArrayOperation<Array>>,
    }

    impl RegionDriver<Array, ArrayOperation<Array>> for TestTranspositionDriver<'_> {
        fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, Array, ArrayOperation<Array>>>
        where
            Array: 'r,
            ArrayOperation<Array>: 'r,
        {
            std::iter::once(self.region)
        }
    }

    impl TranspositionDriver<Array, ArrayOperation<Array>> for TestTranspositionDriver<'_> {
        fn transpose_program(
            &self,
            _region: RegionRef<'_, Array, ArrayOperation<Array>>,
            _input_indices: &[usize],
            _materialize_zeros: &[bool],
            _destination_kinds: &[CotangentDestinationKind],
        ) -> Result<TransposedProgram<Array, ArrayOperation<Array>>, DifferentiationError> {
            Err(ProgramError::UnsupportedOperation {
                message: "test driver does not transpose nested regions".to_string(),
            }
            .into())
        }
    }

    /// Builds a program over `(residual, tangent)` that applies a carrier whose attached backward rule is
    /// [`scalar_multiply_program`], with the residual as its leading input.
    fn attached_carrier_program() -> FlatProgram<ArrayContext> {
        let r#type = ArrayType::scalar(DataType::F64);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let backward = builder.import_region(scalar_multiply_program().entry_region_ref());
        let residual = builder.add_input(r#type.clone());
        let tangent = builder.add_input(r#type.clone());
        let output = builder
            .add_instruction(
                ArrayCompositeTranspose::from_backward_region(1, vec![r#type.clone()], vec![r#type]),
                vec![backward],
                vec![residual, tangent],
                None,
            )
            .unwrap()[0];
        builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap()
    }

    /// Registration of a retained-rule definition in the array IR family.
    type IrRegistration = CustomRuleRegistration<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Program builder in the array IR family.
    type IrBuilder = ProgramBuilder<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Returns a reverse-mode carrier of `definition`, as reverse-mode linearization of a call would stage it, whose
    /// first `leading_input_count` inputs are known.
    fn retained_carrier(
        rules: CustomRuleReference<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
        leading_input_count: usize,
        input_tangent_types: Vec<ArrayIrType>,
        output_tangent_types: Vec<ArrayIrType>,
    ) -> ArrayIrOperation<Array> {
        ArrayIrOperation::CompositeTranspose(CompositeTransposeOperation::new(
            rules,
            leading_input_count,
            input_tangent_types,
            output_tangent_types,
        ))
    }

    /// Returns a definition whose backward rule increments the known leading reference by one exactly when its
    /// differentiated input's cotangent is ignored and its output seed is a structural zero, so the effect exists only
    /// in that specialization.
    fn ignored_zero_effect_definition() -> IrRegistration {
        CustomRuleRegistration::new(IrDefinition::new("ignored_zero_effect").with_accumulating_vjp(
            |_| unreachable!("the tests construct the carrier directly"),
            |context, inputs, seeds, accumulators| {
                if !accumulators[1].is_needed() && matches!(&seeds[0], MaybeZero::Zero(_)) {
                    let stash = inputs[0].as_known().unwrap().clone();
                    let increment = context.lift(ArrayIrValue::Array(Array::scalar(1.0f64)?))?;
                    context.bind(
                        ArrayIrOperation::ReferenceAddUpdate(ReferenceAddUpdateOperation::new()),
                        Vec::new(),
                        &[stash, increment],
                    )?;
                }
                Ok(())
            },
        ))
    }

    /// Returns the carrier of [`ignored_zero_effect_definition`] over a known reference and a scalar tangent.
    fn ignored_zero_effect_carrier() -> ArrayIrOperation<Array> {
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F64).into();
        retained_carrier(ignored_zero_effect_definition().reference(), 1, vec![scalar_type.clone()], vec![scalar_type])
    }

    /// Returns a definition whose backward rule adds the output seed to the element at index 1 of its single
    /// differentiated input's cotangent, counting its invocations in `invocations`. A caller buffer receives a slice
    /// update without any full-sized temporary, while a returned cotangent is padded to the input extent.
    fn slice_at_one_definition(invocations: &Arc<AtomicUsize>) -> IrRegistration {
        let invocations = invocations.clone();
        CustomRuleRegistration::new(IrDefinition::new("slice_at_one").with_accumulating_vjp(
            |_| unreachable!("the tests construct the carrier directly"),
            move |context, inputs, seeds, accumulators| {
                invocations.fetch_add(1, Ordering::SeqCst);
                if !accumulators[0].is_needed() {
                    return Ok(());
                }
                let MaybeZero::Value(seed) = &seeds[0] else {
                    return Ok(());
                };
                if let Some(reference) = accumulators[0].reference(context)? {
                    let operation =
                        ReferenceAddUpdateOperation::new().with_transforms(vec![ArrayReferenceTransform::Slice {
                            axes: vec![ArraySliceAxis::new(1, 1, 1)],
                        }]);
                    context.bind(
                        ArrayIrOperation::ReferenceAddUpdate(operation),
                        Vec::new(),
                        &[reference, seed.clone()],
                    )?;
                } else {
                    let ArrayIrType::Array(input_type) = inputs[0].r#type().into_owned() else { unreachable!() };
                    let extent = input_type.shape().dimensions()[0].value().unwrap();
                    let zero = context.lift(ArrayIrValue::Array(Array::scalar(0.0f64)?))?;
                    let operation = ArrayIrOperation::from(ArrayOperation::Pad(PadOperation::new(
                        vec![1],
                        vec![extent as i64 - 2],
                        vec![0],
                    )?));
                    let output = context.bind(operation, Vec::new(), &[seed.clone(), zero])?.remove(0);
                    accumulators[0].accumulate(context, MaybeZero::Value(output))?;
                }
                Ok(())
            },
        ))
    }

    /// Stages the carrier of `definition` over one differentiated input of type `input_type` with an `f64[1]` output.
    fn slice_at_one_program(
        definition: &IrRegistration,
        input_type: ArrayType,
    ) -> Program<ArrayIrValue<Array>, ArrayIrOperation<Array>, Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>> {
        let mut builder = IrBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let carrier = retained_carrier(
            definition.reference(),
            0,
            vec![input_type.into()],
            vec![ArrayType::new_static(DataType::F64, [1]).into()],
        );
        let output = builder.add_instruction(carrier, Vec::new(), vec![input], None).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_composite_transpose() {
        let scalar_type = ArrayType::scalar(DataType::F64);

        // A carrier with an attached backward rule attaches that rule as its only, deferred rule region, whose own
        // effects decide whether transposition must run it (and whether the carrier's application carries deferred
        // work).
        let carrier =
            ArrayCompositeTranspose::from_backward_region(1, vec![scalar_type.clone()], vec![scalar_type.clone()]);
        assert_eq!(carrier.name(), COMPOSITE_TRANSPOSE_OPERATION_NAME);
        assert_eq!(carrier.leading_input_count(), 1);
        assert_eq!(carrier.input_tangent_types(), std::slice::from_ref(&scalar_type));
        assert_eq!(carrier.output_tangent_types(), std::slice::from_ref(&scalar_type));
        assert_eq!(carrier.rules(), None);
        assert_eq!(carrier.region_slots(), &[RegionSlot::deferred_rule("backward")]);
        assert!(!carrier.effects().summary().has_deferred_work());
        assert_eq!(carrier.to_string(), "composite_transpose [leading_input_count=1]");

        // A carrier with a retained backward rule attaches no region and carries deferred work until transposition
        // replaces it with the specialized backward program.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, false, true));
        let carrier =
            CompositeTransposeOperation::new(definition.reference(), 1, vec![scalar_type.clone()], vec![scalar_type]);
        assert_eq!(carrier.rules(), Some(&definition.reference()));
        assert!(carrier.region_slots().is_empty());
        assert!(carrier.effects().summary().has_deferred_work());
        assert_eq!(carrier.to_string(), "composite_transpose [name=\"cube\", leading_input_count=1]");
    }

    #[test]
    fn test_composite_transpose_into_family() {
        // A carrier with a retained backward rule follows its program into a containing family with the source that the
        // conversion derives from its rules, converting its tangent types.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let ir_scalar_type = ArrayIrType::from(scalar_type.clone());
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(member_cube_definition(&counters));
        let carrier = ArrayCompositeTranspose::new(
            definition.reference(),
            1,
            vec![scalar_type.clone()],
            vec![scalar_type.clone()],
        );
        let converted: CompositeTransposeOperation<
            ArrayIrValue<Array>,
            ArrayIrOperation<Array>,
            LiftedCustomRules<Array, ArrayOperation<Array>>,
        > = carrier.clone().into_family(LiftedCustomRules::Member);
        assert_eq!(converted.leading_input_count(), 1);
        assert_eq!(converted.seed_extent_count(), 0);
        assert_eq!(converted.input_tangent_types(), std::slice::from_ref(&ir_scalar_type));
        assert_eq!(converted.output_tangent_types(), std::slice::from_ref(&ir_scalar_type));
        assert_eq!(converted.rules(), Some(&LiftedCustomRules::Member(definition.reference())));
        assert_eq!(converted.to_string(), carrier.to_string());

        // A carrier with an attached backward rule carries no source, so the conversion never derives one.
        let attached = ArrayCompositeTranspose::from_backward_region(1, vec![scalar_type.clone()], vec![scalar_type]);
        let converted: ArrayIrCompositeTranspose =
            attached.into_family(|_| unreachable!("attached backward rules carry no source"));
        assert_eq!(
            converted,
            ArrayIrCompositeTranspose::from_backward_region(1, vec![ir_scalar_type.clone()], vec![ir_scalar_type]),
        );
    }

    #[test]
    fn test_composite_transpose_into_attached_family() {
        // A carrier with an attached backward rule converts into any family with any source type, while a carrier with
        // a retained backward rule is returned unchanged.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let ir_scalar_type = ArrayIrType::from(scalar_type.clone());
        let attached =
            ArrayCompositeTranspose::from_backward_region(1, vec![scalar_type.clone()], vec![scalar_type.clone()]);
        assert_eq!(
            attached.into_attached_family::<
                ArrayIrValue<Array>,
                ArrayIrOperation<Array>,
                CustomRuleReference<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
            >(),
            Ok(ArrayIrCompositeTranspose::from_backward_region(1, vec![ir_scalar_type.clone()], vec![ir_scalar_type])),
        );
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(member_cube_definition(&counters));
        let retained =
            ArrayCompositeTranspose::new(definition.reference(), 1, vec![scalar_type.clone()], vec![scalar_type]);
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
    fn test_composite_transpose_type_inference() {
        // The attached backward region receives the leading inputs followed by one cotangent per output and returns one
        // cotangent per differentiated input. The tangent types are differential types rather than primal storage
        // types, which the cotangent mapping must respect.
        let residual_type = ArrayType::scalar(DataType::F64);
        let tangent_type = ArrayType::scalar(DataType::F32);
        let carrier =
            ArrayCompositeTranspose::from_backward_region(1, vec![tangent_type.clone()], vec![tangent_type.clone()]);
        let input_types = [residual_type.clone(), tangent_type.clone()];
        let backward = RegionInterface::new(
            vec![residual_type.clone(), tangent_type.clone()],
            vec![tangent_type.clone()],
            EffectClasses::NONE,
        );
        assert_eq!(
            carrier.infer_region_input_types(&input_types, std::slice::from_ref(&backward)),
            Ok(vec![Some(vec![residual_type.clone(), tangent_type.clone()])]),
        );
        assert_eq!(
            carrier.infer_output_types(&input_types, std::slice::from_ref(&backward)),
            Ok(vec![tangent_type.clone()]),
        );

        // The carrier requires exactly its backward region, its leading inputs, and its declared tangent inputs.
        assert_eq!(
            carrier.infer_output_types(&input_types, &[]),
            Err(TypeError::invalid("expected 1 region but got 0")),
        );
        assert_eq!(
            carrier.infer_output_types(std::slice::from_ref(&tangent_type), std::slice::from_ref(&backward)),
            Err(TypeError::invalid("expected 2 inputs but got 1")),
        );
        assert_eq!(
            carrier
                .infer_output_types(&[residual_type.clone(), residual_type.clone()], std::slice::from_ref(&backward)),
            Err(TypeError::invalid(
                "`composite_transpose` tangent input type signature mismatch: expected [f32[]] but got [f64[]]",
            )),
        );
        assert_eq!(
            carrier.infer_output_types(
                &input_types,
                &[RegionInterface::new(
                    vec![tangent_type.clone(), tangent_type.clone()],
                    vec![tangent_type.clone()],
                    EffectClasses::NONE,
                )],
            ),
            Err(TypeError::invalid(
                "`composite_transpose` backward rule input type signature mismatch: expected [f64[], f32[]] \
                 but got [f32[], f32[]]",
            )),
        );
    }

    #[test]
    fn test_composite_transpose_type_inference_exact_backward_outputs() {
        let scalar = ArrayType::scalar(DataType::F32);
        let vector = ArrayType::new_static(DataType::F32, [2]);
        let exact = DimensionVariable::new("exact", DimensionBounds::new(2, Some(3)).unwrap());
        let broad = DimensionVariable::new("broad", DimensionBounds::new(0, Some(4)).unwrap());
        let other = DimensionVariable::new("other", DimensionBounds::new(0, Some(4)).unwrap());
        let carrier = ArrayCompositeTranspose::from_backward_region(0, vec![vector.clone()], vec![scalar.clone()]);
        let infer = |carrier: &ArrayCompositeTranspose, input_type: &ArrayType, output_type: ArrayType| {
            carrier.infer_output_types(
                std::slice::from_ref(input_type),
                &[RegionInterface::new(vec![scalar.clone()], vec![output_type], EffectClasses::NONE)],
            )
        };

        // An exact named extent and its static representation describe precisely the same cotangent space.
        assert_eq!(
            infer(&carrier, &vector, ArrayType::new(DataType::F32, Shape::new(vec![exact.into()]))),
            Ok(vec![scalar.clone()]),
        );

        // A possible extent is not proof of equality, and neither shape nor element type may change.
        let mismatch = |actual: &str| {
            Err(TypeError::invalid(format!(
                "`composite_transpose` backward rule output type signature mismatch: expected [f32[2]] but got \
                 [{actual}]",
            )))
        };
        assert_eq!(
            infer(&carrier, &vector, ArrayType::new(DataType::F32, Shape::new(vec![broad.clone().into()]))),
            mismatch("f32[broad]"),
        );
        assert_eq!(infer(&carrier, &vector, ArrayType::new_static(DataType::F32, [3])), mismatch("f32[3]"));
        assert_eq!(infer(&carrier, &vector, ArrayType::new_static(DataType::F64, [2])), mismatch("f64[2]"));

        // Equal bounds do not equate unrelated identities or erase a repeated-axis relationship.
        let square = ArrayType::new(DataType::F32, Shape::new(vec![broad.clone().into(), broad.clone().into()]));
        let carrier = ArrayCompositeTranspose::from_backward_region(0, vec![square.clone()], vec![scalar.clone()]);
        assert_eq!(
            infer(
                &carrier,
                &square,
                ArrayType::new(DataType::F32, Shape::new(vec![other.clone().into(), other.clone().into()]))
            ),
            Err(TypeError::invalid(
                "`composite_transpose` backward rule output type signature mismatch: expected \
                 [f32[broad, broad]] but got [f32[other, other]]",
            )),
        );
        assert_eq!(
            infer(&carrier, &square, ArrayType::new(DataType::F32, Shape::new(vec![broad.into(), other.into()]))),
            Err(TypeError::invalid(
                "`composite_transpose` backward rule output type signature mismatch: expected \
                 [f32[broad, broad]] but got [f32[broad, other]]",
            )),
        );
    }

    #[test]
    fn test_composite_transpose_type_inference_retained_rules() {
        // A carrier takes its known leading inputs followed by the tangents of the differentiated inputs, and it
        // returns the output tangents. It has no regions.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, false, true));
        let scalar_type = ArrayType::scalar(DataType::F64);
        let carrier = CompositeTransposeOperation::new(
            definition.reference(),
            1,
            vec![scalar_type.clone()],
            vec![scalar_type.clone()],
        );
        assert_eq!(
            carrier.infer_output_types(&[scalar_type.clone(), scalar_type.clone()], &[]),
            Ok(vec![scalar_type.clone()]),
        );
        assert_eq!(
            carrier.infer_output_types(&[scalar_type.clone()], &[]),
            Err(TypeError::invalid("expected 2 inputs but got 1")),
        );
        assert_eq!(
            carrier.infer_output_types(&[scalar_type.clone(), ArrayType::scalar(DataType::F32)], &[]),
            Err(TypeError::invalid(
                "`composite_transpose` tangent input type signature mismatch: expected [f64[]] but got [f32[]]",
            )),
        );
        let region = RegionInterface::new(vec![scalar_type.clone()], vec![scalar_type.clone()], EffectClasses::NONE);
        assert_eq!(
            carrier.infer_output_types(&[scalar_type.clone(), scalar_type], std::slice::from_ref(&region)),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
        assert_eq!(counters.counts(), (0, 0, 0));
    }

    #[test]
    fn test_composite_transpose_region_data_flow() {
        let scalar = ArrayType::scalar(DataType::F64);
        let operation = ArrayCompositeTranspose::from_backward_region(0, vec![scalar.clone()], vec![scalar]);
        // The backward region is deferred transformation work rather than a declared ordinary computation source.
        assert!(matches!(operation.region_data_flow(), RegionDataFlow::Opaque));
        assert_eq!(operation.region_slots(), &[RegionSlot::deferred_rule("backward")]);
        assert_eq!(operation.output_region_provenance(0), Vec::new());
    }

    #[test]
    fn test_composite_transpose_interpretation() {
        // A carrier has no forward program, so it supports only reverse-mode differentiation.
        assert_eq!(
            attached_carrier_program().interpret(vec![Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()]),
            Err(ProgramError::UnsupportedOperation {
                message: "`composite_transpose` has no forward program to execute; it supports only \
                          reverse-mode differentiation (e.g., `vjp`, `value_and_gradient`, or `jacobian_reverse`)"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_composite_transpose_interpretation_retained_rules() {
        // The carrier has no forward program, so ordinary execution and forward mode reject it.
        let stash = ArrayReference::new(Array::scalar(0.0f64).unwrap());
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        assert!(matches!(
            context.bind(
                ignored_zero_effect_carrier(),
                Vec::new(),
                &[ArrayIrValue::Reference(stash.clone()), ArrayIrValue::Array(Array::scalar(3.0f64).unwrap())],
            ),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "`composite_transpose` `ignored_zero_effect` has no forward program to execute; it \
                               supports only reverse-mode differentiation (e.g., `vjp`, `value_and_gradient`, or \
                               `jacobian_reverse`)",
        ));
        assert_eq!(stash.read(), Ok(Array::scalar(0.0f64).unwrap()));
    }

    #[test]
    fn test_composite_transpose_partial_evaluation() {
        // A carrier must survive partial evaluation as one unknown-producing instruction. Splitting or folding it would
        // separate the backward region from the leading inputs that parameterize it, so the default opaque rule is the
        // correct behavior here.
        let r#type = ArrayType::scalar(DataType::F64);
        let evaluation = attached_carrier_program()
            .partially_evaluate(&[PartialValue::Unknown(r#type.clone()), PartialValue::Unknown(r#type)])
            .unwrap();
        assert!(matches!(evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = composite_transpose [leading_input_count=1] %0 %1 [
                    backward={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = mul %1 %0
                        in (%2)
                    },
                ]
                in (%2)
            "}
            .trim_end()
        );
    }

    #[test]
    fn test_composite_transpose_partial_evaluation_retained_rules_disconnected() {
        let stash = ArrayReference::new(Array::scalar(0.0f64).unwrap());
        let context =
            PartialEvaluationContext::new(EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new());
        let tangent = context.unknown_input(ArrayType::scalar(DataType::F64).into(), 0);
        context
            .residualize(
                ignored_zero_effect_carrier(),
                Vec::new(),
                &[PartialEvaluationValue::known(ArrayIrValue::Reference(stash.clone())), tangent],
            )
            .unwrap();
        let evaluation = context.into_evaluation(Vec::new()).unwrap();
        let transposed =
            evaluation.program().transpose_with_respect_to(&[0], &[CotangentDestinationKind::Ignore]).unwrap();
        assert_eq!(stash.read(), Ok(Array::scalar(0.0f64).unwrap()));
        assert_eq!(transposed.interpret(vec![ArrayIrValue::Reference(stash.clone())]), Ok(Vec::new()));
        assert_eq!(stash.read(), Ok(Array::scalar(1.0f64).unwrap()));
    }

    #[test]
    fn test_composite_transpose_partial_evaluation_retained_rules_known_computation() {
        let stash = ArrayReference::new(Array::scalar(0.0f64).unwrap());
        let reference_type = ArrayIrValue::Reference(stash.clone()).r#type().into_owned();
        let tangent_type: ArrayIrType = ArrayType::scalar(DataType::F64).into();
        let mut body = IrBuilder::new();
        let reference = body.add_input(reference_type.clone());
        let tangent = body.add_input(tangent_type.clone());
        body.add_instruction(ignored_zero_effect_carrier(), Vec::new(), vec![reference, tangent], None)
            .unwrap();
        let body = body
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                Vec::new(),
                vec![Placeholder, Placeholder],
                Vec::new(),
            )
            .unwrap();
        // A composite call without rules is a plain region-carrying computation whose reference input must be one of
        // its leading non-differentiated inputs.
        let call = CompositeOperation::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::from_rule_regions(
            CompositeJvpRule::Absent,
            false,
        )
        .with_non_differentiated_count(1)
        .unwrap();
        let mut builder = IrBuilder::new();
        let region = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(reference_type.clone());
        let tangent = builder.add_input(tangent_type.clone());
        builder
            .add_instruction(ArrayIrOperation::Composite(call), vec![region], vec![reference, tangent], None)
            .unwrap();
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                Vec::new(),
                vec![Placeholder, Placeholder],
                Vec::new(),
            )
            .unwrap();
        assert!(program.entry_region_ref().effects().has_deferred_work());

        // Partial evaluation never folds deferred work, even when every input is known.
        let parent = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let evaluation = program
            .partially_evaluate_in_context(
                &parent,
                &[PartialValue::Known(parent.input(reference_type)), PartialValue::Known(parent.input(tangent_type))],
            )
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:ref<f64[]>, %1:f64[] .
                let () = composite [non_differentiated_count=1] %0 %1 [
                    primal={
                        lambda %0:ref<f64[]>, %1:f64[] .
                        let %2:f64[] = composite_transpose [name=\"ignored_zero_effect\", leading_input_count=1] %0 %1
                        in ()
                    },
                ]
                in ()
            "}
            .trim_end(),
        );
        assert!(evaluation.program().entry_region_ref().effects().has_deferred_work());
        assert_eq!(stash.read(), Ok(Array::scalar(0.0f64).unwrap()));
    }

    #[test]
    fn test_composite_transpose_batching() {
        // A completely replicated carrier needs no structural region rewrite, so it rebinds itself over its untouched
        // backward region.
        let program = attached_carrier_program();
        let (batched, output_axes) = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::replicated()]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = composite_transpose [leading_input_count=1] %0 %1 [
                    backward={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = mul %1 %0
                        in (%2)
                    },
                ]
                in (%2)
            "}
            .trim_end()
        );

        // Any other carrier is rejected, because no forward program determines the batch axes of its outputs.
        assert!(matches!(
            program.batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            ),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "a `composite_transpose` with an attached backward rule cannot be batched \
                               structurally, because it has no forward program to determine the batch axes of its \
                               outputs; it is preserved unchanged only when every input is replicated at an unnamed \
                               batching level",
        ));
    }

    #[test]
    fn test_composite_transpose_batching_retained_rules() {
        // Batching a linearization batches its carrier, whose backward specialization is batched when it is transposed.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, false, true));
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        let reverse = program
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose)
            .unwrap();
        let squares = Array::vector(vec![1f64, 4f64, 9f64]).unwrap();
        let seeds = Array::vector(vec![1f64, 2f64, 3f64]).unwrap();
        let batched = batched_program(reverse.tangent(), 3, &[BatchAxis::new(0), BatchAxis::new(0)]);
        assert_eq!(
            batched.to_string(),
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
        // The batched carrier still declares deferred work, so shared transform boundaries retain it.
        assert!(batched.instructions()[0].operation().effects().summary().has_deferred_work());
        let pullback = batched.transpose_with_respect_to(&[0], &[]).unwrap();
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
            pullback.interpret(vec![seeds.clone(), squares.clone()]),
            Ok(vec![Array::vector(vec![2f64, 16f64, 54f64]).unwrap()]),
        );

        // A replicated tangent input receives the sum of its per-item cotangents.
        let batched = batched_program(reverse.tangent(), 3, &[BatchAxis::replicated(), BatchAxis::new(0)]);
        let pullback = batched.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(pullback.interpret(vec![seeds, squares]), Ok(vec![Array::scalar(72f64).unwrap()]));
        assert_eq!(counters.counts(), (0, 1, 1));
    }

    #[test]
    fn test_composite_transpose_batching_retained_rules_caller_buffers() {
        // `x ↦ x[1]` over `f64[4]`, whose accumulating backward rule adds its seed to only the affected entry of a
        // caller buffer. Batching the linearization batches the carrier, and transposing it with a caller buffer
        // destination batches the buffer-updating specialization, so each item's row receives its own seed.
        let element = |x: &IrTracer| -> Result<IrTracer, ProgramError> {
            Ok(ValueProjection::<ArrayType>::into_projected(x.clone())?.slice(&[1], &[2usize], &[1])?.into_value())
        };
        let definition = CustomRuleRegistration::new(
            IrDefinition::new("element")
                .with_accumulating_vjp(
                    move |primals| Ok((vec![element(&primals[0])?], Vec::new())),
                    |context, _, seeds, accumulators| {
                        let MaybeZero::Value(seed) = &seeds[0] else {
                            return Ok(());
                        };
                        let Some(buffer) = accumulators[0].reference(context)? else {
                            return Err(ProgramError::InvalidArgument {
                                message: "expected a caller buffer".to_string(),
                            }
                            .into());
                        };
                        let operation =
                            ReferenceAddUpdateOperation::new().with_transforms(vec![ArrayReferenceTransform::Slice {
                                axes: vec![ArraySliceAxis::new(1, 1, 1)],
                            }]);
                        context.bind(ArrayIrOperation::from(operation), Vec::new(), &[buffer, seed.clone()])?;
                        Ok(())
                    },
                )
                .with_batching_support(),
        );
        let input_type: ArrayIrType = ArrayType::new_static(DataType::F64, [4]).into();
        let (_, primal) =
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(|x: IrTracer| element(&x), input_type)
                .unwrap();
        let program = ir_custom_rule_program(&definition, 0, primal.into_flat_program());
        let reverse = program
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose)
            .unwrap();
        let extent = DimensionValue::constant(3).unwrap();
        let batched = reverse
            .tangent()
            .batched_with_threaded_extent(
                extent.r#type().into_owned(),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        let pullback = batched.transpose_with_respect_to(&[1], &[CotangentDestinationKind::Reference]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:zero[], %1:f64[3, 1], %2:ref<f64[3, 4]>, %3:dimension<3> .
                let () = reference_add_update [transforms=[slice(axes=[0:3, 1:2])]] %2 %1
                in ()
            "}
            .trim_end(),
        );
        let buffer = ArrayReference::new(Array::matrix(3, 4, vec![0.0f64; 12]).unwrap());
        let zero = Array::from_logical_bytes(ArrayType::scalar(DataType::Zero), &[]).unwrap();
        assert_eq!(
            pullback.interpret(vec![
                ArrayIrValue::Array(zero),
                ArrayIrValue::Array(Array::matrix(3, 1, vec![1.0f64, 2.0, 3.0]).unwrap()),
                ArrayIrValue::Reference(buffer.clone()),
                ArrayIrValue::Dimension(extent),
            ]),
            Ok(Vec::new()),
        );
        assert_eq!(
            buffer.read(),
            Ok(Array::matrix(3, 4, vec![0.0f64, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0]).unwrap()),
        );
    }

    #[test]
    fn test_composite_transpose_batching_retained_rules_source_carrier() {
        // The source carriers from which backward specializations are derived invoke the backward rule directly and are
        // only ever transposed in an unbatched source program, so batching one is rejected as a malformed program.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, false, true));
        let scalar_type = ArrayType::scalar(DataType::F64);
        let carrier = CompositeTransposeOperation::direct(
            definition.reference(),
            1,
            0,
            vec![scalar_type.clone()],
            vec![scalar_type.clone()],
            false,
        );
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = vec![builder.add_input(scalar_type.clone()), builder.add_input(scalar_type)];
        let outputs = builder.add_instruction(carrier, Vec::new(), inputs, None).unwrap().to_vec();
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], vec![Placeholder]).unwrap();
        assert!(matches!(
            program.batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            ),
            Err(BatchingError::Program(ProgramError::MalformedProgram(message)))
                if message == "`composite_transpose` `cube` source carriers are never batched",
        ));
        assert_eq!(counters.counts(), (0, 0, 0));
    }

    #[test]
    fn test_composite_transpose_batching_retained_rules_deferred_forward() {
        // Batching a linearization whose carrier defers its forward rule is rejected, because the carrier's forward key
        // describes only the batching of the call from which it was staged. Transposing the linearization first yields
        // a pullback of ordinary operations, which batches like any other program.
        let counters = Arc::new(RuleCounters::default());
        let definition =
            CustomRuleRegistration::new(cube_definition(&counters, false, true).with_deferred_forward_rule());
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        let reverse = program
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose)
            .unwrap();
        assert!(matches!(
            reverse.tangent().batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            ),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "`composite_transpose` `cube` defers its forward rule to the backward pass (refer to \
                               `with_deferred_forward`), so it cannot be batched after it is differentiated; batch \
                               the call before differentiating it, or transpose the derivative before batching it",
        ));
        let pullback = reverse.tangent().transpose_with_respect_to(&[0], &[]).unwrap();
        let batched = batched_program(&pullback, 3, &[BatchAxis::new(0), BatchAxis::new(0)]);
        assert_eq!(
            batched.interpret(vec![
                Array::vector(vec![1f64, 2f64, 3f64]).unwrap(),
                Array::vector(vec![1f64, 2f64, 3f64]).unwrap(),
            ]),
            Ok(vec![Array::vector(vec![2f64, 16f64, 54f64]).unwrap()]),
        );
        assert_eq!(counters.counts(), (0, 1, 1));
    }

    #[test]
    fn test_composite_transpose_differentiation() {
        // A carrier has no forward program, so forward mode rejects it whether its backward rule is attached or
        // retained.
        assert!(matches!(
            attached_carrier_program().jvp(),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`composite_transpose` has no forward-mode (i.e., JVP) rule; it supports only \
                               reverse-mode differentiation",
        ));
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, false, true));
        let scalar_type = ArrayType::scalar(DataType::F64);
        let carrier = CompositeTransposeOperation::new(
            definition.reference(),
            1,
            vec![scalar_type.clone()],
            vec![scalar_type.clone()],
        );
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = vec![builder.add_input(scalar_type.clone()), builder.add_input(scalar_type)];
        let outputs = builder.add_instruction(carrier, Vec::new(), inputs, None).unwrap().to_vec();
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], vec![Placeholder]).unwrap();
        assert!(matches!(
            program.jvp(),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`composite_transpose` `cube` has no forward-mode (i.e., JVP) rule; it supports only \
                               reverse-mode differentiation",
        ));
        assert_eq!(counters.counts(), (0, 0, 0));
    }

    #[test]
    fn test_composite_transpose_transposition() {
        let r#type = ArrayType::scalar(DataType::F64);
        let backward = scalar_multiply_program();
        let driver = TestTranspositionDriver { region: backward.entry_region_ref() };
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();

        // Transposition validates the carrier's input split independently of type inference, because a pullback may be
        // built from an imported carrier whose inputs were pruned after inference ran.
        let carrier = ArrayCompositeTranspose::from_backward_region(1, vec![r#type.clone()], vec![r#type.clone()]);
        assert_eq!(
            carrier.transpose(&mut TranspositionContext::new(context.clone()), &driver, &[], &[], &[]),
            Err(DifferentiationError::Program(ProgramError::InvalidInputCount { expected: 2, actual: 0 })),
        );

        // Every leading input must be known during transposition, because the replayed backward region consumes the
        // leading values themselves rather than cotangents for them.
        let seed = context.input(r#type.clone());
        let inputs = [PartialValue::Unknown(r#type.clone()), PartialValue::Unknown(r#type.clone())];
        let mut rule_context = TranspositionContext::new(context.clone());
        let accumulators = rule_context.cotangent_accumulators(&inputs, &[]).unwrap();
        assert_eq!(
            carrier.transpose(&mut rule_context, &driver, &inputs, &[MaybeZero::Value(seed)], &accumulators),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(
                "`composite_transpose` leading input 0 is not known during transposition".to_string(),
            ))),
        );
    }

    #[test]
    fn test_composite_transpose_transposition_known_input_cotangents() {
        // The backward region of a carrier returns one cotangent per differentiated input, in input order after the
        // leading inputs. Leading inputs are fixed primal parameters rather than linear inputs, and a *known*
        // differentiated input is not being differentiated, so both receive structural zeros while the replayed
        // region's cotangents are assigned only to the unknown linear inputs.
        let r#type = ArrayType::scalar(DataType::F64);
        let mut transpose_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let residual = transpose_builder.add_input(r#type.clone());
        let output_cotangent = transpose_builder.add_input(r#type.clone());
        let first_input_cotangent = transpose_builder
            .add_instruction(MulOperation::new(), Vec::new(), vec![residual, output_cotangent], None)
            .unwrap()[0];
        let transpose = transpose_builder
            .build::<Vec<Array>, Vec<Array>>(
                vec![first_input_cotangent, output_cotangent],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let driver = TestTranspositionDriver { region: transpose.entry_region_ref() };
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let residual = context.input(r#type.clone());
        let known_linear = context.input(r#type.clone());
        let output_cotangent = context.input(r#type.clone());

        let linear_types = vec![r#type.clone(), r#type.clone()];
        let cotangents = {
            let mut rule_context = TranspositionContext::new(context.clone());
            let rule_inputs = &[
                PartialValue::Known(residual),
                PartialValue::Unknown(r#type.clone()),
                PartialValue::Known(known_linear),
            ];
            let accumulators = rule_context.cotangent_accumulators(rule_inputs, &[]).unwrap();
            ArrayCompositeTranspose::from_backward_region(1, linear_types, vec![r#type.clone()])
                .transpose(
                    &mut rule_context,
                    &driver,
                    rule_inputs,
                    &[MaybeZero::Value(output_cotangent)],
                    &accumulators,
                )
                .unwrap();
            rule_context.take_cotangents(&accumulators).unwrap()
        };

        assert_eq!(cotangents.len(), 3);
        assert!(cotangents[0].is_zero());
        assert!(matches!(cotangents[1], MaybeZero::Value(_)));
        assert!(cotangents[2].is_zero());
    }

    #[test]
    fn test_composite_transpose_transposition_structural_zero_outputs() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let primal_type = ArrayType::scalar(DataType::F64)
            .with_sharding(Sharding::new(mesh, Vec::new()).unwrap().with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let tangent_type = primal_type.tangent().unwrap();
        let cotangent_type = primal_type.cotangent().unwrap();
        assert_ne!(tangent_type, cotangent_type);

        // A canonical `zero` transpose-region output is already typed in the linear input's cotangent space.
        // Recovering its structural zero must retain that type instead of dualizing its sharding a second time.
        let mut transpose_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        transpose_builder.add_input(cotangent_type.clone());
        let zero = transpose_builder
            .add_instruction(ZeroOperation::new(cotangent_type.clone()), Vec::new(), Vec::new(), None)
            .unwrap()[0];
        let transpose = transpose_builder
            .build::<Vec<Array>, Vec<Array>>(vec![zero], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let driver = TestTranspositionDriver { region: transpose.entry_region_ref() };
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = context.input(cotangent_type.clone());
        let cotangents = {
            let mut rule_context = TranspositionContext::new(context.clone());
            let rule_inputs = &[PartialValue::Unknown(tangent_type.clone())];
            let accumulators = rule_context.cotangent_accumulators(rule_inputs, &[]).unwrap();
            ArrayCompositeTranspose::from_backward_region(0, vec![tangent_type.clone()], vec![tangent_type.clone()])
                .transpose(
                    &mut rule_context,
                    &driver,
                    rule_inputs,
                    &[MaybeZero::Value(output_cotangent)],
                    &accumulators,
                )
                .unwrap();
            rule_context.take_cotangents(&accumulators).unwrap()
        };
        assert!(matches!(&cotangents[0], MaybeZero::Zero(r#type) if r#type == &cotangent_type));

        // `zero_like` is equally structural even though it consumes an exemplar input. Opaque region replay must
        // recognize it instead of turning the result into a live zero-valued tracer.
        let mut transpose_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = transpose_builder.add_input(cotangent_type.clone());
        let zero_like = transpose_builder
            .add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![output_cotangent], None)
            .unwrap()[0];
        let transpose = transpose_builder
            .build::<Vec<Array>, Vec<Array>>(vec![zero_like], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let driver = TestTranspositionDriver { region: transpose.entry_region_ref() };
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = context.input(cotangent_type.clone());
        let cotangents = {
            let mut rule_context = TranspositionContext::new(context.clone());
            let rule_inputs = &[PartialValue::Unknown(primal_type.tangent().unwrap())];
            let accumulators = rule_context.cotangent_accumulators(rule_inputs, &[]).unwrap();
            ArrayCompositeTranspose::from_backward_region(0, vec![tangent_type.clone()], vec![tangent_type])
                .transpose(
                    &mut rule_context,
                    &driver,
                    rule_inputs,
                    &[MaybeZero::Value(output_cotangent)],
                    &accumulators,
                )
                .unwrap();
            rule_context.take_cotangents(&accumulators).unwrap()
        };
        assert!(matches!(&cotangents[0], MaybeZero::Zero(r#type) if r#type == &cotangent_type));
    }

    #[test]
    fn test_composite_transpose_transposition_dynamic_structural_zero() {
        // An attached backward region returns a canonical zero built from its retained dimension. Its dynamic type
        // does not require accumulating the zero as a live value or returning it across the transposed boundary.
        type IrValue = ArrayIrValue<Array>;
        type IrOperation = ArrayIrOperation<Array>;
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![extent_type.simplified_dimension()]));
        let scalar_type = ArrayType::scalar(DataType::F32);
        let mut backward = ProgramBuilder::<IrValue, IrOperation>::new();
        let extent = backward.add_input(extent_type.clone().into());
        backward.add_input(scalar_type.clone().into());
        let zero = backward
            .add_instruction(ZeroOperation::new(input_type.clone()), Vec::new(), vec![extent], None)
            .unwrap()[0];
        let backward = backward
            .build::<Vec<IrValue>, Vec<IrValue>>(vec![zero], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<IrValue, IrOperation>::new();
        let extent = builder.add_input(extent_type.into());
        let tangent = builder.add_input(input_type.clone().into());
        let backward = builder.import_program(backward);
        let carrier = CompositeTransposeOperation::<IrValue, IrOperation>::from_backward_region(
            1,
            vec![input_type.clone().into()],
            vec![scalar_type.clone().into()],
        );
        let output = builder.add_instruction(carrier, vec![backward], vec![extent, tangent], None).unwrap()[0];
        let program = builder
            .build::<Vec<IrValue>, Vec<IrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let transposed = program.entry_region_ref().transpose(&[1], &[], &[]).unwrap();
        assert_eq!(transposed.cotangents(), &[TransposedProgramCotangent::Zero]);
        assert_eq!(
            transposed.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:dimension<extent ∈ [1, 8)> .
                let %2:f32[extent] = zero [type=f32[extent]] %1
                in ()"},
        );
        assert_eq!(
            transposed.assemble_cotangents(&[input_type.clone().into()], Vec::<IrValue>::new()),
            Ok(vec![Some(MaybeZero::Zero(ArrayIrType::Array(input_type)))]),
        );
    }

    #[test]
    fn test_composite_transpose_transposition_unrequested_backward_effects() {
        // The backward rule asserts a dimension bound even though its contribution is unused. Its forward call has
        // no computation region, so ordinary forward-effect summaries cannot keep this assertion alive.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let mut backward = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let seed = backward.add_input(scalar_type.clone());
        let invalid_extent = backward.add_constant(Array::scalar(-1i32).unwrap().into());
        backward
            .add_instruction(
                DimensionFromScalarOperation::new(
                    DimensionVariable::new("extent", DimensionBounds::new(0, None).unwrap()).into(),
                ),
                Vec::new(),
                vec![invalid_extent],
                None,
            )
            .unwrap();
        let backward = backward
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![seed],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(scalar_type.clone());
        let backward = builder.import_program(backward);
        builder
            .add_instruction(
                ArrayIrCompositeTranspose::from_backward_region(0, vec![scalar_type.clone()], vec![scalar_type]),
                vec![backward],
                vec![input],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        // The unused carrier declares deferred work, so simplification keeps it and its backward effects still run.
        for program in [program.clone(), program.simplified().unwrap()] {
            let transposed = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Ignore]).unwrap();
            assert!(transposed.input_ids().is_empty());
            assert!(transposed.output_ids().is_empty());
            assert!(matches!(transposed.interpret(Vec::new()), Err(ProgramError::Concretization { message })
                if message == "cannot extract a concrete `usize` from `i32[]`; value `-1` is out of range"));
        }
    }

    #[test]
    fn test_composite_transpose_transposition_retained_rules() {
        let invocations = Arc::new(AtomicUsize::new(0));
        let definition = slice_at_one_definition(&invocations);
        let retained = Arc::downgrade(&definition.definition());
        let program = slice_at_one_program(&definition, ArrayType::new_static(DataType::F64, [3]));
        assert_eq!(invocations.load(Ordering::SeqCst), 0);

        // One backward rule specializes to a returned cotangent and a caller buffer. An ignored sole destination leaves
        // the seed unused, so transposition passes the carrier, which has no leading inputs, only a structural-zero
        // seed, whose cotangent is a structural zero without specializing the rule.
        let returned = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Return]).unwrap();
        let buffered = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap();
        let ignored = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Ignore]).unwrap();
        assert_eq!(
            buffered.to_string(),
            indoc! {"
                lambda %0:f64[1], %1:ref<f64[3]> .
                let () = reference_add_update [transforms=[slice(axes=[1:2])]] %1 %0
                in ()
            "}
            .trim_end(),
        );
        assert_eq!(
            returned.to_string(),
            indoc! {"
                lambda %0:f64[1] .
                let %1:f64[] = const 0.0
                    %2:f64[3] = pad [edge_padding_low=[1], edge_padding_high=[1], interior_padding=[0]] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            ignored.to_string(),
            indoc! {"
                lambda %0:f64[1] .
                in ()
            "}
            .trim_end(),
        );
        let seed = ArrayIrValue::Array(Array::vector(vec![3.0f64]).unwrap());
        assert_eq!(
            returned.interpret(vec![seed.clone()]),
            Ok(vec![Array::vector(vec![0.0f64, 3.0, 0.0]).unwrap().into()]),
        );

        // The buffered specialization preserves existing contributions, and a second buffer reuses it.
        let first = ArrayReference::new(Array::vector(vec![10.0f64, 20.0, 30.0]).unwrap());
        let second = ArrayReference::new(Array::vector(vec![40.0f64, 50.0, 60.0]).unwrap());
        assert_eq!(buffered.interpret(vec![seed.clone(), first.clone().into()]), Ok(vec![]));
        assert_eq!(buffered.interpret(vec![seed.clone(), first.clone().into()]), Ok(vec![]));
        assert_eq!(first.read(), Ok(Array::vector(vec![10.0f64, 26.0, 30.0]).unwrap()));
        assert_eq!(buffered.interpret(vec![seed.clone(), second.clone().into()]), Ok(vec![]));
        assert_eq!(second.read(), Ok(Array::vector(vec![40.0f64, 53.0, 60.0]).unwrap()));
        assert_eq!(ignored.interpret(vec![seed.clone()]), Ok(vec![]));
        assert_eq!(invocations.load(Ordering::SeqCst), 2);
        let repeated = program.clone().transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap();
        assert_eq!(repeated.to_string(), buffered.to_string());
        assert_eq!(invocations.load(Ordering::SeqCst), 2);

        // A wider input is a different carrier signature, so its returned cotangent is rebuilt from its own extent
        // even though its seed type and destination kind are unchanged.
        let wider = slice_at_one_program(&definition, ArrayType::new_static(DataType::F64, [4]));
        let wider_pullback = wider.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Return]).unwrap();
        assert_eq!(
            wider_pullback.interpret(vec![seed]),
            Ok(vec![Array::vector(vec![0.0f64, 3.0, 0.0, 0.0]).unwrap().into()]),
        );
        assert_eq!(invocations.load(Ordering::SeqCst), 3);

        // The rule receives the current renamed boundary, but existing reference slicing requires a static referent
        // type. Both failures report their current identity and do not retain failed specializations.
        let original = DimensionVariable::new("original", DimensionBounds::new(2, Some(6)).unwrap());
        let relocated = DimensionVariable::new("relocated", DimensionBounds::new(2, Some(6)).unwrap());
        let original_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(original.clone())]));
        let relocated_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(relocated.clone())]));
        let symbolic = slice_at_one_program(&definition, original_type);
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(original, relocated).unwrap();
        let renamed = symbolic.clone().rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed.input_types(), vec![ArrayIrType::Array(relocated_type)]);
        assert!(matches!(
            symbolic.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "reference slicing requires a static referent type but got `f64[original]`",
        ));
        assert!(matches!(
            renamed.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "reference slicing requires a static referent type but got `f64[relocated]`",
        ));
        assert_eq!(invocations.load(Ordering::SeqCst), 5);
        let seed_type: ArrayIrType = ArrayType::new_static(DataType::F64, [1]).into();
        let key = |input_extent: usize, seed_type: Option<ArrayIrType>, destination_kind| {
            CustomRuleBackwardSpecializationKey {
                leading_input_types: Vec::new(),
                seed_extent_count: 0,
                input_tangent_types: vec![ArrayType::new_static(DataType::F64, [input_extent]).into()],
                output_tangent_types: vec![ArrayType::new_static(DataType::F64, [1]).into()],
                seed_types: vec![seed_type],
                destination_kinds: vec![destination_kind],
                levels: Vec::new(),
                discharged: false,
            }
        };
        assert_eq!(
            definition.caches().backward_specializations.keys(),
            vec![
                key(4, Some(seed_type.clone()), CotangentDestinationKind::Return),
                key(3, Some(seed_type.clone()), CotangentDestinationKind::Reference),
                key(3, Some(seed_type), CotangentDestinationKind::Return),
            ],
        );

        // The specializations contain only ordinary operations, so dropping the programs releases the definition.
        drop((definition, program, wider, symbolic, renamed));
        assert!(retained.upgrade().is_none());
    }

    #[test]
    fn test_composite_transpose_transposition_retained_rules_disconnected() {
        let stash = ArrayReference::new(Array::scalar(0.0f64).unwrap());
        let mut builder = IrBuilder::new();
        let reference = builder.add_input(ArrayIrValue::Reference(stash.clone()).r#type().into_owned());
        let tangent = builder.add_input(ArrayType::scalar(DataType::F64).into());
        builder
            .add_instruction(ignored_zero_effect_carrier(), Vec::new(), vec![reference, tangent], None)
            .unwrap();
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                Vec::new(),
                vec![Placeholder, Placeholder],
                Vec::new(),
            )
            .unwrap();
        assert!(program.entry_region_ref().effects().has_deferred_work());

        // Simplification retains the selected obligation even though the carrier's output is unused.
        assert_eq!(
            program.simplified().unwrap().to_string(),
            indoc! {"
                lambda %0:ref<f64[]>, %1:f64[] .
                let %2:f64[] = composite_transpose [name=\"ignored_zero_effect\", leading_input_count=1] %0 %1
                in ()
            "}
            .trim_end(),
        );
        let transposed = program.transpose_with_respect_to(&[1], &[CotangentDestinationKind::Ignore]).unwrap();
        assert_eq!(stash.read(), Ok(Array::scalar(0.0f64).unwrap()));
        assert_eq!(transposed.interpret(vec![ArrayIrValue::Reference(stash.clone())]), Ok(Vec::new()));
        assert_eq!(stash.read(), Ok(Array::scalar(1.0f64).unwrap()));
        assert_eq!(transposed.interpret(vec![ArrayIrValue::Reference(stash.clone())]), Ok(Vec::new()));
        assert_eq!(stash.read(), Ok(Array::scalar(2.0f64).unwrap()));
    }

    #[test]
    fn test_composite_transpose_transposition_retained_rules_condition() {
        let stash = ArrayReference::new(Array::scalar(0.0f64).unwrap());
        let reference_type = ArrayIrValue::Reference(stash.clone()).r#type().into_owned();
        let tangent_type: ArrayIrType = ArrayType::scalar(DataType::F64).into();
        let mut true_branch = IrBuilder::new();
        let tangent = true_branch.add_input(tangent_type.clone());
        let reference = true_branch.add_input(reference_type.clone());
        true_branch
            .add_instruction(ignored_zero_effect_carrier(), Vec::new(), vec![reference, tangent], None)
            .unwrap();
        let true_branch = true_branch
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                Vec::new(),
                vec![Placeholder, Placeholder],
                Vec::new(),
            )
            .unwrap();
        let mut false_branch = IrBuilder::new();
        false_branch.add_input(tangent_type.clone());
        false_branch.add_input(reference_type.clone());
        let false_branch = false_branch
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                Vec::new(),
                vec![Placeholder, Placeholder],
                Vec::new(),
            )
            .unwrap();
        let mut builder = IrBuilder::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayIrValue::Array(Array::scalar(true).unwrap()).r#type().into_owned());
        let tangent = builder.add_input(tangent_type);
        let reference = builder.add_input(reference_type);
        builder
            .add_instruction(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, tangent, reference],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                Vec::new(),
                vec![Placeholder, Placeholder, Placeholder],
                Vec::new(),
            )
            .unwrap();
        assert!(program.entry_region_ref().effects().has_deferred_work());
        let parent = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let known_inputs = program
            .input_types()
            .into_iter()
            .map(|input_type| PartialValue::Known(parent.input(input_type)))
            .collect::<Vec<_>>();
        let evaluation = program.partially_evaluate_in_context(&parent, &known_inputs).unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[], %2:ref<f64[]> .
                let () = condition %0 %1 %2 [
                    true={
                        lambda %0:f64[], %1:ref<f64[]> .
                        let %2:f64[] = composite_transpose [name=\"ignored_zero_effect\", leading_input_count=1] %1 %0
                        in ()
                    },
                    false={
                        lambda %0:f64[], %1:ref<f64[]> .
                        in ()
                    },
                ]
                in ()
            "}
            .trim_end(),
        );
        assert!(evaluation.program().entry_region_ref().effects().has_deferred_work());
        let transposed =
            evaluation.program().transpose_with_respect_to(&[1], &[CotangentDestinationKind::Ignore]).unwrap();
        assert_eq!(stash.read(), Ok(Array::scalar(0.0f64).unwrap()));
        assert_eq!(
            transposed.interpret(vec![
                ArrayIrValue::Array(Array::scalar(false).unwrap()),
                ArrayIrValue::Reference(stash.clone()),
            ]),
            Ok(Vec::new()),
        );
        assert_eq!(stash.read(), Ok(Array::scalar(0.0f64).unwrap()));
        assert_eq!(
            transposed.interpret(vec![
                ArrayIrValue::Array(Array::scalar(true).unwrap()),
                ArrayIrValue::Reference(stash.clone()),
            ]),
            Ok(Vec::new()),
        );
        assert_eq!(stash.read(), Ok(Array::scalar(1.0f64).unwrap()));
    }

    #[test]
    fn test_composite_transpose_transposition_retained_rules_in_attached_backward_rule() {
        let stash = ArrayReference::new(Array::scalar(0.0f64).unwrap());
        let reference_type = ArrayIrValue::Reference(stash.clone()).r#type().into_owned();
        let tangent_type: ArrayIrType = ArrayType::scalar(DataType::F64).into();
        let attached_carrier =
            ArrayIrOperation::<Array>::CompositeTranspose(CompositeTransposeOperation::from_backward_region(
                1,
                vec![tangent_type.clone()],
                vec![tangent_type.clone()],
            ));

        // Builds a backward program over `(reference, seed)` that returns its seed after the instructions that
        // `populate` appends.
        let backward = |populate: &dyn Fn(&mut IrBuilder, AtomId, AtomId)| {
            let mut builder = IrBuilder::new();
            let reference = builder.add_input(reference_type.clone());
            let seed = builder.add_input(tangent_type.clone());
            populate(&mut builder, reference, seed);
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![seed],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let deferred_work = |builder: &mut IrBuilder, reference: AtomId, seed: AtomId| {
            builder
                .add_instruction(ignored_zero_effect_carrier(), Vec::new(), vec![reference, seed], None)
                .unwrap();
        };

        // Builds a program over `(reference, tangent)` that applies a carrier with an attached backward rule, with the
        // reference as its leading input, whose backward program is `backward`.
        let attached_carrier_program =
            |backward: Program<ArrayIrValue<Array>, ArrayIrOperation<Array>, Vec<_>, Vec<_>>| {
                let mut builder = IrBuilder::new();
                let backward = builder.import_program(backward);
                let reference = builder.add_input(reference_type.clone());
                let tangent = builder.add_input(tangent_type.clone());
                let output = builder
                    .add_instruction(attached_carrier.clone(), vec![backward], vec![reference, tangent], None)
                    .unwrap()[0];
                builder
                    .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                        vec![output],
                        vec![Placeholder, Placeholder],
                        vec![Placeholder],
                    )
                    .unwrap()
            };

        // The backward program is a deferred rule of the carrier, so its deferred work makes the forward program carry
        // an obligation. Transposition selects it, and its deferred work must be staged even though the ignored
        // destination makes every accumulator unneeded, which would otherwise let the attached carrier skip its
        // backward program.
        let program = attached_carrier_program(backward(&deferred_work));
        assert!(program.entry_region_ref().effects().has_deferred_work());
        let transposed = program.transpose_with_respect_to(&[1], &[CotangentDestinationKind::Ignore]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[], %1:ref<f64[]> .
                let %2:f64[] = composite_transpose [name=\"ignored_zero_effect\", leading_input_count=1] %1 %0
                in ()
            "}
            .trim_end(),
        );

        // A nested attached carrier whose backward program has deferred work is itself an obligation, so the selected
        // backward program is replayed and stages that carrier into the pullback, where it remains an obligation of any
        // later transposition (rather than disappearing with the unneeded accumulators).
        let program = attached_carrier_program(backward(&|builder, reference, seed| {
            let nested = builder.import_program(backward(&deferred_work));
            builder
                .add_instruction(attached_carrier.clone(), vec![nested], vec![reference, seed], None)
                .unwrap();
        }));
        let transposed = program.transpose_with_respect_to(&[1], &[CotangentDestinationKind::Ignore]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[], %1:ref<f64[]> .
                let %2:f64[] = composite_transpose [leading_input_count=1] %1 %0 [
                    backward={
                        lambda %0:ref<f64[]>, %1:f64[] .
                        let %2:f64[] = composite_transpose [name=\"ignored_zero_effect\", leading_input_count=1] %0 %1
                        in (%1)
                    },
                ]
                in ()
            "}
            .trim_end(),
        );

        // A nested attached carrier with a pure backward program is no obligation, so the backward program is skipped.
        let program = attached_carrier_program(backward(&|builder, reference, seed| {
            let nested = builder.import_program(backward(&|_, _, _| {}));
            builder
                .add_instruction(attached_carrier.clone(), vec![nested], vec![reference, seed], None)
                .unwrap();
        }));
        assert!(!program.entry_region_ref().effects().has_deferred_work());
        let transposed = program.transpose_with_respect_to(&[1], &[CotangentDestinationKind::Ignore]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[], %1:ref<f64[]> .
                in ()
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_composite_transpose_transposition_retained_rules_effectful_buffered_zero_seed() {
        // The backward rule records a reference effect whenever its differentiated input's cotangent goes to a caller
        // buffer, including for a structural-zero seed that contributes nothing to that buffer.
        let definition = CustomRuleRegistration::new(IrDefinition::new("buffered_zero_effect").with_accumulating_vjp(
            |_| unreachable!("the tests construct the carrier directly"),
            |context, inputs, seeds, accumulators| {
                if accumulators[1].reference(context)?.is_some() && matches!(&seeds[0], MaybeZero::Zero(_)) {
                    let stash = inputs[0].as_known().unwrap().clone();
                    let increment = context.lift(ArrayIrValue::Array(Array::scalar(1.0f64)?))?;
                    context.bind(
                        ArrayIrOperation::ReferenceAddUpdate(ReferenceAddUpdateOperation::new()),
                        Vec::new(),
                        &[stash, increment],
                    )?;
                }
                Ok(())
            },
        ));
        let stash = ArrayReference::new(Array::scalar(0.0f64).unwrap());
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F64).into();
        let mut builder = IrBuilder::new();
        let reference = builder.add_input(ArrayIrValue::Reference(stash.clone()).r#type().into_owned());
        let tangent = builder.add_input(scalar_type.clone());
        let carrier = retained_carrier(definition.reference(), 1, vec![scalar_type.clone()], vec![scalar_type]);
        builder.add_instruction(carrier, Vec::new(), vec![reference, tangent], None).unwrap();
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                Vec::new(),
                vec![Placeholder, Placeholder],
                Vec::new(),
            )
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[1], &[CotangentDestinationKind::Reference]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:ref<f64[]>, %1:ref<f64[]> .
                let %2:f64[] = const 1.0
                    () = reference_add_update %1 %2
                in ()
            "}
            .trim_end(),
        );
        let buffer = ArrayReference::new(Array::scalar(5.0f64).unwrap());
        assert_eq!(
            transposed.interpret(vec![ArrayIrValue::Reference(buffer.clone()), ArrayIrValue::Reference(stash.clone())]),
            Ok(Vec::new()),
        );
        assert_eq!(stash.read(), Ok(Array::scalar(1.0f64).unwrap()));
        assert_eq!(buffer.read(), Ok(Array::scalar(5.0f64).unwrap()));

        // The returned specialization of the same rule records no effect.
        let returned = program.transpose_with_respect_to(&[1], &[CotangentDestinationKind::Return]).unwrap();
        assert_eq!(
            returned.interpret(vec![ArrayIrValue::Reference(stash.clone())]),
            Ok(vec![Array::scalar(0.0f64).unwrap().into()]),
        );
        assert_eq!(stash.read(), Ok(Array::scalar(1.0f64).unwrap()));
    }

    #[test]
    fn test_composite_transpose_transposition_retained_rules_zero_seeds() {
        // The duplicating rule `x ↦ (x, x)` saves no residuals and returns the cotangent `ȳ₁` of its second output,
        // which the caller leaves unused, so its seed is a structural zero.
        let duplicate_program = |definition: &CustomRuleRegistration<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
                                 r#type: &ArrayIrType| {
            let mut primal = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let input = primal.add_input(r#type.clone());
            let primal = primal
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![input, input],
                    vec![Placeholder],
                    vec![Placeholder; 2],
                )
                .unwrap();
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let primal = builder.import_program(primal);
            let input = builder.add_input(r#type.clone());
            let operation = ArrayIrOperation::Composite(CompositeOperation::new(definition.reference()));
            let output = builder.add_instruction(operation, vec![primal], vec![input], None).unwrap()[0];
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![output],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let renderings = |program: &ConvertedProgram| {
            let reverse = program
                .entry_region_ref()
                .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose)
                .unwrap();
            let pullback = reverse.tangent().transpose_with_respect_to(&[0], &[]).unwrap();
            (reverse.primal().to_string(), reverse.tangent().to_string(), pullback.to_string())
        };

        // A rule that receives materialized seeds sees a zero of the dynamically shaped output. No known leading input
        // carries its runtime extent, so the carrier receives that extent as a seed extent, which is read from the
        // primal output and which the rule never sees.
        let extent = DimensionVariable::new("n", DimensionBounds::new(1, Some(8)).unwrap());
        let r#type: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)])).into();
        let definition = CustomRuleRegistration::new(IrDefinition::new("duplicate").with_vjp(
            |primals| Ok((vec![primals[0].clone(), primals[0].clone()], Vec::new())),
            |leading_inputs, seeds| {
                assert!(leading_inputs.is_empty());
                Ok(vec![seeds[1].clone()])
            },
        ));
        assert_eq!(
            renderings(&duplicate_program(&definition, &r#type)),
            (
                indoc! {"
                    lambda %0:f32[n] .
                    let %1:dimension<n ∈ [1, 8)> = dimension_size [axis=0] %0
                    in (%0, %1)
                "}
                .trim_end()
                .to_string(),
                indoc! {"
                    lambda %0:f32[n], %1:dimension<n ∈ [1, 8)> .
                    let %2:f32[n], %3:f32[n] = composite_transpose [name=\"duplicate\", leading_input_count=1, \
                    seed_extent_count=1] %1 %0
                    in (%2)
                "}
                .trim_end()
                .to_string(),
                indoc! {"
                    lambda %0:f32[n], %1:dimension<n ∈ [1, 8)> .
                    let %2:f32[n] = zero [type=f32[n]] %1
                    in (%2)
                "}
                .trim_end()
                .to_string(),
            ),
        );

        // A rule that receives structural-zero seeds sees the unused output's seed as a `MaybeZero::Zero` leaf.
        let observed = Arc::new(Mutex::new(Vec::new()));
        let definition = CustomRuleRegistration::new(IrDefinition::new("duplicate").with_symbolic_zero_vjp(
            |primals| Ok((vec![primals[0].clone(), primals[0].clone()], Vec::new())),
            {
                let observed = observed.clone();
                move |_, seeds| {
                    observed.lock().unwrap().push(seeds.iter().map(MaybeZero::is_zero).collect::<Vec<_>>());
                    let MaybeZero::Value(seed) = &seeds[0] else {
                        return Err(ProgramError::InvalidArgument { message: "unexpected zero seed".to_string() });
                    };
                    Ok(vec![MaybeZero::Value(seed.clone())])
                }
            },
        ));
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        assert_eq!(
            renderings(&duplicate_program(&definition, &scalar_type)).2,
            indoc! {"
                lambda %0:f32[] .
                in (%0)
            "}
            .trim_end(),
        );
        assert_eq!(*observed.lock().unwrap(), vec![vec![false, true]]);
    }

    #[test]
    fn test_composite_transpose_transposition_retained_rules_after_un_projection() {
        /// Array-member carrier payload retaining a definition declared in the array IR family.
        #[derive(Clone, Debug)]
        struct MemberCarrier {
            /// Definition whose backward rule the converted carrier applies.
            rules: CustomRuleReference<ArrayIrValue<Array>, ArrayIrOperation<Array>>,

            /// Tangent type of the differentiated input.
            input_tangent_type: ArrayType,
        }

        impl Operation for MemberCarrier {
            type Type = ArrayType;

            fn name(&self) -> &'static str {
                "member_carrier"
            }

            fn infer_output_types(
                &self,
                _input_types: &[ArrayType],
                _region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                Ok(vec![ArrayType::new_static(DataType::F64, [1])])
            }
        }

        impl From<MemberCarrier> for ArrayIrOperation<Array> {
            fn from(operation: MemberCarrier) -> Self {
                retained_carrier(
                    operation.rules,
                    0,
                    vec![operation.input_tangent_type.into()],
                    vec![ArrayType::new_static(DataType::F64, [1]).into()],
                )
            }
        }

        let invocations = Arc::new(AtomicUsize::new(0));
        let definition = slice_at_one_definition(&invocations);
        let retained = Arc::downgrade(&definition.definition());
        let input_type = ArrayType::new_static(DataType::F64, [4]);
        let mut builder = ProgramBuilder::<Array, MemberCarrier>::new();
        let input = builder.add_input(input_type.clone());
        let carrier = MemberCarrier { rules: definition.reference(), input_tangent_type: input_type };
        let output = builder.add_instruction(carrier, Vec::new(), vec![input], None).unwrap()[0];
        let member =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        // This canonical conversion changes the stored value and type families as well as the operation payload. It
        // carries a definition already typed over array IR values; it does not make a Rust closure family-polymorphic.
        let converted = member.into_unprojected::<ArrayIrValue<Array>, ArrayIrOperation<Array>>().unwrap();
        let ArrayIrOperation::CompositeTranspose(operation) = converted.instructions()[0].operation() else {
            unreachable!()
        };
        assert_eq!(operation.rules(), Some(&definition.reference()));
        assert_eq!(invocations.load(Ordering::SeqCst), 0);

        // The first specialization happens after the conversion.
        let buffered = converted.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap();
        assert_eq!(invocations.load(Ordering::SeqCst), 1);
        assert_eq!(
            buffered.to_string(),
            indoc! {"
                lambda %0:f64[1], %1:ref<f64[4]> .
                let () = reference_add_update [transforms=[slice(axes=[1:2])]] %1 %0
                in ()
            "}
            .trim_end(),
        );
        let seed = ArrayIrValue::Array(Array::vector(vec![3.0f64]).unwrap());
        let buffer = ArrayReference::new(Array::vector(vec![10.0f64, 20.0, 30.0, 40.0]).unwrap());
        assert_eq!(buffered.interpret(vec![seed, buffer.clone().into()]), Ok(vec![]));
        assert_eq!(buffer.read(), Ok(Array::vector(vec![10.0f64, 23.0, 30.0, 40.0]).unwrap()));
        let repeated =
            converted.clone().transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap();
        assert_eq!(repeated.to_string(), buffered.to_string());
        assert_eq!(invocations.load(Ordering::SeqCst), 1);
        drop((definition, converted));
        assert!(retained.upgrade().is_none());
    }
}
