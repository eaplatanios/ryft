//! The [`CustomFunctionOperation`] that calls a primal program with custom derivative rules and the
//! [`CustomFunctionTransposeOperation`] carrier that reverse mode stages for its reverse-mode rules. Both operations
//! hold their rules either as attached rule regions or as retained rule sets (refer to [`CustomRuleSource`]).

// TODO(eaplatanios): Review this module.

use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::LazyLock;

use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchedProgram, BatchingContext, BatchingDriver, BatchingError,
    ProgramBatchingOutputAxesPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    CotangentAccumulator, CotangentBatchingPolicy, CotangentDestinationKind, DifferentiableOperation,
    DifferentiableType, DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    DifferentiationPolicy, ResidualZeroProvider, TransposableOperation, TranspositionContext, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, check_types, impl_non_transposable_operation};
use crate::operations::arithmetic::AddOperation;
use crate::operations::constants::zero::Zero;
use crate::operations::custom_functions::rules::{
    CustomRuleBackwardSpecializationKey, CustomRuleBatching, CustomRuleBatchingLevel, CustomRuleBatchingRuleKey,
    CustomRuleDerivationKey, CustomRuleDerivationKind, CustomRuleReference, CustomRuleSource,
    CustomRuleSpecializationKey, CustomRuleSpecializer, CustomRuleTracer, CustomVjpBackward, boundary_input_count,
    validate_rule_output_types,
};
use crate::parameters::{Parameter, Placeholder};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    Effects, EffectsSummary, InputRegionProvenance, MaybeZero, Operation, OperationFormatter, OutputRegionProvenance,
    ProgramBuilder, ProgramError, ReferenceBoundary, ReferenceDischargeContext, ReferenceDischargeDriver,
    ReferenceDischargePolicy, ReferenceDischargeValue, ReferenceDischargeableOperation, RegionInterface, RegionRef,
    RegionSlot, Type, TypeError, TypeIdentityRenaming, TypeRefinements, Typed, Value,
    discharge_local_reference_operation, discharge_reference_free_operation,
};

/// Canonical operation name for [`CustomFunctionOperation`].
pub const CUSTOM_FUNCTION_OPERATION_NAME: &str = "custom_function";

/// Forward-mode (i.e., Jacobian-Vector Product or JVP) rule of a [`CustomFunctionOperation`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum CustomFunctionJvpRule {
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

/// Derivative rules of a [`CustomFunctionOperation`], in one of its two representations.
#[derive(Clone, PartialEq, Eq, Hash)]
enum CustomFunctionRules<T, S> {
    /// Rule programs attached as rule regions that follow the primal region.
    Attached {
        /// Forward-mode rule, whose explicit form is a `"jvp"` rule region.
        jvp_rule: CustomFunctionJvpRule,

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
    },
}

/// Higher-order [`Operation`] that calls a primal program with custom derivative rules. It is the operation that
/// [`custom_function`](fn@crate::custom_function) functions stage, and it is the one boundary through which every
/// transform treats a function whose derivative is not the derivative of its implementation (e.g., a fused or foreign
/// kernel, or a numerically stabilized derivative). The primal program is attached as the computation region
/// `"primal"`, and ordinary execution, lowering, and partial evaluation use it exclusively. A call has any of the
/// following rules:
///
///   - a forward-mode rule (refer to [`CustomFunctionJvpRule`]), which is either an explicit Jacobian-Vector Product
///     (JVP) rule or the derivative of the primal region, and
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
/// [`CustomRuleDefinition`](crate::CustomRuleDefinition)), so an ordinary call traces only its primal and never
/// declares the effects of its
///     rules. Each rule is traced on the first request of each static specialization and cached by the definition's
///     [`CustomRuleRegistration`](crate::CustomRuleRegistration).
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
/// [`CustomFunction::call`](crate::CustomFunction::call) checks on the traced forward program). Keeping `p` explicit
/// while omitting its tangent and cotangent from the rules distinguishes an input that parameterizes the rules from an
/// ordinary input whose derivative merely happens to be zero. This is the same input split that
/// [`LinearCallOperation`](crate::LinearCallOperation) draws with its residual count. Batching is a canonical producer
/// of such inputs: a batching policy that threads batching state through a structurally batched region's boundary
/// (e.g., a composite universe's first-class mapped extent) reintroduces that state as additional leading
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
/// is instead an explicit registration choice (i.e., [`CustomFunctionJvpRule::Primal`]). Reverse mode replays the
/// forward rule and stages a [`CustomFunctionTransposeOperation`] carrier for the backward rule, in the same
/// representation as the call.
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
/// level (refer to [`CustomRuleDefinition::with_batching`](crate::CustomRuleDefinition::with_batching)). A derivative
/// whose natural layout is incompatible with the declared layout is therefore rejected when it is first requested, not
/// when the call is batched. In both representations, any boundary inputs of the batching policy (e.g., a first-class
/// batch extent) become leading non-differentiated inputs.
///
/// A custom batching rule also stays on the path of forward-mode derivatives that are batched after they are taken.
/// Forward-mode differentiation of an unbatched call with retained rules whose forward-mode rule is derived from its
/// primal stages a derived call (with rules that
/// [`CustomRuleSpecializer::derived`] derives) instead of inlining the derivative
/// of the primal region, and batching that call applies the derivative of the batching rule. The derived call's primal
/// region is that derivative: in fused contexts, it computes the outputs and their tangents, while partitioned contexts
/// compute the outputs with the call itself and stage a pushforward that computes only the tangents (for a pure primal
/// region, since the pushforward recomputes the primal; other primals are linearized inline, so that their effects run
/// once). Reverse mode inlines the derivative, because derived calls are not transposable.
///
/// Equality and hashing of calls with retained rules use the identity of the shared definition (i.e.,
/// [`Arc::ptr_eq`](std::sync::Arc::ptr_eq)), while rendering prints its human-readable name.
pub struct CustomFunctionOperation<V: Typed + Parameter, O, S = CustomRuleReference<V, O>> {
    /// Number of leading inputs that parameterize the call without being differentiated, including the boundary inputs
    /// that the batching levels of a call with retained rules prepended to its inputs.
    non_differentiated_count: usize,

    /// Derivative rules of the call.
    rules: CustomFunctionRules<V::Type, S>,

    /// Marker for the operation family, which only the rule source and the rule regions name.
    marker: PhantomData<fn() -> O>,
}

impl<V: Typed<Type: DifferentiableType + Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>>
    CustomFunctionOperation<V, O, S>
{
    /// Creates a new call of the retained rules of `rules`, whose inputs are all differentiated.
    #[inline]
    pub fn new(rules: S) -> Self {
        Self {
            non_differentiated_count: 0,
            rules: CustomFunctionRules::Retained { rules, batching: None, discharged: false },
            marker: PhantomData,
        }
    }

    /// Creates a new call with attached rule programs, whose inputs are all differentiated. An explicit forward-mode
    /// rule (i.e., [`CustomFunctionJvpRule::Explicit`]) is a `"jvp"` rule region that follows the primal region,
    /// and reverse-mode rules are `"forward"` and `"backward"` rule regions that follow the primal region and the
    /// optional `"jvp"` rule region.
    #[inline]
    pub fn from_rule_regions(jvp_rule: CustomFunctionJvpRule, has_vjp_rule: bool) -> Self {
        Self {
            non_differentiated_count: 0,
            rules: CustomFunctionRules::Attached { jvp_rule, has_vjp_rule },
            marker: PhantomData,
        }
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
            CustomFunctionRules::Attached { .. } => None,
            CustomFunctionRules::Retained { rules, .. } => Some(rules),
        }
    }

    /// Returns the forward-mode rule of this call.
    pub fn jvp_rule(&self) -> CustomFunctionJvpRule {
        match &self.rules {
            CustomFunctionRules::Attached { jvp_rule, .. } => *jvp_rule,
            CustomFunctionRules::Retained { rules, .. } if rules.has_jvp() => CustomFunctionJvpRule::Explicit,
            CustomFunctionRules::Retained { rules, .. } if rules.derives_jvp_from_primal() => {
                CustomFunctionJvpRule::Primal
            }
            CustomFunctionRules::Retained { .. } => CustomFunctionJvpRule::Absent,
        }
    }

    /// Returns whether this call has reverse-mode forward and backward rules.
    pub fn has_vjp_rule(&self) -> bool {
        match &self.rules {
            CustomFunctionRules::Attached { has_vjp_rule, .. } => *has_vjp_rule,
            CustomFunctionRules::Retained { rules, .. } => rules.has_vjp(),
        }
    }

    /// Returns this call with the provided number of leading non-differentiated inputs. Refer to the documentation of
    /// [`CustomFunctionOperation`] for the impact of this property on the rule interfaces. For a batched call with
    /// retained rules, this count includes the boundary inputs that its batching levels prepended to its inputs.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] when the call is a batched call with retained rules and `non_differentiated_count` is
    /// smaller than the number of those boundary inputs, which are always non-differentiated.
    pub fn with_non_differentiated_count(mut self, non_differentiated_count: usize) -> Result<Self, TypeError> {
        if let CustomFunctionRules::Retained { rules, batching: Some(batching), .. } = &self.rules {
            let boundary_input_count = batching.boundary_input_count();
            if non_differentiated_count < boundary_input_count {
                return Err(TypeError::invalid(format!(
                    "batched `{CUSTOM_FUNCTION_OPERATION_NAME}` `{}` must have at least {boundary_input_count} \
                     non-differentiated inputs, which are the boundary inputs that its batching levels prepended to \
                     its inputs, but got {non_differentiated_count}",
                    rules.name(),
                )));
            }
        }
        self.non_differentiated_count = non_differentiated_count;
        Ok(self)
    }

    /// Converts this call into the family `(V2, O2)` with the rule source that `rules_fn` derives from its current
    /// source. This is how a call follows its program into a family that contains its family (e.g., with
    /// [`LiftedCustomRules::Member`](crate::LiftedCustomRules::Member) when an array program is converted into an
    /// array IR program). A call with attached rules carries no source, so `rules_fn` is only invoked for retained
    /// rules. Refer to [`Self::into_attached_family`] for converting calls with attached rules into a family whose
    /// source type the caller chooses independently.
    pub fn into_family<V2, O2, S2, F>(self, rules_fn: F) -> CustomFunctionOperation<V2, O2, S2>
    where
        V2: Typed<Type: DifferentiableType + Eq + Hash + From<V::Type>> + Parameter,
        F: FnOnce(S) -> S2,
    {
        CustomFunctionOperation {
            non_differentiated_count: self.non_differentiated_count,
            rules: match self.rules {
                CustomFunctionRules::Attached { jvp_rule, has_vjp_rule } => {
                    CustomFunctionRules::Attached { jvp_rule, has_vjp_rule }
                }
                CustomFunctionRules::Retained { rules, batching, discharged } => CustomFunctionRules::Retained {
                    rules: rules_fn(rules),
                    batching: batching.map(CustomRuleBatching::map_types),
                    discharged,
                },
            },
            marker: PhantomData,
        }
    }

    /// Converts this call into the family `(V2, O2)` with any rule source type when its rules are attached, because
    /// attached rules are regions that follow their call independently of any source. Returns this call unchanged when
    /// its rules are retained, which [`Self::into_family`] converts instead. Family conversions use this function to
    /// normalize every call with attached rules into the family's native variant.
    pub fn into_attached_family<V2, O2, S2>(self) -> Result<CustomFunctionOperation<V2, O2, S2>, Self>
    where
        V2: Typed + Parameter,
    {
        match self.rules {
            CustomFunctionRules::Attached { jvp_rule, has_vjp_rule } => Ok(CustomFunctionOperation {
                non_differentiated_count: self.non_differentiated_count,
                rules: CustomFunctionRules::Attached { jvp_rule, has_vjp_rule },
                marker: PhantomData,
            }),
            CustomFunctionRules::Retained { .. } => Err(self),
        }
    }

    /// Returns a description of this call for diagnostics, which names the definition of retained rules unless it has
    /// the default name of [`custom_function`](crate::custom_function) functions.
    fn description(&self) -> String {
        match &self.rules {
            CustomFunctionRules::Retained { rules, .. } if rules.name() != CUSTOM_FUNCTION_OPERATION_NAME => {
                format!("a `{CUSTOM_FUNCTION_OPERATION_NAME}` call of `{}`", rules.name())
            }
            _ => format!("a `{CUSTOM_FUNCTION_OPERATION_NAME}` call"),
        }
    }

    /// Returns the index of the `"jvp"` rule region, if this call has attached rules with one.
    #[inline]
    fn jvp_region_index(&self) -> Option<usize> {
        matches!(self.rules, CustomFunctionRules::Attached { jvp_rule: CustomFunctionJvpRule::Explicit, .. })
            .then_some(1)
    }

    /// Returns the indices of the `"forward"` and `"backward"` rule regions, if this call has attached rules with
    /// them.
    #[inline]
    fn vjp_region_indices(&self) -> Option<(usize, usize)> {
        let CustomFunctionRules::Attached { has_vjp_rule: true, .. } = self.rules else {
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

    /// Validates the custom function contract over the attached region interfaces (refer to the documentation of
    /// [`CustomFunctionOperation`] for that contract) and returns the primal interface.
    fn validated_interfaces<'i>(
        &self,
        region_interfaces: &'i [RegionInterface<V::Type>],
    ) -> Result<&'i RegionInterface<V::Type>, TypeError> {
        check_count!("region", region_interfaces, self.region_slots().len(), TypeError);
        let primal_interface = &region_interfaces[0];
        let input_types = primal_interface.input_types();
        let output_types = primal_interface.output_types();
        let (non_differentiated_types, differentiated_types) = self.split_inputs(input_types)?;
        validate_custom_function_reference_boundary(
            CUSTOM_FUNCTION_OPERATION_NAME,
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
            check_types!(@same, format!("`{CUSTOM_FUNCTION_OPERATION_NAME}` JVP rule input"), [
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
            check_types!(@same, format!("`{CUSTOM_FUNCTION_OPERATION_NAME}` JVP rule output"), [
                &expected_jvp_output_types,
                jvp_interface.output_types(),
            ]);
        }

        if let Some((forward_region_index, backward_region_index)) = self.vjp_region_indices() {
            let forward_interface = &region_interfaces[forward_region_index];
            let backward_interface = &region_interfaces[backward_region_index];
            check_types!(@same, format!("`{CUSTOM_FUNCTION_OPERATION_NAME}` forward rule input"), [
                input_types,
                forward_interface.input_types(),
            ]);
            let forward_output_types = forward_interface.output_types();
            if forward_output_types.len() < output_types.len() {
                return Err(TypeError::invalid(format!(
                    "`{}` forward rule must produce at least the {} primal output(s) but produced {} value(s)",
                    CUSTOM_FUNCTION_OPERATION_NAME,
                    output_types.len(),
                    forward_output_types.len(),
                )));
            }
            check_types!(@same, format!("`{CUSTOM_FUNCTION_OPERATION_NAME}` forward rule output"), [
                output_types,
                &forward_output_types[..output_types.len()],
            ]);

            // Reference residuals must have types compatible with distinct leading non-differentiated inputs. This is
            // only a necessary interface constraint: two inputs can have the same reference type. Tracing the forward
            // rule of a `CustomFunction` checks the actual forwarded input identities; raw program producers must
            // ensure that same contract when attaching their forward regions.
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
                            "`{CUSTOM_FUNCTION_OPERATION_NAME}` forward rule returns residual {index} of reference \
                             type `{residual_type}`, but every leading non-differentiated input of that type is \
                             already forwarded by an earlier residual",
                        )));
                    }
                    None => {
                        return Err(TypeError::invalid(format!(
                            "`{CUSTOM_FUNCTION_OPERATION_NAME}` forward rule returns residual {index} of reference \
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
            check_types!(@same, format!("`{CUSTOM_FUNCTION_OPERATION_NAME}` backward rule input"), [
                &expected_backward_input_types,
                backward_interface.input_types(),
            ]);
            let expected_backward_output_types = differentiated_types
                .iter()
                .map(DifferentiableType::cotangent)
                .collect::<Result<Vec<_>, DifferentiationError>>()?;
            check_types!(@same, format!("`{CUSTOM_FUNCTION_OPERATION_NAME}` backward rule output"), [
                &expected_backward_output_types,
                backward_interface.output_types(),
            ]);
        }
        Ok(primal_interface)
    }

    /// Returns the specialization key of this call with retained rules, given the current types of its inputs and
    /// outputs.
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
        input_types: Vec<V::Type>,
        output_types: Vec<V::Type>,
    ) -> Result<CustomRuleSpecializationKey<V::Type>, ProgramError> {
        Ok(match batching {
            None => CustomRuleSpecializationKey {
                input_types,
                output_types,
                non_differentiated_count: self.non_differentiated_count,
                tangent_activity: Vec::new(),
                levels: Vec::new(),
                discharged,
            },
            Some(batching) => CustomRuleSpecializationKey {
                input_types: batching.input_types.clone(),
                output_types: batching.output_types.clone(),
                non_differentiated_count: self
                    .non_differentiated_count
                    .checked_sub(batching.boundary_input_count())
                    .ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "batched `{CUSTOM_FUNCTION_OPERATION_NAME}` `{}` has fewer non-differentiated inputs \
                             than boundary inputs",
                            rules.name(),
                        ))
                    })?,
                tangent_activity: Vec::new(),
                levels: batching.levels.clone(),
                discharged,
            },
        })
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> Clone for CustomFunctionOperation<V, O, S> {
    fn clone(&self) -> Self {
        Self { non_differentiated_count: self.non_differentiated_count, rules: self.rules.clone(), marker: PhantomData }
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> Debug for CustomFunctionOperation<V, O, S> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Retained rules are identified by the name of their definition, whose own rendering would include its rules.
        let mut debug = formatter.debug_struct("CustomFunctionOperation");
        match &self.rules {
            CustomFunctionRules::Attached { jvp_rule, has_vjp_rule } => debug
                .field("non_differentiated_count", &self.non_differentiated_count)
                .field("jvp_rule", jvp_rule)
                .field("has_vjp_rule", has_vjp_rule),
            CustomFunctionRules::Retained { rules, batching, discharged } => debug
                .field("name", &rules.name())
                .field("non_differentiated_count", &self.non_differentiated_count)
                .field("batching", batching)
                .field("discharged", discharged),
        }
        .finish()
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> PartialEq
    for CustomFunctionOperation<V, O, S>
{
    fn eq(&self, other: &Self) -> bool {
        self.non_differentiated_count == other.non_differentiated_count && self.rules == other.rules
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> Eq for CustomFunctionOperation<V, O, S> {}

impl<V: Typed<Type: Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> Hash for CustomFunctionOperation<V, O, S> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.non_differentiated_count.hash(state);
        self.rules.hash(state);
    }
}

impl<V: Typed<Type: DifferentiableType + Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> Display
    for CustomFunctionOperation<V, O, S>
{
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<V: Typed<Type: DifferentiableType + Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> Operation
    for CustomFunctionOperation<V, O, S>
{
    type Type = V::Type;

    #[inline]
    fn name(&self) -> &'static str {
        CUSTOM_FUNCTION_OPERATION_NAME
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
                    CUSTOM_FUNCTION_OPERATION_NAME,
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
            CustomFunctionRules::Attached { .. } => {
                let primal_interface = self.validated_interfaces(region_interfaces)?;
                check_types!(@same, format!("`{CUSTOM_FUNCTION_OPERATION_NAME}` input"), [
                    primal_interface.input_types(),
                    input_types,
                ]);
                Ok(primal_interface.output_types().to_vec())
            }
            CustomFunctionRules::Retained { rules, batching, .. } => {
                check_count!("region", region_interfaces, 1, TypeError);
                validate_non_differentiated_count(
                    CUSTOM_FUNCTION_OPERATION_NAME,
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
                            "batched `{CUSTOM_FUNCTION_OPERATION_NAME}` `{}` has {} inputs but its batching levels \
                             record {} boundary inputs and {} unbatched inputs",
                            rules.name(),
                            input_types.len(),
                            boundary_input_count,
                            batching.input_types.len(),
                        )));
                    }
                }
                check_types!(@same, format!("`{CUSTOM_FUNCTION_OPERATION_NAME}` input"), [
                    region_interfaces[0].input_types(),
                    input_types,
                ]);
                validate_custom_function_reference_boundary(
                    CUSTOM_FUNCTION_OPERATION_NAME,
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

    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<V::Type as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        // Attached rules store no types, while retained rules store the types of their batching levels.
        let rules = match &self.rules {
            CustomFunctionRules::Attached { jvp_rule, has_vjp_rule } => {
                CustomFunctionRules::Attached { jvp_rule: *jvp_rule, has_vjp_rule: *has_vjp_rule }
            }
            CustomFunctionRules::Retained { rules, batching, discharged } => CustomFunctionRules::Retained {
                rules: rules.clone(),
                batching: batching.as_ref().map(|batching| batching.rename_identities(renaming)).transpose()?,
                discharged: *discharged,
            },
        };
        Ok(Self { non_differentiated_count: self.non_differentiated_count, rules, marker: PhantomData })
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        // Attached rule regions render under their slot names, so their fields appear only for the non-differentiated
        // split and for a forward-mode rule derived from the primal region, both exactly where they exist. Retained
        // rules render the name of their definition and the state of the call that their specializations depend on.
        let operation = OperationFormatter::new(formatter, indentation, CUSTOM_FUNCTION_OPERATION_NAME)?;
        match &self.rules {
            CustomFunctionRules::Attached { jvp_rule, .. } => {
                if self.non_differentiated_count == 0 && *jvp_rule != CustomFunctionJvpRule::Primal {
                    return Ok(());
                }
                operation.bracketed(|operation| {
                    if self.non_differentiated_count > 0 {
                        operation.field("non_differentiated_count", self.non_differentiated_count)?;
                    }
                    if *jvp_rule == CustomFunctionJvpRule::Primal {
                        operation.field("jvp_from_primal", true)?;
                    }
                    Ok(())
                })
            }
            CustomFunctionRules::Retained { rules, batching, discharged } => operation.bracketed(|operation| {
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
                Ok(())
            }),
        }
    }
}

impl<C, P, V, O, S> ReferenceDischargeableOperation<C, P> for CustomFunctionOperation<V, O, S>
where
    C: Context<Operation: From<CustomFunctionOperation<V, O, S>>>,
    P: ReferenceDischargePolicy<C>,
    V: Typed<Type: DifferentiableType + Eq + Hash> + Parameter,
    S: CustomRuleSource<V, O>,
    CustomFunctionOperation<V, O, S>: Operation<Type = C::Type>,
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
        match &self.rules {
            CustomFunctionRules::Attached { .. } => discharge_local_reference_operation(self, context, driver, inputs),
            CustomFunctionRules::Retained { rules, batching, .. } => {
                let discharged = Self {
                    non_differentiated_count: self.non_differentiated_count,
                    rules: CustomFunctionRules::Retained {
                        rules: rules.clone(),
                        batching: batching.clone(),
                        discharged: true,
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
impl<C: Domain, V: Typed<Type: DifferentiableType + Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>>
    InterpretableOperation<C> for CustomFunctionOperation<V, O, S>
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
    PartiallyEvaluatableOperation<C> for CustomFunctionOperation<C::Constant, C::Operation, S>
where
    C::Operation: From<CustomFunctionOperation<C::Constant, C::Operation, S>>,
{
}

impl<C, P, S> BatchableOperation<C, P> for CustomFunctionOperation<C::Constant, C::Operation, S>
where
    C: Context<
            Type: DifferentiableType + Eq + Hash,
            Operation: From<CustomFunctionOperation<C::Constant, C::Operation, S>>,
        >,
    P: CotangentBatchingPolicy<C>,
    S: CustomRuleSpecializer<C::Constant, C::Operation>,
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        let input_axes = inputs.iter().map(P::batch_axis).collect::<Vec<_>>();
        let primal_region = driver.region(0)?;
        let boundary_inputs = P::boundary_inputs(context.axis_extent());
        let (operation, regions, output_axes) = match &self.rules {
            CustomFunctionRules::Attached { .. } => {
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
                                CUSTOM_FUNCTION_OPERATION_NAME,
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
                                "batched `{CUSTOM_FUNCTION_OPERATION_NAME}` backward rule output axes \
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
            CustomFunctionRules::Retained { rules, batching, discharged } => {
                // Batch the primal now and record the level, so that the derivative rules are batched only when they
                // are first traced. A custom batching rule takes precedence over structurally batching the primal
                // region, and it declares the batch axes of the outputs. Otherwise, the declared layout is required
                // exactly, so an incompatible primal is rejected here. Boundary inputs reach the primal and every
                // batched rule program as leading non-differentiated inputs.
                let level = driver.batching_level(context)?;
                let (primal, output_axes) = if rules.has_batching_rule() {
                    let key = CustomRuleBatchingRuleKey {
                        level: level.clone(),
                        boundary_input_types: boundary_inputs.iter().map(|value| value.r#type().into_owned()).collect(),
                        input_types: inputs.iter().map(|input| P::value(input).r#type().into_owned()).collect(),
                        unbatched_input_types: primal_region.input_types(),
                        input_axes: input_axes.clone(),
                    };
                    let specialization = rules.batching_rule_specialization(key).map_err(ProgramError::from)?;
                    check_count!(
                        "output",
                        specialization.output_axes,
                        primal_region.output_types().len(),
                        ProgramError,
                    );
                    (specialization.program.clone(), specialization.output_axes.clone())
                } else {
                    let output_axes = rules.batched_call_output_axes(primal_region.output_types().len())?;
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
                    rules: CustomFunctionRules::Retained {
                        rules: rules.clone(),
                        batching: Some(batching),
                        discharged: *discharged,
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
        Ok(outputs
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| P::batch(output, axis))
            .collect::<Result<Vec<_>, _>>()?
            .into())
    }
}

impl<C, S> DifferentiableOperation<C> for CustomFunctionOperation<C::Constant, C::Operation, S>
where
    C: Context<
            Type: DifferentiableType + Eq + Hash,
            Operation: ResidualZeroProvider<C::Type, Operation = C::Operation>
                           + From<CustomFunctionOperation<C::Constant, C::Operation, S>>
                           + From<CustomFunctionTransposeOperation<C::Constant, C::Operation, S>>,
        > + Zero<C::Value>,
    S: CustomRuleSpecializer<C::Constant, C::Operation>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        validate_non_differentiated_count(CUSTOM_FUNCTION_OPERATION_NAME, self.non_differentiated_count, inputs.len())?;
        match (self.jvp_rule(), &self.rules) {
            // Apply the explicit pushforward directly instead of differentiating the primal region.
            (CustomFunctionJvpRule::Explicit, CustomFunctionRules::Attached { .. }) => replay_custom_jvp_rule(
                CUSTOM_FUNCTION_OPERATION_NAME,
                self.non_differentiated_count,
                context,
                driver,
                driver.region(1)?,
                inputs,
                None,
            ),

            // A retained rule is specialized for the tangent activity of the differentiated inputs, so structural zero
            // tangents stay symbolic inside the specialization.
            (CustomFunctionJvpRule::Explicit, CustomFunctionRules::Retained { rules, batching, discharged }) => {
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
                        input_types,
                        driver.region(0)?.output_types(),
                    )?
                };
                let specialization = rules.jvp_specialization(key)?;
                replay_custom_jvp_rule(
                    CUSTOM_FUNCTION_OPERATION_NAME,
                    self.non_differentiated_count,
                    context,
                    driver,
                    specialization.program.entry_region_ref(),
                    inputs,
                    Some(tangent_activity.as_slice()),
                )
            }
            // An unbatched call whose rules have a custom batching rule stages a derived call, which keeps that rule on
            // the path of derivatives that are batched after they are taken (refer to
            // `CustomRuleSpecializer::derived`). A batched call already computes its primal with its batching rule,
            // whose derivative is inlined.
            (CustomFunctionJvpRule::Primal, CustomFunctionRules::Retained { rules, batching: None, .. })
                if rules.has_batching_rule() =>
            {
                differentiate_primal_region(
                    CUSTOM_FUNCTION_OPERATION_NAME,
                    self.non_differentiated_count,
                    context,
                    driver,
                    inputs,
                    Some(self),
                )
            }
            (CustomFunctionJvpRule::Primal, _) => differentiate_primal_region(
                CUSTOM_FUNCTION_OPERATION_NAME,
                self.non_differentiated_count,
                context,
                driver,
                inputs,
                None,
            ),
            (CustomFunctionJvpRule::Absent, _) => {
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
        // derived from the primal is inlined, because the derived calls that forward mode stages are not transposable.
        if !self.has_vjp_rule() {
            if self.jvp_rule() == CustomFunctionJvpRule::Primal {
                return differentiate_primal_region(
                    CUSTOM_FUNCTION_OPERATION_NAME,
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
        // `validate_custom_function_replay`).
        let (non_differentiated_inputs, differentiated_inputs) = self.split_inputs(inputs)?;
        let primal_region = driver.region(0)?;
        let primal_output_types = primal_region.output_types();
        let output_count = primal_output_types.len();
        validate_custom_function_replay(
            CUSTOM_FUNCTION_OPERATION_NAME,
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
        // geometry of the output tangent types (e.g., the extents of a dynamically shaped output that the residuals do
        // not mention), so that geometry is read from the primal outputs and appended to the leading inputs as hidden
        // seed geometry, which the backward rule never receives. Equal geometry types name the same runtime quantity,
        // so each one is passed once.
        let capture_seed_geometry = |outputs: &[C::Value],
                                     geometry_types: &[C::Type],
                                     leading_values: &mut Vec<C::Value>|
         -> Result<Vec<C::Type>, ProgramError> {
            let mut seed_geometry_types = Vec::new();
            for (output, r#type) in outputs.iter().zip(geometry_types) {
                for geometry_type in C::Operation::zero_residual_types(r#type) {
                    if seed_geometry_types.contains(&geometry_type) {
                        continue;
                    }
                    let value = C::Operation::capture_zero_residual_value(context.primal(), output, &geometry_type)?
                        .ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "seed geometry of type `{geometry_type}` cannot be captured from output of type `{}`",
                                output.r#type().as_ref(),
                            ))
                        })?;
                    leading_values.push(value);
                    seed_geometry_types.push(geometry_type);
                }
            }
            Ok(seed_geometry_types)
        };
        let (outputs, seed_geometry_count, carrier_rules, carrier_regions) = match &self.rules {
            CustomFunctionRules::Attached { .. } => {
                // Replay the forward region on the dual primals, recovering the primal outputs followed by the
                // residuals. The carrier transposes by replaying the backward region, whose own inputs are exactly the
                // leading known inputs followed by the output cotangents.
                let (forward_region_index, backward_region_index) = self.vjp_region_indices().unwrap();
                let mut outputs =
                    driver.region(forward_region_index)?.interpret_in_context(context.primal(), primal_inputs)?;
                if outputs.len() < output_count {
                    return Err(ProgramError::MalformedProgram(format!(
                        "`{}` forward rule produced {} outputs which is fewer than its {} primal output(s)",
                        CUSTOM_FUNCTION_OPERATION_NAME,
                        outputs.len(),
                        output_count,
                    ))
                    .into());
                }
                leading_values.extend(outputs.split_off(output_count));
                let geometry_types =
                    outputs.iter().map(|output| output.r#type().tangent()).collect::<Result<Vec<_>, _>>()?;
                let seed_geometry_types = capture_seed_geometry(&outputs, &geometry_types, &mut leading_values)?;
                (
                    outputs,
                    seed_geometry_types.len(),
                    CustomFunctionTransposeRules::Attached,
                    vec![driver.region(backward_region_index)?.to_program()],
                )
            }
            CustomFunctionRules::Retained { rules, batching, discharged } => {
                // Replay the forward rule specialization on the dual primals, returning the primal outputs followed by
                // the residuals.
                let input_types = inputs.iter().map(|input| input.primal().r#type().into_owned()).collect::<Vec<_>>();
                let key = self.specialization_key(rules, batching, *discharged, input_types, primal_output_types)?;
                let specialization = rules.forward_specialization(key.clone())?;
                let mut outputs = specialization.program.interpret_in_context(context.primal(), primal_inputs)?;
                leading_values.extend(outputs.split_off(output_count));

                // The seed geometry of a batched call is that of its unbatched output tangent types, read by identity
                // from its outputs.
                let geometry_types = match batching {
                    None => outputs.iter().map(|output| output.r#type().tangent()).collect::<Result<Vec<_>, _>>()?,
                    Some(batching) => {
                        batching.output_types.iter().map(DifferentiableType::tangent).collect::<Result<Vec<_>, _>>()?
                    }
                };
                let seed_geometry_types = capture_seed_geometry(&outputs, &geometry_types, &mut leading_values)?;
                let seed_geometry_count = seed_geometry_types.len();

                // A batched call stages a batched carrier. At each level, the carrier's inputs have the batch axes of
                // the corresponding call inputs, except for the residuals, which have the axes of that level's forward
                // specialization, and its outputs have the call's batch axes.
                let carrier_batching = match batching {
                    None => None,
                    Some(batching) => {
                        let non_differentiated_count = key.non_differentiated_count;
                        let unbatched_key = CustomRuleSpecializationKey { levels: Vec::new(), ..key.clone() };
                        let unbatched = rules.forward_specialization(unbatched_key)?;
                        let mut carrier_input_types = batching.input_types[..non_differentiated_count].to_vec();
                        carrier_input_types.extend_from_slice(&unbatched.program.output_types()[output_count..]);
                        carrier_input_types.extend_from_slice(&seed_geometry_types);
                        for r#type in &batching.input_types[non_differentiated_count..] {
                            carrier_input_types.push(r#type.tangent()?);
                        }
                        let mut levels = Vec::with_capacity(batching.levels.len());
                        for (index, level) in batching.levels.iter().enumerate() {
                            let level_key = CustomRuleSpecializationKey {
                                levels: batching.levels[..=index].to_vec(),
                                ..key.clone()
                            };
                            let level_specialization = rules.forward_specialization(level_key)?;
                            let leading_input_count =
                                non_differentiated_count + boundary_input_count(&batching.levels[..index]);
                            let mut input_axes = level.input_axes[..leading_input_count].to_vec();
                            input_axes.extend_from_slice(&level_specialization.output_axes[output_count..]);
                            input_axes.extend(std::iter::repeat_n(BatchAxis::replicated(), seed_geometry_count));
                            input_axes.extend_from_slice(&level.input_axes[leading_input_count..]);
                            levels.push(CustomRuleBatchingLevel { input_axes, ..level.clone() });
                        }
                        Some(CustomRuleBatching {
                            input_types: carrier_input_types,
                            output_types: batching
                                .output_types
                                .iter()
                                .map(DifferentiableType::tangent)
                                .collect::<Result<Vec<_>, _>>()?,
                            levels,
                        })
                    }
                };
                (
                    outputs,
                    seed_geometry_count,
                    CustomFunctionTransposeRules::Retained {
                        rules: rules.clone(),
                        batching: carrier_batching,
                        invokes_rule_directly: false,
                        discharged: *discharged,
                    },
                    Vec::new(),
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
        let carrier = CustomFunctionTransposeOperation {
            leading_input_count,
            seed_geometry_count,
            input_tangent_types,
            output_tangent_types: outputs
                .iter()
                .map(|output| output.r#type().tangent())
                .collect::<Result<Vec<_>, _>>()?,
            rules: carrier_rules,
            marker: PhantomData,
        };
        let output_tangents = context.tangent().bind(carrier, carrier_regions, &carrier_inputs)?;
        check_count!("output", output_tangents, output_count, ProgramError);
        outputs
            .into_iter()
            .zip(output_tangents)
            .map(|(primal, tangent)| DifferentiationDual::new(primal, tangent))
            .collect::<Result<Vec<_>, _>>()
    }
}

// The call itself is intentionally non-transposable, which does not restrict reverse-mode differentiation. Reverse mode
// linearizes first: a forward-mode rule replaces `f(p, x)` with the ordinary primitive program computing the linear
// map `ẋ ↦ (∂f/∂x)(p, x) · ẋ`, and reverse-mode rules replace it with a `CustomFunctionTransposeOperation` carrier,
// whose transpose evaluates `backward(p, r, ȳ) = x̄`. Therefore, only an invalid direct transpose of an un-linearized
// call can reach this rejection path.
impl_non_transposable_operation!(
    <V, O, S> CustomFunctionOperation<V, O, S>
    where V: Typed<Type: DifferentiableType + Eq + Hash> + Parameter, S: CustomRuleSource<V, O>
);

/// Canonical operation name for [`CustomFunctionTransposeOperation`].
pub const CUSTOM_FUNCTION_TRANSPOSE_OPERATION_NAME: &str = "custom_function_transpose";

/// Effects of every [`CustomFunctionTransposeOperation`] with retained rules, which carries an unresolved backward
/// rule.
static CUSTOM_FUNCTION_TRANSPOSE_EFFECTS: LazyLock<Effects> =
    LazyLock::new(|| Effects::empty().clone().with_deferred_work());

/// Backward rule of a [`CustomFunctionTransposeOperation`], in the representation of the call that staged it.
#[derive(Clone, PartialEq, Eq, Hash)]
enum CustomFunctionTransposeRules<T, S> {
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
    },
}

/// Selected reverse-mode carrier of a [`CustomFunctionOperation`] with reverse-mode rules, which its
/// [`jvp_for_transpose`](DifferentiableOperation::jvp_for_transpose) rule stages in place of the call. It represents
/// the linear map `ẋ ↦ ẏ` from the differentiated inputs' tangents to the output tangents, whose transpose `ȳ ↦ x̄` is
/// the call's backward rule. This is the role of the `custom_lin` primitive of [JAX's custom
/// derivatives](https://github.com/jax-ml/jax/blob/main/jax/_src/custom_derivatives.py), which has no rendered
/// documentation page and is therefore linked at its source. The map has no forward program: reverse-mode rules supply
/// only its transpose, and deriving the map itself would differentiate the primal, which the custom rules exist to
/// avoid. The carrier therefore rejects execution, lowering, and forward mode, and it exists only to be transposed. A
/// call without reverse-mode rules stages no carrier, because reverse mode transposes the linearization of its
/// forward-mode rule instead.
///
/// Its inputs are the known leading inputs (i.e., any batching boundary inputs, the non-differentiated inputs, the
/// residuals, and the seed geometry) followed by the differentiated inputs' tangents. The seed geometry names the
/// runtime quantities of the output tangent types (e.g., the extents of dynamically shaped outputs) that
/// structural-zero seeds need when no other leading input names them, and the backward rule never receives it.
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
///     `(leading, ȳ) → x̄` over the leading inputs other than the seed geometry, which transposition replays inline
///     into the pullback, since a user-supplied backward program has no linearity contract of its own that would let it
///     be transposed again. The region's own effects decide whether transposition must run it when no cotangent is
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
/// axes, and each returned cotangent is aligned to its input, summing the per-item cotangents of a replicated input.
pub struct CustomFunctionTransposeOperation<V: Typed + Parameter, O, S = CustomRuleReference<V, O>> {
    /// Number of leading known inputs (i.e., boundary inputs, non-differentiated inputs, residuals, and seed geometry).
    leading_input_count: usize,

    /// Number of trailing leading inputs that carry the runtime geometry of the output tangent types (e.g., the
    /// extents of dynamically shaped outputs), from which structural-zero seeds are materialized when no other leading
    /// input carries it. The backward rule never receives them.
    seed_geometry_count: usize,

    /// Tangent types of the differentiated inputs.
    input_tangent_types: Vec<V::Type>,

    /// Tangent types of the outputs.
    output_tangent_types: Vec<V::Type>,

    /// Backward rule of the carrier.
    rules: CustomFunctionTransposeRules<V::Type, S>,

    /// Marker for the operation family, which only the rule source and the rule region name.
    marker: PhantomData<fn() -> O>,
}

impl<V: Typed<Type: DifferentiableType + Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>>
    CustomFunctionTransposeOperation<V, O, S>
{
    /// Creates a new unbatched [`CustomFunctionTransposeOperation`] that applies the retained backward rule of the
    /// provided rules, exactly as the [`jvp_for_transpose`](DifferentiableOperation::jvp_for_transpose) rule of a
    /// [`CustomFunctionOperation`] with retained rules stages it.
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
            seed_geometry_count: 0,
            input_tangent_types,
            output_tangent_types,
            rules: CustomFunctionTransposeRules::Retained {
                rules,
                batching: None,
                invokes_rule_directly: false,
                discharged: false,
            },
            marker: PhantomData,
        }
    }

    /// Creates a new [`CustomFunctionTransposeOperation`] whose backward rule is its attached `"backward"` rule
    /// region, exactly as the [`jvp_for_transpose`](DifferentiableOperation::jvp_for_transpose) rule of a
    /// [`CustomFunctionOperation`] with attached rules stages it, except that it has no seed geometry. The region
    /// maps the leading inputs followed by one cotangent per output to one cotangent per differentiated input. When
    /// the zero seeds of dead outputs need runtime quantities (e.g., dynamic extents) that no leading input names,
    /// the region can receive them as additional leading inputs.
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
            seed_geometry_count: 0,
            input_tangent_types,
            output_tangent_types,
            rules: CustomFunctionTransposeRules::Attached,
            marker: PhantomData,
        }
    }

    /// Creates the source carrier from which the backward specializations of `rules` are derived, whose
    /// transposition invokes the retained backward rule directly.
    pub(super) fn direct(
        rules: S,
        leading_input_count: usize,
        seed_geometry_count: usize,
        input_tangent_types: Vec<V::Type>,
        output_tangent_types: Vec<V::Type>,
        discharged: bool,
    ) -> Self {
        Self {
            leading_input_count,
            seed_geometry_count,
            input_tangent_types,
            output_tangent_types,
            rules: CustomFunctionTransposeRules::Retained {
                rules,
                batching: None,
                invokes_rule_directly: true,
                discharged,
            },
            marker: PhantomData,
        }
    }

    /// Returns the number of leading known inputs of this carrier.
    #[inline]
    pub fn leading_input_count(&self) -> usize {
        self.leading_input_count
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
            CustomFunctionTransposeRules::Attached => None,
            CustomFunctionTransposeRules::Retained { rules, .. } => Some(rules),
        }
    }

    /// Converts this carrier into the family `(V2, O2)` with the rule source that `rules_fn` derives from its current
    /// source (refer to [`CustomFunctionOperation::into_family`]).
    pub fn into_family<V2, O2, S2, F>(self, rules_fn: F) -> CustomFunctionTransposeOperation<V2, O2, S2>
    where
        V2: Typed<Type: DifferentiableType + Eq + Hash + From<V::Type>> + Parameter,
        F: FnOnce(S) -> S2,
    {
        let map_types = |types: Vec<V::Type>| types.into_iter().map(V2::Type::from).collect::<Vec<_>>();
        CustomFunctionTransposeOperation {
            leading_input_count: self.leading_input_count,
            seed_geometry_count: self.seed_geometry_count,
            input_tangent_types: map_types(self.input_tangent_types),
            output_tangent_types: map_types(self.output_tangent_types),
            rules: match self.rules {
                CustomFunctionTransposeRules::Attached => CustomFunctionTransposeRules::Attached,
                CustomFunctionTransposeRules::Retained { rules, batching, invokes_rule_directly, discharged } => {
                    CustomFunctionTransposeRules::Retained {
                        rules: rules_fn(rules),
                        batching: batching.map(CustomRuleBatching::map_types),
                        invokes_rule_directly,
                        discharged,
                    }
                }
            },
            marker: PhantomData,
        }
    }

    /// Converts this carrier into the family `(V2, O2)` with any rule source type when its backward rule is attached,
    /// or returns it unchanged when its backward rule is retained (refer to
    /// [`CustomFunctionOperation::into_attached_family`]).
    pub fn into_attached_family<V2, O2, S2>(self) -> Result<CustomFunctionTransposeOperation<V2, O2, S2>, Self>
    where
        V2: Typed<Type: From<V::Type>> + Parameter,
    {
        match self.rules {
            CustomFunctionTransposeRules::Attached => {
                let map_types = |types: Vec<V::Type>| types.into_iter().map(V2::Type::from).collect::<Vec<_>>();
                Ok(CustomFunctionTransposeOperation {
                    leading_input_count: self.leading_input_count,
                    seed_geometry_count: self.seed_geometry_count,
                    input_tangent_types: map_types(self.input_tangent_types),
                    output_tangent_types: map_types(self.output_tangent_types),
                    rules: CustomFunctionTransposeRules::Attached,
                    marker: PhantomData,
                })
            }
            CustomFunctionTransposeRules::Retained { .. } => Err(self),
        }
    }

    /// Returns a description of this carrier for diagnostics, which names the definition of a retained backward rule.
    fn description(&self) -> String {
        match &self.rules {
            CustomFunctionTransposeRules::Attached => format!("`{CUSTOM_FUNCTION_TRANSPOSE_OPERATION_NAME}`"),
            CustomFunctionTransposeRules::Retained { rules, .. } => {
                format!("`{CUSTOM_FUNCTION_TRANSPOSE_OPERATION_NAME}` `{}`", rules.name())
            }
        }
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> Clone
    for CustomFunctionTransposeOperation<V, O, S>
{
    fn clone(&self) -> Self {
        Self {
            leading_input_count: self.leading_input_count,
            seed_geometry_count: self.seed_geometry_count,
            input_tangent_types: self.input_tangent_types.clone(),
            output_tangent_types: self.output_tangent_types.clone(),
            rules: self.rules.clone(),
            marker: PhantomData,
        }
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> Debug
    for CustomFunctionTransposeOperation<V, O, S>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // A retained backward rule is identified by the name of its definition, whose own rendering would include its
        // rules.
        let mut debug = formatter.debug_struct("CustomFunctionTransposeOperation");
        if let CustomFunctionTransposeRules::Retained { rules, .. } = &self.rules {
            debug.field("name", &rules.name());
        }
        debug
            .field("leading_input_count", &self.leading_input_count)
            .field("seed_geometry_count", &self.seed_geometry_count)
            .field("input_tangent_types", &self.input_tangent_types)
            .field("output_tangent_types", &self.output_tangent_types);
        if let CustomFunctionTransposeRules::Retained { batching, invokes_rule_directly, discharged, .. } = &self.rules
        {
            debug
                .field("batching", batching)
                .field("invokes_rule_directly", invokes_rule_directly)
                .field("discharged", discharged);
        }
        debug.finish()
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> PartialEq
    for CustomFunctionTransposeOperation<V, O, S>
{
    fn eq(&self, other: &Self) -> bool {
        self.leading_input_count == other.leading_input_count
            && self.seed_geometry_count == other.seed_geometry_count
            && self.input_tangent_types == other.input_tangent_types
            && self.output_tangent_types == other.output_tangent_types
            && self.rules == other.rules
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> Eq
    for CustomFunctionTransposeOperation<V, O, S>
{
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> Hash
    for CustomFunctionTransposeOperation<V, O, S>
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.leading_input_count.hash(state);
        self.seed_geometry_count.hash(state);
        self.input_tangent_types.hash(state);
        self.output_tangent_types.hash(state);
        self.rules.hash(state);
    }
}

impl<V: Typed<Type: DifferentiableType + Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> Display
    for CustomFunctionTransposeOperation<V, O, S>
{
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<V: Typed<Type: DifferentiableType + Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>> Operation
    for CustomFunctionTransposeOperation<V, O, S>
{
    type Type = V::Type;

    #[inline]
    fn name(&self) -> &'static str {
        CUSTOM_FUNCTION_TRANSPOSE_OPERATION_NAME
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        match self.rules {
            CustomFunctionTransposeRules::Attached => const { &[RegionSlot::deferred_rule("backward")] },
            CustomFunctionTransposeRules::Retained { .. } => &[],
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
            CustomFunctionTransposeRules::Attached => Ok(vec![Some(self.backward_input_types(input_types)?)]),
            CustomFunctionTransposeRules::Retained { .. } => Ok(Vec::new()),
        }
    }

    fn infer_output_types(
        &self,
        input_types: &[V::Type],
        region_interfaces: &[RegionInterface<V::Type>],
    ) -> Result<Vec<V::Type>, TypeError> {
        check_count!("region", region_interfaces, self.region_slots().len(), TypeError);
        check_count!("input", input_types, self.leading_input_count + self.input_tangent_types.len(), TypeError);
        check_types!(@same, format!("`{CUSTOM_FUNCTION_TRANSPOSE_OPERATION_NAME}` tangent input"), [
            self.input_tangent_types.as_slice(),
            &input_types[self.leading_input_count..],
        ]);
        if let CustomFunctionTransposeRules::Attached = self.rules {
            let backward = &region_interfaces[0];
            check_types!(@same, format!("`{CUSTOM_FUNCTION_TRANSPOSE_OPERATION_NAME}` backward rule input"), [
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
                && (<V::Type as Type>::Refinements::establish(cotangent_types.iter(), backward.output_types().iter())
                    .is_err()
                    || <V::Type as Type>::Refinements::establish(
                        backward.output_types().iter(),
                        cotangent_types.iter(),
                    )
                    .is_err())
            {
                check_types!(@same, format!("`{CUSTOM_FUNCTION_TRANSPOSE_OPERATION_NAME}` backward rule output"), [
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
            CustomFunctionTransposeRules::Attached => Cow::Borrowed(Effects::empty()),
            CustomFunctionTransposeRules::Retained { .. } => Cow::Borrowed(&CUSTOM_FUNCTION_TRANSPOSE_EFFECTS),
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
            CustomFunctionTransposeRules::Attached => CustomFunctionTransposeRules::Attached,
            CustomFunctionTransposeRules::Retained { rules, batching, invokes_rule_directly, discharged } => {
                CustomFunctionTransposeRules::Retained {
                    rules: rules.clone(),
                    batching: batching.as_ref().map(|batching| batching.rename_identities(renaming)).transpose()?,
                    invokes_rule_directly: *invokes_rule_directly,
                    discharged: *discharged,
                }
            }
        };
        Ok(Self {
            leading_input_count: self.leading_input_count,
            seed_geometry_count: self.seed_geometry_count,
            input_tangent_types: rename(&self.input_tangent_types)?,
            output_tangent_types: rename(&self.output_tangent_types)?,
            rules,
            marker: PhantomData,
        })
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        // The tangent types coincide with the types of the carrier's tangent inputs and outputs, so they are not
        // rendered.
        let operation = OperationFormatter::new(formatter, indentation, CUSTOM_FUNCTION_TRANSPOSE_OPERATION_NAME)?;
        operation.bracketed(|operation| {
            if let CustomFunctionTransposeRules::Retained { rules, .. } = &self.rules {
                operation.field("name", format_args!("{:?}", rules.name()))?;
            }
            operation.field("leading_input_count", self.leading_input_count)?;
            if self.seed_geometry_count != 0 {
                operation.field("seed_geometry_count", self.seed_geometry_count)?;
            }
            match &self.rules {
                CustomFunctionTransposeRules::Attached => Ok(()),
                CustomFunctionTransposeRules::Retained { batching, invokes_rule_directly, discharged, .. } => {
                    if let Some(batching) = batching {
                        operation.field("batching", batching)?;
                    }
                    if *invokes_rule_directly {
                        operation.field("direct", true)?;
                    }
                    if *discharged {
                        operation.field("discharged", true)?;
                    }
                    Ok(())
                }
            }
        })
    }
}

impl<V: Typed<Type: DifferentiableType + Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>>
    CustomFunctionTransposeOperation<V, O, S>
{
    /// Returns the input types of an attached backward region: the types of the leading inputs among `input_types`,
    /// except for the seed geometry, followed by the cotangent types of the outputs.
    fn backward_input_types(&self, input_types: &[V::Type]) -> Result<Vec<V::Type>, TypeError> {
        if self.leading_input_count > input_types.len() {
            return Err(TypeError::invalid(format!(
                "`{}` leading input count {} exceeds input count {}",
                CUSTOM_FUNCTION_TRANSPOSE_OPERATION_NAME,
                self.leading_input_count,
                input_types.len(),
            )));
        }
        let Some(rule_leading_input_count) = self.leading_input_count.checked_sub(self.seed_geometry_count) else {
            return Err(TypeError::invalid(format!(
                "`{}` seed geometry count {} exceeds leading input count {}",
                CUSTOM_FUNCTION_TRANSPOSE_OPERATION_NAME, self.seed_geometry_count, self.leading_input_count,
            )));
        };
        let mut backward_input_types = input_types[..rule_leading_input_count].to_vec();
        backward_input_types.extend(
            self.output_tangent_types.iter().map(DifferentiableType::cotangent).collect::<Result<Vec<_>, _>>()?,
        );
        Ok(backward_input_types)
    }
}

impl<C, P, V, O, S> ReferenceDischargeableOperation<C, P> for CustomFunctionTransposeOperation<V, O, S>
where
    C: Context<Operation: From<CustomFunctionTransposeOperation<V, O, S>>>,
    P: ReferenceDischargePolicy<C>,
    V: Typed<Type: DifferentiableType + Eq + Hash> + Parameter,
    S: CustomRuleSource<V, O>,
    CustomFunctionTransposeOperation<V, O, S>: Operation<Type = C::Type>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        // An attached backward region discharges its local reference state now. A carrier with a retained backward
        // rule holds no reference state of its own, and its rebound form records that its lazily traced backward rule
        // programs must be discharged once they are traced.
        match &self.rules {
            CustomFunctionTransposeRules::Attached => {
                discharge_local_reference_operation(self, context, driver, inputs)
            }
            CustomFunctionTransposeRules::Retained { rules, batching, invokes_rule_directly, .. } => {
                let discharged = Self {
                    rules: CustomFunctionTransposeRules::Retained {
                        rules: rules.clone(),
                        batching: batching.clone(),
                        invokes_rule_directly: *invokes_rule_directly,
                        discharged: true,
                    },
                    ..self.clone()
                };
                discharge_reference_free_operation(&discharged, context, driver, inputs)
            }
        }
    }
}

impl<C: Domain, V: Typed<Type: DifferentiableType + Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>>
    InterpretableOperation<C> for CustomFunctionTransposeOperation<V, O, S>
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
    PartiallyEvaluatableOperation<C> for CustomFunctionTransposeOperation<C::Constant, C::Operation, S>
where
    C::Operation: From<CustomFunctionTransposeOperation<C::Constant, C::Operation, S>>,
{
    // A carrier always has an unknown tangent input when it is staged, and a retained backward rule's deferred work is
    // always residualized by the shared folding boundary, so the default rule suffices.
}

impl<C, P, S> BatchableOperation<C, P> for CustomFunctionTransposeOperation<C::Constant, C::Operation, S>
where
    C: Context<
            Type: DifferentiableType + Eq + Hash,
            Operation: From<CustomFunctionTransposeOperation<C::Constant, C::Operation, S>>,
        >,
    P: CotangentBatchingPolicy<C>,
    S: CustomRuleSource<C::Constant, C::Operation>,
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        check_count!("input", inputs, self.leading_input_count + self.input_tangent_types.len(), ProgramError);
        let input_axes = inputs.iter().map(P::batch_axis).collect::<Vec<_>>();
        let (rules, batching, discharged) = match &self.rules {
            CustomFunctionTransposeRules::Attached => {
                // A completely replicated carrier at an unnamed batching level needs no structural region rewrite, and
                // keeping it avoids manufacturing a batch axis that its backward region does not observe. What makes
                // this shortcut sound is the level being unnamed: a region's value can vary per batch item with no
                // mapped input only by addressing the level by name (e.g., an `axis_index` or a collective over the
                // level's axis). Any other carrier is rejected, because no forward program determines the batch axes of
                // its outputs.
                if input_axes.iter().all(BatchAxis::is_replicated) && context.axis_name().is_none() {
                    let outputs = context.parent().bind(
                        self.clone(),
                        driver.regions().map(|region| region.to_program()).collect::<Vec<_>>(),
                        inputs.iter().map(P::value).cloned().collect::<Vec<_>>().as_slice(),
                    )?;
                    return Ok(outputs.into_iter().map(P::replicated).collect::<Vec<_>>().into());
                }
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "a `{CUSTOM_FUNCTION_TRANSPOSE_OPERATION_NAME}` with an attached backward rule cannot be \
                         batched structurally, because it has no forward program to determine the batch axes of its \
                         outputs; it is preserved unchanged only when every input is replicated at an unnamed \
                         batching level",
                    ),
                });
            }
            CustomFunctionTransposeRules::Retained { invokes_rule_directly: true, rules, .. } => {
                return Err(ProgramError::MalformedProgram(format!(
                    "`{CUSTOM_FUNCTION_TRANSPOSE_OPERATION_NAME}` `{}` source carriers are never batched",
                    rules.name(),
                ))
                .into());
            }
            CustomFunctionTransposeRules::Retained { rules, batching, discharged, .. } => {
                (rules, batching, *discharged)
            }
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
            seed_geometry_count: self.seed_geometry_count,
            input_tangent_types: inputs[self.leading_input_count..]
                .iter()
                .map(|input| P::value(input).r#type().into_owned())
                .collect(),
            output_tangent_types: batched_identity.output_types(),
            rules: CustomFunctionTransposeRules::Retained {
                rules: rules.clone(),
                batching: Some(batching),
                invokes_rule_directly: false,
                discharged,
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
            .collect::<Result<Vec<_>, _>>()?
            .into())
    }
}

impl<C: Context<Type: DifferentiableType + Eq + Hash>, S: CustomRuleSource<C::Constant, C::Operation>>
    DifferentiableOperation<C> for CustomFunctionTransposeOperation<C::Constant, C::Operation, S>
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

impl<V, O, S> TransposableOperation<V, O> for CustomFunctionTransposeOperation<V, O, S>
where
    V: Value<Type: DifferentiableType + Eq + Hash>,
    O: Operation<Type = V::Type>
        + From<CustomFunctionTransposeOperation<V, O>>
        + ResidualZeroProvider<V::Type, Operation = O>
        + From<AddOperation<V::Type>>,
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
                    "`{CUSTOM_FUNCTION_TRANSPOSE_OPERATION_NAME}` leading input {index} is not known during \
                     transposition",
                ))
            })?);
        }
        let seed_geometry_count = self.seed_geometry_count;
        let (rules, batching, invokes_rule_directly, discharged) = match &self.rules {
            CustomFunctionTransposeRules::Attached => {
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
                // so zero cotangents stay symbolic instead of accumulating. Outputs with no producing instruction
                // (forwarded region inputs and constants) conservatively classify as nonzero.
                let output_is_zero = backward
                    .output_ids()
                    .iter()
                    .map(|output| {
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
                // inputs, whose seed geometry names every runtime quantity of the output tangent types, do, so the zero
                // is assembled from them one identity at a time before falling back to the nullary zero that every
                // identity-free type keeps. The backward region never receives the seed geometry.
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
                backward_inputs.truncate(self.leading_input_count - seed_geometry_count);
                backward_inputs.extend(seeds);

                // A user-supplied backward program has no linearity contract of its own, so it cannot be transposed
                // again and is replayed inline into the pullback.
                let cotangents = backward.interpret_in_context(&**context, backward_inputs)?;
                check_count!("output", cotangents, self.input_tangent_types.len(), ProgramError);

                // Leading inputs are known and contribute nothing. Preserve symbolic zeros reported by the backward
                // program when their types suffice to reconstruct them. A dynamic zero must retain the already
                // materialized value as its extent inputs live in this carrier's leading graph and may otherwise
                // disappear before the outer pullback constructs its disconnected-input zeros.
                return inputs[self.leading_input_count..]
                    .iter()
                    .zip(cotangents)
                    .zip(output_is_zero)
                    .zip(&accumulators[self.leading_input_count..])
                    .filter(|(((input, cotangent), is_zero), _)| {
                        input.is_unknown()
                            && (!is_zero || !O::zero_residual_types(cotangent.r#type().as_ref()).is_empty())
                    })
                    .try_for_each(|(((_, cotangent), _), accumulator)| {
                        accumulator.accumulate(context, MaybeZero::Value(cotangent))
                    });
            }
            CustomFunctionTransposeRules::Retained { rules, batching, invokes_rule_directly, discharged } => {
                (rules, batching, *invokes_rule_directly, *discharged)
            }
        };
        if invokes_rule_directly {
            let Some(backward) = rules.native_definition().and_then(|definition| definition.backward.as_ref()) else {
                return Err(ProgramError::MalformedProgram(format!(
                    "`{CUSTOM_FUNCTION_TRANSPOSE_OPERATION_NAME}` `{}` has no backward rule",
                    rules.name(),
                ))
                .into());
            };

            // The seed geometry is used only to materialize zero seeds, so the rule sees neither those inputs nor their
            // accumulators.
            let rule_leading_input_count = self.leading_input_count - seed_geometry_count;
            let rule_leading_values = &leading_values[..rule_leading_input_count];
            let cotangents = match backward {
                CustomVjpBackward::Accumulating(rule) => {
                    if seed_geometry_count == 0 {
                        return rule.apply(context, inputs, outputs, accumulators);
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
                    return rule.apply(context, &inputs, outputs, &accumulators);
                }

                // A pure rule receives the known leading inputs and materialized seeds, and its cotangents are added
                // to each differentiated input's destination, which also discards the cotangents of ignored inputs.
                // A zero seed whose type names runtime extents reads them from the leading inputs, including the seed
                // geometry.
                CustomVjpBackward::Pure(rule) => {
                    let seeds = outputs
                        .iter()
                        .map(|seed| {
                            O::materialize_zero_from_residual_sources(&**context, seed.clone(), &leading_values)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    rule.apply(rule_leading_values, &seeds)?
                }
                CustomVjpBackward::PureWithSymbolicZeros(rule) => rule.apply(rule_leading_values, outputs)?,
            };

            // Each cotangent has the cotangent type of its differentiated input, which may differ from the input's
            // tangent type (e.g., sharded arrays swap their reduced and unreduced axes).
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
                accumulator.accumulate(context, MaybeZero::Value(cotangent))?;
            }
            return Ok(());
        }

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
                    seed_geometry_count,
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
                        seed_geometry_count,
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
        // destinations.
        let mut leading_values = leading_values.into_iter();
        let mut arguments = leading_values.by_ref().take(boundary_input_count).collect::<Vec<_>>();
        arguments.extend(outputs.iter().filter_map(MaybeZero::as_value).cloned());
        arguments.extend(references);
        arguments.extend(leading_values);
        let mut returned = specialization.program.interpret_in_context(&**context, arguments)?.into_iter();
        for (accumulator, kind) in accumulators[self.leading_input_count..].iter().zip(destination_kinds) {
            if kind == CotangentDestinationKind::Return {
                let cotangent = returned.next().ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "`{CUSTOM_FUNCTION_TRANSPOSE_OPERATION_NAME}` specialization returned too few cotangents",
                    ))
                })?;
                accumulator.accumulate(context, MaybeZero::Value(cotangent))?;
            }
        }
        Ok(())
    }
}

/// Validates that the leading `non_differentiated_count` input positions of a custom function call named `name` fit
/// within its `input_count` inputs.
///
/// # Errors
///
/// Returns a [`TypeError`] when `non_differentiated_count` exceeds `input_count`.
pub(super) fn validate_non_differentiated_count(
    name: &str,
    non_differentiated_count: usize,
    input_count: usize,
) -> Result<(), TypeError> {
    if non_differentiated_count > input_count {
        return Err(TypeError::invalid(format!(
            "`{name}` non-differentiated input count {non_differentiated_count} exceeds input count {input_count}",
        )));
    }
    Ok(())
}

/// Validates the reference contract that the custom function operations share over one primal boundary. A
/// reference-typed input is accepted only in the leading `non_differentiated_count` positions, which correspond to
/// _plumbing_ that every attached rule region receives unchanged (the rule interfaces define no tangent or cotangent
/// slot for a reference, so an active reference input would have a derivative that no user-supplied rule can express).
/// No output may be a reference, because a rule region would then have to produce that output's tangent reference or
/// consume its cotangent reference, and a user-supplied rule can neither allocate nor receive one. The custom
/// derivative operations apply this contract during type inference, so it holds for every constructed call, and their
/// derivative rules apply it again to the replayed inputs as defense in depth.
///
/// # Parameters
///
///   - `name`: Operation name used in diagnostics.
///   - `non_differentiated_count`: Number of leading inputs that parameterize the call without being differentiated.
///   - `input_types`: Primal input types in input order.
///   - `output_types`: Primal output types in output order.
///
/// # Errors
///
/// Returns a [`TypeError`] naming the first reference-typed input in the differentiated segment, or otherwise the
/// first reference-typed output.
pub(super) fn validate_custom_function_reference_boundary<T: Type>(
    name: &str,
    non_differentiated_count: usize,
    input_types: &[T],
    output_types: &[T],
) -> Result<(), TypeError> {
    if let Some((index, r#type)) = input_types
        .iter()
        .enumerate()
        .skip(non_differentiated_count)
        .find(|(_, r#type)| r#type.is_reference())
    {
        return Err(TypeError::invalid(format!(
            "`{name}` accepts reference inputs only in its leading non-differentiated segment; move input {index} \
             of type `{type}` before the differentiated inputs",
        )));
    }
    if let Some((index, r#type)) = output_types.iter().enumerate().find(|(_, r#type)| r#type.is_reference()) {
        return Err(TypeError::invalid(format!(
            "`{name}` cannot return a reference, but output {index} has type `{type}`",
        )));
    }
    Ok(())
}

/// Validates the inputs before replaying a custom derivative rule. Rule regions bypass ordinary differentiation
/// dispatch, so replay checks the reference contract of [`validate_custom_function_reference_boundary`] that type
/// inference already enforces and uses [`ReferenceBoundary`] to reject aliased concrete or staged references.
/// Non-differentiated numeric inputs must have zero tangents, while non-differentiated references carry state whose
/// tangent the rule leaves untouched.
///
/// # Parameters
///
///   - `name`: Operation name used in diagnostics.
///   - `non_differentiated_count`: Number of leading inputs that parameterize the call without being differentiated.
///   - `context`: Context in which the rule is replayed, which resolves the inputs.
///   - `inputs`: Dual inputs of the replayed call, in input order.
///   - `output_types`: Primal output types in output order.
///
/// # Errors
///
/// Returns the [`ProgramError`] of the first violated contract: the [`TypeError`] of
/// [`validate_custom_function_reference_boundary`], a reference boundary error, or an unsupported non-zero tangent.
pub(super) fn validate_custom_function_replay<C: Context<Type: DifferentiableType>>(
    name: &str,
    non_differentiated_count: usize,
    context: &C,
    inputs: &[DifferentiationDual<C::Value>],
    output_types: &[C::Type],
) -> Result<(), ProgramError> {
    let primal_types = inputs.iter().map(|input| input.primal().r#type().into_owned()).collect::<Vec<_>>();
    validate_custom_function_reference_boundary(name, non_differentiated_count, primal_types.as_slice(), output_types)?;
    ReferenceBoundary::new_for_differentiation(context, inputs.iter().map(DifferentiationDual::primal), [], [])?;
    if let Some(input) = inputs.iter().take(non_differentiated_count).find(|input| {
        !input.primal().r#type().is_reference()
            && !input.tangent().is_zero()
            && !input.tangent().r#type().is_zero_space()
    }) {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "`{}` cannot propagate the non-zero tangent of type `{}` supplied for one of its \
                 {} leading non-differentiated inputs, because its rule has no tangent slot for them",
                name,
                input.tangent().r#type(),
                non_differentiated_count,
            ),
        });
    }
    Ok(())
}

/// Differentiates the primal region at slot 0 of a custom function call in forward mode, which is the forward-mode rule
/// of a call whose JVP is derived from its primal (i.e., [`CustomFunctionJvpRule::Primal`]). The derivative is taken
/// with respect to the differentiated inputs with active tangents, while the leading `non_differentiated_count` inputs,
/// which must have zero tangents, parameterize the derivative.
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
    call: Option<&CustomFunctionOperation<C::Constant, C::Operation, S>>,
) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>
where
    C: Context<
            Type: DifferentiableType + Eq + Hash,
            Operation: ResidualZeroProvider<C::Type, Operation = C::Operation>
                           + From<CustomFunctionOperation<C::Constant, C::Operation, S>>,
        > + Zero<C::Value>,
    P: DifferentiationPolicy<C>,
    D: DifferentiationDriver<C>,
    S: CustomRuleSpecializer<C::Constant, C::Operation>,
{
    let primal_region = driver.region(0)?;
    let output_types = primal_region.output_types();
    let output_count = output_types.len();
    validate_custom_function_replay(name, non_differentiated_count, context.primal(), inputs, output_types.as_slice())?;
    check_count!("input", inputs, primal_region.input_types().len(), ProgramError);

    // Differentiate with respect to the differentiated inputs whose tangents are active. Inactive inputs (e.g.,
    // zero-space inputs) receive no tangent input, and outputs that depend on no active input have no tangent.
    let input_indices = inputs
        .iter()
        .enumerate()
        .skip(non_differentiated_count)
        .filter_map(|(index, input)| input.is_tangent_active().then_some(index))
        .collect::<Vec<_>>();
    let output_activity = primal_region.tangent_output_mask(&input_indices)?;
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
        Some(CustomFunctionOperation { rules: CustomFunctionRules::Retained { rules, discharged, .. }, .. }) => {
            let rules = rules.derived(CustomRuleDerivationKey {
                kind,
                input_count: inputs.len(),
                active_input_indices: input_indices.clone(),
                output_tangent_mask: output_activity.clone(),
            })?;
            Ok::<_, DifferentiationError>(Some(CustomFunctionOperation {
                non_differentiated_count,
                rules: CustomFunctionRules::Retained { rules, batching: None, discharged: *discharged },
                marker: PhantomData,
            }))
        }
        _ => Ok(None),
    };

    // A fused differentiation context stages primal and tangent work in the same context, so the fused JVP program
    // replays there directly. Otherwise, the linearized primal replays in the primal context and its tangent map
    // replays in the tangent context over the residuals.
    let (outputs, output_tangents) = if std::ptr::eq(context.primal(), context.tangent()) {
        let program = driver.jvp_program(primal_region, &input_indices)?;
        let mut values = primals;
        values.extend(tangents);
        let mut outputs = match derived_call(CustomRuleDerivationKind::Jvp)? {
            Some(operation) => context.primal().bind(operation, vec![(*program).clone()], values.as_slice())?,
            None => program.interpret_in_context(context.primal(), values)?,
        };
        let output_tangents = outputs.split_off(output_count);
        (outputs, output_tangents)
    } else if let Some(call) = call.filter(|_| primal_region.effects() == EffectsSummary::PURE) {
        // The known side computes the outputs with the call itself, and the tangent side stages a pushforward call
        // that recomputes the primal internally, because the call is opaque to partial evaluation. Recomputing is only
        // valid for a pure primal: the effects of any other primal (e.g., reference updates, or reference reads that
        // later effects would change) must run once, when the known side runs, so it is linearized inline instead.
        let program = driver.jvp_program(primal_region, &input_indices)?;
        let program_inputs = program.input_ids().to_vec();
        let (pushforward, _) =
            program.filtered(&program_inputs, &program.output_ids()[output_count..], &program_inputs)?;
        let outputs = context.primal().bind(call.clone(), vec![primal_region.to_program()], primals.as_slice())?;
        let mut tangent_inputs =
            primals.into_iter().map(|primal| context.primal_to_tangent(primal)).collect::<Result<Vec<_>, _>>()?;
        tangent_inputs.extend(tangents);
        let operation = derived_call(CustomRuleDerivationKind::Pushforward)?.unwrap();
        (outputs, context.tangent().bind(operation, vec![pushforward], tangent_inputs.as_slice())?)
    } else {
        let linearization = driver.linearize_program(primal_region, &input_indices)?;
        let mut outputs = linearization.primal().interpret_in_context(context.primal(), primals)?;
        let residuals = outputs.split_off(output_count);
        let mut tangent_inputs = tangents;
        for residual in residuals {
            tangent_inputs.push(context.primal_to_tangent(residual)?);
        }
        (outputs, linearization.tangent().interpret_in_context(context.tangent(), tangent_inputs)?)
    };
    check_count!("output", outputs, output_count, ProgramError);
    check_count!("output", output_tangents, output_activity.iter().filter(|&&active| active).count(), ProgramError);
    let mut output_tangents = output_tangents.into_iter();
    outputs
        .into_iter()
        .zip(output_activity)
        .map(|(primal, active)| {
            if active {
                DifferentiationDual::new(primal, MaybeZero::Value(output_tangents.next().unwrap()))
            } else {
                DifferentiationDual::new_with_zero_tangent(primal)
            }
        })
        .collect::<Result<Vec<_>, _>>()
}

/// Returns the error that forward-mode differentiation of a custom function call without a forward-mode rule
/// reports, where `call` describes the call (e.g., ``a `custom_function` call``). A call with reverse-mode rules
/// supports only reverse-mode differentiation, while a call without any rule is not differentiable. A missing rule
/// never falls back to differentiating the primal.
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

/// Replays the flat custom JVP rule `jvp_region`, which computes `(p, x, ẋ) ↦ (y, ẏ)`, for an operation whose leading
/// `non_differentiated_count` inputs `p` are not differentiated. This is the shared forward-mode rule of
/// [`CustomFunctionOperation`] and of other custom-rule operations that obtain their JVP program differently (e.g.,
/// by tracing a retained callback lazily). A fused [`DifferentiationContext`] replays the rule directly, while a
/// partitioned one replays its known and tangent parts in their own contexts. Structural-zero output tangents are
/// recovered from literal zeros and zero-producing instructions of the rule.
///
/// # Parameters
///
///   - `operation_name`: Name of the operation applying the rule, used in diagnostics.
///   - `non_differentiated_count`: Number of leading inputs that parameterize the rule without being differentiated.
///   - `context`: [`DifferentiationContext`] in which the rule is applied.
///   - `driver`: Instruction-scoped [`DifferentiationDriver`] used to partition the rule.
///   - `jvp_region`: Rule program over every primal input followed by the differentiated inputs' tangents, returning
///     the primal outputs followed by their tangents.
///   - `inputs`: Input [`DifferentiationDual`]s aligned with the operation's inputs.
///   - `tangent_activity`: Whether `jvp_region` takes the tangent of each differentiated input, or [`None`] when it
///     takes every one of them. A region that takes only the active tangents materializes the structural zeros
///     itself.
pub(super) fn replay_custom_jvp_rule<
    C: Context<Type: DifferentiableType, Operation: ResidualZeroProvider<C::Type, Operation = C::Operation>>
        + Zero<C::Value>,
    P: DifferentiationPolicy<C>,
    D: DifferentiationDriver<C>,
>(
    operation_name: &str,
    non_differentiated_count: usize,
    context: &DifferentiationContext<C, P>,
    driver: &D,
    jvp_region: RegionRef<'_, C::Constant, C::Operation>,
    inputs: &[DifferentiationDual<C::Value>],
    tangent_activity: Option<&[bool]>,
) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
    // Apply the user-supplied pushforward directly. For `f(p, x) = y`, the rule implements:
    //
    //   j(p, x, ẋ) = (f(p, x), (∂f/∂x)(p, x) · ẋ) = (y, ẏ).
    //
    // Feed every primal value, followed only by the differentiated inputs' tangents; `p` has no tangent slot in the
    // rule, so a non-zero tangent for a numeric `p` is rejected below. Replay stages the rule's ordinary primitive
    // operations directly in the active context, so it introduces no symbolic capture. Consequently, reverse mode
    // differentiation can transpose the resulting linear map in `ẋ` exactly like any other tangent program, and
    // no nested differentiation request or special reverse rule is needed here.
    let output_types = jvp_region.output_types();
    let output_count = output_types.len() / 2;
    validate_non_differentiated_count(operation_name, non_differentiated_count, inputs.len())?;
    let differentiated_inputs = &inputs[non_differentiated_count..];

    // The rule region is replayed directly rather than differentiated, so the replayed inputs are validated as
    // defense in depth (refer to the documentation of `validate_custom_function_replay`).
    validate_custom_function_replay(
        operation_name,
        non_differentiated_count,
        context.primal(),
        inputs,
        &output_types[..output_count],
    )?;
    let tangent_activity = match tangent_activity {
        Some(tangent_activity) => {
            if tangent_activity.len() != differentiated_inputs.len() {
                return Err(ProgramError::MalformedProgram(format!(
                    "`{operation_name}` has {} differentiated inputs but {} tangent activity flags",
                    differentiated_inputs.len(),
                    tangent_activity.len(),
                ))
                .into());
            }
            tangent_activity.to_vec()
        }
        None => vec![true; differentiated_inputs.len()],
    };
    let active_count = tangent_activity.iter().filter(|active| **active).count();
    check_count!("input", jvp_region.input_types(), inputs.len() + active_count, ProgramError);

    // The JVP region consumes `(primals..., active_differentiated_input_tangents...)`, so feed every dual primal
    // followed by the active differentiated duals' tangents.
    let mut jvp_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();

    // The JVP region takes every active tangent as a real region input, so materialize structural zeros against their
    // own primal, which names every runtime quantity a reference-bearing tangent type omits; static inputs keep the
    // nullary zero.
    for (input, _) in differentiated_inputs.iter().zip(&tangent_activity).filter(|(_, active)| **active) {
        let source = context.primal_to_tangent(input.primal().clone())?;
        jvp_inputs.push(C::Operation::materialize_zero_from_residual_sources(
            context.tangent(),
            input.tangent().clone(),
            std::iter::once(&source),
        )?);
    }

    // A fused differentiation context stages primal and tangent work in the same context, so the rule replays
    // there directly. Otherwise, partition the rule into its known part, which depends only on the primal inputs
    // and computes the primal outputs, and its tangent part, and replay each part in its own context.
    let shares_context = std::ptr::eq(context.primal(), context.tangent());
    let mut outputs = if shares_context {
        jvp_region.interpret_in_context(context.primal(), jvp_inputs)?
    } else {
        let mut known = vec![true; inputs.len()];
        known.resize(jvp_inputs.len(), false);
        let partition = driver.partition_jvp_program(jvp_region, &known, &(0..output_count).collect::<Vec<_>>())?;
        partition.interpret_in_context(context, &jvp_inputs, output_count)?
    };
    check_count!("output", outputs, 2 * output_count, ProgramError);
    let tangents = outputs.split_off(output_count);
    outputs
        .into_iter()
        .zip(tangents)
        .enumerate()
        .map(|(index, (primal, tangent))| {
            // Replaying a stored rule loses the structural-zero representation. Recover it from a literal zero
            // or a zero-producing instruction, without treating every known tangent as zero: a non-zero constant
            // tangent is affine and must still be rejected by linearization. Replay has already preserved effects.
            let output = jvp_region.output_ids()[output_count + index];
            let is_zero = jvp_region.atoms()[output.index()].as_constant().is_some_and(Value::is_zero)
                || jvp_region.instructions().iter().any(|instruction| {
                    instruction
                        .outputs()
                        .iter()
                        .position(|candidate| *candidate == output)
                        .is_some_and(|index| instruction.operation().is_zero(index))
                })
                // Only partitioning needs to recover zeros produced by constant folding. Inspecting arbitrary
                // eager tangent results would otherwise add a full array scan to each ordinary JVP call.
                || (!shares_context
                    && context.tangent().resolve(&tangent).into_constant().is_some_and(|value| value.is_zero()));

            // Keep a materialized dynamic zero when its reconstruction requires runtime dimension values.
            if is_zero && C::Operation::zero_residual_types(tangent.r#type().as_ref()).is_empty() {
                DifferentiationDual::new_with_zero_tangent(primal)
            } else {
                DifferentiationDual::new(primal, tangent)
            }
        })
        .collect::<Result<Vec<_>, _>>()
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayBatch, ArrayBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation,
        ArrayReference, ArrayReferenceTransform, ArrayReferenceTransformIndex, ArraySliceAxis, ArrayType, DataType,
        Dimension, DimensionBounds, DimensionType, DimensionValue, DimensionVariable, LogicalMesh, MeshAxis,
        MeshAxisType, Shape, Sharding, ShardingDimension,
    };
    use crate::batching::{
        BatchAxis, BatchingContext, ProgramBatchingOutputAxesPolicy, RecursiveBatchingDriver, batch,
    };
    use crate::contexts::{Context, EagerContext, StagingContext};
    use crate::differentiation::forward::JvpPartitionTransform;
    use crate::differentiation::{
        CotangentDestination, CotangentDestinationKind, CotangentSeed, DifferentiationRule, ForwardModeDifferentiate,
        ReverseModeDifferentiate, TranspositionDriver, differentiate_at,
    };
    use crate::interpretation::{InterpretableOperation, InterpretationDriver};
    use crate::operations::arithmetic::{AddOperation, MulOperation};
    use crate::operations::assertions::{AssertOperation, AssertionError};
    use crate::operations::collectives::axis_index::AxisIndexOperation;
    use crate::operations::comparisons::{CompareOperation, ComparisonDirection};
    use crate::operations::constants::zero::ZeroOperation;
    use crate::operations::constants::zero_like::ZeroLikeOperation;
    use crate::operations::control_flow::condition::{ConditionOperation, transpose_primal_condition};
    use crate::operations::control_flow::scan::ScanOperation;
    use crate::operations::custom_functions::functions::custom_function;
    use crate::operations::custom_functions::rules::{CustomRuleDefinition, CustomRuleRegistration, CustomRuleSource};
    use crate::operations::custom_functions::tests::{
        MemberDefinition, ReferenceRuleDifferentiationDriver, RuleCounters, TestContext, TestDefinition,
        TestRegistration, call_jvp, cube_definition, cube_program, custom_function_call_program, custom_rule_program,
        member_cube_definition,
    };
    use crate::operations::differentiation::linear_call::LinearCallOperation;
    use crate::operations::dimensions::dimension_from_scalar::DimensionFromScalarOperation;
    use crate::operations::manipulation::padding::PadOperation;
    use crate::operations::manipulation::slicing::Slice;
    use crate::operations::reductions::{Reduce, ReduceOperation, ReductionKind};
    use crate::operations::references::{
        ReferenceAddUpdate, ReferenceAddUpdateOperation, ReferenceFreezeOperation, ReferenceNew, ReferenceNewOperation,
        ReferenceRead, ReferenceReadOperation,
    };
    use crate::operations::trigonometric::{CosOperation, SinOperation};
    use crate::parameters::Placeholder;
    use crate::partial::{
        PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationOutput, PartialEvaluationValue,
        PartialValue, PartiallyEvaluatableOperation,
    };
    use crate::programs::{
        AtomId, EffectClass, EffectClasses, Effects, FlatProgram, InputRegionProvenance, MaybeZero, OperationProvider,
        OutputRegionProvenance, Program, ProgramBuilder, ReferenceAccessDescriptor, ReferenceAccessOperation,
        ReferenceType, RegionDriver, RegionInterface, RegionRole, RegionSlot, ValueProjection,
    };
    use crate::tests::{TestArrayOperation, hash_of};
    use crate::tracing::{DomainTracer, Trace, TracingContext};

    use crate::operations::differentiation::linear_call::tests::scalar_multiply_program;

    use super::*;

    /// Eager context whose values are arrays.
    type ArrayContext = EagerContext<Array, ArrayOperation<Array>>;

    /// Custom function call in the array operation family.
    type ArrayCustomFunction = CustomFunctionOperation<Array, ArrayOperation<Array>>;

    /// Reverse-mode carrier of a custom function call in the array operation family.
    type ArrayCustomFunctionTranspose = CustomFunctionTransposeOperation<Array, ArrayOperation<Array>>;

    /// Custom function call in the array IR operation family.
    type ArrayIrCustomFunction = CustomFunctionOperation<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Reverse-mode carrier of a custom function call in the array IR operation family.
    type ArrayIrCustomFunctionTranspose =
        CustomFunctionTransposeOperation<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Eager composite context whose values may be arrays or references.
    type EagerArrayIrContext = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

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
    fn nested_custom_function_state_program(
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
                ArrayIrCustomFunction::from_rule_regions(CustomFunctionJvpRule::Explicit, false),
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
        let program = custom_function_call_program(
            ArrayIrCustomFunction::from_rule_regions(CustomFunctionJvpRule::Explicit, false),
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
                let %1:f32[] = custom_function %0 [
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
        let operation = ArrayIrCustomFunction::from_rule_regions(CustomFunctionJvpRule::Explicit, with_vjp_rule);
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
        let program = custom_function_call_program(operation, regions, vec![ArrayType::scalar(DataType::F32).into()]);

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
        let program = custom_function_call_program(
            ArrayIrCustomFunction::from_rule_regions(CustomFunctionJvpRule::Explicit, false)
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

    /// Tracer of the composite universe used by the plumbing-reference tests.
    type ArrayIrTracer = DomainTracer<EagerArrayIrContext>;

    /// Error message of the forward-mode rejection of calls that have only reverse-mode rules.
    const FORWARD_MODE_REJECTION: &str = "cannot apply forward-mode differentiation to a `custom_function` call that \
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
    fn sin_regions(operation: &ArrayCustomFunction) -> Vec<FlatProgram<ArrayContext>> {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut regions = vec![sin_program(&scalar_type)];
        if operation.jvp_rule() == CustomFunctionJvpRule::Explicit {
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
            _destination_kinds: &[CotangentDestinationKind],
        ) -> Result<Arc<Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>>, DifferentiationError> {
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
                ArrayCustomFunctionTranspose::from_backward_region(1, vec![r#type.clone()], vec![r#type]),
                vec![backward],
                vec![residual, tangent],
                None,
            )
            .unwrap()[0];
        builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap()
    }

    // Fixtures of the tests of calls and carriers with retained rules.

    /// Array IR program family into which member programs are converted.
    type ConvertedProgram = FlatProgram<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>;

    /// Retained-rule definition over the production [`ArrayIrOperation`] family, which supports reference state.
    type IrDefinition = CustomRuleDefinition<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Tracer over which the rules of an [`IrDefinition`] are written.
    type IrTracer = CustomRuleTracer<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Multiplies two array IR tracers, whose composite values have no operator sugar.
    fn ir_multiply(left: &IrTracer, right: &IrTracer) -> Result<IrTracer, ProgramError> {
        let operation = ArrayIrOperation::from(ArrayOperation::<Array>::Mul(MulOperation::new()));
        Ok(left.context().bind(operation, Vec::new(), &[left.clone(), right.clone()])?.remove(0))
    }

    /// Stages one call of the array IR `definition` over `primal` with the provided input types.
    fn ir_custom_rule_program(
        definition: &CustomRuleRegistration<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
        non_differentiated_count: usize,
        primal: ConvertedProgram,
    ) -> ConvertedProgram {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input_types = primal.input_types();
        let primal = builder.import_program(primal);
        let inputs = input_types.into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let input_count = inputs.len();
        let operation = ArrayIrOperation::CustomFunction(
            CustomFunctionOperation::new(definition.reference())
                .with_non_differentiated_count(non_differentiated_count)
                .unwrap(),
        );
        let output = builder.add_instruction(operation, vec![primal], inputs, None).unwrap()[0];
        builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder; input_count],
                vec![Placeholder],
            )
            .unwrap()
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
        let operation = ArrayOperation::CustomFunction(CustomFunctionOperation::new(definition.reference()));
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
                let operation = CustomFunctionOperation::new(definition.reference());
                x.context().bind(operation, vec![cube_program(&r#type)], &[x.clone()])
            },
            x,
            (),
        )?;
        Ok((outputs, pullback.apply(vec![seed])?))
    }

    /// Batches `program` over a batch axis of size `extent` with the provided input batch axes.
    fn batched_program(
        program: &FlatProgram<TestContext>,
        extent: usize,
        input_axes: &[BatchAxis],
    ) -> FlatProgram<TestContext> {
        program
            .batched(extent, ShardingDimension::Replicated, input_axes, ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts()
            .0
    }

    /// Retained-rule definition over [`ReferenceTestOperation`].
    type ReferenceTestDefinition = CustomRuleDefinition<ArrayIrValue<Array>, ReferenceTestOperation>;

    /// Registration of a [`ReferenceTestDefinition`].
    type ReferenceTestRegistration = CustomRuleRegistration<ArrayIrValue<Array>, ReferenceTestOperation>;

    /// Program builder over [`ReferenceTestOperation`].
    type ReferenceTestBuilder = ProgramBuilder<ArrayIrValue<Array>, ReferenceTestOperation>;

    /// Reverse-mode carrier in the [`ReferenceTestOperation`] family.
    type ReferenceTestCarrier = CustomFunctionTransposeOperation<ArrayIrValue<Array>, ReferenceTestOperation>;

    /// Reference-capable operation family for the carrier tests that need caller-owned cotangent buffers or reference
    /// effects, which [`TestArrayOperation`] lacks. It hosts the reverse-mode carrier, which tests construct directly
    /// as reverse-mode linearization would stage it, together with the ordinary composite operations that backward
    /// rules emit, a region-carrying computation, and the production condition and linear call. It supports direct
    /// transposition rather than differentiation.
    #[derive(Clone, Debug)]
    enum ReferenceTestOperation {
        Base(ArrayIrOperation<Array>),
        Computation,
        Condition(ConditionOperation<ArrayIrType>),
        LinearCall(LinearCallOperation<ArrayIrType>),
        CustomFunctionTranspose(ReferenceTestCarrier),
    }

    impl Operation for ReferenceTestOperation {
        type Type = ArrayIrType;

        fn name(&self) -> &'static str {
            match self {
                Self::Base(operation) => operation.name(),
                Self::Computation => "computation",
                Self::Condition(operation) => operation.name(),
                Self::LinearCall(operation) => operation.name(),
                Self::CustomFunctionTranspose(operation) => operation.name(),
            }
        }

        fn region_slots(&self) -> &'static [RegionSlot] {
            match self {
                Self::Computation => const { &[RegionSlot::computation("body")] },
                Self::Condition(operation) => operation.region_slots(),
                Self::LinearCall(operation) => operation.region_slots(),
                Self::CustomFunctionTranspose(operation) => operation.region_slots(),
                Self::Base(_) => &[],
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[ArrayIrType],
            region_interfaces: &[RegionInterface<ArrayIrType>],
        ) -> Result<Vec<ArrayIrType>, TypeError> {
            match self {
                Self::Base(operation) => operation.infer_output_types(input_types, region_interfaces),
                Self::Computation => Ok(region_interfaces[0].output_types().to_vec()),
                Self::Condition(operation) => operation.infer_output_types(input_types, region_interfaces),
                Self::LinearCall(operation) => operation.infer_output_types(input_types, region_interfaces),
                Self::CustomFunctionTranspose(operation) => {
                    operation.infer_output_types(input_types, region_interfaces)
                }
            }
        }

        fn input_region_provenance(&self, region_index: usize, input_index: usize) -> InputRegionProvenance {
            match self {
                Self::Base(operation) => operation.input_region_provenance(region_index, input_index),
                Self::Computation => InputRegionProvenance::Input { index: input_index },
                Self::Condition(operation) => operation.input_region_provenance(region_index, input_index),
                Self::LinearCall(operation) => operation.input_region_provenance(region_index, input_index),
                Self::CustomFunctionTranspose(operation) => {
                    operation.input_region_provenance(region_index, input_index)
                }
            }
        }

        fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
            match self {
                Self::Base(operation) => operation.output_region_provenance(output_index),
                Self::Computation => vec![OutputRegionProvenance { region_index: 0, output_index }],
                Self::Condition(operation) => operation.output_region_provenance(output_index),
                Self::LinearCall(operation) => operation.output_region_provenance(output_index),
                Self::CustomFunctionTranspose(operation) => operation.output_region_provenance(output_index),
            }
        }

        fn effects(&self) -> Cow<'_, Effects> {
            match self {
                Self::Base(operation) => operation.effects(),
                Self::Computation => Cow::Borrowed(Effects::empty()),
                Self::Condition(operation) => operation.effects(),
                Self::LinearCall(operation) => operation.effects(),
                Self::CustomFunctionTranspose(operation) => operation.effects(),
            }
        }

        fn rename_type_identities(
            &self,
            renaming: &TypeIdentityRenaming<<ArrayIrType as Type>::Identity>,
        ) -> Result<Self, TypeError> {
            match self {
                Self::Base(operation) => operation.rename_type_identities(renaming).map(Self::Base),
                Self::Computation => Ok(Self::Computation),
                Self::Condition(operation) => operation.rename_type_identities(renaming).map(Self::Condition),
                Self::LinearCall(operation) => operation.rename_type_identities(renaming).map(Self::LinearCall),
                Self::CustomFunctionTranspose(operation) => {
                    operation.rename_type_identities(renaming).map(Self::CustomFunctionTranspose)
                }
            }
        }

        fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
            match self {
                Self::Base(operation) => operation.render(formatter, indentation),
                Self::Computation => write!(formatter, "computation"),
                Self::Condition(operation) => operation.render(formatter, indentation),
                Self::LinearCall(operation) => operation.render(formatter, indentation),
                Self::CustomFunctionTranspose(operation) => operation.render(formatter, indentation),
            }
        }
    }

    impl<Request> OperationProvider<ArrayIrType, Request> for ReferenceTestOperation
    where
        ArrayIrOperation<Array>: OperationProvider<ArrayIrType, Request, Operation = ArrayIrOperation<Array>>,
    {
        type Operation = Self;

        fn provide(request: Request, input_types: &[&ArrayIrType]) -> Result<Self, ProgramError> {
            ArrayIrOperation::<Array>::provide(request, input_types).map(Self::Base)
        }
    }

    impl From<AddOperation<ArrayIrType>> for ReferenceTestOperation {
        fn from(operation: AddOperation<ArrayIrType>) -> Self {
            Self::Base(operation.into())
        }
    }

    impl From<ConditionOperation<ArrayIrType>> for ReferenceTestOperation {
        fn from(operation: ConditionOperation<ArrayIrType>) -> Self {
            Self::Condition(operation)
        }
    }

    impl From<LinearCallOperation<ArrayIrType>> for ReferenceTestOperation {
        fn from(operation: LinearCallOperation<ArrayIrType>) -> Self {
            Self::LinearCall(operation)
        }
    }

    impl From<ReferenceTestCarrier> for ReferenceTestOperation {
        fn from(operation: ReferenceTestCarrier) -> Self {
            Self::CustomFunctionTranspose(operation)
        }
    }

    // Only static array shapes are used where structural cotangent zeros must be materialized.
    impl ResidualZeroProvider<ArrayIrType> for ReferenceTestOperation {}

    impl ReferenceAccessOperation for ReferenceTestOperation {
        type Transform = ArrayReferenceTransform;

        fn base_input_count(&self) -> usize {
            match self {
                Self::Base(operation) => operation.base_input_count(),
                Self::Computation => 2,
                Self::Condition(_) => 3,
                Self::LinearCall(_) => 0,
                Self::CustomFunctionTranspose(operation) => {
                    operation.leading_input_count + operation.input_tangent_types.len()
                }
            }
        }

        fn reference_access_descriptor(
            &self,
            input_index: usize,
        ) -> Option<ReferenceAccessDescriptor<'_, Self::Transform>> {
            match self {
                Self::Base(operation) => operation.reference_access_descriptor(input_index),
                Self::Computation | Self::Condition(_) | Self::LinearCall(_) | Self::CustomFunctionTranspose(_) => None,
            }
        }

        fn with_reference_access_transforms(
            &self,
            input_index: usize,
            transforms: Vec<Self::Transform>,
        ) -> Result<Self, ProgramError> {
            match self {
                Self::Base(operation) => {
                    operation.with_reference_access_transforms(input_index, transforms).map(Self::Base)
                }
                Self::Computation | Self::Condition(_) | Self::LinearCall(_) | Self::CustomFunctionTranspose(_) => {
                    Err(ProgramError::UnsupportedOperation {
                        message: format!("`{}` has no reference access inputs", self.name()),
                    })
                }
            }
        }
    }

    impl InterpretableOperation<EagerContext<ArrayIrValue<Array>, Self>> for ReferenceTestOperation {
        fn interpret<D: InterpretationDriver<EagerContext<ArrayIrValue<Array>, Self>>>(
            &self,
            context: &EagerContext<ArrayIrValue<Array>, Self>,
            driver: &D,
            inputs: &[ArrayIrValue<Array>],
        ) -> Result<Vec<ArrayIrValue<Array>>, ProgramError> {
            match self {
                Self::Base(operation) => EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new().bind(
                    operation.clone(),
                    Vec::new(),
                    inputs,
                ),
                Self::Computation => driver.interpret_region(context, 0, inputs.to_vec()),
                Self::Condition(operation) => operation.interpret(context, driver, inputs),
                Self::LinearCall(operation) => operation.interpret(context, driver, inputs),
                Self::CustomFunctionTranspose(operation) => operation.interpret(context, driver, inputs),
            }
        }
    }

    // Every operation other than the production condition uses the default rule, whose shared folding boundary
    // residualizes deferred work without carrier-specific checks.
    impl<C: Context<Type = ArrayIrType, Constant = ArrayIrValue<Array>, Operation = Self>>
        PartiallyEvaluatableOperation<C> for ReferenceTestOperation
    {
        fn partially_evaluate<D: PartialEvaluationDriver<C>>(
            &self,
            context: &PartialEvaluationContext<C>,
            driver: &D,
            inputs: &[PartialEvaluationValue<C::Value>],
        ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
            if let Self::Condition(operation) = self {
                return operation.partially_evaluate(context, driver, inputs);
            }
            context.fold_or_residualize(
                self.clone(),
                driver.regions().map(|region| region.to_program()).collect(),
                inputs,
            )
        }
    }

    impl TransposableOperation<ArrayIrValue<Array>, Self> for ReferenceTestOperation {
        fn transpose<D: TranspositionDriver<ArrayIrValue<Array>, Self>>(
            &self,
            context: &mut TranspositionContext<ArrayIrValue<Array>, Self>,
            driver: &D,
            inputs: &[PartialValue<CustomRuleTracer<ArrayIrValue<Array>, Self>>],
            outputs: &[MaybeZero<CustomRuleTracer<ArrayIrValue<Array>, Self>>],
            accumulators: &[CotangentAccumulator],
        ) -> Result<(), DifferentiationError> {
            match self {
                Self::Condition(_) => {
                    // Use the production branch transform with the same destination construction as its type-family
                    // rule.
                    let destinations = context.cotangent_destinations(driver, inputs, accumulators)?;
                    let contributions = transpose_primal_condition(context, driver, inputs, outputs, &destinations)?;
                    for (accumulator, contribution) in accumulators.iter().zip(contributions) {
                        accumulator.accumulate(context, contribution)?;
                    }
                    Ok(())
                }
                Self::LinearCall(operation) => operation.transpose(context, driver, inputs, outputs, accumulators),
                Self::CustomFunctionTranspose(operation) => {
                    operation.transpose(context, driver, inputs, outputs, accumulators)
                }
                Self::Base(_) | Self::Computation => Err(ProgramError::UnsupportedOperation {
                    message: format!("`{}` is not transposed by these tests", self.name()),
                }
                .into()),
            }
        }
    }

    /// Returns a reverse-mode carrier of `definition`, as reverse-mode linearization of a call would stage it, whose
    /// first `leading_input_count` inputs are known.
    fn reference_test_carrier(
        rules: CustomRuleReference<ArrayIrValue<Array>, ReferenceTestOperation>,
        leading_input_count: usize,
        input_tangent_types: Vec<ArrayIrType>,
        output_tangent_types: Vec<ArrayIrType>,
    ) -> ReferenceTestOperation {
        ReferenceTestOperation::CustomFunctionTranspose(CustomFunctionTransposeOperation::new(
            rules,
            leading_input_count,
            input_tangent_types,
            output_tangent_types,
        ))
    }

    /// Returns a definition whose backward rule increments the known leading reference by one exactly when its
    /// differentiated input's cotangent is ignored and its output seed is a structural zero, so the effect exists only
    /// in that specialization.
    fn ignored_zero_effect_definition() -> ReferenceTestRegistration {
        CustomRuleRegistration::new(ReferenceTestDefinition::new("ignored_zero_effect").with_accumulating_vjp(
            |_| unreachable!("the tests construct the carrier directly"),
            |context, inputs, seeds, accumulators| {
                if !accumulators[1].is_needed() && matches!(&seeds[0], MaybeZero::Zero(_)) {
                    let stash = inputs[0].as_known().unwrap().clone();
                    let increment = context.lift(ArrayIrValue::Array(Array::scalar(1.0f64)?))?;
                    context.bind(
                        ReferenceTestOperation::Base(ReferenceAddUpdateOperation::new().into()),
                        Vec::new(),
                        &[stash, increment],
                    )?;
                }
                Ok(())
            },
        ))
    }

    /// Returns the carrier of [`ignored_zero_effect_definition`] over a known reference and a scalar tangent.
    fn ignored_zero_effect_carrier() -> ReferenceTestOperation {
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F64).into();
        reference_test_carrier(
            ignored_zero_effect_definition().reference(),
            1,
            vec![scalar_type.clone()],
            vec![scalar_type],
        )
    }

    /// Returns a definition whose backward rule adds the output seed to the element at index 1 of its single
    /// differentiated input's cotangent, counting its invocations in `invocations`. A caller buffer receives a slice
    /// update without any full-sized temporary, while a returned cotangent is padded to the input extent.
    fn slice_at_one_definition(invocations: &Arc<AtomicUsize>) -> ReferenceTestRegistration {
        let invocations = invocations.clone();
        CustomRuleRegistration::new(ReferenceTestDefinition::new("slice_at_one").with_accumulating_vjp(
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
                        ReferenceTestOperation::Base(operation.into()),
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
                    let output = context
                        .bind(ReferenceTestOperation::Base(operation), Vec::new(), &[seed.clone(), zero])?
                        .remove(0);
                    accumulators[0].accumulate(context, MaybeZero::Value(output))?;
                }
                Ok(())
            },
        ))
    }

    /// Stages the carrier of `definition` over one differentiated input of type `input_type` with an `f64[1]` output.
    fn slice_at_one_program(
        definition: &ReferenceTestRegistration,
        input_type: ArrayType,
    ) -> Program<ArrayIrValue<Array>, ReferenceTestOperation, Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>> {
        let mut builder = ReferenceTestBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let carrier = reference_test_carrier(
            definition.reference(),
            0,
            vec![input_type.into()],
            vec![ArrayType::new_static(DataType::F64, [1]).into()],
        );
        let output = builder.add_instruction(carrier, Vec::new(), vec![input], None).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_custom_function() {
        let operation = ArrayCustomFunction::from_rule_regions(CustomFunctionJvpRule::Absent, false);
        assert_eq!(operation.name(), CUSTOM_FUNCTION_OPERATION_NAME);
        assert_eq!(operation.non_differentiated_count(), 0);
        assert_eq!(operation.rules(), None);
        assert_eq!(operation.jvp_rule(), CustomFunctionJvpRule::Absent);
        assert!(!operation.has_vjp_rule());
        assert_eq!(format!("{operation}"), "custom_function");
        assert_eq!(operation.region_slots(), &[RegionSlot::computation("primal")]);

        // Rule regions follow the primal region in the order `["jvp", "forward", "backward"]`, omitting the ones that a
        // call does not have, and every one of them is a dormant rule region without input provenance.
        let combined = ArrayCustomFunction::from_rule_regions(CustomFunctionJvpRule::Explicit, true);
        assert_eq!(combined.jvp_rule(), CustomFunctionJvpRule::Explicit);
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
        assert_eq!(format!("{combined}"), "custom_function");

        // A forward-mode rule derived from the primal region adds no region, so its rendering records it.
        let derived = ArrayCustomFunction::from_rule_regions(CustomFunctionJvpRule::Primal, false);
        assert_eq!(derived.region_slots(), &[RegionSlot::computation("primal")]);
        assert_eq!(format!("{derived}"), "custom_function [jvp_from_primal=true]");
        let derived = ArrayCustomFunction::from_rule_regions(CustomFunctionJvpRule::Primal, true);
        assert_eq!(
            format!("{}", derived.clone().with_non_differentiated_count(1).unwrap()),
            "custom_function [non_differentiated_count=1, jvp_from_primal=true]",
        );
        assert_eq!(
            derived.region_slots(),
            &[RegionSlot::computation("primal"), RegionSlot::rule("forward"), RegionSlot::rule("backward")],
        );
    }

    #[test]
    fn test_custom_function_jvp_rule() {
        let operation = ArrayCustomFunction::from_rule_regions(CustomFunctionJvpRule::Explicit, false);
        assert_eq!(operation.name(), CUSTOM_FUNCTION_OPERATION_NAME);
        assert_eq!(operation.non_differentiated_count(), 0);
        assert_eq!(operation.jvp_rule(), CustomFunctionJvpRule::Explicit);
        assert!(!operation.has_vjp_rule());
        assert_eq!(format!("{operation}"), "custom_function");
        assert_eq!(
            format!("{operation:?}"),
            "CustomFunctionOperation { non_differentiated_count: 0, jvp_rule: Explicit, has_vjp_rule: false }",
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
    fn test_custom_function_vjp_rule() {
        let operation = ArrayCustomFunction::from_rule_regions(CustomFunctionJvpRule::Absent, true);
        assert_eq!(operation.name(), CUSTOM_FUNCTION_OPERATION_NAME);
        assert_eq!(operation.non_differentiated_count(), 0);
        assert_eq!(operation.jvp_rule(), CustomFunctionJvpRule::Absent);
        assert!(operation.has_vjp_rule());
        assert_eq!(format!("{operation}"), "custom_function");
        assert_eq!(
            format!("{operation:?}"),
            "CustomFunctionOperation { non_differentiated_count: 0, jvp_rule: Absent, has_vjp_rule: true }",
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
    fn test_custom_function_with_non_differentiated_count() {
        let operation = ArrayCustomFunction::from_rule_regions(CustomFunctionJvpRule::Explicit, false)
            .with_non_differentiated_count(1)
            .unwrap();
        assert_eq!(operation.non_differentiated_count(), 1);
        assert_eq!(format!("{operation}"), "custom_function [non_differentiated_count=1]");

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
    fn test_custom_function_with_non_differentiated_count_vjp_rule() {
        let operation = ArrayCustomFunction::from_rule_regions(CustomFunctionJvpRule::Absent, true)
            .with_non_differentiated_count(1)
            .unwrap();
        assert_eq!(operation.non_differentiated_count(), 1);
        assert_eq!(format!("{operation}"), "custom_function [non_differentiated_count=1]");

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
    fn test_custom_function_retained_rules() {
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, true));
        let operation = CustomFunctionOperation::new(definition.reference());
        assert_eq!(operation, CustomFunctionOperation::new(definition.reference()));
        assert_eq!(hash_of(&operation), hash_of(&CustomFunctionOperation::new(definition.reference())));
        assert_ne!(operation, operation.clone().with_non_differentiated_count(1).unwrap());

        // Identity is the definition allocation, so an equally named definition built from the same closure types is a
        // different rule set, even though it renders identically.
        let twin = CustomFunctionOperation::new(
            CustomRuleRegistration::new(cube_definition(&counters, true, true)).reference(),
        );
        assert_ne!(operation, twin);
        assert_eq!(operation.to_string(), twin.to_string());
        assert_eq!(operation.to_string(), "custom_function [name=\"cube\"]");
        assert_eq!(
            operation.clone().with_non_differentiated_count(1).unwrap().to_string(),
            "custom_function [name=\"cube\", non_differentiated_count=1]",
        );
        assert_eq!(operation.rules().unwrap().name(), "cube");
        assert_eq!(
            format!("{:?}", operation),
            "CustomFunctionOperation { name: \"cube\", non_differentiated_count: 0, batching: None, discharged: \
             false }",
        );
    }

    #[test]
    fn test_custom_function_type_inference() {
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
        let combined = ArrayCustomFunction::from_rule_regions(CustomFunctionJvpRule::Explicit, true);
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
        let derived = ArrayCustomFunction::from_rule_regions(CustomFunctionJvpRule::Primal, false);
        assert_eq!(derived.infer_output_types(&[scalar_type.clone()], &[primal]), Ok(vec![scalar_type]));
    }

    #[test]
    fn test_custom_function_type_inference_jvp_rule() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let vector_type = ArrayType::new_static(DataType::F64, [2]);
        let operation = ArrayCustomFunction::from_rule_regions(CustomFunctionJvpRule::Explicit, false);
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
                "`custom_function` JVP rule input type signature mismatch: expected [f64[], f64[]] but got [f64[]]"
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
                "`custom_function` JVP rule output type signature mismatch: expected [f64[], f64[]] but got [f64[]]"
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
                "`custom_function` input type signature mismatch: expected [f64[]] but got [f64[2]]".to_string(),
            )),
        );

        // Output inference independently rejects an invalid non-differentiated count, even when region input
        // inference has not run first.
        assert_eq!(
            operation.clone().with_non_differentiated_count(2).unwrap().infer_output_types(
                std::slice::from_ref(&scalar_type),
                &[primal_interface.clone(), jvp_interface.clone()],
            ),
            Err(TypeError::invalid(
                "`custom_function` non-differentiated input count 2 exceeds input count 1".to_string()
            )),
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
            Err(TypeError::invalid(
                "`custom_function` non-differentiated input count 2 exceeds input count 1".to_string()
            )),
        );
    }

    #[test]
    fn test_custom_function_type_inference_jvp_rule_references() {
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
            ArrayIrCustomFunction::from_rule_regions(CustomFunctionJvpRule::Explicit, false)
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
            ArrayIrCustomFunction::from_rule_regions(CustomFunctionJvpRule::Explicit, false)
                .infer_output_types(&input_types, &[primal_interface, active_jvp_interface]),
            Err(TypeError::invalid(
                "`custom_function` accepts reference inputs only in its leading non-differentiated segment; move \
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
            ArrayIrCustomFunction::from_rule_regions(CustomFunctionJvpRule::Explicit, false)
                .with_non_differentiated_count(1)
                .unwrap()
                .infer_output_types(&input_types, &[forwarding_primal_interface, forwarding_jvp_interface]),
            Err(TypeError::invalid(
                "`custom_function` cannot return a reference, but output 0 has type `ref<f32[]>`".to_string(),
            )),
        );
    }

    #[test]
    fn test_custom_function_type_inference_vjp_rule() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let vector_type = ArrayType::new_static(DataType::F64, [2]);
        let operation = ArrayCustomFunction::from_rule_regions(CustomFunctionJvpRule::Absent, true);
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
                "`custom_function` forward rule input type signature mismatch: expected [f64[]] but got [f64[2]]"
                    .to_string(),
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
                "`custom_function` forward rule must produce at least the 1 primal output(s) but produced 0 value(s)"
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
                "`custom_function` forward rule output type signature mismatch: expected [f64[]] but got [f64[2]]"
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
                "`custom_function` backward rule input type signature mismatch: expected [f64[], f64[]] but got \
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
                "`custom_function` backward rule output type signature mismatch: expected [f64[]] but got []"
                    .to_string(),
            )),
        );

        // The call inputs must match the primal region inputs.
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&vector_type),
                &[primal_interface.clone(), forward_interface.clone(), backward_interface.clone()],
            ),
            Err(TypeError::invalid(
                "`custom_function` input type signature mismatch: expected [f64[]] but got [f64[2]]".to_string(),
            )),
        );

        // Standalone output inference must reject an excessive non-differentiated count as well.
        assert_eq!(
            operation.clone().with_non_differentiated_count(2).unwrap().infer_output_types(
                std::slice::from_ref(&scalar_type),
                &[primal_interface.clone(), forward_interface.clone(), backward_interface.clone()],
            ),
            Err(TypeError::invalid(
                "`custom_function` non-differentiated input count 2 exceeds input count 1".to_string()
            )),
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
            Err(TypeError::invalid(
                "`custom_function` non-differentiated input count 2 exceeds input count 1".to_string()
            )),
        );
    }

    #[test]
    fn test_custom_function_type_inference_vjp_rule_references() {
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
        let operation = ArrayIrCustomFunction::from_rule_regions(CustomFunctionJvpRule::Absent, true)
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
                "`custom_function` forward rule returns residual 0 of reference type `ref<f64[]>`, which matches \
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
                "`custom_function` forward rule returns residual 1 of reference type `ref<f32[]>`, but every leading \
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
            ArrayIrCustomFunction::from_rule_regions(CustomFunctionJvpRule::Absent, true)
                .infer_output_types(&input_types, &[primal_interface, forward_interface, active_backward_interface]),
            Err(TypeError::invalid(
                "`custom_function` accepts reference inputs only in its leading non-differentiated segment; move \
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
                "`custom_function` cannot return a reference, but output 0 has type `ref<f32[]>`".to_string(),
            )),
        );
    }

    #[test]
    fn test_custom_function_type_inference_retained_rules() {
        // A call returns the outputs of its primal region, whose inputs are the call's inputs.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, false));
        let scalar_type = ArrayType::scalar(DataType::F64);
        let region = RegionInterface::new(vec![scalar_type.clone()], vec![scalar_type.clone()], EffectClasses::NONE);
        let operation = CustomFunctionOperation::new(definition.reference());
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
            Err(TypeError::invalid(
                "`custom_function` input type signature mismatch: expected [f64[]] but got [f32[]]"
            )),
        );
        assert_eq!(
            Operation::infer_output_types(
                &operation.clone().with_non_differentiated_count(2).unwrap(),
                &[scalar_type.clone()],
                std::slice::from_ref(&region),
            ),
            Err(TypeError::invalid("`custom_function` non-differentiated input count 2 exceeds input count 1")),
        );

        // A batched call's leading inputs are the boundary inputs of its batching levels, which a call that is
        // reconfigured after it was batched must still declare as non-differentiated.
        let definition = CustomRuleRegistration::new(
            IrDefinition::new("identity")
                .with_jvp(|primals, tangents| Ok((primals.to_vec(), tangents.to_vec())))
                .with_batching(),
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
        let ArrayIrOperation::CustomFunction(batched_operation) = batched.instructions()[0].operation().clone() else {
            panic!("batching should stage a custom rule call");
        };
        assert!(matches!(
            batched.map_operations(|operation| {
                Ok(match operation {
                    ArrayIrOperation::CustomFunction(operation) => {
                        ArrayIrOperation::CustomFunction(operation.clone().with_non_differentiated_count(0)?)
                    }
                    operation => operation.clone(),
                })
            }),
            Err(ProgramError::Type(error)) if error == TypeError::invalid(
                "batched `custom_function` `identity` must have at least 1 non-differentiated inputs, which are the \
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
                "batched `custom_function` `identity` has 1 inputs but its batching levels record 1 boundary \
                 inputs and 1 unbatched inputs",
            )),
        );
    }

    #[test]
    fn test_custom_function_type_inference_retained_rules_reference_boundary() {
        // The reference contract is validated when a call is staged: references are accepted only in the leading
        // non-differentiated segment.
        let definition = CustomRuleRegistration::new(IrDefinition::new("counter"));
        let program = ir_counter_square_program();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let primal = builder.import_program(program.clone());
        let inputs = program.input_types().into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        assert_eq!(
            builder
                .add_instruction(
                    ArrayIrOperation::CustomFunction(CustomFunctionOperation::new(definition.reference())),
                    vec![primal],
                    inputs,
                    None,
                )
                .unwrap_err()
                .to_string(),
            "`custom_function` accepts reference inputs only in its leading non-differentiated segment; move input 0 \
             of type `ref<f32[]>` before the differentiated inputs",
        );
        let staged = ir_custom_rule_program(&definition, 1, program);
        assert_eq!(
            staged.to_string(),
            indoc! {"
                lambda %0:ref<f32[]>, %1:f32[] .
                let %2:f32[] = custom_function [name=\"counter\", non_differentiated_count=1] %0 %1 [
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
    fn test_custom_function_reference_discharge_jvp_rule() {
        // Discharge preserves both a read of the accumulator and a consuming freeze.
        check_custom_jvp_reference_discharge(false);
        check_custom_jvp_reference_discharge(true);
    }

    #[test]
    fn test_custom_function_reference_discharge_vjp_rule() {
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
        let program = custom_function_call_program(
            ArrayIrCustomFunction::from_rule_regions(CustomFunctionJvpRule::Absent, true),
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
                let %1:f32[] = custom_function %0 [
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
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose)
            .unwrap();
        let mut primal_outputs = linearization.primal().interpret(vec![input]).unwrap();
        let mut cotangents = vec![ArrayIrValue::Array(Array::scalar(1.0f32).unwrap())];
        cotangents.extend(primal_outputs.split_off(1));
        assert_eq!(
            linearization.pullback().unwrap().interpret(cotangents),
            Ok(vec![ArrayIrValue::Array(Array::scalar(3.0f32).unwrap())]),
        );
    }

    #[test]
    fn test_custom_function_reference_discharge_vjp_rule_rejects_external_references() {
        // A custom-VJP call threads a plumbing reference into its dormant forward and backward rules, whose
        // reference-typed inputs are bound by the transform that instantiates them and therefore declare no input
        // provenance. Summarizing a condition branch containing such a call skips those rules exactly as the reference
        // analysis does, so discharging the program reaches the call's own discharge rule, which reports that a
        // caller reference cannot cross the custom-VJP boundary, instead of failing on undeclared provenance.
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));

        let branch = custom_function_call_program(
            ArrayIrCustomFunction::from_rule_regions(CustomFunctionJvpRule::Absent, true)
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
        let program = custom_function_call_program(
            ConditionOperation::new(),
            vec![branch.clone(), branch],
            vec![ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)), reference_type, scalar_type],
        );
        assert!(matches!(
            program.discharge_references(0),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "`custom_function` does not thread external references through discharge, but input 0 \
                    is a reference; pass reference-free inputs or discharge external references first",
        ));
    }

    #[test]
    fn test_custom_function_reference_discharge_retained_rules() {
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
                let %1:f32[] = custom_function [name=\"doubled\", discharged=true] %0 [
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
        assert_eq!(
            discharged.jvp().unwrap_err().to_string(),
            "`custom_function` `doubled` cannot differentiate a call whose reference state was discharged, because \
             its rule programs contain reference state and its definition has no reference discharge support",
        );
    }

    #[test]
    fn test_custom_function_interpretation_jvp_rule() {
        // Interpretation replays the primal region only, so an un-differentiated call never pays for the tangent
        // computation of the JVP region.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let operation = ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
            CustomFunctionJvpRule::Explicit,
            false,
        ));
        let regions = vec![sin_program(&scalar_type), doubled_sin_jvp_program(&scalar_type)];
        assert_eq!(
            ArrayContext::new().bind(operation, regions, &[Array::scalar(2.0).unwrap()]),
            Ok(vec![Array::scalar(2.0f64.sin()).unwrap()]),
        );
    }

    #[test]
    fn test_custom_function_interpretation_vjp_rule() {
        // Assertions in both derivative rules make accidental execution observable: interpreting the primal must
        // succeed without running either dormant rule.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let operation = ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
            CustomFunctionJvpRule::Absent,
            true,
        ));
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
    fn test_custom_function_interpretation_retained_rules() {
        // An ordinary call computes the primal without invoking any retained rule.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, true));
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = custom_function [name=\"cube\"] %0 [
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
    fn test_custom_function_partial_evaluation_jvp_rule() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let operation = ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
            CustomFunctionJvpRule::Explicit,
            false,
        ));
        let regions = vec![sin_program(&scalar_type), doubled_sin_jvp_program(&scalar_type)];
        let program = custom_function_call_program(operation, regions, vec![scalar_type.clone()]);

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
                let %1:f64[] = custom_function %0 [
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
    fn test_custom_function_partial_evaluation_vjp_rule() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let operation = ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
            CustomFunctionJvpRule::Absent,
            true,
        ));
        let regions = vec![
            sin_program(&scalar_type),
            sin_forward_program(&scalar_type),
            tripled_sin_backward_program(&scalar_type),
        ];
        let program = custom_function_call_program(operation, regions, vec![scalar_type.clone()]);

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
                let %1:f64[] = custom_function %0 [
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
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose)
            .unwrap();
        let mut outputs = linearization.primal().interpret(vec![Array::scalar(2.0).unwrap()]).unwrap();
        assert_eq!(outputs.remove(0), Array::scalar(2.0f64.sin()).unwrap());
        let mut cotangents = vec![Array::scalar(1.0).unwrap()];
        cotangents.extend(outputs);
        assert_eq!(
            linearization.pullback().unwrap().interpret(cotangents),
            Ok(vec![Array::scalar(3.0 * 2.0f64.cos()).unwrap()]),
        );
    }

    #[test]
    fn test_custom_function_batching() {
        // Batching a call that has every kind of rule batches all of its regions and preserves both rules, so forward
        // mode through the batched call still uses the doubled JVP rule and reverse mode still uses the tripled
        // reverse-mode rules.
        let operation = CustomFunctionOperation::from_rule_regions(CustomFunctionJvpRule::Explicit, true);
        let regions = sin_regions(&operation);
        let (value, tangent) = differentiate_at(Array::vector(vec![0.5, 1.0]).unwrap())
            .jvp(Array::vector(vec![1.0, 1.0]).unwrap(), |x| {
                batch(
                    |item| {
                        let operation = ArrayOperation::CustomFunction(operation.clone());
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
                        let operation = ArrayOperation::CustomFunction(operation.clone());
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
    fn test_custom_function_batching_jvp_rule() {
        let output: Array = batch(
            |x| {
                let scalar_type = ArrayType::scalar(DataType::F64);
                let operation = ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
                    CustomFunctionJvpRule::Explicit,
                    false,
                ));
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
    fn test_custom_function_batching_jvp_rule_natural_output_axes() {
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
        let program: FlatProgram<ArrayContext> = custom_function_call_program(
            ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
                CustomFunctionJvpRule::Explicit,
                false,
            )),
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
                let %1:f64[3, 2], %2:f64[3], %3:f64[3, 2] = custom_function %0 [
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
    fn test_custom_function_batching_jvp_rule_replicated_input_custom_cotangent_reduction() {
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
        let program = custom_function_call_program(
            ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
                CustomFunctionJvpRule::Explicit,
                false,
            )),
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
                                    ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
                                        CustomFunctionJvpRule::Explicit,
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
            Ok((Array::vector(vec![0.0, 0.0, 0.0]).unwrap(), Array::scalar(28.0).unwrap())),
        );
    }

    #[test]
    fn test_custom_function_batching_jvp_rule_named_axis_outputs() {
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
        let regions = vec![primal, jvp];
        let driver = RecursiveBatchingDriver::new(&regions);
        let context =
            BatchingContext::<_, ArrayBatchingPolicy>::new(ArrayContext::new(), 3).with_axis_name("items".to_string());

        // No input carries a mapped axis, but `axis_index("items")` observes the active transform inside both regions
        // and naturally produces a mapped output. Batching must therefore inspect the regions rather than assuming
        // that all-replicated call inputs imply all-replicated call outputs.
        let outputs = CustomFunctionOperation::from_rule_regions(CustomFunctionJvpRule::Explicit, false)
            .batch(&context, &driver, &[ArrayBatch::replicated(Array::scalar(1.0).unwrap())])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &Array::vector(vec![0u64, 1, 2]).unwrap());
    }

    #[test]
    fn test_custom_function_batching_jvp_rule_preserves_custom_function() {
        // Differentiating through a batched custom call must still use the deliberately doubled custom rule, because
        // batching preserves the call around batched regions instead of inlining its primal region.
        let (value, gradient) = differentiate_at(Array::vector(vec![0.5, 1.0]).unwrap())
            .value_and_gradient(|x| {
                let mapped = batch(
                    |item| {
                        let scalar_type = ArrayType::scalar(DataType::F64);
                        let operation = ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
                            CustomFunctionJvpRule::Explicit,
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
    fn test_custom_function_batching_vjp_rule() {
        let output: Array = batch(
            |x| {
                let scalar_type = ArrayType::scalar(DataType::F64);
                let operation = ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
                    CustomFunctionJvpRule::Absent,
                    true,
                ));
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
    fn test_custom_function_batching_vjp_rule_residual_axes_and_replicated_cotangents() {
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
        let program: FlatProgram<ArrayContext> = custom_function_call_program(
            ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
                CustomFunctionJvpRule::Absent,
                true,
            )),
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
                let %2:f64[2] = custom_function %0 %1 [
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
            .linearize_shared_for_rule(&[0, 1], DifferentiationRule::JvpForTranspose)
            .unwrap();
        let mut outputs = linearization
            .primal()
            .interpret(vec![Array::vector(vec![2.0, 3.0]).unwrap(), Array::scalar(5.0).unwrap()])
            .unwrap();
        assert_eq!(outputs.remove(0), Array::vector(vec![10.0, 15.0]).unwrap());
        let mut cotangents = vec![Array::vector(vec![7.0, 11.0]).unwrap()];
        cotangents.extend(outputs);
        assert_eq!(
            linearization.pullback().unwrap().interpret(cotangents),
            Ok(vec![Array::vector(vec![35.0, 55.0]).unwrap(), Array::scalar(47.0).unwrap()]),
        );
    }

    #[test]
    fn test_custom_function_batching_vjp_rule_preserves_custom_function() {
        // Differentiating through a batched custom call must still use the deliberately tripled custom backward rule,
        // because batching preserves the call around batched regions instead of inlining its primal region.
        let (value, gradient) = differentiate_at(Array::vector(vec![0.5, 1.0]).unwrap())
            .value_and_gradient(|x| {
                let mapped = batch(
                    |item| {
                        let scalar_type = ArrayType::scalar(DataType::F64);
                        let operation = ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
                            CustomFunctionJvpRule::Absent,
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
    fn test_custom_function_batching_retained_rules() {
        // Batching a call batches its primal and records the level without invoking any rule.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, true));
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        let batched = batched_program(&program, 3, &[BatchAxis::new(0)]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[3] = custom_function [name=\"cube\", batching=[(extent=3, input_axes=[axis 0], output_axes=[axis 0])]] %0 [
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
    fn test_custom_function_batching_retained_rules_custom_batching_rule() {
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
                let %1:f64[3] = custom_function [name=\"cube\", batching=[(extent=3, input_axes=[axis 0], output_axes=[axis 0])]] %0 [
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
    fn test_custom_function_batching_retained_rules_differentiation() {
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
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose)
            .unwrap();
        assert_eq!(
            reverse.tangent().to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3] .
                let %2:f64[3] = custom_function_transpose [
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
    fn test_custom_function_batching_retained_rules_output_layout() {
        // `f(c, p) = p` with the rule `ḟ = c ṗ`, where `c` is non-differentiated: batching a mapped `c` and a
        // replicated `p` leaves the primal replicated, while the derivative is mapped.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let primal = {
            let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
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
            .with_batching();
        let program = |definition: TestDefinition| {
            let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
            let primal = builder.import_program(primal.clone());
            let coefficient = builder.add_input(scalar_type.clone());
            let parameter = builder.add_input(scalar_type.clone());
            let operation = CustomFunctionOperation::new(CustomRuleRegistration::new(definition).reference())
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
            .with_batching()
            .with_batched_output_axes(vec![BatchAxis::replicated()]);
        let batched = batched_program(&program(definition), 3, &[BatchAxis::new(0), BatchAxis::replicated()]);
        assert_eq!(batched.interpret(vec![coefficient.clone(), parameter.clone()]), Ok(vec![parameter.clone()]));
        assert_eq!(
            batched.entry_region_ref().jvp(&[1]).unwrap_err().to_string(),
            "a custom rule output is mapped along axis 0 but its batched layout requires it to be replicated",
        );

        // A declared replicated output is valid when the derivative agrees with it. Here, the second output is `p`
        // itself, whose tangent `ṗ` stays replicated, while the first output is mapped.
        let definition = TestDefinition::new("scaled_parameter_pair")
            .with_jvp(|primals, tangents| {
                let outputs = vec![primals[1].clone(), primals[1].clone()];
                Ok((outputs, vec![primals[0].clone() * tangents[0].clone(), tangents[0].clone()]))
            })
            .with_batching()
            .with_batched_output_axes(vec![BatchAxis::new(0), BatchAxis::replicated()]);
        let pair_primal = {
            let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
            builder.add_input(scalar_type.clone());
            let parameter = builder.add_input(scalar_type.clone());
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![parameter, parameter], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let pair_program = {
            let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
            let primal = builder.import_program(pair_primal);
            let coefficient = builder.add_input(scalar_type.clone());
            let parameter = builder.add_input(scalar_type.clone());
            let operation = CustomFunctionOperation::new(CustomRuleRegistration::new(definition).reference())
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
            .with_batching()
            .with_batched_output_axes(vec![BatchAxis::replicated()]);
        assert_eq!(
            program(definition)
                .batched(
                    3,
                    ShardingDimension::Replicated,
                    &[BatchAxis::replicated(), BatchAxis::new(0)],
                    ProgramBatchingOutputAxesPolicy::Natural,
                )
                .map(|_| ())
                .unwrap_err()
                .to_string(),
            "batched region output axes [axis 0] do not match the target output axes [replicated]",
        );
    }

    #[test]
    fn test_custom_function_batching_retained_rules_nested() {
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
    fn test_custom_function_batching_retained_rules_without_batcher() {
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
        assert_eq!(
            batched.jvp().unwrap_err().to_string(),
            "`custom_function` `cube` cannot differentiate a batched call because its definition has no batching \
             support",
        );
    }

    #[test]
    fn test_custom_function_batching_retained_rules_renaming() {
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
    fn test_custom_function_differentiation() {
        // The rule-selection contract: every layout of rules selects exactly the rule that each mode requires, and a
        // missing rule never falls back to differentiating the primal region. The rule regions of the sine fixture
        // double (JVP rule) or triple (reverse-mode rules) the true derivative `cos(x)`, which a derived JVP computes.
        let no_rule = "cannot differentiate a `custom_function` call that has no derivative rule";
        let (absent, explicit, primal) =
            (CustomFunctionJvpRule::Absent, CustomFunctionJvpRule::Explicit, CustomFunctionJvpRule::Primal);
        let x = 0.5f64;
        for (jvp_rule, has_vjp_rule, tangent, gradient) in [
            (absent, false, Err(no_rule), Err(no_rule)),
            (primal, false, Ok(x.cos()), Ok(x.cos())),
            (explicit, false, Ok(2.0 * x.cos()), Ok(2.0 * x.cos())),
            (absent, true, Err(FORWARD_MODE_REJECTION), Ok(3.0 * x.cos())),
            (explicit, true, Ok(2.0 * x.cos()), Ok(3.0 * x.cos())),
            (primal, true, Ok(x.cos()), Ok(3.0 * x.cos())),
        ] {
            let operation = ArrayCustomFunction::from_rule_regions(jvp_rule, has_vjp_rule);
            let regions = sin_regions(&operation);
            let result = differentiate_at(Array::scalar(x).unwrap()).jvp(Array::scalar(1f64).unwrap(), |x| {
                Ok(x.context()
                    .bind(ArrayOperation::CustomFunction(operation.clone()), regions.clone(), &[x.clone()])?
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
                        .bind(ArrayOperation::CustomFunction(operation.clone()), regions.clone(), &[x.clone()])?
                        .remove(0))
                })
                .and_then(|(value, pullback)| Ok((value, pullback.apply(Array::scalar(1f64).unwrap())?)));
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
            let operation = ArrayCustomFunction::from_rule_regions(jvp_rule, has_vjp_rule);
            let regions = sin_regions(&operation);
            let (_, second_derivative) = differentiate_at(Array::scalar(x).unwrap())
                .jvp(Array::scalar(1f64).unwrap(), |x| {
                    Ok(differentiate_at(x)
                        .gradient(|y| {
                            let operation = ArrayOperation::CustomFunction(operation.clone());
                            y.context().bind(operation, regions.clone(), &[y.clone()]).unwrap().remove(0)
                        })
                        .unwrap())
                })
                .unwrap();
            assert_eq!(second_derivative, Array::scalar(expected).unwrap());
        }
    }

    #[test]
    fn test_zz_probe_custom_function_zero_tangents() {
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
        let operation = CustomFunctionOperation::from_rule_regions(CustomFunctionJvpRule::Primal, false);
        let program = custom_function_call_program(
            ArrayOperation::CustomFunction(operation),
            vec![primal],
            vec![scalar_type.clone(), scalar_type],
        );
        println!("PROBE custom mixed linearize: {:?}", program.linearize_with_respect_to(&[0]).map(|_| ()));
        println!("PROBE custom mixed jvp: {:?}", program.jvp_with_respect_to(&[0]).map(|p| p.to_string()));
    }

    #[test]
    fn test_custom_function_differentiation_jvp_from_primal_linearization() {
        // Reusable linearization partitions a derived JVP like the primal region's own derivative: the primal program
        // saves `cos(x)` and the tangent program applies it.
        let operation = CustomFunctionOperation::from_rule_regions(CustomFunctionJvpRule::Primal, false);
        let program = custom_function_call_program(
            ArrayOperation::CustomFunction(operation.clone()),
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
    fn test_custom_function_differentiation_jvp_from_primal_non_differentiated_inputs() {
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
        let operation = ArrayOperation::CustomFunction(
            CustomFunctionOperation::from_rule_regions(CustomFunctionJvpRule::Primal, false)
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
    fn test_custom_function_differentiation_combined_rules_fresh_tangent_state() {
        // Adding reverse-mode rules does not change forward-mode linearization: the tangent program still allocates
        // fresh JVP rule state on every application, exactly as for a call that has only a JVP rule.
        check_custom_jvp_linearization_fresh_tangent_state(false, true);
        check_custom_jvp_linearization_fresh_tangent_state(true, true);
    }

    #[test]
    fn test_custom_function_differentiation_combined_rules_repeated_pullback_effects() {
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
        let operation = ArrayIrOperation::CustomFunction(
            CustomFunctionOperation::from_rule_regions(CustomFunctionJvpRule::Explicit, true)
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
                Ok((None, Some(ArrayIrValue::Array(Array::scalar(6.0f32).unwrap())))),
            );
            assert_eq!(counter.read(), Ok(Array::scalar(application as f32).unwrap()));
        }
    }

    #[test]
    fn test_custom_function_differentiation_jvp_rule() {
        // The custom rule doubles the true derivative, which proves that it governs forward-mode differentiation.
        let (primal, tangent) = differentiate_at(Array::scalar(2.0).unwrap())
            .jvp(Array::scalar(1.0).unwrap(), |x| {
                let scalar_type = ArrayType::scalar(DataType::F64);
                let operation = ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
                    CustomFunctionJvpRule::Explicit,
                    false,
                ));
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
                let operation = ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
                    CustomFunctionJvpRule::Explicit,
                    false,
                ));
                let regions = vec![sin_program(&scalar_type), doubled_sin_jvp_program(&scalar_type)];
                x.context().bind(operation, regions, &[x.clone()]).unwrap().remove(0)
            })
            .unwrap();
        assert_eq!(value, Array::scalar(3.0f64.sin()).unwrap());
        assert_eq!(gradient, Array::scalar(2.0 * 3.0f64.cos()).unwrap());
    }

    #[test]
    fn test_custom_function_differentiation_jvp_rule_second_order() {
        // The JVP rule replays the user program as plain primitive operations, so the gradient program that it
        // produces is itself differentiable. The doubled rule makes the first derivative `2 cos(x)`, and so the second
        // derivative is `-2 sin(x)`.
        let (gradient, second_derivative) = differentiate_at(Array::scalar(0.7).unwrap())
            .value_and_gradient(|x| {
                differentiate_at(x)
                    .gradient(|y| {
                        let scalar_type = ArrayType::scalar(DataType::F64);
                        let operation = ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
                            CustomFunctionJvpRule::Explicit,
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
    fn test_custom_function_differentiation_jvp_rule_forward_over_reverse() {
        // Forward differentiation of the gradient differentiates the ordinary operations that the JVP rule replayed
        // into the pullback, so the doubled rule yields the second derivative `-2 sin(x)`.
        let (gradient, second_derivative) = differentiate_at(Array::scalar(0.7).unwrap())
            .jvp(Array::scalar(1.0).unwrap(), |x| {
                Ok(differentiate_at(x)
                    .gradient(|y| {
                        let scalar_type = ArrayType::scalar(DataType::F64);
                        let operation = ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
                            CustomFunctionJvpRule::Explicit,
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
    fn test_custom_function_differentiation_jvp_rule_reverse_over_forward() {
        // Reverse differentiation of the directional derivative differentiates the rule's replayed coefficient.
        let (tangent, second_derivative) = differentiate_at(Array::scalar(0.7).unwrap())
            .value_and_gradient(|x| {
                let direction = x.context().lift(Array::scalar(1.0).unwrap()).unwrap();
                differentiate_at(x)
                    .jvp(direction, |y| {
                        let scalar_type = ArrayType::scalar(DataType::F64);
                        let operation = ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
                            CustomFunctionJvpRule::Explicit,
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
    fn test_custom_function_differentiation_jvp_rule_forward_over_forward() {
        let (tangent, second_derivative) = differentiate_at(Array::scalar(0.7).unwrap())
            .jvp(Array::scalar(1.0).unwrap(), |x| {
                let direction = x.context().lift(Array::scalar(1.0).unwrap())?;
                Ok(differentiate_at(x)
                    .jvp(direction, |y| {
                        let scalar_type = ArrayType::scalar(DataType::F64);
                        let operation = ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
                            CustomFunctionJvpRule::Explicit,
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
    fn test_custom_function_differentiation_jvp_rule_pullback() {
        // The pullback of a JVP-only rule is linear in its seed, so its derivative with respect to the seed is the
        // custom coefficient `2 cos(x)`.
        let (cotangent, seed_derivative) = differentiate_at(Array::scalar(3.0).unwrap())
            .jvp(Array::scalar(1.0).unwrap(), |seed| {
                let input = seed.context().lift(Array::scalar(0.7).unwrap())?;
                let (_, pullback) = differentiate_at(input)
                    .vjp(|y| {
                        let scalar_type = ArrayType::scalar(DataType::F64);
                        let operation = ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
                            CustomFunctionJvpRule::Explicit,
                            false,
                        ));
                        let regions = vec![sin_program(&scalar_type), doubled_sin_jvp_program(&scalar_type)];
                        Ok(y.context().bind(operation, regions, &[y.clone()])?.remove(0))
                    })
                    .unwrap();
                pullback.apply(seed)
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
                CustomFunctionOperation::from_rule_regions(CustomFunctionJvpRule::Explicit, false),
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
    fn test_custom_function_differentiation_jvp_rule_local_reference_state() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let regions = custom_jvp_regions_with_reference_state(&scalar_type);

        // A custom derivative rule may allocate and use local reference state: the rule is replayed directly when it
        // consumes the active input, so its state executes like any other primitive operation of the identity rule.
        assert_eq!(
            differentiate_at(ArrayIrValue::Array(Array::scalar(1.0f32).unwrap())).jvp(
                ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
                |input| {
                    let operation = ArrayIrOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
                        CustomFunctionJvpRule::Explicit,
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
                let operation = ArrayIrOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
                    CustomFunctionJvpRule::Explicit,
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
    fn test_custom_function_differentiation_jvp_rule_staged_local_reference_state() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let program = custom_function_call_program(
            ArrayIrOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
                CustomFunctionJvpRule::Explicit,
                false,
            )),
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
    fn test_custom_function_differentiation_jvp_rule_nested_local_reference_state() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let input = DifferentiationDual::new(
            ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
            ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
        )
        .unwrap();

        // A rule region may allocate and use local reference state inside a dormant nested rule: the rule is replayed
        // directly (the driver makes recursive differentiation an assertion failure), so its state executes like any
        // other primitive operation and the identity rule yields the identity dual.
        let jvp = nested_custom_function_state_program(&scalar_type, true);
        assert!(jvp.entry_region_ref().contains_effect_in_closure(EffectClass::OrderedState));
        let driver =
            ReferenceRuleDifferentiationDriver { programs: vec![array_ir_identity_program(&scalar_type), jvp] };
        let outputs = ArrayIrCustomFunction::from_rule_regions(CustomFunctionJvpRule::Explicit, false)
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
    fn test_custom_function_differentiation_jvp_rule_plumbing_references() {
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
        let outputs = ArrayIrCustomFunction::from_rule_regions(CustomFunctionJvpRule::Explicit, false)
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
            ArrayIrCustomFunction::from_rule_regions(CustomFunctionJvpRule::Explicit, false)
                .with_non_differentiated_count(2)
                .unwrap()
                .jvp(&context, &aliasing_driver, &aliased),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "input 1 and input 0 bind the same reference allocation",
        ));
        assert_eq!(counter.read(), Ok(Array::scalar(3.0f32).unwrap()));

        // The same reference in the differentiated segment is rejected by the replayed rule as well.
        assert!(matches!(
            ArrayIrCustomFunction::from_rule_regions(CustomFunctionJvpRule::Explicit, false)
                .jvp(&context, &driver, &inputs),
            Err(DifferentiationError::Program(ProgramError::Type(error)))
                if error == TypeError::invalid(
                    "`custom_function` accepts reference inputs only in its leading non-differentiated segment; \
                     move input 0 of type `ref<f32[]>` before the differentiated inputs"
                        .to_string(),
                ),
        ));
    }

    #[test]
    fn test_custom_function_differentiation_jvp_rule_dormant_assertions() {
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
        let program = custom_function_call_program(
            CustomFunctionOperation::from_rule_regions(CustomFunctionJvpRule::Explicit, false),
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
    fn test_custom_function_differentiation_jvp_rule_linearization_fresh_tangent_state() {
        // Repeated calls must neither reuse accumulated state nor access an accumulator consumed by an earlier call.
        check_custom_jvp_linearization_fresh_tangent_state(false, false);
        check_custom_jvp_linearization_fresh_tangent_state(true, false);
    }

    #[test]
    fn test_custom_function_differentiation_jvp_rule_linearization_stateful_scan() {
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
                custom_function_call_program(
                    ScanOperation::new(0, 3),
                    vec![body],
                    vec![vector.clone(); body_inputs.len()],
                )
            })
            .collect::<Vec<_>>();
        let program = custom_function_call_program(
            ArrayIrCustomFunction::from_rule_regions(CustomFunctionJvpRule::Explicit, false),
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
                let %1:f32[3], %2:f32[3] = scan [carry_count=0, length=3, reverse=false] %0 [
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
                let %2:f32[3] = scan [carry_count=0, length=3, reverse=false] %0 %1 [
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
    fn test_custom_function_differentiation_jvp_rule_linearization_nested_aliased_reference_carries() {
        // Exercise whole-root carry updates and indexed updates of a stacked root independently.
        check_custom_jvp_linearization_aliased_reference_carries(false);
        check_custom_jvp_linearization_aliased_reference_carries(true);
    }

    #[test]
    fn test_custom_function_differentiation_jvp_rule_zero_tangent_outputs() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(scalar_type.clone());
        builder.add_input(scalar_type.clone());
        let zero = builder.add_constant(Array::scalar(0.0).unwrap());
        let rule = builder.build(vec![input, zero], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();
        let mut primal = ProgramBuilder::new();
        let input = primal.add_input(scalar_type.clone());
        let primal = primal.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let operation = ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
            CustomFunctionJvpRule::Explicit,
            false,
        ));
        let regions = vec![primal, rule];
        let program = custom_function_call_program(operation.clone(), regions.clone(), vec![scalar_type]);

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
    fn test_custom_function_differentiation_jvp_rule_rejects_known_tangent_outputs() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let operation = ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
            CustomFunctionJvpRule::Explicit,
            false,
        ));

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
        let program = custom_function_call_program(operation.clone(), regions.clone(), vec![scalar_type.clone()]);
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
    fn test_custom_function_differentiation_vjp_rule() {
        // The custom backward rule triples the true gradient, which proves that it governs reverse-mode
        // differentiation.
        let (value, gradient) = differentiate_at(Array::scalar(2.0).unwrap())
            .value_and_gradient(|x| {
                let scalar_type = ArrayType::scalar(DataType::F64);
                let operation = ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
                    CustomFunctionJvpRule::Absent,
                    true,
                ));
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
    fn test_custom_function_differentiation_vjp_rule_second_order() {
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
                                ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
                                    CustomFunctionJvpRule::Absent,
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
                                ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
                                    CustomFunctionJvpRule::Absent,
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
    fn test_custom_function_differentiation_vjp_rule_dead_output_geometry() {
        // The attached identity rules of `(x: f32[n], y: f32[m]) ↦ (x, y)` save no residuals, so when the call's
        // output `x` is dead, only the seed geometry that the carrier captures from the primal outputs names the
        // extent `n` of its zero seed. The backward rule never receives that geometry.
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
                    .add_instruction(DimensionFromScalarOperation::new(variable), Vec::new(), vec![extent], None)
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
                    ArrayIrCustomFunction::from_rule_regions(CustomFunctionJvpRule::Absent, true),
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
                .linearize_shared_for_rule(&[0, 1], DifferentiationRule::JvpForTranspose)
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
                let %4:f32[n], %5:f32[m] = custom_function_transpose [leading_input_count=2, seed_geometry_count=2] %2 %3 %0 %1 [
                    backward={
                        lambda %0:f32[n], %1:f32[m] .
                        in (%0, %1)
                    },
                ]
                in (%5)
            "}
            .trim_end(),
        );
        assert_eq!(
            used.to_string(),
            indoc! {"
                lambda %0:f32[m], %1:dimension<n ∈ [1, 8)>, %2:dimension<m ∈ [1, 8)> .
                let %3:f32[n] = zero [type=f32[n]] %1
                in (%3, %0)
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

        // An effectful backward rule runs even when every seed is dead, so both zero seeds need the seed geometry.
        let (_, dead) = linearize(false);
        assert_eq!(
            dead.to_string(),
            indoc! {"
                lambda %0:f32[n], %1:dimension<n ∈ [1, 8)>, %2:dimension<m ∈ [1, 8)>, %3:dimension<m ∈ [1, 8)> .
                let %4:f32[n] = zero [type=f32[n]] %1
                    %5:f32[m] = zero [type=f32[m]] %2
                    %6:i32[] = const 1
                    %7:dimension<extent ∈ [0, ∞)> = dimension_from_scalar [bounds=[0, ∞)] %6
                    %8:f32[n] = add %0 %4
                in (%8, %5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_function_differentiation_vjp_rule_nonlinear_backward_seed() {
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
                CustomFunctionOperation::from_rule_regions(CustomFunctionJvpRule::Absent, true),
                vec![primal, forward, backward],
                vec![input],
                None,
            )
            .unwrap()[0];
        let custom =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let linearization = custom
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose)
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
    fn test_custom_function_differentiation_vjp_rule_re_transposition() {
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
                CustomFunctionOperation::from_rule_regions(CustomFunctionJvpRule::Absent, true),
                vec![primal, forward, backward],
                vec![input],
                None,
            )
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let linearization = program
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose)
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
    fn test_custom_function_differentiation_vjp_rule_rejects_forward_mode() {
        // A custom VJP supplies no executable tangent rule. Forward mode rejects before the forward rule executes,
        // with the same custom-VJP diagnostic whether evaluating immediately or constructing staged derivatives.
        assert!(matches!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| {
                let scalar_type = ArrayType::scalar(DataType::F64);
                let operation =
                    ArrayOperation::CustomFunction(ArrayCustomFunction::from_rule_regions(CustomFunctionJvpRule::Absent, true));
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
    fn test_custom_function_differentiation_vjp_rule_rejects_reverse_over_forward() {
        // Reverse differentiation of a directional derivative first applies forward mode to the custom VJP call,
        // which is rejected with the same diagnostic as immediate forward mode.
        assert!(matches!(
            differentiate_at(Array::scalar(2.0).unwrap()).vjp(|x| {
                let direction = x.context().lift(Array::scalar(1.0).unwrap())?;
                Ok(differentiate_at(x)
                    .jvp(direction, |y| {
                        let scalar_type = ArrayType::scalar(DataType::F64);
                        let operation = ArrayOperation::CustomFunction(ArrayCustomFunction::from_rule_regions(
                            CustomFunctionJvpRule::Absent,
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
    fn test_custom_function_differentiation_vjp_rule_rejects_forward_before_rule_execution() {
        let function = custom_function(|(counter, input): (ArrayIrTracer, ArrayIrTracer)| {
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
        let rejection = "cannot apply forward-mode differentiation to a `custom_function` call that \
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
            Ok((None, Some(ArrayIrValue::Array(Array::scalar(2.0f32).unwrap())))),
        );
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(ArrayIrValue::Array(Array::scalar(5.0f32).unwrap())),
                (CotangentDestination::Ignore, CotangentDestination::Return),
            ),
            Ok((None, Some(ArrayIrValue::Array(Array::scalar(5.0f32).unwrap())))),
        );
        assert_eq!(counter.read(), Ok(Array::scalar(3.0f32).unwrap()));
    }

    #[test]
    fn test_custom_function_differentiation_vjp_rule_nested_local_reference_state() {
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
        let forward = nested_custom_function_state_program(&scalar_type, false);
        assert!(forward.entry_region_ref().contains_effect_in_closure(EffectClass::OrderedState));
        let program = custom_function_call_program(
            ArrayIrCustomFunction::from_rule_regions(CustomFunctionJvpRule::Absent, true),
            vec![identity.clone(), forward, identity],
            vec![scalar_type],
        );
        let linearization = program
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose)
            .unwrap();
        let mut outputs =
            linearization.primal().interpret(vec![ArrayIrValue::Array(Array::scalar(1.0f32).unwrap())]).unwrap();
        let mut inputs = vec![ArrayIrValue::Array(Array::scalar(2.0f32).unwrap())];
        inputs.extend(outputs.split_off(1));
        assert_eq!(outputs, vec![ArrayIrValue::Array(Array::scalar(1.0f32).unwrap())]);
        assert_eq!(
            linearization.pullback().unwrap().interpret(inputs),
            Ok(vec![ArrayIrValue::Array(Array::scalar(2.0f32).unwrap())]),
        );
    }

    #[test]
    fn test_custom_function_differentiation_vjp_rule_reverse_jacobian() {
        // `jacobian_reverse` interprets the pullback with batch-stacked cotangent bases, which exercises the batched
        // replay of the custom backward program. The Jacobian of elementwise `sin` with the tripled rule is the
        // diagonal matrix `diag(3 * cos(x))`.
        let vector_type = ArrayType::new_static(DataType::F64, [2]);
        let jacobian = differentiate_at(Array::vector(vec![0.5, 1.0]).unwrap())
            .jacobian_reverse(|x| {
                let operation = ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
                    CustomFunctionJvpRule::Absent,
                    true,
                ));
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
    fn test_custom_function_differentiation_retained_rules() {
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
    fn test_custom_function_differentiation_retained_rules_rule_selection() {
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
        let no_rule = "cannot differentiate a `custom_function` call of `cube` that has no derivative rule";
        let forward_rejection = "cannot apply forward-mode differentiation to a `custom_function` call of `cube` \
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
                    let operation =
                        ArrayOperation::CustomFunction(CustomFunctionOperation::new(definition.reference()));
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
                Err(expected) => assert_eq!(result.unwrap_err().to_string(), expected, "{forward:?} {reverse}"),
            }
            let result = crate::differentiation::differentiate_at(Array::scalar(x).unwrap())
                .vjp(|input| {
                    let operation =
                        ArrayOperation::CustomFunction(CustomFunctionOperation::new(definition.reference()));
                    Ok(input
                        .context()
                        .bind(operation, vec![member_cube_program(&scalar_type)], &[input.clone()])?
                        .remove(0))
                })
                .and_then(|(value, pullback)| Ok((value, pullback.apply(Array::scalar(1f64).unwrap())?)));
            match gradient {
                Ok(gradient) => assert_eq!(
                    result,
                    Ok((Array::scalar(x * x * x).unwrap(), Array::scalar(gradient).unwrap())),
                    "{forward:?} {reverse}",
                ),
                Err(expected) => assert_eq!(result.unwrap_err().to_string(), expected, "{forward:?} {reverse}"),
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
                            let operation =
                                ArrayOperation::CustomFunction(CustomFunctionOperation::new(definition.reference()));
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
    fn test_custom_function_differentiation_retained_rules_error_timing() {
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
        assert!(call_vjp(&definition, x.clone(), one.clone()).is_ok());
        assert_eq!(call_jvp(&definition, x.clone(), one.clone()).unwrap_err().to_string(), "rule failure");

        // A JVP rule with the wrong result structure surfaces at the same stage with a type error.
        let definition = CustomRuleRegistration::new(
            TestDefinition::new("cube").with_jvp(|primals, _| Ok((primals.to_vec(), vec![]))),
        );
        assert_eq!(
            call_jvp(&definition, x.clone(), one.clone()).unwrap_err().to_string(),
            "`custom_function` `cube` JVP rule returned 1 outputs and 0 output tangents but the primal has 1 outputs",
        );

        // A failing forward rule surfaces at the first reverse-mode request, while a failing backward rule surfaces
        // only when the first pullback application transposes the carrier.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(
            cube_definition(&counters, true, false).with_vjp(move |_| Err(failure()), |_, _| unreachable!()),
        );
        assert!(call_jvp(&definition, x.clone(), one.clone()).is_ok());
        assert_eq!(call_vjp(&definition, x.clone(), one.clone()).unwrap_err().to_string(), "rule failure");
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, false).with_vjp(
            |primals| Ok((vec![primals[0].clone() * primals[0].clone() * primals[0].clone()], vec![])),
            move |_, _| Err(failure()),
        ));
        assert!(call_jvp(&definition, x.clone(), one.clone()).is_ok());
        let r#type = x.r#type().into_owned();
        let (_, pullback) = TestContext::new()
            .vjp(
                |x, ()| {
                    let operation = CustomFunctionOperation::new(definition.reference());
                    x.context().bind(operation, vec![cube_program(&r#type)], &[x.clone()])
                },
                x,
                (),
            )
            .unwrap();
        assert_eq!(pullback.apply(vec![one]).unwrap_err().to_string(), "rule failure");
    }

    #[test]
    fn test_custom_function_differentiation_retained_rules_rule_result_partitions() {
        // Each rule result partition is validated on its own, so a rule that moves a primal output into its other
        // partition is rejected even though the flattened result types match the expected ones exactly.
        let x = Array::scalar(2f64).unwrap();
        let one = Array::scalar(1f64).unwrap();
        let definition = CustomRuleRegistration::new(TestDefinition::new("cube").with_jvp(|primals, tangents| {
            let square = primals[0].clone() * primals[0].clone();
            Ok((vec![], vec![square.clone() * primals[0].clone(), square * tangents[0].clone()]))
        }));
        assert_eq!(
            call_jvp(&definition, x.clone(), one.clone()).unwrap_err().to_string(),
            "`custom_function` `cube` JVP rule returned 0 outputs and 2 output tangents but the primal has 1 outputs",
        );

        // Correctly partitioned results still have their types validated.
        let definition = CustomRuleRegistration::new(TestDefinition::new("cube").with_jvp(|primals, tangents| {
            let tangent = tangents[0].context().lift(Array::scalar(1f32).unwrap())?;
            Ok((primals.to_vec(), vec![tangent]))
        }));
        assert_eq!(
            call_jvp(&definition, x.clone(), one.clone()).unwrap_err().to_string(),
            "`custom_function` `cube` JVP rule output type signature mismatch: expected [f64[], f64[]] but got \
             [f64[], f32[]]",
        );
        let definition = CustomRuleRegistration::new(TestDefinition::new("cube").with_vjp(
            |primals| Ok((vec![], vec![primals[0].clone() * primals[0].clone() * primals[0].clone()])),
            |_, _| unreachable!(),
        ));
        assert_eq!(
            call_vjp(&definition, x, one).unwrap_err().to_string(),
            "`custom_function` `cube` forward rule returned 0 outputs but the primal has 1",
        );
    }

    #[test]
    fn test_custom_function_differentiation_retained_rules_structural_zero_tangents() {
        // Forward-mode specializations receive only the active input tangents, and each activity pattern is a separate
        // specialization. The pair rule forwards each input tangent to the corresponding output.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let pair_program = |definition: &TestRegistration| {
            let mut primal = ProgramBuilder::<Array, TestArrayOperation>::new();
            let inputs = vec![primal.add_input(scalar_type.clone()), primal.add_input(scalar_type.clone())];
            let primal =
                primal.build::<Vec<Array>, Vec<Array>>(inputs, vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();
            let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
            let primal = builder.import_program(primal);
            let inputs = vec![builder.add_input(scalar_type.clone()), builder.add_input(scalar_type.clone())];
            let outputs = builder
                .add_instruction(CustomFunctionOperation::new(definition.reference()), vec![primal], inputs, None)
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

        // A rule that receives structural zeros sees them as `MaybeZero::Zero` leaves.
        let observed = Arc::new(Mutex::new(Vec::new()));
        let definition = TestRegistration::new(TestDefinition::new("pair").with_symbolic_zero_jvp({
            let observed = observed.clone();
            move |primals, tangents| {
                observed.lock().unwrap().push(tangents.iter().map(MaybeZero::is_zero).collect::<Vec<_>>());
                let tangents = tangents
                    .iter()
                    .zip(primals)
                    .map(|(tangent, primal)| match tangent {
                        MaybeZero::Value(tangent) => Ok(tangent.clone()),
                        MaybeZero::Zero(r#type) => primal.context().zero(r#type),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((primals.to_vec(), tangents))
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
        assert!(program.jvp().is_ok());
        assert!(program.jvp_with_respect_to(&[1]).is_ok());
        assert_eq!(*observed.lock().unwrap(), vec![vec![true, false], vec![false, false]]);
    }

    #[test]
    fn test_custom_function_differentiation_retained_rules_retained_partition() {
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
    fn test_custom_function_differentiation_retained_rules_linearization() {
        // Forward-mode linearization replays the traced JVP rule, while reverse-mode linearization stages the carrier
        // over the forward rule's residual and the input tangent.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, true));
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        let forward = program.linearize().unwrap();
        let reverse = program
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose)
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
                let %2:f64[] = custom_function_transpose [name=\"cube\", leading_input_count=1] %1 %0
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
    fn test_custom_function_differentiation_retained_rules_reference_residuals() {
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
                let operation = ArrayIrOperation::CustomFunction(
                    CustomFunctionOperation::new(definition.reference()).with_non_differentiated_count(1).unwrap(),
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
                Some(ArrayIrValue::Array(Array::scalar(6.0f32).unwrap())),
            )),
        );

        // A residual reference that does not forward a leading input, or that forwards an input already forwarded by an
        // earlier residual, is rejected when the forward rule is first traced.
        let definition = registration(|primals| {
            Ok((vec![ir_multiply(&primals[1], &primals[1])?], vec![primals[1].reference_new()?, primals[1].clone()]))
        });
        assert_eq!(
            vjp(&definition).unwrap_err().to_string(),
            "`custom_function` `scaled` forward rule returns residual 0 of reference type `ref<f32[]>` that is not a \
             leading non-differentiated input forwarded by identity",
        );
        let definition = registration(|primals| {
            Ok((vec![ir_multiply(&primals[1], &primals[1])?], vec![primals[0].clone(), primals[0].clone()]))
        });
        assert_eq!(
            vjp(&definition).unwrap_err().to_string(),
            "`custom_function` `scaled` forward rule returns residual 1 of reference type `ref<f32[]>` from an input \
             already forwarded by an earlier residual",
        );
    }

    #[test]
    fn test_custom_function_differentiation_retained_rules_specialization_after_program_changes() {
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
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose)
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
                    non_differentiated_count: 0,
                    tangent_activity: vec![true],
                    levels: Vec::new(),
                    discharged: false,
                },
                CustomRuleSpecializationKey {
                    input_types: vec![relocated_type.clone()],
                    output_types: vec![relocated_type.clone()],
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
                seed_geometry_count: 0,
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
    fn test_custom_function_differentiation_retained_rules_specialization_output_signature() {
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
            let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
            let input = builder.add_input(r#type.clone());
            builder
                .build(vec![input; output_count], vec![Placeholder], vec![Placeholder; output_count])
                .unwrap()
        };
        let jvp = |primal: FlatProgram<TestContext>| {
            TestContext::new().jvp(
                |x, ()| {
                    x.context().bind(
                        CustomFunctionOperation::new(definition.reference()),
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
                            CustomFunctionOperation::new(definition.reference()),
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
        assert_eq!(
            jvp(primal(2)).unwrap_err().to_string(),
            "`custom_function` `forward_input` JVP rule returned 1 outputs and 1 output tangents but the primal has \
             2 outputs",
        );
        assert_eq!(
            vjp(primal(2)).unwrap_err().to_string(),
            "`custom_function` `forward_input` forward rule returned 1 outputs but the primal has 2",
        );
        assert_eq!(definition.caches().jvp_specializations.keys().len(), 1);
        assert_eq!(definition.caches().forward_specializations.keys().len(), 1);
    }

    #[test]
    fn test_custom_function_transposition_jvp_rule() {
        // Differentiation replaces the call with its replayed rule before transposition, so only a direct transpose of
        // an un-linearized call reaches the operation, which rejects it.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let operation = ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
            CustomFunctionJvpRule::Explicit,
            false,
        ));
        let regions = vec![sin_program(&scalar_type), doubled_sin_jvp_program(&scalar_type)];
        let program = custom_function_call_program(operation, regions, vec![scalar_type]);
        assert!(matches!(
            program.transpose_with_respect_to(&[0], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `custom_function` is not transposable",
        ));
    }

    #[test]
    fn test_custom_function_transposition_vjp_rule() {
        // Differentiation replaces the call with its reverse-mode carrier before transposition, so only a direct
        // transpose of an un-linearized call reaches the operation, which rejects it.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let operation = ArrayOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
            CustomFunctionJvpRule::Absent,
            true,
        ));
        let regions = vec![
            sin_program(&scalar_type),
            sin_forward_program(&scalar_type),
            tripled_sin_backward_program(&scalar_type),
        ];
        let program = custom_function_call_program(operation, regions, vec![scalar_type]);
        assert!(matches!(
            program.transpose_with_respect_to(&[0], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `custom_function` is not transposable",
        ));
    }

    #[test]
    fn test_custom_function_lifting_retained_rules() {
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
                let %1:f64[] = custom_function [name=\"cube\"] %0 [
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
        assert!(matches!(converted.instructions()[0].operation(), ArrayIrOperation::LiftedCustomFunction(_)));
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
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose)
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
            ArrayIrOperation::LiftedCustomFunction(again_operation),
            ArrayIrOperation::LiftedCustomFunction(converted_operation),
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

    #[test]
    fn test_custom_function_transpose() {
        let scalar_type = ArrayType::scalar(DataType::F64);

        // A carrier with an attached backward rule attaches that rule as its only, deferred rule region, whose own
        // effects decide whether transposition must run it (and whether the carrier's application carries deferred
        // work).
        let carrier =
            ArrayCustomFunctionTranspose::from_backward_region(1, vec![scalar_type.clone()], vec![scalar_type.clone()]);
        assert_eq!(carrier.name(), CUSTOM_FUNCTION_TRANSPOSE_OPERATION_NAME);
        assert_eq!(carrier.leading_input_count(), 1);
        assert_eq!(carrier.input_tangent_types(), std::slice::from_ref(&scalar_type));
        assert_eq!(carrier.output_tangent_types(), std::slice::from_ref(&scalar_type));
        assert_eq!(carrier.rules(), None);
        assert_eq!(carrier.region_slots(), &[RegionSlot::deferred_rule("backward")]);
        assert!(!carrier.effects().summary().has_deferred_work());
        assert_eq!(carrier.to_string(), "custom_function_transpose [leading_input_count=1]");

        // A carrier with a retained backward rule attaches no region and carries deferred work until transposition
        // replaces it with the specialized backward program.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, false, true));
        let carrier = CustomFunctionTransposeOperation::new(
            definition.reference(),
            1,
            vec![scalar_type.clone()],
            vec![scalar_type],
        );
        assert_eq!(carrier.rules(), Some(&definition.reference()));
        assert!(carrier.region_slots().is_empty());
        assert!(carrier.effects().summary().has_deferred_work());
        assert_eq!(carrier.to_string(), "custom_function_transpose [name=\"cube\", leading_input_count=1]");
    }

    #[test]
    fn test_custom_function_transpose_type_inference() {
        // The attached backward region receives the leading inputs followed by one cotangent per output and returns one
        // cotangent per differentiated input. The tangent types are differential types rather than primal storage
        // types, which the cotangent mapping must respect.
        let residual_type = ArrayType::scalar(DataType::F64);
        let tangent_type = ArrayType::scalar(DataType::F32);
        let carrier = ArrayCustomFunctionTranspose::from_backward_region(
            1,
            vec![tangent_type.clone()],
            vec![tangent_type.clone()],
        );
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
                "`custom_function_transpose` tangent input type signature mismatch: expected [f32[]] but got [f64[]]",
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
                "`custom_function_transpose` backward rule input type signature mismatch: expected [f64[], f32[]] \
                 but got [f32[], f32[]]",
            )),
        );
    }

    #[test]
    fn test_custom_function_transpose_type_inference_exact_backward_outputs() {
        let scalar = ArrayType::scalar(DataType::F32);
        let vector = ArrayType::new_static(DataType::F32, [2]);
        let exact = DimensionVariable::new("exact", DimensionBounds::new(2, Some(3)).unwrap());
        let broad = DimensionVariable::new("broad", DimensionBounds::new(0, Some(4)).unwrap());
        let other = DimensionVariable::new("other", DimensionBounds::new(0, Some(4)).unwrap());
        let carrier = ArrayCustomFunctionTranspose::from_backward_region(0, vec![vector.clone()], vec![scalar.clone()]);
        let infer = |carrier: &ArrayCustomFunctionTranspose, input_type: &ArrayType, output_type: ArrayType| {
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
                "`custom_function_transpose` backward rule output type signature mismatch: expected [f32[2]] but got \
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
        let carrier = ArrayCustomFunctionTranspose::from_backward_region(0, vec![square.clone()], vec![scalar.clone()]);
        assert_eq!(
            infer(
                &carrier,
                &square,
                ArrayType::new(DataType::F32, Shape::new(vec![other.clone().into(), other.clone().into()]))
            ),
            Err(TypeError::invalid(
                "`custom_function_transpose` backward rule output type signature mismatch: expected \
                 [f32[broad, broad]] but got [f32[other, other]]",
            )),
        );
        assert_eq!(
            infer(&carrier, &square, ArrayType::new(DataType::F32, Shape::new(vec![broad.into(), other.into()]))),
            Err(TypeError::invalid(
                "`custom_function_transpose` backward rule output type signature mismatch: expected \
                 [f32[broad, broad]] but got [f32[broad, other]]",
            )),
        );
    }

    #[test]
    fn test_custom_function_transpose_type_inference_retained_rules() {
        // A carrier takes its known leading inputs followed by the tangents of the differentiated inputs, and it
        // returns the output tangents. It has no regions.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, false, true));
        let scalar_type = ArrayType::scalar(DataType::F64);
        let carrier = CustomFunctionTransposeOperation::new(
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
                "`custom_function_transpose` tangent input type signature mismatch: expected [f64[]] but got [f32[]]",
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
    fn test_custom_function_transpose_interpretation() {
        // A carrier has no forward program, so it supports only reverse-mode differentiation.
        assert_eq!(
            attached_carrier_program().interpret(vec![Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()]),
            Err(ProgramError::UnsupportedOperation {
                message: "`custom_function_transpose` has no forward program to execute; it supports only \
                          reverse-mode differentiation (e.g., `vjp`, `value_and_gradient`, or `jacobian_reverse`)"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_custom_function_transpose_interpretation_retained_rules() {
        // The carrier has no forward program, so ordinary execution and forward mode reject it.
        let stash = ArrayReference::new(Array::scalar(0.0f64).unwrap());
        let context = EagerContext::<ArrayIrValue<Array>, ReferenceTestOperation>::new();
        assert_eq!(
            context
                .bind(
                    ignored_zero_effect_carrier(),
                    Vec::new(),
                    &[ArrayIrValue::Reference(stash.clone()), ArrayIrValue::Array(Array::scalar(3.0f64).unwrap())],
                )
                .unwrap_err()
                .to_string(),
            "`custom_function_transpose` `ignored_zero_effect` has no forward program to execute; it supports only \
             reverse-mode differentiation (e.g., `vjp`, `value_and_gradient`, or `jacobian_reverse`)",
        );
        assert_eq!(stash.read(), Ok(Array::scalar(0.0f64).unwrap()));
    }

    #[test]
    fn test_custom_function_transpose_partial_evaluation() {
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
                let %2:f64[] = custom_function_transpose [leading_input_count=1] %0 %1 [
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
    fn test_custom_function_transpose_partial_evaluation_retained_rules_disconnected() {
        let stash = ArrayReference::new(Array::scalar(0.0f64).unwrap());
        let context = PartialEvaluationContext::new(EagerContext::<ArrayIrValue<Array>, ReferenceTestOperation>::new());
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
    fn test_custom_function_transpose_partial_evaluation_retained_rules_known_computation() {
        let stash = ArrayReference::new(Array::scalar(0.0f64).unwrap());
        let reference_type = ArrayIrValue::Reference(stash.clone()).r#type().into_owned();
        let tangent_type: ArrayIrType = ArrayType::scalar(DataType::F64).into();
        let mut body = ReferenceTestBuilder::new();
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
        let mut builder = ReferenceTestBuilder::new();
        let region = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(reference_type.clone());
        let tangent = builder.add_input(tangent_type.clone());
        builder
            .add_instruction(ReferenceTestOperation::Computation, vec![region], vec![reference, tangent], None)
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
        let parent = TracingContext::<ArrayIrValue<Array>, ReferenceTestOperation>::new();
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
                let () = computation %0 %1 [
                    body={
                        lambda %0:ref<f64[]>, %1:f64[] .
                        let %2:f64[] = custom_function_transpose [name=\"ignored_zero_effect\", leading_input_count=1] %0 %1
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
    fn test_custom_function_transpose_batching() {
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
                let %2:f64[] = custom_function_transpose [leading_input_count=1] %0 %1 [
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
                if message == "a `custom_function_transpose` with an attached backward rule cannot be batched \
                               structurally, because it has no forward program to determine the batch axes of its \
                               outputs; it is preserved unchanged only when every input is replicated at an unnamed \
                               batching level",
        ));
    }

    #[test]
    fn test_custom_function_transpose_batching_retained_rules() {
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
                let %2:f64[3] = custom_function_transpose [
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
    fn test_custom_function_transpose_batching_retained_rules_caller_buffers() {
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
                .with_batching(),
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
    fn test_custom_function_transpose_transposition() {
        let r#type = ArrayType::scalar(DataType::F64);
        let backward = scalar_multiply_program();
        let driver = TestTranspositionDriver { region: backward.entry_region_ref() };
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();

        // Transposition validates the carrier's input split independently of type inference, because a pullback may be
        // built from an imported carrier whose inputs were pruned after inference ran.
        let carrier = ArrayCustomFunctionTranspose::from_backward_region(1, vec![r#type.clone()], vec![r#type.clone()]);
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
                "`custom_function_transpose` leading input 0 is not known during transposition".to_string(),
            ))),
        );
    }

    #[test]
    fn test_custom_function_transpose_transposition_known_input_cotangents() {
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
            ArrayCustomFunctionTranspose::from_backward_region(1, linear_types, vec![r#type.clone()])
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
    fn test_custom_function_transpose_transposition_structural_zero_outputs() {
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
            ArrayCustomFunctionTranspose::from_backward_region(
                0,
                vec![tangent_type.clone()],
                vec![tangent_type.clone()],
            )
            .transpose(&mut rule_context, &driver, rule_inputs, &[MaybeZero::Value(output_cotangent)], &accumulators)
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
            ArrayCustomFunctionTranspose::from_backward_region(0, vec![tangent_type.clone()], vec![tangent_type])
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
    fn test_custom_function_transpose_transposition_unrequested_backward_effects() {
        // The backward rule asserts a dimension bound even though its contribution is unused. Its forward call has
        // no computation region, so ordinary forward-effect summaries cannot keep this assertion alive.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let mut backward = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let seed = backward.add_input(scalar_type.clone());
        let invalid_extent = backward.add_constant(Array::scalar(-1i32).unwrap().into());
        backward
            .add_instruction(
                DimensionFromScalarOperation::new(DimensionVariable::new(
                    "extent",
                    DimensionBounds::new(0, None).unwrap(),
                )),
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
                ArrayIrCustomFunctionTranspose::from_backward_region(0, vec![scalar_type.clone()], vec![scalar_type]),
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
    fn test_custom_function_transpose_transposition_retained_rules() {
        let invocations = Arc::new(AtomicUsize::new(0));
        let definition = slice_at_one_definition(&invocations);
        let retained = Arc::downgrade(&definition.definition());
        let program = slice_at_one_program(&definition, ArrayType::new_static(DataType::F64, [3]));
        assert_eq!(invocations.load(Ordering::SeqCst), 0);

        // One backward rule specializes to a returned cotangent, a caller buffer, and an ignored destination.
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
        assert_eq!(invocations.load(Ordering::SeqCst), 3);
        let repeated = program.clone().transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap();
        assert_eq!(repeated.to_string(), buffered.to_string());
        assert_eq!(invocations.load(Ordering::SeqCst), 3);

        // A wider input is a different carrier signature, so its returned cotangent is rebuilt from its own extent
        // even though its seed type and destination kind are unchanged.
        let wider = slice_at_one_program(&definition, ArrayType::new_static(DataType::F64, [4]));
        let wider_pullback = wider.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Return]).unwrap();
        assert_eq!(
            wider_pullback.interpret(vec![seed]),
            Ok(vec![Array::vector(vec![0.0f64, 3.0, 0.0, 0.0]).unwrap().into()]),
        );
        assert_eq!(invocations.load(Ordering::SeqCst), 4);

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
        assert_eq!(
            symbolic
                .transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference])
                .unwrap_err()
                .to_string(),
            "reference slicing requires a static referent type but got `f64[original]`",
        );
        assert_eq!(
            renamed
                .transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference])
                .unwrap_err()
                .to_string(),
            "reference slicing requires a static referent type but got `f64[relocated]`",
        );
        assert_eq!(invocations.load(Ordering::SeqCst), 6);
        // An ignored sole destination leaves the seed unused, so transposition passes the carrier a structural zero.
        let seed_type: ArrayIrType = ArrayType::new_static(DataType::F64, [1]).into();
        let key = |input_extent: usize, seed_type: Option<ArrayIrType>, destination_kind| {
            CustomRuleBackwardSpecializationKey {
                leading_input_types: Vec::new(),
                seed_geometry_count: 0,
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
                key(3, None, CotangentDestinationKind::Ignore),
                key(3, Some(seed_type), CotangentDestinationKind::Return),
            ],
        );

        // The specializations contain only ordinary operations, so dropping the programs releases the definition.
        drop((definition, program, wider, symbolic, renamed));
        assert!(retained.upgrade().is_none());
    }

    #[test]
    fn test_custom_function_transpose_transposition_retained_rules_disconnected() {
        let stash = ArrayReference::new(Array::scalar(0.0f64).unwrap());
        let mut builder = ReferenceTestBuilder::new();
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
                let %2:f64[] = custom_function_transpose [name=\"ignored_zero_effect\", leading_input_count=1] %0 %1
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
    fn test_custom_function_transpose_transposition_retained_rules_condition() {
        let stash = ArrayReference::new(Array::scalar(0.0f64).unwrap());
        let reference_type = ArrayIrValue::Reference(stash.clone()).r#type().into_owned();
        let tangent_type: ArrayIrType = ArrayType::scalar(DataType::F64).into();
        let mut true_branch = ReferenceTestBuilder::new();
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
        let mut false_branch = ReferenceTestBuilder::new();
        false_branch.add_input(tangent_type.clone());
        false_branch.add_input(reference_type.clone());
        let false_branch = false_branch
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                Vec::new(),
                vec![Placeholder, Placeholder],
                Vec::new(),
            )
            .unwrap();
        let mut builder = ReferenceTestBuilder::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayIrValue::Array(Array::scalar(true).unwrap()).r#type().into_owned());
        let tangent = builder.add_input(tangent_type);
        let reference = builder.add_input(reference_type);
        builder
            .add_instruction(
                ReferenceTestOperation::Condition(ConditionOperation::new()),
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
        let parent = TracingContext::<ArrayIrValue<Array>, ReferenceTestOperation>::new();
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
                        let %2:f64[] = custom_function_transpose [name=\"ignored_zero_effect\", leading_input_count=1] %1 %0
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
    fn test_custom_function_transpose_transposition_retained_rules_in_attached_backward_rule() {
        let stash = ArrayReference::new(Array::scalar(0.0f64).unwrap());
        let reference_type = ArrayIrValue::Reference(stash.clone()).r#type().into_owned();
        let tangent_type: ArrayIrType = ArrayType::scalar(DataType::F64).into();
        let attached_carrier = CustomFunctionTransposeOperation::from_backward_region(
            1,
            vec![tangent_type.clone()],
            vec![tangent_type.clone()],
        );

        // Builds a backward program over `(reference, seed)` that returns its seed after the instructions that
        // `populate` appends.
        let backward = |populate: &dyn Fn(&mut ReferenceTestBuilder, AtomId, AtomId)| {
            let mut builder = ReferenceTestBuilder::new();
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
        let deferred_work = |builder: &mut ReferenceTestBuilder, reference: AtomId, seed: AtomId| {
            builder
                .add_instruction(ignored_zero_effect_carrier(), Vec::new(), vec![reference, seed], None)
                .unwrap();
        };

        // Builds a program over `(reference, tangent)` that applies a carrier with an attached backward rule, with the
        // reference as its leading input, whose backward program is `backward`.
        let attached_carrier_program =
            |backward: Program<ArrayIrValue<Array>, ReferenceTestOperation, Vec<_>, Vec<_>>| {
                let mut builder = ReferenceTestBuilder::new();
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
                let %2:f64[] = custom_function_transpose [name=\"ignored_zero_effect\", leading_input_count=1] %1 %0
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
                let %2:f64[] = custom_function_transpose [leading_input_count=1] %1 %0 [
                    backward={
                        lambda %0:ref<f64[]>, %1:f64[] .
                        let %2:f64[] = custom_function_transpose [name=\"ignored_zero_effect\", leading_input_count=1] %0 %1
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
    fn test_custom_function_transpose_transposition_retained_rules_effectful_buffered_zero_seed() {
        // The backward rule records a reference effect whenever its differentiated input's cotangent goes to a caller
        // buffer, including for a structural-zero seed that contributes nothing to that buffer.
        let definition =
            CustomRuleRegistration::new(ReferenceTestDefinition::new("buffered_zero_effect").with_accumulating_vjp(
                |_| unreachable!("the tests construct the carrier directly"),
                |context, inputs, seeds, accumulators| {
                    if accumulators[1].reference(context)?.is_some() && matches!(&seeds[0], MaybeZero::Zero(_)) {
                        let stash = inputs[0].as_known().unwrap().clone();
                        let increment = context.lift(ArrayIrValue::Array(Array::scalar(1.0f64)?))?;
                        context.bind(
                            ReferenceTestOperation::Base(ReferenceAddUpdateOperation::new().into()),
                            Vec::new(),
                            &[stash, increment],
                        )?;
                    }
                    Ok(())
                },
            ));
        let stash = ArrayReference::new(Array::scalar(0.0f64).unwrap());
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F64).into();
        let mut builder = ReferenceTestBuilder::new();
        let reference = builder.add_input(ArrayIrValue::Reference(stash.clone()).r#type().into_owned());
        let tangent = builder.add_input(scalar_type.clone());
        let carrier = reference_test_carrier(definition.reference(), 1, vec![scalar_type.clone()], vec![scalar_type]);
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
    fn test_custom_function_transpose_transposition_retained_rules_zero_seeds() {
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
            let operation = ArrayIrOperation::CustomFunction(CustomFunctionOperation::new(definition.reference()));
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

        // A rule that receives materialized seeds sees a zero of the dynamically shaped output. No known leading
        // input carries its runtime extent, so the carrier receives that extent as seed geometry, which is read from
        // the primal output and which the rule never sees.
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
                    let %2:f32[n], %3:f32[n] = custom_function_transpose [name=\"duplicate\", leading_input_count=1, \
                    seed_geometry_count=1] %1 %0
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
                    Ok(vec![seed.clone()])
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
    fn test_custom_function_transpose_transposition_retained_rules_after_un_projection() {
        /// Array-member carrier payload retaining a definition declared in the composite family.
        #[derive(Clone, Debug)]
        struct MemberCarrier {
            /// Definition whose backward rule the converted carrier applies.
            rules: CustomRuleReference<ArrayIrValue<Array>, ReferenceTestOperation>,

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

        impl From<MemberCarrier> for ReferenceTestOperation {
            fn from(operation: MemberCarrier) -> Self {
                reference_test_carrier(
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
        // carries an already composite-typed definition; it does not make a Rust closure family-polymorphic.
        let converted = member.into_unprojected::<ArrayIrValue<Array>, ReferenceTestOperation>().unwrap();
        let ReferenceTestOperation::CustomFunctionTranspose(operation) = converted.instructions()[0].operation() else {
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

    #[test]
    fn test_validate_non_differentiated_count() {
        assert_eq!(validate_non_differentiated_count("custom_function", 0, 0), Ok(()));
        assert_eq!(validate_non_differentiated_count("custom_function", 2, 2), Ok(()));
        assert_eq!(
            validate_non_differentiated_count("custom_function", 3, 2),
            Err(TypeError::invalid(
                "`custom_function` non-differentiated input count 3 exceeds input count 2".to_string()
            )),
        );
    }

    #[test]
    fn test_validate_custom_function_reference_boundary() {
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));

        // References are accepted in the leading non-differentiated input segment.
        assert_eq!(
            validate_custom_function_reference_boundary(
                "custom_function",
                1,
                &[reference_type.clone(), scalar_type.clone()],
                std::slice::from_ref(&scalar_type),
            ),
            Ok(()),
        );

        // References are rejected among the differentiated inputs and among the outputs.
        assert_eq!(
            validate_custom_function_reference_boundary(
                "custom_function",
                1,
                &[scalar_type.clone(), reference_type.clone()],
                std::slice::from_ref(&scalar_type),
            ),
            Err(TypeError::invalid(
                "`custom_function` accepts reference inputs only in its leading non-differentiated segment; move input \
                 1 of type `ref<f32[]>` before the differentiated inputs"
                    .to_string(),
            )),
        );
        assert_eq!(
            validate_custom_function_reference_boundary(
                "custom_function",
                1,
                &[reference_type.clone(), scalar_type.clone()],
                &[scalar_type, reference_type],
            ),
            Err(TypeError::invalid(
                "`custom_function` cannot return a reference, but output 1 has type `ref<f32[]>`".to_string(),
            )),
        );
    }

    #[test]
    fn test_validate_custom_function_replay() {
        // Structural zeros and zero-space values require no tangent slot for a non-differentiated input.
        let input =
            DifferentiationDual::new_with_zero_tangent(ArrayIrValue::Array(Array::scalar(3.0f32).unwrap())).unwrap();
        assert_eq!(
            validate_custom_function_replay("custom_function", 1, &EagerArrayIrContext::new(), &[input], &[]),
            Ok(()),
        );
        let input = DifferentiationDual::new(
            ArrayIrValue::Array(Array::from_logical_bytes(ArrayType::scalar(DataType::Token), &[]).unwrap()),
            ArrayIrValue::Array(Array::from_logical_bytes(ArrayType::scalar(DataType::Zero), &[]).unwrap()),
        )
        .unwrap();
        assert_eq!(
            validate_custom_function_replay("custom_function", 1, &EagerArrayIrContext::new(), &[input], &[]),
            Ok(()),
        );

        // A non-differentiated numeric input with a non-zero tangent has no tangent slot in the rule.
        let input = DifferentiationDual::new(
            ArrayIrValue::Array(Array::scalar(3.0f32).unwrap()),
            ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
        )
        .unwrap();
        assert!(matches!(
            validate_custom_function_replay("custom_function", 1, &EagerArrayIrContext::new(), &[input], &[]),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "`custom_function` cannot propagate the non-zero tangent of type `f32[]` supplied for \
                    one of its 1 leading non-differentiated inputs, because its rule has no tangent slot for them",
        ));
    }

    #[test]
    fn test_validate_custom_function_replay_staged_references() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        EagerArrayIrContext::trace(
            |inputs: Vec<_>| {
                let context = inputs[0].context();
                let mut duals = inputs
                    .iter()
                    .cloned()
                    .map(DifferentiationDual::new_with_zero_tangent)
                    .collect::<Result<Vec<_>, _>>()?;
                assert_eq!(validate_custom_function_replay("custom_function", 2, context, &duals, &[]), Ok(()));

                // Replaying the same valid boundary is allowed, including a fresh root allocated inside the trace.
                let local = context.bind(ReferenceNewOperation::new(), vec![], &inputs[2..])?.remove(0);
                duals[1] = DifferentiationDual::new_with_zero_tangent(local)?;
                assert_eq!(validate_custom_function_replay("custom_function", 2, context, &duals, &[]), Ok(()));
                assert_eq!(validate_custom_function_replay("custom_function", 2, context, &duals, &[]), Ok(()));

                // An ordinary staged input must not hide two reference inputs naming the same allocation.
                duals[1] = duals[0].clone();
                assert!(matches!(
                    validate_custom_function_replay("custom_function", 2, context, &duals, &[]),
                    Err(ProgramError::InvalidArgument { message })
                        if message == "input 1 and input 0 bind the same reference allocation",
                ));
                Ok(vec![inputs[2].clone()])
            },
            vec![reference_type.clone(), reference_type, scalar_type],
        )
        .unwrap();
    }
}
