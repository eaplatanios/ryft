//! Rule sources for registered ([`CustomRuleReference`]), unavailable ([`UnavailableCustomRules`]), and lifted
//! ([`LiftedCustomRules`]) rule sets, which composite calls and carriers inspect through the sealed [`CustomRuleSource`]
//! interface, together with the internal specialization machinery that traces, batches, and transposes retained rules
//! into cached programs.

use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Weak};

use crate::arrays::ShardingDimension;
use crate::axes::Axis;
use crate::batching::{BatchAxis, BatchingError, BatchingLevel, BatchingLevelExtent};
use crate::contexts::StagingContext;
use crate::differentiation::{
    CotangentDestinationKind, DifferentiableType, DifferentiationError, ResidualZeroProvider,
    TransposedProgramCotangent,
};
use crate::operations::composites::derive_bijective_identity_renaming;
use crate::operations::composites::describe_composite;
use crate::operations::composites::operations::{
    COMPOSITE_OPERATION_NAME, COMPOSITE_TRANSPOSE_OPERATION_NAME, CompositeTransposeOperation,
};
use crate::operations::composites::rules::definitions::{
    CustomRuleBroadcastSensitivity, CustomRuleDefinition, CustomRuleJvp, CustomRuleJvpRule, CustomRuleVjp,
};
use crate::operations::composites::rules::registrations::CustomRuleReference;
use crate::operations::composites::rules::{
    CustomRuleBackwardSpecializationKey, CustomRuleBatchedOutputs, CustomRuleBatchingLevel,
    CustomRuleBatchingSpecializationKey, CustomRuleDerivationKind, CustomRuleDerivedDefinitionKey,
    CustomRuleForwardSpecializationKey, CustomRuleSpecialization, CustomRuleSpecializationRequest, CustomRuleTracer,
};
use crate::parameters::{Parameter, Placeholder};
use crate::programs::{
    MaybeZero, Operation, Program, ProgramBuilder, ProgramError, Type, TypeError, Typed, Value, ValueProjection,
};
use crate::tracing::TracingContext;

#[cfg(doc)]
use crate::batching::DiagonalBatchingPolicy;

/// Derivation of a definition from a source definition, which is stored in the derived definition as its batching rule.
pub(super) struct CustomRuleDerivation<V: Typed + Parameter, O> {
    /// Reference to the source [`CustomRuleDefinition`] whose batching rule is differentiated for this derivation.
    source: CustomRuleReference<V, O>,

    /// Key of this derivation.
    key: CustomRuleDerivedDefinitionKey,
}

/// Seals [`CustomRuleSource`] to the source representations supported by composites.
mod sealed {
    /// Marker for the supported native and lifted rule source representations.
    pub trait Sealed {}
}

impl<V: Typed + Parameter, O> sealed::Sealed for CustomRuleReference<V, O> {}

/// Source of the retained rules that a [`CompositeOperation`](crate::CompositeOperation) and its
/// [`CompositeTransposeOperation`] apply, which provides their specializations as programs of the operations'
/// own family `(V, O)`. [`CustomRuleReference`] is the source of a definition registered in that family, and
/// [`LiftedCustomRules`] is the source of a call that was converted into that family from a member family (e.g., from
/// array programs into array IR programs), whose specializations the member definition produces and caches before they
/// are converted. Equality and hashing use the identity of the source's definition.
///
/// This trait exposes metadata without requiring the bounds needed to produce specialized rule programs. The internal
/// specialization interface carries the heavier bounds of converted sources (e.g., on the member family's operations).
/// Keeping those bounds separate lets the operations implement [`Operation`] without introducing a cycle with the
/// payload bounds of the families that contain them.
///
/// This trait is sealed and implemented only by [`CustomRuleReference`] and [`LiftedCustomRules`]. New kinds of rules
/// are registered through [`CustomRuleDefinition`].
pub trait CustomRuleSource<V: Typed<Type: Eq + Hash> + Parameter, O>:
    sealed::Sealed + Clone + Debug + Eq + Hash
{
    /// Returns the process-unique identity of the source's definition.
    fn id(&self) -> u64;

    /// Returns the human-readable label of the source's definition.
    fn name(&self) -> &str;

    /// Returns whether the source has a retained forward mode differentiation
    /// (i.e., Jacobian-Vector Product or JVP) rule.
    fn has_jvp(&self) -> bool;

    /// Returns whether the source derives its forward mode differentiation rule from the primal region of each call.
    fn derives_jvp_from_primal(&self) -> bool;

    /// Returns whether the source has reverse mode differentiation (i.e., Vector-Jacobian Product or VJP)
    /// forward and backward rules.
    fn has_vjp(&self) -> bool;

    /// Returns whether the source has reverse mode differentiation rules whose forward rule reverse mode defers
    /// to the backward pass.
    fn defers_forward(&self) -> bool;

    /// Returns the declared batch axes of the outputs of batched calls, or [`None`] when they are not declared.
    fn batched_output_axes(&self) -> Option<&[BatchAxis]>;

    /// Returns whether the source has a custom batching rule, which takes precedence over structurally batching
    /// the primal region of its calls.
    fn has_batching_rule(&self) -> bool;

    /// Returns whether the source batches its calls sequentially when it has no custom batching rule.
    fn has_sequential_batching(&self) -> bool;

    /// Returns the definition of this source if it was registered in the source's own family `(V, O)`, and [`None`] for
    /// a [`LiftedCustomRules`] source, whose definition belongs to a member family. Only a native definition's rules
    /// can be applied directly to values of this family (e.g., to invoke a backward rule while transposing).
    fn native_definition(&self) -> Option<&CustomRuleDefinition<V, O>>;

    /// Returns whether backward specializations of the source can write cotangents to caller-provided buffers. A
    /// source that cannot do that returns those cotangents instead, and the carrier's accumulators add them to the
    /// buffers.
    fn supports_reference_destinations(&self) -> bool;
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> CustomRuleSource<V, O> for CustomRuleReference<V, O> {
    #[inline]
    fn id(&self) -> u64 {
        self.definition().id()
    }

    #[inline]
    fn name(&self) -> &str {
        self.definition().name()
    }

    #[inline]
    fn has_jvp(&self) -> bool {
        matches!(self.definition().jvp(), Some(CustomRuleJvp::Rule(_) | CustomRuleJvp::SymbolicZeroRule(_)))
    }

    #[inline]
    fn derives_jvp_from_primal(&self) -> bool {
        matches!(self.definition().jvp(), Some(CustomRuleJvp::Primal))
    }

    #[inline]
    fn has_vjp(&self) -> bool {
        self.definition().vjp().is_some()
    }

    #[inline]
    fn defers_forward(&self) -> bool {
        self.definition().deferred_forward() && self.has_vjp()
    }

    #[inline]
    fn batched_output_axes(&self) -> Option<&[BatchAxis]> {
        self.definition().batched_output_axes()
    }

    #[inline]
    fn has_batching_rule(&self) -> bool {
        self.definition().batching_rule().is_some()
    }

    #[inline]
    fn has_sequential_batching(&self) -> bool {
        self.definition().sequential_batching_scan().is_some()
    }

    #[inline]
    fn native_definition(&self) -> Option<&CustomRuleDefinition<V, O>> {
        Some(self.definition())
    }

    #[inline]
    fn supports_reference_destinations(&self) -> bool {
        true
    }
}

/// [`CustomRuleSource`] that produces the specializations of its rules, as programs of the operations' own family
/// `(V, O)`. The functions that produce specializations carry their bounds themselves rather than on the trait, like
/// the rule traits, which keeps the operations free of bounds on `O` that would form a cycle with the payload bounds
/// of the families that contain them. This interface is internal because its specialization keys and results describe
/// composite implementation details.
pub(in crate::operations::composites) trait CustomRuleSpecializer<V: Typed<Type: Eq + Hash> + Parameter, O>:
    CustomRuleSource<V, O>
{
    /// Returns the forward mode differentiation rule specialization for `key`, whose program takes the call's inputs
    /// followed by the active tangents of its differentiated inputs (refer to the tangent activity of the key) and
    /// returns the call's outputs followed by their tangents.
    fn jvp_specialization(
        &self,
        key: CustomRuleForwardSpecializationKey<V::Type>,
    ) -> Result<Arc<CustomRuleSpecialization<V, O>>, DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type> + ResidualZeroProvider<V::Type, Operation = O>,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>;

    /// Returns the reverse mode differentiation forward rule specialization for `key`, whose program takes the call's
    /// inputs and returns the call's outputs followed by the residuals.
    fn forward_specialization(
        &self,
        key: CustomRuleForwardSpecializationKey<V::Type>,
    ) -> Result<Arc<CustomRuleSpecialization<V, O>>, DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type>,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>;

    /// Returns the destination-specialized backward differentiation rule for `key`, whose program consumes the boundary
    /// inputs, then the live seeds, then the caller buffers, then the remaining known leading inputs, and returns the
    /// cotangents of the differentiated inputs with returned destinations.
    fn backward_specialization(
        &self,
        key: CustomRuleBackwardSpecializationKey<V::Type>,
    ) -> Result<Arc<CustomRuleSpecialization<V, O>>, DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type> + From<CompositeTransposeOperation<V, O>>,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>;

    /// Returns the batched primal program that the custom batching rule produces for `key`, whose program takes the
    /// level's boundary inputs followed by the call's batched inputs and returns the batched outputs, together with the
    /// batch axes of those outputs.
    fn batching_rule_specialization(
        &self,
        key: CustomRuleBatchingSpecializationKey<V::Type>,
    ) -> Result<Arc<CustomRuleSpecialization<V, O>>, DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type>,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>;

    /// Returns the scan operation with which a call of this source loops over `length` batch items when it is batched
    /// sequentially (refer to [`CustomRuleDefinition::with_sequential_batching`]), carrying `carry_count` replicated
    /// inputs and stacking `stacked_output_count` outputs along a batch axis placed by `axis_sharding`, together with
    /// the type of the slice index that the scan body receives first.
    fn sequential_batching_scan(
        &self,
        length: usize,
        carry_count: usize,
        axis_sharding: &ShardingDimension,
        stacked_output_count: usize,
    ) -> Result<(O, V::Type), DifferentiationError>;

    /// Returns the rules of the definition derived from this source for `key`, with which forward mode differentiation
    /// stages the derived call of a call whose forward mode differentiation rule is derived from its primal and whose
    /// source has a custom batching rule or batches its calls sequentially. This keeps that rule, or the sequential
    /// batching, on the path of forward mode derivatives that are batched after they are taken (i.e., the analogue of
    /// JAX's `custom_vmap_jvp`): the derived call's primal region is the derivative of the call's primal region (i.e.,
    /// its Jacobian-Vector Product (JVP), or the tangent half of it), and the derived definition holds no program. It
    /// derives its forward mode differentiation rule from each call's primal region, so that higher orders recurse
    /// through each call's own region, and its batching rule applies the JVP of the source's batching rule, while the
    /// derived definition of a source that batches its calls sequentially batches its own calls sequentially (i.e., it
    /// loops over the derivative one batch item at a time). Calls whose primal regions differ therefore share one
    /// derived definition.
    ///
    /// The derived batching rule always computes the derived call's primal outputs with the source rule applied at the
    /// batch axes of the primal inputs, which is exactly how batching the source call computes them, so a rule whose
    /// result depends on its input axes gives a derivative whose primal outputs are those of the batched call. Its
    /// tangents are computed in one of two modes:
    ///
    ///   - **Structural:** When every primal input is replicated, and so is every output of the rule applied to them
    ///     (e.g., the replicated primals and mapped tangents of a forward mode Jacobian), the JVP of the rule's program
    ///     is batched structurally over the given batch axes.
    ///   - **Rule:** Otherwise, each primal and tangent pair with a mapped side is aligned to one axis (i.e., its
    ///     replicated side is broadcast, and a tangent mapped along another axis than its primal is moved), and the JVP
    ///     of the source rule applied at the aligned batch axes computes the tangents. Inputs without a tangent keep
    ///     their batch axes. Alignment changes the rule's input axes only for a replicated primal whose tangent is
    ///     mapped, because the rule cannot apply per-item perturbations of a shared input otherwise; only then is the
    ///     rule applied twice (i.e., once at the primal axes for the primal outputs and once at the aligned axes for
    ///     the tangents). When the source's rule depends on its input batch axes (refer to
    ///     [`CustomRuleDefinition::with_broadcast_sensitive_batching_rule`]), the rule is instead applied once, at the
    ///     primal axes, and the tangents of such primals are recovered exactly from an outer product (refer to
    ///     [`DiagonalBatchingPolicy`]).
    ///
    /// Derived definitions are registered by the source's registration, which retains a bounded number of derived
    /// registrations like its specializations. Every request for the same key returns the same derived definition
    /// while that definition is alive (i.e., while its registration or a call holds it), whether or not the source's
    /// registration is alive, and a fresh one otherwise.
    fn derived(&self, key: CustomRuleDerivedDefinitionKey) -> Result<Self, DifferentiationError>
    where
        Self: Sized;
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> CustomRuleSpecializer<V, O> for CustomRuleReference<V, O> {
    fn jvp_specialization(
        &self,
        key: CustomRuleForwardSpecializationKey<V::Type>,
    ) -> Result<Arc<CustomRuleSpecialization<V, O>>, DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type> + ResidualZeroProvider<V::Type, Operation = O>,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    {
        // Returns the JVP rule specialization for `key`, whose program takes the call's inputs followed by the active
        // tangents of its differentiated inputs and returns the call's outputs followed by their tangents. An unbatched
        // key traces the rule on its first request, while a batched key batches the specialization of its inner key at
        // its outermost level, so that every batching level shares one trace of the rule.

        // The traced rule is validated against the unbatched output types of `key`.
        let rule = match self.definition().jvp() {
            Some(CustomRuleJvp::Rule(rule)) => CustomRuleJvpRule::Materialized(rule),
            Some(CustomRuleJvp::SymbolicZeroRule(rule)) => CustomRuleJvpRule::SymbolicZeros(rule),
            _ => {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "{} has no forward-mode (i.e., JVP) rule",
                        describe_composite(COMPOSITE_OPERATION_NAME, self.name())
                    ),
                }
                .into());
            }
        };

        // The inner specialization is requested only when the outer one is not cached.
        self.specialization(CustomRuleSpecializationRequest::Jvp(key.clone()), || {
            let inner = match key.levels.split_last() {
                Some((level, inner_levels)) => Some((
                    self.jvp_specialization(CustomRuleForwardSpecializationKey {
                        levels: inner_levels.to_vec(),
                        ..key.clone()
                    })?,
                    level.clone(),
                )),
                None => None,
            };

            let Some((inner, level)) = inner else {
                // The traced program takes every primal input followed by the active tangents of the differentiated
                // inputs and returns the primal outputs followed by their tangents, which is exactly the rule program
                // that `replay_custom_jvp_rule` replays. The structural zeros are materialized inside the program, from
                // their own primals (which name every runtime quantity that their tangent types omit), so that the
                // zeros that the rule propagates remain recognizable as zeros when the program is replayed.
                let differentiated_input_types = &key.input_types[key.non_differentiated_count..];
                if key.tangent_activity.len() != differentiated_input_types.len() {
                    return Err(ProgramError::MalformedProgram(format!(
                        "{} JVP specialization key has {} tangent activity \
                         flags for {} differentiated inputs",
                        describe_composite(COMPOSITE_OPERATION_NAME, self.name()),
                        key.tangent_activity.len(),
                        differentiated_input_types.len(),
                    ))
                    .into());
                }

                let mut rule_input_types = key.input_types.clone();
                for (r#type, active) in differentiated_input_types.iter().zip(&key.tangent_activity) {
                    if *active {
                        rule_input_types.push(r#type.tangent()?);
                    }
                }

                // The rule's two result partitions are validated separately before they are flattened,
                // because the flattened types alone cannot tell a missing output from a misplaced tangent.
                let output_count = key.rule_output_count()?;
                let (_, program) = TracingContext::<V, O>::trace_with_named_axes(
                    |values: Vec<CustomRuleTracer<V, O>>| {
                        let (primals, active_tangents) = values.split_at(key.input_types.len());
                        let mut active_tangents = active_tangents.iter().cloned();
                        let tangents = primals[key.non_differentiated_count..]
                            .iter()
                            .zip(&key.tangent_activity)
                            .map(|(primal, active)| match active {
                                true => Ok(MaybeZero::Value(active_tangents.next().unwrap())),
                                false => Ok(MaybeZero::Zero(primal.r#type().tangent()?)),
                            })
                            .collect::<Result<Vec<_>, ProgramError>>()?;

                        let (mut outputs, output_tangents) = match rule {
                            CustomRuleJvpRule::Materialized(rule) => {
                                let tangents = primals[key.non_differentiated_count..]
                                    .iter()
                                    .zip(tangents)
                                    .map(|(primal, tangent)| {
                                        O::materialize_zero_from_residual_sources(
                                            primal.context(),
                                            tangent,
                                            std::iter::once(primal),
                                        )
                                    })
                                    .collect::<Result<Vec<_>, _>>()?;
                                let (outputs, output_tangents) = rule.apply(primals, &tangents)?;
                                (outputs, output_tangents.into_iter().map(MaybeZero::Value).collect::<Vec<_>>())
                            }
                            CustomRuleJvpRule::SymbolicZeros(rule) => rule.apply(primals, &tangents)?,
                        };

                        if outputs.len() != output_count || output_tangents.len() != output_count {
                            return Err(TypeError::invalid(format!(
                                "{} JVP rule returned {} outputs and {} output tangents but the primal has {} outputs",
                                describe_composite(COMPOSITE_OPERATION_NAME, self.name()),
                                outputs.len(),
                                output_tangents.len(),
                                output_count,
                            ))
                            .into());
                        }

                        // A structural-zero output tangent is materialized here, at the boundary of the program, from
                        // its own output and the primal inputs (which name every runtime extent of its type), rather
                        // than inside the rule. The zero-producing operation that this stages lets the replay of the
                        // program recover the structural zero (refer to `replay_custom_jvp_rule`).
                        let output_tangents = output_tangents
                            .into_iter()
                            .zip(&outputs)
                            .map(|(tangent, output)| {
                                O::materialize_zero_from_residual_sources(
                                    output.context(),
                                    tangent,
                                    std::iter::once(output).chain(primals),
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        outputs.extend(output_tangents);
                        Ok(outputs)
                    },
                    rule_input_types,
                    self.definition().named_axes().to_vec(),
                )?;

                // A call that keeps only some of the rule's outputs receives only those outputs and their tangents,
                // whose computation is all that remains of the program.
                let program = match key.kept_outputs {
                    None => program,
                    Some(_) => {
                        let kept_output_indices = key.kept_output_indices(output_count);
                        let selected = kept_output_indices
                            .iter()
                            .copied()
                            .chain(kept_output_indices.iter().map(|index| output_count + index))
                            .collect::<Vec<_>>();
                        program.with_outputs(&selected)?
                    }
                };

                let mut expected_output_types = key.output_types.clone();
                for r#type in &key.output_types {
                    expected_output_types.push(r#type.tangent()?);
                }

                let program =
                    align_rule_output_type_identities(self.name(), "JVP", program, &expected_output_types, None)?;
                let program = self.definition().discharge_if(key.discharged, program)?;
                return Ok(Arc::new(CustomRuleSpecialization::new(program, Vec::new())));
            };

            // Each active tangent follows the batch axis of its primal, and each output tangent follows its output.
            let differentiated_input_start = key.non_differentiated_count
                + key.levels[..key.levels.len() - 1].iter().map(|level| level.boundary_input_count).sum::<usize>();
            let differentiated_input_axes = level.input_axes.get(differentiated_input_start..).ok_or_else(|| {
                let description = describe_composite(COMPOSITE_OPERATION_NAME, self.name());
                ProgramError::MalformedProgram(format!("{description} batching level has too few input axes"))
            })?;

            let mut input_axes = level.input_axes.clone();
            input_axes.extend(
                differentiated_input_axes
                    .iter()
                    .zip(&key.tangent_activity)
                    .filter_map(|(axis, active)| active.then_some(*axis)),
            );
            let output_axes = level.output_axes.iter().chain(&level.output_axes).copied().map(Some).collect::<Vec<_>>();
            self.definition().batched_specialization(
                &inner.program,
                &input_axes,
                &output_axes,
                &level,
                CustomRuleBatchedOutputs::Values,
                Vec::new(),
            )
        })
    }

    fn forward_specialization(
        &self,
        key: CustomRuleForwardSpecializationKey<V::Type>,
    ) -> Result<Arc<CustomRuleSpecialization<V, O>>, DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type>,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    {
        // Returns the reverse mode forward rule specialization for `key`, whose program takes the call's inputs
        // and returns the call's outputs followed by the residuals. Specializations are traced and batched as for
        // `Self::jvp_specialization`, except that residuals keep the batch axes that batching naturally produces,
        // which the specialization records.

        // The traced rule is validated against the unbatched output types of `key`. A call that keeps only some of
        // the rule's outputs still receives every output of the rule, because the backward rule receives a seed for
        // each of them, but only the kept outputs are validated against the call's output types.
        let Some(forward) = self.definition().vjp().map(|vjp| vjp.forward()) else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "{} has no reverse-mode forward rule",
                    describe_composite(COMPOSITE_OPERATION_NAME, self.name())
                ),
            }
            .into());
        };

        // The inner specialization is requested only when the outer one is not cached.
        let output_count = key.rule_output_count()?;
        self.specialization(CustomRuleSpecializationRequest::Forward(key.clone()), || {
            let inner = match key.levels.split_last() {
                Some((level, inner_levels)) => Some((
                    self.forward_specialization(CustomRuleForwardSpecializationKey {
                        levels: inner_levels.to_vec(),
                        ..key.clone()
                    })?,
                    level.clone(),
                )),
                None => None,
            };

            let Some((inner, level)) = inner else {
                // The primal outputs are validated as their own partition before the residuals are appended,
                // because a residual whose type matches a missing output would otherwise pass as that output.
                let (_, program) = TracingContext::<V, O>::trace_with_named_axes(
                    |values: Vec<CustomRuleTracer<V, O>>| {
                        let (mut outputs, residuals) = forward.apply(&values)?;
                        if outputs.len() != output_count {
                            return Err(TypeError::invalid(format!(
                                "{} forward rule returned {} outputs but the primal has {}",
                                describe_composite(COMPOSITE_OPERATION_NAME, self.name()),
                                outputs.len(),
                                output_count,
                            ))
                            .into());
                        }
                        outputs.extend(residuals);
                        Ok(outputs)
                    },
                    key.input_types.clone(),
                    self.definition().named_axes().to_vec(),
                )?;

                let kept_output_indices = key.kept_output_indices(output_count);
                let program = align_rule_output_type_identities(
                    self.name(),
                    "forward",
                    program,
                    &key.output_types,
                    key.kept_outputs.as_ref().map(|_| kept_output_indices.as_slice()),
                )?;

                // Reference residuals preserve handles, not snapshots, so each one must forward a distinct leading
                // non-differentiated input by identity, which the backward rule then receives as plumbing.
                let leading_inputs = &program.input_ids()[..key.non_differentiated_count];
                let mut forwarded_inputs = HashSet::new();
                for (index, residual) in program.output_ids().iter().skip(output_count).enumerate() {
                    let r#type = program.atoms()[residual.index()].r#type();
                    if !r#type.is_reference() {
                        continue;
                    }

                    if !leading_inputs.contains(residual) {
                        return Err(TypeError::invalid(format!(
                            "{} forward rule returns residual {} of reference type `{}` that is not a leading \
                             non-differentiated input forwarded by identity",
                            describe_composite(COMPOSITE_OPERATION_NAME, self.name()),
                            index,
                            r#type,
                        ))
                        .into());
                    }

                    if !forwarded_inputs.insert(*residual) {
                        return Err(TypeError::invalid(format!(
                            "{} forward rule returns residual {} of reference type `{}` \
                             from an input already forwarded by an earlier residual",
                            describe_composite(COMPOSITE_OPERATION_NAME, self.name()),
                            index,
                            r#type,
                        ))
                        .into());
                    }
                }

                let program = self.definition().discharge_if(key.discharged, program)?;
                return Ok(Arc::new(CustomRuleSpecialization::new(program, Vec::new())));
            };

            // The outputs that the call does not keep, like the residuals, keep their natural batch axes.
            let residual_count = inner.program.output_types().len() - output_count;
            let mut kept_output_axes = level.output_axes.iter().copied();
            let output_axes = (0..output_count)
                .map(|index| {
                    if key.kept_outputs.as_ref().is_none_or(|kept_outputs| kept_outputs[index]) {
                        kept_output_axes.next()
                    } else {
                        None
                    }
                })
                .chain(std::iter::repeat_n(None, residual_count))
                .collect::<Vec<_>>();
            self.definition().batched_specialization(
                &inner.program,
                &level.input_axes,
                &output_axes,
                &level,
                CustomRuleBatchedOutputs::Values,
                Vec::new(),
            )
        })
    }

    fn backward_specialization(
        &self,
        key: CustomRuleBackwardSpecializationKey<V::Type>,
    ) -> Result<Arc<CustomRuleSpecialization<V, O>>, DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type> + From<CompositeTransposeOperation<V, O>>,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    {
        // Returns the backward rule specialization for `key`. An unbatched key transposes a source program that applies
        // a directly invoking `CompositeTransposeOperation` and returns only its live-seed outputs, so transposition
        // hands the rule a structural zero for every dropped output. A batched key batches the specialization of its
        // inner key at its outermost level.
        let Some(CustomRuleVjp { transposer, .. }) = self.definition().vjp() else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "{} has no backward rule",
                    describe_composite(COMPOSITE_TRANSPOSE_OPERATION_NAME, self.name())
                ),
            }
            .into());
        };

        // The inner specialization is requested only when the outer one is not cached.
        self.specialization(CustomRuleSpecializationRequest::Backward(key.clone()), || {
            let inner = match key.levels.split_last() {
                Some((level, inner_levels)) => Some((
                    self.backward_specialization(CustomRuleBackwardSpecializationKey {
                        levels: inner_levels.to_vec(),
                        ..key.clone()
                    })?,
                    level.clone(),
                )),
                None => None,
            };
            let leading_input_count = key.leading_input_types.len();
            let input_count = leading_input_count + key.input_tangent_types.len();
            let Some((inner, level)) = inner else {
                let source_carrier = CompositeTransposeOperation::direct(
                    self.clone(),
                    leading_input_count,
                    key.seed_extent_count,
                    key.input_tangent_types.clone(),
                    key.output_tangent_types.clone(),
                    key.discharged,
                );
                let mut builder = ProgramBuilder::<V, O>::new();
                let source_inputs = key
                    .leading_input_types
                    .iter()
                    .chain(&key.input_tangent_types)
                    .map(|r#type| builder.add_input(r#type.clone()))
                    .collect::<Vec<_>>();
                let source_outputs = builder.add_instruction(source_carrier, Vec::new(), source_inputs, None)?;
                let live_outputs = source_outputs
                    .iter()
                    .zip(&key.seed_types)
                    .filter_map(|(output, seed_type)| seed_type.as_ref().map(|_| *output))
                    .collect::<Vec<_>>();
                let live_output_count = live_outputs.len();
                let source = builder.build::<Vec<V>, Vec<V>>(
                    live_outputs,
                    vec![Placeholder; input_count],
                    vec![Placeholder; live_output_count],
                )?;
                let linear_inputs = (leading_input_count..input_count).collect::<Vec<_>>();
                let (program, cotangents) = transposer(&source, &linear_inputs, &key.destination_kinds)?;
                let program = self.definition().discharge_if(key.discharged, program)?;
                return Ok(Arc::new(CustomRuleSpecialization { program, output_axes: Vec::new(), cotangents }));
            };

            // The inner program consumes its inner boundary inputs, then the live seeds, then the caller buffers, then
            // the remaining leading inputs. Each seed follows its output, each caller buffer follows its input, and
            // each leading input keeps its recorded axis. Each returned cotangent is aligned to its input, while
            // structural-zero cotangents have no output and stay structural zeros at every level.
            let inner_boundary_input_count =
                key.levels[..key.levels.len() - 1].iter().map(|level| level.boundary_input_count).sum::<usize>();
            let (leading_axes, tangent_axes) =
                level.input_axes.split_at(inner_boundary_input_count + leading_input_count);
            let mut input_axes = leading_axes[..inner_boundary_input_count].to_vec();
            input_axes.extend(
                level
                    .output_axes
                    .iter()
                    .zip(&key.seed_types)
                    .filter_map(|(axis, seed_type)| seed_type.as_ref().map(|_| *axis)),
            );
            input_axes.extend(
                tangent_axes
                    .iter()
                    .zip(&key.destination_kinds)
                    .filter_map(|(axis, kind)| (*kind == CotangentDestinationKind::Reference).then_some(*axis)),
            );
            input_axes.extend_from_slice(&leading_axes[inner_boundary_input_count..]);
            let output_axes = tangent_axes
                .iter()
                .zip(&inner.cotangents)
                .filter_map(|(axis, cotangent)| {
                    (*cotangent == TransposedProgramCotangent::Output).then_some(Some(*axis))
                })
                .collect::<Vec<_>>();
            self.definition().batched_specialization(
                &inner.program,
                &input_axes,
                &output_axes,
                &level,
                CustomRuleBatchedOutputs::Cotangents,
                inner.cotangents.clone(),
            )
        })
    }

    fn batching_rule_specialization(
        &self,
        key: CustomRuleBatchingSpecializationKey<V::Type>,
    ) -> Result<Arc<CustomRuleSpecialization<V, O>>, DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type>,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    {
        // Traces the custom batching rule over the level's boundary inputs followed by the batched inputs, recording
        // the batch axes that it declares for its outputs. A derived definition applies the JVP of its source's rule
        // instead. The program becomes the primal region of the batched call, so it is discharged with the call.
        let Some(batching_rule) = self.definition().batching_rule() else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "{} has no custom batching rule",
                    describe_composite(COMPOSITE_OPERATION_NAME, self.name())
                ),
            }
            .into());
        };

        let Some(rule) = batching_rule.rule() else {
            let derivation = batching_rule.derivation().unwrap();
            return self.specialization(CustomRuleSpecializationRequest::BatchingRule(key.clone()), || {
                let CustomRuleSpecialization { program, output_axes, .. } =
                    derivation.batching_rule_specialization(self.name(), &key)?;
                let program = self.definition().discharge_if(key.discharged, program)?;
                Ok(Arc::new(CustomRuleSpecialization::new(program, output_axes)))
            });
        };

        self.specialization(CustomRuleSpecializationRequest::BatchingRule(key.clone()), || {
            let boundary_input_count = key.boundary_input_types.len();
            let mut output_axes = Vec::new();
            let (_, program) = TracingContext::<V, O>::trace_with_named_axes(
                |values: Vec<CustomRuleTracer<V, O>>| {
                    let (boundary_inputs, inputs) = values.split_at(boundary_input_count);
                    let (outputs, axes) = rule.apply(inputs, &key.input_axes, &key.level, boundary_inputs)?;
                    if axes.len() != outputs.len() {
                        return Err(TypeError::invalid(format!(
                            "{} batching rule returned {} outputs but {} output batch axes",
                            describe_composite(COMPOSITE_OPERATION_NAME, self.name()),
                            outputs.len(),
                            axes.len(),
                        ))
                        .into());
                    }
                    output_axes = axes;
                    Ok(outputs)
                },
                key.boundary_input_types.iter().chain(&key.input_types).cloned().collect::<Vec<_>>(),
                self.definition().named_axes().to_vec(),
            )?;
            let program = self.definition().discharge_if(key.discharged, program)?;
            Ok(Arc::new(CustomRuleSpecialization::new(program, output_axes)))
        })
    }

    fn sequential_batching_scan(
        &self,
        length: usize,
        carry_count: usize,
        axis_sharding: &ShardingDimension,
        stacked_output_count: usize,
    ) -> Result<(O, V::Type), DifferentiationError> {
        let Some(sequential_batching_scan) = self.definition().sequential_batching_scan() else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "{} does not batch its calls sequentially",
                    describe_composite(COMPOSITE_OPERATION_NAME, self.name())
                ),
            }
            .into());
        };
        Ok(sequential_batching_scan(length, carry_count, axis_sharding, stacked_output_count)?)
    }

    fn derived(&self, key: CustomRuleDerivedDefinitionKey) -> Result<Self, DifferentiationError> {
        // A derived definition copies the capabilities that its calls need from the source definition and holds no
        // program, so calls whose primal regions differ share it (refer to `CustomRuleSpecializer::derived`). A source
        // that batches its calls sequentially, without a custom batching rule, derives a definition that batches its
        // calls sequentially as well, which needs no derivation.
        if !self.has_batching_rule() && !self.has_sequential_batching() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "{} has neither a custom batching rule nor sequential batching from which to derive rules",
                    describe_composite(COMPOSITE_OPERATION_NAME, self.name()),
                ),
            }
            .into());
        }

        // A derived definition that is still alive is reused, so that its calls keep one identity (and with it the
        // entries of enclosing caches) even after its registration was evicted or the source registration dropped.
        let definition = {
            let mut definitions =
                self.definition().derived_definitions().lock().expect("custom rule derivation mutex is poisoned");
            definitions.retain(|_, definition| definition.strong_count() > 0);
            match definitions.get(&key).and_then(Weak::upgrade) {
                Some(definition) => definition,
                None => {
                    let prefix = match key.kind {
                        CustomRuleDerivationKind::Jvp => "jvp",
                        CustomRuleDerivationKind::Pushforward => "pushforward",
                    };
                    let batching_rule = self.definition().batching_rule().map(|batching_rule| {
                        batching_rule.new_derived(CustomRuleDerivation { source: self.clone(), key: key.clone() })
                    });
                    let definition =
                        Arc::new(self.definition().new_derived(format!("{}({})", prefix, self.name()), batching_rule));
                    definitions.insert(key.clone(), Arc::downgrade(&definition));
                    definition
                }
            }
        };
        Ok(self.derived_reference(key, definition))
    }
}

/// Rules of a definition registered in a family whose values the target family cannot represent (e.g., array IR rules
/// converted into a backend whose constants cannot represent every array IR value). Calls can still execute their
/// converted primal programs. Requests that need the retained rules fail with an error explaining why they are
/// unavailable. Differentiation explicitly derived from the primal can still succeed when it does not need those rules.
#[derive(Clone, Debug)]
pub struct UnavailableCustomRules {
    /// Process-unique identity of the original definition.
    id: u64,

    /// Human-readable label of the original definition.
    name: Cow<'static, str>,

    /// Whether the original definition has a retained forward mode differentiation (i.e., Jacobian-Vector Product
    /// or JVP) rule.
    has_jvp: bool,

    /// Whether the original definition derives its forward mode differentiation rule from the primal.
    derives_jvp_from_primal: bool,

    /// Whether the original definition has reverse mode (i.e., Vector-Jacobian Product or VJP) rules.
    has_vjp: bool,

    /// Whether the original definition defers its reverse mode differentiation forward rule to the backward pass.
    defers_forward: bool,

    /// Declared batch axes of the outputs of batched calls of the original definition.
    batched_output_axes: Option<Vec<BatchAxis>>,

    /// Whether the original definition has a custom batching rule.
    has_batching_rule: bool,

    /// Whether the original definition batches its calls sequentially.
    has_sequential_batching: bool,
}

impl UnavailableCustomRules {
    /// Creates [`UnavailableCustomRules`] that keep the identity and description of the provided source.
    #[inline]
    pub fn from_source<V: Typed<Type: Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>>(rules: &S) -> Self {
        Self {
            id: rules.id(),
            name: Cow::Owned(rules.name().to_owned()),
            has_jvp: rules.has_jvp(),
            derives_jvp_from_primal: rules.derives_jvp_from_primal(),
            has_vjp: rules.has_vjp(),
            defers_forward: rules.defers_forward(),
            batched_output_axes: rules.batched_output_axes().map(<[BatchAxis]>::to_vec),
            has_batching_rule: rules.has_batching_rule(),
            has_sequential_batching: rules.has_sequential_batching(),
        }
    }

    /// Returns the error reported when a request needs these unavailable retained rules.
    #[inline]
    fn error(&self) -> DifferentiationError {
        ProgramError::UnsupportedOperation {
            message: format!(
                "{} was converted from a family whose values its current family \
                 cannot represent, so its derivative rules are unavailable",
                describe_composite(COMPOSITE_OPERATION_NAME, &self.name)
            ),
        }
        .into()
    }
}

/// Source of the rules of a call that was converted from a member family `(V, O)` into a family that contains
/// it (e.g., from array programs into array IR programs, or into a backend family). Each specialization request
/// projects its key into the member family, obtains the specialization from the member definition (so that the member
/// definition's caches, batcher, and transposer serve every family that its calls reach), and converts the program
/// into the requesting family with [`Program::into_unprojected`]. Nested member calls inside that program are converted
/// in turn, by the same conversion that converted the call. The converted programs are cached per requesting family
/// with the member definition's specializations, so repeated requests convert each specialization once.
///
/// A member family cannot represent every value of the families that contain it, so some requests of converted calls
/// fail with exact errors at their first derivative request: rule inputs whose types have no member representation
/// (e.g., references or first-class dimensions of array IR programs), and batching levels whose dynamic extents have no
/// member representation. A backward rule of a converted call returns cotangents whose destinations are caller-provided
/// buffers, and the carrier's accumulators add them to those buffers.
pub enum LiftedCustomRules<V: Typed + Parameter, O> {
    /// Rules of a definition registered in the member family.
    Member(CustomRuleReference<V, O>),

    /// Rules of a definition that cannot be converted into the requesting family.
    Unavailable(UnavailableCustomRules),
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> LiftedCustomRules<V, O> {
    /// Returns the process-unique identity of the source definition, preserved when its rules become unavailable.
    #[inline]
    pub fn id(&self) -> u64 {
        match self {
            Self::Member(rules) => rules.definition().id(),
            Self::Unavailable(rules) => rules.id,
        }
    }

    /// Returns the rules of the member definition, or an error explaining why its retained rules are unavailable.
    #[inline]
    pub fn member(&self) -> Result<&CustomRuleReference<V, O>, DifferentiationError> {
        match self {
            Self::Member(rules) => Ok(rules),
            Self::Unavailable(rules) => Err(rules.error()),
        }
    }

    /// Projects the provided types of a converted call into the member family, rejecting types that have no member
    /// representation.
    fn project_types<T: Type>(&self, types: &[T]) -> Result<Vec<V::Type>, DifferentiationError>
    where
        for<'t> &'t V::Type: TryFrom<&'t T, Error = TypeError>,
    {
        types
            .iter()
            .map(|r#type| {
                <&V::Type>::try_from(r#type).cloned().map_err(|_| {
                    ProgramError::UnsupportedOperation {
                        message: format!(
                            "{} cannot be differentiated with respect to type `{}`, \
                             which the family in which its rules were registered cannot represent",
                            describe_composite(COMPOSITE_OPERATION_NAME, self.name_or_unavailable()),
                            r#type,
                        ),
                    }
                    .into()
                })
            })
            .collect()
    }

    /// Projects the provided batching levels of a converted call into the member family, rejecting dynamic extents
    /// whose types have no member representation.
    fn project_levels<T: Type>(
        &self,
        levels: &[CustomRuleBatchingLevel<T>],
    ) -> Result<Vec<CustomRuleBatchingLevel<V::Type>>, DifferentiationError>
    where
        for<'t> &'t V::Type: TryFrom<&'t T, Error = TypeError>,
    {
        levels
            .iter()
            .map(|level| {
                let extent = match level.level.extent() {
                    BatchingLevelExtent::Static(extent) => BatchingLevelExtent::Static(*extent),
                    BatchingLevelExtent::Dynamic(r#type) => {
                        BatchingLevelExtent::Dynamic(<&V::Type>::try_from(r#type).cloned().map_err(|_| {
                            ProgramError::UnsupportedOperation {
                                message: format!(
                                    "{} cannot differentiate a batched call whose batch extent has type `{}`, \
                                     which the family in which its rules were registered cannot represent; \
                                     differentiate the call before batching it instead",
                                    describe_composite(COMPOSITE_OPERATION_NAME, self.name_or_unavailable()),
                                    r#type,
                                ),
                            }
                        })?)
                    }
                };
                Ok(CustomRuleBatchingLevel {
                    level: BatchingLevel::new(
                        extent,
                        level.level.axis_name().map(str::to_owned),
                        level.level.axis_sharding().clone(),
                    ),
                    boundary_input_count: level.boundary_input_count,
                    input_axes: level.input_axes.clone(),
                    output_axes: level.output_axes.clone(),
                })
            })
            .collect()
    }

    /// Returns the name of the member definition or of the unavailable definition.
    fn name_or_unavailable(&self) -> &str {
        match self {
            Self::Member(rules) => rules.name(),
            Self::Unavailable(rules) => &rules.name,
        }
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Clone for LiftedCustomRules<V, O> {
    #[inline]
    fn clone(&self) -> Self {
        match self {
            Self::Member(rules) => Self::Member(rules.clone()),
            Self::Unavailable(rules) => Self::Unavailable(rules.clone()),
        }
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Debug for LiftedCustomRules<V, O> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Member(rules) => formatter.debug_tuple("Member").field(rules).finish(),
            Self::Unavailable(rules) => formatter.debug_tuple("Unavailable").field(rules).finish(),
        }
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> PartialEq for LiftedCustomRules<V, O> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Eq for LiftedCustomRules<V, O> {}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Hash for LiftedCustomRules<V, O> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}

impl<V: Typed + Parameter, O> sealed::Sealed for LiftedCustomRules<V, O> {}

impl<V, O, VC, OC> CustomRuleSource<VC, OC> for LiftedCustomRules<V, O>
where
    V: Typed<Type: Eq + Hash> + Parameter,
    VC: Typed<Type: Eq + Hash> + Parameter,
{
    #[inline]
    fn id(&self) -> u64 {
        self.id()
    }

    #[inline]
    fn name(&self) -> &str {
        self.name_or_unavailable()
    }

    #[inline]
    fn has_jvp(&self) -> bool {
        match self {
            Self::Member(rules) => rules.has_jvp(),
            Self::Unavailable(rules) => rules.has_jvp,
        }
    }

    #[inline]
    fn derives_jvp_from_primal(&self) -> bool {
        match self {
            Self::Member(rules) => rules.derives_jvp_from_primal(),
            Self::Unavailable(rules) => rules.derives_jvp_from_primal,
        }
    }

    #[inline]
    fn has_vjp(&self) -> bool {
        match self {
            Self::Member(rules) => rules.has_vjp(),
            Self::Unavailable(rules) => rules.has_vjp,
        }
    }

    #[inline]
    fn defers_forward(&self) -> bool {
        match self {
            Self::Member(rules) => rules.defers_forward(),
            Self::Unavailable(rules) => rules.defers_forward,
        }
    }

    #[inline]
    fn batched_output_axes(&self) -> Option<&[BatchAxis]> {
        match self {
            Self::Member(rules) => rules.batched_output_axes(),
            Self::Unavailable(rules) => rules.batched_output_axes.as_deref(),
        }
    }

    #[inline]
    fn has_batching_rule(&self) -> bool {
        match self {
            Self::Member(rules) => rules.has_batching_rule(),
            Self::Unavailable(rules) => rules.has_batching_rule,
        }
    }

    #[inline]
    fn has_sequential_batching(&self) -> bool {
        match self {
            Self::Member(rules) => rules.has_sequential_batching(),
            Self::Unavailable(rules) => rules.has_sequential_batching,
        }
    }

    #[inline]
    fn native_definition(&self) -> Option<&CustomRuleDefinition<VC, OC>> {
        None
    }

    #[inline]
    fn supports_reference_destinations(&self) -> bool {
        false
    }
}

impl<V, O, VC, OC> CustomRuleSpecializer<VC, OC> for LiftedCustomRules<V, O>
where
    V: Value<Type: DifferentiableType + Eq + Hash>,
    O: Operation<Type = V::Type>
        + From<CompositeTransposeOperation<V, O>>
        + ResidualZeroProvider<V::Type, Operation = O>,
    VC: 'static + ValueProjection<V::Type, Projected = V, Type: Eq + Hash + From<V::Type>>,
    OC: 'static + From<O>,
    for<'t> &'t V::Type: TryFrom<&'t VC::Type, Error = TypeError>,
    CustomRuleSpecialization<VC, OC>: Send + Sync,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    // State the source contract explicitly to aid RustRover's trait resolution.
    Self: CustomRuleSource<VC, OC>,
{
    fn jvp_specialization(
        &self,
        key: CustomRuleForwardSpecializationKey<VC::Type>,
    ) -> Result<Arc<CustomRuleSpecialization<VC, OC>>, DifferentiationError>
    where
        VC: Value<Type: DifferentiableType>,
        OC: Operation<Type = VC::Type> + ResidualZeroProvider<VC::Type, Operation = OC>,
    {
        let member = self.member()?;
        let member_key = CustomRuleForwardSpecializationKey {
            input_types: self.project_types(&key.input_types)?,
            output_types: self.project_types(&key.output_types)?,
            kept_outputs: key.kept_outputs.clone(),
            non_differentiated_count: key.non_differentiated_count,
            tangent_activity: key.tangent_activity.clone(),
            levels: self.project_levels(&key.levels)?,
            discharged: key.discharged,
        };
        member.converted_specialization(CustomRuleSpecializationRequest::Jvp(member_key.clone()), || {
            let specialization = member.jvp_specialization(member_key)?;
            Ok(Arc::new(CustomRuleSpecialization {
                program: specialization.program.clone().into_unprojected::<VC, OC>()?,
                output_axes: specialization.output_axes.clone(),
                cotangents: specialization.cotangents.clone(),
            }))
        })
    }

    fn forward_specialization(
        &self,
        key: CustomRuleForwardSpecializationKey<VC::Type>,
    ) -> Result<Arc<CustomRuleSpecialization<VC, OC>>, DifferentiationError>
    where
        VC: Value<Type: DifferentiableType>,
        OC: Operation<Type = VC::Type>,
    {
        let member = self.member()?;
        let member_key = CustomRuleForwardSpecializationKey {
            input_types: self.project_types(&key.input_types)?,
            output_types: self.project_types(&key.output_types)?,
            kept_outputs: key.kept_outputs.clone(),
            non_differentiated_count: key.non_differentiated_count,
            tangent_activity: key.tangent_activity.clone(),
            levels: self.project_levels(&key.levels)?,
            discharged: key.discharged,
        };
        member.converted_specialization(CustomRuleSpecializationRequest::Forward(member_key.clone()), || {
            let specialization = member.forward_specialization(member_key)?;
            Ok(Arc::new(CustomRuleSpecialization {
                program: specialization.program.clone().into_unprojected::<VC, OC>()?,
                output_axes: specialization.output_axes.clone(),
                cotangents: specialization.cotangents.clone(),
            }))
        })
    }

    fn backward_specialization(
        &self,
        key: CustomRuleBackwardSpecializationKey<VC::Type>,
    ) -> Result<Arc<CustomRuleSpecialization<VC, OC>>, DifferentiationError>
    where
        VC: Value<Type: DifferentiableType>,
        OC: Operation<Type = VC::Type> + From<CompositeTransposeOperation<VC, OC>>,
    {
        // Carriers of converted calls request returned cotangents for caller buffers (refer to
        // `supports_reference_destinations`), so a buffer destination here means that a carrier skipped that step.
        let member = self.member()?;
        if key.destination_kinds.contains(&CotangentDestinationKind::Reference) {
            return Err(ProgramError::MalformedProgram(format!(
                "{} requested a caller-buffer destination from rules that cannot write caller buffers",
                describe_composite(COMPOSITE_TRANSPOSE_OPERATION_NAME, self.name_or_unavailable()),
            ))
            .into());
        }
        let seed_types = key
            .seed_types
            .iter()
            .map(|r#type| {
                r#type
                    .as_ref()
                    .map(|r#type| Ok(self.project_types(std::slice::from_ref(r#type))?.remove(0)))
                    .transpose()
            })
            .collect::<Result<Vec<_>, DifferentiationError>>()?;
        let member_key = CustomRuleBackwardSpecializationKey {
            leading_input_types: self.project_types(&key.leading_input_types)?,
            seed_extent_count: key.seed_extent_count,
            input_tangent_types: self.project_types(&key.input_tangent_types)?,
            output_tangent_types: self.project_types(&key.output_tangent_types)?,
            seed_types,
            destination_kinds: key.destination_kinds.clone(),
            levels: self.project_levels(&key.levels)?,
            discharged: key.discharged,
        };
        member.converted_specialization(CustomRuleSpecializationRequest::Backward(member_key.clone()), || {
            let specialization = member.backward_specialization(member_key)?;
            Ok(Arc::new(CustomRuleSpecialization {
                program: specialization.program.clone().into_unprojected::<VC, OC>()?,
                output_axes: specialization.output_axes.clone(),
                cotangents: specialization.cotangents.clone(),
            }))
        })
    }

    fn batching_rule_specialization(
        &self,
        key: CustomRuleBatchingSpecializationKey<VC::Type>,
    ) -> Result<Arc<CustomRuleSpecialization<VC, OC>>, DifferentiationError>
    where
        VC: Value<Type: DifferentiableType>,
        OC: Operation<Type = VC::Type>,
    {
        // The member rule batches the call at the projected input types, and its program is converted back.
        let member = self.member()?;
        let project = |r#type: &VC::Type| {
            <&V::Type>::try_from(r#type).cloned().map_err(|_| {
                DifferentiationError::from(ProgramError::UnsupportedOperation {
                    message: format!(
                        "{} cannot batch a call over type `{}`, which the family in \
                         which its rules were registered cannot represent",
                        describe_composite(COMPOSITE_OPERATION_NAME, self.name_or_unavailable()),
                        r#type,
                    ),
                })
            })
        };
        let extent = match key.level.extent() {
            BatchingLevelExtent::Static(extent) => BatchingLevelExtent::Static(*extent),
            BatchingLevelExtent::Dynamic(r#type) => BatchingLevelExtent::Dynamic(project(r#type)?),
        };
        let member_key = CustomRuleBatchingSpecializationKey {
            level: BatchingLevel::new(
                extent,
                key.level.axis_name().map(str::to_owned),
                key.level.axis_sharding().clone(),
            ),
            boundary_input_types: key.boundary_input_types.iter().map(project).collect::<Result<_, _>>()?,
            input_types: key.input_types.iter().map(project).collect::<Result<_, _>>()?,
            unbatched_input_types: key.unbatched_input_types.iter().map(project).collect::<Result<_, _>>()?,
            input_axes: key.input_axes.clone(),
            discharged: key.discharged,
        };
        member.converted_specialization(CustomRuleSpecializationRequest::BatchingRule(member_key.clone()), || {
            let specialization = member.batching_rule_specialization(member_key)?;
            Ok(Arc::new(CustomRuleSpecialization {
                program: specialization.program.clone().into_unprojected::<VC, OC>()?,
                output_axes: specialization.output_axes.clone(),
                cotangents: specialization.cotangents.clone(),
            }))
        })
    }

    #[inline]
    fn sequential_batching_scan(
        &self,
        length: usize,
        carry_count: usize,
        axis_sharding: &ShardingDimension,
        stacked_output_count: usize,
    ) -> Result<(OC, VC::Type), DifferentiationError> {
        // The member definition constructs the scan in its own family, which contains the scan operation and the type
        // of its slice index of this family.
        let (scan, index_type) =
            self.member()?.sequential_batching_scan(length, carry_count, axis_sharding, stacked_output_count)?;
        Ok((OC::from(scan), index_type.into()))
    }

    #[inline]
    fn derived(&self, key: CustomRuleDerivedDefinitionKey) -> Result<Self, DifferentiationError> {
        // The member definition derives in its own family, and the derived rules follow the call like the member's
        // rules do. Unavailable rules stay unavailable, so the derived call reports the same error on its first
        // request.
        match self {
            Self::Member(rules) => Ok(Self::Member(rules.derived(key)?)),
            Self::Unavailable(rules) => Ok(Self::Unavailable(rules.clone())),
        }
    }
}

impl<V: Value<Type: DifferentiableType + Eq + Hash>, O: Operation<Type = V::Type>> CustomRuleDerivation<V, O>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    /// Returns the batched program of a derived call for `key` (refer to [`CustomRuleDerivation`] for its two modes),
    /// which consumes the level's boundary inputs followed by the derived call's batched inputs (i.e., its primal
    /// inputs, including the boundary inputs of earlier levels, followed by its active tangents) and returns its
    /// batched outputs, together with their batch axes. The source rule programs that it applies are discharged
    /// when `key` is, while the returned program is left for the caller to discharge.
    ///
    /// # Parameters
    ///
    ///   - `name`: Name of the derived definition, used in diagnostics.
    ///   - `key`: Specialization key of the batched derived call.
    fn batching_rule_specialization(
        &self,
        name: &str,
        key: &CustomRuleBatchingSpecializationKey<V::Type>,
    ) -> Result<CustomRuleSpecialization<V, O>, DifferentiationError> {
        let source = self.source.definition();

        // A source has a derivation only when it has a batching rule, so only its structural batcher can be missing.
        let (Some(batching_rule), Some(batcher)) = (source.batching_rule(), source.structural_batcher()) else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "{} cannot batch a derivative because its source definition has no batching support (refer \
                     to `CustomRuleDefinition::with_batching_support`)",
                    describe_composite(COMPOSITE_OPERATION_NAME, name),
                ),
            }
            .into());
        };
        let differentiator = &batching_rule.differentiator;
        let input_count = key.input_types.len();
        let tangent_count = self.key.active_input_indices.len();
        let primal_count = input_count
            .checked_sub(tangent_count)
            .filter(|count| *count >= self.key.input_count)
            .filter(|_| key.unbatched_input_types.len() == input_count && key.input_axes.len() == input_count)
            .ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "batched {} has {} inputs, which do not cover its {} primal inputs and {} tangents",
                    describe_composite(COMPOSITE_OPERATION_NAME, name),
                    input_count,
                    self.key.input_count,
                    tangent_count,
                ))
            })?;

        // The primal inputs of a derived call that was already batched start with the boundary inputs of the earlier
        // levels, which have no tangents.
        let earlier_boundary_input_count = primal_count - self.key.input_count;
        let tangent_primal_indices = self
            .key
            .active_input_indices
            .iter()
            .map(|index| earlier_boundary_input_count + index)
            .collect::<Vec<_>>();
        let (primal_axes, tangent_axes) = key.input_axes.split_at(primal_count);
        let mut primal_tangents = vec![None; primal_count];
        for (tangent, &primal) in tangent_primal_indices.iter().enumerate() {
            primal_tangents[primal] = Some(tangent);
        }
        let boundary_input_count = key.boundary_input_types.len();
        let jvp_input_indices =
            tangent_primal_indices.iter().map(|index| boundary_input_count + index).collect::<Vec<_>>();
        let rule_key = |input_types: Vec<V::Type>, input_axes: Vec<BatchAxis>| CustomRuleBatchingSpecializationKey {
            level: key.level.clone(),
            boundary_input_types: key.boundary_input_types.clone(),
            input_types,
            unbatched_input_types: key.unbatched_input_types[..primal_count].to_vec(),
            input_axes,
            discharged: key.discharged,
        };
        let program_types = key.boundary_input_types.iter().chain(&key.input_types).cloned().collect::<Vec<_>>();
        let tangents_only = self.key.kind == CustomRuleDerivationKind::Pushforward;

        // The source rule applied at the primal inputs' own batch axes is the batched primal of the source call,
        // so it computes the derived call's primal outputs in every mode.
        let primal_rule = self
            .source
            .batching_rule_specialization(rule_key(key.input_types[..primal_count].to_vec(), primal_axes.to_vec()))?;

        // If every primal input is replicated, then so is every output of the rule applied to them.
        if primal_axes.iter().chain(&primal_rule.output_axes).all(BatchAxis::is_replicated) {
            let jvp = self.jvp_program(
                name,
                differentiator.as_ref(),
                &primal_rule.program,
                &jvp_input_indices,
                tangents_only,
            )?;
            let mut input_axes = vec![BatchAxis::replicated(); boundary_input_count];
            input_axes.extend_from_slice(&key.input_axes);
            let (batched, output_axes) = batcher(
                &jvp,
                &input_axes,
                &vec![None; jvp.output_types().len()],
                &key.level,
                CustomRuleBatchedOutputs::Values,
            )
            .map_err(ProgramError::from)?;

            // The batched program consumes the level's boundary inputs before the JVP program's own copy of them.
            let (_, program) = TracingContext::<V, O>::trace_with_named_axes(
                |values: Vec<CustomRuleTracer<V, O>>| {
                    let context = values[0].context().clone();
                    let mut inputs = values[..boundary_input_count].to_vec();
                    inputs.extend(values);
                    batched.interpret_in_context(&context, inputs)
                },
                program_types,
                self.source.definition().named_axes().to_vec(),
            )?;
            return Ok(CustomRuleSpecialization::new(program, output_axes));
        }

        // Align each primal and tangent pair with a mapped side to one axis, preferring the primal's axis. Only a
        // replicated primal whose tangent is mapped is broadcast, in which case the rule applied at the aligned axes
        // computes the tangents, while the primal outputs still come from the rule applied at the primal axes.
        let mut aligned_axes = key.input_axes.clone();
        for (tangent, &primal) in tangent_primal_indices.iter().enumerate() {
            let axis = if primal_axes[primal].is_replicated() { tangent_axes[tangent] } else { primal_axes[primal] };
            aligned_axes[primal] = axis;
            aligned_axes[primal_count + tangent] = axis;
        }
        let broadcasts_primals = aligned_axes[..primal_count] != *primal_axes;
        if broadcasts_primals && let Some(broadcast_sensitivity) = &batching_rule.broadcast_sensitivity {
            return self.broadcast_sensitive_batching_rule_specialization(
                name,
                key,
                &primal_rule,
                &tangent_primal_indices,
                broadcast_sensitivity.as_ref(),
                differentiator.as_ref(),
                batcher,
            );
        }
        let aligned_inputs =
            (0..input_count).filter(|&index| aligned_axes[index] != key.input_axes[index]).collect::<Vec<_>>();
        let mut aligned_types = key.input_types.clone();
        let alignment = match aligned_inputs.is_empty() {
            true => None,
            false => {
                let alignment = Self::alignment_program(
                    name,
                    batcher,
                    &aligned_inputs.iter().map(|&index| key.unbatched_input_types[index].clone()).collect::<Vec<_>>(),
                    &aligned_inputs.iter().map(|&index| key.input_axes[index]).collect::<Vec<_>>(),
                    &aligned_inputs.iter().map(|&index| aligned_axes[index]).collect::<Vec<_>>(),
                    &key.level,
                )?;
                for (&index, r#type) in aligned_inputs.iter().zip(alignment.output_types()) {
                    aligned_types[index] = r#type;
                }
                Some(alignment)
            }
        };

        let rule = match broadcasts_primals {
            true => self.source.batching_rule_specialization(rule_key(
                aligned_types[..primal_count].to_vec(),
                aligned_axes[..primal_count].to_vec(),
            ))?,
            false => primal_rule.clone(),
        };

        let separate_primal = broadcasts_primals && !tangents_only;
        let jvp = self.jvp_program(
            name,
            differentiator.as_ref(),
            &rule.program,
            &jvp_input_indices,
            tangents_only || separate_primal,
        )?;

        let tangent_output_axes = rule
            .output_axes
            .iter()
            .zip(&self.key.output_tangent_mask)
            .filter_map(|(axis, live)| live.then_some(*axis));

        let output_axes = match tangents_only {
            true => tangent_output_axes.collect(),
            false => primal_rule.output_axes.iter().copied().chain(tangent_output_axes).collect(),
        };

        let (_, program) = TracingContext::<V, O>::trace_with_named_axes(
            |values: Vec<CustomRuleTracer<V, O>>| {
                let context = values[0].context().clone();
                let (boundary_inputs, inputs) = values.split_at(boundary_input_count);
                let mut inputs = inputs.to_vec();
                if let Some(alignment) = &alignment {
                    let mut alignment_inputs = boundary_inputs.to_vec();
                    alignment_inputs.extend(aligned_inputs.iter().map(|&index| inputs[index].clone()));
                    let aligned = alignment.interpret_in_context(&context, alignment_inputs)?;
                    for (&index, value) in aligned_inputs.iter().zip(aligned) {
                        inputs[index] = value;
                    }
                }
                let mut outputs = Vec::new();
                if separate_primal {
                    let primal_inputs = boundary_inputs.iter().chain(&values[boundary_input_count..][..primal_count]);
                    outputs = primal_rule.program.interpret_in_context(&context, primal_inputs.cloned().collect())?;
                }
                let mut jvp_inputs = boundary_inputs.to_vec();
                jvp_inputs.extend(inputs);
                outputs.extend(jvp.interpret_in_context(&context, jvp_inputs)?);
                Ok(outputs)
            },
            program_types,
            self.source.definition().named_axes().to_vec(),
        )?;

        Ok(CustomRuleSpecialization::new(program, output_axes))
    }

    /// Returns the batched program of a derived call for `key` when its source batching rule
    /// depends on its input batch axes and some replicated primal input has a mapped tangent (refer to
    /// [`CustomRuleDefinition::with_broadcast_sensitive_batching_rule`]). It is applied once, at the primal inputs'
    /// own batch axes, as `P`, which computes the primal outputs. The tangents are the sum of two contributions:
    ///
    ///   - **Following:** The Jacobian-Vector Product (JVP) of `P` with respect to the tangents that can follow their
    ///     primal's batch axis, after aligning them to it (i.e., broadcasting or moving them).
    ///   - **Tangent-Only:** The Jacobian-Vector Product (JVP) of `P` with respect to the mapped tangents of replicated
    ///     primals. Those tangents differ per batch item while their primals are shared, so the JVP is batched
    ///     structurally over them at a second level of the same extent, which yields the outer product of their batch
    ///     with the batch of `P`. Item `i` of the result is item `i` of that outer product along both axes, which is
    ///     its diagonal.
    ///
    /// # Parameters
    ///
    ///   - `name`: Name of the derived definition, used in diagnostics.
    ///   - `key`: Specialization key of the batched derived call.
    ///   - `primal_rule`: Source rule applied at the primal inputs' batch axes (i.e., `P`).
    ///   - `tangent_primal_indices`: Index of the primal input of each active tangent, among the derived call's primal
    ///     inputs.
    ///   - `broadcast_sensitivity`: Operations of the source definition that take diagonals and add contributions.
    ///   - `differentiator`: Differentiator of the source definition.
    ///   - `batcher`: Structural batcher of the source definition.
    #[allow(clippy::type_complexity)]
    fn broadcast_sensitive_batching_rule_specialization(
        &self,
        name: &str,
        key: &CustomRuleBatchingSpecializationKey<V::Type>,
        primal_rule: &Arc<CustomRuleSpecialization<V, O>>,
        tangent_primal_indices: &[usize],
        broadcast_sensitivity: &dyn CustomRuleBroadcastSensitivity<V, O>,
        differentiator: &(
             dyn Fn(
            &Program<V, O, Vec<V>, Vec<V>>,
            &[usize],
        ) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError>
                 + 'static
                 + Send
                 + Sync
         ),
        batcher: &(
             dyn Fn(
            &Program<V, O, Vec<V>, Vec<V>>,
            &[BatchAxis],
            &[Option<BatchAxis>],
            &BatchingLevel<<V as Typed>::Type>,
            CustomRuleBatchedOutputs,
        ) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Vec<BatchAxis>), BatchingError>
                 + 'static
                 + Send
                 + Sync
         ),
    ) -> Result<CustomRuleSpecialization<V, O>, DifferentiationError> {
        let tangents_only = self.key.kind == CustomRuleDerivationKind::Pushforward;
        let boundary_input_count = key.boundary_input_types.len();
        let primal_count = key.input_types.len() - tangent_primal_indices.len();
        let (primal_axes, tangent_axes) = key.input_axes.split_at(primal_count);

        // Batching policies report normalized (i.e., non-negative) batch axis positions.
        let position = |axis: Axis| {
            usize::try_from(axis.value()).map_err(|_| {
                ProgramError::MalformedProgram(format!(
                    "batched {} has the unnormalized batch axis {axis}",
                    describe_composite(COMPOSITE_OPERATION_NAME, name),
                ))
            })
        };
        let (following, tangent_only): (Vec<usize>, Vec<usize>) =
            (0..tangent_primal_indices.len()).partition(|&tangent| {
                !primal_axes[tangent_primal_indices[tangent]].is_replicated() || tangent_axes[tangent].is_replicated()
            });
        let jvp_input_indices = |tangents: &[usize]| {
            tangents
                .iter()
                .map(|&tangent| boundary_input_count + tangent_primal_indices[tangent])
                .collect::<Vec<_>>()
        };

        // The following contribution takes the JVP of `P` after moving or broadcasting each tangent to its primal's
        // batch axis.
        let (following_jvp, following_mask) = match following.is_empty() {
            true => (None, vec![false; self.key.output_tangent_mask.len()]),
            false => {
                let (jvp, mask) = Self::partial_jvp_program(
                    differentiator,
                    &primal_rule.program,
                    &jvp_input_indices(&following),
                    true,
                )?;
                (Some(jvp), mask)
            }
        };
        let misaligned_following = following
            .iter()
            .copied()
            .filter(|&tangent| tangent_axes[tangent] != primal_axes[tangent_primal_indices[tangent]])
            .collect::<Vec<_>>();
        let following_alignment = match misaligned_following.is_empty() {
            true => None,
            false => Some(Self::alignment_program(
                name,
                batcher,
                &misaligned_following
                    .iter()
                    .map(|&tangent| key.unbatched_input_types[primal_count + tangent].clone())
                    .collect::<Vec<_>>(),
                &misaligned_following.iter().map(|&tangent| tangent_axes[tangent]).collect::<Vec<_>>(),
                &misaligned_following
                    .iter()
                    .map(|&tangent| primal_axes[tangent_primal_indices[tangent]])
                    .collect::<Vec<_>>(),
                &key.level,
            )?),
        };

        // The tangent-only contribution batches the JVP of `P` over the mapped tangents at a second, anonymous and
        // replicated level of the same extent, which consumes the same boundary inputs as the call's level.
        let (tangent_only_jvp, tangent_only_mask) =
            Self::partial_jvp_program(differentiator, &primal_rule.program, &jvp_input_indices(&tangent_only), true)?;
        let outer_level = BatchingLevel::new(key.level.extent().clone(), None, ShardingDimension::Replicated);
        let mut outer_input_axes = vec![BatchAxis::replicated(); boundary_input_count + primal_count];
        outer_input_axes.extend(tangent_only.iter().map(|&tangent| tangent_axes[tangent]));
        let (outer_product, outer_axes) = batcher(
            &tangent_only_jvp,
            &outer_input_axes,
            &vec![None; tangent_only_jvp.output_types().len()],
            &outer_level,
            CustomRuleBatchedOutputs::Values,
        )
        .map_err(ProgramError::from)?;
        let combined_mask = following_mask
            .iter()
            .zip(&tangent_only_mask)
            .map(|(following, tangent_only)| *following || *tangent_only);
        self.validate_output_tangent_mask(name, &combined_mask.collect::<Vec<_>>())?;

        // Each live output tangent is the sum of the contributions that it has. The batch axis of `P`'s output is
        // kept when it is mapped. Otherwise, the tangent-only contribution varies only along the second level's
        // axis, which becomes the output's batch axis, and the following contribution is broadcast along it.
        let mut following_index = 0;
        let mut tangent_only_index = 0;
        let mut tangents = Vec::new();
        for (output, axis) in primal_rule.output_axes.iter().enumerate() {
            let following = following_mask[output].then(|| {
                following_index += 1;
                following_index - 1
            });
            let tangent_only = tangent_only_mask[output].then(|| {
                tangent_only_index += 1;
                tangent_only_index - 1
            });
            if following.is_none() && tangent_only.is_none() {
                continue;
            }
            let outer_axis = match tangent_only.and_then(|index| outer_axes[index].axis()) {
                None => None,
                Some(outer_axis) => Some(position(outer_axis)?),
            };
            tangents.push((following, tangent_only, outer_axis, *axis));
        }
        let output_axes = tangents
            .iter()
            .map(|(_, _, outer_axis, axis)| match (axis.is_replicated(), outer_axis) {
                (true, Some(outer_axis)) => BatchAxis::new(*outer_axis),
                _ => *axis,
            })
            .collect::<Vec<_>>();
        let broadcast_following = tangents
            .iter()
            .zip(&output_axes)
            .filter_map(|((following, _, _, axis), output_axis)| {
                following
                    .filter(|_| axis.is_replicated() && !output_axis.is_replicated())
                    .map(|index| (index, *output_axis))
            })
            .collect::<Vec<_>>();
        let following_broadcast = match broadcast_following.is_empty() {
            true => None,
            false => {
                let following_types = following_jvp.as_ref().unwrap().output_types();
                Some(Self::alignment_program(
                    name,
                    batcher,
                    &broadcast_following.iter().map(|(index, _)| following_types[*index].clone()).collect::<Vec<_>>(),
                    &vec![BatchAxis::replicated(); broadcast_following.len()],
                    &broadcast_following.iter().map(|(_, axis)| *axis).collect::<Vec<_>>(),
                    &key.level,
                )?)
            }
        };
        let mut all_output_axes = match tangents_only {
            true => Vec::new(),
            false => primal_rule.output_axes.clone(),
        };
        all_output_axes.extend(output_axes.iter().copied());

        let (_, program) = TracingContext::<V, O>::trace_with_named_axes(
            |values: Vec<CustomRuleTracer<V, O>>| {
                let context = values[0].context().clone();
                let (boundary_inputs, inputs) = values.split_at(boundary_input_count);
                let (primals, input_tangents) = inputs.split_at(primal_count);
                let leading_inputs = boundary_inputs.iter().chain(primals).cloned().collect::<Vec<_>>();
                let mut outputs = Vec::new();
                if !tangents_only {
                    outputs = primal_rule.program.interpret_in_context(&context, leading_inputs.clone())?;
                }

                // Following contribution.
                let mut following_tangents =
                    following.iter().map(|&tangent| input_tangents[tangent].clone()).collect::<Vec<_>>();
                if let Some(alignment) = &following_alignment {
                    let mut alignment_inputs = boundary_inputs.to_vec();
                    alignment_inputs
                        .extend(misaligned_following.iter().map(|&tangent| input_tangents[tangent].clone()));
                    let aligned = alignment.interpret_in_context(&context, alignment_inputs)?;
                    for (&tangent, value) in misaligned_following.iter().zip(aligned) {
                        let position = following.iter().position(|candidate| *candidate == tangent).unwrap();
                        following_tangents[position] = value;
                    }
                }
                let mut following_outputs = match &following_jvp {
                    None => Vec::new(),
                    Some(jvp) => {
                        let mut jvp_inputs = leading_inputs.clone();
                        jvp_inputs.extend(following_tangents);
                        jvp.interpret_in_context(&context, jvp_inputs)?
                    }
                };
                if let Some(broadcast) = &following_broadcast {
                    let mut broadcast_inputs = boundary_inputs.to_vec();
                    broadcast_inputs
                        .extend(broadcast_following.iter().map(|(index, _)| following_outputs[*index].clone()));
                    let broadcast = broadcast.interpret_in_context(&context, broadcast_inputs)?;
                    for ((index, _), value) in broadcast_following.iter().zip(broadcast) {
                        following_outputs[*index] = value;
                    }
                }

                // Tangent-only contribution.
                let mut outer_inputs = boundary_inputs.to_vec();
                outer_inputs.extend(leading_inputs.iter().cloned());
                outer_inputs.extend(tangent_only.iter().map(|&tangent| input_tangents[tangent].clone()));
                let outer_outputs = outer_product.interpret_in_context(&context, outer_inputs)?;
                for (following, tangent_only, outer_axis, axis) in &tangents {
                    let tangent_only = match (tangent_only, axis.axis(), *outer_axis) {
                        (None, _, _) => None,
                        (Some(index), None, _) => Some(outer_outputs[*index].clone()),
                        (Some(index), Some(_), None) => {
                            // An output that does not depend on the tangent-only inputs (e.g., whose contribution is a
                            // zero) is replicated along the second level, so its contribution already has `P`'s batch
                            // axis.
                            Some(outer_outputs[*index].clone())
                        }
                        (Some(index), Some(axis), Some(outer_axis)) => {
                            // The second level's axis precedes `P`'s batch axis in the outer product exactly when its
                            // position does not exceed that axis's position in `P`'s output.
                            let axis = position(axis)?;
                            let kept_axis = if outer_axis <= axis { axis + 1 } else { axis };
                            Some(broadcast_sensitivity.diagonal(&outer_outputs[*index], kept_axis, outer_axis)?)
                        }
                    };

                    let following = following.map(|index| following_outputs[index].clone());

                    // Every recorded output has at least one contribution.
                    outputs.push(match (following, tangent_only) {
                        (Some(following), Some(tangent_only)) => {
                            broadcast_sensitivity.add(&following, &tangent_only)?
                        }
                        (Some(tangent), None) => tangent,
                        (None, tangent) => tangent.unwrap(),
                    });
                }
                Ok(outputs)
            },
            key.boundary_input_types.iter().chain(&key.input_types).cloned().collect::<Vec<_>>(),
            self.source.definition().named_axes().to_vec(),
        )?;

        Ok(CustomRuleSpecialization::new(program, all_output_axes))
    }

    /// Returns the Jacobian-Vector Product (JVP) program of the source rule's batched `program` with respect to
    /// `input_indices`, restricted to its tangent outputs when `tangents_only` is set, after validating that its live
    /// tangent outputs are those of the derived calls. The derivatives with respect to subsets of the active inputs
    /// (refer to [`Self::broadcast_sensitive_batching_rule_specialization`]) use [`Self::partial_jvp_program`] instead.
    ///
    /// # Parameters
    ///
    ///   - `name`: Name of the derived definition, used in diagnostics.
    ///   - `differentiator`: Differentiator of the source definition.
    ///   - `program`: Batched program of the source rule.
    ///   - `input_indices`: Indices of the program inputs whose tangents are active.
    ///   - `tangents_only`: Whether the returned program returns only the tangent outputs (e.g., for pushforward
    ///     derivations, or when another program computes the primal outputs).
    #[allow(clippy::type_complexity)]
    #[inline]
    fn jvp_program(
        &self,
        name: &str,
        differentiator: &(
             dyn Fn(
            &Program<V, O, Vec<V>, Vec<V>>,
            &[usize],
        ) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError>
                 + 'static
                 + Send
                 + Sync
         ),
        program: &Program<V, O, Vec<V>, Vec<V>>,
        input_indices: &[usize],
        tangents_only: bool,
    ) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError> {
        let (jvp, output_tangent_mask) =
            Self::partial_jvp_program(differentiator, program, input_indices, tangents_only)?;
        self.validate_output_tangent_mask(name, &output_tangent_mask)?;
        Ok(jvp)
    }

    /// Returns the Jacobian-Vector Product (JVP) program of the source rule's batched `program` with respect to
    /// `input_indices`, restricted to its tangent outputs when `tangents_only` is set, together with whether each
    /// output has a live tangent with respect to those inputs (i.e., which outputs the JVP program's tangent outputs
    /// belong to).
    #[allow(clippy::type_complexity)]
    fn partial_jvp_program(
        differentiator: &(
             dyn Fn(
            &Program<V, O, Vec<V>, Vec<V>>,
            &[usize],
        ) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError>
                 + 'static
                 + Send
                 + Sync
         ),
        program: &Program<V, O, Vec<V>, Vec<V>>,
        input_indices: &[usize],
        tangents_only: bool,
    ) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Vec<bool>), DifferentiationError> {
        let output_tangent_mask = program.entry_region_ref().tangent_output_mask(input_indices)?;
        let jvp = differentiator(program, input_indices)?;
        if !tangents_only {
            return Ok((jvp, output_tangent_mask));
        }
        let tangent_outputs = (program.output_count()..jvp.output_count()).collect::<Vec<_>>();
        Ok((jvp.with_outputs(&tangent_outputs)?, output_tangent_mask))
    }

    /// Validates that the source rule's batched program has live tangents for exactly the outputs whose tangents the
    /// derivative of the derived calls' primal has.
    fn validate_output_tangent_mask(
        &self,
        name: &str,
        output_tangent_mask: &[bool],
    ) -> Result<(), DifferentiationError> {
        if output_tangent_mask != self.key.output_tangent_mask {
            return Err(TypeError::invalid(format!(
                "{} batching rule program has live tangents for the outputs {:?}, \
                 but the derivative of its primal has them for the outputs {:?}",
                describe_composite(COMPOSITE_OPERATION_NAME, name),
                output_tangent_mask,
                self.key.output_tangent_mask,
            ))
            .into());
        }
        Ok(())
    }

    /// Returns the program that aligns values of the provided per-item `types` from `input_axes` to `output_axes` at
    /// `level` (i.e., that broadcasts a replicated value or moves a mapped axis), which consumes the level's boundary
    /// inputs followed by the values and returns the aligned values.
    ///
    /// # Parameters
    ///
    ///   - `name`: Name of the derived definition, used in diagnostics.
    ///   - `batcher`: Structural batcher of the source definition.
    ///   - `types`: Per-item types of the values.
    ///   - `input_axes`: Batch axes of the values.
    ///   - `output_axes`: Batch axes of the aligned values.
    ///   - `level`: Batching level of the values.
    #[allow(clippy::type_complexity)]
    fn alignment_program(
        name: &str,
        batcher: &(
             dyn Fn(
            &Program<V, O, Vec<V>, Vec<V>>,
            &[BatchAxis],
            &[Option<BatchAxis>],
            &BatchingLevel<<V as Typed>::Type>,
            CustomRuleBatchedOutputs,
        ) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Vec<BatchAxis>), BatchingError>
                 + 'static
                 + Send
                 + Sync
         ),
        types: &[V::Type],
        input_axes: &[BatchAxis],
        output_axes: &[BatchAxis],
        level: &BatchingLevel<V::Type>,
    ) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError> {
        let mut builder = ProgramBuilder::<V, O>::new();
        let inputs = types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let identity =
            builder.build::<Vec<V>, Vec<V>>(inputs, vec![Placeholder; types.len()], vec![Placeholder; types.len()])?;
        let required_axes = output_axes.iter().copied().map(Some).collect::<Vec<_>>();
        let (alignment, aligned_axes) =
            batcher(&identity, input_axes, &required_axes, level, CustomRuleBatchedOutputs::Values)
                .map_err(ProgramError::from)?;
        if aligned_axes != output_axes {
            return Err(ProgramError::MalformedProgram(format!(
                "batched {} could not align values from the batch axes {:?} to {:?}",
                describe_composite(COMPOSITE_OPERATION_NAME, name),
                input_axes,
                output_axes,
            ))
            .into());
        }
        Ok(alignment)
    }
}

impl<V: Value<Type: DifferentiableType + Eq + Hash>, O: Operation<Type = V::Type>> CustomRuleDefinition<V, O>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    /// Batches `program` at the provided level with the retained structural batcher, rejecting the request when this
    /// definition has no batching support. The batched specialization has the provided `cotangents`, which are those
    /// of the specialization that `program` belongs to (i.e., empty unless it is a backward-rule specialization).
    fn batched_specialization(
        &self,
        program: &Program<V, O, Vec<V>, Vec<V>>,
        input_axes: &[BatchAxis],
        output_axes: &[Option<BatchAxis>],
        level: &CustomRuleBatchingLevel<V::Type>,
        outputs: CustomRuleBatchedOutputs,
        cotangents: Vec<TransposedProgramCotangent>,
    ) -> Result<Arc<CustomRuleSpecialization<V, O>>, DifferentiationError> {
        let Some(batcher) = self.structural_batcher() else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "{} cannot differentiate a batched call because its definition has no batching support \
                     (refer to `CustomRuleDefinition::with_batching_support`)",
                    describe_composite(COMPOSITE_OPERATION_NAME, self.name()),
                ),
            }
            .into());
        };
        let (program, output_axes) =
            batcher(program, input_axes, output_axes, &level.level, outputs).map_err(ProgramError::from)?;
        Ok(Arc::new(CustomRuleSpecialization { program, output_axes, cotangents }))
    }
}

/// Validates that `program`, a retained rule of the custom rule set named `name` traced in its own trace, returns the
/// `expected` types as its leading outputs (or as its outputs at the `selected` positions, when provided), and returns
/// it with its type identities renamed onto those of `expected`. The trace establishes fresh identities for the extents
/// that the rule computes (e.g., a recomputed `n · n` extent of the primal outputs), so its outputs can agree with
/// `expected` only up to a renaming of those identities. The renaming is derived across the complete boundary, so that
/// the identities of the program inputs map to themselves, and it must be bijective, so that it preserves which outputs
/// share an identity. It is applied to the whole program, including any trailing outputs (e.g., residuals) that name
/// the renamed identities.
fn align_rule_output_type_identities<V: Value, O: Operation<Type = V::Type>>(
    name: &str,
    rule: &str,
    program: Program<V, O, Vec<V>, Vec<V>>,
    expected: &[V::Type],
    selected: Option<&[usize]>,
) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    let input_types = program.input_types();
    let output_types = program.output_types();
    let selected_types = selected
        .map(|selected| selected.iter().filter_map(|&index| output_types.get(index).cloned()).collect::<Vec<_>>());
    let actual = match &selected_types {
        Some(selected_types) => selected_types.as_slice(),
        None => output_types.get(..expected.len()).unwrap_or(&output_types),
    };
    let boundary = |outputs: &[V::Type]| input_types.iter().chain(outputs).cloned().collect::<Vec<_>>();
    let Some(renaming) = derive_bijective_identity_renaming(&boundary(actual), &boundary(expected)) else {
        // Signatures that render equally differ only in their identities (e.g., when the rule computes two extents
        // separately that the expected outputs share), which the rendering alone would not show.
        let render = |types: &[V::Type]| types.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ");
        let (expected, actual) = (render(expected), render(actual));
        let reason = if expected == actual { " with different type identities" } else { "" };
        return Err(TypeError::invalid(format!(
            "{} {} rule output type signature mismatch: expected [{}] but got [{}]{}",
            describe_composite(COMPOSITE_OPERATION_NAME, name),
            rule,
            expected,
            actual,
            reason,
        ))
        .into());
    };
    if renaming.is_identity() {
        return Ok(program);
    }
    program.rename_type_identities(&renaming)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType, DimensionBounds,
        DimensionType, DimensionVariable, Shape, ShardingDimension,
    };
    use crate::contexts::Context;
    use crate::operations::arithmetic::MulOperation;
    use crate::operations::composites::rules::registrations::CustomRuleRegistration;
    use crate::operations::composites::tests::{MemberDefinition, RuleCounters, member_cube_definition};
    use crate::programs::ReferenceType;
    use crate::tests::hash_of;

    use super::*;

    /// Returns a registration of `(x, y) ↦ x · y` whose forward mode rule is derived from its primal and whose custom
    /// batching rule maps the product along the first mapped input axis. The registration uses broadcast sensitive
    /// batching when `broadcast_sensitive` is set and has batching support when `batching_support` is set.
    fn product_registration(
        batching_support: bool,
        broadcast_sensitive: bool,
    ) -> CustomRuleRegistration<Array, ArrayOperation<Array>> {
        let rule = |inputs: &[CustomRuleTracer<Array, ArrayOperation<Array>>],
                    axes: &[BatchAxis],
                    _: &BatchingLevel<ArrayType>,
                    _: &[CustomRuleTracer<Array, ArrayOperation<Array>>]| {
            let axis = if axes[0].is_replicated() { axes[1] } else { axes[0] };
            Ok::<_, ProgramError>((vec![inputs[0].clone() * inputs[1].clone()], vec![axis]))
        };
        let mut definition = MemberDefinition::new("product").with_jvp_from_primal();
        if batching_support {
            definition = definition.with_batching_support();
        }
        CustomRuleRegistration::new(match broadcast_sensitive {
            true => definition.with_broadcast_sensitive_batching_rule(rule),
            false => definition.with_batching_rule(rule),
        })
    }

    /// Returns the key of the definition derived from a [`product_registration`] for calls of the provided kind with
    /// the provided active inputs.
    fn product_derivation_key(
        kind: CustomRuleDerivationKind,
        active_input_indices: Vec<usize>,
    ) -> CustomRuleDerivedDefinitionKey {
        CustomRuleDerivedDefinitionKey { kind, input_count: 2, active_input_indices, output_tangent_mask: vec![true] }
    }

    /// Applies the batching rule of the definition derived from `registration` for `derivation_key` directly (i.e.,
    /// without the caches and the discharge of the derived definition) to the call batched as `key` describes.
    fn derived_batching_rule_specialization(
        registration: &CustomRuleRegistration<Array, ArrayOperation<Array>>,
        derivation_key: CustomRuleDerivedDefinitionKey,
        key: &CustomRuleBatchingSpecializationKey<ArrayType>,
    ) -> Result<CustomRuleSpecialization<Array, ArrayOperation<Array>>, DifferentiationError> {
        let derived = registration.reference().derived(derivation_key)?;
        let Some(derivation) = derived.definition().batching_rule().and_then(|rule| rule.derivation()) else {
            panic!("the derived definition has no derivation");
        };
        derivation.batching_rule_specialization(derived.name(), key)
    }

    /// Rules of calls of array definitions that were converted into array IR programs.
    type LiftedArrayRules = LiftedCustomRules<Array, ArrayOperation<Array>>;

    /// Returns `rules` as the rules of the array IR calls into which their calls were converted. Lifted rules serve
    /// every family that contains their member family, so their source and specializer functions name that family.
    fn in_array_ir(
        rules: &LiftedArrayRules,
    ) -> &impl CustomRuleSpecializer<ArrayIrValue<Array>, ArrayIrOperation<Array>> {
        rules
    }

    /// Returns the two lifted forms of the rules of `reference`, whose definition is available and unavailable in the
    /// family into which its calls were converted, respectively.
    fn lifted_forms(reference: &CustomRuleReference<Array, ArrayOperation<Array>>) -> [LiftedArrayRules; 2] {
        [
            LiftedCustomRules::Member(reference.clone()),
            LiftedCustomRules::Unavailable(UnavailableCustomRules::from_source(reference)),
        ]
    }

    /// Returns the specialization key of an unbatched and undischarged call with one active input of type `r#type`
    /// and one output of that same type.
    fn unbatched_key<T: Clone>(r#type: T) -> CustomRuleForwardSpecializationKey<T> {
        CustomRuleForwardSpecializationKey {
            input_types: vec![r#type.clone()],
            output_types: vec![r#type],
            kept_outputs: None,
            non_differentiated_count: 0,
            tangent_activity: vec![true],
            levels: Vec::new(),
            discharged: false,
        }
    }

    /// Returns the specialization key of the unbatched backward rule of a call of the cube (refer to
    /// [`member_cube_definition`]) with one input of type `r#type`, whose residual `x²` is its only leading input and
    /// whose input cotangent has the destination kind `destination_kind`.
    fn cube_backward_key<T: Clone>(
        r#type: T,
        destination_kind: CotangentDestinationKind,
    ) -> CustomRuleBackwardSpecializationKey<T> {
        CustomRuleBackwardSpecializationKey {
            leading_input_types: vec![r#type.clone()],
            seed_extent_count: 0,
            input_tangent_types: vec![r#type.clone()],
            output_tangent_types: vec![r#type.clone()],
            seed_types: vec![Some(r#type)],
            destination_kinds: vec![destination_kind],
            levels: Vec::new(),
            discharged: false,
        }
    }

    /// Returns the specialization key of the custom batching rule of a call of a [`product_registration`] whose `f64[]`
    /// inputs are batched over 3 batch items, the first one mapped along axis 0 and the second one replicated.
    fn product_batching_key<T: From<ArrayType>>() -> CustomRuleBatchingSpecializationKey<T> {
        CustomRuleBatchingSpecializationKey {
            level: BatchingLevel::new(BatchingLevelExtent::Static(3), None, ShardingDimension::Replicated),
            boundary_input_types: Vec::new(),
            input_types: vec![
                ArrayType::new_static(DataType::F64, [3]).into(),
                ArrayType::scalar(DataType::F64).into(),
            ],
            unbatched_input_types: vec![
                ArrayType::scalar(DataType::F64).into(),
                ArrayType::scalar(DataType::F64).into(),
            ],
            input_axes: vec![BatchAxis::new(0), BatchAxis::replicated()],
            discharged: false,
        }
    }

    #[test]
    fn test_custom_rule_source_id() {
        // Every source of a definition reports the identity of that definition, including after its rules became
        // unavailable, while an equally named definition is another rule set.
        let registration = CustomRuleRegistration::new(MemberDefinition::new("cube"));
        let id = registration.definition().id();
        assert_eq!(registration.reference().id(), id);
        for rules in lifted_forms(&registration.reference()) {
            assert_eq!(in_array_ir(&rules).id(), id);
        }
        assert_ne!(CustomRuleRegistration::new(MemberDefinition::new("cube")).reference().id(), id);
    }

    #[test]
    fn test_custom_rule_source_name() {
        let registration = CustomRuleRegistration::new(MemberDefinition::new("cube"));
        assert_eq!(registration.reference().name(), "cube");
        for rules in lifted_forms(&registration.reference()) {
            assert_eq!(in_array_ir(&rules).name(), "cube");
        }
    }

    #[test]
    fn test_custom_rule_source_has_jvp() {
        // Retained forward mode rules, including those that receive structural zeros, count as forward mode rules,
        // while a forward mode rule derived from the primal does not.
        let counters = Arc::new(RuleCounters::default());
        let retained = CustomRuleRegistration::new(member_cube_definition(&counters));
        let symbolic_zero = CustomRuleRegistration::new(
            MemberDefinition::new("identity")
                .with_symbolic_zero_jvp(|primals, tangents| Ok((primals.to_vec(), tangents.to_vec()))),
        );
        let primal = CustomRuleRegistration::new(MemberDefinition::new("identity").with_jvp_from_primal());
        assert!(retained.reference().has_jvp());
        assert!(symbolic_zero.reference().has_jvp());
        assert!(!primal.reference().has_jvp());
        for rules in lifted_forms(&retained.reference()) {
            assert!(in_array_ir(&rules).has_jvp());
        }
        for rules in lifted_forms(&primal.reference()) {
            assert!(!in_array_ir(&rules).has_jvp());
        }
        assert_eq!(counters.counts(), (0, 0, 0));
    }

    #[test]
    fn test_custom_rule_source_derives_jvp_from_primal() {
        let counters = Arc::new(RuleCounters::default());
        let retained = CustomRuleRegistration::new(member_cube_definition(&counters));
        let primal = CustomRuleRegistration::new(MemberDefinition::new("identity").with_jvp_from_primal());
        assert!(primal.reference().derives_jvp_from_primal());
        assert!(!retained.reference().derives_jvp_from_primal());
        for rules in lifted_forms(&primal.reference()) {
            assert!(in_array_ir(&rules).derives_jvp_from_primal());
        }
        for rules in lifted_forms(&retained.reference()) {
            assert!(!in_array_ir(&rules).derives_jvp_from_primal());
        }
    }

    #[test]
    fn test_custom_rule_source_has_vjp() {
        let counters = Arc::new(RuleCounters::default());
        let retained = CustomRuleRegistration::new(member_cube_definition(&counters));
        let without = CustomRuleRegistration::new(MemberDefinition::new("identity").with_jvp_from_primal());
        assert!(retained.reference().has_vjp());
        assert!(!without.reference().has_vjp());
        for rules in lifted_forms(&retained.reference()) {
            assert!(in_array_ir(&rules).has_vjp());
        }
        for rules in lifted_forms(&without.reference()) {
            assert!(!in_array_ir(&rules).has_vjp());
        }
    }

    #[test]
    fn test_custom_rule_source_defers_forward() {
        // A deferred forward rule is reported only for a definition with reverse mode rules, and lifted rules report
        // the deferral of their definition even when they are unavailable.
        let counters = Arc::new(RuleCounters::default());
        let deferred = CustomRuleRegistration::new(member_cube_definition(&counters).with_deferred_forward_rule());
        let eager = CustomRuleRegistration::new(member_cube_definition(&counters));
        let without_vjp = CustomRuleRegistration::new(
            MemberDefinition::new("identity").with_jvp_from_primal().with_deferred_forward_rule(),
        );
        assert!(deferred.reference().defers_forward());
        assert!(!eager.reference().defers_forward());
        assert!(!without_vjp.reference().defers_forward());
        for rules in lifted_forms(&deferred.reference()) {
            assert!(in_array_ir(&rules).defers_forward());
        }
        for rules in lifted_forms(&eager.reference()) {
            assert!(!in_array_ir(&rules).defers_forward());
        }
        assert_eq!(counters.counts(), (0, 0, 0));
    }

    #[test]
    fn test_custom_rule_source_batched_output_axes() {
        let output_axes = vec![BatchAxis::new(0), BatchAxis::replicated()];
        let declared =
            CustomRuleRegistration::new(MemberDefinition::new("pair").with_batched_output_axes(output_axes.clone()));
        let default = CustomRuleRegistration::new(MemberDefinition::new("pair"));
        assert_eq!(declared.reference().batched_output_axes(), Some(output_axes.as_slice()));
        assert_eq!(default.reference().batched_output_axes(), None);
        for rules in lifted_forms(&declared.reference()) {
            assert_eq!(in_array_ir(&rules).batched_output_axes(), Some(output_axes.as_slice()));
        }
        for rules in lifted_forms(&default.reference()) {
            assert_eq!(in_array_ir(&rules).batched_output_axes(), None);
        }
    }

    #[test]
    fn test_custom_rule_source_has_batching_rule() {
        // A custom batching rule and the derivation of a derived definition are batching rules, while sequential
        // batching is not.
        let registration = product_registration(true, false);
        let derived = registration
            .reference()
            .derived(product_derivation_key(CustomRuleDerivationKind::Jvp, vec![0]))
            .unwrap();
        let sequential = CustomRuleRegistration::new(MemberDefinition::new("identity").with_sequential_batching());
        assert!(registration.reference().has_batching_rule());
        assert!(derived.has_batching_rule());
        assert!(!sequential.reference().has_batching_rule());
        for rules in lifted_forms(&registration.reference()) {
            assert!(in_array_ir(&rules).has_batching_rule());
        }
        for rules in lifted_forms(&sequential.reference()) {
            assert!(!in_array_ir(&rules).has_batching_rule());
        }
    }

    #[test]
    fn test_custom_rule_source_has_sequential_batching() {
        let sequential = CustomRuleRegistration::new(MemberDefinition::new("identity").with_sequential_batching());
        let structural = CustomRuleRegistration::new(MemberDefinition::new("identity").with_batching_support());
        assert!(sequential.reference().has_sequential_batching());
        assert!(!structural.reference().has_sequential_batching());
        for rules in lifted_forms(&sequential.reference()) {
            assert!(in_array_ir(&rules).has_sequential_batching());
        }
        for rules in lifted_forms(&structural.reference()) {
            assert!(!in_array_ir(&rules).has_sequential_batching());
        }
    }

    #[test]
    fn test_custom_rule_source_native_definition() {
        // Only a reference has a definition in its own family, while the definition of lifted rules belongs to their
        // member family.
        let registration = CustomRuleRegistration::new(MemberDefinition::new("cube"));
        let reference = registration.reference();
        assert!(std::ptr::eq(reference.native_definition().unwrap(), registration.definition()));
        for rules in lifted_forms(&reference) {
            assert!(in_array_ir(&rules).native_definition().is_none());
        }
    }

    #[test]
    fn test_custom_rule_source_supports_reference_destinations() {
        // Only the backward rules of a reference write caller buffers, and the carriers of lifted rules return those
        // cotangents instead.
        let registration = CustomRuleRegistration::new(MemberDefinition::new("cube"));
        assert!(registration.reference().supports_reference_destinations());
        for rules in lifted_forms(&registration.reference()) {
            assert!(!in_array_ir(&rules).supports_reference_destinations());
        }
    }

    #[test]
    fn test_custom_rule_reference_jvp_specialization() {
        // The program takes every input followed by the active tangents and returns the outputs followed by their
        // tangents, and a repeated request reuses the specialization instead of tracing the rule again.
        let counters = Arc::new(RuleCounters::default());
        let registration = CustomRuleRegistration::new(member_cube_definition(&counters));
        let key = unbatched_key(ArrayType::scalar(DataType::F64));
        let specialization = registration.reference().jvp_specialization(key.clone()).unwrap();
        assert!(specialization.output_axes.is_empty());
        assert_eq!(
            specialization.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %0 %0
                    %3:f64[] = mul %2 %0
                    %4:f64[] = mul %2 %1
                in (%3, %4)
            "}
            .trim_end(),
        );
        assert!(Arc::ptr_eq(&specialization, &registration.reference().jvp_specialization(key).unwrap()));
        assert_eq!(counters.counts(), (1, 0, 0));
    }

    #[test]
    fn test_custom_rule_reference_jvp_specialization_malformed_keys() {
        // A key whose tangent activity does not cover the differentiated inputs, or whose batching level does not
        // describe the batch axes of every input, is malformed.
        let counters = Arc::new(RuleCounters::default());
        let reference = CustomRuleRegistration::new(member_cube_definition(&counters)).reference();
        let scalar_type = ArrayType::scalar(DataType::F64);
        assert!(matches!(
            reference.jvp_specialization(CustomRuleForwardSpecializationKey {
                tangent_activity: vec![true, true],
                ..unbatched_key(scalar_type.clone())
            }),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message)))
                if message == "`composite` `cube` JVP specialization key has 2 tangent activity flags for 1 \
                               differentiated inputs",
        ));
        assert!(matches!(
            reference.jvp_specialization(CustomRuleForwardSpecializationKey {
                input_types: vec![scalar_type.clone(), scalar_type.clone()],
                non_differentiated_count: 1,
                levels: vec![CustomRuleBatchingLevel {
                    level: BatchingLevel::new(BatchingLevelExtent::Static(3), None, ShardingDimension::Replicated),
                    boundary_input_count: 0,
                    input_axes: Vec::new(),
                    output_axes: vec![BatchAxis::new(0)],
                }],
                ..unbatched_key(scalar_type)
            }),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message)))
                if message == "`composite` `cube` batching level has too few input axes",
        ));
    }

    #[test]
    fn test_custom_rule_reference_jvp_specialization_dynamic_extent() {
        // A first-class batch extent reaches batched specializations as a leading boundary input, so the traced rule is
        // batched at a level that records only the extent's type.
        let invocations = Arc::new(AtomicUsize::new(0));
        let definition = CustomRuleDefinition::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new("cube")
            .with_jvp({
                let invocations = invocations.clone();
                move |primals, tangents| {
                    invocations.fetch_add(1, Ordering::SeqCst);
                    let multiply = |left: &CustomRuleTracer<_, _>, right: &CustomRuleTracer<_, _>| {
                        let operation = ArrayIrOperation::Array(ArrayOperation::Mul(MulOperation::new()));
                        Ok::<_, ProgramError>(
                            left.context().bind(operation, Vec::new(), &[left.clone(), right.clone()])?.remove(0),
                        )
                    };
                    let square = multiply(&primals[0], &primals[0])?;
                    Ok((vec![multiply(&square, &primals[0])?], vec![multiply(&square, &tangents[0])?]))
                }
            })
            .with_batching_support();
        let item_type: ArrayIrType = ArrayType::new_static(DataType::F32, [3]).into();
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
        let level = CustomRuleBatchingLevel {
            level: BatchingLevel::new(
                BatchingLevelExtent::Dynamic(DimensionType::from(items).into()),
                None,
                ShardingDimension::Replicated,
            ),
            boundary_input_count: 1,
            input_axes: vec![BatchAxis::new(0)],
            output_axes: vec![BatchAxis::new(0)],
        };
        let key = CustomRuleForwardSpecializationKey {
            input_types: vec![item_type.clone()],
            output_types: vec![item_type],
            kept_outputs: None,
            non_differentiated_count: 0,
            tangent_activity: vec![true],
            levels: vec![level],
            discharged: false,
        };
        let specialization = CustomRuleRegistration::new(definition).reference().jvp_specialization(key).unwrap();
        assert_eq!(specialization.output_axes, vec![BatchAxis::new(0), BatchAxis::new(0)]);
        assert_eq!(
            specialization.program.to_string(),
            indoc! {"
                lambda %0:dimension<items ∈ [1, 9)>, %1:f32[items, 3], %2:f32[items, 3] .
                let %3:f32[items, 3] = mul %1 %1
                    %4:f32[items, 3] = mul %3 %1
                    %5:f32[items, 3] = mul %3 %2
                in (%4, %5)
            "}
            .trim_end(),
        );
        assert_eq!(invocations.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_custom_rule_reference_forward_specialization() {
        // The program takes the call's inputs and returns its outputs followed by the residuals (here, `x²`).
        let counters = Arc::new(RuleCounters::default());
        let registration = CustomRuleRegistration::new(member_cube_definition(&counters));
        let specialization = registration
            .reference()
            .forward_specialization(CustomRuleForwardSpecializationKey {
                tangent_activity: Vec::new(),
                ..unbatched_key(ArrayType::scalar(DataType::F64))
            })
            .unwrap();
        assert!(specialization.output_axes.is_empty());
        assert_eq!(
            specialization.program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = mul %0 %0
                    %2:f64[] = mul %1 %0
                in (%2, %1)
            "}
            .trim_end(),
        );
        assert_eq!(counters.counts(), (0, 1, 0));
    }

    #[test]
    fn test_custom_rule_reference_backward_specialization() {
        // The program consumes the live seed followed by the leading inputs (here, the residual `x²`) and returns the
        // cotangent `2x²ȳ` of the differentiated input.
        let counters = Arc::new(RuleCounters::default());
        let registration = CustomRuleRegistration::new(member_cube_definition(&counters));
        let specialization = registration
            .reference()
            .backward_specialization(cube_backward_key(
                ArrayType::scalar(DataType::F64),
                CotangentDestinationKind::Return,
            ))
            .unwrap();
        assert_eq!(specialization.cotangents, vec![TransposedProgramCotangent::Output]);
        assert_eq!(
            specialization.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %0
                    %3:f64[] = add %2 %2
                in (%3)
            "}
            .trim_end(),
        );
        assert_eq!(counters.counts(), (0, 0, 1));
    }

    #[test]
    fn test_custom_rule_reference_batching_rule_specialization() {
        // The program takes the batched inputs and returns the batched outputs of the rule, with the batch axes that
        // the rule declares.
        let registration = product_registration(true, false);
        let specialization = registration.reference().batching_rule_specialization(product_batching_key()).unwrap();
        assert_eq!(specialization.output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            specialization.program.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[] .
                let %2:f64[3] = mul %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_rule_reference_sequential_batching_scan() {
        // The scan carries the replicated inputs and stacks the outputs, and its body receives an `i64[]` slice index
        // first. A source that does not batch its calls sequentially has no scan.
        let sequential = CustomRuleRegistration::new(MemberDefinition::new("square").with_sequential_batching());
        let (operation, index_type) =
            sequential.reference().sequential_batching_scan(3, 1, &ShardingDimension::Replicated, 2).unwrap();
        assert_eq!(operation.to_string(), "scan [carry_count=1, iteration_count=3, reverse=false]");
        assert_eq!(index_type, ArrayType::scalar(DataType::I64));
        assert!(matches!(
            CustomRuleRegistration::new(MemberDefinition::new("square"))
                .reference()
                .sequential_batching_scan(3, 1, &ShardingDimension::Replicated, 2),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`composite` `square` does not batch its calls sequentially",
        ));
    }

    #[test]
    fn test_custom_rule_reference_derived() {
        // Derivations of one key share a definition that the source registration owns, together with its caches, and
        // the definition keeps only its source definition alive (i.e., there is no strong cycle back to the caches).
        let registration = CustomRuleRegistration::new(
            MemberDefinition::new("sine")
                .with_jvp_from_primal()
                .with_batching_support()
                .with_batching_rule(|inputs, input_axes, _, _| Ok((inputs.to_vec(), input_axes.to_vec()))),
        );
        let key = |kind| CustomRuleDerivedDefinitionKey {
            kind,
            input_count: 1,
            active_input_indices: vec![0],
            output_tangent_mask: vec![true],
        };
        let derived = registration.reference().derived(key(CustomRuleDerivationKind::Jvp)).unwrap();
        assert_eq!(derived.name(), "jvp(sine)");
        assert!(derived.derives_jvp_from_primal());
        assert!(derived.has_batching_rule());
        assert!(!derived.has_jvp());
        assert!(!derived.has_vjp());
        assert_eq!(registration.reference().derived(key(CustomRuleDerivationKind::Jvp)).unwrap(), derived);
        let pushforward = registration.reference().derived(key(CustomRuleDerivationKind::Pushforward)).unwrap();
        assert_eq!(pushforward.name(), "pushforward(sine)");
        assert_ne!(pushforward, derived);
        assert!(derived.is_cached());
        assert!(pushforward.is_cached());

        // Dropping the source registration frees the derived registrations and their caches, and later derivations
        // reuse the derived definitions that are still alive, without caches, exactly like specializations.
        let source = registration.reference();
        drop(registration);
        assert!(!derived.is_cached());
        let fresh = source.derived(key(CustomRuleDerivationKind::Jvp)).unwrap();
        assert_eq!(fresh, derived);
        assert!(!fresh.is_cached());

        // Nothing but the derived references keeps a derived definition alive, so once they are dropped, a later
        // derivation creates a new definition with a new identity.
        let derived_id = derived.definition().id();
        drop((derived, pushforward, fresh));
        assert_ne!(source.derived(key(CustomRuleDerivationKind::Jvp)).unwrap().definition().id(), derived_id);

        // A source without a custom batching rule has nothing to derive from.
        assert!(matches!(
            CustomRuleRegistration::new(MemberDefinition::new("sine").with_jvp_from_primal())
                .reference()
                .derived(key(CustomRuleDerivationKind::Jvp)),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`composite` `sine` has neither a custom batching rule nor sequential batching from \
                               which to derive rules",
        ));
    }

    #[test]
    fn test_unavailable_custom_rules_from_source() {
        // Unavailable rules keep the identity and the description of their source.
        let counters = Arc::new(RuleCounters::default());
        let registration = CustomRuleRegistration::new(
            member_cube_definition(&counters)
                .with_deferred_forward_rule()
                .with_batched_output_axes(vec![BatchAxis::replicated()])
                .with_sequential_batching(),
        );
        let source = registration.reference();
        let unavailable = UnavailableCustomRules::from_source(&source);
        assert_eq!(unavailable.id, source.id());
        assert_eq!(unavailable.name, "cube");
        assert!(unavailable.has_jvp);
        assert!(!unavailable.derives_jvp_from_primal);
        assert!(unavailable.has_vjp);
        assert!(unavailable.defers_forward);
        assert_eq!(unavailable.batched_output_axes, Some(vec![BatchAxis::replicated()]));
        assert!(!unavailable.has_batching_rule);
        assert!(unavailable.has_sequential_batching);

        let registration = CustomRuleRegistration::new(
            MemberDefinition::new("sine")
                .with_jvp_from_primal()
                .with_batching_rule(|inputs, input_axes, _, _| Ok((inputs.to_vec(), input_axes.to_vec()))),
        );
        let source = registration.reference();
        let unavailable = UnavailableCustomRules::from_source(&source);
        assert_eq!(unavailable.id, source.id());
        assert_eq!(unavailable.name, "sine");
        assert!(!unavailable.has_jvp);
        assert!(unavailable.derives_jvp_from_primal);
        assert!(!unavailable.has_vjp);
        assert!(!unavailable.defers_forward);
        assert_eq!(unavailable.batched_output_axes, None);
        assert!(unavailable.has_batching_rule);
        assert!(!unavailable.has_sequential_batching);
        assert_eq!(counters.counts(), (0, 0, 0));
    }

    #[test]
    fn test_lifted_custom_rules_id() {
        // Lifted rules have the identity of their definition, whether or not its rules are available.
        let registration = CustomRuleRegistration::new(MemberDefinition::new("cube"));
        let id = registration.definition().id();
        for rules in lifted_forms(&registration.reference()) {
            assert_eq!(rules.id(), id);
        }
        let other = CustomRuleRegistration::new(MemberDefinition::new("cube"));
        assert_ne!(LiftedArrayRules::Member(other.reference()).id(), id);
    }

    #[test]
    fn test_lifted_custom_rules_member() {
        // The rules of an available member definition are its reference, while unavailable rules explain why their
        // retained rules cannot be used.
        let registration = CustomRuleRegistration::new(MemberDefinition::new("cube"));
        let [member, unavailable] = lifted_forms(&registration.reference());
        assert_eq!(member.member(), Ok(&registration.reference()));
        assert!(matches!(
            unavailable.member(),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`composite` `cube` was converted from a family whose values its current family \
                               cannot represent, so its derivative rules are unavailable",
        ));
    }

    #[test]
    fn test_lifted_custom_rules_clone() {
        // A clone keeps the availability of the rules and the identity of their definition.
        let registration = CustomRuleRegistration::new(MemberDefinition::new("cube"));
        let [member, unavailable] = lifted_forms(&registration.reference());
        assert!(matches!(member.clone(), LiftedCustomRules::Member(rules) if rules == registration.reference()));
        assert!(matches!(unavailable.clone(), LiftedCustomRules::Unavailable(rules) if rules.id == unavailable.id()));
    }

    #[test]
    fn test_lifted_custom_rules_debug() {
        let registration = CustomRuleRegistration::new(MemberDefinition::new("cube"));
        let id = registration.definition().id();
        let [member, unavailable] = lifted_forms(&registration.reference());
        assert_eq!(
            format!("{member:?}"),
            "Member(CustomRuleReference(CustomRuleDefinition { name: \"cube\", has_jvp: false, \
             derives_jvp_from_primal: false, has_vjp: false, .. }))",
        );
        assert_eq!(
            format!("{unavailable:?}"),
            format!(
                "Unavailable(UnavailableCustomRules {{ id: {id}, name: \"cube\", has_jvp: false, \
                 derives_jvp_from_primal: false, has_vjp: false, defers_forward: false, batched_output_axes: None, \
                 has_batching_rule: false, has_sequential_batching: false }})",
            ),
        );
    }

    #[test]
    fn test_lifted_custom_rules_eq_and_hash() {
        // Lifted rules compare and hash by the identity of their definition, so availability does not change which map
        // entry they address, while the rules of an equally named definition differ.
        let registration = CustomRuleRegistration::new(MemberDefinition::new("cube"));
        let [member, unavailable] = lifted_forms(&registration.reference());
        let other = LiftedArrayRules::Member(CustomRuleRegistration::new(MemberDefinition::new("cube")).reference());
        assert_eq!(member, member);
        assert_eq!(member, unavailable);
        assert_ne!(member, other);
        assert_eq!(hash_of(&member), hash_of(&unavailable));
        let definitions = HashMap::from([(member, 1), (other, 2)]);
        assert_eq!(definitions.get(&unavailable), Some(&1));
        assert_eq!(definitions.len(), 2);
    }

    #[test]
    fn test_lifted_custom_rules_jvp_specialization() {
        // The member definition specializes its rule at the projected types, and its program is converted into the
        // requesting family.
        let counters = Arc::new(RuleCounters::default());
        let registration = CustomRuleRegistration::new(member_cube_definition(&counters));
        let [member, unavailable] = lifted_forms(&registration.reference());
        let scalar_type = ArrayIrType::from(ArrayType::scalar(DataType::F64));
        let specialization = in_array_ir(&member).jvp_specialization(unbatched_key(scalar_type.clone())).unwrap();
        assert_eq!(
            specialization.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %0 %0
                    %3:f64[] = mul %2 %0
                    %4:f64[] = mul %2 %1
                in (%3, %4)
            "}
            .trim_end(),
        );
        assert_eq!(counters.counts(), (1, 0, 0));

        // Types without a member representation are rejected at the first derivative request.
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F64)));
        assert!(matches!(
            in_array_ir(&member).jvp_specialization(unbatched_key(reference_type)),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`composite` `cube` cannot be differentiated with respect to type `ref<f64[]>`, which \
                               the family in which its rules were registered cannot represent",
        ));

        // So are first-class dimension batch extents, which array IR batching produces.
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
        let level = CustomRuleBatchingLevel {
            level: BatchingLevel::new(
                BatchingLevelExtent::Dynamic(DimensionType::from(items).into()),
                None,
                ShardingDimension::Replicated,
            ),
            boundary_input_count: 1,
            input_axes: vec![BatchAxis::new(0)],
            output_axes: vec![BatchAxis::new(0)],
        };
        assert!(matches!(
            in_array_ir(&member).jvp_specialization(CustomRuleForwardSpecializationKey {
                levels: vec![level],
                ..unbatched_key(scalar_type.clone())
            }),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`composite` `cube` cannot differentiate a batched call whose batch extent has type \
                               `dimension<items ∈ [1, 9)>`, which the family in which its rules were registered \
                               cannot represent; differentiate the call before batching it instead",
        ));

        // Unavailable rules reject every request to specialize their rules.
        assert!(matches!(
            in_array_ir(&unavailable).jvp_specialization(unbatched_key(scalar_type)),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`composite` `cube` was converted from a family whose values its current family \
                               cannot represent, so its derivative rules are unavailable",
        ));
        assert_eq!(counters.counts(), (1, 0, 0));
    }

    #[test]
    fn test_lifted_custom_rules_jvp_specialization_cache() {
        // The converted program of each specialization is cached with the member definition's specializations, so
        // repeated requests share it instead of converting the member program again.
        let counters = Arc::new(RuleCounters::default());
        let registration = CustomRuleRegistration::new(member_cube_definition(&counters));
        let member = LiftedArrayRules::Member(registration.reference());
        let key = unbatched_key(ArrayIrType::from(ArrayType::scalar(DataType::F64)));
        let first = in_array_ir(&member).jvp_specialization(key.clone()).unwrap();
        let second = in_array_ir(&member).jvp_specialization(key).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(counters.counts(), (1, 0, 0));
    }

    #[test]
    fn test_lifted_custom_rules_forward_specialization() {
        let counters = Arc::new(RuleCounters::default());
        let registration = CustomRuleRegistration::new(member_cube_definition(&counters));
        let member = LiftedArrayRules::Member(registration.reference());
        let specialization = in_array_ir(&member)
            .forward_specialization(CustomRuleForwardSpecializationKey {
                tangent_activity: Vec::new(),
                ..unbatched_key(ArrayIrType::from(ArrayType::scalar(DataType::F64)))
            })
            .unwrap();
        assert_eq!(
            specialization.program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = mul %0 %0
                    %2:f64[] = mul %1 %0
                in (%2, %1)
            "}
            .trim_end(),
        );
        assert_eq!(counters.counts(), (0, 1, 0));
    }

    #[test]
    fn test_lifted_custom_rules_backward_specialization() {
        let counters = Arc::new(RuleCounters::default());
        let registration = CustomRuleRegistration::new(member_cube_definition(&counters));
        let member = LiftedArrayRules::Member(registration.reference());
        let specialization = in_array_ir(&member)
            .backward_specialization(cube_backward_key(
                ArrayIrType::from(ArrayType::scalar(DataType::F64)),
                CotangentDestinationKind::Return,
            ))
            .unwrap();
        assert_eq!(specialization.cotangents, vec![TransposedProgramCotangent::Output]);
        assert_eq!(
            specialization.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %0
                    %3:f64[] = add %2 %2
                in (%3)
            "}
            .trim_end(),
        );
        assert_eq!(counters.counts(), (0, 0, 1));
    }

    #[test]
    fn test_lifted_custom_rules_backward_specialization_reference_destination() {
        // The carriers of converted calls request returned cotangents for caller buffers (refer to
        // `supports_reference_destinations`), so a caller buffer destination means that a carrier skipped that step.
        let counters = Arc::new(RuleCounters::default());
        let registration = CustomRuleRegistration::new(member_cube_definition(&counters));
        let member = LiftedArrayRules::Member(registration.reference());
        assert!(matches!(
            in_array_ir(&member).backward_specialization(cube_backward_key(
                ArrayIrType::from(ArrayType::scalar(DataType::F64)),
                CotangentDestinationKind::Reference,
            )),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message)))
                if message == "`composite_transpose` `cube` requested a caller-buffer destination from rules that \
                               cannot write caller buffers",
        ));
        assert_eq!(counters.counts(), (0, 0, 0));
    }

    #[test]
    fn test_lifted_custom_rules_batching_rule_specialization() {
        let registration = product_registration(true, false);
        let member = LiftedArrayRules::Member(registration.reference());
        let specialization = in_array_ir(&member).batching_rule_specialization(product_batching_key()).unwrap();
        assert_eq!(specialization.output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            specialization.program.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[] .
                let %2:f64[3] = mul %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_lifted_custom_rules_batching_rule_specialization_unrepresentable_type() {
        // A batched input whose type has no member representation cannot be batched by the member rule.
        let registration = product_registration(true, false);
        let member = LiftedArrayRules::Member(registration.reference());
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F64)));
        let mut key = product_batching_key::<ArrayIrType>();
        key.unbatched_input_types[1] = reference_type.clone();
        key.input_types[1] = reference_type;
        assert!(matches!(
            in_array_ir(&member).batching_rule_specialization(key),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`composite` `product` cannot batch a call over type `ref<f64[]>`, which the family in \
                               which its rules were registered cannot represent",
        ));
    }

    #[test]
    fn test_lifted_custom_rules_sequential_batching_scan() {
        // The member definition constructs the scan in its own family, which is converted into the requesting family.
        let registration = CustomRuleRegistration::new(MemberDefinition::new("square").with_sequential_batching());
        let member = LiftedArrayRules::Member(registration.reference());
        let (operation, index_type) =
            in_array_ir(&member).sequential_batching_scan(3, 1, &ShardingDimension::Replicated, 2).unwrap();
        assert_eq!(operation.to_string(), "scan [carry_count=1, iteration_count=3, reverse=false]");
        assert_eq!(index_type, ArrayIrType::from(ArrayType::scalar(DataType::I64)));
    }

    #[test]
    fn test_lifted_custom_rules_derived() {
        // A lifted source derives in the family that owns its definition, and unavailable rules stay unavailable.
        let registration = CustomRuleRegistration::new(
            MemberDefinition::new("sine")
                .with_jvp_from_primal()
                .with_batching_support()
                .with_batching_rule(|inputs, input_axes, _, _| Ok((inputs.to_vec(), input_axes.to_vec()))),
        );
        let key = CustomRuleDerivedDefinitionKey {
            kind: CustomRuleDerivationKind::Jvp,
            input_count: 1,
            active_input_indices: vec![0],
            output_tangent_mask: vec![true],
        };
        let derive = |rules: &LiftedCustomRules<Array, ArrayOperation<Array>>| {
            <LiftedCustomRules<_, _> as CustomRuleSpecializer<ArrayIrValue<Array>, ArrayIrOperation<Array>>>::derived(
                rules,
                key.clone(),
            )
            .unwrap()
        };
        let lifted = LiftedCustomRules::Member(registration.reference());
        assert_eq!(derive(&lifted), LiftedCustomRules::Member(registration.reference().derived(key.clone()).unwrap()));
        let unavailable = LiftedCustomRules::<Array, ArrayOperation<Array>>::Unavailable(
            UnavailableCustomRules::from_source(&registration.reference()),
        );
        assert!(matches!(derive(&unavailable), LiftedCustomRules::Unavailable(_)));
        assert_eq!(derive(&unavailable), unavailable);
    }

    #[test]
    fn test_custom_rule_derivation_batching_rule_specialization() {
        // Structural mode: with replicated primals, the source rule applied to them is replicated as well, so the JVP
        // of its program is batched structurally over the mapped tangents (e.g., in a forward mode Jacobian), which
        // gives `ẋy + xẏ` per item.
        let registration = product_registration(true, false);
        let scalar_type = ArrayType::scalar(DataType::F64);
        let vector_type = ArrayType::new_static(DataType::F64, [3]);
        let (mapped, replicated) = (BatchAxis::new(0), BatchAxis::replicated());
        let scalar = |value: f64| Array::scalar(value).unwrap();
        let vector = |values: Vec<f64>| Array::vector(values).unwrap();
        let specialization = derived_batching_rule_specialization(
            &registration,
            product_derivation_key(CustomRuleDerivationKind::Jvp, vec![0, 1]),
            &CustomRuleBatchingSpecializationKey {
                level: BatchingLevel::new(BatchingLevelExtent::Static(3), None, ShardingDimension::Replicated),
                boundary_input_types: Vec::new(),
                input_types: vec![scalar_type.clone(), scalar_type.clone(), vector_type.clone(), vector_type],
                unbatched_input_types: vec![scalar_type; 4],
                input_axes: vec![replicated, replicated, mapped, mapped],
                discharged: false,
            },
        )
        .unwrap();
        assert_eq!(specialization.output_axes, vec![replicated, mapped]);
        assert_eq!(
            specialization.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[3], %3:f64[3] .
                let %4:f64[] = mul %0 %1
                    %5:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %1
                    %6:f64[3] = mul %5 %2
                    %7:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %0
                    %8:f64[3] = mul %7 %3
                    %9:f64[3] = add %6 %8
                in (%4, %9)
            "}
            .trim_end(),
        );
        assert_eq!(
            specialization.program.interpret(vec![
                scalar(2.0),
                scalar(3.0),
                vector(vec![1.0, 2.0, 3.0]),
                vector(vec![1.0, 0.0, 1.0]),
            ]),
            Ok(vec![scalar(6.0), vector(vec![5.0, 6.0, 11.0])]),
        );
    }

    #[test]
    fn test_custom_rule_derivation_batching_rule_specialization_rule_mode() {
        // Rule mode without alignment: the tangent of the mapped `x` follows its primal, so the JVP of the source rule
        // applied at the primal axes computes both the primal outputs and their tangents `ẋy`.
        let registration = product_registration(true, false);
        let scalar_type = ArrayType::scalar(DataType::F64);
        let vector_type = ArrayType::new_static(DataType::F64, [3]);
        let (mapped, replicated) = (BatchAxis::new(0), BatchAxis::replicated());
        let specialization = derived_batching_rule_specialization(
            &registration,
            product_derivation_key(CustomRuleDerivationKind::Jvp, vec![0]),
            &CustomRuleBatchingSpecializationKey {
                level: BatchingLevel::new(BatchingLevelExtent::Static(3), None, ShardingDimension::Replicated),
                boundary_input_types: Vec::new(),
                input_types: vec![vector_type.clone(), scalar_type.clone(), vector_type],
                unbatched_input_types: vec![scalar_type; 3],
                input_axes: vec![mapped, replicated, mapped],
                discharged: false,
            },
        )
        .unwrap();
        assert_eq!(specialization.output_axes, vec![mapped, mapped]);
        assert_eq!(
            specialization.program.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[], %2:f64[3] .
                let %3:f64[3] = mul %0 %1
                    %4:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %1
                    %5:f64[3] = mul %4 %2
                in (%3, %5)
            "}
            .trim_end(),
        );
        let vector = |values: Vec<f64>| Array::vector(values).unwrap();
        assert_eq!(
            specialization.program.interpret(vec![
                vector(vec![1.0, 2.0, 3.0]),
                Array::scalar(3.0).unwrap(),
                vector(vec![1.0, 1.0, 1.0]),
            ]),
            Ok(vec![vector(vec![3.0, 6.0, 9.0]), vector(vec![3.0, 3.0, 3.0])]),
        );
    }

    #[test]
    fn test_custom_rule_derivation_batching_rule_specialization_alignment() {
        // A tangent mapped along another axis than its primal is moved to the primal's axis before the JVP of the
        // source rule is applied, so item `i` of the tangent `ẋy` is column `i` of `ẋ` scaled by `y`.
        let registration = product_registration(true, false);
        let item_type = ArrayType::new_static(DataType::F64, [2]);
        let scalar_type = ArrayType::scalar(DataType::F64);
        let specialization = derived_batching_rule_specialization(
            &registration,
            product_derivation_key(CustomRuleDerivationKind::Jvp, vec![0]),
            &CustomRuleBatchingSpecializationKey {
                level: BatchingLevel::new(BatchingLevelExtent::Static(3), None, ShardingDimension::Replicated),
                boundary_input_types: Vec::new(),
                input_types: vec![
                    ArrayType::new_static(DataType::F64, [3, 2]),
                    scalar_type.clone(),
                    ArrayType::new_static(DataType::F64, [2, 3]),
                ],
                unbatched_input_types: vec![item_type.clone(), scalar_type, item_type],
                input_axes: vec![BatchAxis::new(0), BatchAxis::replicated(), BatchAxis::new(1)],
                discharged: false,
            },
        )
        .unwrap();
        assert_eq!(specialization.output_axes, vec![BatchAxis::new(0), BatchAxis::new(0)]);
        assert_eq!(
            specialization.program.to_string(),
            indoc! {"
                lambda %0:f64[3, 2], %1:f64[], %2:f64[2, 3] .
                let %3:f64[3, 2] = transpose [permutation=[1, 0]] %2
                    %4:f64[3, 2] = mul %0 %1
                    %5:f64[3, 2] = broadcast [output_type=f64[3, 2], output_axes=[]] %1
                    %6:f64[3, 2] = mul %5 %3
                in (%4, %6)
            "}
            .trim_end(),
        );
        assert_eq!(
            specialization.program.interpret(vec![
                Array::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
                Array::scalar(2.0).unwrap(),
                Array::matrix(2, 3, vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]).unwrap(),
            ]),
            Ok(vec![
                Array::matrix(3, 2, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]).unwrap(),
                Array::matrix(3, 2, vec![2.0, 0.0, 0.0, 2.0, 2.0, 0.0]).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_custom_rule_derivation_batching_rule_specialization_separate_primal() {
        // A replicated `y` whose tangent is mapped is broadcast for the tangents `xẏ`, so the primal outputs come from
        // a separate application of the source rule at the primal axes. A pushforward computes only the tangents, so
        // it needs no separate application.
        let registration = product_registration(true, false);
        let scalar_type = ArrayType::scalar(DataType::F64);
        let vector_type = ArrayType::new_static(DataType::F64, [3]);
        let (mapped, replicated) = (BatchAxis::new(0), BatchAxis::replicated());
        let key = CustomRuleBatchingSpecializationKey {
            level: BatchingLevel::new(BatchingLevelExtent::Static(3), None, ShardingDimension::Replicated),
            boundary_input_types: Vec::new(),
            input_types: vec![vector_type.clone(), scalar_type.clone(), vector_type],
            unbatched_input_types: vec![scalar_type; 3],
            input_axes: vec![mapped, replicated, mapped],
            discharged: false,
        };
        let vector = |values: Vec<f64>| Array::vector(values).unwrap();
        let inputs = vec![vector(vec![1.0, 2.0, 3.0]), Array::scalar(3.0).unwrap(), vector(vec![1.0, 0.0, 2.0])];
        let jvp = derived_batching_rule_specialization(
            &registration,
            product_derivation_key(CustomRuleDerivationKind::Jvp, vec![1]),
            &key,
        )
        .unwrap();
        assert_eq!(jvp.output_axes, vec![mapped, mapped]);
        assert_eq!(
            jvp.program.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[], %2:f64[3] .
                let %3:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %1
                    %4:f64[3] = mul %0 %1
                    %5:f64[3] = mul %0 %2
                in (%4, %5)
            "}
            .trim_end(),
        );
        assert_eq!(
            jvp.program.interpret(inputs.clone()),
            Ok(vec![vector(vec![3.0, 6.0, 9.0]), vector(vec![1.0, 0.0, 6.0])])
        );
        let pushforward = derived_batching_rule_specialization(
            &registration,
            product_derivation_key(CustomRuleDerivationKind::Pushforward, vec![1]),
            &key,
        )
        .unwrap();
        assert_eq!(pushforward.output_axes, vec![mapped]);
        assert_eq!(
            pushforward.program.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[], %2:f64[3] .
                let %3:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %1
                    %4:f64[3] = mul %0 %2
                in (%4)
            "}
            .trim_end(),
        );
        assert_eq!(pushforward.program.interpret(inputs), Ok(vec![vector(vec![1.0, 0.0, 6.0])]));
    }

    #[test]
    fn test_custom_rule_derivation_batching_rule_specialization_without_batching_support() {
        // A derivation batches the source rule's derivative with the source's structural batcher, which it cannot do
        // without it.
        let registration = product_registration(false, false);
        let scalar_type = ArrayType::scalar(DataType::F64);
        let vector_type = ArrayType::new_static(DataType::F64, [3]);
        assert!(matches!(
            derived_batching_rule_specialization(
                &registration,
                product_derivation_key(CustomRuleDerivationKind::Jvp, vec![0]),
                &CustomRuleBatchingSpecializationKey {
                    level: BatchingLevel::new(BatchingLevelExtent::Static(3), None, ShardingDimension::Replicated),
                    boundary_input_types: Vec::new(),
                    input_types: vec![vector_type.clone(), scalar_type.clone(), vector_type],
                    unbatched_input_types: vec![scalar_type; 3],
                    input_axes: vec![BatchAxis::new(0), BatchAxis::replicated(), BatchAxis::new(0)],
                    discharged: false,
                },
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`composite` `jvp(product)` cannot batch a derivative because its source definition has \
                               no batching support (refer to `CustomRuleDefinition::with_batching_support`)",
        ));
    }

    #[test]
    fn test_custom_rule_derivation_batching_rule_specialization_output_tangent_mismatch() {
        // The derivative of the source rule must have live tangents for exactly the outputs whose tangents the derived
        // calls declare.
        let registration = product_registration(true, false);
        let scalar_type = ArrayType::scalar(DataType::F64);
        let vector_type = ArrayType::new_static(DataType::F64, [3]);
        assert!(matches!(
            derived_batching_rule_specialization(
                &registration,
                CustomRuleDerivedDefinitionKey {
                    output_tangent_mask: vec![false],
                    ..product_derivation_key(CustomRuleDerivationKind::Jvp, vec![0])
                },
                &CustomRuleBatchingSpecializationKey {
                    level: BatchingLevel::new(BatchingLevelExtent::Static(3), None, ShardingDimension::Replicated),
                    boundary_input_types: Vec::new(),
                    input_types: vec![vector_type.clone(), scalar_type.clone(), vector_type],
                    unbatched_input_types: vec![scalar_type; 3],
                    input_axes: vec![BatchAxis::new(0), BatchAxis::replicated(), BatchAxis::new(0)],
                    discharged: false,
                },
            ),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "`composite` `jvp(product)` batching rule program has live tangents for the outputs \
                               [true], but the derivative of its primal has them for the outputs [false]",
        ));
    }

    #[test]
    fn test_custom_rule_derivation_broadcast_sensitive_batching_rule_specialization() {
        // A broadcast sensitive source rule is applied once, at the primal axes, and the tangent `xẏ` of the replicated
        // `y` is the diagonal of the outer product of its mapped tangent with the rule's batch.
        let registration = product_registration(true, true);
        let scalar_type = ArrayType::scalar(DataType::F64);
        let vector_type = ArrayType::new_static(DataType::F64, [3]);
        let (mapped, replicated) = (BatchAxis::new(0), BatchAxis::replicated());
        let vector = |values: Vec<f64>| Array::vector(values).unwrap();
        let primals = vec![vector(vec![1.0, 2.0, 3.0]), Array::scalar(3.0).unwrap()];
        let specialization = derived_batching_rule_specialization(
            &registration,
            product_derivation_key(CustomRuleDerivationKind::Jvp, vec![1]),
            &CustomRuleBatchingSpecializationKey {
                level: BatchingLevel::new(BatchingLevelExtent::Static(3), None, ShardingDimension::Replicated),
                boundary_input_types: Vec::new(),
                input_types: vec![vector_type.clone(), scalar_type.clone(), vector_type.clone()],
                unbatched_input_types: vec![scalar_type.clone(); 3],
                input_axes: vec![mapped, replicated, mapped],
                discharged: false,
            },
        )
        .unwrap();
        assert_eq!(specialization.output_axes, vec![mapped, mapped]);
        assert_eq!(
            specialization.program.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[], %2:f64[3] .
                let %3:f64[3] = mul %0 %1
                    %4:f64[3, 3] = broadcast [output_type=f64[3, 3], output_axes=[1]] %0
                    %5:f64[3, 3] = broadcast [output_type=f64[3, 3], output_axes=[0]] %2
                    %6:f64[3, 3] = mul %4 %5
                    %7:i64[3, 1] = iota [type=i64[3, 1], dimension=0]
                    %8:f64[3] = gather [
                        dimensions=(offset=[], collapsed_slice=[0], start_index_map=[0], batching=[(1, 0)]),
                        slice_sizes=[1, 1],
                    ] %6 %7
                in (%3, %8)
            "}
            .trim_end(),
        );
        let mut inputs = primals.clone();
        inputs.push(vector(vec![1.0, 0.0, 2.0]));
        assert_eq!(
            specialization.program.interpret(inputs),
            Ok(vec![vector(vec![3.0, 6.0, 9.0]), vector(vec![1.0, 0.0, 6.0])]),
        );

        // With the tangent of the mapped `x` active as well, the following contribution `ẋy` is added to the
        // tangent-only contribution `xẏ`.
        let specialization = derived_batching_rule_specialization(
            &registration,
            product_derivation_key(CustomRuleDerivationKind::Jvp, vec![0, 1]),
            &CustomRuleBatchingSpecializationKey {
                level: BatchingLevel::new(BatchingLevelExtent::Static(3), None, ShardingDimension::Replicated),
                boundary_input_types: Vec::new(),
                input_types: vec![vector_type.clone(), scalar_type.clone(), vector_type.clone(), vector_type],
                unbatched_input_types: vec![scalar_type; 4],
                input_axes: vec![mapped, replicated, mapped, mapped],
                discharged: false,
            },
        )
        .unwrap();
        assert_eq!(specialization.output_axes, vec![mapped, mapped]);
        assert_eq!(
            specialization.program.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[], %2:f64[3], %3:f64[3] .
                let %4:f64[3] = mul %0 %1
                    %5:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %1
                    %6:f64[3] = mul %5 %2
                    %7:f64[3, 3] = broadcast [output_type=f64[3, 3], output_axes=[1]] %0
                    %8:f64[3, 3] = broadcast [output_type=f64[3, 3], output_axes=[0]] %3
                    %9:f64[3, 3] = mul %7 %8
                    %10:i64[3, 1] = iota [type=i64[3, 1], dimension=0]
                    %11:f64[3] = gather [
                        dimensions=(offset=[], collapsed_slice=[0], start_index_map=[0], batching=[(1, 0)]),
                        slice_sizes=[1, 1],
                    ] %9 %10
                    %12:f64[3] = add %6 %11
                in (%4, %12)
            "}
            .trim_end(),
        );
        let mut inputs = primals;
        inputs.extend([vector(vec![1.0, 1.0, 1.0]), vector(vec![1.0, 0.0, 2.0])]);
        assert_eq!(
            specialization.program.interpret(inputs),
            Ok(vec![vector(vec![3.0, 6.0, 9.0]), vector(vec![4.0, 3.0, 9.0])]),
        );
    }

    #[test]
    fn test_align_rule_output_type_identities() {
        // A rule program whose outputs have the expected types is returned unchanged. Only its leading outputs (e.g.,
        // the primal outputs of a forward rule, which its residuals follow) or the selected ones are compared.
        let (f64_type, f32_type) = (ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F32));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = vec![builder.add_input(f64_type.clone()), builder.add_input(f32_type.clone())];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(inputs, vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();
        let rendering = program.to_string();
        let align = |expected: &[ArrayType], selected: Option<&[usize]>| {
            align_rule_output_type_identities("pair", "forward", program.clone(), expected, selected)
                .map(|program| program.to_string())
        };
        assert_eq!(align(&[f64_type.clone(), f32_type.clone()], None), Ok(rendering.clone()));
        assert_eq!(align(std::slice::from_ref(&f64_type), None), Ok(rendering.clone()));
        assert_eq!(align(std::slice::from_ref(&f32_type), Some(&[1])), Ok(rendering));
        assert!(matches!(
            align(std::slice::from_ref(&f32_type), None),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`composite` `pair` forward rule output type signature mismatch: expected [f32[]] but \
                               got [f64[]]",
        ));
    }

    #[test]
    fn test_align_rule_output_type_identities_different_type_identities() {
        // Outputs that render like the expected types but do not share their identities are rejected with a diagnostic
        // that names the identities, which the renderings alone would not show.
        let bounds = DimensionBounds::new(1, Some(9)).unwrap();
        let vector_type =
            |variable: &DimensionVariable| ArrayType::new(DataType::F32, Shape::new(vec![variable.clone().into()]));
        let (first, second) = (DimensionVariable::new("n", bounds), DimensionVariable::new("n", bounds));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = vec![builder.add_input(vector_type(&first)), builder.add_input(vector_type(&second))];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(inputs, vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();
        assert!(matches!(
            align_rule_output_type_identities(
                "pair",
                "JVP",
                program,
                &[vector_type(&first), vector_type(&first)],
                None,
            ),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`composite` `pair` JVP rule output type signature mismatch: expected [f32[n], f32[n]] \
                               but got [f32[n], f32[n]] with different type identities",
        ));
    }
}
