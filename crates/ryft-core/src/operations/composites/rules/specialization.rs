//! The sealed [`CustomRuleSource`] and [`CustomRuleSpecializer`] interfaces through which composite calls and carriers
//! request rule programs, their implementations for registered ([`CustomRuleReference`]), unavailable
//! ([`UnavailableCustomRules`]), and lifted ([`LiftedCustomRules`]) rule sets, and the derivations that trace, batch,
//! and transpose the retained rules into cached specializations.

use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Weak};

use crate::arrays::ShardingDimension;
use crate::axes::Axis;
use crate::batching::{BatchAxis, BatchingLevel, BatchingLevelExtent};
use crate::contexts::StagingContext;
use crate::differentiation::{
    CotangentDestinationKind, DifferentiableType, DifferentiationError, ResidualZeroProvider,
    TransposedProgramCotangent,
};
use crate::macros::{check_count, check_types};
use crate::operations::composites::rules::definition::{
    CustomRuleAxisDependence, CustomRuleDefinition, CustomRuleJvp, CustomRuleJvpRule, CustomVjpOperation,
};
use crate::operations::composites::rules::registration::{CustomRuleReference, CustomRuleRegistration};
use crate::operations::composites::rules::{
    CustomRuleBackwardSpecializationKey, CustomRuleBatchedOutputs, CustomRuleBatcher, CustomRuleBatchingLevel,
    CustomRuleBatchingRuleKey, CustomRuleDerivationKey, CustomRuleDerivationKind, CustomRuleDifferentiator,
    CustomRuleProgram, CustomRuleSpecialization, CustomRuleSpecializationKey, CustomRuleSpecializationRequest,
    CustomRuleTracer,
};
use crate::operations::composites::transpose::CompositeTransposeOperation;
use crate::operations::composites::{describe_composite, describe_composite_transpose};
use crate::parameters::{Parameter, Placeholder};
use crate::programs::{
    MaybeZero, Operation, Program, ProgramBuilder, ProgramError, ReferenceMemberType, Type, TypeError,
    TypeIdentityRenaming, Typed, Value, ValueProjection,
};
use crate::specialization::SpecializationCacheError;
use crate::tracing::TracingContext;

#[cfg(doc)]
use crate::batching::DiagonalBatchingPolicy;

/// Derivation of a definition from a source definition, which is stored in the derived definition in place of a
/// batching rule (refer to [`CustomRuleSpecializer::derived`]).
pub(super) struct CustomRuleDerivation<V: Typed + Parameter, O> {
    /// Rules of the source definition.
    source: CustomRuleReference<V, O>,

    /// Key of this derivation.
    key: CustomRuleDerivationKey,
}

/// Seals [`CustomRuleSource`] and [`CustomRuleSpecializer`] (refer to the documentation of [`CustomRuleSource`]).
mod sealed {
    /// Supertrait of [`CustomRuleSource`](super::CustomRuleSource) that only this module can implement.
    pub trait Sealed {}
}

impl<V: Typed + Parameter, O> sealed::Sealed for CustomRuleReference<V, O> {}

impl<Vm: Typed + Parameter, Om> sealed::Sealed for LiftedCustomRules<Vm, Om> {}

/// Source of the retained rules that a [`CompositeOperation`](crate::CompositeOperation) and its
/// [`CompositeTransposeOperation`] apply, which provides their specializations as programs of the operations' own
/// family `(V, O)`. [`CustomRuleReference`] is the source of a definition registered in that family, and
/// [`LiftedCustomRules`] is the source of a call that was converted into that family from a member family (e.g., from
/// array programs into array IR programs), whose specializations the member definition produces and caches before they
/// are converted. Equality and hashing use the identity of the source's definition.
///
/// This trait describes the rules only, which is all that every transform except differentiation needs, and
/// [`CustomRuleSpecializer`] adds the production of their specializations. Keeping the two apart keeps the heavier
/// bounds of converted sources (e.g., on the member family's operations) out of the [`Operation`] implementation of the
/// operations, where they would form a cycle with the payload bounds of the families that contain them.
///
/// This trait and [`CustomRuleSpecializer`] are sealed: their only implementations are [`CustomRuleReference`] and
/// [`LiftedCustomRules`]. Their specialization requests and results are internal to the
/// [`rules`](crate::operations::composites::rules) module (their keys have no public constructors or accessors), so a
/// source outside of it could only delegate to one of these two. New kinds of rules are registered through
/// [`CustomRuleDefinition`] instead.
pub trait CustomRuleSource<V: Typed<Type: Eq + Hash> + Parameter, O>:
    sealed::Sealed + Clone + Debug + Eq + Hash
{
    /// Returns the process-unique identity of the source's definition.
    fn id(&self) -> u64;

    /// Returns the human-readable label of the source's definition.
    fn name(&self) -> &str;

    /// Returns whether the source has a retained forward-mode (i.e., JVP) rule.
    fn has_jvp(&self) -> bool;

    /// Returns whether the source derives its forward-mode rule from the primal region of each call (refer to
    /// [`CustomRuleDefinition::with_jvp_from_primal`]).
    fn derives_jvp_from_primal(&self) -> bool;

    /// Returns whether the source has reverse-mode (i.e., VJP) forward and backward rules.
    fn has_vjp(&self) -> bool;

    /// Returns whether the source has reverse-mode rules whose forward rule reverse mode defers to the backward pass
    /// (refer to [`CustomRuleDefinition::with_deferred_forward_rule`]).
    fn defers_forward(&self) -> bool;

    /// Returns the declared batch axes of the outputs of batched calls (refer to
    /// [`CustomRuleDefinition::with_batched_output_axes`]), or [`None`] when they are not declared.
    fn batched_output_axes(&self) -> Option<&[BatchAxis]>;

    /// Returns whether the source has a custom batching rule (refer to [`CustomRuleDefinition::with_batching_rule`]),
    /// which takes precedence over structurally batching the primal region of its calls.
    fn has_batching_rule(&self) -> bool;

    /// Returns whether the source batches its calls sequentially (refer to
    /// [`CustomRuleDefinition::with_sequential_batching`]) when it has no custom batching rule.
    fn has_sequential_batching(&self) -> bool;

    /// Returns the definition whose rules the source applies directly, or [`None`] when the source obtains its
    /// specializations elsewhere (e.g., from a member definition).
    fn native_definition(&self) -> Option<&CustomRuleDefinition<V, O>>;

    /// Returns whether backward specializations of the source can write cotangents to caller-provided buffers. A
    /// source that cannot returns those cotangents instead, and the carrier's accumulators add them to the buffers.
    fn supports_reference_destinations(&self) -> bool;

    /// Returns the batch axes of the outputs of batched calls that have `output_count` outputs, which map every output
    /// at axis 0 unless the source declares them.
    fn batched_call_output_axes(&self, output_count: usize) -> Result<Vec<BatchAxis>, TypeError> {
        let Some(output_axes) = self.batched_output_axes() else {
            return Ok(vec![BatchAxis::new(0); output_count]);
        };
        check_count!("batched output axis", output_axes, output_count, TypeError);
        Ok(output_axes.to_vec())
    }
}

/// [`CustomRuleSource`] that produces the specializations of its rules, as programs of the operations' own family
/// `(V, O)`. The functions that produce specializations carry their bounds themselves rather than on the trait, like
/// the rule traits, which keeps the operations free of bounds on `O` that would form a cycle with the payload bounds of
/// the families that contain them. It is sealed (refer to the documentation of [`CustomRuleSource`]).
pub trait CustomRuleSpecializer<V: Typed<Type: Eq + Hash> + Parameter, O>: CustomRuleSource<V, O> {
    /// Returns the forward-mode rule specialization for `key`, whose program takes the call's inputs followed by the
    /// active tangents of its differentiated inputs (refer to the tangent activity of the key) and returns the call's
    /// outputs followed by their tangents.
    fn jvp_specialization(
        &self,
        key: CustomRuleSpecializationKey<V::Type>,
    ) -> Result<CustomRuleProgram<V, O>, DifferentiationError>
    where
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type> + ResidualZeroProvider<V::Type, Operation = O>;

    /// Returns the reverse-mode forward rule specialization for `key`, whose program takes the call's inputs and
    /// returns the call's outputs followed by the residuals.
    fn forward_specialization(
        &self,
        key: CustomRuleSpecializationKey<V::Type>,
    ) -> Result<CustomRuleProgram<V, O>, DifferentiationError>
    where
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type>;

    /// Returns the destination-specialized backward rule for `key`, whose program consumes the boundary inputs, then
    /// the live seeds, then the caller buffers, then the remaining known leading inputs, and returns the cotangents of
    /// the differentiated inputs with returned destinations.
    fn backward_specialization(
        &self,
        key: CustomRuleBackwardSpecializationKey<V::Type>,
    ) -> Result<CustomRuleProgram<V, O>, DifferentiationError>
    where
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type> + From<CompositeTransposeOperation<V, O>>;

    /// Returns the batched primal program that the custom batching rule produces for `key`, whose program takes the
    /// level's boundary inputs followed by the call's batched inputs and returns the batched outputs, together with the
    /// batch axes of those outputs.
    fn batching_rule_specialization(
        &self,
        key: CustomRuleBatchingRuleKey<V::Type>,
    ) -> Result<CustomRuleProgram<V, O>, DifferentiationError>
    where
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type>;

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

    /// Returns the rules of the definition derived from this source for `key`, with which forward-mode differentiation
    /// stages the derived call of a call whose forward-mode rule is derived from its primal and whose source has a
    /// custom batching rule or batches its calls sequentially. This keeps that rule, or the sequential batching, on the
    /// path of forward-mode derivatives that are batched after they are taken (i.e., the analogue of JAX's
    /// `custom_vmap_jvp`): the derived call's primal region is the derivative of the call's primal region (i.e., its
    /// Jacobian-Vector Product (JVP), or the tangent half of it), and the derived definition holds no program. It
    /// derives its forward-mode rule from each call's primal region, so that higher orders recurse through each call's
    /// own region, and its batching rule applies the JVP of the source's batching rule, while the derived definition of
    /// a source that batches its calls sequentially batches its own calls sequentially (i.e., it loops over the
    /// derivative one batch item at a time). Calls whose primal regions differ therefore share one derived definition.
    ///
    /// The derived batching rule always computes the derived call's primal outputs with the source rule applied at the
    /// batch axes of the primal inputs, which is exactly how batching the source call computes them, so a rule whose
    /// result depends on its input axes gives a derivative whose primal outputs are those of the batched call. Its
    /// tangents are computed in one of two modes:
    ///
    ///   - **Structural:** when every primal input is replicated, and so is every output of the rule applied to them
    ///     (e.g., the replicated primals and mapped tangents of a forward-mode Jacobian), the JVP of the rule's program
    ///     is batched structurally over the given batch axes.
    ///   - **Rule:** otherwise, each primal and tangent pair with a mapped side is aligned to one axis (i.e., its
    ///     replicated side is broadcast, and a tangent mapped along another axis than its primal is moved), and the JVP
    ///     of the source rule applied at the aligned batch axes computes the tangents. Inputs without a tangent keep
    ///     their batch axes. Alignment changes the rule's input axes only for a replicated primal whose tangent is
    ///     mapped, because the rule cannot apply per-item perturbations of a shared input otherwise; only then is the
    ///     rule applied twice (i.e., once at the primal axes for the primal outputs and once at the aligned axes for
    ///     the tangents). When the source's rule depends on its input batch axes (refer to
    ///     [`CustomRuleDefinition::with_axis_dependent_batching_rule`]), the rule is instead applied once, at the
    ///     primal axes, and the tangents of such primals are recovered exactly from an outer product (refer to
    ///     [`DiagonalBatchingPolicy`]).
    ///
    /// Derived definitions are registered by the source's registration, which retains a bounded number of derived
    /// registrations like its specializations. Every request for the same key returns the same derived definition while
    /// that definition is alive (i.e., while its registration or a call holds it), whether or not the source's
    /// registration is alive, and a fresh one otherwise.
    fn derived(&self, key: CustomRuleDerivationKey) -> Result<Self, DifferentiationError>
    where
        Self: Sized;
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> CustomRuleSource<V, O> for CustomRuleReference<V, O> {
    #[inline]
    fn id(&self) -> u64 {
        self.definition.id
    }

    #[inline]
    fn name(&self) -> &str {
        self.definition.name()
    }

    #[inline]
    fn has_jvp(&self) -> bool {
        matches!(self.definition.jvp, Some(CustomRuleJvp::Rule(_) | CustomRuleJvp::SymbolicZeroRule(_)))
    }

    #[inline]
    fn derives_jvp_from_primal(&self) -> bool {
        matches!(self.definition.jvp, Some(CustomRuleJvp::Primal))
    }

    #[inline]
    fn has_vjp(&self) -> bool {
        self.definition.backward.is_some()
    }

    #[inline]
    fn defers_forward(&self) -> bool {
        self.definition.deferred_forward && self.has_vjp()
    }

    #[inline]
    fn batched_output_axes(&self) -> Option<&[BatchAxis]> {
        self.definition.batched_output_axes.as_deref()
    }

    #[inline]
    fn has_batching_rule(&self) -> bool {
        self.definition.batching_rule.is_some() || self.definition.derivation.is_some()
    }

    #[inline]
    fn has_sequential_batching(&self) -> bool {
        self.definition.sequential_scan.is_some()
    }

    #[inline]
    fn native_definition(&self) -> Option<&CustomRuleDefinition<V, O>> {
        Some(&self.definition)
    }

    #[inline]
    fn supports_reference_destinations(&self) -> bool {
        true
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> CustomRuleSpecializer<V, O> for CustomRuleReference<V, O> {
    fn jvp_specialization(
        &self,
        key: CustomRuleSpecializationKey<V::Type>,
    ) -> Result<CustomRuleProgram<V, O>, DifferentiationError>
    where
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type> + ResidualZeroProvider<V::Type, Operation = O>,
    {
        // Returns the JVP rule specialization for `key`, whose program takes the call's inputs followed by the active
        // tangents of its differentiated inputs and returns the call's outputs followed by their tangents. An unbatched
        // key traces the rule on its first request, while a batched key batches the specialization of its inner key at
        // its outermost level, so that every batching level shares one trace of the rule.
        //
        // The traced rule is validated against the unbatched output types of `key`.
        let rule = match &self.definition.jvp {
            Some(CustomRuleJvp::Rule(rule)) => CustomRuleJvpRule::Materialized(rule),
            Some(CustomRuleJvp::SymbolicZeroRule(rule)) => CustomRuleJvpRule::SymbolicZeros(rule),
            _ => {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "{} has no forward-mode (i.e., JVP) rule",
                        describe_composite(self.definition.name())
                    ),
                }
                .into());
            }
        };
        // The inner specialization is requested only when the outer one is not cached.
        self.specialize(CustomRuleSpecializationRequest::Jvp(key.clone()), || {
            let inner = match key.split_outermost_level() {
                Some((inner_key, level)) => Some((self.jvp_specialization(inner_key)?, level.clone())),
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
                        describe_composite(self.definition.name()),
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
                // The rule's two result partitions are validated separately before they are flattened, because
                // the flattened types alone cannot tell a missing output from a misplaced tangent.
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
                                "{} JVP rule returned {} outputs and {} output \
                                     tangents but the primal has {output_count} outputs",
                                describe_composite(self.definition.name()),
                                outputs.len(),
                                output_tangents.len(),
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
                    self.definition.named_axes.clone(),
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
                    conform_rule_output_types(self.definition.name(), "JVP", program, &expected_output_types, None)?;
                let program = self.definition.discharge_if(key.discharged, program)?;
                return Ok(Arc::new(CustomRuleSpecialization::new(program, Vec::new())));
            };

            // Each active tangent follows the batch axis of its primal, and each output tangent follows its output.
            let differentiated_input_start =
                key.non_differentiated_count + boundary_input_count(&key.levels[..key.levels.len() - 1]);
            let differentiated_input_axes = level.input_axes.get(differentiated_input_start..).ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "{} batching level has too few input axes",
                    describe_composite(self.definition.name()),
                ))
            })?;
            let mut input_axes = level.input_axes.clone();
            input_axes.extend(
                differentiated_input_axes
                    .iter()
                    .zip(&key.tangent_activity)
                    .filter_map(|(axis, active)| active.then_some(*axis)),
            );
            let output_axes = level.output_axes.iter().chain(&level.output_axes).copied().map(Some).collect::<Vec<_>>();
            self.definition.batched_specialization(
                &level,
                &inner.program,
                &input_axes,
                &output_axes,
                CustomRuleBatchedOutputs::Values,
                Vec::new(),
            )
        })
    }

    fn forward_specialization(
        &self,
        key: CustomRuleSpecializationKey<V::Type>,
    ) -> Result<CustomRuleProgram<V, O>, DifferentiationError>
    where
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type>,
    {
        // Returns the reverse-mode forward rule specialization for `key`, whose program takes the call's inputs and
        // returns the call's outputs followed by the residuals. Specializations are traced and batched as for
        // [`Self::jvp_specialization`], except that residuals keep the batch axes that batching naturally produces,
        // which the specialization records.
        //
        // The traced rule is validated against the unbatched output types of `key`. A call that keeps only some of
        // the rule's outputs still receives every output of the rule, because the backward rule receives a seed for
        // each of them, but only the kept outputs are validated against the call's output types.
        let Some(forward) = &self.definition.forward else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("{} has no reverse-mode forward rule", describe_composite(self.definition.name())),
            }
            .into());
        };
        // The inner specialization is requested only when the outer one is not cached.
        let output_count = key.rule_output_count()?;
        self.specialize(CustomRuleSpecializationRequest::Forward(key.clone()), || {
            let inner = match key.split_outermost_level() {
                Some((inner_key, level)) => Some((self.forward_specialization(inner_key)?, level.clone())),
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
                                "{} forward rule returned {} outputs but the \
                                     primal has {output_count}",
                                describe_composite(self.definition.name()),
                                outputs.len(),
                            ))
                            .into());
                        }
                        outputs.extend(residuals);
                        Ok(outputs)
                    },
                    key.input_types.clone(),
                    self.definition.named_axes.clone(),
                )?;
                let kept_output_indices = key.kept_output_indices(output_count);
                let program = conform_rule_output_types(
                    self.definition.name(),
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
                            "{} forward rule returns residual {index} of \
                             reference type `{type}` that is not a leading non-differentiated input forwarded by \
                             identity",
                            describe_composite(self.definition.name()),
                        ))
                        .into());
                    }
                    if !forwarded_inputs.insert(*residual) {
                        return Err(TypeError::invalid(format!(
                            "{} forward rule returns residual {index} of \
                             reference type `{type}` from an input already forwarded by an earlier residual",
                            describe_composite(self.definition.name()),
                        ))
                        .into());
                    }
                }
                let program = self.definition.discharge_if(key.discharged, program)?;
                return Ok(Arc::new(CustomRuleSpecialization::new(program, Vec::new())));
            };
            // The outputs that the call does not keep, like the residuals, keep their natural batch axes.
            let residual_count = inner.program.output_types().len() - output_count;
            let mut kept_output_axes = level.output_axes.iter().copied();
            let output_axes = (0..output_count)
                .map(|index| key.is_output_kept(index).then(|| kept_output_axes.next()).flatten())
                .chain(std::iter::repeat_n(None, residual_count))
                .collect::<Vec<_>>();
            self.definition.batched_specialization(
                &level,
                &inner.program,
                &level.input_axes,
                &output_axes,
                CustomRuleBatchedOutputs::Values,
                Vec::new(),
            )
        })
    }

    fn backward_specialization(
        &self,
        key: CustomRuleBackwardSpecializationKey<V::Type>,
    ) -> Result<CustomRuleProgram<V, O>, DifferentiationError>
    where
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type> + From<CompositeTransposeOperation<V, O>>,
    {
        // Returns the backward rule specialization for `key`. An unbatched key transposes a source program that applies
        // a directly invoking [`CompositeTransposeOperation`] and returns only its live-seed outputs, so transposition
        // hands the rule a structural zero for every dropped output. A batched key batches the specialization of its
        // inner key at its outermost level.
        let Some(transposer) = &self.definition.transposer else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("{} has no backward rule", describe_composite_transpose(self.definition.name())),
            }
            .into());
        };
        // The inner specialization is requested only when the outer one is not cached.
        self.specialize(CustomRuleSpecializationRequest::Backward(key.clone()), || {
            let inner = match key.split_outermost_level() {
                Some((inner_key, level)) => Some((self.backward_specialization(inner_key)?, level.clone())),
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
                let program = self.definition.discharge_if(key.discharged, program)?;
                return Ok(Arc::new(CustomRuleSpecialization { program, output_axes: Vec::new(), cotangents }));
            };

            // The inner program consumes its inner boundary inputs, then the live seeds, then the caller buffers, then
            // the remaining leading inputs. Each seed follows its output, each caller buffer follows its input, and
            // each leading input keeps its recorded axis. Each returned cotangent is aligned to its input, while
            // structural-zero cotangents have no output and stay structural zeros at every level.
            let inner_boundary_input_count = boundary_input_count(&key.levels[..key.levels.len() - 1]);
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
            self.definition.batched_specialization(
                &level,
                &inner.program,
                &input_axes,
                &output_axes,
                CustomRuleBatchedOutputs::Cotangents,
                inner.cotangents.clone(),
            )
        })
    }

    fn batching_rule_specialization(
        &self,
        key: CustomRuleBatchingRuleKey<V::Type>,
    ) -> Result<CustomRuleProgram<V, O>, DifferentiationError>
    where
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type>,
    {
        // Traces the custom batching rule over the level's boundary inputs followed by the batched inputs, recording
        // the batch axes that it declares for its outputs. A derived definition applies the JVP of its source's rule
        // instead. The program becomes the primal region of the batched call, so it is discharged with the call.
        let Some(rule) = &self.definition.batching_rule else {
            let Some(derivation) = &self.definition.derivation else {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!("{} has no custom batching rule", describe_composite(self.definition.name())),
                }
                .into());
            };
            return self.specialize(CustomRuleSpecializationRequest::BatchingRule(key.clone()), || {
                let CustomRuleSpecialization { program, output_axes, .. } =
                    derivation.batching_rule_specialization(self.definition.name(), &key)?;
                let program = self.definition.discharge_if(key.discharged, program)?;
                Ok(Arc::new(CustomRuleSpecialization::new(program, output_axes)))
            });
        };
        self.specialize(CustomRuleSpecializationRequest::BatchingRule(key.clone()), || {
            let boundary_input_count = key.boundary_input_types.len();
            let mut output_axes = Vec::new();
            let (_, program) = TracingContext::<V, O>::trace_with_named_axes(
                |values: Vec<CustomRuleTracer<V, O>>| {
                    let (boundary_inputs, inputs) = values.split_at(boundary_input_count);
                    let (outputs, axes) = rule.apply(&key.level, boundary_inputs, inputs, &key.input_axes)?;
                    if axes.len() != outputs.len() {
                        return Err(TypeError::invalid(format!(
                            "{} batching rule returned {} outputs but {} output \
                             batch axes",
                            describe_composite(self.definition.name()),
                            outputs.len(),
                            axes.len(),
                        ))
                        .into());
                    }
                    output_axes = axes;
                    Ok(outputs)
                },
                key.boundary_input_types.iter().chain(&key.input_types).cloned().collect::<Vec<_>>(),
                self.definition.named_axes.clone(),
            )?;
            let program = self.definition.discharge_if(key.discharged, program)?;
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
        let Some(sequential_scan) = &self.definition.sequential_scan else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "{} does not batch its calls sequentially",
                    describe_composite(self.definition.name())
                ),
            }
            .into());
        };
        Ok(sequential_scan(length, carry_count, axis_sharding, stacked_output_count)?)
    }

    fn derived(&self, key: CustomRuleDerivationKey) -> Result<Self, DifferentiationError> {
        // A derived definition copies the capabilities that its calls need from the source definition and holds no
        // program, so calls whose primal regions differ share it (refer to `CustomRuleSpecializer::derived`). A source
        // that batches its calls sequentially, without a custom batching rule, derives a definition that batches its
        // calls sequentially as well, which needs no derivation.
        if !self.has_batching_rule() && !self.has_sequential_batching() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "{} has neither a custom batching rule nor sequential batching from which to derive rules",
                    describe_composite(self.definition.name()),
                ),
            }
            .into());
        }
        let derive = |key: CustomRuleDerivationKey| {
            let prefix = match key.kind {
                CustomRuleDerivationKind::Jvp => "jvp",
                CustomRuleDerivationKind::Pushforward => "pushforward",
            };
            let derivation = self.has_batching_rule().then(|| CustomRuleDerivation { source: self.clone(), key });
            CustomRuleDefinition {
                name: Cow::Owned(format!("{prefix}({})", self.definition.name())),
                named_axes: self.definition.named_axes.clone(),
                jvp: Some(CustomRuleJvp::Primal),
                batcher: self.definition.batcher.clone(),
                differentiator: self.definition.differentiator.clone(),
                axis_dependence: self.definition.axis_dependence.clone(),
                derivation,
                sequential_scan: self.definition.sequential_scan.clone(),
                discharger: self.definition.discharger.clone(),
                ..CustomRuleDefinition::new("")
            }
        };
        // A derived definition that is still alive is reused, so that its calls keep one identity (and with it the
        // entries of enclosing caches) even after its registration was evicted or the source registration dropped.
        let definition = {
            let mut definitions =
                self.definition.derived_definitions.lock().expect("custom rule derivation mutex is poisoned");
            definitions.retain(|_, definition| definition.strong_count() > 0);
            match definitions.get(&key).and_then(Weak::upgrade) {
                Some(definition) => definition,
                None => {
                    let definition = Arc::new(derive(key.clone()));
                    definitions.insert(key.clone(), Arc::downgrade(&definition));
                    definition
                }
            }
        };
        let Some(caches) = self.caches.upgrade() else {
            return Ok(CustomRuleReference { definition, caches: Weak::new() });
        };
        let mut registrations = caches.derived_registrations.lock().expect("custom rule derivation mutex is poisoned");
        if let Some(registration) = registrations.get(&key) {
            return Ok(registration.reference());
        }
        let registration = CustomRuleRegistration::from_definition(definition);
        let reference = registration.reference();

        // An evicted registration is dropped after the lock is released, because dropping its caches may drop
        // programs whose destructors reach other definitions.
        let evicted = registrations.push(key, registration);
        drop(registrations);
        drop(evicted);
        Ok(reference)
    }
}

/// Rules of a definition that cannot be converted into a family (i.e., a definition registered in a family whose
/// values that family cannot represent, such as array IR programs converted into a backend whose constants are only
/// captures). Its calls still execute their primal, but every derivative request fails with an exact error.
#[derive(Clone, Debug)]
pub struct UnavailableCustomRules {
    /// Process-unique identity of the original definition.
    id: u64,

    /// Human-readable label of the original definition.
    name: Cow<'static, str>,

    /// Whether the original definition has a retained forward-mode (i.e., JVP) rule.
    has_jvp: bool,

    /// Whether the original definition derives its forward-mode rule from the primal.
    derives_jvp_from_primal: bool,

    /// Whether the original definition has reverse-mode (i.e., VJP) rules.
    has_vjp: bool,

    /// Whether the original definition defers its reverse-mode forward rule to the backward pass.
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

    /// Returns the error that every derivative request of these rules reports.
    fn error(&self) -> DifferentiationError {
        ProgramError::UnsupportedOperation {
            message: format!(
                "{} was converted from a family whose values its current family \
                 cannot represent, so its derivative rules are unavailable",
                describe_composite(&self.name),
            ),
        }
        .into()
    }
}

/// Source of the rules of a call that was converted from a member family `(Vm, Om)` into a family that contains it
/// (e.g., from array programs into array IR programs, or into a backend family). Each specialization request projects
/// its key into the member family, obtains the specialization from the member definition (so that the member
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
pub enum LiftedCustomRules<Vm: Typed + Parameter, Om> {
    /// Rules of a definition registered in the member family.
    Member(CustomRuleReference<Vm, Om>),

    /// Rules of a definition that cannot be converted into the requesting family.
    Unavailable(UnavailableCustomRules),
}

impl<Vm: Typed<Type: Eq + Hash> + Parameter, Om> LiftedCustomRules<Vm, Om> {
    /// Returns the rules of the member definition, or the error that a derivative request of unavailable rules reports.
    fn member(&self) -> Result<&CustomRuleReference<Vm, Om>, DifferentiationError> {
        match self {
            Self::Member(rules) => Ok(rules),
            Self::Unavailable(rules) => Err(rules.error()),
        }
    }

    /// Projects the provided types of a converted call into the member family, rejecting types that have no member
    /// representation.
    fn project_types<Tc>(&self, types: &[Tc]) -> Result<Vec<Vm::Type>, DifferentiationError>
    where
        Tc: Type,
        for<'t> &'t Vm::Type: TryFrom<&'t Tc, Error = TypeError>,
    {
        types
            .iter()
            .map(|r#type| {
                <&Vm::Type>::try_from(r#type).cloned().map_err(|_| {
                    ProgramError::UnsupportedOperation {
                        message: format!(
                            "{} cannot be differentiated with respect to type \
                             `{type}`, which the family in which its rules were registered cannot represent",
                            describe_composite(self.name_or_unavailable()),
                        ),
                    }
                    .into()
                })
            })
            .collect()
    }

    /// Projects the provided batching levels of a converted call into the member family, rejecting dynamic extents
    /// whose types have no member representation.
    fn project_levels<Tc>(
        &self,
        levels: &[CustomRuleBatchingLevel<Tc>],
    ) -> Result<Vec<CustomRuleBatchingLevel<Vm::Type>>, DifferentiationError>
    where
        Tc: Type,
        for<'t> &'t Vm::Type: TryFrom<&'t Tc, Error = TypeError>,
    {
        levels
            .iter()
            .map(|level| {
                let extent = match level.level.extent() {
                    BatchingLevelExtent::Static(extent) => BatchingLevelExtent::Static(*extent),
                    BatchingLevelExtent::Dynamic(r#type) => {
                        BatchingLevelExtent::Dynamic(<&Vm::Type>::try_from(r#type).cloned().map_err(|_| {
                            ProgramError::UnsupportedOperation {
                                message: format!(
                                    "{} cannot differentiate a batched call whose \
                                     batch extent has type `{type}`, which the family in which its rules were \
                                     registered cannot represent; differentiate the call before batching it instead",
                                    describe_composite(self.name_or_unavailable()),
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
            Self::Member(rules) => rules.definition.name(),
            Self::Unavailable(rules) => &rules.name,
        }
    }
}

impl<Vm: Typed<Type: Eq + Hash> + Parameter, Om> Clone for LiftedCustomRules<Vm, Om> {
    fn clone(&self) -> Self {
        match self {
            Self::Member(rules) => Self::Member(rules.clone()),
            Self::Unavailable(rules) => Self::Unavailable(rules.clone()),
        }
    }
}

impl<Vm: Typed<Type: Eq + Hash> + Parameter, Om> Debug for LiftedCustomRules<Vm, Om> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Member(rules) => formatter.debug_tuple("Member").field(rules).finish(),
            Self::Unavailable(rules) => formatter.debug_tuple("Unavailable").field(rules).finish(),
        }
    }
}

impl<Vm: Typed<Type: Eq + Hash> + Parameter, Om> PartialEq for LiftedCustomRules<Vm, Om> {
    fn eq(&self, other: &Self) -> bool {
        let id = |rules: &Self| match rules {
            Self::Member(rules) => rules.definition.id,
            Self::Unavailable(rules) => rules.id,
        };
        id(self) == id(other)
    }
}

impl<Vm: Typed<Type: Eq + Hash> + Parameter, Om> Eq for LiftedCustomRules<Vm, Om> {}

impl<Vm: Typed<Type: Eq + Hash> + Parameter, Om> Hash for LiftedCustomRules<Vm, Om> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::Member(rules) => rules.definition.id.hash(state),
            Self::Unavailable(rules) => rules.id.hash(state),
        }
    }
}

impl<Vm, Om, Vc, Oc> CustomRuleSource<Vc, Oc> for LiftedCustomRules<Vm, Om>
where
    Vm: Typed<Type: Eq + Hash> + Parameter,
    Vc: Typed<Type: Eq + Hash> + Parameter,
{
    fn id(&self) -> u64 {
        match self {
            Self::Member(rules) => rules.definition.id,
            Self::Unavailable(rules) => rules.id,
        }
    }

    #[inline]
    fn name(&self) -> &str {
        self.name_or_unavailable()
    }

    fn has_jvp(&self) -> bool {
        match self {
            Self::Member(rules) => rules.has_jvp(),
            Self::Unavailable(rules) => rules.has_jvp,
        }
    }

    fn derives_jvp_from_primal(&self) -> bool {
        match self {
            Self::Member(rules) => rules.derives_jvp_from_primal(),
            Self::Unavailable(rules) => rules.derives_jvp_from_primal,
        }
    }

    fn has_vjp(&self) -> bool {
        match self {
            Self::Member(rules) => rules.has_vjp(),
            Self::Unavailable(rules) => rules.has_vjp,
        }
    }

    fn defers_forward(&self) -> bool {
        match self {
            Self::Member(rules) => rules.defers_forward(),
            Self::Unavailable(rules) => rules.defers_forward,
        }
    }

    fn batched_output_axes(&self) -> Option<&[BatchAxis]> {
        match self {
            Self::Member(rules) => rules.batched_output_axes(),
            Self::Unavailable(rules) => rules.batched_output_axes.as_deref(),
        }
    }

    fn has_batching_rule(&self) -> bool {
        match self {
            Self::Member(rules) => rules.has_batching_rule(),
            Self::Unavailable(rules) => rules.has_batching_rule,
        }
    }

    fn has_sequential_batching(&self) -> bool {
        match self {
            Self::Member(rules) => rules.has_sequential_batching(),
            Self::Unavailable(rules) => rules.has_sequential_batching,
        }
    }

    #[inline]
    fn native_definition(&self) -> Option<&CustomRuleDefinition<Vc, Oc>> {
        None
    }

    #[inline]
    fn supports_reference_destinations(&self) -> bool {
        false
    }
}

impl<Vm, Om, Vc, Oc> CustomRuleSpecializer<Vc, Oc> for LiftedCustomRules<Vm, Om>
where
    Vm: Value<Type: DifferentiableType + Eq + Hash>,
    TracingContext<Vm, Om>: StagingContext<Type = Vm::Type, Constant = Vm, Operation = Om>,
    Om: Operation<Type = Vm::Type>
        + From<CompositeTransposeOperation<Vm, Om>>
        + ResidualZeroProvider<Vm::Type, Operation = Om>,
    Vc: 'static + ValueProjection<Vm::Type, Projected = Vm, Type: Eq + Hash + From<Vm::Type>>,
    Oc: 'static + From<Om>,
    CustomRuleSpecialization<Vc, Oc>: Send + Sync,
    for<'t> &'t Vm::Type: TryFrom<&'t Vc::Type, Error = TypeError>,
{
    fn jvp_specialization(
        &self,
        key: CustomRuleSpecializationKey<Vc::Type>,
    ) -> Result<CustomRuleProgram<Vc, Oc>, DifferentiationError>
    where
        Vc: Value<Type: DifferentiableType>,
        Oc: Operation<Type = Vc::Type> + ResidualZeroProvider<Vc::Type, Operation = Oc>,
    {
        let member = self.member()?;
        let member_key = CustomRuleSpecializationKey {
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
                program: specialization.program.clone().into_unprojected::<Vc, Oc>()?,
                output_axes: specialization.output_axes.clone(),
                cotangents: specialization.cotangents.clone(),
            }))
        })
    }

    fn forward_specialization(
        &self,
        key: CustomRuleSpecializationKey<Vc::Type>,
    ) -> Result<CustomRuleProgram<Vc, Oc>, DifferentiationError>
    where
        Vc: Value<Type: DifferentiableType>,
        Oc: Operation<Type = Vc::Type>,
    {
        let member = self.member()?;
        let member_key = CustomRuleSpecializationKey {
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
                program: specialization.program.clone().into_unprojected::<Vc, Oc>()?,
                output_axes: specialization.output_axes.clone(),
                cotangents: specialization.cotangents.clone(),
            }))
        })
    }

    fn backward_specialization(
        &self,
        key: CustomRuleBackwardSpecializationKey<Vc::Type>,
    ) -> Result<CustomRuleProgram<Vc, Oc>, DifferentiationError>
    where
        Vc: Value<Type: DifferentiableType>,
        Oc: Operation<Type = Vc::Type> + From<CompositeTransposeOperation<Vc, Oc>>,
    {
        // Carriers of converted calls request returned cotangents for caller buffers (refer to
        // `supports_reference_destinations`), so a buffer destination here means that a carrier skipped that step.
        let member = self.member()?;
        if key.destination_kinds.contains(&CotangentDestinationKind::Reference) {
            return Err(ProgramError::MalformedProgram(format!(
                "{} requested a caller-buffer destination from rules \
                 that cannot write caller buffers",
                describe_composite_transpose(self.name_or_unavailable()),
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
                program: specialization.program.clone().into_unprojected::<Vc, Oc>()?,
                output_axes: specialization.output_axes.clone(),
                cotangents: specialization.cotangents.clone(),
            }))
        })
    }

    fn batching_rule_specialization(
        &self,
        key: CustomRuleBatchingRuleKey<Vc::Type>,
    ) -> Result<CustomRuleProgram<Vc, Oc>, DifferentiationError>
    where
        Vc: Value<Type: DifferentiableType>,
        Oc: Operation<Type = Vc::Type>,
    {
        // The member rule batches the call at the projected input types, and its program is converted back.
        let member = self.member()?;
        let project = |r#type: &Vc::Type| {
            <&Vm::Type>::try_from(r#type).cloned().map_err(|_| {
                DifferentiationError::from(ProgramError::UnsupportedOperation {
                    message: format!(
                        "{} cannot batch a call over type `{type}`, which the \
                         family in which its rules were registered cannot represent",
                        describe_composite(self.name_or_unavailable()),
                    ),
                })
            })
        };
        let extent = match key.level.extent() {
            BatchingLevelExtent::Static(extent) => BatchingLevelExtent::Static(*extent),
            BatchingLevelExtent::Dynamic(r#type) => BatchingLevelExtent::Dynamic(project(r#type)?),
        };
        let member_key = CustomRuleBatchingRuleKey {
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
                program: specialization.program.clone().into_unprojected::<Vc, Oc>()?,
                output_axes: specialization.output_axes.clone(),
                cotangents: specialization.cotangents.clone(),
            }))
        })
    }

    fn sequential_batching_scan(
        &self,
        length: usize,
        carry_count: usize,
        axis_sharding: &ShardingDimension,
        stacked_output_count: usize,
    ) -> Result<(Oc, Vc::Type), DifferentiationError> {
        // The member definition constructs the scan in its own family, which contains the scan operation and the type
        // of its slice index of this family.
        let (scan, index_type) =
            self.member()?.sequential_batching_scan(length, carry_count, axis_sharding, stacked_output_count)?;
        Ok((Oc::from(scan), Vc::Type::from(index_type)))
    }

    fn derived(&self, key: CustomRuleDerivationKey) -> Result<Self, DifferentiationError> {
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
    /// batched outputs, together with their batch axes. The source rule programs that it applies are discharged when
    /// `key` is, while the returned program is left for the caller to discharge.
    ///
    /// # Parameters
    ///
    ///   - `name`: Name of the derived definition, used in diagnostics.
    ///   - `key`: Specialization key of the batched derived call.
    fn batching_rule_specialization(
        &self,
        name: &str,
        key: &CustomRuleBatchingRuleKey<V::Type>,
    ) -> Result<CustomRuleSpecialization<V, O>, DifferentiationError> {
        let source = &self.source.definition;
        let (Some(differentiator), Some(batcher)) = (&source.differentiator, &source.batcher) else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "{} cannot batch a derivative because its source definition has no batching support (refer \
                     to `CustomRuleDefinition::with_batching_support`)",
                    describe_composite(name),
                ),
            }
            .into());
        };
        let input_count = key.input_types.len();
        let tangent_count = self.key.active_input_indices.len();
        let primal_count = input_count
            .checked_sub(tangent_count)
            .filter(|count| *count >= self.key.input_count)
            .filter(|_| key.unbatched_input_types.len() == input_count && key.input_axes.len() == input_count)
            .ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "batched {} has {input_count} inputs, which do not cover \
                     its {} primal inputs and {tangent_count} tangents",
                    describe_composite(name),
                    self.key.input_count,
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
        let rule_key = |input_types: Vec<V::Type>, input_axes: Vec<BatchAxis>| CustomRuleBatchingRuleKey {
            level: key.level.clone(),
            boundary_input_types: key.boundary_input_types.clone(),
            input_types,
            unbatched_input_types: key.unbatched_input_types[..primal_count].to_vec(),
            input_axes,
            discharged: key.discharged,
        };
        let program_types = key.boundary_input_types.iter().chain(&key.input_types).cloned().collect::<Vec<_>>();
        let tangents_only = self.key.kind == CustomRuleDerivationKind::Pushforward;

        // The source rule applied at the primal inputs' own batch axes is the batched primal of the source call, so it
        // computes the derived call's primal outputs in every mode.
        let primal_rule = self
            .source
            .batching_rule_specialization(rule_key(key.input_types[..primal_count].to_vec(), primal_axes.to_vec()))?;

        // Structural mode: every primal input is replicated, and so is every output of the rule applied to them.
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
                &key.level,
                &jvp,
                &input_axes,
                &vec![None; jvp.output_types().len()],
                CustomRuleBatchedOutputs::Values,
            )
            .map_err(ProgramError::from)?;

            // The batched program consumes the level's boundary inputs before the JVP program's own copy of them.
            let (_, program) = TracingContext::<V, O>::trace_with_named_axes(
                |values: Vec<CustomRuleTracer<V, O>>| {
                    let context = values[0].context().clone();
                    let mut inputs = values[..boundary_input_count].to_vec();
                    inputs.extend(values);
                    Ok(batched.interpret_in_context(&context, inputs)?)
                },
                program_types,
                self.source.definition.named_axes.clone(),
            )?;
            return Ok(CustomRuleSpecialization::new(program, output_axes));
        }

        // Rule mode: align each primal and tangent pair with a mapped side to one axis, preferring the primal's axis.
        // Only a replicated primal whose tangent is mapped is broadcast, in which case the rule applied at the aligned
        // axes computes the tangents, while the primal outputs still come from the rule applied at the primal axes.
        let mut aligned_axes = key.input_axes.clone();
        for (tangent, &primal) in tangent_primal_indices.iter().enumerate() {
            let axis = if primal_axes[primal].is_replicated() { tangent_axes[tangent] } else { primal_axes[primal] };
            aligned_axes[primal] = axis;
            aligned_axes[primal_count + tangent] = axis;
        }
        let broadcasts_primals = aligned_axes[..primal_count] != *primal_axes;
        if broadcasts_primals && let Some(axis_dependence) = &source.axis_dependence {
            return self.axis_dependent_batching_rule_specialization(
                name,
                key,
                &primal_rule,
                &tangent_primal_indices,
                axis_dependence.as_ref(),
                differentiator.as_ref(),
                batcher.as_ref(),
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
                    batcher.as_ref(),
                    &key.level,
                    &aligned_inputs.iter().map(|&index| key.unbatched_input_types[index].clone()).collect::<Vec<_>>(),
                    &aligned_inputs.iter().map(|&index| key.input_axes[index]).collect::<Vec<_>>(),
                    &aligned_inputs.iter().map(|&index| aligned_axes[index]).collect::<Vec<_>>(),
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
            self.source.definition.named_axes.clone(),
        )?;
        Ok(CustomRuleSpecialization::new(program, output_axes))
    }

    /// Returns the batched program of a derived call for `key` when its source batching rule depends on its input batch
    /// axes and some replicated primal input has a mapped tangent (refer to
    /// [`CustomRuleDefinition::with_axis_dependent_batching_rule`]). The rule is applied once, at the primal inputs'
    /// own batch axes, as `P`, which computes the primal outputs. The tangents are the sum of two contributions:
    ///
    ///   - **Following:** the JVP of `P` with respect to the tangents that can follow their primal's batch axis, after
    ///     aligning them to it (i.e., broadcasting or moving them).
    ///   - **Tangent-only:** the JVP of `P` with respect to the mapped tangents of replicated primals. Those tangents
    ///     differ per batch item while their primals are shared, so the JVP is batched structurally over them at a
    ///     second level of the same extent, which yields the outer product of their batch with the batch of `P`. Item
    ///     `i` of the result is item `i` of that outer product along both axes, which is its diagonal.
    ///
    /// # Parameters
    ///
    ///   - `name`: Name of the derived definition, used in diagnostics.
    ///   - `key`: Specialization key of the batched derived call.
    ///   - `primal_rule`: Source rule applied at the primal inputs' batch axes (i.e., `P`).
    ///   - `tangent_primal_indices`: Index of the primal input of each active tangent, among the derived call's primal
    ///     inputs.
    ///   - `axis_dependence`: Operations of the source definition that take diagonals and add contributions.
    ///   - `differentiator`: Differentiator of the source definition.
    ///   - `batcher`: Batcher of the source definition.
    fn axis_dependent_batching_rule_specialization(
        &self,
        name: &str,
        key: &CustomRuleBatchingRuleKey<V::Type>,
        primal_rule: &CustomRuleProgram<V, O>,
        tangent_primal_indices: &[usize],
        axis_dependence: &dyn CustomRuleAxisDependence<V, O>,
        differentiator: &CustomRuleDifferentiator<V, O>,
        batcher: &CustomRuleBatcher<V, O>,
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
                    describe_composite(name),
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
                &key.level,
                &misaligned_following
                    .iter()
                    .map(|&tangent| key.unbatched_input_types[primal_count + tangent].clone())
                    .collect::<Vec<_>>(),
                &misaligned_following.iter().map(|&tangent| tangent_axes[tangent]).collect::<Vec<_>>(),
                &misaligned_following
                    .iter()
                    .map(|&tangent| primal_axes[tangent_primal_indices[tangent]])
                    .collect::<Vec<_>>(),
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
            &outer_level,
            &tangent_only_jvp,
            &outer_input_axes,
            &vec![None; tangent_only_jvp.output_types().len()],
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
                    &key.level,
                    &broadcast_following.iter().map(|(index, _)| following_types[*index].clone()).collect::<Vec<_>>(),
                    &vec![BatchAxis::replicated(); broadcast_following.len()],
                    &broadcast_following.iter().map(|(_, axis)| *axis).collect::<Vec<_>>(),
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

                        // An output that does not depend on the tangent-only inputs (e.g., whose contribution is a
                        // zero) is replicated along the second level, so its contribution already has `P`'s batch axis.
                        (Some(index), Some(_), None) => Some(outer_outputs[*index].clone()),
                        (Some(index), Some(axis), Some(outer_axis)) => {
                            // The second level's axis precedes `P`'s batch axis in the outer product exactly when its
                            // position does not exceed that axis's position in `P`'s output.
                            let axis = position(axis)?;
                            let kept_axis = if outer_axis <= axis { axis + 1 } else { axis };
                            Some(axis_dependence.diagonal(&outer_outputs[*index], kept_axis, outer_axis)?)
                        }
                    };
                    let following = following.map(|index| following_outputs[index].clone());
                    // Every recorded output has at least one contribution.
                    outputs.push(match (following, tangent_only) {
                        (Some(following), Some(tangent_only)) => axis_dependence.add(&following, &tangent_only)?,
                        (Some(tangent), None) => tangent,
                        (None, tangent) => tangent.unwrap(),
                    });
                }
                Ok(outputs)
            },
            key.boundary_input_types.iter().chain(&key.input_types).cloned().collect::<Vec<_>>(),
            self.source.definition.named_axes.clone(),
        )?;
        Ok(CustomRuleSpecialization::new(program, all_output_axes))
    }

    /// Returns the Jacobian-Vector Product (JVP) program of the source rule's batched `program` with respect to
    /// `input_indices`, restricted to its tangent outputs when `tangents_only` is set, after validating that its live
    /// tangent outputs are those of the derived calls. The derivatives with respect to subsets of the active inputs
    /// (refer to [`Self::axis_dependent_batching_rule_specialization`]) use [`Self::partial_jvp_program`] instead.
    ///
    /// # Parameters
    ///
    ///   - `name`: Name of the derived definition, used in diagnostics.
    ///   - `differentiator`: Differentiator of the source definition.
    ///   - `program`: Batched program of the source rule.
    ///   - `input_indices`: Indices of the program inputs whose tangents are active.
    ///   - `tangents_only`: Whether the returned program returns only the tangent outputs (e.g., for pushforward
    ///     derivations, or when another program computes the primal outputs).
    fn jvp_program(
        &self,
        name: &str,
        differentiator: &CustomRuleDifferentiator<V, O>,
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
    fn partial_jvp_program(
        differentiator: &CustomRuleDifferentiator<V, O>,
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
                "{} batching rule program has live tangents for the outputs \
                 {output_tangent_mask:?}, but the derivative of its primal has them for the outputs {:?}",
                describe_composite(name),
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
    ///   - `batcher`: Batcher of the source definition.
    ///   - `level`: Batching level of the values.
    ///   - `types`: Per-item types of the values.
    ///   - `input_axes`: Batch axes of the values.
    ///   - `output_axes`: Batch axes of the aligned values.
    fn alignment_program(
        name: &str,
        batcher: &CustomRuleBatcher<V, O>,
        level: &BatchingLevel<V::Type>,
        types: &[V::Type],
        input_axes: &[BatchAxis],
        output_axes: &[BatchAxis],
    ) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError> {
        let mut builder = ProgramBuilder::<V, O>::new();
        let inputs = types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let identity =
            builder.build::<Vec<V>, Vec<V>>(inputs, vec![Placeholder; types.len()], vec![Placeholder; types.len()])?;
        let required_axes = output_axes.iter().copied().map(Some).collect::<Vec<_>>();
        let (alignment, aligned_axes) =
            batcher(level, &identity, input_axes, &required_axes, CustomRuleBatchedOutputs::Values)
                .map_err(ProgramError::from)?;
        if aligned_axes != output_axes {
            return Err(ProgramError::MalformedProgram(format!(
                "batched {} could not align values from the batch axes \
                 {input_axes:?} to {output_axes:?}",
                describe_composite(name),
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
    /// Batches `program` at the provided level with the retained batcher, rejecting the request when this definition
    /// has no batching support. The batched specialization has the provided `cotangents`, which are those of the
    /// specialization that `program` belongs to (i.e., empty unless it is a backward-rule specialization).
    fn batched_specialization(
        &self,
        level: &CustomRuleBatchingLevel<V::Type>,
        program: &Program<V, O, Vec<V>, Vec<V>>,
        input_axes: &[BatchAxis],
        output_axes: &[Option<BatchAxis>],
        outputs: CustomRuleBatchedOutputs,
        cotangents: Vec<TransposedProgramCotangent>,
    ) -> Result<CustomRuleProgram<V, O>, DifferentiationError> {
        let Some(batcher) = &self.batcher else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "{} cannot differentiate a batched call because its definition has no batching support \
                     (refer to `CustomRuleDefinition::with_batching_support`)",
                    describe_composite(self.name()),
                ),
            }
            .into());
        };
        let (program, output_axes) =
            batcher(&level.level, program, input_axes, output_axes, outputs).map_err(ProgramError::from)?;
        Ok(Arc::new(CustomRuleSpecialization { program, output_axes, cotangents }))
    }
}

/// Transposes the source program of an unbatched backward-rule specialization with respect to the inputs at
/// `input_indices`, whose cotangents have the provided destination kinds, and returns the transposed program together
/// with the representation of each selected input's cotangent. Structural-zero cotangents (e.g., those that a backward
/// rule declares with [`MaybeZero::Zero`]) are not materialized, so they consume no output of the transposed program
/// and reach the carrier as [`TransposedProgramCotangent::Zero`]s, which contribute nothing to their destinations.
pub(super) fn transpose_backward_source_program<V, O>(
    program: &Program<V, O, Vec<V>, Vec<V>>,
    input_indices: &[usize],
    destination_kinds: &[CotangentDestinationKind],
) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Vec<TransposedProgramCotangent>), DifferentiationError>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: Value<Type: DifferentiableType + ReferenceMemberType>,
    O: CustomVjpOperation<V>,
{
    let transposed = program.entry_region_ref().transpose(input_indices, &[], destination_kinds)?;
    let cotangents = transposed.cotangents().to_vec();
    Ok((Arc::unwrap_or_clone(transposed.into_program()), cotangents))
}

/// Returns the number of boundary inputs that the provided batching levels together prepended to an operation's inputs.
pub(in crate::operations::composites) fn boundary_input_count<T>(levels: &[CustomRuleBatchingLevel<T>]) -> usize {
    levels.iter().map(|level| level.boundary_input_count).sum()
}

/// Validates that a retained rule of the custom rule set named `name` returned the `expected` output types.
pub(in crate::operations::composites) fn validate_rule_output_types<T: Type>(
    name: &str,
    rule: &str,
    expected: &[T],
    actual: &[T],
) -> Result<(), TypeError> {
    check_types!(@same, format!("{} {rule} rule output", describe_composite(name)), [
        expected,
        actual,
    ]);
    Ok(())
}

/// Validates that `program`, a retained rule of the custom rule set named `name` traced in its own trace, returns the
/// `expected` types as its leading outputs (or as its outputs at the `selected` positions, when provided), and returns
/// it with its type identities renamed onto those of `expected`.
/// The trace establishes fresh identities for the extents that the rule computes (e.g., a recomputed `n · n` extent of
/// the primal outputs), so its outputs can agree with `expected` only up to a renaming of those identities. The
/// renaming is derived across the complete boundary, so that the identities of the program inputs map to themselves,
/// and it must be bijective, so that it preserves which outputs share an identity. It is applied to the whole program,
/// including any trailing outputs (e.g., residuals) that name the renamed identities.
fn conform_rule_output_types<V: Value, O: Operation<Type = V::Type>>(
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
            "{} {rule} rule output type signature mismatch: expected \
             [{expected}] but got [{actual}]{reason}",
            describe_composite(name),
        ))
        .into());
    };
    if renaming.is_identity() {
        return Ok(program);
    }
    program.rename_type_identities(&renaming)
}

/// Returns the renaming of the type identities of `source` onto those of `target` when the two type signatures are
/// equal up to a bijective renaming of their identities, which holds exactly when each renames into the other.
pub(in crate::operations::composites) fn derive_bijective_identity_renaming<T: Type>(
    source: &[T],
    target: &[T],
) -> Option<TypeIdentityRenaming<T::Identity>> {
    let renames_into = |declared: &[T], actual: &[T]| {
        let renaming = T::derive_identity_renaming(declared, actual).ok()?;
        declared
            .iter()
            .zip(actual)
            .all(|(declared, actual)| declared.rename_identities(&renaming).is_ok_and(|renamed| &renamed == actual))
            .then_some(renaming)
    };
    if source.len() != target.len() {
        return None;
    }
    renames_into(target, source)?;
    renames_into(source, target)
}

/// Converts a failed or reentrant specialization of a retained rule of the custom rule set named `name` into a
/// [`DifferentiationError`], keeping the rule's own error when it produced one.
pub(super) fn specialization_error<E: Debug + Display + Into<DifferentiationError>>(
    name: &str,
    error: SpecializationCacheError<E>,
) -> DifferentiationError {
    match error {
        SpecializationCacheError::Production(error) => error.into(),
        error @ SpecializationCacheError::Reentrant(_) => {
            ProgramError::UnsupportedOperation { message: format!("{} makes a {error}", describe_composite(name)) }
                .into()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType, DimensionBounds,
        DimensionType, DimensionVariable, Shape, ShardingDimension,
    };
    use crate::contexts::Context;
    use crate::operations::arithmetic::MulOperation;
    use crate::operations::composites::rules::CUSTOM_RULE_SPECIALIZATION_CAPACITY;
    use crate::operations::composites::rules::tests::{MemberTracer, batching_rule_key};
    use crate::operations::composites::tests::{
        MemberDefinition, RuleCounters, TestDefinition, call_jvp, cube_definition, custom_rule_program,
        member_cube_definition,
    };
    use crate::programs::ReferenceType;

    use super::*;

    /// Returns a registration of `(x, y) ↦ x · y` whose forward-mode rule is derived from its primal and whose custom
    /// batching rule maps the product along the first mapped input axis, which is axis-dependent when `axis_dependent`
    /// is set and has batching support when `batching_support` is set.
    fn product_registration(
        batching_support: bool,
        axis_dependent: bool,
    ) -> CustomRuleRegistration<Array, ArrayOperation<Array>> {
        let rule = |_: &BatchingLevel<ArrayType>, _: &[MemberTracer], inputs: &[MemberTracer], axes: &[BatchAxis]| {
            let axis = if axes[0].is_replicated() { axes[1] } else { axes[0] };
            Ok::<_, ProgramError>((vec![inputs[0].clone() * inputs[1].clone()], vec![axis]))
        };
        let mut definition = MemberDefinition::new("product").with_jvp_from_primal();
        if batching_support {
            definition = definition.with_batching_support();
        }
        CustomRuleRegistration::new(match axis_dependent {
            true => definition.with_axis_dependent_batching_rule(rule),
            false => definition.with_batching_rule(rule),
        })
    }

    /// Returns the key of the definition derived from a [`product_registration`] for calls of the provided kind with
    /// the provided active inputs.
    fn product_derivation_key(
        kind: CustomRuleDerivationKind,
        active_input_indices: Vec<usize>,
    ) -> CustomRuleDerivationKey {
        CustomRuleDerivationKey { kind, input_count: 2, active_input_indices, output_tangent_mask: vec![true] }
    }

    /// Applies the batching rule of the definition derived from `registration` for `derivation_key` directly (i.e.,
    /// without the caches and the discharge of the derived definition) to the call batched as `key` describes.
    fn derived_batching_rule_specialization(
        registration: &CustomRuleRegistration<Array, ArrayOperation<Array>>,
        derivation_key: CustomRuleDerivationKey,
        key: &CustomRuleBatchingRuleKey<ArrayType>,
    ) -> Result<CustomRuleSpecialization<Array, ArrayOperation<Array>>, DifferentiationError> {
        let derived = registration.reference().derived(derivation_key)?;
        derived.definition.derivation.as_ref().unwrap().batching_rule_specialization(derived.name(), key)
    }

    #[test]
    fn test_custom_rule_reference_specialization_eviction() {
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, false));
        let jvp = |extent: usize| {
            let x = Array::vector(vec![2f64; extent]).unwrap();
            let tangent = Array::vector(vec![1f64; extent]).unwrap();
            assert_eq!(
                call_jvp(&definition, x, tangent),
                Ok((
                    vec![Array::vector(vec![8f64; extent]).unwrap()],
                    vec![Array::vector(vec![4f64; extent]).unwrap()]
                )),
            );
        };
        let specialization = |extent: usize| {
            let r#type = ArrayType::new_static(DataType::F64, [extent]);
            let key = CustomRuleSpecializationKey {
                input_types: vec![r#type.clone()],
                output_types: vec![r#type],
                kept_outputs: None,
                non_differentiated_count: 0,
                tangent_activity: vec![true],
                levels: Vec::new(),
                discharged: false,
            };
            Arc::downgrade(&definition.reference().jvp_specialization(key).unwrap())
        };
        let baseline = Arc::strong_count(&definition.definition());
        let mut probes = Vec::new();
        for extent in 1..=CUSTOM_RULE_SPECIALIZATION_CAPACITY + 1 {
            jvp(extent);
            probes.push(specialization(extent));
        }
        assert_eq!(definition.caches().jvp_specializations.len(), CUSTOM_RULE_SPECIALIZATION_CAPACITY);
        assert_eq!(counters.counts(), (CUSTOM_RULE_SPECIALIZATION_CAPACITY + 1, 0, 0));

        // Bounded caches bound the retained memory: the evicted specialization is freed, and no cached program retains
        // the definition.
        assert!(probes[0].upgrade().is_none());
        assert!(probes[1..].iter().all(|probe| probe.upgrade().is_some()));
        assert_eq!(Arc::strong_count(&definition.definition()), baseline);

        // The evicted oldest specialization is traced again, while a resident one is reused.
        jvp(1);
        jvp(CUSTOM_RULE_SPECIALIZATION_CAPACITY + 1);
        assert_eq!(counters.counts(), (CUSTOM_RULE_SPECIALIZATION_CAPACITY + 2, 0, 0));
    }

    #[test]
    fn test_custom_rule_reference_specialization_outer_cache_hit() {
        // A batched specialization that is still cached is served without requesting the unbatched specialization from
        // which it was batched, so evicting the latter neither traces the rule again nor caches it again.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, false));
        let scalar_type = ArrayType::scalar(DataType::F64);
        let key = CustomRuleSpecializationKey {
            input_types: vec![scalar_type.clone()],
            output_types: vec![scalar_type],
            kept_outputs: None,
            non_differentiated_count: 0,
            tangent_activity: vec![true],
            levels: vec![CustomRuleBatchingLevel {
                level: BatchingLevel::new(BatchingLevelExtent::Static(3), None, ShardingDimension::Replicated),
                boundary_input_count: 0,
                input_axes: vec![BatchAxis::new(0)],
                output_axes: vec![BatchAxis::new(0)],
            }],
            discharged: false,
        };
        let first = definition.reference().jvp_specialization(key.clone()).unwrap();
        assert_eq!(definition.caches().jvp_specializations.len(), 2);
        assert_eq!(definition.caches().jvp_specializations.invalidate_entries_if(|key| key.levels.is_empty()), 1);
        let second = definition.reference().jvp_specialization(key).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(definition.caches().jvp_specializations.len(), 1);
        assert_eq!(counters.counts(), (1, 0, 0));
    }

    #[test]
    fn test_custom_rule_reference_specialization_retry_after_failure() {
        // A failed specialization is not retained, so the next request invokes the rule again.
        let attempts = Arc::new(AtomicUsize::new(0));
        let definition = CustomRuleRegistration::new(TestDefinition::new("flaky").with_jvp({
            let attempts = attempts.clone();
            move |primals, tangents| {
                if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                    return Err(ProgramError::InvalidArgument { message: "transient failure".to_string() });
                }
                let square = primals[0].clone() * primals[0].clone();
                Ok((vec![square.clone() * primals[0].clone()], vec![square * tangents[0].clone()]))
            }
        }));
        let x = Array::scalar(2f64).unwrap();
        let one = Array::scalar(1f64).unwrap();
        assert!(matches!(
            call_jvp(&definition, x.clone(), one.clone()),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "transient failure",
        ));
        assert!(definition.caches().jvp_specializations.is_empty());
        for _ in 0..2 {
            assert_eq!(
                call_jvp(&definition, x.clone(), one.clone()),
                Ok((vec![Array::scalar(8f64).unwrap()], vec![Array::scalar(4f64).unwrap()])),
            );
        }
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_custom_rule_reference_specialization_concurrent_cold_requests() {
        // Both threads enter the rule before either finishes, so both derive the same cold specialization. The
        // duplicate derivations produce one resident result, which later requests reuse.
        let barrier = Arc::new(Barrier::new(2));
        let invocations = Arc::new(AtomicUsize::new(0));
        let definition = CustomRuleRegistration::new(TestDefinition::new("concurrent").with_jvp({
            let barrier = barrier.clone();
            let invocations = invocations.clone();
            move |primals, tangents| {
                if invocations.fetch_add(1, Ordering::SeqCst) < 2 {
                    barrier.wait();
                }
                let square = primals[0].clone() * primals[0].clone();
                Ok((vec![square.clone() * primals[0].clone()], vec![square * tangents[0].clone()]))
            }
        }));
        let x = Array::scalar(2f64).unwrap();
        let one = Array::scalar(1f64).unwrap();
        let expected = (vec![Array::scalar(8f64).unwrap()], vec![Array::scalar(4f64).unwrap()]);
        std::thread::scope(|scope| {
            let threads = (0..2)
                .map(|_| {
                    let (definition, x, one) = (definition.clone(), x.clone(), one.clone());
                    scope.spawn(move || call_jvp(&definition, x, one))
                })
                .collect::<Vec<_>>();
            for thread in threads {
                assert_eq!(thread.join().unwrap(), Ok(expected.clone()));
            }
        });
        assert_eq!(invocations.load(Ordering::SeqCst), 2);
        assert_eq!(definition.caches().jvp_specializations.len(), 1);
        assert_eq!(call_jvp(&definition, x, one), Ok(expected));
        assert_eq!(invocations.load(Ordering::SeqCst), 2);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_custom_rule_reference_specialization_debug_recheck() {
        // A region-transform cache hit re-derives the linearization in debug builds and compares renderings. The
        // re-derivation reuses the retained specialization, so it neither invokes the rule nor changes the rendering.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, true));
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        let first = program.linearize().unwrap();
        let second = program.linearize().unwrap();
        assert_eq!(second.tangent().to_string(), first.tangent().to_string());
        assert_eq!(counters.counts(), (1, 0, 0));
    }

    #[test]
    fn test_custom_rule_reference_derived() {
        // Derivations of one key share a definition that the source registration owns, together with its caches, and
        // the definition keeps only its source definition alive (i.e., there is no strong cycle back to the caches).
        let registration = CustomRuleRegistration::new(
            MemberDefinition::new("sine")
                .with_jvp_from_primal()
                .with_batching_support()
                .with_batching_rule(|_, _, inputs, input_axes| Ok((inputs.to_vec(), input_axes.to_vec()))),
        );
        let key = |kind| CustomRuleDerivationKey {
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
        assert_eq!(registration.caches().derived_registrations.lock().unwrap().len(), 2);
        assert!(derived.caches.upgrade().is_some());

        // Dropping the source registration frees the derived registrations and their caches, and later derivations
        // reuse the derived definitions that are still alive, without caches, exactly like specializations.
        let source = registration.reference();
        drop(registration);
        assert!(derived.caches.upgrade().is_none());
        let fresh = source.derived(key(CustomRuleDerivationKind::Jvp)).unwrap();
        assert_eq!(fresh, derived);
        assert!(fresh.caches.upgrade().is_none());
        let derived_definition = Arc::downgrade(&derived.definition);
        drop((derived, pushforward, fresh));
        assert!(derived_definition.upgrade().is_none());

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
    fn test_custom_rule_reference_derived_capacity() {
        // The source registration retains as many derived registrations as specializations of each rule. A derived
        // definition that is still alive is reused for its key, whether or not its registration was evicted or the
        // source registration was dropped, so that it keeps its identity.
        let registration = CustomRuleRegistration::new(
            MemberDefinition::new("sine")
                .with_jvp_from_primal()
                .with_batching_support()
                .with_batching_rule(|_, _, inputs, input_axes| Ok((inputs.to_vec(), input_axes.to_vec()))),
        );
        let key = |input_count| CustomRuleDerivationKey {
            kind: CustomRuleDerivationKind::Jvp,
            input_count,
            active_input_indices: vec![0],
            output_tangent_mask: vec![true],
        };
        let derived = (1..=CUSTOM_RULE_SPECIALIZATION_CAPACITY + 1)
            .map(|input_count| registration.reference().derived(key(input_count)).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            registration.caches().derived_registrations.lock().unwrap().len(),
            CUSTOM_RULE_SPECIALIZATION_CAPACITY,
        );
        assert_eq!(registration.reference().derived(key(1)).unwrap(), derived[0]);
        let source = registration.reference();
        drop(registration);
        assert_eq!(source.derived(key(2)).unwrap(), derived[1]);
    }

    #[test]
    fn test_unavailable_custom_rules_from_source() {
        // Unavailable rules keep the identity and the description of their source.
        let counters = Arc::new(RuleCounters::default());
        let registration = CustomRuleRegistration::new(
            member_cube_definition(&counters)
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
        assert_eq!(unavailable.batched_output_axes, Some(vec![BatchAxis::replicated()]));
        assert!(!unavailable.has_batching_rule);
        assert!(unavailable.has_sequential_batching);

        let registration = CustomRuleRegistration::new(
            MemberDefinition::new("sine")
                .with_jvp_from_primal()
                .with_batching_rule(|_, _, inputs, input_axes| Ok((inputs.to_vec(), input_axes.to_vec()))),
        );
        let source = registration.reference();
        let unavailable = UnavailableCustomRules::from_source(&source);
        assert_eq!(unavailable.id, source.id());
        assert_eq!(unavailable.name, "sine");
        assert!(!unavailable.has_jvp);
        assert!(unavailable.derives_jvp_from_primal);
        assert!(!unavailable.has_vjp);
        assert_eq!(unavailable.batched_output_axes, None);
        assert!(unavailable.has_batching_rule);
        assert!(!unavailable.has_sequential_batching);
        assert_eq!(counters.counts(), (0, 0, 0));
    }

    #[test]
    fn test_lifted_custom_rules() {
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(member_cube_definition(&counters));
        let lifted = LiftedCustomRules::Member(definition.reference());
        let source = |rules: &LiftedCustomRules<Array, ArrayOperation<Array>>| {
            <LiftedCustomRules<_, _> as CustomRuleSource<ArrayIrValue<Array>, ArrayIrOperation<Array>>>::name(rules)
                .to_owned()
        };
        assert_eq!(source(&lifted), "cube");
        let key = |input_type: ArrayIrType, levels| CustomRuleSpecializationKey {
            input_types: vec![input_type.clone()],
            output_types: vec![input_type],
            kept_outputs: None,
            non_differentiated_count: 0,
            tangent_activity: vec![true],
            levels,
            discharged: false,
        };
        let specialize = |rules: &LiftedCustomRules<Array, ArrayOperation<Array>>, key| {
            CustomRuleSpecializer::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::jvp_specialization(rules, key)
                .map(|_| ())
        };

        // Types without a member representation are rejected at the first derivative request.
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F64)));
        assert!(matches!(
            specialize(&lifted, key(reference_type, Vec::new())),
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
            specialize(&lifted, key(ArrayType::scalar(DataType::F64).into(), vec![level])),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`composite` `cube` cannot differentiate a batched call whose batch extent has type \
                               `dimension<items ∈ [1, 9)>`, which the family in which its rules were registered \
                               cannot represent; differentiate the call before batching it instead",
        ));

        // Unavailable rules keep the identity and description of their definition but reject every derivative request.
        let unavailable = LiftedCustomRules::<Array, ArrayOperation<Array>>::Unavailable(
            UnavailableCustomRules::from_source(&definition.reference()),
        );
        assert_eq!(unavailable, lifted);
        assert_eq!(source(&unavailable), "cube");
        assert!(matches!(
            specialize(&unavailable, key(ArrayType::scalar(DataType::F64).into(), Vec::new())),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`composite` `cube` was converted from a family whose values its current family \
                               cannot represent, so its derivative rules are unavailable",
        ));
        assert_eq!(counters.counts(), (0, 0, 0));
    }

    #[test]
    fn test_lifted_custom_rules_derived() {
        // A lifted source derives in the family that owns its definition, and unavailable rules stay unavailable.
        let registration = CustomRuleRegistration::new(
            MemberDefinition::new("sine")
                .with_jvp_from_primal()
                .with_batching_support()
                .with_batching_rule(|_, _, inputs, input_axes| Ok((inputs.to_vec(), input_axes.to_vec()))),
        );
        let key = CustomRuleDerivationKey {
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
    fn test_lifted_custom_rules_converted_specialization_cache() {
        // The converted program of each specialization is cached with the member definition's specializations, so
        // repeated requests share it instead of converting the member program again.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(member_cube_definition(&counters));
        let lifted = LiftedCustomRules::Member(definition.reference());
        let scalar_type = ArrayIrType::from(ArrayType::scalar(DataType::F64));
        let key = CustomRuleSpecializationKey {
            input_types: vec![scalar_type.clone()],
            output_types: vec![scalar_type],
            kept_outputs: None,
            non_differentiated_count: 0,
            tangent_activity: vec![true],
            levels: Vec::new(),
            discharged: false,
        };
        let specialize = || {
            CustomRuleSpecializer::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::jvp_specialization(
                &lifted,
                key.clone(),
            )
            .unwrap()
        };
        let first = specialize();
        let second = specialize();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(counters.counts(), (1, 0, 0));
    }

    #[test]
    fn test_custom_rule_derivation_batching_rule_specialization() {
        // Structural mode: with replicated primals, the source rule applied to them is replicated as well, so the JVP
        // of its program is batched structurally over the mapped tangents (e.g., in a forward-mode Jacobian), which
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
            &batching_rule_key(
                vec![scalar_type.clone(), scalar_type.clone(), vector_type.clone(), vector_type],
                vec![scalar_type; 4],
                vec![replicated, replicated, mapped, mapped],
            ),
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
            &batching_rule_key(
                vec![vector_type.clone(), scalar_type.clone(), vector_type],
                vec![scalar_type; 3],
                vec![mapped, replicated, mapped],
            ),
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
            &batching_rule_key(
                vec![
                    ArrayType::new_static(DataType::F64, [3, 2]),
                    scalar_type.clone(),
                    ArrayType::new_static(DataType::F64, [2, 3]),
                ],
                vec![item_type.clone(), scalar_type, item_type],
                vec![BatchAxis::new(0), BatchAxis::replicated(), BatchAxis::new(1)],
            ),
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
        let key = batching_rule_key(
            vec![vector_type.clone(), scalar_type.clone(), vector_type],
            vec![scalar_type; 3],
            vec![mapped, replicated, mapped],
        );
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
        // A derivation batches the source rule's derivative with the source's batcher, which it cannot do without it.
        let registration = product_registration(false, false);
        let scalar_type = ArrayType::scalar(DataType::F64);
        let vector_type = ArrayType::new_static(DataType::F64, [3]);
        assert!(matches!(
            derived_batching_rule_specialization(
                &registration,
                product_derivation_key(CustomRuleDerivationKind::Jvp, vec![0]),
                &batching_rule_key(
                    vec![vector_type.clone(), scalar_type.clone(), vector_type],
                    vec![scalar_type; 3],
                    vec![BatchAxis::new(0), BatchAxis::replicated(), BatchAxis::new(0)],
                ),
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
                CustomRuleDerivationKey {
                    output_tangent_mask: vec![false],
                    ..product_derivation_key(CustomRuleDerivationKind::Jvp, vec![0])
                },
                &batching_rule_key(
                    vec![vector_type.clone(), scalar_type.clone(), vector_type],
                    vec![scalar_type; 3],
                    vec![BatchAxis::new(0), BatchAxis::replicated(), BatchAxis::new(0)],
                ),
            ),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "`composite` `jvp(product)` batching rule program has live tangents for the outputs \
                               [true], but the derivative of its primal has them for the outputs [false]",
        ));
    }

    #[test]
    fn test_custom_rule_derivation_axis_dependent_batching_rule_specialization() {
        // An axis-dependent source rule is applied once, at the primal axes, and the tangent `xẏ` of the replicated `y`
        // is the diagonal of the outer product of its mapped tangent with the rule's batch.
        let registration = product_registration(true, true);
        let scalar_type = ArrayType::scalar(DataType::F64);
        let vector_type = ArrayType::new_static(DataType::F64, [3]);
        let (mapped, replicated) = (BatchAxis::new(0), BatchAxis::replicated());
        let vector = |values: Vec<f64>| Array::vector(values).unwrap();
        let primals = vec![vector(vec![1.0, 2.0, 3.0]), Array::scalar(3.0).unwrap()];
        let specialization = derived_batching_rule_specialization(
            &registration,
            product_derivation_key(CustomRuleDerivationKind::Jvp, vec![1]),
            &batching_rule_key(
                vec![vector_type.clone(), scalar_type.clone(), vector_type.clone()],
                vec![scalar_type.clone(); 3],
                vec![mapped, replicated, mapped],
            ),
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
            &batching_rule_key(
                vec![vector_type.clone(), scalar_type.clone(), vector_type.clone(), vector_type],
                vec![scalar_type; 4],
                vec![mapped, replicated, mapped, mapped],
            ),
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
    fn test_custom_rule_definition_batched_specialization_dynamic_extent() {
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
        let key = CustomRuleSpecializationKey {
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
    fn test_conform_rule_output_types() {
        // A rule program whose outputs have the expected types is returned unchanged. Only its leading outputs (e.g.,
        // the primal outputs of a forward rule, which its residuals follow) or the selected ones are compared.
        let (f64_type, f32_type) = (ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F32));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = vec![builder.add_input(f64_type.clone()), builder.add_input(f32_type.clone())];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(inputs, vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();
        let rendering = program.to_string();
        let conform = |expected: &[ArrayType], selected: Option<&[usize]>| {
            conform_rule_output_types("pair", "forward", program.clone(), expected, selected)
                .map(|program| program.to_string())
        };
        assert_eq!(conform(&[f64_type.clone(), f32_type.clone()], None), Ok(rendering.clone()));
        assert_eq!(conform(std::slice::from_ref(&f64_type), None), Ok(rendering.clone()));
        assert_eq!(conform(std::slice::from_ref(&f32_type), Some(&[1])), Ok(rendering));
        assert!(matches!(
            conform(std::slice::from_ref(&f32_type), None),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`composite` `pair` forward rule output type signature mismatch: expected [f32[]] but \
                               got [f64[]]",
        ));
    }

    #[test]
    fn test_conform_rule_output_types_different_type_identities() {
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
            conform_rule_output_types("pair", "JVP", program, &[vector_type(&first), vector_type(&first)], None),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`composite` `pair` JVP rule output type signature mismatch: expected [f32[n], f32[n]] \
                               but got [f32[n], f32[n]] with different type identities",
        ));
    }

    #[test]
    fn test_derive_bijective_identity_renaming() {
        let bounds = DimensionBounds::new(1, Some(9)).unwrap();
        let (a, b, c) = (
            DimensionVariable::new("a", bounds),
            DimensionVariable::new("b", bounds),
            DimensionVariable::new("c", bounds),
        );
        let vector_type =
            |variable: &DimensionVariable| ArrayType::new(DataType::F32, Shape::new(vec![variable.clone().into()]));

        // Equal signatures are related by the identity renaming.
        let renaming = derive_bijective_identity_renaming(&[vector_type(&a)], &[vector_type(&a)]).unwrap();
        assert!(renaming.is_identity());

        // A bijective renaming maps each source identity onto its target identity.
        let renaming = derive_bijective_identity_renaming(
            &[vector_type(&a), vector_type(&b)],
            &[vector_type(&c), vector_type(&b)],
        )
        .unwrap();
        assert!(!renaming.is_identity());
        assert_eq!(vector_type(&a).rename_identities(&renaming), Ok(vector_type(&c)));
        assert_eq!(vector_type(&b).rename_identities(&renaming), Ok(vector_type(&b)));

        // Renamings that merge or split identities, and signatures that differ beyond their identities, have none.
        assert!(
            derive_bijective_identity_renaming(
                &[vector_type(&a), vector_type(&b)],
                &[vector_type(&c), vector_type(&c)]
            )
            .is_none()
        );
        assert!(
            derive_bijective_identity_renaming(
                &[vector_type(&a), vector_type(&a)],
                &[vector_type(&b), vector_type(&c)]
            )
            .is_none()
        );
        assert!(derive_bijective_identity_renaming(&[vector_type(&a)], &[vector_type(&a), vector_type(&a)]).is_none());
        assert!(
            derive_bijective_identity_renaming(
                &[vector_type(&a)],
                &[ArrayType::new(DataType::F64, Shape::new(vec![a.clone().into()]))]
            )
            .is_none()
        );
    }
}
