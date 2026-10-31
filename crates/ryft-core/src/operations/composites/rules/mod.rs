//! Retained custom rule sets, whose derivative rules are callbacks rather than programs. A [`CustomRuleDefinition`]
//! holds a human-readable name, an optional forward mode differentiation (i.e., Jacobian-Vector Product) rule, optional
//! reverse mode differentiation (i.e., Vector-Jacobian Product) forward and backward rules, and optional batching
//! support. A [`CompositeOperation`](crate::CompositeOperation) with retained rules calls a definition over a traced
//! primal program: ordinary execution uses only the primal, forward mode traces and replays the Jacobian-Vector Product
//! (JVP) rule, and reverse mode traces the Vector-Jacobian Product (VJP) forward rule and stages a
//! [`CompositeTransposeOperation`] whose transposition specializes the backward rule against the actual cotangent
//! destinations. Every rule is traced lazily, on the first request of each static specialization, and cached by the
//! definition's [`CustomRuleRegistration`]. Batching a call or carrier likewise traces no derivative rule: it records
//! the batching level, and a later derivative request structurally batches the rule program traced at the unbatched
//! types. Only a custom batching rule (set via [`CustomRuleDefinition::with_batching_rule`]) is traced when a call is
//! batched, because its program becomes the batched call's primal region. Refer to the documentation of
//! [`CompositeOperation`](crate::CompositeOperation) and [`CompositeTransposeOperation`] for more information.
//!
//! This module contains flat, family-level representations that are hidden behind the public
//! [`composite`](fn@crate::composite) builder. That builder adapts structured closures to the flat rule interfaces
//! of this module. Also, rules may receive materialized zeros (e.g., [`CustomRuleDefinition::with_jvp`]) or structural
//! zeros (e.g., [`CustomRuleDefinition::with_symbolic_zero_jvp`]).

use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::sync::Arc;

use crate::arrays::ShardingDimension;
use crate::batching::{BatchAxis, BatchingError, BatchingLevel, BatchingLevelExtent};
use crate::differentiation::{CotangentDestinationKind, DifferentiationError, TransposedProgramCotangent};
use crate::parameters::Parameter;
use crate::programs::{Program, ProgramError, Type, TypeError, TypeIdentityRenaming, Typed};
use crate::tracing::{Tracer, TracingContext};

#[cfg(doc)]
use crate::operations::composites::operations::CompositeTransposeOperation;

pub mod definition;
pub mod registration;
pub mod specialization;

// TODO(eaplatanios): Review from here onwards.

pub(in crate::operations::composites) use definition::CustomVjpBackward;
pub use definition::{CustomRuleDefinition, CustomVjpOperation};
pub use registration::{CustomRuleReference, CustomRuleRegistration, WeakCustomRuleRegistration};
pub use specialization::{CustomRuleSource, CustomRuleSpecializer, LiftedCustomRules, UnavailableCustomRules};
pub(in crate::operations::composites) use specialization::{boundary_input_count, validate_rule_output_types};

/// Maximum number of specializations that each rule of a [`CustomRuleDefinition`] retains.
pub(super) const CUSTOM_RULE_SPECIALIZATION_CAPACITY: usize = 16;

/// Tracer over which retained custom rules are written. It is typed by the registration family's staged constant and
/// operation types only, so a rule can be traced lazily under whichever transform later requests it.
pub type CustomRuleTracer<V, O> = Tracer<TracingContext<V, O>>;

/// Transposition of a backward-rule specialization source program with respect to the provided inputs and destination
/// kinds, which returns the transposed program together with the representation of each selected input's cotangent
/// (refer to [`TransposedProgram`](crate::TransposedProgram)), so that the cotangents that a backward rule declares to
/// be structural zeros stay structural zeros instead of being materialized. [`CustomRuleDefinition::with_vjp`] captures
/// it where the operation family's transposition bounds are known, because requiring them on the carrier's own
/// transposition rule would form a cycle with the family's payload bounds.
type CustomRuleTransposer<V, O> = dyn Fn(
        &Program<V, O, Vec<V>, Vec<V>>,
        &[usize],
        &[CotangentDestinationKind],
    ) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Vec<TransposedProgramCotangent>), DifferentiationError>
    + Send
    + Sync;

/// Reference discharge of a traced rule program whose call was discharged, which replaces the program's local reference
/// state with explicitly threaded values. [`CustomRuleDefinition::with_reference_discharge`] captures it where the
/// operation family's discharge bounds are known, because requiring them on the operations' own rules would form a
/// cycle with the family's payload bounds (and would exclude families without references).
type CustomRuleDischarger<V, O> =
    dyn Fn(Program<V, O, Vec<V>, Vec<V>>) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError> + Send + Sync;

/// Kind of values that a batched rule program returns, which determines how an output that is mapped while its
/// required batch axis is replicated is reconciled.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CustomRuleBatchedOutputs {
    /// Primal or tangent values, for which such an output violates the required layout and is rejected.
    Values,

    /// Cotangents of replicated inputs, for which such an output is summed along its mapped axis (i.e., the transpose
    /// of broadcasting the input across the batch).
    Cotangents,
}

/// Structural batching of a traced rule program at one recorded [`BatchingLevel`] with the provided input batch axes.
/// Each output is aligned to its required batch axis, or keeps its natural batch axis when the requirement is [`None`].
/// The result is the batched program, which consumes the level's boundary inputs before the source program's inputs,
/// together with its output batch axes. [`CustomRuleDefinition::with_batching_support`] captures it where the
/// operation family's batching bounds are known, because requiring them on the operations' own rules would form a cycle
/// with the family's payload bounds.
type CustomRuleBatcher<V, O> = dyn Fn(
        &Program<V, O, Vec<V>, Vec<V>>,
        &[BatchAxis],
        &[Option<BatchAxis>],
        &BatchingLevel<<V as Typed>::Type>,
        CustomRuleBatchedOutputs,
    ) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Vec<BatchAxis>), BatchingError>
    + Send
    + Sync;

/// Forward mode differentiation of a traced rule program with respect to the provided inputs, which returns its fused
/// Jacobian-Vector Product (JVP) program (refer to [`Program::jvp_with_respect_to`]). The batching rules of derived
/// definitions (refer to [`CustomRuleDerivation`]) differentiate the batched programs of their source's custom
/// batching rule with it. [`CustomRuleDefinition::with_batching_rule`] captures it where the operation family's
/// differentiation bounds are known, because requiring them on the operations' own rules would form a cycle with the
/// family's payload bounds.
type CustomRuleDifferentiator<V, O> = dyn Fn(
        // Program to differentiate.
        &Program<V, O, Vec<V>, Vec<V>>,
        // Indices of the inputs with respect to which the program is differentiated.
        &[usize],
    ) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError>
    + Send
    + Sync;

/// Construction of the scan with which a call batched sequentially loops over its batch items (refer to
/// [`CustomRuleDefinition::with_sequential_batching`]). It receives the extent of the batch axis (i.e., the scan's trip
/// count), the number of carries (i.e., of the call's replicated inputs), the placement of the batch axis, and the
/// number of stacked outputs (i.e., of the call's outputs), and it returns the scan operation together with the type
/// of the slice index that the scan body receives first. [`CustomRuleDefinition::with_sequential_batching`] captures
/// it where the family's scan bounds are known, because requiring them on the operations' own rules would restrict
/// every family that contains the operations to scannable types.
type CustomRuleSequentialScan<V, O> =
    dyn Fn(usize, usize, &ShardingDimension, usize) -> Result<(O, <V as Typed>::Type), ProgramError> + Send + Sync;

/// One batching level applied to a [`CompositeOperation`](crate::CompositeOperation) or
/// [`CompositeTransposeOperation`] after it was staged. Batching such an operation batches no derivative rule. Instead,
/// it records this level so that a later derivative request can batch the rule program that it traces at the unbatched
/// types.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CustomRuleBatchingLevel<T> {
    /// Context-neutral description of the level.
    pub(super) level: BatchingLevel<T>,

    /// Number of policy boundary inputs (e.g., a first-class batch extent) that batching at this level prepended to the
    /// operation's inputs (refer to the documentation of [`BatchingPolicy::boundary_inputs`]).
    pub(super) boundary_input_count: usize,

    /// Batch axes of the operation's inputs at this level, excluding the boundary inputs prepended at this level.
    pub(super) input_axes: Vec<BatchAxis>,

    /// Batch axes of the operation's outputs at this level.
    pub(super) output_axes: Vec<BatchAxis>,
}

/// Batching applied to a [`CompositeOperation`](crate::CompositeOperation) or [`CompositeTransposeOperation`]
/// after it was staged, which determines how its lazily traced rule programs are batched.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct CustomRuleBatching<T> {
    /// Unbatched types of the operation's inputs, excluding all boundary inputs.
    pub(super) input_types: Vec<T>,

    /// Unbatched types of the operation's outputs.
    pub(super) output_types: Vec<T>,

    /// Batching levels, from the innermost (i.e., the first one applied) to the outermost.
    pub(super) levels: Vec<CustomRuleBatchingLevel<T>>,
}

impl<T: Type> CustomRuleBatching<T> {
    /// Returns the number of boundary inputs that all levels together prepended to the operation's inputs.
    pub(super) fn boundary_input_count(&self) -> usize {
        boundary_input_count(&self.levels)
    }

    /// Returns this [`CustomRuleBatching`] with its types converted into the type family `U`, as when its operation is
    /// converted into a family that contains the operation's family.
    pub(super) fn map_types<U: Type + From<T>>(self) -> CustomRuleBatching<U> {
        let map_types = |types: Vec<T>| types.into_iter().map(U::from).collect::<Vec<_>>();
        CustomRuleBatching {
            input_types: map_types(self.input_types),
            output_types: map_types(self.output_types),
            levels: self
                .levels
                .into_iter()
                .map(|level| CustomRuleBatchingLevel {
                    level: BatchingLevel::new(
                        match level.level.extent() {
                            BatchingLevelExtent::Static(extent) => BatchingLevelExtent::Static(*extent),
                            BatchingLevelExtent::Dynamic(r#type) => {
                                BatchingLevelExtent::Dynamic(U::from(r#type.clone()))
                            }
                        },
                        level.level.axis_name().map(str::to_owned),
                        level.level.axis_sharding().clone(),
                    ),
                    boundary_input_count: level.boundary_input_count,
                    input_axes: level.input_axes,
                    output_axes: level.output_axes,
                })
                .collect(),
        }
    }

    /// Returns this [`CustomRuleBatching`] after renaming the type identities in its types and levels.
    pub(super) fn rename_identities(&self, renaming: &TypeIdentityRenaming<T::Identity>) -> Result<Self, TypeError> {
        let rename =
            |types: &[T]| types.iter().map(|r#type| r#type.rename_identities(renaming)).collect::<Result<Vec<_>, _>>();
        Ok(Self {
            input_types: rename(&self.input_types)?,
            output_types: rename(&self.output_types)?,
            levels: self
                .levels
                .iter()
                .map(|level| {
                    Ok(CustomRuleBatchingLevel { level: level.level.rename_identities(renaming)?, ..level.clone() })
                })
                .collect::<Result<Vec<_>, TypeError>>()?,
        })
    }
}

impl<T: Display> Display for CustomRuleBatching<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let axes = |axes: &[BatchAxis]| axes.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ");
        write!(formatter, "[")?;
        for (index, level) in self.levels.iter().enumerate() {
            if index > 0 {
                write!(formatter, ", ")?;
            }
            write!(formatter, "(extent=")?;
            match level.level.extent() {
                BatchingLevelExtent::Static(extent) => write!(formatter, "{extent}")?,
                BatchingLevelExtent::Dynamic(r#type) => write!(formatter, "{type}")?,
            }
            if let Some(axis_name) = level.level.axis_name() {
                write!(formatter, ", axis_name={axis_name:?}")?;
            }
            write!(
                formatter,
                ", input_axes=[{}], output_axes=[{}])",
                axes(&level.input_axes),
                axes(&level.output_axes)
            )?;
        }
        write!(formatter, "]")
    }
}

/// Specialization key of the forward mode and reverse mode primal rules of a [`CustomRuleDefinition`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CustomRuleSpecializationKey<T> {
    /// Unbatched types of every input of the call, excluding boundary inputs.
    pub(super) input_types: Vec<T>,

    /// Unbatched types of every output of the call. The key must include them, even though rules are traced from the
    /// inputs alone, because a specialization is validated against them and its results are partitioned by them: two
    /// calls with the same inputs but different primal outputs must never share a specialization.
    pub(super) output_types: Vec<T>,

    /// Whether each output of the rule is an output of the call, or [`None`] when the call returns every output of the
    /// rule (refer to [`CompositeOperation`](crate::CompositeOperation)'s boundary pruning). When present, it has one
    /// entry per output of the rule, and [`Self::output_types`] lists the types of the kept outputs only, against which
    /// the specialization is validated. A forward mode specialization then returns only the kept outputs and their
    /// tangents, while a reverse mode forward specialization keeps returning every output of the rule, because the
    /// backward rule receives a seed for each of them.
    pub(super) kept_outputs: Option<Vec<bool>>,

    /// Number of leading inputs that parameterize the rule without being differentiated, excluding boundary inputs.
    pub(super) non_differentiated_count: usize,

    /// Whether each differentiated input's tangent is active (i.e., not a structural zero) in a forward mode request,
    /// which is empty for reverse mode forward requests, whose rules receive no tangents. A forward mode specialization
    /// receives only the active tangents, and it materializes the structural zeros itself (or hands them to a rule that
    /// receives structural zeros), so that the zeros that the rule propagates remain recognizable as zeros.
    pub(super) tangent_activity: Vec<bool>,

    /// Batching levels of the call, from the innermost to the outermost, which are empty for an unbatched call.
    pub(super) levels: Vec<CustomRuleBatchingLevel<T>>,

    /// Whether the call's reference state was discharged, in which case the specialization is discharged as well.
    pub(super) discharged: bool,
}

impl<T: Type + Eq + Hash> CustomRuleSpecializationKey<T> {
    /// Returns the key of the specialization from which this key's specialization is batched (i.e., without its
    /// outermost level), together with that level, or [`None`] when this key is unbatched.
    fn split_outermost_level(&self) -> Option<(Self, &CustomRuleBatchingLevel<T>)> {
        let (outermost, inner) = self.levels.split_last()?;
        Some((Self { levels: inner.to_vec(), ..self.clone() }, outermost))
    }

    /// Returns the number of outputs of the rule, which exceeds the number of outputs of the call when the call keeps
    /// only some of them (refer to [`Self::kept_outputs`]).
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when the kept outputs do not match the number of output types.
    pub(super) fn rule_output_count(&self) -> Result<usize, ProgramError> {
        let Some(kept_outputs) = &self.kept_outputs else {
            return Ok(self.output_types.len());
        };
        let kept_output_count = kept_outputs.iter().filter(|kept| **kept).count();
        if kept_output_count != self.output_types.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "custom rule specialization key keeps {kept_output_count} rule outputs but has {} output types",
                self.output_types.len(),
            )));
        }
        Ok(kept_outputs.len())
    }

    /// Returns the positions of the outputs of the call among the `rule_output_count` outputs of the rule.
    pub(super) fn kept_output_indices(&self, rule_output_count: usize) -> Vec<usize> {
        (0..rule_output_count).filter(|&index| self.is_output_kept(index)).collect()
    }

    /// Returns whether the output of the rule at `index` is an output of the call.
    fn is_output_kept(&self, index: usize) -> bool {
        self.kept_outputs.as_ref().is_none_or(|kept_outputs| kept_outputs[index])
    }

    /// Returns this key with its types converted into the type family `U` (refer to [`CustomRuleBatching::map_types`]),
    /// as when the carrier that stores it is converted into a family that contains the carrier's family.
    pub(super) fn map_types<U: Type + From<T>>(self) -> CustomRuleSpecializationKey<U> {
        let Self {
            input_types,
            output_types,
            kept_outputs,
            non_differentiated_count,
            tangent_activity,
            levels,
            discharged,
        } = self;
        let CustomRuleBatching { input_types, output_types, levels } =
            CustomRuleBatching { input_types, output_types, levels }.map_types();
        CustomRuleSpecializationKey {
            input_types,
            output_types,
            kept_outputs,
            non_differentiated_count,
            tangent_activity,
            levels,
            discharged,
        }
    }

    /// Returns this key after renaming the type identities in its types and levels (refer to
    /// [`CustomRuleBatching::rename_identities`]).
    pub(super) fn rename_identities(&self, renaming: &TypeIdentityRenaming<T::Identity>) -> Result<Self, TypeError> {
        let batching = CustomRuleBatching {
            input_types: self.input_types.clone(),
            output_types: self.output_types.clone(),
            levels: self.levels.clone(),
        }
        .rename_identities(renaming)?;
        Ok(Self {
            input_types: batching.input_types,
            output_types: batching.output_types,
            levels: batching.levels,
            ..self.clone()
        })
    }
}

/// Specialization key of the backward rule of a [`CustomRuleDefinition`]. It contains only static information, and
/// never a runtime reference identity or the contents of a value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CustomRuleBackwardSpecializationKey<T> {
    /// Unbatched types of the known leading inputs (i.e., the non-differentiated inputs, the residuals, and the seed
    /// extents), excluding boundary inputs.
    pub(super) leading_input_types: Vec<T>,

    /// Number of trailing leading inputs that carry seed extents (refer to [`CompositeTransposeOperation`]).
    pub(super) seed_extent_count: usize,

    /// Unbatched tangent types of the differentiated inputs.
    pub(super) input_tangent_types: Vec<T>,

    /// Unbatched tangent types of the outputs.
    pub(super) output_tangent_types: Vec<T>,

    /// Unbatched type of each output cotangent seed, or [`None`] for a structural-zero seed.
    pub(super) seed_types: Vec<Option<T>>,

    /// Destination kind of each differentiated input's cotangent.
    pub(super) destination_kinds: Vec<CotangentDestinationKind>,

    /// Batching levels of the carrier, from the innermost to the outermost, which are empty for an unbatched carrier.
    pub(super) levels: Vec<CustomRuleBatchingLevel<T>>,

    /// Whether the carrier's reference state was discharged, in which case the specialization is discharged as well.
    pub(super) discharged: bool,
}

impl<T: Type + Eq + Hash> CustomRuleBackwardSpecializationKey<T> {
    /// Returns the key of the specialization from which this key's specialization is batched (i.e., without its
    /// outermost level), together with that level, or [`None`] when this key is unbatched.
    fn split_outermost_level(&self) -> Option<(Self, &CustomRuleBatchingLevel<T>)> {
        let (outermost, inner) = self.levels.split_last()?;
        Some((Self { levels: inner.to_vec(), ..self.clone() }, outermost))
    }
}

/// Specialization key of the custom batching rule of a [`CustomRuleDefinition`] (refer to
/// [`CustomRuleDefinition::with_batching_rule`]). It contains only static information: the level, the types and batch
/// axes of the batched call's inputs, and whether the call was discharged.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CustomRuleBatchingRuleKey<T> {
    /// Batching level at which the call is batched.
    pub(super) level: BatchingLevel<T>,

    /// Types of the level's boundary inputs (e.g., a first-class batch extent).
    pub(super) boundary_input_types: Vec<T>,

    /// Types of the call's batched inputs, which include any boundary inputs of earlier levels.
    pub(super) input_types: Vec<T>,

    /// Per-item types of the call's batched inputs at this level (i.e., the input types of the call's primal region
    /// before this level batched it), which derived batching rules need to batch programs structurally (refer to
    /// [`CustomRuleDerivation`]).
    pub(super) unbatched_input_types: Vec<T>,

    /// Batch axis of each batched input.
    pub(super) input_axes: Vec<BatchAxis>,

    /// Whether the call's reference state was discharged, in which case the specialization, which becomes the primal
    /// region of the batched call, is discharged as well.
    pub(super) discharged: bool,
}

/// Kind of call that a derived definition (refer to [`CustomRuleDerivation`]) stages for the forward mode derivative
/// of a call whose forward mode rule is derived from its primal.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(super) enum CustomRuleDerivationKind {
    /// Call that maps the source call's inputs and the active tangents of its differentiated inputs to its outputs and
    /// their live tangents, which fused differentiation contexts stage.
    Jvp,

    /// Call that maps the source call's inputs and the active tangents of its differentiated inputs to the live
    /// tangents of its outputs only, which partitioned differentiation contexts (e.g., linearization) stage on their
    /// tangent side.
    Pushforward,
}

/// Key of a definition derived from a source [`CustomRuleDefinition`] (refer to [`CustomRuleSpecializer::derived`]). It
/// contains only static information about the derived calls, so that calls whose primal regions differ share one
/// derived definition.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CustomRuleDerivationKey {
    /// Kind of the derived calls.
    pub(super) kind: CustomRuleDerivationKind,

    /// Number of inputs of the source call.
    pub(super) input_count: usize,

    /// Indices of the source call's inputs whose tangents are active, in increasing order, which is the order of the
    /// derived calls' tangent inputs.
    pub(super) active_input_indices: Vec<usize>,

    /// Whether each output of the source call has a live tangent, in output order, which determines the derived calls'
    /// tangent outputs.
    pub(super) output_tangent_mask: Vec<bool>,
}

/// Request for one specialization of a retained rule of a [`CustomRuleDefinition`], which identifies both the cache
/// that retains it and, when no cache is alive, the in-flight marker that rejects its recursive production.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum CustomRuleSpecializationRequest<T> {
    /// Specialization of the forward mode rule.
    Jvp(CustomRuleSpecializationKey<T>),

    /// Specialization of the reverse mode forward rule.
    Forward(CustomRuleSpecializationKey<T>),

    /// Destination-specialized transpose of the reverse mode backward rule.
    Backward(CustomRuleBackwardSpecializationKey<T>),

    /// Batched primal program produced by the custom batching rule.
    BatchingRule(CustomRuleBatchingRuleKey<T>),
}

/// Program produced by specializing a retained custom rule, together with the batch axes of its outputs at the
/// outermost batching level of its key (which are empty for an unbatched specialization).
pub struct CustomRuleSpecialization<V: Typed + Parameter, O> {
    /// Specialized program.
    pub(super) program: Program<V, O, Vec<V>, Vec<V>>,

    /// Batch axes of the specialized program's outputs.
    pub(super) output_axes: Vec<BatchAxis>,

    /// Representation of the cotangent of each differentiated input of a backward-rule specialization, in input order,
    /// of which only the [`TransposedProgramCotangent::Output`]s consume an output of the program, and which is empty
    /// for every other specialization. A structural-zero cotangent (e.g., one that the backward rule declares with
    /// [`MaybeZero::Zero`]) is [`TransposedProgramCotangent::Zero`] and contributes nothing to its destination.
    pub(super) cotangents: Vec<TransposedProgramCotangent>,
}

impl<V: Typed + Parameter, O> CustomRuleSpecialization<V, O> {
    /// Creates a new [`CustomRuleSpecialization`] of a rule other than a backward rule, which has no cotangents.
    pub(super) fn new(program: Program<V, O, Vec<V>, Vec<V>>, output_axes: Vec<BatchAxis>) -> Self {
        Self { program, output_axes, cotangents: Vec::new() }
    }
}

/// Shared [`CustomRuleSpecialization`], as retained by the specialization caches of a [`CustomRuleDefinition`].
pub type CustomRuleProgram<V, O> = Arc<CustomRuleSpecialization<V, O>>;

#[cfg(test)]
mod tests {
    use crate::arrays::{Array, ArrayOperation, ArrayType, ShardingDimension};

    use super::*;

    /// Tracer over which the rules of a [`MemberDefinition`] are written.
    pub(super) type MemberTracer = CustomRuleTracer<Array, ArrayOperation<Array>>;

    /// Returns the key of the custom batching rule of an undischarged call batched at a static level of three items,
    /// whose batched inputs have the provided types and batch axes.
    pub(super) fn batching_rule_key(
        input_types: Vec<ArrayType>,
        unbatched_input_types: Vec<ArrayType>,
        input_axes: Vec<BatchAxis>,
    ) -> CustomRuleBatchingRuleKey<ArrayType> {
        CustomRuleBatchingRuleKey {
            level: BatchingLevel::new(BatchingLevelExtent::Static(3), None, ShardingDimension::Replicated),
            boundary_input_types: Vec::new(),
            input_types,
            unbatched_input_types,
            input_axes,
            discharged: false,
        }
    }
}
