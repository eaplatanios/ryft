//! Contains functionality for building _composite_ operations. Composites in Ryft are functions with custom transform
//! rules. Ryft composites follow the StableHLO [`composite`](https://openxla.org/stablehlo/spec#composite) design, with
//! the operation carrying its decomposition, where here the decomposition is the primal function. A composite pairs
//! that primal with handwritten rules for the transforms whose automatic treatment of it is inadequate (e.g., a fused
//! or foreign kernel, a numerically stabilized derivative, or a batching strategy that structural batching cannot
//! discover). Every transform without a configured rule treats the primal as it treats any other function, except that
//! a composite with only reverse mode rules rejects forward mode. This module has three layers:
//!
//!   - **Functions:** [`composite`] creates a [`Composite`] from its primal, and configuration functions attach its
//!     rules, including support for a Jacobian-Vector Product (JVP) rule that governs both forward and reverse mode
//!     differentiation (via [`Composite::with_jvp`]), forward and backward rules that govern reverse mode
//!     differentiation (via [`Composite::with_vjp`]), or both, and a batching rule (via [`Composite::with_batching`])
//!     or sequential batching (via [`Composite::with_sequential_batching`]). [`Composite::from_custom_call`] creates
//!     a composite whose primal calls a foreign kernel.
//!   - **Operations:** Each call of a composite stages a [`CompositeOperation`], which carries the primal program
//!     as an attached region and retains the rules as callbacks that are traced on the first request of each
//!     specialization. Callers whose rules are already programs (e.g., kernel staging with explicit rule programs)
//!     stage the same operation with the rule programs attached as regions instead. Reverse mode replaces a call
//!     with reverse mode rules, in either representation, with a [`CompositeTransposeOperation`] carrier whose
//!     transposition applies the backward rule, while it transposes the linearized forward mode rule of a call
//!     without reverse mode rules.
//!   - **Rules:** [`CustomRuleDefinition`] and its [`CustomRuleRegistration`] are the flat, family-level rule sets
//!     behind the retained rules, which trace and cache the rule programs of each specialization.
//!
//! Outside of the transforms that their rules govern, composites are _transparent_ meaning that interpretation and
//! backend lowering simply replay the primal program. Refer to the documentation of [`composite`] for the calling
//! convention and for how the rules interact with references, batching, and partial evaluation.
//!
//! # Examples
//!
//! ```rust
//! # use ryft_core::{
//! #     Array, ArrayOperation, Cos, DomainTracer, EagerContext, ProgramError, Sin, composite, differentiate_at,
//! # };
//! # fn main() -> Result<(), ProgramError> {
//! type Tracer = DomainTracer<EagerContext<Array, ArrayOperation<Array>>>;
//!
//! // A custom JVP rule for `sin` that computes the wrong derivative, so that its effect is visible.
//! let sine = composite(|x: Tracer| Ok(x.sin()?)).with_jvp(|x, tangent| {
//!     let tangent = x.cos()? * tangent;
//!     Ok((x.sin()?, tangent.clone() + tangent))
//! });
//! let (value, tangent) = differentiate_at(Array::scalar(0.5f64)?).jvp(Array::scalar(1.0f64)?, |x| sine.call(x))?;
//! assert_eq!(value, Array::scalar(0.5f64.sin())?);
//! assert_eq!(tangent, Array::scalar(2.0 * 0.5f64.cos())?);
//! # Ok(())
//! # }
//! ```

use std::hash::Hash;

use crate::batching::BatchAxis;
use crate::contexts::{Context, StagingContext};
use crate::differentiation::{
    DifferentiableType, DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    DifferentiationPolicy, ResidualZeroProvider,
};
use crate::macros::check_count;
use crate::operations::composites::rules::{
    CustomRuleBatching, CustomRuleBatchingLevel, CustomRuleSpecializationKey, boundary_input_count,
};
use crate::operations::constants::zero::Zero;
use crate::programs::{
    EffectsSummary, Operation, ProgramError, ReferenceBoundary, ReferenceDischargePolicy, ReferenceDischargeValue,
    RegionRef, Type, TypeError, Typed, Value,
};
use crate::tracing::TracingContext;

pub mod builders;
pub mod configurations;
pub mod operations;
pub mod rules;

pub use builders::{
    BatchedComposite, BatchingRuleConsistencyError, Composite, CompositeCallContext, JvpComposite, VjpComposite,
    check_batching_rule_consistency, composite,
};
pub use configurations::{
    AccumulatingVjp, AxisDependentBatching, CompositeBatching, CompositeJvp, CompositeVjp, CustomBatching,
    DefaultBatching, DefaultJvp, DefaultVjp, ExplicitJvp, ExplicitVjp, JvpFromPrimal, SequentialBatching,
    SymbolicZeroJvp, SymbolicZeroVjp,
};
pub use operations::{
    COMPOSITE_OPERATION_NAME, COMPOSITE_TRANSPOSE_OPERATION_NAME, CompositeJvpRule, CompositeOperation,
    CompositeTransposeOperation,
};
pub use rules::{
    CustomRuleDefinition, CustomRuleReference, CustomRuleRegistration, CustomRuleSource, CustomRuleSpecializer,
    CustomRuleTracer, CustomVjpOperation, LiftedCustomRules, UnavailableCustomRules, WeakCustomRuleRegistration,
};

// TODO(eaplatanios): Review from here onwards.

/// Validates that the leading `non_differentiated_count` input positions of a composite call named `name` fit within
/// its `input_count` inputs.
///
/// # Errors
///
/// Returns a [`TypeError`] when `non_differentiated_count` exceeds `input_count`.
fn validate_non_differentiated_count(
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

/// Validates the reference contract that the composite operations share over one primal boundary. A reference-typed
/// input is accepted only in the leading `non_differentiated_count` positions, which correspond to _plumbing_ that
/// every attached rule region receives unchanged (the rule interfaces define no tangent or cotangent slot for a
/// reference, so an active reference input would have a derivative that no user-supplied rule can express). No output
/// may be a reference, because a rule region would then have to produce that output's tangent reference or consume its
/// cotangent reference, and a user-supplied rule can neither allocate nor receive one. The custom derivative operations
/// apply this contract during type inference, so it holds for every constructed call, and their derivative rules apply
/// it again to the replayed inputs as defense in depth.
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
fn validate_composite_reference_boundary<T: Type>(
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

/// Validates that a composite call or carrier, described by `description` for diagnostics, has no reference input when
/// its reference state is discharged. Its rules receive the non-differentiated reference inputs by identity (e.g., to
/// forward a plumbing reference from the reverse-mode forward rule to the backward rule), so they cannot thread the
/// state of those references through explicit inputs and outputs, which discharging them would require.
///
/// # Errors
///
/// Returns [`ProgramError::UnsupportedOperation`] naming the first reference input.
fn validate_composite_discharge_inputs<C: Context, P: ReferenceDischargePolicy<C>>(
    description: &str,
    inputs: &[ReferenceDischargeValue<C, P>],
) -> Result<(), ProgramError> {
    if let Some(index) = inputs.iter().position(|input| matches!(input, ReferenceDischargeValue::Reference(_))) {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "{description} cannot discharge the reference state of its input {index}, because its rules receive \
                 non-differentiated reference inputs by identity and cannot thread their state explicitly; pass \
                 reference-free inputs or discharge external references first",
            ),
        });
    }
    Ok(())
}

/// Validates the inputs before replaying a custom derivative rule. Rule regions bypass ordinary differentiation
/// dispatch, so replay checks the reference contract of [`validate_composite_reference_boundary`] that type inference
/// already enforces and uses [`ReferenceBoundary`] to reject aliased concrete or staged references. Non-differentiated
/// numeric inputs must have zero tangents, while non-differentiated references carry state whose tangent the rule
/// leaves untouched.
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
/// [`validate_composite_reference_boundary`], a reference boundary error, or an unsupported non-zero tangent.
fn validate_composite_replay<C: Context<Type: DifferentiableType>>(
    name: &str,
    non_differentiated_count: usize,
    context: &C,
    inputs: &[DifferentiationDual<C::Value>],
    output_types: &[C::Type],
) -> Result<(), ProgramError> {
    let primal_types = inputs.iter().map(|input| input.primal().r#type().into_owned()).collect::<Vec<_>>();
    validate_composite_reference_boundary(name, non_differentiated_count, primal_types.as_slice(), output_types)?;
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

/// Validates that the forward rule of the retained definition named `name`, which defers that rule to the backward pass
/// (refer to [`CustomRuleSource::defers_forward`]), has a specialization with the pure `effects`, because effects of a
/// forward rule that runs in the backward pass would execute out of order with the rest of the program.
///
/// # Errors
///
/// Returns a [`ProgramError::UnsupportedOperation`] that names `with_deferred_forward` when `effects` is not pure.
fn validate_deferred_forward_rule(name: &str, effects: EffectsSummary) -> Result<(), ProgramError> {
    if effects.is_pure() {
        return Ok(());
    }
    Err(ProgramError::UnsupportedOperation {
        message: format!(
            "{} defers its forward rule to the backward pass (refer to `with_deferred_forward`), which requires a \
             pure forward rule, but its forward rule has effects, which would run out of order in the backward pass",
            if name == COMPOSITE_OPERATION_NAME {
                format!("`{COMPOSITE_OPERATION_NAME}`")
            } else {
                format!("`{COMPOSITE_OPERATION_NAME}` `{name}`")
            },
        ),
    })
}

/// Replays the flat custom JVP rule `jvp_region`, which computes `(p, x, ẋ) ↦ (y, ẏ)`, for an operation whose leading
/// `non_differentiated_count` inputs `p` are not differentiated. This is the shared forward-mode rule of
/// [`CompositeOperation`] and of other custom-rule operations that obtain their JVP program differently (e.g., by
/// tracing a retained callback lazily). A fused [`DifferentiationContext`] replays the rule directly, while a
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
fn replay_custom_jvp_rule<
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
) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>
where
    TracingContext<C::Constant, C::Operation>:
        StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
{
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

    // The rule region is replayed directly rather than differentiated, so the replayed inputs are validated as defense
    // in depth (refer to the documentation of `validate_composite_replay`).
    validate_composite_replay(
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

/// Returns the batching record of the [`CompositeTransposeOperation`] that reverse mode stages for a batched call with
/// retained rules whose forward rule specialization key is `key` (whose levels are those of the call), whose rules have
/// `rule_output_count` outputs, whose carrier receives seed extents of types `seed_extent_types`, and whose output
/// tangents have the unbatched types `extent_types`. At each level, the carrier's inputs have the batch axes of the
/// corresponding call inputs, except for the residuals, which have the axes of that level's forward specialization, and
/// for the seed extents, which are replicated, while its outputs have the call's batch axes, or those of the forward
/// specialization for the outputs that the call does not keep. The carrier's unbatched input types are those of the
/// unbatched forward rule.
fn retained_carrier_batching<V, O, S>(
    rules: &S,
    key: &CustomRuleSpecializationKey<V::Type>,
    rule_output_count: usize,
    seed_extent_types: &[V::Type],
    extent_types: Vec<V::Type>,
) -> Result<CustomRuleBatching<V::Type>, DifferentiationError>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: Value<Type: DifferentiableType + Eq + Hash>,
    O: Operation<Type = V::Type>,
    S: CustomRuleSpecializer<V, O>,
{
    let unbatched = rules.forward_specialization(CustomRuleSpecializationKey { levels: Vec::new(), ..key.clone() })?;
    let non_differentiated_count = key.non_differentiated_count;
    let mut input_types = key.input_types[..non_differentiated_count].to_vec();
    input_types.extend_from_slice(&unbatched.program.output_types()[rule_output_count..]);
    input_types.extend_from_slice(seed_extent_types);
    for r#type in &key.input_types[non_differentiated_count..] {
        input_types.push(r#type.tangent()?);
    }
    let mut levels = Vec::with_capacity(key.levels.len());
    for (index, level) in key.levels.iter().enumerate() {
        let level_key = CustomRuleSpecializationKey { levels: key.levels[..=index].to_vec(), ..key.clone() };
        let level_specialization = rules.forward_specialization(level_key)?;
        let leading_input_count = non_differentiated_count + boundary_input_count(&key.levels[..index]);
        let mut input_axes = level.input_axes[..leading_input_count].to_vec();
        input_axes.extend_from_slice(&level_specialization.output_axes[rule_output_count..]);
        input_axes.extend(std::iter::repeat_n(BatchAxis::replicated(), seed_extent_types.len()));
        input_axes.extend_from_slice(&level.input_axes[leading_input_count..]);
        let output_axes = match key.kept_outputs {
            None => level.output_axes.clone(),
            Some(_) => level_specialization.output_axes[..rule_output_count].to_vec(),
        };
        levels.push(CustomRuleBatchingLevel { input_axes, output_axes, ..level.clone() });
    }
    Ok(CustomRuleBatching { input_types, output_types: extent_types, levels })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType, ShardingDimension,
    };
    use crate::batching::{BatchAxis, BatchedProgram, ProgramBatchingOutputAxesPolicy};
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::{
        DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationError,
        DifferentiationPolicy, ForwardModeDifferentiate, Linearization,
    };
    use crate::operations::arithmetic::MulOperation;
    use crate::operations::references::ReferenceNewOperation;
    use crate::parameters::Placeholder;
    use crate::partial::PartitionedProgram;
    use crate::programs::{
        FlatProgram, Operation, Program, ProgramBuilder, ProgramError, ReferenceType, RegionDriver, RegionRef,
        TypeError, Typed, Value,
    };
    use crate::tracing::Trace;

    use super::*;

    /// Builds one flat program that binds the composite call `operation` with `regions` to inputs of `input_types` and
    /// returns all of its outputs.
    pub(super) fn composite_call_program<V: Value, O: Operation<Type = V::Type>>(
        operation: impl Into<O>,
        regions: Vec<Program<V, O, Vec<V>, Vec<V>>>,
        input_types: Vec<V::Type>,
    ) -> Program<V, O, Vec<V>, Vec<V>> {
        let mut builder = ProgramBuilder::<V, O>::new();
        let regions = regions.into_iter().map(|region| builder.import_program(region)).collect::<Vec<_>>();
        let input_count = input_types.len();
        let inputs = input_types.into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let outputs = builder.add_instruction(operation, regions, inputs, None).unwrap().to_vec();
        let output_count = outputs.len();
        builder
            .build::<Vec<V>, Vec<V>>(outputs, vec![Placeholder; input_count], vec![Placeholder; output_count])
            .unwrap()
    }

    /// Supplies the regions of composite calls while making recursive differentiation an assertion failure: the custom
    /// derivative rules replay their rule regions directly and must never differentiate them recursively.
    pub(super) struct ReferenceRuleDifferentiationDriver {
        pub(super) programs: Vec<FlatProgram<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>>,
    }

    impl RegionDriver<ArrayIrValue<Array>, ArrayIrOperation<Array>> for ReferenceRuleDifferentiationDriver {
        fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, ArrayIrValue<Array>, ArrayIrOperation<Array>>>
        where
            ArrayIrValue<Array>: 'r,
            ArrayIrOperation<Array>: 'r,
        {
            self.programs.iter().map(Program::entry_region_ref)
        }
    }

    impl DifferentiationDriver<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>
        for ReferenceRuleDifferentiationDriver
    {
        fn jvp_program(
            &self,
            _region: RegionRef<'_, ArrayIrValue<Array>, ArrayIrOperation<Array>>,
            _input_indices: &[usize],
        ) -> Result<Arc<FlatProgram<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>>, DifferentiationError>
        {
            unreachable!("custom derivative rules replay their rule regions instead of differentiating them")
        }

        fn linearize_program(
            &self,
            _region: RegionRef<'_, ArrayIrValue<Array>, ArrayIrOperation<Array>>,
            _input_indices: &[usize],
            _instantiated_outputs: Option<&[bool]>,
        ) -> Result<Linearization<ArrayIrValue<Array>, ArrayIrOperation<Array>>, DifferentiationError> {
            unreachable!("custom derivative rules replay their rule regions instead of linearizing them")
        }

        fn partition_jvp_program(
            &self,
            region: RegionRef<'_, ArrayIrValue<Array>, ArrayIrOperation<Array>>,
            input_known: &[bool],
            required_known_outputs: &[usize],
        ) -> Result<PartitionedProgram<ArrayIrValue<Array>, ArrayIrOperation<Array>>, DifferentiationError> {
            Ok(region.partition_with_configuration(input_known, true, true, Some(required_known_outputs), None)?.0)
        }

        fn bind_jvp_operation<P: DifferentiationPolicy<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>>(
            &self,
            _context: &DifferentiationContext<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>, P>,
            _operation: &ArrayIrOperation<Array>,
            _programs: Vec<FlatProgram<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>>,
            _inputs: &[DifferentiationDual<ArrayIrValue<Array>>],
        ) -> Result<Vec<DifferentiationDual<ArrayIrValue<Array>>>, DifferentiationError> {
            unreachable!("custom derivative rules replay their rule regions instead of differentiating them")
        }
    }

    /// Eager context whose values are arrays.
    pub(super) type ArrayContext = EagerContext<Array, ArrayOperation<Array>>;

    /// Eager array IR context whose values may be arrays or references.
    pub(super) type EagerArrayIrContext = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    // Fixtures of the tests of calls and carriers with retained rules.

    /// Eager context whose operation family registers the retained-rule operations.
    pub(super) type TestContext = EagerContext<Array, ArrayOperation<Array>>;

    /// Retained-rule definition over [`TestContext`].
    pub(super) type TestDefinition = CustomRuleDefinition<Array, ArrayOperation<Array>>;

    /// Registration of a [`TestDefinition`].
    pub(super) type TestRegistration = CustomRuleRegistration<Array, ArrayOperation<Array>>;

    /// Retained-rule definition over the production [`ArrayOperation`] member family.
    pub(super) type MemberDefinition = CustomRuleDefinition<Array, ArrayOperation<Array>>;

    /// Invocation counters of the retained rules of a [`cube_definition`].
    #[derive(Default)]
    pub(super) struct RuleCounters {
        /// Number of JVP rule invocations.
        pub(super) jvp: AtomicUsize,

        /// Number of VJP forward rule invocations.
        pub(super) forward: AtomicUsize,

        /// Number of VJP backward rule invocations.
        pub(super) backward: AtomicUsize,
    }

    impl RuleCounters {
        /// Returns the JVP, forward, and backward invocation counts.
        pub(super) fn counts(&self) -> (usize, usize, usize) {
            (self.jvp.load(Ordering::SeqCst), self.forward.load(Ordering::SeqCst), self.backward.load(Ordering::SeqCst))
        }
    }

    /// Builds the primal `f(x) = x³`.
    pub(super) fn cube_program(r#type: &ArrayType) -> FlatProgram<TestContext> {
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(r#type.clone());
        let square = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
        let cube = builder.add_instruction(MulOperation::new(), Vec::new(), vec![square, input], None).unwrap()[0];
        builder.build(vec![cube], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Returns a definition of `f(x) = x³` whose deliberately wrong rules distinguish the selected rule: the JVP rule
    /// computes `ẏ = x² ẋ` and the VJP rule computes `x̄ = 2 x² ȳ`, while the true derivative is `3 x²`. Each rule
    /// counts its invocations in `counters`.
    pub(super) fn cube_definition(counters: &Arc<RuleCounters>, jvp: bool, vjp: bool) -> TestDefinition {
        let mut definition = TestDefinition::new("cube").with_batching_support();
        if jvp {
            let counters = counters.clone();
            definition = definition.with_jvp(move |primals, tangents| {
                counters.jvp.fetch_add(1, Ordering::SeqCst);
                let square = primals[0].clone() * primals[0].clone();
                Ok((vec![square.clone() * primals[0].clone()], vec![square * tangents[0].clone()]))
            });
        }
        if vjp {
            let forward_counters = counters.clone();
            let backward_counters = counters.clone();
            definition = definition.with_vjp(
                move |primals| {
                    forward_counters.forward.fetch_add(1, Ordering::SeqCst);
                    let square = primals[0].clone() * primals[0].clone();
                    Ok((vec![square.clone() * primals[0].clone()], vec![square]))
                },
                move |leading_inputs, seeds| {
                    backward_counters.backward.fetch_add(1, Ordering::SeqCst);
                    let contribution = leading_inputs[0].clone() * seeds[0].clone();
                    Ok(vec![contribution.clone() + contribution])
                },
            );
        }
        definition
    }

    /// Returns the member-family counterpart of [`cube_definition`], whose rules are written against member tracers.
    pub(super) fn member_cube_definition(counters: &Arc<RuleCounters>) -> MemberDefinition {
        let (jvp_counters, forward_counters, backward_counters) =
            (counters.clone(), counters.clone(), counters.clone());
        MemberDefinition::new("cube")
            .with_jvp(move |primals, tangents| {
                jvp_counters.jvp.fetch_add(1, Ordering::SeqCst);
                let square = primals[0].clone() * primals[0].clone();
                Ok((vec![square.clone() * primals[0].clone()], vec![square * tangents[0].clone()]))
            })
            .with_vjp(
                move |primals| {
                    forward_counters.forward.fetch_add(1, Ordering::SeqCst);
                    let square = primals[0].clone() * primals[0].clone();
                    Ok((vec![square.clone() * primals[0].clone()], vec![square]))
                },
                move |leading_inputs, seeds| {
                    backward_counters.backward.fetch_add(1, Ordering::SeqCst);
                    let contribution = leading_inputs[0].clone() * seeds[0].clone();
                    Ok(vec![contribution.clone() + contribution])
                },
            )
            .with_batching_support()
    }

    /// Stages one call of `definition` over an input of type `r#type`.
    pub(super) fn custom_rule_program(definition: &TestRegistration, r#type: ArrayType) -> FlatProgram<TestContext> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let primal = builder.import_program(cube_program(&r#type));
        let input = builder.add_input(r#type);
        let output = builder
            .add_instruction(CompositeOperation::new(definition.reference()), vec![primal], vec![input], None)
            .unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Returns the outputs and tangents of the forward-mode differentiation of one call of `definition` at `x` with
    /// tangent `tangent`.
    pub(super) fn call_jvp(
        definition: &TestRegistration,
        x: Array,
        tangent: Array,
    ) -> Result<(Vec<Array>, Vec<Array>), DifferentiationError> {
        let r#type = x.r#type().into_owned();
        TestContext::new().jvp(
            |x, ()| {
                let operation = CompositeOperation::new(definition.reference());
                x.context().bind(operation, vec![cube_program(&r#type)], &[x.clone()])
            },
            x,
            tangent,
            (),
        )
    }

    /// Array IR program family into which member programs are converted.
    pub(super) type ConvertedProgram = FlatProgram<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>;

    /// Retained-rule definition over the production [`ArrayIrOperation`] family, which supports reference state.
    pub(super) type IrDefinition = CustomRuleDefinition<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Tracer over which the rules of an [`IrDefinition`] are written.
    pub(super) type IrTracer = CustomRuleTracer<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Stages one call of the array IR `definition` over `primal` with the provided input types.
    pub(super) fn ir_custom_rule_program(
        definition: &CustomRuleRegistration<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
        non_differentiated_count: usize,
        primal: ConvertedProgram,
    ) -> ConvertedProgram {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input_types = primal.input_types();
        let primal = builder.import_program(primal);
        let inputs = input_types.into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let input_count = inputs.len();
        let operation = ArrayIrOperation::Composite(
            CompositeOperation::new(definition.reference())
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

    /// Batches `program` over a batch axis of size `extent` with the provided input batch axes.
    pub(super) fn batched_program(
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

    #[test]
    fn test_validate_non_differentiated_count() {
        assert_eq!(validate_non_differentiated_count("composite", 0, 0), Ok(()));
        assert_eq!(validate_non_differentiated_count("composite", 2, 2), Ok(()));
        assert_eq!(
            validate_non_differentiated_count("composite", 3, 2),
            Err(TypeError::invalid("`composite` non-differentiated input count 3 exceeds input count 2".to_string())),
        );
    }

    #[test]
    fn test_validate_composite_reference_boundary() {
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));

        // References are accepted in the leading non-differentiated input segment.
        assert_eq!(
            validate_composite_reference_boundary(
                "composite",
                1,
                &[reference_type.clone(), scalar_type.clone()],
                std::slice::from_ref(&scalar_type),
            ),
            Ok(()),
        );

        // References are rejected among the differentiated inputs and among the outputs.
        assert_eq!(
            validate_composite_reference_boundary(
                "composite",
                1,
                &[scalar_type.clone(), reference_type.clone()],
                std::slice::from_ref(&scalar_type),
            ),
            Err(TypeError::invalid(
                "`composite` accepts reference inputs only in its leading non-differentiated segment; move input \
                 1 of type `ref<f32[]>` before the differentiated inputs"
                    .to_string(),
            )),
        );
        assert_eq!(
            validate_composite_reference_boundary(
                "composite",
                1,
                &[reference_type.clone(), scalar_type.clone()],
                &[scalar_type, reference_type],
            ),
            Err(TypeError::invalid(
                "`composite` cannot return a reference, but output 1 has type `ref<f32[]>`".to_string(),
            )),
        );
    }

    #[test]
    fn test_validate_composite_replay() {
        // Structural zeros and zero-space values require no tangent slot for a non-differentiated input.
        let input =
            DifferentiationDual::new_with_zero_tangent(ArrayIrValue::Array(Array::scalar(3.0f32).unwrap())).unwrap();
        assert_eq!(validate_composite_replay("composite", 1, &EagerArrayIrContext::new(), &[input], &[]), Ok(()),);
        let input = DifferentiationDual::new(
            ArrayIrValue::Array(Array::from_logical_bytes(ArrayType::scalar(DataType::Token), &[]).unwrap()),
            ArrayIrValue::Array(Array::from_logical_bytes(ArrayType::scalar(DataType::Zero), &[]).unwrap()),
        )
        .unwrap();
        assert_eq!(validate_composite_replay("composite", 1, &EagerArrayIrContext::new(), &[input], &[]), Ok(()),);

        // A non-differentiated numeric input with a non-zero tangent has no tangent slot in the rule.
        let input = DifferentiationDual::new(
            ArrayIrValue::Array(Array::scalar(3.0f32).unwrap()),
            ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
        )
        .unwrap();
        assert!(matches!(
            validate_composite_replay("composite", 1, &EagerArrayIrContext::new(), &[input], &[]),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "`composite` cannot propagate the non-zero tangent of type `f32[]` supplied for \
                    one of its 1 leading non-differentiated inputs, because its rule has no tangent slot for them",
        ));
    }

    #[test]
    fn test_validate_composite_replay_staged_references() {
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
                assert_eq!(validate_composite_replay("composite", 2, context, &duals, &[]), Ok(()));

                // Replaying the same valid boundary is allowed, including a fresh root allocated inside the trace.
                let local = context.bind(ReferenceNewOperation::new(), vec![], &inputs[2..])?.remove(0);
                duals[1] = DifferentiationDual::new_with_zero_tangent(local)?;
                assert_eq!(validate_composite_replay("composite", 2, context, &duals, &[]), Ok(()));
                assert_eq!(validate_composite_replay("composite", 2, context, &duals, &[]), Ok(()));

                // An ordinary staged input must not hide two reference inputs naming the same allocation.
                duals[1] = duals[0].clone();
                assert!(matches!(
                    validate_composite_replay("composite", 2, context, &duals, &[]),
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
