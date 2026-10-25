//! The type-state markers of the [`Composite`] configurations (e.g., [`ExplicitJvp`] and [`ExplicitVjp`]), which store
//! the structured rule closures type-erased, and the adapters that register those closures as the flat, family-level
//! rules of [`CustomRuleDefinition`]s.

use std::any::Any;
use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use thiserror::Error;

use crate::axes::{NamedAxes, NamedAxis};
use crate::batching::{
    Batch, BatchAxis, BatchAxisSpecification, BatchableType, BatchingError, BatchingLevel, BatchingLevelExtent,
    BatchingPolicy, BatchingTracer, DiagonalBatchingPolicy, RecursiveBatchingPolicy, batch,
};
use crate::contexts::{Context, EagerContext, StagingContext};
use crate::differentiation::{
    CotangentAccumulator, CotangentBatchingPolicy, DifferentiableOperation, DifferentiableType, ResidualZeroProvider,
    TranspositionContext,
};
use crate::macros::check_count;
use crate::operations::arithmetic::AddOperation;
use crate::operations::control_flow::scan::{ScanOperation, ScanType};
use crate::operations::custom_call::{CustomCall, CustomCallOperation};
use crate::parameters::{Parameter, Parameterized, ParameterizedFamily};
use crate::partial::{PartialEvaluationContext, PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, Operation, OperationProvider, Program, ProgramError, ReferenceMemberType, Type, TypeError, Typed, Value,
};
use crate::tracing::{DomainTracer, DomainTracingContext, TracingContext};
use crate::operations::composites::describe_composite;
use crate::operations::composites::builder::{Composite, composite};
use crate::operations::composites::call::CompositeOperation;
use crate::operations::composites::rules::{CustomRuleDefinition, CustomRuleTracer, CustomVjpOperation, derive_bijective_identity_renaming};

/// Forward-mode configuration of a [`Composite`] without a configured forward-mode rule. A function without
/// reverse-mode rules then derives its forward-mode rule from its primal (as [`JvpFromPrimal`] does), while a function
/// with reverse-mode rules rejects forward mode (refer to [`composite`] for the rule selection).
#[derive(Copy, Clone, Debug, Default)]
pub struct DefaultJvp;

/// Forward-mode configuration of a [`Composite`] with an explicit, user-supplied Jacobian-Vector Product (JVP) rule
/// closure implementing `(x, ẋ) ↦ (y, ẏ)` (refer to [`Composite::with_jvp`]).
pub struct ExplicitJvp<Input, Output> {
    /// Closure computing `(outputs, output_tangents)` from the primal input value and the input tangent value.
    jvp: Arc<dyn Fn(Input, Input) -> Result<(Output, Output), ProgramError> + Send + Sync>,
}

impl<Input, Output> Debug for ExplicitJvp<Input, Output> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("ExplicitJvp").finish_non_exhaustive()
    }
}

/// Forward-mode configuration of a [`Composite`] whose forward-mode rule is derived from its primal (refer to
/// [`Composite::with_jvp_from_primal`]).
#[derive(Copy, Clone, Debug, Default)]
pub struct JvpFromPrimal;

/// Forward-mode configuration of a [`Composite`] with a user-supplied Jacobian-Vector Product (JVP) rule closure
/// implementing `(x, ẋ) ↦ (y, ẏ)` that receives `ẋ` and returns `ẏ` with [`MaybeZero`] leaves, of which the
/// structurally zero ones are [`MaybeZero::Zero`]s (refer to [`Composite::with_symbolic_zero_jvp`]). `Tracer` is the
/// tracer type of the closure's leaves, which the leaves of its tangents wrap.
pub struct SymbolicZeroJvp<Tracer: Typed + Parameter, Input, Output>
where
    Input: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
    Output: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
{
    /// Closure computing `(outputs, output_tangents)` from the primal input value and the input tangent value.
    jvp: Arc<
        dyn Fn(Input, Input::To<MaybeZero<Tracer>>) -> Result<(Output, Output::To<MaybeZero<Tracer>>), ProgramError>
            + Send
            + Sync,
    >,
}

impl<Tracer: Typed + Parameter, Input, Output> Debug for SymbolicZeroJvp<Tracer, Input, Output>
where
    Input: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
    Output: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("SymbolicZeroJvp").finish_non_exhaustive()
    }
}

/// Reverse-mode configuration of a [`Composite`] without reverse-mode rules. Reverse mode then transposes the
/// forward-mode rule, which is derived from the primal when none is configured.
#[derive(Copy, Clone, Debug, Default)]
pub struct DefaultVjp;

/// Reverse-mode configuration of a [`Composite`] with explicit, user-supplied forward and backward (i.e.,
/// Vector-Jacobian Product or VJP) rule closures implementing `x ↦ (y, r)` and `(r, ȳ) ↦ x̄`, where `r` is a value of
/// type `Residual` (refer to [`Composite::with_vjp`]).
pub struct ExplicitVjp<Input, Output, Residual> {
    /// Closure computing `(outputs, residuals)` from the primal input value.
    forward: Arc<dyn Fn(Input) -> Result<(Output, Residual), ProgramError> + Send + Sync>,

    /// Closure computing the input cotangent value from `(residuals, output_cotangents)`.
    backward: Arc<dyn Fn(Residual, Output) -> Result<Input, ProgramError> + Send + Sync>,

    /// Whether reverse mode defers `forward` to the backward pass (refer to [`Composite::with_deferred_forward`]).
    pub(super) deferred_forward: bool,
}

impl<Input, Output, Residual> Debug for ExplicitVjp<Input, Output, Residual> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = formatter.debug_struct("ExplicitVjp");
        if self.deferred_forward {
            debug.field("deferred_forward", &true);
        }
        debug.finish_non_exhaustive()
    }
}

/// Reverse-mode configuration of a [`Composite`] with user-supplied forward and backward (i.e., Vector-Jacobian Product
/// or VJP) rule closures implementing `x ↦ (y, r)` and `(r, ȳ) ↦ x̄`, whose backward rule receives `ȳ` and returns
/// `x̄` with [`MaybeZero`] leaves, of which the structurally zero ones are [`MaybeZero::Zero`]s (refer to
/// [`Composite::with_symbolic_zero_vjp`]). `Tracer` is the tracer type of the closures' leaves, which the leaves of the
/// seeds and cotangents wrap.
pub struct SymbolicZeroVjp<Tracer: Typed + Parameter, Input, Output, Residual>
where
    Input: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
    Output: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
{
    /// Closure computing `(outputs, residuals)` from the primal input value.
    forward: Arc<dyn Fn(Input) -> Result<(Output, Residual), ProgramError> + Send + Sync>,

    /// Closure computing the input cotangent value from `(residuals, output_cotangent_seeds)`.
    backward: Arc<
        dyn Fn(Residual, Output::To<MaybeZero<Tracer>>) -> Result<Input::To<MaybeZero<Tracer>>, ProgramError>
            + Send
            + Sync,
    >,

    /// Whether reverse mode defers `forward` to the backward pass (refer to [`Composite::with_deferred_forward`]).
    pub(super) deferred_forward: bool,
}

impl<Tracer: Typed + Parameter, Input, Output, Residual> Debug for SymbolicZeroVjp<Tracer, Input, Output, Residual>
where
    Input: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
    Output: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = formatter.debug_struct("SymbolicZeroVjp");
        if self.deferred_forward {
            debug.field("deferred_forward", &true);
        }
        debug.finish_non_exhaustive()
    }
}

/// Reverse-mode configuration of a [`Composite`] with a user-supplied forward rule closure implementing `x ↦ (y, r)`
/// and an accumulating backward rule closure, which submits the input cotangents to their accumulators (refer to
/// [`Composite::with_accumulating_vjp`]). `Tracer` is the tracer type of the closures' leaves, which the leaves of the
/// seeds wrap, and `Transposition` is the type of the transposition context that the backward rule closure receives.
pub struct AccumulatingVjp<Tracer: Typed + Parameter, Transposition, Input, Output, Residual>
where
    Input: Parameterized<Tracer, Family: ParameterizedFamily<CotangentAccumulator>>,
    Output: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
{
    /// Closure computing `(outputs, residuals)` from the primal input value.
    forward: Arc<dyn Fn(Input) -> Result<(Output, Residual), ProgramError> + Send + Sync>,

    /// Closure submitting the input cotangents, computed from `(residuals, output_cotangent_seeds)`, to their
    /// accumulators.
    backward: Arc<
        dyn Fn(
                &mut Transposition,
                Residual,
                Output::To<MaybeZero<Tracer>>,
                Input::To<CotangentAccumulator>,
            ) -> Result<(), ProgramError>
            + Send
            + Sync,
    >,

    /// Whether reverse mode defers `forward` to the backward pass (refer to [`Composite::with_deferred_forward`]).
    pub(super) deferred_forward: bool,
}

impl<Tracer: Typed + Parameter, Transposition, Input, Output, Residual> Debug
    for AccumulatingVjp<Tracer, Transposition, Input, Output, Residual>
where
    Input: Parameterized<Tracer, Family: ParameterizedFamily<CotangentAccumulator>>,
    Output: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = formatter.debug_struct("AccumulatingVjp");
        if self.deferred_forward {
            debug.field("deferred_forward", &true);
        }
        debug.finish_non_exhaustive()
    }
}

/// Forward-mode configuration of a [`Composite`] (i.e., [`DefaultJvp`], [`ExplicitJvp`], [`SymbolicZeroJvp`], or
/// [`JvpFromPrimal`]), which installs its forward-mode rule in the retained definition that a call registers for the
/// operation family `(V, O)`. Its user-supplied rule is adapted to the flat interface of retained rules by tracing it,
/// with the complete structured interface of the call, at the types of each specialization and replaying the traced
/// program.
pub trait CompositeJvp<V: Value, O: Operation<Type = V::Type>, Input, Output>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    Input: Parameterized<CustomRuleTracer<V, O>>,
    Output: Parameterized<CustomRuleTracer<V, O>>,
{
    /// Returns `definition` with this forward-mode rule.
    ///
    /// # Parameters
    ///
    ///   - `definition`: Definition that the call registers.
    ///   - `name`: Name of the user-facing function, used in diagnostics.
    ///   - `input_structure`: Structure of the call's inputs.
    ///   - `output_structure`: Structure of the call's outputs.
    ///   - `non_differentiated_count`: Number of leading non-differentiated input leaves.
    fn configure(
        &self,
        definition: CustomRuleDefinition<V, O>,
        name: &Cow<'static, str>,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
        non_differentiated_count: usize,
    ) -> CustomRuleDefinition<V, O>;
}

impl<V: Value, O: Operation<Type = V::Type>, Input, Output> CompositeJvp<V, O, Input, Output> for DefaultJvp
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    Input: Parameterized<CustomRuleTracer<V, O>>,
    Output: Parameterized<CustomRuleTracer<V, O>>,
{
    #[inline]
    fn configure(
        &self,
        definition: CustomRuleDefinition<V, O>,
        _name: &Cow<'static, str>,
        _input_structure: &Input::ParameterStructure,
        _output_structure: &Output::ParameterStructure,
        _non_differentiated_count: usize,
    ) -> CustomRuleDefinition<V, O> {
        definition
    }
}

impl<V: Value<Type: Eq + Hash>, O: Operation<Type = V::Type>, Input, Output> CompositeJvp<V, O, Input, Output>
    for JvpFromPrimal
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    Input: Parameterized<CustomRuleTracer<V, O>>,
    Output: Parameterized<CustomRuleTracer<V, O>>,
{
    #[inline]
    fn configure(
        &self,
        definition: CustomRuleDefinition<V, O>,
        _name: &Cow<'static, str>,
        _input_structure: &Input::ParameterStructure,
        _output_structure: &Output::ParameterStructure,
        _non_differentiated_count: usize,
    ) -> CustomRuleDefinition<V, O> {
        definition.with_jvp_from_primal()
    }
}

impl<V, O, Input, Output> CompositeJvp<V, O, Input, Output> for ExplicitJvp<Input, Output>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: 'static + Value<Type: DifferentiableType + Eq + Hash>,
    O: 'static + Operation<Type = V::Type>,
    Input: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Send + Sync>,
    Input::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Input::To<V::Type>: Clone + Parameterized<V::Type, Family = Input::Family, To<CustomRuleTracer<V, O>> = Input>,
    Output: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send + Sync>,
    Output::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
{
    fn configure(
        &self,
        definition: CustomRuleDefinition<V, O>,
        name: &Cow<'static, str>,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
        non_differentiated_count: usize,
    ) -> CustomRuleDefinition<V, O> {
        let (jvp, name, named_axes) = (self.jvp.clone(), name.clone(), definition.named_axes().to_vec());
        let (input_structure, output_structure) = (input_structure.clone(), output_structure.clone());
        definition.with_jvp(move |primals, tangents| {
            let input_types = Input::To::<V::Type>::from_parameters(
                input_structure.clone(),
                primals.iter().map(|primal| primal.r#type().into_owned()),
            )?;
            let program = trace_custom_jvp_rule::<V, O, Input, Output, _>(
                &name,
                jvp.as_ref(),
                input_types,
                &output_structure,
                non_differentiated_count,
                named_axes.clone(),
            )?;
            let mut values = primals.to_vec();
            values.extend_from_slice(tangents);
            let mut outputs = program.interpret_in_context(primals[0].context(), values)?;
            let output_tangents = outputs.split_off(output_structure.parameter_count());
            Ok((outputs, output_tangents))
        })
    }
}

impl<V, O, Input, Output> CompositeJvp<V, O, Input, Output> for SymbolicZeroJvp<CustomRuleTracer<V, O>, Input, Output>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: 'static + Value<Type: DifferentiableType + Eq + Hash>,
    O: 'static + Operation<Type = V::Type>,
    Input: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Send + Sync>,
    Input::Family:
        ParameterizedFamily<V::Type> + ParameterizedFamily<V> + ParameterizedFamily<MaybeZero<CustomRuleTracer<V, O>>>,
    Input::To<V::Type>: Clone + Parameterized<V::Type, Family = Input::Family, To<CustomRuleTracer<V, O>> = Input>,
    Output: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send + Sync>,
    Output::Family:
        ParameterizedFamily<V::Type> + ParameterizedFamily<V> + ParameterizedFamily<MaybeZero<CustomRuleTracer<V, O>>>,
{
    fn configure(
        &self,
        definition: CustomRuleDefinition<V, O>,
        name: &Cow<'static, str>,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
        non_differentiated_count: usize,
    ) -> CustomRuleDefinition<V, O> {
        let (jvp, name, named_axes) = (self.jvp.clone(), name.clone(), definition.named_axes().to_vec());
        let (input_structure, output_structure) = (input_structure.clone(), output_structure.clone());
        definition.with_symbolic_zero_jvp(move |primals, tangents| {
            let input_types = Input::To::<V::Type>::from_parameters(
                input_structure.clone(),
                primals.iter().map(|primal| primal.r#type().into_owned()),
            )?;
            let tangent_activity = tangents.iter().map(|tangent| !tangent.is_zero()).collect::<Vec<_>>();
            let (program, output_tangent_zeros) = trace_symbolic_zero_custom_jvp_rule::<V, O, Input, Output, _>(
                &name,
                jvp.as_ref(),
                input_types,
                &tangent_activity,
                &output_structure,
                non_differentiated_count,
                named_axes.clone(),
            )?;
            let mut values = primals.to_vec();
            values.extend(tangents.iter().filter_map(MaybeZero::as_value).cloned());
            let mut outputs = program.interpret_in_context(primals[0].context(), values)?;
            let mut live_output_tangents = outputs.split_off(output_structure.parameter_count()).into_iter();

            // The traced program returns only the live output tangents, so the structural zeros are reinserted at
            // their positions, typed by the tangent types of the replayed outputs.
            let output_tangents = outputs
                .iter()
                .zip(output_tangent_zeros)
                .map(|(output, is_zero)| match is_zero {
                    true => Ok(MaybeZero::Zero(output.r#type().tangent()?)),
                    false => Ok(MaybeZero::Value(live_output_tangents.next().unwrap())),
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            Ok((outputs, output_tangents))
        })
    }
}

/// Reverse-mode configuration of a [`Composite`] (i.e., [`DefaultVjp`], [`ExplicitVjp`], [`SymbolicZeroVjp`], or
/// [`AccumulatingVjp`]), which installs its reverse-mode rules in the retained definition that a call registers for the
/// operation family `(V, O)`. Its user-supplied rules are adapted to the flat interface of retained rules as for
/// [`CompositeJvp`]. The backward rule receives the residuals with the structure that the forward rule returned for the
/// same specialization. Different specializations of one call structure may return differently structured residuals, as
/// long as residuals whose flat types are equal up to a renaming of their type identities (e.g., their dimension
/// variables) always have the same structure; the forward rule is rejected otherwise.
pub trait CompositeVjp<V: Value, O: Operation<Type = V::Type>, Input, Output>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    Input: Parameterized<CustomRuleTracer<V, O>>,
    Output: Parameterized<CustomRuleTracer<V, O>>,
{
    /// Returns `definition` with these reverse-mode rules (refer to [`CompositeJvp::configure`] for the parameters).
    fn configure(
        &self,
        definition: CustomRuleDefinition<V, O>,
        name: &Cow<'static, str>,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
        non_differentiated_count: usize,
    ) -> CustomRuleDefinition<V, O>;
}

impl<V: Value, O: Operation<Type = V::Type>, Input, Output> CompositeVjp<V, O, Input, Output> for DefaultVjp
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    Input: Parameterized<CustomRuleTracer<V, O>>,
    Output: Parameterized<CustomRuleTracer<V, O>>,
{
    #[inline]
    fn configure(
        &self,
        definition: CustomRuleDefinition<V, O>,
        _name: &Cow<'static, str>,
        _input_structure: &Input::ParameterStructure,
        _output_structure: &Output::ParameterStructure,
        _non_differentiated_count: usize,
    ) -> CustomRuleDefinition<V, O> {
        definition
    }
}

impl<V, O, Input, Output, Residual> CompositeVjp<V, O, Input, Output> for ExplicitVjp<Input, Output, Residual>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: 'static + Value<Type: Eq + Hash + Send + DifferentiableType + ReferenceMemberType>,
    O: 'static + CustomVjpOperation<V>,
    Input: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send + Sync>,
    Input::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Input::To<V::Type>: Clone + Parameterized<V::Type, Family = Input::Family, To<CustomRuleTracer<V, O>> = Input>,
    Output: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send + Sync>,
    Output::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Output::To<V::Type>: Parameterized<V::Type, Family = Output::Family, To<CustomRuleTracer<V, O>> = Output>,
    Residual: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send>,
    Residual::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Residual::To<V::Type>: Parameterized<V::Type, Family = Residual::Family, To<CustomRuleTracer<V, O>> = Residual>,
{
    fn configure(
        &self,
        definition: CustomRuleDefinition<V, O>,
        name: &Cow<'static, str>,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
        non_differentiated_count: usize,
    ) -> CustomRuleDefinition<V, O> {
        let residual_structures = CompositeResidualStructures::<V::Type, Residual::ParameterStructure>::default();
        let forward = retained_custom_vjp_forward_rule::<V, O, Input, Output, Residual, _>(
            self.forward.clone(),
            name,
            input_structure,
            output_structure,
            &residual_structures,
            definition.named_axes(),
        );
        let backward = {
            let (backward, name, named_axes) = (self.backward.clone(), name.clone(), definition.named_axes().to_vec());
            let (input_structure, output_structure) = (input_structure.clone(), output_structure.clone());
            move |leading_inputs: &[CustomRuleTracer<V, O>], seeds: &[CustomRuleTracer<V, O>]| {
                let types = |values: &[CustomRuleTracer<V, O>]| {
                    values.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>()
                };
                let residual_types = types(&leading_inputs[non_differentiated_count..]);
                let structure = residual_structures.get(&name, &residual_types)?;
                let Some(context) = leading_inputs.iter().chain(seeds).next().map(|value| value.context().clone())
                else {
                    return Err(ProgramError::MalformedProgram(format!(
                        "`{name}` backward rule has neither leading inputs nor seeds",
                    )));
                };
                let program = trace_custom_vjp_backward_rule::<V, O, Input, Output, Residual, _>(
                    &name,
                    backward.as_ref(),
                    &input_structure,
                    types(&leading_inputs[..non_differentiated_count]),
                    structure,
                    residual_types,
                    output_structure.clone(),
                    types(seeds),
                    named_axes.clone(),
                )?;
                let mut values = leading_inputs.to_vec();
                values.extend_from_slice(seeds);
                program.interpret_in_context(&context, values)
            }
        };
        let definition = definition.with_vjp(forward, backward);
        if self.deferred_forward { definition.with_deferred_forward_rule() } else { definition }
    }
}

impl<V, O, Input, Output, Residual> CompositeVjp<V, O, Input, Output>
    for SymbolicZeroVjp<CustomRuleTracer<V, O>, Input, Output, Residual>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: 'static + Value<Type: Eq + Hash + Send + DifferentiableType + ReferenceMemberType>,
    O: 'static + CustomVjpOperation<V>,
    Input: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send + Sync>,
    Input::Family:
        ParameterizedFamily<V::Type> + ParameterizedFamily<V> + ParameterizedFamily<MaybeZero<CustomRuleTracer<V, O>>>,
    Input::To<V::Type>: Clone + Parameterized<V::Type, Family = Input::Family, To<CustomRuleTracer<V, O>> = Input>,
    Output: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send + Sync>,
    Output::Family:
        ParameterizedFamily<V::Type> + ParameterizedFamily<V> + ParameterizedFamily<MaybeZero<CustomRuleTracer<V, O>>>,
    Output::To<V::Type>: Parameterized<V::Type, Family = Output::Family, To<CustomRuleTracer<V, O>> = Output>,
    Residual: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send>,
    Residual::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Residual::To<V::Type>: Parameterized<V::Type, Family = Residual::Family, To<CustomRuleTracer<V, O>> = Residual>,
{
    fn configure(
        &self,
        definition: CustomRuleDefinition<V, O>,
        name: &Cow<'static, str>,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
        non_differentiated_count: usize,
    ) -> CustomRuleDefinition<V, O> {
        let residual_structures = CompositeResidualStructures::<V::Type, Residual::ParameterStructure>::default();
        let forward = retained_custom_vjp_forward_rule::<V, O, Input, Output, Residual, _>(
            self.forward.clone(),
            name,
            input_structure,
            output_structure,
            &residual_structures,
            definition.named_axes(),
        );
        let backward = {
            let (backward, name, named_axes) = (self.backward.clone(), name.clone(), definition.named_axes().to_vec());
            let (input_structure, output_structure) = (input_structure.clone(), output_structure.clone());
            move |leading_inputs: &[CustomRuleTracer<V, O>], seeds: &[MaybeZero<CustomRuleTracer<V, O>>]| {
                let types = |values: &[CustomRuleTracer<V, O>]| {
                    values.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>()
                };
                let residual_types = types(&leading_inputs[non_differentiated_count..]);
                let structure = residual_structures.get(&name, &residual_types)?;
                let live_seeds = seeds.iter().filter_map(MaybeZero::as_value).cloned().collect::<Vec<_>>();
                let Some(context) =
                    leading_inputs.iter().chain(&live_seeds).next().map(|value| value.context().clone())
                else {
                    return Err(ProgramError::MalformedProgram(format!(
                        "`{name}` backward rule has neither leading inputs nor non-zero seeds",
                    )));
                };
                let (program, cotangent_zeros) =
                    trace_symbolic_zero_custom_vjp_backward_rule::<V, O, Input, Output, Residual, _>(
                        &name,
                        backward.as_ref(),
                        &input_structure,
                        types(&leading_inputs[..non_differentiated_count]),
                        structure,
                        residual_types,
                        output_structure.clone(),
                        seeds,
                        named_axes.clone(),
                    )?;
                let mut values = leading_inputs.to_vec();
                values.extend(live_seeds);
                let mut live_cotangents = program.interpret_in_context(&context, values)?.into_iter();

                // The traced program returns only the live cotangents, so the structural zeros are reinserted at their
                // positions. Their types were traced at the actual types of the leading inputs and seeds.
                Ok(cotangent_zeros
                    .into_iter()
                    .map(|zero| match zero {
                        Some(r#type) => MaybeZero::Zero(r#type),
                        None => MaybeZero::Value(live_cotangents.next().unwrap()),
                    })
                    .collect::<Vec<_>>())
            }
        };
        let definition = definition.with_symbolic_zero_vjp(forward, backward);
        if self.deferred_forward { definition.with_deferred_forward_rule() } else { definition }
    }
}

impl<V, O, Input, Output, Residual> CompositeVjp<V, O, Input, Output>
    for AccumulatingVjp<CustomRuleTracer<V, O>, TranspositionContext<V, O>, Input, Output, Residual>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: 'static + Value<Type: Eq + Hash + Send + DifferentiableType + ReferenceMemberType>,
    O: 'static + CustomVjpOperation<V>,
    Input: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send + Sync>,
    Input::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V> + ParameterizedFamily<CotangentAccumulator>,
    Input::To<V::Type>: Clone + Parameterized<V::Type, Family = Input::Family, To<CustomRuleTracer<V, O>> = Input>,
    Output: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send + Sync>,
    Output::Family:
        ParameterizedFamily<V::Type> + ParameterizedFamily<V> + ParameterizedFamily<MaybeZero<CustomRuleTracer<V, O>>>,
    Output::To<V::Type>: Parameterized<V::Type, Family = Output::Family, To<CustomRuleTracer<V, O>> = Output>,
    Residual: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send>,
    Residual::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Residual::To<V::Type>: Parameterized<V::Type, Family = Residual::Family, To<CustomRuleTracer<V, O>> = Residual>,
{
    fn configure(
        &self,
        definition: CustomRuleDefinition<V, O>,
        name: &Cow<'static, str>,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
        non_differentiated_count: usize,
    ) -> CustomRuleDefinition<V, O> {
        let residual_structures = CompositeResidualStructures::<V::Type, Residual::ParameterStructure>::default();
        let forward = retained_custom_vjp_forward_rule::<V, O, Input, Output, Residual, _>(
            self.forward.clone(),
            name,
            input_structure,
            output_structure,
            &residual_structures,
            definition.named_axes(),
        );

        // The rule is invoked directly while its carrier is transposed, with the carrier's inputs (i.e., the
        // non-differentiated inputs, the residuals, and the differentiated inputs' tangents) and one accumulator per
        // carrier input, so it receives the known residuals and the accumulators of the call's inputs.
        let backward = {
            let (backward, name) = (self.backward.clone(), name.clone());
            let (input_structure, output_structure) = (input_structure.clone(), output_structure.clone());
            move |context: &mut TranspositionContext<V, O>,
                  inputs: &[PartialValue<CustomRuleTracer<V, O>>],
                  seeds: &[MaybeZero<CustomRuleTracer<V, O>>],
                  accumulators: &[CotangentAccumulator]| {
                let differentiated_count = input_structure.parameter_count() - non_differentiated_count;
                let leading_input_count = inputs.len() - differentiated_count;
                let residuals = inputs[non_differentiated_count..leading_input_count]
                    .iter()
                    .enumerate()
                    .map(|(index, input)| {
                        input.as_known().cloned().ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "`{name}` backward rule residual {index} is not known during transposition",
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let residual_types = residuals.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
                let residuals = Residual::from_parameters(residual_structures.get(&name, &residual_types)?, residuals)?;
                let seeds = Output::To::<MaybeZero<CustomRuleTracer<V, O>>>::from_parameters(
                    output_structure.clone(),
                    seeds.iter().cloned(),
                )?;
                let accumulators = Input::To::<CotangentAccumulator>::from_parameters(
                    input_structure.clone(),
                    accumulators[..non_differentiated_count]
                        .iter()
                        .chain(&accumulators[leading_input_count..])
                        .cloned(),
                )?;
                backward(context, residuals, seeds, accumulators)
            }
        };
        let definition = definition.with_accumulating_vjp(forward, backward);
        if self.deferred_forward { definition.with_deferred_forward_rule() } else { definition }
    }
}

/// Batching configuration of a [`Composite`] without a custom batching rule. Batching a call structurally batches its
/// primal region.
#[derive(Copy, Clone, Debug, Default)]
pub struct DefaultBatching;

/// Batching configuration of a [`Composite`] with a user-supplied custom batching rule closure implementing
/// `(extent, x, x_axes) ↦ (y, y_axes)` (refer to [`Composite::with_batching`]). `Tracer` is the tracer type of the
/// closure's leaves, which is also the type of dynamic extents.
///
/// The name follows the vocabulary of JAX's
/// [`custom_vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_batching.custom_vmap.html) for a
/// user-supplied batching rule, whereas the `Explicit*` configurations (e.g., [`ExplicitJvp`]) name user-supplied
/// derivative rules that replace the ones otherwise derived from the primal.
pub struct CustomBatching<Tracer: Parameter, Input, Output>
where
    Input: Parameterized<Tracer, Family: ParameterizedFamily<BatchAxis>>,
    Output: Parameterized<Tracer, Family: ParameterizedFamily<BatchAxis>>,
{
    /// Closure computing the batched outputs and their batch axes from the batched inputs and their batch axes.
    rule: Arc<
        dyn Fn(
                BatchingLevelExtent<Tracer>,
                Input,
                Input::To<BatchAxis>,
            ) -> Result<(Output, Output::To<BatchAxis>), ProgramError>
            + Send
            + Sync,
    >,
}

impl<Tracer: Parameter, Input, Output> Debug for CustomBatching<Tracer, Input, Output>
where
    Input: Parameterized<Tracer, Family: ParameterizedFamily<BatchAxis>>,
    Output: Parameterized<Tracer, Family: ParameterizedFamily<BatchAxis>>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("CustomBatching").finish_non_exhaustive()
    }
}

/// Batching configuration of a [`Composite`] with a user-supplied custom batching rule closure implementing
/// `(extent, x, x_axes) ↦ (y, y_axes)` whose result may depend on its input batch axes (refer to
/// [`Composite::with_axis_dependent_batching`]). `Tracer` is the tracer type of the closure's leaves, which is also the
/// type of dynamic extents.
pub struct AxisDependentBatching<Tracer: Parameter, Input, Output>
where
    Input: Parameterized<Tracer, Family: ParameterizedFamily<BatchAxis>>,
    Output: Parameterized<Tracer, Family: ParameterizedFamily<BatchAxis>>,
{
    /// Closure computing the batched outputs and their batch axes from the batched inputs and their batch axes.
    rule: Arc<
        dyn Fn(
                BatchingLevelExtent<Tracer>,
                Input,
                Input::To<BatchAxis>,
            ) -> Result<(Output, Output::To<BatchAxis>), ProgramError>
            + Send
            + Sync,
    >,
}

impl<Tracer: Parameter, Input, Output> Debug for AxisDependentBatching<Tracer, Input, Output>
where
    Input: Parameterized<Tracer, Family: ParameterizedFamily<BatchAxis>>,
    Output: Parameterized<Tracer, Family: ParameterizedFamily<BatchAxis>>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("AxisDependentBatching").finish_non_exhaustive()
    }
}

/// Batching configuration of a [`Composite`] that batches its calls sequentially, by looping over the batch items with
/// a scan (refer to [`Composite::with_sequential_batching`]).
#[derive(Copy, Clone, Debug, Default)]
pub struct SequentialBatching;

/// Batching configuration of a [`Composite`] (i.e., [`DefaultBatching`], [`CustomBatching`], [`AxisDependentBatching`],
/// or [`SequentialBatching`]), which installs its custom batching rule or sequential batching, if any, in the retained
/// definition that a call registers for the operation family `(V, O)`.
pub trait CompositeBatching<V: Value, O: Operation<Type = V::Type>, Input, Output>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    Input: Parameterized<CustomRuleTracer<V, O>>,
    Output: Parameterized<CustomRuleTracer<V, O>>,
{
    /// Returns `definition` with this batching rule.
    ///
    /// # Parameters
    ///
    ///   - `definition`: Definition that the call registers.
    ///   - `name`: Name of the user-facing function, used in diagnostics.
    ///   - `input_structure`: Structure of the call's inputs.
    ///   - `output_structure`: Structure of the call's outputs.
    fn configure(
        &self,
        definition: CustomRuleDefinition<V, O>,
        name: &Cow<'static, str>,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
    ) -> CustomRuleDefinition<V, O>;
}

impl<V: Value, O: Operation<Type = V::Type>, Input, Output> CompositeBatching<V, O, Input, Output> for DefaultBatching
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    Input: Parameterized<CustomRuleTracer<V, O>>,
    Output: Parameterized<CustomRuleTracer<V, O>>,
{
    #[inline]
    fn configure(
        &self,
        definition: CustomRuleDefinition<V, O>,
        _name: &Cow<'static, str>,
        _input_structure: &Input::ParameterStructure,
        _output_structure: &Output::ParameterStructure,
    ) -> CustomRuleDefinition<V, O> {
        definition
    }
}

impl<V, O, Input, Output> CompositeBatching<V, O, Input, Output>
    for CustomBatching<CustomRuleTracer<V, O>, Input, Output>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: 'static + Value<Type: DifferentiableType + Eq + Hash>,
    O: 'static
        + Operation<Type = V::Type>
        + PartiallyEvaluatableOperation<TracingContext<V, O>>
        + DifferentiableOperation<TracingContext<V, O>>
        + DifferentiableOperation<PartialEvaluationContext<TracingContext<V, O>>>
        + ResidualZeroProvider<V::Type, Operation = O>,
    Input: 'static
        + Parameterized<CustomRuleTracer<V, O>, Family: ParameterizedFamily<BatchAxis>, ParameterStructure: Send + Sync>,
    Output: 'static
        + Parameterized<
            CustomRuleTracer<V, O>,
            Family: ParameterizedFamily<BatchAxis>,
            ParameterStructure: Debug + PartialEq + Send + Sync,
        >,
{
    fn configure(
        &self,
        definition: CustomRuleDefinition<V, O>,
        name: &Cow<'static, str>,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
    ) -> CustomRuleDefinition<V, O> {
        definition.with_batching_rule(retained_custom_batching_rule::<V, O, Input, Output, _>(
            self.rule.clone(),
            name,
            input_structure,
            output_structure,
        ))
    }
}

impl<V, O, Input, Output> CompositeBatching<V, O, Input, Output>
    for AxisDependentBatching<CustomRuleTracer<V, O>, Input, Output>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: 'static
        + Value<
            Type: DifferentiableType + BatchableType<Policy: DiagonalBatchingPolicy<TracingContext<V, O>>> + Eq + Hash,
        >,
    O: 'static
        + Operation<Type = V::Type>
        + PartiallyEvaluatableOperation<TracingContext<V, O>>
        + DifferentiableOperation<TracingContext<V, O>>
        + DifferentiableOperation<PartialEvaluationContext<TracingContext<V, O>>>
        + ResidualZeroProvider<V::Type, Operation = O>
        + OperationProvider<V::Type, AddOperation<V::Type>, Operation = O>,
    Input: 'static
        + Parameterized<CustomRuleTracer<V, O>, Family: ParameterizedFamily<BatchAxis>, ParameterStructure: Send + Sync>,
    Output: 'static
        + Parameterized<
            CustomRuleTracer<V, O>,
            Family: ParameterizedFamily<BatchAxis>,
            ParameterStructure: Debug + PartialEq + Send + Sync,
        >,
{
    fn configure(
        &self,
        definition: CustomRuleDefinition<V, O>,
        name: &Cow<'static, str>,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
    ) -> CustomRuleDefinition<V, O> {
        definition.with_axis_dependent_batching_rule(retained_custom_batching_rule::<V, O, Input, Output, _>(
            self.rule.clone(),
            name,
            input_structure,
            output_structure,
        ))
    }
}

impl<V, O, Input, Output> CompositeBatching<V, O, Input, Output> for SequentialBatching
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: 'static + Value<Type: Eq + Hash + ScanType>,
    O: 'static + Operation<Type = V::Type> + From<ScanOperation<V::Type>>,
    Input: Parameterized<CustomRuleTracer<V, O>>,
    Output: Parameterized<CustomRuleTracer<V, O>>,
{
    #[inline]
    fn configure(
        &self,
        definition: CustomRuleDefinition<V, O>,
        _name: &Cow<'static, str>,
        _input_structure: &Input::ParameterStructure,
        _output_structure: &Output::ParameterStructure,
    ) -> CustomRuleDefinition<V, O> {
        definition.with_sequential_batching()
    }
}

/// Adapts the structured custom batching rule closure `rule` of a [`Composite`] to the flat interface of retained
/// batching rules: each invocation passes the closure the level's extent, the call's own batched inputs with the call's
/// input structure, and their batch axes, and validates that its outputs and their batch axes have the call's output
/// structure.
///
/// # Parameters
///
///   - `rule`: Closure implementing `(extent, x, x_axes) ↦ (y, y_axes)`.
///   - `name`: Name of the function, used in diagnostics.
///   - `input_structure`: Structure of the call's inputs.
///   - `output_structure`: Structure of the call's outputs.
fn retained_custom_batching_rule<V, O, Input, Output, Rule>(
    rule: Arc<Rule>,
    name: &Cow<'static, str>,
    input_structure: &Input::ParameterStructure,
    output_structure: &Output::ParameterStructure,
) -> impl 'static
+ Fn(
    &BatchingLevel<V::Type>,
    &[CustomRuleTracer<V, O>],
    &[CustomRuleTracer<V, O>],
    &[BatchAxis],
) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<BatchAxis>), ProgramError>
+ Send
+ Sync
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: 'static + Value,
    O: 'static + Operation<Type = V::Type>,
    Input: 'static
        + Parameterized<CustomRuleTracer<V, O>, Family: ParameterizedFamily<BatchAxis>, ParameterStructure: Send + Sync>,
    Output: 'static
        + Parameterized<
            CustomRuleTracer<V, O>,
            Family: ParameterizedFamily<BatchAxis>,
            ParameterStructure: Debug + PartialEq + Send + Sync,
        >,
    Rule: ?Sized
        + 'static
        + Fn(
            BatchingLevelExtent<CustomRuleTracer<V, O>>,
            Input,
            Input::To<BatchAxis>,
        ) -> Result<(Output, Output::To<BatchAxis>), ProgramError>
        + Send
        + Sync,
{
    let name = name.clone();
    let (input_structure, output_structure) = (input_structure.clone(), output_structure.clone());
    move |level: &BatchingLevel<V::Type>,
          boundary_inputs: &[CustomRuleTracer<V, O>],
          inputs: &[CustomRuleTracer<V, O>],
          input_axes: &[BatchAxis]| {
        // A dynamic extent reaches the rule as the level's leading boundary input, and the level's other boundary
        // inputs do not reach the rule. The inputs of a call that was batched at earlier levels start with those
        // levels' boundary inputs, which the rule does not receive either.
        let extent = match (level.extent(), boundary_inputs) {
            (BatchingLevelExtent::Static(extent), _) => BatchingLevelExtent::Static(*extent),
            (BatchingLevelExtent::Dynamic(_), [extent, ..]) => BatchingLevelExtent::Dynamic(extent.clone()),
            (BatchingLevelExtent::Dynamic(_), []) => {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "{} batching rule requires a dynamic batch extent to be the leading boundary input of its \
                         batching level, but the level has no boundary inputs",
                        describe_composite(&name),
                    ),
                });
            }
        };
        let leading_input_count = inputs
            .len()
            .checked_sub(input_structure.parameter_count())
            .ok_or_else(|| ProgramError::MalformedProgram(format!("`{name}` batching rule received too few inputs")))?;
        let (outputs, output_axes) = rule(
            extent,
            Input::from_parameters(input_structure.clone(), inputs[leading_input_count..].iter().cloned())?,
            Input::To::<BatchAxis>::from_parameters(
                input_structure.clone(),
                input_axes[leading_input_count..].iter().copied(),
            )?,
        )?;
        for (value, structure) in
            [("output", outputs.parameter_structure()), ("output axis", output_axes.parameter_structure())]
        {
            if structure != output_structure {
                return Err(rule_structure_mismatch(&name, "batching", value, &output_structure, &structure).into());
            }
        }
        Ok((outputs.into_parameters().collect(), output_axes.into_parameters().collect()))
    }
}

/// Residual structures that the forward rule of one [`Composite`] registration returned, from which its backward rule
/// learns the structure of the residuals that it receives as flat leading inputs. The backward rule of each
/// specialization is traced after the forward rule of that specialization, but specializations of one call structure
/// may return differently structured residuals (e.g., collections whose lengths depend on the input shapes), so the
/// structures are recorded by the flat residual types that the forward rule returned, and only two structures with
/// equivalent flat residual types are ambiguous.
///
/// Flat residual types are compared up to a bijective renaming of their type identities, because the backward rule
/// observes the residuals of a replay of the traced forward rule. Replaying a program that computes type identities
/// (e.g., a dimension `n * n`) mints fresh identities, so the residual types that the backward rule observes differ
/// from the recorded ones by exactly such a renaming.
struct CompositeResidualStructures<T, Structure> {
    /// Recorded flat residual types, each with the structure of the residuals that had them.
    structures: Arc<Mutex<Vec<(Vec<T>, Structure)>>>,
}

impl<T: Type, Structure: Clone + Debug + PartialEq> CompositeResidualStructures<T, Structure> {
    /// Records that the forward rule of the function named `name` returned residuals of the provided flat types with
    /// the provided structure.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] when residuals of equivalent flat types were recorded with a different structure.
    fn record(&self, name: &str, residual_types: Vec<T>, structure: Structure) -> Result<(), TypeError> {
        let mut structures = self.structures.lock().expect("composite residual mutex is poisoned");
        let recorded = structures
            .iter()
            .find(|(types, _)| derive_bijective_identity_renaming(types, &residual_types).is_some());
        match recorded {
            Some((_, recorded)) if recorded != &structure => {
                Err(rule_structure_mismatch(name, "forward", "residual", recorded, &structure))
            }
            Some(_) => Ok(()),
            None => {
                structures.push((residual_types, structure));
                Ok(())
            }
        }
    }

    /// Returns the structure of residuals of the provided flat types for the backward rule of the function named
    /// `name`.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when no forward rule returned residuals of equivalent types, which means that the
    /// backward rule was traced before the forward rule of its specialization.
    fn get(&self, name: &str, residual_types: &[T]) -> Result<Structure, ProgramError> {
        self.structures
            .lock()
            .expect("composite residual mutex is poisoned")
            .iter()
            .find(|(types, _)| derive_bijective_identity_renaming(types, residual_types).is_some())
            .map(|(_, structure)| structure.clone())
            .ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "`{name}` backward rule was traced before the forward rule that produces its residuals",
                ))
            })
    }
}

impl<T, Structure> Clone for CompositeResidualStructures<T, Structure> {
    fn clone(&self) -> Self {
        Self { structures: self.structures.clone() }
    }
}

impl<T, Structure> Default for CompositeResidualStructures<T, Structure> {
    fn default() -> Self {
        Self { structures: Arc::new(Mutex::new(Vec::new())) }
    }
}

/// Adapts the structured reverse-mode forward closure `forward` of a [`Composite`] to the flat interface of retained
/// forward rules: each invocation traces the closure at the types of its primal inputs, records the structure of its
/// residuals in `residual_structures`, and replays the traced program on the primal inputs.
///
/// # Parameters
///
///   - `forward`: Closure implementing `x ↦ (y, r)`.
///   - `name`: Name of the function, used in diagnostics.
///   - `input_structure`: Structure of the call's inputs.
///   - `output_structure`: Structure of the call's primal outputs.
///   - `residual_structures`: Residual structures shared with the backward rule of the same registration.
///   - `named_axes`: Named axes with which the closure is traced (refer to
///     [`CustomRuleDefinition::with_named_axes`]).
fn retained_custom_vjp_forward_rule<V, O, Input, Output, Residual, Forward>(
    forward: Arc<Forward>,
    name: &Cow<'static, str>,
    input_structure: &Input::ParameterStructure,
    output_structure: &Output::ParameterStructure,
    residual_structures: &CompositeResidualStructures<V::Type, Residual::ParameterStructure>,
    named_axes: &[(String, NamedAxis)],
) -> impl 'static
+ Fn(&[CustomRuleTracer<V, O>]) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>), ProgramError>
+ Send
+ Sync
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: 'static + Value<Type: Eq + Hash + Send + DifferentiableType + ReferenceMemberType>,
    O: 'static + Operation<Type = V::Type>,
    Input: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send + Sync>,
    Input::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Input::To<V::Type>: Clone + Parameterized<V::Type, Family = Input::Family, To<CustomRuleTracer<V, O>> = Input>,
    Output: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send + Sync>,
    Output::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Residual: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send>,
    Residual::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Residual::To<V::Type>: Parameterized<V::Type, Family = Residual::Family, To<CustomRuleTracer<V, O>> = Residual>,
    Forward: ?Sized + 'static + Fn(Input) -> Result<(Output, Residual), ProgramError> + Send + Sync,
{
    let (name, residual_structures, named_axes) = (name.clone(), residual_structures.clone(), named_axes.to_vec());
    let (input_structure, output_structure) = (input_structure.clone(), output_structure.clone());
    move |primals: &[CustomRuleTracer<V, O>]| {
        let input_types = Input::To::<V::Type>::from_parameters(
            input_structure.clone(),
            primals.iter().map(|primal| primal.r#type().into_owned()),
        )?;
        let (structure, residual_types, program) =
            trace_custom_vjp_forward_rule::<V, O, Input, Output, Residual, Forward>(
                &name,
                forward.as_ref(),
                input_types,
                &output_structure,
                named_axes.clone(),
            )?;
        residual_structures.record(&name, residual_types, structure)?;
        let mut outputs = program.interpret_in_context(primals[0].context(), primals.to_vec())?;
        let residuals = outputs.split_off(output_structure.parameter_count());
        Ok((outputs, residuals))
    }
}

/// Traces the structured custom Vector-Jacobian Product (VJP) forward closure `forward` at `input_types` in the
/// `(V, O)` family and returns the structure and flat types of its residuals together with its flat rule program over
/// the call's inputs, which returns `[outputs..., residuals...]`. The closure's outputs must have the primal output
/// structure `output_structure`, because matching flattened types do not establish matching nested collection lengths,
/// and every reference-typed residual must be an input forwarded by identity, because reference residuals preserve
/// handles rather than snapshots (the reference boundary separately requires reference inputs to be
/// non-differentiated).
///
/// # Parameters
///
///   - `name`: Name of the user-facing function, used in diagnostics.
///   - `forward`: Closure implementing `x ↦ (y, r)`.
///   - `input_types`: Types of the call's inputs, with the call's input structure.
///   - `output_structure`: Structure of the call's primal outputs.
///   - `named_axes`: Named axes with which the closure is traced (refer to
///     [`CustomRuleDefinition::with_named_axes`]).
fn trace_custom_vjp_forward_rule<V, O, Input, Output, Residual, Forward>(
    name: &str,
    forward: &Forward,
    input_types: Input::To<V::Type>,
    output_structure: &Output::ParameterStructure,
    named_axes: Vec<(String, NamedAxis)>,
) -> Result<(Residual::ParameterStructure, Vec<V::Type>, Program<V, O, Vec<V>, Vec<V>>), ProgramError>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: Value<Type: DifferentiableType>,
    O: Operation<Type = V::Type>,
    Input: Parameterized<CustomRuleTracer<V, O>>,
    Input::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Input::To<V::Type>: Parameterized<V::Type, Family = Input::Family, To<CustomRuleTracer<V, O>> = Input>,
    Output: Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq>,
    Output::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Residual: Parameterized<CustomRuleTracer<V, O>>,
    Residual::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Forward: ?Sized + Fn(Input) -> Result<(Output, Residual), ProgramError>,
{
    let mut residual_structure = None;
    let ((forward_output_types, residual_types), program) =
        DomainTracingContext::<EagerContext<V, O>>::trace_with_named_axes(
            |input| {
                let (outputs, residuals) = forward(input)?;
                residual_structure = Some(residuals.parameter_structure());
                Ok((outputs, residuals))
            },
            input_types,
            named_axes,
        )?;
    let forward_output_structure = forward_output_types.parameter_structure();
    if &forward_output_structure != output_structure {
        return Err(
            rule_structure_mismatch(name, "forward", "output", output_structure, &forward_output_structure).into()
        );
    }

    // Reference residuals preserve handles, not snapshots. Validate their input identities while the traced atoms are
    // available; matching types alone cannot distinguish two inputs of the same reference type. The operation
    // subsequently checks that reference inputs belong to the non-differentiated prefix.
    let output_count = forward_output_types.parameter_count();
    let mut forwarded_inputs = HashSet::new();
    for (index, residual) in program.output_ids().iter().skip(output_count).enumerate() {
        let r#type = program.atoms()[residual.index()].r#type();
        if !r#type.is_reference() {
            continue;
        }
        if !program.input_ids().contains(residual) {
            return Err(TypeError::invalid(format!(
                "`{name}` forward rule returns residual {index} of reference type `{type}` that is not a leading \
                 non-differentiated input forwarded by identity",
            ))
            .into());
        }
        if !forwarded_inputs.insert(*residual) {
            return Err(TypeError::invalid(format!(
                "`{name}` forward rule returns residual {index} of reference type `{type}` from an input already \
                 forwarded by an earlier residual",
            ))
            .into());
        }
    }

    // The closure ran exactly once, so the structure of its residuals is always recorded.
    Ok((residual_structure.unwrap(), residual_types.parameters().cloned().collect(), program.into_flat_program()))
}

/// Traces the structured custom Vector-Jacobian Product (VJP) backward closure `backward` in the `(V, O)` family and
/// returns its flat rule program over `[non_differentiated_inputs..., residuals..., output_cotangents...]`, which
/// returns one cotangent per differentiated input. The closure sees only the residuals and the output cotangents, so
/// the leading non-differentiated inputs are declared as unused program inputs (a plumbing value that the backward rule
/// needs is forwarded to it as a residual), and the input-shaped cotangent value that the closure returns must have the
/// call's input structure `input_structure` before its leading non-differentiated leaves are dropped.
///
/// # Parameters
///
///   - `name`: Name of the user-facing function, used in diagnostics.
///   - `backward`: Closure implementing `(r, ȳ) ↦ x̄`.
///   - `input_structure`: Structure of the call's inputs.
///   - `non_differentiated_types`: Types of the leading non-differentiated inputs.
///   - `residual_structure`: Structure of the residuals that the forward closure returned.
///   - `residual_types`: Flat types of those residuals.
///   - `output_structure`: Structure of the call's outputs.
///   - `output_cotangent_types`: Flat cotangent types of the call's outputs.
///   - `named_axes`: Named axes with which the closure is traced (refer to
///     [`CustomRuleDefinition::with_named_axes`]).
fn trace_custom_vjp_backward_rule<V, O, Input, Output, Residual, Backward>(
    name: &str,
    backward: &Backward,
    input_structure: &Input::ParameterStructure,
    non_differentiated_types: Vec<V::Type>,
    residual_structure: Residual::ParameterStructure,
    residual_types: Vec<V::Type>,
    output_structure: Output::ParameterStructure,
    output_cotangent_types: Vec<V::Type>,
    named_axes: Vec<(String, NamedAxis)>,
) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: Value<Type: DifferentiableType>,
    O: Operation<Type = V::Type>,
    Input: Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq>,
    Output: Parameterized<CustomRuleTracer<V, O>>,
    Output::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Output::To<V::Type>: Parameterized<V::Type, Family = Output::Family, To<CustomRuleTracer<V, O>> = Output>,
    Residual: Parameterized<CustomRuleTracer<V, O>>,
    Residual::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Residual::To<V::Type>: Parameterized<V::Type, Family = Residual::Family, To<CustomRuleTracer<V, O>> = Residual>,
    Backward: ?Sized + Fn(Residual, Output) -> Result<Input, ProgramError>,
{
    let non_differentiated_count = non_differentiated_types.len();
    let residual_types = Residual::To::<V::Type>::from_parameters(residual_structure, residual_types)?;
    let output_cotangent_types = Output::To::<V::Type>::from_parameters(output_structure, output_cotangent_types)?;
    let (_, program) = DomainTracingContext::<EagerContext<V, O>>::trace_with_named_axes(
        |(_, residuals, cotangents): (Vec<CustomRuleTracer<V, O>>, Residual, Output)| {
            let cotangents = backward(residuals, cotangents)?;

            // Check the full input structure before dropping the non-differentiated prefix. Otherwise, a misplaced
            // cotangent could silently become the derivative of a different input with the same leaf type.
            let cotangent_structure = cotangents.parameter_structure();
            if &cotangent_structure != input_structure {
                return Err(rule_structure_mismatch(
                    name,
                    "backward",
                    "cotangent",
                    input_structure,
                    &cotangent_structure,
                )
                .into());
            }
            Ok(cotangents.into_parameters().skip(non_differentiated_count).collect::<Vec<_>>())
        },
        (non_differentiated_types, residual_types, output_cotangent_types),
        named_axes,
    )?;
    Ok(program.into_flat_program())
}

/// Traces the structured custom Vector-Jacobian Product (VJP) backward closure `backward`, which receives and returns
/// structural zeros, in the `(V, O)` family and returns its flat rule program over
/// `[non_differentiated_inputs..., residuals..., non_zero_output_cotangents...]`, which returns the cotangent of each
/// differentiated input that is not a structural zero, together with the type of each differentiated input's
/// structural-zero cotangent (or [`None`] for a cotangent that the program returns). The closure receives a seed value
/// whose leaves are the non-zero seeds and [`MaybeZero::Zero`]s at the positions of the structural-zero seeds, it
/// returns a cotangent value whose leaves are [`MaybeZero`]s (i.e., the analogue of a JAX backward rule returning
/// `None` or a `SymbolicZero` for an input), and it is otherwise validated as by [`trace_custom_vjp_backward_rule`].
///
/// # Parameters
///
///   - `name`: Name of the user-facing function, used in diagnostics.
///   - `backward`: Closure implementing `(r, ȳ) ↦ x̄`.
///   - `input_structure`: Structure of the call's inputs.
///   - `non_differentiated_types`: Types of the leading non-differentiated inputs.
///   - `residual_structure`: Structure of the residuals that the forward closure returned.
///   - `residual_types`: Flat types of those residuals.
///   - `output_structure`: Structure of the call's outputs.
///   - `seeds`: Output cotangent seeds, of which only the types and the structural zeros are used.
///   - `named_axes`: Named axes with which the closure is traced (refer to
///     [`CustomRuleDefinition::with_named_axes`]).
fn trace_symbolic_zero_custom_vjp_backward_rule<V, O, Input, Output, Residual, Backward>(
    name: &str,
    backward: &Backward,
    input_structure: &Input::ParameterStructure,
    non_differentiated_types: Vec<V::Type>,
    residual_structure: Residual::ParameterStructure,
    residual_types: Vec<V::Type>,
    output_structure: Output::ParameterStructure,
    seeds: &[MaybeZero<CustomRuleTracer<V, O>>],
    named_axes: Vec<(String, NamedAxis)>,
) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Vec<Option<V::Type>>), ProgramError>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: Value<Type: DifferentiableType>,
    O: Operation<Type = V::Type>,
    Input: Parameterized<
            CustomRuleTracer<V, O>,
            Family: ParameterizedFamily<MaybeZero<CustomRuleTracer<V, O>>>,
            ParameterStructure: Debug + PartialEq,
        >,
    Output: Parameterized<CustomRuleTracer<V, O>, Family: ParameterizedFamily<MaybeZero<CustomRuleTracer<V, O>>>>,
    Residual: Parameterized<CustomRuleTracer<V, O>>,
    Residual::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Residual::To<V::Type>: Parameterized<V::Type, Family = Residual::Family, To<CustomRuleTracer<V, O>> = Residual>,
    Backward: ?Sized
        + Fn(
            Residual,
            Output::To<MaybeZero<CustomRuleTracer<V, O>>>,
        ) -> Result<Input::To<MaybeZero<CustomRuleTracer<V, O>>>, ProgramError>,
{
    let non_differentiated_count = non_differentiated_types.len();
    let residual_types = Residual::To::<V::Type>::from_parameters(residual_structure, residual_types)?;
    let live_seed_types = seeds
        .iter()
        .filter_map(MaybeZero::as_value)
        .map(|seed| seed.r#type().into_owned())
        .collect::<Vec<_>>();
    let mut cotangent_zeros = None;
    let (_, program) = DomainTracingContext::<EagerContext<V, O>>::trace_with_named_axes(
        |(_, residuals, live_seeds): (Vec<CustomRuleTracer<V, O>>, Residual, Vec<CustomRuleTracer<V, O>>)| {
            let mut live_seeds = live_seeds.into_iter();
            let leaves = seeds.iter().map(|seed| match seed {
                MaybeZero::Value(_) => MaybeZero::Value(live_seeds.next().unwrap()),
                MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type.clone()),
            });
            let seeds =
                Output::To::<MaybeZero<CustomRuleTracer<V, O>>>::from_parameters(output_structure.clone(), leaves)?;
            let cotangents = backward(residuals, seeds)?;

            // Check the full input structure before dropping the non-differentiated prefix, as for materialized seeds.
            let cotangent_structure = cotangents.parameter_structure();
            if &cotangent_structure != input_structure {
                return Err(rule_structure_mismatch(
                    name,
                    "backward",
                    "cotangent",
                    input_structure,
                    &cotangent_structure,
                )
                .into());
            }

            // Only the live cotangents become program outputs, while the structural zeros are recorded by type.
            let mut zeros = Vec::new();
            let mut live_cotangents = Vec::new();
            for cotangent in cotangents.into_parameters().skip(non_differentiated_count) {
                match cotangent {
                    MaybeZero::Value(cotangent) => {
                        zeros.push(None);
                        live_cotangents.push(cotangent);
                    }
                    MaybeZero::Zero(r#type) => zeros.push(Some(r#type)),
                }
            }
            cotangent_zeros = Some(zeros);
            Ok(live_cotangents)
        },
        (non_differentiated_types, residual_types, live_seed_types),
        named_axes,
    )?;

    // The closure ran exactly once, so the structural zeros are always recorded.
    Ok((program.into_flat_program(), cotangent_zeros.unwrap()))
}

/// Traces the structured custom Jacobian-Vector Product (JVP) closure `jvp` at `input_types` in the `(V, O)` family and
/// returns its flat rule program over `[inputs..., differentiated_input_tangents...]`, which returns
/// `[outputs..., output_tangents...]`. Both halves of the closure's result must have the primal output structure
/// `output_structure`, because flattened signatures cannot distinguish differently nested collections with identical
/// leaf types, and the tangent placeholders of the leading `non_differentiated_count` inputs must be unused (refer to
/// [`without_non_differentiated_tangent_inputs`]). The retained JVP rules of [`Composite`]s are traced with this
/// function on their first derivative request.
///
/// # Parameters
///
///   - `name`: Name of the user-facing function, used in diagnostics.
///   - `jvp`: Closure implementing `(x, ẋ) ↦ (y, ẏ)`.
///   - `input_types`: Types of the call's inputs, with the call's input structure.
///   - `output_structure`: Structure of the call's primal outputs.
///   - `non_differentiated_count`: Number of leading non-differentiated input leaves.
///   - `named_axes`: Named axes with which the closure is traced (refer to
///     [`CustomRuleDefinition::with_named_axes`]).
fn trace_custom_jvp_rule<V, O, Input, Output, Jvp>(
    name: &str,
    jvp: &Jvp,
    input_types: Input::To<V::Type>,
    output_structure: &Output::ParameterStructure,
    non_differentiated_count: usize,
    named_axes: Vec<(String, NamedAxis)>,
) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: Value<Type: DifferentiableType>,
    O: Operation<Type = V::Type>,
    Input: Parameterized<CustomRuleTracer<V, O>>,
    Input::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Input::To<V::Type>: Clone + Parameterized<V::Type, Family = Input::Family, To<CustomRuleTracer<V, O>> = Input>,
    Output: Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq>,
    Output::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Jvp: ?Sized + Fn(Input, Input) -> Result<(Output, Output), ProgramError>,
{
    let input_count = input_types.parameter_count();
    let input_tangent_types = input_types.clone().try_map_parameters(|r#type| r#type.tangent())?;
    let ((output_types, tangent_types), program) = DomainTracingContext::<EagerContext<V, O>>::trace_with_named_axes(
        |(x, t)| jvp(x, t),
        (input_types, input_tangent_types),
        named_axes,
    )?;
    for (value, rule_structure) in
        [("output", output_types.parameter_structure()), ("output tangent", tangent_types.parameter_structure())]
    {
        if &rule_structure != output_structure {
            return Err(rule_structure_mismatch(name, "JVP", value, output_structure, &rule_structure).into());
        }
    }
    without_non_differentiated_tangent_inputs(name, program.into_flat_program(), input_count, non_differentiated_count)
}

/// Traces the structured custom Jacobian-Vector Product (JVP) closure `jvp`, which receives and returns structural
/// zeros, at `input_types` in the `(V, O)` family and returns its flat rule program over
/// `[inputs..., active_tangents...]`, which returns `[outputs..., live_output_tangents...]`, together with whether each
/// output tangent is a structural zero that the program does not return. The closure receives a tangent value whose
/// leaves are the active tangents of the differentiated inputs and [`MaybeZero::Zero`]s at the positions of their
/// inactive tangents and of every non-differentiated input, and it returns an output tangent value with [`MaybeZero`]
/// leaves, whose structural zeros must have the tangent types of their outputs. Both halves of the closure's result
/// must have the primal output structure `output_structure`.
///
/// # Parameters
///
///   - `name`: Name of the user-facing function, used in diagnostics.
///   - `jvp`: Closure implementing `(x, ẋ) ↦ (y, ẏ)`.
///   - `input_types`: Types of the call's inputs, with the call's input structure.
///   - `tangent_activity`: Whether the tangent of each differentiated input is active.
///   - `output_structure`: Structure of the call's primal outputs.
///   - `non_differentiated_count`: Number of leading non-differentiated input leaves.
///   - `named_axes`: Named axes with which the closure is traced (refer to
///     [`CustomRuleDefinition::with_named_axes`]).
fn trace_symbolic_zero_custom_jvp_rule<V, O, Input, Output, Jvp>(
    name: &str,
    jvp: &Jvp,
    input_types: Input::To<V::Type>,
    tangent_activity: &[bool],
    output_structure: &Output::ParameterStructure,
    non_differentiated_count: usize,
    named_axes: Vec<(String, NamedAxis)>,
) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Vec<bool>), ProgramError>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: Value<Type: DifferentiableType>,
    O: Operation<Type = V::Type>,
    Input: Parameterized<CustomRuleTracer<V, O>>,
    Input::Family:
        ParameterizedFamily<V::Type> + ParameterizedFamily<V> + ParameterizedFamily<MaybeZero<CustomRuleTracer<V, O>>>,
    Input::To<V::Type>: Clone + Parameterized<V::Type, Family = Input::Family, To<CustomRuleTracer<V, O>> = Input>,
    Output: Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq>,
    Output::Family:
        ParameterizedFamily<V::Type> + ParameterizedFamily<V> + ParameterizedFamily<MaybeZero<CustomRuleTracer<V, O>>>,
    Jvp: ?Sized
        + Fn(
            Input,
            Input::To<MaybeZero<CustomRuleTracer<V, O>>>,
        ) -> Result<(Output, Output::To<MaybeZero<CustomRuleTracer<V, O>>>), ProgramError>,
{
    let input_structure = input_types.parameter_structure();
    let tangent_types = input_types.parameters().map(|r#type| r#type.tangent()).collect::<Result<Vec<_>, _>>()?;
    let active_tangent_types = tangent_types[non_differentiated_count..]
        .iter()
        .zip(tangent_activity)
        .filter_map(|(r#type, active)| active.then(|| r#type.clone()))
        .collect::<Vec<_>>();
    let mut output_tangent_zeros = None;
    let ((output_types, _), program) = DomainTracingContext::<EagerContext<V, O>>::trace_with_named_axes(
        |(inputs, active_tangents): (Input, Vec<CustomRuleTracer<V, O>>)| {
            let mut active_tangents = active_tangents.into_iter();
            let leaves = tangent_types.iter().enumerate().map(|(index, r#type)| {
                let active = index.checked_sub(non_differentiated_count).is_some_and(|index| tangent_activity[index]);
                match active {
                    true => MaybeZero::Value(active_tangents.next().unwrap()),
                    false => MaybeZero::Zero(r#type.clone()),
                }
            });
            let (outputs, output_tangents) = jvp(
                inputs,
                Input::To::<MaybeZero<CustomRuleTracer<V, O>>>::from_parameters(input_structure.clone(), leaves)?,
            )?;
            let output_tangent_structure = output_tangents.parameter_structure();
            if &output_tangent_structure != output_structure {
                return Err(rule_structure_mismatch(
                    name,
                    "JVP",
                    "output tangent",
                    output_structure,
                    &output_tangent_structure,
                )
                .into());
            }

            // Only the live output tangents become program outputs, while each structural zero must have the tangent
            // type of its output, because the replay reinserts it with that type.
            let mut zeros = Vec::new();
            let mut live_output_tangents = Vec::new();
            for (index, (output, tangent)) in outputs.parameters().zip(output_tangents.into_parameters()).enumerate() {
                match tangent {
                    MaybeZero::Value(tangent) => {
                        zeros.push(false);
                        live_output_tangents.push(tangent);
                    }
                    MaybeZero::Zero(r#type) => {
                        let expected = output.r#type().tangent()?;
                        if r#type != expected {
                            return Err(TypeError::invalid(format!(
                                "{} JVP rule returned a structural-zero tangent of type `{type}` for output {index}, \
                                 whose tangent type is `{expected}`",
                                describe_composite(name),
                            ))
                            .into());
                        }
                        zeros.push(true);
                    }
                }
            }
            output_tangent_zeros = Some(zeros);
            Ok((outputs, live_output_tangents))
        },
        (input_types, active_tangent_types),
        named_axes,
    )?;
    let output_type_structure = output_types.parameter_structure();
    if &output_type_structure != output_structure {
        return Err(rule_structure_mismatch(name, "JVP", "output", output_structure, &output_type_structure).into());
    }

    // The closure ran exactly once, so the structural zeros are always recorded.
    Ok((program.into_flat_program(), output_tangent_zeros.unwrap()))
}

/// Returns the error reported when the `value` structure (e.g., `"output"`) of the `rule` rule (e.g., `"JVP"`) of the
/// function named `name` differs from the `expected` one. It is reported like the type mismatches of rule outputs,
/// because flattened signatures cannot distinguish differently nested collections with identical leaf types.
fn rule_structure_mismatch<Structure: Debug>(
    name: &str,
    rule: &str,
    value: &str,
    expected: &Structure,
    actual: &Structure,
) -> TypeError {
    TypeError::invalid(format!(
        "{} {rule} rule {value} structure mismatch: expected {expected:?} but got {actual:?}",
        describe_composite(name),
    ))
}

/// Removes the tangent inputs that a traced JVP rule declares for the leading `non_differentiated_count` inputs. The
/// rule closure receives one tangent per input so that its signature mirrors the primal signature, but a
/// non-differentiated input has no tangent slot in the [`CompositeOperation`] contract, so its tangent input is a
/// placeholder that the rule must ignore. The traced program has inputs `[inputs..., input_tangents...]`, and the
/// placeholders are the tangents at positions `input_count..input_count + non_differentiated_count`, which are
/// projected away so that the remaining boundary is exactly `[inputs..., differentiated_input_tangents...]`.
///
/// # Parameters
///
///   - `name`: Name of the user-facing function, used in diagnostics.
///   - `program`: Traced JVP rule program over `[inputs..., input_tangents...]`.
///   - `input_count`: Number of primal inputs.
///   - `non_differentiated_count`: Number of leading non-differentiated inputs whose tangent placeholders are removed.
///
/// # Errors
///
/// Returns a [`TypeError`] when the rule consumes or returns a placeholder tangent, and propagates program projection
/// errors otherwise.
fn without_non_differentiated_tangent_inputs<V: Value, O: Operation<Type = V::Type>>(
    name: &str,
    program: Program<V, O, Vec<V>, Vec<V>>,
    input_count: usize,
    non_differentiated_count: usize,
) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError> {
    if non_differentiated_count == 0 {
        return Ok(program);
    }
    let input_ids = program.input_ids();
    check_count!("input", input_ids, 2 * input_count, ProgramError);
    let (primal_ids, tangent_ids) = input_ids.split_at(input_count);
    let (placeholder_ids, differentiated_tangent_ids) = tangent_ids.split_at(non_differentiated_count);
    for (index, placeholder) in placeholder_ids.iter().enumerate() {
        // Attached regions are closed over their own inputs, so a use of an entry input is always a direct input or
        // output of the entry region.
        let used = program.output_ids().contains(placeholder)
            || program.instructions().iter().any(|instruction| instruction.inputs().contains(placeholder));
        if used {
            return Err(TypeError::invalid(format!(
                "`{name}` rule uses the tangent of leading non-differentiated input {index}, which has no tangent \
                 slot because non-differentiated inputs parameterize the rule without being differentiated",
            ))
            .into());
        }
    }
    let kept_ids = primal_ids.iter().chain(differentiated_tangent_ids).copied().collect::<Vec<_>>();
    let (pruned, _) = program.filtered(kept_ids.as_slice(), program.output_ids(), kept_ids.as_slice())?;
    Ok(pruned)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReference, ArrayReferenceTransform,
        ArraySliceAxis, ArrayType, DataType, Dimension, DimensionBounds, DimensionType, DimensionValue, Shape,
        ShardingDimension,
    };
    use crate::axes::AxisError;
    use crate::batching::{
        BatchAxis, BatchAxisSpecification, BatchedProgram, BatchingError, ProgramBatchingOutputAxesPolicy, batch,
    };
    use crate::contexts::{Context, EagerContext, ProjectedContext};
    use crate::differentiation::{
        CotangentDestination, CotangentDestinationKind, CotangentSeed, DifferentiationError, DifferentiationRule,
        differentiate_at,
    };
    use crate::macros::check_gradient;
    use crate::operations::arithmetic::{AddOperation, MulOperation};
    use crate::operations::collectives::axis_index::AxisIndex;
    use crate::operations::collectives::parallel_reduce::ParallelReduce;
    use crate::operations::constants::zero::Zero;
    use crate::operations::custom_call::CustomCallBatching;
    use crate::operations::differentiation::stop_gradient::StopGradient;
    use crate::operations::dimensions::dimension_size::DimensionSize;
    use crate::operations::dot::{Dot, DotDimensionNumbers};
    use crate::operations::manipulation::broadcasting::DynamicBroadcast;
    use crate::operations::manipulation::conversions::{Cast, ConvertElementType};
    use crate::operations::manipulation::padding::PadOperation;
    use crate::operations::manipulation::slicing::Slice;
    use crate::operations::reductions::{Reduce, ReductionKind};
    use crate::operations::references::{
        ReferenceAddUpdate, ReferenceAddUpdateOperation, ReferenceNew, ReferenceRead, ReferenceWrite,
    };
    use crate::operations::trigonometric::{Cos, Sin};
    use crate::parameters::Placeholder;
    use crate::programs::{FlatProgram, MaybeZero, ReferenceType, ValueProjection};
    use crate::tracing::Trace;
    use crate::operations::composites::tests::IrTracer;

    use super::*;

    #[test]
    fn test_retained_custom_batching_rule_leading_boundary_extent() {
        // A dynamic batch extent is the leading boundary input of its batching level, which may have further boundary
        // inputs that the rule does not receive.
        type IrTracer = CustomRuleTracer<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let rule = |extent: BatchingLevelExtent<IrTracer>, _input: IrTracer, _axis: BatchAxis| match extent {
            BatchingLevelExtent::Dynamic(extent) => Ok((extent, BatchAxis::replicated())),
            BatchingLevelExtent::Static(_) => {
                Err(ProgramError::InvalidArgument { message: "the test rule expects a dynamic extent".to_string() })
            }
        };
        let adapter = retained_custom_batching_rule::<_, _, IrTracer, IrTracer, _>(
            Arc::new(rule),
            &Cow::Borrowed("rule"),
            &Placeholder,
            &Placeholder,
        );
        let extent_type = ArrayIrType::from(DimensionType::new("n", DimensionBounds::new(1, Some(8)).unwrap()));
        let other_type = ArrayIrType::from(DimensionType::new("m", DimensionBounds::new(1, Some(8)).unwrap()));
        let level =
            BatchingLevel::new(BatchingLevelExtent::Dynamic(extent_type.clone()), None, ShardingDimension::Replicated);
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |values: Vec<IrTracer>| {
                let (outputs, output_axes) = adapter(&level, &values[..2], &values[2..], &[BatchAxis::replicated()])?;
                assert_eq!(output_axes, vec![BatchAxis::replicated()]);
                Ok(outputs)
            },
            vec![extent_type, other_type, ArrayType::scalar(DataType::F64).into()],
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<n ∈ [1, 8)>, %1:dimension<m ∈ [1, 8)>, %2:f64[] .
                in (%0)
            "}
            .trim_end(),
        );
    }
}
