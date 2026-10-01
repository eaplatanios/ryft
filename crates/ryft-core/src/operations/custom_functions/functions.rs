//! The [`CustomFunction`] builder, which pairs a primal closure with structured rule closures, together with the
//! type-state markers of its configurations (e.g., [`WithJvp`] and [`WithVjp`]) and the adapters that register the
//! structured closures as the flat, family-level rules of [`CustomRuleDefinition`]s. Each call of a function stages a
//! [`CustomFunctionOperation`] with retained rules.

// TODO(eaplatanios): Review this module.

use std::any::Any;
use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use crate::axes::{NamedAxes, NamedAxis};
use crate::batching::{BatchAxis, BatchableType, BatchingLevelExtent, RecursiveBatchingPolicy};
use crate::contexts::{Context, EagerContext};
use crate::differentiation::{
    CotangentAccumulator, CotangentBatchingPolicy, DifferentiableOperation, DifferentiableType, DifferentiationError,
    ResidualZeroProvider, TransposableOperation, TranspositionContext,
};
use crate::macros::check_count;
use crate::operations::arithmetic::AddOperation;
use crate::operations::custom_call::{CustomCall, CustomCallOperation};
use crate::operations::custom_functions::operations::{
    CUSTOM_FUNCTION_OPERATION_NAME, CustomFunctionOperation, validate_non_differentiated_count,
};
use crate::operations::custom_functions::rules::{
    CustomRuleDefinition, CustomRuleReference, CustomRuleRegistration, CustomRuleTracer,
    derive_bijective_identity_renaming,
};
use crate::operations::references::{ReferenceAddUpdateOperation, ReferenceNewOperation};
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily};
use crate::partial::{PartialEvaluationContext, PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, Operation, OperationProvider, Program, ProgramError, ReferenceAccessOperation, ReferenceMemberType,
    ReferenceTransform, Type, TypeError, Typed, Value,
};
use crate::tracing::{DomainTracer, DomainTracingContext, TracingContext};

/// Forward-mode configuration of a [`CustomFunction`] without a configured forward-mode rule. A function without
/// reverse-mode rules then derives its forward-mode rule from its primal (as [`JvpFromPrimal`] does), while a function
/// with reverse-mode rules rejects forward mode (refer to [`custom_function`] for the rule selection).
#[derive(Copy, Clone, Debug, Default)]
pub struct DefaultJvp;

/// Forward-mode configuration of a [`CustomFunction`] with a user-supplied Jacobian-Vector Product (JVP) rule closure
/// implementing `(x, ẋ) ↦ (y, ẏ)`.
pub struct WithJvp<Jvp>(Arc<Jvp>);

/// Forward-mode configuration of a [`CustomFunction`] whose forward-mode rule is derived from its primal (refer to
/// [`CustomFunction::with_jvp_from_primal`]).
#[derive(Copy, Clone, Debug, Default)]
pub struct JvpFromPrimal;

/// Forward-mode configuration of a [`CustomFunction`] with a user-supplied Jacobian-Vector Product (JVP) rule closure
/// implementing `(x, ẋ) ↦ (y, ẏ)` that receives `ẋ` with [`MaybeZero`] leaves, of which the structurally zero ones are
/// [`MaybeZero::Zero`]s (refer to [`CustomFunction::with_symbolic_zero_jvp`]).
pub struct WithSymbolicZeroJvp<Tracer, Jvp> {
    /// Closure computing `(outputs, output_tangents)` from the primal input value and the input tangent value.
    jvp: Arc<Jvp>,

    /// Phantom marker pinning the tracer type of the closure's input leaves, which the leaves of its tangents wrap.
    marker: PhantomData<fn() -> Tracer>,
}

/// Reverse-mode configuration of a [`CustomFunction`] without reverse-mode rules. Reverse mode then transposes the
/// forward-mode rule, which is derived from the primal when none is configured.
#[derive(Copy, Clone, Debug, Default)]
pub struct DefaultVjp;

/// Reverse-mode configuration of a [`CustomFunction`] with user-supplied forward and backward (i.e., Vector-Jacobian
/// Product or VJP) rule closures implementing `x ↦ (y, r)` and `(r, ȳ) ↦ x̄`.
pub struct WithVjp<Residual, Forward, Backward> {
    /// Closure computing `(outputs, residuals)` from the primal input value.
    forward: Arc<Forward>,

    /// Closure computing the input cotangent value from `(residuals, output_cotangents)`.
    backward: Arc<Backward>,

    /// Phantom marker pinning the residual tracer type named by the closure signatures.
    marker: PhantomData<fn() -> Residual>,
}

/// Reverse-mode configuration of a [`CustomFunction`] with user-supplied forward and backward (i.e., Vector-Jacobian
/// Product or VJP) rule closures implementing `x ↦ (y, r)` and `(r, ȳ) ↦ x̄`, whose backward rule receives `ȳ` with
/// [`MaybeZero`] leaves, of which the structurally zero ones are [`MaybeZero::Zero`]s (refer to
/// [`CustomFunction::with_symbolic_zero_vjp`]).
pub struct WithSymbolicZeroVjp<Residual, Tracer, Forward, Backward> {
    /// Closure computing `(outputs, residuals)` from the primal input value.
    forward: Arc<Forward>,

    /// Closure computing the input cotangent value from `(residuals, output_cotangent_seeds)`.
    backward: Arc<Backward>,

    /// Phantom marker pinning the residual type and the tracer type of the closures' output leaves, which the leaves
    /// of the seeds wrap.
    marker: PhantomData<fn() -> (Residual, Tracer)>,
}

/// Reverse-mode configuration of a [`CustomFunction`] with a user-supplied forward rule closure implementing
/// `x ↦ (y, r)` and an accumulating backward rule closure, which submits the input cotangents to their accumulators
/// (refer to [`CustomFunction::with_accumulating_vjp`]).
pub struct WithAccumulatingVjp<Residual, Tracer, Transposition, Forward, Backward> {
    /// Closure computing `(outputs, residuals)` from the primal input value.
    forward: Arc<Forward>,

    /// Closure submitting the input cotangents, computed from `(residuals, output_cotangent_seeds)`, to their
    /// accumulators.
    backward: Arc<Backward>,

    /// Phantom marker pinning the residual type, the tracer type of the closures' leaves, which the leaves of the seeds
    /// wrap, and the transposition context type named by the closure signatures.
    marker: PhantomData<fn() -> (Residual, Tracer, Transposition)>,
}

/// Forward-mode configuration of a [`CustomFunction`] (i.e., [`DefaultJvp`], [`WithJvp`], [`WithSymbolicZeroJvp`], or
/// [`JvpFromPrimal`]), which
/// installs its forward-mode rule in the retained definition that a call registers for the operation family `(V, O)`.
/// Its user-supplied rule is adapted to the flat interface of retained rules by tracing it, with the complete
/// structured interface of the call, at the types of each specialization and replaying the traced program.
pub trait CustomFunctionJvp<V: Value, O: Operation<Type = V::Type>, Input, Output>
where
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

impl<V: Value, O: Operation<Type = V::Type>, Input, Output> CustomFunctionJvp<V, O, Input, Output> for DefaultJvp
where
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

impl<V: Value<Type: Eq + Hash>, O: Operation<Type = V::Type>, Input, Output> CustomFunctionJvp<V, O, Input, Output>
    for JvpFromPrimal
where
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

impl<V, O, Input, Output, Jvp> CustomFunctionJvp<V, O, Input, Output> for WithJvp<Jvp>
where
    V: 'static + Value<Type: DifferentiableType + Eq + Hash>,
    O: 'static + Clone + Operation<Type = V::Type>,
    Input: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Send + Sync>,
    Input::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Input::To<V::Type>: Clone + Parameterized<V::Type, Family = Input::Family, To<CustomRuleTracer<V, O>> = Input>,
    Output: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send + Sync>,
    Output::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Jvp: 'static + Fn(Input, Input) -> Result<(Output, Output), ProgramError> + Send + Sync,
{
    fn configure(
        &self,
        definition: CustomRuleDefinition<V, O>,
        name: &Cow<'static, str>,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
        non_differentiated_count: usize,
    ) -> CustomRuleDefinition<V, O> {
        let (jvp, name, named_axes) = (self.0.clone(), name.clone(), definition.named_axes().to_vec());
        let (input_structure, output_structure) = (input_structure.clone(), output_structure.clone());
        definition.with_jvp(move |primals, tangents| {
            let input_types = Input::To::<V::Type>::from_parameters(
                input_structure.clone(),
                primals.iter().map(|primal| primal.r#type().into_owned()),
            )?;
            let program = trace_custom_jvp_rule::<V, O, Input, Output, Jvp>(
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

impl<V, O, Input, Output, Jvp> CustomFunctionJvp<V, O, Input, Output>
    for WithSymbolicZeroJvp<CustomRuleTracer<V, O>, Jvp>
where
    V: 'static + Value<Type: DifferentiableType + Eq + Hash>,
    O: 'static + Clone + Operation<Type = V::Type>,
    Input: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Send + Sync>,
    Input::Family:
        ParameterizedFamily<V::Type> + ParameterizedFamily<V> + ParameterizedFamily<MaybeZero<CustomRuleTracer<V, O>>>,
    Input::To<V::Type>: Clone + Parameterized<V::Type, Family = Input::Family, To<CustomRuleTracer<V, O>> = Input>,
    Output: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send + Sync>,
    Output::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Jvp: 'static
        + Fn(Input, Input::To<MaybeZero<CustomRuleTracer<V, O>>>) -> Result<(Output, Output), ProgramError>
        + Send
        + Sync,
{
    fn configure(
        &self,
        definition: CustomRuleDefinition<V, O>,
        _name: &Cow<'static, str>,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
        non_differentiated_count: usize,
    ) -> CustomRuleDefinition<V, O> {
        let (jvp, named_axes) = (self.jvp.clone(), definition.named_axes().to_vec());
        let (input_structure, output_structure) = (input_structure.clone(), output_structure.clone());
        definition.with_symbolic_zero_jvp(move |primals, tangents| {
            let input_types = Input::To::<V::Type>::from_parameters(
                input_structure.clone(),
                primals.iter().map(|primal| primal.r#type().into_owned()),
            )?;
            let tangent_activity = tangents.iter().map(|tangent| !tangent.is_zero()).collect::<Vec<_>>();
            let program = trace_symbolic_zero_custom_jvp_rule::<V, O, Input, Output, Jvp>(
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
            let output_tangents = outputs.split_off(output_structure.parameter_count());
            Ok((outputs, output_tangents))
        })
    }
}

/// Reverse-mode configuration of a [`CustomFunction`] (i.e., [`DefaultVjp`], [`WithVjp`], [`WithSymbolicZeroVjp`], or
/// [`WithAccumulatingVjp`]), which installs its reverse-mode rules in the retained definition that a call registers for
/// the operation family `(V, O)`. Its user-supplied rules are adapted to the flat interface of retained rules as for
/// [`CustomFunctionJvp`]. The backward rule receives the residuals with the structure that the forward rule returned
/// for the same specialization. Different specializations of one call structure may return differently structured
/// residuals, as long as residuals whose flat types are equal up to a renaming of their type identities (e.g., their
/// dimension variables) always have the same structure; the forward rule is rejected otherwise.
pub trait CustomFunctionVjp<V: Value, O: Operation<Type = V::Type>, Input, Output>
where
    Input: Parameterized<CustomRuleTracer<V, O>>,
    Output: Parameterized<CustomRuleTracer<V, O>>,
{
    /// Returns `definition` with these reverse-mode rules (refer to [`CustomFunctionJvp::configure`] for the
    /// parameters).
    fn configure(
        &self,
        definition: CustomRuleDefinition<V, O>,
        name: &Cow<'static, str>,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
        non_differentiated_count: usize,
    ) -> CustomRuleDefinition<V, O>;
}

impl<V: Value, O: Operation<Type = V::Type>, Input, Output> CustomFunctionVjp<V, O, Input, Output> for DefaultVjp
where
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

impl<V, O, Input, Output, Residual, Forward, Backward> CustomFunctionVjp<V, O, Input, Output>
    for WithVjp<Residual, Forward, Backward>
where
    V: 'static + Value<Type: DifferentiableType + ReferenceMemberType + Eq + Hash + Send>,
    O: 'static
        + TransposableOperation<V, O>
        + ResidualZeroProvider<V::Type, Operation = O>
        + ReferenceAccessOperation<Transform: ReferenceTransform<Referent = <V::Type as ReferenceMemberType>::Referent>>
        + OperationProvider<
            V::Type,
            ReferenceNewOperation<<V::Type as ReferenceMemberType>::Referent, V::Type>,
            Operation = O,
        >
        + OperationProvider<
            V::Type,
            ReferenceAddUpdateOperation<
                <V::Type as ReferenceMemberType>::Referent,
                V::Type,
                <O as ReferenceAccessOperation>::Transform,
            >,
            Operation = O,
        >
        + From<AddOperation<V::Type>>,
    Input: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send + Sync>,
    Input::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Input::To<V::Type>: Clone + Parameterized<V::Type, Family = Input::Family, To<CustomRuleTracer<V, O>> = Input>,
    Output: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send + Sync>,
    Output::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Output::To<V::Type>: Parameterized<V::Type, Family = Output::Family, To<CustomRuleTracer<V, O>> = Output>,
    Residual: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send>,
    Residual::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Residual::To<V::Type>: Parameterized<V::Type, Family = Residual::Family, To<CustomRuleTracer<V, O>> = Residual>,
    Forward: 'static + Fn(Input) -> Result<(Output, Residual), ProgramError> + Send + Sync,
    Backward: 'static + Fn(Residual, Output) -> Result<Input, ProgramError> + Send + Sync,
{
    fn configure(
        &self,
        definition: CustomRuleDefinition<V, O>,
        name: &Cow<'static, str>,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
        non_differentiated_count: usize,
    ) -> CustomRuleDefinition<V, O> {
        let residual_structures = CustomFunctionResidualStructures::<V::Type, Residual::ParameterStructure>::default();
        let forward = retained_custom_vjp_forward_rule::<V, O, Input, Output, Residual, Forward>(
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
                let program = trace_custom_vjp_backward_rule::<V, O, Input, Output, Residual, Backward>(
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
        definition.with_vjp(forward, backward)
    }
}

impl<V, O, Input, Output, Residual, Forward, Backward> CustomFunctionVjp<V, O, Input, Output>
    for WithSymbolicZeroVjp<Residual, CustomRuleTracer<V, O>, Forward, Backward>
where
    V: 'static + Value<Type: DifferentiableType + ReferenceMemberType + Eq + Hash + Send>,
    O: 'static
        + TransposableOperation<V, O>
        + ResidualZeroProvider<V::Type, Operation = O>
        + ReferenceAccessOperation<Transform: ReferenceTransform<Referent = <V::Type as ReferenceMemberType>::Referent>>
        + OperationProvider<
            V::Type,
            ReferenceNewOperation<<V::Type as ReferenceMemberType>::Referent, V::Type>,
            Operation = O,
        >
        + OperationProvider<
            V::Type,
            ReferenceAddUpdateOperation<
                <V::Type as ReferenceMemberType>::Referent,
                V::Type,
                <O as ReferenceAccessOperation>::Transform,
            >,
            Operation = O,
        >
        + From<AddOperation<V::Type>>,
    Input: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send + Sync>,
    Input::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Input::To<V::Type>: Clone + Parameterized<V::Type, Family = Input::Family, To<CustomRuleTracer<V, O>> = Input>,
    Output: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send + Sync>,
    Output::Family:
        ParameterizedFamily<V::Type> + ParameterizedFamily<V> + ParameterizedFamily<MaybeZero<CustomRuleTracer<V, O>>>,
    Output::To<V::Type>: Parameterized<V::Type, Family = Output::Family, To<CustomRuleTracer<V, O>> = Output>,
    Residual: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send>,
    Residual::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Residual::To<V::Type>: Parameterized<V::Type, Family = Residual::Family, To<CustomRuleTracer<V, O>> = Residual>,
    Forward: 'static + Fn(Input) -> Result<(Output, Residual), ProgramError> + Send + Sync,
    Backward: 'static
        + Fn(Residual, Output::To<MaybeZero<CustomRuleTracer<V, O>>>) -> Result<Input, ProgramError>
        + Send
        + Sync,
{
    fn configure(
        &self,
        definition: CustomRuleDefinition<V, O>,
        name: &Cow<'static, str>,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
        non_differentiated_count: usize,
    ) -> CustomRuleDefinition<V, O> {
        let residual_structures = CustomFunctionResidualStructures::<V::Type, Residual::ParameterStructure>::default();
        let forward = retained_custom_vjp_forward_rule::<V, O, Input, Output, Residual, Forward>(
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
                let program = trace_symbolic_zero_custom_vjp_backward_rule::<V, O, Input, Output, Residual, Backward>(
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
                program.interpret_in_context(&context, values)
            }
        };
        definition.with_symbolic_zero_vjp(forward, backward)
    }
}

impl<V, O, Input, Output, Residual, Forward, Backward> CustomFunctionVjp<V, O, Input, Output>
    for WithAccumulatingVjp<Residual, CustomRuleTracer<V, O>, TranspositionContext<V, O>, Forward, Backward>
where
    V: 'static + Value<Type: DifferentiableType + ReferenceMemberType + Eq + Hash + Send>,
    O: 'static
        + TransposableOperation<V, O>
        + ResidualZeroProvider<V::Type, Operation = O>
        + ReferenceAccessOperation<Transform: ReferenceTransform<Referent = <V::Type as ReferenceMemberType>::Referent>>
        + OperationProvider<
            V::Type,
            ReferenceNewOperation<<V::Type as ReferenceMemberType>::Referent, V::Type>,
            Operation = O,
        >
        + OperationProvider<
            V::Type,
            ReferenceAddUpdateOperation<
                <V::Type as ReferenceMemberType>::Referent,
                V::Type,
                <O as ReferenceAccessOperation>::Transform,
            >,
            Operation = O,
        >
        + From<AddOperation<V::Type>>,
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
    Forward: 'static + Fn(Input) -> Result<(Output, Residual), ProgramError> + Send + Sync,
    Backward: 'static
        + Fn(
            &mut TranspositionContext<V, O>,
            Residual,
            Output::To<MaybeZero<CustomRuleTracer<V, O>>>,
            Input::To<CotangentAccumulator>,
        ) -> Result<(), DifferentiationError>
        + Send
        + Sync,
{
    fn configure(
        &self,
        definition: CustomRuleDefinition<V, O>,
        name: &Cow<'static, str>,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
        non_differentiated_count: usize,
    ) -> CustomRuleDefinition<V, O> {
        let residual_structures = CustomFunctionResidualStructures::<V::Type, Residual::ParameterStructure>::default();
        let forward = retained_custom_vjp_forward_rule::<V, O, Input, Output, Residual, Forward>(
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
        definition.with_accumulating_vjp(forward, backward)
    }
}

/// Batching configuration of a [`CustomFunction`] without a custom batching rule. Batching a call structurally
/// batches its primal region.
#[derive(Copy, Clone, Debug, Default)]
pub struct DefaultBatching;

/// Batching configuration of a [`CustomFunction`] with a user-supplied custom batching rule closure implementing
/// `(extent, x, x_axes) ↦ (y, y_axes)` (refer to [`CustomFunction::with_batching`]).
pub struct WithBatching<Tracer, Rule> {
    /// Closure computing the batched outputs and their batch axes from the batched inputs and their batch axes.
    rule: Arc<Rule>,

    /// Phantom marker pinning the tracer type of the closure's leaves, which is also the type of dynamic extents.
    marker: PhantomData<fn() -> Tracer>,
}

/// Batching configuration of a [`CustomFunction`] (i.e., [`DefaultBatching`] or [`WithBatching`]), which installs its
/// custom batching rule, if any, in the retained definition that a call registers for the operation family `(V, O)`.
pub trait CustomFunctionBatching<V: Value, O: Operation<Type = V::Type>, Input, Output>
where
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

impl<V: Value, O: Operation<Type = V::Type>, Input, Output> CustomFunctionBatching<V, O, Input, Output>
    for DefaultBatching
where
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

impl<V, O, Input, Output, Rule> CustomFunctionBatching<V, O, Input, Output>
    for WithBatching<CustomRuleTracer<V, O>, Rule>
where
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
    Rule: 'static
        + Fn(
            BatchingLevelExtent<CustomRuleTracer<V, O>>,
            Input,
            Input::To<BatchAxis>,
        ) -> Result<(Output, Output::To<BatchAxis>), ProgramError>
        + Send
        + Sync,
{
    fn configure(
        &self,
        definition: CustomRuleDefinition<V, O>,
        name: &Cow<'static, str>,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
    ) -> CustomRuleDefinition<V, O> {
        let (rule, name) = (self.rule.clone(), name.clone());
        let (input_structure, output_structure) = (input_structure.clone(), output_structure.clone());
        definition.with_batching_rule(move |level, boundary_operands, inputs, input_axes| {
            // A dynamic extent reaches the rule as the level's only boundary operand. The inputs of a call that was
            // batched at earlier levels start with those levels' boundary operands, which the rule does not receive.
            let extent = match (level.extent(), boundary_operands) {
                (BatchingLevelExtent::Static(extent), _) => BatchingLevelExtent::Static(*extent),
                (BatchingLevelExtent::Dynamic(_), [extent]) => BatchingLevelExtent::Dynamic(extent.clone()),
                (BatchingLevelExtent::Dynamic(_), _) => {
                    return Err(ProgramError::UnsupportedOperation {
                        message: format!(
                            "`{name}` batching rule requires a dynamic batch extent to be the only boundary operand of \
                             its batching level, but the level has {} boundary operands",
                            boundary_operands.len(),
                        ),
                    });
                }
            };
            let leading_input_count = inputs.len().checked_sub(input_structure.parameter_count()).ok_or_else(|| {
                ProgramError::MalformedProgram(format!("`{name}` batching rule received too few inputs"))
            })?;
            let (outputs, output_axes) = rule(
                extent,
                Input::from_parameters(input_structure.clone(), inputs[leading_input_count..].iter().cloned())?,
                Input::To::<BatchAxis>::from_parameters(
                    input_structure.clone(),
                    input_axes[leading_input_count..].iter().copied(),
                )?,
            )?;
            for structure in [outputs.parameter_structure(), output_axes.parameter_structure()] {
                if structure != output_structure {
                    return Err(ParameterError::MismatchedParameterStructures {
                        left_structure: format!("{output_structure:?}"),
                        right_structure: format!("{structure:?}"),
                    }
                    .into());
                }
            }
            Ok((outputs.into_parameters().collect(), output_axes.into_parameters().collect()))
        })
    }
}

/// Residual structures that the forward rule of one [`CustomFunction`] registration returned, from which its
/// backward rule learns the structure of the residuals that it receives as flat leading inputs. The backward rule of
/// each specialization is traced after the forward rule of that specialization, but specializations of one call
/// structure may return differently structured residuals (e.g., collections whose lengths depend on the input shapes),
/// so the structures are recorded by the flat residual types that the forward rule returned, and only two structures
/// with equivalent flat residual types are ambiguous.
///
/// Flat residual types are compared up to a bijective renaming of their type identities, because the backward rule
/// observes the residuals of a replay of the traced forward rule. Replaying a program that computes type identities
/// (e.g., a dimension `n * n`) mints fresh identities, so the residual types that the backward rule observes differ
/// from the recorded ones by exactly such a renaming.
struct CustomFunctionResidualStructures<T, Structure> {
    /// Recorded flat residual types, each with the structure of the residuals that had them.
    structures: Arc<Mutex<Vec<(Vec<T>, Structure)>>>,
}

impl<T: Type, Structure: Clone + Debug + PartialEq> CustomFunctionResidualStructures<T, Structure> {
    /// Records that a forward rule returned residuals of the provided flat types with the provided structure.
    ///
    /// # Errors
    ///
    /// Returns a [`ParameterError`] when residuals of equivalent flat types were recorded with a different structure.
    fn record(&self, residual_types: Vec<T>, structure: Structure) -> Result<(), ParameterError> {
        let mut structures = self.structures.lock().expect("custom function residual mutex is poisoned");
        let recorded = structures
            .iter()
            .find(|(types, _)| derive_bijective_identity_renaming(types, &residual_types).is_some());
        match recorded {
            Some((_, recorded)) if recorded != &structure => Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{recorded:?}"),
                right_structure: format!("{structure:?}"),
            }),
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
            .expect("custom function residual mutex is poisoned")
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

impl<T, Structure> Clone for CustomFunctionResidualStructures<T, Structure> {
    fn clone(&self) -> Self {
        Self { structures: self.structures.clone() }
    }
}

impl<T, Structure> Default for CustomFunctionResidualStructures<T, Structure> {
    fn default() -> Self {
        Self { structures: Arc::new(Mutex::new(Vec::new())) }
    }
}

/// Adapts the structured reverse-mode forward closure `forward` of a [`CustomFunction`] to the flat interface of
/// retained forward rules: each invocation traces the closure at the types of its primal inputs, records the structure
/// of its residuals in `residual_structures`, and replays the traced program on the primal inputs.
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
    residual_structures: &CustomFunctionResidualStructures<V::Type, Residual::ParameterStructure>,
    named_axes: &[(String, NamedAxis)],
) -> impl 'static
+ Fn(&[CustomRuleTracer<V, O>]) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>), ProgramError>
+ Send
+ Sync
where
    V: 'static + Value<Type: DifferentiableType + ReferenceMemberType + Eq + Hash + Send>,
    O: 'static + Operation<Type = V::Type>,
    Input: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send + Sync>,
    Input::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Input::To<V::Type>: Clone + Parameterized<V::Type, Family = Input::Family, To<CustomRuleTracer<V, O>> = Input>,
    Output: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send + Sync>,
    Output::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Residual: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq + Send>,
    Residual::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Residual::To<V::Type>: Parameterized<V::Type, Family = Residual::Family, To<CustomRuleTracer<V, O>> = Residual>,
    Forward: 'static + Fn(Input) -> Result<(Output, Residual), ProgramError> + Send + Sync,
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
        residual_structures.record(residual_types, structure)?;
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
    V: Value<Type: DifferentiableType>,
    O: Operation<Type = V::Type>,
    Input: Parameterized<CustomRuleTracer<V, O>>,
    Input::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Input::To<V::Type>: Parameterized<V::Type, Family = Input::Family, To<CustomRuleTracer<V, O>> = Input>,
    Output: Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq>,
    Output::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Residual: Parameterized<CustomRuleTracer<V, O>>,
    Residual::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Forward: Fn(Input) -> Result<(Output, Residual), ProgramError>,
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
        return Err(ParameterError::MismatchedParameterStructures {
            left_structure: format!("{output_structure:?}"),
            right_structure: format!("{forward_output_structure:?}"),
        }
        .into());
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
    V: Value<Type: DifferentiableType>,
    O: Operation<Type = V::Type>,
    Input: Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq>,
    Output: Parameterized<CustomRuleTracer<V, O>>,
    Output::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Output::To<V::Type>: Parameterized<V::Type, Family = Output::Family, To<CustomRuleTracer<V, O>> = Output>,
    Residual: Parameterized<CustomRuleTracer<V, O>>,
    Residual::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Residual::To<V::Type>: Parameterized<V::Type, Family = Residual::Family, To<CustomRuleTracer<V, O>> = Residual>,
    Backward: Fn(Residual, Output) -> Result<Input, ProgramError>,
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
                return Err(ParameterError::MismatchedParameterStructures {
                    left_structure: format!("{input_structure:?}"),
                    right_structure: format!("{cotangent_structure:?}"),
                }
                .into());
            }
            Ok(cotangents.into_parameters().skip(non_differentiated_count).collect::<Vec<_>>())
        },
        (non_differentiated_types, residual_types, output_cotangent_types),
        named_axes,
    )?;
    Ok(program.into_flat_program())
}

/// Traces the structured custom Vector-Jacobian Product (VJP) backward closure `backward`, which receives structural
/// zeros, in the `(V, O)` family and returns its flat rule program over
/// `[non_differentiated_inputs..., residuals..., non_zero_output_cotangents...]`, which returns one cotangent per
/// differentiated input. The closure receives a seed value whose leaves are the non-zero seeds and [`MaybeZero::Zero`]s
/// at the positions of the structural-zero seeds, and it is otherwise validated as by
/// [`trace_custom_vjp_backward_rule`].
///
/// # Parameters
///
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
    backward: &Backward,
    input_structure: &Input::ParameterStructure,
    non_differentiated_types: Vec<V::Type>,
    residual_structure: Residual::ParameterStructure,
    residual_types: Vec<V::Type>,
    output_structure: Output::ParameterStructure,
    seeds: &[MaybeZero<CustomRuleTracer<V, O>>],
    named_axes: Vec<(String, NamedAxis)>,
) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError>
where
    V: Value<Type: DifferentiableType>,
    O: Operation<Type = V::Type>,
    Input: Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq>,
    Output: Parameterized<CustomRuleTracer<V, O>, Family: ParameterizedFamily<MaybeZero<CustomRuleTracer<V, O>>>>,
    Residual: Parameterized<CustomRuleTracer<V, O>>,
    Residual::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Residual::To<V::Type>: Parameterized<V::Type, Family = Residual::Family, To<CustomRuleTracer<V, O>> = Residual>,
    Backward: Fn(Residual, Output::To<MaybeZero<CustomRuleTracer<V, O>>>) -> Result<Input, ProgramError>,
{
    let non_differentiated_count = non_differentiated_types.len();
    let residual_types = Residual::To::<V::Type>::from_parameters(residual_structure, residual_types)?;
    let live_seed_types = seeds
        .iter()
        .filter_map(MaybeZero::as_value)
        .map(|seed| seed.r#type().into_owned())
        .collect::<Vec<_>>();
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
                return Err(ParameterError::MismatchedParameterStructures {
                    left_structure: format!("{input_structure:?}"),
                    right_structure: format!("{cotangent_structure:?}"),
                }
                .into());
            }
            Ok(cotangents.into_parameters().skip(non_differentiated_count).collect::<Vec<_>>())
        },
        (non_differentiated_types, residual_types, live_seed_types),
        named_axes,
    )?;
    Ok(program.into_flat_program())
}

/// Traces the structured custom Jacobian-Vector Product (JVP) closure `jvp` at `input_types` in the `(V, O)` family and
/// returns its flat rule program over `[inputs..., differentiated_input_tangents...]`, which returns
/// `[outputs..., output_tangents...]`. Both halves of the closure's result must have the primal output structure
/// `output_structure`, because flattened signatures cannot distinguish differently nested collections with identical
/// leaf types, and the tangent placeholders of the leading `non_differentiated_count` inputs must be unused (refer to
/// [`without_non_differentiated_tangent_inputs`]). The retained JVP rules of [`CustomFunction`] functions are traced
/// with this function on their first derivative request.
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
    V: Value<Type: DifferentiableType>,
    O: Clone + Operation<Type = V::Type>,
    Input: Parameterized<CustomRuleTracer<V, O>>,
    Input::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Input::To<V::Type>: Clone + Parameterized<V::Type, Family = Input::Family, To<CustomRuleTracer<V, O>> = Input>,
    Output: Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq>,
    Output::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Jvp: Fn(Input, Input) -> Result<(Output, Output), ProgramError>,
{
    let input_count = input_types.parameter_count();
    let input_tangent_types = input_types.clone().try_map_parameters(|r#type| r#type.tangent())?;
    let ((output_types, tangent_types), program) = DomainTracingContext::<EagerContext<V, O>>::trace_with_named_axes(
        |(x, t)| jvp(x, t),
        (input_types, input_tangent_types),
        named_axes,
    )?;
    for rule_structure in [output_types.parameter_structure(), tangent_types.parameter_structure()] {
        if &rule_structure != output_structure {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{output_structure:?}"),
                right_structure: format!("{rule_structure:?}"),
            }
            .into());
        }
    }
    without_non_differentiated_tangent_inputs(name, program.into_flat_program(), input_count, non_differentiated_count)
}

/// Traces the structured custom Jacobian-Vector Product (JVP) closure `jvp`, which receives structural zeros, at
/// `input_types` in the `(V, O)` family and returns its flat rule program over `[inputs..., active_tangents...]`, which
/// returns `[outputs..., output_tangents...]`. The closure receives a tangent value whose leaves are the active
/// tangents of the differentiated inputs and [`MaybeZero::Zero`]s at the positions of their inactive tangents and of
/// every non-differentiated input. Both halves of the closure's result must have the primal output structure
/// `output_structure`.
///
/// # Parameters
///
///   - `jvp`: Closure implementing `(x, ẋ) ↦ (y, ẏ)`.
///   - `input_types`: Types of the call's inputs, with the call's input structure.
///   - `tangent_activity`: Whether the tangent of each differentiated input is active.
///   - `output_structure`: Structure of the call's primal outputs.
///   - `non_differentiated_count`: Number of leading non-differentiated input leaves.
///   - `named_axes`: Named axes with which the closure is traced (refer to
///     [`CustomRuleDefinition::with_named_axes`]).
fn trace_symbolic_zero_custom_jvp_rule<V, O, Input, Output, Jvp>(
    jvp: &Jvp,
    input_types: Input::To<V::Type>,
    tangent_activity: &[bool],
    output_structure: &Output::ParameterStructure,
    non_differentiated_count: usize,
    named_axes: Vec<(String, NamedAxis)>,
) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError>
where
    V: Value<Type: DifferentiableType>,
    O: Clone + Operation<Type = V::Type>,
    Input: Parameterized<CustomRuleTracer<V, O>>,
    Input::Family:
        ParameterizedFamily<V::Type> + ParameterizedFamily<V> + ParameterizedFamily<MaybeZero<CustomRuleTracer<V, O>>>,
    Input::To<V::Type>: Clone + Parameterized<V::Type, Family = Input::Family, To<CustomRuleTracer<V, O>> = Input>,
    Output: Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Debug + PartialEq>,
    Output::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Jvp: Fn(Input, Input::To<MaybeZero<CustomRuleTracer<V, O>>>) -> Result<(Output, Output), ProgramError>,
{
    let input_structure = input_types.parameter_structure();
    let tangent_types = input_types.parameters().map(|r#type| r#type.tangent()).collect::<Result<Vec<_>, _>>()?;
    let active_tangent_types = tangent_types[non_differentiated_count..]
        .iter()
        .zip(tangent_activity)
        .filter_map(|(r#type, active)| active.then(|| r#type.clone()))
        .collect::<Vec<_>>();
    let ((output_types, output_tangent_types), program) =
        DomainTracingContext::<EagerContext<V, O>>::trace_with_named_axes(
            |(inputs, active_tangents): (Input, Vec<CustomRuleTracer<V, O>>)| {
                let mut active_tangents = active_tangents.into_iter();
                let leaves = tangent_types.iter().enumerate().map(|(index, r#type)| {
                    let active =
                        index.checked_sub(non_differentiated_count).is_some_and(|index| tangent_activity[index]);
                    match active {
                        true => MaybeZero::Value(active_tangents.next().unwrap()),
                        false => MaybeZero::Zero(r#type.clone()),
                    }
                });
                jvp(
                    inputs,
                    Input::To::<MaybeZero<CustomRuleTracer<V, O>>>::from_parameters(input_structure.clone(), leaves)?,
                )
            },
            (input_types, active_tangent_types),
            named_axes,
        )?;
    for rule_structure in [output_types.parameter_structure(), output_tangent_types.parameter_structure()] {
        if &rule_structure != output_structure {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{output_structure:?}"),
                right_structure: format!("{rule_structure:?}"),
            }
            .into());
        }
    }
    Ok(program.into_flat_program())
}

/// Removes the tangent inputs that a traced JVP rule declares for the leading `non_differentiated_count` inputs. The
/// rule closure receives one tangent per input so that its signature mirrors the primal signature, but a
/// non-differentiated input has no tangent slot in the [`CustomFunctionOperation`] contract, so its tangent input is a
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
fn without_non_differentiated_tangent_inputs<V: Value, O: Clone + Operation<Type = V::Type>>(
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

/// Retained definitions that one [`CustomFunction`] function registered, one per operation family, call structure
/// (i.e., input and output parameter structure), and visible named axes. The registration handles live as long as the
/// function, so every call staged by the function shares the specialization caches of its structure and named axes
/// until the function is dropped.
#[derive(Default)]
struct CustomFunctionRegistrations {
    /// Registrations, each a [`CustomFunctionRegistration`] of some family and structure types.
    entries: Mutex<Vec<Box<dyn Any + Send>>>,
}

/// Retained definition that a [`CustomFunction`] function registered for one operation family and call structure.
struct CustomFunctionRegistration<V: Typed + Parameter, O, InputStructure, OutputStructure> {
    /// Structure of the calls' inputs.
    input_structure: InputStructure,

    /// Structure of the calls' outputs.
    output_structure: OutputStructure,

    /// Named axes visible where the calls are made, with which the definition traces its rules.
    named_axes: Vec<(String, NamedAxis)>,

    /// Registration handle, which owns the definition's specialization caches.
    registration: CustomRuleRegistration<V, O>,
}

impl CustomFunctionRegistrations {
    /// Returns a reference to the definition registered for the family `(V, O)`, the provided call structure, and the
    /// provided named axes, registering the definition returned by `register_fn` if there is none yet.
    fn get_or_register<V, O, InputStructure, OutputStructure, F>(
        &self,
        input_structure: &InputStructure,
        output_structure: &OutputStructure,
        named_axes: &[(String, NamedAxis)],
        register_fn: F,
    ) -> CustomRuleReference<V, O>
    where
        V: 'static + Typed<Type: Eq + Hash> + Parameter,
        O: 'static,
        InputStructure: 'static + Clone + PartialEq + Send,
        OutputStructure: 'static + Clone + PartialEq + Send,
        CustomRuleRegistration<V, O>: Send,
        F: FnOnce() -> CustomRuleDefinition<V, O>,
    {
        let mut entries = self.entries.lock().expect("custom function registration mutex is poisoned");
        let existing = entries.iter().find_map(|entry| {
            entry
                .downcast_ref::<CustomFunctionRegistration<V, O, InputStructure, OutputStructure>>()
                .filter(|entry| {
                    &entry.input_structure == input_structure
                        && &entry.output_structure == output_structure
                        && entry.named_axes == named_axes
                })
                .map(|entry| entry.registration.reference())
        });
        existing.unwrap_or_else(|| {
            let registration = CustomRuleRegistration::new(register_fn());
            let reference = registration.reference();
            entries.push(Box::new(CustomFunctionRegistration {
                input_structure: input_structure.clone(),
                output_structure: output_structure.clone(),
                named_axes: named_axes.to_vec(),
                registration,
            }));
            reference
        })
    }
}

/// Function with custom derivative or batching rules, built by [`custom_function`]. It stores a primal closure together
/// with an optional forward-mode rule (refer to [`Self::with_jvp`] and [`Self::with_jvp_from_primal`]), optional
/// reverse-mode rules (refer to [`Self::with_vjp`]), and an optional batching rule (refer to [`Self::with_batching`]),
/// and each [`call`](Self::call) stages one [`CustomFunctionOperation`] with retained rules over the traced primal.
///
/// The rules are retained callbacks rather than traced programs: a call traces only its primal, and each rule is traced
/// on the first derivative request of each specialization (e.g., each distinct input signature or batching level) and
/// cached, so a program that is never differentiated never traces its rules. Consequently, a malformed rule is reported
/// when it is first needed rather than when the function is called. The rules must be `'static`, [`Send`], and
/// [`Sync`], because staged programs retain them.
///
/// The function owns the specialization caches of its calls: calls that share an operation family and a call
/// structure (i.e., input and output parameter structure) share one registered definition and its caches for as long
/// as the function lives. Once the function is dropped, the derivative requests of calls that it staged trace their
/// rules again on every request. The rules of a call cannot discharge reference state, so differentiating a call whose
/// reference state was discharged is rejected when its rule programs contain reference state (refer to
/// [`CustomRuleDefinition::with_reference_discharge`]).
///
/// The forward-mode configuration is part of the type (i.e., `Jvp` is [`DefaultJvp`], [`WithJvp`], or
/// [`JvpFromPrimal`]), so an explicit forward-mode rule and a forward-mode rule derived from the primal cannot both be
/// configured, whatever the order of the configuration calls. Refer to the documentation of the [`custom_function`]
/// function for the rule selection of every configuration.
pub struct CustomFunction<Input, Output, Primal, Jvp = DefaultJvp, Vjp = DefaultVjp, Batching = DefaultBatching> {
    /// Closure computing the primal output value from the primal input value.
    primal: Primal,

    /// Forward-mode configuration.
    jvp: Jvp,

    /// Reverse-mode configuration.
    vjp: Vjp,

    /// Batching configuration.
    batching: Batching,

    /// Number of leading flattened input leaves that parameterize the call without being differentiated.
    non_differentiated_count: usize,

    /// Name of the function, used in rendering and diagnostics.
    name: Cow<'static, str>,

    /// Definitions registered by the calls of this function.
    registrations: CustomFunctionRegistrations,

    /// Phantom marker pinning the input and output tracer types named by the closure signatures. The [`Context`]
    /// whose universe the rules are traced into is recovered from the values passed to [`CustomFunction::call`], and
    /// so the function stores neither a context value nor a context type witness.
    marker: PhantomData<fn() -> (Input, Output)>,
}

/// Primal closure of a [`CustomFunction`] built by [`CustomFunction::from_custom_call`], which calls one foreign
/// kernel on a vector of input values.
pub type CustomCallPrimal<V> = Box<dyn Fn(Vec<V>) -> Result<Vec<V>, ProgramError> + Send + Sync>;

impl<V: CustomCall> CustomFunction<Vec<V>, Vec<V>, CustomCallPrimal<V>> {
    /// Creates a [`CustomFunction`], without any custom rule, whose primal calls the foreign kernel described by
    /// `operation` (i.e., it stages a [`CustomCallOperation`] whose outputs have the operation's declared output
    /// types). Foreign kernels are opaque, so a custom call rejects differentiation, and this constructor is how a
    /// differentiable foreign kernel is defined: its configuration functions add the derivative rules (e.g.,
    /// [`Self::with_vjp`]), which may themselves call foreign kernels. Without derivative rules, differentiating the
    /// function differentiates its primal, which reaches the custom call and reports its error. This is the analogue of
    /// wrapping JAX's [`jax.ffi.ffi_call`](https://docs.jax.dev/en/latest/_autosummary/jax.ffi.ffi_call.html) with
    /// [`jax.custom_vjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html) or
    /// [`jax.custom_jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html).
    ///
    /// The function takes and returns vectors of values, and it is named after the operation's target, which labels
    /// its staged calls. Interpretation and backend lowering execute the kernel, while derivative requests replay the
    /// rules. The values must be arrays (i.e., implement [`CustomCall`]), so composite contexts (e.g., array IR
    /// contexts) call the function through their [`ProjectedContext`](crate::ProjectedContext) onto arrays, and every
    /// call requires at least one input, whose context the kernel call dispatches through.
    pub fn from_custom_call(operation: CustomCallOperation) -> Self {
        let name = operation.target_name().to_owned();
        let primal: CustomCallPrimal<V> = Box::new(move |inputs| CustomCall::custom_call(&operation, &inputs));
        custom_function(primal).with_name(name)
    }
}

impl<Input, Output, Primal, Jvp, Vjp, Batching> CustomFunction<Input, Output, Primal, Jvp, Vjp, Batching> {
    /// Returns this function with the leading `non_differentiated_count` flattened input leaves treated as
    /// non-differentiated _plumbing_ inputs. Refer to the documentation of the [`custom_function`] function for the
    /// semantics of non-differentiated inputs. Calls staged before this reconfiguration keep their original rules.
    #[inline]
    pub fn with_non_differentiated_count(self, non_differentiated_count: usize) -> Self {
        let Self { primal, jvp, vjp, batching, name, .. } = self;
        Self::with_configuration(primal, jvp, vjp, batching, non_differentiated_count, name)
    }

    /// Returns this function with the provided name, which labels its staged calls in rendering and diagnostics.
    /// Calls staged before this reconfiguration keep their original name.
    #[inline]
    pub fn with_name<N: Into<Cow<'static, str>>>(self, name: N) -> Self {
        let Self { primal, jvp, vjp, batching, non_differentiated_count, .. } = self;
        Self::with_configuration(primal, jvp, vjp, batching, non_differentiated_count, name.into())
    }

    /// Returns this function with the provided configurations and without registered definitions, because the
    /// definitions registered by earlier calls were configured with the previous configurations. The calls staged
    /// earlier keep their definitions, but since this drops the registrations that own their specialization caches,
    /// their derivative requests trace their rules again on every request (refer to [`CustomFunction`]).
    fn with_configuration<NewJvp, NewVjp, NewBatching>(
        primal: Primal,
        jvp: NewJvp,
        vjp: NewVjp,
        batching: NewBatching,
        non_differentiated_count: usize,
        name: Cow<'static, str>,
    ) -> CustomFunction<Input, Output, Primal, NewJvp, NewVjp, NewBatching> {
        CustomFunction {
            primal,
            jvp,
            vjp,
            batching,
            non_differentiated_count,
            name,
            registrations: CustomFunctionRegistrations::default(),
            marker: PhantomData,
        }
    }
}

impl<Input, Output, Primal, Vjp, Batching> CustomFunction<Input, Output, Primal, DefaultJvp, Vjp, Batching> {
    /// Returns this function with the provided Jacobian-Vector Product (JVP) rule, which governs forward-mode
    /// differentiation and, unless reverse-mode rules are configured as well, reverse-mode differentiation. This is the
    /// analogue of JAX's [`jax.custom_jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html) /
    /// [`defjvp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.defjvp.html) decorator pair.
    ///
    /// For `y = f(x)`, let `J_f(x) = ∂f/∂x` denote the Jacobian of `f` at `x`. The rule implements:
    ///
    /// ```text
    /// JVP:        (x, ẋ) ↦ (y, ẏ) = (f(x), J_f(x) · ẋ)
    /// ```
    ///
    /// Thus, `jvp` receives the input value `x` and an input-tangent value `ẋ`, then returns the primal output `y`
    /// together with the Jacobian-vector product `ẏ = J_f(x) · ẋ`, which must be linear in `ẋ`. The tangent values have
    /// the same parameter structures as their corresponding primal values, and Ryft validates these structural and type
    /// relationships when it traces the rule. The rule keeps receiving a full `ẋ` value when the function has
    /// non-differentiated inputs, so that its signature mirrors the primal signature, but the tangent leaves of those
    /// inputs are placeholders that it must not use, because the staged [`CustomFunctionOperation`] has no tangent
    /// slot for them. A rule that consumes or returns such a placeholder is rejected when it is first traced.
    ///
    /// # When to Use
    ///
    /// Reach for a custom JVP when the function _is_ forward-differentiable but its automatically derived tangent is
    /// numerically unstable or wasteful and you want to supply a stable, efficient one by hand. Classic cases are a
    /// `log`-`sum`-`exp`, a softmax, or a normalization, where a handwritten tangent avoids the cancellation or
    /// redundant work that the generic rule incurs. A single custom JVP serves **both** differentiation modes: reverse
    /// mode obtains its gradient by transposing the supplied tangent map, so the one rule composes with forward mode,
    /// reverse mode, and their higher-order combinations. Prefer it over [`Self::with_vjp`] whenever the function is
    /// naturally forward-differentiable, and use reverse-mode rules only when just the reverse rule is natural (e.g.,
    /// for implicit differentiation or adjoint solvers).
    ///
    /// # Transform Semantics
    ///
    /// Differentiation replays the rule program instead of differentiating the primal. The replayed rule consists of
    /// ordinary primitive operations, so reverse mode transposes the linear map in `ẋ` that it computes, exactly like
    /// any other tangent program, and the staged call itself is never transposed. Higher-order differentiation
    /// differentiates those replayed operations, so every combination of forward and reverse mode applies to the
    /// resulting derivatives, including differentiation of a pullback with respect to its cotangent seeds and
    /// transposing a pullback again (which restores the tangent map). The primal is kept separate from the rule for
    /// efficiency rather than necessity: the rule computes both the outputs and their tangents, so deriving the primal
    /// from it would make every un-differentiated call pay for tangent computation.
    ///
    /// # Parameters
    ///
    ///   - `jvp`: Closure implementing `(x, ẋ) ↦ (y, ẏ)`, where `ẏ = J_f(x) · ẋ`.
    #[inline]
    pub fn with_jvp<Jvp: 'static + Fn(Input, Input) -> Result<(Output, Output), ProgramError> + Send + Sync>(
        self,
        jvp: Jvp,
    ) -> CustomFunction<Input, Output, Primal, WithJvp<Jvp>, Vjp, Batching> {
        let Self { primal, vjp, batching, non_differentiated_count, name, .. } = self;
        Self::with_configuration(primal, WithJvp(Arc::new(jvp)), vjp, batching, non_differentiated_count, name)
    }

    /// Returns this function with a forward-mode rule derived from its primal: forward mode differentiates the primal,
    /// reusable linearization partitions that derivative once, and reverse mode transposes it unless reverse-mode rules
    /// are configured as well. A function without derivative rules already derives its forward-mode rule from its
    /// primal, so this configuration is needed only to pair reverse-mode rules with the derived forward-mode rule,
    /// which would otherwise make forward mode reject the calls.
    #[inline]
    pub fn with_jvp_from_primal(self) -> CustomFunction<Input, Output, Primal, JvpFromPrimal, Vjp, Batching> {
        let Self { primal, vjp, batching, non_differentiated_count, name, .. } = self;
        Self::with_configuration(primal, JvpFromPrimal, vjp, batching, non_differentiated_count, name)
    }

    /// Returns this function with the provided Jacobian-Vector Product (JVP) rule `(x, ẋ) ↦ (y, ẏ)`, which is used as
    /// the rule of [`Self::with_jvp`] except that it receives `ẋ` with [`MaybeZero`] leaves (i.e., as an
    /// `Input::To<MaybeZero<Tracer>>` value, such as `(MaybeZero<DomainTracer<C>>, MaybeZero<DomainTracer<C>>)` for an
    /// input `(DomainTracer<C>, DomainTracer<C>)`): the leaves of structurally zero input tangents (e.g., those of
    /// inputs that are not being differentiated, and those of every non-differentiated input) are
    /// [`MaybeZero::Zero`]s, which lets the rule skip the work that they would otherwise require. Its closure
    /// parameters must usually be annotated. Each pattern of structurally zero input tangents is a separate
    /// specialization of the rule. This is the analogue of JAX's `defjvp(..., symbolic_zeros=True)`.
    #[inline]
    pub fn with_symbolic_zero_jvp<Tracer, Jvp>(
        self,
        jvp: Jvp,
    ) -> CustomFunction<Input, Output, Primal, WithSymbolicZeroJvp<Tracer, Jvp>, Vjp, Batching>
    where
        Tracer: Typed + Parameter,
        Input: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
        Jvp: 'static + Fn(Input, Input::To<MaybeZero<Tracer>>) -> Result<(Output, Output), ProgramError> + Send + Sync,
    {
        let Self { primal, vjp, batching, non_differentiated_count, name, .. } = self;
        let jvp = WithSymbolicZeroJvp { jvp: Arc::new(jvp), marker: PhantomData };
        Self::with_configuration(primal, jvp, vjp, batching, non_differentiated_count, name)
    }
}

impl<Input, Output, Primal, Jvp, Batching> CustomFunction<Input, Output, Primal, Jvp, DefaultVjp, Batching> {
    /// Returns this function with the provided reverse-mode forward and backward (i.e., Vector-Jacobian Product or VJP)
    /// rules, which govern reverse-mode differentiation and take precedence over any forward-mode rule for it. This is
    /// the analogue of JAX's [`jax.custom_vjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html) /
    /// [`defvjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.defvjp.html) decorator pair.
    ///
    /// For `y = f(x)`, let `J_f(x) = ∂f/∂x` denote the Jacobian of `f` at `x`. The VJP, or pullback, maps an output
    /// cotangent `ȳ` to the input cotangent `x̄ = J_f(x)ᵀ · ȳ`. The two rules factor that computation through a
    /// residual value `r`:
    ///
    /// ```text
    /// Forward:      x      ↦ (y, r) = (f(x), r)
    /// Backward:     (r, ȳ) ↦ x̄ = J_f(x)ᵀ · ȳ
    /// ```
    ///
    /// Thus, `forward` recomputes `y` and saves exactly the residual value `r` needed by the reverse rule, and
    /// `backward` receives `r` and the output-cotangent value `ȳ`, then returns the input-cotangent value `x̄`. When
    /// tracing, Ryft validates matching parameter structures and the corresponding primal, residual, and cotangent
    /// types. The caller must ensure that the primal and `forward` compute the same value of `y`; tracing cannot
    /// establish their numerical equivalence. A dynamic value that `backward` needs must be preserved as a residual in
    /// `r`, and `backward` returns a zero cotangent for an input that it treats as a parameter.
    ///
    /// The primal is kept separate from `forward` for efficiency rather than necessity: an un-differentiated call
    /// should not pay for residual computation. Callers that do not care about the distinction can pass the same body
    /// for both, accepting that the residual outputs are dead code outside of differentiation (e.g., by writing
    /// `forward` as `|x| Ok((f(x)?, residuals))`).
    ///
    /// # When to Use
    ///
    /// Reach for reverse-mode rules when only the _reverse_ rule is natural, or when the function is not (efficiently)
    /// forward-differentiable. Common cases are:
    ///
    ///   - **Implicit Differentiation:** Differentiate through a solver, optimizer, or fixed point via the implicit
    ///     function theorem rather than unrolling its iterations.
    ///   - **Adjoint Methods:** Backpropagate through an Ordinary Differential Equation (ODE) or Partial Differential
    ///     Equation (PDE) solution via the adjoint system instead of differentiating the individual steps of the
    ///     integrator.
    ///   - **External or Black-Box Calls:** Supply the reverse rule for a custom kernel or for a computation that has
    ///     no automatic derivative (refer to [`Self::from_custom_call`]). The call must still trace as a Ryft operation
    ///     with interpretation and backend lowering support; the closures cannot execute arbitrary external code on
    ///     tracer values.
    ///   - **Numerical Stability:** Replace an unstable or wasteful automatically derived gradient with a handwritten
    ///     one.
    ///
    /// Without a forward-mode rule, active forward-mode differentiation and reusable forward linearization reject a
    /// call before executing its forward rule, because no pushforward rule was supplied. This rejection also occurs
    /// when constructing staged forward derivatives, rather than waiting for their execution. Calls with no active
    /// derivative inputs can still execute their primal directly. Add a forward-mode rule (e.g., with
    /// [`Self::with_jvp`] or [`Self::with_jvp_from_primal`]) when callers need direct forward-mode differentiation as
    /// well as reverse mode.
    ///
    /// # Non-Differentiated Inputs and References
    ///
    /// Non-differentiated inputs reach the primal and `forward` at their usual positions and receive no cotangent:
    /// `backward` keeps returning a full `x̄` value so that its signature mirrors the primal signature, but the leaves
    /// that it returns at non-differentiated positions are ignored, because the staged [`CustomFunctionOperation`]
    /// produces no cotangents for them. A non-differentiated value that `backward` needs must be forwarded to it as a
    /// residual in `r`.
    ///
    /// `forward` may return a non-differentiated reference inside `r`, in which case the reference itself (rather
    /// than a snapshot of its contents) is forwarded to `backward`. Every reference-typed residual must be a distinct
    /// non-differentiated input forwarded by identity, which the first reverse-mode derivative request validates when
    /// it traces `forward`. This interface supports externally owned mutable state, not privately allocated reference
    /// residuals; save immutable values when the backward rule needs a snapshot. This enables the _stash-gradients_
    /// pattern: a `stash` reference enters as a non-differentiated input, `forward` returns it as a residual, and
    /// `backward` writes the incoming `ȳ` into it before returning `x̄`.
    ///
    /// # Transform Semantics
    ///
    /// Reverse mode traces `forward` when it linearizes a call and `backward` when it transposes the resulting carrier.
    /// It replays the forward program for the primal outputs and residuals and stages a
    /// [`CustomFunctionTransposeOperation`](crate::CustomFunctionTransposeOperation) carrier whose transpose replays
    /// the backward program, so reverse mode uses exactly the user-supplied gradient, and the original call is never
    /// transposed. A call batched before it is differentiated batches both rule programs traced at the unbatched types
    /// and sums the cotangents that the batched backward program produces for replicated inputs over the batch axis.
    ///
    /// Higher-order differentiation differentiates the backward program's ordinary operations, because the pullback
    /// replays that program inline. Both forward and reverse mode apply to a pullback, with respect to the primal
    /// inputs (through the saved residuals) and to the cotangent seeds, even when the backward program is nonlinear in
    /// its seeds. This includes reverse-over-reverse and forward-over-reverse differentiation, but it does not supply a
    /// forward rule for a nested reverse-only call encountered along either path. Transposing a pullback again requires
    /// the backward program to be linear in its seeds and is rejected otherwise.
    ///
    /// # Parameters
    ///
    ///   - `forward`: Closure implementing `x ↦ (y, r)` for reverse-mode residual production.
    ///   - `backward`: Closure implementing `(r, ȳ) ↦ x̄ = J_f(x)ᵀ · ȳ`.
    #[inline]
    pub fn with_vjp<Residual, Forward, Backward>(
        self,
        forward: Forward,
        backward: Backward,
    ) -> CustomFunction<Input, Output, Primal, Jvp, WithVjp<Residual, Forward, Backward>, Batching>
    where
        Forward: 'static + Fn(Input) -> Result<(Output, Residual), ProgramError> + Send + Sync,
        Backward: 'static + Fn(Residual, Output) -> Result<Input, ProgramError> + Send + Sync,
    {
        let Self { primal, jvp, batching, non_differentiated_count, name, .. } = self;
        let vjp = WithVjp { forward: Arc::new(forward), backward: Arc::new(backward), marker: PhantomData };
        Self::with_configuration(primal, jvp, vjp, batching, non_differentiated_count, name)
    }

    /// Returns this function with the provided reverse-mode forward rule `x ↦ (y, r)` and backward rule `(r, ȳ) ↦ x̄`,
    /// which are used as the rules of [`Self::with_vjp`] except that the backward rule receives `ȳ` with [`MaybeZero`]
    /// leaves (i.e., as an `Output::To<MaybeZero<Tracer>>` value): the leaves of structural-zero cotangent seeds (e.g.,
    /// those of unused outputs) are [`MaybeZero::Zero`]s. Its closure parameters must usually be annotated. This is the
    /// analogue of JAX's `defvjp(..., symbolic_zeros=True)`, except that the forward rule receives no per-input
    /// differentiation flags.
    #[inline]
    pub fn with_symbolic_zero_vjp<Residual, Tracer, Forward, Backward>(
        self,
        forward: Forward,
        backward: Backward,
    ) -> CustomFunction<Input, Output, Primal, Jvp, WithSymbolicZeroVjp<Residual, Tracer, Forward, Backward>, Batching>
    where
        Tracer: Typed + Parameter,
        Output: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
        Forward: 'static + Fn(Input) -> Result<(Output, Residual), ProgramError> + Send + Sync,
        Backward: 'static + Fn(Residual, Output::To<MaybeZero<Tracer>>) -> Result<Input, ProgramError> + Send + Sync,
    {
        let Self { primal, jvp, batching, non_differentiated_count, name, .. } = self;
        let vjp = WithSymbolicZeroVjp { forward: Arc::new(forward), backward: Arc::new(backward), marker: PhantomData };
        Self::with_configuration(primal, jvp, vjp, batching, non_differentiated_count, name)
    }

    /// Returns this function with the provided reverse-mode forward rule `x ↦ (y, r)` and accumulating backward rule,
    /// which govern reverse-mode differentiation as the rules of [`Self::with_vjp`] do, except that the backward rule
    /// submits each input cotangent to its [`CotangentAccumulator`] instead of returning it. One rule therefore serves
    /// every destination kind of an input's cotangent: it can update only the affected entries of a caller-provided
    /// buffer (refer to [`CotangentAccumulator::reference`]) instead of returning a full-sized cotangent, and it can
    /// skip the inputs whose cotangents are ignored (refer to [`CotangentAccumulator::is_needed`]). The rule is
    /// specialized once per pattern of destination kinds, and the buffers themselves are never part of a
    /// specialization.
    ///
    /// The backward rule is invoked while the call's carrier is transposed, with the [`TranspositionContext`], the
    /// residuals `r` (with the structure that the forward rule returned), the output cotangent seeds with
    /// [`MaybeZero`] leaves (i.e., as an `Output::To<MaybeZero<Tracer>>` value), and the accumulators with
    /// [`CotangentAccumulator`] leaves (i.e., as an `Input::To<CotangentAccumulator>` value; the accumulators of
    /// non-differentiated inputs are never needed). Its closure parameters must usually be annotated.
    #[inline]
    pub fn with_accumulating_vjp<Residual, Tracer, Transposition, Forward, Backward>(
        self,
        forward: Forward,
        backward: Backward,
    ) -> CustomFunction<
        Input,
        Output,
        Primal,
        Jvp,
        WithAccumulatingVjp<Residual, Tracer, Transposition, Forward, Backward>,
        Batching,
    >
    where
        Tracer: Typed + Parameter,
        Input: Parameterized<Tracer, Family: ParameterizedFamily<CotangentAccumulator>>,
        Output: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
        Forward: 'static + Fn(Input) -> Result<(Output, Residual), ProgramError> + Send + Sync,
        Backward: 'static
            + Fn(
                &mut Transposition,
                Residual,
                Output::To<MaybeZero<Tracer>>,
                Input::To<CotangentAccumulator>,
            ) -> Result<(), DifferentiationError>
            + Send
            + Sync,
    {
        let Self { primal, jvp, batching, non_differentiated_count, name, .. } = self;
        let vjp = WithAccumulatingVjp { forward: Arc::new(forward), backward: Arc::new(backward), marker: PhantomData };
        Self::with_configuration(primal, jvp, vjp, batching, non_differentiated_count, name)
    }
}

impl<Input, Output, Primal, Jvp, Vjp> CustomFunction<Input, Output, Primal, Jvp, Vjp, DefaultBatching> {
    /// Returns this function with the provided custom batching rule `(extent, x, x_axes) ↦ (y, y_axes)`, which batches
    /// its calls instead of structurally batching its primal (i.e., the analogue of JAX's
    /// [`custom_vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_batching.custom_vmap.html)). The rule
    /// receives the extent of the batch axis (i.e., [`BatchingLevelExtent::Static`] for a host extent, or
    /// [`BatchingLevelExtent::Dynamic`] with a first-class extent value), the batched inputs (as values of the context
    /// in which the call is batched), and their [`BatchAxis`]es (i.e., an `Input::To<BatchAxis>` value), and it
    /// returns the batched outputs together with their batch axes (i.e., an `Output::To<BatchAxis>` value). A
    /// replicated input axis means that the input is the same for every batch item. Its closure parameters must
    /// usually be annotated.
    ///
    /// The rule is traced when a call is batched, once per batching level and operand signature, and its program
    /// becomes the batched call's primal, so the rule takes precedence over structurally batching the primal (e.g.,
    /// over the batching strategy of a foreign kernel called by [`Self::from_custom_call`]). Batching preserves the
    /// call and its derivative rules, and the rule must compute the batched primal consistently with them.
    ///
    /// # Derivatives
    ///
    /// How derivatives interact with the rule depends on the forward-mode rule (refer to [`custom_function`] for the
    /// rule selection):
    ///
    ///   - **Derived from the primal** (i.e., without derivative rules, or with [`Self::with_jvp_from_primal`]): the
    ///     rule stays on the path of forward-mode derivatives in either order of the two transforms, which is the
    ///     analogue of JAX's `custom_vmap_jvp`. Differentiating a batched call differentiates the rule's program, and
    ///     differentiating an unbatched call stages a derived call whose batching applies the derivative of the rule,
    ///     including when a forward-mode Jacobian batches it or when a linearized pushforward is batched. The derived
    ///     rule applies the rule with the inputs that are mapped on both their primal and tangent side mapped
    ///     (broadcasting the replicated side of a pair with one mapped side), so the batched derivative has the batch
    ///     axes of the batched call. This deliberately differs from JAX when an input is mapped on only one of its two
    ///     sides while another input is mapped on both: JAX then applies the rule with the primal-side axes and batches
    ///     the tangents separately, which yields the outer product of the two batches, of which only the diagonal
    ///     holds the per-item tangents. For a rule whose result depends on its input axes, the broadcast inputs also
    ///     change the rule's result (e.g., the primal outputs of the batched derivative can differ from those of the
    ///     batched call), whereas JAX's diagonal reflects the primal-side axes. When no input is mapped on both sides
    ///     (e.g., in a forward-mode Jacobian, whose primals are replicated), it applies the rule with every input
    ///     replicated and batches the rule's derivative structurally, so the rule must also accept calls without mapped
    ///     inputs. Linearization computes the outputs with the call itself and recomputes the primal inside the staged
    ///     pushforward, because the call is opaque to partial evaluation, which is only valid for a primal without
    ///     effects: a primal with effects (e.g., one that updates or reads references) is linearized inline, so that
    ///     its effects run once, and batching that linearization's pushforward batches it structurally. Reverse mode
    ///     inlines the derivative of the primal instead, because derived calls are not transposable.
    ///   - **Explicit** (i.e., [`Self::with_jvp`] or reverse-mode rules): differentiating a batched call traces the
    ///     derivative rules at the unbatched types and batches them structurally, aligned to the batch axes that the
    ///     rule declared, and batching a derivative batches the explicit rule structurally, which is JAX's
    ///     `custom_jvp(custom_vmap(f))` nesting order. The other order nests two functions: an outer function with the
    ///     batching rule, whose primal calls an inner function with the derivative rule.
    #[inline]
    pub fn with_batching<Tracer, Rule>(
        self,
        rule: Rule,
    ) -> CustomFunction<Input, Output, Primal, Jvp, Vjp, WithBatching<Tracer, Rule>>
    where
        Tracer: Parameter,
        Input: Parameterized<Tracer, Family: ParameterizedFamily<BatchAxis>>,
        Output: Parameterized<Tracer, Family: ParameterizedFamily<BatchAxis>>,
        Rule: 'static
            + Fn(
                BatchingLevelExtent<Tracer>,
                Input,
                Input::To<BatchAxis>,
            ) -> Result<(Output, Output::To<BatchAxis>), ProgramError>
            + Send
            + Sync,
    {
        let Self { primal, jvp, vjp, non_differentiated_count, name, .. } = self;
        let batching = WithBatching { rule: Arc::new(rule), marker: PhantomData };
        Self::with_configuration(primal, jvp, vjp, batching, non_differentiated_count, name)
    }
}

impl<Input, Output, Primal: Fn(Input) -> Result<Output, ProgramError>, Jvp, Vjp, Batching>
    CustomFunction<Input, Output, Primal, Jvp, Vjp, Batching>
{
    /// Stages one call of this function on the provided `input` value and returns its output value. Refer to the
    /// documentation of the [`custom_function`] function for the tracing semantics and for how the transforms treat
    /// the staged call.
    ///
    /// The [`Context`] `C` whose universe the rules are traced into is the [`DispatchDomain`](Value::DispatchDomain)
    /// of the values in `input`, which is exactly the context the call is staged into. It is therefore never named at
    /// a construction or call site, while the stored closures still pin the tracers that this universe must produce.
    ///
    /// The primal and the rules are traced in fresh traces rather than in `C` itself, so each trace is seeded with the
    /// named axes in scope in `C` (refer to [`NamedAxes::named_axes`]). Code in the closures therefore resolves the
    /// axes of enclosing transforms (e.g., an `axis_index` or a collective over the axis of an enclosing `batch`) as it
    /// would if it were inlined. Rules traced under different bindings may differ, so calls under different named axes
    /// register separate rule definitions.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when `input` has no leaves, when the non-differentiated count exceeds the number of
    /// input leaves, when tracing the primal closure fails, or when the staged [`CustomFunctionOperation`] rejects
    /// the call (e.g., because the call violates the reference contract). The rules are validated when they are first
    /// traced, so their errors are reported by the derivative requests that trace them.
    pub fn call<
        V: Value<Type = C::Type, DispatchDomain = C>,
        C: Context<Type: DifferentiableType + BatchableType + Eq + Hash, Value = V> + NamedAxes,
        InputValues: Parameterized<V, Family = Input::Family, To<C::Type> = Input::To<C::Type>>,
    >(
        &self,
        input: InputValues,
    ) -> Result<<Output::To<C::Type> as Parameterized<C::Type>>::To<V>, ProgramError>
    where
        C::Constant: 'static,
        C::Operation: 'static + From<CustomFunctionOperation<C::Constant, C::Operation>>,
        <C::Type as BatchableType>::Policy: RecursiveBatchingPolicy<TracingContext<C::Constant, C::Operation>>
            + CotangentBatchingPolicy<TracingContext<C::Constant, C::Operation>>,
        CustomRuleRegistration<C::Constant, C::Operation>: Send,
        Jvp: CustomFunctionJvp<C::Constant, C::Operation, Input, Output>,
        Vjp: CustomFunctionVjp<C::Constant, C::Operation, Input, Output>,
        Batching: CustomFunctionBatching<C::Constant, C::Operation, Input, Output>,
        Input: Parameterized<DomainTracer<C>, ParameterStructure: 'static + PartialEq + Send>,
        Input::Family: ParameterizedFamily<C::Type> + ParameterizedFamily<C::Constant>,
        Input::To<C::Type>: Clone + Parameterized<C::Type, Family = Input::Family, To<DomainTracer<C>> = Input>,
        Output: Parameterized<DomainTracer<C>, ParameterStructure: 'static + PartialEq + Send>,
        Output::Family: ParameterizedFamily<C::Type> + ParameterizedFamily<C::Constant> + ParameterizedFamily<V>,
        Output::To<C::Type>: Parameterized<C::Type, Family = Output::Family, To<DomainTracer<C>> = Output>,
    {
        let mut input_values = Vec::new();
        let input_types = input
            .map_parameters(|value| {
                let r#type = value.r#type().into_owned();
                input_values.push(value);
                r#type
            })
            .map_err(ProgramError::from)?;
        let Some(first) = input_values.first() else {
            return Err(TypeError::invalid(format!("`{}` requires at least one input", self.name)).into());
        };
        validate_non_differentiated_count(&self.name, self.non_differentiated_count, input_values.len())?;

        // Only the primal is traced now. The rules are registered with the call structure and traced lazily. The
        // primal and the rules are traced in fresh traces, which are seeded with the named axes that are visible where
        // the function is called (e.g., the axis of an enclosing batching level), so that they resolve the same names
        // as the function would if it were inlined. Rules traced under different bindings may differ, so the named
        // axes are part of the registration key.
        let named_axes = first.dispatch_domain().named_axes();
        let (output_types, primal) =
            DomainTracingContext::<C>::trace_with_named_axes(&self.primal, input_types.clone(), named_axes.clone())?;
        let input_structure = input_types.parameter_structure();
        let output_structure = output_types.parameter_structure();
        let rules = self.registrations.get_or_register(&input_structure, &output_structure, &named_axes, || {
            let definition =
                CustomRuleDefinition::new(self.name.clone()).with_named_axes(named_axes.clone()).with_batching();
            let definition = self.jvp.configure(
                definition,
                &self.name,
                &input_structure,
                &output_structure,
                self.non_differentiated_count,
            );
            let definition = self.vjp.configure(
                definition,
                &self.name,
                &input_structure,
                &output_structure,
                self.non_differentiated_count,
            );

            // Every transform without a configured rule treats the primal as it treats any other function, so a
            // function without derivative rules differentiates its primal. A function with only reverse-mode rules
            // keeps rejecting forward mode instead, because a derived pushforward would contradict its pullback.
            let definition = match definition.has_derivative_rules() {
                true => definition,
                false => definition.with_jvp_from_primal(),
            };
            self.batching.configure(definition, &self.name, &input_structure, &output_structure)
        });

        // The call binds through whatever context the input values flow through (e.g., a staging trace, a batching
        // context, or a differentiation context), so the batching or differentiation rule of the bound operation fires
        // and the function composes with those transforms.
        let operation =
            CustomFunctionOperation::new(rules).with_non_differentiated_count(self.non_differentiated_count)?;
        let outputs =
            first.dispatch_domain().bind(operation, vec![primal.into_flat_program()], input_values.as_slice())?;
        Ok(Parameterized::from_parameters(output_structure, outputs)?)
    }
}

/// Creates a [`CustomFunction`] from a primal closure `x ↦ y = f(x)` over values of [`DomainTracer`]s, without any
/// custom rule. It is the one entry point for custom rules of every transform, and its configuration functions add
/// them, where each configuration is the analogue of one JAX entry point: [`CustomFunction::with_jvp`] of
/// [`jax.custom_jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html), [`CustomFunction::with_vjp`] of
/// [`jax.custom_vjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html), and
/// [`CustomFunction::with_batching`] of
/// [`jax.custom_batching.custom_vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_batching.custom_vmap.html),
/// and they combine in one registration:
///
///   - [`CustomFunction::with_jvp`] adds a Jacobian-Vector Product (JVP) rule `(x, ẋ) ↦ (y, ẏ)`, and
///     [`CustomFunction::with_symbolic_zero_jvp`] adds one that receives structural-zero tangents,
///   - [`CustomFunction::with_jvp_from_primal`] derives the forward-mode rule from the primal instead,
///   - [`CustomFunction::with_vjp`] adds reverse-mode forward and backward rules `x ↦ (y, r)` and `(r, ȳ) ↦ x̄`,
///     [`CustomFunction::with_symbolic_zero_vjp`] adds ones whose backward rule receives structural-zero seeds, and
///     [`CustomFunction::with_accumulating_vjp`] adds ones whose backward rule submits its cotangents to their
///     destinations (e.g., caller-provided buffers) instead of returning them, and
///   - [`CustomFunction::with_batching`] adds a custom batching rule, which batches calls instead of structurally
///     batching their primal.
///
/// [`CustomFunction::from_custom_call`] creates a function whose primal calls a foreign kernel. Every transform
/// without a configured rule treats the primal as it treats any other function, so a function without derivative
/// rules (e.g., one with only a batching rule, which is JAX's bare `custom_vmap`) differentiates its primal. The one
/// exception is a function with only reverse-mode rules, which rejects forward mode (as JAX's `custom_vjp` does),
/// because a derived pushforward would contradict its custom pullback. Each derivative mode selects exactly one rule:
///
/// | Configuration                          | Forward mode   | Reverse mode                          | Batching      |
/// |----------------------------------------|----------------|---------------------------------------|---------------|
/// | primal only                            | derived JVP    | transposed derived linearization      | structural    |
/// | `with_batching`                        | derived JVP    | transposed derived linearization      | batching rule |
/// | `with_jvp_from_primal`                 | derived JVP    | transposed derived linearization      | structural    |
/// | `with_jvp`                             | JVP rule       | transposed JVP rule linearization     | structural    |
/// | `with_vjp`                             | rejected       | forward and backward rules            | structural    |
/// | `with_jvp` and `with_vjp`              | JVP rule       | forward and backward rules            | structural    |
/// | `with_jvp_from_primal` and `with_vjp`  | derived JVP    | forward and backward rules            | structural    |
///
/// Any of the forms of a forward-mode or reverse-mode rule listed above takes the place of `with_jvp` or `with_vjp` in
/// this table, respectively, and `with_batching` combines with every configuration, replacing structural batching. The
/// user is responsible for the numerical consistency of the rules (e.g., that a forward rule computes the same outputs
/// as the primal, that a JVP rule and reverse-mode rules describe the same derivative, or that a batching rule computes
/// the batched primal). Higher-order differentiation differentiates the selected rule programs rather than switching
/// back to the primal.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::{
/// #     Array, ArrayOperation, BatchAxis, BatchingLevelExtent, Cos, DomainTracer, EagerContext, ProgramError, Sin,
/// #     batch, custom_function, differentiate_at,
/// # };
/// # fn main() -> Result<(), ProgramError> {
/// type Tracer = DomainTracer<EagerContext<Array, ArrayOperation<Array>>>;
///
/// // `sin` with a JVP rule for forward mode and reverse-mode rules that save `cos(x)` as their residual.
/// let sine = custom_function(|x: Tracer| Ok(x.sin()?))
///     .with_jvp(|x, tangent| Ok((x.sin()?, x.cos()? * tangent)))
///     .with_vjp(|x| Ok((x.sin()?, x.cos()?)), |cosine, cotangent| Ok(cosine * cotangent));
/// let (_, tangent) = differentiate_at(Array::scalar(0.5f64)?).jvp(Array::scalar(1.0f64)?, |x| sine.call(x))?;
/// assert_eq!(tangent, Array::scalar(0.5f64.cos())?);
/// let gradient = differentiate_at(Array::scalar(0.5f64)?).gradient(|x| sine.call(x))?;
/// assert_eq!(gradient, Array::scalar(0.5f64.cos())?);
///
/// // A custom batching rule receives the batched input and its batch axis, and it declares the batch axis of the
/// // output (i.e., the output is mapped exactly when the input is).
/// let square = custom_function(|x: Tracer| Ok(x.clone() * x))
///     .with_batching(|_: BatchingLevelExtent<Tracer>, x: Tracer, axis: BatchAxis| Ok((x.clone() * x, axis)));
/// let squares: Array = batch(
///     |x| square.call(x),
///     Array::vector(vec![1.0f64, 2.0, 3.0])?,
///     BatchAxis::new(0),
///     BatchAxis::new(0),
///     None,
/// )?;
/// assert_eq!(squares, Array::vector(vec![1.0f64, 4.0, 9.0])?);
/// # Ok(())
/// # }
/// ```
///
/// # Calling Convention
///
/// All closures operate on [`Parameterized`] values of [`DomainTracer`]s (i.e., Ryft's analogue of JAX pytrees), so
/// inputs, outputs, and residuals may each be a single tracer, a tuple, or any other parameterized structure. Tangents
/// and cotangents have the parameter structures of their primal values. Because this function builds a reusable
/// function before any input is known, the primal closure must annotate the tracer type of its input (e.g.,
/// `|x: DomainTracer<C>| ...`), which then also fixes the input types of the rule closures. Static configuration
/// should be captured by the closures, while a dynamic value should remain an explicit input, either as a
/// non-differentiated input (see below) or as an ordinary input whose derivative the rules treat as zero.
///
/// # Non-Differentiated Inputs
///
/// [`CustomFunction::with_non_differentiated_count`] declares the leading flattened input leaves as _plumbing_ that
/// parameterizes the call without being differentiated. These leaves remain dynamic inputs. Unlike JAX's
/// `nondiff_argnums`, this is a prefix of flattened leaves, not a set of static argument positions. Plumbing leaves
/// reach every closure at their usual positions, and the rules keep signatures that mirror the primal signature: the
/// tangent leaves that a JVP rule receives for them are placeholders that it must not use, and the cotangent leaves
/// that a backward rule returns for them are ignored (refer to [`CustomFunction::with_jvp`] and
/// [`CustomFunction::with_vjp`]). Differentiating a call with a non-zero tangent for a numeric plumbing input is
/// rejected because the rules cannot propagate that tangent.
///
/// # References
///
/// A reference-typed input is accepted only as plumbing, where every closure that receives it may read or write it.
/// An active reference input is rejected, because the rule interfaces define no tangent or cotangent reference for it
/// and so user-supplied rules could not express its derivative. A live tangent reference supplied for a plumbing input
/// by an enclosing transform is left untouched by the call, since the rules declare no derivative through the state it
/// denotes. No output may be a reference, because the rules would then have to produce that output's tangent
/// reference or consume its cotangent reference. The closures may also allocate and use local reference state, which
/// executes like any other primitive operation whenever the corresponding program is replayed. When the call is
/// differentiated, no two reference inputs may bind the same allocation.
///
/// # Tracing Semantics
///
/// Nothing is traced at construction time. Each [`CustomFunction::call`] recovers the tracing [`Context`] from the
/// values that it is called with, traces only the primal closure at their types, and stages one
/// [`CustomFunctionOperation`] that retains the rules into the context through which those values flow. The rules are
/// traced lazily, at the types of each specialization (e.g., each distinct input signature): forward mode traces the
/// JVP rule, reverse mode traces the forward rule when it stages its reverse-mode carrier and the backward rule when
/// that carrier is transposed, and batching traces the batching rule. Tracing a rule also validates its signature, so
/// a malformed rule is reported by the request that first needs it rather than by the call. The traced programs are
/// cached with the function, and a program that is never differentiated never traces its derivative rules (refer to
/// [`CustomFunction`] for the ownership of that cache).
///
/// # Transform Semantics
///
/// The transforms treat a staged call as follows:
///
///   - _interpretation_ and backend lowering replay the lean primal program only,
///   - _partial evaluation_ folds a call when its inputs are known and effect-ordering constraints permit execution,
///     and it otherwise preserves the call and its retained rules for later differentiation,
///   - _batching_ batches the primal (with the batching rule, if there is one), preserves the call around the batched
///     primal program, and records the batching level, and a later derivative request batches the derivative rule
///     programs traced at the unbatched types, aligned to the batch axes of the batched call's outputs, so that the
///     custom derivative survives batching applied _before_ differentiation, while a batching rule survives batching
///     applied _after_ forward-mode differentiation (refer to [`CustomFunction::with_batching`]), and
///   - _differentiation_ replays the selected derivative rule programs instead of differentiating the primal (refer
///     to [`CustomFunction::with_jvp`] and [`CustomFunction::with_vjp`] for the semantics of each mode, including
///     higher-order differentiation).
///
/// # Parameters
///
///   - `primal`: Closure implementing `f(x) = y`.
#[inline]
pub fn custom_function<Input, Output, Primal: Fn(Input) -> Result<Output, ProgramError>>(
    primal: Primal,
) -> CustomFunction<Input, Output, Primal> {
    CustomFunction {
        primal,
        jvp: DefaultJvp,
        vjp: DefaultVjp,
        batching: DefaultBatching,
        non_differentiated_count: 0,
        name: Cow::Borrowed(CUSTOM_FUNCTION_OPERATION_NAME),
        registrations: CustomFunctionRegistrations::default(),
        marker: PhantomData,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReference, ArrayReferenceTransform,
        ArraySliceAxis, ArrayType, DataType, DimensionBounds, DimensionType, DimensionValue, ShardingDimension,
    };
    use crate::axes::{AxisError, AxisIndex};
    use crate::batching::{
        BatchAxis, BatchAxisSpecification, BatchedProgram, BatchingError, ProgramBatchingOutputAxesPolicy, batch,
    };
    use crate::contexts::{Context, EagerContext, ProjectedContext};
    use crate::differentiation::{
        CotangentDestination, CotangentDestinationKind, CotangentSeed, DifferentiationRule, differentiate_at,
    };
    use crate::operations::arithmetic::{AddOperation, MulOperation};
    use crate::operations::collectives::parallel_reduce::{ParallelReduce, ParallelReductionKind};
    use crate::operations::constants::zero::Zero;
    use crate::operations::custom_call::CustomCallBatching;
    use crate::operations::custom_functions::rules::CustomRuleSource;
    use crate::operations::dimensions::dimension_size::DimensionSize;
    use crate::operations::dot::{Dot, DotDimensionNumbers};
    use crate::operations::manipulation::broadcasting::DynamicBroadcast;
    use crate::operations::manipulation::conversions::ConvertElementType;
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

    use super::*;

    /// Eager context whose values are arrays.
    /// Eager context whose values are arrays.
    type ArrayContext = EagerContext<Array, ArrayOperation<Array>>;

    /// Eager composite context whose values may be arrays or references.
    type EagerArrayIrContext = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Tracer of [`EagerArrayIrContext`] values.
    type ArrayIrTracer = DomainTracer<EagerArrayIrContext>;

    #[test]
    fn test_custom_function_from_custom_call() {
        // The primal calls the foreign kernel `my_sin`, which the reference backend cannot execute, while the rules
        // are ordinary programs, so both differentiation modes execute without calling the kernel.
        type Tracer = DomainTracer<ArrayContext>;
        let scalar_type = ArrayType::scalar(DataType::F64);
        let function = CustomFunction::from_custom_call(CustomCallOperation::new("my_sin", vec![scalar_type.clone()]))
            .with_jvp(|inputs: Vec<Tracer>, tangents: Vec<Tracer>| {
                Ok((vec![inputs[0].sin()?], vec![inputs[0].cos()? * tangents[0].clone()]))
            })
            .with_vjp(
                |inputs: Vec<Tracer>| Ok((vec![inputs[0].sin()?], inputs[0].cos()?)),
                |cosine, cotangents: Vec<Tracer>| Ok(vec![cosine * cotangents[0].clone()]),
            );
        let (_, program) = ArrayContext::trace(|inputs| function.call(inputs), vec![scalar_type]).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = custom_function [name=\"my_sin\"] %0 [
                    primal={
                        lambda %0:f64[] .
                        let %1:f64[] = custom_call [target=my_sin] %0
                        in (%1)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.interpret(vec![Array::scalar(2.0).unwrap()]),
            Err(ProgramError::UnsupportedOperation {
                message: "the reference array backend cannot execute the foreign kernel `my_sin`".to_string(),
            }),
        );
        assert_eq!(
            differentiate_at(vec![Array::scalar(2.0).unwrap()])
                .jvp(vec![Array::scalar(1.0).unwrap()], |inputs| function.call(inputs)),
            Ok((vec![Array::scalar(2.0f64.sin()).unwrap()], vec![Array::scalar(2.0f64.cos()).unwrap()])),
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap())
                .gradient(|x| function.call(vec![x]).map(|outputs| { outputs.into_iter().next().unwrap() })),
            Ok(Array::scalar(2.0f64.cos()).unwrap()),
        );
    }

    #[test]
    fn test_custom_function_from_custom_call_batching() {
        // Batching a function built from a foreign kernel defers to the kernel's own strategy unless the function has a
        // custom batching rule, which takes precedence and here calls a batched kernel instead.
        type Tracer = DomainTracer<ArrayContext>;
        let scalar_type = ArrayType::scalar(DataType::F64);
        let axes = [BatchAxis::new(0)];
        let natural = ProgramBatchingOutputAxesPolicy::Natural;
        let operation = CustomCallOperation::new("my_sin", vec![scalar_type.clone()]);
        let jvp = |inputs: Vec<Tracer>, tangents: Vec<Tracer>| {
            Ok((vec![inputs[0].sin()?], vec![inputs[0].cos()? * tangents[0].clone()]))
        };

        // The kernel rejects batching by default.
        let function = CustomFunction::from_custom_call(operation.clone()).with_jvp(jvp);
        let (_, program) = ArrayContext::trace(|inputs| function.call(inputs), vec![scalar_type.clone()]).unwrap();
        assert!(matches!(
            program.into_flat_program().batched(3, ShardingDimension::Replicated, &axes, natural.clone()),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "custom call `my_sin` has no batching rule for operand 0 mapped at batch axis 0; invoke \
                               a kernel that understands the batch axis, or select an explicit batching behavior with \
                               `CustomCallOperation::with_batching`",
        ));

        // Its broadcasting strategy applies when the function has no batching rule.
        let function =
            CustomFunction::from_custom_call(operation.clone().with_batching(CustomCallBatching::BroadcastAll))
                .with_jvp(jvp);
        let (_, program) = ArrayContext::trace(|inputs| function.call(inputs), vec![scalar_type.clone()]).unwrap();
        let batched = program
            .into_flat_program()
            .batched(3, ShardingDimension::Replicated, &axes, natural.clone())
            .unwrap();
        assert_eq!(
            batched.into_parts().0.to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[3] = custom_function [name=\"my_sin\", batching=[(extent=3, input_axes=[axis 0], output_axes=[axis 0])]] %0 [
                    primal={
                        lambda %0:f64[3] .
                        let %1:f64[3] = custom_call [target=my_sin, batching=broadcast_all] %0
                        in (%1)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );

        // A batching rule takes precedence over the kernel's strategy, including under nested batching, and the batched
        // call keeps its JVP rule.
        let function = CustomFunction::from_custom_call(operation).with_jvp(jvp).with_batching(
            |_: BatchingLevelExtent<Tracer>, inputs: Vec<Tracer>, axes: Vec<BatchAxis>| {
                let operation = CustomCallOperation::new("my_batched_sin", vec![inputs[0].r#type().into_owned()]);
                Ok((CustomCall::custom_call(&operation, &inputs)?, axes))
            },
        );
        let (_, program) = ArrayContext::trace(|inputs| function.call(inputs), vec![scalar_type]).unwrap();
        let batched = program
            .into_flat_program()
            .batched(3, ShardingDimension::Replicated, &axes, natural.clone())
            .unwrap();
        let nested = batched.into_parts().0.batched(2, ShardingDimension::Replicated, &axes, natural).unwrap();
        let nested = nested.into_parts().0;
        assert_eq!(
            nested.to_string(),
            indoc! {"
                lambda %0:f64[2, 3] .
                let %1:f64[2, 3] = custom_function [
                    name=\"my_sin\",
                    batching=[(extent=3, input_axes=[axis 0], output_axes=[axis 0]), (extent=2, input_axes=[axis 0], output_axes=[axis 0])],
                ] %0 [
                    primal={
                        lambda %0:f64[2, 3] .
                        let %1:f64[2, 3] = custom_call [target=my_batched_sin] %0
                        in (%1)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(
            nested.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[2, 3], %1:f64[2, 3] .
                let %2:f64[2, 3] = sin %0
                    %3:f64[2, 3] = cos %0
                    %4:f64[2, 3] = mul %3 %1
                in (%2, %4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_function_from_custom_call_without_rules() {
        // Without derivative rules, differentiation differentiates the primal, which reaches the opaque custom call and
        // reports its own diagnostic in both modes.
        type Tracer = DomainTracer<ArrayContext>;
        let scalar_type = ArrayType::scalar(DataType::F64);
        let function: CustomFunction<Vec<Tracer>, Vec<Tracer>, CustomCallPrimal<Tracer>> =
            CustomFunction::from_custom_call(CustomCallOperation::new("my_sin", vec![scalar_type]));
        let rejection = "custom call `my_sin` has no differentiation rule; call it through a `custom_function` with \
                         derivative rules to provide one";
        assert!(matches!(
            differentiate_at(vec![Array::scalar(2.0).unwrap()])
                .jvp(vec![Array::scalar(1.0).unwrap()], |inputs| function.call(inputs)),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message })) if message == rejection,
        ));
        assert!(matches!(
            differentiate_at(Array::scalar(2.0).unwrap())
                .gradient(|x| function.call(vec![x]).map(|outputs| outputs.into_iter().next().unwrap())),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message })) if message == rejection,
        ));
    }

    #[test]
    fn test_custom_function_with_name() {
        // Without a name, calls are labeled by the operation name. Renaming a function that was already called must not
        // reuse the definition registered by that call, which carries the previous name. A function without derivative
        // rules rejects both differentiation modes and names itself in the diagnostic.
        let function = custom_function(|x: DomainTracer<ArrayContext>| Ok(x.sin()?));
        let (_, program) = ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64)).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = custom_function [name=\"custom_function\"] %0 [
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
        let function = function.with_name("sine");
        let (_, program) = ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64)).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = custom_function [name=\"sine\"] %0 [
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

        // The name also labels the diagnostics of the call (e.g., the forward-mode rejection of reverse-only rules).
        let function = function.with_vjp(|x| Ok((x.sin()?, x.cos()?)), |cosine, cotangent| Ok(cosine * cotangent));
        assert!(matches!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| function.call(x)),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "cannot apply forward-mode differentiation to a `custom_function` call of `sine` that \
                               has only reverse-mode rules; it supports only reverse-mode differentiation (e.g., \
                               `vjp`, `value_and_gradient`, or `jacobian_reverse`)",
        ));
    }

    #[test]
    fn test_custom_function_with_jvp() {
        // An explicit JVP rule governs forward mode and, by transposition, reverse mode. The rule doubles the true
        // derivative (expressed through addition to avoid constant lifting), which proves that the rule is in control.
        let function = custom_function(|x: DomainTracer<ArrayContext>| Ok(x.sin()?)).with_jvp(|x, tangent| {
            let tangent = x.cos()? * tangent;
            Ok((x.sin()?, tangent.clone() + tangent))
        });
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| function.call(x)),
            Ok((Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(2.0 * 2.0f64.cos()).unwrap())),
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).gradient(|x| function.call(x)),
            Ok(Array::scalar(2.0 * 2.0f64.cos()).unwrap()),
        );
    }

    #[test]
    fn test_custom_function_with_jvp_zero_space_inputs() {
        // The tangent of an integer input belongs to a zero differential space, so it is always a structural zero,
        // which the specialization materializes for a rule that receives materialized tangents.
        type Tracer = DomainTracer<ArrayContext>;
        let function = custom_function(|(x, _): (Tracer, Tracer)| Ok(x.sin()?)).with_jvp(
            |(x, _): (Tracer, Tracer), (tangent, count_tangent): (Tracer, Tracer)| {
                assert_eq!(count_tangent.r#type().into_owned(), ArrayType::scalar(DataType::Zero));
                Ok((x.sin()?, x.cos()? * tangent))
            },
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| {
                let count = x.context().lift(Array::scalar(3i32).unwrap())?;
                function.call((x, count))
            }),
            Ok((Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(2.0f64.cos()).unwrap())),
        );
    }

    #[test]
    fn test_custom_function_with_jvp_non_differentiated_inputs() {
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));

        // The leading counter is plumbing: it reaches both closures at its usual position, and the rule closure keeps
        // receiving a full tangent value whose counter leaf is a placeholder that it leaves unused.
        let function =
            custom_function(|(counter, x): (DomainTracer<EagerArrayIrContext>, DomainTracer<EagerArrayIrContext>)| {
                counter.add_update(&x)?;
                Ok(x)
            })
            .with_jvp(|(counter, x), (_, tangent)| {
                counter.add_update(&x)?;
                Ok((x, tangent))
            })
            .with_non_differentiated_count(1);
        let counter = ArrayReference::new(Array::scalar(0.0f32).unwrap());
        let counter_tangent = ArrayReference::new(Array::scalar(0.0f32).unwrap());
        assert_eq!(
            differentiate_at((
                ArrayIrValue::Reference(counter.clone()),
                ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()),
            ))
            .jvp(
                (
                    ArrayIrValue::Reference(counter_tangent.clone()),
                    ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
                ),
                |(counter, x)| function.call((counter, x)),
            ),
            Ok((
                ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
            )),
        );

        // Forward mode replays only the rule region, which increments the counter once and never touches its tangent.
        assert_eq!(counter.read(), Ok(Array::scalar(2.0f32).unwrap()));
        assert_eq!(counter_tangent.read(), Ok(Array::scalar(0.0f32).unwrap()));

        // A rule that uses the placeholder tangent of a plumbing input is rejected when it is first traced (i.e., by
        // the first derivative request), because the staged rule has no tangent slot for it.
        let function =
            custom_function(|(counter, x): (DomainTracer<EagerArrayIrContext>, DomainTracer<EagerArrayIrContext>)| {
                counter.add_update(&x)?;
                Ok(x)
            })
            .with_jvp(|(_, x), (counter_tangent, tangent)| {
                counter_tangent.add_update(&tangent)?;
                Ok((x, tangent))
            })
            .with_non_differentiated_count(1);
        assert_eq!(
            EagerArrayIrContext::trace(
                |(counter, x)| function.call((counter, x)),
                (reference_type.clone(), scalar_type.clone()),
            )
            .and_then(|(_, program)| {
                program.into_flat_program().jvp_with_respect_to(&[1]).map(|_| ()).map_err(ProgramError::from)
            }),
            Err(ProgramError::Type(TypeError::invalid(
                "`custom_function` rule uses the tangent of leading non-differentiated input 0, which has no tangent \
                 slot because non-differentiated inputs parameterize the rule without being differentiated"
                    .to_string(),
            ))),
        );

        // Returning the placeholder is also a use, even when no instruction consumes it.
        let function =
            custom_function(|(parameter, _): (DomainTracer<ArrayContext>, DomainTracer<ArrayContext>)| Ok(parameter))
                .with_jvp(|(parameter, _), (placeholder, _)| Ok((parameter, placeholder)))
                .with_non_differentiated_count(1);
        assert_eq!(
            ArrayContext::trace(
                |inputs| function.call(inputs),
                (ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)),
            )
            .and_then(|(_, program)| {
                program.into_flat_program().jvp_with_respect_to(&[1]).map(|_| ()).map_err(ProgramError::from)
            }),
            Err(ProgramError::Type(TypeError::invalid(
                "`custom_function` rule uses the tangent of leading non-differentiated input 0, which has no tangent \
                 slot because non-differentiated inputs parameterize the rule without being differentiated"
                    .to_string(),
            ))),
        );

        // Without the declaration, the reference is an active input, which the staged operation rejects.
        let function =
            custom_function(|(counter, x): (DomainTracer<EagerArrayIrContext>, DomainTracer<EagerArrayIrContext>)| {
                counter.add_update(&x)?;
                Ok(x)
            })
            .with_jvp(|(_, x), (_, tangent)| Ok((x, tangent)));
        assert_eq!(
            EagerArrayIrContext::trace(|(counter, x)| function.call((counter, x)), (reference_type, scalar_type))
                .map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "`custom_function` accepts reference inputs only in its leading non-differentiated segment; move \
                 input 0 of type `ref<f32[]>` before the differentiated inputs"
                    .to_string(),
            ))),
        );

        // The non-differentiated count cannot exceed the number of input leaves.
        let function = custom_function(|x: DomainTracer<ArrayContext>| Ok(x.sin()?))
            .with_jvp(|x, tangent| Ok((x.sin()?, x.cos()? * tangent)))
            .with_non_differentiated_count(2);
        assert_eq!(
            ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64)).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "`custom_function` non-differentiated input count 2 exceeds input count 1".to_string(),
            ))),
        );
    }

    #[test]
    fn test_custom_function_with_jvp_call() {
        // The wrapper traces the closures at the call site, specialized to the input types. The custom rule
        // `jvp(x, ẋ) = (sin(x), cos(x) * ẋ + cos(x) * ẋ)` doubles the mathematical derivative (expressed through
        // addition to avoid constant lifting), which proves that the rule is in control.
        let function = custom_function(|x: DomainTracer<ArrayContext>| Ok(x.sin()?)).with_jvp(|x, tangent| {
            let tangent = x.cos()? * tangent;
            Ok((x.sin()?, tangent.clone() + tangent))
        });
        let (_, program) = ArrayContext::trace(|input| function.call(input), ArrayType::scalar(DataType::F64)).unwrap();
        assert_eq!(program.interpret(Array::scalar(2.0).unwrap()), Ok(Array::scalar(2.0f64.sin()).unwrap()));
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| function.call(x)),
            Ok((Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(2.0 * 2.0f64.cos()).unwrap())),
        );

        // Reverse mode transposes the linearized custom rule, so the doubled derivative carries over.
        assert_eq!(
            differentiate_at(Array::scalar(3.0).unwrap()).value_and_gradient(|x| function.call(x).unwrap()),
            Ok((Array::scalar(3.0f64.sin()).unwrap(), Array::scalar(2.0 * 3.0f64.cos()).unwrap())),
        );
    }

    #[test]
    fn test_custom_function_with_jvp_structured_outputs() {
        // Distinct tuple leaves make flattening and reconstruction order observable.
        let function = custom_function(|x: DomainTracer<ArrayContext>| Ok((x.sin()?, vec![x.cos()?, x]))).with_jvp(
            |x, tangent| Ok(((x.sin()?, vec![x.cos()?, x]), (tangent.clone(), vec![tangent.clone(), tangent]))),
        );
        let (_, program) = ArrayContext::trace(|input| function.call(input), ArrayType::scalar(DataType::F64)).unwrap();
        assert_eq!(
            program.interpret(Array::scalar(2.0).unwrap()),
            Ok((
                Array::scalar(2.0f64.sin()).unwrap(),
                vec![Array::scalar(2.0f64.cos()).unwrap(), Array::scalar(2.0).unwrap()]
            )),
        );
    }

    #[test]
    fn test_custom_function_with_jvp_tracing_errors() {
        // Empty collections cannot identify the context in which the call should execute.
        let function = custom_function(|inputs: Vec<DomainTracer<ArrayContext>>| Ok(inputs))
            .with_jvp(|inputs, tangents| Ok((inputs, tangents)));
        assert_eq!(
            ArrayContext::trace(|inputs| function.call(inputs), Vec::<ArrayType>::new()).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid("`custom_function` requires at least one input".to_string()))),
        );

        // Both closures propagate their original error without replacing it with a signature error.
        let function =
            custom_function(|_: DomainTracer<ArrayContext>| -> Result<DomainTracer<ArrayContext>, ProgramError> {
                Err(ProgramError::InvalidArgument { message: "primal tracing failed".to_string() })
            })
            .with_jvp(|input, tangent| Ok((input, tangent)));
        assert_eq!(
            ArrayContext::trace(|input| function.call(input), ArrayType::scalar(DataType::F64)).map(|_| ()),
            Err(ProgramError::InvalidArgument { message: "primal tracing failed".to_string() }),
        );
        let function = custom_function(|input: DomainTracer<ArrayContext>| Ok(input))
            .with_jvp(|_, _| Err(ProgramError::InvalidArgument { message: "jvp tracing failed".to_string() }));
        assert_eq!(
            ArrayContext::trace(|input| function.call(input), ArrayType::scalar(DataType::F64))
                .and_then(|(_, program)| program.into_flat_program().jvp().map(|_| ()).map_err(ProgramError::from)),
            Err(ProgramError::InvalidArgument { message: "jvp tracing failed".to_string() }),
        );
    }

    #[test]
    fn test_custom_function_with_jvp_rule_signature_mismatch() {
        // Fixed tuple structures are checked statically, but array shapes and collection lengths require runtime
        // validation. This rule produces a scalar tangent for a vector output, which fails signature validation when
        // the rule is first traced (i.e., by the first derivative request) rather than when the function is called.
        let function = custom_function(|x: DomainTracer<ArrayContext>| Ok(x.sin()?))
            .with_jvp(|x, tangent| Ok((x.sin()?, tangent.dot(&tangent, &DotDimensionNumbers::inner_product())?)));
        let (_, program) =
            ArrayContext::trace(|x| function.call(x), ArrayType::new_static(DataType::F64, [2])).unwrap();
        assert_eq!(
            program.into_flat_program().jvp().map(|_| ()).map_err(ProgramError::from),
            Err(ProgramError::Type(TypeError::invalid(
                "`custom_function` `custom_function` JVP rule output type signature mismatch: expected [f64[2], \
                 f64[2]] but got [f64[2], f64[]]"
                    .to_string(),
            ))),
        );

        // Equal flattened leaf types cannot distinguish different nested vector lengths.
        let function = custom_function(|input: DomainTracer<ArrayContext>| {
            Ok(vec![vec![input.clone(), input.clone()], vec![input]])
        })
        .with_jvp(|input, tangent| {
            Ok((
                vec![vec![input.clone()], vec![input.clone(), input]],
                vec![vec![tangent.clone()], vec![tangent.clone(), tangent]],
            ))
        });
        assert_eq!(
            ArrayContext::trace(|input| function.call(input), ArrayType::scalar(DataType::F64))
                .and_then(|(_, program)| program.into_flat_program().jvp().map(|_| ()).map_err(ProgramError::from)),
            Err(ProgramError::Parameter(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{:?}", vec![vec![Placeholder; 2], vec![Placeholder]]),
                right_structure: format!("{:?}", vec![vec![Placeholder], vec![Placeholder; 2]]),
            })),
        );

        // The primal half may agree while the tangent half has the wrong structure.
        let function = custom_function(|input: DomainTracer<ArrayContext>| {
            Ok(vec![vec![input.clone(), input.clone()], vec![input]])
        })
        .with_jvp(|input, tangent| {
            Ok((
                vec![vec![input.clone(), input.clone()], vec![input]],
                vec![vec![tangent.clone()], vec![tangent.clone(), tangent]],
            ))
        });
        assert_eq!(
            ArrayContext::trace(|input| function.call(input), ArrayType::scalar(DataType::F64))
                .and_then(|(_, program)| program.into_flat_program().jvp().map(|_| ()).map_err(ProgramError::from)),
            Err(ProgramError::Parameter(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{:?}", vec![vec![Placeholder; 2], vec![Placeholder]]),
                right_structure: format!("{:?}", vec![vec![Placeholder], vec![Placeholder; 2]]),
            })),
        );
    }

    #[test]
    fn test_custom_function_with_jvp_computed_dimension_outputs() {
        // The primal broadcasts `x` to the computed extent `n · n`, and the JVP rule recomputes that extent in its own
        // trace, which mints a fresh identity for it. The rule outputs therefore agree with the primal outputs only up
        // to a renaming of that identity, which is applied to the rule program.
        let broadcast = |dimension: &ArrayIrTracer, x: ArrayIrTracer| -> Result<_, ProgramError> {
            let dimension = ValueProjection::<DimensionType>::into_projected(dimension.clone())?;
            let extent = (dimension.clone() * dimension).into_value();
            Ok((extent.clone(), x.dynamic_broadcast(&[extent], &[])?))
        };
        let function = |shares_extent: bool| {
            custom_function(move |(dimension, x): (ArrayIrTracer, ArrayIrTracer)| broadcast(&dimension, x))
                .with_non_differentiated_count(1)
                .with_jvp(
                    move |(dimension, x): (ArrayIrTracer, ArrayIrTracer),
                          (_, tangent): (ArrayIrTracer, ArrayIrTracer)| {
                        let (extent, output) = broadcast(&dimension, x)?;
                        let extent_tangent = extent.context().zero(&extent.r#type().tangent()?)?;
                        let output_tangent = match shares_extent {
                            true => tangent.dynamic_broadcast(&[extent.clone()], &[])?,
                            false => broadcast(&dimension, tangent)?.1,
                        };
                        Ok(((extent, output), (extent_tangent, output_tangent)))
                    },
                )
        };
        let dimension = DimensionType::new("n", DimensionBounds::new(2, Some(5)).unwrap());
        let input_types = (ArrayIrType::from(dimension.clone()), ArrayIrType::from(ArrayType::scalar(DataType::F64)));
        let jvp = |shares_extent: bool| {
            let function = function(shares_extent);
            let (_, program) = EagerArrayIrContext::trace(|inputs| function.call(inputs), input_types.clone())?;
            program.into_flat_program().jvp_with_respect_to(&[1]).map_err(ProgramError::from)
        };
        let program = jvp(true).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<n ∈ [2, 5)>, %1:f64[], %2:f64[] .
                let %3:dimension<n * n ∈ [4, 17)> = dimension_mul %0 %0
                    %4:f64[n * n] = broadcast [output_axes=[]] %1 %3
                    %5:zero[] = zero [type=zero[]]
                    %6:f64[n * n] = broadcast [output_axes=[]] %2 %3
                in (%3, %4, %6)
            "}
            .trim_end(),
        );
        let outputs = program
            .interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::new(dimension, 2).unwrap()),
                ArrayIrValue::Array(Array::scalar(3.0f64).unwrap()),
                ArrayIrValue::Array(Array::scalar(1.0f64).unwrap()),
            ])
            .unwrap();
        assert!(matches!(&outputs[0], ArrayIrValue::Dimension(extent) if extent.extent() == 4));
        assert_eq!(
            outputs[1..],
            [
                ArrayIrValue::Array(Array::vector(vec![3.0f64; 4]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![1.0f64; 4]).unwrap()),
            ],
        );

        // A rule that computes the extent of its output tangent separately establishes a second identity for it,
        // which the primal output and its tangent share, so its outputs are rejected.
        assert_eq!(
            jvp(false).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "`custom_function` `custom_function` JVP rule output type signature mismatch: expected \
                 [dimension<n * n ∈ [4, 17)>, f64[n * n], zero[], f64[n * n]] but got [dimension<n * n ∈ [4, 17)>, \
                 f64[n * n], zero[], f64[n * n]] with different type identities"
                    .to_string(),
            ))),
        );
    }

    #[test]
    fn test_custom_function_with_jvp_zero_space_boundaries() {
        // Token primals and zero-space tangents carry no payload, so the wrapper must pass them through the traced
        // rule unchanged instead of demanding a dense tangent space.
        let token = Array::from_logical_bytes(ArrayType::scalar(DataType::Token), &[]).unwrap();
        let zero = Array::from_logical_bytes(ArrayType::scalar(DataType::Zero), &[]).unwrap();
        let function = custom_function(|token: DomainTracer<ArrayContext>| Ok(token))
            .with_jvp(|token, tangent| Ok((token, tangent)));
        assert_eq!(differentiate_at(token.clone()).jvp(zero.clone(), |token| function.call(token)), Ok((token, zero)));
    }

    #[test]
    fn test_custom_function_with_jvp_and_vjp() {
        // With both rule kinds, forward mode uses the JVP rule (which doubles the derivative) and reverse mode uses the
        // reverse-mode rules (which triple it), whatever the order of the configuration calls.
        let doubled_jvp = |x: DomainTracer<ArrayContext>, tangent: DomainTracer<ArrayContext>| {
            let tangent = x.cos()? * tangent;
            Ok((x.sin()?, tangent.clone() + tangent))
        };
        let forward = |x: DomainTracer<ArrayContext>| Ok((x.sin()?, x.cos()?));
        let tripled_backward = |cosine: DomainTracer<ArrayContext>, cotangent| {
            let gradient = cosine * cotangent;
            Ok(gradient.clone() + gradient.clone() + gradient)
        };
        let primal = |x: DomainTracer<ArrayContext>| Ok(x.sin()?);
        let jvp_first = custom_function(primal).with_jvp(doubled_jvp).with_vjp(forward, tripled_backward);
        let vjp_first = custom_function(primal).with_vjp(forward, tripled_backward).with_jvp(doubled_jvp);
        for (jvp, gradient) in [
            (
                differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| jvp_first.call(x)),
                differentiate_at(Array::scalar(2.0).unwrap()).gradient(|x| jvp_first.call(x)),
            ),
            (
                differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| vjp_first.call(x)),
                differentiate_at(Array::scalar(2.0).unwrap()).gradient(|x| vjp_first.call(x)),
            ),
        ] {
            assert_eq!(jvp, Ok((Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(2.0 * 2.0f64.cos()).unwrap())));
            assert_eq!(gradient, Ok(Array::scalar(3.0 * 2.0f64.cos()).unwrap()));
        }
    }

    #[test]
    fn test_custom_function_with_jvp_from_primal() {
        // Deriving the forward-mode rule from the primal yields the true derivative in both modes, unless reverse-mode
        // rules are configured as well, in which case they govern reverse mode (and triple the derivative here).
        let function = custom_function(|x: DomainTracer<ArrayContext>| Ok(x.sin()?)).with_jvp_from_primal();
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| function.call(x)),
            Ok((Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(2.0f64.cos()).unwrap())),
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).gradient(|x| function.call(x)),
            Ok(Array::scalar(2.0f64.cos()).unwrap()),
        );
        let function = function.with_vjp(
            |x: DomainTracer<ArrayContext>| Ok((x.sin()?, x.cos()?)),
            |cosine, cotangent| {
                let gradient = cosine * cotangent;
                Ok(gradient.clone() + gradient.clone() + gradient)
            },
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| function.call(x)),
            Ok((Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(2.0f64.cos()).unwrap())),
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).gradient(|x| function.call(x)),
            Ok(Array::scalar(3.0 * 2.0f64.cos()).unwrap()),
        );
    }

    #[test]
    fn test_custom_function_with_symbolic_zero_jvp() {
        // The product rule receives the structural zero of an input that is not being differentiated as a
        // `MaybeZero::Zero` leaf and skips its term, and each activity pattern is a separate specialization.
        type Tracer = DomainTracer<ArrayContext>;
        let observed = Arc::new(Mutex::new(Vec::new()));
        let function = custom_function(|(x, y): (Tracer, Tracer)| Ok(x * y)).with_symbolic_zero_jvp({
            let observed = observed.clone();
            move |(x, y): (Tracer, Tracer), tangents: (MaybeZero<Tracer>, MaybeZero<Tracer>)| {
                observed.lock().unwrap().push((tangents.0.is_zero(), tangents.1.is_zero()));
                let tangent = match tangents {
                    (MaybeZero::Value(dx), MaybeZero::Value(dy)) => dx * y.clone() + x.clone() * dy,
                    (MaybeZero::Value(dx), MaybeZero::Zero(_)) => dx * y.clone(),
                    (MaybeZero::Zero(_), MaybeZero::Value(dy)) => x.clone() * dy,
                    (MaybeZero::Zero(r#type), MaybeZero::Zero(_)) => x.context().zero(&r#type)?,
                };
                Ok((x * y, tangent))
            }
        });
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| {
                let y = x.context().lift(Array::scalar(3.0).unwrap())?;
                function.call((x, y))
            }),
            Ok((Array::scalar(6.0).unwrap(), Array::scalar(3.0).unwrap())),
        );
        assert_eq!(
            differentiate_at((Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()))
                .jvp((Array::scalar(1.0).unwrap(), Array::scalar(1.0).unwrap()), |inputs| function.call(inputs)),
            Ok((Array::scalar(6.0).unwrap(), Array::scalar(5.0).unwrap())),
        );
        assert_eq!(*observed.lock().unwrap(), vec![(false, true), (false, false)]);
    }

    #[test]
    fn test_custom_function_with_vjp() {
        // Reverse-mode rules govern reverse mode, while forward mode rejects the call because no forward-mode rule was
        // configured.
        let function = custom_function(|x: DomainTracer<ArrayContext>| Ok(x.sin()?)).with_vjp(
            |x| Ok((x.sin()?, x.cos()?)),
            |cosine, cotangent| {
                let gradient = cosine * cotangent;
                Ok(gradient.clone() + gradient.clone() + gradient)
            },
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).gradient(|x| function.call(x)),
            Ok(Array::scalar(3.0 * 2.0f64.cos()).unwrap()),
        );
        assert!(matches!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| function.call(x)),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "cannot apply forward-mode differentiation to a `custom_function` call that has only \
                               reverse-mode rules; it supports only reverse-mode differentiation (e.g., `vjp`, \
                               `value_and_gradient`, or `jacobian_reverse`)",
        ));
    }

    #[test]
    fn test_custom_function_with_vjp_specialized_residual_structures() {
        // Specializations of one call structure may return differently structured residuals: here, one residual per
        // input dimension, so the scalar and vector specializations save zero and two residuals, respectively.
        let function = custom_function(|x: DomainTracer<ArrayContext>| Ok(x.sin()?)).with_vjp(
            |x| {
                let cosine = x.cos()?;
                Ok((x.sin()?, (cosine.clone(), vec![cosine; x.r#type().rank()])))
            },
            |(cosine, _): (DomainTracer<ArrayContext>, Vec<DomainTracer<ArrayContext>>), cotangent| {
                Ok(cosine * cotangent)
            },
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).gradient(|x| function.call(x)),
            Ok(Array::scalar(2.0f64.cos()).unwrap()),
        );
        let (_, pullback) =
            differentiate_at(Array::vector(vec![1.0f64, 2.0]).unwrap()).vjp(|x| function.call(x)).unwrap();
        assert_eq!(
            pullback.apply(Array::vector(vec![1.0f64, 1.0]).unwrap()),
            Ok(Array::vector(vec![1.0f64.cos(), 2.0f64.cos()]).unwrap()),
        );

        // Residual structures are recorded by their flat residual types, so two specializations whose residuals have
        // identical flat types but different structures are ambiguous. Here, the residual type is always `f64[]`.
        let function = custom_function(|x: DomainTracer<ArrayContext>| Ok(x.sin()?)).with_vjp(
            |x| {
                let total = x.cos()?.reduce(&(0..x.r#type().rank()).collect::<Vec<_>>(), ReductionKind::Sum)?;
                let residuals = if x.r#type().rank() == 0 { vec![vec![total]] } else { vec![vec![], vec![total]] };
                Ok((x.sin()?, residuals))
            },
            |_: Vec<Vec<DomainTracer<ArrayContext>>>, cotangent| Ok(cotangent),
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).gradient(|x| function.call(x)),
            Ok(Array::scalar(1.0).unwrap()),
        );
        assert_eq!(
            differentiate_at(Array::vector(vec![1.0f64, 2.0]).unwrap())
                .vjp(|x| function.call(x))
                .map(|_| ())
                .map_err(ProgramError::from),
            Err(ProgramError::Parameter(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{:?}", vec![vec![Placeholder]]),
                right_structure: format!("{:?}", vec![vec![], vec![Placeholder]]),
            })),
        );
    }

    #[test]
    fn test_custom_function_with_vjp_dimension_residuals() {
        // The forward rule saves the dimension `n · n` of its input's extent `n`. Replaying the traced forward rule
        // mints a fresh identity for that dimension, so the backward rule must find the residual structure by type
        // equivalence up to identity renaming.
        let saved_extent = |x: ArrayIrTracer| -> Result<(ArrayIrTracer, ArrayIrTracer), ProgramError> {
            let extent = ValueProjection::<DimensionType>::into_projected(x.dimension_size(0)?)?;
            Ok((x, (extent.clone() * extent).into_value()))
        };
        let function = custom_function(|x: ArrayIrTracer| Ok(x))
            .with_vjp(saved_extent, |_: ArrayIrTracer, cotangent: ArrayIrTracer| Ok(cotangent));
        let input = ArrayIrValue::Array(Array::vector(vec![1.0f64, 2.0]).unwrap());
        let seed = ArrayIrValue::Array(Array::vector(vec![3.0f64, 4.0]).unwrap());
        let (_, pullback) = differentiate_at(input).vjp(|x| function.call(x)).unwrap();
        assert_eq!(pullback.apply(seed.clone()), Ok(seed));
    }

    #[test]
    fn test_custom_function_with_vjp_computed_dimension_outputs() {
        // The primal broadcasts `x` to the computed extent `n · n`, and the forward rule recomputes that extent in its
        // own trace, which mints a fresh identity for it. The rule outputs therefore agree with the primal outputs
        // only up to a renaming of that identity, which is applied to the rule program, including its residuals.
        let broadcast = |dimension: &ArrayIrTracer, x: ArrayIrTracer| -> Result<_, ProgramError> {
            let dimension = ValueProjection::<DimensionType>::into_projected(dimension.clone())?;
            let extent = (dimension.clone() * dimension).into_value();
            Ok((extent.clone(), x.dynamic_broadcast(&[extent], &[])?))
        };
        let function = custom_function(move |(dimension, x): (ArrayIrTracer, ArrayIrTracer)| broadcast(&dimension, x))
            .with_vjp(
                move |(dimension, x): (ArrayIrTracer, ArrayIrTracer)| {
                    let (extent, output) = broadcast(&dimension, x)?;
                    Ok(((extent.clone(), output), (dimension, extent)))
                },
                |(dimension, _): (ArrayIrTracer, ArrayIrTracer), (_, cotangent): (ArrayIrTracer, ArrayIrTracer)| {
                    let cotangent = ValueProjection::<ArrayType>::into_projected(cotangent)?;
                    Ok((dimension, cotangent.reduce(&[0], ReductionKind::Sum)?.into_value()))
                },
            )
            .with_non_differentiated_count(1);
        let dimension = DimensionType::new("n", DimensionBounds::new(2, Some(5)).unwrap());
        let input_types = (ArrayIrType::from(dimension.clone()), ArrayIrType::from(ArrayType::scalar(DataType::F64)));
        let (_, program) = EagerArrayIrContext::trace(|inputs| function.call(inputs), input_types).unwrap();
        let linearization = program
            .into_flat_program()
            .entry_region_ref()
            .linearize_shared_for_rule(&[1], DifferentiationRule::JvpForTranspose)
            .unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:dimension<n ∈ [2, 5)>, %1:f64[] .
                let %2:dimension<n * n ∈ [4, 17)> = dimension_mul %0 %0
                    %3:f64[n * n] = broadcast [output_axes=[]] %1 %2
                    %4:dimension<n * n ∈ [4, 17)> = dimension_size [axis=0] %3
                in (%2, %3, %0, %2, %4)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[], %1:dimension<n ∈ [2, 5)>, %2:dimension<n * n ∈ [4, 17)>, %3:dimension<n * n ∈ [4, 17)> .
                let %4:zero[], %5:f64[n * n] = custom_function_transpose [name=\"custom_function\", leading_input_count=4, seed_geometry_count=1] %1 %1 %2 %3 %0
                in (%5)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().transpose_with_respect_to(&[0], &[]).unwrap().to_string(),
            indoc! {"
                lambda %0:f64[n * n], %1:dimension<n ∈ [2, 5)>, %2:dimension<n * n ∈ [4, 17)>, %3:dimension<n * n ∈ [4, 17)> .
                let %4:zero[] = zero [type=zero[]]
                    %5:f64[] = reduce [kind=sum, axes=[0]] %0
                in (%5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_function_with_vjp_non_differentiated_inputs() {
        // The stash-gradients pattern: the leading `stash` reference is plumbing, the forward rule forwards it as a
        // residual by identity rather than saving a snapshot, and the backward rule writes the incoming cotangent into
        // it before returning `x̄ = cos(x) · ȳ`. The cotangent value that it returns is input-shaped, so its leading
        // plumbing leaf is ignored.
        let function = custom_function(|(_, x): (ArrayIrTracer, ArrayIrTracer)| {
            Ok(ValueProjection::<ArrayType>::into_projected(x)?.sin()?.into_value())
        })
        .with_vjp(
            |(stash, x)| {
                let x = ValueProjection::<ArrayType>::into_projected(x)?;
                Ok((x.sin()?.into_value(), (stash, x.cos()?.into_value())))
            },
            |(stash, cosine), cotangent| {
                stash.write(&cotangent)?;
                let cosine = ValueProjection::<ArrayType>::into_projected(cosine)?;
                let cotangent = ValueProjection::<ArrayType>::into_projected(cotangent)?;
                Ok((stash, (cosine * cotangent).into_value()))
            },
        )
        .with_non_differentiated_count(1);
        let stash = ArrayReference::new(Array::scalar(0.0f32).unwrap());
        let (value, pullback) = differentiate_at((
            ArrayIrValue::Reference(stash.clone()),
            ArrayIrValue::Array(Array::scalar(0.5f32).unwrap()),
        ))
        .vjp(|(stash, x)| function.call((stash, x)))
        .unwrap();
        assert_eq!(value, ArrayIrValue::Array(Array::scalar(0.5f32.sin()).unwrap()));

        // Linearization replays only the forward rule, which does not touch the stash.
        assert_eq!(stash.read(), Ok(Array::scalar(0.0f32).unwrap()));

        // The stash is a plumbing input, so its own cotangent is ignored, while applying the pullback replays the
        // backward rule: `x̄` is the custom gradient and the stash now holds the cotangent that was pulled back.
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(ArrayIrValue::Array(Array::scalar(2.0f32).unwrap())),
                (CotangentDestination::Ignore, CotangentDestination::Return),
            ),
            Ok((None, Some(ArrayIrValue::Array(Array::scalar(0.5f32.cos() * 2.0).unwrap())))),
        );
        assert_eq!(stash.read(), Ok(Array::scalar(2.0f32).unwrap()));

        // Every application writes the stash anew.
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(ArrayIrValue::Array(Array::scalar(3.0f32).unwrap())),
                (CotangentDestination::Ignore, CotangentDestination::Return),
            ),
            Ok((None, Some(ArrayIrValue::Array(Array::scalar(0.5f32.cos() * 3.0).unwrap())))),
        );
        assert_eq!(stash.read(), Ok(Array::scalar(3.0f32).unwrap()));

        // The non-differentiated count cannot exceed the number of input leaves.
        let function = custom_function(|x: DomainTracer<ArrayContext>| Ok(x.sin()?))
            .with_vjp(|x| Ok((x.sin()?, x.cos()?)), |residual, cotangent| Ok(residual * cotangent))
            .with_non_differentiated_count(2);
        assert_eq!(
            ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64)).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "`custom_function` non-differentiated input count 2 exceeds input count 1".to_string(),
            ))),
        );
    }

    #[test]
    fn test_custom_function_with_vjp_call() {
        // The wrapper traces the closures at the call site, specialized to the input types. The custom
        // rule `backward(residual, cotangent) = 3 * residual * cotangent` triples the true gradient (expressed through
        // addition to avoid constant lifting), which proves that the rule is in control.
        let function = custom_function(|x: DomainTracer<ArrayContext>| Ok(x.sin()?)).with_vjp(
            |x| Ok((x.sin()?, x.cos()?)),
            |residual, cotangent| {
                let product = residual * cotangent;
                Ok(product.clone() + product.clone() + product)
            },
        );
        let (_, program) = ArrayContext::trace(|input| function.call(input), ArrayType::scalar(DataType::F64)).unwrap();
        assert_eq!(program.interpret(Array::scalar(2.0).unwrap()), Ok(Array::scalar(2.0f64.sin()).unwrap()));
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).value_and_gradient(|x| function.call(x).unwrap()),
            Ok((Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(3.0 * 2.0f64.cos()).unwrap())),
        );
    }

    #[test]
    fn test_custom_function_with_vjp_tracing_errors() {
        // An empty collection provides no value from which the call can recover its execution context.
        let function = custom_function(|inputs: Vec<DomainTracer<ArrayContext>>| Ok(inputs))
            .with_vjp(|inputs| Ok((inputs, ())), |(), cotangents| Ok(cotangents));
        assert_eq!(
            ArrayContext::trace(|inputs| function.call(inputs), Vec::<ArrayType>::new()).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid("`custom_function` requires at least one input".to_string()))),
        );

        // Each closure's original error must propagate rather than becoming a rule-interface diagnostic. The primal
        // closure is traced when the function is called, while the forward and backward closures are traced by the
        // first reverse-mode derivative request.
        let function =
            custom_function(|_: DomainTracer<ArrayContext>| -> Result<DomainTracer<ArrayContext>, ProgramError> {
                Err(ProgramError::InvalidArgument { message: "primal tracing failed".to_string() })
            })
            .with_vjp(|input| Ok((input, ())), |(), cotangent| Ok(cotangent));
        assert_eq!(
            ArrayContext::trace(|input| function.call(input), ArrayType::scalar(DataType::F64)).map(|_| ()),
            Err(ProgramError::InvalidArgument { message: "primal tracing failed".to_string() }),
        );
        let function = custom_function(|input: DomainTracer<ArrayContext>| Ok(input)).with_vjp(
            |_| -> Result<(DomainTracer<ArrayContext>, ()), ProgramError> {
                Err(ProgramError::InvalidArgument { message: "forward tracing failed".to_string() })
            },
            |(), cotangent| Ok(cotangent),
        );
        assert_eq!(
            differentiate_at(Array::scalar(1.0).unwrap())
                .vjp(|input| function.call(input))
                .and_then(|(_, pullback)| pullback.into_transposed_parts())
                .map(|_| ())
                .map_err(ProgramError::from),
            Err(ProgramError::InvalidArgument { message: "forward tracing failed".to_string() }),
        );
        let function = custom_function(|input: DomainTracer<ArrayContext>| Ok(input)).with_vjp(
            |input| Ok((input, ())),
            |(), _| Err(ProgramError::InvalidArgument { message: "backward tracing failed".to_string() }),
        );
        assert_eq!(
            differentiate_at(Array::scalar(1.0).unwrap())
                .vjp(|input| function.call(input))
                .and_then(|(_, pullback)| pullback.into_transposed_parts())
                .map(|_| ())
                .map_err(ProgramError::from),
            Err(ProgramError::InvalidArgument { message: "backward tracing failed".to_string() }),
        );
    }

    #[test]
    fn test_custom_function_with_vjp_rule_signature_mismatch() {
        // The closure signatures fix the parameter families, but array shapes still require validation after tracing,
        // which happens when the first reverse-mode derivative request specializes the rules.
        let function = custom_function(|input: DomainTracer<ArrayContext>| Ok(input))
            .with_vjp(|input| Ok((input.reduce(&[0], ReductionKind::Sum)?, ())), |(), cotangent| Ok(cotangent));
        assert_eq!(
            differentiate_at(Array::vector(vec![1.0f64, 2.0]).unwrap())
                .vjp(|input| function.call(input))
                .and_then(|(_, pullback)| pullback.into_transposed_parts())
                .map(|_| ())
                .map_err(ProgramError::from),
            Err(ProgramError::Type(TypeError::invalid(
                "`custom_function` `custom_function` forward rule output type signature mismatch: expected [f64[2]] \
                 but got [f64[]]"
                    .to_string(),
            ))),
        );

        // A valid forward interface must not hide a backward rule that produces the wrong input-cotangent shape.
        let function = custom_function(|input: DomainTracer<ArrayContext>| Ok(input))
            .with_vjp(|input| Ok((input, ())), |(), cotangent| Ok(cotangent.reduce(&[0], ReductionKind::Sum)?));
        assert_eq!(
            differentiate_at(Array::vector(vec![1.0f64, 2.0]).unwrap())
                .vjp(|input| function.call(input))
                .and_then(|(_, pullback)| pullback.into_transposed_parts())
                .map(|_| ())
                .map_err(ProgramError::from),
            Err(ProgramError::Type(TypeError::invalid(
                "`custom_function` `custom_function` backward rule output type signature mismatch: expected [f64[2]] \
                 but got [f64[]]"
                    .to_string(),
            ))),
        );

        // The forward result must preserve nested collection lengths, not just the flattened types.
        let function = custom_function(|input: DomainTracer<ArrayContext>| {
            Ok(vec![vec![input.clone(), input.clone()], vec![input]])
        })
        .with_vjp(
            |input| Ok((vec![vec![input.clone()], vec![input.clone(), input]], ())),
            |(), cotangents| Ok(cotangents[0][0].clone()),
        );
        assert_eq!(
            differentiate_at(Array::scalar(1.0).unwrap())
                .vjp(|input| function.call(input))
                .map(|_| ())
                .map_err(ProgramError::from),
            Err(ProgramError::Parameter(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{:?}", vec![vec![Placeholder; 2], vec![Placeholder]]),
                right_structure: format!("{:?}", vec![vec![Placeholder], vec![Placeholder; 2]]),
            })),
        );

        // Check the full backward structure even when the first flattened leaf is non-differentiated. Reverse mode
        // linearizes with respect to the two differentiated leaves only and then transposes both tangent inputs.
        let function = custom_function(|input: Vec<Vec<DomainTracer<ArrayContext>>>| Ok(input[1][0].clone()))
            .with_vjp(
                |input| Ok((input[1][0].clone(), ())),
                |(), cotangent| Ok(vec![vec![cotangent.clone()], vec![cotangent.clone(), cotangent]]),
            )
            .with_non_differentiated_count(1);
        let (_, program) = ArrayContext::trace(
            |input| function.call(input),
            vec![vec![ArrayType::scalar(DataType::F64); 2], vec![ArrayType::scalar(DataType::F64)]],
        )
        .unwrap();
        assert_eq!(
            program
                .into_flat_program()
                .entry_region_ref()
                .linearize_shared_for_rule(&[1, 2], DifferentiationRule::JvpForTranspose)
                .and_then(|linearization| linearization.tangent().transpose_with_respect_to(&[0, 1], &[]))
                .map(|_| ())
                .map_err(ProgramError::from),
            Err(ProgramError::Parameter(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{:?}", vec![vec![Placeholder; 2], vec![Placeholder]]),
                right_structure: format!("{:?}", vec![vec![Placeholder], vec![Placeholder; 2]]),
            })),
        );
    }

    #[test]
    fn test_custom_function_with_vjp_reference_contract() {
        let input_types = (
            ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))),
            ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
        );

        // A reference input that is not declared as plumbing is an active input, which the staged operation rejects.
        let function = custom_function(|(_, x): (ArrayIrTracer, ArrayIrTracer)| Ok(x))
            .with_vjp(|(stash, x)| Ok((x, stash)), |stash, cotangent| Ok((stash, cotangent)));
        assert_eq!(
            EagerArrayIrContext::trace(|(stash, x)| function.call((stash, x)), input_types.clone()).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "`custom_function` accepts reference inputs only in its leading non-differentiated segment; move \
                 input 0 of type `ref<f32[]>` before the differentiated inputs"
                    .to_string(),
            ))),
        );

        // A forward rule may forward the plumbing reference as a residual but not return a reference it allocated.
        // Residuals are only produced by the forward rule, which the first reverse-mode derivative request traces.
        let inputs = (
            ArrayIrValue::Reference(ArrayReference::new(Array::scalar(0.0f32).unwrap())),
            ArrayIrValue::Array(Array::scalar(0.5f32).unwrap()),
        );
        let function = custom_function(|(_, x): (ArrayIrTracer, ArrayIrTracer)| Ok(x))
            .with_vjp(|(_, x)| Ok((x.clone(), x.reference_new()?)), |allocated, cotangent| Ok((allocated, cotangent)))
            .with_non_differentiated_count(1);
        assert_eq!(
            differentiate_at(inputs.clone())
                .vjp(|(stash, x)| function.call((stash, x)))
                .and_then(|(_, pullback)| pullback.into_transposed_parts())
                .map(|_| ())
                .map_err(ProgramError::from),
            Err(ProgramError::Type(TypeError::invalid(
                "`custom_function` forward rule returns residual 0 of reference type `ref<f32[]>` that is not a \
                 leading non-differentiated input forwarded by identity"
                    .to_string(),
            ))),
        );

        // Every reference-typed residual is held to that rule, not only the first one: forwarding the plumbing
        // reference and then returning an allocated reference beside it is rejected at the allocated residual.
        let function = custom_function(|(_, x): (ArrayIrTracer, ArrayIrTracer)| Ok(x))
            .with_vjp(
                |(stash, x)| Ok((x.clone(), (stash, x.reference_new()?))),
                |(stash, _allocated), cotangent| Ok((stash, cotangent)),
            )
            .with_non_differentiated_count(1);
        assert_eq!(
            differentiate_at(inputs.clone())
                .vjp(|(stash, x)| function.call((stash, x)))
                .and_then(|(_, pullback)| pullback.into_transposed_parts())
                .map(|_| ())
                .map_err(ProgramError::from),
            Err(ProgramError::Type(TypeError::invalid(
                "`custom_function` forward rule returns residual 1 of reference type `ref<f32[]>` that is not a \
                 leading non-differentiated input forwarded by identity"
                    .to_string(),
            ))),
        );

        // No rule may return a reference as a primal output, not even a plumbing input forwarded by identity.
        let function = custom_function(|(stash, _): (ArrayIrTracer, ArrayIrTracer)| Ok(stash))
            .with_vjp(|(stash, _)| Ok((stash, ())), |(), cotangent| Ok((cotangent.clone(), cotangent.read()?)))
            .with_non_differentiated_count(1);
        assert_eq!(
            EagerArrayIrContext::trace(|(stash, x)| function.call((stash, x)), input_types).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "`custom_function` cannot return a reference, but output 0 has type `ref<f32[]>`".to_string(),
            ))),
        );

        // Two same-typed inputs do not permit forwarding the first one twice. Type multiplicity alone cannot detect
        // this alias, so the specialized forward rule checks which input each residual forwards.
        let function = custom_function(|(_, _, input): (ArrayIrTracer, ArrayIrTracer, ArrayIrTracer)| Ok(input))
            .with_vjp(
                |(first, _, input)| Ok((input, (first.clone(), first))),
                |(first, second), cotangent| Ok((first, second, cotangent)),
            )
            .with_non_differentiated_count(2);
        assert_eq!(
            differentiate_at((
                ArrayIrValue::Reference(ArrayReference::new(Array::scalar(0.0f32).unwrap())),
                ArrayIrValue::Reference(ArrayReference::new(Array::scalar(0.0f32).unwrap())),
                ArrayIrValue::Array(Array::scalar(0.5f32).unwrap()),
            ))
            .vjp(|input| function.call(input))
            .and_then(|(_, pullback)| pullback.into_transposed_parts())
            .map(|_| ())
            .map_err(ProgramError::from),
            Err(ProgramError::Type(TypeError::invalid(
                "`custom_function` forward rule returns residual 1 of reference type `ref<f32[]>` from an input \
                 already forwarded by an earlier residual",
            ))),
        );
    }

    #[test]
    fn test_custom_function_with_vjp_pullback() {
        // Extracting the stored pullback must preserve the custom backward rule and its residual inputs. The forward
        // rule saves `cos(x)`, and the backward rule returns `2 * residual * cotangent`, so a unit cotangent at
        // `x = 0.7` produces `2 * cos(0.7)` rather than the mathematical derivative.
        let function = custom_function(|x: DomainTracer<ArrayContext>| Ok(x.sin()?)).with_vjp(
            |x| Ok((x.sin()?, x.cos()?)),
            |residual, cotangent| Ok((residual.clone() + residual) * cotangent),
        );
        let (_, pullback) = differentiate_at(Array::scalar(0.7).unwrap()).vjp(|x| function.call(x)).unwrap();
        let (pullback, residuals) = pullback.into_transposed_parts().unwrap();
        let mut pullback_inputs = vec![Array::scalar(1.0).unwrap()];
        pullback_inputs.extend(residuals);
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![Array::scalar(2.0 * 0.7f64.cos()).unwrap()]));
    }

    #[test]
    fn test_custom_function_with_vjp_multiple_outputs() {
        // A two-output custom VJP exercises the output/residual split of the forward region: its leading values are the
        // primal outputs and the rest are residuals, and the backward region consumes one cotangent per output. The
        // custom rule scales the contribution of the first output by 2 and that of the second by 3, so
        // seeding one output cotangent at a time isolates each term of the custom backward rule.
        let function = custom_function(|x: DomainTracer<ArrayContext>| Ok((x.sin()?, x.cos()?))).with_vjp(
            |x| Ok(((x.sin()?, x.cos()?), (x.cos()?, x.sin()?))),
            |(cosine, sine), (first, second)| {
                let from_first = cosine * first;
                let from_second = sine * second;
                Ok(from_first.clone() + from_first + from_second.clone() + from_second.clone() + from_second)
            },
        );
        let ((sine, cosine), pullback) =
            differentiate_at(Array::scalar(0.5).unwrap()).vjp(|x| function.call(x)).unwrap();
        assert_eq!(sine, Array::scalar(0.5f64.sin()).unwrap());
        assert_eq!(cosine, Array::scalar(0.5f64.cos()).unwrap());
        assert_eq!(
            pullback.apply((Array::scalar(1.0).unwrap(), Array::scalar(0.0).unwrap())),
            Ok(Array::scalar(2.0 * 0.5f64.cos()).unwrap()),
        );
        assert_eq!(
            pullback.apply((Array::scalar(0.0).unwrap(), Array::scalar(1.0).unwrap())),
            Ok(Array::scalar(3.0 * 0.5f64.sin()).unwrap()),
        );
    }

    #[test]
    fn test_custom_function_with_vjp_structured_signatures() {
        // Tuple inputs and tuple residuals exercise the `Parameterized` calling convention, and the captured `repeats`
        // count plays the role of static configuration that is visible to the rule closures without being
        // differentiated or stored as a residual.
        let repeats = 3usize;
        let function = custom_function(|(x, y): (DomainTracer<ArrayContext>, DomainTracer<ArrayContext>)| Ok(x * y))
            .with_vjp(
                |(x, y)| Ok((x.clone() * y.clone(), (x, y))),
                move |(x, y), cotangent| {
                    // The custom rule repeats both cotangents `repeats` times.
                    let (base_x, base_y) = (y * cotangent.clone(), x * cotangent);
                    let (mut scaled_x, mut scaled_y) = (base_x.clone(), base_y.clone());
                    for _ in 1..repeats {
                        scaled_x = scaled_x + base_x.clone();
                        scaled_y = scaled_y + base_y.clone();
                    }
                    Ok((scaled_x, scaled_y))
                },
            );

        // The custom rule triples the true gradients `(y, x)`.
        assert_eq!(
            differentiate_at((Array::scalar(2.0).unwrap(), Array::scalar(5.0).unwrap()))
                .value_and_gradient(|(x, y)| function.call((x, y)).unwrap()),
            Ok((Array::scalar(10.0).unwrap(), (Array::scalar(15.0).unwrap(), Array::scalar(6.0).unwrap()))),
        );
    }

    #[test]
    fn test_custom_function_with_vjp_empty_residuals() {
        // A forward rule that saves nothing (i.e., `Residuals = ()`) exercises the zero-residual carrier path: the
        // backward rule depends only on the output cotangent, so the custom rule
        // `backward(cotangent) = 2 * cotangent` makes the gradient the constant `2` instead of `cos(x)`.
        let function = custom_function(|x: DomainTracer<ArrayContext>| Ok(x.sin()?))
            .with_vjp(|x| Ok((x.sin()?, ())), |(), cotangent| Ok(cotangent.clone() + cotangent));
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).value_and_gradient(|x| function.call(x).unwrap()),
            Ok((Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(2.0).unwrap())),
        );
    }

    #[test]
    fn test_custom_function_with_vjp_batching_replicated_inputs() {
        // Mapping only the first input verifies that the replicated input remains shared at the call boundary while
        // operations inside its regions broadcast it only where per-item multiplication requires alignment.
        let function = custom_function(|(x, y): (DomainTracer<ArrayContext>, DomainTracer<ArrayContext>)| Ok(x * y))
            .with_vjp(
                |(x, y)| Ok((x.clone() * y.clone(), (x, y))),
                |(x, y), cotangent| Ok((y * cotangent.clone(), x * cotangent)),
            );
        let output: Array = batch(
            |(x, y)| function.call((x, y)),
            (Array::vector(vec![2.0, 3.0, 4.0]).unwrap(), Array::scalar(5.0).unwrap()),
            (BatchAxis::new(0), BatchAxis::replicated()),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(output, Array::vector(vec![10.0, 15.0, 20.0]).unwrap());
    }

    #[test]
    fn test_custom_function_with_vjp_zero_space_boundaries() {
        // Token primals, residuals, and zero-space cotangents carry no payload, so the wrapper must pass them through
        // the traced forward and backward rules unchanged instead of demanding a dense cotangent space.
        let token = Array::from_logical_bytes(ArrayType::scalar(DataType::Token), &[]).unwrap();
        let zero = Array::from_logical_bytes(ArrayType::scalar(DataType::Zero), &[]).unwrap();
        let function = custom_function(|token: DomainTracer<ArrayContext>| Ok(token))
            .with_vjp(|token| Ok((token.clone(), token)), |_residual, cotangent| Ok(cotangent));
        let (value, pullback) = differentiate_at(token.clone()).vjp(|token| function.call(token)).unwrap();
        assert_eq!(value, token);
        assert_eq!(pullback.apply(zero.clone()), Ok(zero));
    }

    #[test]
    fn test_custom_function_with_vjp_effectful_backward_destinations() {
        // This is one custom backward closure with an observable reference effect and an intentionally nonlinear
        // seed formula. Destination specialization must preserve the closure's execution and additive semantics.
        let stash = ArrayReference::new(Array::scalar(-1.0f32).unwrap());
        let function = custom_function(|(_, value): (ArrayIrTracer, ArrayIrTracer)| Ok(value))
            .with_vjp(
                |(stash, value)| Ok((value, stash)),
                |stash: ArrayIrTracer, seed: ArrayIrTracer| {
                    stash.write(&seed)?;
                    let contribution = seed
                        .context()
                        .bind(
                            ArrayOperation::<Array>::Mul(MulOperation::new()),
                            Vec::new(),
                            &[seed.clone(), seed.clone()],
                        )?
                        .remove(0);
                    Ok((stash, contribution))
                },
            )
            .with_non_differentiated_count(1);
        let (_, pullback) = differentiate_at((
            ArrayIrValue::Reference(stash.clone()),
            ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()),
        ))
        .vjp(|input| function.call(input))
        .unwrap();
        let first = ArrayReference::new(Array::scalar(10.0f32).unwrap());
        let second = ArrayReference::new(Array::scalar(20.0f32).unwrap());
        let seed = ArrayIrValue::Array(Array::scalar(3.0f32).unwrap());
        let destinations = [CotangentDestinationKind::Ignore, CotangentDestinationKind::Reference];
        let retained = pullback.transposed_program(&destinations).unwrap();
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(seed.clone()),
                (CotangentDestination::Ignore, CotangentDestination::Reference(ArrayIrValue::Reference(first.clone()))),
            ),
            Ok((None, None)),
        );
        assert_eq!(first.read(), Ok(Array::scalar(19.0f32).unwrap()));
        assert_eq!(stash.read(), Ok(Array::scalar(3.0f32).unwrap()));
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(seed.clone()),
                (
                    CotangentDestination::Ignore,
                    CotangentDestination::Reference(ArrayIrValue::Reference(second.clone()))
                ),
            ),
            Ok((None, None)),
        );
        assert_eq!(second.read(), Ok(Array::scalar(29.0f32).unwrap()));
        assert_eq!(first.read(), Ok(Array::scalar(19.0f32).unwrap()));
        assert!(Arc::ptr_eq(&retained, &pullback.transposed_program(&destinations).unwrap()));
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(seed),
                (CotangentDestination::Ignore, CotangentDestination::Return),
            ),
            Ok((None, Some(ArrayIrValue::Array(Array::scalar(9.0f32).unwrap())))),
        );

        // A numerical zero is still a live seed. Dropping every returned gradient must not erase the stash write.
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(ArrayIrValue::Array(Array::scalar(0.0f32).unwrap())),
                (CotangentDestination::Ignore, CotangentDestination::Ignore),
            ),
            Ok((None, None)),
        );
        assert_eq!(stash.read(), Ok(Array::scalar(0.0f32).unwrap()));
    }

    #[test]
    fn test_custom_function_with_symbolic_zero_vjp() {
        // The backward rule of `x ↦ (sin(x), cos(x))` receives the structural-zero seed of the unused second output as
        // a `MaybeZero::Zero` leaf and skips its term.
        type Tracer = DomainTracer<ArrayContext>;
        let observed = Arc::new(Mutex::new(Vec::new()));
        let function = custom_function(|x: Tracer| Ok((x.sin()?, x.cos()?))).with_symbolic_zero_vjp(
            |x| Ok(((x.sin()?, x.cos()?), (x.cos()?, x.sin()?))),
            {
                let observed = observed.clone();
                move |(cosine, sine): (Tracer, Tracer), seeds: (MaybeZero<Tracer>, MaybeZero<Tracer>)| {
                    observed.lock().unwrap().push((seeds.0.is_zero(), seeds.1.is_zero()));
                    Ok(match seeds {
                        (MaybeZero::Value(sine_seed), MaybeZero::Value(cosine_seed)) => {
                            cosine * sine_seed - sine * cosine_seed
                        }
                        (MaybeZero::Value(sine_seed), MaybeZero::Zero(_)) => cosine * sine_seed,
                        (MaybeZero::Zero(_), MaybeZero::Value(cosine_seed)) => -(sine * cosine_seed),
                        (MaybeZero::Zero(r#type), MaybeZero::Zero(_)) => cosine.context().zero(&r#type)?,
                    })
                }
            },
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).gradient(|x| function.call(x).map(|(sine, _)| sine)),
            Ok(Array::scalar(2.0f64.cos()).unwrap()),
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap())
                .gradient(|x| function.call(x).map(|(sine, cosine)| sine + cosine)),
            Ok(Array::scalar(2.0f64.cos() - 2.0f64.sin()).unwrap()),
        );
        assert_eq!(*observed.lock().unwrap(), vec![(false, true), (false, false)]);
    }

    #[test]
    fn test_custom_function_with_symbolic_zero_vjp_dimension_residuals() {
        // The forward rule saves the dimension `n · n` of its input's extent `n`. Replaying the traced forward rule
        // mints a fresh identity for that dimension, so the backward rule must find the residual structure by type
        // equivalence up to identity renaming.
        let saved_extent = |x: ArrayIrTracer| -> Result<(ArrayIrTracer, ArrayIrTracer), ProgramError> {
            let extent = ValueProjection::<DimensionType>::into_projected(x.dimension_size(0)?)?;
            Ok((x, (extent.clone() * extent).into_value()))
        };
        let function = custom_function(|x: ArrayIrTracer| Ok(x)).with_symbolic_zero_vjp(
            saved_extent,
            |_: ArrayIrTracer, seed: MaybeZero<ArrayIrTracer>| match seed {
                MaybeZero::Value(seed) => Ok(seed),
                MaybeZero::Zero(_) => {
                    Err(ProgramError::InvalidArgument { message: "unexpected zero seed".to_string() })
                }
            },
        );
        let input = ArrayIrValue::Array(Array::vector(vec![1.0f64, 2.0]).unwrap());
        let seed = ArrayIrValue::Array(Array::vector(vec![3.0f64, 4.0]).unwrap());
        let (_, pullback) = differentiate_at(input).vjp(|x| function.call(x)).unwrap();
        assert_eq!(pullback.apply(seed.clone()), Ok(seed));
    }

    #[test]
    fn test_custom_function_with_accumulating_vjp() {
        // `(x, y) ↦ x[1] + y[1]` over `f64[4]` inputs. One accumulating backward rule serves every destination: it adds
        // the seed to only the affected entry of a caller buffer (without a full-sized temporary) and pads it into a
        // full cotangent for a returned destination. It is specialized once per pattern of destination kinds.
        type Transposition = TranspositionContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let element = |x: &ArrayIrTracer| -> Result<ArrayIrTracer, ProgramError> {
            Ok(ValueProjection::<ArrayType>::into_projected(x.clone())?.slice(&[1], &[2usize], &[1])?.into_value())
        };
        let add = |left: ArrayIrTracer, right: ArrayIrTracer| -> Result<ArrayIrTracer, ProgramError> {
            let operation = ArrayIrOperation::from(ArrayOperation::<Array>::Add(AddOperation::new()));
            Ok(left.context().bind(operation, Vec::new(), &[left.clone(), right])?.remove(0))
        };
        let invocations = Arc::new(AtomicUsize::new(0));
        let function = custom_function(move |(x, y): (ArrayIrTracer, ArrayIrTracer)| add(element(&x)?, element(&y)?))
            .with_accumulating_vjp(
                move |(x, y): (ArrayIrTracer, ArrayIrTracer)| Ok((add(element(&x)?, element(&y)?)?, ())),
                {
                    let invocations = invocations.clone();
                    move |context: &mut Transposition,
                          (): (),
                          seed: MaybeZero<ArrayIrTracer>,
                          accumulators: (CotangentAccumulator, CotangentAccumulator)| {
                        invocations.fetch_add(1, Ordering::SeqCst);
                        let MaybeZero::Value(seed) = seed else {
                            return Ok(());
                        };
                        for accumulator in [accumulators.0, accumulators.1] {
                            if !accumulator.is_needed() {
                                continue;
                            }
                            if let Some(buffer) = accumulator.reference(context)? {
                                let operation = ReferenceAddUpdateOperation::new().with_transforms(vec![
                                    ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 1, 1)] },
                                ]);
                                context.bind(ArrayIrOperation::from(operation), Vec::new(), &[buffer, seed.clone()])?;
                            } else {
                                let zero = context.lift(ArrayIrValue::Array(Array::scalar(0.0f64)?))?;
                                let operation = ArrayIrOperation::from(ArrayOperation::<Array>::Pad(
                                    PadOperation::new(vec![1], vec![2], vec![0])?,
                                ));
                                let cotangent = context.bind(operation, Vec::new(), &[seed.clone(), zero])?.remove(0);
                                accumulator.accumulate(context, MaybeZero::Value(cotangent))?;
                            }
                        }
                        Ok(())
                    }
                },
            );
        let vector = |values: [f64; 4]| ArrayIrValue::Array(Array::vector(values.to_vec()).unwrap());
        let seed = || CotangentSeed::Value(ArrayIrValue::Array(Array::vector(vec![5.0f64]).unwrap()));
        let (value, pullback) = differentiate_at((vector([1.0, 2.0, 3.0, 4.0]), vector([10.0, 20.0, 30.0, 40.0])))
            .vjp(|inputs| function.call(inputs))
            .unwrap();
        assert_eq!(value, ArrayIrValue::Array(Array::vector(vec![22.0f64]).unwrap()));

        // Mixed destinations: a caller buffer receives only its affected entry, and a returned cotangent is padded.
        let buffer = ArrayReference::new(Array::vector(vec![1.0f64; 4]).unwrap());
        assert_eq!(
            pullback.apply_with_destinations(
                seed(),
                (
                    CotangentDestination::Reference(ArrayIrValue::Reference(buffer.clone())),
                    CotangentDestination::Return
                ),
            ),
            Ok((None, Some(vector([0.0, 5.0, 0.0, 0.0])))),
        );
        assert_eq!(buffer.read(), Ok(Array::vector(vec![1.0f64, 6.0, 1.0, 1.0]).unwrap()));
        assert_eq!(invocations.load(Ordering::SeqCst), 1);

        // Reapplying the same destination pattern reuses its specialization, while another pattern specializes again.
        assert_eq!(
            pullback.apply_with_destinations(
                seed(),
                (
                    CotangentDestination::Reference(ArrayIrValue::Reference(buffer.clone())),
                    CotangentDestination::Return
                ),
            ),
            Ok((None, Some(vector([0.0, 5.0, 0.0, 0.0])))),
        );
        assert_eq!(buffer.read(), Ok(Array::vector(vec![1.0f64, 11.0, 1.0, 1.0]).unwrap()));
        assert_eq!(
            pullback.apply_with_destinations(seed(), (CotangentDestination::Ignore, CotangentDestination::Return)),
            Ok((None, Some(vector([0.0, 5.0, 0.0, 0.0])))),
        );
        assert_eq!(invocations.load(Ordering::SeqCst), 2);

        // Repeated inputs: both accumulators of `x ↦ f(x, x)` add to the same caller buffer.
        let (_, pullback) =
            differentiate_at(vector([1.0, 2.0, 3.0, 4.0])).vjp(|x| function.call((x.clone(), x))).unwrap();
        let buffer = ArrayReference::new(Array::vector(vec![0.0f64; 4]).unwrap());
        assert_eq!(
            pullback.apply_with_destinations(
                seed(),
                CotangentDestination::Reference(ArrayIrValue::Reference(buffer.clone())),
            ),
            Ok(None),
        );
        assert_eq!(buffer.read(), Ok(Array::vector(vec![0.0f64, 10.0, 0.0, 0.0]).unwrap()));
    }

    #[test]
    fn test_custom_function_with_accumulating_vjp_dimension_residuals() {
        // The forward rule saves the dimension `n · n` of its input's extent `n`. Replaying the traced forward rule
        // mints a fresh identity for that dimension, so the backward rule must find the residual structure by type
        // equivalence up to identity renaming.
        let saved_extent = |x: ArrayIrTracer| -> Result<(ArrayIrTracer, ArrayIrTracer), ProgramError> {
            let extent = ValueProjection::<DimensionType>::into_projected(x.dimension_size(0)?)?;
            Ok((x, (extent.clone() * extent).into_value()))
        };
        type Transposition = TranspositionContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let function = custom_function(|x: ArrayIrTracer| Ok(x)).with_accumulating_vjp(
            saved_extent,
            |context: &mut Transposition,
             _: ArrayIrTracer,
             seed: MaybeZero<ArrayIrTracer>,
             accumulator: CotangentAccumulator| { accumulator.accumulate(context, seed) },
        );
        let input = ArrayIrValue::Array(Array::vector(vec![1.0f64, 2.0]).unwrap());
        let seed = ArrayIrValue::Array(Array::vector(vec![3.0f64, 4.0]).unwrap());
        let (_, pullback) = differentiate_at(input).vjp(|x| function.call(x)).unwrap();
        assert_eq!(pullback.apply(seed.clone()), Ok(seed));
    }

    #[test]
    fn test_custom_function_with_batching() {
        // The rule of `(x, y) ↦ x · y` computes `y · x`, which shows that it takes precedence over structurally
        // batching the primal, and declares that the output follows the mapped input. It is traced once per batching
        // level and operand signature, which the extents that it records show.
        type Tracer = DomainTracer<ArrayContext>;
        let extents = Arc::new(Mutex::new(Vec::new()));
        let function = custom_function(|(x, y): (Tracer, Tracer)| Ok(x * y))
            .with_jvp(|(x, y): (Tracer, Tracer), (dx, dy): (Tracer, Tracer)| {
                let tangent = dx * y.clone() + x.clone() * dy;
                Ok((x * y, tangent.clone() + tangent))
            })
            .with_batching({
                let extents = extents.clone();
                move |extent: BatchingLevelExtent<Tracer>, (x, y): (Tracer, Tracer), axes: (BatchAxis, BatchAxis)| {
                    let BatchingLevelExtent::Static(extent) = extent else {
                        return Err(ProgramError::InvalidArgument { message: "unexpected dynamic extent".to_string() });
                    };
                    extents.lock().unwrap().push(extent);
                    Ok((y * x, if axes.0 == BatchAxis::replicated() { axes.1 } else { axes.0 }))
                }
            });
        assert_eq!(
            batch(
                |(x, y)| function.call((x, y)),
                (Array::vector(vec![2.0, 3.0, 4.0]).unwrap(), Array::scalar(5.0).unwrap()),
                (BatchAxis::new(0), BatchAxis::replicated()),
                BatchAxis::new(0),
                None,
            ),
            Ok(Array::vector(vec![10.0, 15.0, 20.0]).unwrap()),
        );
        assert_eq!(*extents.lock().unwrap(), vec![3]);

        // The staged batched call shares that trace, because it has the same level and operand signature.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let (_, program) =
            ArrayContext::trace(|inputs| function.call(inputs), (scalar_type.clone(), scalar_type)).unwrap();
        let batched = program
            .into_flat_program()
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[] .
                let %2:f64[3] = custom_function [
                    name=\"custom_function\",
                    batching=[(extent=3, input_axes=[axis 0, replicated], output_axes=[axis 0])],
                ] %0 %1 [
                    primal={
                        lambda %0:f64[3], %1:f64[] .
                        let %2:f64[3] = mul %1 %0
                        in (%2)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(*extents.lock().unwrap(), vec![3]);

        // Batching preserves the call's derivative rules: the JVP rule, which doubles the derivative, governs the
        // differentiation of the batched call.
        let x = Array::vector(vec![2.0, 3.0, 4.0]).unwrap();
        let (y, x_tangent, y_tangent) =
            (Array::scalar(5.0).unwrap(), Array::vector(vec![1.0; 3]).unwrap(), Array::scalar(1.0).unwrap());
        assert_eq!(
            batched.jvp().unwrap().interpret(vec![x, y, x_tangent, y_tangent]),
            Ok(vec![Array::vector(vec![10.0, 15.0, 20.0]).unwrap(), Array::vector(vec![14.0, 16.0, 18.0]).unwrap()]),
        );

        // Differentiating with respect to one input of the batched call specializes the JVP rule for that activity
        // pattern, materializing the other input's structural-zero tangent inside the batched specialization.
        assert_eq!(
            batched.jvp_with_respect_to(&[0]).unwrap().interpret(vec![
                Array::vector(vec![2.0, 3.0, 4.0]).unwrap(),
                Array::scalar(5.0).unwrap(),
                Array::vector(vec![1.0; 3]).unwrap(),
            ]),
            Ok(vec![Array::vector(vec![10.0, 15.0, 20.0]).unwrap(), Array::vector(vec![10.0; 3]).unwrap()]),
        );

        // Nested batching applies the rule again at the new level.
        let nested = batched
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(
            nested.interpret(vec![
                Array::matrix(2, 3, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap(),
                Array::scalar(5.0).unwrap(),
            ]),
            Ok(vec![Array::matrix(2, 3, vec![10.0, 15.0, 20.0, 25.0, 30.0, 35.0]).unwrap()]),
        );
        assert_eq!(*extents.lock().unwrap(), vec![3, 2]);
    }

    #[test]
    fn test_custom_function_named_axes() {
        // The primal and the rules are traced in fresh traces that are seeded with the named axes visible where the
        // function is called, so a function called under `batch` resolves the batch axis `items` in its primal
        // (`x ↦ x · i` at batch item `i`), its JVP rule (`(x, ẋ) ↦ (x · i, ẋ · i)`), its VJP rules (which save `i`
        // and pull back `ȳ ↦ ȳ · i`), and its named collectives (`x ↦ Σᵢ xᵢ`).
        type Tracer = DomainTracer<ArrayContext>;
        let index = |x: &Tracer| -> Result<Tracer, ProgramError> {
            x.context().axis_index("items")?.convert_element_type(DataType::F64)
        };
        let scaled = move |x: Tracer| -> Result<Tracer, ProgramError> { Ok(x.clone() * index(&x)?) };
        let inputs = Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap();
        let ones = Array::vector(vec![1.0f64; 3]).unwrap();
        let items = || BatchAxisSpecification::named("items");

        // Outside of `batch`, nothing binds `items`.
        let function = custom_function(scaled);
        assert_eq!(
            ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64)).map(|_| ()),
            Err(BatchingError::Axis(AxisError::UnboundAxisName { name: "items".to_string() }).into()),
        );
        assert_eq!(
            batch(|x| function.call(x), inputs.clone(), BatchAxis::new(0), BatchAxis::new(0), items()),
            Ok(Array::vector(vec![0.0f64, 2.0, 6.0]).unwrap()),
        );
        let (_, program) = ArrayContext::trace(
            |x| Ok(batch(|x| function.call(x), x, BatchAxis::new(0), BatchAxis::new(0), items())?),
            ArrayType::new_static(DataType::F64, [3]),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[3] = custom_function [
                    name=\"custom_function\",
                    batching=[(extent=3, axis_name=\"items\", input_axes=[axis 0], output_axes=[axis 0])],
                ] %0 [
                    primal={
                        lambda %0:f64[3] .
                        let %1:u64[3] = iota [type=u64[3], dimension=0]
                            %2:f64[3] = convert_element_type [data_type=f64] %1
                            %3:f64[3] = mul %0 %2
                        in (%3)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );

        let function = custom_function(scaled)
            .with_jvp(move |x: Tracer, tangent: Tracer| Ok((scaled(x.clone())?, tangent * index(&x)?)));
        assert_eq!(
            differentiate_at(inputs.clone()).jvp(ones.clone(), |x| {
                Ok(batch(|x| function.call(x), x, BatchAxis::new(0), BatchAxis::new(0), items())?)
            }),
            Ok((Array::vector(vec![0.0f64, 2.0, 6.0]).unwrap(), Array::vector(vec![0.0f64, 1.0, 2.0]).unwrap())),
        );

        let function = custom_function(scaled).with_vjp(
            move |x: Tracer| Ok((scaled(x.clone())?, index(&x)?)),
            |index: Tracer, cotangent: Tracer| Ok(cotangent * index),
        );
        assert_eq!(
            differentiate_at(inputs.clone()).gradient(|x| {
                batch(|x| function.call(x), x, BatchAxis::new(0), BatchAxis::new(0), items())?
                    .reduce(&[0], ReductionKind::Sum)
            }),
            Ok(Array::vector(vec![0.0f64, 1.0, 2.0]).unwrap()),
        );

        let function = custom_function(|x: Tracer| x.parallel_reduce("items", ParallelReductionKind::Sum));
        assert_eq!(
            batch(|x| function.call(x), inputs, BatchAxis::new(0), BatchAxis::new(0), items()),
            Ok(Array::vector(vec![6.0f64; 3]).unwrap()),
        );
    }

    #[test]
    fn test_custom_function_with_batching_references() {
        // The primal `(counter, x) ↦ x` adds `x` into its plumbing counter. Structurally batching it with a replicated
        // counter and a mapped `x` is rejected, because a replicated reference cannot receive a batched value, while
        // the batching rule adds the sum of the batch into the counter instead.
        let primal = |(counter, x): (ArrayIrTracer, ArrayIrTracer)| {
            counter.add_update(&x)?;
            Ok(x)
        };
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let input_types = (reference_type, ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let extent = DimensionValue::constant(3).unwrap();
        let axes = [BatchAxis::replicated(), BatchAxis::new(0)];
        let batched = |program: FlatProgram<EagerArrayIrContext>| {
            program.batched_with_threaded_extent(
                extent.r#type().into_owned(),
                ShardingDimension::Replicated,
                &axes,
                ProgramBatchingOutputAxesPolicy::Natural,
            )
        };
        let function = custom_function(primal).with_non_differentiated_count(1);
        let (_, program) = EagerArrayIrContext::trace(|inputs| function.call(inputs), input_types.clone()).unwrap();
        assert!(matches!(
            batched(program.into_flat_program()),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "`reference_add_update` cannot store a batched value into an unbatched reference; pass \
                               the reference as a batched input instead",
        ));

        let function = custom_function(primal).with_non_differentiated_count(1).with_batching(
            |_: BatchingLevelExtent<ArrayIrTracer>,
             (counter, x): (ArrayIrTracer, ArrayIrTracer),
             (_, x_axis): (BatchAxis, BatchAxis)| {
                let sum = ValueProjection::<ArrayType>::into_projected(x.clone())?.reduce(&[0], ReductionKind::Sum)?;
                counter.add_update(&sum.into_value())?;
                Ok((x, x_axis))
            },
        );
        let (_, program) = EagerArrayIrContext::trace(|inputs| function.call(inputs), input_types).unwrap();
        let batched = batched(program.into_flat_program()).unwrap().into_parts().0;
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:dimension<3>, %1:ref<f32[]>, %2:f32[3] .
                let %3:f32[3] = custom_function [
                    name=\"custom_function\",
                    non_differentiated_count=2,
                    batching=[(extent=dimension<3>, input_axes=[replicated, axis 0], output_axes=[axis 0])],
                ] %0 %1 %2 [
                    primal={
                        lambda %0:dimension<3>, %1:ref<f32[]>, %2:f32[3] .
                        let %3:f32[] = reduce [kind=sum, axes=[0]] %2
                            () = reference_add_update %1 %3
                        in (%2)
                    },
                ]
                in (%0, %3)
            "}
            .trim_end(),
        );
        let counter = ArrayReference::new(Array::scalar(1.0f32).unwrap());
        assert_eq!(
            batched.interpret(vec![
                ArrayIrValue::Dimension(extent.clone()),
                ArrayIrValue::Reference(counter.clone()),
                ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()),
            ]),
            Ok(vec![
                ArrayIrValue::Dimension(extent),
                ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()),
            ]),
        );
        assert_eq!(counter.read(), Ok(Array::scalar(7.0f32).unwrap()));
    }

    #[test]
    fn test_custom_function_with_batching_differentiation() {
        // A function with only a batching rule (i.e., bare `custom_vmap`) differentiates its primal in both modes.
        type Tracer = DomainTracer<ArrayContext>;
        let function = custom_function(|x: Tracer| Ok(x.sin()?))
            .with_batching(|_: BatchingLevelExtent<Tracer>, x: Tracer, axis: BatchAxis| Ok((x.sin()?, axis)));
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| function.call(x)),
            Ok((Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(2.0f64.cos()).unwrap())),
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).gradient(|x| function.call(x)),
            Ok(Array::scalar(2.0f64.cos()).unwrap()),
        );
    }

    /// Returns `sin` with a custom batching rule that computes `2 · sin(x)` instead (expressed through addition to
    /// avoid constant lifting), which makes visible whether a batched computation and each of its derivatives go
    /// through the rule. The rule maps its output exactly when its input is mapped.
    fn doubled_sine_when_batched() -> CustomFunction<
        DomainTracer<ArrayContext>,
        DomainTracer<ArrayContext>,
        impl Fn(DomainTracer<ArrayContext>) -> Result<DomainTracer<ArrayContext>, ProgramError>,
        DefaultJvp,
        DefaultVjp,
        WithBatching<
            DomainTracer<ArrayContext>,
            impl 'static
            + Fn(
                BatchingLevelExtent<DomainTracer<ArrayContext>>,
                DomainTracer<ArrayContext>,
                BatchAxis,
            ) -> Result<(DomainTracer<ArrayContext>, BatchAxis), ProgramError>
            + Send
            + Sync,
        >,
    > {
        type Tracer = DomainTracer<ArrayContext>;
        custom_function(|x: Tracer| Ok(x.sin()?))
            .with_name("sine")
            .with_batching(|_: BatchingLevelExtent<Tracer>, x: Tracer, axis: BatchAxis| Ok((x.sin()? + x.sin()?, axis)))
    }

    /// Returns `x · y` with a custom batching rule that computes `2 · x · y` instead (refer to
    /// [`doubled_sine_when_batched`]), whose output is mapped along the first mapped input axis.
    fn doubled_product_when_batched() -> CustomFunction<
        (DomainTracer<ArrayContext>, DomainTracer<ArrayContext>),
        DomainTracer<ArrayContext>,
        impl Fn(
            (DomainTracer<ArrayContext>, DomainTracer<ArrayContext>),
        ) -> Result<DomainTracer<ArrayContext>, ProgramError>,
        DefaultJvp,
        DefaultVjp,
        WithBatching<
            DomainTracer<ArrayContext>,
            impl 'static
            + Fn(
                BatchingLevelExtent<DomainTracer<ArrayContext>>,
                (DomainTracer<ArrayContext>, DomainTracer<ArrayContext>),
                (BatchAxis, BatchAxis),
            ) -> Result<(DomainTracer<ArrayContext>, BatchAxis), ProgramError>
            + Send
            + Sync,
        >,
    > {
        type Tracer = DomainTracer<ArrayContext>;
        custom_function(|(x, y): (Tracer, Tracer)| Ok(x * y)).with_name("product").with_batching(
            |_: BatchingLevelExtent<Tracer>, (x, y): (Tracer, Tracer), (x_axis, y_axis): (BatchAxis, BatchAxis)| {
                let product = x * y;
                Ok((product.clone() + product, if x_axis.is_replicated() { y_axis } else { x_axis }))
            },
        )
    }

    /// Returns the elementwise values `2 · sin(x)` and `2 · cos(x)` of `x`.
    fn doubled_sine_and_cosine(x: &[f64]) -> (Array, Array) {
        (
            Array::vector(x.iter().map(|x| 2.0 * f64::sin(*x)).collect()).unwrap(),
            Array::vector(x.iter().map(|x| 2.0 * f64::cos(*x)).collect()).unwrap(),
        )
    }

    #[test]
    fn test_custom_function_with_batching_batch_then_differentiate() {
        // Batching the call applies its batching rule, so differentiating the batched call differentiates the rule's
        // program in both modes (i.e., `d(2 · sin(x)) = 2 · cos(x) · ẋ`).
        let function = doubled_sine_when_batched();
        let (_, program) = ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64)).unwrap();
        let batched = program
            .into_flat_program()
            .batched(3, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts()
            .0;
        let x = [0.5, 1.0, 1.5];
        let (doubled_sine, doubled_cosine) = doubled_sine_and_cosine(&x);
        let jvp = batched.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3] .
                let %2:f64[3] = sin %0
                    %3:f64[3] = cos %0
                    %4:f64[3] = mul %3 %1
                    %5:f64[3] = sin %0
                    %6:f64[3] = cos %0
                    %7:f64[3] = mul %6 %1
                    %8:f64[3] = add %2 %5
                    %9:f64[3] = add %4 %7
                in (%8, %9)
            "}
            .trim_end(),
        );
        let ones = Array::vector(vec![1.0; 3]).unwrap();
        assert_eq!(
            jvp.interpret(vec![Array::vector(x.to_vec()).unwrap(), ones.clone()]),
            Ok(vec![doubled_sine, doubled_cosine.clone()]),
        );
        let reverse = batched
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose)
            .unwrap();
        let mut pullback_inputs = vec![ones];
        pullback_inputs
            .extend(reverse.primal().interpret(vec![Array::vector(x.to_vec()).unwrap()]).unwrap().into_iter().skip(1));
        let pullback = reverse.tangent().transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![doubled_cosine]));
    }

    #[test]
    fn test_custom_function_with_batching_differentiate_then_batch() {
        // Differentiating an unbatched call stages a derived call whose primal region is the derivative of the call's
        // primal region, and batching it applies the derivative of the batching rule, whatever the order in which the
        // two transforms are applied.
        let function = doubled_sine_when_batched();
        let (_, program) = ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64)).unwrap();
        let jvp = program.into_flat_program().jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[], %3:f64[] = custom_function [name=\"jvp(sine)\"] %0 %1 [
                    primal={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = sin %0
                            %3:f64[] = cos %0
                            %4:f64[] = mul %3 %1
                        in (%2, %4)
                    },
                ]
                in (%2, %3)
            "}
            .trim_end(),
        );
        let x = [0.5, 1.0, 1.5];
        let (doubled_sine, doubled_cosine) = doubled_sine_and_cosine(&x);
        let ones = Array::vector(vec![1.0; 3]).unwrap();
        let (batched, output_axes) = jvp
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0), BatchAxis::new(0)]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3] .
                let %2:f64[3], %3:f64[3] = custom_function [
                    name=\"jvp(sine)\",
                    batching=[(extent=3, input_axes=[axis 0, axis 0], output_axes=[axis 0, axis 0])],
                ] %0 %1 [
                    primal={
                        lambda %0:f64[3], %1:f64[3] .
                        let %2:f64[3] = sin %0
                            %3:f64[3] = sin %0
                            %4:f64[3] = add %2 %3
                            %5:f64[3] = cos %0
                            %6:f64[3] = mul %5 %1
                            %7:f64[3] = cos %0
                            %8:f64[3] = mul %7 %1
                            %9:f64[3] = add %6 %8
                        in (%4, %9)
                    },
                ]
                in (%2, %3)
            "}
            .trim_end(),
        );
        assert_eq!(
            batched.interpret(vec![Array::vector(x.to_vec()).unwrap(), ones.clone()]),
            Ok(vec![doubled_sine.clone(), doubled_cosine.clone()]),
        );

        // Batching an eager derivative stages the derived call in the batching context directly.
        assert_eq!(
            batch(
                |(x, tangent)| Ok(differentiate_at(x).jvp(tangent, |x| function.call(x))?),
                (Array::vector(x.to_vec()).unwrap(), ones),
                (BatchAxis::new(0), BatchAxis::new(0)),
                (BatchAxis::new(0), BatchAxis::new(0)),
                None,
            ),
            Ok((doubled_sine, doubled_cosine)),
        );

        // Unbatched derivatives do not apply the batching rule.
        assert_eq!(
            differentiate_at(Array::scalar(0.5).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| function.call(x)),
            Ok((Array::scalar(0.5f64.sin()).unwrap(), Array::scalar(0.5f64.cos()).unwrap())),
        );
    }

    #[test]
    fn test_custom_function_with_batching_differentiate_then_batch_jacobian_forward() {
        // Linearization computes the outputs with the call itself and stages a pushforward call on its tangent side, so
        // the forward-mode Jacobian (which batches that pushforward over its tangents) applies the rule's derivative.
        let function = doubled_sine_when_batched();
        let x = [0.5, 1.0, 1.5];
        let (_, doubled_cosine) = doubled_sine_and_cosine(&x);
        let (_, program) =
            ArrayContext::trace(|x| function.call(x), ArrayType::new_static(DataType::F64, [3])).unwrap();
        let linearization = program.into_flat_program().linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[3] = custom_function [name=\"sine\"] %0 [
                    primal={
                        lambda %0:f64[3] .
                        let %1:f64[3] = sin %0
                        in (%1)
                    },
                ]
                in (%1, %0)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3] .
                let %2:f64[3] = custom_function [name=\"pushforward(sine)\"] %1 %0 [
                    primal={
                        lambda %0:f64[3], %1:f64[3] .
                        let %2:f64[3] = cos %0
                            %3:f64[3] = mul %2 %1
                        in (%3)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        let batched = linearization
            .tangent()
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        let tangents = Array::matrix(2, 3, vec![1.0, 1.0, 1.0, 0.5, 0.5, 0.5]).unwrap();
        let residual = Array::vector(x.to_vec()).unwrap();
        let doubled_cosine = doubled_cosine.to_f64s();
        assert_eq!(
            batched.interpret(vec![tangents, residual.clone()]),
            Ok(vec![
                Array::matrix(
                    2,
                    3,
                    doubled_cosine
                        .iter()
                        .chain(&doubled_cosine)
                        .enumerate()
                        .map(|(index, value)| { if index < 3 { *value } else { 0.5 * value } })
                        .collect()
                )
                .unwrap(),
            ]),
        );
        let jacobian = differentiate_at(residual).jacobian_forward(|x| function.call(x)).unwrap();
        assert_eq!(
            jacobian.values()[0],
            Array::matrix(
                3,
                3,
                (0..9).map(|index| if index % 4 == 0 { doubled_cosine[index / 4] } else { 0.0 }).collect()
            )
            .unwrap(),
        );
    }

    #[test]
    fn test_custom_function_with_batching_differentiate_then_batch_mixed_axes() {
        // Every combination of mapped and replicated primals and tangents batches through the rule's derivative. The
        // first two broadcast the replicated side of a pair with one mapped side, and the last two batch the derivative
        // of the rule applied to replicated inputs structurally, because no input is mapped on both sides.
        let function = doubled_product_when_batched();
        let scalar_type = ArrayType::scalar(DataType::F64);
        let (_, program) =
            ArrayContext::trace(|inputs| function.call(inputs), (scalar_type.clone(), scalar_type)).unwrap();
        let jvp = program.into_flat_program().jvp().unwrap();
        let mapped = BatchAxis::new(0);
        let replicated = BatchAxis::replicated();
        let values = |axis: BatchAxis, value: [f64; 3]| match axis.is_replicated() {
            true => (Array::scalar(value[0]).unwrap(), vec![value[0]; 3]),
            false => (Array::vector(value.to_vec()).unwrap(), value.to_vec()),
        };
        for axes in [
            [mapped, mapped, mapped, mapped],
            [mapped, mapped, mapped, replicated],
            [mapped, replicated, mapped, mapped],
            [replicated, replicated, mapped, mapped],
            [mapped, replicated, replicated, mapped],
        ] {
            let (x, x_values) = values(axes[0], [1.0, 2.0, 3.0]);
            let (y, y_values) = values(axes[1], [4.0, 5.0, 6.0]);
            let (x_tangent, x_tangent_values) = values(axes[2], [1.0, 0.5, 0.25]);
            let (y_tangent, y_tangent_values) = values(axes[3], [2.0, 3.0, 4.0]);
            let (batched, output_axes) = jvp
                .batched(3, ShardingDimension::Replicated, &axes, ProgramBatchingOutputAxesPolicy::Natural)
                .unwrap()
                .into_parts();
            // The output is replicated exactly when both primals are, and its tangent is always mapped.
            let output_axis = if axes[0].is_replicated() && axes[1].is_replicated() { replicated } else { mapped };
            assert_eq!(output_axes, vec![output_axis, mapped], "{axes:?}");
            let expected_outputs = (0..3).map(|index| 2.0 * x_values[index] * y_values[index]).collect::<Vec<_>>();
            let expected_outputs = match output_axis.is_replicated() {
                true => Array::scalar(expected_outputs[0]).unwrap(),
                false => Array::vector(expected_outputs).unwrap(),
            };
            let expected_tangents = (0..3)
                .map(|index| {
                    2.0 * (x_tangent_values[index] * y_values[index] + x_values[index] * y_tangent_values[index])
                })
                .collect();
            assert_eq!(
                batched.interpret(vec![x, y, x_tangent, y_tangent]),
                Ok(vec![expected_outputs, Array::vector(expected_tangents).unwrap()]),
                "{axes:?}",
            );
        }
    }

    #[test]
    fn test_custom_function_with_batching_differentiate_then_batch_flag_sensitive_rule() {
        // The rule computes `2xy` when both inputs are mapped and `3xy` otherwise. For a mapped `x` and a replicated
        // `y` whose tangent is mapped, the derived rule broadcasts `y` and applies the rule with both inputs mapped, so
        // the batched derivative returns `2xy` and `2(ẋy + xẏ)`, while batching the call alone returns `3xy`. JAX
        // returns `3xy` and the outer product of the flags' two input sets, whose diagonal is `3(ẋy + xẏ)` (i.e.,
        // `[18, 24, 39]` here). The two coincide for rules that do not depend on the flags (e.g., in
        // `test_custom_function_with_batching_differentiate_then_batch_mixed_axes`).
        type Tracer = DomainTracer<ArrayContext>;
        let function = custom_function(|(x, y): (Tracer, Tracer)| Ok(x * y)).with_batching(
            |_: BatchingLevelExtent<Tracer>, (x, y): (Tracer, Tracer), (x_axis, y_axis): (BatchAxis, BatchAxis)| {
                let product = x * y;
                let scale = if x_axis.is_replicated() || y_axis.is_replicated() { 3.0 } else { 2.0 };
                let scale = product.context().lift(Array::scalar(scale)?)?;
                Ok((product * scale, if x_axis.is_replicated() { y_axis } else { x_axis }))
            },
        );
        let scalar_type = ArrayType::scalar(DataType::F64);
        let (_, program) =
            ArrayContext::trace(|inputs| function.call(inputs), (scalar_type.clone(), scalar_type)).unwrap();
        let program = program.into_flat_program();
        let x = Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap();
        let y = Array::scalar(4.0f64).unwrap();
        assert_eq!(
            program
                .batched(
                    3,
                    ShardingDimension::Replicated,
                    &[BatchAxis::new(0), BatchAxis::replicated()],
                    ProgramBatchingOutputAxesPolicy::Natural,
                )
                .unwrap()
                .into_parts()
                .0
                .interpret(vec![x.clone(), y.clone()]),
            Ok(vec![Array::vector(vec![12.0f64, 24.0, 36.0]).unwrap()]),
        );
        let (batched, output_axes) = program
            .jvp()
            .unwrap()
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated(), BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0); 2]);
        assert_eq!(
            batched.interpret(vec![
                x,
                y,
                Array::vector(vec![1.0f64, 0.5, 0.25]).unwrap(),
                Array::vector(vec![2.0f64, 3.0, 4.0]).unwrap(),
            ]),
            Ok(vec![
                Array::vector(vec![8.0f64, 16.0, 24.0]).unwrap(),
                Array::vector(vec![12.0f64, 16.0, 26.0]).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_custom_function_with_batching_differentiate_then_batch_non_differentiated_inputs() {
        // A derived call receives no tangents for non-differentiated inputs, which keep their batch axes when the rule
        // is applied.
        let function = doubled_product_when_batched().with_non_differentiated_count(1);
        let scalar_type = ArrayType::scalar(DataType::F64);
        let (_, program) =
            ArrayContext::trace(|inputs| function.call(inputs), (scalar_type.clone(), scalar_type)).unwrap();
        let jvp = program.into_flat_program().jvp_with_respect_to(&[1]).unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[] .
                let %3:f64[], %4:f64[] = custom_function [name=\"jvp(product)\", non_differentiated_count=1] %0 %1 %2 [
                    primal={
                        lambda %0:f64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = mul %0 %1
                            %4:f64[] = mul %0 %2
                        in (%3, %4)
                    },
                ]
                in (%3, %4)
            "}
            .trim_end(),
        );
        let batched = jvp
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(
            batched.interpret(vec![
                Array::vector(vec![1.0, 2.0, 3.0]).unwrap(),
                Array::vector(vec![4.0, 5.0, 6.0]).unwrap(),
                Array::vector(vec![1.0, 1.0, 1.0]).unwrap(),
            ]),
            Ok(vec![Array::vector(vec![8.0, 20.0, 36.0]).unwrap(), Array::vector(vec![2.0, 4.0, 6.0]).unwrap(),]),
        );
    }

    #[test]
    fn test_custom_function_with_batching_differentiate_then_batch_zero_space_inputs() {
        // Zero-space inputs (e.g., integers) have no tangents, so a derived call receives none for them, while the rule
        // still receives them with their batch axes.
        type Tracer = DomainTracer<ArrayContext>;
        let function = custom_function(|(x, _): (Tracer, Tracer)| Ok(x.sin()?)).with_name("sine").with_batching(
            |_: BatchingLevelExtent<Tracer>, (x, _): (Tracer, Tracer), (axis, _): (BatchAxis, BatchAxis)| {
                Ok((x.sin()? + x.sin()?, axis))
            },
        );
        let (_, program) = ArrayContext::trace(
            |inputs| function.call(inputs),
            (ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::I32)),
        )
        .unwrap();
        let jvp = program.into_flat_program().jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[], %1:i32[], %2:f64[] .
                let %3:f64[], %4:f64[] = custom_function [name=\"jvp(sine)\"] %0 %1 %2 [
                    primal={
                        lambda %0:f64[], %1:i32[], %2:f64[] .
                        let %3:f64[] = sin %0
                            %4:f64[] = cos %0
                            %5:f64[] = mul %4 %2
                        in (%3, %5)
                    },
                ]
                in (%3, %4)
            "}
            .trim_end(),
        );
        let batched = jvp
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0); 3],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        let x = [0.5, 1.0, 1.5];
        let (doubled_sine, doubled_cosine) = doubled_sine_and_cosine(&x);
        assert_eq!(
            batched.interpret(vec![
                Array::vector(x.to_vec()).unwrap(),
                Array::vector(vec![1i32, 2, 3]).unwrap(),
                Array::vector(vec![1.0; 3]).unwrap(),
            ]),
            Ok(vec![doubled_sine, doubled_cosine]),
        );
    }

    #[test]
    fn test_custom_function_with_batching_differentiate_then_batch_non_batchable_primal() {
        // The primal's derivative calls a foreign kernel, which cannot be batched (nor executed by the reference
        // backend), so batching a derivative succeeds only through the derivative of the batching rule, in both modes.
        type Tracer = DomainTracer<ArrayContext>;
        let kernel = CustomFunction::from_custom_call(CustomCallOperation::new(
            "my_sin",
            vec![ArrayType::scalar(DataType::F64)],
        ))
        .with_jvp(|inputs: Vec<Tracer>, tangents: Vec<Tracer>| {
            let kernel = CustomCallOperation::new("my_sin", vec![ArrayType::scalar(DataType::F64)]);
            Ok((CustomCall::custom_call(&kernel, &inputs)?, vec![inputs[0].cos()? * tangents[0].clone()]))
        });
        let function = custom_function(move |x: Tracer| Ok(kernel.call(vec![x])?.remove(0)))
            .with_name("sine")
            .with_batching(|_: BatchingLevelExtent<Tracer>, x: Tracer, axis: BatchAxis| {
                Ok((x.sin()? + x.sin()?, axis))
            });
        let (_, program) = ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64)).unwrap();
        let jvp = program.into_flat_program().jvp().unwrap();
        let x = [0.5, 1.0, 1.5];
        let (doubled_sine, doubled_cosine) = doubled_sine_and_cosine(&x);
        for axes in [[BatchAxis::new(0), BatchAxis::new(0)], [BatchAxis::replicated(), BatchAxis::new(0)]] {
            let batched = jvp
                .batched(3, ShardingDimension::Replicated, &axes, ProgramBatchingOutputAxesPolicy::Natural)
                .unwrap()
                .into_parts()
                .0;
            let expected = match axes[0].is_replicated() {
                true => vec![
                    Array::scalar(2.0 * 0.5f64.sin()).unwrap(),
                    Array::vector(vec![2.0 * 0.5f64.cos(), 1.0 * 0.5f64.cos(), 0.5 * 0.5f64.cos()]).unwrap(),
                ],
                false => vec![doubled_sine.clone(), doubled_cosine.clone()],
            };
            let primal = match axes[0].is_replicated() {
                true => Array::scalar(0.5).unwrap(),
                false => Array::vector(x.to_vec()).unwrap(),
            };
            let tangents = match axes[0].is_replicated() {
                true => Array::vector(vec![1.0, 0.5, 0.25]).unwrap(),
                false => Array::vector(vec![1.0; 3]).unwrap(),
            };
            assert_eq!(batched.interpret(vec![primal, tangents]), Ok(expected), "{axes:?}");
        }
    }

    #[test]
    fn test_custom_function_with_batching_differentiate_then_batch_projection() {
        // Projections of composite contexts lift the regions of the member operations that they bind, so a function
        // written against a projection stages its derived call in the composite family.
        type Tracer = DomainTracer<ProjectedContext<EagerArrayIrContext, ArrayType>>;
        let function = custom_function(|x: Tracer| Ok(x.sin()?)).with_batching(
            |_: BatchingLevelExtent<Tracer>, x: Tracer, axis: BatchAxis| Ok((x.sin()? + x.sin()?, axis)),
        );
        let (_, program) = EagerArrayIrContext::trace(
            |x: ArrayIrTracer| {
                let x = ValueProjection::<ArrayType>::into_projected(x)?;
                let (value, tangent) = differentiate_at(x.clone()).jvp(x, |x| function.call(x))?;
                Ok::<_, ProgramError>((value.into_value(), tangent.into_value()))
            },
            ArrayIrType::Array(ArrayType::scalar(DataType::F64)),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[], %2:f64[] = custom_function [name=\"jvp(custom_function)\"] %0 %0 [
                    primal={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = sin %0
                            %3:f64[] = cos %0
                            %4:f64[] = mul %3 %1
                        in (%2, %4)
                    },
                ]
                in (%1, %2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_function_with_batching_differentiate_then_batch_output_axes() {
        // The derived batching rule keeps the batch axes that the rule declares, including nonzero ones, and moves a
        // tangent that is mapped along another axis than its primal to its primal's axis.
        let function = doubled_sine_when_batched();
        let (_, program) =
            ArrayContext::trace(|x| function.call(x), ArrayType::new_static(DataType::F64, [2])).unwrap();
        let jvp = program.into_flat_program().jvp().unwrap();
        let (batched, output_axes) = jvp
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(1), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(1), BatchAxis::new(1)]);
        let x = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let (doubled_sine, doubled_cosine) = doubled_sine_and_cosine(&x);
        let tangents = Array::matrix(3, 2, vec![1.0; 6]).unwrap();
        assert_eq!(
            batched.interpret(vec![Array::matrix(2, 3, x.to_vec()).unwrap(), tangents]),
            Ok(vec![
                Array::matrix(2, 3, doubled_sine.to_f64s()).unwrap(),
                Array::matrix(2, 3, doubled_cosine.to_f64s()).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_custom_function_with_batching_differentiate_then_batch_dynamic_extent() {
        // A first-class batch extent reaches the derived batching rule as the level's boundary operand, which the
        // source rule and the batched derivative both consume.
        type Tracer = DomainTracer<EagerArrayIrContext>;
        let sine = |x: Tracer| ValueProjection::<ArrayType>::into_projected(x)?.sin();
        let function = custom_function(move |x: Tracer| Ok(sine(x)?.into_value())).with_name("sine").with_batching(
            move |_: BatchingLevelExtent<Tracer>, x: Tracer, axis: BatchAxis| {
                Ok(((sine(x.clone())? + sine(x)?).into_value(), axis))
            },
        );
        let (_, program) =
            EagerArrayIrContext::trace(|x| function.call(x), ArrayIrType::Array(ArrayType::scalar(DataType::F64)))
                .unwrap();
        let jvp = program.into_flat_program().jvp().unwrap();
        let extent = DimensionValue::constant(3).unwrap();
        let (batched, output_axes) = jvp
            .batched_with_threaded_extent(
                extent.r#type().into_owned(),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0), BatchAxis::new(0)]);
        let x = [0.5, 1.0, 1.5];
        let (doubled_sine, doubled_cosine) = doubled_sine_and_cosine(&x);
        assert_eq!(
            batched.interpret(vec![
                ArrayIrValue::Dimension(extent.clone()),
                ArrayIrValue::Array(Array::vector(x.to_vec()).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![1.0; 3]).unwrap()),
            ]),
            Ok(vec![
                ArrayIrValue::Dimension(extent),
                ArrayIrValue::Array(doubled_sine),
                ArrayIrValue::Array(doubled_cosine),
            ]),
        );
    }

    #[test]
    fn test_custom_function_with_batching_differentiate_then_batch_shared_derivation() {
        // Calls whose primal regions differ share one derived definition, while each keeps its own primal region.
        let function = doubled_sine_when_batched();
        let (_, program) = ArrayContext::trace(
            |(x, y)| Ok((function.call(x)?, function.call(y)?)),
            (ArrayType::scalar(DataType::F64), ArrayType::new_static(DataType::F64, [2])),
        )
        .unwrap();
        let jvp = program.into_flat_program().jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[2], %2:f64[], %3:f64[2] .
                let %4:f64[], %5:f64[] = custom_function [name=\"jvp(sine)\"] %0 %2 [
                    primal={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = sin %0
                            %3:f64[] = cos %0
                            %4:f64[] = mul %3 %1
                        in (%2, %4)
                    },
                ]
                    %6:f64[2], %7:f64[2] = custom_function [name=\"jvp(sine)\"] %1 %3 [
                        primal={
                            lambda %0:f64[2], %1:f64[2] .
                            let %2:f64[2] = sin %0
                                %3:f64[2] = cos %0
                                %4:f64[2] = mul %3 %1
                            in (%2, %4)
                        },
                    ]
                in (%4, %6, %5, %7)
            "}
            .trim_end(),
        );

        // The rendering names the rules of both calls but cannot show that they share one derived definition, which
        // only their rule references record.
        let derived_rules = jvp
            .instructions()
            .iter()
            .map(|instruction| match instruction.operation() {
                ArrayOperation::CustomFunction(operation) => operation.rules().unwrap().clone(),
                operation => panic!("unexpected operation `{operation}`"),
            })
            .collect::<Vec<_>>();
        assert_eq!(derived_rules.len(), 2);
        assert_eq!(derived_rules[0], derived_rules[1]);
        assert_ne!(jvp.instructions()[0].regions()[0], jvp.instructions()[1].regions()[0]);
        let batched = jvp
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0); 4],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        let x = [0.5, 1.0, 1.5];
        let y = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let (doubled_sine_x, doubled_cosine_x) = doubled_sine_and_cosine(&x);
        let (doubled_sine_y, doubled_cosine_y) = doubled_sine_and_cosine(&y);
        assert_eq!(
            batched.interpret(vec![
                Array::vector(x.to_vec()).unwrap(),
                Array::matrix(3, 2, y.to_vec()).unwrap(),
                Array::vector(vec![1.0; 3]).unwrap(),
                Array::matrix(3, 2, vec![1.0; 6]).unwrap(),
            ]),
            Ok(vec![
                doubled_sine_x,
                Array::matrix(3, 2, doubled_sine_y.to_f64s()).unwrap(),
                doubled_cosine_x,
                Array::matrix(3, 2, doubled_cosine_y.to_f64s()).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_custom_function_with_batching_differentiate_then_batch_effectful_primal() {
        // A pushforward that recomputed this primal would repeat its reference update and read the updated counter, so
        // linearization inlines the derivative of an effectful primal instead: the effect runs once, when the known
        // side runs, and every pushforward sees the coefficient that the known side observed.
        type Tracer = DomainTracer<EagerArrayIrContext>;
        let scaled_by_incremented_counter = |(counter, x): (Tracer, Tracer)| {
            let one = counter.context().lift(ArrayIrValue::Array(Array::scalar(1.0)?))?;
            counter.add_update(&one)?;
            let scale = ValueProjection::<ArrayType>::into_projected(counter.read()?)?;
            Ok::<_, ProgramError>((scale * ValueProjection::<ArrayType>::into_projected(x)?).into_value())
        };
        let function = custom_function(scaled_by_incremented_counter).with_non_differentiated_count(1).with_batching(
            move |_: BatchingLevelExtent<Tracer>, inputs: (Tracer, Tracer), (_, axis): (BatchAxis, BatchAxis)| {
                Ok((scaled_by_incremented_counter(inputs)?, axis))
            },
        );
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F64)));
        let (_, program) = EagerArrayIrContext::trace(
            |inputs| function.call(inputs),
            (reference_type, ArrayIrType::Array(ArrayType::scalar(DataType::F64))),
        )
        .unwrap();
        let linearization = program.into_flat_program().linearize_with_respect_to(&[1]).unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:ref<f64[]>, %1:f64[] .
                let %2:f64[] = const 1.0
                    () = reference_add_update %0 %2
                    %3:f64[] = reference_read %0
                    %4:f64[] = mul %3 %1
                in (%4, %3)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %0
                in (%2)
            "}
            .trim_end(),
        );
        let counter = ArrayReference::new(Array::scalar(0.0).unwrap());
        let outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Reference(counter.clone()), ArrayIrValue::Array(Array::scalar(2.0).unwrap())])
            .unwrap();
        assert_eq!(counter.read(), Ok(Array::scalar(1.0).unwrap()));
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::scalar(1.0).unwrap())];
        tangent_inputs.extend(outputs.into_iter().skip(1));
        for _ in 0..2 {
            assert_eq!(
                linearization.tangent().interpret(tangent_inputs.clone()),
                Ok(vec![ArrayIrValue::Array(Array::scalar(1.0).unwrap())]),
            );
        }
        assert_eq!(counter.read(), Ok(Array::scalar(1.0).unwrap()));
    }

    #[test]
    fn test_custom_function_with_batching_differentiate_then_batch_higher_order() {
        // A derived call is differentiated like its source, so higher-order derivatives keep the rule on their batching
        // path. Nested batching applies the derived rule at every level, where the source rule batches the source
        // function at the batched types (i.e., the rule is not composed with itself), as for the source call.
        let function = doubled_sine_when_batched();
        let (_, program) = ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64)).unwrap();
        let second_order = program.into_flat_program().jvp().unwrap().jvp().unwrap();
        assert_eq!(
            second_order.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                let %4:f64[], %5:f64[], %6:f64[], %7:f64[] = custom_function [name=\"jvp(jvp(sine))\"] %0 %1 %2 %3 [
                    primal={
                        lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = sin %0
                            %5:f64[] = cos %0
                            %6:f64[] = mul %5 %2
                            %7:f64[] = cos %0
                            %8:f64[] = sin %0
                            %9:f64[] = mul %8 %2
                            %10:f64[] = neg %9
                            %11:f64[] = mul %7 %1
                            %12:f64[] = mul %1 %10
                            %13:f64[] = mul %7 %3
                            %14:f64[] = add %12 %13
                        in (%4, %11, %6, %14)
                    },
                ]
                in (%4, %5, %6, %7)
            "}
            .trim_end(),
        );
        let batched = second_order
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0); 4],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        let x = [0.5, 1.0, 1.5];
        let ones = Array::vector(vec![1.0; 3]).unwrap();
        let (doubled_sine, doubled_cosine) = doubled_sine_and_cosine(&x);
        let negated_doubled_sine = Array::vector(x.iter().map(|x| -2.0 * f64::sin(*x)).collect()).unwrap();
        assert_eq!(
            batched.interpret(vec![
                Array::vector(x.to_vec()).unwrap(),
                ones.clone(),
                ones.clone(),
                Array::vector(vec![0.0; 3]).unwrap()
            ]),
            Ok(vec![doubled_sine, doubled_cosine.clone(), doubled_cosine, negated_doubled_sine]),
        );

        let nested = ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64))
            .unwrap()
            .1
            .into_flat_program()
            .jvp()
            .unwrap()
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0); 2],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0); 2],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        let x = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let (doubled_sine, doubled_cosine) = doubled_sine_and_cosine(&x);
        assert_eq!(
            nested
                .interpret(vec![Array::matrix(2, 3, x.to_vec()).unwrap(), Array::matrix(2, 3, vec![1.0; 6]).unwrap()]),
            Ok(vec![
                Array::matrix(2, 3, doubled_sine.to_f64s()).unwrap(),
                Array::matrix(2, 3, doubled_cosine.to_f64s()).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_custom_function_with_batching_differentiate_then_batch_reverse_mode() {
        // Reverse mode inlines the derivative of the primal, since derived calls are not transposable, and a function
        // with reverse-mode rules keeps using them.
        let function = doubled_sine_when_batched();
        let (_, program) = ArrayContext::trace(|x| function.call(x), ArrayType::scalar(DataType::F64)).unwrap();
        let reverse = program
            .into_flat_program()
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose)
            .unwrap();
        assert_eq!(
            reverse.tangent().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %0
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            differentiate_at(Array::scalar(0.5).unwrap()).gradient(|x| function.call(x)),
            Ok(Array::scalar(0.5f64.cos()).unwrap()),
        );

        type Tracer = DomainTracer<ArrayContext>;
        let function = custom_function(|x: Tracer| Ok(x.sin()?))
            .with_jvp_from_primal()
            .with_vjp(
                |x| Ok((x.sin()?, x.cos()?)),
                |cosine, cotangent| Ok(cosine.clone() * cotangent.clone() + cosine * cotangent),
            )
            .with_batching(|_: BatchingLevelExtent<Tracer>, x: Tracer, axis: BatchAxis| {
                Ok((x.sin()? + x.sin()?, axis))
            });
        assert_eq!(
            differentiate_at(Array::scalar(0.5).unwrap()).gradient(|x| function.call(x)),
            Ok(Array::scalar(2.0 * 0.5f64.cos()).unwrap()),
        );
        assert_eq!(
            batch(
                |(x, tangent)| Ok(differentiate_at(x).jvp(tangent, |x| function.call(x))?),
                (Array::vector(vec![0.5, 1.0]).unwrap(), Array::vector(vec![1.0, 1.0]).unwrap()),
                (BatchAxis::new(0), BatchAxis::new(0)),
                (BatchAxis::new(0), BatchAxis::new(0)),
                None,
            ),
            Ok(doubled_sine_and_cosine(&[0.5, 1.0])),
        );
    }

    #[test]
    fn test_custom_function_call() {
        // Nothing but the primal is traced when the function is called. Calls that share a call structure share one
        // registered definition, whose rule is traced once per specialization and cached while the function lives.
        let traces = Arc::new(AtomicUsize::new(0));
        let function = custom_function(|x: DomainTracer<ArrayContext>| Ok(x.sin()?)).with_jvp({
            let traces = traces.clone();
            move |x, tangent| {
                traces.fetch_add(1, Ordering::SeqCst);
                Ok((x.sin()?, x.cos()? * tangent))
            }
        });
        let (_, program) =
            ArrayContext::trace(|x| function.call(function.call(x)?), ArrayType::scalar(DataType::F64)).unwrap();
        let program = program.into_flat_program();

        // Renderings name the rules of calls but not the definitions that they share, which only their rule
        // references record.
        let rule_ids = program
            .instructions()
            .iter()
            .map(|instruction| match instruction.operation() {
                ArrayOperation::CustomFunction(operation) => operation.rules().unwrap().id(),
                operation => panic!("unexpected operation `{operation}`"),
            })
            .collect::<Vec<_>>();
        assert_eq!(rule_ids.len(), 2);
        assert_eq!(rule_ids[0], rule_ids[1]);
        assert_eq!(traces.load(Ordering::SeqCst), 0);
        assert!(program.jvp().is_ok());
        assert_eq!(traces.load(Ordering::SeqCst), 1);
        assert!(program.jvp().is_ok());
        assert_eq!(traces.load(Ordering::SeqCst), 1);

        // A new input signature is a new specialization of the same definition.
        let (_, vector_program) =
            ArrayContext::trace(|x| function.call(x), ArrayType::new_static(DataType::F64, [2])).unwrap();
        assert!(vector_program.into_flat_program().jvp().is_ok());
        assert_eq!(traces.load(Ordering::SeqCst), 2);

        // Once the function is dropped, the calls that it staged trace their rule again on every derivative request.
        drop(function);
        assert!(program.jvp().is_ok());
        assert_eq!(traces.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn test_custom_function_call_structures() {
        // Each call structure registers its own definition, because the rules are adapted to the structured closures.
        let function =
            custom_function(|xs: Vec<DomainTracer<ArrayContext>>| Ok(xs)).with_jvp(|xs, tangents| Ok((xs, tangents)));
        let (_, program) = ArrayContext::trace(
            |x: DomainTracer<ArrayContext>| {
                let single = function.call(vec![x.clone()])?;
                let pair = function.call(vec![x.clone(), x.clone()])?;
                let other_single = function.call(vec![x])?;
                Ok((single, pair, other_single))
            },
            ArrayType::scalar(DataType::F64),
        )
        .unwrap();

        // Renderings name the rules of calls but not the definitions that they share, which only their rule
        // references record.
        let rule_ids = program
            .instructions()
            .iter()
            .map(|instruction| match instruction.operation() {
                ArrayOperation::CustomFunction(operation) => operation.rules().unwrap().id(),
                operation => panic!("unexpected operation `{operation}`"),
            })
            .collect::<Vec<_>>();
        assert_eq!(rule_ids.len(), 3);
        assert_ne!(rule_ids[0], rule_ids[1]);
        assert_eq!(rule_ids[0], rule_ids[2]);
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| {
                let pair = function.call(vec![x.clone(), x])?;
                Ok(pair[0].clone() + pair[1].clone())
            }),
            Ok((Array::scalar(4.0).unwrap(), Array::scalar(2.0).unwrap())),
        );
    }

    #[test]
    fn test_custom_function() {
        // A function without custom rules differentiates its primal in both modes and to higher orders, and it batches
        // its primal structurally.
        let function = custom_function(|x: DomainTracer<ArrayContext>| Ok(x.sin()?));
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| function.call(x)),
            Ok((Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(2.0f64.cos()).unwrap())),
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).gradient(|x| function.call(x)),
            Ok(Array::scalar(2.0f64.cos()).unwrap()),
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap())
                .gradient(|x| differentiate_at(x).gradient(|x| function.call(x)).unwrap()),
            Ok(Array::scalar(-2.0f64.sin()).unwrap()),
        );
        assert_eq!(
            batch(
                |x| function.call(x),
                Array::vector(vec![1.0, 2.0]).unwrap(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None
            ),
            Ok(Array::vector(vec![1.0f64.sin(), 2.0f64.sin()]).unwrap()),
        );
    }
}
