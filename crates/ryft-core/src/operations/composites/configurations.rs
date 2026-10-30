use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use crate::axes::NamedAxis;
use crate::batching::{BatchAxis, BatchableType, BatchingLevel, BatchingLevelExtent, DiagonalBatchingPolicy};
use crate::contexts::{EagerContext, StagingContext};
use crate::differentiation::{
    CotangentAccumulator, DifferentiableOperation, DifferentiableType, ResidualZeroProvider, TranspositionContext,
};
use crate::macros::check_count;
use crate::operations::arithmetic::AddOperation;
use crate::operations::composites::operations::COMPOSITE_OPERATION_NAME;
use crate::operations::composites::rules::{
    CustomRuleDefinition, CustomRuleTracer, CustomVjpOperation, derive_bijective_identity_renaming,
};
use crate::operations::control_flow::scan::{ScanOperation, ScanType};
use crate::parameters::{Parameter, Parameterized, ParameterizedFamily};
use crate::partial::{PartialEvaluationContext, PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, Operation, OperationProvider, ProgramError, ReferenceMemberType, Type, TypeError, Typed, Value,
};
use crate::tracing::{DomainTracingContext, TracingContext};

#[cfg(doc)]
use crate::operations::composites::builders::{Composite, composite};

/// Installs a forward mode differentiation configuration in a [`CustomRuleDefinition`]. A user-supplied rule is adapted
/// to the flat interface of retained rules by tracing it with the complete structured interface of the call at the
/// types of each specialization and replaying the traced program.
pub trait CompositeJvp<Jvp, Tracer: Parameter, Input: Parameterized<Tracer>, Output: Parameterized<Tracer>>:
    Sized
{
    /// Returns this definition with the provided forward mode differentiation configuration applied. [`DefaultJvp`]
    /// leaves the definition unchanged. Retained rules use the definition's name and named axes.
    ///
    /// # Parameters
    ///
    ///   - `configuration`: Forward mode differentiation configuration to apply.
    ///   - `input_structure`: Structure of the call's inputs.
    ///   - `output_structure`: Structure of the call's outputs.
    ///   - `non_differentiated_count`: Number of leading non-differentiated input leaves.
    fn configure_jvp(
        self,
        configuration: &Jvp,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
        non_differentiated_count: usize,
    ) -> Self;
}

/// Forward mode configuration of a [`Composite`] without a configured forward mode differentiation rule. A function
/// without reverse mode rules then derives its forward mode rule from its primal (as [`JvpFromPrimal`] does), while a
/// function with reverse mode rules rejects forward mode differentiation. Refer to the documentation of [`composite`]
/// for more information on rule selection.
#[derive(Copy, Clone, Debug, Default)]
pub struct DefaultJvp;

impl<V: Value, O: Operation<Type = V::Type>, Input, Output>
    CompositeJvp<DefaultJvp, CustomRuleTracer<V, O>, Input, Output> for CustomRuleDefinition<V, O>
where
    Input: Parameterized<CustomRuleTracer<V, O>>,
    Output: Parameterized<CustomRuleTracer<V, O>>,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    #[inline]
    fn configure_jvp(
        self,
        _configuration: &DefaultJvp,
        _input_structure: &Input::ParameterStructure,
        _output_structure: &Output::ParameterStructure,
        _non_differentiated_count: usize,
    ) -> Self {
        self
    }
}

/// Forward mode differentiation configuration of a [`Composite`] with an explicit, user-supplied Jacobian-Vector
/// Product (JVP) rule closure implementing `(x, ẋ) ↦ (y, ẏ)`. Refer to [`Composite::with_jvp`] for more information.
pub struct ExplicitJvp<Input, Output> {
    /// Closure computing `(outputs, output_tangents)` from the primal input value and the input tangent value.
    pub(super) jvp: Arc<dyn Fn(Input, Input) -> Result<(Output, Output), ProgramError> + Send + Sync>,
}

impl<Input, Output> Debug for ExplicitJvp<Input, Output> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("ExplicitJvp").finish_non_exhaustive()
    }
}

impl<V, O, Input, Output> CompositeJvp<ExplicitJvp<Input, Output>, CustomRuleTracer<V, O>, Input, Output>
    for CustomRuleDefinition<V, O>
where
    V: 'static + Value<Type: DifferentiableType + Eq + Hash>,
    O: 'static + Operation<Type = V::Type>,
    Input: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Send + Sync>,
    Input::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Input::To<V::Type>: Clone + Parameterized<V::Type, Family = Input::Family, To<CustomRuleTracer<V, O>> = Input>,
    Output: 'static + Parameterized<CustomRuleTracer<V, O>>,
    Output::ParameterStructure: Debug + PartialEq + Send + Sync,
    Output::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    fn configure_jvp(
        self,
        configuration: &ExplicitJvp<Input, Output>,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
        non_differentiated_count: usize,
    ) -> Self {
        let (jvp, name, named_axes) = (configuration.jvp.clone(), self.name.clone(), self.named_axes().to_vec());
        let (input_structure, output_structure) = (input_structure.clone(), output_structure.clone());
        self.with_jvp(move |primals, tangents| {
            let input_types = Input::To::<V::Type>::from_parameters(
                input_structure.clone(),
                primals.iter().map(|primal| primal.r#type().into_owned()),
            )?;
            let program = {
                let name: &str = &name;
                let jvp = jvp.as_ref();
                let output_structure = &output_structure;
                let named_axes = named_axes.clone();
                let input_count = input_types.parameter_count();
                let input_tangent_types = input_types.clone().try_map_parameters(|r#type| r#type.tangent())?;
                let ((output_types, tangent_types), program) =
                    DomainTracingContext::<EagerContext<V, O>>::trace_with_named_axes(
                        |(x, t)| jvp(x, t),
                        (input_types, input_tangent_types),
                        named_axes,
                    )?;

                // Matching flat types cannot establish the nested output and tangent structures.
                for (value, rule_structure) in [
                    ("output", output_types.parameter_structure()),
                    ("output tangent", tangent_types.parameter_structure()),
                ] {
                    if &rule_structure != output_structure {
                        return Err(TypeError::invalid(format!(
                            "{} JVP rule {} structure mismatch: expected {:?} but got {:?}",
                            if name == COMPOSITE_OPERATION_NAME {
                                format!("`{COMPOSITE_OPERATION_NAME}`")
                            } else {
                                format!("`{COMPOSITE_OPERATION_NAME}` `{name}`")
                            },
                            value,
                            output_structure,
                            rule_structure,
                        ))
                        .into());
                    }
                }
                let program = program.into_flat_program();

                // Non-differentiated tangent slots are tracing placeholders that the rule must leave unused.
                // Remove them so the retained program accepts only differentiated input tangents.
                if non_differentiated_count == 0 {
                    program
                } else {
                    let input_ids = program.input_ids();
                    check_count!("input", input_ids, 2 * input_count, ProgramError);
                    let (primal_ids, tangent_ids) = input_ids.split_at(input_count);
                    let (placeholder_ids, differentiated_tangent_ids) = tangent_ids.split_at(non_differentiated_count);
                    for (index, placeholder) in placeholder_ids.iter().enumerate() {
                        // Attached regions are closed over their own inputs, so a use of an entry input is always
                        // a direct input or output of the entry region.
                        let used = program.output_ids().contains(placeholder)
                            || program
                                .instructions()
                                .iter()
                                .any(|instruction| instruction.inputs().contains(placeholder));
                        if used {
                            return Err(TypeError::invalid(format!(
                                "`{name}` rule uses the tangent of leading non-differentiated input {index}, which has \
                                 no tangent slot because non-differentiated inputs parameterize the rule without being \
                                 differentiated",
                            ))
                            .into());
                        }
                    }
                    let kept_ids = primal_ids.iter().chain(differentiated_tangent_ids).copied().collect::<Vec<_>>();
                    let (pruned, _) =
                        program.filtered(kept_ids.as_slice(), program.output_ids(), kept_ids.as_slice())?;
                    pruned
                }
            };

            let mut values = primals.to_vec();
            values.extend_from_slice(tangents);
            let mut outputs = program.interpret_in_context(primals[0].context(), values)?;
            let output_tangents = outputs.split_off(output_structure.parameter_count());
            Ok((outputs, output_tangents))
        })
    }
}

/// Forward mode differentiation configuration of a [`Composite`] whose forward mode differentiation rule is derived
/// from its primal. Refer to [`Composite::with_jvp_from_primal`] for more information.
#[derive(Copy, Clone, Debug, Default)]
pub struct JvpFromPrimal;

impl<V: Value<Type: Eq + Hash>, O: Operation<Type = V::Type>, Input, Output>
    CompositeJvp<JvpFromPrimal, CustomRuleTracer<V, O>, Input, Output> for CustomRuleDefinition<V, O>
where
    Input: Parameterized<CustomRuleTracer<V, O>>,
    Output: Parameterized<CustomRuleTracer<V, O>>,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    #[inline]
    fn configure_jvp(
        self,
        _configuration: &JvpFromPrimal,
        _input_structure: &Input::ParameterStructure,
        _output_structure: &Output::ParameterStructure,
        _non_differentiated_count: usize,
    ) -> Self {
        self.with_jvp_from_primal()
    }
}

/// Forward mode differentiation configuration of a [`Composite`] with a user-supplied Jacobian-Vector Product (JVP)
/// rule closure implementing `(x, ẋ) ↦ (y, ẏ)` that receives `ẋ` and returns `ẏ` with [`MaybeZero`] leaves, of which
/// the structurally zero ones are [`MaybeZero::Zero`]s. `Tracer` is the tracer type of the closure's leaves, which the
/// leaves of its tangents wrap. Refer to [`Composite::with_symbolic_zero_jvp`] for more information.
pub struct SymbolicZeroJvp<Tracer: Typed + Parameter, Input, Output>
where
    Input: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
    Output: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
{
    /// Closure computing `(outputs, output_tangents)` from the primal input value and the input tangent value.
    #[allow(clippy::type_complexity)]
    pub(super) jvp: Arc<
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
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("SymbolicZeroJvp").finish_non_exhaustive()
    }
}

impl<V, O, Input, Output>
    CompositeJvp<SymbolicZeroJvp<CustomRuleTracer<V, O>, Input, Output>, CustomRuleTracer<V, O>, Input, Output>
    for CustomRuleDefinition<V, O>
where
    V: 'static + Value<Type: DifferentiableType + Eq + Hash>,
    O: 'static + Operation<Type = V::Type>,
    Input: 'static + Parameterized<CustomRuleTracer<V, O>, ParameterStructure: Send + Sync>,
    Input::Family:
        ParameterizedFamily<V::Type> + ParameterizedFamily<V> + ParameterizedFamily<MaybeZero<CustomRuleTracer<V, O>>>,
    Input::To<V::Type>: Clone + Parameterized<V::Type, Family = Input::Family, To<CustomRuleTracer<V, O>> = Input>,
    Output: 'static + Parameterized<CustomRuleTracer<V, O>>,
    Output::ParameterStructure: Debug + PartialEq + Send + Sync,
    Output::Family:
        ParameterizedFamily<V::Type> + ParameterizedFamily<V> + ParameterizedFamily<MaybeZero<CustomRuleTracer<V, O>>>,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    fn configure_jvp(
        self,
        configuration: &SymbolicZeroJvp<CustomRuleTracer<V, O>, Input, Output>,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
        non_differentiated_count: usize,
    ) -> Self {
        let (jvp, name, named_axes) = (configuration.jvp.clone(), self.name.clone(), self.named_axes().to_vec());
        let (input_structure, output_structure) = (input_structure.clone(), output_structure.clone());
        self.with_symbolic_zero_jvp(move |primals, tangents| {
            let input_types = Input::To::<V::Type>::from_parameters(
                input_structure.clone(),
                primals.iter().map(|primal| primal.r#type().into_owned()),
            )?;
            let tangent_activity = tangents.iter().map(|tangent| !tangent.is_zero()).collect::<Vec<_>>();
            let (program, output_tangent_zeros) = {
                let name: &str = &name;
                let jvp = jvp.as_ref();
                let tangent_activity = tangent_activity.as_slice();
                let output_structure = &output_structure;
                let named_axes = named_axes.clone();

                // Only active differentiated tangents cross the traced boundary. Non-differentiated and inactive
                // tangents remain structural zeros inside the rule.
                let input_structure = input_types.parameter_structure();
                let tangent_types =
                    input_types.parameters().map(|r#type| r#type.tangent()).collect::<Result<Vec<_>, _>>()?;
                let active_tangent_types = tangent_types[non_differentiated_count..]
                    .iter()
                    .zip(tangent_activity)
                    .filter(|&(r#type, active)| *active)
                    .map(|(r#type, active)| r#type.clone())
                    .collect::<Vec<_>>();
                let mut output_tangent_zeros = None;
                let ((output_types, _), program) = DomainTracingContext::<EagerContext<V, O>>::trace_with_named_axes(
                    |(inputs, active_tangents): (Input, Vec<CustomRuleTracer<V, O>>)| {
                        let mut active_tangents = active_tangents.into_iter();
                        let leaves = tangent_types.iter().enumerate().map(|(index, r#type)| {
                            let active = index
                                .checked_sub(non_differentiated_count)
                                .is_some_and(|index| tangent_activity[index]);
                            match active {
                                true => MaybeZero::Value(active_tangents.next().unwrap()),
                                false => MaybeZero::Zero(r#type.clone()),
                            }
                        });
                        let (outputs, output_tangents) = jvp(
                            inputs,
                            Input::To::<MaybeZero<CustomRuleTracer<V, O>>>::from_parameters(
                                input_structure.clone(),
                                leaves,
                            )?,
                        )?;
                        let output_tangent_structure = output_tangents.parameter_structure();
                        if &output_tangent_structure != output_structure {
                            return Err(TypeError::invalid(format!(
                                "{} JVP rule output tangent structure mismatch: expected {:?} but got {:?}",
                                if name == COMPOSITE_OPERATION_NAME {
                                    format!("`{COMPOSITE_OPERATION_NAME}`")
                                } else {
                                    format!("`{COMPOSITE_OPERATION_NAME}` `{name}`")
                                },
                                output_structure,
                                output_tangent_structure,
                            ))
                            .into());
                        }

                        // Only the live output tangents become program outputs, while each structural zero must have
                        // the tangent type of its output, because the replay reinserts it with that type.
                        let mut zeros = Vec::new();
                        let mut live_output_tangents = Vec::new();
                        for (index, (output, tangent)) in
                            outputs.parameters().zip(output_tangents.into_parameters()).enumerate()
                        {
                            match tangent {
                                MaybeZero::Value(tangent) => {
                                    zeros.push(false);
                                    live_output_tangents.push(tangent);
                                }
                                MaybeZero::Zero(r#type) => {
                                    let expected = output.r#type().tangent()?;
                                    if r#type != expected {
                                        return Err(TypeError::invalid(format!(
                                            "{} JVP rule returned a structural-zero tangent of type `{}` \
                                             for output {}, whose tangent type is `{}`",
                                            if name == COMPOSITE_OPERATION_NAME {
                                                format!("`{COMPOSITE_OPERATION_NAME}`")
                                            } else {
                                                format!("`{COMPOSITE_OPERATION_NAME}` `{name}`")
                                            },
                                            r#type,
                                            index,
                                            expected,
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
                    return Err(TypeError::invalid(format!(
                        "{} JVP rule output structure mismatch: expected {:?} but got {:?}",
                        if name == COMPOSITE_OPERATION_NAME {
                            format!("`{COMPOSITE_OPERATION_NAME}`")
                        } else {
                            format!("`{COMPOSITE_OPERATION_NAME}` `{name}`")
                        },
                        output_structure,
                        output_type_structure,
                    ))
                    .into());
                }

                // The closure ran exactly once, so the structural zeros are always recorded.
                (program.into_flat_program(), output_tangent_zeros.unwrap())
            };

            let mut values = primals.to_vec();
            values.extend(tangents.iter().filter_map(MaybeZero::as_value).cloned());
            let mut outputs = program.interpret_in_context(primals[0].context(), values)?;
            let mut live_output_tangents = outputs.split_off(output_structure.parameter_count()).into_iter();

            // The traced program returns only the live output tangents, so the structural zeros are reinserted
            // at their positions, typed by the tangent types of the replayed outputs.
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

/// Installs a reverse mode differentiation configuration in a [`CustomRuleDefinition`]. User-supplied rules are adapted
/// to the flat interface of retained rules as described for [`CompositeJvp`]. The backward rule receives the residuals
/// with the structure that the forward rule returned for the same specialization. Different specializations of one call
/// structure may return differently structured residuals. Residuals whose flat types are equal up to a renaming of
/// their type identities (e.g., their dimension variables) must always have the same structure. The forward rule
/// is rejected otherwise.
pub trait CompositeVjp<Vjp, Tracer: Parameter, Input: Parameterized<Tracer>, Output: Parameterized<Tracer>>:
    Sized
{
    /// Returns this definition with the provided reverse mode differentiation configuration applied. [`DefaultVjp`]
    /// leaves the definition unchanged. Retained rules use the definition's name and named axes.
    ///
    /// # Parameters
    ///
    ///   - `configuration`: Reverse mode differentiation configuration to apply.
    ///   - `input_structure`: Structure of the call's inputs.
    ///   - `output_structure`: Structure of the call's outputs.
    ///   - `non_differentiated_count`: Number of leading non-differentiated input leaves.
    fn configure_vjp(
        self,
        configuration: &Vjp,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
        non_differentiated_count: usize,
    ) -> Self;
}

/// Reverse mode differentiation configuration of a [`Composite`] without configured reverse mode differentiation
/// rules. Reverse mode differentiation then transposes the forward mode rule, which is derived from the primal when
/// none is configured. Refer to the documentation of [`composite`] for more information on rule selection.
#[derive(Copy, Clone, Debug, Default)]
pub struct DefaultVjp;

impl<V: Value, O: Operation<Type = V::Type>, Input, Output>
    CompositeVjp<DefaultVjp, CustomRuleTracer<V, O>, Input, Output> for CustomRuleDefinition<V, O>
where
    Input: Parameterized<CustomRuleTracer<V, O>>,
    Output: Parameterized<CustomRuleTracer<V, O>>,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    #[inline]
    fn configure_vjp(
        self,
        _configuration: &DefaultVjp,
        _input_structure: &Input::ParameterStructure,
        _output_structure: &Output::ParameterStructure,
        _non_differentiated_count: usize,
    ) -> Self {
        self
    }
}

/// Reverse mode differentiation configuration of a [`Composite`] with explicit, user-supplied forward and backward
/// rule closures implementing `x ↦ (y, r)` and the Vector-Jacobian Product (VJP) `(r, ȳ) ↦ x̄`, where `r` is a value
/// of type `Residual`. Refer to [`Composite::with_vjp`] for more information.
pub struct ExplicitVjp<Input, Output, Residual> {
    /// Closure computing `(outputs, residuals)` from the primal input value.
    pub(super) forward: Arc<dyn Fn(Input) -> Result<(Output, Residual), ProgramError> + Send + Sync>,

    /// Closure computing the input cotangent value from `(residuals, output_cotangents)`.
    pub(super) backward: Arc<dyn Fn(Residual, Output) -> Result<Input, ProgramError> + Send + Sync>,

    /// Whether reverse mode differentiation defers `forward` to the backward pass. Refer to
    /// [`Composite::with_deferred_forward`] for more information.
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

impl<V, O, Input, Output, Residual>
    CompositeVjp<ExplicitVjp<Input, Output, Residual>, CustomRuleTracer<V, O>, Input, Output>
    for CustomRuleDefinition<V, O>
where
    V: 'static + Value<Type: Eq + Hash + Send + DifferentiableType + ReferenceMemberType>,
    O: 'static + CustomVjpOperation<V>,
    Input: 'static + Parameterized<CustomRuleTracer<V, O>>,
    Input::ParameterStructure: Debug + PartialEq + Send + Sync,
    Input::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Input::To<V::Type>: Clone + Parameterized<V::Type, Family = Input::Family, To<CustomRuleTracer<V, O>> = Input>,
    Output: 'static + Parameterized<CustomRuleTracer<V, O>>,
    Output::ParameterStructure: Debug + PartialEq + Send + Sync,
    Output::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Output::To<V::Type>: Parameterized<V::Type, Family = Output::Family, To<CustomRuleTracer<V, O>> = Output>,
    Residual: 'static + Parameterized<CustomRuleTracer<V, O>>,
    Residual::ParameterStructure: Debug + PartialEq + Send,
    Residual::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Residual::To<V::Type>: Parameterized<V::Type, Family = Residual::Family, To<CustomRuleTracer<V, O>> = Residual>,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    fn configure_vjp(
        self,
        configuration: &ExplicitVjp<Input, Output, Residual>,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
        non_differentiated_count: usize,
    ) -> Self {
        let residual_structures = CompositeResidualStructures::<V::Type, Residual::ParameterStructure>::default();
        let forward = retained_custom_vjp_forward_rule::<V, O, Input, Output, Residual, _>(
            configuration.forward.clone(),
            self.name.clone(),
            input_structure,
            output_structure,
            &residual_structures,
            self.named_axes(),
        );
        let backward = {
            let (backward, name, named_axes) =
                (configuration.backward.clone(), self.name.clone(), self.named_axes().to_vec());
            let (input_structure, output_structure) = (input_structure.clone(), output_structure.clone());
            move |leading_inputs: &[CustomRuleTracer<V, O>], seeds: &[CustomRuleTracer<V, O>]| {
                let residual_types = leading_inputs[non_differentiated_count..]
                    .iter()
                    .map(|value| value.r#type().into_owned())
                    .collect::<Vec<_>>();
                let structure = residual_structures.get(&name, &residual_types)?;
                let Some(context) = leading_inputs.iter().chain(seeds).next().map(|value| value.context().clone())
                else {
                    return Err(ProgramError::MalformedProgram(format!(
                        "`{name}` backward rule has neither leading inputs nor seeds",
                    )));
                };
                let program = {
                    let name: &str = &name;
                    let backward = backward.as_ref();
                    let input_structure = &input_structure;
                    let non_differentiated_types = leading_inputs[..non_differentiated_count]
                        .iter()
                        .map(|value| value.r#type().into_owned())
                        .collect::<Vec<_>>();
                    let residual_structure = structure;
                    let output_structure = output_structure.clone();
                    let output_cotangent_types =
                        seeds.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
                    let named_axes = named_axes.clone();
                    let non_differentiated_count = non_differentiated_types.len();
                    let residual_types = Residual::To::<V::Type>::from_parameters(residual_structure, residual_types)?;
                    let output_cotangent_types =
                        Output::To::<V::Type>::from_parameters(output_structure, output_cotangent_types)?;

                    // Keep the non-differentiated inputs as unused leading program inputs. The backward
                    // closure obtains required primal values from its residuals.
                    let (_, program) = DomainTracingContext::<EagerContext<V, O>>::trace_with_named_axes(
                        |(_, residuals, cotangents): (Vec<CustomRuleTracer<V, O>>, Residual, Output)| {
                            let cotangents = backward(residuals, cotangents)?;

                            // Check the full input structure before dropping the non-differentiated prefix. Otherwise,
                            // a misplaced cotangent could silently become the derivative of a different input with the
                            // same leaf type.
                            let cotangent_structure = cotangents.parameter_structure();
                            if &cotangent_structure != input_structure {
                                return Err(TypeError::invalid(format!(
                                    "{} backward rule cotangent structure mismatch: expected {:?} but got {:?}",
                                    if name == COMPOSITE_OPERATION_NAME {
                                        format!("`{COMPOSITE_OPERATION_NAME}`")
                                    } else {
                                        format!("`{COMPOSITE_OPERATION_NAME}` `{name}`")
                                    },
                                    input_structure,
                                    cotangent_structure,
                                ))
                                .into());
                            }
                            Ok(cotangents.into_parameters().skip(non_differentiated_count).collect::<Vec<_>>())
                        },
                        (non_differentiated_types, residual_types, output_cotangent_types),
                        named_axes,
                    )?;
                    program.into_flat_program()
                };
                let mut values = leading_inputs.to_vec();
                values.extend_from_slice(seeds);
                program.interpret_in_context(&context, values)
            }
        };
        let definition = self.with_vjp(forward, backward);
        if configuration.deferred_forward { definition.with_deferred_forward_rule() } else { definition }
    }
}

/// Reverse mode differentiation configuration of a [`Composite`] with user-supplied forward and backward rule closures
/// implementing `x ↦ (y, r)` and the Vector-Jacobian Product (VJP) `(r, ȳ) ↦ x̄`. The backward rule receives `ȳ` and
/// returns `x̄` with [`MaybeZero`] leaves, of which the structurally zero ones are [`MaybeZero::Zero`]s. `Tracer` is the
/// tracer type of the closures' leaves, which the leaves of the seeds and cotangents wrap. Refer to
/// [`Composite::with_symbolic_zero_vjp`] for more information.
pub struct SymbolicZeroVjp<Tracer: Typed + Parameter, Input, Output, Residual>
where
    Input: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
    Output: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
{
    /// Closure computing `(outputs, residuals)` from the primal input value.
    pub(super) forward: Arc<dyn Fn(Input) -> Result<(Output, Residual), ProgramError> + Send + Sync>,

    /// Closure computing the input cotangent value from `(residuals, output_cotangent_seeds)`.
    pub(super) backward: Arc<
        dyn Fn(Residual, Output::To<MaybeZero<Tracer>>) -> Result<Input::To<MaybeZero<Tracer>>, ProgramError>
            + Send
            + Sync,
    >,

    /// Whether reverse mode differentiation defers `forward` to the backward pass. Refer to
    /// [`Composite::with_deferred_forward`] for more information.
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

impl<V, O, Input, Output, Residual>
    CompositeVjp<
        SymbolicZeroVjp<CustomRuleTracer<V, O>, Input, Output, Residual>,
        CustomRuleTracer<V, O>,
        Input,
        Output,
    > for CustomRuleDefinition<V, O>
where
    V: 'static + Value<Type: Eq + Hash + Send + DifferentiableType + ReferenceMemberType>,
    O: 'static + CustomVjpOperation<V>,
    Input: 'static + Parameterized<CustomRuleTracer<V, O>>,
    Input::ParameterStructure: Debug + PartialEq + Send + Sync,
    Input::Family:
        ParameterizedFamily<V::Type> + ParameterizedFamily<V> + ParameterizedFamily<MaybeZero<CustomRuleTracer<V, O>>>,
    Input::To<V::Type>: Clone + Parameterized<V::Type, Family = Input::Family, To<CustomRuleTracer<V, O>> = Input>,
    Output: 'static + Parameterized<CustomRuleTracer<V, O>>,
    Output::ParameterStructure: Debug + PartialEq + Send + Sync,
    Output::Family:
        ParameterizedFamily<V::Type> + ParameterizedFamily<V> + ParameterizedFamily<MaybeZero<CustomRuleTracer<V, O>>>,
    Output::To<V::Type>: Parameterized<V::Type, Family = Output::Family, To<CustomRuleTracer<V, O>> = Output>,
    Residual: 'static + Parameterized<CustomRuleTracer<V, O>>,
    Residual::ParameterStructure: Debug + PartialEq + Send,
    Residual::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Residual::To<V::Type>: Parameterized<V::Type, Family = Residual::Family, To<CustomRuleTracer<V, O>> = Residual>,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    fn configure_vjp(
        self,
        configuration: &SymbolicZeroVjp<CustomRuleTracer<V, O>, Input, Output, Residual>,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
        non_differentiated_count: usize,
    ) -> Self {
        let residual_structures = CompositeResidualStructures::<V::Type, Residual::ParameterStructure>::default();
        let forward = retained_custom_vjp_forward_rule::<V, O, Input, Output, Residual, _>(
            configuration.forward.clone(),
            self.name.clone(),
            input_structure,
            output_structure,
            &residual_structures,
            self.named_axes(),
        );
        let backward = {
            let (backward, name, named_axes) =
                (configuration.backward.clone(), self.name.clone(), self.named_axes().to_vec());
            let (input_structure, output_structure) = (input_structure.clone(), output_structure.clone());
            move |leading_inputs: &[CustomRuleTracer<V, O>], seeds: &[MaybeZero<CustomRuleTracer<V, O>>]| {
                let residual_types = leading_inputs[non_differentiated_count..]
                    .iter()
                    .map(|value| value.r#type().into_owned())
                    .collect::<Vec<_>>();
                let structure = residual_structures.get(&name, &residual_types)?;
                let live_seeds = seeds.iter().filter_map(MaybeZero::as_value).cloned().collect::<Vec<_>>();
                let Some(context) =
                    leading_inputs.iter().chain(&live_seeds).next().map(|value| value.context().clone())
                else {
                    return Err(ProgramError::MalformedProgram(format!(
                        "`{name}` backward rule has neither leading inputs nor non-zero seeds",
                    )));
                };
                let (program, cotangent_zeros) = {
                    let name: &str = &name;
                    let backward = backward.as_ref();
                    let input_structure = &input_structure;
                    let non_differentiated_types = leading_inputs[..non_differentiated_count]
                        .iter()
                        .map(|value| value.r#type().into_owned())
                        .collect::<Vec<_>>();
                    let residual_structure = structure;
                    let output_structure = output_structure.clone();
                    let named_axes = named_axes.clone();
                    let non_differentiated_count = non_differentiated_types.len();
                    let residual_types = Residual::To::<V::Type>::from_parameters(residual_structure, residual_types)?;
                    let live_seed_types = seeds
                        .iter()
                        .filter_map(MaybeZero::as_value)
                        .map(|seed| seed.r#type().into_owned())
                        .collect::<Vec<_>>();
                    let mut cotangent_zeros = None;

                    // Keep non-differentiated inputs in the boundary even though the closure receives only
                    // residuals and cotangent seeds.
                    let (_, program) = DomainTracingContext::<EagerContext<V, O>>::trace_with_named_axes(
                        |(_, residuals, live_seeds): (
                            Vec<CustomRuleTracer<V, O>>,
                            Residual,
                            Vec<CustomRuleTracer<V, O>>,
                        )| {
                            let mut live_seeds = live_seeds.into_iter();
                            let leaves = seeds.iter().map(|seed| match seed {
                                MaybeZero::Value(_) => MaybeZero::Value(live_seeds.next().unwrap()),
                                MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type.clone()),
                            });
                            let seeds = Output::To::<MaybeZero<CustomRuleTracer<V, O>>>::from_parameters(
                                output_structure.clone(),
                                leaves,
                            )?;
                            let cotangents = backward(residuals, seeds)?;

                            // Check the full input structure before dropping the non-differentiated prefix, as for
                            // materialized seeds.
                            let cotangent_structure = cotangents.parameter_structure();
                            if &cotangent_structure != input_structure {
                                return Err(TypeError::invalid(format!(
                                    "{} backward rule cotangent structure mismatch: expected {:?} but got {:?}",
                                    if name == COMPOSITE_OPERATION_NAME {
                                        format!("`{COMPOSITE_OPERATION_NAME}`")
                                    } else {
                                        format!("`{COMPOSITE_OPERATION_NAME}` `{name}`")
                                    },
                                    input_structure,
                                    cotangent_structure,
                                ))
                                .into());
                            }

                            // Only the live cotangents become program outputs, while the structural zeros are recorded
                            // by type. This preserves symbolic zeros as in a JAX backward rule returning `None` or
                            // `SymbolicZero`.
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
                    (program.into_flat_program(), cotangent_zeros.unwrap())
                };

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
        let definition = self.with_symbolic_zero_vjp(forward, backward);
        if configuration.deferred_forward { definition.with_deferred_forward_rule() } else { definition }
    }
}

/// Reverse mode differentiation configuration of a [`Composite`] with a user-supplied forward rule closure implementing
/// `x ↦ (y, r)` and an accumulating backward rule closure, which submits the input cotangents to their accumulators.
/// `Tracer` is the tracer type of the closures' leaves, which the leaves of the seeds wrap, and `Transposition` is the
/// type of the transposition context that the backward rule closure receives. Refer to
/// [`Composite::with_accumulating_vjp`] for more information.
pub struct AccumulatingVjp<Tracer: Typed + Parameter, Transposition, Input, Output, Residual>
where
    Input: Parameterized<Tracer, Family: ParameterizedFamily<CotangentAccumulator>>,
    Output: Parameterized<Tracer, Family: ParameterizedFamily<MaybeZero<Tracer>>>,
{
    /// Closure computing `(outputs, residuals)` from the primal input value.
    pub(super) forward: Arc<dyn Fn(Input) -> Result<(Output, Residual), ProgramError> + Send + Sync>,

    /// Closure submitting the input cotangents, computed from `(residuals, output_cotangent_seeds)`, to their
    /// accumulators.
    #[allow(clippy::type_complexity)]
    pub(super) backward: Arc<
        dyn Fn(
                &mut Transposition,
                Residual,
                Output::To<MaybeZero<Tracer>>,
                Input::To<CotangentAccumulator>,
            ) -> Result<(), ProgramError>
            + Send
            + Sync,
    >,

    /// Whether reverse mode differentiation defers `forward` to the backward pass. Refer to
    /// [`Composite::with_deferred_forward`] for more information.
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

impl<V, O, Input, Output, Residual>
    CompositeVjp<
        AccumulatingVjp<CustomRuleTracer<V, O>, TranspositionContext<V, O>, Input, Output, Residual>,
        CustomRuleTracer<V, O>,
        Input,
        Output,
    > for CustomRuleDefinition<V, O>
where
    V: 'static + Value<Type: Eq + Hash + Send + DifferentiableType + ReferenceMemberType>,
    O: 'static + CustomVjpOperation<V>,
    Input: 'static + Parameterized<CustomRuleTracer<V, O>>,
    Input::ParameterStructure: Debug + PartialEq + Send + Sync,
    Input::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V> + ParameterizedFamily<CotangentAccumulator>,
    Input::To<V::Type>: Clone + Parameterized<V::Type, Family = Input::Family, To<CustomRuleTracer<V, O>> = Input>,
    Output: 'static + Parameterized<CustomRuleTracer<V, O>>,
    Output::ParameterStructure: Debug + PartialEq + Send + Sync,
    Output::Family:
        ParameterizedFamily<V::Type> + ParameterizedFamily<V> + ParameterizedFamily<MaybeZero<CustomRuleTracer<V, O>>>,
    Output::To<V::Type>: Parameterized<V::Type, Family = Output::Family, To<CustomRuleTracer<V, O>> = Output>,
    Residual: 'static + Parameterized<CustomRuleTracer<V, O>>,
    Residual::ParameterStructure: Debug + PartialEq + Send,
    Residual::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Residual::To<V::Type>: Parameterized<V::Type, Family = Residual::Family, To<CustomRuleTracer<V, O>> = Residual>,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    fn configure_vjp(
        self,
        configuration: &AccumulatingVjp<CustomRuleTracer<V, O>, TranspositionContext<V, O>, Input, Output, Residual>,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
        non_differentiated_count: usize,
    ) -> Self {
        let residual_structures = CompositeResidualStructures::<V::Type, Residual::ParameterStructure>::default();
        let forward = retained_custom_vjp_forward_rule::<V, O, Input, Output, Residual, _>(
            configuration.forward.clone(),
            self.name.clone(),
            input_structure,
            output_structure,
            &residual_structures,
            self.named_axes(),
        );

        // The rule is invoked directly while its carrier is transposed, with the carrier's inputs (i.e., the
        // non-differentiated inputs, the residuals, and the differentiated inputs' tangents) and one accumulator
        // per carrier input, so it receives the known residuals and the accumulators of the call's inputs.
        let backward = {
            let (backward, name) = (configuration.backward.clone(), self.name.clone());
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
        let definition = self.with_accumulating_vjp(forward, backward);
        if configuration.deferred_forward { definition.with_deferred_forward_rule() } else { definition }
    }
}

/// Installs a batching configuration in a [`CustomRuleDefinition`], selecting a custom batching rule or sequential
/// batching when configured.
pub trait CompositeBatching<Batching, Tracer: Parameter, Input: Parameterized<Tracer>, Output: Parameterized<Tracer>>:
    Sized
{
    /// Returns this definition with the provided batching configuration applied. [`DefaultBatching`] leaves the
    /// definition unchanged. Retained rules use the definition's name.
    ///
    /// # Parameters
    ///
    ///   - `configuration`: Batching configuration to apply.
    ///   - `input_structure`: Structure of the call's inputs.
    ///   - `output_structure`: Structure of the call's outputs.
    fn configure_batching(
        self,
        configuration: &Batching,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
    ) -> Self;
}

/// Batching configuration of a [`Composite`] without a custom batching rule. Batching a call structurally batches its
/// primal region.
#[derive(Copy, Clone, Debug, Default)]
pub struct DefaultBatching;

impl<V: Value, O: Operation<Type = V::Type>, Input, Output>
    CompositeBatching<DefaultBatching, CustomRuleTracer<V, O>, Input, Output> for CustomRuleDefinition<V, O>
where
    Input: Parameterized<CustomRuleTracer<V, O>>,
    Output: Parameterized<CustomRuleTracer<V, O>>,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    #[inline]
    fn configure_batching(
        self,
        _configuration: &DefaultBatching,
        _input_structure: &Input::ParameterStructure,
        _output_structure: &Output::ParameterStructure,
    ) -> Self {
        self
    }
}

/// Batching configuration of a [`Composite`] with a user-supplied custom batching rule closure implementing
/// `(x, x_axes, extent) ↦ (y, y_axes)` whose per-item numerical behavior is preserved when a replicated input is
/// broadcast. `Tracer` is the tracer type of the closure's leaves, which is also the type of dynamic extents. Refer
/// to [`Composite::with_batching`] for more information. Rules whose numerical behavior may change when a replicated
/// input is broadcast use [`BroadcastSensitiveBatching`] instead.
pub struct BroadcastInvariantBatching<Tracer: Parameter, Input, Output>
where
    Input: Parameterized<Tracer, Family: ParameterizedFamily<BatchAxis>>,
    Output: Parameterized<Tracer, Family: ParameterizedFamily<BatchAxis>>,
{
    /// Closure computing the batched outputs and their batch axes from the batched inputs and their batch axes.
    #[allow(clippy::type_complexity)]
    pub(super) rule: Arc<
        dyn Fn(
                Input,
                Input::To<BatchAxis>,
                BatchingLevelExtent<Tracer>,
            ) -> Result<(Output, Output::To<BatchAxis>), ProgramError>
            + Send
            + Sync,
    >,
}

impl<Tracer: Parameter, Input, Output> Debug for BroadcastInvariantBatching<Tracer, Input, Output>
where
    Input: Parameterized<Tracer, Family: ParameterizedFamily<BatchAxis>>,
    Output: Parameterized<Tracer, Family: ParameterizedFamily<BatchAxis>>,
{
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("BroadcastInvariantBatching").finish_non_exhaustive()
    }
}

impl<V, O, Input, Output>
    CompositeBatching<
        BroadcastInvariantBatching<CustomRuleTracer<V, O>, Input, Output>,
        CustomRuleTracer<V, O>,
        Input,
        Output,
    > for CustomRuleDefinition<V, O>
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
    Output: 'static + Parameterized<CustomRuleTracer<V, O>, Family: ParameterizedFamily<BatchAxis>>,
    Output::ParameterStructure: Debug + PartialEq + Send + Sync,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    #[inline]
    fn configure_batching(
        self,
        configuration: &BroadcastInvariantBatching<CustomRuleTracer<V, O>, Input, Output>,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
    ) -> Self {
        let rule = retained_custom_batching_rule::<V, O, Input, Output, _>(
            configuration.rule.clone(),
            self.name.clone(),
            input_structure,
            output_structure,
        );
        self.with_batching_rule(rule)
    }
}

/// Batching configuration of a [`Composite`] with a user-supplied custom batching rule closure implementing
/// `(x, x_axes, extent) ↦ (y, y_axes)` whose per-item numerical behavior may change when a replicated input is
/// broadcast. `Tracer` is the tracer type of the closure's leaves, which is also the type of dynamic extents.
/// Refer to [`Composite::with_broadcast_sensitive_batching`] for more information.
pub struct BroadcastSensitiveBatching<Tracer: Parameter, Input, Output>
where
    Input: Parameterized<Tracer, Family: ParameterizedFamily<BatchAxis>>,
    Output: Parameterized<Tracer, Family: ParameterizedFamily<BatchAxis>>,
{
    /// Closure computing the batched outputs and their batch axes from the batched inputs and their batch axes.
    #[allow(clippy::type_complexity)]
    pub(super) rule: Arc<
        dyn Fn(
                Input,
                Input::To<BatchAxis>,
                BatchingLevelExtent<Tracer>,
            ) -> Result<(Output, Output::To<BatchAxis>), ProgramError>
            + Send
            + Sync,
    >,
}

impl<Tracer: Parameter, Input, Output> Debug for BroadcastSensitiveBatching<Tracer, Input, Output>
where
    Input: Parameterized<Tracer, Family: ParameterizedFamily<BatchAxis>>,
    Output: Parameterized<Tracer, Family: ParameterizedFamily<BatchAxis>>,
{
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("BroadcastSensitiveBatching").finish_non_exhaustive()
    }
}

impl<V, O, Input, Output>
    CompositeBatching<
        BroadcastSensitiveBatching<CustomRuleTracer<V, O>, Input, Output>,
        CustomRuleTracer<V, O>,
        Input,
        Output,
    > for CustomRuleDefinition<V, O>
where
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
    Output: 'static + Parameterized<CustomRuleTracer<V, O>, Family: ParameterizedFamily<BatchAxis>>,
    Output::ParameterStructure: Debug + PartialEq + Send + Sync,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    #[inline]
    fn configure_batching(
        self,
        configuration: &BroadcastSensitiveBatching<CustomRuleTracer<V, O>, Input, Output>,
        input_structure: &Input::ParameterStructure,
        output_structure: &Output::ParameterStructure,
    ) -> Self {
        let rule = retained_custom_batching_rule::<V, O, Input, Output, _>(
            configuration.rule.clone(),
            self.name.clone(),
            input_structure,
            output_structure,
        );
        self.with_broadcast_sensitive_batching_rule(rule)
    }
}

/// Batching configuration of a [`Composite`] that batches its calls sequentially, by looping over the batch items with
/// a scan. Refer to [`Composite::with_sequential_batching`] for more information.
#[derive(Copy, Clone, Debug, Default)]
pub struct SequentialBatching;

impl<V, O, Input, Output> CompositeBatching<SequentialBatching, CustomRuleTracer<V, O>, Input, Output>
    for CustomRuleDefinition<V, O>
where
    V: 'static + Value<Type: Eq + Hash + ScanType>,
    O: 'static + Operation<Type = V::Type> + From<ScanOperation<V::Type>>,
    Input: Parameterized<CustomRuleTracer<V, O>>,
    Output: Parameterized<CustomRuleTracer<V, O>>,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    #[inline]
    fn configure_batching(
        self,
        _configuration: &SequentialBatching,
        _input_structure: &Input::ParameterStructure,
        _output_structure: &Output::ParameterStructure,
    ) -> Self {
        self.with_sequential_batching()
    }
}

// TODO(eaplatanios): Review from here onwards.

/// Residual structures that the forward rule of one [`Composite`] registration returned, from which its backward rule
/// learns the structure of the residuals that it receives as flat leading inputs. The backward rule of each
/// specialization is traced after the forward rule of that specialization, but specializations of one call structure
/// may return differently structured residuals (e.g., collections whose lengths depend on the input shapes), so the
/// structures are recorded by the flat residual types that the forward rule returned. Different structures are
/// ambiguous only when their flat residual types are equivalent.
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
            Some((_, recorded)) if recorded != &structure => Err(TypeError::invalid(format!(
                "{} forward rule residual structure mismatch: expected {:?} but got {:?}",
                if name == COMPOSITE_OPERATION_NAME {
                    format!("`{COMPOSITE_OPERATION_NAME}`")
                } else {
                    format!("`{COMPOSITE_OPERATION_NAME}` `{name}`")
                },
                recorded,
                structure,
            ))),
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

/// Adapts the structured forward rule closure `forward` for reverse mode differentiation of a [`Composite`] to the
/// flat interface of retained forward rules. Each invocation traces the closure at the types of its primal inputs,
/// records the residual structure in `residual_structures`, and replays the traced program on the primal inputs.
///
/// # Parameters
///
///   - `forward`: Closure implementing `x ↦ (y, r)`.
///   - `name`: Name of the function, used in diagnostics.
///   - `input_structure`: Structure of the call's inputs.
///   - `output_structure`: Structure of the call's primal outputs.
///   - `residual_structures`: Residual structures shared with the backward rule of the same registration.
///   - `named_axes`: Named axes with which the closure is traced. Refer to [`CustomRuleDefinition::with_named_axes`]
///     for more information.
#[allow(clippy::type_complexity)]
fn retained_custom_vjp_forward_rule<V, O, Input, Output, Residual, Forward>(
    forward: Arc<Forward>,
    name: Cow<'static, str>,
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
    Input: 'static + Parameterized<CustomRuleTracer<V, O>>,
    Input::ParameterStructure: Debug + PartialEq + Send + Sync,
    Input::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Input::To<V::Type>: Clone + Parameterized<V::Type, Family = Input::Family, To<CustomRuleTracer<V, O>> = Input>,
    Output: 'static + Parameterized<CustomRuleTracer<V, O>>,
    Output::ParameterStructure: Debug + PartialEq + Send + Sync,
    Output::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Residual: 'static + Parameterized<CustomRuleTracer<V, O>>,
    Residual::ParameterStructure: Debug + PartialEq + Send,
    Residual::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>,
    Residual::To<V::Type>: Parameterized<V::Type, Family = Residual::Family, To<CustomRuleTracer<V, O>> = Residual>,
    Forward: ?Sized + 'static + Fn(Input) -> Result<(Output, Residual), ProgramError> + Send + Sync,
{
    let (name, residual_structures, named_axes) = (name, residual_structures.clone(), named_axes.to_vec());
    let (input_structure, output_structure) = (input_structure.clone(), output_structure.clone());
    move |primals: &[CustomRuleTracer<V, O>]| {
        let input_types = Input::To::<V::Type>::from_parameters(
            input_structure.clone(),
            primals.iter().map(|primal| primal.r#type().into_owned()),
        )?;
        let (structure, residual_types, program) = {
            let name: &str = &name;
            let forward = forward.as_ref();
            let output_structure = &output_structure;
            let named_axes = named_axes.clone();
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
            // Matching flat types cannot establish the nested output structure.
            let forward_output_structure = forward_output_types.parameter_structure();
            if &forward_output_structure != output_structure {
                return Err(TypeError::invalid(format!(
                    "{} forward rule output structure mismatch: expected {:?} but got {:?}",
                    if name == COMPOSITE_OPERATION_NAME {
                        format!("`{COMPOSITE_OPERATION_NAME}`")
                    } else {
                        format!("`{COMPOSITE_OPERATION_NAME}` `{name}`")
                    },
                    output_structure,
                    forward_output_structure,
                ))
                .into());
            }

            // Reference residuals preserve handles, not snapshots. Validate their input identities while the traced
            // atoms are available; matching types alone cannot distinguish two inputs of the same reference type. The
            // operation subsequently checks that reference inputs belong to the non-differentiated prefix.
            let output_count = forward_output_types.parameter_count();
            let mut forwarded_inputs = HashSet::new();
            for (index, residual) in program.output_ids().iter().skip(output_count).enumerate() {
                let r#type = program.atoms()[residual.index()].r#type();
                if !r#type.is_reference() {
                    continue;
                }
                if !program.input_ids().contains(residual) {
                    return Err(TypeError::invalid(format!(
                        "`{name}` forward rule returns residual {index} of reference type `{type}` \
                         that is not a leading non-differentiated input forwarded by identity",
                    ))
                    .into());
                }
                if !forwarded_inputs.insert(*residual) {
                    return Err(TypeError::invalid(format!(
                        "`{name}` forward rule returns residual {index} of reference type `{type}` \
                         from an input already forwarded by an earlier residual",
                    ))
                    .into());
                }
            }

            // The closure ran exactly once, so the structure of its residuals is always recorded.
            (residual_structure.unwrap(), residual_types.parameters().cloned().collect(), program.into_flat_program())
        };
        residual_structures.record(&name, residual_types, structure)?;
        let mut outputs = program.interpret_in_context(primals[0].context(), primals.to_vec())?;
        let residuals = outputs.split_off(output_structure.parameter_count());
        Ok((outputs, residuals))
    }
}

/// Adapts the structured custom batching rule closure `rule` of a [`Composite`] to the flat interface of retained
/// batching rules. Each invocation passes the closure the call's own batched inputs with the call's input structure,
/// their batch axes, and the level's extent. It validates that the outputs and their batch axes have the call's output
/// structure.
///
/// # Parameters
///
///   - `rule`: Closure implementing `(x, x_axes, extent) ↦ (y, y_axes)`.
///   - `name`: Name of the function, used in diagnostics.
///   - `input_structure`: Structure of the call's inputs.
///   - `output_structure`: Structure of the call's outputs.
#[allow(clippy::type_complexity)]
fn retained_custom_batching_rule<V, O, Input, Output, Rule>(
    rule: Arc<Rule>,
    name: Cow<'static, str>,
    input_structure: &Input::ParameterStructure,
    output_structure: &Output::ParameterStructure,
) -> impl 'static
+ Fn(
    &[CustomRuleTracer<V, O>],
    &[BatchAxis],
    &BatchingLevel<V::Type>,
    &[CustomRuleTracer<V, O>],
) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<BatchAxis>), ProgramError>
+ Send
+ Sync
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: 'static + Value,
    O: 'static + Operation<Type = V::Type>,
    Input: 'static
        + Parameterized<CustomRuleTracer<V, O>, Family: ParameterizedFamily<BatchAxis>, ParameterStructure: Send + Sync>,
    Output: 'static + Parameterized<CustomRuleTracer<V, O>, Family: ParameterizedFamily<BatchAxis>>,
    Output::ParameterStructure: Debug + PartialEq + Send + Sync,
    Rule: ?Sized
        + 'static
        + Fn(
            Input,
            Input::To<BatchAxis>,
            BatchingLevelExtent<CustomRuleTracer<V, O>>,
        ) -> Result<(Output, Output::To<BatchAxis>), ProgramError>
        + Send
        + Sync,
{
    let (input_structure, output_structure) = (input_structure.clone(), output_structure.clone());
    move |inputs: &[CustomRuleTracer<V, O>],
          input_axes: &[BatchAxis],
          level: &BatchingLevel<V::Type>,
          boundary_inputs: &[CustomRuleTracer<V, O>]| {
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
                        {
                            let name: &str = &name;
                            if name == COMPOSITE_OPERATION_NAME {
                                format!("`{COMPOSITE_OPERATION_NAME}`")
                            } else {
                                format!("`{COMPOSITE_OPERATION_NAME}` `{name}`")
                            }
                        },
                    ),
                });
            }
        };
        let leading_input_count = inputs
            .len()
            .checked_sub(input_structure.parameter_count())
            .ok_or_else(|| ProgramError::MalformedProgram(format!("`{name}` batching rule received too few inputs")))?;
        let (outputs, output_axes) = rule(
            Input::from_parameters(input_structure.clone(), inputs[leading_input_count..].iter().cloned())?,
            Input::To::<BatchAxis>::from_parameters(
                input_structure.clone(),
                input_axes[leading_input_count..].iter().copied(),
            )?,
            extent,
        )?;
        for (value, structure) in
            [("output", outputs.parameter_structure()), ("output axis", output_axes.parameter_structure())]
        {
            if structure != output_structure {
                return Err(TypeError::invalid(format!(
                    "{} batching rule {} structure mismatch: expected {:?} but got {:?}",
                    {
                        let name: &str = &name;
                        if name == COMPOSITE_OPERATION_NAME {
                            format!("`{COMPOSITE_OPERATION_NAME}`")
                        } else {
                            format!("`{COMPOSITE_OPERATION_NAME}` `{name}`")
                        }
                    },
                    value,
                    output_structure,
                    structure,
                ))
                .into());
            }
        }
        Ok((outputs.into_parameters().collect(), output_axes.into_parameters().collect()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType, DimensionBounds,
        DimensionType, ShardingDimension,
    };
    use crate::batching::BatchAxis;
    use crate::differentiation::DifferentiationError;
    use crate::operations::composites::rules::{
        CustomRuleBatchingRuleKey, CustomRuleRegistration, CustomRuleSpecializationKey, CustomRuleSpecializer,
    };
    use crate::parameters::Placeholder;

    use super::*;

    type ArrayTracer = CustomRuleTracer<Array, ArrayOperation<Array>>;
    type ArrayDefinition = CustomRuleDefinition<Array, ArrayOperation<Array>>;

    #[test]
    fn test_custom_rule_definition_configure_jvp() {
        // The configured callback outlives its configuration and reports the definition's owned name.
        let registration = {
            let configuration = ExplicitJvp::<ArrayTracer, Vec<ArrayTracer>> {
                jvp: Arc::new(|input, tangent| Ok((vec![input.clone(), input], vec![tangent.clone(), tangent]))),
            };
            CustomRuleRegistration::new(ArrayDefinition::new(String::from("jvp_definition")).configure_jvp(
                &configuration,
                &Placeholder,
                &vec![Placeholder],
                0,
            ))
        };
        let key = CustomRuleSpecializationKey {
            input_types: vec![ArrayType::scalar(DataType::F64)],
            output_types: vec![ArrayType::scalar(DataType::F64)],
            kept_outputs: None,
            non_differentiated_count: 0,
            tangent_activity: vec![true],
            levels: Vec::new(),
            discharged: false,
        };
        assert!(matches!(
            registration.reference().jvp_specialization(key),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == format!(
                    "`{COMPOSITE_OPERATION_NAME}` `jvp_definition` JVP rule output structure mismatch: expected \
                     {:?} but got {:?}",
                    [Placeholder],
                    [Placeholder, Placeholder],
                ),
        ));
    }

    #[test]
    fn test_custom_rule_definition_configure_vjp() {
        let registration = {
            let configuration = ExplicitVjp::<ArrayTracer, Vec<ArrayTracer>, ArrayTracer> {
                forward: Arc::new(|input| Ok((vec![input.clone(), input.clone()], input))),
                backward: Arc::new(|residual, _| Ok(residual)),
                deferred_forward: false,
            };
            CustomRuleRegistration::new(ArrayDefinition::new(String::from("vjp_definition")).configure_vjp(
                &configuration,
                &Placeholder,
                &vec![Placeholder],
                0,
            ))
        };
        let key = CustomRuleSpecializationKey {
            input_types: vec![ArrayType::scalar(DataType::F64)],
            output_types: vec![ArrayType::scalar(DataType::F64)],
            kept_outputs: None,
            non_differentiated_count: 0,
            tangent_activity: Vec::new(),
            levels: Vec::new(),
            discharged: false,
        };
        assert!(matches!(
            registration.reference().forward_specialization(key),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == format!(
                    "`{COMPOSITE_OPERATION_NAME}` `vjp_definition` forward rule output structure mismatch: expected \
                     {:?} but got {:?}",
                    [Placeholder],
                    [Placeholder, Placeholder],
                ),
        ));
    }

    #[test]
    fn test_custom_rule_definition_configure_batching() {
        let registration = {
            let configuration = BroadcastInvariantBatching::<ArrayTracer, ArrayTracer, Vec<ArrayTracer>> {
                rule: Arc::new(|input, axis, _| Ok((vec![input.clone(), input], vec![axis, axis]))),
            };
            CustomRuleRegistration::new(ArrayDefinition::new(String::from("batching_definition")).configure_batching(
                &configuration,
                &Placeholder,
                &vec![Placeholder],
            ))
        };
        let key = CustomRuleBatchingRuleKey {
            level: BatchingLevel::new(BatchingLevelExtent::Static(2), None, ShardingDimension::Replicated),
            boundary_input_types: Vec::new(),
            input_types: vec![ArrayType::scalar(DataType::F64)],
            unbatched_input_types: vec![ArrayType::scalar(DataType::F64)],
            input_axes: vec![BatchAxis::replicated()],
            discharged: false,
        };
        assert!(matches!(
            registration.reference().batching_rule_specialization(key),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == format!(
                    "`{COMPOSITE_OPERATION_NAME}` `batching_definition` batching rule output structure mismatch: \
                     expected {:?} but got {:?}",
                    [Placeholder],
                    [Placeholder, Placeholder],
                ),
        ));
    }

    #[test]
    fn test_retained_custom_batching_rule_leading_boundary_extent() {
        // A dynamic batch extent is the leading boundary input of its batching level, which may have further boundary
        // inputs that the rule does not receive.
        type IrTracer = CustomRuleTracer<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let rule = |_input: IrTracer, _axis: BatchAxis, extent: BatchingLevelExtent<IrTracer>| match extent {
            BatchingLevelExtent::Dynamic(extent) => Ok((extent, BatchAxis::replicated())),
            BatchingLevelExtent::Static(_) => {
                Err(ProgramError::InvalidArgument { message: "the test rule expects a dynamic extent".to_string() })
            }
        };
        let adapter = retained_custom_batching_rule::<_, _, IrTracer, IrTracer, _>(
            Arc::new(rule),
            Cow::Borrowed("rule"),
            &Placeholder,
            &Placeholder,
        );
        let extent_type = ArrayIrType::from(DimensionType::new("n", DimensionBounds::new(1, Some(8)).unwrap()));
        let other_type = ArrayIrType::from(DimensionType::new("m", DimensionBounds::new(1, Some(8)).unwrap()));
        let level =
            BatchingLevel::new(BatchingLevelExtent::Dynamic(extent_type.clone()), None, ShardingDimension::Replicated);
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |values: Vec<IrTracer>| {
                let (outputs, output_axes) = adapter(&values[2..], &[BatchAxis::replicated()], &level, &values[..2])?;
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
