use std::fmt::{Debug, Display};

use crate::contexts::StagingContext;
use crate::differentiation::{Cotangent, Tangent, TransposableOperation};
use crate::domains::Domain;
use crate::macros::check_count;
use crate::operations::constants::{SupportsZero, ZeroLike};
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::programs::{ProgramError, Value};
use crate::tracing::{AbstractTracingContext, DomainTracer, Tracer, TracingContext};
use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation, BatchingContext};
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, ResidualizedOperation, TangentContext};
use crate::tracing_v2::operations::control_flow::{
    FlatProgram, ensure_types_match, flat_program_input_types, flat_program_output_types, stage_cotangent,
};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext, ResidualFactor};
use crate::types::{ArrayType, Type, TypeError, Typed};

/// Higher-order operation pairing a primal program with a user-supplied JVP program — the direct analogue of JAX's
/// [`custom_jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html).
///
/// The JVP program follows JAX's calling convention: it receives the primal inputs followed by one tangent per
/// primal input, and returns the primal outputs followed by one tangent per primal output. The primal program is
/// kept separate from the JVP program so that un-differentiated calls do not pay for tangent computation:
/// interpretation, batching, and backend lowering replay the lean primal program; linearization replays the JVP
/// program instead of differentiating the primal body, so the user-supplied derivative governs both forward and
/// reverse mode (reverse mode transposes the linearization of the JVP program, which therefore must be linear in
/// its tangent arguments).
///
/// Note that batching inlines the primal program through the standard per-operation batching rules, so the custom
/// derivative does not survive a `batch` applied *before* differentiation; differentiate first or avoid batching
/// custom-derivative calls when the custom rule must be preserved.
#[derive(Clone, Debug)]
pub struct CustomJvpOperation<V, O, T = ArrayType>
where
    T: PartialEq + Type,
    V: Value<T>,
{
    /// Program computing the primal outputs from the primal inputs.
    primal: FlatProgram<V, O, T>,

    /// Program computing `(outputs..., output_tangents...)` from `(inputs..., input_tangents...)`.
    jvp: FlatProgram<V, O, T>,
}

impl<T: PartialEq + Type, V: Value<T>, O: Operation<T>> CustomJvpOperation<V, O, T> {
    /// Creates a custom-JVP operation after validating that the JVP program's signature matches the primal
    /// program's: its inputs must be the primal inputs followed by their tangents (same types), and its outputs the
    /// primal outputs followed by their tangents.
    pub fn new(primal: FlatProgram<V, O, T>, jvp: FlatProgram<V, O, T>) -> Result<Self, TypeError> {
        let input_types = flat_program_input_types(&primal);
        let output_types = flat_program_output_types(&primal);
        let expected_jvp_input_types: Vec<T> = input_types.iter().chain(input_types.iter()).cloned().collect();
        ensure_types_match("custom_jvp rule input", &expected_jvp_input_types, &flat_program_input_types(&jvp))?;
        let expected_jvp_output_types: Vec<T> = output_types.iter().chain(output_types.iter()).cloned().collect();
        ensure_types_match("custom_jvp rule output", &expected_jvp_output_types, &flat_program_output_types(&jvp))?;
        Ok(Self { primal, jvp })
    }

    /// Returns the primal program.
    #[inline]
    pub fn primal(&self) -> &FlatProgram<V, O, T> {
        &self.primal
    }

    /// Returns the user-supplied JVP program.
    #[inline]
    pub fn jvp_program(&self) -> &FlatProgram<V, O, T> {
        &self.jvp
    }

    /// Returns the primal input types.
    #[inline]
    pub fn input_types(&self) -> Vec<T> {
        flat_program_input_types(&self.primal)
    }

    /// Returns the primal output types.
    #[inline]
    pub fn output_types(&self) -> Vec<T> {
        flat_program_output_types(&self.primal)
    }
}

impl<T: PartialEq + Type, V: Value<T>, O> Display for CustomJvpOperation<V, O, T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("custom_jvp")
    }
}

impl<T: PartialEq + Type, V: Value<T>, O: Operation<T>> Operation<T> for CustomJvpOperation<V, O, T> {
    #[inline]
    fn name(&self) -> &'static str {
        "custom_jvp"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        ensure_types_match("custom_jvp input", &self.input_types(), input_types)?;
        Ok(self.output_types())
    }
}

impl<T, V, O> InterpretableOperation<T, V> for CustomJvpOperation<V, O, T>
where
    T: PartialEq + Type,
    V: Value<T>,
    O: InterpretableOperation<T, V> + Operation<T>,
    Vec<V>: Parameterized<V, ParameterStructure: Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        self.primal.interpret(inputs.to_vec())
    }
}

/// Shared implementation of the [`CustomJvpOperation`] JVP rule, generic over the linearization context's value
/// type so the operation enum dispatchers ([`ArrayOperation`](crate::tracing_v2::ArrayOperation),
/// [`ScalarOperation`](crate::operations::scalars::ScalarOperation)) can invoke it for any
/// [`DifferentiationContext`] whose constants match the captured programs.
///
/// The rule evaluates the JVP program's primal at `(x̂, 0)` (its first half yields the rule's primal outputs) and
/// seeds its pushforward with `(0, t̂)` so that only the user-defined — and therefore necessarily linear — tangent
/// map survives in the staged linear program.
pub(crate) fn custom_jvp_rule<'jvp, D, O>(
    operation: &CustomJvpOperation<<D as Domain>::Constant, O, D::Type>,
    context: &mut TangentContext<'jvp, D>,
    inputs: &[JvpTracer<'jvp, D>],
) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
where
    D: DifferentiationContext<Type: PartialEq> + 'jvp,
    <D as Domain>::Value: ZeroLike,
    O: Operation<D::Type> + DifferentiableOperation<D>,
    LinearOperationOf<D>: ResidualizedOperation<D>,
    Vec<<D as Domain>::Constant>: Parameterized<
            <D as Domain>::Constant,
            Family: ParameterizedFamily<D::Tangent> + ParameterizedFamily<<D as Domain>::Value>,
            To<<D as Domain>::Value> = Vec<<D as Domain>::Value>,
            To<D::Tangent> = Vec<D::Tangent>,
            ParameterStructure: Debug + PartialEq,
        >,
{
    let input_count = operation.input_types().len();
    let output_count = operation.output_types().len();
    check_count!("input", inputs, input_count, ProgramError);
    // Evaluate the JVP program's primal at `(x̂, 0)`; its first half yields the rule's primal outputs.
    let mut jvp_primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
    for input in inputs.iter() {
        jvp_primal_inputs.push(input.primal().zero_like());
    }
    // Seed the pushforward with `(0, t̂)`: zero tangents for the primal slots and the incoming tangents for the
    // tangent slots, so only the user-defined (linear) tangent map survives.
    let mut tangent_seeds = Vec::with_capacity(2 * inputs.len());
    for input in inputs.iter() {
        tangent_seeds.push(context.materialize_tangent(Tangent::Zero(input.primal().r#type().into_owned()))?);
    }
    for input in inputs.iter() {
        tangent_seeds.push(context.materialize_tangent(input.tangent().clone())?);
    }
    let (jvp_primal_outputs, pushforward) =
        context.differentiable().linearize_program(operation.jvp_program(), jvp_primal_inputs)?;
    let pushforward_program = pushforward.program_with_residual_constants()?;
    let tangent_outputs = context.stage_program(&pushforward_program, tangent_seeds)?;
    Ok(jvp_primal_outputs
        .into_iter()
        .take(output_count)
        .zip(tangent_outputs.into_iter().skip(output_count))
        .map(|(primal, tangent)| JvpTracer::from_value(primal, tangent))
        .collect())
}

/// Value-level batching for [`CustomJvpOperation`]: inlines the primal program through the per-operation batching
/// rules. The custom derivative does not survive this inlining; see the type-level documentation.
impl<V, O> BatchableOperation<V, ()> for CustomJvpOperation<V, O, ArrayType>
where
    V: Value<ArrayType>,
    O: Operation<ArrayType> + BatchableOperation<V>,
{
    fn batch(&self, _context: &(), inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        batch_program_inline(&self.primal, inputs)
    }
}

/// Traced batching for [`CustomJvpOperation`]: inlines the primal program into the parent trace through
/// `BatchingContext::interpret_program`. The custom derivative does not survive this inlining; see the type-level
/// documentation.
impl<C, O> BatchableOperation<Tracer<C>, BatchingContext<C>> for CustomJvpOperation<C::Constant, O, ArrayType>
where
    C: StagingContext<Type = ArrayType>,
    C::Constant: Value<ArrayType>,
    O: Operation<ArrayType> + BatchableOperation<Tracer<C>, BatchingContext<C>>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<Tracer<C>>],
    ) -> Result<Vec<ArrayBatch<Tracer<C>>>, ProgramError> {
        context.interpret_program(&self.primal, inputs.to_vec())
    }
}

/// Replays `program` over packed batch values, dispatching every instruction through its value-level batching rule.
fn batch_program_inline<V, O>(
    program: &FlatProgram<V, O, ArrayType>,
    inputs: &[ArrayBatch<V>],
) -> Result<Vec<ArrayBatch<V>>, ProgramError>
where
    V: Value<ArrayType>,
    O: Operation<ArrayType> + BatchableOperation<V>,
{
    program.interpret_with(
        inputs.to_vec(),
        |_, constant: &V| Ok(ArrayBatch::unbatched(constant.clone())),
        |instruction, instruction_inputs| instruction.operation().batch(&(), instruction_inputs),
    )
}

/// Higher-order operation pairing a primal program with user-supplied forward/backward (VJP) programs — the direct
/// analogue of JAX's [`custom_vjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html).
///
/// The forward program maps the primal inputs to the primal outputs followed by arbitrarily many residual values;
/// the backward program maps those residuals followed by one cotangent per primal output to one cotangent per primal
/// input. The primal program is kept separate from the forward program so that un-differentiated calls do not pay
/// for residual computation: interpretation, batching, and backend lowering replay the lean primal program, and the
/// forward program runs only under reverse-mode differentiation. Linearization evaluates the
/// forward program, captures its residuals as factors, and stages one opaque linear call whose transpose replays the
/// backward program — so reverse mode uses exactly the user-supplied gradient. Forward-mode differentiation
/// (interpreting the staged linear call) is rejected, matching JAX's `custom_vjp` semantics.
///
/// Note that batching inlines the primal program through the standard per-operation batching rules, so the custom
/// derivative does not survive a `batch` applied *before* differentiation; differentiate first or avoid batching
/// custom-derivative calls when the custom rule must be preserved.
#[derive(Clone, Debug)]
pub struct CustomVjpOperation<V, O, T = ArrayType>
where
    T: PartialEq + Type,
    V: Value<T>,
{
    /// Program computing the primal outputs from the primal inputs.
    primal: FlatProgram<V, O, T>,

    /// Program computing `(outputs..., residuals...)` from the primal inputs.
    forward: FlatProgram<V, O, T>,

    /// Program computing one input cotangent per primal input from `(residuals..., output_cotangents...)`.
    backward: FlatProgram<V, O, T>,
}

impl<T: PartialEq + Type, V: Value<T>, O: Operation<T>> CustomVjpOperation<V, O, T> {
    /// Creates a custom-VJP operation after validating the forward/backward program signatures against the primal
    /// program's: `forward` must consume the primal inputs and produce the primal outputs followed by the residuals,
    /// and `backward` must consume those residuals followed by one cotangent per primal output and produce one
    /// cotangent per primal input.
    pub fn new(
        primal: FlatProgram<V, O, T>,
        forward: FlatProgram<V, O, T>,
        backward: FlatProgram<V, O, T>,
    ) -> Result<Self, TypeError> {
        let input_types = flat_program_input_types(&primal);
        let output_types = flat_program_output_types(&primal);
        ensure_types_match("custom_vjp forward input", &input_types, &flat_program_input_types(&forward))?;
        let forward_output_types = flat_program_output_types(&forward);
        if forward_output_types.len() < output_types.len() {
            return Err(TypeError {
                message: format!(
                    "custom_vjp forward must produce at least the {} primal output(s) but produced {} value(s)",
                    output_types.len(),
                    forward_output_types.len(),
                ),
            });
        }
        ensure_types_match("custom_vjp forward output", &output_types, &forward_output_types[..output_types.len()])?;
        let residual_types = &forward_output_types[output_types.len()..];
        let expected_backward_input_types: Vec<T> = residual_types.iter().chain(output_types.iter()).cloned().collect();
        ensure_types_match(
            "custom_vjp backward input",
            &expected_backward_input_types,
            &flat_program_input_types(&backward),
        )?;
        ensure_types_match("custom_vjp backward output", &input_types, &flat_program_output_types(&backward))?;
        Ok(Self { primal, forward, backward })
    }

    /// Returns the primal program.
    #[inline]
    pub fn primal(&self) -> &FlatProgram<V, O, T> {
        &self.primal
    }

    /// Returns the forward (residual-producing) program.
    #[inline]
    pub fn forward(&self) -> &FlatProgram<V, O, T> {
        &self.forward
    }

    /// Returns the backward (cotangent-producing) program.
    #[inline]
    pub fn backward(&self) -> &FlatProgram<V, O, T> {
        &self.backward
    }

    /// Returns the primal input types.
    #[inline]
    pub fn input_types(&self) -> Vec<T> {
        flat_program_input_types(&self.primal)
    }

    /// Returns the primal output types.
    #[inline]
    pub fn output_types(&self) -> Vec<T> {
        flat_program_output_types(&self.primal)
    }
}

impl<T: PartialEq + Type, V: Value<T>, O> Display for CustomVjpOperation<V, O, T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("custom_vjp")
    }
}

impl<T: PartialEq + Type, V: Value<T>, O: Operation<T>> Operation<T> for CustomVjpOperation<V, O, T> {
    #[inline]
    fn name(&self) -> &'static str {
        "custom_vjp"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        ensure_types_match("custom_vjp input", &self.input_types(), input_types)?;
        Ok(self.output_types())
    }
}

impl<T, V, O> InterpretableOperation<T, V> for CustomVjpOperation<V, O, T>
where
    T: PartialEq + Type,
    V: Value<T>,
    O: InterpretableOperation<T, V> + Operation<T>,
    Vec<V>: Parameterized<V, ParameterStructure: Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        self.primal.interpret(inputs.to_vec())
    }
}

/// Trait for linear operation types that include or can wrap [`CustomVjpCallOperation`]. Linear operation enums
/// implement this trait so that the [`CustomVjpOperation`] JVP rule and the [`CustomVjpCallOperation`] transpose rule
/// can stage calls without knowing the concrete operation enum.
#[doc(hidden)]
pub trait SupportsCustomVjpCall<T: PartialEq + Type, C: Value<T>, O, F: Value<T>> {
    /// Constructs the backend-specific representation of [`CustomVjpCallOperation`].
    fn custom_vjp_call_operation(backward: FlatProgram<C, O, T>, residuals: Vec<F>, transposed: bool) -> Self;
}

/// Shared implementation of the [`CustomVjpOperation`] JVP rule, generic over the linearization context's value type
/// so the operation enum dispatchers can invoke it for any [`DifferentiationContext`] whose constants match the
/// captured programs.
///
/// The rule linearizes the forward program at the primal inputs — discarding the resulting pushforward, so the
/// forward body is never differentiated beyond what its primal evaluation requires — captures the trailing residual
/// outputs as factors, and stages one opaque [`CustomVjpCallOperation`] mapping the input tangents to the output
/// tangents. The staged call rejects forward-mode interpretation; its transpose replays the user's backward program.
pub(crate) fn custom_vjp_rule<'jvp, D, O>(
    operation: &CustomVjpOperation<<D as Domain>::Constant, O, D::Type>,
    context: &mut TangentContext<'jvp, D>,
    inputs: &[JvpTracer<'jvp, D>],
) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
where
    D: DifferentiationContext<Type: PartialEq> + 'jvp,
    O: Clone + Operation<D::Type> + DifferentiableOperation<D>,
    LinearOperationOf<D>: ResidualizedOperation<D>
        + SupportsCustomVjpCall<D::Type, <D as Domain>::Constant, O, ResidualFactor<D::Type, <D as Domain>::Value>>,
    Vec<<D as Domain>::Constant>: Parameterized<
            <D as Domain>::Constant,
            Family: ParameterizedFamily<D::Tangent> + ParameterizedFamily<<D as Domain>::Value>,
            To<<D as Domain>::Value> = Vec<<D as Domain>::Value>,
            To<D::Tangent> = Vec<D::Tangent>,
            ParameterStructure: Debug + PartialEq,
        >,
{
    let output_count = operation.output_types().len();
    check_count!("input", inputs, operation.input_types().len(), ProgramError);
    let primal_operands = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
    let tangent_operands = inputs
        .iter()
        .map(|input| context.materialize_tangent(input.tangent().clone()))
        .collect::<Result<Vec<_>, _>>()?;
    let (mut forward_values, _pushforward) =
        context.differentiable().linearize_program(&operation.forward, primal_operands)?;
    let residuals = forward_values.split_off(output_count);
    let factors = residuals.into_iter().map(|residual| context.factor(residual)).collect::<Vec<_>>();
    let call = LinearOperationOf::<D>::custom_vjp_call_operation(operation.backward.clone(), factors, false);
    let tangent_outputs = context.stage_operation(call, tangent_operands.as_slice())?;
    check_count!("output", tangent_outputs, output_count, ProgramError);
    Ok(forward_values
        .into_iter()
        .zip(tangent_outputs)
        .map(|(primal, tangent)| JvpTracer::from_value(primal, tangent))
        .collect())
}

/// Value-level batching for [`CustomVjpOperation`]: inlines the primal program; see [`CustomJvpOperation`]'s
/// batching documentation for the custom-derivative caveat.
impl<V, O> BatchableOperation<V, ()> for CustomVjpOperation<V, O, ArrayType>
where
    V: Value<ArrayType>,
    O: Operation<ArrayType> + BatchableOperation<V>,
{
    fn batch(&self, _context: &(), inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        batch_program_inline(&self.primal, inputs)
    }
}

/// Traced batching for [`CustomVjpOperation`]: inlines the primal program into the parent trace; see
/// [`CustomJvpOperation`]'s batching documentation for the custom-derivative caveat.
impl<C, O> BatchableOperation<Tracer<C>, BatchingContext<C>> for CustomVjpOperation<C::Constant, O, ArrayType>
where
    C: StagingContext<Type = ArrayType>,
    C::Constant: Value<ArrayType>,
    O: Operation<ArrayType> + BatchableOperation<Tracer<C>, BatchingContext<C>>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<Tracer<C>>],
    ) -> Result<Vec<ArrayBatch<Tracer<C>>>, ProgramError> {
        context.interpret_program(&self.primal, inputs.to_vec())
    }
}

/// Access to a custom-VJP residual payload as a concrete value during pullback interpretation.
///
/// Implemented by plain values (identity) and by [`ResidualFactor`] (whose `Constant` form yields its payload and
/// whose `Reference` form errors, since references are only meaningful before residual instantiation).
#[doc(hidden)]
pub trait CustomVjpResidual<T: Type, V: Value<T>>: Value<T> {
    /// Returns the concrete residual value.
    fn residual_value(&self) -> Result<V, ProgramError>;
}

impl<T: Type, V: Value<T>> CustomVjpResidual<T, V> for V {
    #[inline]
    fn residual_value(&self) -> Result<V, ProgramError> {
        Ok(self.clone())
    }
}

impl<T: Type, V: Value<T>> CustomVjpResidual<T, V> for ResidualFactor<T, V> {
    fn residual_value(&self) -> Result<V, ProgramError> {
        match self {
            ResidualFactor::Constant(value) => Ok(value.clone()),
            ResidualFactor::Reference { .. } => Err(TypeError {
                message: "custom_vjp pullback interpretation requires instantiated residuals".to_string(),
            }
            .into()),
        }
    }
}

/// Opaque linear operation staged by [`CustomVjpOperation`]'s JVP rule.
///
/// In its un-transposed form it stands for the (unknown) tangent map of the custom function and rejects
/// interpretation: `custom_vjp` functions are reverse-mode-only, matching JAX. Transposition replaces it with its
/// transposed form, whose interpretation replays the user's backward program on the captured residuals and the
/// incoming output cotangents.
#[derive(Clone, Debug)]
pub struct CustomVjpCallOperation<V, O, F, T = ArrayType>
where
    T: PartialEq + Type,
    V: Value<T>,
    F: Value<T>,
{
    /// The user's backward program, mapping `(residuals..., output_cotangents...)` to input cotangents.
    backward: FlatProgram<V, O, T>,

    /// Captured residual factors consumed by the backward program.
    residuals: Vec<F>,

    /// Whether this call has been transposed into its executable (pullback) form.
    transposed: bool,
}

impl<T: PartialEq + Type, V: Value<T>, F: Value<T>, O> CustomVjpCallOperation<V, O, F, T> {
    /// Creates a custom-VJP call. Use `transposed = false` for the opaque pushforward form and `transposed = true`
    /// for the executable pullback form.
    pub fn new(backward: FlatProgram<V, O, T>, residuals: Vec<F>, transposed: bool) -> Self {
        Self { backward, residuals, transposed }
    }

    /// Returns the user's backward program.
    #[inline]
    pub fn backward(&self) -> &FlatProgram<V, O, T> {
        &self.backward
    }

    /// Returns the captured residual factors.
    #[inline]
    pub fn residuals(&self) -> &[F] {
        self.residuals.as_slice()
    }

    /// Returns whether this call is in its transposed (executable pullback) form.
    #[inline]
    pub fn transposed(&self) -> bool {
        self.transposed
    }

    /// Maps the residual factor payloads with `map_factor`, preserving the backward program and direction.
    pub fn map_factors<MappedFactor: Value<T>, MapFactorFn>(
        &self,
        map_factor: &mut MapFactorFn,
    ) -> Result<CustomVjpCallOperation<V, O, MappedFactor, T>, ProgramError>
    where
        MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>,
        V: Clone,
        O: Clone,
    {
        Ok(CustomVjpCallOperation {
            backward: self.backward.clone(),
            residuals: self.residuals.iter().map(map_factor).collect::<Result<Vec<_>, _>>()?,
            transposed: self.transposed,
        })
    }
}

impl<T: PartialEq + Type, V: Value<T>, F: Value<T>, O: Operation<T>> CustomVjpCallOperation<V, O, F, T> {
    /// Returns the cotangent types flowing *into* the backward program (one per primal output).
    fn cotangent_types(&self) -> Vec<T> {
        flat_program_input_types(&self.backward).split_off(self.residuals.len())
    }
}

impl<T: PartialEq + Type, V: Value<T>, F: Value<T>, O> Display for CustomVjpCallOperation<V, O, F, T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.transposed {
            formatter.write_str("custom_vjp_backward")
        } else {
            formatter.write_str("custom_vjp_tangent")
        }
    }
}

impl<T: PartialEq + Type, V: Value<T>, F: Value<T>, O: Operation<T>> Operation<T>
    for CustomVjpCallOperation<V, O, F, T>
{
    #[inline]
    fn name(&self) -> &'static str {
        if self.transposed { "custom_vjp_backward" } else { "custom_vjp_tangent" }
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        if self.transposed {
            ensure_types_match("custom_vjp backward cotangent", &self.cotangent_types(), input_types)?;
            Ok(flat_program_output_types(&self.backward))
        } else {
            // The un-transposed call maps input tangents (typed like the primal inputs, which are the backward
            // program's outputs) to output tangents (typed like the primal outputs, which are the backward
            // program's trailing inputs).
            ensure_types_match("custom_vjp tangent", &flat_program_output_types(&self.backward), input_types)?;
            Ok(self.cotangent_types())
        }
    }
}

impl<T, V, O, F> InterpretableOperation<T, V> for CustomVjpCallOperation<V, O, F, T>
where
    T: PartialEq + Type,
    V: Value<T>,
    F: CustomVjpResidual<T, V>,
    O: InterpretableOperation<T, V> + Operation<T>,
    Vec<V>: Parameterized<V, ParameterStructure: Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        if !self.transposed {
            return Err(TypeError {
                message: "custom_vjp does not support forward-mode differentiation; use reverse mode (vjp, \
                    value_and_grad, or jacrev) instead"
                    .to_string(),
            }
            .into());
        }
        let mut values = self.residuals.iter().map(CustomVjpResidual::residual_value).collect::<Result<Vec<_>, _>>()?;
        values.extend(inputs.iter().cloned());
        self.backward.interpret(values)
    }
}

/// Transpose rule for [`CustomVjpCallOperation`]: stages the transposed (executable) form of the call on the output
/// cotangents, materializing structural zeros so the backward program receives every cotangent input. The rule is
/// generic over the cotangent value type `W`, which need not match the backward program's value type `V`: the staged
/// transposed call carries the program and residuals along unchanged.
impl<T, V, O, F, W, OLinear> TransposableOperation<T, W, OLinear> for CustomVjpCallOperation<V, O, F, T>
where
    T: PartialEq + Type,
    V: Value<T>,
    F: Value<T>,
    W: Value<T>,
    O: Clone + Operation<T>,
    OLinear: Operation<T> + SupportsZero<T> + SupportsCustomVjpCall<T, V, O, F>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<'transpose, T, W, OLinear>,
        _input_types: &[&T],
        output_cotangents: &[Cotangent<'transpose, T, W, OLinear>],
    ) -> Result<Vec<Cotangent<'transpose, T, W, OLinear>>, ProgramError> {
        if self.transposed {
            return Err(TypeError {
                message: "transposing a custom_vjp pullback (second-order reverse mode through custom_vjp) is not \
                    supported"
                    .to_string(),
            }
            .into());
        }
        let cotangent_types = self.cotangent_types();
        check_count!("output", output_cotangents, cotangent_types.len(), ProgramError);
        let cotangent_tracers = output_cotangents
            .iter()
            .zip(cotangent_types.iter())
            .map(|(cotangent, r#type)| stage_cotangent(context, cotangent, r#type))
            .collect::<Vec<_>>();
        let call = OLinear::custom_vjp_call_operation(self.backward.clone(), self.residuals.to_vec(), true);
        let outputs = context.stage_operation(call, cotangent_tracers.as_slice())?;
        Ok(outputs.into_iter().map(Cotangent::Staged).collect())
    }
}

/// Value-level batching for the transposed [`CustomVjpCallOperation`]: replays the backward program through the
/// per-operation batching rules with the captured residuals as lane-uniform values. Used when a pullback containing
/// custom-VJP calls is interpreted with batched cotangents (e.g., by `jacrev`). The un-transposed form rejects
/// batching just as it rejects interpretation.
impl<V, O, F> BatchableOperation<V, ()> for CustomVjpCallOperation<V, O, F, ArrayType>
where
    V: Value<ArrayType>,
    F: CustomVjpResidual<ArrayType, V>,
    O: Operation<ArrayType> + BatchableOperation<V>,
{
    fn batch(&self, _context: &(), inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        if !self.transposed {
            return Err(TypeError {
                message: "custom_vjp does not support forward-mode differentiation; use reverse mode (vjp, \
                    value_and_grad, or jacrev) instead"
                    .to_string(),
            }
            .into());
        }
        let mut values = self
            .residuals
            .iter()
            .map(|residual| Ok(ArrayBatch::unbatched(residual.residual_value()?)))
            .collect::<Result<Vec<_>, ProgramError>>()?;
        values.extend(inputs.iter().cloned());
        self.backward.interpret_with(
            values,
            |_, constant: &V| Ok(ArrayBatch::unbatched(constant.clone())),
            |instruction, instruction_inputs| instruction.operation().batch(&(), instruction_inputs),
        )
    }
}

/// Trait that represents [`Operation`] types that support/include [`CustomJvpOperation`]. Backend-owned closed
/// [`Operation`] types implement this trait so that generic code — most notably [`CustomJvp::call`] — can stage
/// custom-JVP calls without knowing which operation enum is in use.
pub trait SupportsCustomJvp<T: PartialEq + Type, V: Value<T>>: Sized {
    /// Wraps `operation` into this [`Operation`] type.
    fn custom_jvp_operation(operation: CustomJvpOperation<V, Self, T>) -> Self;
}

/// Trait that represents [`Operation`] types that support/include [`CustomVjpOperation`]. Backend-owned closed
/// [`Operation`] types implement this trait so that generic code — most notably [`CustomVjp::call`] — can stage
/// custom-VJP calls without knowing which operation enum is in use.
pub trait SupportsCustomVjp<T: PartialEq + Type, V: Value<T>>: Sized {
    /// Wraps `operation` into this [`Operation`] type.
    fn custom_vjp_operation(operation: CustomVjpOperation<V, Self, T>) -> Self;
}

/// Function with a user-supplied JVP rule — the ergonomic analogue of JAX's
/// [`jax.custom_jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html) /
/// [`defjvp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.defjvp.html) decorator pair, built by
/// [`custom_jvp`].
///
/// The primal function and its JVP rule are stored as plain closures over [`DomainTracer`]s. Nothing is traced at
/// construction time: each [`call`](Self::call) reads the input types off its tracer arguments, traces both closures
/// into [`FlatProgram`]s specialized to those types, validates the rule signature, and stages one
/// [`CustomJvpOperation`] into the caller's staging context — mirroring how JAX traces rule functions into jaxprs
/// lazily at transform time. The closures follow the operation's flat calling convention: `primal` maps the inputs
/// to the outputs, and `jvp` maps `(inputs..., input_tangents...)` to `(outputs..., output_tangents...)`.
///
/// The primal closure is kept separate from the JVP closure for efficiency rather than necessity: the JVP rule
/// computes both the outputs and their tangents, so deriving the primal from it would make every un-differentiated
/// call pay for tangent computation. Interpretation, batching, and backend lowering replay the lean primal program;
/// the JVP program runs only under differentiation.
pub struct CustomJvp<'d, D: Domain, P, J> {
    /// Domain whose constant and operation types the captured programs are traced over.
    domain: &'d D,

    /// Closure computing the primal outputs from the primal inputs.
    primal: P,

    /// Closure computing `(outputs..., output_tangents...)` from `(inputs..., input_tangents...)`.
    jvp: J,
}

/// Creates a [`CustomJvp`] function from a primal closure and a JVP-rule closure over `domain`'s tracers. Refer to
/// the documentation of [`CustomJvp`] for the calling convention and tracing semantics.
pub fn custom_jvp<'d, D, P, J>(domain: &'d D, primal: P, jvp: J) -> CustomJvp<'d, D, P, J>
where
    D: Domain,
    P: Fn(Vec<DomainTracer<'d, D>>) -> Result<Vec<DomainTracer<'d, D>>, ProgramError>,
    J: Fn(Vec<DomainTracer<'d, D>>) -> Result<Vec<DomainTracer<'d, D>>, ProgramError>,
{
    CustomJvp { domain, primal, jvp }
}

impl<'d, D, P, J> CustomJvp<'d, D, P, J>
where
    D: Domain<Type: PartialEq>,
    P: Fn(Vec<DomainTracer<'d, D>>) -> Result<Vec<DomainTracer<'d, D>>, ProgramError>,
    J: Fn(Vec<DomainTracer<'d, D>>) -> Result<Vec<DomainTracer<'d, D>>, ProgramError>,
    D::Operation: Operation<D::Type> + SupportsCustomJvp<D::Type, D::Constant>,
    Vec<D::Type>: Parameterized<
            D::Type,
            Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<DomainTracer<'d, D>>,
            To<DomainTracer<'d, D>> = Vec<DomainTracer<'d, D>>,
            To<D::Constant> = Vec<D::Constant>,
        >,
    Vec<DomainTracer<'d, D>>: Parameterized<
            DomainTracer<'d, D>,
            Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>,
            To<D::Type> = Vec<D::Type>,
            To<D::Constant> = Vec<D::Constant>,
        >,
{
    /// Stages this custom-JVP function on the provided tracer inputs and returns its outputs, tracing the stored
    /// closures into programs specialized to the inputs' types. Differentiation of the staged call replays the JVP
    /// rule instead of differentiating the primal body, in both forward and reverse mode.
    pub fn call<C>(&self, inputs: &[Tracer<C>]) -> Result<Vec<Tracer<C>>, ProgramError>
    where
        C: StagingContext<Type = D::Type, Constant = D::Constant, Operation = D::Operation>,
    {
        let Some(first) = inputs.first() else {
            return Err(TypeError { message: "custom_jvp requires at least one input".to_string() }.into());
        };
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let (_, primal) = TracingContext::trace(self.domain, |xs| (self.primal)(xs), input_types.clone())?;
        let jvp_input_types = input_types.iter().chain(input_types.iter()).cloned().collect::<Vec<_>>();
        let (_, jvp) = TracingContext::trace(self.domain, |xs| (self.jvp)(xs), jvp_input_types)?;
        let operation = D::Operation::custom_jvp_operation(CustomJvpOperation::new(primal, jvp)?);
        first.context().stage_operation(operation, inputs)
    }
}

/// Function with user-supplied forward/backward (VJP) rules — the ergonomic analogue of JAX's
/// [`jax.custom_vjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html) /
/// [`defvjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.defvjp.html) decorator pair, built by
/// [`custom_vjp`].
///
/// The primal function and its forward/backward rules are stored as plain closures over [`DomainTracer`]s. Nothing
/// is traced at construction time: each [`call`](Self::call) reads the input types off its tracer arguments, traces
/// the closures into [`FlatProgram`]s specialized to those types, validates the rule signatures, and stages one
/// [`CustomVjpOperation`] into the caller's staging context — mirroring how JAX traces rule functions into jaxprs
/// lazily at transform time. The closures follow the operation's flat calling convention: `primal` maps the inputs
/// to the outputs, `forward` maps the inputs to the outputs followed by arbitrarily many residuals, and `backward`
/// maps `(residuals..., output_cotangents...)` to one cotangent per primal input. As in JAX, the resulting function
/// supports reverse mode only; forward-mode differentiation of a staged call is rejected.
///
/// The primal closure is kept separate from the forward closure for efficiency rather than necessity: an
/// un-differentiated call should not pay for residual computation. Interpretation, batching, and backend lowering
/// replay the lean primal program; the residual-producing forward program runs only under reverse-mode
/// differentiation. Callers that do not care about the distinction can pass the same body for both — accepting that
/// the residual outputs are dead code outside of differentiation — which mirrors the common JAX idiom of writing
/// `f_fwd` as `return f(x), residuals`.
pub struct CustomVjp<'d, D: Domain, P, F, B> {
    /// Domain whose constant and operation types the captured programs are traced over.
    domain: &'d D,

    /// Closure computing the primal outputs from the primal inputs.
    primal: P,

    /// Closure computing `(outputs..., residuals...)` from the primal inputs.
    forward: F,

    /// Closure computing one input cotangent per primal input from `(residuals..., output_cotangents...)`.
    backward: B,
}

/// Creates a [`CustomVjp`] function from primal, forward, and backward closures over `domain`'s tracers. Refer to
/// the documentation of [`CustomVjp`] for the calling convention and tracing semantics.
pub fn custom_vjp<'d, D, P, F, B>(domain: &'d D, primal: P, forward: F, backward: B) -> CustomVjp<'d, D, P, F, B>
where
    D: Domain,
    P: Fn(Vec<DomainTracer<'d, D>>) -> Result<Vec<DomainTracer<'d, D>>, ProgramError>,
    F: Fn(Vec<DomainTracer<'d, D>>) -> Result<Vec<DomainTracer<'d, D>>, ProgramError>,
    B: Fn(Vec<DomainTracer<'d, D>>) -> Result<Vec<DomainTracer<'d, D>>, ProgramError>,
{
    CustomVjp { domain, primal, forward, backward }
}

impl<'d, D, P, F, B> CustomVjp<'d, D, P, F, B>
where
    D: Domain<Type: PartialEq>,
    P: Fn(Vec<DomainTracer<'d, D>>) -> Result<Vec<DomainTracer<'d, D>>, ProgramError>,
    F: Fn(Vec<DomainTracer<'d, D>>) -> Result<Vec<DomainTracer<'d, D>>, ProgramError>,
    B: Fn(Vec<DomainTracer<'d, D>>) -> Result<Vec<DomainTracer<'d, D>>, ProgramError>,
    D::Operation: Operation<D::Type> + SupportsCustomVjp<D::Type, D::Constant>,
    Vec<D::Type>: Parameterized<
            D::Type,
            Family: ParameterizedFamily<D::Constant> + ParameterizedFamily<DomainTracer<'d, D>>,
            To<DomainTracer<'d, D>> = Vec<DomainTracer<'d, D>>,
            To<D::Constant> = Vec<D::Constant>,
        >,
    Vec<DomainTracer<'d, D>>: Parameterized<
            DomainTracer<'d, D>,
            Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>,
            To<D::Type> = Vec<D::Type>,
            To<D::Constant> = Vec<D::Constant>,
        >,
{
    /// Stages this custom-VJP function on the provided tracer inputs and returns its outputs, tracing the stored
    /// closures into programs specialized to the inputs' types. Reverse-mode differentiation of the staged call
    /// replays the backward rule on the forward rule's residuals instead of differentiating the primal body.
    pub fn call<C>(&self, inputs: &[Tracer<C>]) -> Result<Vec<Tracer<C>>, ProgramError>
    where
        C: StagingContext<Type = D::Type, Constant = D::Constant, Operation = D::Operation>,
    {
        let Some(first) = inputs.first() else {
            return Err(TypeError { message: "custom_vjp requires at least one input".to_string() }.into());
        };
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let (primal_output_types, primal) =
            TracingContext::trace(self.domain, |xs| (self.primal)(xs), input_types.clone())?;
        let (forward_output_types, forward) =
            TracingContext::trace(self.domain, |xs| (self.forward)(xs), input_types.clone())?;
        if forward_output_types.len() < primal_output_types.len() {
            return Err(TypeError {
                message: format!(
                    "custom_vjp forward must produce at least the {} primal output(s) but produced {} value(s)",
                    primal_output_types.len(),
                    forward_output_types.len(),
                ),
            }
            .into());
        }
        let backward_input_types = forward_output_types[primal_output_types.len()..]
            .iter()
            .chain(primal_output_types.iter())
            .cloned()
            .collect::<Vec<_>>();
        let (_, backward) = TracingContext::trace(self.domain, |xs| (self.backward)(xs), backward_input_types)?;
        let operation = D::Operation::custom_vjp_operation(CustomVjpOperation::new(primal, forward, backward)?);
        first.context().stage_operation(operation, inputs)
    }
}

#[cfg(test)]
mod tests {
    use crate::contexts::StagingContext;
    use crate::operations::scalars::ScalarOperation;
    use crate::operations::trigonometric::{Cos, Sin};
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::scalars::ScalarDomain;
    use crate::tracing_v2::test_util::{TestArray, TestArrayDomain, assert_close};
    use crate::tracing_v2::{ArrayOperation, Batch, DifferentiationContext, value_and_grad};
    use crate::types::{DataType, Shape, Size};

    use super::*;

    /// Returns the canonical test array type with the provided dimensions.
    fn test_type(dimensions: &[usize]) -> ArrayType {
        ArrayType::new(
            DataType::F64,
            Shape::new(dimensions.iter().map(|dimension| Size::Static(*dimension)).collect()),
            None,
            None,
        )
        .unwrap()
    }

    /// Builds `f(x) = sin(x)` over one input of the provided type.
    fn sin_program(r#type: &ArrayType) -> FlatProgram<TestArray, ArrayOperation<TestArray, ArrayType>> {
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(r#type.clone());
        let output = builder.add_instruction(ArrayOperation::Sin, vec![input]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds the deliberately wrong rule `jvp(x, dx) = (sin(x), 2 * cos(x) * dx)`, detectably different from the
    /// true derivative so tests can prove the custom rule is used.
    fn doubled_sin_jvp_program(r#type: &ArrayType) -> FlatProgram<TestArray, ArrayOperation<TestArray, ArrayType>> {
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(r#type.clone());
        let dx = builder.add_input(r#type.clone());
        let y = builder.add_instruction(ArrayOperation::Sin, vec![x]).unwrap()[0];
        let cosine = builder.add_instruction(ArrayOperation::Cos, vec![x]).unwrap()[0];
        let two = builder.add_constant(TestArray::scalar(2.0));
        let scaled = builder.add_instruction(ArrayOperation::Mul, vec![two, cosine]).unwrap()[0];
        let tangent = builder.add_instruction(ArrayOperation::Mul, vec![scaled, dx]).unwrap()[0];
        builder
            .build(vec![y, tangent], vec![Placeholder, Placeholder], vec![Placeholder, Placeholder])
            .unwrap()
    }

    /// Builds the forward rule `forward(x) = (sin(x), cos(x))`, with the cosine as the residual.
    fn sin_forward_program(r#type: &ArrayType) -> FlatProgram<TestArray, ArrayOperation<TestArray, ArrayType>> {
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(r#type.clone());
        let y = builder.add_instruction(ArrayOperation::Sin, vec![x]).unwrap()[0];
        let residual = builder.add_instruction(ArrayOperation::Cos, vec![x]).unwrap()[0];
        builder.build(vec![y, residual], vec![Placeholder], vec![Placeholder, Placeholder]).unwrap()
    }

    /// Builds the deliberately wrong rule `backward(residual, cotangent) = 3 * residual * cotangent`, detectably
    /// different from the true gradient so tests can prove the custom rule is used.
    fn tripled_sin_backward_program(
        r#type: &ArrayType,
    ) -> FlatProgram<TestArray, ArrayOperation<TestArray, ArrayType>> {
        let mut builder = ProgramBuilder::new();
        let residual = builder.add_input(r#type.clone());
        let cotangent = builder.add_input(r#type.clone());
        let three = builder.add_constant(TestArray::scalar(3.0));
        let scaled = builder.add_instruction(ArrayOperation::Mul, vec![three, residual]).unwrap()[0];
        let gradient = builder.add_instruction(ArrayOperation::Mul, vec![scaled, cotangent]).unwrap()[0];
        builder.build(vec![gradient], vec![Placeholder, Placeholder], vec![Placeholder]).unwrap()
    }

    fn custom_jvp_sin(r#type: &ArrayType) -> ArrayOperation<TestArray, ArrayType> {
        ArrayOperation::CustomJvp(Box::new(
            CustomJvpOperation::new(sin_program(r#type), doubled_sin_jvp_program(r#type)).unwrap(),
        ))
    }

    fn custom_vjp_sin(r#type: &ArrayType) -> ArrayOperation<TestArray, ArrayType> {
        ArrayOperation::CustomVjp(Box::new(
            CustomVjpOperation::new(
                sin_program(r#type),
                sin_forward_program(r#type),
                tripled_sin_backward_program(r#type),
            )
            .unwrap(),
        ))
    }

    #[test]
    fn test_custom_jvp_construction_validates_the_rule_signature() {
        let scalar = test_type(&[]);
        // The JVP program must take `(inputs..., tangents...)`; a primal-only signature is rejected.
        assert!(CustomJvpOperation::new(sin_program(&scalar), sin_program(&scalar)).is_err());
    }

    #[test]
    fn test_custom_vjp_construction_validates_the_rule_signatures() {
        let scalar = test_type(&[]);
        // The backward program must consume `(residuals..., output cotangents...)`; a single-input program whose
        // signature cannot line up with the forward residuals is rejected.
        assert!(
            CustomVjpOperation::new(sin_program(&scalar), sin_forward_program(&scalar), sin_program(&scalar)).is_err()
        );
    }

    #[test]
    fn test_custom_jvp_interprets_the_primal_program() {
        let scalar = test_type(&[]);
        let outputs = custom_jvp_sin(&scalar).interpret(&[TestArray::scalar(2.0)]).unwrap();
        assert_close(outputs[0].values[0], 2.0f64.sin());
    }

    #[test]
    fn test_custom_jvp_governs_forward_mode() {
        let scalar = test_type(&[]);
        let (primal, tangent) = TestArrayDomain
            .jvp(
                |x| {
                    let operation = custom_jvp_sin(&test_type(&[]));
                    x.context().stage_operation(operation, &[&x]).unwrap().into_iter().next().unwrap()
                },
                TestArray::scalar(2.0),
                TestArray::scalar(1.0),
            )
            .unwrap();
        let _ = scalar;
        assert_close(primal.values[0], 2.0f64.sin());
        // The custom rule doubles the true derivative, proving it is in control.
        assert_close(tangent.values[0], 2.0 * 2.0f64.cos());
    }

    #[test]
    fn test_custom_jvp_governs_reverse_mode() {
        let (value, gradient) = value_and_grad(
            &TestArrayDomain,
            |x| {
                let operation = custom_jvp_sin(&test_type(&[]));
                x.context().stage_operation(operation, &[&x]).unwrap().into_iter().next().unwrap()
            },
            TestArray::scalar(3.0),
        )
        .unwrap();
        assert_close(value.values[0], 3.0f64.sin());
        // Reverse mode transposes the linearized custom rule, so the doubled derivative carries over.
        assert_close(gradient.values[0], 2.0 * 3.0f64.cos());
    }

    #[test]
    fn test_custom_vjp_governs_reverse_mode() {
        let (value, gradient) = value_and_grad(
            &TestArrayDomain,
            |x| {
                let operation = custom_vjp_sin(&test_type(&[]));
                x.context().stage_operation(operation, &[&x]).unwrap().into_iter().next().unwrap()
            },
            TestArray::scalar(2.0),
        )
        .unwrap();
        assert_close(value.values[0], 2.0f64.sin());
        // The custom backward rule triples the true gradient, proving it is in control.
        assert_close(gradient.values[0], 3.0 * 2.0f64.cos());
    }

    #[test]
    fn test_custom_vjp_rejects_forward_mode() {
        // The staged linear call refuses interpretation in its un-transposed (pushforward) form, which is exactly
        // the operation `jvp` would need to execute; reverse mode transposes it first and replays `backward`.
        let scalar = test_type(&[]);
        let call = CustomVjpCallOperation::<TestArray, ArrayOperation<TestArray, ArrayType>, TestArray>::new(
            tripled_sin_backward_program(&scalar),
            vec![TestArray::scalar(2.0f64.cos())],
            false,
        );
        assert!(matches!(
            call.interpret(&[TestArray::scalar(1.0)]),
            Err(ProgramError::Type(TypeError { message }))
                if message.starts_with("custom_vjp does not support forward-mode differentiation"),
        ));
    }

    #[test]
    fn test_jacrev_through_custom_vjp_uses_the_custom_backward_rule() {
        use crate::tracing_v2::jacrev;

        // jacrev interprets the pullback with lane-stacked cotangent bases, exercising the batched replay of the
        // custom backward program. The Jacobian of elementwise `sin` with the tripled rule is the diagonal matrix
        // `diag(3 * cos(x))`.
        let vector = test_type(&[2]);
        let jacobian = jacrev(
            &TestArrayDomain,
            |x| {
                let operation = custom_vjp_sin(&test_type(&[2]));
                Ok(x.context().stage_operation(operation, &[&x])?.into_iter().next().unwrap())
            },
            TestArray::new(vector, vec![0.5, 1.0]),
        )
        .unwrap();
        let (_, _, block) = jacobian.iter_blocks().next().unwrap();
        assert_close(block.values()[0], 3.0 * 0.5f64.cos());
        assert_close(block.values()[1], 0.0);
        assert_close(block.values()[2], 0.0);
        assert_close(block.values()[3], 3.0 * 1.0f64.cos());
    }

    /// Builds the scalar `f(x) = sin(x)` program.
    fn scalar_sin_program() -> FlatProgram<f64, ScalarOperation<f64>, DataType> {
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(ScalarOperation::Sin, vec![input]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds the deliberately wrong scalar rule `jvp(x, dx) = (sin(x), 2 * cos(x) * dx)`.
    fn scalar_doubled_sin_jvp_program() -> FlatProgram<f64, ScalarOperation<f64>, DataType> {
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(DataType::F64);
        let dx = builder.add_input(DataType::F64);
        let y = builder.add_instruction(ScalarOperation::Sin, vec![x]).unwrap()[0];
        let cosine = builder.add_instruction(ScalarOperation::Cos, vec![x]).unwrap()[0];
        let two = builder.add_constant(2.0);
        let scaled = builder.add_instruction(ScalarOperation::Mul, vec![two, cosine]).unwrap()[0];
        let tangent = builder.add_instruction(ScalarOperation::Mul, vec![scaled, dx]).unwrap()[0];
        builder
            .build(vec![y, tangent], vec![Placeholder, Placeholder], vec![Placeholder, Placeholder])
            .unwrap()
    }

    /// Builds the scalar forward rule `forward(x) = (sin(x), cos(x))`, with the cosine as the residual.
    fn scalar_sin_forward_program() -> FlatProgram<f64, ScalarOperation<f64>, DataType> {
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(DataType::F64);
        let y = builder.add_instruction(ScalarOperation::Sin, vec![x]).unwrap()[0];
        let residual = builder.add_instruction(ScalarOperation::Cos, vec![x]).unwrap()[0];
        builder.build(vec![y, residual], vec![Placeholder], vec![Placeholder, Placeholder]).unwrap()
    }

    /// Builds the deliberately wrong scalar rule `backward(residual, cotangent) = 3 * residual * cotangent`.
    fn scalar_tripled_sin_backward_program() -> FlatProgram<f64, ScalarOperation<f64>, DataType> {
        let mut builder = ProgramBuilder::new();
        let residual = builder.add_input(DataType::F64);
        let cotangent = builder.add_input(DataType::F64);
        let three = builder.add_constant(3.0);
        let scaled = builder.add_instruction(ScalarOperation::Mul, vec![three, residual]).unwrap()[0];
        let gradient = builder.add_instruction(ScalarOperation::Mul, vec![scaled, cotangent]).unwrap()[0];
        builder.build(vec![gradient], vec![Placeholder, Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_scalar_custom_jvp_governs_forward_mode() {
        let (primal, tangent) = ScalarDomain::<f64>::new()
            .jvp(
                |x| {
                    let operation = ScalarOperation::CustomJvp(Box::new(
                        CustomJvpOperation::new(scalar_sin_program(), scalar_doubled_sin_jvp_program()).unwrap(),
                    ));
                    x.context().stage_operation(operation, &[&x]).unwrap().into_iter().next().unwrap()
                },
                2.0,
                1.0,
            )
            .unwrap();
        assert_close(primal, 2.0f64.sin());
        // The custom rule doubles the true derivative, proving it is in control.
        assert_close(tangent, 2.0 * 2.0f64.cos());
    }

    #[test]
    fn test_scalar_custom_vjp_governs_reverse_mode() {
        let (value, gradient) = value_and_grad(
            &ScalarDomain::<f64>::new(),
            |x| {
                let operation = ScalarOperation::CustomVjp(Box::new(
                    CustomVjpOperation::new(
                        scalar_sin_program(),
                        scalar_sin_forward_program(),
                        scalar_tripled_sin_backward_program(),
                    )
                    .unwrap(),
                ));
                x.context().stage_operation(operation, &[&x]).unwrap().into_iter().next().unwrap()
            },
            2.0,
        )
        .unwrap();
        assert_close(value, 2.0f64.sin());
        // The custom backward rule triples the true gradient, proving it is in control.
        assert_close(gradient, 3.0 * 2.0f64.cos());
    }

    #[test]
    fn test_custom_jvp_wrapper_traces_closures_lazily() {
        // No manual programs: the wrapper traces the closures at the call site, specialized to the input types.
        let domain = TestArrayDomain;
        let function = custom_jvp(
            &domain,
            |inputs| Ok(vec![inputs[0].clone().sin()]),
            |inputs| {
                // The deliberately wrong rule `jvp(x, dx) = (sin(x), cos(x) * dx + cos(x) * dx)` doubles the true
                // derivative (expressed through addition to avoid constant lifting), proving the rule is in control.
                let tangent = inputs[0].clone().cos() * inputs[1].clone();
                Ok(vec![inputs[0].clone().sin(), tangent.clone() + tangent])
            },
        );
        let (primal, tangent) = TestArrayDomain
            .jvp(
                |x| function.call(&[x]).unwrap().into_iter().next().unwrap(),
                TestArray::scalar(2.0),
                TestArray::scalar(1.0),
            )
            .unwrap();
        assert_close(primal.values[0], 2.0f64.sin());
        assert_close(tangent.values[0], 2.0 * 2.0f64.cos());
        // Reverse mode transposes the linearized custom rule, so the doubled derivative carries over.
        let (value, gradient) = value_and_grad(
            &TestArrayDomain,
            |x| function.call(&[x]).unwrap().into_iter().next().unwrap(),
            TestArray::scalar(3.0),
        )
        .unwrap();
        assert_close(value.values[0], 3.0f64.sin());
        assert_close(gradient.values[0], 2.0 * 3.0f64.cos());
    }

    #[test]
    fn test_custom_vjp_wrapper_governs_reverse_mode() {
        let domain = TestArrayDomain;
        let function = custom_vjp(
            &domain,
            |inputs| Ok(vec![inputs[0].clone().sin()]),
            |inputs| Ok(vec![inputs[0].clone().sin(), inputs[0].clone().cos()]),
            |inputs| {
                // The deliberately wrong rule `backward(residual, cotangent) = 3 * residual * cotangent` triples the
                // true gradient (expressed through addition to avoid constant lifting).
                let product = inputs[0].clone() * inputs[1].clone();
                Ok(vec![product.clone() + product.clone() + product])
            },
        );
        let (value, gradient) = value_and_grad(
            &TestArrayDomain,
            |x| function.call(&[x]).unwrap().into_iter().next().unwrap(),
            TestArray::scalar(2.0),
        )
        .unwrap();
        assert_close(value.values[0], 2.0f64.sin());
        assert_close(gradient.values[0], 3.0 * 2.0f64.cos());
    }

    #[test]
    fn test_scalar_custom_vjp_wrapper_governs_reverse_mode() {
        let domain = ScalarDomain::<f64>::new();
        let function = custom_vjp(
            &domain,
            |inputs| Ok(vec![inputs[0].clone().sin()]),
            |inputs| Ok(vec![inputs[0].clone().sin(), inputs[0].clone().cos()]),
            |inputs| {
                let product = inputs[0].clone() * inputs[1].clone();
                Ok(vec![product.clone() + product.clone() + product])
            },
        );
        let (value, gradient) =
            value_and_grad(&domain, |x| function.call(&[x]).unwrap().into_iter().next().unwrap(), 2.0).unwrap();
        assert_close(value, 2.0f64.sin());
        assert_close(gradient, 3.0 * 2.0f64.cos());
    }

    #[test]
    fn test_custom_jvp_wrapper_surfaces_rule_signature_mismatches() {
        // The rule closure produces only the primal output (no tangent), so the traced JVP program fails the
        // signature validation that `CustomJvpOperation::new` performs at the call site.
        let domain = TestArrayDomain;
        let function =
            custom_jvp(&domain, |inputs| Ok(vec![inputs[0].clone().sin()]), |inputs| Ok(vec![inputs[0].clone().sin()]));
        let error = crate::tracing::TracingContext::trace(
            &domain,
            |inputs: Vec<_>| function.call(&inputs),
            vec![test_type(&[])],
        )
        .unwrap_err();
        assert!(error.to_string().contains("custom_jvp rule output"));
    }

    #[test]
    fn test_custom_jvp_batches_by_inlining_the_primal() {
        let scalar = test_type(&[]);
        let output: TestArray = TestArrayDomain
            .batch(
                |x| {
                    let operation = custom_jvp_sin(&scalar);
                    Ok(x.context().stage_operation(operation, &[&x])?.into_iter().next().unwrap())
                },
                TestArray::vector(vec![0.5, 1.0, 1.5]),
                Some(0),
                Some(0),
                None,
            )
            .unwrap();
        for (actual, input) in output.values.iter().zip([0.5f64, 1.0, 1.5]) {
            assert_close(*actual, input.sin());
        }
    }
}
