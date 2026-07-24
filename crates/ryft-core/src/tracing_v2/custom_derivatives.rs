//! Contains higher-order custom-derivative operations and their transformation rules.

use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::axes::Axis;
use crate::batching::{
    ArrayBatch, BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError,
    ProgramBatchingOutputAxesPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    TransposableOperation, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, check_types};
use crate::operations::constants::{Zero, ZeroOperation};
use crate::operations::manipulation::{Broadcast, BroadcastOperation, Transpose, TransposeOperation};
use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::operations::Operation;
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::{MaybeZero, Program, ProgramError, Value};
use crate::tracing::{DomainTracer, Trace, Tracer, TracingContext};
use crate::types::ArrayType;

/// Higher-order operation pairing a primal program with a user-supplied JVP program — the direct analogue of JAX's
/// [`custom_jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html).
///
/// The JVP program follows JAX's calling convention: it receives the primal inputs followed by one tangent per
/// primal input, and returns the primal outputs followed by one tangent per primal output. The primal program is
/// kept separate from the JVP program so that un-differentiated calls do not pay for tangent computation:
/// interpretation and backend lowering replay the lean primal program; linearization replays the JVP
/// program instead of differentiating the primal body, so the user-supplied derivative governs both forward and
/// reverse mode (reverse mode transposes the linearization of the JVP program, which therefore must be linear in
/// its tangent arguments).
///
/// Batching (`batch`) re-wraps the call around batched primal/JVP programs — mirroring JAX's
/// `custom_jvp_call_jaxpr` batching rule — so the custom derivative survives a `batch` applied *before*
/// differentiation, under eager and staging parents alike.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CustomJvpOperation;

impl CustomJvpOperation {
    /// Creates a custom-JVP operation. The primal and JVP [`Program`]s are supplied separately as the operation's
    /// attached regions (via the region driver passed to [`Context::bind`]) in the region order
    /// `["primal", "jvp"]`; [`Operation::infer_output_types`] validates that the JVP interface matches the primal
    /// interface (its inputs are the primal inputs followed by their tangents, and its outputs the primal outputs
    /// followed by their tangents).
    #[inline]
    pub fn new() -> Self {
        Self
    }
}

impl Default for CustomJvpOperation {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Display for CustomJvpOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("custom_jvp")
    }
}

/// Validates the custom-JVP contract over the two attached region interfaces (`["primal", "jvp"]` region order) and
/// returns the primal interface: the JVP interface's inputs must be the primal inputs followed by values using their
/// tangent boundary types, and its outputs must be the primal outputs followed by correspondingly typed tangent
/// values.
fn validated_custom_jvp_interfaces<T: DifferentiableType>(
    region_interfaces: &[RegionInterface<T>],
) -> Result<&RegionInterface<T>, TypeError> {
    if region_interfaces.len() != 2 {
        return Err(TypeError::invalid(format!(
            "custom_jvp expects 2 attached regions but got {}",
            region_interfaces.len()
        )));
    }
    let primal_interface = &region_interfaces[0];
    let jvp_interface = &region_interfaces[1];
    let input_types = primal_interface.input_types();
    let output_types = primal_interface.output_types();
    let input_tangent_types = input_types.iter().map(|r#type| r#type.tangent());
    let expected_jvp_input_types: Vec<T> = input_types.iter().cloned().chain(input_tangent_types).collect();
    check_types!(@same, "custom_jvp rule input", [&expected_jvp_input_types, jvp_interface.input_types()]);
    let output_tangent_types = output_types.iter().map(|r#type| r#type.tangent());
    let expected_jvp_output_types: Vec<T> = output_types.iter().cloned().chain(output_tangent_types).collect();
    check_types!(@same, "custom_jvp rule output", [&expected_jvp_output_types, jvp_interface.output_types()]);
    Ok(primal_interface)
}

impl<T: DifferentiableType> Operation<T> for CustomJvpOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "custom_jvp"
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        let primal_interface = validated_custom_jvp_interfaces(region_interfaces)?;
        check_types!(@same, "custom_jvp input", [primal_interface.input_types(), input_types]);
        Ok(primal_interface.output_types().to_vec())
    }

    #[inline]
    fn region_names(&self) -> &'static [&'static str] {
        &["primal", "jvp"]
    }
}

impl<C: Domain<Type: DifferentiableType>> InterpretableOperation<C> for CustomJvpOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        driver.interpret_region(context, 0, inputs.to_vec())
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of [`Program::partially_evaluate`] for a
/// [`CustomJvpOperation`]: a call with all-known operands folds by interpreting its primal, and otherwise
/// residualizes unchanged.
impl<C: Context<Type: DifferentiableType>> PartiallyEvaluatableOperation<C> for CustomJvpOperation where
    C::Operation: From<CustomJvpOperation>
{
}

/// Capture-free forward-mode (JVP) rule for [`CustomJvpOperation`]: replays the user-supplied JVP program through the
/// active context, staging its operations in the shared builder.
///
/// The JVP program is already JVP-shaped over the primal operation family — it maps
/// `(inputs..., input_tangents...)` to `(outputs..., output_tangents...)` — so the rule simply replays it
/// through [`Program::interpret_in_context`](crate::Program::interpret_in_context) over the dual inputs: the primal tracers followed by the
/// tangent tracers feed the JVP
/// program, and its outputs split into the primal outputs and the staged output tangents. Because the replayed program
/// is straight-line primal-enum operations referencing those tracers directly, it introduces no symbolic capture and
/// the enclosing partial-evaluation split discovers the residual operand edges structurally — so
/// the rule is a leaf needing no nested differentiation or linearization request, and reverse mode transposes the
/// replayed bilinear operations exactly as it does for any other straight-line
/// tangent program.
impl<C: Context<Type: DifferentiableType> + Zero<C::Value>> DifferentiableOperation<C> for CustomJvpOperation {
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The user's JVP computation is region 1 (region 0 is the primal), mapping
        // `(inputs..., input_tangents...)` to `(outputs..., output_tangents...)`.
        let jvp_region = driver.region(1)?;
        let output_count = jvp_region.output_types().len() / 2;
        check_count!("input", inputs, jvp_region.input_types().len() / 2, ProgramError);
        // The JVP region consumes `(primals..., input_tangents...)`, so feed the dual primals followed by the dual
        // tangents.
        let mut jvp_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        // The user's JVP region takes every input tangent as a real region input, so materialize structural
        // zeros.
        for input in inputs {
            jvp_inputs.push(input.tangent().clone().materialize(context)?);
        }
        let mut outputs = jvp_region.interpret_in_context(context, jvp_inputs)?;
        check_count!("output", outputs, 2 * output_count, ProgramError);
        let tangents = outputs.split_off(output_count);
        Ok(outputs
            .into_iter()
            .zip(tangents)
            .map(|(primal, tangent)| DifferentiationDual::new(primal, tangent))
            .collect::<Result<Vec<_>, _>>()?)
    }
}

crate::impl_non_transposable_operation!(CustomJvpOperation);

/// Binds a re-wrapped custom-derivative call into the batching context's parent.
///
/// This is the shared body of the `batch` rules for [`CustomJvpOperation`] and [`CustomVjpOperation`],
/// mirroring JAX's `custom_jvp_call_jaxpr` / `custom_vjp_call_jaxpr` batching rules: instead of inlining the primal
/// program (which would lose the custom derivative and any rematerialization structure), the rule binds one new
/// custom-derivative call whose captured programs have been batched — interpreted eagerly under an eager parent and
/// staged into the enclosing trace under a staging parent. When no input carries the mapped batch axis the
/// original operation is bound unchanged and the outputs stay replicated. Otherwise every input is aligned to
/// carry the batch axis at axis `0` (replicated inputs are broadcast, matching the convention that every
/// custom-call input is mapped at axis `0`) and every output carries the batch axis at axis `0`.
pub(crate) fn stage_rewrapped_custom_call<C, MakeOperationFn>(
    context: &BatchingContext<C>,
    inputs: &[ArrayBatch<<C as Domain>::Value>],
    make_operation: MakeOperationFn,
) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError>
where
    C: Context<Type = ArrayType>,
    <C as Domain>::Value: Broadcast + Transpose,
    MakeOperationFn: FnOnce(
        Option<usize>,
    ) -> Result<
        (C::Operation, Vec<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>),
        BatchingError,
    >,
{
    if inputs.iter().all(|input| input.batch_axis().is_replicated()) {
        let (operation, operation_regions) = make_operation(None)?;
        let parent_inputs = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
        let outputs = context.parent().bind(operation, operation_regions, &parent_inputs)?;
        return outputs
            .into_iter()
            .map(|tracer| {
                let physical_type = tracer.r#type().into_owned();
                ArrayBatch::new(physical_type, tracer, BatchAxis::replicated())
            })
            .collect();
    }
    let axis_size = context.axis_size();
    let aligned_inputs = inputs
        .iter()
        .map(|input| match input.batch_axis_position() {
            Some(_) => input.move_axis(0),
            None => input.broadcast(0, axis_size, context.axis_sharding().clone()),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (operation, operation_regions) = make_operation(Some(axis_size))?;
    let parent_inputs = aligned_inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
    let outputs = context.parent().bind(operation, operation_regions, &parent_inputs)?;
    outputs
        .into_iter()
        .map(|tracer| {
            let physical_type = tracer.r#type().into_owned();
            ArrayBatch::new(physical_type, tracer, Some(0))
        })
        .collect()
}

/// Batches the region at `index` using the custom-derivative rewrapping convention: every input and output is mapped
/// at axis `0`.
pub(crate) fn batch_rewrapped_program<C: Context<Type = ArrayType>, D: BatchingDriver<C>>(
    context: &BatchingContext<C>,
    driver: &D,
    index: usize,
) -> Result<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>, BatchingError> {
    let region = driver.region(index)?;
    let input_count = region.input_types().len();
    let input_batch_axes = vec![BatchAxis::new(0); input_count];
    let (program, _) = driver.batch_program(
        context,
        region,
        input_batch_axes.as_slice(),
        ProgramBatchingOutputAxesPolicy::AlignAllTo(Axis::from(0)),
    )?;
    Ok(program)
}

/// Batching rule for [`CustomJvpOperation`]: re-wraps the call around batched primal/JVP programs so the custom
/// derivative survives `batch`; see `stage_rewrapped_custom_call`.
impl<C, O> BatchableOperation<C> for CustomJvpOperation
where
    C: Context<Type = ArrayType, Operation = O>,
    <C as Domain>::Value: Broadcast + Transpose,
    O: Operation<ArrayType> + From<TransposeOperation> + From<BroadcastOperation> + From<CustomJvpOperation>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError> {
        stage_rewrapped_custom_call(context, inputs, |batched| match batched {
            None => Ok((O::from(*self), driver.regions().map(|region| region.to_program()).collect())),
            Some(_) => Ok((
                O::from(CustomJvpOperation::new()),
                vec![batch_rewrapped_program(context, driver, 0)?, batch_rewrapped_program(context, driver, 1)?],
            )),
        })
    }
}

/// Higher-order operation pairing a primal program with user-supplied forward/backward (VJP) programs — the direct
/// analogue of JAX's [`custom_vjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html).
///
/// The forward program maps the primal inputs to the primal outputs followed by arbitrarily many residual values;
/// the backward program maps those residuals followed by one cotangent per primal output to one cotangent per primal
/// input. The primal program is kept separate from the forward program so that un-differentiated calls do not pay
/// for residual computation: interpretation and backend lowering replay the lean primal program, and the
/// forward program runs only under reverse-mode differentiation. Linearization evaluates the
/// forward program, captures its residuals as factors, and stages one opaque linear call whose transpose replays the
/// backward program — so reverse mode uses exactly the user-supplied gradient. Forward-mode differentiation
/// (interpreting the staged linear call) is rejected, matching JAX's `custom_vjp` semantics.
///
/// Batching (`batch`) re-wraps the call around batched primal/forward/backward programs — mirroring JAX's
/// `custom_vjp_call_jaxpr` batching rule — so the custom derivative survives a `batch` applied *before*
/// differentiation, under eager and staging parents alike.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CustomVjpOperation;

impl CustomVjpOperation {
    /// Creates a custom-VJP operation. The primal, forward, and backward [`Program`]s are supplied separately as
    /// the operation's attached regions (via the region driver passed to [`Context::bind`]) in
    /// the region order `["primal", "forward", "backward"]`; [`Operation::infer_output_types`] validates that the
    /// forward interface consumes the primal inputs and produces the primal outputs followed by the residuals, and
    /// that the backward interface consumes those residuals followed by one cotangent per primal output and
    /// produces one cotangent per primal input.
    #[inline]
    pub fn new() -> Self {
        Self
    }
}

impl Default for CustomVjpOperation {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Display for CustomVjpOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("custom_vjp")
    }
}

/// Validates the custom-VJP contract over the three attached region interfaces
/// (`["primal", "forward", "backward"]` region order) and returns the primal interface; refer to the documentation of
/// [`CustomVjpOperation::new`] for the contract.
fn validated_custom_vjp_interfaces<T: DifferentiableType>(
    region_interfaces: &[RegionInterface<T>],
) -> Result<&RegionInterface<T>, TypeError> {
    if region_interfaces.len() != 3 {
        return Err(TypeError::invalid(format!(
            "custom_vjp expects 3 attached regions but got {}",
            region_interfaces.len()
        )));
    }
    let primal_interface = &region_interfaces[0];
    let forward_interface = &region_interfaces[1];
    let backward_interface = &region_interfaces[2];
    let input_types = primal_interface.input_types();
    let output_types = primal_interface.output_types();
    check_types!(@same, "custom_vjp forward input", [input_types, forward_interface.input_types()]);
    let forward_output_types = forward_interface.output_types();
    if forward_output_types.len() < output_types.len() {
        return Err(TypeError::invalid(format!(
            "custom_vjp forward must produce at least the {} primal output(s) but produced {} value(s)",
            output_types.len(),
            forward_output_types.len(),
        )));
    }
    check_types!(@same, "custom_vjp forward output", [
        output_types,
        &forward_output_types[..output_types.len()],
    ]);
    let residual_types = &forward_output_types[output_types.len()..];
    let output_cotangent_types = output_types.iter().map(|r#type| r#type.cotangent());
    let expected_backward_input_types: Vec<T> = residual_types.iter().cloned().chain(output_cotangent_types).collect();
    check_types!(@same, "custom_vjp backward input", [
        &expected_backward_input_types,
        backward_interface.input_types(),
    ]);
    let expected_backward_output_types = input_types.iter().map(|r#type| r#type.cotangent()).collect::<Vec<_>>();
    check_types!(@same, "custom_vjp backward output", [
        &expected_backward_output_types,
        backward_interface.output_types(),
    ]);
    Ok(primal_interface)
}

impl<T: DifferentiableType> Operation<T> for CustomVjpOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "custom_vjp"
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        let primal_interface = validated_custom_vjp_interfaces(region_interfaces)?;
        check_types!(@same, "custom_vjp input", [primal_interface.input_types(), input_types]);
        Ok(primal_interface.output_types().to_vec())
    }

    #[inline]
    fn region_names(&self) -> &'static [&'static str] {
        &["primal", "forward", "backward"]
    }
}

impl<C: Domain<Type: DifferentiableType>> InterpretableOperation<C> for CustomVjpOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        driver.interpret_region(context, 0, inputs.to_vec())
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of [`Program::partially_evaluate`] for a
/// [`CustomVjpOperation`]: a call with all-known operands folds by interpreting its primal, and otherwise
/// residualizes unchanged.
impl<C: Context<Type: DifferentiableType>> PartiallyEvaluatableOperation<C> for CustomVjpOperation where
    C::Operation: From<CustomVjpOperation>
{
}

/// Batching rule for [`CustomVjpOperation`]: re-wraps the call around batched primal/forward/backward programs so
/// the custom derivative survives `batch`; see `stage_rewrapped_custom_call`.
impl<C, O> BatchableOperation<C> for CustomVjpOperation
where
    C: Context<Type = ArrayType, Operation = O>,
    <C as Domain>::Value: Broadcast + Transpose,
    O: Operation<ArrayType> + From<TransposeOperation> + From<BroadcastOperation> + From<CustomVjpOperation>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError> {
        stage_rewrapped_custom_call(context, inputs, |batched| match batched {
            None => Ok((O::from(*self), driver.regions().map(|region| region.to_program()).collect())),
            Some(_) => Ok((
                O::from(CustomVjpOperation::new()),
                vec![
                    batch_rewrapped_program(context, driver, 0)?,
                    batch_rewrapped_program(context, driver, 1)?,
                    batch_rewrapped_program(context, driver, 2)?,
                ],
            )),
        })
    }
}

/// Access to a custom-VJP residual payload as a concrete value during pullback interpretation.
///
/// Implemented by plain values as the identity.
#[doc(hidden)]
pub trait CustomVjpResidual<V: Value>: Value<Type = V::Type> {
    /// Returns the concrete residual value.
    fn residual_value(&self) -> Result<V, ProgramError>;
}

impl<V: Value> CustomVjpResidual<V> for V {
    #[inline]
    fn residual_value(&self) -> Result<V, ProgramError> {
        Ok(self.clone())
    }
}

/// Capture-free forward-mode (JVP) rule for [`CustomVjpOperation`]: replays the user-supplied forward program through
/// the active context and stages one opaque [`CustomVjpTangentOperation`] carrier for the output tangents.
///
/// Unlike [`CustomJvpOperation`], a `custom_vjp` function has no forward tangent program, so the forward cannot
/// compute the output tangents straight-line. Instead it reproduces — under the capture-free direct-transpose path —
/// the same structure the capture-based reverse rule builds: the forward program (already an ordinary primal-enum
/// program mapping `inputs -> (outputs..., residuals...)`) is replayed through
/// [`Program::interpret_in_context`](crate::Program::interpret_in_context) over the dual
/// primals, recovering the primal outputs and the residuals; then one [`CustomVjpTangentOperation`] is staged over
/// `[input_tangents..., residuals...]` with the residuals as ordinary *operands* (not capture factors). That carrier
/// is opaque: it stands for the unknown tangent map and rejects interpretation, so a forward-mode use through it
/// fails with the canonical reverse-only error, while [`transpose_primal_custom_vjp`] replays the user's `backward`
/// program to produce the input cotangents. Because the residuals flow as operand edges and the carrier is a leaf
/// primal-enum operation, the rule introduces no symbolic capture and needs no
/// nested differentiation or linearization request.
impl<C: Context + Zero<C::Value>> DifferentiableOperation<C> for CustomVjpOperation
where
    C::Type: DifferentiableType,
    C::Operation: From<CustomVjpTangentOperation<C::Type>>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The attached regions are `["primal", "forward", "backward"]`; the primal interface provides the boundary
        // types.
        let primal_region = driver.region(0)?;
        let forward_region = driver.region(1)?;
        let backward_region = driver.region(2)?;
        let output_count = primal_region.output_types().len();
        check_count!("input", inputs, primal_region.input_types().len(), ProgramError);
        // Replay the forward region on the dual primals, recovering the primal outputs followed by the residuals.
        let primal_operands = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let mut forward_outputs = forward_region.interpret_in_context(context, primal_operands)?;
        if forward_outputs.len() < output_count {
            return Err(ProgramError::MalformedProgram(format!(
                "custom_vjp forward region produced {} outputs which is fewer than its {output_count} primal \
                 output(s)",
                forward_outputs.len(),
            ))
            .into());
        }
        let residuals = forward_outputs.split_off(output_count);
        let primal_outputs = forward_outputs;
        let residual_count = residuals.len();

        let input_tangent_types = inputs.iter().map(|input| input.primal().r#type().tangent()).collect::<Vec<_>>();
        let output_tangent_types = primal_outputs.iter().map(|output| output.r#type().tangent()).collect::<Vec<_>>();

        // Stage one opaque carrier over `[input_tangents..., residuals...]`, producing the output tangents. The
        // carrier rejects forward interpretation and transposes by replaying the user's backward region.
        // The opaque carrier takes every input tangent as a real operand, so materialize structural zeros.
        let mut carrier_operands = inputs
            .iter()
            .map(|input| input.tangent().clone().materialize(context))
            .collect::<Result<Vec<_>, _>>()?;
        carrier_operands.extend(residuals);
        let carrier = CustomVjpTangentOperation::new(residual_count, false, input_tangent_types, output_tangent_types);
        let output_tangents = context.bind(carrier, vec![backward_region.to_program()], &carrier_operands)?;
        check_count!("output", output_tangents, output_count, ProgramError);

        Ok(primal_outputs
            .into_iter()
            .zip(output_tangents)
            .map(|(primal, tangent)| DifferentiationDual::new(primal, tangent))
            .collect::<Result<Vec<_>, _>>()?)
    }
}

crate::impl_non_transposable_operation!(CustomVjpOperation);

/// Opaque primal-enum carrier staged by [`CustomVjpOperation`]'s capture-free forward-mode rule.
///
/// The tangent program stays in the primal operation family `O` and carries its residuals as ordinary program
/// operands: the carrier receives them as the trailing operands of the staged operation, after the input tangents.
/// (The deleted capture-based `CustomVjpCallOperation` reverse path instead closed residuals into captured factors;
/// the operand form is the sole surviving mechanism.)
///
/// In its un-transposed form it stands for the (unknown) tangent map of the custom function and rejects
/// interpretation: `custom_vjp` functions are reverse-mode-only. Transposition (see [`transpose_primal_custom_vjp`])
/// reads the residual operands from the pullback and replays the user's backward program on them and the incoming
/// output cotangents, producing the input cotangents — so reverse mode uses exactly the user-supplied gradient.
/// The carrier stores its input and output tangent types because they can differ from both the primal types and
/// the backward region's cotangent types.
#[derive(Clone, Debug, PartialEq)]
pub struct CustomVjpTangentOperation<T: Type> {
    /// Number of residual operands, used to split the backward program's inputs into the residual prefix and the
    /// output-cotangent suffix.
    residual_count: usize,

    /// Whether this carrier has been transposed into its executable (pullback) form.
    transposed: bool,

    /// Expected input tangent types of the un-transposed carrier, in primal-input order.
    input_tangent_types: Vec<T>,

    /// Output tangent types produced by the un-transposed carrier, in primal-output order.
    output_tangent_types: Vec<T>,
}

impl<T: Type> CustomVjpTangentOperation<T> {
    /// Creates a custom-VJP tangent carrier. Use `transposed = false` for the opaque pushforward form and
    /// `transposed = true` for the transposed pullback form. The user's backward program (mapping
    /// `(residuals..., output_cotangents...)` to input cotangents) is supplied separately as the operation's
    /// single attached `backward` region.
    ///
    /// # Parameters
    ///
    ///   - `residual_count`: Number of trailing residual operands carried alongside the tangents.
    ///   - `transposed`: Whether this carrier is in its transposed (pullback) form.
    ///   - `input_tangent_types`: Types of the un-transposed carrier's input tangents.
    ///   - `output_tangent_types`: Types of the un-transposed carrier's output tangents.
    pub fn new(
        residual_count: usize,
        transposed: bool,
        input_tangent_types: Vec<T>,
        output_tangent_types: Vec<T>,
    ) -> Self {
        Self { residual_count, transposed, input_tangent_types, output_tangent_types }
    }

    /// Returns the number of trailing residual operands.
    #[inline]
    pub fn residual_count(&self) -> usize {
        self.residual_count
    }

    /// Returns whether this carrier is in its transposed (pullback) form.
    #[inline]
    pub fn transposed(&self) -> bool {
        self.transposed
    }

    /// Splits the backward interface's input types into the leading residual types and the trailing cotangent
    /// types (one per primal output).
    fn split_backward_inputs(&self, backward_interface: &RegionInterface<T>) -> Result<(Vec<T>, Vec<T>), TypeError> {
        if self.residual_count > backward_interface.input_types().len() {
            return Err(TypeError::invalid(format!(
                "custom_vjp residual count {} exceeds backward region input count {}",
                self.residual_count,
                backward_interface.input_types().len(),
            )));
        }
        let mut residual_types = backward_interface.input_types().to_vec();
        let cotangent_types = residual_types.split_off(self.residual_count);
        Ok((residual_types, cotangent_types))
    }
}

impl<T: Type> Display for CustomVjpTangentOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.transposed {
            formatter.write_str("custom_vjp_backward")
        } else {
            formatter.write_str("custom_vjp_tangent")
        }
    }
}

impl<T: Type> Operation<T> for CustomVjpTangentOperation<T> {
    #[inline]
    fn name(&self) -> &'static str {
        if self.transposed { "custom_vjp_backward" } else { "custom_vjp_tangent" }
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        if region_interfaces.len() != 1 {
            return Err(TypeError::invalid(format!(
                "{} expects 1 attached region but got {}",
                self.name(),
                region_interfaces.len(),
            )));
        }
        let backward_interface = &region_interfaces[0];
        let (residual_types, cotangent_types) = self.split_backward_inputs(backward_interface)?;
        check_count!(
            "custom_vjp tangent input type",
            self.input_tangent_types,
            backward_interface.output_types().len(),
            TypeError,
        );
        check_count!("custom_vjp tangent output type", self.output_tangent_types, cotangent_types.len(), TypeError,);
        if self.transposed {
            // The transposed (pullback) carrier maps `[output_cotangents..., residuals...]` to the input cotangents
            // carried by the backward program's outputs.
            let expected: Vec<T> = cotangent_types.iter().chain(residual_types.iter()).cloned().collect();
            check_types!(@same, "custom_vjp backward", [&expected, input_types]);
            Ok(backward_interface.output_types().to_vec())
        } else {
            // The un-transposed (tangent-map) carrier maps `[input_tangents..., residuals...]` to the output tangents.
            // Those types are stored explicitly because they need not match the backward program's cotangent
            // boundary.
            let expected: Vec<T> = self.input_tangent_types.iter().chain(residual_types.iter()).cloned().collect();
            check_types!(@same, "custom_vjp tangent", [&expected, input_types]);
            Ok(self.output_tangent_types.clone())
        }
    }

    #[inline]
    fn region_names(&self) -> &'static [&'static str] {
        &["backward"]
    }
}

/// Interprets a [`CustomVjpTangentOperation`] by rejecting forward-mode interpretation in both forms.
///
/// The un-transposed carrier is the opaque tangent map of a reverse-mode-only `custom_vjp`, so interpreting it is the
/// operation forward mode would need and is rejected. The transposed carrier is never interpreted as a staged
/// operation either: [`transpose_primal_custom_vjp`] replays the user's backward program directly through the
/// pullback builder rather than leaving a transposed carrier in the program, so reaching this interpret path means a
/// forward-mode use slipped through and the same reverse-only error applies.
impl<C: Domain> InterpretableOperation<C> for CustomVjpTangentOperation<C::Type> {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        _inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        Err(TypeError::invalid(
            "custom_vjp does not support forward-mode differentiation; use reverse mode (vjp, \
                value_and_gradient, or jacobian_reverse) instead"
                .to_string(),
        )
        .into())
    }
}

/// Batching rule for [`CustomVjpTangentOperation`]: the opaque custom-VJP tangent carrier is a forward-mode tangent
/// map that never appears in a batched primal program — reverse mode consumes it during transposition rather than
/// batching it — so batching is rejected for every context.
impl<C> BatchableOperation<C> for CustomVjpTangentOperation<ArrayType>
where
    C: Context<Type = ArrayType>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        _context: &BatchingContext<C>,
        _driver: &D,
        _inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        Err(BatchingError::UnsupportedOperation { message: format!("operation `{}` cannot be batched", self.name()) })
    }
}

/// Forward-mode (JVP) rule for [`CustomVjpTangentOperation`]: a `custom_vjp` function is reverse-mode only and has no
/// forward tangent program, so its tangent carrier rejects forward-mode linearization. The carrier never reaches this
/// rule on the supported path — the [`CustomVjpOperation`] JVP rule stages it on the tangent side — but the rule is
/// implemented so the enum dispatch can forward to it uniformly.
impl<C: Context> DifferentiableOperation<C> for CustomVjpTangentOperation<C::Type> {
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        _inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!("{} has no forward-mode (jvp) rule; custom_vjp is reverse-mode only", self.name(),),
        }
        .into())
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of [`Program::partially_evaluate`] for a
/// [`CustomVjpTangentOperation`]; the opaque carrier never has all-known operands in practice (its tangent operands
/// are unknown by construction), so it residualizes unchanged.
impl<C: Context> PartiallyEvaluatableOperation<C> for CustomVjpTangentOperation<C::Type> where
    C::Operation: From<CustomVjpTangentOperation<C::Type>>
{
}

/// Partition-aware transpose rule for an opaque [`CustomVjpTangentOperation`], used by the direct reverse path
/// when it transposes a tangent program in the primal operation family `O` rather than re-keying it into the linear
/// family. The residuals are ordinary *operands* (known values supplied through `operand_values`) rather than the
/// deleted capture-based path's capture factors, so the rule reads them from the pullback and replays the user's
/// backward program forward into the pullback builder.
///
/// The forward stages the carrier over `[input_tangents..., residuals...]`, with one tangent operand per primal input.
/// Partial evaluation may prove some tangent operands known, while the residuals are always known. This rule
/// therefore:
///
///   1. Splits the operands at the carrier's declared tangent count, retaining the linearity of each tangent operand
///      and reading the trailing residual values from the pullback.
///   2. Stages the output cotangents (materializing structural zeros so the backward program receives every
///      cotangent input).
///   3. Replays the user's `backward` program over `[residuals..., output_cotangents...]` through
///      [`Program::interpret_in_context`](crate::Program::interpret_in_context), producing the input cotangents. The backward program is
///      *not* transposed — it already
///      is the pullback — so it is replayed forward into the active pullback builder.
///
/// The returned cotangents place each backward result at its corresponding unknown tangent operand and a structural
/// [`MaybeZero::Zero`] at known tangent and residual positions. Because the backward program is replayed in the same
/// operation family `O` through the context's [`bind`](Context::bind), the rule is value-level and introduces no
/// recursive transposition obligation on `O`.
///
/// # Parameters
///
///   - `operation`: Opaque custom-VJP tangent carrier staged into the tangent program.
///   - `context`: Active transpose tracing context the pullback is staged into.
///   - `inputs`: Per-operand [`PartialValue`] knowledge. The fixed leading tangent prefix may contain both
///     [`Unknown`](PartialValue::Unknown) and [`Known`](PartialValue::Known) entries; the trailing known entries carry
///     the residual tracers the pullback reads.
///   - `outputs`: Symbolic cotangents for the carrier's outputs (one per primal output).
pub fn transpose_primal_custom_vjp<V, O, D: TranspositionDriver<V, O>>(
    operation: &CustomVjpTangentOperation<V::Type>,
    context: &mut TracingContext<V, O>,
    driver: &D,
    inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError>
where
    V: Value<Type: DifferentiableType>,
    O: Operation<V::Type> + From<ZeroOperation<V::Type>>,
{
    if operation.transposed {
        return Err(TypeError::invalid(
            "transposing a custom_vjp pullback (second-order reverse mode through custom_vjp) is not \
                supported"
                .to_string(),
        )
        .into());
    }

    // Operand layout is `[input_tangents..., residuals...]`. The tangent prefix has one operand per primal input,
    // including tangent operands that partial evaluation proved known. The residuals are the trailing known operands
    // read from the pullback. The dispatch guarantees a `Known` operand carries its pullback value, so each residual
    // tracer is read directly.
    let backward = driver.region(0)?;
    let operand_linear = inputs.iter().map(PartialValue::is_unknown).collect::<Vec<_>>();
    let tangent_count = operation.input_tangent_types.len();
    let residual_count = operation.residual_count;
    let backward_input_count = backward.input_ids().len();
    let cotangent_count = backward_input_count.checked_sub(residual_count).ok_or_else(|| {
        ProgramError::MalformedProgram(format!(
            "custom_vjp residual count {residual_count} exceeds backward region input count {backward_input_count}",
        ))
    })?;
    let expected_input_count = tangent_count.checked_add(residual_count).ok_or_else(|| {
        ProgramError::MalformedProgram("custom_vjp tangent input count exceeds usize capacity".to_string())
    })?;
    check_count!("input", operand_linear, expected_input_count, ProgramError);
    let residuals = (tangent_count..inputs.len())
        .map(|index| {
            inputs[index]
                .as_known()
                .ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "custom_vjp tangent transpose operand {index} has no known residual value"
                    ))
                })
                .cloned()
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;

    // Stage the output cotangents, then replay the user's backward program on `[residuals..., outputs...]`.
    // The backward program is the pullback already, so it is replayed forward into the pullback builder rather than
    // transposed.
    check_count!("output", outputs, cotangent_count, ProgramError);
    let mut backward_inputs = residuals;
    for cotangent in outputs {
        backward_inputs.push(cotangent.clone().materialize(context)?);
    }
    let input_cotangents = backward.interpret_in_context(context, backward_inputs)?;
    check_count!("output", input_cotangents, tangent_count, ProgramError);

    // The user's backward program is an opaque replay: its outputs come back as plain values, so any
    // structural zero-ness a user backward expresses (a `zero`/`zero_like` output for a non-differentiated input)
    // would otherwise be lost at this boundary and stage wasted adjoint work upstream. Recover it here with one local
    // pass over the *backward program* itself: an output produced by a canonical `zero` or `zero_like` instruction is
    // a structural zero. This is the reverse-mode analogue of JAX's `custom_vjp` symbolic zeros, recovered
    // automatically instead of through an opt-in.
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

    // Reassemble one cotangent per operand: unknown tangent operands receive their corresponding backward-program
    // output, known tangent operands and residuals carry structural zeros, and canonical zero outputs remain structural
    // zeros.
    let input_cotangents = input_cotangents.into_iter().zip(output_is_zero).map(|(cotangent, is_zero)| {
        if is_zero { MaybeZero::Zero(cotangent.r#type().into_owned()) } else { MaybeZero::Value(cotangent) }
    });
    let mut cotangents = operand_linear[..tangent_count]
        .iter()
        .zip(&inputs[..tangent_count])
        .zip(input_cotangents)
        .map(
            |((&linear, input), cotangent)| {
                if linear { cotangent } else { MaybeZero::Zero(input.r#type().cotangent()) }
            },
        )
        .collect::<Vec<_>>();
    cotangents.extend(inputs[tangent_count..].iter().map(|input| MaybeZero::Zero(input.r#type().cotangent())));
    Ok(cotangents)
}

/// Transpose rule for the opaque [`CustomVjpTangentOperation`] carrier, forwarding to
/// [`transpose_primal_custom_vjp`]. The recursion stays value-level (the user's backward program is replayed forward
/// into the pullback builder through [`Context::bind`]), so instantiating this implementation for a closed operation
/// enum introduces no recursive [`TransposableOperation`] obligation on `O`.
impl<V, O> TransposableOperation<V, O> for CustomVjpTangentOperation<V::Type>
where
    V: Value<Type: DifferentiableType>,
    O: Operation<V::Type> + From<ZeroOperation<V::Type>>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        transpose_primal_custom_vjp(self, context, driver, inputs, outputs).map_err(DifferentiationError::from)
    }
}

/// Function with a user-supplied JVP rule — the ergonomic analogue of JAX's
/// [`jax.custom_jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html) /
/// [`defjvp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.defjvp.html) decorator pair, built by
/// [`custom_jvp`].
///
/// # When to use
///
/// Reach for a custom JVP when the function *is* forward-differentiable but its automatically derived tangent is
/// numerically unstable or wasteful and you want to supply a stable, efficient one by hand — classic cases are a
/// `log`-`sum`-`exp`, a softmax, or a normalization, where a hand-written tangent avoids the cancellation or
/// redundant work the generic rule incurs. A single custom JVP serves **both** differentiation modes: reverse mode
/// obtains its gradient by transposing the supplied tangent map, so the one rule composes with forward mode, reverse
/// mode, and their higher-order combinations. Prefer it over [`CustomVjp`] whenever the function is naturally
/// forward-differentiable, and reach for [`CustomVjp`] only when just the reverse rule is natural (for example
/// implicit differentiation or adjoint solvers).
///
/// The primal function and its JVP rule are stored as plain closures over [`Parameterized`] trees of
/// [`DomainTracer`]s — `ryft`'s analogue of JAX pytrees — so inputs and outputs can be single tracers, tuples, or
/// any other parameterized structure. Nothing is traced at construction time: each [`call`](Self::call) reads the
/// input types off its tracer arguments, traces both closures into programs specialized to those types, validates
/// the rule signature, and stages one [`CustomJvpOperation`] into the caller's staging context — mirroring how JAX
/// traces rule functions into jaxprs lazily at transform time. `primal` maps the input tree to the output tree, and
/// `jvp` maps `(inputs, input_tangents)` to `(outputs, output_tangents)`, exactly like a JAX `defjvp` rule.
///
/// The primal closure is kept separate from the JVP closure for efficiency rather than necessity: the JVP rule
/// computes both the outputs and their tangents, so deriving the primal from it would make every un-differentiated
/// call pay for tangent computation. Interpretation, batching, and backend lowering replay the lean primal program;
/// the JVP program runs only under differentiation.
///
/// There is no analogue of JAX's `nondiff_argnums` because closure capture subsumes it: static, non-differentiated
/// configuration is simply captured by the closures (all of them can see it), exactly like JAX threads
/// non-differentiated arguments through to the rule functions.
pub struct CustomJvp<D: Domain, P, J, IT, OT> {
    /// Closure computing the primal output tree from the primal input tree.
    primal: P,

    /// Closure computing `(outputs, output_tangents)` from `(inputs, input_tangents)`.
    jvp: J,

    /// Phantom marker pinning the [`Domain`] and the input and output tracer-tree types named by the closure
    /// signatures. The domain is a pure type witness, so no domain value is stored.
    marker: PhantomData<fn() -> (D, IT, OT)>,
}

/// Creates a [`CustomJvp`] function from a primal closure and a JVP-rule closure over trees of the [`Domain`] `D`'s
/// tracers. Refer to the documentation of [`CustomJvp`] for the calling convention and tracing semantics.
pub fn custom_jvp<D, P, J, IT, OT>(primal: P, jvp: J) -> CustomJvp<D, P, J, IT, OT>
where
    D: Domain<Type: DifferentiableType>,
    P: Fn(IT) -> Result<OT, ProgramError>,
    J: Fn(IT, IT) -> Result<(OT, OT), ProgramError>,
{
    CustomJvp { primal, jvp, marker: PhantomData }
}

impl<D, P, J, IT, OT> CustomJvp<D, P, J, IT, OT>
where
    D: Domain<Type: DifferentiableType>,
    P: Fn(IT) -> Result<OT, ProgramError>,
    J: Fn(IT, IT) -> Result<(OT, OT), ProgramError>,
    D::Operation: From<CustomJvpOperation>,
    IT: Parameterized<DomainTracer<D>>,
    IT::Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>,
    OT: Parameterized<DomainTracer<D>>,
    OT::Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>,
    IT::To<D::Type>: Clone
        + Parameterized<D::Type, Family = IT::Family, To<DomainTracer<D>> = IT, To<D::Constant> = IT::To<D::Constant>>,
    OT::To<D::Type>:
        Parameterized<D::Type, Family = OT::Family, To<DomainTracer<D>> = OT, To<D::Constant> = OT::To<D::Constant>>,
{
    /// Stages this custom-JVP function on the provided tracer input tree and returns its output tree, tracing the
    /// stored closures into programs specialized to the input types. Differentiation of the staged call replays the
    /// JVP rule instead of differentiating the primal body, in both forward and reverse mode.
    pub fn call<V, ICT>(&self, input: ICT) -> Result<<OT::To<D::Type> as Parameterized<D::Type>>::To<V>, ProgramError>
    where
        V: Value<Type = D::Type>,
        V::DispatchDomain: Context<Type = D::Type, Constant = D::Constant, Operation = D::Operation>,
        IT::Family: ParameterizedFamily<V>,
        OT::Family: ParameterizedFamily<V>,
        ICT: Parameterized<V, Family = IT::Family, To<D::Type> = IT::To<D::Type>>,
        <OT::To<D::Type> as Parameterized<D::Type>>::To<V>: Parameterized<
                V,
                Family = OT::Family,
                ParameterStructure = <OT::To<D::Type> as Parameterized<D::Type>>::ParameterStructure,
            >,
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
            return Err(TypeError::invalid("custom_jvp requires at least one input".to_string()).into());
        };
        let (_, primal) = D::trace(|xs| (self.primal)(xs), input_types.clone())?;
        let input_tangent_types =
            input_types.clone().map_parameters(|r#type| r#type.tangent()).map_err(ProgramError::from)?;
        let (output_types, jvp) = D::trace(|(x, t)| (self.jvp)(x, t), (input_types, input_tangent_types))?;
        let operation = D::Operation::from(CustomJvpOperation::new());
        // The call binds through whatever context the input values flow (a staged trace, a batching context, or a
        // JVP context), so `custom_jvp` composes under `vmap`/`jvp` — the batch/JVP rule of the bound operation fires.
        let context = first.dispatch_domain();
        let outputs = context.bind(operation, vec![primal.to_flat_program(), jvp.to_flat_program()], &input_values)?;
        let output_structure = output_types.0.parameter_structure();
        Ok(Parameterized::from_parameters(output_structure, outputs)?)
    }
}

/// Function with user-supplied forward/backward (VJP) rules — the ergonomic analogue of JAX's
/// [`jax.custom_vjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html) /
/// [`defvjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.defvjp.html) decorator pair, built by
/// [`custom_vjp`].
///
/// # When to use
///
/// Reach for a custom VJP when only the *reverse* rule is natural, or when the function is not (efficiently)
/// forward-differentiable. Common cases:
///
///   - **Implicit differentiation** — differentiate through a solver, optimizer, or fixed point via the implicit
///     function theorem rather than unrolling its iterations.
///   - **Adjoint methods** — backpropagate through an ODE or PDE solve via the adjoint system instead of
///     differentiating the integrator's individual steps.
///   - **External or black-box calls** — supply the reverse rule for a custom kernel or a computation that does not
///     itself trace into `ryft` programs.
///   - **Numerical stability** — replace an unstable or wasteful automatically derived gradient with a hand-written
///     one.
///
/// A custom VJP is reverse-mode only: forward-mode differentiation of a staged call is rejected, and the current
/// transpose implementation also rejects transposing its generated pullback, so higher-order derivatives through a
/// custom VJP are not yet supported. When the function is forward-differentiable or must participate in higher-order
/// differentiation, use [`CustomJvp`] instead.
///
/// The primal function and its forward/backward rules are stored as plain closures over [`Parameterized`] trees of
/// [`DomainTracer`]s — `ryft`'s analogue of JAX pytrees — so inputs, outputs, and residuals can be single tracers,
/// tuples, or any other parameterized structure. Nothing is traced at construction time: each [`call`](Self::call)
/// reads the input types off its tracer arguments, traces the closures into programs specialized to those types,
/// validates the rule signatures, and stages one [`CustomVjpOperation`] into the caller's staging context —
/// mirroring how JAX traces rule functions into jaxprs lazily at transform time. `primal` maps the input tree to
/// the output tree, `forward` maps the input tree to `(outputs, residuals)` (the same structural split as a JAX
/// `f_fwd`), and `backward` maps `(residuals, output_cotangents)` to the input cotangent tree. As in JAX, the
/// resulting function supports reverse mode only; forward-mode differentiation of a staged call is rejected.
///
/// The primal closure is kept separate from the forward closure for efficiency rather than necessity: an
/// un-differentiated call should not pay for residual computation. Interpretation, batching, and backend lowering
/// replay the lean primal program; the residual-producing forward program runs only under reverse-mode
/// differentiation. Callers that do not care about the distinction can pass the same body for both — accepting that
/// the residual outputs are dead code outside of differentiation — which mirrors the common JAX idiom of writing
/// `f_fwd` as `return f(x), residuals`.
///
/// There is no analogue of JAX's `nondiff_argnums` because closure capture subsumes it: static, non-differentiated
/// configuration is simply captured by the closures (all of them can see it), exactly like JAX threads
/// non-differentiated arguments through to the rule functions.
pub struct CustomVjp<D: Domain, P, F, B, IT, OT, RT> {
    /// Closure computing the primal output tree from the primal input tree.
    primal: P,

    /// Closure computing `(outputs, residuals)` from the primal input tree.
    forward: F,

    /// Closure computing the input cotangent tree from `(residuals, output_cotangents)`.
    backward: B,

    /// Phantom marker pinning the [`Domain`] and the input, output, and residual tracer-tree types named by the
    /// closure signatures. The domain is a pure type witness, so no domain value is stored.
    marker: PhantomData<fn() -> (D, IT, OT, RT)>,
}

/// Creates a [`CustomVjp`] function from primal, forward, and backward closures over trees of the [`Domain`] `D`'s
/// tracers. Refer to the documentation of [`CustomVjp`] for the calling convention and tracing semantics.
pub fn custom_vjp<D, P, F, B, IT, OT, RT>(primal: P, forward: F, backward: B) -> CustomVjp<D, P, F, B, IT, OT, RT>
where
    D: Domain<Type: DifferentiableType>,
    P: Fn(IT) -> Result<OT, ProgramError>,
    F: Fn(IT) -> Result<(OT, RT), ProgramError>,
    B: Fn(RT, OT) -> Result<IT, ProgramError>,
{
    CustomVjp { primal, forward, backward, marker: PhantomData }
}

impl<D, P, F, B, IT, OT, RT> CustomVjp<D, P, F, B, IT, OT, RT>
where
    D: Domain<Type: DifferentiableType>,
    P: Fn(IT) -> Result<OT, ProgramError>,
    F: Fn(IT) -> Result<(OT, RT), ProgramError>,
    B: Fn(RT, OT) -> Result<IT, ProgramError>,
    D::Operation: From<CustomVjpOperation>,
    IT: Parameterized<DomainTracer<D>>,
    IT::Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>,
    OT: Parameterized<DomainTracer<D>>,
    OT::Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>,
    RT: Parameterized<DomainTracer<D>>,
    RT::Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>,
    IT::To<D::Type>: Clone
        + Parameterized<D::Type, Family = IT::Family, To<DomainTracer<D>> = IT, To<D::Constant> = IT::To<D::Constant>>,
    OT::To<D::Type>: Clone
        + Parameterized<D::Type, Family = OT::Family, To<DomainTracer<D>> = OT, To<D::Constant> = OT::To<D::Constant>>,
    RT::To<D::Type>:
        Parameterized<D::Type, Family = RT::Family, To<DomainTracer<D>> = RT, To<D::Constant> = RT::To<D::Constant>>,
{
    /// Stages this custom-VJP function on the provided tracer input tree and returns its output tree, tracing the
    /// stored closures into programs specialized to the input types. Reverse-mode differentiation of the staged
    /// call replays the backward rule on the forward rule's residuals instead of differentiating the primal body.
    pub fn call<V, ICT>(&self, input: ICT) -> Result<<OT::To<D::Type> as Parameterized<D::Type>>::To<V>, ProgramError>
    where
        V: Value<Type = D::Type>,
        V::DispatchDomain: Context<Type = D::Type, Constant = D::Constant, Operation = D::Operation>,
        IT::Family: ParameterizedFamily<V>,
        OT::Family: ParameterizedFamily<V>,
        ICT: Parameterized<V, Family = IT::Family, To<D::Type> = IT::To<D::Type>>,
        <OT::To<D::Type> as Parameterized<D::Type>>::To<V>: Parameterized<
                V,
                Family = OT::Family,
                ParameterStructure = <OT::To<D::Type> as Parameterized<D::Type>>::ParameterStructure,
            >,
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
            return Err(TypeError::invalid("custom_vjp requires at least one input".to_string()).into());
        };
        let (output_types, primal) = D::trace(|xs| (self.primal)(xs), input_types.clone())?;
        let (forward_output_types, forward) = D::trace(|xs| (self.forward)(xs), input_types.clone())?;
        let (_, residual_types) = forward_output_types;
        let output_cotangent_types =
            output_types.clone().map_parameters(|r#type| r#type.cotangent()).map_err(ProgramError::from)?;
        let (_, backward) = D::trace(
            |(residuals, cotangents)| (self.backward)(residuals, cotangents),
            (residual_types, output_cotangent_types),
        )?;
        let operation = D::Operation::from(CustomVjpOperation::new());
        // Bind through whatever context the inputs flow, so `custom_vjp` composes under `vmap`/`jvp`.
        let context = first.dispatch_domain();
        let outputs = context.bind(
            operation,
            vec![primal.to_flat_program(), forward.to_flat_program(), backward.to_flat_program()],
            &input_values,
        )?;
        let output_structure = output_types.parameter_structure();
        Ok(Parameterized::from_parameters(output_structure, outputs)?)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::batching::{Batch, BatchAxis};
    use crate::contexts::{Context, EagerContext, StagingContext};
    use crate::differentiation::{ForwardModeDifferentiate, ReverseModeDifferentiate};
    use crate::operations::constants::ZeroLikeOperation;
    use crate::operations::math::{Cos, CosOperation, MulOperation, Sin, SinOperation};
    use crate::operations::math::{Dot, DotDimensionNumbers};
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationOutput, PartialValue};
    use crate::programs::ProgramBuilder;
    use crate::programs::effects::Effects;
    use crate::programs::regions::{RegionDriver, RegionRef};
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding};
    use crate::types::{DataType, Shape, Size};

    use super::*;

    /// Returns the canonical test array type with the provided dimensions.
    fn test_type(dimensions: &[usize]) -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(dimensions.iter().map(|dimension| Size::Static(*dimension)).collect()))
    }

    /// Builds `f(x) = sin(x)` over one input of the provided type.
    fn sin_program(r#type: &ArrayType) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(r#type.clone());
        let output = builder.add_instruction(SinOperation, Vec::new(), vec![input]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds the deliberately wrong rule `jvp(x, dx) = (sin(x), 2 * cos(x) * dx)`, detectably different from the
    /// true derivative so tests can prove the custom rule is used.
    fn doubled_sin_jvp_program(r#type: &ArrayType) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(r#type.clone());
        let dx = builder.add_input(r#type.clone());
        let y = builder.add_instruction(SinOperation, Vec::new(), vec![x]).unwrap()[0];
        let cosine = builder.add_instruction(CosOperation, Vec::new(), vec![x]).unwrap()[0];
        let two = builder.add_constant(Array::scalar(2.0));
        let scaled = builder.add_instruction(MulOperation, Vec::new(), vec![two, cosine]).unwrap()[0];
        let tangent = builder.add_instruction(MulOperation, Vec::new(), vec![scaled, dx]).unwrap()[0];
        builder
            .build(vec![y, tangent], vec![Placeholder, Placeholder], vec![Placeholder, Placeholder])
            .unwrap()
    }

    /// Builds the forward rule `forward(x) = (sin(x), cos(x))`, with the cosine as the residual.
    fn sin_forward_program(r#type: &ArrayType) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(r#type.clone());
        let y = builder.add_instruction(SinOperation, Vec::new(), vec![x]).unwrap()[0];
        let residual = builder.add_instruction(CosOperation, Vec::new(), vec![x]).unwrap()[0];
        builder.build(vec![y, residual], vec![Placeholder], vec![Placeholder, Placeholder]).unwrap()
    }

    /// Builds the deliberately wrong rule `backward(residual, cotangent) = 3 * residual * cotangent`, detectably
    /// different from the true gradient so tests can prove the custom rule is used.
    fn tripled_sin_backward_program(
        r#type: &ArrayType,
    ) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::new();
        let residual = builder.add_input(r#type.clone());
        let cotangent = builder.add_input(r#type.clone());
        let three = builder.add_constant(Array::scalar(3.0));
        let scaled = builder.add_instruction(MulOperation, Vec::new(), vec![three, residual]).unwrap()[0];
        let gradient = builder.add_instruction(MulOperation, Vec::new(), vec![scaled, cotangent]).unwrap()[0];
        builder.build(vec![gradient], vec![Placeholder, Placeholder], vec![Placeholder]).unwrap()
    }

    fn custom_jvp_sin(
        r#type: &ArrayType,
    ) -> (ArrayOperation<Array>, Vec<Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>>) {
        (
            ArrayOperation::CustomJvp(CustomJvpOperation::new()),
            vec![sin_program(r#type), doubled_sin_jvp_program(r#type)],
        )
    }

    fn custom_vjp_sin(
        r#type: &ArrayType,
    ) -> (ArrayOperation<Array>, Vec<Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>>) {
        (
            ArrayOperation::CustomVjp(CustomVjpOperation::new()),
            vec![sin_program(r#type), sin_forward_program(r#type), tripled_sin_backward_program(r#type)],
        )
    }

    /// Returns the [`RegionInterface`] of the provided flat region program.
    fn custom_region_interface(
        program: &Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>,
    ) -> RegionInterface<ArrayType> {
        program.interface()
    }

    /// Test-only transposition driver exposing one backward region to direct custom-VJP transpose-rule tests.
    struct TestTranspositionDriver<'r> {
        /// Backward region exposed by this driver.
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
            _input_linearity: &[bool],
        ) -> Result<Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>, DifferentiationError> {
            Err(ProgramError::UnsupportedOperation {
                message: "test driver does not transpose nested regions".to_string(),
            }
            .into())
        }
    }

    #[test]
    fn test_custom_jvp_inference_validates_the_rule_signature() {
        let scalar = test_type(&[]);
        // The JVP interface must take `(inputs..., tangents...)`; a primal-only signature is rejected.
        assert!(
            CustomJvpOperation::new()
                .infer_output_types(
                    std::slice::from_ref(&scalar),
                    &[custom_region_interface(&sin_program(&scalar)), custom_region_interface(&sin_program(&scalar))],
                )
                .is_err()
        );
    }

    #[test]
    fn test_custom_vjp_inference_validates_the_rule_signatures() {
        let scalar = test_type(&[]);
        // The backward interface must consume `(residuals..., output cotangents...)`; a single-input program whose
        // signature cannot line up with the forward residuals is rejected.
        assert!(
            CustomVjpOperation::new()
                .infer_output_types(
                    std::slice::from_ref(&scalar),
                    &[
                        custom_region_interface(&sin_program(&scalar)),
                        custom_region_interface(&sin_forward_program(&scalar)),
                        custom_region_interface(&sin_program(&scalar)),
                    ],
                )
                .is_err()
        );
    }

    #[test]
    fn test_custom_derivative_inference_uses_differential_types() {
        let primal_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(Vec::new()));
        let tangent_type = ArrayType::new(DataType::F32, Shape::new(Vec::new()));
        let residual_type = test_type(&[]);
        let primal_interface =
            RegionInterface::new(vec![primal_type.clone()], vec![primal_type.clone()], Effects::PURE);
        let jvp_interface = RegionInterface::new(
            vec![primal_type.clone(), tangent_type.clone()],
            vec![primal_type.clone(), tangent_type.clone()],
            Effects::PURE,
        );
        assert_eq!(
            CustomJvpOperation::new()
                .infer_output_types(std::slice::from_ref(&primal_type), &[primal_interface.clone(), jvp_interface],),
            Ok(vec![primal_type.clone()]),
        );

        let forward_interface = RegionInterface::new(
            vec![primal_type.clone()],
            vec![primal_type.clone(), residual_type.clone()],
            Effects::PURE,
        );
        let backward_interface = RegionInterface::new(
            vec![residual_type.clone(), tangent_type.clone()],
            vec![tangent_type.clone()],
            Effects::PURE,
        );
        assert_eq!(
            CustomVjpOperation::new().infer_output_types(
                std::slice::from_ref(&primal_type),
                &[primal_interface, forward_interface, backward_interface.clone()],
            ),
            Ok(vec![primal_type]),
        );
        assert_eq!(
            CustomVjpTangentOperation::new(1, false, vec![tangent_type.clone()], vec![tangent_type.clone()],)
                .infer_output_types(&[tangent_type.clone(), residual_type], std::slice::from_ref(&backward_interface),),
            Ok(vec![tangent_type]),
        );
    }

    #[test]
    fn test_custom_derivative_calls_remain_opaque_to_partial_evaluation() {
        let scalar = test_type(&[]);
        for (operation, operation_regions) in [custom_jvp_sin(&scalar), custom_vjp_sin(&scalar)] {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let region_ids = operation_regions
                .iter()
                .map(|region| builder.import_region(region.entry_region_ref()))
                .collect::<Vec<_>>();
            let input = builder.add_input(scalar.clone());
            let output = builder.add_instruction(operation, region_ids, vec![input]).unwrap()[0];
            let program =
                builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

            let evaluation = program.partially_evaluate(&[PartialValue::Unknown(scalar.clone())]).unwrap();

            assert!(matches!(evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
            assert_eq!(evaluation.program.instructions().len(), 1);
            assert!(matches!(
                evaluation.program.instructions()[0].operation(),
                ArrayOperation::CustomJvp(_) | ArrayOperation::CustomVjp(_),
            ));
        }
    }

    #[test]
    fn test_custom_vjp_tangent_carrier_remains_opaque_to_partial_evaluation() {
        let scalar = test_type(&[]);
        let operation = ArrayOperation::CustomVjpTangent(CustomVjpTangentOperation::new(
            1,
            false,
            vec![scalar.clone()],
            vec![scalar.clone()],
        ));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let backward_region = builder.import_region(tripled_sin_backward_program(&scalar).entry_region_ref());
        let tangent = builder.add_input(scalar.clone());
        let residual = builder.add_input(scalar.clone());
        let output = builder.add_instruction(operation, vec![backward_region], vec![tangent, residual]).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let evaluation = program
            .partially_evaluate(&[PartialValue::Unknown(scalar.clone()), PartialValue::Unknown(scalar)])
            .unwrap();

        assert!(matches!(evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), ArrayOperation::CustomVjpTangent(_)));
    }

    #[test]
    fn test_custom_vjp_tangent_validates_residual_count() {
        let scalar = test_type(&[]);
        let backward_interface = RegionInterface::new(vec![scalar.clone()], vec![scalar.clone()], Effects::PURE);
        for transposed in [false, true] {
            assert!(matches!(
                CustomVjpTangentOperation::new(2, transposed, Vec::new(), Vec::new())
                    .infer_output_types(&[], std::slice::from_ref(&backward_interface)),
                Err(TypeError::Invalid { message })
                    if message == "custom_vjp residual count 2 exceeds backward region input count 1",
            ));
        }

        let backward = tripled_sin_backward_program(&scalar);
        let driver = TestTranspositionDriver { region: backward.entry_region_ref() };
        let mut context = TracingContext::<Array, ArrayOperation<Array>>::new();
        assert!(matches!(
            transpose_primal_custom_vjp(
                &CustomVjpTangentOperation::new(3, false, vec![scalar.clone()], vec![scalar.clone()]),
                &mut context,
                &driver,
                &[],
                &[],
            ),
            Err(ProgramError::MalformedProgram(message))
                if message == "custom_vjp residual count 3 exceeds backward region input count 2",
        ));
    }

    #[test]
    fn test_custom_vjp_transpose_preserves_known_tangent_operands_before_residuals() {
        let scalar = test_type(&[]);
        let mut backward_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let residual = backward_builder.add_input(scalar.clone());
        let output_cotangent = backward_builder.add_input(scalar.clone());
        let first_input_cotangent = backward_builder
            .add_instruction(MulOperation, Vec::new(), vec![residual, output_cotangent])
            .unwrap()[0];
        let backward = backward_builder
            .build::<Vec<Array>, Vec<Array>>(
                vec![first_input_cotangent, output_cotangent],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let driver = TestTranspositionDriver { region: backward.entry_region_ref() };
        let mut context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let known_tangent = context.input(scalar.clone());
        let residual = context.input(scalar.clone());
        let output_cotangent = context.input(scalar.clone());

        let contributions = transpose_primal_custom_vjp(
            &CustomVjpTangentOperation::new(1, false, vec![scalar.clone(), scalar.clone()], vec![scalar.clone()]),
            &mut context,
            &driver,
            &[PartialValue::Unknown(scalar), PartialValue::Known(known_tangent), PartialValue::Known(residual)],
            &[MaybeZero::Value(output_cotangent)],
        )
        .unwrap();

        assert_eq!(contributions.len(), 3);
        assert!(matches!(contributions[0], MaybeZero::Value(_)));
        assert!(contributions[1].is_zero());
        assert!(contributions[2].is_zero());
    }

    #[test]
    fn test_custom_vjp_transpose_preserves_structural_zero_outputs() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let primal_type = ArrayType::scalar(DataType::F64)
            .with_sharding(Sharding::new(mesh, Vec::new()).unwrap().with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let tangent_type = primal_type.tangent();
        let cotangent_type = primal_type.cotangent();
        assert_ne!(tangent_type, cotangent_type);

        // A canonical `zero` backward output is already typed in the primal input's cotangent space. Recovering its
        // structural zero must retain that type instead of dualizing its sharding a second time.
        let mut backward_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        backward_builder.add_input(cotangent_type.clone());
        let zero = backward_builder
            .add_instruction(ZeroOperation::new(cotangent_type.clone()), Vec::new(), Vec::new())
            .unwrap()[0];
        let backward = backward_builder
            .build::<Vec<Array>, Vec<Array>>(vec![zero], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let driver = TestTranspositionDriver { region: backward.entry_region_ref() };
        let mut context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = context.input(cotangent_type.clone());
        let contributions = transpose_primal_custom_vjp(
            &CustomVjpTangentOperation::new(0, false, vec![tangent_type.clone()], vec![tangent_type.clone()]),
            &mut context,
            &driver,
            &[PartialValue::Unknown(tangent_type.clone())],
            &[MaybeZero::Value(output_cotangent)],
        )
        .unwrap();
        assert!(matches!(&contributions[0], MaybeZero::Zero(r#type) if r#type == &cotangent_type));

        // `zero_like` is equally structural even though it consumes an exemplar input. Opaque backward replay must
        // recognize it instead of turning the result into a live zero-valued tracer.
        let mut backward_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = backward_builder.add_input(cotangent_type.clone());
        let zero_like =
            backward_builder.add_instruction(ZeroLikeOperation, Vec::new(), vec![output_cotangent]).unwrap()[0];
        let backward = backward_builder
            .build::<Vec<Array>, Vec<Array>>(vec![zero_like], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let driver = TestTranspositionDriver { region: backward.entry_region_ref() };
        let mut context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = context.input(cotangent_type.clone());
        let contributions = transpose_primal_custom_vjp(
            &CustomVjpTangentOperation::new(0, false, vec![tangent_type.clone()], vec![tangent_type]),
            &mut context,
            &driver,
            &[PartialValue::Unknown(primal_type.tangent())],
            &[MaybeZero::Value(output_cotangent)],
        )
        .unwrap();
        assert!(matches!(&contributions[0], MaybeZero::Zero(r#type) if r#type == &cotangent_type));
    }

    #[test]
    fn test_custom_jvp_interprets_the_primal_program() {
        let scalar = test_type(&[]);
        let (operation, operation_regions) = custom_jvp_sin(&scalar);
        let outputs = crate::EagerContext::<Array, ArrayOperation<Array>>::new()
            .bind(operation, operation_regions, &[Array::scalar(2.0)])
            .unwrap();
        assert_abs_diff_eq!(outputs[0].to_f64s()[0], 2.0f64.sin(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_jvp_governs_forward_mode() {
        let scalar = test_type(&[]);
        let (primal, tangent) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jvp(
                |x| {
                    let (operation, operation_regions) = custom_jvp_sin(&test_type(&[]));
                    Ok(x.context().bind(operation, operation_regions, &[x.clone()])?.into_iter().next().unwrap())
                },
                Array::scalar(2.0),
                Array::scalar(1.0),
            )
            .unwrap();
        let _ = scalar;
        assert_abs_diff_eq!(primal.to_f64s()[0], 2.0f64.sin(), epsilon = 1e-9);
        // The custom rule doubles the true derivative, proving it is in control.
        assert_abs_diff_eq!(tangent.to_f64s()[0], 2.0 * 2.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_jvp_governs_reverse_mode() {
        let (value, gradient) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .value_and_gradient(
                |x| {
                    let (operation, operation_regions) = custom_jvp_sin(&test_type(&[]));
                    x.context().bind(operation, operation_regions, &[x.clone()]).unwrap().into_iter().next().unwrap()
                },
                Array::scalar(3.0),
            )
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 3.0f64.sin(), epsilon = 1e-9);
        // Reverse mode transposes the linearized custom rule, so the doubled derivative carries over.
        assert_abs_diff_eq!(gradient.to_f64s()[0], 2.0 * 3.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_vjp_governs_reverse_mode() {
        let (value, gradient) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .value_and_gradient(
                |x| {
                    let (operation, operation_regions) = custom_vjp_sin(&test_type(&[]));
                    x.context().bind(operation, operation_regions, &[x.clone()]).unwrap().into_iter().next().unwrap()
                },
                Array::scalar(2.0),
            )
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 2.0f64.sin(), epsilon = 1e-9);
        // The custom backward rule triples the true gradient, proving it is in control.
        assert_abs_diff_eq!(gradient.to_f64s()[0], 3.0 * 2.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_vjp_rejects_forward_mode() {
        // The staged tangent carrier refuses interpretation in its un-transposed (pushforward) form, which is
        // exactly the operation `jvp` would need to execute; reverse mode transposes it first and replays
        // `backward`.
        let scalar = test_type(&[]);
        let carrier = CustomVjpTangentOperation::new(1, false, vec![scalar.clone()], vec![scalar]);
        assert!(matches!(
        InterpretableOperation::<_>::interpret(
                        &carrier,
                        &crate::EagerContext::<Array>::new(), &crate::EmptyRegionDriver,
                                        &[Array::scalar(1.0)],
                    ),
                    Err(ProgramError::Type(TypeError::Invalid { message }))
                        if message.starts_with("custom_vjp does not support forward-mode differentiation"),
                ));
    }

    #[test]
    fn test_jacobian_reverse_through_custom_vjp_uses_the_custom_backward_rule() {
        use crate::differentiation::jacobian::jacobian_reverse;

        // jacobian_reverse interprets the pullback with batch-stacked cotangent bases, exercising the batched replay
        // of the custom backward program. The Jacobian of elementwise `sin` with the tripled rule is the diagonal
        // matrix `diag(3 * cos(x))`.
        let vector = test_type(&[2]);
        let jacobian = jacobian_reverse(
            |x| {
                let (operation, operation_regions) = custom_vjp_sin(&test_type(&[2]));
                Ok(x.context().bind(operation, operation_regions, &[x.clone()])?.into_iter().next().unwrap())
            },
            Array::from_f64s(vector, vec![0.5, 1.0]),
        )
        .unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_abs_diff_eq!(block.value().values()[0], 3.0 * 0.5f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().values()[1], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().values()[2], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().values()[3], 3.0 * 1.0f64.cos(), epsilon = 1e-9);
    }

    /// Builds the scalar `f(x) = sin(x)` program.
    fn scalar_sin_program() -> Program<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>> {
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(SinOperation, Vec::new(), vec![input]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds the deliberately wrong scalar rule `jvp(x, dx) = (sin(x), 2 * cos(x) * dx)`.
    fn scalar_doubled_sin_jvp_program() -> Program<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>> {
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(DataType::F64);
        let dx = builder.add_input(DataType::F64);
        let y = builder.add_instruction(SinOperation, Vec::new(), vec![x]).unwrap()[0];
        let cosine = builder.add_instruction(CosOperation, Vec::new(), vec![x]).unwrap()[0];
        let two = builder.add_constant(Scalar::from(2.0));
        let scaled = builder.add_instruction(MulOperation, Vec::new(), vec![two, cosine]).unwrap()[0];
        let tangent = builder.add_instruction(MulOperation, Vec::new(), vec![scaled, dx]).unwrap()[0];
        builder
            .build(vec![y, tangent], vec![Placeholder, Placeholder], vec![Placeholder, Placeholder])
            .unwrap()
    }

    /// Builds the malformed rule `jvp(x, dx) = (sin(x), 1)`. Its tangent ignores `dx` and is therefore an affine
    /// constant rather than a linear tangent map.
    fn scalar_known_tangent_jvp_program() -> Program<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>> {
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(DataType::F64);
        builder.add_input(DataType::F64);
        let y = builder.add_instruction(SinOperation, Vec::new(), vec![x]).unwrap()[0];
        let tangent = builder.add_constant(Scalar::from(1.0));
        builder
            .build(vec![y, tangent], vec![Placeholder, Placeholder], vec![Placeholder, Placeholder])
            .unwrap()
    }

    /// Builds the scalar forward rule `forward(x) = (sin(x), cos(x))`, with the cosine as the residual.
    fn scalar_sin_forward_program() -> Program<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>> {
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(DataType::F64);
        let y = builder.add_instruction(SinOperation, Vec::new(), vec![x]).unwrap()[0];
        let residual = builder.add_instruction(CosOperation, Vec::new(), vec![x]).unwrap()[0];
        builder.build(vec![y, residual], vec![Placeholder], vec![Placeholder, Placeholder]).unwrap()
    }

    /// Builds the deliberately wrong scalar rule `backward(residual, cotangent) = 3 * residual * cotangent`.
    fn scalar_tripled_sin_backward_program() -> Program<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>> {
        let mut builder = ProgramBuilder::new();
        let residual = builder.add_input(DataType::F64);
        let cotangent = builder.add_input(DataType::F64);
        let three = builder.add_constant(Scalar::from(3.0));
        let scaled = builder.add_instruction(MulOperation, Vec::new(), vec![three, residual]).unwrap()[0];
        let gradient = builder.add_instruction(MulOperation, Vec::new(), vec![scaled, cotangent]).unwrap()[0];
        builder.build(vec![gradient], vec![Placeholder, Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_scalar_custom_jvp_governs_forward_mode() {
        let (primal, tangent) = EagerContext::<Scalar, ScalarOperation<Scalar>>::new()
            .jvp(
                |x| {
                    let operation = ScalarOperation::CustomJvp(CustomJvpOperation::new());
                    let operation_regions = vec![scalar_sin_program(), scalar_doubled_sin_jvp_program()];
                    Ok(x.context().bind(operation, operation_regions, &[x.clone()])?.into_iter().next().unwrap())
                },
                Scalar::from(2.0),
                Scalar::from(1.0),
            )
            .unwrap();
        assert_abs_diff_eq!(primal, 2.0f64.sin(), epsilon = 1e-9);
        // The custom rule doubles the true derivative, proving it is in control.
        assert_abs_diff_eq!(tangent, 2.0 * 2.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_linearization_rejects_known_custom_jvp_tangents() {
        let operation = ScalarOperation::CustomJvp(CustomJvpOperation::new());
        let operation_regions = || vec![scalar_sin_program(), scalar_known_tangent_jvp_program()];
        let expected = "linearization produced a known tangent output; differentiation rules must represent \
                        input-independent zero tangents structurally";

        // Program-level direct linearization must reject the malformed rule rather than silently replacing its
        // constant tangent with zero.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let region_ids = operation_regions()
            .iter()
            .map(|region| builder.import_region(region.entry_region_ref()))
            .collect::<Vec<_>>();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation.clone(), region_ids, vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            program.linearize(),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message))) if message == expected,
        ));

        // Value-level direct linearization enforces the same rule contract before exposing a reusable pushforward.
        let result = EagerContext::<Scalar, ScalarOperation<Scalar>>::new().linearize(
            |input| {
                let mut outputs = input.context().bind(operation, operation_regions(), &[input.clone()])?;
                Ok(outputs.remove(0))
            },
            Scalar::from(2.0),
        );
        assert!(matches!(
            result,
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message))) if message == expected,
        ));
    }

    #[test]
    fn test_scalar_custom_vjp_governs_reverse_mode() {
        let (value, gradient) = EagerContext::<Scalar, ScalarOperation<Scalar>>::new()
            .value_and_gradient(
                |x| {
                    let operation = ScalarOperation::CustomVjp(CustomVjpOperation::new());
                    let operation_regions =
                        vec![scalar_sin_program(), scalar_sin_forward_program(), scalar_tripled_sin_backward_program()];
                    x.context().bind(operation, operation_regions, &[x.clone()]).unwrap().into_iter().next().unwrap()
                },
                Scalar::from(2.0),
            )
            .unwrap();
        assert_abs_diff_eq!(value, 2.0f64.sin(), epsilon = 1e-9);
        // The custom backward rule triples the true gradient, proving it is in control.
        assert_abs_diff_eq!(gradient, 3.0 * 2.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_jvp_wrapper_traces_closures_lazily() {
        // No manual programs: the wrapper traces the closures at the call site, specialized to the input types.
        let function = custom_jvp::<EagerContext<Array, ArrayOperation<Array>>, _, _, _, _>(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok(x.sin()?),
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>, dx| {
                // The deliberately wrong rule `jvp(x, dx) = (sin(x), cos(x) * dx + cos(x) * dx)` doubles the true
                // derivative (expressed through addition to avoid constant lifting), proving the rule is in control.
                let tangent = x.cos()? * dx;
                Ok((x.sin()?, tangent.clone() + tangent))
            },
        );
        let (primal, tangent) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jvp(|x| function.call(x), Array::scalar(2.0), Array::scalar(1.0))
            .unwrap();
        assert_abs_diff_eq!(primal.to_f64s()[0], 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(tangent.to_f64s()[0], 2.0 * 2.0f64.cos(), epsilon = 1e-9);
        // Reverse mode transposes the linearized custom rule, so the doubled derivative carries over.
        let (value, gradient) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .value_and_gradient(|x| function.call(x).unwrap(), Array::scalar(3.0))
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 3.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.to_f64s()[0], 2.0 * 3.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_derivative_wrappers_use_zero_space_boundaries() {
        type ScalarContext = EagerContext<Scalar, ScalarOperation<Scalar>>;

        let function = custom_jvp::<ScalarContext, _, _, _, _>(
            |token: DomainTracer<ScalarContext>| Ok(token),
            |token: DomainTracer<ScalarContext>, tangent| Ok((token, tangent)),
        );
        assert_eq!(
            ScalarContext::new().jvp(|token| function.call(token), Scalar::Token, Scalar::Zero),
            Ok((Scalar::Token, Scalar::Zero)),
        );

        let function = custom_vjp::<ScalarContext, _, _, _, _, _, _>(
            |token: DomainTracer<ScalarContext>| Ok(token),
            |token: DomainTracer<ScalarContext>| Ok((token.clone(), token)),
            |_residual: DomainTracer<ScalarContext>, cotangent| Ok(cotangent),
        );
        let (value, pullback) = ScalarContext::new().vjp(|token| function.call(token), Scalar::Token).unwrap();
        assert_eq!(value, Scalar::Token);
        assert_eq!(pullback.apply(Scalar::Zero), Ok(Scalar::Zero));
    }

    #[test]
    fn test_custom_vjp_wrapper_governs_reverse_mode() {
        let function = custom_vjp::<EagerContext<Array, ArrayOperation<Array>>, _, _, _, _, _, _>(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok(x.sin()?),
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok((x.sin()?, x.cos()?)),
            |residual, cotangent| {
                // The deliberately wrong rule `backward(residual, cotangent) = 3 * residual * cotangent` triples the
                // true gradient (expressed through addition to avoid constant lifting).
                let product = residual * cotangent;
                Ok(product.clone() + product.clone() + product)
            },
        );
        let (value, gradient) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .value_and_gradient(|x| function.call(x).unwrap(), Array::scalar(2.0))
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.to_f64s()[0], 3.0 * 2.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_vjp_wrapper_supports_structured_signatures_and_captured_configuration() {
        // Tuple inputs and tuple residuals exercise the `Parameterized` (pytree) calling convention, and the
        // captured `triple` closure plays the role of a JAX `nondiff_argnums` argument: static configuration
        // visible to the rule closures without being differentiated or stored as a residual.
        let repeats = 3usize;
        let function = custom_vjp::<EagerContext<Array, ArrayOperation<Array>>, _, _, _, _, _, _>(
            |(x, y): (
                DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
                DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
            )| Ok(x * y),
            |(x, y): (
                DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
                DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
            )| { Ok((x.clone() * y.clone(), (x, y))) },
            move |(x, y), cotangent| {
                // The deliberately wrong rule repeats both cotangents `repeats` times via the captured count.
                let (base_x, base_y) = (y * cotangent.clone(), x * cotangent);
                let (mut scaled_x, mut scaled_y) = (base_x.clone(), base_y.clone());
                for _ in 1..repeats {
                    scaled_x = scaled_x + base_x.clone();
                    scaled_y = scaled_y + base_y.clone();
                }
                Ok((scaled_x, scaled_y))
            },
        );
        let (value, (gradient_x, gradient_y)) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .value_and_gradient(|(x, y)| function.call((x, y)).unwrap(), (Array::scalar(2.0), Array::scalar(5.0)))
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 10.0, epsilon = 1e-9);
        // The custom rule triples the true gradients `(y, x)`.
        assert_abs_diff_eq!(gradient_x.to_f64s()[0], 3.0 * 5.0, epsilon = 1e-9);
        assert_abs_diff_eq!(gradient_y.to_f64s()[0], 3.0 * 2.0, epsilon = 1e-9);
    }

    #[test]
    fn test_scalar_custom_vjp_wrapper_governs_reverse_mode() {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let function = custom_vjp::<EagerContext<Scalar, ScalarOperation<Scalar>>, _, _, _, _, _, _>(
            |x: DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>| Ok(x.sin()?),
            |x: DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>| Ok((x.sin()?, x.cos()?)),
            |residual, cotangent| {
                let product = residual * cotangent;
                Ok(product.clone() + product.clone() + product)
            },
        );
        let (value, gradient) = domain.value_and_gradient(|x| function.call(x).unwrap(), Scalar::from(2.0)).unwrap();
        assert_abs_diff_eq!(value, 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient, 3.0 * 2.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_jvp_wrapper_surfaces_rule_signature_mismatches() {
        // Arity mismatches are compile-time errors under the structured signatures, but shape mismatches remain
        // runtime concerns: this rule produces a scalar tangent for a vector-valued function, so the traced JVP
        // program fails the signature validation that `CustomJvpOperation::new` performs at the call site.
        let function = custom_jvp::<EagerContext<Array, ArrayOperation<Array>>, _, _, _, _>(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok(x.sin()?),
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>, dx| {
                Ok((x.sin()?, dx.dot(&dx, &DotDimensionNumbers::inner_product())))
            },
        );
        let error =
            EagerContext::<Array, ArrayOperation<Array>>::trace(|x| function.call(x), test_type(&[2])).unwrap_err();
        assert!(error.to_string().contains("custom_jvp rule output"));
    }

    #[test]
    fn test_custom_jvp_batches_by_rewrapping_the_call() {
        let scalar = test_type(&[]);
        let output: Array = EagerContext::<Array, ArrayOperation<Array>>::new()
            .batch(
                |x| {
                    let (operation, operation_regions) = custom_jvp_sin(&scalar);
                    Ok(x.context().bind(operation, operation_regions, &[x.clone()])?.into_iter().next().unwrap())
                },
                Array::vector(vec![0.5, 1.0, 1.5]),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
        for (actual, input) in output.to_f64s().iter().zip([0.5f64, 1.0, 1.5]) {
            assert_abs_diff_eq!(*actual, input.sin(), epsilon = 1e-9);
        }
    }

    #[test]
    fn test_custom_jvp_survives_batching_and_governs_the_batched_gradient() {
        use crate::batching::Batch;
        use crate::differentiation::LinearizationTracer;
        use crate::operations::math::{Reduce, ReductionKind};

        // Differentiating *through* a batch of the custom call must still use the (deliberately doubled) custom
        // rule: batching re-wraps the call around batched programs instead of inlining the primal, so the
        // custom derivative survives `batch` — mirroring JAX's `vmap`-of-`custom_jvp` semantics.
        let (value, gradient) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .value_and_gradient(
                |x| {
                    let context = x.context().clone();
                    let mapped: LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>> = Batch::batch(
                        &context,
                        |item| {
                            let (operation, operation_regions) = custom_jvp_sin(&test_type(&[]));
                            Ok(item
                                .context()
                                .bind(operation, operation_regions, &[item.clone()])?
                                .into_iter()
                                .next()
                                .unwrap())
                        },
                        x,
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        None,
                    )
                    .unwrap();
                    mapped.reduce(&[0], ReductionKind::Sum)
                },
                Array::vector(vec![0.5, 1.0]),
            )
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 0.5f64.sin() + 1.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.to_f64s()[0], 2.0 * 0.5f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.to_f64s()[1], 2.0 * 1.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_vjp_survives_batching_and_governs_the_batched_gradient() {
        use crate::batching::Batch;
        use crate::differentiation::LinearizationTracer;
        use crate::operations::math::{Reduce, ReductionKind};

        // The reverse-mode analogue of the test above: the (deliberately tripled) custom backward rule governs the
        // gradient through the batched call — mirroring JAX's `vmap`-of-`custom_vjp` semantics.
        let (value, gradient) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .value_and_gradient(
                |x| {
                    let context = x.context().clone();
                    let mapped: LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>> = Batch::batch(
                        &context,
                        |item| {
                            let (operation, operation_regions) = custom_vjp_sin(&test_type(&[]));
                            Ok(item
                                .context()
                                .bind(operation, operation_regions, &[item.clone()])?
                                .into_iter()
                                .next()
                                .unwrap())
                        },
                        x,
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        None,
                    )
                    .unwrap();
                    mapped.reduce(&[0], ReductionKind::Sum)
                },
                Array::vector(vec![0.5, 1.0]),
            )
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 0.5f64.sin() + 1.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.to_f64s()[0], 3.0 * 0.5f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.to_f64s()[1], 3.0 * 1.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_user_custom_vjp_pullback_applies_the_user_backward_program() {
        // First-order reverse mode through a user custom VJP applies the user-supplied backward program. The
        // reverse entry stages an opaque tangent carrier and the direct transpose replays the backward program forward
        // into the pullback, so seeding the pullback at `[cotangent ++ residuals]` recovers `residual * cotangent`. The
        // user backward defines the residual as `cos(x)`, so at `x = 0.7` and a unit cotangent the input cotangent is
        // `cos(0.7)`.
        let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
        let function = custom_vjp::<EagerContext<Array, ArrayOperation<Array>>, _, _, _, _, _, _>(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok(x.sin()?),
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok((x.sin()?, x.cos()?)),
            |residual, cotangent| Ok(residual * cotangent),
        );
        let (_, pullback) = domain.vjp(|x| function.call(x), Array::scalar(0.7)).unwrap();
        let (pullback, residuals) = pullback.into_parts();
        let mut pullback_inputs = vec![Array::scalar(1.0)];
        pullback_inputs.extend(residuals);
        let input_cotangents = pullback.interpret(pullback_inputs).unwrap();
        assert_abs_diff_eq!(input_cotangents[0].to_f64s()[0], 0.7f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_vjp_batching_broadcasts_replicated_inputs() {
        // Mapping only the first input exercises the replicated broadcast in the re-wrapping batch rule: the
        // unmapped operand is broadcast into the batch (the all-inputs-mapped-at-0 convention) and the batched call
        // still computes per-item products.
        let function = custom_vjp::<EagerContext<Array, ArrayOperation<Array>>, _, _, _, _, _, _>(
            |(x, y): (
                DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
                DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
            )| Ok(x * y),
            |(x, y): (
                DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
                DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
            )| { Ok((x.clone() * y.clone(), (x, y))) },
            |(x, y), cotangent| Ok((y * cotangent.clone(), x * cotangent)),
        );
        let output: Array = EagerContext::<Array, ArrayOperation<Array>>::new()
            .batch(
                |(x, y)| function.call((x, y)),
                (Array::vector(vec![2.0, 3.0, 4.0]), Array::scalar(5.0)),
                (BatchAxis::new(0), BatchAxis::replicated()),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
        assert_eq!(output.to_f64s(), vec![10.0, 15.0, 20.0]);
    }
}
