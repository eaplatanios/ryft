use std::fmt::{Debug, Display};

use crate::batching::BatchingError;
use crate::contexts::{EagerContext, StagingContext};
use crate::differentiation::{Cotangent, TransposableOperation};
use crate::domains::Domain;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::AddOperation;
use crate::operations::constants::{MaybeZeroOperation, Zero, ZeroOperation};
use crate::operations::control_flow::ScanOperation;
use crate::operations::control_flow::scan::{
    ScanTypeSemantics, interpret_scan_lanes, read_scan_lane, stacked_scan_type,
};
use crate::operations::manipulation::{BroadcastOperation, Reshape, Slice, UpdateSlice};
use crate::parameters::Parameterized;
use crate::programs::{Program, ProgramError, Value};
use crate::tracing::{AbstractTracingContext, Tracer};
use crate::tracing_v2::ValueOrCapture;
use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation, BatchingContext};
use crate::tracing_v2::differentiation::{
    CaptureParameterizedOperation, DifferentiableOperation, DifferentiationContext, JvpTracer, LinearOperationOf,
    NestedLinearization, ProgramLinearizableOperation, ResidualizedOperation, TangentContext,
};
use crate::tracing_v2::operations::custom_derivatives::CustomVjpResidual;
use crate::types::{ArrayType, DataType, Shape, Size, Type, TypeError, Typed};

/// Renders a compact comma-separated list of capture-like payloads.
pub(crate) fn render_factor_list<C: Display>(factors: &[C]) -> String {
    let mut rendered = String::from("[");
    for (index, factor) in factors.iter().enumerate() {
        if index > 0 {
            rendered.push_str(", ");
        }
        rendered.push_str(&factor.to_string());
    }
    rendered.push(']');
    rendered
}

/// Internal blanket capability for building the linear-operation representation of a residualized scan pushforward.
///
/// This trait is intentionally private to the scan rule. Operation families do not implement it directly; the blanket
/// implementation below derives it from the existing capture-mapping contract and the generated
/// `From<ScanOperation<...>>` conversion for the family's `Scan` variant.
pub(crate) trait LinearScanOperation<T: ScanTypeSemantics, V: Value<T>, R: Value<T>>:
    CaptureParameterizedOperation<T, ValueOrCapture<T, R>> + Sized
{
    /// Builds the linear scan operation.
    fn linear_scan_operation(
        body: Program<T, V, Self, Vec<V>, Vec<V>>,
        residual_stacks: Vec<ValueOrCapture<T, R>>,
        carry_count: usize,
        length: usize,
        reverse: bool,
        unroll: usize,
    ) -> Result<Self, ProgramError>;
}

impl<T, V, R, O> LinearScanOperation<T, V, R> for O
where
    T: ScanTypeSemantics,
    V: Value<T>,
    R: Value<T>,
    O: CaptureParameterizedOperation<T, ValueOrCapture<T, R>>
        + From<
            ScanOperation<
                T,
                V,
                <O as CaptureParameterizedOperation<T, ValueOrCapture<T, R>>>::WithCapture<ValueOrCapture<T, V>>,
                ValueOrCapture<T, R>,
            >,
        >,
{
    fn linear_scan_operation(
        body: Program<T, V, Self, Vec<V>, Vec<V>>,
        residual_stacks: Vec<ValueOrCapture<T, R>>,
        carry_count: usize,
        length: usize,
        reverse: bool,
        unroll: usize,
    ) -> Result<Self, ProgramError> {
        let body = body.map_operations(|operation| {
            operation.try_map_captures(&mut |factor| match factor {
                ValueOrCapture::Capture { index, r#type } => {
                    Ok(ValueOrCapture::Capture { index: *index, r#type: r#type.clone() })
                }
                ValueOrCapture::Value(_) => Err(ProgramError::UnsupportedOperation {
                    message: "scan body pushforwards must reference residual stacks instead of carrying closed \
                              constant captures"
                        .to_string(),
                }),
            })
        })?;
        let scan = ScanOperation::<
            T,
            V,
            <Self as CaptureParameterizedOperation<T, ValueOrCapture<T, R>>>::WithCapture<ValueOrCapture<T, V>>,
        >::new(body, carry_count, length)?
        .with_reverse(reverse)
        .with_unroll(unroll)?
        .with_captures(residual_stacks);
        Ok(Self::from(scan))
    }
}

/// Builds a linear scan operation through [`LinearScanOperation`].
pub(crate) fn linear_scan_operation<T, V, R, O>(
    body: Program<T, V, O, Vec<V>, Vec<V>>,
    residual_stacks: Vec<ValueOrCapture<T, R>>,
    carry_count: usize,
    length: usize,
    reverse: bool,
    unroll: usize,
) -> Result<O, ProgramError>
where
    T: ScanTypeSemantics,
    V: Value<T>,
    R: Value<T>,
    O: LinearScanOperation<T, V, R>,
{
    O::linear_scan_operation(body, residual_stacks, carry_count, length, reverse, unroll)
}

/// Represents scan-local linear operation families that can transpose the captured linear scan body program.
///
/// The body of a linear scan is always expressed over a scan-local residual-reference namespace. This trait names the
/// recursive fixed point needed to transpose that body without making callers restate every variant-level transposition
/// requirement of the operation family. It intentionally acts as a trait-solver recursion breaker for operation enums
/// that contain `Scan` variants.
pub trait LinearScanBodyTransposable<T: Type, V: Value<T>>:
    Operation<T> + MaybeZeroOperation<T> + From<ZeroOperation<T>> + From<AddOperation>
{
    /// Transposes a scan-local linear body program.
    ///
    /// # Parameters
    ///
    ///   - `body`: Scan-local linear body program to transpose.
    fn transpose_linear_scan_body(
        body: &Program<T, V, Self, Vec<V>, Vec<V>>,
    ) -> Result<Program<T, V, Self, Vec<V>, Vec<V>>, ProgramError>
    where
        Self: Sized;
}

/// Represents operation families that can recursively interpret nested flat programs.
///
/// This trait names the recursive fixed point needed by higher-order interpretation helpers without requiring the
/// full operation enum's [`InterpretableOperation`](crate::operations::InterpretableOperation) impl while proving
/// that impl. Operation families implement it by replaying a nested flat [`Program`] through their operation-owned
/// interpretation rules.
pub trait InterpretableNestedProgram<T: Type, V: Value<T>>: Operation<T> {
    /// Interprets a nested flat program.
    ///
    /// # Parameters
    ///
    ///   - `context`: Interpretation context used for body operations.
    ///   - `program`: Nested program to interpret.
    ///   - `input`: Input values for `program`.
    fn interpret_nested_program(
        context: &V::InterpretationContext,
        program: &Program<T, V, Self, Vec<V>, Vec<V>>,
        input: Vec<V>,
    ) -> Result<Vec<V>, ProgramError>
    where
        Self: Sized;
}

impl<V, O, F> ScanOperation<DataType, V, O, F>
where
    V: Value<DataType>,
    F: Value<DataType>,
    O: CaptureParameterizedOperation<DataType, ValueOrCapture<DataType, V>>,
    <O as CaptureParameterizedOperation<DataType, ValueOrCapture<DataType, V>>>::WithCapture<V>:
        InterpretableNestedProgram<DataType, V>,
{
    /// Interprets a carry-only linear scalar scan whose body payloads use scan-local captured factors.
    ///
    /// Scalar scans currently have no scalar-stack representation, so this path only supports residual-free linear
    /// scan bodies. The regular [`Operation::infer_output_types`] call enforces that any stored captures are rejected
    /// before the body is replayed. The body operations are then instantiated into their ordinary value-payload form
    /// and interpreted through the operation family's nested-program interpreter.
    ///
    /// # Parameters
    ///
    ///   - `context`: Interpretation context used for instantiated body operations.
    ///   - `inputs`: Initial carry values.
    pub(crate) fn interpret(
        &self,
        context: &<V as Value<DataType>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(input_types.as_slice())?;
        let body = self.body().map_operations(|operation| {
            operation.try_map_captures(&mut |capture: &ValueOrCapture<DataType, V>| capture.instantiate(&[]))
        })?;
        let mut state = inputs.to_vec();
        for _ in 0..self.length() {
            state = <<O as CaptureParameterizedOperation<DataType, ValueOrCapture<DataType, V>>>::WithCapture<
                V,
            > as InterpretableNestedProgram<DataType, V>>::interpret_nested_program(context, &body, state)?;
            check_count!("output", state, self.carry_count(), ProgramError);
        }
        Ok(state)
    }
}

// TODO(eaplatanios): Is this actually necessary?
/// Interprets a captured linear [`ScanOperation`] in an active interpretation context.
///
/// Linear scan interpretation is context-mediated because the operation owns a nested body program whose captures must
/// be instantiated per lane before the body is replayed.
pub trait LinearScanInterpretation<V: Value<ArrayType>, F: Value<ArrayType>, O: Operation<ArrayType>> {
    /// Interprets `operation` over `inputs`.
    ///
    /// # Parameters
    ///
    ///   - `operation`: Linear scan operation to interpret.
    ///   - `inputs`: Carry tangents followed by stacked input tangents.
    fn interpret_linear_scan(
        &self,
        operation: &ScanOperation<ArrayType, V, O, F>,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError>;
}

impl<V, F, O, ContextT> LinearScanInterpretation<V, F, O> for ContextT
where
    V: Value<ArrayType, InterpretationContext = ContextT> + Slice + UpdateSlice + Reshape,
    ContextT: Zero<ArrayType, V>,
    F: CustomVjpResidual<ArrayType, V>,
    O: CaptureParameterizedOperation<ArrayType, ValueOrCapture<ArrayType, V>>,
    <O as CaptureParameterizedOperation<ArrayType, ValueOrCapture<ArrayType, V>>>::WithCapture<V>:
        InterpretableNestedProgram<ArrayType, V>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    fn interpret_linear_scan(
        &self,
        operation: &ScanOperation<ArrayType, V, O, F>,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        operation.infer_output_types(input_types.as_slice())?;
        let stack_values =
            operation.captures().iter().map(CustomVjpResidual::residual_value).collect::<Result<Vec<_>, _>>()?;
        let body = operation.body();
        let carry_count = operation.carry_count();
        let y_slice_types = body.output_types().split_off(carry_count);
        interpret_scan_lanes(
            carry_count,
            operation.length(),
            operation.reverse(),
            y_slice_types.as_slice(),
            inputs,
            |stacked_type| self.zero(stacked_type),
            |lane, lane_inputs| {
                let lane_residuals =
                    stack_values.iter().map(|stack| read_scan_lane(stack, lane)).collect::<Result<Vec<_>, _>>()?;
                let lane_body = body.map_operations(|operation| {
                    operation.try_map_captures(&mut |capture: &ValueOrCapture<ArrayType, V>| {
                        capture.instantiate(&lane_residuals)
                    })
                })?;
                <<O as CaptureParameterizedOperation<ArrayType, ValueOrCapture<ArrayType, V>>>::WithCapture<
                    V,
                > as InterpretableNestedProgram<ArrayType, V>>::interpret_nested_program(
                    self,
                    &lane_body,
                    lane_inputs,
                )
            },
        )
    }
}

/// JVP rule for [`ScanOperation`] with full JAX
/// [`scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html) JVP-plus-partial-evaluation parity.
/// Like every `ryft` JVP rule it always replays the body symbolically rather than concretizing it, but scan also
/// stages a *single* strategy: it has a static trip count and no predicate, so unlike the
/// [`WhileOperation`](crate::operations::control_flow::WhileOperation) rule — which picks between a bounded
/// masked-scan path and an unbounded fused-loop path at rule time — there is no branching at all, and one
/// residual-extended primal scan plus one linear scan are staged in every domain. Because scan stores its residuals
/// statically, reverse mode through it is always total (`scan` and bounded `while` are the reverse-capable loops;
/// unbounded `while` is forward-mode only — see the [`WhileOperation`] rule).
///
///   1. The body is linearized *symbolically* once at the body slice types via [`linearize_program`](
///      crate::tracing_v2::linearize_program) — no primal values are involved and no iteration runs here.
///   2. The residual-extended primal body becomes the body of a new primal [`ScanOperation`] with the same carry
///      count, length, direction, and (lowering-only) unroll factor: the appended residual outputs become *extra
///      scanned outputs*, so the primal scan **stores** every per-iteration residual as a statically shaped
///      `[length, …]` stack. This is the static-trip-count payoff over unbounded `while`, whose staged linear loop
///      must *recompute* its loop-varying residuals forward and therefore cannot transpose (a `while` with a
///      semantic [`iteration_bound`](crate::operations::control_flow::WhileOperation::with_iteration_bound) recovers
///      the same payoff by storing its residuals into `[bound, …]` stacks and staging a masked linear scan).
///   3. The stacked residual outputs are registered in the enclosing linearization residual environment, and one
///      linear [`ScanOperation`] is staged over the operand tangents with those stacks as capture payloads. The body
///      pushforward keeps its residual references *scan-local* (reference `i` resolves to slice `lane` of stack `i`
///      while iteration `lane` runs), so no remapping onto the enclosing namespace happens. Closed constant captures
///      from body constants are broadcast into lane-uniform stacks first, keeping the scan-local namespace
///      reference-only.
///
/// **Alignment invariant.** When [`reverse`](ScanOperation::reverse) is set, the primal scan visits lanes from
/// `length - 1` down to `0` but still writes output and residual slice `i` aligned with input slice `i`, and the
/// linear scan runs with the *same* `reverse`, so residual lane `i` is consumed exactly while tangent lane `i` is
/// processed. Transposing the linear scan flips `reverse` (and transposes the body program), which pairs cotangent
/// lane `i` with residual lane `i` in the opposite visit order — making reverse-mode differentiation through `scan`
/// total, with no array-reversal operation anywhere.
impl<V, D, O> DifferentiableOperation<D> for ScanOperation<ArrayType, V, O>
where
    V: Value<ArrayType>,
    D: DifferentiationContext<Type = ArrayType, Constant = V> + Domain<Operation = O>,
    O: Clone
        + Operation<ArrayType>
        + From<BroadcastOperation>
        + From<ScanOperation<ArrayType, V, O>>
        + ProgramLinearizableOperation<D>,
    LinearOperationOf<D>: ResidualizedOperation<D> + LinearScanOperation<ArrayType, D::Tangent, D::Value>,
    LinearOperationOf<D>: From<ZeroOperation<ArrayType>> + MaybeZeroOperation<ArrayType>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        let carry_count = self.carry_count();
        let length = self.length();
        let reverse = self.reverse();
        let unroll = self.unroll();
        let input_count = self.body().input_types().len();
        let output_count = self.body().output_types().len();
        check_count!("input", inputs, input_count, ProgramError);

        let all_tangents_zero = inputs
            .iter()
            .try_fold(true, |all_zero, input| Ok::<_, ProgramError>(all_zero && context.is_zero(input.tangent())?))?;
        if all_tangents_zero {
            let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            let primal_outputs = context.bind_primal(O::from(self.clone()), primal_inputs.as_slice())?;
            check_count!("output", primal_outputs, output_count, ProgramError);
            let mut tangent_outputs = Vec::with_capacity(output_count);
            for primal in &primal_outputs {
                let mut tangent = context.stage_nullary_operation(ZeroOperation::new(primal.r#type().into_owned()))?;
                check_count!("output", tangent, 1, ProgramError);
                tangent_outputs.push(tangent.remove(0));
            }
            return Ok(primal_outputs
                .into_iter()
                .zip(tangent_outputs)
                .map(|(primal, tangent)| JvpTracer::from_value(primal, tangent))
                .collect());
        }

        // Linearize the body symbolically once at the body slice types.
        let NestedLinearization { primal_program, pushforward_program, residual_types } =
            self.body().linearize(context.differentiable())?;
        let residual_count = residual_types.len();

        // Bind the residual-extended primal scan: the appended residual outputs become extra scanned outputs, so
        // the primal scan stores every per-iteration residual as a statically shaped stack. The lowering-only
        // unroll factor carries over from the differentiated scan.
        let extended_scan = ScanOperation::<ArrayType, V, O>::new(primal_program, carry_count, length)?
            .with_reverse(reverse)
            .with_unroll(unroll)?;
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let mut bound_outputs = context.bind_primal(O::from(extended_scan), primal_inputs.as_slice())?;
        check_count!("output", bound_outputs, output_count + residual_count, ProgramError);
        let residual_stack_values = bound_outputs.split_off(output_count);
        let primal_outputs = bound_outputs;

        // Register the stacked residuals in the enclosing residual environment, then rewrite any closed constant
        // factors in the body pushforward into lane-uniform residual stacks (broadcast along a fresh leading
        // length axis) so the scan-local factor namespace stays reference-only.
        let mut stack_factors =
            residual_stack_values.into_iter().map(|value| context.factor(value)).collect::<Vec<_>>();
        let body = pushforward_program.map_operations(|operation| {
            operation.try_map_captures(&mut |factor| match factor {
                ValueOrCapture::Capture { index, r#type } => {
                    Ok(ValueOrCapture::Capture { index: *index, r#type: r#type.clone() })
                }
                ValueOrCapture::Value(value) => {
                    let value_type = value.r#type().into_owned();
                    if value_type.static_shape().is_none() {
                        return Err(TypeError {
                            message: format!(
                                "scan body pushforwards cannot capture a constant factor of dynamically sized type \
                                 {value_type}",
                            ),
                        }
                        .into());
                    }
                    let stacked_type = stacked_scan_type(&value_type, length);
                    let output_axes = (1..=value_type.rank()).collect::<Vec<_>>();
                    let mut broadcasted = context.bind_primal(
                        O::from(BroadcastOperation::new(stacked_type, output_axes)),
                        std::slice::from_ref(value),
                    )?;
                    check_count!("output", broadcasted, 1, ProgramError);
                    let scan_local_index = stack_factors.len();
                    stack_factors.push(context.factor(broadcasted.remove(0)));
                    Ok(ValueOrCapture::Capture { index: scan_local_index, r#type: value_type })
                }
            })
        })?;

        // Stage one linear scan over the operand tangents and pair its outputs with the primal outputs of the
        // residual-extended scan.
        let tangent_operands = inputs.iter().map(|input| input.tangent().clone()).collect::<Vec<_>>();
        let linear_scan: LinearOperationOf<D> =
            linear_scan_operation(body, stack_factors, carry_count, length, reverse, unroll)?;
        let tangent_outputs = context.stage_operation(linear_scan, tangent_operands.as_slice())?;
        check_count!("output", tangent_outputs, output_count, ProgramError);
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| JvpTracer::from_value(primal, tangent))
            .collect())
    }
}

impl<V, D, O> DifferentiableOperation<D> for ScanOperation<DataType, V, O>
where
    V: Value<DataType>,
    D: DifferentiationContext<Type = DataType, Constant = V> + Domain<Operation = O>,
    O: Clone + Operation<DataType> + From<ScanOperation<DataType, V, O>> + ProgramLinearizableOperation<D>,
    LinearOperationOf<D>: ResidualizedOperation<D> + LinearScanOperation<DataType, D::Tangent, D::Value>,
    LinearOperationOf<D>: From<ZeroOperation<DataType>> + MaybeZeroOperation<DataType>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        let carry_count = self.carry_count();
        let length = self.length();
        let reverse = self.reverse();
        let unroll = self.unroll();
        let input_count = self.body().input_types().len();
        let output_count = self.body().output_types().len();
        check_count!("input", inputs, input_count, ProgramError);

        let all_tangents_zero = inputs
            .iter()
            .try_fold(true, |all_zero, input| Ok::<_, ProgramError>(all_zero && context.is_zero(input.tangent())?))?;
        if all_tangents_zero {
            let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            let primal_outputs = context.bind_primal(O::from(self.clone()), primal_inputs.as_slice())?;
            check_count!("output", primal_outputs, output_count, ProgramError);
            let mut tangent_outputs = Vec::with_capacity(output_count);
            for primal in &primal_outputs {
                let mut tangent = context.stage_nullary_operation(ZeroOperation::new(primal.r#type().into_owned()))?;
                check_count!("output", tangent, 1, ProgramError);
                tangent_outputs.push(tangent.remove(0));
            }
            return Ok(primal_outputs
                .into_iter()
                .zip(tangent_outputs)
                .map(|(primal, tangent)| JvpTracer::from_value(primal, tangent))
                .collect());
        }

        let NestedLinearization { primal_program, pushforward_program, residual_types } =
            self.body().linearize(context.differentiable())?;
        if !residual_types.is_empty() {
            return Err(ProgramError::UnsupportedOperation {
                message: "scalar scan JVP with per-iteration residuals requires a scalar stack representation"
                    .to_string(),
            });
        }

        let extended_scan = ScanOperation::<DataType, V, O>::new(primal_program, carry_count, length)?
            .with_reverse(reverse)
            .with_unroll(unroll)?;
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal_outputs = context.bind_primal(O::from(extended_scan), primal_inputs.as_slice())?;
        check_count!("output", primal_outputs, output_count, ProgramError);

        let body = pushforward_program.map_operations(|operation| {
            let mut has_factor = false;
            let operation = operation.try_map_captures(&mut |factor| {
                has_factor = true;
                Ok(factor.clone())
            })?;
            if has_factor {
                return Err(ProgramError::UnsupportedOperation {
                    message: "scalar scan JVP with captured factors requires a scalar stack representation".to_string(),
                });
            }
            Ok(operation)
        })?;
        let tangent_operands = inputs.iter().map(|input| input.tangent().clone()).collect::<Vec<_>>();
        let linear_scan: LinearOperationOf<D> =
            linear_scan_operation(body, vec![], carry_count, length, reverse, unroll)?;
        let tangent_outputs = context.stage_operation(linear_scan, tangent_operands.as_slice())?;
        check_count!("output", tangent_outputs, output_count, ProgramError);
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| JvpTracer::from_value(primal, tangent))
            .collect())
    }
}

/// Extracts slice `lane` of a stacked batch along its *logical* leading axis and drops that axis.
///
/// The logical leading axis is the scan length axis: physical axis `1` when the batch axis sits at physical axis
/// `0`, and physical axis `0` otherwise. The lane batch keeps the input's batch axis, decremented when it sat after
/// the dropped axis.
fn read_scan_lane_batch<V>(stack: &ArrayBatch<V>, lane: usize) -> Result<ArrayBatch<V>, ProgramError>
where
    V: Value<ArrayType> + Slice + Reshape,
{
    let stack_axis = match stack.batch_axis() {
        Some(0) => 1,
        _ => 0,
    };
    let stack_type = stack.r#type().into_owned();
    let dimensions = stack_type
        .shape()
        .dimensions()
        .iter()
        .map(|dimension| {
            dimension.value().ok_or_else(|| {
                BatchingError::UnsupportedOperation {
                    message: format!("scan batching requires static stacked input types but got {stack_type}"),
                }
                .into()
            })
        })
        .collect::<Result<Vec<usize>, ProgramError>>()?;
    let mut start_indices = vec![0; dimensions.len()];
    start_indices[stack_axis] = lane;
    let mut limit_indices = dimensions.clone();
    limit_indices[stack_axis] = lane + 1;
    let unit_strides = vec![1; dimensions.len()];
    let lane_value =
        stack
            .value()
            .clone()
            .slice(start_indices.as_slice(), limit_indices.as_slice(), unit_strides.as_slice())?;
    let lane_dimensions = dimensions
        .iter()
        .enumerate()
        .filter(|(axis, _)| *axis != stack_axis)
        .map(|(_, &dimension)| Size::Static(dimension))
        .collect::<Vec<_>>();
    let lane_value = lane_value.reshape(Shape::new(lane_dimensions))?;
    let lane_type = lane_value.r#type().into_owned();
    let lane_axis = stack.batch_axis().map(|axis| if axis > stack_axis { axis - 1 } else { axis });
    ArrayBatch::new(lane_type, lane_value, lane_axis)
}

/// Per-output stacking state used by [`batch_scan_with_interpreter`]: the accumulator batch holding the lanes
/// written so far, together with the lane batch axis every lane must agree on.
struct ScanOutputAccumulator<V: Typed<ArrayType>> {
    /// Stacked accumulator; its leading physical axis is the scan length axis.
    accumulator: V,

    /// Batch axis of the per-lane values written into the accumulator, if the output is lane-varying.
    lane_axis: Option<usize>,
}

/// Drives one batched scan loop over `[carry..., stacked_xs...]` input batches, delegating each iteration's body
/// evaluation to `interpret_lane` and allocating stacked output accumulators through `allocate_zero`.
///
/// Per-iteration slices of the stacked inputs are read along their *logical* leading axis (see
/// [`read_scan_lane_batch`]) so the batch axis threads through untouched, and the per-iteration outputs are stacked
/// along a fresh leading physical axis, shifting each output's batch axis right by one. The visit order reverses
/// when `reverse` is `true` while output slice `i` stays aligned with input slice `i`, exactly like the unbatched
/// scan loop.
pub(crate) fn batch_scan_with_interpreter<V, AllocateZeroFn, InterpretLaneFn>(
    carry_count: usize,
    length: usize,
    reverse: bool,
    y_slice_types: &[ArrayType],
    inputs: &[ArrayBatch<V>],
    mut allocate_zero: AllocateZeroFn,
    mut interpret_lane: InterpretLaneFn,
) -> Result<Vec<ArrayBatch<V>>, ProgramError>
where
    V: Value<ArrayType> + Slice + UpdateSlice + Reshape,
    AllocateZeroFn: FnMut(&ArrayType) -> Result<V, ProgramError>,
    InterpretLaneFn: FnMut(usize, Vec<ArrayBatch<V>>) -> Result<Vec<ArrayBatch<V>>, ProgramError>,
{
    let (initial_carries, stacks) = inputs.split_at(carry_count);
    let mut carries = initial_carries.to_vec();
    let mut accumulators: Vec<Option<ScanOutputAccumulator<V>>> = (0..y_slice_types.len()).map(|_| None).collect();
    let mut lanes: Vec<usize> = (0..length).collect();
    if reverse {
        lanes.reverse();
    }
    for lane in lanes {
        let mut lane_inputs = carries.clone();
        for stack in stacks {
            lane_inputs.push(read_scan_lane_batch(stack, lane)?);
        }
        let mut lane_outputs = interpret_lane(lane, lane_inputs)?;
        check_count!("output", lane_outputs, carry_count + y_slice_types.len(), ProgramError);
        let lane_ys = lane_outputs.split_off(carry_count);
        carries = lane_outputs;
        for (accumulator, lane_y) in accumulators.iter_mut().zip(lane_ys.into_iter()) {
            let lane_axis = lane_y.batch_axis();
            let lane_type = lane_y.r#type().into_owned();
            let accumulator = match accumulator {
                Some(accumulator) => {
                    if accumulator.lane_axis != lane_axis {
                        return Err(BatchingError::MisalignedBatchAxes {
                            message: format!(
                                "scan body produced stacked output lanes at mismatched batch axes ({:?} vs \
                                 {lane_axis:?})",
                                accumulator.lane_axis,
                            ),
                        }
                        .into());
                    }
                    accumulator
                }
                None => accumulator.insert(ScanOutputAccumulator {
                    accumulator: allocate_zero(&stacked_scan_type(&lane_type, length))?,
                    lane_axis,
                }),
            };
            let mut expanded_dimensions = Vec::with_capacity(lane_type.rank() + 1);
            expanded_dimensions.push(Size::Static(1));
            expanded_dimensions.extend(lane_type.shape().dimensions().iter().cloned());
            let expanded = lane_y.into_value().reshape(Shape::new(expanded_dimensions))?;
            let mut start_indices = vec![0; lane_type.rank() + 1];
            start_indices[0] = lane;
            accumulator.accumulator = accumulator.accumulator.update_slice(&expanded, start_indices.as_slice())?;
        }
    }
    let mut outputs = carries;
    for (accumulator, y_slice_type) in accumulators.into_iter().zip(y_slice_types.iter()) {
        match accumulator {
            Some(ScanOutputAccumulator { accumulator, lane_axis }) => {
                let stacked_type = accumulator.r#type().into_owned();
                outputs.push(ArrayBatch::new(stacked_type, accumulator, lane_axis.map(|axis| axis + 1))?);
            }
            None => {
                // A zero-length scan writes no lanes, so each stacked output is the lane-uniform empty stack of
                // the body's per-iteration output type.
                let stacked_type = stacked_scan_type(y_slice_type, length);
                outputs.push(ArrayBatch::unbatched(allocate_zero(&stacked_type)?));
            }
        }
    }
    Ok(outputs)
}

impl<V, O> BatchableOperation<V, EagerContext<ArrayType, V, O>> for ScanOperation<ArrayType, V, O>
where
    V: Value<ArrayType> + Slice + UpdateSlice + Reshape,
    EagerContext<ArrayType, V, O>: Zero<ArrayType, V>,
    O: BatchableOperation<V, EagerContext<ArrayType, V, O>>,
{
    fn batch(
        &self,
        context: &EagerContext<ArrayType, V, O>,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        let y_slice_types = self.body().output_types().split_off(self.carry_count());
        batch_scan_with_interpreter(
            self.carry_count(),
            self.length(),
            self.reverse(),
            y_slice_types.as_slice(),
            inputs,
            |stacked_type| context.zero(stacked_type),
            |_, lane_inputs| {
                self.body().interpret_with(
                    lane_inputs,
                    |_, constant| Ok(ArrayBatch::unbatched(constant.clone())),
                    |instruction, instruction_inputs| instruction.operation().batch(context, instruction_inputs),
                )
            },
        )
    }
}

impl<C, O> BatchableOperation<Tracer<C>, BatchingContext<C>> for ScanOperation<ArrayType, C::Constant, O>
where
    C: StagingContext<Type = ArrayType>,
    C::Constant: Value<ArrayType>,
    C::Operation: From<ZeroOperation<ArrayType>>,
    Tracer<C>: Slice + UpdateSlice + Reshape,
    O: BatchableOperation<Tracer<C>, BatchingContext<C>>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<Tracer<C>>],
    ) -> Result<Vec<ArrayBatch<Tracer<C>>>, ProgramError> {
        let y_slice_types = self.body().output_types().split_off(self.carry_count());
        batch_scan_with_interpreter(
            self.carry_count(),
            self.length(),
            self.reverse(),
            y_slice_types.as_slice(),
            inputs,
            |stacked_type| context.parent_context().zero(stacked_type),
            |_, lane_inputs| context.interpret_program(self.body(), lane_inputs),
        )
    }
}

/// Transpose rule for captured linear scans. Linear-scan transposition is total: the body pushforward maps
/// `[carry..., x_slice...]` to `[carry..., y_slice...]`, so its program transpose maps
/// `[carry_cotangent..., y_slice_cotangent...]` to `[carry_cotangent..., x_slice_cotangent...]` — the same scan-body
/// signature with the same carry count. Flipping `reverse` pairs cotangent lane `i` with residual stack lane `i`
/// exactly when the forward scan consumed them, so the same residual stacks (and the lowering-only unroll factor)
/// carry over verbatim. The body transpose recurses through [`LinearScanBodyTransposable`], keeping the scan-local
/// fixed point owned by the operation family.
impl<V, F, O, Target> TransposableOperation<ArrayType, V, Target> for ScanOperation<ArrayType, V, O, F>
where
    V: Value<ArrayType>,
    F: Value<ArrayType>,
    O: Clone + LinearScanBodyTransposable<ArrayType, V>,
    Target: Operation<ArrayType> + From<ZeroOperation<ArrayType>> + From<ScanOperation<ArrayType, V, O, F>>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<'transpose, ArrayType, V, Target>,
        input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, Target>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, Target>>, ProgramError> {
        // A scan with only zero output cotangents is a zero linear map, so every input cotangent is zero.
        if output_cotangents.iter().all(Cotangent::is_zero) {
            return Ok(vec![Cotangent::Zero; input_types.len()]);
        }
        let body = self.body();
        let carry_count = self.carry_count();
        let length = self.length();
        let transposed_body = <O as LinearScanBodyTransposable<ArrayType, V>>::transpose_linear_scan_body(body)?;
        let transposed = ScanOperation::<ArrayType, V, O>::new(transposed_body, carry_count, length)?
            .with_reverse(!self.reverse())
            .with_unroll(self.unroll())?
            .with_captures(self.captures().to_vec());
        let mut output_types = body.output_types();
        let y_slice_types = output_types.split_off(carry_count);
        output_types.extend(y_slice_types.iter().map(|slice_type| stacked_scan_type(slice_type, length)));
        check_count!("output", output_cotangents, output_types.len(), ProgramError);
        let materialized = output_cotangents
            .iter()
            .zip(output_types.iter())
            .map(|(cotangent, output_type)| {
                crate::tracing_v2::operations::control_flow::stage_cotangent(context, cotangent, output_type)
            })
            .collect::<Vec<_>>();
        let cotangents = context.stage_operation(Target::from(transposed), materialized.as_slice())?;
        check_count!("output", cotangents, input_types.len(), ProgramError);
        Ok(cotangents.into_iter().map(Cotangent::Staged).collect())
    }
}

impl<V, F, O, Target> TransposableOperation<DataType, V, Target> for ScanOperation<DataType, V, O, F>
where
    V: Value<DataType>,
    F: Value<DataType>,
    O: Clone + LinearScanBodyTransposable<DataType, V>,
    Target: Operation<DataType> + From<ZeroOperation<DataType>> + From<ScanOperation<DataType, V, O, F>>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<'transpose, DataType, V, Target>,
        input_types: &[&DataType],
        output_cotangents: &[Cotangent<'transpose, DataType, V, Target>],
    ) -> Result<Vec<Cotangent<'transpose, DataType, V, Target>>, ProgramError> {
        if output_cotangents.iter().all(Cotangent::is_zero) {
            return Ok(vec![Cotangent::Zero; input_types.len()]);
        }
        if !self.captures().is_empty() {
            return Err(ProgramError::UnsupportedOperation {
                message: "scalar linear scan transposition with residual stacks requires a scalar stack representation"
                    .to_string(),
            });
        }
        let body = self.body();
        let output_types = body.output_types();
        check_count!("output", output_cotangents, output_types.len(), ProgramError);
        let transposed_body = <O as LinearScanBodyTransposable<DataType, V>>::transpose_linear_scan_body(body)?;
        let transposed = ScanOperation::<DataType, V, O>::new(transposed_body, self.carry_count(), self.length())?
            .with_reverse(!self.reverse())
            .with_unroll(self.unroll())?
            .with_captures(self.captures().to_vec());
        let materialized = output_cotangents
            .iter()
            .zip(output_types.iter())
            .map(|(cotangent, output_type)| {
                crate::tracing_v2::operations::control_flow::stage_cotangent(context, cotangent, output_type)
            })
            .collect::<Vec<_>>();
        let cotangents = context.stage_operation(Target::from(transposed), materialized.as_slice())?;
        check_count!("output", cotangents, input_types.len(), ProgramError);
        Ok(cotangents.into_iter().map(Cotangent::Staged).collect())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use pretty_assertions::assert_eq;

    use crate::contexts::{EagerContext, StagingContext};
    use crate::operations::InterpretableOperation;
    use crate::operations::arithmetic::{AddOperation, MulOperation, ScaleOperation};
    use crate::operations::constants::ConstantOperation;
    use crate::operations::control_flow::WhileOperation;
    use crate::parameters::Placeholder;
    use crate::payloads::Input;
    use crate::programs::ProgramBuilder;
    use crate::tests::{TestArray, TestArrayDomain};
    use crate::tracing::{DomainTracer, TracingContext};
    use crate::tracing_v2::{ArrayOperation, LinearArrayOperation};
    use crate::types::DataType;

    use super::*;

    type TestOperation = ArrayOperation<TestArray>;
    type TestEagerContext = EagerContext<ArrayType, TestArray, TestOperation>;
    type TestScanOperation = ScanOperation<ArrayType, TestArray, TestOperation>;

    /// Builds a cumulative-product body program that maps `[carry, x]` to `[carry * x, carry * x]`.
    fn product_body() -> Program<ArrayType, TestArray, TestOperation, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestOperation>::new();
        let carry = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let product = builder.add_instruction(MulOperation, vec![carry, x]).unwrap()[0];
        builder
            .build(vec![product, product], vec![Placeholder, Placeholder], vec![Placeholder, Placeholder])
            .unwrap()
    }

    /// Builds the `f64` array type with the provided static dimensions.
    fn f64_type(dimensions: &[usize]) -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(dimensions.iter().copied().map(Size::Static).collect()))
    }

    /// Builds a cumulative-product [`ScanOperation`] whose nested depth matches `lengths`.
    fn product_scan_with_lengths(lengths: &[usize]) -> ScanOperation<ArrayType, TestArray, TestOperation> {
        assert!(!lengths.is_empty());
        if lengths.len() == 1 {
            return TestScanOperation::new(product_body(), 1, lengths[0]).unwrap();
        }
        let inner_scan = product_scan_with_lengths(&lengths[1..]);
        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestOperation>::new();
        let carry = builder.add_input(ArrayType::scalar(DataType::F64));
        let xs = builder.add_input(f64_type(&lengths[1..]));
        let outputs = builder
            .add_instruction(TestOperation::Scan(Box::new(inner_scan)), vec![carry, xs])
            .unwrap()
            .to_vec();
        let body = builder.build(outputs, vec![Placeholder, Placeholder], vec![Placeholder, Placeholder]).unwrap();
        TestScanOperation::new(body, 1, lengths[0]).unwrap()
    }

    /// Builds the cumulative-product [`ScanOperation`] over three lanes used by the differentiation tests.
    fn product_scan() -> ScanOperation<ArrayType, TestArray, TestOperation> {
        product_scan_with_lengths(&[3])
    }

    #[test]
    fn test_scan_jvp_propagates_tangents_through_linear_scan() {
        // Cumulative product over `xs = [2, 3, 4]` starting at `init = 1`: the final carry is 24 and the running
        // products are `[2, 6, 24]`. A unit tangent on `init` propagates as `d(init * x0 * x1 * x2)/d(init) = 24`
        // on the final carry and `[2, 6, 24]` on the stacked outputs.
        let scan = product_scan();
        let ((carry, ys), (carry_tangent, ys_tangent)) = TestArrayDomain
            .jvp(
                move |(init, xs)| {
                    let mut outputs =
                        init.context().stage_operation(TestOperation::Scan(Box::new(scan)), &[&init, &xs]).unwrap();
                    let ys = outputs.remove(1);
                    (outputs.remove(0), ys)
                },
                (TestArray::scalar(1.0), TestArray::vector(vec![2.0, 3.0, 4.0])),
                (TestArray::scalar(1.0), TestArray::vector(vec![0.0, 0.0, 0.0])),
            )
            .unwrap();
        assert_eq!(carry.values, vec![24.0]);
        assert_eq!(ys.values, vec![2.0, 6.0, 24.0]);
        assert_eq!(carry_tangent.values, vec![24.0]);
        assert_eq!(ys_tangent.values, vec![2.0, 6.0, 24.0]);

        // A unit tangent on `xs[1]` propagates as `d(init * x0 * x1 * x2)/d(x1) = init * x0 * x2 = 8` on the final
        // carry and `[0, 2, 8]` on the stacked outputs (`y0` does not depend on `x1`).
        let scan = product_scan();
        let ((carry, _), (carry_tangent, ys_tangent)) = TestArrayDomain
            .jvp(
                move |(init, xs)| {
                    let mut outputs =
                        init.context().stage_operation(TestOperation::Scan(Box::new(scan)), &[&init, &xs]).unwrap();
                    let ys = outputs.remove(1);
                    (outputs.remove(0), ys)
                },
                (TestArray::scalar(1.0), TestArray::vector(vec![2.0, 3.0, 4.0])),
                (TestArray::scalar(0.0), TestArray::vector(vec![0.0, 1.0, 0.0])),
            )
            .unwrap();
        assert_eq!(carry.values, vec![24.0]);
        assert_eq!(carry_tangent.values, vec![8.0]);
        assert_eq!(ys_tangent.values, vec![0.0, 2.0, 8.0]);
    }

    #[test]
    fn test_scan_jvp_supports_nested_scans_in_linear_scan_bodies() {
        // Nested scans differentiate by recursively replaying the inner linear scan inside each outer scan lane. The
        // final carry is the product of every element, and a unit tangent on the initial carry follows the same
        // cumulative-product path through both scan levels.
        let scan = product_scan_with_lengths(&[2, 3]);
        let ((carry, ys), (carry_tangent, ys_tangent)) = TestArrayDomain
            .jvp(
                move |(init, xs)| {
                    let mut outputs =
                        init.context().stage_operation(TestOperation::Scan(Box::new(scan)), &[&init, &xs]).unwrap();
                    let ys = outputs.remove(1);
                    (outputs.remove(0), ys)
                },
                (TestArray::scalar(1.0), TestArray::matrix(2, 3, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0])),
                (TestArray::scalar(1.0), TestArray::matrix(2, 3, vec![0.0; 6])),
            )
            .unwrap();
        assert_eq!(carry.values, vec![5040.0]);
        assert_eq!(ys.values, vec![2.0, 6.0, 24.0, 120.0, 720.0, 5040.0]);
        assert_eq!(carry_tangent.values, vec![5040.0]);
        assert_eq!(ys_tangent.values, vec![2.0, 6.0, 24.0, 120.0, 720.0, 5040.0]);
    }

    #[test]
    fn test_scan_jvp_supports_three_nested_scans_in_linear_scan_bodies() {
        // Three levels catches the recursive fixed point that failed for nested scan bodies: the middle scan's
        // linear body contains another scan whose body also has scan-local residual references.
        let scan = product_scan_with_lengths(&[2, 2, 2]);
        let xs_type = f64_type(&[2, 2, 2]);
        let ((carry, ys), (carry_tangent, ys_tangent)) = TestArrayDomain
            .jvp(
                move |(init, xs)| {
                    let mut outputs =
                        init.context().stage_operation(TestOperation::Scan(Box::new(scan)), &[&init, &xs]).unwrap();
                    let ys = outputs.remove(1);
                    (outputs.remove(0), ys)
                },
                (TestArray::scalar(1.0), TestArray::new(xs_type.clone(), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])),
                (TestArray::scalar(1.0), TestArray::new(xs_type, vec![0.0; 8])),
            )
            .unwrap();
        assert_eq!(carry.values, vec![362880.0]);
        assert_eq!(ys.values, vec![2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0]);
        assert_eq!(carry_tangent.values, vec![362880.0]);
        assert_eq!(ys_tangent.values, vec![2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0]);
    }

    #[test]
    fn test_linear_scan_interpretation_supports_nested_while_body() {
        // The instantiated linear scan body interpreter must recursively replay nested linear while programs rather
        // than rejecting them. This loop's predicate is false, so each scan lane forwards the carry and returns the
        // current input slice as the stacked output.
        type DirectLinearOperation = LinearArrayOperation<TestArray, TestArray, TestArray, ArrayOperation<TestArray>>;
        type ScanBodyOperation =
            LinearArrayOperation<TestArray, TestArray, ValueOrCapture<ArrayType, TestArray>, ArrayOperation<TestArray>>;

        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<ArrayType, TestArray, ScanBodyOperation>::new();
        condition_builder.add_input(scalar_f64.clone());
        let predicate = condition_builder
            .add_instruction(ZeroOperation::new(ArrayType::scalar(DataType::Boolean)), vec![])
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut while_body_builder = ProgramBuilder::<ArrayType, TestArray, ScanBodyOperation>::new();
        let state = while_body_builder.add_input(scalar_f64.clone());
        let while_body = while_body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![state], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let while_operation = WhileOperation::new(condition, while_body).unwrap();

        let mut scan_body_builder = ProgramBuilder::<ArrayType, TestArray, ScanBodyOperation>::new();
        let carry = scan_body_builder.add_input(scalar_f64);
        let x = scan_body_builder.add_input(ArrayType::scalar(DataType::F64));
        let forwarded_carry = scan_body_builder
            .add_instruction(ScanBodyOperation::While(Box::new(while_operation)), vec![carry])
            .unwrap()[0];
        let scan_body = scan_body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(
                vec![forwarded_carry, x],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let linear_scan = DirectLinearOperation::Scan(Box::new(
            ScanOperation::<ArrayType, TestArray, ScanBodyOperation>::new(scan_body, 1, 3).unwrap(),
        ));
        let context = EagerContext::<ArrayType, TestArray, ConstantOperation<ArrayType, TestArray>>::new();
        let outputs = linear_scan
            .interpret(&context, &[TestArray::scalar(1.0), TestArray::vector(vec![2.0, 3.0, 4.0])])
            .unwrap();
        assert_eq!(outputs[0].values, vec![1.0]);
        assert_eq!(outputs[1].values, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_scan_jvp_stages_residual_extended_scan_under_abstract_tracing() {
        // The scan JVP rule is uniform across domains: under abstract tracing (a tracer-valued differentiation
        // context) it stages exactly one residual-extended primal `scan` — whose body gains one extra stacked
        // output per pushforward residual — and exactly one linear `scan` carrying the registered stacks as
        // factors, with no concretization anywhere. The differentiated scan carries a lowering-only unroll factor,
        // which both staged scans must inherit.
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let stacked_f64 = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let scan = product_scan().with_unroll(3).unwrap();

        let outer_builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestArray, TestOperation>::new()));
        let init_input = outer_builder.borrow_mut().add_input(scalar_f64.clone());
        let xs_input = outer_builder.borrow_mut().add_input(stacked_f64.clone());
        let outer_context = TracingContext::new(&TestArrayDomain, outer_builder.clone());
        let primal_init = outer_context.tracer(init_input, None);
        let primal_xs = outer_context.tracer(xs_input, None);

        let linear_builder = Rc::new(RefCell::new(ProgramBuilder::<
            ArrayType,
            DomainTracer<TestArrayDomain>,
            LinearArrayOperation<
                DomainTracer<TestArrayDomain>,
                TestArray,
                ValueOrCapture<ArrayType, DomainTracer<TestArrayDomain>>,
                TestOperation,
            >,
        >::new()));
        let mut context = TangentContext::new(&outer_context, linear_builder.clone());
        let init_tangent = context.input(scalar_f64.clone());
        let xs_tangent = context.input(stacked_f64.clone());

        let outputs = scan
            .jvp(
                &mut context,
                &[JvpTracer::from_value(primal_init, init_tangent), JvpTracer::from_value(primal_xs, xs_tangent)],
            )
            .expect("the scan JVP rule should stage scan structure under abstract tracing");
        assert_eq!(outputs.len(), 2);

        // The primal trace gained exactly one residual-extended scan: the product body captures both operand
        // primals, so the extended body has two extra stacked outputs.
        let outer_builder = outer_builder.borrow();
        assert_eq!(outer_builder.instructions().len(), 1);
        let staged_primal = outer_builder.instructions()[0].operation();
        let ArrayOperation::Scan(staged_scan) = staged_primal else {
            panic!("expected the staged primal operation to be a scan");
        };
        assert_eq!(staged_scan.carry_count(), 1);
        assert_eq!(staged_scan.length(), 3);
        assert!(!staged_scan.reverse());
        assert_eq!(staged_scan.unroll(), 3);
        assert_eq!(staged_scan.body().input_types(), vec![scalar_f64.clone(), scalar_f64.clone()]);
        assert_eq!(staged_scan.body().output_types().len(), 4);

        // The linear trace gained exactly one linear scan carrying the two registered residual stacks and the
        // inherited unroll factor, which also shows up in its rendered form.
        let linear_builder = linear_builder.borrow();
        assert_eq!(linear_builder.instructions().len(), 1);
        let staged_linear = linear_builder.instructions()[0].operation();
        let LinearArrayOperation::Scan(operation) = staged_linear else {
            panic!("expected the staged linear operation to be a scan");
        };
        assert_eq!(operation.captures().len(), 2);
        assert_eq!(operation.carry_count(), 1);
        assert_eq!(operation.length(), 3);
        assert!(!operation.reverse());
        assert_eq!(operation.unroll(), 3);
        assert!(staged_linear.to_string().contains("unroll=3"), "{staged_linear}");
    }

    #[test]
    fn test_scan_jvp_skips_linear_scan_when_all_input_tangents_are_zero() {
        // Canonical zero input tangents make the entire scan tangent structurally zero. The JVP rule should still
        // stage the ordinary primal scan, but it should avoid the residual-extended primal body and the linear scan.
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let stacked_f64 = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let scan = product_scan();

        let outer_builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestArray, TestOperation>::new()));
        let init_input = outer_builder.borrow_mut().add_input(scalar_f64.clone());
        let xs_input = outer_builder.borrow_mut().add_input(stacked_f64.clone());
        let outer_context = TracingContext::new(&TestArrayDomain, outer_builder.clone());
        let primal_init = outer_context.tracer(init_input, None);
        let primal_xs = outer_context.tracer(xs_input, None);

        let linear_builder = Rc::new(RefCell::new(ProgramBuilder::<
            ArrayType,
            DomainTracer<TestArrayDomain>,
            LinearArrayOperation<
                DomainTracer<TestArrayDomain>,
                TestArray,
                ValueOrCapture<ArrayType, DomainTracer<TestArrayDomain>>,
                TestOperation,
            >,
        >::new()));
        let mut context = TangentContext::new(&outer_context, linear_builder.clone());
        let mut init_zero = context.stage_nullary_operation(ZeroOperation::new(scalar_f64.clone())).unwrap();
        let mut xs_zero = context.stage_nullary_operation(ZeroOperation::new(stacked_f64.clone())).unwrap();

        let outputs = scan
            .jvp(
                &mut context,
                &[
                    JvpTracer::from_value(primal_init, init_zero.remove(0)),
                    JvpTracer::from_value(primal_xs, xs_zero.remove(0)),
                ],
            )
            .expect("zero tangent scan JVP should stage only primal scan plus zero tangent outputs");
        assert_eq!(outputs.len(), 2);

        let outer_builder = outer_builder.borrow();
        assert_eq!(outer_builder.instructions().len(), 1);
        let ArrayOperation::Scan(staged_scan) = outer_builder.instructions()[0].operation() else {
            panic!("expected the staged primal operation to be an ordinary scan");
        };
        assert_eq!(staged_scan.body().output_types().len(), 2);

        let linear_builder = linear_builder.borrow();
        assert_eq!(linear_builder.instructions().len(), 4);
        assert!(
            linear_builder
                .instructions()
                .iter()
                .all(|instruction| matches!(instruction.operation(), LinearArrayOperation::Zero(_)))
        );
    }

    #[test]
    fn test_scan_transpose_round_trip_flips_reverse_back() {
        // Transposing a linear scan flips `reverse` and transposes the body, so transposing twice restores the
        // original direction and the original linear map. The body `carry' = r[lane] * carry + x` references its
        // residual stack scan-locally, which both transposes carry verbatim.
        type DirectLinearOperation = LinearArrayOperation<TestArray, TestArray, TestArray, ArrayOperation<TestArray>>;
        type ScanBodyOperation =
            LinearArrayOperation<TestArray, TestArray, ValueOrCapture<ArrayType, TestArray>, ArrayOperation<TestArray>>;

        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let stacked_f64 = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let mut body_builder = ProgramBuilder::<ArrayType, TestArray, ScanBodyOperation>::new();
        let tangent_carry = body_builder.add_input(scalar_f64.clone());
        let tangent_x = body_builder.add_input(scalar_f64.clone());
        let scaled = body_builder
            .add_instruction(
                ScaleOperation::<ArrayType, ValueOrCapture<ArrayType, TestArray>, Input>::new(
                    ValueOrCapture::Capture { index: 0, r#type: scalar_f64.clone() },
                ),
                vec![tangent_carry],
            )
            .unwrap()[0];
        let summed = body_builder.add_instruction(AddOperation, vec![scaled, tangent_x]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(
                vec![summed, summed],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<ArrayType, TestArray, DirectLinearOperation>::new();
        let tangent_init = builder.add_input(scalar_f64);
        let tangent_xs = builder.add_input(stacked_f64);
        let linear_scan = ScanOperation::<ArrayType, TestArray, ScanBodyOperation>::new(body, 1, 3)
            .unwrap()
            .with_unroll(3)
            .unwrap()
            .with_captures(vec![TestArray::vector(vec![2.0, 3.0, 4.0])]);
        let outputs = builder
            .add_instruction(DirectLinearOperation::Scan(Box::new(linear_scan)), vec![tangent_init, tangent_xs])
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(
                outputs,
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();

        // One transposition flips the direction; a second one flips it back. The lowering-only unroll factor is
        // preserved verbatim by every transposition.
        let transposed = program.transpose().unwrap();
        assert!(transposed.to_string().contains("reverse=true"), "{transposed}");
        assert!(transposed.to_string().contains("unroll=3"), "{transposed}");
        let round_tripped = transposed.transpose().unwrap();
        assert!(round_tripped.to_string().contains("reverse=false"), "{round_tripped}");
        assert!(round_tripped.to_string().contains("unroll=3"), "{round_tripped}");

        // The double transpose is the original linear map: forward over `(t_init, t_xs) = (1, [1, 1, 1])` with
        // `r = [2, 3, 4]` computes carries `c0 = 2 * 1 + 1 = 3`, `c1 = 3 * 3 + 1 = 10`, `c2 = 4 * 10 + 1 = 41`.
        let inputs = vec![TestArray::scalar(1.0), TestArray::vector(vec![1.0, 1.0, 1.0])];
        let expected = program.interpret(inputs.clone()).unwrap();
        assert_eq!(expected[0].values, vec![41.0]);
        assert_eq!(expected[1].values, vec![3.0, 10.0, 41.0]);
        let round_tripped_outputs = round_tripped.interpret(inputs).unwrap();
        assert_eq!(round_tripped_outputs[0].values, expected[0].values);
        assert_eq!(round_tripped_outputs[1].values, expected[1].values);
    }

    #[test]
    fn test_scan_batching_lifts_batched_carries() {
        // Batching a scan whose carry is mapped at axis 0 threads the lane axis through every iteration: each
        // batch lane runs its own cumulative product over the shared `xs = [2, 3, 4]`, and the stacked outputs
        // gain the scan axis in front of the lane axis.
        let scan = product_scan();
        let context = TestEagerContext::new();
        let carries = ArrayBatch::mapped(TestArray::vector(vec![1.0, 2.0, 3.0]), 0).unwrap();
        let stacked_inputs = ArrayBatch::unbatched(TestArray::vector(vec![2.0, 3.0, 4.0]));
        let outputs = scan.batch(&context, &[carries, stacked_inputs]).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].value().values, vec![24.0, 48.0, 72.0]);
        assert_eq!(outputs[1].batch_axis(), Some(1));
        assert_eq!(outputs[1].r#type().shape().dimensions(), &[Size::Static(3), Size::Static(3)]);
        assert_eq!(outputs[1].value().values, vec![2.0, 4.0, 6.0, 6.0, 12.0, 18.0, 24.0, 48.0, 72.0]);
    }

    #[test]
    fn test_scan_batching_lifts_batched_stacked_inputs() {
        // Batching a scan whose stacked input is mapped at axis 0 reads each iteration's slice along the logical
        // leading axis (physical axis 1 when the batch axis sits at 0), so every batch lane scans its own row.
        let scan = product_scan();
        let context = TestEagerContext::new();
        let carries = ArrayBatch::unbatched(TestArray::scalar(1.0));
        let stacked_inputs =
            ArrayBatch::mapped(TestArray::matrix(2, 3, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), 0).unwrap();
        let outputs = scan.batch(&context, &[carries, stacked_inputs]).unwrap();
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].value().values, vec![24.0, 210.0]);
        assert_eq!(outputs[1].batch_axis(), Some(1));
        assert_eq!(outputs[1].r#type().shape().dimensions(), &[Size::Static(3), Size::Static(2)]);
        assert_eq!(outputs[1].value().values, vec![2.0, 5.0, 6.0, 30.0, 24.0, 210.0]);

        // A trailing batch axis (physical `[3, 2]` with the lane axis at 1) reads the same logical lanes, so the
        // outputs are identical.
        let scan = product_scan();
        let context = TestEagerContext::new();
        let carries = ArrayBatch::unbatched(TestArray::scalar(1.0));
        let stacked_inputs =
            ArrayBatch::mapped(TestArray::matrix(3, 2, vec![2.0, 5.0, 3.0, 6.0, 4.0, 7.0]), 1).unwrap();
        let outputs = scan.batch(&context, &[carries, stacked_inputs]).unwrap();
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].value().values, vec![24.0, 210.0]);
        assert_eq!(outputs[1].batch_axis(), Some(1));
        assert_eq!(outputs[1].value().values, vec![2.0, 5.0, 6.0, 30.0, 24.0, 210.0]);
    }

    #[test]
    fn test_scan_batching_threads_batched_carries_and_inputs() {
        // Batching both operands pairs batch lane `i` of the carries with batch lane `i` of the stacked inputs.
        let scan = product_scan();
        let context = TestEagerContext::new();
        let carries = ArrayBatch::mapped(TestArray::vector(vec![1.0, 10.0]), 0).unwrap();
        let stacked_inputs =
            ArrayBatch::mapped(TestArray::matrix(2, 3, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), 0).unwrap();
        let outputs = scan.batch(&context, &[carries, stacked_inputs]).unwrap();
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].value().values, vec![24.0, 2100.0]);
        assert_eq!(outputs[1].batch_axis(), Some(1));
        assert_eq!(outputs[1].value().values, vec![2.0, 50.0, 6.0, 300.0, 24.0, 2100.0]);
    }

    #[test]
    fn test_scan_batching_respects_reverse_visit_order() {
        // A reversed batched scan visits the logical lanes from the back while keeping output lane `i` aligned
        // with input lane `i`: the reversed cumulative product over `[2, 3, 4]` is `[24, 12, 4]` per batch lane.
        let scan = product_scan().with_reverse(true);
        let context = TestEagerContext::new();
        let carries = ArrayBatch::unbatched(TestArray::scalar(1.0));
        let stacked_inputs =
            ArrayBatch::mapped(TestArray::matrix(2, 3, vec![2.0, 3.0, 4.0, 2.0, 3.0, 4.0]), 0).unwrap();
        let outputs = scan.batch(&context, &[carries, stacked_inputs]).unwrap();
        assert_eq!(outputs[0].value().values, vec![24.0, 24.0]);
        assert_eq!(outputs[1].batch_axis(), Some(1));
        assert_eq!(outputs[1].value().values, vec![24.0, 24.0, 12.0, 12.0, 4.0, 4.0]);
    }

    #[test]
    fn test_scan_defactorize_moves_reference_stacks_into_operands() {
        // Direct coverage for the scan defactorization arm with mixed stacks: a linear scan whose residual stacks
        // mix a loop-varying reference (moved into operand position as an extra scanned input) and a closed
        // constant (kept as a factor payload) rewrites its body so references to the moved stack become recomputed
        // operand-form products against a new trailing lane input, while references to the surviving constant stack
        // are re-indexed against the compacted constant-only stack list.
        use crate::tracing_v2::{DefactorizedOperation, ResidualizedOperation, SupportsLinearWhile};

        type ScanBodyOperation =
            LinearArrayOperation<TestArray, TestArray, ValueOrCapture<ArrayType, TestArray>, ArrayOperation<TestArray>>;

        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let stacked_f64 = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]));

        // Body: `carry' = kept[lane] * (moved[lane] * carry)`, with scan-local reference 0 pointing at the moved
        // stack and scan-local reference 1 pointing at the kept constant stack.
        let mut body_builder = ProgramBuilder::<ArrayType, TestArray, ScanBodyOperation>::new();
        let tangent_carry = body_builder.add_input(scalar_f64.clone());
        let scaled_by_moved = body_builder
            .add_instruction(
                ScaleOperation::<ArrayType, ValueOrCapture<ArrayType, TestArray>, Input>::new(
                    ValueOrCapture::Capture { index: 0, r#type: scalar_f64.clone() },
                ),
                vec![tangent_carry],
            )
            .unwrap()[0];
        let scaled_by_kept = body_builder
            .add_instruction(
                ScaleOperation::<ArrayType, ValueOrCapture<ArrayType, TestArray>, Input>::new(
                    ValueOrCapture::Capture { index: 1, r#type: scalar_f64.clone() },
                ),
                vec![scaled_by_moved],
            )
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![scaled_by_kept], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // The enclosing fused while body carries the tangent input and the recomputed moved-stack atom.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, ScanBodyOperation>::new();
        let tangent_input = builder.add_input(scalar_f64.clone());
        let moved_stack_atom = builder.add_input(stacked_f64.clone());
        let scan_operation = ScanOperation::<ArrayType, TestArray, ScanBodyOperation>::new(body, 1, 2)
            .unwrap()
            .with_unroll(2)
            .unwrap()
            .with_captures(vec![
                ValueOrCapture::Capture { index: 0, r#type: stacked_f64 },
                ValueOrCapture::Value(TestArray::vector(vec![5.0, 7.0])),
            ]);
        let scan = ScanBodyOperation::Scan(Box::new(scan_operation));
        let DefactorizedOperation::Operation { operation, inputs } =
            scan.defactorize(&[moved_stack_atom], vec![tangent_input]).unwrap()
        else {
            panic!("expected the scan defactorization to produce an operand-form scan");
        };

        // The moved stack was appended as an extra scanned operand and only the constant stack survived as a
        // payload; the body gained one trailing lane input, its reference to the moved stack became a recomputed
        // operand-form product, and its reference to the kept stack was re-indexed to compacted position 0. The
        // lowering-only unroll factor is preserved by the rewrite.
        assert_eq!(inputs, vec![tangent_input, moved_stack_atom]);
        let ScanBodyOperation::Scan(scan_operation) = &operation else {
            panic!("expected the rewritten operation to stay a scan");
        };
        let rewritten_body = scan_operation.body();
        assert_eq!(scan_operation.unroll(), 2);
        assert_eq!(scan_operation.captures().len(), 1);
        assert!(matches!(scan_operation.captures()[0], ValueOrCapture::Value(_)));
        assert_eq!(rewritten_body.input_types(), vec![scalar_f64.clone(), scalar_f64]);
        assert!(
            rewritten_body
                .instructions()
                .iter()
                .any(|instruction| matches!(instruction.operation(), ScanBodyOperation::Recompute(_))),
            "{rewritten_body}",
        );
        assert!(
            rewritten_body.instructions().iter().any(|instruction| matches!(
                instruction.operation(),
                ScanBodyOperation::Scale(scale) if matches!(
                    scale.factor(),
                    ValueOrCapture::Capture { index: 0, .. },
                ),
            )),
            "{rewritten_body}",
        );

        // Interpreting the rewritten scan over `moved = [2, 3]` and `kept = [5, 7]` matches the factor-form
        // semantics: `c1 = 5 * (2 * 1) = 10` and `c2 = 7 * (3 * 10) = 210`.
        let outputs = builder.add_instruction(operation, inputs).unwrap().to_vec();
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(outputs, vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        let direct_program = program
            .map_operations(|operation| ResidualizedOperation::<TestArrayDomain>::instantiate_residuals(operation, &[]))
            .unwrap();
        let outputs =
            direct_program.interpret(vec![TestArray::scalar(1.0), TestArray::vector(vec![2.0, 3.0])]).unwrap();
        assert_eq!(outputs[0].values, vec![210.0]);
    }

    #[test]
    fn test_linear_scan_batching_binds_residual_lanes_per_iteration() {
        // Batching the staged linear scan binds each iteration's body against that lane's residual slices (the
        // stacks are lane-uniform across *batch* lanes but vary across *scan* lanes) and reuses the shared scan
        // batching loop: with `r = [2, 3, 4]` and body `carry' = r[lane] * carry + x`, batched tangent carries
        // `[1, 2]` produce final carries `[c2, 2 * 24 + ...]` computed per batch lane.
        type DirectLinearOperation = LinearArrayOperation<TestArray, TestArray, TestArray, ArrayOperation<TestArray>>;
        type ScanBodyOperation =
            LinearArrayOperation<TestArray, TestArray, ValueOrCapture<ArrayType, TestArray>, ArrayOperation<TestArray>>;

        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut body_builder = ProgramBuilder::<ArrayType, TestArray, ScanBodyOperation>::new();
        let tangent_carry = body_builder.add_input(scalar_f64.clone());
        let tangent_x = body_builder.add_input(scalar_f64.clone());
        let scaled = body_builder
            .add_instruction(
                ScaleOperation::<ArrayType, ValueOrCapture<ArrayType, TestArray>, Input>::new(
                    ValueOrCapture::Capture { index: 0, r#type: scalar_f64.clone() },
                ),
                vec![tangent_carry],
            )
            .unwrap()[0];
        let summed = body_builder.add_instruction(AddOperation, vec![scaled, tangent_x]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![summed], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        let scan = ScanOperation::<ArrayType, TestArray, ScanBodyOperation>::new(body, 1, 3)
            .unwrap()
            .with_captures(vec![TestArray::vector(vec![2.0, 3.0, 4.0])]);
        let linear_scan = DirectLinearOperation::Scan(Box::new(scan));

        // Per batch lane: `c0 = 2 t + 1`, `c1 = 3 c0 + 1`, `c2 = 4 c1 + 1` so `c2 = 24 t + 17`.
        let tangent_carries = ArrayBatch::mapped(TestArray::vector(vec![1.0, 2.0]), 0).unwrap();
        let tangent_inputs = ArrayBatch::unbatched(TestArray::vector(vec![1.0, 1.0, 1.0]));
        let context = EagerContext::<ArrayType, TestArray, DirectLinearOperation>::new();
        let outputs = linear_scan.batch(&context, &[tangent_carries, tangent_inputs]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].value().values, vec![41.0, 65.0]);
    }
}
