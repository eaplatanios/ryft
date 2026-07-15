use std::fmt::{Debug, Display};

use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, check_types};
use crate::operations::constants::Zero;
use crate::operations::manipulation::{Reshape, Slice, UpdateSlice};
use crate::parameters::Placeholder;
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationInput, PartialEvaluationOutput,
    PartialEvaluationValue, PartialValue, PartiallyEvaluatableOperation, PartitionedProgram,
};
use crate::programs::ProgramError;
use crate::programs::builders::ProgramBuilder;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::{OutputRegionProvenance, RegionInterface, RegionRef};
use crate::programs::types::{Type, TypeError};
use crate::programs::values::Value;
use crate::types::{ArrayType, DataType, Shape, Size};

// TODO(eaplatanios): Review from here onwards.

/// Canonical operation name for [`ScanOperation`].
pub const SCAN_OPERATION_NAME: &str = "scan";

/// [`Operation`] that applies a nested body [`Program`](crate::Program) a static number of times over a loop-carried
/// state while
/// consuming one slice of each stacked input per iteration and stacking the body's per-iteration outputs. This is the
/// statically shaped loop primitive analogous to JAX's
/// [`lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html): where [`WhileOperation`] iterates a
/// data-dependent number of times and therefore supports reverse-mode differentiation only by unrolling in eager
/// domains, `scan` has a static trip count, so its linearization can *store* per-iteration residuals as statically
/// shaped stacks and its linear form transposes totally (see the `tracing_v2` scan rules).
///
/// The body [`Program`](crate::Program) maps `[carry..., x_slice...]` to `[carry..., y_slice...]`: the first
/// [`carry_count`](Self::carry_count) inputs and outputs are the loop-carried state (with identical type signatures),
/// the remaining inputs consume one slice of each stacked input per iteration, and the remaining outputs produce one
/// slice of each stacked output per iteration. The operation's inputs are `[carry..., stacked_xs...]` and its outputs
/// are `[final_carry..., stacked_ys...]`, where each stacked type prepends a static
/// [`length`](Self::length) dimension to the corresponding body slice type. Iteration `i` consumes slice `i` of every
/// stacked input and produces slice `i` of every stacked output; when [`reverse`](Self::reverse) is `true` the
/// iterations visit the slices from `length - 1` down to `0`, but slice `i` of every stacked output still corresponds
/// to slice `i` of the stacked inputs (the visit *order* reverses, the slice *pairing* does not). Transposition of
/// the linear form simply flips [`reverse`](Self::reverse), so no array-reversal operation is ever needed.
///
/// The `length` is stored explicitly so that scans without stacked inputs (pure carry loops with stacked outputs)
/// remain well-defined. All body slice types must be fully static because stacking prepends a static `length`
/// dimension and per-iteration slice extraction must be provably in bounds.
///
/// The optional [`unroll`](Self::unroll) factor (attached via [`with_unroll`](Self::with_unroll)) is a
/// **lowering-only** attribute: interpretation and every transform rule (differentiation, transposition, batching)
/// ignore it semantically but preserve it on whatever scan they re-stage, while lowerings emit `unroll` body copies
/// per loop trip — and a fully unrolled straight-line lowering with no loop at all when `unroll` equals `length`.
///
/// The body computation is not part of this payload: it is a [`Region`](crate::Region) attached to the
/// [`Instruction`](crate::Instruction) applying the operation (the single [`region_names`](Operation::region_names)
/// slot `["body"]`), and semantic rules reach it through their driver-granted region access. Scans with owned bodies
/// supply the body [`Program`](crate::Program) through the `regions` argument of [`Context::bind`];
/// [`Operation::infer_output_types`] validates the body signature over the attached [`RegionInterface`].
///
/// [`WhileOperation`]: crate::operations::control_flow::WhileOperation
#[derive(Clone, Debug)]
pub struct ScanOperation<Capture: Value> {
    /// Captured values used by the body operation payloads.
    ///
    /// Ordinary primal scans leave this empty. Linearized scans use it for values captured from the primal program,
    /// such as per-iteration residual stacks. Keeping the captures on the ordinary scan operation avoids a separate
    /// linear scan operation while preserving the fact that captures can have a different value family from the
    /// tangent inputs carried by the scan body.
    pub(crate) captures: Vec<Capture>,

    /// Number of loop-carried state leaves at the front of the body's inputs and outputs.
    pub(crate) carry_count: usize,

    /// Static trip count of this [`ScanOperation`].
    pub(crate) length: usize,

    /// Boolean indicating whether iterations visit the stacked slices in reverse order.
    pub(crate) reverse: bool,

    /// Lowering-only unroll factor: the number of body copies emitted per loop trip (`1` keeps one body per trip).
    pub(crate) unroll: usize,
}

/// Validates that every dimension of `r#type` is static, reporting a precise error that names the scan `role` (for
/// example, `input 1` or `output 0`) when one is not. Scan body slice types must be fully static because stacking
/// prepends a static length dimension and per-iteration slice extraction must be provably in bounds.
fn check_static_scan_type(role: &str, index: usize, r#type: &ArrayType) -> Result<(), TypeError> {
    for (axis, dimension) in r#type.shape().dimensions().iter().enumerate() {
        if dimension.value().is_none() {
            return Err(TypeError {
                message: format!(
                    "scan body {role} {index} must have a fully static type but axis {axis} of {type} has size \
                     {dimension}",
                    r#type = r#type,
                ),
            });
        }
    }
    Ok(())
}

/// Validates a scan unroll factor against the scan's static trip count: the factor must be at least `1` and must
/// evenly divide `length` (remainder handling is an explicit non-goal). This backs [`ScanOperation::with_unroll`] and
/// enum-payload validation of scan variants whose fields are public and therefore cannot rely on builder-time
/// validation alone.
pub(crate) fn validate_scan_unroll(unroll: usize, length: usize) -> Result<(), TypeError> {
    if unroll == 0 {
        return Err(TypeError { message: "scan unroll factor must be at least 1".to_string() });
    }
    if length % unroll != 0 {
        return Err(TypeError {
            message: format!("scan unroll factor {unroll} must evenly divide the scan length {length}"),
        });
    }
    Ok(())
}

/// Returns the stacked variant of a scan body slice type, prepending a static `length` dimension to its shape. The
/// stacked type carries no optional layout or sharding metadata, so it is a declared type whose optional components
/// are unspecified and scan input validation compares it against actual input types with
/// [`Type::is_refined_by`](crate::programs::types::Type::is_refined_by).
pub(crate) fn stacked_scan_type(slice_type: &ArrayType, length: usize) -> ArrayType {
    let mut dimensions = Vec::with_capacity(slice_type.rank() + 1);
    dimensions.push(Size::Static(length));
    dimensions.extend(slice_type.shape().dimensions().iter().cloned());
    ArrayType::new(slice_type.data_type(), Shape::new(dimensions))
}

/// Validates `[carry..., stacked_xs...]` input types against a scan body signature and returns the
/// `[carry..., stacked_ys...]` output types. This backs type inference for [`ScanOperation`].
///
/// The expected input types are declared types derived from the body signature (with stacked types built by
/// [`stacked_scan_type`], which carries no optional metadata), while the provided `input_types` may be actual runtime
/// value types carrying more precise optional metadata, such as the normalized [`Sharding`](crate::Sharding)s that
/// every concrete backend array type carries. Validation therefore uses the directional declared-vs-actual
/// [`Type::is_refined_by`] relation instead of strict type equality. The returned output types are declared types
/// built the same way and thus leave optional metadata unspecified for downstream consumers (e.g., sharding
/// propagation) to resolve.
pub(crate) fn scan_output_types(
    body_input_types: &[ArrayType],
    body_output_types: &[ArrayType],
    carry_count: usize,
    length: usize,
    input_types: &[ArrayType],
) -> Result<Vec<ArrayType>, TypeError> {
    let mut expected_input_types = body_input_types[..carry_count].to_vec();
    expected_input_types
        .extend(body_input_types[carry_count..].iter().map(|slice_type| stacked_scan_type(slice_type, length)));
    check_count!("input", input_types, expected_input_types.len(), TypeError);
    for (index, (expected, actual)) in expected_input_types.iter().zip(input_types).enumerate() {
        if !expected.is_refined_by(actual) {
            return Err(TypeError {
                message: format!(
                    "scan input {index} has type {actual} which is incompatible with the expected type {expected}",
                ),
            });
        }
    }
    let mut output_types = body_output_types[..carry_count].to_vec();
    output_types
        .extend(body_output_types[carry_count..].iter().map(|slice_type| stacked_scan_type(slice_type, length)));
    Ok(output_types)
}

/// Validates a carry-only scalar scan and returns the final carry types.
///
/// `DataType` has no length-indexed stack metadata, so scalar scans can only represent loop-carried state. Scanned
/// scalar inputs and stacked scalar outputs require a separate stack value representation and are rejected here.
pub(crate) fn scalar_scan_output_types(
    body_input_types: &[DataType],
    body_output_types: &[DataType],
    carry_count: usize,
    input_types: &[DataType],
) -> Result<Vec<DataType>, TypeError> {
    if carry_count != body_input_types.len() {
        return Err(TypeError {
            message: format!(
                "scalar scan requires every body input to be loop-carried, but carry count {carry_count} is smaller \
                 than the body input count {}",
                body_input_types.len(),
            ),
        });
    }
    if carry_count != body_output_types.len() {
        return Err(TypeError {
            message: format!(
                "scalar scan requires every body output to be loop-carried, but carry count {carry_count} is smaller \
                 than the body output count {}",
                body_output_types.len(),
            ),
        });
    }
    check_types!("scan body carry", body_input_types, body_output_types);
    check_count!("input", input_types, carry_count, TypeError);
    check_types!("scan input", body_input_types, input_types);
    Ok(body_output_types.to_vec())
}

/// Type-family semantics for [`ScanOperation`].
///
/// [`ArrayType`] can represent scanned values by prepending a static leading axis to each per-iteration value type,
/// while [`DataType`] currently has no stack metadata and therefore supports only carry-only scalar scans. This trait
/// keeps those type rules local to the scan operation so the operation dispatcher itself can be generic over `T`.
pub trait ScanTypeSemantics: Type {
    /// Validates a scan body signature for this type family.
    fn validate_scan_body(
        body_input_types: &[Self],
        body_output_types: &[Self],
        carry_count: usize,
        length: usize,
    ) -> Result<(), TypeError>;

    /// Returns this scan's input types from its body input types.
    fn scan_input_types(body_input_types: &[Self], carry_count: usize, length: usize) -> Vec<Self>;

    /// Returns this scan's declared output types from its body output types.
    fn scan_declared_output_types(body_output_types: &[Self], carry_count: usize, length: usize) -> Vec<Self>;

    /// Infers this scan's output types from concrete input types.
    fn infer_scan_output_types(
        body_input_types: &[Self],
        body_output_types: &[Self],
        carry_count: usize,
        length: usize,
        input_types: &[Self],
    ) -> Result<Vec<Self>, TypeError>;

    /// Validates one capture value stored on this scan.
    fn validate_scan_capture<C: Value<Type = Self>>(capture: &C, index: usize, length: usize) -> Result<(), TypeError>;
}

impl ScanTypeSemantics for ArrayType {
    fn validate_scan_body(
        body_input_types: &[Self],
        body_output_types: &[Self],
        carry_count: usize,
        _length: usize,
    ) -> Result<(), TypeError> {
        if carry_count > body_input_types.len() {
            return Err(TypeError {
                message: format!(
                    "scan carry count {carry_count} exceeds the body input count {}",
                    body_input_types.len(),
                ),
            });
        }
        if carry_count > body_output_types.len() {
            return Err(TypeError {
                message: format!(
                    "scan carry count {carry_count} exceeds the body output count {}",
                    body_output_types.len(),
                ),
            });
        }
        check_types!("scan body carry", &body_input_types[..carry_count], &body_output_types[..carry_count]);
        for (index, input_type) in body_input_types.iter().enumerate() {
            check_static_scan_type("input", index, input_type)?;
        }
        for (index, output_type) in body_output_types.iter().enumerate() {
            check_static_scan_type("output", index, output_type)?;
        }
        Ok(())
    }

    fn scan_input_types(body_input_types: &[Self], carry_count: usize, length: usize) -> Vec<Self> {
        let mut input_types = body_input_types[..carry_count].to_vec();
        input_types
            .extend(body_input_types[carry_count..].iter().map(|slice_type| stacked_scan_type(slice_type, length)));
        input_types
    }

    fn scan_declared_output_types(body_output_types: &[Self], carry_count: usize, length: usize) -> Vec<Self> {
        let mut output_types = body_output_types[..carry_count].to_vec();
        output_types
            .extend(body_output_types[carry_count..].iter().map(|slice_type| stacked_scan_type(slice_type, length)));
        output_types
    }

    fn infer_scan_output_types(
        body_input_types: &[Self],
        body_output_types: &[Self],
        carry_count: usize,
        length: usize,
        input_types: &[Self],
    ) -> Result<Vec<Self>, TypeError> {
        scan_output_types(body_input_types, body_output_types, carry_count, length, input_types)
    }

    fn validate_scan_capture<C: Value<Type = Self>>(capture: &C, index: usize, length: usize) -> Result<(), TypeError> {
        let capture_type = capture.r#type();
        if capture_type.rank() == 0 || capture_type.dimension(0) != Size::Static(length) {
            return Err(TypeError {
                message: format!(
                    "scan capture {index} must have leading dimension {length} but has type {capture_type}",
                    capture_type = capture_type.as_ref(),
                ),
            });
        }
        Ok(())
    }
}

impl ScanTypeSemantics for DataType {
    fn validate_scan_body(
        body_input_types: &[Self],
        body_output_types: &[Self],
        carry_count: usize,
        _length: usize,
    ) -> Result<(), TypeError> {
        scalar_scan_output_types(body_input_types, body_output_types, carry_count, body_input_types)?;
        Ok(())
    }

    fn scan_input_types(body_input_types: &[Self], _carry_count: usize, _length: usize) -> Vec<Self> {
        body_input_types.to_vec()
    }

    fn scan_declared_output_types(body_output_types: &[Self], _carry_count: usize, _length: usize) -> Vec<Self> {
        body_output_types.to_vec()
    }

    fn infer_scan_output_types(
        body_input_types: &[Self],
        body_output_types: &[Self],
        carry_count: usize,
        _length: usize,
        input_types: &[Self],
    ) -> Result<Vec<Self>, TypeError> {
        scalar_scan_output_types(body_input_types, body_output_types, carry_count, input_types)
    }

    fn validate_scan_capture<C: Value<Type = Self>>(
        _capture: &C,
        _index: usize,
        _length: usize,
    ) -> Result<(), TypeError> {
        Err(TypeError { message: "scalar scan captures require a scalar stack representation".to_string() })
    }
}

impl<Capture: Value> ScanOperation<Capture> {
    /// Creates a new [`ScanOperation`] with the provided carry count and static trip count, visiting iterations in
    /// increasing order (use [`Self::with_reverse`] to flip the visit order). The body [`Program`](crate::Program)
    /// mapping
    /// `[carry..., x_slice...]` to `[carry..., y_slice...]` is supplied separately as the operation's attached
    /// region (via the `regions` argument of [`Context::bind`]);
    /// [`Operation::infer_output_types`] validates its signature against `carry_count` and `length`.
    ///
    /// # Parameters
    ///
    ///   - `carry_count`: Number of loop-carried state leaves at the front of the body's inputs and outputs.
    ///   - `length`: Static trip count.
    #[inline]
    pub fn new(carry_count: usize, length: usize) -> Self {
        Self { captures: Vec::new(), carry_count, length, reverse: false, unroll: 1 }
    }
    /// Returns this [`ScanOperation`] with the slice visit order set to `reverse`. For carry-only scalar scans this
    /// only preserves lowering metadata because all iterations consume and produce loop-carried state.
    #[inline]
    pub fn with_reverse(mut self, reverse: bool) -> Self {
        self.reverse = reverse;
        self
    }

    /// Returns this [`ScanOperation`] with the lowering unroll factor set to `unroll`. The factor must be at least
    /// `1` and must evenly divide [`length`](Self::length) (remainder handling is an explicit non-goal). Unrolling
    /// is lowering-only: interpretation and transform rules ignore the factor but preserve it, while lowerings emit
    /// `unroll` body copies per loop trip — and a fully unrolled straight-line lowering with no loop at all when
    /// `unroll` equals [`length`](Self::length).
    pub fn with_unroll(mut self, unroll: usize) -> Result<Self, ProgramError> {
        validate_scan_unroll(unroll, self.length)?;
        self.unroll = unroll;
        Ok(self)
    }

    /// Returns this [`ScanOperation`] with the provided capture environment.
    ///
    /// Captures are interpreted by operation payloads inside the body. The scan operation itself only stores and
    /// preserves them; linear interpretation, transposition, and batching rules decide how to instantiate each
    /// captured value for an iteration.
    #[inline]
    pub fn with_captures<MappedCapture: Value>(self, captures: Vec<MappedCapture>) -> ScanOperation<MappedCapture> {
        ScanOperation {
            captures,
            carry_count: self.carry_count,
            length: self.length,
            reverse: self.reverse,
            unroll: self.unroll,
        }
    }

    /// Returns the capture environment used by this [`ScanOperation`]'s body payloads.
    #[inline]
    pub fn captures(&self) -> &[Capture] {
        self.captures.as_slice()
    }

    /// Returns the number of loop-carried state leaves of this [`ScanOperation`].
    #[inline]
    pub fn carry_count(&self) -> usize {
        self.carry_count
    }

    /// Returns the static trip count of this [`ScanOperation`].
    #[inline]
    pub fn length(&self) -> usize {
        self.length
    }

    /// Returns `true` when iterations of this [`ScanOperation`] visit the stacked slices in reverse order.
    #[inline]
    pub fn reverse(&self) -> bool {
        self.reverse
    }

    /// Returns the lowering-only unroll factor of this [`ScanOperation`] (the number of body copies emitted per
    /// loop trip; `1` when no unrolling was requested via [`with_unroll`](Self::with_unroll)).
    #[inline]
    pub fn unroll(&self) -> usize {
        self.unroll
    }
}

impl<Capture: Value<Type: ScanTypeSemantics>> Display for ScanOperation<Capture> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Operation::<Capture::Type>::render(self, formatter, 0)
    }
}

fn render_capture_list<C: Display>(captures: &[C]) -> String {
    let mut rendered = String::from("[");
    for (index, capture) in captures.iter().enumerate() {
        if index > 0 {
            rendered.push_str(", ");
        }
        rendered.push_str(&capture.to_string());
    }
    rendered.push(']');
    rendered
}

/// Validates the scan contract over the single attached body region interface (the `["body"]` slot) and returns
/// it: the body's first `carry_count` input and output types must agree, every body type must satisfy the type
/// family's scan rules (fully static for [`ArrayType`]; carry-only for [`DataType`]), and the interface is what the
/// scan's boundary types derive from.
fn validated_scan_interface<'i, T: ScanTypeSemantics>(
    region_interfaces: &'i [RegionInterface<T>],
    carry_count: usize,
    length: usize,
) -> Result<&'i RegionInterface<T>, TypeError> {
    if region_interfaces.len() != 1 {
        return Err(TypeError {
            message: format!("scan expects 1 attached region but got {}", region_interfaces.len()),
        });
    }
    let body_interface = &region_interfaces[0];
    T::validate_scan_body(body_interface.input_types(), body_interface.output_types(), carry_count, length)?;
    Ok(body_interface)
}

impl<T, Capture> Operation<T> for ScanOperation<Capture>
where
    T: ScanTypeSemantics,
    Capture: Value<Type = T>,
{
    #[inline]
    fn name(&self) -> &'static str {
        SCAN_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        let body_interface = validated_scan_interface(region_interfaces, self.carry_count, self.length)?;
        let output_types = T::infer_scan_output_types(
            body_interface.input_types(),
            body_interface.output_types(),
            self.carry_count,
            self.length,
            input_types,
        )?;
        for (index, capture) in self.captures.iter().enumerate() {
            T::validate_scan_capture(capture, index, self.length)?;
        }
        Ok(output_types)
    }

    #[inline]
    fn region_names(&self) -> &'static [&'static str] {
        &["body"]
    }

    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        vec![OutputRegionProvenance { region_index: 0, output_index }]
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, SCAN_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("carry_count", self.carry_count)?;
            operation.field("length", self.length)?;
            operation.field("reverse", self.reverse)?;
            if self.unroll > 1 {
                operation.field("unroll", self.unroll)?;
            }
            if !self.captures.is_empty() {
                operation.field("captures", format_args!("{}", render_capture_list(&self.captures)))?;
            }
            Ok(())
        })
    }
}

/// Extracts slice `iteration` of a stacked value along its leading axis and drops that axis.
///
/// The slice bounds and the squeezed shape are derived from the stacked value's own type, which must be fully static
/// with a leading axis of extent greater than `iteration` (guaranteed for stacked scan values by construction).
pub fn read_scan_iteration<V>(stack: &V, iteration: usize) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + Slice + Reshape,
{
    let stack_type = stack.r#type().into_owned();
    let dimensions = stack_type
        .shape()
        .dimensions()
        .iter()
        .map(|dimension| {
            dimension.value().ok_or_else(|| {
                TypeError {
                    message: format!("scan iteration extraction requires a static stacked type but got {stack_type}"),
                }
                .into()
            })
        })
        .collect::<Result<Vec<usize>, ProgramError>>()?;
    let mut start_indices = vec![0; dimensions.len()];
    start_indices[0] = iteration;
    let mut limit_indices = dimensions.clone();
    limit_indices[0] = iteration + 1;
    let unit_strides = vec![1; dimensions.len()];
    let iteration_value = stack.slice(start_indices.as_slice(), limit_indices.as_slice(), unit_strides.as_slice())?;
    iteration_value.reshape(Shape::new(dimensions[1..].iter().map(|&dimension| Size::Static(dimension)).collect()))
}

/// Writes `value` as slice `iteration` of `accumulator` along its leading axis, prepending a unit axis to `value`
/// first.
pub(super) fn write_scan_iteration<V>(accumulator: V, iteration: usize, value: V) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + UpdateSlice + Reshape,
{
    let value_type = value.r#type().into_owned();
    let mut dimensions = Vec::with_capacity(value_type.rank() + 1);
    dimensions.push(Size::Static(1));
    dimensions.extend(value_type.shape().dimensions().iter().cloned());
    let expanded = value.reshape(Shape::new(dimensions))?;
    let mut start_indices = vec![0; value_type.rank() + 1];
    start_indices[0] = iteration;
    accumulator.update_slice(&expanded, start_indices.as_slice())
}

/// Drives one array scan loop over `[carry..., stacked_xs...]` inputs, delegating each iteration's body evaluation to
/// `interpret_iteration`.
///
/// This is the single source of truth for scan iteration arithmetic: iteration `iteration` consumes slice `iteration`
/// of every stacked input and writes slice `iteration` of every stacked output, visiting iterations from `length - 1`
/// down to `0` when `reverse` is `true` (the visit order reverses while the slice pairing does not).
/// [`ScanOperation`]'s interpretation evaluates the body program directly, while the linear scan interpretation arms
/// instantiate the body's scan-local residual references against each iteration's residual values before evaluating
/// it; both share this loop.
///
/// # Parameters
///
///   - `context`: Interpretation context used to allocate output stacks.
///   - `carry_count`: Number of loop-carried state leaves at the front of `inputs`.
///   - `length`: Static trip count.
///   - `reverse`: Whether iterations are visited in reverse order.
///   - `y_slice_types`: Per-iteration output slice types used to allocate the output stacks.
///   - `inputs`: Flat `[carry..., stacked_xs...]` input values.
///   - `interpret_iteration`: Evaluates one iteration, mapping `(iteration, [carry..., x_slice...])` to `[carry...,
///     y_slice...]`.
pub fn interpret_scan_iterations<V, C, InterpretIterationFn>(
    context: &C,
    carry_count: usize,
    length: usize,
    reverse: bool,
    y_slice_types: &[ArrayType],
    inputs: &[V],
    mut interpret_iteration: InterpretIterationFn,
) -> Result<Vec<V>, ProgramError>
where
    V: Value<Type = ArrayType> + Slice + UpdateSlice + Reshape,
    C: Zero<V>,
    InterpretIterationFn: FnMut(usize, Vec<V>) -> Result<Vec<V>, ProgramError>,
{
    let (carries, stacks) = inputs.split_at(carry_count);
    let mut carries = carries.to_vec();
    let mut accumulators = y_slice_types
        .iter()
        .map(|slice_type| {
            let stack_type = stacked_scan_type(slice_type, length);
            context.zero(&stack_type)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut iterations: Vec<usize> = (0..length).collect();
    if reverse {
        iterations.reverse();
    }
    for iteration in iterations {
        let mut iteration_inputs = carries.clone();
        for stack in stacks {
            iteration_inputs.push(read_scan_iteration(stack, iteration)?);
        }
        let mut iteration_outputs = interpret_iteration(iteration, iteration_inputs)?;
        check_count!("output", iteration_outputs, carry_count + y_slice_types.len(), ProgramError);
        let iteration_ys = iteration_outputs.split_off(carry_count);
        carries = iteration_outputs;
        for (accumulator, iteration_y) in accumulators.iter_mut().zip(iteration_ys.into_iter()) {
            *accumulator = write_scan_iteration(accumulator.clone(), iteration, iteration_y)?;
        }
    }
    carries.extend(accumulators);
    Ok(carries)
}

/// Type-family interpretation semantics for [`ScanOperation`], mirroring the `while` module's type-family dispatch:
/// [`ArrayType`] scans drive the stacked-slice loop of [`interpret_scan_iterations`] (allocating the output stacks
/// from the body interface's slice types, so zero-trip scans still shape their outputs), while [`DataType`] scans are
/// carry-only loops that thread the state through the body `length` times.
pub(crate) trait ScanInterpretation<C: Domain<Type = Self>>: ScanTypeSemantics {
    /// Interprets one scan over the attached body region; refer to the documentation of
    /// [`InterpretableOperation::interpret`] for the contract.
    fn interpret_scan<D: InterpretationDriver<C>>(
        carry_count: usize,
        length: usize,
        reverse: bool,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError>;
}

impl<C> ScanInterpretation<C> for ArrayType
where
    C: Domain<Type = ArrayType> + Zero<C::Value>,
    C::Value: Slice + UpdateSlice + Reshape,
{
    fn interpret_scan<D: InterpretationDriver<C>>(
        carry_count: usize,
        length: usize,
        reverse: bool,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        let body_interface = driver.region(0)?.interface();
        let y_slice_types = body_interface.output_types()[carry_count..].to_vec();
        interpret_scan_iterations(
            context,
            carry_count,
            length,
            reverse,
            y_slice_types.as_slice(),
            inputs,
            |_, iteration_inputs| driver.interpret_region(context, 0, iteration_inputs),
        )
    }
}

impl<C: Domain<Type = DataType>> ScanInterpretation<C> for DataType {
    fn interpret_scan<D: InterpretationDriver<C>>(
        carry_count: usize,
        length: usize,
        _reverse: bool,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        let mut state = inputs.to_vec();
        for _ in 0..length {
            state = driver.interpret_region(context, 0, state)?;
            check_count!("output", state, carry_count, ProgramError);
        }
        Ok(state)
    }
}

impl<Capture, C: Domain> InterpretableOperation<C> for ScanOperation<Capture>
where
    C::Type: ScanInterpretation<C>,
    Capture: Value<Type = C::Type>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        <C::Type>::interpret_scan(self.carry_count, self.length, self.reverse, context, driver, inputs)
    }
}

/// Partial-evaluation override for a [`Captured`](crate::operations::payloads::Captured)-payload [`ScanOperation`] over [`ArrayType`].
///
/// A scan's inputs are `[carry_init..., stacked_xs...]` and its body maps `[carry..., x_slice...]` to
/// `[next_carry..., y_slice...]`. Partial evaluation folds the known value of every *loop-invariant-known* carry into
/// the body: a carry is loop-invariant-known iff its init input is [`Known`](PartialValue::Known) and, with the
/// loop-invariant-known carries bound to their init values and everything else [`Unknown`](PartialValue::Unknown), its
/// body next-carry output is itself a known value equal to that init. Such a carry holds its init value on every
/// iteration, so binding it to that constant inside the body is sound and collapses every subcomputation that depended
/// only on it.
///
/// The invariant carries are found by a monotonic fixed point (a carry can only be demoted from invariant to
/// non-invariant as more carries are admitted, so it converges): on each round every currently-invariant carry is
/// bound to `Known(init)`, every other carry and every scanned element is `Unknown`, the body is partially evaluated
/// through the partial-evaluation driver's split requests (not
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate) directly, so the rule
/// carries no operation-enum semantic bounds), and a carry survives the round only if its next-carry output is
/// known and equal to its init. A carry init that is known but does not [`resolve`](Context::resolve) to a
/// [`Concrete`](crate::ValueResolution::Concrete) constant in the known-side context is never an invariance
/// candidate: its value could not be embedded into the rebuilt body as a
/// constant, and skipping it also keeps the fixed point's probe rounds from folding symbolic known work into a live
/// staging context. Under a staging known-side context, the surviving equality check is [`Tracer`](crate::Tracer)
/// identity, which degrades invariance detection to syntactic pass-through.
///
/// The residual scan keeps the *same* carry set and therefore the same output arity as the original operation. A
/// reduced carry set would change the scan's output count (`carry_count + scanned_outputs`). Instead, each
/// loop-invariant-known carry's body input is left dead and its body next-carry output is rebuilt as the constant init
/// value, so the carry slot survives while every use of its value folds away. The residual body is rebuilt from the
/// final round's residual body program: the carries and scanned elements feed it as inputs in scan body order, the
/// intensive residuals it consumes (the folded invariant carry values and any other value the body closed over) are
/// rebuilt inline as residual-body constants (so the residual scan needs no captures), and folded next-carry and
/// scanned outputs are rebuilt inline as constants exactly as the residual program reports them. The rewrite is
/// emitted over the original scan inputs unchanged.
///
/// Beyond the invariants, *time-varying* known work — known non-invariant carry chains and known stacked inputs —
/// is split off by `split_scan_by_knownness` into a *known scan* bound in the enclosing known-side context and an
/// *unknown scan* left in the residual program, connected by per-iteration residual edges the known scan stacks over
/// the scan length; see that function's documentation for the full recipe. If no carry is loop-invariant-known, no
/// time-varying known work exists, and the body has no other foldable subcomputation, the rule defers to the default
/// residualize-unchanged behavior.
impl<V, O, C> PartiallyEvaluatableOperation<C> for ScanOperation<V>
where
    V: Value<Type = ArrayType>,
    C: Context<Type = ArrayType, Constant = V, Operation = O>,
    C::Value: PartialEq,
    O: Operation<ArrayType> + From<ScanOperation<V>>,
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // The rule requests all nested-computation work through its region access (region 0 is the body), which keeps
        // its bounds free of the operation family's own semantic traits.
        //
        // When every input is known the whole scan folds by binding it in the known-side context; defer to that
        // default behavior.
        if inputs.iter().all(PartialEvaluationValue::is_known) {
            return context.fold_or_residualize(
                O::from(self.clone()),
                driver.regions().map(|region| region.to_program()).collect(),
                inputs,
            );
        }

        let carry_count = self.carry_count;
        let body = driver.region(0)?;
        let body_input_types = body.input_types();

        // The invariance fixed point below probes by folding the body through the *live* known-side context. For an
        // effectful body each probe round would execute (eager) or stage (staging) the body's effects once more, so
        // effectful bodies skip invariance probing entirely: the known-ness split's probes run through fresh,
        // discarded contexts and remain safe (see the effect placement contract on
        // `PartialEvaluationContext::fold_or_residualize`).
        if !body.effects().is_pure() {
            let time_varying_known = inputs.iter().any(PartialEvaluationValue::is_known);
            if time_varying_known {
                return split_scan_by_knownness(context, self, body, inputs, |input_known| {
                    driver.partition_program(body, input_known)
                });
            }
            return context.fold_or_residualize(O::from(self.clone()), vec![body.to_program()], inputs);
        }

        // A carry can only fold if its init input is known *and* concretizable in the known-side context: the folded
        // value must be embeddable as a rebuilt-body constant, and skipping symbolic knowns also keeps the fixed
        // point's probe rounds from folding symbolic known work into a live staging context.
        let carry_inits = (0..carry_count)
            .map(|index| {
                inputs[index].as_known().filter(|value| context.parent().resolve(value).is_concrete()).cloned()
            })
            .collect::<Vec<Option<C::Value>>>();

        // Monotonically narrow the set of loop-invariant-known carries to a fixed point. A round binds each invariant
        // carry to its init, leaves everything else unknown, and keeps a carry only if the body reproduces its init.
        // With no invariance candidates at all there is nothing the rebuild below could embed, so skip the
        // live-context probe entirely and go straight to the known-ness split (or the default).
        let mut invariant = carry_inits.iter().map(Option::is_some).collect::<Vec<bool>>();
        if invariant.iter().all(|candidate| !candidate) {
            if inputs.iter().any(PartialEvaluationValue::is_known) {
                return split_scan_by_knownness(context, self, body, inputs, |input_known| {
                    driver.partition_program(body, input_known)
                });
            }
            return context.fold_or_residualize(O::from(self.clone()), vec![body.to_program()], inputs);
        }
        let body_knowledge = |invariant: &[bool]| -> Vec<PartialValue<C::Value>> {
            let mut knowledge = Vec::with_capacity(body_input_types.len());
            for index in 0..carry_count {
                match (invariant[index], &carry_inits[index]) {
                    (true, Some(value)) => knowledge.push(PartialValue::Known(value.clone())),
                    _ => knowledge.push(PartialValue::Unknown(body_input_types[index].clone())),
                }
            }
            for slice_type in body_input_types[carry_count..].iter() {
                knowledge.push(PartialValue::Unknown(slice_type.clone()));
            }
            knowledge
        };

        let mut body_evaluation = driver.partially_evaluate_program(context, body, &body_knowledge(&invariant))?;
        loop {
            let refined = (0..carry_count)
                .map(|index| {
                    invariant[index]
                        && matches!(
                            &body_evaluation.outputs[index],
                            PartialEvaluationOutput::Known(value) if Some(value) == carry_inits[index].as_ref()
                        )
                })
                .collect::<Vec<bool>>();
            if refined == invariant {
                break;
            }
            invariant = refined;
            body_evaluation = driver.partially_evaluate_program(context, body, &body_knowledge(&invariant))?;
        }

        // Beyond the invariants, the remaining knowledge may still contain *time-varying* known work: known
        // non-invariant carry inits or known stacked inputs. Those cannot fold once, but they can ride a *known
        // scan* that runs per iteration, so after the invariant rewrite below the known-ness split takes over.
        let time_varying_known = (0..carry_count).any(|index| inputs[index].is_known() && !invariant[index])
            || inputs[carry_count..].iter().any(PartialEvaluationValue::is_known);

        // Nothing folded into the body: defer to the known-ness split when time-varying known work remains, and to
        // the default residualize-unchanged behavior otherwise. A loop-invariant-known carry always shrinks the body
        // (its uses fold to constants), so the only way nothing folds is an empty invariant set whose residual body
        // did not shrink either. The rebuild below embeds the probe's known values as inline body constants, which
        // is only possible when they are all concrete — under a staging known-side context the probe can fold a
        // constant-only chain into a live-trace tracer — so a non-concrete probe takes the same fallback.
        if (invariant.iter().all(|folded| !folded)
            && body_evaluation.program.instructions().len() >= body.instructions().len())
            || !context.all_knowns_are_concrete(&body_evaluation)
        {
            if time_varying_known {
                return split_scan_by_knownness(context, self, body, inputs, |input_known| {
                    driver.partition_program(body, input_known)
                });
            }
            return context.fold_or_residualize(O::from(self.clone()), vec![body.to_program()], inputs);
        }

        // The residual scan keeps the same carry set, so its output arity matches the original scan. A
        // loop-invariant-known carry is not dropped; instead its body next-carry output is rebuilt as the constant
        // init value and its body input is left dead, while its known value is folded into the body wherever it was
        // used. The body's per-iteration inputs are `[carry..., scanned_elem...]`.
        let mut builder = ProgramBuilder::<V, O>::new();
        let body_input_atoms =
            body_input_types.iter().map(|input_type| builder.add_input(input_type.clone())).collect::<Vec<_>>();

        // Feed the residual body program's inputs in its own input order. A surviving unknown body input is a
        // non-invariant carry or a scanned element and maps to the matching body input atom; a known residual (a
        // folded invariant carry value or another value the body closed over) is rebuilt as an inline constant by
        // recovering its staged payload through the known-side context.
        let mut residual_body_inputs = Vec::with_capacity(body_evaluation.inputs.len());
        for residual_input in body_evaluation.inputs.iter() {
            match residual_input {
                PartialEvaluationInput::Unknown(body_input) => residual_body_inputs.push(body_input_atoms[*body_input]),
                PartialEvaluationInput::Known(value) => {
                    residual_body_inputs.push(builder.add_constant(context.known_constant(value)?))
                }
            }
        }
        let spliced_outputs = builder.splice_program(&body_evaluation.program, &residual_body_inputs)?;

        // Assemble the residual body outputs as `[next_carry..., scanned_out...]`: a folded output (an invariant
        // carry's next value, or any output the body closed over) becomes an inline constant, and an unknown output
        // reads the spliced residual program's corresponding output.
        let body_output_atoms = (0..body.output_types().len())
            .map(|output_index| match &body_evaluation.outputs[output_index] {
                PartialEvaluationOutput::Known(value) => Ok(builder.add_constant(context.known_constant(value)?)),
                PartialEvaluationOutput::Unknown(index) => Ok(spliced_outputs[*index]),
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;

        let body_output_count = body_output_atoms.len();
        let residual_body = builder.build::<Vec<V>, Vec<V>>(
            body_output_atoms,
            vec![Placeholder; body_input_atoms.len()],
            vec![Placeholder; body_output_count],
        )?;

        let scan = ScanOperation::<V>::new(carry_count, self.length)
            .with_reverse(self.reverse)
            .with_unroll(self.unroll)?;

        // With time-varying known work remaining, the known-ness split of the invariant-folded scan finishes the
        // job; otherwise the residual scan's inputs are exactly the original scan's inputs: each carry init (now a
        // known residual for the folded carries) followed by each stacked input.
        if time_varying_known {
            return split_scan_by_knownness(context, &scan, residual_body.entry_region_ref(), inputs, |input_known| {
                driver.partition_program(residual_body.entry_region_ref(), input_known)
            });
        }
        context.fold_or_residualize(O::from(scan), vec![residual_body], inputs)
    }
}

/// Splits `scan` into a *known scan* bound in the enclosing known-side context and an *unknown scan* emitted into the
/// residual program, by a fixed point over carry known-ness — ryft's analogue of JAX's `_scan_partial_eval`, which
/// keeps time-varying known chains known instead of demoting them.
///
/// A carry stays known iff the body computes its next value from known values alone (its init must be known); known
/// stacked inputs are known throughout. Unlike the loop-*invariance* fixed point (which folds a carry once and
/// therefore requires a concretizable, value-equal init), known-ness needs neither concretizability nor value
/// equality, so symbolic known inits (tracers into a live outer trace) participate fully. Each fixed-point round
/// splits the body through a **fresh** staging context whose inputs stand in for the known body inputs, so no probe
/// or split work can leak into the caller's context.
///
/// From the converged split, the *known scan*'s body maps the known carries and known stacked slices to the known
/// next-carries, the known per-iteration outputs, and the **residual edges** — every known per-iteration value the
/// unknown side consumes — which the known scan *stacks* over the scan length as extra scanned outputs. The known
/// scan is bound whole into the enclosing known-side context over the original known inputs (interpreting it under an
/// eager context and staging it into the outer program under a staging one). The *unknown scan*'s body consumes the
/// unknown carries, the unknown stacked slices, and one slice of each stacked edge per iteration; a known body
/// next-carry belonging to an *unknown* carry (one whose value the unknown side threads) is instantiated as one more
/// residual edge that the unknown body passes through, mirroring JAX's `instantiate` flag. An effectful unknown body
/// still produces a zero-output residual scan when every boundary result belongs to the known side. If the split turns
/// out to have an empty known side, the scan residualizes unchanged through the default rule instead.
fn split_scan_by_knownness<V, O, C, PartitionRegion>(
    context: &PartialEvaluationContext<C>,
    scan: &ScanOperation<V>,
    body: RegionRef<'_, V, O>,
    inputs: &[PartialEvaluationValue<C::Value>],
    mut partition_region: PartitionRegion,
) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>
where
    V: Value<Type = ArrayType>,
    C: Context<Type = ArrayType, Constant = V, Operation = O>,
    O: Operation<ArrayType> + From<ScanOperation<V>>,
    PartitionRegion: FnMut(&[bool]) -> Result<PartitionedProgram<V, O>, ProgramError>,
{
    let carry_count = scan.carry_count;
    let body_input_types = body.input_types();
    let body_output_count = body.output_types().len();
    let input_known = inputs.iter().map(PartialEvaluationValue::is_known).collect::<Vec<bool>>();

    // Fixed point over carry known-ness, each round partitioning the borrowed body through a fresh staging context.
    let mut partition_body = |carry_known: &[bool]| -> Result<PartitionedProgram<V, O>, ProgramError> {
        let body_known = (0..body_input_types.len())
            .map(|index| if index < carry_count { carry_known[index] } else { input_known[index] })
            .collect::<Vec<bool>>();
        partition_region(body_known.as_slice())
    };

    let mut carry_known = input_known[..carry_count].to_vec();
    let partition = loop {
        let partition = partition_body(&carry_known)?;
        let refined = (0..carry_count)
            .map(|index| {
                carry_known[index] && matches!(partition.outputs().get(index), Some(PartialEvaluationOutput::Known(_)))
            })
            .collect::<Vec<bool>>();
        if refined == carry_known {
            break partition;
        }
        carry_known = refined;
    };
    let (known_program, residual_program, known_input_indices, residual_inputs, partition_outputs) =
        partition.into_parts();
    check_count!("output", partition_outputs, body_output_count, ProgramError);

    let expected_known_input_indices = (0..body_input_types.len())
        .filter(|&index| if index < carry_count { carry_known[index] } else { input_known[index] })
        .collect::<Vec<_>>();
    if known_input_indices != expected_known_input_indices {
        return Err(ProgramError::MalformedProgram(format!(
            "scan body partition reported known input indices {known_input_indices:?} but expected \
             {expected_known_input_indices:?}",
        )));
    }
    check_count!("input", residual_program.input_ids(), residual_inputs.len(), ProgramError);

    let known_result_count = partition_outputs
        .iter()
        .filter(|output| matches!(output, PartialEvaluationOutput::Known(_)))
        .count();
    let feeder_edge_count =
        residual_inputs.iter().filter(|input| matches!(input, PartialEvaluationInput::Known(_))).count();
    check_count!("output", known_program.output_ids(), known_result_count + feeder_edge_count, ProgramError);
    let known_program_output_types = known_program.output_types();

    // Assemble the known body's outputs: the known next-carries, then the known per-iteration outputs, then the
    // residual edges (the known feeders the unknown side consumes, plus the instantiated known next-carries of
    // unknown carries). Absolute positions into this list equal positions into the known scan's outputs, because the
    // known scan's outputs are its final carries followed by its stacked per-iteration outputs in the same order.
    let mut known_program_output_indices = Vec::with_capacity(known_program.output_ids().len());
    let mut known_carry_output_positions = vec![None; carry_count];
    for index in 0..carry_count {
        if carry_known[index] {
            match &partition_outputs[index] {
                PartialEvaluationOutput::Known(output) => {
                    known_carry_output_positions[index] = Some(known_program_output_indices.len());
                    known_program_output_indices.push(*output);
                }
                PartialEvaluationOutput::Unknown(_) => {
                    return Err(ProgramError::MalformedProgram(
                        "scan known-ness fixed point converged with an unknown next value for a known carry"
                            .to_string(),
                    ));
                }
            }
        }
    }
    let mut known_y_output_positions = vec![None; body_output_count - carry_count];
    for (position, output) in partition_outputs[carry_count..].iter().enumerate() {
        if let PartialEvaluationOutput::Known(output) = output {
            known_y_output_positions[position] = Some(known_program_output_indices.len());
            known_program_output_indices.push(*output);
        }
    }
    let mut edge_types = Vec::new();
    let mut feeder_edge_positions = Vec::with_capacity(residual_inputs.len());
    for input in residual_inputs.iter() {
        match input {
            PartialEvaluationInput::Known(edge) => {
                if *edge != edge_types.len() {
                    return Err(ProgramError::MalformedProgram(format!(
                        "scan body partition reported residual edge {edge} out of order",
                    )));
                }
                let output = known_result_count + edge;
                let output_type = known_program_output_types.get(output).ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "scan body partition residual edge {edge} has no known-program output",
                    ))
                })?;
                feeder_edge_positions.push(Some((*edge, known_program_output_indices.len())));
                edge_types.push(output_type.clone());
                known_program_output_indices.push(output);
            }
            PartialEvaluationInput::Unknown(_) => feeder_edge_positions.push(None),
        }
    }
    let mut instantiated_edge_positions = vec![None; carry_count];
    for index in 0..carry_count {
        if !carry_known[index] {
            if let PartialEvaluationOutput::Known(output) = &partition_outputs[index] {
                let output_type = known_program_output_types.get(*output).ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "scan body partition output {index} references missing known-program output {output}",
                    ))
                })?;
                instantiated_edge_positions[index] = Some((edge_types.len(), known_program_output_indices.len()));
                edge_types.push(output_type.clone());
                known_program_output_indices.push(*output);
            }
        }
    }

    // An empty known side means the split folds nothing; residualize unchanged through the default rule.
    if known_program_output_indices.is_empty() {
        return context.fold_or_residualize(O::from(scan.clone()), vec![body.to_program()], inputs);
    }

    // Bind the known scan into the enclosing known-side context over the original known inputs.
    let known_carry_count = carry_known.iter().filter(|&&known| known).count();
    let known_scan_inputs = known_input_indices
        .iter()
        .map(|&index| {
            inputs.get(index).cloned().ok_or_else(|| {
                ProgramError::MalformedProgram(format!("scan body partition references missing scan input {index}",))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut known_body_builder = ProgramBuilder::<V, O>::new();
    let known_body_inputs = known_program
        .input_types()
        .into_iter()
        .map(|input_type| known_body_builder.add_input(input_type))
        .collect::<Vec<_>>();
    let known_program_outputs = known_body_builder.splice_program(&known_program, known_body_inputs.as_slice())?;
    let known_output_atoms = known_program_output_indices
        .iter()
        .map(|&output| {
            known_program_outputs.get(output).copied().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "scan body partition references missing known-program output {output}",
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let known_body = known_body_builder.build::<Vec<V>, Vec<V>>(
        known_output_atoms,
        vec![Placeholder; known_body_inputs.len()],
        vec![Placeholder; known_program_output_indices.len()],
    )?;
    let known_scan = ScanOperation::<V>::new(known_carry_count, scan.length)
        .with_reverse(scan.reverse)
        .with_unroll(scan.unroll)?;
    let known_outputs =
        context.fold_or_residualize(O::from(known_scan), vec![known_body], known_scan_inputs.as_slice())?;

    // Assemble the unknown body over `[unknown carries..., unknown stacked slices..., edge slices...]`, splicing the
    // residual body program over its unknown inputs and edge inputs, with instantiated known next-carries passed
    // through from their edge slices.
    let mut unknown_output_ordinals = vec![None; body_output_count];
    let mut residual_outputs = Vec::new();
    let needs_unknown_scan =
        partition_outputs.iter().any(|output| matches!(output, PartialEvaluationOutput::Unknown(_)))
            || (0..carry_count).any(|index| !carry_known[index])
            || !residual_program.effects().is_pure();
    if needs_unknown_scan {
        let mut builder = ProgramBuilder::<V, O>::new();
        let mut unknown_body_input_atoms = vec![None; body_input_types.len()];
        for (index, input_type) in body_input_types.iter().enumerate() {
            let known = if index < carry_count { carry_known[index] } else { input_known[index] };
            if !known {
                unknown_body_input_atoms[index] = Some(builder.add_input(input_type.clone()));
            }
        }
        let edge_input_atoms =
            edge_types.iter().map(|edge_type| builder.add_input(edge_type.clone())).collect::<Vec<_>>();

        let mut spliced_inputs = Vec::with_capacity(residual_inputs.len());
        for input in residual_inputs.iter() {
            match input {
                PartialEvaluationInput::Unknown(index) => {
                    spliced_inputs.push(unknown_body_input_atoms.get(*index).copied().flatten().ok_or_else(|| {
                        ProgramError::MalformedProgram(
                            "scan known-ness split saw a residual feeder for a known body input".to_string(),
                        )
                    })?);
                }
                PartialEvaluationInput::Known(edge) => {
                    spliced_inputs.push(*edge_input_atoms.get(*edge).ok_or_else(|| {
                        ProgramError::MalformedProgram("scan known-ness split lost a residual edge".to_string())
                    })?)
                }
            }
        }
        let spliced_outputs = builder.splice_program(&residual_program, &spliced_inputs)?;

        let mut unknown_output_atoms = Vec::new();
        for index in 0..body_output_count {
            let owned_by_unknown_side = if index < carry_count {
                !carry_known[index]
            } else {
                matches!(&partition_outputs[index], PartialEvaluationOutput::Unknown(_))
            };
            if !owned_by_unknown_side {
                continue;
            }
            unknown_output_ordinals[index] = Some(unknown_output_atoms.len());
            match &partition_outputs[index] {
                PartialEvaluationOutput::Unknown(spliced) => unknown_output_atoms.push(spliced_outputs[*spliced]),
                PartialEvaluationOutput::Known(_) => {
                    let (edge, _) = instantiated_edge_positions[index].ok_or_else(|| {
                        ProgramError::MalformedProgram(
                            "scan known-ness split lost an instantiated carry edge".to_string(),
                        )
                    })?;
                    unknown_output_atoms.push(edge_input_atoms[edge]);
                }
            }
        }

        let unknown_body_input_count =
            unknown_body_input_atoms.iter().filter(|atom| atom.is_some()).count() + edge_input_atoms.len();
        let unknown_output_count = unknown_output_atoms.len();
        let unknown_body = builder.build::<Vec<V>, Vec<V>>(
            unknown_output_atoms,
            vec![Placeholder; unknown_body_input_count],
            vec![Placeholder; unknown_output_count],
        )?;
        let unknown_carry_count = carry_known.iter().filter(|&&known| !known).count();
        let unknown_scan = ScanOperation::<V>::new(unknown_carry_count, scan.length)
            .with_reverse(scan.reverse)
            .with_unroll(scan.unroll)?;

        // The unknown scan consumes the unknown original inputs followed by one stacked edge per residual edge, each
        // edge fed by the known scan's matching stacked output.
        let mut unknown_scan_inputs = Vec::new();
        for (index, input) in inputs.iter().enumerate() {
            let known = if index < carry_count { carry_known[index] } else { input_known[index] };
            if !known {
                unknown_scan_inputs.push(input.clone());
            }
        }
        let mut edge_known_output_positions = feeder_edge_positions
            .iter()
            .flatten()
            .chain(instantiated_edge_positions.iter().flatten())
            .collect::<Vec<_>>();
        edge_known_output_positions.sort_by_key(|(edge, _)| *edge);
        for (_, known_output_position) in edge_known_output_positions {
            unknown_scan_inputs.push(known_outputs.get(*known_output_position).cloned().ok_or_else(|| {
                ProgramError::MalformedProgram(
                    "scan known-ness split known scan produced no output for a residual edge".to_string(),
                )
            })?);
        }
        residual_outputs =
            context.residualize(O::from(unknown_scan), vec![unknown_body], unknown_scan_inputs.as_slice())?;
    }

    // Reassemble the original scan's outputs from the two sides.
    (0..body_output_count)
        .map(|index| {
            let known_position = if index < carry_count {
                known_carry_output_positions[index]
            } else {
                known_y_output_positions[index - carry_count]
            };
            if let Some(position) = known_position {
                return known_outputs.get(position).cloned().ok_or_else(|| {
                    ProgramError::MalformedProgram(
                        "scan known-ness split known scan produced no output for a known result".to_string(),
                    )
                });
            }
            let ordinal = unknown_output_ordinals[index].ok_or_else(|| {
                ProgramError::MalformedProgram(
                    "scan known-ness split produced a result owned by neither side".to_string(),
                )
            })?;
            residual_outputs.get(ordinal).cloned().ok_or_else(|| {
                ProgramError::MalformedProgram(
                    "scan known-ness split unknown scan produced no output for a residual result".to_string(),
                )
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::{EagerContext, StagingContext};
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::constants::ZeroLikeOperation;
    use crate::operations::math::{AddOperation, MulOperation};
    use crate::parameters::Placeholder;
    use crate::programs::{Program, ProgramBuilder};
    use crate::tests::TestArray;
    use crate::tracing::DomainTracingContext;
    use crate::tracing_v2::ArrayOperation;
    use crate::types::DataType;

    use super::*;

    type TestScanOperation = ScanOperation<TestArray>;

    /// Returns the [`RegionInterface`] of the provided flat region program.
    fn region_interface(
        program: &Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>>,
    ) -> RegionInterface<ArrayType> {
        program.interface()
    }

    /// Builds a cumulative-product body program that maps `[carry, x]` to `[carry * x, carry * x]`: the new carry is
    /// the running product and each iteration also emits that product as a stacked output slice.
    fn product_body() -> Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let carry = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let product = builder.add_instruction(MulOperation, Vec::new(), vec![carry, x]).unwrap()[0];
        builder
            .build(vec![product, product], vec![Placeholder, Placeholder], vec![Placeholder, Placeholder])
            .unwrap()
    }

    /// Builds a carry-only body program that maps `[carry]` to `[carry + carry]` with no stacked inputs or outputs.
    fn doubling_body() -> Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let carry = builder.add_input(ArrayType::scalar(DataType::F64));
        let doubled = builder.add_instruction(AddOperation, Vec::new(), vec![carry, carry]).unwrap()[0];
        builder.build(vec![doubled], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_scan() {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let stacked_f64 = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let operation = TestScanOperation::new(1, 3);
        let body = product_body();
        let interfaces = vec![region_interface(&body)];

        // Operation identity, declared region slots, output provenance, and accessors.
        assert_eq!(Operation::<ArrayType>::name(&operation), SCAN_OPERATION_NAME);
        assert_eq!(Operation::<ArrayType>::region_names(&operation), &["body"]);
        assert_eq!(
            Operation::<ArrayType>::output_region_provenance(&operation, 1),
            vec![OutputRegionProvenance { region_index: 0, output_index: 1 }],
        );
        assert_eq!(operation.carry_count(), 1);
        assert_eq!(operation.length(), 3);
        assert!(!operation.reverse());
        assert_eq!(operation.unroll(), 1);
        assert!(operation.clone().with_reverse(true).reverse());
        assert_eq!(format!("{operation}"), "scan [carry_count=1, length=3, reverse=false]");

        // Type inference validates the body interface, the carry, and the stacked input types, and returns the
        // stacked output types.
        assert_eq!(
            operation.infer_output_types(&[scalar_f64.clone(), stacked_f64.clone()], interfaces.as_slice()),
            Ok(vec![scalar_f64.clone(), stacked_f64.clone()]),
        );
        assert_eq!(
            operation.infer_output_types(&[scalar_f64.clone(), stacked_f64.clone()], &[]),
            Err(TypeError { message: "scan expects 1 attached region but got 0".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&scalar_f64), interfaces.as_slice()),
            Err(TypeError { message: "expected 2 inputs but got 1".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[scalar_f64.clone(), scalar_f64.clone()], interfaces.as_slice()),
            Err(TypeError {
                message: "scan input 1 has type f64[] which is incompatible with the expected type f64[3]".to_string(),
            }),
        );

        // The lowering-only unroll factor must be at least 1 and must evenly divide the scan length; valid factors
        // render only when greater than 1 and interpretation ignores them entirely.
        assert_eq!(
            TestScanOperation::new(1, 3).with_unroll(0).map(|_| ()),
            Err(ProgramError::Type(TypeError { message: "scan unroll factor must be at least 1".to_string() })),
        );
        assert_eq!(
            TestScanOperation::new(1, 3).with_unroll(2).map(|_| ()),
            Err(ProgramError::Type(TypeError {
                message: "scan unroll factor 2 must evenly divide the scan length 3".to_string(),
            })),
        );
        let unrolled = TestScanOperation::new(1, 3).with_unroll(3).unwrap();
        assert_eq!(unrolled.unroll(), 3);
        assert_eq!(format!("{unrolled}"), "scan [carry_count=1, length=3, reverse=false, unroll=3]");
        let context = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let outputs = context
            .bind(
                ArrayOperation::Scan(unrolled),
                vec![body.clone()],
                &[TestArray::scalar(1.0), TestArray::vector(vec![2.0, 3.0, 4.0])],
            )
            .unwrap();
        assert_eq!(outputs[0].values, vec![24.0]);
        assert_eq!(outputs[1].values, vec![2.0, 6.0, 24.0]);

        // Inference rejects carry counts that exceed the body signature, mismatched carry types, and dynamically
        // sized body slice types over the attached region interface.
        assert_eq!(
            TestScanOperation::new(3, 3)
                .infer_output_types(&[scalar_f64.clone(), stacked_f64.clone()], interfaces.as_slice()),
            Err(TypeError { message: "scan carry count 3 exceeds the body input count 2".to_string() }),
        );
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let carry = builder.add_input(scalar_f64.clone());
        let x = builder.add_input(scalar_f64.clone());
        let product = builder.add_instruction(MulOperation, Vec::new(), vec![carry, x]).unwrap()[0];
        let no_output_body = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![product], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            TestScanOperation::new(2, 3)
                .infer_output_types(&[scalar_f64.clone(), scalar_f64.clone()], &[region_interface(&no_output_body)],),
            Err(TypeError { message: "scan carry count 2 exceeds the body output count 1".to_string() }),
        );
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let mismatched_carry = builder.add_input(scalar_f64.clone());
        let mismatched_output =
            builder.add_instruction(ZeroLikeOperation, Vec::new(), vec![mismatched_carry]).unwrap()[0];
        let mismatched_output = builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::Equal),
                Vec::new(),
                vec![mismatched_output, mismatched_carry],
            )
            .unwrap()[0];
        let mismatched_body = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![mismatched_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            TestScanOperation::new(1, 3)
                .infer_output_types(std::slice::from_ref(&scalar_f64), &[region_interface(&mismatched_body)]),
            Err(TypeError {
                message: "scan body carry type signature mismatch: expected [f64[]] but got [bool[]]".to_string(),
            }),
        );
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let dynamic_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None)]));
        let dynamic_carry = builder.add_input(dynamic_type.clone());
        let dynamic_body = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![dynamic_carry], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            TestScanOperation::new(1, 3)
                .infer_output_types(std::slice::from_ref(&dynamic_type), &[region_interface(&dynamic_body)]),
            Err(TypeError {
                message: "scan body input 0 must have a fully static type but axis 0 of f64[*] has size *".to_string(),
            }),
        );

        // Eager binding threads the carry while stacking the per-iteration outputs: a cumulative product over
        // `xs = [2, 3, 4]` starting at `1` produces the final carry `24` and the running products `[2, 6, 24]`.
        let outputs = context
            .bind(
                ArrayOperation::Scan(operation.clone()),
                vec![body.clone()],
                &[TestArray::scalar(1.0), TestArray::vector(vec![2.0, 3.0, 4.0])],
            )
            .unwrap();
        assert_eq!(outputs[0].values, vec![24.0]);
        assert_eq!(outputs[1].values, vec![2.0, 6.0, 24.0]);

        // A reversed scan visits the slices from the back while keeping output slice `i` aligned with input slice
        // `i`: the running products visit `4, 3, 2` and land in iterations `2, 1, 0`.
        let reversed = TestScanOperation::new(1, 3).with_reverse(true);
        let outputs = context
            .bind(
                ArrayOperation::Scan(reversed),
                vec![body.clone()],
                &[TestArray::scalar(1.0), TestArray::vector(vec![2.0, 3.0, 4.0])],
            )
            .unwrap();
        assert_eq!(outputs[0].values, vec![24.0]);
        assert_eq!(outputs[1].values, vec![24.0, 12.0, 4.0]);

        // A carry-only scan with no stacked inputs or outputs applies the body `length` times.
        let carry_only = TestScanOperation::new(1, 3);
        let outputs = context
            .bind(ArrayOperation::Scan(carry_only), vec![doubling_body()], &[TestArray::scalar(1.0)])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].values, vec![8.0]);

        // A zero-length scan returns the initial carries and empty stacked outputs.
        let empty = TestScanOperation::new(1, 0);
        let empty_stacked_f64 = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(0)]));
        let outputs = context
            .bind(
                ArrayOperation::Scan(empty),
                vec![body.clone()],
                &[TestArray::scalar(1.0), TestArray::new(empty_stacked_f64, vec![])],
            )
            .unwrap();
        assert_eq!(outputs[0].values, vec![1.0]);
        assert_eq!(outputs[1].values, Vec::<f64>::new());

        // Staging imports the body program as an attached region of the staged instruction instead of running scan
        // iterations eagerly over staged values.
        let context = DomainTracingContext::<EagerContext<TestArray, ArrayOperation<TestArray>>>::new();
        let builder = context.builder().clone();
        let staged_carry = context.input(scalar_f64.clone());
        let staged_xs = context.input(stacked_f64.clone());
        let outputs = context
            .stage_operation(operation.clone(), [body.clone()], &[staged_carry.clone(), staged_xs.clone()])
            .unwrap();
        assert_eq!(outputs.len(), 2);
        let builder = builder.borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert!(matches!(builder.instructions()[0].operation(), ArrayOperation::Scan(_)));
        assert_eq!(builder.instructions()[0].regions().len(), 1);
        assert_eq!(
            builder.instructions()[0].inputs(),
            &[staged_carry.atom_id().unwrap(), staged_xs.atom_id().unwrap()],
        );
        assert_eq!(outputs[0].atom_id(), Ok(builder.instructions()[0].outputs()[0]));
        assert_eq!(outputs[1].atom_id(), Ok(builder.instructions()[0].outputs()[1]));

        // Program rendering shows the attached body region at the instruction with its declared slot name.
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let program_carry = builder.add_input(scalar_f64);
        let program_xs = builder.add_input(stacked_f64);
        let program_outputs = builder
            .add_instruction(ArrayOperation::Scan(operation), vec![body_region], vec![program_carry, program_xs])
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(
                program_outputs,
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3] .
                let %2:f64[], %3:f64[3] = scan [carry_count=1, length=3, reverse=false] %0 %1 [
                    body={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = mul %0 %1
                        in (%2, %2)
                    },
                ]
                in (%2, %3)
            "}
            .trim_end(),
        );
    }

    /// Scan input validation compares the declared types derived from the body signature against actual input types
    /// with `Type::is_refined_by`, so actual types carrying optional metadata that the declared types leave
    /// unspecified (e.g., the normalized shardings every concrete backend array type carries) are accepted, while
    /// data type and shape mismatches are still rejected.
    #[test]
    fn test_scan_input_type_refinement() {
        use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};

        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let stacked_f64 = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let operation = TestScanOperation::new(1, 3);
        let interfaces = vec![region_interface(&product_body())];

        // Sharded actual input types refine the metadata-free declared carry and stacked input types, and the
        // inferred output types stay declared (i.e., metadata-free) rather than inheriting the input shardings.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharded_carry = scalar_f64.clone().with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
        let sharded_stacked = stacked_f64
            .clone()
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert_eq!(
            operation.infer_output_types(&[sharded_carry, sharded_stacked], interfaces.as_slice()),
            Ok(vec![scalar_f64.clone(), stacked_f64.clone()]),
        );

        // Data type and shape mismatches are still rejected with the declared-vs-actual framing.
        assert_eq!(
            operation
                .infer_output_types(&[ArrayType::scalar(DataType::F32), stacked_f64.clone()], interfaces.as_slice(),),
            Err(TypeError {
                message: "scan input 0 has type f32[] which is incompatible with the expected type f64[]".to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(
                &[scalar_f64, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4)]))],
                interfaces.as_slice(),
            ),
            Err(TypeError {
                message: "scan input 1 has type f64[4] which is incompatible with the expected type f64[3]".to_string(),
            }),
        );
    }

    /// A scan whose body prints inside its known chain keeps the effect in the *known scan* of the known-ness
    /// split: effectful bodies skip the live-context invariance probes and go straight to the split, whose fresh
    /// probe contexts fold the all-known print into the known side. The known scan staged into the live outer trace
    /// owns the print (running it once per iteration, all before the residual side, per the effect placement
    /// contract), and the residual scan stays pure.
    #[test]
    fn test_partially_evaluate_scan_keeps_effectful_known_work_in_the_known_scan() {
        use crate::operations::debugging::PrintOperation;
        use crate::tracing::TracingContext;

        let scalar = || ArrayType::scalar(DataType::F64);
        let stacked = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));

        // Body `[acc, k, x] -> [acc + (print(k) * k) * x, k, acc + (print(k) * k) * x]`: the print sits inside the
        // otherwise-known `k * k` chain.
        let body = {
            let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
            let acc = builder.add_input(scalar());
            let k = builder.add_input(scalar());
            let x = builder.add_input(scalar());
            let printed = builder.add_instruction(PrintOperation::new("k"), Vec::new(), vec![k]).unwrap()[0];
            let ksq = builder.add_instruction(MulOperation, Vec::new(), vec![printed, k]).unwrap()[0];
            let kx = builder.add_instruction(MulOperation, Vec::new(), vec![ksq, x]).unwrap()[0];
            let next_acc = builder.add_instruction(AddOperation, Vec::new(), vec![acc, kx]).unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(
                    vec![next_acc, k, next_acc],
                    vec![Placeholder; 3],
                    vec![Placeholder; 3],
                )
                .unwrap()
        };

        let scan = TestScanOperation::new(2, 3);
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let acc_init = builder.add_input(scalar());
        let k_init = builder.add_input(scalar());
        let xs = builder.add_input(stacked.clone());
        let outputs = builder
            .add_instruction(ArrayOperation::Scan(scan), vec![body_region], vec![acc_init, k_init, xs])
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(outputs, vec![Placeholder; 3], vec![Placeholder; 3])
            .unwrap();

        let outer = TracingContext::<TestArray, ArrayOperation<TestArray>>::new();
        let known_carry = outer.input(scalar());
        let knowledge =
            vec![PartialValue::Unknown(scalar()), PartialValue::Known(known_carry), PartialValue::Unknown(stacked)];
        let evaluation = program.partially_evaluate_in_context(&outer, knowledge.as_slice()).unwrap();

        // The known scan owns the print (visible through the nested-program effects union) and the residual scan
        // is pure, consuming the stacked known-chain edge instead.
        {
            let outer_builder = outer.builder().borrow();
            assert_eq!(outer_builder.instructions().len(), 1);
            let known_instruction = &outer_builder.instructions()[0];
            assert!(matches!(known_instruction.operation(), ArrayOperation::Scan(_)));
            let known_body = outer_builder.region_ref(known_instruction.regions()[0]).unwrap().to_program();
            assert!(known_body.effects().is_ordered());
        }
        assert!(evaluation.program.effects().is_pure());
        let residual_scans = evaluation
            .program
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), ArrayOperation::Scan(_)))
            .count();
        assert_eq!(residual_scans, 1);
    }

    /// A split scan retains an effectful unknown body as a zero-output residual scan even when every boundary result
    /// belongs to the known side.
    #[test]
    fn test_partially_evaluate_scan_preserves_zero_output_residual_effects() {
        use crate::operations::debugging::PrintOperation;
        use crate::partial::{PartialEvaluationOutput, PartialValue};
        use crate::tracing::TracingContext;

        let scalar = || ArrayType::scalar(DataType::F64);
        let stacked = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));

        let mut body_builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let carry = body_builder.add_input(scalar());
        let input = body_builder.add_input(scalar());
        body_builder.add_instruction(PrintOperation::new("x"), Vec::new(), vec![input]).unwrap();
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![carry], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert!(body.partition(&[true, false]).unwrap().residual_program().effects().is_ordered());

        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let carry_init = builder.add_input(scalar());
        let inputs = builder.add_input(stacked.clone());
        let output = builder
            .add_instruction(
                ArrayOperation::Scan(TestScanOperation::new(1, 3)),
                vec![body_region],
                vec![carry_init, inputs],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let outer = TracingContext::<TestArray, ArrayOperation<TestArray>>::new();
        let symbolic_carry = outer.input(scalar());
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(symbolic_carry), PartialValue::Unknown(stacked)],
            )
            .unwrap();

        assert!(matches!(evaluation.outputs.as_slice(), [PartialEvaluationOutput::Known(_)]));
        assert!(evaluation.program.effects().is_ordered());
        assert_eq!(evaluation.program.output_ids().len(), 0);
        assert_eq!(evaluation.program.instructions().len(), 1);
        let residual_scan = &evaluation.program.instructions()[0];
        assert!(matches!(residual_scan.operation(), ArrayOperation::Scan(_)));
        assert_eq!(residual_scan.outputs().len(), 0);
        let residual_body = evaluation.program.region_ref(residual_scan.regions()[0]).unwrap();
        assert!(residual_body.effects().is_ordered());
        assert_eq!(residual_body.output_types().len(), 0);
    }

    /// Under a *staging* known-side context, a *symbolic* known carry (a genuine outer tracer) participates in the
    /// known-ness split: the known chain (`k` and `k * k`) rides a known scan staged into the live outer trace,
    /// stacking the per-iteration `k * k` values the unknown side consumes as a residual edge, while the unknown
    /// accumulator chain stays behind a residual scan — JAX's `_scan_partial_eval` behavior. The fixed-point probes
    /// run through fresh contexts, so the only instruction the outer trace gains is the known scan itself.
    #[test]
    fn test_partially_evaluate_scan_splits_symbolic_known_carries_under_staging() {
        use crate::tracing::TracingContext;

        let scalar = || ArrayType::scalar(DataType::F64);
        let stacked = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));

        // Body `[acc, k, x] -> [acc + (k * k) * x, k, acc + (k * k) * x]`, as in the loop-invariant test below.
        let body = || {
            let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
            let acc = builder.add_input(scalar());
            let k = builder.add_input(scalar());
            let x = builder.add_input(scalar());
            let ksq = builder.add_instruction(MulOperation, Vec::new(), vec![k, k]).unwrap()[0];
            let kx = builder.add_instruction(MulOperation, Vec::new(), vec![ksq, x]).unwrap()[0];
            let next_acc = builder.add_instruction(AddOperation, Vec::new(), vec![acc, kx]).unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(
                    vec![next_acc, k, next_acc],
                    vec![Placeholder; 3],
                    vec![Placeholder; 3],
                )
                .unwrap()
        };

        let scan = TestScanOperation::new(2, 3);
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let body_region = builder.import_region(body().entry_region_ref());
        let acc_init = builder.add_input(scalar());
        let k_init = builder.add_input(scalar());
        let xs = builder.add_input(stacked.clone());
        let outputs = builder
            .add_instruction(ArrayOperation::Scan(scan), vec![body_region], vec![acc_init, k_init, xs])
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(outputs, vec![Placeholder; 3], vec![Placeholder; 3])
            .unwrap();

        let outer = TracingContext::<TestArray, ArrayOperation<TestArray>>::new();
        let known_carry = outer.input(scalar());
        let knowledge = vec![
            PartialValue::Unknown(scalar()),
            PartialValue::Known(known_carry),
            PartialValue::Unknown(stacked.clone()),
        ];
        let evaluation = program.partially_evaluate_in_context(&outer, knowledge.as_slice()).unwrap();

        // The known chain landed in the outer program as one known scan whose body carries `k` and stacks `k * k`
        // per iteration as the residual edge — the fixed-point probes leaked nothing else.
        {
            let outer_builder = outer.builder().borrow();
            assert_eq!(outer_builder.instructions().len(), 1);
            let known_instruction = &outer_builder.instructions()[0];
            let ArrayOperation::Scan(known_scan) = known_instruction.operation() else {
                panic!("expected the outer program to contain the known scan");
            };
            assert_eq!(known_scan.carry_count(), 1);
            let known_body = outer_builder.region_ref(known_instruction.regions()[0]).unwrap().to_program();
            assert_eq!(known_body.input_types().len(), 1);
            assert_eq!(known_body.output_types().len(), 2);
            assert_eq!(known_body.instructions().len(), 1);
        }

        // The unknown accumulator chain stays behind one residual scan over `[acc, x_slice, stacked k * k slice]`.
        assert_eq!(evaluation.program.instructions().len(), 1);
        let residual_instruction = &evaluation.program.instructions()[0];
        let ArrayOperation::Scan(residual_scan) = residual_instruction.operation() else {
            panic!("expected the residual program to contain the unknown scan");
        };
        assert_eq!(residual_scan.carry_count(), 1);
        let residual_body = evaluation.program.region_ref(residual_instruction.regions()[0]).unwrap().to_program();
        assert_eq!(residual_body.input_types().len(), 3);
        assert_eq!(residual_body.output_types().len(), 2);
        assert_eq!(residual_body.instructions().len(), 2);
        assert_eq!(evaluation.inputs.len(), 3);
        assert!(matches!(&evaluation.inputs[0], PartialEvaluationInput::Unknown(0)));
        assert!(matches!(&evaluation.inputs[1], PartialEvaluationInput::Unknown(2)));
        assert!(matches!(&evaluation.inputs[2], PartialEvaluationInput::Known(value) if value.atom_id().is_ok()));

        // The final `k` is a *known* output (the known scan's final carry), and the accumulator outputs stay residual.
        assert_eq!(evaluation.outputs.len(), 3);
        assert!(matches!(&evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert!(matches!(&evaluation.outputs[1], PartialEvaluationOutput::Known(value) if value.atom_id().is_ok()));
        assert!(matches!(&evaluation.outputs[2], PartialEvaluationOutput::Unknown(1)));
    }

    /// The known-ness split keeps *time-varying* known work known under an eager context too: a known stacked input
    /// whose per-iteration squares feed the unknown accumulator executes during partial evaluation inside a known
    /// scan, the folded stacked output surfaces as a concrete known value, and the unknown scan consumes the concrete
    /// stacked squares as a residual edge.
    #[test]
    fn test_partially_evaluate_scan_splits_time_varying_known_work() {
        let scalar = || ArrayType::scalar(DataType::F64);
        let stacked = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));

        // Body `[c, x] -> [c + x * x, x * x]` over an unknown accumulator `c` and known stacked `xs`.
        let body = {
            let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
            let c = builder.add_input(scalar());
            let x = builder.add_input(scalar());
            let xsq = builder.add_instruction(MulOperation, Vec::new(), vec![x, x]).unwrap()[0];
            let next = builder.add_instruction(AddOperation, Vec::new(), vec![c, xsq]).unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(vec![next, xsq], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let scan = TestScanOperation::new(1, 3);
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let c_init = builder.add_input(scalar());
        let xs = builder.add_input(stacked.clone());
        let outputs = builder
            .add_instruction(ArrayOperation::Scan(scan), vec![body_region], vec![c_init, xs])
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let knowledge =
            vec![PartialValue::Unknown(scalar()), PartialValue::Known(TestArray::vector(vec![1.0, 2.0, 3.0]))];
        let evaluation = program.partially_evaluate(knowledge.as_slice()).unwrap();

        // The stacked squares were computed *during* partial evaluation by the known scan: they surface both as the
        // folded stacked output and as the residual edge feeding the unknown scan.
        assert_eq!(evaluation.outputs.len(), 2);
        assert!(matches!(&evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert!(matches!(
            &evaluation.outputs[1],
            PartialEvaluationOutput::Known(value) if value.values == vec![1.0, 4.0, 9.0]
        ));
        assert_eq!(evaluation.inputs.len(), 2);
        assert!(matches!(&evaluation.inputs[0], PartialEvaluationInput::Unknown(0)));
        assert!(matches!(
            &evaluation.inputs[1],
            PartialEvaluationInput::Known(value) if value.values == vec![1.0, 4.0, 9.0]
        ));

        // The residual (unknown) scan accumulates the stacked squares: interpreting it at `c = 10` reproduces the
        // full interpretation of the original program.
        let residual_outputs = evaluation
            .program
            .interpret(vec![TestArray::scalar(10.0), TestArray::vector(vec![1.0, 4.0, 9.0])])
            .unwrap();
        let expected =
            program.interpret(vec![TestArray::scalar(10.0), TestArray::vector(vec![1.0, 2.0, 3.0])]).unwrap();
        assert_eq!(residual_outputs[0].values, expected[0].values);
        assert_eq!(residual_outputs[0].values, vec![24.0]);
    }

    /// With a *loop-invariant known* carry, a scan partially evaluates by folding that carry's value into the body: the
    /// residual scan keeps the same carry set (so its output arity is preserved) but its body shrinks because every
    /// subcomputation that depended only on the known carry collapses to a constant.
    ///
    /// The body over `[acc, k, x]` computes `ksq = k * k`, `kx = ksq * x`, `next_acc = acc + kx`, and returns
    /// `[next_acc, k, next_acc]`: `acc` is a running accumulator, `k` is forwarded unchanged (loop-invariant), and the
    /// stacked output is the running accumulator. With `k` known (`2`) and `acc` and `xs` unknown, the `k` carry is
    /// loop-invariant-known (its next-carry equals its init), so `ksq` folds to the constant `4` and the body shrinks
    /// from three instructions to two, with `final_k` folded to the constant `2` inside the residual scan body.
    /// Interpreting the residual program reproduces the original scan over the same inputs.
    #[test]
    fn test_partially_evaluate_folds_loop_invariant_known_carry() {
        let scalar = || ArrayType::scalar(DataType::F64);
        let stacked = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));

        // Body `[acc, k, x] -> [acc + (k * k) * x, k, acc + (k * k) * x]`.
        let body = || {
            let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
            let acc = builder.add_input(scalar());
            let k = builder.add_input(scalar());
            let x = builder.add_input(scalar());
            let ksq = builder.add_instruction(MulOperation, Vec::new(), vec![k, k]).unwrap()[0];
            let kx = builder.add_instruction(MulOperation, Vec::new(), vec![ksq, x]).unwrap()[0];
            let next_acc = builder.add_instruction(AddOperation, Vec::new(), vec![acc, kx]).unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(
                    vec![next_acc, k, next_acc],
                    vec![Placeholder; 3],
                    vec![Placeholder; 3],
                )
                .unwrap()
        };

        // Flat program over `[acc_init, k_init, xs]` staging the scan (two carries, one scanned input, length 3); its
        // outputs are `[final_acc, final_k, stacked_acc]`.
        let scan = TestScanOperation::new(2, 3);
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let body_region = builder.import_region(body().entry_region_ref());
        let acc_init = builder.add_input(scalar());
        let k_init = builder.add_input(scalar());
        let xs = builder.add_input(stacked.clone());
        let outputs = builder
            .add_instruction(ArrayOperation::Scan(scan), vec![body_region], vec![acc_init, k_init, xs])
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(outputs, vec![Placeholder; 3], vec![Placeholder; 3])
            .unwrap();

        let knowledge = vec![
            PartialValue::Unknown(scalar()),
            PartialValue::Known(TestArray::scalar(2.0)),
            PartialValue::Unknown(stacked.clone()),
        ];
        let evaluation = program.partially_evaluate(knowledge.as_slice()).unwrap();

        // All three scan outputs are produced by the residual program: the scan instruction itself residualizes
        // (its inputs are not all known), so even the loop-invariant `final_k` is computed by the residual scan
        // (whose body folds it to the constant `2`) rather than folded at the top level.
        assert!(matches!(&evaluation.outputs[0], PartialEvaluationOutput::Unknown(_)));
        assert!(matches!(&evaluation.outputs[1], PartialEvaluationOutput::Unknown(_)));
        assert!(matches!(&evaluation.outputs[2], PartialEvaluationOutput::Unknown(_)));

        // The residual program's only instruction is the rewritten scan, carrying its rewritten body as an
        // attached region.
        assert_eq!(evaluation.program.instructions().len(), 1);
        let residual_instruction = &evaluation.program.instructions()[0];
        let ArrayOperation::Scan(residual_scan) = residual_instruction.operation() else {
            panic!("expected the residual program to contain a rewritten scan");
        };

        // The carry set is preserved (so output arity matches), but the body shrank: `k * k` folded to a constant, so
        // the body drops from three instructions to two.
        assert_eq!(residual_scan.carry_count(), 2);
        assert_eq!(residual_scan.length(), 3);
        let residual_body = evaluation.program.region_ref(residual_instruction.regions()[0]).unwrap().to_program();
        assert!(residual_body.instructions().len() < body().instructions().len());
        assert_eq!(residual_body.instructions().len(), 2);

        // Correctness: interpreting the residual program reproduces the original program on the same concrete inputs.
        let runtime = |acc: f64, xs: Vec<f64>| -> Vec<TestArray> {
            let arguments = evaluation
                .inputs
                .iter()
                .map(|residual_input| match residual_input {
                    PartialEvaluationInput::Known(value) => value.clone(),
                    PartialEvaluationInput::Unknown(index) => match index {
                        0 => TestArray::scalar(acc),
                        _ => TestArray::vector(xs.clone()),
                    },
                })
                .collect::<Vec<_>>();
            let residual_outputs = evaluation.program.interpret(arguments).unwrap();
            evaluation
                .outputs
                .iter()
                .map(|output| match output {
                    PartialEvaluationOutput::Known(value) => value.clone(),
                    PartialEvaluationOutput::Unknown(index) => residual_outputs[*index].clone(),
                })
                .collect()
        };
        let original = |acc: f64, k: f64, xs: Vec<f64>| {
            program
                .interpret(vec![TestArray::scalar(acc), TestArray::scalar(k), TestArray::vector(xs)])
                .unwrap()
        };

        let reassembled = runtime(1.0, vec![5.0, 6.0, 7.0]);
        let expected = original(1.0, 2.0, vec![5.0, 6.0, 7.0]);
        assert_eq!(
            reassembled.iter().map(|value| value.values.clone()).collect::<Vec<_>>(),
            expected.iter().map(|value| value.values.clone()).collect::<Vec<_>>()
        );
        // `acc` threads `1 -> 1 + 4*5 -> 21 + 4*6 -> 45 + 4*7 = 73`; the stacked output records `[21, 45, 73]`; the
        // loop-invariant `k` final carry stays `2`.
        assert_eq!(reassembled[0].values, vec![73.0]);
        assert_eq!(reassembled[1].values, vec![2.0]);
        assert_eq!(reassembled[2].values, vec![21.0, 45.0, 73.0]);
    }
}
