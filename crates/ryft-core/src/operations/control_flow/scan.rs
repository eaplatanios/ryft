//! Contains the `scan` control-flow operation: [`ScanOperation`], a shape-determined loop that threads
//! `carry_count` loop-carried values through an attached body [`Region`](crate::Region) while consuming one slice of
//! every stacked input and producing one slice of every stacked output per iteration, together with its
//! interpretation, partial-evaluation, batching, forward-mode differentiation, and transposition rules. This is the
//! analogue of [JAX's `lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html) (including
//! `reverse` and the lowering-only `unroll` factor) and lowers to a
//! [StableHLO `while`](https://openxla.org/stablehlo/spec#while) loop with counter-indexed slice reads and writes.

use std::fmt::{Debug, Display};
use std::sync::Arc;

use crate::arrays::batching::align_array_batch;
use crate::arrays::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, ArrayIrBatch, ArrayIrBatching, ArrayIrType, ArrayIrValue,
    ArrayType, Dimension, DimensionType, DimensionValue, Shape,
};
use crate::axes::Axis;
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchedProgram, BatchingContext, BatchingDriver, BatchingError,
    ProgramBatchingOutputAxesPolicy,
};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    ResidualZeroProvider, TransposableOperation, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, check_types};
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::constants::zero::{Zero, ZeroOperationProvider};
use crate::operations::control_flow::{TemporalResidualOperation, TemporalResidualType};
use crate::operations::dimensions::dimension_size::DimensionSizeOperation;
use crate::operations::manipulation::broadcasting::{Broadcast, BroadcastOperation, DynamicBroadcastOperation};
use crate::operations::manipulation::reshaping::{Reshape, ReshapeOperation};
use crate::operations::manipulation::slicing::{Slice, SliceOperation, UpdateSlice, UpdateSliceOperation};
use crate::operations::manipulation::transposition::{Transpose, TransposeOperation};
use crate::parameters::Placeholder;
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationInput, PartialEvaluationOutput,
    PartialEvaluationValue, PartialValue, PartiallyEvaluatableOperation, PartitionedProgram,
};
use crate::programs::{
    AtomId, CalleeRegionDriver, MaybeZero, Operation, OperationFormatter, OperationProjection, OutputRegionProvenance,
    Program, ProgramBuilder, ProgramError, RegionInterface, RegionRef, RegionSlot, Type, TypeError,
    TypeIdentityPosition, TypeIdentityRenaming, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

// TODO(eaplatanios): Review this.

/// Canonical operation name for [`ScanOperation`].
pub const SCAN_OPERATION_NAME: &str = "scan";

/// [`Operation`] that applies a nested body [`Program`] a shape-determined number of times over loop-carried state
/// while
/// consuming one slice of each stacked input per iteration and stacking the body's per-iteration outputs. This is the
/// shape-determined loop primitive analogous to JAX's
/// [`lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html): where [`WhileOperation`] iterates a
/// data-dependent number of times and therefore supports reverse-mode differentiation only by unrolling in eager
/// domains, `scan` has a shape-determined trip count, so its linearization can *store* per-iteration residuals as
/// shape-determined stacks and its linear form transposes totally (see the `tracing_v2` scan rules).
///
/// The body [`Program`] maps `[carry..., x_slice...]` to `[carry..., y_slice...]`: the first
/// [`carry_count`](Self::carry_count) inputs and outputs are the loop-carried state (with identical type signatures),
/// the remaining inputs consume one slice of each stacked input per iteration, and the remaining outputs produce one
/// slice of each stacked output per iteration. The operation's inputs are `[carry..., stacked_xs...]` and its outputs
/// are `[final_carry..., stacked_ys...]`, where each stacked type prepends the
/// [`length`](Self::length) dimension to the corresponding body slice type. Iteration `i` consumes slice `i` of every
/// stacked input and produces slice `i` of every stacked output; when [`reverse`](Self::reverse) is `true` the
/// iterations visit the slices from `length - 1` down to `0`, but slice `i` of every stacked output still corresponds
/// to slice `i` of the stacked inputs (the visit *order* reverses, the slice *pairing* does not). Transposition of
/// the linear form simply flips [`reverse`](Self::reverse), so no array-reversal operation is ever needed.
///
/// The `length` is stored explicitly so that scans without stacked inputs (pure carry loops with stacked outputs)
/// remain well-defined. Homogeneous [`ArrayType`] scans require a static length. Composite [`ArrayIrType`] scans may
/// instead use one dynamic dimension identity and then
/// consume its matching first-class dimension value as a trailing runtime operand; this is the scalar-SSA trip-count
/// contract used by structurally batched scans.
///
/// The optional [`unroll`](Self::unroll) factor (attached via [`with_unroll`](Self::with_unroll)) is a
/// **lowering-only** attribute: interpretation and every transform rule (differentiation, transposition, batching)
/// ignore it semantically but preserve it on whatever scan they re-stage, while lowerings emit `unroll` body copies
/// per loop trip — and a fully unrolled straight-line lowering with no loop at all when `unroll` equals `length`.
///
/// The body computation is not part of this payload: it is a [`Region`](crate::Region) attached to the
/// [`Instruction`](crate::Instruction) applying the operation (the single [`region_slots`](Operation::region_slots)
/// slot `["body"]`), and semantic rules reach it through their driver-granted region access. Scans with owned bodies
/// supply the body [`Program`] through the region driver passed to [`Context::bind`];
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

    /// Shape-determined trip count of this [`ScanOperation`].
    pub(crate) length: Dimension,

    /// Boolean indicating whether iterations visit the stacked slices in reverse order.
    pub(crate) reverse: bool,

    /// Lowering-only unroll factor: the number of body copies emitted per loop trip (`1` keeps one body per trip).
    pub(crate) unroll: usize,
}

impl<Capture: Value> ScanOperation<Capture> {
    /// Creates a new [`ScanOperation`] with the provided carry count and shape-determined trip count, visiting
    /// iterations in
    /// increasing order (use [`Self::with_reverse`] to flip the visit order). The body [`Program`]
    /// mapping
    /// `[carry..., x_slice...]` to `[carry..., y_slice...]` is supplied separately as the operation's attached
    /// region (via the region driver passed to [`Context::bind`]);
    /// [`Operation::infer_output_types`] validates its signature against `carry_count` and `length`.
    ///
    /// # Parameters
    ///
    ///   - `carry_count`: Number of loop-carried state leaves at the front of the body's inputs and outputs.
    ///   - `length`: Shape-determined trip count.
    #[inline]
    pub fn new<L: Into<Dimension>>(carry_count: usize, length: L) -> Self {
        Self { captures: Vec::new(), carry_count, length: length.into(), reverse: false, unroll: 1 }
    }
    /// Returns this [`ScanOperation`] with the slice visit order set to `reverse`. For carry-only scans this only
    /// preserves lowering metadata because all iterations consume and produce loop-carried state.
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
        validate_scan_unroll(unroll, &self.length)?;
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

    /// Returns the shape-determined trip count of this [`ScanOperation`].
    #[inline]
    pub fn length(&self) -> &Dimension {
        &self.length
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
        self.render(formatter, 0)
    }
}

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

/// Validates that every dimension of `r#type` is static, reporting a precise error that names the scan `role` (for
/// example, `input 1` or `output 0`) when one is not. Scan body slice types must be fully static because stacking
/// prepends a static length dimension and per-iteration slice extraction must be provably in bounds.
fn check_static_scan_type(role: &str, index: usize, r#type: &ArrayType) -> Result<(), TypeError> {
    for (axis, dimension) in r#type.shape().dimensions().iter().enumerate() {
        if dimension.value().is_none() {
            return Err(TypeError::invalid(format!(
                "{SCAN_OPERATION_NAME} body {role} {index} must have a fully static type but axis {axis} of {type} has size \
                     {dimension}",
                r#type = r#type,
            )));
        }
    }
    Ok(())
}

/// Validates a scan unroll factor against the scan's trip count. The factor must be at least `1`; a static length must
/// be evenly divisible by it, while a dynamic length only supports `1`. This backs [`ScanOperation::with_unroll`] and
/// enum-payload validation of scan variants whose fields are public and therefore cannot rely on builder-time
/// validation alone.
pub(crate) fn validate_scan_unroll(unroll: usize, length: &Dimension) -> Result<(), TypeError> {
    if unroll == 0 {
        return Err(TypeError::invalid(format!("{SCAN_OPERATION_NAME} unroll factor must be at least 1")));
    }
    let Some(length) = length.value() else {
        if unroll == 1 {
            return Ok(());
        }
        return Err(TypeError::invalid(format!(
            "{SCAN_OPERATION_NAME} with dynamic length {length} only supports unroll factor 1",
        )));
    };
    if length % unroll != 0 {
        return Err(TypeError::invalid(format!(
            "{SCAN_OPERATION_NAME} unroll factor {unroll} must evenly divide the {SCAN_OPERATION_NAME} length {length}"
        )));
    }
    Ok(())
}

/// Returns the stacked variant of a scan body slice type, prepending `length` to its shape. The
/// stacked type preserves the slice's memory placement but carries no optional layout or sharding metadata, so it is
/// a declared type whose optional components are unspecified. Scan input validation compares it against actual input
/// types with [`Type::is_refined_by`](crate::programs::types::Type::is_refined_by).
pub(crate) fn stacked_scan_type<L: Into<Dimension>>(slice_type: &ArrayType, length: L) -> ArrayType {
    let mut dimensions = Vec::with_capacity(slice_type.rank() + 1);
    dimensions.push(length.into());
    dimensions.extend(slice_type.shape().dimensions().iter().cloned());
    ArrayType::new(slice_type.data_type(), Shape::new(dimensions)).with_memory(slice_type.memory())
}

/// Validates `[carry..., stacked_xs...]` input types against a scan body signature and returns the
/// `[carry..., stacked_ys...]` output types. This backs type inference for [`ScanOperation`].
///
/// The expected input types are declared types derived from the body signature (with stacked types built by
/// [`stacked_scan_type`], which carries no optional layout or sharding metadata), while the provided `input_types` may
/// be actual runtime value types carrying more precise optional metadata, such as the normalized
/// [`Sharding`](crate::arrays::Sharding)s that every concrete backend array type carries. Validation therefore uses the
/// directional declared-vs-actual [`Type::is_refined_by`] relation instead of strict type equality. The returned output
/// types are declared types built the same way and thus leave optional metadata unspecified for downstream consumers
/// (e.g., sharding propagation) to resolve.
pub(crate) fn scan_output_types(
    body_input_types: &[ArrayType],
    body_output_types: &[ArrayType],
    carry_count: usize,
    length: &Dimension,
    input_types: &[ArrayType],
) -> Result<Vec<ArrayType>, TypeError> {
    let mut expected_input_types = body_input_types[..carry_count].to_vec();
    expected_input_types
        .extend(body_input_types[carry_count..].iter().map(|slice_type| stacked_scan_type(slice_type, length)));
    check_count!("input", input_types, expected_input_types.len(), TypeError);
    for (index, (expected, actual)) in expected_input_types.iter().zip(input_types).enumerate() {
        if !expected.is_refined_by(actual) {
            return Err(TypeError::invalid(format!(
                "{SCAN_OPERATION_NAME} input {index} has type {actual} which is incompatible with the expected type {expected}",
            )));
        }
    }
    let mut output_types = body_output_types[..carry_count].to_vec();
    output_types
        .extend(body_output_types[carry_count..].iter().map(|slice_type| stacked_scan_type(slice_type, length)));
    Ok(output_types)
}

/// Type-family semantics for [`ScanOperation`].
///
/// [`ArrayType`] can represent scanned values by prepending a static leading axis to each per-iteration value type,
/// and requires a static trip count. [`ArrayIrType`] permits arrays and first-class dimensions in carry positions and
/// a dynamic trip count backed by one trailing dimension operand, but requires every stacked input, output, and
/// capture to be an array because the composite domain has no ragged or stacked dimension value. This trait keeps
/// those type rules local to the scan operation so the operation dispatcher itself can be generic over `T`.
pub trait ScanTypeSemantics: Type {
    /// Renames any dynamic identity referenced by a scan length.
    fn rename_scan_length(length: &Dimension, _renaming: &TypeIdentityRenaming<Self::Identity>) -> Dimension {
        length.clone()
    }

    /// Validates a scan body signature for this type family.
    fn validate_scan_body(
        body_input_types: &[Self],
        body_output_types: &[Self],
        carry_count: usize,
        length: &Dimension,
    ) -> Result<(), TypeError>;

    /// Derives the instantiated body input types from this scan's boundary input types.
    fn scan_body_input_types(
        input_types: &[Self],
        body_input_count: usize,
        carry_count: usize,
        length: &Dimension,
    ) -> Result<Vec<Self>, TypeError>;

    /// Infers this scan's output types from concrete input types.
    fn infer_scan_output_types(
        body_input_types: &[Self],
        body_output_types: &[Self],
        carry_count: usize,
        length: &Dimension,
        input_types: &[Self],
    ) -> Result<Vec<Self>, TypeError>;

    /// Returns the boundary type that stores one per-iteration value across the scan length.
    fn stacked_scan_type(r#type: &Self, length: &Dimension) -> Result<Self, TypeError>;

    /// Validates one capture value stored on this scan.
    fn validate_scan_capture<C: Value<Type = Self>>(
        capture: &C,
        index: usize,
        length: &Dimension,
    ) -> Result<(), TypeError>;

    /// Returns the [`Dimension`] every stacked value of this scan must actually be laid out over. This is the declared
    /// `length` except under a composite scan's exact-refinement arm, where the runtime length operand pins a concrete
    /// extent that differs from the still-symbolic declared variable; captures are validated against this effective
    /// dimension so a concretely refined trip count cannot over-read or truncate a capture stacked over the symbolic
    /// declared length. The default returns the declared `length` unchanged, which is exact for type families whose
    /// scan lengths are always static.
    #[inline]
    fn effective_scan_length(
        length: &Dimension,
        _input_types: &[Self],
        _carry_count: usize,
    ) -> Result<Dimension, TypeError> {
        Ok(length.clone())
    }
}

impl ScanTypeSemantics for ArrayType {
    fn rename_scan_length(length: &Dimension, renaming: &TypeIdentityRenaming<Self::Identity>) -> Dimension {
        length.rename_type_identities(renaming)
    }

    fn validate_scan_body(
        body_input_types: &[Self],
        body_output_types: &[Self],
        carry_count: usize,
        length: &Dimension,
    ) -> Result<(), TypeError> {
        if length.variable().is_some() {
            return Err(TypeError::invalid(format!(
                "homogeneous array {SCAN_OPERATION_NAME} requires a static length but got {length}; use a composite {SCAN_OPERATION_NAME} with a \
                 trailing first-class dimension operand for a dynamic trip count",
            )));
        }
        if carry_count > body_input_types.len() {
            return Err(TypeError::invalid(format!(
                "{} carry count {} exceeds the body input count {}",
                SCAN_OPERATION_NAME,
                carry_count,
                body_input_types.len(),
            )));
        }
        if carry_count > body_output_types.len() {
            return Err(TypeError::invalid(format!(
                "{} carry count {} exceeds the body output count {}",
                SCAN_OPERATION_NAME,
                carry_count,
                body_output_types.len(),
            )));
        }
        check_types!(@same, format!("{SCAN_OPERATION_NAME} body carry"), [
            &body_input_types[..carry_count],
            &body_output_types[..carry_count],
        ]);
        for (index, input_type) in body_input_types.iter().enumerate() {
            check_static_scan_type("input", index, input_type)?;
        }
        for (index, output_type) in body_output_types.iter().enumerate() {
            check_static_scan_type("output", index, output_type)?;
        }
        Ok(())
    }

    fn scan_body_input_types(
        input_types: &[Self],
        body_input_count: usize,
        carry_count: usize,
        length: &Dimension,
    ) -> Result<Vec<Self>, TypeError> {
        check_count!("input", input_types, body_input_count, TypeError);
        if carry_count > body_input_count {
            return Err(TypeError::invalid(format!(
                "{SCAN_OPERATION_NAME} carry count {carry_count} exceeds the body input count {body_input_count}",
            )));
        }
        let mut body_input_types = input_types[..carry_count].to_vec();
        for (index, input_type) in input_types[carry_count..].iter().enumerate() {
            if input_type.rank() == 0 {
                return Err(TypeError::invalid(format!(
                    "{} stacked input {} must have rank at least 1",
                    SCAN_OPERATION_NAME,
                    carry_count + index,
                )));
            }
            let (slice_type, leading_dimension) = input_type.without_dimension(0)?;
            if !length.is_refined_by(&leading_dimension) {
                return Err(TypeError::invalid(format!(
                    "{} stacked input {} must have leading dimension {} but has type {}",
                    SCAN_OPERATION_NAME,
                    carry_count + index,
                    length,
                    input_type,
                )));
            }
            body_input_types.push(slice_type);
        }
        Ok(body_input_types)
    }

    fn infer_scan_output_types(
        body_input_types: &[Self],
        body_output_types: &[Self],
        carry_count: usize,
        length: &Dimension,
        input_types: &[Self],
    ) -> Result<Vec<Self>, TypeError> {
        scan_output_types(body_input_types, body_output_types, carry_count, length, input_types)
    }

    #[inline]
    fn stacked_scan_type(r#type: &Self, length: &Dimension) -> Result<Self, TypeError> {
        Ok(stacked_scan_type(r#type, length))
    }

    fn validate_scan_capture<C: Value<Type = Self>>(
        capture: &C,
        index: usize,
        length: &Dimension,
    ) -> Result<(), TypeError> {
        let capture_type = capture.r#type();
        if capture_type.rank() == 0 || !length.is_refined_by(&capture_type.dimension(0)) {
            return Err(TypeError::invalid(format!(
                "{} capture {} must have leading dimension {} but has type {}",
                SCAN_OPERATION_NAME,
                index,
                length,
                capture_type.as_ref(),
            )));
        }
        Ok(())
    }
}

/// Builds one composite scan boundary from per-iteration body types, preserving carry entries and stacking every
/// trailing array entry along the scan's shape-determined leading axis.
fn composite_scan_boundary_types(
    role: &str,
    body_types: &[ArrayIrType],
    carry_count: usize,
    length: &Dimension,
) -> Result<Vec<ArrayIrType>, TypeError> {
    if carry_count > body_types.len() {
        return Err(TypeError::invalid(format!(
            "{} carry count {} exceeds the body {} count {}",
            SCAN_OPERATION_NAME,
            carry_count,
            role,
            body_types.len(),
        )));
    }
    let mut boundary_types = body_types[..carry_count].to_vec();
    for (index, r#type) in body_types[carry_count..].iter().enumerate() {
        let ArrayIrType::Array(r#type) = r#type else {
            return Err(TypeError::invalid(format!(
                "{} stacked body {} {} must be an array but got {}",
                SCAN_OPERATION_NAME,
                role,
                carry_count + index,
                r#type,
            )));
        };
        boundary_types.push(ArrayIrType::Array(stacked_scan_type(r#type, length)));
    }
    Ok(boundary_types)
}

/// Validates the trailing runtime length operand of a dynamic-length composite scan against the declared `length` and
/// the actual stacked operand types, and returns the concrete trip count when the operand only refines `length`'s
/// bounds instead of carrying its nominal identity.
///
/// An operand that carries `length`'s own [`DimensionVariable`](crate::arrays::DimensionVariable) defines exactly the
/// runtime extent that every stacked axis typed `length` has, so the scan is consistent by construction and [`None`]
/// is returned. An operand of an unrelated identity is admissible only when its bounds pin exactly one extent inside
/// `length`'s bounds. The trip count is then that extent, which agrees with the stacked operands only if each of them
/// is itself refined to the very same extent: a stacked axis left symbolic would be read `extent` times regardless of
/// its actual runtime size, over-reading the stacks that turn out shorter and silently truncating the longer ones.
///
/// This is the single definition of the runtime-length safety rule. The values space calls it with the types of the
/// scan's actual operands, so both spaces produce identical diagnostics.
///
/// # Parameters
///   - `length`: Declared scan length.
///   - `input_types`: All scan input types, whose last entry is the runtime length operand when `length` is dynamic.
///   - `carry_count`: Number of leading loop-carried inputs, which are not stacked.
///   - `stacked_input_end`: Exclusive end of the stacked input range, i.e. the index of the runtime length operand.
pub(crate) fn validate_scan_runtime_length<T: std::borrow::Borrow<ArrayIrType>>(
    length: &Dimension,
    input_types: &[T],
    carry_count: usize,
    stacked_input_end: usize,
) -> Result<Option<usize>, TypeError> {
    let Some(variable) = length.variable() else {
        return Ok(None);
    };
    let runtime_length_type = <&DimensionType>::try_from(input_types.last().unwrap().borrow())?;
    if runtime_length_type.variable() == variable {
        return Ok(None);
    }
    let extent = runtime_length_type
        .extent()
        .filter(|_| DimensionType::new(variable.clone()).is_refined_by(runtime_length_type))
        .ok_or_else(|| {
            TypeError::invalid(format!(
                "`{SCAN_OPERATION_NAME}` runtime length operand has type {runtime_length_type} but {SCAN_OPERATION_NAME} length \
                 requires {variable}",
            ))
        })?;
    for (index, r#type) in input_types[carry_count..stacked_input_end].iter().enumerate() {
        let r#type = r#type.borrow();
        let leading_extent = match r#type {
            ArrayIrType::Array(r#type) if r#type.rank() > 0 => {
                let bounds = r#type.dimension(0).bounds();
                (bounds.lower().checked_add(1) == bounds.upper()).then_some(bounds.lower())
            }
            _ => None,
        };
        if leading_extent != Some(extent) {
            return Err(TypeError::invalid(format!(
                "`{}` runtime length operand has type {} but stacked input {} has type {} whose leading dimension is \
                 not refined to extent {}",
                SCAN_OPERATION_NAME,
                runtime_length_type,
                carry_count + index,
                r#type,
                extent,
            )));
        }
    }
    Ok(Some(extent))
}

impl ScanTypeSemantics for ArrayIrType {
    fn rename_scan_length(length: &Dimension, renaming: &TypeIdentityRenaming<Self::Identity>) -> Dimension {
        length.rename_type_identities(renaming)
    }

    fn validate_scan_body(
        body_input_types: &[Self],
        body_output_types: &[Self],
        carry_count: usize,
        length: &Dimension,
    ) -> Result<(), TypeError> {
        composite_scan_boundary_types("input", body_input_types, carry_count, length)?;
        composite_scan_boundary_types("output", body_output_types, carry_count, length)?;
        check_types!(@same, format!("{SCAN_OPERATION_NAME} body carry"), [
            &body_input_types[..carry_count],
            &body_output_types[..carry_count],
        ]);
        Ok(())
    }

    fn scan_body_input_types(
        input_types: &[Self],
        body_input_count: usize,
        carry_count: usize,
        length: &Dimension,
    ) -> Result<Vec<Self>, TypeError> {
        let runtime_length_count = usize::from(length.variable().is_some());
        check_count!("input", input_types, body_input_count + runtime_length_count, TypeError);
        if carry_count > body_input_count {
            return Err(TypeError::invalid(format!(
                "{SCAN_OPERATION_NAME} carry count {carry_count} exceeds the body input count {body_input_count}",
            )));
        }
        let mut body_input_types = input_types[..carry_count].to_vec();
        for (index, r#type) in input_types[carry_count..body_input_count].iter().enumerate() {
            let Self::Array(r#type) = r#type else {
                return Err(TypeError::invalid(format!(
                    "{} stacked input {} must be an array but got {}",
                    SCAN_OPERATION_NAME,
                    carry_count + index,
                    r#type,
                )));
            };
            if r#type.rank() == 0 {
                return Err(TypeError::invalid(format!(
                    "{} stacked input {} must have rank at least 1",
                    SCAN_OPERATION_NAME,
                    carry_count + index,
                )));
            }
            let (slice_type, leading_dimension) = r#type.without_dimension(0)?;
            if !length.is_refined_by(&leading_dimension) {
                return Err(TypeError::invalid(format!(
                    "{} stacked input {} must have leading dimension {} but has type {}",
                    SCAN_OPERATION_NAME,
                    carry_count + index,
                    length,
                    r#type,
                )));
            }
            body_input_types.push(Self::Array(slice_type));
        }
        validate_scan_runtime_length(length, input_types, carry_count, body_input_count)?;
        Ok(body_input_types)
    }

    fn infer_scan_output_types(
        body_input_types: &[Self],
        body_output_types: &[Self],
        carry_count: usize,
        length: &Dimension,
        input_types: &[Self],
    ) -> Result<Vec<Self>, TypeError> {
        let expected_input_types = composite_scan_boundary_types("input", body_input_types, carry_count, length)?;
        let output_types = composite_scan_boundary_types("output", body_output_types, carry_count, length)?;
        check_types!(@same, format!("{SCAN_OPERATION_NAME} body carry"), [
            &body_input_types[..carry_count],
            &body_output_types[..carry_count],
        ]);
        let runtime_length_count = usize::from(length.variable().is_some());
        check_count!("input", input_types, expected_input_types.len() + runtime_length_count, TypeError);
        for (index, (expected, actual)) in
            expected_input_types.iter().zip(&input_types[..expected_input_types.len()]).enumerate()
        {
            if !expected.is_refined_by(actual) {
                return Err(TypeError::invalid(format!(
                    "{SCAN_OPERATION_NAME} input {index} has type {actual} which is incompatible with the expected type {expected}",
                )));
            }
        }
        // A runtime operand that only refines the declared bounds fixes the trip count to one exact extent, so the
        // stacked boundary axes are inferred at that extent instead of at the still-symbolic declared length. Leaving
        // them symbolic would type a concretely sized result as an independent runtime extent.
        match validate_scan_runtime_length(length, input_types, carry_count, expected_input_types.len())? {
            Some(extent) => {
                composite_scan_boundary_types("output", body_output_types, carry_count, &Dimension::Static(extent))
            }
            None => Ok(output_types),
        }
    }

    fn stacked_scan_type(r#type: &Self, length: &Dimension) -> Result<Self, TypeError> {
        let Self::Array(r#type) = r#type else {
            return Err(TypeError::invalid(format!(
                "{} cannot stack first-class dimension type {}",
                SCAN_OPERATION_NAME, r#type,
            )));
        };
        Ok(Self::Array(stacked_scan_type(r#type, length)))
    }

    fn validate_scan_capture<C: Value<Type = Self>>(
        capture: &C,
        index: usize,
        length: &Dimension,
    ) -> Result<(), TypeError> {
        let capture_type = capture.r#type();
        let Self::Array(capture_type) = capture_type.as_ref() else {
            return Err(TypeError::invalid(format!(
                "{SCAN_OPERATION_NAME} capture {index} must be a stacked array but got {capture_type}",
            )));
        };
        if capture_type.rank() == 0 || !length.is_refined_by(&capture_type.dimension(0)) {
            return Err(TypeError::invalid(format!(
                "{SCAN_OPERATION_NAME} capture {index} must have leading dimension {length} but has type {capture_type}",
            )));
        }
        Ok(())
    }

    fn effective_scan_length(
        length: &Dimension,
        input_types: &[Self],
        carry_count: usize,
    ) -> Result<Dimension, TypeError> {
        // The trailing input is the runtime length operand whenever the declared length is dynamic, so the stacked
        // inputs end one before it.
        let stacked_input_end = input_types.len() - usize::from(length.variable().is_some());
        Ok(match validate_scan_runtime_length(length, input_types, carry_count, stacked_input_end)? {
            Some(extent) => Dimension::Static(extent),
            None => length.clone(),
        })
    }
}

/// Validates the scan contract over the single attached body region interface (the `["body"]` slot) and returns
/// it: the body's first `carry_count` input and output types must agree, every body type must satisfy the type family's
/// scan rules (fully static for [`ArrayType`] and mixed carries with array-only stacks for [`ArrayIrType`]), and the
/// interface is what the scan's boundary types derive from.
fn validated_scan_interface<'i, T: ScanTypeSemantics>(
    region_interfaces: &'i [RegionInterface<T>],
    carry_count: usize,
    length: &Dimension,
) -> Result<&'i RegionInterface<T>, TypeError> {
    check_count!("region", region_interfaces, 1, TypeError);
    let body_interface = &region_interfaces[0];
    T::validate_scan_body(body_interface.input_types(), body_interface.output_types(), carry_count, length)?;
    Ok(body_interface)
}

impl<T, Capture> Operation for ScanOperation<Capture>
where
    T: ScanTypeSemantics,
    Capture: Value<Type = T>,
{
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        SCAN_OPERATION_NAME
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        const { &[RegionSlot::computation("body")] }
    }

    fn infer_region_input_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<Option<Vec<T>>>, TypeError> {
        check_count!("region", region_interfaces, 1, TypeError);
        let body_input_types = T::scan_body_input_types(
            input_types,
            region_interfaces[0].input_types().len(),
            self.carry_count,
            &self.length,
        )?;
        let mut declared_identities = Vec::new();
        for r#type in region_interfaces[0].input_types() {
            declared_identities.extend(r#type.identities().map(|(position, identity)| (position, identity.clone())));
        }
        let mut instantiated_identities = Vec::new();
        for r#type in &body_input_types {
            instantiated_identities
                .extend(r#type.identities().map(|(position, identity)| (position, identity.clone())));
        }
        if declared_identities == instantiated_identities {
            return Ok(vec![None]);
        }
        Ok(vec![Some(body_input_types)])
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        let body_interface = validated_scan_interface(region_interfaces, self.carry_count, &self.length)?;
        let output_types = T::infer_scan_output_types(
            body_interface.input_types(),
            body_interface.output_types(),
            self.carry_count,
            &self.length,
            input_types,
        )?;
        let effective_length = T::effective_scan_length(&self.length, input_types, self.carry_count)?;
        for (index, capture) in self.captures.iter().enumerate() {
            T::validate_scan_capture(capture, index, &effective_length)?;
        }
        Ok(output_types)
    }

    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        vec![OutputRegionProvenance { region_index: 0, output_index }]
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<T::Identity>) -> Result<Self, TypeError> {
        Ok(Self {
            captures: self
                .captures
                .iter()
                .map(|capture| capture.rename_type_identities(renaming))
                .collect::<Result<Vec<_>, _>>()?,
            carry_count: self.carry_count,
            length: T::rename_scan_length(&self.length, renaming),
            reverse: self.reverse,
            unroll: self.unroll,
        })
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, SCAN_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("carry_count", self.carry_count)?;
            operation.field("length", &self.length)?;
            operation.field("reverse", self.reverse)?;
            if self.unroll > 1 {
                operation.field("unroll", self.unroll)?;
            }
            if !self.captures.is_empty() {
                operation.field("captures", format_args!("{}", render_factor_list(&self.captures)))?;
            }
            Ok(())
        })
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
        <C::Type>::interpret_scan(self.carry_count, &self.length, self.reverse, context, driver, inputs)
    }
}

/// Type-family interpretation semantics for [`ScanOperation`], mirroring the `while` module's type-family dispatch.
/// Array-backed scans drive the stacked-slice loop of [`interpret_scan_iterations`], allocating output stacks from the
/// body interface's slice types so zero-trip scans still shape their outputs.
pub(crate) trait ScanInterpretation<C: Domain<Type = Self>>: ScanTypeSemantics {
    /// Interprets one scan over the attached body region; refer to the documentation of
    /// [`InterpretableOperation::interpret`] for the contract.
    fn interpret_scan<D: InterpretationDriver<C>>(
        carry_count: usize,
        length: &Dimension,
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
        length: &Dimension,
        reverse: bool,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        let length = length.value().ok_or_else(|| ProgramError::UnsupportedOperation {
            message: format!(
                "cannot eagerly interpret homogeneous array {SCAN_OPERATION_NAME} with dynamic length {length} without an explicit \
                 first-class dimension operand",
            ),
        })?;
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
                TypeError::invalid(format!(
                    "{SCAN_OPERATION_NAME} iteration extraction requires a static stacked type but got {stack_type}"
                ))
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
    iteration_value.reshape(Shape::new(dimensions[1..].iter().map(|&dimension| Dimension::Static(dimension)).collect()))
}

/// Writes `value` as slice `iteration` of `accumulator` along its leading axis, prepending a unit axis to `value`
/// first.
pub(crate) fn write_scan_iteration<V>(accumulator: V, iteration: usize, value: V) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + UpdateSlice + Reshape,
{
    let value_type = value.r#type().into_owned();
    let mut dimensions = Vec::with_capacity(value_type.rank() + 1);
    dimensions.push(Dimension::Static(1));
    dimensions.extend(value_type.shape().dimensions().iter().cloned());
    let expanded = value.reshape(Shape::new(dimensions))?;
    let mut start_indices = vec![0; value_type.rank() + 1];
    start_indices[0] = iteration;
    accumulator.update_slice(&expanded, start_indices.as_slice())
}

/// Partial-evaluation override for a [`ScanOperation`].
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
/// [`Constant`](crate::ValueResolution::Constant) in the known-side context is never an invariant
/// candidate: its value could not be embedded into the rebuilt body as a
/// constant, and skipping it also keeps the fixed point's probe rounds from folding symbolic known work into a live
/// staging context. Under a staging known-side context, the surviving equality check is [`Tracer`]
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
    V: Value<Type: ScanTypeSemantics + TemporalResidualType>,
    C: Context<Type = V::Type, Constant = V, Operation = O>,
    C::Value: PartialEq,
    O: Operation<Type = V::Type> + From<ScanOperation<V>> + TemporalResidualOperation<V::Type>,
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

        // A zero-length scan runs no iteration, so probing the body for carry invariance below could execute (or
        // stage) body work — and surface its errors — for iterations that never run; the scan residualizes
        // unchanged instead.
        if self.length.value() == Some(0) {
            return context.fold_or_residualize(O::from(self.clone()), vec![body.to_program()], inputs);
        }

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

        // A carry can only fold if its init input is known *and* resolves to a constant in the known-side context: the
        // folded value must be embeddable as a rebuilt-body constant, and skipping symbolic knowns also keeps the
        // fixed point's probe rounds from folding symbolic known work into a live staging context.
        let carry_inits = (0..carry_count)
            .map(|index| {
                inputs[index].as_known().filter(|value| context.parent().resolve(value).is_constant()).cloned()
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
        // is only possible when they all resolve to constants — under a staging known-side context the probe can fold
        // a constant-only chain into a live-trace tracer — so a non-constant probe takes the same fallback.
        if (invariant.iter().all(|folded| !folded)
            && body_evaluation.program.instructions().len() >= body.instructions().len())
            || !context.all_knowns_are_constants(&body_evaluation)
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

        let scan = ScanOperation::<V>::new(carry_count, self.length.clone())
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
/// therefore requires a constant-resolved, value-equal init), known-ness needs neither constant resolution nor value
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
    V: Value<Type: ScanTypeSemantics + TemporalResidualType>,
    C: Context<Type = V::Type, Constant = V, Operation = O>,
    O: Operation<Type = V::Type> + From<ScanOperation<V>> + TemporalResidualOperation<V::Type>,
    PartitionRegion: FnMut(&[bool]) -> Result<PartitionedProgram<V, O>, ProgramError>,
{
    let carry_count = scan.carry_count;
    let body_input_types = body.input_types();
    let body_output_count = body.output_types().len();
    let runtime_length_count = usize::from(scan.length.variable().is_some());
    check_count!("input", inputs, body_input_types.len() + runtime_length_count, ProgramError);
    let (body_inputs, runtime_length_inputs) = inputs.split_at(body_input_types.len());
    let input_known = body_inputs.iter().map(PartialEvaluationValue::is_known).collect::<Vec<bool>>();

    // Fixed point over carry known-ness, each round partitioning the borrowed body through a fresh staging context.
    // Unlike the body's derived forward-mode and transposed programs, a partition is not retained by the body region's
    // transform cache: it carries known outputs that are values of the live parent context, so the body and the
    // known-ness mask alone do not determine it.
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
            "{SCAN_OPERATION_NAME} body partition reported known input indices {known_input_indices:?} but expected \
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
    let mut known_program_output_edges = Vec::with_capacity(known_program.output_ids().len());
    let mut known_carry_output_positions = vec![None; carry_count];
    for index in 0..carry_count {
        if carry_known[index] {
            match &partition_outputs[index] {
                PartialEvaluationOutput::Known(output) => {
                    known_carry_output_positions[index] = Some(known_program_output_indices.len());
                    known_program_output_indices.push(*output);
                    known_program_output_edges.push(None);
                }
                PartialEvaluationOutput::Unknown(_) => {
                    return Err(ProgramError::MalformedProgram(format!(
                        "{SCAN_OPERATION_NAME} known-ness fixed point converged with an unknown next value for a known carry"
                    )));
                }
            }
        }
    }
    let mut known_y_output_positions = vec![None; body_output_count - carry_count];
    for (position, output) in partition_outputs[carry_count..].iter().enumerate() {
        if let PartialEvaluationOutput::Known(output) = output {
            known_y_output_positions[position] = Some(known_program_output_indices.len());
            known_program_output_indices.push(*output);
            known_program_output_edges.push(None);
        }
    }
    let mut edge_types = Vec::new();
    let mut edge_carry_sources = Vec::new();
    let mut feeder_edge_positions = Vec::with_capacity(residual_inputs.len());
    for input in residual_inputs.iter() {
        match input {
            PartialEvaluationInput::Known(edge) => {
                if *edge != edge_types.len() {
                    return Err(ProgramError::MalformedProgram(format!(
                        "{SCAN_OPERATION_NAME} body partition reported residual edge {edge} out of order",
                    )));
                }
                let output = known_result_count + edge;
                let output_type = known_program_output_types.get(output).ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "{SCAN_OPERATION_NAME} body partition residual edge {edge} has no known-program output",
                    ))
                })?;
                let carry_source = (0..carry_count).find(|&index| {
                    carry_known[index]
                        && matches!(
                            &partition_outputs[index],
                            PartialEvaluationOutput::Known(carry)
                                if known_program.output_ids()[*carry] == known_program.output_ids()[output]
                        )
                });
                edge_types.push(output_type.clone());
                edge_carry_sources.push(carry_source);
                if carry_source.is_some() {
                    feeder_edge_positions.push(None);
                } else {
                    feeder_edge_positions.push(Some((*edge, known_program_output_indices.len())));
                    known_program_output_indices.push(output);
                    known_program_output_edges.push(Some(*edge));
                }
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
                        "{SCAN_OPERATION_NAME} body partition output {index} references missing known-program output {output}",
                    ))
                })?;
                instantiated_edge_positions[index] = Some((edge_types.len(), known_program_output_indices.len()));
                let edge = edge_types.len();
                edge_types.push(output_type.clone());
                edge_carry_sources.push(None);
                known_program_output_indices.push(*output);
                known_program_output_edges.push(Some(edge));
            }
        }
    }
    let mut invariant_carry_sources = Vec::new();
    let edge_invariant_carry_positions = edge_carry_sources
        .iter()
        .map(|source| {
            source.map(|source| {
                invariant_carry_sources.iter().position(|&candidate| candidate == source).unwrap_or_else(|| {
                    invariant_carry_sources.push(source);
                    invariant_carry_sources.len() - 1
                })
            })
        })
        .collect::<Vec<_>>();
    let edge_storage_types = edge_types
        .iter()
        .zip(&edge_invariant_carry_positions)
        .map(
            |(edge_type, invariant_position)| {
                if invariant_position.is_some() { Ok(edge_type.clone()) } else { edge_type.temporal_storage_type() }
            },
        )
        .collect::<Result<Vec<_>, TypeError>>()?;

    // An empty known side means the split folds nothing; residualize unchanged through the default rule.
    if known_program_output_indices.is_empty() {
        return context.fold_or_residualize(O::from(scan.clone()), vec![body.to_program()], inputs);
    }

    // Bind the known scan into the enclosing known-side context over the original known inputs.
    let known_carry_count = carry_known.iter().filter(|&&known| known).count();
    let known_scan_inputs = known_input_indices
        .iter()
        .map(|&index| {
            body_inputs.get(index).cloned().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "{SCAN_OPERATION_NAME} body partition references missing {SCAN_OPERATION_NAME} input {index}",
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut known_scan_inputs = known_scan_inputs;
    known_scan_inputs.extend_from_slice(runtime_length_inputs);
    let mut known_body_builder = ProgramBuilder::<V, O>::new();
    let known_body_inputs = known_program
        .input_types()
        .into_iter()
        .map(|input_type| known_body_builder.add_input(input_type))
        .collect::<Vec<_>>();
    let known_program_outputs = known_body_builder.splice_program(&known_program, known_body_inputs.as_slice())?;
    let known_output_atoms = known_program_output_indices
        .iter()
        .zip(&known_program_output_edges)
        .map(|(&output, &edge)| -> Result<AtomId, ProgramError> {
            let output = known_program_outputs.get(output).copied().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "{SCAN_OPERATION_NAME} body partition references missing known-program output {output}",
                ))
            })?;
            let Some(edge) = edge else { return Ok(output) };
            let Some(operation) = O::residual_to_storage(&edge_types[edge])? else { return Ok(output) };
            let converted = known_body_builder.add_instruction(operation, Vec::new(), vec![output])?;
            check_count!("output", converted, 1, ProgramError);
            Ok(converted[0])
        })
        .collect::<Result<Vec<_>, _>>()?;
    let known_body = known_body_builder.build::<Vec<V>, Vec<V>>(
        known_output_atoms,
        vec![Placeholder; known_body_inputs.len()],
        vec![Placeholder; known_program_output_indices.len()],
    )?;
    let known_scan = ScanOperation::<V>::new(known_carry_count, scan.length.clone())
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
        let invariant_carry_atoms = invariant_carry_sources
            .iter()
            .map(|&source| builder.add_input(body_input_types[source].clone()))
            .collect::<Vec<_>>();
        let mut unknown_body_input_atoms = vec![None; body_input_types.len()];
        for (index, input_type) in body_input_types.iter().enumerate() {
            let known = if index < carry_count { carry_known[index] } else { input_known[index] };
            if !known {
                unknown_body_input_atoms[index] = Some(builder.add_input(input_type.clone()));
            }
        }
        let mut restored_identity_edges = Vec::new();
        let mut edge_input_atoms = Vec::with_capacity(edge_types.len());
        for (edge, edge_type) in edge_types.iter().enumerate() {
            if let Some(position) = edge_invariant_carry_positions[edge] {
                edge_input_atoms.push(invariant_carry_atoms[position]);
                continue;
            }
            let storage = builder.add_input(edge_storage_types[edge].clone());
            let Some(operation) = O::residual_from_storage(edge_type)? else {
                edge_input_atoms.push(storage);
                continue;
            };

            // A residual partition may expose the same identity-bearing value through multiple feeder edges. Restore
            // that value once so the generated region forwards one SSA definition instead of redefining the nominal
            // identity for every use.
            let defines_identity =
                edge_type.identities().any(|(position, _)| position == TypeIdentityPosition::Definition);
            if defines_identity
                && let Some((_, restored)) =
                    restored_identity_edges.iter().find(|(restored_type, _)| restored_type == edge_type)
            {
                edge_input_atoms.push(*restored);
                continue;
            }
            let restored = builder.add_instruction(operation, Vec::new(), vec![storage])?;
            check_count!("output", restored, 1, ProgramError);
            edge_input_atoms.push(restored[0]);
            if defines_identity {
                restored_identity_edges.push((edge_type.clone(), restored[0]));
            }
        }

        let mut spliced_inputs = Vec::with_capacity(residual_inputs.len());
        for input in residual_inputs.iter() {
            match input {
                PartialEvaluationInput::Unknown(index) => {
                    spliced_inputs.push(unknown_body_input_atoms.get(*index).copied().flatten().ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "{SCAN_OPERATION_NAME} known-ness split saw a residual feeder for a known body input"
                        ))
                    })?);
                }
                PartialEvaluationInput::Known(edge) => {
                    spliced_inputs.push(*edge_input_atoms.get(*edge).ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "{SCAN_OPERATION_NAME} known-ness split lost a residual edge"
                        ))
                    })?)
                }
            }
        }
        let spliced_outputs = builder.splice_program(&residual_program, &spliced_inputs)?;

        let mut unknown_output_atoms = invariant_carry_atoms.clone();
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
                        ProgramError::MalformedProgram(format!(
                            "{SCAN_OPERATION_NAME} known-ness split lost an instantiated carry edge"
                        ))
                    })?;
                    unknown_output_atoms.push(edge_input_atoms[edge]);
                }
            }
        }

        let unknown_body_input_count = invariant_carry_atoms.len()
            + unknown_body_input_atoms.iter().filter(|atom| atom.is_some()).count()
            + edge_invariant_carry_positions.iter().filter(|position| position.is_none()).count();
        let unknown_output_count = unknown_output_atoms.len();
        let unknown_body = builder.build::<Vec<V>, Vec<V>>(
            unknown_output_atoms,
            vec![Placeholder; unknown_body_input_count],
            vec![Placeholder; unknown_output_count],
        )?;
        let unknown_carry_count = invariant_carry_atoms.len() + carry_known.iter().filter(|&&known| !known).count();
        let unknown_scan = ScanOperation::<V>::new(unknown_carry_count, scan.length.clone())
            .with_reverse(scan.reverse)
            .with_unroll(scan.unroll)?;

        // The unknown scan consumes the unknown original inputs followed by one stacked edge per residual edge, each
        // edge fed by the known scan's matching stacked output.
        let mut unknown_scan_inputs =
            invariant_carry_sources.iter().map(|&source| body_inputs[source].clone()).collect::<Vec<_>>();
        for (index, input) in body_inputs.iter().enumerate() {
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
                    format!("{SCAN_OPERATION_NAME} known-ness split known {SCAN_OPERATION_NAME} produced no output for a residual edge"),
                )
            })?);
        }
        unknown_scan_inputs.extend_from_slice(runtime_length_inputs);
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
                        format!("{SCAN_OPERATION_NAME} known-ness split known {SCAN_OPERATION_NAME} produced no output for a known result"),
                    )
                });
            }
            let ordinal = unknown_output_ordinals[index].ok_or_else(|| {
                ProgramError::MalformedProgram(
                    format!("{SCAN_OPERATION_NAME} known-ness split produced a result owned by neither side"),
                )
            })?;
            residual_outputs.get(ordinal).cloned().ok_or_else(|| {
                ProgramError::MalformedProgram(
                    format!("{SCAN_OPERATION_NAME} known-ness split unknown {SCAN_OPERATION_NAME} produced no output for a residual result"),
                )
            })
        })
        .collect()
}

/// Batching rule for [`ScanOperation`]. Under a *staging* parent, a capture-free scan is batched *structurally*,
/// staging one batched scan into the enclosing trace (the shape of JAX's `_scan_batching_rule`), so the batched
/// program's size stays independent of the trip count:
///
///   1. Every batched carry init is realigned to batch axis 0, and every stacked input whose batch axis would
///      displace the leading scan dimension is realigned to batch axis 1, so per-iteration slices keep their batch
///      placement when the leading scan dimension is dropped.
///   2. The body is batched at `[carry_axes..., slice_axes...]` and the carry axes are iterated to a fixed point: a
///      scan's carry types are loop-invariant, so a replicated carry whose next-carry output is batched *becomes*
///      batched, and the rule widens that carry's input axis and re-batches until the body is axis-invariant (the
///      iteration count is bounded by the carry count because every non-final pass widens at least one carry —
///      JAX's `carry_bat` fixed point). The body's outputs are then instantiated at the joined axes
///      ([`ProgramBatchingOutputAxesPolicy::AlignEachTo`], mirroring JAX's `instantiate=carry_bat`), reusing the
///      stabilizing pass's own program when its natural axes already are those joined axes.
///   3. Widened parent carry inits gain their batch axis through staged broadcasts, and one [`ScanOperation`] over
///      the batched body is bound into the parent with the same carry count, length, `reverse`, and (lowering-only)
///      `unroll` factor. Final carries come back at the carry axes, and stacked outputs at their per-iteration axes
///      shifted right by the new leading scan dimension. The staged stacked outputs carry the scan's *declared*
///      output types, whose optional sharding metadata is left for sharding propagation to resolve (the
///      `scan_output_types` contract).
///
/// Under an *eager* parent — and for *captured* linear scans under any parent, whose bodies read scan-local capture
/// references that a structurally batched body cannot re-slice — the scan loop is instead replayed per iteration
/// through `batch_scan_with_interpreter`, with each body instruction re-entering this operation family's batching
/// rules against the same active context. This is the operational path eager batched scans execute either way, and
/// its packed stacked accumulators retain per-item placement metadata exactly. Constants lift and stacked-output
/// accumulators seed (via the parent's [`Zero`]) through `context.parent()`.
impl<C, P: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>> for ScanOperation<C::Constant>
where
    C: Context<Type = ArrayType> + Zero<<C as Domain>::Value>,
    <C as Domain>::Value: Broadcast + Transpose + Slice + UpdateSlice + Reshape,
    C::Operation: ZeroOperationProvider<ArrayType>
        + From<BroadcastOperation>
        + From<TransposeOperation>
        + From<SliceOperation>
        + From<UpdateSliceOperation>
        + From<ReshapeOperation>
        + From<ScanOperation<C::Constant>>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<P>>, BatchingError> {
        let body = driver.region(0)?;
        let carry_count = self.carry_count();
        if self.captures().is_empty() && !context.parent().is_eager() {
            check_count!("input", inputs, body.input_types().len(), ProgramError);
            let body_output_count = body.output_types().len();

            // Realign batched carries to batch axis 0 and batched stacks off the leading scan dimension, so the
            // fixed point below only ever distinguishes replicated from batched-at-0 carries and every
            // per-iteration slice keeps its batch placement when the leading scan dimension is dropped.
            let mut carries =
                inputs[..carry_count].iter().map(|input| input.move_axis(0)).collect::<Result<Vec<_>, _>>()?;
            let stacks = inputs[carry_count..]
                .iter()
                .map(
                    |input| if input.batch_axis_position() == Some(0) { input.move_axis(1) } else { Ok(input.clone()) },
                )
                .collect::<Result<Vec<_>, _>>()?;
            let mut carry_axes = carries.iter().map(ArrayBatch::batch_axis).collect::<Vec<_>>();
            let slice_axes =
                stacks.iter().map(|stack| scan_iteration_batch_axis(stack.batch_axis())).collect::<Vec<_>>();

            // Iterate the carry batch axes to a fixed point (bounded by the carry count; see the rule doc). Each
            // pass discovers the body's natural output axes; the pass that widens nothing determines the stacked
            // outputs' per-iteration axes.
            let mut stabilized = None;
            for _ in 0..=carry_count {
                let mut iteration_axes = carry_axes.clone();
                iteration_axes.extend(slice_axes.iter().copied());
                let candidate = driver.batch_program(
                    context,
                    body,
                    iteration_axes.as_slice(),
                    ProgramBatchingOutputAxesPolicy::Natural,
                )?;
                check_count!("output", candidate.output_axes(), body_output_count, ProgramError);
                let mut widened = false;
                for (carry_axis, output_axis) in carry_axes.iter_mut().zip(candidate.output_axes()) {
                    if carry_axis.is_replicated() && !output_axis.is_replicated() {
                        *carry_axis = BatchAxis::new(0);
                        widened = true;
                    }
                }
                if !widened {
                    let y_axes = candidate.output_axes()[carry_count..].to_vec();
                    stabilized = Some((candidate, y_axes));
                    break;
                }
            }
            let Some((stabilized_body, y_axes)) = stabilized else {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "{SCAN_OPERATION_NAME} batching failed to stabilize the carry batch axes within {carry_count} widening passes",
                    ),
                });
            };

            // Instantiate the body's outputs at the joined axes so its next-carry outputs align with its carry
            // inputs across iterations. The stabilizing pass already used these input axes, so when its discovered
            // (normalized) output axes equal the joined targets it *is* the aligned program and is kept as-is.
            let mut iteration_axes = carry_axes.clone();
            iteration_axes.extend(slice_axes.iter().copied());
            let mut target_axes = carry_axes.clone();
            target_axes.extend(y_axes.iter().copied());
            let batched_body = context.align_batched_program_outputs(
                driver,
                body,
                iteration_axes.as_slice(),
                stabilized_body,
                target_axes.as_slice(),
            )?;

            // Widen the parent carry inits whose elements became batched (their batch axis is materialized through
            // a staged broadcast) and stage one batched scan over the batched body.
            for (carry, carry_axis) in carries.iter_mut().zip(carry_axes.iter()) {
                if !carry_axis.is_replicated() && carry.batch_axis().is_replicated() {
                    *carry = carry.broadcast(0, P::axis_size(context)?, context.axis_sharding().clone())?;
                }
            }
            let batched_scan = ScanOperation::<C::Constant>::new(carry_count, self.length())
                .with_reverse(self.reverse())
                .with_unroll(self.unroll())?;
            let mut values = carries.iter().map(|carry| carry.value().clone()).collect::<Vec<_>>();
            values.extend(stacks.iter().map(|stack| stack.value().clone()));
            let outputs = context.parent().bind(batched_scan, vec![batched_body], &values)?;
            check_count!("output", outputs, carry_count + y_axes.len(), ProgramError);

            // Final carries come back at the carry axes; each stacked output gains the leading scan dimension,
            // shifting its per-iteration batch axis right by one.
            let mut output_axes = carry_axes;
            output_axes.extend(y_axes.iter().map(|axis| match axis.axis() {
                Some(axis) => BatchAxis::new(axis.value() + 1),
                None => BatchAxis::replicated(),
            }));
            return Ok(outputs
                .into_iter()
                .zip(output_axes)
                .map(|(output, axis)| ArrayBatch::new(output, axis))
                .collect::<Result<Vec<_>, _>>()?
                .into());
        }

        if self.length().value() == Some(0) {
            check_count!("input", inputs, body.input_types().len(), ProgramError);

            // No iteration executes, but batching the body structurally still determines which per-iteration outputs
            // are mapped and where their packed batch dimensions live. Stacked inputs lose their per-item leading
            // scan dimension before entering the body, so their batch axes must be adjusted in the same way as an
            // actual iteration slice.
            let mut iteration_input_axes =
                inputs[..self.carry_count()].iter().map(ArrayBatch::batch_axis).collect::<Vec<_>>();
            iteration_input_axes
                .extend(inputs[self.carry_count()..].iter().map(|input| scan_iteration_batch_axis(input.batch_axis())));
            let (batched_body, output_axes) = driver
                .batch_program(
                    context,
                    body,
                    iteration_input_axes.as_slice(),
                    ProgramBatchingOutputAxesPolicy::Natural,
                )?
                .into_parts();
            let output_types = batched_body.output_types();
            check_count!("output", output_axes, output_types.len(), ProgramError);
            if output_types.len() < self.carry_count() {
                return Err(ProgramError::MalformedProgram(format!(
                    "{} body has {} outputs but carry count is {}",
                    SCAN_OPERATION_NAME,
                    output_types.len(),
                    self.carry_count(),
                ))
                .into());
            }

            // A zero-length scan returns its initial carries unchanged. Its stacked outputs are empty arrays whose
            // packed element types and batch axes come from the structurally batched body. Inserting the leading
            // scan dimension shifts every mapped output axis right by one while preserving its placement metadata.
            let mut outputs = inputs[..self.carry_count()].to_vec();
            for (output_type, output_axis) in
                output_types.into_iter().zip(output_axes.into_iter()).skip(self.carry_count())
            {
                let stacked_type = output_type.with_inserted_dimension(0, Dimension::Static(0))?;
                let stacked_axis = match output_axis.axis() {
                    Some(axis) => BatchAxis::new(axis.value() + 1),
                    None => BatchAxis::replicated(),
                };
                let stacked_value = context.parent().zero(&stacked_type)?;
                outputs.push(ArrayBatch::new(stacked_value, stacked_axis)?);
            }
            return Ok(outputs.into());
        }

        let y_slice_types = body.output_types().split_off(self.carry_count());
        let length = self.length().value().ok_or_else(|| BatchingError::UnsupportedOperation {
            message: format!(
                "eager homogeneous {} batching requires a concrete trip count but got {}",
                SCAN_OPERATION_NAME,
                self.length(),
            ),
        })?;
        Ok(batch_scan_with_interpreter(
            self.carry_count(),
            length,
            self.reverse(),
            y_slice_types.as_slice(),
            inputs,
            |stacked_type| context.parent().zero(stacked_type),
            |_, iteration_inputs| driver.batch_region(context, 0, iteration_inputs),
        )?
        .into())
    }
}

/// Drives one batched scan loop over `[carry..., stacked_xs...]` input batches, delegating each iteration's body
/// evaluation to `interpret_iteration` and allocating stacked output accumulators through `allocate_zero`.
///
/// Per-iteration slices of the stacked inputs are read along their *per-item* leading axis (see
/// [`read_scan_iteration_batch`]) so the batch axis threads through untouched, and the per-iteration outputs are
/// stacked along a fresh leading axis, shifting each output's batch axis right by one. The visit order
/// reverses when `reverse` is `true` while output slice `i` stays aligned with input slice `i`, exactly like the
/// unbatched scan loop.
pub(crate) fn batch_scan_with_interpreter<V, AllocateZeroFn, InterpretIterationFn>(
    carry_count: usize,
    length: usize,
    reverse: bool,
    y_slice_types: &[ArrayType],
    inputs: &[ArrayBatch<V>],
    mut allocate_zero: AllocateZeroFn,
    mut interpret_iteration: InterpretIterationFn,
) -> Result<Vec<ArrayBatch<V>>, BatchingError>
where
    V: Value<Type = ArrayType> + Slice + UpdateSlice + Reshape,
    AllocateZeroFn: FnMut(&ArrayType) -> Result<V, ProgramError>,
    InterpretIterationFn: FnMut(usize, Vec<ArrayBatch<V>>) -> Result<Vec<ArrayBatch<V>>, BatchingError>,
{
    let (initial_carries, stacks) = inputs.split_at(carry_count);
    let mut carries = initial_carries.to_vec();
    let mut accumulators: Vec<Option<ScanOutputAccumulator<V>>> = (0..y_slice_types.len()).map(|_| None).collect();
    let mut iterations: Vec<usize> = (0..length).collect();
    if reverse {
        iterations.reverse();
    }
    for iteration in iterations {
        let mut iteration_inputs = carries.clone();
        for stack in stacks {
            iteration_inputs.push(read_scan_iteration_batch(stack, iteration)?);
        }
        let mut iteration_outputs = interpret_iteration(iteration, iteration_inputs)?;
        check_count!("output", iteration_outputs, carry_count + y_slice_types.len(), ProgramError);
        let iteration_ys = iteration_outputs.split_off(carry_count);
        carries = iteration_outputs;
        for (accumulator, iteration_y) in accumulators.iter_mut().zip(iteration_ys.into_iter()) {
            let batch_axis = iteration_y.batch_axis_position();
            let iteration_type = iteration_y.r#type().into_owned();
            let accumulator = match accumulator {
                Some(accumulator) => {
                    if accumulator.batch_axis != batch_axis {
                        return Err(BatchingError::MisalignedBatchAxes {
                            message: format!(
                                "{SCAN_OPERATION_NAME} body produced stacked output iterations at mismatched batch axes ({:?} vs \
                                 {batch_axis:?})",
                                accumulator.batch_axis,
                            ),
                        });
                    }
                    accumulator
                }
                None => accumulator.insert(ScanOutputAccumulator {
                    // Unlike the scan operation's unbatched signature helper, this packed accumulator must retain
                    // the iteration value's mapped-dimension placement. The newly inserted scan dimension itself is
                    // replicated.
                    accumulator: allocate_zero(&iteration_type.with_inserted_dimension(0, Dimension::Static(length))?)?,
                    batch_axis,
                }),
            };
            let mut expanded_dimensions = Vec::with_capacity(iteration_type.rank() + 1);
            expanded_dimensions.push(Dimension::Static(1));
            expanded_dimensions.extend(iteration_type.shape().dimensions().iter().cloned());
            let expanded = iteration_y.into_value().reshape(Shape::new(expanded_dimensions))?;
            let mut start_indices = vec![0; iteration_type.rank() + 1];
            start_indices[0] = iteration;
            accumulator.accumulator = accumulator.accumulator.update_slice(&expanded, start_indices.as_slice())?;
        }
    }
    let mut outputs = carries;
    for (accumulator, y_slice_type) in accumulators.into_iter().zip(y_slice_types.iter()) {
        match accumulator {
            Some(ScanOutputAccumulator { accumulator, batch_axis }) => {
                outputs.push(ArrayBatch::new(
                    accumulator,
                    BatchAxis::from_optional_position(batch_axis.map(|axis| axis + 1)),
                )?);
            }
            None => {
                // A zero-length scan writes no iterations, so each stacked output is the replicated empty stack of
                // the body's per-iteration output type.
                let stacked_type = stacked_scan_type(y_slice_type, length);
                outputs.push(ArrayBatch::replicated(allocate_zero(&stacked_type)?));
            }
        }
    }
    Ok(outputs)
}

/// Per-output stacking state used by [`batch_scan_with_interpreter`]: the accumulator batch holding the iterations
/// written so far, together with the batch axis every iteration must agree on.
struct ScanOutputAccumulator<V: Typed<Type = ArrayType>> {
    /// Stacked accumulator; its leading packed axis is the scan length axis.
    accumulator: V,

    /// Batch axis of the per-item values written into the accumulator, if the output is batch-varying.
    batch_axis: Option<usize>,
}

/// Extracts slice `iteration` of a stacked batch along its *per-item* leading axis and drops that axis.
///
/// The per-item leading axis is the scan length axis: packed axis `1` when the batch axis sits at packed axis
/// `0`, and packed axis `0` otherwise. The iteration batch keeps the input's batch axis, decremented when it sat
/// after the dropped axis.
fn read_scan_iteration_batch<V>(stack: &ArrayBatch<V>, iteration: usize) -> Result<ArrayBatch<V>, BatchingError>
where
    V: Value<Type = ArrayType> + Slice + Reshape,
{
    let stack_axis = match stack.batch_axis_position() {
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
                    message: format!(
                        "{SCAN_OPERATION_NAME} batching requires static stacked input types but got {stack_type}"
                    ),
                }
                .into()
            })
        })
        .collect::<Result<Vec<usize>, ProgramError>>()?;
    let mut start_indices = vec![0; dimensions.len()];
    start_indices[stack_axis] = iteration;
    let mut limit_indices = dimensions.clone();
    limit_indices[stack_axis] = iteration + 1;
    let unit_strides = vec![1; dimensions.len()];
    let iteration_value =
        stack
            .value()
            .clone()
            .slice(start_indices.as_slice(), limit_indices.as_slice(), unit_strides.as_slice())?;
    let iteration_dimensions = dimensions
        .iter()
        .enumerate()
        .filter(|(axis, _)| *axis != stack_axis)
        .map(|(_, &dimension)| Dimension::Static(dimension))
        .collect::<Vec<_>>();
    let iteration_value = iteration_value.reshape(Shape::new(iteration_dimensions))?;
    ArrayBatch::new(iteration_value, scan_iteration_batch_axis(stack.batch_axis()))
}

/// Maps a stacked scan input's packed batch axis to the corresponding per-iteration batch axis after removing the
/// per-item leading scan dimension.
pub(crate) fn scan_iteration_batch_axis(batch_axis: BatchAxis) -> BatchAxis {
    match batch_axis.axis() {
        Some(axis) if axis.value() == 0 => BatchAxis::new(0),
        Some(axis) => BatchAxis::new(axis.value() - 1),
        None => BatchAxis::replicated(),
    }
}

/// Composite array IR batching rule for [`ScanOperation`].
///
/// The rule carries the mapped extent as leading replicated state in the transformed scan. Array carries use the
/// same monotonic mapped-axis fixed point as homogeneous scans, while first-class dimension carries remain
/// replicated. Stacked inputs and outputs are necessarily arrays because one shared dimension value cannot represent
/// a different stacked extent for each batch item.
impl<A, C> BatchableOperation<C, ArrayIrBatching> for ScanOperation<ArrayIrValue<A>>
where
    A: Value<Type = ArrayType>,
    C: Context<
            Type = ArrayIrType,
            Operation: From<DynamicBroadcastOperation>
                           + From<ConstantOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + From<ScanOperation<ArrayIrValue<A>>>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<TransposeOperation>,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatching>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatching>,
        driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatching>, BatchingError> {
        let body = driver.region(0)?;
        let (scan_inputs, runtime_length) = if self.length().variable().is_some() {
            let Some((runtime_length, scan_inputs)) = inputs.split_last() else {
                return Err(
                    ProgramError::InvalidInputCount { expected: body.input_types().len() + 1, actual: 0 }.into()
                );
            };
            runtime_length.validate_replicated_dimension()?;
            (scan_inputs, Some(runtime_length))
        } else {
            (inputs, None)
        };
        check_count!("input", scan_inputs, body.input_types().len(), ProgramError);
        let carry_count = self.carry_count();

        // Canonicalize mapped array carries to the leading axis. Dimension carries remain replicated.
        let mut carries = scan_inputs[..carry_count]
            .iter()
            .cloned()
            .map(|input| match input.unbatched_type() {
                ArrayIrType::Array(_) if !input.batch_axis().is_replicated() => {
                    align_array_batch(context, input, Axis::from(0))
                }
                ArrayIrType::Array(_) => Ok(input),
                ArrayIrType::Dimension(_) => {
                    input.validate_replicated_dimension()?;
                    Ok(input)
                }
            })
            .collect::<Result<Vec<_>, BatchingError>>()?;
        let stacks = scan_inputs[carry_count..]
            .iter()
            .cloned()
            .map(|input| {
                <&ArrayType>::try_from(&input.unbatched_type())?;
                if input.batch_axis_position() == Some(0) {
                    align_array_batch(context, input, Axis::from(1))
                } else {
                    Ok(input)
                }
            })
            .collect::<Result<Vec<_>, BatchingError>>()?;
        let mut carry_axes = carries.iter().map(ArrayIrBatch::batch_axis).collect::<Vec<_>>();
        let slice_axes = stacks.iter().map(|stack| scan_iteration_batch_axis(stack.batch_axis())).collect::<Vec<_>>();

        // Iterate carry axes to a fixed point. A first-class dimension cannot widen because composite batching does
        // not admit mapped dimension values.
        let mut stabilized = None;
        for _ in 0..=carry_count {
            let iteration_axes = carry_axes.iter().chain(slice_axes.iter()).copied().collect::<Vec<_>>();
            let candidate = driver.batch_program(
                context,
                body,
                iteration_axes.as_slice(),
                ProgramBatchingOutputAxesPolicy::Natural,
            )?;
            check_count!("output", candidate.output_axes(), body.output_types().len(), ProgramError);
            let mut widened = false;
            for (index, (carry_axis, output_axis)) in
                carry_axes.iter_mut().zip(candidate.output_axes().iter()).enumerate()
            {
                if carry_axis.is_replicated() && !output_axis.is_replicated() {
                    if matches!(scan_inputs[index].unbatched_type(), ArrayIrType::Dimension(_)) {
                        return Err(BatchingError::MappedDimension {
                            r#type: Box::new(<&DimensionType>::try_from(&scan_inputs[index].unbatched_type())?.clone()),
                            axis: *output_axis,
                        });
                    }
                    *carry_axis = BatchAxis::new(0);
                    widened = true;
                }
            }
            if !widened {
                let output_slice_axes = candidate.output_axes()[carry_count..].to_vec();
                stabilized = Some((candidate, output_slice_axes));
                break;
            }
        }
        let Some((stabilized_body, output_slice_axes)) = stabilized else {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "{SCAN_OPERATION_NAME} batching failed to stabilize the carry batch axes within {carry_count} widening passes",
                ),
            });
        };

        // The stabilizing pass already used these input axes, so when its discovered (normalized) output axes equal
        // the joined targets it *is* the aligned program and is kept as-is instead of being rebuilt.
        let iteration_axes = carry_axes.iter().chain(slice_axes.iter()).copied().collect::<Vec<_>>();
        let target_axes = carry_axes.iter().chain(output_slice_axes.iter()).copied().collect::<Vec<_>>();
        let batched_body = context.align_batched_program_outputs(
            driver,
            body,
            iteration_axes.as_slice(),
            stabilized_body,
            target_axes.as_slice(),
        )?;
        for (carry, axis) in carries.iter_mut().zip(carry_axes.iter()) {
            if !axis.is_replicated() && carry.batch_axis().is_replicated() {
                *carry = align_array_batch(context, carry.clone(), Axis::from(0))?;
            }
        }

        let batched_scan = ScanOperation::<ArrayIrValue<A>>::new(carry_count + 1, self.length())
            .with_reverse(self.reverse())
            .with_unroll(self.unroll())?
            .with_captures(self.captures().to_vec());
        let mut packed_inputs = Vec::with_capacity(inputs.len() + 1);
        packed_inputs.push(context.axis_extent().clone());
        packed_inputs.extend(carries.iter().map(|carry| carry.value().clone()));
        packed_inputs.extend(stacks.iter().map(|stack| stack.value().clone()));
        packed_inputs.extend(runtime_length.map(|runtime_length| runtime_length.value().clone()));
        let mut outputs = context.parent().bind(batched_scan, vec![batched_body], packed_inputs.as_slice())?;
        check_count!("output", outputs, 1 + carry_count + output_slice_axes.len(), ProgramError);
        outputs.remove(0);
        let mut output_axes = carry_axes;
        output_axes.extend(output_slice_axes.iter().map(|axis| match axis.axis() {
            Some(axis) => BatchAxis::new(axis.value() + 1),
            None => BatchAxis::replicated(),
        }));
        Ok(outputs
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| ArrayIrBatch::new(output, axis))
            .collect::<Result<Vec<_>, _>>()?
            .into())
    }
}

/// Capture-free forward-mode (JVP) rule for [`ScanOperation`], staging **one fused** jvp `scan` with compact
/// live-tangent carries and scanned inputs as an ordinary primal-enum `scan` operation over the shared builder.
///
/// The rule builds the body's compact fused jvp program through its instruction-scoped differentiation driver
/// (boundary entries whose tangent type is a zero differential space carry no tangent entry) and permutes its
/// signature into scan order, giving a fused body
/// `[primal_carries..., live(tangent_carries)..., primal_slices..., live(tangent_slices)...] ->
/// [primal_next_carries..., live(tangent_next_carries)..., primal_outputs..., live(tangent_outputs)...]`, and stages
/// one scan whose carries are the primal carries followed by the live tangent carries. Pure forward mode therefore
/// runs a single loop pass and stores **no** per-iteration residual stacks — the JAX jvp-of-`scan` shape.
///
/// The primal/tangent separation that reverse mode needs is deferred to partial evaluation: the known-ness split of
/// [`Program::linearize`](crate::Program::linearize) marks the primal halves known and the tangent halves unknown,
/// and the scan known-ness split (ryft's `_scan_partial_eval` analogue) separates the fused scan into a known
/// primal scan — stacking exactly the per-iteration known→unknown edges the tangent side consumes — and a residual
/// tangent scan over `[tangent_carries..., tangent_slices..., edge_slices...]`, the transposable linear-scan shape.
/// Residual stacks therefore exist only when linearization actually demands them.
impl<C: Context<Type: DifferentiableType + ScanTypeSemantics> + Zero<C::Value>> DifferentiableOperation<C>
    for ScanOperation<C::Constant>
where
    C::Operation: ResidualZeroProvider<C::Type> + From<ScanOperation<C::Constant>>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The rule requests all nested-computation work through its driver (region 0 is the body), which keeps
        // its bounds free of the operation family's own semantic traits.
        let carry_count = self.carry_count();
        let length = self.length();
        let reverse = self.reverse();
        let unroll = self.unroll();

        // The fused body is compact: body inputs and outputs whose tangent type is a zero differential space carry
        // no tangent boundary entry. Derive the liveness masks from the same body boundary types that the fused-body
        // construction filtered on.
        let (input_has_tangent, output_has_tangent) = {
            let body = driver.region(0)?;
            let input_has_tangent = body
                .input_types()
                .iter()
                .map(|r#type| Ok(!r#type.tangent()?.is_zero_space()))
                .collect::<Result<Vec<_>, DifferentiationError>>()?;
            let output_has_tangent = body
                .output_types()
                .iter()
                .map(|r#type| Ok(!r#type.tangent()?.is_zero_space()))
                .collect::<Result<Vec<_>, DifferentiationError>>()?;
            (input_has_tangent, output_has_tangent)
        };
        let body_input_count = input_has_tangent.len();
        let body_output_count = output_has_tangent.len();
        let runtime_length_count = usize::from(length.variable().is_some());
        check_count!("input", inputs, body_input_count + runtime_length_count, ProgramError);
        let (body_inputs, runtime_length_inputs) = inputs.split_at(body_input_count);
        let live_carry_count = input_has_tangent[..carry_count].iter().filter(|&&live| live).count();

        // The fused jvp body is over `[primal_body_inputs..., live(tangent_body_inputs)...]`; permute its compact
        // signature into scan order (carries lead scanned inputs on both the primal and live tangent sides). The
        // unpermuted program comes from the body region's retained transform cache, so a body shared by several
        // programs is differentiated once; the permutation into scan order is a boundary convention of this rule
        // rather than a property of the body, so it is reapplied per use instead of being retained against the body.
        let fused_body = driver.jvp_program(driver.region(0)?)?;
        check_count!(
            "input",
            fused_body.input_types(),
            body_input_count + input_has_tangent.iter().filter(|&&live| live).count(),
            ProgramError,
        );
        let fused_body = permute_live_scan_body(&fused_body, &input_has_tangent, &output_has_tangent, carry_count)?;

        // Stage the fused scan over
        // `[primal_carry_inits..., live(tangent_carry_inits)..., primal_stacks..., live(tangent_stacks)...]`.
        let fused_scan = ScanOperation::<C::Constant>::new(carry_count + live_carry_count, length)
            .with_reverse(reverse)
            .with_unroll(unroll)?;
        // The fused scan takes each live carry and scanned tangent as a real program input, so materialize their
        // structural zeros at this sub-program boundary. Each operand's own primal names every runtime quantity a
        // reference-bearing tangent type omits, because the tangent type derivation preserves geometry exactly.
        let mut operands = Vec::with_capacity(fused_body.input_types().len());
        operands.extend(body_inputs[..carry_count].iter().map(|input| input.primal().clone()));
        for (input, &live) in body_inputs[..carry_count].iter().zip(&input_has_tangent[..carry_count]) {
            if live {
                operands.push(C::Operation::materialize_zero_from_residual_sources(
                    context,
                    input.tangent().clone(),
                    std::iter::once(input.primal()),
                )?);
            }
        }
        operands.extend(body_inputs[carry_count..].iter().map(|input| input.primal().clone()));
        for (input, &live) in body_inputs[carry_count..].iter().zip(&input_has_tangent[carry_count..]) {
            if live {
                operands.push(C::Operation::materialize_zero_from_residual_sources(
                    context,
                    input.tangent().clone(),
                    std::iter::once(input.primal()),
                )?);
            }
        }
        operands.extend(runtime_length_inputs.iter().map(|input| input.primal().clone()));
        let outputs = context.bind(C::Operation::from(fused_scan), vec![fused_body], &operands)?;
        let live_scanned_output_count = output_has_tangent[carry_count..].iter().filter(|&&live| live).count();
        check_count!("output", outputs, body_output_count + live_carry_count + live_scanned_output_count, ProgramError,);

        // The fused scan's outputs are `[primal_final_carries..., live(tangent_final_carries)..., primal_stacked...,
        // live(tangent_stacked)...]`; zip the live halves back into `DifferentiationDual`s in the original output
        // order, restoring structural zeros for the omitted zero-space outputs.
        let scanned_output_count = body_output_count - carry_count;
        let stacked_primals_start = carry_count + live_carry_count;
        let stacked_tangents_start = stacked_primals_start + scanned_output_count;
        let mut jvp_outputs = Vec::with_capacity(body_output_count);
        let mut carry_tangents = outputs[carry_count..stacked_primals_start].iter().cloned();
        for index in 0..carry_count {
            // Scan's carry fixed point makes the carry input and output tangent liveness masks identical.
            jvp_outputs.push(if input_has_tangent[index] {
                DifferentiationDual::new(outputs[index].clone(), carry_tangents.next().unwrap())?
            } else {
                DifferentiationDual::new_with_zero_tangent(outputs[index].clone())?
            });
        }
        let mut stacked_tangents = outputs[stacked_tangents_start..].iter().cloned();
        for index in 0..scanned_output_count {
            jvp_outputs.push(if output_has_tangent[carry_count + index] {
                DifferentiationDual::new(
                    outputs[stacked_primals_start + index].clone(),
                    stacked_tangents.next().unwrap(),
                )?
            } else {
                DifferentiationDual::new_with_zero_tangent(outputs[stacked_primals_start + index].clone())?
            });
        }
        Ok(jvp_outputs)
    }
}

/// Rebuilds a fused JVP scan body so its compact boundary uses scan order instead of JVP order. The liveness masks
/// mark which primal boundary entries carry a tangent entry in the compact fused signature.
fn permute_live_scan_body<V, O>(
    program: &Program<V, O, Vec<V>, Vec<V>>,
    input_has_tangent: &[bool],
    output_has_tangent: &[bool],
    carry_count: usize,
) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError>
where
    V: Value,
    O: Operation<Type = V::Type>,
{
    let input_order = live_scan_signature_permutation(input_has_tangent, carry_count)?;
    let output_order = live_scan_signature_permutation(output_has_tangent, carry_count)?;
    reorder_program_boundary(program, input_order.as_slice(), output_order.as_slice())
}

/// Returns the permutation that converts one side of a compact fused JVP body signature from JVP order
/// (`[primal_entries..., live(tangent_entries)...]`) into scan order, where carries lead the scanned entries on both
/// the primal and live tangent sides:
/// `[primal_carries..., live(tangent_carries)..., primal_scanned..., live(tangent_scanned)...]`. The `has_tangent`
/// mask marks the primal entries whose tangent entry exists in the compact signature; the position of the `k`-th
/// live entry's tangent entry is the number of primal entries plus `k`.
fn live_scan_signature_permutation(has_tangent: &[bool], carry_count: usize) -> Result<Vec<usize>, ProgramError> {
    let entry_count = has_tangent.len();
    if carry_count > entry_count {
        return Err(ProgramError::MalformedProgram(format!(
            "{SCAN_OPERATION_NAME} carry count {carry_count} exceeds fused body signature size {entry_count}",
        )));
    }
    let tangent_positions = has_tangent
        .iter()
        .scan(entry_count, |next_position, &live| {
            let position = live.then_some(*next_position);
            *next_position += usize::from(live);
            Some(position)
        })
        .collect::<Vec<_>>();
    let mut permutation = Vec::with_capacity(entry_count + has_tangent.iter().filter(|&&live| live).count());
    permutation.extend(0..carry_count);
    permutation.extend(tangent_positions[..carry_count].iter().flatten());
    permutation.extend(carry_count..entry_count);
    permutation.extend(tangent_positions[carry_count..].iter().flatten());
    Ok(permutation)
}

/// Rebuilds `program` with a new public boundary order. `input_order` and `output_order` list old boundary positions
/// in the desired new order.
fn reorder_program_boundary<V, O>(
    program: &Program<V, O, Vec<V>, Vec<V>>,
    input_order: &[usize],
    output_order: &[usize],
) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError>
where
    V: Value,
    O: Operation<Type = V::Type>,
{
    fn inverse_order(order: &[usize], length: usize, label: &str) -> Result<Vec<usize>, ProgramError> {
        if order.len() != length {
            return Err(ProgramError::MalformedProgram(format!(
                "{label} permutation has length {} but boundary has length {length}",
                order.len(),
            )));
        }
        let mut inverse = vec![None; length];
        for (new_position, &old_position) in order.iter().enumerate() {
            let Some(slot) = inverse.get_mut(old_position) else {
                return Err(ProgramError::MalformedProgram(format!(
                    "{label} permutation references out-of-range position {old_position}",
                )));
            };
            if slot.is_some() {
                return Err(ProgramError::MalformedProgram(format!(
                    "{label} permutation references position {old_position} more than once",
                )));
            }
            *slot = Some(new_position);
        }
        inverse
            .into_iter()
            .enumerate()
            .map(|(old_position, new_position)| {
                new_position.ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "{label} permutation does not reference position {old_position}",
                    ))
                })
            })
            .collect()
    }

    let input_types = program.input_types();
    let output_count = program.output_count();
    let inverse_input_order = inverse_order(input_order, input_types.len(), "input")?;
    let _ = inverse_order(output_order, output_count, "output")?;
    let reordered_input_types = input_order.iter().map(|&index| input_types[index].clone()).collect::<Vec<_>>();
    let mut builder = ProgramBuilder::new();
    let inputs = reordered_input_types.into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
    let original_inputs = inverse_input_order.iter().map(|&new_position| inputs[new_position]).collect::<Vec<_>>();
    let outputs = builder.splice_program(program, original_inputs.as_slice())?;
    let reordered_outputs = output_order.iter().map(|&index| outputs[index]).collect::<Vec<_>>();
    builder.build(reordered_outputs, vec![Placeholder; input_order.len()], vec![Placeholder; output_order.len()])
}

/// Transpose rule for [`ScanOperation`], dispatching to the scan's type family through the crate-private
/// `ScanTransposition` trait: array scans transpose captured linear scans whole and forward operand-form primal scans
/// to [`transpose_primal_scan`].
impl<V, F, Target> TransposableOperation<V, Target> for ScanOperation<F>
where
    V: Value,
    V::Type: ScanTypeSemantics + ScanTransposition<V, F, Target>,
    F: Value<Type = V::Type>,
    Target: Operation<Type = V::Type>,
{
    fn transpose<D: TranspositionDriver<V, Target>>(
        &self,
        context: &mut TracingContext<V, Target>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, Target>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, Target>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, Target>>>>, DifferentiationError> {
        <V::Type>::transpose_scan(self, context, driver, inputs, outputs)
    }
}

/// Type-family transposition semantics for [`ScanOperation`], with the scan's value, body-operation, capture,
/// payload, and staging-target parameters riding as trait inputs and the type family as the implementing type
/// (mirroring [`ScanPayload`](crate::operations::control_flow::scan::ScanPayload)) so that each family implementation
/// carries only the bounds its rule needs.
pub(crate) trait ScanTransposition<V, F, Target>: Type
where
    V: Value<Type = Self>,
    F: Value<Type = Self>,
    Target: Operation<Type = Self>,
{
    /// Applies the type family's `scan` transpose rule using the loop's driver; refer to the documentation of
    /// [`TransposableOperation::transpose`] for the contract.
    fn transpose_scan<D: TranspositionDriver<V, Target>>(
        operation: &ScanOperation<F>,
        context: &mut TracingContext<V, Target>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, Target>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, Target>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, Target>>>>, DifferentiationError>;
}

/// Transpose rule for array scans, covering both scan forms that reach a reverse pass.
///
/// A *captured* linear scan (non-empty [`captures`](ScanOperation::captures)) is transposed whole: linear-scan
/// transposition is total because the body pushforward maps `[carry..., x_slice...]` to `[carry..., y_slice...]`, so
/// its program transpose maps `[carry_cotangent..., y_slice_cotangent...]` to
/// `[carry_cotangent..., x_slice_cotangent...]` — the same scan-body signature with the same carry count. Flipping
/// `reverse` pairs cotangent iteration `i` with residual stack iteration `i` exactly when the forward scan consumed
/// them, so the same residual stacks (and the lowering-only unroll factor) carry over verbatim.
///
/// A capture-free scan is a *primal* operand-form scan whose known residual stacks ride as ordinary operands, so it is
/// forwarded to the partition-aware [`transpose_primal_scan`] rule instead. Both forms recurse into the body through
/// the instruction-scoped driver's transposition requests, keeping the scan-local recursion owned by the operation
/// family with no recursive [`TransposableOperation`] obligation on `O`.
impl<V, F, Target> ScanTransposition<V, F, Target> for ArrayType
where
    V: Value<Type = ArrayType>,
    F: Value<Type = ArrayType>,
    Target: Operation<Type = ArrayType> + ResidualZeroProvider<ArrayType> + From<ScanOperation<F>>,
{
    fn transpose_scan<D: TranspositionDriver<V, Target>>(
        operation: &ScanOperation<F>,
        context: &mut TracingContext<V, Target>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, Target>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, Target>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, Target>>>>, DifferentiationError> {
        transpose_array_scan(operation, context, driver, inputs, outputs)
    }
}

impl<V, F, Target> ScanTransposition<V, F, Target> for ArrayIrType
where
    V: Value<Type = ArrayIrType>,
    F: Value<Type = ArrayIrType>,
    Target: Operation<Type = ArrayIrType> + ResidualZeroProvider<ArrayIrType> + From<ScanOperation<F>>,
{
    fn transpose_scan<D: TranspositionDriver<V, Target>>(
        operation: &ScanOperation<F>,
        context: &mut TracingContext<V, Target>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, Target>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, Target>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, Target>>>>, DifferentiationError> {
        transpose_array_scan(operation, context, driver, inputs, outputs)
    }
}

/// Applies the shared transpose rule for scan type families that can represent stacked values.
fn transpose_array_scan<V, F, Target, D>(
    operation: &ScanOperation<F>,
    context: &mut TracingContext<V, Target>,
    driver: &D,
    inputs: &[PartialValue<Tracer<TracingContext<V, Target>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, Target>>>],
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, Target>>>>, DifferentiationError>
where
    V: Value<Type: DifferentiableType + ScanTypeSemantics>,
    F: Value<Type = V::Type>,
    Target: Operation<Type = V::Type> + ResidualZeroProvider<V::Type> + From<ScanOperation<F>>,
    D: TranspositionDriver<V, Target>,
{
    if outputs.iter().all(MaybeZero::is_zero) {
        return inputs.iter().map(|input| Ok(MaybeZero::Zero(input.r#type().cotangent()?))).collect();
    }
    if operation.captures().is_empty() {
        return transpose_primal_scan(operation, context, driver, inputs, outputs).map_err(DifferentiationError::from);
    }
    let body = driver.region(0)?;
    let runtime_length_count = usize::from(operation.length().variable().is_some());
    check_count!("input", inputs, body.input_types().len() + runtime_length_count, ProgramError);
    let (body_inputs, runtime_length_inputs) = inputs.split_at(body.input_types().len());
    // The body is transposed through its region's retained transform cache, so a body shared by several programs is
    // transposed once per linearity mask and repeated attachments of the result intern by `Arc` identity.
    let transposed_body = driver.transpose_program(body, &vec![true; body.input_ids().len()])?;
    let transposed = ScanOperation::<F>::new(operation.carry_count(), operation.length())
        .with_reverse(!operation.reverse())
        .with_unroll(operation.unroll())?
        .with_captures(operation.captures().to_vec());
    check_count!("output", outputs, body.output_types().len(), ProgramError);
    // A dead output's structural-zero cotangent still becomes a real operand of the reversed scan. Its type alone
    // cannot construct it when it references runtime identities, but the boundary already carries that geometry: at
    // least one peer cotangent is live here (the all-zero case returned above) and every known operand is live too.
    let mut materialized = outputs
        .iter()
        .map(|cotangent| {
            Target::materialize_zero_from_residual_sources(
                context,
                cotangent.clone(),
                outputs
                    .iter()
                    .filter_map(MaybeZero::as_value)
                    .chain(inputs.iter().filter_map(PartialValue::as_known)),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (index, input) in runtime_length_inputs.iter().enumerate() {
        materialized.push(input.as_known().cloned().ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "{SCAN_OPERATION_NAME} transpose runtime length operand {index} is not known"
            ))
        })?);
    }
    let cotangents = context.stage_operation(
        Target::from(transposed),
        CalleeRegionDriver::new(std::slice::from_ref(&transposed_body)),
        materialized.as_slice(),
    )?;
    check_count!("output", cotangents, body_inputs.len(), ProgramError);
    let mut cotangents = cotangents.into_iter().map(MaybeZero::Value).collect::<Vec<_>>();
    cotangents.extend(
        runtime_length_inputs
            .iter()
            .map(|input| Ok(MaybeZero::Zero(input.r#type().cotangent()?)))
            .collect::<Result<Vec<_>, DifferentiationError>>()?,
    );
    Ok(cotangents)
}

/// Partition-aware transpose rule for a *primal* [`ScanOperation`], used when the direct reverse transposes a
/// tangent program in the primal operation family `O` rather than re-keying it into the linear family. This is the
/// operand-form counterpart of the captured-stack linear-scan transpose rule above: the per-iteration residuals are
/// ordinary *scanned operands* (known residual stacks supplied through `operand_values`) instead of capture payloads,
/// so the rule reads them from the pullback and threads them back through as known scanned operands of a transposed
/// scan with the same scan-loop geometry.
///
/// The operands mirror the body's inputs one-to-one as `[carries..., scanned_inputs...]`, and each operand is
/// independently linear (a tangent the reverse accumulates) or known (a residual stack the pullback reads). The
/// forward typically marks the carry-and-scanned tangents linear and the residual stacks known, but the linear
/// operands need not form a leading run: vmapping a bounded `while` threads a non-differentiable Boolean mask as a
/// known *carry*, so a known operand can sit among the linear carries. This rule therefore:
///
///   1. Transposes the body through its instruction-scoped driver under each
///      input's own linearity. The transposed body maps every body output's cotangent followed by every known body
///      input's runtime value to every *linear* body input's cotangent:
///      `[carry_output_cotangent..., y_slice_cotangent..., known_input_value...] -> [linear_input_cotangent...]`.
///   2. Restores the reversed scan's carry-output arity, which
///      [`Program::transpose_with_respect_to`](crate::Program::transpose_with_respect_to) erases for known
///      carries (a known carry is not a linear input, so it contributes no carry cotangent output). Each known carry's
///      actual residual value is inserted into the matching body input and passed through the matching carry output,
///      so the reversed body preserves one carry slot per forward carry without fabricating a temporal zero stack.
///   3. Re-stages a primal [`ScanOperation`] over the restored body with flipped [`reverse`](ScanOperation::reverse)
///      and the same carry count, length, and (lowering-only) unroll factor, over `[outputs...,
///      known_scanned_input_stacks...]`. Known carries remain carries; only known scanned inputs consume residual
///      stacks. Flipping `reverse` pairs cotangent iteration `i` with residual stack iteration `i` exactly when the
///      forward scan consumed them, making reverse mode through the scan total with no array-reversal operation.
///
/// The returned cotangents place the reversed scan's carry cotangents at the carry-operand positions, its
/// scanned-output cotangents at the linear scanned-operand positions, and a structural [`MaybeZero::Zero`] at the
/// known scanned-operand positions, which carry no cotangent. The body recursion happens through the
/// instruction-scoped driver's transposition requests in the same operation family, so it introduces no recursive
/// [`TransposableOperation`] obligation on `O`.
///
/// # Parameters
///
///   - `operation`: Primal scan staged into the tangent program.
///   - `context`: Active transpose tracing context the pullback is staged into.
///   - `inputs`: Per-operand [`PartialValue`] knowledge, mirroring the body inputs as `[carries..., scanned_inputs...]`.
///     A linear operand is [`Unknown`](PartialValue::Unknown); a known operand is
///     [`Known`](PartialValue::Known) of the residual-stack tracer the pullback reads.
///   - `outputs`: Symbolic cotangents for the scan's outputs.
pub fn transpose_primal_scan<V, O, F, D: TranspositionDriver<V, O>>(
    operation: &ScanOperation<F>,
    context: &mut TracingContext<V, O>,
    driver: &D,
    inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError>
where
    V: Value<Type: DifferentiableType + ScanTypeSemantics>,
    F: Value<Type = V::Type>,
    O: Operation<Type = V::Type> + ResidualZeroProvider<V::Type> + From<ScanOperation<F>>,
{
    // A scan with only zero output cotangents is a zero linear map, so every operand cotangent is zero.
    if outputs.iter().all(MaybeZero::is_zero) {
        return inputs
            .iter()
            .map(|input| {
                let input_type = input.r#type();
                Ok(MaybeZero::Zero(input_type.cotangent()?))
            })
            .collect();
    }

    // Operand layout is `[carries..., scanned_inputs...]`, mirroring the body's input order one-to-one, where each
    // operand is independently linear (a tangent the reverse must accumulate) or known (a residual stack the pullback
    // reads). Linear operands need not form a leading run: vmapping a bounded `while` threads a non-differentiable
    // Boolean mask as a known *carry*, so a known operand can sit among the linear carries. The leading `carry_count`
    // operands are the carries and the rest are scanned inputs.
    let body = driver.region(0)?;
    let carry_count = operation.carry_count();
    let length = operation.length();
    let runtime_length_count = usize::from(length.variable().is_some());
    check_count!("input", inputs, body.input_types().len() + runtime_length_count, ProgramError);
    let (scan_inputs, runtime_length_inputs) = inputs.split_at(body.input_types().len());
    let operand_linear = scan_inputs.iter().map(PartialValue::is_unknown).collect::<Vec<_>>();
    check_count!("input", operand_linear, body.input_types().len(), ProgramError);
    if carry_count > operand_linear.len() {
        return Err(ProgramError::MalformedProgram(format!(
            "{} transpose found carry count {} exceeding its {} operands",
            SCAN_OPERATION_NAME,
            carry_count,
            operand_linear.len(),
        )));
    }

    // Transpose the body with each input's own linearity. The transposed body maps the cotangent of every body output
    // followed by every known body input's runtime value to the cotangent of every *linear* body input only:
    // `[carry_output_cotangent..., y_slice_cotangent..., known_input_value...] -> [linear_input_cotangent...]`, in body
    // order on each side.
    // The body is transposed through its region's retained transform cache, so a body shared by several programs is
    // transposed once per linearity mask and repeated attachments of the result intern by `Arc` identity.
    let mut transposed_body =
        driver.transpose_program(body, operand_linear.as_slice()).map_err(|error| match error {
            crate::differentiation::DifferentiationError::Program(error) => error,
            error => ProgramError::UnsupportedOperation { message: error.to_string() },
        })?;

    // A known carry is loop state, not a stacked operand. Move its exposed known-value input into the matching carry
    // slot and pass it through as the matching body output. Linear carries retain their cotangent slots, while known
    // scanned inputs remain trailing per-iteration slices. This avoids fabricating a zero stack for a known carry and
    // lets first-class dimension carries define the identities referenced by dynamic tangent-array carries.
    // Threading is a per-attachment rewrite of the retained transposition rather than a property of the body, so the
    // shared artifact is rebuilt here instead of being retained in its threaded form.
    let linear_carry_count = operand_linear[..carry_count].iter().filter(|&&linear| linear).count();
    if linear_carry_count != carry_count {
        transposed_body = Arc::new(thread_known_carries(
            transposed_body.as_ref().clone(),
            body.output_types().as_slice(),
            operand_linear.as_slice(),
            carry_count,
        )?);
    }

    let transposed = ScanOperation::<F>::new(carry_count, length)
        .with_reverse(!operation.reverse())
        .with_unroll(operation.unroll())?;

    // Stage the reversed scan over `[outputs..., known_input_value_stacks...]`, matching the transposed
    // body's input order. The output cotangents are typed by the *scan operation's* outputs, not the body's
    // per-iteration outputs: the leading carries keep their per-iteration shape while each trailing y-slice output is
    // stacked along the scan length. Using the body's per-iteration y-slice types here would materialize a zero
    // cotangent for a dead y-output with the un-stacked slice type, desyncing the reversed scan's operand signature.
    check_count!("output", outputs, body.output_types().len(), ProgramError);
    // A dead output's structural-zero cotangent still becomes a real operand of the reversed scan. Its type alone
    // cannot construct it when it references runtime identities, but the boundary collectively names every such
    // quantity, so the zero is assembled from the peers one identity at a time. Carry cotangents keep the
    // per-iteration geometry that live peer carries and known carries also carry, while a dead *stacked* cotangent's
    // scan-length-prefixed geometry is split across the boundary: the length identity rides the runtime length
    // operand (a first-class dimension) and any known residual stack, while its inner extents ride the carries and
    // per-iteration peers. No peer has the stacked type as a whole, which is exactly why exemplar matching cannot
    // close this case and identity-directed capture can.
    let geometry_sources = || {
        outputs
            .iter()
            .filter_map(MaybeZero::as_value)
            .chain(inputs.iter().filter_map(PartialValue::as_known))
    };
    let mut operands = Vec::with_capacity(outputs.len() + operand_linear.len());
    for index in 0..carry_count {
        if operand_linear[index] {
            operands.push(O::materialize_zero_from_residual_sources(
                context,
                outputs[index].clone(),
                geometry_sources(),
            )?);
        } else {
            operands.push(scan_inputs[index].as_known().cloned().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "{SCAN_OPERATION_NAME} transpose carry operand {index} has no known residual value",
                ))
            })?);
        }
    }
    for (cotangent, output_type) in outputs[carry_count..].iter().zip(&body.output_types()[carry_count..]) {
        if !output_type.cotangent()?.is_zero_space() {
            operands.push(O::materialize_zero_from_residual_sources(context, cotangent.clone(), geometry_sources())?);
        }
    }

    // Append one scanned operand per known body input, in body order, to feed the transposed body's known-value
    // inputs. A known *scanned* input is a residual stack read from the pullback; known carries were already placed in
    // their carry slots above and therefore add no trailing operand here. A known intermediate without a pullback
    // value is one the partial-evaluation split must never leave in a tangent program, so its absence is malformed.
    for (index, &linear) in operand_linear.iter().enumerate() {
        if linear {
            continue;
        }
        if index >= carry_count {
            // A known scanned operand is a residual stack; the dispatch guarantees it carries its pullback value.
            let residual = scan_inputs[index].as_known().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "{SCAN_OPERATION_NAME} transpose operand {index} has no known residual value"
                ))
            })?;
            operands.push(residual.clone());
        }
    }
    for (index, input) in runtime_length_inputs.iter().enumerate() {
        operands.push(input.as_known().cloned().ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "{SCAN_OPERATION_NAME} transpose runtime length operand {index} is not known"
            ))
        })?);
    }

    // The reversed scan outputs one carry cotangent per carry and one stacked scanned-output cotangent per *linear*
    // scanned input.
    let linear_scanned_count = operand_linear[carry_count..].iter().filter(|&&linear| linear).count();
    let scan_cotangents = context.stage_operation(
        O::from(transposed),
        CalleeRegionDriver::new(std::slice::from_ref(&transposed_body)),
        operands.as_slice(),
    )?;
    check_count!("output", scan_cotangents, carry_count + linear_scanned_count, ProgramError);

    // Reassemble one cotangent per operand. The reversed scan outputs `[carry_cotangent..., scanned_input_cotangent...]`,
    // the carry cotangents (including the re-inserted zeros for known carries) leading the scanned-input cotangents over
    // the *linear* scanned inputs. Every carry operand precedes every scanned operand, so a single sequential drain
    // hands each carry operand the next carry cotangent and each linear scanned operand the next scanned-input
    // cotangent in turn; known scanned operands carry a structural zero (they are residual stacks, which carry no
    // cotangent).
    let mut scan_cotangents = scan_cotangents.into_iter();
    let cotangents = operand_linear
        .iter()
        .zip(scan_inputs)
        .enumerate()
        .map(|(index, (&linear, input))| -> Result<_, DifferentiationError> {
            if index < carry_count {
                let cotangent = scan_cotangents.next().unwrap();
                Ok(if linear { MaybeZero::Value(cotangent) } else { MaybeZero::Zero(input.r#type().cotangent()?) })
            } else if linear {
                Ok(MaybeZero::Value(scan_cotangents.next().unwrap()))
            } else {
                Ok(MaybeZero::Zero(input.r#type().cotangent()?))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut cotangents = cotangents;
    cotangents.extend(
        runtime_length_inputs
            .iter()
            .map(|input| Ok(MaybeZero::Zero(input.r#type().cotangent()?)))
            .collect::<Result<Vec<_>, DifferentiationError>>()?,
    );
    Ok(cotangents)
}

/// Rebuilds a transposed scan body so known carry values occupy and pass through their carry slots.
fn thread_known_carries<V, O>(
    program: Program<V, O, Vec<V>, Vec<V>>,
    body_output_types: &[V::Type],
    operand_linear: &[bool],
    carry_count: usize,
) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError>
where
    V: Value<Type: DifferentiableType>,
    O: Operation<Type = V::Type>,
{
    let linear_carry_count = operand_linear[..carry_count].iter().filter(|&&linear| linear).count();
    let body_output_count = body_output_types.len();
    let output_cotangent_positions = body_output_types
        .iter()
        .enumerate()
        .map(|(position, output_type)| Ok((position, output_type.cotangent()?)))
        .collect::<Result<Vec<_>, DifferentiationError>>()?
        .into_iter()
        .map(|(position, cotangent_type)| (!cotangent_type.is_zero_space()).then_some(position))
        .collect::<Vec<_>>();
    let known_input_positions = operand_linear
        .iter()
        .scan(body_output_count, |position, &linear| {
            let result = (!linear).then_some(*position);
            *position += usize::from(!linear);
            Some(result)
        })
        .collect::<Vec<_>>();
    let input_order =
        operand_linear[..carry_count]
            .iter()
            .enumerate()
            .map(|(index, &linear)| {
                if linear { output_cotangent_positions[index].unwrap() } else { known_input_positions[index].unwrap() }
            })
            .chain(output_cotangent_positions[carry_count..body_output_count].iter().flatten().copied())
            .chain(known_input_positions[carry_count..].iter().flatten().copied())
            .collect::<Vec<_>>();

    // Zero-space output-cotangent inputs carry no information and cannot affect a well-formed transposed body. Project
    // them out instead of fabricating typed values merely to satisfy the old boundary while splicing. Keeping every
    // selected input alive preserves known carry slots even when the derivative body does not otherwise read them.
    let selected_inputs = input_order
        .iter()
        .map(|&index| {
            program.input_ids().get(index).copied().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "{SCAN_OPERATION_NAME} transpose boundary references missing input position {index}",
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output_ids = program.output_ids().to_vec();
    let selected_input_count = selected_inputs.len();
    let (program, live_inputs) =
        program.into_filtered(selected_inputs.as_slice(), output_ids.as_slice(), selected_inputs.as_slice())?;
    if !live_inputs.iter().copied().eq(0..selected_input_count) {
        return Err(ProgramError::MalformedProgram(format!(
            "{SCAN_OPERATION_NAME} transpose boundary projection dropped a retained input"
        )));
    }

    let input_types = program.input_types();
    let mut builder = ProgramBuilder::new();
    let inputs = input_types.into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
    let mut outputs = builder.splice_program(&program, inputs.as_slice())?;
    check_count!("output", outputs, program.output_count(), ProgramError);
    let trailing_outputs = outputs.split_off(linear_carry_count);
    let mut linear_carry_outputs = outputs.into_iter();
    let mut restored_outputs = Vec::with_capacity(carry_count + trailing_outputs.len());
    for (carry_index, &carry_is_linear) in operand_linear[..carry_count].iter().enumerate() {
        if carry_is_linear {
            restored_outputs.push(linear_carry_outputs.next().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "{SCAN_OPERATION_NAME} transpose missing linear carry cotangent output {carry_index}",
                ))
            })?);
        } else {
            restored_outputs.push(inputs[carry_index]);
        }
    }
    restored_outputs.extend(trailing_outputs);
    let output_count = restored_outputs.len();
    builder.build(restored_outputs, vec![Placeholder; inputs.len()], vec![Placeholder; output_count])
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, DataType, DimensionBounds, DimensionType,
        DimensionVariable, LogicalMesh, Memory, MeshAxis, MeshAxisType, Sharding, ShardingDimension,
    };
    use crate::batching::{BatchingTracer, batch};
    use crate::captures::CaptureReference;
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::forward::JvpTransform;
    use crate::differentiation::reverse::TranspositionTransform;
    use crate::differentiation::{Differentiate, LinearizationTracer, ReverseModeDifferentiate, differentiate_at};
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::constants::zero_like::ZeroLikeOperation;
    use crate::operations::math::add::AddOperation;
    use crate::operations::math::div::DivOperation;
    use crate::operations::math::mul::MulOperation;
    use crate::operations::math::sin::SinOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{Effects, Program, ProgramBuilder};
    use crate::tests::CountingBatchingDriver;
    use crate::tracing::{DomainTracingContext, Trace};

    use super::*;

    type TestScanOperation = ScanOperation<Array>;

    /// Returns the [`RegionInterface`] of the provided flat region program.
    fn region_interface(
        program: &Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>,
    ) -> RegionInterface<ArrayType> {
        program.interface()
    }

    /// Builds a cumulative-product body program that maps `[carry, x]` to `[carry * x, carry * x]`: the new carry is
    /// the running product and each iteration also emits that product as a stacked output slice.
    fn product_body() -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        product_body_with_type(ArrayType::scalar(DataType::F64))
    }

    /// Builds a cumulative-product body over `r#type` that maps `[carry, x]` to `[carry * x, carry * x]`.
    fn product_body_with_type(r#type: ArrayType) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let carry = builder.add_input(r#type.clone());
        let x = builder.add_input(r#type);
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![carry, x]).unwrap()[0];
        builder
            .build(vec![product, product], vec![Placeholder, Placeholder], vec![Placeholder, Placeholder])
            .unwrap()
    }

    /// Builds a carry-only body program that maps `[carry]` to `[carry + carry]` with no stacked inputs or outputs.
    fn doubling_body() -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let carry = builder.add_input(ArrayType::scalar(DataType::F64));
        let doubled = builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, carry]).unwrap()[0];
        builder.build(vec![doubled], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Applies the three-iteration cumulative-product scan and returns its final carry.
    fn stage_product_scan<V: Value<Type = ArrayType>>(initial: V, values: V) -> Result<V, ProgramError>
    where
        V::DispatchDomain: Context<Type = ArrayType, Constant = Array, Operation = ArrayOperation<Array>>,
    {
        let mut outputs = initial.dispatch_domain().bind(
            ArrayOperation::Scan(ScanOperation::new(1, 3)),
            vec![product_body()],
            &[initial.clone(), values],
        )?;
        Ok(outputs.remove(0))
    }

    #[test]
    fn test_scan_composite_type_contract() {
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let dimension_type = ArrayIrType::Dimension(DimensionType::new(extent.clone()));
        let slice_type =
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone())])));
        let stacked_type = ArrayIrType::Array(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Static(3), Dimension::Dynamic(extent.clone())]),
        ));
        let body_input_types = vec![dimension_type.clone(), slice_type.clone()];
        let body_output_types = vec![dimension_type.clone(), slice_type];
        let body_interface = RegionInterface::new(body_input_types, body_output_types, Effects::PURE);
        let operation = ScanOperation::<CaptureReference<ArrayIrType>>::new(1, 3);
        let input_types = vec![dimension_type.clone(), stacked_type.clone()];

        assert_eq!(
            operation.infer_region_input_types(input_types.as_slice(), std::slice::from_ref(&body_interface)),
            Ok(vec![None]),
        );
        assert_eq!(
            operation.infer_output_types(input_types.as_slice(), std::slice::from_ref(&body_interface)),
            Ok(vec![dimension_type.clone(), stacked_type.clone()]),
        );

        let captured_dimension = ScanOperation::<CaptureReference<ArrayIrType>>::new(1, 3)
            .with_captures(vec![CaptureReference::new(0, dimension_type.clone())]);
        assert_eq!(
            captured_dimension.infer_output_types(input_types.as_slice(), std::slice::from_ref(&body_interface)),
            Err(TypeError::invalid(
                "scan capture 0 must be a stacked array but got dimension<extent ∈ [1, 8)>".to_string(),
            )),
        );

        let invalid_body_interface = RegionInterface::new(
            vec![dimension_type.clone(), dimension_type.clone()],
            vec![dimension_type.clone()],
            Effects::PURE,
        );
        assert_eq!(
            ScanOperation::<CaptureReference<ArrayIrType>>::new(1, 3)
                .infer_output_types(&[dimension_type.clone(), dimension_type.clone()], &[invalid_body_interface]),
            Err(TypeError::invalid(
                "scan stacked body input 1 must be an array but got dimension<extent ∈ [1, 8)>".to_string(),
            )),
        );

        // A fresh dimension produced by the body cannot replace the declared carry identity on each iteration.
        // Supporting such shape-varying state would require an explicit widening contract. `scan` instead keeps its
        // loop-carried type invariant and reports the exact incompatible signatures.
        let carry = DimensionVariable::new("carry", DimensionBounds::positive(Some(8)).unwrap());
        let next = DimensionVariable::new("next", DimensionBounds::positive(Some(8)).unwrap());
        let carry_type =
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(carry.clone())])));
        let next_type = ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(next)])));
        let shape_varying_body = RegionInterface::new(vec![carry_type.clone()], vec![next_type], Effects::PURE);
        assert_eq!(
            ScanOperation::<CaptureReference<ArrayIrType>>::new(1, 3)
                .infer_output_types(std::slice::from_ref(&carry_type), &[shape_varying_body]),
            Err(TypeError::invalid(
                "scan body carry type signature mismatch: expected [f32[carry]] but got [f32[next]]".to_string(),
            )),
        );

        // A dynamic runtime-length operand must carry the scan length's nominal identity. An unrelated dynamic
        // identity with compatible bounds cannot redefine the stacked axis. An operand whose bounds pin one exact
        // extent fixes the trip count to that extent and is admissible only when every stacked operand is refined to
        // the same extent; a stacked axis left symbolic would otherwise be read a fixed number of times regardless of
        // its independently determined runtime size. The accepted refinement also types the stacked outputs at the
        // concrete extent instead of at the still-symbolic declared length.
        let length = DimensionVariable::new("length", DimensionBounds::positive(Some(5)).unwrap());
        let unrelated = DimensionVariable::new("unrelated", DimensionBounds::positive(Some(5)).unwrap());
        let three = DimensionType::new(DimensionVariable::new("three", DimensionBounds::new(3, Some(4)).unwrap()));
        let dynamic_stacked_type = ArrayIrType::Array(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(length.clone()), Dimension::Dynamic(extent)]),
        ));
        let dynamic_scan = ScanOperation::<CaptureReference<ArrayIrType>>::new(1, Dimension::Dynamic(length.clone()));
        assert_eq!(
            dynamic_scan.infer_output_types(
                &[dimension_type.clone(), dynamic_stacked_type.clone(), DimensionType::new(unrelated).into()],
                std::slice::from_ref(&body_interface),
            ),
            Err(TypeError::invalid(
                "`scan` runtime length operand has type dimension<unrelated ∈ [1, 5)> but scan length requires length"
                    .to_string(),
            )),
        );
        assert_eq!(
            dynamic_scan.infer_output_types(
                &[dimension_type.clone(), dynamic_stacked_type.clone(), three.clone().into()],
                std::slice::from_ref(&body_interface),
            ),
            Err(TypeError::invalid(
                "`scan` runtime length operand has type dimension<3> but stacked input 1 has type f32[length, extent] \
                 whose leading dimension is not refined to extent 3"
                    .to_string(),
            )),
        );
        // The same consistency requirement applies to captures: a capture stacked over the still-symbolic declared
        // length cannot be driven by a concretely refined trip count, while a capture refined to the exact extent is
        // admissible.
        let captured_symbolic =
            ScanOperation::<CaptureReference<ArrayIrType>>::new(1, Dimension::Dynamic(length.clone()))
                .with_captures(vec![CaptureReference::new(0, dynamic_stacked_type.clone())]);
        assert_eq!(
            captured_symbolic.infer_output_types(
                &[dimension_type.clone(), stacked_type.clone(), three.clone().into()],
                std::slice::from_ref(&body_interface),
            ),
            Err(TypeError::invalid(
                "scan capture 0 must have leading dimension 3 but has type f32[length, extent]".to_string(),
            )),
        );
        let captured_refined = ScanOperation::<CaptureReference<ArrayIrType>>::new(1, Dimension::Dynamic(length))
            .with_captures(vec![CaptureReference::new(0, stacked_type.clone())]);
        assert_eq!(
            captured_refined.infer_output_types(
                &[dimension_type.clone(), stacked_type.clone(), three.clone().into()],
                std::slice::from_ref(&body_interface),
            ),
            Ok(vec![dimension_type.clone(), stacked_type.clone()]),
        );
        assert_eq!(
            dynamic_scan
                .infer_output_types(&[dimension_type.clone(), stacked_type.clone(), three.into()], &[body_interface]),
            Ok(vec![dimension_type, stacked_type]),
        );
    }

    #[test]
    fn test_scan_stacked_type_preserves_memory() {
        let slice_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]))
            .with_memory(Memory::Host { pinned: true });

        assert_eq!(
            stacked_scan_type(&slice_type, 2),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
                .with_memory(Memory::Host { pinned: true }),
        );
    }

    #[test]
    fn test_scan() {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let stacked_f64 = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let operation = TestScanOperation::new(1, 3);
        let body = product_body();
        let interfaces = vec![region_interface(&body)];

        // Operation identity, declared region slots, output provenance, and accessors.
        assert_eq!(operation.name(), SCAN_OPERATION_NAME);
        assert_eq!(operation.region_slots(), &[RegionSlot::computation("body")]);
        assert_eq!(
            operation.output_region_provenance(1),
            vec![OutputRegionProvenance { region_index: 0, output_index: 1 }],
        );
        assert_eq!(operation.carry_count(), 1);
        assert_eq!(operation.length(), &Dimension::Static(3));
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
            Err(TypeError::invalid("expected 1 region but got 0")),
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&scalar_f64), interfaces.as_slice()),
            Err(TypeError::invalid("expected 2 inputs but got 1".to_string())),
        );
        assert_eq!(
            operation.infer_output_types(&[scalar_f64.clone(), scalar_f64.clone()], interfaces.as_slice()),
            Err(TypeError::invalid(
                "scan input 1 has type f64[] which is incompatible with the expected type f64[3]".to_string()
            )),
        );

        // The lowering-only unroll factor must be at least 1 and must evenly divide the scan length; valid factors
        // render only when greater than 1 and interpretation ignores them entirely.
        assert_eq!(
            TestScanOperation::new(1, 3).with_unroll(0).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid("scan unroll factor must be at least 1".to_string()))),
        );
        assert_eq!(
            TestScanOperation::new(1, 3).with_unroll(2).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid(
                "scan unroll factor 2 must evenly divide the scan length 3".to_string()
            ))),
        );
        let unrolled = TestScanOperation::new(1, 3).with_unroll(3).unwrap();
        assert_eq!(unrolled.unroll(), 3);
        assert_eq!(format!("{unrolled}"), "scan [carry_count=1, length=3, reverse=false, unroll=3]");
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let outputs = context
            .bind(
                ArrayOperation::Scan(unrolled),
                vec![body.clone()],
                &[Array::scalar(1.0), Array::vector(vec![2.0, 3.0, 4.0])],
            )
            .unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![24.0]);
        assert_eq!(outputs[1].to_f64s(), vec![2.0, 6.0, 24.0]);

        // Inference rejects carry counts that exceed the body signature, mismatched carry types, and dynamically
        // sized body slice types over the attached region interface.
        assert_eq!(
            TestScanOperation::new(3, 3)
                .infer_output_types(&[scalar_f64.clone(), stacked_f64.clone()], interfaces.as_slice()),
            Err(TypeError::invalid("scan carry count 3 exceeds the body input count 2".to_string())),
        );
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let carry = builder.add_input(scalar_f64.clone());
        let x = builder.add_input(scalar_f64.clone());
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![carry, x]).unwrap()[0];
        let no_output_body = builder
            .build::<Vec<Array>, Vec<Array>>(vec![product], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            TestScanOperation::new(2, 3)
                .infer_output_types(&[scalar_f64.clone(), scalar_f64.clone()], &[region_interface(&no_output_body)],),
            Err(TypeError::invalid("scan carry count 2 exceeds the body output count 1".to_string())),
        );
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let mismatched_carry = builder.add_input(scalar_f64.clone());
        let mismatched_output =
            builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![mismatched_carry]).unwrap()[0];
        let mismatched_output = builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::Equal),
                Vec::new(),
                vec![mismatched_output, mismatched_carry],
            )
            .unwrap()[0];
        let mismatched_body = builder
            .build::<Vec<Array>, Vec<Array>>(vec![mismatched_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            TestScanOperation::new(1, 3)
                .infer_output_types(std::slice::from_ref(&scalar_f64), &[region_interface(&mismatched_body)]),
            Err(TypeError::invalid(
                "scan body carry type signature mismatch: expected [f64[]] but got [bool[]]".to_string()
            )),
        );
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let dynamic_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded()))]),
        );
        let dynamic_carry = builder.add_input(dynamic_type.clone());
        let dynamic_body = builder
            .build::<Vec<Array>, Vec<Array>>(vec![dynamic_carry], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            TestScanOperation::new(1, 3)
                .infer_output_types(std::slice::from_ref(&dynamic_type), &[region_interface(&dynamic_body)]),
            Err(TypeError::invalid(
                "scan body input 0 must have a fully static type but axis 0 of f64[dynamic] has size dynamic"
                    .to_string()
            )),
        );

        // Eager binding threads the carry while stacking the per-iteration outputs: a cumulative product over
        // `xs = [2, 3, 4]` starting at `1` produces the final carry `24` and the running products `[2, 6, 24]`.
        let outputs = context
            .bind(
                ArrayOperation::Scan(operation.clone()),
                vec![body.clone()],
                &[Array::scalar(1.0), Array::vector(vec![2.0, 3.0, 4.0])],
            )
            .unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![24.0]);
        assert_eq!(outputs[1].to_f64s(), vec![2.0, 6.0, 24.0]);

        // A reversed scan visits the slices from the back while keeping output slice `i` aligned with input slice
        // `i`: the running products visit `4, 3, 2` and land in iterations `2, 1, 0`.
        let reversed = TestScanOperation::new(1, 3).with_reverse(true);
        let outputs = context
            .bind(
                ArrayOperation::Scan(reversed),
                vec![body.clone()],
                &[Array::scalar(1.0), Array::vector(vec![2.0, 3.0, 4.0])],
            )
            .unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![24.0]);
        assert_eq!(outputs[1].to_f64s(), vec![24.0, 12.0, 4.0]);

        // A carry-only scan with no stacked inputs or outputs applies the body `length` times.
        let carry_only = TestScanOperation::new(1, 3);
        let outputs = context
            .bind(ArrayOperation::Scan(carry_only), vec![doubling_body()], &[Array::scalar(1.0)])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].to_f64s(), vec![8.0]);

        // A zero-length scan returns the initial carries and empty stacked outputs.
        let empty = TestScanOperation::new(1, 0);
        let empty_stacked_f64 = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0)]));
        let outputs = context
            .bind(
                ArrayOperation::Scan(empty),
                vec![body.clone()],
                &[Array::scalar(1.0), Array::from_f64s(empty_stacked_f64, vec![])],
            )
            .unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![1.0]);
        assert_eq!(outputs[1].to_f64s(), Vec::<f64>::new());

        // Staging imports the body program as an attached region of the staged instruction instead of running scan
        // iterations eagerly over staged values.
        let context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
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
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let program_carry = builder.add_input(scalar_f64);
        let program_xs = builder.add_input(stacked_f64);
        let program_outputs = builder
            .add_instruction(ArrayOperation::Scan(operation), vec![body_region], vec![program_carry, program_xs])
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(
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

    #[test]
    fn test_scan_linearization_and_transposition_preserve_carry_derivatives() {
        type TestContext = EagerContext<Array, ArrayOperation<Array>>;
        type TestTracer = LinearizationTracer<TestContext>;

        let function = |(carry, values): (TestTracer, TestTracer)| {
            let mut outputs = carry.context().bind(
                ArrayOperation::Scan(ScanOperation::new(1, 3)),
                vec![product_body()],
                &[carry.clone(), values],
            )?;
            Ok((outputs.remove(0), outputs.remove(0)))
        };
        let primals = (Array::scalar(1.0), Array::vector(vec![2.0, 3.0, 4.0]));
        let (outputs, pushforward) = differentiate_at(primals.clone()).linearize(function).unwrap();
        assert_eq!(outputs, (Array::scalar(24.0), Array::vector(vec![2.0, 6.0, 24.0])));
        assert_eq!(
            pushforward.apply((Array::scalar(1.0), Array::vector(vec![0.0, 0.0, 0.0]))),
            Ok((Array::scalar(24.0), Array::vector(vec![2.0, 6.0, 24.0]))),
        );

        let (final_carry, pullback) = differentiate_at(primals)
            .vjp(|(carry, values)| {
                let mut outputs = carry.context().bind(
                    ArrayOperation::Scan(ScanOperation::new(1, 3)),
                    vec![product_body()],
                    &[carry.clone(), values],
                )?;
                Ok(outputs.remove(0))
            })
            .unwrap();
        assert_eq!(final_carry, Array::scalar(24.0));
        assert_eq!(pullback.apply(Array::scalar(1.0)), Ok((Array::scalar(24.0), Array::vector(vec![12.0, 8.0, 6.0]))),);
    }

    #[test]
    fn test_scan_vjp_stages_reusable_reversed_scan_pullback() {
        let (output, pullback) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .vjp(
                |(initial, values), ()| stage_product_scan(initial, values),
                (Array::scalar(1.0), Array::vector(vec![2.0, 3.0, 4.0])),
                (),
            )
            .unwrap();
        let (pullback, residuals) = pullback.into_parts();
        assert_eq!(output.to_f64s(), vec![24.0]);
        let rendered_pullback = pullback.to_string();
        assert!(rendered_pullback.contains("scan"), "{rendered_pullback}");
        assert!(rendered_pullback.contains("reverse=true"), "{rendered_pullback}");

        let mut pullback_inputs = vec![Array::scalar(1.0)];
        pullback_inputs.extend(residuals.iter().cloned());
        let cotangents = pullback.interpret(pullback_inputs).unwrap();
        assert_eq!(cotangents[0].to_f64s(), vec![24.0]);
        assert_eq!(cotangents[1].to_f64s(), vec![12.0, 8.0, 6.0]);

        let mut pullback_inputs = vec![Array::scalar(2.0)];
        pullback_inputs.extend(residuals);
        let cotangents = pullback.interpret(pullback_inputs).unwrap();
        assert_eq!(cotangents[0].to_f64s(), vec![48.0]);
        assert_eq!(cotangents[1].to_f64s(), vec![24.0, 16.0, 12.0]);
    }

    #[test]
    fn test_scan_dense_jacobians_replay_body_region() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let primals = (Array::scalar(1.0), Array::vector(vec![2.0, 3.0, 4.0]));
        let forward = context
            .differentiate_at(primals.clone())
            .jacobian_forward(|(initial, values)| stage_product_scan(initial, values))
            .unwrap();
        let reverse = context
            .differentiate_at(primals)
            .jacobian_reverse(|(initial, values)| stage_product_scan(initial, values))
            .unwrap();

        let blocks = forward.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].value().to_f64s(), vec![24.0]);
        assert_eq!(blocks[1].value().to_f64s(), vec![12.0, 8.0, 6.0]);

        let blocks = reverse.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].value().to_f64s(), vec![24.0]);
        assert_eq!(blocks[1].value().to_f64s(), vec![12.0, 8.0, 6.0]);
    }

    #[test]
    fn test_scan_hessian_replays_body_region() {
        // For `f(initial, values) = initial * product(values)`, same-variable second derivatives vanish. Mixed
        // derivatives with `initial` are products excluding the corresponding value, while mixed derivatives between
        // values are `initial` times the remaining value.
        let hessian = EagerContext::<Array, ArrayOperation<Array>>::new()
            .differentiate_at((Array::scalar(1.0), Array::vector(vec![2.0, 3.0, 4.0])))
            .hessian(|(initial, values)| stage_product_scan(initial, values))
            .unwrap();

        let blocks = hessian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 4);
        assert_eq!(blocks[0].value().to_f64s(), vec![0.0]);
        assert_eq!(blocks[1].value().to_f64s(), vec![12.0, 8.0, 6.0]);
        assert_eq!(blocks[2].value().to_f64s(), vec![12.0, 8.0, 6.0]);
        assert_eq!(
            blocks[3].value().to_f64s(),
            vec![
                0.0, 4.0, 3.0, //
                4.0, 0.0, 2.0, //
                3.0, 2.0, 0.0, //
            ],
        );
    }

    /// A zero-length scan runs no iteration, so partial evaluation must not probe its body: a body whose known-side
    /// fold errors (here an integer division by a known zero carry) residualizes unchanged instead of failing.
    #[test]
    fn test_scan_partial_evaluation_residualizes_zero_length_scans_without_probing_the_body() {
        let carry_type = ArrayType::scalar(DataType::I32);
        let stack_type = stacked_scan_type(&carry_type, 0);
        let body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let carry = builder.add_input(carry_type.clone());
            let x = builder.add_input(carry_type.clone());
            let one = builder.add_constant(Array::from_f64s(carry_type.clone(), vec![1.0]));
            let inverse = builder.add_instruction(DivOperation::new(), Vec::new(), vec![one, carry]).unwrap()[0];
            let y = builder.add_instruction(MulOperation::new(), Vec::new(), vec![inverse, x]).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![carry, y], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let carry = builder.add_input(carry_type.clone());
        let xs = builder.add_input(stack_type.clone());
        let outputs = builder
            .add_instruction(ArrayOperation::Scan(ScanOperation::new(1, 0)), vec![body_region], vec![carry, xs])
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        // The known zero carry would fold `1 / carry` during an invariance probe; with no iteration to run, the
        // partial evaluation must succeed and keep the scan whole.
        let knowledge =
            vec![PartialValue::Known(Array::from_f64s(carry_type, vec![0.0])), PartialValue::Unknown(stack_type)];
        let evaluation = program.partially_evaluate(knowledge.as_slice()).unwrap();
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), ArrayOperation::Scan(_)));
    }

    /// Forward- and reverse-mode differentiation flow through a `reverse` scan: the visit order flips while slice
    /// `i` of every stacked value stays paired with iteration `i`, and the transposed scan flips `reverse` back.
    #[test]
    fn test_scan_differentiation_flows_through_reverse_scans() {
        type TestContext = EagerContext<Array, ArrayOperation<Array>>;
        type TestTracer = LinearizationTracer<TestContext>;

        let function = |(carry, values): (TestTracer, TestTracer)| {
            let mut outputs = carry.context().bind(
                ArrayOperation::Scan(ScanOperation::new(1, 3).with_reverse(true)),
                vec![product_body()],
                &[carry.clone(), values],
            )?;
            Ok((outputs.remove(0), outputs.remove(0)))
        };
        // Reverse visit order: `c = 2·5 = 10 → 10·4 = 40 → 40·3 = 120`, with `ys[i]` still paired with `xs[i]`.
        let primals = (Array::scalar(2.0), Array::vector(vec![3.0, 4.0, 5.0]));
        let (outputs, pushforward) = differentiate_at(primals.clone()).linearize(function).unwrap();
        assert_eq!(outputs, (Array::scalar(120.0), Array::vector(vec![120.0, 40.0, 10.0])));
        // A pure carry tangent scales by the running product of the slices consumed after each visit.
        assert_eq!(
            pushforward.apply((Array::scalar(1.0), Array::vector(vec![0.0, 0.0, 0.0]))),
            Ok((Array::scalar(60.0), Array::vector(vec![60.0, 20.0, 5.0]))),
        );

        let (final_carry, pullback) = differentiate_at(primals)
            .vjp(|(carry, values)| {
                let mut outputs = carry.context().bind(
                    ArrayOperation::Scan(ScanOperation::new(1, 3).with_reverse(true)),
                    vec![product_body()],
                    &[carry.clone(), values],
                )?;
                Ok(outputs.remove(0))
            })
            .unwrap();
        assert_eq!(final_carry, Array::scalar(120.0));
        // `∂(2·3·4·5)/∂carry = 60` and `∂/∂xs = [40, 30, 24]`.
        assert_eq!(
            pullback.apply(Array::scalar(1.0)),
            Ok((Array::scalar(60.0), Array::vector(vec![40.0, 30.0, 24.0]))),
        );
    }

    /// Scan input validation compares the declared types derived from the body signature against actual input types
    /// with `Type::is_refined_by`, so actual types carrying optional metadata that the declared types leave
    /// unspecified (e.g., the normalized shardings every concrete backend array type carries) are accepted, while
    /// data type and shape mismatches are still rejected.
    #[test]
    fn test_scan_input_type_refinement() {
        use crate::arrays::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};

        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let stacked_f64 = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
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
            Err(TypeError::invalid(
                "scan input 0 has type f32[] which is incompatible with the expected type f64[]".to_string()
            )),
        );
        assert_eq!(
            operation.infer_output_types(
                &[scalar_f64, ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(4)]))],
                interfaces.as_slice(),
            ),
            Err(TypeError::invalid(
                "scan input 1 has type f64[4] which is incompatible with the expected type f64[3]".to_string()
            )),
        );
    }

    /// A scan whose body prints inside its known chain keeps the effect in the *known scan* of the known-ness
    /// split: effectful bodies skip the live-context invariance probes and go straight to the split, whose fresh
    /// probe contexts fold the all-known print into the known side. The known scan staged into the live outer trace
    /// owns the print (running it once per iteration, all before the residual side, per the effect placement
    /// contract), and the residual scan stays pure.
    #[test]
    fn test_scan_partial_evaluation_keeps_effectful_known_work_in_the_known_scan() {
        use crate::operations::debugging::PrintOperation;
        use crate::tracing::TracingContext;

        let scalar = || ArrayType::scalar(DataType::F64);
        let stacked = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));

        // Body `[acc, k, x] -> [acc + (print(k) * k) * x, k, acc + (print(k) * k) * x]`: the print sits inside the
        // otherwise-known `k * k` chain.
        let body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let acc = builder.add_input(scalar());
            let k = builder.add_input(scalar());
            let x = builder.add_input(scalar());
            let printed = builder.add_instruction(PrintOperation::new("k"), Vec::new(), vec![k]).unwrap()[0];
            let ksq = builder.add_instruction(MulOperation::new(), Vec::new(), vec![printed, k]).unwrap()[0];
            let kx = builder.add_instruction(MulOperation::new(), Vec::new(), vec![ksq, x]).unwrap()[0];
            let next_acc = builder.add_instruction(AddOperation::new(), Vec::new(), vec![acc, kx]).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(
                    vec![next_acc, k, next_acc],
                    vec![Placeholder; 3],
                    vec![Placeholder; 3],
                )
                .unwrap()
        };

        let scan = TestScanOperation::new(2, 3);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let acc_init = builder.add_input(scalar());
        let k_init = builder.add_input(scalar());
        let xs = builder.add_input(stacked.clone());
        let outputs = builder
            .add_instruction(ArrayOperation::Scan(scan), vec![body_region], vec![acc_init, k_init, xs])
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 3], vec![Placeholder; 3])
            .unwrap();

        let outer = TracingContext::<Array, ArrayOperation<Array>>::new();
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
    fn test_scan_partial_evaluation_preserves_zero_output_residual_effects() {
        use crate::operations::debugging::PrintOperation;
        use crate::partial::{PartialEvaluationOutput, PartialValue};
        use crate::tracing::TracingContext;

        let scalar = || ArrayType::scalar(DataType::F64);
        let stacked = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));

        let mut body_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let carry = body_builder.add_input(scalar());
        let input = body_builder.add_input(scalar());
        body_builder.add_instruction(PrintOperation::new("x"), Vec::new(), vec![input]).unwrap();
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![carry], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert!(body.partition(&[true, false]).unwrap().residual_program().effects().is_ordered());

        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
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
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let outer = TracingContext::<Array, ArrayOperation<Array>>::new();
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
    fn test_scan_partial_evaluation_splits_symbolic_known_carries_under_staging() {
        use crate::tracing::TracingContext;

        let scalar = || ArrayType::scalar(DataType::F64);
        let stacked = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));

        // Body `[acc, k, x] -> [acc + (k * k) * x, k, acc + (k * k) * x]`, as in the loop-invariant test below.
        let body = || {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let acc = builder.add_input(scalar());
            let k = builder.add_input(scalar());
            let x = builder.add_input(scalar());
            let ksq = builder.add_instruction(MulOperation::new(), Vec::new(), vec![k, k]).unwrap()[0];
            let kx = builder.add_instruction(MulOperation::new(), Vec::new(), vec![ksq, x]).unwrap()[0];
            let next_acc = builder.add_instruction(AddOperation::new(), Vec::new(), vec![acc, kx]).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(
                    vec![next_acc, k, next_acc],
                    vec![Placeholder; 3],
                    vec![Placeholder; 3],
                )
                .unwrap()
        };

        let scan = TestScanOperation::new(2, 3);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let body_region = builder.import_region(body().entry_region_ref());
        let acc_init = builder.add_input(scalar());
        let k_init = builder.add_input(scalar());
        let xs = builder.add_input(stacked.clone());
        let outputs = builder
            .add_instruction(ArrayOperation::Scan(scan), vec![body_region], vec![acc_init, k_init, xs])
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 3], vec![Placeholder; 3])
            .unwrap();

        let outer = TracingContext::<Array, ArrayOperation<Array>>::new();
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
    fn test_scan_partial_evaluation_splits_time_varying_known_work() {
        let scalar = || ArrayType::scalar(DataType::F64);
        let stacked = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));

        // Body `[c, x] -> [c + x * x, x * x]` over an unknown accumulator `c` and known stacked `xs`.
        let body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let c = builder.add_input(scalar());
            let x = builder.add_input(scalar());
            let xsq = builder.add_instruction(MulOperation::new(), Vec::new(), vec![x, x]).unwrap()[0];
            let next = builder.add_instruction(AddOperation::new(), Vec::new(), vec![c, xsq]).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![next, xsq], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let scan = TestScanOperation::new(1, 3);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let c_init = builder.add_input(scalar());
        let xs = builder.add_input(stacked.clone());
        let outputs = builder
            .add_instruction(ArrayOperation::Scan(scan), vec![body_region], vec![c_init, xs])
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let knowledge = vec![PartialValue::Unknown(scalar()), PartialValue::Known(Array::vector(vec![1.0, 2.0, 3.0]))];
        let evaluation = program.partially_evaluate(knowledge.as_slice()).unwrap();

        // The stacked squares were computed *during* partial evaluation by the known scan: they surface both as the
        // folded stacked output and as the residual edge feeding the unknown scan.
        assert_eq!(evaluation.outputs.len(), 2);
        assert!(matches!(&evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert!(matches!(
            &evaluation.outputs[1],
            PartialEvaluationOutput::Known(value) if value.to_f64s() == vec![1.0, 4.0, 9.0]
        ));
        assert_eq!(evaluation.inputs.len(), 2);
        assert!(matches!(&evaluation.inputs[0], PartialEvaluationInput::Unknown(0)));
        assert!(matches!(
            &evaluation.inputs[1],
            PartialEvaluationInput::Known(value) if value.to_f64s() == vec![1.0, 4.0, 9.0]
        ));

        // The residual (unknown) scan accumulates the stacked squares: interpreting it at `c = 10` reproduces the
        // full interpretation of the original program.
        let residual_outputs =
            evaluation.program.interpret(vec![Array::scalar(10.0), Array::vector(vec![1.0, 4.0, 9.0])]).unwrap();
        let expected = program.interpret(vec![Array::scalar(10.0), Array::vector(vec![1.0, 2.0, 3.0])]).unwrap();
        assert_eq!(residual_outputs[0].to_f64s(), expected[0].to_f64s());
        assert_eq!(residual_outputs[0].to_f64s(), vec![24.0]);
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
    fn test_scan_partial_evaluation_folds_loop_invariant_known_carry() {
        let scalar = || ArrayType::scalar(DataType::F64);
        let stacked = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));

        // Body `[acc, k, x] -> [acc + (k * k) * x, k, acc + (k * k) * x]`.
        let body = || {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let acc = builder.add_input(scalar());
            let k = builder.add_input(scalar());
            let x = builder.add_input(scalar());
            let ksq = builder.add_instruction(MulOperation::new(), Vec::new(), vec![k, k]).unwrap()[0];
            let kx = builder.add_instruction(MulOperation::new(), Vec::new(), vec![ksq, x]).unwrap()[0];
            let next_acc = builder.add_instruction(AddOperation::new(), Vec::new(), vec![acc, kx]).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(
                    vec![next_acc, k, next_acc],
                    vec![Placeholder; 3],
                    vec![Placeholder; 3],
                )
                .unwrap()
        };

        // Flat program over `[acc_init, k_init, xs]` staging the scan (two carries, one scanned input, length 3); its
        // outputs are `[final_acc, final_k, stacked_acc]`.
        let scan = TestScanOperation::new(2, 3);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let body_region = builder.import_region(body().entry_region_ref());
        let acc_init = builder.add_input(scalar());
        let k_init = builder.add_input(scalar());
        let xs = builder.add_input(stacked.clone());
        let outputs = builder
            .add_instruction(ArrayOperation::Scan(scan), vec![body_region], vec![acc_init, k_init, xs])
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 3], vec![Placeholder; 3])
            .unwrap();

        let knowledge = vec![
            PartialValue::Unknown(scalar()),
            PartialValue::Known(Array::scalar(2.0)),
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
        assert_eq!(residual_scan.length(), &Dimension::Static(3));
        let residual_body = evaluation.program.region_ref(residual_instruction.regions()[0]).unwrap().to_program();
        assert!(residual_body.instructions().len() < body().instructions().len());
        assert_eq!(residual_body.instructions().len(), 2);

        // Correctness: interpreting the residual program reproduces the original program on the same concrete inputs.
        let runtime = |acc: f64, xs: Vec<f64>| -> Vec<Array> {
            let arguments = evaluation
                .inputs
                .iter()
                .map(|residual_input| match residual_input {
                    PartialEvaluationInput::Known(value) => value.clone(),
                    PartialEvaluationInput::Unknown(index) => match index {
                        0 => Array::scalar(acc),
                        _ => Array::vector(xs.clone()),
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
            program.interpret(vec![Array::scalar(acc), Array::scalar(k), Array::vector(xs)]).unwrap()
        };

        let reassembled = runtime(1.0, vec![5.0, 6.0, 7.0]);
        let expected = original(1.0, 2.0, vec![5.0, 6.0, 7.0]);
        assert_eq!(
            reassembled.iter().map(|value| value.to_f64s()).collect::<Vec<_>>(),
            expected.iter().map(|value| value.to_f64s()).collect::<Vec<_>>()
        );
        // `acc` threads `1 -> 1 + 4*5 -> 21 + 4*6 -> 45 + 4*7 = 73`; the stacked output records `[21, 45, 73]`; the
        // loop-invariant `k` final carry stays `2`.
        assert_eq!(reassembled[0].to_f64s(), vec![73.0]);
        assert_eq!(reassembled[1].to_f64s(), vec![2.0]);
        assert_eq!(reassembled[2].to_f64s(), vec![21.0, 45.0, 73.0]);
    }

    type TestOperation = ArrayOperation<Array>;
    type TestEagerContext = EagerContext<Array, TestOperation>;

    /// Builds a body for zero-length scan tests whose first stacked result follows the carry's mapped axis and whose
    /// second stacked result is a replicated constant.
    fn zero_length_body(r#type: ArrayType) -> Program<Array, TestOperation, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let carry = builder.add_input(r#type.clone());
        let _x = builder.add_input(r#type.clone());
        let constant = builder.add_constant(Array::from_f64s(r#type, vec![7.0]));
        builder
            .build(
                vec![carry, carry, constant],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder, Placeholder],
            )
            .unwrap()
    }

    /// Builds the `f64` array type with the provided static dimensions.
    fn f64_type(dimensions: &[usize]) -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(dimensions.iter().copied().map(Dimension::Static).collect()))
    }

    /// Builds a cumulative-product [`ScanOperation`] whose nested depth matches `lengths`, returning the
    /// payload-free operation together with its body region program.
    fn product_scan_with_lengths(
        lengths: &[usize],
    ) -> (ScanOperation<Array>, Program<Array, TestOperation, Vec<Array>, Vec<Array>>) {
        assert!(!lengths.is_empty());
        if lengths.len() == 1 {
            return (TestScanOperation::new(1, lengths[0]), product_body());
        }
        let (inner_scan, inner_body) = product_scan_with_lengths(&lengths[1..]);
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let inner_body_region = builder.import_region(inner_body.entry_region_ref());
        let carry = builder.add_input(ArrayType::scalar(DataType::F64));
        let xs = builder.add_input(f64_type(&lengths[1..]));
        let outputs = builder
            .add_instruction(TestOperation::Scan(inner_scan), vec![inner_body_region], vec![carry, xs])
            .unwrap()
            .to_vec();
        let body = builder.build(outputs, vec![Placeholder, Placeholder], vec![Placeholder, Placeholder]).unwrap();
        (TestScanOperation::new(1, lengths[0]), body)
    }

    /// Builds the cumulative-product [`ScanOperation`] over three iterations used by the differentiation tests,
    /// returning the payload-free operation together with its body region program.
    fn product_scan() -> (ScanOperation<Array>, Program<Array, TestOperation, Vec<Array>, Vec<Array>>) {
        product_scan_with_lengths(&[3])
    }

    /// Batches `scan` through the public [`BatchingContext::bind`] path with `body` as an owned attached region.
    fn batch_scan(
        context: &BatchingContext<TestEagerContext, ArrayBatching>,
        scan: ScanOperation<Array>,
        body: Program<Array, TestOperation, Vec<Array>, Vec<Array>>,
        inputs: Vec<ArrayBatch<Array>>,
    ) -> Vec<ArrayBatch<Array>> {
        let tracer_inputs =
            inputs.into_iter().map(|input| BatchingTracer::new(context.clone(), input)).collect::<Vec<_>>();
        context
            .bind(TestOperation::Scan(scan), [body], tracer_inputs.as_slice())
            .unwrap()
            .into_iter()
            .map(|output| output.batch().clone())
            .collect()
    }

    #[test]
    fn test_scan_reorder_program_boundary_supports_nullary_programs() {
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let first = builder.add_constant(Array::scalar(1.0));
        let second = builder.add_constant(Array::scalar(2.0));
        let program = builder.build(vec![first, second], Vec::new(), vec![Placeholder, Placeholder]).unwrap();

        let reordered = reorder_program_boundary(&program, &[], &[1, 0]).unwrap();

        assert_eq!(reordered.input_count(), 0);
        assert_eq!(
            reordered.outputs().map(|output| output.as_constant().unwrap().to_f64s()).collect::<Vec<_>>(),
            vec![vec![2.0], vec![1.0]],
        );
    }

    #[test]
    fn test_scan_thread_known_carries_projects_out_dynamic_zero_space_inputs() {
        type CompositeValue = ArrayIrValue<Array>;
        type CompositeOperation = ArrayIrOperation<Array>;

        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let key_type = ArrayIrType::Array(ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Dynamic(extent)])));
        let accumulator_type = ArrayIrType::Array(ArrayType::scalar(DataType::F64));
        let mut builder = ProgramBuilder::<CompositeValue, CompositeOperation>::new();
        let _key_cotangent = builder.add_input(key_type.cotangent().unwrap());
        let accumulator_cotangent = builder.add_input(accumulator_type.cotangent().unwrap());
        let _known_key = builder.add_input(key_type.clone());
        let transposed = builder
            .build::<Vec<CompositeValue>, Vec<CompositeValue>>(
                vec![accumulator_cotangent],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap();

        // The zero-space key cotangent is an unused transpose-boundary input. The reversed scan body should erase
        // that slot and thread the real key value through its carry slot without constructing a dynamic zero.
        let threaded =
            thread_known_carries(transposed, &[key_type.clone(), accumulator_type.clone()], &[false, true], 2).unwrap();
        assert_eq!(threaded.input_types(), vec![key_type.clone(), accumulator_type.clone()]);
        assert_eq!(threaded.output_types(), vec![key_type, accumulator_type]);
        assert_eq!(threaded.instructions().len(), 0);
    }

    /// A dead *stacked* output's cotangent has the scan-length-prefixed type `f64[length, k]`, which no single value at
    /// the transpose boundary carries: the length rides the runtime length operand and the inner extent rides the
    /// per-iteration carry cotangent. Identity-directed materialization assembles the zero from both, so the reversed
    /// scan gets a well-typed operand instead of failing on an unconstructible nullary zero.
    #[test]
    fn test_scan_transpose_materializes_dead_dynamic_stacked_cotangent() {
        type CompositeValue = ArrayIrValue<Array>;
        type CompositeOperation = ArrayIrOperation<Array>;

        let length = DimensionVariable::new("length", DimensionBounds::positive(Some(8)).unwrap());
        let k = DimensionVariable::new("k", DimensionBounds::positive(Some(8)).unwrap());
        let item_type =
            ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(k.clone())])));
        let stacked_type = ArrayIrType::Array(ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(length.clone()), Dimension::Dynamic(k)]),
        ));

        // The body maps `[carry, x]` to `[carry + x, carry]`, so the scan produces a final carry and a stacked
        // per-iteration output. Both halves are linear in the operands.
        let mut body_builder = ProgramBuilder::<CompositeValue, CompositeOperation>::new();
        let carry = body_builder.add_input(item_type.clone());
        let x = body_builder.add_input(item_type.clone());
        let sum = body_builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![carry, x])
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<CompositeValue>, Vec<CompositeValue>>(
                vec![sum, carry],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        // Only the final carry is a program output, so the stacked output is dead and its cotangent is a structural
        // zero of the dynamically shaped stacked type.
        let mut builder = ProgramBuilder::<CompositeValue, CompositeOperation>::new();
        let runtime_length = builder.add_input(ArrayIrType::Dimension(DimensionType::new(length.clone())));
        let carry_init = builder.add_input(item_type);
        let stacked_input = builder.add_input(stacked_type.clone());
        let body_region = builder.import_region(body.entry_region_ref());
        let outputs = builder
            .add_instruction(
                CompositeOperation::Scan(ScanOperation::new(1, Dimension::Dynamic(length))),
                vec![body_region],
                vec![carry_init, stacked_input, runtime_length],
            )
            .unwrap()
            .to_vec();
        assert_eq!(builder.atoms()[outputs[1].index()].r#type().as_ref(), &stacked_type);
        let program = builder
            .build::<Vec<CompositeValue>, Vec<CompositeValue>>(
                vec![outputs[0]],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap();

        // Transposing with respect to the carry initializer and the stacked operand reaches the dead stacked
        // cotangent. The pullback reads the inner extent off the live carry cotangent, reuses the runtime length
        // operand, and stages the mixed dynamic zero over both.
        let pullback = program.transpose_with_respect_to(&[1, 2]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[k], %1:dimension<length \u{2208} [1, 8)> .
                let %2:dimension<k \u{2208} [1, 8)> = dimension_size [axis=0] %0
                    %3:f64[length, k] = zero [type=f64[length, k]] %1 %2
                    %4:f64[k], %5:f64[length, k] = scan [carry_count=1, length=length, reverse=true] %0 %3 %1 [
                        body={
                            lambda %0:f64[k], %1:f64[k] .
                            let %2:f64[k] = add %1 %0
                            in (%2, %0)
                        },
                    ]
                in (%4, %5)
            "}
            .trim_end(),
        );
    }

    /// The fused JVP rule stages exactly one scan with doubled carries and **no** per-iteration residual stacks:
    /// pure forward mode pays a single loop pass and no reverse-mode storage. Residual stacks appear only when
    /// [`Program::linearize`] directly differentiates over partial evaluation (its known scan then stacks the
    /// known→unknown edges), which the trailing assertion pins.
    #[test]
    fn test_scan_jvp_stages_one_fused_scan_with_no_residual_stacks() {
        use crate::arrays::{Dimension, Shape};
        use crate::tracing::DomainTracer;

        let (scan, scan_body) = product_scan();
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |(init, xs): (
                DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
                DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
            )| {
                let mut outputs = init.context().stage_operation(
                    TestOperation::Scan(scan),
                    vec![scan_body.clone()],
                    &[&init, &xs],
                )?;
                let ys = outputs.remove(1);
                Ok((outputs.remove(0), ys))
            },
            (ArrayType::scalar(DataType::F64), ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]))),
        )
        .unwrap();
        let program = program.to_flat_program();

        let jvp = program.jvp().unwrap().into_simplified().unwrap();
        let scans = jvp
            .instructions()
            .iter()
            .filter_map(|instruction| match instruction.operation() {
                TestOperation::Scan(operation) => Some((operation, instruction)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(scans.len(), 1);
        let (fused_scan, fused_instruction) = scans[0];
        assert_eq!(fused_scan.carry_count(), 2);
        // The fused body is `[primal_carry, tangent_carry, primal_x, tangent_x] ->
        // [primal_carry', tangent_carry', primal_y, tangent_y]`: doubled arity and nothing else.
        let fused_body = jvp.region_ref(fused_instruction.regions()[0]).unwrap().to_program();
        assert_eq!(fused_body.input_types().len(), 4);
        assert_eq!(fused_body.output_types().len(), 4);

        // Linearizing the same program is what materializes residual stacks, as known-scan edges.
        let linearization = program.linearize().unwrap();
        assert!(linearization.residual_count() >= 1);
    }

    #[test]
    fn test_scan_differentiation_with_zero_space_key_carry() {
        use crate::arrays::{Dimension, Shape};

        // A scan whose carries mix a differentiable accumulator with a zero-differential-space element — here a
        // `u64` key, the shape of every keyed training loop. The compact fused-JVP contract omits the key's tangent
        // slot on both the carry and output boundaries, and reverse mode returns a typed zero-space cotangent for
        // the key input at the public boundary.
        fn keyed_product_body() -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let accumulator = builder.add_input(ArrayType::scalar(DataType::F64));
            let key = builder.add_input(ArrayType::scalar(DataType::U64));
            let slice = builder.add_input(ArrayType::scalar(DataType::F64));
            let product =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![accumulator, slice]).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(
                    vec![product, key, product],
                    vec![Placeholder; 3],
                    vec![Placeholder; 3],
                )
                .unwrap()
        }
        fn stage_keyed_product_scan<V: Value<Type = ArrayType>>(
            accumulator: V,
            key: V,
            values: V,
        ) -> Result<(V, V), ProgramError>
        where
            V::DispatchDomain: Context<Type = ArrayType, Constant = Array, Operation = ArrayOperation<Array>>,
        {
            let mut outputs = accumulator.dispatch_domain().bind(
                ArrayOperation::Scan(ScanOperation::new(2, 3)),
                vec![keyed_product_body()],
                &[accumulator.clone(), key, values],
            )?;
            let stacked = outputs.remove(2);
            Ok((outputs.remove(0), stacked))
        }

        // Forward mode: the fused scan carries `[acc, key, live(ȧcc)]` and its body omits the key tangents.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |(accumulator, key, values)| stage_keyed_product_scan(accumulator, key, values),
            (
                ArrayType::scalar(DataType::F64),
                ArrayType::scalar(DataType::U64),
                ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])),
            ),
        )
        .unwrap();
        let program = program.to_flat_program();
        let jvp = program.jvp().unwrap().into_simplified().unwrap();
        let scans = jvp
            .instructions()
            .iter()
            .filter_map(|instruction| match instruction.operation() {
                TestOperation::Scan(operation) => Some((operation, instruction)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(scans.len(), 1);
        let (fused_scan, fused_instruction) = scans[0];
        assert_eq!(fused_scan.carry_count(), 3);
        let fused_body = jvp.region_ref(fused_instruction.regions()[0]).unwrap().to_program();
        assert_eq!(fused_body.input_types().len(), 5);
        assert_eq!(fused_body.output_types().len(), 5);

        // Reverse mode through the same scan: the accumulator and slice cotangents match the keyless product scan,
        // while the key input receives a typed zero-space cotangent at the public boundary.
        let ((output, stacked), pullback) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .vjp(
                |(accumulator, key, values), ()| stage_keyed_product_scan(accumulator, key, values),
                (
                    Array::scalar(1.0),
                    Array::from_elements(ArrayType::scalar(DataType::U64), &[7u64]).unwrap(),
                    Array::vector(vec![2.0, 3.0, 4.0]),
                ),
                (),
            )
            .unwrap();
        assert_eq!(output.to_f64s(), vec![24.0]);
        assert_eq!(stacked.to_f64s(), vec![2.0, 6.0, 24.0]);
        let (accumulator_cotangent, key_cotangent, values_cotangent) =
            pullback.apply((Array::scalar(1.0), Array::vector(vec![0.0, 0.0, 0.0]))).unwrap();
        assert_eq!(accumulator_cotangent.to_f64s(), vec![24.0]);
        assert_eq!(key_cotangent, Array::new(ArrayType::scalar(DataType::Zero), Vec::new()).unwrap());
        assert_eq!(values_cotangent.to_f64s(), vec![12.0, 8.0, 6.0]);
    }

    #[test]
    fn test_scan_jvp_propagates_tangents_through_linear_scan() {
        // Cumulative product over `xs = [2, 3, 4]` starting at `init = 1`: the final carry is 24 and the running
        // products are `[2, 6, 24]`. A unit tangent on `init` propagates as `d(init * x0 * x1 * x2)/d(init) = 24`
        // on the final carry and `[2, 6, 24]` on the stacked outputs.
        let (scan, scan_body) = product_scan();
        let ((carry, ys), (carry_tangent, ys_tangent)) = crate::EagerContext::<
            crate::Array,
            crate::ArrayOperation<crate::Array>,
        >::new()
        .differentiate_at((Array::scalar(1.0), Array::vector(vec![2.0, 3.0, 4.0])))
        .jvp((Array::scalar(1.0), Array::vector(vec![0.0, 0.0, 0.0])), move |(init, xs)| {
            let mut outputs =
                init.context()
                    .bind(TestOperation::Scan(scan), vec![scan_body.clone()], &[init.clone(), xs.clone()])?;
            let ys = outputs.remove(1);
            Ok((outputs.remove(0), ys))
        })
        .unwrap();
        assert_eq!(carry.to_f64s(), vec![24.0]);
        assert_eq!(ys.to_f64s(), vec![2.0, 6.0, 24.0]);
        assert_eq!(carry_tangent.to_f64s(), vec![24.0]);
        assert_eq!(ys_tangent.to_f64s(), vec![2.0, 6.0, 24.0]);

        // A unit tangent on `xs[1]` propagates as `d(init * x0 * x1 * x2)/d(x1) = init * x0 * x2 = 8` on the final
        // carry and `[0, 2, 8]` on the stacked outputs (`y0` does not depend on `x1`).
        let (scan, scan_body) = product_scan();
        let ((carry, _), (carry_tangent, ys_tangent)) = crate::EagerContext::<
            crate::Array,
            crate::ArrayOperation<crate::Array>,
        >::new()
        .differentiate_at((Array::scalar(1.0), Array::vector(vec![2.0, 3.0, 4.0])))
        .jvp((Array::scalar(0.0), Array::vector(vec![0.0, 1.0, 0.0])), move |(init, xs)| {
            let mut outputs =
                init.context()
                    .bind(TestOperation::Scan(scan), vec![scan_body.clone()], &[init.clone(), xs.clone()])?;
            let ys = outputs.remove(1);
            Ok((outputs.remove(0), ys))
        })
        .unwrap();
        assert_eq!(carry.to_f64s(), vec![24.0]);
        assert_eq!(carry_tangent.to_f64s(), vec![8.0]);
        assert_eq!(ys_tangent.to_f64s(), vec![0.0, 2.0, 8.0]);
    }

    #[test]
    fn test_scan_jvp_supports_nested_scans_in_linear_scan_bodies() {
        // Nested scans differentiate by recursively replaying the inner linear scan inside each outer scan iteration.
        // The final carry is the product of every element, and a unit tangent on the initial carry follows the same
        // cumulative-product path through both scan levels.
        let (scan, scan_body) = product_scan_with_lengths(&[2, 3]);
        let ((carry, ys), (carry_tangent, ys_tangent)) = crate::EagerContext::<
            crate::Array,
            crate::ArrayOperation<crate::Array>,
        >::new()
        .differentiate_at((Array::scalar(1.0), Array::matrix(2, 3, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0])))
        .jvp((Array::scalar(1.0), Array::matrix(2, 3, vec![0.0; 6])), move |(init, xs)| {
            let mut outputs =
                init.context()
                    .bind(TestOperation::Scan(scan), vec![scan_body.clone()], &[init.clone(), xs.clone()])?;
            let ys = outputs.remove(1);
            Ok((outputs.remove(0), ys))
        })
        .unwrap();
        assert_eq!(carry.to_f64s(), vec![5040.0]);
        assert_eq!(ys.to_f64s(), vec![2.0, 6.0, 24.0, 120.0, 720.0, 5040.0]);
        assert_eq!(carry_tangent.to_f64s(), vec![5040.0]);
        assert_eq!(ys_tangent.to_f64s(), vec![2.0, 6.0, 24.0, 120.0, 720.0, 5040.0]);
    }

    #[test]
    fn test_scan_jvp_supports_three_nested_scans_in_linear_scan_bodies() {
        // Three levels catches the recursive fixed point that failed for nested scan bodies: the middle scan's
        // linear body contains another scan whose body also has scan-local residual references.
        let (scan, scan_body) = product_scan_with_lengths(&[2, 2, 2]);
        let xs_type = f64_type(&[2, 2, 2]);
        let ((carry, ys), (carry_tangent, ys_tangent)) = crate::EagerContext::<
            crate::Array,
            crate::ArrayOperation<crate::Array>,
        >::new()
        .differentiate_at((
            Array::scalar(1.0),
            Array::from_f64s(xs_type.clone(), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        ))
        .jvp((Array::scalar(1.0), Array::from_f64s(xs_type, vec![0.0; 8])), move |(init, xs)| {
            let mut outputs =
                init.context()
                    .bind(TestOperation::Scan(scan), vec![scan_body.clone()], &[init.clone(), xs.clone()])?;
            let ys = outputs.remove(1);
            Ok((outputs.remove(0), ys))
        })
        .unwrap();
        assert_eq!(carry.to_f64s(), vec![362880.0]);
        assert_eq!(ys.to_f64s(), vec![2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0]);
        assert_eq!(carry_tangent.to_f64s(), vec![362880.0]);
        assert_eq!(ys_tangent.to_f64s(), vec![2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0]);
    }

    #[test]
    fn test_scan_batching_lifts_batched_carries() {
        // Batching a scan whose carry is mapped at axis 0 threads the batch axis through every iteration: each
        // batch item runs its own cumulative product over the shared `xs = [2, 3, 4]`, and the stacked outputs
        // gain the scan axis in front of the batch axis.
        let (scan, scan_body) = product_scan();
        let context = BatchingContext::new(TestEagerContext::new(), 3);
        let carries = {
            let value = Array::vector(vec![1.0, 2.0, 3.0]);
            ArrayBatch::new(value, Some(0))
        }
        .unwrap();
        let stacked_inputs = ArrayBatch::replicated(Array::vector(vec![2.0, 3.0, 4.0]));
        let outputs = batch_scan(&context, scan, scan_body, vec![carries, stacked_inputs]);
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().to_f64s(), vec![24.0, 48.0, 72.0]);
        assert_eq!(outputs[1].batch_axis(), BatchAxis::new(1));
        assert_eq!(outputs[1].r#type().shape().dimensions(), &[Dimension::Static(3), Dimension::Static(3)]);
        assert_eq!(outputs[1].value().to_f64s(), vec![2.0, 4.0, 6.0, 6.0, 12.0, 18.0, 24.0, 48.0, 72.0]);
    }

    #[test]
    fn test_scan_batching_lifts_batched_stacked_inputs() {
        // Batching a scan whose stacked input is mapped at axis 0 reads each iteration's slice along the per-item
        // leading axis (packed axis 1 when the batch axis sits at 0), so every batch item scans its own row.
        let (scan, scan_body) = product_scan();
        let context = BatchingContext::new(TestEagerContext::new(), 2);
        let carries = ArrayBatch::replicated(Array::scalar(1.0));
        let stacked_inputs = {
            let value = Array::matrix(2, 3, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
            ArrayBatch::new(value, Some(0))
        }
        .unwrap();
        let outputs = batch_scan(&context, scan, scan_body, vec![carries, stacked_inputs]);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().to_f64s(), vec![24.0, 210.0]);
        assert_eq!(outputs[1].batch_axis(), BatchAxis::new(1));
        assert_eq!(outputs[1].r#type().shape().dimensions(), &[Dimension::Static(3), Dimension::Static(2)]);
        assert_eq!(outputs[1].value().to_f64s(), vec![2.0, 5.0, 6.0, 30.0, 24.0, 210.0]);

        // A trailing batch axis (packed `[3, 2]` with the batch axis at 1) reads the same iterations, so the
        // outputs are identical.
        let (scan, scan_body) = product_scan();
        let context = BatchingContext::new(TestEagerContext::new(), 2);
        let carries = ArrayBatch::replicated(Array::scalar(1.0));
        let stacked_inputs = {
            let value = Array::matrix(3, 2, vec![2.0, 5.0, 3.0, 6.0, 4.0, 7.0]);
            ArrayBatch::new(value, Some(1))
        }
        .unwrap();
        let outputs = batch_scan(&context, scan, scan_body, vec![carries, stacked_inputs]);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().to_f64s(), vec![24.0, 210.0]);
        assert_eq!(outputs[1].batch_axis(), BatchAxis::new(1));
        assert_eq!(outputs[1].value().to_f64s(), vec![2.0, 5.0, 6.0, 30.0, 24.0, 210.0]);
    }

    #[test]
    fn test_scan_batching_threads_batched_carries_and_inputs() {
        // Batching both operands pairs batch item `i` of the carries with batch item `i` of the stacked inputs.
        let (scan, scan_body) = product_scan();
        let context = BatchingContext::new(TestEagerContext::new(), 2);
        let carries = {
            let value = Array::vector(vec![1.0, 10.0]);
            ArrayBatch::new(value, Some(0))
        }
        .unwrap();
        let stacked_inputs = {
            let value = Array::matrix(2, 3, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
            ArrayBatch::new(value, Some(0))
        }
        .unwrap();
        let outputs = batch_scan(&context, scan, scan_body, vec![carries, stacked_inputs]);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().to_f64s(), vec![24.0, 2100.0]);
        assert_eq!(outputs[1].batch_axis(), BatchAxis::new(1));
        assert_eq!(outputs[1].value().to_f64s(), vec![2.0, 50.0, 6.0, 300.0, 24.0, 2100.0]);
    }

    #[test]
    fn test_scan_batching_respects_reverse_visit_order() {
        // A reversed batched scan visits the iterations from the back while keeping output iteration `i`
        // aligned with input iteration `i`: the reversed cumulative product over `[2, 3, 4]` is `[24, 12, 4]` per
        // batch item.
        let (scan, scan_body) = product_scan();
        let scan = scan.with_reverse(true);
        let context = BatchingContext::new(TestEagerContext::new(), 2);
        let carries = ArrayBatch::replicated(Array::scalar(1.0));
        let stacked_inputs = {
            let value = Array::matrix(2, 3, vec![2.0, 3.0, 4.0, 2.0, 3.0, 4.0]);
            ArrayBatch::new(value, Some(0))
        }
        .unwrap();
        let outputs = batch_scan(&context, scan, scan_body, vec![carries, stacked_inputs]);
        assert_eq!(outputs[0].value().to_f64s(), vec![24.0, 24.0]);
        assert_eq!(outputs[1].batch_axis(), BatchAxis::new(1));
        assert_eq!(outputs[1].value().to_f64s(), vec![24.0, 24.0, 12.0, 12.0, 4.0, 4.0]);
    }

    #[test]
    fn test_scan_batching_preserves_stacked_output_batch_placement() {
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let logical_type =
                ArrayType::scalar(DataType::F64).with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
            let carry_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
                .unwrap()
                .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                .unwrap();
            let carry_type = f64_type(&[2]).with_sharding(carry_sharding.clone()).unwrap();
            let carries = ArrayBatch::new(Array::from_f64s(carry_type, vec![1.0, 2.0]), BatchAxis::new(0)).unwrap();
            let stack_type = f64_type(&[3]).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
            let stacked_inputs = ArrayBatch::replicated(Array::from_f64s(stack_type, vec![2.0, 3.0, 4.0]));
            let context =
                BatchingContext::new(TestEagerContext::new(), 2).with_axis_sharding(ShardingDimension::sharded(["x"]));

            let outputs = batch_scan(
                &context,
                TestScanOperation::new(1, 3),
                product_body_with_type(logical_type),
                vec![carries, stacked_inputs],
            );

            assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
            assert_eq!(outputs[0].r#type().sharding().unwrap().dimensions(), carry_sharding.dimensions());
            assert_eq!(outputs[0].value().to_f64s(), vec![24.0, 48.0]);
            assert_eq!(outputs[1].batch_axis(), BatchAxis::new(1));
            assert_eq!(outputs[1].r#type().shape().dimensions(), &[Dimension::Static(3), Dimension::Static(2)]);
            assert_eq!(
                outputs[1].r#type().sharding().unwrap().dimensions(),
                &[ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
            );
            assert_eq!(outputs[1].value().to_f64s(), vec![2.0, 4.0, 6.0, 12.0, 24.0, 48.0]);
        }
    }

    /// Batching a capture-free scan under a staging parent stages exactly one batched scan — with the replicated
    /// carry widened through a staged broadcast and the batched stacked input realigned off the leading scan
    /// dimension — instead of unrolling the loop into per-iteration body copies.
    #[test]
    fn test_scan_batching_stages_one_batched_scan_under_tracing() {
        let parent = DomainTracingContext::<TestEagerContext>::new();
        let builder = parent.builder().clone();
        let carry_atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let xs_atom = builder.borrow_mut().add_input(f64_type(&[2, 3]));
        let carry_tracer = parent.tracer(carry_atom, None);
        let xs_tracer = parent.tracer(xs_atom, None);
        let (final_carry, ys) = batch(
            |(carry, xs)| {
                let mut outputs = carry.context().bind(
                    TestOperation::Scan(TestScanOperation::new(1, 3)),
                    vec![product_body()],
                    &[carry.clone(), xs.clone()],
                )?;
                Ok((outputs.remove(0), outputs.remove(0)))
            },
            (carry_tracer, xs_tracer),
            (BatchAxis::replicated(), BatchAxis::new(0)),
            (BatchAxis::new(0), BatchAxis::new(0)),
            None,
        )
        .unwrap();
        let output_atoms = vec![final_carry.atom_id().unwrap(), ys.atom_id().unwrap()];
        let program = builder
            .borrow()
            .clone()
            .build::<(Array, Array), Vec<Array>>(
                output_atoms,
                (Placeholder, Placeholder),
                vec![Placeholder, Placeholder],
            )
            .unwrap();

        // Exactly one scan is staged and the loop body is not unrolled into the enclosing trace.
        let scan_count =
            program.instructions().iter().filter(|instruction| instruction.operation().name() == "scan").count();
        assert_eq!(scan_count, 1, "{program}");
        let unrolled_body_count =
            program.instructions().iter().filter(|instruction| instruction.operation().name() == "mul").count();
        assert_eq!(unrolled_body_count, 0, "{program}");

        // Interpreting the staged program computes per-item cumulative products, with the replicated carry
        // broadcast across the batch.
        let xs = Array::from_f64s(f64_type(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let outputs = program.interpret((Array::scalar(1.0), xs)).unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![24.0, 210.0]);
        assert_eq!(outputs[1].to_f64s(), vec![2.0, 6.0, 24.0, 5.0, 30.0, 210.0]);
    }

    /// The structural rule iterates the body's carry axes to a fixed point with natural output axes and then
    /// instantiates the body at the joined carry and stacked-slice axes. `AlignEachTo` stages axis movement only where
    /// a natural axis differs from a mapped target, so when the stabilizing pass already discovered those targets its
    /// program *is* the aligned body and is not rebuilt.
    #[test]
    fn test_scan_batching_reuses_the_stabilized_body_discovery_program() {
        let regions = vec![product_body()];

        // Both carries and stacked slices are batched from the start, and the stacked input already carries its batch
        // axis off the leading scan dimension, so the first pass widens nothing and its discovered axes already equal
        // the joined targets: exactly one structural pass.
        let parent = DomainTracingContext::<TestEagerContext>::new();
        let builder = parent.builder().clone();
        let carry_atom = builder.borrow_mut().add_input(f64_type(&[2]));
        let xs_atom = builder.borrow_mut().add_input(f64_type(&[3, 2]));
        let carry = parent.tracer(carry_atom, None);
        let xs = parent.tracer(xs_atom, None);
        let context = BatchingContext::new(parent, 2);
        let inputs =
            vec![ArrayBatch::new(carry, BatchAxis::new(0)).unwrap(), ArrayBatch::new(xs, BatchAxis::new(1)).unwrap()];
        let driver = CountingBatchingDriver::new(&regions);
        let outputs = TestScanOperation::new(1, 3).batch(&context, &driver, inputs.as_slice()).unwrap().into_parts().0;
        assert_eq!(driver.batch_program_calls(), 1);
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[1].batch_axis(), BatchAxis::new(1));
        let program = builder
            .borrow()
            .clone()
            .build::<(Array, Array), Vec<Array>>(
                vec![outputs[0].value().atom_id().unwrap(), outputs[1].value().atom_id().unwrap()],
                (Placeholder, Placeholder),
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2], %1:f64[3, 2] .
                let %2:f64[2], %3:f64[3, 2] = scan [carry_count=1, length=3, reverse=false] %0 %1 [
                    body={
                        lambda %0:f64[2], %1:f64[2] .
                        let %2:f64[2] = mul %0 %1
                        in (%2, %2)
                    },
                ]
                in (%2, %3)"},
        );
        let outputs = program
            .interpret((
                Array::from_f64s(f64_type(&[2]), vec![1.0, 1.0]),
                Array::from_f64s(f64_type(&[3, 2]), vec![2.0, 5.0, 3.0, 6.0, 4.0, 7.0]),
            ))
            .unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![24.0, 210.0]);
        assert_eq!(outputs[1].to_f64s(), vec![2.0, 5.0, 6.0, 30.0, 24.0, 210.0]);

        // A replicated carry whose next-carry output is batched widens once, so the fixed point runs two natural
        // passes. The second (stabilizing) pass is still reused instead of being replayed a third time.
        let parent = DomainTracingContext::<TestEagerContext>::new();
        let builder = parent.builder().clone();
        let carry_atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let xs_atom = builder.borrow_mut().add_input(f64_type(&[2, 3]));
        let carry = parent.tracer(carry_atom, None);
        let xs = parent.tracer(xs_atom, None);
        let context = BatchingContext::new(parent, 2);
        let inputs = vec![ArrayBatch::replicated(carry), ArrayBatch::new(xs, BatchAxis::new(0)).unwrap()];
        let driver = CountingBatchingDriver::new(&regions);
        let outputs = TestScanOperation::new(1, 3).batch(&context, &driver, inputs.as_slice()).unwrap().into_parts().0;
        assert_eq!(driver.batch_program_calls(), 2);
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[1].batch_axis(), BatchAxis::new(1));
    }

    #[test]
    fn test_scan_batching_infers_zero_length_mapped_and_replicated_outputs_eagerly() {
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let logical_type =
                ArrayType::scalar(DataType::F64).with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
            let carry_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
                .unwrap()
                .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                .unwrap();
            let carry_type = f64_type(&[2]).with_sharding(carry_sharding.clone()).unwrap();
            let carries = ArrayBatch::new(Array::from_f64s(carry_type, vec![1.0, 2.0]), BatchAxis::new(0)).unwrap();
            let stack_type = f64_type(&[0]).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
            let stacked_inputs = ArrayBatch::replicated(Array::from_f64s(stack_type, Vec::new()));
            let context =
                BatchingContext::new(TestEagerContext::new(), 2).with_axis_sharding(ShardingDimension::sharded(["x"]));

            let outputs = batch_scan(
                &context,
                TestScanOperation::new(1, 0),
                zero_length_body(logical_type),
                vec![carries, stacked_inputs],
            );

            assert_eq!(outputs.len(), 3);
            assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
            assert_eq!(outputs[0].r#type().shape().dimensions(), &[Dimension::Static(2)]);
            assert_eq!(outputs[0].r#type().sharding().unwrap().dimensions(), carry_sharding.dimensions());
            assert_eq!(outputs[0].value().to_f64s(), vec![1.0, 2.0]);
            assert_eq!(outputs[1].batch_axis(), BatchAxis::new(1));
            assert_eq!(outputs[1].r#type().shape().dimensions(), &[Dimension::Static(0), Dimension::Static(2)]);
            assert_eq!(
                outputs[1].r#type().sharding().unwrap().dimensions(),
                &[ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
            );
            assert!(outputs[1].value().storage_bytes().is_empty());
            assert_eq!(outputs[2].batch_axis(), BatchAxis::replicated());
            assert_eq!(outputs[2].r#type().shape().dimensions(), &[Dimension::Static(0)]);
            assert_eq!(outputs[2].r#type().sharding().unwrap().dimensions(), &[ShardingDimension::replicated()],);
            assert!(outputs[2].value().storage_bytes().is_empty());
        }
    }

    #[test]
    fn test_scan_batching_infers_zero_length_mapped_and_replicated_outputs_while_tracing() {
        use std::rc::Rc;

        use crate::tracing::TracingContext;

        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let carry_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
                .unwrap()
                .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                .unwrap();
            let carry_type = f64_type(&[2]).with_sharding(carry_sharding.clone()).unwrap();
            let stack_type = f64_type(&[0]).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
            let parent = TracingContext::<Array, TestOperation>::new();
            let builder = parent.builder().clone();
            let carry_atom = builder.borrow_mut().add_input(carry_type.clone());
            let stack_atom = builder.borrow_mut().add_input(stack_type.clone());
            let context = BatchingContext::new(parent.clone(), 2).with_axis_sharding(ShardingDimension::sharded(["x"]));
            let carries = ArrayBatch::new(parent.tracer(carry_atom, None), BatchAxis::new(0)).unwrap();
            let stacked_inputs = ArrayBatch::replicated(parent.tracer(stack_atom, None));
            // The body's boundary types derive from the carry's unbatched per-item type (like a traced-over-inputs
            // body would), so its metadata — including any varying-manual-axes marker — matches the actual carries.
            let logical_type = carries.unbatched_type();
            let tracer_inputs =
                [BatchingTracer::new(context.clone(), carries), BatchingTracer::new(context.clone(), stacked_inputs)];
            let outputs = context
                .bind(
                    TestOperation::Scan(TestScanOperation::new(1, 0)),
                    [zero_length_body(logical_type)],
                    &tracer_inputs,
                )
                .unwrap();
            let output_axes = outputs.iter().map(|output| output.batch().batch_axis()).collect::<Vec<_>>();
            let output_atoms =
                outputs.iter().map(|output| output.batch().value().atom_id().unwrap()).collect::<Vec<_>>();
            drop(outputs);
            drop(tracer_inputs);
            drop(context);
            drop(parent);

            let builder = Rc::try_unwrap(builder).expect("batching should not retain the tracing builder").into_inner();
            let program = builder
                .build::<Vec<Array>, Vec<Array>>(
                    output_atoms,
                    vec![Placeholder, Placeholder],
                    vec![Placeholder, Placeholder, Placeholder],
                )
                .unwrap();
            let output_types = program.output_types();

            assert_eq!(output_axes, vec![BatchAxis::new(0), BatchAxis::new(1), BatchAxis::replicated()]);
            assert_eq!(output_types[0].shape().dimensions(), &[Dimension::Static(2)]);
            assert_eq!(output_types[0].sharding().unwrap().dimensions(), carry_sharding.dimensions());
            // The staged batched scan's stacked outputs carry the scan's *declared* output types, whose optional
            // sharding metadata is left unspecified for sharding propagation to resolve (the `scan_output_types`
            // contract); only the batch axes and shapes are pinned structurally.
            assert_eq!(output_types[1].shape().dimensions(), &[Dimension::Static(0), Dimension::Static(2)]);
            assert_eq!(output_types[1].sharding(), None);
            assert_eq!(output_types[2].shape().dimensions(), &[Dimension::Static(0)]);
            assert_eq!(output_types[2].sharding(), None);
        }
    }

    /// The `scan` differentiation rules reach their body through the per-[`Region`](crate::Region) transform cache,
    /// so several programs attaching one shared body derive its fused forward-mode program once and its transposition
    /// once per linearity mask, while staging exactly the programs the uncached path stages from independently built
    /// copies of the same body.
    #[test]
    fn test_scan_differentiation_reuses_the_shared_body_transforms() {
        /// Builds a program that scans the provided body over three slices and then applies `epilogue` sines to the
        /// final carry, so that programs sharing one body still have distinct derived programs.
        fn scanning_program(
            body: &Arc<Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>>,
            epilogue: usize,
        ) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let initial = builder.add_input(ArrayType::scalar(DataType::F64));
            let values = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])));
            let body_region = builder.intern_callee(body, None).unwrap();
            let mut value = builder
                .add_instruction(
                    ArrayOperation::Scan(ScanOperation::new(1, 3)),
                    vec![body_region],
                    vec![initial, values],
                )
                .unwrap()[0];
            for _ in 0..epilogue {
                value = builder.add_instruction(SinOperation::new(), Vec::new(), vec![value]).unwrap()[0];
            }
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![value], vec![Placeholder, Placeholder], vec![Placeholder])
                .unwrap()
        }

        let body = Arc::new(product_body());
        let first = scanning_program(&body, 1).linearize().unwrap();
        let second = scanning_program(&body, 2).linearize().unwrap();
        assert_ne!(first.tangent().to_string(), second.tangent().to_string());

        // The body's fused forward-mode program is derived by the first program and served to the second.
        let statistics = body.entry_region_ref().transform_statistics::<JvpTransform>().unwrap();
        assert_eq!((statistics.productions, statistics.hits), (1, 1));

        // An independently built copy of the same body shares no retained transforms, so it exercises the uncached
        // path and pins that caching changed nothing about what is staged.
        let uncached = scanning_program(&Arc::new(product_body()), 1).linearize().unwrap();
        assert_eq!(first.primal().to_string(), uncached.primal().to_string());
        assert_eq!(first.tangent().to_string(), uncached.tangent().to_string());
        assert_eq!(first.residual_count(), uncached.residual_count());

        // Transposing the tangent program twice transposes its scan body once: the second pass is served from the
        // body region's retained transposition and produces the identical pullback.
        let pullback = first.tangent().transpose_with_trailing_residuals(first.residual_count()).unwrap();
        let repeated = first.tangent().transpose_with_trailing_residuals(first.residual_count()).unwrap();
        assert_eq!(pullback.to_string(), repeated.to_string());
        let tangent_scan = first
            .tangent()
            .instructions()
            .iter()
            .find(|instruction| matches!(instruction.operation(), ArrayOperation::Scan(_)))
            .unwrap();
        let statistics = first
            .tangent()
            .region_ref(tangent_scan.regions()[0])
            .unwrap()
            .transform_statistics::<TranspositionTransform>()
            .unwrap();
        assert_eq!((statistics.productions, statistics.hits), (1, 1));
        assert_eq!(
            pullback.to_string(),
            uncached.tangent().transpose_with_trailing_residuals(uncached.residual_count()).unwrap().to_string(),
        );
    }

    /// Gate measurement for extending the per-[`Region`](crate::Region) transform cache to the `scan`
    /// differentiation rules. Several distinct outer programs attach *one shared* body region, which is exactly the
    /// sharing a region-keyed cache can serve, and each outer program is then linearized and transposed from cold.
    /// The printed table reports the frontend cost of each transform per outer program and how it scales with the
    /// body's instruction count, which is the input to deciding whether retaining the body's derived programs is
    /// worth its complexity.
    #[test]
    #[ignore = "region transform cache gate measurement"]
    fn test_baseline_repeated_scan_body_transformation() {
        /// Body instruction counts swept by the measurement.
        const BODY_OPERATION_COUNTS: [usize; 2] = [2, 200];

        /// Number of distinct outer programs that attach the one shared body region.
        const OUTER_SPECIALIZATIONS: usize = 4;

        let scalar_type = ArrayType::scalar(DataType::F64);
        let stacked_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let mut measurements = Vec::new();
        for body_operations in BODY_OPERATION_COUNTS {
            // One shared body that maps `[carry, slice]` through a product followed by a chain of sines.
            let body = {
                let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
                let carry = builder.add_input(scalar_type.clone());
                let slice = builder.add_input(scalar_type.clone());
                let mut value =
                    builder.add_instruction(MulOperation::new(), Vec::new(), vec![carry, slice]).unwrap()[0];
                for _ in 1..body_operations {
                    value = builder.add_instruction(SinOperation::new(), Vec::new(), vec![value]).unwrap()[0];
                }
                Arc::new(
                    builder
                        .build::<Vec<Array>, Vec<Array>>(
                            vec![value, value],
                            vec![Placeholder, Placeholder],
                            vec![Placeholder, Placeholder],
                        )
                        .unwrap(),
                )
            };

            // Each outer program interns that one body and differs only in the length of its sine epilogue, so their
            // derived programs are genuinely distinct while the scanned body region is shared.
            let outers = (0..OUTER_SPECIALIZATIONS)
                .map(|index| {
                    let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
                    let initial = builder.add_input(scalar_type.clone());
                    let values = builder.add_input(stacked_type.clone());
                    let body_region = builder.intern_callee(&body, None).unwrap();
                    let mut value = builder
                        .add_instruction(
                            ArrayOperation::Scan(ScanOperation::new(1, 3)),
                            vec![body_region],
                            vec![initial, values],
                        )
                        .unwrap()[0];
                    for _ in 0..=index {
                        value = builder.add_instruction(SinOperation::new(), Vec::new(), vec![value]).unwrap()[0];
                    }
                    builder
                        .build::<Vec<Array>, Vec<Array>>(vec![value], vec![Placeholder, Placeholder], vec![Placeholder])
                        .unwrap()
                })
                .collect::<Vec<_>>();

            let mut rows = Vec::with_capacity(OUTER_SPECIALIZATIONS);
            for outer in &outers {
                let start = Instant::now();
                let linearization = outer.linearize().unwrap();
                let linearized = start.elapsed();
                let start = Instant::now();
                linearization.tangent().transpose_with_trailing_residuals(linearization.residual_count()).unwrap();
                rows.push((linearized, start.elapsed()));
            }
            measurements.push((body_operations, rows));
        }

        println!("scan body transform gate: one shared body region, {OUTER_SPECIALIZATIONS} outer programs");
        for (body_operations, rows) in &measurements {
            println!("  body with {body_operations} operations (all times in milliseconds):");
            println!("    outer |    linearize |    transpose |        total");
            for (index, (linearized, transposed)) in rows.iter().enumerate() {
                println!(
                    "    {index:>5} | {:>12.3} | {:>12.3} | {:>12.3}",
                    linearized.as_secs_f64() * 1e3,
                    transposed.as_secs_f64() * 1e3,
                    (*linearized + *transposed).as_secs_f64() * 1e3,
                );
            }
        }

        // Repeated-outer cost is the mean over the outer programs after the first, which is what a retained body
        // transform could serve; the per-body-operation column reports how much of it is body-proportional.
        let repeated_mean = |rows: &[(Duration, Duration)]| {
            rows[1..]
                .iter()
                .map(|(linearized, transposed)| (*linearized + *transposed).as_secs_f64() * 1e3)
                .sum::<f64>()
                / (rows.len() - 1) as f64
        };
        let (small_operations, small_rows) = &measurements[0];
        let (large_operations, large_rows) = &measurements[1];
        let small_mean = repeated_mean(small_rows);
        let large_mean = repeated_mean(large_rows);
        println!(
            "  repeated-outer summary (mean over outers 1..{}, milliseconds): {small_operations}-op body \
             {small_mean:.3}, {large_operations}-op body {large_mean:.3}, per body operation {:.4}",
            OUTER_SPECIALIZATIONS,
            (large_mean - small_mean) / (large_operations - small_operations) as f64,
        );
    }
}
