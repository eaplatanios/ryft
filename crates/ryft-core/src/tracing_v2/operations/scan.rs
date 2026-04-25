use std::{
    borrow::Cow,
    fmt::{Debug, Display},
};

use thiserror::Error;

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily, Placeholder},
    tracing::{Program, Traceable, TracingError, Value},
    tracing_v2::{
        EngineTangent, JvpTracer, LinearPrimitiveOperation, LinearTerm, PrimitiveOperation, Tracer,
        engines::{DifferentiableEngine, Engine},
        forward::{Differentiable, TangentSpace},
        linear::{
            Linearized, linearize_program, replay_program_linearized_jit, trace_flat_program_from_input_types,
            transpose_linear_program,
        },
        operations::{
            CoreLinearProgramOperation, DifferentiableOperation, LinearAddOperation, LinearOperation,
            constants::ZeroLike,
        },
    },
    types::{ArrayType, Size, Type, TypeError, Typed},
};

use super::{InterpretableOperation, Operation};

/// Errors emitted while tracing, interpreting, or transforming `scan` operations.
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScanError {
    /// A scan with no `xs` leaves did not provide an explicit static length.
    #[error("scan requires an explicit length when xs has no leaves")]
    MissingLength,

    /// One scanned input leaf has no leading axis to scan over.
    #[error("scan input #{input_index} has rank 0 and cannot provide a leading scan axis")]
    MissingLeadingAxis { input_index: usize },

    /// One scanned input leaf has a dynamic leading axis.
    #[error("scan input #{input_index} has dynamic leading-axis size {size}")]
    DynamicLength { input_index: usize, size: Size },

    /// The explicit scan length disagrees with one scanned input leaf.
    #[error("scan input #{input_index} has leading-axis length {got}, but scan length is {expected}")]
    LengthMismatch { expected: usize, got: usize, input_index: usize },

    /// The scan body changed the carry metadata.
    #[error("scan body carry output #{index} has type {got}, but expected {expected}")]
    CarryTypeMismatch { index: usize, expected: ArrayType, got: ArrayType },

    /// The scan body returned the wrong number of carry leaves.
    #[error("scan body returned {got} carry leaf/leaves, but expected {expected}")]
    CarryCountMismatch { expected: usize, got: usize },

    /// Eager execution of a zero-length scan cannot infer non-empty output metadata.
    #[error("zero-length eager scan cannot infer stacked output metadata")]
    MissingEagerOutputMetadata,

    /// Eager scan interpretation requires value-level leading-axis support.
    #[error("scan value does not support {capability}")]
    UnsupportedValueCapability { capability: &'static str },

    /// The current transform path does not yet support scan.
    #[error("scan does not yet provide a {transform} rule")]
    MissingTransformRule { transform: &'static str },

    /// Traced scan staging needs at least one traced input leaf to supply the enclosing context.
    #[error("traced scan with non-empty outputs requires at least one traced input leaf")]
    MissingTracedInvocationContext,
}

/// Normalized scan unroll behavior.
///
/// `Rolled` corresponds to JAX's `unroll=False`, `Full` corresponds to `unroll=True` or
/// `unroll=0`, and `Count(n)` corresponds to a positive integer unroll chunk size.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ScanUnroll {
    /// Keep the loop rolled.
    Rolled,

    /// Fully unroll the statically known trip count.
    Full,

    /// Unroll a positive number of iterations per rolled loop iteration.
    Count(usize),
}

impl ScanUnroll {
    /// Returns the effective unroll factor for a scan of `length` iterations.
    #[inline]
    pub fn factor(self, length: usize) -> usize {
        match self {
            Self::Rolled => 1,
            Self::Full => length.max(1),
            Self::Count(0) => length.max(1),
            Self::Count(count) => count,
        }
    }
}

impl Default for ScanUnroll {
    #[inline]
    fn default() -> Self {
        Self::Count(1)
    }
}

impl From<bool> for ScanUnroll {
    #[inline]
    fn from(value: bool) -> Self {
        if value { Self::Full } else { Self::Rolled }
    }
}

impl From<usize> for ScanUnroll {
    #[inline]
    fn from(value: usize) -> Self {
        if value == 0 { Self::Full } else { Self::Count(value) }
    }
}

/// Options controlling the static trip count and traversal order of [`scan`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ScanOptions {
    /// Optional explicit scan length.
    pub length: Option<usize>,

    /// Whether to traverse the leading axis from the last element to the first element.
    pub reverse: bool,

    /// Loop unroll policy.
    pub unroll: ScanUnroll,

    /// Experimental reverse-mode transpose partitioning flag matching JAX's `_split_transpose`.
    pub split_transpose: bool,
}

impl ScanOptions {
    /// Returns default scan options.
    #[inline]
    pub const fn new() -> Self {
        Self { length: None, reverse: false, unroll: ScanUnroll::Count(1), split_transpose: false }
    }

    /// Returns options with an explicit static `length`.
    #[inline]
    pub const fn with_length(mut self, length: usize) -> Self {
        self.length = Some(length);
        self
    }

    /// Returns options with `reverse` traversal set.
    #[inline]
    pub const fn with_reverse(mut self, reverse: bool) -> Self {
        self.reverse = reverse;
        self
    }

    /// Returns options with a specific unroll policy.
    #[inline]
    pub const fn with_unroll(mut self, unroll: ScanUnroll) -> Self {
        self.unroll = unroll;
        self
    }

    /// Returns options with `_split_transpose` set.
    #[inline]
    pub const fn with_split_transpose(mut self, split_transpose: bool) -> Self {
        self.split_transpose = split_transpose;
        self
    }
}

impl Default for ScanOptions {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Value-level leading-axis behavior needed by eager scan interpretation.
pub trait ScanValue: Traceable<ArrayType> {
    /// Returns the `index`-th slice along the leading axis.
    fn scan_slice_leading_axis(&self, index: usize) -> Result<Self, TracingError>;

    /// Returns a metadata-only leading-axis slice exemplar for zero-length scans.
    fn scan_empty_slice_leading_axis(&self) -> Result<Self, TracingError> {
        Err(ScanError::UnsupportedValueCapability { capability: "zero-length leading-axis slice exemplar" }.into())
    }

    /// Stacks `values` along a new leading axis with metadata `output_type`.
    fn scan_stack_leading_axis(output_type: &ArrayType, values: Vec<Self>) -> Result<Self, TracingError>;

    /// Stacks `values` along a new leading axis, using `exemplar` when `values` is empty and the
    /// implementation needs an ambient tracing context.
    fn scan_stack_leading_axis_with_exemplar(
        output_type: &ArrayType,
        values: Vec<Self>,
        exemplar: Option<&Self>,
    ) -> Result<Self, TracingError> {
        Self::scan_stack_axis_with_exemplar(0, output_type, values, exemplar)
    }

    /// Returns the `index`-th slice along `axis`.
    fn scan_slice_axis(&self, axis: usize, index: usize) -> Result<Self, TracingError> {
        if axis == 0 {
            self.scan_slice_leading_axis(index)
        } else {
            Err(ScanError::UnsupportedValueCapability { capability: "non-leading-axis slicing" }.into())
        }
    }

    /// Returns a metadata-only slice exemplar along `axis` for zero-length scans.
    fn scan_empty_slice_axis(&self, axis: usize) -> Result<Self, TracingError> {
        if axis == 0 {
            self.scan_empty_slice_leading_axis()
        } else {
            Err(ScanError::UnsupportedValueCapability { capability: "zero-length non-leading-axis slice exemplar" }
                .into())
        }
    }

    /// Stacks `values` along `axis` with metadata `output_type`.
    fn scan_stack_axis(axis: usize, output_type: &ArrayType, values: Vec<Self>) -> Result<Self, TracingError> {
        if axis == 0 {
            Self::scan_stack_leading_axis(output_type, values)
        } else {
            Err(ScanError::UnsupportedValueCapability { capability: "non-leading-axis stacking" }.into())
        }
    }

    /// Stacks `values` along `axis`, using `exemplar` when `values` is empty and the implementation
    /// needs an ambient tracing context.
    fn scan_stack_axis_with_exemplar(
        axis: usize,
        output_type: &ArrayType,
        values: Vec<Self>,
        _exemplar: Option<&Self>,
    ) -> Result<Self, TracingError> {
        Self::scan_stack_axis(axis, output_type, values)
    }
}

macro_rules! impl_scalar_scan_value {
    ($($ty:ty),* $(,)?) => {
        $(
            impl ScanValue for $ty {
                fn scan_slice_leading_axis(&self, _index: usize) -> Result<Self, TracingError> {
                    Err(ScanError::UnsupportedValueCapability {
                        capability: "leading-axis slicing for scalar leaves",
                    }
                    .into())
                }

                fn scan_stack_leading_axis(_output_type: &ArrayType, _values: Vec<Self>) -> Result<Self, TracingError> {
                    Err(ScanError::UnsupportedValueCapability {
                        capability: "leading-axis stacking for scalar leaves",
                    }
                    .into())
                }
            }
        )*
    };
}

impl_scalar_scan_value!(bool, i8, i16, i32, i64, u8, u16, u32, u64, half::bf16, half::f16, f32, f64);

/// Carrier capability for leading-axis slice, scatter, and stack operations.
#[doc(hidden)]
pub trait LeadingAxisTracingOperation<V: Traceable<ArrayType>>: Clone + Operation<ArrayType> {
    /// Constructs a leading-axis slice operation.
    fn slice_leading_axis_op(op: SliceLeadingAxisOperation) -> Self;

    /// Constructs a leading-axis scatter operation.
    fn scatter_leading_axis_slice_op(op: ScatterLeadingAxisSliceOperation) -> Self;

    /// Constructs a leading-axis stack operation.
    fn stack_leading_axis_op(op: StackLeadingAxisOperation) -> Self;
}

/// Linear slice of one statically indexed leading-axis element.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SliceLeadingAxisOperation {
    /// Abstract type of the stacked input.
    input_type: ArrayType,

    /// Abstract type of the sliced output.
    output_type: ArrayType,

    /// Static leading-axis index.
    index: usize,

    /// Static axis being sliced.
    axis: usize,
}

impl SliceLeadingAxisOperation {
    /// Creates a leading-axis slice operation.
    pub fn new(input_type: ArrayType, index: usize) -> Result<Self, TracingError> {
        Self::new_at_axis(input_type, 0, index)
    }

    /// Creates a static-axis slice operation.
    pub fn new_at_axis(input_type: ArrayType, axis: usize, index: usize) -> Result<Self, TracingError> {
        let (output_type, length) = input_type.without_dimension(axis)?;
        if let Some(length) = length.value()
            && index >= length
        {
            return Err(TypeError {
                message: format!("slice leading-axis index {index} is out of bounds for length {length}"),
            }
            .into());
        }
        Ok(Self { input_type, output_type, index, axis })
    }

    /// Returns the stacked input type.
    #[inline]
    pub fn input_type(&self) -> &ArrayType {
        &self.input_type
    }

    /// Returns the sliced output type.
    #[inline]
    pub fn output_type(&self) -> &ArrayType {
        &self.output_type
    }

    /// Returns the static leading-axis index.
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns the static axis being sliced.
    #[inline]
    pub fn axis(&self) -> usize {
        self.axis
    }
}

impl Display for SliceLeadingAxisOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "slice")
    }
}

impl Operation<ArrayType> for SliceLeadingAxisOperation {
    fn name(&self) -> &'static str {
        "slice"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        if input_types.len() != 1 {
            return Err(TypeError { message: format!("slice expected 1 input type but got {}", input_types.len()) });
        }
        if input_types[0] != self.input_type {
            return Err(TypeError { message: "slice input type does not match the captured type".to_string() });
        }
        Ok(vec![self.output_type.clone()])
    }
}

impl<V: ScanValue> InterpretableOperation<ArrayType, V> for SliceLeadingAxisOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        if inputs.len() != 1 {
            return Err(TracingError::InvalidInputCount { expected: 1, got: inputs.len() });
        }
        Ok(vec![inputs[0].scan_slice_axis(self.axis, self.index)?])
    }
}

impl<E> DifferentiableOperation<E> for SliceLeadingAxisOperation
where
    E: DifferentiableEngine<Type = ArrayType> + ?Sized,
    E::Value: ScanValue + Differentiable<ArrayType>,
    EngineTangent<E>: ScanValue,
{
    fn jvp(
        &self,
        _engine: &E,
        inputs: &[JvpTracer<E::Value, EngineTangent<E>>],
    ) -> Result<Vec<JvpTracer<E::Value, EngineTangent<E>>>, TracingError> {
        if inputs.len() != 1 {
            return Err(TracingError::InvalidInputCount { expected: 1, got: inputs.len() });
        }
        Ok(vec![JvpTracer {
            primal: inputs[0].primal.scan_slice_axis(self.axis, self.index)?,
            tangent: inputs[0].tangent.scan_slice_axis(self.axis, self.index)?,
        }])
    }
}

impl<V: ScanValue + ZeroLike, LinearCarrier: LeadingAxisTracingOperation<V>>
    LinearOperation<ArrayType, V, LinearCarrier> for SliceLeadingAxisOperation
{
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V, LinearCarrier>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V, LinearCarrier>>>, TracingError>
    where
        LinearCarrier: Operation<ArrayType>,
    {
        if output_cotangents.len() != 1 {
            return Err(TracingError::InvalidInputCount { expected: 1, got: output_cotangents.len() });
        }
        let scatter = ScatterLeadingAxisSliceOperation::new_at_axis(
            self.output_type.clone(),
            self.input_type.clone(),
            self.axis,
            self.index,
        )?;
        let contribution = LinearTerm::apply_staged_op(
            output_cotangents[0].builder.clone(),
            output_cotangents,
            LinearCarrier::scatter_leading_axis_slice_op(scatter),
            1,
        )?
        .into_iter()
        .next()
        .expect("scatter should produce one cotangent contribution");
        Ok(vec![Some(contribution)])
    }
}

/// Linear scatter of one leading-axis slice into an otherwise-zero stacked value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScatterLeadingAxisSliceOperation {
    /// Abstract type of the slice input.
    input_type: ArrayType,

    /// Abstract type of the stacked output.
    output_type: ArrayType,

    /// Static leading-axis index to populate.
    index: usize,

    /// Static axis being scattered into.
    axis: usize,
}

impl ScatterLeadingAxisSliceOperation {
    /// Creates a leading-axis slice scatter operation.
    pub fn new(input_type: ArrayType, output_type: ArrayType, index: usize) -> Result<Self, TracingError> {
        Self::new_at_axis(input_type, output_type, 0, index)
    }

    /// Creates a static-axis slice scatter operation.
    pub fn new_at_axis(
        input_type: ArrayType,
        output_type: ArrayType,
        axis: usize,
        index: usize,
    ) -> Result<Self, TracingError> {
        let (expected_input_type, length) = output_type.without_dimension(axis)?;
        if expected_input_type != input_type {
            return Err(TypeError {
                message: "scatter slice input type does not match the output element type".to_string(),
            }
            .into());
        }
        let Some(length) = length.value() else {
            return Err(ScanError::DynamicLength { input_index: 0, size: length }.into());
        };
        if index >= length {
            return Err(TypeError {
                message: format!("scatter leading-axis index {index} is out of bounds for length {length}"),
            }
            .into());
        }
        Ok(Self { input_type, output_type, index, axis })
    }

    /// Returns the slice input type.
    #[inline]
    pub fn input_type(&self) -> &ArrayType {
        &self.input_type
    }

    /// Returns the stacked output type.
    #[inline]
    pub fn output_type(&self) -> &ArrayType {
        &self.output_type
    }

    /// Returns the static leading-axis index.
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns the static axis being scattered into.
    #[inline]
    pub fn axis(&self) -> usize {
        self.axis
    }
}

impl Display for ScatterLeadingAxisSliceOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "scatter")
    }
}

impl Operation<ArrayType> for ScatterLeadingAxisSliceOperation {
    fn name(&self) -> &'static str {
        "scatter"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        if input_types.len() != 1 {
            return Err(TypeError { message: format!("scatter expected 1 input type but got {}", input_types.len()) });
        }
        if input_types[0] != self.input_type {
            return Err(TypeError { message: "scatter input type does not match the captured type".to_string() });
        }
        Ok(vec![self.output_type.clone()])
    }
}

impl<V: ScanValue + ZeroLike> InterpretableOperation<ArrayType, V> for ScatterLeadingAxisSliceOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        if inputs.len() != 1 {
            return Err(TracingError::InvalidInputCount { expected: 1, got: inputs.len() });
        }
        let axis = self.axis as i32;
        let Some(length) = self.output_type.dimension(axis).value() else {
            return Err(ScanError::DynamicLength { input_index: 0, size: self.output_type.dimension(axis) }.into());
        };
        let mut values = vec![inputs[0].zero_like(); length];
        values[self.index] = inputs[0].clone();
        Ok(vec![V::scan_stack_axis(self.axis, &self.output_type, values)?])
    }
}

impl<E> DifferentiableOperation<E> for ScatterLeadingAxisSliceOperation
where
    E: DifferentiableEngine<Type = ArrayType> + ?Sized,
    E::Value: ScanValue + ZeroLike + Differentiable<ArrayType>,
    EngineTangent<E>: ScanValue + ZeroLike,
{
    fn jvp(
        &self,
        _engine: &E,
        inputs: &[JvpTracer<E::Value, EngineTangent<E>>],
    ) -> Result<Vec<JvpTracer<E::Value, EngineTangent<E>>>, TracingError> {
        if inputs.len() != 1 {
            return Err(TracingError::InvalidInputCount { expected: 1, got: inputs.len() });
        }
        let primal = <Self as InterpretableOperation<ArrayType, E::Value>>::interpret(
            self,
            std::slice::from_ref(&inputs[0].primal),
        )?
        .remove(0);
        let tangent = <Self as InterpretableOperation<ArrayType, EngineTangent<E>>>::interpret(
            self,
            std::slice::from_ref(&inputs[0].tangent),
        )?
        .remove(0);
        Ok(vec![JvpTracer { primal, tangent }])
    }
}

impl<V: ScanValue + ZeroLike, LinearCarrier: LeadingAxisTracingOperation<V>>
    LinearOperation<ArrayType, V, LinearCarrier> for ScatterLeadingAxisSliceOperation
{
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V, LinearCarrier>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V, LinearCarrier>>>, TracingError>
    where
        LinearCarrier: Operation<ArrayType>,
    {
        if output_cotangents.len() != 1 {
            return Err(TracingError::InvalidInputCount { expected: 1, got: output_cotangents.len() });
        }
        let slice = SliceLeadingAxisOperation::new_at_axis(self.output_type.clone(), self.axis, self.index)?;
        let contribution = LinearTerm::apply_staged_op(
            output_cotangents[0].builder.clone(),
            output_cotangents,
            LinearCarrier::slice_leading_axis_op(slice),
            1,
        )?
        .into_iter()
        .next()
        .expect("slice should produce one cotangent contribution");
        Ok(vec![Some(contribution)])
    }
}

/// Linear stack of same-typed leaves along a new leading axis.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StackLeadingAxisOperation {
    /// Abstract type of every input slice.
    input_type: ArrayType,

    /// Abstract type of the stacked output.
    output_type: ArrayType,

    /// Static number of input slices.
    length: usize,

    /// Static axis being stacked.
    axis: usize,
}

impl StackLeadingAxisOperation {
    /// Creates a leading-axis stack operation.
    pub fn new(output_type: ArrayType) -> Result<Self, TracingError> {
        Self::new_at_axis(output_type, 0)
    }

    /// Creates a static-axis stack operation.
    pub fn new_at_axis(output_type: ArrayType, axis: usize) -> Result<Self, TracingError> {
        let (input_type, length) = output_type.without_dimension(axis)?;
        let Some(length) = length.value() else {
            return Err(ScanError::DynamicLength { input_index: 0, size: length }.into());
        };
        Ok(Self { input_type, output_type, length, axis })
    }

    /// Returns the slice input type.
    #[inline]
    pub fn input_type(&self) -> &ArrayType {
        &self.input_type
    }

    /// Returns the stacked output type.
    #[inline]
    pub fn output_type(&self) -> &ArrayType {
        &self.output_type
    }

    /// Returns the static stack length.
    #[inline]
    pub fn length(&self) -> usize {
        self.length
    }

    /// Returns the static axis being stacked.
    #[inline]
    pub fn axis(&self) -> usize {
        self.axis
    }
}

impl Display for StackLeadingAxisOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "stack")
    }
}

impl Operation<ArrayType> for StackLeadingAxisOperation {
    fn name(&self) -> &'static str {
        "stack"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        if input_types.len() != self.length {
            return Err(TypeError {
                message: format!("stack expected {} input type(s) but got {}", self.length, input_types.len()),
            });
        }
        if input_types.iter().any(|input_type| input_type != &self.input_type) {
            return Err(TypeError { message: "stack input types do not match the captured type".to_string() });
        }
        Ok(vec![self.output_type.clone()])
    }
}

impl<V: ScanValue> InterpretableOperation<ArrayType, V> for StackLeadingAxisOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        if inputs.len() != self.length {
            return Err(TracingError::InvalidInputCount { expected: self.length, got: inputs.len() });
        }
        Ok(vec![V::scan_stack_axis(self.axis, &self.output_type, inputs.to_vec())?])
    }
}

impl<E> DifferentiableOperation<E> for StackLeadingAxisOperation
where
    E: DifferentiableEngine<Type = ArrayType> + ?Sized,
    E::Value: ScanValue + Differentiable<ArrayType>,
    EngineTangent<E>: ScanValue,
{
    fn jvp(
        &self,
        _engine: &E,
        inputs: &[JvpTracer<E::Value, EngineTangent<E>>],
    ) -> Result<Vec<JvpTracer<E::Value, EngineTangent<E>>>, TracingError> {
        if inputs.len() != self.length {
            return Err(TracingError::InvalidInputCount { expected: self.length, got: inputs.len() });
        }
        let primals = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let tangents = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
        Ok(vec![JvpTracer {
            primal: E::Value::scan_stack_axis(self.axis, &self.output_type, primals)?,
            tangent: EngineTangent::<E>::scan_stack_axis(self.axis, &self.output_type, tangents)?,
        }])
    }
}

impl<V: ScanValue + ZeroLike, LinearCarrier: LeadingAxisTracingOperation<V>>
    LinearOperation<ArrayType, V, LinearCarrier> for StackLeadingAxisOperation
{
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V, LinearCarrier>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V, LinearCarrier>>>, TracingError>
    where
        LinearCarrier: Operation<ArrayType>,
    {
        if output_cotangents.len() != 1 {
            return Err(TracingError::InvalidInputCount { expected: 1, got: output_cotangents.len() });
        }
        let builder = output_cotangents[0].builder.clone();
        (0..self.length)
            .map(|index| {
                let slice = SliceLeadingAxisOperation::new_at_axis(self.output_type.clone(), self.axis, index)?;
                Ok(Some(
                    LinearTerm::apply_staged_op(
                        builder.clone(),
                        output_cotangents,
                        LinearCarrier::slice_leading_axis_op(slice),
                        1,
                    )?
                    .into_iter()
                    .next()
                    .expect("slice should produce one cotangent contribution"),
                ))
            })
            .collect()
    }
}

impl<'engine, E: Engine<Type = ArrayType> + ?Sized, O: LeadingAxisTracingOperation<E::Value>> ScanValue
    for Tracer<'engine, E, O>
where
    E::Value: Traceable<ArrayType>,
{
    fn scan_slice_leading_axis(&self, index: usize) -> Result<Self, TracingError> {
        self.scan_slice_axis(0, index)
    }

    fn scan_slice_axis(&self, axis: usize, index: usize) -> Result<Self, TracingError> {
        let op = SliceLeadingAxisOperation::new_at_axis(self.r#type().into_owned(), axis, index)?;
        Ok(Tracer::apply_staged_op(
            self.engine,
            self.builder.clone(),
            std::slice::from_ref(self),
            O::slice_leading_axis_op(op),
        )?
        .into_iter()
        .next()
        .expect("slice should produce one traced output"))
    }

    fn scan_stack_leading_axis(output_type: &ArrayType, values: Vec<Self>) -> Result<Self, TracingError> {
        Self::scan_stack_axis(0, output_type, values)
    }

    fn scan_stack_axis(axis: usize, output_type: &ArrayType, values: Vec<Self>) -> Result<Self, TracingError> {
        let Some(first_value) = values.first() else {
            return Err(ScanError::MissingTracedInvocationContext.into());
        };
        let op = StackLeadingAxisOperation::new_at_axis(output_type.clone(), axis)?;
        Ok(Tracer::apply_staged_op(
            first_value.engine,
            first_value.builder.clone(),
            values.as_slice(),
            O::stack_leading_axis_op(op),
        )?
        .into_iter()
        .next()
        .expect("stack should produce one traced output"))
    }

    fn scan_stack_axis_with_exemplar(
        axis: usize,
        output_type: &ArrayType,
        values: Vec<Self>,
        exemplar: Option<&Self>,
    ) -> Result<Self, TracingError> {
        let (engine, builder) = match values.first().or(exemplar) {
            Some(value) => (value.engine, value.builder.clone()),
            None => return Err(ScanError::MissingTracedInvocationContext.into()),
        };
        let op = StackLeadingAxisOperation::new_at_axis(output_type.clone(), axis)?;
        Ok(Tracer::apply_staged_op(engine, builder, values.as_slice(), O::stack_leading_axis_op(op))?
            .into_iter()
            .next()
            .expect("stack should produce one traced output"))
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone + Operation<T>> Typed<T> for LinearTerm<T, V, O> {
    fn r#type(&self) -> Cow<'_, T> {
        Cow::Owned(self.builder.borrow().atoms[self.atom.index].r#type().into_owned())
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone + Operation<T>> Traceable<T> for LinearTerm<T, V, O> {}

impl<V: Traceable<ArrayType>, O: LeadingAxisTracingOperation<V>> ScanValue for LinearTerm<ArrayType, V, O> {
    fn scan_slice_leading_axis(&self, index: usize) -> Result<Self, TracingError> {
        self.scan_slice_axis(0, index)
    }

    fn scan_slice_axis(&self, axis: usize, index: usize) -> Result<Self, TracingError> {
        let op = SliceLeadingAxisOperation::new_at_axis(self.r#type().into_owned(), axis, index)?;
        Ok(LinearTerm::apply_staged_op(
            self.builder.clone(),
            std::slice::from_ref(self),
            O::slice_leading_axis_op(op),
            1,
        )?
        .into_iter()
        .next()
        .expect("slice should produce one linear output"))
    }

    fn scan_stack_leading_axis(output_type: &ArrayType, values: Vec<Self>) -> Result<Self, TracingError> {
        Self::scan_stack_axis(0, output_type, values)
    }

    fn scan_stack_axis(axis: usize, output_type: &ArrayType, values: Vec<Self>) -> Result<Self, TracingError> {
        let Some(first_value) = values.first() else {
            return Err(ScanError::MissingTracedInvocationContext.into());
        };
        let op = StackLeadingAxisOperation::new_at_axis(output_type.clone(), axis)?;
        Ok(LinearTerm::apply_staged_op(
            first_value.builder.clone(),
            values.as_slice(),
            O::stack_leading_axis_op(op),
            1,
        )?
        .into_iter()
        .next()
        .expect("stack should produce one linear output"))
    }

    fn scan_stack_axis_with_exemplar(
        axis: usize,
        output_type: &ArrayType,
        values: Vec<Self>,
        exemplar: Option<&Self>,
    ) -> Result<Self, TracingError> {
        let builder = match values.first().or(exemplar) {
            Some(value) => value.builder.clone(),
            None => return Err(ScanError::MissingTracedInvocationContext.into()),
        };
        let op = StackLeadingAxisOperation::new_at_axis(output_type.clone(), axis)?;
        Ok(LinearTerm::apply_staged_op(builder, values.as_slice(), O::stack_leading_axis_op(op), 1)?
            .into_iter()
            .next()
            .expect("stack should produce one linear output"))
    }
}

impl<V: ScanValue, T: ScanValue + TangentSpace<ArrayType, V>> ScanValue for JvpTracer<V, T> {
    fn scan_slice_leading_axis(&self, index: usize) -> Result<Self, TracingError> {
        self.scan_slice_axis(0, index)
    }

    fn scan_empty_slice_leading_axis(&self) -> Result<Self, TracingError> {
        self.scan_empty_slice_axis(0)
    }

    fn scan_slice_axis(&self, axis: usize, index: usize) -> Result<Self, TracingError> {
        Ok(Self {
            primal: self.primal.scan_slice_axis(axis, index)?,
            tangent: self.tangent.scan_slice_axis(axis, index)?,
        })
    }

    fn scan_empty_slice_axis(&self, axis: usize) -> Result<Self, TracingError> {
        Ok(Self {
            primal: self.primal.scan_empty_slice_axis(axis)?,
            tangent: self.tangent.scan_empty_slice_axis(axis)?,
        })
    }

    fn scan_stack_leading_axis(output_type: &ArrayType, values: Vec<Self>) -> Result<Self, TracingError> {
        Self::scan_stack_axis(0, output_type, values)
    }

    fn scan_stack_axis(axis: usize, output_type: &ArrayType, values: Vec<Self>) -> Result<Self, TracingError> {
        let mut primals = Vec::with_capacity(values.len());
        let mut tangents = Vec::with_capacity(values.len());
        for value in values {
            primals.push(value.primal);
            tangents.push(value.tangent);
        }
        Ok(Self {
            primal: V::scan_stack_axis(axis, output_type, primals)?,
            tangent: T::scan_stack_axis(axis, output_type, tangents)?,
        })
    }

    fn scan_stack_axis_with_exemplar(
        axis: usize,
        output_type: &ArrayType,
        values: Vec<Self>,
        exemplar: Option<&Self>,
    ) -> Result<Self, TracingError> {
        let mut primals = Vec::with_capacity(values.len());
        let mut tangents = Vec::with_capacity(values.len());
        for value in values {
            primals.push(value.primal);
            tangents.push(value.tangent);
        }
        Ok(Self {
            primal: V::scan_stack_axis_with_exemplar(axis, output_type, primals, exemplar.map(|value| &value.primal))?,
            tangent: T::scan_stack_axis_with_exemplar(
                axis,
                output_type,
                tangents,
                exemplar.map(|value| &value.tangent),
            )?,
        })
    }
}

/// Hidden staging trait for the `scan` higher-order primitive.
#[doc(hidden)]
pub trait ScanTracingOperation<T: Type + Display, V: Traceable<T>>: Clone + Operation<T> {
    /// Constructs the carrier-specific representation of a captured [`ScanOperation`].
    fn scan_op(op: ScanOperation<T, V, Self>) -> Self;
}

/// Erased traced body for one scan operation.
#[derive(Clone)]
pub struct FlatTracedScan<T: Type, V: Traceable<T>, O: Clone + Operation<T> = PrimitiveOperation<V>> {
    /// Canonical carry leaf types.
    carry_types: Vec<T>,

    /// Canonical per-step `x` leaf types.
    x_types: Vec<T>,

    /// Canonical per-step `y` leaf types.
    y_types: Vec<T>,

    /// Canonical scanned `xs` input leaf types, including the leading scan axis.
    xs_types: Vec<T>,

    /// Canonical stacked `ys` output leaf types, including the leading scan axis.
    ys_types: Vec<T>,

    /// Static trip count.
    length: usize,

    /// Flat body sub-program over `carry` leaves followed by one `x` slice.
    program: Program<T, V, O, Vec<V>, Vec<V>>,
}

impl<T: Type, V: Traceable<T>, O: Clone + Operation<T>> FlatTracedScan<T, V, O> {
    /// Builds one erased traced scan body from explicit staged parts.
    #[inline]
    pub fn from_parts(
        carry_types: Vec<T>,
        x_types: Vec<T>,
        y_types: Vec<T>,
        xs_types: Vec<T>,
        ys_types: Vec<T>,
        length: usize,
        program: Program<T, V, O, Vec<V>, Vec<V>>,
    ) -> Self {
        Self { carry_types, x_types, y_types, xs_types, ys_types, length, program }
    }

    /// Returns the flat carry leaf types.
    #[inline]
    pub fn carry_types(&self) -> &[T] {
        self.carry_types.as_slice()
    }

    /// Returns the flat per-step `x` leaf types.
    #[inline]
    pub fn x_types(&self) -> &[T] {
        self.x_types.as_slice()
    }

    /// Returns the flat per-step `y` leaf types.
    #[inline]
    pub fn y_types(&self) -> &[T] {
        self.y_types.as_slice()
    }

    /// Returns the flat scanned `xs` input leaf types.
    #[inline]
    pub fn xs_types(&self) -> &[T] {
        self.xs_types.as_slice()
    }

    /// Returns the flat stacked `ys` output leaf types.
    #[inline]
    pub fn ys_types(&self) -> &[T] {
        self.ys_types.as_slice()
    }

    /// Returns the static trip count.
    #[inline]
    pub fn length(&self) -> usize {
        self.length
    }

    /// Returns the flat scan body sub-program.
    #[inline]
    pub fn program(&self) -> &Program<T, V, O, Vec<V>, Vec<V>> {
        &self.program
    }
}

/// Higher-order scan operation carrying one traced loop body.
#[derive(Clone)]
pub struct ScanOperation<
    T: Type + Display,
    V: Traceable<T> + Parameter,
    O: Clone + Operation<T> = PrimitiveOperation<V>,
> {
    /// The captured flat body.
    body: FlatTracedScan<T, V, O>,

    /// Whether the leading axis is traversed in reverse.
    reverse: bool,

    /// Loop unroll policy.
    unroll: ScanUnroll,

    /// Experimental transpose partitioning flag.
    split_transpose: bool,
}

impl<T: Type + Display, V: Traceable<T>, O: Clone + Operation<T>> ScanOperation<T, V, O> {
    /// Builds one scan operation wrapping `body` and normalized options.
    #[inline]
    pub fn new(body: FlatTracedScan<T, V, O>, options: ScanOptions) -> Self {
        Self { body, reverse: options.reverse, unroll: options.unroll, split_transpose: options.split_transpose }
    }

    /// Returns the captured scan body.
    #[inline]
    pub fn body(&self) -> &FlatTracedScan<T, V, O> {
        &self.body
    }

    /// Returns whether this scan traverses the leading axis in reverse.
    #[inline]
    pub fn reverse(&self) -> bool {
        self.reverse
    }

    /// Returns the normalized unroll policy.
    #[inline]
    pub fn unroll(&self) -> ScanUnroll {
        self.unroll
    }

    /// Returns the `_split_transpose` flag.
    #[inline]
    pub fn split_transpose(&self) -> bool {
        self.split_transpose
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone + Operation<T>> Debug for ScanOperation<T, V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Scan")
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone + Operation<T>> Display for ScanOperation<T, V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "scan")
    }
}

type LinearScanBodyProgram<V> = Program<ArrayType, V, LinearPrimitiveOperation<V>, Vec<V>, Vec<V>>;

/// Compact linear pushforward for one scan operation.
#[derive(Clone)]
pub struct LinearizedScanJvpOperation<V: Traceable<ArrayType> + Parameter> {
    /// Captured scan metadata and body signature.
    scan: ScanOperation<ArrayType, V, PrimitiveOperation<V>>,

    /// Per-step linearized body pushforwards in forward scan execution order.
    pushforwards: Vec<LinearScanBodyProgram<V>>,

    /// Per-step body pullbacks matching [`LinearizedScanJvpOperation::pushforwards`].
    pullbacks: Vec<LinearScanBodyProgram<V>>,
}

/// Compact linear transpose for one scan pushforward.
#[derive(Clone)]
pub struct LinearizedScanTransposeOperation<V: Traceable<ArrayType> + Parameter> {
    /// Pushforward operation whose precomputed body pullbacks define this transpose.
    jvp: LinearizedScanJvpOperation<V>,
}

impl<V: Traceable<ArrayType>> LinearizedScanJvpOperation<V> {
    /// Builds one compact linearized scan operation by linearizing each body step once at the
    /// concrete primal point.
    pub(crate) fn from_scan<E>(
        engine: &E,
        scan: ScanOperation<ArrayType, V, PrimitiveOperation<V>>,
        primal_inputs: Vec<V>,
    ) -> Result<Self, TracingError>
    where
        V: ScanValue + ZeroLike,
        PrimitiveOperation<V>: DifferentiableOperation<E> + InterpretableOperation<ArrayType, V>,
        LinearPrimitiveOperation<V>: CoreLinearProgramOperation<V> + LinearAddOperation<ArrayType, V>,
        Vec<V>: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
        E: DifferentiableEngine<
                Type = ArrayType,
                Value = V,
                DifferentiableOperation = PrimitiveOperation<V>,
                LinearOperation = LinearPrimitiveOperation<V>,
            > + ?Sized,
    {
        let abstract_inputs = primal_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let _ = scan.infer_output_types(abstract_inputs.as_slice())?;
        let carry_count = scan.body.carry_types.len();
        let x_count = scan.body.x_types.len();
        let mut carry_primals = primal_inputs[..carry_count].to_vec();
        let xs_primals = primal_inputs[carry_count..].to_vec();
        let mut pushforwards = Vec::with_capacity(scan.body.length);
        let mut pullbacks = Vec::with_capacity(scan.body.length);

        for step in 0..scan.body.length {
            let scan_index = if scan.reverse { scan.body.length - 1 - step } else { step };
            let x_primals = xs_primals
                .iter()
                .map(|input| input.scan_slice_leading_axis(scan_index))
                .collect::<Result<Vec<_>, _>>()?;
            let mut body_primals = Vec::with_capacity(carry_count + x_count);
            body_primals.extend(carry_primals);
            body_primals.extend(x_primals);
            let pushforward = linearize_program(engine, scan.body.program(), body_primals.clone())?;
            let pullback = transpose_linear_program(engine, &pushforward)?;
            let body_outputs = scan.body.program().interpret(body_primals)?;
            carry_primals = body_outputs[..carry_count].to_vec();
            pushforwards.push(pushforward);
            pullbacks.push(pullback);
        }

        Ok(Self { scan, pushforwards, pullbacks })
    }

    fn input_types(&self) -> Vec<ArrayType> {
        self.scan.body.carry_types.iter().chain(self.scan.body.xs_types.iter()).cloned().collect()
    }

    fn output_types(&self) -> Vec<ArrayType> {
        self.scan.body.carry_types.iter().chain(self.scan.body.ys_types.iter()).cloned().collect()
    }

    /// Returns the captured primal scan metadata.
    #[inline]
    pub(crate) fn scan(&self) -> &ScanOperation<ArrayType, V, PrimitiveOperation<V>> {
        &self.scan
    }

    /// Returns the per-step body pushforward programs in forward execution order.
    #[inline]
    pub(crate) fn pushforwards(&self) -> &[Program<ArrayType, V, LinearPrimitiveOperation<V>, Vec<V>, Vec<V>>] {
        self.pushforwards.as_slice()
    }

    /// Returns the per-step body pullback programs in forward execution order.
    #[inline]
    pub(crate) fn pullbacks(&self) -> &[Program<ArrayType, V, LinearPrimitiveOperation<V>, Vec<V>, Vec<V>>] {
        self.pullbacks.as_slice()
    }
}

impl<V: Traceable<ArrayType>> LinearizedScanTransposeOperation<V> {
    /// Builds the compact transpose of `jvp`.
    #[inline]
    pub fn new(jvp: LinearizedScanJvpOperation<V>) -> Self {
        Self { jvp }
    }

    fn input_types(&self) -> Vec<ArrayType> {
        self.jvp.output_types()
    }

    fn output_types(&self) -> Vec<ArrayType> {
        self.jvp.input_types()
    }

    /// Returns the captured compact pushforward whose transpose this operation represents.
    #[inline]
    pub(crate) fn jvp(&self) -> &LinearizedScanJvpOperation<V> {
        &self.jvp
    }
}

impl<V: Traceable<ArrayType>> Debug for LinearizedScanJvpOperation<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("LinearizedScanJvp")
    }
}

impl<V: Traceable<ArrayType>> Debug for LinearizedScanTransposeOperation<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("LinearizedScanTranspose")
    }
}

impl<V: Traceable<ArrayType>> Display for LinearizedScanJvpOperation<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("linear_scan_jvp")
    }
}

impl<V: Traceable<ArrayType>> Display for LinearizedScanTransposeOperation<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("linear_scan_transpose")
    }
}

impl<V: Traceable<ArrayType>> Operation<ArrayType> for LinearizedScanJvpOperation<V> {
    fn name(&self) -> &'static str {
        "linear_scan_jvp"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        let expected_input_types = self.input_types();
        if input_types != expected_input_types.as_slice() {
            return Err(TypeError {
                message: "linear scan JVP input types do not match the captured scan signature".to_string(),
            });
        }
        Ok(self.output_types())
    }
}

impl<V: Traceable<ArrayType>> Operation<ArrayType> for LinearizedScanTransposeOperation<V> {
    fn name(&self) -> &'static str {
        "linear_scan_transpose"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        let expected_input_types = self.input_types();
        if input_types != expected_input_types.as_slice() {
            return Err(TypeError {
                message: "linear scan transpose input types do not match the captured scan signature".to_string(),
            });
        }
        Ok(self.output_types())
    }
}

impl<V: Traceable<ArrayType> + ScanValue> InterpretableOperation<ArrayType, V> for LinearizedScanJvpOperation<V>
where
    LinearPrimitiveOperation<V>: InterpretableOperation<ArrayType, V>,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        let _ = self.infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>())?;
        let carry_count = self.scan.body.carry_types.len();
        let x_count = self.scan.body.x_types.len();
        let y_count = self.scan.body.y_types.len();
        let mut carry_tangents = inputs[..carry_count].to_vec();
        let xs_tangents = inputs[carry_count..].to_vec();
        let mut stacked_y_tangents = vec![vec![None; self.scan.body.length]; y_count];

        for step in 0..self.scan.body.length {
            let scan_index = if self.scan.reverse { self.scan.body.length - 1 - step } else { step };
            let x_tangents = xs_tangents
                .iter()
                .map(|input| input.scan_slice_leading_axis(scan_index))
                .collect::<Result<Vec<_>, _>>()?;
            let mut body_tangents = Vec::with_capacity(carry_count + x_count);
            body_tangents.extend(carry_tangents);
            body_tangents.extend(x_tangents);
            let body_outputs = self.pushforwards[step].interpret(body_tangents)?;
            carry_tangents = body_outputs[..carry_count].to_vec();
            for (output_index, output) in body_outputs[carry_count..].iter().cloned().enumerate() {
                stacked_y_tangents[output_index][scan_index] = Some(output);
            }
        }

        let mut outputs = carry_tangents;
        for (output_type, values) in self.scan.body.ys_types.iter().zip(stacked_y_tangents) {
            let values = values.into_iter().collect::<Option<Vec<_>>>().ok_or(ScanError::MissingEagerOutputMetadata)?;
            outputs.push(V::scan_stack_leading_axis_with_exemplar(output_type, values, inputs.first())?);
        }
        Ok(outputs)
    }
}

impl<V: Traceable<ArrayType> + ScanValue> InterpretableOperation<ArrayType, V> for LinearizedScanTransposeOperation<V>
where
    LinearPrimitiveOperation<V>: InterpretableOperation<ArrayType, V>,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        let _ = self.infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>())?;
        let carry_count = self.jvp.scan.body.carry_types.len();
        let x_count = self.jvp.scan.body.x_types.len();
        let y_count = self.jvp.scan.body.y_types.len();
        let mut carry_cotangents = inputs[..carry_count].to_vec();
        let ys_cotangents = inputs[carry_count..].to_vec();
        let mut stacked_x_cotangents = vec![vec![None; self.jvp.scan.body.length]; x_count];

        for step in (0..self.jvp.scan.body.length).rev() {
            let scan_index = if self.jvp.scan.reverse { self.jvp.scan.body.length - 1 - step } else { step };
            let y_cotangents = ys_cotangents
                .iter()
                .map(|input| input.scan_slice_leading_axis(scan_index))
                .collect::<Result<Vec<_>, _>>()?;
            let mut body_output_cotangents = Vec::with_capacity(carry_count + y_count);
            body_output_cotangents.extend(carry_cotangents);
            body_output_cotangents.extend(y_cotangents);
            let body_input_cotangents = self.jvp.pullbacks[step].interpret(body_output_cotangents)?;
            carry_cotangents = body_input_cotangents[..carry_count].to_vec();
            for (input_index, cotangent) in body_input_cotangents[carry_count..].iter().cloned().enumerate() {
                stacked_x_cotangents[input_index][scan_index] = Some(cotangent);
            }
        }

        let mut outputs = carry_cotangents;
        for (output_type, values) in self.jvp.scan.body.xs_types.iter().zip(stacked_x_cotangents) {
            let values = values.into_iter().collect::<Option<Vec<_>>>().ok_or(ScanError::MissingEagerOutputMetadata)?;
            outputs.push(V::scan_stack_leading_axis_with_exemplar(output_type, values, inputs.first())?);
        }
        Ok(outputs)
    }
}

impl<V: Traceable<ArrayType> + ScanValue> LinearOperation<ArrayType, V, LinearPrimitiveOperation<V>>
    for LinearizedScanJvpOperation<V>
{
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V, LinearPrimitiveOperation<V>>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V, LinearPrimitiveOperation<V>>>>, TracingError> {
        let Some(first_cotangent) = output_cotangents.first() else {
            return if self.input_types().is_empty() {
                Ok(Vec::new())
            } else {
                Err(ScanError::MissingTracedInvocationContext.into())
            };
        };
        Ok(LinearTerm::apply_staged_op(
            first_cotangent.builder.clone(),
            output_cotangents,
            LinearPrimitiveOperation::LinearScanTranspose(Box::new(LinearizedScanTransposeOperation::new(
                self.clone(),
            ))),
            self.input_types().len(),
        )?
        .into_iter()
        .map(Some)
        .collect())
    }
}

impl<V: Traceable<ArrayType> + ScanValue> LinearOperation<ArrayType, V, LinearPrimitiveOperation<V>>
    for LinearizedScanTransposeOperation<V>
{
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V, LinearPrimitiveOperation<V>>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V, LinearPrimitiveOperation<V>>>>, TracingError> {
        let Some(first_cotangent) = output_cotangents.first() else {
            return if self.output_types().is_empty() {
                Ok(Vec::new())
            } else {
                Err(ScanError::MissingTracedInvocationContext.into())
            };
        };
        Ok(LinearTerm::apply_staged_op(
            first_cotangent.builder.clone(),
            output_cotangents,
            LinearPrimitiveOperation::LinearScanJvp(Box::new(self.jvp.clone())),
            self.output_types().len(),
        )?
        .into_iter()
        .map(Some)
        .collect())
    }
}

impl<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>> Operation<ArrayType> for ScanOperation<ArrayType, V, O> {
    fn name(&self) -> &'static str {
        "scan"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        let expected_input_count = self.body.carry_types.len() + self.body.xs_types.len();
        if input_types.len() != expected_input_count {
            return Err(TypeError {
                message: format!("scan expected {expected_input_count} input types but got {}", input_types.len()),
            });
        }
        let expected_input_types =
            self.body.carry_types.iter().chain(self.body.xs_types.iter()).cloned().collect::<Vec<_>>();
        if input_types != expected_input_types.as_slice() {
            return Err(TypeError { message: "scan input types do not match the captured body signature".to_string() });
        }
        Ok(self.body.carry_types.iter().chain(self.body.ys_types.iter()).cloned().collect())
    }
}

impl<V: ScanValue, O: Clone + Operation<ArrayType>> InterpretableOperation<ArrayType, V>
    for ScanOperation<ArrayType, V, O>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    O: InterpretableOperation<ArrayType, V>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        let abstract_inputs = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let _ = self.infer_output_types(abstract_inputs.as_slice())?;

        let carry_count = self.body.carry_types.len();
        let x_count = self.body.x_types.len();
        let y_count = self.body.y_types.len();
        let mut carry = inputs[..carry_count].to_vec();
        let xs = &inputs[carry_count..];
        let mut stacked_y_leaves = vec![vec![None; self.body.length]; y_count];

        for step in 0..self.body.length {
            let scan_index = if self.reverse { self.body.length - 1 - step } else { step };
            let x_slices = xs.iter().map(|x| x.scan_slice_leading_axis(scan_index)).collect::<Result<Vec<_>, _>>()?;
            let mut body_inputs = Vec::with_capacity(carry_count + x_count);
            body_inputs.extend(carry.iter().cloned());
            body_inputs.extend(x_slices);
            let body_outputs = self.body.program.interpret(body_inputs)?;
            let got_carry_count = body_outputs.len().min(carry_count);
            if got_carry_count != carry_count || body_outputs.len() != carry_count + y_count {
                return Err(TracingError::InvalidOutputCount {
                    expected: carry_count + y_count,
                    got: body_outputs.len(),
                });
            }
            carry = body_outputs[..carry_count].to_vec();
            for (output_index, value) in body_outputs[carry_count..].iter().cloned().enumerate() {
                stacked_y_leaves[output_index][scan_index] = Some(value);
            }
        }

        let mut outputs = carry;
        for (output_type, values) in self.body.ys_types.iter().zip(stacked_y_leaves) {
            let values = values.into_iter().collect::<Option<Vec<_>>>().ok_or(ScanError::MissingEagerOutputMetadata)?;
            outputs.push(V::scan_stack_leading_axis_with_exemplar(output_type, values, inputs.first())?);
        }
        Ok(outputs)
    }
}

impl<V, E> DifferentiableOperation<E> for ScanOperation<ArrayType, V, PrimitiveOperation<V>>
where
    V: ScanValue
        + ZeroLike
        + Differentiable<
            ArrayType,
            Tangent<LinearPrimitiveOperation<V>> = LinearTerm<ArrayType, V, LinearPrimitiveOperation<V>>,
        >,
    PrimitiveOperation<V>: DifferentiableOperation<E> + InterpretableOperation<ArrayType, V>,
    LinearPrimitiveOperation<V>: CoreLinearProgramOperation<V> + LinearAddOperation<ArrayType, V>,
    E: DifferentiableEngine<
            Type = ArrayType,
            Value = V,
            DifferentiableOperation = PrimitiveOperation<V>,
            LinearOperation = LinearPrimitiveOperation<V>,
        > + ?Sized,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
{
    fn jvp(
        &self,
        engine: &E,
        inputs: &[JvpTracer<V, EngineTangent<E>>],
    ) -> Result<Vec<JvpTracer<V, EngineTangent<E>>>, TracingError> {
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let primal_outputs = <Self as InterpretableOperation<ArrayType, V>>::interpret(self, primal_inputs.as_slice())?;
        let tangent_inputs = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
        let output_count = self.body.carry_types.len() + self.body.ys_types.len();
        let tangent_outputs = if output_count == 0 {
            Vec::new()
        } else {
            let Some(first_tangent) = tangent_inputs.first() else {
                return Err(ScanError::MissingTracedInvocationContext.into());
            };
            let linear_scan = LinearizedScanJvpOperation::from_scan(engine, self.clone(), primal_inputs)?;
            LinearTerm::apply_staged_op(
                first_tangent.builder.clone(),
                tangent_inputs.as_slice(),
                LinearPrimitiveOperation::LinearScanJvp(Box::new(linear_scan)),
                output_count,
            )?
        };
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| JvpTracer { primal, tangent })
            .collect())
    }
}

impl<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>> ScanOperation<ArrayType, V, O> {
    /// Replays this scan on traced dual values inside an enclosing JIT trace.
    pub fn interpret_linearized_jit<'engine, E>(
        &self,
        inputs: &[Linearized<Tracer<'engine, E, O>>],
    ) -> Result<Vec<Linearized<Tracer<'engine, E, O>>>, TracingError>
    where
        V: ZeroLike,
        E: Engine<Type = ArrayType, Value = V> + ?Sized + 'static,
        O: ScanTracingOperation<ArrayType, V>
            + InterpretableOperation<ArrayType, Linearized<Tracer<'engine, E, O>>>
            + 'static,
        LinearTerm<ArrayType, Tracer<'engine, E, O>, LinearPrimitiveOperation<Tracer<'engine, E, O>>>: ScanValue,
        Tracer<'engine, E, O>: ScanValue,
    {
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let tangent_inputs = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
        let Some(exemplar_primal_input) = primal_inputs.first().cloned() else {
            return if self.body.carry_types().is_empty() && self.body.ys_types().is_empty() {
                Ok(Vec::new())
            } else {
                Err(ScanError::MissingTracedInvocationContext.into())
            };
        };
        let primal_outputs = Tracer::apply_staged_op(
            exemplar_primal_input.engine,
            exemplar_primal_input.builder.clone(),
            primal_inputs.as_slice(),
            O::scan_op(self.clone()),
        )?;
        let linear_builder = inputs[0].tangent.builder.clone();
        let carry_count = self.body.carry_types.len();
        let x_count = self.body.x_types.len();
        let y_count = self.body.y_types.len();
        let mut carry_primals = primal_inputs[..carry_count].to_vec();
        let xs_primals = &primal_inputs[carry_count..];
        let mut carry_tangents = tangent_inputs[..carry_count].to_vec();
        let xs_tangents = &tangent_inputs[carry_count..];
        let mut stacked_y_tangents = vec![vec![None; self.body.length]; y_count];

        for step in 0..self.body.length {
            let scan_index = if self.reverse { self.body.length - 1 - step } else { step };
            let x_primals = xs_primals
                .iter()
                .map(|input| input.scan_slice_leading_axis(scan_index))
                .collect::<Result<Vec<_>, _>>()?;
            let x_tangents = xs_tangents
                .iter()
                .map(|input| input.scan_slice_leading_axis(scan_index))
                .collect::<Result<Vec<_>, _>>()?;
            let mut body_inputs = Vec::with_capacity(carry_count + x_count);
            body_inputs.extend(
                carry_primals
                    .into_iter()
                    .zip(carry_tangents)
                    .map(|(primal, tangent)| Linearized { primal, tangent }),
            );
            body_inputs
                .extend(x_primals.into_iter().zip(x_tangents).map(|(primal, tangent)| Linearized { primal, tangent }));
            let body_outputs = replay_program_linearized_jit(
                exemplar_primal_input.engine,
                exemplar_primal_input.builder.clone(),
                linear_builder.clone(),
                self.body.program(),
                body_inputs,
            )?;
            carry_primals = body_outputs[..carry_count].iter().map(|output| output.primal.clone()).collect();
            carry_tangents = body_outputs[..carry_count].iter().map(|output| output.tangent.clone()).collect();
            for (output_index, output) in body_outputs[carry_count..].iter().enumerate() {
                stacked_y_tangents[output_index][scan_index] = Some(output.tangent.clone());
            }
        }

        let mut tangent_outputs = carry_tangents;
        for (output_type, values) in self.body.ys_types.iter().zip(stacked_y_tangents) {
            let values = values.into_iter().collect::<Option<Vec<_>>>().ok_or(ScanError::MissingEagerOutputMetadata)?;
            tangent_outputs.push(LinearTerm::scan_stack_leading_axis_with_exemplar(
                output_type,
                values,
                tangent_inputs.first(),
            )?);
        }
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| Linearized { primal, tangent })
            .collect())
    }
}

fn scan_length_and_step_types(
    explicit_length: Option<usize>,
    xs_types: &[ArrayType],
) -> Result<(usize, Vec<ArrayType>), TracingError> {
    let mut length = explicit_length;
    let mut x_types = Vec::with_capacity(xs_types.len());
    for (input_index, xs_type) in xs_types.iter().enumerate() {
        if xs_type.rank() == 0 {
            return Err(ScanError::MissingLeadingAxis { input_index }.into());
        }
        let (x_type, axis_size) = xs_type.without_dimension(0)?;
        let Some(axis_length) = axis_size.value() else {
            return Err(ScanError::DynamicLength { input_index, size: axis_size }.into());
        };
        match length {
            Some(expected) if expected != axis_length => {
                return Err(ScanError::LengthMismatch { expected, got: axis_length, input_index }.into());
            }
            Some(_) => {}
            None => length = Some(axis_length),
        }
        x_types.push(x_type);
    }
    let length = length.ok_or(ScanError::MissingLength)?;
    Ok((length, x_types))
}

fn stacked_y_types(length: usize, y_types: &[ArrayType]) -> Result<Vec<ArrayType>, TracingError> {
    y_types
        .iter()
        .map(|y_type| y_type.with_inserted_dimension(0, Size::Static(length)).map_err(TracingError::from))
        .collect()
}

fn validate_body_carry_types(carry_types: &[ArrayType], body_output_types: &[ArrayType]) -> Result<(), TracingError> {
    if body_output_types.len() < carry_types.len() {
        return Err(ScanError::CarryCountMismatch { expected: carry_types.len(), got: body_output_types.len() }.into());
    }
    for (index, (expected, got)) in carry_types.iter().zip(body_output_types).enumerate() {
        if got != expected {
            return Err(ScanError::CarryTypeMismatch { index, expected: expected.clone(), got: got.clone() }.into());
        }
    }
    Ok(())
}

/// Dispatch trait used by [`scan_with_options`] to handle concrete and traced leaves.
#[doc(hidden)]
pub trait ScanInvocationLeaf<
    Carry: Parameterized<Self, ParameterStructure: Clone>,
    Xs: Parameterized<Self, ParameterStructure: Clone>,
    Y: Parameterized<Self, ParameterStructure: Clone>,
>: Parameter + Sized
{
    /// Invokes scan for one leaf regime.
    fn invoke_scan<F>(function: F, init: Carry, xs: Xs, options: ScanOptions) -> Result<(Carry, Y), TracingError>
    where
        F: FnMut((Carry, Xs)) -> (Carry, Y);
}

impl<
    V: Value<ArrayType> + ScanValue,
    Carry: Clone + Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    Xs: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    Y: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
> ScanInvocationLeaf<Carry, Xs, Y> for V
{
    fn invoke_scan<F>(mut function: F, init: Carry, xs: Xs, options: ScanOptions) -> Result<(Carry, Y), TracingError>
    where
        F: FnMut((Carry, Xs)) -> (Carry, Y),
    {
        let carry_structure = init.parameter_structure();
        let xs_structure = xs.parameter_structure();
        let carry_types = init.parameters().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
        let xs_values = xs.into_parameters().collect::<Vec<_>>();
        let xs_types = xs_values.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
        let (length, _) = scan_length_and_step_types(options.length, xs_types.as_slice())?;

        if length == 0 {
            let x_slices =
                xs_values.iter().map(ScanValue::scan_empty_slice_leading_axis).collect::<Result<Vec<_>, _>>()?;
            let x_step = Xs::from_parameters(xs_structure, x_slices)?;
            let (next_carry, y) = function((init.clone(), x_step));
            if next_carry.parameter_structure() != carry_structure {
                return Err(crate::parameters::ParameterError::MismatchedParameterStructures {
                    left_structure: format!("{:?}", carry_structure),
                    right_structure: format!("{:?}", next_carry.parameter_structure()),
                }
                .into());
            }
            let next_carry_types = next_carry.parameters().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
            validate_body_carry_types(carry_types.as_slice(), next_carry_types.as_slice())?;
            let y_structure = y.parameter_structure();
            let y_types = y.parameters().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
            let ys_types = stacked_y_types(length, y_types.as_slice())?;
            let ys_leaves = ys_types
                .iter()
                .map(|output_type| V::scan_stack_leading_axis_with_exemplar(output_type, Vec::new(), xs_values.first()))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok((init, Y::from_parameters(y_structure, ys_leaves)?));
        }

        let mut carry = init;
        let mut y_structure = None;
        let mut y_types = None;
        let mut stacked_y_leaves = None::<Vec<Vec<Option<V>>>>;
        for step in 0..length {
            let scan_index = if options.reverse { length - 1 - step } else { step };
            let x_slices =
                xs_values.iter().map(|x| x.scan_slice_leading_axis(scan_index)).collect::<Result<Vec<_>, _>>()?;
            let x_step = Xs::from_parameters(xs_structure.clone(), x_slices)?;
            let (next_carry, y) = function((carry, x_step));
            if next_carry.parameter_structure() != carry_structure {
                return Err(crate::parameters::ParameterError::MismatchedParameterStructures {
                    left_structure: format!("{:?}", carry_structure),
                    right_structure: format!("{:?}", next_carry.parameter_structure()),
                }
                .into());
            }
            let next_carry_types = next_carry.parameters().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
            validate_body_carry_types(carry_types.as_slice(), next_carry_types.as_slice())?;
            carry = next_carry;

            let current_y_structure = y.parameter_structure();
            if let Some(expected_y_structure) = &y_structure {
                if current_y_structure != *expected_y_structure {
                    return Err(crate::parameters::ParameterError::MismatchedParameterStructures {
                        left_structure: format!("{expected_y_structure:?}"),
                        right_structure: format!("{current_y_structure:?}"),
                    }
                    .into());
                }
            } else {
                let current_y_types = y.parameters().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
                y_types = Some(current_y_types.clone());
                stacked_y_leaves = Some(vec![vec![None; length]; current_y_types.len()]);
                y_structure = Some(current_y_structure.clone());
            }

            let y_leaves = y.into_parameters().collect::<Vec<_>>();
            let stacked_y_leaves = stacked_y_leaves.as_mut().ok_or(ScanError::MissingEagerOutputMetadata)?;
            for (output_index, value) in y_leaves.into_iter().enumerate() {
                stacked_y_leaves[output_index][scan_index] = Some(value);
            }
        }

        let Some(y_structure) = y_structure else {
            return Err(ScanError::MissingEagerOutputMetadata.into());
        };
        let y_types = y_types.expect("non-empty eager scan should record y types");
        let ys_types = stacked_y_types(length, y_types.as_slice())?;
        let ys_leaves = ys_types
            .iter()
            .zip(stacked_y_leaves.expect("non-empty eager scan should record y leaves"))
            .map(|(output_type, values)| {
                let values =
                    values.into_iter().collect::<Option<Vec<_>>>().ok_or(ScanError::MissingEagerOutputMetadata)?;
                V::scan_stack_leading_axis_with_exemplar(output_type, values, xs_values.first())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((carry, Y::from_parameters(y_structure, ys_leaves)?))
    }
}

impl<
    'engine,
    E,
    V: Traceable<ArrayType>,
    Carry: Parameterized<
            Tracer<'engine, E>,
            ParameterStructure: Clone + std::fmt::Debug + PartialEq,
            To<Tracer<'engine, E>> = Carry,
        >,
    Xs: Parameterized<Tracer<'engine, E>, ParameterStructure: Clone, To<Tracer<'engine, E>> = Xs>,
    Y: Parameterized<Tracer<'engine, E>, ParameterStructure: Clone, To<Tracer<'engine, E>> = Y>,
> ScanInvocationLeaf<Carry, Xs, Y> for Tracer<'engine, E>
where
    E: Engine<Type = ArrayType, Value = V> + ?Sized + 'static,
    V: Parameterized<V, ParameterStructure = Placeholder>,
    Carry::Family: ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Xs::Family: ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Y::Family: ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Carry::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Carry, To<V> = Carry::To<V>>,
    Xs::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Xs, To<V> = Xs::To<V>>,
    Y::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Y, To<V> = Y::To<V>>,
    E::TracingOperation: ScanTracingOperation<ArrayType, V>,
{
    fn invoke_scan<F>(mut function: F, init: Carry, xs: Xs, options: ScanOptions) -> Result<(Carry, Y), TracingError>
    where
        F: FnMut((Carry, Xs)) -> (Carry, Y),
    {
        let carry_structure = init.parameter_structure();
        let xs_structure = xs.parameter_structure();
        let traced_carry = init.into_parameters().collect::<Vec<_>>();
        let traced_xs = xs.into_parameters().collect::<Vec<_>>();
        let carry_count = traced_carry.len();
        let carry_types = traced_carry.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let xs_types = traced_xs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let (length, x_types) = scan_length_and_step_types(options.length, xs_types.as_slice())?;
        let x_step_structure = xs_structure.clone();
        let traced_inputs = traced_carry.into_iter().chain(traced_xs).collect::<Vec<_>>();
        let Some(exemplar_traced_input) = traced_inputs.first().cloned() else {
            return Err(ScanError::MissingTracedInvocationContext.into());
        };
        let mut body_input_types = Vec::with_capacity(carry_types.len() + x_types.len());
        body_input_types.extend(carry_types.iter().cloned());
        body_input_types.extend(x_types.iter().cloned());
        let mut y_structure = None;

        let (body_output_types, body_program) =
            trace_flat_program_from_input_types::<Vec<ArrayType>, Vec<ArrayType>, V, E, E::TracingOperation, _>(
                exemplar_traced_input.engine,
                |body_input| {
                    let mut body_input = body_input.into_iter();
                    let carry_input =
                        Carry::from_parameters(carry_structure.clone(), body_input.by_ref().take(carry_count))?;
                    let x_input = Xs::from_parameters(x_step_structure.clone(), body_input)?;
                    let (carry_output, y_output) = function((carry_input, x_input));
                    let carry_output_structure = carry_output.parameter_structure();
                    if carry_output_structure != carry_structure {
                        return Err(crate::parameters::ParameterError::MismatchedParameterStructures {
                            left_structure: format!("{:?}", carry_structure),
                            right_structure: format!("{carry_output_structure:?}"),
                        }
                        .into());
                    }
                    y_structure = Some(y_output.parameter_structure());
                    Ok(carry_output.into_parameters().chain(y_output.into_parameters()).collect())
                },
                body_input_types,
            )?;
        let body_output_leaves = body_output_types;
        validate_body_carry_types(carry_types.as_slice(), body_output_leaves.as_slice())?;
        let y_types = body_output_leaves[carry_types.len()..].to_vec();
        let ys_types = stacked_y_types(length, y_types.as_slice())?;
        let y_structure = y_structure.ok_or(ScanError::MissingTracedInvocationContext)?;
        let ys_output_count = y_structure.parameter_count();
        let body = FlatTracedScan::from_parts(carry_types, x_types, y_types, xs_types, ys_types, length, body_program);

        let staged_outputs = Tracer::apply_staged_op(
            exemplar_traced_input.engine,
            exemplar_traced_input.builder.clone(),
            traced_inputs.as_slice(),
            E::TracingOperation::scan_op(ScanOperation::new(body, options)),
        )?;
        let mut staged_outputs = staged_outputs.into_iter();
        let carry_outputs = staged_outputs.by_ref().take(carry_structure.parameter_count()).collect::<Vec<_>>();
        let y_outputs = staged_outputs.take(ys_output_count).collect::<Vec<_>>();
        Ok((Carry::from_parameters(carry_structure, carry_outputs)?, Y::from_parameters(y_structure, y_outputs)?))
    }
}

/// Runs `function` over the leading axis of `xs`, carrying state between iterations.
#[allow(private_bounds)]
pub fn scan<F, Carry, Xs, Y, V>(function: F, init: Carry, xs: Xs) -> Result<(Carry, Y), TracingError>
where
    V: ScanInvocationLeaf<Carry, Xs, Y>,
    Carry: Parameterized<V, ParameterStructure: Clone>,
    Xs: Parameterized<V, ParameterStructure: Clone>,
    Y: Parameterized<V, ParameterStructure: Clone>,
    F: FnMut((Carry, Xs)) -> (Carry, Y),
{
    scan_with_options(function, init, xs, ScanOptions::default())
}

/// Runs `function` over the leading axis of `xs` with explicit scan options.
#[allow(private_bounds)]
pub fn scan_with_options<F, Carry, Xs, Y, V>(
    function: F,
    init: Carry,
    xs: Xs,
    options: ScanOptions,
) -> Result<(Carry, Y), TracingError>
where
    V: ScanInvocationLeaf<Carry, Xs, Y>,
    Carry: Parameterized<V, ParameterStructure: Clone>,
    Xs: Parameterized<V, ParameterStructure: Clone>,
    Y: Parameterized<V, ParameterStructure: Clone>,
    F: FnMut((Carry, Xs)) -> (Carry, Y),
{
    V::invoke_scan(function, init, xs, options)
}

/// Runs `function` for a static number of iterations without scanned inputs.
#[allow(private_bounds)]
pub fn scan_without_xs<F, Carry, Y, V>(function: F, init: Carry, length: usize) -> Result<(Carry, Y), TracingError>
where
    V: ScanInvocationLeaf<Carry, (), Y>,
    Carry: Parameterized<V, ParameterStructure: Clone>,
    Y: Parameterized<V, ParameterStructure: Clone>,
    F: FnMut((Carry, ())) -> (Carry, Y),
{
    scan_without_xs_with_options(function, init, ScanOptions::default().with_length(length))
}

/// Runs `function` without scanned inputs using explicit scan options.
#[allow(private_bounds)]
pub fn scan_without_xs_with_options<F, Carry, Y, V>(
    function: F,
    init: Carry,
    options: ScanOptions,
) -> Result<(Carry, Y), TracingError>
where
    V: ScanInvocationLeaf<Carry, (), Y>,
    Carry: Parameterized<V, ParameterStructure: Clone>,
    Y: Parameterized<V, ParameterStructure: Clone>,
    F: FnMut((Carry, ())) -> (Carry, Y),
{
    V::invoke_scan(function, init, (), options)
}

#[cfg(test)]
mod tests {
    use std::{
        borrow::Cow,
        ops::{Add, Mul, Neg},
    };

    use pretty_assertions::assert_eq;

    use crate::{
        parameters::Parameter,
        tracing::{Program, Traceable, Value},
        tracing_v2::{
            Cos, DifferentiableEngine, Engine, LinearPrimitiveOperation, MatrixOps, PrimitiveOperation, Sin, jvp,
            linear::{jvp_program, transpose_linear_program, vjp},
            operations::{
                ControlFlowError, ControlFlowValue,
                constants::{OneLike, ZeroLike},
                reshape::ReshapeOps,
            },
            trace,
        },
        types::{DataType, Shape, Typed},
    };

    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct TestArray {
        r#type: ArrayType,
        values: Vec<f64>,
    }

    impl TestArray {
        fn scalar(value: f64) -> Self {
            Self { r#type: scalar_type(), values: vec![value] }
        }

        fn vector(values: Vec<f64>) -> Self {
            Self { r#type: vector_type(values.len()), values }
        }

        fn element_count(r#type: &ArrayType) -> usize {
            if r#type.rank() == 0 {
                1
            } else {
                r#type
                    .shape
                    .dimensions
                    .iter()
                    .map(|dimension| dimension.value().expect("test arrays use static dimensions"))
                    .product()
            }
        }
    }

    impl Parameter for TestArray {}

    impl Typed<ArrayType> for TestArray {
        fn r#type(&self) -> Cow<'_, ArrayType> {
            Cow::Borrowed(&self.r#type)
        }
    }

    impl Traceable<ArrayType> for TestArray {}

    impl Value<ArrayType> for TestArray {}

    impl ControlFlowValue for TestArray {
        fn control_flow_predicate(&self) -> Result<bool, TracingError> {
            Err(ControlFlowError::InvalidPredicateValue { type_: self.r#type().into_owned() }.into())
        }
    }

    impl Add for TestArray {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            assert_eq!(self.r#type, rhs.r#type);
            Self {
                r#type: self.r#type,
                values: self.values.into_iter().zip(rhs.values).map(|(left, right)| left + right).collect(),
            }
        }
    }

    impl Mul for TestArray {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            assert_eq!(self.r#type, rhs.r#type);
            Self {
                r#type: self.r#type,
                values: self.values.into_iter().zip(rhs.values).map(|(left, right)| left * right).collect(),
            }
        }
    }

    impl Neg for TestArray {
        type Output = Self;

        fn neg(self) -> Self::Output {
            Self { r#type: self.r#type, values: self.values.into_iter().map(|value| -value).collect() }
        }
    }

    impl Sin for TestArray {
        fn sin(self) -> Self {
            Self { r#type: self.r#type, values: self.values.into_iter().map(f64::sin).collect() }
        }
    }

    impl Cos for TestArray {
        fn cos(self) -> Self {
            Self { r#type: self.r#type, values: self.values.into_iter().map(f64::cos).collect() }
        }
    }

    impl ZeroLike for TestArray {
        fn zero_like(&self) -> Self {
            Self { r#type: self.r#type.clone(), values: vec![0.0; self.values.len()] }
        }
    }

    impl OneLike for TestArray {
        fn one_like(&self) -> Self {
            Self { r#type: self.r#type.clone(), values: vec![1.0; self.values.len()] }
        }
    }

    impl MatrixOps for TestArray {
        fn matmul(self, rhs: Self) -> Self {
            self * rhs
        }

        fn transpose_matrix(self) -> Self {
            self
        }
    }

    impl ReshapeOps for TestArray {
        fn reshape(self, target_shape: Shape) -> Result<Self, TracingError> {
            let output_type = ArrayType::new(self.r#type.data_type, target_shape, None, None).unwrap();
            assert_eq!(Self::element_count(&self.r#type), Self::element_count(&output_type));
            Ok(Self { r#type: output_type, values: self.values })
        }
    }

    impl ScanValue for TestArray {
        fn scan_slice_leading_axis(&self, index: usize) -> Result<Self, TracingError> {
            let (slice_type, length) = self.r#type.without_dimension(0)?;
            let Some(length) = length.value() else {
                return Err(ScanError::DynamicLength { input_index: 0, size: length }.into());
            };
            assert!(index < length);
            let slice_element_count = Self::element_count(&slice_type);
            let offset = index * slice_element_count;
            Ok(Self { r#type: slice_type, values: self.values[offset..offset + slice_element_count].to_vec() })
        }

        fn scan_empty_slice_leading_axis(&self) -> Result<Self, TracingError> {
            let (slice_type, _) = self.r#type.without_dimension(0)?;
            Ok(Self { values: vec![0.0; Self::element_count(&slice_type)], r#type: slice_type })
        }

        fn scan_stack_leading_axis(output_type: &ArrayType, values: Vec<Self>) -> Result<Self, TracingError> {
            let (element_type, length) = output_type.without_dimension(0)?;
            assert_eq!(length.value(), Some(values.len()));
            let mut stacked_values = Vec::new();
            for value in values {
                assert_eq!(value.r#type, element_type);
                stacked_values.extend(value.values);
            }
            Ok(Self { r#type: output_type.clone(), values: stacked_values })
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct TestArrayEngine;

    impl Engine for TestArrayEngine {
        type Type = ArrayType;
        type Value = TestArray;
        type TracingOperation = PrimitiveOperation<TestArray>;

        fn zero(&self, r#type: &ArrayType) -> Result<Self::Value, TracingError> {
            Ok(TestArray { r#type: r#type.clone(), values: vec![0.0; TestArray::element_count(r#type)] })
        }

        fn one(&self, r#type: &ArrayType) -> Result<Self::Value, TracingError> {
            Ok(TestArray { r#type: r#type.clone(), values: vec![1.0; TestArray::element_count(r#type)] })
        }
    }

    impl DifferentiableEngine for TestArrayEngine {
        type DifferentiableOperation = PrimitiveOperation<TestArray>;
        type LinearOperation = LinearPrimitiveOperation<TestArray>;
    }

    fn scalar_type() -> ArrayType {
        ArrayType::scalar(DataType::F64)
    }

    fn vector_type(length: usize) -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(length)]), None, None).unwrap()
    }

    #[test]
    fn test_scan_eager_reference_stacks_outputs() {
        let (carry, ys): (TestArray, TestArray) = scan(
            |(carry, x)| {
                let next_carry = carry + x;
                (next_carry.clone(), next_carry)
            },
            TestArray::scalar(0.0),
            TestArray::vector(vec![1.0, 2.0, 3.0]),
        )
        .unwrap();

        assert_eq!(carry, TestArray::scalar(6.0));
        assert_eq!(ys, TestArray::vector(vec![1.0, 3.0, 6.0]));
    }

    #[test]
    fn test_scan_reverse_preserves_output_axis_order() {
        let (carry, ys): (TestArray, TestArray) = scan_with_options(
            |(carry, x)| {
                let next_carry = carry + x;
                (next_carry.clone(), next_carry)
            },
            TestArray::scalar(0.0),
            TestArray::vector(vec![1.0, 2.0, 3.0]),
            ScanOptions::default().with_reverse(true),
        )
        .unwrap();

        assert_eq!(carry, TestArray::scalar(6.0));
        assert_eq!(ys, TestArray::vector(vec![6.0, 5.0, 3.0]));
    }

    #[test]
    fn test_scan_eager_zero_length_infers_output_metadata_from_body() {
        let (carry, ys): (TestArray, TestArray) = scan(
            |(carry, x)| {
                let next_carry = carry + x;
                (next_carry.clone(), next_carry)
            },
            TestArray::scalar(5.0),
            TestArray::vector(vec![]),
        )
        .unwrap();

        assert_eq!(carry, TestArray::scalar(5.0));
        assert_eq!(ys, TestArray::vector(vec![]));
    }

    #[test]
    fn test_scan_rejects_mismatched_explicit_length() {
        let result: Result<(TestArray, TestArray), TracingError> = scan_with_options(
            |(carry, x)| {
                let next_carry = carry + x;
                (next_carry.clone(), next_carry)
            },
            TestArray::scalar(0.0),
            TestArray::vector(vec![1.0, 2.0, 3.0]),
            ScanOptions::default().with_length(2),
        );

        assert!(matches!(
            result,
            Err(TracingError::Scan(ScanError::LengthMismatch { expected: 2, got: 3, input_index: 0 }))
        ));
    }

    #[test]
    fn test_scan_rejects_dynamic_leading_axis() {
        let engine = TestArrayEngine;
        let result: Result<
            (
                (ArrayType, ArrayType),
                Program<
                    ArrayType,
                    TestArray,
                    PrimitiveOperation<TestArray>,
                    (TestArray, TestArray),
                    (TestArray, TestArray),
                >,
            ),
            TracingError,
        > = trace(
            &engine,
            |(carry, xs)| {
                scan(
                    |(carry, x)| {
                        let next_carry = carry + x;
                        (next_carry.clone(), next_carry)
                    },
                    carry,
                    xs,
                )
            },
            (scalar_type(), ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None)]), None, None).unwrap()),
        );

        assert!(matches!(
            result,
            Err(TracingError::Scan(ScanError::DynamicLength { input_index: 0, size: Size::Dynamic(None) }))
        ));
    }

    #[test]
    fn test_scan_without_xs_eager_uses_explicit_length() {
        let (carry, ys): (TestArray, TestArray) = scan_without_xs(
            |(carry, ())| {
                let next_carry = carry.clone() + carry;
                (next_carry.clone(), next_carry)
            },
            TestArray::scalar(1.0),
            3,
        )
        .unwrap();

        assert_eq!(carry, TestArray::scalar(8.0));
        assert_eq!(ys, TestArray::vector(vec![2.0, 4.0, 8.0]));
    }

    #[test]
    fn test_scan_without_xs_eager_zero_length_infers_output_metadata_from_body() {
        let (carry, ys): (TestArray, TestArray) = scan_without_xs(
            |(carry, ())| {
                let next_carry = carry.clone() + carry;
                (next_carry.clone(), next_carry)
            },
            TestArray::scalar(5.0),
            0,
        )
        .unwrap();

        assert_eq!(carry, TestArray::scalar(5.0));
        assert_eq!(ys, TestArray::vector(vec![]));
    }

    #[test]
    fn test_scan_jvp_propagates_carry_and_stacked_output_tangents() {
        let engine = TestArrayEngine;
        let (primal, tangent): ((TestArray, TestArray), (TestArray, TestArray)) = jvp(
            &engine,
            |(carry, xs)| {
                scan(
                    |(carry, x)| {
                        let next_carry = carry + x;
                        (next_carry.clone(), next_carry)
                    },
                    carry,
                    xs,
                )
                .unwrap()
            },
            (TestArray::scalar(0.0), TestArray::vector(vec![1.0, 2.0, 3.0])),
            (TestArray::scalar(10.0), TestArray::vector(vec![1.0, 1.0, 1.0])),
        )
        .unwrap();

        assert_eq!(primal, (TestArray::scalar(6.0), TestArray::vector(vec![1.0, 3.0, 6.0])));
        assert_eq!(tangent, (TestArray::scalar(13.0), TestArray::vector(vec![11.0, 12.0, 13.0])));
    }

    #[test]
    fn test_scan_jvp_program_preserves_compact_linear_scan() {
        let engine = TestArrayEngine;
        let (primal, pushforward): (
            (TestArray, TestArray),
            Program<
                ArrayType,
                TestArray,
                LinearPrimitiveOperation<TestArray>,
                (TestArray, TestArray),
                (TestArray, TestArray),
            >,
        ) = jvp_program(
            &engine,
            |(carry, xs)| {
                scan(
                    |(carry, x)| {
                        let next_carry = carry + x;
                        (next_carry.clone(), next_carry)
                    },
                    carry,
                    xs,
                )
            },
            (TestArray::scalar(0.0), TestArray::vector(vec![1.0, 2.0, 3.0])),
        )
        .unwrap();

        assert_eq!(primal, (TestArray::scalar(6.0), TestArray::vector(vec![1.0, 3.0, 6.0])));
        assert_eq!(pushforward.instructions.len(), 1);
        assert_eq!(pushforward.instructions[0].operation.name(), "linear_scan_jvp");
        assert!(!pushforward.to_string().contains("slice_leading_axis"));
        assert!(!pushforward.to_string().contains("stack_leading_axis"));

        let tangent = pushforward.interpret((TestArray::scalar(10.0), TestArray::vector(vec![1.0, 1.0, 1.0]))).unwrap();
        assert_eq!(tangent, (TestArray::scalar(13.0), TestArray::vector(vec![11.0, 12.0, 13.0])));
    }

    #[test]
    fn test_scan_transpose_program_preserves_compact_linear_scan() {
        let engine = TestArrayEngine;
        let (_, pushforward): (
            (TestArray, TestArray),
            Program<
                ArrayType,
                TestArray,
                LinearPrimitiveOperation<TestArray>,
                (TestArray, TestArray),
                (TestArray, TestArray),
            >,
        ) = jvp_program(
            &engine,
            |(carry, xs)| {
                scan(
                    |(carry, x)| {
                        let next_carry = carry + x;
                        (next_carry.clone(), next_carry)
                    },
                    carry,
                    xs,
                )
            },
            (TestArray::scalar(0.0), TestArray::vector(vec![1.0, 2.0, 3.0])),
        )
        .unwrap();
        let pullback = transpose_linear_program(&engine, &pushforward).unwrap();

        assert_eq!(pullback.instructions.len(), 1);
        assert_eq!(pullback.instructions[0].operation.name(), "linear_scan_transpose");
        assert!(!pullback.to_string().contains("slice_leading_axis"));
        assert!(!pullback.to_string().contains("stack_leading_axis"));

        let cotangents = pullback.interpret((TestArray::scalar(1.0), TestArray::vector(vec![0.0, 0.0, 0.0]))).unwrap();
        assert_eq!(cotangents, (TestArray::scalar(1.0), TestArray::vector(vec![1.0, 1.0, 1.0])));
    }

    #[test]
    fn test_scan_transpose_pullback_propagates_final_carry_cotangent() {
        let engine = TestArrayEngine;
        let (output, pullback): (
            (TestArray, TestArray),
            Program<
                ArrayType,
                TestArray,
                LinearPrimitiveOperation<TestArray>,
                (TestArray, TestArray),
                (TestArray, TestArray),
            >,
        ) = vjp(
            &engine,
            |(carry, xs)| {
                scan(
                    |(carry, x)| {
                        let next_carry = carry + x;
                        (next_carry.clone(), next_carry)
                    },
                    carry,
                    xs,
                )
            },
            (TestArray::scalar(0.0), TestArray::vector(vec![1.0, 2.0, 3.0])),
        )
        .unwrap();
        let cotangents = pullback.interpret((TestArray::scalar(1.0), TestArray::vector(vec![0.0, 0.0, 0.0]))).unwrap();

        assert_eq!(output, (TestArray::scalar(6.0), TestArray::vector(vec![1.0, 3.0, 6.0])));
        assert_eq!(cotangents, (TestArray::scalar(1.0), TestArray::vector(vec![1.0, 1.0, 1.0])));
    }

    #[test]
    fn test_scan_traces_as_single_higher_order_operation() {
        let engine = TestArrayEngine;
        let (_, program): (
            (ArrayType, ArrayType),
            Program<
                ArrayType,
                TestArray,
                PrimitiveOperation<TestArray>,
                (TestArray, TestArray),
                (TestArray, TestArray),
            >,
        ) = trace(
            &engine,
            |(carry, xs)| {
                scan(
                    |(carry, x)| {
                        let next_carry = carry + x;
                        (next_carry.clone(), next_carry)
                    },
                    carry,
                    xs,
                )
            },
            (scalar_type(), vector_type(3)),
        )
        .unwrap();

        assert_eq!(program.instructions.len(), 1);
        assert_eq!(program.instructions[0].operation.name(), "scan");
        assert_eq!(
            program.outputs().map(|output| output.r#type().into_owned()).collect::<Vec<_>>(),
            vec![scalar_type(), vector_type(3),]
        );
        assert!(program.to_string().contains("scan"));
    }

    #[test]
    fn test_scan_traces_zero_length_xs_as_single_higher_order_operation() {
        let engine = TestArrayEngine;
        let (_, program): (
            (ArrayType, ArrayType),
            Program<
                ArrayType,
                TestArray,
                PrimitiveOperation<TestArray>,
                (TestArray, TestArray),
                (TestArray, TestArray),
            >,
        ) = trace(
            &engine,
            |(carry, xs)| {
                scan(
                    |(carry, x)| {
                        let next_carry = carry + x;
                        (next_carry.clone(), next_carry)
                    },
                    carry,
                    xs,
                )
            },
            (scalar_type(), vector_type(0)),
        )
        .unwrap();

        assert_eq!(program.instructions.len(), 1);
        assert_eq!(
            program.outputs().map(|output| output.r#type().into_owned()).collect::<Vec<_>>(),
            vec![scalar_type(), vector_type(0),]
        );

        let output = program.interpret((TestArray::scalar(5.0), TestArray::vector(vec![]))).unwrap();
        assert_eq!(output, (TestArray::scalar(5.0), TestArray::vector(vec![])));
    }

    #[test]
    fn test_scan_tracing_rejects_changed_vec_carry_structure() {
        let engine = TestArrayEngine;
        let result: Result<
            (
                (Vec<ArrayType>, ArrayType),
                Program<
                    ArrayType,
                    TestArray,
                    PrimitiveOperation<TestArray>,
                    (Vec<TestArray>, TestArray),
                    (Vec<TestArray>, TestArray),
                >,
            ),
            TracingError,
        > = trace(
            &engine,
            |(carry, xs): (Vec<_>, _)| {
                scan(
                    |(carry, x): (Vec<_>, _)| {
                        let next_carry = carry[0].clone() + x;
                        (vec![next_carry.clone()], next_carry)
                    },
                    carry,
                    xs,
                )
            },
            (vec![scalar_type(), scalar_type()], vector_type(3)),
        );

        assert!(matches!(
            result,
            Err(TracingError::Parameter(crate::parameters::ParameterError::MismatchedParameterStructures { .. }))
        ));
    }

    #[test]
    fn test_scan_traces_options_metadata() {
        let engine = TestArrayEngine;
        let (_, program): (
            (ArrayType, ArrayType),
            Program<
                ArrayType,
                TestArray,
                PrimitiveOperation<TestArray>,
                (TestArray, TestArray),
                (TestArray, TestArray),
            >,
        ) = trace(
            &engine,
            |(carry, xs)| {
                scan_with_options(
                    |(carry, x)| {
                        let next_carry = carry + x;
                        (next_carry.clone(), next_carry)
                    },
                    carry,
                    xs,
                    ScanOptions::default().with_unroll(ScanUnroll::Full).with_split_transpose(true),
                )
            },
            (scalar_type(), vector_type(3)),
        )
        .unwrap();

        let PrimitiveOperation::Scan(scan) = &program.instructions[0].operation else {
            panic!("expected scan operation");
        };
        assert_eq!(scan.unroll(), ScanUnroll::Full);
        assert!(scan.split_transpose());
    }

    #[test]
    fn test_scan_without_xs_traces_as_single_higher_order_operation() {
        let engine = TestArrayEngine;
        let (_, program): (
            (ArrayType, ArrayType),
            Program<ArrayType, TestArray, PrimitiveOperation<TestArray>, TestArray, (TestArray, TestArray)>,
        ) = trace(
            &engine,
            |carry| {
                scan_without_xs(
                    |(carry, ())| {
                        let next_carry = carry.clone() + carry;
                        (next_carry.clone(), next_carry)
                    },
                    carry,
                    3,
                )
            },
            scalar_type(),
        )
        .unwrap();

        assert_eq!(program.instructions.len(), 1);
        assert_eq!(program.instructions[0].operation.name(), "scan");
        assert_eq!(
            program.outputs().map(|output| output.r#type().into_owned()).collect::<Vec<_>>(),
            vec![scalar_type(), vector_type(3),]
        );
    }

    #[test]
    fn test_scan_without_xs_traces_zero_length_as_single_higher_order_operation() {
        let engine = TestArrayEngine;
        let (_, program): (
            (ArrayType, ArrayType),
            Program<ArrayType, TestArray, PrimitiveOperation<TestArray>, TestArray, (TestArray, TestArray)>,
        ) = trace(
            &engine,
            |carry| {
                scan_without_xs(
                    |(carry, ())| {
                        let next_carry = carry.clone() + carry;
                        (next_carry.clone(), next_carry)
                    },
                    carry,
                    0,
                )
            },
            scalar_type(),
        )
        .unwrap();

        assert_eq!(program.instructions.len(), 1);
        assert_eq!(
            program.outputs().map(|output| output.r#type().into_owned()).collect::<Vec<_>>(),
            vec![scalar_type(), vector_type(0),]
        );

        let output = program.interpret(TestArray::scalar(5.0)).unwrap();
        assert_eq!(output, (TestArray::scalar(5.0), TestArray::vector(vec![])));
    }
}
