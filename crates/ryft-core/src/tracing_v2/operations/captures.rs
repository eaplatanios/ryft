use std::borrow::Cow;
use std::fmt::Display;

use ryft_macros::Parameter;

use crate::differentiation::{Cotangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::control_flow::SelectCondition;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::parameters::Parameter;
use crate::programs::{AtomId, ProgramError, Value};
use crate::tracing::AbstractTracingContext;
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

/// Payload used inside residualized linear programs when a value is either embedded directly or supplied by a
/// capture environment.
///
/// A [`ValueOrCapture`] represents a primal value that a linearized tangent or cotangent program closes over. The
/// value can be carried directly in the operation payload via [`Value`](Self::Value), or it can be represented as a
/// [`Capture`](Self::Capture) into an external capture environment, such as a
/// [`Pushforward`](crate::tracing_v2::Pushforward)'s residual table or a higher-order program's capture list.
#[derive(Clone, Debug, Parameter)]
pub enum ValueOrCapture<T: Type, V: Value<T>> {
    /// Embedded value that is independent of a capture environment.
    Value(V),

    /// Reference to a value supplied by the owning capture environment.
    Capture {
        /// Zero-based capture index inside the owning environment.
        index: usize,

        /// Type metadata for the captured value.
        r#type: T,
    },
}

impl<T: Type, V: Value<T>> ValueOrCapture<T, V> {
    /// Instantiates this payload into a concrete value using `captures`.
    pub fn instantiate(&self, captures: &[V]) -> Result<V, ProgramError> {
        match self {
            Self::Value(value) => Ok(value.clone()),
            Self::Capture { index, .. } => {
                captures.get(*index).cloned().ok_or(ProgramError::UnboundAtomId { id: AtomId::new(*index) }.into())
            }
        }
    }

    /// Returns this payload's capture index, if it references one.
    pub(crate) fn residual_index(&self) -> Option<usize> {
        match self {
            Self::Value(_) => None,
            Self::Capture { index, .. } => Some(*index),
        }
    }

    /// Remaps this payload through a compacted capture-index table.
    pub(crate) fn remap_residuals(&self, mapping: &[Option<usize>]) -> Result<Self, ProgramError> {
        match self {
            Self::Value(value) => Ok(Self::Value(value.clone())),
            Self::Capture { index: old_index, r#type } => {
                let Some(Some(index)) = mapping.get(*old_index) else {
                    return Err(ProgramError::UnboundAtomId { id: AtomId::new(*old_index) }.into());
                };
                Ok(Self::Capture { index: *index, r#type: r#type.clone() })
            }
        }
    }
}

impl<T: Type, V: Value<T>> Display for ValueOrCapture<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Value(value) => Display::fmt(value, formatter),
            Self::Capture { index, .. } => write!(formatter, "capture[{index}]"),
        }
    }
}

impl<T: Type, V: Value<T>> Typed<T> for ValueOrCapture<T, V> {
    #[inline]
    fn r#type(&self) -> Cow<'_, T> {
        match self {
            Self::Value(value) => value.r#type(),
            Self::Capture { r#type, .. } => Cow::Borrowed(r#type),
        }
    }
}

impl<T: Type, V: Value<T>> Value<T> for ValueOrCapture<T, V> {
    type InterpretationContext = V::InterpretationContext;

    #[inline]
    fn interpretation_context(&self) -> Option<V::InterpretationContext> {
        match self {
            Self::Value(value) => value.interpretation_context(),
            // Captures carry no payload value to recover a context from. They are instantiated to concrete values
            // before any interpretation, and callers scan their other operands for one that yields a context.
            Self::Capture { .. } => None,
        }
    }
}

/// Scalar captured-condition payloads carry the [`SelectOperation`](crate::operations::control_flow::SelectOperation)
/// condition as an in-band scalar value, so the linear select interprets embedded values by delegating to their
/// [`SelectCondition`] implementation. Captures are residuals of the primal computation and must be instantiated
/// before interpretation, so the capture form errors here, matching
/// [`CustomVjpResidual::residual_value`](crate::tracing_v2::operations::CustomVjpResidual).
impl<V: Value<DataType> + SelectCondition> SelectCondition for ValueOrCapture<DataType, V> {
    type Condition = V::Condition;

    fn select_condition(&self) -> Result<Self::Condition, ProgramError> {
        match self {
            Self::Value(value) => value.select_condition(),
            Self::Capture { .. } => Err(ProgramError::Concretization {
                message: "captured select condition requires instantiated captures".to_string(),
            }),
        }
    }
}

/// Canonical operation name for [`MaterializeCaptureOperation`].
pub const MATERIALIZE_CAPTURE_OPERATION_NAME: &str = "materialize_capture";

/// Nullary operation that materializes a captured payload as an ordinary program value.
///
/// Most linear operations keep primal values in their operation payloads: for example, a scale stores its factor, and
/// a captured-condition select stores its condition. Sometimes a captured value must instead become an atom in the
/// linear program so later instructions or nested control-flow bodies can use it as a normal operand. This operation
/// provides that adapter: it has no inputs, resolves its captured payload, and produces one output with the payload's
/// type.
///
/// This differs from [`ConstantOperation`](crate::operations::constants::ConstantOperation), which embeds an
/// already-materialized value as a literal program constant. A captured payload may instead be a
/// [`ValueOrCapture::Capture`] into the owning capture environment, so the actual value is not available until that
/// environment is instantiated or interpreted.
///
/// The `Residual` variant of [`LinearArrayOperation`](crate::tracing_v2::LinearArrayOperation) wraps this operation so
/// array linear programs can inject such values. It is used by higher-order rules such as
/// [`WhileOperation`](crate::operations::control_flow::WhileOperation) when fusing or defactorizing nested
/// linearizations: if the capture is already available as a forwarded operand, defactorization can replace the
/// operation with that atom; otherwise interpretation resolves the payload from the owning capture environment.
///
/// This operation is not a linear map with respect to any operand. It is a constant injection into the linear IR, so
/// transposition rejects it directly.
#[derive(Clone, Debug)]
pub struct MaterializeCaptureOperation<F> {
    /// Captured payload materialized as this operation's single output.
    capture: F,
}

impl<F> MaterializeCaptureOperation<F> {
    /// Creates a new [`MaterializeCaptureOperation`] capturing the provided payload.
    #[inline]
    pub fn new(capture: F) -> Self {
        Self { capture }
    }

    /// Returns the captured payload materialized as this operation's single output.
    #[inline]
    pub fn capture(&self) -> &F {
        &self.capture
    }
}

impl<F: Value<ArrayType>> Display for MaterializeCaptureOperation<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<F: Value<ArrayType>> Operation<ArrayType> for MaterializeCaptureOperation<F> {
    #[inline]
    fn name(&self) -> &'static str {
        MATERIALIZE_CAPTURE_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 0, TypeError);
        Ok(vec![self.capture.r#type().into_owned()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("capture", &self.capture))
    }
}

impl<V, F> InterpretableOperation<ArrayType, V> for MaterializeCaptureOperation<F>
where
    V: Value<ArrayType>,
    F: crate::tracing_v2::operations::custom_derivatives::CustomVjpResidual<ArrayType, V>,
{
    fn interpret(
        &self,
        _context: &<V as Value<ArrayType>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 0, ProgramError);
        Ok(vec![self.capture().residual_value()?])
    }
}

/// Transpose rule for capture materialization (the `Residual` variant of
/// [`LinearArrayOperation`](crate::tracing_v2::LinearArrayOperation)). The injected payload is a captured primal value
/// rather than a linear operand, so the operation is not a linear map and rejects transposition. This rule is only
/// reachable behind the while transpose error, which fires first.
impl<V: Value<ArrayType>, O: Operation<ArrayType>, F: Value<ArrayType>> TransposableOperation<ArrayType, V, O>
    for MaterializeCaptureOperation<F>
{
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
        _input_types: &[&ArrayType],
        _output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: "capture materialization is not a linear map and does not support transposition".to_string(),
        })
    }
}
