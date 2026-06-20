use std::borrow::Cow;
use std::fmt::Display;

use ryft_macros::Parameter;

use crate::differentiation::{Cotangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::control_flow::SelectCondition;
use crate::operations::{BooleanLike, Operation, OperationFormatter};
use crate::parameters::Parameter;
use crate::programs::{AtomId, ProgramError, Value};
use crate::tracing::AbstractTracingContext;
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

/// Factor payload used inside residualized linear programs.
///
/// Captured factors represent primal values that a linearized tangent or cotangent program closes over. A captured
/// factor is either a closed constant value that can be cloned directly, or a reference into the owning
/// [`Pushforward`](crate::tracing_v2::Pushforward)'s residual table that is instantiated later.
#[derive(Clone, Debug, Parameter)]
pub enum CapturedFactor<T: Type, V: Value<T>> {
    /// Closed constant factor that is independent of primal inputs.
    Constant(V),

    /// Reference to a primal residual saved by the owning [`Pushforward`](crate::tracing_v2::Pushforward).
    Reference {
        /// Zero-based residual index inside the owning [`Pushforward`](crate::tracing_v2::Pushforward).
        index: usize,

        /// Type metadata for the residual value.
        r#type: T,
    },
}

impl<T: Type, V: Value<T>> CapturedFactor<T, V> {
    /// Instantiates this factor into a concrete value using `residuals`.
    pub(crate) fn instantiate(&self, residuals: &[V]) -> Result<V, ProgramError> {
        match self {
            Self::Constant(value) => Ok(value.clone()),
            Self::Reference { index, .. } => {
                residuals.get(*index).cloned().ok_or(ProgramError::UnboundAtomId { id: AtomId::new(*index) }.into())
            }
        }
    }

    /// Returns this factor's residual index, if it references one.
    pub(crate) fn residual_index(&self) -> Option<usize> {
        match self {
            Self::Constant(_) => None,
            Self::Reference { index, .. } => Some(*index),
        }
    }

    /// Remaps this factor through a compacted residual-index table.
    pub(crate) fn remap_residuals(&self, mapping: &[Option<usize>]) -> Result<Self, ProgramError> {
        match self {
            Self::Constant(value) => Ok(Self::Constant(value.clone())),
            Self::Reference { index: old_index, r#type } => {
                let Some(Some(index)) = mapping.get(*old_index) else {
                    return Err(ProgramError::UnboundAtomId { id: AtomId::new(*old_index) }.into());
                };
                Ok(Self::Reference { index: *index, r#type: r#type.clone() })
            }
        }
    }
}

impl<T: Type, V: Value<T>> Display for CapturedFactor<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Constant(value) => Display::fmt(value, formatter),
            Self::Reference { index, .. } => write!(formatter, "captured[{index}]"),
        }
    }
}

impl<T: Type, V: Value<T>> Typed<T> for CapturedFactor<T, V> {
    #[inline]
    fn r#type(&self) -> Cow<'_, T> {
        match self {
            Self::Constant(value) => value.r#type(),
            Self::Reference { r#type, .. } => Cow::Borrowed(r#type),
        }
    }
}

impl<T: Type, V: Value<T>> Value<T> for CapturedFactor<T, V> {
    type InterpretationContext = V::InterpretationContext;

    #[inline]
    fn interpretation_context(&self) -> Option<V::InterpretationContext> {
        match self {
            Self::Constant(value) => value.interpretation_context(),
            // References carry no payload value to recover a context from. They are instantiated to concrete
            // constants before any interpretation, and callers scan their other operands for one that yields a
            // context (see the `SelectCondition`/`residual_value` impls, which reject references for the same reason).
            Self::Reference { .. } => None,
        }
    }
}

/// Scalar captured-condition factors carry the [`SelectOperation`](crate::operations::control_flow::SelectOperation)
/// condition as an in-band Boolean over a [`DataType`] value, so the linear select interprets them by decoding that
/// Boolean. References are residuals of the primal computation and must be instantiated before interpretation, so the
/// reference form errors here, matching
/// [`CustomVjpResidual::residual_value`](crate::tracing_v2::operations::CustomVjpResidual).
impl<V: Value<DataType> + BooleanLike> SelectCondition for CapturedFactor<DataType, V> {
    type Condition = bool;

    fn select_condition(&self) -> Result<bool, ProgramError> {
        match self {
            Self::Constant(value) => value.boolean(),
            Self::Reference { .. } => Err(ProgramError::Concretization {
                message: "captured select condition requires instantiated residuals".to_string(),
            }),
        }
    }
}

/// Canonical operation name for [`MaterializeCapturedFactorOperation`].
pub const MATERIALIZE_CAPTURED_FACTOR_OPERATION_NAME: &str = "materialize_captured_factor";

/// Nullary operation that materializes a captured factor as an ordinary program value.
///
/// Most linear operations keep primal values in their operation payloads: for example, a scale stores its factor, and
/// a captured-condition select stores its condition. Sometimes a captured value must instead become an atom in the
/// linear program so later instructions or nested control-flow bodies can use it as a normal operand. This operation
/// provides that adapter: it has no inputs, resolves its captured factor, and produces one output with the factor's
/// type.
///
/// This differs from [`ConstantOperation`](crate::operations::constants::ConstantOperation), which embeds an
/// already-materialized value as a literal program constant. A captured factor may instead be a
/// [`CapturedFactor::Reference`] into the owning pushforward's residual
/// table, so the actual value is not available until the pushforward is instantiated or interpreted.
///
/// The `Residual` variant of [`LinearArrayOperation`](crate::tracing_v2::LinearArrayOperation) wraps this operation so
/// array linear programs can inject such values. It is used by higher-order rules such as
/// [`WhileOperation`](crate::operations::control_flow::WhileOperation) when fusing or defactorizing nested
/// linearizations: if the residual is already available as a forwarded operand, defactorization can replace the
/// operation with that atom; otherwise interpretation resolves the factor from the owning residual environment.
///
/// This operation is not a linear map with respect to any operand. It is a constant injection into the linear IR, so
/// transposition rejects it directly.
#[derive(Clone, Debug)]
pub struct MaterializeCapturedFactorOperation<F> {
    /// Captured factor materialized as this operation's single output.
    factor: F,
}

impl<F> MaterializeCapturedFactorOperation<F> {
    /// Creates a new [`MaterializeCapturedFactorOperation`] capturing the provided factor.
    #[inline]
    pub fn new(factor: F) -> Self {
        Self { factor }
    }

    /// Returns the captured factor materialized as this operation's single output.
    #[inline]
    pub fn factor(&self) -> &F {
        &self.factor
    }
}

impl<F: Value<ArrayType>> Display for MaterializeCapturedFactorOperation<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<F: Value<ArrayType>> Operation<ArrayType> for MaterializeCapturedFactorOperation<F> {
    #[inline]
    fn name(&self) -> &'static str {
        MATERIALIZE_CAPTURED_FACTOR_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 0, TypeError);
        Ok(vec![self.factor.r#type().into_owned()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("factor", &self.factor))
    }
}

/// Transpose rule for captured-factor materialization (the `Residual` variant of
/// [`LinearArrayOperation`](crate::tracing_v2::LinearArrayOperation)). The injected factor is a captured primal value
/// rather than a linear operand, so the operation is not a linear map and rejects transposition. This rule is only
/// reachable behind the while transpose error, which fires first.
impl<V: Value<ArrayType>, O: Operation<ArrayType>, F: Value<ArrayType>> TransposableOperation<ArrayType, V, O>
    for MaterializeCapturedFactorOperation<F>
{
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
        _input_types: &[&ArrayType],
        _output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: "captured-factor materialization is not a linear map and does not support transposition"
                .to_string(),
        })
    }
}
