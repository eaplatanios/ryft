use half::{bf16, f16};
use thiserror::Error;

use crate::tracing::engines::{Engine, ScalarEngine, Tracer, TracingContext, TracingEngine};
use crate::tracing::{InterpretableOperation, TracingError};
use crate::tracing_v2::LinearOperation as LinearOperationTrait;
use crate::tracing_v2::operations::{
    SupportsAdd, SupportsNeg, SupportsScale, SupportsZero, TracedLinearizationCarrier,
};
use crate::tracing_v2::{Differentiable, LinearScalarOperation, ScalarOperation};
use crate::types::ArrayType;

/// Errors emitted by the differentiation helpers in [`crate::tracing_v2`].
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DifferentiationError {
    /// Reverse-mode gradient was requested for a function with an invalid number of output leaves.
    #[error("gradient output must have exactly {expected} leaf but got {got}")]
    InvalidGradientOutputLeafCount { expected: usize, got: usize },

    /// Reverse-mode gradient was requested for a non-scalar array output.
    #[error("gradient output must be a rank-0 scalar array but got {output_type}")]
    NonScalarGradientOutput { output_type: ArrayType },

    /// Traced forward-mode differentiation was invoked without any staged input leaves.
    #[error("traced jvp requires at least one input leaf to recover the tracing context")]
    MissingTracedJvpInputLeaves,

    /// Traced reverse-mode differentiation was invoked without any staged input leaves.
    #[error("traced reverse-mode requires at least one input leaf to recover the tracing context")]
    MissingTracedReverseModeInputLeaves,

    /// Traced rematerialization was invoked without any staged input leaves.
    #[error("traced rematerialize requires at least one input leaf to recover the tracing context")]
    MissingTracedRematerializeInputLeaves,

    /// Linear rematerialization replay was invoked without any tangent leaves.
    #[error("linear rematerialize replay requires at least one tangent leaf to recover the tracing context")]
    MissingLinearRematerializeReplayTangentLeaves,

    /// Linear rematerialization transpose was invoked without any output cotangent leaves.
    #[error(
        "linear rematerialize transpose requires at least one output cotangent leaf to recover the tracing context"
    )]
    MissingLinearRematerializeTransposeCotangentLeaves,

    /// Dense Jacobian materialization produced an unexpected number of rows.
    #[error("invalid Jacobian row count; expected {expected} but got {got}")]
    InvalidJacobianRowCount { expected: usize, got: usize },

    /// Dense Jacobian materialization produced a row with an unexpected width.
    #[error("invalid Jacobian row width; expected {expected} but got {got}")]
    InvalidJacobianRowWidth { expected: usize, got: usize },

    /// Dense Jacobian materialization produced an unexpected number of columns.
    #[error("invalid Jacobian column count; expected {expected} but got {got}")]
    InvalidJacobianColumnCount { expected: usize, got: usize },

    /// Dense Jacobian materialization produced a column with an unexpected height.
    #[error("invalid Jacobian column height; expected {expected} but got {got}")]
    InvalidJacobianColumnHeight { expected: usize, got: usize },
}

/// Optional extension for tracing engines that support differentiation inside an active trace.
///
/// Plain tracing engines do not need to choose any linear carrier. This trait is the additional
/// contract required when a [`TracingContext`](crate::tracing::engines::TracingContext) itself needs to act
/// as a differentiable engine: tangent and cotangent programs then operate on
/// [`Tracer`] values, so the underlying tracing engine must select a linear operation carrier for
/// those traced leaves.
pub trait DifferentiableTracingEngine: TracingEngine {
    /// Linear operation carrier used for tangent and cotangent programs over traced values.
    type LinearOperation<'engine>: Clone
        + LinearOperationTrait<Self::Type, Tracer<'engine, Self>, Self::LinearOperation<'engine>>
        + SupportsAdd<Self::Type, Tracer<'engine, Self>>
        + SupportsNeg<Self::Type, Tracer<'engine, Self>>
        + SupportsScale<Self::Type, Tracer<'engine, Self>>
        + SupportsZero<Self::Type, Tracer<'engine, Self>>
    where
        Self: 'engine;
}

/// Extension of [`Engine`] for backends that support automatic differentiation.
///
/// Engines that only need ordinary tracing implement [`TracingEngine`] without this extension. AD
/// transforms such as [`grad`](crate::tracing_v2::grad), [`jvp`](crate::tracing_v2::jvp), and
/// [`vjp`](crate::tracing_v2::vjp) require this trait so non-differentiable backends do not need to
/// define fake tangent carriers.
///
/// Differentiated closures are traced through [`DifferentiableOperationTracingEngine`], whose
/// [`TracingEngine::Operation`] is [`DifferentiableEngine::DifferentiableOperation`]. That keeps
/// ordinary tracing free to use a wider operation carrier while making differentiation reject
/// unsupported operations at type-check time when the differentiation carrier omits them.
pub trait DifferentiableEngine: Engine {
    /// Staged operation type selected by this engine for tracing differentiable primal programs.
    type DifferentiableOperation: Clone + InterpretableOperation<Self::Type, Self::Value>;

    /// Linear staged operation type selected by this engine for tangent and cotangent programs.
    ///
    /// Linear programs produced by [`jvp_program`](crate::tracing_v2::jvp_program),
    /// [`vjp`](crate::tracing_v2::vjp), and related transforms store this carrier.
    type LinearOperation: Clone
        + LinearOperationTrait<Self::Type, Self::Value, Self::LinearOperation>
        + SupportsAdd<Self::Type, Self::Value>
        + SupportsNeg<Self::Type, Self::Value>
        + SupportsScale<Self::Type, Self::Value>;
}

/// Transparent tracing view used while tracing differentiable primal programs.
///
/// Automatic-differentiation transforms need to stage the user's primal closure with
/// [`DifferentiableEngine::DifferentiableOperation`] rather than the ordinary
/// [`TracingEngine::Operation`] selected by the backend. Those carriers may intentionally differ:
/// an engine can support a broad ordinary tracing universe while exposing a narrower
/// differentiable carrier whose variants all have differentiation rules. This adapter is the small
/// bridge between those two contracts.
///
/// [`DifferentiableOperationTracingEngine::new`] reborrows an `E: DifferentiableEngine` as a
/// [`TracingEngine`] without allocation or ownership. AD entry points construct this view at trace
/// boundaries such as [`jvp_program`](crate::tracing_v2::jvp_program),
/// [`vjp`](crate::tracing_v2::vjp), and [`grad`](crate::tracing_v2::grad), pass it immediately to
/// ordinary tracing helpers, and keep backend implementations centered on their real engine type.
/// User-facing ordinary tracing should keep using the backend's own [`TracingEngine`]
/// implementation; traced tangent and cotangent programs are selected separately through
/// [`DifferentiableTracingEngine`].
///
/// This type is public today because the public AD closure bounds still mention
/// `Tracer<'engine, DifferentiableOperationTracingEngine<E>>`. Once those APIs hide the concrete
/// active tracer carrier, this adapter can become a `pub(crate)` implementation detail.
#[repr(transparent)]
pub struct DifferentiableOperationTracingEngine<E: DifferentiableEngine + ?Sized> {
    /// Engine viewed through its differentiable operation carrier.
    engine: E,
}

impl<E: DifferentiableEngine + ?Sized> DifferentiableOperationTracingEngine<E> {
    /// Reborrows `engine` as a differentiable operation tracing view.
    #[inline]
    pub const fn new(engine: &E) -> &Self {
        // SAFETY: `DifferentiableOperationTracingEngine<E>` is `repr(transparent)` over `E` and adds no
        // fields, so references to `E` and references to this view have identical layout.
        unsafe { &*(std::ptr::from_ref(engine) as *const Self) }
    }

    /// Returns the wrapped engine.
    #[inline]
    pub const fn inner(&self) -> &E {
        // SAFETY: `DifferentiableOperationTracingEngine<E>` is `repr(transparent)` over `E` and adds no
        // fields, so references to this view and references to `E` have identical layout.
        unsafe { &*(std::ptr::from_ref(self) as *const E) }
    }
}

impl<E: DifferentiableEngine + ?Sized> std::fmt::Debug for DifferentiableOperationTracingEngine<E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("DifferentiableOperationTracingEngine").finish_non_exhaustive()
    }
}

impl<E: DifferentiableEngine + ?Sized> Engine for DifferentiableOperationTracingEngine<E> {
    type Type = E::Type;
    type Value = E::Value;

    #[inline]
    fn zero(&self, r#type: &Self::Type) -> Result<Self::Value, TracingError> {
        self.inner().zero(r#type)
    }

    #[inline]
    fn one(&self, r#type: &Self::Type) -> Result<Self::Value, TracingError> {
        self.inner().one(r#type)
    }
}

impl<E: DifferentiableEngine + ?Sized> TracingEngine for DifferentiableOperationTracingEngine<E> {
    type Operation = E::DifferentiableOperation;
}

impl<E: DifferentiableEngine + ?Sized> DifferentiableEngine for DifferentiableOperationTracingEngine<E> {
    type DifferentiableOperation = E::DifferentiableOperation;
    type LinearOperation = E::LinearOperation;
}

impl<'engine, E> DifferentiableEngine for TracingContext<'engine, E>
where
    E: DifferentiableTracingEngine + ?Sized,
    E::Value: Differentiable<E::Type, Tangent = E::Value>,
    E::Operation: TracedLinearizationCarrier<E::Type, E::Value>,
    crate::tracing_v2::operations::AddOperation: InterpretableOperation<E::Type, Tracer<'engine, E>>,
{
    type DifferentiableOperation = crate::tracing_v2::operations::AddOperation;
    type LinearOperation = E::LinearOperation<'engine>;
}

macro_rules! impl_differentiable_engine_for_scalar {
    ($ty:ty) => {
        impl DifferentiableEngine for ScalarEngine<$ty> {
            type DifferentiableOperation = ScalarOperation<$ty>;
            type LinearOperation = LinearScalarOperation<$ty>;
        }

        impl DifferentiableTracingEngine for ScalarEngine<$ty> {
            type LinearOperation<'engine>
                = LinearScalarOperation<Tracer<'engine, Self>>
            where
                Self: 'engine;
        }
    };
}

impl_differentiable_engine_for_scalar!(bf16);
impl_differentiable_engine_for_scalar!(f16);
impl_differentiable_engine_for_scalar!(f32);
impl_differentiable_engine_for_scalar!(f64);
