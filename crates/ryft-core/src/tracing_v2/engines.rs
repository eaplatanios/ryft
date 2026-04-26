use std::marker::PhantomData;

use crate::{
    parameters::Parameter,
    tracing::{InterpretableOperation, Operation, Traceable, TracingError},
    tracing_v2::{
        LinearOperation as LinearOperationTrait, LinearPrimitiveOperation, PrimitiveOperation,
        jit::Tracer,
        operations::{SupportsAdd, SupportsNeg, SupportsScale, SupportsZero},
    },
    types::{ArrayType, Type},
};

/// Synthesizes concrete leaf values from abstract type metadata.
///
/// [`Engine`] is the backend token threaded through the public `tracing_v2` transforms. It has
/// one narrow job: synthesize representative zero and one values from abstract metadata when a
/// transform needs an exemplar but only knows a leaf's type. That responsibility is what lets
/// higher-order transforms stay generic. Linearization,
/// reverse-mode transposition, and rematerialization all occasionally need to rebuild a value from
/// shape/type information alone; [`Engine`] is the narrow seam where backend-specific knowledge
/// enters the otherwise backend-agnostic transform code.
///
/// Per-instruction evaluation stays outside this trait: replay and abstract-eval continue to go
/// straight through [`crate::tracing::InterpretableOperation`] and [`crate::tracing::Operation`]
/// so the common fast path never needs an extra dispatch layer.
///
/// Engines are passed by shared reference to user-facing transforms. Implementations should be
/// cheap to clone (the common case is a [`Copy`] zero-sized type) and must return values whose type
/// metadata agrees with the input descriptor.
pub trait Engine {
    /// Abstract type metadata interpreted by this engine.
    ///
    /// This is the descriptor carried by staged atoms and used during abstract evaluation. For the
    /// default core pipeline it is usually [`ArrayType`](crate::types::ArrayType), but the trait is
    /// generic so backends can substitute a richer metadata type if needed.
    type Type: Type + Parameter;

    /// Concrete leaf value produced by this engine.
    ///
    /// The value is what program replay and eager transforms actually operate on. In other words,
    /// [`Engine::Type`] is the abstract description used while staging, while [`Engine::Value`] is
    /// the runtime leaf that inhabits traced programs once they are executed.
    type Value: Traceable<Self::Type>;

    /// Returns the additive-identity value corresponding to the provided type metadata.
    ///
    /// Transforms use this when they need a representative value for a leaf without having a
    /// concrete witness available, for example when replaying a staged program from retained input
    /// types or constructing zero cotangents in a transposed linear program.
    fn zero(&self, r#type: &Self::Type) -> Result<Self::Value, TracingError>;

    /// Returns the multiplicative-identity value corresponding to the provided type metadata.
    ///
    /// This is used less frequently than [`Engine::zero`] but plays the same architectural role:
    /// it lets traced code materialize identity seeds without depending on an existing exemplar.
    fn one(&self, r#type: &Self::Type) -> Result<Self::Value, TracingError>;
}

/// Engine capability for selecting a staged operation carrier.
///
/// [`StagingEngine`] extends [`Engine`] with the closed operation carrier that the paired
/// [`ProgramBuilder`](crate::tracing::ProgramBuilder) stores. This keeps carrier selection on the
/// engine value that is actually threaded through tracers instead of splitting it across a separate
/// generic parameter.
pub trait StagingEngine: Engine {
    /// Staged operation type selected by this staging engine.
    type Operation: Clone + Operation<Self::Type>;
}

/// Optional extension for staging engines that support differentiation inside an active trace.
///
/// Plain staging engines do not need to choose any linear carrier. This trait is the additional
/// contract required when a [`TracingEngine`](crate::tracing_v2::TracingEngine) itself needs to act
/// as a differentiable engine: tangent and cotangent programs then operate on
/// [`Tracer`] values, so the underlying staging engine must select a linear operation carrier for
/// those traced leaves.
pub trait DifferentiableStagingEngine: StagingEngine {
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
/// Engines that only need ordinary tracing implement [`StagingEngine`] without this extension. AD
/// transforms such as [`grad`](crate::tracing_v2::grad), [`jvp`](crate::tracing_v2::jvp), and
/// [`vjp`](crate::tracing_v2::vjp) require this trait so non-differentiable backends do not need to
/// define fake tangent carriers.
///
/// Differentiated closures are traced through [`DifferentiationStagingEngine`], whose
/// [`StagingEngine::Operation`] is [`DifferentiableEngine::DifferentiableOperation`]. That keeps
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

/// Active tracing view used while staging differentiable primal programs.
///
/// This transparent view selects [`DifferentiableEngine::DifferentiableOperation`] as the active
/// staged operation carrier for an existing [`DifferentiableEngine`]. It does not store its own
/// reference or state: [`DifferentiationStagingEngine::new`] reborrows an engine as this view, so a
/// [`TracingEngine`](crate::tracing_v2::jit::TracingEngine) can keep using ordinary engine references
/// without allocating or owning a wrapper.
#[repr(transparent)]
pub struct DifferentiationStagingEngine<E: DifferentiableEngine + ?Sized> {
    /// Engine viewed through the differentiation tracing carrier.
    engine: E,
}

impl<E: DifferentiableEngine + ?Sized> DifferentiationStagingEngine<E> {
    /// Reborrows `engine` as a differentiation tracing view.
    #[inline]
    pub const fn new(engine: &E) -> &Self {
        // SAFETY: `DifferentiationStagingEngine<E>` is `repr(transparent)` over `E` and adds no
        // fields, so references to `E` and references to this view have identical layout.
        unsafe { &*(std::ptr::from_ref(engine) as *const Self) }
    }

    /// Returns the wrapped differentiation engine.
    #[inline]
    pub const fn inner(&self) -> &E {
        // SAFETY: `DifferentiationStagingEngine<E>` is `repr(transparent)` over `E` and adds no
        // fields, so references to this view and references to `E` have identical layout.
        unsafe { &*(std::ptr::from_ref(self) as *const E) }
    }
}

impl<E: DifferentiableEngine + ?Sized> std::fmt::Debug for DifferentiationStagingEngine<E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("DifferentiationStagingEngine").finish_non_exhaustive()
    }
}

impl<E: DifferentiableEngine + ?Sized> Engine for DifferentiationStagingEngine<E> {
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

impl<E: DifferentiableEngine + ?Sized> StagingEngine for DifferentiationStagingEngine<E> {
    type Operation = E::DifferentiableOperation;
}

impl<E: DifferentiableEngine + ?Sized> DifferentiableEngine for DifferentiationStagingEngine<E> {
    type DifferentiableOperation = E::DifferentiableOperation;
    type LinearOperation = E::LinearOperation;
}

/// Stateless engine that synthesizes scalar-compatible values from [`ArrayType`] metadata.
///
/// [`ArrayScalarEngine<V>`] is the "minimal backend" used throughout tests and scalar-only
/// examples. It demonstrates the intended role of an [`Engine`] in the smallest possible form:
/// there is no device handle, no mesh state, and no backend registry, just the choice of the
/// built-in primitive carriers plus metadata-driven construction of scalar zeros and ones.
///
/// The engine ignores most of the supplied [`ArrayType`] metadata because scalar leaves have a
/// single canonical runtime representation. That makes it a good teaching example for the rest of
/// the tracing stack: if a transform works against [`ArrayScalarEngine`], the same code path can be
/// reused by richer engines that need sharding, device, or runtime context.
#[derive(Clone, Copy, Debug, Default)]
pub struct ArrayScalarEngine<V> {
    /// Phantom marker that ties the zero-sized engine to its scalar leaf type.
    marker: PhantomData<fn() -> V>,
}

impl<V> ArrayScalarEngine<V> {
    /// Returns a new [`ArrayScalarEngine<V>`].
    ///
    /// This is a no-op at runtime because the engine is zero-sized; the method mainly exists to
    /// give examples and tests an explicit, readable backend token.
    #[inline]
    pub const fn new() -> Self {
        Self { marker: PhantomData }
    }
}

macro_rules! impl_engine_for_array_scalar_engine {
    ($ty:ty, $zero:expr, $one:expr) => {
        impl Engine for ArrayScalarEngine<$ty> {
            type Type = ArrayType;
            type Value = $ty;

            #[inline]
            fn zero(&self, _type: &ArrayType) -> Result<$ty, TracingError> {
                Ok($zero)
            }

            #[inline]
            fn one(&self, _type: &ArrayType) -> Result<$ty, TracingError> {
                Ok($one)
            }
        }

        impl StagingEngine for ArrayScalarEngine<$ty> {
            type Operation = PrimitiveOperation<$ty>;
        }

        impl DifferentiableEngine for ArrayScalarEngine<$ty> {
            type DifferentiableOperation = PrimitiveOperation<$ty>;
            type LinearOperation = LinearPrimitiveOperation<$ty>;
        }

        impl DifferentiableStagingEngine for ArrayScalarEngine<$ty> {
            type LinearOperation<'engine>
                = LinearPrimitiveOperation<Tracer<'engine, Self>>
            where
                Self: 'engine;
        }
    };
}

impl_engine_for_array_scalar_engine!(f32, 0.0, 1.0);
impl_engine_for_array_scalar_engine!(f64, 0.0, 1.0);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DataType;

    #[test]
    fn test_array_scalar_engine_is_zero_sized() {
        assert_eq!(size_of::<ArrayScalarEngine<f64>>(), 0);
        assert_eq!(size_of::<ArrayScalarEngine<f32>>(), 0);
    }

    #[test]
    fn test_array_scalar_engine_produces_canonical_zero_and_one() {
        let engine = ArrayScalarEngine::<f64>::new();
        let r#type = ArrayType::scalar(DataType::F64);
        assert_eq!(Engine::zero(&engine, &r#type), Ok(0.0));
        assert_eq!(Engine::one(&engine, &r#type), Ok(1.0));
    }
}
