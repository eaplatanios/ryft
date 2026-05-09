use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::ops::{Add, Mul, Neg, Sub};
use std::sync::Arc;

use thiserror::Error;

use crate::operations::arithmetic::Scale;
use crate::operations::constants::{One, OneLike, Zero, ZeroLike};
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameter, Parameterized};
use crate::tracing::engines::{Tracer, TracingContext};
use crate::tracing::transposition::LinearOperation;
use crate::tracing::{Traceable, TracingError, Value};
use crate::tracing_v2::differentiation::{Differentiable, JvpTracer};
use crate::tracing_v2::{DifferentiableEngine, DifferentiableOperation, DifferentiableTracingEngine};
use crate::types::{ArrayType, Type, TypeError, Typed};

use super::control_flow::ControlFlowValue;
use super::matrix::MatrixOps;
use super::primitive::{ArrayOperation, LinearArrayOperation};
use super::reshape::ReshapeOps;
/// Error type for rule-based custom staged operations.
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CustomOperationError {
    /// Error returned when a custom primitive is used by a transform without registering the
    /// required rule.
    #[error("custom primitive '{op}' does not provide a '{transform}' rule")]
    MissingRule { op: &'static str, transform: &'static str },
}

/// Trait that represents [`Operation`] carrier types that support/include [`CustomPrimitive`]. Backend-owned
/// closed [`Operation`] carrier types (such as [`ArrayOperation`](super::ArrayOperation), for example) implement this
/// trait so that generic transform code can stage [`CustomPrimitive`] without knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsCustom<T: PartialEq + Type, V: Traceable<T> + Parameter> {
    /// Constructs the carrier-specific representation of the custom-primitive [`Operation`].
    fn custom_operation(primitive: Arc<CustomPrimitive<T, V>>) -> Self;
}

/// Trait that represents [`Operation`] carrier types that support/include [`LinearCustomPrimitive`].
/// Backend-owned closed [`Operation`] carrier types (such as [`LinearArrayOperation`](super::LinearArrayOperation), for
/// example) implement this trait so that generic transform code can stage [`LinearCustomPrimitive`] without knowing
/// which carrier is in use.
#[doc(hidden)]
pub trait SupportsLinearCustom<T: PartialEq + Type, V: Traceable<T> + Parameter>: Sized {
    /// Constructs the carrier-specific representation of the linear custom-primitive [`Operation`].
    fn custom_operation(primitive: CustomPrimitive<T, V>) -> Result<Self, TracingError>;

    /// Constructs the carrier-specific representation of the shared linear custom-primitive [`Operation`].
    fn custom_arc_operation(primitive: Arc<CustomPrimitive<T, V>>) -> Result<Self, TracingError>;
}

/// Typed extension registry carried by one [`CustomPrimitive`].
///
/// The registry is how custom primitives attach optional transform-specific rules without forcing
/// the core [`CustomPrimitive`] struct to know about every possible backend or higher-order
/// transform ahead of time.
#[derive(Clone, Default)]
pub struct CustomPrimitiveExtensions<T: Type, V: Typed<T>> {
    /// Type-indexed extension entries carried by the custom primitive.
    entries: HashMap<TypeId, Arc<dyn Any>>,

    /// Phantom marker tying the registry to the primitive's abstract and concrete leaf types.
    _marker: std::marker::PhantomData<(T, V)>,
}

impl<T: Type, V: Traceable<T>> Debug for CustomPrimitiveExtensions<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("CustomPrimitiveExtensions").field("count", &self.entries.len()).finish()
    }
}

impl<T: Type, V: Traceable<T>> CustomPrimitiveExtensions<T, V> {
    /// Inserts one typed extension into the registry, replacing any previous extension of the same type.
    pub fn insert<E: 'static>(&mut self, extension: E) {
        self.entries.insert(TypeId::of::<E>(), Arc::new(extension));
    }

    /// Returns the registered extension of type `E`, if present.
    pub fn get<E: 'static>(&self) -> Option<&E> {
        self.entries.get(&TypeId::of::<E>()).and_then(|extension| extension.as_ref().downcast_ref::<E>())
    }
}

/// Engine-keyed wrapper for one forward-mode JVP rule stored inside [`CustomPrimitiveExtensions`].
///
/// Custom primitives now key JVP rules by the concrete engine type instead of the `(O, L)` carrier
/// family pair so the public differentiation surface stays fully engine-driven.
struct JvpRule<E>(Arc<dyn DifferentiableOperation<E>>)
where
    E: DifferentiableEngine<Type = ArrayType> + 'static,
    E::Value: Differentiable<ArrayType>;

impl<E> JvpRule<E>
where
    E: DifferentiableEngine<Type = ArrayType> + 'static,
    E::Value: Differentiable<ArrayType>,
{
    fn rule(&self) -> &dyn DifferentiableOperation<E> {
        self.0.as_ref()
    }
}

/// Rule for differentiating a custom primitive while its primals are already staged tracers.
#[doc(hidden)]
pub trait CustomTracedLinearizationRule<V: Value<ArrayType>, E>
where
    V: Differentiable<ArrayType>,
    E: DifferentiableTracingEngine<Type = ArrayType, Value = V, OperationCarrier = ArrayOperation<V, ArrayType>>
        + 'static,
{
    /// Applies the custom primitive's traced-linearization JVP rule.
    fn jvp_traced_linearization<'engine>(
        &self,
        context: &mut crate::tracing_v2::JvpContext<'_, TracingContext<'engine, E>>,
        inputs: &[JvpTracer<Tracer<'engine, E>, crate::tracing::AtomId>],
    ) -> Result<Vec<JvpTracer<Tracer<'engine, E>, crate::tracing::AtomId>>, TracingError>;
}

/// Engine-keyed wrapper for one traced-linearization rule stored inside [`CustomPrimitiveExtensions`].
struct TracedLinearizationRule<V: Value<ArrayType>, E>(Arc<dyn CustomTracedLinearizationRule<V, E>>)
where
    V: Differentiable<ArrayType>,
    E: DifferentiableTracingEngine<Type = ArrayType, Value = V, OperationCarrier = ArrayOperation<V, ArrayType>>
        + 'static;

impl<V: Value<ArrayType>, E> TracedLinearizationRule<V, E>
where
    V: Differentiable<ArrayType>,
    E: DifferentiableTracingEngine<Type = ArrayType, Value = V, OperationCarrier = ArrayOperation<V, ArrayType>>
        + 'static,
{
    fn rule(&self) -> &dyn CustomTracedLinearizationRule<V, E> {
        self.0.as_ref()
    }
}

/// Base operation contract wrapped by one [`CustomPrimitive`].
pub trait CustomBaseOperation<T: Type, V: Typed<T>>: Operation<T> + InterpretableOperation<T, V> {}

impl<Ty: Type, V: Traceable<Ty>, O: Operation<Ty> + InterpretableOperation<Ty, V>> CustomBaseOperation<Ty, V> for O {}

/// Rule-based registration object used by [`ArrayOperation::Custom`].
///
/// [`CustomPrimitive`] is the main extensibility seam for the operation system. The base op always
/// supplies shape metadata and eager interpretation; optional transform rules are registered using
/// the existing tracing traits directly:
///
/// - [`LinearOperation<ArrayType, V, LinearArrayOperation<V, ArrayType>>`] for reverse-mode transpose,
/// - [`DifferentiableOperation<E>`] for forward-mode JVP under engine `E`,
/// - [`CustomTracedLinearizationRule`] for JVPs whose primals are already staged tracers.
#[derive(Clone)]
pub struct CustomPrimitive<T: PartialEq + Type, V: Traceable<T> + Parameter> {
    /// Required base op providing abstract evaluation and eager interpretation.
    pub base: Arc<dyn CustomBaseOperation<T, V>>,

    /// Optional reverse-mode transpose rule for the primitive.
    pub transpose_rule: Option<Arc<dyn LinearOperation<T, V, LinearArrayOperation<V, T>>>>,

    /// Typed extension registry carrying backend- or transform-specific extra rules.
    pub extensions: CustomPrimitiveExtensions<T, V>,
}

impl<T: PartialEq + Type + 'static, V: Traceable<T> + Parameter + 'static> CustomPrimitive<T, V> {
    /// Creates one custom primitive from its required base operation.
    pub fn new<Base>(base: Base) -> Self
    where
        Base: Operation<T> + InterpretableOperation<T, V> + 'static,
    {
        Self {
            base: Arc::new(base),
            transpose_rule: None,
            extensions: CustomPrimitiveExtensions { entries: HashMap::new(), _marker: std::marker::PhantomData },
        }
    }

    /// Registers one transpose rule for reverse-mode differentiation.
    pub fn with_transpose_rule<Rule>(mut self, rule: Rule) -> Self
    where
        LinearArrayOperation<V, T>: Operation<T>,
        Rule: LinearOperation<T, V, LinearArrayOperation<V, T>> + 'static,
    {
        self.transpose_rule = Some(Arc::new(rule));
        self
    }

    /// Registers one engine-specific forward-mode JVP rule.
    pub fn with_jvp_rule_for<E, Rule>(mut self, rule: Rule) -> Self
    where
        E: DifferentiableEngine<Type = ArrayType, Value = V> + 'static,
        V: Differentiable<ArrayType>,
        Rule: DifferentiableOperation<E> + 'static,
    {
        self.extensions.insert(JvpRule::<E>(Arc::new(rule)));
        self
    }

    /// Registers one typed extension.
    pub fn with_extension<E: 'static>(mut self, extension: E) -> Self {
        self.extensions.insert(extension);
        self
    }

    /// Returns one linear-only wrapper for this primitive after verifying that it provides a transpose rule.
    pub fn into_linear(self) -> Result<LinearCustomPrimitive<T, V>, TracingError> {
        LinearCustomPrimitive::from_custom_primitive(Arc::new(self))
    }

    /// Clones this primitive into one linear-only wrapper after verifying that it provides a transpose rule.
    pub fn to_linear(&self) -> Result<LinearCustomPrimitive<T, V>, TracingError> {
        self.clone().into_linear()
    }

    pub(super) fn missing_rule(&self, transform: &'static str) -> CustomOperationError {
        CustomOperationError::MissingRule { op: self.base.name(), transform }
    }

    fn jvp_rule<E>(&self) -> Result<&dyn DifferentiableOperation<E>, TracingError>
    where
        E: DifferentiableEngine<Type = ArrayType, Value = V> + 'static,
        V: Differentiable<ArrayType>,
    {
        self.extensions
            .get::<JvpRule<E>>()
            .map(JvpRule::rule)
            .ok_or_else(|| TracingError::from(self.missing_rule("jvp")))
    }

    fn traced_linearization_rule<E>(&self) -> Result<&dyn CustomTracedLinearizationRule<V, E>, TracingError>
    where
        V: Value<ArrayType> + Differentiable<ArrayType>,
        E: DifferentiableTracingEngine<Type = ArrayType, Value = V, OperationCarrier = ArrayOperation<V, ArrayType>>
            + 'static,
    {
        self.extensions
            .get::<TracedLinearizationRule<V, E>>()
            .map(TracedLinearizationRule::rule)
            .ok_or_else(|| TracingError::from(self.missing_rule("traced linearization")))
    }
}

impl<V: Traceable<ArrayType> + Parameter + 'static> CustomPrimitive<ArrayType, V> {
    /// Registers one forward-mode JVP rule for the canonical core staged carriers.
    pub fn with_jvp_rule<E, Rule>(self, rule: Rule) -> Self
    where
        E: DifferentiableEngine<
                Type = ArrayType,
                Value = V,
                Tangent = V,
                LinearOperationCarrier = LinearArrayOperation<V, ArrayType>,
            > + 'static,
        V: Differentiable<ArrayType, Tangent = V>
            + Add<Output = V>
            + Sub<Output = V>
            + Mul<Output = V>
            + Neg<Output = V>
            + Scale<Output = V>
            + Zero<ArrayType>
            + One<ArrayType>
            + ZeroLike
            + OneLike
            + MatrixOps
            + ReshapeOps
            + ControlFlowValue,
        Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
        Rule: DifferentiableOperation<E> + 'static,
    {
        self.with_jvp_rule_for::<E, _>(rule)
    }

    /// Registers one traced-linearization rule for the canonical core staged carriers.
    #[doc(hidden)]
    pub fn with_traced_linearization_rule<E, Rule>(mut self, rule: Rule) -> Self
    where
        V: Value<ArrayType> + Differentiable<ArrayType>,
        E: DifferentiableTracingEngine<Type = ArrayType, Value = V, OperationCarrier = ArrayOperation<V, ArrayType>>
            + 'static,
        Rule: CustomTracedLinearizationRule<V, E> + 'static,
    {
        self.extensions.insert(TracedLinearizationRule::<V, E>(Arc::new(rule)));
        self
    }

    /// Registers one custom derivative rule for the canonical core staged carriers.
    ///
    /// This is a convenience wrapper for the common case where one rule type can provide both the
    /// eager forward-mode [`DifferentiableOperation`] rule and the nested traced-linearization rule.
    /// It is equivalent to calling [`Self::with_jvp_rule`] followed by
    /// [`Self::with_traced_linearization_rule`] with clones of the same rule.
    ///
    /// This does not register a transpose rule for treating the custom primitive itself as a linear
    /// operation. Use [`Self::with_transpose_rule`] when a custom primitive must appear directly in a
    /// transposed linear program.
    pub fn with_derivative_rule<E, Rule>(self, rule: Rule) -> Self
    where
        E: DifferentiableEngine<
                Type = ArrayType,
                Value = V,
                Tangent = V,
                LinearOperationCarrier = LinearArrayOperation<V, ArrayType>,
            > + DifferentiableTracingEngine<Type = ArrayType, Value = V, OperationCarrier = ArrayOperation<V, ArrayType>>
            + 'static,
        V: Value<ArrayType>
            + Differentiable<ArrayType, Tangent = V>
            + Add<Output = V>
            + Sub<Output = V>
            + Mul<Output = V>
            + Neg<Output = V>
            + Scale<Output = V>
            + Zero<ArrayType>
            + One<ArrayType>
            + ZeroLike
            + OneLike
            + MatrixOps
            + ReshapeOps
            + ControlFlowValue,
        Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
        Rule: Clone + DifferentiableOperation<E> + CustomTracedLinearizationRule<V, E> + 'static,
    {
        self.with_jvp_rule::<E, _>(rule.clone()).with_traced_linearization_rule::<E, _>(rule)
    }
}

impl<T: PartialEq + Type, V: Traceable<T> + Parameter> Debug for CustomPrimitive<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self.base.as_ref(), formatter)
    }
}

impl<T: PartialEq + Type, V: Traceable<T> + Parameter> Display for CustomPrimitive<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.base.name())
    }
}

impl<T: PartialEq + Type, V: Traceable<T> + Parameter> Operation<T> for CustomPrimitive<T, V> {
    #[inline]
    fn name(&self) -> &'static str {
        self.base.name()
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        self.base.infer_output_types(input_types)
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        self.base.render(formatter, indentation)
    }
}

impl<T: PartialEq + Type, V: Traceable<T> + Parameter> InterpretableOperation<T, V> for CustomPrimitive<T, V> {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        self.base.interpret(inputs)
    }
}

impl<T: PartialEq + Type + 'static, V: Traceable<T> + Parameter + 'static>
    LinearOperation<T, V, LinearArrayOperation<V, T>> for CustomPrimitive<T, V>
where
    LinearArrayOperation<V, T>: Operation<T>,
{
    fn transpose(
        &self,
        context: &mut crate::tracing::transposition::TranspositionContext<T, V, LinearArrayOperation<V, T>>,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        self.transpose_rule
            .as_deref()
            .ok_or_else(|| TracingError::from(self.missing_rule("transpose")))?
            .transpose(context, output_cotangents)
    }
}

impl<V, E> DifferentiableOperation<E> for CustomPrimitive<ArrayType, V>
where
    V: Differentiable<ArrayType> + 'static,
    E: DifferentiableEngine<Type = ArrayType, Value = V> + 'static,
{
    fn jvp(
        &self,
        context: &mut crate::tracing_v2::JvpContext<'_, E>,
        inputs: &[JvpTracer<V, crate::tracing::AtomId>],
    ) -> Result<Vec<JvpTracer<V, crate::tracing::AtomId>>, TracingError> {
        self.jvp_rule::<E>()?.jvp(context, inputs)
    }
}

/// JVP rule for `CustomPrimitive` under [`TracingContext`].
impl<'engine, V, EInner> DifferentiableOperation<crate::tracing::engines::TracingContext<'engine, EInner>>
    for CustomPrimitive<ArrayType, V>
where
    V: Value<ArrayType> + Differentiable<ArrayType> + 'static,
    EInner: DifferentiableTracingEngine<Type = ArrayType, Value = V, OperationCarrier = ArrayOperation<V, ArrayType>>
        + 'static,
{
    fn jvp(
        &self,
        context: &mut crate::tracing_v2::JvpContext<'_, crate::tracing::engines::TracingContext<'engine, EInner>>,
        inputs: &[JvpTracer<Tracer<'engine, EInner>, crate::tracing::AtomId>],
    ) -> Result<Vec<JvpTracer<Tracer<'engine, EInner>, crate::tracing::AtomId>>, TracingError> {
        self.traced_linearization_rule::<EInner>()?.jvp_traced_linearization(context, inputs)
    }
}

/// Linear-only wrapper around one [`CustomPrimitive`] that guarantees a transpose rule is present.
///
/// Linear programs cannot store an op unless reverse-mode transposition is known to exist. This
/// wrapper is the proof object that a custom primitive has satisfied that requirement.
#[derive(Clone)]
pub struct LinearCustomPrimitive<T: PartialEq + Type, V: Traceable<T> + Parameter> {
    /// Wrapped custom primitive known to provide a transpose rule.
    pub primitive: Arc<CustomPrimitive<T, V>>,
}

impl<T: PartialEq + Type + 'static, V: Traceable<T> + Parameter + 'static> LinearCustomPrimitive<T, V> {
    /// Creates one linear-only wrapper from a custom primitive that already provides a transpose rule.
    pub fn from_custom_primitive(primitive: Arc<CustomPrimitive<T, V>>) -> Result<Self, TracingError> {
        primitive
            .transpose_rule
            .as_ref()
            .ok_or_else(|| TracingError::from(primitive.missing_rule("transpose")))?;
        Ok(Self { primitive })
    }
}

impl<T: PartialEq + Type, V: Traceable<T> + Parameter> Debug for LinearCustomPrimitive<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self.primitive.as_ref(), formatter)
    }
}

impl<T: PartialEq + Type, V: Traceable<T> + Parameter> Display for LinearCustomPrimitive<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self.primitive.as_ref(), formatter)
    }
}

impl<T: PartialEq + Type, V: Traceable<T> + Parameter> Operation<T> for LinearCustomPrimitive<T, V> {
    #[inline]
    fn name(&self) -> &'static str {
        self.primitive.name()
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        self.primitive.infer_output_types(input_types)
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        self.primitive.render(formatter, indentation)
    }
}

impl<T: PartialEq + Type, V: Traceable<T> + Parameter> InterpretableOperation<T, V> for LinearCustomPrimitive<T, V> {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        self.primitive.interpret(inputs)
    }
}

impl<T: PartialEq + Type, V: Traceable<T> + Parameter> LinearOperation<T, V, LinearArrayOperation<V, T>>
    for LinearCustomPrimitive<T, V>
where
    LinearArrayOperation<V, T>: Operation<T>,
{
    fn transpose(
        &self,
        context: &mut crate::tracing::transposition::TranspositionContext<T, V, LinearArrayOperation<V, T>>,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        self.primitive
            .transpose_rule
            .as_deref()
            .expect("linear custom primitives must carry a transpose rule")
            .transpose(context, output_cotangents)
    }
}
