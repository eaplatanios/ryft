use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::sync::Arc;

use thiserror::Error;

use crate::parameters::Parameter;
use crate::tracing::{Traceable, TracingError, Value};
use crate::tracing_v2::engines::Tracer;
use crate::tracing_v2::forward::{Differentiable, JvpTracer};
use crate::tracing_v2::{DifferentiableEngine, DifferentiableTracingEngine, TracingContext};
use crate::types::{ArrayType, Type, TypeError, Typed};

use super::primitive::{LinearPrimitiveOperation, PrimitiveOperation};
use super::{DifferentiableOperation, InterpretableOperation, LinearOperation, Operation};

/// Error type for rule-based custom staged operations.
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CustomOperationError {
    /// Error returned when a custom primitive is used by a transform without registering the
    /// required rule.
    #[error("custom primitive '{op}' does not provide a '{transform}' rule")]
    MissingRule { op: &'static str, transform: &'static str },
}

/// Hidden carrier capability for staging the custom-primitive escape hatch.
#[doc(hidden)]
pub trait SupportsCustom<T: Type + PartialEq, V: Traceable<T> + Parameter>: Clone {
    /// Constructs the carrier-specific representation of one custom primitive.
    fn custom_operation(primitive: Arc<CustomPrimitive<T, V>>) -> Self;
}

/// Hidden carrier capability for staging the custom-primitive escape hatch in linear programs.
#[doc(hidden)]
pub trait SupportsLinearCustom<T: Type + PartialEq, V: Traceable<T> + Parameter>: Clone {
    /// Constructs the carrier-specific representation of one custom primitive in the linear universe.
    fn custom_operation(primitive: CustomPrimitive<T, V>) -> Result<Self, TracingError>;

    /// Constructs the carrier-specific representation of one shared custom primitive in the linear universe.
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
struct JvpRule<E: DifferentiableEngine<Type = ArrayType> + 'static>(Arc<dyn DifferentiableOperation<E>>)
where
    E::Value: Differentiable<ArrayType, Tangent = E::Value>;

impl<E: DifferentiableEngine<Type = ArrayType> + 'static> JvpRule<E>
where
    E::Value: Differentiable<ArrayType, Tangent = E::Value>,
{
    fn rule(&self) -> &dyn DifferentiableOperation<E> {
        self.0.as_ref()
    }
}

/// Rule for differentiating a custom primitive while its primals are already staged tracers.
#[doc(hidden)]
pub trait CustomTracedLinearizationRule<
    V: Value<ArrayType> + Differentiable<ArrayType, Tangent = V>,
    E: DifferentiableTracingEngine<Type = ArrayType, Value = V, Operation = PrimitiveOperation<V>> + ?Sized + 'static,
>
{
    /// Applies the custom primitive's traced-linearization JVP rule.
    fn jvp_traced_linearization<'engine>(
        &self,
        engine: &TracingContext<'engine, E>,
        context: &mut crate::tracing_v2::JvpContext<
            '_,
            Tracer<'engine, E>,
            <E as crate::tracing_v2::DifferentiableTracingEngine>::LinearOperation<'engine>,
        >,
        inputs: &[JvpTracer<Tracer<'engine, E>, crate::tracing::AtomId>],
    ) -> Result<Vec<JvpTracer<Tracer<'engine, E>, crate::tracing::AtomId>>, TracingError>;
}

/// Engine-keyed wrapper for one traced-linearization rule stored inside [`CustomPrimitiveExtensions`].
struct TracedLinearizationRule<
    V: Value<ArrayType> + Differentiable<ArrayType, Tangent = V>,
    E: DifferentiableTracingEngine<Type = ArrayType, Value = V, Operation = PrimitiveOperation<V>> + ?Sized + 'static,
>(Arc<dyn CustomTracedLinearizationRule<V, E>>);

impl<
    V: Value<ArrayType> + Differentiable<ArrayType, Tangent = V>,
    E: DifferentiableTracingEngine<Type = ArrayType, Value = V, Operation = PrimitiveOperation<V>> + ?Sized + 'static,
> TracedLinearizationRule<V, E>
{
    fn rule(&self) -> &dyn CustomTracedLinearizationRule<V, E> {
        self.0.as_ref()
    }
}

trait CustomBaseOperation<T: Type, V: Typed<T>>: Operation<T> + InterpretableOperation<T, V> {}

impl<Ty: Type, V: Traceable<Ty>, O: Operation<Ty> + InterpretableOperation<Ty, V>> CustomBaseOperation<Ty, V> for O {}

/// Rule-based registration object used by [`PrimitiveOperation::Custom`].
///
/// [`CustomPrimitive`] is the main extensibility seam for the operation system. The base op always
/// supplies shape metadata and eager interpretation; optional transform rules are registered using
/// the existing tracing traits directly:
///
/// - [`LinearOperation<ArrayType, V>`] for reverse-mode transpose,
/// - [`DifferentiableOperation<E>`] for forward-mode JVP under engine `E`,
/// - [`CustomTracedLinearizationRule`] for JVPs whose primals are already staged tracers.
#[derive(Clone)]
pub struct CustomPrimitive<T: Type + PartialEq, V: Traceable<T> + Parameter> {
    /// Required base op providing abstract evaluation and eager interpretation.
    base: Arc<dyn CustomBaseOperation<T, V>>,

    /// Optional reverse-mode transpose rule for the primitive.
    transpose_rule: Option<Arc<dyn LinearOperation<T, V>>>,

    /// Typed extension registry carrying backend- or transform-specific extra rules.
    extensions: CustomPrimitiveExtensions<T, V>,
}

impl<T: Type + PartialEq + 'static, V: Traceable<T> + Parameter + 'static> CustomPrimitive<T, V> {
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
        Rule: LinearOperation<T, V> + 'static,
    {
        self.transpose_rule = Some(Arc::new(rule));
        self
    }

    /// Registers one engine-specific forward-mode JVP rule.
    pub fn with_jvp_rule_for<E, Rule>(mut self, rule: Rule) -> Self
    where
        E: DifferentiableEngine<Type = ArrayType, Value = V> + 'static,
        V: Differentiable<ArrayType, Tangent = V>,
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

    /// Returns the typed extension registry carried by this primitive.
    #[inline]
    pub fn extensions(&self) -> &CustomPrimitiveExtensions<T, V> {
        &self.extensions
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
        V: Differentiable<ArrayType, Tangent = V>,
    {
        self.extensions
            .get::<JvpRule<E>>()
            .map(JvpRule::rule)
            .ok_or_else(|| TracingError::from(self.missing_rule("jvp")))
    }

    fn traced_linearization_rule<E>(&self) -> Result<&dyn CustomTracedLinearizationRule<V, E>, TracingError>
    where
        V: Value<ArrayType> + Differentiable<ArrayType, Tangent = V>,
        E: DifferentiableTracingEngine<Type = ArrayType, Value = V, Operation = PrimitiveOperation<V>>
            + ?Sized
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
                DifferentiableOperation = PrimitiveOperation<V>,
                LinearOperation = LinearPrimitiveOperation<V>,
            > + 'static,
        V: Differentiable<ArrayType, Tangent = V>,
        Rule: DifferentiableOperation<E> + 'static,
    {
        self.with_jvp_rule_for::<E, _>(rule)
    }

    /// Registers one traced-linearization rule for the canonical core staged carriers.
    #[doc(hidden)]
    pub fn with_traced_linearization_rule<E, Rule>(mut self, rule: Rule) -> Self
    where
        V: Value<ArrayType> + Differentiable<ArrayType, Tangent = V>,
        E: DifferentiableTracingEngine<Type = ArrayType, Value = V, Operation = PrimitiveOperation<V>>
            + ?Sized
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
        V: Value<ArrayType> + Differentiable<ArrayType, Tangent = V>,
        E: DifferentiableEngine<
                Type = ArrayType,
                Value = V,
                DifferentiableOperation = PrimitiveOperation<V>,
                LinearOperation = LinearPrimitiveOperation<V>,
            > + DifferentiableTracingEngine<Type = ArrayType, Value = V, Operation = PrimitiveOperation<V>>
            + 'static,
        Rule: Clone + DifferentiableOperation<E> + CustomTracedLinearizationRule<V, E> + 'static,
    {
        self.with_jvp_rule::<E, _>(rule.clone()).with_traced_linearization_rule::<E, _>(rule)
    }
}

impl<T: Type + PartialEq, V: Traceable<T> + Parameter> Debug for CustomPrimitive<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self.base.as_ref(), formatter)
    }
}

impl<T: Type + PartialEq, V: Traceable<T> + Parameter> Display for CustomPrimitive<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.base.name())
    }
}

impl<T: Type + PartialEq, V: Traceable<T> + Parameter> Operation<T> for CustomPrimitive<T, V> {
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

impl<T: Type + PartialEq, V: Traceable<T> + Parameter> InterpretableOperation<T, V> for CustomPrimitive<T, V> {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        self.base.interpret(inputs)
    }
}

impl<T: Type + PartialEq + 'static, V: Traceable<T> + Parameter + 'static> LinearOperation<T, V>
    for CustomPrimitive<T, V>
where
    LinearPrimitiveOperation<V, T>: Operation<T>,
{
    fn transpose(
        &self,
        context: &mut crate::tracing_v2::operations::TranspositionContext<'_, T, V, LinearPrimitiveOperation<V, T>>,
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
    V: Differentiable<ArrayType, Tangent = V> + 'static,
    E: DifferentiableEngine<Type = ArrayType, Value = V> + 'static,
{
    fn jvp(
        &self,
        engine: &E,
        context: &mut crate::tracing_v2::JvpContext<'_, V, E::LinearOperation>,
        inputs: &[JvpTracer<V, crate::tracing::AtomId>],
    ) -> Result<Vec<JvpTracer<V, crate::tracing::AtomId>>, TracingError> {
        self.jvp_rule::<E>()?.jvp(engine, context, inputs)
    }
}

/// JVP rule for `CustomPrimitive` under [`TracingContext`].
impl<'engine, V, EInner> DifferentiableOperation<crate::tracing_v2::TracingContext<'engine, EInner>>
    for CustomPrimitive<ArrayType, V>
where
    V: Value<ArrayType> + Differentiable<ArrayType, Tangent = V> + 'static,
    EInner:
        DifferentiableTracingEngine<Type = ArrayType, Value = V, Operation = PrimitiveOperation<V>> + ?Sized + 'static,
{
    fn jvp(
        &self,
        _engine: &crate::tracing_v2::TracingContext<'engine, EInner>,
        context: &mut crate::tracing_v2::JvpContext<
            '_,
            Tracer<'engine, EInner>,
            <EInner as DifferentiableTracingEngine>::LinearOperation<'engine>,
        >,
        inputs: &[JvpTracer<Tracer<'engine, EInner>, crate::tracing::AtomId>],
    ) -> Result<Vec<JvpTracer<Tracer<'engine, EInner>, crate::tracing::AtomId>>, TracingError> {
        self.traced_linearization_rule::<EInner>()?.jvp_traced_linearization(_engine, context, inputs)
    }
}

/// Linear-only wrapper around one [`CustomPrimitive`] that guarantees a transpose rule is present.
///
/// Linear programs cannot store an op unless reverse-mode transposition is known to exist. This
/// wrapper is the proof object that a custom primitive has satisfied that requirement.
#[derive(Clone)]
pub struct LinearCustomPrimitive<T: Type + PartialEq, V: Traceable<T> + Parameter> {
    /// Wrapped custom primitive known to provide a transpose rule.
    primitive: Arc<CustomPrimitive<T, V>>,
}

impl<T: Type + PartialEq + 'static, V: Traceable<T> + Parameter + 'static> LinearCustomPrimitive<T, V> {
    /// Creates one linear-only wrapper from a custom primitive that already provides a transpose rule.
    pub fn from_custom_primitive(primitive: Arc<CustomPrimitive<T, V>>) -> Result<Self, TracingError> {
        primitive
            .transpose_rule
            .as_ref()
            .ok_or_else(|| TracingError::from(primitive.missing_rule("transpose")))?;
        Ok(Self { primitive })
    }

    /// Returns the wrapped custom primitive.
    #[inline]
    pub fn primitive(&self) -> &Arc<CustomPrimitive<T, V>> {
        &self.primitive
    }
}

impl<T: Type + PartialEq, V: Traceable<T> + Parameter> Debug for LinearCustomPrimitive<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self.primitive.as_ref(), formatter)
    }
}

impl<T: Type + PartialEq, V: Traceable<T> + Parameter> Display for LinearCustomPrimitive<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self.primitive.as_ref(), formatter)
    }
}

impl<T: Type + PartialEq, V: Traceable<T> + Parameter> Operation<T> for LinearCustomPrimitive<T, V> {
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

impl<T: Type + PartialEq, V: Traceable<T> + Parameter> InterpretableOperation<T, V> for LinearCustomPrimitive<T, V> {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        self.primitive.interpret(inputs)
    }
}

impl<T: Type + PartialEq, V: Traceable<T> + Parameter> LinearOperation<T, V> for LinearCustomPrimitive<T, V>
where
    LinearPrimitiveOperation<V, T>: Operation<T>,
{
    fn transpose(
        &self,
        context: &mut crate::tracing_v2::operations::TranspositionContext<'_, T, V, LinearPrimitiveOperation<V, T>>,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        self.primitive
            .transpose_rule
            .as_deref()
            .expect("linear custom primitives must carry a transpose rule")
            .transpose(context, output_cotangents)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::Arc;

    use pretty_assertions::assert_eq;

    use super::*;
    use crate::tracing::{Program, ProgramBuilder};
    use crate::tracing_v2::engines::{Engine, TracingEngine};
    use crate::tracing_v2::operations::TranspositionContext;
    use crate::tracing_v2::operations::constants::OneLike;
    use crate::tracing_v2::{
        DifferentiableEngine, DifferentiableTracingEngine, LinearPrimitiveOperation, PrimitiveOperation, Tracer, grad,
        jvp,
    };
    use crate::types::{ArrayType, DataType, Shape, Typed};

    #[derive(Copy, Clone)]
    struct ArrayScalarEngine;

    impl Engine for ArrayScalarEngine {
        type Type = ArrayType;
        type Value = f64;

        fn zero(&self, _type: &ArrayType) -> Result<f64, TracingError> {
            Ok(0.0)
        }

        fn one(&self, _type: &ArrayType) -> Result<f64, TracingError> {
            Ok(1.0)
        }
    }

    impl TracingEngine for ArrayScalarEngine {
        type Operation = PrimitiveOperation<f64>;
    }

    impl DifferentiableEngine for ArrayScalarEngine {
        type DifferentiableOperation = PrimitiveOperation<f64>;
        type LinearOperation = LinearPrimitiveOperation<f64>;
    }

    impl DifferentiableTracingEngine for ArrayScalarEngine {
        type LinearOperation<'engine>
            = LinearPrimitiveOperation<Tracer<'engine, Self>>
        where
            Self: 'engine;
    }

    /// Simple unary custom op used to exercise the rule-based custom primitive API.
    #[derive(Clone, Debug)]
    struct ShiftOp {
        amount: f64,
    }

    impl ShiftOp {
        /// Creates one shift op with the provided additive amount.
        fn new(amount: f64) -> Self {
            Self { amount }
        }
    }

    impl Display for ShiftOp {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "test_shift")
        }
    }

    impl Operation<ArrayType> for ShiftOp {
        fn name(&self) -> &'static str {
            "test_shift"
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
            if input_types.len() != 1 {
                return Err(TypeError {
                    message: format!("test_shift expected 1 input type but got {}", input_types.len()),
                });
            }
            Ok(vec![input_types[0].clone()])
        }
    }

    impl InterpretableOperation<ArrayType, f64> for ShiftOp {
        fn interpret(&self, inputs: &[f64]) -> Result<Vec<f64>, TracingError> {
            if inputs.len() != 1 {
                return Err(TracingError::InvalidInputCount { expected: 1, got: inputs.len() });
            }
            Ok(vec![inputs[0] + self.amount])
        }
    }

    impl LinearOperation<ArrayType, f64> for ShiftOp {
        fn transpose(
            &self,
            _context: &mut crate::tracing_v2::operations::TranspositionContext<
                '_,
                ArrayType,
                f64,
                LinearPrimitiveOperation<f64>,
            >,
            output_cotangents: &[Option<crate::tracing::AtomId>],
        ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
            if output_cotangents.len() != 1 {
                return Err(TracingError::InvalidInputCount { expected: 1, got: output_cotangents.len() });
            }
            Ok(vec![output_cotangents[0]])
        }
    }

    impl DifferentiableOperation<ArrayScalarEngine> for ShiftOp {
        fn jvp(
            &self,
            _engine: &ArrayScalarEngine,
            _context: &mut crate::tracing_v2::JvpContext<'_, f64, LinearPrimitiveOperation<f64>>,
            inputs: &[JvpTracer<f64, crate::tracing::AtomId>],
        ) -> Result<Vec<JvpTracer<f64, crate::tracing::AtomId>>, TracingError> {
            if inputs.len() != 1 {
                return Err(TracingError::InvalidInputCount { expected: 1, got: inputs.len() });
            }
            Ok(vec![JvpTracer { primal: inputs[0].primal + self.amount, tangent: inputs[0].tangent }])
        }
    }

    impl CustomTracedLinearizationRule<f64, ArrayScalarEngine> for ShiftOp {
        fn jvp_traced_linearization<'engine>(
            &self,
            _engine: &TracingContext<'engine, ArrayScalarEngine>,
            _context: &mut crate::tracing_v2::JvpContext<
                '_,
                Tracer<'engine, ArrayScalarEngine>,
                LinearPrimitiveOperation<Tracer<'engine, ArrayScalarEngine>>,
            >,
            inputs: &[JvpTracer<Tracer<'engine, ArrayScalarEngine>, crate::tracing::AtomId>],
        ) -> Result<Vec<JvpTracer<Tracer<'engine, ArrayScalarEngine>, crate::tracing::AtomId>>, TracingError> {
            if inputs.len() != 1 {
                return Err(TracingError::InvalidInputCount { expected: 1, got: inputs.len() });
            }
            let primal = apply_custom_traced_unary(
                inputs[0].primal.clone(),
                CustomPrimitive::<ArrayType, f64>::new(self.clone()),
            )?;
            Ok(vec![JvpTracer { primal, tangent: inputs[0].tangent }])
        }
    }

    /// Applies one unary custom primitive to one traced scalar.
    fn apply_custom_traced_unary<'engine, E>(
        input: Tracer<'engine, E>,
        primitive: CustomPrimitive<ArrayType, f64>,
    ) -> Result<Tracer<'engine, E>, TracingError>
    where
        E: TracingEngine<Type = ArrayType, Value = f64, Operation = PrimitiveOperation<f64>> + ?Sized,
    {
        let context = input.context.clone();
        Ok(context
            .trace(PrimitiveOperation::Custom(Arc::new(primitive)), std::slice::from_ref(&input))?
            .into_iter()
            .next()
            .expect("unary custom primitive should produce one output"))
    }

    /// Applies one unary custom primitive to one traced scalar and expects staging to succeed.
    fn stage_custom_traced_unary<'engine, E>(
        input: Tracer<'engine, E>,
        primitive: CustomPrimitive<ArrayType, f64>,
    ) -> Tracer<'engine, E>
    where
        E: TracingEngine<Type = ArrayType, Value = f64, Operation = PrimitiveOperation<f64>> + ?Sized,
    {
        apply_custom_traced_unary(input, primitive).expect("custom primitive staging should succeed")
    }

    /// Returns one scalar array type used by these custom-primitive tests.
    fn scalar_type() -> ArrayType {
        ArrayType::new(DataType::F64, Shape::scalar(), None, None).expect("scalar array types should be valid")
    }

    fn test_transposition_context(
        builder: Rc<RefCell<ProgramBuilder<ArrayType, f64, LinearPrimitiveOperation<f64>>>>,
    ) -> TranspositionContext<'static, ArrayType, f64, LinearPrimitiveOperation<f64>> {
        TranspositionContext::new(builder)
    }

    #[test]
    fn test_linear_custom_primitive_requires_transpose_rule_up_front() {
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0));

        assert!(matches!(
            primitive.into_linear(),
            Err(TracingError::CustomOperation(CustomOperationError::MissingRule {
                op: "test_shift",
                transform: "transpose",
            }))
        ));
    }

    #[test]
    fn test_custom_primitive_base_execution_replays_without_optional_rules() {
        let engine = ArrayScalarEngine;
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0));
        let (output, compiled): (f64, Program<ArrayType, f64, PrimitiveOperation<f64>, f64, f64>) = engine
            .interpret_and_trace(
                {
                    let primitive = primitive.clone();
                    move |x| Ok(stage_custom_traced_unary(x, primitive.clone()))
                },
                3.0f64,
            )
            .unwrap();

        assert_eq!(output, 5.0);
        assert_eq!(compiled.interpret(4.0f64), Ok(6.0));
    }

    #[test]
    fn test_custom_primitive_missing_transpose_rule_reports_targeted_error() {
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0));
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, f64, LinearPrimitiveOperation<f64>>::new()));
        let cotangent_atom = builder.borrow_mut().add_input(<f64 as Typed<ArrayType>>::r#type(&0.0f64).into_owned());
        let mut context = test_transposition_context(builder);

        assert!(matches!(
            primitive.transpose(&mut context, &[Some(cotangent_atom)]),
            Err(TracingError::CustomOperation(CustomOperationError::MissingRule {
                op: "test_shift",
                transform: "transpose",
            }))
        ));
    }

    #[test]
    fn test_custom_primitive_missing_jvp_rule_reports_targeted_error() {
        let engine = ArrayScalarEngine;
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0));
        let result: Result<(f64, f64), TracingError> = jvp(
            &engine,
            {
                let primitive = primitive.clone();
                move |x| stage_custom_traced_unary(x, primitive.clone())
            },
            3.0f64,
            1.0f64,
        );

        assert_eq!(
            result,
            Err(TracingError::CustomOperation(CustomOperationError::MissingRule {
                op: "test_shift",
                transform: "jvp",
            })),
        );
    }

    #[test]
    fn test_custom_primitive_missing_traced_linearization_rule_reports_targeted_error() {
        let engine = ArrayScalarEngine;
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0))
            .with_jvp_rule::<ArrayScalarEngine, _>(ShiftOp::new(2.0));
        let result: Result<(f64, Program<ArrayType, f64, PrimitiveOperation<f64>, f64, f64>), TracingError> = engine
            .interpret_and_trace(
                {
                    let primitive = primitive.clone();
                    move |x: Tracer<ArrayScalarEngine>| {
                        let (primal, tangent) = jvp(
                            &engine,
                            {
                                let primitive = primitive.clone();
                                move |inner| stage_custom_traced_unary(inner, primitive.clone())
                            },
                            x.clone(),
                            x.one_like(),
                        )?;
                        Ok(primal + tangent)
                    }
                },
                3.0f64,
            );

        assert!(matches!(
            result,
            Err(TracingError::CustomOperation(CustomOperationError::MissingRule {
                op: "test_shift",
                transform: "traced linearization",
            }))
        ));
    }

    #[test]
    fn test_custom_primitive_jvp_rule_participates_in_grad_and_traced_linearization() {
        let engine = ArrayScalarEngine;
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0))
            .with_derivative_rule::<ArrayScalarEngine, _>(ShiftOp::new(2.0));

        assert_eq!(
            grad(
                &engine,
                {
                    let primitive = primitive.clone();
                    move |x| stage_custom_traced_unary(x, primitive.clone())
                },
                3.0f64,
            ),
            Ok(1.0f64),
        );

        let (output, compiled): (f64, Program<ArrayType, f64, PrimitiveOperation<f64>, f64, f64>) = engine
            .interpret_and_trace(
                {
                    let primitive = primitive.clone();
                    move |x: Tracer<ArrayScalarEngine>| {
                        let (primal, tangent) = jvp(
                            &engine,
                            {
                                let primitive = primitive.clone();
                                move |inner| stage_custom_traced_unary(inner, primitive.clone())
                            },
                            x.clone(),
                            x.one_like(),
                        )?;
                        Ok(primal + tangent)
                    }
                },
                3.0f64,
            )
            .unwrap();

        assert_eq!(output, 6.0);
        assert_eq!(compiled.interpret(4.0f64), Ok(7.0));
    }

    #[test]
    fn test_custom_primitive_abstract_eval_uses_the_registered_base_op() {
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0));

        assert_eq!(primitive.infer_output_types(&[scalar_type()]), Ok(vec![scalar_type()]));
    }
}
