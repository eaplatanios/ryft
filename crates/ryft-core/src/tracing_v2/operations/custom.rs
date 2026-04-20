//! Custom-primitive subsystem for `tracing_v2`.
//!
//! [`CustomPrimitive`] is the rule-based escape hatch that lets user code (or downstream backends)
//! introduce new staged operations without touching the closed
//! [`PrimitiveOperation`](super::PrimitiveOperation) carrier. Each primitive carries a required base op for
//! shape metadata and concrete interpretation, plus optional rules for the various transforms
//! ([`LinearOperation`] for transpose, [`DifferentiableOperation`] for JVP, [`VectorizableOperation`] for
//! batching, and an optional linearized-JIT replay rule for nested transforms).
//!
//! [`LinearCustomPrimitive`] is the linear-only sibling that guarantees a transpose rule is
//! present, so it can be stored in linear programs without runtime checks.

use std::{
    any::{Any, TypeId},
    collections::HashMap,
    fmt::{Debug, Display},
    sync::Arc,
};

use crate::{
    batching::Batch,
    parameters::Parameter,
    tracing_v2::{
        AtomId, Traceable, TracingError, Value, ZeroLike,
        engine::Engine,
        forward::JvpTracer,
        jit::Tracer,
        linear::{LinearTerm, Linearized},
    },
    types::{ArrayType, Type, Typed},
};

use super::{
    DifferentiableOperation, InterpretableOperation, LinearAddOperation, LinearNegOperation, LinearOperation,
    LinearScaleOperation, Operation, VectorizableOperation,
    primitive::{LinearPrimitiveOperation, PrimitiveOperation},
};

/// Hidden staging trait for the custom-primitive escape hatch.
#[doc(hidden)]
pub trait CustomTracingOperation<T: Type + Display, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of one custom primitive.
    fn custom_op(primitive: Arc<CustomPrimitive<T, V>>) -> Self;
}

/// Hidden staging trait for the custom-primitive escape hatch in linear programs.
#[doc(hidden)]
pub trait LinearCustomOperation<T: Type + Display, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of one custom primitive in the linear universe.
    fn linear_custom_op(primitive: CustomPrimitive<T, V>) -> Result<Self, TracingError>;

    /// Constructs the carrier-specific representation of one shared custom primitive in the linear universe.
    fn linear_custom_arc_op(primitive: Arc<CustomPrimitive<T, V>>) -> Result<Self, TracingError>;
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

/// Type-erased wrapper for a linearized-JIT replay rule stored inside [`CustomPrimitiveExtensions`].
///
/// This wrapper is `'static` so it can live inside the extension registry. The `V: Traceable<ArrayType>`
/// bound is required at construction time but does not appear on the outer [`CustomPrimitive`] struct.
struct LinearizedJitRule<
    V: Traceable<ArrayType> + ZeroLike,
    O: Clone + Operation<ArrayType> + 'static,
    OuterLinearOperation: Clone + 'static,
    InnerLinearOperation: Clone
        + Operation<ArrayType>
        + LinearAddOperation<ArrayType, Tracer<'static, E>>
        + LinearNegOperation<ArrayType, Tracer<'static, E>>
        + LinearScaleOperation<ArrayType, Tracer<'static, E>>
        + 'static,
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = OuterLinearOperation>
        + ?Sized
        + 'static,
>(Arc<dyn InterpretableOperation<ArrayType, Linearized<Tracer<'static, E>, InnerLinearOperation>>>);

impl<
    V: Traceable<ArrayType> + ZeroLike,
    O: Clone + Operation<ArrayType> + 'static,
    OuterLinearOperation: Clone + 'static,
    InnerLinearOperation: Clone
        + Operation<ArrayType>
        + LinearAddOperation<ArrayType, Tracer<'static, E>>
        + LinearNegOperation<ArrayType, Tracer<'static, E>>
        + LinearScaleOperation<ArrayType, Tracer<'static, E>>
        + 'static,
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = OuterLinearOperation>
        + ?Sized
        + 'static,
> LinearizedJitRule<V, O, OuterLinearOperation, InnerLinearOperation, E>
{
    fn interpret(
        &self,
        inputs: &[Linearized<Tracer<'static, E>, InnerLinearOperation>],
    ) -> Result<Vec<Linearized<Tracer<'static, E>, InnerLinearOperation>>, TracingError> {
        self.0.interpret(inputs)
    }
}

/// Type-erased wrapper for the canonical core linearized-JIT replay rule stored inside
/// [`CustomPrimitiveExtensions`].
struct CanonicalLinearizedJitRule<
    V: Value<ArrayType> + ZeroLike,
    E: Engine<
            Type = ArrayType,
            Value = V,
            TracingOperation = PrimitiveOperation<ArrayType, V>,
            LinearOperation = LinearPrimitiveOperation<ArrayType, V>,
        > + ?Sized
        + 'static,
>(Arc<dyn for<'engine> InterpretableOperation<ArrayType, Linearized<Tracer<'engine, E>>>>);

impl<
    V: Value<ArrayType> + ZeroLike,
    E: Engine<
            Type = ArrayType,
            Value = V,
            TracingOperation = PrimitiveOperation<ArrayType, V>,
            LinearOperation = LinearPrimitiveOperation<ArrayType, V>,
        > + ?Sized
        + 'static,
> CanonicalLinearizedJitRule<V, E>
{
    fn interpret<'engine>(
        &self,
        inputs: &[Linearized<Tracer<'engine, E>>],
    ) -> Result<Vec<Linearized<Tracer<'engine, E>>>, TracingError> {
        self.0.interpret(inputs)
    }
}

/// Type-erased wrapper for a staged-carrier-specific JVP rule stored inside [`CustomPrimitiveExtensions`].
struct JvpRule<T: Type + Display, V: Traceable<T> + Parameter, O: Clone, L: Clone + Operation<T>>(
    Arc<dyn DifferentiableOperation<T, V, LinearTerm<T, V, L>, O, L>>,
);

impl<T: Type + Display, V: Traceable<T> + Parameter, O: Clone, L: Clone + Operation<T>> JvpRule<T, V, O, L> {
    fn rule(&self) -> &dyn DifferentiableOperation<T, V, LinearTerm<T, V, L>, O, L> {
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
/// - [`DifferentiableOperation<ArrayType, V, LinearTerm<ArrayType, V>>`] for forward-mode JVP,
/// - [`VectorizableOperation<ArrayType, V>`] for `vmap`, and
/// - [`InterpretableOperation<ArrayType, Linearized<Tracer<'engine, E>>>`] for fully general linearized-JIT replay.
#[derive(Clone)]
pub struct CustomPrimitive<T: Type + Display, V: Traceable<T> + Parameter> {
    /// Required base op providing abstract evaluation and eager interpretation.
    base: Arc<dyn CustomBaseOperation<T, V>>,

    /// Optional reverse-mode transpose rule for the primitive.
    transpose_rule: Option<Arc<dyn LinearOperation<T, V>>>,

    /// Optional batching rule for the primitive.
    vectorization_rule: Option<Arc<dyn VectorizableOperation<T, V>>>,

    /// Typed extension registry carrying backend- or transform-specific extra rules.
    extensions: CustomPrimitiveExtensions<T, V>,
}

impl<T: Type + Display + 'static, V: Traceable<T> + Parameter + 'static> CustomPrimitive<T, V> {
    /// Creates one custom primitive from its required base operation.
    pub fn new<Base>(base: Base) -> Self
    where
        Base: Operation<T> + InterpretableOperation<T, V> + 'static,
    {
        Self {
            base: Arc::new(base),
            transpose_rule: None,
            vectorization_rule: None,
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

    /// Registers one staged-carrier-specific forward-mode JVP rule.
    pub fn with_jvp_rule_for<O, L, Rule>(mut self, rule: Rule) -> Self
    where
        O: Clone + 'static,
        L: Clone + Operation<T> + 'static,
        Rule: DifferentiableOperation<T, V, LinearTerm<T, V, L>, O, L> + 'static,
    {
        self.extensions.insert(JvpRule::<T, V, O, L>(Arc::new(rule)));
        self
    }

    /// Registers one batching rule.
    pub fn with_vectorization_rule<Rule>(mut self, rule: Rule) -> Self
    where
        Rule: VectorizableOperation<T, V> + 'static,
    {
        self.vectorization_rule = Some(Arc::new(rule));
        self
    }

    /// Registers one staged-carrier-specific linearized-JIT replay rule for nested custom primitives.
    #[doc(hidden)]
    pub fn with_linearized_jit_rule_for<O, OuterLinearOperation, InnerLinearOperation, E, Rule>(
        mut self,
        rule: Rule,
    ) -> Self
    where
        O: Clone + Operation<ArrayType> + 'static,
        OuterLinearOperation: Clone + 'static,
        E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = OuterLinearOperation>
            + ?Sized
            + 'static,
        InnerLinearOperation: Clone
            + Operation<ArrayType>
            + LinearAddOperation<ArrayType, Tracer<'static, E>>
            + LinearNegOperation<ArrayType, Tracer<'static, E>>
            + LinearScaleOperation<ArrayType, Tracer<'static, E>>
            + 'static,
        Rule: InterpretableOperation<ArrayType, Linearized<Tracer<'static, E>, InnerLinearOperation>> + 'static,
        Linearized<Tracer<'static, E>, InnerLinearOperation>: Traceable<ArrayType>,
        V: Traceable<ArrayType> + ZeroLike,
    {
        self.extensions
            .insert(LinearizedJitRule::<V, O, OuterLinearOperation, InnerLinearOperation, E>(Arc::new(rule)));
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

    pub(super) fn missing_rule(&self, transform: &'static str) -> TracingError {
        TracingError::MissingCustomRule { op: self.base.name(), transform }
    }

    fn jvp_rule<O: Clone + 'static, L: Clone + Operation<T> + 'static>(
        &self,
    ) -> Result<&dyn DifferentiableOperation<T, V, LinearTerm<T, V, L>, O, L>, TracingError> {
        self.extensions
            .get::<JvpRule<T, V, O, L>>()
            .map(JvpRule::rule)
            .ok_or_else(|| self.missing_rule("jvp"))
    }
}

impl<V: Traceable<ArrayType> + Parameter + ZeroLike + 'static> CustomPrimitive<ArrayType, V> {
    /// Registers one forward-mode JVP rule for the canonical core staged carriers.
    pub fn with_jvp_rule<Rule>(self, rule: Rule) -> Self
    where
        Rule: DifferentiableOperation<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOperation<ArrayType, V>,
                LinearPrimitiveOperation<ArrayType, V>,
            > + 'static,
    {
        self.with_jvp_rule_for::<PrimitiveOperation<ArrayType, V>, LinearPrimitiveOperation<ArrayType, V>, _>(rule)
    }

    /// Registers one linearized-JIT replay rule for the canonical core staged carriers.
    #[doc(hidden)]
    pub fn with_linearized_jit_rule<E, Rule>(mut self, rule: Rule) -> Self
    where
        V: Value<ArrayType>,
        E: Engine<
                Type = ArrayType,
                Value = V,
                TracingOperation = PrimitiveOperation<ArrayType, V>,
                LinearOperation = LinearPrimitiveOperation<ArrayType, V>,
            > + ?Sized
            + 'static,
        Rule: for<'engine> InterpretableOperation<ArrayType, Linearized<Tracer<'engine, E>>> + 'static,
        for<'engine> Linearized<Tracer<'engine, E>>: Traceable<ArrayType>,
    {
        self.extensions.insert(CanonicalLinearizedJitRule::<V, E>(Arc::new(rule)));
        self
    }

    /// Runs the canonical core linearized-JIT replay rule for this primitive.
    #[doc(hidden)]
    pub(crate) fn interpret_linearized_jit<'engine, E>(
        &self,
        inputs: &[Linearized<Tracer<'engine, E>>],
    ) -> Result<Vec<Linearized<Tracer<'engine, E>>>, TracingError>
    where
        V: Value<ArrayType>,
        E: Engine<
                Type = ArrayType,
                Value = V,
                TracingOperation = PrimitiveOperation<ArrayType, V>,
                LinearOperation = LinearPrimitiveOperation<ArrayType, V>,
            > + ?Sized
            + 'static,
    {
        self.extensions
            .get::<CanonicalLinearizedJitRule<V, E>>()
            .ok_or_else(|| self.missing_rule("linearized JIT replay"))?
            .interpret(inputs)
    }
}

impl<T: Type + Display, V: Traceable<T>> Debug for CustomPrimitive<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self.base.as_ref(), formatter)
    }
}

impl<T: Type + Display, V: Traceable<T>> Display for CustomPrimitive<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self.base.as_ref(), formatter)
    }
}

impl<V: Traceable<ArrayType>> Operation for CustomPrimitive<ArrayType, V> {
    #[inline]
    fn name(&self) -> &'static str {
        self.base.name()
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TracingError> {
        self.base.infer_output_types(input_types)
    }

    #[inline]
    fn try_simplify(
        &self,
        inputs: &[AtomId],
        is_zero_constant: &dyn Fn(AtomId) -> bool,
        is_one_constant: &dyn Fn(AtomId) -> bool,
    ) -> Option<Vec<AtomId>> {
        self.base.try_simplify(inputs, is_zero_constant, is_one_constant)
    }
}

impl<V: Traceable<ArrayType>> InterpretableOperation<ArrayType, V> for CustomPrimitive<ArrayType, V> {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        self.base.interpret(inputs)
    }
}

impl<V: Traceable<ArrayType> + 'static> LinearOperation<ArrayType, V> for CustomPrimitive<ArrayType, V> {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V>>>, TracingError> {
        self.transpose_rule
            .as_deref()
            .ok_or_else(|| self.missing_rule("transpose"))?
            .transpose(output_cotangents)
    }
}

impl<V: Traceable<ArrayType> + 'static> VectorizableOperation<ArrayType, V> for CustomPrimitive<ArrayType, V> {
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TracingError> {
        self.vectorization_rule.as_deref().ok_or_else(|| self.missing_rule("vectorize"))?.batch(inputs)
    }
}

impl<V: Traceable<ArrayType> + 'static, O: Clone + 'static, L: Clone + Operation<ArrayType> + 'static>
    DifferentiableOperation<ArrayType, V, LinearTerm<ArrayType, V, L>, O, L> for CustomPrimitive<ArrayType, V>
{
    fn jvp(
        &self,
        engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L>,
        inputs: &[JvpTracer<V, LinearTerm<ArrayType, V, L>>],
    ) -> Result<Vec<JvpTracer<V, LinearTerm<ArrayType, V, L>>>, TracingError> {
        self.jvp_rule::<O, L>()?.jvp(engine, inputs)
    }
}

impl<
    V: Value<ArrayType> + ZeroLike + 'static,
    O: Clone + Operation<ArrayType> + 'static,
    OuterLinearOperation: Clone + 'static,
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = OuterLinearOperation>
        + ?Sized
        + 'static,
    InnerLinearOperation: Clone
        + Operation<ArrayType>
        + LinearAddOperation<ArrayType, Tracer<'static, E>>
        + LinearNegOperation<ArrayType, Tracer<'static, E>>
        + LinearScaleOperation<ArrayType, Tracer<'static, E>>
        + 'static,
> InterpretableOperation<ArrayType, Linearized<Tracer<'static, E>, InnerLinearOperation>>
    for CustomPrimitive<ArrayType, V>
where
    Linearized<Tracer<'static, E>, InnerLinearOperation>: Traceable<ArrayType>,
{
    fn interpret(
        &self,
        inputs: &[Linearized<Tracer<'static, E>, InnerLinearOperation>],
    ) -> Result<Vec<Linearized<Tracer<'static, E>, InnerLinearOperation>>, TracingError> {
        self.extensions
            .get::<LinearizedJitRule<V, O, OuterLinearOperation, InnerLinearOperation, E>>()
            .ok_or_else(|| self.missing_rule("linearized JIT replay"))?
            .interpret(inputs)
    }
}

/// Linear-only wrapper around one [`CustomPrimitive`] that guarantees a transpose rule is present.
///
/// Linear programs cannot store an op unless reverse-mode transposition is known to exist. This
/// wrapper is the proof object that a custom primitive has satisfied that requirement.
#[derive(Clone)]
pub struct LinearCustomPrimitive<T: Type + Display, V: Traceable<T> + Parameter> {
    /// Wrapped custom primitive known to provide a transpose rule.
    primitive: Arc<CustomPrimitive<T, V>>,
}

impl<T: Type + Display + 'static, V: Traceable<T> + 'static> LinearCustomPrimitive<T, V> {
    /// Creates one linear-only wrapper from a custom primitive that already provides a transpose rule.
    pub fn from_custom_primitive(primitive: Arc<CustomPrimitive<T, V>>) -> Result<Self, TracingError> {
        primitive.transpose_rule.as_ref().ok_or_else(|| primitive.missing_rule("transpose"))?;
        Ok(Self { primitive })
    }

    /// Returns the wrapped custom primitive.
    #[inline]
    pub fn primitive(&self) -> &Arc<CustomPrimitive<T, V>> {
        &self.primitive
    }
}

impl<T: Type + Display, V: Traceable<T>> Debug for LinearCustomPrimitive<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self.primitive.as_ref(), formatter)
    }
}

impl<T: Type + Display, V: Traceable<T>> Display for LinearCustomPrimitive<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self.primitive.as_ref(), formatter)
    }
}

impl<V: Traceable<ArrayType>> Operation for LinearCustomPrimitive<ArrayType, V> {
    #[inline]
    fn name(&self) -> &'static str {
        self.primitive.name()
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TracingError> {
        self.primitive.infer_output_types(input_types)
    }

    #[inline]
    fn try_simplify(
        &self,
        inputs: &[AtomId],
        is_zero_constant: &dyn Fn(AtomId) -> bool,
        is_one_constant: &dyn Fn(AtomId) -> bool,
    ) -> Option<Vec<AtomId>> {
        self.primitive.try_simplify(inputs, is_zero_constant, is_one_constant)
    }
}

impl<V: Traceable<ArrayType>> InterpretableOperation<ArrayType, V> for LinearCustomPrimitive<ArrayType, V> {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        self.primitive.interpret(inputs)
    }
}

impl<V: Traceable<ArrayType>> LinearOperation<ArrayType, V> for LinearCustomPrimitive<ArrayType, V> {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V>>>, TracingError> {
        self.primitive
            .transpose_rule
            .as_deref()
            .expect("linear custom primitives must carry a transpose rule")
            .transpose(output_cotangents)
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc, sync::Arc};

    use pretty_assertions::assert_eq;

    use super::*;
    use crate::batching::{Batch, vmap};
    use crate::tracing_v2::{
        LinearPrimitiveOperation, OneLike, PrimitiveOperation, Program, ProgramBuilder, Tracer, TracingError,
        engine::ArrayScalarEngine, grad, interpret_and_trace, jvp,
    };
    use crate::types::{ArrayType, DataType, Shape};

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

    impl Operation for ShiftOp {
        fn name(&self) -> &'static str {
            "test_shift"
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TracingError> {
            if input_types.len() != 1 {
                return Err(TracingError::InvalidInputCount { expected: 1, got: input_types.len() });
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
            output_cotangents: &[LinearTerm<ArrayType, f64>],
        ) -> Result<Vec<Option<LinearTerm<ArrayType, f64>>>, TracingError> {
            if output_cotangents.len() != 1 {
                return Err(TracingError::InvalidInputCount { expected: 1, got: output_cotangents.len() });
            }
            Ok(vec![Some(output_cotangents[0].clone())])
        }
    }

    impl
        DifferentiableOperation<
            ArrayType,
            f64,
            LinearTerm<ArrayType, f64>,
            PrimitiveOperation<ArrayType, f64>,
            LinearPrimitiveOperation<ArrayType, f64>,
        > for ShiftOp
    {
        fn jvp(
            &self,
            _engine: &dyn Engine<
                Type = ArrayType,
                Value = f64,
                TracingOperation = PrimitiveOperation<ArrayType, f64>,
                LinearOperation = LinearPrimitiveOperation<ArrayType, f64>,
            >,
            inputs: &[JvpTracer<f64, LinearTerm<ArrayType, f64>>],
        ) -> Result<Vec<JvpTracer<f64, LinearTerm<ArrayType, f64>>>, TracingError> {
            if inputs.len() != 1 {
                return Err(TracingError::InvalidInputCount { expected: 1, got: inputs.len() });
            }
            Ok(vec![JvpTracer { primal: inputs[0].primal + self.amount, tangent: inputs[0].tangent.clone() }])
        }
    }

    impl VectorizableOperation<ArrayType, f64> for ShiftOp {
        fn batch(&self, inputs: &[Batch<f64>]) -> Result<Vec<Batch<f64>>, TracingError> {
            if inputs.len() != 1 {
                return Err(TracingError::InvalidInputCount { expected: 1, got: inputs.len() });
            }
            Ok(vec![Batch::new(inputs[0].lanes().iter().map(|lane| lane + self.amount).collect::<Vec<_>>())])
        }
    }

    impl<'engine, E> InterpretableOperation<ArrayType, Linearized<Tracer<'engine, E>>> for ShiftOp
    where
        E: Engine<
                Type = ArrayType,
                Value = f64,
                TracingOperation = PrimitiveOperation<ArrayType, f64>,
                LinearOperation = LinearPrimitiveOperation<ArrayType, f64>,
            > + ?Sized
            + 'static,
    {
        fn interpret(
            &self,
            inputs: &[Linearized<Tracer<'engine, E>>],
        ) -> Result<Vec<Linearized<Tracer<'engine, E>>>, TracingError> {
            if inputs.len() != 1 {
                return Err(TracingError::InvalidInputCount { expected: 1, got: inputs.len() });
            }
            let primal = apply_custom_traced_unary(
                inputs[0].primal.clone(),
                CustomPrimitive::<ArrayType, f64>::new(self.clone()),
            )?;
            Ok(vec![Linearized { primal, tangent: inputs[0].tangent.clone() }])
        }
    }

    /// Applies one unary custom primitive to one traced scalar.
    fn apply_custom_traced_unary<'engine, E>(
        input: Tracer<'engine, E>,
        primitive: CustomPrimitive<ArrayType, f64>,
    ) -> Result<Tracer<'engine, E>, TracingError>
    where
        E: Engine<
                Type = ArrayType,
                Value = f64,
                TracingOperation = PrimitiveOperation<ArrayType, f64>,
                LinearOperation = LinearPrimitiveOperation<ArrayType, f64>,
            > + ?Sized
            + 'static,
    {
        Ok(Tracer::apply_staged_op(
            input.engine,
            input.builder.clone(),
            std::slice::from_ref(&input),
            PrimitiveOperation::Custom(Arc::new(primitive)),
        )?
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
        E: Engine<
                Type = ArrayType,
                Value = f64,
                TracingOperation = PrimitiveOperation<ArrayType, f64>,
                LinearOperation = LinearPrimitiveOperation<ArrayType, f64>,
            > + ?Sized
            + 'static,
    {
        apply_custom_traced_unary(input, primitive).expect("custom primitive staging should succeed")
    }

    /// Applies one unary custom primitive to one batched scalar.
    fn apply_custom_batched_unary(
        input: Batch<f64>,
        primitive: CustomPrimitive<ArrayType, f64>,
    ) -> Result<Batch<f64>, TracingError> {
        Ok(VectorizableOperation::batch(&PrimitiveOperation::Custom(Arc::new(primitive)), &[input])?
            .into_iter()
            .next()
            .expect("unary custom primitive should produce one batched output"))
    }

    /// Returns one scalar array type used by these custom-primitive tests.
    fn scalar_type() -> ArrayType {
        ArrayType::new(DataType::F64, Shape::scalar(), None, None).expect("scalar array types should be valid")
    }

    #[test]
    fn test_linear_custom_primitive_requires_transpose_rule_up_front() {
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0));

        assert!(matches!(
            primitive.into_linear(),
            Err(TracingError::MissingCustomRule { op: "test_shift", transform: "transpose" })
        ));
    }

    #[test]
    fn test_custom_primitive_base_execution_replays_without_optional_rules() {
        let engine = ArrayScalarEngine::<f64>::new();
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0));
        let (output, compiled): (f64, Program<ArrayType, f64, PrimitiveOperation<ArrayType, f64>, f64, f64>) =
            interpret_and_trace(
                &engine,
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
        let builder =
            Rc::new(RefCell::new(ProgramBuilder::<ArrayType, f64, LinearPrimitiveOperation<ArrayType, f64>>::new()));
        let cotangent_atom = builder.borrow_mut().add_input(0.0f64.r#type().into_owned());
        let cotangent = LinearTerm::from_staged_parts(cotangent_atom, builder);

        assert!(matches!(
            primitive.transpose(&[cotangent]),
            Err(TracingError::MissingCustomRule { op: "test_shift", transform: "transpose" })
        ));
    }

    #[test]
    fn test_custom_primitive_missing_jvp_rule_reports_targeted_error() {
        let engine = ArrayScalarEngine::<f64>::new();
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

        assert_eq!(result, Err(TracingError::MissingCustomRule { op: "test_shift", transform: "jvp" }),);
    }

    #[test]
    fn test_custom_primitive_missing_linearized_jit_rule_reports_targeted_error() {
        let engine = ArrayScalarEngine::<f64>::new();
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0)).with_jvp_rule(ShiftOp::new(2.0));
        let result: Result<(f64, Program<ArrayType, f64, PrimitiveOperation<ArrayType, f64>, f64, f64>), TracingError> =
            interpret_and_trace(
                &engine,
                {
                    let primitive = primitive.clone();
                    move |x: Tracer<ArrayScalarEngine<f64>>| {
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
            Err(TracingError::MissingCustomRule { op: "test_shift", transform: "linearized JIT replay" })
        ));
    }

    #[test]
    fn test_custom_primitive_jvp_rule_participates_in_grad_and_linearized_jit_replay() {
        let engine = ArrayScalarEngine::<f64>::new();
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0))
            .with_jvp_rule(ShiftOp::new(2.0))
            .with_linearized_jit_rule::<ArrayScalarEngine<f64>, _>(ShiftOp::new(2.0));

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

        let (output, compiled): (f64, Program<ArrayType, f64, PrimitiveOperation<ArrayType, f64>, f64, f64>) =
            interpret_and_trace(
                &engine,
                {
                    let primitive = primitive.clone();
                    move |x: Tracer<ArrayScalarEngine<f64>>| {
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
    fn test_custom_primitive_batch_rule_reports_targeted_error_when_missing() {
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0));

        assert_eq!(
            apply_custom_batched_unary(Batch::new(vec![1.0f64, 2.0]), primitive),
            Err(TracingError::MissingCustomRule { op: "test_shift", transform: "vectorize" }),
        );
    }

    #[test]
    fn test_custom_primitive_batch_rule_participates_in_vmap() {
        let primitive =
            CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0)).with_vectorization_rule(ShiftOp::new(2.0));
        let engine = ArrayScalarEngine::<f64>::new();
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));

        assert_eq!(
            vmap(
                &engine,
                builder,
                {
                    let primitive = primitive.clone();
                    move |batch: Batch<f64>| apply_custom_batched_unary(batch, primitive.clone()).unwrap()
                },
                vec![1.0f64, 2.0, 3.0],
            ),
            Ok(vec![3.0f64, 4.0, 5.0]),
        );
    }

    #[test]
    fn test_custom_primitive_abstract_eval_uses_the_registered_base_op() {
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0));

        assert_eq!(primitive.infer_output_types(&[scalar_type()]), Ok(vec![scalar_type()]));
    }
}
