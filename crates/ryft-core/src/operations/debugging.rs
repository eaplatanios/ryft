use std::borrow::Cow;
use std::fmt::Display;
use std::marker::PhantomData;

use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_elementwise_operation};
use crate::operations::ElementwiseOperation;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    EffectClass, EffectClasses, Effects, Operation, OperationFormatter, ProgramError, RegionInterface, Type, TypeError,
    Value,
};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`PrintOperation`].
pub const PRINT_OPERATION_NAME: &str = "print";

/// [`Operation`] that returns its input unchanged while printing it to standard error with a label — the analogue of
/// [`jax.debug.print`](https://docs.jax.dev/en/latest/debugging/print_breakpoint.html). Refer to the documentation of
/// [`Print`] for more information.
///
/// By default, [`Operation::effects`] reports [`EffectClass::OrderedIo`], so program transforms never eliminate it as
/// dead code (even when nothing consumes its output) and preserve its execution order relative to other ordered-I/O
/// operations across every participating device. [`with_effect_class`](Self::with_effect_class) selects a weaker
/// contract instead: [`EffectClass::DeviceOrderedIo`] keeps program order only among the ordered I/O executing on
/// the same device, which allows the print to run once per device inside `shard_map` bodies, and
/// [`EffectClass::UnorderedIo`] retains the print without ordering it relative to other I/O. Partial evaluation
/// places it by input known-ness like any other operation — an all-known print folds into the known side (printing
/// at partial-evaluation time under an eager known-side context, which is also what makes linearization print during
/// the forward pass), while a mixed-input print residualizes. Differentiation passes the tangent through unchanged
/// while re-printing the primal value, and transposition is the identity on the cotangent (adjoints are not printed).
///
/// Eager interpretation prints directly. The XLA backend lowers this operation to a StableHLO host-callback custom
/// call (`@ryft.print`) threaded on a token chain that preserves ordered-I/O execution order within one dispatch,
/// including through `while`/`if` regions; refer to `ryft-xla`'s `experimental::debugging` module for the calling
/// convention and the capturable output sink.
///
/// The `T` parameter fixes this payload's type universe, so each concrete [`PrintOperation`] implements exactly one
/// [`Operation`] contract.
#[derive(Clone, Debug)]
pub struct PrintOperation<T: Type> {
    /// Label printed before the value.
    label: String,

    /// Observable I/O effect class of this print.
    effect_class: EffectClass,

    /// Type universe in which this operation is valid.
    marker: PhantomData<fn() -> T>,
}

impl<T: Type> PrintOperation<T> {
    /// Creates a new [`PrintOperation`] with the provided label and the default [`EffectClass::OrderedIo`] effect.
    #[inline]
    pub fn new<L: Into<String>>(label: L) -> Self {
        Self { label: label.into(), effect_class: EffectClass::OrderedIo, marker: PhantomData }
    }

    /// Returns the label carried by this [`PrintOperation`].
    #[inline]
    pub fn label(&self) -> &str {
        self.label.as_str()
    }

    /// Returns the [`EffectClass`] declared by this [`PrintOperation`].
    #[inline]
    pub fn effect_class(&self) -> EffectClass {
        self.effect_class
    }

    /// Returns this print declaring the provided I/O effect class, replacing the default [`EffectClass::OrderedIo`].
    /// Every class keeps the print observable; they differ only in ordering scope, as described in the type
    /// documentation.
    #[inline]
    pub fn with_effect_class(mut self, effect_class: EffectClass) -> Self {
        self.effect_class = effect_class;
        self
    }
}

impl<T: Type> Display for PrintOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type> Operation for PrintOperation<T> {
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        PRINT_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[T],
        _region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![input_types[0].clone()])
    }

    #[inline]
    fn effects(&self) -> Cow<'_, Effects> {
        Cow::Owned(Effects::explicit(EffectClasses::single(self.effect_class)))
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, PRINT_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("label", &self.label)?;
            if self.effect_class != EffectClass::OrderedIo {
                operation.field("effect_class", self.effect_class)?;
            }
            Ok(())
        })
    }
}

impl ElementwiseOperation for PrintOperation<crate::arrays::ArrayType> {
    #[inline]
    fn input_count(&self) -> usize {
        1
    }
}

impl<C: Domain> InterpretableOperation<C> for PrintOperation<C::Type> {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        eprintln!("{}: {}", self.label, inputs[0]);
        Ok(vec![inputs[0].clone()])
    }
}

// Partial evaluation defers to the default behavior of
// [`Program::partially_evaluate`](crate::Program::partially_evaluate): the print folds into the known side when its
// input is known and residualizes otherwise, with dead-code elimination keeping residual prints alive because
// [`Operation::effects`] is not pure.
impl<C: Context> PartiallyEvaluatableOperation<C> for PrintOperation<C::Type> where
    C::Operation: From<PrintOperation<C::Type>>
{
}

impl_differentiable_elementwise_operation! {
    @linear<T>
    PrintOperation<T>,
    rule = [@positive]
}

/// Represents the ability to print values in programs with labels. [`Print`] stages a [`PrintOperation`], which is
/// effectively an identity function that prints its input to standard error when executed. Because the staged
/// operation defaults to [`EffectClass::OrderedIo`], the print survives dead-code elimination and keeps its execution
/// order relative to other ordered prints.
pub trait Print: Sized {
    /// Returns this value unchanged while printing it to standard error with `label`.
    fn print(self, label: &str) -> Self {
        self.print_with_effect_class(label, EffectClass::OrderedIo)
    }

    /// Returns this value unchanged while printing it with the provided I/O [`EffectClass`].
    ///
    /// [`EffectClass::DeviceOrderedIo`] keeps the print ordered only relative to the ordered I/O on the same device,
    /// and [`EffectClass::UnorderedIo`] lets it run independently of other prints while remaining observable. The
    /// effect class does not determine whether a backend executes the call inline or asynchronously.
    fn print_with_effect_class(self, label: &str, effect_class: EffectClass) -> Self;
}

// Any context-carrying value prints by binding a [`PrintOperation`] through its own context. The
// `From<PrintOperation<V::Type>>` bound makes this disjoint from the eager value types (whose context operation is
// [`ConstantOperation`](crate::operations::constants::ConstantOperation)), so it covers the transform tracers
// without conflicting with concrete implementations.
impl<V: Value> Print for V
where
    V::DispatchDomain: Context<Operation: From<PrintOperation<V::Type>>>,
{
    #[inline]
    fn print_with_effect_class(self, label: &str, effect_class: EffectClass) -> Self {
        self.dispatch_domain()
            .bind(PrintOperation::new(label).with_effect_class(effect_class), Vec::new(), std::slice::from_ref(&self))
            .expect("`print` operation failed")
            .remove(0)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayOperation, ArrayType, DataType};
    use crate::contexts::EagerContext;
    use crate::differentiation::differentiate_at;
    use crate::programs::EmptyRegionDriver;
    use crate::tracing::{DomainTracer, Trace};

    use super::*;

    /// Computes `f(x) = x * x` while printing `x` and discarding the printed value, so the staged `print` is dead
    /// code that only its effect keeps alive.
    fn print_square<V: Clone + Print + std::ops::Mul<Output = V>>(x: V) -> V {
        let _printed = x.clone().print("x");
        x.clone() * x
    }

    #[test]
    fn test_print() {
        let operation = PrintOperation::new("x");
        let scalar_type = ArrayType::scalar(DataType::F64);

        assert_eq!(operation.label(), "x");
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedIo));
        assert_eq!(Operation::infer_output_types(&operation, &[scalar_type.clone()], &[]), Ok(vec![scalar_type]));
        assert_eq!(operation.to_string(), "print [label=x]");
    }

    #[test]
    fn test_print_operation_with_effect_class() {
        let operation = PrintOperation::<ArrayType>::new("x");
        assert_eq!(operation.effect_class(), EffectClass::OrderedIo);
        for effect_class in [EffectClass::DeviceOrderedIo, EffectClass::UnorderedIo] {
            let operation = operation.clone().with_effect_class(effect_class);
            assert_eq!(operation.effect_class(), effect_class);
            assert_eq!(operation.effects().classes(), EffectClasses::single(effect_class));
            assert_eq!(operation.to_string(), format!("print [label=x, effect_class={effect_class}]"));
        }
        let restored = operation.with_effect_class(EffectClass::UnorderedIo).with_effect_class(EffectClass::OrderedIo);
        assert_eq!(restored.effects().classes(), EffectClasses::single(EffectClass::OrderedIo));
        assert_eq!(restored.to_string(), "print [label=x]");
    }

    #[test]
    fn test_print_with_effect_class() {
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| {
                let _printed = x.clone().print_with_effect_class("x", EffectClass::UnorderedIo);
                Ok(x.clone() * x)
            },
            ArrayType::scalar(DataType::F64),
        )
        .unwrap();
        let program = program.to_flat_program();
        assert_eq!(program.effects().classes(), EffectClasses::single(EffectClass::UnorderedIo));
        // An unused unordered print survives the primal partition, retaining its class through differentiation.
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.primal().effects().classes(), EffectClasses::single(EffectClass::UnorderedIo));
        assert_eq!(linearization.tangent().effects().classes(), EffectClasses::NONE);
    }

    #[test]
    fn test_print_interpretation() {
        let context = EagerContext::<Array>::new();
        let input = Array::scalar(3.0);
        let outputs =
            PrintOperation::new("x").interpret(&context.clone(), &EmptyRegionDriver, &[input.clone()]).unwrap();
        assert_eq!(outputs, vec![input]);
    }

    #[test]
    fn test_print_stages_through_the_tracer_capability() {
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok(x.print("x")),
            ArrayType::scalar(DataType::F64),
        )
        .unwrap();
        let program = program.to_flat_program();
        assert_eq!(program.instructions().len(), 1);
        assert!(matches!(
            program.instructions()[0].operation(),
            ArrayOperation::Print(operation) if operation.label() == "x",
        ));
        assert_eq!(program.effects().classes(), EffectClasses::single(EffectClass::OrderedIo));
    }

    #[test]
    fn test_print_differentiation() {
        // The JVP rule re-prints the primal and passes the tangent through, so the effect survives on the primal
        // side of the linearization without perturbing the gradient. The dead primal print (its output is unused by
        // the gradient) exercises the effect keep-alive of the partition projections.
        let (value, gradient) = differentiate_at(Array::scalar(3.0)).value_and_gradient(print_square).unwrap();
        assert_eq!(value.to_f64s()[0], 9.0);
        assert_eq!(gradient.to_f64s()[0], 6.0);
    }

    #[test]
    fn test_dead_prints_survive_linearization_on_the_primal_side() {
        // Linearizing `print_square` partitions its jvp program into primal and tangent stages: the dead print rides
        // the primal (known) stage through the partition projections' effect keep-alive, and the tangent stage stays
        // print-free because the JVP rule keeps the effect on the primal side.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok(print_square(x)),
            ArrayType::scalar(DataType::F64),
        )
        .unwrap();
        let linearization = program.to_flat_program().linearize().unwrap();
        let primal_prints = linearization
            .primal()
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), ArrayOperation::Print(_)))
            .count();
        let tangent_prints = linearization
            .tangent()
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), ArrayOperation::Print(_)))
            .count();
        assert_eq!(primal_prints, 1);
        assert_eq!(tangent_prints, 0);
    }
}
