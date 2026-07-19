use std::fmt::Display;

use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_elementwise_operation};
use crate::operations::ElementwiseOperation;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::ProgramError;
use crate::programs::effects::{Effect, Effects};
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError};
use crate::programs::values::Value;
use crate::types::ArrayType;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`PrintOperation`].
pub const PRINT_OPERATION_NAME: &str = "print";

/// [`Operation`] that returns its input unchanged while printing it to standard error with a label — the analogue of
/// [`jax.debug.print`](https://docs.jax.dev/en/latest/debugging/print_breakpoint.html). Refer to the documentation of
/// [`Print`] for more information.
///
/// This is ryft's first operation with observable effects: [`Operation::effects`] reports [`Effect::OrderedIo`], so
/// program transforms never eliminate it as dead code (even when nothing consumes its output) and preserve its
/// execution order relative to other ordered-I/O operations. Partial evaluation places it by input known-ness like
/// any other operation — an all-known print folds into the known side (printing at partial-evaluation time under an
/// eager known-side context, which is also what makes linearization print during the forward pass), while a
/// mixed-input print residualizes. Differentiation passes the tangent through unchanged while re-printing the primal
/// value, and transposition is the identity on the cotangent (adjoints are not printed).
///
/// Eager interpretation prints directly. The XLA backend lowers this operation to a StableHLO host-callback custom
/// call (`@ryft.print`) threaded on a token chain that preserves ordered-I/O execution order within one dispatch,
/// including through `while`/`if` regions; refer to `ryft-xla`'s `experimental::debugging` module for the calling
/// convention and the capturable output sink.
#[derive(Clone, Debug)]
pub struct PrintOperation {
    /// Label printed before the value.
    label: String,
}

impl PrintOperation {
    /// Creates a new [`PrintOperation`] with the provided label.
    #[inline]
    pub fn new<L: Into<String>>(label: L) -> Self {
        Self { label: label.into() }
    }

    /// Returns the label carried by this [`PrintOperation`].
    #[inline]
    pub fn label(&self) -> &str {
        self.label.as_str()
    }
}

impl Display for PrintOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Operation::<ArrayType>::render(self, formatter, 0)
    }
}

impl<T: Type> Operation<T> for PrintOperation {
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
    fn effects(&self) -> Effects {
        Effects::single(Effect::OrderedIo)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, PRINT_OPERATION_NAME)?
            .bracketed(|operation| operation.field("label", &self.label))
    }
}

impl ElementwiseOperation for PrintOperation {
    #[inline]
    fn input_count(&self) -> usize {
        1
    }
}

impl<C: Domain> InterpretableOperation<C> for PrintOperation {
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

/// Partial evaluation defers to the default behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate): the print folds into the known side when its
/// input is known and residualizes otherwise, with dead-code elimination keeping residual prints alive because
/// [`Operation::effects`] is not [`Effects::PURE`].
impl<C: Context> PartiallyEvaluatableOperation<C> for PrintOperation where C::Operation: From<PrintOperation> {}

/// Represents the ability to print values in programs with labels. [`Print`] stages a [`PrintOperation`], which is
/// effectively an identity function that prints its input to standard error when executed. Because the staged
/// operation reports [`Effect::OrderedIo`], the print survives dead-code elimination and keeps its execution order
/// relative to other prints.
pub trait Print: Sized {
    /// Returns this value unchanged while printing it to standard error with `label`.
    fn print(self, label: &str) -> Self;
}

/// Any context-carrying value prints by binding a [`PrintOperation`] through its own context. The
/// `From<PrintOperation>` bound makes this disjoint from the eager value types (whose context operation is
/// [`ConstantOperation`](crate::operations::constants::ConstantOperation)), so it covers the transform tracers
/// without conflicting with concrete implementations.
impl<V: Value> Print for V
where
    V::DispatchDomain: Context<Operation: From<PrintOperation>>,
{
    #[inline]
    fn print(self, label: &str) -> Self {
        self.dispatch_domain()
            .bind(PrintOperation::new(label), Vec::new(), std::slice::from_ref(&self))
            .expect("`print` operation failed")
            .remove(0)
    }
}

impl_differentiable_elementwise_operation! {
    @linear
    PrintOperation,
    rule = [@positive]
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::contexts::EagerContext;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::tracing::{DomainTracer, Trace};
    use crate::tracing_v2::ReverseModeDifferentiate;
    use crate::types::{ArrayType, DataType};

    use super::*;

    /// Computes `f(x) = x * x` while printing `x` and discarding the printed value, so the staged `print` is dead
    /// code that only its effect keeps alive.
    fn print_square<V: Clone + Print + std::ops::Mul<Output = V>>(x: V) -> V {
        let _printed = x.clone().print("x");
        x.clone() * x
    }

    #[test]
    fn test_print_operation_contract() {
        let operation = PrintOperation::new("x");
        let scalar_type = ArrayType::scalar(DataType::F64);

        assert_eq!(operation.label(), "x");
        assert_eq!(Operation::<ArrayType>::name(&operation), PRINT_OPERATION_NAME);
        assert_eq!(Operation::<ArrayType>::effects(&operation), Effects::single(Effect::OrderedIo));
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[scalar_type.clone()], &[]),
            Ok(vec![scalar_type])
        );
        assert_eq!(operation.to_string(), "print [label=x]");
    }

    #[test]
    fn test_print_interprets_as_the_identity() {
        let context = EagerContext::<Array>::new();
        let input = Array::scalar(3.0);
        let outputs =
            PrintOperation::new("x").interpret(&context.clone(), &EmptyRegionDriver, &[input.clone()]).unwrap();
        assert_eq!(outputs, vec![input]);
    }

    #[test]
    fn test_print_is_transparent_to_differentiation() {
        // The JVP rule re-prints the primal and passes the tangent through, so the effect survives on the primal
        // side of the linearization without perturbing the gradient. The dead primal print (its output is unused by
        // the gradient) exercises the effect keep-alive of the partition projections.
        let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
        let (value, gradient) = domain.value_and_gradient(print_square, Array::scalar(3.0)).unwrap();
        assert_eq!(value.to_f64s()[0], 9.0);
        assert_eq!(gradient.to_f64s()[0], 6.0);
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
        assert_eq!(program.effects(), Effects::single(Effect::OrderedIo));
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
