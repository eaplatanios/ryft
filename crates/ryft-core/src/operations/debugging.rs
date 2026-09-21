//! Operations that expose intermediate values while a program runs without changing what it computes. Debugging
//! operations behave as identity functions on their inputs and exist only for their side effects, so they can be
//! inserted anywhere in a computation, including inside traced and transformed programs, and later removed without
//! touching the surrounding code. This module provides the following:
//!
//!   - The [`Print`] value capability, whose [`print`](Print::print) returns its input unchanged while printing it to
//!     standard error under a label, and whose [`print_with_effect_class`](Print::print_with_effect_class) selects how
//!     strongly the print is ordered relative to other I/O.
//!   - The [`PrintOperation`] that the capability stages. By default, it declares the [`EffectClass::OrderedIo`]
//!     effect, so dead-code elimination never removes it and prints appear in program order across all participating
//!     devices. [`EffectClass::DeviceOrderedIo`] keeps program order only among the prints executing on the same
//!     device, which lets a print inside a `shard_map` body run once per device, and
//!     [`EffectClass::UnorderedIo`] retains the print without ordering it at all.
//!
//! Program transforms treat print operations as identities on their data. Batching prints the whole batch, partial
//! evaluation prints known inputs when it encounters them and residualizes unknown ones, and differentiation prints
//! the primal value while passing the tangent through unchanged. Backends decide how the effect is realized. For
//! example, the XLA backend lowers a print to a host callback that is threaded through a token chain so that ordered
//! prints stay ordered within one dispatch.
//!
//! # Example
//!
//! A traced value stages a `print` instruction that prints whenever the program is interpreted or executed. Its output
//! value is the input, so the print is transparent to consumers:
//!
//! ```rust
//! # use indoc::indoc;
//! # use ryft_core::{
//! #     Array, ArrayOperation, ArrayType, DataType, EffectClass, Mul, Print, ProgramError, Trace, TracingContext,
//! # };
//! # fn main() -> Result<(), ProgramError> {
//! let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
//!     |input| {
//!         let squared = input.clone().mul(&input)?;
//!         squared.print_with_effect_class("squared", EffectClass::UnorderedIo)
//!     },
//!     ArrayType::scalar(DataType::F64),
//! )?;
//!
//! assert_eq!(
//!     program.to_string(),
//!     indoc! {"
//!         lambda %0:f64[] .
//!         let %1:f64[] = mul %0 %0
//!             %2:f64[] = print [label=squared, effect_class=unordered_io] %1
//!         in (%2)"},
//! );
//!
//! // Interpreting the program prints `squared = 9` to standard error and returns the squared input.
//! assert_eq!(program.interpret(Array::scalar(3.0_f64)?)?, Array::scalar(9.0_f64)?);
//! # Ok(())
//! # }
//! ```

use std::borrow::Cow;
use std::fmt::Display;
use std::marker::PhantomData;

use crate::arrays::ArrayType;
use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_elementwise_operation};
use crate::operations::ElementwiseOperation;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    EffectClass, EffectClasses, Effects, Operation, OperationFormatter, ProgramError, RegionInterface, Type, TypeError,
    Value,
};

/// Canonical operation name for [`PrintOperation`].
pub const PRINT_OPERATION_NAME: &str = "print";

/// [`Operation`] that returns its input unchanged while printing it to standard error with a label.
///
/// By default, [`Operation::effects`] reports [`EffectClass::OrderedIo`], so program transforms never eliminate it as
/// dead code (even when nothing consumes its output) and preserve its execution order relative to other ordered-I/O
/// operations across every participating device. [`with_effect_class`](Self::with_effect_class) selects a weaker
/// contract instead: [`EffectClass::DeviceOrderedIo`] keeps program order only among the ordered I/O executing on
/// the same device, which allows the print to run once per device inside "shard map" bodies, for example, and
/// [`EffectClass::UnorderedIo`] retains the print without ordering it relative to other I/O. Partial evaluation places
/// it on the known side when its input is known (printing at partial-evaluation time under an eager known-side context,
/// which also makes linearization print during the forward pass), and residualizes it otherwise. Differentiation passes
/// the tangent through unchanged while printing the primal value. The primitive transposition rule passes the cotangent
/// through without printing it. Whole-program transposition rejects effectful linear instructions, so reverse
/// differentiation transposes the print-free tangent program.
///
/// Eager interpretation prints directly. The XLA backend lowers this operation to a StableHLO host-callback custom
/// call (i.e., `@ryft.print`), using a token chain for ordered I/O to preserve execution order within one dispatch,
/// including through `if`/`while` regions.
///
/// The `T` parameter fixes this payload's type universe, so each concrete [`PrintOperation`] implements exactly one
/// [`Operation`] contract.
///
/// Refer to the documentation of [`Print`] for more information.
#[derive(Clone, Debug)]
pub struct PrintOperation<T: Type> {
    /// Refer to the documentation of [`label`](Self::label) for more information.
    label: String,

    /// Refer to the documentation of [`effect_class`](Self::effect_class) for more information.
    effect_class: EffectClass,

    /// [`PhantomData`] marker tying this [`Operation`] to the [`Type`] universe in which it is valid.
    marker: PhantomData<fn() -> T>,
}

impl<T: Type> PrintOperation<T> {
    /// Creates a new [`PrintOperation`] with the provided label and the default [`EffectClass::OrderedIo`] effect.
    #[inline]
    pub fn new<L: Into<String>>(label: L) -> Self {
        Self { label: label.into(), effect_class: EffectClass::OrderedIo, marker: PhantomData }
    }

    /// Returns a copy of this [`PrintOperation`] with its effect class set to the provided `effect_class`. Use
    /// [`EffectClass::OrderedIo`], [`EffectClass::DeviceOrderedIo`], or [`EffectClass::UnorderedIo`] to select
    /// the I/O ordering contract described in the type documentation.
    #[inline]
    pub fn with_effect_class(mut self, effect_class: EffectClass) -> Self {
        self.effect_class = effect_class;
        self
    }

    /// Returns the label printed before the input value, separated from it by `: `.
    #[inline]
    pub fn label(&self) -> &str {
        self.label.as_str()
    }

    /// Returns the [`EffectClass`] declared by this [`PrintOperation`].
    #[inline]
    pub fn effect_class(&self) -> EffectClass {
        self.effect_class
    }
}

impl<T: Type> Display for PrintOperation<T> {
    #[inline]
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

    #[inline]
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

impl ElementwiseOperation for PrintOperation<ArrayType> {
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

impl<C: Context> PartiallyEvaluatableOperation<C> for PrintOperation<C::Type> where
    C::Operation: From<PrintOperation<C::Type>>
{
}

impl_differentiable_elementwise_operation! {
    @linear<T>
    PrintOperation<T>,
    rule = [@positive],
}

/// Represents the ability to print values in programs with labels. [`Print`] stages a [`PrintOperation`], which is
/// effectively an identity function that prints its input to standard error when executed. Because the staged operation
/// defaults to [`EffectClass::OrderedIo`], the print survives dead-code elimination and keeps its execution order
/// relative to other ordered print instructions.
pub trait Print: Sized {
    /// Returns this value unchanged while printing it to standard error with `label`, and a [`ProgramError`] if the
    /// value's context fails to bind the operation. Tracing contexts never fail here, because the value is always
    /// native to its own context, but contexts that execute operations eagerly (e.g., a backend running operation
    /// by operation) may.
    #[inline]
    fn print(self, label: &str) -> Result<Self, ProgramError> {
        self.print_with_effect_class(label, EffectClass::OrderedIo)
    }

    /// Returns this value unchanged while printing it with the provided I/O [`EffectClass`], and a [`ProgramError`]
    /// if the value's context fails to bind the operation.
    ///
    /// [`EffectClass::DeviceOrderedIo`] keeps the print ordered only relative to the ordered I/O on the same device,
    /// and [`EffectClass::UnorderedIo`] lets it run independently of other prints while remaining observable. The
    /// effect class does not determine whether a backend executes the call inline or asynchronously.
    fn print_with_effect_class(self, label: &str, effect_class: EffectClass) -> Result<Self, ProgramError>;
}

impl<V: Value<DispatchDomain: Context<Operation: From<PrintOperation<V::Type>>>>> Print for V {
    #[inline]
    fn print_with_effect_class(self, label: &str, effect_class: EffectClass) -> Result<Self, ProgramError> {
        // Any context-carrying value prints by binding a `PrintOperation` through its own context. The
        // `From<PrintOperation<V::Type>>` bound makes this disjoint from the eager value types (whose context
        // operation is `ConstantOperation`), so it covers the transform tracers without conflicting with
        // concrete implementations.
        let mut outputs = self.dispatch_domain().bind(
            PrintOperation::new(label).with_effect_class(effect_class),
            Vec::new(),
            std::slice::from_ref(&self),
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayOperation, DataType};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{TransposableOperation, TranspositionContext, differentiate_at};
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_type_inference,
    };
    use crate::partial::PartialValue;
    use crate::programs::{EmptyRegionDriver, MaybeZero, Program};
    use crate::tracing::{DomainTracer, Trace, TracingContext};

    use super::*;

    #[test]
    fn test_print() {
        let operation = PrintOperation::<ArrayType>::new("x");

        // Operation identity, accessors, the default ordered-I/O effect, and rendering, which omits the default class.
        assert_eq!(operation.name(), PRINT_OPERATION_NAME);
        assert_eq!(operation.label(), "x");
        assert_eq!(operation.effect_class(), EffectClass::OrderedIo);
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedIo));
        assert_eq!(operation.input_count(), 1);
        assert_eq!(operation.to_string(), "print [label=x]");
    }

    #[test]
    fn test_print_with_effect_class() {
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
    fn test_print_type_inference() {
        check_operation_type_inference!(
            operation = PrintOperation::<ArrayType>::new("x"),
            cases = [{
                input_types = [ArrayType::scalar(DataType::F64)],
                output_types = [ArrayType::scalar(DataType::F64)],
            }, {
                input_types = [ArrayType::new_static(DataType::I32, [2, 3])],
                output_types = [ArrayType::new_static(DataType::I32, [2, 3])],
            }, {
                input_types = [],
                error = "expected 1 input but got 0",
            }, {
                input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
                error = "expected 1 input but got 2",
            }],
        );
    }

    #[test]
    fn test_print_interpretation() {
        let context = EagerContext::<Array>::new();
        let input = Array::scalar(3.0).unwrap();
        let operation = PrintOperation::new("x");

        // Standard error capture is not available here; verify the returned value and input validation.
        assert_eq!(
            operation.interpret(&context, &EmptyRegionDriver, std::slice::from_ref(&input)),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            operation.interpret(&context, &EmptyRegionDriver, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
        assert_eq!(
            operation.interpret(&context, &EmptyRegionDriver, &[input.clone(), input]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 2 }),
        );
    }

    #[test]
    fn test_print_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = PrintOperation::new("x"),
            inputs = [Array::scalar(3.0).unwrap()],
            expected = Array::scalar(3.0).unwrap(),
        );
    }

    #[test]
    fn test_print_batching() {
        check_operation_batching!(
            @exact,
            operation = PrintOperation::new("x"),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![1.0, 2.0]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![1.0, 2.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_print_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = PrintOperation::new("x"),
            cases = [{
                primals = [Array::scalar(2.0).unwrap()],
                tangents = [Array::scalar(3.0).unwrap()],
                primal_outputs = [Array::scalar(2.0).unwrap()],
                tangent_outputs = [Array::scalar(3.0).unwrap()],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = print [label=x] %0
                    in (%2, %1)
                "},
            }],
        );
    }

    #[test]
    fn test_print_differentiation_unused_output() {
        // A print whose output nothing consumes must not perturb the gradient: the JVP rule re-prints the primal and
        // passes the tangent through, so the effect rides the primal side of the linearization.
        let (value, gradient) = differentiate_at(Array::scalar(3.0).unwrap())
            .value_and_gradient(|input| {
                input.clone().print("x").unwrap();
                input.clone() * input
            })
            .unwrap();
        assert_eq!(value, Array::scalar(9.0).unwrap());
        assert_eq!(gradient, Array::scalar(6.0).unwrap());

        // Linearizing the same computation partitions its JVP program into primal and tangent stages: the dead print
        // survives in the primal stage through the partition projections' effect keep-alive, carrying its effect
        // class with it, and the tangent stage stays print-free.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |input: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| {
                input.clone().print("x").unwrap();
                Ok(input.clone() * input)
            },
            ArrayType::scalar(DataType::F64),
        )
        .unwrap();
        let linearization = program.to_flat_program().linearize().unwrap();
        let prints = |program: &Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>| {
            program
                .instructions()
                .iter()
                .filter(|instruction| matches!(instruction.operation(), ArrayOperation::Print(_)))
                .count()
        };
        assert_eq!(prints(linearization.primal()), 1);
        assert_eq!(linearization.primal().effects().classes(), EffectClasses::single(EffectClass::OrderedIo));
        assert_eq!(prints(linearization.tangent()), 0);
        assert_eq!(linearization.tangent().effects().classes(), EffectClasses::NONE);
    }

    #[test]
    fn test_print_transposition() {
        // Whole-program transposition rejects observable effects before calling the primitive rule.
        // Check directly that this rule passes the cotangent through without staging another print.
        let tracing = TracingContext::<Array, ArrayOperation<Array>>::new();
        let cotangent = tracing.input(ArrayType::scalar(DataType::F64));
        let mut context = TranspositionContext::new(tracing);
        let inputs = [PartialValue::Unknown(ArrayType::scalar(DataType::F64))];
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        assert_eq!(
            PrintOperation::new("x").transpose(
                &mut context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Value(cotangent.clone())],
                &accumulators,
            ),
            Ok(()),
        );
        let outputs = context.take_cotangents(&accumulators).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(matches!(&outputs[0], MaybeZero::Value(output) if output.atom_id() == cotangent.atom_id()));
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_print_staging() {
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |input: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| Ok(input.print("x")?),
            ArrayType::scalar(DataType::F64),
        )
        .unwrap();
        let program = program.to_flat_program();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = print [label=x] %0
                in (%1)"},
        );
        assert_eq!(program.effects().classes(), EffectClasses::single(EffectClass::OrderedIo));
    }

    #[test]
    fn test_print_staging_with_effect_class() {
        // The selected class is recorded on the staged instruction and summarized as the program's effect.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |input: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| {
                Ok(input.print_with_effect_class("x", EffectClass::UnorderedIo)?)
            },
            ArrayType::scalar(DataType::F64),
        )
        .unwrap();
        let program = program.to_flat_program();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = print [label=x, effect_class=unordered_io] %0
                in (%1)"},
        );
        assert_eq!(program.effects().classes(), EffectClasses::single(EffectClass::UnorderedIo));
    }
}
