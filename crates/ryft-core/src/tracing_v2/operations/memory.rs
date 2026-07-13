use std::fmt::Display;

use half::{bf16, f16};

use crate::batching::BatchingContext;
use crate::batching::BatchingError;
use crate::batching::{ArrayBatch, BatchableOperation};
use crate::contexts::Context;
use crate::contexts::Domain;
use crate::contexts::StagingContext;
use crate::differentiation::{DifferentiableOperation, DifferentiationError, TransposableOperation};
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::{Operation, OperationFormatter};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

use crate::differentiation::DifferentiationDual;
use crate::types::{ArrayType, Memory, TypeError, Typed};

/// Canonical operation name for [`TransferToMemoryOperation`].
pub const TRANSFER_TO_MEMORY_OPERATION_NAME: &'static str = "transfer_to_memory";

/// [`Operation`] that moves its operand into a destination [`Memory`] — the analogue of placing a value with
/// `jax.device_put` and a memory-kind-bearing sharding.
///
/// Placement is metadata about *where* a value lives, never about its contents, so this operation is shape- and
/// value-preserving: type inference returns the input type with its [`Memory`] replaced by the destination, and
/// interpretation in eager domains — which have no memory hierarchy — keeps the payload unchanged while re-placing
/// the value's carried type in the destination so that interpreted values stay faithful to the declared output
/// types. Backends that do have a memory hierarchy lower the staged operation into their native placement
/// annotations (for example, XLA's device placement annotations consumed by its host-offloading pipeline).
///
/// Differentiation moves derivatives along with the value: the JVP transfers the primal and the tangent to the
/// destination, and the staged linear transfer transposes into a transfer that moves the cotangent back to the
/// operand's source memory (read off the operand type during transposition).
#[derive(Copy, Clone, Debug)]
pub struct TransferToMemoryOperation {
    /// Destination [`Memory`] that the operand is moved into.
    destination: Memory,
}

impl TransferToMemoryOperation {
    /// Creates a new [`TransferToMemoryOperation`] with the provided destination [`Memory`].
    pub fn new(destination: Memory) -> Self {
        Self { destination }
    }

    /// Returns the destination [`Memory`] that the operand is moved into.
    #[inline]
    pub fn destination(&self) -> Memory {
        self.destination
    }
}

impl Display for TransferToMemoryOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for TransferToMemoryOperation {
    #[inline]
    fn name(&self) -> &'static str {
        TRANSFER_TO_MEMORY_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![input_types[0].clone().with_memory(self.destination)])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("destination", self.destination))
    }
}

impl<V: Clone + Value<Type = ArrayType> + TransferToMemory, C> InterpretableOperation<V, C>
    for TransferToMemoryOperation
{
    /// Interprets the transfer by delegating to the value-level [`TransferToMemory`] capability. Eager values keep
    /// their payload unchanged but must re-place their carried type in the destination [`Memory`], so that the
    /// interpreted value's type stays faithful to the instruction's declared output type.
    #[inline]
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].transfer_to_memory(self.destination)])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for TransferToMemoryOperation where
    C::Operation: From<TransferToMemoryOperation>
{
}

/// Value-level memory-transfer capability. [`TransferToMemory`] fills the same role for
/// [`TransferToMemoryOperation`] that [`Sin`](crate::operations::math::Sin) fills for
/// [`SinOperation`](crate::operations::math::SinOperation): on concrete values it keeps the payload
/// unchanged (eager domains have no memory hierarchy) while re-placing the carried type in the destination when
/// the value stores one, and on traced values it stages a transfer whose staged type carries the destination
/// [`Memory`]. Tracers only implement this capability when their operation type implements
/// [`From<TransferToMemoryOperation>`], so transfers over (for example) scalar staging contexts are type errors
/// rather than silent passthroughs.
pub trait TransferToMemory: Sized {
    /// Returns this value moved into the provided destination [`Memory`].
    fn transfer_to_memory(&self, destination: Memory) -> Self;
}

macro_rules! impl_transfer_to_memory_identity {
    ($($ty:ty),* $(,)?) => {
        $(
            impl TransferToMemory for $ty {
                #[inline]
                fn transfer_to_memory(&self, _destination: Memory) -> Self {
                    *self
                }
            }
        )*
    };
}

impl_transfer_to_memory_identity!(bf16, f16, f32, f64);

/// Any context-carrying value transfers to memory by binding a [`TransferToMemoryOperation`] through its own context.
/// The `From<TransferToMemoryOperation>` bound makes this disjoint from the eager value types (whose context operation
/// is `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete implementations.
impl<V: Value<Type = ArrayType>> TransferToMemory for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<TransferToMemoryOperation>,
{
    fn transfer_to_memory(&self, destination: Memory) -> Self {
        self.dispatch_domain()
            .bind(TransferToMemoryOperation::new(destination), &[], &[], &[self.clone()])
            .expect("`transfer_to_memory` operation failed")
            .remove(0)
    }
}

/// Batching rule for [`TransferToMemoryOperation`]: memory placement is metadata that applies identically to every
/// batch item, so the rule moves the packed value through the value-level [`TransferToMemory`] capability and
/// preserves the operand's batch axis. On traced values this stages the transfer on the batched physical value; on
/// concrete values it keeps the payload unchanged while re-placing the carried type in the destination, exactly like
/// interpretation.
impl<C: Context<Type = ArrayType, Value: TransferToMemory>> BatchableOperation<C> for TransferToMemoryOperation {
    fn batch(
        &self,
        _context: &BatchingContext<C>,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        let value = inputs[0].value().transfer_to_memory(self.destination);
        let physical_type = value.r#type().into_owned();
        Ok(vec![ArrayBatch::new(physical_type, value, inputs[0].batch_axis())?])
    }
}

/// Forward-mode rule for [`TransferToMemoryOperation`]: a memory transfer is structural-linear, so the tangent is
/// transferred to the same destination as the primal. The shared all-zero fast path handles a zero operand tangent
/// before this rule is consulted, so the operand tangent reaching here is always live.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for TransferToMemoryOperation
where
    C::Operation: Clone + From<TransferToMemoryOperation>,
    C::Value: TransferToMemory,
{
    fn jvp(
        &self,
        _context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let primal = inputs[0].primal().transfer_to_memory(self.destination());
        let tangent = match inputs[0].tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().into_owned()),
            MaybeZero::Value(tangent) => MaybeZero::Value(tangent.transfer_to_memory(self.destination())),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Transpose rule for [`TransferToMemoryOperation`]. A memory transfer is the identity linear map
/// between two memories, so its transpose moves the output cotangent back to the operand's source memory by staging a
/// transfer to `input_types[0]`'s memory. Symbolic-zero cotangents propagate unchanged.
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for TransferToMemoryOperation
where
    O: Operation<ArrayType> + From<TransferToMemoryOperation>,
{
    fn transpose(
        &self,
        context: &mut TracingContext<V, O>,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        match &outputs[0] {
            MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(inputs[0].r#type().into_owned())]),
            MaybeZero::Value(cotangent) => {
                let outputs = context.stage_operation(
                    TransferToMemoryOperation::new(inputs[0].r#type().memory()),
                    std::slice::from_ref(cotangent),
                )?;
                check_count!("output", outputs, 1, ProgramError);
                Ok(vec![MaybeZero::Value(outputs.into_iter().next().unwrap())])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::batching::ArrayBatch;
    use crate::batching::Batch;
    use crate::batching::BatchAxis;
    use crate::batching::BatchableOperation;
    use crate::contexts::EagerContext;
    use crate::differentiation::{ForwardModeDifferentiate, ReverseModeDifferentiate};
    use crate::tests::TestArray;
    use crate::tracing_v2::ArrayOperation;
    use crate::tracing_v2::operations::dot::{Dot, DotDimensionNumbers};
    use crate::types::{DataType, Shape, Size, Typed};

    use crate::tracing::Trace;

    use super::*;

    const PINNED_HOST: Memory = Memory::Host { pinned: true };

    fn vector_type(size: usize) -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(size)]))
    }

    fn matrix_type(rows: usize, columns: usize) -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(rows), Size::Static(columns)]))
    }

    #[test]
    fn test_transfer_to_memory_operation() {
        let operation = TransferToMemoryOperation::new(PINNED_HOST);
        assert_eq!(operation.name(), TRANSFER_TO_MEMORY_OPERATION_NAME);
        assert_eq!(operation.destination(), PINNED_HOST);
        assert_eq!(operation.to_string(), "transfer_to_memory [destination=Host[Pinned]]");
        let inferred = operation.infer_output_types(&[vector_type(2)]).unwrap();
        assert_eq!(inferred, vec![vector_type(2).with_memory(PINNED_HOST)]);
        assert!(operation.infer_output_types(&[]).is_err());
        // Eager domains have no memory hierarchy, so interpretation keeps the payload unchanged while re-placing
        // the value's carried type in the destination so that it matches the declared output type.
        let input = TestArray::vector(vec![1.0, 2.0]);
        let outputs =
            operation.interpret(&crate::EagerContext::<TestArray>::new(), std::slice::from_ref(&input)).unwrap();
        assert_eq!(outputs, vec![input.transfer_to_memory(PINNED_HOST)]);
        assert_eq!(*outputs[0].r#type(), vector_type(2).with_memory(PINNED_HOST));
        assert_eq!(outputs[0].values, vec![1.0, 2.0]);
    }

    #[test]
    fn test_transfer_to_memory_staging_replaces_the_memory() {
        let (output_type, program) = EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(
            |x| Ok(x.transfer_to_memory(PINNED_HOST)),
            vector_type(2),
        )
        .unwrap();
        assert_eq!(output_type, vector_type(2).with_memory(PINNED_HOST));
        assert_eq!(program.instructions().len(), 1);
        let ArrayOperation::TransferToMemory(operation) = program.instructions()[0].operation() else {
            panic!("expected a staged transfer_to_memory operation");
        };
        assert_eq!(operation.destination(), PINNED_HOST);
        let output_types: Vec<_> = program.outputs().map(|atom| atom.r#type().into_owned()).collect();
        assert_eq!(output_types, vec![vector_type(2).with_memory(PINNED_HOST)]);
    }

    #[test]
    fn test_transfer_to_memory_jvp_moves_the_primal_and_the_tangent() {
        // Eagerly the transfer is the identity on both the primal and the tangent.
        let (primal, tangent) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(
                |x| Ok(x.transfer_to_memory(PINNED_HOST)),
                TestArray::vector(vec![2.0, 3.0]),
                TestArray::vector(vec![1.0, 0.5]),
            )
            .unwrap();
        assert_eq!(primal.values, vec![2.0, 3.0]);
        assert_eq!(tangent.values, vec![1.0, 0.5]);
    }

    #[test]
    fn test_transfer_to_memory_transposition_moves_the_cotangent_back_to_the_source_memory() {
        let (output, pullback) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .vjp(|x| Ok(x.transfer_to_memory(PINNED_HOST)), TestArray::vector(vec![2.0, 3.0]))
            .unwrap();
        let (pullback, residuals) = pullback.into_parts();
        assert_eq!(output.values, vec![2.0, 3.0]);
        // The linear transfer carries no residual, so the direct-transpose pullback consumes only the pinned-host
        // cotangent and transfers it back to the operand's source memory.
        assert!(residuals.is_empty(), "transfer_to_memory has no residual");
        let input_types: Vec<_> = pullback.inputs().map(|atom| atom.r#type().into_owned()).collect();
        assert_eq!(input_types, vec![vector_type(2).with_memory(PINNED_HOST)]);
        let destination = pullback
            .instructions()
            .iter()
            .find_map(|instruction| match instruction.operation() {
                ArrayOperation::TransferToMemory(operation) => Some(operation.destination()),
                _ => None,
            })
            .expect("expected the pullback to stage a transfer_to_memory transposition");
        assert_eq!(destination, Memory::Device);
        let output_types: Vec<_> = pullback.outputs().map(|atom| atom.r#type().into_owned()).collect();
        assert_eq!(output_types, vec![vector_type(2)]);
    }

    #[test]
    fn test_transfer_to_memory_round_trip_differentiates_like_the_identity() {
        let (value, gradient) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .value_and_gradient(
                |x| {
                    let on_host = x.transfer_to_memory(Memory::Host { pinned: false });
                    let back = on_host.transfer_to_memory(Memory::Device);
                    back.dot(&back, &DotDimensionNumbers::inner_product())
                },
                TestArray::vector(vec![0.5, 1.5]),
            )
            .unwrap();
        assert_abs_diff_eq!(value.values[0], 2.5, epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.values[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.values[1], 3.0, epsilon = 1e-9);
    }

    #[test]
    fn test_transfer_to_memory_batching_preserves_the_operation_and_the_memory() {
        // Batching over concrete values keeps the payload unchanged while re-placing the carried type in the
        // destination — exactly like interpretation — and preserves the batch axis.
        let input = {
            let value = TestArray::matrix(2, 3, vec![1.0; 6]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let operation = ArrayOperation::<TestArray>::TransferToMemory(TransferToMemoryOperation::new(PINNED_HOST));
        let context = BatchingContext::new(EagerContext::<TestArray, ArrayOperation<TestArray>>::new(), 2, None);
        let outputs = operation.batch(&context, std::slice::from_ref(&input)).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &input.value().transfer_to_memory(PINNED_HOST));
        assert_eq!(outputs[0].r#type().memory(), PINNED_HOST);
        assert_eq!(outputs[0].value().values, vec![1.0; 6]);

        // Batching under a staging parent stages the same transfer on the physical batched value with its batch
        // axis preserved.
        let (output_type, program) = EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(
            |x| {
                let context = x.context().clone();
                Ok(Batch::batch(
                    &context,
                    |item| Ok(item.transfer_to_memory(PINNED_HOST)),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )
                .unwrap())
            },
            matrix_type(2, 3),
        )
        .unwrap();
        assert_eq!(output_type, matrix_type(2, 3).with_memory(PINNED_HOST));
        assert_eq!(program.instructions().len(), 1);
        let ArrayOperation::TransferToMemory(operation) = program.instructions()[0].operation() else {
            panic!("expected the batched program to stage a transfer_to_memory operation");
        };
        assert_eq!(operation.destination(), PINNED_HOST);
    }
}
