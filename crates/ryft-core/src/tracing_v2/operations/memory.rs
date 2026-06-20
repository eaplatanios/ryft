use std::fmt::Display;

use half::{bf16, f16};

use crate::contexts::StagingContext;
use crate::differentiation::{Cotangent, Tangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::programs::{ProgramError, Value};
use crate::tracing::{AbstractTracingContext, Tracer};
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext};
use crate::types::{ArrayType, Memory, TypeError};

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

impl<V: Clone + Value<ArrayType> + TransferToMemory> InterpretableOperation<ArrayType, V>
    for TransferToMemoryOperation
{
    /// Interprets the transfer by delegating to the value-level [`TransferToMemory`] capability. Eager values keep
    /// their payload unchanged but must re-place their carried type in the destination [`Memory`], so that the
    /// interpreted value's type stays faithful to the instruction's declared output type.
    #[inline]
    fn interpret(
        &self,
        _context: &<V as Value<ArrayType>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].transfer_to_memory(self.destination)])
    }
}

/// Value-level memory-transfer capability. [`TransferToMemory`] fills the same role for
/// [`TransferToMemoryOperation`] that [`Sin`](crate::operations::trigonometric::Sin) fills for
/// [`SinOperation`](crate::operations::trigonometric::SinOperation): on concrete values it keeps the payload
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

impl<C: StagingContext<Operation: From<TransferToMemoryOperation>>> TransferToMemory for Tracer<C> {
    fn transfer_to_memory(&self, destination: Memory) -> Self {
        self.unary(TransferToMemoryOperation::new(destination))
    }
}

/// JVP rule for [`TransferToMemoryOperation`]: derivatives move along with the value, so the primal and the tangent
/// are both transferred to the destination (mirroring the JVP of `jax.device_put`). Symbolic zero tangents stay
/// symbolic with their types re-placed, so no transfer is staged for them. The staged linear transfer transposes
/// into a transfer that moves the cotangent back to the operand's source memory.
impl<D> DifferentiableOperation<D> for TransferToMemoryOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: TransferToMemory,
    LinearOperationOf<D>: From<TransferToMemoryOperation>,
{
    #[inline]
    fn jvp<'jvp>(
        &self,
        _context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        let tangent = match input.tangent().clone() {
            Tangent::Zero(r#type) => Tangent::Zero(r#type.with_memory(self.destination)),
            Tangent::Value(tangent) => Tangent::Value(tangent.transfer_to_memory(self.destination)),
        };
        Ok(vec![JvpTracer::new(input.primal().transfer_to_memory(self.destination), tangent)])
    }
}

/// Transpose rule for [`TransferToMemoryOperation`] (the `TransferToMemory` variant of
/// [`LinearArrayOperation`](crate::tracing_v2::LinearArrayOperation)). A memory transfer is the identity linear map
/// between two memories, so its transpose moves the output cotangent back to the operand's source memory by staging a
/// transfer to `input_types[0]`'s memory. Symbolic-zero cotangents propagate unchanged.
impl<V: Value<ArrayType>, O> TransposableOperation<ArrayType, V, O> for TransferToMemoryOperation
where
    O: Operation<ArrayType> + From<TransferToMemoryOperation>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
        input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError> {
        check_count!("input", input_types, 1, ProgramError);
        check_count!("output", output_cotangents, 1, ProgramError);
        match &output_cotangents[0] {
            Cotangent::Zero => Ok(vec![Cotangent::Zero]),
            Cotangent::Staged(cotangent) => {
                let outputs = context.stage_operation(
                    TransferToMemoryOperation::new(input_types[0].memory()),
                    std::slice::from_ref(cotangent),
                )?;
                check_count!("output", outputs, 1, ProgramError);
                Ok(vec![Cotangent::Staged(outputs.into_iter().next().unwrap())])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::contexts::EagerContext;
    use crate::tests::{TestArray, TestArrayDomain};
    use crate::tracing::trace;
    use crate::tracing_v2::batching::{ArrayBatch, BatchContext, BatchableOperation};
    use crate::tracing_v2::operations::dot::{Dot, DotDimensionNumbers};
    use crate::tracing_v2::test_util::assert_close;
    use crate::tracing_v2::{ArrayOperation, LinearArrayOperation, value_and_grad};
    use crate::types::{DataType, Shape, Size, Typed};

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
        let outputs = operation.interpret(&crate::EagerContext::new(), std::slice::from_ref(&input)).unwrap();
        assert_eq!(outputs, vec![input.transfer_to_memory(PINNED_HOST)]);
        assert_eq!(*outputs[0].r#type(), vector_type(2).with_memory(PINNED_HOST));
        assert_eq!(outputs[0].values, vec![1.0, 2.0]);
    }

    #[test]
    fn test_transfer_to_memory_staging_replaces_the_memory() {
        let (output_type, program) =
            trace(&TestArrayDomain, |x| Ok(x.transfer_to_memory(PINNED_HOST)), vector_type(2)).unwrap();
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
        let (primal, tangent) = TestArrayDomain
            .jvp(
                |x| x.transfer_to_memory(PINNED_HOST),
                TestArray::vector(vec![2.0, 3.0]),
                TestArray::vector(vec![1.0, 0.5]),
            )
            .unwrap();
        assert_eq!(primal.values, vec![2.0, 3.0]);
        assert_eq!(tangent.values, vec![1.0, 0.5]);

        // The staged pushforward transfers the tangent to the same destination as the primal.
        let (output, pushforward) = TestArrayDomain
            .linearize(|x| Ok(x.transfer_to_memory(PINNED_HOST)), TestArray::vector(vec![2.0, 3.0]))
            .unwrap();
        assert_eq!(output.values, vec![2.0, 3.0]);
        let program = pushforward.program();
        assert!(
            program
                .instructions()
                .iter()
                .any(|instruction| instruction.operation().name() == TRANSFER_TO_MEMORY_OPERATION_NAME),
            "expected the pushforward to stage a transfer_to_memory operation",
        );
        let output_types: Vec<_> = program.outputs().map(|atom| atom.r#type().into_owned()).collect();
        assert_eq!(output_types, vec![vector_type(2).with_memory(PINNED_HOST)]);
    }

    #[test]
    fn test_transfer_to_memory_transposition_moves_the_cotangent_back_to_the_source_memory() {
        let (output, pullback) = TestArrayDomain
            .vjp(|x| Ok(x.transfer_to_memory(PINNED_HOST)), TestArray::vector(vec![2.0, 3.0]))
            .unwrap();
        assert_eq!(output.values, vec![2.0, 3.0]);
        // The pullback consumes a pinned-host cotangent and transfers it back to the operand's source memory.
        let input_types: Vec<_> = pullback.inputs().map(|atom| atom.r#type().into_owned()).collect();
        assert_eq!(input_types, vec![vector_type(2).with_memory(PINNED_HOST)]);
        let destination = pullback
            .instructions()
            .iter()
            .find_map(|instruction| match instruction.operation() {
                LinearArrayOperation::TransferToMemory(operation) => Some(operation.destination()),
                _ => None,
            })
            .expect("expected the pullback to stage a transfer_to_memory transposition");
        assert_eq!(destination, Memory::Device);
        let output_types: Vec<_> = pullback.outputs().map(|atom| atom.r#type().into_owned()).collect();
        assert_eq!(output_types, vec![vector_type(2)]);
    }

    #[test]
    fn test_transfer_to_memory_round_trip_differentiates_like_the_identity() {
        let (value, gradient) = value_and_grad(
            &TestArrayDomain,
            |x| {
                let on_host = x.transfer_to_memory(Memory::Host { pinned: false });
                let back = on_host.transfer_to_memory(Memory::Device);
                back.dot(&back, &DotDimensionNumbers::inner_product())
            },
            TestArray::vector(vec![0.5, 1.5]),
        )
        .unwrap();
        assert_close(value.values[0], 2.5);
        assert_close(gradient.values[0], 1.0);
        assert_close(gradient.values[1], 3.0);
    }

    #[test]
    fn test_transfer_to_memory_batching_preserves_the_operation_and_the_memory() {
        // Value-level batching is the identity and preserves the lane axis.
        let input = ArrayBatch::mapped(TestArray::matrix(2, 3, vec![1.0; 6]), 0).unwrap();
        let operation =
            ArrayOperation::<ArrayType, TestArray>::TransferToMemory(TransferToMemoryOperation::new(PINNED_HOST));
        let context = EagerContext::<ArrayType, TestArray, ArrayOperation<ArrayType, TestArray>>::new();
        let outputs = operation.batch(&context, std::slice::from_ref(&input)).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].value(), input.value());

        // Traced batching stages the same transfer on the physical batched value with its lane axis preserved.
        let (output_type, program) = trace(
            &TestArrayDomain,
            |x| {
                let context = x.context().clone();
                Ok(BatchContext::batch(
                    &context,
                    |lane| Ok(lane.transfer_to_memory(PINNED_HOST)),
                    x,
                    Some(0),
                    Some(0),
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
