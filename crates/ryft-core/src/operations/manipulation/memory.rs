use std::fmt::Display;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayType, Complex, Memory, RaggedAxis, bf16,
    f4e2m1fn, f6e2m3fn, f6e3m2fn, f8e3m4, f8e4m3, f8e4m3b11fnuz, f8e4m3fn, f8e4m3fnuz, f8e5m2, f8e5m2fnuz, f8e8m0fnu,
    f16, i1, i2, i4, u1, u2, u4,
};
use crate::batching::{BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{DifferentiableType, DifferentiationDual};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, ProgramError, RegionInterface, TypeError, Typed, Value,
};

// TODO(eaplatanios): Review this.

/// Canonical operation name for [`TransferToMemoryOperation`].
pub const TRANSFER_TO_MEMORY_OPERATION_NAME: &str = "transfer_to_memory";

/// [`Operation`] that moves its input into a destination [`Memory`].
///
/// Placement is metadata about _where_ a value lives, never about its contents, so this operation is shape- and
/// value-preserving: type inference returns the input type with its [`Memory`] replaced by the destination, and
/// interpretation for reference arrays keeps the payload unchanged while updating the value's carried type to the
/// destination so that interpreted values agree with the declared output types. Backends with a memory hierarchy lower
/// the staged operation into their native placement annotations (for example, XLA's device placement annotations
/// consumed by its host-offloading pipeline).
///
/// Differentiation moves derivatives along with the value: the JVP transfers the primal and the tangent to the
/// destination, and the staged linear transfer transposes into a transfer that moves the cotangent back to the
/// input's source memory (read off the input type during transposition).
#[derive(Copy, Clone, Debug)]
pub struct TransferToMemoryOperation {
    /// Destination [`Memory`] that the input is moved into.
    destination: Memory,
}

impl TransferToMemoryOperation {
    /// Creates a new [`TransferToMemoryOperation`] with the provided destination [`Memory`].
    pub fn new(destination: Memory) -> Self {
        Self { destination }
    }

    /// Returns the destination [`Memory`] that the input is moved into.
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

impl Operation for TransferToMemoryOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        TRANSFER_TO_MEMORY_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        Ok(vec![input_types[0].clone().with_memory(self.destination)])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("destination", self.destination))
    }
}

impl<C: Domain<Type = ArrayType, Value: TransferToMemory>> InterpretableOperation<C> for TransferToMemoryOperation {
    // Interprets the transfer by delegating to the value-level [`TransferToMemory`] capability. Eager values keep
    // their payload unchanged but must re-place their carried type in the destination [`Memory`], so that the
    // interpreted value's type stays faithful to the instruction's declared output type.
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].transfer_to_memory(self.destination)?])
    }
}

// Partial evaluation defers to the default fold-or-residualize behavior of
// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for TransferToMemoryOperation where
    C::Operation: From<TransferToMemoryOperation>
{
}

// Batching rule for [`TransferToMemoryOperation`]: memory placement is metadata that applies identically to every
// batch item, so the rule moves the packed value through the value-level [`TransferToMemory`] capability and
// preserves the input's batch axis. On traced values this stages the transfer on the batched physical value; on
// concrete values it keeps the payload unchanged while re-placing the carried type in the destination, exactly like
// interpretation.
impl<C: Context<Type = ArrayType, Value: TransferToMemory>, P: ArrayExtentBatchingPolicy<C>>
    BatchableOperation<C, ArrayBatchingPolicy<P>> for TransferToMemoryOperation
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        _context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        let value = inputs[0].value().transfer_to_memory(self.destination)?;
        let ragged_axes = inputs[0]
            .ragged_axes()
            .iter()
            .map(|ragged_axis| {
                Ok(RaggedAxis::new(
                    ragged_axis.axis(),
                    ragged_axis.extents().transfer_to_memory(self.destination)?,
                    ragged_axis.dimension().clone(),
                    ragged_axis.extent_axes().to_vec(),
                ))
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        Ok(vec![ArrayBatch::new(value, inputs[0].batch_axis())?.with_ragged_axes(ragged_axes)?].into())
    }
}

impl_differentiable_operation! {
    TransferToMemoryOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Operation: From<TransferToMemoryOperation>,
        C::Value: TransferToMemory,
    {
        |operation, _context, _driver, inputs| {
            // Forward-mode rule for [`TransferToMemoryOperation`]: a memory transfer is structural-linear, so the
            // tangent is transferred to the same destination as the primal, retaining symbolic zeros without allocation.
            check_count!("input", inputs, 1, ProgramError);
            let primal = inputs[0].primal().transfer_to_memory(operation.destination())?;
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                MaybeZero::Value(tangent) => MaybeZero::Value(tangent.transfer_to_memory(operation.destination())?),
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType> + From<TransferToMemoryOperation>,
    {
        |_operation, context, _driver, inputs, outputs, accumulators| {
            // Transpose rule for [`TransferToMemoryOperation`]. A memory transfer is the identity linear map between two
            // memories, so its transpose moves the output cotangent back to the input's source memory by staging a
            // transfer to `input_types[0]`'s memory. Symbolic-zero cotangents propagate unchanged.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 1, DifferentiationError);
            match &outputs[0] {
                MaybeZero::Zero(_) => Ok(()),
                MaybeZero::Value(cotangent) => {
                    let outputs = context.stage_operation(
                        TransferToMemoryOperation::new(inputs[0].r#type().memory()),
                        Vec::new(),
                        std::slice::from_ref(cotangent),
                    )?;
                    check_count!("output", outputs, 1, ProgramError);
                    {
                        let contribution = MaybeZero::Value(outputs.into_iter().next().unwrap());
                        accumulators[0].accumulate(context, contribution)?;
                        Ok(())
                    }
                }
            }
        }
    },
}

/// Transfers a value to a destination [`Memory`] without changing its elements, shape, layout, or sharding.
///
/// Reference [`Array`] values retain their shared host storage and update the memory recorded in their type. Traced
/// values stage a [`TransferToMemoryOperation`]; execution on a backend performs the transfer if the destination is
/// supported. Transfer and staging failures are returned to the caller. Native scalar values have no memory-placement
/// metadata and are returned unchanged.
///
/// # Examples
///
/// ```
/// # use ryft_core::{Array, ArrayType, DataType, Memory, ProgramError, TransferToMemory, Typed};
/// # fn main() -> Result<(), ProgramError> {
/// let input = Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[1i32, 2])?;
/// let output = input.transfer_to_memory(Memory::Host { pinned: true })?;
/// assert_eq!(output.r#type().memory(), Memory::Host { pinned: true });
/// assert_eq!(output.elements::<i32>()?, vec![1, 2]);
/// # Ok(())
/// # }
/// ```
pub trait TransferToMemory: Sized {
    /// Returns this value placed in `destination`, preserving its contents and all other type metadata.
    ///
    /// # Parameters
    ///
    ///   - `destination`: Memory space for the resulting value. Backend execution may fail if the requested memory
    ///     space is unavailable. Reference arrays model the placement in their type without physically moving storage.
    fn transfer_to_memory(&self, destination: Memory) -> Result<Self, ProgramError>;
}

// Context-carrying values bind through their owning context. The operation conversion bound keeps this disjoint
// from reference arrays, whose dispatch domain does not provide a memory-transfer operation.
impl<V: Value<Type = ArrayType>> TransferToMemory for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<TransferToMemoryOperation>,
{
    fn transfer_to_memory(&self, destination: Memory) -> Result<Self, ProgramError> {
        let outputs = self.dispatch_domain().bind(
            TransferToMemoryOperation::new(destination),
            Vec::new(),
            std::slice::from_ref(self),
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.into_iter().next().unwrap())
    }
}

/// Implements placement as an identity for scalar types that have no memory metadata.
macro_rules! impl_transfer_to_memory_identity {
    // Each listed scalar has no associated allocation to move.
    ($($element:ty),* $(,)?) => {
        $(impl TransferToMemory for $element {
            #[inline]
            fn transfer_to_memory(&self, _destination: Memory) -> Result<Self, ProgramError> {
                Ok(*self)
            }
        })*
    };
}

impl_transfer_to_memory_identity!(
    bool,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    i1,
    i2,
    i4,
    u1,
    u2,
    u4,
    f4e2m1fn,
    f6e2m3fn,
    f6e3m2fn,
    f8e3m4,
    f8e4m3,
    f8e4m3fn,
    f8e4m3fnuz,
    f8e4m3b11fnuz,
    f8e5m2,
    f8e5m2fnuz,
    f8e8m0fnu,
    bf16,
    f16,
    f32,
    f64,
    Complex<f32>,
    Complex<f64>,
);

impl TransferToMemory for Array {
    #[inline]
    fn transfer_to_memory(&self, destination: Memory) -> Result<Self, ProgramError> {
        // Reference storage stays host-resident. Updating only placement metadata retains exact physical bytes,
        // layout and sharding while making interpreted output types agree with staged output types.
        Ok(Self::new_unchecked(self.r#type().into_owned().with_memory(destination), self.shared_storage().clone()))
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::sync::Arc;

    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayBatch, ArrayElement, ArrayOperation, DataType, DimensionBounds, DimensionVariable, Layout,
        LogicalMesh, MeshAxis, MeshAxisType, RaggedAxis, Sharding, TiledLayout,
    };
    use crate::batching::{BatchAxis, BatchableOperation, BatchingContext, batch};
    use crate::contexts::EagerContext;
    use crate::differentiation::{
        DifferentiableOperation, DifferentiationContext, TransposableOperation, TranspositionContext, differentiate_at,
    };
    use crate::macros::{check_operation_partial_evaluation, dispatch_on_array_element_type};
    use crate::parameters::Parameter;
    use crate::partial::PartialValue;
    use crate::programs::{BindingRegionDriver, EffectClasses, EmptyRegionDriver, Provenance, ProvenanceScope, Typed};
    use crate::tracing::{Trace, TracingContext};

    use super::*;

    /// Destination shared by transfer fixtures.
    const PINNED_HOST: Memory = Memory::Host { pinned: true };

    #[test]
    fn test_transfer_to_memory() {
        let operation = TransferToMemoryOperation::new(PINNED_HOST);
        assert_eq!(operation.name(), TRANSFER_TO_MEMORY_OPERATION_NAME);
        assert_eq!(operation.destination(), PINNED_HOST);
        assert_eq!(operation.to_string(), "transfer_to_memory [destination=Host[Pinned]]");
    }

    #[test]
    fn test_transfer_to_memory_type_inference() {
        let operation = TransferToMemoryOperation::new(PINNED_HOST);
        let inferred = operation.infer_output_types(&[ArrayType::new_static(DataType::F64, [2])], &[]).unwrap();
        assert_eq!(inferred, vec![ArrayType::new_static(DataType::F64, [2]).with_memory(PINNED_HOST)]);
        assert_eq!(operation.infer_output_types(&[], &[]), Err(TypeError::invalid("expected 1 input but got 0")));
        assert_eq!(
            operation.infer_output_types(&[ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)], &[]),
            Err(TypeError::invalid("expected 1 input but got 2")),
        );
        assert_eq!(
            operation.infer_output_types(
                &[ArrayType::new_static(DataType::F64, [])],
                &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_transfer_to_memory_interpretation() {
        let operation = TransferToMemoryOperation::new(PINNED_HOST);
        // Reference arrays have no memory hierarchy, so interpretation keeps the payload unchanged while updating
        // the value's carried type in the destination so that it matches the declared output type.
        let input = Array::from_elements(ArrayType::new_static(DataType::F64, [2]), &[1.0f64, 2.0]).unwrap();
        let outputs = operation
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(outputs, vec![input.transfer_to_memory(PINNED_HOST).unwrap()]);
        assert_eq!(*outputs[0].r#type(), ArrayType::new_static(DataType::F64, [2]).with_memory(PINNED_HOST));
        assert_eq!(outputs[0].to_f64s(), vec![1.0, 2.0]);

        assert_eq!(
            operation.interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // Every element representation is preserved exactly, including encoded low-precision and complex values.
        for data_type in [
            DataType::Boolean,
            DataType::I1,
            DataType::I2,
            DataType::I4,
            DataType::I8,
            DataType::I16,
            DataType::I32,
            DataType::I64,
            DataType::U1,
            DataType::U2,
            DataType::U4,
            DataType::U8,
            DataType::U16,
            DataType::U32,
            DataType::U64,
            DataType::F4E2M1FN,
            DataType::F6E2M3FN,
            DataType::F6E3M2FN,
            DataType::F8E3M4,
            DataType::F8E4M3,
            DataType::F8E4M3FN,
            DataType::F8E4M3FNUZ,
            DataType::F8E4M3B11FNUZ,
            DataType::F8E5M2,
            DataType::F8E5M2FNUZ,
            DataType::F8E8M0FNU,
            DataType::BF16,
            DataType::F16,
            DataType::F32,
            DataType::F64,
            DataType::C64,
            DataType::C128,
        ] {
            dispatch_on_array_element_type!(data_type, |Element| {
                let element = Element::one().unwrap();
                assert_eq!(element.transfer_to_memory(PINNED_HOST), Ok(element));
                let input = Array::from_elements(ArrayType::new_static(data_type, [1]), &[element]).unwrap();
                for destination in [Memory::Device, Memory::Host { pinned: false }, PINNED_HOST] {
                    let output = input.transfer_to_memory(destination).unwrap();
                    assert_eq!(output.r#type().as_ref(), &input.r#type().clone().into_owned().with_memory(destination));
                    assert_eq!(output.elements::<Element>().unwrap(), vec![element]);
                    assert!(Arc::ptr_eq(input.shared_storage(), output.shared_storage()));
                }
            });
        }

        // Empty arrays and non-default physical layouts retain their exact storage representation.
        let empty = Array::from_elements::<i64>(ArrayType::new_static(DataType::I64, [0]), &[]).unwrap();
        assert_eq!(empty.transfer_to_memory(PINNED_HOST).unwrap().elements::<i64>().unwrap(), Vec::<i64>::new());

        let column_major = Array::from_elements(
            ArrayType::new_static(DataType::I32, [2, 2])
                .with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]))),
            &[1i32, 2, 3, 4],
        )
        .unwrap();
        let transferred = column_major.transfer_to_memory(PINNED_HOST).unwrap();
        assert_eq!(transferred.r#type().as_ref(), &column_major.r#type().into_owned().with_memory(PINNED_HOST));
        assert_eq!(transferred.elements::<i32>().unwrap(), vec![1, 2, 3, 4]);
        assert!(Arc::ptr_eq(column_major.shared_storage(), transferred.shared_storage()));

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded_type =
            ArrayType::new_static(DataType::I32, [2]).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        let sharded = Array::from_elements(sharded_type.clone(), &[1i32, 2]).unwrap();
        let transferred = sharded.transfer_to_memory(PINNED_HOST).unwrap();
        assert_eq!(transferred.r#type().as_ref(), &sharded_type.with_memory(PINNED_HOST));
        assert!(Arc::ptr_eq(sharded.shared_storage(), transferred.shared_storage()));

        // Tracing carries the same placement in the result type and stages one transfer.
        let (output_type, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |input| input.transfer_to_memory(PINNED_HOST),
            ArrayType::new_static(DataType::F64, [2]),
        )
        .unwrap();
        assert_eq!(output_type, ArrayType::new_static(DataType::F64, [2]).with_memory(PINNED_HOST));
        assert_eq!(program.instructions().len(), 1);
        let ArrayOperation::TransferToMemory(operation) = program.instructions()[0].operation() else {
            panic!("expected a staged transfer_to_memory operation");
        };
        assert_eq!(operation.destination(), PINNED_HOST);
        let output_types: Vec<_> = program.outputs().map(|atom| atom.r#type().into_owned()).collect();
        assert_eq!(output_types, vec![ArrayType::new_static(DataType::F64, [2]).with_memory(PINNED_HOST)]);
    }

    #[test]
    fn test_transfer_to_memory_interpretation_dispatch_errors() {
        /// Array wrapper that dispatches through a deliberately malformed context.
        #[derive(Clone, Debug)]
        struct DispatchArray {
            array: Array,
            output_count: usize,
        }

        impl Display for DispatchArray {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "{}", self.array)
            }
        }

        impl Parameter for DispatchArray {}

        impl Typed for DispatchArray {
            type Type = ArrayType;

            fn r#type(&self) -> Cow<'_, ArrayType> {
                self.array.r#type()
            }
        }

        impl Value for DispatchArray {
            type DispatchDomain = InvalidOutputContext;
            type ExecutionDomain = InvalidOutputContext;

            fn dispatch_domain(&self) -> Self::DispatchDomain {
                InvalidOutputContext(self.output_count)
            }

            fn execution_domain(&self) -> Self::ExecutionDomain {
                self.dispatch_domain()
            }
        }

        /// Context that violates the transfer output arity while accepting otherwise valid inputs.
        #[derive(Clone)]
        struct InvalidOutputContext(usize);

        impl Domain for InvalidOutputContext {
            type Type = ArrayType;
            type Value = DispatchArray;
            type Constant = Array;
            type Operation = TransferToMemoryOperation;
        }

        impl Context for InvalidOutputContext {
            fn lift(&self, array: Array) -> Result<DispatchArray, ProgramError> {
                Ok(DispatchArray { array, output_count: self.0 })
            }

            fn bind<O: Into<Self::Operation>, D: BindingRegionDriver<Self::Constant, Self::Operation>>(
                &self,
                _operation: O,
                _driver: D,
                inputs: &[DispatchArray],
            ) -> Result<Vec<DispatchArray>, ProgramError> {
                if self.0 == 1 {
                    return Err(ProgramError::InvalidArgument { message: "injected transfer failure".to_string() });
                }
                Ok(vec![inputs[0].clone(); self.0])
            }

            fn is_eager(&self) -> bool {
                true
            }

            fn provenance(&self) -> Provenance {
                Provenance::unknown()
            }

            fn invoke_with_provenance_origin<R, F: FnOnce() -> R>(&self, _origin: Provenance, function: F) -> R {
                function()
            }

            fn invoke_with_provenance_scope<R, F: FnOnce() -> R>(&self, _scope: ProvenanceScope, function: F) -> R {
                function()
            }
        }

        let array = Array::from_elements(ArrayType::new_static(DataType::I32, [1, 2]), &[3_i32, 7]).unwrap();
        let input = InvalidOutputContext(0).lift(array.clone()).unwrap();
        assert!(matches!(
            input.transfer_to_memory(PINNED_HOST),
            Err(ProgramError::InvalidOutputCount { expected: 1, actual: 0 }),
        ));
        let input = InvalidOutputContext(2).lift(array.clone()).unwrap();
        assert!(matches!(
            input.transfer_to_memory(PINNED_HOST),
            Err(ProgramError::InvalidOutputCount { expected: 1, actual: 2 }),
        ));
        let input = InvalidOutputContext(1).lift(array).unwrap();
        assert!(matches!(
            input.transfer_to_memory(PINNED_HOST),
            Err(ProgramError::InvalidArgument { message }) if message == "injected transfer failure",
        ));
    }

    #[test]
    fn test_transfer_to_memory_partial_evaluation() {
        let input = Array::from_elements(ArrayType::new_static(DataType::F64, [2]), &[1.0f64, 2.0]).unwrap();
        check_operation_partial_evaluation!(
            operation = TransferToMemoryOperation::new(PINNED_HOST),
            inputs = [input.clone()],
            expected = input.transfer_to_memory(PINNED_HOST).unwrap(),
        );
    }

    #[test]
    fn test_transfer_to_memory_batching() {
        // Batching over concrete values keeps the payload unchanged while re-placing the carried type in the
        // destination — exactly like interpretation — and preserves the batch axis.
        let input = {
            let value = Array::from_elements(ArrayType::new_static(DataType::F64, [2, 3]), &[1.0f64; 6]).unwrap();
            ArrayBatch::new(value, Some(0))
        }
        .unwrap();
        let operation = ArrayOperation::<Array>::TransferToMemory(TransferToMemoryOperation::new(PINNED_HOST));
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);
        let outputs =
            operation.batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input)).unwrap().into_parts().0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &input.value().transfer_to_memory(PINNED_HOST).unwrap());
        assert_eq!(outputs[0].r#type().memory(), PINNED_HOST);
        assert_eq!(outputs[0].value().to_f64s(), vec![1.0; 6]);

        // Memory placement changes neither logical geometry nor the values carrying per-item extents.
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let ragged_input = ArrayBatch::new(
            Array::from_elements(ArrayType::new_static(DataType::F64, [2, 3]), &[1.0f64; 6]).unwrap(),
            BatchAxis::new(0),
        )
        .unwrap()
        .with_ragged_axes(vec![RaggedAxis::new(
            1,
            Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[1i32, 3]).unwrap(),
            variable,
            vec![0],
        )])
        .unwrap();
        let ragged_outputs = operation
            .batch(&context, &EmptyRegionDriver, std::slice::from_ref(&ragged_input))
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(ragged_outputs[0].ragged_axes()[0].axis(), 1);
        assert_eq!(ragged_outputs[0].ragged_axes()[0].dimension(), ragged_input.ragged_axes()[0].dimension());
        assert_eq!(ragged_outputs[0].ragged_axes()[0].extent_axes(), &[0]);
        assert_eq!(ragged_outputs[0].ragged_axes()[0].extents().r#type().memory(), PINNED_HOST);
        assert_eq!(ragged_outputs[0].unbatched_type().memory(), PINNED_HOST);

        // Batching under a staging parent stages the same transfer on the physical batched value with its batch
        // axis preserved.
        let (output_type, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |x| {
                Ok(batch(|item| item.transfer_to_memory(PINNED_HOST), x, BatchAxis::new(0), BatchAxis::new(0), None)
                    .unwrap())
            },
            ArrayType::new_static(DataType::F64, [2, 3]),
        )
        .unwrap();
        assert_eq!(output_type, ArrayType::new_static(DataType::F64, [2, 3]).with_memory(PINNED_HOST));
        assert_eq!(program.instructions().len(), 1);
        let ArrayOperation::TransferToMemory(operation) = program.instructions()[0].operation() else {
            panic!("expected the batched program to stage a transfer_to_memory operation");
        };
        assert_eq!(operation.destination(), PINNED_HOST);
    }

    #[test]
    fn test_transfer_to_memory_differentiation() {
        // Eagerly the transfer is the identity on both the primal and the tangent.
        let (primal, tangent) =
            differentiate_at(Array::from_elements(ArrayType::new_static(DataType::F64, [2]), &[2.0f64, 3.0]).unwrap())
                .jvp(
                    Array::from_elements(ArrayType::new_static(DataType::F64, [2]), &[1.0f64, 0.5]).unwrap(),
                    |input| input.transfer_to_memory(PINNED_HOST),
                )
                .unwrap();
        assert_eq!(primal.elements::<f64>().unwrap(), vec![2.0, 3.0]);
        assert_eq!(tangent.elements::<f64>().unwrap(), vec![1.0, 0.5]);
        assert_eq!(primal.r#type().memory(), PINNED_HOST);
        assert_eq!(tangent.r#type().memory(), PINNED_HOST);

        // Symbolic tangents retain the destination type without staging a redundant allocation or transfer.
        let context = TracingContext::<Array, TransferToMemoryOperation>::new();
        let primal = context.input(ArrayType::new_static(DataType::F64, [2]));
        let duals = TransferToMemoryOperation::new(PINNED_HOST)
            .jvp(
                &DifferentiationContext::fused(context.clone()),
                &EmptyRegionDriver,
                &[DifferentiationDual::new_with_zero_tangent(primal).unwrap()],
            )
            .unwrap();
        assert!(duals[0].tangent().is_zero());
        assert_eq!(duals[0].tangent().r#type().memory(), PINNED_HOST);
        assert_eq!(context.builder().borrow().instructions().len(), 1);
    }

    #[test]
    fn test_transfer_to_memory_transposition() {
        let (output, pullback) =
            differentiate_at(Array::from_elements(ArrayType::new_static(DataType::F64, [2]), &[2.0f64, 3.0]).unwrap())
                .vjp(|input| input.transfer_to_memory(PINNED_HOST))
                .unwrap();
        let cotangent =
            Array::from_elements(ArrayType::new_static(DataType::F64, [2]).with_memory(PINNED_HOST), &[5.0f64, 7.0])
                .unwrap();
        let (pullback, residuals) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.interpret(vec![cotangent]),
            Ok(vec![Array::from_elements(ArrayType::new_static(DataType::F64, [2]), &[5.0f64, 7.0]).unwrap()]),
        );
        assert_eq!(output.elements::<f64>().unwrap(), vec![2.0, 3.0]);
        // The linear transfer carries no residual, so the direct-transpose pullback consumes only the pinned-host
        // cotangent and transfers it back to the input's source memory.
        assert!(residuals.is_empty(), "transfer_to_memory has no residual");
        let input_types: Vec<_> = pullback.inputs().map(|atom| atom.r#type().into_owned()).collect();
        assert_eq!(input_types, vec![ArrayType::new_static(DataType::F64, [2]).with_memory(PINNED_HOST)]);
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
        assert_eq!(output_types, vec![ArrayType::new_static(DataType::F64, [2])]);

        // A symbolic-zero cotangent preserves even a non-default source placement without staging a transfer.
        let source_type = ArrayType::new_static(DataType::F64, [2]).with_memory(Memory::Host { pinned: false });
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let mut rule_context = TranspositionContext::new(context.clone());
        let inputs = [PartialValue::Unknown(source_type.clone())];
        let accumulators = rule_context.cotangent_accumulators(&inputs, &[]).unwrap();
        TransferToMemoryOperation::new(PINNED_HOST)
            .transpose(
                &mut rule_context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Zero(source_type.clone().with_memory(PINNED_HOST).cotangent().unwrap())],
                &accumulators,
            )
            .unwrap();
        let contributions = rule_context.take_cotangents(&accumulators).unwrap();
        assert!(contributions[0].is_zero());
        assert_eq!(contributions[0].r#type().as_ref(), &source_type.cotangent().unwrap());
        assert!(context.builder().borrow().instructions().is_empty());
    }
}
