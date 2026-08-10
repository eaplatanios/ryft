//! Reference [`Array`] answer to the custom-call operation family contract.
//!
//! The reference backend has no foreign-kernel registry, so it reports every custom call as an unsupported
//! operation instead of silently producing a value.

use crate::arrays::arrays::Array;
use crate::arrays::ir::ArrayIrValue;
use crate::arrays::operations::ArrayIrOperation;
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::dimensions::{Dimension, DimensionType};
use crate::arrays::types::ir::ArrayIrType;
use crate::contexts::EagerContext;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::custom_call::{CUSTOM_CALL_OPERATION_NAME, CustomCall, CustomCallOperation};
use crate::operations::dimensions::dimension_size::DimensionSize;
use crate::programs::{Operation, ProgramError, TypeError, Typed, Value, ValueProjection};

// TODO(eaplatanios): Review this.

impl CustomCall for Array {
    /// The reference array backend has no foreign-kernel registry, so custom calls always report an
    /// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
    fn custom_call<'a, I: IntoIterator<Item = &'a Self>>(
        operation: &CustomCallOperation<ArrayType>,
        _inputs: I,
    ) -> Result<Vec<Self>, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "the reference array backend cannot execute the foreign kernel '{}'",
                operation.target_name(),
            ),
        })
    }
}

impl<A: CustomCall + DimensionSize<usize> + Value<Type = ArrayType>>
    InterpretableOperation<EagerContext<ArrayIrValue<A>, ArrayIrOperation<A>>> for CustomCallOperation<ArrayIrType>
{
    fn interpret<D: InterpretationDriver<EagerContext<ArrayIrValue<A>, ArrayIrOperation<A>>>>(
        &self,
        _context: &EagerContext<ArrayIrValue<A>, ArrayIrOperation<A>>,
        driver: &D,
        inputs: &[ArrayIrValue<A>],
    ) -> Result<Vec<ArrayIrValue<A>>, ProgramError> {
        if driver.region_count() != 0 {
            return Err(TypeError::invalid(format!("expected 0 regions but got {}", driver.region_count())).into());
        }
        self.infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(), &[])?;
        let dynamic_output_dimension_count = self
            .output_types()
            .iter()
            .flat_map(|output_type| output_type.shape().dimensions())
            .filter(|dimension| matches!(dimension, Dimension::Dynamic(_)))
            .count();
        let array_input_count = inputs.len() - dynamic_output_dimension_count;
        let array_inputs = inputs[..array_input_count]
            .iter()
            .map(<ArrayIrValue<A> as ValueProjection<ArrayType>>::projected)
            .collect::<Result<Vec<_>, _>>()?;
        let output_extents = inputs[array_input_count..]
            .iter()
            .map(<ArrayIrValue<A> as ValueProjection<DimensionType>>::projected)
            .collect::<Result<Vec<_>, _>>()?;
        let kernel_operation = CustomCallOperation::<ArrayType>::from(self.clone());
        let outputs = A::custom_call(&kernel_operation, array_inputs.iter().copied())?;
        check_count!("output", outputs, self.output_types().len(), ProgramError);
        let mut output_extents = output_extents.into_iter();
        for (output_index, (output, output_type)) in outputs.iter().zip(self.output_types()).enumerate() {
            for (axis, dimension) in output_type.shape().dimensions().iter().enumerate() {
                if matches!(dimension, Dimension::Dynamic(_)) {
                    let expected_extent = output_extents.next().unwrap().extent();
                    let actual_extent = output.dimension_size(axis)?;
                    if actual_extent != expected_extent {
                        return Err(ProgramError::InvalidArgument {
                            message: format!(
                                "'{CUSTOM_CALL_OPERATION_NAME}' output {output_index} axis {axis} has extent \
                                 {actual_extent}, but its explicit extent operand is {expected_extent}",
                            ),
                        });
                    }
                }
            }
        }
        Ok(outputs.into_iter().map(ArrayIrValue::Array).collect())
    }
}
