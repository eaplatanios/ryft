use std::fmt::Display;

use ryft_macros::Parameter;

use crate::arrays::{ArrayIrBatch, ArrayIrBatching, ArrayIrType, ArrayType, DataType, DimensionType};
use crate::batching::{BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_differentiable_operation, impl_non_transposable_operation};
use crate::parameters::Parameter;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    Operation, OperationFormatter, ProgramError, ProjectedValue, RegionInterface, TypeError, Typed, Value,
};

/// Canonical element type used when first-class dimensions become ordinary array data.
///
/// Although dimensions are nonnegative, Ryft represents compiled dimension SSA as signed 64-bit integers and caps
/// [`MAX_DIMENSION_EXTENT`](crate::arrays::MAX_DIMENSION_EXTENT) accordingly. Therefore, unsigned 64-bit data would
/// admit no additional dimension values, would require a conversion instead of letting this gateway lower to an
/// identity, and would interact poorly with ordinary signed index and offset arithmetic.
pub const RUNTIME_DIMENSION_DATA_TYPE: DataType = DataType::I64;

/// Canonical operation name for [`DimensionToScalarOperation`].
pub const DIMENSION_TO_SCALAR_OPERATION_NAME: &str = "dimension_to_scalar";

/// Converts a first-class dimension into ordinary rank-zero signed 64-bit array data.
///
/// This is the explicit boundary from a first-class dimension to numerical data. Composite program values produce an
/// array member in their parent carrier, while concrete dimension backends can select a concrete array representation
/// through `Output`. The returned scalar cannot define an array extent; converting it back into a first-class
/// dimension requires a separate checked gateway.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{ArrayIrValue, DimensionToScalar, DimensionValue, ProgramError};
/// # use ryft_core::arrays::Array;
/// # fn main() -> Result<(), ProgramError> {
/// let dimension = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(3)?);
/// let scalar = dimension.to_scalar()?;
/// let ArrayIrValue::Array(scalar) = scalar else {
///     unreachable!("dimension_to_scalar always returns an array member");
/// };
/// assert_eq!(scalar, Array::scalar(3_i64));
/// # Ok(())
/// # }
/// ```
pub trait DimensionToScalar<Output = Self>: Typed + Sized {
    /// Returns this dimension as ordinary rank-zero signed 64-bit array data represented by `Output`.
    fn to_scalar(&self) -> Result<Output, ProgramError>;
}

impl<V: Value<Type = ArrayIrType>> DimensionToScalar<V> for V
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<DimensionToScalarOperation>,
{
    fn to_scalar(&self) -> Result<V, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(DimensionToScalarOperation, Vec::new(), std::slice::from_ref(self))?
            .remove(0))
    }
}

impl<V: Value<Type = ArrayIrType>> DimensionToScalar<V> for ProjectedValue<DimensionType, V>
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<DimensionToScalarOperation>,
{
    fn to_scalar(&self) -> Result<V, ProgramError> {
        Ok(self
            .value()
            .dispatch_domain()
            .bind(DimensionToScalarOperation, Vec::new(), std::slice::from_ref(self.value()))?
            .remove(0))
    }
}

/// Mixed dimension-to-array operation used by [`DimensionToScalar`].
///
/// Refer to [`DimensionToScalar`] for semantic details and an example.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct DimensionToScalarOperation;

impl Display for DimensionToScalarOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for DimensionToScalarOperation {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        DIMENSION_TO_SCALAR_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        <&DimensionType>::try_from(&input_types[0])?;
        Ok(vec![ArrayType::scalar(RUNTIME_DIMENSION_DATA_TYPE).into()])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, DIMENSION_TO_SCALAR_OPERATION_NAME).map(|_| ())
    }
}

impl<C: Domain<Type = ArrayIrType, Value: DimensionToScalar<C::Value>>> InterpretableOperation<C>
    for DimensionToScalarOperation
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        self.infer_output_types(&[inputs[0].r#type().into_owned()], &[])?;
        Ok(vec![inputs[0].to_scalar()?])
    }
}

impl<C: Context<Type = ArrayIrType, Operation: From<DimensionToScalarOperation>>> PartiallyEvaluatableOperation<C>
    for DimensionToScalarOperation
{
}

/// Batching converts a replicated first-class dimension into one replicated scalar array. A mapped dimension already
/// stores its per-item extents as packed integer array data on the batch carrier, so conversion exposes that same value
/// as a mapped scalar array without staging another operation.
impl<C: Context<Type = ArrayIrType, Operation: From<DimensionToScalarOperation>>> BatchableOperation<C, ArrayIrBatching>
    for DimensionToScalarOperation
{
    fn batch<D: BatchingDriver<C, ArrayIrBatching>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatching>,
        _driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatching>, BatchingError> {
        let [input] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
        };
        if input.mapped_dimension_extents().is_some() {
            return Ok(vec![ArrayIrBatch::new(input.value().clone(), input.batch_axis())?].into());
        }
        input.validate_replicated_dimension()?;
        Ok(context
            .parent()
            .bind(*self, Vec::new(), std::slice::from_ref(input.value()))?
            .into_iter()
            .map(ArrayIrBatch::replicated)
            .collect::<Vec<_>>()
            .into())
    }
}

impl_non_differentiable_operation!(DimensionToScalarOperation);
impl_non_transposable_operation!(DimensionToScalarOperation);

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, DimensionBounds, DimensionValue, DimensionVariable, MAX_DIMENSION_EXTENT,
    };
    use crate::contexts::{Context, EagerContext, StagingContext};
    use crate::differentiation::TransposableOperation;
    use crate::macros::check_operation_partial_evaluation;
    use crate::parameters::Placeholder;
    use crate::programs::{Effects, EmptyRegionDriver, ProgramBuilder, RegionInterface};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_dimension_to_scalar() {
        let operation = DimensionToScalarOperation;
        let dimension_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap()));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::I64));

        assert_eq!(operation.name(), DIMENSION_TO_SCALAR_OPERATION_NAME);
        assert_eq!(operation.to_string(), DIMENSION_TO_SCALAR_OPERATION_NAME);
        assert_eq!(operation.infer_output_types(&[dimension_type.clone().into()], &[]), Ok(vec![scalar_type.clone()]),);
        assert_eq!(
            operation.infer_output_types(&[ArrayType::scalar(DataType::I64).into()], &[]),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );
        assert_eq!(operation.infer_output_types(&[], &[]), Err(TypeError::invalid("expected 1 input but got 0")),);
        assert_eq!(
            operation.infer_output_types(
                &[dimension_type.clone().into()],
                &[RegionInterface::new(Vec::new(), Vec::new(), Effects::PURE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );

        let zero = DimensionValue::new(dimension_type.clone(), 0).unwrap();
        assert_eq!(zero.to_scalar(), Ok(Array::scalar(0_i64)));
        let maximum_type = DimensionType::new(DimensionVariable::new(
            "maximum",
            DimensionBounds::new(0, Some(MAX_DIMENSION_EXTENT + 1)).unwrap(),
        ));
        let maximum = DimensionValue::new(maximum_type, MAX_DIMENSION_EXTENT).unwrap();
        assert_eq!(maximum.to_scalar(), Ok(Array::scalar(i64::try_from(MAX_DIMENSION_EXTENT).unwrap())));

        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = ArrayIrValue::Dimension(DimensionValue::new(dimension_type.clone(), 7).unwrap());
        assert_eq!(context.bind(operation, Vec::new(), &[input]), Ok(vec![ArrayIrValue::Array(Array::scalar(7_i64))]),);
        assert_eq!(
            context.bind(operation, Vec::new(), &[ArrayIrValue::Array(Array::scalar(7_i64))]),
            Err(TypeError::invalid("expected dimension type but got array type").into()),
        );
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = operation,
            cases = [
                {
                    inputs = [
                        (@known, ArrayIrValue::Dimension(
                            DimensionValue::new(dimension_type.clone(), 7).unwrap()
                        )),
                    ],
                    outputs = [
                        (@known, ArrayIrValue::Array(Array::scalar(7_i64))),
                    ],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(
                            type = ArrayIrType::Dimension(dimension_type.clone()),
                            replay = ArrayIrValue::Dimension(
                                DimensionValue::new(dimension_type.clone(), 7).unwrap()
                            )
                        )),
                    ],
                    outputs = [
                        (@residual, ArrayIrValue::Array(Array::scalar(7_i64))),
                    ],
                    residual_instructions = 1,
                },
            ],
        );

        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let context = TestContext::new();
        let input = context.input(dimension_type.clone().into());
        let input_id = input.atom_id().unwrap();
        let output = input.to_scalar().unwrap();
        let output_id = output.atom_id().unwrap();
        let builder = context.builder().borrow();
        let [instruction] = builder.instructions() else {
            panic!("expected one dimension-to-scalar instruction");
        };
        assert_eq!(instruction.inputs(), &[input_id]);
        assert_eq!(instruction.outputs(), &[output_id]);
        assert!(instruction.regions().is_empty());
        assert!(matches!(instruction.operation(), ArrayIrOperation::DimensionToScalar(_)));
        assert_eq!(output.r#type().as_ref(), &scalar_type);

        let program = builder
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output_id],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        drop(builder);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<extent ∈ [0, 9)> .
                let %1:i64[] = dimension_to_scalar %0
                in (%1)"},
        );

        let mut relocated_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let relocated_input = relocated_builder.add_input(ArrayIrType::Dimension(dimension_type.clone()));
        let relocated_outputs = relocated_builder.splice_program(&program, &[relocated_input]).unwrap();
        let [relocated_instruction] = relocated_builder.instructions() else {
            panic!("expected one relocated dimension-to-scalar instruction");
        };
        assert_eq!(relocated_instruction.inputs(), &[relocated_input]);
        assert_eq!(relocated_instruction.outputs(), relocated_outputs.as_slice());
        assert!(relocated_instruction.regions().is_empty());
        assert!(matches!(
            relocated_instruction.operation(),
            ArrayIrOperation::DimensionToScalar(DimensionToScalarOperation),
        ));
        let relocated = relocated_builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                relocated_outputs,
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(relocated.to_string(), program.to_string());

        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_ids().len(), 1);
        assert_eq!(jvp.output_ids().len(), 1);
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 0);
        let pullback = linearization.pullback().unwrap();
        assert!(pullback.input_ids().is_empty());
        assert!(pullback.output_ids().is_empty());

        let mut transposition_context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        assert!(matches!(
            <DimensionToScalarOperation as TransposableOperation<
                ArrayIrValue<Array>,
                ArrayIrOperation<Array>,
            >>::transpose(
                &operation,
                &mut transposition_context,
                &EmptyRegionDriver,
                &[],
                &[],
            ),
            Err(crate::DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `dimension_to_scalar` is not transposable",
        ));
    }
}
