use std::fmt::Display;

use ryft_macros::Parameter;

use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_differentiable_operation, impl_non_transposable_operation};
use crate::parameters::Parameter;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::ProgramError;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{TypeError, Typed};
use crate::programs::values::{ProjectedValue, Value};
use crate::types::{ArrayProgramType, ArrayType, DataType, DimensionType};

/// Canonical element type used when first-class dimensions become ordinary array data.
///
/// Although dimensions are nonnegative, Ryft represents compiled dimension SSA as signed 64-bit integers and caps
/// [`MAX_DIMENSION_EXTENT`](crate::MAX_DIMENSION_EXTENT) accordingly. Therefore, unsigned 64-bit data would admit no
/// additional dimension values, would require a conversion instead of letting this gateway lower to an identity, and
/// would interact poorly with ordinary signed index and offset arithmetic.
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
/// # use ryft_core::{ArrayProgramValue, DimensionToScalar, DimensionValue, ProgramError};
/// # use ryft_core::backends::arrays::Array;
/// # fn main() -> Result<(), ProgramError> {
/// let dimension = ArrayProgramValue::<Array>::Dimension(DimensionValue::constant(3)?);
/// let scalar = dimension.to_scalar()?;
/// let ArrayProgramValue::Array(scalar) = scalar else {
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

impl<V: Value<Type = ArrayProgramType>> DimensionToScalar<V> for V
where
    V::DispatchDomain: Context<Type = ArrayProgramType>,
    <V::DispatchDomain as Domain>::Operation: From<DimensionToScalarOperation>,
{
    fn to_scalar(&self) -> Result<V, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(DimensionToScalarOperation, Vec::new(), std::slice::from_ref(self))?
            .remove(0))
    }
}

impl<V: Value<Type = ArrayProgramType>> DimensionToScalar<V> for ProjectedValue<DimensionType, V>
where
    V::DispatchDomain: Context<Type = ArrayProgramType>,
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
    type Type = ArrayProgramType;

    #[inline]
    fn name(&self) -> &'static str {
        DIMENSION_TO_SCALAR_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayProgramType],
        region_interfaces: &[RegionInterface<ArrayProgramType>],
    ) -> Result<Vec<ArrayProgramType>, TypeError> {
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

impl<C: Domain<Type = ArrayProgramType, Value: DimensionToScalar<C::Value>>> InterpretableOperation<C>
    for DimensionToScalarOperation
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let region_count = driver.region_count();
        if region_count != 0 {
            return Err(TypeError::invalid(format!("expected 0 regions but got {region_count}")).into());
        }
        self.infer_output_types(&[inputs[0].r#type().into_owned()], &[])?;
        Ok(vec![inputs[0].to_scalar()?])
    }
}

impl<C: Context<Type = ArrayProgramType, Operation: From<DimensionToScalarOperation>>> PartiallyEvaluatableOperation<C>
    for DimensionToScalarOperation
{
}

impl_non_differentiable_operation!(DimensionToScalarOperation);
impl_non_transposable_operation!(DimensionToScalarOperation);

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::array_programs::{ArrayProgramOperation, ArrayProgramValue};
    use crate::backends::arrays::Array;
    use crate::backends::dimensions::DimensionValue;
    use crate::contexts::{Context, EagerContext, StagingContext};
    use crate::differentiation::TransposableOperation;
    use crate::macros::check_operation_partial_evaluation;
    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::effects::Effects;
    use crate::programs::regions::{EmptyRegionDriver, RegionInterface};
    use crate::tracing::TracingContext;
    use crate::types::dimensions::{DimensionBounds, DimensionVariable, MAX_DIMENSION_EXTENT};

    use super::*;

    #[test]
    fn test_dimension_to_scalar() {
        let operation = DimensionToScalarOperation;
        let dimension_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap()));
        let scalar_type = ArrayProgramType::Array(ArrayType::scalar(DataType::I64));

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

        let context = EagerContext::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let input = ArrayProgramValue::Dimension(DimensionValue::new(dimension_type.clone(), 7).unwrap());
        assert_eq!(
            context.bind(operation, Vec::new(), &[input]),
            Ok(vec![ArrayProgramValue::Array(Array::scalar(7_i64))]),
        );
        assert_eq!(
            context.bind(operation, Vec::new(), &[ArrayProgramValue::Array(Array::scalar(7_i64))]),
            Err(TypeError::invalid("expected dimension type but got array type").into()),
        );
        check_operation_partial_evaluation!(
            backend = (ArrayProgramValue<Array>, ArrayProgramOperation<Array>),
            operation = operation,
            cases = [
                {
                    inputs = [
                        (@known, ArrayProgramValue::Dimension(
                            DimensionValue::new(dimension_type.clone(), 7).unwrap()
                        )),
                    ],
                    outputs = [
                        (@known, ArrayProgramValue::Array(Array::scalar(7_i64))),
                    ],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(
                            type = ArrayProgramType::Dimension(dimension_type.clone()),
                            replay = ArrayProgramValue::Dimension(
                                DimensionValue::new(dimension_type.clone(), 7).unwrap()
                            )
                        )),
                    ],
                    outputs = [
                        (@residual, ArrayProgramValue::Array(Array::scalar(7_i64))),
                    ],
                    residual_instructions = 1,
                },
            ],
        );

        type TestContext = TracingContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;
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
        assert!(matches!(instruction.operation(), ArrayProgramOperation::DimensionToScalar(_)));
        assert_eq!(output.r#type().as_ref(), &scalar_type);

        let program = builder
            .clone()
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output_id],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        drop(builder);

        let mut relocated_builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let relocated_input = relocated_builder.add_input(ArrayProgramType::Dimension(dimension_type.clone()));
        let relocated_outputs = relocated_builder.splice_program(&program, &[relocated_input]).unwrap();
        let [relocated_instruction] = relocated_builder.instructions() else {
            panic!("expected one relocated dimension-to-scalar instruction");
        };
        assert_eq!(relocated_instruction.inputs(), &[relocated_input]);
        assert_eq!(relocated_instruction.outputs(), relocated_outputs.as_slice());
        assert!(relocated_instruction.regions().is_empty());
        assert!(matches!(
            relocated_instruction.operation(),
            ArrayProgramOperation::DimensionToScalar(DimensionToScalarOperation),
        ));

        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_ids().len(), 1);
        assert_eq!(jvp.output_ids().len(), 1);
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 0);
        let pullback = linearization.pullback().unwrap();
        assert!(pullback.input_ids().is_empty());
        assert!(pullback.output_ids().is_empty());

        let mut transposition_context = TracingContext::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        assert!(matches!(
            <DimensionToScalarOperation as TransposableOperation<
                ArrayProgramValue<Array>,
                ArrayProgramOperation<Array>,
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
