//! Checked conversion from ordinary scalar-array data into first-class dimension authority.
//!
//! The user-facing semantics and example live on [`DimensionFromScalar`]. The operation records the produced
//! [`DimensionType`] directly so its identity and authoritative bounds remain one structural SSA definition.

use std::fmt::Display;

use ryft_macros::Parameter;

use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_differentiable_operation, impl_non_transposable_operation};
use crate::parameters::Parameter;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::ProgramError;
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::{ProjectedValue, Value};
use crate::types::{ArrayProgramType, ArrayType, DimensionType, DimensionVariable};

/// Canonical operation name for [`DimensionFromScalarOperation`].
pub const DIMENSION_FROM_SCALAR_OPERATION_NAME: &str = "dimension_from_scalar";

/// Converts ordinary rank-zero integer array data into a checked first-class dimension.
///
/// This is the explicit boundary that grants shape authority to numerical data. `result` declares the fresh
/// [`DimensionVariable`] and authoritative bounds of the produced dimension. Eager execution rejects negative,
/// out-of-bounds, host-unrepresentable, and backend-width-incompatible values before returning the dimension.
///
/// The input may use any signed or unsigned integer element type. It must be rank zero. A mapped scalar cannot pass
/// through this gateway under batching because that would create a different shape for each batch item and require a
/// ragged-array representation.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{
/// #     ArrayProgramValue, DimensionBounds, DimensionFromScalar, DimensionMul, DimensionValue, DimensionVariable,
/// #     ProgramError,
/// # };
/// # use ryft_core::backends::arrays::Array;
/// # fn main() -> Result<(), ProgramError> {
/// let scalar = ArrayProgramValue::Array(Array::scalar(5_i32));
/// let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
/// let dimension = scalar.to_dimension(batch)?;
/// let ArrayProgramValue::Dimension(dimension) = dimension else {
///     unreachable!("dimension_from_scalar always returns a dimension member");
/// };
/// assert_eq!(dimension.extent(), 5);
/// let doubled = dimension.mul(&DimensionValue::constant(2)?)?;
/// assert_eq!(doubled.extent(), 10);
/// # Ok(())
/// # }
/// ```
pub trait DimensionFromScalar<Output = Self>: Typed + Sized {
    /// Returns this rank-zero integer array as a first-class dimension described by `result`.
    fn to_dimension(&self, result: DimensionVariable) -> Result<Output, ProgramError>;
}

impl<V: Value<Type = ArrayProgramType>> DimensionFromScalar<V> for V
where
    V::DispatchDomain: Context<Type = ArrayProgramType>,
    <V::DispatchDomain as Domain>::Operation: From<DimensionFromScalarOperation>,
{
    fn to_dimension(&self, result: DimensionVariable) -> Result<V, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(DimensionFromScalarOperation::new(result), Vec::new(), std::slice::from_ref(self))?
            .remove(0))
    }
}

impl<V: Value<Type = ArrayProgramType>> DimensionFromScalar<V> for ProjectedValue<ArrayType, V>
where
    V::DispatchDomain: Context<Type = ArrayProgramType>,
    <V::DispatchDomain as Domain>::Operation: From<DimensionFromScalarOperation>,
{
    fn to_dimension(&self, result: DimensionVariable) -> Result<V, ProgramError> {
        Ok(self
            .value()
            .dispatch_domain()
            .bind(DimensionFromScalarOperation::new(result), Vec::new(), std::slice::from_ref(self.value()))?
            .remove(0))
    }
}

/// Mixed scalar-array-to-dimension operation used by [`DimensionFromScalar`].
///
/// Refer to [`DimensionFromScalar`] for semantic details and an example.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct DimensionFromScalarOperation {
    /// First-class dimension type produced by this operation.
    result_type: DimensionType,
}

impl DimensionFromScalarOperation {
    /// Creates a checked scalar-data gateway that produces a dimension described by `result`.
    #[inline]
    pub fn new(result: DimensionVariable) -> Self {
        Self { result_type: DimensionType::new(result) }
    }

    /// Returns the first-class dimension type produced by this operation.
    #[inline]
    pub fn result_type(&self) -> &DimensionType {
        &self.result_type
    }

    /// Validates the array member accepted by this gateway without allocating an inferred output collection.
    pub(crate) fn validate_input_type(input_type: &ArrayType) -> Result<(), TypeError> {
        if input_type.rank() != 0 || !input_type.data_type().is_integer() {
            return Err(TypeError::invalid(format!(
                "'{DIMENSION_FROM_SCALAR_OPERATION_NAME}' input must be a rank-0 integer array but has type \
                 {input_type}",
            )));
        }
        Ok(())
    }
}

impl Display for DimensionFromScalarOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as Operation<ArrayProgramType>>::render(self, formatter, 0)
    }
}

impl Operation<ArrayProgramType> for DimensionFromScalarOperation {
    #[inline]
    fn name(&self) -> &'static str {
        DIMENSION_FROM_SCALAR_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayProgramType],
        region_interfaces: &[RegionInterface<ArrayProgramType>],
    ) -> Result<Vec<ArrayProgramType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let input_type = <&ArrayType>::try_from(&input_types[0])?;
        Self::validate_input_type(input_type)?;
        Ok(vec![self.result_type.clone().into()])
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        Ok(Self { result_type: self.result_type.rename_identities(renaming)? })
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, DIMENSION_FROM_SCALAR_OPERATION_NAME)?
            .bracketed(|operation| operation.field("bounds", self.result_type.bounds()))
    }
}

impl<C: Domain<Type = ArrayProgramType, Value: DimensionFromScalar<C::Value>>> InterpretableOperation<C>
    for DimensionFromScalarOperation
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
        Ok(vec![inputs[0].to_dimension(self.result_type.variable().clone())?])
    }
}

impl<C: Context<Type = ArrayProgramType, Operation: From<DimensionFromScalarOperation>>>
    PartiallyEvaluatableOperation<C> for DimensionFromScalarOperation
{
}

impl_non_differentiable_operation!(DimensionFromScalarOperation);
impl_non_transposable_operation!(DimensionFromScalarOperation);

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
    use crate::types::dimensions::{DimensionBounds, DimensionError, MAX_DIMENSION_EXTENT};
    use crate::types::{DataType, Shape};

    use super::*;

    #[test]
    fn test_dimension_from_scalar() {
        let bounds = DimensionBounds::new(0, Some(9)).unwrap();
        let variable = DimensionVariable::new("extent", bounds);
        let operation = DimensionFromScalarOperation::new(variable.clone());
        let scalar_type = ArrayProgramType::Array(ArrayType::scalar(DataType::I32));

        assert_eq!(operation.name(), DIMENSION_FROM_SCALAR_OPERATION_NAME);
        assert_eq!(operation.result_type(), &DimensionType::new(variable.clone()));
        assert_eq!(operation.to_string(), "dimension_from_scalar [bounds=[0, 9)]");
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&scalar_type), &[]),
            Ok(vec![operation.result_type().clone().into()]),
        );
        for data_type in [
            DataType::I8,
            DataType::I16,
            DataType::I32,
            DataType::I64,
            DataType::U8,
            DataType::U16,
            DataType::U32,
            DataType::U64,
        ] {
            assert_eq!(
                operation.infer_output_types(&[ArrayType::scalar(data_type).into()], &[]),
                Ok(vec![operation.result_type().clone().into()]),
            );
        }
        assert_eq!(
            operation.infer_output_types(
                &[ArrayType::new(DataType::I32, Shape::new(vec![crate::Dimension::Static(1)])).into()],
                &[],
            ),
            Err(
                TypeError::invalid("'dimension_from_scalar' input must be a rank-0 integer array but has type i32[1]",)
            ),
        );
        assert_eq!(
            operation.infer_output_types(&[ArrayType::scalar(DataType::F32).into()], &[]),
            Err(TypeError::invalid("'dimension_from_scalar' input must be a rank-0 integer array but has type f32[]",)),
        );
        assert_eq!(
            operation.infer_output_types(&[operation.result_type().clone().into()], &[]),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        assert_eq!(operation.infer_output_types(&[], &[]), Err(TypeError::invalid("expected 1 input but got 0")));
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&scalar_type),
                &[RegionInterface::new(Vec::new(), Vec::new(), Effects::PURE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );

        let renamed_variable = DimensionVariable::new("renamed", bounds);
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(variable.clone(), renamed_variable.clone()).unwrap();
        let renamed_operation = operation.rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed_operation.result_type().variable(), &renamed_variable);
        assert_eq!(renamed_operation.result_type().bounds(), bounds);

        // Every integer element type uses the same checked conversion contract.
        for array in [
            Array::scalar(7_i8),
            Array::scalar(7_i16),
            Array::scalar(7_i32),
            Array::scalar(7_i64),
            Array::scalar(7_u8),
            Array::scalar(7_u16),
            Array::scalar(7_u32),
            Array::scalar(7_u64),
        ] {
            assert_eq!(array.to_dimension(variable.clone()).unwrap().extent(), 7);
        }
        assert_eq!(Array::scalar(0_i32).to_dimension(variable.clone()).unwrap().extent(), 0);
        let bounded_variable = DimensionVariable::new("bounded", DimensionBounds::new(2, Some(5)).unwrap());
        assert_eq!(Array::scalar(2_i32).to_dimension(bounded_variable.clone()).unwrap().extent(), 2);
        assert_eq!(Array::scalar(4_i32).to_dimension(bounded_variable.clone()).unwrap().extent(), 4);
        assert_eq!(
            Array::scalar(-1_i32).to_dimension(variable.clone()),
            Err(ProgramError::InvalidArgument {
                message: "'dimension_from_scalar' scalar input must be a nonnegative host-representable extent but is \
                          -1"
                .to_string(),
            }),
        );
        let error = Array::scalar(5_i32).to_dimension(bounded_variable.clone()).unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::BindingOutOfBounds {
                variable: "bounded".to_string(),
                value: 5,
                bounds: bounded_variable.bounds(),
            }),
        );
        if let Some(unrepresentable) = MAX_DIMENSION_EXTENT.checked_add(1) {
            let error = Array::scalar(u64::try_from(unrepresentable).unwrap())
                .to_dimension(DimensionVariable::new("wide", DimensionBounds::at_least(0)))
                .unwrap_err();
            assert_eq!(
                error.downcast_custom::<DimensionError>(),
                Some(&DimensionError::ExtentExceedsBackendWidth {
                    value: unrepresentable,
                    maximum: MAX_DIMENSION_EXTENT,
                }),
            );
        }

        let context = EagerContext::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        assert_eq!(
            context.bind(operation.clone(), Vec::new(), &[ArrayProgramValue::Array(Array::scalar(7_i32))],),
            Ok(vec![ArrayProgramValue::Dimension(DimensionValue::new(operation.result_type().clone(), 7).unwrap(),)]),
        );
        assert_eq!(
            context.bind(
                operation.clone(),
                Vec::new(),
                &[ArrayProgramValue::Dimension(DimensionValue::new(operation.result_type().clone(), 7).unwrap(),)],
            ),
            Err(TypeError::invalid("expected array type but got dimension type").into()),
        );
        check_operation_partial_evaluation!(
            backend = (ArrayProgramValue<Array>, ArrayProgramOperation<Array>),
            operation = operation.clone(),
            cases = [
                {
                    inputs = [
                        (@known, ArrayProgramValue::Array(Array::scalar(7_i32))),
                    ],
                    outputs = [
                        (@known, ArrayProgramValue::Dimension(
                            DimensionValue::new(operation.result_type().clone(), 7).unwrap()
                        )),
                    ],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(
                            type = scalar_type.clone(),
                            replay = ArrayProgramValue::Array(Array::scalar(7_i32))
                        )),
                    ],
                    outputs = [
                        (@residual, ArrayProgramValue::Dimension(
                            DimensionValue::new(operation.result_type().clone(), 7).unwrap()
                        )),
                    ],
                    residual_instructions = 1,
                },
            ],
        );

        type TestContext = TracingContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;
        let context = TestContext::new();
        let input = context.input(scalar_type);
        let input_id = input.atom_id().unwrap();
        let output = input.to_dimension(variable.clone()).unwrap();
        let output_id = output.atom_id().unwrap();
        let builder = context.builder().borrow();
        let [instruction] = builder.instructions() else {
            panic!("expected one dimension-from-scalar instruction");
        };
        assert_eq!(instruction.inputs(), &[input_id]);
        assert_eq!(instruction.outputs(), &[output_id]);
        assert!(instruction.regions().is_empty());
        assert!(matches!(instruction.operation(), ArrayProgramOperation::DimensionFromScalar(_)));
        assert_eq!(output.r#type().as_ref(), &ArrayProgramType::Dimension(DimensionType::new(variable.clone())));

        let program = builder
            .clone()
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output_id],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        drop(builder);
        assert_eq!(program.type_identity_signature().internal_identities(), std::slice::from_ref(&variable));

        let mut relocated_builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let relocated_input = relocated_builder.add_input(ArrayType::scalar(DataType::I32).into());
        let relocated_outputs = relocated_builder.splice_program(&program, &[relocated_input]).unwrap();
        let [relocated_instruction] = relocated_builder.instructions() else {
            panic!("expected one relocated dimension-from-scalar instruction");
        };
        assert_eq!(relocated_instruction.inputs(), &[relocated_input]);
        assert_eq!(relocated_instruction.outputs(), relocated_outputs.as_slice());
        let ArrayProgramOperation::DimensionFromScalar(relocated_operation) = relocated_instruction.operation() else {
            panic!("expected a relocated dimension-from-scalar operation");
        };
        assert_eq!(relocated_operation.result_type().variable(), &variable);
        assert_eq!(relocated_operation.result_type().bounds(), bounds);

        let projected_context = TestContext::new();
        let projected_input = projected_context.input(ArrayType::scalar(DataType::I32).into());
        let projected_input =
            <crate::Tracer<TestContext> as crate::ValueProjection<ArrayType>>::into_projected(projected_input).unwrap();
        let projected_variable = DimensionVariable::new("projected", bounds);
        let projected_output = projected_input.to_dimension(projected_variable.clone()).unwrap();
        assert_eq!(
            projected_output.r#type().as_ref(),
            &ArrayProgramType::Dimension(DimensionType::new(projected_variable)),
        );
        assert!(matches!(
            projected_context.builder().borrow().instructions(),
            [instruction] if matches!(instruction.operation(), ArrayProgramOperation::DimensionFromScalar(_)),
        ));

        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_ids().len(), 2);
        assert_eq!(jvp.output_ids().len(), 2);
        assert_eq!(
            jvp.outputs().last().unwrap().r#type().as_ref(),
            &ArrayProgramType::Array(ArrayType::scalar(DataType::Zero)),
        );

        let mut transposition_context = TestContext::new();
        assert!(matches!(
            <DimensionFromScalarOperation as TransposableOperation<
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
                if message == "operation `dimension_from_scalar` is not transposable",
        ));
    }
}
