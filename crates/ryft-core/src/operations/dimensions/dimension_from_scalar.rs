use std::borrow::Cow;
use std::fmt::Display;

use ryft_macros::capability;

use crate::arrays::{
    Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrType, ArrayIrValue, ArrayType, DimensionType, DimensionValue,
    DimensionVariable,
};
use crate::batching::{BatchableOperation, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{
    check_count, impl_non_differentiable_operation, impl_non_transposable_operation,
    impl_reference_dischargeable_operation,
};
use crate::operations::Capability;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    Concretizable, EffectClass, EffectClasses, Effects, Operation, OperationFormatter, ProgramError, ProjectedValue,
    RegionInterface, Type, TypeError, TypeIdentityRenaming, Typed, Value, ValueDomainDispatch, ValueProjection,
};

/// Canonical operation name for [`DimensionFromScalarOperation`].
pub const DIMENSION_FROM_SCALAR_OPERATION_NAME: &str = "dimension_from_scalar";

/// Scalar-array-to-dimension operation used by [`DimensionFromScalar`].
/// Refer to [`DimensionFromScalar`] for semantic details and an example.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DimensionFromScalarOperation {
    /// Output [`DimensionType`] of this [`Operation`].
    output_type: DimensionType,
}

impl DimensionFromScalarOperation {
    /// Creates a new [`DimensionFromScalarOperation`] that decodes a dimension of type `output_type`. A static
    /// `output_type` (e.g., that of a stored residual whose extent is known) defines no identity, and decoding
    /// checks that the scalar equals its extent, just as it checks a symbolic output against its bounds.
    #[inline]
    pub fn new(output_type: DimensionType) -> Self {
        Self { output_type }
    }

    /// Returns the output [`DimensionType`] of this [`DimensionFromScalarOperation`].
    #[inline]
    pub fn output_type(&self) -> &DimensionType {
        &self.output_type
    }

    /// Validates the provided input [`ArrayType`].
    fn validate_input_type(input_type: &ArrayType) -> Result<(), TypeError> {
        if input_type.rank() != 0 || !input_type.data_type().is_integer() {
            return Err(TypeError::invalid(format!(
                "`{DIMENSION_FROM_SCALAR_OPERATION_NAME}` input must be a rank-0 integer array but has type \
                 `{input_type}`",
            )));
        }
        Ok(())
    }
}

impl Display for DimensionFromScalarOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for DimensionFromScalarOperation {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        DIMENSION_FROM_SCALAR_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let input_type = <&ArrayType>::try_from(&input_types[0])?;
        Self::validate_input_type(input_type)?;
        Ok(vec![self.output_type.clone().into()])
    }

    #[inline]
    fn effects(&self) -> Cow<'_, Effects> {
        Cow::Owned(Effects::explicit(EffectClasses::single(EffectClass::OrderedAssertion)))
    }

    #[inline]
    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        Ok(Self { output_type: self.output_type.rename_identities(renaming)? })
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, DIMENSION_FROM_SCALAR_OPERATION_NAME)?
            .bracketed(|operation| operation.field("bounds", self.output_type.bounds()))
    }
}

impl_reference_dischargeable_operation!(@reference_free DimensionFromScalarOperation);

impl<C: Domain<Type = ArrayIrType, Value: DimensionFromScalar<C::Value>>> InterpretableOperation<C>
    for DimensionFromScalarOperation
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        self.infer_output_types(&[inputs[0].r#type().into_owned()], &[])?;
        Ok(vec![inputs[0].to_dimension(self.output_type.to_variable())?])
    }
}

impl<C: Context<Type = ArrayIrType, Operation: From<DimensionFromScalarOperation>>> PartiallyEvaluatableOperation<C>
    for DimensionFromScalarOperation
{
}

impl<C: Context<Type = ArrayIrType, Operation: From<DimensionFromScalarOperation>>>
    BatchableOperation<C, ArrayIrBatchingPolicy> for DimensionFromScalarOperation
{
    fn batch<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        _driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<Vec<ArrayIrBatch<C::Value>>, BatchingError> {
        // Batching rejects mapped scalar inputs because converting one scalar per batch item would produce a different
        // first-class dimension for each item. A replicated scalar is converted once and remains replicated.
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];
        <&ArrayType>::try_from(&input.unbatched_type())?;
        if !input.batch_axis().is_replicated() {
            return Err(BatchingError::MappedDimension {
                r#type: Box::new(self.output_type().clone()),
                axis: input.batch_axis(),
            });
        }
        Ok(context
            .parent()
            .bind(self.clone(), Vec::new(), std::slice::from_ref(input.value()))?
            .into_iter()
            .map(ArrayIrBatch::replicated)
            .collect())
    }
}

impl_non_differentiable_operation!(DimensionFromScalarOperation);
impl_non_transposable_operation!(DimensionFromScalarOperation);

/// Capability to convert ordinary rank-zero integer [`Array`]s into [`DimensionValue`]s.
///
/// This is the explicit boundary that converts numerical data into a dimension value. `output` declares the fresh
/// [`DimensionVariable`] and authoritative bounds of the produced dimension. Eager execution rejects negative,
/// out-of-bounds, host-unrepresentable, and backend-width-incompatible values before returning the dimension.
///
/// The input may use any signed or unsigned integer element type. It must be rank zero. Under batching, a mapped
/// scalar is rejected because array shapes must be shared by every batch item.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values (including the projected array members of array IR values) implement this capability for [`ArrayType`] and
/// array IR values implement it for [`ArrayIrType`].
///
/// # Example
///
/// ```rust
/// # use ryft_core::{
/// #     Array, ArrayIrValue, DimensionBounds, DimensionFromScalar, DimensionValue, DimensionVariable, Mul,
/// #     ProgramError,
/// # };
/// # fn main() -> Result<(), ProgramError> {
/// let scalar = ArrayIrValue::Array(Array::scalar(5_i32).unwrap());
/// let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
/// let dimension = scalar.to_dimension(batch)?;
/// let ArrayIrValue::Dimension(dimension) = dimension else {
///     unreachable!("`dimension_from_scalar` always returns a dimension member");
/// };
/// assert_eq!(dimension.extent(), 5);
/// let doubled = dimension.mul(&DimensionValue::constant(2)?)?;
/// assert_eq!(doubled.extent(), 10);
/// # Ok(())
/// # }
/// ```
///
/// Extract vector elements with ordinary array operations before crossing this gateway. This keeps indexing a general
/// array concern and keeps this operation the numerical-data-to-scalar-dimension boundary:
///
/// ```rust
/// # use ryft_core::{
/// #     Array, DimensionBounds, DimensionFromScalar, DimensionVariable, ProgramError, Shape, Reshape, Slice,
/// # };
/// # fn main() -> Result<(), ProgramError> {
/// let extents = Array::vector(vec![3_i32, 5_i32]).unwrap();
/// let sequence = extents.slice(&[1], &[2], &[1])?.reshape(Shape::scalar())?;
/// let sequence = sequence.to_dimension(DimensionVariable::new(
///     "sequence",
///     DimensionBounds::new(1, Some(9))?,
/// ))?;
/// assert_eq!(sequence.extent(), 5);
/// # Ok(())
/// # }
/// ```
#[capability]
pub trait DimensionFromScalar<Output = Self, T = <Self as Capability>::Universe>: Capability + Sized {
    /// Returns this rank-zero integer array as a first-class dimension value described by `output`.
    fn to_dimension(&self, output: DimensionVariable) -> Result<Output, ProgramError>;
}

impl<
    V: Value<
            Dispatch = ValueDomainDispatch,
            Type = ArrayIrType,
            Domain: Context<Type = ArrayIrType, Operation: From<DimensionFromScalarOperation>>,
        >,
> DimensionFromScalar<V, ArrayIrType> for V
{
    fn to_dimension(&self, output: DimensionVariable) -> Result<V, ProgramError> {
        Ok(self
            .domain()
            .bind(DimensionFromScalarOperation::new(output.into()), Vec::new(), std::slice::from_ref(self))?
            .remove(0))
    }
}

impl DimensionFromScalar<DimensionValue, ArrayType> for Array {
    fn to_dimension(&self, output: DimensionVariable) -> Result<DimensionValue, ProgramError> {
        let operation = DimensionFromScalarOperation::new(output.into());
        DimensionFromScalarOperation::validate_input_type(self.r#type().as_ref())?;
        Ok(DimensionValue::new(operation.output_type().clone(), self.concretize()?)?)
    }
}

impl<A: Value<Type = ArrayType> + DimensionFromScalar<DimensionValue>> DimensionFromScalar<Self, ArrayIrType>
    for ArrayIrValue<A>
{
    fn to_dimension(&self, output: DimensionVariable) -> Result<Self, ProgramError> {
        let array = <Self as ValueProjection<ArrayType>>::projected(self)?;
        Ok(Self::Dimension(array.to_dimension(output)?))
    }
}

impl<
    V: Value<
            Dispatch = ValueDomainDispatch,
            Type = ArrayIrType,
            Domain: Context<Type = ArrayIrType, Operation: From<DimensionFromScalarOperation>>,
        >,
> DimensionFromScalar<V, ArrayType> for ProjectedValue<ArrayType, V>
{
    fn to_dimension(&self, output: DimensionVariable) -> Result<V, ProgramError> {
        Ok(self
            .value()
            .domain()
            .bind(DimensionFromScalarOperation::new(output.into()), Vec::new(), std::slice::from_ref(self.value()))?
            .remove(0))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, DataType, Dimension, DimensionBounds, DimensionError, DimensionValue,
        MAX_DIMENSION_EXTENT, Shape, i1, i2, i4, u1, u2, u4,
    };
    use crate::batching::{BatchAxis, batch};
    use crate::contexts::{Context, EagerContext, StagingContext};
    use crate::differentiation::{DifferentiationError, TransposableOperation, TranspositionContext};
    use crate::macros::{check_operation_partial_evaluation, check_operation_type_inference};
    use crate::parameters::Placeholder;
    use crate::programs::{EffectClasses, EmptyRegionDriver, ProgramBuilder, RegionInterface, ValueProjection};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_dimension_from_scalar() {
        let bounds = DimensionBounds::new(0, Some(9)).unwrap();
        let variable = DimensionVariable::new("extent", bounds);
        let operation = DimensionFromScalarOperation::new(variable.clone().into());

        assert_eq!(operation.name(), DIMENSION_FROM_SCALAR_OPERATION_NAME);
        assert_eq!(operation.output_type(), &DimensionType::from(variable.clone()));
        assert_eq!(operation.to_string(), "dimension_from_scalar [bounds=[0, 9)]");
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));

        let (output_type, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |input| input.to_dimension(variable.clone()),
            ArrayIrType::Array(ArrayType::scalar(DataType::I32)),
        )
        .unwrap();
        let [instruction] = program.instructions() else {
            panic!("expected one dimension-from-scalar instruction");
        };
        assert_eq!(instruction.inputs(), program.input_ids());
        assert_eq!(instruction.outputs(), program.output_ids());
        assert!(instruction.regions().is_empty());
        assert!(
            matches!(instruction.operation(), ArrayIrOperation::DimensionFromScalar(actual) if actual == &operation)
        );
        assert_eq!(output_type, ArrayIrType::Dimension(DimensionType::from(variable.clone())));

        assert_eq!(program.type_identity_signature().internal_identities(), std::slice::from_ref(&variable));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:i32[] .
                let %1:dimension<extent ∈ [0, 9)> = dimension_from_scalar [bounds=[0, 9)] %0
                in (%1)"},
        );
    }

    #[test]
    fn test_dimension_from_scalar_type_inference() {
        let bounds = DimensionBounds::new(0, Some(9)).unwrap();
        let variable = DimensionVariable::new("extent", bounds);
        let operation = DimensionFromScalarOperation::new(variable.clone().into());
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    input_types = [ArrayType::scalar(DataType::I1).into()],
                    output_types = [DimensionType::from(variable.clone()).into()],
                },
                {
                    input_types = [ArrayType::scalar(DataType::I2).into()],
                    output_types = [DimensionType::from(variable.clone()).into()],
                },
                {
                    input_types = [ArrayType::scalar(DataType::I4).into()],
                    output_types = [DimensionType::from(variable.clone()).into()],
                },
                {
                    input_types = [ArrayType::scalar(DataType::U1).into()],
                    output_types = [DimensionType::from(variable.clone()).into()],
                },
                {
                    input_types = [ArrayType::scalar(DataType::U2).into()],
                    output_types = [DimensionType::from(variable.clone()).into()],
                },
                {
                    input_types = [ArrayType::scalar(DataType::U4).into()],
                    output_types = [DimensionType::from(variable.clone()).into()],
                },
                {
                    input_types = [ArrayType::scalar(DataType::I8).into()],
                    output_types = [DimensionType::from(variable.clone()).into()],
                },
                {
                    input_types = [ArrayType::scalar(DataType::I16).into()],
                    output_types = [DimensionType::from(variable.clone()).into()],
                },
                {
                    input_types = [ArrayType::scalar(DataType::I32).into()],
                    output_types = [DimensionType::from(variable.clone()).into()],
                },
                {
                    input_types = [ArrayType::scalar(DataType::I64).into()],
                    output_types = [DimensionType::from(variable.clone()).into()],
                },
                {
                    input_types = [ArrayType::scalar(DataType::U8).into()],
                    output_types = [DimensionType::from(variable.clone()).into()],
                },
                {
                    input_types = [ArrayType::scalar(DataType::U16).into()],
                    output_types = [DimensionType::from(variable.clone()).into()],
                },
                {
                    input_types = [ArrayType::scalar(DataType::U32).into()],
                    output_types = [DimensionType::from(variable.clone()).into()],
                },
                {
                    input_types = [ArrayType::scalar(DataType::U64).into()],
                    output_types = [DimensionType::from(variable.clone()).into()],
                },
                {
                    input_types = [ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(1)])).into()],
                    error = "`dimension_from_scalar` input must be a rank-0 integer array but has type `i32[1]`",
                },
                {
                    input_types = [ArrayType::scalar(DataType::Boolean).into()],
                    error = "`dimension_from_scalar` input must be a rank-0 integer array but has type `bool[]`",
                },
                {
                    input_types = [ArrayType::scalar(DataType::F32).into()],
                    error = "`dimension_from_scalar` input must be a rank-0 integer array but has type `f32[]`",
                },
                {
                    input_types = [DimensionType::from(variable.clone()).into()],
                    error = "expected array type but got dimension type",
                },
                {
                    input_types = [],
                    error = "expected 1 input but got 0",
                },
            ],
        );
        assert_eq!(
            operation.infer_output_types(
                &[ArrayType::scalar(DataType::I32).into()],
                &[RegionInterface::new(Vec::new(), Vec::new(), EffectClasses::NONE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_dimension_from_scalar_rename_type_identities() {
        let bounds = DimensionBounds::new(0, Some(9)).unwrap();
        let variable = DimensionVariable::new("extent", bounds);
        let operation = DimensionFromScalarOperation::new(variable.clone().into());
        let renamed_variable = DimensionVariable::new("renamed", bounds);
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(variable.clone(), renamed_variable.clone()).unwrap();
        let renamed_operation = operation.rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed_operation.output_type().variable(), Some(&renamed_variable));
        assert_eq!(renamed_operation.output_type().bounds(), bounds);
    }

    #[test]
    fn test_dimension_from_scalar_rename_type_identities_splicing() {
        let bounds = DimensionBounds::new(0, Some(9)).unwrap();
        let variable = DimensionVariable::new("extent", bounds);
        let operation = DimensionFromScalarOperation::new(variable.clone().into());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let mut relocated_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let relocated_input = relocated_builder.add_input(ArrayType::scalar(DataType::I32).into());
        let relocated_outputs = relocated_builder.splice_program(&program, &[relocated_input]).unwrap();
        let [relocated_instruction] = relocated_builder.instructions() else {
            panic!("expected one relocated dimension-from-scalar instruction");
        };
        assert_eq!(relocated_instruction.inputs(), &[relocated_input]);
        assert_eq!(relocated_instruction.outputs(), relocated_outputs.as_slice());
        let ArrayIrOperation::DimensionFromScalar(relocated_operation) = relocated_instruction.operation() else {
            panic!("expected a relocated dimension-from-scalar operation");
        };
        assert_ne!(relocated_operation.output_type().variable(), Some(&variable));
        assert_eq!(relocated_operation.output_type().variable().unwrap().name(), variable.name());
        assert_eq!(relocated_operation.output_type().bounds(), bounds);

        // The relocated identity is fresh but retains the same bounds and relationships, so the relocated program
        // renders exactly like its source.
        let relocated = relocated_builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                relocated_outputs,
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(relocated.to_string(), program.to_string());
    }

    #[test]
    fn test_dimension_from_scalar_interpretation() {
        let operation = DimensionFromScalarOperation::new(
            DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap()).into(),
        );
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        assert_eq!(
            context.bind(operation.clone(), Vec::new(), &[ArrayIrValue::Array(Array::scalar(7_i32).unwrap())],),
            Ok(vec![ArrayIrValue::Dimension(DimensionValue::new(operation.output_type().clone(), 7).unwrap(),)]),
        );
        assert_eq!(
            context.bind(
                operation.clone(),
                Vec::new(),
                &[ArrayIrValue::Dimension(DimensionValue::new(operation.output_type().clone(), 7).unwrap(),)],
            ),
            Err(TypeError::invalid("expected array type but got dimension type").into()),
        );
    }

    #[test]
    fn test_dimension_from_scalar_partial_evaluation() {
        let bounds = DimensionBounds::new(0, Some(9)).unwrap();
        let variable = DimensionVariable::new("extent", bounds);
        let operation = DimensionFromScalarOperation::new(variable.clone().into());
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::I32));
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = operation.clone(),
            cases = [
                {
                    inputs = [
                        (@known, ArrayIrValue::Array(Array::scalar(7_i32).unwrap())),
                    ],
                    outputs = [
                        (@known, ArrayIrValue::Dimension(
                            DimensionValue::new(operation.output_type().clone(), 7).unwrap()
                        )),
                    ],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(
                            type = scalar_type.clone(),
                            replay = ArrayIrValue::Array(Array::scalar(7_i32).unwrap())
                        )),
                    ],
                    outputs = [
                        (@residual, ArrayIrValue::Dimension(
                            DimensionValue::new(operation.output_type().clone(), 7).unwrap()
                        )),
                    ],
                    residual_instructions = 1,
                },
            ],
        );
    }

    #[test]
    fn test_dimension_from_scalar_batching() {
        let variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(4)).unwrap());
        let operation = DimensionFromScalarOperation::new(variable.clone().into());
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            trace.clone(),
            trace.constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap())),
        );
        let input = ArrayIrBatch::new(trace.input(ArrayType::new_static(DataType::I32, [2]).into()), BatchAxis::new(0))
            .unwrap();
        assert_eq!(
            operation.batch(&context, &EmptyRegionDriver, &[input]).map(|_| ()),
            Err(BatchingError::MappedDimension {
                r#type: Box::new(operation.output_type().clone()),
                axis: BatchAxis::new(0),
            }),
        );
        assert!(trace.builder().borrow().instructions().is_empty());

        // Empty batches cannot make data-dependent dimensions representable either.
        let empty_context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            trace.clone(),
            trace.constant(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap())),
        );
        let empty = ArrayIrBatch::new(trace.input(ArrayType::new_static(DataType::I32, [0]).into()), BatchAxis::new(0))
            .unwrap();
        assert_eq!(
            operation.batch(&empty_context, &EmptyRegionDriver, &[empty]).map(|_| ()),
            Err(BatchingError::MappedDimension {
                r#type: Box::new(operation.output_type().clone()),
                axis: BatchAxis::new(0),
            }),
        );
        assert!(trace.builder().borrow().instructions().is_empty());

        let eager = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let input = ArrayIrBatch::replicated(ArrayIrValue::Array(Array::scalar(3i32).unwrap()));
        let outputs = operation.batch(&eager, &EmptyRegionDriver, &[input]).unwrap();
        assert_eq!(
            outputs,
            vec![ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::new(variable.into(), 3).unwrap(),))],
        );
    }

    #[test]
    fn test_dimension_from_scalar_batching_rejects_mapped_input() {
        // The public entry point surfaces the same rejection for a scalar that varies across batch items.
        let variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(4)).unwrap());
        let output: Result<ArrayIrValue<Array>, BatchingError> = batch(
            |input| input.to_dimension(variable.clone()),
            ArrayIrValue::Array(Array::vector(vec![1_i32, 3]).unwrap()),
            BatchAxis::new(0),
            BatchAxis::replicated(),
            None,
        );
        assert_eq!(
            output,
            Err(BatchingError::MappedDimension { r#type: Box::new(variable.into()), axis: BatchAxis::new(0) }),
        );
    }

    #[test]
    fn test_dimension_from_scalar_differentiation() {
        let bounds = DimensionBounds::new(0, Some(9)).unwrap();
        let variable = DimensionVariable::new("extent", bounds);
        let operation = DimensionFromScalarOperation::new(variable.clone().into());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let differentiated = program.jvp().unwrap();
        assert_eq!(differentiated.input_ids().len(), 1);
        assert_eq!(differentiated.output_ids().len(), 1);
        assert_eq!(differentiated.to_string(), program.to_string());
        assert_eq!(
            differentiated.interpret(vec![ArrayIrValue::Array(Array::scalar(7_i32).unwrap())]),
            Ok(vec![ArrayIrValue::Dimension(DimensionValue::new(DimensionType::from(variable), 7).unwrap())]),
        );
    }

    #[test]
    fn test_dimension_from_scalar_transposition() {
        let bounds = DimensionBounds::new(0, Some(9)).unwrap();
        let variable = DimensionVariable::new("extent", bounds);
        let operation = DimensionFromScalarOperation::new(variable.clone().into());
        let transposition_context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        assert!(matches!(
            <DimensionFromScalarOperation as TransposableOperation<
                ArrayIrValue<Array>,
                ArrayIrOperation<Array>,
            >>::transpose(
                &operation,
                &mut TranspositionContext::new(transposition_context),
                &EmptyRegionDriver,
                &[],
                &[],
                &[],
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `dimension_from_scalar` is not transposable",
        ));
    }

    #[test]
    fn test_dimension_from_scalar_to_dimension() {
        let bounds = DimensionBounds::new(0, Some(9)).unwrap();
        let variable = DimensionVariable::new("extent", bounds);
        let operation = DimensionFromScalarOperation::new(variable.clone().into());

        // Every integer element type preserves the output identity and checked extent.
        for (array, extent) in [
            (Array::scalar(7_i8).unwrap(), 7),
            (Array::scalar(7_i16).unwrap(), 7),
            (Array::scalar(7_i32).unwrap(), 7),
            (Array::scalar(7_i64).unwrap(), 7),
            (Array::scalar(7_u8).unwrap(), 7),
            (Array::scalar(7_u16).unwrap(), 7),
            (Array::scalar(7_u32).unwrap(), 7),
            (Array::scalar(7_u64).unwrap(), 7),
            (Array::scalar(i1::new(0).unwrap()).unwrap(), 0),
            (Array::scalar(i2::new(1).unwrap()).unwrap(), 1),
            (Array::scalar(i4::new(7).unwrap()).unwrap(), 7),
            (Array::scalar(u1::new(1).unwrap()).unwrap(), 1),
            (Array::scalar(u2::new(3).unwrap()).unwrap(), 3),
            (Array::scalar(u4::new(7).unwrap()).unwrap(), 7),
        ] {
            assert_eq!(
                array.to_dimension(variable.clone()),
                Ok(DimensionValue::new(operation.output_type().clone(), extent).unwrap()),
            );
        }
        assert_eq!(
            Array::scalar(0_i32).unwrap().to_dimension(variable.clone()),
            Ok(DimensionValue::new(DimensionType::from(variable.clone()), 0).unwrap()),
        );
        let bounded_variable = DimensionVariable::new("bounded", DimensionBounds::new(2, Some(5)).unwrap());
        assert_eq!(
            Array::scalar(2_i32).unwrap().to_dimension(bounded_variable.clone()),
            Ok(DimensionValue::new(DimensionType::from(bounded_variable.clone()), 2).unwrap()),
        );
        assert_eq!(
            Array::scalar(4_i32).unwrap().to_dimension(bounded_variable.clone()),
            Ok(DimensionValue::new(DimensionType::from(bounded_variable.clone()), 4).unwrap()),
        );
        assert_eq!(
            Array::scalar(-1_i32).unwrap().to_dimension(variable.clone()),
            Err(ProgramError::Concretization {
                message: "cannot extract a concrete `usize` from `i32[]`; value `-1` is out of range".to_string(),
            }),
        );

        let error = Array::scalar(1_i32).unwrap().to_dimension(bounded_variable.clone()).unwrap_err();
        assert!(matches!(
            error.downcast_custom::<DimensionError>(),
            Some(DimensionError::BindingOutOfBounds { variable, value: 1, bounds })
                if variable == "bounded" && *bounds == bounded_variable.bounds(),
        ));

        let error = Array::scalar(5_i32).unwrap().to_dimension(bounded_variable.clone()).unwrap_err();
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
                .unwrap()
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
    }

    #[test]
    fn test_dimension_from_scalar_to_dimension_projected() {
        let bounds = DimensionBounds::new(0, Some(9)).unwrap();
        let projected_context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let projected_input = projected_context.input(ArrayType::scalar(DataType::I32).into());
        let projected_input = ValueProjection::<ArrayType>::into_projected(projected_input).unwrap();
        let projected_variable = DimensionVariable::new("projected", bounds);
        let projected_output = projected_input.to_dimension(projected_variable.clone()).unwrap();
        assert_eq!(
            projected_output.r#type().as_ref(),
            &ArrayIrType::Dimension(DimensionType::from(projected_variable)),
        );
        assert!(matches!(
            projected_context.builder().borrow().instructions(),
            [instruction] if matches!(instruction.operation(), ArrayIrOperation::DimensionFromScalar(_)),
        ));
    }
}
