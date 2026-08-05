//! First-class extraction of an array axis extent.
//!
//! The user-facing semantics and example live on [`DimensionSize`]. The capability's output type lets concrete
//! backends expose host shape metadata while composite program values produce first-class dimensions.

use std::fmt::Display;

use ryft_macros::Parameter;

use crate::axes::Axis;
use crate::backends::array_programs::batching::{ArrayIrBatch, ArrayIrBatching};
use crate::batching::{BatchableOperation, BatchingContext, BatchingDriver, BatchingError};
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
use crate::types::{
    ArrayIrType, ArrayType, Dimension, DimensionError, DimensionType, DimensionVariable, MAX_DIMENSION_EXTENT,
};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`DimensionSizeOperation`].
pub const DIMENSION_SIZE_OPERATION_NAME: &str = "dimension_size";

/// Reads the runtime extent of one array axis.
///
/// This is the equivalent of using an array shape component such as `x.shape[axis]`. The concrete representation is
/// selected by `Output`: a materialized array backend can return its host extent, while a composite program value
/// returns a dimension SSA value that can be passed explicitly to shape-carrying operations and combined with other
/// dimension operations. A program result is not an integer array; convert it to ordinary scalar data explicitly when
/// a numerical computation needs the extent as data.
///
/// In a composite trace, the capability works both on the outer array member and on a
/// [`ProjectedValue<ArrayType, V>`]. The projected form stages into and returns the parent composite carrier, allowing
/// an array operation result to feed shape computation without exposing an adapter conversion in user code.
///
/// Negative axes index from the final array axis. A dynamic selected axis preserves its existing
/// [`DimensionVariable`], while a static selected axis produces a fresh dimension with exact bounds.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{ArrayIrValue, DimensionSize, ProgramError};
/// # use ryft_core::backends::arrays::Array;
/// # fn main() -> Result<(), ProgramError> {
/// let array = ArrayIrValue::Array(Array::matrix(2, 3, vec![0.0; 6]));
/// let columns = array.dimension_size(-1)?;
/// let ArrayIrValue::Dimension(columns) = columns else {
///     unreachable!("dimension_size always returns a dimension member");
/// };
/// assert_eq!(columns.extent(), 3);
/// # Ok(())
/// # }
/// ```
pub trait DimensionSize<Output = Self>: Typed + Sized {
    /// Returns the runtime extent of `axis` in the representation selected by `Output`.
    fn dimension_size<AxisValue: Into<Axis>>(&self, axis: AxisValue) -> Result<Output, ProgramError>;
}

impl<V: Value<Type = ArrayIrType>> DimensionSize<V> for V
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<DimensionSizeOperation>,
{
    fn dimension_size<AxisValue: Into<Axis>>(&self, axis: AxisValue) -> Result<V, ProgramError> {
        let r#type = self.r#type();
        let input_type = <&ArrayType>::try_from(r#type.as_ref())?;
        let operation = DimensionSizeOperation::new(input_type, axis)?;
        Ok(self.dispatch_domain().bind(operation, Vec::new(), std::slice::from_ref(self))?.remove(0))
    }
}

impl<V: Value<Type = ArrayIrType>> DimensionSize<V> for ProjectedValue<ArrayType, V>
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<DimensionSizeOperation>,
{
    fn dimension_size<AxisValue: Into<Axis>>(&self, axis: AxisValue) -> Result<V, ProgramError> {
        let operation = DimensionSizeOperation::new(self.r#type().as_ref(), axis)?;
        Ok(self
            .value()
            .dispatch_domain()
            .bind(operation, Vec::new(), std::slice::from_ref(self.value()))?
            .remove(0))
    }
}

/// Mixed array-to-dimension operation used by [`DimensionSize`].
///
/// Refer to [`DimensionSize`] for semantic details and an example.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct DimensionSizeOperation {
    /// Normalized nonnegative input axis.
    axis: usize,

    /// Declared selected dimension against which later inference inputs are checked as refinements.
    input_dimension: Dimension,

    /// First-class dimension type produced by this operation.
    result_type: DimensionType,
}

impl DimensionSizeOperation {
    /// Creates an operation that reads `axis` of `input_type`.
    pub fn new<AxisValue: Into<Axis>>(input_type: &ArrayType, axis: AxisValue) -> Result<Self, TypeError> {
        let axis = axis.into();
        let position = axis.normalize(input_type.rank()).map_err(|_| {
            TypeError::invalid(format!(
                "'{DIMENSION_SIZE_OPERATION_NAME}' axis {axis} is out of bounds for rank {}",
                input_type.rank(),
            ))
        })?;
        let input_dimension = input_type.shape().dimensions()[position].clone();
        let minimum_extent = input_dimension.bounds().lower();
        if minimum_extent > MAX_DIMENSION_EXTENT {
            return Err(DimensionError::ExtentExceedsBackendWidth {
                value: minimum_extent,
                maximum: MAX_DIMENSION_EXTENT,
            }
            .into());
        }
        let result_variable = match &input_dimension {
            Dimension::Static(_) => DimensionVariable::new(format!("size(axis={position})"), input_dimension.bounds()),
            Dimension::Dynamic(variable) => variable.clone(),
        };
        Ok(Self { axis: position, input_dimension, result_type: DimensionType::new(result_variable) })
    }

    /// Returns the normalized nonnegative input axis.
    #[inline]
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// Returns the declared selected input dimension.
    #[inline]
    pub fn input_dimension(&self) -> &Dimension {
        &self.input_dimension
    }

    /// Returns the first-class dimension type produced by this operation.
    #[inline]
    pub fn result_type(&self) -> &DimensionType {
        &self.result_type
    }

    /// Validates one complete composite input type against this operation's selected declared axis.
    fn validate_input_type(&self, input_type: &ArrayIrType) -> Result<(), TypeError> {
        let input_type = <&ArrayType>::try_from(input_type)?;
        let actual_dimension = input_type.shape().dimensions().get(self.axis).ok_or_else(|| {
            TypeError::invalid(format!(
                "'{DIMENSION_SIZE_OPERATION_NAME}' axis {} is out of bounds for rank {}",
                self.axis,
                input_type.rank(),
            ))
        })?;
        let minimum_extent = actual_dimension.bounds().lower();
        if minimum_extent > MAX_DIMENSION_EXTENT {
            return Err(DimensionError::ExtentExceedsBackendWidth {
                value: minimum_extent,
                maximum: MAX_DIMENSION_EXTENT,
            }
            .into());
        }
        if self.input_dimension.is_refined_by(actual_dimension) {
            return Ok(());
        }
        if let (Dimension::Dynamic(variable), Dimension::Static(extent)) = (&self.input_dimension, actual_dimension)
            && !variable.bounds().contains(*extent)
        {
            return Err(DimensionError::BindingOutOfBounds {
                variable: variable.to_string(),
                value: *extent,
                bounds: variable.bounds(),
            }
            .into());
        }
        Err(TypeError::invalid(format!(
            "'{DIMENSION_SIZE_OPERATION_NAME}' input axis {} dimension {actual_dimension} does not refine declared \
             dimension {}",
            self.axis, self.input_dimension,
        )))
    }
}

impl Display for DimensionSizeOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for DimensionSizeOperation {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        DIMENSION_SIZE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        self.validate_input_type(&input_types[0])?;
        Ok(vec![self.result_type.clone().into()])
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        Ok(Self {
            axis: self.axis,
            input_dimension: self.input_dimension.rename_type_identities(renaming),
            result_type: self.result_type.rename_identities(renaming)?,
        })
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, DIMENSION_SIZE_OPERATION_NAME)?
            .bracketed(|operation| operation.field("axis", self.axis))
    }
}

impl<C: Domain<Type = ArrayIrType, Value: DimensionSize<C::Value>>> InterpretableOperation<C>
    for DimensionSizeOperation
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
        let input_type = inputs[0].r#type();
        self.validate_input_type(input_type.as_ref())?;
        Ok(vec![inputs[0].dimension_size(self.axis)?])
    }
}

impl<C: Context<Type = ArrayIrType, Operation: From<DimensionSizeOperation>>> PartiallyEvaluatableOperation<C>
    for DimensionSizeOperation
{
}

/// Batching reads the same logical array axis after accounting for an inserted packed batch axis. The resulting
/// first-class dimension is shared shape metadata and therefore remains replicated.
impl<C: Context<Type = ArrayIrType, Operation: From<DimensionSizeOperation>>> BatchableOperation<C, ArrayIrBatching>
    for DimensionSizeOperation
{
    fn batch<D: BatchingDriver<C, ArrayIrBatching>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatching>,
        _driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<Vec<ArrayIrBatch<C::Value>>, BatchingError> {
        let [input] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
        };
        let input_type = input.value().r#type();
        let batched_type = <&ArrayType>::try_from(input_type.as_ref())?;
        let packed_axis = match input.batch_axis().axis() {
            Some(batch_axis) => {
                let batch_axis = batch_axis.normalize(batched_type.rank())?;
                if self.axis() < batch_axis { self.axis() } else { self.axis() + 1 }
            }
            None => self.axis(),
        };
        let operation = Self::new(batched_type, packed_axis)?;
        Ok(context
            .parent()
            .bind(operation, Vec::new(), std::slice::from_ref(input.value()))?
            .into_iter()
            .map(ArrayIrBatch::replicated)
            .collect())
    }
}

impl_non_differentiable_operation!(DimensionSizeOperation);
impl_non_transposable_operation!(DimensionSizeOperation);

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::array_programs::{ArrayIrOperation, ArrayIrValue};
    use crate::backends::arrays::Array;
    use crate::contexts::{Context, EagerContext};
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationOutput, PartialValue};
    use crate::programs::effects::Effects;
    use crate::programs::operations::Operation;
    use crate::programs::regions::RegionInterface;
    use crate::programs::{ProgramBuilder, TypeIdentityRenaming};
    use crate::tracing::TracingContext;
    use crate::types::{DataType, DimensionBounds, Shape};

    use super::*;

    #[test]
    fn test_dimension_size_operation() {
        let bounds = DimensionBounds::new(2, Some(8)).unwrap();
        let variable = DimensionVariable::new("extent", bounds);
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable.clone())]));
        let operation = DimensionSizeOperation::new(&dynamic_type, -1).unwrap();

        // Dynamic axes preserve their identity and accept compatible static refinements.
        assert_eq!(operation.axis(), 0);
        assert_eq!(operation.input_dimension(), &Dimension::Dynamic(variable.clone()));
        assert_eq!(operation.result_type().variable(), &variable);
        assert_eq!(operation.to_string(), "dimension_size [axis=0]");
        assert_eq!(
            operation.infer_output_types(&[dynamic_type.clone().into()], &[]),
            Ok(vec![operation.result_type().clone().into()]),
        );
        assert_eq!(
            operation.infer_output_types(
                &[ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(5)])).into()],
                &[],
            ),
            Ok(vec![operation.result_type().clone().into()]),
        );

        // Static axes produce a fresh exact-bounds dimension.
        let static_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]));
        let static_operation = DimensionSizeOperation::new(&static_type, 0).unwrap();
        assert_eq!(static_operation.input_dimension(), &Dimension::Static(3));
        assert_eq!(static_operation.result_type().bounds(), DimensionBounds::new(3, Some(4)).unwrap());
        assert_ne!(static_operation.result_type().variable(), operation.result_type().variable());

        assert_eq!(
            DimensionSizeOperation::new(&static_type, 1),
            Err(TypeError::invalid("'dimension_size' axis 1 is out of bounds for rank 1")),
        );
        assert_eq!(
            DimensionSizeOperation::new(&static_type, -2),
            Err(TypeError::invalid("'dimension_size' axis -2 is out of bounds for rank 1")),
        );
        if let Some(unrepresentable) = MAX_DIMENSION_EXTENT.checked_add(1) {
            let unrepresentable_type =
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(unrepresentable)]));
            let expected = Err(DimensionError::ExtentExceedsBackendWidth {
                value: unrepresentable,
                maximum: MAX_DIMENSION_EXTENT,
            }
            .into());
            assert_eq!(DimensionSizeOperation::new(&unrepresentable_type, 0), expected);
            let unrepresentable_variable =
                DimensionVariable::new("unrepresentable", DimensionBounds::at_least(unrepresentable));
            let unrepresentable_dynamic_type =
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(unrepresentable_variable)]));
            assert_eq!(
                DimensionSizeOperation::new(&unrepresentable_dynamic_type, 0),
                Err(DimensionError::ExtentExceedsBackendWidth {
                    value: unrepresentable,
                    maximum: MAX_DIMENSION_EXTENT,
                }
                .into()),
            );
            assert_eq!(
                operation.infer_output_types(&[unrepresentable_type.into()], &[]),
                Err(DimensionError::ExtentExceedsBackendWidth {
                    value: unrepresentable,
                    maximum: MAX_DIMENSION_EXTENT,
                }
                .into()),
            );
        }

        // The mixed signature requires one array input, no regions, and a compatible selected dimension.
        let dimension_type = DimensionType::new(DimensionVariable::new("other", bounds));
        assert_eq!(operation.infer_output_types(&[], &[]), Err(TypeError::invalid("expected 1 input but got 0")),);
        assert_eq!(
            operation.infer_output_types(&[dimension_type.clone().into()], &[]),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        assert_eq!(
            operation.infer_output_types(
                &[dynamic_type.clone().into()],
                &[RegionInterface::new(Vec::new(), Vec::new(), Effects::PURE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
        assert_eq!(
            operation.infer_output_types(&[ArrayType::scalar(DataType::F32).into()], &[]),
            Err(TypeError::invalid("'dimension_size' axis 0 is out of bounds for rank 0")),
        );
        assert_eq!(
            operation.infer_output_types(
                &[ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(9)])).into()],
                &[],
            ),
            Err(DimensionError::BindingOutOfBounds { variable: "extent".to_string(), value: 9, bounds }.into()),
        );
        assert_eq!(
            operation.infer_output_types(
                &[ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(dimension_type.variable().clone())]),
                )
                .into()],
                &[],
            ),
            Err(TypeError::invalid(
                "'dimension_size' input axis 0 dimension other does not refine declared dimension extent",
            )),
        );
        assert_eq!(
            static_operation.infer_output_types(
                &[ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)])).into()],
                &[],
            ),
            Err(TypeError::invalid("'dimension_size' input axis 0 dimension 4 does not refine declared dimension 3",)),
        );

        // Renaming preserves the relationship between the selected dynamic input dimension and the result.
        let renamed = DimensionVariable::new("renamed", bounds);
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(variable, renamed.clone()).unwrap();
        let renamed_operation = operation.rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed_operation.input_dimension(), &Dimension::Dynamic(renamed.clone()));
        assert_eq!(renamed_operation.result_type().variable(), &renamed);

        // Eager execution reads shape metadata without consuming or copying the array payload.
        let reference_array = Array::matrix(2, 3, vec![0.0f32; 6]);
        assert_eq!(reference_array.dimension_size(-1), Ok(3));
        let array = ArrayIrValue::Array(Array::matrix(2, 3, vec![0.0f32; 6]));
        let payload = <ArrayIrValue<Array> as crate::ValueProjection<ArrayType>>::projected(&array)
            .unwrap()
            .values()
            .as_ptr();
        let result = array.dimension_size(-1).unwrap();
        let ArrayIrValue::Dimension(result) = result else {
            panic!("expected one dimension result");
        };
        assert_eq!(result.extent(), 3);
        assert_eq!(
            <ArrayIrValue<Array> as crate::ValueProjection<ArrayType>>::projected(&array)
                .unwrap()
                .values()
                .as_ptr(),
            payload,
        );

        let dimension = ArrayIrValue::<Array>::Dimension(crate::DimensionValue::new(dimension_type, 3).unwrap());
        assert_eq!(
            dimension.dimension_size(0),
            Err(ProgramError::Type(TypeError::invalid("expected array type but got dimension type"))),
        );

        // A dynamic staged declaration executes against its compatible concrete static refinement.
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let result =
            context.bind(operation, Vec::new(), &[ArrayIrValue::Array(Array::vector(vec![0.0f32; 5]))]).unwrap();
        let [ArrayIrValue::Dimension(result)] = result.as_slice() else {
            panic!("expected one dimension result");
        };
        assert_eq!(result.extent(), 5);
    }

    #[test]
    fn test_dimension_size_program() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let bounds = DimensionBounds::new(2, Some(8)).unwrap();
        let variable = DimensionVariable::new("extent", bounds);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable.clone())]));
        let (output_types, program) = TestContext::trace(
            |array| {
                let projected =
                    <crate::Tracer<TestContext> as crate::ValueProjection<ArrayType>>::into_projected(array.clone())?;
                Ok((array.dimension_size(0)?, projected.dimension_size(0)?))
            },
            ArrayIrType::from(input_type.clone()),
        )
        .unwrap();
        let (ArrayIrType::Dimension(first), ArrayIrType::Dimension(second)) = output_types else {
            panic!("expected two dimension result types");
        };
        assert_eq!(first.variable(), &variable);
        assert_eq!(second.variable(), &variable);
        assert!(program.type_identity_signature().internal_identities().is_empty());

        let [first_instruction, second_instruction] = program.instructions() else {
            panic!("expected two dimension-size instructions");
        };
        assert_eq!(first_instruction.inputs(), program.input_ids());
        assert_eq!(second_instruction.inputs(), program.input_ids());
        assert!(first_instruction.regions().is_empty());
        assert!(second_instruction.regions().is_empty());
        assert!(matches!(first_instruction.operation(), ArrayIrOperation::DimensionSize(_),));
        assert!(matches!(second_instruction.operation(), ArrayIrOperation::DimensionSize(_),));

        let concrete = ArrayIrValue::Array(Array::vector(vec![0.0f32; 5]));
        let (first, second) = program.interpret(concrete.clone()).unwrap();
        let (ArrayIrValue::Dimension(first), ArrayIrValue::Dimension(second)) = (first, second) else {
            panic!("expected two concrete dimension results");
        };
        assert_eq!(first.extent(), 5);
        assert_eq!(second.extent(), 5);

        // Relocation preserves both explicit reader edges and their shared forwarded identity.
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let outputs = builder.splice_program(&program, &[input]).unwrap();
        let imported = builder
            .build::<ArrayIrValue<Array>, (ArrayIrValue<Array>, ArrayIrValue<Array>)>(
                outputs,
                Placeholder,
                (Placeholder, Placeholder),
            )
            .unwrap();
        assert!(imported.type_identity_signature().internal_identities().is_empty());
        let output_types = imported.output_types();
        let [ArrayIrType::Dimension(first), ArrayIrType::Dimension(second)] = output_types.as_slice() else {
            panic!("expected two imported dimension result types");
        };
        assert_eq!(first.variable(), &variable);
        assert_eq!(second.variable(), &variable);
        assert_eq!(imported.instructions().len(), 2);
    }

    #[test]
    fn test_dimension_size_partial_evaluation() {
        type TestContext = TracingContext<ArrayIrValue<Array>, DimensionSizeOperation>;

        let variable = DimensionVariable::new("extent", DimensionBounds::new(2, Some(8)).unwrap());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let (_, program) =
            TestContext::trace(|array| array.dimension_size(0), ArrayIrType::from(input_type.clone())).unwrap();
        let program = program.to_flat_program();

        let known = program
            .partially_evaluate(&[PartialValue::Known(ArrayIrValue::Array(Array::vector(vec![0.0f32; 5])))])
            .unwrap();
        assert!(known.program().instructions().is_empty());
        let [PartialEvaluationOutput::Known(ArrayIrValue::Dimension(result))] = known.outputs() else {
            panic!("expected one known dimension result");
        };
        assert_eq!(result.extent(), 5);

        let unknown = program.partially_evaluate(&[PartialValue::Unknown(input_type.into())]).unwrap();
        assert_eq!(unknown.program().instructions().len(), 1);
        assert!(matches!(unknown.outputs(), [PartialEvaluationOutput::Unknown(_)],));
    }
}
