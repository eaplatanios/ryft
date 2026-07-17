use std::fmt::Display;

use crate::batching::BatchableOperation;
use crate::contexts::{Context, Domain};
use crate::differentiation::forward::{DifferentiableOperation, DifferentiationDriver, DifferentiationDual};
use crate::differentiation::reverse::{TransposableOperation, TranspositionDriver};
use crate::differentiation::{DifferentiableType, DifferentiationError};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::ProgramError;
use crate::programs::atoms::MaybeZero;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::Value;
use crate::tracing::{Tracer, TracingContext};
use crate::types::{ArrayType, DataType};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`ConvertElementTypeOperation`].
pub const CONVERT_ELEMENT_TYPE_OPERATION_NAME: &str = "convert_element_type";

/// Type descriptor whose values have a numeric element [`DataType`] that can be converted independently of the
/// descriptor's remaining structure and placement metadata.
pub trait ElementType: Type {
    /// Returns the element [`DataType`].
    fn element_type(&self) -> DataType;

    /// Returns this descriptor with its element [`DataType`] replaced by `data_type`.
    ///
    /// # Parameters
    ///
    ///   - `data_type`: New element data type.
    fn with_element_type(&self, data_type: DataType) -> Self;
}

impl ElementType for DataType {
    #[inline]
    fn element_type(&self) -> DataType {
        *self
    }

    #[inline]
    fn with_element_type(&self, data_type: DataType) -> Self {
        data_type
    }
}

impl ElementType for ArrayType {
    #[inline]
    fn element_type(&self) -> DataType {
        self.data_type()
    }

    #[inline]
    fn with_element_type(&self, data_type: DataType) -> Self {
        let mut output = self.clone();
        output.data_type = data_type;
        output
    }
}

/// Unary operation that converts the numeric element type of a value while preserving its shape and placement
/// metadata.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConvertElementTypeOperation {
    /// Element data type produced by this conversion.
    data_type: DataType,
}

impl ConvertElementTypeOperation {
    /// Creates a conversion to `data_type`.
    ///
    /// # Parameters
    ///
    ///   - `data_type`: Element data type produced by the operation.
    #[inline]
    pub fn new(data_type: DataType) -> Self {
        Self { data_type }
    }

    /// Returns the output element data type.
    #[inline]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }
}

impl Display for ConvertElementTypeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        OperationFormatter::new(formatter, 0, CONVERT_ELEMENT_TYPE_OPERATION_NAME)?
            .bracketed(|operation| operation.field("data_type", &self.data_type))
    }
}

impl<T: ElementType> Operation<T> for ConvertElementTypeOperation {
    #[inline]
    fn name(&self) -> &'static str {
        CONVERT_ELEMENT_TYPE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        _region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        if input_types[0].element_type() == DataType::Token || self.data_type == DataType::Token {
            return Err(TypeError { message: "cannot convert values to or from the token data type".to_string() });
        }
        Ok(vec![input_types[0].with_element_type(self.data_type)])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, CONVERT_ELEMENT_TYPE_OPERATION_NAME)?
            .bracketed(|operation| operation.field("data_type", &self.data_type))
    }
}

/// Value-level capability for converting a value's numeric element type.
pub trait ConvertElementType: Sized {
    /// Promotes this value's elements to `data_type`. Unlike [`Self::convert_element_type`], this method rejects
    /// conversions that are not permitted by the element-type promotion lattice.
    ///
    /// # Parameters
    ///
    ///   - `data_type`: Element data type to which the returned value is promoted.
    fn promote_element_type(&self, data_type: DataType) -> Result<Self, ProgramError>
    where
        Self: Typed,
        Self::Type: ElementType,
    {
        self.r#type()
            .element_type()
            .promote_to(data_type)
            .map_err(|error| TypeError { message: error.to_string() })?;
        self.convert_element_type(data_type)
    }

    /// Converts this value's elements to `data_type`.
    ///
    /// # Parameters
    ///
    ///   - `data_type`: Element data type of the returned value.
    fn convert_element_type(&self, data_type: DataType) -> Result<Self, ProgramError>;
}

impl<V> ConvertElementType for V
where
    V: Value,
    V::Type: ElementType,
    V::DispatchDomain: Context,
    <V::DispatchDomain as Domain>::Operation: From<ConvertElementTypeOperation>,
{
    #[inline]
    fn convert_element_type(&self, data_type: DataType) -> Result<Self, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(ConvertElementTypeOperation::new(data_type), Vec::new(), std::slice::from_ref(self))?
            .remove(0))
    }
}

impl<C> InterpretableOperation<C> for ConvertElementTypeOperation
where
    C: Domain,
    C::Type: ElementType,
    C::Value: ConvertElementType,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].convert_element_type(self.data_type)?])
    }
}

impl<C: Context> PartiallyEvaluatableOperation<C> for ConvertElementTypeOperation
where
    C::Type: ElementType,
    C::Operation: From<ConvertElementTypeOperation>,
{
}

impl<C: Context> DifferentiableOperation<C> for ConvertElementTypeOperation
where
    C::Type: DifferentiableType + ElementType,
    C::Value: ConvertElementType,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let primal = inputs[0].primal().convert_element_type(self.data_type)?;
        let primal_type = primal.r#type();
        let output_tangent_type = primal_type.tangent();
        let tangent = match (inputs[0].tangent(), output_tangent_type) {
            (_, None) => MaybeZero::Zero(primal_type.tangent_slot()),
            (MaybeZero::Zero(_), Some(tangent_type)) => MaybeZero::Zero(tangent_type),
            (MaybeZero::Value(tangent), Some(tangent_type)) => {
                MaybeZero::Value(tangent.convert_element_type(tangent_type.element_type())?)
            }
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

impl<V, O> TransposableOperation<V, O> for ConvertElementTypeOperation
where
    V: Value,
    V::Type: DifferentiableType + ElementType,
    O: Operation<V::Type> + From<ConvertElementTypeOperation>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        let input_cotangent_type =
            inputs[0].r#type().cotangent().ok_or_else(|| ProgramError::UnsupportedOperation {
                message: format!("'{CONVERT_ELEMENT_TYPE_OPERATION_NAME}' input 0 has no cotangent space"),
            })?;
        Ok(vec![match &outputs[0] {
            MaybeZero::Zero(_) => MaybeZero::Zero(input_cotangent_type),
            MaybeZero::Value(cotangent) => {
                MaybeZero::Value(cotangent.convert_element_type(input_cotangent_type.element_type())?)
            }
        }])
    }
}

// Conversion changes element type rather than participating in ordinary elementwise output-type broadcasting, so it
// has an explicit batching rule instead of implementing `ElementwiseOperation` and inheriting its type inference.
impl<C> BatchableOperation<C> for ConvertElementTypeOperation
where
    C: Context<Type = ArrayType>,
    C::Value: ConvertElementType,
{
    fn batch<D: crate::batching::BatchingDriver<C>>(
        &self,
        _context: &crate::batching::BatchingContext<C>,
        _driver: &D,
        inputs: &[crate::batching::ArrayBatch<C::Value>],
    ) -> Result<Vec<crate::batching::ArrayBatch<C::Value>>, crate::batching::BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        let mut output_type = inputs[0].r#type().into_owned();
        output_type.data_type = self.data_type;
        let output = inputs[0].value().convert_element_type(self.data_type)?;
        Ok(vec![crate::batching::ArrayBatch::new(output_type, output, inputs[0].batch_axis())?])
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::tests::TestArray;
    use crate::tracing_v2::{ArrayOperation, ForwardModeDifferentiate};
    use crate::types::{ArrayType, DataType};

    use super::ConvertElementType;

    #[test]
    fn test_convert_element_type_jvp_uses_differential_representations() {
        let context = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();

        let primal = TestArray::new(ArrayType::scalar(DataType::F8E8M0FNU), vec![2.0]);
        let tangent = TestArray::new(ArrayType::scalar(DataType::F32), vec![3.0]);
        let (output, output_tangent) =
            context.jvp(|value| value.convert_element_type(DataType::F32), primal, tangent).unwrap();
        assert_eq!(output.r#type, ArrayType::scalar(DataType::F32));
        assert_eq!(output_tangent, TestArray::new(ArrayType::scalar(DataType::F32), vec![3.0]));

        let primal = TestArray::new(ArrayType::scalar(DataType::F32), vec![2.0]);
        let tangent = TestArray::new(ArrayType::scalar(DataType::F32), vec![3.0]);
        let (_, output_tangent) =
            context.jvp(|value| value.convert_element_type(DataType::F8E8M0FNU), primal, tangent).unwrap();
        assert_eq!(output_tangent, TestArray::new(ArrayType::scalar(DataType::F32), vec![3.0]));
    }

    #[test]
    fn test_convert_element_type_jvp_converts_narrower_real_and_complex_tangents() {
        let context = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();

        let (_, tangent) = context
            .jvp(
                |value| value.convert_element_type(DataType::F32),
                TestArray::new(ArrayType::scalar(DataType::F64), vec![2.0]),
                TestArray::new(ArrayType::scalar(DataType::F64), vec![3.0]),
            )
            .unwrap();
        assert_eq!(tangent, TestArray::new(ArrayType::scalar(DataType::F32), vec![3.0]));

        let (_, tangent) = context
            .jvp(
                |value| value.convert_element_type(DataType::C64),
                TestArray::new(ArrayType::scalar(DataType::C128), vec![2.0]),
                TestArray::new(ArrayType::scalar(DataType::C128), vec![3.0]),
            )
            .unwrap();
        assert_eq!(tangent, TestArray::new(ArrayType::scalar(DataType::C64), vec![3.0]));
    }

    #[test]
    fn test_convert_element_type_jvp_is_zero_through_non_differentiable_types() {
        let context = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let primal = TestArray::new(ArrayType::scalar(DataType::F64), vec![2.75]);
        let tangent = TestArray::new(ArrayType::scalar(DataType::F64), vec![3.0]);
        let (output, output_tangent) = context
            .jvp(
                |value| value.convert_element_type(DataType::I32)?.convert_element_type(DataType::F64),
                primal,
                tangent,
            )
            .unwrap();
        assert_eq!(output.r#type, ArrayType::scalar(DataType::F64));
        assert_eq!(output_tangent, TestArray::new(ArrayType::scalar(DataType::F64), vec![0.0]));
    }
}
