use std::fmt::Display;
use std::marker::PhantomData;

use crate::arrays::{ArrayIrType, ArrayType, DataType, DimensionType};
use crate::batching::{
    ArrayIrBatch, ArrayIrBatching, BatchableOperation, BatchingContext, BatchingDriver, BatchingError,
};
use crate::broadcasting::Broadcastable;
use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::ElementwiseOperation;
use crate::operations::manipulation::conversion::ElementType;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    Operation, OperationFormatter, ProgramError, ProjectedValue, RegionInterface, Type, TypeError, Value,
};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`CompareOperation`].
pub const COMPARE_OPERATION_NAME: &str = "compare";

/// Direction of the pairwise comparison performed by a [`CompareOperation`]. Each direction corresponds to one
/// comparison predicate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ComparisonDirection {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

impl Display for ComparisonDirection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Equal => "Equal",
            Self::NotEqual => "NotEqual",
            Self::LessThan => "LessThan",
            Self::LessThanOrEqual => "LessThanOrEqual",
            Self::GreaterThan => "GreaterThan",
            Self::GreaterThanOrEqual => "GreaterThanOrEqual",
        })
    }
}

/// [`Operation`] that performs pairwise comparisons in the `T` type universe. [`DataType`] and [`ArrayType`]
/// instantiations provide homogeneous elementwise comparison, while [`ArrayIrType`] provides the mixed
/// first-class-dimension comparison whose Boolean predicate is ordinary array data. Refer to [`Compare`] for the
/// corresponding value-level semantics.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct CompareOperation<T: Type> {
    /// [`ComparisonDirection`] used by this [`CompareOperation`].
    direction: ComparisonDirection,

    /// Type universe whose comparison contract this payload represents.
    type_marker: PhantomData<T>,
}

impl<T: Type> Copy for CompareOperation<T> {}

impl<T: Type> Clone for CompareOperation<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Type> CompareOperation<T> {
    /// Creates a new [`CompareOperation`] with the provided [`ComparisonDirection`].
    #[inline]
    pub fn new(direction: ComparisonDirection) -> Self {
        Self { direction, type_marker: PhantomData }
    }

    /// Returns the [`ComparisonDirection`] used by this [`CompareOperation`].
    #[inline]
    pub fn direction(&self) -> ComparisonDirection {
        self.direction
    }
}

impl<T: Type> Display for CompareOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        OperationFormatter::new(formatter, 0, COMPARE_OPERATION_NAME)?
            .bracketed(|operation| operation.field("direction", self.direction))
    }
}

/// Homogeneous comparison contract: the two operands are broadcast together and the broadcasted element type is
/// replaced by [`DataType::Boolean`]. This covers every element-bearing type universe, including [`DataType`] and
/// [`ArrayType`].
impl<T: Broadcastable + ElementType> Operation for CompareOperation<T> {
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        COMPARE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        _region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 2, TypeError);

        // Complex operands are unordered, so only the equality comparison directions are defined for them.
        if !matches!(self.direction, ComparisonDirection::Equal | ComparisonDirection::NotEqual)
            && input_types.iter().any(|input_type| input_type.is_complex())
        {
            return Err(TypeError::invalid(format!(
                "cannot apply an ordered comparison to unordered complex operands of types {} and {}",
                input_types[0], input_types[1],
            )));
        }

        let broadcasted = T::broadcasted(input_types)
            .map_err(|_| TypeError::invalid("comparison input types are not broadcast-compatible".to_string()))?;
        Ok(vec![broadcasted.with_element_type(DataType::Boolean)])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, COMPARE_OPERATION_NAME)?
            .bracketed(|operation| operation.field("direction", self.direction))
    }
}

/// Composite comparison contract: both operands are first-class dimensions and the predicate is ordinary rank-zero
/// Boolean array data rather than another dimension value.
impl Operation for CompareOperation<ArrayIrType> {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        COMPARE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        input_types.iter().try_for_each(|r#type| <&DimensionType>::try_from(r#type).map(|_| ()))?;
        // Comparing first-class dimensions produces ordinary predicate data rather than another dimension value.
        Ok(vec![ArrayType::scalar(DataType::Boolean).into()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, COMPARE_OPERATION_NAME)?
            .bracketed(|operation| operation.field("direction", self.direction))
    }
}

impl ElementwiseOperation for CompareOperation<ArrayType> {
    #[inline]
    fn input_count(&self) -> usize {
        2
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        Operation::infer_output_types(self, input_types, &[])
    }
}

impl<C: Domain> InterpretableOperation<C> for CompareOperation<C::Type>
where
    CompareOperation<C::Type>: Operation<Type = C::Type>,
    C::Value: Compare<C::Value>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].compare(&inputs[1], self.direction)?])
    }
}

impl<C: Context<Operation: From<CompareOperation<C::Type>>>> PartiallyEvaluatableOperation<C>
    for CompareOperation<C::Type>
where
    CompareOperation<C::Type>: Operation<Type = C::Type>,
{
}

crate::impl_non_differentiable_operation!(<T> CompareOperation<T> where T: Type);
crate::impl_non_transposable_operation!(<T> CompareOperation<T> where T: Type);

/// Batching rule for first-class dimension comparison. Dimension operands describe one shared array shape and must
/// therefore remain replicated; their Boolean array result is replicated ordinary data.
impl<C: Context<Type = ArrayIrType>> BatchableOperation<C, ArrayIrBatching> for CompareOperation<ArrayIrType>
where
    C::Operation: From<CompareOperation<ArrayIrType>>,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatching>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatching>,
        _driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<Vec<ArrayIrBatch<C::Value>>, BatchingError> {
        let [left, right] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
        };
        left.validate_replicated_dimension()?;
        right.validate_replicated_dimension()?;
        Ok(context
            .parent()
            .bind(self.clone(), Vec::new(), &[left.value().clone(), right.value().clone()])?
            .into_iter()
            .map(ArrayIrBatch::replicated)
            .collect())
    }
}

/// Represents the ability to perform a pairwise comparison between two values. For array values,
/// `left.compare(right, direction)` produces a Boolean-valued result whose `i`-th element is the result of comparing
/// the `i`-th elements of `left` and `right` according to `direction`. The input arrays must have broadcast-compatible
/// shapes and promotable [`DataType`](crate::arrays::DataType)s. The result has
/// [`DataType::Boolean`](crate::arrays::DataType::Boolean) and the broadcasted shape of the two input arrays.
///
/// First-class dimensions use the same comparison operation but return ordinary rank-zero Boolean array data. This
/// keeps the predicate available to selection and control-flow operations without making the result a dimension:
///
/// ```rust
/// # use ryft_core::{ArrayIrValue, Compare, DimensionValue, ProgramError};
/// # use ryft_core::backends::arrays::Array;
/// # fn main() -> Result<(), ProgramError> {
/// let left = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(3)?);
/// let right = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(5)?);
/// let ArrayIrValue::Array(result) = left.less_than(&right)? else {
///     unreachable!("comparing dimensions always returns an array member");
/// };
/// assert_eq!(result, Array::scalar(true));
/// # Ok(())
/// # }
/// ```
///
/// The `Output` type parameter lets the comparison result use a different value carrier when the input carrier cannot
/// represent Boolean data. Array values use the default `Output = Self` and return honestly Boolean-typed values.
/// [`DimensionValue`](crate::DimensionValue), by contrast, uses an array output because a first-class dimension
/// describes an array extent rather than serving as a general scalar-data carrier.
pub trait Compare<Output = Self>: Sized {
    /// Compares `self` and `rhs` using a predicate determined by the provided `direction`.
    fn compare(&self, rhs: &Self, direction: ComparisonDirection) -> Result<Output, ProgramError>;

    /// Computes `self == rhs` using [`CompareOperation`].
    #[inline]
    fn equal(&self, rhs: &Self) -> Result<Output, ProgramError> {
        self.compare(rhs, ComparisonDirection::Equal)
    }

    /// Computes `self != rhs` using [`CompareOperation`].
    #[inline]
    fn not_equal(&self, rhs: &Self) -> Result<Output, ProgramError> {
        self.compare(rhs, ComparisonDirection::NotEqual)
    }

    /// Computes `self < rhs` using [`CompareOperation`].
    #[inline]
    fn less_than(&self, rhs: &Self) -> Result<Output, ProgramError> {
        self.compare(rhs, ComparisonDirection::LessThan)
    }

    /// Computes `self <= rhs` using [`CompareOperation`].
    #[inline]
    fn less_than_or_equal(&self, rhs: &Self) -> Result<Output, ProgramError> {
        self.compare(rhs, ComparisonDirection::LessThanOrEqual)
    }

    /// Computes `self > rhs` using [`CompareOperation`].
    #[inline]
    fn greater_than(&self, rhs: &Self) -> Result<Output, ProgramError> {
        self.compare(rhs, ComparisonDirection::GreaterThan)
    }

    /// Computes `self >= rhs` using [`CompareOperation`].
    #[inline]
    fn greater_than_or_equal(&self, rhs: &Self) -> Result<Output, ProgramError> {
        self.compare(rhs, ComparisonDirection::GreaterThanOrEqual)
    }
}

impl<V: Value<DispatchDomain: Context<Operation: From<CompareOperation<V::Type>>>>> Compare<V> for V {
    #[inline]
    fn compare(&self, rhs: &Self, direction: ComparisonDirection) -> Result<Self, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(CompareOperation::new(direction), Vec::new(), &[self.clone(), rhs.clone()])?
            .remove(0))
    }
}

impl<V: Value<Type = ArrayIrType>> Compare<V> for ProjectedValue<DimensionType, V>
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<CompareOperation<V::Type>>,
{
    fn compare(&self, rhs: &Self, direction: ComparisonDirection) -> Result<V, ProgramError> {
        Ok(self
            .value()
            .dispatch_domain()
            .bind(CompareOperation::new(direction), Vec::new(), &[self.value().clone(), rhs.value().clone()])?
            .remove(0))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayType, DataType, Dimension, DimensionBounds, DimensionType,
        DimensionValue, DimensionVariable, Layout, Memory, Shape, StridedLayout,
    };
    use crate::backends::{Array, ArrayOperation};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{DifferentiationError, DifferentiationTracer, TransposableOperation, jvp};
    use crate::macros::{check_operation_batching, check_operation_partial_evaluation};
    use crate::operations::constants::zero_like::ZeroLike;
    use crate::operations::control_flow::select::Select;
    use crate::parameters::Placeholder;
    use crate::programs::{
        Effects, EmptyRegionDriver, ProgramBuilder, ProgramError, RegionInterface, Typed, ValueProjection,
    };
    use crate::tracing::{Tracer, TracingContext};

    use super::*;

    /// `f(x) = select(x > 0, 2x, 3x)` expressed over JVP duals of the eager [`Array`] context.
    fn piecewise_select(
        x: DifferentiationTracer<EagerContext<Array, ArrayOperation<Array>>>,
    ) -> Result<DifferentiationTracer<EagerContext<Array, ArrayOperation<Array>>>, ProgramError> {
        let mask = x.compare(&x.zero_like(), ComparisonDirection::GreaterThan)?;
        Select::select(&mask, &(x.clone() + x.clone()), &(x.clone() + x.clone() + x))
    }

    #[test]
    fn test_compare() {
        let left = || Array::vector(vec![1.0, 2.0, 3.0]);
        let right = || Array::vector(vec![2.0, 2.0, 2.0]);
        assert_eq!(left().equal(&right()).unwrap().elements::<bool>(), Ok(vec![false, true, false]));
        assert_eq!(left().not_equal(&right()).unwrap().elements::<bool>(), Ok(vec![true, false, true]));
        assert_eq!(left().less_than(&right()).unwrap().elements::<bool>(), Ok(vec![true, false, false]));
        assert_eq!(left().less_than_or_equal(&right()).unwrap().elements::<bool>(), Ok(vec![true, true, false]));
        assert_eq!(left().greater_than(&right()).unwrap().elements::<bool>(), Ok(vec![false, false, true]));
        assert_eq!(left().greater_than_or_equal(&right()).unwrap().elements::<bool>(), Ok(vec![false, true, true]));

        let left = DimensionValue::constant(3).unwrap();
        let right = DimensionValue::constant(5).unwrap();
        assert_eq!(left.equal(&right), Ok(Array::scalar(false)));
        assert_eq!(left.not_equal(&right), Ok(Array::scalar(true)));
        assert_eq!(left.less_than(&right), Ok(Array::scalar(true)));
        assert_eq!(left.less_than_or_equal(&right), Ok(Array::scalar(true)));
        assert_eq!(left.greater_than(&right), Ok(Array::scalar(false)));
        assert_eq!(left.greater_than_or_equal(&right), Ok(Array::scalar(false)));

        let left = ArrayIrValue::<Array>::Dimension(left);
        let right = ArrayIrValue::<Array>::Dimension(right);
        assert_eq!(left.less_than(&right), Ok(ArrayIrValue::Array(Array::scalar(true))));
    }

    #[test]
    fn test_compare_type_inference() {
        let ordered_scalar = CompareOperation::<DataType>::new(ComparisonDirection::LessThan);
        assert_eq!(
            ordered_scalar.infer_output_types(&[DataType::F32, DataType::F64], &[]),
            Ok(vec![DataType::Boolean]),
        );
        assert_eq!(
            ordered_scalar.infer_output_types(&[DataType::F8E3M4, DataType::F32], &[]),
            Err(TypeError::invalid("comparison input types are not broadcast-compatible")),
        );
        assert_eq!(
            ordered_scalar.infer_output_types(&[DataType::C64, DataType::C64], &[]),
            Err(TypeError::invalid(
                "cannot apply an ordered comparison to unordered complex operands of types c64 and c64",
            )),
        );

        // The array contract applies the same element-type rule while preserving the broadcasted structural metadata.
        let left = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![3, 1])))
            .with_memory(Memory::Host { pinned: true });
        let right = left.clone().with_data_type(DataType::F64);
        let ordered_array = CompareOperation::<ArrayType>::new(ComparisonDirection::LessThan);
        assert_eq!(
            Operation::infer_output_types(&ordered_array, &[left.clone(), right], &[]),
            Ok(vec![left.clone().with_data_type(DataType::Boolean)]),
        );

        let equality_scalar = CompareOperation::<DataType>::new(ComparisonDirection::Equal);
        assert_eq!(
            equality_scalar.infer_output_types(&[DataType::C64, DataType::C64], &[]),
            Ok(vec![DataType::Boolean]),
        );
        let equality_array = CompareOperation::<ArrayType>::new(ComparisonDirection::Equal);
        let complex = left.with_data_type(DataType::C64);
        assert_eq!(
            Operation::infer_output_types(&equality_array, &[complex.clone(), complex.clone()], &[]),
            Ok(vec![complex.with_data_type(DataType::Boolean)]),
        );

        let bounds = DimensionBounds::new(0, Some(9)).unwrap();
        let left = DimensionType::new(DimensionVariable::new("left", bounds));
        let right = DimensionType::new(DimensionVariable::new("right", bounds));
        let operation = CompareOperation::<ArrayIrType>::new(ComparisonDirection::LessThan);
        assert_eq!(
            operation.infer_output_types(&[left.clone().into(), right.clone().into()], &[]),
            Ok(vec![ArrayType::scalar(DataType::Boolean).into()]),
        );
        assert_eq!(
            operation.infer_output_types(&[ArrayType::scalar(DataType::I64).into(), right.clone().into()], &[]),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );
        assert_eq!(
            operation.infer_output_types(&[left.clone().into()], &[]),
            Err(TypeError::invalid("expected 2 inputs but got 1")),
        );
        assert_eq!(
            operation.infer_output_types(
                &[left.into(), right.into()],
                &[RegionInterface::new(Vec::new(), Vec::new(), Effects::PURE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_compare_batching() {
        check_operation_batching!(
            @exact,
            operation = CompareOperation::new(ComparisonDirection::GreaterThan),
            axis_size = 2,
            cases = [
                {
                    inputs = [
                        (@mapped(axis = 0), Array::vector(vec![1.0, -2.0])),
                        (@replicated, Array::scalar(0.0)),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![true, false]))],
                },
                {
                    inputs = [
                        (@replicated, Array::scalar(0.0)),
                        (@mapped(axis = 0), Array::vector(vec![1.0, -2.0])),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![false, true]))],
                },
            ],
        );
    }

    #[test]
    fn test_compare_differentiation() {
        // `f(x) = select(x > 0, 2x, 3x)`: the comparison output is Boolean, so its tangent is symbolically zero and
        // the derivative comes entirely from the selected branch (2 for x > 0 and 3 for x <= 0).
        let (primal, tangent) = jvp(piecewise_select, Array::scalar(2.0), Array::scalar(1.0)).unwrap();
        assert_eq!(primal.to_f64s(), vec![4.0]);
        assert_eq!(tangent.to_f64s(), vec![2.0]);

        let (primal, tangent) = jvp(piecewise_select, Array::scalar(-2.0), Array::scalar(1.0)).unwrap();
        assert_eq!(primal.to_f64s(), vec![-6.0]);
        assert_eq!(tangent.to_f64s(), vec![3.0]);
    }

    #[test]
    fn test_compare_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = CompareOperation::new(ComparisonDirection::GreaterThan),
            inputs = [Array::scalar(1.0), Array::scalar(0.0)],
            expected = Array::scalar(true),
        );

        let bounds = DimensionBounds::new(0, Some(9)).unwrap();
        let left_type = DimensionType::new(DimensionVariable::new("left", bounds));
        let right_type = DimensionType::new(DimensionVariable::new("right", bounds));
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = CompareOperation::new(ComparisonDirection::LessThan),
            cases = [
                {
                    inputs = [
                        (@known, ArrayIrValue::Dimension(
                            DimensionValue::new(left_type.clone(), 3).unwrap()
                        )),
                        (@known, ArrayIrValue::Dimension(
                            DimensionValue::new(right_type.clone(), 5).unwrap()
                        )),
                    ],
                    outputs = [
                        (@known, ArrayIrValue::Array(Array::scalar(true))),
                    ],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(
                            type = ArrayIrType::Dimension(left_type.clone()),
                            replay = ArrayIrValue::Dimension(
                                DimensionValue::new(left_type.clone(), 3).unwrap()
                            )
                        )),
                        (@unknown(
                            type = ArrayIrType::Dimension(right_type.clone()),
                            replay = ArrayIrValue::Dimension(
                                DimensionValue::new(right_type.clone(), 5).unwrap()
                            )
                        )),
                    ],
                    outputs = [
                        (@residual, ArrayIrValue::Array(Array::scalar(true))),
                    ],
                    residual_instructions = 1,
                },
            ],
        );
    }

    #[test]
    fn test_compare_array_ir() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let bounds = DimensionBounds::new(0, Some(9)).unwrap();
        let left_type = DimensionType::new(DimensionVariable::new("left", bounds));
        let right_type = DimensionType::new(DimensionVariable::new("right", bounds));
        let context = TestContext::new();
        let left = context.input(left_type.clone().into());
        let right = context.input(right_type.clone().into());
        let left_id = left.atom_id().unwrap();
        let right_id = right.atom_id().unwrap();
        let left = <Tracer<TestContext> as ValueProjection<DimensionType>>::into_projected(left).unwrap();
        let right = <Tracer<TestContext> as ValueProjection<DimensionType>>::into_projected(right).unwrap();
        let output = left.less_than(&right).unwrap();
        let output_id = output.atom_id().unwrap();
        assert_eq!(output.r#type().as_ref(), &ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));

        let builder = context.builder().borrow();
        let [instruction] = builder.instructions() else {
            panic!("expected one comparison instruction");
        };
        assert_eq!(instruction.inputs(), &[left_id, right_id]);
        assert_eq!(instruction.outputs(), &[output_id]);
        assert!(instruction.regions().is_empty());
        assert!(matches!(instruction.operation(), ArrayIrOperation::Compare(_)));
        let program = builder
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output_id],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        drop(builder);

        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::new(left_type.clone(), 3).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::new(right_type.clone(), 5).unwrap()),
            ]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(true))]),
        );

        let mut relocated_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let relocated_left =
            relocated_builder.add_input(DimensionType::new(DimensionVariable::new("relocated_left", bounds)).into());
        let relocated_right =
            relocated_builder.add_input(DimensionType::new(DimensionVariable::new("relocated_right", bounds)).into());
        let relocated_outputs = relocated_builder.splice_program(&program, &[relocated_left, relocated_right]).unwrap();
        let [relocated_instruction] = relocated_builder.instructions() else {
            panic!("expected one relocated comparison instruction");
        };
        assert_eq!(relocated_instruction.inputs(), &[relocated_left, relocated_right]);
        assert_eq!(relocated_instruction.outputs(), relocated_outputs.as_slice());
        assert!(matches!(relocated_instruction.operation(), ArrayIrOperation::Compare(_)));

        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_ids().len(), 2);
        assert_eq!(jvp.output_ids().len(), 1);

        let operation = ArrayIrOperation::<Array>::from(CompareOperation::new(ComparisonDirection::LessThan));
        let mut transposition_context = TestContext::new();
        assert!(matches!(
            <ArrayIrOperation<Array> as TransposableOperation<
                ArrayIrValue<Array>,
                ArrayIrOperation<Array>,
            >>::transpose(
                &operation,
                &mut transposition_context,
                &EmptyRegionDriver,
                &[],
                &[],
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `compare` is not transposable",
        ));
    }
}
