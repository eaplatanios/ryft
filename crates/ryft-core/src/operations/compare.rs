use std::fmt::Display;

use crate::broadcasting::Broadcastable;
use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_elementwise_operation};
use crate::operations::ElementwiseOperation;
use crate::operations::manipulation::conversion::ElementType;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::ProgramError;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::TypeError;
use crate::programs::values::{ProjectedValue, Value};
use crate::types::{ArrayProgramType, ArrayType, DataType, DimensionType};

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

/// [`Operation`] that performs pairwise comparisons. Refer to [`Compare`] for its elementwise and first-class
/// dimension semantics.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CompareOperation {
    /// [`ComparisonDirection`] used by this [`CompareOperation`].
    direction: ComparisonDirection,
}

impl CompareOperation {
    /// Creates a new [`CompareOperation`] with the provided [`ComparisonDirection`].
    #[inline]
    pub fn new(direction: ComparisonDirection) -> Self {
        Self { direction }
    }

    /// Returns the [`ComparisonDirection`] used by this [`CompareOperation`].
    #[inline]
    pub fn direction(&self) -> ComparisonDirection {
        self.direction
    }
}

impl Display for CompareOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Operation::<ArrayType>::render(self, formatter, 0)
    }
}

impl<T: Broadcastable + ElementType> Operation<T> for CompareOperation {
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

impl Operation<ArrayProgramType> for CompareOperation {
    #[inline]
    fn name(&self) -> &'static str {
        COMPARE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayProgramType],
        region_interfaces: &[RegionInterface<ArrayProgramType>],
    ) -> Result<Vec<ArrayProgramType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        input_types.iter().try_for_each(|r#type| <&DimensionType>::try_from(r#type).map(|_| ()))?;
        // Comparing first-class dimensions produces ordinary predicate data rather than another dimension value.
        Ok(vec![ArrayType::scalar(DataType::Boolean).into()])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, COMPARE_OPERATION_NAME)?
            .bracketed(|operation| operation.field("direction", self.direction))
    }
}

impl ElementwiseOperation for CompareOperation {
    #[inline]
    fn input_count(&self) -> usize {
        2
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        Operation::<ArrayType>::infer_output_types(self, input_types, &[])
    }
}

impl<C: Domain> InterpretableOperation<C> for CompareOperation
where
    CompareOperation: Operation<C::Type>,
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

impl<C: Context<Operation: From<CompareOperation>>> PartiallyEvaluatableOperation<C> for CompareOperation where
    CompareOperation: Operation<C::Type>
{
}

impl_differentiable_elementwise_operation!(@non_differentiable CompareOperation);

/// Represents the ability to perform a pairwise comparison between two values. For array values,
/// `left.compare(right, direction)` produces a Boolean-valued result whose `i`-th element is the result of comparing
/// the `i`-th elements of `left` and `right` according to `direction`. The input arrays must have broadcast-compatible
/// shapes and promotable [`DataType`](crate::DataType)s. The result has [`DataType::Boolean`](crate::DataType::Boolean)
/// and the broadcasted shape of the two input arrays.
///
/// First-class dimensions use the same comparison operation but return ordinary rank-zero Boolean array data. This
/// keeps the predicate available to selection and control-flow operations without making the result a dimension:
///
/// ```rust
/// # use ryft_core::{ArrayProgramValue, Compare, DimensionValue, ProgramError};
/// # use ryft_core::backends::arrays::Array;
/// # fn main() -> Result<(), ProgramError> {
/// let left = ArrayProgramValue::<Array>::Dimension(DimensionValue::constant(3)?);
/// let right = ArrayProgramValue::<Array>::Dimension(DimensionValue::constant(5)?);
/// let ArrayProgramValue::Array(result) = left.less_than(&right)? else {
///     unreachable!("comparing dimensions always returns an array member");
/// };
/// assert_eq!(result, Array::scalar(true));
/// # Ok(())
/// # }
/// ```
///
/// The `Output` type parameter lets the comparison result use a different value carrier when the input carrier cannot
/// represent Boolean data. Scalar and array backends use the default `Output = Self` and return honestly
/// Boolean-typed values. [`DimensionValue`](crate::DimensionValue), by contrast, uses an array output because a
/// first-class dimension describes an array extent rather than serving as a general scalar-data carrier.
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

impl<V: Value<DispatchDomain: Context<Operation: From<CompareOperation>>>> Compare<V> for V {
    #[inline]
    fn compare(&self, rhs: &Self, direction: ComparisonDirection) -> Result<Self, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(CompareOperation::new(direction), Vec::new(), &[self.clone(), rhs.clone()])?
            .remove(0))
    }
}

impl<V: Value<Type = ArrayProgramType>> Compare<V> for ProjectedValue<DimensionType, V>
where
    V::DispatchDomain: Context<Type = ArrayProgramType>,
    <V::DispatchDomain as Domain>::Operation: From<CompareOperation>,
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

    use crate::backends::array_programs::{ArrayProgramOperation, ArrayProgramValue};
    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::dimensions::DimensionValue;
    use crate::backends::scalars::Scalar;
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::DifferentiationError;
    use crate::differentiation::forward::{DifferentiationTracer, jvp};
    use crate::differentiation::reverse::TransposableOperation;
    use crate::macros::{check_operation_batching, check_operation_partial_evaluation, check_operation_type_inference};
    use crate::operations::constants::ZeroLike;
    use crate::operations::control_flow::Select;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::effects::Effects;
    use crate::programs::regions::{EmptyRegionDriver, RegionInterface};
    use crate::programs::types::Typed;
    use crate::programs::values::ValueProjection;
    use crate::tracing::{Tracer, TracingContext};
    use crate::types::{ArrayProgramType, ArrayType, DataType, DimensionBounds, DimensionType, DimensionVariable};

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
        assert_eq!(Scalar::from(2.0).less_than(&Scalar::from(3.0)).unwrap(), Scalar::from(true));
        assert_eq!(Scalar::from(2.0f32).greater_than(&Scalar::from(3.0f32)).unwrap(), Scalar::from(false));
        let left = || Array::vector(vec![1.0, 2.0, 3.0]);
        let right = || Array::vector(vec![2.0, 2.0, 2.0]);
        assert_eq!(left().equal(&right()).unwrap().values(), &[false, true, false]);
        assert_eq!(left().not_equal(&right()).unwrap().values(), &[true, false, true]);
        assert_eq!(left().less_than(&right()).unwrap().values(), &[true, false, false]);
        assert_eq!(left().less_than_or_equal(&right()).unwrap().values(), &[true, true, false]);
        assert_eq!(left().greater_than(&right()).unwrap().values(), &[false, false, true]);
        assert_eq!(left().greater_than_or_equal(&right()).unwrap().values(), &[false, true, true]);

        let left = DimensionValue::constant(3).unwrap();
        let right = DimensionValue::constant(5).unwrap();
        assert_eq!(left.equal(&right), Ok(Array::scalar(false)));
        assert_eq!(left.not_equal(&right), Ok(Array::scalar(true)));
        assert_eq!(left.less_than(&right), Ok(Array::scalar(true)));
        assert_eq!(left.less_than_or_equal(&right), Ok(Array::scalar(true)));
        assert_eq!(left.greater_than(&right), Ok(Array::scalar(false)));
        assert_eq!(left.greater_than_or_equal(&right), Ok(Array::scalar(false)));

        let left = ArrayProgramValue::<Array>::Dimension(left);
        let right = ArrayProgramValue::<Array>::Dimension(right);
        assert_eq!(left.less_than(&right), Ok(ArrayProgramValue::Array(Array::scalar(true))));
    }

    #[test]
    fn test_compare_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = CompareOperation::new(ComparisonDirection::LessThan),
            cases = [
                {
                    input_data_types = [DataType::F32, DataType::F64],
                    output_data_types = [DataType::Boolean],
                },
                {
                    input_data_types = [DataType::F8E3M4, DataType::F32],
                    error = "comparison input types are not broadcast-compatible",
                },
                {
                    input_data_types = [DataType::C64, DataType::C64],
                    error = "cannot apply an ordered comparison to unordered complex operands of types c64 and c64",
                },
            ],
        );
        check_operation_type_inference!(
            @elementwise @binary,
            operation = CompareOperation::new(ComparisonDirection::Equal),
            cases = [{
                input_data_types = [DataType::C64, DataType::C64],
                output_data_types = [DataType::Boolean],
            }],
        );

        let bounds = DimensionBounds::new(0, Some(9)).unwrap();
        let left = DimensionType::new(DimensionVariable::new("left", bounds));
        let right = DimensionType::new(DimensionVariable::new("right", bounds));
        let operation = CompareOperation::new(ComparisonDirection::LessThan);
        assert_eq!(
            Operation::<ArrayProgramType>::infer_output_types(
                &operation,
                &[left.clone().into(), right.clone().into()],
                &[],
            ),
            Ok(vec![ArrayType::scalar(DataType::Boolean).into()]),
        );
        assert_eq!(
            Operation::<ArrayProgramType>::infer_output_types(
                &operation,
                &[ArrayType::scalar(DataType::I64).into(), right.clone().into()],
                &[],
            ),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );
        assert_eq!(
            Operation::<ArrayProgramType>::infer_output_types(&operation, &[left.clone().into()], &[]),
            Err(TypeError::invalid("expected 2 inputs but got 1")),
        );
        assert_eq!(
            Operation::<ArrayProgramType>::infer_output_types(
                &operation,
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
            inputs = [Scalar::from(1.0), Scalar::from(0.0)],
            expected = Scalar::from(true),
        );

        let bounds = DimensionBounds::new(0, Some(9)).unwrap();
        let left_type = DimensionType::new(DimensionVariable::new("left", bounds));
        let right_type = DimensionType::new(DimensionVariable::new("right", bounds));
        check_operation_partial_evaluation!(
            backend = (ArrayProgramValue<Array>, ArrayProgramOperation<Array>),
            operation = CompareOperation::new(ComparisonDirection::LessThan),
            cases = [
                {
                    inputs = [
                        (@known, ArrayProgramValue::Dimension(
                            DimensionValue::new(left_type.clone(), 3).unwrap()
                        )),
                        (@known, ArrayProgramValue::Dimension(
                            DimensionValue::new(right_type.clone(), 5).unwrap()
                        )),
                    ],
                    outputs = [
                        (@known, ArrayProgramValue::Array(Array::scalar(true))),
                    ],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(
                            type = ArrayProgramType::Dimension(left_type.clone()),
                            replay = ArrayProgramValue::Dimension(
                                DimensionValue::new(left_type.clone(), 3).unwrap()
                            )
                        )),
                        (@unknown(
                            type = ArrayProgramType::Dimension(right_type.clone()),
                            replay = ArrayProgramValue::Dimension(
                                DimensionValue::new(right_type.clone(), 5).unwrap()
                            )
                        )),
                    ],
                    outputs = [
                        (@residual, ArrayProgramValue::Array(Array::scalar(true))),
                    ],
                    residual_instructions = 1,
                },
            ],
        );
    }

    #[test]
    fn test_compare_array_program() {
        type TestContext = TracingContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;

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
        assert_eq!(output.r#type().as_ref(), &ArrayProgramType::Array(ArrayType::scalar(DataType::Boolean)));

        let builder = context.builder().borrow();
        let [instruction] = builder.instructions() else {
            panic!("expected one comparison instruction");
        };
        assert_eq!(instruction.inputs(), &[left_id, right_id]);
        assert_eq!(instruction.outputs(), &[output_id]);
        assert!(instruction.regions().is_empty());
        assert!(matches!(instruction.operation(), ArrayProgramOperation::Compare(_)));
        let program = builder
            .clone()
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output_id],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        drop(builder);

        assert_eq!(
            program.interpret(vec![
                ArrayProgramValue::Dimension(DimensionValue::new(left_type.clone(), 3).unwrap()),
                ArrayProgramValue::Dimension(DimensionValue::new(right_type.clone(), 5).unwrap()),
            ]),
            Ok(vec![ArrayProgramValue::Array(Array::scalar(true))]),
        );

        let mut relocated_builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
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
        assert!(matches!(relocated_instruction.operation(), ArrayProgramOperation::Compare(_)));

        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_ids().len(), 4);
        assert_eq!(jvp.output_ids().len(), 2);
        assert_eq!(
            jvp.outputs().last().unwrap().r#type().as_ref(),
            &ArrayProgramType::Array(ArrayType::scalar(DataType::Zero)),
        );

        let operation = ArrayProgramOperation::<Array>::from(CompareOperation::new(ComparisonDirection::LessThan));
        let mut transposition_context = TestContext::new();
        assert!(matches!(
            <ArrayProgramOperation<Array> as TransposableOperation<
                ArrayProgramValue<Array>,
                ArrayProgramOperation<Array>,
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
