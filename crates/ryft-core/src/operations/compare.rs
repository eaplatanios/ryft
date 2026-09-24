use std::fmt::Display;
use std::marker::PhantomData;

use crate::arrays::{
    Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrType, ArrayIrValue, ArrayType, Broadcastable, DataType,
    DimensionType, DimensionValue,
};
use crate::batching::{BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain, ValueResolution};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{
    check_count, impl_non_differentiable_operation, impl_non_transposable_operation,
    impl_reference_dischargeable_operation,
};
use crate::operations::ElementwiseOperation;
use crate::operations::collectives::parallel_vary::ManualVariationAlignment;
use crate::operations::manipulation::conversions::ElementType;
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue, PartiallyEvaluatableOperation,
};
use crate::programs::{
    Operation, OperationFormatter, ProgramError, ProjectedValue, RegionInterface, Type, TypeError, Typed, Value,
    ValueProjection,
};

// TODO(eaplatanios): Review this module.

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

/// Canonical operation name for [`CompareOperation`].
pub const COMPARE_OPERATION_NAME: &str = "compare";

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

impl Operation for CompareOperation<DataType> {
    type Type = DataType;

    #[inline]
    fn name(&self) -> &'static str {
        COMPARE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[DataType],
        _region_interfaces: &[RegionInterface<DataType>],
    ) -> Result<Vec<DataType>, TypeError> {
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

        let broadcasted = DataType::broadcasted(input_types)
            .map_err(|_| TypeError::invalid("comparison input types are not broadcast-compatible".to_string()))?;
        Ok(vec![broadcasted.with_element_type(DataType::Boolean)])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, COMPARE_OPERATION_NAME)?
            .bracketed(|operation| operation.field("direction", self.direction))
    }
}

// Array comparisons preserve broadcast geometry and matching manual variation while producing Boolean elements.
impl Operation for CompareOperation<ArrayType> {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        COMPARE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
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

        ArrayType::check_matching_manual_variation(self.name(), &input_types.iter().collect::<Vec<_>>())?;
        let broadcasted = ArrayType::broadcasted(input_types)
            .map_err(|_| TypeError::invalid("comparison input types are not broadcast-compatible".to_string()))?;
        Ok(vec![broadcasted.with_element_type(DataType::Boolean)])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, COMPARE_OPERATION_NAME)?
            .bracketed(|operation| operation.field("direction", self.direction))
    }
}

// Composite comparison contract: both operands are first-class dimensions and the predicate is ordinary rank-zero
// Boolean array data rather than another dimension value.
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

impl_reference_dischargeable_operation!(@reference_free <T> CompareOperation<T> where T: Type);

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

impl<C: Context<Type = DataType, Operation: From<CompareOperation<DataType>>>> PartiallyEvaluatableOperation<C>
    for CompareOperation<DataType>
{
}

impl<C: Context<Type = ArrayType, Operation: From<CompareOperation<ArrayType>>>> PartiallyEvaluatableOperation<C>
    for CompareOperation<ArrayType>
{
}

impl<C: Context<Type = ArrayIrType>> PartiallyEvaluatableOperation<C> for CompareOperation<ArrayIrType>
where
    C::Operation: From<Self>,
    C::Constant: TryFrom<bool, Error = ProgramError> + ValueProjection<DimensionType, Projected = DimensionValue>,
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let regions = driver.regions().map(|region| region.to_program()).collect::<Vec<_>>();
        self.infer_output_types(&input_types, &regions.iter().map(|region| region.interface()).collect::<Vec<_>>())?;
        let left = <&DimensionType>::try_from(&input_types[0])?;
        let right = <&DimensionType>::try_from(&input_types[1])?;
        let mut exact = [None, None];
        for (input, extent) in inputs.iter().zip(exact.iter_mut()) {
            if let Some(value) = input.as_known()
                && let ValueResolution::Constant(value) = context.parent().resolve(value)
                && value.capture_index().is_none()
            {
                // Captures name runtime data, so only immediate constants refine the declared extent interval.
                *extent = Some(value.into_projected()?.extent());
            }
        }
        if let Some(output) = prove_dimension_comparison(left, right, exact, self.direction)? {
            return Ok(vec![PartialEvaluationValue::known(context.parent().lift(C::Constant::try_from(output)?)?)]);
        }
        context.fold_or_residualize(*self, regions, inputs)
    }
}

// Batching rule for first-class dimension comparison. Dimension operands describe one shared array shape and must
// therefore remain replicated; their Boolean array result is replicated ordinary data.
impl<C: Context<Type = ArrayIrType>> BatchableOperation<C, ArrayIrBatchingPolicy> for CompareOperation<ArrayIrType>
where
    C::Operation: From<CompareOperation<ArrayIrType>>,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        _driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatchingPolicy>, BatchingError> {
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
            .collect::<Vec<_>>()
            .into())
    }
}

impl_non_differentiable_operation!(<T> CompareOperation<T> where T: Type);
impl_non_transposable_operation!(<T> CompareOperation<T> where T: Type);

/// Represents the ability to perform a pairwise comparison between two values. For array values,
/// `left.compare(right, direction)` produces a Boolean-valued result whose `i`-th element is the result of comparing
/// the `i`-th elements of `left` and `right` according to `direction`. The input arrays must have broadcast-compatible
/// shapes and promotable [`DataType`]s. The result has [`DataType::Boolean`] and the broadcasted shape of the two
/// input arrays.
///
/// First-class dimensions use the same comparison operation but return ordinary rank-zero Boolean array data. This
/// keeps the predicate available to selection and control-flow operations without making the result a dimension:
///
/// ```rust
/// # use ryft_core::{ArrayIrValue, Compare, DimensionValue, ProgramError};
/// # use ryft_core::arrays::Array;
/// # fn main() -> Result<(), ProgramError> {
/// let left = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(3)?);
/// let right = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(5)?);
/// let ArrayIrValue::Array(result) = left.less_than(&right)? else {
///     unreachable!("comparing dimensions always returns an array member");
/// };
/// assert_eq!(result, Array::scalar(true).unwrap());
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

impl<T: Type, V: Value<Type = T> + ManualVariationAlignment<T>> Compare<V> for V
where
    V::DispatchDomain: Context<Operation: From<CompareOperation<T>>>,
{
    #[inline]
    fn compare(&self, rhs: &Self, direction: ComparisonDirection) -> Result<Self, ProgramError> {
        let inputs = [self.clone(), rhs.clone()];
        let inputs = ManualVariationAlignment::align_manual_variation(&inputs)?;
        Ok(self.dispatch_domain().bind(CompareOperation::new(direction), Vec::new(), &inputs)?.remove(0))
    }
}

impl<V: Value<Type = ArrayIrType>> Compare<V> for ProjectedValue<DimensionType, V>
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<CompareOperation<V::Type>>,
    <V::DispatchDomain as Domain>::Constant: TryFrom<bool, Error = ProgramError>,
{
    fn compare(&self, rhs: &Self, direction: ComparisonDirection) -> Result<V, ProgramError> {
        if let Some(output) =
            prove_dimension_comparison(self.r#type().as_ref(), rhs.r#type().as_ref(), [None, None], direction)?
        {
            return self.value().dispatch_domain().lift(<V::DispatchDomain as Domain>::Constant::try_from(output)?);
        }
        Ok(self
            .value()
            .dispatch_domain()
            .bind(CompareOperation::new(direction), Vec::new(), &[self.value().clone(), rhs.value().clone()])?
            .remove(0))
    }
}

// TODO(eaplatanios): Review this.

impl Compare<Array> for DimensionValue {
    fn compare(&self, rhs: &Self, direction: ComparisonDirection) -> Result<Array, ProgramError> {
        let result = match direction {
            ComparisonDirection::Equal => self.extent() == rhs.extent(),
            ComparisonDirection::NotEqual => self.extent() != rhs.extent(),
            ComparisonDirection::LessThan => self.extent() < rhs.extent(),
            ComparisonDirection::LessThanOrEqual => self.extent() <= rhs.extent(),
            ComparisonDirection::GreaterThan => self.extent() > rhs.extent(),
            ComparisonDirection::GreaterThanOrEqual => self.extent() >= rhs.extent(),
        };
        Array::scalar(result)
    }
}

impl<A: Value<Type = ArrayType> + TryFrom<bool, Error = ProgramError>> Compare<ArrayIrValue<A>> for DimensionValue {
    fn compare(&self, rhs: &Self, direction: ComparisonDirection) -> Result<ArrayIrValue<A>, ProgramError> {
        let output = prove_dimension_comparison(
            self.r#type().as_ref(),
            rhs.r#type().as_ref(),
            [Some(self.extent()), Some(rhs.extent())],
            direction,
        )?
        .unwrap();
        Ok(ArrayIrValue::Array(A::try_from(output)?))
    }
}

/// Proves comparisons using dimension identities and inclusive representable extent intervals.
fn prove_dimension_comparison(
    left: &DimensionType,
    right: &DimensionType,
    exact: [Option<usize>; 2],
    direction: ComparisonDirection,
) -> Result<Option<bool>, ProgramError> {
    let left_range = left.bounds().representable_extent_range()?;
    let right_range = right.bounds().representable_extent_range()?;
    let (left_minimum, left_maximum) = exact[0].map_or(left_range, |extent| (extent, extent));
    let (right_minimum, right_maximum) = exact[1].map_or(right_range, |extent| (extent, extent));
    // Resolved extents take precedence over symbolic identity, including inconsistent concrete inputs supplied
    // directly to partial evaluation rather than through a program's dimension binding validation.
    let identical = left.variable() == right.variable() && !matches!(exact, [Some(left), Some(right)] if left != right);
    let equal = if identical
        || (left_minimum == left_maximum && right_minimum == right_maximum && left_minimum == right_minimum)
    {
        Some(true)
    } else if left_maximum < right_minimum || right_maximum < left_minimum {
        Some(false)
    } else {
        None
    };
    Ok(match direction {
        ComparisonDirection::Equal => equal,
        ComparisonDirection::NotEqual => equal.map(|equal| !equal),
        ComparisonDirection::LessThan => {
            if identical || left_minimum >= right_maximum {
                Some(false)
            } else if left_maximum < right_minimum {
                Some(true)
            } else {
                None
            }
        }
        ComparisonDirection::LessThanOrEqual => {
            if identical || left_maximum <= right_minimum {
                Some(true)
            } else if left_minimum > right_maximum {
                Some(false)
            } else {
                None
            }
        }
        ComparisonDirection::GreaterThan => {
            prove_dimension_comparison(right, left, [exact[1], exact[0]], ComparisonDirection::LessThan)?
        }
        ComparisonDirection::GreaterThanOrEqual => {
            prove_dimension_comparison(right, left, [exact[1], exact[0]], ComparisonDirection::LessThanOrEqual)?
        }
    })
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType, Dimension,
        DimensionBounds, DimensionType, DimensionValue, Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType, Shape,
        Sharding, StridedLayout,
    };
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{
        DifferentiationError, DifferentiationTracer, TransposableOperation, TranspositionContext, differentiate_at,
    };
    use crate::macros::{check_operation_batching, check_operation_partial_evaluation};
    use crate::operations::constants::zero_like::ZeroLike;
    use crate::operations::control_flow::select::Select;
    use crate::parameters::Placeholder;
    use crate::programs::{
        EffectClasses, EmptyRegionDriver, ProgramBuilder, ProgramError, RegionInterface, Typed, ValueProjection,
    };
    use crate::tracing::{Tracer, TracingContext};

    use super::*;

    /// `f(x) = select(x > 0, 2x, 3x)` expressed over JVP duals of the eager [`Array`] context.
    fn piecewise_select(
        x: DifferentiationTracer<EagerContext<Array, ArrayOperation<Array>>>,
    ) -> Result<DifferentiationTracer<EagerContext<Array, ArrayOperation<Array>>>, ProgramError> {
        let mask = x.compare(&x.zero_like()?, ComparisonDirection::GreaterThan)?;
        Select::select(&mask, &(x.clone() + x.clone()), &(x.clone() + x.clone() + x))
    }

    #[test]
    fn test_compare() {
        let left = || Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        let right = || Array::vector(vec![2.0, 2.0, 2.0]).unwrap();
        assert_eq!(left().equal(&right()).unwrap().elements::<bool>(), Ok(vec![false, true, false]));
        assert_eq!(left().not_equal(&right()).unwrap().elements::<bool>(), Ok(vec![true, false, true]));
        assert_eq!(left().less_than(&right()).unwrap().elements::<bool>(), Ok(vec![true, false, false]));
        assert_eq!(left().less_than_or_equal(&right()).unwrap().elements::<bool>(), Ok(vec![true, true, false]));
        assert_eq!(left().greater_than(&right()).unwrap().elements::<bool>(), Ok(vec![false, false, true]));
        assert_eq!(left().greater_than_or_equal(&right()).unwrap().elements::<bool>(), Ok(vec![false, true, true]));

        let left = DimensionValue::constant(3).unwrap();
        let right = DimensionValue::constant(5).unwrap();
        assert_eq!(left.equal(&right), Ok(Array::scalar(false).unwrap()));
        assert_eq!(left.not_equal(&right), Ok(Array::scalar(true).unwrap()));
        assert_eq!(left.less_than(&right), Ok(Array::scalar(true).unwrap()));
        assert_eq!(left.less_than_or_equal(&right), Ok(Array::scalar(true).unwrap()));
        assert_eq!(left.greater_than(&right), Ok(Array::scalar(false).unwrap()));
        assert_eq!(left.greater_than_or_equal(&right), Ok(Array::scalar(false).unwrap()));

        let left = ArrayIrValue::<Array>::Dimension(left);
        let right = ArrayIrValue::<Array>::Dimension(right);
        assert_eq!(left.less_than(&right), Ok(ArrayIrValue::Array(Array::scalar(true).unwrap())));
    }

    #[test]
    fn test_compare_manual_variation() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let invariant_type = ArrayType::scalar(DataType::F32);
        let varying_type = invariant_type
            .clone()
            .with_sharding(Sharding::replicated(mesh, 0).with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        let operation = CompareOperation::<ArrayType>::new(ComparisonDirection::Equal);
        let error = TypeError::invalid(
            "`compare` inputs must have matching varying manual axes; insert `parallel_vary` on the inputs that lack \
             an axis, as `align_manual_variation` does",
        );
        assert_eq!(
            Operation::infer_output_types(&operation, &[invariant_type.clone(), varying_type.clone()], &[]),
            Err(error.clone()),
        );
        assert_eq!(
            Operation::infer_output_types(&operation, &[varying_type.clone(), invariant_type.clone()], &[]),
            Err(error.clone()),
        );
        let invariant = Array::from_elements(invariant_type, &[1_f32]).unwrap();
        let varying = Array::from_elements(varying_type.clone(), &[1_f32]).unwrap();
        assert_eq!(invariant.equal(&varying), Err(error.clone().into()));
        assert_eq!(varying.equal(&invariant), Err(error.into()));
        assert_eq!(
            varying.equal(&varying),
            Array::from_elements(varying_type.with_data_type(DataType::Boolean), &[true]),
        );
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

        // Comparison preserves shape and placement but clears byte strides when the element storage width changes.
        let left = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![24, 8])))
            .with_memory(Memory::Host { pinned: true });
        let right = left.clone().with_data_type(DataType::F64);
        let ordered_array = CompareOperation::<ArrayType>::new(ComparisonDirection::LessThan);
        assert_eq!(
            Operation::infer_output_types(&ordered_array, &[left.clone(), right], &[]),
            Ok(vec![left.clone().with_data_type(DataType::Boolean).with_layout(None)]),
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
            Ok(vec![complex.with_data_type(DataType::Boolean).with_layout(None)]),
        );

        let bounds = DimensionBounds::new(0, Some(9)).unwrap();
        let left = DimensionType::new("left", bounds);
        let right = DimensionType::new("right", bounds);
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
                &[RegionInterface::new(Vec::new(), Vec::new(), EffectClasses::NONE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_dimension_comparison_proofs() {
        let directions = [
            ComparisonDirection::Equal,
            ComparisonDirection::NotEqual,
            ComparisonDirection::LessThan,
            ComparisonDirection::LessThanOrEqual,
            ComparisonDirection::GreaterThan,
            ComparisonDirection::GreaterThanOrEqual,
        ];
        for left_minimum in 0..4 {
            for left_maximum in left_minimum..4 {
                for right_minimum in 0..4 {
                    for right_maximum in right_minimum..4 {
                        let left = DimensionType::new(
                            "left",
                            DimensionBounds::new(left_minimum, Some(left_maximum + 1)).unwrap(),
                        );
                        let right = DimensionType::new(
                            "right",
                            DimensionBounds::new(right_minimum, Some(right_maximum + 1)).unwrap(),
                        );
                        for direction in directions {
                            let mut outcomes = Vec::new();
                            for left in left_minimum..=left_maximum {
                                for right in right_minimum..=right_maximum {
                                    outcomes.push(match direction {
                                        ComparisonDirection::Equal => left == right,
                                        ComparisonDirection::NotEqual => left != right,
                                        ComparisonDirection::LessThan => left < right,
                                        ComparisonDirection::LessThanOrEqual => left <= right,
                                        ComparisonDirection::GreaterThan => left > right,
                                        ComparisonDirection::GreaterThanOrEqual => left >= right,
                                    });
                                }
                            }
                            let expected =
                                outcomes.iter().all(|outcome| *outcome == outcomes[0]).then_some(outcomes[0]);
                            assert_eq!(
                                prove_dimension_comparison(&left, &right, [None, None], direction).unwrap(),
                                expected
                            );
                        }
                    }
                }
            }
        }
        let dimension = DimensionType::new("dimension", DimensionBounds::new(0, None).unwrap());
        for (direction, expected) in directions.into_iter().zip([true, false, false, true, false, true]) {
            assert_eq!(
                prove_dimension_comparison(&dimension, &dimension, [None, None], direction).unwrap(),
                Some(expected)
            );
        }
        for (direction, expected) in directions.into_iter().zip([false, true, true, true, false, false]) {
            assert_eq!(
                prove_dimension_comparison(&dimension, &dimension, [Some(4), Some(5)], direction).unwrap(),
                Some(expected),
            );
        }
    }

    #[test]
    fn test_compare_dimension_staging_proof() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let context = TestContext::new();
        let dimension = context.input(DimensionType::new("extent", DimensionBounds::new(0, None).unwrap()).into());
        let dimension = ValueProjection::<DimensionType>::into_projected(dimension).unwrap();
        let predicate = dimension.equal(&dimension).unwrap();
        assert!(
            matches!(context.resolve(&predicate), ValueResolution::Constant(ArrayIrValue::Array(value)) if value == Array::scalar(true).unwrap())
        );
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_compare_interpretation() {
        assert_eq!(
            CompareOperation::new(ComparisonDirection::GreaterThan).interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::vector(vec![1.0, -2.0]).unwrap(), Array::scalar(0.0).unwrap()],
            ),
            Ok(vec![Array::vector(vec![true, false]).unwrap()]),
        );
    }

    #[test]
    fn test_compare_array_ir() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let bounds = DimensionBounds::new(0, Some(9)).unwrap();
        let left_type = DimensionType::new("left", bounds);
        let right_type = DimensionType::new("right", bounds);
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
            Ok(vec![ArrayIrValue::Array(Array::scalar(true).unwrap())]),
        );

        let mut relocated_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let relocated_left = relocated_builder.add_input(DimensionType::new("relocated_left", bounds).into());
        let relocated_right = relocated_builder.add_input(DimensionType::new("relocated_right", bounds).into());
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
        let transposition_context = TestContext::new();
        assert!(matches!(
            <ArrayIrOperation<Array> as TransposableOperation<
                ArrayIrValue<Array>,
                ArrayIrOperation<Array>,
            >>::transpose(
                &operation,
                &mut TranspositionContext::new(transposition_context.clone()),
                &EmptyRegionDriver,
                &[],
                &[],
                &[],
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `compare` is not transposable",
        ));
    }

    #[test]
    fn test_compare_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = CompareOperation::new(ComparisonDirection::GreaterThan),
            inputs = [Array::scalar(1.0).unwrap(), Array::scalar(0.0).unwrap()],
            expected = Array::scalar(true).unwrap(),
        );

        let bounds = DimensionBounds::new(0, Some(9)).unwrap();
        let left_type = DimensionType::new("left", bounds);
        let right_type = DimensionType::new("right", bounds);
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
                        (@known, ArrayIrValue::Array(Array::scalar(true).unwrap())),
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
                        (@residual, ArrayIrValue::Array(Array::scalar(true).unwrap())),
                    ],
                    residual_instructions = 1,
                },
                {
                    inputs = [
                        (@unknown(
                            type = ArrayIrType::Dimension(left_type.clone()),
                            replay = ArrayIrValue::Dimension(DimensionValue::new(left_type.clone(), 3).unwrap())
                        )),
                        (@unknown(
                            type = ArrayIrType::Dimension(left_type.clone()),
                            replay = ArrayIrValue::Dimension(DimensionValue::new(left_type.clone(), 3).unwrap())
                        )),
                    ],
                    outputs = [(@known, ArrayIrValue::Array(Array::scalar(false).unwrap()))],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@known, ArrayIrValue::Dimension(DimensionValue::new(left_type.clone(), 8).unwrap())),
                        (@unknown(
                            type = ArrayIrType::Dimension(right_type.clone()),
                            replay = ArrayIrValue::Dimension(DimensionValue::new(right_type.clone(), 5).unwrap())
                        )),
                    ],
                    outputs = [(@known, ArrayIrValue::Array(Array::scalar(false).unwrap()))],
                    residual_instructions = 0,
                },
            ],
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
                        (@mapped(axis = 0), Array::vector(vec![1.0, -2.0]).unwrap()),
                        (@replicated, Array::scalar(0.0).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![true, false]).unwrap())],
                },
                {
                    inputs = [
                        (@replicated, Array::scalar(0.0).unwrap()),
                        (@mapped(axis = 0), Array::vector(vec![1.0, -2.0]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![false, true]).unwrap())],
                },
            ],
        );
    }

    #[test]
    fn test_compare_differentiation() {
        // `f(x) = select(x > 0, 2x, 3x)`: the comparison output is Boolean, so its tangent is symbolically zero and
        // the derivative comes entirely from the selected branch (2 for x > 0 and 3 for x <= 0).
        let (primal, tangent) = differentiate_at(Array::scalar(2.0).unwrap())
            .jvp(Array::scalar(1.0).unwrap(), piecewise_select)
            .unwrap();
        assert_eq!(primal.to_f64s(), vec![4.0]);
        assert_eq!(tangent.to_f64s(), vec![2.0]);

        let (primal, tangent) = differentiate_at(Array::scalar(-2.0).unwrap())
            .jvp(Array::scalar(1.0).unwrap(), piecewise_select)
            .unwrap();
        assert_eq!(primal.to_f64s(), vec![-6.0]);
        assert_eq!(tangent.to_f64s(), vec![3.0]);
    }
}
