//! Operations that compare pairs of values and produce Boolean data. Every comparison is defined by the
//! [`CompareOperation`] type together with the [`Compare`] value capability trait, whose functions apply it to eager
//! [`Array`]s and traced values alike, so the same code executes immediately or records into a program depending on
//! the value it runs on. A [`ComparisonDirection`] selects the predicate, and [`Compare`] additionally provides one
//! named function per direction (e.g., [`less_than`](Compare::less_than)). Comparisons apply to two kinds of inputs:
//!
//!   - **Arrays:** The inputs are broadcast to a common shape and their element types are promoted before
//!     they are compared elementwise, producing [`Boolean`](DataType::Boolean) elements, as in StableHLO's
//!     [`compare`](https://openxla.org/stablehlo/spec#compare). Equality and inequality support complex elements,
//!     whereas ordered comparisons reject them, and a comparison involving a floating-point NaN is false for every
//!     direction except [`NotEqual`](ComparisonDirection::NotEqual).
//!   - **Dimensions:** [`DimensionValue`]s and dimension-typed traced values compare their extents and produce
//!     rank-zero Boolean arrays. Predicates that dimension identities and extent bounds already prove (e.g., that
//!     a dimension is equal to itself) become Boolean constants without staging an operation, both when tracing
//!     and during partial evaluation.
//!
//! Batching maps array comparisons elementwise over the batch axis, whereas dimension comparisons require replicated
//! inputs, since a dimension describes one array shape that every batch item shares, and produce replicated outputs.
//! Comparisons are not differentiable (i.e., their outputs have zero tangents) and cannot be transposed.
//!
//! # Example
//!
//! A scalar threshold broadcasts across an array, producing one Boolean per element:
//!
//! ```rust
//! # use ryft_core::{Array, Compare, ProgramError};
//! # fn main() -> Result<(), ProgramError> {
//! let input = Array::vector(vec![1.0f32, 2.0, 3.0])?;
//! let threshold = Array::scalar(2.0f32)?;
//! assert_eq!(input.less_than(&threshold)?, Array::vector(vec![true, false, false])?);
//! # Ok(())
//! # }
//! ```

use std::cmp::Ordering;
use std::fmt::Display;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::arrays::{
    Array, ArrayAddressing, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrType, ArrayIrValue, ArrayType, Broadcastable,
    DataType, DimensionType, DimensionValue,
};
use crate::batching::{BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain, ValueResolution};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{
    check_count, check_types, dispatch_on_array_element_type, impl_non_differentiable_operation,
    impl_non_transposable_operation, impl_reference_dischargeable_operation,
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

/// Direction of the pairwise comparison performed by a [`CompareOperation`].
/// Each direction corresponds to a comparison predicate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ComparisonDirection {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

impl ComparisonDirection {
    /// Returns whether this predicate holds between `left` and `right` as proved by their dimension identities and
    /// representable extent intervals, or [`None`] when it depends on runtime extents. An entry in `exact` narrows
    /// the corresponding input to a known extent. Conflicting exact extents take precedence over a shared symbolic
    /// identity. Invalid extent bounds return an error.
    fn prove_for_dimensions(
        self,
        left: &DimensionType,
        right: &DimensionType,
        exact: [Option<usize>; 2],
    ) -> Result<Option<bool>, ProgramError> {
        let left_range = left.bounds().representable_extent_range()?;
        let right_range = right.bounds().representable_extent_range()?;
        let (left_minimum, left_maximum) = exact[0].map_or(left_range, |extent| (extent, extent));
        let (right_minimum, right_maximum) = exact[1].map_or(right_range, |extent| (extent, extent));

        // Resolved extents take precedence over symbolic identity, including inconsistent concrete inputs supplied
        // directly to partial evaluation rather than through a program's dimension binding validation.
        let identical =
            left.variable() == right.variable() && !matches!(exact, [Some(left), Some(right)] if left != right);
        let equal = if identical
            || (left_minimum == left_maximum && right_minimum == right_maximum && left_minimum == right_minimum)
        {
            Some(true)
        } else if left_maximum < right_minimum || right_maximum < left_minimum {
            Some(false)
        } else {
            None
        };

        Ok(match self {
            Self::Equal => equal,
            Self::NotEqual => equal.map(|equal| !equal),
            Self::LessThan => {
                if identical || left_minimum >= right_maximum {
                    Some(false)
                } else if left_maximum < right_minimum {
                    Some(true)
                } else {
                    None
                }
            }
            Self::LessThanOrEqual => {
                if identical || left_maximum <= right_minimum {
                    Some(true)
                } else if left_minimum > right_maximum {
                    Some(false)
                } else {
                    None
                }
            }
            Self::GreaterThan => Self::LessThan.prove_for_dimensions(right, left, [exact[1], exact[0]])?,
            Self::GreaterThanOrEqual => {
                Self::LessThanOrEqual.prove_for_dimensions(right, left, [exact[1], exact[0]])?
            }
        })
    }
}

impl Display for ComparisonDirection {
    #[inline]
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
/// instantiations compare scalar element types and arrays, respectively. The [`ArrayIrType`] instantiation accepts
/// two dimension inputs and produces a rank-zero Boolean array. It does not accept array-member inputs. Refer to
/// [`Compare`] for the corresponding value-level semantics.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct CompareOperation<T: Type> {
    /// [`ComparisonDirection`] used by this [`CompareOperation`].
    direction: ComparisonDirection,

    /// [`PhantomData`] marker tying this [`Operation`] to the [`Type`] universe in which it is valid.
    marker: PhantomData<T>,
}

impl<T: Type> CompareOperation<T> {
    /// Creates a new [`CompareOperation`] with the provided [`ComparisonDirection`].
    #[inline]
    pub fn new(direction: ComparisonDirection) -> Self {
        Self { direction, marker: PhantomData }
    }

    /// Returns the [`ComparisonDirection`] used by this [`CompareOperation`].
    #[inline]
    pub fn direction(&self) -> ComparisonDirection {
        self.direction
    }
}

impl<T: Type> Copy for CompareOperation<T> {}

impl<T: Type> Clone for CompareOperation<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Type> Display for CompareOperation<T>
where
    Self: Operation<Type = T>,
{
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: ComparisonTypeSemantics> Operation for CompareOperation<T> {
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        COMPARE_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        Ok(vec![input_types[0].infer_comparison_output_type(&input_types[1], self.direction)?])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
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

impl_reference_dischargeable_operation!(@reference_free <T> CompareOperation<T> where T: Type);

impl<D: Domain<Type: ComparisonTypeSemantics, Value: Compare<D::Value>>> InterpretableOperation<D>
    for CompareOperation<D::Type>
{
    #[inline]
    fn interpret<I: InterpretationDriver<D>>(
        &self,
        _context: &D,
        _driver: &I,
        inputs: &[D::Value],
    ) -> Result<Vec<D::Value>, ProgramError> {
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

impl<
    C: Context<
            Type = ArrayIrType,
            Constant: TryFrom<bool, Error = ProgramError> + ValueProjection<DimensionType, Projected = DimensionValue>,
            Operation: From<Self>,
        >,
> PartiallyEvaluatableOperation<C> for CompareOperation<ArrayIrType>
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // Validate inputs and regions before attempting to eliminate the comparison.
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

        // A proven predicate is a Boolean constant. Unresolved predicates follow ordinary partial evaluation.
        if let Some(output) = self.direction.prove_for_dimensions(left, right, exact)? {
            return Ok(vec![PartialEvaluationValue::known(context.parent().lift(C::Constant::try_from(output)?)?)]);
        }

        context.fold_or_residualize(*self, regions, inputs)
    }
}

impl<C: Context<Type = ArrayIrType, Operation: From<CompareOperation<ArrayIrType>>>>
    BatchableOperation<C, ArrayIrBatchingPolicy> for CompareOperation<ArrayIrType>
{
    fn batch<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        _driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatchingPolicy>, BatchingError> {
        // Dimension inputs describe one shared array shape and must therefore remain replicated;
        // their Boolean array output is replicated ordinary data.
        check_count!("input", inputs, 2, ProgramError);
        let left = &inputs[0];
        let right = &inputs[1];
        left.validate_replicated_dimension()?;
        right.validate_replicated_dimension()?;
        Ok(context
            .parent()
            .bind(*self, Vec::new(), &[left.value().clone(), right.value().clone()])?
            .into_iter()
            .map(ArrayIrBatch::replicated)
            .collect::<Vec<_>>()
            .into())
    }
}

impl_non_differentiable_operation!(<T> CompareOperation<T> where T: Type);
impl_non_transposable_operation!(<T> CompareOperation<T> where T: Type);

/// Type-family comparison semantics for [`CompareOperation`].
pub trait ComparisonTypeSemantics: Type {
    /// Infers the Boolean output type for comparing this type with `other` in the given `direction`.
    /// Returns an error if the inputs are incompatible or the direction is unsupported for their types.
    ///
    /// # Parameters
    ///
    ///   - `other`: Type of the second comparison input.
    ///   - `direction`: Predicate applied to the two inputs.
    fn infer_comparison_output_type(&self, other: &Self, direction: ComparisonDirection) -> Result<Self, TypeError>;
}

impl ComparisonTypeSemantics for DataType {
    fn infer_comparison_output_type(&self, other: &Self, direction: ComparisonDirection) -> Result<Self, TypeError> {
        // Complex inputs are unordered, so only the equality comparison directions are defined for them.
        if !matches!(direction, ComparisonDirection::Equal | ComparisonDirection::NotEqual)
            && (self.is_complex() || other.is_complex())
        {
            return Err(TypeError::invalid(format!(
                "cannot apply an ordered comparison to unordered complex inputs of types `{}` and `{}`",
                self, other,
            )));
        }

        let broadcasted = DataType::broadcasted(&[self, other])
            .map_err(|_| TypeError::invalid("comparison input types are not broadcast-compatible"))?;

        if matches!(broadcasted, DataType::Token | DataType::Zero) {
            return Err(TypeError::invalid(format!("cannot compare `{broadcasted}` scalars")));
        }

        Ok(broadcasted.with_element_type(DataType::Boolean))
    }
}

impl ComparisonTypeSemantics for ArrayType {
    fn infer_comparison_output_type(&self, other: &Self, direction: ComparisonDirection) -> Result<Self, TypeError> {
        // Complex inputs are unordered, so only the equality comparison directions are defined for them.
        if !matches!(direction, ComparisonDirection::Equal | ComparisonDirection::NotEqual)
            && (self.is_complex() || other.is_complex())
        {
            return Err(TypeError::invalid(format!(
                "cannot apply an ordered comparison to unordered complex inputs of types `{}` and `{}`",
                self, other,
            )));
        }

        // Comparisons cannot operate on partial sums or implicitly change an input's reduction state.
        check_types!(@no_unreduced, COMPARE_OPERATION_NAME, [self, other]);
        check_types!(@same_reduced_axes, COMPARE_OPERATION_NAME, [self, other]);
        ArrayType::check_matching_manual_variation(COMPARE_OPERATION_NAME, &[self, other])?;

        let broadcasted = ArrayType::broadcasted(&[self, other])
            .map_err(|_| TypeError::invalid("comparison input types are not broadcast-compatible"))?;
        let data_type = broadcasted.data_type();

        if matches!(data_type, DataType::Token | DataType::Zero) && broadcasted.shape().element_count()? != Some(0) {
            return Err(TypeError::invalid(format!("cannot compare `{data_type}` scalars")));
        }

        Ok(broadcasted.with_element_type(DataType::Boolean))
    }
}

impl ComparisonTypeSemantics for ArrayIrType {
    fn infer_comparison_output_type(&self, other: &Self, _direction: ComparisonDirection) -> Result<Self, TypeError> {
        // Validate that both inputs are dimensions. Array comparisons use the projected `ArrayType` operation instead.
        <&DimensionType>::try_from(self)?;
        <&DimensionType>::try_from(other)?;

        // Comparing first-class dimensions produces ordinary predicate data rather than another dimension value.
        Ok(ArrayType::scalar(DataType::Boolean).into())
    }
}

/// Represents the ability to compare two values and produce Boolean data. Array inputs are broadcast to a common
/// shape and their element types are promoted before comparison. The output has that shape and [`DataType::Boolean`]
/// elements. Equality and inequality support complex elements while ordered comparisons reject them. A comparison
/// involving a floating-point NaN value is false except for inequality. Empty eager arrays perform no element
/// comparisons and return an empty Boolean array, including for payload-free element types. Ordered complex comparisons
/// are rejected even for empty arrays. Inputs carrying unreduced mesh axes are rejected, and both inputs must carry the
/// same reduced-axis set.
///
/// Concrete arrays compare immediately. Context-carrying values apply [`CompareOperation`] through their context,
/// aligning manual variation first. First-class dimensions produce rank-zero Boolean arrays, and predicates proved
/// by their identities and extent bounds can be returned as constants without staging an operation. `Output` permits
/// an array output when the input type, such as [`DimensionValue`], cannot represent Boolean data.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{Array, ArrayIrValue, Compare, DimensionValue, ProgramError};
/// # fn main() -> Result<(), ProgramError> {
/// let left = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(3)?);
/// let right = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(5)?);
/// assert_eq!(left.less_than(&right)?, ArrayIrValue::Array(Array::scalar(true)?));
/// # Ok(())
/// # }
/// ```
pub trait Compare<Output = Self>: Sized {
    /// Compares this value with `other` according to `direction`, returning Boolean data. Returns an error if the
    /// input types cannot be promoted or broadcast together, or if the direction is unsupported for their elements.
    /// Refer to [`Compare`] for dimension and contextual behavior.
    ///
    /// # Parameters
    ///
    ///   - `other`: Value to compare with this value.
    ///   - `direction`: Predicate to apply to each pair of elements or to the two dimension extents.
    fn compare(&self, other: &Self, direction: ComparisonDirection) -> Result<Output, ProgramError>;

    /// Returns the Boolean output of `self == other`, with the broadcasting and type rules of [`Self::compare`].
    #[inline]
    fn equal(&self, other: &Self) -> Result<Output, ProgramError> {
        self.compare(other, ComparisonDirection::Equal)
    }

    /// Returns the Boolean output of `self != other`, with the broadcasting and type rules of [`Self::compare`].
    #[inline]
    fn not_equal(&self, other: &Self) -> Result<Output, ProgramError> {
        self.compare(other, ComparisonDirection::NotEqual)
    }

    /// Returns the Boolean output of `self < other`, with the broadcasting and type rules of [`Self::compare`].
    #[inline]
    fn less_than(&self, other: &Self) -> Result<Output, ProgramError> {
        self.compare(other, ComparisonDirection::LessThan)
    }

    /// Returns the Boolean output of `self <= other`, with the broadcasting and type rules of [`Self::compare`].
    #[inline]
    fn less_than_or_equal(&self, other: &Self) -> Result<Output, ProgramError> {
        self.compare(other, ComparisonDirection::LessThanOrEqual)
    }

    /// Returns the Boolean output of `self > other`, with the broadcasting and type rules of [`Self::compare`].
    #[inline]
    fn greater_than(&self, other: &Self) -> Result<Output, ProgramError> {
        self.compare(other, ComparisonDirection::GreaterThan)
    }

    /// Returns the Boolean output of `self >= other`, with the broadcasting and type rules of [`Self::compare`].
    #[inline]
    fn greater_than_or_equal(&self, other: &Self) -> Result<Output, ProgramError> {
        self.compare(other, ComparisonDirection::GreaterThanOrEqual)
    }
}

impl Compare for Array {
    fn compare(&self, other: &Self, direction: ComparisonDirection) -> Result<Self, ProgramError> {
        // Broadcast the input types together (including element-type promotion) so that mixed-precision comparisons
        // mirror the `CompareOperation` type-inference contract, and then compare the promoted elements pairwise.
        // The output type is the Boolean-typed counterpart of the broadcast type.
        let output_type = self.r#type().infer_comparison_output_type(other.r#type().as_ref(), direction)?;
        let (broadcast_type, inputs) = Self::broadcast_promoted(&[self, other])?;
        let data_type = broadcast_type.data_type();

        // Empty comparisons inspect no elements, so they succeed vacuously even for payload-free data types.
        if Self::element_count(&output_type) == 0 {
            let addressing = ArrayAddressing::new(output_type.clone())?;
            return Ok(Self::new_unchecked(output_type, Arc::new(vec![0; addressing.storage_byte_len()])));
        }

        // `broadcast_promoted` converts only mismatched inputs, so equal-typed inputs retain their exact physical
        // storage and are decoded one addressed element at a time by the shared binary loop.
        let [left, right] = <[_; 2]>::try_from(inputs).unwrap();
        if data_type.is_complex() {
            // The shared inference rule has already restricted complex comparisons to equality and inequality.
            let equal = matches!(direction, ComparisonDirection::Equal);
            return dispatch_on_array_element_type!(@complex data_type, |Element| {
                left.map_element_pairs::<Element, bool>(&right, output_type, |left, right| {
                    Ok(if equal { left == right } else { left != right })
                })
            });
        }

        dispatch_on_array_element_type!(@ordered data_type, |Element| {
            left.map_element_pairs::<Element, bool>(&right, output_type, |left, right| {
                // An unordered pair (e.g., a comparison involving a floating-point NaN) satisfies only `NotEqual`.
                let ordering = left.partial_cmp(&right);
                Ok(match direction {
                    ComparisonDirection::Equal => ordering == Some(Ordering::Equal),
                    ComparisonDirection::NotEqual => ordering != Some(Ordering::Equal),
                    ComparisonDirection::LessThan => ordering == Some(Ordering::Less),
                    ComparisonDirection::LessThanOrEqual => matches!(ordering, Some(Ordering::Less | Ordering::Equal)),
                    ComparisonDirection::GreaterThan => ordering == Some(Ordering::Greater),
                    ComparisonDirection::GreaterThanOrEqual => {
                        matches!(ordering, Some(Ordering::Greater | Ordering::Equal))
                    }
                })
            })
        })
    }
}

impl Compare<Array> for DimensionValue {
    #[inline]
    fn compare(&self, other: &Self, direction: ComparisonDirection) -> Result<Array, ProgramError> {
        Array::scalar(match direction {
            ComparisonDirection::Equal => self.extent() == other.extent(),
            ComparisonDirection::NotEqual => self.extent() != other.extent(),
            ComparisonDirection::LessThan => self.extent() < other.extent(),
            ComparisonDirection::LessThanOrEqual => self.extent() <= other.extent(),
            ComparisonDirection::GreaterThan => self.extent() > other.extent(),
            ComparisonDirection::GreaterThanOrEqual => self.extent() >= other.extent(),
        })
    }
}

impl<A: Value<Type = ArrayType> + TryFrom<bool, Error = ProgramError>> Compare<ArrayIrValue<A>> for DimensionValue {
    fn compare(&self, other: &Self, direction: ComparisonDirection) -> Result<ArrayIrValue<A>, ProgramError> {
        let output = direction.prove_for_dimensions(
            self.r#type().as_ref(),
            other.r#type().as_ref(),
            [Some(self.extent()), Some(other.extent())],
        )?;

        // Both extents are known, so the comparison always resolves to a Boolean and so the `.unwrap()` is safe.
        let output = output.unwrap();
        Ok(ArrayIrValue::Array(A::try_from(output)?))
    }
}

impl<A: Value<Type = ArrayType>> Compare for ArrayIrValue<A>
where
    DimensionValue: Compare<A>,
{
    fn compare(&self, other: &Self, direction: ComparisonDirection) -> Result<Self, ProgramError> {
        let left = <Self as ValueProjection<DimensionType>>::projected(self)?;
        let right = <Self as ValueProjection<DimensionType>>::projected(other)?;
        Ok(Self::Array(left.compare(right, direction)?))
    }
}

impl<
    V: Value<
            Type = ArrayIrType,
            DispatchDomain: Context<
                Type = ArrayIrType,
                Constant: TryFrom<bool, Error = ProgramError>
                              + ValueProjection<DimensionType, Projected = DimensionValue>,
                Operation: From<CompareOperation<V::Type>>,
            >,
        >,
> Compare<V> for ProjectedValue<DimensionType, V>
{
    fn compare(&self, other: &Self, direction: ComparisonDirection) -> Result<V, ProgramError> {
        // Immediate extents override nominal identity, just as in partial evaluation. Captures remain runtime data.
        let mut exact = [None, None];
        for (value, extent) in [self.value(), other.value()].into_iter().zip(exact.iter_mut()) {
            if let ValueResolution::Constant(value) = value.dispatch_domain().resolve(value)
                && value.capture_index().is_none()
            {
                *extent = Some(value.into_projected()?.extent());
            }
        }

        if let Some(output) = direction.prove_for_dimensions(self.r#type().as_ref(), other.r#type().as_ref(), exact)? {
            return self.value().dispatch_domain().lift(<V::DispatchDomain as Domain>::Constant::try_from(output)?);
        }

        Ok(self
            .value()
            .dispatch_domain()
            .bind(CompareOperation::new(direction), Vec::new(), &[self.value().clone(), other.value().clone()])?
            .remove(0))
    }
}

impl<
    T: Type,
    V: Value<Type = T, DispatchDomain: Context<Operation: From<CompareOperation<T>>>> + ManualVariationAlignment<T>,
> Compare<V> for V
{
    #[inline]
    fn compare(&self, other: &Self, direction: ComparisonDirection) -> Result<Self, ProgramError> {
        let inputs = [self.clone(), other.clone()];
        let inputs = ManualVariationAlignment::align_manual_variation(&inputs)?;
        Ok(self.dispatch_domain().bind(CompareOperation::new(direction), Vec::new(), &inputs)?.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType, Dimension,
        DimensionBounds, DimensionError, DimensionType, DimensionValue, Layout, LogicalMesh, MAX_DIMENSION_EXTENT,
        Memory, MeshAxis, MeshAxisType, Shape, Sharding, StridedLayout, f8e5m2, i2,
    };
    use crate::axes::NamedAxis;
    use crate::batching::BatchAxis;
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{
        DifferentiableOperation, DifferentiationContext, DifferentiationDual, DifferentiationError,
        TransposableOperation, TranspositionContext, differentiate_at,
    };
    use crate::macros::{check_operation_batching, check_operation_partial_evaluation, check_operation_type_inference};
    use crate::operations::constants::zero_like::ZeroLike;
    use crate::operations::control_flow::select::Select;
    use crate::partial::PartialValue;
    use crate::programs::{
        EffectClasses, EmptyRegionDriver, MaybeZero, ProgramError, RegionInterface, Typed, ValueProjection,
    };
    use crate::tracing::{DomainTracingContext, TracingContext};

    use super::*;

    #[test]
    fn test_comparison_direction() {
        for (direction, expected) in [
            (ComparisonDirection::Equal, "Equal"),
            (ComparisonDirection::NotEqual, "NotEqual"),
            (ComparisonDirection::LessThan, "LessThan"),
            (ComparisonDirection::LessThanOrEqual, "LessThanOrEqual"),
            (ComparisonDirection::GreaterThan, "GreaterThan"),
            (ComparisonDirection::GreaterThanOrEqual, "GreaterThanOrEqual"),
        ] {
            assert_eq!(direction.to_string(), expected);
            assert_eq!(format!("{direction:?}"), expected);
        }
    }

    #[test]
    fn test_comparison_direction_prove_for_dimensions() {
        let directions = [
            ComparisonDirection::Equal,
            ComparisonDirection::NotEqual,
            ComparisonDirection::LessThan,
            ComparisonDirection::LessThanOrEqual,
            ComparisonDirection::GreaterThan,
            ComparisonDirection::GreaterThanOrEqual,
        ];

        // Exhaustively compare interval proofs with concrete outcomes over a small domain. This checks
        // endpoint strictness and overlapping intervals independently of the implementation's interval rules.
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
                            assert_eq!(direction.prove_for_dimensions(&left, &right, [None, None]).unwrap(), expected);
                        }
                    }
                }
            }
        }

        // Shared identities prove reflexive predicates even without finite bounds, including when only one side has a
        // known extent. Conflicting exact extents take precedence over the shared identity.
        let dimension = DimensionType::new("dimension", DimensionBounds::new(0, None).unwrap());
        for exact in [[None, None], [Some(4), None]] {
            for (direction, expected) in directions.into_iter().zip([true, false, false, true, false, true]) {
                assert_eq!(direction.prove_for_dimensions(&dimension, &dimension, exact).unwrap(), Some(expected));
            }
        }
        for (direction, expected) in directions.into_iter().zip([false, true, true, true, false, false]) {
            assert_eq!(
                direction.prove_for_dimensions(&dimension, &dimension, [Some(4), Some(5)]).unwrap(),
                Some(expected),
            );
        }

        // Lower bounds beyond the portable backend width have no representable extent interval.
        let oversized = DimensionType::new("oversized", DimensionBounds::new(usize::MAX, None).unwrap());
        assert_eq!(
            ComparisonDirection::Equal.prove_for_dimensions(&oversized, &dimension, [None, None]),
            Err(DimensionError::ExtentExceedsBackendWidth { value: usize::MAX, maximum: MAX_DIMENSION_EXTENT }.into()),
        );
    }

    #[test]
    fn test_compare() {
        let operation = CompareOperation::<ArrayType>::new(ComparisonDirection::LessThan);
        assert_eq!(operation.name(), COMPARE_OPERATION_NAME);
        assert_eq!(operation.direction(), ComparisonDirection::LessThan);
        assert_eq!(operation.to_string(), indoc! {"compare [direction=LessThan]"});
    }

    #[test]
    fn test_compare_type_inference() {
        // Scalar element types promote to a common type, and only equality comparisons accept complex inputs.
        let ordered_scalar = CompareOperation::<DataType>::new(ComparisonDirection::LessThan);
        check_operation_type_inference!(
            operation = ordered_scalar,
            cases = [{
                input_types = [DataType::F32, DataType::F64],
                output_types = [DataType::Boolean],
            }, {
                input_types = [DataType::F8E3M4, DataType::F32],
                error = "comparison input types are not broadcast-compatible",
            }, {
                input_types = [DataType::C64, DataType::C64],
                error = "cannot apply an ordered comparison to unordered complex inputs of types `c64` and `c64`",
            }],
        );

        check_operation_type_inference!(
            operation = CompareOperation::<DataType>::new(ComparisonDirection::Equal),
            cases = [{
                input_types = [DataType::C64, DataType::C64],
                output_types = [DataType::Boolean],
            }],
        );

        // Array comparisons preserve shape and placement but clear byte strides because Boolean elements change the
        // storage width. Only equality comparisons accept complex arrays.
        let left = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![24, 8])))
            .with_memory(Memory::Host { pinned: true });
        let right = left.clone().with_data_type(DataType::F64);
        let ordered_array = CompareOperation::<ArrayType>::new(ComparisonDirection::LessThan);
        assert_eq!(
            Operation::infer_output_types(&ordered_array, &[left.clone(), right], &[]),
            Ok(vec![left.clone().with_data_type(DataType::Boolean).with_layout(None)]),
        );
        let complex = left.with_data_type(DataType::C64);
        assert_eq!(
            Operation::infer_output_types(
                &CompareOperation::<ArrayType>::new(ComparisonDirection::Equal),
                &[complex.clone(), complex.clone()],
                &[],
            ),
            Ok(vec![complex.clone().with_data_type(DataType::Boolean).with_layout(None)]),
        );
        assert_eq!(
            Operation::infer_output_types(
                &ordered_array,
                &[ArrayType::scalar(DataType::C64), ArrayType::scalar(DataType::C64)],
                &[],
            ),
            Err(TypeError::invalid(
                "cannot apply an ordered comparison to unordered complex inputs of types `c64[]` and `c64[]`",
            )),
        );

        // Dimension comparisons accept only dimension inputs and produce a rank-zero Boolean array.
        let bounds = DimensionBounds::new(0, Some(9)).unwrap();
        let left = DimensionType::new("left", bounds);
        let right = DimensionType::new("right", bounds);
        let dimension_operation = CompareOperation::<ArrayIrType>::new(ComparisonDirection::LessThan);
        check_operation_type_inference!(
            operation = dimension_operation,
            cases = [{
                input_types = [left.clone().into(), right.clone().into()],
                output_types = [ArrayType::scalar(DataType::Boolean).into()],
            }, {
                input_types = [ArrayType::scalar(DataType::I64).into(), right.clone().into()],
                error = "expected dimension type but got array type",
            }, {
                input_types = [left.clone().into()],
                error = "expected 2 inputs but got 1",
            }],
        );

        // The shared operation contract rejects attached regions for every type universe.
        assert_eq!(
            ordered_scalar.infer_output_types(
                &[DataType::F32, DataType::F32],
                &[RegionInterface::new(Vec::new(), Vec::new(), EffectClasses::NONE)]
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
        assert_eq!(
            Operation::infer_output_types(
                &ordered_array,
                &[ArrayType::scalar(DataType::F32), ArrayType::scalar(DataType::F32)],
                &[RegionInterface::new(Vec::new(), Vec::new(), EffectClasses::NONE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
        assert_eq!(
            dimension_operation.infer_output_types(
                &[left.into(), right.into()],
                &[RegionInterface::new(Vec::new(), Vec::new(), EffectClasses::NONE)]
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_compare_type_inference_reduction_state() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let plain = ArrayType::scalar(DataType::F32);
        let replicated = plain.clone().with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
        let unreduced = plain
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 0).with_unreduced_axes(["m"]).unwrap())
            .unwrap();
        let reduced = plain
            .clone()
            .with_sharding(Sharding::replicated(mesh, 0).with_reduced_axes(["m"]).unwrap())
            .unwrap();
        check_operation_type_inference!(
            operation = CompareOperation::<ArrayType>::new(ComparisonDirection::Equal),
            cases = [{
                input_types = [unreduced.clone(), unreduced.clone()],
                error = "`compare` does not support unreduced operands",
            }, {
                input_types = [unreduced.clone(), plain.clone()],
                error = "`compare` does not support unreduced operands",
            }, {
                input_types = [reduced.clone(), plain.clone()],
                error = "`compare` operands must be reduced over the same axes",
            }, {
                input_types = [plain.clone(), reduced.clone()],
                error = "`compare` operands must be reduced over the same axes",
            }, {
                input_types = [reduced.clone(), replicated],
                error = "`compare` operands must be reduced over the same axes",
            }, {
                input_types = [reduced.clone(), reduced.clone()],
                output_types = [reduced.with_data_type(DataType::Boolean)],
            }],
        );
    }

    #[test]
    fn test_compare_type_inference_payload_free() {
        check_operation_type_inference!(
            operation = CompareOperation::<DataType>::new(ComparisonDirection::Equal),
            cases = [{
                input_types = [DataType::Token, DataType::Token],
                error = "cannot compare `token` scalars",
            }, {
                input_types = [DataType::Zero, DataType::Zero],
                error = "cannot compare `zero` scalars",
            }],
        );

        check_operation_type_inference!(
            operation = CompareOperation::<ArrayType>::new(ComparisonDirection::Equal),
            cases = [{
                input_types = [ArrayType::scalar(DataType::Token), ArrayType::scalar(DataType::Token)],
                error = "cannot compare `token` scalars",
            }, {
                input_types = [ArrayType::scalar(DataType::Zero), ArrayType::scalar(DataType::Zero)],
                error = "cannot compare `zero` scalars",
            }, {
                input_types = [ArrayType::new_static(DataType::Token, [0]), ArrayType::scalar(DataType::Token)],
                output_types = [ArrayType::new_static(DataType::Boolean, [0])],
            }, {
                input_types = [ArrayType::new_static(DataType::Zero, [0]), ArrayType::scalar(DataType::Zero)],
                output_types = [ArrayType::new_static(DataType::Boolean, [0])],
            }],
        );
    }

    #[test]
    fn test_compare_type_inference_manual_variation() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let invariant_type = ArrayType::scalar(DataType::F32);
        let varying_type = invariant_type
            .clone()
            .with_sharding(Sharding::replicated(mesh, 0).with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        let operation = CompareOperation::<ArrayType>::new(ComparisonDirection::Equal);
        assert_eq!(
            Operation::infer_output_types(&operation, &[varying_type.clone(), varying_type.clone()], &[]),
            Ok(vec![varying_type.clone().with_data_type(DataType::Boolean)]),
        );
        let error = TypeError::invalid(
            "`compare` inputs must have matching varying manual axes; insert `parallel_vary` on the inputs that lack \
             an axis, as `align_manual_variation` does",
        );
        assert_eq!(
            Operation::infer_output_types(&operation, &[invariant_type.clone(), varying_type.clone()], &[]),
            Err(error.clone()),
        );
        assert_eq!(Operation::infer_output_types(&operation, &[varying_type, invariant_type], &[]), Err(error));
    }

    #[test]
    fn test_compare_interpretation() {
        // Array inputs broadcast before comparing, whereas dimension inputs compare their extents.
        assert_eq!(
            CompareOperation::new(ComparisonDirection::GreaterThan).interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::vector(vec![1.0, -2.0]).unwrap(), Array::scalar(0.0).unwrap()],
            ),
            Ok(vec![Array::vector(vec![true, false]).unwrap()]),
        );
        assert_eq!(
            CompareOperation::new(ComparisonDirection::LessThan).interpret(
                &EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
                &EmptyRegionDriver,
                &[
                    ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
                    ArrayIrValue::Dimension(DimensionValue::constant(5).unwrap()),
                ],
            ),
            Ok(vec![ArrayIrValue::Array(Array::scalar(true).unwrap())]),
        );
    }

    #[test]
    fn test_compare_partial_evaluation() {
        // Known array inputs fold to a Boolean constant.
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
                // Concrete dimensions fold to an ordinary Boolean array.
                {
                    inputs = [
                        (@known, ArrayIrValue::Dimension(DimensionValue::new(left_type.clone(), 3).unwrap())),
                        (@known, ArrayIrValue::Dimension(DimensionValue::new(right_type.clone(), 5).unwrap())),
                    ],
                    outputs = [
                        (@known, ArrayIrValue::Array(Array::scalar(true).unwrap())),
                    ],
                    residual_instructions = 0,
                },
                // Overlapping bounds of independent runtime dimensions require a residual comparison.
                {
                    inputs = [
                        (@unknown(
                            type = ArrayIrType::Dimension(left_type.clone()),
                            replay = ArrayIrValue::Dimension(DimensionValue::new(left_type.clone(), 3).unwrap()),
                        )),
                        (@unknown(
                            type = ArrayIrType::Dimension(right_type.clone()),
                            replay = ArrayIrValue::Dimension(DimensionValue::new(right_type.clone(), 5).unwrap()),
                        )),
                    ],
                    outputs = [
                        (@residual, ArrayIrValue::Array(Array::scalar(true).unwrap())),
                    ],
                    residual_instructions = 1,
                },
                // A shared dimension identity proves that strict self-comparison is false.
                {
                    inputs = [
                        (@unknown(
                            type = ArrayIrType::Dimension(left_type.clone()),
                            replay = ArrayIrValue::Dimension(DimensionValue::new(left_type.clone(), 3).unwrap()),
                        )),
                        (@unknown(
                            type = ArrayIrType::Dimension(left_type.clone()),
                            replay = ArrayIrValue::Dimension(DimensionValue::new(left_type.clone(), 3).unwrap()),
                        )),
                    ],
                    outputs = [(@known, ArrayIrValue::Array(Array::scalar(false).unwrap()))],
                    residual_instructions = 0,
                },
                // A known endpoint can prove a predicate against every extent in the unknown input's bounds.
                {
                    inputs = [
                        (@known, ArrayIrValue::Dimension(DimensionValue::new(left_type.clone(), 8).unwrap())),
                        (@unknown(
                            type = ArrayIrType::Dimension(right_type.clone()),
                            replay = ArrayIrValue::Dimension(DimensionValue::new(right_type.clone(), 5).unwrap()),
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
        // Array comparisons map elementwise over the batch axis, whichever input carries it.
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
    fn test_compare_batching_dimensions() {
        // A comparison of shared dimensions produces one replicated Boolean, regardless of the batch extent.
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let left = ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        let right = ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(5).unwrap()));
        let operation = CompareOperation::<ArrayIrType>::new(ComparisonDirection::LessThan);
        let (outputs, _) = operation.batch(&context, &EmptyRegionDriver, &[left, right.clone()]).unwrap().into_parts();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value(), &ArrayIrValue::Array(Array::scalar(true).unwrap()));

        // A mapped dimension carries one extent per batch item, so it cannot stand in for one shared shape extent.
        let dimension_type = DimensionType::new("extent", DimensionBounds::new(0, Some(8)).unwrap());
        let mapped = ArrayIrBatch::mapped_dimension(
            ArrayIrValue::Array(Array::vector(vec![1i32, 3]).unwrap()),
            BatchAxis::new(0),
            dimension_type.clone(),
        )
        .unwrap();
        assert_eq!(
            operation.batch(&context, &EmptyRegionDriver, &[mapped, right]).map(|_| ()),
            Err(BatchingError::MappedDimension { r#type: Box::new(dimension_type), axis: BatchAxis::new(0) }),
        );
    }

    #[test]
    fn test_compare_differentiation() {
        // Comparisons are non-differentiable, so their Boolean output has a structural zero tangent even when the
        // input tangents are nonzero.
        let outputs = CompareOperation::<ArrayType>::new(ComparisonDirection::GreaterThan)
            .jvp(
                &DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new()),
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new(Array::scalar(2.0).unwrap(), Array::scalar(1.0).unwrap()).unwrap(),
                    DifferentiationDual::new(Array::scalar(0.0).unwrap(), Array::scalar(1.0).unwrap()).unwrap(),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &Array::scalar(true).unwrap());
        assert!(
            matches!(outputs[0].tangent(), MaybeZero::Zero(r#type) if r#type == &ArrayType::scalar(DataType::Zero))
        );

        // A comparison that selects between differentiable branches contributes nothing to the derivative, so
        // `select(x > 0, 2x, 3x)` differentiates to the slope of the selected branch.
        for (input, primal, tangent) in [(2.0, 4.0, 2.0), (-2.0, -6.0, 3.0)] {
            assert_eq!(
                differentiate_at(Array::scalar(input).unwrap()).jvp(Array::scalar(1.0).unwrap(), |input| {
                    let condition = input.greater_than(&input.zero_like()?)?;
                    Select::select(
                        &condition,
                        &(input.clone() + input.clone()),
                        &(input.clone() + input.clone() + input),
                    )
                }),
                Ok((Array::scalar(primal).unwrap(), Array::scalar(tangent).unwrap())),
            );
        }
    }

    #[test]
    fn test_compare_differentiation_dimensions() {
        // Dimension inputs and Boolean outputs have no tangent slots, so the differentiated program only replays the
        // primal comparison.
        let bounds = DimensionBounds::new(0, Some(9)).unwrap();
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |inputs: Vec<_>| Ok(vec![inputs[0].less_than(&inputs[1])?]),
            vec![
                ArrayIrType::from(DimensionType::new("left", bounds)),
                ArrayIrType::from(DimensionType::new("right", bounds)),
            ],
        )
        .unwrap();
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:dimension<left ∈ [0, 9)>, %1:dimension<right ∈ [0, 9)> .
                let %2:bool[] = compare [direction=LessThan] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_compare_transposition() {
        // Program transposition elides the zero-space cotangents of Boolean outputs, so check the primitive's
        // rejection directly.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        assert!(matches!(
            CompareOperation::<ArrayType>::new(ComparisonDirection::LessThan).transpose(
                &mut TranspositionContext::new(context),
                &EmptyRegionDriver,
                &[
                    PartialValue::Unknown(ArrayType::scalar(DataType::F64)),
                    PartialValue::Unknown(ArrayType::scalar(DataType::F64)),
                ],
                &[MaybeZero::Zero(ArrayType::scalar(DataType::Zero))],
                &[],
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `compare` is not transposable",
        ));
    }

    #[test]
    fn test_compare_for_array() {
        let left = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        let right = Array::vector(vec![2.0, 2.0, 2.0]).unwrap();
        assert_eq!(
            left.compare(&right, ComparisonDirection::LessThan),
            Ok(Array::vector(vec![true, false, false]).unwrap()),
        );
        assert_eq!(left.equal(&right), Ok(Array::vector(vec![false, true, false]).unwrap()));
        assert_eq!(left.not_equal(&right), Ok(Array::vector(vec![true, false, true]).unwrap()));
        assert_eq!(left.less_than(&right), Ok(Array::vector(vec![true, false, false]).unwrap()));
        assert_eq!(left.less_than_or_equal(&right), Ok(Array::vector(vec![true, true, false]).unwrap()));
        assert_eq!(left.greater_than(&right), Ok(Array::vector(vec![false, false, true]).unwrap()));
        assert_eq!(left.greater_than_or_equal(&right), Ok(Array::vector(vec![false, true, true]).unwrap()));
    }

    #[test]
    fn test_compare_for_array_reduction_state() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let unreduced_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.clone(), 0).with_unreduced_axes(["m"]).unwrap())
            .unwrap();
        let unreduced = Array::from_elements(unreduced_type, &[1f32]).unwrap();
        assert_eq!(
            unreduced.equal(&unreduced),
            Err(TypeError::invalid("`compare` does not support unreduced operands").into()),
        );

        // A replicated peer cannot acquire another input's reduced state implicitly.
        let reduced_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh, 0).with_reduced_axes(["m"]).unwrap())
            .unwrap();
        let reduced = Array::from_elements(reduced_type.clone(), &[1f32]).unwrap();
        assert_eq!(
            reduced.equal(&Array::scalar(1f32).unwrap()),
            Err(TypeError::invalid("`compare` operands must be reduced over the same axes").into()),
        );
        assert_eq!(
            reduced.equal(&reduced),
            Array::from_elements(reduced_type.with_data_type(DataType::Boolean), &[true]),
        );
    }

    #[test]
    fn test_compare_for_array_manual_variation() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let invariant_type = ArrayType::scalar(DataType::F32);
        let varying_type = invariant_type
            .clone()
            .with_sharding(Sharding::replicated(mesh, 0).with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        let invariant = Array::from_elements(invariant_type, &[1f32]).unwrap();
        let varying = Array::from_elements(varying_type.clone(), &[1f32]).unwrap();
        assert_eq!(
            varying.equal(&varying),
            Array::from_elements(varying_type.with_data_type(DataType::Boolean), &[true]),
        );

        // Eager arrays have no context in which to insert `parallel_vary`, so mismatched variation is rejected.
        let error = TypeError::invalid(
            "`compare` inputs must have matching varying manual axes; insert `parallel_vary` on the inputs that lack \
             an axis, as `align_manual_variation` does",
        );
        assert_eq!(invariant.equal(&varying), Err(error.clone().into()));
        assert_eq!(varying.equal(&invariant), Err(error.into()));
    }

    #[test]
    fn test_compare_for_array_broadcasting() {
        // Inputs broadcast and promote to a common element type before comparing.
        assert_eq!(
            Array::vector(vec![1.0f32, 3.0]).unwrap().greater_than(&Array::scalar(2.0f64).unwrap()),
            Ok(Array::vector(vec![false, true]).unwrap()),
        );

        // Sub-byte integers compare by their signed values and broadcast along every size-one axis.
        let left = Array::matrix(2, 1, vec![i2::new(-1).unwrap(), i2::new(1).unwrap()]).unwrap();
        let right = Array::matrix(1, 3, vec![i2::new(-2).unwrap(), i2::new(0).unwrap(), i2::new(1).unwrap()]).unwrap();
        assert_eq!(
            left.less_than(&right),
            Ok(Array::matrix(2, 3, vec![false, true, true, false, false, false]).unwrap()),
        );
    }

    #[test]
    fn test_compare_for_array_layout() {
        // Addressed inputs retain their physical layout, whereas the narrower Boolean output clears byte strides.
        let strided_type =
            ArrayType::new_static(DataType::U16, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let left = Array::from_elements(strided_type.clone(), &[1u16, 3]).unwrap();
        let right = Array::from_elements(strided_type, &[2u16, 2]).unwrap();
        let compared = left.less_than(&right).unwrap();
        assert_eq!(compared.elements::<bool>(), Ok(vec![true, false]));
        assert_eq!(compared.r#type().layout(), None);
        assert_eq!(compared.storage_bytes(), [1, 0]);
    }

    #[test]
    fn test_compare_for_array_nan() {
        // Floating-point NaNs are unordered, so a comparison involving one is false for every direction except
        // `NotEqual`, whether the other element is a NaN or an ordinary value.
        let left = Array::vector(vec![f8e5m2::NAN, f8e5m2::NAN]).unwrap();
        let right = Array::vector(vec![f8e5m2::NAN, f8e5m2::from_f64(1.0).unwrap()]).unwrap();
        for (direction, expected) in [
            (ComparisonDirection::Equal, false),
            (ComparisonDirection::NotEqual, true),
            (ComparisonDirection::LessThan, false),
            (ComparisonDirection::LessThanOrEqual, false),
            (ComparisonDirection::GreaterThan, false),
            (ComparisonDirection::GreaterThanOrEqual, false),
        ] {
            assert_eq!(left.compare(&right, direction), Ok(Array::vector(vec![expected, expected]).unwrap()));
        }
    }

    #[test]
    fn test_compare_for_array_complex() {
        // Complex elements are unordered, so they support only equality and inequality.
        let left = Array::vector(vec![ComplexNumber::new(1.0f32, 2.0), ComplexNumber::new(1.0f32, -2.0)]).unwrap();
        let right = Array::vector(vec![ComplexNumber::new(1.0f32, 2.0), ComplexNumber::new(1.0f32, 2.0)]).unwrap();
        assert_eq!(left.equal(&right), Ok(Array::vector(vec![true, false]).unwrap()));
        assert_eq!(left.not_equal(&right), Ok(Array::vector(vec![false, true]).unwrap()));
        assert_eq!(
            left.less_than(&right),
            Err(TypeError::invalid(
                "cannot apply an ordered comparison to unordered complex inputs of types `c64[2]` and `c64[2]`",
            )
            .into()),
        );
    }

    #[test]
    fn test_compare_for_array_complex_empty() {
        // The element-domain contract applies even when there are no elements to compare.
        let empty = Array::vector(Vec::<ComplexNumber<f32>>::new()).unwrap();
        assert_eq!(empty.equal(&empty), Ok(Array::vector(Vec::<bool>::new()).unwrap()));
        let expected = TypeError::invalid(
            "cannot apply an ordered comparison to unordered complex inputs of types `c64[0]` and `c64[0]`",
        );
        assert_eq!(empty.less_than(&empty), Err(expected.clone().into()));
        assert_eq!(
            Operation::infer_output_types(
                &CompareOperation::<ArrayType>::new(ComparisonDirection::LessThan),
                &[empty.r#type().into_owned(), empty.r#type().into_owned()],
                &[],
            ),
            Err(expected),
        );
    }

    #[test]
    fn test_compare_for_array_payload_free() {
        // Empty payload-free comparisons inspect no elements, so they succeed vacuously, whereas nonempty
        // payload-free arrays have no elements to compare.
        let empty_token = Array::from_logical_bytes(ArrayType::new_static(DataType::Token, [0]), &[]).unwrap();
        assert_eq!(empty_token.equal(&empty_token), Ok(Array::vector(Vec::<bool>::new()).unwrap()));
        let token = Array::from_logical_bytes(ArrayType::new_static(DataType::Token, [1]), &[]).unwrap();
        assert_eq!(token.equal(&token), Err(TypeError::invalid("cannot compare `token` scalars").into()));
        let zero = Array::from_logical_bytes(ArrayType::new_static(DataType::Zero, [1]), &[]).unwrap();
        assert_eq!(zero.equal(&zero), Err(TypeError::invalid("cannot compare `zero` scalars").into()));
    }

    #[test]
    fn test_compare_for_dimension_value() {
        let left = DimensionValue::constant(3).unwrap();
        let right = DimensionValue::constant(5).unwrap();
        assert_eq!(left.equal(&right), Ok(Array::scalar(false).unwrap()));
        assert_eq!(left.not_equal(&right), Ok(Array::scalar(true).unwrap()));
        assert_eq!(left.less_than(&right), Ok(Array::scalar(true).unwrap()));
        assert_eq!(left.less_than_or_equal(&right), Ok(Array::scalar(true).unwrap()));
        assert_eq!(left.greater_than(&right), Ok(Array::scalar(false).unwrap()));
        assert_eq!(left.greater_than_or_equal(&right), Ok(Array::scalar(false).unwrap()));

        // Mixed-family outputs are resolved by the dimension proof, in which concrete extents take precedence over a
        // shared dimension identity.
        let dimension_type = DimensionType::new("extent", DimensionBounds::new(0, Some(9)).unwrap());
        let three = DimensionValue::new(dimension_type.clone(), 3).unwrap();
        let five = DimensionValue::new(dimension_type, 5).unwrap();
        assert_eq!(
            Compare::<ArrayIrValue<Array>>::less_than(&three, &five),
            Ok(ArrayIrValue::Array(Array::scalar(true).unwrap())),
        );
        assert_eq!(
            Compare::<ArrayIrValue<Array>>::equal(&three, &three),
            Ok(ArrayIrValue::Array(Array::scalar(true).unwrap())),
        );
    }

    #[test]
    fn test_compare_for_array_ir_value() {
        // Dimension members compare their extents, whereas array members are rejected because array comparisons
        // use the projected `ArrayType` operation.
        let left = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(3).unwrap());
        let right = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(5).unwrap());
        assert_eq!(left.less_than(&right), Ok(ArrayIrValue::Array(Array::scalar(true).unwrap())));
        let array = ArrayIrValue::Array(Array::scalar(1.0).unwrap());
        assert_eq!(
            array.less_than(&array),
            Err(TypeError::invalid("expected dimension type but got array type").into())
        );
    }

    #[test]
    fn test_compare_for_projected_value() {
        // An unproven dimension predicate stages one comparison that produces a rank-zero Boolean array.
        let bounds = DimensionBounds::new(0, Some(9)).unwrap();
        let left_type = DimensionType::new("left", bounds);
        let right_type = DimensionType::new("right", bounds);
        let (output_type, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(left, right): (_, _)| {
                let left = ValueProjection::<DimensionType>::into_projected(left)?;
                let right = ValueProjection::<DimensionType>::into_projected(right)?;
                left.less_than(&right)
            },
            (ArrayIrType::from(left_type.clone()), ArrayIrType::from(right_type.clone())),
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<left ∈ [0, 9)>, %1:dimension<right ∈ [0, 9)> .
                let %2:bool[] = compare [direction=LessThan] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.interpret((
                ArrayIrValue::Dimension(DimensionValue::new(left_type, 3).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::new(right_type, 5).unwrap()),
            )),
            Ok(ArrayIrValue::Array(Array::scalar(true).unwrap())),
        );
    }

    #[test]
    fn test_compare_for_projected_value_dimension_proof() {
        // A predicate proved by a shared dimension identity is lifted as a constant without staging an instruction.
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let dimension = context.input(DimensionType::new("extent", DimensionBounds::new(0, None).unwrap()).into());
        let dimension = ValueProjection::<DimensionType>::into_projected(dimension).unwrap();
        let predicate = dimension.equal(&dimension).unwrap();
        assert!(matches!(
            context.resolve(&predicate),
            ValueResolution::Constant(ArrayIrValue::Array(value)) if value == Array::scalar(true).unwrap(),
        ));
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_compare_for_projected_value_constant_dimensions() {
        // Concrete extents take precedence over a shared symbolic identity during tracing, as they do eagerly.
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let dimension_type = DimensionType::new("extent", DimensionBounds::new(0, Some(9)).unwrap());
        let three = context
            .lift(ArrayIrValue::Dimension(DimensionValue::new(dimension_type.clone(), 3).unwrap()))
            .unwrap();
        let five = context.lift(ArrayIrValue::Dimension(DimensionValue::new(dimension_type, 5).unwrap())).unwrap();
        let three = ValueProjection::<DimensionType>::into_projected(three).unwrap();
        let five = ValueProjection::<DimensionType>::into_projected(five).unwrap();
        for (direction, expected) in [
            (ComparisonDirection::Equal, false),
            (ComparisonDirection::NotEqual, true),
            (ComparisonDirection::LessThan, true),
            (ComparisonDirection::LessThanOrEqual, true),
            (ComparisonDirection::GreaterThan, false),
            (ComparisonDirection::GreaterThanOrEqual, false),
        ] {
            let output = three.compare(&five, direction).unwrap();
            assert!(matches!(
                context.resolve(&output),
                ValueResolution::Constant(ArrayIrValue::Array(value)) if value == Array::scalar(expected).unwrap(),
            ));
        }
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_compare_for_tracer() {
        // Traced arrays stage the comparison through their context.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(left, right): (_, _)| left.less_than(&right),
            (ArrayType::new_static(DataType::F32, [2]), ArrayType::scalar(DataType::F64)),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f64[] .
                let %2:bool[2] = compare [direction=LessThan] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_compare_for_tracer_manual_variation() {
        // Traced comparisons align manual variation before staging by inserting `parallel_vary` on the invariant input.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        let (_, program) = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::trace_with_named_axes(
            |(invariant, varying): (_, _)| invariant.less_than(&varying),
            (ArrayType::scalar(DataType::F32), varying_type),
            vec![("m".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh })],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[][sharding={mesh<['m'=2:manual]>, [], varying_manual={'m'}}] .
                let %2:f32[][sharding={mesh<['m'=2:manual]>, []}] = broadcast \
                        [output_type=f32[][sharding={mesh<['m'=2:manual]>, []}], output_axes=[]] %0
                    %3:f32[][sharding={mesh<['m'=2:manual]>, [], varying_manual={'m'}}] = parallel_vary \
                        [axis_name=\"m\"] %2
                    %4:bool[][sharding={mesh<['m'=2:manual]>, [], varying_manual={'m'}}] = compare \
                        [direction=LessThan] %3 %1
                in (%4)
            "}
            .trim_end(),
        );
    }
}
