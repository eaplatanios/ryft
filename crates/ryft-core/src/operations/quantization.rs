//! Block-scaled (i.e., microscaling) quantization through [`BlockQuantize`] and [`ScaledDot`]. [`BlockQuantize`]
//! splits a floating-point array into an array of narrow elements and an array of block scales, and
//! [`ScaledDotOperation`] contracts such element and scale pairs as a generalized dot product of their dequantized
//! values. Together they cover the NVFP4 and [OCP MX] (e.g., MXFP8) formats that low precision matrix units consume
//! natively.
//!
//! [`ScaledDotOperation`] supports eager arrays, staged programs, array IR projections, partial evaluation, and
//! batching, and the XLA backend lowers it to the `xla.scaled_dot` composite. It intentionally has no differentiation
//! rule. [`BlockQuantize`] composes existing operations and therefore inherits all of their transform rules.
//!
//! # Example
//!
//! ```rust
//! # use ryft_core::{Array, BlockQuantize, DataType, DotDimensionNumbers, ProgramError, ScaledDot};
//! # fn main() -> Result<(), ProgramError> {
//! let input = Array::matrix(1, 8, vec![3.0f32, 1.5, 0.5, 6.0, 0.5, 1.0, 0.25, 1.5])?;
//! let (elements, scales) = input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN)?;
//! assert_eq!(scales.to_f64s(), vec![1.0, 0.25]);
//! let dimensions = DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new());
//! let product =
//!     elements.scaled_dot(&elements, Some(&scales), Some(&scales), Some(&dimensions), Some(DataType::F32))?;
//! assert_eq!(product.to_f64s(), vec![51.0625]);
//! # Ok(())
//! # }
//! ```
//!
//! [OCP MX]: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

use std::fmt::Display;

use ryft_macros::capability;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrType, ArrayIrValue, ArrayMemberType,
    ArrayType, DataType, Dimension, DimensionType, Shape,
};
use crate::axes::Axis;
use crate::batching::{
    BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError, InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    DifferentiationPolicy,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_transposable_operation};
use crate::operations::Capability;
use crate::operations::arithmetic::{Abs, Div, Mul, Rem, Sub};
use crate::operations::assertions::Assert;
use crate::operations::collectives::parallel_vary::ManualVariationAlignment;
use crate::operations::comparisons::{Compare, ComparisonDirection};
use crate::operations::constants::constant::DimensionConstant;
use crate::operations::constants::fill::Fill;
use crate::operations::dimensions::dimension_max::DimensionMax;
use crate::operations::dimensions::dimension_size::DimensionSize;
use crate::operations::dot::{Dot, DotDimensionNumbers, infer_dot_output_type, validate_common_mapped_dimension};
use crate::operations::exponential::{Exp, Log};
use crate::operations::extrema::Clamp;
use crate::operations::manipulation::broadcasting::{Broadcast, DynamicBroadcast};
use crate::operations::manipulation::conversions::ConvertElementType;
use crate::operations::manipulation::reshaping::{DynamicReshape, Reshape};
use crate::operations::reductions::{Reduce, ReductionKind};
use crate::operations::rounding::Floor;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    Operation, OperationFormatter, ProgramError, RegionInterface, TypeError, Typed, Value, ValueDomainDispatch,
    ValueProjection,
};

/// Canonical operation name for [`ScaledDotOperation`].
pub const SCALED_DOT_OPERATION_NAME: &str = "scaled_dot";

/// Primitive representing a generalized block-scaled dot product. Its first two inputs are the left and right element
/// arrays. The optional left and right scale arrays follow in that order, and [`Self::has_lhs_scale`] and
/// [`Self::has_rhs_scale`] record which of them are present. Element and scale arrays may have any real numeric or
/// Boolean element type, and all inputs must have the same rank and matching varying manual axes. Unreduced inputs are
/// rejected.
///
/// Each scale array has the same rank as its element array. Its non-contracting dimensions match the element array
/// exactly, and the extent of every contracting element dimension is an integer multiple of at least two of the
/// corresponding scale extent. That ratio is the block size, and it is inferred independently for every side and
/// contracting dimension. Type inference validates statically known ratios, and [`scaled_dot_ir_composition`] asserts
/// the ratios of dynamic extents at runtime.
///
/// Semantically, this operation expands each present scale to the shape of its element array by repeating every scale
/// value over its block, converts the elements and expanded scales to `bf16`, and multiplies them. It then contracts
/// the resulting arrays with a [`DotOperation`](crate::operations::DotOperation) that uses [`Self::dimensions`] and
/// accumulates at [`Self::preferred_element_type`], which is also the output element type. An absent scale leaves its
/// element array unchanged apart from the `bf16` conversion. [`scaled_dot_ir_composition`] implements this definition
/// once, and the XLA backend uses it as the decomposition of the `xla.scaled_dot` composite that it lowers this
/// operation to.
///
/// Partial evaluation folds the operation when all inputs are known and stages it otherwise. Batching moves every
/// mapped axis to the front and materializes replicated inputs along a new leading axis, because each scale array must
/// keep the rank of its element array. The lifted operation pairs those leading axes as an additional batching
/// dimension, so repeated batching applies the same lift again without a rank ceiling.
///
/// This operation has no differentiation rule. Its scales are quantization metadata rather than differentiable
/// parameters, and treating them as constants would silently define a straight-through estimator that is not part of
/// the operation's semantics. Callers that need a differentiable approximation can write the dequantization
/// composition explicitly and choose their own gradient policy. The `scaled_matmul` primitive of JAX likewise has no
/// differentiation rule.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScaledDotOperation {
    /// Contracting and batching dimension specification.
    dimensions: DotDimensionNumbers,

    /// Data type of the contraction accumulator and the output.
    preferred_element_type: DataType,

    /// Whether the operation has a left scale input.
    has_lhs_scale: bool,

    /// Whether the operation has a right scale input.
    has_rhs_scale: bool,
}

impl ScaledDotOperation {
    /// Creates a new [`ScaledDotOperation`].
    ///
    /// # Parameters
    ///
    ///   - `dimensions`: Contracting and batching dimensions of the generalized dot product.
    ///   - `preferred_element_type`: Data type of the contraction accumulator and the output. Type inference validates
    ///     that the `bf16` dequantized inputs can accumulate at this type.
    ///   - `has_lhs_scale`: Whether the inputs include a left scale array after the two element arrays.
    ///   - `has_rhs_scale`: Whether the inputs include a right scale array after the optional left scale array.
    #[inline]
    pub fn new(
        dimensions: DotDimensionNumbers,
        preferred_element_type: DataType,
        has_lhs_scale: bool,
        has_rhs_scale: bool,
    ) -> Self {
        Self { dimensions, preferred_element_type, has_lhs_scale, has_rhs_scale }
    }

    /// Returns the default dimension numbers that [`ScaledDot::scaled_dot`] uses for element arrays of rank `rank`.
    /// They contract the last left dimension with the penultimate right dimension and pair every preceding dimension
    /// as a batching dimension. For example, rank-two inputs describe the matrix product `[M, K] × [K, N] → [M, N]`
    /// and rank-three inputs describe the batched matrix product `[B, M, K] × [B, K, N] → [B, M, N]`. Returns a
    /// [`TypeError`] when `rank` is smaller than two.
    pub fn default_dimensions(rank: usize) -> Result<DotDimensionNumbers, TypeError> {
        if rank < 2 {
            return Err(TypeError::invalid(format!(
                "default `{SCALED_DOT_OPERATION_NAME}` dimension numbers require inputs of rank at least 2 but got \
                 rank {rank}",
            )));
        }
        Ok(DotDimensionNumbers::new(vec![rank - 1], vec![rank - 2], (0..rank - 2).collect(), (0..rank - 2).collect()))
    }

    /// Returns the contracting and batching dimension specification of this [`ScaledDotOperation`].
    #[inline]
    pub fn dimensions(&self) -> &DotDimensionNumbers {
        &self.dimensions
    }

    /// Returns the data type of the contraction accumulator and the output of this [`ScaledDotOperation`].
    #[inline]
    pub fn preferred_element_type(&self) -> DataType {
        self.preferred_element_type
    }

    /// Returns whether this [`ScaledDotOperation`] has a left scale input.
    #[inline]
    pub fn has_lhs_scale(&self) -> bool {
        self.has_lhs_scale
    }

    /// Returns whether this [`ScaledDotOperation`] has a right scale input.
    #[inline]
    pub fn has_rhs_scale(&self) -> bool {
        self.has_rhs_scale
    }

    /// Returns the number of inputs of this [`ScaledDotOperation`].
    #[inline]
    fn input_count(&self) -> usize {
        2 + usize::from(self.has_lhs_scale) + usize::from(self.has_rhs_scale)
    }

    /// Splits `inputs` into the left and right element inputs followed by the optional left and right scale inputs.
    fn split_inputs<'o, Input>(
        &self,
        inputs: &'o [Input],
    ) -> Result<(&'o Input, &'o Input, Option<&'o Input>, Option<&'o Input>), TypeError> {
        if inputs.len() != self.input_count() {
            return Err(TypeError::invalid(format!(
                "`{SCALED_DOT_OPERATION_NAME}` expects {} inputs but got {}",
                self.input_count(),
                inputs.len(),
            )));
        }
        let lhs_scale = self.has_lhs_scale.then(|| &inputs[2]);
        let rhs_scale = self.has_rhs_scale.then(|| &inputs[inputs.len() - 1]);
        Ok((&inputs[0], &inputs[1], lhs_scale, rhs_scale))
    }
}

impl Display for ScaledDotOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ScaledDotOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        SCALED_DOT_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        let (lhs, rhs, lhs_scale, rhs_scale) = self.split_inputs(input_types)?;
        ArrayType::check_matching_manual_variation(SCALED_DOT_OPERATION_NAME, &input_types.iter().collect::<Vec<_>>())?;
        let rank = lhs.rank();
        if std::iter::once(rhs).chain(lhs_scale).chain(rhs_scale).any(|input_type| input_type.rank() != rank) {
            return Err(TypeError::invalid(format!("`{SCALED_DOT_OPERATION_NAME}` inputs must have the same rank")));
        }
        for (index, input_type) in input_types.iter().enumerate() {
            let data_type = input_type.data_type();
            // Complex inputs are rejected because the `bf16` dequantization would silently drop their imaginary
            // parts.
            if !data_type.is_real() && data_type != DataType::Boolean {
                return Err(TypeError::invalid(format!(
                    "`{SCALED_DOT_OPERATION_NAME}` input {index} must have a real numeric or Boolean element type but \
                     got `{data_type}`",
                )));
            }
            if !input_type.unreduced_axes().is_empty() {
                return Err(TypeError::invalid(format!(
                    "`{SCALED_DOT_OPERATION_NAME}` does not support unreduced inputs",
                )));
            }
        }

        // The contraction is a dot of the `bf16` dequantized element arrays, so the dot type inference rule validates
        // the dimension numbers and placements and determines the output type.
        let output_type = infer_dot_output_type(
            SCALED_DOT_OPERATION_NAME,
            &lhs.clone().with_data_type(DataType::BF16),
            &rhs.clone().with_data_type(DataType::BF16),
            &self.dimensions,
            Some(self.preferred_element_type),
            None,
        )?;

        // Each present scale matches its element array along non-contracting dimensions. Along contracting dimensions,
        // static extents must have an integer ratio of at least two. Ratios that involve a dynamic extent are asserted
        // at runtime, except that two identical extents can never have such a ratio.
        for (side, input_type, scale_type, contracting_dimensions) in [
            ("left", lhs, lhs_scale, self.dimensions.lhs_contracting_dimensions()),
            ("right", rhs, rhs_scale, self.dimensions.rhs_contracting_dimensions()),
        ] {
            let Some(scale_type) = scale_type else { continue };
            for axis in 0..rank {
                let input_dimension = input_type.dimension(axis);
                let scale_dimension = scale_type.dimension(axis);
                if !contracting_dimensions.contains(&axis) {
                    if input_dimension != scale_dimension {
                        return Err(TypeError::invalid(format!(
                            "`{SCALED_DOT_OPERATION_NAME}` {side} dimension {axis} has size `{input_dimension}` but \
                             its scale has size `{scale_dimension}`",
                        )));
                    }
                    continue;
                }
                match (input_dimension.value(), scale_dimension.value()) {
                    (_, Some(0)) => {
                        return Err(TypeError::invalid(format!(
                            "`{SCALED_DOT_OPERATION_NAME}` {side} contracting dimension {axis} has an empty scale, \
                             but scale sizes must be positive",
                        )));
                    }
                    (Some(input_size), Some(scale_size)) if input_size % scale_size != 0 => {
                        return Err(TypeError::invalid(format!(
                            "`{SCALED_DOT_OPERATION_NAME}` {side} contracting dimension {axis} has size {input_size}, \
                             which is not divisible by its scale size {scale_size}",
                        )));
                    }
                    (Some(input_size), Some(scale_size)) if input_size / scale_size < 2 => {
                        return Err(TypeError::invalid(format!(
                            "`{SCALED_DOT_OPERATION_NAME}` {side} contracting dimension {axis} must be at least twice \
                             as large as its scale but has ratio {}",
                            input_size / scale_size,
                        )));
                    }
                    _ if input_dimension == scale_dimension => {
                        return Err(TypeError::invalid(format!(
                            "`{SCALED_DOT_OPERATION_NAME}` {side} contracting dimension {axis} must be at least twice \
                             as large as its scale but both have size `{input_dimension}`",
                        )));
                    }
                    _ => {}
                }
            }
        }
        Ok(vec![output_type])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("dimensions", &self.dimensions)?;
            operation.field("preferred_element_type", &self.preferred_element_type)?;
            operation.field("lhs_scale", &self.has_lhs_scale)?;
            operation.field("rhs_scale", &self.has_rhs_scale)
        })
    }
}

impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for ScaledDotOperation
where
    C::Value: ScaledDot,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        let (lhs, rhs, lhs_scale, rhs_scale) = self.split_inputs(inputs)?;
        Ok(vec![lhs.scaled_dot(
            rhs,
            lhs_scale,
            rhs_scale,
            Some(&self.dimensions),
            Some(self.preferred_element_type),
        )?])
    }
}

impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for ScaledDotOperation where
    C::Operation: From<ScaledDotOperation>
{
}

impl<C: Context<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>>
    for ScaledDotOperation
where
    ScaledDotOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        if inputs.len() != self.input_count() {
            return Err(ProgramError::InvalidInputCount { expected: self.input_count(), actual: inputs.len() }.into());
        }
        self.dimensions.validate(
            SCALED_DOT_OPERATION_NAME,
            inputs[0].unbatched_type().rank(),
            inputs[1].unbatched_type().rank(),
        )?;
        if inputs.iter().all(|input| input.batch_axis().is_replicated()) {
            return self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()]);
        }
        validate_common_mapped_dimension(SCALED_DOT_OPERATION_NAME, inputs)?;

        // Every scale array must keep the rank of its element array, so every input moves its mapped axis to the front
        // or, when it is replicated, materializes one there through the active policy, which keeps a dynamic mapped
        // extent a first-class value. The lifted dimension numbers then pair those leading axes as a new batching
        // dimension and shift the existing dimension numbers past them.
        let aligned_inputs = inputs
            .iter()
            .map(|input| P::match_axis(context, input, Axis::from(0)))
            .collect::<Result<Vec<_>, _>>()?;
        let (dimensions, output_axis) = self.dimensions.batched(Some(0), Some(0)).unwrap();
        let lifted_operation =
            Self::new(dimensions, self.preferred_element_type, self.has_lhs_scale, self.has_rhs_scale);
        lifted_operation.interpret_with_batch_axes(
            context,
            aligned_inputs.as_slice(),
            &[BatchAxis::from_optional_position(output_axis)],
        )
    }
}

impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for ScaledDotOperation
where
    C::Operation: From<ScaledDotOperation>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        _context: &DifferentiationContext<C, P>,
        _driver: &D,
        _inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // Refer to the documentation of `ScaledDotOperation` for why this operation has no differentiation rule.
        // Pairing its outputs with zero tangents instead would silently treat the scales as constants.
        Err(ProgramError::UnsupportedOperation {
            message: format!("`{SCALED_DOT_OPERATION_NAME}` does not support differentiation"),
        }
        .into())
    }
}

impl_non_transposable_operation!(ScaledDotOperation);

/// Value-level generalized block-scaled dot product capability. [`ScaledDot`] is the receiver-style entry point for
/// staging or executing [`ScaledDotOperation`]s, which contract element arrays after scaling every block of their
/// contracting dimensions by the corresponding scale value. [`BlockQuantize`] produces element and scale arrays in the
/// layout that this capability consumes.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and array IR values implement it for [`ArrayIrType`].
#[capability(projection(ArrayIrType => ArrayType))]
pub trait ScaledDot<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Computes the generalized dot product `(self * lhs_scale) · (rhs * rhs_scale)`, where each scale is repeated over
    /// its blocks before the multiplication and an absent scale is the multiplicative identity. Refer to
    /// [`ScaledDotOperation`] for the exact semantics and input constraints. Returns a [`ProgramError`] if the inputs
    /// are incompatible with each other or with `dimensions`, or if the operation cannot be recorded in the context of
    /// the value.
    ///
    /// # Parameters
    ///
    ///   - `rhs`: Right element array.
    ///   - `lhs_scale`: Optional scale array of the left element array (i.e., of `self`).
    ///   - `rhs_scale`: Optional scale array of the right element array.
    ///   - `dimensions`: Optional dimension numbers of the generalized dot product. They default to
    ///     [`ScaledDotOperation::default_dimensions`] for the rank of `self`.
    ///   - `preferred_element_type`: Optional data type of the contraction accumulator and the output, which defaults
    ///     to `f32` so that the contraction of the `bf16` dequantized inputs does not lose precision.
    fn scaled_dot(
        &self,
        rhs: &Self,
        lhs_scale: Option<&Self>,
        rhs_scale: Option<&Self>,
        dimensions: Option<&DotDimensionNumbers>,
        preferred_element_type: Option<DataType>,
    ) -> Result<Self, ProgramError>;

    /// Computes the rank-three block-scaled batched matrix product `[B, M, K] × [B, N, K] → [B, M, N]` of `self` and
    /// `rhs`, where both inputs contract their last dimension and pair their first dimension as a batching dimension.
    /// Unlike [`Self::scaled_dot`], both scales are required. Returns a [`ProgramError`] if any input does not have rank
    /// three or if [`Self::scaled_dot`] fails.
    #[inline]
    fn scaled_matmul(
        &self,
        rhs: &Self,
        lhs_scale: &Self,
        rhs_scale: &Self,
        preferred_element_type: Option<DataType>,
    ) -> Result<Self, ProgramError>
    where
        Self: Typed<Type: ArrayMemberType>,
    {
        for value in [self, rhs, lhs_scale, rhs_scale] {
            if value.r#type().as_array_type()?.rank() != 3 {
                return Err(TypeError::invalid("`scaled_matmul` expects rank-3 inputs").into());
            }
        }
        self.scaled_dot(
            rhs,
            Some(lhs_scale),
            Some(rhs_scale),
            Some(&DotDimensionNumbers::new(vec![2], vec![2], vec![0], vec![0])),
            preferred_element_type,
        )
    }
}

impl ScaledDot for Array {
    fn scaled_dot(
        &self,
        rhs: &Self,
        lhs_scale: Option<&Self>,
        rhs_scale: Option<&Self>,
        dimensions: Option<&DotDimensionNumbers>,
        preferred_element_type: Option<DataType>,
    ) -> Result<Self, ProgramError> {
        // Concrete arrays evaluate the canonical composition eagerly. They are lifted into the array IR so that the
        // composition can read their extents as dimension values, and the composition validates them through the type
        // inference rule of `ScaledDotOperation` before dequantizing and contracting them.
        let dimensions = match dimensions {
            Some(dimensions) => dimensions.clone(),
            None => ScaledDotOperation::default_dimensions(self.r#type().rank())?,
        };
        let preferred_element_type = preferred_element_type.unwrap_or(DataType::F32);
        let lhs_scale = lhs_scale.cloned().map(ArrayIrValue::from);
        let rhs_scale = rhs_scale.cloned().map(ArrayIrValue::from);
        scaled_dot_ir_composition(
            &ArrayIrValue::from(self.clone()),
            &ArrayIrValue::from(rhs.clone()),
            lhs_scale.as_ref(),
            rhs_scale.as_ref(),
            &dimensions,
            preferred_element_type,
        )
    }
}

impl<V: Value<Type = ArrayType, Dispatch = ValueDomainDispatch> + ManualVariationAlignment<ArrayType>>
    ScaledDot<ArrayType> for V
where
    V::Domain: Context<Operation: From<ScaledDotOperation>>,
{
    fn scaled_dot(
        &self,
        rhs: &Self,
        lhs_scale: Option<&Self>,
        rhs_scale: Option<&Self>,
        dimensions: Option<&DotDimensionNumbers>,
        preferred_element_type: Option<DataType>,
    ) -> Result<Self, ProgramError> {
        // Values that carry a context stage a scaled dot through that context. The `ValueDomainDispatch` marker keeps
        // this implementation disjoint from eager values, which implement the capability directly.
        let dimensions = match dimensions {
            Some(dimensions) => dimensions.clone(),
            None => ScaledDotOperation::default_dimensions(self.r#type().rank())?,
        };
        let preferred_element_type = preferred_element_type.unwrap_or(DataType::F32);
        let mut inputs = vec![self.clone(), rhs.clone()];
        inputs.extend(lhs_scale.cloned());
        inputs.extend(rhs_scale.cloned());
        let inputs = ManualVariationAlignment::align_manual_variation(&inputs)?;
        let operation =
            ScaledDotOperation::new(dimensions, preferred_element_type, lhs_scale.is_some(), rhs_scale.is_some());
        let mut outputs = self.domain().bind(operation, Vec::new(), inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Evaluates a [`ScaledDotOperation`] over array IR values as its canonical composition, returning the projected array
/// member of the result. This is the authoritative definition of the operation, which the eager [`Array`]
/// implementation of [`ScaledDot`] and the XLA `xla.scaled_dot` composite decomposition both use.
///
/// The composition first validates the input types with the type inference rule of [`ScaledDotOperation`]. It then
/// reads every element and scale extent through [`DimensionSize`], proves each contracting ratio with comparisons and
/// [`Assert`] (i.e., that every scale extent is positive, that it divides the element extent, and that the quotient is
/// at least two), and supplies every broadcast and reshape extent as an ordinary dimension value, so that dynamic
/// extents need no special handling. The array arithmetic is delegated to the projected [`ArrayType`] member, so this
/// function introduces neither another value universe nor backend-specific shape logic.
///
/// # Parameters
///
///   - `lhs`: Left element array.
///   - `rhs`: Right element array.
///   - `lhs_scale`: Optional scale array of `lhs`.
///   - `rhs_scale`: Optional scale array of `rhs`.
///   - `dimensions`: Dimension numbers of the generalized dot product.
///   - `preferred_element_type`: Data type of the contraction accumulator and the output.
pub fn scaled_dot_ir_composition<V>(
    lhs: &V,
    rhs: &V,
    lhs_scale: Option<&V>,
    rhs_scale: Option<&V>,
    dimensions: &DotDimensionNumbers,
    preferred_element_type: DataType,
) -> Result<<V as ValueProjection<ArrayType>>::Projected, ProgramError>
where
    V: Value<Type = ArrayIrType>
        + Assert
        + DimensionSize
        + DynamicBroadcast
        + DynamicReshape
        + ValueProjection<ArrayType>
        + ValueProjection<DimensionType>,
    <V as ValueProjection<ArrayType>>::Projected: Value<Type = ArrayType> + ConvertElementType + Dot + Mul,
    <V as ValueProjection<DimensionType>>::Projected:
        Value<Type = DimensionType> + Compare<V> + DimensionMax + Div + Rem,
    V::Domain: Context<Type = ArrayIrType> + DimensionConstant,
{
    let array_type = |value: &V| -> Result<ArrayType, ProgramError> {
        let r#type = value.r#type();
        Ok(<&ArrayType>::try_from(r#type.as_ref())?.clone())
    };
    let operation =
        ScaledDotOperation::new(dimensions.clone(), preferred_element_type, lhs_scale.is_some(), rhs_scale.is_some());
    let mut input_types = vec![array_type(lhs)?, array_type(rhs)?];
    input_types.extend(lhs_scale.map(array_type).transpose()?);
    input_types.extend(rhs_scale.map(array_type).transpose()?);
    operation.infer_output_types(input_types.as_slice(), &[])?;

    let lhs = dequantize_block_scaled_ir(lhs, lhs_scale, dimensions.lhs_contracting_dimensions())?;
    let rhs = dequantize_block_scaled_ir(rhs, rhs_scale, dimensions.rhs_contracting_dimensions())?;
    Ok(if preferred_element_type == DataType::BF16 {
        lhs.dot(&rhs, dimensions)?
    } else {
        lhs.dot_with_accumulation_type(&rhs, dimensions, preferred_element_type)?
    })
}

/// Dequantizes one element array of [`scaled_dot_ir_composition`] by converting it to `bf16` and, when `scale` is
/// present, multiplying it by that scale after repeating each scale value over its block along every dimension in
/// `contracting_dimensions`.
fn dequantize_block_scaled_ir<V>(
    elements: &V,
    scale: Option<&V>,
    contracting_dimensions: &[usize],
) -> Result<<V as ValueProjection<ArrayType>>::Projected, ProgramError>
where
    V: Value<Type = ArrayIrType>
        + Assert
        + DimensionSize
        + DynamicBroadcast
        + DynamicReshape
        + ValueProjection<ArrayType>
        + ValueProjection<DimensionType>,
    <V as ValueProjection<ArrayType>>::Projected: Value<Type = ArrayType> + ConvertElementType + Mul,
    <V as ValueProjection<DimensionType>>::Projected:
        Value<Type = DimensionType> + Compare<V> + DimensionMax + Div + Rem,
    V::Domain: Context<Type = ArrayIrType> + DimensionConstant,
{
    let Some(scale) = scale else {
        return ValueProjection::<ArrayType>::into_projected(elements.clone())?.convert_element_type(DataType::BF16);
    };
    let context = elements.domain();
    let element_type = elements.r#type();
    let element_type = <&ArrayType>::try_from(element_type.as_ref())?;
    let element_dimensions =
        (0..element_type.rank()).map(|axis| elements.dimension_size(axis)).collect::<Result<Vec<_>, _>>()?;
    let elements =
        ValueProjection::<ArrayType>::into_projected(elements.clone())?.convert_element_type(DataType::BF16)?;
    let scale_type = scale.r#type();
    let scale_type = <&ArrayType>::try_from(scale_type.as_ref())?;
    let scale_dimensions = (0..scale_type.rank())
        .map(|scale_axis| scale.dimension_size(scale_axis))
        .collect::<Result<Vec<_>, _>>()?;

    // The scale is expanded by inserting a block dimension after every contracting dimension, broadcasting into that
    // shape, and reshaping the result to the shape of the elements, which repeats each scale value over its block.
    let mut expanded_dimensions = Vec::with_capacity(scale_type.rank() + contracting_dimensions.len());
    let mut output_axes = Vec::with_capacity(scale_type.rank());
    for (axis, scale_dimension) in scale_dimensions.iter().enumerate() {
        output_axes.push(expanded_dimensions.len());
        expanded_dimensions.push(scale_dimension.clone());
        if contracting_dimensions.contains(&axis) {
            // The remainder and the quotient divide by `max(scale, 1)` rather than by the scale extent itself, so that
            // they stay defined for an empty scale and the positivity assertion reports that violation instead.
            let element_extent = ValueProjection::<DimensionType>::into_projected(element_dimensions[axis].clone())?;
            let scale_extent = ValueProjection::<DimensionType>::into_projected(scale_dimension.clone())?;
            let zero = ValueProjection::<DimensionType>::into_projected(context.dimension_constant(0)?)?;
            let one = ValueProjection::<DimensionType>::into_projected(context.dimension_constant(1)?)?;
            scale_extent.compare(&zero, ComparisonDirection::GreaterThan)?.assert(
                "scale extent must be positive",
                &[("scale", ValueProjection::<DimensionType>::from_projected(scale_extent.clone()))],
            )?;
            let divisor = scale_extent.dimension_max(&one)?;
            element_extent.rem(&divisor)?.compare(&zero, ComparisonDirection::Equal)?.assert(
                "element extent must be divisible by scale extent",
                &[
                    ("elements", ValueProjection::<DimensionType>::from_projected(element_extent.clone())),
                    ("scale", ValueProjection::<DimensionType>::from_projected(scale_extent)),
                ],
            )?;
            let ratio = element_extent.div(&divisor)?;
            let two = ValueProjection::<DimensionType>::into_projected(context.dimension_constant(2)?)?;
            two.compare(&ratio, ComparisonDirection::LessThanOrEqual)?.assert(
                "contracting ratio must be at least two",
                &[("ratio", ValueProjection::<DimensionType>::from_projected(ratio.clone()))],
            )?;
            expanded_dimensions.push(ValueProjection::<DimensionType>::from_projected(ratio));
        }
    }
    let expanded_scale = scale
        .dynamic_broadcast(expanded_dimensions.as_slice(), output_axes.as_slice())?
        .dynamic_reshape(element_dimensions.as_slice())?;
    let expanded_scale =
        ValueProjection::<ArrayType>::into_projected(expanded_scale)?.convert_element_type(DataType::BF16)?;
    elements.mul(&expanded_scale)
}

/// Value-level block quantization capability. [`BlockQuantize::block_quantize`] splits an `f32` or `f64` array into an
/// array of narrow elements and an array of scales with one scale per block of consecutive values along the last
/// dimension, which are the inputs that [`ScaledDot`] consumes. This supports quantizing values on the fly (e.g., a KV
/// cache) without a dedicated primitive, because the recipe composes existing operations and inherits their transform
/// rules. Both sides of a [`ScaledDot`] contraction quantized this way must contract their last dimension, which
/// [`ScaledDot::scaled_matmul`] does, while the default dimension numbers of [`ScaledDot::scaled_dot`] contract the
/// penultimate dimension of the right input.
///
/// The scale data type selects one of two standard microscaling recipes.
///
///   - **`f8e4m3fn` scales** (the NVFP4 recipe of NVIDIA) set the scale of each block to
///     `max_abs(block) / element_max`, where `element_max` is the largest finite magnitude of the element type (e.g.,
///     `6.0` for `f4e2m1fn`), so that the largest element of each block quantizes to the top of the element grid.
///     Scales are clamped to `[2^-6, 448]`, which is the range of positive normal `f8e4m3fn` values.
///   - **`f8e8m0fnu` scales** (the [OCP MX] recipe, e.g., MXFP8) set the shared scale of each block to the power of
///     two `2^(floor(log2(max_abs(block))) - emax)`, where `emax` is the largest exponent of the element type (e.g.,
///     `8` for `f8e4m3fn` and `15` for `f8e5m2`), so that the largest element of each block lands in the top binade of
///     the element type. Scales are clamped to `[2^-126, 2^127]`. The upper bound is the largest `f8e8m0fnu` value,
///     and the lower bound is the smallest normal `f32` value rather than the smallest `f8e8m0fnu` value `2^-127`, so
///     that a backend that flushes subnormal values to zero cannot produce a zero scale, which `f8e8m0fnu` cannot
///     represent.
///
/// Both recipes divide each input value by the scale of its block after narrowing that scale to the scale type, which
/// is exactly the scale that [`ScaledDotOperation`] dequantizes with. They then clamp the quotients to the finite range
/// of the element type before converting them, so that dequantization reproduces the input up to the quantization
/// error of the element type and no recipe relies on the overflow behavior of a floating-point conversion. The OCP MX
/// specification prescribes that clamp for the quotients between the largest finite magnitude and `2^(emax + 1)`, and
/// in the NVFP4 recipe it absorbs the rounding of the scales. Blocks whose values are all zero (or tiny) and blocks
/// whose values are huge use the smallest and largest scale, respectively, instead of a zero or non-finite scale.
///
/// The `log2` of the MX recipe is computed as `log(x) / log(2)`, and `1e-4` is added to it before the floor so that
/// block maxima that are exact powers of two stay in their own binade despite rounding in the quotient. Converting
/// `exp(exponent · log(2))` to `f8e8m0fnu` rounds to the nearest power of two, which absorbs the remaining
/// approximation error. The `quantize` helper of JAX instead rounds `max_abs(block) / element_max` up to a power of
/// two, and it also supports a global scale for NVFP4, which this capability does not.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and array IR values implement it for [`ArrayIrType`].
///
/// [OCP MX]: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
#[capability(projection(ArrayIrType => ArrayType))]
pub trait BlockQuantize<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Quantizes `self` into a pair `(elements, scales)`, with one scale per block of `block_size` consecutive values
    /// along the last dimension. `elements` has the shape of `self` and element type `element_type`, while `scales` has
    /// the shape of `self` with its last extent divided by `block_size` and element type `scale_type`. Refer to the
    /// documentation of [`BlockQuantize`] for the exact recipes.
    ///
    /// `self` must be an `f32` or `f64` array with a static shape of rank at least one, `block_size` must be at least
    /// two, and the last extent of `self` must be a positive multiple of `block_size`, because [`ScaledDot`] requires
    /// non-empty scales whose blocks have at least two values. Returns a [`ProgramError`] if these requirements are not
    /// met, if `element_type` is not one of `f4e2m1fn`, `f8e4m3fn`, and `f8e5m2`, if `scale_type` is not one of
    /// `f8e4m3fn` and `f8e8m0fnu`, or if a composed operation fails.
    ///
    /// # Parameters
    ///
    ///   - `block_size`: Number of consecutive values along the last dimension that share one scale.
    ///   - `element_type`: Data type of the quantized elements.
    ///   - `scale_type`: Data type of the scales, which selects the quantization recipe.
    fn block_quantize(
        &self,
        block_size: usize,
        element_type: DataType,
        scale_type: DataType,
    ) -> Result<(Self, Self), ProgramError>;
}

impl<V> BlockQuantize<ArrayType> for V
where
    V: Value<Type = ArrayType>
        + Abs
        + Broadcast
        + Clamp
        + ConvertElementType
        + Div
        + Exp
        + Floor
        + Log
        + Mul
        + Reduce
        + Reshape
        + Sub,
    V::Domain: Fill<f64, V>,
{
    fn block_quantize(
        &self,
        block_size: usize,
        element_type: DataType,
        scale_type: DataType,
    ) -> Result<(Self, Self), ProgramError> {
        // Every value with the capabilities used by the recipe, which include both the concrete reference `Array`
        // backend and the transform tracers, quantizes through this composition. All validation happens before the
        // first operation, so that rejected requests stage nothing.
        let (element_max, element_max_exponent) = match element_type {
            DataType::F4E2M1FN => (6.0, 2.0),
            DataType::F8E4M3FN => (448.0, 8.0),
            DataType::F8E5M2 => (57344.0, 15.0),
            element_type => {
                return Err(TypeError::invalid(format!(
                    "`block_quantize` does not support element data type `{element_type}`",
                ))
                .into());
            }
        };
        let (smallest_scale, largest_scale) = match scale_type {
            DataType::F8E4M3FN => ((-6.0f64).exp2(), 448.0),
            DataType::F8E8M0FNU => ((-126.0f64).exp2(), 127.0f64.exp2()),
            scale_type => {
                return Err(TypeError::invalid(format!(
                    "`block_quantize` does not support scale data type `{scale_type}`",
                ))
                .into());
            }
        };
        let input_type = self.r#type().into_owned();
        let compute_type = input_type.data_type();
        if !matches!(compute_type, DataType::F32 | DataType::F64) {
            return Err(TypeError::invalid(format!(
                "`block_quantize` expects an `f32` or `f64` input but got `{compute_type}`",
            ))
            .into());
        }
        let Some(shape) = input_type.static_shape() else {
            return Err(TypeError::invalid("`block_quantize` input must have a static shape").into());
        };
        let dimensions = shape.dimensions().to_vec();
        let Some(&trailing_size) = dimensions.last() else {
            return Err(TypeError::invalid("`block_quantize` does not support rank-0 inputs").into());
        };
        if block_size < 2 {
            return Err(TypeError::invalid(format!(
                "`block_quantize` block size must be at least 2 but got {block_size}"
            ))
            .into());
        }
        if trailing_size == 0 || trailing_size % block_size != 0 {
            return Err(TypeError::invalid(format!(
                "`block_quantize` trailing dimension size {trailing_size} is not a positive multiple of block size \
                 {block_size}",
            ))
            .into());
        }
        let mut scale_dimensions = dimensions;
        *scale_dimensions.last_mut().unwrap() = trailing_size / block_size;
        let block_shape = Shape::new(
            scale_dimensions
                .iter()
                .map(|&size| Dimension::Static(size))
                .chain(std::iter::once(Dimension::Static(block_size)))
                .collect(),
        );
        let scale_value_type = ArrayType::new(
            compute_type,
            Shape::new(scale_dimensions.iter().map(|&size| Dimension::Static(size)).collect()),
        );
        let domain = self.domain();
        let fill = |value: f64| domain.fill(&scale_value_type, value);
        let fill_scalar = |value: f64| domain.fill(&ArrayType::scalar(compute_type), value);

        // The maximum magnitude of each block along the trailing dimension determines its scale.
        let block_max = self.reshape(block_shape)?.abs()?.reduce(&[scale_dimensions.len()], ReductionKind::Max)?;
        let scales = if scale_type == DataType::F8E4M3FN {
            // NVFP4 linear scaling maps the block maximum to the largest finite magnitude of the element type.
            block_max.div(&fill(element_max)?)?
        } else {
            // OCP MX power of two scaling computes `2^(floor(log2(max_abs)) - emax)` with the boundary nudge that the
            // trait documentation describes, folded into one subtraction because `floor(x + ε) - emax` equals
            // `floor(x + ε - emax)` for the integer `emax`. The lower clamp below also absorbs the `exp(-inf) = 0`
            // that blocks whose values are all zero produce.
            let log_2 = fill(std::f64::consts::LN_2)?;
            let exponent = block_max.log()?.div(&log_2)?.sub(&fill(element_max_exponent - 1e-4)?)?.floor()?;
            exponent.mul(&log_2)?.exp()?
        };
        let scales = scales
            .clamp(&fill_scalar(smallest_scale)?, &fill_scalar(largest_scale)?)?
            .convert_element_type(scale_type)?;

        // The elements are divided by the stored (i.e., already narrowed) scales, which are exactly the values that
        // `ScaledDot` dequantizes with. The scales derive from the input, so the expanded scales keep its memory space,
        // placement, and manual variation and only add the replicated block dimension, because broadcasting cannot
        // change manual variation.
        let stored_scales = scales.convert_element_type(compute_type)?;
        let expanded_type = stored_scales
            .r#type()
            .with_inserted_dimension(scale_dimensions.len(), Dimension::Static(block_size))?;
        let scale_axes = (0..scale_dimensions.len()).collect::<Vec<_>>();
        let expanded_scales =
            stored_scales.broadcast(expanded_type, scale_axes.as_slice())?.reshape(input_type.shape().clone())?;
        let elements = self
            .div(&expanded_scales)?
            .clamp(&fill_scalar(-element_max)?, &fill_scalar(element_max)?)?
            .convert_element_type(element_type)?;
        Ok((elements, scales))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrOperation, ArrayOperation, DimensionBounds, DimensionVariable, LogicalMesh, MeshAxis, MeshAxisType,
        Sharding, ShardingDimension,
    };
    use crate::axes::NamedAxis;
    use crate::contexts::{EagerContext, StagingContext};
    use crate::macros::{check_operation_batching, check_operation_partial_evaluation, check_operation_type_inference};
    use crate::operations::assertions::AssertionError;
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::EmptyRegionDriver;
    use crate::tracing::{Tracer, TracingContext};

    use super::*;

    #[test]
    fn test_scaled_dot() {
        let dimensions = DotDimensionNumbers::matmul();
        let operation = ScaledDotOperation::new(dimensions.clone(), DataType::F32, true, false);

        // Operation identity, accessors, and formatting.
        assert_eq!(operation.name(), SCALED_DOT_OPERATION_NAME);
        assert_eq!(operation.dimensions(), &dimensions);
        assert_eq!(operation.preferred_element_type(), DataType::F32);
        assert!(operation.has_lhs_scale());
        assert!(!operation.has_rhs_scale());
        assert_eq!(
            operation.to_string(),
            indoc! {"
                scaled_dot [
                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    preferred_element_type=f32,
                    lhs_scale=true,
                    rhs_scale=false,
                ]
            "}
            .trim_end(),
        );

        // Staging records the default dimension numbers, the default `f32` preferred element type, and the scales.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(lhs, rhs, lhs_scale, rhs_scale): (Tracer<_>, Tracer<_>, Tracer<_>, Tracer<_>)| {
                lhs.scaled_dot(&rhs, Some(&lhs_scale), Some(&rhs_scale), None, None)
            },
            (
                ArrayType::new_static(DataType::F8E4M3FN, [2, 4]),
                ArrayType::new_static(DataType::F8E5M2, [4, 3]),
                ArrayType::new_static(DataType::F8E8M0FNU, [2, 2]),
                ArrayType::new_static(DataType::F8E8M0FNU, [2, 3]),
            ),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f8e4m3fn[2, 4], %1:f8e5m2[4, 3], %2:f8e8m0fnu[2, 2], %3:f8e8m0fnu[2, 3] .
                let %4:f32[2, 3] = scaled_dot [
                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    preferred_element_type=f32,
                    lhs_scale=true,
                    rhs_scale=true,
                ] %0 %1 %2 %3
                in (%4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_scaled_dot_operation_default_dimensions() {
        assert_eq!(ScaledDotOperation::default_dimensions(2), Ok(DotDimensionNumbers::matmul()));
        assert_eq!(
            ScaledDotOperation::default_dimensions(4),
            Ok(DotDimensionNumbers::new(vec![3], vec![2], vec![0, 1], vec![0, 1])),
        );
        assert_eq!(
            ScaledDotOperation::default_dimensions(1),
            Err(TypeError::invalid(format!(
                "default `{SCALED_DOT_OPERATION_NAME}` dimension numbers require inputs of rank at least 2 but got \
                 rank 1",
            ))),
        );
    }

    #[test]
    fn test_scaled_dot_type_inference() {
        let dimensions = DotDimensionNumbers::new(vec![2, 3], vec![1, 2], vec![0], vec![0]);
        let lhs = ArrayType::new_static(DataType::F8E4M3FN, [2, 3, 4, 6]);
        let rhs = ArrayType::new_static(DataType::F8E5M2, [2, 4, 6, 5]);
        let lhs_scale = ArrayType::new_static(DataType::F8E8M0FNU, [2, 3, 2, 2]);
        let rhs_scale = ArrayType::new_static(DataType::F8E8M0FNU, [2, 2, 2, 5]);
        check_operation_type_inference!(
            operation = ScaledDotOperation::new(dimensions.clone(), DataType::BF16, true, true),
            cases = [
                {
                    input_types = [lhs.clone(), rhs.clone(), lhs_scale.clone(), rhs_scale.clone()],
                    output_types = [ArrayType::new_static(DataType::BF16, [2, 3, 5])],
                },
                {
                    input_types = [lhs.clone(), rhs.clone(), lhs_scale.clone()],
                    error = format!("`{SCALED_DOT_OPERATION_NAME}` expects 4 inputs but got 3"),
                },
                {
                    input_types = [
                        lhs.clone(),
                        rhs.clone(),
                        ArrayType::new_static(DataType::F8E8M0FNU, [2, 3, 2]),
                        rhs_scale.clone(),
                    ],
                    error = format!("`{SCALED_DOT_OPERATION_NAME}` inputs must have the same rank"),
                },
                {
                    input_types = [
                        lhs.clone(),
                        rhs.clone(),
                        lhs_scale.clone(),
                        ArrayType::new_static(DataType::Token, [2, 2, 2, 5]),
                    ],
                    error = format!(
                        "`{SCALED_DOT_OPERATION_NAME}` input 3 must have a real numeric or Boolean element type but \
                         got `token`",
                    ),
                },
                {
                    input_types = [
                        ArrayType::new_static(DataType::C64, [2, 3, 4, 6]),
                        rhs.clone(),
                        lhs_scale.clone(),
                        rhs_scale.clone(),
                    ],
                    error = format!(
                        "`{SCALED_DOT_OPERATION_NAME}` input 0 must have a real numeric or Boolean element type but \
                         got `c64`",
                    ),
                },
                {
                    input_types = [
                        ArrayType::new_static(DataType::Boolean, [2, 3, 4, 6]),
                        rhs.clone(),
                        lhs_scale.clone(),
                        rhs_scale.clone(),
                    ],
                    output_types = [ArrayType::new_static(DataType::BF16, [2, 3, 5])],
                },
                {
                    input_types = [
                        lhs.clone(),
                        ArrayType::new_static(DataType::F8E5M2, [2, 5, 6, 5]),
                        lhs_scale.clone(),
                        rhs_scale.clone(),
                    ],
                    error = format!(
                        "`{SCALED_DOT_OPERATION_NAME}` contracting dimension sizes do not match (left-hand side axis \
                         2, right-hand side axis 1)",
                    ),
                },
                {
                    input_types = [
                        lhs.clone(),
                        rhs.clone(),
                        ArrayType::new_static(DataType::F8E8M0FNU, [2, 1, 2, 2]),
                        rhs_scale.clone(),
                    ],
                    error = format!(
                        "`{SCALED_DOT_OPERATION_NAME}` left dimension 1 has size `3` but its scale has size `1`",
                    ),
                },
                {
                    input_types = [
                        lhs.clone(),
                        rhs.clone(),
                        ArrayType::new_static(DataType::F8E8M0FNU, [2, 3, 3, 2]),
                        rhs_scale.clone(),
                    ],
                    error = format!(
                        "`{SCALED_DOT_OPERATION_NAME}` left contracting dimension 2 has size 4, which is not \
                         divisible by its scale size 3",
                    ),
                },
                {
                    input_types = [
                        lhs.clone(),
                        rhs.clone(),
                        ArrayType::new_static(DataType::F8E8M0FNU, [2, 3, 0, 2]),
                        rhs_scale.clone(),
                    ],
                    error = format!(
                        "`{SCALED_DOT_OPERATION_NAME}` left contracting dimension 2 has an empty scale, but scale \
                         sizes must be positive",
                    ),
                },
                {
                    input_types = [
                        lhs.clone(),
                        rhs.clone(),
                        lhs_scale.clone(),
                        ArrayType::new_static(DataType::F8E8M0FNU, [2, 4, 2, 5]),
                    ],
                    error = format!(
                        "`{SCALED_DOT_OPERATION_NAME}` right contracting dimension 1 must be at least twice as large \
                         as its scale but has ratio 1",
                    ),
                },
            ],
        );

        // Without scales, the operation is a dot of the `bf16` element arrays at the preferred element type.
        check_operation_type_inference!(
            operation = ScaledDotOperation::new(dimensions.clone(), DataType::F64, false, false),
            cases = [{
                input_types = [lhs.clone(), rhs.clone()],
                output_types = [ArrayType::new_static(DataType::F64, [2, 3, 5])],
            }],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = ScaledDotOperation::new(dimensions, DataType::BF16, true, true),
            input_types = [lhs, rhs, lhs_scale, rhs_scale],
        );

        // Ratios that involve a dynamic extent are asserted at runtime, but identical extents are rejected statically.
        let block = Dimension::Dynamic(DimensionVariable::new("block", DimensionBounds::new(1, Some(8)).unwrap()));
        let lhs = ArrayType::new(DataType::F32, Shape::new(vec![2.into(), block.clone()]));
        let rhs = ArrayType::new(DataType::F32, Shape::new(vec![block.clone(), 3.into()]));
        check_operation_type_inference!(
            operation = ScaledDotOperation::new(DotDimensionNumbers::matmul(), DataType::F32, true, false),
            cases = [
                {
                    input_types = [lhs.clone(), rhs.clone(), ArrayType::new_static(DataType::F32, [2, 1])],
                    output_types = [ArrayType::new_static(DataType::F32, [2, 3])],
                },
                {
                    input_types = [lhs.clone(), rhs, lhs],
                    error = format!(
                        "`{SCALED_DOT_OPERATION_NAME}` left contracting dimension 1 must be at least twice as large \
                         as its scale but both have size `block`",
                    ),
                },
            ],
        );
    }

    #[test]
    fn test_scaled_dot_type_inference_manual_variation() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 2).with_varying_manual_axes(["x"]).unwrap();
        let lhs = ArrayType::new_static(DataType::F32, [2, 4]);
        let rhs = ArrayType::new_static(DataType::F32, [4, 2]);
        let scale = ArrayType::new_static(DataType::F32, [2, 2]).with_sharding(sharding.clone()).unwrap();
        check_operation_type_inference!(
            operation = ScaledDotOperation::new(DotDimensionNumbers::matmul(), DataType::F32, true, false),
            cases = [
                {
                    input_types = [lhs.clone(), rhs.clone(), scale.clone()],
                    error = format!(
                        "`{SCALED_DOT_OPERATION_NAME}` inputs must have matching varying manual axes; insert \
                         `parallel_vary` on the inputs that lack an axis, as `align_manual_variation` does",
                    ),
                },
                {
                    input_types = [
                        lhs.with_sharding(sharding.clone()).unwrap(),
                        rhs.with_sharding(sharding.clone()).unwrap(),
                        scale,
                    ],
                    output_types = [ArrayType::new_static(DataType::F32, [2, 2]).with_sharding(sharding).unwrap()],
                },
            ],
        );
    }

    #[test]
    fn test_scaled_dot_interpretation() {
        // Rank-two inputs use the default dimension numbers, which contract the last left dimension with the first
        // right dimension. The two sides infer independent block sizes of two along different scale dimensions.
        let lhs = Array::matrix(2, 4, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let rhs = Array::matrix(4, 3, (1..=12).map(|value| value as f32).collect()).unwrap();
        let lhs_scale = Array::matrix(2, 2, vec![1.0f32, 2.0, 0.5, 1.0]).unwrap();
        let rhs_scale = Array::matrix(2, 3, vec![1.0f32, 1.0, 1.0, 2.0, 2.0, 2.0]).unwrap();
        assert_eq!(
            lhs.scaled_dot(&rhs, Some(&lhs_scale), Some(&rhs_scale), None, Some(DataType::F32)),
            Ok(Array::matrix(2, 3, vec![253.0f32, 284.0, 315.0, 272.5, 308.0, 343.5]).unwrap()),
        );

        // Each scale is independently optional, and an absent scale is the multiplicative identity.
        assert_eq!(
            lhs.scaled_dot(&rhs, Some(&lhs_scale), None, None, Some(DataType::F32)),
            Ok(Array::matrix(2, 3, vec![131.0f32, 148.0, 165.0, 143.5, 164.0, 184.5]).unwrap()),
        );
        assert_eq!(
            lhs.scaled_dot(&rhs, None, None, None, Some(DataType::F32)),
            Ok(Array::matrix(2, 3, vec![70.0f32, 80.0, 90.0, 158.0, 184.0, 210.0]).unwrap()),
        );

        // Without a preferred element type, the contraction accumulates at and returns `f32`, while an explicit `bf16`
        // preferred element type keeps the contraction at the dequantized input type.
        assert_eq!(
            lhs.scaled_dot(&rhs, Some(&lhs_scale), Some(&rhs_scale), None, None),
            Ok(Array::matrix(2, 3, vec![253.0f32, 284.0, 315.0, 272.5, 308.0, 343.5]).unwrap()),
        );
        let product = lhs.scaled_dot(&rhs, None, None, None, Some(DataType::BF16)).unwrap();
        assert_eq!(product.r#type().into_owned(), ArrayType::new_static(DataType::BF16, [2, 3]));
        assert_eq!(product.to_f64s(), vec![70.0, 80.0, 90.0, 158.0, 184.0, 210.0]);

        // The type inference rule rejects invalid inputs before anything is evaluated.
        assert_eq!(
            lhs.scaled_dot(&rhs, Some(&rhs_scale), None, None, None),
            Err(ProgramError::from(TypeError::invalid(format!(
                "`{SCALED_DOT_OPERATION_NAME}` left contracting dimension 1 has size 4, which is not divisible by its \
                 scale size 3",
            )))),
        );

        // Both contracting dimensions of each side carry independent block sizes (two and three on the left), and every
        // scale value is repeated over its two-dimensional block.
        let lhs = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 2, 4, 6]), &[1.0; 48]).unwrap();
        let rhs = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 4, 6, 3]), &[1.0; 72]).unwrap();
        let lhs_scale = Array::from_elements::<f32>(
            ArrayType::new_static(DataType::F32, [1, 2, 2, 2]),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let rhs_scale =
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 2, 2, 3]), &[1.0; 12]).unwrap();
        let dimensions = DotDimensionNumbers::new(vec![2, 3], vec![1, 2], vec![0], vec![0]);
        assert_eq!(
            lhs.scaled_dot(&rhs, Some(&lhs_scale), Some(&rhs_scale), Some(&dimensions), Some(DataType::F32)),
            Ok(Array::from_elements::<f32>(
                ArrayType::new_static(DataType::F32, [1, 2, 3]),
                &[60.0, 60.0, 60.0, 156.0, 156.0, 156.0],
            )
            .unwrap()),
        );
    }

    #[test]
    fn test_scaled_dot_interpretation_array_ir() {
        // Concrete array IR values contract their array members.
        let lhs = Array::matrix(2, 4, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let rhs = Array::matrix(4, 2, vec![1.0f32; 8]).unwrap();
        let lhs_scale = Array::matrix(2, 2, vec![1.0f32, 2.0, 0.5, 1.0]).unwrap();
        let expected = lhs.scaled_dot(&rhs, Some(&lhs_scale), None, None, Some(DataType::F32)).unwrap();
        assert_eq!(
            ArrayIrValue::Array(lhs).scaled_dot(
                &ArrayIrValue::Array(rhs),
                Some(&ArrayIrValue::Array(lhs_scale)),
                None,
                None,
                Some(DataType::F32),
            ),
            Ok(ArrayIrValue::Array(expected)),
        );
    }

    #[test]
    fn test_scaled_dot_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = ScaledDotOperation::new(DotDimensionNumbers::inner_product(), DataType::F32, true, false),
            inputs = [
                Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap(),
                Array::vector(vec![1.0f32; 4]).unwrap(),
                Array::vector(vec![2.0f32, 0.5]).unwrap(),
            ],
            expected = Array::scalar(9.5f32).unwrap(),
        );
    }

    #[test]
    fn test_scaled_dot_batching() {
        check_operation_batching!(
            @exact,
            operation = ScaledDotOperation::new(DotDimensionNumbers::inner_product(), DataType::F32, true, true),
            axis_size = 2,
            cases = [
                {
                    // Mapped scales keep one set of block scales per batch item, and replicated inputs are shared.
                    inputs = [
                        (
                            @mapped(axis = 0),
                            Array::matrix(2, 4, vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0]).unwrap(),
                        ),
                        (@replicated, Array::vector(vec![1.0f32; 4]).unwrap()),
                        (@mapped(axis = 1), Array::matrix(2, 2, vec![1.0f32, 2.0, 2.0, 1.0]).unwrap()),
                        (@replicated, Array::vector(vec![1.0f32, 1.0]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![6.0f32, 13.0]).unwrap())],
                },
                {
                    inputs = [
                        (@replicated, Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap()),
                        (@replicated, Array::vector(vec![1.0f32; 4]).unwrap()),
                        (@replicated, Array::vector(vec![2.0f32, 0.5]).unwrap()),
                        (@replicated, Array::vector(vec![1.0f32, 1.0]).unwrap()),
                    ],
                    outputs = [(@replicated, Array::scalar(9.5f32).unwrap())],
                },
            ],
        );

        // Existing batching dimensions shift past the new leading batching dimension.
        check_operation_batching!(
            @exact,
            operation = ScaledDotOperation::new(
                DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]),
                DataType::F32,
                true,
                false,
            ),
            axis_size = 2,
            cases = [{
                inputs = [
                    (
                        @mapped(axis = 0),
                        Array::from_elements::<f32>(
                            ArrayType::new_static(DataType::F32, [2, 1, 1, 4]),
                            &[1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0],
                        )
                        .unwrap(),
                    ),
                    (
                        @replicated,
                        Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 4, 1]), &[1.0; 4])
                            .unwrap(),
                    ),
                    (
                        @mapped(axis = 0),
                        Array::from_elements::<f32>(
                            ArrayType::new_static(DataType::F32, [2, 1, 1, 2]),
                            &[1.0, 2.0, 2.0, 1.0],
                        )
                        .unwrap(),
                    ),
                ],
                outputs = [(
                    @mapped(axis = 0),
                    Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [2, 1, 1, 1]), &[6.0, 13.0])
                        .unwrap(),
                )],
            }],
        );
    }

    #[test]
    fn test_scaled_dot_batching_validates_original_dimensions() {
        let input = ArrayBatch::new(Array::matrix(2, 1, vec![1.0f32, 2.0]).unwrap(), BatchAxis::new(0)).unwrap();
        let operation = ScaledDotOperation::new(
            DotDimensionNumbers::new(vec![usize::MAX], vec![0], Vec::new(), Vec::new()),
            DataType::F32,
            false,
            false,
        );
        assert_eq!(
            operation.batch(
                &BatchingContext::new(EagerContext::<Array>::new(), 2),
                &EmptyRegionDriver,
                &[input.clone(), input],
            ),
            Err(BatchingError::Type(TypeError::invalid(format!(
                "`{SCALED_DOT_OPERATION_NAME}` left-hand side dimension index out of bounds",
            )))),
        );
    }

    #[test]
    fn test_scaled_dot_batching_validates_mapped_extents() {
        let lhs = ArrayBatch::new(Array::matrix(2, 4, vec![1.0f32; 8]).unwrap(), BatchAxis::new(0)).unwrap();
        let rhs = ArrayBatch::new(Array::matrix(3, 4, vec![1.0f32; 12]).unwrap(), BatchAxis::new(0)).unwrap();
        let operation = ScaledDotOperation::new(DotDimensionNumbers::inner_product(), DataType::F32, false, false);
        assert_eq!(
            operation.batch(&BatchingContext::new(EagerContext::<Array>::new(), 2), &EmptyRegionDriver, &[lhs, rhs]),
            Err(BatchingError::MismatchedBatchSizes { expected: 2, actual: 3 }),
        );
    }

    #[test]
    fn test_scaled_dot_differentiation() {
        // Scaled dot has no differentiation rule, so it rejects live tangents instead of treating its scales as
        // constants.
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let dual = DifferentiationDual::new(input.clone(), input).unwrap();
        assert!(matches!(
            ScaledDotOperation::new(DotDimensionNumbers::inner_product(), DataType::F32, false, false).jvp(
                &DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new()),
                &EmptyRegionDriver,
                &[dual.clone(), dual],
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == format!("`{SCALED_DOT_OPERATION_NAME}` does not support differentiation"),
        ));
    }

    #[test]
    fn test_scaled_dot_scaled_matmul() {
        // Both inputs contract their last dimension, and the output defaults to `f32`.
        let lhs = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 1, 4]), &[1.0, 2.0, 3.0, 4.0])
            .unwrap();
        let rhs = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 2, 4]), &[1.0; 8]).unwrap();
        let lhs_scale =
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 1, 2]), &[1.0, 2.0]).unwrap();
        let rhs_scale =
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 2, 2]), &[1.0, 1.0, 2.0, 2.0])
                .unwrap();
        assert_eq!(
            lhs.scaled_matmul(&rhs, &lhs_scale, &rhs_scale, None),
            Ok(Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 1, 2]), &[17.0, 34.0]).unwrap()),
        );

        // Each side infers its own block size, so the right scale may use larger blocks than the left one.
        let rhs_scale =
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 2, 1]), &[3.0, 1.0]).unwrap();
        assert_eq!(
            lhs.scaled_matmul(&rhs, &lhs_scale, &rhs_scale, Some(DataType::F64)),
            Ok(Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [1, 1, 2]), &[51.0, 17.0]).unwrap()),
        );

        // Inputs of any other rank are rejected.
        let matrix = Array::matrix(2, 2, vec![1.0f32; 4]).unwrap();
        assert_eq!(
            matrix.scaled_matmul(&matrix, &matrix, &matrix, None),
            Err(ProgramError::from(TypeError::invalid("`scaled_matmul` expects rank-3 inputs"))),
        );
    }

    #[test]
    fn test_scaled_dot_ir_composition() {
        // A dynamic scale extent is validated at runtime by the staged assertions.
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let lhs = context.input(ArrayType::new_static(DataType::F32, [8]).into());
        let rhs = context.input(ArrayType::new_static(DataType::F32, [8]).into());
        let scale_extent = DimensionVariable::new("scale", DimensionBounds::new(0, Some(5)).unwrap());
        let lhs_scale =
            context.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(scale_extent)])).into());
        let dimensions = DotDimensionNumbers::inner_product();
        let output = scaled_dot_ir_composition(&lhs, &rhs, Some(&lhs_scale), None, &dimensions, DataType::F32).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.value().atom_id().unwrap()],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {r#"
                lambda %0:f32[8], %1:f32[8], %2:f32[scale] .
                let %3:dimension<8> = constant [value=8]
                    %4:bf16[8] = convert_element_type [data_type=bf16] %0
                    %5:dimension<scale ∈ [0, 5)> = dimension_size [axis=0] %2
                    %6:dimension<0> = constant [value=0]
                    %7:dimension<1> = constant [value=1]
                    %8:bool[] = compare [direction=GreaterThan] %5 %6
                    () = assert [message="scale extent must be positive", labels=["scale"]] %8 %5
                    %9:dimension<max(scale, 1) ∈ [1, 5)> = dimension_max %5 %7
                    %10:dimension<8 % max(scale, 1) ∈ [0, 4)> = dimension_rem %3 %9
                    %11:bool[] = compare [direction=Equal] %10 %6
                    () = assert [
                        message="element extent must be divisible by scale extent",
                        labels=["elements", "scale"],
                    ] %11 %3 %5
                    %12:dimension<8 / max(scale, 1) ∈ [2, 9)> = dimension_div %3 %9
                    %13:dimension<2> = constant [value=2]
                    %14:bool[] = const true
                    %15:f32[scale, 8 / max(scale, 1)] = broadcast [output_axes=[0]] %2 %5 %12
                    %16:f32[8] = reshape %15 %3
                    %17:bf16[8] = convert_element_type [data_type=bf16] %16
                    %18:bf16[8] = mul %4 %17
                    %19:bf16[8] = convert_element_type [data_type=bf16] %1
                    %20:f32[] = dot [
                        dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        accumulation_type=f32,
                    ] %18 %19
                in (%20)
            "#}
            .trim_end(),
        );

        // Valid inputs dequantize with the observed block size and contract.
        let inputs = |scale: Vec<f32>| {
            vec![
                ArrayIrValue::Array(Array::vector(vec![1.0f32; 8]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![1.0f32; 8]).unwrap()),
                ArrayIrValue::Array(Array::vector(scale).unwrap()),
            ]
        };
        assert_eq!(
            program.interpret(inputs(vec![1.0, 2.0, 3.0, 4.0])),
            Ok(vec![ArrayIrValue::Array(Array::scalar(20.0f32).unwrap())]),
        );

        // An empty scale reports the positivity assertion rather than a failed remainder or division, and a positive
        // scale extent that does not divide the element extent reports the divisibility assertion. Partially evaluating
        // the program with unknown inputs preserves both diagnostics.
        let evaluation = program
            .partially_evaluate(&program.input_types().into_iter().map(PartialValue::Unknown).collect::<Vec<_>>())
            .unwrap();
        for (scale, message, observations) in [
            (Vec::new(), "scale extent must be positive", vec![("scale".to_string(), "0".to_string())]),
            (
                vec![1.0f32; 3],
                "element extent must be divisible by scale extent",
                vec![("elements".to_string(), "8".to_string()), ("scale".to_string(), "3".to_string())],
            ),
        ] {
            let expected = AssertionError::Failed { message: message.to_string(), observations };
            let error = program.interpret(inputs(scale.clone())).unwrap_err();
            assert_eq!(error.downcast_custom::<AssertionError>(), Some(&expected));
            let residual_error = evaluation
                .interpret(&EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(), &inputs(scale))
                .unwrap_err();
            assert_eq!(residual_error.downcast_custom::<AssertionError>(), Some(&expected));
        }
    }

    #[test]
    fn test_block_quantize() {
        // The NVFP4 recipe uses `f4e2m1fn` elements with `f8e4m3fn` scales `max_abs(block) / 6`. Every block below is a
        // scaled copy of `f4e2m1fn` grid points whose scale is exactly representable in `f8e4m3fn`, so quantization is
        // exact. The block whose values are all zero uses the smallest normal scale, `2^-6`.
        let input = Array::matrix(
            2,
            8,
            vec![
                3.0f32, 1.5, 0.5, 6.0, 0.5, 1.0, 0.25, 1.5, // Blocks with scales `1.0` and `0.25`.
                -12.0, 6.0, 3.0, -1.0, 0.0, 0.0, 0.0, 0.0, // Blocks with scale `2.0` and the smallest scale.
            ],
        )
        .unwrap();
        let (elements, scales) = input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN).unwrap();
        assert_eq!(elements.r#type().into_owned(), ArrayType::new_static(DataType::F4E2M1FN, [2, 8]));
        assert_eq!(scales.r#type().into_owned(), ArrayType::new_static(DataType::F8E4M3FN, [2, 2]));
        assert_eq!(scales.to_f64s(), vec![1.0, 0.25, 2.0, 0.015625]);
        assert_eq!(
            elements.to_f64s(),
            vec![3.0, 1.5, 0.5, 6.0, 2.0, 4.0, 1.0, 6.0, -6.0, 3.0, 1.5, -0.5, 0.0, 0.0, 0.0, 0.0],
        );

        // The quantized inputs contract through `scaled_dot` to the exact full precision dot product.
        let dimensions = DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new());
        assert_eq!(
            elements.scaled_dot(&elements, Some(&scales), Some(&scales), Some(&dimensions), Some(DataType::F32)),
            input.dot(&input, &dimensions),
        );
    }

    #[test]
    fn test_block_quantize_mxfp8() {
        // The OCP MX recipe uses `f8e4m3fn` elements with power of two `f8e8m0fnu` scales
        // `2^(floor(log2(max_abs(block))) - 8)`. The block maxima below are exact powers of two, which exercises the
        // boundary nudge of the `log2` composition, and every quotient is exactly representable in `f8e4m3fn`, so
        // quantization is exact. The block whose values are all zero uses the smallest scale, `2^-126`.
        let input = Array::matrix(
            2,
            8,
            vec![
                4.0f32, 2.0, 1.0, 0.5, 1.75, 0.5, -1.0, 0.25, // Blocks with scales `2^-6` and `2^-8`.
                -8.0, 4.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, // Blocks with scale `2^-5` and the smallest scale.
            ],
        )
        .unwrap();
        let (elements, scales) = input.block_quantize(4, DataType::F8E4M3FN, DataType::F8E8M0FNU).unwrap();
        assert_eq!(elements.r#type().into_owned(), ArrayType::new_static(DataType::F8E4M3FN, [2, 8]));
        assert_eq!(scales.r#type().into_owned(), ArrayType::new_static(DataType::F8E8M0FNU, [2, 2]));
        assert_eq!(scales.to_f64s(), vec![(-6.0f64).exp2(), (-8.0f64).exp2(), (-5.0f64).exp2(), (-126.0f64).exp2()]);
        assert_eq!(
            elements.to_f64s(),
            vec![256.0, 128.0, 64.0, 32.0, 448.0, 128.0, -256.0, 64.0, -256.0, 128.0, 64.0, 32.0, 0.0, 0.0, 0.0, 0.0],
        );

        // The quantized inputs contract through `scaled_dot` to the exact full precision dot product.
        let dimensions = DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new());
        assert_eq!(
            elements.scaled_dot(&elements, Some(&scales), Some(&scales), Some(&dimensions), Some(DataType::F32)),
            input.dot(&input, &dimensions),
        );
    }

    #[test]
    fn test_block_quantize_clamps_scales() {
        // An NVFP4 block maximum above `6 · 448` would need a scale beyond the largest finite `f8e4m3fn` value, so the
        // scale saturates at `448` and the quotients saturate at the largest element magnitude instead of becoming NaN.
        let input = Array::vector(vec![6000.0f32, 3000.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
        let (elements, scales) = input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN).unwrap();
        assert_eq!(scales.to_f64s(), vec![448.0, 0.171875]);
        assert_eq!(elements.to_f64s(), vec![6.0, 6.0, 0.0, 0.0, 6.0, 6.0, 6.0, 6.0]);

        // Likewise, an `f64` MX block maximum beyond the `f8e8m0fnu` range saturates at the largest scale, `2^127`.
        let input = Array::vector(vec![1e300f64, 0.0, 0.0, 0.0]).unwrap();
        let (elements, scales) = input.block_quantize(4, DataType::F8E4M3FN, DataType::F8E8M0FNU).unwrap();
        assert_eq!(scales.to_f64s(), vec![127.0f64.exp2()]);
        assert_eq!(elements.to_f64s(), vec![448.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_block_quantize_round_trip_tolerance() {
        // Values off the storage grids round-trip within the quantization error of the element type. `f8e4m3fn` has
        // three mantissa bits, so each dequantized element is within about 6% of its input (apart from the OCP MX
        // saturation of a block maximum that lands past the finite range), and the contraction of eight such products
        // stays within a proportional tolerance of the full precision dot product.
        let values = [1.1f32, -2.3, 0.7, 3.9, 0.013, -0.27, 5.4, 8.9];
        let input = Array::matrix(1, 8, values.to_vec()).unwrap();
        let (elements, scales) = input.block_quantize(4, DataType::F8E4M3FN, DataType::F8E8M0FNU).unwrap();
        assert_eq!(elements.r#type().into_owned(), ArrayType::new_static(DataType::F8E4M3FN, [1, 8]));
        assert_eq!(scales.r#type().into_owned(), ArrayType::new_static(DataType::F8E8M0FNU, [1, 2]));
        let dimensions = DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new());
        let product = elements
            .scaled_dot(&elements, Some(&scales), Some(&scales), Some(&dimensions), Some(DataType::F32))
            .unwrap()
            .to_f64s()[0];
        let expected = input.dot(&input, &dimensions).unwrap().to_f64s()[0];
        assert_abs_diff_eq!(product, expected, epsilon = 0.05 * expected);
    }

    #[test]
    fn test_block_quantize_ranks() {
        // Inputs of any positive rank quantize each block of their last dimension identically.
        let values = [1.1f32, -2.3, 0.7, 3.9, 0.013, -0.27, 5.4, 8.9];
        let (elements, scales) = Array::vector(values.to_vec())
            .unwrap()
            .block_quantize(4, DataType::F8E4M3FN, DataType::F8E8M0FNU)
            .unwrap();
        assert_eq!(elements.r#type().into_owned(), ArrayType::new_static(DataType::F8E4M3FN, [8]));
        assert_eq!(scales.r#type().into_owned(), ArrayType::new_static(DataType::F8E8M0FNU, [2]));
        let rank_four =
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 1, 1, 8]), &values).unwrap();
        let (rank_four_elements, rank_four_scales) =
            rank_four.block_quantize(4, DataType::F8E4M3FN, DataType::F8E8M0FNU).unwrap();
        assert_eq!(rank_four_elements.r#type().into_owned(), ArrayType::new_static(DataType::F8E4M3FN, [1, 1, 1, 8]));
        assert_eq!(rank_four_scales.r#type().into_owned(), ArrayType::new_static(DataType::F8E8M0FNU, [1, 1, 1, 2]));
        assert_eq!(rank_four_elements.to_f64s(), elements.to_f64s());
        assert_eq!(rank_four_scales.to_f64s(), scales.to_f64s());
    }

    #[test]
    fn test_block_quantize_array_ir() {
        // Concrete array IR values quantize their array members exactly like the arrays themselves.
        let input = Array::matrix(1, 8, vec![3.0f32, 1.5, 0.5, 6.0, 0.5, 1.0, 0.25, 1.5]).unwrap();
        let (elements, scales) = input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN).unwrap();
        assert_eq!(
            ArrayIrValue::Array(input).block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN),
            Ok((ArrayIrValue::Array(elements), ArrayIrValue::Array(scales))),
        );
    }

    #[test]
    fn test_block_quantize_validation() {
        let input = Array::matrix(2, 8, vec![1.0f32; 16]).unwrap();
        let error = |message: &str| Err(ProgramError::from(TypeError::invalid(message)));
        assert_eq!(
            input.block_quantize(4, DataType::F16, DataType::F8E4M3FN),
            error("`block_quantize` does not support element data type `f16`"),
        );
        assert_eq!(
            input.block_quantize(4, DataType::F4E2M1FN, DataType::F16),
            error("`block_quantize` does not support scale data type `f16`"),
        );
        assert_eq!(
            Array::vector(vec![1i32; 8]).unwrap().block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN),
            error("`block_quantize` expects an `f32` or `f64` input but got `i32`"),
        );
        assert_eq!(
            Array::scalar(1.0f32).unwrap().block_quantize(2, DataType::F4E2M1FN, DataType::F8E4M3FN),
            error("`block_quantize` does not support rank-0 inputs"),
        );

        // `ScaledDot` requires blocks of at least two values and non-empty scales.
        assert_eq!(
            input.block_quantize(1, DataType::F4E2M1FN, DataType::F8E4M3FN),
            error("`block_quantize` block size must be at least 2 but got 1"),
        );
        assert_eq!(
            input.block_quantize(3, DataType::F4E2M1FN, DataType::F8E4M3FN),
            error("`block_quantize` trailing dimension size 8 is not a positive multiple of block size 3"),
        );
        assert_eq!(
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [2, 0]), &[])
                .unwrap()
                .block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN),
            error("`block_quantize` trailing dimension size 0 is not a positive multiple of block size 4"),
        );

        // Staged inputs must have a static shape, and rejected requests stage nothing.
        let extent = Dimension::Dynamic(DimensionVariable::new("n", DimensionBounds::new(0, Some(8)).unwrap()));
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input: Tracer<_>| {
                assert!(matches!(
                    input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN),
                    Err(error) if error
                        == ProgramError::from(TypeError::invalid("`block_quantize` input must have a static shape")),
                ));
                Ok(input)
            },
            ArrayType::new(DataType::F32, Shape::new(vec![extent])),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[n] .
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_block_quantize_staging() {
        // Quantizing a tracer stages the composition, and the staged outputs carry the element and scale types.
        let (output_types, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input: Tracer<_>| {
                let (elements, scales) = input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN)?;
                Ok(vec![elements, scales])
            },
            ArrayType::new_static(DataType::F32, [2, 8]),
        )
        .unwrap();
        assert_eq!(
            output_types,
            vec![ArrayType::new_static(DataType::F4E2M1FN, [2, 8]), ArrayType::new_static(DataType::F8E4M3FN, [2, 2])],
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 8] .
                let %1:f32[2, 2, 4] = reshape [shape=[2, 2, 4]] %0
                    %2:f32[2, 2, 4] = abs %1
                    %3:f32[2, 2] = reduce [kind=max, axes=[2]] %2
                    %4:f32[] = constant [value=6.0]
                    %5:f32[2, 2] = broadcast [output_type=f32[2, 2], output_axes=[]] %4
                    %6:f32[2, 2] = div %3 %5
                    %7:f32[] = constant [value=0.015625]
                    %8:f32[] = constant [value=448.0]
                    %9:f32[2, 2] = clamp %7 %6 %8
                    %10:f8e4m3fn[2, 2] = convert_element_type [data_type=f8e4m3fn] %9
                    %11:f32[2, 2] = convert_element_type [data_type=f32] %10
                    %12:f32[2, 2, 4] = broadcast [output_type=f32[2, 2, 4], output_axes=[0, 1]] %11
                    %13:f32[2, 8] = reshape [shape=[2, 8]] %12
                    %14:f32[2, 8] = div %0 %13
                    %15:f32[] = constant [value=-6.0]
                    %16:f32[] = constant [value=6.0]
                    %17:f32[2, 8] = clamp %15 %14 %16
                    %18:f4e2m1fn[2, 8] = convert_element_type [data_type=f4e2m1fn] %17
                in (%18, %10)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_block_quantize_sharding() {
        // Inside a manual region, the elements and scales of a varying input keep its variation for both recipes,
        // because the scales are broadcast back over their blocks with their own variation rather than to an invariant
        // type and the recipe constants are aligned with the varying values that they combine with.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying = |data_type: DataType, size: usize| {
            ArrayType::new_static(data_type, [size])
                .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
                .unwrap()
        };
        let (output_types, _) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input: Tracer<_>| {
                let (elements, scales) = input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN)?;
                Ok(vec![elements, scales])
            },
            varying(DataType::F32, 8),
            vec![("x".to_string(), NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2 })],
        )
        .unwrap();
        assert_eq!(output_types, vec![varying(DataType::F4E2M1FN, 8), varying(DataType::F8E4M3FN, 2)]);
        let (output_types, _) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input: Tracer<_>| {
                let (elements, scales) = input.block_quantize(4, DataType::F8E4M3FN, DataType::F8E8M0FNU)?;
                Ok(vec![elements, scales])
            },
            varying(DataType::F32, 8),
            vec![("x".to_string(), NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2 })],
        )
        .unwrap();
        assert_eq!(output_types, vec![varying(DataType::F8E4M3FN, 8), varying(DataType::F8E8M0FNU, 2)]);

        // Explicit placement of the leading dimension is likewise kept rather than dropped by the scale broadcast.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded = |data_type: DataType, columns: usize| {
            ArrayType::new_static(data_type, [2, columns])
                .with_sharding(
                    Sharding::new(
                        mesh.clone(),
                        vec![ShardingDimension::sharded(["y"]), ShardingDimension::replicated()],
                    )
                    .unwrap(),
                )
                .unwrap()
        };
        let (output_types, _) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input: Tracer<_>| {
                let (elements, scales) = input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN)?;
                Ok(vec![elements, scales])
            },
            sharded(DataType::F32, 8),
        )
        .unwrap();
        assert_eq!(output_types, vec![sharded(DataType::F4E2M1FN, 8), sharded(DataType::F8E4M3FN, 2)]);
    }
}
