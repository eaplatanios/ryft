//! Generalized and grouped tensor contractions through [`Dot`] and [`RaggedDot`]. [`DotOperation`] pairs batching
//! dimensions, sums over contracting dimensions, and preserves each input's remaining dimensions. Its output axes are
//! ordered as batching dimensions, left result dimensions, then right result dimensions. Inputs have identical numeric
//! element types, while an optional accumulation type selects the result representation.
//!
//! [`RaggedDotOperation`] additionally consumes integer group sizes that partition one left input dimension into
//! consecutive intervals. Depending on that dimension's role, groups select right input slices, produce a leading group
//! output dimension, or leave ordinary batched dot semantics unchanged. Explicit group sizes describe dense packed
//! buffers rather than introducing array extents that vary across batch items.
//!
//! Both operations support eager arrays, staged programs, array IR projections, partial evaluation, batching, and
//! forward mode differentiation. Reverse mode transposes the bilinear linearizations. Ragged transposition currently
//! supports only a non-contracting ragged dimension. Refer to the operation types for placement and metadata
//! constraints.
//!
//! # Example
//!
//! ```rust
//! # use ryft_core::{Array, Dot, DotDimensionNumbers, ProgramError};
//! # fn main() -> Result<(), ProgramError> {
//! let left = Array::matrix(1, 2, vec![2.0f32, 3.0])?;
//! let right = Array::matrix(2, 1, vec![5.0f32, 7.0])?;
//! assert_eq!(left.dot(&right, &DotDimensionNumbers::matmul())?.elements::<f32>()?, vec![31.0]);
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeSet;
use std::fmt::Display;
use std::sync::Arc;

use ryft_macros::capability;

use crate::arrays::{
    Array, ArrayAddressing, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrType, ArrayType,
    DataType, Dimension, DimensionSource, MeshAxisType, NumericArrayElement, Shape, Sharding, ShardingDimension,
    StaticShape, UnreducedKind,
};
use crate::axes::Axis;
use crate::batching::{
    BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError, InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{
    DifferentiableType, DifferentiationContext, DifferentiationDual, DifferentiationError, DifferentiationPolicy,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, check_types, dispatch_on_array_element_type, impl_differentiable_operation};
use crate::operations::Capability;
use crate::operations::arithmetic::Add;
use crate::operations::collectives::parallel_vary::ManualVariationAlignment;
use crate::operations::manipulation::conversions::{ConvertElementType, ConvertElementTypeOperation};
use crate::operations::manipulation::reshaping::Reshape;
use crate::operations::manipulation::slicing::Slice;
use crate::operations::manipulation::transposition::{Permutation, Transpose, TransposeOperation};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, ProgramError, RegionInterface, TypeError, Typed, Value,
    ValueDomainDispatch,
};

/// Specification of contracting and batching dimensions for a generalized dot product. The contracting dimensions index
/// axes that are summed over, such as the `K` axes in matrix multiplication. The batching dimensions index axes aligned
/// pairwise between the inputs and preserved in the output. These are the dimension numbers of StableHLO's
/// [`dot_general`](https://openxla.org/stablehlo/spec#dot_general).
///
/// Both `lhs_contracting_dimensions` and `rhs_contracting_dimensions` must have the same length and their
/// corresponding dimensions in the two inputs must match in size. The same applies to `lhs_batching_dimensions` and
/// `rhs_batching_dimensions`. Every list must contain distinct axes, and an input's batching and contracting lists must
/// be disjoint. Construction stores the specification without validating input ranks. [`DotOperation`] validates it
/// against the input types before execution or staging.
///
/// The output shape is `[batching..., lhs_result..., rhs_result...]`, where each input's result dimensions are those
/// that are neither contracting nor batching, preserved in their original order.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DotDimensionNumbers {
    /// Axes on the left-hand side that contract with `rhs_contracting_dimensions` on the right-hand side.
    lhs_contracting_dimensions: Vec<usize>,

    /// Axes on the right-hand side that contract with `lhs_contracting_dimensions` on the left-hand side.
    rhs_contracting_dimensions: Vec<usize>,

    /// Axes on the left-hand side that are aligned 1:1 with `rhs_batching_dimensions` on the right-hand side and that
    /// are preserved in the output.
    lhs_batching_dimensions: Vec<usize>,

    /// Axes on the right-hand side that are aligned 1:1 with `lhs_batching_dimensions` on the left-hand side and that
    /// are preserved in the output.
    rhs_batching_dimensions: Vec<usize>,
}

impl DotDimensionNumbers {
    /// Creates a new [`DotDimensionNumbers`] instance from the provided explicit contracting and batching dimensions.
    #[inline]
    pub fn new(
        lhs_contracting_dimensions: Vec<usize>,
        rhs_contracting_dimensions: Vec<usize>,
        lhs_batching_dimensions: Vec<usize>,
        rhs_batching_dimensions: Vec<usize>,
    ) -> Self {
        Self {
            lhs_contracting_dimensions,
            rhs_contracting_dimensions,
            lhs_batching_dimensions,
            rhs_batching_dimensions,
        }
    }

    /// Returns the [`DotDimensionNumbers`] of a rank-two matrix multiplication `[M, K] × [K, N] → [M, N]`. The last
    /// left input dimension contracts with the first right input dimension, and neither input has batching dimensions.
    #[inline]
    pub fn matmul() -> Self {
        Self::new(vec![1], vec![0], Vec::new(), Vec::new())
    }

    /// Returns the [`DotDimensionNumbers`] of a rank-one inner product `[K] · [K] → []`, which contracts the single
    /// dimension of each input.
    #[inline]
    pub fn inner_product() -> Self {
        Self::new(vec![0], vec![0], Vec::new(), Vec::new())
    }

    /// Returns the contracting dimensions of the left-hand side.
    #[inline]
    pub fn lhs_contracting_dimensions(&self) -> &[usize] {
        &self.lhs_contracting_dimensions
    }

    /// Returns the contracting dimensions of the right-hand side.
    #[inline]
    pub fn rhs_contracting_dimensions(&self) -> &[usize] {
        &self.rhs_contracting_dimensions
    }

    /// Returns the batching dimensions of the left-hand side.
    #[inline]
    pub fn lhs_batching_dimensions(&self) -> &[usize] {
        &self.lhs_batching_dimensions
    }

    /// Returns the batching dimensions of the right-hand side.
    #[inline]
    pub fn rhs_batching_dimensions(&self) -> &[usize] {
        &self.rhs_batching_dimensions
    }

    /// Returns the result dimensions of a left-hand side with rank `lhs_rank`, which are its dimensions that are
    /// neither batching nor contracting, in their original order. These dimensions follow the batching dimensions
    /// in the output. This function does not validate the specification against `lhs_rank`.
    pub fn lhs_result_dimensions(&self, lhs_rank: usize) -> Vec<usize> {
        (0..lhs_rank)
            .filter(|axis| {
                !self.lhs_batching_dimensions.contains(axis) && !self.lhs_contracting_dimensions.contains(axis)
            })
            .collect()
    }

    /// Returns the result dimensions of a right-hand side with rank `rhs_rank`, which are its dimensions that are
    /// neither batching nor contracting, in their original order. These dimensions are the trailing output dimensions.
    /// This function does not validate the specification against `rhs_rank`.
    pub fn rhs_result_dimensions(&self, rhs_rank: usize) -> Vec<usize> {
        (0..rhs_rank)
            .filter(|axis| {
                !self.rhs_batching_dimensions.contains(axis) && !self.rhs_contracting_dimensions.contains(axis)
            })
            .collect()
    }

    /// Lifts these dimension numbers through one batching level. Batch axis positions refer to the physical inputs,
    /// while these dimension numbers describe each logical item. A missing position denotes a replicated input.
    /// When both inputs are mapped, their inserted axes become the first batching pair and every existing axis at or
    /// after its insertion position shifts by one. The output's mapped axis is then zero. Two replicated inputs keep
    /// the original dimensions and have a replicated output. A mixed pair returns [`None`], so callers must first
    /// materialize the mapped axis on the replicated input. This function does not validate ranks or mapped extents.
    pub fn batched(
        &self,
        lhs_batch_axis: Option<usize>,
        rhs_batch_axis: Option<usize>,
    ) -> Option<(Self, Option<usize>)> {
        let shift = |axes: &[usize], batch_axis: usize| -> Vec<usize> {
            axes.iter().map(|axis| if *axis >= batch_axis { *axis + 1 } else { *axis }).collect()
        };
        match (lhs_batch_axis, rhs_batch_axis) {
            (Some(lhs_axis), Some(rhs_axis)) => {
                let mut lhs_batching_dimensions = vec![lhs_axis];
                lhs_batching_dimensions.extend(shift(&self.lhs_batching_dimensions, lhs_axis));
                let mut rhs_batching_dimensions = vec![rhs_axis];
                rhs_batching_dimensions.extend(shift(&self.rhs_batching_dimensions, rhs_axis));
                let dimensions = Self::new(
                    shift(&self.lhs_contracting_dimensions, lhs_axis),
                    shift(&self.rhs_contracting_dimensions, rhs_axis),
                    lhs_batching_dimensions,
                    rhs_batching_dimensions,
                );
                Some((dimensions, Some(0)))
            }
            (None, None) => Some((self.clone(), None)),
            _ => None,
        }
    }

    /// Validates these dimension numbers against the ranks of the left-hand side and right-hand side inputs before
    /// indexing or lifting. Paired batching lists and paired contracting lists must have equal lengths. Each list must
    /// contain distinct indices within its input's rank, and an input's batching and contracting lists must be
    /// disjoint.
    ///
    /// Validation covers list lengths and dimension indices. The operation's type inference additionally validates
    /// paired extents, element types, and sharding.
    ///
    /// # Parameters
    ///
    ///   - `operation_name`: Name used in diagnostics, allowing other dot-based operations to reuse this validation.
    ///   - `lhs_rank`: Rank of the left-hand side input.
    ///   - `rhs_rank`: Rank of the right-hand side input.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if paired lists have different lengths, an index is out of bounds, a list repeats an
    /// axis, or an input uses the same axis for batching and contraction.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{DOT_OPERATION_NAME, DotDimensionNumbers};
    /// assert_eq!(DotDimensionNumbers::matmul().validate(DOT_OPERATION_NAME, 2, 2), Ok(()));
    /// ```
    pub fn validate(&self, operation_name: &str, lhs_rank: usize, rhs_rank: usize) -> Result<(), TypeError> {
        if self.lhs_batching_dimensions.len() != self.rhs_batching_dimensions.len() {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` batching dimensions have different lengths on the two inputs",
            )));
        }
        if self.lhs_contracting_dimensions.len() != self.rhs_contracting_dimensions.len() {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` contracting dimensions have different lengths on the two inputs",
            )));
        }
        validate_dot_axes(
            operation_name,
            &self.lhs_batching_dimensions,
            &self.lhs_contracting_dimensions,
            lhs_rank,
            "left-hand side",
        )?;
        validate_dot_axes(
            operation_name,
            &self.rhs_batching_dimensions,
            &self.rhs_contracting_dimensions,
            rhs_rank,
            "right-hand side",
        )
    }
}

impl Display for DotDimensionNumbers {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "(lhs_contracting={:?}, rhs_contracting={:?}, lhs_batching={:?}, rhs_batching={:?})",
            self.lhs_contracting_dimensions,
            self.rhs_contracting_dimensions,
            self.lhs_batching_dimensions,
            self.rhs_batching_dimensions,
        )
    }
}

/// Canonical operation name for [`DotOperation`].
pub const DOT_OPERATION_NAME: &str = "dot";

/// Primitive representing a generalized dot operation (i.e., a tensor contraction) of two identically typed numeric
/// inputs. Boolean, token, and structural zero element types are rejected. [`DotOperation`] is the unified primitive
/// for matrix multiplication, batched matrix multiplication, vector inner products, and arbitrary tensor contractions.
/// The XLA backend lowers it to StableHLO's [`dot_general`](https://openxla.org/stablehlo/spec#dot_general)
/// instruction.
///
/// Without an accumulation type, the output keeps the input element type. The eager [`Array`] kernel accumulates
/// floating-point inputs narrower than `f32` at `f32` and rounds each result once, which is how the XLA backend
/// evaluates such dot operations. Integer products and sums wrap at the declared element width, and every other
/// element type accumulates at itself.
///
/// A dot is bilinear, so for `y = dot(a, b)`, forward mode differentiation computes `ẏ = dot(ȧ, b) + dot(a, ḃ)`.
/// Transposition accepts exactly one linear input and contracts the output cotangent `ȳ` with the other, known input
/// using the corresponding adjoint dimension numbers. For a matrix multiplication `y = a · b`, for example, it computes
/// `ā = ȳ · bᵀ` when `a` is linear and `b̄ = aᵀ · ȳ` when `b` is linear. Dots with an accumulation type perform those
/// contractions at the widened cotangent type before converting the result back to the linear input's cotangent
/// representation.
///
/// Each forward mode tangent term is staged as an ordinary dot that preserves the primal dimension numbers,
/// accumulation type, and requested output sharding without introducing captures. The adjoint contraction emits the
/// linear input's axes in dot order, so transposition requests the cotangent dual of the linear input's sharding in
/// that order and then restores the original axis order and placement with the inverse permutation. A structural zero
/// output cotangent remains structural zero.
///
/// Batching aligns the inputs onto a common mapped axis and lifts the dimension numbers past it. Materializing an axis
/// on a replicated input preserves a dynamic mapped extent as a first-class value. Two mapped inputs must describe the
/// same mapped extent. For dynamic extents, they must share the same [`DimensionVariable`](crate::DimensionVariable).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DotOperation {
    /// Contracting and batching dimension specification.
    dimensions: DotDimensionNumbers,

    /// Optional accumulation data type. Refer to the documentation of [`Self::with_accumulation_type`].
    accumulation_type: Option<DataType>,

    /// Optional requested output [`Sharding`]. Refer to the documentation of [`Self::with_output_sharding`].
    output_sharding: Option<Sharding>,
}

impl DotOperation {
    /// Creates a new [`DotOperation`] with the provided dimension numbers.
    #[inline]
    pub fn new(dimensions: DotDimensionNumbers) -> Self {
        Self { dimensions, accumulation_type: None, output_sharding: None }
    }

    /// Returns a [`DotOperation`] configured for rank-two matrix multiplication.
    #[inline]
    pub fn matmul() -> Self {
        Self::new(DotDimensionNumbers::matmul())
    }

    /// Returns this [`DotOperation`] with the provided accumulation data type. The input element types must still
    /// match each other and must support the accumulation type, which becomes the output element type. The eager
    /// kernel upcasts the inputs and accumulates the contraction at the wider type. This is the
    /// `preferred_element_type` contract of XLA, which its low precision matrix units implement natively (e.g.,
    /// `f8 × f8 → f32` and `bf16 × bf16 → f32`). Dots with an accumulation type differentiate like ordinary dots,
    /// with tangents and cotangents carried at the accumulation type (refer to the forward mode and transposition
    /// documentation of this operation), and they can also request an output sharding (refer to
    /// [`Self::with_output_sharding`]).
    #[inline]
    pub fn with_accumulation_type<D: Into<Option<DataType>>>(mut self, accumulation_type: D) -> Self {
        self.accumulation_type = accumulation_type.into();
        self
    }

    /// Returns this [`DotOperation`] with a requested output [`Sharding`]. Type inference validates its rank, its mesh,
    /// the absence of auto axes, and the rule for unreduced outputs before using it instead of the inferred sharding.
    /// The request bypasses the ordinary consistency checks for batching and contracting dimension shardings. This is
    /// the only way to produce an output with unreduced axes, which represents partial results on each device whose
    /// reduction across devices is delayed. Logical dependence on manual shards still comes from the input values.
    #[inline]
    pub fn with_output_sharding<S: Into<Option<Sharding>>>(mut self, output_sharding: S) -> Self {
        self.output_sharding = output_sharding.into();
        self
    }

    /// Returns the contracting and batching dimension specification of this [`DotOperation`].
    #[inline]
    pub fn dimensions(&self) -> &DotDimensionNumbers {
        &self.dimensions
    }

    /// Returns the optional accumulation data type of this [`DotOperation`]. Refer to the documentation of
    /// [`Self::with_accumulation_type`] for more information on what the accumulation data type is and how it is used.
    #[inline]
    pub fn accumulation_type(&self) -> Option<DataType> {
        self.accumulation_type
    }

    /// Returns the requested output sharding of this [`DotOperation`], if any.
    #[inline]
    pub fn output_sharding(&self) -> Option<&Sharding> {
        self.output_sharding.as_ref()
    }

    /// Applies this operation to `lhs` and `rhs` through the [`Dot`] capability, forwarding the accumulation type and
    /// the requested output sharding, so that interpretation and differentiation over staging values preserve them.
    fn apply<V: Dot>(&self, lhs: &V, rhs: &V) -> Result<V, ProgramError> {
        lhs.dot_with_configuration(rhs, &self.dimensions, self.accumulation_type, self.output_sharding.as_ref())
    }
}

impl Display for DotOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for DotOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        DOT_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        Ok(vec![infer_dot_output_type(
            DOT_OPERATION_NAME,
            &input_types[0],
            &input_types[1],
            &self.dimensions,
            self.accumulation_type,
            self.output_sharding.as_ref(),
        )?])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("dimensions", &self.dimensions)?;
            if let Some(accumulation_type) = self.accumulation_type {
                operation.field("accumulation_type", &accumulation_type)?;
            }
            if let Some(output_sharding) = &self.output_sharding {
                operation.field("output_sharding", output_sharding)?;
            }
            Ok(())
        })
    }
}

impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for DotOperation
where
    C::Value: Dot,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // Concrete values record the requested sharding and upcast for the accumulation type, while staging values
        // (e.g., during program batching) stage an operation that preserves both.
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![self.apply(&inputs[0], &inputs[1])?])
    }
}

impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for DotOperation where
    C::Operation: From<DotOperation>
{
}

impl<C: Context<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>>
    for DotOperation
where
    DotOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        // Alignment delegates mapped axis materialization to `ArrayExtentBatchingPolicy::match_axis`, so a dynamic
        // mapped extent remains a first-class value. The lifted contraction keeps each item independent across the
        // inserted axis.
        check_count!("input", inputs, 2, ProgramError);
        self.dimensions.validate(
            DOT_OPERATION_NAME,
            inputs[0].unbatched_type().rank(),
            inputs[1].unbatched_type().rank(),
        )?;
        validate_common_mapped_dimension(DOT_OPERATION_NAME, inputs)?;

        // A mixed pair needs a physical batch axis on its replicated input before the logical dimension numbers can be
        // lifted. Insert that axis at position zero through the active policy, so a dynamic mapped extent remains an
        // extent value grounding the broadcast instead of requiring a statically known host size.
        let batch_axes: Vec<Option<usize>> = inputs.iter().map(|input| input.batch_axis_position()).collect();
        let aligned_inputs: Vec<ArrayBatch<C::Value>> = match (batch_axes[0], batch_axes[1]) {
            (Some(_), Some(_)) | (None, None) => inputs.to_vec(),
            (Some(_), None) => vec![inputs[0].clone(), P::match_axis(context, &inputs[1], Axis::from(0))?],
            (None, Some(_)) => vec![P::match_axis(context, &inputs[0], Axis::from(0))?, inputs[1].clone()],
        };
        let (lifted_dimensions, output_axis) = self
            .dimensions
            .batched(aligned_inputs[0].batch_axis_position(), aligned_inputs[1].batch_axis_position())
            .ok_or_else(|| BatchingError::MisalignedBatchAxes {
                message: format!(
                    "`{DOT_OPERATION_NAME}` batching failed to lift its dimension numbers for the aligned batch axes",
                ),
            })?;

        // The output batch axis carries the sharding of the inputs' mapped axes, including in a requested output
        // sharding.
        let axis_sharding = ArrayBatch::sharding_for_inputs(inputs)?;
        let output_sharding = match (self.output_sharding(), output_axis) {
            (Some(output_sharding), Some(axis)) => {
                Some(output_sharding.batched(axis, axis_sharding).map_err(ProgramError::from)?)
            }
            (Some(output_sharding), None) => Some(output_sharding.clone()),
            (None, _) => None,
        };
        let lifted_operation = DotOperation::new(lifted_dimensions)
            .with_accumulation_type(self.accumulation_type)
            .with_output_sharding(output_sharding);
        let output_batch_axes = [BatchAxis::from_optional_position(output_axis)];
        lifted_operation.interpret_with_batch_axes(context, &aligned_inputs, &output_batch_axes)
    }
}

impl_differentiable_operation! {
    DotOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Value: ConvertElementType + Dot + Add,
        C::Operation: From<DotOperation>,
    {
        |operation, context, _driver, inputs| {
            // Forward mode differentiation applies the product rule. Each term holds the corresponding primal input
            // fixed on its original contracting side and stages an ordinary dot with the primal operation's dimensions,
            // accumulation type, and requested output sharding.
            check_count!("input", inputs, 2, ProgramError);
            let left = &inputs[0];
            let right = &inputs[1];
            let primal = operation.apply(left.primal(), right.primal())?;
            Ok(vec![bilinear_array_jvp(context, primal, left, right, |left, right| Ok(operation.apply(left, right)?))?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: From<ConvertElementTypeOperation<ArrayType>> + From<DotOperation> + From<TransposeOperation>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // A generalized dot is bilinear rather than jointly linear, so a valid pullback has exactly one linear
            // input. The known input selects the corresponding adjoint dimensions, and the result is pinned to the
            // linear input's cotangent sharding and element representation.
            check_count!("input", inputs, 2, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 2, DifferentiationError);
            operation.infer_output_types(
                &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
                &[],
            )?;

            let (linear_index, known_index) = match (inputs[0].is_unknown(), inputs[1].is_unknown()) {
                // Both inputs linear is a bilinear product, which is not a linear map in both inputs jointly and so
                // never appears in a valid pushforward.
                (true, true) => {
                    return Err(ProgramError::UnsupportedOperation {
                        message: format!("bilinear `{DOT_OPERATION_NAME}` with two linear inputs cannot be transposed"),
                    }
                    .into());
                }
                (false, false) => return Ok(()),
                (true, false) => (0, 1),
                (false, true) => (1, 0),
            };

            // Demand is independent of primal knowledge. An ignored linear input stays unknown, but no matrix
            // multiplication or element conversion is needed for its cotangent. A zero output cotangent stays
            // a structural zero.
            if !accumulators[linear_index].is_needed() {
                return Ok(());
            }

            let MaybeZero::Value(output_cotangent) = &outputs[0] else {
                return Ok(());
            };

            // The dispatch guarantees that a known input carries its pullback value, and a primal with an accumulation
            // type contracts its adjoint at the widened cotangent type.
            let known_value = inputs[known_index].as_known().unwrap();
            let known_value = if known_value.r#type().data_type() == output_cotangent.r#type().data_type() {
                known_value.clone()
            } else {
                known_value.convert_element_type(output_cotangent.r#type().data_type())?
            };

            // The adjoint contraction emits its axes in dot order, which may differ from the linear input's order.
            // Request placement in that intermediate order, then restore both the shape and the placement with the
            // inverse permutation.
            let left_rank = inputs[0].r#type().rank();
            let right_rank = inputs[1].r#type().rank();
            let primal_dimensions = operation.dimensions();
            let (dimensions, output_axes) = if linear_index == 0 {
                // Contract the output cotangent with the known right-hand side. The adjoint emits batching axes,
                // left-hand side result axes, and contracting axes sorted by their paired right-hand side axes.
                let batching_count = primal_dimensions.rhs_batching_dimensions.len();
                let lhs_result_dimensions = primal_dimensions.lhs_result_dimensions(left_rank);
                let rhs_result_dimensions = primal_dimensions.rhs_result_dimensions(right_rank);
                let cotangent_rhs_result_start = batching_count + lhs_result_dimensions.len();
                let dimensions = DotDimensionNumbers::new(
                    (cotangent_rhs_result_start..(cotangent_rhs_result_start + rhs_result_dimensions.len())).collect(),
                    rhs_result_dimensions,
                    (0..batching_count).collect(),
                    primal_dimensions.rhs_batching_dimensions.clone(),
                );
                let mut contracting_pairs = primal_dimensions
                    .lhs_contracting_dimensions
                    .iter()
                    .copied()
                    .zip(primal_dimensions.rhs_contracting_dimensions.iter().copied())
                    .collect::<Vec<_>>();
                contracting_pairs.sort_by_key(|(_, rhs_axis)| *rhs_axis);
                let output_axes = primal_dimensions
                    .lhs_batching_dimensions
                    .iter()
                    .copied()
                    .chain(lhs_result_dimensions)
                    .chain(contracting_pairs.into_iter().map(|(lhs_axis, _)| lhs_axis))
                    .collect::<Vec<_>>();
                (dimensions, output_axes)
            } else {
                // Contract the known left-hand side with the output cotangent. The adjoint emits batching axes,
                // contracting axes sorted by their paired left-hand side axes, and right-hand side result axes.
                let batching_count = primal_dimensions.lhs_batching_dimensions.len();
                let lhs_result_dimensions = primal_dimensions.lhs_result_dimensions(left_rank);
                let cotangent_lhs_result_count = lhs_result_dimensions.len();
                let dimensions = DotDimensionNumbers::new(
                    lhs_result_dimensions,
                    (batching_count..(batching_count + cotangent_lhs_result_count)).collect(),
                    primal_dimensions.lhs_batching_dimensions.clone(),
                    (0..batching_count).collect(),
                );
                let mut contracting_pairs = primal_dimensions
                    .rhs_contracting_dimensions
                    .iter()
                    .copied()
                    .zip(primal_dimensions.lhs_contracting_dimensions.iter().copied())
                    .collect::<Vec<_>>();
                contracting_pairs.sort_by_key(|(_, lhs_axis)| *lhs_axis);
                let output_axes = primal_dimensions
                    .rhs_batching_dimensions
                    .iter()
                    .copied()
                    .chain(contracting_pairs.into_iter().map(|(rhs_axis, _)| rhs_axis))
                    .chain(primal_dimensions.rhs_result_dimensions(right_rank))
                    .collect::<Vec<_>>();
                (dimensions, output_axes)
            };

            let linear_cotangent_type = inputs[linear_index].r#type().cotangent()?;
            let adjoint_output_sharding = linear_cotangent_type
                .sharding()
                .map(|sharding| sharding.transpose(output_axes.clone()))
                .transpose()?;
            let adjoint = DotOperation::new(dimensions).with_output_sharding(adjoint_output_sharding);
            let adjoint_inputs = if linear_index == 0 {
                [output_cotangent.clone(), known_value]
            } else {
                [known_value, output_cotangent.clone()]
            };
            let mut adjoint_outputs = context.stage_operation(adjoint, Vec::new(), &adjoint_inputs)?;
            check_count!("output", adjoint_outputs, 1, ProgramError);
            let adjoint_value = adjoint_outputs.remove(0).transpose(Permutation::from(output_axes).inverse()?)?;

            // Convert the adjoint back to the linear input's cotangent element type when the two differ.
            let adjoint_value = if adjoint_value.r#type().data_type() == linear_cotangent_type.data_type() {
                adjoint_value
            } else {
                adjoint_value.convert_element_type(linear_cotangent_type.data_type())?
            };

            accumulators[linear_index].accumulate(context, MaybeZero::Value(adjoint_value))
        }
    },
}

/// Value-level generalized dot product capability. [`Dot`] is the receiver-style entry point for staging or executing
/// [`DotOperation`]s. It performs the contraction described by `dimensions`, supporting standard matrix multiplication,
/// batched matrix multiplication, vector inner products, and arbitrary tensor contractions.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and array IR values implement it for [`ArrayIrType`].
#[capability(projection(ArrayIrType => ArrayType))]
pub trait Dot<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Computes the generalized dot product of `self` and `rhs` using `dimensions`, and returns a [`ProgramError`] if
    /// the inputs are incompatible with `dimensions` or the contraction cannot be recorded in the value's context.
    #[inline]
    fn dot(&self, rhs: &Self, dimensions: &DotDimensionNumbers) -> Result<Self, ProgramError> {
        self.dot_with_configuration(rhs, dimensions, None, None)
    }

    /// Computes the generalized dot product of `self` and `rhs` using `dimensions`, upcasting the inputs to
    /// `accumulation_type` and accumulating the contraction there, so the result carries the accumulation type.
    /// Refer to the documentation of [`DotOperation::with_accumulation_type`] for the exact contract.
    #[inline]
    fn dot_with_accumulation_type(
        &self,
        rhs: &Self,
        dimensions: &DotDimensionNumbers,
        accumulation_type: DataType,
    ) -> Result<Self, ProgramError> {
        self.dot_with_configuration(rhs, dimensions, Some(accumulation_type), None)
    }

    /// Computes the generalized dot product of `self` and `rhs` using `dimensions`, requesting `output_sharding` for
    /// the result. The requested sharding overrides the inferred output sharding and is validated by the operation's
    /// type inference (this is implemented using [`DotOperation::with_output_sharding`]). Staging values attach it to
    /// the staged operation, array IR values project it onto their array members, and concrete [`Array`]s record it on
    /// their result.
    #[inline]
    fn dot_with_output_sharding(
        &self,
        rhs: &Self,
        dimensions: &DotDimensionNumbers,
        output_sharding: &Sharding,
    ) -> Result<Self, ProgramError> {
        self.dot_with_configuration(rhs, dimensions, None, Some(output_sharding))
    }

    /// Computes the generalized dot product of `self` and `rhs` using `dimensions`, accumulating the contraction at
    /// `accumulation_type` when one is provided (refer to [`Self::dot_with_accumulation_type`]) and requesting
    /// `output_sharding` for the result when one is provided (refer to [`Self::dot_with_output_sharding`]). The two
    /// options are independent, because the accumulation type determines the element type of the result while the
    /// requested sharding determines its placement. This is the most general function of this capability, and the
    /// other functions are implemented using it.
    fn dot_with_configuration(
        &self,
        rhs: &Self,
        dimensions: &DotDimensionNumbers,
        accumulation_type: Option<DataType>,
        output_sharding: Option<&Sharding>,
    ) -> Result<Self, ProgramError>;
}

impl Dot for Array {
    fn dot_with_configuration(
        &self,
        rhs: &Self,
        dimensions: &DotDimensionNumbers,
        accumulation_type: Option<DataType>,
        output_sharding: Option<&Sharding>,
    ) -> Result<Self, ProgramError> {
        // The kernel accumulates at the input element type, so an accumulation type upcasts the inputs first, after the
        // `DotOperation` type inference rule validated it against the original input types, which the upcast inputs no
        // longer carry.
        if let Some(accumulation_type) = accumulation_type {
            DotOperation::new(dimensions.clone())
                .with_accumulation_type(accumulation_type)
                .with_output_sharding(output_sharding.cloned())
                .infer_output_types(&[self.r#type().into_owned(), rhs.r#type().into_owned()], &[])?;
            let lhs = self.convert_element_type(accumulation_type)?;
            let rhs = rhs.convert_element_type(accumulation_type)?;
            return lhs.dot_with_configuration(&rhs, dimensions, None, output_sharding);
        }

        // An `Array` is a concrete value on a single device whose payload is always its logical (i.e., fully reduced)
        // value, so a requested sharding only describes the placement of its result, exactly as for `Reshard`. That
        // holds for unreduced axes as well, which describe partial sums whose total is that payload. The `DotOperation`
        // type inference rule validates the request exactly as it does for staged programs and determines the result
        // type.
        let data_type = self.r#type().data_type();
        validate_dot_data_type(DOT_OPERATION_NAME, data_type)?;
        dispatch_on_array_element_type!(
            @numeric data_type,
            |Element| self.dot_elements::<Element>(rhs, dimensions, output_sharding),
        )
    }
}

impl<V: Value<Type = ArrayType, Dispatch = ValueDomainDispatch> + ManualVariationAlignment<ArrayType>> Dot<ArrayType>
    for V
where
    V::Domain: Context<Type = ArrayType>,
    <V::Domain as Domain>::Operation: From<DotOperation>,
{
    fn dot_with_configuration(
        &self,
        rhs: &Self,
        dimensions: &DotDimensionNumbers,
        accumulation_type: Option<DataType>,
        output_sharding: Option<&Sharding>,
    ) -> Result<Self, ProgramError> {
        // Values that carry a context stage a dot through that context. The `ValueDomainDispatch` marker keeps this
        // implementation disjoint from eager values, which implement the capability directly.
        let inputs = [self.clone(), rhs.clone()];
        let inputs = ManualVariationAlignment::align_manual_variation(&inputs)?;
        let operation = DotOperation::new(dimensions.clone())
            .with_accumulation_type(accumulation_type)
            .with_output_sharding(output_sharding.cloned());
        let mut outputs = self.domain().bind(operation, Vec::new(), &inputs)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Role of the single ragged left-hand side dimension in a [`RaggedDotDimensionNumbers`] specification.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RaggedDotMode {
    /// The ragged dimension is a non-contracting result dimension.
    NonContracting,

    /// The ragged dimension is contracted with the right-hand side.
    Contracting,

    /// The ragged dimension is one of the paired batching dimensions.
    Batch,
}

impl Display for RaggedDotMode {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::NonContracting => "non-contracting",
            Self::Contracting => "contracting",
            Self::Batch => "batch",
        })
    }
}

/// Dimension numbers of a ragged dot operation (i.e., a [`RaggedDotOperation`]). They extend the ordinary
/// [`DotDimensionNumbers`], which describe the contracting and batching axes, with the axes that carry the groups.
///
///   - The left-hand side ragged dimension is the single left-hand side axis that the group sizes input partitions
///     into consecutive groups. Its role among the dot dimensions selects the [`RaggedDotMode`].
///   - The optional right-hand side group dimension is the right-hand side axis that indexes groups. A valid
///     non-contracting specification has one, which must be neither a batching nor a contracting axis, and the
///     contracting and batching modes have none.
///
/// These fields correspond to the `lhs_ragged_dimensions` and `rhs_group_dimensions` lists of JAX's
/// [`ragged_dot_general`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_dot_general.html) and of
/// `chlo.ragged_dot`, which mirror the repeated fields of XLA's `RaggedDotDimensionNumbers` proto but accept exactly
/// one ragged dimension and at most one group dimension. Construction stores the specification without validating it.
/// [`Self::mode`] validates the ragged dimension, and [`RaggedDotOperation`] validates the complete specification
/// against the input types before execution or staging.
///
/// For example, [`Self::matmul`] describes `[M, K] × [G, K, N] → [M, N]`, which partitions `M` into `G` groups and
/// contracts each group with its own `[K, N]` slice of the right-hand side. Refer to [`RaggedDotOperation`] for the
/// semantics of each mode.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RaggedDotDimensionNumbers {
    /// Underlying generalized dot dimension numbers.
    dot_dimensions: DotDimensionNumbers,

    /// Left-hand side axis that the group sizes partition into consecutive groups.
    lhs_ragged_dimension: usize,

    /// Right-hand side axis that indexes groups. Non-contracting mode requires one, and the other modes require none.
    rhs_group_dimension: Option<usize>,
}

impl RaggedDotDimensionNumbers {
    /// Creates a ragged dot dimension number specification.
    #[inline]
    pub fn new(
        dot_dimensions: DotDimensionNumbers,
        lhs_ragged_dimension: usize,
        rhs_group_dimension: Option<usize>,
    ) -> Self {
        Self { dot_dimensions, lhs_ragged_dimension, rhs_group_dimension }
    }

    /// Returns the specification of the basic non-contracting matrix form `[M, K] × [G, K, N] → [M, N]`.
    #[inline]
    pub fn matmul() -> Self {
        Self::new(DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()), 0, Some(0))
    }

    /// Returns the underlying generalized dot dimension numbers.
    #[inline]
    pub fn dot_dimensions(&self) -> &DotDimensionNumbers {
        &self.dot_dimensions
    }

    /// Returns the left-hand side ragged dimension.
    #[inline]
    pub fn lhs_ragged_dimension(&self) -> usize {
        self.lhs_ragged_dimension
    }

    /// Returns the right-hand side group dimension, if any.
    #[inline]
    pub fn rhs_group_dimension(&self) -> Option<usize> {
        self.rhs_group_dimension
    }

    /// Classifies this specification according to the role of its left-hand side ragged dimension for a left-hand
    /// side with rank `lhs_rank`. [`RaggedDotOperation`] validates the remaining dimension number constraints.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] when the ragged dimension is outside `lhs_rank`.
    pub fn mode(&self, lhs_rank: usize) -> Result<RaggedDotMode, TypeError> {
        let axis = self.lhs_ragged_dimension;
        if axis >= lhs_rank {
            return Err(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` left-hand side ragged dimension {axis} is out of bounds for rank \
                 {lhs_rank}",
            )));
        }
        if self.dot_dimensions.lhs_contracting_dimensions().contains(&axis) {
            Ok(RaggedDotMode::Contracting)
        } else if self.dot_dimensions.lhs_batching_dimensions().contains(&axis) {
            Ok(RaggedDotMode::Batch)
        } else {
            Ok(RaggedDotMode::NonContracting)
        }
    }

    /// Returns the left input axes indexing the prefix of a `group_sizes` input for a left-hand side with rank
    /// `lhs_rank`. The prefix contains batching axes and the result or contracting axes preceding the ragged position
    /// in iteration order. A batching ragged axis uses only the batching axes before it. [`RaggedDotOperation`]
    /// validates the right input's indices.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] for an invalid ragged dimension or for invalid left batching or contracting indices.
    pub fn group_sizes_prefix_dimensions(&self, lhs_rank: usize) -> Result<Vec<usize>, TypeError> {
        let mode = self.mode(lhs_rank)?;
        let ragged_axis = self.lhs_ragged_dimension;
        let dimensions = self.dot_dimensions();
        validate_dot_axes(
            RAGGED_DOT_OPERATION_NAME,
            dimensions.lhs_batching_dimensions(),
            dimensions.lhs_contracting_dimensions(),
            lhs_rank,
            "left-hand side",
        )?;
        Ok(match mode {
            RaggedDotMode::NonContracting => {
                let result_axes = dimensions.lhs_result_dimensions(lhs_rank);
                let position = result_axes.iter().position(|axis| *axis == ragged_axis).unwrap();
                dimensions
                    .lhs_batching_dimensions()
                    .iter()
                    .copied()
                    .chain(result_axes[..position].iter().copied())
                    .collect()
            }
            RaggedDotMode::Contracting => {
                let position =
                    dimensions.lhs_contracting_dimensions().iter().position(|axis| *axis == ragged_axis).unwrap();
                dimensions
                    .lhs_batching_dimensions()
                    .iter()
                    .copied()
                    .chain(dimensions.lhs_contracting_dimensions()[..position].iter().copied())
                    .collect()
            }
            RaggedDotMode::Batch => {
                let position =
                    dimensions.lhs_batching_dimensions().iter().position(|axis| *axis == ragged_axis).unwrap();
                dimensions.lhs_batching_dimensions()[..position].to_vec()
            }
        })
    }

    /// Validates this specification against the ranks of the left-hand side and right-hand side inputs before
    /// indexing or lifting, and returns its [`RaggedDotMode`]. The underlying [`DotDimensionNumbers`] must pass
    /// [`DotDimensionNumbers::validate`], and the ragged dimension must be within the left-hand side rank. A
    /// non-contracting specification must have a group dimension within the right-hand side rank that is neither a
    /// batching nor a contracting axis, and the contracting and batching modes must have none.
    ///
    /// Validation covers dimension indices. The operation's type inference additionally validates paired extents,
    /// the group sizes input, element types, and sharding.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{RaggedDotDimensionNumbers, RaggedDotMode};
    /// assert_eq!(RaggedDotDimensionNumbers::matmul().validate(2, 3), Ok(RaggedDotMode::NonContracting));
    /// ```
    ///
    /// # Parameters
    ///
    ///   - `lhs_rank`: Rank of the left-hand side input.
    ///   - `rhs_rank`: Rank of the right-hand side input.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if the underlying dot dimension numbers are invalid, the ragged or group dimension is
    /// out of bounds, the group dimension is a batching or contracting axis, or the presence of the group dimension
    /// does not match the mode.
    pub fn validate(&self, lhs_rank: usize, rhs_rank: usize) -> Result<RaggedDotMode, TypeError> {
        self.dot_dimensions.validate(RAGGED_DOT_OPERATION_NAME, lhs_rank, rhs_rank)?;
        let mode = self.mode(lhs_rank)?;
        match (mode, self.rhs_group_dimension) {
            (RaggedDotMode::NonContracting, None) => {
                return Err(TypeError::invalid(format!(
                    "`{RAGGED_DOT_OPERATION_NAME}` requires a right-hand side group dimension when the left-hand side \
                     ragged dimension is non-contracting",
                )));
            }
            (RaggedDotMode::NonContracting, Some(group_axis)) => {
                if group_axis >= rhs_rank {
                    return Err(TypeError::invalid(format!(
                        "`{RAGGED_DOT_OPERATION_NAME}` right-hand side group dimension {group_axis} is out of bounds \
                         for rank {rhs_rank}",
                    )));
                }
                let dot_dimensions = self.dot_dimensions();
                if dot_dimensions.rhs_batching_dimensions().contains(&group_axis)
                    || dot_dimensions.rhs_contracting_dimensions().contains(&group_axis)
                {
                    return Err(TypeError::invalid(format!(
                        "`{RAGGED_DOT_OPERATION_NAME}` right-hand side group dimension must be distinct from batching \
                         and contracting dimensions",
                    )));
                }
            }
            (RaggedDotMode::Contracting | RaggedDotMode::Batch, Some(_)) => {
                return Err(TypeError::invalid(format!(
                    "`{RAGGED_DOT_OPERATION_NAME}` does not accept a right-hand side group dimension when the \
                     left-hand side ragged dimension is contracting or batching",
                )));
            }
            (RaggedDotMode::Contracting | RaggedDotMode::Batch, None) => {}
        }
        Ok(mode)
    }

    /// Returns the ragged dot dimension numbers that map the output cotangent `ȳ` of `y = ragged_dot(a, b, g)` to the
    /// cotangent `ā` of its linear left-hand side `a`, together with the axes of `a` in the order in which that adjoint
    /// ragged dot emits them. This specification must be a valid non-contracting specification for a left-hand side
    /// with rank `lhs_rank` and a right-hand side with rank `rhs_rank`.
    ///
    /// The adjoint computes `ā` as a non-contracting ragged dot of `ȳ` with `b`, whose ragged dimension is the ragged
    /// output dimension. It emits the batching axes, the left-hand side result axes, and then the left-hand side
    /// contracting axes sorted by their paired right-hand side axes. Callers must restore the original axis order of
    /// `a` with the inverse permutation.
    fn lhs_cotangent_dimensions(&self, lhs_rank: usize, rhs_rank: usize) -> (Self, Vec<usize>) {
        let dot_dimensions = self.dot_dimensions();
        let lhs_result_dimensions = dot_dimensions.lhs_result_dimensions(lhs_rank);
        let rhs_result_dimensions = dot_dimensions
            .rhs_result_dimensions(rhs_rank)
            .into_iter()
            .filter(|axis| self.rhs_group_dimension != Some(*axis))
            .collect::<Vec<_>>();
        let batching_count = dot_dimensions.lhs_batching_dimensions().len();
        let cotangent_rhs_result_start = batching_count + lhs_result_dimensions.len();
        let ragged_output_axis =
            batching_count + lhs_result_dimensions.iter().position(|axis| *axis == self.lhs_ragged_dimension).unwrap();
        let dimensions = Self::new(
            DotDimensionNumbers::new(
                (cotangent_rhs_result_start..(cotangent_rhs_result_start + rhs_result_dimensions.len())).collect(),
                rhs_result_dimensions,
                (0..batching_count).collect(),
                dot_dimensions.rhs_batching_dimensions().to_vec(),
            ),
            ragged_output_axis,
            self.rhs_group_dimension,
        );
        let mut contracting_pairs = dot_dimensions
            .lhs_contracting_dimensions()
            .iter()
            .copied()
            .zip(dot_dimensions.rhs_contracting_dimensions().iter().copied())
            .collect::<Vec<_>>();
        contracting_pairs.sort_by_key(|(_, rhs_axis)| *rhs_axis);
        let output_axes = dot_dimensions
            .lhs_batching_dimensions()
            .iter()
            .copied()
            .chain(lhs_result_dimensions)
            .chain(contracting_pairs.into_iter().map(|(lhs_axis, _)| lhs_axis))
            .collect();
        (dimensions, output_axes)
    }

    /// Returns the ragged dot dimension numbers that map the output cotangent `ȳ` of `y = ragged_dot(a, b, g)` to
    /// the cotangent `b̄` of its linear right-hand side `b`, together with the axes of `b` in the order in which that
    /// adjoint ragged dot emits them. This specification must be a valid non-contracting specification for a
    /// left-hand side with rank `lhs_rank` and a right-hand side with rank `rhs_rank`.
    ///
    /// The adjoint computes `b̄` as a contracting ragged dot of `a` with `ȳ` over the ragged dimension, so it emits
    /// the group axis, the batching axes, the right-hand side contracting axes sorted by their paired left-hand side
    /// axes, and then the right-hand side result axes. Callers must restore the original axis order of `b` with the
    /// inverse permutation.
    fn rhs_cotangent_dimensions(&self, lhs_rank: usize, rhs_rank: usize) -> (Self, Vec<usize>) {
        let dot_dimensions = self.dot_dimensions();
        let lhs_result_dimensions = dot_dimensions.lhs_result_dimensions(lhs_rank);
        let rhs_result_dimensions = dot_dimensions
            .rhs_result_dimensions(rhs_rank)
            .into_iter()
            .filter(|axis| self.rhs_group_dimension != Some(*axis))
            .collect::<Vec<_>>();
        let batching_count = dot_dimensions.lhs_batching_dimensions().len();
        let cotangent_lhs_result_count = lhs_result_dimensions.len();
        let dimensions = Self::new(
            DotDimensionNumbers::new(
                lhs_result_dimensions,
                (batching_count..(batching_count + cotangent_lhs_result_count)).collect(),
                dot_dimensions.lhs_batching_dimensions().to_vec(),
                (0..batching_count).collect(),
            ),
            self.lhs_ragged_dimension,
            None,
        );
        let mut contracting_pairs = dot_dimensions
            .rhs_contracting_dimensions()
            .iter()
            .copied()
            .zip(dot_dimensions.lhs_contracting_dimensions().iter().copied())
            .collect::<Vec<_>>();
        contracting_pairs.sort_by_key(|(_, lhs_axis)| *lhs_axis);
        let output_axes = self
            .rhs_group_dimension
            .into_iter()
            .chain(dot_dimensions.rhs_batching_dimensions().iter().copied())
            .chain(contracting_pairs.into_iter().map(|(rhs_axis, _)| rhs_axis))
            .chain(rhs_result_dimensions)
            .collect();
        (dimensions, output_axes)
    }
}

impl Display for RaggedDotDimensionNumbers {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The group dimension is rendered only when present, like the optional fields of operation renderings.
        write!(
            formatter,
            "(lhs_contracting={:?}, rhs_contracting={:?}, lhs_batching={:?}, rhs_batching={:?}, lhs_ragged={}",
            self.dot_dimensions.lhs_contracting_dimensions,
            self.dot_dimensions.rhs_contracting_dimensions,
            self.dot_dimensions.lhs_batching_dimensions,
            self.dot_dimensions.rhs_batching_dimensions,
            self.lhs_ragged_dimension,
        )?;
        if let Some(rhs_group_dimension) = self.rhs_group_dimension {
            write!(formatter, ", rhs_group={rhs_group_dimension}")?;
        }
        formatter.write_str(")")
    }
}

/// Canonical operation name for [`RaggedDotOperation`].
pub const RAGGED_DOT_OPERATION_NAME: &str = "ragged_dot";

/// Operation type representing a ragged dot product with explicit group sizes, which is also known as a grouped matrix
/// multiplication (or grouped GEMM) in mixture-of-experts models. Exactly one left-hand side dimension is _ragged_. Its
/// role selects one of three modes:
///
///   - A non-contracting ragged dimension is partitioned into consecutive groups. The corresponding right-hand side
///     group dimension selects one right-hand side slice per group, and the grouped products are written back along the
///     left-hand side result dimension. An empty group contributes nothing and any uncovered suffix of that dimension
///     is zero.
///   - A contracting ragged dimension partitions the paired contracting dimensions into consecutive groups.
///     The output gains a leading group dimension, and an empty group produces a zero slice.
///   - A batching ragged dimension has ordinary batched dot semantics. `group_sizes` participates in type inference
///     but its values do not affect the result.
///
/// `group_sizes` is either a rank-one `[group_count]` array shared by every prefix or an array whose trailing axis is
/// `group_count` and whose prefix matches the dimensions preceding the ragged position in the ragged dot iteration
/// space.
///
/// In non-contracting and contracting modes every size must be non-negative. The eager interpreter rejects negative
/// metadata in those modes. The decomposition lowering of the XLA backend additionally clamps signed negatives to zero
/// before unsigned accumulation, so invalid runtime metadata cannot become a large interval. The instruction lowering
/// passes metadata unchanged to `chlo.ragged_dot` and therefore relies on the contract that sizes are non-negative.
///
/// The sizes define consecutive raw cumulative intervals. Each interval is intersected with the physical left-hand side
/// ragged extent, so a group that extends past that extent is clipped, and every later group is empty once its raw
/// start reaches or exceeds that extent.
///
/// Grouped expansion modes require an element type that can represent zero. In particular, they reject `f8e8m0fnu`.
/// Refer to [`RaggedDotDimensionNumbers`] for the dimension number contract.
///
/// The operation is linear in either data input separately, while the group sizes are nondifferentiable metadata.
/// For `y = ragged_dot(a, b, g)` with group sizes `g`, forward mode differentiation therefore computes
/// `ẏ = ragged_dot(ȧ, b, g) + ragged_dot(a, ḃ, g)`. Transposition is defined only in non-contracting mode, and it
/// computes `ā` or `b̄` from the output cotangent `ȳ` with another ragged dot followed by the inverse permutation of
/// its output axes.
///
/// Batching accepts any mix of replicated and mapped inputs with mapped axes at any position. It materializes a
/// replicated data input across the mapped extent and moves every mapped axis to the front, so the mapped axes become
/// a new leading batching pair, and `group_sizes` remains the sole source of the group extents of every item.
/// Replicated rank-one `group_sizes` stay shared by every item, while mapped rank-one `group_sizes` are broadcast
/// across the remaining `group_sizes` prefix of the lifted operation. Unlike JAX, which requires every input to be
/// mapped over axis zero, this also covers replicated inputs and non-leading mapped axes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RaggedDotOperation {
    /// Ragged dot dimension number specification.
    dimensions: RaggedDotDimensionNumbers,
}

impl RaggedDotOperation {
    /// Creates a new [`RaggedDotOperation`].
    #[inline]
    pub fn new(dimensions: RaggedDotDimensionNumbers) -> Self {
        Self { dimensions }
    }

    /// Returns the [`RaggedDotDimensionNumbers`] of this [`RaggedDotOperation`].
    #[inline]
    pub fn dimensions(&self) -> &RaggedDotDimensionNumbers {
        &self.dimensions
    }
}

impl Display for RaggedDotOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for RaggedDotOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        RAGGED_DOT_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 3, TypeError);

        let lhs = &input_types[0];
        let rhs = &input_types[1];
        let group_sizes = &input_types[2];
        let dimensions = &self.dimensions;

        ArrayType::check_matching_manual_variation(RAGGED_DOT_OPERATION_NAME, &[lhs, rhs, group_sizes])?;

        if !group_sizes.data_type().is_integer() {
            return Err(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` group sizes must have an integer data type",
            )));
        }

        if group_sizes.rank() == 0 {
            return Err(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` group sizes must have rank at least one",
            )));
        }

        let mode = dimensions.validate(lhs.rank(), rhs.rank())?;
        let mut output =
            infer_dot_output_type(RAGGED_DOT_OPERATION_NAME, lhs, rhs, dimensions.dot_dimensions(), None, None)?;
        let prefix_axes = dimensions.group_sizes_prefix_dimensions(lhs.rank())?;
        if group_sizes.rank() != 1 {
            let expected_prefix = Shape::new(prefix_axes.iter().map(|axis| lhs.dimension(*axis)).collect());
            let actual_prefix = Shape::new(group_sizes.shape().dimensions()[..group_sizes.rank() - 1].to_vec());
            if actual_prefix != expected_prefix {
                return Err(TypeError::invalid(format!(
                    "`{RAGGED_DOT_OPERATION_NAME}` group sizes prefix must be `{expected_prefix}`, \
                     but got `{actual_prefix}`",
                )));
            }
        }

        let group_count = group_sizes.dimension(group_sizes.rank() - 1);
        if let Some(group_axis) = dimensions.rhs_group_dimension()
            && rhs.dimension(group_axis) != group_count
        {
            return Err(TypeError::invalid(format!(
                "`{}` right-hand side group dimension has extent {} but group sizes describe {}",
                RAGGED_DOT_OPERATION_NAME,
                rhs.dimension(group_axis),
                group_count,
            )));
        }

        if mode != RaggedDotMode::Batch && lhs.data_type() == DataType::F8E8M0FNU {
            return Err(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` does not support element data type `f8e8m0fnu` in grouped expansion \
                 modes because it cannot represent zero",
            )));
        }

        match mode {
            RaggedDotMode::NonContracting => {
                let rhs_result = dimensions.dot_dimensions().rhs_result_dimensions(rhs.rank());
                let group_position =
                    rhs_result.iter().position(|axis| Some(*axis) == dimensions.rhs_group_dimension()).unwrap();
                let output_axis = dimensions.dot_dimensions().lhs_batching_dimensions().len()
                    + dimensions.dot_dimensions().lhs_result_dimensions(lhs.rank()).len()
                    + group_position;
                output = output.without_dimension(output_axis)?.0;
            }
            RaggedDotMode::Contracting => output = output.with_inserted_dimension(0, group_count)?,
            RaggedDotMode::Batch => {}
        }

        Ok(vec![output])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("dimensions", &self.dimensions))
    }
}

impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for RaggedDotOperation
where
    C::Value: RaggedDot,
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 3, ProgramError);
        Ok(vec![inputs[0].ragged_dot(&inputs[1], &inputs[2], &self.dimensions)?])
    }
}

impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for RaggedDotOperation where
    C::Operation: From<RaggedDotOperation>
{
}

impl<C: Context<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>>
    for RaggedDotOperation
where
    RaggedDotOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        // Alignment delegates mapped axis materialization to `ArrayExtentBatchingPolicy::match_axis` and
        // `ArrayExtentBatchingPolicy::broadcast_input`, so a dynamic mapped extent remains a first-class value.
        // The mapped axes become a new leading batching pair, and the explicit `group_sizes` keep specifying the
        // groups of every item.
        check_count!("input", inputs, 3, ProgramError);
        let mode = self.dimensions.validate(inputs[0].unbatched_type().rank(), inputs[1].unbatched_type().rank())?;
        if inputs.iter().all(|input| input.batch_axis_position().is_none()) {
            return self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()]);
        }

        validate_common_mapped_dimension(RAGGED_DOT_OPERATION_NAME, inputs)?;

        // Both data inputs need the mapped axis at position zero before the logical dimension numbers can be lifted,
        // so a replicated data input is materialized across the mapped extent and a mapped one moves its axis there.
        let lhs = P::match_axis(context, &inputs[0], Axis::from(0))?;
        let rhs = P::match_axis(context, &inputs[1], Axis::from(0))?;
        let (dot_dimensions, _) = self.dimensions.dot_dimensions().batched(Some(0), Some(0)).unwrap();
        let dimensions = RaggedDotDimensionNumbers::new(
            dot_dimensions,
            self.dimensions.lhs_ragged_dimension() + 1,
            self.dimensions.rhs_group_dimension().map(|axis| axis + 1),
        );

        // Replicated rank-one group sizes are shared by every prefix of the lifted operation as well, so they need no
        // materialization. Other group sizes need the mapped axis leading their prefix, and mapped rank-one group sizes
        // are additionally broadcast across the remaining prefix of the lifted operation.
        let group_sizes = &inputs[2];
        let group_sizes = match group_sizes.batch_axis_position() {
            None if group_sizes.r#type().rank() == 1 => group_sizes.clone(),
            None => P::match_axis(context, group_sizes, Axis::from(0))?,
            Some(_) => {
                let group_sizes = P::match_axis(context, group_sizes, Axis::from(0))?;
                let prefix_axes = dimensions.group_sizes_prefix_dimensions(lhs.r#type().rank())?;
                if inputs[2].unbatched_type().rank() == 1 && prefix_axes.len() > 1 {
                    // Broadcast mapped rank-one metadata `[batch, group_count]` across the remaining prefix, so
                    // every item shares its group sizes there. The policy's extent value grounds the leading mapped
                    // axis, and dynamic prefix dimensions are read from the aligned left input itself.
                    let lhs_type = lhs.r#type();
                    let group_sizes_type = group_sizes.r#type();
                    let dimension_source = |source: &C::Value, dimension: Dimension, axis: usize| match dimension {
                        Dimension::Static(extent) => DimensionSource::Static(extent),
                        _ => DimensionSource::Value { source: source.clone(), axis },
                    };
                    let dimension_sources = prefix_axes
                        .iter()
                        .map(|axis| match axis {
                            0 => DimensionSource::BatchExtent,
                            _ => dimension_source(lhs.value(), lhs_type.dimension(*axis), *axis),
                        })
                        .chain(std::iter::once(dimension_source(group_sizes.value(), group_sizes_type.dimension(1), 1)))
                        .collect::<Vec<_>>();

                    // The group sizes keep their memory space and manual variation, while each new prefix axis
                    // is replicated.
                    let output_rank = prefix_axes.len() + 1;
                    let output_axes = vec![0, prefix_axes.len()];
                    let output_sharding = group_sizes_type
                        .sharding()
                        .map(|sharding| sharding.with_broadcasted_dimensions(output_rank, &output_axes))
                        .transpose()
                        .map_err(TypeError::from)?;
                    let output_shape = Shape::new(
                        prefix_axes
                            .iter()
                            .map(|axis| lhs_type.dimension(*axis))
                            .chain(std::iter::once(group_sizes_type.dimension(1)))
                            .collect(),
                    );
                    let output_type = ArrayType::new(group_sizes_type.data_type(), output_shape)
                        .with_memory(group_sizes_type.memory())
                        .with_sharding(output_sharding)
                        .map_err(TypeError::from)?;
                    P::broadcast_input(
                        context,
                        &group_sizes,
                        output_type,
                        output_axes,
                        Axis::from(0),
                        dimension_sources,
                    )?
                } else {
                    group_sizes
                }
            }
        };

        // A contracting ragged dimension keeps its leading group output axis ahead of the batch axis.
        let output_axis = if mode == RaggedDotMode::Contracting { 1 } else { 0 };

        RaggedDotOperation::new(dimensions).interpret_with_batch_axes(
            context,
            &[lhs, rhs, group_sizes],
            &[BatchAxis::from_position(output_axis)],
        )
    }
}

impl_differentiable_operation! {
    RaggedDotOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Operation: From<RaggedDotOperation>,
        C::Value: ConvertElementType + RaggedDot + Add,
    {
        |operation, context, _driver, inputs| {
            // A ragged dot is linear in either data input separately. Group sizes are integer metadata and therefore
            // carry no tangent. The two product rule terms reuse the same ragged dot dimensions.
            check_count!("input", inputs, 3, ProgramError);
            let lhs = &inputs[0];
            let rhs = &inputs[1];
            let group_sizes = inputs[2].primal();
            let primal = lhs.primal().ragged_dot(rhs.primal(), group_sizes, operation.dimensions())?;
            if lhs.tangent().is_zero() && rhs.tangent().is_zero() {
                return Ok(vec![DifferentiationDual::new_with_zero_tangent(primal)?]);
            }
            let group_sizes = context.primal_to_tangent(group_sizes.clone())?;
            let apply = |lhs: &C::Value, rhs: &C::Value| {
                lhs.ragged_dot(rhs, &group_sizes, operation.dimensions()).map_err(DifferentiationError::from)
            };
            Ok(vec![bilinear_array_jvp(context, primal, lhs, rhs, apply)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: From<ConvertElementTypeOperation<ArrayType>> + From<RaggedDotOperation> + From<TransposeOperation>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // Ragged dot transposition is defined only for the non-contracting mode. Each data input adjoint is
            // another ragged dot followed by the inverse permutation of the axis order that its adjoint dimension
            // numbers produce.
            check_count!("input", inputs, 3, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 3, DifferentiationError);
            operation.infer_output_types(
                &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
                &[],
            )?;
            let mode = operation.dimensions().mode(inputs[0].r#type().rank())?;

            if mode != RaggedDotMode::NonContracting {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!("`{RAGGED_DOT_OPERATION_NAME}` transposition is unsupported in `{mode}` mode"),
                }
                .into());
            }

            let group_sizes = inputs[2].as_known().ok_or_else(|| ProgramError::UnsupportedOperation {
                message: format!("`{RAGGED_DOT_OPERATION_NAME}` group sizes must be known during transposition"),
            })?;

            let linear_index = match (inputs[0].is_unknown(), inputs[1].is_unknown()) {
                (true, true) => {
                    // Both data inputs linear is a bilinear product, which is not a linear map in both inputs jointly
                    // and so never appears in a valid pushforward.
                    return Err(ProgramError::UnsupportedOperation {
                        message: format!(
                            "bilinear `{RAGGED_DOT_OPERATION_NAME}` with two linear inputs cannot be transposed",
                        ),
                    }
                    .into());
                }
                (false, false) => return Ok(()),
                (true, false) => 0,
                (false, true) => 1,
            };

            // The known group metadata contract is retained, but the adjoint kernel is omitted when its value is not
            // requested. A zero output cotangent stays a structural zero.
            if !accumulators[linear_index].is_needed() {
                return Ok(());
            }

            let MaybeZero::Value(cotangent) = &outputs[0] else {
                return Ok(());
            };

            let left_rank = inputs[0].r#type().rank();
            let right_rank = inputs[1].r#type().rank();
            let (dimensions, output_axes, mut adjoint_inputs) = if linear_index == 0 {
                let (dimensions, output_axes) = operation.dimensions().lhs_cotangent_dimensions(left_rank, right_rank);
                let known_rhs = inputs[1].as_known().unwrap();
                (dimensions, output_axes, [cotangent.clone(), known_rhs.clone(), group_sizes.clone()])
            } else {
                let (dimensions, output_axes) = operation.dimensions().rhs_cotangent_dimensions(left_rank, right_rank);
                let known_lhs = inputs[0].as_known().unwrap();
                (dimensions, output_axes, [known_lhs.clone(), cotangent.clone(), group_sizes.clone()])
            };

            // An adjoint over widened tangents contracts both data inputs at the cotangent element type,
            // and the result is converted back to the linear input's cotangent element type.
            let linear_cotangent_type = inputs[linear_index].r#type().cotangent()?;
            for adjoint_input in &mut adjoint_inputs[..2] {
                if adjoint_input.r#type().data_type() != cotangent.r#type().data_type() {
                    *adjoint_input = adjoint_input.convert_element_type(cotangent.r#type().data_type())?;
                }
            }
            let mut adjoint_outputs =
                context.stage_operation(RaggedDotOperation::new(dimensions), Vec::new(), &adjoint_inputs)?;
            check_count!("output", adjoint_outputs, 1, ProgramError);

            let adjoint_value = adjoint_outputs.remove(0).transpose(Permutation::from(output_axes).inverse()?)?;
            let adjoint_value = if adjoint_value.r#type().data_type() == linear_cotangent_type.data_type() {
                adjoint_value
            } else {
                adjoint_value.convert_element_type(linear_cotangent_type.data_type())?
            };

            accumulators[linear_index].accumulate(context, MaybeZero::Value(adjoint_value))
        }
    },
}

/// Value-level ragged dot product capability. [`RaggedDot`] is the receiver-style entry point for staging or executing
/// [`RaggedDotOperation`], which contracts two data inputs group by group according to explicit integer `group_sizes`.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and array IR values implement it for [`ArrayIrType`].
#[capability(projection(ArrayIrType => ArrayType))]
pub trait RaggedDot<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Computes the ragged dot product of `self` and `rhs` using the explicit `group_sizes` and `dimensions`, and
    /// returns a [`ProgramError`] if the inputs are incompatible with `dimensions` or the contraction cannot be
    /// recorded in the value's context. Use [`RaggedDotDimensionNumbers::matmul`] for the basic non-contracting form
    /// `[M, K] × [G, K, N] → [M, N]`. Refer to the documentation of [`RaggedDotOperation`] for information on the
    /// three modes, the metadata shapes, the clipping of cumulative intervals, and the semantics of empty groups and
    /// uncovered positions.
    fn ragged_dot(
        &self,
        rhs: &Self,
        group_sizes: &Self,
        dimensions: &RaggedDotDimensionNumbers,
    ) -> Result<Self, ProgramError>;
}

impl RaggedDot for Array {
    #[inline]
    fn ragged_dot(
        &self,
        rhs: &Self,
        group_sizes: &Self,
        dimensions: &RaggedDotDimensionNumbers,
    ) -> Result<Self, ProgramError> {
        let data_type = self.r#type().data_type();
        validate_dot_data_type(RAGGED_DOT_OPERATION_NAME, data_type)?;
        dispatch_on_array_element_type!(
            @numeric data_type,
            |Element| self.ragged_dot_elements::<Element>(rhs, group_sizes, dimensions),
        )
    }
}

impl<V: Value<Type = ArrayType, Dispatch = ValueDomainDispatch> + ManualVariationAlignment<ArrayType>>
    RaggedDot<ArrayType> for V
where
    V::Domain: Context<Type = ArrayType>,
    <V::Domain as Domain>::Operation: From<RaggedDotOperation>,
{
    fn ragged_dot(
        &self,
        rhs: &Self,
        group_sizes: &Self,
        dimensions: &RaggedDotDimensionNumbers,
    ) -> Result<Self, ProgramError> {
        // Values that carry a context stage a ragged dot through that context. The `ValueDomainDispatch` marker keeps
        // this implementation disjoint from eager values, which implement the capability directly.
        let inputs = [self.clone(), rhs.clone(), group_sizes.clone()];
        let inputs = ManualVariationAlignment::align_manual_variation(&inputs)?;
        let mut outputs = self.domain().bind(RaggedDotOperation::new(dimensions.clone()), Vec::new(), &inputs)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Group of the dot product capabilities [`Dot`] and [`RaggedDot`]. It is implemented automatically for every type that
/// implements all of its members.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this bundle for [`ArrayType`] and array IR values implement it for [`ArrayIrType`].
pub trait DotOperations<T = <Self as Capability>::Universe>: Capability + Dot<T> + RaggedDot<T> {}

impl<T, V: Dot<T> + RaggedDot<T>> DotOperations<T> for V {}

impl Array {
    /// Contracts typed elements using each input's physical layout and the declared contraction dimensions. The result
    /// type comes from the type inference of the corresponding [`DotOperation`], which uses `output_sharding` (when
    /// provided) instead of the inferred output sharding. Floating-point elements narrower than `f32` accumulate at
    /// `f32` and are rounded once to `T`.
    fn dot_elements<T: NumericArrayElement>(
        &self,
        rhs: &Self,
        dimensions: &DotDimensionNumbers,
        output_sharding: Option<&Sharding>,
    ) -> Result<Self, ProgramError> {
        let mut output_types = DotOperation::new(dimensions.clone())
            .with_output_sharding(output_sharding.cloned())
            .infer_output_types(&[self.r#type().into_owned(), rhs.r#type().into_owned()], &[])?;
        let output_type = output_types.remove(0);

        // An empty output or an empty contraction never evaluates a product, and the latter materializes one additive
        // identity for every result coordinate.
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        if output_addressing.element_count() == 0
            || dimensions
                .lhs_contracting_dimensions()
                .iter()
                .any(|axis| self.r#type().dimension(*axis).value() == Some(0))
        {
            return Self::from_fn_elements(output_type, |_| Ok(T::zero()?));
        }

        // Floating-point elements narrower than `f32` accumulate at `f32` and round once to their own element type,
        // which is how XLA backends evaluate such dots when no accumulation type is requested.
        if T::data_type().is_floating_point() && T::data_type().bit_width() < 32 {
            let lhs = self.convert_element_type(DataType::F32)?;
            let rhs = rhs.convert_element_type(DataType::F32)?;
            return lhs.dot_elements::<f32>(&rhs, dimensions, output_sharding)?.convert_element_type(T::data_type());
        }

        debug_assert_eq!(self.r#type().data_type(), T::data_type());
        debug_assert_eq!(rhs.r#type().data_type(), T::data_type());
        let lhs_shape = self.r#type().static_shape().unwrap();
        let rhs_shape = rhs.r#type().static_shape().unwrap();
        let output_shape = output_type.static_shape().unwrap();
        let output_strides = output_shape.row_major_strides();
        let lhs_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let rhs_addressing = ArrayAddressing::new(rhs.r#type().into_owned())?;
        let lhs_batching = dimensions.lhs_batching_dimensions();
        let rhs_batching = dimensions.rhs_batching_dimensions();
        let lhs_contracting = dimensions.lhs_contracting_dimensions();
        let rhs_contracting = dimensions.rhs_contracting_dimensions();
        let lhs_result = dimensions.lhs_result_dimensions(lhs_shape.rank());
        let rhs_result = dimensions.rhs_result_dimensions(rhs_shape.rank());
        let contracting_shape =
            StaticShape::new(lhs_contracting.iter().map(|axis| lhs_shape[*axis]).collect::<Vec<_>>());
        let contracting_strides = contracting_shape.row_major_strides();
        let contracting_count = contracting_shape.dimensions().iter().product();

        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        let mut lhs_index = vec![0usize; lhs_shape.rank()];
        let mut rhs_index = vec![0usize; rhs_shape.rank()];
        for output_flat in 0..output_addressing.element_count() {
            // Decode the result coordinate directly into the corresponding batch and non-contracting input axes.
            let mut output_axis = 0usize;
            for (&lhs_axis, &rhs_axis) in lhs_batching.iter().zip(rhs_batching) {
                let coordinate = (output_flat / output_strides[output_axis]) % output_shape[output_axis];
                lhs_index[lhs_axis] = coordinate;
                rhs_index[rhs_axis] = coordinate;
                output_axis += 1;
            }

            for &lhs_axis in &lhs_result {
                lhs_index[lhs_axis] = (output_flat / output_strides[output_axis]) % output_shape[output_axis];
                output_axis += 1;
            }

            for &rhs_axis in &rhs_result {
                rhs_index[rhs_axis] = (output_flat / output_strides[output_axis]) % output_shape[output_axis];
                output_axis += 1;
            }

            let mut accumulator = T::zero()?;
            for contracting_flat in 0..contracting_count {
                for (contracting_axis, (&lhs_axis, &rhs_axis)) in
                    lhs_contracting.iter().zip(rhs_contracting).enumerate()
                {
                    let coordinate = (contracting_flat / contracting_strides[contracting_axis])
                        % contracting_shape[contracting_axis];
                    lhs_index[lhs_axis] = coordinate;
                    rhs_index[rhs_axis] = coordinate;
                }
                let lhs_value = T::decode(&self.storage_bytes()[lhs_addressing.byte_range_unchecked(&lhs_index)]);
                let rhs_value = T::decode(&rhs.storage_bytes()[rhs_addressing.byte_range_unchecked(&rhs_index)]);
                let product = lhs_value.mul(rhs_value)?;
                accumulator = accumulator.add(product)?;
            }
            accumulator.encode(&mut bytes[output_addressing.byte_range_for_flat_index(output_flat)]);
        }

        Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
    }

    /// Evaluates a ragged dot over the exact physical extents. Each concrete group's raw cumulative interval is clipped
    /// to the physical ragged extent, the resulting pair of left-hand side and right-hand side slices is contracted by
    /// the ordinary generalized dot kernel, and the result is written into its output window. This keeps temporary
    /// storage proportional to one group rather than to a whole input times the group count.
    fn ragged_dot_elements<T: NumericArrayElement>(
        &self,
        rhs: &Self,
        group_sizes: &Self,
        dimensions: &RaggedDotDimensionNumbers,
    ) -> Result<Self, ProgramError> {
        let mut output_types = RaggedDotOperation::new(dimensions.clone()).infer_output_types(
            &[self.r#type().into_owned(), rhs.r#type().into_owned(), group_sizes.r#type().into_owned()],
            &[],
        )?;
        let output_type = output_types.remove(0);
        let dot_dimensions = dimensions.dot_dimensions();
        let ragged_axis = dimensions.lhs_ragged_dimension();

        let mode = dimensions.mode(self.r#type().rank())?;
        if mode == RaggedDotMode::Batch {
            return self.dot_elements::<T>(rhs, dot_dimensions, None);
        }

        let prefix_axes = dimensions.group_sizes_prefix_dimensions(self.r#type().rank())?;
        let prefix_shape = prefix_axes
            .iter()
            .map(|axis| self.r#type().shape().dimensions()[*axis].value().unwrap())
            .collect::<Vec<_>>();
        let group_count = group_sizes.r#type().shape().dimensions().last().unwrap().value().ok_or_else(|| {
            ProgramError::InvalidArgument {
                message: format!("`{RAGGED_DOT_OPERATION_NAME}` requires a static group count for eager evaluation"),
            }
        })?;

        let sizes = group_sizes.non_negative_integer_elements("group_sizes")?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        if output_addressing.element_count() == 0 {
            return Ok(Self::new_unchecked(output_type, Arc::new(Vec::new())));
        }

        let prefix_count = prefix_shape
            .iter()
            .try_fold(1usize, |count, extent| count.checked_mul(*extent))
            .ok_or_else(|| ProgramError::InvalidArgument {
                message: format!(
                    "`{RAGGED_DOT_OPERATION_NAME}` group sizes prefix element count does not fit in `usize`",
                ),
            })?;

        let expected_size_count = if group_sizes.r#type().rank() == 1 {
            group_count
        } else {
            prefix_count.checked_mul(group_count).ok_or_else(|| ProgramError::InvalidArgument {
                message: format!("`{RAGGED_DOT_OPERATION_NAME}` group sizes element count does not fit in `usize`"),
            })?
        };
        if sizes.len() != expected_size_count {
            return Err(ProgramError::InvalidArgument {
                message: format!("`{RAGGED_DOT_OPERATION_NAME}` group sizes storage does not match its shape"),
            });
        }

        let ragged_extent = self.r#type().shape().dimensions()[ragged_axis].value().unwrap();
        let lhs_shape = self.r#type().static_shape().unwrap();
        let rhs_shape = rhs.r#type().static_shape().unwrap();
        let lhs_strides = vec![1; lhs_shape.rank()];
        let rhs_strides = vec![1; rhs_shape.rank()];
        let output_strides = vec![1; output_type.rank()];
        let lhs_result = dot_dimensions.lhs_result_dimensions(self.r#type().rank());

        // Non-contracting groups contract one RHS group slice, which drops the group axis and keeps a unit extent
        // on every batching axis that belongs to the group sizes prefix.
        let non_contracting_metadata = (mode == RaggedDotMode::NonContracting).then(|| {
            let rhs_group_axis = dimensions.rhs_group_dimension().unwrap();
            let rhs_slice_shape = Shape::new(
                rhs_shape
                    .dimensions()
                    .iter()
                    .enumerate()
                    .filter_map(|(axis, dimension)| {
                        (axis != rhs_group_axis).then(|| {
                            let is_prefix_axis = dot_dimensions
                                .lhs_batching_dimensions()
                                .iter()
                                .zip(dot_dimensions.rhs_batching_dimensions())
                                .any(|(lhs_axis, rhs_axis)| *rhs_axis == axis && prefix_axes.contains(lhs_axis));
                            Dimension::Static(if is_prefix_axis { 1 } else { *dimension })
                        })
                    })
                    .collect(),
            );
            let remap_rhs_axis = |axis: usize| if axis < rhs_group_axis { axis } else { axis - 1 };
            let dense_dimensions = DotDimensionNumbers::new(
                dot_dimensions.lhs_contracting_dimensions().to_vec(),
                dot_dimensions.rhs_contracting_dimensions().iter().map(|axis| remap_rhs_axis(*axis)).collect(),
                dot_dimensions.lhs_batching_dimensions().to_vec(),
                dot_dimensions.rhs_batching_dimensions().iter().map(|axis| remap_rhs_axis(*axis)).collect(),
            );
            let ragged_position = lhs_result.iter().position(|axis| *axis == ragged_axis).unwrap();
            let ragged_output_axis = dot_dimensions.lhs_batching_dimensions().len() + ragged_position;
            (rhs_group_axis, rhs_slice_shape, dense_dimensions, ragged_output_axis)
        });

        let contracting_rhs_ragged_axis = (mode == RaggedDotMode::Contracting).then(|| {
            let contracting_position =
                dot_dimensions.lhs_contracting_dimensions().iter().position(|axis| *axis == ragged_axis).unwrap();
            dot_dimensions.rhs_contracting_dimensions()[contracting_position]
        });

        let mut output = Self::from_fn_elements(output_type, |_| Ok(T::zero()?))?;
        for prefix in 0..prefix_count {
            // Decode the group sizes prefix coordinate and restrict both inputs and the output window to it.
            let mut remainder = prefix;
            let mut prefix_coordinates = vec![0; prefix_axes.len()];
            for (coordinate, extent) in prefix_coordinates.iter_mut().zip(prefix_shape.iter()).rev() {
                *coordinate = remainder % extent;
                remainder /= extent;
            }

            let metadata_prefix = if group_sizes.r#type().rank() == 1 { 0 } else { prefix };
            let group_range = metadata_prefix * group_count..(metadata_prefix + 1) * group_count;
            let mut lhs_starts = vec![0; lhs_shape.rank()];
            let mut lhs_limits = lhs_shape.dimensions().to_vec();
            for (&axis, &coordinate) in prefix_axes.iter().zip(prefix_coordinates.iter()) {
                lhs_starts[axis] = coordinate;
                lhs_limits[axis] = coordinate + 1;
            }

            let mut rhs_starts = vec![0; rhs_shape.rank()];
            let mut rhs_limits = rhs_shape.dimensions().to_vec();
            for (&lhs_axis, &rhs_axis) in
                dot_dimensions.lhs_batching_dimensions().iter().zip(dot_dimensions.rhs_batching_dimensions())
            {
                if let Some(prefix_position) = prefix_axes.iter().position(|axis| *axis == lhs_axis) {
                    let coordinate = prefix_coordinates[prefix_position];
                    rhs_starts[rhs_axis] = coordinate;
                    rhs_limits[rhs_axis] = coordinate + 1;
                }
            }

            if mode == RaggedDotMode::Contracting {
                for (&lhs_axis, &rhs_axis) in
                    dot_dimensions.lhs_contracting_dimensions().iter().zip(dot_dimensions.rhs_contracting_dimensions())
                {
                    if let Some(prefix_position) = prefix_axes.iter().position(|axis| *axis == lhs_axis) {
                        let coordinate = prefix_coordinates[prefix_position];
                        rhs_starts[rhs_axis] = coordinate;
                        rhs_limits[rhs_axis] = coordinate + 1;
                    }
                }
            }

            let mut output_starts = vec![0; output.r#type().rank()];
            let mut output_limits = output.r#type().static_shape().unwrap().dimensions().to_vec();
            match mode {
                RaggedDotMode::NonContracting => {
                    for (&axis, &coordinate) in prefix_axes.iter().zip(prefix_coordinates.iter()) {
                        if let Some(position) =
                            dot_dimensions.lhs_batching_dimensions().iter().position(|candidate| *candidate == axis)
                        {
                            output_starts[position] = coordinate;
                        } else {
                            let position = lhs_result.iter().position(|candidate| *candidate == axis).unwrap();
                            let position = dot_dimensions.lhs_batching_dimensions().len() + position;
                            output_starts[position] = coordinate;
                        }
                    }
                }
                RaggedDotMode::Contracting => {
                    for (position, lhs_axis) in dot_dimensions.lhs_batching_dimensions().iter().enumerate() {
                        if let Some(prefix_position) = prefix_axes.iter().position(|axis| axis == lhs_axis) {
                            output_starts[position + 1] = prefix_coordinates[prefix_position];
                            output_limits[position + 1] = prefix_coordinates[prefix_position] + 1;
                        }
                    }
                }
                RaggedDotMode::Batch => unreachable!(),
            }

            // Walk the raw cumulative intervals of this prefix, clipping each one to the physical ragged extent.
            let mut raw_ragged_start = 0usize;
            for (group, &group_size) in sizes[group_range].iter().enumerate() {
                if raw_ragged_start >= ragged_extent {
                    break;
                }

                let ragged_start = raw_ragged_start;
                let ragged_limit = raw_ragged_start.saturating_add(group_size).min(ragged_extent);
                raw_ragged_start = ragged_limit;
                if ragged_start == ragged_limit {
                    continue;
                }

                lhs_starts[ragged_axis] = ragged_start;
                lhs_limits[ragged_axis] = ragged_limit;
                let lhs_slice = self.slice(&lhs_starts, &lhs_limits, &lhs_strides)?;
                let dot = match mode {
                    RaggedDotMode::NonContracting => {
                        let (rhs_group_axis, rhs_slice_shape, dense_dimensions, ragged_output_axis) =
                            non_contracting_metadata.as_ref().unwrap();
                        rhs_starts[*rhs_group_axis] = group;
                        rhs_limits[*rhs_group_axis] = group + 1;
                        let rhs_slice = rhs.slice(&rhs_starts, &rhs_limits, &rhs_strides)?;
                        let rhs_slice = rhs_slice.reshape(rhs_slice_shape.clone())?;
                        output_starts[*ragged_output_axis] = ragged_start;
                        lhs_slice.dot_elements::<T>(&rhs_slice, dense_dimensions, None)?
                    }
                    RaggedDotMode::Contracting => {
                        let rhs_ragged_axis = contracting_rhs_ragged_axis.unwrap();
                        rhs_starts[rhs_ragged_axis] = ragged_start;
                        rhs_limits[rhs_ragged_axis] = ragged_limit;
                        let rhs_slice = rhs.slice(&rhs_starts, &rhs_limits, &rhs_strides)?;
                        output_starts[0] = group;
                        output_limits[0] = group + 1;
                        let dot = lhs_slice.dot_elements::<T>(&rhs_slice, dot_dimensions, None)?;
                        let mut dimensions = vec![Dimension::Static(1)];
                        dimensions.extend_from_slice(dot.r#type().shape().dimensions());
                        let dot = dot.reshape(Shape::new(dimensions))?;
                        let current = output.slice(&output_starts, &output_limits, &output_strides)?;
                        Add::add(&current, &dot)?
                    }
                    RaggedDotMode::Batch => unreachable!(),
                };
                output = output.replace_block(&dot, &output_starts);
            }
        }

        Ok(output)
    }
}

/// Validates that every mapped input of the operation named `operation_name` maps the same extent. Comparing the
/// mapped dimensions validates static extents exactly as [`ArrayBatch::common_batch_size`] does, and it additionally
/// admits a dynamic mapped extent, which two inputs share exactly when it is the same dimension variable.
pub(crate) fn validate_common_mapped_dimension<V: Value<Type = ArrayType>>(
    operation_name: &str,
    inputs: &[ArrayBatch<V>],
) -> Result<(), BatchingError> {
    let mut mapped_dimensions = inputs
        .iter()
        .filter_map(|input| input.batch_axis_position().map(|axis| input.r#type().dimension(axis)));
    let Some(first) = mapped_dimensions.next() else {
        return Ok(());
    };
    match mapped_dimensions.find(|dimension| *dimension != first) {
        None => Ok(()),
        Some(other) => Err(match (first.value(), other.value()) {
            (Some(expected), Some(actual)) => BatchingError::MismatchedBatchSizes { expected, actual },
            _ => BatchingError::MisalignedBatchAxes {
                message: format!("`{operation_name}` inputs map different batch extents `{first}` and `{other}`"),
            },
        }),
    }
}

/// Infers the output type of one generalized dot product operation, naming `operation_name` in diagnostics. The result
/// shape is `[batching..., lhs_result..., rhs_result...]`, where the result dimensions are the input axes that are
/// neither batching nor contracting, in their original order. The output element type is the requested compatible
/// accumulation type when one is provided, or otherwise the common input element type.
///
/// Floating-point compatibility deliberately does not use the standard elementwise promotion lattice. Backend dot
/// instructions expose an independent preferred accumulator type, so even sub-byte and 8-bit inputs may accumulate
/// at `f32` or `f64`. Integer accumulation preserves signedness and may only widen the input type.
///
/// The output [`Sharding`] follows the `dot_general` sharding rule of JAX (i.e., `_dot_general_sharding_rule`
/// in the [JAX implementation](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py)). The StableHLO
/// [`dot_general`](https://openxla.org/stablehlo/spec#dot_general) specification describes the underlying
/// contraction semantics. Placement is inferred as follows:
///
///   - When `output_sharding` is provided, it is validated and returned directly, bypassing the ordinary input
///     consistency checks. Validation checks its rank and mesh, rejects auto axes, and accepts unreduced axes only
///     when the contracting dimensions are sharded identically over exactly those axes.
///   - Inputs must not be unreduced over explicit mesh axes. Reduced inputs are legal, and reduced and varying manual
///     axes are "unioned" across the inputs into the output sharding.
///   - When neither input carries a sharding, the output carries none. When exactly one does, the rule runs with the
///     absent side treated as fully replicated on the present input's mesh. Input meshes must match.
///   - Batch dimension entries are merged preferring the more informative entry (i.e., `Sharded` over `Replicated` over
///     `Unconstrained`). Two different `Sharded` entries are an error when an explicit mesh axis is involved. This
///     intentionally diverges from JAX, which reads batch dimension specs from the left-hand side only.
///   - Contracting dimensions sharded over explicit mesh axes on both inputs are an error. Identically sharded ones
///     make the output sharding ambiguous and require an explicit output sharding. A contracting dimension sharded on
///     only one input is allowed, and its sharding is dropped from the output.
///   - Result dimension entries are copied from the owning input, and auto mesh axes are stripped from the final
///     sharding.
pub(crate) fn infer_dot_output_type(
    operation_name: &str,
    lhs: &ArrayType,
    rhs: &ArrayType,
    dimensions: &DotDimensionNumbers,
    accumulation_type: Option<DataType>,
    output_sharding: Option<&Sharding>,
) -> Result<ArrayType, TypeError> {
    ArrayType::check_matching_manual_variation(operation_name, &[lhs, rhs])?;
    if lhs.data_type() != rhs.data_type() {
        return Err(TypeError::invalid(format!("`{operation_name}` input element types are incompatible")));
    }

    validate_dot_data_type(operation_name, lhs.data_type())?;

    if let Some(accumulation_type) = accumulation_type {
        // An identical type is valid. Floating-point inputs may accumulate at `f32` or `f64`, and integer inputs may
        // accumulate at an integer type with the same signedness that is at least as wide.
        let input_type = lhs.data_type();
        let compatible = input_type == accumulation_type
            || (input_type.is_floating_point() && matches!(accumulation_type, DataType::F32 | DataType::F64))
            || (input_type.is_integer()
                && accumulation_type.is_integer()
                && input_type.is_signed() == accumulation_type.is_signed()
                && accumulation_type.bit_width() >= input_type.bit_width());
        if !compatible {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` input data type `{input_type}` cannot accumulate at data type \
                 `{accumulation_type}`",
            )));
        }
    }

    let lhs_batching = dimensions.lhs_batching_dimensions();
    let rhs_batching = dimensions.rhs_batching_dimensions();
    let lhs_contracting = dimensions.lhs_contracting_dimensions();
    let rhs_contracting = dimensions.rhs_contracting_dimensions();
    dimensions.validate(operation_name, lhs.rank(), rhs.rank())?;

    for (lhs_axis, rhs_axis) in lhs_batching.iter().zip(rhs_batching) {
        if lhs.dimension(*lhs_axis) != rhs.dimension(*rhs_axis) {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` batching dimension sizes do not match (left-hand side axis {lhs_axis}, \
                 right-hand side axis {rhs_axis})",
            )));
        }
    }

    for (lhs_axis, rhs_axis) in lhs_contracting.iter().zip(rhs_contracting) {
        if lhs.dimension(*lhs_axis) != rhs.dimension(*rhs_axis) {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` contracting dimension sizes do not match (left-hand side axis {lhs_axis}, \
                 right-hand side axis {rhs_axis})",
            )));
        }
    }

    let lhs_result = dimensions.lhs_result_dimensions(lhs.rank());
    let rhs_result = dimensions.rhs_result_dimensions(rhs.rank());
    let output_dimensions: Vec<Dimension> = lhs_batching
        .iter()
        .chain(&lhs_result)
        .map(|axis| lhs.dimension(*axis))
        .chain(rhs_result.iter().map(|axis| rhs.dimension(*axis)))
        .collect();

    // Inputs unreduced over an explicit axis are rejected on every path, including the bypass through a requested
    // output sharding, because a pending reduction across devices must be discharged (e.g., through a sharding
    // constraint) before the value is contracted. The check covers only explicit axes, because a `shard_map` value
    // unreduced over a manual axis is the user's to manage. Reduced inputs are always legal, which lets adjoint dots
    // consume reduced cotangents.
    let lhs_sharding = lhs.sharding();
    let rhs_sharding = rhs.sharding();
    for sharding in [lhs_sharding, rhs_sharding].into_iter().flatten() {
        if sharding
            .unreduced_axes()
            .iter()
            .any(|axis_name| sharding.mesh().axis_type(axis_name) == Some(MeshAxisType::Explicit))
        {
            return Err(TypeError::invalid(format!("`{operation_name}` inputs cannot be unreduced")));
        }
    }

    // A dot product is bilinear, so it distributes only over pending sums, including manual ones.
    check_types!(@sum_unreduced, operation_name, [lhs, rhs]);

    let mesh = match (lhs_sharding, rhs_sharding) {
        (Some(left), Some(right)) if left.mesh() != right.mesh() => {
            return Err(TypeError::invalid(format!("`{operation_name}` input shardings must use the same mesh")));
        }
        (Some(left), _) => Some(left.mesh()),
        (_, Some(right)) => Some(right.mesh()),
        (None, None) => None,
    };

    // A missing input sharding is treated as fully replicated, so that the sharding of a single input still propagates.
    let dimension_of = |sharding: Option<&Sharding>, axis: usize| -> ShardingDimension {
        sharding.map_or(ShardingDimension::Replicated, |sharding| sharding.dimensions()[axis].clone())
    };

    let sharding = if let Some(output_sharding) = output_sharding {
        // A requested output sharding is validated and returned directly, bypassing the consistency checks for
        // batching and contracting dimension shardings, like the `out_sharding` argument of JAX.
        if output_sharding.rank() != output_dimensions.len() {
            return Err(TypeError::invalid(format!(
                "`{}` output sharding rank ({}) does not match the output rank ({})",
                operation_name,
                output_sharding.rank(),
                output_dimensions.len(),
            )));
        }
        if let Some(mesh) = mesh
            && output_sharding.mesh() != mesh
        {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` output sharding must use the same mesh as the inputs",
            )));
        }

        let mut referenced_axes: Vec<&String> = output_sharding.unreduced_axes().iter().collect();
        referenced_axes.extend(output_sharding.reduced_axes());
        for dimension in output_sharding.dimensions() {
            if let ShardingDimension::Sharded(axis_names) = dimension {
                referenced_axes.extend(axis_names);
            }
        }

        if referenced_axes
            .iter()
            .any(|name| output_sharding.mesh().axis_type(name) == Some(MeshAxisType::Auto))
        {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` output sharding cannot reference auto mesh axes",
            )));
        }

        // A requested unreduced output defers an all-reduce. Contracting over a dimension sharded across mesh axes
        // leaves each device holding only a partial product sum over its shard of the contraction, whose true value
        // is the sum across devices. Marking the output unreduced over exactly those axes defers that sum to a later
        // operation instead of materializing it here, so the request is valid only when it names precisely the axes
        // that shard the (identically sharded) contracting dimensions. This is JAX's `_dot_general_unreduced_rule`.
        if !output_sharding.unreduced_axes().is_empty() {
            if output_sharding.unreduced_kind() != UnreducedKind::Sum {
                return Err(TypeError::invalid(format!(
                    "`{}` output sharding supports only unreduced axes of kind `{}` but got kind `{}`",
                    operation_name,
                    UnreducedKind::Sum,
                    output_sharding.unreduced_kind(),
                )));
            }

            let lhs_contracting_spec: Vec<ShardingDimension> =
                lhs_contracting.iter().map(|axis| dimension_of(lhs_sharding, *axis)).collect();
            let rhs_contracting_spec: Vec<ShardingDimension> =
                rhs_contracting.iter().map(|axis| dimension_of(rhs_sharding, *axis)).collect();

            if lhs_contracting_spec != rhs_contracting_spec {
                return Err(TypeError::invalid(format!(
                    "`{operation_name}` contracting dimensions must be sharded identically when the output \
                     sharding is unreduced",
                )));
            }

            let mut contracting_axes = BTreeSet::new();
            for dimension in &lhs_contracting_spec {
                if let ShardingDimension::Sharded(axis_names) = dimension {
                    contracting_axes.extend(axis_names.iter().cloned());
                }
            }

            if output_sharding.unreduced_axes() != &contracting_axes {
                return Err(TypeError::invalid(format!(
                    "`{operation_name}` output sharding unreduced axes must equal the axes that shard the \
                     contracting dimensions",
                )));
            }
        }

        // Requested placement does not override logical dependence on manual shards. Variation is determined by the
        // input values, and changing it requires an explicit variation operation or collective.
        let varying = [lhs_sharding, rhs_sharding]
            .into_iter()
            .flatten()
            .flat_map(|sharding| sharding.varying_manual_axes().iter().cloned())
            .collect::<BTreeSet<_>>();
        Some(
            output_sharding
                .clone()
                .with_varying_manual_axes(varying)
                .map_err(|error| TypeError::invalid(error.to_string()))?,
        )
    } else if let Some(mesh) = mesh {
        for (lhs_axis, rhs_axis) in lhs_contracting.iter().zip(rhs_contracting) {
            let left = dimension_of(lhs_sharding, *lhs_axis);
            let right = dimension_of(rhs_sharding, *rhs_axis);

            // Only explicit axis sharding of both contracting inputs triggers the ambiguity and consistency errors.
            // Contracting shardings over only manual or auto axes are left to `shard_map` and the compiler.
            // A contracting dimension sharded on only one input is allowed, and its sharding is dropped from
            // the output, because the partitioner inserts the necessary communication.
            if left.has_explicit_axis(mesh) && right.has_explicit_axis(mesh) {
                let (ShardingDimension::Sharded(left_axes), ShardingDimension::Sharded(right_axes)) = (&left, &right)
                else {
                    unreachable!("`has_explicit_axis` only returns true for sharded dimensions")
                };

                if left_axes != right_axes {
                    return Err(TypeError::invalid(format!(
                        "`{operation_name}` contracting dimensions must have consistent shardings, but got \
                         `{left}` and `{right}`",
                    )));
                }

                return Err(TypeError::invalid(format!(
                    "`{operation_name}` contracting dimensions are sharded, making the output sharding ambiguous; \
                     request an explicit output sharding (e.g., one with unreduced axes) to resolve it",
                )));
            }
        }

        let mut placement = Vec::with_capacity(output_dimensions.len());
        for (lhs_axis, rhs_axis) in lhs_batching.iter().zip(rhs_batching) {
            let left = dimension_of(lhs_sharding, *lhs_axis);
            let right = dimension_of(rhs_sharding, *rhs_axis);

            // Prefer the more informative entry, `Sharded` over `Replicated` over `Unconstrained`. Taking a sharded
            // entry from either input deliberately differs from reading only the left input's batch sharding.
            let merged = match (&left, &right) {
                (ShardingDimension::Sharded(left_axes), ShardingDimension::Sharded(right_axes))
                    if left_axes == right_axes =>
                {
                    left.clone()
                }
                (ShardingDimension::Sharded(_), ShardingDimension::Sharded(_))
                    if left.has_explicit_axis(mesh) || right.has_explicit_axis(mesh) =>
                {
                    // Conflicts over an explicit axis are errors. Conflicts purely over manual or auto axes become
                    // replicated and are left to `shard_map` and the compiler.
                    return Err(TypeError::invalid(format!(
                        "`{operation_name}` batching dimensions must have consistent shardings, but got \
                             `{left}` and `{right}`",
                    )));
                }
                (ShardingDimension::Sharded(_), ShardingDimension::Sharded(_)) => ShardingDimension::Replicated,
                (ShardingDimension::Sharded(_), _) => left.clone(),
                (_, ShardingDimension::Sharded(_)) => right.clone(),
                (ShardingDimension::Replicated, _) | (_, ShardingDimension::Replicated) => {
                    ShardingDimension::Replicated
                }
                (ShardingDimension::Unconstrained, ShardingDimension::Unconstrained) => {
                    ShardingDimension::Unconstrained
                }
            };

            placement.push(merged);
        }
        placement.extend(lhs_result.iter().map(|axis| dimension_of(lhs_sharding, *axis)));
        placement.extend(rhs_result.iter().map(|axis| dimension_of(rhs_sharding, *axis)));

        let merged_axes = |select: fn(&Sharding) -> &BTreeSet<String>| -> BTreeSet<String> {
            match (lhs_sharding, rhs_sharding) {
                (Some(left), Some(right)) => select(left).union(select(right)).cloned().collect(),
                (Some(left), None) => select(left).clone(),
                (None, Some(right)) => select(right).clone(),
                (None, None) => BTreeSet::new(),
            }
        };
        let reduced_axes = merged_axes(Sharding::reduced_axes);
        let varying_manual_axes = merged_axes(Sharding::varying_manual_axes);
        let sharding = Sharding::new(mesh.clone(), placement)
            .and_then(|output| output.with_reduced_axes(reduced_axes))
            .and_then(|output| output.with_varying_manual_axes(varying_manual_axes))
            .map_err(|error| {
                TypeError::invalid(format!("`{operation_name}` output sharding construction failed: {error}"))
            })?;
        Some(sharding.without_auto_axes())
    } else {
        None
    };

    ArrayType::new(accumulation_type.unwrap_or(lhs.data_type()), Shape::new(output_dimensions))
        .with_sharding(sharding)
        .map_err(|error| TypeError::invalid(error.to_string()))
}

/// Validates the batching and contracting dimension indices of one input with rank `rank` before inference or metadata
/// queries index its shape. `operation_name` names the validating operation and `side` names the input in diagnostics.
fn validate_dot_axes(
    operation_name: &str,
    batching: &[usize],
    contracting: &[usize],
    rank: usize,
    side: &str,
) -> Result<(), TypeError> {
    if batching.iter().chain(contracting).any(|axis| *axis >= rank) {
        return Err(TypeError::invalid(format!("`{operation_name}` {side} dimension index out of bounds")));
    }
    for (role, axes) in [("batching", batching), ("contracting", contracting)] {
        if axes.iter().copied().collect::<BTreeSet<_>>().len() != axes.len() {
            return Err(TypeError::invalid(format!("`{operation_name}` {side} {role} dimensions must be distinct")));
        }
    }
    if batching.iter().any(|axis| contracting.contains(axis)) {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` {side} batching and contracting dimensions must be disjoint",
        )));
    }
    Ok(())
}

/// Rejects element types for which the dot kernels of `operation_name` have no arithmetic implementation.
fn validate_dot_data_type(operation_name: &str, data_type: DataType) -> Result<(), TypeError> {
    if data_type.is_numeric() {
        Ok(())
    } else {
        Err(TypeError::invalid(format!("`{operation_name}` does not support input data type `{data_type}`")))
    }
}

/// Applies the product rule `ẏ = f(ȧ, b) + f(a, ḃ)` of a bilinear dot operation `y = f(a, b)`, where `apply` evaluates
/// `f`. Structural zero tangents drop their term, so `ẏ` is a structural zero only when both `ȧ` and `ḃ` are. Dense
/// and ragged dot operations share this rule. When one factor of a tangent term uses the widened tangent
/// representation, both factors are converted to the output tangent element type before applying the operation.
fn bilinear_array_jvp<C, V, ApplyFn, P: DifferentiationPolicy<C>>(
    context: &DifferentiationContext<C, P>,
    primal: V,
    lhs: &DifferentiationDual<V>,
    rhs: &DifferentiationDual<V>,
    apply_fn: ApplyFn,
) -> Result<DifferentiationDual<V>, DifferentiationError>
where
    C: Context<Type = ArrayType, Value = V>,
    V: Value<Type = ArrayType> + ConvertElementType + Add,
    ApplyFn: Fn(&V, &V) -> Result<V, DifferentiationError>,
{
    let tangent_type = primal.r#type().tangent()?;

    let convert_to_tangent_type = |value: &V| {
        if value.r#type().data_type() == tangent_type.data_type() {
            Ok(value.clone())
        } else {
            value.convert_element_type(tangent_type.data_type()).map_err(DifferentiationError::from)
        }
    };

    let apply_tangent = |lhs: &V, rhs: &V| {
        if lhs.r#type().data_type() == rhs.r#type().data_type() {
            apply_fn(lhs, rhs)
        } else {
            apply_fn(&convert_to_tangent_type(lhs)?, &convert_to_tangent_type(rhs)?)
        }
    };

    let lhs_term = lhs
        .tangent()
        .as_value()
        .map(|tangent| apply_tangent(tangent, &context.primal_to_tangent(rhs.primal().clone())?))
        .transpose()?;

    let rhs_term = rhs
        .tangent()
        .as_value()
        .map(|tangent| apply_tangent(&context.primal_to_tangent(lhs.primal().clone())?, tangent))
        .transpose()?;

    let tangent = match (lhs_term, rhs_term) {
        (Some(left), Some(right)) => MaybeZero::Value(left.add(&right)?),
        (Some(term), None) | (None, Some(term)) => MaybeZero::Value(term),
        (None, None) => MaybeZero::Zero(tangent_type),
    };

    DifferentiationDual::new(primal, tangent)
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::{formatdoc, indoc};
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation,
        ArrayType, DataType, Dimension, DimensionBounds, DimensionValue, DimensionVariable, Layout, LogicalMesh,
        MeshAxis, MeshAxisType, Shape, Sharding, ShardingDimension, StridedLayout, f8e4m3fn, f8e8m0fnu, i4,
    };
    use crate::batching::{
        BatchAxis, BatchableOperation, BatchedProgram, BatchingContext, BatchingTracer,
        ProgramBatchingOutputAxesPolicy, batch,
    };
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::{TransposableOperation, TranspositionContext, differentiate_at};
    use crate::macros::{
        check_gradient, check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::operations::reductions::Reduce;
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{
        EmptyRegionDriver, Operation, ProgramBuilder, ProgramError, TypeError, Value, ValueProjection,
    };
    use crate::tracing::{Tracer, TracingContext};

    use super::*;

    /// Creates the explicit mesh used by dot placement tests.
    fn test_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("b", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("n", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("k", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap()
    }

    /// Creates an unplaced floating-point type with the supplied static extents.
    fn plain_array(sizes: &[usize]) -> ArrayType {
        ArrayType::new_static(DataType::F32, sizes)
    }

    /// Creates a validated floating-point type with explicit placement.
    fn sharded_array(mesh: &LogicalMesh, sizes: &[usize], dimensions: Vec<ShardingDimension>) -> ArrayType {
        plain_array(sizes).with_sharding(Sharding::new(mesh.clone(), dimensions).unwrap()).unwrap()
    }

    type ArrayIrContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    #[test]
    fn test_dot_dimension_numbers_new() {
        let dimensions = DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]);
        assert_eq!(dimensions.lhs_contracting_dimensions(), &[2]);
        assert_eq!(dimensions.rhs_contracting_dimensions(), &[1]);
        assert_eq!(dimensions.lhs_batching_dimensions(), &[0]);
        assert_eq!(dimensions.rhs_batching_dimensions(), &[0]);
        assert_eq!(
            dimensions.to_string(),
            "(lhs_contracting=[2], rhs_contracting=[1], lhs_batching=[0], rhs_batching=[0])",
        );
        assert_eq!(
            format!("{dimensions:?}"),
            "DotDimensionNumbers { lhs_contracting_dimensions: [2], rhs_contracting_dimensions: [1], \
                lhs_batching_dimensions: [0], rhs_batching_dimensions: [0] }",
        );
    }

    #[test]
    fn test_dot_dimension_numbers_matmul() {
        assert_eq!(DotDimensionNumbers::matmul(), DotDimensionNumbers::new(vec![1], vec![0], vec![], vec![]));
    }

    #[test]
    fn test_dot_dimension_numbers_inner_product() {
        assert_eq!(DotDimensionNumbers::inner_product(), DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![]));
    }

    #[test]
    fn test_dot_dimension_numbers_lhs_result_dimensions() {
        let dimensions = DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]);
        assert_eq!(dimensions.lhs_result_dimensions(4), vec![1, 3]);
        assert_eq!(DotDimensionNumbers::default().lhs_result_dimensions(0), Vec::<usize>::new());
    }

    #[test]
    fn test_dot_dimension_numbers_rhs_result_dimensions() {
        let dimensions = DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]);
        assert_eq!(dimensions.rhs_result_dimensions(4), vec![2, 3]);
        assert_eq!(DotDimensionNumbers::default().rhs_result_dimensions(0), Vec::<usize>::new());
    }

    #[test]
    fn test_dot_dimension_numbers_batched() {
        let dimensions = DotDimensionNumbers::matmul();
        assert_eq!(
            dimensions.batched(Some(1), Some(2)),
            Some((DotDimensionNumbers::new(vec![2], vec![0], vec![1], vec![2]), Some(0))),
        );
        assert_eq!(dimensions.batched(None, None), Some((dimensions.clone(), None)));
        assert_eq!(dimensions.batched(Some(0), None), None);
        assert_eq!(dimensions.batched(None, Some(0)), None);

        // The inserted axes lead the batching pairs, and existing axes at or after an insertion position shift by one.
        assert_eq!(
            DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]).batched(Some(1), Some(0)),
            Some((DotDimensionNumbers::new(vec![3], vec![2], vec![1, 0], vec![0, 1]), Some(0))),
        );
    }

    #[test]
    fn test_dot_dimension_numbers_validate() {
        assert_eq!(DotDimensionNumbers::matmul().validate(DOT_OPERATION_NAME, 2, 2), Ok(()));
        assert_eq!(DotDimensionNumbers::default().validate(DOT_OPERATION_NAME, 0, 0), Ok(()));

        // Paired lists must have equal lengths.
        assert_eq!(
            DotDimensionNumbers::new(vec![1], vec![0], vec![0], vec![]).validate(DOT_OPERATION_NAME, 2, 2),
            Err(TypeError::invalid(format!(
                "`{DOT_OPERATION_NAME}` batching dimensions have different lengths on the two inputs",
            ))),
        );
        assert_eq!(
            DotDimensionNumbers::new(vec![1], vec![], vec![], vec![]).validate(DOT_OPERATION_NAME, 2, 2),
            Err(TypeError::invalid(format!(
                "`{DOT_OPERATION_NAME}` contracting dimensions have different lengths on the two inputs",
            ))),
        );

        // Indices must be within the rank of their input, and errors use the supplied operation name.
        assert_eq!(
            DotDimensionNumbers::matmul().validate("custom_dot", 1, 2),
            Err(TypeError::invalid("`custom_dot` left-hand side dimension index out of bounds")),
        );
        assert_eq!(
            DotDimensionNumbers::matmul().validate(DOT_OPERATION_NAME, 2, 0),
            Err(TypeError::invalid(format!("`{DOT_OPERATION_NAME}` right-hand side dimension index out of bounds"))),
        );

        // Each list must be distinct, and the batching and contracting lists of an input must be disjoint.
        assert_eq!(
            DotDimensionNumbers::new(vec![1, 1], vec![0, 2], vec![], vec![]).validate(DOT_OPERATION_NAME, 2, 3),
            Err(TypeError::invalid(format!(
                "`{DOT_OPERATION_NAME}` left-hand side contracting dimensions must be distinct",
            ))),
        );
        assert_eq!(
            DotDimensionNumbers::new(vec![1, 0], vec![0, 0], vec![], vec![]).validate(DOT_OPERATION_NAME, 2, 2),
            Err(TypeError::invalid(format!(
                "`{DOT_OPERATION_NAME}` right-hand side contracting dimensions must be distinct",
            ))),
        );
        assert_eq!(
            DotDimensionNumbers::new(vec![1], vec![1], vec![0, 0], vec![0, 0]).validate(DOT_OPERATION_NAME, 2, 2),
            Err(TypeError::invalid(format!(
                "`{DOT_OPERATION_NAME}` left-hand side batching dimensions must be distinct",
            ))),
        );
        assert_eq!(
            DotDimensionNumbers::new(vec![1], vec![1], vec![1], vec![0]).validate(DOT_OPERATION_NAME, 2, 2),
            Err(TypeError::invalid(format!(
                "`{DOT_OPERATION_NAME}` left-hand side batching and contracting dimensions must be disjoint",
            ))),
        );
        assert_eq!(
            DotDimensionNumbers::new(vec![1], vec![0], vec![0], vec![0]).validate(DOT_OPERATION_NAME, 2, 2),
            Err(TypeError::invalid(format!(
                "`{DOT_OPERATION_NAME}` right-hand side batching and contracting dimensions must be disjoint",
            ))),
        );
    }

    #[test]
    fn test_dot() {
        let mesh = test_mesh();
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::sharded(["m"]), ShardingDimension::Replicated]).unwrap();
        let operation =
            DotOperation::matmul().with_accumulation_type(DataType::F32).with_output_sharding(sharding.clone());

        // Operation identity, accessors, and formatting. The rendering lists the optional fields after the dimension
        // numbers.
        assert_eq!(operation.name(), DOT_OPERATION_NAME);
        assert_eq!(operation.dimensions(), &DotDimensionNumbers::matmul());
        assert_eq!(operation.accumulation_type(), Some(DataType::F32));
        assert_eq!(operation.output_sharding(), Some(&sharding));
        assert_eq!(DotOperation::matmul().accumulation_type(), None);
        assert_eq!(DotOperation::matmul().output_sharding(), None);
        assert_eq!(
            operation.to_string(),
            formatdoc! {"
                dot [
                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    accumulation_type=f32,
                    output_sharding={sharding},
                ]
            "}
            .trim_end(),
        );
        assert_eq!(
            format!("{:?}", DotOperation::matmul()),
            "DotOperation { dimensions: DotDimensionNumbers { lhs_contracting_dimensions: [1], \
             rhs_contracting_dimensions: [0], lhs_batching_dimensions: [], rhs_batching_dimensions: [] }, \
             accumulation_type: None, output_sharding: None }",
        );

        // Program rendering uses the canonical operation name and includes the dimension numbers.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(left, right): (Tracer<_>, Tracer<_>)| left.dot(&right, &DotDimensionNumbers::matmul()),
            (ArrayType::new_static(DataType::F32, [2, 3]), ArrayType::new_static(DataType::F32, [3, 4])),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 3], %1:f32[3, 4] .
                let %2:f32[2, 4] = dot [
                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // Staged dots carry both the accumulation type and the requested output sharding.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(left, right): (Tracer<_>, Tracer<_>)| {
                left.dot_with_configuration(
                    &right,
                    &DotDimensionNumbers::matmul(),
                    Some(DataType::F32),
                    Some(&sharding),
                )
            },
            (ArrayType::new_static(DataType::BF16, [2, 3]), ArrayType::new_static(DataType::BF16, [3, 4])),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda %0:bf16[2, 3], %1:bf16[3, 4] .
                let %2:f32[2, 4][sharding={sharding}] = dot [
                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    accumulation_type=f32,
                    output_sharding={sharding},
                ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dot_type_inference() {
        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying = Sharding::replicated(manual_mesh, 2).with_varying_manual_axes(["devices"]).unwrap();
        check_operation_type_inference!(
            operation = DotOperation::matmul(),
            cases = [
                { input_types = [plain_array(&[2, 3]), plain_array(&[3, 4])], output_types = [plain_array(&[2, 4])], },
                { input_types = [plain_array(&[2, 0]), plain_array(&[0, 4])], output_types = [plain_array(&[2, 4])], },
                { input_types = [plain_array(&[2, 3])], error = "expected 2 inputs but got 1", },
                {
                    input_types = [plain_array(&[2, 3]), plain_array(&[4, 2])],
                    error = format!(
                        "`{DOT_OPERATION_NAME}` contracting dimension sizes do not match (left-hand side axis 1, \
                         right-hand side axis 0)",
                    ),
                },
                {
                    input_types = [plain_array(&[2, 3]), ArrayType::new_static(DataType::F64, [3, 4])],
                    error = format!("`{DOT_OPERATION_NAME}` input element types are incompatible"),
                },
                {
                    input_types = [
                        ArrayType::new_static(DataType::Boolean, [2, 3]),
                        ArrayType::new_static(DataType::Boolean, [3, 4]),
                    ],
                    error = format!("`{DOT_OPERATION_NAME}` does not support input data type `bool`"),
                },
                {
                    input_types = [
                        ArrayType::new_static(DataType::Token, [2, 3]),
                        ArrayType::new_static(DataType::Token, [3, 4]),
                    ],
                    error = format!("`{DOT_OPERATION_NAME}` does not support input data type `token`"),
                },
                {
                    input_types = [
                        ArrayType::new_static(DataType::Zero, [2, 3]),
                        ArrayType::new_static(DataType::Zero, [3, 4]),
                    ],
                    error = format!("`{DOT_OPERATION_NAME}` does not support input data type `zero`"),
                },
                // An input without a sharding is invariant over every manual axis.
                {
                    input_types = [plain_array(&[2, 3]).with_sharding(varying).unwrap(), plain_array(&[3, 4])],
                    error = format!(
                        "`{DOT_OPERATION_NAME}` inputs must have matching varying manual axes; insert \
                         `parallel_vary` on the inputs that lack an axis, as `align_manual_variation` does",
                    ),
                },
            ],
        );

        // Type inference validates the dimension numbers against the input ranks.
        check_operation_type_inference!(
            operation = DotOperation::new(DotDimensionNumbers::new(vec![1, 1], vec![0, 0], vec![], vec![])),
            cases = [{
                input_types = [plain_array(&[2, 3]), plain_array(&[3, 4])],
                error = format!("`{DOT_OPERATION_NAME}` left-hand side contracting dimensions must be distinct"),
            }],
        );
    }

    #[test]
    fn test_dot_type_inference_dynamic_dimensions() {
        let batch = DimensionVariable::new("batch", DimensionBounds::unbounded());
        let contracting = DimensionVariable::new("contracting", DimensionBounds::non_negative(Some(4)).unwrap());
        let lhs = ArrayType::new(
            DataType::F64,
            Shape::new(vec![
                Dimension::Dynamic(batch.clone()),
                Dimension::Static(2),
                Dimension::Dynamic(contracting.clone()),
            ]),
        );
        let rhs = ArrayType::new(
            DataType::F64,
            Shape::new(vec![
                Dimension::Dynamic(batch.clone()),
                Dimension::Dynamic(contracting.clone()),
                Dimension::Static(3),
            ]),
        );
        let static_rhs = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(4), Dimension::Static(3)]),
        );
        let mismatched_batch_rhs = ArrayType::new(
            DataType::F64,
            Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("other_batch", DimensionBounds::unbounded())),
                Dimension::Dynamic(contracting),
                Dimension::Static(3),
            ]),
        );

        // Batched matrix multiplication contracts axis 2 of the left-hand side with axis 1 of the right-hand side over
        // batching axis 0.
        check_operation_type_inference!(
            operation = DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0])),
            cases = [
                // Dynamic extents that compare equal flow through inference into the output type. The dynamic
                // batching dimension is preserved and the equal bounded dynamic contracting dimensions are dropped.
                {
                    input_types = [lhs.clone(), rhs],
                    output_types = [ArrayType::new(
                        DataType::F64,
                        Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(2), Dimension::Static(3)]),
                    )],
                },
                // Pairs of a static and a dynamic dimension and pairs of distinct dynamic dimensions keep erroring
                // under the strict size equality used for batching and contracting dimensions.
                {
                    input_types = [lhs.clone(), static_rhs],
                    error = format!(
                        "`{DOT_OPERATION_NAME}` contracting dimension sizes do not match (left-hand side axis 2, \
                         right-hand side axis 1)",
                    ),
                },
                {
                    input_types = [lhs, mismatched_batch_rhs],
                    error = format!(
                        "`{DOT_OPERATION_NAME}` batching dimension sizes do not match (left-hand side axis 0, \
                         right-hand side axis 0)",
                    ),
                },
            ],
        );
    }

    #[test]
    fn test_dot_type_inference_accumulation_type() {
        let f8_input = ArrayType::new_static(DataType::F8E4M3FN, [2, 2]);
        let bf16_input = ArrayType::new_static(DataType::BF16, [2, 2]);
        let output_type = ArrayType::new_static(DataType::F32, [2, 2]);

        // Floating-point inputs may accumulate at `f32` or `f64`, so the output widens to the accumulation type.
        check_operation_type_inference!(
            operation = DotOperation::matmul().with_accumulation_type(DataType::F32),
            cases = [
                { input_types = [f8_input.clone(), f8_input.clone()], output_types = [output_type.clone()] },
                { input_types = [bf16_input.clone(), bf16_input], output_types = [output_type] },
            ],
        );
        check_operation_type_inference!(
            operation = DotOperation::matmul().with_accumulation_type(DataType::F16),
            cases = [{
                input_types = [plain_array(&[2, 2]), plain_array(&[2, 2])],
                error = format!("`{DOT_OPERATION_NAME}` input data type `f32` cannot accumulate at data type `f16`"),
            }],
        );

        // Integer inputs may accumulate at an integer type with the same signedness that is at least as wide.
        check_operation_type_inference!(
            operation = DotOperation::matmul().with_accumulation_type(DataType::I32),
            cases = [{
                input_types = [
                    ArrayType::new_static(DataType::I8, [2, 2]),
                    ArrayType::new_static(DataType::I8, [2, 2]),
                ],
                output_types = [ArrayType::new_static(DataType::I32, [2, 2])],
            }],
        );
        check_operation_type_inference!(
            operation = DotOperation::matmul().with_accumulation_type(DataType::I16),
            cases = [{
                input_types = [
                    ArrayType::new_static(DataType::I32, [2, 2]),
                    ArrayType::new_static(DataType::I32, [2, 2]),
                ],
                error = format!("`{DOT_OPERATION_NAME}` input data type `i32` cannot accumulate at data type `i16`"),
            }],
        );

        // A requested output sharding places the widened output.
        let sharding =
            Sharding::new(test_mesh(), vec![ShardingDimension::sharded(["m"]), ShardingDimension::Replicated]).unwrap();
        check_operation_type_inference!(
            operation = DotOperation::matmul()
                .with_accumulation_type(DataType::F32)
                .with_output_sharding(sharding.clone()),
            cases = [{
                input_types = [f8_input.clone(), f8_input],
                output_types = [ArrayType::new_static(DataType::F32, [2, 2]).with_sharding(sharding).unwrap()],
            }],
        );
    }

    #[test]
    fn test_dot_type_inference_batched_sharding_propagation() {
        let mesh = test_mesh();
        let lhs = sharded_array(
            &mesh,
            &[2, 4, 8],
            vec![ShardingDimension::sharded(["b"]), ShardingDimension::sharded(["m"]), ShardingDimension::Replicated],
        );
        let rhs = sharded_array(
            &mesh,
            &[2, 8, 16],
            vec![ShardingDimension::Replicated, ShardingDimension::Replicated, ShardingDimension::sharded(["n"])],
        );
        // The batch dimension merges the more informative entry (left-hand side `b` over right-hand side replicated),
        // and the result dimensions are copied from their owning inputs.
        check_operation_type_inference!(
            operation = DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0])),
            cases = [{
                input_types = [lhs, rhs],
                output_types = [sharded_array(
                    &mesh,
                    &[2, 4, 16],
                    vec![
                        ShardingDimension::sharded(["b"]),
                        ShardingDimension::sharded(["m"]),
                        ShardingDimension::sharded(["n"]),
                    ],
                )],
            }],
        );
    }

    #[test]
    fn test_dot_type_inference_matmul_sharding_propagation() {
        let mesh = test_mesh();
        let replicated_lhs =
            sharded_array(&mesh, &[4, 8], vec![ShardingDimension::Replicated, ShardingDimension::Replicated]);
        let replicated_rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::Replicated, ShardingDimension::Replicated]);
        let lhs = sharded_array(&mesh, &[4, 8], vec![ShardingDimension::sharded(["m"]), ShardingDimension::Replicated]);
        let rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::Replicated, ShardingDimension::sharded(["n"])]);
        check_operation_type_inference!(
            operation = DotOperation::matmul(),
            cases = [
                // Fully replicated inputs stay fully replicated.
                {
                    input_types = [replicated_lhs, replicated_rhs],
                    output_types = [sharded_array(
                        &mesh,
                        &[4, 16],
                        vec![ShardingDimension::Replicated, ShardingDimension::Replicated],
                    )],
                },
                // `[M@m, K] · [K, N@n] -> [M@m, N@n]`.
                {
                    input_types = [lhs, rhs],
                    output_types = [sharded_array(
                        &mesh,
                        &[4, 16],
                        vec![ShardingDimension::sharded(["m"]), ShardingDimension::sharded(["n"])],
                    )],
                },
            ],
        );
    }

    #[test]
    fn test_dot_type_inference_one_sided_sharding_propagation() {
        let mesh = test_mesh();
        let lhs = sharded_array(&mesh, &[4, 8], vec![ShardingDimension::sharded(["m"]), ShardingDimension::Replicated]);
        check_operation_type_inference!(
            operation = DotOperation::matmul(),
            cases = [
                // A missing input sharding is treated as fully replicated on the present input's mesh.
                {
                    input_types = [lhs, plain_array(&[8, 16])],
                    output_types = [sharded_array(
                        &mesh,
                        &[4, 16],
                        vec![ShardingDimension::sharded(["m"]), ShardingDimension::Replicated],
                    )],
                },
                // Without any input shardings, the output carries none.
                {
                    input_types = [plain_array(&[4, 8]), plain_array(&[8, 16])],
                    output_types = [plain_array(&[4, 16])],
                },
            ],
        );
    }

    #[test]
    fn test_dot_type_inference_batch_sharding_conflict() {
        let mesh = test_mesh();
        let lhs = sharded_array(
            &mesh,
            &[2, 4, 8],
            vec![ShardingDimension::sharded(["b"]), ShardingDimension::Replicated, ShardingDimension::Replicated],
        );
        let rhs = sharded_array(
            &mesh,
            &[2, 8, 16],
            vec![ShardingDimension::sharded(["m"]), ShardingDimension::Replicated, ShardingDimension::Replicated],
        );
        check_operation_type_inference!(
            operation = DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0])),
            cases = [{
                input_types = [lhs, rhs],
                error = format!(
                    "`{DOT_OPERATION_NAME}` batching dimensions must have consistent shardings, but got \
                     `{{'b'}}` and `{{'m'}}`",
                ),
            }],
        );
    }

    #[test]
    fn test_dot_type_inference_contracting_sharding() {
        let mesh = test_mesh();
        let lhs = sharded_array(&mesh, &[4, 8], vec![ShardingDimension::Replicated, ShardingDimension::sharded(["k"])]);
        let rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::sharded(["k"]), ShardingDimension::Replicated]);
        let mismatched_rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::sharded(["m"]), ShardingDimension::Replicated]);
        let replicated_rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::Replicated, ShardingDimension::Replicated]);
        check_operation_type_inference!(
            operation = DotOperation::matmul(),
            cases = [
                // Identically sharded contracting dimensions make the output sharding ambiguous.
                {
                    input_types = [lhs.clone(), rhs],
                    error = format!(
                        "`{DOT_OPERATION_NAME}` contracting dimensions are sharded, making the output sharding \
                         ambiguous; request an explicit output sharding (e.g., one with unreduced axes) to resolve it",
                    ),
                },
                // Differently sharded contracting dimensions are inconsistent.
                {
                    input_types = [lhs.clone(), mismatched_rhs],
                    error = format!(
                        "`{DOT_OPERATION_NAME}` contracting dimensions must have consistent shardings, but got \
                         `{{'k'}}` and `{{'m'}}`",
                    ),
                },
                // A contracting dimension sharded on only one input is allowed, and its sharding is dropped.
                {
                    input_types = [lhs, replicated_rhs],
                    output_types = [sharded_array(
                        &mesh,
                        &[4, 16],
                        vec![ShardingDimension::Replicated, ShardingDimension::Replicated],
                    )],
                },
            ],
        );
    }

    #[test]
    fn test_dot_type_inference_mesh_mismatch() {
        let mesh = test_mesh();
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("m", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let lhs = sharded_array(&mesh, &[4, 8], vec![ShardingDimension::sharded(["m"]), ShardingDimension::Replicated]);
        let rhs = sharded_array(
            &other_mesh,
            &[8, 16],
            vec![ShardingDimension::sharded(["m"]), ShardingDimension::Replicated],
        );
        check_operation_type_inference!(
            operation = DotOperation::matmul(),
            cases = [{
                input_types = [lhs, rhs],
                error = format!("`{DOT_OPERATION_NAME}` input shardings must use the same mesh"),
            }],
        );
    }

    #[test]
    fn test_dot_type_inference_unreduced_and_reduced_inputs() {
        let mesh = test_mesh();
        let unreduced_lhs = plain_array(&[4, 8])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::Replicated, ShardingDimension::Replicated])
                    .unwrap()
                    .with_unreduced_axes(["k"])
                    .unwrap(),
            )
            .unwrap();
        let reduced_lhs = plain_array(&[4, 8])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::Replicated, ShardingDimension::Replicated])
                    .unwrap()
                    .with_reduced_axes(["k"])
                    .unwrap(),
            )
            .unwrap();
        check_operation_type_inference!(
            operation = DotOperation::matmul(),
            cases = [
                // Unreduced inputs are rejected because the pending reduction must be discharged before the
                // contraction.
                {
                    input_types = [unreduced_lhs, plain_array(&[8, 16])],
                    error = format!("`{DOT_OPERATION_NAME}` inputs cannot be unreduced"),
                },
                // Reduced inputs are legal (this is what lets adjoint dots consume reduced cotangents), and their
                // reduced axes are unioned into the output sharding.
                {
                    input_types = [reduced_lhs, plain_array(&[8, 16])],
                    output_types = [
                        plain_array(&[4, 16])
                            .with_sharding(
                                Sharding::new(
                                    mesh,
                                    vec![ShardingDimension::Replicated, ShardingDimension::Replicated],
                                )
                                .unwrap()
                                .with_reduced_axes(["k"])
                                .unwrap(),
                            )
                            .unwrap(),
                    ],
                },
            ],
        );
    }

    #[test]
    fn test_dot_type_inference_strips_auto_axes() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let lhs = sharded_array(&mesh, &[4, 8], vec![ShardingDimension::sharded(["a"]), ShardingDimension::Replicated]);
        let rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::Replicated, ShardingDimension::sharded(["m"])]);
        check_operation_type_inference!(
            operation = DotOperation::matmul(),
            cases = [{
                input_types = [lhs, rhs],
                output_types = [sharded_array(
                    &mesh,
                    &[4, 16],
                    vec![ShardingDimension::Replicated, ShardingDimension::sharded(["m"])],
                )],
            }],
        );
    }

    #[test]
    fn test_dot_type_inference_output_sharding_bypass_and_validation() {
        let mesh = test_mesh();
        let lhs = sharded_array(
            &mesh,
            &[2, 4, 8],
            vec![ShardingDimension::sharded(["b"]), ShardingDimension::Replicated, ShardingDimension::Replicated],
        );
        let rhs = sharded_array(
            &mesh,
            &[2, 8, 16],
            vec![ShardingDimension::sharded(["m"]), ShardingDimension::Replicated, ShardingDimension::Replicated],
        );

        // The requested output sharding bypasses the batch consistency checks (here, conflicting batch shardings).
        let requested = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::sharded(["b"]), ShardingDimension::Replicated, ShardingDimension::sharded(["n"])],
        )
        .unwrap();
        check_operation_type_inference!(
            operation = DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]))
                .with_output_sharding(requested.clone()),
            cases = [{
                input_types = [lhs.clone(), rhs.clone()],
                output_types = [plain_array(&[2, 4, 16]).with_sharding(requested).unwrap()],
            }],
        );

        // Rank validation.
        let rank_mismatched = Sharding::new(mesh.clone(), vec![ShardingDimension::Replicated]).unwrap();
        check_operation_type_inference!(
            operation = DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]))
                .with_output_sharding(rank_mismatched),
            cases = [{
                input_types = [lhs.clone(), rhs.clone()],
                error = format!("`{DOT_OPERATION_NAME}` output sharding rank (1) does not match the output rank (3)"),
            }],
        );

        // Mesh validation.
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("m", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other_mesh_sharding = Sharding::replicated(other_mesh, 3);
        check_operation_type_inference!(
            operation = DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]))
                .with_output_sharding(other_mesh_sharding),
            cases = [{
                input_types = [lhs, rhs],
                error = format!("`{DOT_OPERATION_NAME}` output sharding must use the same mesh as the inputs"),
            }],
        );

        // Auto mesh axes cannot be requested explicitly.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let auto_sharding =
            Sharding::new(auto_mesh, vec![ShardingDimension::sharded(["a"]), ShardingDimension::Replicated]).unwrap();
        check_operation_type_inference!(
            operation = DotOperation::matmul().with_output_sharding(auto_sharding),
            cases = [{
                input_types = [plain_array(&[4, 8]), plain_array(&[8, 16])],
                error = format!("`{DOT_OPERATION_NAME}` output sharding cannot reference auto mesh axes"),
            }],
        );
    }

    #[test]
    fn test_dot_type_inference_output_sharding_preserves_manual_variation() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying = Sharding::replicated(mesh.clone(), 2).with_varying_manual_axes(["devices"]).unwrap();
        let left = plain_array(&[2, 3]).with_sharding(varying.clone()).unwrap();
        let right = plain_array(&[3, 4]).with_sharding(varying.clone()).unwrap();
        check_operation_type_inference!(
            operation = DotOperation::matmul().with_output_sharding(Sharding::replicated(mesh, 2)),
            cases = [{
                input_types = [left, right],
                output_types = [plain_array(&[2, 4]).with_sharding(varying).unwrap()],
            }],
        );
    }

    #[test]
    fn test_dot_type_inference_unreduced_output_sharding() {
        let mesh = test_mesh();
        let lhs = sharded_array(&mesh, &[4, 8], vec![ShardingDimension::Replicated, ShardingDimension::sharded(["k"])]);
        let rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::sharded(["k"]), ShardingDimension::Replicated]);
        let replicated_rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::Replicated, ShardingDimension::Replicated]);
        let unreduced = Sharding::new(mesh.clone(), vec![ShardingDimension::Replicated, ShardingDimension::Replicated])
            .unwrap()
            .with_unreduced_axes(["k"])
            .unwrap();
        check_operation_type_inference!(
            operation = DotOperation::matmul().with_output_sharding(unreduced.clone()),
            cases = [
                // Identically sharded contracting dimensions plus a matching unreduced set produce an unreduced
                // output.
                {
                    input_types = [lhs.clone(), rhs.clone()],
                    output_types = [plain_array(&[4, 16]).with_sharding(unreduced).unwrap()],
                },
                // The contracting dimensions must be sharded identically.
                {
                    input_types = [lhs.clone(), replicated_rhs.clone()],
                    error = format!(
                        "`{DOT_OPERATION_NAME}` contracting dimensions must be sharded identically when the \
                         output sharding is unreduced",
                    ),
                },
                // Unsharded contracting dimensions cannot produce an unreduced output.
                {
                    input_types = [
                        replicated_rhs,
                        sharded_array(
                            &mesh,
                            &[16, 4],
                            vec![ShardingDimension::Replicated, ShardingDimension::Replicated],
                        ),
                    ],
                    error = format!(
                        "`{DOT_OPERATION_NAME}` output sharding unreduced axes must equal the axes that shard \
                         the contracting dimensions",
                    ),
                },
            ],
        );

        // The unreduced set must equal the axes that shard the contracting dimensions.
        let mismatched = Sharding::new(mesh, vec![ShardingDimension::Replicated, ShardingDimension::Replicated])
            .unwrap()
            .with_unreduced_axes(["n"])
            .unwrap();
        check_operation_type_inference!(
            operation = DotOperation::matmul().with_output_sharding(mismatched),
            cases = [{
                input_types = [lhs, rhs],
                error = format!(
                    "`{DOT_OPERATION_NAME}` output sharding unreduced axes must equal the axes that shard \
                     the contracting dimensions",
                ),
            }],
        );
    }

    #[test]
    fn test_dot_interpretation() {
        // Matrix multiplication contracts the columns of the left-hand side with the rows of the right-hand side.
        let lhs = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs = Array::matrix(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        assert_eq!(
            lhs.dot(&rhs, &DotDimensionNumbers::matmul()),
            Ok(Array::matrix(2, 2, vec![58.0, 64.0, 139.0, 154.0]).unwrap()),
        );

        // Both inputs are decoded through their physical layouts rather than through dense logical payload copies.
        let lhs_type =
            ArrayType::new_static(DataType::U16, [2, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![-6, 2])));
        let rhs_type =
            ArrayType::new_static(DataType::U16, [3, 2]).with_layout(Layout::Strided(StridedLayout::new(vec![4, -2])));
        let lhs = Array::from_elements(lhs_type, &[1u16, 2, 3, 4, 5, 6]).unwrap();
        let rhs = Array::from_elements(rhs_type, &[7u16, 8, 9, 10, 11, 12]).unwrap();
        assert_eq!(
            lhs.dot(&rhs, &DotDimensionNumbers::matmul()).unwrap().elements::<u16>(),
            Ok(vec![58, 64, 139, 154]),
        );

        // The output axes are the batching axes, then the left-hand side result axes, and then the right-hand side
        // result axes.
        let lhs = Array::from_elements(ArrayType::new_static(DataType::I32, [2, 2, 2]), &[1i32, 2, 3, 4, 5, 6, 7, 8])
            .unwrap();
        let rhs = Array::from_elements(ArrayType::new_static(DataType::I32, [2, 2, 1]), &[2i32, 3, 4, 5]).unwrap();
        assert_eq!(
            lhs.dot(&rhs, &DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0])),
            Ok(Array::from_elements(ArrayType::new_static(DataType::I32, [2, 2, 1]), &[8i32, 18, 50, 68]).unwrap()),
        );

        // Narrow integer products and sums wrap at the declared element width, and complex accumulation retains both
        // components.
        let lhs = Array::from_elements(
            ArrayType::new_static(DataType::I4, [1, 2]),
            &[i4::new(7).unwrap(), i4::new(7).unwrap()],
        )
        .unwrap();
        let rhs = Array::from_elements(
            ArrayType::new_static(DataType::I4, [2, 1]),
            &[i4::new(2).unwrap(), i4::new(2).unwrap()],
        )
        .unwrap();
        assert_eq!(
            lhs.dot(&rhs, &DotDimensionNumbers::matmul()).unwrap().elements::<i4>(),
            Ok(vec![i4::new(-4).unwrap()]),
        );
        let lhs = Array::matrix(1, 2, vec![ComplexNumber::new(1.0f32, 2.0), ComplexNumber::new(3.0, -1.0)]).unwrap();
        let rhs = Array::matrix(2, 1, vec![ComplexNumber::new(2.0f32, -1.0), ComplexNumber::new(0.5, 4.0)]).unwrap();
        assert_eq!(
            lhs.dot(&rhs, &DotDimensionNumbers::matmul()).unwrap().elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(9.5, 14.5)]),
        );

        // An empty contracting dimension materializes one additive identity for every result coordinate.
        let lhs = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [2, 0]), &[]).unwrap();
        let rhs = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0, 3]), &[]).unwrap();
        assert_eq!(lhs.dot(&rhs, &DotDimensionNumbers::matmul()).unwrap().elements::<f32>(), Ok(vec![0.0; 6]));
    }

    #[test]
    fn test_dot_interpretation_accumulates_narrow_floats_at_f32() {
        // Each small term is below half an ulp of the running sum at `1`, so rounding after every addition would keep
        // the sum at `1`. Accumulating at `f32` reaches exactly one more ulp, which is then rounded once.
        let lhs =
            Array::vector([1.0f32, 0.001953125, 0.001953125, 0.001953125, 0.001953125].map(bf16::from_f32).to_vec())
                .unwrap();
        let rhs = Array::vector(vec![bf16::from_f32(1.0); 5]).unwrap();
        let product = lhs.dot(&rhs, &DotDimensionNumbers::inner_product()).unwrap();
        assert_eq!(product.r#type().into_owned(), ArrayType::scalar(DataType::BF16));
        assert_eq!(product.elements::<bf16>(), Ok(vec![bf16::from_f32(1.0078125)]));

        let lhs = Array::vector(
            [1.0f32, 0.000244140625, 0.000244140625, 0.000244140625, 0.000244140625].map(f16::from_f32).to_vec(),
        )
        .unwrap();
        let rhs = Array::vector(vec![f16::from_f32(1.0); 5]).unwrap();
        let product = lhs.dot(&rhs, &DotDimensionNumbers::inner_product()).unwrap();
        assert_eq!(product.r#type().into_owned(), ArrayType::scalar(DataType::F16));
        assert_eq!(product.elements::<f16>(), Ok(vec![f16::from_f32(1.0009765625)]));
    }

    #[test]
    fn test_dot_interpretation_empty_outputs_and_contractions_skip_large_products() {
        // These enormous extents require no storage. Their products must never be evaluated for an empty result.
        let extent = 1usize << (usize::BITS / 2);
        let left = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0, extent, extent]), &[]).unwrap();
        let right =
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [extent, extent, 0]), &[]).unwrap();
        assert_eq!(
            left.dot(&right, &DotDimensionNumbers::new(vec![1, 2], vec![0, 1], vec![], vec![])),
            Ok(Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0, 0]), &[]).unwrap()),
        );
        assert_eq!(
            left.dot(&right, &DotDimensionNumbers::new(vec![0, 1, 2], vec![2, 0, 1], vec![], vec![])),
            Ok(Array::scalar(0.0f32).unwrap()),
        );
    }

    #[test]
    fn test_dot_interpretation_rejects_invalid_input_types() {
        let left = Array::matrix(1, 1, vec![2.0f32]).unwrap();
        let right = Array::matrix(1, 1, vec![3.0f64]).unwrap();
        assert_eq!(
            left.dot(&right, &DotDimensionNumbers::matmul()),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{DOT_OPERATION_NAME}` input element types are incompatible",
            )))),
        );
        let boolean = Array::matrix(1, 1, vec![true]).unwrap();
        assert_eq!(
            boolean.dot(&boolean, &DotDimensionNumbers::matmul()),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{DOT_OPERATION_NAME}` does not support input data type `bool`",
            )))),
        );
    }

    #[test]
    fn test_dot_interpretation_accumulation_type() {
        // An accumulation type promotes both inputs and then runs the same typed contraction at the wider element data
        // type, which becomes the element data type of the result.
        let lhs = Array::matrix(1, 2, vec![f16::from_f32(1.5), f16::from_f32(2.0)]).unwrap();
        let rhs = Array::matrix(2, 1, vec![f16::from_f32(2.0), f16::from_f32(3.0)]).unwrap();
        assert_eq!(
            lhs.dot_with_accumulation_type(&rhs, &DotDimensionNumbers::matmul(), DataType::F32),
            Ok(Array::matrix(1, 1, vec![9.0f32]).unwrap()),
        );

        // Every value below is exactly representable in `f8e4m3fn`, so the `f32` results are exact.
        let input_type = ArrayType::new_static(DataType::F8E4M3FN, [2, 2]);
        let lhs = Array::from_elements::<f8e4m3fn>(
            input_type.clone(),
            &[0.5, 1.0, 1.5, 2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        let rhs = Array::from_elements::<f8e4m3fn>(
            input_type,
            &[1.0, 0.5, 0.5, 1.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        assert_eq!(
            lhs.dot_with_accumulation_type(&rhs, &DotDimensionNumbers::matmul(), DataType::F32),
            Ok(Array::matrix(2, 2, vec![1.0f32, 1.25, 2.5, 2.75]).unwrap()),
        );
    }

    #[test]
    fn test_dot_interpretation_accumulation_type_rejects_invalid_input_types() {
        let left = Array::matrix(1, 1, vec![2.0f32]).unwrap();
        let right = Array::matrix(1, 1, vec![3.0f64]).unwrap();
        assert_eq!(
            left.dot_with_accumulation_type(&right, &DotDimensionNumbers::matmul(), DataType::F64),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{DOT_OPERATION_NAME}` input element types are incompatible",
            )))),
        );
        assert_eq!(
            left.dot_with_accumulation_type(&left, &DotDimensionNumbers::matmul(), DataType::F16),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{DOT_OPERATION_NAME}` input data type `f32` cannot accumulate at data type `f16`",
            )))),
        );
    }

    #[test]
    fn test_dot_interpretation_output_sharding() {
        // The requested sharding is validated by the operation's type inference and recorded on the computed result.
        let mesh = test_mesh();
        let lhs = Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs = Array::matrix(3, 2, vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["m"]), ShardingDimension::Replicated])
                .unwrap();
        assert_eq!(
            lhs.dot_with_output_sharding(&rhs, &DotDimensionNumbers::matmul(), &sharding),
            Ok(Array::from_elements(
                plain_array(&[2, 2]).with_sharding(sharding).unwrap(),
                &[58.0f32, 64.0, 139.0, 154.0]
            )
            .unwrap()),
        );

        // Invalid requests are rejected exactly as they are for staged programs.
        let rank_one = Sharding::new(mesh.clone(), vec![ShardingDimension::Replicated]).unwrap();
        assert_eq!(
            lhs.dot_with_output_sharding(&rhs, &DotDimensionNumbers::matmul(), &rank_one),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{DOT_OPERATION_NAME}` output sharding rank (1) does not match the output rank (2)",
            )))),
        );

        // Unreduced outputs are recorded as well, because a concrete array's payload is its fully reduced logical
        // value, which is exactly the total of the partial sums that an unreduced sharding describes.
        let lhs = Array::from_elements(
            sharded_array(&mesh, &[2, 2], vec![ShardingDimension::Replicated, ShardingDimension::sharded(["k"])]),
            &[1.0f32, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let rhs = Array::from_elements(
            sharded_array(&mesh, &[2, 2], vec![ShardingDimension::sharded(["k"]), ShardingDimension::Replicated]),
            &[5.0f32, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let unreduced = Sharding::new(mesh, vec![ShardingDimension::Replicated, ShardingDimension::Replicated])
            .unwrap()
            .with_unreduced_axes(["k"])
            .unwrap();
        assert_eq!(
            lhs.dot_with_output_sharding(&rhs, &DotDimensionNumbers::matmul(), &unreduced),
            Ok(Array::from_elements(
                plain_array(&[2, 2]).with_sharding(unreduced).unwrap(),
                &[19.0f32, 22.0, 43.0, 50.0]
            )
            .unwrap()),
        );
    }

    #[test]
    fn test_dot_interpretation_accumulation_type_and_output_sharding() {
        // The accumulation type determines the element type of the result and the requested sharding its placement,
        // including an unreduced placement, which a concrete array records on its fully reduced value. Every value
        // below is exactly representable in `f8e4m3fn`, so the `f32` results are exact.
        let mesh = test_mesh();
        let lhs = Array::from_elements::<f8e4m3fn>(
            ArrayType::new_static(DataType::F8E4M3FN, [2, 2])
                .with_sharding(
                    Sharding::new(mesh.clone(), vec![ShardingDimension::Replicated, ShardingDimension::sharded(["k"])])
                        .unwrap(),
                )
                .unwrap(),
            &[0.5, 1.0, 1.5, 2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        let rhs = Array::from_elements::<f8e4m3fn>(
            ArrayType::new_static(DataType::F8E4M3FN, [2, 2])
                .with_sharding(
                    Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["k"]), ShardingDimension::Replicated])
                        .unwrap(),
                )
                .unwrap(),
            &[1.0, 0.5, 0.5, 1.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        let unreduced = Sharding::new(mesh, vec![ShardingDimension::Replicated, ShardingDimension::Replicated])
            .unwrap()
            .with_unreduced_axes(["k"])
            .unwrap();
        assert_eq!(
            lhs.dot_with_configuration(&rhs, &DotDimensionNumbers::matmul(), Some(DataType::F32), Some(&unreduced)),
            Ok(Array::from_elements(
                plain_array(&[2, 2]).with_sharding(unreduced).unwrap(),
                &[1.0f32, 1.25, 2.5, 2.75]
            )
            .unwrap()),
        );
    }

    #[test]
    fn test_dot_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = DotOperation::new(DotDimensionNumbers::inner_product()),
            inputs = [Array::vector(vec![2.0, 3.0]).unwrap(), Array::vector(vec![5.0, 7.0]).unwrap()],
            expected = Array::scalar(31.0).unwrap(),
        );
    }

    #[test]
    fn test_dot_batching() {
        check_operation_batching!(
            @exact,
            operation = DotOperation::new(DotDimensionNumbers::inner_product()),
            axis_size = 2,
            cases = [
                {
                    inputs = [
                        (@mapped(axis = 0), Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap()),
                        (@replicated, Array::vector(vec![5.0, 7.0]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![19.0, 43.0]).unwrap())],
                },
                {
                    inputs = [
                        (@replicated, Array::vector(vec![5.0, 7.0]).unwrap()),
                        (@mapped(axis = 1), Array::matrix(2, 2, vec![1.0, 3.0, 2.0, 4.0]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![19.0, 43.0]).unwrap())],
                },
                {
                    inputs = [
                        (@mapped(axis = 0), Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap()),
                        (@mapped(axis = 1), Array::matrix(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![19.0, 50.0]).unwrap())],
                },
                {
                    inputs = [
                        (@replicated, Array::vector(vec![2.0, 3.0]).unwrap()),
                        (@replicated, Array::vector(vec![5.0, 7.0]).unwrap()),
                    ],
                    outputs = [(@replicated, Array::scalar(31.0).unwrap())],
                },
            ],
        );
    }

    #[test]
    fn test_dot_batching_accumulation_type() {
        // Batching lifts the dimension numbers while carrying the accumulation type, so the products of each item still
        // accumulate at the widened type. Both batch items repeat the same exactly representable `f8e4m3fn` values.
        let batched_type = ArrayType::new_static(DataType::F8E4M3FN, [2, 2, 2]);
        check_operation_batching!(
            @exact,
            operation = DotOperation::matmul().with_accumulation_type(DataType::F32),
            axis_size = 2,
            cases = [{
                inputs = [
                    (
                        @mapped(axis = 0),
                        Array::from_elements::<f8e4m3fn>(
                            batched_type.clone(),
                            &[0.5, 1.0, 1.5, 2.0, 0.5, 1.0, 1.5, 2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
                        )
                        .unwrap(),
                    ),
                    (
                        @mapped(axis = 0),
                        Array::from_elements::<f8e4m3fn>(
                            batched_type,
                            &[1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
                        )
                        .unwrap(),
                    ),
                ],
                outputs = [(
                    @mapped(axis = 0),
                    Array::from_elements::<f32>(
                        ArrayType::new_static(DataType::F32, [2, 2, 2]),
                        &[1.0, 1.25, 2.5, 2.75, 1.0, 1.25, 2.5, 2.75],
                    )
                    .unwrap(),
                )],
            }],
        );

        let input_type = ArrayType::new_static(DataType::F8E4M3FN, [2, 2]);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let lhs_input = builder.add_input(input_type.clone());
        let rhs_input = builder.add_input(input_type);
        let output = builder
            .add_instruction(
                DotOperation::matmul().with_accumulation_type(DataType::F32),
                vec![],
                vec![lhs_input, rhs_input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let lifted = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(
            lifted.to_string(),
            indoc! {"
                lambda %0:f8e4m3fn[2, 2, 2], %1:f8e4m3fn[2, 2, 2] .
                let %2:f32[2, 2, 2] = dot [
                    dimensions=(lhs_contracting=[2], rhs_contracting=[1], lhs_batching=[0], rhs_batching=[0]),
                    accumulation_type=f32,
                ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dot_batching_accumulation_type_and_output_sharding() {
        // Batching lifts the dimension numbers while carrying both the accumulation type and the requested output
        // sharding, which gains the placement of the inserted batch axis.
        let input_type = ArrayType::new_static(DataType::F8E4M3FN, [2, 2]);
        let mesh = test_mesh();
        let sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["m"]), ShardingDimension::Replicated])
                .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let lhs_input = builder.add_input(input_type.clone());
        let rhs_input = builder.add_input(input_type);
        let output = builder
            .add_instruction(
                DotOperation::matmul().with_accumulation_type(DataType::F32).with_output_sharding(sharding),
                vec![],
                vec![lhs_input, rhs_input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let lifted_sharding = Sharding::new(
            mesh,
            vec![ShardingDimension::Replicated, ShardingDimension::sharded(["m"]), ShardingDimension::Replicated],
        )
        .unwrap();
        let lifted = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(
            lifted.to_string(),
            formatdoc! {"
                lambda %0:f8e4m3fn[2, 2, 2], %1:f8e4m3fn[2, 2, 2] .
                let %2:f32[2, 2, 2][sharding={lifted_sharding}] = dot [
                    dimensions=(lhs_contracting=[2], rhs_contracting=[1], lhs_batching=[0], rhs_batching=[0]),
                    accumulation_type=f32,
                    output_sharding={lifted_sharding},
                ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dot_batching_output_sharding() -> Result<(), ProgramError> {
        let mesh = test_mesh();
        let output_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["m"]), ShardingDimension::sharded(["n"])])
                .unwrap();
        let operation = DotOperation::matmul().with_output_sharding(output_sharding.clone());

        // Two replicated inputs keep the original operation, including its requested output sharding.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(left, right)| {
                let context = BatchingContext::new(left.context().clone(), 2);
                let mut outputs = operation.batch(
                    &context,
                    &EmptyRegionDriver,
                    &[ArrayBatch::replicated(left), ArrayBatch::replicated(right)],
                )?;
                Ok(outputs.remove(0).into_value())
            },
            (plain_array(&[4, 8]), plain_array(&[8, 16])),
        )?;
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda %0:f32[4, 8], %1:f32[8, 16] .
                let %2:f32[4, 16][sharding={output_sharding}] = dot [
                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    output_sharding={output_sharding},
                ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // Two mapped inputs lift the requested output sharding with the sharding of their mapped axes.
        let batch_sharding =
            vec![ShardingDimension::sharded(["b"]), ShardingDimension::Replicated, ShardingDimension::Replicated];
        let lifted_sharding = Sharding::new(
            mesh.clone(),
            vec![
                ShardingDimension::sharded(["b"]),
                ShardingDimension::sharded(["m"]),
                ShardingDimension::sharded(["n"]),
            ],
        )
        .unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(left, right)| {
                let context = BatchingContext::new(left.context().clone(), 2)
                    .with_axis_sharding(ShardingDimension::sharded(["b"]));
                let mut outputs = operation.batch(
                    &context,
                    &EmptyRegionDriver,
                    &[ArrayBatch::new(left, BatchAxis::new(0))?, ArrayBatch::new(right, BatchAxis::new(0))?],
                )?;
                Ok(outputs.remove(0).into_value())
            },
            (
                sharded_array(&mesh, &[2, 4, 8], batch_sharding.clone()),
                sharded_array(&mesh, &[2, 8, 16], batch_sharding.clone()),
            ),
        )?;
        let left_sharding = Sharding::new(mesh, batch_sharding).unwrap();
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda %0:f32[2, 4, 8][sharding={left_sharding}], %1:f32[2, 8, 16][sharding={left_sharding}] .
                let %2:f32[2, 4, 16][sharding={lifted_sharding}] = dot [
                    dimensions=(lhs_contracting=[2], rhs_contracting=[1], lhs_batching=[0], rhs_batching=[0]),
                    output_sharding={lifted_sharding},
                ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_dot_batching_preserves_materialized_batch_placement() -> Result<(), ProgramError> {
        // Materializing the mapped axis on the replicated right-hand side places it like the mapped left-hand side.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let left_sharding = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated, ShardingDimension::Replicated],
        )
        .unwrap();
        let right_sharding = Sharding::replicated(mesh, 2);
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(left, right)| {
                let context = BatchingContext::new(left.context().clone(), 2)
                    .with_axis_sharding(ShardingDimension::sharded(["x"]));
                let mut outputs = DotOperation::matmul().batch(
                    &context,
                    &EmptyRegionDriver,
                    &[ArrayBatch::new(left, BatchAxis::new(0))?, ArrayBatch::replicated(right)],
                )?;
                Ok(outputs.remove(0).into_value())
            },
            (
                ArrayType::new_static(DataType::F64, [2, 2, 2]).with_sharding(left_sharding.clone()).unwrap(),
                ArrayType::new_static(DataType::F64, [2, 1]).with_sharding(right_sharding.clone()).unwrap(),
            ),
        )?;
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda %0:f64[2, 2, 2][sharding={left_sharding}], %1:f64[2, 1][sharding={right_sharding}] .
                let %2:f64[2, 2, 1][sharding={left_sharding}] = broadcast [
                    output_type=f64[2, 2, 1][sharding={left_sharding}],
                    output_axes=[1, 2],
                ] %1
                    %3:f64[2, 2, 1][sharding={left_sharding}] = dot [
                        dimensions=(lhs_contracting=[2], rhs_contracting=[1], lhs_batching=[0], rhs_batching=[0]),
                    ] %0 %2
                in (%3)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_dot_batching_aligns_materialized_batch_placement_with_manual_variation() -> Result<(), ProgramError> {
        // Materializing the mapped axis on the replicated right-hand side gives it the batch placement on `y`, which
        // adds no variation, so `parallel_vary` aligns it with the left-hand side, which varies along `x`.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let left_sharding = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::sharded(["y"]), ShardingDimension::Replicated, ShardingDimension::Replicated],
        )
        .unwrap()
        .with_varying_manual_axes(["x"])
        .unwrap();
        let right_sharding = Sharding::replicated(mesh.clone(), 2);
        let materialized_sharding = Sharding::new(
            mesh,
            vec![ShardingDimension::sharded(["y"]), ShardingDimension::Replicated, ShardingDimension::Replicated],
        )
        .unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(left, right)| {
                let context = BatchingContext::new(left.context().clone(), 2)
                    .with_axis_sharding(ShardingDimension::sharded(["y"]));
                let mut outputs = DotOperation::matmul().batch(
                    &context,
                    &EmptyRegionDriver,
                    &[ArrayBatch::new(left, BatchAxis::new(0))?, ArrayBatch::replicated(right)],
                )?;
                Ok(outputs.remove(0).into_value())
            },
            (
                ArrayType::new_static(DataType::F64, [2, 2, 2]).with_sharding(left_sharding.clone()).unwrap(),
                ArrayType::new_static(DataType::F64, [2, 1]).with_sharding(right_sharding.clone()).unwrap(),
            ),
        )?;
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda %0:f64[2, 2, 2][sharding={left_sharding}], %1:f64[2, 1][sharding={right_sharding}] .
                let %2:f64[2, 2, 1][sharding={materialized_sharding}] = broadcast [
                    output_type=f64[2, 2, 1][sharding={materialized_sharding}],
                    output_axes=[1, 2],
                ] %1
                    %3:f64[2, 2, 1][sharding={left_sharding}] = parallel_vary [axis_name=\"x\"] %2
                    %4:f64[2, 2, 1][sharding={left_sharding}] = dot [
                        dimensions=(lhs_contracting=[2], rhs_contracting=[1], lhs_batching=[0], rhs_batching=[0]),
                    ] %0 %3
                in (%4)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_dot_batching_aligns_mapped_manual_variation() -> Result<(), ProgramError> {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let left_sharding = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated, ShardingDimension::Replicated],
        )
        .unwrap()
        .with_varying_manual_axes(["x"])
        .unwrap();
        let right_sharding = Sharding::replicated(mesh, 3);
        let varying_right_sharding = right_sharding.clone().with_varying_manual_axes(["x"]).unwrap();
        // Both inputs already have mapped axes, so axis materialization adds no variation. The [`Dot`] capability owns
        // the transition on the invariant input.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(left, right)| {
                let context = BatchingContext::new(left.context().clone(), 2)
                    .with_axis_sharding(ShardingDimension::sharded(["x"]));
                let mut outputs = DotOperation::matmul().batch(
                    &context,
                    &EmptyRegionDriver,
                    &[ArrayBatch::new(left, BatchAxis::new(0))?, ArrayBatch::new(right, BatchAxis::new(0))?],
                )?;
                Ok(outputs.remove(0).into_value())
            },
            (
                ArrayType::new_static(DataType::F32, [2, 2, 2]).with_sharding(left_sharding.clone()).unwrap(),
                ArrayType::new_static(DataType::F32, [2, 2, 1]).with_sharding(right_sharding.clone()).unwrap(),
            ),
        )?;
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda %0:f32[2, 2, 2][sharding={left_sharding}], %1:f32[2, 2, 1][sharding={right_sharding}] .
                let %2:f32[2, 2, 1][sharding={varying_right_sharding}] = parallel_vary [axis_name=\"x\"] %1
                    %3:f32[2, 2, 1][sharding={left_sharding}] = dot [
                        dimensions=(lhs_contracting=[2], rhs_contracting=[1], lhs_batching=[0], rhs_batching=[0]),
                    ] %0 %2
                in (%3)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_dot_batching_through_batch_transform() {
        // Mapping axis zero of a [3, 4] matrix gives three vector inner products. Each logical item contracts its only
        // axis, while the physical operation preserves the leading mapped axis. The batch items contract to
        // [1, 2, 3, 4]·[1, 2, 3, 4] = 30, [5, 6, 7, 8]·[5, 6, 7, 8] = 174, and
        // [9, 10, 11, 12]·[9, 10, 11, 12] = 446.
        let rows = Array::matrix(3, 4, (1..=12).map(|value| value as f64).collect()).unwrap();
        assert_eq!(
            batch(
                |row| Ok(row.dot(&row, &DotDimensionNumbers::inner_product())?),
                rows,
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            ),
            Ok(Array::vector(vec![30.0, 174.0, 446.0]).unwrap()),
        );
    }

    #[test]
    fn test_dot_batching_validates_original_dimensions() {
        let input = ArrayBatch::new(Array::matrix(2, 1, vec![1.0f32, 2.0]).unwrap(), BatchAxis::new(0)).unwrap();
        let operation = DotOperation::new(DotDimensionNumbers::new(vec![usize::MAX], vec![0], vec![], vec![]));
        assert_eq!(
            operation.batch(
                &BatchingContext::new(EagerContext::<Array>::new(), 2),
                &EmptyRegionDriver,
                &[input.clone(), input],
            ),
            Err(BatchingError::Type(TypeError::invalid(format!(
                "`{DOT_OPERATION_NAME}` left-hand side dimension index out of bounds",
            )))),
        );
    }

    #[test]
    fn test_dot_batching_validates_mapped_extents() {
        let context = TracingContext::<ArrayType, ArrayOperation<ArrayType>>::new();
        let batching_context = BatchingContext::new(context.clone(), 2);
        let operation = DotOperation::new(DotDimensionNumbers::inner_product());

        // Two mapped inputs whose mapped axes carry different statically known extents cannot describe one batch.
        assert!(matches!(
            operation.batch(
                &batching_context,
                &EmptyRegionDriver,
                &[
                    ArrayBatch::new(context.input(plain_array(&[2, 3])), BatchAxis::new(0)).unwrap(),
                    ArrayBatch::new(context.input(plain_array(&[3, 3])), BatchAxis::new(0)).unwrap(),
                ],
            ),
            Err(BatchingError::MismatchedBatchSizes { expected: 2, actual: 3 }),
        ));

        // Dynamic extents are compared by variable identity, so two independently declared variables are two
        // independent extents even when their bounds agree.
        let batch_variable = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5)).unwrap());
        let other_variable = DimensionVariable::new("other", DimensionBounds::new(1, Some(5)).unwrap());
        assert!(matches!(
            operation.batch(
                &batching_context,
                &EmptyRegionDriver,
                &[
                    ArrayBatch::new(
                        context.input(ArrayType::new(
                            DataType::F32,
                            Shape::new(vec![Dimension::Dynamic(batch_variable), Dimension::Static(3)]),
                        )),
                        BatchAxis::new(0),
                    )
                    .unwrap(),
                    ArrayBatch::new(
                        context.input(ArrayType::new(
                            DataType::F32,
                            Shape::new(vec![Dimension::Dynamic(other_variable), Dimension::Static(3)]),
                        )),
                        BatchAxis::new(0),
                    )
                    .unwrap(),
                ],
            ),
            Err(BatchingError::MisalignedBatchAxes { message })
                if message == format!("`{DOT_OPERATION_NAME}` inputs map different batch extents `batch` and `other`"),
        ));
    }

    #[test]
    fn test_dot_batching_dynamic_mapped_extent() -> Result<(), ProgramError> {
        // A dense contraction is batched under a dynamic mapped extent as well. The mapped input keeps the dynamic
        // batch dimension it already carries, and the replicated right-hand side vector gains one through the policy's
        // dynamic broadcast, whose inserted axis is grounded by the transform's first-class extent value (read off the
        // mapped input's own axis). The lifted dot then contracts the trailing axis under a dynamic batching dimension.
        let trace = ArrayIrContext::new();
        let batch_variable = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5))?);
        let rows = trace.input(
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(batch_variable.clone()), Dimension::Static(3)]),
            )
            .into(),
        );
        let vector = trace.input(ArrayType::new_static(DataType::F32, [3]).into());
        let output = batch(
            |(row, vector)| {
                let row = ValueProjection::<ArrayType>::into_projected(row)?;
                let vector = ValueProjection::<ArrayType>::into_projected(vector)?;
                let mut outputs = row.domain().bind(
                    DotOperation::new(DotDimensionNumbers::inner_product()),
                    vec![],
                    &[row, vector],
                )?;
                Ok(outputs.remove(0).into_value())
            },
            (rows, vector),
            (BatchAxis::new(0), BatchAxis::replicated()),
            BatchAxis::new(0),
            None,
        )?;
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, ArrayIrValue<Array>>(
            vec![output.atom_id()?],
            vec![Placeholder, Placeholder],
            Placeholder,
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[batch, 3], %1:f32[3] .
                let %2:dimension<batch ∈ [1, 5)> = dimension_size [axis=0] %0
                    %3:dimension<3> = constant [value=3]
                    %4:f32[batch, 3] = broadcast [output_axes=[1]] %1 %2 %3
                    %5:f32[batch] = dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[0], rhs_batching=[0]),
                    ] %0 %4
                in (%5)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])?),
                ArrayIrValue::Array(Array::vector(vec![10.0f32, 100.0, 1000.0])?),
            ])?,
            ArrayIrValue::Array(Array::vector(vec![3210.0f32, 6540.0])?),
        );

        // Two mapped inputs that share one dynamic mapped extent describe the same batch. The transform reads both
        // extents and compares them, which folds to a constant because they are the same variable, and the lifted
        // contraction is staged with that dimension on its batching dimension.
        let trace = ArrayIrContext::new();
        let row_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch_variable), Dimension::Static(3)]));
        let left = trace.input(row_type.clone().into());
        let right = trace.input(row_type.into());
        let output = batch(
            |(left, right)| {
                let left = ValueProjection::<ArrayType>::into_projected(left)?;
                let right = ValueProjection::<ArrayType>::into_projected(right)?;
                let mut outputs = left.domain().bind(
                    DotOperation::new(DotDimensionNumbers::inner_product()),
                    vec![],
                    &[left, right],
                )?;
                Ok(outputs.remove(0).into_value())
            },
            (left, right),
            (BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        )?;
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, ArrayIrValue<Array>>(
            vec![output.atom_id()?],
            vec![Placeholder, Placeholder],
            Placeholder,
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[batch, 3], %1:f32[batch, 3] .
                let %2:dimension<batch ∈ [1, 5)> = dimension_size [axis=0] %0
                    %3:dimension<batch ∈ [1, 5)> = dimension_size [axis=0] %1
                    %4:bool[] = const true
                    %5:f32[batch] = dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[0], rhs_batching=[0]),
                    ] %0 %1
                in (%5)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_dot_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-5, epsilon = 1e-7),
            operation = DotOperation::new(DotDimensionNumbers::inner_product()),
            cases = [{
                primals = [Array::vector(vec![2.0, 3.0]).unwrap(), Array::vector(vec![5.0, 7.0]).unwrap()],
                tangents = [Array::vector(vec![1.0, 2.0]).unwrap(), Array::vector(vec![3.0, 4.0]).unwrap()],
                primal_outputs = [Array::scalar(31.0).unwrap()],
                tangent_outputs = [Array::scalar(37.0).unwrap()],
                jvp = indoc! {"
                    lambda %0:f64[2], %1:f64[2], %2:f64[2], %3:f64[2] .
                    let %4:f64[] = dot [
                        dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    ] %0 %1
                        %5:f64[] = dot [
                            dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        ] %2 %1
                        %6:f64[] = dot [
                            dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        ] %0 %3
                        %7:f64[] = add %5 %6
                    in (%4, %7)
                "},
            }],
        );

        // A structural zero tangent drops its product rule term. Here, the constant right-hand side has a zero tangent.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let lhs_input = builder.add_input(ArrayType::new_static(DataType::F64, [2]));
        let rhs_constant = builder.add_constant(Array::vector(vec![5.0, 7.0]).unwrap());
        let output = builder
            .add_instruction(
                DotOperation::new(DotDimensionNumbers::inner_product()),
                vec![],
                vec![lhs_input, rhs_constant],
                None,
            )
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[2], %1:f64[2] .
                let %2:f64[2] = const [5.0, 7.0]
                    %3:f64[] = dot [
                        dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    ] %0 %2
                    %4:f64[] = dot [
                        dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    ] %1 %2
                in (%3, %4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dot_differentiation_jacobians() {
        let inputs = (Array::vector(vec![2.0, 3.0, 5.0]).unwrap(), Array::vector(vec![7.0, 11.0, 13.0]).unwrap());

        // Reverse mode batches the pullback's adjoint dots over the cotangents of the output coordinates.
        let jacobian = differentiate_at(inputs.clone())
            .jacobian_reverse(|(left, right)| Ok(left.dot(&right, &DotDimensionNumbers::inner_product())?))
            .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        let [left, right] = blocks.as_slice() else { unreachable!() };
        assert_eq!(left.value(), &Array::vector(vec![7.0, 11.0, 13.0]).unwrap());
        assert_eq!(right.value(), &Array::vector(vec![2.0, 3.0, 5.0]).unwrap());

        // Forward mode batches the basis tangents of the input coordinates through the dot pushforward.
        let jacobian = differentiate_at(inputs)
            .jacobian_forward(|(left, right)| Ok(left.dot(&right, &DotDimensionNumbers::inner_product())?))
            .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        let [left, right] = blocks.as_slice() else { unreachable!() };
        assert_eq!(left.value(), &Array::vector(vec![7.0, 11.0, 13.0]).unwrap());
        assert_eq!(right.value(), &Array::vector(vec![2.0, 3.0, 5.0]).unwrap());
    }

    #[test]
    fn test_dot_differentiation_widened_tangents() {
        // The tangents of `f8e8m0fnu` primals are widened to `f32`, so each product rule term converts its primal
        // factor to the tangent element type before the contraction.
        let lhs_type = ArrayType::new_static(DataType::F8E8M0FNU, [3, 2]);
        let rhs_type = ArrayType::new_static(DataType::F8E8M0FNU, [2, 1]);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let lhs = builder.add_input(lhs_type.clone());
        let rhs = builder.add_input(rhs_type.clone());
        let output = builder.add_instruction(DotOperation::matmul(), vec![], vec![lhs, rhs], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f8e8m0fnu[3, 2], %1:f8e8m0fnu[2, 1] .
                let %2:f8e8m0fnu[3, 1] = dot [
                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f8e8m0fnu[3, 2], %1:f8e8m0fnu[2, 1], %2:f32[3, 2], %3:f32[2, 1] .
                let %4:f8e8m0fnu[3, 1] = dot [
                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                ] %0 %1
                    %5:f32[2, 1] = convert_element_type [data_type=f32] %1
                    %6:f32[3, 1] = dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    ] %2 %5
                    %7:f32[3, 2] = convert_element_type [data_type=f32] %0
                    %8:f32[3, 1] = dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    ] %7 %3
                    %9:f32[3, 1] = add %6 %8
                in (%4, %9)
            "}
            .trim_end(),
        );
        assert_eq!(
            jvp.interpret(vec![
                Array::from_elements(lhs_type, &[f8e8m0fnu::from_bits(127); 6]).unwrap(),
                Array::from_elements(rhs_type, &[f8e8m0fnu::from_bits(127); 2]).unwrap(),
                Array::from_elements(ArrayType::new_static(DataType::F32, [3, 2]), &[1.0f32; 6]).unwrap(),
                Array::from_elements(ArrayType::new_static(DataType::F32, [2, 1]), &[1.0f32; 2]).unwrap(),
            ]),
            Ok(vec![
                Array::from_elements(
                    ArrayType::new_static(DataType::F8E8M0FNU, [3, 1]),
                    &[f8e8m0fnu::from_bits(128); 3],
                )
                .unwrap(),
                Array::from_elements(ArrayType::new_static(DataType::F32, [3, 1]), &[4.0f32; 3]).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_dot_differentiation_accumulation_type() {
        // Forward mode differentiation stages tangent dots with the accumulation type over tangents of the input type,
        // so the output tangent lives at the accumulation type exactly like the primal output. Every value below is
        // exactly representable in `f8e4m3fn` and every product sum is exact in `f32`.
        let input_type = ArrayType::new_static(DataType::F8E4M3FN, [2, 2]);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let lhs_input = builder.add_input(input_type.clone());
        let rhs_input = builder.add_input(input_type.clone());
        let output = builder
            .add_instruction(
                DotOperation::matmul().with_accumulation_type(DataType::F32),
                vec![],
                vec![lhs_input, rhs_input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f8e4m3fn[2, 2], %1:f8e4m3fn[2, 2], %2:f8e4m3fn[2, 2], %3:f8e4m3fn[2, 2] .
                let %4:f32[2, 2] = dot [
                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    accumulation_type=f32,
                ] %0 %1
                    %5:f32[2, 2] = dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        accumulation_type=f32,
                    ] %2 %1
                    %6:f32[2, 2] = dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        accumulation_type=f32,
                    ] %0 %3
                    %7:f32[2, 2] = add %5 %6
                in (%4, %7)
            "}
            .trim_end(),
        );

        // The tangent is `d_lhs · rhs + lhs · d_rhs = [[1.5, 1.5], [1.5, 1.5]] + [[0.75, 0.75], [1.75, 1.75]]`.
        assert_eq!(
            jvp.interpret(vec![
                Array::from_elements::<f8e4m3fn>(
                    input_type.clone(),
                    &[0.5, 1.0, 1.5, 2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
                )
                .unwrap(),
                Array::from_elements::<f8e4m3fn>(
                    input_type.clone(),
                    &[1.0, 0.5, 0.5, 1.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
                )
                .unwrap(),
                Array::from_elements::<f8e4m3fn>(
                    input_type.clone(),
                    &[1.0, 1.0, 1.0, 1.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
                )
                .unwrap(),
                Array::from_elements::<f8e4m3fn>(
                    input_type,
                    &[0.5, 0.5, 0.5, 0.5].map(|value| f8e4m3fn::from_f64(value).unwrap()),
                )
                .unwrap(),
            ]),
            Ok(vec![
                Array::matrix(2, 2, vec![1.0f32, 1.25, 2.5, 2.75]).unwrap(),
                Array::matrix(2, 2, vec![2.25f32, 2.25, 3.25, 3.25]).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_dot_differentiation_accumulation_type_and_output_sharding() {
        // Forward mode differentiation stages tangent dots with both the accumulation type and the requested output
        // sharding, so the output tangent has exactly the type of the primal output.
        let input_type = ArrayType::new_static(DataType::F8E4M3FN, [2, 2]);
        let sharding =
            Sharding::new(test_mesh(), vec![ShardingDimension::sharded(["m"]), ShardingDimension::Replicated]).unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let lhs_input = builder.add_input(input_type.clone());
        let rhs_input = builder.add_input(input_type);
        let output = builder
            .add_instruction(
                DotOperation::matmul().with_accumulation_type(DataType::F32).with_output_sharding(sharding.clone()),
                vec![],
                vec![lhs_input, rhs_input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.jvp().unwrap().to_string(),
            formatdoc! {"
                lambda %0:f8e4m3fn[2, 2], %1:f8e4m3fn[2, 2], %2:f8e4m3fn[2, 2], %3:f8e4m3fn[2, 2] .
                let %4:f32[2, 2][sharding={sharding}] = dot [
                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    accumulation_type=f32,
                    output_sharding={sharding},
                ] %0 %1
                    %5:f32[2, 2][sharding={sharding}] = dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        accumulation_type=f32,
                        output_sharding={sharding},
                    ] %2 %1
                    %6:f32[2, 2][sharding={sharding}] = dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        accumulation_type=f32,
                        output_sharding={sharding},
                    ] %0 %3
                    %7:f32[2, 2][sharding={sharding}] = add %5 %6
                in (%4, %7)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dot_transposition() {
        let left = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let right = Array::matrix(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let cotangent = Array::matrix(2, 2, vec![1.0, -2.0, 0.5, 3.0]).unwrap();
        check_operation_transposition!(
            @exact,
            operation = DotOperation::matmul(),
            cases = [
                {
                    inputs = [
                        (@known, left),
                        (@linear(type = right.r#type().into_owned())),
                    ],
                    output_cotangents = [cotangent.clone()],
                    input_cotangents = [Array::matrix(3, 2, vec![3.0, 10.0, 4.5, 11.0, 6.0, 12.0]).unwrap()],
                    pullback = indoc! {"
                        lambda %0:f64[2, 2], %1:f64[2, 3] .
                        let %2:f64[3, 2] = dot [
                            dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        ] %1 %0
                        in (%2)
                    "},
                },
                {
                    inputs = [
                        (@linear(type = ArrayType::new_static(DataType::F64, [2, 3]))),
                        (@known, right),
                    ],
                    output_cotangents = [cotangent],
                    input_cotangents = [Array::matrix(2, 3, vec![-9.0, -11.0, -13.0, 27.5, 34.5, 41.5]).unwrap()],
                    pullback = indoc! {"
                        lambda %0:f64[2, 2], %1:f64[3, 2] .
                        let %2:f64[2, 3] = dot [
                            dimensions=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[], rhs_batching=[]),
                        ] %0 %1
                        in (%2)
                    "},
                },
            ],
        );
    }

    #[test]
    fn test_dot_transposition_accumulation_type() {
        let input_type = ArrayType::new_static(DataType::F8E4M3FN, [2, 2]);
        // The transpose rule contracts the adjoint at the accumulation type and converts the result back to the linear
        // input's `f8e4m3fn` cotangent representation. With an identity output cotangent, the adjoint of the linear
        // right-hand side is exactly `lhsᵀ`.
        check_operation_transposition!(
            @exact,
            operation = DotOperation::matmul().with_accumulation_type(DataType::F32),
            cases = [{
                inputs = [
                    (@known, Array::from_elements::<f8e4m3fn>(
                        input_type.clone(),
                        &[0.5, 1.0, 1.5, 2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
                    ).unwrap()),
                    (@linear(type = input_type.clone())),
                ],
                output_cotangents = [Array::matrix(2, 2, vec![1.0f32, 0.0, 0.0, 1.0]).unwrap()],
                input_cotangents = [Array::from_elements::<f8e4m3fn>(
                    input_type,
                    &[0.5, 1.5, 1.0, 2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
                ).unwrap()],
                pullback = indoc! {"
                    lambda %0:f32[2, 2], %1:f8e4m3fn[2, 2] .
                    let %2:f32[2, 2] = convert_element_type [data_type=f32] %1
                        %3:f32[2, 2] = dot [
                            dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        ] %2 %0
                        %4:f8e4m3fn[2, 2] = convert_element_type [data_type=f8e4m3fn] %3
                    in (%4)
                "},
            }],
        );
    }

    #[test]
    fn test_dot_transposition_accumulation_type_and_output_sharding() {
        // The transpose rule contracts the adjoint at the accumulation type, requests the placement of the linear
        // input's cotangent, and converts the result back to its `f8e4m3fn` element type. With an identity output
        // cotangent, the adjoint of the linear right-hand side is exactly `lhsᵀ`.
        let sharding =
            Sharding::new(test_mesh(), vec![ShardingDimension::Replicated, ShardingDimension::sharded(["n"])]).unwrap();
        let rhs_type = ArrayType::new_static(DataType::F8E4M3FN, [2, 2]).with_sharding(sharding.clone()).unwrap();
        check_operation_transposition!(
            @exact,
            operation = DotOperation::matmul()
                .with_accumulation_type(DataType::F32)
                .with_output_sharding(sharding.clone()),
            cases = [{
                inputs = [
                    (@known, Array::from_elements::<f8e4m3fn>(
                        ArrayType::new_static(DataType::F8E4M3FN, [2, 2]),
                        &[0.5, 1.0, 1.5, 2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
                    ).unwrap()),
                    (@linear(type = rhs_type.clone())),
                ],
                output_cotangents = [Array::from_elements::<f32>(
                    ArrayType::new_static(DataType::F32, [2, 2]).with_sharding(sharding.clone()).unwrap(),
                    &[1.0, 0.0, 0.0, 1.0],
                ).unwrap()],
                input_cotangents = [Array::from_elements::<f8e4m3fn>(
                    rhs_type,
                    &[0.5, 1.5, 1.0, 2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
                ).unwrap()],
                pullback = formatdoc! {"
                    lambda %0:f32[2, 2][sharding={sharding}], %1:f8e4m3fn[2, 2] .
                    let %2:f32[2, 2] = convert_element_type [data_type=f32] %1
                        %3:f32[2, 2][sharding={sharding}] = dot [
                            dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            output_sharding={sharding},
                        ] %2 %0
                        %4:f8e4m3fn[2, 2][sharding={sharding}] = convert_element_type [data_type=f8e4m3fn] %3
                    in (%4)
                "},
            }],
        );
    }

    #[test]
    fn test_dot_transposition_unrequested_cotangent() {
        // An unrequested linear input stages no adjoint and keeps a structural zero cotangent.
        let mut context = TranspositionContext::new(TracingContext::<Array, ArrayOperation<Array>>::new());
        let right = context.input(ArrayType::new_static(DataType::F64, [3, 2]));
        let seed = context.input(ArrayType::new_static(DataType::F64, [2, 2]));
        let inputs = [PartialValue::Unknown(ArrayType::new_static(DataType::F64, [2, 3])), PartialValue::Known(right)];
        let accumulators = context.cotangent_accumulators(&inputs, &[false, false]).unwrap();
        DotOperation::matmul()
            .transpose(&mut context, &EmptyRegionDriver, &inputs, &[MaybeZero::Value(seed)], &accumulators)
            .unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[3, 2], %1:f64[2, 2] .
                in ()
            "}
            .trim_end(),
        );
        assert!(context.take_cotangents(&accumulators).unwrap().iter().all(MaybeZero::is_zero));
    }

    #[test]
    fn test_dot_transposition_structural_zero() {
        // A structural zero output cotangent stages no adjoint and leaves the requested cotangent zero.
        let mut context = TranspositionContext::new(TracingContext::<Array, ArrayOperation<Array>>::new());
        let right = context.input(ArrayType::new_static(DataType::F64, [3, 2]));
        let inputs = [PartialValue::Unknown(ArrayType::new_static(DataType::F64, [2, 3])), PartialValue::Known(right)];
        let accumulators = context.cotangent_accumulators(&inputs, &[true, false]).unwrap();
        DotOperation::matmul()
            .transpose(
                &mut context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Zero(ArrayType::new_static(DataType::F64, [2, 2]))],
                &accumulators,
            )
            .unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![], vec![Placeholder], vec![])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[3, 2] .
                in ()
            "}
            .trim_end(),
        );
        assert!(context.take_cotangents(&accumulators).unwrap().iter().all(MaybeZero::is_zero));
    }

    #[test]
    fn test_dot_transposition_known_inputs() {
        // A dot with two known inputs is not part of the linear map, so transposition stages nothing.
        let mut context = TranspositionContext::new(TracingContext::<Array, ArrayOperation<Array>>::new());
        let left = context.input(ArrayType::new_static(DataType::F64, [2, 3]));
        let right = context.input(ArrayType::new_static(DataType::F64, [3, 2]));
        let seed = context.input(ArrayType::new_static(DataType::F64, [2, 2]));
        let inputs = [PartialValue::Known(left), PartialValue::Known(right)];
        let accumulators = context.cotangent_accumulators(&inputs, &[false, false]).unwrap();
        DotOperation::matmul()
            .transpose(&mut context, &EmptyRegionDriver, &inputs, &[MaybeZero::Value(seed)], &accumulators)
            .unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![], vec![Placeholder; 3], vec![])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 3], %1:f64[3, 2], %2:f64[2, 2] .
                in ()
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dot_transposition_rejects_two_linear_inputs() {
        let mut context = TranspositionContext::new(TracingContext::<Array, ArrayOperation<Array>>::new());
        let seed = context.input(ArrayType::new_static(DataType::F64, [2, 2]));
        let inputs = [
            PartialValue::Unknown(ArrayType::new_static(DataType::F64, [2, 3])),
            PartialValue::Unknown(ArrayType::new_static(DataType::F64, [3, 2])),
        ];
        let accumulators = context.cotangent_accumulators(&inputs, &[true, true]).unwrap();
        assert_eq!(
            DotOperation::matmul().transpose(
                &mut context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Value(seed)],
                &accumulators,
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation {
                message: format!("bilinear `{DOT_OPERATION_NAME}` with two linear inputs cannot be transposed"),
            })),
        );
    }

    #[test]
    fn test_dot_transposition_restores_input_axis_order() {
        let right = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let seed = Array::matrix(2, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
        // Equal extents conceal the wrong permutation in type checks, so pin the element order too.
        check_operation_transposition!(
            @exact,
            operation = DotOperation::new(DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![])),
            cases = [{
                inputs = [(@linear(type = ArrayType::new_static(DataType::F64, [2, 2]))), (@known, right)],
                output_cotangents = [seed],
                input_cotangents = [Array::matrix(2, 2, vec![1.0, 2.0, 4.0, 5.0]).unwrap()],
            }],
        );

        check_operation_transposition!(
            @exact,
            operation = DotOperation::new(DotDimensionNumbers::new(vec![1], vec![1], vec![], vec![])),
            cases = [{
                inputs = [
                    (@known, Array::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
                    (@linear(type = ArrayType::new_static(DataType::F64, [4, 2]))),
                ],
                output_cotangents = [Array::matrix(
                    3, 4, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                ).unwrap()],
                input_cotangents = [Array::matrix(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0]).unwrap()],
            }],
        );
    }

    #[test]
    fn test_dot_transposition_multiple_contracting_axes() {
        let input_type = plain_array(&[2, 2, 2, 1]);
        let seed = Array::from_elements::<f32>(plain_array(&[2, 1, 1]), &[3.0, 5.0]).unwrap();

        // Both adjoints must sort contracting axes by their paired input axes and restore the linear input's order.
        // Pin the complete pullbacks as well as the gradients, because equal extents can hide a wrong permutation.
        check_operation_transposition!(
            @exact,
            operation = DotOperation::new(DotDimensionNumbers::new(vec![2, 0], vec![0, 2], vec![1], vec![1])),
            cases = [
                {
                    inputs = [
                        (@linear(type = input_type.clone())),
                        (@known, Array::from_elements::<f32>(
                            input_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                        ).unwrap()),
                    ],
                    output_cotangents = [seed.clone()],
                    input_cotangents = [Array::from_elements::<f32>(
                        input_type.clone(), &[3.0, 15.0, 15.0, 35.0, 6.0, 18.0, 20.0, 40.0],
                    ).unwrap()],
                    pullback = indoc! {"
                        lambda %0:f32[2, 1, 1], %1:f32[2, 2, 2, 1] .
                        let %2:f32[2, 1, 2, 2] = dot [
                            dimensions=(lhs_contracting=[2], rhs_contracting=[3], lhs_batching=[0], rhs_batching=[1]),
                        ] %0 %1
                            %3:f32[2, 2, 2, 1] = transpose [permutation=[3, 0, 2, 1]] %2
                        in (%3)
                    "},
                },
                {
                    inputs = [
                        (@known, Array::from_elements::<f32>(
                            input_type.clone(), &[11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                        ).unwrap()),
                        (@linear(type = input_type.clone())),
                    ],
                    output_cotangents = [seed],
                    input_cotangents = [Array::from_elements::<f32>(
                        input_type, &[33.0, 45.0, 65.0, 85.0, 36.0, 48.0, 70.0, 90.0],
                    ).unwrap()],
                    pullback = indoc! {"
                        lambda %0:f32[2, 1, 1], %1:f32[2, 2, 2, 1] .
                        let %2:f32[2, 2, 2, 1] = dot [
                            dimensions=(lhs_contracting=[3], rhs_contracting=[1], lhs_batching=[1], rhs_batching=[0]),
                        ] %1 %0
                            %3:f32[2, 2, 2, 1] = transpose [permutation=[2, 0, 1, 3]] %2
                        in (%3)
                    "},
                },
            ],
        );
    }

    #[test]
    fn test_dot_transposition_multiple_contracting_axes_and_sharding() {
        // Each adjoint requests the linear input's sharding permuted into the adjoint axis order, and the restoring
        // transpose then recovers the linear input's own sharding.
        let mesh = test_mesh();
        let input_sharding = Sharding::new(
            mesh.clone(),
            vec![
                ShardingDimension::sharded(["m"]),
                ShardingDimension::sharded(["b"]),
                ShardingDimension::Replicated,
                ShardingDimension::Replicated,
            ],
        )
        .unwrap();
        let seed_sharding = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::sharded(["b"]), ShardingDimension::Replicated, ShardingDimension::Replicated],
        )
        .unwrap();
        let left_adjoint_sharding = Sharding::new(
            mesh.clone(),
            vec![
                ShardingDimension::sharded(["b"]),
                ShardingDimension::Replicated,
                ShardingDimension::Replicated,
                ShardingDimension::sharded(["m"]),
            ],
        )
        .unwrap();
        let right_adjoint_sharding = Sharding::new(
            mesh,
            vec![
                ShardingDimension::sharded(["b"]),
                ShardingDimension::Replicated,
                ShardingDimension::sharded(["m"]),
                ShardingDimension::Replicated,
            ],
        )
        .unwrap();
        let input_type = plain_array(&[2, 2, 2, 1]).with_sharding(input_sharding.clone()).unwrap();
        let seed = Array::from_elements::<f32>(
            plain_array(&[2, 1, 1]).with_sharding(seed_sharding.clone()).unwrap(),
            &[3.0, 5.0],
        )
        .unwrap();

        check_operation_transposition!(
            @exact,
            operation = DotOperation::new(DotDimensionNumbers::new(vec![2, 0], vec![0, 2], vec![1], vec![1])),
            cases = [
                {
                    inputs = [
                        (@linear(type = input_type.clone())),
                        (@known, Array::from_elements::<f32>(
                            plain_array(&[2, 2, 2, 1]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                        ).unwrap()),
                    ],
                    output_cotangents = [seed.clone()],
                    input_cotangents = [Array::from_elements::<f32>(
                        input_type.clone(), &[3.0, 15.0, 15.0, 35.0, 6.0, 18.0, 20.0, 40.0],
                    ).unwrap()],
                    pullback = formatdoc! {"
                        lambda %0:f32[2, 1, 1][sharding={seed_sharding}], %1:f32[2, 2, 2, 1] .
                        let %2:f32[2, 1, 2, 2][sharding={left_adjoint_sharding}] = dot [
                            dimensions=(lhs_contracting=[2], rhs_contracting=[3], lhs_batching=[0], rhs_batching=[1]),
                            output_sharding={left_adjoint_sharding},
                        ] %0 %1
                            %3:f32[2, 2, 2, 1][sharding={input_sharding}] = transpose [permutation=[3, 0, 2, 1]] %2
                        in (%3)
                    "},
                },
                {
                    inputs = [
                        (@known, Array::from_elements::<f32>(
                            plain_array(&[2, 2, 2, 1]), &[11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                        ).unwrap()),
                        (@linear(type = input_type.clone())),
                    ],
                    output_cotangents = [seed],
                    input_cotangents = [Array::from_elements::<f32>(
                        input_type, &[33.0, 45.0, 65.0, 85.0, 36.0, 48.0, 70.0, 90.0],
                    ).unwrap()],
                    pullback = formatdoc! {"
                        lambda %0:f32[2, 1, 1][sharding={seed_sharding}], %1:f32[2, 2, 2, 1] .
                        let %2:f32[2, 2, 2, 1][sharding={right_adjoint_sharding}] = dot [
                            dimensions=(lhs_contracting=[3], rhs_contracting=[1], lhs_batching=[1], rhs_batching=[0]),
                            output_sharding={right_adjoint_sharding},
                        ] %1 %0
                            %3:f32[2, 2, 2, 1][sharding={input_sharding}] = transpose [permutation=[2, 0, 1, 3]] %2
                        in (%3)
                    "},
                },
            ],
        );
    }

    #[test]
    fn test_dot_array_ir() {
        // Array IR tracers stage the homogeneous dot product as an array member instruction.
        let vector_type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2]));
        let (_, program) = ArrayIrContext::trace(
            |inputs: Vec<Tracer<ArrayIrContext>>| inputs[0].dot(&inputs[1], &DotDimensionNumbers::inner_product()),
            vec![vector_type.clone(), vector_type],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[2] .
                let %2:f32[] = dot [
                    dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // Concrete array IR values contract their array members, while first-class dimension members are rejected.
        let left = ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0]).unwrap());
        let right = ArrayIrValue::Array(Array::vector(vec![3.0f32, 4.0]).unwrap());
        assert_eq!(
            left.dot(&right, &DotDimensionNumbers::inner_product()),
            Ok(ArrayIrValue::Array(Array::scalar(11.0f32).unwrap())),
        );
        let singleton = ArrayIrValue::Array(Array::vector(vec![3.0f32]).unwrap());
        assert_eq!(
            singleton.dot(&singleton, &DotDimensionNumbers::inner_product()),
            Ok(ArrayIrValue::Array(Array::scalar(9.0f32).unwrap())),
        );
        let empty = ArrayIrValue::Array(Array::vector(Vec::<f32>::new()).unwrap());
        assert_eq!(
            empty.dot(&empty, &DotDimensionNumbers::inner_product()),
            Ok(ArrayIrValue::Array(Array::scalar(0.0f32).unwrap())),
        );
        let dimension = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(2).unwrap());
        assert_eq!(
            dimension.dot(&dimension, &DotDimensionNumbers::inner_product()),
            Err(ProgramError::Type(TypeError::invalid("expected array type but got dimension type"))),
        );

        // Batching and differentiation reach the homogeneous rule through the array IR member projection.
        assert_eq!(
            batch(
                |item: BatchingTracer<
                    EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
                    ArrayIrBatchingPolicy,
                >| { item.dot(&item, &DotDimensionNumbers::inner_product()) },
                ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0f32, 2.0, 3.0, 4.0]).unwrap()),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            ),
            Ok(ArrayIrValue::Array(Array::vector(vec![5.0f32, 25.0]).unwrap())),
        );
        assert_eq!(
            differentiate_at(left).jvp(right, |input| input.dot(&input, &DotDimensionNumbers::inner_product())),
            Ok((
                ArrayIrValue::Array(Array::scalar(5.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(22.0f32).unwrap()),
            )),
        );
    }

    #[test]
    fn test_ragged_dot_mode_display() {
        assert_eq!(RaggedDotMode::NonContracting.to_string(), "non-contracting");
        assert_eq!(RaggedDotMode::Contracting.to_string(), "contracting");
        assert_eq!(RaggedDotMode::Batch.to_string(), "batch");
        assert_eq!(
            format!("{:?}", [RaggedDotMode::NonContracting, RaggedDotMode::Contracting, RaggedDotMode::Batch]),
            "[NonContracting, Contracting, Batch]",
        );
    }

    #[test]
    fn test_ragged_dot_dimension_numbers_new() {
        let dimensions = RaggedDotDimensionNumbers::new(DotDimensionNumbers::inner_product(), 0, None);
        assert_eq!(dimensions.dot_dimensions(), &DotDimensionNumbers::inner_product());
        assert_eq!(dimensions.lhs_ragged_dimension(), 0);
        assert_eq!(dimensions.rhs_group_dimension(), None);
        assert_eq!(
            dimensions.to_string(),
            "(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[], lhs_ragged=0)",
        );
        let dimensions = RaggedDotDimensionNumbers::matmul();
        assert_eq!(
            dimensions.to_string(),
            "(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[], rhs_batching=[], lhs_ragged=0, \
             rhs_group=0)",
        );
        assert_eq!(
            format!("{dimensions:?}"),
            "RaggedDotDimensionNumbers { dot_dimensions: DotDimensionNumbers { lhs_contracting_dimensions: [1], \
             rhs_contracting_dimensions: [1], lhs_batching_dimensions: [], rhs_batching_dimensions: [] }, \
             lhs_ragged_dimension: 0, rhs_group_dimension: Some(0) }",
        );
    }

    #[test]
    fn test_ragged_dot_dimension_numbers_matmul() {
        assert_eq!(
            RaggedDotDimensionNumbers::matmul(),
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![1], vec![1], vec![], vec![]), 0, Some(0)),
        );
    }

    #[test]
    fn test_ragged_dot_dimension_numbers_mode() {
        assert_eq!(RaggedDotDimensionNumbers::matmul().mode(2), Ok(RaggedDotMode::NonContracting));
        assert_eq!(
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::inner_product(), 0, None).mode(1),
            Ok(RaggedDotMode::Contracting),
        );
        assert_eq!(
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![], vec![], vec![0], vec![0]), 0, None).mode(1),
            Ok(RaggedDotMode::Batch),
        );
        assert_eq!(
            RaggedDotDimensionNumbers::matmul().mode(0),
            Err(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` left-hand side ragged dimension 0 is out of bounds for rank 0",
            ))),
        );
    }

    #[test]
    fn test_ragged_dot_dimension_numbers_group_sizes_prefix_dimensions() {
        let non_contracting =
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![3], vec![0], vec![1], vec![1]), 2, Some(2));
        assert_eq!(non_contracting.group_sizes_prefix_dimensions(4), Ok(vec![1, 0]));
        let contracting =
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![3, 0], vec![0, 2], vec![1], vec![1]), 0, None);
        assert_eq!(contracting.group_sizes_prefix_dimensions(4), Ok(vec![1, 3]));
        let batch =
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![], vec![], vec![2, 0], vec![1, 0]), 0, None);
        assert_eq!(batch.group_sizes_prefix_dimensions(3), Ok(vec![2]));

        // Invalid left-hand side batching or contracting indices are rejected before they index the prefix.
        let invalid =
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![1], vec![1], vec![3], vec![0]), 0, Some(0));
        assert_eq!(
            invalid.group_sizes_prefix_dimensions(2),
            Err(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` left-hand side dimension index out of bounds",
            ))),
        );
    }

    #[test]
    fn test_ragged_dot_dimension_numbers_validate() {
        let dimensions = DotDimensionNumbers::new(vec![1], vec![1], vec![], vec![]);
        assert_eq!(RaggedDotDimensionNumbers::matmul().validate(2, 3), Ok(RaggedDotMode::NonContracting));
        assert_eq!(
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::matmul(), 1, None).validate(2, 2),
            Ok(RaggedDotMode::Contracting),
        );
        assert_eq!(
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]), 0, None)
                .validate(2, 3),
            Ok(RaggedDotMode::Batch),
        );

        // Invalid underlying dot dimension numbers are reported with the ragged dot operation name.
        assert_eq!(
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![1], vec![1], vec![], vec![0]), 0, Some(0))
                .validate(2, 3),
            Err(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` batching dimensions have different lengths on the two inputs",
            ))),
        );
        assert_eq!(
            RaggedDotDimensionNumbers::new(dimensions.clone(), 2, Some(0)).validate(2, 3),
            Err(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` left-hand side ragged dimension 2 is out of bounds for rank 2",
            ))),
        );

        // The group dimension must be present exactly in non-contracting mode and must index a result axis.
        assert_eq!(
            RaggedDotDimensionNumbers::new(dimensions.clone(), 0, None).validate(2, 3),
            Err(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` requires a right-hand side group dimension when the left-hand side \
                 ragged dimension is non-contracting",
            ))),
        );
        assert_eq!(
            RaggedDotDimensionNumbers::new(dimensions.clone(), 0, Some(3)).validate(2, 3),
            Err(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` right-hand side group dimension 3 is out of bounds for rank 3",
            ))),
        );
        assert_eq!(
            RaggedDotDimensionNumbers::new(dimensions, 0, Some(1)).validate(2, 3),
            Err(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` right-hand side group dimension must be distinct from batching and \
                 contracting dimensions",
            ))),
        );
        assert_eq!(
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]), 2, Some(0))
                .validate(3, 3),
            Err(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` right-hand side group dimension must be distinct from batching and \
                 contracting dimensions",
            ))),
        );
        assert_eq!(
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::matmul(), 1, Some(1)).validate(2, 2),
            Err(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` does not accept a right-hand side group dimension when the \
                 left-hand side ragged dimension is contracting or batching",
            ))),
        );
        assert_eq!(
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]), 0, Some(2))
                .validate(2, 3),
            Err(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` does not accept a right-hand side group dimension when the \
                 left-hand side ragged dimension is contracting or batching",
            ))),
        );
    }

    #[test]
    fn test_ragged_dot_dimension_numbers_lhs_cotangent_dimensions() {
        // The adjoint is another non-contracting ragged dot whose ragged dimension is the ragged output dimension.
        assert_eq!(
            RaggedDotDimensionNumbers::matmul().lhs_cotangent_dimensions(2, 3),
            (
                RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![1], vec![2], vec![], vec![]), 0, Some(0)),
                vec![0, 1],
            ),
        );

        // The adjoint emits the batching axes first, so the left-hand side axes must be restored with a permutation.
        assert_eq!(
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![1], vec![1], vec![2], vec![2]), 0, Some(0))
                .lhs_cotangent_dimensions(3, 4),
            (
                RaggedDotDimensionNumbers::new(
                    DotDimensionNumbers::new(vec![2], vec![3], vec![0], vec![2]),
                    1,
                    Some(0)
                ),
                vec![2, 0, 1],
            ),
        );
    }

    #[test]
    fn test_ragged_dot_dimension_numbers_rhs_cotangent_dimensions() {
        // The adjoint is a contracting ragged dot, so it emits the group axis first.
        assert_eq!(
            RaggedDotDimensionNumbers::matmul().rhs_cotangent_dimensions(2, 3),
            (
                RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![]), 0, None),
                vec![0, 1, 2],
            ),
        );

        // The batching axes follow the group axis, ahead of the right-hand side contracting axes.
        assert_eq!(
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![1], vec![1], vec![2], vec![2]), 0, Some(0))
                .rhs_cotangent_dimensions(3, 4),
            (
                RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![0], vec![1], vec![2], vec![0]), 0, None),
                vec![0, 2, 1, 3],
            ),
        );
    }

    #[test]
    fn test_ragged_dot() {
        let operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::matmul());

        // Operation identity, accessors, and formatting.
        assert_eq!(operation.name(), RAGGED_DOT_OPERATION_NAME);
        assert_eq!(operation.dimensions(), &RaggedDotDimensionNumbers::matmul());
        assert_eq!(
            operation.to_string(),
            indoc! {"
                ragged_dot [
                    dimensions=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[], rhs_batching=[], \
                        lhs_ragged=0, rhs_group=0),
                ]
            "}
            .trim_end(),
        );
        assert_eq!(
            format!("{operation:?}"),
            "RaggedDotOperation { dimensions: RaggedDotDimensionNumbers { dot_dimensions: DotDimensionNumbers { \
             lhs_contracting_dimensions: [1], rhs_contracting_dimensions: [1], lhs_batching_dimensions: [], \
             rhs_batching_dimensions: [] }, lhs_ragged_dimension: 0, rhs_group_dimension: Some(0) } }",
        );

        // Program rendering uses the canonical operation name and includes the dimension numbers.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(left, right, group_sizes): (Tracer<_>, Tracer<_>, Tracer<_>)| {
                left.ragged_dot(&right, &group_sizes, &RaggedDotDimensionNumbers::matmul())
            },
            (plain_array(&[3, 2]), plain_array(&[2, 2, 1]), ArrayType::new_static(DataType::I32, [2])),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3, 2], %1:f32[2, 2, 1], %2:i32[2] .
                let %3:f32[3, 1] = ragged_dot [
                    dimensions=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[], rhs_batching=[], \
                        lhs_ragged=0, rhs_group=0),
                ] %0 %1 %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_ragged_dot_type_inference() {
        let group_sizes = ArrayType::new_static(DataType::I32, [3]);

        // Non-contracting mode drops the right-hand side group dimension from the dot output.
        check_operation_type_inference!(
            operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::matmul()),
            cases = [
                {
                    input_types = [plain_array(&[5, 2]), plain_array(&[3, 2, 4]), group_sizes.clone()],
                    output_types = [plain_array(&[5, 4])],
                },
                {
                    input_types = [
                        plain_array(&[5, 2]),
                        plain_array(&[3, 2, 4]),
                        ArrayType::new_static(DataType::F32, [3]),
                    ],
                    error = format!("`{RAGGED_DOT_OPERATION_NAME}` group sizes must have an integer data type"),
                },
                {
                    input_types = [
                        plain_array(&[5, 2]),
                        plain_array(&[3, 2, 4]),
                        ArrayType::scalar(DataType::I32),
                    ],
                    error = format!("`{RAGGED_DOT_OPERATION_NAME}` group sizes must have rank at least one"),
                },
                {
                    input_types = [plain_array(&[5, 2]), plain_array(&[2, 2, 4]), group_sizes.clone()],
                    error = format!(
                        "`{RAGGED_DOT_OPERATION_NAME}` right-hand side group dimension has extent 2 but group sizes \
                         describe 3",
                    ),
                },
                {
                    input_types = [
                        ArrayType::new_static(DataType::F8E8M0FNU, [5, 2]),
                        ArrayType::new_static(DataType::F8E8M0FNU, [3, 2, 4]),
                        group_sizes.clone(),
                    ],
                    error = format!(
                        "`{RAGGED_DOT_OPERATION_NAME}` does not support element data type `f8e8m0fnu` in \
                         grouped expansion modes because it cannot represent zero",
                    ),
                },
            ],
        );

        // The right-hand side group dimension may follow the right-hand side result dimensions.
        check_operation_type_inference!(
            operation =
                RaggedDotOperation::new(RaggedDotDimensionNumbers::new(DotDimensionNumbers::matmul(), 0, Some(2))),
            cases = [{
                input_types = [plain_array(&[5, 2]), plain_array(&[2, 4, 3]), group_sizes.clone()],
                output_types = [plain_array(&[5, 4])],
            }],
        );

        // Type inference validates the dimension numbers, reporting errors with the ragged dot operation name.
        check_operation_type_inference!(
            operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
                DotDimensionNumbers::new(vec![1], vec![1], vec![], vec![]),
                0,
                None,
            )),
            cases = [{
                input_types = [plain_array(&[5, 2]), plain_array(&[3, 2, 4]), group_sizes.clone()],
                error = format!(
                    "`{RAGGED_DOT_OPERATION_NAME}` requires a right-hand side group dimension when the left-hand side \
                     ragged dimension is non-contracting",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
                DotDimensionNumbers::new(vec![1], vec![1], vec![3], vec![0]),
                0,
                Some(0),
            )),
            cases = [{
                input_types = [
                    plain_array(&[3, 2]),
                    plain_array(&[2, 2, 1]),
                    ArrayType::new_static(DataType::I32, [2, 2]),
                ],
                error = format!("`{RAGGED_DOT_OPERATION_NAME}` left-hand side dimension index out of bounds"),
            }],
        );

        // Contracting mode inserts a leading group dimension and accepts any integer group sizes.
        check_operation_type_inference!(
            operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(DotDimensionNumbers::matmul(), 1, None)),
            cases = [
                {
                    input_types = [plain_array(&[2, 5]), plain_array(&[5, 4]), group_sizes.clone()],
                    output_types = [plain_array(&[3, 2, 4])],
                },
                {
                    input_types = [
                        plain_array(&[2, 5]),
                        plain_array(&[5, 4]),
                        ArrayType::new_static(DataType::I64, [3]),
                    ],
                    output_types = [plain_array(&[3, 2, 4])],
                },
                {
                    input_types = [
                        ArrayType::new_static(DataType::F8E8M0FNU, [2, 5]),
                        ArrayType::new_static(DataType::F8E8M0FNU, [5, 4]),
                        group_sizes.clone(),
                    ],
                    error = format!(
                        "`{RAGGED_DOT_OPERATION_NAME}` does not support element data type `f8e8m0fnu` in \
                         grouped expansion modes because it cannot represent zero",
                    ),
                },
            ],
        );
        check_operation_type_inference!(
            operation =
                RaggedDotOperation::new(RaggedDotDimensionNumbers::new(DotDimensionNumbers::matmul(), 1, Some(1))),
            cases = [{
                input_types = [plain_array(&[2, 5]), plain_array(&[5, 4]), group_sizes.clone()],
                error = format!(
                    "`{RAGGED_DOT_OPERATION_NAME}` does not accept a right-hand side group dimension when the \
                     left-hand side ragged dimension is contracting or batching",
                ),
            }],
        );

        // Batch mode keeps the ordinary batched dot output and accepts `f8e8m0fnu`, because it expands no groups.
        check_operation_type_inference!(
            operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
                DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]),
                0,
                None,
            )),
            cases = [
                {
                    input_types = [plain_array(&[5, 2]), plain_array(&[5, 2, 4]), group_sizes.clone()],
                    output_types = [plain_array(&[5, 4])],
                },
                {
                    input_types = [
                        ArrayType::new_static(DataType::F8E8M0FNU, [5, 2]),
                        ArrayType::new_static(DataType::F8E8M0FNU, [5, 2, 4]),
                        group_sizes,
                    ],
                    output_types = [ArrayType::new_static(DataType::F8E8M0FNU, [5, 4])],
                },
            ],
        );

        // Group sizes of rank greater than one must have a prefix that matches the left-hand side dimensions preceding
        // the ragged position in each mode.
        check_operation_type_inference!(
            operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
                DotDimensionNumbers::new(vec![2], vec![2], vec![0], vec![1]),
                1,
                Some(0),
            )),
            cases = [
                {
                    input_types = [
                        plain_array(&[2, 5, 3]),
                        plain_array(&[4, 2, 3, 6]),
                        ArrayType::new_static(DataType::I32, [2, 4]),
                    ],
                    output_types = [plain_array(&[2, 5, 6])],
                },
                {
                    input_types = [
                        plain_array(&[2, 5, 3]),
                        plain_array(&[4, 2, 3, 6]),
                        ArrayType::new_static(DataType::I32, [3, 4]),
                    ],
                    error = format!("`{RAGGED_DOT_OPERATION_NAME}` group sizes prefix must be `[2]`, but got `[3]`"),
                },
            ],
        );
        check_operation_type_inference!(
            operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
                DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]),
                2,
                None,
            )),
            cases = [{
                input_types = [
                    plain_array(&[2, 3, 4]),
                    plain_array(&[2, 4, 5]),
                    ArrayType::new_static(DataType::I32, [2, 3]),
                ],
                output_types = [plain_array(&[3, 2, 3, 5])],
            }],
        );
        check_operation_type_inference!(
            operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
                DotDimensionNumbers::new(vec![2], vec![2], vec![0, 1], vec![0, 1]),
                1,
                None,
            )),
            cases = [{
                input_types = [
                    plain_array(&[2, 4, 3]),
                    plain_array(&[2, 4, 3, 5]),
                    ArrayType::new_static(DataType::I32, [2, 3]),
                ],
                output_types = [plain_array(&[2, 4, 5])],
            }],
        );
    }

    #[test]
    fn test_ragged_dot_type_inference_sharding() {
        let mesh = test_mesh();
        let lhs = sharded_array(&mesh, &[4, 8], vec![ShardingDimension::sharded(["m"]), ShardingDimension::Replicated]);

        // Non-contracting mode drops the sharding of the right-hand side group dimension with the dimension itself.
        check_operation_type_inference!(
            operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::matmul()),
            cases = [{
                input_types = [
                    lhs.clone(),
                    sharded_array(
                        &mesh,
                        &[2, 8, 16],
                        vec![
                            ShardingDimension::Replicated,
                            ShardingDimension::Replicated,
                            ShardingDimension::sharded(["n"]),
                        ],
                    ),
                    ArrayType::new_static(DataType::I32, [2]),
                ],
                output_types = [sharded_array(
                    &mesh,
                    &[4, 16],
                    vec![ShardingDimension::sharded(["m"]), ShardingDimension::sharded(["n"])],
                )],
            }],
        );

        // Contracting mode inserts a replicated leading group dimension.
        check_operation_type_inference!(
            operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(DotDimensionNumbers::matmul(), 1, None)),
            cases = [{
                input_types = [
                    lhs,
                    sharded_array(
                        &mesh,
                        &[8, 16],
                        vec![ShardingDimension::Replicated, ShardingDimension::sharded(["n"])],
                    ),
                    ArrayType::new_static(DataType::I32, [3]),
                ],
                output_types = [sharded_array(
                    &mesh,
                    &[3, 4, 16],
                    vec![
                        ShardingDimension::Replicated,
                        ShardingDimension::sharded(["m"]),
                        ShardingDimension::sharded(["n"]),
                    ],
                )],
            }],
        );

        // The group sizes take part in the manual variation check of the data inputs.
        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        check_operation_type_inference!(
            operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::matmul()),
            cases = [{
                input_types = [
                    plain_array(&[5, 2])
                        .with_sharding(
                            Sharding::replicated(manual_mesh.clone(), 2).with_varying_manual_axes(["devices"]).unwrap(),
                        )
                        .unwrap(),
                    plain_array(&[3, 2, 4])
                        .with_sharding(
                            Sharding::replicated(manual_mesh, 3).with_varying_manual_axes(["devices"]).unwrap(),
                        )
                        .unwrap(),
                    ArrayType::new_static(DataType::I32, [3]),
                ],
                error = format!(
                    "`{RAGGED_DOT_OPERATION_NAME}` inputs must have matching varying manual axes; insert \
                     `parallel_vary` on the inputs that lack an axis, as `align_manual_variation` does",
                ),
            }],
        );
    }

    #[test]
    fn test_ragged_dot_interpretation() {
        // Consecutive groups of left-hand side rows use their right-hand side slices, and the uncovered suffix is zero.
        let lhs = Array::matrix(5, 2, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).unwrap();
        let rhs = Array::from_elements::<f32>(plain_array(&[3, 2, 1]), &[10.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(
            lhs.ragged_dot(&rhs, &Array::vector(vec![2i32, 0, 2]).unwrap(), &RaggedDotDimensionNumbers::matmul()),
            Ok(Array::matrix(5, 1, vec![12.0f32, 34.0, 50.0, 68.0, 0.0]).unwrap()),
        );

        // Sizes that cover more than the extent clip the intersecting group and leave every later raw cumulative
        // interval empty.
        let lhs = Array::matrix(4, 2, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        assert_eq!(
            lhs.ragged_dot(&rhs, &Array::vector(vec![3i32, 3, 2]).unwrap(), &RaggedDotDimensionNumbers::matmul()),
            Ok(Array::matrix(4, 1, vec![12.0f32, 34.0, 56.0, 38.0]).unwrap()),
        );

        // Sizes in intervals that begin after the physical extent are unobservable, even when their cumulative sum
        // would overflow the host index type.
        let lhs = Array::matrix(1, 1, vec![2.0f32]).unwrap();
        let rhs = Array::from_elements::<f32>(plain_array(&[2, 1, 1]), &[3.0, 100.0]).unwrap();
        assert_eq!(
            lhs.ragged_dot(&rhs, &Array::vector(vec![1u64, u64::MAX]).unwrap(), &RaggedDotDimensionNumbers::matmul()),
            Ok(Array::matrix(1, 1, vec![6.0f32]).unwrap()),
        );

        // The right-hand side group dimension may follow the right-hand side result dimensions.
        let lhs = Array::matrix(3, 2, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs = Array::from_elements::<f32>(plain_array(&[2, 1, 2]), &[10.0, 2.0, 1.0, 3.0]).unwrap();
        assert_eq!(
            lhs.ragged_dot(
                &rhs,
                &Array::vector(vec![1i32, 2]).unwrap(),
                &RaggedDotDimensionNumbers::new(DotDimensionNumbers::matmul(), 0, Some(2)),
            ),
            Ok(Array::matrix(3, 1, vec![12.0f32, 18.0, 28.0]).unwrap()),
        );
    }

    #[test]
    fn test_ragged_dot_interpretation_contracting_mode() {
        // Consecutive groups of contracting positions produce one leading output slice each, and an empty group
        // produces a zero slice.
        let dimensions = RaggedDotDimensionNumbers::new(DotDimensionNumbers::matmul(), 1, None);
        let lhs = Array::matrix(2, 4, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let rhs = Array::matrix(4, 1, vec![10.0f32, 20.0, 30.0, 40.0]).unwrap();
        assert_eq!(
            lhs.ragged_dot(&rhs, &Array::vector(vec![1i32, 0, 2]).unwrap(), &dimensions),
            Ok(Array::from_elements(plain_array(&[3, 2, 1]), &[10.0f32, 50.0, 0.0, 0.0, 130.0, 330.0]).unwrap()),
        );

        // Contracting groups use the same clipped raw intervals, with empty trailing groups retaining zero output
        // slices.
        assert_eq!(
            lhs.ragged_dot(&rhs, &Array::vector(vec![3i32, 3, 2]).unwrap(), &dimensions),
            Ok(Array::from_elements(plain_array(&[3, 2, 1]), &[140.0f32, 380.0, 160.0, 320.0, 0.0, 0.0]).unwrap()),
        );
    }

    #[test]
    fn test_ragged_dot_interpretation_batch_mode() {
        // Batch mode is the ordinary batched dot, so group size values do not participate in its runtime semantics.
        let dimensions =
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]), 0, None);
        let lhs = Array::matrix(4, 1, vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let rhs = Array::from_elements::<f32>(plain_array(&[4, 1, 1]), &[10.0, 20.0, 30.0, 40.0]).unwrap();
        assert_eq!(
            lhs.ragged_dot(&rhs, &Array::vector(vec![5i32, -1, 99]).unwrap(), &dimensions),
            Ok(Array::matrix(4, 1, vec![10.0f32, 40.0, 90.0, 160.0]).unwrap()),
        );

        // Batch mode expands no groups, so it also accepts `f8e8m0fnu`, which cannot represent zero.
        let lhs = Array::from_elements::<f8e8m0fnu>(
            ArrayType::new_static(DataType::F8E8M0FNU, [2, 1]),
            &[1.0, 2.0].map(|value| f8e8m0fnu::from_f64(value).unwrap()),
        )
        .unwrap();
        let rhs = Array::from_elements::<f8e8m0fnu>(
            ArrayType::new_static(DataType::F8E8M0FNU, [2, 1, 1]),
            &[2.0, 4.0].map(|value| f8e8m0fnu::from_f64(value).unwrap()),
        )
        .unwrap();
        assert_eq!(
            lhs.ragged_dot(&rhs, &Array::vector(vec![-1i32]).unwrap(), &dimensions),
            Ok(Array::from_elements::<f8e8m0fnu>(
                ArrayType::new_static(DataType::F8E8M0FNU, [2, 1]),
                &[2.0, 8.0].map(|value| f8e8m0fnu::from_f64(value).unwrap()),
            )
            .unwrap()),
        );
    }

    #[test]
    fn test_ragged_dot_interpretation_group_sizes_prefix() {
        // Group sizes with a prefix select separate group sizes for every prefix coordinate. In contracting mode, the
        // prefix is the contracting dimension preceding the ragged one.
        let dimensions =
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![0, 1], vec![0, 1], vec![], vec![]), 1, None);
        let lhs = Array::from_elements::<f32>(plain_array(&[2, 3, 1]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs = Array::from_elements::<f32>(plain_array(&[2, 3, 1]), &[10.0, 20.0, 30.0, 1.0, 2.0, 3.0]).unwrap();
        assert_eq!(
            lhs.ragged_dot(&rhs, &Array::matrix(2, 2, vec![1i32, 2, 2, 1]).unwrap(), &dimensions),
            Ok(Array::from_elements(plain_array(&[2, 1, 1]), &[24.0f32, 148.0]).unwrap()),
        );
        assert_eq!(
            lhs.ragged_dot(&rhs, &Array::matrix(2, 2, vec![2i32, 3, 4, 1]).unwrap(), &dimensions),
            Ok(Array::from_elements(plain_array(&[2, 1, 1]), &[82.0f32, 90.0]).unwrap()),
        );

        // In non-contracting mode, the prefix is the batching dimension.
        let dimensions =
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![2], vec![2], vec![0], vec![1]), 1, Some(0));
        let lhs = Array::from_elements::<f32>(plain_array(&[2, 3, 1]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs = Array::from_elements::<f32>(plain_array(&[2, 2, 1, 1]), &[10.0, 100.0, 20.0, 200.0]).unwrap();
        assert_eq!(
            lhs.ragged_dot(&rhs, &Array::matrix(2, 2, vec![1i32, 1, 2, 0]).unwrap(), &dimensions),
            Ok(Array::from_elements(plain_array(&[2, 3, 1]), &[10.0f32, 40.0, 0.0, 400.0, 500.0, 0.0]).unwrap()),
        );

        // In batch mode, the prefix is the batching dimension preceding the ragged one.
        let dimensions =
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![2], vec![2], vec![0, 1], vec![0, 1]), 1, None);
        let lhs =
            Array::from_elements::<f32>(plain_array(&[2, 4, 1]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let rhs =
            Array::from_elements::<f32>(plain_array(&[2, 4, 1, 1]), &[10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0])
                .unwrap();
        assert_eq!(
            lhs.ragged_dot(&rhs, &Array::matrix(2, 3, vec![1i32, 0, 2, 0, 2, 2]).unwrap(), &dimensions),
            Ok(Array::from_elements(plain_array(&[2, 4, 1]), &[10.0f32, 40.0, 90.0, 160.0, 5.0, 12.0, 21.0, 32.0])
                .unwrap()),
        );
    }

    #[test]
    fn test_ragged_dot_interpretation_rejects_negative_group_sizes() {
        let lhs = Array::matrix(2, 1, vec![1.0f32, 2.0]).unwrap();
        let rhs = Array::from_elements::<f32>(plain_array(&[2, 1, 1]), &[3.0, 4.0]).unwrap();
        assert!(matches!(
            lhs.ragged_dot(&rhs, &Array::vector(vec![1i32, -1]).unwrap(), &RaggedDotDimensionNumbers::matmul()),
            Err(ProgramError::InvalidArgument { message })
                if message == "`group_sizes[1]` must be non-negative but got -1",
        ));
    }

    #[test]
    fn test_ragged_dot_interpretation_rejects_invalid_input_types() {
        let lhs = Array::matrix(2, 1, vec![true, false]).unwrap();
        let rhs = Array::from_elements(ArrayType::new_static(DataType::Boolean, [2, 1, 1]), &[true, true]).unwrap();
        assert_eq!(
            lhs.ragged_dot(&rhs, &Array::vector(vec![1i32, 1]).unwrap(), &RaggedDotDimensionNumbers::matmul()),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` does not support input data type `bool`",
            )))),
        );
    }

    #[test]
    fn test_ragged_dot_interpretation_empty_output_skips_large_prefix_products() {
        // These enormous extents require no storage. Their products must never be evaluated for an empty result.
        let extent = 1usize << (usize::BITS / 2);
        let left =
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0, extent, extent, 1]), &[]).unwrap();
        let right = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 0, 0]), &[]).unwrap();
        let dimensions =
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![0], vec![1], vec![], vec![]), 3, Some(0));
        assert_eq!(
            left.ragged_dot(&right, &Array::vector(vec![1i32]).unwrap(), &dimensions),
            Ok(Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [extent, extent, 1, 0]), &[]).unwrap()),
        );

        // Negative sizes are rejected before the empty output shortcut.
        assert_eq!(
            left.ragged_dot(&right, &Array::vector(vec![-1i32]).unwrap(), &dimensions),
            Err(ProgramError::InvalidArgument {
                message: "`group_sizes[0]` must be non-negative but got -1".to_string(),
            }),
        );
    }

    #[test]
    fn test_ragged_dot_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::matmul()),
            inputs = [
                Array::matrix(2, 1, vec![2.0, 3.0]).unwrap(),
                Array::from_elements(ArrayType::new_static(DataType::F64, [2, 1, 1]), &[5.0, 7.0]).unwrap(),
                Array::vector(vec![1i32, 1]).unwrap(),
            ],
            expected = Array::matrix(2, 1, vec![10.0, 21.0]).unwrap(),
        );
    }

    #[test]
    fn test_ragged_dot_batching() {
        // Non-contracting mode keeps the mapped axis first.
        check_operation_batching!(
            @exact,
            operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::matmul()),
            axis_size = 2,
            cases = [{
                inputs = [
                    (
                        @mapped(axis = 0),
                        Array::from_elements::<f32>(plain_array(&[2, 2, 2]), &[1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 4.0, 3.0])
                            .unwrap(),
                    ),
                    (
                        @mapped(axis = 0),
                        Array::from_elements::<f32>(
                            plain_array(&[2, 2, 2, 1]),
                            &[10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                        )
                        .unwrap(),
                    ),
                    (@mapped(axis = 0), Array::matrix(2, 2, vec![1i32, 1, 1, 1]).unwrap()),
                ],
                outputs = [(
                    @mapped(axis = 0),
                    Array::from_elements::<f32>(plain_array(&[2, 2, 1]), &[12.0, 18.0, 13.0, 45.0]).unwrap(),
                )],
            }],
        );

        // Contracting mode keeps its leading group output axis ahead of the mapped axis.
        check_operation_batching!(
            @exact,
            operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(DotDimensionNumbers::matmul(), 1, None)),
            axis_size = 2,
            cases = [{
                inputs = [
                    (
                        @mapped(axis = 0),
                        Array::from_elements::<f32>(
                            plain_array(&[2, 2, 4]),
                            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 0.0],
                        )
                        .unwrap(),
                    ),
                    (
                        @mapped(axis = 0),
                        Array::from_elements::<f32>(
                            plain_array(&[2, 4, 1]),
                            &[10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0],
                        )
                        .unwrap(),
                    ),
                    (@mapped(axis = 0), Array::matrix(2, 2, vec![1i32, 2, 2, 2]).unwrap()),
                ],
                outputs = [(
                    @mapped(axis = 1),
                    Array::from_elements::<f32>(
                        plain_array(&[2, 2, 2, 1]),
                        &[10.0, 50.0, 4.0, 5.0, 130.0, 330.0, 4.0, 3.0],
                    )
                    .unwrap(),
                )],
            }],
        );

        // Batch mode keeps the mapped axis first, and its replicated group sizes stay shared by every item.
        check_operation_batching!(
            @exact,
            operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
                DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]),
                0,
                None,
            )),
            axis_size = 2,
            cases = [{
                inputs = [
                    (
                        @mapped(axis = 0),
                        Array::from_elements::<f32>(plain_array(&[2, 2, 1]), &[1.0, 2.0, 3.0, 4.0]).unwrap(),
                    ),
                    (
                        @mapped(axis = 0),
                        Array::from_elements::<f32>(plain_array(&[2, 2, 1, 1]), &[10.0, 20.0, 30.0, 40.0]).unwrap(),
                    ),
                    (@replicated, Array::vector(vec![1i32]).unwrap()),
                ],
                outputs = [(
                    @mapped(axis = 0),
                    Array::from_elements::<f32>(plain_array(&[2, 2, 1]), &[10.0, 40.0, 90.0, 160.0]).unwrap(),
                )],
            }],
        );
    }

    #[test]
    fn test_ragged_dot_batching_validates_original_dimensions() {
        let left = ArrayBatch::new(
            Array::from_elements::<f32>(plain_array(&[2, 1, 1]), &[1.0, 2.0]).unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let right = ArrayBatch::new(
            Array::from_elements::<f32>(plain_array(&[2, 1, 1, 1]), &[3.0, 4.0]).unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let group_sizes = ArrayBatch::new(Array::matrix(2, 1, vec![1i32, 1]).unwrap(), BatchAxis::new(0)).unwrap();
        let operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![usize::MAX], vec![1], vec![], vec![]),
            0,
            Some(0),
        ));
        let inputs = [left, right, group_sizes];
        let context = BatchingContext::new(EagerContext::<Array>::new(), 2);
        assert_eq!(
            operation.batch(&context, &EmptyRegionDriver, &inputs),
            Err(BatchingError::Type(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` left-hand side dimension index out of bounds",
            )))),
        );
        let invalid_group = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![1], vec![1], vec![], vec![]),
            0,
            Some(usize::MAX),
        ));
        assert_eq!(
            invalid_group.batch(&context, &EmptyRegionDriver, &inputs),
            Err(BatchingError::Type(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` right-hand side group dimension {} is out of bounds for rank 3",
                usize::MAX,
            )))),
        );
    }

    #[test]
    fn test_ragged_dot_batching_validates_mapped_extents() {
        let lhs = ArrayBatch::new(
            Array::from_elements::<f32>(plain_array(&[2, 3, 2]), &[1.0; 12]).unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let rhs = ArrayBatch::replicated(Array::from_elements::<f32>(plain_array(&[2, 2, 1]), &[1.0; 4]).unwrap());
        let group_sizes =
            ArrayBatch::new(Array::matrix(3, 2, vec![1i32, 2, 2, 1, 3, 0]).unwrap(), BatchAxis::new(0)).unwrap();
        assert_eq!(
            RaggedDotOperation::new(RaggedDotDimensionNumbers::matmul()).batch(
                &BatchingContext::new(EagerContext::<Array>::new(), 2),
                &EmptyRegionDriver,
                &[lhs, rhs, group_sizes],
            ),
            Err(BatchingError::MismatchedBatchSizes { expected: 2, actual: 3 }),
        );
    }

    #[test]
    fn test_ragged_dot_batching_aligns_mixed_batch_axes() {
        // Replicated data inputs are materialized across the mapped extent and mapped inputs at non-leading axes move
        // to the front. Replicated rank-one group sizes stay shared by every item.
        check_operation_batching!(
            @exact,
            operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::matmul()),
            axis_size = 2,
            cases = [
                {
                    inputs = [
                        (
                            @mapped(axis = 0),
                            Array::from_elements::<f32>(
                                plain_array(&[2, 3, 2]),
                                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
                            )
                            .unwrap(),
                        ),
                        (
                            @replicated,
                            Array::from_elements::<f32>(plain_array(&[2, 2, 1]), &[1.0, 10.0, 100.0, 1000.0]).unwrap(),
                        ),
                        (@mapped(axis = 0), Array::matrix(2, 2, vec![1i32, 2, 2, 1]).unwrap()),
                    ],
                    outputs = [(
                        @mapped(axis = 0),
                        Array::from_elements::<f32>(
                            plain_array(&[2, 3, 1]),
                            &[21.0, 4300.0, 6500.0, 12.0, 34.0, 5600.0],
                        )
                        .unwrap(),
                    )],
                },
                {
                    inputs = [
                        (@replicated, Array::matrix(3, 2, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
                        (
                            @mapped(axis = 1),
                            Array::from_elements::<f32>(
                                plain_array(&[2, 2, 2, 1]),
                                &[1.0, 10.0, 2.0, 20.0, 100.0, 1000.0, 200.0, 2000.0],
                            )
                            .unwrap(),
                        ),
                        (@replicated, Array::vector(vec![1i32, 2]).unwrap()),
                    ],
                    outputs = [(
                        @mapped(axis = 0),
                        Array::from_elements::<f32>(
                            plain_array(&[2, 3, 1]),
                            &[21.0, 4300.0, 6500.0, 42.0, 8600.0, 13000.0],
                        )
                        .unwrap(),
                    )],
                },
                {
                    inputs = [
                        (@replicated, Array::matrix(3, 2, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
                        (
                            @replicated,
                            Array::from_elements::<f32>(plain_array(&[2, 2, 1]), &[1.0, 10.0, 100.0, 1000.0]).unwrap(),
                        ),
                        (@mapped(axis = 1), Array::matrix(2, 2, vec![1i32, 3, 2, 0]).unwrap()),
                    ],
                    outputs = [(
                        @mapped(axis = 0),
                        Array::from_elements::<f32>(plain_array(&[2, 3, 1]), &[21.0, 4300.0, 6500.0, 21.0, 43.0, 65.0])
                            .unwrap(),
                    )],
                },
            ],
        );
    }

    #[test]
    fn test_ragged_dot_batching_broadcasts_shared_group_sizes_across_prefix() {
        // Each item shares rank-one group sizes across its batching prefix, so mapped rank-one group sizes are
        // broadcast across that prefix of the lifted operation, while replicated ones stay rank-one.
        check_operation_batching!(
            @exact,
            operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
                DotDimensionNumbers::new(vec![2], vec![2], vec![0], vec![1]),
                1,
                Some(0),
            )),
            axis_size = 2,
            cases = [
                {
                    inputs = [
                        (
                            @mapped(axis = 0),
                            Array::from_elements::<f32>(
                                plain_array(&[2, 2, 3, 1]),
                                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                            )
                            .unwrap(),
                        ),
                        (
                            @mapped(axis = 0),
                            Array::from_elements::<f32>(
                                plain_array(&[2, 2, 2, 1, 1]),
                                &[10.0, 20.0, 100.0, 200.0, 10.0, 20.0, 100.0, 200.0],
                            )
                            .unwrap(),
                        ),
                        (@mapped(axis = 0), Array::matrix(2, 2, vec![1i32, 2, 3, 0]).unwrap()),
                    ],
                    outputs = [(
                        @mapped(axis = 0),
                        Array::from_elements::<f32>(
                            plain_array(&[2, 2, 3, 1]),
                            &[10.0, 200.0, 300.0, 80.0, 1000.0, 1200.0, 10.0, 20.0, 30.0, 80.0, 100.0, 120.0],
                        )
                        .unwrap(),
                    )],
                },
                {
                    inputs = [
                        (
                            @replicated,
                            Array::from_elements::<f32>(plain_array(&[2, 3, 1]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                                .unwrap(),
                        ),
                        (
                            @replicated,
                            Array::from_elements::<f32>(plain_array(&[2, 2, 1, 1]), &[10.0, 20.0, 100.0, 200.0])
                                .unwrap(),
                        ),
                        (@mapped(axis = 1), Array::matrix(2, 2, vec![1i32, 3, 2, 0]).unwrap()),
                    ],
                    outputs = [(
                        @mapped(axis = 0),
                        Array::from_elements::<f32>(
                            plain_array(&[2, 2, 3, 1]),
                            &[10.0, 200.0, 300.0, 80.0, 1000.0, 1200.0, 10.0, 20.0, 30.0, 80.0, 100.0, 120.0],
                        )
                        .unwrap(),
                    )],
                },
                {
                    inputs = [
                        (
                            @mapped(axis = 0),
                            Array::from_elements::<f32>(
                                plain_array(&[2, 2, 3, 1]),
                                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
                            )
                            .unwrap(),
                        ),
                        (
                            @replicated,
                            Array::from_elements::<f32>(plain_array(&[2, 2, 1, 1]), &[10.0, 20.0, 100.0, 200.0])
                                .unwrap(),
                        ),
                        (@replicated, Array::vector(vec![1i32, 2]).unwrap()),
                    ],
                    outputs = [(
                        @mapped(axis = 0),
                        Array::from_elements::<f32>(
                            plain_array(&[2, 2, 3, 1]),
                            &[10.0, 200.0, 300.0, 80.0, 1000.0, 1200.0, 20.0, 400.0, 600.0, 160.0, 2000.0, 2400.0],
                        )
                        .unwrap(),
                    )],
                },
            ],
        );
    }

    #[test]
    fn test_ragged_dot_batching_per_prefix_group_sizes() {
        // Group sizes that already have a per-prefix shape only need the mapped axis to lead their prefix, whether
        // they are mapped or replicated.
        check_operation_batching!(
            @exact,
            operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
                DotDimensionNumbers::new(vec![2], vec![2], vec![0], vec![1]),
                1,
                Some(0),
            )),
            axis_size = 2,
            cases = [
                {
                    inputs = [
                        (
                            @mapped(axis = 0),
                            Array::from_elements::<f32>(
                                plain_array(&[2, 2, 3, 1]),
                                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
                            )
                            .unwrap(),
                        ),
                        (
                            @mapped(axis = 0),
                            Array::from_elements::<f32>(
                                plain_array(&[2, 2, 2, 1, 1]),
                                &[10.0, 100.0, 20.0, 200.0, 10.0, 100.0, 20.0, 200.0],
                            )
                            .unwrap(),
                        ),
                        (
                            @mapped(axis = 0),
                            Array::from_elements::<i32>(
                                ArrayType::new_static(DataType::I32, [2, 2, 2]),
                                &[1, 1, 2, 0, 2, 1, 1, 1],
                            )
                            .unwrap(),
                        ),
                    ],
                    outputs = [(
                        @mapped(axis = 0),
                        Array::from_elements::<f32>(
                            plain_array(&[2, 2, 3, 1]),
                            &[10.0, 40.0, 0.0, 400.0, 500.0, 0.0, 20.0, 40.0, 120.0, 800.0, 2000.0, 0.0],
                        )
                        .unwrap(),
                    )],
                },
                {
                    inputs = [
                        (
                            @mapped(axis = 0),
                            Array::from_elements::<f32>(
                                plain_array(&[2, 2, 3, 1]),
                                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
                            )
                            .unwrap(),
                        ),
                        (
                            @replicated,
                            Array::from_elements::<f32>(plain_array(&[2, 2, 1, 1]), &[10.0, 100.0, 20.0, 200.0])
                                .unwrap(),
                        ),
                        (@replicated, Array::matrix(2, 2, vec![1i32, 1, 2, 0]).unwrap()),
                    ],
                    outputs = [(
                        @mapped(axis = 0),
                        Array::from_elements::<f32>(
                            plain_array(&[2, 2, 3, 1]),
                            &[10.0, 40.0, 0.0, 400.0, 500.0, 0.0, 20.0, 80.0, 0.0, 800.0, 1000.0, 0.0],
                        )
                        .unwrap(),
                    )],
                },
            ],
        );
    }

    #[test]
    fn test_ragged_dot_batching_dynamic_mapped_extent() -> Result<(), ProgramError> {
        // The replicated right-hand side gains the dynamic mapped axis through the policy's dynamic broadcast, and the
        // mapped rank-one group sizes are broadcast across the batching prefix of the lifted operation. Both
        // broadcasts are grounded by the transform's first-class extent value.
        let trace = ArrayIrContext::new();
        let batch_variable = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5))?);
        let batch_dimension = Dimension::Dynamic(batch_variable);
        let lhs = trace.input(
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![
                    batch_dimension.clone(),
                    Dimension::Static(2),
                    Dimension::Static(3),
                    Dimension::Static(1),
                ]),
            )
            .into(),
        );
        let rhs = trace.input(ArrayType::new_static(DataType::F32, [2, 2, 1, 1]).into());
        let group_sizes =
            trace.input(ArrayType::new(DataType::I32, Shape::new(vec![batch_dimension, Dimension::Static(2)])).into());
        let output = batch(
            |(lhs, rhs, group_sizes)| {
                let lhs = ValueProjection::<ArrayType>::into_projected(lhs)?;
                let rhs = ValueProjection::<ArrayType>::into_projected(rhs)?;
                let group_sizes = ValueProjection::<ArrayType>::into_projected(group_sizes)?;
                let mut outputs = lhs.domain().bind(
                    RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
                        DotDimensionNumbers::new(vec![2], vec![2], vec![0], vec![1]),
                        1,
                        Some(0),
                    )),
                    vec![],
                    &[lhs, rhs, group_sizes],
                )?;
                Ok(outputs.remove(0).into_value())
            },
            (lhs, rhs, group_sizes),
            (BatchAxis::new(0), BatchAxis::replicated(), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        )?;
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, ArrayIrValue<Array>>(
            vec![output.atom_id()?],
            vec![Placeholder, Placeholder, Placeholder],
            Placeholder,
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[batch, 2, 3, 1], %1:f32[2, 2, 1, 1], %2:i32[batch, 2] .
                let %3:dimension<batch ∈ [1, 5)> = dimension_size [axis=0] %0
                    %4:dimension<batch ∈ [1, 5)> = dimension_size [axis=0] %2
                    %5:bool[] = const true
                    %6:dimension<2> = constant [value=2]
                    %7:dimension<2> = constant [value=2]
                    %8:dimension<1> = constant [value=1]
                    %9:dimension<1> = constant [value=1]
                    %10:f32[batch, 2, 2, 1, 1] = broadcast [output_axes=[1, 2, 3, 4]] %1 %3 %6 %7 %8 %9
                    %11:dimension<2> = constant [value=2]
                    %12:dimension<2> = constant [value=2]
                    %13:i32[batch, 2, 2] = broadcast [output_axes=[0, 2]] %2 %3 %11 %12
                    %14:f32[batch, 2, 3, 1] = ragged_dot [
                        dimensions=(lhs_contracting=[3], rhs_contracting=[3], lhs_batching=[0, 1], \
                            rhs_batching=[0, 2], lhs_ragged=2, rhs_group=1),
                    ] %0 %10 %13
                in (%14)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Array(Array::from_elements::<f32>(
                    plain_array(&[2, 2, 3, 1]),
                    &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                )?),
                ArrayIrValue::Array(Array::from_elements::<f32>(
                    plain_array(&[2, 2, 1, 1]),
                    &[10.0, 20.0, 100.0, 200.0],
                )?),
                ArrayIrValue::Array(Array::matrix(2, 2, vec![1i32, 2, 3, 0])?),
            ])?,
            ArrayIrValue::Array(Array::from_elements::<f32>(
                plain_array(&[2, 2, 3, 1]),
                &[10.0, 200.0, 300.0, 80.0, 1000.0, 1200.0, 10.0, 20.0, 30.0, 80.0, 100.0, 120.0],
            )?),
        );
        Ok(())
    }

    #[test]
    fn test_ragged_dot_differentiation() {
        // These checks stay explicit because the finite difference oracle of `check_operation_differentiation!`
        // perturbs every operation input along a tangent, and the integer group sizes have none.
        let left = Array::matrix(2, 1, vec![2.0, 3.0]).unwrap();
        let right = Array::from_elements(ArrayType::new_static(DataType::F64, [2, 1, 1]), &[5.0, 7.0]).unwrap();
        let group_sizes = Array::vector(vec![1i32, 1]).unwrap();
        assert_eq!(
            differentiate_at((left, right)).jvp(
                (
                    Array::matrix(2, 1, vec![1.0, 2.0]).unwrap(),
                    Array::from_elements(ArrayType::new_static(DataType::F64, [2, 1, 1]), &[3.0, 4.0]).unwrap(),
                ),
                |(left, right)| {
                    let group_sizes = left.domain().lift(group_sizes.clone())?;
                    left.ragged_dot(&right, &group_sizes, &RaggedDotDimensionNumbers::matmul())
                },
            ),
            Ok((Array::matrix(2, 1, vec![10.0, 21.0]).unwrap(), Array::matrix(2, 1, vec![11.0, 26.0]).unwrap())),
        );
        check_gradient!(
            |left| {
                let right = left
                    .domain()
                    .lift(Array::from_elements(ArrayType::new_static(DataType::F64, [2, 1, 1]), &[5.0, 7.0])?)?;
                let group_sizes = left.domain().lift(Array::vector(vec![1i32, 1])?)?;
                left.ragged_dot(&right, &group_sizes, &RaggedDotDimensionNumbers::matmul())?
                    .reduce_sum(&[0, 1], None)
            },
            at = Array::matrix(2, 1, vec![2.0, 3.0]).unwrap(),
            step = 1e-5,
            tolerance = 1e-7,
        );
        check_gradient!(
            |right| {
                let left = right.domain().lift(Array::matrix(2, 1, vec![2.0, 3.0])?)?;
                let group_sizes = right.domain().lift(Array::vector(vec![1i32, 1])?)?;
                left.ragged_dot(&right, &group_sizes, &RaggedDotDimensionNumbers::matmul())?
                    .reduce_sum(&[0, 1], None)
            },
            at = Array::from_elements(ArrayType::new_static(DataType::F64, [2, 1, 1]), &[5.0, 7.0]).unwrap(),
            step = 1e-5,
            tolerance = 1e-7,
        );

        // Contracting mode has no transposition rule, so its product rule is checked in forward mode. The tangent is
        // `ragged_dot(ȧ, b, g) + ragged_dot(a, ḃ, g) = [10, 10, 50, 50] + [1, 4, 13, 28]`.
        let left = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let right = Array::matrix(3, 1, vec![10.0, 20.0, 30.0]).unwrap();
        let group_sizes = Array::vector(vec![1i32, 2]).unwrap();
        let output_type = ArrayType::new_static(DataType::F64, [2, 2, 1]);
        assert_eq!(
            differentiate_at((left, right)).jvp(
                (Array::matrix(2, 3, vec![1.0; 6]).unwrap(), Array::matrix(3, 1, vec![1.0, 2.0, 3.0]).unwrap()),
                |(left, right)| {
                    let group_sizes = left.domain().lift(group_sizes.clone())?;
                    left.ragged_dot(
                        &right,
                        &group_sizes,
                        &RaggedDotDimensionNumbers::new(DotDimensionNumbers::matmul(), 1, None),
                    )
                },
            ),
            Ok((
                Array::from_elements(output_type.clone(), &[10.0, 40.0, 130.0, 280.0]).unwrap(),
                Array::from_elements(output_type, &[11.0, 14.0, 63.0, 78.0]).unwrap(),
            )),
        );

        // A staged JVP reuses the known group metadata in both product rule terms.
        let lhs = Array::matrix(3, 2, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs = Array::from_elements::<f32>(plain_array(&[2, 2, 1]), &[10.0, 1.0, 2.0, 3.0]).unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let lhs_input = builder.add_input(lhs.r#type().into_owned());
        let rhs_input = builder.add_input(rhs.r#type().into_owned());
        let group_sizes = builder.add_constant(Array::vector(vec![1i32, 2]).unwrap());
        let output = builder
            .add_instruction(
                RaggedDotOperation::new(RaggedDotDimensionNumbers::matmul()),
                vec![],
                vec![lhs_input, rhs_input, group_sizes],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f32[3, 2], %1:f32[2, 2, 1], %2:f32[3, 2], %3:f32[2, 2, 1] .
                let %4:i32[2] = const [1, 2]
                    %5:f32[3, 1] = ragged_dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[], \
                            rhs_batching=[], lhs_ragged=0, rhs_group=0),
                    ] %0 %1 %4
                    %6:f32[3, 1] = ragged_dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[], \
                            rhs_batching=[], lhs_ragged=0, rhs_group=0),
                    ] %2 %1 %4
                    %7:f32[3, 1] = ragged_dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[], \
                            rhs_batching=[], lhs_ragged=0, rhs_group=0),
                    ] %0 %3 %4
                    %8:f32[3, 1] = add %6 %7
                in (%5, %8)
            "}
            .trim_end(),
        );
        assert_eq!(
            jvp.interpret(vec![
                lhs.clone(),
                rhs.clone(),
                Array::matrix(3, 2, vec![1.0f32; 6]).unwrap(),
                Array::from_elements::<f32>(plain_array(&[2, 2, 1]), &[1.0; 4]).unwrap(),
            ]),
            Ok(vec![
                Array::matrix(3, 1, vec![12.0f32, 18.0, 28.0]).unwrap(),
                Array::matrix(3, 1, vec![14.0f32, 12.0, 16.0]).unwrap(),
            ]),
        );

        // A structural zero tangent drops its product rule term. Here, the constant right-hand side has a zero tangent.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let lhs_input = builder.add_input(lhs.r#type().into_owned());
        let rhs_constant = builder.add_constant(rhs);
        let group_sizes = builder.add_constant(Array::vector(vec![1i32, 2]).unwrap());
        let output = builder
            .add_instruction(
                RaggedDotOperation::new(RaggedDotDimensionNumbers::matmul()),
                vec![],
                vec![lhs_input, rhs_constant, group_sizes],
                None,
            )
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[3, 2], %1:f32[3, 2] .
                let %2:f32[2, 2, 1] = const [[[10.0], [1.0]], [[2.0], [3.0]]]
                    %3:i32[2] = const [1, 2]
                    %4:f32[3, 1] = ragged_dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[], \
                            rhs_batching=[], lhs_ragged=0, rhs_group=0),
                    ] %0 %2 %3
                    %5:f32[3, 1] = ragged_dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[], \
                            rhs_batching=[], lhs_ragged=0, rhs_group=0),
                    ] %1 %2 %3
                in (%4, %5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_ragged_dot_differentiation_widened_tangents() {
        let lhs_type = ArrayType::new_static(DataType::F8E8M0FNU, [3, 2]);
        let rhs_type = ArrayType::new_static(DataType::F8E8M0FNU, [3, 2, 1]);
        let dimensions =
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]), 0, None);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let lhs = builder.add_input(lhs_type.clone());
        let rhs = builder.add_input(rhs_type.clone());
        let group_sizes = builder.add_constant(Array::vector(vec![1i32, 2]).unwrap());
        let output = builder
            .add_instruction(RaggedDotOperation::new(dimensions), vec![], vec![lhs, rhs, group_sizes], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f8e8m0fnu[3, 2], %1:f8e8m0fnu[3, 2, 1] .
                let %2:i32[2] = const [1, 2]
                    %3:f8e8m0fnu[3, 1] = ragged_dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[0], \
                            rhs_batching=[0], lhs_ragged=0),
                    ] %0 %1 %2
                in (%3)
            "}
            .trim_end(),
        );

        // The tangents of `f8e8m0fnu` primals are widened to `f32`, so each product rule term converts its primal
        // factor to the tangent element type before the contraction.
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f8e8m0fnu[3, 2], %1:f8e8m0fnu[3, 2, 1], %2:f32[3, 2], %3:f32[3, 2, 1] .
                let %4:i32[2] = const [1, 2]
                    %5:f8e8m0fnu[3, 1] = ragged_dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[0], \
                            rhs_batching=[0], lhs_ragged=0),
                    ] %0 %1 %4
                    %6:f32[3, 2, 1] = convert_element_type [data_type=f32] %1
                    %7:f32[3, 1] = ragged_dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[0], \
                            rhs_batching=[0], lhs_ragged=0),
                    ] %2 %6 %4
                    %8:f32[3, 2] = convert_element_type [data_type=f32] %0
                    %9:f32[3, 1] = ragged_dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[0], \
                            rhs_batching=[0], lhs_ragged=0),
                    ] %8 %3 %4
                    %10:f32[3, 1] = add %7 %9
                in (%5, %10)
            "}
            .trim_end(),
        );
        assert_eq!(
            jvp.interpret(vec![
                Array::from_elements(lhs_type, &[f8e8m0fnu::from_bits(127); 6]).unwrap(),
                Array::from_elements(rhs_type, &[f8e8m0fnu::from_bits(127); 6]).unwrap(),
                Array::from_elements(ArrayType::new_static(DataType::F32, [3, 2]), &[1.0f32; 6]).unwrap(),
                Array::from_elements(ArrayType::new_static(DataType::F32, [3, 2, 1]), &[1.0f32; 6]).unwrap(),
            ]),
            Ok(vec![
                Array::from_elements(
                    ArrayType::new_static(DataType::F8E8M0FNU, [3, 1]),
                    &[f8e8m0fnu::from_bits(128); 3],
                )
                .unwrap(),
                Array::from_elements(ArrayType::new_static(DataType::F32, [3, 1]), &[4.0f32; 3]).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_ragged_dot_transposition() {
        let lhs = Array::matrix(3, 2, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs = Array::from_elements::<f32>(plain_array(&[2, 2, 1]), &[10.0, 1.0, 2.0, 3.0]).unwrap();
        let group_sizes = Array::vector(vec![1i32, 2]).unwrap();
        // The left-hand side adjoint is a non-contracting ragged dot of the output cotangent with the right-hand side,
        // and the right-hand side adjoint is a contracting ragged dot of the left-hand side with the output cotangent.
        check_operation_transposition!(
            @exact,
            operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::matmul()),
            cases = [
                {
                    inputs = [
                        (@linear(type = lhs.r#type().into_owned())),
                        (@known, rhs.clone()),
                        (@known, group_sizes.clone()),
                    ],
                    output_cotangents = [Array::matrix(3, 1, vec![2.0f32, 3.0, 5.0]).unwrap()],
                    input_cotangents = [Array::matrix(3, 2, vec![20.0f32, 2.0, 6.0, 9.0, 10.0, 15.0]).unwrap()],
                    pullback = indoc! {"
                        lambda %0:f32[3, 1], %1:f32[2, 2, 1], %2:i32[2] .
                        let %3:f32[3, 2] = ragged_dot [
                            dimensions=(lhs_contracting=[1], rhs_contracting=[2], lhs_batching=[], \
                                rhs_batching=[], lhs_ragged=0, rhs_group=0),
                        ] %0 %1 %2
                        in (%3)
                    "},
                },
                {
                    inputs = [
                        (@known, lhs),
                        (@linear(type = rhs.r#type().into_owned())),
                        (@known, group_sizes),
                    ],
                    output_cotangents = [Array::matrix(3, 1, vec![2.0f32, 3.0, 5.0]).unwrap()],
                    input_cotangents = [Array::from_elements::<f32>(
                        plain_array(&[2, 2, 1]),
                        &[2.0, 4.0, 34.0, 42.0],
                    ).unwrap()],
                    pullback = indoc! {"
                        lambda %0:f32[3, 1], %1:f32[3, 2], %2:i32[2] .
                        let %3:f32[2, 2, 1] = ragged_dot [
                            dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], \
                                rhs_batching=[], lhs_ragged=0),
                        ] %1 %0 %2
                        in (%3)
                    "},
                },
            ],
        );
    }

    #[test]
    fn test_ragged_dot_transposition_inverts_permuted_output_axes() {
        let dimensions =
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![1], vec![1], vec![2], vec![2]), 0, Some(0));
        let lhs = Array::from_elements::<f32>(
            plain_array(&[3, 2, 2]),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )
        .unwrap();
        let rhs =
            Array::from_elements::<f32>(plain_array(&[2, 2, 2, 1]), &[2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0])
                .unwrap();
        let group_sizes = Array::vector(vec![1i32, 2]).unwrap();
        let output_cotangent =
            Array::from_elements::<f32>(plain_array(&[2, 3, 1]), &[1.0, 2.0, 3.0, 5.0, 7.0, 11.0]).unwrap();

        // The left-hand side adjoint is initially ordered `[batch, ragged, contracting]` and must be transposed back
        // to `[ragged, contracting, batch]` with `[1, 2, 0]`.
        check_operation_transposition!(
            @exact,
            operation = RaggedDotOperation::new(dimensions.clone()),
            cases = [{
                inputs = [
                    (@linear(type = lhs.r#type().into_owned())),
                    (@known, rhs.clone()),
                    (@known, group_sizes.clone()),
                ],
                output_cotangents = [output_cotangent.clone()],
                input_cotangents = [Array::from_elements::<f32>(
                    plain_array(&[3, 2, 2]),
                    &[2.0, 15.0, 5.0, 35.0, 22.0, 91.0, 34.0, 133.0, 33.0, 143.0, 51.0, 209.0],
                ).unwrap()],
                pullback = indoc! {"
                    lambda %0:f32[2, 3, 1], %1:f32[2, 2, 2, 1], %2:i32[2] .
                    let %3:f32[2, 3, 2] = ragged_dot [
                        dimensions=(lhs_contracting=[2], rhs_contracting=[3], lhs_batching=[0], \
                            rhs_batching=[2], lhs_ragged=1, rhs_group=0),
                    ] %0 %1 %2
                        %4:f32[3, 2, 2] = transpose [permutation=[1, 2, 0]] %3
                    in (%4)
                "},
            }],
        );

        // The right-hand side adjoint is initially ordered `[group, batch, contracting, result]` and must be
        // transposed back to `[group, contracting, batch, result]` with `[0, 2, 1, 3]`.
        check_operation_transposition!(
            @exact,
            operation = RaggedDotOperation::new(dimensions),
            cases = [{
                inputs = [
                    (@known, lhs),
                    (@linear(type = rhs.r#type().into_owned())),
                    (@known, group_sizes),
                ],
                output_cotangents = [output_cotangent],
                input_cotangents = [Array::from_elements::<f32>(
                    plain_array(&[2, 2, 2, 1]),
                    &[1.0, 10.0, 3.0, 20.0, 37.0, 152.0, 47.0, 188.0],
                ).unwrap()],
                pullback = indoc! {"
                    lambda %0:f32[2, 3, 1], %1:f32[3, 2, 2], %2:i32[2] .
                    let %3:f32[2, 2, 2, 1] = ragged_dot [
                        dimensions=(lhs_contracting=[0], rhs_contracting=[1], lhs_batching=[2], \
                            rhs_batching=[0], lhs_ragged=0),
                    ] %1 %0 %2
                        %4:f32[2, 2, 2, 1] = transpose [permutation=[0, 2, 1, 3]] %3
                    in (%4)
                "},
            }],
        );
    }

    #[test]
    fn test_ragged_dot_transposition_rejects_contracting_and_batch_modes() {
        let cases = [
            (
                RaggedDotOperation::new(RaggedDotDimensionNumbers::new(DotDimensionNumbers::matmul(), 1, None)),
                [plain_array(&[2, 5]), plain_array(&[5, 4]), ArrayType::new_static(DataType::I32, [3])],
                RaggedDotMode::Contracting,
            ),
            (
                RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
                    DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]),
                    0,
                    None,
                )),
                [plain_array(&[5, 2]), plain_array(&[5, 2, 4]), ArrayType::new_static(DataType::I32, [3])],
                RaggedDotMode::Batch,
            ),
        ];
        for (operation, input_types, mode) in cases {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let inputs = input_types.map(|input_type| builder.add_input(input_type));
            let outputs = builder.add_instruction(operation, vec![], inputs.to_vec(), None).unwrap().to_vec();
            let program =
                builder.build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 3], vec![Placeholder]).unwrap();
            assert_eq!(
                program.transpose_with_respect_to(&[0], &[]).unwrap_err(),
                DifferentiationError::Program(ProgramError::UnsupportedOperation {
                    message: format!("`{RAGGED_DOT_OPERATION_NAME}` transposition is unsupported in `{mode}` mode"),
                }),
            );
        }
    }

    #[test]
    fn test_ragged_dot_transposition_rejects_two_linear_inputs() {
        // The bilinear rejection precedes the early return for a structural zero output cotangent.
        let mut context = TranspositionContext::new(TracingContext::<Array, ArrayOperation<Array>>::new());
        let group_sizes = context.input(ArrayType::new_static(DataType::I32, [2]));
        let inputs = [
            PartialValue::Unknown(plain_array(&[3, 2])),
            PartialValue::Unknown(plain_array(&[2, 2, 1])),
            PartialValue::Known(group_sizes),
        ];
        let accumulators = context.cotangent_accumulators(&inputs, &[true, true, false]).unwrap();
        assert_eq!(
            RaggedDotOperation::new(RaggedDotDimensionNumbers::matmul()).transpose(
                &mut context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Zero(plain_array(&[3, 1]))],
                &accumulators,
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation {
                message: format!("bilinear `{RAGGED_DOT_OPERATION_NAME}` with two linear inputs cannot be transposed"),
            })),
        );
    }

    #[test]
    fn test_ragged_dot_transposition_rejects_unknown_group_sizes() {
        let mut context = TranspositionContext::new(TracingContext::<Array, ArrayOperation<Array>>::new());
        let right = context.input(plain_array(&[2, 2, 1]));
        let seed = context.input(plain_array(&[3, 1]));
        let inputs = [
            PartialValue::Unknown(plain_array(&[3, 2])),
            PartialValue::Known(right),
            PartialValue::Unknown(ArrayType::new_static(DataType::I32, [2])),
        ];
        let accumulators = context.cotangent_accumulators(&inputs, &[true, false, false]).unwrap();
        assert_eq!(
            RaggedDotOperation::new(RaggedDotDimensionNumbers::matmul()).transpose(
                &mut context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Value(seed)],
                &accumulators,
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation {
                message: format!("`{RAGGED_DOT_OPERATION_NAME}` group sizes must be known during transposition"),
            })),
        );
    }

    #[test]
    fn test_ragged_dot_transposition_unrequested_cotangent() {
        // An unrequested linear input stages no adjoint and keeps a structural zero cotangent.
        let mut context = TranspositionContext::new(TracingContext::<Array, ArrayOperation<Array>>::new());
        let right = context.input(plain_array(&[2, 2, 1]));
        let group_sizes = context.input(ArrayType::new_static(DataType::I32, [2]));
        let seed = context.input(plain_array(&[3, 1]));
        let inputs =
            [PartialValue::Unknown(plain_array(&[3, 2])), PartialValue::Known(right), PartialValue::Known(group_sizes)];
        let accumulators = context.cotangent_accumulators(&inputs, &[false, false, false]).unwrap();
        RaggedDotOperation::new(RaggedDotDimensionNumbers::matmul())
            .transpose(&mut context, &EmptyRegionDriver, &inputs, &[MaybeZero::Value(seed)], &accumulators)
            .unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![], vec![Placeholder; 3], vec![])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 2, 1], %1:i32[2], %2:f32[3, 1] .
                in ()
            "}
            .trim_end(),
        );
        assert!(context.take_cotangents(&accumulators).unwrap().iter().all(MaybeZero::is_zero));
    }

    #[test]
    fn test_ragged_dot_transposition_structural_zero() {
        // A structural zero output cotangent stages no adjoint and leaves the requested cotangent zero.
        let mut context = TranspositionContext::new(TracingContext::<Array, ArrayOperation<Array>>::new());
        let right = context.input(plain_array(&[2, 2, 1]));
        let group_sizes = context.input(ArrayType::new_static(DataType::I32, [2]));
        let inputs =
            [PartialValue::Unknown(plain_array(&[3, 2])), PartialValue::Known(right), PartialValue::Known(group_sizes)];
        let accumulators = context.cotangent_accumulators(&inputs, &[true, false, false]).unwrap();
        RaggedDotOperation::new(RaggedDotDimensionNumbers::matmul())
            .transpose(
                &mut context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Zero(plain_array(&[3, 1]))],
                &accumulators,
            )
            .unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 2, 1], %1:i32[2] .
                in ()
            "}
            .trim_end(),
        );
        assert!(context.take_cotangents(&accumulators).unwrap().iter().all(MaybeZero::is_zero));
    }

    #[test]
    fn test_ragged_dot_transposition_known_inputs() {
        // A ragged dot with two known data inputs is not part of the linear map, so transposition stages nothing.
        let mut context = TranspositionContext::new(TracingContext::<Array, ArrayOperation<Array>>::new());
        let left = context.input(plain_array(&[3, 2]));
        let right = context.input(plain_array(&[2, 2, 1]));
        let group_sizes = context.input(ArrayType::new_static(DataType::I32, [2]));
        let seed = context.input(plain_array(&[3, 1]));
        let inputs = [PartialValue::Known(left), PartialValue::Known(right), PartialValue::Known(group_sizes)];
        let accumulators = context.cotangent_accumulators(&inputs, &[false, false, false]).unwrap();
        RaggedDotOperation::new(RaggedDotDimensionNumbers::matmul())
            .transpose(&mut context, &EmptyRegionDriver, &inputs, &[MaybeZero::Value(seed)], &accumulators)
            .unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![], vec![Placeholder; 4], vec![])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3, 2], %1:f32[2, 2, 1], %2:i32[2], %3:f32[3, 1] .
                in ()
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_ragged_dot_transposition_validates_original_dimensions() {
        let mut context = TranspositionContext::new(TracingContext::<Array, ArrayOperation<Array>>::new());
        let right = context.input(plain_array(&[2, 2, 1]));
        let group_sizes = context.input(ArrayType::new_static(DataType::I32, [2]));
        let seed = context.input(plain_array(&[3, 1]));
        let inputs =
            [PartialValue::Unknown(plain_array(&[3, 2])), PartialValue::Known(right), PartialValue::Known(group_sizes)];
        let accumulators = context.cotangent_accumulators(&inputs, &[true, false, false]).unwrap();
        let operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![4], vec![1], vec![], vec![]),
            0,
            Some(0),
        ));
        assert_eq!(
            operation.transpose(&mut context, &EmptyRegionDriver, &inputs, &[MaybeZero::Value(seed)], &accumulators),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` left-hand side dimension index out of bounds",
            ))))),
        );
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![], vec![Placeholder; 3], vec![])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 2, 1], %1:i32[2], %2:f32[3, 1] .
                in ()
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_ragged_dot_array_ir() {
        // Array IR tracers stage the homogeneous ragged dot as an array member instruction.
        let (_, program) = ArrayIrContext::trace(
            |inputs: Vec<Tracer<ArrayIrContext>>| {
                inputs[0].ragged_dot(&inputs[1], &inputs[2], &RaggedDotDimensionNumbers::matmul())
            },
            vec![
                ArrayIrType::Array(plain_array(&[3, 2])),
                ArrayIrType::Array(plain_array(&[2, 2, 1])),
                ArrayIrType::Array(ArrayType::new_static(DataType::I32, [2])),
            ],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3, 2], %1:f32[2, 2, 1], %2:i32[2] .
                let %3:f32[3, 1] = ragged_dot [
                    dimensions=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[], rhs_batching=[], \
                        lhs_ragged=0, rhs_group=0),
                ] %0 %1 %2
                in (%3)
            "}
            .trim_end(),
        );

        // Concrete array IR values contract their array members.
        let lhs = ArrayIrValue::Array(Array::matrix(3, 2, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        let rhs =
            ArrayIrValue::Array(Array::from_elements::<f32>(plain_array(&[2, 2, 1]), &[10.0, 1.0, 2.0, 3.0]).unwrap());
        let group_sizes = ArrayIrValue::Array(Array::vector(vec![1i32, 2]).unwrap());
        assert_eq!(
            lhs.ragged_dot(&rhs, &group_sizes, &RaggedDotDimensionNumbers::matmul()),
            Ok(ArrayIrValue::Array(Array::matrix(3, 1, vec![12.0f32, 18.0, 28.0]).unwrap())),
        );
    }
}
