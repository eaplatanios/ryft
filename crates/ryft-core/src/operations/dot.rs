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
    StaticShape,
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
use crate::macros::{check_count, dispatch_on_array_element_type, impl_differentiable_operation};
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

    // TODO(eaplatanios): Review from here onwards.

    /// Validates paired list lengths and the distinct, disjoint axis indices on both inputs before indexing or lifting.
    /// `operation_name` names the operation that owns these dimension numbers in diagnostics.
    pub(crate) fn validate_for_ranks(
        &self,
        operation_name: &str,
        lhs_rank: usize,
        rhs_rank: usize,
    ) -> Result<(), TypeError> {
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
            "LHS",
        )?;
        validate_dot_axes(
            operation_name,
            &self.rhs_batching_dimensions,
            &self.rhs_contracting_dimensions,
            rhs_rank,
            "RHS",
        )
    }

    /// Returns the dimension numbers that map the output cotangent of `t ↦ dot(t, rhs)` back toward the linear
    /// left-hand side `t`, together with the left-hand side axes in the order in which that adjoint contraction emits
    /// them. These dimension numbers must describe a valid contraction of a left-hand side with rank `lhs_rank` and a
    /// right-hand side with rank `rhs_rank`.
    ///
    /// The adjoint contraction is `dot(cotangent, rhs)`, and it emits the batching axes, the left-hand side result
    /// axes, and then the left-hand side contracting axes sorted by their paired right-hand side axes. Callers must
    /// request any output sharding in that intermediate order and then restore the original left-hand side axis order
    /// with the inverse permutation.
    fn lhs_cotangent_dimensions(&self, lhs_rank: usize, rhs_rank: usize) -> (Self, Vec<usize>) {
        let batching_count = self.rhs_batching_dimensions.len();
        let lhs_result_dimensions = self.lhs_result_dimensions(lhs_rank);
        let rhs_result_dimensions = self.rhs_result_dimensions(rhs_rank);
        let cotangent_rhs_result_start = batching_count + lhs_result_dimensions.len();
        let dimensions = Self::new(
            (cotangent_rhs_result_start..(cotangent_rhs_result_start + rhs_result_dimensions.len())).collect(),
            rhs_result_dimensions,
            (0..batching_count).collect(),
            self.rhs_batching_dimensions.clone(),
        );
        let mut contracting_pairs = self
            .lhs_contracting_dimensions
            .iter()
            .copied()
            .zip(self.rhs_contracting_dimensions.iter().copied())
            .collect::<Vec<_>>();
        contracting_pairs.sort_by_key(|(_, rhs_axis)| *rhs_axis);
        let output_axes = self
            .lhs_batching_dimensions
            .iter()
            .copied()
            .chain(lhs_result_dimensions)
            .chain(contracting_pairs.into_iter().map(|(lhs_axis, _)| lhs_axis))
            .collect();
        (dimensions, output_axes)
    }

    /// Returns the dimension numbers that map the output cotangent of `t ↦ dot(lhs, t)` back toward the linear
    /// right-hand side `t`, together with the right-hand side axes in the order in which that adjoint contraction emits
    /// them. These dimension numbers must describe a valid contraction of a left-hand side with rank `lhs_rank` and a
    /// right-hand side with rank `rhs_rank`.
    ///
    /// The adjoint contraction is `dot(lhs, cotangent)`, and it emits the batching axes, the right-hand side
    /// contracting axes sorted by their paired left-hand side axes, and then the right-hand side result axes. Callers
    /// must request any output sharding in that intermediate order and then restore the original right-hand side axis
    /// order with the inverse permutation.
    fn rhs_cotangent_dimensions(&self, lhs_rank: usize, rhs_rank: usize) -> (Self, Vec<usize>) {
        let batching_count = self.lhs_batching_dimensions.len();
        let lhs_result_dimensions = self.lhs_result_dimensions(lhs_rank);
        let cotangent_lhs_result_count = lhs_result_dimensions.len();
        let dimensions = Self::new(
            lhs_result_dimensions,
            (batching_count..(batching_count + cotangent_lhs_result_count)).collect(),
            self.lhs_batching_dimensions.clone(),
            (0..batching_count).collect(),
        );
        let mut contracting_pairs = self
            .rhs_contracting_dimensions
            .iter()
            .copied()
            .zip(self.lhs_contracting_dimensions.iter().copied())
            .collect::<Vec<_>>();
        contracting_pairs.sort_by_key(|(_, lhs_axis)| *lhs_axis);
        let output_axes = self
            .rhs_batching_dimensions
            .iter()
            .copied()
            .chain(contracting_pairs.into_iter().map(|(rhs_axis, _)| rhs_axis))
            .chain(self.rhs_result_dimensions(rhs_rank))
            .collect();
        (dimensions, output_axes)
    }
}

impl Display for DotDimensionNumbers {
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

/// Primitive representing a generalized dot (tensor contraction) of two identically typed numeric inputs. Boolean,
/// token, and structural zero element types are rejected.
///
/// [`DotOperation`] is the unified primitive for matrix multiplication, batched matrix multiplication, vector inner
/// products, and arbitrary tensor contractions. The XLA backend lowers it to StableHLO's
/// [`dot_general`](https://openxla.org/stablehlo/spec#dot_general) instruction.
///
/// Without an accumulation type, the output keeps the input element type. The eager kernel accumulates floating-point
/// inputs narrower than `f32` at `f32` and rounds each result once, which is how XLA backends evaluate such dots.
/// Integer products and sums wrap at the declared element width, and every other element type accumulates at itself.
///
/// A dot is bilinear. Forward mode differentiation applies `d(dot(lhs, rhs)) = dot(dlhs, rhs) + dot(lhs, drhs)`.
/// Transposition therefore accepts exactly one linear input and contracts the output cotangent with the other, known
/// input using the corresponding adjoint dimension numbers. Dots with an accumulation type perform those contractions
/// at the widened cotangent type before converting the result back to the linear input's cotangent representation.
///
/// Each forward mode tangent term is staged as an ordinary dot that preserves the primal dimension numbers,
/// accumulation type, and requested output sharding without introducing captures. The adjoint contraction emits the
/// linear input's axes in dot order, so transposition requests the cotangent dual of the linear input's sharding in
/// that order and then restores the original axis order and placement with the inverse permutation. A structural zero
/// output cotangent remains structural zero.
///
/// Batching aligns the inputs onto a common mapped axis and lifts the dimension numbers past it. Materializing an axis
/// on a replicated input preserves a dynamic mapped extent as a first-class value. Two mapped inputs must describe the
/// same mapped extent. For dynamic extents, they must share the same
/// [`DimensionVariable`](crate::arrays::DimensionVariable).
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
    /// Creates a new [`DotOperation`] with the supplied dimension numbers.
    #[inline]
    pub fn new(dimensions: DotDimensionNumbers) -> Self {
        Self { dimensions, accumulation_type: None, output_sharding: None }
    }

    /// Returns a [`DotOperation`] configured for rank-two matrix multiplication.
    #[inline]
    pub fn matmul() -> Self {
        Self::new(DotDimensionNumbers::matmul())
    }

    /// Returns this [`DotOperation`] with a requested output [`Sharding`]. Type inference validates its rank, its mesh,
    /// the absence of auto axes, and the rule for unreduced outputs before using it instead of the inferred sharding.
    /// The request bypasses the ordinary consistency checks for batching and contracting dimension shardings. This is
    /// the only way to produce an output with unreduced axes, which represents partial results on each device whose
    /// reduction across devices is delayed. Logical dependence on manual shards still comes from the input values.
    #[inline]
    pub fn with_output_sharding(mut self, output_sharding: impl Into<Option<Sharding>>) -> Self {
        self.output_sharding = output_sharding.into();
        self
    }

    /// Returns this [`DotOperation`] with the provided accumulation data type. The input element types must still match
    /// each other and must support the accumulation type, which becomes the output element type. The eager kernel
    /// upcasts the inputs and accumulates the contraction at the wider type. This is the `preferred_element_type`
    /// contract of XLA, which its low precision matrix units implement natively (e.g., `f8 × f8 → f32` and
    /// `bf16 × bf16 → f32`). Dots with an accumulation type differentiate like ordinary dots, with tangents and
    /// cotangents carried at the accumulation type (refer to the forward mode and transposition documentation of this
    /// operation), and cannot yet be combined with a requested output sharding.
    #[inline]
    pub fn with_accumulation_type(mut self, accumulation_type: impl Into<Option<DataType>>) -> Self {
        self.accumulation_type = accumulation_type.into();
        self
    }

    /// Returns the contracting and batching dimension specification.
    #[inline]
    pub fn dimensions(&self) -> &DotDimensionNumbers {
        &self.dimensions
    }

    /// Returns the optional accumulation data type. Refer to the documentation of [`Self::with_accumulation_type`].
    #[inline]
    pub fn accumulation_type(&self) -> Option<DataType> {
        self.accumulation_type
    }

    /// Returns the requested output sharding, if any.
    #[inline]
    pub fn output_sharding(&self) -> Option<&Sharding> {
        self.output_sharding.as_ref()
    }

    /// Applies this operation to `lhs` and `rhs` through the [`Dot`] capability, forwarding the accumulation type or
    /// the requested output sharding, so that interpretation and differentiation over staging values preserve them.
    fn apply<V: Dot>(&self, lhs: &V, rhs: &V) -> Result<V, ProgramError> {
        validate_dot_options(self.accumulation_type, self.output_sharding.as_ref())?;
        match (self.accumulation_type, &self.output_sharding) {
            (Some(accumulation_type), _) => lhs.dot_with_accumulation_type(rhs, &self.dimensions, accumulation_type),
            (None, Some(output_sharding)) => lhs.dot_with_output_sharding(rhs, &self.dimensions, output_sharding),
            (None, None) => lhs.dot(rhs, &self.dimensions),
        }
    }
}

impl Display for DotOperation {
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
        Ok(vec![dot_abstract(
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

// Partial evaluation uses the default fold-or-residualize behavior.
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for DotOperation where
    C::Operation: From<DotOperation>
{
}

impl<C: Context<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>>
    for DotOperation
where
    DotOperation: InterpretableOperation<C>,
{
    // Alignment delegates mapped axis materialization to `ArrayExtentBatchingPolicy::match_axis`, so a dynamic mapped
    // extent remains a first-class value. The lifted contraction keeps each item independent across the inserted axis.
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 2, ProgramError);
        self.dimensions.validate_for_ranks(
            DOT_OPERATION_NAME,
            inputs[0].unbatched_type().rank(),
            inputs[1].unbatched_type().rank(),
        )?;
        let batch_axes: Vec<Option<usize>> = inputs.iter().map(|input| input.batch_axis_position()).collect();

        check_common_mapped_dimension(DOT_OPERATION_NAME, inputs)?;

        // A mixed pair needs a physical batch axis on its replicated input before the logical dimension numbers can be
        // lifted. Insert that axis at position zero through the active policy, so a dynamic mapped extent remains an
        // extent value grounding the broadcast instead of requiring a statically known host size.
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
        Ok(lifted_operation.interpret_with_batch_axes(context, &aligned_inputs, &output_batch_axes)?)
    }
}

// Forward mode differentiation applies the product rule. Each term holds the corresponding primal input fixed on its
// original contracting side and stages an ordinary dot with the primal operation's dimensions, accumulation type, and
// requested output sharding.
impl_differentiable_operation! {
    DotOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Operation: From<DotOperation>,
        C::Value: ConvertElementType + Dot + Add,
    {
        |operation, context, _driver, inputs| {
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
            // multiplication or element conversion is needed for its cotangent. A zero output cotangent stays a
            // structural zero.
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
            let (dimensions, output_axes) = if linear_index == 0 {
                operation.dimensions().lhs_cotangent_dimensions(left_rank, right_rank)
            } else {
                operation.dimensions().rhs_cotangent_dimensions(left_rank, right_rank)
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

/// Value-level generalized dot capability.
///
/// [`Dot`] is the receiver-style entry point for staging or executing [`DotOperation`]. It performs the contraction
/// described by `dimensions`, supporting standard matrix multiplication, batched matrix multiplication, vector inner
/// products, and arbitrary tensor contractions.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and array IR values implement it for [`ArrayIrType`].
#[capability(projection(ArrayIrType => ArrayType))]
pub trait Dot<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Computes the generalized dot product of `self` and `rhs` using `dimensions`, and returns a [`ProgramError`] if
    /// the inputs are incompatible with `dimensions` or the contraction cannot be recorded in the value's context.
    fn dot(&self, rhs: &Self, dimensions: &DotDimensionNumbers) -> Result<Self, ProgramError>;

    /// Computes the generalized dot product of `self` and `rhs` using `dimensions`, requesting `output_sharding` for
    /// the result. The requested sharding overrides the inferred output sharding and is validated by the operation's
    /// type inference (refer to the documentation of [`DotOperation::with_output_sharding`]). Staging values attach it
    /// to the staged operation, array IR values project it onto their array members, and concrete [`Array`]s record it
    /// on their result.
    fn dot_with_output_sharding(
        &self,
        rhs: &Self,
        dimensions: &DotDimensionNumbers,
        output_sharding: &Sharding,
    ) -> Result<Self, ProgramError>;

    /// Computes the generalized dot product of `self` and `rhs` using `dimensions`, upcasting the inputs to
    /// `accumulation_type` and accumulating the contraction there, so the result carries the accumulation type. Refer
    /// to the documentation of [`DotOperation::with_accumulation_type`] for the exact contract.
    fn dot_with_accumulation_type(
        &self,
        rhs: &Self,
        dimensions: &DotDimensionNumbers,
        accumulation_type: DataType,
    ) -> Result<Self, ProgramError>;
}

impl Dot for Array {
    fn dot(&self, rhs: &Self, dimensions: &DotDimensionNumbers) -> Result<Self, ProgramError> {
        let data_type = self.r#type().data_type();
        validate_dot_data_type(DOT_OPERATION_NAME, data_type)?;
        dispatch_on_array_element_type!(
            @numeric data_type,
            |Element| self.dot_elements::<Element>(rhs, dimensions, None),
        )
    }

    fn dot_with_output_sharding(
        &self,
        rhs: &Self,
        dimensions: &DotDimensionNumbers,
        output_sharding: &Sharding,
    ) -> Result<Self, ProgramError> {
        // An `Array` is a concrete value on a single device whose payload is always its logical (i.e., fully reduced)
        // value, so the requested sharding only describes the placement of its result, exactly as for `Reshard`. That
        // holds for unreduced axes as well, which describe partial sums whose total is that payload. The `DotOperation`
        // type inference rule validates the request exactly as it does for staged programs and determines the result
        // type.
        let data_type = self.r#type().data_type();
        validate_dot_data_type(DOT_OPERATION_NAME, data_type)?;
        dispatch_on_array_element_type!(
            @numeric data_type,
            |Element| self.dot_elements::<Element>(rhs, dimensions, Some(output_sharding)),
        )
    }

    fn dot_with_accumulation_type(
        &self,
        rhs: &Self,
        dimensions: &DotDimensionNumbers,
        accumulation_type: DataType,
    ) -> Result<Self, ProgramError> {
        DotOperation::new(dimensions.clone())
            .with_accumulation_type(accumulation_type)
            .infer_output_types(&[self.r#type().into_owned(), rhs.r#type().into_owned()], &[])?;
        let lhs = self.convert_element_type(accumulation_type)?;
        let rhs = rhs.convert_element_type(accumulation_type)?;
        lhs.dot(&rhs, dimensions)
    }
}

// Values that carry a context stage a dot through that context. The `ValueDomainDispatch` marker keeps this
// implementation disjoint from eager values, which implement the capability directly.
impl<V: Value<Type = ArrayType, Dispatch = ValueDomainDispatch> + ManualVariationAlignment<ArrayType>> Dot<ArrayType>
    for V
where
    V::Domain: Context<Type = ArrayType>,
    <V::Domain as Domain>::Operation: From<DotOperation>,
{
    fn dot(&self, rhs: &Self, dimensions: &DotDimensionNumbers) -> Result<Self, ProgramError> {
        let inputs = [self.clone(), rhs.clone()];
        let inputs = ManualVariationAlignment::align_manual_variation(&inputs)?;
        let mut outputs = self.domain().bind(DotOperation::new(dimensions.clone()), Vec::new(), &inputs)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }

    fn dot_with_output_sharding(
        &self,
        rhs: &Self,
        dimensions: &DotDimensionNumbers,
        output_sharding: &Sharding,
    ) -> Result<Self, ProgramError> {
        let inputs = [self.clone(), rhs.clone()];
        let inputs = ManualVariationAlignment::align_manual_variation(&inputs)?;
        let mut outputs = self.domain().bind(
            DotOperation::new(dimensions.clone()).with_output_sharding(output_sharding.clone()),
            Vec::new(),
            &inputs,
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }

    fn dot_with_accumulation_type(
        &self,
        rhs: &Self,
        dimensions: &DotDimensionNumbers,
        accumulation_type: DataType,
    ) -> Result<Self, ProgramError> {
        let inputs = [self.clone(), rhs.clone()];
        let inputs = ManualVariationAlignment::align_manual_variation(&inputs)?;
        let mut outputs = self.domain().bind(
            DotOperation::new(dimensions.clone()).with_accumulation_type(accumulation_type),
            Vec::new(),
            &inputs,
        )?;
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
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::NonContracting => "non-contracting",
            Self::Contracting => "contracting",
            Self::Batch => "batch",
        })
    }
}

/// Dimension number specification for a grouped generalized dot product.
///
/// The ordinary [`DotDimensionNumbers`] describe the contraction and batching axes. Exactly one left-hand side axis is
/// marked ragged, and right-hand side group dimensions identify the explicit group axis used by non-contracting ragged
/// dots. Refer to [`RaggedDotOperation`] for the semantics of each [`RaggedDotMode`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RaggedDotDimensionNumbers {
    /// Underlying generalized dot dimension numbers.
    dot_dimensions: DotDimensionNumbers,

    /// Left-hand side axes marked ragged. Type inference requires exactly one entry.
    lhs_ragged_dimensions: Vec<usize>,

    /// Right-hand side axes indexing groups. Non-contracting mode requires exactly one, and the other modes require
    /// none.
    rhs_group_dimensions: Vec<usize>,
}

impl RaggedDotDimensionNumbers {
    /// Creates a grouped dot dimension number specification.
    #[inline]
    pub fn new(
        dot_dimensions: DotDimensionNumbers,
        lhs_ragged_dimensions: Vec<usize>,
        rhs_group_dimensions: Vec<usize>,
    ) -> Self {
        Self { dot_dimensions, lhs_ragged_dimensions, rhs_group_dimensions }
    }

    /// Returns the specification of the basic non-contracting matrix form `[M, K] × [G, K, N] → [M, N]`.
    #[inline]
    pub fn matmul() -> Self {
        Self::new(DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()), vec![0], vec![0])
    }

    /// Returns the underlying generalized dot dimension numbers.
    #[inline]
    pub fn dot_dimensions(&self) -> &DotDimensionNumbers {
        &self.dot_dimensions
    }

    /// Returns the left-hand side ragged dimensions.
    #[inline]
    pub fn lhs_ragged_dimensions(&self) -> &[usize] {
        &self.lhs_ragged_dimensions
    }

    /// Returns the right-hand side group dimensions.
    #[inline]
    pub fn rhs_group_dimensions(&self) -> &[usize] {
        &self.rhs_group_dimensions
    }

    /// Classifies this specification according to the role of its single left-hand side ragged dimension for a
    /// left-hand side with rank `lhs_rank`. [`RaggedDotOperation`] validates the remaining dimension number
    /// constraints.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] when there is not exactly one ragged dimension or when its index is outside `lhs_rank`.
    pub fn mode(&self, lhs_rank: usize) -> Result<RaggedDotMode, TypeError> {
        if self.lhs_ragged_dimensions.len() != 1 {
            return Err(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` expects exactly one LHS ragged dimension, but got {}",
                self.lhs_ragged_dimensions.len(),
            )));
        }
        let axis = self.lhs_ragged_dimensions[0];
        if axis >= lhs_rank {
            return Err(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` LHS ragged dimension {axis} is out of bounds for rank {lhs_rank}",
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
        let ragged_axis = self.lhs_ragged_dimensions[0];
        let dimensions = self.dot_dimensions();
        validate_dot_axes(
            RAGGED_DOT_OPERATION_NAME,
            dimensions.lhs_batching_dimensions(),
            dimensions.lhs_contracting_dimensions(),
            lhs_rank,
            "LHS",
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

    /// Validates every ordinary, ragged, and group axis before dimension number lifting, returning the ragged mode.
    fn validate_for_ranks(&self, lhs_rank: usize, rhs_rank: usize) -> Result<RaggedDotMode, TypeError> {
        self.dot_dimensions.validate_for_ranks(RAGGED_DOT_OPERATION_NAME, lhs_rank, rhs_rank)?;
        let mode = self.mode(lhs_rank)?;
        let rhs_group_dimensions = self.rhs_group_dimensions();
        match mode {
            RaggedDotMode::NonContracting => {
                if rhs_group_dimensions.len() != 1 {
                    return Err(TypeError::invalid(format!(
                        "`{RAGGED_DOT_OPERATION_NAME}` requires exactly one RHS group dimension when the LHS ragged \
                         dimension is non-contracting",
                    )));
                }
                let group_axis = rhs_group_dimensions[0];
                if group_axis >= rhs_rank {
                    return Err(TypeError::invalid(format!(
                        "`{RAGGED_DOT_OPERATION_NAME}` RHS group dimension {group_axis} is out of bounds for rank \
                         {rhs_rank}",
                    )));
                }
                let dot_dimensions = self.dot_dimensions();
                if dot_dimensions.rhs_batching_dimensions().contains(&group_axis)
                    || dot_dimensions.rhs_contracting_dimensions().contains(&group_axis)
                {
                    return Err(TypeError::invalid(format!(
                        "`{RAGGED_DOT_OPERATION_NAME}` RHS group dimension must be distinct from batching and \
                         contracting dimensions",
                    )));
                }
            }
            RaggedDotMode::Contracting | RaggedDotMode::Batch if !rhs_group_dimensions.is_empty() => {
                return Err(TypeError::invalid(format!(
                    "`{RAGGED_DOT_OPERATION_NAME}` requires zero RHS group dimensions when the LHS ragged dimension is \
                     contracting or batching",
                )));
            }
            RaggedDotMode::Contracting | RaggedDotMode::Batch => {}
        }
        Ok(mode)
    }

    /// Returns the grouped dot dimension numbers that map the output cotangent of `t ↦ ragged_dot(t, rhs)` back toward
    /// the linear left-hand side `t`, together with the left-hand side axes in the order in which that adjoint grouped
    /// dot emits them. This specification must be a valid non-contracting specification for a left-hand side with rank
    /// `lhs_rank` and a right-hand side with rank `rhs_rank`.
    ///
    /// The adjoint is a non-contracting grouped dot of the cotangent with the right-hand side, whose ragged dimension
    /// is the ragged output dimension. It emits the batching axes, the left-hand side result axes, and then the
    /// left-hand side contracting axes sorted by their paired right-hand side axes. Callers must restore the original
    /// left-hand side axis order with the inverse permutation.
    fn lhs_cotangent_dimensions(&self, lhs_rank: usize, rhs_rank: usize) -> (Self, Vec<usize>) {
        let dot_dimensions = self.dot_dimensions();
        let lhs_result_dimensions = dot_dimensions.lhs_result_dimensions(lhs_rank);
        let rhs_result_dimensions = dot_dimensions
            .rhs_result_dimensions(rhs_rank)
            .into_iter()
            .filter(|axis| !self.rhs_group_dimensions.contains(axis))
            .collect::<Vec<_>>();
        let batching_count = dot_dimensions.lhs_batching_dimensions().len();
        let cotangent_rhs_result_start = batching_count + lhs_result_dimensions.len();
        let lhs_ragged_axis = self.lhs_ragged_dimensions[0];
        let ragged_output_axis =
            batching_count + lhs_result_dimensions.iter().position(|axis| *axis == lhs_ragged_axis).unwrap();
        let dimensions = Self::new(
            DotDimensionNumbers::new(
                (cotangent_rhs_result_start..(cotangent_rhs_result_start + rhs_result_dimensions.len())).collect(),
                rhs_result_dimensions,
                (0..batching_count).collect(),
                dot_dimensions.rhs_batching_dimensions().to_vec(),
            ),
            vec![ragged_output_axis],
            self.rhs_group_dimensions.clone(),
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

    /// Returns the grouped dot dimension numbers that map the output cotangent of `t ↦ ragged_dot(lhs, t)` back toward
    /// the linear right-hand side `t`, together with the right-hand side axes in the order in which that adjoint
    /// grouped dot emits them. This specification must be a valid non-contracting specification for a left-hand side
    /// with rank `lhs_rank` and a right-hand side with rank `rhs_rank`.
    ///
    /// The adjoint is a contracting grouped dot of the left-hand side with the cotangent over the ragged dimension, so
    /// it emits the group axis, the batching axes, the right-hand side contracting axes sorted by their paired
    /// left-hand side axes, and then the right-hand side result axes. Callers must restore the original right-hand side
    /// axis order with the inverse permutation.
    fn rhs_cotangent_dimensions(&self, lhs_rank: usize, rhs_rank: usize) -> (Self, Vec<usize>) {
        let dot_dimensions = self.dot_dimensions();
        let lhs_result_dimensions = dot_dimensions.lhs_result_dimensions(lhs_rank);
        let rhs_result_dimensions = dot_dimensions
            .rhs_result_dimensions(rhs_rank)
            .into_iter()
            .filter(|axis| !self.rhs_group_dimensions.contains(axis))
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
            self.lhs_ragged_dimensions.clone(),
            Vec::new(),
        );
        let mut contracting_pairs = dot_dimensions
            .rhs_contracting_dimensions()
            .iter()
            .copied()
            .zip(dot_dimensions.lhs_contracting_dimensions().iter().copied())
            .collect::<Vec<_>>();
        contracting_pairs.sort_by_key(|(_, lhs_axis)| *lhs_axis);
        let output_axes = self
            .rhs_group_dimensions
            .iter()
            .copied()
            .chain(dot_dimensions.rhs_batching_dimensions().iter().copied())
            .chain(contracting_pairs.into_iter().map(|(rhs_axis, _)| rhs_axis))
            .chain(rhs_result_dimensions)
            .collect();
        (dimensions, output_axes)
    }
}

impl Display for RaggedDotDimensionNumbers {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "(dot={}, lhs_ragged={:?}, rhs_group={:?})",
            self.dot_dimensions, self.lhs_ragged_dimensions, self.rhs_group_dimensions,
        )
    }
}

/// Canonical operation name for [`RaggedDotOperation`].
pub const RAGGED_DOT_OPERATION_NAME: &str = "ragged_dot_general";

/// Primitive representing a grouped generalized dot with explicit group sizes.
///
/// Exactly one left-hand side dimension is ragged. Its role selects one of three modes:
///
///   - A non-contracting ragged dimension is partitioned into consecutive groups. The corresponding right-hand side
///     group dimension selects one right-hand side slice per group, and the grouped products are written back along the
///     left-hand side result dimension. An empty group contributes nothing and any uncovered suffix of that dimension
///     is zero.
///   - A contracting ragged dimension partitions the paired contracting dimensions into consecutive groups. The
///     output gains a leading group dimension, and an empty group produces a zero slice.
///   - A batching ragged dimension has ordinary batched dot semantics. `group_sizes` participates in type inference
///     but its values do not affect the result.
///
/// `group_sizes` is either a rank-one `[group_count]` array shared by every prefix or an array whose trailing axis is
/// `group_count` and whose prefix matches the dimensions preceding the ragged position in the grouped dot iteration
/// space.
///
/// In non-contracting and contracting modes every size must be non-negative. The eager interpreter rejects negative
/// metadata in those modes. The decomposition lowering of XLA additionally clamps signed negatives to zero before
/// unsigned accumulation, so invalid runtime metadata cannot become a large interval. The instruction lowering passes
/// metadata unchanged to `chlo.ragged_dot` and therefore relies on the contract that sizes are non-negative.
///
/// The sizes define consecutive raw cumulative intervals. Each interval is intersected with the physical left-hand side
/// ragged extent, so a group that extends past that extent is clipped, and every later group is empty once its raw
/// start reaches or exceeds that extent.
///
/// Grouped expansion modes require an element type that can represent zero. In particular, they reject `f8e8m0fnu`.
/// Refer to [`RaggedDotDimensionNumbers`] for the dimension number contract.
///
/// The operation is linear in either data input separately, while `group_sizes` is nondifferentiable metadata. Forward
/// mode differentiation applies the product rule with the same group metadata. Transposition is defined only in
/// non-contracting mode and applies another grouped dot followed by the inverse permutation of its output axes.
///
/// Batching accepts any mix of replicated and mapped inputs with mapped axes at any position. It materializes a
/// replicated data input across the mapped extent and moves every mapped axis to the front, so the mapped axes become
/// a new leading batching pair, and `group_sizes` remains the sole source of the group extents of every item.
/// Replicated rank-one `group_sizes` stay shared by every item, while mapped rank-one `group_sizes` are broadcast
/// across the remaining `group_sizes` prefix of the lifted operation. Unlike JAX, which requires every input to be
/// mapped over axis zero, this also covers replicated inputs and non-leading mapped axes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RaggedDotOperation {
    /// Grouped dot dimension number specification.
    dimensions: RaggedDotDimensionNumbers,
}

impl RaggedDotOperation {
    /// Creates a grouped generalized dot.
    #[inline]
    pub fn new(dimensions: RaggedDotDimensionNumbers) -> Self {
        Self { dimensions }
    }

    /// Returns the grouped dot dimension number specification.
    #[inline]
    pub fn dimensions(&self) -> &RaggedDotDimensionNumbers {
        &self.dimensions
    }
}

impl Display for RaggedDotOperation {
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

        let mode = dimensions.validate_for_ranks(lhs.rank(), rhs.rank())?;
        let mut output = dot_abstract(RAGGED_DOT_OPERATION_NAME, lhs, rhs, dimensions.dot_dimensions(), None, None)?;
        let prefix_axes = dimensions.group_sizes_prefix_dimensions(lhs.rank())?;
        if group_sizes.rank() != 1 {
            let expected_prefix = Shape::new(prefix_axes.iter().map(|axis| lhs.dimension(*axis)).collect());
            let actual_prefix = Shape::new(group_sizes.shape().dimensions()[..group_sizes.rank() - 1].to_vec());
            if actual_prefix != expected_prefix {
                return Err(TypeError::invalid(format!(
                    "`{RAGGED_DOT_OPERATION_NAME}` group sizes prefix must be `{expected_prefix}`, but got \
                     `{actual_prefix}`",
                )));
            }
        }
        let group_count = group_sizes.dimension(group_sizes.rank() - 1);
        let rhs_group_dimensions = dimensions.rhs_group_dimensions();
        if mode == RaggedDotMode::NonContracting {
            let group_axis = rhs_group_dimensions[0];
            if rhs.dimension(group_axis) != group_count {
                return Err(TypeError::invalid(format!(
                    "`{RAGGED_DOT_OPERATION_NAME}` RHS group dimension has extent {} but group sizes describe \
                     {group_count}",
                    rhs.dimension(group_axis),
                )));
            }
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
                let group_position = rhs_result.iter().position(|axis| *axis == rhs_group_dimensions[0]).unwrap();
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

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("dimensions", &self.dimensions))
    }
}

impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for RaggedDotOperation
where
    C::Value: RaggedDot,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 3, ProgramError);
        Ok(vec![inputs[0].ragged_dot_general(&inputs[1], &inputs[2], &self.dimensions)?])
    }
}

// Partial evaluation uses the default fold-or-residualize behavior.
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for RaggedDotOperation where
    C::Operation: From<RaggedDotOperation>
{
}

impl<C: Context<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>>
    for RaggedDotOperation
where
    RaggedDotOperation: InterpretableOperation<C>,
{
    // Alignment delegates mapped axis materialization to `ArrayExtentBatchingPolicy::match_axis` and
    // `ArrayExtentBatchingPolicy::broadcast_input`, so a dynamic mapped extent remains a first-class value. The mapped
    // axes become a new leading batching pair, and the explicit `group_sizes` keep specifying the groups of every item.
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 3, ProgramError);
        let mode = self
            .dimensions
            .validate_for_ranks(inputs[0].unbatched_type().rank(), inputs[1].unbatched_type().rank())?;
        if inputs.iter().all(|input| input.batch_axis_position().is_none()) {
            return Ok(self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()])?);
        }
        check_common_mapped_dimension(RAGGED_DOT_OPERATION_NAME, inputs)?;

        // Both data inputs need the mapped axis at position zero before the logical dimension numbers can be lifted,
        // so a replicated data input is materialized across the mapped extent and a mapped one moves its axis there.
        let lhs = P::match_axis(context, &inputs[0], Axis::from(0))?;
        let rhs = P::match_axis(context, &inputs[1], Axis::from(0))?;
        let (dot_dimensions, _) = self.dimensions.dot_dimensions().batched(Some(0), Some(0)).unwrap();
        let dimensions = RaggedDotDimensionNumbers::new(
            dot_dimensions,
            self.dimensions.lhs_ragged_dimensions().iter().map(|axis| axis + 1).collect(),
            self.dimensions.rhs_group_dimensions().iter().map(|axis| axis + 1).collect(),
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

                    // The group sizes keep their memory space and manual variation, while each new prefix axis is
                    // replicated.
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
        Ok(RaggedDotOperation::new(dimensions).interpret_with_batch_axes(
            context,
            &[lhs, rhs, group_sizes],
            &[BatchAxis::from_position(output_axis)],
        )?)
    }
}

// A grouped dot is linear in either data input separately. Group sizes are integer metadata and therefore carry no
// tangent. The two product rule terms reuse the same grouped dot dimensions.
impl_differentiable_operation! {
    RaggedDotOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Operation: From<RaggedDotOperation>,
        C::Value: ConvertElementType + RaggedDot + Add,
    {
        |operation, context, _driver, inputs| {
            check_count!("input", inputs, 3, ProgramError);
            let lhs = &inputs[0];
            let rhs = &inputs[1];
            let group_sizes = inputs[2].primal();
            let primal = lhs.primal().ragged_dot_general(rhs.primal(), group_sizes, operation.dimensions())?;
            if lhs.tangent().is_zero() && rhs.tangent().is_zero() {
                return Ok(vec![DifferentiationDual::new_with_zero_tangent(primal)?]);
            }
            let group_sizes = context.primal_to_tangent(group_sizes.clone())?;
            let apply = |lhs: &C::Value, rhs: &C::Value| {
                lhs.ragged_dot_general(rhs, &group_sizes, operation.dimensions()).map_err(DifferentiationError::from)
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
            // Grouped dot transposition is defined only for the non-contracting mode. Each data input adjoint is
            // another grouped dot followed by the inverse permutation of the axis order that its adjoint dimension
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
                // Both data inputs linear is a bilinear product, which is not a linear map in both inputs jointly and
                // so never appears in a valid pushforward.
                (true, true) => {
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

            // An adjoint over widened tangents contracts both data inputs at the cotangent element type, and the result
            // is converted back to the linear input's cotangent element type.
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

/// Value-level grouped generalized dot capability.
///
/// [`RaggedDot`] is the receiver-style entry point for staging or executing [`RaggedDotOperation`], which contracts
/// two data inputs group by group according to explicit integer `group_sizes`.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and array IR values implement it for [`ArrayIrType`].
#[capability(projection(ArrayIrType => ArrayType))]
pub trait RaggedDot<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Computes a grouped generalized dot using explicit `group_sizes`. Refer to [`RaggedDotOperation`] for the three
    /// modes, the metadata shapes, the clipping of cumulative intervals, and the semantics of empty groups and
    /// uncovered positions.
    fn ragged_dot_general(
        &self,
        rhs: &Self,
        group_sizes: &Self,
        dimensions: &RaggedDotDimensionNumbers,
    ) -> Result<Self, ProgramError>;

    /// Computes the basic non-contracting form `[M, K] × [G, K, N] → [M, N]`. Refer to [`RaggedDotOperation`] for the
    /// behavior of empty groups and uncovered rows.
    #[inline]
    fn ragged_dot(&self, rhs: &Self, group_sizes: &Self) -> Result<Self, ProgramError> {
        self.ragged_dot_general(rhs, group_sizes, &RaggedDotDimensionNumbers::matmul())
    }
}

impl RaggedDot for Array {
    fn ragged_dot_general(
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

// Values that carry a context stage a grouped dot through that context. The `ValueDomainDispatch` marker keeps this
// implementation disjoint from eager values, which implement the capability directly.
impl<V: Value<Type = ArrayType, Dispatch = ValueDomainDispatch> + ManualVariationAlignment<ArrayType>>
    RaggedDot<ArrayType> for V
where
    V::Domain: Context<Type = ArrayType>,
    <V::Domain as Domain>::Operation: From<RaggedDotOperation>,
{
    fn ragged_dot_general(
        &self,
        rhs: &Self,
        group_sizes: &Self,
        dimensions: &RaggedDotDimensionNumbers,
    ) -> Result<Self, ProgramError> {
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
            return Err(TypeError::invalid(format!("`{operation_name}` {side} {role} dimensions must be distinct",)));
        }
    }
    if batching.iter().any(|axis| contracting.contains(axis)) {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` {side} batching and contracting dimensions must be disjoint",
        )));
    }
    Ok(())
}

/// Computes the abstract output type of one generalized dot product, naming `operation_name` in diagnostics.
///
/// The result shape is `[batching..., lhs_result..., rhs_result...]`, where the result dimensions are the input axes
/// that are neither batching nor contracting, in their original order. The output element type is the requested
/// compatible accumulation type when one is provided, or otherwise the common input element type.
///
/// Floating-point compatibility deliberately does not use the standard elementwise promotion lattice. Backend dot
/// instructions expose an independent preferred accumulator type, so even sub-byte and 8-bit inputs may accumulate at
/// `f32` or `f64`. Integer accumulation preserves signedness and may only widen the input type.
///
/// The output [`Sharding`] follows the `dot_general` sharding rule of JAX (`_dot_general_sharding_rule` in the
/// [JAX implementation](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py)). The [StableHLO `dot_general`
/// specification](https://openxla.org/stablehlo/spec#dot_general) describes the underlying contraction semantics.
/// Placement is inferred as follows.
///
///   - When `output_sharding` is provided, it is validated and returned directly, bypassing the ordinary input
///     consistency checks. Validation checks its rank and mesh, rejects auto axes, and accepts unreduced axes only when
///     the contracting dimensions are sharded identically over exactly those axes.
///   - Inputs must not be unreduced over explicit mesh axes. Reduced inputs are legal, and reduced and varying manual
///     axes are unioned across the inputs into the output sharding.
///   - When neither input carries a sharding, the output carries none. When exactly one does, the rule runs with the
///     absent side treated as fully replicated on the present input's mesh. Input meshes must match.
///   - Batch dimension entries are merged preferring the more informative entry (`Sharded` over `Replicated` over
///     `Unconstrained`). Two different `Sharded` entries are an error when an explicit mesh axis is involved. This
///     intentionally diverges from JAX, which reads batch dimension specs from the left-hand side only.
///   - Contracting dimensions sharded over explicit mesh axes on both inputs are an error. Identically sharded ones
///     make the output sharding ambiguous and require an explicit output sharding. A contracting dimension sharded on
///     only one input is allowed, and its sharding is dropped from the output.
///   - Result dimension entries are copied from the owning input, and auto mesh axes are stripped from the final
///     sharding.
pub(crate) fn dot_abstract(
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
    validate_dot_options(accumulation_type, output_sharding)?;
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
    dimensions.validate_for_ranks(operation_name, lhs.rank(), rhs.rank())?;
    for (lhs_axis, rhs_axis) in lhs_batching.iter().zip(rhs_batching) {
        if lhs.dimension(*lhs_axis) != rhs.dimension(*rhs_axis) {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` batching dimension sizes do not match (LHS axis {lhs_axis}, RHS axis \
                 {rhs_axis})",
            )));
        }
    }
    for (lhs_axis, rhs_axis) in lhs_contracting.iter().zip(rhs_contracting) {
        if lhs.dimension(*lhs_axis) != rhs.dimension(*rhs_axis) {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` contracting dimension sizes do not match (LHS axis {lhs_axis}, RHS axis \
                 {rhs_axis})",
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
                "`{operation_name}` output sharding rank ({}) does not match the output rank ({})",
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
        // leaves each device holding only a partial product sum over its shard of the contraction, whose true value is
        // the sum across devices. Marking the output unreduced over exactly those axes defers that sum to a later
        // operation instead of materializing it here, so the request is valid only when it names precisely the axes
        // that shard the (identically sharded) contracting dimensions. This is the `_dot_general_unreduced_rule` of
        // JAX.
        if !output_sharding.unreduced_axes().is_empty() {
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
            // Contracting shardings over only manual or auto axes are left to `shard_map` and the compiler. A
            // contracting dimension sharded on only one input is allowed, and its sharding is dropped from the output,
            // because the partitioner inserts the necessary communication.
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
                // Conflicts over an explicit axis are errors. Conflicts purely over manual or auto axes become
                // replicated and are left to `shard_map` and the compiler.
                (ShardingDimension::Sharded(_), ShardingDimension::Sharded(_))
                    if left.has_explicit_axis(mesh) || right.has_explicit_axis(mesh) =>
                {
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

/// Rejects element types for which the dot kernels of `operation_name` have no arithmetic implementation.
fn validate_dot_data_type(operation_name: &str, data_type: DataType) -> Result<(), TypeError> {
    if data_type.is_numeric() {
        Ok(())
    } else {
        Err(TypeError::invalid(format!("`{operation_name}` does not support input data type `{data_type}`")))
    }
}

/// Rejects dot option combinations that have no shared interpretation and transformation contract.
fn validate_dot_options(
    accumulation_type: Option<DataType>,
    output_sharding: Option<&Sharding>,
) -> Result<(), TypeError> {
    if accumulation_type.is_some() && output_sharding.is_some() {
        return Err(TypeError::invalid(format!(
            "`{DOT_OPERATION_NAME}` does not support combining an accumulation type with a requested output sharding \
             yet",
        )));
    }
    Ok(())
}

/// Checks that every mapped input of the operation named `operation_name` maps the same extent. Comparing the mapped
/// dimensions validates static extents exactly as [`ArrayBatch::common_batch_size`] does, and it additionally admits a
/// dynamic mapped extent, which two inputs share exactly when it is the same dimension variable.
fn check_common_mapped_dimension<V: Value<Type = ArrayType>>(
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

/// Applies the product rule shared by dense and grouped bilinear dot operations.
///
/// When one factor of a tangent term uses the widened tangent representation, both factors are converted to the output
/// tangent element type before applying the operation.
fn bilinear_array_jvp<C, V, Apply, P: DifferentiationPolicy<C>>(
    context: &DifferentiationContext<C, P>,
    primal: V,
    lhs: &DifferentiationDual<V>,
    rhs: &DifferentiationDual<V>,
    apply: Apply,
) -> Result<DifferentiationDual<V>, DifferentiationError>
where
    C: Context<Type = ArrayType, Value = V>,
    V: Value<Type = ArrayType> + ConvertElementType + Add,
    Apply: Fn(&V, &V) -> Result<V, DifferentiationError>,
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
            apply(lhs, rhs)
        } else {
            apply(&convert_to_tangent_type(lhs)?, &convert_to_tangent_type(rhs)?)
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

    /// Evaluates a grouped generalized dot over the exact physical extents. Each concrete group's raw cumulative
    /// interval is clipped to the physical ragged extent, the resulting pair of left-hand side and right-hand side
    /// slices is contracted by the ordinary generalized dot kernel, and the result is written into its output window.
    /// This keeps temporary storage proportional to one group rather than to a whole input times the group count.
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
        let ragged_axis = dimensions.lhs_ragged_dimensions()[0];
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

        // Non-contracting groups contract one RHS group slice, which drops the group axis and keeps a unit extent on
        // every batching axis that belongs to the group sizes prefix.
        let non_contracting_metadata = (mode == RaggedDotMode::NonContracting).then(|| {
            let rhs_group_axis = dimensions.rhs_group_dimensions()[0];
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

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use half::{bf16, f16};
    use indoc::{formatdoc, indoc};
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayBatch, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType, Dimension,
        DimensionBounds, DimensionVariable, Layout, LogicalMesh, MeshAxis, MeshAxisType, Shape, Sharding,
        ShardingDimension, StridedLayout, f8e4m3fn, f8e8m0fnu, i4,
    };
    use crate::batching::{
        BatchAxis, BatchableOperation, BatchedProgram, BatchingContext, ProgramBatchingOutputAxesPolicy, batch,
    };
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::{
        CotangentDestination, CotangentDestinationKind, CotangentSeed, TransposableOperation, TranspositionContext,
        differentiate_at,
    };
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
        ArrayType::new(DataType::F32, Shape::new(sizes.iter().map(|size| Dimension::Static(*size)).collect()))
    }

    /// Creates a validated floating-point type with explicit placement.
    fn sharded_array(mesh: &LogicalMesh, sizes: &[usize], dimensions: Vec<ShardingDimension>) -> ArrayType {
        plain_array(sizes).with_sharding(Sharding::new(mesh.clone(), dimensions).unwrap()).unwrap()
    }

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
    }

    #[test]
    fn test_dot_dimension_numbers_lhs_cotangent_dimensions() {
        assert_eq!(
            DotDimensionNumbers::matmul().lhs_cotangent_dimensions(2, 2),
            (DotDimensionNumbers::new(vec![1], vec![1], vec![], vec![]), vec![0, 1]),
        );
        // The adjoint emits batching, result, and then contracting axes sorted by their paired RHS axes, which differs
        // from the original LHS axis order.
        assert_eq!(
            DotDimensionNumbers::new(vec![2, 0], vec![0, 2], vec![1], vec![1]).lhs_cotangent_dimensions(4, 4),
            (DotDimensionNumbers::new(vec![2], vec![3], vec![0], vec![1]), vec![1, 3, 2, 0]),
        );
    }

    #[test]
    fn test_dot_dimension_numbers_rhs_cotangent_dimensions() {
        assert_eq!(
            DotDimensionNumbers::matmul().rhs_cotangent_dimensions(2, 2),
            (DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![]), vec![0, 1]),
        );
        // The adjoint emits batching, contracting axes sorted by their paired LHS axes, and then result axes, which
        // differs from the original RHS axis order.
        assert_eq!(
            DotDimensionNumbers::new(vec![2, 0], vec![0, 2], vec![1], vec![1]).rhs_cotangent_dimensions(4, 4),
            (DotDimensionNumbers::new(vec![3], vec![1], vec![1], vec![0]), vec![1, 2, 0, 3]),
        );
    }

    #[test]
    fn test_dot() {
        let mesh = test_mesh();
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated()]).unwrap();
        let operation = DotOperation::matmul().with_output_sharding(sharding.clone());

        // Operation identity, accessors, and formatting.
        assert_eq!(operation.name(), DOT_OPERATION_NAME);
        assert_eq!(operation.dimensions(), &DotDimensionNumbers::matmul());
        assert_eq!(operation.accumulation_type(), None);
        assert_eq!(operation.output_sharding(), Some(&sharding));
        assert_eq!(DotOperation::matmul().output_sharding(), None);
        assert_eq!(
            operation.to_string(),
            formatdoc! {"
                dot [
                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
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
    }

    #[test]
    fn test_dot_type_inference() {
        check_operation_type_inference!(
            operation = DotOperation::matmul(),
            cases = [
                { input_types = [plain_array(&[2, 3]), plain_array(&[3, 4])], output_types = [plain_array(&[2, 4])], },
                { input_types = [plain_array(&[2, 0]), plain_array(&[0, 4])], output_types = [plain_array(&[2, 4])], },
                { input_types = [plain_array(&[2, 3])], error = "expected 2 inputs but got 1", },
                {
                    input_types = [plain_array(&[2, 3]), plain_array(&[4, 2])],
                    error = format!(
                        "`{DOT_OPERATION_NAME}` contracting dimension sizes do not match (LHS axis 1, RHS axis 0)",
                    ),
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
            ],
        );

        check_operation_type_inference!(
            operation = DotOperation::new(DotDimensionNumbers::new(vec![1, 1], vec![0, 0], vec![], vec![])),
            cases = [{
                input_types = [plain_array(&[2, 3]), plain_array(&[3, 4])],
                error = format!("`{DOT_OPERATION_NAME}` LHS contracting dimensions must be distinct"),
            }],
        );

        check_operation_type_inference!(
            operation = DotOperation::new(DotDimensionNumbers::new(vec![1, 0], vec![0, 0], vec![], vec![])),
            cases = [{
                input_types = [plain_array(&[3, 3]), plain_array(&[3, 4])],
                error = format!("`{DOT_OPERATION_NAME}` RHS contracting dimensions must be distinct"),
            }],
        );

        check_operation_type_inference!(
            operation = DotOperation::new(DotDimensionNumbers::new(vec![1], vec![1], vec![0, 0], vec![0, 0])),
            cases = [{
                input_types = [plain_array(&[2, 3]), plain_array(&[2, 3])],
                error = format!("`{DOT_OPERATION_NAME}` LHS batching dimensions must be distinct"),
            }],
        );

        check_operation_type_inference!(
            operation = DotOperation::new(DotDimensionNumbers::new(vec![1], vec![1], vec![1], vec![0])),
            cases = [{
                input_types = [plain_array(&[3, 3]), plain_array(&[3, 3])],
                error = format!("`{DOT_OPERATION_NAME}` LHS batching and contracting dimensions must be disjoint"),
            }],
        );

        check_operation_type_inference!(
            operation = DotOperation::new(DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![1])),
            cases = [{
                input_types = [plain_array(&[3, 3]), plain_array(&[3, 3])],
                error = format!("`{DOT_OPERATION_NAME}` RHS batching and contracting dimensions must be disjoint"),
            }],
        );
    }

    #[test]
    fn test_dot_type_inference_with_dynamic_dimensions() {
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
            Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Dynamic(contracting), Dimension::Static(3)]),
        );
        let static_rhs = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(4), Dimension::Static(3)]),
        );
        let mismatched_batch_rhs = ArrayType::new(
            DataType::F64,
            Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::non_negative(Some(8)).unwrap())),
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::non_negative(Some(4)).unwrap())),
                Dimension::Static(3),
            ]),
        );
        // Batched matrix multiplication contracting axis 2 of the LHS with axis 1 of the RHS over batching axis 0.
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
                // Pairs of a static and a dynamic dimension and pairs of unequal dynamic dimensions keep erroring
                // under the strict size equality used for batching and contracting dimensions.
                {
                    input_types = [lhs.clone(), static_rhs],
                    error = format!(
                        "`{DOT_OPERATION_NAME}` contracting dimension sizes do not match (LHS axis 2, RHS axis 1)",
                    ),
                },
                {
                    input_types = [lhs, mismatched_batch_rhs],
                    error = format!(
                        "`{DOT_OPERATION_NAME}` batching dimension sizes do not match (LHS axis 0, RHS axis 0)",
                    ),
                },
            ],
        );
    }

    #[test]
    fn test_dot_type_inference_batched_sharding_propagation() {
        let mesh = test_mesh();
        let lhs = sharded_array(
            &mesh,
            &[2, 4, 8],
            vec![ShardingDimension::sharded(["b"]), ShardingDimension::sharded(["m"]), ShardingDimension::replicated()],
        );
        let rhs = sharded_array(
            &mesh,
            &[2, 8, 16],
            vec![ShardingDimension::replicated(), ShardingDimension::replicated(), ShardingDimension::sharded(["n"])],
        );
        // The batch dimension merges the more informative entry (LHS `b` over RHS replicated), and the result
        // dimensions are copied from their owning inputs.
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
            sharded_array(&mesh, &[4, 8], vec![ShardingDimension::replicated(), ShardingDimension::replicated()]);
        let replicated_rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::replicated(), ShardingDimension::replicated()]);
        let lhs =
            sharded_array(&mesh, &[4, 8], vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated()]);
        let rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::replicated(), ShardingDimension::sharded(["n"])]);
        check_operation_type_inference!(
            operation = DotOperation::matmul(),
            cases = [
                // Fully replicated inputs stay fully replicated.
                {
                    input_types = [replicated_lhs, replicated_rhs],
                    output_types = [sharded_array(
                        &mesh,
                        &[4, 16],
                        vec![ShardingDimension::replicated(), ShardingDimension::replicated()],
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
        let lhs =
            sharded_array(&mesh, &[4, 8], vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated()]);
        check_operation_type_inference!(
            operation = DotOperation::matmul(),
            cases = [
                // A missing input sharding is treated as fully replicated on the present input's mesh.
                {
                    input_types = [lhs, plain_array(&[8, 16])],
                    output_types = [sharded_array(
                        &mesh,
                        &[4, 16],
                        vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated()],
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
            vec![ShardingDimension::sharded(["b"]), ShardingDimension::replicated(), ShardingDimension::replicated()],
        );
        let rhs = sharded_array(
            &mesh,
            &[2, 8, 16],
            vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated(), ShardingDimension::replicated()],
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
    fn test_dot_type_inference_contracting_sharding_errors() {
        let mesh = test_mesh();
        let lhs =
            sharded_array(&mesh, &[4, 8], vec![ShardingDimension::replicated(), ShardingDimension::sharded(["k"])]);
        let rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::sharded(["k"]), ShardingDimension::replicated()]);
        let mismatched_rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated()]);
        let replicated_rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::replicated(), ShardingDimension::replicated()]);
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
                        vec![ShardingDimension::replicated(), ShardingDimension::replicated()],
                    )],
                },
            ],
        );
    }

    #[test]
    fn test_dot_type_inference_mesh_mismatch() {
        let mesh = test_mesh();
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("m", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let lhs =
            sharded_array(&mesh, &[4, 8], vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated()]);
        let rhs = sharded_array(
            &other_mesh,
            &[8, 16],
            vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated()],
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
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
                    .unwrap()
                    .with_unreduced_axes(["k"])
                    .unwrap(),
            )
            .unwrap();
        let reduced_lhs = plain_array(&[4, 8])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
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
                                    vec![ShardingDimension::replicated(), ShardingDimension::replicated()],
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
        let lhs =
            sharded_array(&mesh, &[4, 8], vec![ShardingDimension::sharded(["a"]), ShardingDimension::replicated()]);
        let rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::replicated(), ShardingDimension::sharded(["m"])]);
        check_operation_type_inference!(
            operation = DotOperation::matmul(),
            cases = [{
                input_types = [lhs, rhs],
                output_types = [sharded_array(
                    &mesh,
                    &[4, 16],
                    vec![ShardingDimension::replicated(), ShardingDimension::sharded(["m"])],
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
            vec![ShardingDimension::sharded(["b"]), ShardingDimension::replicated(), ShardingDimension::replicated()],
        );
        let rhs = sharded_array(
            &mesh,
            &[2, 8, 16],
            vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated(), ShardingDimension::replicated()],
        );

        // The requested output sharding bypasses the batch consistency checks (here, conflicting batch shardings).
        let requested = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::sharded(["b"]), ShardingDimension::replicated(), ShardingDimension::sharded(["n"])],
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
        let rank_mismatched = Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()]).unwrap();
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
            Sharding::new(auto_mesh, vec![ShardingDimension::sharded(["a"]), ShardingDimension::replicated()]).unwrap();
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
        let lhs =
            sharded_array(&mesh, &[4, 8], vec![ShardingDimension::replicated(), ShardingDimension::sharded(["k"])]);
        let rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::sharded(["k"]), ShardingDimension::replicated()]);
        let replicated_rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::replicated(), ShardingDimension::replicated()]);
        let unreduced =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
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
                            vec![ShardingDimension::replicated(), ShardingDimension::replicated()],
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
        let mismatched = Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
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
    fn test_dot_type_inference_accumulation_type() {
        // Type inference widens the output to the accumulation type for promotable input types and rejects
        // non-promotable ones and combining with a requested output sharding.
        let operation = DotOperation::matmul().with_accumulation_type(DataType::F32);
        assert_eq!(operation.accumulation_type(), Some(DataType::F32));
        let lhs = ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let rhs = lhs.clone();
        let bf16_input = ArrayType::new(DataType::BF16, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        check_operation_type_inference!(
            operation = operation,
            cases = [
                {
                    input_types = [lhs.clone(), rhs.clone()],
                    output_types = [output_type.clone()],
                },
                {
                    input_types = [bf16_input.clone(), bf16_input],
                    output_types = [output_type],
                },
            ],
        );
        let narrowing = DotOperation::matmul().with_accumulation_type(DataType::F16);
        let f32_input = plain_array(&[2, 2]);
        check_operation_type_inference!(
            operation = narrowing,
            cases = [{
                input_types = [f32_input.clone(), f32_input],
                error = format!("`{DOT_OPERATION_NAME}` input data type `f32` cannot accumulate at data type `f16`"),
            }],
        );
        let mesh = test_mesh();
        let sharded = DotOperation::matmul().with_accumulation_type(DataType::F32).with_output_sharding(
            Sharding::new(mesh, vec![ShardingDimension::Replicated, ShardingDimension::Replicated]).unwrap(),
        );

        check_operation_type_inference!(
            operation = sharded,
            cases = [{
                input_types = [lhs.clone(), rhs.clone()],
                error = format!(
                    "`{DOT_OPERATION_NAME}` does not support combining an accumulation type with a \
                    requested output sharding yet",
                ),
            }],
        );
    }

    #[test]
    fn test_dot_interpretation() {
        // Ordinary matrix multiplication uses the generalized contraction order.
        let lhs = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs = Array::matrix(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let dimensions = DotDimensionNumbers::new(vec![1], vec![0], vec![], vec![]);
        let product = lhs.dot(&rhs, &dimensions).unwrap();
        assert_eq!(product.r#type().into_owned(), ArrayType::new_static(DataType::F64, [2, 2]));
        assert_eq!(product.to_f64s(), vec![58.0, 64.0, 139.0, 154.0]);

        // Both inputs are decoded through their physical layouts rather than through dense logical payload copies.
        let lhs_type =
            ArrayType::new_static(DataType::U16, [2, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![-6, 2])));
        let rhs_type =
            ArrayType::new_static(DataType::U16, [3, 2]).with_layout(Layout::Strided(StridedLayout::new(vec![4, -2])));
        let lhs = Array::from_elements(lhs_type, &[1u16, 2, 3, 4, 5, 6]).unwrap();
        let rhs = Array::from_elements(rhs_type, &[7u16, 8, 9, 10, 11, 12]).unwrap();
        assert_eq!(lhs.dot(&rhs, &dimensions).unwrap().elements::<u16>(), Ok(vec![58, 64, 139, 154]));

        // Batched generalized contraction places batch axes before both inputs' non-contracting axes.
        let lhs = Array::from_elements(ArrayType::new_static(DataType::I32, [2, 2, 2]), &[1i32, 2, 3, 4, 5, 6, 7, 8])
            .unwrap();
        let rhs = Array::from_elements(ArrayType::new_static(DataType::I32, [2, 2, 1]), &[2i32, 3, 4, 5]).unwrap();
        let batched = DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]);
        let product = lhs.dot(&rhs, &batched).unwrap();
        assert_eq!(product.r#type().into_owned(), ArrayType::new_static(DataType::I32, [2, 2, 1]));
        assert_eq!(product.elements::<i32>(), Ok(vec![8, 18, 50, 68]));

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
        assert_eq!(lhs.dot(&rhs, &dimensions).unwrap().elements::<i4>(), Ok(vec![i4::new(-4).unwrap()]));
        let lhs = Array::matrix(1, 2, vec![ComplexNumber::new(1.0f32, 2.0), ComplexNumber::new(3.0, -1.0)]).unwrap();
        let rhs = Array::matrix(2, 1, vec![ComplexNumber::new(2.0f32, -1.0), ComplexNumber::new(0.5, 4.0)]).unwrap();
        assert_eq!(
            lhs.dot(&rhs, &dimensions).unwrap().elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(9.5, 14.5)]),
        );

        // An empty contracting dimension materializes one additive identity for every result coordinate.
        let lhs = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [2, 0]), &[]).unwrap();
        let rhs = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0, 3]), &[]).unwrap();
        assert_eq!(lhs.dot(&rhs, &dimensions).unwrap().elements::<f32>(), Ok(vec![0.0; 6]));
    }

    #[test]
    fn test_dot_interpretation_low_precision_accumulation() {
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
        let dimensions = DotDimensionNumbers::new(vec![1, 2], vec![0, 1], vec![], vec![]);
        assert_eq!(
            left.dot(&right, &dimensions),
            Ok(Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0, 0]), &[]).unwrap()),
        );
        let dimensions = DotDimensionNumbers::new(vec![0, 1, 2], vec![2, 0, 1], vec![], vec![]);
        assert_eq!(left.dot(&right, &dimensions), Ok(Array::scalar(0.0f32).unwrap()));
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
        let boolean = Array::matrix(1, 1, vec![true]).unwrap();
        assert_eq!(
            boolean.dot(&boolean, &DotDimensionNumbers::matmul()),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{DOT_OPERATION_NAME}` does not support input data type `bool`",
            )))),
        );
        let signed = Array::matrix(1, 1, vec![2i32]).unwrap();
        assert_eq!(
            signed.dot_with_accumulation_type(&signed, &DotDimensionNumbers::matmul(), DataType::U32),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{DOT_OPERATION_NAME}` input data type `i32` cannot accumulate at data type `u32`",
            )))),
        );
    }

    #[test]
    fn test_dot_interpretation_with_output_sharding() {
        // The requested sharding is validated by the operation's type inference and recorded on the computed result.
        let mesh = test_mesh();
        let dimensions = DotDimensionNumbers::new(vec![1], vec![0], vec![], vec![]);
        let lhs = Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs = Array::matrix(3, 2, vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated()])
                .unwrap();
        let product = lhs.dot_with_output_sharding(&rhs, &dimensions, &sharding).unwrap();
        assert_eq!(product.r#type().into_owned(), plain_array(&[2, 2]).with_sharding(sharding).unwrap());
        assert_eq!(product.elements::<f32>(), Ok(vec![58.0, 64.0, 139.0, 154.0]));

        // Invalid requests are rejected exactly as they are for staged programs.
        let rank_one = Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()]).unwrap();
        assert_eq!(
            lhs.dot_with_output_sharding(&rhs, &dimensions, &rank_one),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{DOT_OPERATION_NAME}` output sharding rank (1) does not match the output rank (2)",
            )))),
        );
        // Unreduced outputs are recorded as well, because a concrete array's payload is its fully reduced logical
        // value, which is exactly the total of the partial sums that an unreduced sharding describes.
        let lhs = Array::from_elements(
            sharded_array(&mesh, &[2, 2], vec![ShardingDimension::replicated(), ShardingDimension::sharded(["k"])]),
            &[1.0f32, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let rhs = Array::from_elements(
            sharded_array(&mesh, &[2, 2], vec![ShardingDimension::sharded(["k"]), ShardingDimension::replicated()]),
            &[5.0f32, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let unreduced = Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
            .unwrap()
            .with_unreduced_axes(["k"])
            .unwrap();
        let product = lhs.dot_with_output_sharding(&rhs, &dimensions, &unreduced).unwrap();
        assert_eq!(product.r#type().into_owned(), plain_array(&[2, 2]).with_sharding(unreduced).unwrap());
        assert_eq!(product.elements::<f32>(), Ok(vec![19.0, 22.0, 43.0, 50.0]));
    }

    #[test]
    fn test_dot_interpretation_with_accumulation_type() {
        let dimensions = DotDimensionNumbers::new(vec![1], vec![0], vec![], vec![]);
        // Preferred accumulation first promotes both inputs and then runs the same typed contraction at the wider
        // element data type.
        let lhs = Array::matrix(1, 2, vec![f16::from_f32(1.5), f16::from_f32(2.0)]).unwrap();
        let rhs = Array::matrix(2, 1, vec![f16::from_f32(2.0), f16::from_f32(3.0)]).unwrap();
        let product = lhs.dot_with_accumulation_type(&rhs, &dimensions, DataType::F32).unwrap();
        assert_eq!(product.r#type().data_type(), DataType::F32);
        assert_eq!(product.elements::<f32>(), Ok(vec![9.0]));
    }

    #[test]
    fn test_dot_interpretation_with_accumulation_type_low_precision() {
        let lhs = ArrayType::new_static(DataType::F8E4M3FN, [2, 2]);
        let rhs = lhs.clone();
        // The eager reference backend upcasts the inputs and accumulates at the accumulation type. Every value below is
        // exactly representable in `f8e4m3fn`, so the `f32` results are exact.
        let lhs_values = Array::from_elements::<f8e4m3fn>(
            lhs.clone(),
            &[0.5, 1.0, 1.5, 2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        let rhs_values = Array::from_elements::<f8e4m3fn>(
            rhs.clone(),
            &[1.0, 0.5, 0.5, 1.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        let product = lhs_values
            .dot_with_accumulation_type(&rhs_values, &DotDimensionNumbers::matmul(), DataType::F32)
            .unwrap();
        assert_eq!(
            product.r#type().as_ref(),
            &ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
        );
        assert_eq!(product.to_f64s(), vec![1.0, 1.25, 2.5, 2.75]);
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
        let lhs = ArrayType::new_static(DataType::F8E4M3FN, [2, 2]);
        let rhs = lhs.clone();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let lhs_input = builder.add_input(lhs.clone());
        let rhs_input = builder.add_input(rhs.clone());
        let output = builder
            .add_instruction(
                DotOperation::matmul().with_accumulation_type(DataType::F32),
                Vec::new(),
                vec![lhs_input, rhs_input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:f8e4m3fn[2, 2], %1:f8e4m3fn[2, 2] .
            let %2:f32[2, 2] = dot [
                dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                accumulation_type=f32,
            ] %0 %1
            in (%2)
        "}
            .trim_end(),
        );

        // Batching lifts the dimension numbers while carrying the accumulation type, so the products of each item still
        // accumulate at the widened type.
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
        let batched_lhs_type = ArrayType::new(
            DataType::F8E4M3FN,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(2)]),
        );
        let batched_lhs = Array::from_elements::<f8e4m3fn>(
            batched_lhs_type.clone(),
            &[0.5, 1.0, 1.5, 2.0, 0.5, 1.0, 1.5, 2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        let batched_rhs = Array::from_elements::<f8e4m3fn>(
            batched_lhs_type,
            &[1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        let outputs = lifted.interpret(vec![batched_lhs, batched_rhs]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].r#type().data_type(), DataType::F32);
        // Both batch items repeat the unbatched case, whose exact product is [[1, 1.25], [2.5, 2.75]].
        assert_eq!(outputs[0].to_f64s(), vec![1.0, 1.25, 2.5, 2.75, 1.0, 1.25, 2.5, 2.75]);
    }

    #[test]
    fn test_dot_batching_stages_the_lifted_output_sharding() {
        let mesh = test_mesh();
        let output_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["m"]), ShardingDimension::sharded(["n"])])
                .unwrap();
        let operation = DotOperation::matmul().with_output_sharding(output_sharding.clone());
        // Program batching applies lifted operations to tracer inputs. The staged batched dot must carry the lifted
        // output sharding instead of dropping it.
        let context = TracingContext::<ArrayType, ArrayOperation<ArrayType>>::new();
        let builder = context.builder().clone();
        let lhs_atom = builder.borrow_mut().add_input(plain_array(&[2, 4, 8]));
        let rhs_atom = builder.borrow_mut().add_input(plain_array(&[2, 8, 16]));
        let batching_context = BatchingContext::new(context.clone(), 2);
        let lhs = {
            let value = context.tracer(lhs_atom, None);
            ArrayBatch::new(value, Some(0))
        }
        .unwrap();
        let rhs = {
            let value = context.tracer(rhs_atom, None);
            ArrayBatch::new(value, Some(0))
        }
        .unwrap();
        let outputs = operation.batch(&batching_context, &EmptyRegionDriver, &[lhs, rhs]).unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        let output_atom = outputs[0].value().atom_id().unwrap();
        drop(outputs);
        drop(batching_context);
        drop(context);

        let builder = Rc::try_unwrap(builder).expect("batching should not hold on to the builder").into_inner();
        let program = builder
            .build::<Vec<ArrayType>, Vec<ArrayType>>(vec![output_atom], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:f32[2, 4, 8], %1:f32[2, 8, 16] .
            let %2:f32[2, 4, 16][sharding={mesh<['b'=2:explicit, 'm'=2:explicit, 'n'=2:explicit, \
                'k'=2:explicit]>, [{}, {'m'}, {'n'}]}] = dot [
                dimensions=(lhs_contracting=[2], rhs_contracting=[1], lhs_batching=[0], rhs_batching=[0]),
                output_sharding={mesh<['b'=2:explicit, 'm'=2:explicit, 'n'=2:explicit, 'k'=2:explicit]>, [{}, \
                    {'m'}, {'n'}]},
            ] %0 %1
            in (%2)
        "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dot_batching_preserves_materialized_batch_placement() {
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let lhs_sharding = Sharding::new(
                mesh.clone(),
                vec![
                    ShardingDimension::sharded(["x"]),
                    ShardingDimension::replicated(),
                    ShardingDimension::replicated(),
                ],
            )
            .unwrap()
            .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
            .unwrap();
            let lhs_type = ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(2)]),
            )
            .with_sharding(lhs_sharding)
            .unwrap();
            let rhs_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]))
                .with_sharding(Sharding::replicated(mesh, 2))
                .unwrap();
            let parent = TracingContext::<Array, ArrayOperation<Array>>::new();
            let builder = parent.builder().clone();
            let lhs_atom = builder.borrow_mut().add_input(lhs_type.clone());
            let rhs_atom = builder.borrow_mut().add_input(rhs_type);
            let lhs = ArrayBatch::new(parent.tracer(lhs_atom, None), BatchAxis::new(0)).unwrap();
            let rhs = ArrayBatch::replicated(parent.tracer(rhs_atom, None));
            let context = BatchingContext::new(parent.clone(), 2).with_axis_sharding(ShardingDimension::sharded(["x"]));
            let outputs = DotOperation::matmul().batch(&context, &EmptyRegionDriver, &[lhs, rhs]).unwrap();
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
            let output_atom = outputs[0].value().atom_id().unwrap();
            drop(outputs);
            drop(context);
            drop(parent);

            let builder = Rc::try_unwrap(builder).expect("batching should not retain the tracing builder").into_inner();
            let program = builder
                .build::<Vec<Array>, Vec<Array>>(vec![output_atom], vec![Placeholder, Placeholder], vec![Placeholder])
                .unwrap();
            let expected = match axis_type {
                MeshAxisType::Explicit => indoc! {"
                    lambda %0:f64[2, 2, 2][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}, {}]}], \
                    %1:f64[2, 1][sharding={mesh<['x'=2:explicit]>, [{}, {}]}] .
                    let %2:f64[2, 2, 1][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}, {}]}] = broadcast [
                        output_type=f64[2, 2, 1][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}, {}]}],
                        output_axes=[1, 2],
                    ] %1
                        %3:f64[2, 2, 1][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}, {}]}] = dot [
                            dimensions=(lhs_contracting=[2], rhs_contracting=[1], lhs_batching=[0], rhs_batching=[0]),
                        ] %0 %2
                    in (%3)
                "},
                MeshAxisType::Manual => indoc! {"
                    lambda %0:f64[2, 2, 2][sharding={mesh<['x'=2:manual]>, [{'x'}, {}, {}], varying_manual={'x'}}], \
                    %1:f64[2, 1][sharding={mesh<['x'=2:manual]>, [{}, {}]}] .
                    let %2:f64[2, 2, 1][sharding={mesh<['x'=2:manual]>, [{'x'}, {}, {}], \
                    varying_manual={'x'}}] = broadcast [
                        output_type=f64[2, 2, 1][sharding={mesh<['x'=2:manual]>, [{'x'}, {}, {}], \
                        varying_manual={'x'}}],
                        output_axes=[1, 2],
                    ] %1
                        %3:f64[2, 2, 1][sharding={mesh<['x'=2:manual]>, [{'x'}, {}, {}], \
                        varying_manual={'x'}}] = dot [
                            dimensions=(lhs_contracting=[2], rhs_contracting=[1], lhs_batching=[0], rhs_batching=[0]),
                        ] %0 %2
                    in (%3)
                "},
                MeshAxisType::Auto => unreachable!(),
            };
            assert_eq!(program.to_string(), expected.trim_end());
        }
    }

    #[test]
    fn test_dot_batching_aligns_mapped_manual_variation() -> Result<(), ProgramError> {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let left_type = ArrayType::new_static(DataType::F32, [2, 2, 2])
            .with_sharding(
                Sharding::new(
                    mesh.clone(),
                    vec![
                        ShardingDimension::sharded(["x"]),
                        ShardingDimension::replicated(),
                        ShardingDimension::replicated(),
                    ],
                )
                .unwrap()
                .with_varying_manual_axes(["x"])
                .unwrap(),
            )
            .unwrap();
        let right_type = ArrayType::new_static(DataType::F32, [2, 2, 1])
            .with_sharding(Sharding::replicated(mesh, 3))
            .unwrap();
        // Both inputs already have mapped axes, so axis materialization adds no variation. The Dot capability owns the
        // transition on the invariant input.
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
            (left_type, right_type),
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 2, 2][sharding={mesh<['x'=2:manual]>, [{'x'}, {}, {}], varying_manual={'x'}}], \
                %1:f32[2, 2, 1][sharding={mesh<['x'=2:manual]>, [{}, {}, {}]}] .
                let %2:f32[2, 2, 1][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], varying_manual={'x'}}] = \
                parallel_vary [axis_name=\"x\"] %1
                    %3:f32[2, 2, 1][sharding={mesh<['x'=2:manual]>, [{'x'}, {}, {}], varying_manual={'x'}}] = dot [
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
        // axis, while the physical operation preserves the leading mapped axis.
        let rows = Array::matrix(3, 4, (1..=12).map(|value| value as f64).collect()).unwrap();
        let output: Array = batch(
            |row| Ok(row.dot(&row, &DotDimensionNumbers::inner_product())?),
            rows,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(output.r#type().into_owned(), ArrayType::new_static(DataType::F64, [3]));
        // The batch items contract to [1,2,3,4]·[1,2,3,4] = 30, [5,6,7,8]·[5,6,7,8] = 174, and 446.
        assert_eq!(output.elements::<f64>(), Ok(vec![30.0, 174.0, 446.0]));
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
                "`{DOT_OPERATION_NAME}` LHS dimension index out of bounds",
            )))),
        );
    }

    #[test]
    fn test_dot_batching_validates_mapped_extents() {
        let context = TracingContext::<ArrayType, ArrayOperation<ArrayType>>::new();
        let batching_context = BatchingContext::new(context.clone(), 2);
        let input = |r#type: ArrayType| {
            let atom = context.builder().borrow_mut().add_input(r#type.clone());
            context.tracer(atom, Some(r#type))
        };
        let operation = DotOperation::new(DotDimensionNumbers::inner_product());
        let dynamic_rows = |variable: &DimensionVariable| {
            input(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(variable.clone()), Dimension::Static(3)]),
            ))
        };

        // Two mapped inputs whose mapped axes carry different statically known extents cannot describe one batch.
        assert_eq!(
            operation
                .batch(
                    &batching_context,
                    &EmptyRegionDriver,
                    &[
                        ArrayBatch::new(input(plain_array(&[2, 3])), BatchAxis::new(0)).unwrap(),
                        ArrayBatch::new(input(plain_array(&[3, 3])), BatchAxis::new(0)).unwrap(),
                    ],
                )
                .unwrap_err(),
            BatchingError::MismatchedBatchSizes { expected: 2, actual: 3 },
        );
        // Two mapped inputs sharing one dynamic mapped extent describe the same batch, so the lifted contraction is
        // staged with that dimension on its batching dimension.
        let variable = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5)).unwrap());
        let outputs = operation
            .batch(
                &batching_context,
                &EmptyRegionDriver,
                &[
                    ArrayBatch::new(dynamic_rows(&variable), BatchAxis::new(0)).unwrap(),
                    ArrayBatch::new(dynamic_rows(&variable), BatchAxis::new(0)).unwrap(),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(
            outputs[0].r#type().into_owned(),
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable.clone())])),
        );

        // Dynamic extents are compared by variable identity, so two independently declared variables are two
        // independent extents even when their bounds agree.
        let other = DimensionVariable::new("other", DimensionBounds::new(1, Some(5)).unwrap());
        assert_eq!(
            operation
                .batch(
                    &batching_context,
                    &EmptyRegionDriver,
                    &[
                        ArrayBatch::new(dynamic_rows(&variable), BatchAxis::new(0)).unwrap(),
                        ArrayBatch::new(dynamic_rows(&other), BatchAxis::new(0)).unwrap(),
                    ],
                )
                .unwrap_err(),
            BatchingError::MisalignedBatchAxes {
                message: format!("`{DOT_OPERATION_NAME}` inputs map different batch extents `batch` and `other`"),
            },
        );
    }

    #[test]
    fn test_dot_batching_under_a_dynamic_mapped_extent() -> Result<(), ProgramError> {
        // A dense contraction is batched under a dynamic mapped extent as well. The mapped input keeps the dynamic
        // batch dimension it already carries, and the replicated right-hand vector gains one through the policy's
        // dynamic broadcast, whose inserted axis is grounded by the transform's first-class extent value (read off the
        // mapped input's own axis). The lifted dot then contracts the trailing axis under a dynamic batching dimension.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let batch_variable = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5))?);
        let rows = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch_variable), Dimension::Static(3)]))
                .into(),
        );
        let vector = trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let output = batch(
            |(row, vector)| {
                let row = ValueProjection::<ArrayType>::into_projected(row)?;
                let vector = ValueProjection::<ArrayType>::into_projected(vector)?;
                let mut outputs = row.domain().bind(
                    DotOperation::new(DotDimensionNumbers::inner_product()),
                    Vec::new(),
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
                ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![10.0f32, 100.0, 1000.0]).unwrap()),
            ])?,
            ArrayIrValue::Array(Array::vector(vec![3210.0f32, 6540.0]).unwrap()),
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
        assert_eq!(left.value().to_f64s(), vec![7.0, 11.0, 13.0]);
        assert_eq!(right.value().to_f64s(), vec![2.0, 3.0, 5.0]);

        // Forward mode batches the basis tangents of the input coordinates through the dot pushforward.
        let jacobian = differentiate_at(inputs)
            .jacobian_forward(|(left, right)| Ok(left.dot(&right, &DotDimensionNumbers::inner_product())?))
            .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        let [left, right] = blocks.as_slice() else { unreachable!() };
        assert_eq!(left.value().to_f64s(), vec![7.0, 11.0, 13.0]);
        assert_eq!(right.value().to_f64s(), vec![2.0, 3.0, 5.0]);
    }

    #[test]
    fn test_dot_differentiation_accumulation_type() {
        let lhs = ArrayType::new_static(DataType::F8E4M3FN, [2, 2]);
        let rhs = lhs.clone();
        let lhs_values = Array::from_elements::<f8e4m3fn>(
            lhs.clone(),
            &[0.5, 1.0, 1.5, 2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        let rhs_values = Array::from_elements::<f8e4m3fn>(
            rhs.clone(),
            &[1.0, 0.5, 0.5, 1.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        // Forward mode differentiation stages tangent dots with the accumulation type over tangents of the input type,
        // so the output tangent lives at the accumulation type exactly like the primal output. Every value below is
        // exactly representable in `f8e4m3fn` and every product sum is exact in `f32`.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let lhs_input = builder.add_input(lhs.clone());
        let rhs_input = builder.add_input(rhs.clone());
        let output = builder
            .add_instruction(
                DotOperation::matmul().with_accumulation_type(DataType::F32),
                Vec::new(),
                vec![lhs_input, rhs_input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:f8e4m3fn[2, 2], %1:f8e4m3fn[2, 2] .
            let %2:f32[2, 2] = dot [
                dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                accumulation_type=f32,
            ] %0 %1
            in (%2)
        "}
            .trim_end(),
        );
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
        let jvp_outputs = jvp
            .interpret(vec![
                lhs_values.clone(),
                rhs_values.clone(),
                Array::from_elements::<f8e4m3fn>(
                    lhs.clone(),
                    &[1.0, 1.0, 1.0, 1.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
                )
                .unwrap(),
                Array::from_elements::<f8e4m3fn>(
                    rhs.clone(),
                    &[0.5, 0.5, 0.5, 0.5].map(|value| f8e4m3fn::from_f64(value).unwrap()),
                )
                .unwrap(),
            ])
            .unwrap();
        assert_eq!(jvp_outputs[0].to_f64s(), vec![1.0, 1.25, 2.5, 2.75]);
        assert_eq!(jvp_outputs[1].r#type().data_type(), DataType::F32);
        // Tangent = d_lhs · rhs + lhs · d_rhs = [[1.5, 1.5], [1.5, 1.5]] + [[0.75, 0.75], [1.75, 1.75]].
        assert_eq!(jvp_outputs[1].to_f64s(), vec![2.25, 2.25, 3.25, 3.25]);
    }

    #[test]
    fn test_dot_transposition() {
        let matmul = DotDimensionNumbers::matmul();
        let left = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let right = Array::matrix(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let cotangent = Array::matrix(2, 2, vec![1.0, -2.0, 0.5, 3.0]).unwrap();
        check_operation_transposition!(
            @exact,
            operation = DotOperation::new(matmul),
            cases = [
                {
                    inputs = [
                        (@known, left),
                        (@linear(type = right.r#type().into_owned())),
                    ],
                    output_cotangents = [cotangent.clone()],
                    input_cotangents = [Array::matrix(3, 2, vec![3.0, 10.0, 4.5, 11.0, 6.0, 12.0]).unwrap()],
                },
                {
                    inputs = [
                        (@linear(type = ArrayType::new(
                            DataType::F64,
                            Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]),
                        ))),
                        (@known, right),
                    ],
                    output_cotangents = [cotangent],
                    input_cotangents = [Array::matrix(2, 3, vec![-9.0, -11.0, -13.0, 27.5, 34.5, 41.5]).unwrap()],
                },
            ],
        );
    }

    #[test]
    fn test_dot_transposition_accumulation_type() {
        let lhs = ArrayType::new_static(DataType::F8E4M3FN, [2, 2]);
        let rhs = lhs.clone();
        let lhs_values = Array::from_elements::<f8e4m3fn>(
            lhs.clone(),
            &[0.5, 1.0, 1.5, 2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
        )
        .unwrap();
        // The transpose rule contracts the adjoint at the accumulation type and converts the result back to the linear
        // input's `f8e4m3fn` cotangent representation. With an identity output cotangent, the adjoint of the linear RHS
        // is exactly `lhsᵀ`.
        check_operation_transposition!(
            @exact,
            operation = DotOperation::matmul().with_accumulation_type(DataType::F32),
            cases = [{
                inputs = [
                    (@known, lhs_values),
                    (@linear(type = rhs.clone())),
                ],
                output_cotangents = [Array::from_elements::<f32>(
                    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
                    &[1.0, 0.0, 0.0, 1.0],
                ).unwrap()],
                input_cotangents = [Array::from_elements::<f8e4m3fn>(
                    rhs,
                    &[0.5, 1.5, 1.0, 2.0].map(|value| f8e4m3fn::from_f64(value).unwrap()),
                ).unwrap()],
            }],
        );
    }

    #[test]
    fn test_dot_transposition_prunes_unrequested_pullback_gradient() {
        let left = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let right = Array::matrix(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let (_, pullback) = differentiate_at((left, right))
            .vjp(|(left, right)| Ok(left.dot(&right, &DotDimensionNumbers::matmul())?))
            .unwrap();
        let both = pullback.transposed_program(&[CotangentDestinationKind::Return; 2]).unwrap().into_program();
        let selected = pullback
            .transposed_program(&[CotangentDestinationKind::Return, CotangentDestinationKind::Ignore])
            .unwrap()
            .into_program();
        // Ignoring the right gradient removes its contraction from the pullback itself.
        assert_eq!(
            both.to_string(),
            indoc! {"
            lambda %0:f64[2, 2], %1:f64[3, 2], %2:f64[2, 3] .
            let %3:f64[3, 2] = dot [
                dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
            ] %2 %0
                %4:f64[2, 3] = dot [
                    dimensions=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[], rhs_batching=[]),
                ] %0 %1
            in (%4, %3)
            "}
            .trim_end(),
        );
        assert_eq!(
            selected.to_string(),
            indoc! {"
            lambda %0:f64[2, 2], %1:f64[3, 2], %2:f64[2, 3] .
            let %3:f64[2, 3] = dot [
                dimensions=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[], rhs_batching=[]),
            ] %0 %1
            in (%3)
            "}
            .trim_end(),
        );
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(Array::matrix(2, 2, vec![1.0, -2.0, 0.5, 3.0]).unwrap()),
                (CotangentDestination::Return, CotangentDestination::Ignore),
            ),
            Ok((
                Some(MaybeZero::Value(Array::matrix(2, 3, vec![-9.0, -11.0, -13.0, 27.5, 34.5, 41.5]).unwrap())),
                None,
            )),
        );
    }

    #[test]
    fn test_dot_transposition_omits_unrequested_cotangent() {
        let mut context = TranspositionContext::new(TracingContext::<Array, ArrayOperation<Array>>::new());
        let right = context.input(ArrayType::new_static(DataType::F64, [3, 2]));
        let seed = context.input(ArrayType::new_static(DataType::F64, [2, 2]));
        let inputs = [PartialValue::Unknown(ArrayType::new_static(DataType::F64, [2, 3])), PartialValue::Known(right)];
        let accumulators = context.cotangent_accumulators(&inputs, &[false, false]).unwrap();
        DotOperation::matmul()
            .transpose(&mut context, &EmptyRegionDriver, &inputs, &[MaybeZero::Value(seed.clone())], &accumulators)
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
        // Requesting the same unknown input constructs its adjoint matrix multiplication with the same knownness.
        let accumulators = context.cotangent_accumulators(&inputs, &[true, false]).unwrap();
        DotOperation::matmul()
            .transpose(&mut context, &EmptyRegionDriver, &inputs, &[MaybeZero::Value(seed)], &accumulators)
            .unwrap();
        let contributions = context.take_cotangents(&accumulators).unwrap();
        let atom = contributions[0].as_value().unwrap().atom_id().unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![atom], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:f64[3, 2], %1:f64[2, 2] .
            let %2:f64[2, 3] = dot [
                dimensions=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[], rhs_batching=[]),
            ] %1 %0
            in (%2)
        "}
            .trim_end(),
        );
        assert!(!contributions[0].is_zero());
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
        assert!(context.builder().borrow().instructions().is_empty());
        assert!(context.take_cotangents(&accumulators).unwrap().iter().all(MaybeZero::is_zero));
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
    fn test_dot_transposition_multiple_contracting_axes_and_sharding() {
        let mesh = test_mesh();
        let left_type = sharded_array(
            &mesh,
            &[2, 2, 2, 1],
            vec![
                ShardingDimension::sharded(["m"]),
                ShardingDimension::sharded(["b"]),
                ShardingDimension::Replicated,
                ShardingDimension::Replicated,
            ],
        );
        let seed_type = sharded_array(
            &mesh,
            &[2, 1, 1],
            vec![ShardingDimension::sharded(["b"]), ShardingDimension::Replicated, ShardingDimension::Replicated],
        );

        check_operation_transposition!(
            @exact,
            operation = DotOperation::new(DotDimensionNumbers::new(vec![2, 0], vec![0, 2], vec![1], vec![1])),
            cases = [{
                inputs = [
                    (@linear(type = left_type.clone())),
                    (@known, Array::from_elements::<f32>(
                        plain_array(&[2, 2, 2, 1]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                    ).unwrap()),
                ],
                output_cotangents = [Array::from_elements::<f32>(seed_type, &[3.0, 5.0]).unwrap()],
                input_cotangents = [Array::from_elements::<f32>(
                    left_type, &[3.0, 15.0, 15.0, 35.0, 6.0, 18.0, 20.0, 40.0],
                ).unwrap()],
            }],
        );
    }

    #[test]
    fn test_dot_array_ir() {
        type ArrayIrContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

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
                in (%2)"
            },
        );
        let empty = ArrayIrValue::Array(Array::vector(Vec::<f32>::new()).unwrap());
        assert_eq!(
            empty.dot(&empty, &DotDimensionNumbers::inner_product()),
            Ok(ArrayIrValue::Array(Array::scalar(0.0f32).unwrap())),
        );
        let singleton = ArrayIrValue::Array(Array::vector(vec![3.0f32]).unwrap());
        assert_eq!(
            singleton.dot(&singleton, &DotDimensionNumbers::inner_product()),
            Ok(ArrayIrValue::Array(Array::scalar(9.0f32).unwrap())),
        );
        // Concrete array IR values contract their array members, and differentiation reaches the homogeneous rule
        // through the array IR member projection.
        let left = ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0]).unwrap());
        let right = ArrayIrValue::Array(Array::vector(vec![3.0f32, 4.0]).unwrap());
        assert_eq!(
            left.dot(&right, &DotDimensionNumbers::inner_product()),
            Ok(ArrayIrValue::Array(Array::scalar(11.0f32).unwrap())),
        );
        assert_eq!(
            differentiate_at(left.clone())
                .jvp(right.clone(), |input| input.dot(&input, &DotDimensionNumbers::inner_product())),
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
        let dimensions = RaggedDotDimensionNumbers::new(DotDimensionNumbers::inner_product(), vec![0], vec![]);
        assert_eq!(dimensions.dot_dimensions(), &DotDimensionNumbers::inner_product());
        assert_eq!(dimensions.lhs_ragged_dimensions(), &[0]);
        assert!(dimensions.rhs_group_dimensions().is_empty());
        let dimensions = RaggedDotDimensionNumbers::matmul();
        assert_eq!(
            dimensions.to_string(),
            "(dot=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[], rhs_batching=[]), lhs_ragged=[0], \
             rhs_group=[0])",
        );
        assert_eq!(
            format!("{dimensions:?}"),
            "RaggedDotDimensionNumbers { dot_dimensions: DotDimensionNumbers { lhs_contracting_dimensions: [1], \
             rhs_contracting_dimensions: [1], lhs_batching_dimensions: [], rhs_batching_dimensions: [] }, \
             lhs_ragged_dimensions: [0], rhs_group_dimensions: [0] }",
        );
    }

    #[test]
    fn test_ragged_dot_dimension_numbers_matmul() {
        assert_eq!(
            RaggedDotDimensionNumbers::matmul(),
            RaggedDotDimensionNumbers::new(
                DotDimensionNumbers::new(vec![1], vec![1], vec![], vec![]),
                vec![0],
                vec![0],
            ),
        );
    }

    #[test]
    fn test_ragged_dot_dimension_numbers_mode() {
        assert_eq!(RaggedDotDimensionNumbers::matmul().mode(2), Ok(RaggedDotMode::NonContracting));
        assert_eq!(
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::inner_product(), vec![0], vec![]).mode(1),
            Ok(RaggedDotMode::Contracting),
        );
        assert_eq!(
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![], vec![], vec![0], vec![0]), vec![0], vec![])
                .mode(1),
            Ok(RaggedDotMode::Batch),
        );
        assert_eq!(
            RaggedDotDimensionNumbers::new(DotDimensionNumbers::default(), vec![], vec![]).mode(2),
            Err(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` expects exactly one LHS ragged dimension, but got 0",
            ))),
        );
        assert_eq!(
            RaggedDotDimensionNumbers::matmul().mode(0),
            Err(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` LHS ragged dimension 0 is out of bounds for rank 0",
            ))),
        );
    }

    #[test]
    fn test_ragged_dot_dimension_numbers_group_sizes_prefix_dimensions() {
        let noncontracting = RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![3], vec![0], vec![1], vec![1]),
            vec![2],
            vec![2],
        );
        assert_eq!(noncontracting.group_sizes_prefix_dimensions(4), Ok(vec![1, 0]));
        let contracting = RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![3, 0], vec![0, 2], vec![1], vec![1]),
            vec![0],
            vec![],
        );
        assert_eq!(contracting.group_sizes_prefix_dimensions(4), Ok(vec![1, 3]));
        let batching = RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![], vec![], vec![2, 0], vec![1, 0]),
            vec![0],
            vec![],
        );
        assert_eq!(batching.group_sizes_prefix_dimensions(3), Ok(vec![2]));
        // Invalid LHS batching or contracting indices are rejected before they index the prefix.
        let invalid = RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![1], vec![1], vec![3], vec![0]),
            vec![0],
            vec![0],
        );
        assert_eq!(
            invalid.group_sizes_prefix_dimensions(2),
            Err(TypeError::invalid(format!("`{RAGGED_DOT_OPERATION_NAME}` LHS dimension index out of bounds"))),
        );
    }

    #[test]
    fn test_ragged_dot_dimension_numbers_lhs_cotangent_dimensions() {
        // The adjoint is another non-contracting grouped dot whose ragged dimension is the ragged output dimension.
        assert_eq!(
            RaggedDotDimensionNumbers::matmul().lhs_cotangent_dimensions(2, 3),
            (
                RaggedDotDimensionNumbers::new(
                    DotDimensionNumbers::new(vec![1], vec![2], vec![], vec![]),
                    vec![0],
                    vec![0],
                ),
                vec![0, 1],
            ),
        );
    }

    #[test]
    fn test_ragged_dot_dimension_numbers_rhs_cotangent_dimensions() {
        // The adjoint is a contracting grouped dot, so it emits the group axis first.
        assert_eq!(
            RaggedDotDimensionNumbers::matmul().rhs_cotangent_dimensions(2, 3),
            (
                RaggedDotDimensionNumbers::new(
                    DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![]),
                    vec![0],
                    vec![],
                ),
                vec![0, 1, 2],
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
                ragged_dot_general [
                    dimensions=(dot=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[], rhs_batching=[]), \
                        lhs_ragged=[0], rhs_group=[0]),
                ]
            "}
            .trim_end(),
        );
        assert_eq!(
            format!("{operation:?}"),
            "RaggedDotOperation { dimensions: RaggedDotDimensionNumbers { dot_dimensions: DotDimensionNumbers { \
             lhs_contracting_dimensions: [1], rhs_contracting_dimensions: [1], lhs_batching_dimensions: [], \
             rhs_batching_dimensions: [] }, lhs_ragged_dimensions: [0], rhs_group_dimensions: [0] } }",
        );

        // Program rendering uses the canonical operation name and includes the dimension numbers.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(left, right, sizes): (Tracer<_>, Tracer<_>, Tracer<_>)| left.ragged_dot(&right, &sizes),
            (plain_array(&[3, 2]), plain_array(&[2, 2, 1]), ArrayType::new_static(DataType::I32, [2])),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3, 2], %1:f32[2, 2, 1], %2:i32[2] .
                let %3:f32[3, 1] = ragged_dot_general [
                    dimensions=(dot=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[], rhs_batching=[]), \
                        lhs_ragged=[0], rhs_group=[0]),
                ] %0 %1 %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_ragged_dot_type_inference() {
        let group_sizes = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(3)]));
        let dimensions = RaggedDotDimensionNumbers::matmul();
        check_operation_type_inference!(
            operation = RaggedDotOperation::new(dimensions),
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
                        "`{RAGGED_DOT_OPERATION_NAME}` RHS group dimension has extent 2 but group \
                        sizes describe 3",
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

        check_operation_type_inference!(
            operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
                DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()),
                vec![0],
                Vec::new(),
            )),
            cases = [{
                input_types = [plain_array(&[5, 2]), plain_array(&[3, 2, 4]), group_sizes.clone()],
                error = format!(
                    "`{RAGGED_DOT_OPERATION_NAME}` requires exactly one RHS group dimension when the LHS ragged \
                     dimension is non-contracting",
                ),
            }],
        );

        let contracting = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![1], vec![0], Vec::new(), Vec::new()),
            vec![1],
            Vec::new(),
        ));
        check_operation_type_inference!(
            operation = contracting,
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
            operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
                DotDimensionNumbers::new(vec![1], vec![0], Vec::new(), Vec::new()),
                vec![1],
                vec![1],
            )),
            cases = [{
                input_types = [plain_array(&[2, 5]), plain_array(&[5, 4]), group_sizes.clone()],
                error = format!(
                    "`{RAGGED_DOT_OPERATION_NAME}` requires zero RHS group dimensions when the LHS ragged dimension is \
                         contracting or batching",
                ),
            }],
        );

        let batch = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]),
            vec![0],
            Vec::new(),
        ));
        check_operation_type_inference!(
            operation = batch,
            cases = [
                {
                    input_types = [plain_array(&[5, 2]), plain_array(&[5, 2, 4]), group_sizes.clone()],
                    output_types = [plain_array(&[5, 4])],
                },
                {
                    input_types = [
                        ArrayType::new_static(DataType::F8E8M0FNU, [5, 2]),
                        ArrayType::new_static(DataType::F8E8M0FNU, [5, 2, 4]),
                        group_sizes.clone(),
                    ],
                    output_types = [ArrayType::new_static(DataType::F8E8M0FNU, [5, 4])],
                },
            ],
        );

        let prefixed = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![2], vec![2], vec![0], vec![1]),
            vec![1],
            vec![0],
        ));
        check_operation_type_inference!(
            operation = prefixed,
            cases = [
                {
                    input_types = [
                        plain_array(&[2, 5, 3]),
                        plain_array(&[4, 2, 3, 6]),
                        ArrayType::new(
                            DataType::I32,
                            Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]),
                        ),
                    ],
                    output_types = [plain_array(&[2, 5, 6])],
                },
                {
                    input_types = [
                        plain_array(&[2, 5, 3]),
                        plain_array(&[4, 2, 3, 6]),
                        ArrayType::new(
                            DataType::I32,
                            Shape::new(vec![Dimension::Static(3), Dimension::Static(4)]),
                        ),
                    ],
                    error = format!("`{RAGGED_DOT_OPERATION_NAME}` group sizes prefix must be `[2]`, but got `[3]`"),
                },
            ],
        );

        let prefixed_contracting = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]),
            vec![2],
            Vec::new(),
        ));
        check_operation_type_inference!(
            operation = prefixed_contracting,
            cases = [{
                input_types = [
                    plain_array(&[2, 3, 4]),
                    plain_array(&[2, 4, 5]),
                    ArrayType::new_static(DataType::I32, [2, 3]),
                ],
                output_types = [plain_array(&[3, 2, 3, 5])],
            }],
        );

        let prefixed_batch = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![2], vec![2], vec![0, 1], vec![0, 1]),
            vec![1],
            Vec::new(),
        ));
        check_operation_type_inference!(
            operation = prefixed_batch,
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
    fn test_ragged_dot_type_inference_rejects_invalid_prefix_axes() {
        let dimensions = RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![1], vec![1], vec![3], vec![0]),
            vec![0],
            vec![0],
        );
        check_operation_type_inference!(
            operation = RaggedDotOperation::new(dimensions),
            cases = [{
                input_types = [
                    plain_array(&[3, 2]),
                    plain_array(&[2, 2, 1]),
                    ArrayType::new_static(DataType::I32, [2, 2]),
                ],
                error = format!("`{RAGGED_DOT_OPERATION_NAME}` LHS dimension index out of bounds"),
            }],
        );
    }

    #[test]
    fn test_ragged_dot_interpretation() {
        let lhs = Array::matrix(5, 2, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).unwrap();
        let rhs = Array::from_elements::<f32>(plain_array(&[3, 2, 1]), &[10.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let output = lhs.ragged_dot(&rhs, &Array::vector(vec![2i32, 0, 2]).unwrap()).unwrap();
        assert_eq!(output.r#type().into_owned(), plain_array(&[5, 1]));
        assert_eq!(output.to_f64s(), vec![12.0, 34.0, 50.0, 68.0, 0.0]);

        // Sizes that cover more than the extent clip the intersecting group and leave every later raw cumulative
        // interval empty.
        let lhs = Array::matrix(4, 2, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let output = lhs.ragged_dot(&rhs, &Array::vector(vec![3i32, 3, 2]).unwrap()).unwrap();
        assert_eq!(output, Array::matrix(4, 1, vec![12.0f32, 34.0, 56.0, 38.0]).unwrap());
        // Sizes in intervals that begin after the physical extent are unobservable, even when their cumulative sum
        // would overflow the host index type.
        let lhs = Array::matrix(1, 1, vec![2.0f32]).unwrap();
        let rhs = Array::from_elements::<f32>(plain_array(&[2, 1, 1]), &[3.0, 100.0]).unwrap();
        let output = lhs.ragged_dot(&rhs, &Array::vector(vec![1u64, u64::MAX]).unwrap()).unwrap();
        assert_eq!(output, Array::matrix(1, 1, vec![6.0f32]).unwrap());
        let lhs = Array::from_elements::<f8e8m0fnu>(
            ArrayType::new_static(DataType::F8E8M0FNU, [2, 1]),
            &[1.0, 2.0].map(|value| f8e8m0fnu::from_f64(value).unwrap()),
        )
        .unwrap();
        let rhs = Array::from_elements::<f8e8m0fnu>(
            ArrayType::new_static(DataType::F8E8M0FNU, [1, 1, 1]),
            &[1.0].map(|value| f8e8m0fnu::from_f64(value).unwrap()),
        )
        .unwrap();
        assert!(matches!(
            lhs.ragged_dot(&rhs, &Array::vector(vec![1i32]).unwrap()),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == format!(
                    "`{RAGGED_DOT_OPERATION_NAME}` does not support element data type `f8e8m0fnu` in grouped \
                     expansion modes because it cannot represent zero",
                ),
        ));
    }

    #[test]
    fn test_ragged_dot_interpretation_rejects_negative_group_sizes() {
        let lhs = Array::matrix(2, 1, vec![1.0f32, 2.0]).unwrap();
        let rhs = Array::from_elements::<f32>(plain_array(&[2, 1, 1]), &[3.0, 4.0]).unwrap();
        assert!(matches!(
            lhs.ragged_dot(&rhs, &Array::vector(vec![1i32, -1]).unwrap()),
            Err(ProgramError::InvalidArgument { message })
                if message == "`group_sizes[1]` must be non-negative but got -1",
        ));
    }

    #[test]
    fn test_ragged_dot_interpretation_contracting_batch_and_group_prefixes() {
        let contracting_dimensions = RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![1], vec![0], Vec::new(), Vec::new()),
            vec![1],
            Vec::new(),
        );
        let lhs = Array::matrix(2, 4, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let rhs = Array::matrix(4, 1, vec![10.0f32, 20.0, 30.0, 40.0]).unwrap();
        let output = lhs
            .ragged_dot_general(&rhs, &Array::vector(vec![1i32, 0, 2]).unwrap(), &contracting_dimensions)
            .unwrap();
        assert_eq!(output.r#type().into_owned(), plain_array(&[3, 2, 1]));
        assert_eq!(output.to_f64s(), vec![10.0, 50.0, 0.0, 0.0, 130.0, 330.0]);
        // Contracting groups use the same clipped raw intervals, with empty trailing groups retaining zero output
        // slices.
        let output = lhs
            .ragged_dot_general(&rhs, &Array::vector(vec![3i32, 3, 2]).unwrap(), &contracting_dimensions)
            .unwrap();
        assert_eq!(
            output,
            Array::from_elements(plain_array(&[3, 2, 1]), &[140.0f32, 380.0, 160.0, 320.0, 0.0, 0.0]).unwrap(),
        );

        let prefixed_contracting_dimensions = RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![0, 1], vec![0, 1], Vec::new(), Vec::new()),
            vec![1],
            Vec::new(),
        );
        let lhs = Array::from_elements::<f32>(plain_array(&[2, 3, 1]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs = Array::from_elements::<f32>(plain_array(&[2, 3, 1]), &[10.0, 20.0, 30.0, 1.0, 2.0, 3.0]).unwrap();
        let group_sizes = Array::matrix(2, 2, vec![1i32, 2, 2, 1]).unwrap();
        let output = lhs.ragged_dot_general(&rhs, &group_sizes, &prefixed_contracting_dimensions).unwrap();
        assert_eq!(output.r#type().into_owned(), plain_array(&[2, 1, 1]));
        assert_eq!(output.to_f64s(), vec![24.0, 148.0]);
        let group_sizes = Array::matrix(2, 2, vec![2i32, 3, 4, 1]).unwrap();
        let output = lhs.ragged_dot_general(&rhs, &group_sizes, &prefixed_contracting_dimensions).unwrap();
        assert_eq!(output, Array::from_elements(plain_array(&[2, 1, 1]), &[82.0f32, 90.0]).unwrap());
        let batch_dimensions = RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]),
            vec![0],
            Vec::new(),
        );
        let lhs = Array::matrix(4, 1, vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let rhs = Array::from_elements::<f32>(plain_array(&[4, 1, 1]), &[10.0, 20.0, 30.0, 40.0]).unwrap();
        // Batch mode is the ordinary batched dot, so group size values do not participate in its runtime semantics.
        let output = lhs
            .ragged_dot_general(&rhs, &Array::vector(vec![5i32, -1, 99]).unwrap(), &batch_dimensions)
            .unwrap();
        assert_eq!(output.r#type().into_owned(), plain_array(&[4, 1]));
        assert_eq!(output.to_f64s(), vec![10.0, 40.0, 90.0, 160.0]);

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
        let output = lhs.ragged_dot_general(&rhs, &Array::vector(vec![-1i32]).unwrap(), &batch_dimensions).unwrap();
        assert_eq!(output.to_f64s(), vec![2.0, 8.0]);
        let prefixed_dimensions = RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![2], vec![2], vec![0], vec![1]),
            vec![1],
            vec![0],
        );
        let lhs = Array::from_elements::<f32>(plain_array(&[2, 3, 1]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs = Array::from_elements::<f32>(plain_array(&[2, 2, 1, 1]), &[10.0, 100.0, 20.0, 200.0]).unwrap();
        let group_sizes = Array::matrix(2, 2, vec![1i32, 1, 2, 0]).unwrap();
        let output = lhs.ragged_dot_general(&rhs, &group_sizes, &prefixed_dimensions).unwrap();
        assert_eq!(output.r#type().into_owned(), plain_array(&[2, 3, 1]));
        assert_eq!(output.to_f64s(), vec![10.0, 40.0, 0.0, 400.0, 500.0, 0.0]);

        let prefixed_batch_dimensions = RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![2], vec![2], vec![0, 1], vec![0, 1]),
            vec![1],
            Vec::new(),
        );
        let lhs =
            Array::from_elements::<f32>(plain_array(&[2, 4, 1]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let rhs =
            Array::from_elements::<f32>(plain_array(&[2, 4, 1, 1]), &[10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0])
                .unwrap();
        let group_sizes = Array::matrix(2, 3, vec![1i32, 0, 2, 0, 2, 2]).unwrap();
        let output = lhs.ragged_dot_general(&rhs, &group_sizes, &prefixed_batch_dimensions).unwrap();
        assert_eq!(output.r#type().into_owned(), plain_array(&[2, 4, 1]));
        assert_eq!(output.to_f64s(), vec![10.0, 40.0, 90.0, 160.0, 5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    fn test_ragged_dot_interpretation_empty_output_skips_large_prefix_products() {
        let extent = 1usize << (usize::BITS / 2);
        let left =
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0, extent, extent, 1]), &[]).unwrap();
        let right = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [1, 0, 0]), &[]).unwrap();
        let dimensions = RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![0], vec![1], vec![], vec![]),
            vec![3],
            vec![0],
        );
        assert_eq!(
            left.ragged_dot_general(&right, &Array::vector(vec![1i32]).unwrap(), &dimensions),
            Ok(Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [extent, extent, 1, 0]), &[]).unwrap()),
        );
        assert_eq!(
            left.ragged_dot_general(&right, &Array::vector(vec![-1i32]).unwrap(), &dimensions),
            Err(ProgramError::InvalidArgument {
                message: "`group_sizes[0]` must be non-negative but got -1".to_string()
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
        let operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::matmul());
        let lhs = ArrayBatch::new(
            Array::from_elements::<f32>(plain_array(&[2, 2, 2]), &[1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 4.0, 3.0]).unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let rhs = ArrayBatch::new(
            Array::from_elements::<f32>(plain_array(&[2, 2, 2, 1]), &[10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
                .unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let group_sizes =
            ArrayBatch::new(Array::matrix(2, 2, vec![1i32, 1, 1, 1]).unwrap(), BatchAxis::new(0)).unwrap();
        let context = BatchingContext::new(EagerContext::<Array>::new(), 2);
        let outputs = operation
            .batch(&context, &EmptyRegionDriver, &[lhs.clone(), rhs.clone(), group_sizes.clone()])
            .unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().to_f64s(), vec![12.0, 18.0, 13.0, 45.0]);

        let contracting_operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![1], vec![0], Vec::new(), Vec::new()),
            vec![1],
            Vec::new(),
        ));
        let contracting_lhs = ArrayBatch::new(
            Array::from_elements::<f32>(
                plain_array(&[2, 2, 4]),
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 0.0],
            )
            .unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let contracting_rhs = ArrayBatch::new(
            Array::from_elements::<f32>(plain_array(&[2, 4, 1]), &[10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0])
                .unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let contracting_groups =
            ArrayBatch::new(Array::matrix(2, 2, vec![1i32, 2, 2, 2]).unwrap(), BatchAxis::new(0)).unwrap();
        let outputs = contracting_operation
            .batch(&context, &EmptyRegionDriver, &[contracting_lhs, contracting_rhs, contracting_groups])
            .unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(1));
        assert_eq!(outputs[0].value().to_f64s(), vec![10.0, 50.0, 4.0, 5.0, 130.0, 330.0, 4.0, 3.0]);
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
        let sizes = ArrayBatch::new(Array::matrix(2, 1, vec![1i32, 1]).unwrap(), BatchAxis::new(0)).unwrap();
        let operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![usize::MAX], vec![1], vec![], vec![]),
            vec![0],
            vec![0],
        ));
        let inputs = [left, right, sizes];
        let context = BatchingContext::new(EagerContext::<Array>::new(), 2);
        assert_eq!(
            operation.batch(&context, &EmptyRegionDriver, &inputs),
            Err(BatchingError::Type(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` LHS dimension index out of bounds",
            )))),
        );
        let invalid_group = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![1], vec![1], vec![], vec![]),
            vec![0],
            vec![usize::MAX],
        ));
        assert_eq!(
            invalid_group.batch(&context, &EmptyRegionDriver, &inputs),
            Err(BatchingError::Type(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` RHS group dimension {} is out of bounds for rank 3",
                usize::MAX,
            )))),
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
                            Array::from_elements::<f32>(plain_array(&[2, 2, 1]), &[1.0, 10.0, 100.0, 1000.0])
                            .unwrap(),
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
                            Array::from_elements::<f32>(plain_array(&[2, 2, 1]), &[1.0, 10.0, 100.0, 1000.0])
                            .unwrap(),
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
                vec![1],
                vec![0],
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
    fn test_ragged_dot_batching_under_a_dynamic_mapped_extent() -> Result<(), ProgramError> {
        // The replicated RHS gains the dynamic mapped axis through the policy's dynamic broadcast, and the mapped
        // rank-one group sizes are broadcast across the batching prefix of the lifted operation. Both broadcasts are
        // grounded by the transform's first-class extent value.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
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
                        vec![1],
                        vec![0],
                    )),
                    Vec::new(),
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
                    %14:f32[batch, 2, 3, 1] = ragged_dot_general [
                        dimensions=(dot=(lhs_contracting=[3], rhs_contracting=[3], lhs_batching=[0, 1], rhs_batching=[0, 2]), lhs_ragged=[2], rhs_group=[1]),
                    ] %0 %10 %13
                in (%14)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Array(
                    Array::from_elements::<f32>(
                        plain_array(&[2, 2, 3, 1]),
                        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    )
                    .unwrap(),
                ),
                ArrayIrValue::Array(
                    Array::from_elements::<f32>(plain_array(&[2, 2, 1, 1]), &[10.0, 20.0, 100.0, 200.0]).unwrap(),
                ),
                ArrayIrValue::Array(Array::matrix(2, 2, vec![1i32, 2, 3, 0]).unwrap()),
            ])?,
            ArrayIrValue::Array(
                Array::from_elements::<f32>(
                    plain_array(&[2, 2, 3, 1]),
                    &[10.0, 200.0, 300.0, 80.0, 1000.0, 1200.0, 10.0, 20.0, 30.0, 80.0, 100.0, 120.0],
                )
                .unwrap(),
            ),
        );
        Ok(())
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
    fn test_ragged_dot_differentiation() {
        let left = Array::matrix(2, 1, vec![2.0, 3.0]).unwrap();
        let right = Array::from_elements(ArrayType::new_static(DataType::F64, [2, 1, 1]), &[5.0, 7.0]).unwrap();
        let sizes = Array::vector(vec![1i32, 1]).unwrap();
        assert_eq!(
            differentiate_at((left, right)).jvp(
                (
                    Array::matrix(2, 1, vec![1.0, 2.0]).unwrap(),
                    Array::from_elements(ArrayType::new_static(DataType::F64, [2, 1, 1]), &[3.0, 4.0]).unwrap(),
                ),
                |(left, right)| {
                    let sizes = left.domain().lift(sizes.clone())?;
                    left.ragged_dot(&right, &sizes)
                },
            ),
            Ok((Array::matrix(2, 1, vec![10.0, 21.0]).unwrap(), Array::matrix(2, 1, vec![11.0, 26.0]).unwrap())),
        );
        check_gradient!(
            |left| {
                let right = left
                    .domain()
                    .lift(Array::from_elements(ArrayType::new_static(DataType::F64, [2, 1, 1]), &[5.0, 7.0])?)?;
                let sizes = left.domain().lift(Array::vector(vec![1i32, 1])?)?;
                left.ragged_dot(&right, &sizes)?.reduce_sum(&[0, 1], None)
            },
            at = Array::matrix(2, 1, vec![2.0, 3.0]).unwrap(),
            step = 1e-5,
            tolerance = 1e-7,
        );
        check_gradient!(
            |right| {
                let left = right.domain().lift(Array::matrix(2, 1, vec![2.0, 3.0])?)?;
                let sizes = right.domain().lift(Array::vector(vec![1i32, 1])?)?;
                left.ragged_dot(&right, &sizes)?.reduce_sum(&[0, 1], None)
            },
            at = Array::from_elements(ArrayType::new_static(DataType::F64, [2, 1, 1]), &[5.0, 7.0]).unwrap(),
            step = 1e-5,
            tolerance = 1e-7,
        );

        // A staged JVP reuses the known group metadata in both product rule terms.
        let lhs = Array::matrix(3, 2, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs = Array::from_elements::<f32>(plain_array(&[2, 2, 1]), &[10.0, 1.0, 2.0, 3.0]).unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let lhs_input = builder.add_input(lhs.r#type().into_owned());
        let rhs_input = builder.add_input(rhs.r#type().into_owned());
        let groups = builder.add_constant(Array::vector(vec![1i32, 2]).unwrap());
        let output = builder
            .add_instruction(
                RaggedDotOperation::new(RaggedDotDimensionNumbers::matmul()),
                Vec::new(),
                vec![lhs_input, rhs_input, groups],
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
                    %5:f32[3, 1] = ragged_dot_general [
                        dimensions=(dot=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[], \
                            rhs_batching=[]), lhs_ragged=[0], rhs_group=[0]),
                    ] %0 %1 %4
                    %6:f32[3, 1] = ragged_dot_general [
                        dimensions=(dot=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[], \
                            rhs_batching=[]), lhs_ragged=[0], rhs_group=[0]),
                    ] %2 %1 %4
                    %7:f32[3, 1] = ragged_dot_general [
                        dimensions=(dot=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[], \
                            rhs_batching=[]), lhs_ragged=[0], rhs_group=[0]),
                    ] %0 %3 %4
                    %8:f32[3, 1] = add %6 %7
                in (%5, %8)
            "}
            .trim_end(),
        );
        assert_eq!(
            jvp.interpret(vec![
                lhs,
                rhs,
                Array::matrix(3, 2, vec![1.0f32; 6]).unwrap(),
                Array::from_elements::<f32>(plain_array(&[2, 2, 1]), &[1.0; 4]).unwrap(),
            ]),
            Ok(vec![
                Array::matrix(3, 1, vec![12.0f32, 18.0, 28.0]).unwrap(),
                Array::matrix(3, 1, vec![14.0f32, 12.0, 16.0]).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_ragged_dot_differentiation_widened_tangents() {
        let lhs_type = ArrayType::new_static(DataType::F8E8M0FNU, [3, 2]);
        let rhs_type = ArrayType::new_static(DataType::F8E8M0FNU, [3, 2, 1]);
        let dimensions = RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]),
            vec![0],
            Vec::new(),
        );
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let lhs = builder.add_input(lhs_type.clone());
        let rhs = builder.add_input(rhs_type.clone());
        let group_sizes = builder.add_constant(Array::vector(vec![1i32, 2]).unwrap());
        let output = builder
            .add_instruction(RaggedDotOperation::new(dimensions), Vec::new(), vec![lhs, rhs, group_sizes], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let jvp = program.clone().jvp().unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:f8e8m0fnu[3, 2], %1:f8e8m0fnu[3, 2, 1] .
            let %2:i32[2] = const [1, 2]
                %3:f8e8m0fnu[3, 1] = ragged_dot_general [
                    dimensions=(dot=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[0], \
                        rhs_batching=[0]), lhs_ragged=[0], rhs_group=[]),
                ] %0 %1 %2
            in (%3)
            "}
            .trim_end(),
        );
        assert_eq!(
            jvp.to_string(),
            indoc! {"
            lambda %0:f8e8m0fnu[3, 2], %1:f8e8m0fnu[3, 2, 1], %2:f32[3, 2], %3:f32[3, 2, 1] .
            let %4:i32[2] = const [1, 2]
                %5:f8e8m0fnu[3, 1] = ragged_dot_general [
                    dimensions=(dot=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[0], \
                        rhs_batching=[0]), lhs_ragged=[0], rhs_group=[]),
                ] %0 %1 %4
                %6:f32[3, 2, 1] = convert_element_type [data_type=f32] %1
                %7:f32[3, 1] = ragged_dot_general [
                    dimensions=(dot=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[0], \
                        rhs_batching=[0]), lhs_ragged=[0], rhs_group=[]),
                ] %2 %6 %4
                %8:f32[3, 2] = convert_element_type [data_type=f32] %0
                %9:f32[3, 1] = ragged_dot_general [
                    dimensions=(dot=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[0], \
                        rhs_batching=[0]), lhs_ragged=[0], rhs_group=[]),
                ] %8 %3 %4
                %10:f32[3, 1] = add %7 %9
            in (%5, %10)
            "}
            .trim_end(),
        );
        assert_eq!(
            jvp.input_types(),
            vec![
                lhs_type.clone(),
                rhs_type.clone(),
                ArrayType::new_static(DataType::F32, [3, 2]),
                ArrayType::new_static(DataType::F32, [3, 2, 1]),
            ],
        );
        assert_eq!(
            jvp.output_types(),
            vec![ArrayType::new_static(DataType::F8E8M0FNU, [3, 1]), ArrayType::new_static(DataType::F32, [3, 1])],
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
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let lhs_input = builder.add_input(lhs.r#type().into_owned());
        let rhs_input = builder.add_input(rhs.r#type().into_owned());
        let groups = builder.add_constant(group_sizes.clone());
        let output = builder
            .add_instruction(
                RaggedDotOperation::new(RaggedDotDimensionNumbers::matmul()),
                Vec::new(),
                vec![lhs_input, rhs_input, groups],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let transpose = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transpose.to_string(),
            indoc! {"
                lambda %0:f32[3, 1], %1:f32[2, 2, 1] .
                let %2:i32[2] = const [1, 2]
                    %3:f32[3, 2] = ragged_dot_general [
                        dimensions=(dot=(lhs_contracting=[1], rhs_contracting=[2], lhs_batching=[], \
                        rhs_batching=[]), lhs_ragged=[0], rhs_group=[0]),
                    ] %0 %1 %2
                in (%3)
            "}
            .trim_end(),
        );

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
                },
                {
                    inputs = [
                        (@known, lhs.clone()),
                        (@linear(type = rhs.r#type().into_owned())),
                        (@known, group_sizes),
                    ],
                    output_cotangents = [Array::matrix(3, 1, vec![2.0f32, 3.0, 5.0]).unwrap()],
                    input_cotangents = [Array::from_elements::<f32>(
                        plain_array(&[2, 2, 1]),
                        &[2.0, 4.0, 34.0, 42.0],
                    ).unwrap()],
                },
            ],
        );
    }

    #[test]
    fn test_ragged_dot_transposition_inverts_permuted_output_axes() {
        let dimensions = RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![1], vec![1], vec![2], vec![2]),
            vec![0],
            vec![0],
        );
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
        // The LHS adjoint is initially ordered `[batch, ragged, contracting]` and must be transposed back to `[ragged,
        // contracting, batch]` with `[1, 2, 0]`.
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
            }],
        );

        // The RHS adjoint is initially ordered `[group, batch, contracting, result]` and must be transposed back to
        // `[group, contracting, batch, result]` with `[0, 2, 1, 3]`.
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
            }],
        );
    }

    #[test]
    fn test_ragged_dot_transposition_rejects_contracting_and_batch_modes() {
        let cases = [
            (
                RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
                    DotDimensionNumbers::new(vec![1], vec![0], Vec::new(), Vec::new()),
                    vec![1],
                    Vec::new(),
                )),
                [plain_array(&[2, 5]), plain_array(&[5, 4]), ArrayType::new_static(DataType::I32, [3])],
                RaggedDotMode::Contracting,
            ),
            (
                RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
                    DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]),
                    vec![0],
                    Vec::new(),
                )),
                [plain_array(&[5, 2]), plain_array(&[5, 2, 4]), ArrayType::new_static(DataType::I32, [3])],
                RaggedDotMode::Batch,
            ),
        ];
        for (operation, input_types, mode) in cases {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let inputs = input_types.map(|input_type| builder.add_input(input_type));
            let outputs = builder.add_instruction(operation, Vec::new(), inputs.to_vec(), None).unwrap().to_vec();
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
        let sizes = context.input(ArrayType::new_static(DataType::I32, [2]));
        let inputs = [
            PartialValue::Unknown(plain_array(&[3, 2])),
            PartialValue::Unknown(plain_array(&[2, 2, 1])),
            PartialValue::Known(sizes),
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
    fn test_ragged_dot_transposition_structural_zero() {
        // A structural zero output cotangent stages no adjoint and leaves the requested cotangent zero.
        let mut context = TranspositionContext::new(TracingContext::<Array, ArrayOperation<Array>>::new());
        let right = context.input(plain_array(&[2, 2, 1]));
        let sizes = context.input(ArrayType::new_static(DataType::I32, [2]));
        let inputs =
            [PartialValue::Unknown(plain_array(&[3, 2])), PartialValue::Known(right), PartialValue::Known(sizes)];
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
        assert!(context.builder().borrow().instructions().is_empty());
        assert!(context.take_cotangents(&accumulators).unwrap().iter().all(MaybeZero::is_zero));
    }

    #[test]
    fn test_ragged_dot_transposition_validates_original_dimensions() {
        let mut context = TranspositionContext::new(TracingContext::<Array, ArrayOperation<Array>>::new());
        let right = context.input(plain_array(&[2, 2, 1]));
        let sizes = context.input(ArrayType::new_static(DataType::I32, [2]));
        let seed = context.input(plain_array(&[3, 1]));
        let inputs =
            [PartialValue::Unknown(plain_array(&[3, 2])), PartialValue::Known(right), PartialValue::Known(sizes)];
        let accumulators = context.cotangent_accumulators(&inputs, &[true, false, false]).unwrap();
        let operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![4], vec![1], vec![], vec![]),
            vec![0],
            vec![0],
        ));
        assert_eq!(
            operation.transpose(&mut context, &EmptyRegionDriver, &inputs, &[MaybeZero::Value(seed)], &accumulators),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` LHS dimension index out of bounds",
            ))))),
        );
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_ragged_dot_array_ir() {
        type ArrayIrContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        // Array IR tracers stage the homogeneous grouped dot as an array member instruction.
        let (_, program) = ArrayIrContext::trace(
            |inputs: Vec<Tracer<ArrayIrContext>>| inputs[0].ragged_dot(&inputs[1], &inputs[2]),
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
                let %3:f32[3, 1] = ragged_dot_general [
                    dimensions=(dot=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[], rhs_batching=[]), \
                        lhs_ragged=[0], rhs_group=[0]),
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
            lhs.ragged_dot(&rhs, &group_sizes),
            Ok(ArrayIrValue::Array(Array::matrix(3, 1, vec![12.0f32, 18.0, 28.0]).unwrap())),
        );
    }
}
